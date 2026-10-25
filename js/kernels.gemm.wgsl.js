// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// GEMM con teselado en REGISTROS para el prefill.
//
// El matmul de este motor da una fila de salida por subgrupo y procesa un token
// cada vez: cada peso que se trae de memoria se usa UNA vez. Con un lote de 32
// tokens eso son 32 lecturas de los mismos pesos. Y el decode está limitado por
// traerse los pesos, no por calcular, así que ese factor se paga entero.
//
// Se ve por los dos lados: el prefill va por detrás de llama.cpp en el MISMO
// navegador, y una pasada de 8 tokens cuesta 5,15 veces una de 1 cuando debería
// costar poco más que una — que es también la razón por la que el decodificado
// especulativo no compensa aquí (verificar K borradores es justo esa pasada).
//
// La idea es la de cualquier GEMM por bloques, y es la que usa ggml-webgpu:
//
//   · una tesela de pesos y otra de activaciones se llevan a memoria compartida
//     UNA vez, ya descuantizadas — así el descuantizador también se paga una vez
//     por tesela y no una vez por salida;
//   · cada hilo acumula un bloque TM×TN en REGISTROS, así que cada peso que lee
//     de memoria compartida alimenta TN multiplicaciones en vez de una.
//
// Con TM=TN=4 y un grupo de 8×8 hilos, un peso traído de memoria global acaba
// usándose 32 veces. Ese es todo el cambio.
//
// Las teselas van en f32 y no en f16 a propósito. f16 halvaría el tráfico de
// memoria compartida, pero mezclaría el cambio de MECANISMO con un cambio de
// PRECISIÓN, y entonces una diferencia en los logits no se sabría atribuir.
// Primero que el mecanismo se mida solo; f16 es un interruptor posterior.
//
// ── TK NO es el bloque del tipo ─────────────────────────────────────────────
// La primera versión ataba `TK` —el paso del bucle sobre k— al `blockElems` del
// tipo, porque con Q8_0 (bloques de 32) las dos cosas coincidían y la tirada por
// hilo caía dentro de un solo bloque sin pensarlo. PQ2_0, que es el tipo del
// modelo de producción, tiene bloques de 128, y ahí esa atadura se rompe por
// TRES sitios a la vez:
//
//   1. MEMORIA COMPARTIDA. Con la forma ganadora (2×4 en un grupo de 8×8) y
//      TK=128 la tesela son (16 filas + 32 tokens)·128·4 = 24.576 bytes de los
//      32.768 que suelen dar los adaptadores. CABE —o sea que atarlo no habría fallado,
//      habría ido lento y callado— pero se come el 75% del presupuesto, y eso
//      hunde las ondas residentes. Lo que hizo ganar a 2×4 sobre 4×4 fue
//      exactamente lo contrario. Se estaría pagando con ocupación lo que se
//      gana con reúso, y ese cambio no sale a cuenta.
//   2. EL BARRIDO SE HUNDE. Las formas de 256 hilos (2×4 g16×16) piden
//      (32+64)·128·4 = 49.152 bytes: no caben en ningún caso. O sea que el tipo
//      del modelo decidiría qué formas se pueden ni siquiera probar.
//   3. ES UNA COINCIDENCIA, NO UNA RELACIÓN. TK es una propiedad del BUCLE
//      (cuánta k cabe en la tesela); blockElems es del FORMATO (cada cuántos
//      pesos hay una escala). Atarlas hace que la forma más rápida dependa del
//      tipo de cuantización por accidente.
//
// Y no es un argumento de sobremesa: el punto 1 está MEDIDO. Con todo lo demás
// igual sobre el 27B PQ2_0, atar TK al bloque —la opción que no requería tocar
// nada— habría dejado sobre la mesa un factor 2,4 frente a TK=32. Todas las
// variantes eligen el mismo token, así que se comparan cosas que calculan lo
// mismo.
//
// Así que TK se desacopla: la carga a memoria compartida sabe descuantizar un
// tramo PARCIAL de un bloque. La condición que se impone es que **TK divida a
// blockElems**, y no la general «TK y blockElems cualesquiera». No es pereza:
// mientras una tesela no cruce de bloque, cada tirada lee UNA escala y la
// dirección del peso sigue siendo `(fila·bpr + b)·blockBytes`, sin partir nada.
// Permitir el cruce obligaría a llevar dos escalas y dos direcciones por tirada
// para ganar formas que nadie ha pedido.
//
// Con TK = blockElems (SUB = 1) el bucle interno queda de una sola vuelta y el
// código generado es el de antes: el orden de acumulación de Q8_0 no se mueve,
// así que sus logits siguen siendo los mismos bit a bit que antes del cambio.
//
// Lo que el desacoplo NO regala: que un tipo pueda entrar aquí sigue dependiendo
// de que un tramo parcial se pueda descuantizar por sí solo. PQ2_0 puede porque
// son cuatro pesos consecutivos por byte y una escala por bloque. PTQ1_0, por
// ejemplo, NO: guarda cinco trits por byte y el orden de salida no es el de
// los bytes, así que un tramo contiguo de pesos no es un tramo contiguo de
// fichero. De ahí que `TIRADAS` sea una tabla explícita y no un `decode` genérico.

import { HELPERS, AXIS, quantInfo } from './kernels.wgsl.js';

// Paso por defecto del bucle sobre k. 32 no es el bloque de nadie en concreto:
// es el que sale mejor en el 27B PQ2_0, por delante de 16, 64 y 128, y a la vez
// el que ya estaba medido en Q8_0. Para un tipo de bloque mayor se toma igual
// —min(32, blockElems)— en vez de seguir al bloque, por lo de arriba.
export const TK_POR_DEFECTO = 32;

// Descuantización de UNA tirada de `n` pesos consecutivos de una fila, dentro de
// un solo bloque. Vive aquí y no en el descriptor del tipo porque el contrato es
// distinto: el descriptor entrega valores a un `emit` que acumula; esto los
// ESCRIBE en memoria compartida. Mezclarlos haría que un cambio en el camino de
// decode moviera en silencio el de prefill.
//
// El andamio les da tres nombres: `blk` (offset en bytes del bloque), `e0` (el
// índice del primer peso de la tirada DENTRO del bloque) y `n` como constante.
// `e0` es múltiplo de 4 por construcción — lo garantizan las guardas del
// constructor— y de eso dependen las dos tiradas.
const TIRADAS = {
  // 32 pesos, 34 bytes: escala f16 + 32 enteros de 8 bits con signo.
  Q8_0: ({ n, escribe }) => /* wgsl */`
      // La escala se lee UNA vez por tirada, no una por peso: es lo que permite
      // esta forma y no la del decode, donde la tirada es de 8.
      let d = load_scale(blk);
      for (var i = 0u; i < ${n}u; i = i + 4u) {
        let q = load_u32_at(blk + 2u + e0 + i);
        for (var c = 0u; c < 4u; c = c + 1u) {
          ${escribe('i + c', 'f32(get_byte_i32(q, c)) * d')}
        }
      }`,

  // 128 pesos, 34 bytes: escala f16 + 32 bytes de ranuras de 2 bits. CUATRO
  // pesos consecutivos por byte, así que un tramo parcial alineado a 4 es un
  // tramo de bytes entero: ni se parte un byte ni hace falta una segunda escala.
  // La expresión del valor es CARÁCTER POR CARÁCTER la del `decode` del tipo
  // (`f32(i32((by >> (c*2)) & 3) - 1) * d`): si se escribiera «mejor» —sacar la
  // escala del bucle, por ejemplo— los pesos dejarían de ser los mismos bits y
  // una diferencia en los logits ya no se sabría atribuir al mecanismo.
  PQ2_0: ({ n, escribe }) => /* wgsl */`
      let d = load_scale(blk);
      for (var i = 0u; i < ${n}u; i = i + 4u) {
        let by = load_byte(blk + 2u + (e0 + i) / 4u);
        for (var c = 0u; c < 4u; c = c + 1u) {
          ${escribe('i + c', 'f32(i32((by >> (c * 2u)) & 3u) - 1) * d')}
        }
      }`,
};

// Qué tipos saben pasar por aquí. La lista SON las claves de `TIRADAS`: tener
// dos sitios que decir lo mismo es tener dos sitios que desincronizar.
export const TIPOS_GEMM = new Set(Object.keys(TIRADAS));

// Paso del bucle sobre k para un tipo. `override` es para el barrido.
export function tkPara(type, override = 0) {
  const d = quantInfo(type);
  const tk = (override | 0) || Math.min(TK_POR_DEFECTO, d.blockElems);
  return tk;
}

export function soportaGemm(type, k, TK = 0) {
  if (!TIPOS_GEMM.has(type)) return false;
  const d = quantInfo(type);
  const tk = tkPara(type, TK);
  // TK divide al bloque (una tesela no cruza de bloque), TK es múltiplo de 4
  // (las dos tiradas leen de 4 en 4) y k entero en teselas.
  return tk > 0 && tk % 4 === 0 && d.blockElems % tk === 0 && k % tk === 0;
}

/**
 * Y = W · X con W cuantizada (rows×k) y X token-major (tokens×k).
 * Salida y[token * rows + row], igual que el matmul de siempre: es un sustituto
 * intercambiable, no un camino paralelo con otro contrato.
 *
 * `maxCompartida` es el tope de memoria compartida por grupo del dispositivo
 * (`limits.maxComputeWorkgroupStorageSize`). Se pasa para poder RECHAZAR una
 * forma que no cabe con un mensaje que diga cuánto pedía, en vez de que reviente
 * la validación de WebGPU al crear el pipeline con un texto genérico.
 */
export function matmulGemmShader(type, { TM = 4, TN = 4, WGM = 8, WGN = 8, TK = 0, maxCompartida = 32768 } = {}) {
  const d = quantInfo(type);
  if (!TIPOS_GEMM.has(type)) {
    throw new Error(`matmulGemmShader: «${type}» todavía no pasa por el GEMM (ver TIRADAS)`);
  }
  const tk = tkPara(type, TK);
  if (tk % 4 !== 0) {
    throw new Error(`matmulGemmShader: TK=${tk} tiene que ser múltiplo de 4 (las tiradas leen de 4 en 4)`);
  }
  // Una tesela que cruzara de bloque necesitaría dos escalas y dos direcciones
  // por tirada. No se soporta, así que se rechaza: leerla con una sola escala
  // daría pesos mal escalados SIN lanzar, que es el modo de fallo caro.
  if (d.blockElems % tk !== 0) {
    throw new Error(`matmulGemmShader: TK=${tk} tiene que dividir al bloque de «${type}» (${d.blockElems})`);
  }
  const SUB = d.blockElems / tk;   // sub-teselas por bloque; 1 = el caso de antes
  const HILOS = WGM * WGN;
  const FILAS_WG = WGM * TM;      // filas de salida por grupo
  const TOKS_WG = WGN * TN;       // tokens por grupo
  // ── PASO DE LA TESELA: CON RELLENO ─────────────────────────────────────────
  // El paso entre filas de la tesela era `tk` floats. Con el tk medido como
  // mejor (32) eso son 32 floats = 128 bytes, y en una arquitectura de 32 bancos
  // de 4 bytes el banco es `índice % 32`: todas las filas caen en EL MISMO
  // banco. Conflicto de 32 vías, o sea los accesos de un subgrupo serializados.
  // Es el peor caso posible y era el último candidato sin probar del prefill
  // (el matmul corre al 37 % del techo de cálculo).
  //
  // Un float de relleno por fila rompe la periodicidad: con paso tk+1, la fila
  // r en kk cae en el banco (r·(tk+1) + kk) % 32 = (r + kk) % 32 con tk=32, o
  // sea un banco distinto por fila. Cuesta (FILAS_WG + TOKS_WG) floats de
  // memoria compartida, que es ruido.
  //
  // ⚠️ MEDIDO Y DESCARTADO. Apagado por defecto: con RELLENO=0 el shader
  // generado es EXACTAMENTE el de antes.
  //
  // En un A/B del prefill completo, con el orden alternado en la misma sesión,
  // el ganador cambia de una ronda a otra: es ruido, no efecto. (Se alterna el
  // orden y se exige que el ganador se repita porque la primera medida paga la
  // compilación del shader.)
  //
  // Y la razón de que no haya nada que ganar está en el propio bucle: en
  // `xs[(ln*TN+tn)*SW + kk]` los hilos de un subgrupo que comparten `ln` leen la
  // MISMA dirección, que es un broadcast y no un conflicto. Un conflicto de
  // banco exige hilos leyendo direcciones DISTINTAS del mismo banco, y aquí el
  // patrón es broadcast más paso 1 en kk. Nunca hubo un conflicto de 32 vías
  // que romper, aunque el paso de 128 bytes lo sugiriera sobre el papel.
  //
  // Se conserva el mecanismo (a 1 rellena) porque la medida cuesta una línea y
  // en otro adaptador, u otro reparto de hilos, podría dejar de ser broadcast.
  const RELLENO = globalThis.__ELFFUSS_PAD_TESELA === 1 ? 1 : 0;
  const SW = tk + RELLENO;

  // La memoria compartida se comprueba AQUÍ y no se deja al driver: el barrido
  // necesita distinguir «esta forma no cabe» de «esta forma va lenta».
  // Se cuenta con SW y no con tk: si no, el relleno quedaría fuera de la cuenta
  // y una tesela al límite pasaría la guarda para que la rechazara el driver,
  // que es exactamente lo que esta comprobación existe para evitar.
  const compartida = (FILAS_WG + TOKS_WG) * SW * 4;
  if (compartida > maxCompartida) {
    throw new Error(`matmulGemmShader: la tesela pide ${compartida} bytes de memoria compartida ` +
      `y el tope es ${maxCompartida} (filas ${FILAS_WG} + tokens ${TOKS_WG}, TK=${tk})`);
  }
  // Cuántos elementos pone cada hilo en cada tesela. Tiene que repartirse
  // EXACTO: un reparto con resto dejaría huecos sin escribir en memoria
  // compartida, y lo que se leería de ahí sería basura de la iteración anterior
  // —no ceros—, o sea un resultado incorrecto que además no es reproducible.
  if ((FILAS_WG * tk) % HILOS !== 0 || (TOKS_WG * tk) % HILOS !== 0) {
    throw new Error(`matmulGemmShader: la tesela no se reparte exacta entre ${HILOS} hilos`);
  }
  const POR_HILO_W = (FILAS_WG * tk) / HILOS;
  const POR_HILO_X = (TOKS_WG * tk) / HILOS;
  if (POR_HILO_W % 4 !== 0) {
    throw new Error('matmulGemmShader: la tirada de pesos por hilo debe ser múltiplo de 4 (se leen de 4 en 4)');
  }
  // La carga a memoria compartida calcula UNA fila y UN bloque por hilo, así que
  // la tirada no puede cruzar de fila. Sin esta guarda, una forma con la tirada
  // más larga que el bloque lee los pesos de la fila equivocada y devuelve un
  // resultado incorrecto SIN lanzar — que es justo lo que pasó con 8x4 g16x4:
  // salía a toda velocidad y contestaba otra cosa. Una forma que no vale tiene
  // que ser rechazada, no medida.
  if (POR_HILO_W > tk || POR_HILO_X > tk) {
    throw new Error(`matmulGemmShader: la tirada por hilo (pesos ${POR_HILO_W}, activaciones ${POR_HILO_X}) ` +
      `no puede pasar de la tesela (${tk}): cruzaría de fila`);
  }
  // Y no basta con que quepa: tiene que ENCAJAR. Con una tirada que no divide a
  // TK, el hilo `tid` arranca en `(tid·POR_HILO) % TK` y se sale por el final de
  // su fila, pisando el principio de la siguiente y dejando un hueco detrás. Con
  // formas potencia de dos no pasa nunca, pero «no pasa nunca con las que he
  // probado» no es una guarda.
  if (tk % POR_HILO_W !== 0 || tk % POR_HILO_X !== 0) {
    throw new Error(`matmulGemmShader: la tirada por hilo (pesos ${POR_HILO_W}, activaciones ${POR_HILO_X}) ` +
      `tiene que dividir a la tesela (${tk}): si no, se desborda de fila`);
  }
  // Y la tirada tampoco puede cruzar de BLOQUE dentro de la fila. Con TK
  // dividiendo al bloque y la tirada dividiendo a TK esto ya se cumple, pero se
  // deja escrito porque es la condición de la que depende leer UNA sola escala.
  if (d.blockElems % POR_HILO_W !== 0) {
    throw new Error(`matmulGemmShader: la tirada de pesos (${POR_HILO_W}) cruzaría de bloque (${d.blockElems})`);
  }

  // El bucle del producto, en DOS formas. La de siempre indexa arrays por la
  // variable de bucle; la plana genera un escalar por acumulador y ni un indice
  // dinamico.
  //
  // Importa porque un array en ambito de funcion indexado dinamicamente puede
  // acabar en memoria de RASCADO en vez de en registros, a criterio del
  // compilador. Si los acumuladores no estan en registros, cada
  // multiplicacion-acumulacion es un viaje a memoria.
  //
  // OJO con lo que esto NO es: el teselado por registros YA estaba puesto —los
  // pesos se leen una vez por kk y alimentan las TN columnas—. Esto solo probaba
  // si el indexado dinamico se derrama, que es una duda de compilador y no una
  // palanca nueva.
  //
  // ⚠️ MEDIDO Y DESCARTADO: x1,00 en el prefill completo, y el ganador alterna
  // entre rondas. El compilador ya mantiene los acumuladores en registros.
  // Apagado por defecto; el shader que genera el defecto es el de siempre.
  //
  // Y una leccion del INSTRUMENTO que vale mas que el resultado: sin calentar,
  // la primera medida paga la compilacion del shader y sale muy por debajo, y
  // ese artefacto cae siempre sobre la primera variante medida. Sin calentar,
  // el plano parecia ganar; calentando las DOS formas y descartando esas
  // medidas, la diferencia desaparece. Si una variante compila su propio
  // pipeline, hay que calentarla antes de cronometrarla.
  const plano = globalThis.__ELFFUSS_GEMM_PLANO === true;
  const L = [];
  L.push(`      for (var kk = 0u; kk < ${tk}u; kk = kk + 1u) {`);
  if (plano) {
    for (let tm = 0; tm < TM; tm++) {
      L.push(`        let w${tm} = ws[(lm * ${TM}u + ${tm}u) * ${SW}u + kk];`);
    }
    for (let tn = 0; tn < TN; tn++) {
      L.push(`        let x${tn} = xs[(ln * ${TN}u + ${tn}u) * ${SW}u + kk];`);
      for (let tm = 0; tm < TM; tm++) {
        L.push(`        a${tm}_${tn} = a${tm}_${tn} + w${tm} * x${tn};`);
      }
    }
  } else {
    L.push(`        var wr : array<f32, ${TM}u>;`);
    L.push(`        for (var tm = 0u; tm < ${TM}u; tm = tm + 1u) {`);
    L.push(`          wr[tm] = ws[(lm * ${TM}u + tm) * ${SW}u + kk];`);
    L.push('        }');
    L.push(`        for (var tn = 0u; tn < ${TN}u; tn = tn + 1u) {`);
    L.push(`          let xv = xs[(ln * ${TN}u + tn) * ${SW}u + kk];`);
    L.push(`          for (var tm = 0u; tm < ${TM}u; tm = tm + 1u) {`);
    L.push('            acc[tm][tn] = acc[tm][tn] + wr[tm] * xv;');
    L.push('          }');
    L.push('        }');
  }
  L.push('      }');
  const PRODUCTO = L.join('\n');

  // Declaración de los acumuladores y volcado final, coherentes con la forma
  // elegida: si el producto usa escalares, aquí NO puede quedar un array, o el
  // shader ni compila.
  const D = [];
  if (plano) {
    D.push('  // Acumuladores como ESCALARES: ni un indice dinamico en el kernel.');
    for (let tm = 0; tm < TM; tm++) {
      for (let tn = 0; tn < TN; tn++) D.push(`  var a${tm}_${tn} = 0.0;`);
    }
  } else {
    D.push('  // Acumulador TM x TN en registros. Esto es el kernel entero: lo que lo');
    D.push('  // hace rapido es que estas celdas no vuelven a memoria hasta el final.');
    D.push(`  var acc : array<array<f32, ${TN}u>, ${TM}u>;`);
    D.push(`  for (var tm = 0u; tm < ${TM}u; tm = tm + 1u) {`);
    D.push(`    for (var tn = 0u; tn < ${TN}u; tn = tn + 1u) { acc[tm][tn] = 0.0; }`);
    D.push('  }');
  }
  const DECL_ACC = D.join('\n');

  const V = [];
  if (plano) {
    for (let tn = 0; tn < TN; tn++) {
      V.push(`  {`);
      V.push(`    let tok = tok0 + ln * ${TN}u + ${tn}u;`);
      V.push('    if (tok < params.tokens) {');
      for (let tm = 0; tm < TM; tm++) {
        V.push(`      let f${tm} = fila0 + lm * ${TM}u + ${tm}u;`);
        V.push(`      if (f${tm} < params.rows) { y[tok * params.rows + f${tm}] = a${tm}_${tn}; }`);
      }
      V.push('    }');
      V.push('  }');
    }
  } else {
    V.push(`  for (var tn = 0u; tn < ${TN}u; tn = tn + 1u) {`);
    V.push(`    let tok = tok0 + ln * ${TN}u + tn;`);
    V.push('    if (tok >= params.tokens) { continue; }');
    V.push(`    for (var tm = 0u; tm < ${TM}u; tm = tm + 1u) {`);
    V.push(`      let fila = fila0 + lm * ${TM}u + tm;`);
    V.push('      if (fila < params.rows) {');
    V.push('        y[tok * params.rows + fila] = acc[tm][tn];');
    V.push('      }');
    V.push('    }');
    V.push('  }');
  }
  const VOLCADO = V.join('\n');

  const tirada = TIRADAS[type]({
    n: POR_HILO_W,
    escribe: (i, v) => `ws[fila * ${SW}u + kk0 + ${i}] = ${v};`,
  });

  return /* wgsl */`
struct Params { nBlocks: u32, blocksPerRow: u32, rows: u32, k: u32,
                tokens: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> w : array<u32>;
@group(0) @binding(1) var<storage, read> x : array<f32>;
@group(0) @binding(2) var<storage, read_write> y : array<f32>;
@group(0) @binding(3) var<uniform> params : Params;
${d.table ? '@group(0) @binding(4) var<storage, read> tbl : array<u32>;' : ''}
${HELPERS}${d.helpers}

var<workgroup> ws : array<f32, ${FILAS_WG * SW}u>;   // tesela de pesos, ya descuantizada (paso con relleno)
var<workgroup> xs : array<f32, ${TOKS_WG * SW}u>;    // tesela de activaciones (paso con relleno)

@compute @workgroup_size(${HILOS})
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let tid = lid.x;
  // Reparto del hilo dentro de la tesela de salida.
  let lm = tid % ${WGM}u;
  let ln = tid / ${WGM}u;
  // Las filas van en DOS ejes: uno solo topa en 65.535 y la cabeza de salida
  // son cientos de miles de filas. Los tokens van en el tercero.
  let fila0 = (wid.x + wid.y * ${AXIS}u) * ${FILAS_WG}u;
  let tok0  = wid.z * ${TOKS_WG}u;
  let bpr = params.blocksPerRow;

  // Acumulador TM×TN en registros. Esto es el kernel entero: lo que lo hace
  // rápido es que estas ${TM * TN} celdas no vuelven a memoria hasta el final.
${DECL_ACC}

  for (var b = 0u; b < bpr; b = b + 1u) {
    // Sub-teselas dentro del bloque. Con SUB=1 (TK = blockElems) esta vuelta
    // desaparece y queda exactamente el bucle de antes, incluido el orden de
    // acumulación: por eso Q8_0 no cambia ni un bit al desacoplar TK.
    for (var s = 0u; s < ${SUB}u; s = s + 1u) {
      let kOff = s * ${tk}u;                       // offset dentro del bloque
      let k0 = b * ${d.blockElems}u + kOff;        // offset absoluto en k
      // ── Tesela de PESOS a memoria compartida, descuantizada una sola vez ──
      {
        let plano = tid * ${POR_HILO_W}u;
        let fila = plano / ${tk}u;
        let kk0 = plano % ${tk}u;
        let e0 = kOff + kk0;
        let global = fila0 + fila;
        if (global < params.rows) {
          let blk = (global * bpr + b) * ${d.blockBytes}u;
${tirada}
        } else {
          // Fuera de rango se escriben CEROS, no se deja lo de la vuelta anterior:
          // esas filas se calculan igual y solo se descartan al guardar, así que
          // basura aquí sería basura sumada en acumuladores que sí se guardan.
          for (var i = 0u; i < ${POR_HILO_W}u; i = i + 1u) { ws[fila * ${SW}u + kk0 + i] = 0.0; }
        }
      }
      // ── Tesela de ACTIVACIONES ──
      {
        let plano = tid * ${POR_HILO_X}u;
        let t = plano / ${tk}u;
        let kk0 = plano % ${tk}u;
        let global = tok0 + t;
        let base = global * params.k + k0 + kk0;
        for (var i = 0u; i < ${POR_HILO_X}u; i = i + 1u) {
          xs[t * ${SW}u + kk0 + i] = select(0.0, x[base + i], global < params.tokens);
        }
      }
      workgroupBarrier();

${PRODUCTO}
      workgroupBarrier();
    }
  }

${VOLCADO}
}
`;
}
