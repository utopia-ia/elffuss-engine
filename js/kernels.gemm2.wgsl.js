// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · GEMM del prompt con la descuantización DENTRO (interruptor GEMM2).
// ─────────────────────────────────────────────────────────────────────────────
// El camino DRIFT (kernels.drift.wgsl.js) hace tres despachos por matmul:
// descuantiza la matriz ENTERA a un búfer f16, pasa las activaciones a f16 y
// luego corre el GEMM sin memoria compartida. El primer despacho escribe casi el
// doble de bytes de los que lee, y el GEMM vuelve a leer todo eso: por cada
// pasada del prompt se mueven varias veces los pesos del modelo solo para
// cambiarles el formato. Aquí ese búfer intermedio desaparece. Dos kernels:
//
//   · EXACTO. La MISMA tesela que DRIFT (un grupo de `tpw` hilos en tokens
//     consecutivos, cada hilo `tpt` tokens × `filas` filas, el mismo bucle sobre
//     k de cuatro en cuatro y el mismo `a[j] += f32(dot(x, w))`) pero los pesos
//     llegan cuantizados: los hilos del grupo descuantizan juntos un trozo de
//     `nb` bloques por fila a memoria compartida y el bucle lee de ahí. Cada
//     peso sale con la MISMA expresión que el descuantizador de DRIFT (un valor
//     exacto en f32, porque cuanto × escala cabe en el significando) y se
//     redondea a f16 igual; cada salida acumula los mismos sumandos en el mismo
//     orden. Por eso tiene que dar los mismos bits que DRIFT. Descuantizar en el
//     bucle de cada hilo, sin compartir, repetiría la descodificación en los
//     hilos que comparten el peso: es lo que la cabecera de DRIFT descartó, y
//     con la memoria compartida el trabajo se reparte en vez de repetirse.
//
//   · TESELA (sub-interruptor GEMM2_TESELA). Bloqueo en registros con memoria
//     compartida: una tesela de BM filas de pesos × BN tokens, paso de k de 32
//     (un bloque de cuantización por fila), cada hilo TM filas × TN tokens.
//     Las dos teselas viven en memoria compartida en f16 con k como dimensión
//     lenta, así que un hilo lee de un golpe cuatro filas (o cuatro tokens) del
//     mismo k y cada lectura alimenta TM·TN/4 multiplicaciones. Doble búfer: la
//     tesela siguiente se trae a registros antes de calcular la actual y se
//     guarda después, con una sola barrera por paso. Acumulación en f32
//     producto a producto (más precisa que el `dot` en f16 de DRIFT), así que
//     NO es bit a bit: se valida con tolerancia contra DRIFT y contra una
//     referencia en f64. La acumulación en f16 dentro de cada tesela de k es
//     otro sub-interruptor (GEMM2_ACC16), con su propia tolerancia.
//
// Formatos de pesos: Q8_0 y Q4_0 tal cual vienen del GGUF, y Q8_0R/Q4_0R, los
// mismos bytes reordenados al cargar: primero los cuantos de todos los bloques,
// cada bloque alineado a 16 bytes, y detrás las escalas f16, una por media
// palabra. Los kernels leen SIEMPRE a través de las funciones de bloque de
// `FORMATOS` (escala y cuatro pesos del bloque `ib`), así que cambiar de formato
// no toca el bucle. Las cuatro variantes descuantizan al MISMO valor: el formato
// reordenado tiene que dar los mismos bits que el original.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.
import { HELPERS } from './kernels.wgsl.js';
import { matmulGemmShader } from './kernels.gemm.wgsl.js';

// Tipos que este GEMM sabe leer, y de qué tipo del GGUF sale cada reordenado.
export const GEMM2_TIPOS = new Set(['Q8_0', 'Q4_0', 'Q8_0R', 'Q4_0R']);
const ORIGEN = { Q8_0: 'Q8_0', Q4_0: 'Q4_0', Q8_0R: 'Q8_0', Q4_0R: 'Q4_0' };
export const tipoBase = (type) => ORIGEN[type] || type;

// Palabras u32 de cuantos por bloque en los reordenados. La región de escalas
// empieza justo detrás de la de cuantos: palabra W = ppb · R · B, con R las filas
// del TENSOR (no las de una llamada que pida menos, como la cabeza con el
// vocabulario) y B bloques por fila. ops.gemm2.js no la usa: saca W de la marca
// del cargador (`escalasDesde` con el N del tensor). En los formatos del GGUF no
// hay región aparte y el valor no se usa.
export function baseEscalas(type, rows, bpr) {
  const ppb = type === 'Q8_0R' ? 8 : type === 'Q4_0R' ? 4 : 0;
  return ppb * rows * bpr;
}

// ── Lectura de bloques ──────────────────────────────────────────────────────
// Cada formato da cuatro funciones, partidas para que la tesela pueda PRECARGAR:
// traer las palabras crudas a registros antes de calcular, y descodificarlas
// después, cuando la lectura ya ha llegado. Descodificar justo al cargar haría
// esperar a la memoria en ese mismo sitio y la precarga no serviría de nada.
//   g2_ecruda(ib)            palabra que contiene la escala f16 del bloque ib
//   g2_escala_de(c, ib)      la escala en f32 (exacta: f16 cabe en f32)
//   g2_crudo(ib, e)          palabras con los cuatro cuantos e..e+3 (e múltiplo de 4)
//   g2_cuatro(c, ib, e, d)   los cuatro pesos en f32, ya escalados
// Las expresiones del valor son CARÁCTER POR CARÁCTER las de `driftDequantShader`:
// si se escribieran «mejor», los pesos podrían dejar de ser los mismos bits y una
// diferencia en los logits ya no se sabría atribuir.
//
// En los formatos del GGUF los bloques miden 34 y 18 bytes: la mitad no empieza
// en palabra y los cuantos se recomponen con dos palabras y un desplazamiento,
// igual que `load_u32_at` (que lee la palabra siguiente aunque no haga falta:
// por eso los búferes de pesos llevan cuatro bytes de holgura al final). La
// escala nunca cruza palabra porque 34·i y 18·i son pares.
const FORMATOS = {
  Q8_0: /* wgsl */`
fn g2_ecruda(ib : u32) -> u32 { return w[(ib * 34u) / 4u]; }
fn g2_escala_de(c : u32, ib : u32) -> f32 {
  return unpack2x16float((c >> (((ib * 34u) & 2u) * 8u)) & 0xFFFFu)[0];
}
fn g2_crudo(ib : u32, e : u32) -> vec2<u32> {
  let o = (ib * 34u + 2u + e) / 4u;
  return vec2<u32>(w[o], w[o + 1u]);
}
fn g2_cuatro(c : vec2<u32>, ib : u32, e : u32, d : f32) -> vec4<f32> {
  let sh = ((ib * 34u + 2u + e) & 3u) * 8u;
  let q = select((c.x >> sh) | (c.y << ((32u - sh) & 31u)), c.x, sh == 0u);
  return vec4<f32>(f32(get_byte_i32(q, 0u)), f32(get_byte_i32(q, 1u)), f32(get_byte_i32(q, 2u)), f32(get_byte_i32(q, 3u))) * d;
}`,

  // Byte j del bloque: peso j en el nibble bajo y j+16 en el alto; valor (q−8)·d.
  Q4_0: /* wgsl */`
fn g2_ecruda(ib : u32) -> u32 { return w[(ib * 18u) / 4u]; }
fn g2_escala_de(c : u32, ib : u32) -> f32 {
  return unpack2x16float((c >> (((ib * 18u) & 2u) * 8u)) & 0xFFFFu)[0];
}
fn g2_crudo(ib : u32, e : u32) -> vec2<u32> {
  let o = (ib * 18u + 2u + (e & 15u)) / 4u;
  return vec2<u32>(w[o], w[o + 1u]);
}
fn g2_cuatro(c : vec2<u32>, ib : u32, e : u32, d : f32) -> vec4<f32> {
  let s8 = ((ib * 18u + 2u + (e & 15u)) & 3u) * 8u;
  let q = select((c.x >> s8) | (c.y << ((32u - s8) & 31u)), c.x, s8 == 0u);
  let sh = select(0u, 4u, e >= 16u);
  return (vec4<f32>(f32((q >> sh) & 0xFu), f32((q >> (8u + sh)) & 0xFu), f32((q >> (16u + sh)) & 0xFu), f32((q >> (24u + sh)) & 0xFu)) - 8.0) * d;
}`,

  // Reordenado: cuantos del bloque ib en las palabras 8·ib … 8·ib+7, en el
  // orden del GGUF; escala f16 en la media palabra ib de la región que empieza
  // en `p.baseEsc`. Todo alineado: una palabra por cuarteto, sin recomponer.
  Q8_0R: /* wgsl */`
fn g2_ecruda(ib : u32) -> u32 { return w[p.baseEsc + ib / 2u]; }
fn g2_escala_de(c : u32, ib : u32) -> f32 { return unpack2x16float(c)[ib & 1u]; }
fn g2_crudo(ib : u32, e : u32) -> vec2<u32> { return vec2<u32>(w[ib * 8u + e / 4u], 0u); }
fn g2_cuatro(c : vec2<u32>, ib : u32, e : u32, d : f32) -> vec4<f32> {
  let q = c.x;
  return vec4<f32>(f32(get_byte_i32(q, 0u)), f32(get_byte_i32(q, 1u)), f32(get_byte_i32(q, 2u)), f32(get_byte_i32(q, 3u))) * d;
}`,

  // Reordenado: 16 bytes de nibbles en las palabras 4·ib … 4·ib+3, MISMO
  // orden que Q4_0 (byte j: peso j abajo, j+16 arriba).
  Q4_0R: /* wgsl */`
fn g2_ecruda(ib : u32) -> u32 { return w[p.baseEsc + ib / 2u]; }
fn g2_escala_de(c : u32, ib : u32) -> f32 { return unpack2x16float(c)[ib & 1u]; }
fn g2_crudo(ib : u32, e : u32) -> vec2<u32> { return vec2<u32>(w[ib * 4u + (e & 15u) / 4u], 0u); }
fn g2_cuatro(c : vec2<u32>, ib : u32, e : u32, d : f32) -> vec4<f32> {
  let q = c.x;
  let sh = select(0u, 4u, e >= 16u);
  return (vec4<f32>(f32((q >> sh) & 0xFu), f32((q >> (8u + sh)) & 0xFu), f32((q >> (16u + sh)) & 0xFu), f32((q >> (24u + sh)) & 0xFu)) - 8.0) * d;
}`,
};

export function bloqueWgsl(type) {
  const f = FORMATOS[type];
  if (!f) throw new Error(`gemm2: sin lectura de bloques para «${type}»`);
  return f;
}

// ── 1) EXACTO: la tesela de DRIFT con los pesos descuantizados en el grupo ──
// Reparto de la descuantización de un trozo (`filas` filas × `nb` bloques): cada
// par (fila, bloque) son 8 cuartetos de pesos; se reparten entre los hilos de
// forma que cada uno lea UNA escala y cuartetos consecutivos de UN bloque. Con
// la forma por defecto (32 hilos, 16 filas, 2 bloques) cada hilo descodifica un
// bloque entero.
export function repartoExacto(tpw, filas, nb) {
  const pares = filas * nb;
  let porPar = 1;                       // hilos por par: 1, 2, 4 u 8
  while (porPar < 8 && pares * porPar * 2 <= tpw) porPar *= 2;
  return { pares, porPar, cuartetos: 8 / porPar, unidades: pares * porPar };
}

// `doble`: dos trozos en memoria compartida y UNA barrera por trozo, o uno solo y
// dos barreras. El de un trozo ocupa la mitad: DRIFT no usa memoria compartida y
// sus grupos de un subgrupo caben muchos a la vez en cada núcleo; con la memoria
// compartida eso puede pasar a estar limitado por ella, y en un grupo de un solo
// subgrupo la barrera de más cuesta poco. Por eso el de un trozo es el defecto;
// el otro queda para medirlo (ver `__ELFFUSS_GEMM2_TROZO` en ops.gemm2.js).
//
// `fma32` (sub-interruptor GEMM2_FMA32) cambia SOLO la aritmética del bucle:
// cada producto en f32 con FMA, en orden de k, en vez del `dot` de cuatro en f16.
// Deja de ser bit a bit con DRIFT (es más precisa: la misma cuenta que la
// tesela) y mantiene la tesela de DRIFT, con su ocupación. Es la otra mitad del
// A/B: si la tesela con memoria compartida resulta limitada por ocupación, esto
// separa cuánto venía de la aritmética y cuánto del reparto.
export function gemm2ExactoShader({ type = 'Q8_0', tpt = 2, tpw = 32, filas = 16, nb = 2, doble = false, fma32 = false } = {}) {
  if (!GEMM2_TIPOS.has(type)) throw new Error(`gemm2 exacto: tipo «${type}» no soportado`);
  if (filas !== 16 && filas !== 8) throw new Error(`gemm2 exacto: filas=${filas} (16 u 8)`);
  if (nb !== 1 && nb !== 2) throw new Error(`gemm2 exacto: nb=${nb} (1 o 2 bloques por trozo)`);
  if (!(tpw >= 1 && tpt >= 1)) throw new Error('gemm2 exacto: forma inválida');
  const T = [...Array(tpt).keys()];
  const R = repartoExacto(tpw, filas, nb);
  const KSL = nb * 8;                  // pasos de k (de cuatro en cuatro) por trozo
  const TROZO = KSL * filas;           // vec4<f16> por trozo
  return /* wgsl */`
enable f16;
struct P { ks4 : u32, rows : u32, tokens : u32, bpr : u32, baseEsc : u32, pad0 : u32, pad1 : u32, pad2 : u32 };
@group(0) @binding(0) var<storage, read>       w : array<u32>;          // pesos cuantizados (${type})
@group(0) @binding(1) var<storage, read>       X : array<vec4<f16>>;    // [T][KS]
@group(0) @binding(2) var<storage, read_write> Y : array<f32>;          // [T][rows]
@group(0) @binding(3) var<uniform>             p : P;
${HELPERS}
${bloqueWgsl(type)}

${doble ? `// Dos trozos: mientras unos hilos calculan con uno, nadie escribe en él, así que
// basta UNA barrera por trozo (el siguiente se escribe en el otro).` : `// Un trozo: barrera tras descuantizarlo y otra antes de pisarlo con el siguiente.`}
var<workgroup> Wt : array<vec4<f16>, ${(doble ? 2 : 1) * TROZO}u>;   // [trozo][ksl][fila]

@compute @workgroup_size(${tpw})
fn main(@builtin(workgroup_id) wg : vec3<u32>, @builtin(local_invocation_id) l : vec3<u32>) {
  let og = wg.y + wg.z * 65535u;
  if (og * ${filas}u >= p.rows) { return; }   // uniforme en el grupo: antes de toda barrera
  let t0 = (wg.x * ${tpw}u + l.x) * ${tpt}u;
  let ultimo = p.tokens - 1u;
${T.map(t => `  var a${t} : array<f32, ${filas}>;
  let x${t}i = min(t0 + ${t}u, ultimo) * p.ks4;`).join('\n')}
  for (var j = 0u; j < ${filas}u; j++) { ${T.map(t => `a${t}[j] = 0.0;`).join(' ')} }
  let nTrozos = p.bpr / ${nb}u;
  for (var c = 0u; c < nTrozos; c++) {
    let base = ${doble ? `(c & 1u) * ${TROZO}u` : '0u'};
    // Descuantización del trozo c, repartida: unidad u → (fila, bloque, parte).
    for (var u = l.x; u < ${R.unidades}u; u += ${tpw}u) {
      let par = u / ${R.porPar}u;
      let parte = u % ${R.porPar}u;
      let jf = par % ${filas}u;
      let bl = par / ${filas}u;
      let ib = (og * ${filas}u + jf) * p.bpr + c * ${nb}u + bl;
      let d = g2_escala_de(g2_ecruda(ib), ib);
      for (var m = 0u; m < ${R.cuartetos}u; m++) {
        let qd = parte * ${R.cuartetos}u + m;          // cuarteto dentro del bloque
        Wt[base + (bl * 8u + qd) * ${filas}u + jf] = vec4<f16>(g2_cuatro(g2_crudo(ib, qd * 4u), ib, qd * 4u, d));
      }
    }
    workgroupBarrier();
    let ks0 = c * ${KSL}u;
    for (var ksl = 0u; ksl < ${KSL}u; ksl++) {
${fma32
    ? T.map(t => `      let x${t} = vec4<f32>(X[x${t}i + ks0 + ksl]);`).join('\n')
    : T.map(t => `      let x${t} = X[x${t}i + ks0 + ksl];`).join('\n')}
      for (var j = 0u; j < ${filas}u; j++) {
${fma32 ? `        let wv = vec4<f32>(Wt[base + ksl * ${filas}u + j]);
${T.map(t => `        a${t}[j] = fma(x${t}.w, wv.w, fma(x${t}.z, wv.z, fma(x${t}.y, wv.y, fma(x${t}.x, wv.x, a${t}[j]))));`).join('\n')}`
    : `        let wv = Wt[base + ksl * ${filas}u + j];
${T.map(t => `        a${t}[j] += f32(dot(x${t}, wv));`).join('\n')}`}
      }
    }${doble ? '' : `
    workgroupBarrier();`}
  }
  let n0 = og * ${filas}u;
${T.map(t => `  if (t0 + ${t}u < p.tokens) { for (var j = 0u; j < ${filas}u; j++) { Y[(t0 + ${t}u) * p.rows + n0 + j] = a${t}[j]; } }`).join('\n')}
}`;
}

// ── 2) TESELA: bloqueo en registros con memoria compartida ──────────────────
// Reparto de hilos: t → (tr = t % NTR, tc = t / NTR). El hilo tr lleva los
// cuartetos de filas tr, tr+NTR, … (G = TM/4) y el hilo tc los cuartetos de
// tokens tc, tc+NTC, … (H = TN/4). Con eso, en una lectura de la tesela de pesos
// los hilos de un subgrupo piden cuartetos CONSECUTIVOS (sin conflictos de
// banco) y en la de activaciones casi todos piden el mismo (difusión).
//
// Las teselas en memoria compartida van con k como dimensión lenta: el elemento
// (k, q) es un vec4<f16> con cuatro filas (o cuatro tokens) en ese k. Una fila de
// k mide BM/4 + 1 vec4: el relleno hace que al GUARDAR (cada hilo escribe un
// cuarteto de filas en cuatro k seguidos) los ocho k de un subgrupo no caigan
// todos en los mismos bancos.
export function formaTesela({ BM = 128, BN = 64, TM = 8, TN = 4, doble = true } = {}) {
  const NTR = BM / TM, NTC = BN / TN;
  const SW = BM / 4 + 1, SX = BN / 4 + 1;
  const NBUF = doble ? 2 : 1;
  return {
    BM, BN, TM, TN, doble, NTR, NTC, HILOS: NTR * NTC, G: TM / 4, H: TN / 4, SW, SX, NBUF,
    // Bytes de memoria compartida: lo que se compara con el tope del dispositivo.
    compartida: NBUF * 32 * (SW + SX) * 8,
    unidadesW: (BM / 4) * 8, unidadesX: (BN / 4) * 8,
  };
}

export function gemm2TeselaShader({ type = 'Q8_0', BM = 128, BN = 64, TM = 8, TN = 4, doble = true,
  acc16 = false, partes = 1, maxCompartida = 32768 } = {}) {
  if (!GEMM2_TIPOS.has(type)) throw new Error(`gemm2 tesela: tipo «${type}» no soportado`);
  if (TM % 4 || TN % 4 || BM % TM || BN % TN || BM % 4 || BN % 4) {
    throw new Error(`gemm2 tesela: forma ${BM}×${BN} con ${TM}×${TN} por hilo no encaja (múltiplos de 4)`);
  }
  const F = formaTesela({ BM, BN, TM, TN, doble });
  const { NTR, NTC, HILOS, G, H, SW, SX, NBUF } = F;
  if (HILOS > 256) throw new Error(`gemm2 tesela: ${HILOS} hilos por grupo (tope 256)`);
  if (F.compartida > maxCompartida) {
    throw new Error(`gemm2 tesela: pide ${F.compartida} bytes de memoria compartida y el tope es ${maxCompartida}`);
  }
  // Cada unidad de carga son 4 filas × 4 k (pesos) o 4 tokens × 4 k
  // (activaciones). Tienen que repartirse sin resto: un hueco sin escribir en la
  // tesela se leería como basura de la vuelta anterior, no como cero.
  if (F.unidadesW > HILOS && F.unidadesW % HILOS) throw new Error('gemm2 tesela: la tesela de pesos no se reparte exacta');
  if (F.unidadesX > HILOS && F.unidadesX % HILOS) throw new Error('gemm2 tesela: la tesela de activaciones no se reparte exacta');
  if (!(partes >= 1 && partes <= 64)) throw new Error(`gemm2 tesela: partes=${partes}`);
  const UW = Math.max(1, F.unidadesW / HILOS), UX = Math.max(1, F.unidadesX / HILOS);
  const TW = 32 * SW, TX = 32 * SX;
  const L = [];
  const r4 = [0, 1, 2, 3];

  // Declaración de las unidades de carga de este hilo.
  for (let i = 0; i < UW; i++) {
    L.push(`  let uw${i} = t + ${i * HILOS}u;`);
    L.push(`  let wkq${i} = uw${i} % 8u;`);
    L.push(`  let wrq${i} = uw${i} / 8u;`);
    L.push(`  let wact${i} = uw${i} < ${F.unidadesW}u;`);
    for (const r of r4) L.push(`  var cw${i}_${r} : vec2<u32>;\n  var ce${i}_${r} : u32;`);
  }
  for (let i = 0; i < UX; i++) {
    L.push(`  let ux${i} = t + ${i * HILOS}u;`);
    L.push(`  let xkq${i} = ux${i} % 8u;`);
    L.push(`  let xtq${i} = ux${i} / 8u;`);
    L.push(`  let xact${i} = ux${i} < ${F.unidadesX}u;`);
    for (const r of r4) L.push(`  var cx${i}_${r} : vec4<f16>;`);
  }
  const DECL = L.join('\n');

  // Carga a registros de la tesela `kt` (solo lecturas; nada se descodifica aquí).
  const carga = (kt) => {
    const C = [];
    for (let i = 0; i < UW; i++) {
      C.push(`    if (wact${i}) {`);
      for (const r of r4) {
        C.push(`      { let ib = (fila0 + wrq${i} * 4u + ${r}u) * p.bpr + ${kt}; ce${i}_${r} = g2_ecruda(ib); cw${i}_${r} = g2_crudo(ib, wkq${i} * 4u); }`);
      }
      C.push('    }');
    }
    for (let i = 0; i < UX; i++) {
      C.push(`    if (xact${i}) {`);
      for (const r of r4) {
        // Tokens fuera del lote: CERO, no lo que hubiera de la vuelta anterior.
        C.push(`      { let tk = tok0 + xtq${i} * 4u + ${r}u; cx${i}_${r} = select(vec4<f16>(0.0), X[min(tk, p.tokens - 1u) * p.ks4 + ${kt} * 8u + xkq${i}], tk < p.tokens); }`);
      }
      C.push('    }');
    }
    return C.join('\n');
  };

  // Descodificación de lo precargado y guardado en el búfer `b` de la tesela.
  const guarda = (b, kt) => {
    const C = [];
    for (let i = 0; i < UW; i++) {
      C.push(`    if (wact${i}) {`);
      for (const r of r4) {
        C.push(`      let ib${r} = (fila0 + wrq${i} * 4u + ${r}u) * p.bpr + ${kt};`);
        C.push(`      let v${r} = g2_cuatro(cw${i}_${r}, ib${r}, wkq${i} * 4u, g2_escala_de(ce${i}_${r}, ib${r}));`);
      }
      // Transposición 4×4 en registros: cada vec4 guardado son cuatro FILAS en un k.
      for (let c = 0; c < 4; c++) {
        C.push(`      Ws[${b} * ${TW}u + (wkq${i} * 4u + ${c}u) * ${SW}u + wrq${i}] = vec4<f16>(vec4<f32>(v0[${c}], v1[${c}], v2[${c}], v3[${c}]));`);
      }
      C.push('    }');
    }
    for (let i = 0; i < UX; i++) {
      C.push(`    if (xact${i}) {`);
      for (let c = 0; c < 4; c++) {
        C.push(`      Xs[${b} * ${TX}u + (xkq${i} * 4u + ${c}u) * ${SX}u + xtq${i}] = vec4<f16>(cx${i}_0[${c}], cx${i}_1[${c}], cx${i}_2[${c}], cx${i}_3[${c}]);`);
      }
      C.push('    }');
    }
    return C.join('\n');
  };

  // Acumuladores: token n (0..TN-1) × cuarteto de filas g (0..G-1), vec4 sobre
  // las cuatro filas: así el volcado es una escritura vec4 por token y cuarteto.
  const A = [];
  for (let n = 0; n < TN; n++) {
    for (let g = 0; g < G; g++) {
      A.push(`  var a${n}_${g} = vec4<f32>(0.0);`);
      if (acc16) A.push(`  var h${n}_${g} = vec4<f16>(0.0);`);
    }
  }
  const ACUM = A.join('\n');

  // El producto de una tesela: 32 pasos de k; en cada uno, G lecturas de pesos y
  // H de activaciones alimentan TN·G FMA vectoriales.
  const P = [];
  P.push('    for (var kk = 0u; kk < 32u; kk = kk + 1u) {');
  P.push(`      let wb = cur * ${TW}u + kk * ${SW}u + tr;`);
  P.push(`      let xb = cur * ${TX}u + kk * ${SX}u + tc;`);
  for (let g = 0; g < G; g++) {
    P.push(acc16 ? `      let w${g} = Ws[wb + ${g * NTR}u];` : `      let w${g} = vec4<f32>(Ws[wb + ${g * NTR}u]);`);
  }
  for (let h = 0; h < H; h++) P.push(`      let x${h} = Xs[xb + ${h * NTC}u];`);
  for (let h = 0; h < H; h++) {
    for (let tk = 0; tk < 4; tk++) {
      const n = h * 4 + tk;
      P.push(acc16 ? `      let s${n} = vec4<f16>(x${h}[${tk}]);` : `      let s${n} = vec4<f32>(f32(x${h}[${tk}]));`);
      for (let g = 0; g < G; g++) {
        P.push(acc16 ? `      h${n}_${g} = fma(w${g}, s${n}, h${n}_${g});` : `      a${n}_${g} = fma(w${g}, s${n}, a${n}_${g});`);
      }
    }
  }
  P.push('    }');
  if (acc16) {
    // Volcado a f32 al final de CADA tesela de k: la suma en f16 nunca pasa de
    // 32 productos, que es lo que acota su error.
    for (let n = 0; n < TN; n++) {
      for (let g = 0; g < G; g++) P.push(`    a${n}_${g} = a${n}_${g} + vec4<f32>(h${n}_${g}); h${n}_${g} = vec4<f16>(0.0);`);
    }
  }
  const PRODUCTO = P.join('\n');

  const V = [];
  V.push(`  let off = wg.z * p.tokens * (p.rows / 4u);   // parte de k (1 sola sin partir)`);
  for (let h = 0; h < H; h++) {
    for (let tk = 0; tk < 4; tk++) {
      const n = h * 4 + tk;
      V.push(`  { let tok = tok0 + (tc + ${h * NTC}u) * 4u + ${tk}u;`);
      V.push('    if (tok < p.tokens) {');
      for (let g = 0; g < G; g++) {
        V.push(`      Y[off + tok * (p.rows / 4u) + fila0 / 4u + tr + ${g * NTR}u] = a${n}_${g};`);
      }
      V.push('    } }');
    }
  }
  const VOLCADO = V.join('\n');

  const BUCLE = doble ? /* wgsl */`
  for (var kt = kt0; kt < kt1; kt = kt + 1u) {
    let cur = (kt - kt0) & 1u;
    let hay = kt + 1u < kt1;
    if (hay) {
${carga('(kt + 1u)')}
    }
${PRODUCTO}
    if (hay) {
${guarda('(1u - cur)', '(kt + 1u)')}
    }
    // Una sola barrera: el búfer que se acaba de escribir nadie lo estaba leyendo
    // (se leyó en la vuelta anterior, y todos pasaron ya su barrera).
    workgroupBarrier();
  }` : /* wgsl */`
  for (var kt = kt0; kt < kt1; kt = kt + 1u) {
    let cur = 0u;
    let hay = kt + 1u < kt1;
    if (hay) {
${carga('(kt + 1u)')}
    }
${PRODUCTO}
    workgroupBarrier();
    if (hay) {
${guarda('0u', '(kt + 1u)')}
    }
    workgroupBarrier();
  }`;

  return /* wgsl */`
enable f16;
struct P { rows : u32, tokens : u32, bpr : u32, ks4 : u32, baseEsc : u32, pad0 : u32, pad1 : u32, pad2 : u32 };
@group(0) @binding(0) var<storage, read>       w : array<u32>;          // pesos cuantizados (${type})
@group(0) @binding(1) var<storage, read>       X : array<vec4<f16>>;    // [T][K/4]
@group(0) @binding(2) var<storage, read_write> Y : array<vec4<f32>>;    // [partes][T][rows/4]
@group(0) @binding(3) var<uniform>             p : P;
${HELPERS}
${bloqueWgsl(type)}

var<workgroup> Ws : array<vec4<f16>, ${NBUF * TW}u>;   // [búfer][k][cuarteto de filas] (paso ${SW})
var<workgroup> Xs : array<vec4<f16>, ${NBUF * TX}u>;   // [búfer][k][cuarteto de tokens] (paso ${SX})

@compute @workgroup_size(${HILOS})
fn main(@builtin(workgroup_id) wg : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let t = lid.x;
  let tr = t % ${NTR}u;
  let tc = t / ${NTR}u;
  let fila0 = wg.y * ${BM}u;
  let tok0 = wg.x * ${BN}u;
  // Rango de teselas de k de esta parte (todas si no se parte).
  let kt0 = (wg.z * p.bpr) / ${partes}u;
  let kt1 = ((wg.z + 1u) * p.bpr) / ${partes}u;
${DECL}
${ACUM}
  if (kt0 < kt1) {
${carga('kt0')}
${guarda('0u', 'kt0')}
  }
  workgroupBarrier();
${BUCLE}
${VOLCADO}
}`;
}

// ── Suma de las partes de k (forma partida de la tesela) ────────────────────
// Y = P0 + P1 + … en ese orden fijo: determinista de una pasada a otra.
export function gemm2SumaShader(partes = 2, wg = 256) {
  const S = [...Array(partes).keys()];
  return /* wgsl */`
struct P { n4 : u32, pad0 : u32, pad1 : u32, pad2 : u32 };
@group(0) @binding(0) var<storage, read>       src : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst : array<vec4<f32>>;
@group(0) @binding(2) var<uniform>             p   : P;
@compute @workgroup_size(${wg})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x + gid.y * 65535u * ${wg}u;
  if (i >= p.n4) { return; }
  var s = src[i];
${S.slice(1).map(s => `  s = s + src[i + ${s}u * p.n4];`).join('\n')}
  dst[i] = s;
}`;
}

// ── 5) Q4_0 y Q4_0R en el GEMM teselado de 4 a 63 tokens ────────────────────
// Tiradas para `matmulGemmShader` (kernels.gemm.wgsl.js), con su mismo contrato:
// escriben `n` pesos consecutivos de la fila desde `e0` (múltiplo de 4) dentro
// del bloque, a f32, en memoria compartida. El valor es CARÁCTER POR CARÁCTER el
// del `decode` de Q4_0 (`f32(i32(nibble) - 8) * d`), así que los pesos son los
// mismos bits que en el resto del motor. Un cuarteto alineado a 4 nunca cruza de
// la mitad baja a la alta del bloque (16 es múltiplo de 4).
const TIRADAS_Q4 = {
  Q4_0: ({ n, escribe }) => /* wgsl */`
      let d = load_scale(blk);
      for (var i = 0u; i < ${n}u; i = i + 4u) {
        let e = e0 + i;
        let q = load_u32_at(blk + 2u + (e & 15u));
        let sh = select(0u, 4u, e >= 16u);
        for (var c = 0u; c < 4u; c = c + 1u) {
          ${escribe('i + c', 'f32(i32((q >> (c * 8u + sh)) & 0xFu) - 8) * d')}
        }
      }`,
  // El bloque se localiza por su ÍNDICE (fila · bpr + b), no por bytes; la
  // región de escalas empieza en la palabra 4 · nBlocks, con nBlocks los bloques
  // del TENSOR (ops.gemm2.js los saca de la marca, no del filas · bpr de la
  // llamada).
  Q4_0R: ({ n, escribe }) => /* wgsl */`
      let ib = global * bpr + b;
      let d = unpack2x16float(w[4u * params.nBlocks + ib / 2u])[ib & 1u];
      for (var i = 0u; i < ${n}u; i = i + 4u) {
        let e = e0 + i;
        let q = w[ib * 4u + (e & 15u) / 4u];
        let sh = select(0u, 4u, e >= 16u);
        for (var c = 0u; c < 4u; c = c + 1u) {
          ${escribe('i + c', 'f32(i32((q >> (c * 8u + sh)) & 0xFu) - 8) * d')}
        }
      }`,
};

export const TIPOS_Q4_TESELA = new Set(Object.keys(TIRADAS_Q4));

// El `extra` que entiende `matmulGemmShader`: descriptor del bloque y tirada.
export function extraQ4(type) {
  const tirada = TIRADAS_Q4[type];
  if (!tirada) throw new Error(`gemm2: sin tirada de GEMM teselado para «${type}»`);
  return { info: { blockElems: 32, blockBytes: 18, helpers: '', table: null }, tirada };
}

// El shader del GEMM teselado para Q4_0/Q4_0R. Los mismos números de forma que
// el de Q8_0 (TM, TN, WGM, WGN, TK, maxCompartida); el relleno y el bucle plano
// los sigue leyendo `matmulGemmShader` de sus interruptores, como siempre.
export function gemmQ4Shader(type, forma = {}) {
  return matmulGemmShader(type, { ...forma, extra: extraQ4(type) });
}
