// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · shaders WGSL de pesos cuantizados. Seis tipos.
// ─────────────────────────────────────────────────────────────────────────────
// La decisión de fondo: los pesos NO se descuantizan en CPU. Se suben tal cual
// salen del GGUF y se descuantizan dentro del shader. Eso ahorra la pasada de
// CPU, el pico de memoria del doble y, sobre todo, el ancho de banda — que es lo
// que limita de verdad un matmul cuantizado.
//
// El precio es el layout: un bloque Q8_0 son 34 bytes (escala f16 + 32 int8) y
// en WGSL un storage buffer se lee como array<u32>. Como 34·i alterna resto 0 y
// 2 al dividir por 4, la mitad de los bloques NO empieza en palabra. Se ataca
// combinando dos palabras con desplazamientos, igual que llama.cpp.
//
// Una nota buena: 34·i SIEMPRE es par, así que la escala f16 nunca cruza la
// frontera de palabra — basta una carga y un desplazamiento de 0 o 16 bits. Y
// `unpack2x16float` es del NÚCLEO de WGSL, así que convertir f16→f32 no
// necesita la extensión shader-f16.
//
// ── Por qué hay UN andamio y seis descodificadores ──────────────────────────
// Los seis tipos comparten TODO menos la descodificación: mismo reparto de filas
// en dos ejes, mismo teselado de tokens, misma reducción en árbol. Escribir seis
// matmuls sería mantener seis veces la misma reducción y que se desincronicen.
// Aquí el andamio (`matmulShader`/`dequantShader`) es común y cada tipo aporta
// solo un `decode(emit)` que va sacando (índice dentro del bloque, valor). El
// andamio decide qué hacer con cada valor: acumularlo contra las activaciones
// (matmul) o escribirlo (dequant).
//
// El orden de las operaciones aritméticas es el de ggml, a propósito: `d1·q − m1`
// y no `d·(sc·q − …)`. No es superstición — cambiarlo mueve el último bit y el
// test de igualdad exacta contra la referencia deja de poder ser exacto.
//
// Las rejillas de los i-quants viven en `quant-tables.js` porque son parte del
// FORMATO, no de este código.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.

export const Q8_BLOCK = 32;
export const Q8_BLOCK_BYTES = 34;

// Tope de workgroups por eje del despacho. NO es una constante de rendimiento:
// pasarse es un error de validación. Las filas se reparten en dos ejes porque la
// cabeza de salida son 248.320 y aquí topa en 65.535. El shader la necesita para
// recomponer el índice, así que vive aquí y `ops.js` la importa: dos copias que
// se desincronicen darían filas a cero sin un solo error.
export const AXIS = 65535;

// ── Helpers compartidos ─────────────────────────────────────────────────────
// `w` es el buffer de pesos cuantizados, declarado por quien incluya esto.
export const HELPERS = /* wgsl */`
// Byte con signo dentro de una palabra: se sube al tope y se baja con
// desplazamiento ARITMÉTICO. Es el idiom correcto, no restar 256.
fn get_byte_i32(value: u32, index: u32) -> i32 {
  return bitcast<i32>(((value >> (index * 8u)) & 0xFFu) << 24u) >> 24u;
}

// Byte sin signo en un offset CUALQUIERA. Una sola palabra basta: un byte nunca
// cruza la frontera.
fn load_byte(byte_offset: u32) -> u32 {
  return (w[byte_offset / 4u] >> ((byte_offset & 3u) * 8u)) & 0xFFu;
}

// Palabra de 32 bits en un offset de bytes CUALQUIERA, combinando dos palabras.
// El «& 31u» no es adorno: con shift==0 el desplazamiento sería de 32 bits, que
// en WGSL da un valor indeterminado. Enmascarando, ese caso queda en hi<<0 y el
// select lo descarta. Es el bug número uno de esta técnica.
fn load_u32_at(byte_offset: u32) -> u32 {
  let word_idx = byte_offset / 4u;
  let shift = (byte_offset & 3u) * 8u;
  let lo = w[word_idx];
  let hi = w[word_idx + 1u];
  return select((lo >> shift) | (hi << ((32u - shift) & 31u)), lo, shift == 0u);
}

// Media palabra en un offset PAR. Todos los campos de 16 bits de todos los tipos
// caen en offsets pares, así que nunca cruzan la frontera de palabra.
fn load_u16_at(byte_offset: u32) -> u32 {
  return (w[byte_offset / 4u] >> ((byte_offset & 2u) * 8u)) & 0xFFFFu;
}

// Escala f16 → f32, en un offset par.
fn load_scale(byte_offset: u32) -> f32 {
  return unpack2x16float(load_u16_at(byte_offset))[0];
}
`;

// Helper exclusivo de Q4_K y Q5_K: seis bits de escala y seis de mínimo,
// guardados en 12 bytes para ocho sub-bloques. Los cuatro primeros van
// enteros; los cuatro últimos toman los cuatro bits bajos de un sitio y los dos
// altos de otro. Es la descodificación más fácil de equivocar del formato.
const SCALE_MIN_K4 = /* wgsl */`
fn scale_min_k4(sbase: u32, j: u32) -> vec2<u32> {
  if (j < 4u) {
    return vec2<u32>(load_byte(sbase + j) & 63u, load_byte(sbase + j + 4u) & 63u);
  }
  let a = load_byte(sbase + j + 4u);
  let b = load_byte(sbase + j - 4u);
  let c = load_byte(sbase + j);
  return vec2<u32>((a & 0xFu) | ((b >> 6u) << 4u), (a >> 4u) | ((c >> 6u) << 4u));
}
`;

// ── Descodificadores por tipo ───────────────────────────────────────────────
// Cada uno recibe `emit(idxExpr, valExpr)` y devuelve WGSL. Dentro del cuerpo
// hay dos nombres dados por el andamio: `blk` (offset en bytes del bloque) y el
// índice que emite cada valor, relativo al bloque.

const DECODERS = {
  // Sin cuantizar. No es un caso hipotético: llama.cpp deja en f32 o f16 los
  // tensores cuyo ne[0] no es múltiplo de 256, y basta UNA proyección así para
  // que el modelo no cargue. Aquí el «bloque» es el propio elemento, y el resto
  // del andamio (reparto de filas, teselado, reducción) no se entera.
  F32: {
    blockElems: 1,
    blockBytes: 4,
    helpers: '',
    decode: (emit) => `
${emit('0u', 'bitcast<f32>(w[blk / 4u])')}`,
  },

  F16: {
    blockElems: 2,
    blockBytes: 4,
    helpers: '',
    decode: (emit) => `
      let pair = unpack2x16float(w[blk / 4u]);
${emit('0u', 'pair.x')}
${emit('1u', 'pair.y')}`,
  },

  // BF16 — la conversion mas barata que existe: los 16 bits de un bfloat16 SON
  // la mitad alta de un f32, asi que basta desplazarlos y reinterpretar. Ni
  // tabla ni aritmetica de exponentes.
  //
  // Se lee de dos en dos, como F16, porque el almacen es `array<u32>` y una
  // palabra trae dos pesos. El de indice 0 esta en la mitad BAJA (little-endian).
  //
  // Hace falta para los modelos ternarios: dejan el camino recurrente del SSM
  // en mas precision que el resto, y ahi es donde aparece este tipo.
  BF16: {
    blockElems: 2,
    blockBytes: 4,
    helpers: '',
    decode: (emit) => `
      let par = w[blk / 4u];
${emit('0u', 'bitcast<f32>(par << 16u)')}
${emit('1u', 'bitcast<f32>(par & 0xFFFF0000u)')}`,
  },

  // 32 pesos, 34 bytes: escala f16 + 32 enteros de 8 bits con signo.
  Q8_0: {
    blockElems: Q8_BLOCK,
    blockBytes: Q8_BLOCK_BYTES,
    helpers: '',
    // La escala se saca FUERA del bucle: (Σ q·x)·d en vez de Σ (q·d)·x sería
    // aún mejor, pero rompería el contrato de `emit`, que entrega valores ya
    // escalados. Se paga una multiplicación por peso y se gana un andamio único.
    decode: (emit) => /* wgsl */`
      let d = load_scale(blk);
      for (var i = 0u; i < 8u; i = i + 1u) {
        let q = load_u32_at(blk + 2u + i * 4u);
        for (var c = 0u; c < 4u; c = c + 1u) {
${emit('i * 4u + c', 'f32(get_byte_i32(q, c)) * d')}
        }
      }`,
  },

  // 256 pesos, 84 bytes: 16 escalas/mínimos de 4+4 bits, 64 bytes de 2 bits,
  // y d/dmin al FINAL del bloque (no al principio, como en Q4_K/Q5_K).
  Q2_K: {
    blockElems: 256,
    blockBytes: 84,
    helpers: '',
    decode: (emit) => /* wgsl */`
      let d = load_scale(blk + 80u);
      let dmin = load_scale(blk + 82u);
      for (var n = 0u; n < 2u; n = n + 1u) {
        let qbase = blk + 16u + n * 32u;
        for (var j = 0u; j < 4u; j = j + 1u) {
          // El mismo byte de quants se lee CUATRO veces, dos bits cada vez:
          // j no elige el byte, elige el desplazamiento dentro del byte.
          let shift = j * 2u;
          let sca = load_byte(blk + n * 8u + j * 2u);
          let scb = load_byte(blk + n * 8u + j * 2u + 1u);
          let dl1 = d * f32(sca & 0xFu);   let ml1 = dmin * f32(sca >> 4u);
          let dl2 = d * f32(scb & 0xFu);   let ml2 = dmin * f32(scb >> 4u);
          let obase = n * 128u + j * 32u;
          for (var l = 0u; l < 4u; l = l + 1u) {
            let wa = w[(qbase + l * 4u) / 4u];
            let wb = w[(qbase + 16u + l * 4u) / 4u];
            for (var c = 0u; c < 4u; c = c + 1u) {
${emit('obase + l * 4u + c', 'dl1 * f32((wa >> (c * 8u + shift)) & 3u) - ml1')}
${emit('obase + 16u + l * 4u + c', 'dl2 * f32((wb >> (c * 8u + shift)) & 3u) - ml2')}
            }
          }
        }
      }`,
  },

  // 256 pesos, 144 bytes: d/dmin f16, 12 bytes de escalas de 6 bits, 128 bytes
  // de nibbles. Los nibbles bajos son la primera mitad de cada grupo de 64 y los
  // altos la segunda: NO alternan peso a peso.
  Q4_K: {
    blockElems: 256,
    blockBytes: 144,
    helpers: SCALE_MIN_K4,
    decode: (emit) => /* wgsl */`
      let d = load_scale(blk);
      let dmin = load_scale(blk + 2u);
      for (var j = 0u; j < 4u; j = j + 1u) {
        let sm1 = scale_min_k4(blk + 4u, j * 2u);
        let sm2 = scale_min_k4(blk + 4u, j * 2u + 1u);
        let d1 = d * f32(sm1.x);   let m1 = dmin * f32(sm1.y);
        let d2 = d * f32(sm2.x);   let m2 = dmin * f32(sm2.y);
        let qbase = blk + 16u + j * 32u;
        let obase = j * 64u;
        for (var l = 0u; l < 8u; l = l + 1u) {
          let ql = w[(qbase + l * 4u) / 4u];
          for (var c = 0u; c < 4u; c = c + 1u) {
            let q = (ql >> (c * 8u)) & 0xFFu;
${emit('obase + l * 4u + c', 'd1 * f32(q & 0xFu) - m1')}
${emit('obase + 32u + l * 4u + c', 'd2 * f32(q >> 4u) - m2')}
          }
        }
      }`,
  },

  // 256 pesos, 210 bytes. El que desbloquea las mezclas estándar: Q4_K_M,
  // Q5_K_M y compañía ponen la salida y el embedding en Q6_K, así que SIN este
  // kernel no carga ni un fichero k-quant de los que publica todo el mundo —
  // aunque el resto de sus tensores sí tenga kernel.
  //
  // Tres diferencias con Q4_K/Q5_K, y las tres importan:
  //
  //   · 210 NO es múltiplo de 4, así que los bloques quedan desalineados y hay
  //     que leer con `load_byte`/`load_scale`, no con `w[(blk + x) / 4u]` como
  //     hacen sus hermanos. Usar la forma alineada da pesos de otro bloque.
  //   · las escalas son de 8 bits CON SIGNO. Sin signo el modelo carga y genera
  //     texto degenerado, que es mucho peor que un error.
  //   · no hay mínimo: el valor es `d·escala·(q − 32)` y el −32 es entero, así
  //     que no se usa la forma `d·q − m`.
  //
  // Dentro de cada mitad de 128 pesos, los cuatro que comparten un byte de `qh`
  // van a 0, 32, 64 y 96 del destino con escalas is+0, is+2, is+4, is+6.
  Q6_K: {
    blockElems: 256,
    blockBytes: 210,
    helpers: /* wgsl */`
fn load_i8_at(byte_offset: u32) -> i32 {
  let b = i32(load_byte(byte_offset));
  return select(b, b - 256, b > 127);
}
`,
    // ⚠️ POR QUÉ ESTO LEE PALABRAS Y NO BYTES, que es la diferencia entre
    // «correcto» y «usable»:
    //
    // La primera versión leía byte a byte con `load_byte`, y lo justifiqué
    // diciendo que Q6_K solo aparece en unos pocos tensores y no es camino
    // caliente. **Era falso.** En una mezcla Q4_K_M real, Q6_K es el 38 % del
    // fichero: el embedding (que también es la proyección de salida) y el
    // `ffn_down` de TODAS las capas — o sea camino caliente en cada token y en
    // cada posición del prompt. Medido con un 4B, el prefill caía a ~1/30 del
    // de un modelo con Q8_0, y el banco lo cazó porque el prefill dejaba de ir
    // más rápido que la generación, que es imposible si procesa en bloque.
    //
    // Ahora cada vuelta trae CUATRO pesos con tres lecturas de 32 bits en vez
    // de doce de un byte, y las cuatro escalas salen del bucle interno: `is`
    // solo cambia cada 16 posiciones, así que dentro de un grupo de 4 es
    // constante. De 128 lecturas de escala por bloque se baja a 32.
    //
    // `load_u32_at` sigue siendo obligatorio —no `w[(blk + x) / 4u]`— porque
    // 210 bytes no es múltiplo de 4 y el bloque arranca desalineado la mitad de
    // las veces.
    decode: (emit) => /* wgsl */`
      let d = load_scale(blk + 208u);
      for (var n = 0u; n < 2u; n = n + 1u) {
        let qlb = blk + n * 64u;
        let qhb = blk + 128u + n * 32u;
        let scb = blk + 192u + n * 8u;
        let obase = n * 128u;
        for (var l4 = 0u; l4 < 32u; l4 = l4 + 4u) {
          let is = l4 >> 4u;
          let d1 = d * f32(load_i8_at(scb + is));
          let d2 = d * f32(load_i8_at(scb + is + 2u));
          let d3 = d * f32(load_i8_at(scb + is + 4u));
          let d4 = d * f32(load_i8_at(scb + is + 6u));
          let w0 = load_u32_at(qlb + l4);
          let w32 = load_u32_at(qlb + 32u + l4);
          let wh = load_u32_at(qhb + l4);
          for (var c = 0u; c < 4u; c = c + 1u) {
            let sh = c * 8u;
            let lo0 = (w0 >> sh) & 0xFFu;
            let lo32 = (w32 >> sh) & 0xFFu;
            let h = (wh >> sh) & 0xFFu;
            let q1 = i32((lo0 & 0xFu) | (((h >> 0u) & 3u) << 4u)) - 32;
            let q2 = i32((lo32 & 0xFu) | (((h >> 2u) & 3u) << 4u)) - 32;
            let q3 = i32((lo0 >> 4u) | (((h >> 4u) & 3u) << 4u)) - 32;
            let q4 = i32((lo32 >> 4u) | (((h >> 6u) & 3u) << 4u)) - 32;
${emit('obase + l4 + c', 'd1 * f32(q1)')}
${emit('obase + 32u + l4 + c', 'd2 * f32(q2)')}
${emit('obase + 64u + l4 + c', 'd3 * f32(q3)')}
${emit('obase + 96u + l4 + c', 'd4 * f32(q4)')}
          }
        }
      }`,
  },

  // 256 pesos, 176 bytes: como Q4_K más 32 bytes con el quinto bit. El bit que
  // le toca a cada peso avanza DOS posiciones por grupo de 64 (u1 <<= 2).
  Q5_K: {
    blockElems: 256,
    blockBytes: 176,
    helpers: SCALE_MIN_K4,
    decode: (emit) => /* wgsl */`
      let d = load_scale(blk);
      let dmin = load_scale(blk + 2u);
      for (var j = 0u; j < 4u; j = j + 1u) {
        let sm1 = scale_min_k4(blk + 4u, j * 2u);
        let sm2 = scale_min_k4(blk + 4u, j * 2u + 1u);
        let d1 = d * f32(sm1.x);   let m1 = dmin * f32(sm1.y);
        let d2 = d * f32(sm2.x);   let m2 = dmin * f32(sm2.y);
        let u1 = 1u << (j * 2u);
        let u2 = 2u << (j * 2u);
        let qbase = blk + 48u + j * 32u;
        let obase = j * 64u;
        for (var l = 0u; l < 8u; l = l + 1u) {
          let ql = w[(qbase + l * 4u) / 4u];
          let qh = w[(blk + 16u + l * 4u) / 4u];
          for (var c = 0u; c < 4u; c = c + 1u) {
            let lo = (ql >> (c * 8u)) & 0xFFu;
            let hi = (qh >> (c * 8u)) & 0xFFu;
            let a = (lo & 0xFu) + select(0u, 16u, (hi & u1) != 0u);
            let b = (lo >> 4u) + select(0u, 16u, (hi & u2) != 0u);
${emit('obase + l * 4u + c', 'd1 * f32(a) - m1')}
${emit('obase + 32u + l * 4u + c', 'd2 * f32(b) - m2')}
          }
        }
      }`,
  },

  // 256 pesos, 66 bytes. Cada 32 pesos: dos palabras. La primera son cuatro
  // índices de 8 bits a una rejilla de 256 octetos; la segunda, cuatro paquetes
  // de 7 bits de signos más 4 bits de escala local en el tope.
  IQ2_XXS: {
    blockElems: 256,
    blockBytes: 66,
    table: 'IQ2XXS',
    helpers: '',
    decode: (emit) => /* wgsl */`
      let d = load_scale(blk);
      for (var ib = 0u; ib < 8u; ib = ib + 1u) {
        let a0 = load_u32_at(blk + 2u + ib * 8u);
        let a1 = load_u32_at(blk + 6u + ib * 8u);
        let db = d * (0.5 + f32(a1 >> 28u)) * 0.25;
        for (var l = 0u; l < 4u; l = l + 1u) {
          let gi = (a0 >> (l * 8u)) & 0xFFu;
          // ggml guarda una tabla de 128 bytes de signos; es exactamente el
          // índice completado a paridad PAR, así que se calcula en vez de subirla.
          let bits = (a1 >> (l * 7u)) & 127u;
          let signs = bits | ((countOneBits(bits) & 1u) << 7u);
          let g0 = tbl[gi * 2u];
          let g1 = tbl[gi * 2u + 1u];
          let obase = ib * 32u + l * 8u;
          for (var j = 0u; j < 8u; j = j + 1u) {
            let gw = select(g0, g1, j >= 4u);
            let gv = f32((gw >> ((j & 3u) * 8u)) & 0xFFu);
            let sg = select(1.0, -1.0, (signs & (1u << j)) != 0u);
${emit('obase + j', 'db * gv * sg')}
          }
        }
      }`,
  },

  // 256 pesos, 50 bytes. Cada 32 pesos: cuatro índices de 11 bits (8 en `qs`,
  // 3 en `qh`) a una rejilla de 2048 octetos de valores en {-1,0,1}, más una
  // escala local de 3 bits y un signo que decide el desplazamiento ±1/8.
  // PQ2_0 — ternario en ranuras de 2 bits, bloque de 128. Tampoco es un tipo de
  // llama.cpp. 34 bytes: escala f16 + 32 de cuantizacion.
  //
  // ESTE es el formato que encaja con este kernel, y no por el tamaño —ocupa MAS
  // que el de base 3— sino por como reparte. 128 pesos a 2 bits son 32 bytes
  // EXACTOS y, con subgrupos de 32 carriles, toca un byte por carril: una carga,
  // sin etapas y sin divergencia. Y los cuatro pesos de cada carril son
  // CONSECUTIVOS, asi que sus activaciones entran en una lectura vec4.
  //
  // Comparado con IQ1_S, que por bloque hace tres cargas de peso mas DOS
  // lecturas dispersas de una tabla de rejilla indexada por carril. Aqui el peso
  // sale de un desplazamiento y una mascara: no hay tabla que leer.
  //
  // PTQ1_0 no sirve para esto aunque ocupe menos: guarda cinco trits por byte
  // en etapas de tamaños distintos (16, 8, y los dos bytes de qh), lo que parte
  // los 32 carriles en tres caminos con direcciones distintas.
  PQ2_0: {
    blockElems: 128,
    blockBytes: 34,
    helpers: '',
    decode: (emit) => /* wgsl */`
      let d = load_scale(blk);
      for (var i = 0u; i < 32u; i = i + 1u) {
        let by = load_byte(blk + 2u + i);
        for (var c = 0u; c < 4u; c = c + 1u) {
${emit('i * 4u + c', 'f32(i32((by >> (c * 2u)) & 3u) - 1) * d')}
        }
      }`,
    // Reparto por carril: el carril `sl` se queda el byte `sl`, que son sus
    // cuatro pesos. Una carga de peso por carril y por bloque.
    decodeCarril: (emit, xv = false) => /* wgsl */`
      let d = load_scale(blk);
      let by = load_byte(blk + 2u + sl);
      let obase = sl * 4u;
${xv ? `      // Los 4 pesos del carril son consecutivos y obase es multiplo de 4, igual
      // que k0 = b*128 y toda k del decode: caben en UNA lectura vectorial.
      let xb = (base_t * params.k${globalThis.__ELFFUSS_X_FIJA === true ? '' : ' + k0 + obase'}) >> 2u;
      let xv0 = x[xb];` : ''}
${Array.from({ length: 4 }, (_, j) => `      {
${emit(`obase + ${j}u`, `f32(i32((by >> ${j * 2}u) & 3u) - 1) * d`, j)}
      }`).join('\n')}`,
  },

  // PTQ1_0 — ternario en base 3, bloque de 128. NO es un tipo de llama.cpp:
  // lo define un fork de terceros. 28 bytes: qs[24] · qh[2] · escala f16.
  //
  // LA DIFERENCIA QUE IMPORTA, y no es el tamaño: NO HAY CODEBOOK. En IQ1_S el
  // peso sale de una tabla de rejilla indexada por carril —dos lecturas
  // dependientes y dispersas por bloque— y aquí sale de multiplicar un byte por
  // una potencia de tres y mirarle los bits de arriba. Cero lecturas de tabla.
  //
  // El formato mete CINCO trits en un byte (3^5 = 243 <= 256) con el truco
  // de siempre: al codificar se escala a 256/243 y al descodificar se recupera
  // con `(q * pow3[n] * 3) >> 8`, donde el producto por la potencia va en
  // aritmetica de OCHO bits y envuelve a proposito.
  //
  // El orden de salida NO es el de los bytes: por cada grupo, primero los 16 (u
  // 8) pesos de la potencia 0, luego los de la potencia 1, etc. Las etapas son
  // {32,16,8} y con qs de 24 bytes dan dos grupos —16 y 8—, o sea 80 + 40
  // valores, mas 8 que salen de qh. Total 128.
  //
  // VALIDADO CONTRA UN TERCERO, no contra si mismo: este mismo esquema con los
  // parametros de TQ1_0 (qs de 48 bytes) se reduce al «32 y luego 16» de
  // upstream, y da 16.384 valores IDENTICOS BIT A BIT a los de gguf-py. La ida
  // y vuelta contra uno mismo no habria distinguido un orden mal entendido.
  PTQ1_0: {
    blockElems: 128,
    blockBytes: 28,
    helpers: '',
    // `blk` aqui SI esta alineado a 4 (28 es multiplo de 4), al reves que IQ1_S.
    decode: (emit) => /* wgsl */`
      let d = load_scale(blk + 26u);
      var p3 = 1u;
      for (var n = 0u; n < 5u; n = n + 1u) {
        for (var m = 0u; m < 16u; m = m + 1u) {
          let q = (load_byte(blk + m) * p3) & 0xFFu;
${emit('n * 16u + m', 'f32(i32((q * 3u) >> 8u) - 1) * d')}
        }
        p3 = p3 * 3u;
      }
      p3 = 1u;
      for (var n = 0u; n < 5u; n = n + 1u) {
        for (var m = 0u; m < 8u; m = m + 1u) {
          let q = (load_byte(blk + 16u + m) * p3) & 0xFFu;
${emit('80u + n * 8u + m', 'f32(i32((q * 3u) >> 8u) - 1) * d')}
        }
        p3 = p3 * 3u;
      }
      p3 = 1u;
      for (var n = 0u; n < 4u; n = n + 1u) {
        for (var h = 0u; h < 2u; h = h + 1u) {
          let q = (load_byte(blk + 24u + h) * p3) & 0xFFu;
${emit('120u + n * 2u + h', 'f32(i32((q * 3u) >> 8u) - 1) * d')}
        }
        p3 = p3 * 3u;
      }`,

    // REPARTO POR CARRIL. Cuesta mas que en PQ2_0 y lo pagan las direcciones:
    // alli 128 pesos a 2 bits son 32 bytes JUSTOS y toca un byte por carril; aqui
    // el formato va en tres etapas de tamaños distintos —16 bytes, luego 8, y
    // aparte los dos de `qh`— asi que los carriles caen en tres sitios.
    //
    // Se resuelve con ARITMETICA y no con ramas, para que los 32 carriles emitan
    // las mismas instrucciones y solo cambien los numeros. Cada carril se queda
    // 4 pesos consecutivos (obase = 4*sl), que es lo que permite la lectura vec4
    // de activaciones, y los saca de UNA palabra de 32 bits:
    //   · carriles 0-19  -> 4 bytes seguidos de qs[0..15], una sola potencia
    //   · carriles 20-29 -> 4 bytes seguidos de qs[16..23], una sola potencia
    //   · carriles 30-31 -> los dos bytes de qh, con DOS potencias
    // El ultimo caso es el que ensucia: los demas leen cuatro bytes distintos con
    // la misma potencia y estos leen dos bytes alternos con potencias que cambian
    // a mitad. De ahi que el techo aqui sea peor que en PQ2_0 aunque el fichero
    // ocupe menos.
    //
    // Las tres direcciones se calculan siempre y `select` elige: en WGSL los dos
    // brazos de un select se evaluan igual, asi que `sl - 20u` da la vuelta en
    // los carriles bajos. No importa — ese valor no llega a usarse como
    // direccion, solo el elegido.
    decodeCarril: (emit, xv = false) => /* wgsl */`
      let d = load_scale(blk + 26u);
      let obase = sl * 4u;
      // Base de la palabra que le toca a este carril.
      let baseG = select(
        select(blk + 24u, blk + 16u + ((sl - 20u) & 1u) * 4u, sl < 30u),
        blk + (sl & 3u) * 4u, sl < 20u);
      let w4 = load_u32_at(baseG);
      // Potencia de tres del carril (para los carriles de qh cambia por peso).
      let pBase = select(select(2u * (sl - 30u), (sl - 20u) >> 1u, sl < 30u), sl >> 2u, sl < 20u);
      let deQh = sl >= 30u;
${xv ? `      // Los 4 pesos del carril son consecutivos y obase es multiplo de 4, igual
      // que k0 = b*128 y toda k del decode: una sola lectura vectorial.
      let xb = (base_t * params.k${globalThis.__ELFFUSS_X_FIJA === true ? '' : ' + k0 + obase'}) >> 2u;
      let xv0 = x[xb];` : ''}
${Array.from({ length: 4 }, (_, j) => `      {
        let by = (w4 >> (select(${j}u, ${j & 1}u, deQh) * 8u)) & 0xFFu;
        let pe = pBase + select(0u, ${j >> 1}u, deQh);
        let p3 = select(select(select(select(1u, 3u, pe >= 1u), 9u, pe >= 2u), 27u, pe >= 3u), 81u, pe >= 4u);
        let q = (by * p3) & 0xFFu;
${emit(`obase + ${j}u`, 'f32(i32((q * 3u) >> 8u) - 1) * d', j)}
      }`).join('\n')}`,
  },

  IQ1_S: {
    blockElems: 256,
    blockBytes: 50,
    table: 'IQ1S',
    helpers: '',
    // Reparto ALTERNATIVO: el carril posee un SUB-GRUPO de 8 pesos en vez de un
    // bloque entero. Un bloque son 8 sub-bloques x 4 grupos = 32 unidades de 8
    // pesos, que encaja EXACTO con los 32 carriles de un subgrupo.
    //
    // Por que: con el reparto por bloque, una fila de bpr=20 ocupa 20 carriles de
    // 32 y emite 8.192 ranuras-peso para 5.120 utiles (62,5%). Repartiendo por
    // sub-grupo, todos los carriles trabajan en TODOS los bloques: 5.120 ranuras,
    // todas utiles. El perfil por forma demostro que este kernel esta limitado por
    // emision, asi que esa utilizacion se traduce casi 1:1 en tiempo (~x1,5).
    //
    // Precio: cada carril calcula su propio qh/dl/delta para solo 8 pesos en vez
    // de para 32, asi que las operaciones por peso suben ~4%. Y sobre todo NO es
    // bit a bit: cambia que sumandos caen en que carril, o sea la particion de la
    // suma en coma flotante. Se valida con tolerancia, no con igualdad exacta.
    //
    // Exige ancho de workgroup 32 (un subgrupo); ops.js no lo activa si no.
    decodeCarril: (emit, xv = false) => {
      // DESENROLLAR EL BUCLE DE 8 (`__ELFFUSS_DESENROLLA`).
      //
      // OJO: este comentario decia que «~70% del decode es EMITIR instrucciones».
      // REFUTADO por una medida interna con marcas de tiempo del dispositivo:
      // el descuantizador entero es el 8,3% del matmul, y traer activaciones el
      // 44%. Por eso desenrollar no pagaba: atacaba un 8%. Se conserva el
      // interruptor porque es
      // bit a bit exacto y es requisito de la lectura vec4. Este
      // bucle da 8 vueltas con `j` conocido en compilacion y paga por peso tres
      // cosas que desaparecen al desenrollar:
      //   · el contador, la comparacion y el salto del bucle,
      //   · `select(g0, g1, j >= 4u)` -> queda g0 o g1, decidido en compilacion,
      //   · `get_byte_i32(gw, j & 3u)` -> desplazamiento por cantidad CONSTANTE
      //     en vez de por cantidad variable.
      //
      // Es BIT A BIT identico por construccion: mismos valores, mismo orden de
      // acumulacion (j ascendente), no toca la particion de la suma entre
      // carriles. Mismo perfil que izar la guarda, que ya rindio.
      //
      // Riesgo real: ocho copias del cuerpo suben la presion de registros y
      // pueden hundir la ocupancia — asi se perdio la rejilla compartida (x0,44).
      // Por eso va detras de un interruptor y se mide antes de quedarse.
      // La lectura vec4 exige indices CONSTANTES, asi que fuerza el desenrollado.
      const desenrolla = globalThis.__ELFFUSS_DESENROLLA === true || xv;
      // ORACULO DEL DESCUANTIZADOR — solo medicion, NUNCA en produccion.
      // X_FIJA y W_FIJA acotan el coste de TRAER datos; este acota el de
      // TRANSFORMARLOS. Se emite `f32(gw & 1u)` y no una constante a proposito:
      // `gw` cuelga de g0/g1, o sea de las dos lecturas de `tbl`. Con una
      // constante el compilador mataria tambien esas lecturas y estariamos
      // midiendo memoria de rejilla, que W_FIJA YA cuenta — sumar las dos celdas
      // contaria ese coste dos veces.
      // Acota la cota CONJUNTA de {extraer el byte + la aritmetica por peso},
      // no la de un candidato suelto. Da numeros FALSOS por diseno.
      // OJO CON LA ETIQUETA: mide MAS que «la aritmetica». Al sustituir `wv`,
      // `dl` y `delta` quedan muertas y con ellas `d`, asi que el compilador
      // borra tambien `load_scale`. Mide {una carga por bloque} + {toda la
      // aritmetica por peso}. `qh` sigue viva porque la usa `gi`.
      // ESTA CONTENIDO EN W_FIJA, no es disjunto: con `blk` invariante, todo lo
      // que cuelga de el (incluida esta aritmetica) queda fuera del bucle. Son
      // ANIDADOS, asi que NO se suman — sumarlos cuenta el mismo coste dos veces
      // y es lo que produjo el falso «49% explicado». Cotas validas: x <= 26%,
      // w <= 7%, luego sin explicar >= 67%.
      const deqPlana = globalThis.__ELFFUSS_DEQ_PLANA === true;
      const lazo = desenrolla
        ? Array.from({ length: 8 }, (_, j) => `      {
        let gw = ${j < 4 ? 'g0' : 'g1'};
        let gv = f32(get_byte_i32(gw, ${j % 4}u));
${emit(`obase + ${j}u`, deqPlana ? 'f32(gw & 1u)' : 'dl * (gv + delta)', j)}
      }`).join('\n')
        : `      for (var j = 0u; j < 8u; j = j + 1u) {
        let gw = select(g0, g1, j >= 4u);
        let gv = f32(get_byte_i32(gw, j & 3u));
${emit('obase + j', deqPlana ? 'f32(gw & 1u)' : 'dl * (gv + delta)')}
      }`;
      // ORACULO DE LA CADENA DE LA REJILLA (`__ELFFUSS_GI`) — solo medicion.
      //
      // La cadena real es LECTURA -> LECTURA: `load_byte` (y `qh`) producen `gi`,
      // y `gi` es la DIRECCION de `tbl`. En una ejecucion en orden, la segunda carga no
      // puede emitirse hasta que vuelve la primera: por bloque se pagan DOS
      // latencias en serie en vez de una. Ningun oraculo anterior toca eso:
      // W_FIJA cambia direcciones (trafico) y DEQ_PLANA quita aritmetica.
      //
      // Se sustituye `gi` por un hash de REGISTROS (nada de lecturas), de modo que
      // la carga de `tbl` puede emitirse a la vez que la de los pesos. Tres modos,
      // que separan tres cosas que la etiqueta «rejilla» confunde:
      //   hash — divergente y distinto por bloque: rompe SOLO el encadenamiento.
      //   unif — igual para los 32 carriles: hash-unif = coste de la DIVERGENCIA
      //          (32 lineas distintas por carril contra una sola difundida).
      //   inv  — no depende de `b`, luego es INVARIANTE del bucle y el compilador
      //          puede izar las dos lecturas de `tbl` fuera. hash-inv = trafico de
      //          rejilla por bloque. ADEMAS es el DETECTOR DE IZADO: si inv sale
      //          igual que hash, o el compilador no iza o izar no vale nada.
      //
      // TRAMPA CERRADA: si se quita `load_byte` del camino, el compilador la borra
      // entera y mediriamos «una carga menos», no «una cadena menos». Por eso su
      // valor se conserva vivo por el lado del VALOR (no de la direccion), sumado
      // a `dl` enmascarado con `params.pad0`, que vale 0 pero es un UNIFORME que el
      // compilador no puede plegar. La carga sigue emitiendose y sigue trayendo los
      // mismos bytes; lo unico que desaparece es que `tbl` la ESPERE.
      // Cuesta dos operaciones por bloque (mascara y conversion), ~0,3% del ALU.
      //
      // Los indices salen de la MISMA rejilla de 2048 entradas y sus bytes siguen
      // siendo {-1,0,1}, asi que no hay riesgo de desbordar ni de denormales: lo
      // que sale es NUMERICAMENTE FALSO, pero acotado igual que lo verdadero.
      const modoGi = globalThis.__ELFFUSS_GI;
      const SEMILLAS = {
        hash: 'lid.x * 2246822519u + b * 40503u + row * 2654435761u',
        unif: 'b * 40503u + row * 2654435761u',
        inv: 'lid.x * 2246822519u + row * 2654435761u',
      };
      // Sin este aviso, un modo mal escrito generaria el shader NORMAL bajo una
      // clave de pipeline distinta: dos ramas identicas midiendo x1,00 y una
      // conclusion («la cadena no pesa») que solo dice que hubo una errata.
      if (modoGi !== undefined && !(modoGi in SEMILLAS)) {
        throw new Error(`__ELFFUSS_GI desconocido: «${modoGi}» (hash|unif|inv)`);
      }
      // El orden de las tres lineas del control es el de SIEMPRE, byte a byte: en
      // el oraculo `dl` consume `raw` y tiene que ir detras, pero en produccion no
      // se mueve ni un espacio para no cambiar el reparto de registros.
      const nucleo = modoGi === undefined
        ? `      let dl = d * f32(2u * ((qh >> 12u) & 7u) + 1u);
      let delta = select(0.125, -0.125, (qh & 0x8000u) != 0u);
      let gi = load_byte(blk + 2u + ib * 4u + l) | (((qh >> (3u * l)) & 7u) << 8u);`
        : `      let raw = load_byte(blk + 2u + ib * 4u + l) | (((qh >> (3u * l)) & 7u) << 8u);
      let gi = (((${SEMILLAS[modoGi]}) * 2654435761u) >> 13u) & 2047u;
      let dl = d * f32(2u * ((qh >> 12u) & 7u) + 1u) + f32(raw & params.pad0);
      let delta = select(0.125, -0.125, (qh & 0x8000u) != 0u);`;
      // IZAR LAS CARGAS INDEPENDIENTES (`__ELFFUSS_IZA_CARGAS`).
      //
      // El orden de siempre emite las siete cargas del bloque asi:
      //   load_scale(blk) · load_u16_at(qh) · USA las dos -> PARADA
      //   load_byte(qs) · tbl[gi] x2 -> PARADA · y AL FINAL las dos vec4 de x.
      //
      // Cinco de esas siete cargas son INDEPENDIENTES entre si: `load_scale`,
      // `load_u16_at`, `load_byte` (su direccion sale de blk/ib/l, NO del valor
      // de qh) y las dos de activaciones (que no dependen de nada del cuerpo).
      // Y aun asi las tres ultimas se emiten detras de dos esperas evitables,
      // porque el fuente las escribe despues.
      //
      // POR QUE ESTO Y NO OTRA COSA. Las hipotesis anteriores miden trafico,
      // ALU, registros, ocupancia y encadenamiento; ninguna mide el NUMERO DE
      // CARGAS EXPUESTAS. Con la aritmetica del bloque, la latencia de esas
      // siete cargas explicaria la mayor parte de la espera medida, que es el
      // hueco que ningun oraculo explicaba.
      //
      // No cambia NI UNA operacion: mismos valores, mismo orden de acumulacion,
      // bit a bit identico por construccion. Solo cambia en que orden pide las
      // cosas. Si el compilador ya reordenaba, esto sale x1,00 y cierra el
      // angulo — que tambien es un resultado.
      //
      // Se deja FUERA de los caminos de oraculo (modoGi definido) a proposito:
      // esos bancos comparan contra un orden concreto y moverlo les cambiaria el
      // reparto de registros bajo los pies.
      const izaCargas = globalThis.__ELFFUSS_IZA_CARGAS === true
        && xv && modoGi === undefined;

      // LECTURA COOPERATIVA DEL BLOQUE (`__ELFFUSS_BLOQUE_COOP`).
      //
      // El bloque son 50 bytes —escala, 32 de cuantizacion y 16 de qh— y es el
      // MISMO para los 32 carriles: `blk` sale de la fila y del indice de bloque,
      // no del carril. Aun asi cada carril hace TRES lecturas dentro de el. Entre
      // los 32 sobra de largo para traerlo con UNA: 50 bytes son 13 palabras, las
      // trae un carril cada una y luego se reparten con `subgroupShuffle`, que va
      // a velocidad de ALU y no de memoria.
      //
      // POR QUE SE ESPERA QUE ESTO SI Y EL IZADO NO. Reordenar quito CERO cargas
      // y no cambió nada medible: el planificador ya solapaba. Pero
      // la rama `giinv` del oraculo de la cadena, que saca DOS cargas del bucle,
      // vale -19,3%. Las dos medidas juntas dan un modelo simple: el orden es
      // gratis y cada instruccion de carga cuesta ~10% del matmul. Aqui se pasa
      // de tres lecturas de peso a una: -2 cargas, luego la prediccion es ~19%.
      // Si sale muy por debajo, el modelo de «coste por carga» es falso y hay que
      // tirarlo entero, no ajustarlo.
      //
      // El indice se CLAMPA a la ultima palabra del buffer. `blk` no esta alineado
      // a 4 (50 no es multiplo de 4, asi que `blk & 3` alterna 0 y 2) y la palabra
      // 12 del ultimo bloque roza dos bytes mas alla del final. Esos dos bytes no
      // se usan, pero una lectura fuera de rango podria devolver cero y llevarse
      // por delante los dos que SI se usan.
      const bloqueCoop = globalThis.__ELFFUSS_BLOQUE_COOP === true
        && xv && modoGi === undefined;

      const comentarioX = `      // Los 8 pesos del carril caen en DOS vec4 completos: obase = ib*32 + l*8 es
      // multiplo de 8 y k0 = b*256 de 4, luego (k0 + obase) es multiplo de 4 y
      // toda k del decode (5120, 6144, 17408) tambien. Dos cargas vectoriales en
      // vez de ocho escalares con zancada de 32 bytes.`;
      const dirX = `(base_t * params.k${globalThis.__ELFFUSS_X_FIJA === true ? '' : ' + k0 + obase'}) >> 2u`;

      if (bloqueCoop) {
        // `rel(r)` = palabra RELATIVA donde cae el byte `blk + r`, y `des(r, m)`
        // el desplazamiento dentro de ella. Sale de que blk/4 es suelo:
        //   (blk + r)/4 = blk/4 + ((blk&3) + r)/4
        const rel = (r) => `((off + ${r}) >> 2u)`;
        const des = (r, m) => `(((off + ${r}) & ${m}u) * 8u)`;
        return /* wgsl */`
      let ib = sl >> 2u;
      let l = sl & 3u;
      let obase = ib * 32u + l * 8u;
${comentarioX}
      let xb = ${dirX};
      let xv0 = x[xb];
      let xv1 = x[xb + 1u];
      // UNA lectura de peso para todo el subgrupo: 13 carriles traen una palabra
      // cada uno y el bloque entero queda repartido entre los registros.
      let base = blk >> 2u;
      let off = blk & 3u;
      let tope = arrayLength(&w) - 1u;
      let mio = select(0u, w[min(base + sl, tope)], sl < 13u);
      let pqs = subgroupShuffle(mio, ${rel('2u + ib * 4u + l')});
      let pqh = subgroupShuffle(mio, ${rel('34u + ib * 2u')});
      let pd  = subgroupShuffle(mio, ${rel('0u')});
      let gb = (pqs >> ${des('2u + ib * 4u + l', 3)}) & 0xFFu;
      let qh = (pqh >> ${des('34u + ib * 2u', 2)}) & 0xFFFFu;
      let d = unpack2x16float((pd >> ${des('0u', 2)}) & 0xFFFFu)[0];
      let gi = gb | (((qh >> (3u * l)) & 7u) << 8u);
      let g0 = tbl[gi * 2u];
      let g1 = tbl[gi * 2u + 1u];
      let dl = d * f32(2u * ((qh >> 12u) & 7u) + 1u);
      let delta = select(0.125, -0.125, (qh & 0x8000u) != 0u);
${lazo}`;
      }

      if (izaCargas) {
        return /* wgsl */`
      let ib = sl >> 2u;
      let l = sl & 3u;
      let obase = ib * 32u + l * 8u;
${comentarioX}
      // Las CINCO cargas independientes primero, seguidas, sin que ninguna
      // espere a la anterior. Solo despues empieza lo que depende de ellas.
      let xb = ${dirX};
      let xv0 = x[xb];
      let xv1 = x[xb + 1u];
      let d = load_scale(blk);
      let qh = load_u16_at(blk + 34u + ib * 2u);
      let gb = load_byte(blk + 2u + ib * 4u + l);
      let gi = gb | (((qh >> (3u * l)) & 7u) << 8u);
      let g0 = tbl[gi * 2u];
      let g1 = tbl[gi * 2u + 1u];
      let dl = d * f32(2u * ((qh >> 12u) & 7u) + 1u);
      let delta = select(0.125, -0.125, (qh & 0x8000u) != 0u);
${lazo}`;
      }

      return /* wgsl */`
      let d = load_scale(blk);
      let ib = sl >> 2u;
      let l = sl & 3u;
      let qh = load_u16_at(blk + 34u + ib * 2u);
${nucleo}
      let g0 = tbl[gi * 2u];
      let g1 = tbl[gi * 2u + 1u];
      let obase = ib * 32u + l * 8u;
${xv ? `${comentarioX}
      let xb = ${dirX};
      let xv0 = x[xb];
      let xv1 = x[xb + 1u];` : ''}
${lazo}`;
    },

    decode: (emit) => /* wgsl */`
      let d = load_scale(blk);
      for (var ib = 0u; ib < 8u; ib = ib + 1u) {
        let qh = load_u16_at(blk + 34u + ib * 2u);
        let dl = d * f32(2u * ((qh >> 12u) & 7u) + 1u);
        let delta = select(0.125, -0.125, (qh & 0x8000u) != 0u);
        for (var l = 0u; l < 4u; l = l + 1u) {
          let gi = load_byte(blk + 2u + ib * 4u + l) | (((qh >> (3u * l)) & 7u) << 8u);
          let g0 = tbl[gi * 2u];
          let g1 = tbl[gi * 2u + 1u];
          let obase = ib * 32u + l * 8u;
          for (var j = 0u; j < 8u; j = j + 1u) {
            let gw = select(g0, g1, j >= 4u);
            let gv = f32(get_byte_i32(gw, j & 3u));
${emit('obase + j', 'dl * (gv + delta)')}
          }
        }
      }`,
  },
};

// Metadatos por tipo, para quien tenga que reservar buffers o cortar shards.
export const QUANT_TYPES = Object.fromEntries(
  Object.entries(DECODERS).map(([name, d]) => [name, {
    name, blockElems: d.blockElems, blockBytes: d.blockBytes, table: d.table || null,
  }]));

export function quantInfo(type) {
  const d = DECODERS[type];
  if (!d) throw new Error(`tipo de cuantización no soportado: «${type}»`);
  return d;
}

// ── Andamios ────────────────────────────────────────────────────────────────

// Descuantiza un tensor entero a f32. Existe para el test: es el kernel que
// permite comparar contra una referencia de CPU peso a peso, que es donde se
// caza una descodificación mal hecha. En el forward no se usa — ahí descuantiza
// el matmul, sin materializar nada.
export function dequantShader(type, workgroupSize = 64) {
  const d = quantInfo(type);
  const emit = (idx, val) => `            out[dst + (${idx})] = ${val};`;
  return /* wgsl */`
struct Params { nBlocks: u32, blocksPerRow: u32, rows: u32, k: u32 };
@group(0) @binding(0) var<storage, read> w : array<u32>;
@group(0) @binding(1) var<storage, read_write> out : array<f32>;
@group(0) @binding(2) var<uniform> params : Params;
${d.table ? '@group(0) @binding(3) var<storage, read> tbl : array<u32>;' : ''}
${HELPERS}${d.helpers}
@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let b = gid.x + gid.y * ${AXIS}u * ${workgroupSize}u;
  if (b >= params.nBlocks) { return; }
  let blk = b * ${d.blockBytes}u;
  let dst = b * ${d.blockElems}u;
  {${d.decode(emit)}
  }
}`;
}

// y[fila] = Σ_k A[fila][k] · x[k], con A cuantizada y x/y en f32.
// Un workgroup por fila de salida: cada hilo recorre bloques a saltos y luego se
// reduce en árbol por memoria compartida. Es el kernel del bucle de decode, que
// es donde el modelo pasa el 99 % del tiempo.
//
// Con `tokens = 1` esto ES el matvec del decode. Con más, es el matmul del
// prefill: un workgroup por (fila de salida, GRUPO de TILE tokens). Las
// activaciones van token-major (k contiguo por token), igual que las consume
// RoPE, así que no hace falta transponer nada entre operaciones.
// Cuantos SUBGRUPOS por workgroup llevan cada uno SU fila. Vive aqui, junto al
// shader que lo aplica, y lo llama ops.js para el despacho: si el predicado se
// duplicara en los dos sitios y se desincronizaran, saldrian filas a cero sin un
// solo error. Ya paso una vez.
export function sgFilasPorGrupo(type, workgroupSize, tile, carrilSubbloque, rows, subgroup, sgSize = 32) {
  if (!subgroup || tile !== 1) return 1;
  const d = DECODERS[type];
  const porCarril = carrilSubbloque && workgroupSize % sgSize === 0
    && d && typeof d.decodeCarril === 'function';
  if (!porCarril) return 1;
  const s = Math.max(1, Math.floor(workgroupSize / sgSize));
  // `rows` multiplo de S: sin esto el ultimo grupo escribiria fuera de rango.
  if (s === 1 || (Number.isFinite(rows) && rows % s !== 0)) return 1;
  return s;
}

export function matmulShader(type, workgroupSize = 64, tile = 1, subgroup = false, sgSize = 32, carrilSubbloque = false) {
  const d = quantInfo(type);
  // Cuántos subgrupos hay en el workgroup. Con 1 (ancho = tamaño de subgrupo),
  // `subgroupAdd` da la suma entera sin nada más. Con varios (p. ej. 64 hilos =
  // 2 subgrupos de 32), cada subgrupo suma lo suyo y se combinan los pocos
  // parciales por memoria compartida con UNA barrera — aun así muchísimo menos
  // que la reducción en árbol. Más hilos por fila esconden mejor la latencia de
  // las lecturas dependientes de la rejilla i-quant, que es lo que frena IQ1_S.
  const numSg = subgroup ? Math.max(1, Math.floor(workgroupSize / sgSize)) : 1;
  // Solo tiene sentido con un workgroup de 32 carriles (= las 32 unidades de 8
  // pesos que tiene un bloque) y en los tipos que aportan `decodeCarril`.
  const porCarril = carrilSubbloque && workgroupSize % sgSize === 0
    && typeof d.decodeCarril === 'function';
  // Mismo predicado que usa ops.js. El shader NO conoce `rows`, asi que la
  // comprobacion de «rows multiplo de S» la hace SOLO ops.js: aqui se pasa
  // `Infinity` para no fingir que se comprueba algo que no se comprueba. Si
  // ops.js decide 1 por ese motivo, usa wg=32 y aqui sgFilas sale 1 tambien.
  const sgFilas = sgFilasPorGrupo(type, workgroupSize, tile, carrilSubbloque,
    Infinity, subgroup, sgSize);
  // La lectura vectorial de `x` exige TRES cosas: reparto por carril (los 8
  // pesos del carril son contiguos y alineados), guarda izada (tile===1, o sea
  // decode: en prefill haria falta un vec4 por token) y el interruptor.
  const xv = globalThis.__ELFFUSS_X_VEC4 !== false && porCarril
    && tile === 1 && !globalThis.__ELFFUSS_GUARDA_POR_PESO;
  // Reducción por SUBGRUPO en vez de en árbol. La versión de árbol reduce los
  // acumuladores de los `workgroupSize` hilos con memoria compartida y una
  // barrera por paso (6 con 64 hilos). Si el workgroup es EXACTAMENTE un subgrupo
  // (ops.js lo garantiza fijando el ancho al tamaño de subgrupo del dispositivo,
  // 32 en muchos adaptadores), `subgroupAdd` hace esa misma suma en una instrucción, sin
  // barreras ni memoria compartida. En el matvec del decode la reducción pesa,
  // así que esto va al grano. La numérica es la misma suma (mismo orden no está
  // garantizado a nivel de bit, pero es una suma de flotantes dentro de la misma
  // tolerancia que valida la prueba de cuantización).
  // Un acumulador por token del grupo, en variables SUELTAS y no en un array.
  // Con array<f32,N> e índice dinámico, el compilador no puede dejarlo en
  // registros y lo baja a memoria: cada acumulación pasa a ser un acceso, y el
  // teselado deja de compensar. Desenrollado, son registros de verdad.
  // El valor se ata a un `let` ANTES de repartirlo entre los acumuladores. Si se
  // interpolase la expresión dentro de cada línea, un tile de 8 descuantizaría
  // el mismo peso ocho veces y el teselado costaría más de lo que ahorra.
  // La guarda de token se IZA cuando no puede ser falsa. Con tile=1 el matmul se
  // llama siempre con tokens=1 (ops.js: `tile = tokens > 1 ? TILE_PREFILL : 1`),
  // el despacho lanza ceil(1/1)=1 grupo en z, luego base_t=0 y `0 < 1` es cierto
  // SIEMPRE. Aun asi se evaluaba una vez POR PESO: 20.510 millones de sumas y
  // comparaciones por token que no deciden nada.
  //
  // No es cosmetica: el perfil por forma dice que este kernel esta limitado por
  // EMISION de instrucciones (el rendimiento sigue a la utilizacion de carriles,
  // no es plano como seria si mandaran las transacciones de memoria). Con ~11
  // operaciones-carril por peso, quitar una es ~9% del trabajo emitido, y ademas
  // es el cambio mas limpio posible: no desenrolla, no toca registros, no cambia
  // el orden de acumulacion. Bit a bit identico por construccion.
  //
  // En prefill (tile>1) la guarda SI puede ser falsa —el ultimo grupo de tokens
  // va incompleto— y se conserva tal cual.
  // AVISO para quien quiera izar la guarda tambien en prefill: la condicion
  // natural seria `tokens % tile === 0`, y entonces la clave de pipeline de
  // ops.js TENDRIA que llevarla. Hoy la clave lleva el `tile` pero no si la
  // guarda se izo, y no hace falta porque `tile === 1` equivale a
  // `tokens === 1` (ops.js: `tile = tokens > 1 ? TILE_PREFILL : 1`), donde
  // base_t = 0 < 1 siempre. Si se relaja esa condicion sin tocar la clave, un
  // shader compilado para un lote completo se reutilizaria en el ultimo lote
  // incompleto y descartaria tokens EN SILENCIO.
  const guardaIzable = tile === 1 && !globalThis.__ELFFUSS_GUARDA_POR_PESO;
  // ORACULO DEL COSTE DE `x` — solo para medir, NUNCA en produccion.
  //
  // Una medida interna de patrones de acceso dice que nuestro acceso a `x` (cada
  // carril con zancada de 8 flotantes) rinde x0,31 de un acceso coalescido en
  // cache, y que vectorizarlo daria x4,11. Pero x4,11 sobre el termino de `x`
  // solo importa si ese termino pesa: sobre el 10% del tiempo no es nada, sobre
  // el 60% lo cambia todo. Eso no se puede estimar, hay que medirlo.
  //
  // Con el interruptor puesto, los 32 carriles leen la MISMA posicion de `x` en
  // vez de 32 con zancada. El numero de FMA no cambia; lo que desaparece es la
  // carga divergente (y ademas queda invariante del bucle, asi que el compilador
  // la iza fuera). La diferencia de tiempo contra el kernel normal ES el coste
  // del termino de `x`.
  //
  // Se lee una posicion FIJA en lugar de quitar la lectura a propósito: sin
  // ninguna referencia a `x`, `layout:'auto'` deduciria un layout sin el binding
  // 1 y el bind group dejaria de ser valido — sin lanzar, devolviendo ceros.
  // Devuelve resultados NUMERICAMENTE FALSOS por diseño.
  const xFija = globalThis.__ELFFUSS_X_FIJA === true;
  // OJO CON LA ETIQUETA: esto NO mide «traer pesos» en el sentido de las
  // instrucciones de carga — esas se siguen emitiendo igual. Al fijar la
  // direccion, mide el coste del TRAFICO de pesos distintos: fallos de cache
  // y DRAM. Y CONTIENE a DEQ_PLANA: al perder `blk` el `+ b`, todo lo que
  // cuelga de `blk` queda invariante del bucle. Que DEQ_PLANA mida MAS que
  // este (15,5% > 7,4%) es imposible => uno de los dos esta roto, y lo mas
  // probable es que este salga deprimido porque `blk` constante Y CONOCIDO
  // deja izar el lado del peso entero. NO sumar los dos: cuenta dos veces.
  const wFija = globalThis.__ELFFUSS_W_FIJA === true;
  const xVec4 = globalThis.__ELFFUSS_X_VEC4 === true;
  const desp = xFija ? '' : ' + kk';
  // Con `xv`, los 8 flotantes del carril ya estan en dos vec4 leidos fuera del
  // bucle, asi que aqui solo queda el FMA. Se usa SWIZZLE (.x/.y/.z/.w) y no
  // indice: indexar un vec4 declarado con `let` solo es legal con constante, y
  // el swizzle no tiene esa ambiguedad. `j` solo llega del cuerpo desenrollado.
  const COMP = ['x', 'y', 'z', 'w'];
  // ORACULO DE LA CADENA DEL ACUMULADOR (`__ELFFUSS_ACUMS`) — mide, y ademas
  // PODRIA quedarse si gana (no es bit a bit: cambia la particion de la suma).
  //
  // Hoy `acc0` encadena bpr*8 FMA en serie: 160 de profundidad con bpr=20. Cada
  // FMA tiene que esperar al anterior. Con N acumuladores independientes son N
  // cadenas de 160/N que pueden solaparse, y al final se refunden.
  //
  // Solo tiene sentido en el camino vec4 (decode, reparto por carril), que es
  // donde `j` es constante de compilacion y hay un unico acumulador de token.
  // Con N=1 el texto generado es IDENTICO al de produccion, byte a byte: es el
  // control de la tanda y a la vez la garantia de que el interruptor apagado no
  // toca nada.
  // Lastre de registros: N acumuladores vivos de mas, plegados al final por
  // `params.pad0` (=0 en ejecucion, desconocido para el compilador). Bit a bit
  // exacto: suma cero. Solo para la escalera de ocupancia.
  const nlastre = Math.min(32, Math.max(0, (globalThis.__ELFFUSS_LASTRE | 0) || 0));
  const declLastre = nlastre > 0
    ? '\n' + Array.from({ length: nlastre }, (_, i) => `  var lastre${i} = f32(${i + 1}u);`).join('\n')
    : '';
  const fundeLastre = nlastre > 0
    ? `\n  acc0 = acc0 + (${Array.from({ length: nlastre }, (_, i) => `lastre${i}`).join(' + ')}) * f32(params.pad0);`
    : '';
  const nacum = xv ? Math.min(8, Math.max(1, (globalThis.__ELFFUSS_ACUMS | 0) || 1)) : 1;
  const AC = (j) => (nacum > 1 && (j % nacum) !== 0 ? `accb${j % nacum}` : 'acc0');
  const declAcum = nacum > 1
    ? '\n' + Array.from({ length: nacum - 1 }, (_, i) => `  var accb${i + 1} = 0.0;`).join('\n')
    : '';
  // Refundido en ARBOL y no en cascada: en cascada volveria a crear una cadena
  // serie de N-1 sumas justo donde se quiere demostrar que las cadenas cuestan.
  const arbol = (ns) => (ns.length === 1 ? ns[0]
    : `(${arbol(ns.slice(0, ns.length >> 1))} + ${arbol(ns.slice(ns.length >> 1))})`);
  const fundeAcum = nacum > 1
    ? `\n  acc0 = ${arbol(['acc0', ...Array.from({ length: nacum - 1 }, (_, i) => `accb${i + 1}`)])};`
    : '';
  const emit = (idx, val, j) => (xv && j !== undefined
    ? [
      '            {',
      `              let wv = ${val};`,
      `              ${AC(j)} = ${AC(j)} + wv * ${j < 4 ? 'xv0' : 'xv1'}.${COMP[j % 4]};`,
      '            }',
    ]
    : [
      '            {',
      `              let wv = ${val};`,
      `              let kk = k0 + (${idx});`,
      ...(guardaIzable
        ? [`              acc0 = acc0 + wv * x[base_t * params.k${desp}];`]
        : Array.from({ length: tile }, (_, t) =>
          `              if (base_t + ${t}u < params.tokens) { acc${t} = acc${t} + wv * ` +
          `x[(base_t + ${t}u) * params.k${desp}]; }`)),
      '            }',
    ]).join('\n');

  return /* wgsl */`${subgroup ? 'enable subgroups;\n' : ''}
const TILE : u32 = ${tile}u;
struct Params { nBlocks: u32, blocksPerRow: u32, rows: u32, k: u32,
                tokens: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> w : array<u32>;
${xv ? '@group(0) @binding(1) var<storage, read> x : array<vec4<f32>>;' : '@group(0) @binding(1) var<storage, read> x : array<f32>;'}
@group(0) @binding(2) var<storage, read_write> y : array<f32>;
@group(0) @binding(3) var<uniform> params : Params;
${d.table ? '@group(0) @binding(4) var<storage, read> tbl : array<u32>;' : ''}
${HELPERS}${d.helpers}
${subgroup ? (numSg > 1 && sgFilas === 1 ? `var<workgroup> sgpart : array<f32, ${numSg}u>;` : '') : `var<workgroup> partial : array<f32, ${workgroupSize}>;`}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>${subgroup && numSg > 1 ? ',\n        @builtin(subgroup_invocation_id) sglane : u32' : ''}) {
  // Las filas van en dos ejes porque uno solo topa en 65.535 y la cabeza de
  // salida son 248.320 filas. El tercer eje son GRUPOS de TILE tokens.
  // Carril LOCAL al subgrupo: con varios subgrupos por workgroup, lid.x pasa de
  // 31 y el reparto por carril (ib, l) se saldria de rango. Con un subgrupo es
  // exactamente lid.x.
  ${porCarril ? `let sl = lid.x & ${sgSize - 1}u;` : '// sin reparto por carril no hace falta el indice de carril (ver kernels.wgsl.js)'}
  ${sgFilas > 1
    ? `// Una fila por SUBGRUPO: cada punto de la rejilla son ${sgFilas} filas.
  let row = (wid.x + wid.y * ${AXIS}u) * ${sgFilas}u + lid.x / ${sgSize}u;`
    : `let row = wid.x + wid.y * ${AXIS}u;`}
  let base_t = wid.z * TILE;
  let bpr = params.blocksPerRow;

${Array.from({ length: tile }, (_, t) => `  var acc${t} = 0.0;`).join('\n')}${declAcum}${declLastre}

  if (row < params.rows) {
${porCarril
  ? `    // Reparto por SUB-GRUPO: todos los carriles avanzan por el MISMO bloque y
    // cada uno se ocupa de 8 pesos. Cero carriles ociosos sea cual sea bpr, y de
    // paso los 32 carriles leen un tramo CONTIGUO de activaciones en vez de 32
    // tramos dispersos. Ver decodeCarril.
    for (var b = 0u; b < bpr; b = b + 1u) {`
  : `    for (var b = lid.x; b < bpr; b = b + ${workgroupSize}u) {`}
      let blk = (row * bpr${wFija ? '' : ' + b'}) * ${d.blockBytes}u;
      let k0 = b * ${d.blockElems}u;
      {${porCarril ? d.decodeCarril(emit, xv) : d.decode(emit)}
      }
    }
  }${fundeAcum}${fundeLastre}

  // Reducción por token del grupo. Con subgrupo, una instrucción por token; en
  // árbol, memoria compartida y una barrera por paso. El recorrido del árbol es
  // constante de compilación, así que las barreras quedan en flujo uniforme.
${subgroup
  ? (sgFilas > 1
    ? Array.from({ length: tile }, (_, t) => `  {
    let tot${t} = subgroupAdd(acc${t});
    if (sglane == 0u && row < params.rows && base_t + ${t}u < params.tokens) {
      y[(base_t + ${t}u) * params.rows + row] = tot${t};
    }
  }`).join('\n')
    : numSg === 1
    ? Array.from({ length: tile }, (_, t) => `  {
    let tot${t} = subgroupAdd(acc${t});
    if (lid.x == 0u && row < params.rows && base_t + ${t}u < params.tokens) {
      y[(base_t + ${t}u) * params.rows + row] = tot${t};
    }
  }`).join('\n')
    : Array.from({ length: tile }, (_, t) => `  {
    let s${t} = subgroupAdd(acc${t});
    if (sglane == 0u) { sgpart[lid.x / ${sgSize}u] = s${t}; }
    workgroupBarrier();
    if (lid.x == 0u && row < params.rows && base_t + ${t}u < params.tokens) {
      var tot${t} = 0.0;
      for (var q = 0u; q < ${numSg}u; q = q + 1u) { tot${t} = tot${t} + sgpart[q]; }
      y[(base_t + ${t}u) * params.rows + row] = tot${t};
    }
    workgroupBarrier();
  }`).join('\n'))
  : Array.from({ length: tile }, (_, t) => `  partial[lid.x] = acc${t};
  workgroupBarrier();
  for (var s${t} = ${workgroupSize >> 1}u; s${t} > 0u; s${t} = s${t} >> 1u) {
    if (lid.x < s${t}) { partial[lid.x] = partial[lid.x] + partial[lid.x + s${t}]; }
    workgroupBarrier();
  }
  if (lid.x == 0u && row < params.rows && base_t + ${t}u < params.tokens) {
    y[(base_t + ${t}u) * params.rows + row] = partial[0];
  }
  workgroupBarrier();            // WAR: el token siguiente reescribe partial`).join('\n')}
}`;
}

// Busca las filas `ids` de una tabla cuantizada y las deja en f32.
// La tabla de embeddings es de los tensores más grandes del modelo: descuantizarla
// entera para leer unas pocas filas costaría muchas veces lo que se lee.
//
// El grano es el BLOQUE, no el elemento. Sacar un peso suelto de un K-quant
// obligaría a escribir un segundo descodificador «de un solo elemento» por
// tipo, distinto del que usa el matmul — o sea, seis oportunidades más de
// equivocarse en el layout. Descodificando bloques enteros se reutiliza tal cual
// el descodificador ya verificado, y lo único que se paga es descartar los pesos
// del bloque que no hacían falta, que en un embedding es ninguno: se quiere la
// fila entera.
export function gatherShader(type, workgroupSize = 64) {
  const d = quantInfo(type);
  const emit = (idx, val) => `            dst[base + (${idx})] = ${val};`;
  return /* wgsl */`
struct P { dim : u32, tokens : u32, blocksPerRow : u32, pad : u32 };
@group(0) @binding(0) var<storage, read>       w   : array<u32>;
@group(0) @binding(1) var<storage, read>       ids : array<u32>;
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;
@group(0) @binding(3) var<uniform>             p   : P;
${d.table ? '@group(0) @binding(4) var<storage, read> tbl : array<u32>;' : ''}
${HELPERS}${d.helpers}
@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x + gid.y * ${AXIS}u * ${workgroupSize}u;
  if (i >= p.tokens * p.blocksPerRow) { return; }
  let t = i / p.blocksPerRow;
  let b = i % p.blocksPerRow;
  let blk = (ids[t] * p.blocksPerRow + b) * ${d.blockBytes}u;
  let base = t * p.dim + b * ${d.blockElems}u;
  {${d.decode(emit)}
  }
}`;
}

// Los alias `dequantQ8Shader`/`matmulQ8Shader`/`matvecQ8Shader` ya no existen:
// nadie los llamaba una vez que `ops.js` pasa el tipo, y dejarlos puestos habría
// sido una compatibilidad con nada.
