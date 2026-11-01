// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · lectura de bloques Q8_0 y Q4_0, en el formato del GGUF y en
// el formato REORDENADO al cargar (tipos internos Q8_0R y Q4_0R).
// ─────────────────────────────────────────────────────────────────────────────
// POR QUÉ HAY UN FORMATO PROPIO. En el GGUF un bloque Q8_0 son 34 bytes (escala
// f16 + 32 int8) y uno Q4_0 son 18 (escala + 16 bytes de nibbles). Ninguno de
// los dos es múltiplo de 4: la mitad de los bloques empieza a media palabra, y
// cada palabra de cuantos sale de DOS lecturas, dos desplazamientos y un
// select. El GEMV del decode va limitado por las instrucciones que emite, no por
// la memoria, así que esa recomposición se paga entera, peso a peso.
//
// El formato reordenado deja los MISMOS bytes en otro orden: primero los
// cuantos de todos los bloques, uno tras otro y alineados a 16 bytes; después
// todas las escalas. Ni un byte de más (salvo redondear a 4) ni un valor
// distinto: cambia dónde vive cada cosa, no qué vale. Por eso un lector que lea
// el bloque i en los dos formatos saca los mismos bits.
//
// ── EL CONTRATO (lo leen varios kernels escritos por separado: no se toca sin
// cambiarlos a todos) ─────────────────────────────────────────────────────────
// Sea N el número de bloques del TENSOR (filas × bloques por fila) e
// i = fila·bpr + b el bloque:
//   · Q8_0R: los cuantos del bloque i en los bytes [32i, 32i + 32), con los 32
//     int8 en el orden del GGUF (palabras u32 8i … 8i+7, o sea vec4<u32> 2i y
//     2i+1). Las escalas, a partir de la palabra W = 8N: la del bloque i es la
//     media palabra i de esa región, escala(i) = unpack2x16float(w[W + i/2])[i & 1].
//   · Q4_0R: los cuantos en [16i, 16i + 16) (vec4<u32> i), con el MISMO orden
//     de nibbles que Q4_0 (byte j: peso j en el nibble bajo y peso j+16 en el
//     alto; valor (q − 8)·d). Escalas desde W = 4N, igual que arriba.
//   · Tamaño del formato: 34N (o 18N) redondeado a múltiplo de 4.
//   · Tamaño del BUFFER: el formato redondeado a 16 bytes (`bytesBufferR`), o
//     sea entre cero y tres palabras de holgura detrás de la última escala. Es
//     lo que crea el cargador, y es a propósito: deja que un lector vec4<u32>
//     llegue a la última palabra de escalas. Nadie puede suponer que el buffer
//     mide justo el formato, ni buscar las escalas contando desde el final.
//
// ── DÓNDE EMPIEZAN LAS ESCALAS: lo dice el propio buffer ─────────────────────
// W depende de N, que es el número de bloques del TENSOR y no el del trozo que
// lea un despacho. La cabeza de salida, el gather de embeddings o una
// descuantización parcial leen un subconjunto de filas, y las escalas siguen
// donde las puso el tensor entero. Así que N no puede salir de los uniformes:
// el `rows·bpr` del matmul es el de la LLAMADA, y el gather ni lo tiene. Y no
// es un caso teórico: la cabeza se multiplica con `rows` = el vocabulario del
// tokenizador, y hay familias cuya tabla trae filas de relleno por encima de
// él; un W sacado de ese `rows` leería todas las escalas de la cabeza de otro
// sitio, sin un error. Sale del tamaño del buffer. Con L = arrayLength(&w)
// palabras,
//     Q8_0R:  N = ⌊2L/17⌋        Q4_0R:  N = ⌊2L/9⌋
// y es EXACTO mientras el buffer tenga como mucho tres palabras de holgura sobre
// el tamaño redondeado a 4: la cuenta está en `bloquesDeBuffer` y una prueba de
// node la recorre entera. El buffer de `bytesBufferR` deja entre cero y tres.
//
// Dos condiciones: el buffer se ata ENTERO (sin offset ni tamaño en el bind
// group) y mide lo que dice `bytesBufferR`. No se dejan a la buena fe: cada
// buffer reordenado lleva una MARCA con su N (la pone `creaBufferR`, en
// repack.js) y las operaciones de ops.js no despachan un lector R sobre un
// buffer sin ella (`compruebaBufferR`). Quien conozca N por otra vía puede pasar
// la expresión (`escalasDesde`) y no depender del tamaño; tiene que ser el N del
// TENSOR (`bloquesR(buf)` lo da), nunca el `rows·bpr` de la llamada.
//
// ── LAS FUNCIONES ──────────────────────────────────────────────────────────
// `bloquesWGSL(tipo)` da, para los cuatro tipos, las mismas tres funciones:
//   bq_escala(i)      la escala f16 del bloque i, ya en f32 (conversión exacta)
//   bq_palabra(i, j)  la palabra j de sus cuantos (j < 8 en Q8, j < 4 en Q4),
//                     con los bytes en el orden del GGUF
//   bq_cuantos(i, h)  las palabras 4·h … 4·h + 3 de una vez (h < 2 en Q8, h = 0 en Q4)
// y en los reordenados, además, bq_escalas_desde() (la palabra W). Un kernel
// que lea los pesos SOLO a través de ellas cambia de formato sin tocar el bucle.
//
// Dos opciones para los kernels que no son el matmul de siempre:
//   · `sufijo`: se añade a los cuatro nombres (`sufijo: '_u'` da bq_escala_u…).
//     WGSL no deja pasar un puntero a storage a una función, así que cada
//     función lee de UN almacén con nombre fijo: un kernel que lea dos o más
//     pesos (gate y up a la vez, Q/K/V juntas, o un Q8_0R con un Q4_0R) incluye
//     el bloque una vez por peso, cada una con su `w` y su sufijo. Sin sufijo,
//     el texto es el de siempre.
//   · `vista: 'vec4'`: el almacén se declara `array<vec4<u32>>` y bq_cuantos es
//     UNA lectura alineada de 16 bytes, que es para lo que existe el formato.
//     Solo en los reordenados (en el GGUF los bloques no caen alineados a 16).
//     Ahí arrayLength cuenta vec4, así que L = 4·arrayLength: exacto porque el
//     buffer de `bytesBufferR` es múltiplo de 16 (con otro tamaño, arrayLength
//     redondearía hacia abajo y se perdería la cola de escalas).
//
// Este módulo es texto y tablas, sin importar nada: lo importan los kernels, la
// referencia de CPU y el cargador, y ninguno puede arrastrar a los demás.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.

// Los tipos internos. Nunca aparecen en un GGUF: los pone el cargador cuando
// reordena, y viajan en el descriptor del tensor (`typeName`).
export const TIPOS_R = new Set(['Q8_0R', 'Q4_0R']);

// Tipo del GGUF → su tipo reordenado, y al revés.
export const TIPO_R = Object.freeze({ Q8_0: 'Q8_0R', Q4_0: 'Q4_0R' });
export const TIPO_GGUF = Object.freeze({ Q8_0R: 'Q8_0', Q4_0R: 'Q4_0' });

// Geometría del bloque. `bytes` es lo que ocupa en el GGUF, y también lo que
// ocupa DE MEDIA en el reordenado (cuantos + escala); `cuantos` son los bytes
// de cuantos por bloque y `palabras` las mismas en u32.
const G8 = Object.freeze({ elems: 32, bytes: 34, cuantos: 32, palabras: 8 });
const G4 = Object.freeze({ elems: 32, bytes: 18, cuantos: 16, palabras: 4 });
export const GEOMETRIA = Object.freeze({ Q8_0: G8, Q8_0R: G8, Q4_0: G4, Q4_0R: G4 });

function geo(tipo) {
  const g = GEOMETRIA[tipo];
  if (!g) throw new Error(`bloques: tipo sin geometría: «${tipo}»`);
  return g;
}

// Bytes del formato reordenado de N bloques: los del GGUF, redondeados a 4.
export function bytesReordenado(tipo, nBloques) {
  return Math.ceil(geo(tipo).bytes * nBloques / 4) * 4;
}

// Bytes del BUFFER que crea el cargador: redondeado a 16. Deja entre cero y tres
// palabras de holgura (lo que tolera la deducción de N) y que un lector vec4
// alcance la última palabra de escalas.
export function bytesBufferR(tipo, nBloques) {
  return Math.ceil(geo(tipo).bytes * nBloques / 16) * 16;
}

// Primera palabra de la región de escalas de un tensor de N bloques.
export function escalasDesde(tipo, nBloques) {
  return geo(tipo).palabras * nBloques;
}

// Espejo en JS de lo que hace el shader: N a partir del tamaño del buffer atado.
// Con k palabras de holgura sobre ⌈bytes·N/4⌉ (r = 1 si bytes·N/4 no es entero):
//   Q8_0R: 2L = 17N + r + 2k → ⌊2L/17⌋ = N  mientras r + 2k < 17  (k ≤ 7)
//   Q4_0R: 2L =  9N + r + 2k → ⌊2L/9⌋  = N  mientras r + 2k < 9   (k ≤ 3)
// Q4_0R es el que manda: tres palabras como mucho.
export function bloquesDeBuffer(tipo, bytesBuffer) {
  const L = Math.floor(bytesBuffer / 4);
  return Math.floor((2 * L) / (geo(tipo).cuantos === 32 ? 17 : 9));
}

/**
 * Funciones WGSL para leer el bloque `i` de un tensor de tipo `tipo` guardado en
 * el almacén `w`, declarado por quien incluya esto: `array<u32>`, o
 * `array<vec4<u32>>` con `vista: 'vec4'` (solo en los reordenados).
 *
 * `escalasDesde`: expresión WGSL que da W (la primera palabra u32 de escalas,
 * también en la vista vec4) para quien sepa el N del TENSOR sin mirar el
 * buffer; solo en los reordenados. Por defecto se deduce de `arrayLength` (ver
 * la cabecera).
 *
 * `sufijo`: se añade a los nombres de las funciones, para incluir el bloque una
 * vez por almacén en un kernel que lea varios pesos.
 */
export function bloquesWGSL(tipo, { w = 'w', escalasDesde: base = null, sufijo = '', vista = 'u32' } = {}) {
  const g = geo(tipo);
  const P = g.palabras;
  // Los nombres van pegados al WGSL: uno mal formado daría un shader que no
  // compila lejos de aquí, o peor, uno que compila leyendo otra cosa.
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(w)) throw new Error(`bloquesWGSL: nombre de almacén inválido: «${w}»`);
  if (!/^[A-Za-z0-9_]*$/.test(sufijo)) throw new Error(`bloquesWGSL: sufijo inválido: «${sufijo}»`);
  if (vista !== 'u32' && vista !== 'vec4') throw new Error(`bloquesWGSL: vista desconocida: «${vista}» (u32 o vec4)`);
  const s = sufijo;
  if (TIPOS_R.has(tipo)) {
    const den = g.cuantos === 32 ? 17 : 9;
    if (vista === 'vec4') {
      // Con arrayLength en vec4: L = 4·arrayLength palabras, así que 2L = 8·arrayLength.
      const desde4 = base || `((8u * arrayLength(&${w})) / ${den}u) * ${P}u`;
      return /* wgsl */`
// ${tipo} (vista vec4<u32>): cuantos del bloque i en los vec4 [${P / 4}i, ${P / 4}i + ${P / 4}); escalas desde la palabra W = ${P}N.
fn bq_escalas_desde${s}() -> u32 {
  return ${desde4};
}
fn bq_escala${s}(i: u32) -> f32 {
  let bq_p = bq_escalas_desde${s}() + (i >> 1u);
  let bq_par = ${w}[bq_p >> 2u][bq_p & 3u];
  return unpack2x16float((bq_par >> ((i & 1u) * 16u)) & 0xFFFFu)[0];
}
fn bq_palabra${s}(i: u32, j: u32) -> u32 {
  let bq_p = i * ${P}u + j;
  return ${w}[bq_p >> 2u][bq_p & 3u];
}
fn bq_cuantos${s}(i: u32, h: u32) -> vec4<u32> {
  return ${w}[i * ${P / 4}u + h];
}
`;
    }
    const desde = base || `((2u * arrayLength(&${w})) / ${den}u) * ${P}u`;
    // La escala se saca con el MISMO desplazamiento-y-máscara que `load_scale`
    // del formato GGUF, no indexando el vec2 de `unpack2x16float`: las dos son
    // exactas, pero así el f32 sale literalmente de la misma instrucción.
    return /* wgsl */`
// ${tipo}: cuantos del bloque i en las palabras [${P}i, ${P}i + ${P}); escalas desde W = ${P}N.
fn bq_escalas_desde${s}() -> u32 {
  return ${desde};
}
fn bq_escala${s}(i: u32) -> f32 {
  let bq_par = ${w}[bq_escalas_desde${s}() + (i >> 1u)];
  return unpack2x16float((bq_par >> ((i & 1u) * 16u)) & 0xFFFFu)[0];
}
fn bq_palabra${s}(i: u32, j: u32) -> u32 {
  return ${w}[i * ${P}u + j];
}
fn bq_cuantos${s}(i: u32, h: u32) -> vec4<u32> {
  let bq_ix = i * ${P}u + h * 4u;
  return vec4<u32>(${w}[bq_ix], ${w}[bq_ix + 1u], ${w}[bq_ix + 2u], ${w}[bq_ix + 3u]);
}
`;
  }
  if (vista === 'vec4') {
    throw new Error(`bloquesWGSL: la vista vec4 es solo de los reordenados (Q8_0R, Q4_0R); ` +
      `en el formato del GGUF los bloques de «${tipo}» no caen alineados a 16 bytes`);
  }
  // Formato del GGUF: bloques de 34 (o 18) bytes, la escala delante. La escala
  // cae siempre en un offset par (nunca cruza palabra); los cuantos empiezan a
  // media palabra en la mitad de los bloques y se recomponen con dos lecturas,
  // como `load_u32_at`. El «& 31u» evita el desplazamiento de 32 bits, que en
  // WGSL es indeterminado; con s == 0 el select se queda con `lo`. Igual que en
  // `load_u32_at`, el último bloque lee la palabra SIGUIENTE al tensor: el
  // buffer necesita la holgura de 4 bytes que ya le da el cargador.
  const B = g.bytes;
  return /* wgsl */`
// ${tipo} (formato GGUF): el bloque i empieza en el byte ${B}·i.
fn bq_escala${s}(i: u32) -> f32 {
  let bq_o = i * ${B}u;
  return unpack2x16float((${w}[bq_o >> 2u] >> ((bq_o & 2u) * 8u)) & 0xFFFFu)[0];
}
fn bq_palabra${s}(i: u32, j: u32) -> u32 {
  let bq_o = i * ${B}u + 2u + j * 4u;
  let bq_s = (bq_o & 3u) * 8u;
  let bq_lo = ${w}[bq_o >> 2u];
  let bq_hi = ${w}[(bq_o >> 2u) + 1u];
  return select((bq_lo >> bq_s) | (bq_hi << ((32u - bq_s) & 31u)), bq_lo, bq_s == 0u);
}
fn bq_cuantos${s}(i: u32, h: u32) -> vec4<u32> {
  let bq_j = h * 4u;
  return vec4<u32>(bq_palabra${s}(i, bq_j), bq_palabra${s}(i, bq_j + 1u), bq_palabra${s}(i, bq_j + 2u), bq_palabra${s}(i, bq_j + 3u));
}
`;
}
