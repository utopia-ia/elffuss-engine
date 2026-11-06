// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · GEMV del decode, segunda generación (`__ELFFUSS_GEMV2`).
// ─────────────────────────────────────────────────────────────────────────────
// POR QUÉ OTRO GEMV. El de `kernels.wgsl.js` es el andamio genérico: un bloque
// entero por carril, `x` escalar con una zancada de un bloque entre carriles,
// la escala multiplicada en cada peso y una sola fila por grupo. Con Q8_0 y Q4_0
// eso lo deja limitado por emisión de instrucciones y por la caché, no por la
// memoria: por eso Q4_0 iba igual que Q8_0 aunque lee casi la mitad de bytes. Y
// la cabeza Q6_K del fichero de 4 bits dejaba tres cuartos de los carriles
// mirando, con cada carril activo descodificando un bloque de 256 pesos en serie.
//
// Lo que cambia aquí, en el orden en que pesa:
//   · varios carriles por bloque y bloques CONSECUTIVOS en carriles
//     consecutivos: un paso de subgrupo cubre 512 pesos seguidos de la fila y
//     los 32 carriles leen un tramo contiguo de `x`, en vec4;
//   · `x` se lee UNA vez por paso y se reutiliza en las R filas del grupo (y en
//     las dos matrices de gate/up, o en los T tokens): el tráfico de
//     activaciones se divide entre R;
//   · la escala del bloque se aplica UNA vez sobre Σq·x, no peso a peso, y en
//     Q4_0 el −8 se pliega como d·(Σq·x − 8·Σx), con Σx calculada una vez por
//     paso para todas las filas;
//   · en Q4_0 y Q6_K los cuantos no se desplazan: se enmascaran en su sitio y
//     la activación llega PREESCALADA por la potencia de dos que falta (1,
//     2^−8, 2^−16…). El producto es exactamente el mismo —escalar por una
//     potencia de dos no redondea, salvo que lleve una activación minúscula a
//     subnormal— y cada peso cuesta una máscara, una conversión y una
//     multiplicación-suma;
//   · con pocas filas, K se reparte entre varios subgrupos del grupo
//     (`NSG`), para que el dispositivo no se quede sin trabajo.
//
// ── EL ORDEN DE LA SUMA ES CANÓNICO ─────────────────────────────────────────
// La suma de una fila depende SOLO del formato y de K, nunca de cuántas filas
// lleva el grupo (R), de cuántos subgrupos se reparten K (NSG), de cuántos
// tokens van a la vez (T) ni de si la matriz va sola o fusionada con otras:
//   1. K se parte en pasos de 512 pesos; el paso `s` es de la partición
//      `s mod GEMV2_PARTES`.
//   2. Cada carril acumula, partición a partición, sus contribuciones en orden
//      creciente de `s`.
//   3. Cada carril junta sus particiones en orden: ((p0 + p1) + p2) + p3.
//   4. `subgroupAdd` de los 32 carriles.
// Con NSG = 1 las cuatro particiones viven en registros del mismo carril; con
// NSG > 1 cada subgrupo lleva las suyas y se juntan por memoria de grupo, pero
// el punto 3 se hace igual, carril a carril. Por eso la versión fusionada
// gate+up+SwiGLU y la de Q/K/V/puerta dan BIT A BIT lo mismo que las matrices
// sueltas con este mismo kernel, y la disposición reordenada lo mismo que la
// del GGUF. Contra el kernel de siempre NO es bit a bit: cambia la partición
// de la suma (reasociación), y se valida con tolerancia.
//
// ── DISPOSICIONES DE PESOS ──────────────────────────────────────────────────
//   · GGUF Q8_0 (34 bytes) y Q4_0 (18): el bloque empieza en múltiplo de dos;
//     la mitad de los tramos de cuantos caen a media palabra y se recomponen
//     con dos palabras (`junta`). La escala nunca cruza palabra.
//   · Reordenados al cargar, Q8_0R y Q4_0R: cuantos alineados a 16 bytes en
//     una región y las escalas f16 en otra que empieza en la palabra `esc`. Un
//     carril lee sus 16 cuantos de Q8_0R con UNA carga vec4<u32> y los 8 bytes
//     de Q4_0R con una vec2<u32>.
//   · Q6_K (210 bytes) se queda como en el GGUF: 16 carriles por bloque, cada
//     uno con 16 pesos (4 posiciones × 4 grupos de 32), la disposición de la
//     cabeza que más rinde en otros motores.
// Las lecturas de bloque son locales a este módulo a propósito: otro módulo
// lleva los lectores comunes, y así cada uno cambia sin pisar al otro.
//
// Requisitos: subgrupos de 32 carriles exactos y subgrupos formados por
// invocaciones consecutivas (lo mismo que ya supone `kernels.wgsl.js`). Quien
// no los tenga sigue con el camino de siempre (ver `ops.gemv.js`).
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.

import { AXIS } from './kernels.wgsl.js';

// Particiones canónicas de K (ver la cabecera). Va en el ORDEN de la suma: si
// cambia, cambian los bits de todas las filas, así que no es un parámetro de
// forma sino parte del formato del resultado.
export const GEMV2_PARTES = 4;

// Tokens que caben en un despacho: el decode es 1, y 2-3 son los turnos cortos
// que hoy caen al camino de uno en uno. A partir de 4 manda el GEMM teselado.
export const GEMV2_MAX_TOKENS = 3;

// Por formato: elementos y bytes del bloque, bloques que cubre un subgrupo en
// un paso (siempre 512 pesos), carriles por bloque y cómo se declara el búfer.
const FORMATOS = {
  Q8_0: { bloque: 32, bytes: 34, bloquesPaso: 16, carrilesBloque: 2, repack: false, decl: 'array<u32>' },
  Q8_0R: { bloque: 32, bytes: 34, bloquesPaso: 16, carrilesBloque: 2, repack: true, decl: 'array<vec4<u32>>' },
  Q4_0: { bloque: 32, bytes: 18, bloquesPaso: 16, carrilesBloque: 2, repack: false, decl: 'array<u32>' },
  Q4_0R: { bloque: 32, bytes: 18, bloquesPaso: 16, carrilesBloque: 2, repack: true, decl: 'array<vec2<u32>>' },
  Q6_K: { bloque: 256, bytes: 210, bloquesPaso: 2, carrilesBloque: 16, repack: false, decl: 'array<u32>' },
};

export const GEMV2_TIPOS = new Set(Object.keys(FORMATOS));

export function gemv2Formato(tipo) {
  const f = FORMATOS[tipo];
  if (!f) throw new Error(`gemv2: tipo sin GEMV nuevo: «${tipo}»`);
  return f;
}

// Palabra u32 donde empieza la región de escalas de un tensor reordenado de
// `nBloques` bloques (`8·N` en Q8_0R, `4·N` en Q4_0R); 0 en los demás. N es
// el número de bloques del TENSOR, no el de las filas que pida una llamada.
export function gemv2Escalas(tipo, nBloques) {
  if (tipo === 'Q8_0R') return 8 * nBloques;
  if (tipo === 'Q4_0R') return 4 * nBloques;
  return 0;
}

// Bloques del tensor ENTERO de un búfer reordenado, deducidos de su tamaño. Hace
// falta porque las escalas empiezan detrás de los cuantos de TODOS los bloques
// del tensor, y una llamada puede pedir menos filas de las que tiene (la cabeza
// de salida pide las del vocabulario): sacar N de las filas de la llamada
// pondría las escalas en otro sitio, sin error. Con L palabras en el búfer,
//   Q8_0R: N = ⌊2L/17⌋        Q4_0R: N = ⌊2L/9⌋
// y es exacto mientras el búfer lleve como mucho tres palabras de holgura
// sobre su tamaño redondeado a 4, que es como los crea el cargador (redondeados
// a 16). Es la misma deducción que hacen los demás lectores de estos formatos.
// Ojo: el tamaño SOLO no lo decide todo. Un búfer más grande que el del
// cargador se lee como un tensor de más bloques, y a veces mide justo lo que el
// de ese otro tensor (34 bloques Q4_0R en 768 bytes se leen como 42, y 768 es
// el búfer de 42). Por eso quien despacha usa la marca del cargador cuando la
// tiene, y si no, exige el tamaño exacto (ver ops.gemv.js).
export function gemv2BloquesDeBuffer(tipo, bytes) {
  const L = Math.floor(bytes / 4);
  if (tipo === 'Q8_0R') return Math.floor((2 * L) / 17);
  if (tipo === 'Q4_0R') return Math.floor((2 * L) / 9);
  throw new Error(`gemv2: «${tipo}» no es un formato reordenado`);
}

// Bytes EXACTOS del búfer que el cargador crea para un tensor reordenado de
// `nBloques` bloques: el formato redondeado a 16 (lo que pide la vista vec4).
export function gemv2BytesBufferR(tipo, nBloques) {
  const F = gemv2Formato(tipo);
  if (!F.repack) throw new Error(`gemv2: «${tipo}» no es un formato reordenado`);
  return Math.ceil((F.bytes * nBloques) / 16) * 16;
}

// Bytes que el búfer de un tensor de `nBloques` bloques tiene que poder servir
// en elementos ENTEROS del array con que se declara. Un array<vec4<u32>> no
// llega a los últimos bytes de un búfer que no sea múltiplo de 16, y la última
// escala vive justo ahí: leerla fuera de rango no falla, devuelve otra cosa.
// Quien llama compara con `buf.size`.
export function gemv2BytesNecesarios(tipo, nBloques) {
  const F = gemv2Formato(tipo);
  const n = nBloques;
  if (tipo === 'Q8_0R') {
    const ultima = gemv2Escalas(tipo, n) + ((n - 1) >> 1);
    return ((ultima >> 2) + 1) * 16;
  }
  if (tipo === 'Q4_0R') {
    const ultima = gemv2Escalas(tipo, n) + ((n - 1) >> 1);
    return ((ultima >> 1) + 1) * 8;
  }
  // GGUF: el último tramo de cuantos lee hasta cuatro bytes más allá del
  // bloque (la palabra siguiente al recomponer); el cargador deja esa holgura.
  return n * F.bytes + 4;
}

// ── La forma: filas por grupo (R) y subgrupos que se reparten K (NSG) ───────
// Ninguna de las dos toca la numérica (ver la cabecera), así que se pueden
// barrer libremente. Los valores de partida siguen lo que mejor rinde en otros
// motores con la misma estructura: Q8_0 con 2 filas y K repartida en 4
// subgrupos; Q4_0 con 4 filas por subgrupo; Q6_K con 2. Con pocas filas, K en 4
// subgrupos para no dejar el dispositivo medio vacío. `__ELFFUSS_GEMV2_FORMA =
// { R, NSG }` la fija A MANO para el barrido, y una forma que no vale LANZA en
// vez de cambiarse por otra: un número con una etiqueta que no es la suya es
// peor que no tenerlo.
export function gemv2Forma({ tipo, filas, k, T = 1, nm = 1, maxCompartida = 16384, sg = true }) {
  const F = gemv2Formato(tipo);
  const pasos = Math.ceil(k / F.bloque / F.bloquesPaso);
  const lista = Array.isArray(filas) ? filas : [filas];
  const divide = (R) => lista.every((f) => f % R === 0);
  // Sin subgrupos, la suma final de los 32 carriles va por memoria de grupo (`red`): cuenta también.
  const compartida = (R, NSG) => (NSG > 1 ? nm * R * T * GEMV2_PARTES * 32 * 4 : 0) + (sg ? 0 : nm * R * T * 32 * 4);
  const pedida = globalThis.__ELFFUSS_GEMV2_FORMA;
  if (pedida && typeof pedida === 'object') {
    const { R, NSG } = pedida;
    if (![1, 2, 4, 8].includes(R) || ![1, 2, 4].includes(NSG)) {
      throw new Error(`gemv2: forma pedida inválida R=${R} NSG=${NSG} (R ∈ 1,2,4,8; NSG ∈ 1,2,4)`);
    }
    if (!divide(R)) throw new Error(`gemv2: R=${R} no divide las filas ${lista.join(',')}`);
    if (compartida(R, NSG) > maxCompartida) {
      throw new Error(`gemv2: R=${R} NSG=${NSG} T=${T} pide ${compartida(R, NSG)} bytes de memoria de grupo`);
    }
    return { R, NSG, pasos };
  }
  const q8 = tipo === 'Q8_0' || tipo === 'Q8_0R';
  let R = q8 ? 2 : tipo === 'Q6_K' ? 2 : 4;
  let NSG = q8 ? 4 : 1;
  if (Math.min(...lista) <= 512) { NSG = 4; R = Math.min(R, 2); }
  // Con varios tokens las activaciones de cada token viven en registros (y en
  // Q4_0/Q6_K también su copia preescalada): mejor una partición por carril.
  if (T > 1) NSG = 4;
  // Más subgrupos que pasos dejarían subgrupos sin nada que hacer.
  while (NSG > 1 && NSG > pasos) NSG >>= 1;
  // Registros: acumuladores vivos = matrices · filas · tokens · particiones del
  // subgrupo. Con muchos tokens se bajan filas y, si no basta y hay pasos para
  // todos, se reparte K entre más subgrupos.
  const vivos = () => nm * R * T * (GEMV2_PARTES / NSG);
  while (vivos() > 24 && R > 1) R >>= 1;
  while (vivos() > 24 && NSG < GEMV2_PARTES && NSG * 2 <= pasos) NSG <<= 1;
  while (R > 1 && !divide(R)) R >>= 1;
  while (R > 1 && compartida(R, NSG) > maxCompartida) R >>= 1;
  return { R, NSG, pasos };
}

// La clave del pipeline lleva TODO lo que cambia el WGSL generado, y NADA más.
// Sin lo primero, dos ramas de un A/B compartirían el shader ya compilado y
// medirían lo mismo; sin lo segundo, el mismo shader se compilaría dos veces
// (unpack4xI8 solo existe en la familia Q8).
export function gemv2Clave(o) {
  const u = o.u4x8 && familia(o.tipo) === 'Q8' ? 1 : 0;
  return `gemv2_${o.tipo}_${o.modo}${o.n}_R${o.R}_S${o.NSG}_T${o.T}_c${o.cola ? 1 : 0}_u${u}${o.sg === false ? '_g0' : ''}`;
}

// El uniforme, en el orden exacto del `struct Params` de abajo: cuatro u32 y
// tres vec4<u32> (alineados a 16, así que caen en los desplazamientos 16, 32 y
// 48 sin relleno). `fin` son los grupos acumulados al final de cada sección.
export function gemv2Params({ k, bpr, pasos, filas, esc, fin }) {
  const u = new Uint32Array(16);
  u[0] = k / 4; u[1] = bpr; u[2] = pasos; u[3] = filas.length;
  for (let i = 0; i < 4; i++) {
    u[4 + i] = filas[i] || 0;
    u[8 + i] = esc[i] || 0;
    u[12 + i] = fin[i] || 0;
  }
  return u;
}

const L = (n, f, sep = '\n') => Array.from({ length: n }, (_, i) => f(i)).join(sep);
const COMP = ['x', 'y', 'z', 'w'];

// ── Lo que cada formato hace en un paso ─────────────────────────────────────
// `indices`: bloque de la fila y subíndices del carril (antes de la guarda de
// cola). `activaciones(t)`: lo que se lee de `x` para el token t, compartido por
// todas las filas. `fila(tipo, w, r, e)`: descodifica la fila r de la matriz
// `w` (pesos en registros, compartidos por los T tokens; `e` es la variable con
// la palabra donde empiezan sus escalas, solo en los reordenados) y
// `contrib(t)` devuelve la contribución del bloque al acumulador del token t.
const PASO = {
  // Q8_0 y Q8_0R: 2 carriles por bloque, 16 pesos seguidos cada uno.
  Q8: {
    indices: `
          let b = s * 16u + (lane >> 1u);
          let il = lane & 1u;`,
    activaciones: (t) => `
          let xq${t} = ${t ? `${t}u * params.k4 + ` : ''}b * 8u + il * 4u;
${L(4, (c) => `          let x${t}_${c} = x[xq${t} + ${c}u];`)}`,
    fila: (tipo, w, r, e) => (tipo === 'Q8_0R' ? `
              let i = (row0 + ${r}u) * params.bpr + b;
              let qv = ${w}[i * 2u + il];
              let sw = ${e} + (i >> 1u);
              let d = unpack2x16float(${w}[sw >> 2u][sw & 3u] >> ((i & 1u) * 16u)).x;
              let f0 = i8x4(qv.x);
              let f1 = i8x4(qv.y);
              let f2 = i8x4(qv.z);
              let f3 = i8x4(qv.w);` : `
              let blk = ((row0 + ${r}u) * params.bpr + b) * 34u;
              let d = unpack2x16float(${w}[blk >> 2u] >> ((blk & 2u) * 8u)).x;
              let wb = (blk + 2u + il * 16u) >> 2u;
              let mitad = (blk & 2u) == 0u;
              let v0 = ${w}[wb];
              let v1 = ${w}[wb + 1u];
              let v2 = ${w}[wb + 2u];
              let v3 = ${w}[wb + 3u];
              let v4 = ${w}[wb + 4u];
              let f0 = i8x4(junta(v0, v1, mitad));
              let f1 = i8x4(junta(v1, v2, mitad));
              let f2 = i8x4(junta(v2, v3, mitad));
              let f3 = i8x4(junta(v3, v4, mitad));`),
    contrib: (t) => `d * (((dot(f0, x${t}_0) + dot(f1, x${t}_1)) + dot(f2, x${t}_2)) + dot(f3, x${t}_3))`,
  },

  // Q4_0 y Q4_0R: 2 carriles por bloque; el carril il se queda los bytes
  // 8il..8il+7: sus nibbles bajos son los pesos 8il..8il+7 y los altos los
  // 16+8il..16+8il+7. Dos tramos de 8 activaciones, cuatro vec4.
  Q4: {
    indices: `
          let b = s * 16u + (lane >> 1u);
          let il = lane & 1u;`,
    activaciones: (t) => `
          let xq${t} = ${t ? `${t}u * params.k4 + ` : ''}b * 8u + il * 2u;
          let xl${t}_0 = x[xq${t}];
          let xl${t}_1 = x[xq${t} + 1u];
          let xh${t}_0 = x[xq${t} + 4u];
          let xh${t}_1 = x[xq${t} + 5u];
          let sx${t} = (dot(xl${t}_0, UNO) + dot(xl${t}_1, UNO)) + (dot(xh${t}_0, UNO) + dot(xh${t}_1, UNO));
          let pl${t}_0 = xl${t}_0 * ESC_BAJO;
          let pl${t}_1 = xl${t}_1 * ESC_BAJO;
          let ph${t}_0 = xh${t}_0 * ESC_ALTO;
          let ph${t}_1 = xh${t}_1 * ESC_ALTO;`,
    fila: (tipo, w, r, e) => `${tipo === 'Q4_0R' ? `
              let i = (row0 + ${r}u) * params.bpr + b;
              let qv = ${w}[i * 2u + il];
              let q0 = qv.x;
              let q1 = qv.y;
              let sw = ${e} + (i >> 1u);
              let d = unpack2x16float(${w}[sw >> 1u][sw & 1u] >> ((i & 1u) * 16u)).x;` : `
              let blk = ((row0 + ${r}u) * params.bpr + b) * 18u;
              let d = unpack2x16float(${w}[blk >> 2u] >> ((blk & 2u) * 8u)).x;
              let wb = (blk + 2u + il * 8u) >> 2u;
              let mitad = (blk & 2u) == 0u;
              let v0 = ${w}[wb];
              let v1 = ${w}[wb + 1u];
              let v2 = ${w}[wb + 2u];
              let q0 = junta(v0, v1, mitad);
              let q1 = junta(v1, v2, mitad);`}
              let l0 = vec4<f32>(vec4<u32>(q0) & NIB_BAJO);
              let h0 = vec4<f32>(vec4<u32>(q0) & NIB_ALTO);
              let l1 = vec4<f32>(vec4<u32>(q1) & NIB_BAJO);
              let h1 = vec4<f32>(vec4<u32>(q1) & NIB_ALTO);`,
    contrib: (t) => `d * ((((dot(l0, pl${t}_0) + dot(l1, pl${t}_1)) + dot(h0, ph${t}_0)) + dot(h1, ph${t}_1)) - 8.0 * sx${t})`,
  },

  // Q6_K: 16 carriles por bloque, dos bloques por paso. El carril (ip, il) de
  // su bloque lleva las posiciones l = 4il..4il+3 de la mitad ip en los cuatro
  // grupos de 32 (0, 32, 64, 96): los cuatro pesos que comparten un byte de qh.
  Q6K: {
    indices: `
          let b = s * 2u + (lane >> 4u);
          let ip = (lane >> 3u) & 1u;
          let il = lane & 7u;`,
    activaciones: (t) => `
          let xq${t} = ${t ? `${t}u * params.k4 + ` : ''}b * 64u + ip * 32u + il;
${L(4, (g) => `          let x${t}_${g} = x[xq${t} + ${8 * g}u];
          let p${t}_${g} = x${t}_${g} * ESC_BAJO;
          let sx${t}_${g} = dot(x${t}_${g}, UNO);`)}`,
    fila: (tipo, w, r) => `
              let blk = ((row0 + ${r}u) * params.bpr + b) * 210u;
              let wb = blk >> 2u;
              let mitad = (blk & 2u) != 0u;
              let o0 = wb + ip * 16u + il;
              let oh = wb + 32u + ip * 8u + il;
              let os = wb + 48u + ip * 2u;
              let ql0 = junta(${w}[o0], ${w}[o0 + 1u], mitad);
              let ql1 = junta(${w}[o0 + 8u], ${w}[o0 + 9u], mitad);
              let qh = junta(${w}[oh], ${w}[oh + 1u], mitad);
              let sc0 = junta(${w}[os], ${w}[os + 1u], mitad);
              let sc1 = junta(${w}[os + 1u], ${w}[os + 2u], mitad);
              let d = unpack2x16float(${w}[wb + 52u] >> select(0u, 16u, mitad)).x;
              let sh = (il >> 2u) * 8u;
              let e0 = bitcast<i32>(sc0 >> sh);
              let e1 = bitcast<i32>(sc1 >> sh);
              let c0 = f32(extractBits(e0, 0u, 8u));
              let c1 = f32(extractBits(e0, 16u, 8u));
              let c2 = f32(extractBits(e1, 0u, 8u));
              let c3 = f32(extractBits(e1, 16u, 8u));
              let f0 = vec4<f32>(vec4<u32>((ql0 & 0x0F0F0F0Fu) | ((qh & 0x03030303u) << 4u)) & SEIS);
              let f1 = vec4<f32>(vec4<u32>((ql1 & 0x0F0F0F0Fu) | ((qh & 0x0C0C0C0Cu) << 2u)) & SEIS);
              let f2 = vec4<f32>(vec4<u32>(((ql0 >> 4u) & 0x0F0F0F0Fu) | (qh & 0x30303030u)) & SEIS);
              let f3 = vec4<f32>(vec4<u32>(((ql1 >> 4u) & 0x0F0F0F0Fu) | ((qh >> 2u) & 0x30303030u)) & SEIS);`,
    contrib: (t) => `d * (((c0 * (dot(f0, p${t}_0) - 32.0 * sx${t}_0) + c1 * (dot(f1, p${t}_1) - 32.0 * sx${t}_1)) + ` +
      `c2 * (dot(f2, p${t}_2) - 32.0 * sx${t}_2)) + c3 * (dot(f3, p${t}_3) - 32.0 * sx${t}_3))`,
  },
};

const familia = (tipo) => (tipo.startsWith('Q8') ? 'Q8' : tipo.startsWith('Q4') ? 'Q4' : 'Q6K');

// ── El shader ───────────────────────────────────────────────────────────────
// o = { tipo, R, NSG, T, modo, n, u4x8, cola, sg }
//   · modo 'una': una matriz → y0.
//   · modo 'swiglu': n = 2, gate (w0) y up (w1) con las MISMAS filas; cada
//     grupo hace sus R filas de las dos y escribe silu(gate)·up.
//   · modo 'secciones': n matrices (2-4) con filas distintas y la misma K; los
//     grupos se reparten por secciones y cada sección escribe su salida.
//   · u4x8: usar `unpack4xI8` (extensión de lenguaje `packed_4x8_integer_dot_product`);
//     sin ella, los mismos desplazamientos a mano (mismos enteros, misma suma).
//   · cola: K no es múltiplo del paso y hay que guardar el último bloque.
//   · sg: false para dispositivos sin subgrupos (el teléfono): la suma de los 32
//     carriles se hace en memoria de grupo, en árbol, en vez de con subgroupAdd.
//     Mismos productos y mismas sumas parciales; cambia el orden de la última suma.
// Enlaces: pesos 0..n−1, `x` en n, salidas a continuación, el uniforme al final.
export function gemv2Shader(o) {
  const { tipo, R, NSG, T, modo, n, u4x8, cola } = o;
  const sg = o.sg !== false;
  const F = gemv2Formato(tipo);
  const fam = familia(tipo);
  const P = GEMV2_PARTES;
  if (![1, 2, 4].includes(NSG) || P % NSG) throw new Error(`gemv2: NSG=${NSG} no divide ${P}`);
  if (!(R >= 1 && R <= 8)) throw new Error(`gemv2: R=${R} fuera de rango`);
  if (!(T >= 1 && T <= GEMV2_MAX_TOKENS)) throw new Error(`gemv2: T=${T} fuera de rango`);
  if (modo === 'una' && n !== 1) throw new Error('gemv2: «una» lleva una matriz');
  if (modo === 'swiglu' && n !== 2) throw new Error('gemv2: «swiglu» lleva gate y up');
  if (modo === 'secciones' && !(n >= 2 && n <= 4)) throw new Error('gemv2: «secciones» lleva de 2 a 4 matrices');
  if (!['una', 'swiglu', 'secciones'].includes(modo)) throw new Error(`gemv2: modo «${modo}»`);
  const J = P / NSG;                               // particiones por subgrupo
  const nm = modo === 'swiglu' ? 2 : 1;            // matrices que recorre un grupo
  const nsal = modo === 'secciones' ? n : 1;       // salidas
  const A = (m, r, t, j) => `a${m}_${r}_${t}_${j}`;
  const S = PASO[fam];

  const enlaces = [
    ...L(n, (m) => `@group(0) @binding(${m}) var<storage, read> w${m} : ${F.decl};`).split('\n'),
    `@group(0) @binding(${n}) var<storage, read> x : array<vec4<f32>>;`,
    ...L(nsal, (m) => `@group(0) @binding(${n + 1 + m}) var<storage, read_write> y${m} : array<f32>;`).split('\n'),
    `@group(0) @binding(${n + 1 + nsal}) var<uniform> params : Params;`,
  ].join('\n');

  const ayudas = [];
  if (fam === 'Q8') {
    ayudas.push(u4x8
      ? `// Cuatro int8 con signo de una palabra, el byte 0 en .x.
fn i8x4(q : u32) -> vec4<f32> { return vec4<f32>(unpack4xI8(q)); }`
      : `// Cuatro int8 con signo de una palabra, el byte 0 en .x: cada byte se sube al
// tope y se baja con desplazamiento ARITMÉTICO. Mismos enteros que unpack4xI8.
fn i8x4(q : u32) -> vec4<f32> {
  return vec4<f32>(bitcast<vec4<i32>>(vec4<u32>(q) << vec4<u32>(24u, 16u, 8u, 0u)) >> vec4<u32>(24u));
}`);
  }
  if (!F.repack) {
    ayudas.push(`// Palabra de 4 bytes que empieza a media palabra (\`mitad\`) o en palabra.
// Los dos desplazamientos son de 16 constantes: ningún desplazamiento de 32.
fn junta(a : u32, b : u32, mitad : bool) -> u32 {
  return select(a, (a >> 16u) | (b << 16u), mitad);
}`);
  }
  if (fam !== 'Q8') {
    ayudas.push(`const UNO = vec4<f32>(1.0);
// La potencia de dos que le falta a cada byte de la palabra (0, 8, 16, 24 bits).
const ESC_BAJO = vec4<f32>(1.0, 0x1p-8f, 0x1p-16f, 0x1p-24f);`);
  }
  if (fam === 'Q4') {
    ayudas.push(`// Y a cada nibble alto (4, 12, 20, 28 bits).
const ESC_ALTO = vec4<f32>(0x1p-4f, 0x1p-12f, 0x1p-20f, 0x1p-28f);
const NIB_BAJO = vec4<u32>(0xFu, 0xF00u, 0xF0000u, 0xF000000u);
const NIB_ALTO = vec4<u32>(0xF0u, 0xF000u, 0xF00000u, 0xF0000000u);`);
  }
  if (fam === 'Q6K') ayudas.push('const SEIS = vec4<u32>(0x3Fu, 0x3F00u, 0x3F0000u, 0x3F000000u);');
  if (modo === 'swiglu') {
    ayudas.push(`// La misma silu que el kernel de SwiGLU de siempre: x/(1+exp(-x)).
fn silu(v : f32) -> f32 { return v / (1.0 + exp(-v)); }`);
  }

  // El bucle de pasos para las matrices `mats` (pares binding/índice de
  // acumulador). Una vuelta son GEMV2_PARTES pasos; el subgrupo g hace los de
  // sus particiones, g + NSG·j.
  const bucle = (mats, sangria) => {
    const cuerpo = L(J, (j) => {
      const s = NSG > 1 ? `s0 + g + ${NSG * j}u` : `s0 + ${j}u`;
      const filas = L(R, (r) => mats.map(({ w, m }) => `            {${S.fila(tipo, w, r, `esc${m}`)}
${L(T, (t) => `              ${A(m, r, t, j)} = ${A(m, r, t, j)} + ${S.contrib(t)};`)}
            }`).join('\n'));
      return `      {
        let s = ${s};
        if (s < params.pasos) {${S.indices}
${cola ? '          if (b < params.bpr) {' : '          {'}${L(T, (t) => S.activaciones(t), '')}
${filas}
          }
        }
      }`;
    });
    const txt = `    for (var s0 = 0u; s0 < params.pasos; s0 = s0 + ${P}u) {
${cuerpo}
    }`;
    return sangria ? txt.split('\n').map((l) => (l ? sangria + l : l)).join('\n') : txt;
  };

  let principal;
  if (modo === 'secciones') {
    principal = L(n, (m) => `  ${m ? '} else ' : ''}if (sec == ${m}u) {
${bucle([{ w: `w${m}`, m: 0 }], '  ')}`) + '\n  }';
  } else {
    principal = bucle(modo === 'swiglu' ? [{ w: 'w0', m: 0 }, { w: 'w1', m: 1 }] : [{ w: 'w0', m: 0 }], '');
  }

  // Cada carril junta sus particiones EN ORDEN y luego suma el subgrupo.
  const items = [];
  for (let m = 0; m < nm; m++) for (let r = 0; r < R; r++) for (let t = 0; t < T; t++) items.push({ m, r, t });
  const base = ({ m, r, t }) => ((m * R + r) * T + t) * P;
  const suma4 = (v) => `((${v(0)} + ${v(1)}) + ${v(2)}) + ${v(3)}`;
  const deParc = (it) => { const b0 = base(it) * 32; return (p) => `parc[${b0 + 32 * p}u + lane]`; };
  let reduccion;
  if (sg) {
    reduccion = NSG === 1
      ? items.map((it) => `  let tot${it.m}_${it.r}_${it.t} = subgroupAdd(${suma4((p) => A(it.m, it.r, it.t, p))});`).join('\n')
      : `${items.map((it) => L(J, (j) => `  parc[(${base(it)}u + g + ${NSG * j}u) * 32u + lane] = ${A(it.m, it.r, it.t, j)};`)).join('\n')}
  workgroupBarrier();
${items.map((it) => `  let tot${it.m}_${it.r}_${it.t} = subgroupAdd(${suma4(deParc(it))});`).join('\n')}`;
  } else {
    // Sin subgrupos: cada uno de los 32 carriles deja su suma en \`red\` y se
    // reducen a la mitad cinco veces, con una barrera entre pasos. Las barreras
    // quedan fuera de los \`if\` (control uniforme); solo escriben los 32 primeros.
    const ponRed = (v) => items.map((it, k) => `    red[${k * 32}u + lane] = ${v(it)};`).join('\n');
    const arbol = [16, 8, 4, 2, 1].map((st) => `  if (lid.x < ${st}u) {
${items.map((it, k) => `    red[${k * 32}u + lid.x] = red[${k * 32}u + lid.x] + red[${k * 32 + st}u + lid.x];`).join('\n')}
  }
  workgroupBarrier();`).join('\n');
    reduccion = NSG === 1
      ? `  {
${ponRed((it) => suma4((p) => A(it.m, it.r, it.t, p)))}
  }
  workgroupBarrier();
${arbol}
${items.map((it, k) => `  let tot${it.m}_${it.r}_${it.t} = red[${k * 32}u];`).join('\n')}`
      : `${items.map((it) => L(J, (j) => `  parc[(${base(it)}u + g + ${NSG * j}u) * 32u + lane] = ${A(it.m, it.r, it.t, j)};`)).join('\n')}
  workgroupBarrier();
  if (lid.x < 32u) {
${ponRed((it) => suma4(deParc(it)))}
  }
  workgroupBarrier();
${arbol}
${items.map((it, k) => `  let tot${it.m}_${it.r}_${it.t} = red[${k * 32}u];`).join('\n')}`;
  }

  const destino = (r, t) => `${t ? `${t}u * filas + ` : ''}row0 + ${r}u`;
  let escritura;
  if (modo === 'swiglu') {
    escritura = L(R, (r) => L(T, (t) => `    y0[${destino(r, t)}] = silu(tot0_${r}_${t}) * tot1_${r}_${t};`));
  } else if (modo === 'una') {
    escritura = L(R, (r) => L(T, (t) => `    y0[${destino(r, t)}] = tot0_${r}_${t};`));
  } else {
    escritura = L(n, (m) => `    ${m ? '} else ' : ''}if (sec == ${m}u) {
${L(R, (r) => L(T, (t) => `      y${m}[${destino(r, t)}] = tot0_${r}_${t};`))}`) + '\n    }';
  }

  const prologo = modo === 'secciones'
    ? `  // Sección de este grupo: los grupos van por secciones seguidas, y \`fin\`
  // dice dónde acaba cada una. Todo sale de workgroup_id: uniforme.
  var sec = 0u;
  var g0 = 0u;
${L(n - 1, (m) => `  if (wg >= params.fin.${COMP[m]}) { sec = ${m + 1}u; g0 = params.fin.${COMP[m]}; }`)}
  if (wg >= params.fin.${COMP[n - 1]}) { return; }
  let filas = params.filas[sec];
${F.repack ? '  let esc0 = params.esc[sec];\n' : ''}  let row0 = (wg - g0) * ${R}u;`
    : `  let filas = params.filas.x;
${F.repack ? `  let esc0 = params.esc.x;\n${nm > 1 ? '  let esc1 = params.esc.y;\n' : ''}` : ''}  let row0 = wg * ${R}u;
  if (row0 >= filas) { return; }`;

  return /* wgsl */`${sg ? 'enable subgroups;\n' : ''}
// gemv2 ${tipo} · ${modo} (${n}) · R=${R} · NSG=${NSG} · T=${T}${cola ? ' · cola' : ''}
struct Params {
  k4 : u32, bpr : u32, pasos : u32, nsec : u32,
  filas : vec4<u32>, esc : vec4<u32>, fin : vec4<u32>,
};
${enlaces}
${NSG > 1 ? `var<workgroup> parc : array<f32, ${nm * R * T * P * 32}>;\n` : ''}${sg ? '' : `var<workgroup> red : array<f32, ${items.length * 32}>;\n`}
${ayudas.join('\n')}

@compute @workgroup_size(${32 * NSG})
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let lane = lid.x & 31u;
${NSG > 1 ? '  let g = lid.x >> 5u;\n' : ''}  let wg = wid.x + wid.y * ${AXIS}u;
${prologo}

${items.map(({ m, r, t }) => L(J, (j) => `  var ${A(m, r, t, j)} = 0.0;`)).join('\n')}

${principal}

${reduccion}
  if (lid.x == 0u) {
${escritura}
  }
}`;
}
