// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · los candidatos del muestreo, calculados en el dispositivo.
// ─────────────────────────────────────────────────────────────────────────────
// Para muestrear hacía falta traer el vocabulario entero a la CPU en cada
// token. `topk.js` explica cuántos logits necesita de verdad `sample()` (los
// M = K'+|S| primeros en crudo, más los de la ventana penalizada); estos dos
// kernels los encuentran sin sacar el vector del dispositivo:
//
//   1. TROZOS: cada grupo toma un trozo contiguo del vocabulario y saca SUS M
//      primeros por SELECCIÓN RADIX, con un histograma de 256 casillas en
//      memoria compartida: cuatro rondas de 8 bits sobre una clave entera
//      ordenable y, entre los empatados con el M-ésimo, tres rondas más sobre
//      el índice para quedarse con los de índice MENOR.
//   2. FINAL: un solo grupo repite la selección sobre la unión de lo que dejaron
//      los trozos y copia los logits de los ids penalizados.
//
// «Primeros» es el orden de `sample()`: logit mayor primero y, a igualdad, el
// índice menor. Por qué el resultado es EXACTAMENTE el top-M de ese orden:
//   · la clave es entera: su orden es el de los f32 (con −0 igual a +0, como en
//     JavaScript) sin pasar por comparaciones en coma flotante, que en un
//     shader pueden aplanar subnormales o tratar NaN a su manera;
//   · (clave, índice) es un orden total estricto: la selección no tiene
//     empates que deshacer al azar, y cada trozo devuelve exactamente sus M
//     primeros (o todos, si tiene menos);
//   · cada elemento del top-M global tiene menos de M por delante en todo el
//     vocabulario, así que también en su trozo: ningún trozo lo descarta. Y los
//     M primeros de un superconjunto del top-M son el top-M;
//   · los bits salen tal cual del buffer de logits: la CPU ve los mismos f32.
// Desempatar por índice aquí no es un lujo: con «todos los empatados» un trozo
// lleno de fichas con el mismo logit (las reservadas sin entrenar suelen serlo)
// no cabría en ningún sitio y obligaría a leer el vector entero en cada token.
// Un NaN en cualquier logit se AVISA en vez de adivinarse (el orden con NaN no
// se puede reproducir con un trozo) y la CPU lee el vector entero.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.

// Contratos con `topk.js` y `ops.topk.js`. Como cada kernel devuelve
// exactamente min(M, válidos), basta con que los huecos lleguen a M_MAX.
export const TOPK_WG = 256;          // un hilo por casilla del histograma
export const TOPK_POR_HILO = 16;     // claves por hilo: trozos de 4096
export const TOPK_M_MAX = 192;       // el mayor M (= K'+|S|) que se acepta
export const TOPK_OUTCAP = 256;      // candidatos por trozo (≥ M_MAX)
export const TOPK_CAP = 256;         // candidatos finales (≥ M_MAX)
export const TOPK_P_MAX = 128;       // ids penalizados
// El índice va invertido en 24 bits (tres rondas): el vocabulario no puede
// pasar de 2^24 fichas.
export const TOPK_N_MAX = 1 << 24;

// Cabecera de la salida (u32): [0] candidatos, [1] avisos, [2] ids penalizados,
// [3] trozos, [4] número de serie de la llamada, [5..7] sin uso. Después
// TOPK_P_MAX bits de los penalizados y TOPK_CAP pares (id, bits). Avisos: 2 =
// hay NaN; 1 y 4 = no cupo (no debería pasar nunca).
// La SERIE existe porque, sin ámbitos de error, un envío que no se ejecuta (un
// bind group inválido, por ejemplo) no lanza: el buffer de lectura conserva lo
// de una llamada anterior, con su cabecera bien formada, y saldría un token de
// otro paso. Con la serie eso se ve: la cabecera dice de qué llamada es.
export const TOPK_CABECERA = 8;
export const topkBytesSalida = (pMax = TOPK_P_MAX, cap = TOPK_CAP) => (TOPK_CABECERA + pMax + 2 * cap) * 4;
// Lo que dejan los trozos: [2·G] (cuántos, avisos) y G·OUTCAP pares.
export const topkBytesTrozos = (trozos, outCap = TOPK_OUTCAP) => (2 * trozos + 2 * trozos * outCap) * 4;
export const topkTrozos = (n, porHilo = TOPK_POR_HILO) => Math.ceil(n / (TOPK_WG * porHilo));

// Lo común a los dos kernels: la clave, el NaN y la elección del dígito.
const COMUN = /* wgsl */`
var<workgroup> hist : array<atomic<u32>, 256>;
var<workgroup> grupo16 : array<u32, 16>;
// [0] prefijo de la clave del M-ésimo, [1] cuántos faltan dentro del prefijo,
// [2] 1 si hay menos de M válidos (entonces van todos), [3] prefijo del índice
// invertido del M-ésimo entre los empatados.
var<workgroup> sel : array<u32, 4>;

// Clave entera con el orden de los f32: positivos con el bit de signo puesto,
// negativos invertidos. −0 se cuenta como +0.
fn clave(bits : u32) -> u32 {
  let b = select(bits, 0u, bits == 0x80000000u);
  return select(b | 0x80000000u, ~b, (b & 0x80000000u) != 0u);
}

// Con bits, no con x != x: un compilador con matemática rápida puede dar esa
// comparación por falsa.
fn esNaN(bits : u32) -> bool { return (bits & 0x7fffffffu) > 0x7f800000u; }

// Tras el histograma de una ronda: en qué casilla cae el que falta. Se suman
// grupos de 16 casillas y el primer hilo baja de mayor a menor hasta que la cuenta
// llega; luego, dentro del grupo, casilla a casilla. El dígito va al prefijo
// \`ranura\` (0 la clave, 3 el índice invertido).
fn elige(t : u32, sh : u32, ranura : u32) {
  if (t < 16u) {
    var s = 0u;
    for (var b = 0u; b < 16u; b++) { s += atomicLoad(&hist[t * 16u + b]); }
    grupo16[t] = s;
  }
  workgroupBarrier();
  if (t == 0u && sel[2] == 0u) {
    let falta = sel[1];
    var acum = 0u;
    var d = 256u;
    for (var gq = 16u; gq > 0u; gq--) {
      let gi = gq - 1u;
      let s = grupo16[gi];
      if (acum + s >= falta) {
        for (var b = 16u; b > 0u; b--) {
          let bi = gi * 16u + b - 1u;
          let c = atomicLoad(&hist[bi]);
          if (acum + c >= falta) { d = bi; break; }
          acum += c;
        }
        break;
      }
      acum += s;
    }
    if (d == 256u) { sel[2] = 1u; }
    else { sel[ranura] = sel[ranura] | (d << sh); sel[1] = falta - acum; }
  }
  workgroupBarrier();
}
`;

// Kernel 1: un grupo por trozo de TOPK_WG·porHilo logits contiguos.
export function topkTrozosShader({ porHilo = TOPK_POR_HILO, outCap = TOPK_OUTCAP } = {}) {
  if (porHilo < 1 || porHilo > 32) throw new Error(`topk: ${porHilo} claves por hilo no caben en la máscara`);
  return /* wgsl */`
const WG : u32 = ${TOPK_WG}u;
const E : u32 = ${porHilo}u;
const TROZO : u32 = ${TOPK_WG * porHilo}u;
const OUTCAP : u32 = ${outCap}u;
struct P { n : u32, base : u32, m : u32, trozos : u32 };
@group(0) @binding(0) var<storage, read>       logits : array<u32>;
@group(0) @binding(1) var<storage, read_write> cand   : array<u32>;
@group(0) @binding(2) var<uniform>             p      : P;
var<workgroup> hayNaN : atomic<u32>;
var<workgroup> nOut : atomic<u32>;
${COMUN}
@compute @workgroup_size(${TOPK_WG})
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) t : u32) {
  let g = wid.x;
  let ini = g * TROZO;
  if (t == 0u) {
    sel[0] = 0u; sel[1] = p.m; sel[2] = 0u; sel[3] = 0xFF000000u;
    atomicStore(&hayNaN, 0u); atomicStore(&nOut, 0u);
  }
  workgroupBarrier();
  // Las claves del hilo, en registros; las lecturas, contiguas entre hilos.
  var k : array<u32, E>;
  var validos = 0u;
  var nan = false;
  for (var e = 0u; e < E; e++) {
    let j = ini + e * WG + t;
    k[e] = 0u;
    if (j < p.n) {
      let bits = logits[p.base + j];
      k[e] = clave(bits);
      validos = validos | (1u << e);
      if (esNaN(bits)) { nan = true; }
    }
  }
  if (nan) { atomicOr(&hayNaN, 1u); }
  // La clave del M-ésimo, 8 bits por ronda.
  var mascara = 0u;
  for (var r = 0u; r < 4u; r++) {
    let sh = 24u - 8u * r;
    let prefijo = sel[0];
    atomicStore(&hist[t], 0u);
    workgroupBarrier();
    for (var e = 0u; e < E; e++) {
      if (((validos >> e) & 1u) != 0u && (k[e] & mascara) == prefijo) {
        atomicAdd(&hist[(k[e] >> sh) & 255u], 1u);
      }
    }
    workgroupBarrier();
    elige(t, sh, 0u);
    mascara = mascara | (255u << sh);
  }
  let tau = sel[0];
  // Entre los empatados con él, los de índice menor: el índice invertido
  // mayor. Sus 8 bits altos son 0xFF (el índice cabe en 24).
  var mascaraI = 0xFF000000u;
  for (var r = 0u; r < 3u; r++) {
    let sh = 16u - 8u * r;
    let prefijo = sel[3];
    atomicStore(&hist[t], 0u);
    workgroupBarrier();
    for (var e = 0u; e < E; e++) {
      let inv = ~(ini + e * WG + t);
      if (((validos >> e) & 1u) != 0u && k[e] == tau && (inv & mascaraI) == prefijo) {
        atomicAdd(&hist[(inv >> sh) & 255u], 1u);
      }
    }
    workgroupBarrier();
    elige(t, sh, 3u);
    mascaraI = mascaraI | (255u << sh);
  }
  let iota = sel[3];
  let todos = sel[2] != 0u;
  // Exactamente los M primeros del trozo (o todos sus válidos si no llega).
  let pares = 2u * p.trozos;
  for (var e = 0u; e < E; e++) {
    let j = ini + e * WG + t;
    if (((validos >> e) & 1u) != 0u && (todos || k[e] > tau || (k[e] == tau && ~j >= iota))) {
      let s = atomicAdd(&nOut, 1u);
      if (s < OUTCAP) {
        let o = pares + (g * OUTCAP + s) * 2u;
        cand[o] = j;
        cand[o + 1u] = logits[p.base + j];
      }
    }
  }
  workgroupBarrier();
  if (t == 0u) {
    let c = atomicLoad(&nOut);
    cand[g * 2u] = min(c, OUTCAP);
    cand[g * 2u + 1u] = select(0u, 1u, c > OUTCAP) | select(0u, 2u, atomicLoad(&hayNaN) != 0u);
  }
}`;
}

// Kernel 2: un solo grupo sobre lo que dejaron los trozos.
export function topkFinalShader({ outCap = TOPK_OUTCAP, cap = TOPK_CAP, pMax = TOPK_P_MAX } = {}) {
  if (pMax > TOPK_WG) throw new Error(`topk: ${pMax} ids penalizados son más que hilos`);
  return /* wgsl */`
const WG : u32 = ${TOPK_WG}u;
const OUTCAP : u32 = ${outCap}u;
const CAP : u32 = ${cap}u;
const PMAX : u32 = ${pMax}u;
const CABECERA : u32 = ${TOPK_CABECERA}u;
struct P { n : u32, base : u32, m : u32, trozos : u32, nS : u32, serie : u32, pad1 : u32, pad2 : u32 };
@group(0) @binding(0) var<storage, read>       logits : array<u32>;
@group(0) @binding(1) var<storage, read>       cand   : array<u32>;
@group(0) @binding(2) var<storage, read>       ids    : array<u32>;
@group(0) @binding(3) var<storage, read_write> out    : array<u32>;
@group(0) @binding(4) var<uniform>             p      : P;
var<workgroup> avisos : atomic<u32>;
var<workgroup> nOut : atomic<u32>;
${COMUN}
@compute @workgroup_size(${TOPK_WG})
fn main(@builtin(local_invocation_index) t : u32) {
  let G = p.trozos;
  let pares = 2u * G;
  if (t == 0u) {
    sel[0] = 0u; sel[1] = p.m; sel[2] = 0u; sel[3] = 0xFF000000u;
    atomicStore(&avisos, 0u); atomicStore(&nOut, 0u);
  }
  workgroupBarrier();
  // Los avisos de los trozos (NaN, no cupo) pasan tal cual.
  for (var g = t; g < G; g += WG) {
    let f = cand[g * 2u + 1u];
    if (f != 0u) { atomicOr(&avisos, f); }
  }
  // Los logits crudos de los ids penalizados: la penalización la pone la CPU.
  if (t < p.nS && t < PMAX) {
    let id = ids[t];
    var v = 0u;
    if (id < p.n) { v = logits[p.base + id]; }
    out[CABECERA + t] = v;
  }
  var mascara = 0u;
  for (var r = 0u; r < 4u; r++) {
    let sh = 24u - 8u * r;
    let prefijo = sel[0];
    atomicStore(&hist[t], 0u);
    workgroupBarrier();
    for (var g = 0u; g < G; g++) {
      let c = cand[g * 2u];
      for (var s = t; s < c; s += WG) {
        let kk = clave(cand[pares + (g * OUTCAP + s) * 2u + 1u]);
        if ((kk & mascara) == prefijo) { atomicAdd(&hist[(kk >> sh) & 255u], 1u); }
      }
    }
    workgroupBarrier();
    elige(t, sh, 0u);
    mascara = mascara | (255u << sh);
  }
  let tau = sel[0];
  var mascaraI = 0xFF000000u;
  for (var r = 0u; r < 3u; r++) {
    let sh = 16u - 8u * r;
    let prefijo = sel[3];
    atomicStore(&hist[t], 0u);
    workgroupBarrier();
    for (var g = 0u; g < G; g++) {
      let c = cand[g * 2u];
      for (var s = t; s < c; s += WG) {
        let o = pares + (g * OUTCAP + s) * 2u;
        let inv = ~cand[o];
        if (clave(cand[o + 1u]) == tau && (inv & mascaraI) == prefijo) {
          atomicAdd(&hist[(inv >> sh) & 255u], 1u);
        }
      }
    }
    workgroupBarrier();
    elige(t, sh, 3u);
    mascaraI = mascaraI | (255u << sh);
  }
  let iota = sel[3];
  let todos = sel[2] != 0u;
  for (var g = 0u; g < G; g++) {
    let c = cand[g * 2u];
    for (var s = t; s < c; s += WG) {
      let o = pares + (g * OUTCAP + s) * 2u;
      let id = cand[o];
      let bits = cand[o + 1u];
      let kk = clave(bits);
      if (todos || kk > tau || (kk == tau && ~id >= iota)) {
        let q = atomicAdd(&nOut, 1u);
        if (q < CAP) {
          let w = CABECERA + PMAX + q * 2u;
          out[w] = id;
          out[w + 1u] = bits;
        }
      }
    }
  }
  workgroupBarrier();
  if (t == 0u) {
    let c = atomicLoad(&nOut);
    out[0] = min(c, CAP);
    out[1] = atomicLoad(&avisos) | select(0u, 4u, c > CAP);
    out[2] = p.nS;
    out[3] = G;
    out[4] = p.serie;
  }
}`;
}
