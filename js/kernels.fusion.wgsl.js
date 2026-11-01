// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · kernels de fila FUSIONADOS: varias operaciones pequeñas en un despacho.
// ─────────────────────────────────────────────────────────────────────────────
// Un token de generación son cientos de despachos y la mayoría NO son
// multiplicaciones de matrices: normas, sumas, cortes, copias. Cada uno lee y
// escribe un vector diminuto y paga entero el coste fijo de un despacho
// (grabarlo, su bind group, su uniforme, la barrera con el anterior). Aquí se
// juntan las cadenas que siempre van seguidas:
//
//   F1  post-norma + suma residual + norma siguiente    (3 despachos → 1)
//   F2  embedding + norma de entrada de la primera capa  (2 → 1)
//   F4  la convolución corta de lfm3 tras `in_proj`       (8 → 1)
//   F6  QK-norm + RoPE + escritura de K y V en la caché  (6 → 1, o 6 → 2 partido
//       cuando el dispositivo no da buffers para un solo despacho)
//
// EL CONTRATO ES BIT A BIT. No «parecido»: el mismo resultado que la cadena de
// siempre hasta el último bit, y por eso se puede encender sin volver a validar
// la numérica del modelo. Eso obliga a tres cosas, y las tres son a propósito:
//   1. las MISMAS operaciones en el mismo orden: `(x·s)·w` y no `x·(s·w)`,
//      `res + y` con los mismos operandos, la suma de la conv de r=0 a r=K-1;
//   2. el MISMO árbol de reducción que `rmsnormShader`: grupos de 128 hilos,
//      cada hilo suma EN SERIE sus columnas (c, c+128, c+256, …) y después el
//      árbol por memoria compartida. Otro reparto daría la misma suma con otra
//      partición, y los últimos bits cambiarían;
//   3. los MISMOS redondeos intermedios. En la cadena, cada resultado pasa por
//      memoria como f32. Dentro de un kernel, un compilador con contracción
//      puede fundir `a + b·c` en una sola operación y saltarse el redondeo de
//      `b·c`, o reasociar un producto con el siguiente. `corta()` lo impide: un
//      XOR con un cero que el compilador no puede ver (viene del uniforme)
//      obliga a materializar el f32 justo donde la cadena lo guardaba. Es la
//      identidad en ejecución y cuesta una operación entera por elemento.
// Una fusión que deje de ser exacta no avisa en el modelo: sale el mismo texto
// casi siempre. La caza una prueba que compara bit a bit con la cadena de
// siempre y dice el primer índice distinto.
//
// WG va por SUSTITUCIÓN DE TEXTO, igual que en `kernels.nn.wgsl.js`: el análisis
// de uniformidad exige que el recorrido de un bucle con barreras sea constante.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.

import { HELPERS, AXIS, quantInfo } from './kernels.wgsl.js';
import { ROPE_NEOX } from './kernels.nn.wgsl.js';

// El ancho del grupo de las normas. TIENE que ser el de `createNNOps` (su
// `wgNorm`, 128 por defecto y nadie pasa otro): es lo que fija el árbol de la
// suma de cuadrados. Con otro ancho la norma seguiría siendo correcta, pero
// dejaría de ser bit a bit la de la cadena.
export const WG_FILA = 128;

// Barrera de contracción. Ver la cabecera, punto 3.
const CORTA = /* wgsl */`
// Identidad en ejecución (p.cero vale 0) y opaca para el compilador: obliga a
// redondear a f32 AQUÍ, como cuando la cadena guardaba el valor en memoria.
fn corta(v : f32) -> f32 { return bitcast<f32>(bitcast<u32>(v) ^ p.cero); }
`;

// La suma en árbol, copia literal de la de `kernels.nn.wgsl.js`. La barrera va
// FUERA del if (uniformidad) y deja `scratch[0]` listo para todos los hilos.
const ARBOL = /* wgsl */`
  scratch[lid.x] = acc;
  workgroupBarrier();
  for (var off = WG >> 1u; off > 0u; off = off >> 1u) {
    if (lid.x < off) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + off]; }
    workgroupBarrier();
  }`;

// Declara los bindings en el orden de `nombres`, que es el MISMO array con el que
// ops.fusion.js arma la lista de buffers. Un solo sitio para el orden: si el
// shader y la llamada lo llevaran cada uno por su cuenta y se desincronizaran,
// cada buffer caería en el binding del vecino sin un solo error.
function enlaces(nombres, decl) {
  return nombres.map((n, i) => `@group(0) @binding(${i}) ${decl[n]}`).join('\n');
}

// ── F1: post-norma + suma residual + norma siguiente ────────────────────────
// Sustituye, por fila de `ne0` (un token):
//   lo2 = rmsnorm(lo, wp)        (la post-norma de lfm3; sin ella, lo2 = lo)
//   xo  = res + lo2              (la suma residual, `nn.add(x, lo2)`)
//   nrm = rmsnorm(xo, ws)        (la norma de lo que viene: la del FFN, la de
//                                 entrada de la capa siguiente o la final)
// Un grupo por fila. `xo` se escribe entero porque es el residual que sigue
// vivo; `nrm` es la entrada de la siguiente multiplicación.
//
// Sin `post` (qwen35, llama) es la misma cadena sin la primera norma.
export function enlacesNormaResidual(post) {
  return post ? ['lo', 'res', 'wp', 'ws', 'xo', 'nrm', 'p'] : ['lo', 'res', 'ws', 'xo', 'nrm', 'p'];
}

export function normaResidualShader({ wg = WG_FILA, post = true } = {}) {
  const decl = {
    lo:  'var<storage, read>       lo  : array<f32>;   // salida del bloque o del FFN',
    res: 'var<storage, read>       res : array<f32>;   // residual que entra',
    wp:  'var<storage, read>       wp  : array<f32>;   // peso de la post-norma [ne0]',
    ws:  'var<storage, read>       ws  : array<f32>;   // peso de la norma siguiente [ne0]',
    xo:  'var<storage, read_write> xo  : array<f32>;   // residual que sale',
    nrm: 'var<storage, read_write> nrm : array<f32>;   // entrada normalizada de lo siguiente',
    p:   'var<uniform>             p   : P;',
  };
  return /* wgsl */`
const WG : u32 = ${wg}u;
struct P { ne0 : u32, rows : u32, eps : f32, cero : u32 };
${enlaces(enlacesNormaResidual(post), decl)}
var<workgroup> scratch : array<f32, ${wg}>;
${CORTA}
@compute @workgroup_size(${wg})
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let row = wid.x + wid.y * ${AXIS}u;   // dos ejes: uno solo topa en 65535
  if (row >= p.rows) { return; }        // uniforme en el grupo
  let base = row * p.ne0;
  var acc : f32 = 0.0;
${post ? /* wgsl */`
  // 1) La post-norma: suma de cuadrados de la salida del bloque, con el reparto
  //    y el árbol de rmsnormShader.
  for (var c = lid.x; c < p.ne0; c = c + WG) {
    let v = lo[base + c];
    acc = acc + v * v;
  }
${ARBOL}
  let s1 = inverseSqrt(scratch[0] / f32(p.ne0) + p.eps);
  workgroupBarrier();                   // WAR: el árbol de abajo reescribe scratch
  acc = 0.0;
` : ''}
  // 2) La suma residual, y de paso la suma de cuadrados del residual nuevo (el
  //    mismo reparto: cada hilo vuelve a recorrer SUS columnas, en el mismo orden).
  for (var c = lid.x; c < p.ne0; c = c + WG) {
    let v = res[base + c] + ${post ? 'corta(lo[base + c] * s1 * wp[c])' : 'lo[base + c]'};
    xo[base + c] = v;
    acc = acc + v * v;
  }
${ARBOL}
  let s2 = inverseSqrt(scratch[0] / f32(p.ne0) + p.eps);

  // 3) La norma siguiente. Cada hilo relee lo que ÉL MISMO escribió en el paso 2
  //    (las mismas columnas), así que no hace falta barrera de memoria.
  for (var c = lid.x; c < p.ne0; c = c + WG) {
    nrm[base + c] = xo[base + c] * s2 * ws[c];
  }
}`;
}

// ── F2: embedding + norma de entrada de la primera capa ─────────────────────
// Los tipos cuyo lector de bloques es el de `gatherShader`: bloque `i` en el
// byte `i·blockBytes`, sin rejilla aparte. Es una lista CERRADA a propósito, y
// no «todo lo que hay en QUANT_TYPES»: un tipo nuevo entra cuando se ha mirado
// que su lector vale aquí, no por estar en la tabla.
// Los reordenados al cargar (Q8_0R/Q4_0R, `__ELFFUSS_REPACK`) entran: su
// descodificador respeta el mismo contrato de `decode(emit)` con el `blk` de
// este andamio (de `blk` saca el índice de bloque y lee cuantos y escala de sus
// regiones) y es el MISMO texto que usa `gatherShader`, así que F2 da los mismos
// bits que su gather. Lo que no basta es la lista: sus lectores sacan del
// tamaño del buffer dónde empiezan las escalas, y `gatherNorma` hace antes la
// misma comprobación de la marca que `quant.gather` (`compruebaBufferR`).
export const TIPOS_GATHER_NORMA = new Set([
  'F32', 'F16', 'BF16', 'Q8_0', 'Q4_0', 'Q2_K', 'Q4_K', 'Q5_K', 'Q6_K', 'PQ2_0', 'PTQ1_0', 'Q8_0R', 'Q4_0R',
]);

// Un grupo por token. Fase 1: el gather de siempre (mismo descodificador de
// bloque, mismo `emit`), pero los bloques de la fila los reparten los 128 hilos
// del grupo. Fase 2, tras una barrera de memoria de almacenamiento: la rmsnorm
// de esa fila, idéntica a `rmsnormShader`. `xo` es el residual (hace falta
// entero para la primera suma) y `nrm` la entrada de la capa 0.
export function gatherNormaShader(type, wg = WG_FILA) {
  if (!TIPOS_GATHER_NORMA.has(type)) throw new Error(`gather+norma: «${type}» no tiene lector de bloque aquí`);
  const d = quantInfo(type);
  if (d.table) throw new Error(`gather+norma: «${type}» necesita rejilla y aquí no se enlaza`);
  const emit = (idx, val) => `            xo[bxo + (${idx})] = ${val};`;
  return /* wgsl */`
const WG : u32 = ${wg}u;
struct P { dim : u32, tokens : u32, blocksPerRow : u32, eps : f32 };
@group(0) @binding(0) var<storage, read>       w   : array<u32>;
@group(0) @binding(1) var<storage, read>       ids : array<u32>;
@group(0) @binding(2) var<storage, read>       wn  : array<f32>;   // peso de la norma [dim]
@group(0) @binding(3) var<storage, read_write> xo  : array<f32>;   // embedding [tokens][dim]
@group(0) @binding(4) var<storage, read_write> nrm : array<f32>;   // normalizado [tokens][dim]
@group(0) @binding(5) var<uniform>             p   : P;
${HELPERS}${d.helpers}
var<workgroup> scratch : array<f32, ${wg}>;

@compute @workgroup_size(${wg})
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let t = wid.x + wid.y * ${AXIS}u;     // un grupo por token
  if (t >= p.tokens) { return; }
  for (var b = lid.x; b < p.blocksPerRow; b = b + WG) {
    let blk = (ids[t] * p.blocksPerRow + b) * ${d.blockBytes}u;
    let bxo = t * p.dim + b * ${d.blockElems}u;
    {${d.decode(emit)}
    }
  }
  // Lo escrito por los otros hilos del grupo tiene que verse antes de leerlo.
  storageBarrier();

  let base = t * p.dim;
  var acc : f32 = 0.0;
  for (var c = lid.x; c < p.dim; c = c + WG) {
    let v = xo[base + c];
    acc = acc + v * v;
  }
${ARBOL}
  let scale = inverseSqrt(scratch[0] / f32(p.dim) + p.eps);
  for (var c = lid.x; c < p.dim; c = c + WG) {
    nrm[base + c] = xo[base + c] * scale * wn[c];
  }
}`;
}

// ── F4: la convolución corta de lfm3, entera, tras `in_proj` ────────────────
// Sustituye los ocho despachos de `convBlock` entre `in_proj` y `out_proj`:
// tres cortes (B, C, x), B·x, la ventana (estado ‖ B·x), la conv depthwise
// causal de K, guardar el estado y C·y. Por canal e y token t:
//   bx[t]  = B[t]·x[t]
//   win[τ] = estado[τ]        si τ < K-1
//          = bx[τ-(K-1)]      si no
//   y[t]   = C[t] · Σ_{r=0..K-1} win[t+r]·w[r]      (en ese orden, desde 0)
//   estado'[τ] = win[T+τ]                           (τ < K-1)
//
// Un hilo por (canal, tesela de TT tokens), con la ventana deslizante en
// registros. El reparto no es capricho: el estado VIEJO solo lo lee la tesela
// 0 (las demás empiezan en τ ≥ TT ≥ K-1, todo B·x), y el estado NUEVO lo escribe
// ese mismo hilo al final, después de haber leído. Si lo escribiera otro hilo
// —el de la última tesela, que tiene esos valores a mano— habría una carrera
// entre grupos que ninguna barrera de WGSL puede ordenar.
//
// En generación (T=1) es un hilo por canal, el mismo paralelismo que la conv de
// siempre; en prompt, T/TT teselas por canal. Con TT = 8 cubre kernels de 2 a 9
// (el de lfm3 es de 7); uno más largo va por la cadena de siempre (ver
// `puedeConvCorta` en ops.fusion.js).
export const CONV_TT = 8;

export function convCortaShader({ K = 7, wg = 64, tt = CONV_TT } = {}) {
  if (!(K >= 2) || tt < K - 1) throw new Error(`conv corta: hace falta TT ≥ K-1 (K=${K}, TT=${tt})`);
  const R = Array.from({ length: K }, (_, r) => r);
  const R1 = R.slice(0, K - 1);
  return /* wgsl */`
const K : u32 = ${K}u;
const TT : u32 = ${tt}u;
struct P { E : u32, T : u32, tiles : u32, cero : u32 };
@group(0) @binding(0) var<storage, read>       bcx : array<f32>;   // [T][3E]: B | C | x (orden de llama.cpp)
@group(0) @binding(1) var<storage, read>       wc  : array<f32>;   // [E][K], r=0 el más antiguo
@group(0) @binding(2) var<storage, read_write> st  : array<f32>;   // estado [K-1][E]
@group(0) @binding(3) var<storage, read_write> y   : array<f32>;   // C·conv [T][E]
@group(0) @binding(4) var<uniform>             p   : P;
${CORTA}
// B·x del token t en el canal e: el MISMO producto que guardaba \`attn.mul(b, xs)\`.
fn bx(t : u32, e : u32) -> f32 {
  let f = t * 3u * p.E;
  return corta(bcx[f + e] * bcx[f + 2u * p.E + e]);
}

// La ventana en la posición τ: el estado de los tokens anteriores o B·x.
fn ventana(tau : u32, e : u32) -> f32 {
  if (tau < K - 1u) { return st[tau * p.E + e]; }
  return bx(tau - (K - 1u), e);
}

@compute @workgroup_size(${wg})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x + gid.y * ${AXIS}u * ${wg}u;   // dos ejes: uno solo topa en 65535 grupos
  if (i >= p.E * p.tiles) { return; }
  let e = i % p.E;                      // hilos contiguos → canales contiguos
  let tile = i / p.E;
  let t0 = tile * TT;
  let t1 = min(t0 + TT, p.T);
${R.map((r) => `  let k${r} = wc[e * K + ${r}u];`).join('\n')}
  // La ventana deslizante: v0 es la posición t0 (la más antigua).
${R1.map((r) => `  var v${r} = ventana(t0 + ${r}u, e);`).join('\n')}
  var v${K - 1} = 0.0;
  for (var t = t0; t < t1; t = t + 1u) {
    v${K - 1} = bx(t, e);
    var acc = 0.0;
${R.map((r) => `    acc = acc + v${r} * k${r};`).join('\n')}
    y[t * p.E + e] = bcx[t * 3u * p.E + p.E + e] * corta(acc);
${R1.map((r) => `    v${r} = v${r + 1};`).join('\n')}
  }
  // El estado nuevo, las últimas K-1 posiciones de la ventana. Se LEE todo antes
  // de escribir nada: con T < K-1 parte de lo nuevo sale del estado viejo.
  if (tile == 0u) {
${R1.map((r) => `    let n${r} = ventana(p.T + ${r}u, e);`).join('\n')}
${R1.map((r) => `    st[${r}u * p.E + e] = n${r};`).join('\n')}
  }
}`;
}

// ── F6: QK-norm + RoPE + escritura de K y V en la caché ─────────────────────
// Sustituye, en el bloque de atención: rmsnorm de q por cabeza, RoPE de q,
// rmsnorm de k, RoPE de k, y las dos copias a la caché (`copyInto`, con la
// conversión a f16 si la caché va en media precisión).
//
// Un grupo de 128 hilos por (token, cabeza): primero las H de consulta, luego
// las KVH de clave y luego las KVH de valor. Las de consulta y clave hacen la
// norma con el árbol de siempre, dejan la fila normalizada en memoria de grupo
// (el RoPE mezcla dimensiones de hilos distintos) y rotan como `ropeShader`: un
// hilo por par, `theta` con la MISMA expresión. Las de valor solo copian.
//
// q, k y v se leen con zancada y desplazamiento por token: hoy son tres
// buffers sueltos (zancada = su anchura, desplazamiento 0), pero si un GEMV
// fusionado los deja en un solo buffer [T][q|k|v], basta con pasar ese buffer
// con otros números. En ese caso (`unBuffer`) se ata UNA vez y no tres: cada
// binding cuenta contra el tope de buffers por etapa, y así la capa global de
// lfm3 (QK-norm y rope_freqs) queda en ocho, el mínimo que garantiza WebGPU.
//
// PARTIDO EN DOS (`partes`). Con tres buffers sueltos, QK-norm y `rope_freqs`,
// el despacho único ata diez buffers de almacenamiento (nueve sin rope_freqs),
// y en un dispositivo que solo dé el mínimo de WebGPU ese pipeline ni se puede
// crear. Para eso hay dos despachos con el MISMO código: 'q', las cabezas de
// consulta (q, pos, wq, qo y rope_freqs), y 'kv', las de clave y valor (k, v,
// pos, wk, las dos cachés y rope_freqs): siete como mucho. Cada grupo hace
// exactamente lo que hacía su cabeza en el despacho único; cambia solo qué
// grupos van en cada despacho, y eso lo dice el uniforme: la primera ranura
// (`s0`) y cuántas hay por token (`nRanuras`), en el orden de siempre
// [consulta | clave | valor]. Las ramas de las cabezas que un despacho no lleva
// no se generan, porque sus buffers no están atados.
//
// La caché se escribe con la disposición de `copyInto`: el token t de la cabeza
// h en ((nPast+t)·KVH + h)·D. Si esa disposición cambiara, cambia aquí también.
export const PARTES_QKV = ['qkv', 'q', 'kv'];

export function enlacesQKV({ norma = true, factores = false, unBuffer = false, partes = 'qkv' } = {}) {
  const conQ = partes !== 'kv', conK = partes !== 'q';
  const entradas = unBuffer ? ['qkv'] : [...(conQ ? ['q'] : []), ...(conK ? ['k', 'v'] : [])];
  const pesos = norma ? [...(conQ ? ['wq'] : []), ...(conK ? ['wk'] : [])] : [];
  return [...entradas, 'pos', ...pesos, ...(conQ ? ['qo'] : []), ...(conK ? ['kc', 'vc'] : []), 'p',
    ...(factores ? ['ff'] : [])];
}

export function qkvShader({ D, norma = true, factores = false, kvF16 = true, wg = WG_FILA, unBuffer = false, partes = 'qkv' } = {}) {
  if (!(D > 0) || D % 2) throw new Error(`QK-norm+RoPE: head_dim ${D} tiene que ser par`);
  if (!PARTES_QKV.includes(partes)) throw new Error(`QK-norm+RoPE: partes «${partes}» desconocidas`);
  const conQ = partes !== 'kv', conK = partes !== 'q';
  // Sin cabezas de clave no se toca la caché: su precisión no entra en el texto
  // (y la variante 'q' es la misma con caché f16 o f32).
  const KV = kvF16 && conK ? 'f16' : 'f32';
  // Lo que depende de si la cabeza es de consulta o de clave: con las dos en el
  // despacho, la rama de siempre; con una sola, solo su lado. Las expresiones
  // son las MISMAS en las tres variantes.
  // Un bloque de varias líneas llega con su sangría de bloque (seis espacios);
  // suelto, sube un nivel.
  const rama = (enQ, enK) => {
    if (conQ && conK) return enQ.includes('\n') ? `if (esQ) {${enQ}} else {${enK}}` : `if (esQ) { ${enQ} } else { ${enK} }`;
    const solo = conQ ? enQ : enK;
    return solo.includes('\n') ? solo.trim().replace(/\n {6}/g, '\n    ') : solo;
  };
  const decl = {
    q:   'var<storage, read>       q   : array<f32>;',
    k:   'var<storage, read>       k   : array<f32>;',
    v:   'var<storage, read>       v   : array<f32>;',
    qkv: 'var<storage, read>       qkv : array<f32>;   // [T][q | k | v], con zancadas',
    pos: 'var<storage, read>       pos : array<i32>;',
    wq:  'var<storage, read>       wq  : array<f32>;   // attn_q_norm [D]',
    wk:  'var<storage, read>       wk  : array<f32>;   // attn_k_norm [D]',
    qo:  'var<storage, read_write> qo  : array<f32>;   // q rotada [T][H][D]',
    kc:  `var<storage, read_write> kc  : array<${KV}>;   // caché K [ctx][KVH][D]`,
    vc:  `var<storage, read_write> vc  : array<${KV}>;   // caché V [ctx][KVH][D]`,
    p:   'var<uniform>             p   : P;',
    ff:  'var<storage, read>       ff  : array<f32>;   // rope_freqs [nDims/2]',
  };
  // Las lecturas de la fila de entrada. Con UN buffer, los desplazamientos ya
  // distinguen q de k; con tres, cada cabeza lee el suyo.
  const lee = unBuffer ? 'fn lee(esQ : bool, i : u32) -> f32 { return qkv[i]; }   // los desplazamientos ya distinguen q de k'
    : conQ && conK ? `fn lee(esQ : bool, i : u32) -> f32 {
  if (esQ) { return q[i]; }
  return k[i];
}` : `fn lee(esQ : bool, i : u32) -> f32 { return ${conQ ? 'q' : 'k'}[i]; }`;
  const leeV = !conK ? '' : `fn leeV(i : u32) -> f32 { return ${unBuffer ? 'qkv' : 'v'}[i]; }`;
  return /* wgsl */`${KV === 'f16' ? 'enable f16;\n' : ''}
const WG : u32 = ${wg}u;
struct P {
  heads : u32, kvHeads : u32, tokens : u32, dim : u32,
  nDims : u32, mode : u32, nPast : u32, cero : u32,
  qStride : u32, qOff : u32, kStride : u32, kOff : u32,
  vStride : u32, vOff : u32, eps : f32, thetaScale : f32,
  freqScale : f32, s0 : u32, nRanuras : u32, pad0 : u32,
};
${enlaces(enlacesQKV({ norma, factores, unBuffer, partes }), decl)}
var<workgroup> scratch : array<f32, ${wg}>;
var<workgroup> fila : array<f32, ${D}>;
${CORTA}
${lee}
${leeV}

@compute @workgroup_size(${wg})
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let g = wid.x + wid.y * ${AXIS}u;     // un grupo por (token, ranura)
  if (g >= p.tokens * p.nRanuras) { return; }
  let t = g / p.nRanuras;
  // La ranura en el orden del despacho único: [consulta | clave | valor].
  let s = p.s0 + g % p.nRanuras;
${conK ? /* wgsl */`  let fc = (p.nPast + t) * p.kvHeads * p.dim;   // el token en la caché: a partir de nPast

  // Cabezas de VALOR: la copia de \`copyInto\`, sin nada más.
  if (s >= p.heads + p.kvHeads) {
    let h = s - p.heads - p.kvHeads;
    let src = t * p.vStride + p.vOff + h * p.dim;
    for (var c = lid.x; c < p.dim; c = c + WG) { vc[fc + h * p.dim + c] = ${KV}(leeV(src + c)); }
    return;                             // uniforme: \`s\` sale del índice del grupo
  }
` : ''}
  let esQ = ${conQ && conK ? 's < p.heads' : conQ ? 'true' : 'false'};
  let h = ${conQ && conK ? 'select(s - p.heads, s, esQ)' : conQ ? 's' : 's - p.heads'};
  let src = ${conQ && conK ? 'select(t * p.kStride + p.kOff + h * p.dim, t * p.qStride + p.qOff + h * p.dim, esQ)'
    : conQ ? 't * p.qStride + p.qOff + h * p.dim' : 't * p.kStride + p.kOff + h * p.dim'};
${norma ? /* wgsl */`
  // 1) La rmsnorm de la cabeza (filas de D), con el reparto y el árbol de
  //    rmsnormShader. El ancho sale del uniforme y no de la constante D, igual
  //    que allí: una división por una constante la puede reescribir el
  //    compilador, y la de la cadena no.
  var acc : f32 = 0.0;
  for (var c = lid.x; c < p.dim; c = c + WG) {
    let x = lee(esQ, src + c);
    acc = acc + x * x;
  }
${ARBOL}
  let scale = inverseSqrt(scratch[0] / f32(p.dim) + p.eps);
  for (var c = lid.x; c < p.dim; c = c + WG) {
    var wv : f32;
    ${rama('wv = wq[c];', 'wv = wk[c];')}
    fila[c] = corta(lee(esQ, src + c) * scale * wv);
  }` : /* wgsl */`
  // Sin QK-norm (llama): el RoPE va directo sobre la proyección.
  for (var c = lid.x; c < p.dim; c = c + WG) { fila[c] = lee(esQ, src + c); }`}
  workgroupBarrier();                   // la fila entera, visible para todos

  // 2) RoPE como \`ropeShader\`: un hilo por par y la MISMA expresión de theta.
  let pairs = p.nDims / 2u;
  let tok = f32(pos[t]);
${conQ ? '  let bq = (t * p.heads + h) * p.dim;   // q: [T][H][D]' : ''}
${conK ? '  let bk = fc + h * p.dim;              // k: la caché' : ''}
  for (var pr = lid.x; pr < pairs; pr = pr + WG) {
    let theta = tok * p.freqScale * pow(p.thetaScale, f32(pr))${factores ? ' / ff[pr]' : ''};
    let cs = cos(theta);
    let sn = sin(theta);
    // La única diferencia entre los dos modos: qué dos posiciones forman el par.
    var i0 : u32;
    var i1 : u32;
    if (p.mode == ${ROPE_NEOX}u) { i0 = pr;      i1 = pr + pairs; }
    else                          { i0 = pr * 2u; i1 = pr * 2u + 1u; }
    let x0 = fila[i0];
    let x1 = fila[i1];
    ${rama(/* wgsl */`
      qo[bq + i0] = x0 * cs - x1 * sn;
      qo[bq + i1] = x0 * sn + x1 * cs;
    `, /* wgsl */`
      // En la cadena el resultado del RoPE se guardaba en f32 y la copia lo
      // convertía: \`corta\` fija ese redondeo antes de la conversión.
      kc[bk + i0] = ${KV}(corta(x0 * cs - x1 * sn));
      kc[bk + i1] = ${KV}(corta(x0 * sn + x1 * cs));
    `)}
  }
  // Las dimensiones no rotadas (RoPE parcial) se copian tal cual.
  for (var c = p.nDims + lid.x; c < p.dim; c = c + WG) {
    ${rama('qo[bq + c] = fila[c];', `kc[bk + c] = ${KV}(fila[c]);`)}
  }
}`;
}
