// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · shaders WGSL de las operaciones de red: norm, rope, softmax, swiglu.
// ─────────────────────────────────────────────────────────────────────────────
// Las cuatro son iguales en los dos tipos de bloque de qwen35, así que sirven
// pase lo que pase con el resto del forward.
//
// Las cuatro comparten además un patrón de fallo: un error de indexado que
// PRESERVA la forma y la escala. No petan, no dan NaN, y el modelo genera texto
// gramatical e incoherente. Solo se ven comparando contra una referencia. Por eso
// cada una lleva aquí escrito qué es lo que se rompe en silencio:
//   · rmsnorm  → acumular en f16: el acumulador se estanca o desborda a inf y la
//                fila sale entera a cero.
//   · rope     → confundir NEOX con NORM: el par de rotación son elementos
//                distintos. Texto fluido que se degrada al crecer el contexto.
//   · softmax  → no restar el máximo: exp() desborda con logits > 88.
//   · swiglu   → intercambiar gate y up: silu(g)·u ≠ silu(u)·g, y la perplejidad
//                se dispara sin que nada avise.
//
// WG_SIZE va por SUSTITUCIÓN DE TEXTO y no como `override` a propósito: el
// análisis de uniformidad de WGSL exige que el recorrido de un bucle con
// barreras sea constante de compilación, y con override no lo es.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.

export const ROPE_NORM = 0;      // pares adyacentes:  x[2p], x[2p+1]
export const ROPE_NEOX = 2;      // mitad-mitad:       x[p],  x[p + nDims/2]

// Reducción en árbol. La barrera va FUERA del if: si se mete dentro, WGSL lo
// rechaza al crear el pipeline (análisis de uniformidad) — que es la forma
// amable de fallar. Lo que NINGÚN validador caza es olvidar la barrera WAR al
// reutilizar el scratch, y eso sí es una carrera de datos silenciosa.
function reduce(op) {
  return /* wgsl */`
  scratch[lid.x] = acc;
  workgroupBarrier();
  for (var off = WG >> 1u; off > 0u; off = off >> 1u) {
    if (lid.x < off) { scratch[lid.x] = ${op}; }
    workgroupBarrier();
  }`;
}

// y = (x / sqrt(mean(x²) + eps)) · w, por filas.
// El epsilon se suma a la MEDIA, dentro de la raíz. Sumarlo fuera es otro
// clásico y da un resultado parecido pero distinto.
export function rmsnormShader(wg = 128) {
  return /* wgsl */`
const WG : u32 = ${wg}u;
struct P { ne0 : u32, rows : u32, eps : f32, pad : u32 };
@group(0) @binding(0) var<storage, read>       src : array<f32>;
@group(0) @binding(1) var<storage, read>       w   : array<f32>;
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;
@group(0) @binding(3) var<uniform>             p   : P;
var<workgroup> scratch : array<f32, ${wg}>;

@compute @workgroup_size(${wg})
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let row = wid.x + wid.y * 65535u;   // dos ejes: uno solo topa en 65535                       // uniforme en el workgroup
  if (row >= p.rows) { return; }
  let base = row * p.ne0;

  // Acumulación SIEMPRE en f32. Este bucle tiene salida no uniforme (cada hilo
  // sale en una iteración distinta), así que aquí dentro no puede haber barreras.
  var acc : f32 = 0.0;
  for (var c = lid.x; c < p.ne0; c = c + WG) {
    let v = src[base + c];
    acc = acc + v * v;                   // v*v, no pow(v, 2.0)
  }
${reduce('scratch[lid.x] + scratch[lid.x + off]')}
  let scale = inverseSqrt(scratch[0] / f32(p.ne0) + p.eps);

  for (var c = lid.x; c < p.ne0; c = c + WG) {
    dst[base + c] = src[base + c] * scale * w[c];
  }
}`;
}

// RoPE. theta_scale se calcula en el HOST: en el shader la potencia se
// acumularía con el error de pow() y a posiciones grandes eso se nota.
// Una fila es una cabeza de longitud ne0; solo se rotan las primeras nDims.
export function ropeShader(wg = 64) {
  return /* wgsl */`
const WG : u32 = ${wg}u;
struct P {
  ne0 : u32, rows : u32, heads : u32, nDims : u32,
  mode : u32, pad0 : u32, pad1 : u32, pad2 : u32,
  thetaScale : f32, freqScale : f32, pad3 : f32, pad4 : f32,
};
@group(0) @binding(0) var<storage, read>       src : array<f32>;
@group(0) @binding(1) var<storage, read>       pos : array<i32>;
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;
@group(0) @binding(3) var<uniform>             p   : P;

@compute @workgroup_size(${wg})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let pairs = p.nDims / 2u;
  let total = p.rows * pairs;
  let idx = gid.x + gid.y * 65535u * ${wg}u;   // dos ejes: uno solo topa en 65535 grupos
  if (idx >= total) { return; }

  let row  = idx / pairs;                // cabeza dentro del token
  let pair = idx % pairs;
  let base = row * p.ne0;
  let token = row / p.heads;             // filas ordenadas [cabeza, token]
  let theta = f32(pos[token]) * p.freqScale * pow(p.thetaScale, f32(pair));
  let c = cos(theta);
  let s = sin(theta);

  // AQUÍ está la única diferencia entre los dos modos: qué dos posiciones de
  // memoria forman el par. NORM son adyacentes, NEOX está a media fila.
  var i0 : u32;
  var i1 : u32;
  if (p.mode == ${ROPE_NEOX}u) { i0 = pair;      i1 = pair + pairs; }
  else                          { i0 = pair * 2u; i1 = pair * 2u + 1u; }

  let x0 = src[base + i0];
  let x1 = src[base + i1];
  dst[base + i0] = x0 * c - x1 * s;
  dst[base + i1] = x0 * s + x1 * c;

  // Las dimensiones no rotadas se copian tal cual. Solo lo hace el primer par
  // de cada fila, para no escribir lo mismo desde varios hilos.
  if (pair == 0u) {
    for (var i = p.nDims; i < p.ne0; i = i + 1u) { dst[base + i] = src[base + i]; }
  }
}`;
}

// Softmax por filas con máscara causal calculada EN EL SHADER: materializar una
// máscara [n_q × n_kv] es un tensor entero de ancho de banda tirado.
// El centinela es finito (-1e30) y no -inf: si una fila quedara entera
// enmascarada, con -inf sale NaN; con -1e30 sale una distribución uniforme, que
// es raro pero no envenena el resto de la red.
export function softmaxShader(wg = 256) {
  return /* wgsl */`
const WG : u32 = ${wg}u;
const NEG_BIG : f32 = -1.0e30;
struct P { nKv : u32, nQ : u32, nPast : u32, rows : u32, scale : f32, pad0 : f32, pad1 : f32, pad2 : f32 };
@group(0) @binding(0) var<storage, read>       src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<uniform>             p   : P;
var<workgroup> scratch : array<f32, ${wg}>;

@compute @workgroup_size(${wg})
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let row = wid.x + wid.y * 65535u;   // dos ejes: uno solo topa en 65535
  if (row >= p.rows) { return; }
  let base = row * p.nKv;
  let qAbs = p.nPast + (row % p.nQ);

  // Recorrido UNIFORME (iters igual para todos) porque hay barreras dentro.
  let iters = (p.nKv + WG - 1u) / WG;

  var acc : f32 = NEG_BIG;
  for (var it = 0u; it < iters; it = it + 1u) {
    let k = lid.x + it * WG;
    if (k < p.nKv && k <= qAbs) { acc = max(acc, src[base + k] * p.scale); }
  }
${reduce('max(scratch[lid.x], scratch[lid.x + off])')}
  let rowMax = scratch[0];
  workgroupBarrier();                    // WAR: se va a reutilizar scratch

  acc = 0.0;
  for (var it = 0u; it < iters; it = it + 1u) {
    let k = lid.x + it * WG;
    if (k < p.nKv) {
      // exp(x - max) ∈ (0,1]: imposible que desborde, en cualquier precisión.
      let e = select(0.0, exp(src[base + k] * p.scale - rowMax), k <= qAbs);
      dst[base + k] = e;
      acc = acc + e;
    }
  }
${reduce('scratch[lid.x] + scratch[lid.x + off]')}
  let inv = 1.0 / scratch[0];

  for (var it = 0u; it < iters; it = it + 1u) {
    let k = lid.x + it * WG;
    if (k < p.nKv) { dst[base + k] = dst[base + k] * inv; }
  }
}`;
}

// SwiGLU con gate y up en tensores separados, que es como vienen en el GGUF.
// silu(gate) · up, en ese orden. Invertirlo no peta y arruina el modelo.
export function swigluShader(wg = 256) {
  return /* wgsl */`
const WG : u32 = ${wg}u;
struct P { n : u32, pad0 : u32, pad1 : u32, pad2 : u32 };
@group(0) @binding(0) var<storage, read>       gate : array<f32>;
@group(0) @binding(1) var<storage, read>       up   : array<f32>;
@group(0) @binding(2) var<storage, read_write> dst  : array<f32>;
@group(0) @binding(3) var<uniform>             p    : P;

// x/(1+exp(-x)), no x*exp(x)/(1+exp(x)): la segunda forma desborda con x grande.
fn silu(x : f32) -> f32 { return x / (1.0 + exp(-x)); }

@compute @workgroup_size(${wg})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x + gid.y * 65535u * ${wg}u;   // dos ejes: uno solo topa en 65535 grupos
  if (i >= p.n) { return; }              // sin barreras: el early-return es legal
  dst[i] = silu(gate[i]) * up[i];
}`;
}

// Suma elemento a elemento — la cadena residual. Trivial, pero se acumula 24
// veces: en f16 los incrementos pequeños se pierden por absorción.
export function addShader(wg = 256) {
  return /* wgsl */`
const WG : u32 = ${wg}u;
struct P { n : u32, pad0 : u32, pad1 : u32, pad2 : u32 };
@group(0) @binding(0) var<storage, read>       a   : array<f32>;
@group(0) @binding(1) var<storage, read>       b   : array<f32>;
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;
@group(0) @binding(3) var<uniform>             p   : P;

@compute @workgroup_size(${wg})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x + gid.y * 65535u * ${wg}u;   // dos ejes: uno solo topa en 65535 grupos
  if (i >= p.n) { return; }
  dst[i] = a[i] + b[i];
}`;
}
