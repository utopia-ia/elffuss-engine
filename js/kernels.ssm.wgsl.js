// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · shaders de la capa SSM (gated delta net) de qwen35.
// ─────────────────────────────────────────────────────────────────────────────
// 18 de cada 24 bloques del modelo son ESTO, no atención. En vez de una caché KV
// que crece con el contexto, llevan un estado recurrente de tamaño FIJO: una
// matriz de 128×128 por cabeza que se actualiza token a token. Por eso una
// cuenta de memoria que tratara todos los bloques como de atención se queda
// larga: el estado entero de las 18 capas es una fracción pequeña de lo que
// pesa el modelo.
//
// La regla, por token, cabeza y secuencia (S es el estado, k/q/v los vectores de
// la cabeza, g el decaimiento y β la tasa de escritura):
//   1. DECAIMIENTO   S ← diag(e^g) · S
//   2. LECTURA       u  = v − Sᵀk
//   3. DELTA         δ  = β · u
//   4. ESCRITURA     S ← S + k δᵀ
//   5. SALIDA        o  = (1/√128) · Sᵀq        ← con el S YA actualizado
//
// Dos órdenes que no son negociables: el decaimiento va ANTES de leer, y la
// salida usa el estado DESPUÉS de escribir. Invertir cualquiera de los dos da un
// modelo que genera texto y pierde el hilo, sin ningún error.
//
// El estado se guarda TRASPUESTO, state[j][i] con j = dimensión de value e
// i = dimensión de key, para que cada hilo posea una fila contigua.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.

export const SSM_HEAD_DIM = 128;      // head_k_dim = head_v_dim = ssm_d_state
export const SSM_CONV_K = 4;          // qwen35.ssm.conv_kernel

// L2-norm por filas. OJO: NO es rmsnorm y no comparte fórmula.
//   rmsnorm: scale = 1 / sqrt(mean(x²) + eps)     ← eps DENTRO de la raíz
//   l2norm:  scale = 1 / max(sqrt(sum(x²)), eps)  ← eps como SUELO, fuera
// Se aplica a q y k del SSM (a v no). Confundirlas da valores parecidos y una
// deriva que crece con la profundidad.
export function l2normShader(wg = 128) {
  return /* wgsl */`
const WG : u32 = ${wg}u;
struct P { ne0 : u32, rows : u32, eps : f32, pad : u32 };
@group(0) @binding(0) var<storage, read>       src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<uniform>             p   : P;
var<workgroup> scratch : array<f32, ${wg}>;

@compute @workgroup_size(${wg})
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let row = wid.x + wid.y * 65535u;   // dos ejes: uno solo topa en 65535
  if (row >= p.rows) { return; }
  let base = row * p.ne0;
  var acc = 0.0;
  for (var c = lid.x; c < p.ne0; c = c + WG) { let v = src[base + c]; acc = acc + v * v; }
  scratch[lid.x] = acc;
  workgroupBarrier();
  for (var off = WG >> 1u; off > 0u; off = off >> 1u) {
    if (lid.x < off) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + off]; }
    workgroupBarrier();
  }
  let scale = 1.0 / max(sqrt(scratch[0]), p.eps);
  for (var c = lid.x; c < p.ne0; c = c + WG) { dst[base + c] = src[base + c] * scale; }
}`;
}

// Convolución causal DEPTHWISE de kernel 4, sin bias. Cada canal solo ve su
// propia fila de pesos: no hay mezcla entre canales.
//   y[c][t] = Σ_{r=0..3} buf[c][t+r] · W[c][r]
// `buf` es la ventana [3+T] por canal: los 3 primeros vienen del estado del
// token anterior. r=0 es el más antiguo y r=3 el actual — los pesos van en
// orden temporal directo, NO invertidos como en una convolución de libro.
export function ssmConvShader(wg = 64, k = SSM_CONV_K) {
  return /* wgsl */`
const WG : u32 = ${wg}u;
const K : u32 = ${k}u;
struct P { channels : u32, tokens : u32, window : u32, pad : u32 };
@group(0) @binding(0) var<storage, read>       buf : array<f32>;   // [channels][window]
@group(0) @binding(1) var<storage, read>       w   : array<f32>;   // [channels][K]
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;   // [tokens][channels]
@group(0) @binding(3) var<uniform>             p   : P;

@compute @workgroup_size(${wg})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x + gid.y * 65535u * ${wg}u;   // dos ejes: uno solo topa en 65535 grupos
  let total = p.channels * p.tokens;
  if (idx >= total) { return; }
  let c = idx % p.channels;
  let t = idx / p.channels;
  var acc = 0.0;
  for (var r = 0u; r < K; r = r + 1u) {
    acc = acc + buf[c * p.window + t + r] * w[c * K + r];
  }
  dst[t * p.channels + c] = acc;
}`;
}

// El núcleo: la regla delta con puerta. Un workgroup por cabeza, un hilo por
// dimensión de value. Cada hilo posee la fila j del estado (128 f32 contiguos)
// y la recorre tres veces por token: leer, escribir, y proyectar la salida.
//
// El estado vive en un buffer de storage, no en memoria privada: tiene que
// sobrevivir entre tokens de todas formas, y meterlo en `private` sería la
// matriz entera por workgroup, con riesgo de volcado a memoria.
export function gatedDeltaNetShader(headDim = SSM_HEAD_DIM) {
  return /* wgsl */`
const S : u32 = ${headDim}u;
struct P { heads : u32, tokens : u32, kvHeads : u32, pad1 : u32,
           scale : f32, pad2 : f32, pad3 : f32, pad4 : f32 };
@group(0) @binding(0) var<storage, read>       q     : array<f32>;  // [T][H][S]
@group(0) @binding(1) var<storage, read>       k     : array<f32>;  // [T][H][S]
@group(0) @binding(2) var<storage, read>       v     : array<f32>;  // [T][H][S]
@group(0) @binding(3) var<storage, read>       g     : array<f32>;  // [T][H]
@group(0) @binding(4) var<storage, read>       beta  : array<f32>;  // [T][H]
@group(0) @binding(5) var<storage, read_write> state : array<f32>;  // [H][S][S] (j,i)
@group(0) @binding(6) var<storage, read_write> out   : array<f32>;  // [T][H][S]
@group(0) @binding(7) var<uniform>             p     : P;

var<workgroup> shk : array<f32, ${headDim}>;
var<workgroup> shq : array<f32, ${headDim}>;

@compute @workgroup_size(${headDim})
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let h = wid.x;                        // cabeza de VALOR
  if (h >= p.heads) { return; }
  let j = lid.x;                        // esta invocación posee la columna j
  let row = (h * S + j) * S;            // inicio de la fila j del estado

  // GQA DENTRO DEL SSM. Hay más cabezas de valor que de clave: en el 27B son 48
  // contra 16, o sea tres cabezas de valor comparten cada q/k. En el 0.8B son
  // 16 y 16, y por eso asumir que eran iguales funcionaba ahí y solo ahí — el
  // modelo grande salía con texto plausible y basura.
  //
  // Y el reparto es MÓDULO, no división, que es lo contrario de lo que hace la
  // GQA de la atención. No es un capricho: el checkpoint original guarda las
  // cabezas de valor AGRUPADAS bajo su cabeza de clave (v0,v1,v2 de k0, luego
  // las de k1…), pero el conversor a GGUF las repermuta a orden TESELADO
  // (v0 de k0, v0 de k1, … v1 de k0, …) para que la difusión barata de ggml
  // valga. Ese reordenado lo lleva TODO lo que va por cabeza de valor: v, z,
  // alpha, beta, A_log, dt_bias, la parte v de conv1d y las columnas de
  // out_proj. Se leen tal cual del fichero; lo único que hay que emparejar
  // aquí es q/k.
  //   · ggml: ggml_repeat / iq1 = iv1 % n_k_heads
  //   · conversor: _LinearAttentionVReorderBase._reorder_v_heads
  // Dividir en vez de modular da un modelo que genera palabras del contexto en
  // bucle: cada cabeza lee la clave de otra, y aun así todo tiene la forma
  // correcta y nada peta.
  let hk = h % p.kvHeads;

  // Bucle SECUENCIAL sobre tokens: la recurrencia no se puede paralelizar en t.
  for (var t = 0u; t < p.tokens; t = t + 1u) {
    let vv = (t * p.heads + h) * S;      // v, g, beta y la salida van por cabeza de VALOR
    let vk = (t * p.kvHeads + hk) * S;    // q y k, por cabeza de CLAVE
    shk[j] = k[vk + j];
    shq[j] = q[vk + j];
    workgroupBarrier();

    let gv = exp(g[t * p.heads + h]);
    let bv = beta[t * p.heads + h];

    // Sᵀk con el estado VIEJO. El decaimiento se saca fuera del sumatorio:
    // g·(Sᵀk) = (g·S)ᵀk, así se ahorra una pasada y es idéntico.
    var kv = 0.0;
    for (var i = 0u; i < S; i = i + 1u) { kv = kv + state[row + i] * shk[i]; }
    let delta = (v[vv + j] - gv * kv) * bv;

    // Decaimiento y escritura en la misma pasada, y de paso la salida con el
    // estado YA actualizado.
    var o = 0.0;
    for (var i = 0u; i < S; i = i + 1u) {
      let s = gv * state[row + i] + shk[i] * delta;
      state[row + i] = s;
      o = o + s * shq[i];
    }
    out[vv + j] = o * p.scale;
    workgroupBarrier();               // el token siguiente reescribe shk/shq
  }
}`;
}

// RMSNorm con puerta: rms(x) · W · silu(z).
// El peso se aplica ANTES de multiplicar por silu(z), y silu(z) NO entra en el
// cálculo de la media cuadrática. Es rms(x)·W·silu(z), no rms(x·silu(z))·W.
export function gatedNormShader(wg = 128) {
  return /* wgsl */`
const WG : u32 = ${wg}u;
struct P { ne0 : u32, rows : u32, eps : f32, pad : u32 };
@group(0) @binding(0) var<storage, read>       src : array<f32>;
@group(0) @binding(1) var<storage, read>       w   : array<f32>;   // [ne0]
@group(0) @binding(2) var<storage, read>       z   : array<f32>;   // misma forma que src
@group(0) @binding(3) var<storage, read_write> dst : array<f32>;
@group(0) @binding(4) var<uniform>             p   : P;
var<workgroup> scratch : array<f32, ${wg}>;

@compute @workgroup_size(${wg})
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let row = wid.x + wid.y * 65535u;   // dos ejes: uno solo topa en 65535
  if (row >= p.rows) { return; }
  let base = row * p.ne0;
  var acc = 0.0;
  for (var c = lid.x; c < p.ne0; c = c + WG) { let x = src[base + c]; acc = acc + x * x; }
  scratch[lid.x] = acc;
  workgroupBarrier();
  for (var off = WG >> 1u; off > 0u; off = off >> 1u) {
    if (lid.x < off) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + off]; }
    workgroupBarrier();
  }
  let scale = inverseSqrt(scratch[0] / f32(p.ne0) + p.eps);
  for (var c = lid.x; c < p.ne0; c = c + WG) {
    let zc = z[base + c];
    dst[base + c] = src[base + c] * scale * w[c] * (zc / (1.0 + exp(-zc)));
  }
}`;
}

// alpha → g: softplus(alpha + dt_bias) · a.
// El softplus de ggml corta en 20: por encima devuelve x tal cual, para no
// calcular exp(x) de un número grande. Se replica.
// `a` ya viene NEGATIVO en el fichero (guarda −exp(A_log)), así que aquí no se
// niega ni se exponencia nada: hacerlo daría un decaimiento explosivo.
export function decayGateShader(wg = 64) {
  return /* wgsl */`
const WG : u32 = ${wg}u;
struct P { heads : u32, tokens : u32, pad0 : u32, pad1 : u32 };
@group(0) @binding(0) var<storage, read>       alpha : array<f32>;  // [T][H]
@group(0) @binding(1) var<storage, read>       dt    : array<f32>;  // [H]
@group(0) @binding(2) var<storage, read>       a     : array<f32>;  // [H], negativo
@group(0) @binding(3) var<storage, read_write> dst   : array<f32>;  // [T][H]
@group(0) @binding(4) var<uniform>             p     : P;

fn softplus(x : f32) -> f32 { if (x > 20.0) { return x; } return log(1.0 + exp(x)); }

@compute @workgroup_size(${wg})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x + gid.y * 65535u * ${wg}u;   // dos ejes: uno solo topa en 65535 grupos
  if (i >= p.heads * p.tokens) { return; }
  let h = i % p.heads;
  dst[i] = softplus(alpha[i] + dt[h]) * a[h];
}`;
}

// Sigmoide elemento a elemento — la β del SSM y la puerta de salida de la
// atención salen de aquí.
export function sigmoidShader(wg = 256) {
  return /* wgsl */`
const WG : u32 = ${wg}u;
struct P { n : u32, pad0 : u32, pad1 : u32, pad2 : u32 };
@group(0) @binding(0) var<storage, read>       src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<uniform>             p   : P;

@compute @workgroup_size(${wg})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x + gid.y * 65535u * ${wg}u;   // dos ejes: uno solo topa en 65535 grupos
  if (i >= p.n) { return; }
  dst[i] = 1.0 / (1.0 + exp(-src[i]));
}`;
}
