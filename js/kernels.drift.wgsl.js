// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · GEMM del prefill con la estructura de ML Drift.
// ─────────────────────────────────────────────────────────────────────────────
// El GEMM teselado (kernels.gemm.wgsl.js) sube teselas f32 a memoria compartida,
// pone dos barreras por paso y hace 2×4 salidas escalares por hilo. Este hace lo
// contrario, que es lo que hace ML Drift (Apache-2.0) en su convolución 1×1:
//   · f16 en memoria y vec4<f16>·dot, acumulador en f32;
//   · SIN memoria compartida ni barreras;
//   · cada hilo calcula 16 filas × 2 tokens, y los 32 hilos del grupo en tokens
//     consecutivos leyendo el MISMO peso (difusión: una lectura sirve a todos).
// Medido en un microbanco con pesos densos: ×4 sobre el GEMM teselado.
//
// Los pesos Q8_0 se DESCUANTIZAN a f16 antes, una vez por matriz y lote, en la
// disposición que pide el bucle: [rows/16][K/4][16] vec4. Descuantizar dentro
// del bucle repetiría la descodificación en los 32 hilos que comparten el peso,
// y con solo 2 tokens de reúso por hilo eso se come la ventaja.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.
import { HELPERS } from './kernels.wgsl.js';

export const DRIFT_TPW = 32;   // hilos por grupo, todos en tokens
export const DRIFT_TPT = 2;    // tokens por hilo
export const DRIFT_FILAS = 16; // filas de salida por hilo

// Q8_0 [rows][K] (bloques de 34 bytes) → f16 [rows/16][K/4][16] vec4.
// Un hilo por vec4 de salida.
export function q8aDriftShader(wg = 256) {
  return /* wgsl */`
enable f16;
struct P { rows : u32, k : u32, bpr : u32, total : u32 };
@group(0) @binding(0) var<storage, read>       w   : array<u32>;
@group(0) @binding(1) var<storage, read_write> dst : array<vec4<f16>>;
@group(0) @binding(2) var<uniform>             p   : P;
${HELPERS}
@compute @workgroup_size(${wg})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x + gid.y * 65535u * ${wg}u;
  if (i >= p.total) { return; }
  let ks4 = p.k / 4u;
  let j = i % 16u;
  let ks = (i / 16u) % ks4;
  let og = i / (16u * ks4);
  let row = og * 16u + j;
  let kk = ks * 4u;
  let blk = (row * p.bpr + kk / 32u) * 34u;
  let d = load_scale(blk);
  let q = load_u32_at(blk + 2u + (kk % 32u));
  dst[i] = vec4<f16>(f16(f32(get_byte_i32(q, 0u)) * d), f16(f32(get_byte_i32(q, 1u)) * d),
                     f16(f32(get_byte_i32(q, 2u)) * d), f16(f32(get_byte_i32(q, 3u)) * d));
}`;
}

// Activaciones f32 [T][K] → vec4<f16> [T][K/4].
export function f32aF16Shader(wg = 256) {
  return /* wgsl */`
enable f16;
struct P { n4 : u32, pad0 : u32, pad1 : u32, pad2 : u32 };
@group(0) @binding(0) var<storage, read>       src : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst : array<vec4<f16>>;
@group(0) @binding(2) var<uniform>             p   : P;
@compute @workgroup_size(${wg})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x + gid.y * 65535u * ${wg}u;
  if (i >= p.n4) { return; }
  dst[i] = vec4<f16>(src[i]);
}`;
}

// Y[t][n] = Σ_k X[t][k]·W[n][k]. `tpw` hilos por grupo, todos en tokens; cada
// hilo calcula `tpt` tokens × 16 filas. workgroup_id.y = og (bloque de 16 filas). El
// último grupo de tokens puede pasarse del lote: lee la última fila válida y no
// escribe. Más tokens por hilo = cada peso leído sirve a más cuentas, a cambio
// de más registros vivos (16·tpt acumuladores).
export function driftGemmShader(tpt = DRIFT_TPT, tpw = DRIFT_TPW) {
  const T = [...Array(tpt).keys()];
  return /* wgsl */`
enable f16;
struct P { ks4 : u32, rows : u32, tokens : u32, pad : u32 };
@group(0) @binding(0) var<storage, read>       W : array<vec4<f16>>;   // [OG][KS][16]
@group(0) @binding(1) var<storage, read>       X : array<vec4<f16>>;   // [T][KS]
@group(0) @binding(2) var<storage, read_write> Y : array<f32>;         // [T][rows]
@group(0) @binding(3) var<uniform>             p : P;
@compute @workgroup_size(${tpw})
fn main(@builtin(workgroup_id) wg : vec3<u32>, @builtin(local_invocation_id) l : vec3<u32>) {
  let og = wg.y + wg.z * 65535u;
  if (og * 16u >= p.rows) { return; }   // sin barreras: salir es legal
  let t0 = (wg.x * ${tpw}u + l.x) * ${tpt}u;
  let ultimo = p.tokens - 1u;
${T.map(t => `  var a${t} : array<f32, 16>;
  let x${t}i = min(t0 + ${t}u, ultimo) * p.ks4;`).join('\n')}
  for (var j = 0u; j < 16u; j++) { ${T.map(t => `a${t}[j] = 0.0;`).join(' ')} }
  var wi = og * p.ks4 * 16u;
  for (var ks = 0u; ks < p.ks4; ks++) {
${T.map(t => `    let x${t} = X[x${t}i + ks];`).join('\n')}
    for (var j = 0u; j < 16u; j++) {
      let w = W[wi + j];
${T.map(t => `      a${t}[j] += f32(dot(x${t}, w));`).join('\n')}
    }
    wi += 16u;
  }
  let n0 = og * 16u;
${T.map(t => `  if (t0 + ${t}u < p.tokens) { for (var j = 0u; j < 16u; j++) { Y[(t0 + ${t}u) * p.rows + n0 + j] = a${t}[j]; } }`).join('\n')}
}`;
}
