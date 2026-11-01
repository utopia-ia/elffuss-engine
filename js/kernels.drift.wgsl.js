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
import { bloquesWGSL } from './kernels.bloques.wgsl.js';

export const DRIFT_TPW = 32;   // hilos por grupo, todos en tokens
export const DRIFT_TPT = 2;    // tokens por hilo
export const DRIFT_FILAS = 16; // filas de salida por hilo

// Pesos cuantizados [rows][K] → f16 [rows/16][K/4][16] vec4. Un hilo por vec4
// de salida: cuatro pesos CONSECUTIVOS en k de una fila. Cada tipo dice cómo
// sacar esos cuatro de su bloque (todos son de 32 pesos y kk es múltiplo de 4,
// así que nunca cruzan de bloque).
// Q8_0R/Q4_0R son los mismos tipos REORDENADOS al cargar (`__ELFFUSS_REPACK`,
// ver kernels.bloques.wgsl.js).
export const DRIFT_TIPOS = new Set(['Q8_0', 'Q4_0', 'Q8_0R', 'Q4_0R']);
const CUATRO = {
  // 34 bytes: escala f16 + 32 int8.
  Q8_0: { bytes: 34, wgsl: `
  let d = load_scale(blk);
  let q = load_u32_at(blk + 2u + i0);
  let v = vec4<f32>(f32(get_byte_i32(q, 0u)), f32(get_byte_i32(q, 1u)), f32(get_byte_i32(q, 2u)), f32(get_byte_i32(q, 3u))) * d;` },
  // 18 bytes: escala f16 + 16 bytes de nibbles; byte j = peso j (bajo) y j+16 (alto).
  Q4_0: { bytes: 18, wgsl: `
  let d = load_scale(blk);
  let q = load_u32_at(blk + 2u + (i0 & 15u));
  let sh = select(0u, 4u, i0 >= 16u);
  let v = (vec4<f32>(f32((q >> sh) & 0xFu), f32((q >> (8u + sh)) & 0xFu), f32((q >> (16u + sh)) & 0xFu), f32((q >> (24u + sh)) & 0xFu)) - 8.0) * d;` },
  // Los reordenados: las MISMAS dos expresiones de arriba con las lecturas
  // cambiadas. `blk` sigue saliendo como si el formato fuera el del GGUF y se
  // divide por el tamaño del bloque para tener su índice; la escala y la palabra
  // de cuantos son los mismos bits, así que el f16 que se escribe también. Se
  // llama `bi` y no `i` porque aquí se está en el cuerpo de `main`, donde `i`
  // ya es el índice del hilo.
  Q8_0R: { bytes: 34, helpers: bloquesWGSL('Q8_0R'), wgsl: `
  let bi = blk / 34u;
  let d = bq_escala(bi);
  let q = bq_palabra(bi, i0 / 4u);
  let v = vec4<f32>(f32(get_byte_i32(q, 0u)), f32(get_byte_i32(q, 1u)), f32(get_byte_i32(q, 2u)), f32(get_byte_i32(q, 3u))) * d;` },
  Q4_0R: { bytes: 18, helpers: bloquesWGSL('Q4_0R'), wgsl: `
  let bi = blk / 18u;
  let d = bq_escala(bi);
  let q = bq_palabra(bi, (i0 & 15u) / 4u);
  let sh = select(0u, 4u, i0 >= 16u);
  let v = (vec4<f32>(f32((q >> sh) & 0xFu), f32((q >> (8u + sh)) & 0xFu), f32((q >> (16u + sh)) & 0xFu), f32((q >> (24u + sh)) & 0xFu)) - 8.0) * d;` },
};

export function driftDequantShader(type, wg = 256) {
  const c = CUATRO[type];
  if (!c) throw new Error(`drift: sin descuantización para ${type}`);
  return /* wgsl */`
enable f16;
struct P { rows : u32, k : u32, bpr : u32, total : u32 };
@group(0) @binding(0) var<storage, read>       w   : array<u32>;
@group(0) @binding(1) var<storage, read_write> dst : array<vec4<f16>>;
@group(0) @binding(2) var<uniform>             p   : P;
${HELPERS}${c.helpers || ''}
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
  let blk = (row * p.bpr + kk / 32u) * ${c.bytes}u;
  let i0 = kk % 32u;${c.wgsl}
  dst[i] = vec4<f16>(v);
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
