// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · plomería del forward: embeddings, caché y ventana de conv.
// ─────────────────────────────────────────────────────────────────────────────
// Tres operaciones que no son matemática del modelo pero sin las cuales no hay
// forward. Las tres mueven datos, y las tres tienen una forma de equivocarse que
// no produce ningún error: escribir en el sitio de al lado.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.

// El gather de embeddings vivía aquí, clavado a Q8_0 y con su propia copia de
// los helpers de descodificación. Ahora está en `kernels.wgsl.js`, generado por
// el mismo andamio que el matmul: la tabla del 27B no es Q8_0, y dos copias de
// la descodificación son dos sitios donde equivocarse.

// Copia un bloque de elementos dentro de otro buffer, en un desplazamiento.
// Es cómo se añaden K y V a la caché: se escribe a partir de nPast, sin tocar
// lo anterior. Un desplazamiento mal calculado pisa tokens ya cacheados y el
// modelo empieza a «recordar» cosas que no dijo nadie.
// `dstF16` es para llenar la caché KV en media precisión: la fuente sigue en
// f32 (viene del rope) y la conversión se hace AQUÍ, al guardar, que es el único
// sitio donde ocurre una vez por token en vez de una vez por lectura.
export function copyIntoShader(wg = 256, dstF16 = false) {
  return /* wgsl */`${dstF16 ? 'enable f16;\n' : ''}
const WG : u32 = ${wg}u;
struct P { n : u32, dstOffset : u32, srcOffset : u32, pad : u32 };
@group(0) @binding(0) var<storage, read>       src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<${dstF16 ? 'f16' : 'f32'}>;
@group(0) @binding(2) var<uniform>             p   : P;

@compute @workgroup_size(${wg})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x + gid.y * 65535u * ${wg}u;   // dos ejes: uno solo topa en 65535 grupos
  if (i >= p.n) { return; }
  dst[p.dstOffset + i] = ${dstF16 ? 'f16' : 'f32'}(src[p.srcOffset + i]);
}`;
}

// Monta la ventana de la convolución: por canal, los K-1 valores que quedaron
// del token anterior seguidos de los T nuevos. Esto es a la vez una TRANSPOSICIÓN
// (qkv viene [token][canal] y la conv lee [canal][tiempo]) y una concatenación.
//
// El desplazamiento del estado es implícito: no hay memmove, se copian las
// últimas K-1 columnas de la ventana de vuelta al estado.
export function convWindowShader(wg = 256) {
  return /* wgsl */`
const WG : u32 = ${wg}u;
struct P { channels : u32, tokens : u32, keep : u32, window : u32 };
@group(0) @binding(0) var<storage, read>       state : array<f32>;  // [keep][C]
@group(0) @binding(1) var<storage, read>       qkv   : array<f32>;  // [T][C]
@group(0) @binding(2) var<storage, read_write> dst   : array<f32>;  // [C][window]
@group(0) @binding(3) var<uniform>             p     : P;

@compute @workgroup_size(${wg})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x + gid.y * 65535u * ${wg}u;   // dos ejes: uno solo topa en 65535 grupos
  if (i >= p.channels * p.window) { return; }
  let tau = i % p.window;
  let c = i / p.window;
  if (tau < p.keep) { dst[i] = state[tau * p.channels + c]; }
  else              { dst[i] = qkv[(tau - p.keep) * p.channels + c]; }
}`;
}

// Guarda las últimas keep columnas de la ventana como estado del token siguiente.
export function convSaveShader(wg = 256) {
  return /* wgsl */`
const WG : u32 = ${wg}u;
struct P { channels : u32, tokens : u32, keep : u32, window : u32 };
@group(0) @binding(0) var<storage, read>       win   : array<f32>;  // [C][window]
@group(0) @binding(1) var<storage, read_write> state : array<f32>;  // [keep][C]
@group(0) @binding(2) var<uniform>             p     : P;

@compute @workgroup_size(${wg})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x + gid.y * 65535u * ${wg}u;   // dos ejes: uno solo topa en 65535 grupos
  if (i >= p.channels * p.keep) { return; }
  let c = i % p.channels;
  let tau = i / p.channels;
  state[i] = win[c * p.window + (p.window - p.keep + tau)];
}`;
}

// silu(x) = x·sigmoid(x), elemento a elemento. Se usa sobre los 6144 canales
// justo después de la convolución, antes de cortar q|k|v.
// Puerta POR CABEZA de lfm3: un escalar por cabeza y token que multiplica las D
// salidas de esa cabeza. dst[t][h][d] = a[t][h][d] · sigmoid(g[t][h]). La
// sigmoide va dentro: un despacho y un tensor intermedio menos por capa.
export function mulCabezaShader(wg = 256) {
  return /* wgsl */`
const WG : u32 = ${wg}u;
struct P { n : u32, headDim : u32, pad0 : u32, pad1 : u32 };
@group(0) @binding(0) var<storage, read>       a   : array<f32>;   // [T][H][D]
@group(0) @binding(1) var<storage, read>       g   : array<f32>;   // [T][H]
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;
@group(0) @binding(3) var<uniform>             p   : P;

@compute @workgroup_size(${wg})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x + gid.y * 65535u * ${wg}u;   // dos ejes: uno solo topa en 65535 grupos
  if (i >= p.n) { return; }
  let x = g[i / p.headDim];
  dst[i] = a[i] / (1.0 + exp(-x));
}`;
}

export function siluShader(wg = 256) {
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
  let x = src[i];
  dst[i] = x / (1.0 + exp(-x));
}`;
}
