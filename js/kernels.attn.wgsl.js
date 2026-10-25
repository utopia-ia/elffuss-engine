// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · shaders del bloque de atención de qwen35.
// ─────────────────────────────────────────────────────────────────────────────
// Solo 6 de cada 24 bloques son esto; los otros 18 son SSM. Pero son los únicos
// que necesitan caché KV, así que son los que fijan el coste de contexto largo.
//
// La trampa que define este bloque: `attn_q.weight` proyecta 1024 → 4096, y esa
// fila NO es «Q en la mitad baja, puerta en la alta». Están INTERCALADOS por
// cabeza:
//     [ q_h0(256) | gate_h0(256) | q_h1(256) | gate_h1(256) | … ]
// Partirlo por la mitad da tensores de la forma correcta, magnitudes creíbles y
// un modelo que no funciona. Por eso hay un kernel de extracción con paso en vez
// de una vista contigua.
//
// GQA: 8 cabezas de query contra 2 de clave/valor, o sea 4 queries por cada KV.
// La cabeza KV se elige con h / (H/KVH); no se replican los tensores.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.

// Extrae un sub-tensor con paso: dst[t][h][d] = src[t][h·stride + offset + d].
// Con stride = 2·headDim y offset = 0 saca Q; con offset = headDim, la puerta.
export function strideCopyShader(wg = 256) {
  return /* wgsl */`
const WG : u32 = ${wg}u;
struct P { headDim : u32, heads : u32, tokens : u32, stride : u32,
           offset : u32, pad0 : u32, pad1 : u32, pad2 : u32 };
@group(0) @binding(0) var<storage, read>       src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<uniform>             p   : P;

@compute @workgroup_size(${wg})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x + gid.y * 65535u * ${wg}u;   // dos ejes: uno solo topa en 65535 grupos
  let total = p.headDim * p.heads * p.tokens;
  if (i >= total) { return; }
  let d = i % p.headDim;
  let rest = i / p.headDim;
  let h = rest % p.heads;
  let t = rest / p.heads;
  // La fila del token mide heads·stride; dentro, cada cabeza ocupa «stride» y
  // el sub-tensor que queremos empieza en «offset».
  let srcRow = t * p.heads * p.stride;
  dst[i] = src[srcRow + h * p.stride + p.offset + d];
}`;
}

// scores[h][t][j] = Q[t][h] · K[j][kv(h)], sin escalar y sin enmascarar: la
// escala y la máscara causal las aplica el softmax, igual que en la referencia.
// Compararse contra un `kq` ya escalado sería un falso negativo.
// `kvF16` pone la caché KV en media precisión: la mitad de memoria y la mitad
// de tráfico al leerla, que en el decode se lee entera por token. El acumulador
// sigue en f32 — lo que se guarda en media precisión es el DATO, no la suma, que
// es donde la media precisión sí se notaría.
export function attnScoresShader(wg = 64, kvF16 = false) {
  const KV = kvF16 ? 'f16' : 'f32';
  return /* wgsl */`${kvF16 ? 'enable f16;\n' : ''}
const WG : u32 = ${wg}u;
struct P { headDim : u32, heads : u32, kvHeads : u32, tokens : u32,
           nKv : u32, nPast : u32, pad0 : u32, pad1 : u32 };
@group(0) @binding(0) var<storage, read>       q   : array<f32>;   // [T][H][D]
@group(0) @binding(1) var<storage, read>       k   : array<${KV}>;   // [nKv][KVH][D]
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;   // [H][T][nKv]
@group(0) @binding(3) var<uniform>             p   : P;

@compute @workgroup_size(${wg})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x + gid.y * 65535u * ${wg}u;   // dos ejes: uno solo topa en 65535 grupos
  let total = p.heads * p.tokens * p.nKv;
  if (i >= total) { return; }
  let j = i % p.nKv;
  let rest = i / p.nKv;
  let t = rest % p.tokens;
  let h = rest / p.tokens;

  let group = p.heads / p.kvHeads;          // 4 queries por cada cabeza KV
  let kvh = h / group;
  let qOff = (t * p.heads + h) * p.headDim;
  let kOff = (j * p.kvHeads + kvh) * p.headDim;

  var acc = 0.0;
  for (var d = 0u; d < p.headDim; d = d + 1u) { acc = acc + q[qOff + d] * f32(k[kOff + d]); }
  dst[(h * p.tokens + t) * p.nKv + j] = acc;
}`;
}

// out[t][h][d] = Σ_j probs[h][t][j] · V[j][kv(h)][d].
// Los pesos que caen fuera de la máscara ya vienen a cero del softmax, así que
// no hace falta volver a enmascarar aquí.
export function attnCombineShader(wg = 64, kvF16 = false) {
  const KV = kvF16 ? 'f16' : 'f32';
  return /* wgsl */`${kvF16 ? 'enable f16;\n' : ''}
const WG : u32 = ${wg}u;
struct P { headDim : u32, heads : u32, kvHeads : u32, tokens : u32,
           nKv : u32, nPast : u32, pad0 : u32, pad1 : u32 };
@group(0) @binding(0) var<storage, read>       probs : array<f32>; // [H][T][nKv]
@group(0) @binding(1) var<storage, read>       v     : array<${KV}>; // [nKv][KVH][D]
@group(0) @binding(2) var<storage, read_write> dst   : array<f32>; // [T][H][D]
@group(0) @binding(3) var<uniform>             p     : P;

@compute @workgroup_size(${wg})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x + gid.y * 65535u * ${wg}u;   // dos ejes: uno solo topa en 65535 grupos
  let total = p.tokens * p.heads * p.headDim;
  if (i >= total) { return; }
  let d = i % p.headDim;
  let rest = i / p.headDim;
  let h = rest % p.heads;
  let t = rest / p.heads;

  let group = p.heads / p.kvHeads;
  let kvh = h / group;
  let pOff = (h * p.tokens + t) * p.nKv;

  var acc = 0.0;
  for (var j = 0u; j < p.nKv; j = j + 1u) {
    acc = acc + probs[pOff + j] * f32(v[(j * p.kvHeads + kvh) * p.headDim + d]);
  }
  dst[i] = acc;
}`;
}

// Producto elemento a elemento — la puerta de salida multiplica la atención.
export function mulShader(wg = 256) {
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
  dst[i] = a[i] * b[i];
}`;
}

// ── combine PARTIDO: el mismo resultado, con el contexto repartido ───────────
// El `attnCombineShader` de arriba pone un hilo por (token, cabeza, dimension) y
// recorre las nKv posiciones en un bucle. A contexto largo eso es ruinoso, y
// medido: a 64k de contexto se llevaba el 73 % del tiempo del token, y crecia
// tres veces mas rapido de lo que crecen los bytes que lee (lo mide una prueba
// interna). Tres cosas apiladas, no una:
//
//   1. una cadena de nKv sumas DEPENDIENTES, sin paralelismo de instruccion:
//      cada vuelta espera el resultado de la anterior;
//   2. tokens×cabezas×headDim hilos —6.144 en decode— o sea casi nada de
//      ocupacion para una suma larguisima;
//   3. la V se relee heads/kvHeads veces, porque cada cabeza de consulta
//      recorre por su cuenta la misma V de su cabeza KV.
//
// Este reparte el contexto en BLOQUES trozos y pone un grupo por
// (trozo, cabeza KV, token), con las heads/kvHeads cabezas de consulta de ese
// grupo A LA VEZ. Asi la V se lee UNA vez (ataca el 3), los trozos dan
// ocupacion (el 2) y las cabezas dan cadenas de suma independientes, o sea que
// el paralelismo de instruccion sale gratis de compartir la V (el 1).
//
// Deja sumas parciales que `combineReduceShader` junta. NO es bit a bit igual al
// de arriba: cambia el orden de la suma. Medido contra el, coseno 1.000000 y
// desviaciones del orden de 1e-7 a 1e-6 RELATIVAS a la escala, en toda la
// rejilla de formas que el motor usa (1 y 32 tokens, contextos de 64 a 32.768),
// y en el 27B entero logits bit a bit identicos con seis tokens seguidos
// iguales (lo comprueba una prueba interna).
export function attnCombineSplitShader({ headDim, grupo, subTrozo = 64, kvF16 = false }) {
  const KV = kvF16 ? 'f16' : 'f32';
  return /* wgsl */`${kvF16 ? 'enable f16;\n' : ''}
const D : u32 = ${headDim}u;
const G : u32 = ${grupo}u;
const SJ : u32 = ${subTrozo}u;
struct P { headDim : u32, heads : u32, kvHeads : u32, tokens : u32,
           nKv : u32, bloques : u32, pad0 : u32, pad1 : u32 };
@group(0) @binding(0) var<storage, read>       probs : array<f32>;    // [H][T][nKv]
@group(0) @binding(1) var<storage, read>       v     : array<${KV}>;  // [nKv][KVH][D]
@group(0) @binding(2) var<storage, read_write> parc  : array<f32>;    // [T][H][BLOQUES][D]
@group(0) @binding(3) var<uniform>             p     : P;

// Las probabilidades del sub-trozo, para las G cabezas de este grupo. Sin esto,
// los D hilos leerian de memoria global el MISMO valor D veces por cada j.
var<workgroup> ps : array<f32, ${grupo * subTrozo}u>;

@compute @workgroup_size(${headDim})
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let bloque = wid.x;
  let kvh = wid.y;
  let t = wid.z;
  let d = lid.x;

  // El ultimo trozo puede quedar corto: se recorta con nKv. Rellenarlo con
  // ceros sumaria terminos que no existen.
  let porBloque = (p.nKv + p.bloques - 1u) / p.bloques;
  let j0 = bloque * porBloque;
  let j1 = min(j0 + porBloque, p.nKv);

  var acc : array<f32, G>;
  for (var g = 0u; g < G; g = g + 1u) { acc[g] = 0.0; }

  var jb = j0;
  loop {
    if (jb >= j1) { break; }
    let n = min(SJ, j1 - jb);
    for (var k = d; k < G * n; k = k + D) {
      let g = k / n;
      let jj = k % n;
      let h = kvh * G + g;
      ps[g * SJ + jj] = probs[(h * p.tokens + t) * p.nKv + jb + jj];
    }
    workgroupBarrier();
    for (var jj = 0u; jj < n; jj = jj + 1u) {
      // UNA lectura de V alimenta las G cabezas: ahi esta el ahorro de bytes.
      let vv = f32(v[((jb + jj) * p.kvHeads + kvh) * D + d]);
      for (var g = 0u; g < G; g = g + 1u) {
        acc[g] = acc[g] + ps[g * SJ + jj] * vv;
      }
    }
    workgroupBarrier();
    jb = jb + n;
  }

  for (var g = 0u; g < G; g = g + 1u) {
    let h = kvh * G + g;
    parc[((t * p.heads + h) * p.bloques + bloque) * D + d] = acc[g];
  }
}`;
}

// Junta las sumas parciales del kernel de arriba. Un hilo por salida.
export function combineReduceShader(wg = 256) {
  return /* wgsl */`
struct P { headDim : u32, heads : u32, tokens : u32, bloques : u32 };
@group(0) @binding(0) var<storage, read>       parc : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst  : array<f32>;
@group(0) @binding(2) var<uniform>             p    : P;
@compute @workgroup_size(${wg})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x + gid.y * 65535u * ${wg}u;   // dos ejes: uno solo topa en 65535
  let total = p.tokens * p.heads * p.headDim;
  if (i >= total) { return; }
  let d = i % p.headDim;
  let th = i / p.headDim;
  var s = 0.0;
  for (var b = 0u; b < p.bloques; b = b + 1u) {
    s = s + parc[(th * p.bloques + b) * p.headDim + d];
  }
  dst[i] = s;
}`;
}

// Memoria compartida que pide el scores con K compartida: SOLO las filas de q.
// Se expone para poder RECHAZAR la forma con un mensaje que diga cuanto pedia,
// en vez de dejar que reviente la validacion de WebGPU con un texto generico.
export function compartidaScores(headDim, grupo) {
  return grupo * headDim * 4;
}

// Cuantos trozos para una forma. Sale del barrido y no de un razonamiento: el
// optimo es el MAXIMO de bloques casi siempre, tambien con 32 tokens, con dos
// topes medidos — trozos de menos de 32 posiciones no aportan, y por encima de
// 32 bloques empeora (a nKv 65.000, 64 y 128 bloques salian algo peores).
export function bloquesCombine(nKv) {
  return Math.max(1, Math.min(32, Math.ceil(nKv / 32)));
}

// ── scores PARTIDO: la K leida UNA vez para las cabezas que la comparten ─────
// El `attnScoresShader` de arriba pone un hilo por (cabeza, token, posicion) y
// cada hilo recorre headDim. El paralelismo es de sobra —millones de hilos— y
// el kernel va al 96 % del techo de lectura: NO es ineficiente.
// El problema es que lee SEIS VECES lo que necesita.
//
// La cuenta, con las formas del 27B: cada cabeza de consulta recorre por su
// cuenta la K de su cabeza KV, y hay heads/kvHeads = 6 por cada una. O sea que
// la K de una capa se lee 6 veces. Medido a 64k de contexto, el kernel se
// llevaba el 69 % de lo que el contexto añade al token, y los bytes cuadran con
// el tiempo casi exactamente — por eso afinar la forma no servia de nada: no
// habia nada que afinar, habia que leer menos.
//
// Este pone un grupo por (token, cabeza KV, bloque de posiciones) y calcula las
// G cabezas de consulta de ese grupo A LA VEZ, con la K del bloque en memoria
// compartida. Asi la K se lee una vez, y de paso las G cabezas dan G cadenas de
// multiplicacion-suma independientes sobre el mismo dato.
//
// Es EXACTO: mismo producto, mismos terminos, mismo orden de acumulacion sobre
// d. La unica diferencia es de donde se lee cada valor.
export function attnScoresSharedShader({ headDim, grupo, kvF16 = false, wg = 64 }) {
  const KV = kvF16 ? 'f16' : 'f32';
  return /* wgsl */`${kvF16 ? 'enable f16;\n' : ''}
const D : u32 = ${headDim}u;
const G : u32 = ${grupo}u;
struct P { headDim : u32, heads : u32, kvHeads : u32, tokens : u32,
           nKv : u32, pad0 : u32, pad1 : u32, pad2 : u32 };
@group(0) @binding(0) var<storage, read>       q   : array<f32>;    // [T][H][D]
@group(0) @binding(1) var<storage, read>       k   : array<${KV}>;  // [nKv][KVH][D]
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;    // [H][T][nKv]
@group(0) @binding(3) var<uniform>             p   : P;

// SOLO las G filas de q van a memoria compartida, y es a proposito. La primera
// version de este kernel metia ahi tambien un bloque de K: pedia por grupo mas
// del 90 % del tope de memoria compartida, o sea UN grupo residente por unidad
// de computo, y
// sin varios grupos en vuelo no hay con que esconder la latencia. Resultado:
// leia 6 veces menos bytes y corria 18 veces peor — TRES VECES mas lento que el
// kernel de un hilo por salida. Es el mismo modo de fallo que hundio el intento
// de cachear la rejilla i-quant en memoria compartida, y lo repeti teniendolo
// apuntado.
//
// Asi que la K se comparte por REGISTROS, no por memoria compartida: cada hilo
// lee k[j][d] UNA vez y la usa para las G cabezas. Se mantiene el patron de
// acceso del kernel original —que iba al 96 % del techo de lectura, o sea que
// no habia nada que arreglar ahi— y se le quitan los 5/6 de bytes redundantes
// que le metia la GQA.
var<workgroup> qs : array<f32, ${grupo * headDim}u>;

@compute @workgroup_size(${wg})
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let kvh = wid.y;
  let t = wid.z;
  let tid = lid.x;
  for (var i = tid; i < G * D; i = i + ${wg}u) {
    qs[i] = q[(t * p.heads + kvh * G + i / D) * D + i % D];
  }
  workgroupBarrier();

  let j = wid.x * ${wg}u + tid;
  if (j >= p.nKv) { return; }
  let kOff = (j * p.kvHeads + kvh) * D;
  var acc : array<f32, G>;
  for (var g = 0u; g < G; g = g + 1u) { acc[g] = 0.0; }
  for (var d = 0u; d < D; d = d + 1u) {
    // UNA lectura de K alimenta las G cabezas: ahi esta todo el ahorro, y las G
    // cadenas independientes dan el paralelismo de instruccion de propina.
    let kv = f32(k[kOff + d]);
    for (var g = 0u; g < G; g = g + 1u) { acc[g] = acc[g] + qs[g * D + d] * kv; }
  }
  for (var g = 0u; g < G; g = g + 1u) {
    dst[((kvh * G + g) * p.tokens + t) * p.nKv + j] = acc[g];
  }
}`;
}
