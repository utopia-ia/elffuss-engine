// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · MOSS-TTS-Nano (texto a voz) en la tarjeta.
// ─────────────────────────────────────────────────────────────────────────────
// MOSS-TTS-Nano escribe audio trama a trama (12,5 por segundo de voz). Cada trama
// es un paso de un GPT-2 «global» de 12 capas sobre una fila de 17 ids (un id de
// texto y 16 códigos de audio, cuyos embeddings se suman), y luego hasta 17 pasos
// de un GPT-2 «local» de una capa: el primero decide si la voz sigue o termina, y
// los 16 siguientes sortean un código por libro, cada uno con el embedding del
// anterior como entrada nueva. Es escritura token a token pura, lo que este motor
// hace bien; el códec que convierte los códigos en audio se queda fuera.
//
// Qué va dónde:
//   · tarjeta: las capas (LayerNorm con sesgo, QKV, RoPE de pares adyacentes,
//     atención con caché, GELU), la norma final y las cabezas de logits;
//   · CPU: la suma de embeddings de cada fila (17 filas de 768 por paso, nada) y
//     el muestreo, copiado del grafo ONNX oficial (moss_tts_local_fixed_sampled_
//     frame): penalización 1,2 a los códigos ya dados, temperatura 0,8, top-k 25,
//     top-p 0,95 y el sorteo con el número uniforme que da quien llama. Con los
//     mismos uniformes que el runtime oficial salen las mismas tramas.
//
// Pesos: el GGUF de tools/moss_a_gguf.py, en BF16 (los bits del original).
import { parseGGUF } from './gguf.js';
import { createOps, ROPE_NORM } from './ops.js';
import { readback } from './device.js';
import { copyIntoShader } from './kernels.model.wgsl.js';
export { MOSS_ONNX_MAPA, MOSS_ONNX_REVISION } from './moss-mapa.js';

const H = 768, NH = 12, HD = 64, NVQ = 16, CB = 1024, FFN = 3072, EPS = 1e-5;
const SLOT = 9, END = 7, PAD = 1024;
// Muestreo del grafo oficial.
const PENAL = 1.2, TEMP = 0.8, TOPK = 25, TOPP = 0.95, U_MAX = 0.99999994;

// ── Kernels propios: LayerNorm con sesgo, sesgo, sesgo + GELU (tanh) ────────
const layerNormShader = (wg) => /* wgsl */`
struct P { n : u32, rows : u32, eps : f32, z : u32 };
@group(0) @binding(0) var<storage, read> x : array<f32>;
@group(0) @binding(1) var<storage, read> w : array<f32>;
@group(0) @binding(2) var<storage, read> b : array<f32>;
@group(0) @binding(3) var<storage, read_write> y : array<f32>;
@group(0) @binding(4) var<uniform> p : P;
var<workgroup> s1 : array<f32, ${wg}>;
var<workgroup> s2 : array<f32, ${wg}>;
@compute @workgroup_size(${wg})
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
  let row = wid.x;
  if (row >= p.rows) { return; }
  let base = row * p.n;
  var a = 0.0;
  for (var c = lid.x; c < p.n; c = c + ${wg}u) { a = a + x[base + c]; }
  s1[lid.x] = a;
  workgroupBarrier();
  for (var o = ${wg / 2}u; o > 0u; o = o >> 1u) { if (lid.x < o) { s1[lid.x] = s1[lid.x] + s1[lid.x + o]; } workgroupBarrier(); }
  let media = s1[0] / f32(p.n);
  var v = 0.0;
  for (var c = lid.x; c < p.n; c = c + ${wg}u) { let d = x[base + c] - media; v = v + d * d; }
  s2[lid.x] = v;
  workgroupBarrier();
  for (var o = ${wg / 2}u; o > 0u; o = o >> 1u) { if (lid.x < o) { s2[lid.x] = s2[lid.x] + s2[lid.x + o]; } workgroupBarrier(); }
  let inv = inverseSqrt(s2[0] / f32(p.n) + p.eps);
  for (var c = lid.x; c < p.n; c = c + ${wg}u) { y[base + c] = (x[base + c] - media) * inv * w[c] + b[c]; }
}`;

const sesgoShader = (wg, gelu) => /* wgsl */`
struct P { n : u32, total : u32, z0 : u32, z1 : u32 };
@group(0) @binding(0) var<storage, read> x : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> y : array<f32>;
@group(0) @binding(3) var<uniform> p : P;
@compute @workgroup_size(${wg})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= p.total) { return; }
  let v = x[i] + b[i % p.n];
  ${gelu
    // tanh recortado: hay tarjetas que lo calculan como (e^2x-1)/(e^2x+1) y a
    // partir de x≈44 da inf/inf = NaN. En ±15 tanh ya es ±1 exacto en f32.
    ? 'y[i] = 0.5 * v * (1.0 + tanh(clamp(0.7978845834732056 * (v + 0.044715 * v * v * v), -15.0, 15.0)));'
    : 'y[i] = v;'}
}`;

// bf16 → f32 por una vista compartida: crear arrays por número costaba cientos de ms.
const U32 = new Uint32Array(1), F32 = new Float32Array(U32.buffer);
const bf16 = (u) => { U32[0] = u << 16; return F32[0]; };

// Una fuente de pesos da cada tensor como bf16 en el orden de PyTorch ([salida,
// entrada] en las matrices). Hay dos: el GGUF de tools/moss_a_gguf.py y los
// ficheros .data de la exportación ONNX oficial, que el navegador ya baja para
// el camino ONNX: así no hay que alojar pesos convertidos.

/** Fuente desde el GGUF (BF16 tal cual). */
export async function fuenteGGUF(ranged) {
  const g = await parseGGUF(ranged);
  if (g.meta['general.architecture'] !== 'moss-tts-nano') throw new Error('moss: el fichero no es de MOSS-TTS-Nano');
  const T = new Map(g.tensors.map((t) => [t.name, t]));
  return {
    async bf16(name) {
      const t = T.get(name);
      if (!t) throw new Error(`moss: falta ${name}`);
      return new Uint16Array((await ranged.slice(t.start, t.nBytes)).slice(0));
    },
  };
}

/**
 * Fuente desde los .data de la exportación ONNX oficial (f32). `mapa` dice, por
 * tensor, fichero, desplazamiento, longitud, forma y si va traspuesto (los pesos
 * de MatMul van [entrada, salida]). Los valores son bf16 guardados en f32, así
 * que quedarse con los 16 bits altos es exacto.
 * @param {Record<string, { size: number, slice(o: number, l: number): Promise<ArrayBuffer> }>} ficheros por nombre de .data
 */
export function fuenteOnnx(ficheros, mapa) {
  return {
    async bf16(name) {
      const d = mapa[name];
      if (!d) throw new Error(`moss: el mapa no tiene ${name}`);
      const f32 = new Float32Array(await ficheros[d.file].slice(d.off, d.len));
      const u32 = new Uint32Array(f32.buffer);
      const out = new Uint16Array(f32.length);
      if (!d.T) { for (let i = 0; i < out.length; i++) out[i] = u32[i] >>> 16; return out; }
      const [filas, cols] = d.dims;           // ONNX [entrada, salida] → [salida, entrada]
      for (let r = 0; r < filas; r++) for (let c = 0; c < cols; c++) out[c * filas + r] = u32[r * cols + c] >>> 16;
      return out;
    },
  };
}

/**
 * Carga MOSS-TTS-Nano en la tarjeta desde una fuente (`fuenteGGUF` o `fuenteOnnx`).
 * @param {{ bf16(name: string): Promise<Uint16Array> }} fuente
 * @param {GPUDevice} device
 */
export async function cargaMoss(fuente, device) {
  if (typeof fuente.bf16 !== 'function') fuente = await fuenteGGUF(fuente);   // compat: un GGUF «ranged»
  const ops = createOps(device, { debug: false });
  const { ctx, quant, nn, attn } = ops;
  const bytes = async (name) => { const u = await fuente.bf16(name); return new Uint8Array(u.buffer, u.byteOffset, u.byteLength); };
  const f32De = (u8) => { const u = new Uint16Array(u8.buffer, u8.byteOffset, u8.length / 2); return Float32Array.from(u, bf16); };
  const sube = (data, label) => {
    const buf = device.createBuffer({ label, size: Math.ceil((data.byteLength + 4) / 4) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(buf, 0, data);
    return buf;
  };
  const W = new Map();
  // Matrices en BF16 tal cual; vectores (normas y sesgos) a f32.
  const mat = async (name, filas) => { const u8 = await bytes(name); W.set(name, sube(filas ? u8.subarray(filas[0] * H * 2, filas[1] * H * 2) : u8, name)); };
  const vec = async (name) => W.set(name, sube(f32De(await bytes(name)), name));
  const capas = [['transformer', 12], ['local_transformer', 1]];
  for (const [pre, n] of capas) {
    for (let i = 0; i < n; i++) {
      const p = `${pre}.h.${i}`;
      for (const v of ['ln_1.weight', 'ln_1.bias', 'ln_2.weight', 'ln_2.bias', 'attn.c_attn.bias', 'attn.c_proj.bias', 'mlp.fc_in.bias', 'mlp.fc_out.bias']) await vec(`${p}.${v}`);
      for (const m of ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.fc_in.weight', 'mlp.fc_out.weight']) await mat(`${p}.${m}`);
    }
    await vec(`${pre}.ln_f.weight`); await vec(`${pre}.ln_f.bias`);
  }
  for (let c = 0; c < NVQ; c++) await mat(`audio_lm_heads.${c}.weight`);
  // De la cabeza de texto solo hacen falta dos filas: «sigue» (9) y «fin» (7).
  {
    const u8 = await bytes('text_lm_head.weight');
    const dos = new Uint8Array(2 * H * 2);
    dos.set(u8.subarray(SLOT * H * 2, (SLOT + 1) * H * 2), 0);
    dos.set(u8.subarray(END * H * 2, (END + 1) * H * 2), H * 2);
    W.set('cabeza_texto', sube(dos, 'cabeza_texto'));
  }
  // Embeddings en la CPU, en BF16: la suma de una fila son 17 × 768 sumas.
  const emb = { texto: new Uint16Array((await bytes('transformer.wte.weight')).slice().buffer), audio: [] };
  for (let c = 0; c < NVQ; c++) emb.audio.push(new Uint16Array((await bytes(`audio_embeddings.${c}.weight`)).slice().buffer));

  const pipe = ctx.pipeline;
  const raw = (label, p, buffers, groups) => ops.dispatchAny(label, p, buffers, groups);
  async function layerNorm(x, pre, rows) {
    const p = await pipe('moss_ln', layerNormShader(256));
    const y = ctx.alloc(rows * H * 4, 'moss.ln');
    const u = ctx.uniform([new Uint32Array([H, rows]), new Float32Array([EPS]), new Uint32Array([0])]);
    await raw('moss_ln', p, [x, W.get(`${pre}.weight`), W.get(`${pre}.bias`), y, u], rows);
    return y;
  }
  async function sesgo(x, b, rows, n, gelu = false) {
    const k = gelu ? 'moss_sesgo_gelu' : 'moss_sesgo';
    const p = await pipe(k, sesgoShader(256, gelu));
    const y = ctx.alloc(rows * n * 4, k);
    const u = ctx.uniform([new Uint32Array([n, rows * n, 0, 0])]);
    await raw(k, p, [x, W.get(b), y, u], Math.ceil(rows * n / 256));
    return y;
  }
  async function copia(src, dst, dstOff, n) {
    const p = await pipe('copy_into', copyIntoShader(256, false));
    const u = ctx.uniform([new Uint32Array([n, dstOff, 0, 0])]);
    await raw('copy_into', p, [src, dst, u], Math.ceil(n / 256));
  }
  const mm = (name, x, rows, k, t) => quant.matmul('BF16', W.get(name), x, rows, k, t);
  const persistente = (n, label) => device.createBuffer({ label, size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });

  /** Una pila (global o local) con su caché de atención. */
  function pila(pre, n, maxPos) {
    const k = Array.from({ length: n }, (_, i) => persistente(maxPos * H, `${pre}.k${i}`));
    const v = Array.from({ length: n }, (_, i) => persistente(maxPos * H, `${pre}.v${i}`));
    let pasado = 0;
    return {
      get pasado() { return pasado; },
      reinicia() { pasado = 0; },
      // x: [t, H] en la tarjeta; devuelve la salida tras ln_f, [t, H].
      async paso(x, t) {
        if (pasado + t > maxPos) throw new Error(`moss: ${pre} sin sitio (${pasado}+${t} > ${maxPos})`);
        const pos = ctx.alloc(t * 4, 'pos');
        device.queue.writeBuffer(pos, 0, Int32Array.from({ length: t }, (_, i) => pasado + i));
        const nKv = pasado + t, cfg = { headDim: HD, heads: NH, kvHeads: NH, tokens: t, nKv, nPast: pasado, kvF16: false };
        for (let i = 0; i < n; i++) {
          const p = `${pre}.h.${i}`;
          const h = await layerNorm(x, `${p}.ln_1`, t);
          const qkv = await sesgo(await mm(`${p}.attn.c_attn.weight`, h, 3 * H, H, t), `${p}.attn.c_attn.bias`, t, 3 * H);
          const corte = (off) => attn.strideCopy(qkv, { headDim: H, heads: 1, tokens: t, stride: 3 * H, offset: off });
          const rope = (s) => nn.rope(s, pos, { rows: t * NH, ne0: HD, heads: NH, nDims: HD, mode: ROPE_NORM, freqBase: 10000 });
          const q = await rope(await corte(0));
          await copia(await rope(await corte(H)), k[i], pasado * H, t * H);
          await copia(await corte(2 * H), v[i], pasado * H, t * H);
          const sc = await attn.scores(q, k[i], cfg);
          const pr = await nn.softmax(sc, { rows: NH * t, nKv, nQ: t, nPast: pasado, scale: 1 / Math.sqrt(HD) });
          const at = await attn.combine(pr, v[i], cfg);
          const o = await sesgo(await mm(`${p}.attn.c_proj.weight`, at, H, H, t), `${p}.attn.c_proj.bias`, t, H);
          x = await nn.add(x, o, t * H);
          const h2 = await layerNorm(x, `${p}.ln_2`, t);
          const f = await sesgo(await mm(`${p}.mlp.fc_in.weight`, h2, FFN, H, t), `${p}.mlp.fc_in.bias`, t, FFN, true);
          const o2 = await sesgo(await mm(`${p}.mlp.fc_out.weight`, f, H, FFN, t), `${p}.mlp.fc_out.bias`, t, H);
          x = await nn.add(x, o2, t * H);
        }
        pasado += t;
        return await layerNorm(x, `${pre}.ln_f`, t);
      },
    };
  }

  // La suma de embeddings de las filas (id de texto + 16 códigos; el relleno no suma).
  function embebe(filas) {
    const t = filas.length / (NVQ + 1), out = new Float32Array(t * H);
    for (let r = 0; r < t; r++) {
      const fila = filas.subarray(r * (NVQ + 1), (r + 1) * (NVQ + 1)), o = r * H;
      const tx = fila[0] * H;
      for (let j = 0; j < H; j++) out[o + j] = bf16(emb.texto[tx + j]);
      for (let c = 0; c < NVQ; c++) {
        const id = fila[1 + c];
        if (id === PAD) continue;
        const a = emb.audio[c], b = id * H;
        for (let j = 0; j < H; j++) out[o + j] += bf16(a[b + j]);
      }
    }
    return out;
  }
  const filaDe = (tabla, id) => Float32Array.from(tabla.subarray(id * H, (id + 1) * H), bf16);

  return { ops, device, W, emb, pila, embebe, filaDe, mm, copia, persistente, layerNorm, sesgo, maxGlobal: 0 };
}

// Muestreo de un libro con el número uniforme `u`, como el grafo oficial (f32).
export function sorteaCodigo(logits, visto, u) {
  const f = Math.fround, n = logits.length, l = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let x = logits[i];
    if (visto[i]) x = x < 0 ? f(x * f(PENAL)) : f(x / f(PENAL));
    l[i] = f(x / f(TEMP));
  }
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => l[b] - l[a] || a - b).slice(0, TOPK);
  const sm = (v) => { const m = Math.max(...v.filter(Number.isFinite)); const e = v.map((x) => (Number.isFinite(x) ? f(Math.exp(f(x - m))) : 0)); const s = e.reduce((a, b) => f(a + b), 0); return e.map((x) => f(x / s)); };
  const v = idx.map((i) => l[i]);
  const p = sm(v);
  let acc = 0;
  const v2 = v.map((x, k) => { const antes = acc; acc = f(acc + p[k]); return f(antes) < f(TOPP) ? x : -Infinity; });
  const p2 = sm(v2);
  let c = 0, cnt = 0;
  for (const x of p2) { c = f(c + x); if (c < f(u)) cnt++; }
  return idx[Math.min(cnt, TOPK - 1)];
}

/**
 * Genera las tramas de una petición. `filas`: Int32Array de filas × 17 (las de
 * cloneRows de sdk/audio). `uniforme`: () => [0,1), en el orden del runtime
 * oficial (uno para «sigue», luego uno por libro).
 */
export async function generaTramas(m, filas, uniforme, { max = 375, enTrama = null, maxContexto = 2048, depura = null } = {}) {
  const { ops, device } = m;
  const { ctx } = ops;
  const glob = m.global || (m.global = m.pila('transformer', 12, maxContexto));
  const loc = m.local || (m.local = m.pila('local_transformer', 1, NVQ + 1));
  const oculto = m.oculto || (m.oculto = m.persistente(H, 'moss.oculto'));
  const logits = m.logits || (m.logits = m.persistente(CB, 'moss.logits'));
  glob.reinicia();
  const clamp = (u) => Math.min(U_MAX, Math.max(0, u));
  const sube = (arr) => { const b = ctx.alloc(arr.byteLength, 'moss.x'); device.queue.writeBuffer(b, 0, arr); return b; };

  // Prefill de la petición: queda el oculto de la última fila.
  const nFilas = filas.length / (NVQ + 1);
  await ops.batch('moss.prefill', async () => {
    const y = await glob.paso(sube(m.embebe(filas)), nFilas);
    await m.copia(await ops.attn.strideCopy(y, { headDim: H, heads: 1, tokens: 1, stride: H, offset: (nFilas - 1) * H }), oculto, 0, H);
  });
  ctx.recycle();
  if (depura) depura.oculto = new Float32Array(await readback(device, oculto, 0, H * 4));

  const visto = Array.from({ length: NVQ }, () => new Uint8Array(CB));
  const tramas = [];
  for (let paso = 0; paso < max; paso++) {
    const ua = clamp(uniforme());
    const us = Array.from({ length: NVQ }, () => clamp(uniforme()));
    loc.reinicia();
    // ¿Sigue? Softmax de dos: logit de «sigue» contra el de «fin».
    await ops.batch('moss.local0', async () => {
      const h = await loc.paso(oculto, 1);
      await m.copia(await m.mm('cabeza_texto', h, 2, H, 1), logits, 0, 2);
    });
    const [ls, lf] = new Float32Array(await readback(device, logits, 0, 8));
    if (depura && paso === 0) depura.sigue = [ls, lf];
    ctx.recycle();
    const mx = Math.max(ls, lf), es = Math.exp(ls - mx), ef = Math.exp(lf - mx);
    if (!(ua <= Math.fround(es / (es + ef)))) break;

    const trama = [];
    let x = m.filaDe(m.emb.texto, SLOT);
    for (let c = 0; c < NVQ; c++) {
      await ops.batch('moss.local', async () => {
        const h = await loc.paso(sube(x), 1);
        await m.copia(await m.mm(`audio_lm_heads.${c}.weight`, h, CB, H, 1), logits, 0, CB);
      });
      const lg = new Float32Array(await readback(device, logits, 0, CB * 4));
      ctx.recycle();
      const code = sorteaCodigo(lg, visto[c], us[c]);
      trama.push(code);
      x = m.filaDe(m.emb.audio[c], code);
    }
    trama.forEach((code, c) => { visto[c][code] = 1; });
    tramas.push(trama);
    enTrama?.(tramas.length, trama);
    // Paso global con la fila de la trama.
    const fila = new Int32Array(NVQ + 1).fill(PAD);
    fila[0] = SLOT; trama.forEach((code, c) => { fila[1 + c] = code; });
    await ops.batch('moss.global', async () => {
      const y = await glob.paso(sube(m.embebe(fila)), 1);
      await m.copia(y, oculto, 0, H);
    });
    ctx.recycle();
  }
  return tramas;
}

// ── Muestreo en la tarjeta ───────────────────────────────────────────────────
// El mismo sorteo que `sorteaCodigo`, en un grupo de 256 hilos: penalización y
// temperatura, los 25 mayores (valor de mayor a menor, a igualdad el índice
// menor), y un hilo hace las dos softmax, el top-p y el sorteo en f32. Después
// todos copian el embedding del código elegido a `xsig` (la entrada del paso
// local siguiente) y lo suman a `fila` (la fila de la trama para el paso global),
// y el código queda marcado como visto para las tramas siguientes.
const sorteoShader = () => /* wgsl */`
struct P { c : u32, k : u32, z0 : u32, z1 : u32 };
@group(0) @binding(0) var<storage, read> logits : array<f32>;
@group(0) @binding(1) var<storage, read_write> visto : array<u32>;
@group(0) @binding(2) var<storage, read> us : array<f32>;
@group(0) @binding(3) var<storage, read> tablas : array<u32>;
@group(0) @binding(4) var<storage, read_write> xsig : array<f32>;
@group(0) @binding(5) var<storage, read_write> fila : array<f32>;
@group(0) @binding(6) var<storage, read_write> res : array<f32>;
@group(0) @binding(7) var<uniform> p : P;
const NEG = -3.4028234663852886e38;
var<workgroup> l : array<f32, ${CB}>;
var<workgroup> bv : array<f32, 256>;
var<workgroup> bi : array<u32, 256>;
var<workgroup> topV : array<f32, ${TOPK}>;
var<workgroup> topI : array<u32, ${TOPK}>;
var<workgroup> elegido : u32;
fn mejor(av : f32, ai : u32, bv2 : f32, bi2 : u32) -> bool { return av > bv2 || (av == bv2 && ai < bi2); }
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid : vec3<u32>) {
  let t = lid.x;
  for (var i = t; i < ${CB}u; i = i + 256u) {
    var x = logits[i];
    if (visto[p.c * ${CB}u + i] != 0u) { x = select(x / ${PENAL}, x * ${PENAL}, x < 0.0); }
    l[i] = x / ${TEMP};
  }
  workgroupBarrier();
  for (var r = 0u; r < ${TOPK}u; r = r + 1u) {
    var v = NEG; var ix = 0xffffffffu;
    for (var i = t; i < ${CB}u; i = i + 256u) { if (mejor(l[i], i, v, ix)) { v = l[i]; ix = i; } }
    bv[t] = v; bi[t] = ix;
    workgroupBarrier();
    for (var o = 128u; o > 0u; o = o >> 1u) {
      if (t < o && mejor(bv[t + o], bi[t + o], bv[t], bi[t])) { bv[t] = bv[t + o]; bi[t] = bi[t + o]; }
      workgroupBarrier();
    }
    if (t == 0u) { topV[r] = bv[0]; topI[r] = bi[0]; l[bi[0]] = NEG; }
    workgroupBarrier();
  }
  if (t == 0u) {
    let u = us[p.k * ${NVQ}u + p.c];
    let m = topV[0];
    var s = 0.0;
    for (var j = 0u; j < ${TOPK}u; j = j + 1u) { s = s + exp(topV[j] - m); }
    var acc = 0.0; var vivos = 0u;
    for (var j = 0u; j < ${TOPK}u; j = j + 1u) {
      if (acc < ${TOPP}) { vivos = j + 1u; }
      acc = acc + exp(topV[j] - m) / s;
    }
    var s2 = 0.0;
    for (var j = 0u; j < vivos; j = j + 1u) { s2 = s2 + exp(topV[j] - m); }
    var c2 = 0.0; var n = 0u;
    for (var j = 0u; j < vivos; j = j + 1u) { c2 = c2 + exp(topV[j] - m) / s2; if (c2 < u) { n = n + 1u; } }
    let code = topI[min(n, ${TOPK - 1}u)];
    elegido = code;
    res[p.k * ${NVQ + 2}u + 2u + p.c] = f32(code);
    visto[p.c * ${CB}u + code] = 1u;
  }
  workgroupBarrier();
  let base = (p.c * ${CB}u + elegido) * ${H}u;
  for (var j = t; j < ${H}u; j = j + 256u) {
    let w = tablas[(base + j) >> 1u];
    let v = bitcast<f32>(select(w << 16u, w & 0xffff0000u, ((base + j) & 1u) == 1u));
    xsig[j] = v;
    fila[j] = fila[j] + v;
  }
}`;

/**
 * Lo mismo que `generaTramas`, con el muestreo en la tarjeta y `lote` tramas por
 * envío: una sola lectura por lote. La trama en la que la voz termina, y las que
 * vengan detrás en el mismo lote, se descartan al leer.
 */
export async function generaTramasGPU(m, filas, uniforme, { max = 375, lote = 4, enTrama = null, maxContexto = 2048 } = {}) {
  const { ops, device } = m;
  const { ctx } = ops;
  const glob = m.global || (m.global = m.pila('transformer', 12, maxContexto));
  const loc = m.local || (m.local = m.pila('local_transformer', 1, NVQ + 1));
  const oculto = m.oculto || (m.oculto = m.persistente(H, 'moss.oculto'));
  if (!m.gpu) {
    const tabl = new Uint16Array(NVQ * CB * H);
    m.emb.audio.forEach((t, c) => tabl.set(t, c * CB * H));
    const sube = (data, label) => { const b = device.createBuffer({ label, size: Math.ceil(data.byteLength / 16) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC }); device.queue.writeBuffer(b, 0, data); return b; };
    m.gpu = {
      tablas: sube(tabl, 'moss.tablas'),
      slot: sube(m.filaDe(m.emb.texto, SLOT), 'moss.slot'),
      visto: m.persistente(NVQ * CB, 'moss.visto'),
      xsig: m.persistente(H, 'moss.xsig'),
      fila: m.persistente(H, 'moss.fila'),
    };
  }
  const G = m.gpu;
  const res = device.createBuffer({ label: 'moss.res', size: lote * (NVQ + 2) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const usBuf = device.createBuffer({ label: 'moss.us', size: lote * NVQ * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(G.visto, 0, new Uint32Array(NVQ * CB));
  glob.reinicia();
  const clamp = (u) => Math.min(U_MAX, Math.max(0, u));
  const sube = (arr) => { const b = ctx.alloc(arr.byteLength, 'moss.x'); device.queue.writeBuffer(b, 0, arr); return b; };
  const p = await ctx.pipeline('moss_sorteo', sorteoShader());

  const nFilas = filas.length / (NVQ + 1);
  await ops.batch('moss.prefill', async () => {
    const y = await glob.paso(sube(m.embebe(filas)), nFilas);
    await m.copia(await ops.attn.strideCopy(y, { headDim: H, heads: 1, tokens: 1, stride: H, offset: (nFilas - 1) * H }), oculto, 0, H);
  });
  ctx.recycle();

  const tramas = [];
  try {
    while (tramas.length < max) {
      const k = Math.min(lote, max - tramas.length);
      const ua = [], us = new Float32Array(lote * NVQ);
      for (let i = 0; i < k; i++) { ua.push(clamp(uniforme())); for (let c = 0; c < NVQ; c++) us[i * NVQ + c] = clamp(uniforme()); }
      device.queue.writeBuffer(usBuf, 0, us);
      await ops.batch('moss.lote', async () => {
        for (let i = 0; i < k; i++) {
          loc.reinicia();
          const h0 = await loc.paso(oculto, 1);
          await m.copia(await m.mm('cabeza_texto', h0, 2, H, 1), res, i * (NVQ + 2), 2);
          await m.copia(G.slot, G.xsig, 0, H);
          await m.copia(G.slot, G.fila, 0, H);
          for (let c = 0; c < NVQ; c++) {
            const h = await loc.paso(G.xsig, 1);
            const lg = await m.mm(`audio_lm_heads.${c}.weight`, h, CB, H, 1);
            const u = ctx.uniform([new Uint32Array([c, i, 0, 0])]);
            await ops.dispatchAny('moss_sorteo', p, [lg, G.visto, usBuf, G.tablas, G.xsig, G.fila, res, u], 1);
          }
          await m.copia(await glob.paso(G.fila, 1), oculto, 0, H);
        }
      });
      const r = new Float32Array(await readback(device, res, 0, k * (NVQ + 2) * 4));
      ctx.recycle();
      let fin = false;
      for (let i = 0; i < k; i++) {
        const ls = r[i * (NVQ + 2)], lf = r[i * (NVQ + 2) + 1];
        const mx = Math.max(ls, lf), es = Math.exp(ls - mx), ef = Math.exp(lf - mx);
        if (!(ua[i] <= Math.fround(es / (es + ef)))) { fin = true; break; }
        const trama = Array.from(r.subarray(i * (NVQ + 2) + 2, (i + 1) * (NVQ + 2)), (x) => x | 0);
        tramas.push(trama);
        enTrama?.(tramas.length, trama);
      }
      if (fin) break;
    }
  } finally {
    res.destroy(); usBuf.destroy();
  }
  return tramas;
}
