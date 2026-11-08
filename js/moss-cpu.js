// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · MOSS-TTS-Nano en la CPU (WebAssembly), para quien no tiene tarjeta.
// ─────────────────────────────────────────────────────────────────────────────
// La misma cuenta que moss.js, sin WebGPU. Las multiplicaciones van por
// `matmul_bf16` (cpu/kernels.c): los pesos se quedan en bf16 en la memoria del
// módulo, la mitad de bytes que el f32 que mueve el camino ONNX, y al escribir
// trama a trama lo que manda es mover pesos. La atención es `atencion_cabezas`;
// LayerNorm, sesgos, GELU y RoPE son vectores de 768 y van en JS. Con hilos, las
// multiplicaciones se reparten por filas entre los Workers de cpu-hilo.js (hace
// falta aislamiento entre orígenes y correr dentro de un Worker: el pool espera
// con Atomics.wait).
//
// Pesos: la misma fuente que moss.js (`fuenteOnnx` con el mapa, o `fuenteGGUF`).
import { cargaKernels } from './cpu.js';
import { KERNEL } from './cpu-trabajo.js';
import { fuenteGGUF, sorteaCodigo } from './moss.js';

const H = 768, NH = 12, HD = 64, NVQ = 16, CB = 1024, FFN = 3072, EPS = 1e-5;
const SLOT = 9, END = 7, PAD = 1024, U_MAX = 0.99999994;
const f = Math.fround;
const U32 = new Uint32Array(1), F32 = new Float32Array(U32.buffer);
const bf16 = (u) => { U32[0] = u << 16; return F32[0]; };
// inv_freq en f32, como la referencia: 1 / 10000^(2i/64).
const INV = Float32Array.from({ length: HD / 2 }, (_, i) => f(1 / f(Math.pow(10000, f((2 * i) / HD)))));

/**
 * Carga MOSS-TTS-Nano en la memoria de los kernels de CPU.
 * @param {{ bf16(name: string): Promise<Uint16Array> }} fuente
 * @param {{ hilos?: number, maxContexto?: number, maxFilas?: number }} [opts]
 */
export async function cargaMossCpu(fuente, { hilos = 1, maxContexto = 1024, maxFilas = 512 } = {}) {
  if (typeof fuente.bf16 !== 'function') fuente = await fuenteGGUF(fuente);
  const K = await cargaKernels({ hilos });
  const { e } = K;
  const mem = () => K.buffer();
  const reserva = (bytes) => { const p = e.reserva(Math.ceil(bytes / 16) * 16 + 16); return (p + 15) & ~15; };
  const f32 = (p, n) => new Float32Array(mem(), p, n);

  const W = new Map(), V = new Map();
  const mat = async (name) => {
    const u = await fuente.bf16(name);
    const p = reserva(u.byteLength);
    new Uint16Array(mem(), p, u.length).set(u);
    W.set(name, p);
  };
  const vec = async (name) => {
    const v = Float32Array.from(await fuente.bf16(name), bf16);
    const p = reserva(v.byteLength);
    new Float32Array(mem(), p, v.length).set(v);
    V.set(name, p);
  };
  for (const [pre, n] of [['transformer', 12], ['local_transformer', 1]]) {
    for (let i = 0; i < n; i++) {
      const p = `${pre}.h.${i}`;
      for (const v of ['ln_1.weight', 'ln_1.bias', 'ln_2.weight', 'ln_2.bias', 'attn.c_attn.bias', 'attn.c_proj.bias', 'mlp.fc_in.bias', 'mlp.fc_out.bias']) await vec(`${p}.${v}`);
      for (const m of ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.fc_in.weight', 'mlp.fc_out.weight']) await mat(`${p}.${m}`);
    }
    await vec(`${pre}.ln_f.weight`); await vec(`${pre}.ln_f.bias`);
  }
  for (let c = 0; c < NVQ; c++) await mat(`audio_lm_heads.${c}.weight`);
  const texto = await fuente.bf16('text_lm_head.weight');
  const cabezaTexto = [SLOT, END].map((id) => Float32Array.from(texto.subarray(id * H, (id + 1) * H), bf16));
  const emb = { texto: await fuente.bf16('transformer.wte.weight'), audio: [] };
  for (let c = 0; c < NVQ; c++) emb.audio.push(await fuente.bf16(`audio_embeddings.${c}.weight`));

  // Sitio de trabajo en la memoria del módulo (punteros; las vistas se rehacen al
  // usarlas porque la memoria puede crecer y cambiar de ArrayBuffer).
  const T = maxFilas;
  const P = {
    x: reserva(T * H * 4), h: reserva(T * H * 4), qkv: reserva(T * 3 * H * 4), q: reserva(T * H * 4),
    att: reserva(T * H * 4), o: reserva(T * FFN * 4), ff: reserva(T * FFN * 4), lg: reserva(CB * 4),
    tmp: reserva(maxContexto * 4 * Math.max(1, K.hilos)),
  };
  const cache = (n, ctx) => Array.from({ length: n }, () => ({ k: reserva(ctx * H * 4), v: reserva(ctx * H * 4) }));
  const pilas = { transformer: { n: 12, c: cache(12, maxContexto), max: maxContexto }, local_transformer: { n: 1, c: cache(1, NVQ + 1), max: NVQ + 1 } };

  const mm = (name, act, out, k, filas, t) => {
    if (K.pool) K.pool.ejecuta([{ kernel: KERNEL.matmul_bf16, W: W.get(name), act, out, k, filas, T: t }]);
    else e.matmul_bf16(W.get(name), act, out, k, filas, t, 0, filas);
  };
  const sesgo = (p, name, t, n, gelu = false) => e.sesgo_filas(p, V.get(name), n, 0, t, gelu ? 1 : 0);
  const layerNorm = (src, dst, pre, t) => e.capa_norma(src, V.get(`${pre}.weight`), V.get(`${pre}.bias`), dst, H, t, EPS);
  // RoPE de pares adyacentes (GPT-J) sobre una fila de 768 en su sitio.
  const rope = (a, o, pos) => {
    for (let hh = 0; hh < NH; hh++) for (let i = 0; i < HD / 2; i++) {
      const fr = f(pos * INV[i]), c = f(Math.cos(fr)), s = f(Math.sin(fr));
      const j = o + hh * HD + 2 * i, ev = a[j], od = a[j + 1];
      a[j] = f(f(ev * c) - f(od * s)); a[j + 1] = f(f(od * c) + f(ev * s));
    }
  };

  /** Un paso de una pila sobre `t` filas que ya están en P.x; deja la salida (tras ln_f) en P.h. */
  const reloj = {};
  const mide = (k, fn) => { if (!globalThis.__MOSS_RELOJ) return fn(); const t0 = performance.now(); const r = fn(); reloj[k] = (reloj[k] || 0) + performance.now() - t0; return r; };
  function paso(pre, pasado, t) {
    const pila = pilas[pre];
    if (pasado + t > pila.max) throw new Error(`moss-cpu: ${pre} sin sitio`);
    for (let i = 0; i < pila.n; i++) {
      const p = `${pre}.h.${i}`, { k, v } = pila.c[i];
      mide('norma', () => layerNorm(P.x, P.h, `${p}.ln_1`, t));
      mide('matmul', () => mm(`${p}.attn.c_attn.weight`, P.h, P.qkv, H, 3 * H, t));
      sesgo(P.qkv, `${p}.attn.c_attn.bias`, t, 3 * H);
      const qkv = f32(P.qkv, t * 3 * H), q = f32(P.q, t * H), kc = f32(k, pila.max * H), vc = f32(v, pila.max * H);
      mide('rope', () => { for (let r = 0; r < t; r++) {
        const pos = pasado + r;
        q.set(qkv.subarray(r * 3 * H, r * 3 * H + H), r * H); rope(q, r * H, pos);
        kc.set(qkv.subarray(r * 3 * H + H, r * 3 * H + 2 * H), pos * H); rope(kc, pos * H, pos);
        vc.set(qkv.subarray(r * 3 * H + 2 * H, r * 3 * H + 3 * H), pos * H);
      } });
      mide('atencion', () => e.atencion_cabezas(P.q, k, v, P.att, P.tmp, 0, NH, pasado, t, NH, 1, HD, H, 0, f(1 / 8)));
      mide('matmul', () => mm(`${p}.attn.c_proj.weight`, P.att, P.o, H, H, t));
      sesgo(P.o, `${p}.attn.c_proj.bias`, t, H);
      e.suma_f32(P.x, P.o, P.x, t * H);
      layerNorm(P.x, P.h, `${p}.ln_2`, t);
      mide('matmul', () => mm(`${p}.mlp.fc_in.weight`, P.h, P.ff, H, FFN, t));
      mide('gelu', () => sesgo(P.ff, `${p}.mlp.fc_in.bias`, t, FFN, true));
      mide('matmul', () => mm(`${p}.mlp.fc_out.weight`, P.ff, P.o, FFN, H, t));
      sesgo(P.o, `${p}.mlp.fc_out.bias`, t, H);
      e.suma_f32(P.x, P.o, P.x, t * H);
    }
    layerNorm(P.x, P.h, `${pre}.ln_f`, t);
  }

  function embebe(filas, dst) {
    const t = filas.length / (NVQ + 1);
    for (let r = 0; r < t; r++) {
      const fila = filas.subarray(r * (NVQ + 1), (r + 1) * (NVQ + 1)), o = r * H, tx = fila[0] * H;
      for (let j = 0; j < H; j++) dst[o + j] = bf16(emb.texto[tx + j]);
      for (let c = 0; c < NVQ; c++) {
        const id = fila[1 + c];
        if (id === PAD) continue;
        const a = emb.audio[c], b = id * H;
        for (let j = 0; j < H; j++) dst[o + j] = f(dst[o + j] + bf16(a[b + j]));
      }
    }
  }
  const fila = (tabla, id, dst) => { for (let j = 0; j < H; j++) dst[j] = bf16(tabla[id * H + j]); };
  const punto = (a, b) => { let s = 0; for (let j = 0; j < H; j++) s = f(s + f(a[j] * b[j])); return s; };

  /** Como `generaTramas` de moss.js, en la CPU. */
  async function genera(filas, uniforme, { max = 375, enTrama = null } = {}) {
    const clamp = (u) => Math.min(U_MAX, Math.max(0, u));
    const nFilas = filas.length / (NVQ + 1);
    if (nFilas > T) throw new Error(`moss-cpu: ${nFilas} filas, el máximo es ${T}`);
    embebe(filas, f32(P.x, nFilas * H));
    paso('transformer', 0, nFilas);
    let pasado = nFilas;
    const oculto = Float32Array.from(f32(P.h + (nFilas - 1) * H * 4, H));
    const visto = Array.from({ length: NVQ }, () => new Uint8Array(CB));
    const tramas = [];
    for (let n = 0; n < max; n++) {
      const ua = clamp(uniforme());
      const us = Array.from({ length: NVQ }, () => clamp(uniforme()));
      f32(P.x, H).set(oculto);
      paso('local_transformer', 0, 1);
      const h0 = f32(P.h, H);
      const ls = punto(h0, cabezaTexto[0]), lf = punto(h0, cabezaTexto[1]);
      const mx = Math.max(ls, lf), es = Math.exp(ls - mx), ef = Math.exp(lf - mx);
      if (!(ua <= f(es / (es + ef)))) break;
      const trama = [];
      fila(emb.texto, SLOT, f32(P.x, H));
      for (let c = 0; c < NVQ; c++) {
        paso('local_transformer', 1 + c, 1);
        mm(`audio_lm_heads.${c}.weight`, P.h, P.lg, H, CB, 1);
        const code = sorteaCodigo(f32(P.lg, CB), visto[c], us[c]);
        trama.push(code);
        fila(emb.audio[c], code, f32(P.x, H));
      }
      trama.forEach((code, c) => { visto[c][code] = 1; });
      tramas.push(trama);
      enTrama?.(tramas.length, trama);
      const filaG = new Int32Array(NVQ + 1).fill(PAD);
      filaG[0] = SLOT; trama.forEach((code, c) => { filaG[1 + c] = code; });
      embebe(filaG, f32(P.x, H));
      paso('transformer', pasado, 1);
      pasado += 1;
      oculto.set(f32(P.h, H));
    }
    return tramas;
  }

  return { genera, hilos: K.hilos, relajado: K.relajado, reloj };
}
