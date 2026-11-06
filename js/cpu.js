// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Engine · el mismo modelo en CPU, sin WebGPU.
//
// Para los navegadores y dispositivos sin tarjeta utilizable. El reparto:
//   · WebAssembly con SIMD (cpu/kernels.c → cpu-kernels.js) para lo que pesa:
//     los productos de los pesos Q4_0/Q8_0 por las activaciones y la atención.
//     Si el navegador tiene relaxed SIMD se usa ese módulo; si no, el estricto.
//   · JavaScript para lo que no pesa: normas, RoPE, la convolución corta, las
//     puertas y las sumas, con las MISMAS fórmulas que los shaders de la tarjeta
//     (kernels.nn.wgsl.js, kernels.ssm.wgsl.js, model.js).
//
// Arquitecturas: lfm3 (convolución corta + atención, la de los modelos por
// defecto). Lo que no sepa calcular lo dice al cargar, no a mitad de una frase.
//
// La sesión expone lo mismo que la de la tarjeta (forward, recycle, state con
// sus instantáneas) para que provider.js no distinga entre las dos: plantilla,
// tokenizador, muestreo, reutilización del prefijo y freno son los mismos.
// La caché KV vive en la memoria del módulo, en f32.

import { WASM_ESTRICTO, WASM_RELAJADO, WASM_ESTRICTO_HILOS, WASM_RELAJADO_HILOS } from './cpu-kernels.js';
import { CTL, PASO_DESC, KERNEL, MAX_MATRICES, marca, pideTrozo, ejecutaTrozo, escribeFloat } from './cpu-trabajo.js';

const QK = 32;
const BQ4 = 18, BQ8 = 34, BACT = 40;          // bytes por bloque (ver kernels.c)
// Fichas por pasada del prompt: acota la memoria de los intermedios.
const LOTE_CPU = 64;

const deBase64 = (s) => {
  const bin = atob(s); const u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return u;
};

// Las filas del embedding, a f32: Q6_K (el del GGUF del teléfono), BF16 (el del
// Q8_0), Q8_0, Q4_0 y F16. Redondeo a redondeo como llama.cpp: d·escala redondeado
// a f32 y luego por el cuanto.
const fr = Math.fround;
function deF16(bits) {
  const s = (bits & 0x8000) ? -1 : 1, e = (bits >> 10) & 0x1f, m = bits & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}
function filaAF32(tipo, bytes, nBloques) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (tipo === 'F16') { const o = new Float32Array(nBloques); for (let i = 0; i < nBloques; i++) o[i] = deF16(dv.getUint16(i * 2, true)); return o; }
  if (tipo === 'BF16') { const o = new Uint32Array(nBloques); for (let i = 0; i < nBloques; i++) o[i] = dv.getUint16(i * 2, true) << 16; return new Float32Array(o.buffer); }
  const porBloque = tipo === 'Q6_K' ? 256 : 32;
  const o = new Float32Array(nBloques * porBloque);
  for (let b = 0; b < nBloques; b++) {
    if (tipo === 'Q8_0') {
      const base = b * 34, d = fr(deF16(dv.getUint16(base, true)));
      for (let i = 0; i < 32; i++) o[b * 32 + i] = fr(dv.getInt8(base + 2 + i) * d);
    } else if (tipo === 'Q4_0') {
      const base = b * 18, d = fr(deF16(dv.getUint16(base, true)));
      for (let j = 0; j < 16; j++) {
        const by = dv.getUint8(base + 2 + j);
        o[b * 32 + j] = fr(((by & 0xf) - 8) * d);
        o[b * 32 + j + 16] = fr(((by >> 4) - 8) * d);
      }
    } else {
      // Q6_K, 210 bytes: ql[128] · qh[64] · escalas[16] · d. Los cuatro pesos que
      // comparten un byte de qh van a 0, 32, 64 y 96 del destino.
      const base = b * 210, d = fr(deF16(dv.getUint16(base + 208, true)));
      for (let n = 0; n < 2; n++) {
        const ql = base + n * 64, qh = base + 128 + n * 32, sc = base + 192 + n * 8, ob = b * 256 + n * 128;
        for (let l = 0; l < 32; l++) {
          const is = l >> 4;
          const lo0 = dv.getUint8(ql + l), lo32 = dv.getUint8(ql + 32 + l), h = dv.getUint8(qh + l);
          o[ob + l] = fr(fr(d * dv.getInt8(sc + is)) * (((lo0 & 0xf) | ((h & 3) << 4)) - 32));
          o[ob + 32 + l] = fr(fr(d * dv.getInt8(sc + is + 2)) * (((lo32 & 0xf) | (((h >> 2) & 3) << 4)) - 32));
          o[ob + 64 + l] = fr(fr(d * dv.getInt8(sc + is + 4)) * (((lo0 >> 4) | (((h >> 4) & 3) << 4)) - 32));
          o[ob + 96 + l] = fr(fr(d * dv.getInt8(sc + is + 6)) * (((lo32 >> 4) | (((h >> 6) & 3) << 4)) - 32));
        }
      }
    }
  }
  return o;
}

// f32 → f16 (redondeo al par), para las escalas del Q8_0 que se construye al cargar.
const _f = new Float32Array(1), _u = new Uint32Array(_f.buffer);
function aF16(v) {
  _f[0] = v; const x = _u[0];
  const s = (x >>> 16) & 0x8000, e = (x >>> 23) & 0xFF, m = x & 0x7FFFFF;
  if (e === 0xFF) return s | 0x7C00 | (m ? 0x200 : 0);
  let ee = e - 127 + 15;
  if (ee >= 0x1F) return s | 0x7C00;
  if (ee <= 0) {
    if (ee < -10) return s;
    const mm = (m | 0x800000) >>> (1 - ee);
    return s | ((mm + 0x1000 + ((mm >>> 13) & 1) - 1) >>> 13);
  }
  let r = s | (ee << 10) | (m >>> 13);
  const resto = m & 0x1FFF;
  if (resto > 0x1000 || (resto === 0x1000 && (r & 1))) r++;
  return r;
}

const WASI = { proc_exit() {}, fd_write() { return 0; }, fd_close() { return 0; }, fd_seek() { return 0; } };

// Hilos que tiene sentido usar: la mitad de los núcleos que declara el navegador
// (los otros para la página y el sistema), como mucho 8, y solo si hay memoria
// compartida (página aislada: COOP/COEP) y Workers. Medido con llama.cpp en wasm:
// con todos los núcleos el prompt va algo más rápido y la generación se hunde.
export function hilosDisponibles(pedidos) {
  const puede = typeof SharedArrayBuffer !== 'undefined' && typeof Worker !== 'undefined' &&
    (globalThis.crossOriginIsolated !== false);
  if (!puede) return 1;
  const nucleos = globalThis.navigator?.hardwareConcurrency || 4;
  return Math.max(1, Math.min(8, pedidos || Math.floor(nucleos / 2)));
}

export async function cargaKernels({ hilos = 1 } = {}) {
  const usaHilos = hilos > 1;
  const relajado = deBase64(usaHilos ? WASM_RELAJADO_HILOS : WASM_RELAJADO);
  const conRelaxed = WebAssembly.validate(relajado);
  const bytes = conRelaxed ? relajado : deBase64(usaHilos ? WASM_ESTRICTO_HILOS : WASM_ESTRICTO);
  let memoria = null;
  const imp = { env: { emscripten_notify_memory_growth() {} }, wasi_snapshot_preview1: WASI };
  if (usaHilos) {
    memoria = new WebAssembly.Memory({ initial: 256, maximum: 65536, shared: true });
    imp.env.memory = memoria;
  }
  const { instance } = await WebAssembly.instantiate(bytes, imp);
  const e = instance.exports;
  e._initialize?.();
  e.prepara();
  let pool = null;
  if (usaHilos) pool = await creaPool(e, memoria, bytes, hilos);
  // Con hilos la memoria es la compartida que se creó aquí (el módulo la importa
  // y no la exporta); sin hilos, la del propio módulo.
  const buffer = () => (memoria || e.memory).buffer;
  return { e, relajado: conRelaxed, pool, hilos: pool ? pool.n : 1, buffer };
}

// Trozos por hilo en cada trabajo: los suficientes para que un núcleo rápido
// haga más que uno lento (los móviles mezclan los dos), sin que pedirlos cueste.
const TROZOS_POR_HILO = 4;
// Segundos sin que se termine ningún trozo antes de dar el trabajo por perdido.
// Un trozo es miles de veces más corto: esto solo salta si un hilo murió con uno cogido.
const ESPERA_MAX = 15;

// Los ayudantes: n-1 Workers más el propio jefe, que también coge trozos.
async function creaPool(e, memoria, bytes, n) {
  const ctl = e.reserva(CTL * 4);
  const ayudantes = [];
  const listos = [];
  for (let yo = 1; yo < n; yo++) {
    const pila = e.reserva(256 * 1024) + 256 * 1024;
    const w = new Worker(new URL('./cpu-hilo.js', import.meta.url), { type: 'module' });
    listos.push(new Promise((ok, mal) => { w.onmessage = () => ok(); w.onerror = (ev) => mal(new Error(ev.message || 'hilo')); }));
    w.postMessage({ bytes, memoria, pila, ctl, yo, n });
    ayudantes.push(w);
  }
  await Promise.all(listos);
  let gen = 0;
  return {
    n,
    // Ejecuta un trabajo de hasta MAX_MATRICES productos repartidos por filas:
    // prepara el bloque de control, despierta a los ayudantes, coge trozos como uno
    // más y espera a los que tienen otros. `lista` = [{kernel, W, act, out, k, filas, T}].
    ejecuta(lista) {
      const c = new Int32Array(memoria.buffer, ctl, CTL);
      gen = (gen + 1) | 0;
      // Primero la marca nueva: desde aquí, un hilo que se despertó tarde para el
      // trabajo anterior ya no puede coger trozos con argumentos a medio escribir.
      Atomics.store(c, 9, marca(gen));
      let total = 0;
      lista.forEach((t, m) => {
        const b = 16 + PASO_DESC * m;
        const porTrozo = t.porTrozo || Math.max(4, Math.ceil(t.filas / (n * TROZOS_POR_HILO) / 4) * 4);
        c[b] = t.kernel; c[b + 1] = t.W; c[b + 2] = t.act; c[b + 3] = t.out; c[b + 4] = t.k; c[b + 5] = t.filas; c[b + 6] = t.T;
        c[b + 7] = total; c[b + 8] = porTrozo;
        if (t.kernel === KERNEL.atencion) {
          c[b + 9] = t.tmp; c[b + 10] = t.tmpPaso; c[b + 11] = t.pasado; c[b + 12] = t.grupo; c[b + 13] = t.D;
          c[b + 14] = t.paso; c[b + 15] = t.ventana; escribeFloat(c, b + 16, t.escala);
        } else if (t.kernel === KERNEL.norma) {
          c[b + 9] = t.n; escribeFloat(c, b + 16, t.eps);
        }
        total += Math.ceil(t.filas / porTrozo);
      });
      c[1] = 0; c[2] = lista.length; c[10] = total; c[12] = 0;
      // Un solo trozo lo hace el jefe sin despertar a nadie: despertar a los
      // ayudantes cuesta más que el trozo (la norma o el SwiGLU de un token). La
      // generación no sube, así que los ayudantes siguen durmiendo.
      if (total <= 1) {
        if (total) ejecutaTrozo(e, c, 0, 0);
        return;
      }
      Atomics.store(c, 0, gen);
      Atomics.notify(c, 0);
      for (let i; (i = pideTrozo(c, gen)) >= 0;) { ejecutaTrozo(e, c, i, 0); Atomics.add(c, 1, 1); }
      let visto = Atomics.load(c, 1), quieto = 0;
      while (visto < total) {
        const r = Atomics.wait(c, 1, visto, 1000);
        const ahora = Atomics.load(c, 1);
        if (r === 'timed-out' && ahora === visto && ++quieto >= ESPERA_MAX) {
          throw new Error(`un hilo de la CPU no termina su parte (${total - ahora} de ${total} trozos)`);
        }
        if (ahora !== visto) quieto = 0;
        visto = ahora;
      }
      if (c[12]) throw new Error('un hilo de la CPU falló a mitad de un cálculo');
    },
    cierra() {
      const c = new Int32Array(memoria.buffer, ctl, CTL);
      gen = (gen + 1) | 0;
      Atomics.store(c, 9, marca(gen));
      c[2] = 0; c[10] = 0;
      Atomics.store(c, 0, gen); Atomics.notify(c, 0);
      for (const w of ayudantes) w.terminate();
    },
  };
}

// ── El modelo ──────────────────────────────────────────────────────────────
// `ranged` = { size, slice(offset, length) }, `info` = parseGGUF, `hp` = readHParams.
export async function cargaModeloCpu(ranged, info, hp, onProgress = () => {}, { hilos = 1 } = {}) {
  if (hp.arch !== 'lfm3') throw new Error(`la CPU todavía no calcula la arquitectura «${hp.arch}»`);
  const K = await cargaKernels({ hilos });
  const { e } = K;
  const mem = K.buffer;
  const porNombre = new Map(info.tensors.map((t) => [t.name, t]));
  const tipo = (t) => String(t.typeName ?? t.type);
  const dims = (t) => (t.dims ?? t.ne).map(Number);
  const inicio = (t) => Number(t.start ?? t.offset);

  // Lo que se sube a la memoria del módulo: los pesos grandes, tal cual (los
  // bloques Q4_0/Q8_0 del GGUF son los mismos structs que lee kernels.c).
  const grandes = info.tensors.filter((t) => /^(Q4_0|Q8_0)$/.test(tipo(t)));
  const total = grandes.reduce((n, t) => n + t.nBytes, 0);
  const pesos = new Map();      // nombre → { p, tipo, filas, k }
  let hecho = 0;
  for (const t of grandes) {
    const [k, filas = 1] = dims(t);
    const p = e.reserva(t.nBytes);
    if (!p) throw new Error('no hay memoria para los pesos en la CPU');
    const bytes = new Uint8Array(await ranged.slice(inicio(t), t.nBytes));
    new Uint8Array(mem(), p, t.nBytes).set(bytes);
    pesos.set(t.name, { p, tipo: tipo(t), filas, k });
    hecho += t.nBytes;
    onProgress(hecho, total);
  }
  // Los pequeños (normas, convolución, rope_freqs) en JS, como f32.
  const f32 = new Map();
  for (const t of info.tensors) {
    if (tipo(t) !== 'F32') continue;
    f32.set(t.name, new Float32Array((await ranged.slice(inicio(t), t.nBytes)).slice(0)));
  }
  // El embedding: en el GGUF del teléfono es Q6_K y hace también de cabeza de
  // salida (pesos atados). La fila de cada ficha se descuantiza al vuelo; para la
  // cabeza se construye UNA vez una copia Q8_0 en la memoria del módulo.
  const E = hp.embd;
  const te = porNombre.get('token_embd.weight');
  const teTipo = tipo(te), [teK, vocab] = dims(te);
  if (teK !== E) throw new Error('token_embd no tiene el ancho del modelo');
  const teBytes = new Uint8Array(await ranged.slice(inicio(te), te.nBytes));
  const bytesFila = te.nBytes / vocab;
  const bloquesFila = { Q6_K: E / 256, Q4_0: E / 32, Q8_0: E / 32, F16: E, BF16: E, F32: E }[teTipo];
  if (!bloquesFila) throw new Error(`token_embd en «${teTipo}»: la CPU no lo sabe leer`);
  const filaEmbd = (id) => {
    const b = teBytes.subarray(id * bytesFila, (id + 1) * bytesFila);
    if (teTipo === 'F32') return new Float32Array(b.slice().buffer);
    return filaAF32(teTipo, b, bloquesFila);
  };
  const salida = porNombre.get('output.weight');
  let cabeza;
  if (salida && /^(Q4_0|Q8_0)$/.test(tipo(salida))) cabeza = pesos.get('output.weight');
  else {
    if (salida) throw new Error(`output.weight en «${tipo(salida)}»: la CPU no lo sabe leer`);
    const nb = E / QK, p = e.reserva(vocab * nb * BQ8);
    const dv = new DataView(mem());
    for (let v = 0; v < vocab; v++) {
      const x = filaEmbd(v);
      for (let b = 0; b < nb; b++) {
        let a = 0;
        for (let i = 0; i < QK; i++) a = Math.max(a, Math.abs(x[b * QK + i]));
        const d = a / 127, id = d ? 1 / d : 0;
        const o = p + (v * nb + b) * BQ8;
        dv.setUint16(o, aF16(d), true);
        for (let i = 0; i < QK; i++) dv.setInt8(o + 2 + i, Math.round(x[b * QK + i] * id));
      }
    }
    cabeza = { p, tipo: 'Q8_0', filas: vocab, k: E };
  }
  return { K, e, hp, pesos, f32, filaEmbd, cabeza, vocab, kernels: (K.relajado ? 'relaxed SIMD' : 'SIMD128') + ` · ${K.hilos} hilo${K.hilos > 1 ? 's' : ''}` };
}

// ── La sesión ──────────────────────────────────────────────────────────────
export function creaSesionCpu(modelo, { contextLen }) {
  const { e, hp, pesos, f32, filaEmbd, cabeza, vocab } = modelo;
  const E = hp.embd, H = hp.heads, KVH = hp.kvHeads, D = hp.headDim, F = hp.ffnLength, CK = hp.convKernel;
  const eps = hp.eps;
  const mem = modelo.K.buffer;
  const paso = KVH * D;

  // Memoria de trabajo en el módulo, de una vez y para el lote más largo. Todo el
  // camino de una capa vive aquí dentro: la entrada, las salidas de las matrices,
  // la convolución, la atención y el residuo. El JS solo dice qué va con qué; ya
  // no copia filas de un lado a otro ni crea arrays por paso.
  const L = LOTE_CPU;
  // La salida más ancha es la de un trabajo con varias matrices: gate + up del FFN,
  // o q + k + v + la puerta de la atención.
  const maxK = Math.max(E, F, H * D);
  const maxFilas = Math.max(3 * E, 2 * F, H * D + 2 * paso + H, E);
  const f4 = (n) => { const p = e.reserva(n * 4); if (!p) throw new Error('no hay memoria de trabajo en la CPU'); return p; };
  const pX = f4(L * E), pRes = f4(L * E), pIn = f4(L * maxK), pY = f4(L * Math.max(E, H * D));
  const pAct = e.reserva(L * (maxK / QK) * BACT);
  const pOut = f4(L * maxFilas);
  const pVent = f4((CK - 1 + L) * E);
  const pLog = f4(vocab);
  // Un sitio de puntuaciones por hilo: la atención va por cabezas a cualquier hilo.
  const nHilos = modelo.K.hilos;
  const pTmp = f4(nHilos * contextLen);
  // Los pesos pequeños (normas, convolución) también dentro, para las funciones de C.
  const pf = new Map();
  for (const [n, w] of f32) { const p = f4(w.length); new Float32Array(mem(), p, w.length).set(w); pf.set(n, p); }
  const pPeso = (n) => { const p = pf.get(n); if (!p) throw new Error(`falta «${n}»`); return p; };
  const vista = (p, n) => new Float32Array(mem(), p, n);
  // Caché KV: por capa de atención, K y V de [ctx][KVH·D] en f32.
  const kv = new Map();
  for (const l of hp.layers) {
    if (l.kind !== 'attn') continue;
    const k = e.reserva(contextLen * paso * 4), v = e.reserva(contextLen * paso * 4);
    if (!k || !v) throw new Error(`no cabe la caché de ${contextLen} fichas en la CPU`);
    kv.set(l.index, { k, v });
  }
  // La ventana de la convolución corta: los CK-1 valores anteriores de B·x.
  const conv = new Map();
  for (const l of hp.layers) if (l.kind === 'conv') { const p = f4((CK - 1) * E); vista(p, (CK - 1) * E).fill(0); conv.set(l.index, p); }

  // Perfil por pasos (`__ELFFUSS_CPU_PERFIL = true` en el Worker): milisegundos y
  // veces de cada paso. Lo que no es «kernel» lo hace el jefe SOLO: es tiempo en
  // que los ayudantes esperan. Va aparte para el prompt (lotes) y para la
  // generación (de uno en uno).
  const perfilVacio = () => ({ ms: {}, veces: {}, tokens: 0, pasadas: 0 });
  const perfil = globalThis.__ELFFUSS_CPU_PERFIL === true ? { modo: 'gen', prompt: perfilVacio(), gen: perfilVacio() } : null;
  const mide = perfil
    ? (paso, f) => {
      const a = performance.now(); const r = f(); const p = perfil[perfil.modo];
      p.ms[paso] = (p.ms[paso] || 0) + performance.now() - a; p.veces[paso] = (p.veces[paso] || 0) + 1;
      return r;
    }
    : (paso, f) => f();

  // Trabajos para los hilos (si los hay; si no, se hacen aquí mismo).
  const pool = modelo.K.pool;
  const kernelDe = (w, T) => w.tipo === 'Q4_0' ? (T === 1 ? KERNEL.matvec_q4_0 : KERNEL.matmul_q4_0) : (T === 1 ? KERNEL.matvec_q8_0 : KERNEL.matmul_q8_0);
  function corre(lista, paso) {
    if (perfil) return mide(paso, () => corre0(lista));
    return corre0(lista);
  }
  function corre0(lista) {
    if (pool) return pool.ejecuta(lista);
    for (const t of lista) {
      if (t.kernel === KERNEL.matvec_q4_0) e.matvec_q4_0(t.W, t.act, t.out, t.k, 0, t.filas);
      else if (t.kernel === KERNEL.matvec_q8_0) e.matvec_q8_0(t.W, t.act, t.out, t.k, 0, t.filas);
      else if (t.kernel === KERNEL.matmul_q4_0) e.matmul_q4_0(t.W, t.act, t.out, t.k, t.filas, t.T, 0, t.filas);
      else if (t.kernel === KERNEL.matmul_q8_0) e.matmul_q8_0(t.W, t.act, t.out, t.k, t.filas, t.T, 0, t.filas);
      else if (t.kernel === KERNEL.swiglu) e.swiglu_cuantiza(t.W, t.act, t.out, 0, t.filas);
      else if (t.kernel === KERNEL.norma) e.norma_cuantiza(t.W, t.act, t.out, t.k, t.n, 0, t.filas, t.eps);
      else e.atencion_cabezas(t.W, t.act, t.out, t.k, t.tmp, 0, t.filas, t.pasado, t.T, t.filas, t.grupo, t.D, t.paso, t.ventana, t.escala);
    }
  }
  // [W·entrada por cada nombre] para T filas que ya están en el módulo (`pEnt`):
  // se cuantizan UNA vez para todas las matrices, que van en un solo trabajo (una
  // barrera en vez de una por matriz). Devuelve dónde quedó cada salida, dentro de
  // pOut: valen hasta el siguiente mm.
  // Con `pEnt` = null la entrada ya está cuantizada en pAct (la dejó la norma o el SwiGLU).
  function mm(nombres, pEnt, T) {
    const ws = nombres.map((n) => { const w = pesos.get(n); if (!w) throw new Error(`falta el peso «${n}»`); return w; });
    const k = ws[0].k;
    if (ws.some((w) => w.k !== k) || ws.length > MAX_MATRICES) throw new Error('matrices que no comparten entrada');
    if (pEnt !== null) mide('cuantiza', () => e.cuantiza_filas(pEnt, pAct, k, T));
    let o = pOut;
    const lista = ws.map((w) => { const t = { kernel: kernelDe(w, T), W: w.p, act: pAct, out: o, k, filas: w.filas, T }; o += T * w.filas * 4; return t; });
    corre(lista, 'kernel');
    return lista.map((t) => t.out);
  }
  const norma = (pEnt, nombre, pSal, ancho, filas) => mide('norma', () => e.norma_filas(pEnt, pPeso(nombre), pSal, ancho, filas, eps));
  // La norma de la entrada de una capa y su cuantización, repartidas por filas
  // (en el prompt; con un token es una fila). Deja la activación en pAct. El eps
  // viaja como float en el bloque de control: lo es en el GGUF, así que no cambia.
  const epsF32 = Math.fround(eps) === eps;
  const normaCuantiza = (pEnt, nombre, T) => {
    if (!epsF32) { norma(pEnt, nombre, pIn, E, T); mide('cuantiza', () => e.cuantiza_filas(pIn, pAct, E, T)); return; }
    corre([{ kernel: KERNEL.norma, W: pEnt, act: pPeso(nombre), out: pIn, k: pAct, filas: T, T, n: E, eps,
      porTrozo: Math.max(1, Math.ceil(T / (nHilos * 2))) }], 'norma');
  };

  // RoPE NEOX (pares i, i+mitad), con los factores de `rope_freqs` si la capa los usa.
  // Los cosenos y senos dependen de la posición y de la configuración de la capa,
  // no de la cabeza: se calculan una vez por lote y configuración (las mismas
  // cuentas, en el mismo orden) y los reutilizan todas las cabezas de q y de k y
  // todas las capas con esa configuración. Antes eran 20 veces por capa.
  const mitadRope = hp.ropeDims / 2;
  let tablasRope = new Map(), posTablas = -1, filasTablas = 0;
  function tablaRope(filas, pos0, base, factores) {
    if (pos0 !== posTablas || filas !== filasTablas) { tablasRope = new Map(); posTablas = pos0; filasTablas = filas; }
    const clave = base + (factores ? '|f' : '');
    let t = tablasRope.get(clave);
    if (t) return t;
    const escalaTheta = Math.pow(base, -2 / hp.ropeDims);
    const cos = new Float64Array(filas * mitadRope), sin = new Float64Array(filas * mitadRope);
    for (let r = 0; r < filas; r++) {
      const pos = pos0 + r;
      for (let i = 0; i < mitadRope; i++) {
        let th = pos * Math.pow(escalaTheta, i);
        if (factores) th /= factores[i];
        cos[r * mitadRope + i] = Math.cos(th); sin[r * mitadRope + i] = Math.sin(th);
      }
    }
    t = { cos, sin };
    tablasRope.set(clave, t);
    return t;
  }
  const rope = (...a) => mide('rope', () => rope0(...a));
  function rope0(v, filas, cabezas, pos0, base, factores) {
    const mitad = mitadRope, { cos, sin } = tablaRope(filas, pos0, base, factores);
    for (let r = 0; r < filas; r++) {
      for (let h = 0; h < cabezas; h++) {
        const o = (r * cabezas + h) * D;
        for (let i = 0; i < mitad; i++) {
          const c = cos[r * mitad + i], s = sin[r * mitad + i];
          const x0 = v[o + i], x1 = v[o + i + mitad];
          v[o + i] = x0 * c - x1 * s;
          v[o + i + mitad] = x0 * s + x1 * c;
        }
      }
    }
  }

  // Cada bloque lee el residuo de pX y deja el nuevo en pX.
  function bloqueConv(i, T) {
    normaCuantiza(pX, `blk.${i}.attn_norm.weight`, T);
    const [pBcx] = mm([`blk.${i}.shortconv.in_proj.weight`], null, T);
    mide('conv', () => e.conv_corta(pBcx, conv.get(i), pPeso(`blk.${i}.shortconv.conv.weight`), pY, pVent, E, CK, T));
    const [pO] = mm([`blk.${i}.shortconv.out_proj.weight`], pY, T);
    ffn(i, pO, T);
  }

  function bloqueAtencion(l, T) {
    const i = l.index, pasado = sesion.state.nPast;
    normaCuantiza(pX, `blk.${i}.attn_norm.weight`, T);
    // q, k, v y la puerta salen de la misma entrada: un trabajo.
    const nombres = ['attn_q', 'attn_k', 'attn_v', ...(hp.headGate ? ['attn_gate'] : [])].map((n) => `blk.${i}.${n}.weight`);
    const [pQ, pK, pV, pG] = mm(nombres, null, T);
    if (hp.qkNorm) {
      norma(pQ, `blk.${i}.attn_q_norm.weight`, pQ, D, T * H);
      norma(pK, `blk.${i}.attn_k_norm.weight`, pK, D, T * KVH);
    }
    const factores = l.ropeFactores ? peso('rope_freqs.weight') : null;
    rope(vista(pQ, T * H * D), T, H, pasado, l.ropeBase, factores);
    rope(vista(pK, T * KVH * D), T, KVH, pasado, l.ropeBase, factores);
    const c = kv.get(i);
    mide('copia', () => {
      const m = new Float32Array(mem());
      m.copyWithin((c.k >>> 2) + pasado * paso, pK >>> 2, (pK >>> 2) + T * paso);
      m.copyWithin((c.v >>> 2) + pasado * paso, pV >>> 2, (pV >>> 2) + T * paso);
    });
    // Una cabeza por trozo: con hilos, las 16 cabezas se reparten entre todos.
    corre([{ kernel: KERNEL.atencion, W: pQ, act: c.k, out: c.v, k: pY, filas: H, T, porTrozo: 1, tmp: pTmp, tmpPaso: contextLen,
      pasado, grupo: H / KVH, D, paso, ventana: l.ventana || 0, escala: 1 / Math.sqrt(D) }], 'atencion');
    // La puerta: un valor por cabeza, la exponencial una vez por cabeza.
    if (hp.headGate) mide('puerta', () => e.puerta_cabezas(pY, pG, T * H, D));
    const [pO] = mm([`blk.${i}.attn_output.weight`], pY, T);
    ffn(i, pO, T);
  }

  // El FFN con su residuo: pX + capa → pRes; pRes + FFN → pX. `pCapa` está en pOut.
  function ffn(i, pCapa, T) {
    if (hp.postNorms) norma(pCapa, `blk.${i}.post_attention_norm.weight`, pCapa, E, T);
    mide('suma', () => e.suma_f32(pX, pCapa, pRes, T * E));
    normaCuantiza(pRes, `blk.${i}.${hp.ffnNorm}.weight`, T);
    const [pG, pU] = mm([`blk.${i}.ffn_gate.weight`, `blk.${i}.ffn_up.weight`], null, T);
    // SwiGLU y la cuantización de la entrada de ffn_down, en un paso repartido por
    // bloques de 32: la activación queda en pAct antes de que ffn_down escriba en pOut.
    // Con un token va en un solo trozo: repartirlo costaba más de lo que ahorraba.
    const bloques = T * F / QK;
    const porTrozo = T === 1 ? bloques : Math.max(1, Math.ceil(bloques / (nHilos * TROZOS_POR_HILO)));
    corre([{ kernel: KERNEL.swiglu, W: pG, act: pU, out: pAct, filas: bloques, T, porTrozo }], 'swiglu');
    const [pAb] = mm([`blk.${i}.ffn_down.weight`], null, T);
    if (hp.postNorms) norma(pAb, `blk.${i}.post_ffw_norm.weight`, pAb, E, T);
    mide('suma', () => e.suma_f32(pRes, pAb, pX, T * E));
  }
  const peso = (n) => { const w = f32.get(n); if (!w) throw new Error(`falta «${n}»`); return w; };

  // Instantáneas: lo mismo que state.js en la tarjeta. Se copia el estado de la
  // convolución (lo que no se puede rebobinar); la caché KV no, porque la máscara
  // causal descarta lo posterior y volver atrás es devolver `nPast`.
  const ranuras = new Map();
  const ranura = (n) => { let r = ranuras.get(n); if (!r) { r = { past: -1, conv: null }; ranuras.set(n, r); } return r; };

  const state = {
    ctxLen: contextLen,
    nPast: 0,
    kvF16: false,
    reset() { state.nPast = 0; for (const p of conv.values()) vista(p, (CK - 1) * E).fill(0); ranura('turno').past = -1; },
    advance(n) {
      if (state.nPast + n > contextLen) throw new Error(`la caché se llena: ${state.nPast + n} tokens sobre ${contextLen} de contexto`);
      state.nPast += n;
    },
    instantanea(nombre = 'turno') {
      const r = ranura(nombre);
      r.conv = new Map([...conv].map(([i, p]) => [i, vista(p, (CK - 1) * E).slice()]));
      r.past = state.nPast;
      return r.past;
    },
    restaura(nombre = 'turno') {
      const r = ranuras.get(nombre);
      if (!r || !r.conv || r.past < 0) return false;
      for (const [i, s] of r.conv) vista(conv.get(i), (CK - 1) * E).set(s);
      state.nPast = r.past;
      return true;
    },
    tieneInstantanea(nombre = 'turno') { const r = ranuras.get(nombre); return !!r && r.past >= 0; },
    pasadoInstantanea(nombre = 'turno') { const r = ranuras.get(nombre); return r ? r.past : -1; },
    olvidaInstantanea(nombre) {
      if (nombre === undefined) { for (const r of ranuras.values()) r.past = -1; return; }
      const r = ranuras.get(nombre); if (r) r.past = -1;
    },
    // El prefijo en disco es de la tarjeta (otro formato): en CPU no se guarda.
    exportaInstantanea() { return null; },
    importaInstantanea() { return false; },
    summary() {
      const mb = (kv.size * 2 * contextLen * paso * 4) / 1048576;
      return `CPU (${modelo.kernels}) · ${kv.size} capas KV (${mb.toFixed(0)} MiB a ${contextLen} de contexto, f32) · ${conv.size} capas de convolución`;
    },
    destroy() {},
  };

  const sesion = {
    state,
    recycle() {},
    // Mismo contrato que la de la tarjeta: procesa `ids` desde nPast, avanza y
    // devuelve los logits del ÚLTIMO token (Float32Array de `vocab`).
    // El perfil acumulado desde la última vez, y a cero.
    tomaPerfil() {
      if (!perfil) return null;
      const r = { prompt: perfil.prompt, gen: perfil.gen };
      perfil.prompt = perfilVacio(); perfil.gen = perfilVacio();
      return r;
    },
    async forward(ids, opts = {}) {
      const { signal, onProgress } = opts;
      if (perfil) { perfil.modo = ids.length > 1 ? 'prompt' : 'gen'; perfil[perfil.modo].tokens += ids.length; perfil[perfil.modo].pasadas++; }
      const t0 = perfil ? performance.now() : 0;
      try { return await forward0(ids, signal, onProgress); }
      finally { if (perfil) { const p = perfil[perfil.modo]; p.ms.total = (p.ms.total || 0) + performance.now() - t0; } }
    },
  };
  async function forward0(ids, signal, onProgress) {
    let T = 0, hechos = 0;
    for (let ini = 0; ini < ids.length; ini += L) {
      if (signal?.aborted) throw Object.assign(new Error('parado'), { name: 'AbortError' });
      const trozo = ids.slice(ini, ini + L);
      T = trozo.length;
      mide('embd', () => { for (let t = 0; t < T; t++) vista(pX + t * E * 4, E).set(filaEmbd(trozo[t])); });
      for (const l of hp.layers) l.kind === 'conv' ? bloqueConv(l.index, T) : bloqueAtencion(l, T);
      state.advance(T);
      hechos += T;
      onProgress?.(hechos);
      // Ceder el hilo entre lotes: sin esto el navegador no pinta hasta el final.
      if (ini + L < ids.length) await new Promise((r) => setTimeout(r, 0));
    }
    norma(pX + (T - 1) * E * 4, 'output_norm.weight', pIn, E, 1);
    e.cuantiza_filas(pIn, pAct, E, 1);
    corre([{ kernel: kernelDe(cabeza, 1), W: cabeza.p, act: pAct, out: pLog, k: E, filas: cabeza.filas, T: 1 }], 'kernel');
    return { logits: vista(pLog, vocab).slice(), logitsTokens: 1 };
  }
  return sesion;
}
