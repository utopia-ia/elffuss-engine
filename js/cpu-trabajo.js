// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Engine en CPU · qué es un trabajo y cómo se lo reparten los hilos.
// Lo comparten el jefe (cpu.js) y los ayudantes (cpu-hilo.js).
//
// Un trabajo son hasta MAX_MATRICES productos de pesos por activaciones que se
// pueden hacer a la vez (en LFM3, q, k, v y la puerta de la atención, o gate y
// up del FFN: la misma entrada, otras matrices). Ir juntas ahorra barreras: cada
// trabajo despierta a los hilos y espera al último.
//
// El bloque de control (`ctl`, Int32 en la memoria compartida):
//   0 generación (el jefe la sube en cada trabajo) · 1 trozos terminados
//   2 matrices del trabajo (0 = cerrar) · 9 siguiente trozo libre, con la
//   generación delante · 10 trozos en total · 12 algún hilo falló
//   16 + 18·m, la matriz m: kernel, W, act, out, k, filas, T, su primer trozo y
//   cuántas filas tiene cada trozo suyo; y para la atención (que reparte cabezas,
//   no filas) además: tmp, tmpPaso, pasado, grupo, D, paso, ventana y la escala.
//   SwiGLU reparte bloques de 32 (W = g, act = u, out = activación cuantizada) y
//   la norma reparte filas (W = x, act = peso, out = salida, k = activación; en 9
//   el ancho y en 16 el eps).
//
// Los trozos se COGEN, no se asignan: cada hilo, el jefe incluido, pide el
// siguiente libre hasta que no quedan. Con un reparto fijo el jefe esperaba a
// que CADA hilo hiciera el suyo: un hilo que tardaba en despertar (o un núcleo
// lento de un móvil) paraba a todos, y uno que no despertaba lo colgaba entero.

export const MAX_MATRICES = 4;
export const PASO_DESC = 18;
export const CTL = 16 + PASO_DESC * MAX_MATRICES;
// Kernels repartibles: los productos por filas y la atención por cabezas.
export const KERNEL = { matvec_q4_0: 1, matvec_q8_0: 2, matmul_q4_0: 3, matmul_q8_0: 4, atencion: 5, swiglu: 6, norma: 7, matmul_bf16: 8 };

// ctl[9] = generación (19 bits) · índice del trozo (12 bits). Un hilo que se
// despertó tarde lleva la generación de un trabajo ya acabado: su petición no
// casa y no coge nada, aunque el jefe ya haya escrito los argumentos del siguiente.
const BITS = 12, INDICE = (1 << BITS) - 1, GEN = 0x7FFFF;
export const marca = (gen) => (gen & GEN) << BITS;

// El siguiente trozo libre del trabajo `gen`, o -1 si no quedan o el trabajo es otro.
export function pideTrozo(c, gen) {
  for (;;) {
    const v = Atomics.load(c, 9);
    if ((v >>> BITS) !== (gen & GEN) || (v & INDICE) >= c[10]) return -1;
    if (Atomics.compareExchange(c, 9, v, v + 1) === v) return v & INDICE;
  }
}

// La escala de la atención es un float: va con sus bits en el bloque de enteros.
export const escribeFloat = (c, i, v) => { new Float32Array(c.buffer, c.byteOffset + i * 4, 1)[0] = v; };
const leeFloat = (c, i) => new Float32Array(c.buffer, c.byteOffset + i * 4, 1)[0];

// El trozo `i` del trabajo, hecho por el hilo `yo` (0 el jefe): `yo` elige su
// sitio de puntuaciones en la atención.
export function ejecutaTrozo(e, c, i, yo) {
  let m = c[2] - 1;
  while (m > 0 && i < c[16 + PASO_DESC * m + 7]) m--;
  const b = 16 + PASO_DESC * m;
  const W = c[b + 1] >>> 0, a = c[b + 2] >>> 0, out = c[b + 3] >>> 0, k = c[b + 4], filas = c[b + 5], T = c[b + 6];
  const r0 = (i - c[b + 7]) * c[b + 8], r1 = Math.min(filas, r0 + c[b + 8]);
  switch (c[b]) {
    case 1: e.matvec_q4_0(W, a, out, k, r0, r1); break;
    case 2: e.matvec_q8_0(W, a, out, k, r0, r1); break;
    case 3: e.matmul_q4_0(W, a, out, k, filas, T, r0, r1); break;
    case 4: e.matmul_q8_0(W, a, out, k, filas, T, r0, r1); break;
    case 8: e.matmul_bf16(W, a, out, k, filas, T, r0, r1); break;
    case 6: e.swiglu_cuantiza(W, a, out, r0, r1); break;
    case 7: e.norma_cuantiza(W, a, out, c[b + 4] >>> 0, c[b + 9], r0, r1, leeFloat(c, b + 16)); break;
    case 5: e.atencion_cabezas(W, a, out, c[b + 4] >>> 0, (c[b + 9] >>> 0) + yo * c[b + 10] * 4, r0, r1,
      c[b + 11], T, filas, c[b + 12], c[b + 13], c[b + 14], c[b + 15], leeFloat(c, b + 16)); break;
    default: throw new Error(`kernel desconocido ${c[b]}`);
  }
}

// ¿Es esta foto de prefijo de ESTA sesión de CPU? La forma (bytes por fila,
// capas de caché y de convolución) la da `formaInstantanea()`; la usan el estado
// y el proxy de la página, que tiene que contestar sin preguntar al Worker.
export function cuadraInstantanea(forma, d, ctx) {
  if (!d || d.bpf !== forma.bpf || d.kvF16 !== false || !(d.past > 0) || d.past > ctx || !Array.isArray(d.partes)) return false;
  if (d.partes.length !== forma.conv.length + 2 * forma.kv.length) return false;
  const por = new Map(d.partes.map((p) => [p.n, p.data?.byteLength]));
  return forma.conv.every((i) => por.get(`c${i}`) === forma.convBytes) &&
    forma.kv.every((i) => por.get(`k${i}`) === d.past * forma.bpf && por.get(`v${i}`) === d.past * forma.bpf);
}
