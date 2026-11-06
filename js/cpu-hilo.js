// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Engine en CPU · un hilo ayudante.
//
// Instancia el MISMO módulo de kernels sobre la memoria COMPARTIDA del jefe, con
// su propia pila, y espera trabajo en el bloque de control (ver cpu-trabajo.js).
// Despierto, coge trozos hasta que no quedan, cuenta cada uno y avisa. No hay
// postMessage por trabajo: un token son cientos de trabajos y el mensaje cuesta
// más que el trozo.

import { CTL, pideTrozo, ejecutaTrozo } from './cpu-trabajo.js';

self.onmessage = async ({ data }) => {
  const { bytes, memoria, pila, ctl, yo } = data;
  const imp = {
    env: { memory: memoria, emscripten_notify_memory_growth() {} },
    wasi_snapshot_preview1: { proc_exit() {}, fd_write() { return 0; }, fd_close() { return 0; }, fd_seek() { return 0; } },
  };
  const { instance } = await WebAssembly.instantiate(bytes, imp);
  const e = instance.exports;
  e.pon_pila(pila);
  self.postMessage('listo');
  let gen = 0;
  for (;;) {
    let c = new Int32Array(memoria.buffer, ctl, CTL);
    Atomics.wait(c, 0, gen);
    c = new Int32Array(memoria.buffer, ctl, CTL);
    gen = Atomics.load(c, 0);
    if (c[2] === 0) return;                       // 0 = cerrar
    for (let i; (i = pideTrozo(c, gen)) >= 0;) {
      // Un trozo que falla se cuenta igual, con la bandera puesta: el jefe lanza
      // el error en vez de esperar un trozo que no va a llegar.
      try { ejecutaTrozo(e, c, i, yo); } catch (err) { Atomics.store(c, 12, 1); console.error('[engine] hilo de CPU:', err); }
      Atomics.add(c, 1, 1);
      Atomics.notify(c, 1);
    }
  }
};
