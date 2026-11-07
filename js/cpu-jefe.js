// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Engine en CPU · el Worker que lleva el modelo y la sesión.
//
// Va en un Worker por dos razones: el forward en CPU tardaría en devolver el
// hilo de la página (se congelaría al escribir), y los hilos ayudantes esperan
// con Atomics.wait, que en el hilo de la página no está permitido.
//
// Mensajes de la página: carga {size, ctx, hilos} · trozo {id, buf} (respuesta
// a una petición de bytes) · forward {id, ids} · estado {metodo, args}.
// Mensajes a la página: pide {id, o, n} (bytes del modelo: los lee la página,
// que es quien tiene abierto el almacén) · progreso {h, t} · listo {resumen,
// vocab} · logits {id, logits} · error {id?, msg}.

import { parseGGUF } from './gguf.js';
import { readHParams } from './state.js';
import { cargaModeloCpu, creaSesionCpu, hilosDisponibles } from './cpu.js';

let sesion = null;
let genPerfil = 0;
const pendientes = new Map();
let seq = 0;
const pideBytes = (o, n) => new Promise((ok) => {
  const id = ++seq;
  pendientes.set(id, ok);
  self.postMessage({ op: 'pide', id, o, n });
});

// Los mensajes se atienden de uno en uno y en orden. Un forward de más de un lote
// cede el hilo entre lotes, y sin la cola una orden del estado (una foto, volver
// a una foto) que llegara en ese hueco se colaría delante del forward que la
// precede. Las respuestas de bytes van aparte: la carga las está esperando.
let cola = Promise.resolve();
self.onmessage = ({ data: d }) => {
  if (d.op === 'trozo') {
    const ok = pendientes.get(d.id);
    pendientes.delete(d.id);
    ok?.(d.buf);
    return;
  }
  cola = cola.then(() => atiende(d));
};

async function atiende(d) {
  try {
    if (d.op === 'carga') {
      if (d.perfil) globalThis.__ELFFUSS_CPU_PERFIL = true;
      const ranged = { size: d.size, slice: pideBytes };
      const info = await parseGGUF(ranged);
      const hp = readHParams(info.meta, info.tensors);
      const modelo = await cargaModeloCpu(ranged, info, hp, (h, t) => self.postMessage({ op: 'progreso', h, t }),
        { hilos: hilosDisponibles(d.hilos) });
      sesion = creaSesionCpu(modelo, { contextLen: d.ctx });
      self.postMessage({ op: 'listo', resumen: sesion.state.summary(), vocab: modelo.vocab, forma: sesion.state.formaInstantanea() });
    } else if (d.op === 'forward') {
      // El avance de un prompt largo, lote a lote: en la CPU es la espera más
      // larga de todas, y la página la enseña.
      const r = await sesion.forward(d.ids, d.ids.length > 1 ? { onProgress: (h) => self.postMessage({ op: 'avance', id: d.id, h }) } : {});
      self.postMessage({ op: 'logits', id: d.id, logits: r.logits }, [r.logits.buffer]);
      // Con perfil: el del prompt en cuanto acaba, el de la generación cada 32 tokens.
      if (globalThis.__ELFFUSS_CPU_PERFIL && (d.ids.length > 1 || ++genPerfil % 32 === 0)) {
        self.postMessage({ op: 'perfil', datos: sesion.tomaPerfil() });
      }
    } else if (d.op === 'exporta') {
      const x = sesion.state.exportaInstantanea(d.nombre);
      self.postMessage({ op: 'exportado', id: d.id, datos: x }, x ? x.partes.map((p) => p.data) : []);
    } else if (d.op === 'estado') {
      sesion.state[d.metodo](...(d.args || []));
    }
  } catch (e) {
    self.postMessage({ op: 'error', id: d.id, msg: String(e?.message || e) });
  }
}
