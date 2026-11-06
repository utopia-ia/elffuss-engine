// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Engine en CPU · la sesión vista desde la página.
//
// Expone lo mismo que la sesión de la tarjeta (forward, recycle, state con sus
// instantáneas) y por dentro habla con cpu-jefe.js. Las operaciones del estado
// que provider.js espera síncronas (posición, fotos) se contestan con un ESPEJO
// llevado aquí y se mandan al Worker en orden: el Worker procesa los mensajes uno
// detrás de otro, así que nunca ve una foto antes del forward que la precede.

export async function creaSesionCpuRemota(ranged, { contextLen, hilos = 0, onProgress = () => {} } = {}) {
  const w = new Worker(new URL('./cpu-jefe.js', import.meta.url), { type: 'module' });
  const esperando = new Map();
  let seq = 0, alListo = null, resumen = '';

  w.onmessage = async ({ data: d }) => {
    if (d.op === 'pide') {
      const buf = await ranged.slice(d.o, d.n);
      w.postMessage({ op: 'trozo', id: d.id, buf }, [buf]);
    } else if (d.op === 'progreso') onProgress(d.h, d.t);
    else if (d.op === 'perfil') console.log('[engine] PERFIL CPU ' + JSON.stringify(d.datos));
    else if (d.op === 'listo') { resumen = d.resumen; alListo?.ok(d); }
    else if (d.op === 'logits') { const p = esperando.get(d.id); esperando.delete(d.id); p?.ok(d.logits); }
    else if (d.op === 'error') {
      const err = new Error(d.msg);
      if (d.id && esperando.has(d.id)) { const p = esperando.get(d.id); esperando.delete(d.id); p.mal(err); }
      else alListo?.mal(err);
    }
  };
  w.onerror = (ev) => {
    const err = new Error(ev.message || 'el Worker de CPU falló');
    alListo?.mal(err);
    for (const p of esperando.values()) p.mal(err);
    esperando.clear();
  };

  await new Promise((ok, mal) => {
    alListo = { ok, mal };
    w.postMessage({ op: 'carga', size: ranged.size, ctx: contextLen, hilos, perfil: globalThis.__ELFFUSS_CPU_PERFIL === true });
  });

  const manda = (metodo, ...args) => w.postMessage({ op: 'estado', metodo, args });
  const ranuras = new Map();          // nombre → posición de su foto (-1: no hay)

  const state = {
    ctxLen: contextLen,
    nPast: 0,
    kvF16: false,
    reset() { state.nPast = 0; ranuras.set('turno', -1); manda('reset'); },
    instantanea(nombre = 'turno') { ranuras.set(nombre, state.nPast); manda('instantanea', nombre); return state.nPast; },
    restaura(nombre = 'turno') {
      const p = ranuras.get(nombre);
      if (p === undefined || p < 0) return false;
      state.nPast = p; manda('restaura', nombre);
      return true;
    },
    tieneInstantanea(nombre = 'turno') { return (ranuras.get(nombre) ?? -1) >= 0; },
    pasadoInstantanea(nombre = 'turno') { return ranuras.get(nombre) ?? -1; },
    olvidaInstantanea(nombre) {
      if (nombre === undefined) for (const k of ranuras.keys()) ranuras.set(k, -1);
      else ranuras.set(nombre, -1);
      manda('olvidaInstantanea', ...(nombre === undefined ? [] : [nombre]));
    },
    exportaInstantanea() { return null; },
    importaInstantanea() { return false; },
    summary() { return resumen; },
    destroy() { w.terminate(); },
  };

  return {
    state,
    cpu: true,
    recycle() {},
    async forward(ids, { signal } = {}) {
      if (state.nPast + ids.length > contextLen) {
        throw new Error(`la caché se llena: ${state.nPast + ids.length} tokens sobre ${contextLen} de contexto`);
      }
      const id = ++seq;
      const logits = await new Promise((ok, mal) => {
        esperando.set(id, { ok, mal });
        w.postMessage({ op: 'forward', id, ids: Array.from(ids) });
      });
      state.nPast += ids.length;
      if (signal?.aborted) throw Object.assign(new Error('parado'), { name: 'AbortError' });
      return { logits, logitsTokens: 1 };
    },
    cierra() { w.terminate(); },
    destroy() { w.terminate(); },
  };
}
