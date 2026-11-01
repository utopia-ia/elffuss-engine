// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · muestrear trayendo solo los candidatos (`__ELFFUSS_TOPK_GPU`).
// ─────────────────────────────────────────────────────────────────────────────
// El bucle de generación leía TODOS los logits en cada token: un buffer de
// lectura nuevo, tres ámbitos de error, el mapeo, y después `sample()` recorría
// el vocabulario entero en JavaScript. Aquí el dispositivo elige los
// candidatos (`kernels.topk.wgsl.js`), se lee un buffer pequeño y `sample()` corre sobre
// ese puñado (`topk.js`), con el MISMO token como resultado.
//
// Va DESPUÉS del forward, en su propio envío —el mismo número de envíos que la
// lectura de siempre, que también llevaba el suyo— y antes de reciclar: el
// buffer de logits sigue vivo hasta que se ha leído. Todo lo que puede salir mal
// acaba en el camino de siempre: configuración sin tope (null del plan), un
// aviso de los kernels (NaN, algo que no cupo) o una respuesta incoherente.
//
// Sin ámbitos de error por token: solo la PRIMERA vez se envuelve en uno, que es
// cuando un binding mal puesto se descubre, y al crear los bind groups de un
// buffer de logits nuevo (pocas veces por sesión). Si eso falla, o el shader no
// compila en este navegador, se apaga para el resto de la sesión y se avisa por
// consola: el camino de siempre sigue funcionando. Y como un envío inválido sin
// ámbito no lanza —deja en el buffer de lectura lo de una llamada anterior—,
// cada llamada lleva su número de serie y la cabecera tiene que devolverlo: si
// no, también se apaga, y ese token sale por el camino de siempre.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.
import { guard } from './device.js';
import {
  topkTrozosShader, topkFinalShader, TOPK_POR_HILO, TOPK_OUTCAP, TOPK_CAP, TOPK_P_MAX, TOPK_N_MAX,
  TOPK_CABECERA, topkBytesSalida, topkBytesTrozos, topkTrozos,
} from './kernels.topk.wgsl.js';
import { planTopK, muestraCandidatos, leeSalidaTopK } from './topk.js';

const ahora = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

// Cuentas de TODAS las instancias de este módulo. Las lee una prueba que solo
// ve el proveedor desde fuera: sin ellas, «sale el mismo texto con el
// interruptor puesto» no distingue «los kernels eligieron bien» de «se cayó al
// camino de siempre», que da el mismo texto sin probar nada.
export const estadisticasTopK = { gpu: 0, respaldo: 0, motivos: {} };

export function createTopKGPU(device, ctx) {
  const bytesSalida = topkBytesSalida();
  const bufIds = device.createBuffer({ label: 'topk.ids', size: TOPK_P_MAX * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const bufPar = device.createBuffer({ label: 'topk.params', size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bufOut = device.createBuffer({ label: 'topk.out', size: bytesSalida,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  // Dos buffers de lectura que se turnan: mientras uno se mapea el otro queda
  // libre para el token siguiente. Se crean una vez, no uno por token. `ocupada`
  // marca el que está en uso desde que se elige hasta que se desmapea: el estado
  // de mapeo solo cambia al llamar a mapAsync, y entre la copia y esa llamada
  // puede haber una espera (la del ámbito de error de la primera vez).
  const lectura = [null, null], ocupada = [false, false];
  let turno = 0;
  let serie = 0;
  let bufCand = null, trozosCand = 0;
  let p1 = null, p2 = null;
  // Bind groups por buffer de logits: el pozo entrega pocos distintos (con
  // `__ELFFUSS_PLAN`, el mismo en cada token de generación).
  let grupos = new Map();
  let validado = false, roto = false;
  const params = new Uint32Array(8);
  const stats = { gpu: 0, respaldo: 0, motivos: {} };
  const respaldo = (motivo) => {
    for (const s of [stats, estadisticasTopK]) {
      s.respaldo++;
      s.motivos[motivo] = (s.motivos[motivo] || 0) + 1;
    }
    return null;
  };
  const apaga = (motivo, e) => {
    roto = true;
    console.warn(`[engine] top-k en el dispositivo desactivado (${motivo}): ${e?.message || e}`);
    return respaldo(motivo);
  };
  const nuevaLectura = () => device.createBuffer({
    label: 'topk.lectura', size: bytesSalida, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  // Los dos bind groups de un buffer de logits, creados con ámbito de error: un
  // bind group inválido no lanza, y el envío que lo use no se ejecutaría.
  async function gruposDe(logits) {
    let bg = grupos.get(logits);
    if (bg) return bg;
    if (grupos.size >= 8) grupos = new Map();
    const ent = (bufs) => bufs.map((buffer, binding) => ({ binding, resource: { buffer } }));
    bg = await guard(device, 'topk/bind', async () => [
      device.createBindGroup({ layout: p1.getBindGroupLayout(0), entries: ent([logits, bufCand, bufPar]) }),
      device.createBindGroup({ layout: p2.getBindGroupLayout(0), entries: ent([logits, bufCand, bufIds, bufOut, bufPar]) }),
    ]);
    grupos.set(logits, bg);
    return bg;
  }

  // Los candidatos de UN vector de logits: `base` es el primer logit (en
  // elementos) del token que se muestrea. Devuelve null si hay que leer entero.
  async function candidatos(logits, base, vocab, plan) {
    if (roto) return respaldo('apagado');
    const G = topkTrozos(vocab);
    if (vocab >= TOPK_N_MAX || G > 65535) return respaldo('vocabulario');
    if (!p1) {
      try {
        p1 = await ctx.pipeline(`topk_trozos_${TOPK_POR_HILO}_${TOPK_OUTCAP}`, () => topkTrozosShader());
        p2 = await ctx.pipeline(`topk_final_${TOPK_OUTCAP}_${TOPK_CAP}_${TOPK_P_MAX}_${TOPK_CABECERA}`, () => topkFinalShader());
      } catch (e) { p1 = null; return apaga('compilación', e); }
    }
    if (!bufCand || trozosCand < G) {
      bufCand?.destroy();
      bufCand = device.createBuffer({ label: 'topk.trozos', size: topkBytesTrozos(G), usage: GPUBufferUsage.STORAGE });
      trozosCand = G;
      grupos = new Map();
    }
    let bg;
    try { bg = await gruposDe(logits); } catch (e) { return apaga('validación', e); }

    let k = turno; turno ^= 1;
    if (ocupada[k]) k ^= 1;
    let staging, temporal = false;
    if (ocupada[k]) { staging = nuevaLectura(); temporal = true; }       // las dos en uso
    else {
      if (!lectura[k] || lectura[k].mapState !== 'unmapped') lectura[k] = nuevaLectura();
      staging = lectura[k];
      ocupada[k] = true;
    }
    serie = (serie + 1) >>> 0 || 1;                    // nunca 0: un buffer recién creado lee 0
    const esta = serie;

    const graba = () => {
      params[0] = vocab; params[1] = base; params[2] = plan.M; params[3] = G; params[4] = plan.S.length;
      params[5] = esta;
      device.queue.writeBuffer(bufPar, 0, params);
      if (plan.S.length) device.queue.writeBuffer(bufIds, 0, plan.S);
      const enc = device.createCommandEncoder({ label: 'topk' });
      const pass = enc.beginComputePass({ label: 'topk' });
      pass.setPipeline(p1); pass.setBindGroup(0, bg[0]); pass.dispatchWorkgroups(G);
      pass.setPipeline(p2); pass.setBindGroup(0, bg[1]); pass.dispatchWorkgroups(1);
      pass.end();
      enc.copyBufferToBuffer(bufOut, 0, staging, 0, bytesSalida);
      device.queue.submit([enc.finish()]);
    };
    try {
      if (!validado) {
        try { await guard(device, 'topk', async () => graba()); }
        catch (e) { grupos = new Map(); return apaga('validación', e); }
        validado = true;
      } else {
        graba();
      }
      await staging.mapAsync(GPUMapMode.READ, 0, bytesSalida);
      const u = new Uint32Array(staging.getMappedRange(0, bytesSalida).slice(0));
      staging.unmap();
      const r = leeSalidaTopK(u, plan, G, esta);
      if (r.motivo === 'serie') return apaga('serie', new Error('lo leído no es de esta llamada: el envío no se ejecutó'));
      return r.motivo ? respaldo(r.motivo) : r;
    } catch (e) {
      // Tras un fallo no se sabe en qué estado queda el buffer de lectura: fuera.
      if (!temporal) { try { staging.destroy(); } catch { /* — */ } lectura[k] = null; }
      throw e;
    } finally {
      if (temporal) staging.destroy();
      else ocupada[k] = false;
    }
  }

  return {
    stats,
    candidatos,
    // El paso completo: plan, candidatos, `sample()` sobre el puñado. Devuelve
    // { id, msLectura, msMuestreo } o null, y entonces quien llama lee el
    // vector entero como siempre. `opts` es lo que se le pasaría a `sample()`
    // SIN `previos`, que llega en dos mitades (el prompt y lo ya emitido).
    async muestrea(logits, base, vocab, opts, previosA, previosB) {
      const t0 = ahora();
      const plan = planTopK(vocab, opts, previosA, previosB);
      if (!plan) return respaldo('muestreo sin tope');
      const c = await candidatos(logits, base, vocab, plan);
      if (!c) return null;
      const t1 = ahora();
      const id = muestraCandidatos(plan, c.ids, c.bits, c.bitsS, opts);
      if (id === null) return respaldo('incoherente');
      stats.gpu++; estadisticasTopK.gpu++;
      return { id, msLectura: t1 - t0, msMuestreo: ahora() - t1 };
    },
    destroy() {
      for (const b of [bufIds, bufPar, bufOut, bufCand, ...lectura]) { try { b?.destroy(); } catch { /* — */ } }
    },
  };
}
