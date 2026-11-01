// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · el modelo: carga de pesos y forward de qwen35.
// ─────────────────────────────────────────────────────────────────────────────
// Aquí se encadenan las piezas que ya están verificadas una a una. Esa
// verificación por piezas NO garantiza que el conjunto funcione: puede fallar el
// orden, el layout que espera la siguiente operación, o un residual sumado en el
// sitio equivocado. Por eso el forward está escrito para poder pararse en
// cualquier capa y comparar contra el volcado de un forward real.
//
// Dos tipos de bloque, y el reparto sale del propio fichero:
//   · SSM (18 de 24 en el 0.8B, 48 de 65 en el 27B): qkv fusionado → conv causal
//     → silu → q/k/v → delta net con estado recurrente → norm con puerta.
//   · Atención (el resto): Q y puerta INTERCALADOS en el mismo tensor, QK-norm
//     antes de RoPE, RoPE parcial, GQA, y la puerta multiplicando la salida.
// Los dos acaban igual: residual, norm, SwiGLU, residual.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.

import { parseGGUF } from './gguf.js';
import { leeRotacion } from './hadamard.js';
import { createOps, ROPE_NEOX, ROPE_NORM } from './ops.js';
import { createFlashOps } from './ops.flash.js';
import { readHParams, createState } from './state.js';
import {
  copyIntoShader, convWindowShader, convSaveShader, siluShader, mulCabezaShader,
} from './kernels.model.wgsl.js';
import { QUANT_TYPES } from './kernels.wgsl.js';
// Q8_0/Q4_0 reordenados al subir (`__ELFFUSS_REPACK`, apagado por defecto).
import {
  reordenActivo, tipoReordenado, comoReordenado, bloquesDe, reordena,
  creaBufferR, subeReordenadoPorTrozos,
} from './repack.js';
import { createFusionOps, fusionActiva, puedeGatherNorma } from './ops.fusion.js';

const MiB = 1024 ** 2;

// Sube todos los tensores del fichero al dispositivo, uno por buffer.
// PENDIENTE consciente: un tensor por buffer exige que cada uno quepa en
// maxStorageBufferBindingSize. En escritorio el límite suele sobrar; en otros
// navegadores es mucho menor y la tabla de embeddings, de los tensores más
// grandes del modelo, puede no caber. Ahí hará falta que el matmul lea varios shards, que es
// trabajo del kernel, no de aquí. Se comprueba y se falla claro en vez de
// devolver ceros.
// El tamaño de lote del prefill. Es un `let` con gancho SOLO para poder
// medirlo: comparar dos valores exige cambiarlo en caliente, dentro de la misma
// sesión y alternando el orden. Editando el fichero entre pasadas, lo que se
// mueve la medida de una sesión a otra se mezcla con lo que se compara, y luego
// no hay forma de separar una cosa de la otra.
// 32 y no 128: medido, un lote más corto va algo MÁS rápido (menos posiciones
// enmascaradas que se calculan para tirar) y —lo que importa aquí— cada despacho
// dura menos, que es media defensa contra el cuelgue. La otra media es
// trocear el forward en varios command buffers (ver `cede`/`ctx.corta`).
let loteActual = 32;
// El reparto de la última carga. Lo consume una prueba interna y puede
// consumirlo la UI: es la espera más larga del producto.
let medidaCarga = null;
export function ultimaCarga() { return medidaCarga; }

// Cuántos tensores DEBÍAN subirse en la última carga. No es lo mismo que los del
// fichero: la cabeza MTP (`nextn_predict_layers`) no pertenece al forward y no
// se sube — en el modelo pequeño son 15 de 335. Un banco que compare contra
// `g.tensorCount` falla por construcción, que es lo que pasaba.
let subidosEsperados = 0;
export function ultimosSubidos() { return subidosEsperados; }

export function _setLote(n) { loteActual = Math.max(1, n | 0); }
export function _getLote() { return loteActual; }

// Cada cuántas capas se le da al dispositivo un respiro real para que el compositor
// del sistema pinte. Un forward del 27B son 64 capas; sin trocear, iban todas
// en un solo despacho larguísimo y la pantalla se quedaba sin pintar mientras
// tanto. Ver `ctx.corta`.
let cederCada = 2;
export function _setCeder(n) { cederCada = Math.max(1, n | 0); }
export function _getCeder() { return cederCada; }

// El respiro se cobra por RELOJ, no por capas. Lo que se persigue —que el
// compositor del sistema pinte— es un asunto de milisegundos de pared: da igual
// si en ese rato han pasado dos capas o cuarenta. Contándolo por capas se pagaba
// un frame entero cada dos capas aunque esas dos capas durasen medio frame, y
// como un rAF no se puede partir, el prefill acababa cuantizado al refresco del
// monitor: el reloj lo marcaba la pantalla y no el modelo.
// `cederCada` sigue siendo la granularidad mínima (solo se mira el reloj en esas
// capas) y esto es el límite de frecuencia. A 0 se recupera exactamente el
// comportamiento anterior, que es lo que usan los bancos para el A/B.
let cederMs = 100;
export function _setCederMs(ms) { cederMs = Math.max(0, +ms || 0); }
export function _getCederMs() { return cederMs; }

// Cada cuántas capas se CIERRA y envía el command buffer del prefill. Es otra
// cosa que `cederCada`: eso es cuándo se drena la cola y se cede un frame, esto
// es solo cuántos submits se parten. Va con gancho porque la nota de al lado
// afirmaba que trocear en cada capa es «GRATIS» y eso nadie lo había medido:
// una tanda de prefill aprovecha el ancho de banda de pesos 6,6 veces peor que
// un token suelto (medido con una prueba interna) y el prefill es justo el único
// camino que trocea. A 1 se recupera el comportamiento de siempre.
let cortaCada = 1;
export function _setCorta(n) { cortaCada = Math.max(1, n | 0); }
// Lo mismo para el DECODE (ver el bucle de capas). Con gancho para el A/B.
// En CADA capa por defecto. Medido en la misma sesión, tres rondas alternadas:
// ×1,30 en LFM3 y ×1,52 en Qwen3.5-0.8B. Un solo envío al
// final del token dejaba el dispositivo parado mientras la CPU grababa cientos
// de despachos; cortando, empieza con la primera capa. Cortar cada 2, 4 u 8
// capas queda en medio. No cambia la numérica: la cola respeta el orden.
let cortaDecode = globalThis.__ELFFUSS_CORTA_DECODE ?? 1;
export function _setCortaDecode(n) { cortaDecode = Math.max(0, n | 0); }
export function _getCorta() { return cortaCada; }
// Cede un frame al bucle de eventos: en el navegador deja que el compositor
// del sistema componga; fuera del DOM (tests) basta un turno de macrotarea.
//
// Y vuelve TAMBIÉN en cuanto se pide parar. No es un adorno: con la pestaña en
// segundo plano el navegador deja de dar frames —el rAF no llega hasta que la
// pestaña se vuelve a ver—, y un prefill esperando aquí no podría atender un
// «para» hasta entonces. Con la señal, parar no depende de que haya pantalla.
const nuevoFrame = (senal = null) => new Promise((r) => {
  if (senal?.aborted) { r(); return; }
  let id = null, hecho = false;
  const fin = () => {
    if (hecho) return;
    hecho = true;
    senal?.removeEventListener?.('abort', fin);
    if (id !== null && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(id);
    r();
  };
  senal?.addEventListener?.('abort', fin, { once: true });
  if (typeof requestAnimationFrame === 'function') id = requestAnimationFrame(() => fin());
  else setTimeout(fin, 0);
});

// El error de «se pidió parar». Lleva el NOMBRE estándar, AbortError, para que
// quien llama lo distinga de un fallo de verdad sin leer el texto: un turno
// parado no es un modelo roto, y enseñarlo como «el modelo falló» sería mentir.
export function errorParado(mensaje = 'parado a petición') {
  if (typeof DOMException === 'function') return new DOMException(mensaje, 'AbortError');
  const e = new Error(mensaje);
  e.name = 'AbortError';
  return e;
}

export async function loadModel(ranged, device, opts = {}) {
  const onProgress = opts.onProgress || (() => {});
  const g = await parseGGUF(ranged);
  const hp = readHParams(g.meta, g.tensors);
  const ops = createOps(device, { debug: opts.debug !== false, subgroupSize: opts.subgroupSize || 0 });
  // Atención fusionada (`__ELFFUSS_FLASH`): solo cierres, sin pipelines ni buffers hasta que se use.
  ops.flash = createFlashOps(device, { ctx: ops.ctx, despacha: ops.dispatchAny, subgroupSize: opts.subgroupSize || 0 });
  const limit = device.limits.maxStorageBufferBindingSize;

  // Antes de mover un solo byte: que TODOS los tipos del fichero tengan kernel.
  // Un tipo sin soporte descubierto a mitad del forward sería el modelo entero
  // ya subido y un error tres capas más abajo; aquí es una línea que dice cuál.
  const sinKernel = new Map();
  for (const t of g.tensors) {
    if (!QUANT_TYPES[t.typeName]) {
      if (!sinKernel.has(t.typeName)) sinKernel.set(t.typeName, t.name);
    }
  }
  if (sinKernel.size) {
    throw new Error('el fichero trae tipos sin kernel: ' +
      [...sinKernel].map(([k, n]) => `${k} (p.ej. «${n}»)`).join(', '));
  }
  // Las rejillas de los i-quants, subidas antes del primer forward.
  await ops.quant.prepare(g.tensors.map(t => t.typeName));

  const weights = new Map();
  let uploaded = 0;
  // LA CABEZA MTP NO SE SUBE. El fichero trae `nextn_predict_layers` bloques de
  // mas que el forward: state.js hace `nLayers = block_count - nextn` y nunca los
  // recorre. En el modelo grande son 15 tensores que se suben al dispositivo, se
  // quedan ahi y no se ejecutan NUNCA: memoria y tiempo de subida tirados en
  // cada sesion, sobre un modelo cuya carga ya es la espera mas larga del
  // producto.
  //
  // Se deriva de los METADATOS y no de un «blk.64» escrito a mano: si algun dia
  // se implementa la especulacion con esa cabeza, basta con quitar este filtro y
  // los pesos vuelven. Y se descuentan tambien de `total`, porque si no la barra
  // de progreso no llegaria nunca al 100%.
  const kNextn = Object.keys(g.meta).find((k) => k.endsWith('.nextn_predict_layers'));
  const kBloques = Object.keys(g.meta).find((k) => k.endsWith('.block_count'));
  const nextn = (kNextn && g.meta[kNextn]) || 0;
  const nBloques = (kBloques && g.meta[kBloques]) || 0;
  const saltados = nextn > 0 && nBloques > 0
    ? new Set(Array.from({ length: nextn }, (_, i) => `blk.${nBloques - nextn + i}.`))
    : new Set();
  const sobra = (t) => [...saltados].some((pre) => t.name.startsWith(pre));
  const suben = saltados.size ? g.tensors.filter((t) => !sobra(t)) : g.tensors;
  const total = suben.reduce((n, t) => n + t.nBytes, 0);

  // ── DÓNDE SE VA EL TIEMPO DE CARGA ─────────────────────────────────────────
  // Medido contra llama.cpp en la misma sesión y el mismo fichero, ellos cargan
  // x3,5 más rápido que nosotros. Es la espera más larga que paga el usuario y
  // nadie la había repartido nunca: «la carga tarda» tiene al menos tres causas
  // que desde fuera se ven idénticas —traer los bytes, subirlos al dispositivo, y el
  // número de viajes— y sin separarlas no se puede atacar ninguna.
  //
  // Se cuentan los TROZOS además del tiempo: este bucle pide un rango por
  // tensor como mínimo, y un modelo pequeño tiene cientos. Si el tiempo está en
  // la latencia de cientos de viajes en serie, el arreglo no es un kernel, es
  // dejar de esperar uno por uno.
  const ahoraC = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
  const tCarga0 = ahoraC();
  let msLeer = 0, msSubir = 0, trozos = 0;

  // ── TRAMOS: una lectura para MUCHOS tensores ────────────────────────────────
  // Esto pedía un rango por tensor. Medido en navegador limpio con el modelo
  // pequeño: un viaje por tensor, con el 99,9 % del tiempo en LEER, menos del
  // 1 % en encolar la subida y nada esperando al dispositivo. Y el coste por viaje
  // apenas dependía de su tamaño, que es la firma de la LATENCIA: no se estaba
  // llenando el enlace, se estaba esperando cientos de veces de una en una.
  //
  // En GGUF los tensores van CONTIGUOS, así que se agrupan en tramos de hasta
  // TRAMO bytes y cada tramo se trae de una sola lectura; después se reparte en
  // memoria, que sale gratis (encolar era el 0,1 %).
  //
  // Se agrupa SOLO si el tensor empieza justo donde acaba el anterior: si el
  // fichero trae relleno entre tensores, agrupar a ciegas no daría error, daría
  // pesos de otro sitio. Y un tensor más grande que TRAMO va en su propio tramo
  // y se lee a trozos de CHUNK, igual que antes.
  // Interruptor para poder medir el ANTES y el DESPUÉS en la MISMA sesión: una
  // razón sacada de comparar dos tiradas no vale, porque entre tiradas los
  // absolutos se mueven demasiado. A 1 byte cada tensor va en su propio tramo y
  // se recupera el comportamiento de una lectura por tensor.
  // El tramo por defecto, y no la mitad: barrido en la misma sesión, dos rondas
  // por candidato y orden alternado, con una prueba interna. Contra el camino
  // viejo —un viaje por tensor— agrupar da del orden de x7. Entre candidatos, de
  // la mitad al valor elegido hay una mejora pequeña que se repite en LAS DOS
  // rondas; al doble la diferencia cae dentro del ruido y pediría un buffer
  // transitorio el doble de grande para nada. Ahí la curva ya está saturada:
  // manda el enlace.
  const TRAMO = Math.max(1, (globalThis.__ELFFUSS_TRAMO | 0) || 128 * MiB);
  const CHUNK = 32 * MiB;
  const tramos = [];
  for (const t of suben) {
    if (t.nBytes > limit) {
      throw new Error(`«${t.name}» son ${(t.nBytes / MiB).toFixed(0)} MiB y el ` +
        `límite de binding es ${(limit / MiB).toFixed(0)} MiB: hace falta que el ` +
        `matmul lea varios shards`);
    }
    const ult = tramos[tramos.length - 1];
    if (ult && t.start === ult.fin && (ult.fin - ult.ini) + t.nBytes <= TRAMO) {
      ult.fin = t.start + t.nBytes; ult.ts.push(t);
    } else {
      tramos.push({ ini: t.start, fin: t.start + t.nBytes, ts: [t] });
    }
  }

  // Holgura de 4 bytes en el buffer: la descodificación de bloques de 34 lee la
  // palabra siguiente al recomponer los que no caen alineados.
  const nuevoBuf = (t) => device.createBuffer({
    label: t.name,
    size: Math.ceil((t.nBytes + 4) / 4) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // FORMATO REORDENADO (`__ELFFUSS_REPACK`): los Q8_0/Q4_0 se suben con los
  // cuantos alineados y las escalas aparte, y su descriptor pasa a decir
  // Q8_0R/Q4_0R (ver repack.js). La foto del interruptor se toma UNA vez por
  // carga. Apagado, `tipoReordenado` da null siempre y todo lo de abajo hace
  // exactamente lo de antes.
  const conReorden = reordenActivo();
  let reordenados = 0;

  for (const tr of tramos) {
    const largo = tr.fin - tr.ini;
    if (largo > TRAMO) {
      // Tensor suelto demasiado grande: camino de trozos, uno solo en el tramo.
      const t = tr.ts[0];
      const tipoR = tipoReordenado(t, conReorden);
      if (tipoR) {
        // Los trozos van cortados en bloques enteros y cada uno se escribe en
        // sus dos regiones (cuantos y escalas).
        const buf = creaBufferR(device, t, tipoR);
        await subeReordenadoPorTrozos({
          device, ranged, t, buf, tipoR, trozo: CHUNK, ahora: ahoraC,
          cuenta: (leer, subir, n) => {
            msLeer += leer; msSubir += subir; trozos++;
            uploaded += n;
            onProgress(uploaded, total, t.name);
          },
        });
        weights.set(t.name, { buf, t: comoReordenado(t, tipoR) });
        reordenados++;
        continue;
      }
      const buf = nuevoBuf(t);
      for (let off = 0; off < t.nBytes; off += CHUNK) {
        const len = Math.min(CHUNK, t.nBytes - off);
        const tL = ahoraC();
        const raw = new Uint8Array(await ranged.slice(t.start + off, len));
        const tS = ahoraC(); msLeer += tS - tL; trozos++;
        const pad = len % 4 === 0 ? raw : (() => {
          const p = new Uint8Array(Math.ceil(len / 4) * 4); p.set(raw); return p;
        })();
        device.queue.writeBuffer(buf, off, pad, 0, pad.byteLength);
        msSubir += ahoraC() - tS;
        uploaded += len;
        onProgress(uploaded, total, t.name);
      }
      weights.set(t.name, { buf, t });
      continue;
    }

    const tL = ahoraC();
    const span = new Uint8Array(await ranged.slice(tr.ini, largo));
    msLeer += ahoraC() - tL; trozos++;
    // Un tramo corto se leería sin error y dejaría ceros dentro de un peso.
    if (span.byteLength !== largo) {
      throw new Error(`tramo incompleto: pedidos ${largo} bytes desde ${tr.ini} ` +
        `y llegaron ${span.byteLength}`);
    }
    for (const t of tr.ts) {
      const tipoR = tipoReordenado(t, conReorden);
      const buf = tipoR ? creaBufferR(device, t, tipoR) : nuevoBuf(t);
      const tS = ahoraC();
      const desde = t.start - tr.ini;
      const n4 = Math.ceil(t.nBytes / 4) * 4;
      const pad = new Uint8Array(n4);
      // Formato reordenado: se escribe DENTRO de la copia de relleno que ya se
      // hacía, en vez de copiar tal cual. Mismo tamaño, cero memoria extra.
      if (tipoR) reordena(tipoR, span.subarray(desde, desde + t.nBytes), bloquesDe(t), pad);
      else pad.set(span.subarray(desde, desde + t.nBytes));
      device.queue.writeBuffer(buf, 0, pad, 0, n4);
      msSubir += ahoraC() - tS;
      uploaded += t.nBytes;
      onProgress(uploaded, total, t.name);
      weights.set(t.name, { buf, t: tipoR ? comoReordenado(t, tipoR) : t });
      if (tipoR) reordenados++;
    }
  }
  // `onSubmittedWorkDone` se mide aparte: es donde se paga de verdad el trabajo
  // que `writeBuffer` solo había ENCOLADO. Sin separarlo, subir parecería
  // gratis y el tiempo aparecería como un hueco sin dueño.
  const tEspera = ahoraC();
  await device.queue.onSubmittedWorkDone();
  subidosEsperados = suben.length;
  medidaCarga = {
    tensores: suben.length, saltados: g.tensors.length - suben.length, trozos, bytes: total,
    leerMs: msLeer, subirMs: msSubir, esperaMs: ahoraC() - tEspera,
    totalMs: ahoraC() - tCarga0,
    // Cuántos tensores se subieron reordenados (0 con `__ELFFUSS_REPACK` apagado).
    reordenados,
  };
  console.log(`[engine] carga: ${suben.length} tensores en ${trozos} trozos · ` +
    `leer ${msLeer.toFixed(0)} ms · encolar ${msSubir.toFixed(0)} ms · ` +
    `esperar GPU ${medidaCarga.esperaMs.toFixed(0)} ms · total ${medidaCarga.totalMs.toFixed(0)} ms`);

  // `picoGPU` es el pico de memoria pedida al dispositivo sin reciclar dentro de un
  // solo forward. Un prefill largo acumula TODOS los intermedios de todas las
  // capas, y cada uno escala con el número de tokens.
  return { g, hp, ops, weights, device, bytes: total, picoGPU: () => ops.ctx.picoBytes };
}

export function createSession(model, opts = {}) {
  const { device, hp, ops, weights } = model;
  // Sonda de diagnostico: cuando esta puesta, cada intermedio con nombre se
  // ofrece al llamador. Es lo que permite comparar contra el volcado nodo a
  // nodo de un forward real y localizar el paso exacto donde algo se desvia,
  // en vez de ver solo que la capa entera no cuadra.
  const probe = opts.probe || null;
  let capaActual = -1;
  // Buffers que una sonda ha pedido CONSERVAR hasta el final del forward. Sin
  // esto el mecanismo solo servía para la última capa, y de la forma más
  // traicionera que hay: `ctx.recycleDesde` recicla los intermedios en cuanto
  // la capa termina —protege los de `tap`, pero no los que se entregan por
  // `probe`—, así que leer uno después del forward devolvía el trozo de memoria
  // REUTILIZADO por las capas siguientes. Un banco que sondeó nueve capas sacó
  // la misma cifra en ocho de ellas, idéntica hasta el decimal, y solo la última
  // era real. No fallaba nada: contestaba otra cosa.
  // El contrato: si la sonda devuelve `true`, ese buffer no se recicla.
  let conservadas = new Set();
  const P = (name, buf, n) => {
    if (probe && probe(name, buf, n, capaActual) === true) conservadas.add(buf);
    return buf;
  };
  const { quant, nn, ssm: ssmOps, attn, ctx } = ops;
  // ── FUSIÓN DE KERNELS DE FILA (`__ELFFUSS_FUSION`, apagada por defecto) ─────
  // Cuatro cadenas de despachos pequeños pasan a uno cada una, con la MISMA
  // aritmética en el mismo orden (ver `kernels.fusion.wgsl.js`): post-norma +
  // residual + norma siguiente (F1), embedding + primera norma (F2), la
  // convolución corta tras `in_proj` (F4) y QK-norm + RoPE + escritura en la
  // caché (F6). Se decide por PASADA (`fusionPase`), no por sesión: un A/B
  // alterna en la misma sesión sin recargar el modelo. Apagada, nada de esto
  // se toca y el forward es el de siempre, despacho a despacho.
  const fus = createFusionOps(ops);
  let fusionPase = false;
  // La norma de ENTRADA que ya dejó hecha la capa anterior (F1) o el embedding
  // (F2): { capa, peso, buf }, con `capa` el índice de la capa que la consume o
  // 'salida' para la norma final, y `peso` el tensor con el que se hizo. Se
  // consume una vez: quien la pide la borra. Quien la pide dice además QUÉ peso
  // usaría él: si no coincide, no se usa y se calcula la de siempre.
  let nrmPendiente = null;
  const normaHecha = (capa, peso) => {
    const n = nrmPendiente;
    nrmPendiente = null;
    return n && n.capa === capa && n.peso === peso ? n.buf : null;
  };
  // Qué norma va DESPUÉS de cada capa: la de entrada de la siguiente o la final.
  // Todos los bloques (ssm, conv, atención) empiezan por su `attn_norm`.
  const normaSiguiente = new Map((hp.layers || []).map((l, k) => {
    const sig = hp.layers[k + 1];
    return [l.index, sig ? { capa: sig.index, peso: `blk.${sig.index}.attn_norm.weight` }
      : { capa: 'salida', peso: 'output_norm.weight' }];
  }));
  const state = createState(device, hp, opts);
  const W = (name) => {
    const w = weights.get(name);
    if (!w) throw new Error('falta el tensor «' + name + '»');
    return w;
  };
  const get = (name) => W(name).buf;
  const has = (name) => weights.has(name);

  // Una proyección. El tipo de cuantización lo pone el FICHERO, no esta función:
  // el 0.8B lo trae todo en Q8_0, otros ficheros mezclan varios tipos, y el
  // forward es el mismo. Este es el único punto donde el modelo toca la
  // cuantización; de aquí en adelante todo es f32.
  // ROTACIÓN DE HADAMARD. Si el fichero la declara, las activaciones que entran
  // a los tensores enumerados hay que girarlas antes del matmul. No aplicarla no
  // falla: carga, ejecuta y devuelve ruido con cara de respuesta.
  const rot = leeRotacion((model.g && model.g.meta) || {});
  const bufSignos = new Map();      // anchura -> buffer subido una sola vez
  const signosDe = (ancho) => {
    if (!bufSignos.has(ancho)) {
      const v = rot.signos.get(ancho);
      if (!v) throw new Error(`la rotación no trae signos para la anchura ${ancho}`);
      // NO se pide al pozo de buffers: `ctx.alloc` entrega memoria reciclable y
      // `recycle()` la devuelve, así que el mismo buffer acababa siendo el
      // DESTINO de otra operación dentro del mismo pase — y WebGPU lo rechaza
      // con «includes writable usage and another usage in the same
      // synchronization scope», que no dice ni una palabra de reciclaje. Los
      // signos viven lo que vive el modelo, como los pesos.
      const buf = device.createBuffer({
        label: `signos.${ancho}`, size: v.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buf, 0, v);
      bufSignos.set(ancho, buf);
    }
    return bufSignos.get(ancho);
  };
  // Varios tensores comparten la MISMA activación de entrada (por ejemplo la
  // proyección fusionada y la puerta, las dos tras la misma norma). Sin memoria
  // se giraría lo mismo dos veces: correcto y tirado a la basura.
  let memoRot = new Map();
  // Geometría del reordenado de rasgos que pide `gdn_v_grouped`. Solo lo lleva
  // la salida del SSM: su activación llega por teselas [hd, nk, rep] y el peso
  // se guardó agrupado [hd, rep, nk]. Los nombres salen de la cabecera, no de
  // constantes: n_v es el rango del paso temporal y n_k el número de grupos.
  const meta = (model.g && model.g.meta) || {};
  const clave = (suf) => Object.keys(meta).find((k) => k.endsWith(suf));
  const nV = Number(meta[clave('.ssm.time_step_rank')] || 0);
  const nK = Number(meta[clave('.ssm.group_count')] || 0);
  const permDe = (name, ancho) => {
    if (!rot || !rot.vGrouped || !name.includes('.ssm_out.')) return null;
    if (!nV || !nK || nV % nK !== 0 || ancho % nV !== 0) {
      throw new Error(`geometría de cabezas incompatible con el reordenado: ancho ${ancho}, n_v ${nV}, n_k ${nK}`);
    }
    return { hd: ancho / nV, nk: nK, rep: nV / nK };
  };
  const giraEntrada = async (x, ancho, tokens, perm) => {
    // La clave lleva el reordenado: `ssm_out` comparte la activación de entrada
    // con otros tensores pero su transformada NO es la misma, y compartir la
    // memoria entre los dos daría el resultado del primero que pasara.
    const k = perm ? `${perm.hd}:${perm.nk}:${perm.rep}` : 'llana';
    // La MISMA activación alimenta varios tensores —`nrm` va a siete matmuls en
    // una capa— y para todos salvo `ssm_out` la rotación es idéntica. Sin
    // memoria se calcula una vez por matmul: cientos de veces por token, y cada
    // una escribe su resultado a memoria para que el matmul lo lea de vuelta.
    //
    // La memoria estuvo APAGADA por una razón buena: se indexaba por el buffer,
    // y los buffers se reciclan, así que el que fue una activación podía ser
    // otra distinta más adelante y la segunda recibía la rotación de la primera.
    // No fallaba, no avisaba, y el modelo decía ruido.
    //
    // Ahora la clave es la MARCA DE REPARTO (`ctx.marcaDe`): el pozo le pone un
    // número distinto a cada buffer cada vez que lo entrega, de modo que un
    // buffer reciclado es un valor nuevo y no puede alias con el anterior.
    // Sin marca —un buffer que no salió del pozo— no se memoriza nada: es
    // preferible repetir el trabajo a arriesgarse a devolver lo que no es.
    const marca = ops.ctx.marcaDe?.get(x);
    if (marca === undefined || globalThis.__ELFFUSS_ROT_MEMO === false) {
      return await nn.hadamard(x, signosDe(ancho), { rows: tokens, ancho, bloque: rot.bloque, perm });
    }
    const clave = `${marca}:${k}`;
    // Diagnóstico: sin contar aciertos no se sabe si la memoria sirve. Un x1,00
    // puede ser «no hay nada que ahorrar» o «no está acertando», y son cosas
    // muy distintas.
    ops.ctx.rotStats = ops.ctx.rotStats || { acierto: 0, fallo: 0 };
    if (memoRot.has(clave)) { ops.ctx.rotStats.acierto++; return memoRot.get(clave); }
    ops.ctx.rotStats.fallo++;
    const y = await nn.hadamard(x, signosDe(ancho), { rows: tokens, ancho, bloque: rot.bloque, perm });
    memoRot.set(clave, y);
    return y;
  };
  // Cuenta cuántos tensores rotados se han visto de verdad. Si el fichero
  // declara 401 y el forward toca 12, algo no encaja y hay que enterarse.
  const rotVistos = new Set();

  // GEMV2: proyecciones que ya dejó hechas el despacho fusionado de Q/K/V/puerta
  // (ver `qkvFusion`), por nombre de peso. Sin el interruptor está siempre vacío.
  const hechas = new Map();
  const mm = async (name, x, rows, k, tokens) => {
    // Una proyección ya hecha se entrega UNA vez, y solo a quien la pide sobre
    // la MISMA entrada: misma marca de reparto, así que un búfer reciclado no
    // vale aunque sea el mismo objeto.
    if (hechas.size) {
      const h = hechas.get(name);
      if (h) {
        hechas.delete(name);
        if (h.marca === ctx.marcaDe?.get(x) && h.rows === rows && h.tokens === tokens) return h.y;
      }
    }
    const w = W(name);
    let xin = x;
    if (rot && rot.pesos.has(name)) {
      rotVistos.add(name);
      xin = await giraEntrada(x, k, tokens, permDe(name, k));
    }
    return quant.matmul(w.t.typeName, w.buf, xin, rows, k, tokens);
  };

  // Los pesos F32 (normas, ssm_a, dt) se usan tal cual; los cuantizados pasan
  // por el matmul, que descuantiza dentro del shader.
  const pipe = ctx.pipeline;
  const eps = hp.eps;
  const D = hp.headDim, H = hp.heads, KVH = hp.kvHeads, E = hp.embd;
  const SH = hp.ssmHeads;              // cabezas de CLAVE (q, k)
  const SV = hp.ssmVHeads;             // cabezas de VALOR (v, alpha, beta, salida)
  const SS = hp.ssmState, CC = hp.convChannels, CK = hp.convKernel;
  const KEY = SH * SS;                 // key_dim = value_dim = 2048 en el 0.8B

  // GEMV fusionados del decode (`__ELFFUSS_GEMV2`, apagado por defecto): gate+up
  // con la SwiGLU en el epílogo, y Q/K/V/puerta en un despacho. Están hechos para
  // dar bit a bit lo mismo que las matrices sueltas con el GEMV nuevo: el orden
  // de la suma de cada fila es el mismo (ver kernels.gemv.wgsl.js).
  // Si no aplican —más de 3 tokens, tipos distintos, rotación de Hadamard,
  // dispositivo sin subgrupos de 32— no hacen nada y el bloque sigue como siempre.
  const gemv2 = quant.gemv || null;
  const conRotacion = (name) => !!(rot && (rot.pesos.has(name) || rot.inversos.has(name)));
  async function gateUpFusion(i, nrm, tokens) {
    if (!gemv2 || tokens > 3) return null;
    const ng = `blk.${i}.ffn_gate.weight`, nu = `blk.${i}.ffn_up.weight`;
    if (conRotacion(ng) || conRotacion(nu)) return null;
    const wg = W(ng), wu = W(nu);
    if (wg.t.typeName !== wu.t.typeName) return null;
    return gemv2.gateUp(wg.t.typeName, wg.buf, wu.buf, nrm, hp.ffnLength, E, tokens);
  }
  // Q, K, V y la puerta de una capa de atención en UN despacho. Las salidas no
  // se devuelven: se dejan en `hechas`, por nombre de peso, y `mm` las entrega
  // cuando el bloque pide esas proyecciones sobre la misma entrada. Así el
  // bloque sigue igual línea a línea, y cualquier variante que pida las
  // proyecciones en otro orden o desde otro sitio las encuentra hechas.
  // Con el mínimo de búferes por etapa las cuatro no caben (nueve búferes):
  // entonces van Q/K/V juntas (siete) y la puerta por su GEMV suelto.
  async function qkvFusion(i, nrm, tokens) {
    hechas.clear();
    if (!gemv2 || tokens > 3) return;
    const marca = ctx.marcaDe?.get(nrm);
    if (marca === undefined) return;
    const nombres = [`blk.${i}.attn_q.weight`, `blk.${i}.attn_k.weight`, `blk.${i}.attn_v.weight`];
    const filas = [(hp.qGate ? 2 : 1) * H * D, KVH * D, KVH * D];
    // La puerta por cabeza solo la usa el bloque cuando no hay puerta en Q.
    if (!hp.qGate && hp.headGate) { nombres.push(`blk.${i}.attn_gate.weight`); filas.push(H); }
    if (nombres.some(conRotacion)) return;
    const ws = nombres.map(W);
    const tipo = ws[0].t.typeName;
    if (ws.some((w) => w.t.typeName !== tipo)) return;
    let n = nombres.length;
    let ys = await gemv2.secciones(tipo, ws.map((w) => w.buf), nrm, filas, E, tokens);
    if (!ys && n === 4) {
      n = 3;
      ys = await gemv2.secciones(tipo, ws.slice(0, 3).map((w) => w.buf), nrm, filas.slice(0, 3), E, tokens);
    }
    if (!ys) return;
    for (let j = 0; j < n; j++) hechas.set(nombres[j], { y: ys[j], marca, rows: filas[j], tokens });
  }

  async function gather(idsBuf, tokens) {
    const w = W('token_embd.weight');
    const z = await quant.gather(w.t.typeName, w.buf, idsBuf, E, tokens);
    // La tabla guarda las filas YA giradas, así que aquí se deshace: h = s ⊙ (H z).
    // Los signos van DESPUÉS, al revés que en el camino normal — es la inversa,
    // y confundir el orden no falla, da ruido.
    if (rot && rot.inversos.has('token_embd.weight')) {
      rotVistos.add('token_embd.weight');
      return await nn.hadamard(z, signosDe(E), { rows: tokens, ancho: E, bloque: rot.bloque, inverso: true });
    }
    return z;
  }

  // FUSIÓN (F2): el embedding y la norma de entrada de la primera capa en un
  // despacho, que deja la norma para esa capa (`nrmPendiente`). Solo con el
  // lector de bloques de siempre y sin rotación inversa: una tabla girada,
  // o con rejilla va por `gather` y la norma de siempre (una reordenada al cargar
  // sí entra: ver `puedeGatherNorma`).
  async function gatherFusion(idsBuf, tokens) {
    const w = W('token_embd.weight');
    const l0 = hp.layers[0];
    if (!l0 || (rot && rot.inversos.has('token_embd.weight')) || !puedeGatherNorma(w)) {
      return await gather(idsBuf, tokens);
    }
    const { x, nrm } = await fus.gatherNorma(w.t.typeName, w.buf, idsBuf,
      get(`blk.${l0.index}.attn_norm.weight`), E, tokens, eps);
    nrmPendiente = { capa: l0.index, peso: `blk.${l0.index}.attn_norm.weight`, buf: nrm };
    return x;
  }

  const raw = (label, p, buffers, groups) => ops.dispatchAny(label, p, buffers, groups);

  async function silu(src, n) {
    const p = await pipe('silu', siluShader());
    const dst = ctx.alloc(n * 4, 'silu');
    const params = ctx.uniform([new Uint32Array([n, 0, 0, 0])]);
    await raw('silu', p, [src, dst, params], Math.ceil(n / 256));
    return dst;
  }

  // `dstF16` va en la CLAVE del pipeline, no solo en el shader: con una sola
  // clave, la primera llamada fijaría el compilado y la segunda escribiría con
  // el tipo de la primera. La caché saldría escrita a medias y sin un error.
  async function copyInto(src, dst, dstOffset, n, srcOffset = 0, dstF16 = false) {
    const p = await pipe(`copy_into${dstF16 ? '_h' : ''}`, copyIntoShader(256, dstF16));
    const params = ctx.uniform([new Uint32Array([n, dstOffset, srcOffset, 0])]);
    await raw('copy_into', p, [src, dst, params], Math.ceil(n / 256));
  }

  // Corte contiguo de una fila ancha: q, k y v salen del mismo tensor de 6144.
  const slice = (src, offset, width, tokens, rowWidth) =>
    attn.strideCopy(src, { headDim: width, heads: 1, tokens, stride: rowWidth, offset });

  // El cierre común de los dos tipos de bloque: residual, norm, SwiGLU, residual.
  // Con `postNorms` (lfm3) la salida del bloque y la del FFN pasan por su propia
  // RMSNorm ANTES de sumarse al residual.
  async function feedForward(x, layerOut, i, tokens) {
    const n = tokens * E;
    // FUSIÓN (F1): post-norma + residual + norma del FFN en un despacho. Sin
    // fusión `f1` es null y lo de abajo es exactamente lo de siempre.
    const f1 = fusionPase ? await fus.normaResidual(layerOut, x,
      hp.postNorms ? get(`blk.${i}.post_attention_norm.weight`) : null,
      get(`blk.${i}.${hp.ffnNorm}.weight`), tokens, E, eps) : null;
    if (hp.postNorms && !f1) {
      layerOut = await nn.rmsnorm(layerOut, get(`blk.${i}.post_attention_norm.weight`), tokens, E, eps);
    }
    const res = f1 ? f1.res : await nn.add(x, layerOut, n);
    // El nombre sale del perfil, no de `has(...)`: si se decidiera por presencia
    // del tensor, un modelo al que le faltara dejaría de fallar y empezaría a
    // mentir con una norma menos.
    const nrm = f1 ? f1.nrm : await nn.rmsnorm(res, get(`blk.${i}.${hp.ffnNorm}.weight`), tokens, E, eps);
    // gate+up+SwiGLU en un despacho (`__ELFFUSS_GEMV2`); null = camino de siempre.
    const f3 = globalThis.__ELFFUSS_GEMV2 === true ? await gateUpFusion(i, nrm, tokens) : null;
    const gate = f3 ? null : await mm(`blk.${i}.ffn_gate.weight`, nrm, hp.ffnLength, E, tokens);
    const up = f3 ? null : await mm(`blk.${i}.ffn_up.weight`, nrm, hp.ffnLength, E, tokens);
    // La intermedia del FFN lleva sonda. No es simetría con los otros nodos:
    // las tres matrices del FFN son dos tercios de los pesos del modelo, y si
    // esta activación es dispersa, las columnas de `ffn_down` que multiplican a
    // un canal despreciable no hace falta ni LEERLAS. Medir eso exige poder
    // verla, y sin sonda no se puede.
    const sw = P('ffn_sw', f3 || await nn.swiglu(gate, up, tokens * hp.ffnLength), tokens * hp.ffnLength);
    let down = await mm(`blk.${i}.ffn_down.weight`, sw, E, hp.ffnLength, tokens);
    if (f1) return await cierreFusion(down, res, i, tokens);
    if (hp.postNorms) down = await nn.rmsnorm(down, get(`blk.${i}.post_ffw_norm.weight`), tokens, E, eps);
    return await nn.add(res, down, n);
  }

  // FUSIÓN (F1), el cierre del bloque: post-norma del FFN + residual + la norma
  // de lo que viene después (la `attn_norm` de la capa siguiente o la final),
  // que se deja hecha en `nrmPendiente` para quien la consuma. Esa norma nace
  // con los intermedios del FFN de esta capa aún sin reciclar: en un modelo sin
  // post-normas el pico de memoria puede llevar una fila de activaciones más;
  // con ellas (lfm3) baja, porque las post-normas ya no se materializan.
  async function cierreFusion(down, res, i, tokens) {
    const sig = normaSiguiente.get(i);
    const r = await fus.normaResidual(down, res,
      hp.postNorms ? get(`blk.${i}.post_ffw_norm.weight`) : null, get(sig.peso), tokens, E, eps);
    nrmPendiente = { capa: sig.capa, peso: sig.peso, buf: r.nrm };
    return r.res;
  }

  // ── LA ÚLTIMA CAPA, SOLO PARA EL ÚLTIMO TOKEN (interruptor GEMM2_COLA) ─────
  // De la última capa de un prompt solo se usa la fila del último token: es la
  // única que llega a la cabeza. Sus cachés (KV, ventana de la convolución,
  // estado recurrente) sí necesitan TODOS los tokens, y se escriben antes de la
  // proyección de salida. Así que el corte va justo ahí: la mezcla se calcula
  // entera, y la proyección de salida, las normas sándwich, el FFN y las sumas
  // residuales se hacen para UNA fila.
  //
  // Tiene que dar los MISMOS bits. Las normas, las sumas y SwiGLU van fila a
  // fila o elemento a elemento, así que una fila sola sale igual que dentro del
  // lote. Los matmuls no: con un token caerían en el matvec, que suma en otro
  // orden. Por eso van por `quant.gemm2.cola`, que usa el mismo kernel y la misma
  // forma que habría tocado al lote entero. Si alguno de los cuatro pesos no
  // puede ir así, la capa se calcula como siempre (ver `colaReproducible`).
  //
  // `colaLote` vale el tamaño del lote SOLO mientras se calcula la última capa
  // con el corte puesto; se fija antes de cada capa y se borra al empezar cada
  // pasada, para que un forward que lanzó a medias no deje el corte armado.
  // `colaCortada` lo pone `colaFinal`: dice que el bloque llegó de verdad al
  // corte, que es lo que `unaPasada` exige antes de dar la capa por cortada.
  let colaLote = 0;
  let colaCortada = false;
  const mmCola = async (name, x1, rows, k, lote) => {
    const w = W(name);
    let xin = x1;
    if (rot && rot.pesos.has(name)) {
      rotVistos.add(name);
      xin = await giraEntrada(x1, k, 1, permDe(name, k));
    }
    const y = await quant.gemm2.cola(w.t.typeName, w.buf, xin, rows, k, lote);
    // `colaReproducible` lo comprobó antes de empezar: si aun así no aplica es
    // un fallo de programación, y se dice. Caer al matvec daría otros bits.
    if (!y) throw new Error(`cola: «${name}» no se puede calcular con el kernel del lote`);
    return y;
  };
  async function feedForwardCola(x1, out1, i, lote) {
    let layerOut = out1;
    if (hp.postNorms) {
      layerOut = await nn.rmsnorm(layerOut, get(`blk.${i}.post_attention_norm.weight`), 1, E, eps);
    }
    const res = await nn.add(x1, layerOut, E);
    const nrm = await nn.rmsnorm(res, get(`blk.${i}.${hp.ffnNorm}.weight`), 1, E, eps);
    const gate = await mmCola(`blk.${i}.ffn_gate.weight`, nrm, hp.ffnLength, E, lote);
    const up = await mmCola(`blk.${i}.ffn_up.weight`, nrm, hp.ffnLength, E, lote);
    const sw = await nn.swiglu(gate, up, hp.ffnLength);
    let down = await mmCola(`blk.${i}.ffn_down.weight`, sw, E, hp.ffnLength, lote);
    if (hp.postNorms) down = await nn.rmsnorm(down, get(`blk.${i}.post_ffw_norm.weight`), 1, E, eps);
    return await nn.add(res, down, E);
  }
  // `mezcla` es la salida del bloque ANTES de su proyección de salida, con
  // `ancho` columnas; `x` la entrada del bloque (el residual). Devuelve UNA fila.
  async function colaFinal(x, mezcla, nombreSalida, ancho, i, tokens) {
    colaCortada = true;
    const mezclaFila = ctx.alloc(ancho * 4, 'cola.mezcla');
    await copyInto(mezcla, mezclaFila, 0, ancho, (tokens - 1) * ancho);
    const x1 = ctx.alloc(E * 4, 'cola.x');
    await copyInto(x, x1, 0, E, (tokens - 1) * E);
    const out1 = await mmCola(nombreSalida, mezclaFila, E, ancho, tokens);
    return await feedForwardCola(x1, out1, i, tokens);
  }
  function colaReproducible(l, tokens) {
    const i = l.index;
    const salida = l.kind === 'conv' ? [`blk.${i}.shortconv.out_proj.weight`, E, E]
      : l.kind === 'ssm' ? [`blk.${i}.ssm_out.weight`, E, hp.ssmInner]
      : [`blk.${i}.attn_output.weight`, E, H * D];
    return [salida,
      [`blk.${i}.ffn_gate.weight`, hp.ffnLength, E],
      [`blk.${i}.ffn_up.weight`, hp.ffnLength, E],
      [`blk.${i}.ffn_down.weight`, E, hp.ffnLength],
    ].every(([n, r, k]) => has(n) && quant.gemm2.reproducible(W(n).t.typeName, r, k, tokens));
  }

  // La convolución CORTA de lfm3. Mismo andamio que la conv del SSM (ventana con
  // el estado de los tokens anteriores, conv depthwise causal, guardar el
  // estado), pero sin silu y entre dos puertas:
  //   in_proj → [B | C | x] → B·x → conv(K) → C·y → out_proj
  // El orden de los tres trozos es el de llama.cpp (`build_lfm_shortconv`):
  // B primero, luego C, luego x. Cambiarlo no peta, da ruido con cara de texto.
  async function convBlock(x, i, tokens) {
    capaActual = i;
    const nrm = P('attn_norm', normaHecha(i, `blk.${i}.attn_norm.weight`) ?? await nn.rmsnorm(x, get(`blk.${i}.attn_norm.weight`), tokens, E, eps), tokens * E);
    const bcx = P('in_proj', await mm(`blk.${i}.shortconv.in_proj.weight`, nrm, 3 * E, E, tokens), tokens * 3 * E);
    let y;
    if (fusionPase && fus.convCabe(CK)) {
      // FUSIÓN (F4): de `in_proj` a la entrada de `out_proj` en un despacho —los
      // tres cortes, B·x, la ventana, la conv, guardar el estado y C·y—. Solo
      // sustituye esa fontanería y deja `y` donde la deja la cadena: la cola del
      // bloque (`out_proj` y el FFN) es UNA para los dos caminos, y lo que se le
      // cuelgue a esa cola lo ven los dos. Los intermedios `bx` y `conv` no
      // llegan a existir, así que esas dos sondas no se ofrecen. Un kernel de
      // conv más largo que la tesela del kernel fusionado va por la cadena.
      y = await fus.convCorta(bcx, state.ssm.get(i).conv, get(`blk.${i}.shortconv.conv.weight`), E, tokens, CK);
    } else {
      const b = await slice(bcx, 0, E, tokens, 3 * E);
      const c = await slice(bcx, E, E, tokens, 3 * E);
      const xs = await slice(bcx, 2 * E, E, tokens, 3 * E);
      const bx = P('bx', await attn.mul(b, xs, tokens * E), tokens * E);

      const st = state.ssm.get(i);
      const window = CK - 1 + tokens;
      const pw = await pipe('conv_window', convWindowShader());
      const win = ctx.alloc(E * window * 4, 'convwin');
      const pwp = ctx.uniform([new Uint32Array([E, tokens, CK - 1, window])]);
      await raw('conv_window', pw, [st.conv, bx, win, pwp], Math.ceil(E * window / 256));
      const conv = P('conv', await ssmOps.conv(win, get(`blk.${i}.shortconv.conv.weight`), E, tokens, CK), tokens * E);
      const ps = await pipe('conv_save', convSaveShader());
      const psp = ctx.uniform([new Uint32Array([E, tokens, CK - 1, window])]);
      await raw('conv_save', ps, [win, st.conv, psp], Math.ceil(E * (CK - 1) / 256));

      y = await attn.mul(c, conv, tokens * E);
    }
    if (colaLote) return await colaFinal(x, y, `blk.${i}.shortconv.out_proj.weight`, E, i, tokens);
    const out = P('conv_out', await mm(`blk.${i}.shortconv.out_proj.weight`, y, E, E, tokens), tokens * E);
    return await feedForward(x, out, i, tokens);
  }

  async function ssmBlock(x, i, tokens) {
    capaActual = i;
    const nrm = P('attn_norm', normaHecha(i, `blk.${i}.attn_norm.weight`) ?? await nn.rmsnorm(x, get(`blk.${i}.attn_norm.weight`), tokens, E, eps), tokens * E);

    // qkv fusionado → ventana de conv (transposición + concatenación con el
    // estado del token anterior) → conv depthwise causal → silu.
    const qkv = P('qkv', await mm(`blk.${i}.attn_qkv.weight`, nrm, CC, E, tokens), tokens * CC);
    const st = state.ssm.get(i);
    const window = CK - 1 + tokens;
    const pw = await pipe('conv_window', convWindowShader());
    const win = ctx.alloc(CC * window * 4, 'convwin');
    const pwp = ctx.uniform([new Uint32Array([CC, tokens, CK - 1, window])]);
    await raw('conv_window', pw, [st.conv, qkv, win, pwp], Math.ceil(CC * window / 256));

    const conv = P('conv_raw', await ssmOps.conv(win, get(`blk.${i}.ssm_conv1d.weight`), CC, tokens), tokens * CC);
    const act = P('conv_silu', await silu(conv, CC * tokens), tokens * CC);

    // El estado de conv se guarda DESPUÉS de leer la ventana, no antes.
    const ps = await pipe('conv_save', convSaveShader());
    const psp = ctx.uniform([new Uint32Array([CC, tokens, CK - 1, window])]);
    await raw('conv_save', ps, [win, st.conv, psp], Math.ceil(CC * (CK - 1) / 256));

    const qRawS = P('q_conv', await slice(act, 0, KEY, tokens, CC), tokens * KEY);
    const kRawS = P('k_conv', await slice(act, KEY, KEY, tokens, CC), tokens * KEY);
    const q = P('q_l2', await ssmOps.l2norm(qRawS, tokens * SH, SS, eps), tokens * KEY);
    const k = P('k_l2', await ssmOps.l2norm(kRawS, tokens * SH, SS, eps), tokens * KEY);
    const v = P('v_conv', await slice(act, 2 * KEY, hp.ssmInner, tokens, CC), tokens * hp.ssmInner);

    // alpha y beta salen de la ENTRADA normalizada, no del qkv ni de la conv.
    const alpha = P('alpha', await mm(`blk.${i}.ssm_alpha.weight`, nrm, SV, E, tokens), tokens * SV);
    const gdecay = P('gate', await ssmOps.decayGate(alpha, get(`blk.${i}.ssm_dt.bias`),
      get(`blk.${i}.ssm_a`), SV, tokens), tokens * SV);
    const betaRaw = await mm(`blk.${i}.ssm_beta.weight`, nrm, SV, E, tokens);
    const beta = P('beta', await ssmOps.sigmoid(betaRaw, SV * tokens), tokens * SV);

    const o = P('gdn_out', await ssmOps.gatedDeltaNet(q, k, v, gdecay, beta, st.state, SV, tokens, SH), tokens * hp.ssmInner);

    // La puerta del norm final sale de su propia proyección.
    //
    // Los tres nodos que van de aquí al residual llevan sonda porque son el
    // último tramo del bloque donde el recuento de cabezas de VALOR vuelve a
    // importar: la norma con puerta normaliza por filas de SS sobre tokens×SV.
    // Sin sonda, un fallo aquí no aparecía como «la norma con puerta está mal»,
    // aparecía como una escalera que se desvía desde el bloque 0 sin decir en
    // qué operación.
    const z = P('z', await mm(`blk.${i}.attn_gate.weight`, nrm, hp.ssmInner, E, tokens), tokens * hp.ssmInner);
    const gn = P('gated_norm',
      await ssmOps.gatedNorm(o, get(`blk.${i}.ssm_norm.weight`), z, tokens * SV, SS, eps),
      tokens * hp.ssmInner);
    if (colaLote) return await colaFinal(x, gn, `blk.${i}.ssm_out.weight`, hp.ssmInner, i, tokens);
    const out = P('ssm_out', await mm(`blk.${i}.ssm_out.weight`, gn, E, hp.ssmInner, tokens), tokens * E);
    return await feedForward(x, out, i, tokens);
  }

  // `rope_freqs` (lfm3): un solo tensor para todas las capas globales.
  const factoresRope = has('rope_freqs.weight') ? get('rope_freqs.weight') : null;

  async function attnBlock(x, l, tokens, posBuf) {
    const i = l.index;
    capaActual = i;
    // Base de RoPE y ventana POR CAPA (lfm3); en el resto, las del modelo.
    const ropeBase = l.ropeBase ?? hp.ropeFreqBase;
    const factores = l.ropeFactores ? factoresRope : null;
    if (l.ropeFactores && !factores) throw new Error(`la capa ${i} pide rope_freqs y el fichero no lo trae`);
    const nrm = normaHecha(i, `blk.${i}.attn_norm.weight`) ?? await nn.rmsnorm(x, get(`blk.${i}.attn_norm.weight`), tokens, E, eps);

    // En qwen35 `attn_q` proyecta al DOBLE —Q y una puerta INTERCALADOS por
    // cabeza— y hay que partirlo. En llama proyecta a H*D y ya está, así que no
    // hay nada que partir: el matmul deja [T][H*D], que es justo [T][H][D], la
    // forma que esperan RoPE y los scores. Ni una copia de por medio.
    // GEMV2: Q, K, V y la puerta en un despacho; las recoge `mm` (ver `qkvFusion`).
    if (globalThis.__ELFFUSS_GEMV2 === true) await qkvFusion(i, nrm, tokens);
    let qRaw, gRaw = null;
    if (hp.qGate) {
      const qfull = await mm(`blk.${i}.attn_q.weight`, nrm, 2 * H * D, E, tokens);
      qRaw = await attn.strideCopy(qfull, { headDim: D, heads: H, tokens, stride: 2 * D, offset: 0 });
      gRaw = await attn.strideCopy(qfull, { headDim: D, heads: H, tokens, stride: 2 * D, offset: D });
    } else {
      qRaw = await mm(`blk.${i}.attn_q.weight`, nrm, H * D, E, tokens);
    }

    // QK-norm ANTES de RoPE.
    //
    // Y el RoPE va en NEOX aunque la referencia use MRoPE intercalado (qwen35
    // declara IMROPE, con cuatro secciones de dimensiones). NO es una
    // simplificación: para texto puro las dos son la MISMA función, y conviene
    // dejar por qué, porque parece un bug cada vez que se lee.
    //   1. El MRoPE reparte las dimensiones entre cuatro ejes de posición
    //      (t, h, w y uno extra) — de ahí que sea multimodal.
    //   2. Con tokens de texto, llama.cpp escribe la MISMA posición en las
    //      cuatro ranuras, así que los cuatro ángulos valen lo mismo y da igual
    //      qué eje toque a cada dimensión.
    //   3. Los cuatro avanzan en bloque: `indep_sects` sale de `is_vision`, que
    //      es falso aquí, o sea que ninguna sección reinicia su ángulo.
    //   4. Y el emparejamiento de elementos es literalmente la misma llamada
    //      que el NEOX, `rotate_pairs(n_dims, n_dims/2)`.
    // Deja de valer en cuanto entre una imagen: ahí las posiciones dejan de ser
    // iguales y hace falta el MRoPE de verdad.
    // ⚠️ EL MODO DE RoPE SALE DEL PERFIL, y es el cambio que más miedo da de
    // toda esta lista: equivocarse NO peta. Sale texto gramatical que se
    // degrada al crecer el contexto, o sea el fallo más caro de detectar que
    // hay. qwen35 empareja como NEOX; llama usa NORM porque su conversor a GGUF
    // ya permuta los pesos. Ver `state.js`, ARQUITECTURAS.
    const modoRope = hp.ropeMode === 'norm' ? ROPE_NORM : ROPE_NEOX;
    // A la caché, a partir de nPast: nunca se pisa lo ya cacheado.
    const cache = state.kv.get(i);
    const per = KVH * D;
    // UNA sola fuente de verdad para la precisión de la caché, leída del estado
    // que la reservó. Escribirla y leerla con precisiones distintas no lanza:
    // devuelve basura, y por eso el dato no se repite en cuatro sitios.
    const kvF16 = !!state.kvF16;
    let q;
    if (fusionPase && fus.qkvCabe(D, hp.ropeDims, { norma: !!hp.qkNorm, factores: !!factores })) {
      // FUSIÓN (F6): QK-norm + RoPE de q y k + K y V a la caché, en un despacho.
      // Las proyecciones de k y v se piden aquí, antes, porque el kernel quiere
      // las tres a la vez; el orden entre proyecciones independientes no cambia
      // ningún resultado. La misma precisión de caché que la cadena (`kvF16`).
      // Si el dispositivo no admite tantos buffers por despacho, va partido en
      // dos (consulta, y clave con valor); si ni así, por la cadena.
      const kRaw = await mm(`blk.${i}.attn_k.weight`, nrm, KVH * D, E, tokens);
      const v = await mm(`blk.${i}.attn_v.weight`, nrm, KVH * D, E, tokens);
      q = await fus.qkNormaRopeCache({
        q: qRaw, k: kRaw, v, pos: posBuf,
        wq: hp.qkNorm ? get(`blk.${i}.attn_q_norm.weight`) : null,
        wk: hp.qkNorm ? get(`blk.${i}.attn_k_norm.weight`) : null,
        cacheK: cache.k, cacheV: cache.v,
        heads: H, kvHeads: KVH, headDim: D, tokens, nPast: state.nPast,
        nDims: hp.ropeDims, mode: modoRope, freqBase: ropeBase, factores, kvF16, eps,
      });
    } else {
      // La QK-norm tampoco es universal: llama no trae `attn_q_norm` siquiera.
      const qn = hp.qkNorm
        ? await nn.rmsnorm(qRaw, get(`blk.${i}.attn_q_norm.weight`), tokens * H, D, eps)
        : qRaw;
      q = await nn.rope(qn, posBuf,
        { rows: tokens * H, ne0: D, heads: H, nDims: hp.ropeDims, mode: modoRope, freqBase: ropeBase, factores });

      const kRaw = await mm(`blk.${i}.attn_k.weight`, nrm, KVH * D, E, tokens);
      const kn = hp.qkNorm
        ? await nn.rmsnorm(kRaw, get(`blk.${i}.attn_k_norm.weight`), tokens * KVH, D, eps)
        : kRaw;
      const k = await nn.rope(kn, posBuf,
        { rows: tokens * KVH, ne0: D, heads: KVH, nDims: hp.ropeDims, mode: modoRope, freqBase: ropeBase, factores });
      const v = await mm(`blk.${i}.attn_v.weight`, nrm, KVH * D, E, tokens);

      await copyInto(k, cache.k, state.nPast * per, tokens * per, 0, kvF16);
      await copyInto(v, cache.v, state.nPast * per, tokens * per, 0, kvF16);
    }

    const nKv = state.nPast + tokens;
    const cfg = { headDim: D, heads: H, kvHeads: KVH, tokens, nKv, nPast: state.nPast, kvF16 };
    // ── Atención FUSIONADA (`__ELFFUSS_FLASH`, ver ops.flash.js) ────────────
    // Puntuaciones, softmax con la máscara causal y de ventana y combinación en
    // uno o dos despachos, sin materializar las puntuaciones, y con la puerta
    // por cabeza de lfm3 dentro de su epílogo. Misma matemática salvo redondeo.
    // Sustituye SOLO al núcleo: lo de después (la puerta de Q de qwen35, la
    // proyección de salida, el residual) es el mismo código para las dos rutas,
    // y así lo que se añada ahí vale también para esta. Si la forma no se puede
    // atender, `fusionada` queda en null y va la cadena de siempre, que da lo
    // mismo más despacio. La puerta de cabeza va dentro solo si no hay la de Q:
    // la misma precedencia que abajo. `puede` recibe la misma forma (con la
    // puerta) que `atencion`: si no, preguntaría por otro kernel.
    const fusionada = globalThis.__ELFFUSS_FLASH === true && ops.flash?.puede({ ...cfg, ventana: l.ventana || 0, puerta: !hp.qGate && !!hp.headGate })
      ? await ops.flash.atencion(q, cache.k, cache.v, { ...cfg, ventana: l.ventana || 0,
        puerta: !hp.qGate && hp.headGate ? await mm(`blk.${i}.attn_gate.weight`, nrm, H, E, tokens) : null })
      : null;
    const scores = fusionada ? null : await attn.scores(q, cache.k, cfg);
    const probs = fusionada ? null : await nn.softmax(scores,
      { rows: H * tokens, nKv, nQ: tokens, nPast: state.nPast, scale: 1 / Math.sqrt(D), ventana: l.ventana || 0 });
    const att = fusionada || await attn.combine(probs, cache.v, cfg);

    // La puerta multiplica la salida ANTES de la proyección — cuando la hay.
    // En llama la atención va directa a la proyección de salida.
    let gated = att;
    if (hp.qGate) {
      const gsig = await ssmOps.sigmoid(gRaw, tokens * H * D);
      gated = await attn.mul(att, gsig, tokens * H * D);
    } else if (hp.headGate && !fusionada) {
      // lfm3: un escalar por cabeza, de la ENTRADA normalizada (no de Q).
      const gh = await mm(`blk.${i}.attn_gate.weight`, nrm, H, E, tokens);
      const p = await pipe('mul_cabeza', mulCabezaShader());
      gated = ctx.alloc(tokens * H * D * 4, 'gated');
      const pp = ctx.uniform([new Uint32Array([tokens * H * D, D, 0, 0])]);
      await raw('mul_cabeza', p, [att, gh, gated, pp], Math.ceil(tokens * H * D / 256));
    }
    if (colaLote) return await colaFinal(x, gated, `blk.${i}.attn_output.weight`, H * D, i, tokens);
    const out = await mm(`blk.${i}.attn_output.weight`, gated, E, H * D, tokens);
    return await feedForward(x, out, i, tokens);
  }

  // El reloj del respiro de un prefill troceado, con `__ELFFUSS_PLAN` (si no,
  // null y todo como siempre). Sin él, cada pasada arrancaba en «ya toca» y
  // pagaba, tras su segunda capa, un frame entero con el dispositivo parado:
  // uno por lote. Dos cosas lo evitan sin perder para qué existe el respiro
  // (que el compositor pinte):
  //   · el reloj se COMPARTE entre las pasadas del mismo forward, como si
  //     fueran una sola: un respiro cada `cederMs` como mucho;
  //   · un frame PINTADO cuenta como respiro. Un latido con rAF apunta cuándo
  //     se pintó el último; si el compositor está pintando mientras esperamos
  //     al dispositivo entre lotes, forzar otro hueco no compra nada. Si deja de
  //     pintar, el latido se para y el respiro vuelve a entrar igual que antes.
  // El PRIMERO del forward sigue entrando en cuanto se puede (lo que busca es
  // que lo último de la pantalla se vea antes de que el dispositivo se ocupe),
  // pero con PLAN cede el frame SIN vaciar la cola (`respiroSinVaciar`): en ese
  // momento el dispositivo solo tiene en cola las capas recién enviadas, el
  // compositor entra entre un envío y el siguiente, y el dispositivo sigue
  // trabajando mientras llega el frame en vez de esperarlo parado. Esto vale
  // también para el prefill de una sola pasada, que era un frame ocioso por
  // forward. Los respiros siguientes, si hacen falta (si no se ve pintar),
  // vacían la cola como siempre: ahí el hueco es lo que garantiza el frame.
  // Con `cederMs = 0` (el A/B) nada cambia: respira cada `cederCada` capas y
  // vaciando siempre.
  let relojRespiro = null;
  let respiroSinVaciar = false;
  const nuevoRelojRespiro = () => {
    const r = { t: -Infinity, frame: -Infinity, actividad: performance.now() };
    if (typeof requestAnimationFrame === 'function') {
      const late = () => {
        // Se apaga solo cuando el forward deja de dar señales (acabó o se paró).
        if (performance.now() - r.actividad > 2000 || typeof requestAnimationFrame !== 'function') return;
        r.frame = performance.now();
        requestAnimationFrame(late);
      };
      requestAnimationFrame(late);
    }
    return r;
  };

  // UNA pasada: la implementación de siempre, intacta. La trocea `forward`.
  //
  // `signal` se mira en cada FRONTERA DE CAPA, y parar no rompe el lote a
  // medias: se deja de grabar, el callback devuelve null y `ctx.batch` cierra y
  // envía lo grabado por su camino normal (ops.js no sabe nada de parar). Lo que
  // sí hay que hacer antes de lanzar es esperar al dispositivo: los uniformes del lote
  // solo se pueden reciclar después de una sincronización —ver `forward`—, y
  // quien recibe el error va a reciclar. `state.advance` NO se llama: el lote no
  // terminó, así que nPast sigue donde estaba y lo que se escribió en la caché
  // por encima queda tapado por la máscara causal. El estado recurrente sí quedó
  // a medias, y por eso todo turno empieza con `restaura()` o `reset()`.
  const unaPasada = async (tokenIds, { tap = null, todosLosLogits = false, signal = null } = {}) => {
    let logitsTokens = 1;
    let parado = false;
    const tokens = tokenIds.length;
    const taps = new Map();
    const idsBuf = ctx.alloc(tokens * 4, 'ids');
    device.queue.writeBuffer(idsBuf, 0, new Uint32Array(tokenIds));
    const posBuf = ctx.alloc(tokens * 4, 'pos');
    device.queue.writeBuffer(posBuf, 0,
      Int32Array.from({ length: tokens }, (_, i) => state.nPast + i));

    memoRot = new Map();   // la memoria de rotación vale para UN forward
    conservadas = new Set();   // y lo que las sondas pidan conservar, también
    // FUSIÓN: el interruptor se lee UNA vez por pasada, y una norma pendiente de
    // una pasada anterior (parada a medias) no vale para esta.
    fusionPase = fusionActiva();
    nrmPendiente = null;
    // GEMM2_COLA (ver `colaFinal`). Nunca con sondas, taps ni todos los logits:
    // ahí se leen las filas de todos los tokens de la última capa.
    colaLote = 0;
    colaCortada = false;
    const ultimaCapa = hp.layers[hp.layers.length - 1];
    const cola = globalThis.__ELFFUSS_GEMM2 === true && globalThis.__ELFFUSS_GEMM2_COLA === true &&
      tokens > 1 && !tap && !todosLosLogits && !probe && !!quant.gemm2 && colaReproducible(ultimaCapa, tokens);
    let colaHecha = false;
    const logits = await ops.batch('forward', async () => {
      let x = fusionPase ? await gatherFusion(idsBuf, tokens) : await gather(idsBuf, tokens);
      // El índice −1 es el embedding, antes de la primera capa. No es un
      // capricho de simetría: si los ids o la tabla se leen mal, TODAS las
      // capas se desvían y la escalera señala la 0 sin distinguir «entré mal»
      // de «la capa 0 calcula mal». Con este ancla esa duda se resuelve.
      if (tap && tap.has(-1)) taps.set(-1, x);
      let capaN = 0;
      // Arranca en «ya toca»: la primera capa de un prefill largo es un buen
      // momento para dejar pintar, y no depender del reloj del forward anterior.
      // Con PLAN, las pasadas siguientes del MISMO forward siguen su reloj.
      let ultimoRespiro = relojRespiro ? relojRespiro.t : -Infinity;
      for (const l of hp.layers) {
        if (signal?.aborted) { parado = true; break; }
        // Se recicla lo de CADA capa en cuanto termina, en vez de guardarlo
        // todo hasta el final del forward. Sin esto el pico de memoria del dispositivo
        // crecía con capas × tokens, o sea que con un prompt normal llegaba a
        // pedir varias veces lo que pesa el modelo entero — y el factor empeora
        // cuantas más capas tiene.
        const marca = ctx.scratch.length;
        // FUSIÓN: la norma que dejó hecha la capa anterior muere al acabar ESTA,
        // como la norma que sustituye (ver `reciclableDesde`). Sin fusión no hay
        // pendiente y la marca es la de siempre.
        const marcaCapa = nrmPendiente ? fus.reciclableDesde(nrmPendiente.buf, marca) : marca;
        const previo = x;
        colaLote = cola && l === ultimaCapa ? tokens : 0;
        x = l.kind === 'ssm' ? await ssmBlock(x, l.index, tokens)
          : l.kind === 'conv' ? await convBlock(x, l.index, tokens)
          : await attnBlock(x, l, tokens, posBuf);
        if (colaLote) {
          // Que el corte estuviera armado no basta: el bloque tiene que haber
          // pasado por `colaFinal`. Un camino del bloque que vuelva antes del
          // corte (otra variante del mismo bloque, una fusión) devuelve TODAS
          // las filas, y la cabeza de una fila leería la del PRIMER token:
          // logits de otro token, sin un error. Se dice aquí, antes de la cabeza.
          if (!colaCortada) throw new Error(`cola: la capa ${l.index} volvió sin pasar por el corte (colaFinal)`);
          colaHecha = true; colaLote = 0;   // x es ya UNA fila
        }
        if (tap && tap.has(l.index)) taps.set(l.index, x);
        // Vivos: la salida de esta capa (entra en la siguiente), la entrada si
        // el bloque la devolvió tal cual, y cualquier tensor con sonda puesta,
        // que el llamador va a leer DESPUÉS del forward.
        // Con fusión, también la norma que esta capa deja hecha para la siguiente.
        ctx.recycleDesde(marcaCapa, x, previo, ...taps.values(), ...conservadas,
          ...(nrmPendiente ? [nrmPendiente.buf] : []));
        // Trocear el forward para no colgar el ordenador. SOLO en prefill
        // (tokens>1): el decode es 1 token y nunca colgó, y trocearlo lo
        // ralentizaría sin motivo. Se corta el command buffer en CADA capa
        // (barato: solo un submit más) y de vez en cuando se drena la cola y se
        // cede un frame, para que el compositor tenga su hueco. No cambia la
        // numérica: el orden de la cola se respeta. Ver ctx.corta.
        //
        // El troceo es GRATIS y va en todas las capas; el respiro es CARO (drena
        // la cola, que mata el solape entre la CPU y el dispositivo, y encima espera
        // al refresco) y va
        // limitado por reloj. Ver `cederMs`: antes iba atado al contador de capas
        // y era lo que fijaba el techo del prefill.
        // DECODE: cortar el command buffer cada `cortaDecode` capas deja que el
        // dispositivo empiece mientras la CPU sigue grabando el resto del token.
        // Sin respiro (no hace falta: un token no cuelga nada). 0 = no cortar.
        if (tokens === 1 && cortaDecode > 0 && (capaN + 1) % cortaDecode === 0 && capaN + 1 < hp.layers.length) ctx.corta();
        if (tokens === 1) capaN++;
        if (tokens > 1) {
          if (relojRespiro) {
            relojRespiro.actividad = performance.now();
            if (relojRespiro.frame > ultimoRespiro) ultimoRespiro = relojRespiro.frame;
          }
          const toca = (capaN + 1) % cortaCada === 0;
          // El respiro drena la cola; si el pase sigue abierto, lo que drena es
          // una cola VACÍA —el trabajo ni se ha enviado— y se paga el frame sin
          // conseguir el hueco que se buscaba. Así que cuando el respiro va a
          // entrar, se corta aunque no tocase por intervalo.
          const respira = (capaN + 1) % cederCada === 0 && performance.now() - ultimoRespiro >= cederMs;
          if (toca || respira) ctx.corta();
          capaN++;
          if (respira) {
            // Con PLAN, el primero del forward no vacía la cola (ver arriba).
            if (!(respiroSinVaciar && cederMs > 0 && ultimoRespiro === -Infinity)) await device.queue.onSubmittedWorkDone();
            await nuevoFrame(signal);
            ultimoRespiro = performance.now();
            if (relojRespiro) relojRespiro.t = ultimoRespiro;
          }
        }
      }
      // La última frontera: tras la última capa solo queda la cabeza, y unos
      // logits que nadie va a leer no merecen la espera.
      if (signal?.aborted) parado = true;
      if (parado) return null;
      if (colaHecha) {
        // La última capa ya dejó solo la fila del último token: la misma norma
        // y la misma cabeza que abajo, sin el recorte (ya está hecho).
        const nrm1 = await nn.rmsnorm(x, get('output_norm.weight'), 1, E, eps);
        const cabeza = has('output.weight') ? 'output.weight' : 'token_embd.weight';
        logitsTokens = 1;
        return await mm(cabeza, nrm1, model.g.meta['tokenizer.ggml.tokens'].length, E, 1);
      }
      // Con fusión, la norma final ya la dejó hecha la última capa (F1).
      const nrm = normaHecha('salida', 'output_norm.weight') ?? await nn.rmsnorm(x, get('output_norm.weight'), tokens, E, eps);
      // SOLO EL ÚLTIMO TOKEN. La cabeza es la matriz más grande del modelo
      // (vocabulario × embedding) y para generar solo hace falta la última fila:
      // calcularla para todo el prompt costaba cientos de veces más, y se pagaba
      // por N−1 vectores de logits que se descartan enteros.
      let entradaCabeza = nrm, tokensCabeza = tokens;
      if (!todosLosLogits && tokens > 1) {
        const fila = ctx.alloc(E * 4, 'ultimo');
        await copyInto(nrm, fila, 0, E, (tokens - 1) * E);
        entradaCabeza = fila; tokensCabeza = 1;
      }
      const head = has('output.weight') ? 'output.weight' : 'token_embd.weight';
      logitsTokens = tokensCabeza;
      return await mm(head, entradaCabeza, model.g.meta['tokenizer.ggml.tokens'].length,
        E, tokensCabeza);
    });

    if (parado) {
      // La espera va ANTES de lanzar, no es opcional: ver arriba.
      await device.queue.onSubmittedWorkDone();
      throw errorParado();
    }
    state.advance(tokens);
    // `logitsTokens` dice para CUÁNTOS tokens hay logits: casi siempre 1,
    // aunque `tokens` sea el prompt entero. Sin este dato, quien deduzca el
    // tamaño del vocabulario dividiendo por `tokens` se equivoca.
    return { logits, tokens, logitsTokens, taps };
  };

  // DIAGNÓSTICO DE COBERTURA. El fichero enumera cuántos tensores llevan
  // rotación; si el forward toca muchos menos, hay un camino que se la salta —
  // y eso no falla, devuelve ruido.
  const _rotacion = () => (rot ? {
    declarados: rot.pesos.size, inversos: rot.inversos.size,
    aplicados: rotVistos.size, bloque: rot.bloque,
    faltan: [...rot.pesos, ...rot.inversos].filter((n) => !rotVistos.has(n)),
  } : null);

  return {
    _rotacion,
    state,
    hp,

    // Ejecuta el forward y devuelve los logits del ÚLTIMO token.
    // `tap` permite pedir el residual de una capa concreta para compararlo con
    // el oráculo: es la diferencia entre «el texto suena raro» y «el bloque 7
    // se desvía y los seis anteriores cuadran».
    // Prefill TROCEADO. La máscara causal se aplica en el softmax, DESPUÉS de
    // calcular los scores, así que una pasada de N tokens calcula la matriz
    // N×N entera y tira la mitad. Troceando, cada lote solo se mide contra lo
    // que ya hay: para 1347 tokens en lotes de 512 son 1,22 M de posiciones en
    // vez de 1,81 M, y la diferencia CRECE con la longitud (×1,6 a 2048, ×1,78
    // a 4096). Es la parte «superlineal» que la medida del prefill detectó y la
    // razón de que llama.cpp —que trocea por defecto— fuera 58× más rápido.
    //
    // No toca ningún kernel ni la numérica: es la misma maquinaria del decode,
    // con nPast avanzando entre lotes. Lo que cambia es cuánto trabajo inútil
    // se hace, no el resultado.
    //
    // PARAR: `opts.signal` (un AbortSignal) se mira al entrar, entre lotes y en
    // cada frontera de capa dentro de un lote. Al entrar no se toca nada; entre
    // lotes, nPast queda al final del último lote completo; dentro de uno, donde
    // empezó ese lote. Las tres salidas lanzan un AbortError y dejan el
    // dispositivo en reposo, así que `recycle()` es seguro justo después. Todo lote terminado
    // se avisa por `onProgress`, también el último antes del «para»: lo que se
    // cuenta es trabajo hecho, y nPast lo incluye.
    async forward(tokenIds, opts = {}) {
      const senal = opts.signal || null;
      if (senal?.aborted) throw errorParado();
      // 128 y no 512, y la razón NO es la memoria: es que el dispositivo tiene que poder
      // ATENDER AL ESCRITORIO entre despacho y despacho. El bucle del SSM recorre
      // todos los tokens del lote DENTRO de un solo despacho, sin ceder; con 512
      // ese núcleo corre largo, y el sistema puede matar la sesión gráfica cuando
      // el compositor no consigue pintar dentro del plazo de su vigilante.
      //
      // Trocear más fino además REDUCE trabajo: la máscara causal se aplica en el
      // softmax, o sea que cada lote calcula su matriz entera; a lotes más cortos,
      // menos posiciones enmascaradas se calculan para tirarlas.
      const LOTE = loteActual;
      // Con `tap` NO se trocea: la referencia contra la que se compara sale de
      // UNA pasada, y comparar 64 escalones contra tres pasadas concatenadas no
      // significa nada. Con `todosLosLogits`
      // tampoco, porque troceando solo sobreviven los del último lote.
      // `onProgress(hechos, total)` se llama DESPUÉS de cada lote, que es justo
      // después de esperar a que el dispositivo lo termine. Así lo que se reporta es
      // trabajo HECHO y no encolado: una barra que avanza por encolado llega al
      // 100 % y se queda ahí mientras el dispositivo sigue, que es peor que no tenerla.
      const avisa = typeof opts.onProgress === 'function' ? opts.onProgress : null;
      // Un reloj nuevo por forward troceado (ver `relojRespiro`): nunca se
      // hereda el de otro forward.
      relojRespiro = globalThis.__ELFFUSS_PLAN === true && !opts.tap && !opts.todosLosLogits &&
        tokenIds.length > LOTE ? nuevoRelojRespiro() : null;
      respiroSinVaciar = globalThis.__ELFFUSS_PLAN === true && !opts.tap && !opts.todosLosLogits;
      if (opts.tap || opts.todosLosLogits || tokenIds.length <= LOTE) {
        const r = await unaPasada(tokenIds, opts);
        avisa?.(tokenIds.length, tokenIds.length);
        return r;
      }
      let ultimo = null;
      for (let i = 0; i < tokenIds.length; i += LOTE) {
        ultimo = await unaPasada(tokenIds.slice(i, i + LOTE), opts);
        // ESPERAR a que el lote termine en el dispositivo antes de encolar el siguiente.
        // Los buffers uniform se reciclan ENTRE lotes (ops.js lo dice: «se
        // reciclan entre lotes, no dentro»), y eso solo es seguro si hay una
        // sincronización en medio. El bucle de decode la tiene sin buscarla,
        // porque lee los logits en cada paso; aquí no había ninguna, así que el
        // lote siguiente reutilizaba uniformes que el anterior todavía no había
        // consumido. El resultado no era un error: era OTRO token (197 en vez
        // de 62), que es la peor forma de fallar porque sale texto plausible.
        await device.queue.onSubmittedWorkDone();
        // El aviso va JUSTO DESPUÉS de esa espera y ANTES de mirar la señal: el
        // lote ya está hecho y nPast lo incluye, así que se cuenta aunque el
        // prefill se pare aquí mismo. Callarlo dejaría el progreso un lote por
        // detrás de lo que de verdad entró. Quien no quiera avisos después de
        // un «para» los filtra él: el proveedor deja de pasarlos a `onPrefill`
        // en cuanto nota la señal.
        avisa?.(Math.min(i + LOTE, tokenIds.length), tokenIds.length);
        // Y ahora la señal, con el dispositivo parado y nPast al final del último lote
        // completo. Tras el último lote no se mira: el trabajo ya está hecho y
        // tirarlo no ahorra nada, así que el forward termina y el «para» lo
        // nota quien llama en su siguiente comprobación.
        if (senal?.aborted && i + LOTE < tokenIds.length) throw errorParado();
      }
      // `tokens` se corrige al total: quien lo lea espera cuántos entraron, no
      // cuántos traía el último lote.
      return { ...ultimo, tokens: tokenIds.length };
    },


    recycle: () => ctx.recycle(),
    destroy: () => state.destroy(),
  };
}
