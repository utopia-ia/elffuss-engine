// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · operaciones de los GEMV nuevos del decode (`__ELFFUSS_GEMV2`).
// ─────────────────────────────────────────────────────────────────────────────
// Tres operaciones sobre los kernels de `kernels.gemv.wgsl.js`:
//   · matmul: la proyección de siempre (1 a 3 tokens), mismo contrato que
//     `quant.matmul`: y = W·x con x token-major y la salida [tokens][filas];
//   · gateUp: gate y up del FFN con la SwiGLU en el epílogo: devuelve
//     directamente silu(W_gate·x)·(W_up·x). Tres despachos en uno;
//   · secciones: varias matrices con la MISMA entrada y la misma K (Q, K, V y la
//     puerta de la atención) en un solo despacho, cada una con su salida.
//
// Cualquiera de las tres devuelve `null` cuando no aplica —tipo sin kernel
// nuevo, dispositivo sin subgrupos de 32, búfer sin la holgura que hace falta,
// demasiados enlaces, un búfer reordenado que no se deja comprobar, o un shader
// que el compilador del navegador rechazó— y quien llama sigue por el camino de
// siempre. La idea es que el interruptor no rompa un modelo: como mucho, no lo
// acelera. Dos salvedades, a propósito: una forma fijada a mano para el barrido
// (`__ELFFUSS_GEMV2_FORMA`) que no vale LANZA en vez de cambiarse por otra, y un
// búfer reordenado sin marca hace lanzar al camino de siempre, igual que con
// el interruptor quitado (ver `bloquesTensorR`).
//
// El despacho lo pone quien crea esto (`ops.js`): así estas operaciones pasan
// por el MISMO despachador que las demás —agrupado, perfilado y lo que venga—
// y no hay una segunda copia que se desincronice.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.

import {
  gemv2Shader, gemv2Clave, gemv2Formato, gemv2Forma, gemv2Params, gemv2Escalas,
  gemv2BloquesDeBuffer, gemv2BytesBufferR, gemv2BytesNecesarios, GEMV2_TIPOS, GEMV2_MAX_TOKENS,
} from './kernels.gemv.wgsl.js';
import { AXIS } from './kernels.wgsl.js';

// Los grupos en dos ejes, como el resto del motor: uno solo topa en AXIS.
function dims(grupos) {
  return grupos <= AXIS ? [grupos, 1] : [AXIS, Math.ceil(grupos / AXIS)];
}

// `compruebaR(tipo, buf, { bloques }, que)`: la comprobación de los búferes
// reordenados del módulo que los fabrica (`compruebaBufferR`): lanza si el búfer
// no lleva su marca, es de otro tipo o la llamada se sale del tensor, y si no,
// devuelve el N del tensor. Sin ella (null), N sale del tamaño del búfer.
export function createGemvOps(device, { ctx, subgroupSize = 0, despacha, compruebaR = null }) {
  const get = ctx.pipeline;
  const limites = device.limits || {};

  // `unpack4xI8` es una extensión de lenguaje; sin ella, el shader hace los
  // mismos desplazamientos a mano. `__ELFFUSS_GEMV2_U4X8 = false` fuerza la
  // versión a mano para compararlas (va en la clave del pipeline).
  let u4x8Disp = null;
  const u4x8 = () => {
    if (globalThis.__ELFFUSS_GEMV2_U4X8 === false) return false;
    if (u4x8Disp === null) {
      u4x8Disp = !!globalThis.navigator?.gpu?.wgslLanguageFeatures?.has?.('packed_4x8_integer_dot_product');
    }
    return u4x8Disp;
  };

  // El reparto de carriles supone subgrupos de 32 exactos: con otro ancho,
  // camino de siempre. `__ELFFUSS_NO_SUBGROUP` (lo usan pruebas internas) manda.
  const listo = () => subgroupSize === 32 && !globalThis.__ELFFUSS_NO_SUBGROUP;

  // Shaders que el compilador del navegador rechazó. naga los valida todos,
  // pero el compilador de cada navegador tiene su propio análisis, y un rechazo
  // no puede costar el modelo: esa clave no se vuelve a intentar (compilar cuesta
  // y fallaría igual), se avisa UNA vez y esas llamadas siguen por el camino de
  // siempre. El banco del dispositivo exige que la lista salga vacía.
  const rechazadas = new Map();          // clave → mensaje del compilador
  async function pipeline(o) {
    const clave = gemv2Clave(o);
    if (rechazadas.has(clave)) return null;
    try {
      return await get(clave, () => gemv2Shader(o));
    } catch (e) {
      const msg = e?.message || String(e);
      rechazadas.set(clave, msg);
      console.warn(`[engine] gemv2: el compilador rechazó «${clave}»; esas llamadas siguen por el camino de siempre:`, msg);
      return null;
    }
  }

  // Despachos hechos por cada operación. Solo diagnóstico: con esto un banco
  // con el interruptor puesto comprueba que el GEMV nuevo corrió DE VERDAD y no
  // cayó en silencio al camino de siempre (que da números casi iguales).
  const hechos = { matmul: 0, gateUp: 0, secciones: 0 };

  // Bloques del TENSOR que guarda un búfer reordenado, o null si no se pueden
  // saber con seguridad. Las escalas empiezan tras los cuantos de TODOS sus
  // bloques, y una llamada puede pedir menos filas de las que tiene el tensor
  // (la cabeza pide las del vocabulario): hace falta el N del tensor, no el de
  // la llamada.
  //   · Con `compruebaR` (la marca que el cargador le pone a cada búfer): la
  //     MISMA comprobación que hace el camino de siempre —marca, tipo y que la
  //     llamada no se salga del tensor— y el N de la marca. Si no pasa, null, y
  //     el camino de siempre lanza con su porqué: el interruptor no cambia qué
  //     búferes se aceptan.
  //   · Sin ella: del tamaño, y solo si el búfer mide EXACTAMENTE lo que el
  //     cargador le da a un tensor de ese N. Uno más grande (el pozo redondea a
  //     256) se leería como un tensor de más bloques, con las escalas en otro
  //     sitio y sin un error. Queda un caso que el tamaño no decide —un búfer
  //     que por casualidad mide justo lo que el de otro N (ver
  //     `gemv2BloquesDeBuffer`)—; sin la marca no hay cargador que fabrique
  //     estos búferes, y quien los fabrique a mano tiene que darles ese tamaño.
  function bloquesTensorR(tipo, buf, bloquesLlamada) {
    if (compruebaR) {
      try { return compruebaR(tipo, buf, { bloques: bloquesLlamada }, 'gemv2'); } catch { return null; }
    }
    const n = gemv2BloquesDeBuffer(tipo, buf.size);
    return buf.size === gemv2BytesBufferR(tipo, n) ? n : null;
  }

  // Lo común a las tres: ¿hay kernel para esto?, y su geometría.
  function prepara(tipo, k, tokens, bufs, filasPorBuf) {
    if (!listo() || !GEMV2_TIPOS.has(tipo)) return null;
    if (!Number.isInteger(tokens) || tokens < 1 || tokens > GEMV2_MAX_TOKENS) return null;
    const F = gemv2Formato(tipo);
    if (!(k > 0) || k % F.bloque !== 0) return null;
    const bpr = k / F.bloque;
    // El búfer tiene que servir lo que el kernel lee, holgura incluida. Leer
    // fuera de rango no lanza: devuelve otra palabra y la fila sale mal.
    // En los reordenados, además, dónde empiezan las escalas depende de los
    // bloques del TENSOR (ver `bloquesTensorR`): una llamada con menos filas
    // que el tensor no las puede buscar donde acabarían las suyas.
    const esc = [];
    for (let i = 0; i < bufs.length; i++) {
      const f = filasPorBuf[i];
      const size = bufs[i]?.size;
      if (!(f >= 1) || !(size > 0)) return null;
      const n = F.repack ? bloquesTensorR(tipo, bufs[i], f * bpr) : f * bpr;
      if (!Number.isInteger(n) || n < f * bpr || !(size >= gemv2BytesNecesarios(tipo, n))) return null;
      esc.push(gemv2Escalas(tipo, n));
    }
    return { F, bpr, esc, pasos: Math.ceil(bpr / F.bloquesPaso), cola: bpr % F.bloquesPaso !== 0 };
  }

  return {
    // y = W·x, 1 a 3 tokens. Mismo contrato y misma salida que `quant.matmul`.
    async matmul(tipo, wBuf, xBuf, rows, k, tokens = 1) {
      const g = prepara(tipo, k, tokens, [wBuf], [rows]);
      if (!g) return null;
      const forma = gemv2Forma({ tipo, filas: [rows], k, T: tokens, nm: 1,
        maxCompartida: limites.maxComputeWorkgroupStorageSize || 16384 });
      const o = { tipo, R: forma.R, NSG: forma.NSG, T: tokens, modo: 'una', n: 1, u4x8: u4x8(), cola: g.cola };
      const p = await pipeline(o);
      if (!p) return null;
      const y = ctx.alloc(rows * tokens * 4, 'matmul.y');
      const grupos = rows / forma.R;
      const params = ctx.uniform([gemv2Params({ k, bpr: g.bpr, pasos: g.pasos, filas: [rows],
        esc: g.esc, fin: [grupos] })]);
      const [dx, dy] = dims(grupos);
      await despacha(`matmul:${tipo}:r${rows}b${g.bpr}:g2`, p, [wBuf, xBuf, y, params], dx, dy, 1);
      hechos.matmul++;
      return y;
    },

    // silu(W_gate·x)·(W_up·x) en un despacho: las dos matrices con las mismas
    // filas y la misma K, y cada grupo hace las mismas R filas de las dos.
    async gateUp(tipo, wGate, wUp, xBuf, rows, k, tokens = 1) {
      const g = prepara(tipo, k, tokens, [wGate, wUp], [rows, rows]);
      if (!g) return null;
      if ((limites.maxStorageBuffersPerShaderStage || 8) < 4) return null;
      const forma = gemv2Forma({ tipo, filas: [rows], k, T: tokens, nm: 2,
        maxCompartida: limites.maxComputeWorkgroupStorageSize || 16384 });
      const o = { tipo, R: forma.R, NSG: forma.NSG, T: tokens, modo: 'swiglu', n: 2, u4x8: u4x8(), cola: g.cola };
      const p = await pipeline(o);
      if (!p) return null;
      const y = ctx.alloc(rows * tokens * 4, 'swiglu.out');
      const grupos = rows / forma.R;
      // Una palabra de escalas por matriz (gate en .x, up en .y): dos tensores
      // de la misma forma la tienen igual, pero no se da por supuesto.
      const params = ctx.uniform([gemv2Params({ k, bpr: g.bpr, pasos: g.pasos, filas: [rows],
        esc: g.esc, fin: [grupos] })]);
      const [dx, dy] = dims(grupos);
      await despacha(`matmul:${tipo}:gateup:r${rows}b${g.bpr}`, p, [wGate, wUp, xBuf, y, params], dx, dy, 1);
      hechos.gateUp++;
      return y;
    },

    // Varias matrices (2 a 4) con la misma entrada y la misma K, cada una con su
    // número de filas y su salida, en un despacho. Devuelve las salidas en el
    // mismo orden que los pesos.
    async secciones(tipo, ws, xBuf, filas, k, tokens = 1) {
      const n = ws.length;
      if (n < 2 || n > 4 || filas.length !== n) return null;
      // n pesos + x + n salidas: con cuatro matrices son nueve búferes de
      // almacenamiento, más de lo que garantiza el mínimo de WebGPU.
      if ((limites.maxStorageBuffersPerShaderStage || 8) < 2 * n + 1) return null;
      const g = prepara(tipo, k, tokens, ws, filas);
      if (!g) return null;
      const forma = gemv2Forma({ tipo, filas, k, T: tokens, nm: 1,
        maxCompartida: limites.maxComputeWorkgroupStorageSize || 16384 });
      const o = { tipo, R: forma.R, NSG: forma.NSG, T: tokens, modo: 'secciones', n, u4x8: u4x8(), cola: g.cola };
      const p = await pipeline(o);
      if (!p) return null;
      const ys = filas.map((f) => ctx.alloc(f * tokens * 4, 'matmul.y'));
      const fin = [];
      let acum = 0;
      for (const f of filas) { acum += f / forma.R; fin.push(acum); }
      const params = ctx.uniform([gemv2Params({ k, bpr: g.bpr, pasos: g.pasos, filas,
        esc: g.esc, fin })]);
      const [dx, dy] = dims(acum);
      await despacha(`matmul:${tipo}:secciones:r${filas.join('+')}b${g.bpr}`, p, [...ws, xBuf, ...ys, params], dx, dy, 1);
      hechos.secciones++;
      return ys;
    },

    // Diagnóstico para los bancos; el forward no lo usa.
    conMarca: !!compruebaR,                       // ¿comprueba la marca de los reordenados?
    despachados: () => ({ ...hechos }),           // despachos hechos por operación
    rechazados: () => [...rechazadas.keys()],     // claves que el compilador rechazó
  };
}
