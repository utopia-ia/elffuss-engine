// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · operaciones FUSIONADAS sobre el contexto de siempre.
// ─────────────────────────────────────────────────────────────────────────────
// Las cadenas de despachos pequeños que `kernels.fusion.wgsl.js` junta en uno,
// envueltas igual que las de `ops.js`: pipeline cacheado por clave, un uniforme
// por llamada, buffers del pozo del contexto y el despacho genérico
// (`ops.dispatchAny`). No toca `ops.js`: se monta encima de lo que ya expone.
//
// Interruptor: `globalThis.__ELFFUSS_FUSION === true`, APAGADO por defecto. El
// modelo lo lee una vez por pasada (`fusionActiva`); con él apagado no se llama
// a nada de aquí y el forward es el de siempre, despacho a despacho. Las claves
// de pipeline de este módulo empiezan por `fusion_` y llevan TODO lo que cambia
// el WGSL generado: dos variantes nunca comparten compilado.
//
// Cada operación devuelve buffers NUEVOS del pozo, como las que sustituye: el
// reciclado por capas de `model.js` las trata igual que a cualquier intermedio.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.

import {
  WG_FILA, CONV_TT, TIPOS_GATHER_NORMA,
  normaResidualShader, enlacesNormaResidual, gatherNormaShader, convCortaShader,
  qkvShader, enlacesQKV,
} from './kernels.fusion.wgsl.js';
import { QUANT_TYPES } from './kernels.wgsl.js';
import { TIPOS_R } from './kernels.bloques.wgsl.js';
import { compruebaBufferR } from './repack.js';
import { ROPE_NEOX, ROPE_NORM } from './kernels.nn.wgsl.js';

// El interruptor, leído en el momento: así un A/B lo alterna dentro de la misma
// sesión sin recargar nada.
export function fusionActiva() {
  return globalThis.__ELFFUSS_FUSION === true;
}

// ¿Puede F2 leer esta tabla de embeddings con el lector de bloques de siempre?
// `w` es la entrada del mapa de pesos ({ buf, t }). Lista cerrada de tipos y
// sin rejilla. Los reordenados al cargar (Q8_0R/Q4_0R) entran: llegan con su
// propio `typeName` y un descodificador con el contrato de siempre (el `blk`
// que calcula el gather y las escalas al final del buffer, que se ata entero).
// Como su lector saca del tamaño del buffer dónde empiezan las escalas,
// `gatherNorma` hace antes la MISMA comprobación de la marca que el gather
// (`compruebaBufferR`, la de repack.js, no una copia): un buffer sin la marca
// de un tensor reordenado de filas enteras lanza en los dos. Si algún día se
// marcara el peso sin cambiar el nombre de tipo, esta es la comprobación que hay
// que ampliar.
export function puedeGatherNorma(w) {
  const tipo = w && w.t && w.t.typeName;
  return !!tipo && TIPOS_GATHER_NORMA.has(tipo) && !!QUANT_TYPES[tipo] && !QUANT_TYPES[tipo].table;
}

// ¿Cubre F4 este kernel de convolución? El hilo que lleva la primera tesela
// lee el estado viejo entero y escribe el nuevo, y eso exige que la tesela
// tenga al menos K-1 tokens (ver `convCortaShader`); sin estado (K < 2) no hay
// nada que fusionar. Si no, el bloque va por la cadena de siempre.
export function puedeConvCorta(K) {
  return Number.isInteger(K) && K >= 2 && K - 1 <= CONV_TT;
}

// ¿Cubre F6 esta forma de cabeza? head_dim par y que quepa en memoria de grupo
// sobrada; pares de RoPE dentro de la cabeza. Si no, el modelo va por la cadena.
export function puedeQKV(headDim, nDims) {
  return Number.isInteger(headDim) && headDim > 0 && headDim % 2 === 0 && headDim <= 1024 &&
    Number.isInteger(nDims) && nDims >= 0 && nDims % 2 === 0 && nDims <= headDim;
}

// Cuántos buffers de almacenamiento ata F6: todos sus bindings menos el uniforme.
// Con QK-norm y `rope_freqs` (las capas globales de lfm3) el despacho único ata
// diez, y el mínimo que garantiza WebGPU por etapa son ocho: un dispositivo que
// no dé más NO puede crear ese pipeline. Ver `modoQKV`.
// Con q, k y v en UN buffer (`unBuffer`) se ata una vez: ocho en esa capa.
// `partes` cuenta solo los de un despacho del camino partido ('q' o 'kv').
export function bindingsAlmacenQKV({ norma = true, factores = false, unBuffer = false, partes = 'qkv' } = {}) {
  return enlacesQKV({ norma, factores, unBuffer, partes }).length - 1;
}

// Cómo corre F6 con un tope de buffers de almacenamiento por etapa: 'uno' (un
// despacho), 'partido' (dos: consulta, y clave con valor) o null (no cabe y va
// la cadena de siempre). `partido` fuerza (true) o prohíbe (false) el camino
// partido; sin él se parte solo si el despacho único no cabe. Las dos formas
// dan los mismos bits: cada cabeza hace las mismas cuentas en un despacho o en
// otro.
export function modoQKV(headDim, nDims, { norma = true, factores = false, unBuffer = false, partido = null } = {}, tope = 8) {
  if (!puedeQKV(headDim, nDims)) return null;
  const cabe = (partes) => bindingsAlmacenQKV({ norma, factores, unBuffer, partes }) <= tope;
  const uno = cabe('qkv'), dos = cabe('q') && cabe('kv');
  if (partido === true) return dos ? 'partido' : null;
  if (partido === false) return uno ? 'uno' : null;
  return uno ? 'uno' : dos ? 'partido' : null;
}

export function createFusionOps(ops, opts = {}) {
  const { ctx } = ops;
  // El ancho de grupo de las normas. Ver WG_FILA: tiene que ser el de
  // `createNNOps` para que el árbol de la suma sea el mismo.
  const WG = opts.wgNorm || WG_FILA;
  const WG_CONV = 64;
  const pipe = (key, code) => ctx.pipeline(key, code);
  // `dispatchAny` reparte los grupos en dos ejes pasado el tope por eje, y los
  // shaders recomponen el índice con el mismo tope (AXIS).
  const despacha = (label, p, buffers, grupos) => ops.dispatchAny(label, p, buffers, grupos);

  // El tope del dispositivo de buffers de almacenamiento por etapa (8 es lo que
  // garantiza WebGPU; muchos dan más). F1, F2 y F4 atan seis como mucho.
  const topeAlmacen = (ctx.device && ctx.device.limits && ctx.device.limits.maxStorageBuffersPerShaderStage) || 8;

  return {
    // ¿Puede ESTE dispositivo correr F6 con esta forma? La forma (ver `puedeQKV`)
    // y que sus bindings quepan en el tope, en un despacho o partido en dos (ver
    // `modoQKV`). Si no, la capa va por la cadena de siempre: más despachos, el
    // mismo resultado.
    qkvCabe(headDim, nDims, opts = {}) {
      return modoQKV(headDim, nDims, opts, topeAlmacen) !== null;
    },
    // 'uno', 'partido' o null en ESTE dispositivo (las pruebas cuentan despachos con esto).
    qkvModo(headDim, nDims, opts = {}) {
      return modoQKV(headDim, nDims, opts, topeAlmacen);
    },
    // ¿Cubre F4 este kernel de convolución? (ver `puedeConvCorta`). Si no, la
    // capa va por la cadena de siempre.
    convCabe(K) {
      return puedeConvCorta(K);
    },

    // F1. `lo` es lo que se suma al residual (salida del bloque o del FFN), `res`
    // el residual que entra, `wPost` la post-norma (null si el modelo no la
    // tiene) y `wSig` la norma de lo que viene. Devuelve el residual nuevo y la
    // entrada normalizada de la siguiente operación.
    async normaResidual(lo, res, wPost, wSig, rows, ne0, eps) {
      const post = !!wPost;
      const p = await pipe(`fusion_norma_residual_${WG}_${post ? 'post' : 'llana'}`,
        () => normaResidualShader({ wg: WG, post }));
      const xo = ctx.alloc(rows * ne0 * 4, 'fusion.res');
      const nrm = ctx.alloc(rows * ne0 * 4, 'fusion.nrm');
      // El mismo eps que `nn.rmsnorm` (f32 del número del perfil) y el cero de
      // la barrera de contracción.
      const params = ctx.uniform([new Uint32Array([ne0, rows]), new Float32Array([eps]), new Uint32Array([0])]);
      const por = { lo, res, wp: wPost, ws: wSig, xo, nrm, p: params };
      await despacha('fusion_norma_residual', p, enlacesNormaResidual(post).map((n) => por[n]), rows);
      return { res: xo, nrm };
    },

    // F2. Mismos argumentos que `quant.gather` más el peso de la norma de la
    // primera capa. Quien llama mira antes `puedeGatherNorma`.
    async gatherNorma(type, wBuf, idsBuf, wNorm, dim, tokens, eps) {
      const i = QUANT_TYPES[type];
      if (!i || !TIPOS_GATHER_NORMA.has(type)) throw new Error(`gather+norma: tipo «${type}» sin lector de bloque`);
      if (dim % i.blockElems !== 0) throw new Error(`gather+norma ${type}: dim=${dim} no es múltiplo de ${i.blockElems}`);
      // La misma guarda que `quant.gather` para los reordenados (ver arriba).
      if (TIPOS_R.has(type)) compruebaBufferR(type, wBuf, { porFila: dim / i.blockElems }, 'gather+norma');
      const p = await pipe(`fusion_gather_norma_${type}_${WG}`, () => gatherNormaShader(type, WG));
      const bpr = dim / i.blockElems;
      const x = ctx.alloc(tokens * dim * 4, 'embd');
      const nrm = ctx.alloc(tokens * dim * 4, 'fusion.nrm');
      const params = ctx.uniform([new Uint32Array([dim, tokens, bpr]), new Float32Array([eps])]);
      await despacha('fusion_gather_norma', p, [wBuf, idsBuf, wNorm, x, nrm, params], tokens);
      return { x, nrm };
    },

    // F4. `bcx` es la salida de `in_proj` [T][3E] (B | C | x), `estado` el de la
    // conv [K-1][E] —se actualiza EN SITIO, como hacía `conv_save`— y `wConv`
    // los pesos [E][K]. Devuelve C·conv(B·x) [T][E], la entrada de `out_proj`.
    async convCorta(bcx, estado, wConv, E, tokens, K) {
      // Quien llama mira antes `convCabe`; esto es para que un kernel no cubierto
      // falle aquí y diga por qué, no al crear el pipeline.
      if (!puedeConvCorta(K)) throw new Error(`conv corta: kernel de ${K} no cubierto (hace falta 2 ≤ K ≤ ${CONV_TT + 1})`);
      const p = await pipe(`fusion_conv_corta_${K}_${CONV_TT}_${WG_CONV}`,
        () => convCortaShader({ K, wg: WG_CONV, tt: CONV_TT }));
      const tiles = Math.ceil(tokens / CONV_TT);
      const y = ctx.alloc(tokens * E * 4, 'fusion.conv');
      const params = ctx.uniform([new Uint32Array([E, tokens, tiles, 0])]);
      await despacha('fusion_conv_corta', p, [bcx, wConv, estado, y, params], Math.ceil(E * tiles / WG_CONV));
      return y;
    },

    // F6. q, k y v son las proyecciones [T][…] (zancada y desplazamiento por
    // token: tres buffers sueltos o uno solo con las tres —el mismo objeto en
    // los tres argumentos—, que entonces se ata una sola vez), `wq`/`wk` las
    // QK-norm (null si el modelo no la tiene: las dos o ninguna) y `factores`
    // el `rope_freqs` (o null). Escribe K y V en la caché a partir de `nPast` y
    // devuelve q normalizada y rotada [T][H][D], la que espera `attn.scores`.
    // Un despacho si cabe en el tope de buffers del dispositivo y, si no, dos
    // (ver `modoQKV`); `partido` es para las pruebas: fuerza o prohíbe partir.
    async qkNormaRopeCache({
      q, k, v, pos, wq = null, wk = null, cacheK, cacheV,
      heads, kvHeads, headDim, tokens, nPast, nDims, mode, freqBase, freqScale = 1,
      factores = null, kvF16 = true, eps,
      qStride = heads * headDim, qOff = 0, kStride = kvHeads * headDim, kOff = 0,
      vStride = kvHeads * headDim, vOff = 0, partido = null,
    }) {
      const norma = !!wq;
      if (norma !== !!wk) throw new Error('QK-norm: o las dos normas o ninguna');
      if (!puedeQKV(headDim, nDims)) throw new Error(`QK-norm+RoPE: forma no cubierta (D=${headDim}, nDims=${nDims})`);
      const unBuffer = q === k && k === v;
      const conFf = !!factores;
      const modo = modoQKV(headDim, nDims, { norma, factores: conFf, unBuffer, partido }, topeAlmacen);
      if (!modo) {
        throw new Error(`QK-norm+RoPE: no cabe ${partido === true ? 'partido' : partido === false ? 'en un despacho' : 'ni en un despacho ni partido'}` +
          ` con ${topeAlmacen} buffers de almacenamiento por etapa`);
      }
      if (mode !== ROPE_NEOX && mode !== ROPE_NORM) throw new Error(`RoPE: modo ${mode} desconocido`);
      const qo = ctx.alloc(tokens * heads * headDim * 4, 'fusion.q');
      // theta_scale en el HOST, con la misma cuenta que `nn.rope`: el redondeo a
      // f32 de este número es parte del resultado.
      const thetaScale = Math.pow(freqBase, -2 / nDims);
      const por = { q, k, v, qkv: q, pos, wq, wk, qo, kc: cacheK, vc: cacheV, ff: factores };
      // [parte, primera ranura, ranuras por token], en el orden [consulta | clave | valor].
      const partes = modo === 'uno' ? [['qkv', 0, heads + 2 * kvHeads]]
        : [['q', 0, heads], ['kv', heads, 2 * kvHeads]];
      for (const [parte, s0, nRanuras] of partes) {
        // La parte de consulta no toca la caché: su texto (y su clave) no
        // dependen de la precisión de la caché.
        const h16 = kvF16 && parte !== 'q';
        const p = await pipe(`fusion_qkv_${headDim}_${WG}${norma ? '_n' : ''}${conFf ? '_ff' : ''}${unBuffer ? '_u1' : ''}` +
          `${parte === 'qkv' ? '' : '_p' + parte}${h16 ? '_h' : ''}`,
        () => qkvShader({ D: headDim, norma, factores: conFf, kvF16: h16, wg: WG, unBuffer, partes: parte }));
        const params = ctx.uniform([
          new Uint32Array([heads, kvHeads, tokens, headDim, nDims, mode, nPast, 0,
            qStride, qOff, kStride, kOff, vStride, vOff]),
          new Float32Array([eps, thetaScale, freqScale]),
          new Uint32Array([s0, nRanuras, 0]),
        ]);
        await despacha(parte === 'qkv' ? 'fusion_qkv' : `fusion_qkv:${parte}`, p,
          enlacesQKV({ norma, factores: conFf, unBuffer, partes: parte }).map((n) => (n === 'p' ? params : por[n])),
          tokens * nRanuras);
      }
      return qo;
    },

    // Para el reciclado por capas de `model.js`. La norma que una capa deja hecha
    // para la SIGUIENTE (F1/F2) se reserva antes de la marca de esa siguiente
    // capa, y `ctx.recycleDesde` solo mira de la marca en adelante: sin esto,
    // cada una se quedaría viva hasta el final de la pasada y el pico de memoria
    // de un prompt crecería una fila de activaciones por capa. La mueve justo
    // antes de la marca y devuelve la marca bajada en uno, para que muera al
    // acabar la capa que la consume, como la norma que sustituye.
    // Se llama al EMPEZAR la capa, antes de reservar nada en ella.
    reciclableDesde(buf, marca) {
      const s = ctx.scratch;
      for (let k = Math.min(marca, s.length) - 1; k >= 0; k--) {
        if (s[k].buf !== buf) continue;
        if (k !== marca - 1) { const [e] = s.splice(k, 1); s.splice(marca - 1, 0, e); }
        return marca - 1;
      }
      return marca;
    },
  };
}
