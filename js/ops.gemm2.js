// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · despacho del GEMM del prompt con descuantización dentro (GEMM2).
// ─────────────────────────────────────────────────────────────────────────────
// Lo llama `quant.matmul` (ops.js) al principio, y SOLO si `__ELFFUSS_GEMM2 ===
// true`: con el interruptor apagado no se ejecuta ni una línea de aquí y el motor
// es el de siempre. Si el plan no aplica a la forma pedida devuelve null y
// `quant.matmul` sigue por su camino de siempre.
//
// Interruptores (todos APAGADOS por defecto; los sub-interruptores solo cuentan
// con el principal puesto):
//   __ELFFUSS_GEMM2         el GEMM exacto sustituye a DRIFT: mismos bits, sin
//                           el búfer de pesos f16 ni su despacho.
//   __ELFFUSS_GEMM2_TESELA  la tesela con memoria compartida en vez del exacto
//                           (tolerancia declarada, no bit a bit).
//   __ELFFUSS_GEMM2_ACC16   dentro de la tesela, suma en f16 por tesela de k.
//   __ELFFUSS_GEMM2_FMA32   el kernel de la tesela de DRIFT con la aritmética de
//                           la tesela (FMA f32 por producto): no bit a bit, misma
//                           tolerancia que la tesela; sin efecto con TESELA puesto.
//   __ELFFUSS_GEMM2_PARTE   formas largas y estrechas (k > filas, como ffn_down):
//                           bloques de 8 filas en el exacto (sigue exacto) o k
//                           partida en la tesela (tolerancia de la tesela).
//   __ELFFUSS_GEMM2_Q4      Q4_0/Q4_0R por el GEMM teselado de 4 a 63 tokens.
//   __ELFFUSS_GEMM2_COLA    última capa del prompt solo para el último token
//                           (lo usa model.js a través de `cola`).
//   __ELFFUSS_GEMM2_FORMA   {BM, BN, TM, TN, doble} para barrer la tesela.
//   __ELFFUSS_GEMM2_TROZO   {nb, doble, tpw, tpt} para barrer el exacto. `tpw` y
//                           `tpt` cambian SOLO su tesela (la de DRIFT sigue con
//                           sus `__ELFFUSS_DRIFT_TPW/TPT`); sigue siendo exacto
//                           con cualquier forma, porque cada salida suma lo mismo
//                           en el mismo orden. Más hilos por grupo reparten la
//                           descuantización de cada trozo entre más tokens.
// Todo lo que cambia el WGSL entra en la clave del pipeline: si no, un A/B
// mediría dos veces el mismo shader compilado.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.
import {
  GEMM2_TIPOS, formaTesela, gemm2ExactoShader, gemm2TeselaShader,
  gemm2SumaShader, TIPOS_Q4_TESELA, gemmQ4Shader,
} from './kernels.gemm2.wgsl.js';
import { f32aF16Shader, DRIFT_TPT, DRIFT_TPW } from './kernels.drift.wgsl.js';
import { TK_POR_DEFECTO } from './kernels.gemm.wgsl.js';
import { AXIS } from './kernels.wgsl.js';
import { TIPOS_R, escalasDesde } from './kernels.bloques.wgsl.js';
import { compruebaBufferR } from './repack.js';

// Una sola lectura de los interruptores por decisión. Los del camino al que
// sustituye (DRIFT_DESDE, DRIFT, GEMM, la forma de DRIFT) se leen con la MISMA
// expresión que en ops.js: GEMM2 elige lo mismo que elegiría él, no un rango
// propio que se desincronice.
export function leeInterruptores(g = globalThis) {
  return {
    gemm2: g.__ELFFUSS_GEMM2 === true,
    tesela: g.__ELFFUSS_GEMM2_TESELA === true,
    acc16: g.__ELFFUSS_GEMM2_ACC16 === true,
    fma32: g.__ELFFUSS_GEMM2_FMA32 === true,
    parte: g.__ELFFUSS_GEMM2_PARTE === true,
    q4: g.__ELFFUSS_GEMM2_Q4 === true,
    forma: g.__ELFFUSS_GEMM2_FORMA || null,
    trozo: g.__ELFFUSS_GEMM2_TROZO || {},
    driftDesde: g.__ELFFUSS_DRIFT_DESDE || 64,
    driftApagado: g.__ELFFUSS_DRIFT === false,
    gemmApagado: g.__ELFFUSS_GEMM === false,
    tpt: g.__ELFFUSS_DRIFT_TPT || DRIFT_TPT,
    tpw: g.__ELFFUSS_DRIFT_TPW || DRIFT_TPW,
    formaGemm: g.__ELFFUSS_GEMM_FORMA || {},
    pad: g.__ELFFUSS_PAD_TESELA === 1 ? 1 : 0,
    llano: g.__ELFFUSS_GEMM_PLANO === true ? 1 : 0,
  };
}

// La tesela estrecha del GEMM teselado según el lote: la MISMA tabla que el
// camino GEMM de `quant.matmul` (ops.js). Está copiada a propósito, para no
// tocar aquel bloque; si allí cambia, aquí hay que cambiarlo también.
const anchoTesela = (n) => (n <= 4 ? { TM: 2, TN: 1, WGM: 16, WGN: 4 }
  : n <= 8 ? { TM: 2, TN: 1, WGM: 8, WGN: 8 }
  : n <= 16 ? { TM: 2, TN: 2, WGM: 8, WGN: 8 }
  : { TM: 2, TN: 4, WGM: 8, WGN: 8 });

// Qué camino toma un matmul. Función pura (sin WebGPU) para poder probarla en Node.
// Devuelve null si GEMM2 no aplica y `quant.matmul` debe seguir como siempre.
export function planGemm2({ type, rows, k, tokens, f16 = true, maxCompartida = 32768, sw = leeInterruptores() }) {
  if (!sw.gemm2 || !GEMM2_TIPOS.has(type) || k % 32 !== 0 || !(tokens >= 1)) return null;
  const bpr = k / 32;
  // El rango de DRIFT con su mismo predicado (ops.js): lote largo, f16 en el
  // dispositivo, filas múltiplo de 16. GEMM2 lo sustituye; no lo amplía.
  if (tokens >= sw.driftDesde && !sw.driftApagado && f16 && rows % 16 === 0) {
    // Forma «larga y estrecha»: k mayor que las filas. Con la tesela de siempre
    // salen pocos grupos con bucles largos (ffn_down: la cuarta parte de grupos
    // que gate/up con bucles cuatro veces más largos).
    const larga = sw.parte && k > rows;
    if (sw.tesela) {
      const F = formaTesela(sw.forma || {});
      if (rows % F.BM === 0 && F.HILOS <= 256 && F.compartida <= maxCompartida) {
        // k partida en tantas partes como veces cabe `rows` en k (así cada grupo
        // hace una cantidad de trabajo parecida a la de las formas anchas), con
        // al menos cuatro teselas de k por parte.
        const partes = larga ? Math.max(1, Math.min(8, Math.ceil(k / rows), Math.floor(bpr / 4))) : 1;
        return { via: 'tesela', type, bpr, forma: F, acc16: sw.acc16, partes };
      }
    }
    // Dos bloques por trozo si la fila lo permite (menos barreras por k) y un
    // solo trozo en memoria compartida (ver `gemm2ExactoShader`). Un trozo de 2
    // con un número impar de bloques dejaría el último sin calcular: se baja a 1.
    const nb = (sw.trozo.nb === 1 || bpr % 2 !== 0) ? 1 : 2;
    return {
      via: 'exacto', type, bpr, tpt: (sw.trozo.tpt | 0) || sw.tpt, tpw: (sw.trozo.tpw | 0) || sw.tpw,
      filas: larga ? 8 : 16, nb, doble: sw.trozo.doble === true, fma32: sw.fma32,
    };
  }
  // Q4_0/Q4_0R por el GEMM teselado, con las mismas reglas que Q8_0 allí.
  if (sw.q4 && TIPOS_Q4_TESELA.has(type) && tokens >= 4 && !sw.gemmApagado) {
    const Fg = sw.formaGemm;
    const TK = (Fg.TK | 0) || TK_POR_DEFECTO;
    if (TK % 4 !== 0 || 32 % TK !== 0 || k % TK !== 0) {
      // Igual que ops.js: un TK pedido a mano que no vale LANZA; nunca se mide
      // otro camino con la etiqueta de este.
      if (Fg.TK) throw new Error(`GEMM: TK=${Fg.TK} no vale para «${type}» (bloque 32, k=${k})`);
      return null;
    }
    const A = anchoTesela(tokens);
    const G = { TM: Fg.TM || A.TM, TN: Fg.TN || A.TN, WGM: Fg.WGM || A.WGM, WGN: Fg.WGN || A.WGN, TK, maxCompartida };
    return { via: 'q4', type, bpr, G, pad: sw.pad, llano: sw.llano };
  }
  return null;
}

// Clave del pipeline de cada plan: TODO lo que cambia el WGSL generado.
export function clavePlan(pl) {
  if (pl.via === 'exacto') return `gemm2x_${pl.type}_${pl.tpt}_${pl.tpw}_f${pl.filas}_n${pl.nb}_d${pl.doble ? 1 : 0}_m${pl.fma32 ? 1 : 0}`;
  if (pl.via === 'tesela') {
    const F = pl.forma;
    return `gemm2t_${pl.type}_${F.BM}_${F.BN}_${F.TM}_${F.TN}_d${F.doble ? 1 : 0}_a${pl.acc16 ? 1 : 0}_p${pl.partes}`;
  }
  const G = pl.G;
  return `gemm_${pl.type}_${G.TM}_${G.TN}_${G.WGM}_${G.WGN}_k${G.TK}_p${pl.pad}_f${pl.llano}_g2q4`;
}

const rowsToDims = (rows) => (rows <= AXIS ? [rows, 1] : [AXIS, Math.ceil(rows / AXIS)]);

// `dispatch`, `dispatchXYZ` y `dispatchN` son los de ops.js (se pasan para que
// estos despachos vayan por el mismo sitio que todos: lote, perfilado y scopes).
export function createGemm2Ops(ctx, { dispatch, dispatchXYZ, dispatchN }) {
  const device = ctx.device;
  const get = ctx.pipeline;
  const conF16 = () => !!device.features?.has('shader-f16');
  const maxComp = () => device.limits?.maxComputeWorkgroupStorageSize || 16384;
  const plan = (type, rows, k, tokens) =>
    planGemm2({ type, rows, k, tokens, f16: conF16(), maxCompartida: maxComp() });

  // ── x → f16, con memoria ───────────────────────────────────────────────────
  // La misma activación entra en varios matmuls seguidos (la salida de la norma
  // va a q, k, v y la puerta; la del FFN a gate y up) y DRIFT la convertía otra
  // vez en cada uno. La conversión es la misma operación sobre los mismos datos,
  // así que reutilizarla no cambia ni un bit.
  //
  // La clave es la MARCA DE REPARTO (`ctx.marcaDe`), no el búfer: un búfer
  // reciclado recibe marca nueva al volver a entregarse, así que no puede
  // confundirse con la activación que fue antes. Y el resultado guardado solo
  // vale mientras SU marca siga siendo la misma: si el pozo lo ha vuelto a
  // entregar, alguien lo está reescribiendo. Mientras no lo haya entregado, su
  // contenido sigue intacto aunque esté en el pozo (nadie lo ha pedido), y el
  // despacho que lo lee se graba antes que cualquiera que lo pida después.
  // Supuesto, el mismo que la memoria de rotación de model.js: nadie reescribe
  // en el sitio un búfer del pozo que ya entregó.
  //
  // Y solo se reutiliza una conversión que siga PRESTADA. Una que ya volvió al
  // pozo conserva su contenido mientras nadie la pida, pero el mismo matmul que
  // la reutiliza pide después su búfer de salida, y si mide lo mismo el pozo
  // puede entregarle justo ese: leer y escribir el mismo búfer en un despacho
  // no es válido, y sin los scopes de error eso no lanza, deja la salida sin
  // escribir. Con el flujo de model.js no ocurre (las conversiones se reutilizan
  // dentro de la misma capa, antes de reciclar), pero comprobarlo es barato y
  // solo corre en el prompt.
  const prestado = (buf) => ctx.scratch.some((e) => e.buf === buf);
  const memo = [];
  async function aF16(xBuf, tokens, k) {
    const n4 = tokens * k / 4;
    const marca = ctx.marcaDe?.get(xBuf);
    if (marca !== undefined) {
      for (let i = memo.length - 1; i >= 0; i--) {
        const m = memo[i];
        if (m.marca === marca && m.n4 === n4 && ctx.marcaDe.get(m.xh) === m.marcaXh && prestado(m.xh)) return m.xh;
      }
    }
    // El mismo shader y la misma clave que usa DRIFT: es la misma conversión.
    const px = await get('drift_x16', () => f32aF16Shader(256));
    const xh = ctx.alloc(n4 * 8, 'gemm2.x');
    await dispatchN(ctx, 'gemm2_x16', px, [xBuf, xh, ctx.uniform([new Uint32Array([n4, 0, 0, 0])])],
      Math.ceil(n4 / 256));
    if (marca !== undefined) {
      memo.push({ marca, n4, xh, marcaXh: ctx.marcaDe.get(xh) });
      if (memo.length > 4) memo.shift();
    }
    return xh;
  }

  // Bloques del TENSOR que guarda `wBuf`. En los reordenados (Q8_0R, Q4_0R) las
  // escalas empiezan tras los cuantos de TODOS sus bloques, y una llamada puede
  // pedir menos filas de las que tiene el tensor (la cabeza se pide con el
  // vocabulario del tokenizador): el `rows·bpr` de la llamada no sirve. Sale de
  // la marca del cargador, con la MISMA comprobación que hace el camino de
  // siempre (`compruebaBufferR`: marca, tipo y que la llamada no se salga del
  // tensor), y lanza igual que él: el interruptor no cambia qué búferes se
  // aceptan. En los formatos del GGUF no hay región de escalas aparte.
  const bloquesTensor = (type, wBuf, rows, bpr) => (TIPOS_R.has(type)
    ? compruebaBufferR(type, wBuf, { bloques: rows * bpr }, 'gemm2')
    : rows * bpr);
  // La palabra donde empiezan las escalas: la de `escalasDesde` con el N del
  // tensor; 0 en los del GGUF (el kernel no la lee).
  const baseEscalasDe = (type, n) => (TIPOS_R.has(type) ? escalasDesde(type, n) : 0);

  async function exacto(pl, wBuf, xh, rows, k, tokens, baseEsc) {
    const { type, tpt, tpw, filas, nb, doble, fma32, bpr } = pl;
    const pg = await get(clavePlan(pl), () => gemm2ExactoShader({ type, tpt, tpw, filas, nb, doble, fma32 }));
    const y = ctx.alloc(rows * tokens * 4, 'matmul.y');
    const og = rows / filas;
    await dispatchXYZ(ctx, `gemm2x:${type}:r${rows}b${bpr}`, pg,
      [wBuf, xh, y, ctx.uniform([new Uint32Array([k / 4, rows, tokens, bpr, baseEsc, 0, 0, 0])])],
      Math.ceil(tokens / (tpw * tpt)), Math.min(og, 65535), Math.ceil(og / 65535));
    return y;
  }

  async function tesela(pl, wBuf, xh, rows, k, tokens, baseEsc) {
    const { type, forma: F, acc16, partes, bpr } = pl;
    const pg = await get(clavePlan(pl), () => gemm2TeselaShader({
      type, BM: F.BM, BN: F.BN, TM: F.TM, TN: F.TN, doble: F.doble, acc16, partes, maxCompartida: maxComp(),
    }));
    const n = rows * tokens;
    const dst = ctx.alloc(n * 4 * partes, partes > 1 ? 'gemm2.partes' : 'matmul.y');
    const params = ctx.uniform([new Uint32Array([rows, tokens, bpr, k / 4, baseEsc, 0, 0, 0])]);
    // Teselas de TOKENS en el eje x: los grupos que el dispositivo lanza seguidos
    // comparten la tesela de pesos, que se lee de memoria una vez y el resto de
    // veces sale de la caché.
    await dispatchXYZ(ctx, `gemm2t:${type}:r${rows}b${bpr}`, pg, [wBuf, xh, dst, params],
      Math.ceil(tokens / F.BN), rows / F.BM, partes);
    if (partes === 1) return dst;
    const ps = await get(`gemm2s_${partes}`, () => gemm2SumaShader(partes, 256));
    const y = ctx.alloc(n * 4, 'matmul.y');
    const n4 = n / 4;
    await dispatchN(ctx, `gemm2s:r${rows}`, ps, [dst, y, ctx.uniform([new Uint32Array([n4, 0, 0, 0])])],
      Math.ceil(n4 / 256));
    return y;
  }

  async function q4(pl, wBuf, xBuf, rows, k, tokens, nTensor) {
    const { type, G, bpr } = pl;
    const pg = await get(clavePlan(pl), () => gemmQ4Shader(type, G));
    const y = ctx.alloc(rows * tokens * 4, 'matmul.y');
    // `nBlocks` solo lo lee la tirada de Q4_0R, y como base de sus escalas
    // (4 · nBlocks): tienen que ser los bloques del TENSOR, no los de la llamada.
    const params = ctx.uniform([new Uint32Array([nTensor, bpr, rows, k, tokens, 0, 0, 0])]);
    const [gx, gy] = rowsToDims(Math.ceil(rows / (G.WGM * G.TM)));
    await dispatch(ctx, `gemm:${type}:r${rows}b${bpr}`, pg, [wBuf, xBuf, y, params],
      gx, gy, Math.ceil(tokens / (G.WGN * G.TN)));
    return y;
  }

  return {
    // Diagnóstico: qué haría con esta forma (null = el camino de siempre).
    plan,
    // Para los bancos: vacía la memoria de x→f16, de modo que una medida no se
    // ahorre conversiones que en la otra rama sí se pagan.
    olvida() { memo.length = 0; },

    async matmul(type, wBuf, xBuf, rows, k, tokens) {
      // El decode (un token) nunca es de GEMM2 salvo que alguien baje
      // DRIFT_DESDE a mano: se sale aquí sin leer el resto de interruptores, que
      // en el bucle de decode serían cientos de lecturas por token para nada.
      if (tokens < 4 && tokens < (globalThis.__ELFFUSS_DRIFT_DESDE || 64)) return null;
      const pl = plan(type, rows, k, tokens);
      if (!pl) return null;
      const n = bloquesTensor(type, wBuf, rows, pl.bpr);
      if (pl.via === 'q4') return await q4(pl, wBuf, xBuf, rows, k, tokens, n);
      const xh = await aF16(xBuf, tokens, k);
      const baseEsc = baseEscalasDe(type, n);
      return pl.via === 'tesela'
        ? await tesela(pl, wBuf, xh, rows, k, tokens, baseEsc)
        : await exacto(pl, wBuf, xh, rows, k, tokens, baseEsc);
    },

    // ── La cola de la última capa (interruptor GEMM2_COLA, ver model.js) ──────
    // `x1` es UNA fila: la del último token de un lote de `lote` tokens. Se
    // calcula con el MISMO plan (mismo shader, misma forma, mismas partes de k)
    // que habría tocado al lote entero, despachado para un solo token. En los dos
    // kernels de aquí la salida de un token solo depende de su fila de x y de los
    // pesos, sumados en el mismo orden, así que da los mismos bits que su fila
    // dentro del lote. Con otro camino (el matvec de un token suma en otro
    // orden) eso ya no se cumple: por eso `reproducible` dice antes si se puede.
    reproducible(type, rows, k, lote) {
      const pl = plan(type, rows, k, lote);
      return !!pl && (pl.via === 'exacto' || pl.via === 'tesela');
    },
    async cola(type, wBuf, x1, rows, k, lote) {
      const pl = plan(type, rows, k, lote);
      if (!pl || (pl.via !== 'exacto' && pl.via !== 'tesela')) return null;
      const baseEsc = baseEscalasDe(type, bloquesTensor(type, wBuf, rows, pl.bpr));
      const xh = await aF16(x1, 1, k);
      return pl.via === 'tesela'
        ? await tesela(pl, wBuf, xh, rows, k, 1, baseEsc)
        : await exacto(pl, wBuf, xh, rows, k, 1, baseEsc);
    },
  };
}
