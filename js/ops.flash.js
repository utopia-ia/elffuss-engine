// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · la atención fusionada como operación: elegir la forma del
// kernel, comprobar que el dispositivo la admite y despacharlo.
// ─────────────────────────────────────────────────────────────────────────────
// Sustituye, cuando se enciende `__ELFFUSS_FLASH`, a la cadena
//   attn.scores → nn.softmax → attn.combine (partido + reducción) → mul_cabeza
// por uno o dos despachos (`kernels.flash.wgsl.js`). Con el interruptor apagado
// este módulo ni se llama: el forward es el de siempre, mismos shaders y mismos
// despachos.
//
// La decisión de la forma está en `planFlash`, una función PURA (no toca
// WebGPU): se puede llamar desde Node para ver qué compilaría cada modelo y
// validar justo esos shaders sin dispositivo.
//
// Si la forma no se puede atender —subgrupos que no son de 32, caché en f32,
// otra dimensión de cabeza, memoria de grupo insuficiente— `puede()` dice que no
// y el forward usa la cadena de siempre, que da lo mismo más despacio. Un
// modelo con otra forma tiene que seguir corriendo. Y lo mismo si el
// dispositivo NO COMPILA un kernel que el plan sí admitía: `atencion` devuelve
// null (la cadena hace esa capa), la clave queda apartada y `puede()` dice que
// no a esa forma desde entonces, con un aviso en consola una sola vez.
//
// Interruptores de MEDIDA (solo para elegir defectos; apagados no cambian
// nada, y todo lo que toca el texto del shader va en la clave del pipeline):
//   __ELFFUSS_FLASH_TQ      consultas por grupo en prefill
//   __ELFFUSS_FLASH_NSG     subgrupos por grupo
//   __ELFFUSS_FLASH_QSH     q en memoria de grupo (true) o leída de global (false)
//   __ELFFUSS_FLASH_VSH     tesela de V en memoria de grupo
//   __ELFFUSS_FLASH_PSH     probabilidades a P·V por memoria de grupo
//   __ELFFUSS_FLASH_PREF    prelectura de la tesela de K siguiente
//   __ELFFUSS_FLASH_MEM     presupuesto de memoria de grupo
//   __ELFFUSS_FLASH_PARTES  partes del rango de claves, forzadas
//   __ELFFUSS_FLASH_GRUPOS  grupos que se persiguen antes de partir
//   __ELFFUSS_FLASH_CLAVES  claves por parte al partir (y se parte desde el doble)
// Y uno de PRUEBA:
//   __ELFFUSS_FLASH_EXIGE   con el interruptor puesto, una capa que no pueda ir
//                           por la fusionada LANZA en vez de volver a la cadena:
//                           así una prueba del modelo entero no da por buena la
//                           fusionada habiendo corrido la cadena sin saberlo.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.
import { flashShader, flashReduceShader, flashMemoria, flashPrefetch, FLASH_BK } from './kernels.flash.wgsl.js';

// Memoria de grupo que se permite por defecto, aunque el dispositivo admita más.
// No es un tope del dispositivo sino de ocupación: cuanto más pide un grupo,
// menos grupos caben a la vez en una unidad de cómputo, y con pocos grupos no
// hay con qué esconder la latencia de memoria. Es además el mínimo que la
// especificación garantiza, así que con el presupuesto por defecto la forma
// cabe en cualquier dispositivo. `__ELFFUSS_FLASH_MEM` lo cambia para medir.
const PRESUPUESTO = 16384;

// Cuántos grupos se persiguen antes de partir el rango de claves (y pagar la
// reducción). Con una consulta (decode) hay solo KVH grupos sin partir: casi
// todo el dispositivo parado mientras unos pocos recorren el contexto en serie.
const GRUPOS_OBJETIVO = 128;
// Claves por parte cuando se parte (dos teselas), y no se parte un rango de
// menos del doble: con tres o cuatro teselas, lo que se ahorraría en el camino
// más largo es poco al lado de un despacho más (la reducción) y la espera entre
// los dos. Dónde está de verdad el punto lo dice el barrido
// (`__ELFFUSS_FLASH_CLAVES`), no esta cuenta.
const CLAVES_MIN = 2 * FLASH_BK;
const PARTES_MAX = 32;

// Lee un entero positivo de un interruptor de medida, o null.
const entero = (v) => (Number.isInteger(v) && v > 0 ? v : null);

// La forma del kernel para una llamada. Devuelve { ok:false, motivo } si no se
// puede, y si se puede, todo lo que hace falta para compilar y despachar.
// `dev` = { subgroupSize, limites } con los límites REALES del dispositivo
// (device.limits), no los pedidos.
export function planFlash(cfg, dev) {
  const { headDim: D, heads: H, kvHeads: KVH, tokens: T, nPast = 0, ventana = 0,
    kvF16 = false, puerta = false } = cfg;
  const sg = dev?.subgroupSize || 0;
  const lim = dev?.limites || {};
  const no = (motivo) => ({ ok: false, motivo });
  // El reparto (un carril por clave de la tesela; cada subgrupo, 32 hilos
  // consecutivos del grupo) está escrito para subgrupos de 32 exactos.
  if (sg !== 32) return no(`subgrupo de ${sg} (hace falta 32 fijo)`);
  if (!kvF16) return no('la caché KV no es f16');
  if (D !== 128 && D !== 256) return no(`headDim ${D} sin kernel`);
  if (!Number.isInteger(H) || !Number.isInteger(KVH) || KVH < 1 || H % KVH !== 0) {
    return no(`${H} cabezas no se reparten entre ${KVH} KV`);
  }
  if (!Number.isInteger(T) || T < 1 || !Number.isInteger(nPast) || nPast < 0) return no('tokens o nPast inválidos');
  const G = H / KVH;

  // Consultas por grupo. En prefill, las suficientes para que el grupo tenga
  // del orden de 16 filas (G·TQ): cada tesela de K se lee una vez para todas.
  // En decode hay una.
  let TQ = T === 1 ? 1 : Math.max(1, Math.floor(16 / G));
  const tqPedido = entero(globalThis.__ELFFUSS_FLASH_TQ);
  if (T > 1 && tqPedido) TQ = tqPedido;
  TQ = Math.min(TQ, T);
  const R = G * TQ;
  // Subgrupos por grupo: el mayor divisor de las filas que no pase de 4.
  // `__ELFFUSS_FLASH_NSG` lo fuerza para medir (p. ej. 1 en decode: un solo
  // subgrupo con todas las filas, que lee la V una vez y no una por subgrupo).
  let NSG = [4, 3, 2, 1].find((d) => R % d === 0);
  const nsgPedido = entero(globalThis.__ELFFUSS_FLASH_NSG);
  if (nsgPedido) {
    if (R % nsgPedido !== 0) return no(`${R} filas no se reparten entre ${nsgPedido} subgrupos`);
    NSG = nsgPedido;
  }
  const RS = R / NSG;
  const WG = 32 * NSG;
  // Cada fila de un subgrupo son registros vivos durante todo el bucle (su
  // salida parcial, su máximo y su suma): por encima de 8 se derramarían.
  if (RS > 8) return no(`${RS} filas por subgrupo (máximo 8)`);
  const maxInv = lim.maxComputeInvocationsPerWorkgroup ?? 256;
  if (WG > maxInv) return no(`grupo de ${WG} hilos sobre un tope de ${maxInv}`);
  const maxSB = lim.maxStorageBuffersPerShaderStage ?? 8;
  if (maxSB < 5) return no(`${maxSB} buffers de almacenamiento por etapa`);

  const maxMem = lim.maxComputeWorkgroupStorageSize ?? 16384;
  const memPedida = entero(globalThis.__ELFFUSS_FLASH_MEM);
  const tope = Math.min(maxMem, memPedida || PRESUPUESTO);
  if (flashMemoria({ D, G, TQ, qsh: false, vsh: false }) > maxMem) {
    return no(`la tesela de K no cabe en ${maxMem} bytes de memoria de grupo`);
  }
  // Probabilidades a P·V por memoria de grupo en vez de por barajado, solo si se
  // pide y hay más de una fila por subgrupo (con una, un vec4 no ahorra nada).
  const psh = globalThis.__ELFFUSS_FLASH_PSH === true && RS >= 2;
  // q en memoria de grupo si cabe en el presupuesto; si no, en difusión desde
  // global. Los dos interruptores fuerzan una u otra para medir.
  let qsh = flashMemoria({ D, G, TQ, qsh: true, vsh: false, psh, NSG }) <= tope;
  if (globalThis.__ELFFUSS_FLASH_QSH === true) qsh = true;
  if (globalThis.__ELFFUSS_FLASH_QSH === false) qsh = false;
  // V en memoria de grupo solo si se pide. Por defecto SOLO la K pasa por
  // memoria de grupo: cada subgrupo lee de la caché las filas de V de la tesela
  // (los 32 carriles en un tramo contiguo), así que con NSG subgrupos la misma
  // tesela de V se pide NSG veces —lecturas repetidas y casi a la vez, que en
  // general sirve la caché del dispositivo y no la memoria—. Pasarla una vez por
  // memoria de grupo cuesta otra tesela entera, que con q dentro no cabe en el
  // presupuesto (habría que leer q de global), y una barrera más de copia. Cuál
  // gana no se deduce: lo mide el barrido (`vsh` con q fuera, y en decode
  // NSG = 1, que tiene un solo subgrupo y lee la V una vez).
  const vsh = globalThis.__ELFFUSS_FLASH_VSH === true;
  // Prelectura de la tesela de K a registros, solo si se pide y el reparto lo
  // admite (cuesta registros durante todo el cálculo de la tesela).
  const pref = globalThis.__ELFFUSS_FLASH_PREF === true && flashPrefetch({ D, NSG }) > 0;
  const memoria = flashMemoria({ D, G, TQ, qsh, vsh, psh, NSG });
  if (memoria > maxMem) return no(`${memoria} bytes de memoria de grupo sobre un tope de ${maxMem}`);

  // Partes del rango de claves. El rango más largo es el de la última tesela de
  // consultas (la causal llega más lejos); con ventana, todos rondan ventana+TQ.
  const qTiles = Math.ceil(T / TQ);
  const qIniUlt = nPast + (qTiles - 1) * TQ;
  const kIniUlt = ventana > 0 ? Math.max(0, qIniUlt + 1 - ventana) : 0;
  const span = nPast + T - kIniUlt;
  const grupos = KVH * qTiles;
  const objetivo = entero(globalThis.__ELFFUSS_FLASH_GRUPOS) || GRUPOS_OBJETIVO;
  const clavesMin = entero(globalThis.__ELFFUSS_FLASH_CLAVES) || CLAVES_MIN;
  let partes = 1;
  if (grupos < objetivo && span >= 2 * clavesMin) {
    partes = Math.max(1, Math.min(PARTES_MAX, Math.ceil(objetivo / grupos), Math.ceil(span / clavesMin)));
    // El kernel redondea el trozo de cada parte a teselas enteras (ver
    // `trozo` en el shader). Con ese trozo, las partes que de verdad llevan
    // claves en el rango más largo; las de detrás saldrían vacías, y un grupo
    // vacío cuesta su lanzamiento y una entrada más en la reducción. La última
    // puede quedarse con pocas claves: es el resto, cuesta una tesela como
    // cualquiera y meterlo en otra parte alargaría el camino más largo.
    partes = Math.ceil(span / (Math.ceil(Math.ceil(span / partes) / FLASH_BK) * FLASH_BK));
  }
  // Forzadas para medir, con más margen que la regla: el barrido tiene que
  // poder ver qué hay por encima del tope que la regla se pone.
  const partesPedidas = entero(globalThis.__ELFFUSS_FLASH_PARTES);
  if (partesPedidas) partes = Math.min(8 * PARTES_MAX, partesPedidas);
  if (KVH * qTiles * partes > 65535 * 65535) return no('rejilla demasiado grande');

  const parcial = partes > 1;
  // La puerta va en el kernel solo sin partes; con partes la aplica la reducción.
  const kernel = { D, G, TQ, NSG, puerta: !!puerta && !parcial, parcial, qsh, vsh, pref, psh };
  // TODO lo que cambia el texto del shader va en la clave. Sin eso, un A/B
  // reutilizaría el compilado de la otra rama y mediría dos veces lo mismo.
  // (`psh` solo añade sufijo cuando está puesto: las claves de siempre no cambian.)
  const clave = `flash_${D}_${G}_${TQ}_${NSG}_${kernel.puerta ? 'g' : 'n'}_${parcial ? 'p' : 'd'}` +
    `_${qsh ? 'qs' : 'qg'}_${vsh ? 'vs' : 'vg'}_${pref ? 'pf' : 'np'}${psh ? '_ps' : ''}`;
  const claveReduce = `flash_reduce_${D}_${puerta ? 'g' : 'n'}`;
  return { ok: true, G, TQ, R, NSG, RS, WG, qTiles, partes, span, memoria, kernel, clave, claveReduce };
}

// `despacha(label, pipeline, buffers, grupos)` es el despacho plano del motor
// (ops.dispatchAny): reparte los grupos en dos ejes si pasan del tope.
export function createFlashOps(device, { ctx, despacha, subgroupSize = 0 }) {
  const get = ctx.pipeline;
  const dev = () => ({ subgroupSize, limites: device.limits });
  // Claves que este dispositivo no compiló (clave → mensaje). El plan solo ve
  // límites y formas; que el compilador del navegador acepte el kernel no se
  // sabe hasta pedirlo. Una clave apartada no se vuelve a intentar.
  const rechazadas = new Map();
  const exige = () => globalThis.__ELFFUSS_FLASH_EXIGE === true;
  const compila = async (clave, hazShader) => {
    try {
      return await get(clave, hazShader);
    } catch (e) {
      if (exige()) throw e;
      const msg = String((e && e.message) || e);
      rechazadas.set(clave, msg);
      console.warn(`[flash] ${clave} no compila en este dispositivo; esa forma vuelve a la cadena de siempre: ${msg.slice(0, 300)}`);
      return null;
    }
  };

  return {
    plan: (cfg) => planFlash(cfg, dev()),
    puede(cfg) {
      const pl = planFlash(cfg, dev());
      const motivo = !pl.ok ? pl.motivo
        : rechazadas.has(pl.clave) ? `${pl.clave} no compiló`
        : pl.partes > 1 && rechazadas.has(pl.claveReduce) ? `${pl.claveReduce} no compiló` : null;
      if (motivo && exige()) throw new Error(`flash exigida y no puede: ${motivo}`);
      return !motivo;
    },
    // Solo diagnóstico: lo que no compiló. Una prueba en el dispositivo tiene que verlo vacío.
    rechazadas: () => [...rechazadas.entries()],

    // q: [T][H][D] f32 (tras RoPE). kCache/vCache: la caché f16 [ctx][KVH][D]
    // con las T posiciones nuevas ya escritas a partir de nPast. `puerta`:
    // [T][H] f32 sin sigmoide (lfm3) o null. Devuelve [T][H][D] f32: la entrada
    // de la proyección de salida, con la puerta ya aplicada si la hay. O null si
    // un kernel no compila: no se ha despachado nada, y quien llama hace esa
    // capa por la cadena (la fusionada solo escribe en sus propios buffers).
    async atencion(q, kCache, vCache, { headDim, heads, kvHeads, tokens, nPast = 0, ventana = 0, kvF16 = false, puerta = null }) {
      const pl = planFlash({ headDim, heads, kvHeads, tokens, nPast, ventana, kvF16, puerta: !!puerta }, dev());
      if (!pl.ok) throw new Error(`flash: ${pl.motivo}`);
      const D = headDim, H = heads, KVH = kvHeads, T = tokens;
      // Los dos pipelines ANTES de despachar nada: si el segundo no compila, el
      // primero no ha corrido para nada.
      const p = await compila(pl.clave, () => flashShader(pl.kernel));
      if (!p) return null;
      const pr = pl.partes > 1 ? await compila(pl.claveReduce, () => flashReduceShader({ D, puerta: !!puerta })) : null;
      if (pl.partes > 1 && !pr) return null;
      // Misma escala que la cadena de siempre: 1/√D en doble, a f32 en el uniform.
      const params = ctx.uniform([
        new Uint32Array([T, nPast, H, KVH, ventana, pl.qTiles, pl.partes, 0]),
        new Float32Array([1 / Math.sqrt(D), 0, 0, 0]),
      ]);
      const grupos = KVH * pl.qTiles * pl.partes;
      if (pl.partes === 1) {
        const out = ctx.alloc(T * H * D * 4, 'flash.out');
        const bufs = pl.kernel.puerta ? [q, kCache, vCache, puerta, out, params] : [q, kCache, vCache, out, params];
        await despacha('flash', p, bufs, grupos);
        return out;
      }
      const parcO = ctx.alloc(T * H * pl.partes * D * 4, 'flash.parc');
      const parcML = ctx.alloc(T * H * pl.partes * 8, 'flash.ml');
      await despacha('flash', p, [q, kCache, vCache, parcO, parcML, params], grupos);
      const out = ctx.alloc(T * H * D * 4, 'flash.out');
      const parR = ctx.uniform([new Uint32Array([T * H, pl.partes, 0, 0])]);
      const bufsR = puerta ? [parcO, parcML, puerta, out, parR] : [parcO, parcML, out, parR];
      await despacha('flash_reduce', pr, bufsR, Math.ceil(T * H * (D / 4) / 64));
      return out;
    },
  };
}

// ── Referencia de CPU para las pruebas ──────────────────────────────────────
// Atención con máscara causal y de ventana, en doble y por la definición (sin
// softmax en línea ni teselas): puntuaciones, máximo de las visibles, exp,
// suma, combinación y puerta. Solo calcula las filas (t, h) pedidas en `filas`
// —un prefill entero en JavaScript no cabe en el tiempo de una prueba—.
// k y v son las cachés YA en doble ([nKv][KVH][D]); q es [T][H][D].
export function atencionCPU(q, k, v, { headDim: D, heads: H, kvHeads: KVH, tokens: T, nPast = 0, ventana = 0 },
  puerta = null, filas = null) {
  const G = H / KVH, scale = 1 / Math.sqrt(D);
  const lista = filas || Array.from({ length: T * H }, (_, i) => [Math.floor(i / H), i % H]);
  const out = new Map();
  for (const [t, h] of lista) {
    const kvh = Math.floor(h / G), qAbs = nPast + t;
    const ini = ventana > 0 ? Math.max(0, qAbs - ventana + 1) : 0;
    const s = new Float64Array(qAbs + 1 - ini);
    let mx = -Infinity;
    for (let j = ini; j <= qAbs; j++) {
      let acc = 0;
      for (let d = 0; d < D; d++) acc += q[(t * H + h) * D + d] * k[(j * KVH + kvh) * D + d];
      s[j - ini] = acc * scale;
      if (s[j - ini] > mx) mx = s[j - ini];
    }
    let suma = 0;
    for (let i = 0; i < s.length; i++) { s[i] = Math.exp(s[i] - mx); suma += s[i]; }
    const o = new Float64Array(D);
    for (let j = ini; j <= qAbs; j++) {
      const w = s[j - ini] / suma;
      for (let d = 0; d < D; d++) o[d] += w * v[(j * KVH + kvh) * D + d];
    }
    if (puerta) {
      const sg = 1 / (1 + Math.exp(-puerta[t * H + h]));
      for (let d = 0; d < D; d++) o[d] *= sg;
    }
    out.set(t * H + h, o);
  }
  return out;
}
