// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · operaciones del motor sobre WebGPU.
// ─────────────────────────────────────────────────────────────────────────────
// Envuelve los shaders en pipelines reutilizables. Los pipelines se compilan UNA
// vez y se cachean: crearlos por llamada mete una compilación de shader en el
// bucle de decode, que es un desastre silencioso — no falla, solo va veinte
// veces más lento.
//
// Todo lo que crea recursos va envuelto en scopes de error: en WebGPU un binding
// mal alineado no lanza, devuelve ceros, y una capa que devuelve ceros produce
// texto plausible sin un solo error en consola.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.
import { guard as rawGuard } from './device.js';

// guard() cuesta una ida y vuelta al device timeline por llamada. Imprescindible
// al arrancar y en los tests; en el bucle de decode, con 423 operaciones por
// token, es el propio diagnostico el que domina el tiempo.
async function guard(device, label, fn, on = true) {
  return on ? await rawGuard(device, label, fn) : await fn();
}
import {
  dequantShader, matmulShader, gatherShader, QUANT_TYPES, AXIS, sgFilasPorGrupo,
} from './kernels.wgsl.js';
// Matmul de decode con VARIAS FILAS por workgroup. Existe porque cada fila de
// salida se relee la fila ENTERA de activaciones: por token se mueven unos 20,5
// bytes de activacion por cada byte de peso. Ese trafico no puede estar saliendo
// de DRAM —la razon medida se pasa del ancho de banda disponible— asi que sale de
// cache, y es por lo que el oraculo X_FIJA mide que traer `x` es el termino mayor
// del matmul. Con N filas por workgroup ese trafico se divide entre N.
import { matmulFilasShader, filasPorGrupo } from './kernels.filas.wgsl.js';
import { matmulGemmShader, soportaGemm, tkPara, TIPOS_GEMM } from './kernels.gemm.wgsl.js';
import { hadamardShader } from './kernels.hadamard.wgsl.js';
import {
  rmsnormShader, ropeShader, softmaxShader, swigluShader, addShader,
  ROPE_NORM, ROPE_NEOX,
} from './kernels.nn.wgsl.js';
import {
  l2normShader, ssmConvShader, gatedDeltaNetShader, gatedNormShader,
  decayGateShader, sigmoidShader, SSM_HEAD_DIM, SSM_CONV_K,
} from './kernels.ssm.wgsl.js';
import {
  strideCopyShader, attnScoresShader, attnCombineShader, mulShader,
  attnCombineSplitShader, combineReduceShader, bloquesCombine,
  attnScoresSharedShader, compartidaScores,
} from './kernels.attn.wgsl.js';

export { ROPE_NORM, ROPE_NEOX, SSM_HEAD_DIM, SSM_CONV_K };

// Perezoso a proposito: evaluarlo al cargar el modulo ataria este fichero al
// navegador y ninguna de las referencias de CPU se podria importar desde Node.
let STORAGE_RW = 0;
function storageUsage() {
  if (!STORAGE_RW) {
    STORAGE_RW = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  }
  return STORAGE_RW;
}

// Tope de workgroups por eje. Viene del shader (que recompone el índice con él),
// no de una copia local: dos constantes que se desincronicen darían filas a cero.
const DISPATCH_MAX = AXIS;

// Tokens por tesela en prefill. Cada peso leído se aplica a este número de
// tokens, así que divide el tráfico de memoria de la matriz por otro tanto.
// Tokens por tesela en prefill. ERA 8, y ese 8 costaba casi la mitad del
// matmul del prefill sin que nadie lo hubiera medido.
//
// La aritmética del tráfico por token (la cuenta está en la cabecera de
// `kernels.filas.wgsl.js`): por cada byte de peso leído se leen unos veinte de
// activaciones (`x`).
//
// Teselar divide el PRIMERO entre `tile` y no toca el segundo: ataca como mucho
// el 4,6% del tráfico. Y a cambio APAGA las tres rutas que atacan el 95,4%
// restante, las tres por el mismo predicado `tile === 1`: la lectura vec4 de
// `x` (medida ×1,135), la guarda de token izada y el reparto de filas entre
// subgrupos. Por eso la lectura vec4 solo se notaba en decode: el prefill nunca
// la ejecutaba.
//
// MEDIDO en el 27B con 32 tokens, en dos sesiones distintas y con el orden
// alternado: tile=1 da del orden de +94% en las dos. Una medida hecha con otro
// trabajo pesado en el mismo dispositivo no vale: la contención arrastra a las
// DOS ramas hacia el mismo cuello y comprime las razones hacia 1.
//
// ⚠️ NO ES BIT A BIT EXACTO: tile=1 cambia de estructura de kernel
// (`decodeCarril` en vez de `decode`, reducción por subgrupos), o sea que cambia
// la partición de la suma. Desvío medido: 1,3e-6 de la norma contra un límite
// de 1e-4. Se valida por tolerancia, no por bytes.
//
// ⛔ Y AUN ASÍ SIGUE EN 8, porque poner 1 de defecto NO COMPILA. Se probó y
// la prueba del forward ni llegó a la numérica: error de COMPILACIÓN del WGSL,
// no de precisión. La prueba de unas formas concretas de matmul pasa; el bloque
// SSM usa otra que a tile=1 genera shader inválido. O sea: la ganancia es real
// para lo medido, y el defecto global no es viable hasta arreglar esa
// generación.
//
// ⚠️ TODO LO DE ARRIBA ES HISTORIA, y el valor dice 1, no 8. El comentario
// decía «se revierte a 8» mientras la línea ponía 1, así que durante un tiempo
// nadie podía saber qué corría este camino leyendo el fichero.
//
// Lo que pasó es que el problema se movió de sitio: los dos tipos que usa
// producción —Q8_0 en el modelo pequeño y PQ2_0 en el grande— van por el GEMM
// teselado (`soportaGemm`), que no mira `tile` para nada. Así que el bloque SSM
// ya no llega a la forma que no compilaba, y las pruebas del forward, del modelo
// de juguete y del proveedor con el modelo grande pasan con el 1 puesto.
//
// Pero el camino de `tile` sigue vivo para todo tipo SIN soporte de GEMM
// (Q4_K, Q2_K, Q5_K, Q6_K, IQ1_S, IQ2_XXS, F16…), y ahí `tile = 1` significa
// `z = tokens`: cada peso se vuelve a leer de memoria global UNA VEZ POR TOKEN
// del lote. Con el lote de 32 son 32 relecturas del tensor entero. No muerde en
// producción y es un acantilado silencioso para cualquier GGUF de terceros con
// un solo tensor grande en Q4_K.
//
// PENDIENTE, por orden de valor: (1) añadir Q4_K/Q2_K/Q5_K a `TIRADAS` para
// sacarlos de aquí —`blockElems` 256 es múltiplo de TK=32, así que encaja—;
// (2) arreglar la generación del shader para la forma del bloque SSM a tile=1;
// (3) escribir para esos tipos un kernel teselado propio.
const TILE_PREFILL = 1;

// Contexto compartido por todas las familias de operaciones del mismo device.
// Existe para tres cosas que solo se pueden hacer si alguien las coordina:
// agrupar todo un forward en UN submit, reciclar buffers entre tokens, y poder
// apagar los scopes de error en el bucle de decode.
function createContext(device, opts = {}) {
  const cache = new Map();
  const ctx = {
    device,
    pass: null,                       // no null ⇒ se está grabando un lote
    debug: opts.debug !== false,      // scopes de error: sí al arrancar, no al decodificar
    uniforms: [],                     // buffers uniform prestados en el lote actual
    uniformFree: [],
    scratch: [],
    scratchFree: new Map(),           // tamaño → [buffers]
  };

  // `code` puede ser el WGSL o una FUNCIÓN que lo construya. La diferencia no es
  // de estilo: con la cadena, el argumento se evalúa ANTES de entrar aquí, así
  // que el acierto de caché de la primera línea llegaba tarde — el shader se
  // reconstruía entero en cada operación, miles de veces por pasada, para
  // tirarlo acto seguido. Con la función, el WGSL solo se arma cuando de verdad
  // hay que compilarlo. Se admiten las dos formas: hay llamadas que ya traen la
  // cadena hecha y no tiene sentido envolverlas.
  ctx.pipeline = async function (key, code) {
    if (cache.has(key)) return cache.get(key);
    const p = await guard(device, 'compilar ' + key, async () => {
      const module = device.createShaderModule({ label: key, code: typeof code === 'function' ? code() : code });
      const info = await module.getCompilationInfo?.();
      const errs = (info?.messages || []).filter(m => m.type === 'error');
      if (errs.length) {
        throw new Error(errs.map(m => `${key}:${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
      }
      return await device.createComputePipelineAsync({
        label: key, layout: 'auto', compute: { module, entryPoint: 'main' },
      });
    }, true);
    cache.set(key, p);
    return p;
  };

  // Solo DIAGNOSTICO: las claves de pipeline ya compiladas. Existe porque el modo
  // de fallo mas caro de este repo es que dos ramas de un A/B compartan pipeline
  // —el A/B sale x1,00 y se lee como «no hay ganancia» cuando lo que hay es una
  // clave incompleta— y desde un test no habia forma de verlo. No toca el camino
  // de ejecucion: es un lector de las claves del cache.
  ctx.claves = () => [...cache.keys()];

  // Un uniform por operación, SIEMPRE: reutilizar uno dentro del mismo lote
  // sería una carrera silenciosa. queue.writeBuffer se ordena contra el submit,
  // así que la segunda escritura pisaría a la primera antes de que el
  // dispositivo ejecute ninguna de las dos. Se reciclan entre lotes, no dentro.
  ctx.uniform = function (arrays) {
    const bytes = arrays.reduce((n, a) => n + a.byteLength, 0);
    const size = Math.max(16, Math.ceil(bytes / 16) * 16);
    let buf = ctx.uniformFree.pop();
    if (!buf || buf.size < size) {
      if (buf) buf.destroy();
      buf = device.createBuffer({
        label: 'params', size: Math.max(size, 64),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
    let off = 0;
    for (const a of arrays) { device.queue.writeBuffer(buf, off, a); off += a.byteLength; }
    ctx.uniforms.push(buf);
    return buf;
  };

  // Buffers de trabajo reciclables. Un forward crea cientos de tensores
  // intermedios; sin reciclado se asignan y liberan cientos de buffers por
  // token, que es trabajo del driver por nada.
  // El pico de memoria prestada SIN reciclar. No es decoración: el pozo solo se
  // vacía al terminar un token, así que dentro de un prefill de N tokens se
  // acumulan todos los intermedios de todas las capas, y cada uno escala con N.
  // Sin este número, «cuánta memoria del dispositivo pide esto» era una suposición.
  ctx.picoBytes = 0;
  ctx.vivosBytes = 0;
  // MARCA DE REPARTO. Cada vez que el pozo entrega un buffer —nuevo o
  // reciclado— le pone un número distinto. Sirve para que quien quiera
  // memorizar algo DERIVADO de un buffer tenga una clave que no se pueda alias:
  // indexar por el buffer en sí es inseguro porque el que fue una activación
  // puede ser otra distinta más adelante en el mismo forward, y entonces la
  // memoria devuelve lo que calculó para la anterior. No falla, no avisa, y el
  // modelo dice ruido. Con la marca, un buffer reciclado es un valor nuevo.
  let reparto = 0;
  ctx.marcaDe = new WeakMap();
  ctx.alloc = function (bytes, label) {
    const size = Math.max(4, Math.ceil(bytes / 256) * 256);
    const free = ctx.scratchFree.get(size);
    let buf = free && free.pop();
    if (!buf) buf = device.createBuffer({ label, size, usage: storageUsage() });
    ctx.marcaDe.set(buf, ++reparto);
    ctx.scratch.push({ buf, size });
    ctx.vivosBytes += size;
    if (ctx.vivosBytes > ctx.picoBytes) ctx.picoBytes = ctx.vivosBytes;
    return buf;
  };

  // Devuelve al pozo todo lo prestado. Lo llama el modelo al terminar un token.
  ctx.recycle = function () {
    for (const b of ctx.uniforms) ctx.uniformFree.push(b);
    ctx.uniforms = [];
    for (const { buf, size } of ctx.scratch) {
      if (!ctx.scratchFree.has(size)) ctx.scratchFree.set(size, []);
      ctx.scratchFree.get(size).push(buf);
    }
    ctx.scratch = [];
    ctx.vivosBytes = 0;
  };

  // Devuelve al pozo lo prestado A PARTIR de una marca, salvo los buffers que
  // siguen vivos. Sirve para reciclar ENTRE CAPAS sin esperar al final del
  // forward: los intermedios de la capa N están muertos en cuanto la capa N ha
  // producido su salida, y guardarlos hasta el final hacía que el pico creciera
  // con el número de capas Y con el de tokens a la vez (lineal en los dos, sin
  // techo).
  //
  // Es seguro DENTRO de una misma pasada de cómputo: WebGPU garantiza que los
  // despachos de una pasada se ejecutan en orden y ven lo que escribieron los
  // anteriores, así que reutilizar un buffer en la capa N+1 ocurre después de
  // que la capa N lo haya leído. Lo que NO sería seguro es reciclar entre
  // envíos distintos sin sincronizar.
  ctx.recycleDesde = function (marca, ...vivos) {
    const guardar = new Set(vivos);
    const quedan = [];
    for (let i = marca; i < ctx.scratch.length; i++) {
      const e = ctx.scratch[i];
      if (guardar.has(e.buf)) { quedan.push(e); continue; }
      if (!ctx.scratchFree.has(e.size)) ctx.scratchFree.set(e.size, []);
      ctx.scratchFree.get(e.size).push(e.buf);
      ctx.vivosBytes -= e.size;
    }
    ctx.scratch.length = marca;
    for (const e of quedan) ctx.scratch.push(e);
  };

  // Agrupa todo lo que ocurra dentro en UN command encoder y UN submit. Con ~423
  // operaciones por token, hacerlo por operación son ~423 sincronizaciones por
  // token en vez de una, y eso es coste puro de coordinación.
  ctx.batch = async function (label, fn) {
    if (ctx.pass) return await fn();          // ya estamos dentro de un lote
    if (ctx.profile) return await fn();       // perfilando: cada despacho lleva su propio pass
    if (ctx.debug) { device.pushErrorScope('validation'); }
    ctx._label = label;
    ctx._enc = device.createCommandEncoder({ label });
    ctx.pass = ctx._enc.beginComputePass(ctx._descPase(label));
    let out;
    try { out = await fn(); } finally {
      ctx.pass.end();
      ctx.pass = null;
      device.queue.submit([ctx._enc.finish()]);
      ctx._enc = null;
    }
    if (ctx.debug) {
      const err = await device.popErrorScope();
      if (err) throw new Error(`[${label}] ${err.constructor.name}: ${err.message}`);
    }
    return out;
  };

  // Cierra el command buffer en curso, lo ENVÍA, y abre otro — para que un
  // forward largo NO sea un único despacho larguísimo que monopoliza el
  // dispositivo. Troceado en varios, el dispositivo queda libre un instante entre
  // ellos y el compositor del sistema puede pintar. Sin eso, un prefill largo
  // deja al compositor sin pintar dentro del plazo de su vigilante, y el sistema
  // puede matar la sesión gráfica. Todos los sistemas tienen un vigilante así,
  // con distinto nombre y distinto plazo.
  // NO cambia la numérica: el orden de la cola
  // garantiza que los despachos se ejecutan igual, solo que en varias tandas.
  // Solo actúa dentro de un batch (si no hay pass abierto, no hace nada).
  ctx.corta = function () {
    if (!ctx.pass) return;
    ctx.pass.end();
    device.queue.submit([ctx._enc.finish()]);
    ctx._enc = device.createCommandEncoder({ label: ctx._label });
    ctx.pass = ctx._enc.beginComputePass(ctx._descPase(ctx._label));
  };

  // ── Perfilado por PASE, con el agrupado INTACTO (solo para medir) ───────────
  // El perfilador por despacho de abajo tiene dos límites que lo inhabilitan
  // para el prefill:
  //   · topa en 2048 operaciones, y una sola pasada del 27B son 1652 despachos:
  //     el prefill de 316 tokens (16.520) no cabe ni de lejos, y pasado el tope
  //     los despachos se ejecutan SIN contarse;
  //   · rompe el agrupado — `ctx.batch` se rinde cuando está puesto — así que lo
  //     que mide ya no es la configuración que corre en producción.
  // Esto marca los pases que `ctx.batch` y `ctx.corta` ya crean de todas formas
  // (uno por capa): mismo agrupado, mismo número de submits, mismo troceado. Un
  // prefill de 316 tokens son 10 pasadas × 65 pases = 650 marcas, y cabe entero.
  // Es la única medida cuya suma se puede comparar con el reloj de pared.
  ctx.pases = null;
  ctx.perfilarPases = function (maxPases = 2048) {
    maxPases = Math.min(maxPases, 2048);   // el device admite 4096 queries
    const querySet = device.createQuerySet({ type: 'timestamp', count: maxPases * 2 });
    const resolve = device.createBuffer({ size: maxPases * 2 * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: maxPases * 2 * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    ctx.pases = { querySet, resolve, read, i: 0, vistos: 0, max: maxPases };
  };
  // El descriptor del pass vive en UN SOLO SITIO y lo llaman `batch` y `corta`.
  // Si la decisión de «este pass lleva marca» estuviera escrita en los dos, un
  // día discreparían y la mitad de los pases saldría sin medir sin un solo error.
  ctx._descPase = function (label) {
    const pf = ctx.pases;
    if (!pf) return { label };
    pf.vistos++;                           // se cuenta SIEMPRE, quepa o no
    if (pf.i >= pf.max) return { label };  // el hueco no se oculta: sale en `omitidos`
    const idx = pf.i++;
    return { label, timestampWrites: { querySet: pf.querySet,
      beginningOfPassWriteIndex: idx * 2, endOfPassWriteIndex: idx * 2 + 1 } };
  };
  ctx.finProfilePases = async function () {
    const pf = ctx.pases; if (!pf) return { total: 0, pases: [], vistos: 0, omitidos: 0 };
    await device.queue.onSubmittedWorkDone();
    const n = Math.min(pf.i, pf.max);
    const enc = device.createCommandEncoder();
    enc.resolveQuerySet(pf.querySet, 0, n * 2, pf.resolve, 0);
    enc.copyBufferToBuffer(pf.resolve, 0, pf.read, 0, n * 2 * 8);
    device.queue.submit([enc.finish()]);
    await pf.read.mapAsync(GPUMapMode.READ);
    const ts = new BigUint64Array(pf.read.getMappedRange().slice(0));
    pf.read.unmap();
    const pases = [];
    let total = 0;
    for (let k = 0; k < n; k++) {
      const dt = Number(ts[2 * k + 1] - ts[2 * k]);   // nanosegundos
      pases.push(dt); total += dt;
    }
    ctx.pases = null;
    try { pf.querySet.destroy(); pf.resolve.destroy(); pf.read.destroy(); } catch { /* — */ }
    return { total, pases, vistos: pf.vistos, omitidos: pf.vistos - n };
  };

  // ── Perfilado por despacho con timestamp-query (solo para medir) ────────────
  // Con esto activo, cada despacho corre en su PROPIO pass con marcas de tiempo
  // del dispositivo al principio y al final, y `finProfile()` devuelve los nanosegundos
  // sumados por etiqueta. No es el modo normal (rompe el batching), pero es la
  // única forma de saber QUÉ kernel se come el tiempo en vez de adivinarlo.
  ctx.profile = null;
  // `desde` salta los primeros N despachos SIN medirlos, para poder cubrir una
  // pasada que no cabe en un solo juego de marcas. El tope de 2048 no es una
  // elección: un query set admite 4096 marcas y cada despacho gasta dos. Una
  // pasada del 27B son 2.053 despachos, o sea que se queda a cinco de caber, y
  // sin esto el reparto de ese modelo era sencillamente inobtenible.
  //
  // Dos pasadas con `desde` distinto miden la MISMA secuencia porque el forward
  // es determinista: el despacho número k es el mismo en las dos. Lo que no se
  // puede es sumar sus TIEMPOS como si fueran una sola pasada —cada una es una
  // tanda distinta, con sus propias condiciones—, así que quien las junte tiene
  // que decir que son dos tandas.
  ctx.perfilar = function (maxOps = 2000, desde = 0) {
    maxOps = Math.min(maxOps, 2048);   // tope: el device admite 4096 queries = 2048 ops×2
    const querySet = device.createQuerySet({ type: 'timestamp', count: maxOps * 2 });
    const resolve = device.createBuffer({ size: maxOps * 2 * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: maxOps * 2 * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    ctx.profile = { querySet, resolve, read, i: 0, vistos: 0, labels: [], max: maxOps, desde };
  };
  ctx.finProfile = async function () {
    const pf = ctx.profile; if (!pf) return { total: 0, porLabel: [] };
    await device.queue.onSubmittedWorkDone();
    const n = Math.min(pf.i, pf.max);
    const enc = device.createCommandEncoder();
    enc.resolveQuerySet(pf.querySet, 0, n * 2, pf.resolve, 0);
    enc.copyBufferToBuffer(pf.resolve, 0, pf.read, 0, n * 2 * 8);
    device.queue.submit([enc.finish()]);
    await pf.read.mapAsync(GPUMapMode.READ);
    const ts = new BigUint64Array(pf.read.getMappedRange().slice(0));
    pf.read.unmap();
    const acum = new Map(), veces = new Map();
    let total = 0;
    for (let k = 0; k < n; k++) {
      const dt = Number(ts[2 * k + 1] - ts[2 * k]);   // nanosegundos
      const lab = pf.labels[k];
      acum.set(lab, (acum.get(lab) || 0) + dt);
      veces.set(lab, (veces.get(lab) || 0) + 1);
      total += dt;
    }
    ctx.profile = null;
    try { pf.querySet.destroy(); pf.resolve.destroy(); pf.read.destroy(); } catch { /* — */ }
    const porLabel = [...acum.entries()]
      .map(([lab, ns]) => ({ lab, ns, veces: veces.get(lab) }))
      .sort((a, b) => b.ns - a.ns);
    // `omitidos` son los que esta tanda NO midió, por cualquier razón. `saltados`
    // dice cuántos de ellos fueron a propósito (los de antes de `desde`), para
    // que quien junte dos tandas pueda distinguir «esto lo cubre la otra» de
    // «esto no lo cubre nadie». Sin esa distinción, un reparto incompleto se
    // lee igual que uno completo.
    const desde = pf.desde || 0;
    const saltados = Math.min(desde, pf.vistos);
    return { total, porLabel, vistos: pf.vistos, desde, saltados,
      omitidos: pf.vistos - n, sinCubrir: pf.vistos - n - saltados };
  };

  return ctx;
}

// Graba un despacho. Dentro de un lote solo apunta; fuera, abre y cierra su
// propio encoder para que las llamadas sueltas de los tests sigan funcionando.
async function dispatch(ctx, label, pipe, buffers, x, y = 1, z = 1) {
  const device = ctx.device;
  const record = (pass) => {
    const bg = device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(x, y, z);
  };
  if (ctx.pass) { record(ctx.pass); return; }
  // Se cuentan TODOS los despachos que ve el perfilador, quepan o no. Sin esto,
  // pasado el tope el despacho cae al camino normal y se ejecuta SIN contarse:
  // la etiqueta que desborda sale BARATA, que es la peor forma de mentir que
  // puede tener un instrumento. Con `vistos`, `finProfile` devuelve `omitidos` y
  // el banco puede negarse a publicar un reparto truncado.
  if (ctx.profile) ctx.profile.vistos++;
  // Perfilado: cada despacho en su pass con marcas de tiempo del dispositivo. Los que
  // caen antes de `desde` se ejecutan por el camino normal SIN medirse: es lo
  // que permite cubrir una pasada más larga que el juego de marcas en dos
  // tandas. `vistos` ya los ha contado, así que el hueco sigue siendo visible.
  if (ctx.profile && ctx.profile.vistos > (ctx.profile.desde || 0)
      && ctx.profile.i < ctx.profile.max) {
    const pf = ctx.profile;
    const idx = pf.i++;
    pf.labels.push(label);
    const enc = device.createCommandEncoder({ label });
    const pass = enc.beginComputePass({ timestampWrites: {
      querySet: pf.querySet, beginningOfPassWriteIndex: idx * 2, endOfPassWriteIndex: idx * 2 + 1 } });
    record(pass);
    pass.end();
    device.queue.submit([enc.finish()]);
    return;
  }
  await guard(device, label, async () => {
    const enc = device.createCommandEncoder({ label });
    const pass = enc.beginComputePass();
    record(pass);
    pass.end();
    device.queue.submit([enc.finish()]);
  }, ctx.debug);
}

// Reparte `rows` en dos ejes cuando pasan del tope. El shader recompone con
// row = wid.x + wid.y * DISPATCH_MAX.
// Una fila por workgroup, repartiendo en dos ejes si hacen falta.
async function dispatchRows(ctx, label, pipe, buffers, rows) {
  const [x, y] = rowsToDims(rows);
  return dispatch(ctx, label, pipe, buffers, x, y, 1);
}

// N workgroups planos (operaciones elemento a elemento).
async function dispatchN(ctx, label, pipe, buffers, groups) {
  const [x, y] = rowsToDims(groups);
  return dispatch(ctx, label, pipe, buffers, x, y, 1);
}

// Tres ejes tal cual, para los kernels cuya rejilla ES tridimensional y no un
// contador plano disfrazado. El combine partido reparte (trozo, cabeza KV,
// token), y aplanarlo obligaría al shader a deshacer la cuenta con divisiones
// que el despacho ya sabe hacer gratis. Se comprueba cada eje contra el tope:
// pasarse NO lanza en WebGPU, invalida el command buffer y el pase se ejecuta
// sin hacer nada — ya pasó en un banco, que devolvió Infinity sin un error.
async function dispatchXYZ(ctx, label, pipe, buffers, x, y, z) {
  if (x > DISPATCH_MAX || y > DISPATCH_MAX || z > DISPATCH_MAX) {
    throw new Error(`${label}: la rejilla ${x}x${y}x${z} pasa del tope de ${DISPATCH_MAX} por eje`);
  }
  return dispatch(ctx, label, pipe, buffers, x, y, z);
}

function rowsToDims(rows) {
  return rows <= DISPATCH_MAX
    ? [rows, 1]
    : [DISPATCH_MAX, Math.ceil(rows / DISPATCH_MAX)];
}

// ── Operaciones sobre pesos cuantizados ─────────────────────────────────────
// Seis tipos con el MISMO andamio de shader (ver kernels.wgsl.js): aquí solo se
// eligen el pipeline, el tamaño de bloque y, si el tipo lo pide, la rejilla.
export function createQuantOps(device, opts = {}) {
  const wgQ8 = opts.workgroupSize || 64;
  // Ancho de subgrupo del dispositivo (0 = sin subgrupos o tamaño variable). Si
  // lo hay, el matmul reduce con `subgroupAdd` en vez de con barreras.
  const sgSize = opts.subgroupSize || 0;
  const ctx = opts.ctx || createContext(device, opts);
  const get = ctx.pipeline;

  const info = (type) => {
    const i = QUANT_TYPES[type];
    if (!i) throw new Error(`tipo de cuantización no soportado: «${type}»`);
    return i;
  };

  // Hilos por fila. Cada hilo se come un bloque entero, así que con bloques de
  // 256 pesos una fila de 5.120 son 20 bloques: con 64 hilos, 44 se quedarían
  // mirando. Se baja a la potencia de dos que cubre los bloques que hay.
  //
  // Q8_0 se queda clavado en su tamaño de siempre a propósito: es el kernel del
  // bucle de decode y está medido así. Cambiarlo sin volver a medirlo sería
  // adivinar.
  const threadsFor = (type, bpr) => type === 'Q8_0'
    ? wgQ8
    : Math.min(64, Math.max(4, 2 ** Math.ceil(Math.log2(Math.max(1, bpr)))));

  // Las rejillas de IQ1_S/IQ2_XXS se suben UNA vez por contexto y solo si algún
  // tensor las pide: son tablas que un modelo todo-Q8_0 no tiene por qué
  // descargar siquiera, de ahí el import dinámico.
  const tables = new Map();
  async function tableBuf(name) {
    if (tables.has(name)) return tables.get(name);
    const mod = await import('./quant-tables.js');
    // Sin caso por defecto, un nombre de tabla NUEVO recibiría en silencio la
    // rejilla de IQ2_XXS: mismo layout de bind group, sin error, y lecturas fuera
    // de rango acotadas a cero. Un modelo entero saldría mal sin una sola traza.
    const GRIDS = { IQ1S: mod.IQ1S_GRID, IQ2XXS: mod.IQ2XXS_GRID };
    const data = GRIDS[name];
    if (!data) throw new Error(`rejilla desconocida: «${name}»`);
    const buf = device.createBuffer({
      label: 'grid_' + name, size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, data);
    tables.set(name, buf);
    return buf;
  }

  return {
    // Descuantiza un tensor completo a f32. No se usa en el forward — ahí
    // descuantiza el propio matmul, sin materializar nada. Existe porque es el
    // kernel que permite comparar peso a peso contra la referencia de CPU, que
    // es donde se caza una descodificación mal hecha.
    async dequant(type, wBuf, nBlocks) {
      const i = info(type);
      const p = await get(`dequant_${type}`, () => dequantShader(type, wgQ8));
      const out = ctx.alloc(nBlocks * i.blockElems * 4, 'dequant.out');
      const params = ctx.uniform([new Uint32Array([nBlocks, 0, 0, 0])]);
      const bufs = [wBuf, out, params];
      if (i.table) bufs.push(await tableBuf(i.table));
      await dispatchN(ctx, 'dequant', p, bufs, Math.ceil(nBlocks / wgQ8));
      return out;
    },

    // Y = A·X con A cuantizada y X token-major (k contiguo por token).
    // `tokens = 1` es el matvec del decode; más es el prefill.
    async matmul(type, wBuf, xBuf, rows, k, tokens = 1) {
      const i = info(type);
      if (k % i.blockElems !== 0) {
        throw new Error(`matmul ${type}: k=${k} no es múltiplo de ${i.blockElems}`);
      }
      // Dos pipelines del MISMO shader: uno sin teselar para el decode (un
      // token, teselar solo añadiría ramas muertas) y otro teselado para el
      // prefill, donde el ahorro de relecturas de peso es todo el asunto.
      const tile = tokens > 1
        ? Math.max(1, Math.min(32, (globalThis.__ELFFUSS_TILE | 0) || TILE_PREFILL))
        : 1;
      const bpr = k / i.blockElems;

      // ── Camino GEMM: una tesela en memoria compartida y el acumulador en
      // registros. Es un sustituto del camino de abajo con el MISMO contrato de
      // entrada y salida, no un modo aparte: lo único que cambia es cómo se
      // reparte el trabajo. Opt-in mientras se mide.
      // `TK` (el paso del bucle sobre k) ya NO es el bloque del tipo: se elige
      // aparte, porque atarlo al formato hundía la ocupación en los tipos de
      // bloque grande —PQ2_0 son 128— y dejaba fuera del barrido las formas de
      // 256 hilos. El porqué entero está en `kernels.gemm.wgsl.js`.
      const Fg = globalThis.__ELFFUSS_GEMM_FORMA || {};
      const TKg = tkPara(type, Fg.TK);
      const puedeGemm = soportaGemm(type, k, TKg);
      // Un TK pedido A MANO que este tipo no sabe hacer tiene que LANZAR, no
      // caerse en silencio al camino de siempre. Los otros cuatro números de la
      // forma ya lanzan —los mira el constructor del shader—, pero TK lo mira
      // `soportaGemm`, que devuelve un booleano: sin esto, el barrido mediría el
      // matmul VIEJO y lo escribiría en la tabla como si fuera «GEMM con TK=64».
      // Un número que no es lo que dice su etiqueta es peor que no tenerlo.
      if (tokens > 1 && Fg.TK && !puedeGemm && TIPOS_GEMM.has(type)) {
        throw new Error(`GEMM: TK=${Fg.TK} no vale para «${type}» ` +
          `(bloque ${i.blockElems}, k=${k}): tiene que dividir al bloque y ser múltiplo de 4`);
      }
      // Por debajo de 4 tokens NI SIQUIERA la tesela estrecha alcanza al camino
      // de uno en uno: medido en el modelo grande, con 2 tokens el viejo gana
      // casi al doble, con 4 empatan y a partir de 8 el GEMM se despega. El
      // cruce está en 4 y por eso el umbral es 4 y no «tokens > 1».
      //
      // No es un detalle: con 2 o 3 tokens se cae en cuanto hay un turno corto
      // de conversación, que es de lo más frecuente que hace un chat.
      const GEMM_DESDE = 4;
      if (tokens >= GEMM_DESDE && globalThis.__ELFFUSS_GEMM !== false && puedeGemm) {
        // La forma de la tesela es ajustable para el barrido. Los CINCO números
        // van en la clave del pipeline: si no fueran, dos formas compartirían el
        // shader ya compilado y la tabla del barrido saldría plana — el modo de
        // fallo que en este repo ya produjo una conclusión falsa. TK entró en la
        // clave a la vez que dejó de venir dado por el tipo: sin él, barrer TK
        // mediría cuatro veces el mismo shader.
        // 2×4 en un grupo de 8×8 no es lo que usa ggml-webgpu (4×4): es lo que
        // salió del barrido aquí. Menos filas por hilo = menos registros vivos =
        // más ondas residentes, y eso pesa más que el reúso extra que daría un
        // TM mayor. Medido x2,29 contra x1,55 de 4×4.
        //
        // El grupo de 16×8 —128 hilos— salió por delante en un barrido y se
        // llevó una tanda propia para confirmarlo: empate. La ventaja era ruido.
        // Se queda 8×8, que hace lo mismo con la mitad de hilos.
        //
        // Esa tanda deja además un aviso sobre cómo se lee esta tabla: el MISMO
        // código no da la misma razón en dos tandas distintas. No solo derivan
        // los tiempos: deriva la razón. Dos factores medidos en tandas distintas
        // no se comparan entre sí, ni aunque los dos sean razones.
        // ── La tesela se ESTRECHA cuando el lote es corto ──────────────────
        // La tesela cubre `WGN·TN` columnas de token. Con 32 columnas y un
        // lote de 4 tokens se calculan 32 y se tiran 28, así que un lote de 2
        // costaba casi lo mismo que uno de 16 —medido en el modelo grande— y
        // unas cinco veces lo que cuesta procesar un token solo.
        //
        // Eso convertía el GEMM en el camino LENTO justo donde más aparece: un
        // prompt corto, un turno breve de conversación, o la ventana de una
        // verificación. Medido, con 4 tokens: estrechar de 32 columnas a 4 lo
        // deja casi en la mitad.
        //
        // Se elige por el número de tokens que de verdad hay. Las formas salen
        // del mismo barrido que la ancha y todas respetan las guardas del
        // constructor; la clave del pipeline ya lleva los cinco números, así
        // que cada ancho compila el suyo y no se pisan.
        // ⚠️ LA TESELA NO ES UNA PALANCA, y esta medido dos veces por si acaso.
        //
        // Se intento cambiar a 2x8 en 16x4 porque un barrido de matmuls
        // AISLADOS con marcas de tiempo del dispositivo la daba x1,58 sobre esta.
        // Dos cosas lo tumbaron:
        //
        //   · el ganador del barrido CAMBIA entre pasadas — la segunda dio 2x4
        //     en 32x8 como mejor y la supuesta ganadora bajo a x1,12—, o sea
        //     que el x1,58 era ruido. El tiempo de un matmul medido en AISLADO
        //     se mueve demasiado entre pasadas, con el mismo tensor y los mismos
        //     tokens: solo un A/B en la misma sesion sobre el forward completo
        //     decide algo.
        //   · en el PREFILL COMPLETO, con las dos teselas en la MISMA sesion y
        //     emparejado, el efecto es x1,03. Nada.
        //
        // O sea que el barrido de teselas viejo
        // tenia razon, aunque midiera con reloj de pared y con una premisa
        // equivocada sobre el peso del matmul: su conclusion se sostiene.
        //
        // Y de paso queda descartada mi hipotesis de que lo que limita el bucle
        // interno son las lecturas de memoria compartida por operacion: el
        // orden por tiempo no coincide en NI UNA posicion con el orden por esa
        // razon, y la forma con menos lecturas por operacion es de las mas
        // lentas.
        const anchoTesela = (n) => (n <= 4 ? { TM: 2, TN: 1, WGM: 16, WGN: 4 }
          : n <= 8 ? { TM: 2, TN: 1, WGM: 8, WGN: 8 }
          : n <= 16 ? { TM: 2, TN: 2, WGM: 8, WGN: 8 }
          : { TM: 2, TN: 4, WGM: 8, WGN: 8 });
        const A = anchoTesela(tokens);
        const G = {
          TM: Fg.TM || A.TM, TN: Fg.TN || A.TN, WGM: Fg.WGM || A.WGM, WGN: Fg.WGN || A.WGN, TK: TKg,
          // El tope real del dispositivo, no un 32768 supuesto: así una forma que
          // no cabe se RECHAZA con un mensaje que dice cuánto pedía, en vez de
          // reventar la validación de WebGPU con un texto genérico.
          maxCompartida: device.limits.maxComputeWorkgroupStorageSize,
        };
        const FILAS_WG = G.WGM * G.TM, TOKS_WG = G.WGN * G.TN;
        // El relleno de la tesela ENTRA EN LA CLAVE. Es una constante del
        // shader: sin ella en la clave, cambiar `__ELFFUSS_PAD_TESELA` en
        // caliente devolvería el pipeline ya compilado con el valor viejo, y un
        // A/B mediría el MISMO shader dos veces creyendo medir dos. Vale para
        // cualquier interruptor que cambie el WGSL generado.
        const pad = globalThis.__ELFFUSS_PAD_TESELA === 1 ? 1 : 0;
        const llano = globalThis.__ELFFUSS_GEMM_PLANO === true ? 1 : 0;
        const pg = await get(`gemm_${type}_${G.TM}_${G.TN}_${G.WGM}_${G.WGN}_k${G.TK}_p${pad}_f${llano}`,
          () => matmulGemmShader(type, G));
        const y = ctx.alloc(rows * tokens * 4, 'matmul.y');
        const params = ctx.uniform([new Uint32Array([rows * bpr, bpr, rows, k, tokens, 0, 0, 0])]);
        const bufs = [wBuf, xBuf, y, params];
        if (i.table) bufs.push(await tableBuf(i.table));
        const [gx, gy] = rowsToDims(Math.ceil(rows / FILAS_WG));
        await dispatch(ctx, `gemm:${type}:r${rows}b${bpr}`, pg, bufs,
          gx, gy, Math.ceil(tokens / TOKS_WG));
        return y;
      }

      // Con subgrupos: el workgroup es EXACTAMENTE un subgrupo (ancho = sgSize) y
      // la reducción va sin barreras. Sin ellos, el reparto de hilos de siempre.
      const subgroup = !globalThis.__ELFFUSS_NO_SUBGROUP && sgSize >= 32;
      // Ancho del workgroup en la ruta de subgrupos = múltiplo del tamaño de
      // subgrupo. Más ancho = más hilos por fila = mejor se esconde la latencia
      // de las lecturas dependientes del dequant i-quant. Tunable para A/B.
      // Subgrupos por workgroup: cada uno lleva SU fila de salida. Sube ondas
      // residentes sin quitar trabajo a nadie. La clave de pipeline ya lleva `wg`.
      const sgf = Math.max(1, Math.min(8, (globalThis.__ELFFUSS_SG_FILAS | 0) || 1));
      // El ancho se sube SOLO si ESTE tipo va a tomar de verdad la fila por
      // subgrupo. Aplicarlo a todos cambiaba la zancada del bucle de 32 a 64 en
      // los cinco tipos sin `decodeCarril`: misma suma, OTRA particion en coma
      // flotante, y los ultimos bits difieren. No era incorrecto, pero rompia la
      // igualdad exacta — y el aserto lo achacaba a la reduccion, mandando a
      // buscar dos veces donde no era.
      const anchoSg = sgf > 1 && sgFilasPorGrupo(type, sgSize * sgf, tile,
        globalThis.__ELFFUSS_CARRIL_SUBBLOQUE !== false, rows, subgroup, sgSize) > 1 ? sgf : 1;
      const mult = subgroup ? Math.max(1, anchoSg, Math.round((globalThis.__ELFFUSS_SG_WG || sgSize) / sgSize)) : 1;
      const wg = subgroup ? sgSize * mult : threadsFor(type, bpr);
      // El izado de la guarda entra en la CLAVE: sin esto, al conmutar
      // `__ELFFUSS_GUARDA_POR_PESO` para el A/B se reutilizaria el pipeline ya
      // compilado y las dos ramas medirian el mismo shader (x1,00 indistinguible
      // de «no hay ganancia»).
      // Reparto por SUB-GRUPO (carril = 8 pesos en vez de un bloque entero), que es
      // el camino POR DEFECTO. Sube la utilizacion de carriles del bucle de zancada
      // de 62,5% a 100% con bpr=20, y en un kernel limitado por emision eso se
      // traduce casi directamente en tiempo: medido +35,8% sobre el reparto por
      // bloque, muy por encima del ruido de la medida.
      //
      // `__ELFFUSS_CARRIL_SUBBLOQUE = false` devuelve el reparto por bloque, que es
      // lo que usa el A/B; por eso la comparacion es `!== false` y no un booleano
      // a secas: sin interruptor puesto, gana el camino rapido.
      const carril = globalThis.__ELFFUSS_CARRIL_SUBBLOQUE !== false;
      // Filas por workgroup. Devuelve 1 (kernel de siempre) salvo que se cumpla
      // TODO: interruptor, subgrupos, reparto por carril, decode y `rows`
      // multiplo. El reparto entra en la CLAVE: sin eso, al conmutar el
      // interruptor se reutilizaria el pipeline ya compilado y las dos ramas
      // medirian el mismo shader.
      const filas = filasPorGrupo(type, wg, tile, carril, rows, subgroup);
      const p = await get(`matmul_${type}_${wg}_${tile}_${subgroup ? 'sg' : 'tr'}` +
        (globalThis.__ELFFUSS_GUARDA_POR_PESO ? '_g' : '') + (carril ? '_c' : '') +
        (globalThis.__ELFFUSS_X_FIJA ? '_x0' : '') + (globalThis.__ELFFUSS_W_FIJA ? '_w0' : '') + (globalThis.__ELFFUSS_DESENROLLA ? '_u' : '') + (globalThis.__ELFFUSS_DEQ_PLANA ? '_dp' : '') + (globalThis.__ELFFUSS_X_VEC4 !== false ? '_v4' : '') +
        // El izado de cargas TIENE que entrar en la clave. Sin esto, al conmutar
        // el interruptor para el A/B se reutilizaria el pipeline ya compilado y
        // las dos ramas medirian el MISMO shader: x1,00 indistinguible de «no hay
        // ganancia». Ese modo de fallo ya produjo una conclusion falsa aqui.
        (globalThis.__ELFFUSS_IZA_CARGAS === true ? '_iz' : '') +
        (globalThis.__ELFFUSS_BLOQUE_COOP === true ? '_bc' : '') +
        // Los dos oraculos de cadena (la de la rejilla y la del acumulador) van
        // tambien en la clave. Sin esto, las ramas comparten el
        // shader ya compilado y el A/B sale x1,00 SIEMPRE — que es justo el modo
        // de fallo que costo descubrir con la guarda de token.
        (globalThis.__ELFFUSS_GI ? `_gi${globalThis.__ELFFUSS_GI}` : '') +
        ((globalThis.__ELFFUSS_ACUMS | 0) > 1 ? `_a${globalThis.__ELFFUSS_ACUMS | 0}` : '') +
        ((globalThis.__ELFFUSS_LASTRE | 0) > 0 ? `_L${globalThis.__ELFFUSS_LASTRE | 0}` : '') +
        (filas > 1 ? `_f${filas}` : ''),
        () => (filas > 1 ? matmulFilasShader(type, filas, sgSize)
                         : matmulShader(type, wg, tile, subgroup, sgSize, carril)));
      const y = ctx.alloc(rows * tokens * 4, 'matmul.y');
      const params = ctx.uniform([new Uint32Array([rows * bpr, bpr, rows, k, tokens, 0, 0, 0])]);
      const bufs = [wBuf, xBuf, y, params];
      if (i.table) bufs.push(await tableBuf(i.table));
      // Filas en los dos primeros ejes, tokens en el tercero. Sin esto, la
      // cabeza de salida (248.320 filas) revienta el tope de 65.535 por eje.
      // Con N filas por workgroup la rejilla cubre rows/N puntos: el shader hace
      // `row0 = (wid.x + wid.y*AXIS) * FILAS`. Si esto y el shader se
      // desincronizaran saldrian filas a CERO sin un solo error.
      // Con S subgrupos por workgroup la rejilla cubre rows/S puntos.
      // El MISMO predicado que usa el shader, no una condicion paralela: es lo
      // que fallo antes, y el sintoma fue que las capas de tipos sin reparto por
      // carril calculaban solo 1/S de sus filas.
      const sgFilas = sgFilasPorGrupo(type, wg, tile, carril, rows, subgroup, sgSize);
      const porGrupo = (filas > 1 ? filas : 1) * sgFilas;
      const [dx, dy] = rowsToDims(porGrupo > 1 ? Math.ceil(rows / porGrupo) : rows);
      // Etiqueta DESAGREGADA por forma (rows/bpr). Agregadas bajo `matmul:IQ1_S`,
      // las 312 matmuls de un decode no dejan ver si el coste es por PESO o por
      // DESPACHO, ni comparar Gw/s entre formas — que es justo lo que distingue
      // «limitado por transacciones de memoria» de «limitado por emisión de
      // instrucciones». No toca el shader ni la clave de pipeline: solo el nombre.
      await dispatch(ctx, `matmul:${type}:r${rows}b${bpr}`, p, bufs, dx, dy, Math.ceil(tokens / tile));
      return y;
    },

    // Filas sueltas de una tabla cuantizada (los embeddings) a f32.
    async gather(type, wBuf, idsBuf, dim, tokens) {
      const i = info(type);
      if (dim % i.blockElems !== 0) {
        throw new Error(`gather ${type}: dim=${dim} no es múltiplo de ${i.blockElems}`);
      }
      const p = await get(`gather_${type}`, () => gatherShader(type, wgQ8));
      const bpr = dim / i.blockElems;
      const dst = ctx.alloc(tokens * dim * 4, 'embd');
      const params = ctx.uniform([new Uint32Array([dim, tokens, bpr, 0])]);
      const bufs = [wBuf, idsBuf, dst, params];
      if (i.table) bufs.push(await tableBuf(i.table));
      await dispatchN(ctx, 'gather', p, bufs, Math.ceil(tokens * bpr / wgQ8));
      return dst;
    },

    // Sube por adelantado las rejillas que vaya a necesitar el modelo, para que
    // el primer forward no se encuentre una descarga y una subida a media pasada.
    async prepare(types) {
      for (const t of new Set(types)) {
        const i = QUANT_TYPES[t];
        if (i && i.table) await tableBuf(i.table);
      }
    },

    upload(array, label = 'x') {
      const buf = ctx.alloc(array.byteLength, label);
      device.queue.writeBuffer(buf, 0, array);
      return buf;
    },
    ctx,
  };
}

// Vista Q8_0 de lo anterior, con la firma de siempre (sin el tipo delante).
// El forward del 0.8B llama por aquí y no se entera de que hay seis tipos.
export function createQ8Ops(device, opts = {}) {
  const q = createQuantOps(device, opts);
  return {
    dequant: (wBuf, nBlocks) => q.dequant('Q8_0', wBuf, nBlocks),
    matmul: (wBuf, xBuf, rows, k, tokens = 1) => q.matmul('Q8_0', wBuf, xBuf, rows, k, tokens),
    matvec: (wBuf, xBuf, rows, k) => q.matmul('Q8_0', wBuf, xBuf, rows, k, 1),
    upload: (array, label) => q.upload(array, label),
    ctx: q.ctx,
  };
}

// ── Operaciones de red ──────────────────────────────────────────────────────
export function createNNOps(device, opts = {}) {
  const ctx = opts.ctx || createContext(device, opts);
  const get = ctx.pipeline;
  const WG_NORM = opts.wgNorm || 128;
  const WG_SM = opts.wgSoftmax || 256;
  const WG_EW = opts.wgElementwise || 256;
  const WG_ROPE = opts.wgRope || 64;

  return {
    upload(array, label = 'buf') {
      const buf = ctx.alloc(array.byteLength, label);
      device.queue.writeBuffer(buf, 0, array);
      return buf;
    },
    alloc: (bytes, label) => ctx.alloc(bytes, label),
    ctx,

    // (x / sqrt(mean(x²) + eps)) · w, por filas de longitud ne0.
    // Rotación de Hadamard sobre las activaciones, en bloques de `bloque`.
    // `inverso` elige el orden de los signos: false = s antes de rotar (lo que
    // va delante de un matmul), true = s después (deshacer el embedding).
    async hadamard(src, signos, { rows, ancho, bloque = 1024, inverso = false, perm = null }) {
      if (ancho % bloque !== 0) {
        throw new Error(`hadamard: el ancho ${ancho} no es múltiplo del bloque ${bloque}`);
      }
      const p = await get(`hadamard_${bloque}_${WG_EW}`, () => hadamardShader(bloque, WG_EW));
      const dst = ctx.alloc(rows * ancho * 4, 'hadamard.out');
      const params = ctx.uniform([
        new Uint32Array([ancho, rows, ancho / bloque, inverso ? 1 : 0,
          perm ? perm.hd : 0, perm ? perm.nk : 0, perm ? perm.rep : 0,
          globalThis.__ELFFUSS_PERM_INV === true ? 1 : 0]),
      ]);
      await dispatchN(ctx, 'hadamard', p, [src, signos, dst, params], rows * (ancho / bloque));
      return dst;
    },

    async rmsnorm(src, weight, rows, ne0, eps) {
      const p = await get('rmsnorm_' + WG_NORM, () => rmsnormShader(WG_NORM));
      const dst = ctx.alloc(rows * ne0 * 4, 'rmsnorm.out');
      const params = ctx.uniform([
        new Uint32Array([ne0, rows]), new Float32Array([eps, 0]),
      ]);
      await dispatchRows(ctx, 'rmsnorm', p, [src, weight, dst, params], rows);
      return dst;
    },

    // RoPE sobre filas [head_dim, heads, tokens]. `mode` decide el pareado:
    // ROPE_NORM empareja adyacentes, ROPE_NEOX empareja i con i+nDims/2.
    async rope(src, posBuf, { rows, ne0, heads, nDims, mode, freqBase, freqScale = 1 }) {
      const p = await get('rope_' + WG_ROPE, () => ropeShader(WG_ROPE));
      const dst = ctx.alloc(rows * ne0 * 4, 'rope.out');
      // theta_scale en el host: acumular la potencia en el shader arrastra el
      // error de pow() y a posiciones grandes se nota.
      const thetaScale = Math.pow(freqBase, -2 / nDims);
      const params = ctx.uniform([
        new Uint32Array([ne0, rows, heads, nDims, mode, 0, 0, 0]),
        new Float32Array([thetaScale, freqScale, 0, 0]),
      ]);
      const total = rows * (nDims / 2);
      await dispatchN(ctx, 'rope', p, [src, posBuf, dst, params], Math.ceil(total / WG_ROPE));
      return dst;
    },

    // Softmax por filas con máscara causal calculada en el shader.
    async softmax(src, { rows, nKv, nQ, nPast = 0, scale = 1 }) {
      const p = await get('softmax_' + WG_SM, () => softmaxShader(WG_SM));
      const dst = ctx.alloc(rows * nKv * 4, 'softmax.out');
      const params = ctx.uniform([
        new Uint32Array([nKv, nQ, nPast, rows]), new Float32Array([scale, 0, 0, 0]),
      ]);
      await dispatchRows(ctx, 'softmax', p, [src, dst, params], rows);
      return dst;
    },

    // silu(gate) · up. El orden importa y no avisa si se invierte.
    async swiglu(gate, up, n) {
      const p = await get('swiglu_' + WG_EW, () => swigluShader(WG_EW));
      const dst = ctx.alloc(n * 4, 'swiglu.out');
      const params = ctx.uniform([new Uint32Array([n, 0, 0, 0])]);
      await dispatchN(ctx, 'swiglu', p, [gate, up, dst, params], Math.ceil(n / WG_EW));
      return dst;
    },

    async add(a, b, n) {
      const p = await get('add_' + WG_EW, () => addShader(WG_EW));
      const dst = ctx.alloc(n * 4, 'add.out');
      const params = ctx.uniform([new Uint32Array([n, 0, 0, 0])]);
      await dispatchN(ctx, 'add', p, [a, b, dst, params], Math.ceil(n / WG_EW));
      return dst;
    },
  };
}

// ── Operaciones de la capa SSM ──────────────────────────────────────────────
export function createSSMOps(device, opts = {}) {
  const ctx = opts.ctx || createContext(device, opts);
  const get = ctx.pipeline;
  const S = opts.headDim || SSM_HEAD_DIM;
  const WG = opts.wg || 128;
  const WG_EW = opts.wgElementwise || 256;

  return {
    async l2norm(src, rows, ne0, eps) {
      const p = await get('l2norm_' + WG, () => l2normShader(WG));
      const dst = ctx.alloc(rows * ne0 * 4, 'l2norm.out');
      const params = ctx.uniform([new Uint32Array([ne0, rows]), new Float32Array([eps, 0])]);
      await dispatchRows(ctx, 'l2norm', p, [src, dst, params], rows);
      return dst;
    },

    // buf: [channels][window] con window = 3 + tokens. dst: [tokens][channels].
    async conv(buf, w, channels, tokens) {
      const p = await get('ssm_conv', () => ssmConvShader());
      const dst = ctx.alloc(channels * tokens * 4, 'conv.out');
      const params = ctx.uniform([new Uint32Array([channels, tokens, tokens + SSM_CONV_K - 1, 0])]);
      await dispatchN(ctx, 'ssm_conv', p, [buf, w, dst, params], Math.ceil(channels * tokens / 64));
      return dst;
    },

    // Actualiza `state` EN SITIO y devuelve la salida. El estado sobrevive entre
    // llamadas: es lo que sustituye a la caché KV en estas capas.
    // `heads` son las cabezas de VALOR y `kvHeads` las de clave. En el 0.8B
    // coinciden; en el 27B son 48 contra 16.
    async gatedDeltaNet(q, k, v, g, beta, state, heads, tokens, kvHeads = heads) {
      const p = await get('gdn_' + S, () => gatedDeltaNetShader(S));
      const out = ctx.alloc(tokens * heads * S * 4, 'gdn.out');
      const params = ctx.uniform([
        new Uint32Array([heads, tokens, kvHeads, 0]),
        new Float32Array([1 / Math.sqrt(S), 0, 0, 0]),
      ]);
      await dispatchRows(ctx, 'gated_delta_net', p, [q, k, v, g, beta, state, out, params], heads);
      return out;
    },

    async gatedNorm(src, w, z, rows, ne0, eps) {
      const p = await get('gated_norm_' + WG, () => gatedNormShader(WG));
      const dst = ctx.alloc(rows * ne0 * 4, 'gnorm.out');
      const params = ctx.uniform([new Uint32Array([ne0, rows]), new Float32Array([eps, 0])]);
      await dispatchRows(ctx, 'gated_norm', p, [src, w, z, dst, params], rows);
      return dst;
    },

    async decayGate(alpha, dt, a, heads, tokens) {
      const p = await get('decay_gate', () => decayGateShader());
      const dst = ctx.alloc(heads * tokens * 4, 'g');
      const params = ctx.uniform([new Uint32Array([heads, tokens, 0, 0])]);
      await dispatchN(ctx, 'decay_gate', p, [alpha, dt, a, dst, params], Math.ceil(heads * tokens / 64));
      return dst;
    },

    async sigmoid(src, n) {
      const p = await get('sigmoid_' + WG_EW, () => sigmoidShader(WG_EW));
      const dst = ctx.alloc(n * 4, 'sigmoid.out');
      const params = ctx.uniform([new Uint32Array([n, 0, 0, 0])]);
      await dispatchN(ctx, 'sigmoid', p, [src, dst, params], Math.ceil(n / WG_EW));
      return dst;
    },

    upload(array, label = 'buf') {
      const buf = ctx.alloc(array.byteLength, label);
      device.queue.writeBuffer(buf, 0, array);
      return buf;
    },
    alloc: (bytes, label) => ctx.alloc(bytes, label),
    ctx,
  };
}

// ── Operaciones del bloque de atención ──────────────────────────────────────
export function createAttnOps(device, opts = {}) {
  const ctx = opts.ctx || createContext(device, opts);
  const get = ctx.pipeline;
  const WG = opts.wg || 64;
  const WG_EW = opts.wgElementwise || 256;

  return {
    // Extrae Q o la puerta del tensor intercalado. `offset` 0 = Q,
    // `offset` headDim = puerta.
    async strideCopy(src, { headDim, heads, tokens, stride, offset }) {
      const p = await get('stride_copy_' + WG_EW, () => strideCopyShader(WG_EW));
      const dst = ctx.alloc(headDim * heads * tokens * 4, 'stride.out');
      const params = ctx.uniform([new Uint32Array([headDim, heads, tokens, stride, offset, 0, 0, 0])]);
      await dispatchN(ctx, 'stride_copy', p, [src, dst, params],
        Math.ceil(headDim * heads * tokens / WG_EW));
      return dst;
    },

    // QKᵀ SIN escalar ni enmascarar: de eso se encarga el softmax.
    async scores(q, k, { headDim, heads, kvHeads, tokens, nKv, nPast = 0, kvF16 = false }) {
      // La precisión va en la CLAVE del pipeline. Sin ella, la primera llamada
      // fijaría el shader y la segunda reutilizaría el compilado para la otra
      // precisión: leería la caché con el tipo equivocado y devolvería basura
      // sin un solo error.
      // Camino con la K COMPARTIDA entre las cabezas que la usan. El de un hilo
      // por salida va al 96 % del techo de lectura —no es ineficiente— pero lee
      // la K heads/kvHeads veces: con las formas del 27B, seis. Compartirla
      // divide los bytes por seis, es EXACTO (mismo producto, mismo orden de
      // acumulación) y a contexto largo era el 69 % de lo que el contexto añade
      // al token. Las condiciones son del dispositivo y del modelo, no gustos;
      // si alguna falla se usa el de siempre, que da lo mismo más despacio.
      const grupo = kvHeads > 0 ? heads / kvHeads : 0;
      const puedeCompartir = globalThis.__ELFFUSS_SCORES_SHARED !== false &&
        Number.isInteger(grupo) && grupo >= 1 &&
        compartidaScores(headDim, grupo) <= device.limits.maxComputeWorkgroupStorageSize &&
        Math.ceil(nKv / WG) <= 65535;
      if (puedeCompartir) {
        const pS = await get(`attn_scores_sh_${headDim}_${grupo}_${WG}${kvF16 ? '_h' : ''}`,
          () => attnScoresSharedShader({ headDim, grupo, kvF16, wg: WG }));
        const dst = ctx.alloc(heads * tokens * nKv * 4, 'scores');
        const params = ctx.uniform([new Uint32Array([headDim, heads, kvHeads, tokens, nKv, 0, 0, 0])]);
        await dispatchXYZ(ctx, 'attn_scores_sh', pS, [q, k, dst, params],
          Math.ceil(nKv / WG), kvHeads, tokens);
        return dst;
      }
      const p = await get(`attn_scores_${WG}${kvF16 ? '_h' : ''}`, () => attnScoresShader(WG, kvF16));
      const dst = ctx.alloc(heads * tokens * nKv * 4, 'scores');
      const params = ctx.uniform([new Uint32Array([headDim, heads, kvHeads, tokens, nKv, nPast, 0, 0])]);
      await dispatchN(ctx, 'attn_scores', p, [q, k, dst, params],
        Math.ceil(heads * tokens * nKv / WG));
      return dst;
    },

    // El camino PARTIDO es el normal; el de un hilo por salida se queda como
    // respaldo para las formas que el partido no puede atender. A contexto
    // largo la diferencia no es un detalle: a 64k de contexto el kernel viejo se
    // llevaba el 73 % del tiempo del token, y el partido mide dos ordenes de
    // magnitud menos en el mismo sitio.
    //
    // Tres condiciones para poder usarlo, y las tres son del dispositivo o del
    // modelo, no preferencias:
    //   · un grupo tiene headDim hilos, así que headDim tiene que caber en el
    //     tope de invocaciones por grupo del dispositivo;
    //   · las cabezas de consulta tienen que repartirse exactas entre las KV,
    //     porque un grupo atiende a todas las de una cabeza KV;
    //   · la memoria compartida de las probabilidades tiene que caber.
    // Si alguna falla NO se falla: se usa el camino de siempre, que da el mismo
    // resultado más despacio. Un modelo con otra forma tiene que seguir
    // corriendo.
    async combine(probs, v, { headDim, heads, kvHeads, tokens, nKv, nPast = 0, kvF16 = false }) {
      const grupo = kvHeads > 0 ? heads / kvHeads : 0;
      const SUB = 64;
      const puedePartir = globalThis.__ELFFUSS_COMBINE_SPLIT !== false &&
        Number.isInteger(grupo) && grupo >= 1 &&
        headDim <= device.limits.maxComputeInvocationsPerWorkgroup &&
        grupo * SUB * 4 <= device.limits.maxComputeWorkgroupStorageSize;
      if (puedePartir) {
        const bloques = bloquesCombine(nKv);
        const pS = await get(`attn_combine_split_${headDim}_${grupo}_${SUB}${kvF16 ? '_h' : ''}`,
          () => attnCombineSplitShader({ headDim, grupo, subTrozo: SUB, kvF16 }));
        const pR = await get(`combine_reduce_${WG_EW}`, () => combineReduceShader(WG_EW));
        const parc = ctx.alloc(tokens * heads * bloques * headDim * 4, 'attn.parc');
        const dst = ctx.alloc(tokens * heads * headDim * 4, 'attn.out');
        const parS = ctx.uniform([new Uint32Array([headDim, heads, kvHeads, tokens, nKv, bloques, 0, 0])]);
        const parR = ctx.uniform([new Uint32Array([headDim, heads, tokens, bloques])]);
        await dispatchXYZ(ctx, 'attn_combine_split', pS, [probs, v, parc, parS],
          bloques, kvHeads, tokens);
        await dispatchN(ctx, 'attn_combine_reduce', pR, [parc, dst, parR],
          Math.ceil(tokens * heads * headDim / WG_EW));
        return dst;
      }
      const p = await get(`attn_combine_${WG}${kvF16 ? '_h' : ''}`, () => attnCombineShader(WG, kvF16));
      const dst = ctx.alloc(tokens * heads * headDim * 4, 'attn.out');
      const params = ctx.uniform([new Uint32Array([headDim, heads, kvHeads, tokens, nKv, nPast, 0, 0])]);
      await dispatchN(ctx, 'attn_combine', p, [probs, v, dst, params],
        Math.ceil(tokens * heads * headDim / WG));
      return dst;
    },

    async mul(a, b, n) {
      const p = await get('mul_' + WG_EW, () => mulShader(WG_EW));
      const dst = ctx.alloc(n * 4, 'mul.out');
      const params = ctx.uniform([new Uint32Array([n, 0, 0, 0])]);
      await dispatchN(ctx, 'mul', p, [a, b, dst, params], Math.ceil(n / WG_EW));
      return dst;
    },

    upload(array, label = 'buf') {
      const buf = ctx.alloc(array.byteLength, label);
      device.queue.writeBuffer(buf, 0, array);
      return buf;
    },
    alloc: (bytes, label) => ctx.alloc(bytes, label),
    ctx,
  };
}

// ── Referencias de CPU para los tests ───────────────────────────────────────
// Deliberadamente ABURRIDAS y escritas de otra forma que el shader, para que un
// fallo del WGSL no se cancele con el mismo fallo aquí. Los Number de JS son
// dobles, así que estas referencias acumulan en f64 de verdad.

// La descuantización de referencia de los tipos de peso vive aparte, en un
// módulo que solo usan las pruebas: importarla desde aquí arrastraría las
// rejillas de los i-quants a todo el que solo quiere un rmsnorm.

export function rmsnormCPU(src, w, rows, ne0, eps) {
  const out = new Float64Array(rows * ne0);
  for (let r = 0; r < rows; r++) {
    let sum = 0;
    for (let i = 0; i < ne0; i++) { const v = src[r * ne0 + i]; sum += v * v; }
    const scale = 1 / Math.sqrt(sum / ne0 + eps);
    for (let i = 0; i < ne0; i++) out[r * ne0 + i] = src[r * ne0 + i] * scale * w[i];
  }
  return out;
}

export function ropeCPU(src, pos, { rows, ne0, heads, nDims, mode, freqBase, freqScale = 1 }) {
  const out = Float64Array.from(src);
  const pairs = nDims / 2;
  for (let r = 0; r < rows; r++) {
    const base = r * ne0;
    const t = Math.floor(r / heads);
    for (let p = 0; p < pairs; p++) {
      // Forma cerrada, distinta de la del shader (que usa thetaScale^p).
      const theta = pos[t] * freqScale / Math.pow(freqBase, (2 * p) / nDims);
      const c = Math.cos(theta), s = Math.sin(theta);
      const i0 = mode === ROPE_NEOX ? p : 2 * p;
      const i1 = mode === ROPE_NEOX ? p + pairs : 2 * p + 1;
      const x0 = src[base + i0], x1 = src[base + i1];
      out[base + i0] = x0 * c - x1 * s;
      out[base + i1] = x0 * s + x1 * c;
    }
  }
  return out;
}

export function softmaxCPU(src, { rows, nKv, nQ, nPast = 0, scale = 1 }) {
  const out = new Float64Array(rows * nKv);
  for (let r = 0; r < rows; r++) {
    const base = r * nKv, qAbs = nPast + (r % nQ);
    let max = -Infinity;
    for (let k = 0; k <= Math.min(qAbs, nKv - 1); k++) max = Math.max(max, src[base + k] * scale);
    let sum = 0;
    for (let k = 0; k < nKv; k++) {
      const e = k <= qAbs ? Math.exp(src[base + k] * scale - max) : 0;
      out[base + k] = e; sum += e;
    }
    for (let k = 0; k < nKv; k++) out[base + k] /= sum;
  }
  return out;
}

export function swigluCPU(gate, up, n) {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = (gate[i] / (1 + Math.exp(-gate[i]))) * up[i];
  return out;
}

// ── Referencias de CPU de la capa SSM ───────────────────────────────────────
// Escritas siguiendo el bucle de referencia paso a paso (decaimiento explícito,
// tres pasadas separadas), no la forma optimizada del shader: si las dos
// compartieran el atajo algebraico, el test no lo estaría comprobando.

export function l2normCPU(src, rows, ne0, eps) {
  const out = new Float64Array(rows * ne0);
  for (let r = 0; r < rows; r++) {
    let sum = 0;
    for (let i = 0; i < ne0; i++) { const v = src[r * ne0 + i]; sum += v * v; }
    const scale = 1 / Math.max(Math.sqrt(sum), eps);     // suelo, no eps bajo la raíz
    for (let i = 0; i < ne0; i++) out[r * ne0 + i] = src[r * ne0 + i] * scale;
  }
  return out;
}

export function ssmConvCPU(buf, w, channels, tokens) {
  const window = tokens + SSM_CONV_K - 1;
  const out = new Float64Array(channels * tokens);
  for (let c = 0; c < channels; c++) {
    for (let t = 0; t < tokens; t++) {
      let acc = 0;
      for (let r = 0; r < SSM_CONV_K; r++) acc += buf[c * window + t + r] * w[c * SSM_CONV_K + r];
      out[t * channels + c] = acc;
    }
  }
  return out;
}

// `heads` son las de VALOR y `kvHeads` las de CLAVE. El emparejamiento es por
// MÓDULO (cabeza de valor h ↔ cabeza de clave h % kvHeads), no por división:
// las cabezas de valor vienen del fichero en orden TESELADO. Ver la nota larga
// en el shader. Cuando los dos recuentos coinciden da igual, que es el caso del
// 0.8B y la razón de que el fallo solo apareciera en el modelo grande.
export function gatedDeltaNetCPU(q, k, v, g, beta, state, heads, tokens,
  S = SSM_HEAD_DIM, kvHeads = heads) {
  const st = Float64Array.from(state);                   // [H][S(j)][S(i)]
  const out = new Float64Array(tokens * heads * S);
  const scale = 1 / Math.sqrt(S);
  for (let t = 0; t < tokens; t++) {
    for (let h = 0; h < heads; h++) {
      const vv = (t * heads + h) * S, hb = h * S * S;    // v, g, beta y salida
      const vk = (t * kvHeads + (h % kvHeads)) * S;      // q y k
      const gv = Math.exp(g[t * heads + h]), bv = beta[t * heads + h];
      // 1) decaimiento, explícito
      for (let x = 0; x < S * S; x++) st[hb + x] *= gv;
      // 2) u = v − Sᵀk ; 3) δ = β·u
      const delta = new Float64Array(S);
      for (let j = 0; j < S; j++) {
        let dot = 0;
        for (let i = 0; i < S; i++) dot += st[hb + j * S + i] * k[vk + i];
        delta[j] = (v[vv + j] - dot) * bv;
      }
      // 4) escritura
      for (let j = 0; j < S; j++) {
        for (let i = 0; i < S; i++) st[hb + j * S + i] += k[vk + i] * delta[j];
      }
      // 5) salida con el estado YA actualizado
      for (let j = 0; j < S; j++) {
        let dot = 0;
        for (let i = 0; i < S; i++) dot += st[hb + j * S + i] * q[vk + i];
        out[vv + j] = dot * scale;
      }
    }
  }
  return { out, state: st };
}

export function gatedNormCPU(src, w, z, rows, ne0, eps) {
  const out = new Float64Array(rows * ne0);
  for (let r = 0; r < rows; r++) {
    let sum = 0;
    for (let i = 0; i < ne0; i++) { const x = src[r * ne0 + i]; sum += x * x; }
    const scale = 1 / Math.sqrt(sum / ne0 + eps);
    for (let i = 0; i < ne0; i++) {
      const zc = z[r * ne0 + i];
      out[r * ne0 + i] = src[r * ne0 + i] * scale * w[i] * (zc / (1 + Math.exp(-zc)));
    }
  }
  return out;
}

export function decayGateCPU(alpha, dt, a, heads, tokens) {
  const out = new Float64Array(heads * tokens);
  const sp = x => (x > 20 ? x : Math.log(1 + Math.exp(x)));
  for (let i = 0; i < heads * tokens; i++) out[i] = sp(alpha[i] + dt[i % heads]) * a[i % heads];
  return out;
}

// Las tres familias compartiendo UN contexto: un solo submit puede abarcar
// operaciones cuantizadas, de red y de SSM, que es lo que hace un forward.
export function createOps(device, opts = {}) {
  const ctx = createContext(device, opts);
  const shared = { ...opts, ctx };
  const quant = createQuantOps(device, shared);
  return {
    ctx,
    quant,
    // Vista Q8_0 de `quant`, para el forward del 0.8B, que no pasa el tipo.
    q8: {
      dequant: (w, n) => quant.dequant('Q8_0', w, n),
      matmul: (w, x, rows, k, tokens = 1) => quant.matmul('Q8_0', w, x, rows, k, tokens),
      matvec: (w, x, rows, k) => quant.matmul('Q8_0', w, x, rows, k, 1),
      upload: (a, l) => quant.upload(a, l),
      ctx,
    },
    nn: createNNOps(device, shared),
    ssm: createSSMOps(device, shared),
    attn: createAttnOps(device, shared),
    batch: (label, fn) => ctx.batch(label, fn),
    // Despacho generico para los kernels de plomeria del modelo, que no
    // pertenecen a ninguna de las tres familias.
    dispatchAny: (label, pipe, buffers, groups) => dispatchN(ctx, label, pipe, buffers, groups),
    recycle: () => ctx.recycle(),
    // Los scopes de error cuestan una ida y vuelta por operacion. Se dejan
    // puestos al cargar el modelo y se quitan para el bucle de decode.
    setDebug: (on) => { ctx.debug = !!on; },
  };
}

// ── Referencias de CPU del bloque de atención ───────────────────────────────

export function strideCopyCPU(src, { headDim, heads, tokens, stride, offset }) {
  const out = new Float64Array(headDim * heads * tokens);
  for (let t = 0; t < tokens; t++) {
    for (let h = 0; h < heads; h++) {
      for (let d = 0; d < headDim; d++) {
        out[(t * heads + h) * headDim + d] =
          src[t * heads * stride + h * stride + offset + d];
      }
    }
  }
  return out;
}

export function attnScoresCPU(q, k, { headDim, heads, kvHeads, tokens, nKv }) {
  const out = new Float64Array(heads * tokens * nKv);
  const group = heads / kvHeads;
  for (let h = 0; h < heads; h++) {
    const kvh = Math.floor(h / group);
    for (let t = 0; t < tokens; t++) {
      for (let j = 0; j < nKv; j++) {
        let acc = 0;
        for (let d = 0; d < headDim; d++) {
          acc += q[(t * heads + h) * headDim + d] * k[(j * kvHeads + kvh) * headDim + d];
        }
        out[(h * tokens + t) * nKv + j] = acc;
      }
    }
  }
  return out;
}

export function attnCombineCPU(probs, v, { headDim, heads, kvHeads, tokens, nKv }) {
  const out = new Float64Array(tokens * heads * headDim);
  const group = heads / kvHeads;
  for (let t = 0; t < tokens; t++) {
    for (let h = 0; h < heads; h++) {
      const kvh = Math.floor(h / group);
      for (let d = 0; d < headDim; d++) {
        let acc = 0;
        for (let j = 0; j < nKv; j++) {
          acc += probs[(h * tokens + t) * nKv + j] * v[(j * kvHeads + kvh) * headDim + d];
        }
        out[(t * heads + h) * headDim + d] = acc;
      }
    }
  }
  return out;
}
