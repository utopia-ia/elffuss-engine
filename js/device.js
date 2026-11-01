// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · creación del GPUDevice y utillaje que hace visibles los fallos.
// ─────────────────────────────────────────────────────────────────────────────
// En WebGPU casi nada lanza. Un buffer inválido, un binding mal alineado o una
// lectura fuera de rango no dan excepción: dan CEROS. Un motor de inferencia
// construido sobre eso genera texto plausible y basura numérica sin un solo
// error en consola. Este módulo existe para que eso no pase:
//   · pide los límites de verdad (ver la trampa de abajo, es gorda),
//   · comprueba que el device no nace ya perdido,
//   · envuelve cualquier operación en scopes de error,
//   · y ofrece el round-trip de lectura, que es el único canario fiable.
//
// TRAMPA MAYOR — `requestDevice({ requiredLimits: adapter.limits })` NO falla:
// GPUSupportedLimits expone sus límites como accessors del PROTOTIPO, y la
// conversión a `record<K,V>` de WebIDL solo mira propiedades PROPIAS. Así que
// `{...adapter.limits}` es `{}` y te devuelve un device con los límites POR
// DEFECTO (buffer 256 MiB, binding 128 MiB) sin decir nada. Hay que recorrerlo
// con for..in, que sí pasa por el prototipo. (gpuweb/gpuweb#4277)
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.

// Los dos únicos límites de clase «alignment», donde MEJOR es MENOR y el valor
// debe ser potencia de dos. El resto son de clase «maximum» (mejor = mayor).
// Pedir algo peor que el defecto no da error: se clampa en silencio.
const ALIGNMENT_LIMITS = new Set([
  'minUniformBufferOffsetAlignment',
  'minStorageBufferOffsetAlignment',
]);

// Los límites que de verdad deciden si un modelo cabe y cómo hay que trocearlo.
// Se usan como conjunto de reintento si pedirlo todo falla en algún navegador.
const CORE_LIMITS = new Set([
  'maxBufferSize',
  'maxStorageBufferBindingSize',
  'maxStorageBuffersPerShaderStage',
  'maxBindGroups',
  'minStorageBufferOffsetAlignment',
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupSizeX',
  'maxComputeWorkgroupSizeY',
  'maxComputeWorkgroupStorageSize',
  'maxComputeWorkgroupsPerDimension',
]);

// Copia los límites del adaptador a un objeto JS plano. for..in y no Object.keys
// ni spread: ver la trampa de la cabecera. Como solo copiamos valores que el
// propio adaptador reporta, por construcción nunca pedimos nada «mejor» que él.
export function maxOutLimits(adapter, only = null) {
  const out = {};
  for (const key in adapter.limits) {
    if (only && !only.has(key)) continue;
    const v = adapter.limits[key];
    if (typeof v !== 'number' || !Number.isFinite(v)) continue;
    // Un límite de alineación que no sea potencia de dos es inválido y tumbaría
    // la petición entera; se salta en vez de arrastrarlo.
    if (ALIGNMENT_LIMITS.has(key) && (v <= 0 || !Number.isInteger(Math.log2(v)))) continue;
    out[key] = v;
  }
  return out;
}

// Crea el device con todo lo que el adaptador dé. Devuelve también si hay f16 y
// la información del adaptador, que es lo que permite detectar un adaptador de
// software haciéndose pasar por uno de verdad.
export async function requestDevice(opts = {}) {
  if (!globalThis.navigator?.gpu) {
    // Sin secure context (http:// que no sea localhost, o about:blank) el
    // navegador ni siquiera define navigator.gpu, y parece un fallo del dispositivo.
    throw new Error('sin WebGPU: ¿contexto seguro? (https o 127.0.0.1)');
  }
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: opts.powerPreference || 'high-performance',
  });
  if (!adapter) throw new Error('no hay adaptador WebGPU');

  const wantF16 = adapter.features.has('shader-f16') && opts.f16 !== false;
  const wantSG = adapter.features.has('subgroups') && opts.subgroups !== false;
  // timestamp-query solo se pide si el llamador lo quiere (perfilado); no lo usa
  // producción, así que por defecto no se activa.
  const wantTS = adapter.features.has('timestamp-query') && opts.timestamp === true;
  const required = [...(wantF16 ? ['shader-f16'] : []), ...(wantSG ? ['subgroups'] : []),
                    ...(wantTS ? ['timestamp-query'] : [])];
  // Tamaño de subgrupo SOLO si es fijo (min==max): con un tamaño fijo un
  // workgroup de ese ancho es EXACTAMENTE un subgrupo y `subgroupAdd` da la
  // reducción entera sin barreras. Si varía, no se asume nada (queda en 0 = off).
  const sgSize = wantSG && adapter.info && adapter.info.subgroupMinSize === adapter.info.subgroupMaxSize
    ? (adapter.info.subgroupMinSize || 0) : 0;

  let device;
  try {
    device = await adapter.requestDevice({
      label: opts.label || 'elffuss-engine',
      requiredFeatures: required,
      requiredLimits: maxOutLimits(adapter),
    });
  } catch (e) {
    // requestDevice es de UN SOLO USO: tras fallar hay que pedir OTRO adaptador.
    // El reintento se queda en el núcleo de límites, por si el navegador reporta
    // alguno inconsistente consigo mismo.
    const a2 = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!a2) throw e;
    device = await a2.requestDevice({
      label: (opts.label || 'elffuss-engine') + ' (reintento)',
      requiredFeatures: a2.features.has('shader-f16') && opts.f16 !== false ? ['shader-f16'] : [],
      requiredLimits: maxOutLimits(a2, CORE_LIMITS),
    });
  }

  // requestDevice puede RESOLVER con un device ya perdido en vez de rechazar
  // (la spec lo dice literalmente). Sin esta carrera, el primer síntoma sería
  // un mapAsync que se cuelga treinta llamadas más tarde.
  const lost = device.lost.then(i => i);
  const race = await Promise.race([lost, Promise.resolve(ALIVE)]);
  if (race !== ALIVE) throw new Error('el device nació perdido: ' + race.reason + ' ' + race.message);

  // Red de seguridad. La spec NO garantiza que dispare y permite limitarlo, así
  // que no sirve para controlar nada — pero registrar el listener puede silenciar
  // el aviso de consola, así que si se registra hay que re-emitirlo.
  device.addEventListener?.('uncapturederror', (ev) => {
    console.error('[gpu] error no capturado:', ev.error?.constructor?.name, ev.error?.message);
  });

  return {
    device,
    adapter,
    info: adapter.info || null,
    // Lo que te DIERON, nunca lo que pediste.
    hasF16: device.features.has('shader-f16'),
    // 32 en muchos adaptadores. 0 si no hay subgrupos o el tamaño no es fijo → el motor cae
    // a la reducción en árbol de siempre.
    subgroupSize: device.features.has('subgroups') ? sgSize : 0,
    hasTimestamp: device.features.has('timestamp-query'),
    limits: device.limits,
  };
}

const ALIVE = Symbol('alive');

// Envuelve una operación en los tres scopes de error. Sin esto, «me devuelve
// ceros» nunca se convierte en «createBindGroup falló porque el offset no es
// múltiplo de 256», que es la misma información con veinte veces más valor.
export async function guard(device, label, fn) {
  device.pushErrorScope('validation');
  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('internal');
  let out, thrown = null;
  try { out = await fn(); } catch (e) { thrown = e; }
  // LIFO, y con await individual: la spec no garantiza el orden de resolución.
  const internal = await device.popErrorScope();
  const oom = await device.popErrorScope();
  const validation = await device.popErrorScope();
  if (thrown) throw thrown;
  const first = validation || oom || internal;
  if (first) {
    // El tipo es estable, el mensaje es localizable: se reportan los dos.
    throw new Error(`[${label}] ${first.constructor.name}: ${first.message}`);
  }
  return out;
}

// Lee un rango de un buffer de vuelta a CPU. Es el único canario fiable: en un
// device perdido, mapAsync es la ÚNICA operación que falla de forma observable
// (el resto «aparenta funcionar»).
export async function readback(device, src, offset = 0, size = src.size - offset) {
  if (offset % 8) throw new Error('readback: el offset debe ser múltiplo de 8 (mapAsync)');
  if (size % 4) throw new Error('readback: el tamaño debe ser múltiplo de 4');
  // (Salvo si redondear la talla del buffer de lectura pasara de `maxBufferSize`:
  // ahí, el de siempre, que lo pide exacto.)
  if (globalThis.__ELFFUSS_PLAN === true &&
      Math.max(256, Math.ceil(size / 256) * 256) <= (device.limits?.maxBufferSize ?? Infinity)) {
    return await lecturaPersistente(device, src, offset, size);
  }

  // MAP_READ solo puede combinarse con COPY_DST. Añadir STORAGE o COPY_SRC deja
  // el buffer inválido — y devolviendo ceros, cómo no.
  const staging = device.createBuffer({
    label: 'readback',
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    await guard(device, 'readback/copy', async () => {
      const enc = device.createCommandEncoder({ label: 'readback' });
      enc.copyBufferToBuffer(src, offset, staging, 0, size);
      device.queue.submit([enc.finish()]);
    });
    await staging.mapAsync(GPUMapMode.READ, 0, size);
    const mapped = staging.getMappedRange(0, size);
    // Copiar ANTES de unmap: unmap desacopla el ArrayBuffer (byteLength 0) y
    // cualquier vista que hubieras hecho sobre él se queda a cero. Ese es el
    // origen del clásico «me sale todo ceros» que no es del dispositivo.
    const copy = mapped.slice(0);
    staging.unmap();
    return copy;
  } finally {
    staging.destroy();
  }
}

// La misma lectura con PLAN (`__ELFFUSS_PLAN`): el bucle de generación lee en
// cada token, y cada lectura creaba y destruía su buffer y pagaba tres idas y
// vueltas de ámbitos de error. Aquí los buffers de lectura se GUARDAN, dos por
// tamaño y turnándose (mientras uno está mapeado el otro está libre), y los
// ámbitos de error se quedan donde un fallo de configuración se descubre: la
// primera vez que se usa cada buffer, o si la copia pedida no se puede hacer.
// Una copia inválida sin ámbito no lanza: dejaría en el buffer lo de la lectura
// anterior. Por eso lo que sí se puede comprobar en CPU se comprueba antes, y
// si no cuadra se va por el camino con ámbitos. Lo único que no se puede mirar
// desde la CPU es un origen YA DESTRUIDO: el camino de siempre lanzaría y este
// devolvería datos viejos. Los logits salen del pozo, que no destruye nada.
// `ocupado` marca cada buffer desde que se elige hasta que se desmapea. El
// estado de mapeo NO basta: solo cambia al llamar a mapAsync, y entre la copia
// y esa llamada está la espera del ámbito de la primera vez. Tres lecturas a la
// vez del mismo tamaño elegían así el mismo buffer, y la tercera copia pisaba
// la de la primera antes de que esta lo mapeara.
const pozosLectura = new WeakMap();      // device → Map(tamaño → { bufs, validado, ocupado, turno })
const CLASES_LECTURA = 8;

async function lecturaPersistente(device, src, offset, size) {
  let pozo = pozosLectura.get(device);
  if (!pozo) { pozo = new Map(); pozosLectura.set(device, pozo); }
  const clase = Math.max(256, Math.ceil(size / 256) * 256);
  let e = pozo.get(clase);
  if (!e) {
    // Con demasiados tamaños distintos (un banco que lee de todo) se suelta la
    // clase más vieja que no esté en uso.
    if (pozo.size >= CLASES_LECTURA) {
      for (const [c, v] of pozo) {
        if (!v.ocupado[0] && !v.ocupado[1]) {
          for (const b of v.bufs) b?.destroy();
          pozo.delete(c);
          break;
        }
      }
    }
    e = { bufs: [null, null], validado: [false, false], ocupado: [false, false], turno: 0 };
    pozo.set(clase, e);
  }
  let k = e.turno; e.turno ^= 1;
  if (e.ocupado[k]) k ^= 1;
  let staging, temporal = false;
  if (e.ocupado[k]) {
    // Las dos ocupadas (lecturas concurrentes): una de usar y tirar.
    staging = device.createBuffer({ label: 'readback', size: clase,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    temporal = true;
  } else {
    if (!e.bufs[k] || e.bufs[k].mapState !== 'unmapped') {
      e.bufs[k] = device.createBuffer({ label: 'readback', size: clase,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      e.validado[k] = false;
    }
    staging = e.bufs[k];
    e.ocupado[k] = true;
  }
  const copia = () => {
    const enc = device.createCommandEncoder({ label: 'readback' });
    enc.copyBufferToBuffer(src, offset, staging, 0, size);
    device.queue.submit([enc.finish()]);
  };
  const cuadra = typeof src.usage === 'number' && (src.usage & GPUBufferUsage.COPY_SRC) !== 0 &&
    offset + size <= src.size;
  try {
    if (temporal || !e.validado[k] || !cuadra) await guard(device, 'readback/copy', async () => copia());
    else copia();
    await staging.mapAsync(GPUMapMode.READ, 0, size);
    const copy = staging.getMappedRange(0, size).slice(0);
    staging.unmap();
    if (!temporal) e.validado[k] = true;
    return copy;
  } catch (err) {
    // Tras un fallo no se sabe en qué estado queda el buffer: fuera.
    if (!temporal) { try { staging.destroy(); } catch { /* — */ } e.bufs[k] = null; e.validado[k] = false; }
    throw err;
  } finally {
    if (temporal) staging.destroy();
    else e.ocupado[k] = false;
  }
}

// Preludio del WGSL. Con la extensión se puede ALMACENAR en f16 (que es lo que
// de verdad compra: la mitad de presión en memoria compartida, que es el recurso
// que limita el tamaño de tile del matmul). Sin ella se degrada el rendimiento,
// no se deja de arrancar. Convertir la escala f16→f32 NO necesita la extensión:
// unpack2x16float es del núcleo de WGSL.
export function wgslPrelude(hasF16) {
  return hasF16
    ? 'enable f16;\nalias half = f16;\nalias half4 = vec4<f16>;\n'
    : 'alias half = f32;\nalias half4 = vec4<f32>;\n';
}

// ¿El adaptador es de verdad o es software haciéndose pasar por uno?
// Según cómo se arranque Chromium, `requestAdapter()` puede devolver un
// adaptador SwiftShader —software puro— que funciona y miente en cada medición.
export function isSoftwareAdapter(info) {
  const s = ((info?.vendor || '') + ' ' + (info?.architecture || '') + ' ' +
    (info?.description || '')).toLowerCase();
  return /swiftshader|llvmpipe|lavapipe|software|warp|basic render/.test(s);
}

const GiB = 1024 ** 3, MiB = 1024 ** 2;

// Resumen de una línea para diagnóstico, sin datos del usuario.
export function summarize({ info, hasF16, limits }) {
  const gpu = info ? `${info.vendor || '?'}/${info.architecture || '?'}` : 'adaptador anónimo';
  return `${gpu} · buffer≤${(limits.maxBufferSize / GiB).toFixed(1)}GiB` +
    ` · binding≤${(limits.maxStorageBufferBindingSize / MiB) | 0}MiB` +
    ` · ${limits.maxStorageBuffersPerShaderStage} storage/stage` +
    ` · align ${limits.minStorageBufferOffsetAlignment}` +
    ` · f16 ${hasF16 ? 'sí' : 'no'}`;
}
