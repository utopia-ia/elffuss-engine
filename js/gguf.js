// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · parser de la cabecera GGUF (metadatos + tabla de tensores).
// ─────────────────────────────────────────────────────────────────────────────
// Primer módulo del MOTOR: sin esto no sabemos qué hay dentro de un modelo ni
// en qué byte empieza cada tensor. Lee por RANGOS con el handle que devuelve
// openRanged() de model-store.js ({size, slice(offset,length)}), NUNCA el
// fichero entero: un GGUF de 7B ya ocupa gigas y volcarlo a un ArrayBuffer mata
// la pestaña — que es justo el problema que nos saca de la dependencia actual.
//
// La zona de metadatos (cabecera + pares clave/valor + tabla de tensores) ocupa
// de unos KB a unos MB, dominada por el vocabulario del tokenizer. Se lee con
// una ventana que arranca en 1 MiB y DOBLA cuando el cursor se sale: un modelo
// entero se parsea leyendo una fracción pequeña del fichero, no el fichero.
//
// Paridad con llama.cpp (ggml/src/gguf.cpp): se replican sus validaciones para
// rechazar exactamente los mismos ficheros. Un parser permisivo no te avisa de
// que el fichero está roto, te da tensores en offsets absurdos y lo descubres
// tres capas más arriba con basura en la salida.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.

const MAGIC = [0x47, 0x47, 0x55, 0x46];            // 'G','G','U','F' en disco
const VERSION_MAX = 3;                             // v1 rechazada, no hay ruta
const DEFAULT_ALIGNMENT = 32;                      // GGUF_DEFAULT_ALIGNMENT
const MAX_STRING_LENGTH = 1024 ** 3;               // GGUF_MAX_STRING_LENGTH
const MAX_ARRAY_ELEMENTS = 1024 ** 3;              // GGUF_MAX_ARRAY_ELEMENTS
const MAX_DIMS = 4;                                // GGML_MAX_DIMS
const MAX_NAME = 64;                               // GGML_MAX_NAME (nombre ≤ 63)
const FIRST_WINDOW = 1024 ** 2;                    // 1 MiB de arranque
const MAX_WINDOW = 512 * 1024 ** 2;                // tope del doblado a ciegas
const BOOL_ARRAY_EAGER = 65536;                    // por encima, se deja crudo

// Tipos de valor de metadatos. El índice es el código en disco.
const V_UINT8 = 0, V_INT8 = 1, V_UINT16 = 2, V_INT16 = 3, V_UINT32 = 4,
  V_INT32 = 5, V_FLOAT32 = 6, V_BOOL = 7, V_STRING = 8, V_ARRAY = 9,
  V_UINT64 = 10, V_INT64 = 11, V_FLOAT64 = 12, V_COUNT = 13;

const V_NAMES = ['uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32',
  'float32', 'bool', 'string', 'array', 'uint64', 'int64', 'float64'];

// Tamaño en bytes de cada tipo escalar (los variables van a 0 y se tratan
// aparte). BOOL ocupa UN byte, no cuatro: leerlo como u32 desplaza el resto del
// fichero y el parseo explota mucho más adelante, con un error incomprensible.
const V_SIZES = [1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8];

// Tabla ggml_type: [nombre, blockSize, typeSize]. Los códigos muertos (Q4_2,
// Q4_3, los *_4_4/4_8/8_8 retirados) van con blockSize 0 A PROPÓSITO: dividir
// por su block size daría Infinity/NaN y se propagaría en silencio a todos los
// offsets. Con 0 se rechazan, igual que hace llama.cpp.
const TYPES = [
  ['F32', 1, 4], ['F16', 1, 2], ['Q4_0', 32, 18], ['Q4_1', 32, 20],
  ['Q4_2', 0, 0], ['Q4_3', 0, 0], ['Q5_0', 32, 22], ['Q5_1', 32, 24],
  ['Q8_0', 32, 34], ['Q8_1', 32, 40], ['Q2_K', 256, 84], ['Q3_K', 256, 110],
  ['Q4_K', 256, 144], ['Q5_K', 256, 176], ['Q6_K', 256, 210],
  ['Q8_K', 256, 292], ['IQ2_XXS', 256, 66], ['IQ2_XS', 256, 74],
  ['IQ3_XXS', 256, 98], ['IQ1_S', 256, 50], ['IQ4_NL', 32, 18],
  ['IQ3_S', 256, 110], ['IQ2_S', 256, 82], ['IQ4_XS', 256, 136],
  ['I8', 1, 1], ['I16', 1, 2], ['I32', 1, 4], ['I64', 1, 8], ['F64', 1, 8],
  ['IQ1_M', 256, 56], ['BF16', 1, 2], ['Q4_0_4_4', 0, 0], ['Q4_0_4_8', 0, 0],
  ['Q4_0_8_8', 0, 0], ['TQ1_0', 256, 54], ['TQ2_0', 256, 66],
  ['IQ4_NL_4_4', 0, 0], ['IQ4_NL_4_8', 0, 0], ['IQ4_NL_8_8', 0, 0],
  ['MXFP4', 32, 17], ['NVFP4', 64, 36], ['Q1_0', 128, 18], ['Q2_0', 64, 18],
];

// TIPOS FUERA DEL CATALOGO DE ggml. Los codigos 142/143 no son de llama.cpp:
// los define un fork de terceros para sus formatos ternarios, y por eso van
// APARTE y con su nombre completo. Ponerlos en la tabla de arriba sugeriria que
// son estandar y que cualquier GGUF con ese codigo significa esto, que es falso.
//
//   PQ2_0  (142): 128 pesos, escala f16 + 2 bits por peso        = 34 bytes
//   PTQ1_0 (143): 128 pesos, 5 trits por byte en base 3 + escala = 28 bytes
//
// Ninguno lleva codebook: el peso sale de aritmetica, no de una tabla. Esa es
// la razon de soportarlos, no el tamaño.
TYPES[142] = ['PQ2_0', 128, 34];
TYPES[143] = ['PTQ1_0', 128, 28];

const DEC = new TextDecoder('utf-8');               // fatal:false a propósito
// Los vocabularios BPE llevan tokens de bytes crudos que NO son UTF-8 válido:
// con fatal:true el decodificador LANZA al llegar a ellos y no hay modelo.

// ¿el host es little-endian? GGUF lo es siempre en la práctica, y las vistas
// tipadas usan el orden del host: si algún día corremos en big-endian hay que
// decodificar a mano en vez de reinterpretar memoria.
const HOST_LE = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

// GGML_PAD sin operadores bitwise. La macro original es `((x)+(n)-1) & ~((n)-1)`
// y traducirla literal es LA trampa de este formato en JavaScript: & y ~ truncan
// a 32 bits con signo → cualquier offset por encima de 2^31 bytes sale negativo, o
// sea todos los modelos de 7B en adelante.
function pad(x, n) { return x + ((n - (x % n)) % n); }

function bad(msg) { throw new Error('GGUF: ' + msg); }

// Descripción de un ggml_type por su código, o null si no existe.
export function ggmlType(code) {
  const t = TYPES[code];
  return t ? { code, name: t[0], blockSize: t[1], typeSize: t[2] } : null;
}

// Bytes que ocupa un tensor: (ne0/blockSize) * typeSize * ne1 * ne2 * ne3.
// El tamaño NO es nElements * typeSize: en los tipos cuantizados el «elemento»
// es un bloque de 32 o 256 pesos con sus escalas dentro.
export function tensorBytes(dims, code) {
  const t = ggmlType(code);
  if (!t || t.blockSize === 0) return 0;
  let n = (dims[0] / t.blockSize) * t.typeSize;
  for (let i = 1; i < dims.length; i++) n *= dims[i];
  return n;
}

// Cursor de lectura sobre el handle por rangos. Mantiene una ventana que
// SIEMPRE empieza en el byte 0, así el offset del fichero y el índice del
// buffer son el mismo número: se evita de raíz toda una familia de bugs de
// «offset relativo a qué». Doblar y releer desde 0 cuesta como mucho 2× la
// ventana final, y la ventana final son megas, no gigas.
async function cursor(ranged, firstWindow) {
  let bytes = new Uint8Array(0);
  let view = new DataView(bytes.buffer);
  let loaded = 0, reads = 0, bytesRead = 0, pos = 0;

  async function ensure(end) {
    if (end <= loaded) return;
    if (end > ranged.size) bad('fichero truncado: se piden ' + end +
      ' bytes de un fichero de ' + ranged.size);
    let want = Math.max(firstWindow, loaded * 2);
    while (want < end) want *= 2;
    if (want > MAX_WINDOW && end <= MAX_WINDOW) want = MAX_WINDOW;
    want = Math.min(want, ranged.size);
    // Se lee SOLO la cola nueva y se concatena. La ventana sigue empezando en el
    // byte 0 (offset del fichero == índice del buffer), pero cada byte viaja del
    // disco UNA vez: releer desde 0 en cada doblado costaba ~2,2× el fichero en
    // los modelos pequeños, y eso es exactamente lo que openRanged viene a
    // evitar. Copiar en memoria es un memcpy; releer es E/S.
    const tail = await ranged.slice(loaded, want - loaded);
    reads++; bytesRead += tail.byteLength;
    const grown = new Uint8Array(want);
    grown.set(bytes.subarray(0, loaded));
    grown.set(new Uint8Array(tail), loaded);
    bytes = grown;
    view = new DataView(grown.buffer);
    loaded += tail.byteLength;                     // slice recorta en silencio
    if (end > loaded) bad('lectura corta: ' + loaded + ' de ' + want + ' bytes');
  }

  async function take(n) { await ensure(pos + n); const at = pos; pos += n; return at; }

  // Un u64 que no cabe en un double es, en la práctica, un fichero corrupto:
  // llama.cpp los mete en int64_t con signo y rechaza los negativos.
  function u64at(at, what) {
    const v = view.getBigUint64(at, true);
    if (v > BigInt(Number.MAX_SAFE_INTEGER)) bad(what + ' fuera de rango: ' + v);
    return Number(v);
  }

  return {
    get pos() { return pos; },
    get loaded() { return loaded; },
    get reads() { return reads; },
    get bytesRead() { return bytesRead; },
    get bytes() { return bytes; },
    get view() { return view; },
    size: ranged.size,
    ensure,
    skip(n) { pos += n; },
    async u8() { return view.getUint8(await take(1)); },
    async i8() { return view.getInt8(await take(1)); },
    async u16() { return view.getUint16(await take(2), true); },
    async i16() { return view.getInt16(await take(2), true); },
    async u32() { return view.getUint32(await take(4), true); },
    async i32() { return view.getInt32(await take(4), true); },
    async f32() { return view.getFloat32(await take(4), true); },
    async f64() { return view.getFloat64(await take(8), true); },
    async u64(what) { return u64at(await take(8), what); },
    async i64(what) {
      const at = await take(8);
      const v = view.getBigInt64(at, true);
      if (v > BigInt(Number.MAX_SAFE_INTEGER) || v < -BigInt(Number.MAX_SAFE_INTEGER)) {
        bad(what + ' fuera de rango: ' + v);
      }
      return Number(v);
    },
    // Devuelve {at, len} sin decodificar: el llamador decide si materializa la
    // string o solo la salta. Con 262k tokens de vocabulario esa diferencia es
    // la que hay entre parsear en un suspiro y ahogar al recolector.
    async str(what) {
      const len = u64at(await take(8), what + ' (longitud)');
      if (len > MAX_STRING_LENGTH) bad(what + ': string de ' + len + ' bytes');
      if (len > ranged.size - pos) bad(what + ': string más larga que el fichero');
      const at = await take(len);
      return { at, len };
    },
  };
}

function decode(bytes, at, len) { return DEC.decode(bytes.subarray(at, at + len)); }

// Array de strings PEREZOSO: al parsear solo se guardan offset y longitud de
// cada elemento; las strings se materializan cuando alguien las pide. Un vocab
// de 262k tokens serían 262k objetos String creados de golpe para nada — el
// motor casi siempre quiere los metadatos de arquitectura, no el vocabulario.
function lazyStrings(cur, offs, lens) {
  // Se consulta cur.bytes al decodificar, no al construir: la ventana crece
  // durante el parseo y capturar el buffer de entonces dejaría vivo un buffer
  // viejo además del definitivo. Los offsets valen igual en los dos, porque la
  // ventana siempre arranca en el byte 0.
  return {
    ggufType: 'string[]',
    length: offs.length,
    at(i) {
      if (!(i >= 0 && i < offs.length)) return undefined;
      return decode(cur.bytes, offs[i], lens[i]);
    },
    toArray() {
      const out = new Array(offs.length);
      for (let i = 0; i < offs.length; i++) out[i] = decode(cur.bytes, offs[i], lens[i]);
      return out;
    },
  };
}

// Copia un array de escalares a su vista tipada. Se COPIA (buf.slice) porque un
// Int32Array exige que su byteOffset sea múltiplo de 4 y en GGUF los valores no
// están alineados a nada: construirlo sobre el buffer original lanzaría.
function scalarArray(cur, type, at, n) {
  const size = V_SIZES[type];
  const raw = cur.bytes.buffer.slice(at, at + n * size);
  if (!HOST_LE) {                                  // camino lento y correcto
    const v = new DataView(raw);
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = readScalarView(v, type, i * size);
    return out;
  }
  switch (type) {
    case V_UINT8: return new Uint8Array(raw);
    case V_INT8: return new Int8Array(raw);
    case V_UINT16: return new Uint16Array(raw);
    case V_INT16: return new Int16Array(raw);
    case V_UINT32: return new Uint32Array(raw);
    case V_INT32: return new Int32Array(raw);
    case V_FLOAT32: return new Float32Array(raw);
    case V_UINT64: return new BigUint64Array(raw);
    case V_INT64: return new BigInt64Array(raw);
    case V_FLOAT64: return new Float64Array(raw);
    case V_BOOL: {
      const u = new Uint8Array(raw);
      // Los arrays de bool reales son de decenas de elementos (el patrón de
      // ventana deslizante de Gemma, p.ej.). Si alguna vez llega uno enorme, se
      // deja crudo en vez de inflar un Array de JS con un booleano por byte.
      return n <= BOOL_ARRAY_EAGER ? Array.from(u, b => b !== 0) : u;
    }
    default: return bad('tipo de array no escalar: ' + type);
  }
}

function readScalarView(v, type, at) {
  switch (type) {
    case V_UINT8: return v.getUint8(at);
    case V_INT8: return v.getInt8(at);
    case V_UINT16: return v.getUint16(at, true);
    case V_INT16: return v.getInt16(at, true);
    case V_UINT32: return v.getUint32(at, true);
    case V_INT32: return v.getInt32(at, true);
    case V_FLOAT32: return v.getFloat32(at, true);
    case V_BOOL: return v.getUint8(at) !== 0;
    case V_UINT64: return v.getBigUint64(at, true);
    case V_INT64: return v.getBigInt64(at, true);
    case V_FLOAT64: return v.getFloat64(at, true);
    default: return bad('tipo escalar desconocido: ' + type);
  }
}

async function readValue(cur, type, key) {
  switch (type) {
    case V_UINT8: return await cur.u8();
    case V_INT8: return await cur.i8();
    case V_UINT16: return await cur.u16();
    case V_INT16: return await cur.i16();
    case V_UINT32: return await cur.u32();
    case V_INT32: return await cur.i32();
    case V_FLOAT32: return await cur.f32();
    case V_FLOAT64: return await cur.f64();
    case V_UINT64: return await cur.u64('«' + key + '»');
    case V_INT64: return await cur.i64('«' + key + '»');
    case V_BOOL: return (await cur.u8()) !== 0;
    case V_STRING: { const s = await cur.str('«' + key + '»'); return decode(cur.bytes, s.at, s.len); }
    case V_ARRAY: return await readArray(cur, key);
    default: return bad('«' + key + '» tiene un tipo inválido: ' + type);
  }
}

async function readArray(cur, key) {
  const sub = await cur.i32();
  if (sub === V_ARRAY) bad('«' + key + '»: arrays anidados no permitidos');
  if (sub < 0 || sub >= V_COUNT) bad('«' + key + '»: subtipo inválido: ' + sub);
  const n = await cur.u64('«' + key + '» (nº de elementos)');
  if (n > MAX_ARRAY_ELEMENTS) bad('«' + key + '»: array de ' + n + ' elementos');

  if (sub === V_STRING) {
    // Antes de reservar nada: cada string son 8 bytes de longitud como mínimo.
    // Sin este chequeo, un n falseado nos hace pedir gigas de índice.
    if (n * 8 > cur.size - cur.pos) bad('«' + key + '»: array de strings mayor que el fichero');
    const offs = new Float64Array(n);              // los offsets pasan de 2^32
    const lens = new Uint32Array(n);               // ≤ 1 GiB por string
    for (let i = 0; i < n; i++) {
      const s = await cur.str('«' + key + '»[' + i + ']');
      offs[i] = s.at; lens[i] = s.len;
    }
    return lazyStrings(cur, offs, lens);
  }

  const size = V_SIZES[sub];
  if (n * size > cur.size - cur.pos) bad('«' + key + '»: array mayor que el fichero');
  await cur.ensure(cur.pos + n * size);
  const at = cur.pos;
  cur.skip(n * size);
  return scalarArray(cur, sub, at, n);
}

// Parsea la zona de metadatos de un GGUF desde un handle por rangos
// ({size, slice(offset,length)→ArrayBuffer}). No toca los datos de los tensores:
// devuelve dónde está cada uno para que el loader los suba al dispositivo por shards.
export async function parseGGUF(ranged, opts = {}) {
  if (!ranged || typeof ranged.slice !== 'function') bad('handle por rangos inválido');
  const cur = await cursor(ranged, opts.firstWindow || FIRST_WINDOW);
  await cur.ensure(Math.min(24, ranged.size));

  for (let i = 0; i < 4; i++) {
    if (cur.bytes[i] !== MAGIC[i]) bad('no es un fichero GGUF (magic incorrecto)');
  }
  cur.skip(4);

  const version = await cur.u32();
  // Un fichero escrito en el endianness contrario (s390x) da 0x03000000: el
  // truco es que los 16 bits bajos quedan a cero. Sin esto, el error que ve el
  // usuario es «versión 50331648» y nadie sabe qué hacer con eso.
  if (version !== 0 && version % 65536 === 0) {
    bad('fichero big-endian (versión ' + version + '): no soportado');
  }
  if (version === 0 || version > VERSION_MAX) bad('versión no soportada: ' + version);
  if (version === 1) bad('GGUFv1 no soportada: reconvierte el modelo');

  // En la CABECERA el nº de tensores va ANTES que el de claves, pero en el
  // CUERPO los pares clave/valor van ANTES que la tabla de tensores.
  const tensorCount = await cur.u64('nº de tensores');
  const kvCount = await cur.u64('nº de claves');

  const meta = Object.create(null);
  const kvTypes = Object.create(null);
  for (let i = 0; i < kvCount; i++) {
    const k = await cur.str('clave #' + i);
    if (k.len === 0) bad('clave vacía en la posición ' + i);
    const key = decode(cur.bytes, k.at, k.len);
    if (key in meta) bad('clave duplicada: «' + key + '»');
    const type = await cur.i32();
    if (type < 0 || type >= V_COUNT) bad('«' + key + '» tiene un tipo inválido: ' + type);
    kvTypes[key] = V_NAMES[type];
    meta[key] = await readValue(cur, type, key);
  }
  const kvEnd = cur.pos;

  // La alineación se resuelve aquí, entre las claves y la tabla de tensores:
  // solo afecta al relleno del bloque de datos, nunca a los metadatos.
  let alignment = DEFAULT_ALIGNMENT;
  const alignmentKey = 'general.alignment' in meta;
  if (alignmentKey) {
    if (kvTypes['general.alignment'] !== 'uint32') {
      bad('general.alignment debe ser uint32, es ' + kvTypes['general.alignment']);
    }
    alignment = meta['general.alignment'];
    // Potencia de dos vía log2 y no con `x & (x-1)`: mismo criterio que
    // llama.cpp (donde 1 es válida) sin meter operadores bitwise en un módulo
    // que trabaja con números por encima de 2^32.
    if (alignment === 0 || !Number.isInteger(Math.log2(alignment))) {
      bad('alineación no es potencia de 2: ' + alignment);
    }
  }

  const tensors = [];
  const seen = new Set();
  let expected = 0;                                // offset acumulado esperado
  for (let i = 0; i < tensorCount; i++) {
    const n = await cur.str('nombre del tensor #' + i);
    if (n.len >= MAX_NAME) bad('nombre de tensor demasiado largo (' + n.len + ' bytes)');
    const name = decode(cur.bytes, n.at, n.len);
    if (seen.has(name)) bad('nombre de tensor duplicado: «' + name + '»');
    seen.add(name);

    const nDims = await cur.u32();                 // 4 bytes, no 8
    if (nDims > MAX_DIMS) bad('«' + name + '» tiene ' + nDims + ' dimensiones (máx ' + MAX_DIMS + ')');
    // Las dimensiones que no se escriben valen 1: el escritor solo emite hasta
    // la última con ne>1, así que una norma o un bias vienen con nDims=1.
    const dims = [1, 1, 1, 1];
    for (let j = 0; j < nDims; j++) {
      const ne = await cur.i64('dimensión de «' + name + '»');
      if (ne < 0) bad('«' + name + '» tiene una dimensión negativa');
      dims[j] = ne;
    }
    const type = await cur.i32();
    const offset = await cur.u64('offset de «' + name + '»');

    const t = ggmlType(type);
    if (!t) bad('«' + name + '» usa un tipo desconocido: ' + type);
    if (t.blockSize === 0) bad('«' + name + '» usa el tipo retirado ' + t.name);
    if (dims[0] % t.blockSize !== 0) {
      bad('«' + name + '»: ne[0]=' + dims[0] + ' no es múltiplo del bloque de ' + t.name);
    }

    const nBytes = tensorBytes(dims, type);
    // El offset de cada tensor tiene que ser exactamente el acumulado de los
    // anteriores con su relleno. llama.cpp lo comprueba y rechaza el fichero si
    // no cuadra: es el mejor checksum estructural que hay, y es gratis.
    if (offset !== expected) {
      bad('«' + name + '» está en el offset ' + offset + ', se esperaba ' + expected);
    }
    expected = pad(expected + nBytes, alignment);

    tensors.push({
      name, dims: dims.slice(0, Math.max(nDims, 1)), type, typeName: t.name,
      blockSize: t.blockSize, typeSize: t.typeSize,
      nElements: dims[0] * dims[1] * dims[2] * dims[3],
      nBytes, offset,
    });
  }
  const tensorInfoEnd = cur.pos;

  // Dónde empieza el bloque de datos. llama.cpp SOLO alinea si hay al menos un
  // tensor (el seek vive dentro de `if (n_tensors > 0)`); gguf-py alinea
  // siempre. En un fichero de solo vocabulario los dos discrepan y el valor
  // alineado se sale del final del fichero, así que se devuelven los dos.
  const dataStart = tensorCount > 0 ? pad(tensorInfoEnd, alignment) : tensorInfoEnd;
  const dataStartPadded = pad(tensorInfoEnd, alignment);
  const dataBytes = expected;
  if (dataStart + dataBytes > ranged.size) {
    bad('los datos se salen del fichero: hacen falta ' + (dataStart + dataBytes) +
      ' bytes y hay ' + ranged.size);
  }

  for (const t of tensors) t.start = dataStart + t.offset;

  return {
    version, alignment, alignmentKey, tensorCount, kvCount,
    meta, kvTypes, tensors,
    kvEnd, tensorInfoEnd, dataStart, dataStartPadded, dataBytes,
    fileBytes: ranged.size,
    reads: cur.reads, bytesRead: cur.bytesRead,
    // Rango absoluto de un tensor por nombre, listo para slice() o Range HTTP.
    range(name) {
      const t = tensors.find(x => x.name === name);
      return t ? { offset: t.start, length: t.nBytes } : null;
    },
  };
}

// Atajo para el navegador: abre el modelo desde el almacén y parsea su cabecera.
// El almacén se INYECTA en vez de importarse: model-store.js vive en el repo
// público de Elffuss (web/js/runtime/), no aquí, y duplicar el fichero solo
// garantiza que las dos copias se separen. Quien llame pasa el suyo.
export async function openGGUF(url, store) {
  if (!store?.openRanged) throw new Error('openGGUF necesita un almacén con openRanged');
  const ranged = await store.openRanged(url);
  if (!ranged) return null;                        // sin OPFS: decide el llamador
  const parsed = await parseGGUF(ranged);
  return { ...parsed, ranged };
}
