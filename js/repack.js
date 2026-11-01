// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · reordenar Q8_0 y Q4_0 al cargar (`__ELFFUSS_REPACK`).
// ─────────────────────────────────────────────────────────────────────────────
// El formato está en `kernels.bloques.wgsl.js`: los cuantos de todos los bloques
// primero, alineados a 16 bytes, y las escalas después (tipos internos Q8_0R y
// Q4_0R). Aquí está quien lo FABRICA a partir de los bytes del GGUF, y lo que
// necesita el cargador para subirlo.
//
// ── POR QUÉ EN LA CPU, MIENTRAS LLEGAN LOS BYTES, Y NO EN EL DISPOSITIVO ────
// La alternativa era subir los bytes tal cual a un buffer temporal y que un
// kernel los reordenara en el definitivo. Se descarta por dos motivos, y la
// memoria NO es uno de ellos:
//   · SE PUEDE COMPROBAR SIN DISPOSITIVO. Es una permutación de bytes, y la
//     MISMA función que corre el cargador se prueba bit a bit en node: vectores
//     conocidos, ida y vuelta, bloques impares, relleno, y trocear igual que
//     entero. Un kernel de reordenado solo se podría probar con dispositivo.
//   · NI COMPILAR NI DESPACHAR DURANTE LA CARGA. El reordenado en la CPU es una
//     pasada lineal sobre bytes que ya están en memoria, con lecturas y
//     escrituras de 32 bits y desenrollada por parejas de bloques.
// Lo que cuesta, por camino:
//   · TRAMOS (casi todos los tensores): ya se hacía una copia de cada tensor
//     para rellenarla a 4 bytes (`pad`), y ahora se reordena DENTRO de esa copia
//     en vez de copiar tal cual. Ni un byte más de memoria; la pasada es varias
//     veces más cara que la copia que sustituye.
//   · TROZOS (un tensor mayor que un tramo: la tabla de embeddings): antes cada
//     trozo leído se escribía TAL CUAL, sin copia. Ahora se reordena en dos
//     búferes que se reutilizan entre trozos, así que hay una pasada que antes
//     no había y un trozo más de memoria en la CPU.
// Sobre la memoria, para que nadie lo lea al revés: un reordenado en el
// dispositivo de un tensor ENTERO sí lo tendría dos veces a la vez, pero uno
// por trozos (un buffer temporal del tamaño de un trozo y un kernel que reparta
// cada trozo en sus dos regiones) costaría también un trozo, como aquí. No es lo
// que decide. Lo que cueste de verdad en la carga entera se mide con y sin el
// interruptor (la parte tmodelo del banco del reordenado, que recarga en cada
// ronda) antes de encenderlo.
//
// ── EL INTERRUPTOR SE LEE AL CARGAR, NO AL MULTIPLICAR ─────────────────────
// `__ELFFUSS_REPACK` decide qué formato tiene cada tensor en el momento de
// subirlo, y ese formato queda escrito en su descriptor (`typeName` = 'Q8_0R' o
// 'Q4_0R'). Los kernels eligen por el TIPO, así que un modelo cargado con el
// interruptor puesto sigue siendo coherente aunque luego se quite, y al revés:
// formato y lector no pueden desacoplarse. Y como el tipo va en todas las claves
// de pipeline (`matmul_${type}…`, `drift_deq_${type}`, `gather_${type}`…), los
// shaders de un formato y del otro nunca comparten compilado.
//
// Por lo mismo, un A/B que cargue el modelo UNA vez y conmute el interruptor
// entre rondas mide el mismo formato en las dos y da ×1,00: hay que cargar de
// nuevo en cada ronda (`INTERRUPTORES_DE_CARGA`, abajo, es la lista que miran
// los bancos para negarse).
//
// Apagado, nada de esto se ejecuta: mismos buffers, mismos bytes, mismos
// descriptores que antes.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.
import { TIPO_R, TIPOS_R, GEOMETRIA, bytesReordenado, bytesBufferR } from './kernels.bloques.wgsl.js';

// Un solo sitio que lee el interruptor.
export function reordenActivo() {
  return globalThis.__ELFFUSS_REPACK === true;
}

// Interruptores que actúan AL CARGAR el modelo y no al multiplicar.
export const INTERRUPTORES_DE_CARGA = Object.freeze(['__ELFFUSS_REPACK']);

// El tipo reordenado de un tensor del GGUF, o null si no se reordena
// (interruptor apagado, o un tipo que no es Q8_0 ni Q4_0). `activo` se pasa
// para que una carga entera use la foto que tomó al empezar.
export function tipoReordenado(t, activo = reordenActivo()) {
  return activo ? (TIPO_R[t.typeName] || null) : null;
}

// El descriptor que viaja con el buffer. Es una COPIA del de parseGGUF con el
// tipo interno: la tabla de tensores del fichero no se toca (otra carga del
// mismo `g` tiene que ver los tipos del fichero), y quien elige kernel por
// `w.t.typeName` —el forward, el gather, las pruebas— recibe el que corresponde
// a los bytes que hay en el dispositivo.
export function comoReordenado(t, tipoR) {
  return { ...t, typeName: tipoR, typeNameGGUF: t.typeName };
}

// Bloques de un tensor Q8_0/Q4_0 (32 pesos por bloque en los dos).
export function bloquesDe(t) {
  return t.nElements / GEOMETRIA[t.typeName].elems;
}

const LE = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
const alineado = (a) => a.byteOffset % 4 === 0;

// ── El camino rápido, por PAREJAS de bloques ───────────────────────────────
// Dos bloques Q8_0 son 68 bytes = 17 palabras enteras (Q4_0: 36 = 9). Dentro de
// la pareja:
//   · el bloque PAR tiene la escala en la media palabra baja de w0 y los
//     cuantos desde el byte 2, así que cada palabra de cuantos es la mitad alta
//     de una palabra y la baja de la siguiente;
//   · el bloque IMPAR tiene la escala en la mitad alta de la palabra 8 (Q4: 4),
//     la misma que cierra los cuantos del par, y sus cuantos ya alineados: se
//     copian tal cual;
//   · las dos escalas van juntas en una palabra de la región de escalas: la del
//     par en la mitad baja (media palabra par) y la del impar en la alta.
// Desenrollado a mano y uno por tipo: con el bucle interno de 8 (o 4) vueltas
// la pasada costaba varias veces más, y es la que paga la carga.
function pares8(s32, q32, e32, pares) {
  for (let p = 0, s = 0, q = 0; p < pares; p++, s += 17, q += 16) {
    const w0 = s32[s], w1 = s32[s + 1], w2 = s32[s + 2], w3 = s32[s + 3], w4 = s32[s + 4],
      w5 = s32[s + 5], w6 = s32[s + 6], w7 = s32[s + 7], w8 = s32[s + 8];
    q32[q] = (w0 >>> 16) | (w1 << 16);
    q32[q + 1] = (w1 >>> 16) | (w2 << 16);
    q32[q + 2] = (w2 >>> 16) | (w3 << 16);
    q32[q + 3] = (w3 >>> 16) | (w4 << 16);
    q32[q + 4] = (w4 >>> 16) | (w5 << 16);
    q32[q + 5] = (w5 >>> 16) | (w6 << 16);
    q32[q + 6] = (w6 >>> 16) | (w7 << 16);
    q32[q + 7] = (w7 >>> 16) | (w8 << 16);
    q32[q + 8] = s32[s + 9];
    q32[q + 9] = s32[s + 10];
    q32[q + 10] = s32[s + 11];
    q32[q + 11] = s32[s + 12];
    q32[q + 12] = s32[s + 13];
    q32[q + 13] = s32[s + 14];
    q32[q + 14] = s32[s + 15];
    q32[q + 15] = s32[s + 16];
    e32[p] = (w0 & 0xFFFF) | (w8 & 0xFFFF0000);
  }
}
function pares4(s32, q32, e32, pares) {
  for (let p = 0, s = 0, q = 0; p < pares; p++, s += 9, q += 8) {
    const w0 = s32[s], w1 = s32[s + 1], w2 = s32[s + 2], w3 = s32[s + 3], w4 = s32[s + 4];
    q32[q] = (w0 >>> 16) | (w1 << 16);
    q32[q + 1] = (w1 >>> 16) | (w2 << 16);
    q32[q + 2] = (w2 >>> 16) | (w3 << 16);
    q32[q + 3] = (w3 >>> 16) | (w4 << 16);
    q32[q + 4] = s32[s + 5];
    q32[q + 5] = s32[s + 6];
    q32[q + 6] = s32[s + 7];
    q32[q + 7] = s32[s + 8];
    e32[p] = (w0 & 0xFFFF) | (w4 & 0xFFFF0000);
  }
}

/**
 * Reordena `nb` bloques consecutivos en formato GGUF (`src`, Uint8Array) a dos
 * destinos: `cuantos` (bytes de cuantos, 32·nb o 16·nb) y `escalas` (2·nb
 * bytes, la escala f16 del bloque k en los bytes 2k y 2k+1). Son las dos
 * regiones del formato: el trozo [i0, i0+nb) de un tensor va a la posición
 * cuantos·i0 de la primera y 2·i0 de la segunda.
 *
 * El camino rápido (arriba) exige un host little-endian —los bytes del GGUF lo
 * son, y las palabras se recomponen con desplazamientos— y las tres vistas
 * alineadas a 4. Si no se cumple, o para el bloque suelto de un número impar,
 * se copia byte a byte, que da exactamente los mismos bytes.
 */
export function reordenaBloques(tipo, src, nb, cuantos, escalas) {
  const g = GEOMETRIA[tipo];
  if (!g) throw new Error(`reordena: tipo sin formato reordenado: «${tipo}»`);
  const B = g.bytes, Q = g.cuantos, P = g.palabras;
  if (!(nb >= 0) || nb % 1 !== 0) throw new Error(`reordena: número de bloques inválido: ${nb}`);
  if (src.byteLength < B * nb) throw new Error(`reordena: faltan bytes de origen (${src.byteLength} < ${B * nb})`);
  if (cuantos.byteLength < Q * nb) throw new Error(`reordena: destino de cuantos corto (${cuantos.byteLength} < ${Q * nb})`);
  if (escalas.byteLength < 2 * nb) throw new Error(`reordena: destino de escalas corto (${escalas.byteLength} < ${2 * nb})`);

  let hechos = 0;
  const pares = nb >> 1;
  if (LE && pares > 0 && alineado(src) && alineado(cuantos) && alineado(escalas)) {
    const s32 = new Uint32Array(src.buffer, src.byteOffset, pares * (B >> 1));
    const q32 = new Uint32Array(cuantos.buffer, cuantos.byteOffset, pares * 2 * P);
    const e32 = new Uint32Array(escalas.buffer, escalas.byteOffset, pares);
    (P === 8 ? pares8 : pares4)(s32, q32, e32, pares);
    hechos = 2 * pares;
  }
  // Byte a byte: lo que no fue por parejas.
  for (let i = hechos; i < nb; i++) {
    const s = i * B;
    escalas[2 * i] = src[s];
    escalas[2 * i + 1] = src[s + 1];
    const q = i * Q;
    for (let j = 0; j < Q; j++) cuantos[q + j] = src[s + 2 + j];
  }
}

/**
 * Reordena un tensor ENTERO de `nBloques` bloques. `dst` (opcional) es donde se
 * escribe —el cargador pasa su copia de relleno— y tiene que medir al menos
 * `bytesReordenado`; lo que sobre detrás de las escalas se deja a cero.
 * Devuelve `dst`. Por defecto mide `bytesBufferR`, lo mismo que el buffer.
 */
export function reordena(tipo, src, nBloques, dst = null) {
  const g = GEOMETRIA[tipo];
  if (!g) throw new Error(`reordena: tipo sin formato reordenado: «${tipo}»`);
  const out = dst || new Uint8Array(bytesBufferR(tipo, nBloques));
  const total = bytesReordenado(tipo, nBloques);
  if (out.byteLength < total) {
    throw new Error(`reordena: el destino mide ${out.byteLength} y hacen falta ${total}`);
  }
  const W = g.cuantos * nBloques;
  reordenaBloques(tipo, src, nBloques, out.subarray(0, W), out.subarray(W, W + 2 * nBloques));
  out.fill(0, W + 2 * nBloques);
  return out;
}

// La inversa: del formato reordenado al del GGUF (34 o 18 bytes por bloque).
// No la usa la carga; existe para la ida y vuelta de las pruebas y para quien
// necesite los bytes originales de un tensor ya reordenado.
export function vuelveAlGGUF(tipo, src, nBloques) {
  const g = GEOMETRIA[tipo];
  if (!g) throw new Error(`vuelveAlGGUF: tipo sin formato reordenado: «${tipo}»`);
  const B = g.bytes, Q = g.cuantos, W = Q * nBloques;
  if (src.byteLength < W + 2 * nBloques) {
    throw new Error(`vuelveAlGGUF: faltan bytes (${src.byteLength} < ${W + 2 * nBloques})`);
  }
  const out = new Uint8Array(B * nBloques);
  for (let i = 0; i < nBloques; i++) {
    const d = i * B;
    out[d] = src[W + 2 * i];
    out[d + 1] = src[W + 2 * i + 1];
    out.set(src.subarray(Q * i, Q * i + Q), d + 2);
  }
  return out;
}

// ── LA MARCA: cuántos bloques tiene DE VERDAD cada buffer reordenado ────────
// Los lectores R deducen N —y con él dónde empiezan las escalas— del tamaño del
// buffer atado (kernels.bloques.wgsl.js). Con un buffer de otro tamaño no fallan:
// leen las escalas de otro sitio y sale basura plausible. Y es fácil tener uno:
// `upload` y `ctx.alloc` de ops.js redondean a 256 bytes (un Q8_0R de 64
// bloques necesita 2.176 y el pozo da 2.304, que se lee como 67 bloques), y un
// troceado o un binding con offset harían lo mismo. Ni siquiera basta con mirar
// el tamaño: un Q4_0R de 34 bloques en 768 bytes se lee como 42, y 768 es
// también el buffer correcto de 42. Hace falta saber el N del tensor, y eso solo
// lo sabe quien crea el buffer: lo apunta aquí, y ops.js lo comprueba antes de
// cada despacho de un lector R (`compruebaBufferR`). Es aritmética de JS por
// despacho, sin tocar el dispositivo.
//
// WeakMap y no una propiedad pegada al GPUBuffer: no toca un objeto de la
// plataforma, y la marca se va con el buffer cuando se libera.
const MARCAS = new WeakMap();

// Apunta que `buf` guarda un tensor `tipoR` de `nBloques` bloques. El buffer
// tiene que medir exactamente `bytesBufferR` (múltiplo de 16, que también es lo
// que pide la vista vec4 de los lectores).
export function marcaBufferR(buf, tipoR, nBloques) {
  if (!TIPOS_R.has(tipoR)) throw new Error(`marcaBufferR: «${tipoR}» no es un tipo reordenado`);
  if (!(nBloques > 0) || nBloques % 1 !== 0) throw new Error(`marcaBufferR: número de bloques inválido: ${nBloques}`);
  const esperado = bytesBufferR(tipoR, nBloques);
  if (buf.size !== esperado) {
    throw new Error(`marcaBufferR: «${buf.label || '?'}» mide ${buf.size} bytes y un ${tipoR} de ` +
      `${nBloques} bloques tiene que medir ${esperado}: los lectores sacan del tamaño dónde empiezan las escalas`);
  }
  MARCAS.set(buf, { tipo: tipoR, bloques: nBloques });
  return buf;
}

// El N del TENSOR que guarda un buffer reordenado (undefined si no lleva marca).
// Es el que necesita quien pase W por uniformes en vez de deducirlo del tamaño:
// el `rows·bpr` de una llamada no vale (la cabeza se pide con el vocabulario).
export function bloquesR(buf) {
  const m = MARCAS.get(buf);
  return m ? m.bloques : undefined;
}

/**
 * Antes de despachar un lector R sobre `buf`: que lleve la marca, que sea del
 * tipo que se va a leer y que la llamada no se salga del tensor. `bloques` son
 * los que lee la llamada (matmul: filas·bpr; dequant: nBlocks) y `porFila`, en
 * el gather, los bloques de una fila (la tabla tiene que ser de filas enteras).
 * Lanza con el porqué; devuelve el N del tensor.
 */
export function compruebaBufferR(tipo, buf, { bloques = 0, porFila = 0 } = {}, que = 'lector') {
  const m = MARCAS.get(buf);
  if (!m) {
    throw new Error(`${que} ${tipo}: el buffer «${buf.label || '?'}» no lleva la marca de un tensor reordenado. ` +
      'Los lectores R sacan del tamaño del buffer dónde empiezan las escalas, así que tiene que crearlo ' +
      'creaBufferR (o marcarlo marcaBufferR, con el tamaño de bytesBufferR); `upload` redondea a 256 ' +
      'bytes y con otro tamaño se leerían escalas de otro sitio sin un error');
  }
  if (m.tipo !== tipo) {
    throw new Error(`${que}: el buffer «${buf.label || '?'}» guarda un ${m.tipo} y se iba a leer como ${tipo}`);
  }
  if (bloques > m.bloques) {
    throw new Error(`${que} ${tipo}: la llamada lee ${bloques} bloques y el tensor de «${buf.label || '?'}» tiene ${m.bloques}`);
  }
  if (porFila > 0 && m.bloques % porFila !== 0) {
    throw new Error(`${que} ${tipo}: el tensor de «${buf.label || '?'}» (${m.bloques} bloques) no son filas enteras de ${porFila} bloques`);
  }
  return m.bloques;
}

// El buffer de un tensor reordenado: el formato redondeado a 16 bytes (ver
// `bytesBufferR`), con su marca puesta. No lleva la holgura de 4 bytes de los
// otros: los lectores de este formato leen palabras alineadas y nunca la palabra
// siguiente al tensor. Pasarse del tope de binding no lanzaría aquí sino al
// atarlo, y en WebGPU eso es un despacho que no hace nada: se comprueba antes.
export function creaBufferR(device, t, tipoR) {
  const nBloques = bloquesDe(t);
  const size = bytesBufferR(tipoR, nBloques);
  const tope = device.limits.maxStorageBufferBindingSize;
  if (size > tope) {
    throw new Error(`«${t.name}» reordenado mide ${size} bytes y el límite de binding es ${tope}`);
  }
  const buf = device.createBuffer({
    label: t.name,
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  return marcaBufferR(buf, tipoR, nBloques);
}

/**
 * Sube un tensor demasiado grande para un tramo, por trozos y reordenado. Es el
 * camino de trozos de siempre con dos cambios: cada trozo son bloques ENTEROS y
 * en número PAR (para que sus escalas empiecen en palabra: writeBuffer exige
 * offset y tamaño múltiplos de 4), y cada trozo se escribe en DOS sitios del
 * buffer, su tramo de cuantos y su tramo de escalas.
 *
 * `cuenta(msLeer, msSubir, bytes)` se llama por trozo, para que el cargador
 * siga repartiendo el tiempo y avisando del progreso igual que antes.
 */
export async function subeReordenadoPorTrozos({ device, ranged, t, buf, tipoR, trozo, ahora, cuenta }) {
  const g = GEOMETRIA[tipoR];
  const B = g.bytes, Q = g.cuantos;
  const N = bloquesDe(t);
  const W = Q * N;                                   // byte donde empiezan las escalas
  const porTrozo = Math.max(2, Math.floor(trozo / (2 * B)) * 2);
  // Los dos destinos se reutilizan entre trozos: writeBuffer copia al llamarse.
  const cuantos = new Uint8Array(Q * Math.min(porTrozo, N));
  const escalas = new Uint8Array(Math.ceil(2 * Math.min(porTrozo, N) / 4) * 4);
  for (let i0 = 0; i0 < N; i0 += porTrozo) {
    const nb = Math.min(porTrozo, N - i0);
    const t0 = ahora();
    const raw = new Uint8Array(await ranged.slice(t.start + i0 * B, nb * B));
    const t1 = ahora();
    // Un trozo corto entraría como PESOS sin un solo error.
    if (raw.byteLength !== nb * B) {
      throw new Error(`trozo incompleto en «${t.name}»: pedidos ${nb * B} bytes y llegaron ${raw.byteLength}`);
    }
    const nE = Math.ceil(2 * nb / 4) * 4;
    escalas.fill(0, 0, nE);
    reordenaBloques(tipoR, raw, nb, cuantos, escalas.subarray(0, 2 * nb));
    device.queue.writeBuffer(buf, Q * i0, cuantos, 0, Q * nb);
    // Con nb impar (solo el último trozo de un N impar) se escriben dos bytes de
    // relleno detrás de la última escala: caben, el buffer va redondeado a 16.
    device.queue.writeBuffer(buf, W + 2 * i0, escalas, 0, nE);
    cuenta(t1 - t0, ahora() - t1, nb * B);
  }
}
