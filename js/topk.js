// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · muestreo sobre CANDIDATOS: lo que `sample()` necesita ver.
// ─────────────────────────────────────────────────────────────────────────────
// En cada token se traía el vocabulario ENTERO a la CPU para que `sample()` se
// quedara con unas decenas de logits. Esto decide CUÁLES bastan, para que el
// dispositivo (`ops.topk.js`) devuelva solo esos, y luego llama al MISMO
// `sample()` de siempre sobre ese puñado. Exacto, no aproximado: mismo token, bit a bit, que
// el camino de siempre. El porqué, en tres pasos:
//
//   1. Lo que mira `sample()`. Primero penaliza los ids de la ventana reciente
//      (S, como mucho `repeatLastN` ids distintos) y después solo usa los K'
//      primeros en el orden «logit mayor primero; a igualdad, índice menor»:
//      K' = 1 en voraz (el argmax con `>` estricto se queda con el primero de
//      los empatados) y K' = topK con temperatura. Top-p, min-p y la
//      temperatura solo leen esos K' valores. Nada más del vector influye.
//   2. Por qué basta el top-(K'+|S|) CRUDO más S. La penalización solo cambia
//      los logits de S —puede bajar uno de arriba (divide los positivos y
//      multiplica los negativos) y dejar entrar a otro—, y fuera de S el
//      orden entre dos ids no cambia. Sea j uno de los K' primeros DESPUÉS de
//      penalizar, con j fuera de S: tiene delante, como mucho, K'−1 ids de
//      fuera de S —que en crudo siguen delante de j— más los |S| de S. O sea
//      que en crudo va como mucho en el puesto K'+|S|, y está en el top-(K'+|S|).
//      Los de S van aparte, con su valor crudo, y la penalización se la pone el
//      propio `sample()`.
//   3. Por qué `sample()` sobre el puñado da lo mismo que sobre el vector. Los
//      candidatos se ordenan por id ASCENDENTE: así «índice menor» significa lo
//      mismo en el puñado y en el vocabulario, y los empates se deshacen igual.
//      Los K' primeros de un superconjunto de los K' primeros son esos mismos,
//      en el mismo orden y con los mismos valores f32 (los bits los copia el
//      dispositivo tal cual). Las penalizaciones se aplican con la misma aritmética
//      sobre un Float32Array, igual que antes. Mismo orden de operaciones,
//      mismos números: mismo token.
//
// Lo que NO se hace aquí y por qué se cae al camino de siempre (`null`):
//   · temperatura > 0 sin top-k: `sample()` se queda con todo lo que está a
//     menos de 40 del máximo, y eso no tiene tope;
//   · un id de la ventana que no es entero (no pasa con fichas de verdad, pero
//     `sample()` haría con él cosas que un puñado no reproduce);
//   · más candidatos de los que caben (top-k grande o ventana larga);
//   · NaN en los logits: el orden con NaN depende de la posición en el vector y
//     no se puede reproducir con un trozo (lo detecta el kernel y lo avisa);
//   · una penalización que puede FABRICAR un NaN: factor NaN, ±∞ o 0 (0·∞,
//     −∞·0) o presencia ±∞ (∞−∞). Con factor finito y no nulo y presencia
//     finita, un logit que no es NaN no se vuelve NaN, y el paso 2 vale.
//
// Módulo PURO: sin WebGPU, para poder probarlo en Node contra `sample()`.
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.
import { sample } from './sample.js';
// Topes del contrato con los kernels: sitio para TOPK_CAP candidatos y
// TOPK_P_MAX ids penalizados. Los kernels devuelven EXACTAMENTE los M primeros
// (los empates se deshacen por índice), así que basta con M ≤ TOPK_M_MAX.
import { TOPK_M_MAX, TOPK_P_MAX, TOPK_CAP, TOPK_CABECERA } from './kernels.topk.wgsl.js';

// Qué candidatos hacen falta para este paso, o null si este muestreo no se
// puede hacer con un puñado. `previosA` y `previosB` son las dos mitades de la
// ventana (prompt y lo ya emitido): se recorren como si estuvieran concatenadas,
// sin concatenarlas, que era un coste lineal en el contexto en cada token.
//
// Los valores por defecto son LOS DE `sample()`: si cambian allí tienen que
// cambiar aquí, o el puñado se elegiría con otra ventana u otro K'.
export function planTopK(vocab, opts = {}, previosA = null, previosB = null) {
  const { temperature = 0, topK = 0, repeatPenalty = 1.1, repeatLastN = 64, presence = 0 } = opts;
  if (!(vocab >= 1)) return null;
  let K;
  if (temperature <= 0) K = 1;
  else if (topK > 0 && Number.isInteger(topK)) K = Math.min(topK, vocab);
  else return null;

  const a = previosA || [], b = previosB || [];
  const largo = a.length + b.length;
  // La MISMA condición que `sample()`, con el mismo `previos.length`.
  const penaliza = !!(largo && (repeatPenalty !== 1 || presence));
  const ventana = [];
  const S = [];
  if (penaliza) {
    // El MISMO bucle que `sample()`: mismo `desde` (que puede no ser entero si
    // `repeatLastN` no lo es) y mismos índices.
    const desde = Math.max(0, largo - repeatLastN);
    const vistos = new Set();
    for (let i = desde; i < largo; i++) {
      const id = i < a.length ? a[i] : b[i - a.length];
      if (!Number.isInteger(id)) return null;
      ventana.push(id);
      if (id >= 0 && id < vocab && !vistos.has(id)) { vistos.add(id); S.push(id); }
    }
  }
  if (S.length > TOPK_P_MAX) return null;
  // Las MISMAS dos condiciones con las que `sample()` aplica cada penalización.
  if (S.length && ((repeatPenalty !== 1 && !(Number.isFinite(repeatPenalty) && repeatPenalty !== 0)) ||
      (presence && !Number.isFinite(presence)))) return null;
  const M = K + S.length;
  if (M > TOPK_M_MAX) return null;
  return { vocab, K, M, S: Uint32Array.from(S), penaliza, ventana };
}

// Lo que dejó el kernel final (`u`, u32 con la disposición de
// `kernels.topk.wgsl.js`), o { motivo } si hay que leer el vector entero.
// `serie` es el número de la llamada: si la cabecera trae otro, lo leído es de
// una llamada anterior (el envío no llegó a ejecutarse) y no vale.
export function leeSalidaTopK(u, plan, trozos, serie = undefined) {
  if (serie !== undefined && u[4] !== serie) return { motivo: 'serie' };
  const n = u[0], avisos = u[1];
  if (avisos !== 0) return { motivo: avisos & 2 ? 'NaN' : 'no cupo' };
  if (u[2] !== plan.S.length || u[3] !== trozos || n > TOPK_CAP) return { motivo: 'cabecera' };
  const ini = TOPK_CABECERA + TOPK_P_MAX;
  const ids = new Uint32Array(n), bits = new Uint32Array(n);
  for (let q = 0; q < n; q++) { ids[q] = u[ini + 2 * q]; bits[q] = u[ini + 2 * q + 1]; }
  return { ids, bits, bitsS: u.slice(TOPK_CABECERA, TOPK_CABECERA + plan.S.length) };
}

// El token, a partir de lo que devolvieron los kernels. `ids`/`bits` son los
// candidatos (bits f32 crudos) y `bitsS` los logits crudos de `plan.S`, en su
// orden. Devuelve null si lo recibido no es coherente (y entonces se lee el
// vector entero): mejor un token lento que uno inventado.
export function muestraCandidatos(plan, ids, bits, bitsS, opts = {}) {
  const n = ids.length;
  if (bits.length !== n || bitsS.length !== plan.S.length) return null;
  if (n < Math.min(plan.M, plan.vocab)) return null;
  const porId = new Map();
  for (let q = 0; q < n; q++) {
    const id = ids[q];
    // Un id fuera del vocabulario o repetido no sale de unos kernels sanos.
    if (id >= plan.vocab || porId.has(id)) return null;
    porId.set(id, bits[q]);
  }
  for (let i = 0; i < plan.S.length; i++) {
    const id = plan.S[i], prev = porId.get(id);
    if (prev !== undefined && prev !== bitsS[i]) return null;
    porId.set(id, bitsS[i]);
  }
  // Por id ascendente: el desempate por índice tiene que significar lo mismo
  // aquí que en el vocabulario entero.
  const orden = Uint32Array.from(porId.keys()).sort();
  const sub = new Float32Array(orden.length);
  const subBits = new Uint32Array(sub.buffer);
  const pos = new Map();
  for (let j = 0; j < orden.length; j++) {
    subBits[j] = porId.get(orden[j]);
    pos.set(orden[j], j);
  }
  // La ventana con las posiciones del puñado, en el MISMO orden. Los ids fuera
  // del vocabulario van a −1, que `sample()` se salta igual que se los saltaba.
  // Su longitud no pasa de `repeatLastN`, así que `sample()` la recorre entera.
  let previos = null;
  if (plan.penaliza) {
    previos = new Array(plan.ventana.length);
    for (let i = 0; i < plan.ventana.length; i++) {
      const p = pos.get(plan.ventana[i]);
      previos[i] = p === undefined ? -1 : p;
    }
  }
  const j = sample(sub, { ...opts, previos });
  return orden[j];
}
