// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · muestreo y bucle de generación.
// ─────────────────────────────────────────────────────────────────────────────
// El bucle de decode es donde el motor pasa el 99 % del tiempo, y donde falla lo
// que un forward suelto no puede detectar: que el estado se arrastre bien entre
// llamadas. Un forward aislado empieza con el estado a cero y la caché vacía, así
// que valida la matemática pero no la memoria.
//
// La prueba que sí lo detecta es de consistencia interna: procesar «abc» de una
// vez tiene que dar el MISMO resultado que procesar «a», luego «b», luego «c».
// Si la caché KV se escribe en el sitio equivocado, si el estado recurrente no
// se arrastra o si las posiciones de RoPE se reinician, las dos formas divergen.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.

// Muestreo sobre los logits ya en CPU. Se trabaja sobre el último token, que es
// el único que interesa al generar. Sigue el ORDEN de llama.cpp, que es la
// referencia con la que se mide todo: penalizaciones → top-k → top-p → min-p →
// temperatura. La temperatura va AL FINAL: top-p y min-p se calculan sobre la
// distribución sin temperatura y la temperatura solo reparte entre los que
// sobreviven. Antes iba al principio, y con temperatura < 1 (la ficha del 27B
// pide 0,7 con top-p 0,8) el top-p cortaba sobre una distribución más
// afilada: sobrevivían menos tokens que en llama.cpp, sin error ninguno.
//   · temp = 0 → greedy, determinista (tras las penalizaciones).
//   · repeatPenalty: la de llama.cpp (divide si el logit es positivo y
//     multiplica si es negativo). Su defecto aquí sigue siendo 1,1 para no
//     cambiar a los modelos que no declaran muestreo; el llama.cpp actual trae
//     1,0, y los modelos con ficha la apagan desde el registro.
//   · presence: resta fija a todo token que aparezca en la ventana (la ficha
//     del 27B pide 1,5 sin razonar), como `presence_penalty` de llama.cpp.
//   · minP: descarta los tokens cuya probabilidad sea menor que minP × la del
//     más probable.
// mulberry32 sobre la semilla: buena mezcla también con semillas consecutivas
// (seed + paso), que es como se llama token a token.
function uniforme(seed) {
  let t = ((seed >>> 0) + 0x6D2B79F5) >>> 0;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

// ¿Va `a` antes que `b`? Logit mayor primero; a igualdad, índice menor.
const antes = (logits, a, b) => logits[a] > logits[b] || (logits[a] === logits[b] && a < b);

// Los K mejores índices, ordenados. Montículo de MÍNIMOS de tamaño K (la raíz es
// el peor de los que se guardan): cada logit nuevo solo entra si gana a la raíz.
export function topKIndices(logits, k) {
  const h = new Int32Array(k);
  let size = 0;
  const peor = (a, b) => antes(logits, b, a);          // a es «peor» que b
  const sube = (i) => { while (i > 0) { const p = (i - 1) >> 1; if (!peor(h[i], h[p])) break; [h[i], h[p]] = [h[p], h[i]]; i = p; } };
  const baja = (i) => {
    for (;;) {
      const l = 2 * i + 1, r = l + 1; let m = i;
      if (l < size && peor(h[l], h[m])) m = l;
      if (r < size && peor(h[r], h[m])) m = r;
      if (m === i) return;
      [h[i], h[m]] = [h[m], h[i]]; i = m;
    }
  };
  for (let i = 0; i < logits.length; i++) {
    if (size < k) { h[size] = i; sube(size++); }
    else if (antes(logits, i, h[0])) { h[0] = i; baja(0); }
  }
  return Array.from(h.subarray(0, size)).sort((a, b) => (antes(logits, a, b) ? -1 : 1));
}

// Sin top-k: los índices a menos de `margen` del máximo, ordenados.
function candidatos(logits, margen) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  const out = [];
  for (let i = 0; i < logits.length; i++) if (logits[i] >= max - margen) out.push(i);
  return out.sort((a, b) => (antes(logits, a, b) ? -1 : 1));
}

export function sample(logits, opts = {}) {
  const { temperature = 0, topP = 1, topK = 0, minP = 0, seed = 1,
          repeatPenalty = 1.1, repeatLastN = 64, presence = 0, previos = null } = opts;
  const n = logits.length;

  // PENALIZACIONES, sobre los últimos `repeatLastN` (prompt incluido, como
  // llama.cpp). La de repetición existe porque el decode voraz sobre un modelo
  // muy cuantizado entra en bucle («el color es de color, el color es de…»).
  if (previos && previos.length && (repeatPenalty !== 1 || presence)) {
    const desde = Math.max(0, previos.length - repeatLastN);
    const vistos = new Set();
    for (let i = desde; i < previos.length; i++) vistos.add(previos[i]);
    for (const id of vistos) {
      if (id < 0 || id >= n) continue;
      if (repeatPenalty !== 1) logits[id] = logits[id] > 0 ? logits[id] / repeatPenalty : logits[id] * repeatPenalty;
      if (presence) logits[id] -= presence;
    }
  }
  if (temperature <= 0) {
    let best = 0;
    for (let i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }

  // Los candidatos ordenados por logit (desc; a igual logit, el índice menor
  // primero, que es lo que daba la ordenación estable de antes). NO se ordena el
  // vocabulario entero: eso eran decenas de milisegundos POR TOKEN en JS, más
  // que el forward de un modelo pequeño. Con top-k se seleccionan los K con un
  // montículo en una pasada; sin top-k, solo los que quedan a menos de 40 nats
  // del máximo (fuera de ahí la probabilidad no llega a 1e-17 cada uno).
  const idx = topK > 0 ? topKIndices(logits, Math.min(topK, n)) : candidatos(logits, 40);
  const keep = idx.length;

  // top-p y min-p SIN temperatura. Resta del máximo antes de exp(): sin ella,
  // logits de 30 desbordan.
  const max = logits[idx[0]];
  const base = new Float64Array(keep);
  let total = 0;
  for (let i = 0; i < keep; i++) { base[i] = Math.exp(logits[idx[i]] - max); total += base[i]; }
  let corte = keep;
  if (topP < 1) {
    let acc = 0;
    for (let i = 0; i < keep; i++) {
      acc += base[i] / total;
      if (acc >= topP) { corte = i + 1; break; }
    }
  }
  if (minP > 0) {
    // base[0] es el más probable (están ordenados): el umbral es relativo a él.
    const umbral = minP * base[0];
    let k = 1;
    while (k < corte && base[k] >= umbral) k++;
    corte = k;
  }

  // La temperatura, solo entre los supervivientes.
  const probs = new Float64Array(corte);
  let sub = 0;
  for (let i = 0; i < corte; i++) { probs[i] = Math.exp((logits[idx[i]] - max) / temperature); sub += probs[i]; }

  // Generador propio y con semilla: Math.random() haría irreproducible
  // cualquier prueba de regresión de generación. Uniforme en [0, 1).
  //
  // ⚠️ Antes era un paso de LCG con `(s >>> 8) / 8388608`: el numerador llega a
  // 2²⁴ y el divisor es 2²³, así que el «uniforme» caía en [0, 2). La mitad de
  // las veces r > sub, el bucle de abajo se salía sin elegir y devolvía el
  // ÚLTIMO superviviente: el menos probable (sin top-k, el peor token de todo el
  // vocabulario). Y semillas consecutivas daban valores correlados. Lo cazó
  // una prueba al pedir que un token improbable saliera alguna vez.
  let r = uniforme(seed) * sub;
  for (let i = 0; i < corte; i++) {
    r -= probs[i];
    if (r <= 0) return idx[i];
  }
  return idx[corte - 1];
}

// Genera tokens uno a uno. El prompt entero va en UNA llamada (prefill) y cada
// token nuevo en la suya (decode), que es exactamente el patrón real.
export async function generate(session, readLogits, promptIds, opts = {}) {
  const max = opts.maxTokens || 32;
  const stop = new Set(opts.stopIds || []);
  const onToken = opts.onToken || (() => {});
  const out = [];

  let ids = promptIds;
  for (let step = 0; step < max; step++) {
    const { logits, logitsTokens } = await session.forward(ids);
    // El forward devuelve logits SOLO del último token salvo que se pidan todos:
    // deducir el vocabulario dividiendo por el número de tokens del prompt daría
    // un vocabulario 316 veces más pequeño y una lectura corrida.
    const vocab = logits.size / 4 / logitsTokens;
    const all = await readLogits(logits, logitsTokens * vocab);
    const last = all.subarray((logitsTokens - 1) * vocab, logitsTokens * vocab);
    // Se le pasan los tokens ya emitidos MÁS el prompt: repetir lo que acaba de
    // decir el usuario es tan bucle como repetirse a sí mismo.
    const id = sample(last, { ...opts, seed: (opts.seed || 1) + step,
                              previos: promptIds.concat(out) });
    session.recycle();
    if (stop.has(id)) break;
    out.push(id);
    onToken(id, step);
    ids = [id];
  }
  return out;
}
