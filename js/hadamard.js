// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Rotación de Hadamard declarada por el fichero: qué tensores la llevan y cómo.
// ─────────────────────────────────────────────────────────────────────────────
// Algunos modelos ternarios guardan los pesos en una base girada y declaran la
// rotación en sus metadatos. Un runtime que la ignore NO falla: carga, ejecuta y
// genera ruido — comprobado, produce tokens plausibles y texto sin sentido. Por
// eso esto se decide a partir de lo que dice el FICHERO y nunca se deduce:
//
//   · `weight_names` enumera, uno a uno, los 401 tensores cuya entrada hay que
//     girar. No se adivina por el nombre ni por la forma.
//   · `sign_widths` / `sign_values` traen un vector de signos por anchura de
//     entrada, concatenados. Se elige por la anchura del tensor.
//   · `inverse_weight_names` son las tablas que guardan las filas ya giradas y
//     hay que DESgirar tras la búsqueda.
//
// Si el fichero declara una versión o una transformada que no conocemos, esto
// LANZA en vez de seguir. Es la única respuesta honesta: continuar significaría
// devolver ruido con cara de respuesta.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE).

const CLAVE = 'prism.hadamard.';

// Los arrays de CADENAS del GGUF no son arrays: el parser los devuelve
// perezosos ({length, at(i), toArray()}) para no materializar 248.000 tokens de
// vocabulario cada vez que alguien mira los metadatos. Los de escalares sí son
// vistas tipadas y se iteran solos. Un `new Set(...)` sobre el perezoso no
// lanza nada legible — dice «object is not iterable» — así que se normaliza
// aquí, en un sitio, en vez de esparcir la comprobación.
const lista = (v) => {
  if (v == null) return [];
  if (typeof v.toArray === 'function') return v.toArray();
  if (Array.isArray(v) || ArrayBuffer.isView(v)) return Array.from(v);
  throw new Error('se esperaba una lista en los metadatos de la rotación');
};

export function leeRotacion(meta) {
  const v = meta[CLAVE + 'version'];
  if (v === undefined) return null;                 // modelo sin rotación: normal
  if (v !== 1) throw new Error(`rotación de Hadamard versión ${v}: no soportada`);

  const transformada = meta[CLAVE + 'transform'];
  if (transformada !== 'normalized-sylvester-walsh-hadamard') {
    throw new Error(`transformada desconocida: «${transformada}»`);
  }
  const eje = meta[CLAVE + 'axis'];
  if (eje !== 'input-last-dimension') throw new Error(`eje de rotación desconocido: «${eje}»`);
  const modo = meta[CLAVE + 'sign_mode'];
  if (modo !== 'explicit') throw new Error(`modo de signos desconocido: «${modo}»`);

  const bloque = Number(meta[CLAVE + 'block_size']);
  if (!bloque || (bloque & (bloque - 1)) !== 0) {
    throw new Error(`bloque de rotación inválido: ${bloque}`);
  }
  const pesos = new Set(lista(meta[CLAVE + 'weight_names']));
  if (!pesos.size) throw new Error('la rotación no declara ningún tensor');
  const inversos = new Set(lista(meta[CLAVE + 'inverse_weight_names']));

  // Los signos vienen concatenados por anchura, en el orden de `sign_widths`.
  const anchuras = lista(meta[CLAVE + 'sign_widths']).map(Number);
  const valores = lista(meta[CLAVE + 'sign_values']);
  const signos = new Map();
  let o = 0;
  for (const a of anchuras) {
    if (a <= 0 || a % bloque !== 0) throw new Error(`anchura de signos inválida: ${a}`);
    const v = new Float32Array(a);
    for (let i = 0; i < a; i++) {
      const s = Number(valores[o + i]);
      if (s !== 1 && s !== -1) throw new Error(`signo que no es ±1 en la posición ${o + i}: ${s}`);
      v[i] = s;
    }
    signos.set(a, v);
    o += a;
  }
  if (o !== valores.length) {
    throw new Error(`sobran o faltan signos: ${valores.length} valores para ${o} declarados`);
  }

  return {
    bloque, pesos, inversos, signos,
    vGrouped: meta[CLAVE + 'gdn_v_grouped'] === true,
    // Cuántos tensores declara, para poder comprobar que se aplicaron todos.
    total: pesos.size + inversos.size,
  };
}
