// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
// Elffuss Runtime · tokenizador BPE a nivel de byte, leído del propio GGUF.
// ─────────────────────────────────────────────────────────────────────────────
// El modelo no ve texto, ve enteros. Si el tokenizador difiere del que se usó
// para entrenar, el modelo recibe una secuencia que nunca ha visto y responde
// peor sin que nada falle: no hay excepción, no hay warning, solo calidad que
// baja. Es el mismo patrón de fallo silencioso que el resto del motor, así que
// se valida igual — contra el tokenizador de llama.cpp, caso por caso.
//
// Tres piezas, y las tres tienen su trampa:
//   1. PRE-TOKENIZADOR: un regex parte el texto en trozos ANTES del BPE. Es lo
//      que impide que un merge cruce la frontera entre una palabra y la
//      puntuación siguiente. Cada familia de modelos usa el suyo; usar el que
//      no es da tokenizaciones casi correctas.
//   2. MAPA BYTE→UNICODE de GPT-2: el BPE opera sobre texto, pero el vocabulario
//      cubre los 256 bytes. La tabla de OpenAI mueve los bytes no imprimibles a
//      un hueco de Unicode para que nunca aparezca un carácter de control. El
//      espacio se convierte en «Ġ», que es por qué el vocabulario está lleno de
//      ellas.
//   3. MERGES: el rango es el ORDEN en el array del GGUF. Siempre se aplica el
//      par de menor rango de toda la palabra, no el primero que se encuentre.
//
// Licencia: código propio, Apache-2.0 (ver SPDX arriba y LICENSE). Sin dependencias externas.

// Tipos de token de GGUF (llama_token_type).
const TYPE_NORMAL = 1, TYPE_CONTROL = 3, TYPE_USER_DEFINED = 4, TYPE_BYTE = 6;

// Tabla bytes→unicode de GPT-2. Los 188 bytes imprimibles se quedan donde
// están; los otros 68 se desplazan a partir de U+0100, donde no colisionan con
// nada real. Así una secuencia de bytes cualquiera es siempre texto válido.
function byteMaps() {
  const bs = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const direct = new Set(bs);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!direct.has(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const b2u = new Array(256);
  const u2b = new Map();
  for (let i = 0; i < bs.length; i++) {
    const ch = String.fromCodePoint(cs[i]);
    b2u[bs[i]] = ch;
    u2b.set(ch, bs[i]);
  }
  return { b2u, u2b };
}

const { b2u: BYTE_TO_UNI, u2b: UNI_TO_BYTE } = byteMaps();

// El conjunto de espacios de llama.cpp NO es el `\s` de JavaScript. Barriendo
// los 0x110000 code points difieren en DOS: U+0085 (NEL), que llama.cpp cuenta
// como espacio y JS no, y U+FEFF (BOM), al revés. Parece anecdótico y no lo es:
// un BOM al principio de un fichero cambia la tokenización de todo el trozo
// inicial. Por eso la clase va explícita y `\S` se emula negándola.
const WS = '\\t\\n\\u000B\\f\\r \\u0085\\u00A0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000';

// Pre-tokenizadores por valor de `tokenizer.ggml.pre`. Parten el texto ANTES del
// BPE para que ningún merge cruce de una palabra a la puntuación siguiente.
//
// Detalles que separan «funciona» de «casi funciona»:
//   · `\p{N}` es UN dígito, no `\p{N}{1,3}`. Con el de Llama-3 todo el texto
//     normal sale bien y solo se descuadran los números — el síntoma clásico.
//   · qwen35 consume marcas combinantes junto a las letras (`[\p{L}\p{M}]+`);
//     qwen2 no. Copiar el de qwen2 da resultados idénticos en inglés y ASCII y
//     divergentes en hindi, tailandés, vietnamita o árabe vocalizado.
//   · `(?i:…)` no existe en JS y la flag `i` global NO sirve: haría insensible
//     todo el patrón. La expansión a mano es la del propio llama.cpp.
//   · El orden de las 7 alternativas es semántico (leftmost-first con
//     retroceso, no leftmost-longest). Consecuencia contraintuitiva: el prefijo
//     opcional acepta puntuación, así que «!a» es UN token, y « a» también.
const CONTRACTIONS = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])";
const TAIL = `|\\p{N}| ?[^${WS}\\p{L}%M%\\p{N}]+[\\r\\n]*|[${WS}]*[\\r\\n]+|[${WS}]+(?![^${WS}])|[${WS}]+`;

const PRE_PATTERNS = {
  qwen35: CONTRACTIONS + '|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+' + TAIL.replace('%M%', '\\p{M}'),
  qwen2: CONTRACTIONS + '|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+' + TAIL.replace('%M%', ''),
};
PRE_PATTERNS.default = PRE_PATTERNS.qwen2;

export function createTokenizer(meta, opts = {}) {
  const tokens = meta['tokenizer.ggml.tokens'];
  const merges = meta['tokenizer.ggml.merges'];
  if (!tokens) throw new Error('el GGUF no trae tokenizer.ggml.tokens');
  const model = meta['tokenizer.ggml.model'];
  if (model !== 'gpt2') throw new Error('tokenizador no soportado: ' + model);

  const types = meta['tokenizer.ggml.token_type'];
  const preName = meta['tokenizer.ggml.pre'] || 'default';
  const pattern = opts.pattern || PRE_PATTERNS[preName] || PRE_PATTERNS.default;
  // La flag `u` es OBLIGATORIA y su ausencia no da error: sin ella `\p{L}` se
  // lee como la letra «p» seguida de «{L}», falla en silencio y produce una
  // tokenización distinta que en textos cortos parece casi bien.
  const re = new RegExp(pattern, 'gu');

  const bosId = meta['tokenizer.ggml.bos_token_id'];
  const eosId = meta['tokenizer.ggml.eos_token_id'];
  const addBosDefault = meta['tokenizer.ggml.add_bos_token'] ?? false;
  const addEosDefault = meta['tokenizer.ggml.add_eos_token'] ?? false;

  // Índices perezosos: materializar 248.320 strings cuesta memoria y tiempo, y
  // para DECODIFICAR no hacen falta — el array del GGUF ya indexa por posición.
  let vocab = null;      // string → id
  let ranks = null;      // "a\u0000b" → rango
  let specials = null;   // [{text, id}] ordenados por longitud descendente

  function buildEncode() {
    if (vocab) return;
    vocab = new Map();
    for (let i = 0; i < tokens.length; i++) vocab.set(tokens.at(i), i);
    ranks = new Map();
    if (merges) {
      for (let i = 0; i < merges.length; i++) {
        const m = merges.at(i);
        // Se parte por el PRIMER espacio: las piezas no pueden contenerlo
        // (el espacio real viaja como «Ġ»), así que esto no es ambiguo.
        const sp = m.indexOf(' ');
        if (sp < 0) continue;
        ranks.set(m.slice(0, sp) + '\u0000' + m.slice(sp + 1), i);
      }
    }
    // Índice de tokens especiales por PRIMER CARÁCTER. Este vocabulario tiene
    // cientos de «[PADnnnnnn]», y recorrerlos todos en cada posición del texto
    // convertiría la codificación en cuadrática.
    specials = new Map();
    if (types) {
      for (let i = 0; i < types.length; i++) {
        const t = types[i];
        if (t !== TYPE_CONTROL && t !== TYPE_USER_DEFINED) continue;
        const text = tokens.at(i);
        if (!text) continue;
        const k = text[0];
        if (!specials.has(k)) specials.set(k, []);
        specials.get(k).push({ text, id: i });
      }
      // De mayor a menor longitud: si existieran «<|im_start|>» y «<|im|>», hay
      // que probar el largo primero o se parte el otro por la mitad.
      for (const list of specials.values()) list.sort((a, b) => b.text.length - a.text.length);
    }
  }

  // Parte el texto en trozos normales y tokens especiales literales.
  function splitSpecials(text) {
    const out = [];
    let i = 0, start = 0;
    while (i < text.length) {
      const cands = specials.get(text[i]);
      let hit = null;
      if (cands) { for (const s of cands) if (text.startsWith(s.text, i)) { hit = s; break; } }
      if (hit) {
        if (i > start) out.push({ text: text.slice(start, i) });
        out.push({ id: hit.id });
        i += hit.text.length;
        start = i;
      } else i++;
    }
    if (start < text.length) out.push({ text: text.slice(start) });
    return out;
  }

  // BPE sobre una palabra ya pasada por el mapa de bytes. Se aplica SIEMPRE el
  // par de menor rango de toda la palabra; coger el primero que se encuentre da
  // resultados distintos y plausibles.
  function bpe(word) {
    if (word.length < 2) return word;
    let parts = word.slice();
    for (;;) {
      let best = Infinity, at = -1;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = ranks.get(parts[i] + '\u0000' + parts[i + 1]);
        if (r !== undefined && r < best) { best = r; at = i; }
      }
      if (at < 0) break;
      parts = parts.slice(0, at).concat(parts[at] + parts[at + 1], parts.slice(at + 2));
      if (parts.length === 1) break;
    }
    return parts;
  }

  const enc = new TextEncoder();
  const dec = new TextDecoder('utf-8');

  // Trozos del pre-tokenizador, CONSERVANDO los huecos entre coincidencias:
  // llama.cpp también los conserva, y descartarlos perdería texto en silencio.
  function preSplit(text) {
    const out = [];
    re.lastIndex = 0;
    let last = 0, m;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) out.push(text.slice(last, m.index));
      if (m[0].length === 0) { re.lastIndex++; continue; }
      out.push(m[0]);
      last = m.index + m[0].length;
    }
    if (last < text.length) out.push(text.slice(last));
    return out;
  }

  function encodeChunk(text, out) {
    for (const piece of preSplit(text)) {
      if (!piece) continue;
      // Texto → bytes UTF-8 → caracteres del mapa de GPT-2.
      const bytes = enc.encode(piece);
      const chars = new Array(bytes.length);
      for (let i = 0; i < bytes.length; i++) chars[i] = BYTE_TO_UNI[bytes[i]];
      for (const p of bpe(chars)) {
        const id = vocab.get(p);
        if (id !== undefined) { out.push(id); continue; }
        // Un trozo sin entrada en el vocabulario se parte carácter a carácter:
        // el vocabulario cubre los 256 bytes, así que siempre hay salida.
        for (const ch of p) {
          const single = vocab.get(ch);
          if (single !== undefined) out.push(single);
        }
      }
    }
  }

  return {
    vocabSize: tokens.length,
    bosId, eosId, preName,

    // `special: false` desactiva el reconocimiento de tokens de control en el
    // texto de entrada — útil para no dejar que un usuario inyecte «<|im_end|>»
    // escribiéndolo a mano.
    encode(text, { addBos = addBosDefault, addEos = addEosDefault, special = true } = {}) {
      buildEncode();
      const out = [];
      if (addBos && bosId !== undefined) out.push(bosId);

      if (special && specials.size) {
        for (const part of splitSpecials(text)) {
          if (part.id !== undefined) out.push(part.id);
          else encodeChunk(part.text, out);
        }
      } else {
        encodeChunk(text, out);
      }

      if (addEos && eosId !== undefined) out.push(eosId);
      return out;
    },

    // ids → texto. No necesita el índice de codificación, así que decodificar
    // un modelo recién cargado no paga la construcción del mapa de 248k.
    decode(ids) {
      let s = '';
      for (const id of ids) {
        if (id < 0 || id >= tokens.length) continue;
        s += tokens.at(id);
      }
      const bytes = new Uint8Array(s.length * 4);
      let n = 0;
      for (const ch of s) {
        const b = UNI_TO_BYTE.get(ch);
        if (b !== undefined) { bytes[n++] = b; continue; }
        // Un carácter fuera del mapa (token especial escrito literal) se
        // devuelve tal cual en UTF-8.
        for (const x of enc.encode(ch)) bytes[n++] = x;
      }
      return dec.decode(bytes.subarray(0, n));
    },

    // El texto de UN token, para emitir en streaming. Ojo: un token puede
    // cortar un carácter multibyte por la mitad, así que quien haga streaming
    // tiene que acumular bytes, no strings.
    tokenText(id) { return tokens.at(id); },
    tokenType(id) { return types ? types[id] : TYPE_NORMAL; },
    isControl(id) { return types ? types[id] === TYPE_CONTROL : false; },
  };
}

export { TYPE_NORMAL, TYPE_CONTROL, TYPE_USER_DEFINED, TYPE_BYTE };
