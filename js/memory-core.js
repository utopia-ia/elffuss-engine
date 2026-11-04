// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
//
// User memory: the record, the write gate, slots, the deterministic proposer,
// consolidation decisions and the rendering of the context block. Pure: no
// storage, no DOM. The store (memory-store.js) and the client (memory.js) both
// run this code, so a write is judged by the same rules on both sides.
//
// The safety rules here are code, not configuration and not tuning: the lint,
// the taint rule, the support floor and "only typed user text grounds a fact".
// A brand config or a tuning file can make them stricter, never looser.

import {
  fold, terms, contentTerms, prefixMatch, findDates, startOfDay, endOfDay, addDays,
  estimateTokens, fitBlock, GROUNDING_STOP, ROLES, STOPWORDS,
} from './memory-rank.js';
import { t, raw, formatWhen, fill } from './memory-text.js';

export const SCHEMA = 2;

// Hard limits. A config may lower them (memory-config.js), never raise them.
export const LIMITS = Object.freeze({
  factMax: 300,
  active: 1000,
  history: 2000,
  proposals: 50,
  pinned: 12,
  message: 65536,
  writesPerMinute: 30,
  writesPerDay: 300,
  forgetsPerMinute: 5,
  readsPerMinute: 60,
  searchMax: 20,
  listMax: 50,
  undoGraceMs: 10000,
  expiredPurgeDays: 180,
  proposalDays: 7,
  typedMax: 1500,
  outboxMax: 20,
  outboxDays: 7,
  catchUpDays: 60,
  tombstoneDays: 30,
  forgetCardsPerTurn: 3,
  ridKeep: 64,
});

// The support floor. Tuning may raise `shared.memory.write.support` up to its
// schema maximum; nothing may lower it below this.
export const SUPPORT_FLOOR = 0.6;

// ── Ids and hashes ───────────────────────────────────────────────────────────

const B36 = '0123456789abcdefghijklmnopqrstuvwxyz';
function rand36(n) {
  const a = new Uint8Array(n);
  globalThis.crypto.getRandomValues(a);
  let s = '';
  for (const x of a) s += B36[x % 36];
  return s;
}
export const newId = () => 'm' + rand36(10);
export const newPid = () => 'p' + rand36(10);
export const newRid = () => 'r' + rand36(12);
export const newConv = () => 'c' + rand36(12);

export async function sha256hex(s) {
  const buf = await globalThis.crypto.subtle.digest('SHA-256', new TextEncoder().encode(String(s)));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');
}
export const hashFact = (fact) => sha256hex(fold(fact));

// ── Language guess (only to pick date wording for a fact) ────────────────────

const ES_HINT = new Set(['el', 'la', 'de', 'que', 'en', 'y', 'los', 'las', 'su', 'se', 'es', 'con', 'por', 'una', 'mi', 'vive', 'tiene', 'trabaja', 'prefiere', 'mañana', 'manana']);
const EN_HINT = new Set(['the', 'of', 'and', 'is', 'in', 'to', 'my', 'his', 'her', 'their', 'with', 'lives', 'works', 'prefers', 'has', 'at', 'on']);
export function detectLang(text, fallback = 'und') {
  let e = 0, n = 0;
  for (const w of terms(text)) { if (ES_HINT.has(w)) e++; if (EN_HINT.has(w)) n++; }
  if (/[ñ¿¡áéíóú]/i.test(text)) e += 2;
  if (e > n) return 'es';
  if (n > e) return 'en';
  return fallback;
}

// ── Normalisation ────────────────────────────────────────────────────────────

const QUOTES = /^[\s"'«»“”‘’`]+|[\s"'«»“”‘’`]+$/g;

// NFC, one line, single spaces, no surrounding quotes, no final period, and
// relative dates resolved against `now` («mañana» → «el 2026-11-04,
// miércoles»). Returns the text plus the event dates it found.
export function normalizeFact(fact, { now = Date.now(), lang = null } = {}) {
  let s = String(fact ?? '').normalize('NFC').replace(/[\r\n\t\f\v\u2028\u2029]+/g, ' ').replace(/ {2,}/g, ' ');
  s = s.replace(QUOTES, '').trim();
  if (/[^.]\.$/.test(s)) s = s.slice(0, -1).trim();
  const L = lang && lang !== 'und' ? lang : detectLang(s, 'es');
  const dates = findDates(s, now, L);
  // Replace from the end so earlier indexes stay valid.
  for (const d of [...dates].reverse()) {
    if (!d.rel) continue;
    s = s.slice(0, d.index) + datePhrase(d, L) + s.slice(d.index + d.length);
  }
  s = s.replace(/ {2,}/g, ' ').trim();
  const resolved = findDates(s, now, L);
  const days = resolved.filter((d) => d.kind === 'day' || d.kind === 'range' || d.kind === 'week');
  const eventAt = days.length ? startOfDay(days[0].start) : null;
  const expiresAt = days.length ? addDays(startOfDay(days[0].end), 2) - 1 : null;
  return { text: s, eventAt, expiresAt, lang: L };
}

function datePhrase(d, lang) {
  const day = formatWhen(d.start, lang, null, { weekday: d.kind === 'day' });
  const plain = formatWhen(d.start, lang, null);
  if (lang === 'en') {
    if (d.kind === 'week') return `the week of ${plain}`;
    if (d.kind === 'month') return `in ${plain.slice(0, 7)}`;
    if (d.kind === 'range') return `the weekend of ${plain}`;
    return `on ${day}`;
  }
  if (d.kind === 'week') return `la semana del ${plain}`;
  if (d.kind === 'month') return `en ${plain.slice(0, 7)}`;
  if (d.kind === 'range') return `el fin de semana del ${plain}`;
  return `el ${day}`;
}

// ── Lint (narrow on purpose) ─────────────────────────────────────────────────

// Rejects only what is clearly not a fact about the user: chat markup,
// override phrases, triggers aimed at the assistant, exfiltration with a
// destination, active markup, URLs with a query or fragment, code fences and
// invisible characters. It does NOT reject ordinary words such as «IA»,
// «sistema», «envía», a bare domain or an e-mail address on its own.
const LINT = [
  ['markup', /<\||\|>|\[\/?INST\]|<\/?system>|<\/?assistant>|^\s*(system|assistant|developer)\s*:|\n\s*(system|assistant)\s*:/i],
  ['override', /\bignore\s+(all|any|the\s+)?(previous|prior|above|earlier)?\s*(instructions|rules|prompts?)\b|\bignore (all|previous|prior)\b|\bdisregard\b|\bignora\s+(todo|lo anterior|las instrucciones|tus instrucciones|las reglas)\b|\bolvida (tus|las) instrucciones\b|\bfrom now on,?\s+(you|always|never|reply|respond|answer|say|send|include)\b|\ba partir de ahora,?\s+(responde|envía|envia|incluye|di|siempre|nunca|contesta)\b|\byou must now\b|\bnew instructions\b|\bnuevas instrucciones\b/i],
  ['trigger', /\bwhen(ever)? the user (says|types|asks|writes|replies)\b|\bif the user (says|types|asks|writes|replies)\b|\bcuando el usuario (diga|escriba|pida|responda|conteste)\b|\bsi el usuario (dice|escribe|pide|responde|contesta)\b/i],
  ['exfiltration', /\b(send|post|forward|upload|fetch|leak|exfiltrate|env[ií]a|enviar|manda|mandar|reenv[ií]a|reenviar|sube|subir|filtra)\b[^\n]{0,120}?\b(to|a|al|hacia)\s+(https?:\/\/|www\.|[\w.+-]+@[\w-]+\.[\w.]+|@[\w]+|[\w-]+\.(com|net|org|io|dev|xyz|ru|cn|info|app|site)\b)/i],
  ['active-markup', /!\[|<img\b|<script\b|javascript:|<iframe\b|on(error|load)\s*=/i],
  ['url-query', /\bhttps?:\/\/[^\s]*[?#]/i],
  ['code-fence', /```|~~~/],
  // The memory block's own fence, in the built-in languages: a fact must never
  // open or close it. Brands with another tag are covered by renderBlock,
  // which neutralises every tag-like '<' in a fact.
  ['fence', /<\s*\/?\s*(memoria|memory|mem[oó]ria|m[eé]moire|ged[aä]chtnis)\b/i],
  ['invisible', /[\u{E0000}-\u{E007F}]|[\u202a-\u202e\u2066-\u2069]|[\u200b-\u200d\u2060\ufeff]{3,}/u],
];

export function lintFact(fact) {
  const s = String(fact ?? '');
  for (const [reason, re] of LINT) if (re.test(s)) return { ok: false, reason };
  return { ok: true, reason: null };
}

// ── Remember triggers ────────────────────────────────────────────────────────

const ASKING = /\b(do|did|can|could) you remember\b|\bwhat do you remember\b|\bte acuerdas\b|\bque recuerdas\b|\brecuerdas\b|\bqu[eé] sabes de m[ií]\b/gi;
const TRIGGERS = /\brecuerda(lo|me|te)?\b|\bacuerdate\b|\bno (te )?olvides\b|\bapunta(lo|te|me)?\b|\bguarda(lo)? (que|esto|eso|en (la )?memoria)\b|\bten en cuenta que\b|\bque no se te olvide\b|\bmemoriza(lo)?\b|\bremember\b|\bdon'?t forget\b|\bdo not forget\b|\bnote that\b|\bmake a note\b|\bkeep in mind\b|\bsave (that|this|it)\b|\bmemori[sz]e\b/;

export function hasTrigger(current, lang = 'es', strings = null) {
  const s = fold(current).replace(ASKING, ' ');
  if (TRIGGERS.test(s)) return true;
  const extra = raw(strings, lang, 'triggers');
  if (Array.isArray(extra)) {
    for (const src of extra) {
      try { if (new RegExp(src, 'iu').test(s)) return true; } catch { /* a bad brand pattern never matches */ }
    }
  }
  return false;
}

// ── Slots ────────────────────────────────────────────────────────────────────

const V = (s) => s.split(/\s+/).filter(Boolean);
export const SLOTS = Object.freeze({
  name: { exclusive: true, vocab: V('nombre llamo llamas llama name called') },
  home_city: { exclusive: true, vocab: V('vivir vivo vives vive ciudad casa mudanza mudado live lives living city home moved') },
  employer: { exclusive: true, vocab: V('trabajo trabajar trabaja empresa work works job company employer') },
  job: { exclusive: true, vocab: V('trabajo trabajar trabaja oficio profesion dedico work works job profession') },
  birthday: { exclusive: true, vocab: V('cumpleanos cumple nacio naci birthday born') },
  reply_language: { exclusive: true, vocab: V('idioma lengua responder respuestas language answer answers reply') },
  answer_style: { exclusive: true, vocab: V('respuestas estilo cortas largas answers style short long') },
  diet: { exclusive: true, vocab: V('dieta come comer comida diet eat food') },
  allergy: { exclusive: false, vocab: V('alergia alergico alergica allergy allergic') },
  rel: { exclusive: false, vocab: V('llama nombre called name') },
  pet: { exclusive: false, vocab: V('mascota pet llama called') },
  stack: { exclusive: false, vocab: V('usa uso programa codigo stack uses code') },
});
const slotBase = (slot) => String(slot || '').split(':')[0];
export const isExclusive = (slot) => !!SLOTS[slotBase(slot)]?.exclusive;
export const slotVocab = (slot) => SLOTS[slotBase(slot)]?.vocab || [];

const LANGS = 'ingles|espanol|castellano|frances|aleman|italiano|portugues|catalan|gallego|euskera|english|spanish|french|german|italian|portuguese|catalan';
const DIETS = 'vegetarian[oa]?|vegan[oa]?|celiac[oa]?|coeliac|pescetarian|pescetarian[oa]|flexitarian';
const TECH = 'python|javascript|typescript|golang|java|kotlin|swift|ruby|php|c\\+\\+|c#|pnpm|npm|yarn|react|vue|svelte|node|deno|bun|django|rails|flutter|elixir|haskell|scala|sql|postgres|sqlite|linux|vim|neovim|emacs|docker|kubernetes';
const ROLE_RE = Object.keys(ROLES).join('|');

// Which slot a fact fills, or null. Works on the third-person templates and on
// first-person text.
export function slotOf(fact) {
  const f = fold(fact);
  const rel = new RegExp(`\\b(?:su|mi|my|their|his|her)\\s+(${ROLE_RE})\\s+(?:se llama|is called|is named|'s name is|name is|es)\\b`).exec(f);
  if (rel) {
    const role = ROLES[rel[1]];
    return `rel:${role}`;
  }
  if (/\b(?:alergic[oa]|allergic)\s+(?:a|al|to)\s+(?:la |las |los |el )?(\p{L}+)/u.test(f)) {
    const m = /\b(?:alergic[oa]|allergic)\s+(?:a|al|to)\s+(?:la |las |los |el )?(\p{L}+)/u.exec(f);
    return `allergy:${m[1]}`;
  }
  const named = new RegExp(`\\b(${ROLE_RE})\\s+(?:que\\s+)?(?:se llama|called|named)\\b`).exec(f);
  if (named) return `rel:${ROLES[named[1]]}`;
  if (/^(?:se llama|me llamo|mi nombre es|su nombre es|nombre:|is called|my name is|name is|call me|(?:their|her|his) name is)\b/.test(f)) return 'name';
  if (/\b(?:vive|vivo|vivimos) en\b|\bse ha mudado a\b|\bme he mudado a\b|\bse mudo a\b|\bme mude a\b|\b(?:lives|live|living) in\b|\bmoved to\b|^(?:es|soy) de\s+\p{L}/u.test(f) && !/\b(?:es|soy) de (?:color|tipo|verdad|acuerdo)\b/.test(f)) return 'home_city';
  if (/\btrabaj[ao] (?:de|como)\b|\bworks? as\b|\bjob is\b|\bse dedica a\b|\bme dedico a\b/.test(f)) return 'job';
  if (/\btrabaj[ao] en\b|\bworks? (?:at|for)\b|\bemployer is\b/.test(f)) return 'employer';
  if (/\bcumpleanos\b|\bbirthday\b/.test(f)) return 'birthday';
  if (new RegExp(`\\b(?:respond|contest|habl|answer|repl|speak|talk|prefiere que le respondan|prefers answers)[\\p{L}]*\\b[^.]*\\b(?:en|in)\\s+(?:${LANGS})\\b`, 'u').test(f) || new RegExp(`^(?:${LANGS}) de \\p{L}+$`, 'u').test(f)) return 'reply_language';
  if (/\brespuestas (?:cortas|largas|breves|detalladas|concisas)\b|\b(?:short|long|brief|detailed|concise) answers\b/.test(f)) return 'answer_style';
  if (new RegExp(`\\b(?:es|soy|is|am|i'm)\\s+(?:a\\s+)?(?:${DIETS})\\b`).test(f)) return 'diet';
  if (new RegExp(`\\b(?:usa|uso|programa en|programo en|uses|use|codes in|code in)\\s+(?:${TECH})\\b`).test(f)) return 'stack';
  return null;
}

// The value part of a slot fact: its content terms minus the slot vocabulary
// and the verbs that name the slot. Two facts in the same exclusive slot with
// overlapping values say the same thing.
function slotValueTerms(fact, slot) {
  const vocab = new Set(slotVocab(slot).map(fold));
  const extra = new Set(V('vive vivo en se ha mudado a trabaja trabajo de como lives live in moved to works work at as is called llama llamo nombre name prefiere prefers respondan answers es soy is am'));
  return contentTerms(fact).filter((w) => !vocab.has(w) && !extra.has(w));
}

const NEGATION = /\bya no\b|\bdej[oóe] de\b|\bdejado de\b|\bno longer\b|\bused to\b|\bnot anymore\b|\bno more\b|\bnunca mas\b/;
export const isNegated = (fact) => NEGATION.test(fold(fact));

export function kindOf(fact, slot, eventAt) {
  if (eventAt) return 'event';
  if (slot === 'reply_language' || slot === 'answer_style' || /\bprefier|\bprefer|\ble gusta|\blikes?\b|\bodia\b|\bhates?\b|\bno le gusta\b|\bdislikes?\b/.test(fold(fact))) return 'preference';
  if (slot) return 'profile';
  return 'fact';
}

export function keysFor(fact, slot) {
  const out = [];
  const push = (w) => { const f = fold(w); if (f && !out.includes(f) && out.length < 12) out.push(f); };
  for (const w of slotVocab(slot).slice(0, 8)) push(w);
  for (const w of contentTerms(fact)) push(w);
  return out;
}

// ── Similarity ───────────────────────────────────────────────────────────────

export function similarity(a, b) {
  const A = [...new Set(contentTerms(a))], B = [...new Set(contentTerms(b))];
  if (!A.length || !B.length) return 0;
  let m = 0;
  for (const x of A) if (B.some((y) => prefixMatch(x, y))) m++;
  return m / (A.length + B.length - m);
}

// ── Support: is the fact grounded in what the user typed? ────────────────────

function looseMatch(a, b) {
  if (prefixMatch(a, b)) return true;
  // 1st / 3rd person of the same verb («vivo» / «vive»).
  return a.length >= 4 && b.length >= 4 && a.length === b.length && a.slice(0, -1) === b.slice(0, -1) && /[aeiou]$/.test(a) && /[aeiou]$/.test(b);
}

// The share of the fact's content terms found in the user's typed text. A date
// the gate resolved («el 2026-11-04, miércoles») counts as present when the
// user wrote a date of their own («mañana»). Slot words count as present when
// the user used any word of that slot («vive» against «vivo en»).
export function support(fact, typedTexts, { now = Date.now(), lang = 'es' } = {}) {
  const typed = (Array.isArray(typedTexts) ? typedTexts : [typedTexts]).filter((x) => typeof x === 'string').join('\n');
  let f = String(fact ?? '');
  const userDates = findDates(typed, now, lang).length + findDates(typed, now, lang === 'es' ? 'en' : 'es').length;
  if (userDates) {
    for (const d of [...findDates(f, now, lang)].reverse()) f = f.slice(0, d.index) + ' ' + f.slice(d.index + d.length);
  }
  const factTerms = [...new Set(contentTerms(f))].filter((w) => !GROUNDING_STOP.has(w));
  if (!factTerms.length) return 0;
  const typedTerms = [...new Set(terms(typed))];
  const slot = slotOf(fact);
  const vocab = new Set(slotVocab(slot).map(fold));
  const slotUsed = vocab.size && typedTerms.some((w) => [...vocab].some((v) => looseMatch(w, v)));
  let hit = 0;
  for (const w of factTerms) {
    if (typedTerms.some((u) => looseMatch(w, u))) { hit++; continue; }
    if (slotUsed && [...vocab].some((v) => looseMatch(w, v))) hit++;
  }
  return hit / factTerms.length;
}

// ── The write gate ───────────────────────────────────────────────────────────

// Decides what a save request becomes. Runs in the client and again in the
// store. `shared` is the store-wide tuning (`sharedTuning().memory.write`).
//   -> { ok: false, code, n? }
//   -> { ok: true, status: 'active' | 'proposal', how?, label?, fact, eventAt, expiresAt, support }
export function gateWrite({ fact, evidence, tainted, lang = 'es', shared, now = Date.now(), strings = null, limits = LIMITS }) {
  if (typeof fact !== 'string' || !fact.trim()) return { ok: false, code: 'bad_args' };
  const norm = normalizeFact(fact, { now });
  if (!norm.text) return { ok: false, code: 'bad_args' };
  const max = Math.min(LIMITS.factMax, limits.factMax || LIMITS.factMax);
  if (norm.text.length > max) return { ok: false, code: 'too_long', n: norm.text.length };
  const lint = lintFact(norm.text);
  if (!lint.ok) return { ok: false, code: 'not_a_fact', reason: lint.reason };
  // The raw fact is linted too: normalisation must not hide markup.
  if (!lintFact(fact).ok) return { ok: false, code: 'not_a_fact', reason: lintFact(fact).reason };

  const ev = evidence && typeof evidence === 'object' ? evidence : {};
  const typedTexts = (Array.isArray(ev.typedTexts) ? ev.typedTexts : []).filter((x) => typeof x === 'string').slice(-3);
  const current = typeof ev.current === 'string' ? ev.current : '';
  const s = Math.max(SUPPORT_FLOOR, Number(shared?.support) || SUPPORT_FLOOR);
  const sup = support(norm.text, typedTexts.length ? typedTexts : [current], { now, lang: norm.lang });
  const trigger = !!current && hasTrigger(current, lang, strings);
  const grounded = sup >= s;
  const base = { ok: true, fact: norm.text, eventAt: norm.eventAt, expiresAt: norm.expiresAt, support: sup, lang: norm.lang };
  if (tainted !== false) {
    // Fails closed: anything but an explicit `false` counts as tainted.
    return grounded ? { ...base, status: 'proposal', label: 'from-content' } : { ok: false, code: 'not_from_user' };
  }
  if (trigger && grounded) return { ...base, status: 'active', how: 'user-asked' };
  if (!trigger && grounded) return { ...base, status: 'proposal', label: 'from-user' };
  if (trigger && !grounded) return { ...base, status: 'proposal', label: 'from-content' };
  return { ok: false, code: 'not_from_user' };
}

// ── Deterministic proposer ───────────────────────────────────────────────────

const SENSITIVE = /\b(religi|catolic|musulman|judi[oa]|ateo|cristian|votar|voto|partido|politic|izquierda|derecha|gay|lesbiana|bisexual|trans\b|sexual|embaraz|cancer|depresi|ansiedad|diabet|vih|enfermedad|medicaci|terapia|psiqui|religion|catholic|muslim|jewish|atheist|christian|vote|party|politic|left[- ]wing|right[- ]wing|pregnan|depress|anxiety|disease|medication|therapy|psychiatr|hiv)/i;
const NOT_A_STATEMENT = /\?|¿|^\s*(si|if|cuando|when|quizas|quizá|quiza|maybe|perhaps|igual)\b|\b(dice|dijo|dicen|says|said|told me|me dijo|cree que|thinks)\b|\b(ojal[aá]|i wish|me gustar[ií]a vivir|would like to live)\b/i;
const CAP = (s) => s.charAt(0).toLocaleUpperCase() + s.slice(1);
const ES_ROLES = { hija: 'hija', hijo: 'hijo', pareja: 'pareja', perro: 'perro', perra: 'perra', gato: 'gato', gata: 'gata', mujer: 'mujer', marido: 'marido' };

// Each entry: [slot, keyword pattern (case-insensitive, ends where the value
// starts), value pattern (case-SENSITIVE, so «Valencia de Don Juan» keeps
// its capitalised words and «desde mayo» is left out), language].
const ONE = "[\\p{L}][\\p{L}'’.-]*";
const PLACE_V = new RegExp(`^${ONE}(?:\\s+(?:de(?:l)?\\s+|la\\s+|las\\s+|los\\s+|of\\s+|the\\s+)?\\p{Lu}[\\p{L}'’.-]*)*`, 'u');
const NAME_V = new RegExp(`^${ONE}(?:\\s+\\p{Lu}[\\p{L}'’-]*)?`, 'u');
const WORDS_V = new RegExp(`^${ONE}(?:\\s+(?!(?:y|and|desde|since|en|in|para|for|con|with)\\b)${ONE}){0,2}`, 'u');
const WORD_V = new RegExp(`^${ONE}`, 'u');
const LANG_V = new RegExp(`^(?:${LANGS}|ingl[eé]s|espa[ñn]ol|franc[eé]s|alem[aá]n|portugu[eé]s|catal[aá]n)\\b`, 'iu');
const TECH_V = new RegExp(`^(?:${TECH})\\b`, 'iu');
const ANY_V = /^[^.,;!?\n]{3,30}/u;
const PROPOSALS = [
  ['name', /\b(?:me llamo|mi nombre es)\s+/iu, NAME_V, 'es'],
  ['name', /\b(?:my name is|call me)\s+/iu, NAME_V, 'en'],
  ['home_city', /\b(?:vivo en|me he mudado a|me mudé a|me mude a|soy de)\s+/iu, PLACE_V, 'es'],
  ['home_city', /\b(?:i live in|i moved to|i've moved to|i have moved to|i'm from|i am from)\s+/iu, PLACE_V, 'en'],
  ['job', /\btrabajo (?:de|como)\s+/iu, WORDS_V, 'es'],
  ['job', /\bi work as (?:an? )?/iu, WORDS_V, 'en'],
  ['employer', /\btrabajo en\s+/iu, PLACE_V, 'es'],
  ['employer', /\bi work (?:at|for)\s+/iu, PLACE_V, 'en'],
  ['birthday', /\bmi cumplea[ñn]os es (?:el\s+)?/iu, ANY_V, 'es'],
  ['birthday', /\bmy birthday is (?:on\s+)?/iu, ANY_V, 'en'],
  ['reply_language', /\b(?:resp[oó]ndeme|cont[eé]stame|h[aá]blame|escr[ií]beme)\s+(?:siempre\s+)?en\s+/iu, LANG_V, 'es'],
  ['reply_language', /\b(?:answer|reply|respond|speak|talk|write)\s+(?:to\s+)?(?:me\s+)?(?:always\s+)?in\s+/iu, LANG_V, 'en'],
  ['answer_style', /\bprefiero (?:las )?respuestas\s+/iu, /^(?:cortas|largas|breves|detalladas|concisas)\b/iu, 'es'],
  ['answer_style', /\bi prefer\s+(?=(?:short|long|brief|detailed|concise) answers)/iu, /^(?:short|long|brief|detailed|concise)\b/iu, 'en'],
  ['allergy', /\bsoy al[eé]rgic[oa] (?:a|al)\s+(?:la |las |los |el )?/iu, WORD_V, 'es'],
  ['allergy', /\bi(?:'m| am) allergic to\s+/iu, WORD_V, 'en'],
  ['diet', /\bsoy\s+(?=(?:vegetarian[oa]|vegan[oa]|cel[ií]ac[oa])\b)/iu, /^(?:vegetarian[oa]|vegan[oa]|cel[ií]ac[oa])\b/iu, 'es'],
  ['diet', /\bi(?:'m| am) (?:a )?(?=(?:vegetarian|vegan|coeliac|celiac)\b)/iu, /^(?:vegetarian|vegan|coeliac|celiac)\b/iu, 'en'],
  ['rel', new RegExp(`\\bmi (${Object.keys(ES_ROLES).join('|')})\\s+se llama\\s+`, 'iu'), WORD_V, 'es'],
  ['rel', /\bmy (daughter|son|partner|dog|cat|wife|husband)(?:'s name)? is (?:called |named )?/iu, WORD_V, 'en'],
  ['stack', /\b(?:uso|programo en)\s+/iu, TECH_V, 'es'],
  ['stack', /\bi (?:use|code in|program in)\s+/iu, TECH_V, 'en'],
];
const EVENT_ES = /\btengo (?:una? )?(cita|dentista|examen|viaje|reuni[oó]n|m[eé]dico|entrevista|vuelo|boda)\b/iu;
const EVENT_EN = /\bi have (?:an? |my )?(appointment|dentist|exam|trip|meeting|interview|doctor|flight|wedding)\b/iu;
const TIME = /\b(?:a las|at)\s+(\d{1,2}(?::\d{2})?)\b/i;

// At most one proposal from one typed user message, or null. Only first-person
// statements; never questions, conditionals, negations or reported speech;
// never health beyond allergies, religion, politics or sexuality.
export function proposeFromUser(text, lang = 'es', { now = Date.now(), strings = null } = {}) {
  const s = String(text ?? '').normalize('NFC');
  if (!s.trim() || s.length > LIMITS.typedMax) return null;
  if (NOT_A_STATEMENT.test(s) || NEGATION.test(fold(s)) || SENSITIVE.test(fold(s).replace(/\balergi\w*|\ballerg\w*/g, ''))) return null;
  if (!lintFact(s).ok) return null;
  for (const [slot, re, valueRe, L] of PROPOSALS) {
    const m = re.exec(s);
    if (!m) continue;
    const v = valueRe.exec(s.slice(m.index + m[0].length));
    if (!v) continue;
    let fact;
    if (slot === 'rel') {
      const role = m[1].toLowerCase();
      fact = t(strings, L, 'tpl.rel', { role, x: CAP(v[0]) });
    } else {
      const x = v[0].trim().replace(/[.,;:!]+$/, '');
      if (!x) continue;
      const val = slot === 'answer_style' || slot === 'diet' || slot === 'allergy' || slot === 'job' ? x.toLowerCase()
        : slot === 'reply_language' ? x.toLowerCase() : CAP(x);
      fact = t(strings, L, `tpl.${slot}`, { x: val });
    }
    const norm = normalizeFact(fact, { now, lang: L });
    const sl = slotOf(norm.text);
    return { fact: norm.text, slot: sl, kind: kindOf(norm.text, sl, norm.eventAt), lang: L };
  }
  // Events need a date in the same message.
  for (const [re, L] of [[EVENT_ES, 'es'], [EVENT_EN, 'en']]) {
    const m = re.exec(s);
    if (!m) continue;
    const dates = findDates(s, now, L).filter((d) => d.kind === 'day');
    if (!dates.length) return null;
    const time = TIME.exec(s);
    const what = CAP(m[1].toLowerCase()) + (time ? (L === 'es' ? ` a las ${time[1]}` : ` at ${time[1]}`) : '');
    const fact = t(strings, L, 'tpl.event', { what, date: formatWhen(dates[0].start, L, strings, { weekday: true }) });
    const norm = normalizeFact(fact, { now, lang: L });
    return { fact: norm.text, slot: null, kind: 'event', lang: L };
  }
  return null;
}

// ── Records ──────────────────────────────────────────────────────────────────

const ID_RE = /^m[0-9a-z]{1,10}$/;
const STATUSES = new Set(['active', 'superseded', 'expired', 'review']);
const KINDS = new Set(['profile', 'preference', 'fact', 'event']);
const HOWS = new Set(['user-asked', 'user-confirmed', 'user-typed', 'imported', 'legacy']);

export function validateRecord(r) {
  const p = [];
  if (!r || typeof r !== 'object') return { ok: false, problems: ['not an object'] };
  if (typeof r.id !== 'string' || !ID_RE.test(r.id)) p.push('id');
  if (typeof r.fact !== 'string' || !r.fact.trim()) p.push('fact');
  else if (r.fact.length > LIMITS.factMax && !r.tooLong) p.push('fact length');
  if (!Number.isFinite(r.at)) p.push('at');
  if (r.v === 2) {
    if (!STATUSES.has(r.status)) p.push('status');
    if (!KINDS.has(r.kind)) p.push('kind');
    if (typeof r.scope !== 'string' || !/^(all|app:[a-z][a-z0-9-]{0,23}|project:[a-z0-9]{1,40})$/.test(r.scope)) p.push('scope');
    if (!r.source || !HOWS.has(r.source.how)) p.push('source');
    if (!Array.isArray(r.keys)) p.push('keys');
    if (![1, 2, 3, 4, 5].includes(r.importance)) p.push('importance');
  }
  return { ok: p.length === 0, problems: p };
}

// A fresh v2 record. `hash` is filled by the caller (async).
export function makeRecord({ id = newId(), fact, at, now = Date.now(), kind, slot, scope = 'all', importance = 3, source, eventAt = null, expiresAt = null, lang = 'und', hash = '', status = 'active' }) {
  const sl = slot === undefined ? slotOf(fact) : slot;
  return {
    id, fact, at: at ?? now,
    v: 2, updatedAt: now, lastUsedAt: 0, useCount: 0,
    kind: kind || kindOf(fact, sl, eventAt),
    slot: sl, keys: keysFor(fact, sl), pinned: false,
    importance: Math.min(5, Math.max(1, importance | 0 || 3)),
    scope, status, eventAt, expiresAt, validTo: null,
    supersedes: null, supersededBy: null, possibleDup: null,
    lang, hash, source,
  };
}

// A v1 row {id, fact, at} as a v2 record, with id, fact and at byte for byte.
// Returns null for empty or malformed rows (they are counted, never invented).
export function fromLegacy(row, { hash = '' } = {}) {
  if (!row || typeof row !== 'object') return null;
  if (typeof row.id !== 'string' || !row.id || typeof row.fact !== 'string' || !row.fact.trim()) return null;
  const at = Number.isFinite(row.at) ? row.at : 0;
  const slot = slotOf(row.fact);
  const rec = {
    id: row.id, fact: row.fact, at,
    v: 2, updatedAt: at, lastUsedAt: 0, useCount: 0,
    kind: slot ? 'profile' : 'fact', slot, keys: keysFor(row.fact, slot),
    pinned: false, importance: 3, scope: 'all',
    status: lintFact(row.fact).ok ? 'active' : 'review',
    eventAt: null, expiresAt: null, validTo: null,
    supersedes: null, supersededBy: null, possibleDup: null,
    lang: detectLang(row.fact), hash,
    source: { app: 'legacy', how: 'legacy', conv: null },
  };
  if (row.fact.length > LIMITS.factMax) rec.tooLong = true;
  return rec;
}

// ── Consolidation decision (pure) ────────────────────────────────────────────

// `cand`: { fact, hash, slot, scope, how }. `active`: active records. `write`:
// the store-wide thresholds { dupSim, similarMin }.
//   -> { action: 'duplicate' | 'updated' | 'negate' | 'similar' | 'add', target? }
export function decideWrite(cand, active, write) {
  const same = active.filter((r) => r.scope === cand.scope && r.status === 'active');
  const negated = isNegated(cand.fact);
  const byUser = cand.how === 'user-asked' || cand.how === 'user-confirmed' || cand.how === 'user-typed';
  if (negated && byUser) {
    let target = cand.slot ? same.find((r) => r.slot === cand.slot) : null;
    if (target && isExclusive(cand.slot)) {
      // Only when the negated value is the stored one («ya no vivo en Sevilla»).
      const v = slotValueTerms(cand.fact, cand.slot), w = slotValueTerms(target.fact, target.slot);
      if (v.length && !v.some((x) => w.some((y) => prefixMatch(x, y)))) target = null;
    }
    if (!target) {
      let best = null, bs = 0;
      for (const r of same) { const s = similarity(cand.fact, r.fact); if (s > bs) { bs = s; best = r; } }
      if (best && bs >= write.similarMin) target = best;
    }
    if (target) return { action: 'negate', target };
  }
  const dup = same.find((r) => r.hash && r.hash === cand.hash);
  if (dup) return { action: 'duplicate', target: dup };
  if (cand.slot && isExclusive(cand.slot)) {
    const cur = same.find((r) => r.slot === cand.slot);
    if (cur) {
      const v = slotValueTerms(cand.fact, cand.slot), w = slotValueTerms(cur.fact, cur.slot);
      const overlap = v.filter((x) => w.some((y) => prefixMatch(x, y))).length;
      const sameValue = v.length && w.length && overlap / Math.max(v.length, w.length) >= 0.5;
      return sameValue ? { action: 'duplicate', target: cur } : { action: 'updated', target: cur };
    }
  }
  let best = null, bs = 0;
  for (const r of same) { const s = similarity(cand.fact, r.fact); if (s > bs) { bs = s; best = r; } }
  if (best && bs >= write.dupSim) {
    // Two different values for a non-exclusive slot (two allergies) are not
    // duplicates even if the words overlap.
    if (!(cand.slot && best.slot && cand.slot !== best.slot)) return { action: 'duplicate', target: best };
  }
  if (best && bs >= write.similarMin && !cand.slot) return { action: 'similar', target: best };
  return { action: 'add' };
}

// ── The context block ────────────────────────────────────────────────────────

// Every fact goes into the block as ONE line of inert text: line breaks
// become spaces and a '<' that could start a tag (the block's fence in any
// brand's wording, role tags) becomes '‹'. Legacy facts are kept byte for byte
// in the store; only their rendering is neutralised.
export const inertFact = (fact) => String(fact ?? '')
  .replace(/[\r\n\u2028\u2029\u0085\v\f]+/g, ' ')
  .replace(/<(?=\s*[\/!?|\p{L}])/gu, '‹');
const lineOf = (r, { format, strings, lang }) => {
  const day = formatWhen(r.eventAt ?? r.updatedAt ?? r.at, lang, strings);
  const f = inertFact(r.fact);
  if (format === 'list-plain') return `- ${f}`;
  if (format === 'kv') return `* ${f} (${day})`;
  return r.kind === 'event' && r.eventAt ? `- ${t(strings, lang, 'eventLabel')} ${day} · ${f}` : `- ${day} · ${f}`;
};
const coreText = (r, { strings, lang, format }) =>
  r.supersedes && format !== 'list-plain' ? `${inertFact(r.fact)} (${t(strings, lang, 'since', { date: formatWhen(r.updatedAt, lang, strings) })})` : inertFact(r.fact);

// Builds the block. `core` and `archive` are records already chosen; this
// function only renders and fits them into `budget` ({ total, core }).
//   -> { text, tokens, ids, coreLeft }
export function renderBlock({ core = [], archive = [], strings = null, lang = 'es', wording = 'v1', format = 'list-dated', budget, toolList = 'memory.list', count = estimateTokens }) {
  const tag = t(strings, lang, 'blockTag');
  const head = `<${tag}> ${t(strings, lang, `blockHead.${wording}`)}`;
  const foot = `${t(strings, lang, `blockAbstain.${wording}`)}\n</${tag}>`;
  let used = count(head + '\n') + count(foot + '\n');
  const ids = [];
  const lines = [head];

  // Core: profile first, then preferences; cut at budget.core.
  const prof = core.filter((r) => r.kind !== 'preference');
  const prefs = core.filter((r) => r.kind === 'preference');
  let coreUsed = 0, coreLeft = 0;
  const pickCore = (list, label) => {
    const kept = [];
    for (const r of list) {
      const piece = coreText(r, { strings, lang, format });
      const c = count(piece + ' · ');
      if (coreUsed + c > budget.core) { coreLeft++; continue; }
      coreUsed += c; kept.push(r);
    }
    if (!kept.length) return null;
    kept.forEach((r) => ids.push(r.id));
    return `${t(strings, lang, label)}: ${kept.map((r) => coreText(r, { strings, lang, format })).join(' · ')}`;
  };
  const a = pickCore(prof, 'aboutLabel');
  const p = pickCore(prefs, 'prefsLabel');
  if (a) { lines.push(a); used += count(a + '\n'); }
  if (p) { lines.push(p); used += count(p + '\n'); }
  if (coreLeft) {
    const more = t(strings, lang, 'moreHint', { n: coreLeft, tool: toolList });
    lines.push(more); used += count(more + '\n');
  }
  const fit = fitBlock(archive.map((r) => lineOf(r, { format, strings, lang })), Math.max(0, budget.total - used), count);
  // fitBlock keeps order and skips what does not fit; map kept lines back to ids.
  let k = 0;
  for (const r of archive) {
    if (k < fit.lines.length && fit.lines[k] === lineOf(r, { format, strings, lang })) { ids.push(r.id); k++; }
  }
  lines.push(...fit.lines);
  used += fit.tokens;
  lines.push(foot);
  const text = lines.join('\n');
  return { text, tokens: count(text), ids, coreLeft };
}

// The rule lines for the system head: stable bytes, no facts.
export function renderRules(strings, lang, variant = 'v1', prefix = 'memory.') {
  const tag = t(strings, lang, 'blockTag');
  const lines = raw(strings, lang, `rules.${variant}`) || raw(strings, lang, 'rules.v1');
  return lines.map((l) => fill(l, { save: `${prefix}save`, forget: `${prefix}forget`, list: `${prefix}list`, tag }));
}

export { STOPWORDS };
