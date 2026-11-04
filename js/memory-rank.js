// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
//
// User memory: text matching, dates and retrieval scoring. Pure functions, no
// storage, no DOM, no imports.
//
// Unicode-aware throughout: a term is a run of letters and digits in ANY
// script (`\p{L}\p{N}`), accents are folded away for matching, and the stop
// lists are only the function words of a few languages. An ASCII tokenizer
// would silently drop every name written with «ñ», accents, Cyrillic or CJK.
//
// Every number that changes how retrieval behaves (BM25 constants, weights,
// half-life, abstain threshold, budget) comes in as an argument from the
// model's tuning file; nothing here is a hidden constant.

// ── Text ─────────────────────────────────────────────────────────────────────

export function fold(s) {
  return String(s ?? '').normalize('NFC').toLowerCase().normalize('NFD').replace(/\p{M}/gu, '').normalize('NFC');
}

const TOKEN = /[\p{L}\p{N}]+(?:[-'’][\p{L}\p{N}]+)*/gu;

// Folded terms. A compound («e-mail», "o'brien") is kept whole AND split, so
// either form matches.
export function terms(s) {
  const out = [];
  for (const m of fold(s).matchAll(TOKEN)) {
    const w = m[0];
    out.push(w);
    if (/[-'’]/.test(w)) for (const p of w.split(/[-'’]/)) if (p) out.push(p);
  }
  return out;
}

const SW = (s) => s.split(/\s+/).filter(Boolean).map(fold);
export const STOPWORDS = new Set([
  // Spanish
  ...SW(`a al algo algun alguna algunas alguno algunos ante antes asi aun aunque bajo bien cada como con contra cual cuales
    cuando cuanto de del desde donde dos e el ella ellas ello ellos en entre era eran es esa esas ese eso esos esta estaba
    estan estar este esto estos fue fueron ha habia han hasta hay la las le les lo los mas me mi mia mis mucho muy nada
    ni no nos nosotros o os otra otro para pero poco por porque que quien se sea segun ser si sin sobre son su sus tal
    tambien te tengo tiene tienen ti tu tus un una uno unos unas y ya yo vosotros usted ustedes sera seria esto aqui alli
    dime dimelo sabes recuerdas acuerdas cosa cosas eh oye vale pues entonces luego tan tanto hacer hago hace quiero puedes
    puedo qué cómo cuál cuándo dónde quién cuentame cuéntame explicame explícame muestrame muéstrame enseñame enséñame
    bastante ahora todavia todavía tambien realmente`),
  // English
  ...SW(`a an and are as at be been being but by can could did do does doing for from had has have having he her hers him
    his how i if in into is it its just me more most my no nor not of on once only or other our ours out over own same she
    should so some such than that the their theirs them then there these they this those through to too under until up
    very was we were what when where which while who whom why will with would you your yours yourself am also any
    tell remember know about again please hey ok okay thing things do does did show explain give
    really very quite truly actually now still much lot lots
    what's where's who's how's when's that's it's there's i'm i've i'd i'll you're isn't doesn't don't can't won't didn't
    aren't wasn't weren't haven't hasn't let's`),
  // A few function words of other common languages, so a French, Portuguese,
  // Italian or German query does not count «le», «de», «der» as content.
  ...SW(`le les des du un une et est dans pour pas que qui sur au aux ce cette je tu il elle nous vous ils
    o os as um uma e do da dos das em no na nos nas por para com nao sim
    il lo gli la le di da in con su per tra fra che non e un una
    der die das den dem des ein eine einer und ist nicht zu mit von im am auf fur ich du er sie es wir ihr`),
]);

// Words that never ground a fact: a third-person rewrite («Al usuario le
// gusta...», "The user prefers...") must ground on the rest of the words.
export const GROUNDING_STOP = new Set(SW('usuario usuaria user users el la the persona'));

// Content terms: no stopwords, no single letters ("s" of "what's"), numbers
// kept unless asked otherwise.
export function contentTerms(s, { keepNumbers = true } = {}) {
  return terms(s).filter((w) => !STOPWORDS.has(w) && (w.length > 1 || /^\p{N}$/u.test(w)) &&
    (keepNumbers || !/^\p{N}+$/u.test(w)));
}

// Two folded terms match if equal; or if the shorter has at least 4 letters and
// is a prefix of the other ("live"/"lives"); or if they share a prefix of 5 or
// more and neither runs more than two letters past it («trabajo»/«trabaja»).
export function prefixMatch(a, b) {
  if (a === b) return true;
  const [s, l] = a.length <= b.length ? [a, b] : [b, a];
  if (s.length >= 4 && l.startsWith(s)) return true;
  let i = 0;
  while (i < s.length && s[i] === l[i]) i++;
  return i >= 5 && s.length - i <= 2 && l.length - i <= 2;
}

// Looser, for retrieval only: also the 1st / 3rd person of the same verb
// («toco» / «toca», «vivo» / «vive»), same length and stem, both ending in a
// vowel. Never used to decide that two facts are duplicates.
export function stemMatch(a, b) {
  if (prefixMatch(a, b)) return true;
  return a.length >= 4 && a.length === b.length && a.slice(0, -1) === b.slice(0, -1) && /[aeiou]$/.test(a) && /[aeiou]$/.test(b);
}

export function estimateTokens(s) {
  return Math.ceil(String(s ?? '').length / 3);
}

// ── Dates ────────────────────────────────────────────────────────────────────

const DAY = 864e5;
const startOfDay = (ms) => { const d = new Date(ms); d.setHours(0, 0, 0, 0); return d.getTime(); };
const addDays = (ms, n) => { const d = new Date(ms); d.setDate(d.getDate() + n); return d.getTime(); };
const endOfDay = (ms) => addDays(startOfDay(ms), 1) - 1;
const monthRange = (y, m) => [new Date(y, m, 1).getTime(), new Date(y, m + 1, 1).getTime() - 1];

const MONTHS = {
  enero: 0, febrero: 1, marzo: 2, abril: 3, mayo: 4, junio: 5, julio: 6, agosto: 7, septiembre: 8, setiembre: 8,
  octubre: 9, noviembre: 10, diciembre: 11,
  january: 0, february: 1, march: 2, april: 3, may: 4, june: 5, july: 6, august: 7, september: 8, october: 9,
  november: 10, december: 11, jan: 0, feb: 1, mar: 2, apr: 3, jun: 5, jul: 6, aug: 7, sep: 8, sept: 8, oct: 9, nov: 10, dec: 11,
};
const WEEKDAYS = {
  domingo: 0, lunes: 1, martes: 2, miercoles: 3, jueves: 4, viernes: 5, sabado: 6,
  sunday: 0, monday: 1, tuesday: 2, wednesday: 3, thursday: 4, friday: 5, saturday: 6,
};
const NUMWORDS = {
  un: 1, una: 1, uno: 1, dos: 2, tres: 3, cuatro: 4, cinco: 5, seis: 6, siete: 7, ocho: 8, nueve: 9, diez: 10,
  quince: 15, veinte: 20, treinta: 30,
  a: 1, an: 1, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
  fifteen: 15, twenty: 20, thirty: 30,
};
const num = (w) => (/^\d+$/.test(w) ? +w : NUMWORDS[fold(w)] ?? null);

const MONTH_RE = '(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre|january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)';
const WD_RE = '(lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo|monday|tuesday|wednesday|thursday|friday|saturday|sunday)';
const NUM_RE = '(\\d{1,3}|un|una|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|quince|veinte|treinta|a|an|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|thirty)';

// The nearest occurrence of day/month to `now`, within half a year either way.
function nearestDate(day, month, now, year = null) {
  if (year != null) return new Date(year, month, day).getTime();
  const y = new Date(now).getFullYear();
  const cands = [y - 1, y, y + 1].map((yy) => new Date(yy, month, day).getTime());
  const today = startOfDay(now);
  // Prefer the future one when it is close: «el 4 de enero» said in November
  // means the coming January.
  let best = cands[0];
  for (const c of cands) if (Math.abs(c - today) < Math.abs(best - today)) best = c;
  return best;
}

// Date rules, tried in order on lower-cased text. Each returns [start, end, kind]
// where kind is 'day', 'week', 'month' or 'range'; `rel` says whether the
// expression is relative to today (and so must be resolved when saving).
const RULES = [
  // An ISO date swallows the weekday written after it (the form saved facts
  // use), so normalising a saved fact again never moves its date.
  { re: /\b(\d{4})-(\d{2})-(\d{2})\b(?:,?\s*\(?(?:lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b\)?)?/g, rel: false, at: (m) => { const d = new Date(+m[1], +m[2] - 1, +m[3]).getTime(); return [d, endOfDay(d), 'day']; } },
  { re: /\bpasado ma[ñn]ana\b|\bday after tomorrow\b/g, rel: true, at: (m, now) => { const d = addDays(startOfDay(now), 2); return [d, endOfDay(d), 'day']; } },
  { re: /(?<!\bla\s)(?<!\bpor la\s)(?<!\bde la\s)(?<!\bpasado\s)\bma[ñn]ana\b|\btomorrow\b/g, rel: true, at: (m, now) => { const d = addDays(startOfDay(now), 1); return [d, endOfDay(d), 'day']; } },
  { re: /\banteayer\b/g, rel: true, at: (m, now) => { const d = addDays(startOfDay(now), -2); return [d, endOfDay(d), 'day']; } },
  { re: /\bhoy\b|\btoday\b|\btonight\b|\besta noche\b/g, rel: true, at: (m, now) => { const d = startOfDay(now); return [d, endOfDay(d), 'day']; } },
  { re: /\bayer\b|\byesterday\b/g, rel: true, at: (m, now) => { const d = addDays(startOfDay(now), -1); return [d, endOfDay(d), 'day']; } },
  { re: new RegExp(`\\b(?:en|dentro de|in)\\s+${NUM_RE}\\s+(d[ií]as?|days?|semanas?|weeks?)\\b`, 'g'), rel: true, at: (m, now) => {
    const n = num(m[1]); if (n == null) return null;
    const d = addDays(startOfDay(now), /^(sem|week)/.test(m[2]) ? 7 * n : n); return [d, endOfDay(d), 'day'];
  } },
  { re: /\b(?:la\s+)?semana que viene\b|\bla pr[oó]xima semana\b|\bnext week\b/g, rel: true, at: (m, now) => {
    const t = startOfDay(now), wd = (new Date(t).getDay() + 6) % 7; const mon = addDays(t, 7 - wd); return [mon, endOfDay(addDays(mon, 6)), 'week'];
  } },
  { re: /\bla semana pasada\b|\blast week\b/g, rel: true, at: (m, now) => {
    const t = startOfDay(now), wd = (new Date(t).getDay() + 6) % 7; const mon = addDays(t, -wd - 7); return [mon, endOfDay(addDays(mon, 6)), 'week'];
  } },
  { re: /\besta semana\b|\bthis week\b/g, rel: true, at: (m, now) => {
    const t = startOfDay(now), wd = (new Date(t).getDay() + 6) % 7; const mon = addDays(t, -wd); return [mon, endOfDay(addDays(mon, 6)), 'week'];
  } },
  { re: /\b(?:este|el) fin de semana\b|\bthis weekend\b|\b(?:at|on) the weekend\b/g, rel: true, at: (m, now) => {
    const t = startOfDay(now), wd = new Date(t).getDay(); const sat = addDays(t, wd === 0 ? -1 : 6 - wd); return [sat, endOfDay(addDays(sat, 1)), 'range'];
  } },
  { re: /\bel mes que viene\b|\bel pr[oó]ximo mes\b|\bnext month\b/g, rel: true, at: (m, now) => { const d = new Date(now); return [...monthRange(d.getFullYear(), d.getMonth() + 1), 'month']; } },
  { re: /\bel mes pasado\b|\blast month\b/g, rel: true, at: (m, now) => { const d = new Date(now); return [...monthRange(d.getFullYear(), d.getMonth() - 1), 'month']; } },
  { re: /\beste mes\b|\bthis month\b/g, rel: true, at: (m, now) => { const d = new Date(now); return [...monthRange(d.getFullYear(), d.getMonth()), 'month']; } },
  { re: new RegExp(`\\b(?:el\\s+)?(\\d{1,2})\\s+de\\s+${MONTH_RE}(?:\\s+(?:de|del)\\s+(\\d{4}))?\\b`, 'g'), rel: false, at: (m, now) => {
    const d = nearestDate(+m[1], MONTHS[m[2]], now, m[3] ? +m[3] : null); return [d, endOfDay(d), 'day'];
  } },
  { re: new RegExp(`\\b${MONTH_RE}\\s+(\\d{1,2})(?:st|nd|rd|th)?(?:,?\\s+(\\d{4}))?\\b`, 'g'), rel: false, at: (m, now) => {
    const d = nearestDate(+m[2], MONTHS[m[1]], now, m[3] ? +m[3] : null); return [d, endOfDay(d), 'day'];
  } },
  { re: new RegExp(`\\b(\\d{1,2})(?:st|nd|rd|th)?\\s+(?:of\\s+)?${MONTH_RE}(?:,?\\s+(\\d{4}))?\\b`, 'g'), rel: false, at: (m, now) => {
    const d = nearestDate(+m[1], MONTHS[m[2]], now, m[3] ? +m[3] : null); return [d, endOfDay(d), 'day'];
  } },
  { re: new RegExp(`\\b(?:en|in)\\s+${MONTH_RE}\\b`, 'g'), rel: false, at: (m, now) => {
    const mo = MONTHS[m[1]]; const s = nearestDate(1, mo, now); const d = new Date(s); return [...monthRange(d.getFullYear(), d.getMonth()), 'month'];
  } },
  // «los lunes» and "every Monday" are habits, not dates.
  { re: new RegExp(`(?<!\\blos\\s)(?<!\\bevery\\s)\\b(?:(?:el|este|esta|next|this|on|coming)\\s+)?(?:pr[oó]ximo\\s+)?${WD_RE}(?:\\s+que viene)?\\b`, 'g'), rel: true, at: (m, now) => {
    const target = WEEKDAYS[fold(m[1])]; const t = startOfDay(now); const wd = new Date(t).getDay();
    const ahead = ((target - wd + 7) % 7) || 7; const d = addDays(t, ahead); return [d, endOfDay(d), 'day'];
  } },
  { re: /\b(\d{1,2})\/(\d{1,2})(?:\/(\d{2,4}))?\b/g, rel: false, at: (m, now, lang) => {
    let a = +m[1], b = +m[2];
    let day = a, mon = b;
    if (lang === 'en' && a <= 12 && b <= 31) { day = b; mon = a; }
    if (mon > 12 && day <= 12) [day, mon] = [mon, day];
    if (mon < 1 || mon > 12 || day < 1 || day > 31) return null;
    const y = m[3] ? (m[3].length === 2 ? 2000 + +m[3] : +m[3]) : null;
    const d = nearestDate(day, mon - 1, now, y); return [d, endOfDay(d), 'day'];
  } },
];

// Every date expression in `text`, left to right, without overlaps:
// [{ index, length, text, start, end, kind, rel }].
export function findDates(text, now = Date.now(), lang = 'es') {
  const low = String(text ?? '').toLowerCase();
  const found = [];
  for (const r of RULES) {
    r.re.lastIndex = 0;
    for (const m of low.matchAll(r.re)) {
      const v = r.at(m, now, lang);
      if (!v) continue;
      const span = { index: m.index, length: m[0].length, text: String(text).slice(m.index, m.index + m[0].length), start: v[0], end: v[1], kind: v[2], rel: r.rel };
      if (found.some((f) => span.index < f.index + f.length && f.index < span.index + span.length)) continue;
      found.push(span);
    }
  }
  return found.sort((a, b) => a.index - b.index);
}

// The date range a text talks about, or null: { start, end, kind, rel, past }.
export function parseWhen(text, now = Date.now(), lang = 'es') {
  const all = findDates(text, now, lang);
  if (!all.length) return null;
  const start = Math.min(...all.map((d) => d.start));
  const end = Math.max(...all.map((d) => d.end));
  return { start, end, kind: all.length === 1 ? all[0].kind : 'range', rel: all.some((d) => d.rel), past: end < startOfDay(now) };
}

export { startOfDay, endOfDay, addDays, DAY };

// ── Slot intent ──────────────────────────────────────────────────────────────

// Questions that ask for a known slot. Matched on folded text, so accents in
// the patterns are optional by construction.
const INTENTS = [
  ['name', /\bcomo me llamo\b|\bcual es mi nombre\b|\bmi nombre\b|\bwhat(?:'s| is) my name\b|\bwho am i\b|\bmy name\b/],
  ['home_city', /\bdonde vivo\b|\ben que ciudad vivo\b|\bdonde esta mi casa\b|\bde donde soy\b|\bmi ciudad\b|\bwhere do i live\b|\bwhich city do i live\b|\bwhere(?:'s| is) my home\b|\bmy city\b|\bwhere am i from\b/],
  ['employer', /\bdonde trabajo\b|\ben que empresa\b|\bmi empresa\b|\bwhere do i work\b|\bmy company\b|\bmy employer\b/],
  ['job', /\bde que trabajo\b|\ben que trabajo\b|\ba que me dedico\b|\bmi trabajo\b|\bmi oficio\b|\bwhat(?:'s| is) my job\b|\bwhat do i do for a living\b|\bmy job\b/],
  ['birthday', /\bcuando es mi cumple|\bmi cumpleanos\b|\bwhen(?:'s| is) my birthday\b|\bmy birthday\b/],
  ['reply_language', /\ben que idioma\b|\bque idioma\b|\bwhich language\b|\bwhat language\b/],
  ['answer_style', /\bcomo (?:prefiero|me gustan) las respuestas\b|\bhow do i like (?:my )?answers\b/],
  ['diet', /\bmi dieta\b|\bque como\b|\bque no como\b|\bwhat(?:'s| is) my diet\b|\bwhat do i eat\b|\bwhat can'?t i eat\b/],
  ['allergy:*', /\balergic|\balergia|\ballerg/],
  ['stack', /\bque (?:lenguaje|stack|herramientas) uso\b|\bwhat (?:language|stack|tools) do i use\b/],
];
const ROLES = {
  hija: 'daughter', hijo: 'son', pareja: 'partner', mujer: 'partner', marido: 'partner', novia: 'partner', novio: 'partner',
  perro: 'dog', perra: 'dog', gato: 'cat', gata: 'cat', madre: 'mother', padre: 'father', hermana: 'sister', hermano: 'brother',
  daughter: 'daughter', son: 'son', partner: 'partner', wife: 'partner', husband: 'partner', girlfriend: 'partner', boyfriend: 'partner',
  dog: 'dog', cat: 'cat', mother: 'mother', father: 'father', mum: 'mother', mom: 'mother', dad: 'father', sister: 'sister', brother: 'brother',
};
export { ROLES };

export function slotIntent(query) {
  const q = fold(query);
  const out = [];
  for (const [slot, re] of INTENTS) if (re.test(q)) out.push(slot);
  const rel = /\bcomo se llama mi (\p{L}+)|\bnombre de mi (\p{L}+)|\bwhat(?:'s| is) my (\p{L}+?)(?:'s)? name\b|\bmy (\p{L}+)'s name\b/u.exec(q);
  if (rel) {
    const role = ROLES[rel[1] || rel[2] || rel[3] || rel[4]];
    if (role) out.push(`rel:${role}`);
  }
  return out;
}

export function slotMatches(intent, slot) {
  if (!slot) return false;
  if (intent === slot) return true;
  if (intent.endsWith(':*')) return slot.startsWith(intent.slice(0, -1));
  return false;
}

// ── BM25 with prefix matching ────────────────────────────────────────────────

// docs: [{ id, text, keys: string[] }]. `keys` are repeated `keyBoost` times so
// slot vocabulary counts more than a passing word.
export function buildBM25(docs, { k1, b, keyBoost }) {
  const rows = docs.map((d) => {
    const tf = new Map();
    const add = (w, n) => tf.set(w, (tf.get(w) || 0) + n);
    for (const w of contentTerms(d.text)) add(w, 1);
    for (const k of d.keys || []) for (const w of contentTerms(k)) add(w, keyBoost);
    let len = 0;
    for (const n of tf.values()) len += n;
    return { id: d.id, tf, len };
  });
  const vocab = new Set();
  for (const r of rows) for (const w of r.tf.keys()) vocab.add(w);
  const N = rows.length;
  const avgdl = N ? rows.reduce((s, r) => s + r.len, 0) / N : 1;
  const expandCache = new Map();
  const expand = (q) => {
    if (expandCache.has(q)) return expandCache.get(q);
    const m = [];
    for (const w of vocab) if (stemMatch(q, w)) m.push(w);
    expandCache.set(q, m);
    return m;
  };
  const dfCache = new Map();
  const df = (q) => {
    if (dfCache.has(q)) return dfCache.get(q);
    const m = expand(q);
    let n = 0;
    for (const r of rows) if (m.some((w) => r.tf.has(w))) n++;
    dfCache.set(q, n);
    return n;
  };
  const idf = (q) => { const n = df(q); return Math.log(1 + (N - n + 0.5) / (n + 0.5)); };
  const tfOf = (row, q) => { let s = 0; for (const w of expand(q)) s += row.tf.get(w) || 0; return s; };
  return {
    N, rows, idf, expand,
    // q: Map term -> weight. Returns one score per row.
    score(q) {
      return rows.map((r) => {
        let s = 0;
        for (const [term, wq] of q) {
          const f = tfOf(r, term);
          if (!f) continue;
          s += wq * idf(term) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * r.len / avgdl));
        }
        return s;
      });
    },
    // The IDF-weighted share of the query's content terms this row matches.
    // Independent of how many records the store holds, unlike a raw BM25
    // threshold, which is why the abstain gate uses it.
    coverage(q, i) {
      let all = 0, hit = 0;
      for (const [term, wq] of q) {
        const w = wq * idf(term);
        all += w;
        if (tfOf(rows[i], term)) hit += w;
      }
      return all ? hit / all : 0;
    },
  };
}

// Query terms with weights: the latest typed message at 1, the previous one at
// `prevWeight` (for follow-ups such as «¿y a qué hora?»).
export function queryTerms(query, prevQuery, prevWeight) {
  const q = new Map();
  for (const w of contentTerms(query)) q.set(w, 1);
  if (prevQuery && prevWeight > 0) for (const w of contentTerms(prevQuery)) if (!q.has(w)) q.set(w, prevWeight);
  return q;
}

export function coverage(index, q, i) { return index.coverage(q, i); }

// Score candidate records for one query. Returns the items sorted by score and
// the signal the abstain gate needs.
export function scoreRecords(records, { query, prevQuery = '', now = Date.now(), rank, lang = 'es', index = null }) {
  const idx = index || buildBM25(records.map((r) => ({ id: r.id, text: r.fact, keys: r.keys })), rank);
  // Date words («mañana», "next week") are matched as dates (tmp), never as
  // words: «mañana» must not pull in every fact that says «por las mañanas».
  const noDates = (t) => { let x = String(t || ''); for (const d of findDates(x, now, lang).reverse()) x = x.slice(0, d.index) + ' ' + x.slice(d.index + d.length); return x; };
  const q = queryTerms(noDates(query), noDates(prevQuery), rank.prevWeight);
  const raw = q.size ? idx.score(q) : records.map(() => 0);
  const best = Math.max(0, ...raw);
  const intents = slotIntent(query).concat(prevQuery ? slotIntent(prevQuery) : []);
  const when = parseWhen(query, now, lang);
  let maxCov = 0, slotHit = false, dateHit = false;
  const items = records.map((r, i) => {
    const cov = q.size ? idx.coverage(q, i) : 0;
    maxCov = Math.max(maxCov, cov);
    const slotRel = intents.some((it) => slotMatches(it, r.slot));
    if (slotRel) slotHit = true;
    const rel = slotRel ? 1 : best > 0 ? raw[i] / best : 0;
    const ref = Math.max(r.lastUsedAt || 0, r.updatedAt || r.at || 0);
    const days = Math.max(0, (now - ref) / DAY);
    const rec = Math.pow(2, -days / rank.halfLifeDays);
    const imp = ((r.importance || 3) - 1) / 4;
    const at = r.eventAt ?? r.at;
    const tmp = when && at >= when.start && at <= when.end ? 1 : 0;
    if (tmp) dateHit = true;
    const score = rank.wRel * rel + rank.wRec * rec + rank.wImp * imp + rank.wTmp * tmp;
    return { rec: r, score, rel, cov, tmp, slotRel };
  });
  items.sort((a, b) => b.score - a.score || (b.rec.updatedAt || 0) - (a.rec.updatedAt || 0));
  return { items, signal: { maxCov, slotHit, dateHit, intents, when }, index: idx };
}

// ── Budget ───────────────────────────────────────────────────────────────────

// The memory block budget in tokens, from the registry `ctx` of the selected
// model and its tuning (`tuningFor(id, ctx).memory.budget`). `ctx` only sizes
// the block; it is never a stand-in for how capable the model is.
export function memoryBudget(ctx, b) {
  const total = ctx ? Math.min(b.max, Math.max(b.min, Math.floor(ctx * b.share))) : b.noCtx;
  return {
    total,
    core: Math.min(b.coreMax, Math.floor(total * b.coreShare)),
    maxArchive: ctx && ctx <= b.smallCtx ? b.maxItemsSmall : b.maxItems,
  };
}

// Greedy fit of lines into a token budget. Returns the lines that fit, in
// order, and how many were left out.
export function fitBlock(lines, budget, count = estimateTokens) {
  const out = [];
  let used = 0, left = 0;
  for (const l of lines) {
    const c = count(l + '\n');
    if (used + c <= budget) { out.push(l); used += c; } else left++;
  }
  return { lines: out, tokens: used, left };
}
