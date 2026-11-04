// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
//
// The one config object that makes the shared memory any brand's memory:
// brand name, storage namespace, store page URL, the exact origins of the
// apps allowed to use it, strings in any language, theme tokens and the tools
// the model sees. The same object goes to the store page (`mountStorePage`)
// and to every app (`createMemory`).
//
// `defineMemoryConfig` validates everything at once and throws one error that
// lists every problem, so nothing half-works. What a config can NOT change is
// listed in `FLOORS`: there is simply no key for it, and unknown keys throw.

import { STRINGS, REQUIRED_KEYS } from './memory-text.js';
import { LIMITS, SUPPORT_FLOOR } from './memory-core.js';

export class MemoryConfigError extends Error {
  constructor(problems) {
    super('invalid memory config:\n  - ' + problems.join('\n  - '));
    this.name = 'MemoryConfigError';
    this.problems = problems;
  }
}

// What no config can change. Exported so adopters can see it.
export const FLOORS = Object.freeze({
  supportMin: SUPPORT_FLOOR,          // share of a fact's words the user must have typed
  onlyTypedTextGrounds: true,         // pasted, uploaded, tool, web and assistant text never grounds a fact
  taintFailsClosed: true,             // any non-memory tool result in the turn makes saves proposals at most
  lint: true,                         // instruction-shaped text is never stored
  forgetNeedsClick: true,             // a forget the model starts always waits for the user's click
  bulkOpsManagerOnly: true,           // export, import, erase, settings, revoke: only on the top-level manager page
  exactSameSiteOrigins: true,         // https, exact origins, same registrable domain as the store
  connectSrcNone: true,               // the store page cannot open any network connection
  limitsLowerOnly: true,              // `limits` may lower a limit, never raise it
});

// Multi-part public suffixes (and hosting suffixes where every subdomain is a
// different site). Enough for the common cases; extend it when a brand lives
// under another one.
export const PUBLIC_SUFFIXES = new Set([
  'co.uk', 'org.uk', 'ac.uk', 'gov.uk', 'me.uk', 'ltd.uk', 'plc.uk', 'net.uk',
  'com.es', 'org.es', 'nom.es', 'gob.es', 'edu.es',
  'com.au', 'net.au', 'org.au', 'edu.au', 'gov.au',
  'co.jp', 'ne.jp', 'or.jp', 'ac.jp',
  'com.br', 'net.br', 'org.br', 'com.mx', 'org.mx', 'gob.mx', 'com.ar', 'com.co', 'com.pe', 'cl',
  'co.nz', 'org.nz', 'co.in', 'net.in', 'org.in', 'co.za', 'com.tr', 'com.cn', 'com.tw', 'com.hk', 'co.kr',
  'github.io', 'gitlab.io', 'pages.dev', 'workers.dev', 'netlify.app', 'vercel.app', 'web.app',
  'firebaseapp.com', 'herokuapp.com', 'appspot.com', 'glitch.me', 'onrender.com', 'fly.dev',
]);

// Second-level labels that country registries use as public suffixes.
const GENERIC_SLD = new Set(['com', 'co', 'net', 'org', 'gov', 'gob', 'edu', 'ac', 'nom', 'ltd', 'plc', 'or', 'ne', 'go', 'mil', 'sch', 'biz', 'info']);

const isLocalHost = (h) => h === 'localhost' || h === '127.0.0.1' || h === '[::1]';

export function registrableDomain(hostname) {
  const h = String(hostname).toLowerCase().replace(/\.$/, '');
  if (isLocalHost(h)) return h;
  if (/^\d+\.\d+\.\d+\.\d+$/.test(h) || h.includes(':')) return h;
  const parts = h.split('.');
  for (let i = 0; i < parts.length; i++) {
    const suffix = parts.slice(i).join('.');
    if (PUBLIC_SUFFIXES.has(suffix)) return i > 0 ? parts.slice(i - 1).join('.') : null;
  }
  // Not in the list: fail closed for the usual shape of a multi-part suffix,
  // a generic second level under a two-letter country code (com.sg, co.il,
  // org.my...). Treating it as a suffix can only split one site in two, which
  // the config then refuses; the other way round would accept a cross-site
  // config and silently split the memory.
  if (parts.length >= 3 && /^[a-z]{2}$/.test(parts[parts.length - 1]) && GENERIC_SLD.has(parts[parts.length - 2])) return parts.slice(-3).join('.');
  if (parts.length === 2 && /^[a-z]{2}$/.test(parts[1]) && GENERIC_SLD.has(parts[0])) return null;
  return parts.length >= 2 ? parts.slice(-2).join('.') : null;
}

// Schemeful same-site: same scheme and same registrable domain.
export function sameSite(a, b) {
  try {
    const A = new URL(a), B = new URL(b);
    if (A.protocol !== B.protocol) return false;
    const ra = registrableDomain(A.hostname), rb = registrableDomain(B.hostname);
    return !!ra && ra === rb;
  } catch { return false; }
}

const KNOWN_STRING_KEYS = new Set(Object.keys(STRINGS.en));
const ARRAY_KEYS = new Set(['weekdays', 'triggers', 'patterns', ...Object.keys(STRINGS.en).filter((k) => Array.isArray(STRINGS.en[k]))]);

export function validateStrings(strings, problems = []) {
  if (strings === undefined) return problems;
  if (!strings || typeof strings !== 'object' || Array.isArray(strings)) { problems.push('strings: must be an object of languages'); return problems; }
  for (const [lang, tbl] of Object.entries(strings)) {
    if (!/^[a-z]{2,3}(-[A-Za-z0-9]{2,8})?$/.test(lang)) { problems.push(`strings.${lang}: not a language tag`); continue; }
    if (!tbl || typeof tbl !== 'object' || Array.isArray(tbl)) { problems.push(`strings.${lang}: must be an object`); continue; }
    for (const [k, v] of Object.entries(tbl)) {
      if (!KNOWN_STRING_KEYS.has(k)) { problems.push(`strings.${lang}.${k}: unknown key`); continue; }
      if (ARRAY_KEYS.has(k)) {
        if (!Array.isArray(v) || v.some((x) => typeof x !== 'string')) { problems.push(`strings.${lang}.${k}: must be a list of strings`); continue; }
        if (k === 'triggers' || k === 'patterns') {
          for (const src of v) { try { new RegExp(src, 'iu'); } catch { problems.push(`strings.${lang}.${k}: bad pattern ${JSON.stringify(src)}`); } }
        }
        if (k.startsWith('rules.') && v.length !== STRINGS.en[k].length) problems.push(`strings.${lang}.${k}: must have ${STRINGS.en[k].length} lines`);
        if (k === 'weekdays' && v.length !== 7) problems.push(`strings.${lang}.weekdays: must have 7 names`);
        continue;
      }
      if (typeof v !== 'string') { problems.push(`strings.${lang}.${k}: must be a string`); continue; }
      if (REQUIRED_KEYS.includes(k) && !v.trim()) problems.push(`strings.${lang}.${k}: required and empty`);
      if (k === 'blockTag' && !/^[\p{L}][\p{L}\p{N}_-]{0,23}$/u.test(v)) problems.push(`strings.${lang}.blockTag: one word, letters only`);
    }
    // A language that is neither built in nor complete must still carry the
    // safety lines itself or inherit them; an explicit empty one was caught above.
  }
  return problems;
}

const THEME_KEY = /^--mem-[a-z][a-z0-9-]{0,30}$/;
const THEME_VALUE = /^[#(),.%\w\s-]{1,64}$/;
export function validateTheme(theme, problems = []) {
  if (theme === undefined) return problems;
  if (!theme || typeof theme !== 'object') { problems.push('theme: must be { light, dark? }'); return problems; }
  for (const k of Object.keys(theme)) if (k !== 'light' && k !== 'dark') problems.push(`theme.${k}: only "light" and "dark"`);
  for (const scheme of ['light', 'dark']) {
    const set = theme[scheme];
    if (set === undefined) continue;
    if (!set || typeof set !== 'object') { problems.push(`theme.${scheme}: must be an object`); continue; }
    for (const [k, v] of Object.entries(set)) {
      if (!THEME_KEY.test(k)) problems.push(`theme.${scheme}.${k}: only --mem-* properties`);
      if (typeof v !== 'string' || !THEME_VALUE.test(v) || /url\s*\(|expression|@import|;|[{}<>"'\\]/i.test(v)) problems.push(`theme.${scheme}.${k}: value not allowed`);
    }
  }
  return problems;
}

function checkOrigin(x, where, problems, { storeLocal }) {
  if (typeof x !== 'string') { problems.push(`${where}: must be a string`); return null; }
  let u;
  try { u = new URL(x); } catch { problems.push(`${where}: not a URL`); return null; }
  if (x.includes('*')) { problems.push(`${where}: wildcards are not allowed`); return null; }
  if (u.origin === 'null') { problems.push(`${where}: opaque origin`); return null; }
  if (u.origin !== x) { problems.push(`${where}: must be exactly an origin (${u.origin}), no path, slash or query`); return null; }
  if (u.protocol === 'http:') {
    if (!(storeLocal && isLocalHost(u.hostname))) { problems.push(`${where}: must be https (http only for localhost when the store itself is on localhost)`); return null; }
  } else if (u.protocol !== 'https:') { problems.push(`${where}: must be https`); return null; }
  return u;
}

const TOP_KEYS = new Set(['brand', 'namespace', 'storeUrl', 'apps', 'strings', 'theme', 'tools', 'features', 'limits', 'legacy']);
const TOOLS = ['save', 'list', 'forget'];

function deepFreeze(o) {
  if (o && typeof o === 'object' && !Object.isFrozen(o)) {
    Object.freeze(o);
    for (const v of Object.values(o)) deepFreeze(v);
  }
  return o;
}

export function defineMemoryConfig(cfg) {
  const p = [];
  if (!cfg || typeof cfg !== 'object' || Array.isArray(cfg)) throw new MemoryConfigError(['config must be an object']);
  for (const k of Object.keys(cfg)) if (!TOP_KEYS.has(k)) p.push(`${k}: unknown key (the safety rules are not configurable; see FLOORS)`);

  if (!cfg.brand || typeof cfg.brand.name !== 'string' || !cfg.brand.name.trim() || cfg.brand.name.length > 60) p.push('brand.name: required, 1..60 characters');
  if (typeof cfg.namespace !== 'string' || !/^[a-z][a-z0-9-]{2,40}$/.test(cfg.namespace)) p.push('namespace: required, /^[a-z][a-z0-9-]{2,40}$/');

  let store = null, storeLocal = false;
  if (typeof cfg.storeUrl !== 'string') p.push('storeUrl: required');
  else {
    try {
      store = new URL(cfg.storeUrl);
      storeLocal = isLocalHost(store.hostname);
      if (cfg.storeUrl.includes('*')) p.push('storeUrl: wildcards are not allowed');
      if (store.protocol !== 'https:' && !(store.protocol === 'http:' && storeLocal)) p.push('storeUrl: must be https');
      if (store.search || store.hash) p.push('storeUrl: no query or fragment');
      if (store.username || store.password) p.push('storeUrl: no credentials');
    } catch { p.push('storeUrl: not a URL'); store = null; }
  }

  const ids = new Set(), origins = new Set();
  if (!Array.isArray(cfg.apps) || !cfg.apps.length) p.push('apps: required, at least one app');
  else {
    cfg.apps.forEach((a, i) => {
      const w = `apps[${i}]`;
      if (!a || typeof a !== 'object') { p.push(`${w}: must be an object`); return; }
      for (const k of Object.keys(a)) if (!['id', 'origin', 'label'].includes(k)) p.push(`${w}.${k}: unknown key`);
      if (typeof a.id !== 'string' || !/^[a-z][a-z0-9-]{0,23}$/.test(a.id)) p.push(`${w}.id: /^[a-z][a-z0-9-]{0,23}$/`);
      else if (ids.has(a.id) || ['manager', 'import', 'legacy'].includes(a.id)) p.push(`${w}.id: duplicate or reserved (${a.id})`);
      else ids.add(a.id);
      const u = checkOrigin(a.origin, `${w}.origin`, p, { storeLocal });
      if (u) {
        if (origins.has(u.origin)) p.push(`${w}.origin: duplicate`);
        origins.add(u.origin);
        if (store && !sameSite(store.origin, u.origin)) {
          p.push(`${w}.origin: cross-site (${u.origin} vs ${store.origin}): cross-site origins get a separate, partitioned memory; host the store page under the same parent domain`);
        }
      }
      if (!a.label || typeof a.label !== 'object' || !Object.values(a.label).length || Object.values(a.label).some((x) => typeof x !== 'string' || !x.trim())) p.push(`${w}.label: { [lang]: non-empty string }`);
    });
  }

  validateStrings(cfg.strings, p);
  validateTheme(cfg.theme, p);

  let tools = { expose: [...TOOLS], prefix: 'memory.' };
  if (cfg.tools !== undefined) {
    const tl = cfg.tools;
    if (!tl || typeof tl !== 'object') p.push('tools: must be an object');
    else {
      for (const k of Object.keys(tl)) if (!['expose', 'prefix'].includes(k)) p.push(`tools.${k}: unknown key`);
      if (tl.expose !== undefined) {
        if (!Array.isArray(tl.expose) || tl.expose.some((x) => !TOOLS.includes(x))) p.push(`tools.expose: a subset of ${TOOLS.join(', ')}`);
        else tools.expose = [...new Set(tl.expose)];
      }
      if (tl.prefix !== undefined) {
        if (typeof tl.prefix !== 'string' || !/^[a-z][a-z0-9_]{0,22}[._]$/.test(tl.prefix)) p.push('tools.prefix: like "memory." or "mem_"');
        else tools.prefix = tl.prefix;
      }
    }
  }

  const features = { proposals: true, projectScope: [], incognito: true };
  if (cfg.features !== undefined) {
    const f = cfg.features;
    if (!f || typeof f !== 'object') p.push('features: must be an object');
    else {
      for (const k of Object.keys(f)) if (!['proposals', 'projectScope', 'incognito'].includes(k)) p.push(`features.${k}: unknown key`);
      if (f.proposals !== undefined) { if (typeof f.proposals !== 'boolean') p.push('features.proposals: boolean'); else features.proposals = f.proposals; }
      if (f.incognito !== undefined) { if (typeof f.incognito !== 'boolean') p.push('features.incognito: boolean'); else features.incognito = f.incognito; }
      if (f.projectScope !== undefined) {
        if (!Array.isArray(f.projectScope) || f.projectScope.some((x) => !ids.has(x))) p.push('features.projectScope: a list of app ids');
        else features.projectScope = [...f.projectScope];
      }
    }
  }

  const limits = { ...LIMITS };
  if (cfg.limits !== undefined) {
    if (!cfg.limits || typeof cfg.limits !== 'object') p.push('limits: must be an object');
    else for (const [k, v] of Object.entries(cfg.limits)) {
      if (!(k in LIMITS)) { p.push(`limits.${k}: unknown limit`); continue; }
      if (typeof v !== 'number' || !Number.isFinite(v) || v <= 0) { p.push(`limits.${k}: positive number`); continue; }
      if (v > LIMITS[k]) { p.push(`limits.${k}: may only be lowered (max ${LIMITS[k]})`); continue; }
      limits[k] = v;
    }
  }

  let legacy = null;
  if (cfg.legacy !== undefined && cfg.legacy !== null) {
    const l = cfg.legacy;
    if (!l || typeof l.db !== 'string' || !l.db || typeof l.store !== 'string' || !l.store || typeof l.app !== 'string' || !ids.has(l.app)) {
      p.push('legacy: { db, store, app } with app one of apps[].id');
    } else legacy = { db: l.db, store: l.store, app: l.app };
  }

  if (p.length) throw new MemoryConfigError(p);
  return deepFreeze({
    brand: { name: cfg.brand.name },
    namespace: cfg.namespace,
    storeUrl: store.href,
    storeOrigin: store.origin,
    apps: cfg.apps.map((a) => ({ id: a.id, origin: a.origin, label: { ...a.label } })),
    strings: cfg.strings ? JSON.parse(JSON.stringify(cfg.strings)) : {},
    theme: cfg.theme ? JSON.parse(JSON.stringify(cfg.theme)) : {},
    tools,
    features,
    limits,
    legacy,
    __valid: true,
  });
}

export function assertConfig(cfg) {
  if (!cfg || cfg.__valid !== true || !Object.isFrozen(cfg)) throw new MemoryConfigError(['pass the result of defineMemoryConfig(...)']);
  return cfg;
}

export function appLabel(cfg, id, lang) {
  const a = cfg.apps.find((x) => x.id === id);
  if (!a) return id;
  return a.label[lang] || a.label[String(lang).split('-')[0]] || a.label.en || Object.values(a.label)[0];
}
