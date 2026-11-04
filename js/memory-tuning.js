// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
//
// Tuning: every number the memory uses for READING (budgets, retrieval
// weights, thresholds, block wording) comes from one versioned tuning file per
// model, merged over a default. The files live as JSON under `tuning/` and are
// compiled into the generated module `tuning.js`; only the autotune skill
// writes them.
//
// Safety floors are not tuning. `clampTuning` pulls every value into the range
// its schema allows, so a bad file can make retrieval worse but can never
// cross a floor (the support threshold, for example, never drops below 0.6).
//
// `targets` is open: a later target (tool-router thresholds, prompt pieces per
// model) adds its own section and its code reads `tuningFor(id, ctx).<target>`.

import FILES from './tuning.js';
import { BLOCK_WORDINGS, RULE_VARIANTS, BLOCK_FORMATS } from './memory-text.js';

// Type and range of every tunable key. Keys are dotted paths under
// `targets.memory` (per model) or `shared.memory` (store-wide, default only).
export const TUNING_SCHEMA = Object.freeze({
  memory: Object.freeze({
    'budget.share': { type: 'float', min: 0.015, max: 0.06 },
    'budget.min': { type: 'int', min: 128, max: 320 },
    'budget.max': { type: 'int', min: 256, max: 1024 },
    'budget.noCtx': { type: 'int', min: 128, max: 600 },
    'budget.coreShare': { type: 'float', min: 0.2, max: 0.6 },
    'budget.coreMax': { type: 'int', min: 64, max: 320 },
    'budget.smallCtx': { type: 'int', min: 4096, max: 16384 },
    'budget.maxItemsSmall': { type: 'int', min: 2, max: 8 },
    'budget.maxItems': { type: 'int', min: 4, max: 24 },
    'rank.k1': { type: 'float', min: 0.5, max: 2 },
    'rank.b': { type: 'float', min: 0.2, max: 0.9 },
    'rank.keyBoost': { type: 'float', min: 1, max: 4 },
    'rank.prevWeight': { type: 'float', min: 0, max: 1 },
    'rank.halfLifeDays': { type: 'float', min: 1, max: 180 },
    'rank.wRel': { type: 'float', min: 1, max: 1 },
    'rank.wRec': { type: 'float', min: 0, max: 2 },
    'rank.wImp': { type: 'float', min: 0, max: 2 },
    'rank.wTmp': { type: 'float', min: 0, max: 2 },
    'rank.minScore': { type: 'float', min: 0, max: 0.6 },
    'abstain.minCoverage': { type: 'float', min: 0.2, max: 0.8 },
    'block.format': { type: 'choice', values: BLOCK_FORMATS },
    'block.wording': { type: 'choice', values: BLOCK_WORDINGS },
    'block.rules': { type: 'choice', values: RULE_VARIANTS },
    'proposer.perTurn': { type: 'int', min: 0, max: 1 },
    'proposer.perConv': { type: 'int', min: 0, max: 3 },
  }),
  shared: Object.freeze({
    'memory.write.support': { type: 'float', min: 0.6, max: 0.85 },
    'memory.write.dupSim': { type: 'float', min: 0.75, max: 0.95 },
    'memory.write.similarMin': { type: 'float', min: 0.4, max: 0.75 },
  }),
});

const isObj = (x) => x && typeof x === 'object' && !Array.isArray(x);
function merge(a, b) {
  const out = { ...a };
  for (const [k, v] of Object.entries(b || {})) out[k] = isObj(v) && isObj(a?.[k]) ? merge(a[k], v) : v;
  return out;
}
const getPath = (o, p) => p.split('.').reduce((x, k) => (x == null ? undefined : x[k]), o);
function setPath(o, p, v) {
  const ks = p.split('.');
  let x = o;
  for (const k of ks.slice(0, -1)) { if (!isObj(x[k])) x[k] = {}; x = x[k]; }
  x[ks[ks.length - 1]] = v;
}
const clone = (o) => JSON.parse(JSON.stringify(o));

let warned = new Set();
function warn(code) {
  if (warned.has(code)) return;
  warned.add(code);
  try { console.warn(`[memory] tuning: ${code}`); } catch { /* no console */ }
}

function clampSection(section, schema, prefix, fallback) {
  const out = clone(section || {});
  for (const [key, s] of Object.entries(schema)) {
    let v = getPath(out, key);
    const d = getPath(fallback, key);
    if (v === undefined) { if (d !== undefined) setPath(out, key, d); continue; }
    if (s.type === 'choice') {
      if (!s.values.includes(v)) { warn(`${prefix}${key}:bad-choice`); setPath(out, key, s.values.includes(d) ? d : s.values[0]); }
      continue;
    }
    if (typeof v !== 'number' || !Number.isFinite(v)) { warn(`${prefix}${key}:not-a-number`); v = d; }
    let c = Math.min(s.max, Math.max(s.min, v));
    if (s.type === 'int') c = Math.round(c);
    if (c !== v) warn(`${prefix}${key}:clamped`);
    setPath(out, key, c);
  }
  return out;
}

// Pull every tunable into its range; unknown or broken values fall back to the
// built-in default file. Never throws.
export function clampTuning(tuning) {
  const def = FILES.default || {};
  return {
    ...tuning,
    memory: clampSection(tuning?.memory, TUNING_SCHEMA.memory, 'memory.', def.targets?.memory || {}),
  };
}

// The store-wide write thresholds: only the default file may hold them, so no
// model's file can change how the shared store grounds or dedupes.
export function sharedTuning() {
  const def = FILES.default || {};
  const out = clampSection(def.shared || {}, TUNING_SCHEMA.shared, 'shared.', def.shared || {});
  return out;
}

export function isStale(file, benchVersion, ctx = null) {
  if (!file) return false;
  if (benchVersion != null && file.bench?.version !== benchVersion) return true;
  if (ctx != null && file.ctx != null && file.ctx !== ctx) return true;
  return false;
}

// The tuning for one registry model: default ← <modelId>, clamped. `ctx` is
// the registry ctx of that model; a file tuned for another ctx still applies
// but is reported stale.
export function tuningFor(modelId, ctx = null) {
  const def = FILES.default || {};
  const own = modelId && Object.prototype.hasOwnProperty.call(FILES, modelId) && modelId !== 'default' ? FILES[modelId] : null;
  let targets = merge(def.targets || {}, own?.targets || {});
  const out = clampTuning(targets);
  return Object.freeze({
    ...out,
    model: own ? modelId : 'default',
    rev: own?.rev ?? def.rev ?? 0,
    stale: own ? isStale(own, def.bench?.version, ctx) : false,
  });
}

// A candidate tuning that is not in a file yet (the autotune skill's
// challenger): `override` = { memory?: {...}, shared?: { memory: { write } } },
// merged over what `tuningFor` gives and clamped the same way.
export function resolveTuning(modelId, ctx, override = null) {
  const base = tuningFor(modelId, ctx);
  if (!override) return base;
  const memory = clampTuning({ memory: merge(base.memory, override.memory || {}) }).memory;
  return Object.freeze({ ...base, memory, override: true });
}
export function resolveShared(override = null) {
  const base = sharedTuning();
  if (!override?.shared) return base;
  return clampSection(merge(base, override.shared), TUNING_SCHEMA.shared, 'shared.', base);
}

export function tuningFiles() { return FILES; }
export const __resetWarnings = () => { warned = new Set(); };
