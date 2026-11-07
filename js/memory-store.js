// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
//
// User memory: the store. The ONLY code that writes the memory database.
//
// It runs in the store page (a small page on its own path or subdomain that
// every app embeds as an iframe) and, in-process, in an app that shares the
// store page's origin. Both paths go through `store.call(op, args, ctx)`, which
// applies the same gate, limits and rules.
//
// Storage is IndexedDB on the store origin, named by `cfg.namespace`. Every
// read-modify-write runs inside a Web Lock and commits in one transaction;
// after a commit the store posts the new revision on a BroadcastChannel so
// every open frame can tell its app to re-query. No fact text is broadcast.
//
// Nothing here talks to a network. The store page is served with
// `connect-src 'none'`, so it could not even if it tried.

import {
  LIMITS, SCHEMA, newId, newPid, hashFact, normalizeFact, lintFact, gateWrite, slotOf, kindOf, keysFor,
  makeRecord, fromLegacy, validateRecord, decideWrite, renderBlock, isExclusive, detectLang,
} from './memory-core.js';
import { buildBM25, scoreRecords, memoryBudget, parseWhen, fold, prefixMatch, contentTerms, DAY } from './memory-rank.js';
import { tuningFor, resolveTuning, resolveShared } from './memory-tuning.js';
import { assertConfig } from './memory-config.js';

export const STORES = Object.freeze({
  records: { keyPath: 'id', indexes: ['status', 'hash', 'slot', 'updatedAt'] },
  proposals: { keyPath: 'id', indexes: ['hash', 'createdAt'] },
  pendingDelete: { keyPath: 'id', indexes: ['purgeAt'] },
  tombstones: { keyPath: 'key', indexes: [] },
  dismissed: { keyPath: 'hash', indexes: [] },
  grants: { keyPath: 'origin', indexes: [] },
  meta: { keyPath: 'key', indexes: [] },
});
const NAMES = Object.keys(STORES);

export const ERR = Object.freeze({
  bad_args: 'bad_args', too_long: 'too_long', not_a_fact: 'not_a_fact', not_from_user: 'not_from_user',
  no_grant: 'no_grant', disabled: 'disabled', not_found: 'not_found', ambiguous: 'ambiguous', rate: 'rate',
  full: 'full', quota: 'quota', readonly: 'readonly', unavailable: 'unavailable', incognito: 'incognito',
});
class OpError extends Error {
  constructor(code, extra = {}) { super(code); this.code = code; Object.assign(this, extra); }
}
const fail = (code, extra) => { throw new OpError(code, extra); };

// Ops an embedding app may call. Everything else is refused with `readonly`.
const APP_OPS = new Set(['hello', 'context', 'search', 'list', 'resolve', 'save', 'propose', 'confirm', 'dismiss',
  'forget', 'restore', 'forgetConv', 'byConv', 'grant', 'touch', 'subscribe']);
// Ops of the memory UI that lives on the store origin (panel and manager).
const UI_OPS = new Set(['edit', 'pin', 'setScope', 'keep', 'add', 'merge', 'all']);
// Bulk and destructive ops: only the top-level manager page.
const MANAGER_OPS = new Set(['export', 'import', 'wipe', 'settings', 'revoke', 'purge']);
// `resolve` (forget and replace by text) is a safety decision, not retrieval
// quality: a fixed floor, never tuned.
const RESOLVE_MIN_COVERAGE = 0.5;
// `touch` counts as a read (one per turn), `dismiss` as a write: both commit.
const READ_OPS = new Set(['context', 'search', 'list', 'resolve', 'byConv', 'touch']);
const WRITE_OPS = new Set(['save', 'propose', 'confirm', 'grant', 'dismiss']);
// Ops that remove or replace a current record: one shared, stricter limit.
const FORGET_OPS = new Set(['forget', 'forgetConv', 'restore']);
// How long the app that made an update may undo it (the card's «Deshacer»).
// A safety bound, not a tunable: after it, only the panel or the manager can
// bring an older value back.
export const UPDATE_UNDO_MS = 10 * 60 * 1000;

// ── Backends ─────────────────────────────────────────────────────────────────

// In-memory backend with the same contract as the IndexedDB one. Used by the
// unit suite and as a last resort when storage is blocked (then `writable` is
// false and the store refuses writes instead of pretending).
export function memoryBackend({ writable = true } = {}) {
  const data = Object.fromEntries(NAMES.map((n) => [n, new Map()]));
  const key = (n, v) => v[STORES[n].keyPath];
  const copy = (x) => (x === undefined ? undefined : structuredClone(x));
  return {
    writable,
    kind: 'memory',
    async getAll(n) { return [...data[n].values()].map(copy); },
    async get(n, k) { return copy(data[n].get(k)); },
    async commit(ops) {
      if (!writable) throw new OpError(ERR.unavailable);
      for (const o of ops) if (!data[o.store]) throw new Error(`unknown store ${o.store}`);
      for (const o of ops) {
        if (o.clear) data[o.store].clear();
        else if (o.del !== undefined) data[o.store].delete(o.del);
        else data[o.store].set(key(o.store, o.put), copy(o.put));
      }
    },
    async close() {},
    _data: data,
  };
}

const req = (r) => new Promise((ok, ko) => { r.onsuccess = () => ok(r.result); r.onerror = () => ko(r.error); });

// An IndexedDB connection that survives the browser closing it. WebKit closes
// connections on its own (a tab put to sleep, memory pressure, storage
// evicted), and from then on every `db.transaction()` throws InvalidStateError
// ("the database connection is closing"). So the connection is forgotten when
// it closes or another tab asks for a new version, and a transaction that could
// not start, or that the browser aborted, runs ONCE more on a new connection.
// That retry never writes twice: an aborted transaction commits nothing, and a
// transaction that fails for its own reasons (a constraint, the quota) is not
// retried at all.
//   open()   → Promise<IDBDatabase>, a new connection each call
//   run(db)  → Promise of one transaction, settled when it completes or fails
const CONNECTION_LOST = new Set(['InvalidStateError', 'AbortError', 'UnknownError']);
export function idbConnection(open) {
  let current = null;
  let closed = false;
  const forget = (p) => { if (current === p) current = null; };
  const connect = () => {
    if (current) return current;
    const p = current = open().then((db) => {
      db.onclose = () => forget(p);
      db.onversionchange = () => { db.close(); forget(p); };
      return db;
    });
    p.catch(() => forget(p));
    return p;
  };
  return {
    connect,
    async tx(run) {
      for (let retried = false; ; retried = true) {
        if (closed) throw new OpError(ERR.unavailable);
        const p = connect();
        const db = await p;
        try {
          return await run(db);
        } catch (e) {
          if (retried || !CONNECTION_LOST.has(e?.name)) throw e;
          forget(p);
          try { db.close(); } catch { /* already closing */ }
        }
      }
    },
    async close() {
      closed = true;
      const p = current;
      current = null;
      if (p) (await p.catch(() => null))?.close();
    },
  };
}

export async function idbBackend(name, idb = globalThis.indexedDB) {
  if (!idb) throw new OpError(ERR.unavailable);
  const conn = idbConnection(() => {
    const open = idb.open(name, 1);
    open.onupgradeneeded = () => {
      const db = open.result;
      for (const [n, s] of Object.entries(STORES)) {
        if (db.objectStoreNames.contains(n)) continue;
        const os = db.createObjectStore(n, { keyPath: s.keyPath });
        for (const i of s.indexes) os.createIndex(i, i);
      }
    };
    return req(open);
  });
  await conn.connect();
  return {
    writable: true,
    kind: 'idb',
    getAll(n) { return conn.tx((db) => req(db.transaction(n, 'readonly').objectStore(n).getAll())); },
    get(n, k) { return conn.tx((db) => req(db.transaction(n, 'readonly').objectStore(n).get(k))); },
    commit(ops) {
      if (!ops.length) return Promise.resolve();
      const names = [...new Set(ops.map((o) => o.store))];
      return conn.tx((db) => new Promise((ok, ko) => {
        const tx = db.transaction(names, 'readwrite');
        tx.oncomplete = () => ok();
        tx.onerror = () => ko(tx.error);
        tx.onabort = () => ko(tx.error || new Error('aborted'));
        for (const o of ops) {
          const os = tx.objectStore(o.store);
          if (o.clear) os.clear();
          else if (o.del !== undefined) os.delete(o.del);
          else os.put(o.put);
        }
      }));
    },
    close() { return conn.close(); },
  };
}

// Reads (and, for a forget, deletes from) an older v1 store without ever
// upgrading, creating or blocking it: it opens WITHOUT a version number and
// aborts the open if the database does not exist yet.
export function idbLegacy({ db: dbName, store }, idb = globalThis.indexedDB) {
  const openExisting = async () => {
    if (!idb) return null;
    if (typeof idb.databases === 'function') {
      try { const list = await idb.databases(); if (!list.some((d) => d.name === dbName)) return null; } catch { /* fall through to the guarded open */ }
    }
    return new Promise((ok) => {
      let created = false;
      const r = idb.open(dbName);
      r.onupgradeneeded = () => { created = true; try { r.transaction.abort(); } catch { /* aborting is the point */ } };
      r.onsuccess = () => { if (created) { r.result.close(); ok(null); } else ok(r.result); };
      r.onerror = () => ok(null);
      r.onblocked = () => ok(null);
    });
  };
  return {
    async readAll() {
      const db = await openExisting();
      if (!db) return null;
      try {
        if (!db.objectStoreNames.contains(store)) return null;
        return await req(db.transaction(store, 'readonly').objectStore(store).getAll());
      } finally { db.close(); }
    },
    async delete(ids) {
      const db = await openExisting();
      if (!db) return;
      try {
        if (!db.objectStoreNames.contains(store)) return;
        await new Promise((ok) => {
          const tx = db.transaction(store, 'readwrite');
          for (const id of ids) tx.objectStore(store).delete(id);
          tx.oncomplete = ok; tx.onerror = ok; tx.onabort = ok;
        });
      } finally { db.close(); }
    },
  };
}

// ── Locking ──────────────────────────────────────────────────────────────────

function localMutex() {
  let tail = Promise.resolve();
  return (fn) => {
    const run = tail.then(fn, fn);
    tail = run.catch(() => {});
    return run;
  };
}

// ── The store ────────────────────────────────────────────────────────────────

const isObj = (x) => x && typeof x === 'object' && !Array.isArray(x);
const str = (x, max = 4000) => (typeof x === 'string' ? x.slice(0, max) : '');
const CONV_RE = /^[A-Za-z0-9_-]{1,40}$/;

export async function openStore(cfg, opts = {}) {
  assertConfig(cfg);
  const now = opts.now || (() => Date.now());
  const L = cfg.limits;
  let backend = opts.backend || null;
  if (!backend) {
    try { backend = await idbBackend(cfg.namespace); }
    catch { backend = memoryBackend({ writable: false }); }
  }
  const legacy = opts.legacy !== undefined ? opts.legacy : (cfg.legacy ? idbLegacy(cfg.legacy) : null);
  const navLocks = opts.locks !== undefined ? opts.locks : globalThis.navigator?.locks;
  const mutex = localMutex();
  const withLock = (fn) => (navLocks
    ? navLocks.request(cfg.namespace + '-write', () => mutex(fn))
    : mutex(fn));
  let channel = null;
  if (opts.channel !== undefined) channel = opts.channel;
  else if (typeof globalThis.BroadcastChannel === 'function') { try { channel = new BroadcastChannel(cfg.namespace); } catch { channel = null; } }
  const listeners = new Set();
  // `opts.tuning` is only for the benchmark and the autotune skill: a
  // challenger tuning that is not in a file yet. Pages never pass it.
  const shared = () => resolveShared(opts.tuning).memory.write;
  const tuneFor = (id, ctx) => resolveTuning(id, ctx, opts.tuning);

  // Cache, reloaded whenever `meta.rev` moves (another tab wrote).
  let cache = null;
  async function load(force = false) {
    const rev = (await backend.get('meta', 'rev'))?.value ?? 0;
    if (!force && cache && cache.rev === rev) return cache;
    const [records, proposals, pending, tombs, dismissed, grants, meta] = await Promise.all(NAMES.map((n) => backend.getAll(n)));
    cache = {
      rev,
      records: new Map(records.map((r) => [r.id, r])),
      proposals: new Map(proposals.map((p) => [p.id, p])),
      pending: new Map(pending.map((p) => [p.id, p])),
      tombs: new Map(tombs.map((t) => [t.key, t])),
      dismissed: new Set(dismissed.map((d) => d.hash)),
      grants: new Map(grants.map((g) => [g.origin, g])),
      meta: new Map(meta.map((m) => [m.key, m.value])),
      index: null, indexRev: -1,
    };
    return cache;
  }

  const settingsOf = (c) => {
    const s = c.meta.get('settings') || {};
    const apps = {};
    for (const a of cfg.apps) apps[a.id] = { read: s.apps?.[a.id]?.read !== false, write: s.apps?.[a.id]?.write !== false };
    return { on: s.on !== false, proposals: s.proposals !== false && cfg.features.proposals, apps };
  };
  const originOf = (app) => cfg.apps.find((a) => a.id === app)?.origin || null;

  // One transaction: the ops, plus the new revision. Then tell everyone.
  async function commit(ops, { forgot = [], wiped = false } = {}) {
    const rev = ((await backend.get('meta', 'rev'))?.value ?? 0) + 1;
    const all = [...ops, { store: 'meta', put: { key: 'rev', value: rev } }];
    try { await backend.commit(all); }
    catch (e) {
      cache = null;
      if (e?.name === 'QuotaExceededError' || /quota/i.test(String(e?.message))) fail(ERR.quota);
      if (e instanceof OpError) throw e;
      fail(ERR.unavailable);
    }
    await load(true);
    const ev = { rev, ...(forgot.length ? { forgot } : {}), ...(wiped ? { wiped: true } : {}) };
    try { channel?.postMessage(ev); } catch { /* closed channel */ }
    for (const f of listeners) { try { f(ev); } catch { /* a listener must not break a write */ } }
    return rev;
  }
  if (channel) {
    channel.onmessage = (e) => {
      const d = e.data;
      if (!isObj(d) || typeof d.rev !== 'number') return;
      cache = null;
      const ev = { rev: d.rev, ...(Array.isArray(d.forgot) ? { forgot: d.forgot.filter((h) => typeof h === 'string') } : {}), ...(d.wiped ? { wiped: true } : {}) };
      for (const f of listeners) { try { f(ev); } catch { /* ignore */ } }
    };
  }

  // ── Maintenance: due purges, lazy expiry, caps. Runs inside the lock. ──────
  function maintenanceOps(c) {
    const t = now();
    const ops = [];
    for (const p of c.pending.values()) if (p.purgeAt <= t) ops.push({ store: 'pendingDelete', del: p.id });
    for (const r of c.records.values()) {
      if (r.status === 'active' && r.expiresAt && r.expiresAt < t) {
        ops.push({ store: 'records', put: { ...r, status: 'expired', validTo: r.expiresAt } });
      } else if (r.status === 'expired' && r.validTo && t - r.validTo > L.expiredPurgeDays * DAY) {
        ops.push({ store: 'records', del: r.id });
      }
    }
    for (const p of c.proposals.values()) if (p.expiresAt && p.expiresAt < t) ops.push({ store: 'proposals', del: p.id });
    const hist = [...c.records.values()].filter((r) => r.status === 'superseded' || r.status === 'expired').sort((a, b) => (a.validTo || 0) - (b.validTo || 0));
    for (const r of hist.slice(0, Math.max(0, hist.length - L.history))) ops.push({ store: 'records', del: r.id });
    for (const tb of c.tombs.values()) if (tb.key.startsWith('h:') && t - tb.at > 365 * DAY) ops.push({ store: 'tombstones', del: tb.key });
    return ops;
  }

  // The legacy copy of a forgotten record is deleted only when its forget
  // becomes final (the undo window ends), so an undo keeps it intact.
  const purgedIds = (ops) => ops.filter((o) => o.store === 'pendingDelete' && o.del !== undefined).map((o) => o.del);
  function dropLegacy(ids) {
    if (legacy && cfg.legacy && ids.length) legacy.delete(ids).catch(() => {});
  }
  async function runMaintenance(c) {
    const maint = maintenanceOps(c);
    if (!maint.length) return;
    await commit(maint);
    dropLegacy(purgedIds(maint));
  }

  async function write(fn) {
    return withLock(async () => {
      await runMaintenance(await load());
      return fn(cache);
    });
  }

  // ── Rate limits (per app origin; the UI on the store origin is exempt) ─────
  // Write and forget windows live in `meta`, under the write lock, so every
  // store frame and every reload of the same browser shares them. Read windows
  // stay per frame: reads change nothing and must not cost a database write.
  const windows = new Map();
  const within = (t, w) => (Array.isArray(w) ? w : []).filter((x) => typeof x === 'number' && x <= t && t - x < 864e5);
  function check(recent, t, perMinute, perDay) {
    const lastMinute = recent.filter((x) => t - x < 6e4).length;
    if (lastMinute >= perMinute || recent.length >= perDay) fail(ERR.rate);
    recent.push(t);
    return recent;
  }
  function rate(app, kind, perMinute, perDay = Infinity) {
    const t = now();
    const k = `${app}:${kind}`;
    windows.set(k, check(within(t, windows.get(k)), t, perMinute, perDay));
  }
  async function rateShared(app, kind, perMinute, perDay = Infinity) {
    if (!backend.writable) return rate(app, kind, perMinute, perDay);
    await withLock(async () => {
      const t = now();
      const key = `rate:${app}:${kind}`;
      const recent = check(within(t, (await backend.get('meta', key))?.value), t, perMinute, perDay);
      // No revision bump: rate windows are bookkeeping, not memory content.
      await backend.commit([{ store: 'meta', put: { key, value: recent } }]);
    });
  }

  // ── Helpers ────────────────────────────────────────────────────────────────
  const activeOf = (c) => [...c.records.values()].filter((r) => r.status === 'active');
  const pub = (r) => r && ({
    id: r.id, fact: r.fact, at: r.at, updatedAt: r.updatedAt, kind: r.kind, slot: r.slot, scope: r.scope,
    pinned: r.pinned, importance: r.importance, status: r.status, eventAt: r.eventAt, possibleDup: r.possibleDup,
    supersedes: r.supersedes, supersededBy: r.supersededBy, validTo: r.validTo,
    source: { app: r.source?.app, how: r.source?.how, conv: r.source?.conv ?? null }, ...(r.tooLong ? { tooLong: true } : {}),
  });
  function allowedScopes(app, scopes) {
    const ok = ['all'];
    if (app && app !== 'manager') ok.push(`app:${app}`);
    const list = Array.isArray(scopes) && scopes.length ? scopes : ok;
    return list.filter((s) => typeof s === 'string' && (ok.includes(s) ||
      (/^project:[a-z0-9]{1,40}$/.test(s) && (app === 'manager' || cfg.features.projectScope.includes(app)))));
  }
  function scopeFor(app, scope) {
    if (scope === undefined || scope === null || scope === 'all') return 'all';
    if (typeof scope !== 'string') fail(ERR.bad_args);
    if (scope === `app:${app}` && app !== 'manager') return scope;
    if (/^project:[a-z0-9]{1,40}$/.test(scope) && (app === 'manager' || cfg.features.projectScope.includes(app))) return scope;
    fail(ERR.bad_args);
  }
  function chainOf(c, id) {
    const out = [];
    let r = c.records.get(id);
    const seen = new Set();
    while (r && !seen.has(r.id)) {
      seen.add(r.id);
      out.push(r);
      const prev = r.supersedes ? c.records.get(r.supersedes) : null;
      r = prev && prev.status !== 'active' ? prev : null;
    }
    return out;
  }
  // May this app act on this record at all (forget, restore, touch)?
  const inScope = (app, r) => r.scope === 'all' || r.scope === `app:${app}` ||
    (r.scope.startsWith('project:') && cfg.features.projectScope.includes(app));
  function checkGrant(c, ctx) {
    if (ctx.ui) return;
    const origin = originOf(ctx.app);
    const g = origin ? c.grants.get(origin) : null;
    if (!g) fail(ERR.no_grant);
    if (g.state !== 'granted') fail(ERR.disabled);
    const s = settingsOf(c);
    if (!s.on || !s.apps[ctx.app]?.write) fail(ERR.disabled);
  }
  // A `from-content` proposal changes nothing until the user clicks in the
  // panel, so an app with no grant row yet may leave one (an older client that
  // cannot show the grant card would otherwise lose the fact). Any other label
  // still asks for the grant first, and a denied grant, memory off or the app's
  // write switch off refuse. Returns true when there is no grant row.
  function checkProposeGrant(c, ctx, label) {
    if (ctx.ui) return false;
    const origin = originOf(ctx.app);
    if (origin && !c.grants.get(origin) && label === 'from-content') {
      const s = settingsOf(c);
      if (!s.on || !s.apps[ctx.app]?.write) fail(ERR.disabled);
      return true;
    }
    checkGrant(c, ctx);
    return false;
  }
  // The manager's switches apply to every read an app makes, not only to the
  // context block: memory off, or this app's read switch off, reads nothing.
  const readOff = (c, ctx) => {
    if (ctx.ui) return false;
    const s = settingsOf(c);
    return !s.on || !s.apps[ctx.app]?.read;
  };
  const tombstoned = (c, id, hash) => c.tombs.has(`id:${id}`) || (hash && c.tombs.has(`h:${hash}`));
  const recentTomb = (c, hash) => { const tb = c.tombs.get(`h:${hash}`); return tb && now() - tb.at < L.tombstoneDays * DAY; };

  // The core of every write: consolidation against the active records.
  async function applyWrite(c, cand, ops, { how, app, conv, replaces = null }) {
    const t = now();
    const hash = await hashFact(cand.fact);
    if (recentTomb(c, hash) && !['user-asked', 'user-typed', 'user-confirmed'].includes(how)) {
      return { ok: false, code: ERR.not_from_user, message: 'forgotten' };
    }
    const active = activeOf(c);
    const slot = cand.slot !== undefined ? cand.slot : slotOf(cand.fact);
    const scope = cand.scope || 'all';
    let decision = replaces
      ? { action: 'updated', target: replaces }
      : decideWrite({ fact: cand.fact, hash, slot, scope, how }, active, shared());
    if (decision.target && decision.target.scope !== scope && !replaces) decision = { action: 'add' };
    const dropProps = [...c.proposals.values()].filter((p) => p.hash === hash).map((p) => ({ store: 'proposals', del: p.id }));
    ops.push(...dropProps);
    const source = { app, how, conv: conv || null };
    if (decision.action === 'duplicate') {
      const d = decision.target;
      ops.push({ store: 'records', put: { ...d, updatedAt: t, importance: Math.max(d.importance || 3, cand.importance || 3) } });
      return { ok: true, status: 'duplicate', id: d.id, fact: d.fact };
    }
    if (decision.action === 'negate') {
      const d = decision.target;
      ops.push({ store: 'records', put: { ...d, status: 'superseded', validTo: t, supersededBy: null } });
      return { ok: true, status: 'updated', negated: true, id: d.id, previous: d.fact, fact: d.fact };
    }
    if (active.length >= L.active) return { ok: false, code: ERR.full };
    const rec = makeRecord({
      ...(cand.id && !c.records.has(cand.id) ? { id: cand.id } : {}),
      at: cand.at, fact: cand.fact, now: t, slot, scope, importance: cand.importance || 3, source,
      eventAt: cand.eventAt ?? null, expiresAt: cand.expiresAt ?? null,
      kind: kindOf(cand.fact, slot, cand.eventAt), lang: cand.lang || detectLang(cand.fact), hash,
    });
    if (cand.pinned === true && active.filter((r) => r.pinned).length < L.pinned) { rec.pinned = true; rec.importance = 5; }
    if (decision.action === 'updated') {
      const old = decision.target;
      rec.supersedes = old.id;
      if (old.pinned) rec.pinned = true;
      rec.importance = Math.max(rec.importance, old.importance === 5 ? 5 : rec.importance);
      ops.push({ store: 'records', put: { ...old, status: 'superseded', validTo: t, supersededBy: rec.id, pinned: false } });
      ops.push({ store: 'records', put: rec });
      return { ok: true, status: 'updated', id: rec.id, previous: old.fact, previousId: old.id, fact: rec.fact };
    }
    if (decision.action === 'similar') {
      const s = decision.target;
      rec.possibleDup = s.id;
      ops.push({ store: 'records', put: { ...s, possibleDup: rec.id } });
      ops.push({ store: 'records', put: rec });
      return { ok: true, status: 'similar', id: rec.id, previous: s.fact, previousId: s.id, fact: rec.fact };
    }
    ops.push({ store: 'records', put: rec });
    return { ok: true, status: 'saved', id: rec.id, fact: rec.fact };
  }

  async function addProposal(c, p, ops) {
    const hash = await hashFact(p.fact);
    if (c.dismissed.has(hash)) return { status: 'noop' };
    const active = activeOf(c);
    if (active.some((r) => r.hash === hash)) return { status: 'noop' };
    const d = decideWrite({ fact: p.fact, hash, slot: p.slot ?? slotOf(p.fact), scope: p.scope || 'all', how: 'proposal' }, active, shared());
    if (d.action === 'duplicate') return { status: 'noop' };
    const existing = [...c.proposals.values()].find((x) => x.hash === hash);
    if (existing) return { pid: existing.id };
    const t = now();
    const props = [...c.proposals.values()].sort((a, b) => a.createdAt - b.createdAt);
    for (const old of props.slice(0, Math.max(0, props.length + 1 - L.proposals))) ops.push({ store: 'proposals', del: old.id });
    const id = newPid();
    const slot = p.slot ?? slotOf(p.fact);
    ops.push({ store: 'proposals', put: {
      id, fact: p.fact, kind: p.kind || kindOf(p.fact, slot, p.eventAt), slot, scope: p.scope || 'all',
      source: { app: p.app, how: 'user-confirmed', conv: p.conv || null }, label: p.label === 'from-user' ? 'from-user' : 'from-content',
      hash, createdAt: t, expiresAt: t + L.proposalDays * DAY, eventAt: p.eventAt ?? null, expiresEventAt: p.expiresAt ?? null,
    } });
    return { pid: id };
  }

  async function forgetIds(c, ids, ops) {
    const t = now();
    const purgeAt = t + L.undoGraceMs;
    const pending = [], hashes = [];
    for (const id of ids) {
      for (const r of chainOf(c, id)) {
        if (pending.includes(r.id)) continue;
        pending.push(r.id);
        hashes.push(r.hash);
        ops.push({ store: 'records', del: r.id });
        ops.push({ store: 'pendingDelete', put: { id: r.id, record: r, purgeAt, group: id } });
        ops.push({ store: 'tombstones', put: { key: `id:${r.id}`, at: t } });
        if (r.hash) ops.push({ store: 'tombstones', put: { key: `h:${r.hash}`, at: t } });
        for (const p of c.proposals.values()) if (p.hash === r.hash) ops.push({ store: 'proposals', del: p.id });
        // A record that pointed at it as a possible duplicate no longer does.
        for (const o of c.records.values()) if (o.possibleDup === r.id) ops.push({ store: 'records', put: { ...o, possibleDup: null } });
      }
    }
    return { pending, hashes: hashes.filter(Boolean), purgeAt };
  }

  // ── Retrieval ──────────────────────────────────────────────────────────────
  function context(c, args, ctx) {
    const model = isObj(args.model) ? args.model : {};
    const ctxLen = Number.isFinite(model.ctx) && model.ctx > 0 ? Math.floor(model.ctx) : null;
    const tun = tuneFor(typeof model.id === 'string' ? model.id : null, ctxLen).memory;
    const budget = memoryBudget(ctxLen, tun.budget);
    const lang = typeof args.lang === 'string' ? args.lang : 'es';
    const empty = { text: '', tokens: 0, ids: [], abstained: true, budget: budget.total };
    const s = settingsOf(c);
    if (args.incognito || !s.on || (!ctx.ui && !s.apps[ctx.app]?.read)) return empty;
    const scopes = allowedScopes(ctx.app, args.scopes);
    const query = str(args.query, 4000), prevQuery = str(args.prevQuery, 4000);
    const t = now();
    const when = parseWhen(query, t, lang);
    const live = [...c.records.values()].filter((r) => scopes.includes(r.scope) &&
      (r.status === 'active' && !(r.expiresAt && r.expiresAt < t) ||
       (when?.past && (r.status === 'superseded' || r.status === 'expired' || (r.expiresAt && r.expiresAt < t)))));
    if (!live.length) return empty;
    const core = live.filter((r) => r.status === 'active' && (r.pinned || (r.slot && isExclusive(r.slot))))
      .sort((a, b) => (b.pinned - a.pinned) || (b.importance - a.importance) || (b.updatedAt - a.updatedAt));
    const coreIds = new Set(core.map((r) => r.id));
    const rest = live.filter((r) => !coreIds.has(r.id));
    const scored = scoreRecords(rest, { query, prevQuery, now: t, rank: tun.rank, lang });
    let archive = [], abstained = false;
    const block = (arch) => renderBlock({
      core, archive: arch, strings: cfg.strings, lang, wording: tun.block.wording, format: tun.block.format,
      budget, toolList: `${cfg.tools.prefix}list`,
    });
    const allLines = rest.length;
    const roughAll = renderBlock({ core, archive: scored.items.map((i) => i.rec), strings: cfg.strings, lang, wording: tun.block.wording, format: tun.block.format, budget: { ...budget, total: 1e9 }, toolList: `${cfg.tools.prefix}list` });
    if (allLines && allLines <= budget.maxArchive && roughAll.tokens <= budget.total) {
      archive = scored.items.map((i) => i.rec);
    } else {
      const sig = scored.signal;
      const signal = sig.maxCov >= tun.abstain.minCoverage || sig.slotHit || sig.dateHit;
      if (!signal) abstained = true;
      else {
        const best = scored.items[0]?.score || 0;
        archive = scored.items
          .filter((i) => (i.rel > 0 || i.tmp || i.slotRel) && i.score >= tun.rank.minScore * best)
          .slice(0, budget.maxArchive).map((i) => i.rec);
        if (!archive.length && !sig.slotHit) abstained = true;
      }
    }
    const out = block(archive);
    const archiveIds = out.ids.filter((id) => !coreIds.has(id));
    if (archiveIds.length && !ctx.noTouch) touchLater(archiveIds);
    if (ctx.app && !ctx.ui) servedTo(ctx.app, out.ids);
    const facts = out.ids.map((id) => c.records.get(id)?.fact).filter(Boolean);
    return { text: out.text, tokens: out.tokens, ids: out.ids, facts, abstained, budget: budget.total };
  }

  // The ids each app was given in its last few context blocks: an app may
  // only `touch` those, so it cannot lift arbitrary records in the ranking.
  const served = new Map();
  function servedTo(app, ids) {
    const l = served.get(app) || [];
    l.push(new Set(ids));
    while (l.length > 3) l.shift();
    served.set(app, l);
  }
  const wasServed = (app, id) => (served.get(app) || []).some((s) => s.has(id));

  let touchQueue = new Set(), touchTimer = null;
  function touchLater(ids) {
    for (const id of ids) touchQueue.add(id);
    if (touchTimer) return;
    touchTimer = setTimeout(() => { touchTimer = null; const ids2 = [...touchQueue]; touchQueue = new Set(); doTouch(ids2).catch(() => {}); }, 0);
  }
  async function doTouch(ids) {
    if (!backend.writable || !ids.length) return;
    await write(async (c) => {
      const t = now();
      const ops = [];
      for (const id of ids) { const r = c.records.get(id); if (r && r.status === 'active') ops.push({ store: 'records', put: { ...r, lastUsedAt: t, useCount: (r.useCount || 0) + 1 } }); }
      if (ops.length) await commit(ops);
    });
  }

  function search(c, args, ctx) {
    if (readOff(c, ctx)) return { items: [], total: 0 };
    const scopes = allowedScopes(ctx.app, args.scopes);
    const limit = Math.min(L.searchMax, Math.max(1, args.limit | 0 || 10));
    const list = activeOf(c).filter((r) => scopes.includes(r.scope));
    const q = str(args.query, 2000);
    if (!q.trim()) {
      const items = list.sort((a, b) => (b.pinned - a.pinned) || (b.updatedAt - a.updatedAt)).slice(0, limit).map(pub);
      return { items, total: list.length };
    }
    // Search orders by relevance alone (then date match): recency and
    // importance shape the context block, not what the user looks up.
    const sc = scoreRecords(list, { query: q, now: now(), rank: tuneFor(null).memory.rank });
    const items = sc.items.filter((i) => i.rel > 0 || i.tmp || i.slotRel)
      .sort((a, b) => (b.rel - a.rel) || (b.tmp - a.tmp)).slice(0, limit);
    return { items: items.map((i) => ({ ...pub(i.rec), score: i.score })), total: list.length };
  }

  function resolve(c, args, ctx) {
    const ref = str(args.ref, 300).trim();
    if (!ref) fail(ERR.bad_args);
    if (readOff(c, ctx)) fail(ERR.not_found);
    const scopes = allowedScopes(ctx.app, args.scopes);
    const list = activeOf(c).filter((r) => scopes.includes(r.scope) || ctx.ui);
    const exact = list.find((r) => r.id === ref);
    if (exact) return { record: pub(exact) };
    const bare = ref.replace(/^\[|\]$/g, '');
    if (/^m[0-9a-z]{3,10}$/.test(bare)) {
      const pre = list.filter((r) => r.id.startsWith(bare));
      if (pre.length === 1) return { record: pub(pre[0]) };
    }
    const sc = scoreRecords(list, { query: ref, now: now(), rank: tuneFor(null).memory.rank });
    const hits = sc.items.filter((i) => i.rel > 0 || i.slotRel).sort((a, b) => b.rel - a.rel);
    if (!hits.length) fail(ERR.not_found);
    // The best hit wins only if it clearly beats the second and really
    // overlaps the reference; otherwise the user picks.
    if ((hits.length === 1 || hits[0].rel >= 1.5 * hits[1].rel) && (hits[0].cov >= RESOLVE_MIN_COVERAGE || hits[0].slotRel)) {
      return { record: pub(hits[0].rec) };
    }
    return { ambiguous: true, candidates: hits.slice(0, 3).map((i) => pub(i.rec)) };
  }

  // ── Op dispatch ────────────────────────────────────────────────────────────
  const recentRids = new Map();   // app -> Map(rid -> Promise<result>)

  async function handle(op, args, ctx) {
    const c0 = await load();
    if (op === 'hello') {
      const origin = originOf(ctx.app);
      const g = origin ? c0.grants.get(origin) : null;
      const s = settingsOf(c0);
      const created = c0.meta.get('created') || null;
      return {
        v: 2, minClient: 2, app: ctx.app, rev: c0.rev,
        count: activeOf(c0).filter((r) => r.scope === 'all' || r.scope === `app:${ctx.app}` || ctx.ui ||
          (r.scope.startsWith('project:') && cfg.features.projectScope.includes(ctx.app))).length,
        grant: ctx.ui ? 'granted' : (g?.state || null),
        enabled: { read: s.on && (ctx.ui || s.apps[ctx.app]?.read !== false), write: backend.writable && s.on && (ctx.ui || s.apps[ctx.app]?.write !== false) },
        persisted: !!c0.meta.get('persisted'), epoch: created, engine: opts.engine || null,
        storage: backend.writable ? 'ok' : 'blocked',
      };
    }
    if (op === 'context') return context(c0, args, ctx);
    if (op === 'search') return search(c0, args, ctx);
    if (op === 'list') {
      if (readOff(c0, ctx)) return { items: [], next: null, total: 0 };
      const limit = Math.min(L.listMax, Math.max(1, args.limit | 0 || L.listMax));
      const start = Math.max(0, parseInt(args.cursor || '0', 10) || 0);
      const scopes = allowedScopes(ctx.app, args.scopes);
      const list = activeOf(c0).filter((r) => ctx.ui || scopes.includes(r.scope)).sort((a, b) => (b.pinned - a.pinned) || (b.updatedAt - a.updatedAt));
      const items = list.slice(start, start + limit).map(pub);
      return { items, next: start + limit < list.length ? String(start + limit) : null, total: list.length };
    }
    if (op === 'resolve') return resolve(c0, args, ctx);
    if (op === 'byConv') {
      const conv = str(args.conv, 40);
      if (!CONV_RE.test(conv)) fail(ERR.bad_args);
      if (readOff(c0, ctx)) return { count: 0, items: [] };
      const items = activeOf(c0).filter((r) => r.source?.conv === conv && (ctx.ui || r.source?.app === ctx.app));
      return { count: items.length, items: items.map(pub) };
    }
    if (op === 'subscribe') return { rev: c0.rev };
    if (op === 'all' && ctx.ui) {
      return {
        records: [...c0.records.values()].map(pub), proposals: [...c0.proposals.values()].map((p) => ({ ...p })),
        settings: settingsOf(c0), grants: [...c0.grants.values()], meta: { migration: c0.meta.get('migration') || null, persisted: !!c0.meta.get('persisted') },
        pending: [...c0.pending.values()].map((p) => ({ id: p.id, purgeAt: p.purgeAt, group: p.group })),
      };
    }
    if (!backend.writable) fail(ERR.unavailable);

    return write(async (c) => {
      const ops = [];
      const t = now();
      switch (op) {
        case 'save': {
          checkGrant(c, ctx);
          const conv = args.conv && CONV_RE.test(String(args.conv)) ? String(args.conv) : null;
          const scope = scopeFor(ctx.app, args.scope);
          const g = gateWrite({ fact: args.fact, evidence: args.evidence, tainted: args.tainted, lang: str(args.lang, 10) || 'es', shared: shared(), now: t, strings: cfg.strings, limits: L });
          if (!g.ok) return { ok: false, code: g.code, n: g.n, message: g.code };
          // A replay from an app's outbox carries the time it was queued. An
          // erase-all or a forget of the same fact after that time wins.
          if (Number.isFinite(args.queuedAt)) {
            const queuedAt = Math.min(args.queuedAt, t);
            const wipedAt = c.meta.get('wipedAt') || 0;
            const tb = c.tombs.get(`h:${await hashFact(g.fact)}`);
            if (wipedAt >= queuedAt || (tb && tb.at >= queuedAt)) return { ok: false, code: ERR.not_from_user, message: 'forgotten' };
          }
          const importance = Math.min(4, Math.max(1, args.importance | 0 || 3));
          if (g.status === 'proposal') {
            const s = settingsOf(c);
            if (!s.proposals) return { ok: false, code: ERR.not_from_user, message: 'proposals off' };
            const r = await addProposal(c, { fact: g.fact, scope, conv, label: g.label, app: ctx.app, eventAt: g.eventAt, expiresAt: g.expiresAt }, ops);
            if (ops.length) await commit(ops);
            return r.pid ? { ok: true, status: 'proposed', pid: r.pid, fact: g.fact, label: g.label } : { ok: true, status: 'duplicate', fact: g.fact, id: null };
          }
          let replaces = null;
          if (args.replaces) {
            try { const r = resolve(c, { ref: args.replaces }, ctx); if (r.record) replaces = c.records.get(r.record.id); } catch { replaces = null; }
          }
          const res = await applyWrite(c, { fact: g.fact, scope, importance, eventAt: g.eventAt, expiresAt: g.expiresAt, lang: g.lang }, ops, { how: 'user-asked', app: ctx.app, conv, replaces });
          if (ops.length && res.ok) { await commit(ops); markPersist(); }
          return res;
        }
        case 'propose': {
          checkProposeGrant(c, ctx, args.label);
          const s = settingsOf(c);
          if (!s.proposals) return { status: 'noop' };
          const n = normalizeFact(args.fact, { now: t });
          if (!n.text) fail(ERR.bad_args);
          if (n.text.length > L.factMax) fail(ERR.too_long, { n: n.text.length });
          if (!lintFact(n.text).ok || !lintFact(String(args.fact)).ok) fail(ERR.not_a_fact);
          const conv = args.conv && CONV_RE.test(String(args.conv)) ? String(args.conv) : null;
          const r = await addProposal(c, { fact: n.text, scope: scopeFor(ctx.app, args.scope), conv, label: args.label, app: ctx.app, eventAt: n.eventAt, expiresAt: n.expiresAt }, ops);
          if (ops.length) await commit(ops);
          return r;
        }
        case 'confirm': {
          checkGrant(c, ctx);
          const p = c.proposals.get(str(args.pid, 20));
          if (!p) fail(ERR.not_found);
          let fact = p.fact, eventAt = p.eventAt ?? null, expiresAt = p.expiresEventAt ?? null;
          if (typeof args.fact === 'string' && args.fact.trim() && args.fact !== p.fact) {
            const n = normalizeFact(args.fact, { now: t });
            if (!n.text) fail(ERR.bad_args);
            if (n.text.length > L.factMax) return { ok: false, code: ERR.too_long, n: n.text.length };
            if (!lintFact(n.text).ok) return { ok: false, code: ERR.not_a_fact };
            fact = n.text; eventAt = n.eventAt; expiresAt = n.expiresAt;
          }
          ops.push({ store: 'proposals', del: p.id });
          const res = await applyWrite(c, { fact, scope: p.scope, eventAt, expiresAt }, ops, { how: 'user-confirmed', app: p.source?.app || ctx.app, conv: p.source?.conv });
          if (res.ok || ops.length) await commit(ops);
          markPersist();
          return res;
        }
        case 'dismiss': {
          checkGrant(c, ctx);
          const p = c.proposals.get(str(args.pid, 20));
          if (!p) return {};
          ops.push({ store: 'proposals', del: p.id });
          if (args.never) ops.push({ store: 'dismissed', put: { hash: p.hash } });
          await commit(ops);
          return {};
        }
        case 'forget': {
          checkGrant(c, ctx);
          const id = str(args.id, 20);
          const r = c.records.get(id);
          if (!r || (!ctx.ui && (r.status !== 'active' || !inScope(ctx.app, r)))) fail(ERR.not_found);
          const f = await forgetIds(c, [id], ops);
          await commit(ops, { forgot: f.hashes });
          scheduleSweep(f.purgeAt);
          return { pending: f.pending, purgeAt: f.purgeAt };
        }
        case 'restore': {
          checkGrant(c, ctx);
          const id = str(args.id, 20);
          const p = c.pending.get(id);
          // Pending entries exist only inside the undo window (maintenance
          // purged the due ones before this ran).
          if (p && (ctx.ui || inScope(ctx.app, p.record))) {
            const group = [...c.pending.values()].filter((x) => x.group === p.group);
            for (const g of group) {
              ops.push({ store: 'pendingDelete', del: g.id });
              ops.push({ store: 'records', put: g.record });
              ops.push({ store: 'tombstones', del: `id:${g.id}` });
              if (g.record.hash) ops.push({ store: 'tombstones', del: `h:${g.record.hash}` });
            }
            await commit(ops);
            return { ok: true, status: 'saved', id, fact: p.record.fact };
          }
          // Undo of an update: the old record becomes current again and the
          // one that replaced it is forgotten.
          const old = c.records.get(id);
          const newer0 = old?.supersededBy ? c.records.get(old.supersededBy) : null;
          // An app may only undo its own update, shortly after making it.
          const undoable = ctx.ui || (old && newer0 && newer0.status === 'active' && inScope(ctx.app, old) &&
            newer0.source?.app === ctx.app && t - (old.validTo || 0) <= UPDATE_UNDO_MS);
          if (old && old.status === 'superseded' && undoable) {
            const newer = newer0;
            ops.push({ store: 'records', put: { ...old, status: 'active', validTo: null, supersededBy: null, pinned: !!newer?.pinned || old.pinned } });
            if (newer) {
              const c2 = { ...c, records: new Map(c.records) };
              c2.records.set(old.id, { ...old, status: 'active' });
              const f = await forgetIds(c2, [newer.id], ops);
              await commit(ops, { forgot: f.hashes });
              scheduleSweep(f.purgeAt);
            } else await commit(ops);
            return { ok: true, status: 'saved', id: old.id, fact: old.fact };
          }
          fail(ERR.not_found);
          break;
        }
        case 'forgetConv': {
          checkGrant(c, ctx);
          const conv = str(args.conv, 40);
          if (!CONV_RE.test(conv)) fail(ERR.bad_args);
          const ids = activeOf(c).filter((r) => r.source?.conv === conv && (ctx.ui || r.source?.app === ctx.app)).map((r) => r.id);
          for (const p of c.proposals.values()) if (p.source?.conv === conv) ops.push({ store: 'proposals', del: p.id });
          const f = await forgetIds(c, ids, ops);
          if (ops.length) await commit(ops, { forgot: f.hashes });
          if (f.pending.length) scheduleSweep(f.purgeAt);
          return { pending: f.pending.length };
        }
        case 'grant': {
          if (ctx.ui) fail(ERR.readonly);
          const origin = originOf(ctx.app);
          const state = args.state === 'granted' ? 'granted' : args.state === 'denied' ? 'denied' : null;
          if (!origin || !state) fail(ERR.bad_args);
          ops.push({ store: 'grants', put: { origin, state, at: t } });
          await commit(ops);
          return { grant: state };
        }
        case 'touch': {
          // Bookkeeping of reads: allowed when this app may read, silently skipped otherwise.
          const s = settingsOf(c);
          if (!ctx.ui && (!s.on || !s.apps[ctx.app]?.read)) return {};
          const ids = Array.isArray(args.ids) ? args.ids.filter((x) => typeof x === 'string').slice(0, 64) : [];
          for (const id of ids) { const r = c.records.get(id); if (r && r.status === 'active' && (ctx.ui || (inScope(ctx.app, r) && wasServed(ctx.app, id)))) ops.push({ store: 'records', put: { ...r, lastUsedAt: t, useCount: (r.useCount || 0) + 1 } }); }
          if (ops.length) await commit(ops);
          return {};
        }
        // ── UI on the store origin ──
        case 'add': {
          const n = normalizeFact(args.fact, { now: t });
          if (!n.text) fail(ERR.bad_args);
          if (n.text.length > L.factMax) return { ok: false, code: ERR.too_long, n: n.text.length };
          if (!lintFact(n.text).ok) return { ok: false, code: ERR.not_a_fact };
          const res = await applyWrite(c, { fact: n.text, scope: scopeFor('manager', args.scope), importance: 3, eventAt: n.eventAt, expiresAt: n.expiresAt }, ops, { how: 'user-typed', app: 'manager', conv: null });
          if (res.ok) { await commit(ops); markPersist(); }
          return res;
        }
        case 'edit': {
          const r = c.records.get(str(args.id, 20));
          if (!r) fail(ERR.not_found);
          const n = normalizeFact(args.fact, { now: t });
          if (!n.text) fail(ERR.bad_args);
          if (n.text.length > L.factMax) return { ok: false, code: ERR.too_long, n: n.text.length };
          if (!lintFact(n.text).ok) return { ok: false, code: ERR.not_a_fact };
          const hash = await hashFact(n.text);
          const slot = slotOf(n.text);
          ops.push({ store: 'records', put: { ...r, fact: n.text, hash, slot, keys: keysFor(n.text, slot), kind: kindOf(n.text, slot, n.eventAt), eventAt: n.eventAt, expiresAt: n.expiresAt, updatedAt: t, tooLong: undefined, status: r.status === 'review' ? 'active' : r.status, source: { ...r.source, how: 'user-typed' } } });
          if (r.hash && r.hash !== hash) ops.push({ store: 'tombstones', put: { key: `h:${r.hash}`, at: t } });
          await commit(ops, { forgot: r.hash && r.hash !== hash ? [r.hash] : [] });
          // The old wording must not survive in the legacy store either.
          if (r.hash !== hash) dropLegacy([r.id]);
          return { ok: true, status: 'updated', id: r.id, fact: n.text };
        }
        case 'pin': {
          const r = c.records.get(str(args.id, 20));
          if (!r) fail(ERR.not_found);
          const want = !!args.pinned;
          if (want && !r.pinned && activeOf(c).filter((x) => x.pinned).length >= L.pinned) fail(ERR.full);
          ops.push({ store: 'records', put: { ...r, pinned: want, importance: want ? 5 : Math.min(r.importance, 4) } });
          await commit(ops);
          return { ok: true };
        }
        case 'setScope': {
          const r = c.records.get(str(args.id, 20));
          if (!r) fail(ERR.not_found);
          ops.push({ store: 'records', put: { ...r, scope: scopeFor('manager', args.scope), updatedAt: t } });
          await commit(ops);
          return { ok: true };
        }
        case 'keep': {
          const r = c.records.get(str(args.id, 20));
          if (!r || r.status !== 'review') fail(ERR.not_found);
          ops.push({ store: 'records', put: { ...r, status: 'active', updatedAt: t } });
          await commit(ops);
          return { ok: true };
        }
        case 'merge': {
          const ids = Array.isArray(args.ids) ? args.ids.map((x) => str(x, 20)) : [];
          const recs = ids.map((id) => c.records.get(id)).filter(Boolean);
          if (recs.length < 2) fail(ERR.bad_args);
          const n = normalizeFact(args.fact, { now: t });
          if (!n.text || n.text.length > L.factMax || !lintFact(n.text).ok) fail(ERR.bad_args);
          const f = await forgetIds(c, recs.map((r) => r.id), ops);
          const c2 = { ...c, records: new Map([...c.records].filter(([k]) => !f.pending.includes(k))) };
          const res = await applyWrite(c2, { fact: n.text, scope: recs[0].scope, importance: Math.max(...recs.map((r) => r.importance)) }, ops, { how: 'user-typed', app: 'manager', conv: null });
          await commit(ops, { forgot: f.hashes });
          scheduleSweep(f.purgeAt);
          return res;
        }
        // ── Manager only ──
        case 'settings': {
          const cur = settingsOf(c);
          const next = { on: args.on ?? cur.on, proposals: args.proposals ?? cur.proposals, apps: { ...cur.apps } };
          if (isObj(args.apps)) for (const [id, v] of Object.entries(args.apps)) if (next.apps[id] && isObj(v)) next.apps[id] = { read: v.read ?? next.apps[id].read, write: v.write ?? next.apps[id].write };
          ops.push({ store: 'meta', put: { key: 'settings', value: { on: !!next.on, proposals: !!next.proposals, apps: next.apps } } });
          await commit(ops);
          return settingsOf(cache);
        }
        case 'revoke': {
          const origin = str(args.origin, 200);
          if (!c.grants.has(origin)) fail(ERR.not_found);
          ops.push({ store: 'grants', del: origin });
          await commit(ops);
          return {};
        }
        case 'wipe': {
          const n = activeOf(c).length;
          const ts = [...c.records.values()].map((r) => r.hash).filter(Boolean);
          for (const name of ['records', 'proposals', 'pendingDelete']) ops.push({ store: name, clear: true });
          for (const r of c.records.values()) {
            ops.push({ store: 'tombstones', put: { key: `id:${r.id}`, at: t } });
            if (r.hash) ops.push({ store: 'tombstones', put: { key: `h:${r.hash}`, at: t } });
          }
          // Outbox replays queued before this moment are refused (see save).
          ops.push({ store: 'meta', put: { key: 'wipedAt', value: t } });
          await commit(ops, { forgot: ts, wiped: true });
          dropLegacy([...c.records.keys(), ...c.pending.keys()]);
          return { erased: n };
        }
        case 'purge': {
          await commit([...c.pending.values()].map((p) => ({ store: 'pendingDelete', del: p.id })));
          dropLegacy([...c.pending.keys()]);
          return {};
        }
        case 'export': {
          const records = [...c.records.values()].map((r) => ({ ...r }));
          return { format: 'memory-export', v: 2, namespace: cfg.namespace, exportedAt: t, records };
        }
        case 'import': {
          return importData(c, args, ops);
        }
        default: fail(ERR.bad_args);
      }
      return null;
    });
  }

  async function importData(c, args, ops) {
    const t = now();
    let data = args.data;
    if (typeof data === 'string') { try { data = JSON.parse(data); } catch { fail(ERR.bad_args); } }
    let rows;
    if (Array.isArray(data)) rows = data;
    else if (isObj(data) && data.format === 'memory-export' && Array.isArray(data.records)) rows = data.records;
    else if (isObj(data) && Array.isArray(data.memory)) rows = data.memory;
    else fail(ERR.bad_args);
    const plan = { add: 0, dup: 0, review: 0, invalid: 0, forgotten: 0, history: 0 };
    const work = { ...c, records: new Map(c.records) };
    const num = (x) => (Number.isFinite(x) ? x : null);
    const ID_RE = /^m[0-9a-z]{1,10}$/;
    const SCOPE_RE = /^(all|app:[a-z][a-z0-9-]{0,23}|project:[a-z0-9]{1,40})$/;
    // Replay in the order things happened, so a later value supersedes an
    // earlier one and never the other way round, whatever the file order.
    const when = (row) => num(row?.updatedAt) ?? num(row?.at) ?? 0;
    const ordered = rows.slice(0, L.active + L.history).map((row, i) => ({ row, i }))
      .sort((a, b) => (when(a.row) - when(b.row)) || (a.i - b.i)).map((x) => x.row);
    for (const row of ordered) {
      if (!isObj(row) || typeof row.fact !== 'string' || !row.fact.trim()) { plan.invalid++; continue; }
      const id = typeof row.id === 'string' && ID_RE.test(row.id) ? row.id : newId();
      const fact = row.fact.normalize('NFC').replace(/[\r\n]+/g, ' ').trim();
      const hash = await hashFact(fact);
      if (tombstoned(c, id, hash)) { plan.forgotten++; continue; }
      const scope = typeof row.scope === 'string' && SCOPE_RE.test(row.scope) ? row.scope : 'all';
      const eventAt = num(row.eventAt), expiresAt = num(row.expiresAt);
      // History (superseded, expired, or an event already past) stays history:
      // kept for «Pasado / Past», never active, never retrieved.
      // The lint applies on every write path, history included: a flagged or
      // over-long row is held for review whatever status the file claims.
      const flagged = !lintFact(fact).ok || fact.length > L.factMax || row.status === 'review';
      const past = !flagged && (row.status === 'superseded' || row.status === 'expired' || (expiresAt !== null && expiresAt < t));
      if (past) {
        if (work.records.has(id)) { plan.dup++; continue; }
        const status = row.status === 'superseded' ? 'superseded' : 'expired';
        const rec = makeRecord({
          id, fact, at: num(row.at) ?? t, now: num(row.updatedAt) ?? num(row.at) ?? t, scope,
          importance: [1, 2, 3, 4].includes(row.importance) ? row.importance : 3,
          source: { app: 'import', how: 'imported', conv: null }, eventAt, expiresAt,
          lang: detectLang(fact), hash, status,
        });
        rec.validTo = num(row.validTo) ?? expiresAt ?? t;
        if (typeof row.supersedes === 'string' && ID_RE.test(row.supersedes)) rec.supersedes = row.supersedes;
        if (typeof row.supersededBy === 'string' && ID_RE.test(row.supersededBy)) rec.supersededBy = row.supersededBy;
        plan.history++;
        if (!args.dryRun) ops.push({ store: 'records', put: rec });
        work.records.set(id, rec);
        continue;
      }
      if (work.records.has(id) || activeOf(work).some((r) => r.hash === hash)) { plan.dup++; continue; }
      if (flagged) {
        plan.review++;
        const rec = fromLegacy({ id, fact, at: Number.isFinite(row.at) ? row.at : t }, { hash });
        rec.status = 'review'; rec.source = { app: 'import', how: 'imported', conv: null };
        if (!args.dryRun) ops.push({ store: 'records', put: rec });
        work.records.set(id, rec);
        continue;
      }
      const sub = [];
      const res = await applyWrite(work, {
        id, at: num(row.at) ?? undefined, fact, scope, eventAt, expiresAt, pinned: row.pinned === true,
        importance: [1, 2, 3, 4, 5].includes(row.importance) ? row.importance : 3,
      }, sub, { how: 'imported', app: 'import', conv: null });
      if (!res.ok) { plan.invalid++; continue; }
      if (res.status === 'duplicate') { plan.dup++; continue; }
      plan.add++;
      for (const o of sub) {
        if (o.put && o.store === 'records') work.records.set(o.put.id, o.put);
        if (!args.dryRun) ops.push(o);
      }
    }
    if (!args.dryRun && ops.length) await commit(ops);
    return { dryRun: !!args.dryRun, ...plan };
  }

  // ── Migration from an older store ──────────────────────────────────────────
  async function migrate() {
    if (!legacy || !cfg.legacy || !backend.writable) return null;
    const c = await load();
    const prev = c.meta.get('migration');
    if (prev?.stopped) return prev;
    let rows;
    try { rows = await legacy.readAll(); } catch { return prev || null; }
    if (!rows) return prev || null;
    return write(async (c2) => {
      const t = now();
      const ops = [];
      const st = { count: 0, skipped: 0, invalid: 0, review: 0, at: prev?.at || t, lastNewAt: prev?.lastNewAt || t };
      const byHash = new Map();
      for (const r of c2.records.values()) if (r.hash) byHash.set(r.hash, r);
      for (const row of rows) {
        if (!row || typeof row.id !== 'string' || typeof row.fact !== 'string' || !row.fact.trim()) { st.invalid++; continue; }
        if (c2.records.has(row.id) || c2.pending.has(row.id)) { st.skipped++; continue; }
        const hash = await hashFact(row.fact);
        if (tombstoned(c2, row.id, hash)) { st.skipped++; continue; }
        const rec = fromLegacy(row, { hash });
        if (!rec) { st.invalid++; continue; }
        const dup = byHash.get(hash);
        if (dup) { rec.possibleDup = dup.id; ops.push({ store: 'records', put: { ...dup, possibleDup: rec.id } }); }
        byHash.set(hash, rec);
        if (rec.status === 'review') st.review++;
        ops.push({ store: 'records', put: rec });
        st.count++;
      }
      if (st.count) st.lastNewAt = t;
      st.total = (prev?.total || 0) + st.count;
      if (t - st.lastNewAt > L.catchUpDays * DAY) st.stopped = true;
      ops.push({ store: 'meta', put: { key: 'migration', value: st } });
      await commit(ops);
      return st;
    });
  }

  // ── Timers and persistence ─────────────────────────────────────────────────
  let sweepTimer = null;
  function scheduleSweep(at) {
    if (sweepTimer) clearTimeout(sweepTimer);
    sweepTimer = setTimeout(() => { sweepTimer = null; write(async () => {}).catch(() => {}); }, Math.max(0, at - now()) + 50);
  }
  let persistAsked = false;
  function markPersist() {
    if (persistAsked || !opts.persist) return;
    persistAsked = true;
    Promise.resolve().then(() => opts.persist()).then((ok) => {
      if (ok) write(async () => { await commit([{ store: 'meta', put: { key: 'persisted', value: true } }]); }).catch(() => {});
      else persistAsked = false;
    }).catch(() => { persistAsked = false; });
  }

  // First open: schema and epoch; then purge what is due.
  if (backend.writable) {
    await withLock(async () => {
      const c = await load(true);
      const ops = [];
      if (!c.meta.get('schema')) ops.push({ store: 'meta', put: { key: 'schema', value: SCHEMA } });
      if (!c.meta.get('created')) ops.push({ store: 'meta', put: { key: 'created', value: now() } });
      const maint = maintenanceOps(c);
      ops.push(...maint);
      if (ops.length) await commit(ops);
      dropLegacy(purgedIds(maint));
    });
  } else await load(true);

  const store = {
    cfg,
    backend,
    // `ctx`: { app } for an embedding app (set by serveStore from e.origin),
    // { app, direct: true } for an app on the store origin, { ui: true } for
    // the panel, { ui: true, manager: true } for the top-level manager.
    async call(op, args = {}, ctx = {}) {
      try {
        if (typeof op !== 'string') fail(ERR.bad_args);
        if (!isObj(args)) fail(ERR.bad_args);
        const c = ctx.ui ? { ...ctx, app: 'manager' } : ctx;
        if (!c.ui && !cfg.apps.some((a) => a.id === c.app)) fail(ERR.bad_args);
        if (MANAGER_OPS.has(op) && !c.manager) fail(ERR.readonly);
        if (UI_OPS.has(op) && !c.ui) fail(ERR.readonly);
        if (!APP_OPS.has(op) && !UI_OPS.has(op) && !MANAGER_OPS.has(op)) fail(ERR.bad_args);
        if (!c.ui) {
          if (READ_OPS.has(op)) rate(c.app, 'read', L.readsPerMinute);
          if (WRITE_OPS.has(op)) await rateShared(c.app, 'write', L.writesPerMinute, L.writesPerDay);
          if (FORGET_OPS.has(op)) await rateShared(c.app, 'forget', L.forgetsPerMinute);
        }
        const result = await handle(op, args, c);
        return { ok: true, result };
      } catch (e) {
        if (e instanceof OpError) return { ok: false, error: { code: e.code, message: e.code, ...(e.n ? { n: e.n } : {}) } };
        return { ok: false, error: { code: ERR.unavailable, message: 'unavailable' } };
      }
    },
    onChange(f) { listeners.add(f); return () => listeners.delete(f); },
    migrate,
    askPersist() { markPersist(); },
    async sweep() { return write(async () => {}); },
    rid(app) { if (!recentRids.has(app)) recentRids.set(app, new Map()); return recentRids.get(app); },
    async close() { try { channel?.close(); } catch { /* — */ } if (sweepTimer) clearTimeout(sweepTimer); await backend.close(); },
  };
  return store;
}

// ── Serving embedders ────────────────────────────────────────────────────────

// Installs the message router of a store frame. Answers only:
//   1. exact allowed origins (cfg.apps), by string equality;
//   2. messages whose source is this frame's parent;
//   3. when the parent is the top-level page (no nested or generated frames);
//   4. well-formed envelopes of this namespace within the size limit.
// Anything else is dropped without a reply. Replies go to e.source with
// target e.origin, never '*'. The app id comes from e.origin, never from the
// message.
export function serveStore(win, cfg, store, { engine = null } = {}) {
  assertConfig(cfg);
  const ALLOW = new Map(cfg.apps.map((a) => [a.origin, a.id]));
  let subscribed = null;   // { origin }
  const off = store.onChange((ev) => {
    if (!subscribed) return;
    const msg = { ns: cfg.namespace, v: 2, event: 'changed', rev: ev.rev, ...(ev.forgot ? { forgot: ev.forgot } : {}), ...(ev.wiped ? { wiped: true } : {}) };
    try { win.parent.postMessage(msg, subscribed.origin); } catch { /* parent gone */ }
  });
  async function onMessage(e) {
    const app = ALLOW.get(e.origin);
    if (!app) return;
    if (e.source !== win.parent) return;
    if (win.parent !== win.top) return;
    const d = e.data;
    if (!isObj(d) || d.ns !== cfg.namespace || d.v !== 2 || typeof d.rid !== 'string' || !d.rid || d.rid.length > 40 || typeof d.op !== 'string') return;
    let size = 0;
    try { size = JSON.stringify(d).length; } catch { return; }
    if (size > cfg.limits.message) return;
    const args = isObj(d.args) ? d.args : {};
    const cache = store.rid(app);
    let p = cache.get(d.rid);
    if (!p) {
      p = store.call(d.op, args, { app });
      cache.set(d.rid, p);
      while (cache.size > cfg.limits.ridKeep) cache.delete(cache.keys().next().value);
    }
    const r = await p;
    if (d.op === 'subscribe' && r.ok) subscribed = { origin: e.origin };
    const reply = r.ok ? { ns: cfg.namespace, v: 2, rid: d.rid, ok: true, result: r.result } : { ns: cfg.namespace, v: 2, rid: d.rid, ok: false, error: r.error };
    if (d.op === 'hello' && r.ok && engine) reply.result.engine = engine;
    try { e.source.postMessage(reply, e.origin); } catch { /* source gone */ }
  }
  win.addEventListener('message', onMessage);
  return () => { win.removeEventListener('message', onMessage); off(); };
}

export { fromLegacy, validateRecord };
