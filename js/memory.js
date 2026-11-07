// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
//
// User memory: the client every app uses.
//
// One memory per browser, shared by every app of a brand: the apps never open
// the memory database themselves. They talk to the store page through a hidden
// iframe (or, for an app that shares the store page's origin, through the same
// store code in-process). This module gives them:
//
//   - `createMemory(cfg, { app, lang })`: the connection, with retries, an
//     honest "off" state and a small outbox for explicit saves made while the
//     store is unreachable;
//   - `memoryTools`, `runMemoryTool`: the three tools the model sees;
//   - `memoryPromptRules`: the rule lines for the system head;
//   - `cardFor`: what the UI shows under an answer;
//   - `typedEvidence`, `maybePropose`, `basicIntent`: helpers for the turn loop.
//
// The client runs the same write gate as the store, so a save the store would
// refuse is refused here without a round trip, and the store runs it again.

import { gateWrite, proposeFromUser, newRid, hashFact, LIMITS, renderRules } from './memory-core.js';
import { assertConfig, appLabel } from './memory-config.js';
import { sharedTuning, tuningFor } from './memory-tuning.js';
import { t, formatWhen } from './memory-text.js';
import { idbConnection } from './memory-store.js';

// Transport timeouts, in milliseconds. Code constants, not tunables.
export const READY_TIMEOUT = 8000;
export const CALL_TIMEOUT = 10000;
export const HELLO_INTERVAL = 250;

const isObj = (x) => x && typeof x === 'object' && !Array.isArray(x);

// ── Outbox storage (the app's own origin) ────────────────────────────────────

export function memoryOutbox() {
  const m = new Map();
  return {
    async all() { return [...m.values()].map((x) => structuredClone(x)); },
    async put(x) { m.set(x.id, structuredClone(x)); },
    async del(id) { m.delete(id); },
  };
}

export function idbOutbox(name, idb = globalThis.indexedDB) {
  if (!idb) return memoryOutbox();
  // The connection is reopened if the browser closes it (see idbConnection).
  const conn = idbConnection(() => new Promise((ok, ko) => {
    const r = idb.open(name, 1);
    r.onupgradeneeded = () => { if (!r.result.objectStoreNames.contains('outbox')) r.result.createObjectStore('outbox', { keyPath: 'id' }); };
    r.onsuccess = () => ok(r.result);
    r.onerror = () => ko(r.error);
  }));
  const run = (mode, fn) => conn.tx((d) => new Promise((ok, ko) => {
    const tx = d.transaction('outbox', mode);
    const res = fn(tx.objectStore('outbox'));
    tx.oncomplete = () => ok(res?.result);
    tx.onerror = () => ko(tx.error);
    tx.onabort = () => ko(tx.error || new Error('aborted'));
  }));
  return {
    all: () => run('readonly', (s) => s.getAll()).catch(() => []),
    put: (x) => run('readwrite', (s) => s.put(x)),
    del: (id) => run('readwrite', (s) => s.delete(id)),
  };
}

// ── The client ───────────────────────────────────────────────────────────────

export function createMemory(cfg, {
  app, lang = 'es', direct = null, slowNetwork = false,
  doc = globalThis.document, win = globalThis.window, outbox = null, now = () => Date.now(),
} = {}) {
  assertConfig(cfg);
  if (!cfg.apps.some((a) => a.id === app)) throw new Error(`memory: app "${app}" is not in the config`);
  const L = cfg.limits;
  const storeOrigin = cfg.storeOrigin;
  const slow = slowNetwork ? 2 : 1;
  const box = outbox || idbOutbox(`${cfg.namespace}-outbox`);
  const listeners = new Set();
  let frame = null;
  let frameDead = false;           // the frame stopped answering: replace it on the next connect
  let listening = false;
  let mode = 'unknown';            // 'store' | 'direct' | 'off' | 'unknown'
  let hello = null;                // last hello result
  let readyP = null;
  const waiting = new Map();       // rid -> { resolve }

  const emit = (ev) => { for (const f of listeners) { try { f(ev); } catch { /* listener */ } } };

  async function onChanged(ev) {
    if (ev.wiped) { for (const e of await box.all()) await box.del(e.id); }
    else if (Array.isArray(ev.forgot) && ev.forgot.length) {
      for (const e of await box.all()) if (ev.forgot.includes(e.hash)) await box.del(e.id);
    }
    emit(ev);
  }

  function onMessage(e) {
    if (!frame || e.origin !== storeOrigin || e.source !== frame.contentWindow) return;
    const d = e.data;
    if (!isObj(d) || d.ns !== cfg.namespace || d.v !== 2) return;
    if (d.event === 'changed') { onChanged(d).catch(() => {}); return; }
    const w = waiting.get(d.rid);
    if (w) { waiting.delete(d.rid); w.resolve(d); }
  }

  // A frame that loaded an error page (offline, 404, blocked) never runs the
  // store code, so retrying hello on it is useless: it is replaced instead.
  function ensureFrame() {
    if (frame && frameDead) { try { frame.remove(); } catch { /* already gone */ } frame = null; }
    frameDead = false;
    if (frame || !doc) return frame;
    frame = doc.createElement('iframe');
    frame.src = cfg.storeUrl;
    frame.title = t(cfg.strings, lang, 'frameTitle');
    frame.setAttribute('aria-hidden', 'true');
    frame.setAttribute('tabindex', '-1');
    frame.setAttribute('referrerpolicy', 'no-referrer');
    frame.style.cssText = 'position:absolute;width:0;height:0;border:0;visibility:hidden';
    if (!listening) { win.addEventListener('message', onMessage); listening = true; }
    doc.body.appendChild(frame);
    return frame;
  }

  const post = (op, args, rid) => {
    frame.contentWindow.postMessage({ ns: cfg.namespace, v: 2, rid, op, args }, storeOrigin);
  };
  function once(op, args, rid, ms) {
    return new Promise((resolve) => {
      const timer = setTimeout(() => { waiting.delete(rid); resolve(null); }, ms);
      waiting.set(rid, { resolve: (d) => { clearTimeout(timer); resolve(d); } });
      try { post(op, args, rid); } catch { clearTimeout(timer); waiting.delete(rid); resolve(null); }
    });
  }

  async function raw(op, args) {
    if (direct) {
      const r = await direct.call(op, args, { app, direct: true });
      return r;
    }
    const rid = newRid();
    let d = await once(op, args, rid, CALL_TIMEOUT * slow);
    if (!d) d = await once(op, args, rid, CALL_TIMEOUT * slow);   // same rid: the store returns the stored result
    if (!d) { goOff(); return { ok: false, error: { code: 'unavailable' } }; }
    return d;
  }

  // Off until the next turn (`context`), `online`, focus or `retry()`: later
  // calls in the same turn fail fast instead of waiting for hello again.
  const OFF = () => ({ mode: 'off', count: 0, grant: null, enabled: { read: false, write: false } });
  function goOff(extra = {}) {
    mode = 'off';
    frameDead = true;
    const r = { ...OFF(), ...extra };
    readyP = Promise.resolve(r);
    return r;
  }

  async function connect() {
    if (direct) {
      const r = await direct.call('hello', { client: 2, engine: null }, { app, direct: true });
      if (!r.ok) { mode = 'off'; return { mode, count: 0, grant: null, enabled: { read: false, write: false } }; }
      hello = r.result; mode = 'direct';
      if (!direct.__memSub) { direct.__memSub = true; direct.onChange((ev) => { onChanged(ev).catch(() => {}); }); }
      flush().catch(() => {});
      return { mode, ...pick(hello) };
    }
    if (!doc) { mode = 'off'; return { mode, count: 0, grant: null, enabled: { read: false, write: false } }; }
    ensureFrame();
    const rid = newRid();
    // The budget counts only time the page was free to hear the answer. The store
    // shares this page's process: while the page is busy (loading a model, for
    // example) a hello can come back seconds late through no fault of the store,
    // and a wall-clock deadline gave up just before it answered. Each attempt is
    // charged at most two intervals, however long it really took.
    let gastado = 0;
    let d = null;
    while (!d && gastado < READY_TIMEOUT * slow) {
      const t0 = now();
      d = await once('hello', { client: 2, engine: null }, rid, HELLO_INTERVAL);
      gastado += Math.min(Math.max(0, now() - t0), 2 * HELLO_INTERVAL);
    }
    if (!d || !d.ok) return goOff();
    hello = d.result;
    if ((hello.minClient || 2) > 2) { const r = goOff({ outdated: true }); frameDead = false; return r; }
    mode = 'store';
    raw('subscribe', {}).catch(() => {});
    flush().catch(() => {});
    return { mode, ...pick(hello) };
  }
  const pick = (h) => ({ count: h.count, grant: h.grant, enabled: h.enabled, rev: h.rev, epoch: h.epoch, persisted: h.persisted, storage: h.storage });
  const storageBlocked = () => hello?.storage === 'blocked';

  function ready() {
    if (!readyP) readyP = connect();
    return readyP;
  }
  const newTurn = () => { if (mode === 'off') readyP = null; };
  // After a failure the client stays off until the next turn, `online` or focus.
  if (win && !direct) {
    win.addEventListener?.('online', newTurn);
    win.addEventListener?.('focus', newTurn);
  }

  async function call(op, args) {
    const r = await ready();
    if (r.mode === 'off') return { ok: false, error: { code: 'unavailable' } };
    return raw(op, args);
  }
  const res = (r) => (r.ok ? r.result : { ok: false, code: r.error?.code || 'unavailable', n: r.error?.n });

  // Replays queued saves in order. Each carries the time it was queued, so the
  // store refuses one that an erase-all or a forget made after that time has
  // overruled, even if this app was closed when it happened. A save waiting
  // for this app's grant stays queued and the app is told to ask for it.
  let flushing = null;
  function flush() {
    if (!flushing) flushing = doFlush().finally(() => { flushing = null; });
    return flushing;
  }
  async function doFlush() {
    if (storageBlocked()) return;
    const items = (await box.all()).sort((a, b) => a.at - b.at);
    let needGrant = false;
    for (const e of items) {
      if (now() - e.at > L.outboxDays * 864e5) { await box.del(e.id); continue; }
      const r = await raw('save', { fact: e.fact, evidence: e.evidence, tainted: false, conv: e.conv, scope: e.scope, importance: e.importance, lang: e.lang, queuedAt: e.at });
      const c = r.ok ? (r.result?.ok === false ? r.result.code : null) : r.error?.code;
      if (c === 'no_grant') { needGrant = true; break; }
      if (c === 'unavailable' || c === 'rate') break;
      await box.del(e.id);
    }
    emit({ outbox: (await box.all()).length, ...(needGrant ? { needGrant: true } : {}) });
  }

  const api = {
    get mode() { return mode; },
    get hello() { return hello; },
    ready,
    async context({ query = '', prevQuery = '', model = {}, scopes = null, conv = null, incognito = false } = {}) {
      const empty = { text: '', tokens: 0, ids: [], abstained: true };
      if (incognito) return empty;
      newTurn();
      const r = await call('context', { query, prevQuery, model: { id: model?.id ?? null, ctx: model?.ctx ?? null }, lang, scopes, conv });
      return r.ok ? r.result : { ...empty, unavailable: true };
    },
    async save(fact, { evidence = null, tainted = true, conv = null, scope = null, importance = 3, replaces = null } = {}) {
      const g = gateWrite({ fact, evidence, tainted, lang, shared: sharedTuning().memory.write, now: now(), strings: cfg.strings, limits: L });
      if (!g.ok) return { ok: false, code: g.code, n: g.n };
      const args = { fact: g.fact, evidence, tainted: tainted === false ? false : true, conv, scope, importance, replaces, lang };
      // The store's browser does not allow saving: say so, never queue.
      if (storageBlocked()) return { ok: false, code: 'unavailable', blocked: true };
      const r = await call('save', args);
      if (storageBlocked()) return { ok: false, code: 'unavailable', blocked: true };
      if (!r.ok && r.error?.code === 'unavailable' && g.status === 'active' && tainted === false) {
        const items = await box.all();
        if (items.length >= L.outboxMax) return { ok: false, code: 'unavailable', queued: items.length, dropped: true };
        await box.put({ id: newRid(), fact: g.fact, evidence, conv, scope, importance, lang, hash: await hashFact(g.fact), at: now() });
        return { ok: false, code: 'unavailable', queued: items.length + 1 };
      }
      return res(r);
    },
    async propose(p) { return res(await call('propose', p)); },
    async confirm(pid, fact) { return res(await call('confirm', { pid, ...(fact ? { fact } : {}) })); },
    async dismiss(pid, never = false) { return res(await call('dismiss', { pid, never })); },
    async resolve(ref, scopes) { return res(await call('resolve', { ref, scopes })); },
    async forget(id) { return res(await call('forget', { id })); },
    async restore(id) { return res(await call('restore', { id })); },
    async forgetConversation(conv) { return res(await call('forgetConv', { conv })); },
    async byConversation(conv) { return res(await call('byConv', { conv })); },
    async search(query, limit = 10, scopes = null) { return res(await call('search', { query, limit, scopes })); },
    async list(cursor = null, scopes = null) { return res(await call('list', { cursor, scopes })); },
    async touch(ids) { return res(await call('touch', { ids })); },
    async setGrant(state) {
      const r = res(await call('grant', { state }));
      if (r && r.grant === 'granted') await flush().catch(() => {});
      return r;
    },
    onChange(f) { listeners.add(f); return () => listeners.delete(f); },
    async outboxSize() { return (await box.all()).length; },
    async retry() { readyP = null; mode = 'unknown'; return ready(); },
    close() {
      if (listening) { win.removeEventListener('message', onMessage); listening = false; }
      if (frame) { frame.remove(); frame = null; }
      listeners.clear();
    },
    cfg, app, lang,
  };
  return api;
}

// ── Turn helpers ─────────────────────────────────────────────────────────────

// Evidence for the write gate from the conversation. Only messages the user
// TYPED count (`typed === true`): pasted, dropped or uploaded text, attachments
// and anything longer than the typed limit never ground a fact.
export function typedEvidence(messages) {
  const typed = (messages || []).filter((m) => m && m.role === 'user' && m.typed === true &&
    typeof m.content === 'string' && m.content.length <= LIMITS.typedMax);
  const last = [...(messages || [])].reverse().find((m) => m && m.role === 'user');
  const current = last && last.typed === true && typeof last.content === 'string' && last.content.length <= LIMITS.typedMax ? last.content : '';
  return { current, typedTexts: typed.slice(-3).map((m) => m.content), trigger: undefined };
}

// A turn is tainted as soon as the result of ANY tool other than the memory
// tools enters the context. Apps call this with every tool name they ran.
export function taints(cfg, toolName) {
  return !String(toolName || '').startsWith(cfg.tools.prefix);
}

// Basic mode (no model): «recuerda (que) X», «qué recuerdas», «olvida X».
export function basicIntent(text) {
  const s = String(text || '').trim();
  let m = /^(?:recuerda|acuérdate de|acuerdate de|apunta|remember)(?:\s+(?:que|that))?\s+(.+)$/iu.exec(s);
  if (m) return { op: 'save', fact: m[1] };
  if (/^(?:¿\s*)?(?:qu[eé] recuerdas|qu[eé] sabes de m[ií]|what do you remember|what do you know about me)\s*\??$/iu.test(s)) return { op: 'list' };
  m = /^(?:olvida|olvídate de|olvidate de|forget)(?:\s+(?:que|that))?\s+(.+)$/iu.exec(s);
  if (m) return { op: 'forget', query: m[1] };
  return null;
}

// The deterministic proposer, once per user turn after the final reply, within
// the model's tuning limits. `state` is per conversation: { proposed: n }.
export async function maybePropose(mem, text, { lang = 'es', conv = null, model = null, state = {}, incognito = false, scope = null } = {}) {
  if (incognito || !mem.cfg.features.proposals) return null;
  const tun = tuningFor(model?.id ?? null, model?.ctx ?? null).memory.proposer;
  state.proposed = state.proposed || 0;
  if (tun.perTurn < 1 || state.proposed >= tun.perConv) return null;
  const p = proposeFromUser(text, lang, { strings: mem.cfg.strings });
  if (!p) return null;
  const r = await mem.propose({ fact: p.fact, kind: p.kind, slot: p.slot, conv, label: 'from-user', ...(scope ? { scope } : {}) });
  if (r && r.pid) { state.proposed++; return { type: 'proposal', pid: r.pid, fact: p.fact, label: 'from-user' }; }
  return null;
}

// ── Tools ────────────────────────────────────────────────────────────────────

// Tool definitions for `cfg.tools.expose` only, in the app's language. Each
// entry carries both shapes apps use: `desc` + `params` (name -> description)
// and a JSON schema in `parameters`.
export function memoryTools(cfg, lang = 'es', { app = null } = {}) {
  assertConfig(cfg);
  const S = (k) => t(cfg.strings, lang, k);
  const project = app && cfg.features.projectScope.includes(app);
  const defs = {
    save: {
      desc: S('tool.save'),
      params: { fact: S('tool.save.fact'), importance: S('tool.save.importance'), replaces: S('tool.save.replaces'), ...(project ? { scope: S('tool.save.scope') } : {}) },
      parameters: { type: 'object', required: ['fact'], properties: {
        fact: { type: 'string', maxLength: 300, description: S('tool.save.fact') },
        importance: { type: 'integer', minimum: 1, maximum: 4, description: S('tool.save.importance') },
        replaces: { type: 'string', description: S('tool.save.replaces') },
        ...(project ? { scope: { type: 'string', enum: ['all', 'project'], description: S('tool.save.scope') } } : {}),
      } },
    },
    list: {
      desc: S('tool.list'),
      params: { query: S('tool.list.query'), limit: S('tool.list.limit') },
      parameters: { type: 'object', properties: {
        query: { type: 'string', description: S('tool.list.query') },
        limit: { type: 'integer', minimum: 1, maximum: 20, description: S('tool.list.limit') },
      } },
    },
    forget: {
      desc: S('tool.forget'),
      params: { id: S('tool.forget.id'), query: S('tool.forget.query') },
      parameters: { type: 'object', properties: {
        id: { type: 'string', description: S('tool.forget.id') },
        query: { type: 'string', description: S('tool.forget.query') },
      } },
    },
  };
  const out = {};
  for (const k of cfg.tools.expose) out[cfg.tools.prefix + k] = { name: cfg.tools.prefix + k, ...defs[k] };
  return out;
}

export function memoryPromptRules(cfg, lang = 'es', variant = 'v1') {
  assertConfig(cfg);
  return renderRules(cfg.strings, lang, variant, cfg.tools.prefix).map((l, i) => `${i + 1}. ${l}`).join('\n');
}

const short = (id) => String(id || '').slice(0, 6);

// Runs one memory tool call. `turn` carries what only the app knows:
//   { evidence, tainted, conv, incognito, scope, project, forgetCards }
// Returns { text } for the model and, when the user has to see or decide
// something, { card } for the UI. The text is always the truth.
export async function runMemoryTool(mem, name, args = {}, turn = {}) {
  const cfg = mem.cfg, lang = mem.lang;
  const S = (k, v) => t(cfg.strings, lang, k, v);
  const op = String(name || '').startsWith(cfg.tools.prefix) ? name.slice(cfg.tools.prefix.length) : name;
  if (!cfg.tools.expose.includes(op)) return { text: S('res.error', { code: 'bad_args' }) };
  if (turn.incognito) return { text: S('res.incognito') };
  const a = isObj(args) ? args : {};

  if (op === 'save') {
    let scope = null;
    if (a.scope === 'project' && turn.project && cfg.features.projectScope.includes(mem.app)) scope = `project:${turn.project}`;
    else if (turn.scope) scope = turn.scope;
    const r = await mem.save(typeof a.fact === 'string' ? a.fact : '', {
      evidence: turn.evidence, tainted: turn.tainted === false ? false : true, conv: turn.conv || null,
      scope, importance: Math.min(4, Math.max(1, a.importance | 0 || 3)), replaces: typeof a.replaces === 'string' ? a.replaces : null,
    });
    return { text: resultText(r, lang, cfg), card: cardFor(r, lang, cfg) };
  }

  if (op === 'list') {
    const limit = Math.min(20, Math.max(1, a.limit | 0 || 20));
    const scopes = turn.scopes || null;
    const r = typeof a.query === 'string' && a.query.trim()
      ? await mem.search(a.query, limit, scopes)
      : await mem.search('', limit, scopes);
    if (r.ok === false) return { text: S(r.code === 'unavailable' ? 'res.unavailable_plain' : 'res.error', { code: r.code }) };
    if (!r.total) return { text: S('res.empty') };
    const lines = (r.items || []).map((x) => `[${short(x.id)}] ${formatWhen(x.updatedAt || x.at, lang, cfg.strings)} · ${x.fact}`);
    return { text: `${S('res.count', { n: r.total })}\n` + JSON.stringify({ data_from_memory: lines, note: S('dataNote') }) };
  }

  if (op === 'forget') {
    turn.forgetCards = (turn.forgetCards || 0) + 1;
    if (turn.forgetCards > LIMITS.forgetCardsPerTurn) return { text: S('res.forget_limit') };
    const ref = typeof a.id === 'string' && a.id.trim() ? a.id : typeof a.query === 'string' ? a.query : '';
    if (!ref.trim()) return { text: S('res.not_found') };
    const r = await mem.resolve(ref, turn.scopes || null);
    if (r.ok === false) {
      return { text: r.code === 'not_found' ? S('res.not_found') : r.code === 'unavailable' ? S('res.unavailable_plain') : S('res.error', { code: r.code }) };
    }
    if (r.ambiguous) return { text: S('res.ambiguous'), card: { type: 'ambiguous', candidates: r.candidates.map((x) => ({ id: x.id, fact: x.fact })) } };
    // Nothing is deleted here: the card asks, the user's click forgets.
    return { text: S('res.forget_card'), card: { type: 'forget', id: r.record.id, fact: r.record.fact } };
  }
  return { text: S('res.error', { code: 'bad_args' }) };
}

export function resultText(r, lang, cfg) {
  const S = (k, v) => t(cfg.strings, lang, k, v);
  if (!r) return S('res.unavailable_plain');
  if (r.ok === false) {
    switch (r.code) {
      case 'too_long': return S('res.too_long', { n: r.n });
      case 'not_a_fact': return S('res.not_a_fact');
      case 'not_from_user': return S('res.not_from_user');
      case 'unavailable': return r.queued ? S('res.unavailable') : S('res.unavailable_plain');
      case 'no_grant': case 'disabled': return S('res.no_grant');
      case 'full': return S('res.full');
      case 'rate': return S('res.rate');
      case 'incognito': return S('res.incognito');
      default: return S('res.error', { code: r.code });
    }
  }
  switch (r.status) {
    case 'saved': return S('res.saved', { fact: r.fact });
    case 'duplicate': return S('res.duplicate', { fact: r.fact });
    case 'updated': return r.negated ? S('card.negated', { fact: r.previous }) : S('res.updated', { prev: r.previous, fact: r.fact });
    case 'proposed': return S('res.proposed');
    case 'similar': return S('res.similar', { prev: r.previous });
    default: return S('res.saved', { fact: r.fact });
  }
}

// The card model for a write result (memory-cards.js renders it).
export function cardFor(r, lang, cfg) {
  if (!r) return null;
  if (r.ok === false) {
    if (r.code === 'no_grant') return { type: 'grant' };
    if (r.code === 'unavailable' && r.blocked) return { type: 'error', text: t(cfg.strings, lang, 'fail.blocked') };
    if (r.code === 'unavailable') return r.queued ? { type: 'queued', n: r.queued } : { type: 'unavailable' };
    if (r.code === 'too_long') return { type: 'error', text: t(cfg.strings, lang, 'fail.tooLong', { n: r.n }) };
    if (r.code === 'full') return { type: 'error', text: t(cfg.strings, lang, 'fail.full') };
    return null;
  }
  switch (r.status) {
    case 'saved': return { type: 'saved', id: r.id, fact: r.fact };
    case 'duplicate': return { type: 'duplicate', id: r.id, fact: r.fact };
    case 'updated': return r.negated ? { type: 'negated', id: r.id, fact: r.previous } : { type: 'updated', id: r.id, fact: r.fact, prev: r.previous, prevId: r.previousId };
    case 'proposed': return { type: 'proposal', pid: r.pid, fact: r.fact, label: r.label || 'from-content' };
    case 'similar': return { type: 'similar', id: r.id, fact: r.fact, prev: r.previous };
    default: return null;
  }
}

export { appLabel };
export { todayLine } from './memory-text.js';
