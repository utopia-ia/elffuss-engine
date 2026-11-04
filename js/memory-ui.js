// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
//
// User memory: the store page. A brand hosts a page of a few lines that calls
// `mountStorePage(cfg)`; this module decides what that page is:
//
//   - embedded, `?view=panel`: the visible memory panel, and it serves ops;
//   - embedded otherwise: headless, it only serves ops to its parent app;
//   - top level: the full memory manager, with no message router at all.
//
// The panel and the manager run the store in-process (same origin) and are the
// only places where facts are edited, pinned or merged, and the manager the
// only place for export, import, erase-all, settings and revoking apps.
// Every fact is shown with `textContent`.

import { openStore, serveStore } from './memory-store.js';
import { t, formatWhen, describeSource } from './memory-text.js';
import { isExclusive } from './memory-core.js';
import { assertConfig, appLabel } from './memory-config.js';

const STYLE = `
.mem-root{font-family:var(--mem-font,system-ui,sans-serif);color:var(--mem-fg,CanvasText);background:var(--mem-bg,Canvas);
  margin:0;padding:1rem;box-sizing:border-box;max-width:48rem;font-size:15px;line-height:1.45}
.mem-root *{box-sizing:border-box}
.mem-root h1{font-size:1.25rem;margin:0 0 .5rem}
.mem-root h2{font-size:.95rem;margin:1.25rem 0 .4rem;color:var(--mem-muted,GrayText);font-weight:600}
.mem-root .mem-head{color:var(--mem-muted,GrayText);margin:0 0 .75rem}
.mem-root ul{list-style:none;margin:0;padding:0}
.mem-root li.mem-item{border:1px solid var(--mem-border,GrayText);border-radius:var(--mem-radius,.5rem);padding:.5rem .75rem;margin:.4rem 0;overflow-wrap:anywhere}
.mem-root .mem-fact{display:block}
.mem-root .mem-meta{color:var(--mem-muted,GrayText);font-size:.85em}
.mem-root .mem-actions{display:flex;flex-wrap:wrap;gap:.35rem;margin-top:.35rem}
.mem-root button{font:inherit;font-size:.85em;padding:.25em .7em;border-radius:var(--mem-radius,.5rem);border:1px solid var(--mem-border,GrayText);
  background:transparent;color:var(--mem-fg,CanvasText);cursor:pointer}
.mem-root button.primary{background:var(--mem-accent,AccentColor);color:var(--mem-accent-fg,AccentColorText);border-color:var(--mem-accent,AccentColor)}
.mem-root button.danger{color:var(--mem-danger,crimson)}
.mem-root input[type=text],.mem-root input[type=search],.mem-root select{font:inherit;padding:.35em .5em;border:1px solid var(--mem-border,GrayText);
  border-radius:var(--mem-radius,.5rem);background:var(--mem-bg,Canvas);color:var(--mem-fg,CanvasText);width:100%}
.mem-root .mem-row{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap;margin:.35rem 0}
.mem-root .mem-row>input[type=text]{flex:1 1 12rem}
.mem-root .mem-empty{color:var(--mem-muted,GrayText);padding:1rem 0}
.mem-root .mem-warn{color:var(--mem-danger,crimson)}
.mem-root a{color:var(--mem-accent,AccentColor)}
@media (max-width:420px){.mem-root{padding:.75rem}.mem-root .mem-actions button{flex:1 1 auto}}
`;

function h(doc, tag, props = {}, kids = []) {
  const el = doc.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'text') el.textContent = v;
    else if (k === 'class') el.className = v;
    else if (k === 'on') for (const [ev, f] of Object.entries(v)) el.addEventListener(ev, f);
    else if (v !== false && v !== null && v !== undefined) el.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids) if (c) el.append(c);
  return el;
}

// Record ids are store-made ([A-Za-z0-9_-]); anything else is dropped before
// it reaches a selector.
const CSS_ID = (id) => String(id).replace(/[^A-Za-z0-9_-]/g, '');

function applyTheme(el, cfg, scheme) {
  const set = { ...(cfg.theme?.light || {}), ...(scheme === 'dark' ? cfg.theme?.dark || {} : {}) };
  for (const [k, v] of Object.entries(set)) el.style.setProperty(k, v);
}

// Picks the mode, opens the store, runs the migration and mounts what the mode
// needs. Returns { mode, store, unmount }.
export async function mountStorePage(cfg, { doc = globalThis.document, win = globalThis.window, engine = null } = {}) {
  assertConfig(cfg);
  const params = new URL(win.location.href).searchParams;
  const embedded = win.top !== win;
  const mode = embedded ? (params.get('view') === 'panel' ? 'panel' : 'headless') : 'manager';
  const lang = (params.get('lang') || (doc.documentElement.lang || 'es')).slice(0, 10);
  const scheme = params.get('scheme') === 'dark' ? 'dark' : 'light';
  const persist = async () => { try { return await win.navigator?.storage?.persist?.(); } catch { return false; } };
  const store = await openStore(cfg, { persist, engine });
  // The router is installed before the migration so apps get answers at once;
  // the migration runs inside the write lock and announces its changes.
  let off = null;
  if (mode !== 'manager') off = serveStore(win, cfg, store, { engine });
  store.migrate().catch(() => {});
  let ui = null;
  if (mode !== 'headless') {
    const root = doc.getElementById('root') || doc.body;
    ui = mountPanel(root, store, { cfg, lang, scheme, manager: mode === 'manager', doc });
    if (mode === 'manager') store.askPersist();
  }
  return { mode, store, unmount: () => { off?.(); ui?.unmount(); } };
}

// The panel (and, with `manager: true`, the manager). Runs in the store page.
export function mountPanel(el, store, { cfg = store.cfg, lang = 'es', scheme = 'light', manager = false, doc = el.ownerDocument } = {}) {
  const S = (k, v) => t(cfg.strings, lang, k, v);
  const ctx = manager ? { ui: true, manager: true } : { ui: true };
  const call = async (op, args = {}) => {
    const r = await store.call(op, args, ctx);
    if (!r.ok) { flash(S('res.error', { code: r.error.code })); return null; }
    return r.result;
  };
  if (!doc.getElementById('mem-style')) { const st = h(doc, 'style', { id: 'mem-style' }); st.textContent = STYLE; doc.head.append(st); }
  doc.documentElement.lang = lang;
  const root = h(doc, 'div', { class: 'mem-root', 'data-mode': manager ? 'manager' : 'panel' });
  applyTheme(root, cfg, scheme);
  el.replaceChildren(root);
  let query = '';
  let editing = null;      // id being edited
  let merging = null;      // [ids] being merged
  let msg = '';
  let importPlan = null;   // { data, plan }
  let eraseOpen = false;
  const flash = (m) => { msg = m; render(); };

  const scopeLabel = (s) => (s === 'all' ? S('panel.scope.all') : s.startsWith('app:') ? S('panel.scope.app', { app: appLabel(cfg, s.slice(4), lang) }) : S('panel.scope.project'));
  const appsList = cfg.apps.map((a) => ({ id: a.id, label: appLabel(cfg, a.id, lang) }));

  function item(r, { past = false } = {}) {
    const li = h(doc, 'li', { class: 'mem-item', 'data-id': r.id });
    if (editing === r.id) {
      const input = h(doc, 'input', { type: 'text', maxlength: '300', 'aria-label': S('act.edit') });
      input.value = r.fact;
      const counter = h(doc, 'span', { class: 'mem-meta', text: S('panel.counter', { n: r.fact.length }) });
      input.addEventListener('input', () => { counter.textContent = S('panel.counter', { n: input.value.length }); });
      li.append(h(doc, 'div', { class: 'mem-row' }, [input, counter]), h(doc, 'div', { class: 'mem-actions' }, [
        h(doc, 'button', { class: 'primary', type: 'button', text: S('act.save'), on: { click: async () => { const x = await call('edit', { id: r.id, fact: input.value }); if (x?.ok === false) flash(S('fail.tooLong', { n: input.value.length })); editing = null; render(); } } }),
        h(doc, 'button', { type: 'button', text: S('act.cancel'), on: { click: () => { editing = null; render(); } } }),
      ]));
      queueMicrotask(() => input.focus());
      return li;
    }
    li.append(h(doc, 'span', { class: 'mem-fact', text: r.fact }));
    const meta = [formatWhen(r.eventAt ?? r.updatedAt ?? r.at, lang, cfg.strings), scopeLabel(r.scope)];
    if (r.pinned) meta.unshift(S('panel.pinned'));
    li.append(h(doc, 'span', { class: 'mem-meta', text: meta.join(' · ') }));
    const why = h(doc, 'details', {}, [h(doc, 'summary', { class: 'mem-meta', text: S('act.why') }), h(doc, 'div', { class: 'mem-meta', text: describeSource(r, { strings: cfg.strings, lang, apps: cfg.apps }) })]);
    const actions = h(doc, 'div', { class: 'mem-actions' });
    if (!past) {
      if (r.status === 'review') actions.append(h(doc, 'button', { type: 'button', class: 'primary', text: S('act.keep'), on: { click: () => call('keep', { id: r.id }).then(render) } }));
      actions.append(
        h(doc, 'button', { type: 'button', text: S('act.edit'), on: { click: () => { editing = r.id; render(); } } }),
        h(doc, 'button', { type: 'button', text: r.pinned ? S('act.unpin') : S('act.pin'), on: { click: () => call('pin', { id: r.id, pinned: !r.pinned }).then(render) } }),
      );
      const sel = h(doc, 'select', { 'aria-label': S('panel.scope.all') });
      const opts = ['all', ...appsList.map((a) => `app:${a.id}`)];
      if (r.scope.startsWith('project:')) opts.push(r.scope);
      for (const o of opts) { const op = h(doc, 'option', { value: o, text: scopeLabel(o) }); if (o === r.scope) op.selected = true; sel.append(op); }
      sel.addEventListener('change', () => call('setScope', { id: r.id, scope: sel.value }).then(render));
      actions.append(sel);
    }
    actions.append(h(doc, 'button', { type: 'button', class: 'danger', text: S('act.forget'), on: { click: async () => {
      const x = await call('forget', { id: r.id });
      if (!x) return;
      msg = '';
      undoBar = { id: r.id, fact: r.fact, until: Date.now() + cfg.limits.undoGraceMs };
      // The item and its button are gone: the keyboard goes to «Deshacer».
      focusAfter = '[data-mem-undo]';
      render();
      setTimeout(() => {
        if (!undoBar || undoBar.id !== r.id) return;
        undoBar = null;
        // If the keyboard was on «Deshacer», it lands on the search box, not on the page body.
        if (root.contains(doc.activeElement) && doc.activeElement.matches('[data-mem-undo]')) focusAfter = 'input[type=search]';
        render();
      }, cfg.limits.undoGraceMs);
    } } }));
    li.append(why, actions);
    return li;
  }
  let undoBar = null;
  // After a re-render, where the keyboard goes (a selector inside the panel).
  // Every render replaces the nodes, so without this the focus falls to <body>.
  let focusAfter = null;

  // Where the keyboard is now, as selectors to find the same control after a
  // re-render (another app's change re-renders too).
  function focusKey() {
    const a = doc.activeElement;
    if (!a || a === root || !root.contains(a)) return null;
    if (a.matches('[data-mem-undo]')) return ['[data-mem-undo]', 'input[type=search]'];
    if (a.matches('input[type=search]')) return ['input[type=search]'];
    const li = a.closest('li[data-id]');
    if (li && a.parentElement?.classList.contains('mem-actions')) {
      const k = [...a.parentElement.children].indexOf(a) + 1;
      return [`li[data-id="${CSS_ID(li.dataset.id)}"] .mem-actions > :nth-child(${k})`, 'input[type=search]'];
    }
    return null;
  }

  let seq = 0;
  async function render() {
    const mine = ++seq;
    const all = await store.call('all', {}, ctx);
    if (mine !== seq) return;   // a newer render is on its way; it paints
    if (!all.ok) { root.replaceChildren(h(doc, 'p', { class: 'mem-warn', text: S('fail.blocked') })); return; }
    const { records, proposals, settings, grants, meta } = all.result;
    const pendingIds = new Set(all.result.pending.map((p) => p.id));
    const active = records.filter((r) => r.status === 'active' && !pendingIds.has(r.id));
    const q = query.trim().toLowerCase();
    const match = (r) => !q || r.fact.toLowerCase().includes(q);
    const review = records.filter((r) => r.status === 'review' && match(r));
    const core = active.filter((r) => (r.pinned || (r.slot && isExclusive(r.slot))) && match(r));
    const coreIds = new Set(core.map((r) => r.id));
    const rest = active.filter((r) => !coreIds.has(r.id) && match(r)).sort((a, b) => b.updatedAt - a.updatedAt);
    const similar = active.filter((r) => r.possibleDup && active.some((x) => x.id === r.possibleDup) && r.id < r.possibleDup);
    const past = records.filter((r) => (r.status === 'superseded' || r.status === 'expired') && match(r)).sort((a, b) => (b.validTo || 0) - (a.validTo || 0));
    const kids = [];
    if (manager) kids.push(h(doc, 'h1', { text: S('mgr.title') }));
    kids.push(h(doc, 'p', { class: 'mem-head', text: S('panel.header', { n: active.length, brand: cfg.brand.name }) }));
    if (msg) kids.push(h(doc, 'p', { class: 'mem-warn', role: 'alert', text: msg }));
    if (undoBar) {
      kids.push(h(doc, 'div', { class: 'mem-row', role: 'status' }, [
        h(doc, 'span', { text: S('card.forgotten') + ': ' + undoBar.fact }),
        h(doc, 'button', { type: 'button', text: S('act.undo'), 'data-mem-undo': true, on: { click: async () => {
          const id = undoBar.id;
          await call('restore', { id });
          undoBar = null;
          // Back to the fact that came back (its «Olvidar»), or the search box.
          focusAfter = [`li[data-id="${CSS_ID(id)}"] button.danger`, 'input[type=search]'];
          render();
        } } }),
      ]));
    }
    const search = h(doc, 'input', { type: 'search', placeholder: S('panel.search'), 'aria-label': S('panel.search') });
    search.value = query;
    search.addEventListener('input', () => { query = search.value; render().then(() => { const s = root.querySelector('input[type=search]'); if (s) { s.focus(); s.setSelectionRange(s.value.length, s.value.length); } }); });
    kids.push(search);

    if (proposals.length) {
      kids.push(h(doc, 'h2', { text: S('panel.pending') }));
      const ul = h(doc, 'ul');
      for (const p of proposals) {
        const li = h(doc, 'li', { class: 'mem-item' }, [h(doc, 'span', { class: 'mem-fact', text: p.fact })]);
        if (p.label === 'from-content') li.append(h(doc, 'span', { class: 'mem-meta', text: S('card.fromContent') }));
        li.append(h(doc, 'div', { class: 'mem-actions' }, [
          h(doc, 'button', { type: 'button', class: 'primary', text: S('act.yes'), on: { click: () => call('confirm', { pid: p.id }).then(render) } }),
          h(doc, 'button', { type: 'button', text: S('act.no'), on: { click: () => call('dismiss', { pid: p.id }).then(render) } }),
          h(doc, 'button', { type: 'button', text: S('act.never'), on: { click: () => call('dismiss', { pid: p.id, never: true }).then(render) } }),
        ]));
        ul.append(li);
      }
      kids.push(ul);
    }
    if (review.length) {
      kids.push(h(doc, 'h2', { text: S('panel.review') }));
      kids.push(h(doc, 'ul', {}, review.map((r) => item(r))));
    }
    if (!active.length && !proposals.length && !review.length) kids.push(h(doc, 'p', { class: 'mem-empty', text: S('panel.empty') }));
    if (core.length) { kids.push(h(doc, 'h2', { text: S('panel.about') })); kids.push(h(doc, 'ul', {}, core.map((r) => item(r)))); }
    if (rest.length) { kids.push(h(doc, 'h2', { text: S('panel.rest') })); kids.push(h(doc, 'ul', {}, rest.map((r) => item(r)))); }
    if (similar.length) {
      kids.push(h(doc, 'h2', { text: S('panel.similar') }));
      const ul = h(doc, 'ul');
      for (const r of similar) {
        const o = active.find((x) => x.id === r.possibleDup);
        const li = h(doc, 'li', { class: 'mem-item' }, [h(doc, 'span', { class: 'mem-fact', text: r.fact }), h(doc, 'span', { class: 'mem-fact', text: o.fact })]);
        if (merging && merging[0] === r.id) {
          const input = h(doc, 'input', { type: 'text', maxlength: '300', 'aria-label': S('act.merge') });
          input.value = r.fact.length >= o.fact.length ? r.fact : o.fact;
          li.append(input, h(doc, 'div', { class: 'mem-actions' }, [
            h(doc, 'button', { type: 'button', class: 'primary', text: S('act.save'), on: { click: async () => { await call('merge', { ids: [r.id, o.id], fact: input.value }); merging = null; render(); } } }),
            h(doc, 'button', { type: 'button', text: S('act.cancel'), on: { click: () => { merging = null; render(); } } }),
          ]));
        } else li.append(h(doc, 'div', { class: 'mem-actions' }, [h(doc, 'button', { type: 'button', text: S('act.merge'), on: { click: () => { merging = [r.id, o.id]; render(); } } })]));
        ul.append(li);
      }
      kids.push(ul);
    }
    // Add something (user-typed)
    const add = h(doc, 'input', { type: 'text', maxlength: '300', placeholder: S('act.add'), 'aria-label': S('act.add') });
    const addCount = h(doc, 'span', { class: 'mem-meta', text: S('panel.counter', { n: 0 }) });
    add.addEventListener('input', () => { addCount.textContent = S('panel.counter', { n: add.value.length }); });
    const doAdd = async () => { if (!add.value.trim()) return; const x = await call('add', { fact: add.value }); if (x?.ok === false) flash(x.code === 'too_long' ? S('fail.tooLong', { n: x.n }) : S('res.not_a_fact')); else render(); };
    add.addEventListener('keydown', (e) => { if (e.key === 'Enter') doAdd(); });
    kids.push(h(doc, 'h2', { text: S('act.add') }), h(doc, 'div', { class: 'mem-row' }, [add, addCount, h(doc, 'button', { type: 'button', class: 'primary', text: S('act.save'), on: { click: doAdd } })]));

    if (!manager) {
      kids.push(h(doc, 'p', {}, [h(doc, 'a', { href: cfg.storeUrl, target: '_blank', rel: 'noopener', text: S('act.manage') })]));
    } else {
      kids.push(...managerSection({ settings, grants, meta, past, activeCount: active.length }));
    }
    const want = focusAfter ? [].concat(focusAfter) : focusKey();
    focusAfter = null;
    root.replaceChildren(...kids);
    for (const sel of want || []) { const el = root.querySelector(sel); if (el) { el.focus(); break; } }
  }

  function managerSection({ settings, grants, meta, past, activeCount }) {
    const out = [];
    const sw = (label, checked, onChange) => {
      const id = 'mem-sw-' + Math.random().toString(36).slice(2, 8);
      const inp = h(doc, 'input', { type: 'checkbox', id });
      inp.checked = !!checked;
      inp.addEventListener('change', () => onChange(inp.checked));
      return h(doc, 'label', { for: id, class: 'mem-row' }, [inp, h(doc, 'span', { text: label })]);
    };
    out.push(h(doc, 'h2', { text: S('mgr.title') }));
    out.push(sw(S('mgr.on'), settings.on, (v) => call('settings', { on: v }).then(render)));
    out.push(sw(S('mgr.proposals'), settings.proposals, (v) => call('settings', { proposals: v }).then(render)));
    for (const a of appsList) {
      out.push(h(doc, 'div', { class: 'mem-row' }, [
        h(doc, 'strong', { text: a.label }),
        sw(S('mgr.read'), settings.apps[a.id]?.read, (v) => call('settings', { apps: { [a.id]: { read: v } } }).then(render)),
        sw(S('mgr.write'), settings.apps[a.id]?.write, (v) => call('settings', { apps: { [a.id]: { write: v } } }).then(render)),
      ]));
    }
    out.push(h(doc, 'h2', { text: S('mgr.access') }));
    const granted = grants.filter((g) => g.state === 'granted');
    if (!granted.length) out.push(h(doc, 'p', { class: 'mem-meta', text: S('mgr.noAccess') }));
    for (const g of granted) {
      const a = cfg.apps.find((x) => x.origin === g.origin);
      out.push(h(doc, 'div', { class: 'mem-row' }, [h(doc, 'span', { text: a ? appLabel(cfg, a.id, lang) : g.origin }),
        h(doc, 'button', { type: 'button', text: S('act.revoke'), on: { click: () => call('revoke', { origin: g.origin }).then(render) } })]));
    }
    if (past.length) {
      out.push(h(doc, 'h2', { text: S('panel.past') }));
      out.push(h(doc, 'ul', {}, past.map((r) => item(r, { past: true }))));
    }
    // Export
    out.push(h(doc, 'h2', { text: `${S('act.export')} · ${S('act.import')}` }));
    out.push(h(doc, 'p', { class: 'mem-meta', text: S('mgr.exportWarn') }));
    const exp = h(doc, 'button', { type: 'button', text: S('act.export'), on: { click: async () => {
      const data = await call('export');
      if (!data) return;
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const a = h(doc, 'a', { href: URL.createObjectURL(blob), download: `${cfg.namespace}-export.json` });
      doc.body.append(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    } } });
    const file = h(doc, 'input', { type: 'file', accept: 'application/json,.json', 'aria-label': S('act.import') });
    file.addEventListener('change', async () => {
      const f = file.files?.[0];
      if (!f) return;
      let data = null;
      try { data = JSON.parse(await f.text()); } catch { flash(S('mgr.importBad')); return; }
      const plan = await call('import', { data, dryRun: true });
      if (!plan) { flash(S('mgr.importBad')); return; }
      importPlan = { data, plan };
      render();
    });
    out.push(h(doc, 'div', { class: 'mem-row' }, [exp, file]));
    if (importPlan) {
      out.push(h(doc, 'div', { class: 'mem-row', role: 'status' }, [
        h(doc, 'span', { text: S('mgr.dryRun', { add: importPlan.plan.add, dup: importPlan.plan.dup, review: importPlan.plan.review, history: importPlan.plan.history }) }),
        h(doc, 'button', { type: 'button', class: 'primary', text: S('act.confirmImport'), on: { click: async () => { await call('import', { data: importPlan.data, dryRun: false }); importPlan = null; render(); } } }),
        h(doc, 'button', { type: 'button', text: S('act.cancel'), on: { click: () => { importPlan = null; render(); } } }),
      ]));
    }
    // Erase all: two steps, typing the word.
    out.push(h(doc, 'h2', { text: S('act.erase') }));
    if (!eraseOpen) out.push(h(doc, 'button', { type: 'button', class: 'danger', text: S('act.erase'), on: { click: () => { eraseOpen = true; render(); } } }));
    else {
      const word = S('mgr.eraseWord');
      const inp = h(doc, 'input', { type: 'text', 'aria-label': word });
      const go = h(doc, 'button', { type: 'button', class: 'danger', text: S('act.erase'), disabled: true, on: { click: async () => { if (inp.value.trim().toLowerCase() !== word) return; await call('wipe'); eraseOpen = false; render(); } } });
      inp.addEventListener('input', () => { go.disabled = inp.value.trim().toLowerCase() !== word; });
      out.push(h(doc, 'p', { text: S('mgr.eraseStep', { n: activeCount, word }) }), h(doc, 'div', { class: 'mem-row' }, [inp, go,
        h(doc, 'button', { type: 'button', text: S('act.cancel'), on: { click: () => { eraseOpen = false; render(); } } })]));
    }
    out.push(h(doc, 'p', { class: meta.persisted ? 'mem-meta' : 'mem-warn', text: meta.persisted ? S('mgr.persisted') : S('mgr.notPersisted') }));
    if (meta.migration?.total) out.push(h(doc, 'p', { class: 'mem-meta', text: S('mgr.migrated', { n: meta.migration.total }) }));
    return out;
  }

  const off = store.onChange(() => { render(); });
  render();
  return { render, unmount() { off(); root.remove(); } };
}
