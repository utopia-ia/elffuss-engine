// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 UtopiaIA
//
// User memory: the small pieces every app shows. A card under an answer
// (saved, updated, proposal, grant, forget, unavailable...), the "N memories in
// context" disclosure, the per-conversation "No memory" switch and the panel
// frame (the store page itself in a visible iframe).
//
// Fact text is ALWAYS set with `textContent`, never as HTML or markdown.
// Styling only through `--mem-*` custom properties with neutral fallbacks,
// set through the CSSOM (no inline style attribute, no injected stylesheet),
// so the host app's own content security policy needs no exception.

import { t } from './memory-text.js';
import { appLabel } from './memory-config.js';

const V = {
  bg: 'var(--mem-bg, Canvas)', fg: 'var(--mem-fg, CanvasText)', muted: 'var(--mem-muted, GrayText)',
  accent: 'var(--mem-accent, AccentColor)', accentFg: 'var(--mem-accent-fg, AccentColorText)',
  border: 'var(--mem-border, GrayText)', radius: 'var(--mem-radius, 0.5rem)', font: 'var(--mem-font, system-ui, sans-serif)',
  danger: 'var(--mem-danger, crimson)',
};
const css = (el, props) => { for (const [k, v] of Object.entries(props)) el.style.setProperty(k, v); return el; };

export function applyTheme(el, cfg, scheme = 'light') {
  const set = { ...(cfg.theme?.light || {}), ...(scheme === 'dark' ? cfg.theme?.dark || {} : {}) };
  for (const [k, v] of Object.entries(set)) if (/^--mem-[a-z][a-z0-9-]*$/.test(k)) el.style.setProperty(k, v);
  return el;
}

function h(doc, tag, props = {}, kids = []) {
  const el = doc.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'text') el.textContent = v;
    else if (k === 'class') el.className = v;
    else if (k === 'on') for (const [ev, f] of Object.entries(v)) el.addEventListener(ev, f);
    else if (k === 'style') css(el, v);
    else el.setAttribute(k, v);
  }
  for (const c of kids) if (c) el.appendChild(typeof c === 'string' ? doc.createTextNode(c) : c);
  return el;
}

function button(doc, label, onClick, { primary = false, danger = false } = {}) {
  return h(doc, 'button', {
    type: 'button', text: label, class: 'mem-btn',
    style: {
      font: 'inherit', 'font-size': '0.875em', padding: '0.25em 0.75em', margin: '0.25em 0.5em 0 0',
      'border-radius': V.radius, cursor: 'pointer', border: `1px solid ${primary ? V.accent : V.border}`,
      background: primary ? V.accent : 'transparent', color: primary ? V.accentFg : danger ? V.danger : V.fg,
    },
    // The card's host remembers that the keyboard was here before the button is
    // disabled (a disabled button drops the focus to <body>).
    on: { click: async (e) => { const b = e.currentTarget; const host = b.closest('.mem-card')?.parentNode; if (host) host.__memFocus = true; b.disabled = true; try { await onClick(e); } finally { b.disabled = false; } } },
  });
}

function shell(doc, cfg, opts) {
  const box = h(doc, 'div', {
    class: 'mem-root mem-card', role: 'status', 'aria-live': 'polite',
    style: {
      'font-family': V.font, color: V.fg, background: V.bg, border: `1px solid ${V.border}`,
      'border-radius': V.radius, padding: '0.5em 0.75em', margin: '0.5em 0', 'font-size': '0.9em',
      'overflow-wrap': 'anywhere', 'max-width': '100%', 'box-sizing': 'border-box',
    },
  });
  applyTheme(box, cfg, opts.scheme);
  return box;
}

// Where the keyboard goes when a card closes with nothing left to press:
// `opts.focusOut` (an element or a function returning one), else the app's
// element marked `data-mem-focus-out` (its message box).
function focusOut(doc, opts) {
  let to = typeof opts.focusOut === 'function' ? opts.focusOut() : opts.focusOut;
  if (!to) to = doc.querySelector('[data-mem-focus-out]');
  try { to?.focus?.(); } catch {}
}

// Renders `card` (from `cardFor` or `runMemoryTool`) into `el`, replacing what
// was there. `opts`: { lang, scheme, onSee, onGranted, onDone, focusOut }.
// When the keyboard was on the card, it stays on it: the new card's first
// button, the card itself when it has none, or `focusOut` when it closes.
export function renderCard(el, card, mem, opts = {}) {
  const doc = el.ownerDocument;
  const cfg = mem.cfg;
  const lang = opts.lang || mem.lang;
  const S = (k, v) => t(cfg.strings, lang, k, v);
  const active = doc.activeElement;
  const hadFocus = !!el.__memFocus || (!!active && active !== doc.body && el.contains(active));
  el.__memFocus = false;
  el.replaceChildren();
  if (!card) { if (hadFocus) focusOut(doc, opts); return null; }
  const box = shell(doc, cfg, opts);
  const line = (text) => h(doc, 'div', { text, class: 'mem-line' });
  const actions = h(doc, 'div', { class: 'mem-actions', style: { display: 'flex', 'flex-wrap': 'wrap' } });
  const swap = (next) => renderCard(el, next, mem, opts);
  const close = () => { const f = !!el.__memFocus || el.contains(doc.activeElement); el.__memFocus = false; el.replaceChildren(); if (f) focusOut(doc, opts); };
  const done = (x) => { opts.onDone?.(x); };

  const forgotten = (id, fact) => swap({ type: 'forgotten', id, fact });
  switch (card.type) {
    case 'saved':
    case 'similar':
      box.append(line(card.type === 'saved' ? S('card.saved', { fact: card.fact }) : S('card.similar', { prev: card.prev })));
      actions.append(button(doc, S('act.undo'), async () => { const r = await mem.forget(card.id); if (r?.pending) forgotten(card.id, card.fact); }));
      if (opts.onSee) actions.append(button(doc, S('act.see'), () => opts.onSee()));
      break;
    case 'duplicate':
      box.append(line(S('card.duplicate', { fact: card.fact })));
      break;
    case 'updated':
      box.append(line(S('card.updated', { prev: card.prev })));
      if (card.prevId) actions.append(button(doc, S('act.undo'), async () => { await mem.restore(card.prevId); swap({ type: 'note', text: S('act.undo') + ' ✓' }); }));
      break;
    case 'negated':
      box.append(line(S('card.negated', { fact: card.fact })));
      actions.append(button(doc, S('act.undo'), async () => { await mem.restore(card.id); swap({ type: 'note', text: S('act.undo') + ' ✓' }); }));
      break;
    case 'proposal': {
      box.append(line(S('card.proposal', { fact: card.fact })));
      if (card.label === 'from-content') box.append(h(doc, 'div', { text: S('card.fromContent'), style: { color: V.muted, 'font-size': '0.85em' } }));
      actions.append(
        button(doc, S('act.yes'), async () => { const r = await mem.confirm(card.pid); swap(r?.ok === false ? null : { type: 'saved', id: r.id, fact: r.fact || card.fact }); done(r); }, { primary: true }),
        button(doc, S('act.edit'), () => {
          const input = h(doc, 'input', { type: 'text', maxlength: '300', 'aria-label': S('act.edit'), style: { width: '100%', font: 'inherit', 'box-sizing': 'border-box', margin: '0.25em 0' } });
          input.value = card.fact;
          const counter = h(doc, 'span', { text: S('panel.counter', { n: input.value.length }), style: { color: V.muted, 'font-size': '0.8em' } });
          input.addEventListener('input', () => { counter.textContent = S('panel.counter', { n: input.value.length }); });
          actions.replaceChildren(
            button(doc, S('act.save'), async () => { const r = await mem.confirm(card.pid, input.value); swap(r?.ok === false ? { type: 'error', text: S('fail.tooLong', { n: input.value.length }) } : { type: 'saved', id: r.id, fact: r.fact }); done(r); }, { primary: true }),
            button(doc, S('act.cancel'), () => swap(card)),
          );
          box.insertBefore(input, actions); box.insertBefore(counter, actions);
          input.focus();
        }),
        button(doc, S('act.no'), async () => { await mem.dismiss(card.pid, false); close(); done(null); }),
        button(doc, S('act.never'), async () => { await mem.dismiss(card.pid, true); close(); done(null); }),
      );
      break;
    }
    case 'grant':
      box.append(line(S('card.grant', { app: appLabel(cfg, mem.app, lang), brand: cfg.brand.name })));
      actions.append(
        button(doc, S('act.allow'), async () => { await mem.setGrant('granted'); close(); opts.onGranted?.(true); }, { primary: true }),
        button(doc, S('act.no'), async () => { await mem.setGrant('denied'); close(); opts.onGranted?.(false); }),
      );
      break;
    case 'forget':
      box.append(line(S('card.forget', { fact: card.fact })));
      actions.append(
        button(doc, S('act.forget'), async () => { const r = await mem.forget(card.id); if (r?.pending) forgotten(card.id, card.fact); done(r); }, { danger: true }),
        button(doc, S('act.no'), () => { close(); done(null); }),
      );
      break;
    case 'forgotten': {
      box.append(line(S('card.forgotten')));
      const undo = button(doc, S('act.undo'), async () => { const r = await mem.restore(card.id); if (r?.ok !== false) swap({ type: 'note', text: S('act.undo') + ' ✓' }); });
      actions.append(undo);
      // The undo lives only as long as the store keeps the forgotten record.
      setTimeout(() => {
        const was = doc.activeElement === undo;
        undo.remove();
        if (was) { box.tabIndex = -1; box.focus(); }
      }, cfg.limits.undoGraceMs);
      break;
    }
    case 'ambiguous':
      box.append(line(S('card.ambiguous')));
      for (const c of card.candidates || []) actions.append(button(doc, c.fact, () => swap({ type: 'forget', id: c.id, fact: c.fact })));
      actions.append(button(doc, S('act.no'), () => close()));
      break;
    case 'unavailable':
      box.append(line(S('card.unavailable')));
      actions.append(button(doc, S('act.retry'), async () => { const r = await mem.retry(); if (r.mode !== 'off') close(); }));
      break;
    case 'queued':
      box.append(line(S('card.queued', { n: card.n })));
      break;
    case 'error':
    case 'note':
      box.append(line(card.text));
      break;
    default:
      return null;
  }
  if (actions.childNodes.length) box.append(actions);
  el.append(box);
  if (hadFocus) {
    const first = actions.querySelector('button');
    if (first) first.focus();
    else { box.tabIndex = -1; box.focus(); }
  }
  return box;
}

// "N memories in context ▾": expands to exactly the facts that went into the
// prompt this turn (`used.facts` from `mem.context()`).
export function renderUsed(el, used, opts = {}) {
  const doc = el.ownerDocument;
  el.replaceChildren();
  const facts = used?.facts || [];
  if (!facts.length) return null;
  const S = (k, v) => t(opts.cfg?.strings, opts.lang || 'es', k, v);
  const d = h(doc, 'details', { class: 'mem-root mem-used', style: { 'font-family': V.font, color: V.muted, 'font-size': '0.85em', margin: '0.25em 0' } });
  if (opts.cfg) applyTheme(d, opts.cfg, opts.scheme);
  d.append(h(doc, 'summary', { text: S('card.used', { n: facts.length }), style: { cursor: 'pointer' } }));
  const ul = h(doc, 'ul', { style: { margin: '0.25em 0', 'padding-left': '1.25em' } });
  for (const f of facts) ul.append(h(doc, 'li', { text: f }));
  d.append(ul);
  el.append(d);
  return d;
}

// The «No memory» switch for a conversation or a project.
export function renderComposerToggle(el, { cfg, lang = 'es', checked = false, onChange } = {}) {
  const doc = el.ownerDocument;
  el.replaceChildren();
  if (cfg && cfg.features && cfg.features.incognito === false) return null;
  const id = 'mem-incognito-' + Math.random().toString(36).slice(2, 8);
  const input = h(doc, 'input', { type: 'checkbox', id });
  input.checked = !!checked;
  input.addEventListener('change', () => onChange?.(input.checked));
  const label = h(doc, 'label', { for: id, class: 'mem-root mem-incognito', style: { 'font-family': V.font, display: 'inline-flex', gap: '0.4em', 'align-items': 'center', cursor: 'pointer' } },
    [input, h(doc, 'span', { text: t(cfg?.strings, lang, 'incognito') })]);
  el.append(label);
  return input;
}

// The panel: the store page itself in a visible frame. Edits, pins and forgets
// happen inside the store's origin through real clicks; the app never handles
// the full list. Only `view`, `lang` and `scheme` go in the URL.
export function mountPanelFrame(el, mem, { lang, scheme = 'light', height = '32rem' } = {}) {
  const doc = el.ownerDocument;
  const cfg = mem.cfg;
  const u = new URL(cfg.storeUrl);
  u.searchParams.set('view', 'panel');
  u.searchParams.set('lang', lang || mem.lang);
  u.searchParams.set('scheme', scheme === 'dark' ? 'dark' : 'light');
  const f = h(doc, 'iframe', { src: u.href, title: t(cfg.strings, lang || mem.lang, 'frameTitle'), referrerpolicy: 'no-referrer', class: 'mem-panel-frame',
    style: { width: '100%', height, border: '0', display: 'block', 'max-width': '100%' } });
  el.replaceChildren(f);
  return f;
}
