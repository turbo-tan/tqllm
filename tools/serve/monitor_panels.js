// Trusted, bundled panels only. No script loading, inference calls or credentials.
'use strict';
const monitorPanels = (() => {
  const panels = new Map();
  const pending = 'Waiting for fresh telemetry.';
  function register({id, title, render}) {
    if (typeof id !== 'string' || !/^[a-z][a-z0-9-]{0,47}$/.test(id) || panels.has(id) || typeof title !== 'string' || !title.trim() || typeof render !== 'function') {
      throw new Error('Invalid or duplicate monitor panel');
    }
    const card = document.createElement('section'), heading = document.createElement('h2'), body = document.createElement('div');
    card.className = 'extension-panel'; card.id = 'panel-' + id; card.dataset.state = 'unavailable';
    heading.id = 'panel-heading-' + id; heading.textContent = title;
    card.setAttribute('aria-labelledby', heading.id); body.textContent = pending;
    card.append(heading, body); document.getElementById('monitorPanels').append(card);
    panels.set(id, {card, body, render});
    return () => { if (panels.get(id)?.card === card) { card.remove(); panels.delete(id); } };
  }
  function update(snapshot) {
    for (const panel of panels.values()) {
      try {
        // A panel cannot mutate the snapshot used by the core or other panels.
        panel.render({snapshot: structuredClone(snapshot), element: panel.body});
        panel.card.dataset.state = 'ready';
      } catch (_) {
        panel.card.dataset.state = 'unavailable'; panel.body.textContent = 'Panel unavailable.';
      }
    }
  }
  function invalidate() {
    for (const panel of panels.values()) {
      panel.card.dataset.state = 'unavailable'; panel.body.textContent = pending;
    }
  }
  return Object.freeze({version: 1, register, update, invalidate});
})();
