// Shared novelty archive (both modes), persisted in IndexedDB.
// novelty(x) = mean distance to the k nearest items of the same mode
// (saved finds + known references). Archive keeps the most novel per mode.

const DB = 'emergence-hunt', STORE = 'finds';
let dbp;
function db() {
  return dbp ||= new Promise((res, rej) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(STORE, { keyPath: 'id' });
    r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
  });
}
async function tx(mode, fn) {
  const d = await db();
  return new Promise((res, rej) => {
    const t = d.transaction(STORE, mode), s = t.objectStore(STORE);
    const out = fn(s);
    t.oncomplete = () => res(out?.result ?? out); t.onerror = () => rej(t.error);
  });
}

export const dist = (a, b) => {
  if (a.length !== b.length) return Infinity; // feature schema changed: not comparable
  return Math.sqrt(a.reduce((s, x, i) => s + (x - b[i]) ** 2, 0));
};

export class Gallery {
  constructor({ capacity = 40, k = 3 } = {}) {
    this.items = []; this.capacity = capacity; this.k = k; this.listeners = [];
  }
  async load() {
    try { this.items = await tx('readonly', s => s.getAll()); } catch (e) { console.warn('idb', e); this.items = []; }
    this.items.sort((a, b) => b.time - a.time);
    this.emit();
  }
  on(f) { this.listeners.push(f); }
  emit() { this.listeners.forEach(f => f(this)); }
  pool(mode) { return this.items.filter(i => i.mode === mode); }

  novelty(mode, feat, exclude) {
    const ds = this.pool(mode).filter(i => i !== exclude).map(i => dist(feat, i.features)).sort((a, b) => a - b);
    if (!ds.length || !isFinite(ds[0])) return Infinity;
    const n = Math.min(this.k, ds.length);
    return ds.slice(0, n).reduce((a, b) => a + b, 0) / n;
  }
  nearest(mode, feat) {
    let best = null, bd = Infinity;
    for (const i of this.pool(mode)) { const d = dist(feat, i.features); if (d < bd) { bd = d; best = i; } }
    return { item: best, d: bd };
  }

  // Try to add a candidate; returns the stored item or null.
  async consider(c, { minNovelty = 0.12 } = {}) {
    const nov = this.novelty(c.mode, c.features);
    if (nov < minNovelty) return null;
    const near = this.nearest(c.mode, c.features);
    const item = { ...c, id: c.id || `${c.mode}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
      novelty: Math.min(nov, 9), nearest: near.item ? near.item.title || near.item.label : null, time: Date.now() };
    this.items.unshift(item);
    await this.put(item);
    await this.prune(c.mode);
    this.emit();
    return this.items.includes(item) ? item : null;
  }

  async prune(mode) {
    const finds = this.pool(mode).filter(i => !i.known);
    const cap = Math.ceil(this.capacity * 0.3);
    for (;;) {
      // evict the least novel w.r.t. the rest of the archive — from an over-represented label
      // first (noisy classes like chaos always look "novel" and would crowd out rare ones)
      const cnt = {};
      finds.forEach(i => cnt[i.label] = (cnt[i.label] || 0) + 1);
      const top = Object.keys(cnt).sort((a, b) => cnt[b] - cnt[a])[0];
      const crowded = cnt[top] > cap && finds.length >= this.capacity * 0.75;
      if (finds.length <= this.capacity && !crowded) break;
      const pool = cnt[top] > cap ? finds.filter(i => i.label === top) : finds;
      let worst = null, wn = Infinity;
      for (const i of pool) { const n = this.novelty(mode, i.features, i); if (n < wn) { wn = n; worst = i; } }
      finds.splice(finds.indexOf(worst), 1);
      await this.remove(worst.id, false);
    }
    for (const i of this.pool(mode)) i.novelty = Math.min(9, this.novelty(mode, i.features, i));
  }

  async addKnown(item) { // replaces existing known with same id
    this.items = this.items.filter(i => i.id !== item.id);
    this.items.push({ ...item, known: true, time: Date.now() });
    await this.put(this.items[this.items.length - 1]);
    this.emit();
  }
  async put(item) { try { await tx('readwrite', s => s.put(item)); } catch (e) { console.warn('idb put', e); } }
  async remove(id, emit = true) {
    this.items = this.items.filter(i => i.id !== id);
    try { await tx('readwrite', s => s.delete(id)); } catch (e) { console.warn(e); }
    if (emit) this.emit();
  }
  async clearFinds() {
    for (const i of this.items.filter(i => !i.known)) await this.remove(i.id, false);
    this.emit();
  }
  export() { return JSON.stringify(this.items); }
  async import(json) {
    for (const it of JSON.parse(json)) { this.items = this.items.filter(i => i.id !== it.id); this.items.push(it); await this.put(it); }
    this.emit();
  }
}
