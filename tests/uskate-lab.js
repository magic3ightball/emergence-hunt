// Browser-only lab for planting U-skates (Munafo, arXiv:1501.01990) — not part of the app.
// Usage in the page console:  const L = await import('/tests/uskate-lab.js');
// U-skates live in the blue state (u≈0.42, v≈0.29) as a U of high u; they move ~1 lu / 62,000 steps.
import { blueState } from '../src/gs.js';

// Blue background with shapes drawn in a high-u / low-v state (plus Pearson's 1% noise).
export function negField(T, F, k, draw, { uu = 0.8, vv = 0.05, seed = 1 } = {}) {
  const b = blueState(F, k);
  const u = new Float32Array(T * T).fill(b.u), v = new Float32Array(T * T).fill(b.v);
  let s = seed >>> 0 || 1;
  const r = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  const set = (x, y) => { x = ((x % T) + T) % T; y = ((y % T) + T) % T; u[y * T + x] = uu; v[y * T + x] = vv; };
  draw(set, T, r);
  for (let i = 0; i < T * T; i++) { u[i] += (r() - 0.5) * 0.02; v[i] = Math.max(0, v[i] + (r() - 0.5) * 0.02); }
  return { u, v };
}

// U-shaped band of radius R, half-thickness th, angular span, opening toward +y, centred in the tile.
export const arc = (R, th, span, ox = 0, oy = 0) => (set, T) => {
  const c = T >> 1;
  for (let y = -R - th; y <= R + th; y++) for (let x = -R - th; x <= R + th; x++) {
    if (Math.abs(Math.hypot(x, y) - R) > th) continue;
    const a = (Math.atan2(y, x) + Math.PI * 2) % (Math.PI * 2);
    if (Math.abs(a - Math.PI * 1.5) <= span / 2) set(c + ox + x, c + oy + y);
  }
};

export const spots = (pts, rad) => (set, T) => {
  const c = T >> 1;
  for (const [px, py] of pts) for (let y = -rad; y <= rad; y++) for (let x = -rad; x <= rad; x++)
    if (x * x + y * y <= rad * rad) set(c + px + x, c + py + y);
};

// Munafo's random start: random rectangles with random u, v over the blue state.
export const rects = (n) => (set, T, r) => {
  for (let i = 0; i < n; i++) {
    const w = 4 + ((r() * 14) | 0), h = 4 + ((r() * 14) | 0), x0 = (r() * T) | 0, y0 = (r() * T) | 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) set(x0 + x, y0 + y);
  }
};

// Contact sheet of filmstrips over the page (for screenshots). rows: runBatch results with .film
export function sheet(rows, title = '') {
  let d = document.getElementById('rvSheet');
  if (!d) {
    d = document.createElement('div'); d.id = 'rvSheet';
    d.style.cssText = 'position:fixed;inset:0;z-index:99;background:#0b0d12;overflow:auto;padding:8px;font:11px sans-serif;color:#ddd';
    document.body.appendChild(d);
  }
  d.innerHTML = `<div style="font-weight:bold;margin-bottom:4px">${title}</div>` + rows.map((r, j) =>
    `<div style="display:flex;gap:3px;align-items:center;margin-bottom:3px"><div style="width:170px">#${j} <b>${r.label}</b> ${r.d || ''}<br>` +
    `mot ${r.stats.motion.toFixed(2)} drift ${(r.stats.drift ?? 0).toFixed(2)} n ${r.stats.count} tov ${r.stats.turnover.toFixed(2)}</div>` +
    r.film.map(f => `<img src="${f.img}" width="64" height="64">`).join('') + '</div>').join('');
  d.scrollTop = 0;
}
export const closeSheet = () => document.getElementById('rvSheet')?.remove();
