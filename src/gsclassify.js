// Gray-Scott outcome analysis on a periodic T×T v-field (CPU, pure functions).

// Connected components of mask (Uint8Array) with periodic wrap.
// Returns blobs: {area, perim, cx, cy} (cx,cy circular means in [0,T)).
export function components(mask, T) {
  const lab = new Int32Array(T * T).fill(-1);
  const blobs = [];
  const stack = new Int32Array(T * T);
  for (let s = 0; s < T * T; s++) {
    if (!mask[s] || lab[s] >= 0) continue;
    const id = blobs.length;
    let sp = 0, area = 0, perim = 0, sx = 0, sy = 0, cx = 0, cy = 0;
    stack[sp++] = s; lab[s] = id;
    while (sp) {
      const i = stack[--sp];
      const x = i % T, y = (i / T) | 0;
      area++;
      const a = (2 * Math.PI * x) / T, b = (2 * Math.PI * y) / T;
      sx += Math.sin(a); cx += Math.cos(a); sy += Math.sin(b); cy += Math.cos(b);
      const nb = [y * T + (x + 1) % T, y * T + (x + T - 1) % T, ((y + 1) % T) * T + x, ((y + T - 1) % T) * T + x];
      for (const j of nb) {
        if (!mask[j]) { perim++; continue; }
        if (lab[j] < 0) { lab[j] = id; stack[sp++] = j; }
      }
    }
    const ang = (s, c) => ((Math.atan2(s, c) / (2 * Math.PI)) * T + T) % T;
    blobs.push({ area, perim, cx: ang(sx, cx), cy: ang(sy, cy) });
  }
  return { blobs, lab };
}

export function analyzeField(v, T) {
  let mn = Infinity, mx = -Infinity, sum = 0, sq = 0;
  for (let i = 0; i < v.length; i++) { const x = v[i]; mn = Math.min(mn, x); mx = Math.max(mx, x); sum += x; sq += x * x; }
  const mean = sum / v.length, std = Math.sqrt(Math.max(0, sq / v.length - mean * mean));
  const out = { mean, std, range: mx - mn, highFrac: 0, highCount: 0, fgHigh: 1, count: 0, meanArea: 0, compact: 1, lineFrac: 0, blobs: [] };
  if (mx - mn < 0.04) return out;
  const thr = (mn + mx) / 2;
  const hi = new Uint8Array(v.length);
  let hc = 0;
  for (let i = 0; i < v.length; i++) if (v[i] > thr) { hi[i] = 1; hc++; }
  out.highFrac = hc / v.length;
  const H = components(hi, T);
  const hiBlobs = H.blobs.filter(b => b.area >= 4);
  out.highCount = hiBlobs.length;
  let blobs = hiBlobs;
  if (out.highFrac > 0.5) { // minority phase is the low one (negative spots / holes)
    const lo = hi.map(x => 1 - x);
    blobs = components(lo, T).blobs.filter(b => b.area >= 4);
    out.fgHigh = 0;
  }
  out.blobs = blobs;
  out.count = blobs.length;
  if (!blobs.length) return out;
  let A = 0, cw = 0, line = 0;
  for (const b of blobs) {
    const c = Math.min(1, (4 * Math.PI * b.area) / (b.perim * b.perim));
    b.compact = c;
    A += b.area; cw += c * b.area;
    if (c < 0.35 && b.area > 30) line += b.area;
  }
  out.meanArea = A / blobs.length;
  out.compact = cw / A;
  out.lineFrac = line / A;
  return out;
}

const wrapd = (a, T) => { let d = Math.abs(a); return Math.min(d, T - d); };

// Mean displacement of compact blobs between two analyses separated by dt steps.
function blobMotion(A, B, T) {
  const pa = A.blobs.filter(b => b.compact > 0.45), pb = B.blobs.filter(b => b.compact > 0.45);
  if (!pa.length || !pb.length) return 0;
  let tot = 0;
  for (const b of pb) {
    let best = Infinity;
    for (const a of pa) {
      const dx = wrapd(a.cx - b.cx, T), dy = wrapd(a.cy - b.cy, T);
      best = Math.min(best, Math.hypot(dx, dy));
    }
    tot += Math.min(best, 8);
  }
  return tot / pb.length;
}

// Mean displacement of every blob (any shape) between two analyses, in cells, or null when the blobs
// don't pair up one-to-one with the same area (±8 %: split, merged, died, or growing like a hairpin).
function blobShift(P, Q, T) {
  // tile-spanning blobs (labyrinths) have no meaningful centroid
  if (!P.blobs.length || P.blobs.length !== Q.blobs.length || P.blobs.length > 40 || Q.blobs.some(b => b.area > T * T / 16)) return null;
  let tot = 0;
  for (const q of Q.blobs) {
    let best = Infinity, bp = null;
    for (const p of P.blobs) {
      const d = Math.hypot(wrapd(p.cx - q.cx, T), wrapd(p.cy - q.cy, T));
      if (d < best) { best = d; bp = p; }
    }
    // a glider keeps its form; slowly lengthening arcs shift their centroid without travelling
    if (best > T / 4 || q.area > bp.area * 1.08 || q.area < bp.area * 0.92) return null;
    tot += best;
  }
  return tot / Q.blobs.length;
}

// Slow steady drift (U-skates move ~0.3 cells per 200 steps, too slow for blobMotion's last-200-step view):
// shift over the last two checkpoint intervals, in cells per DT steps, and whether both rates agree.
function slowDrift(history, B, T, DT, tEnd) {
  const n = history.length;
  if (tEnd == null || n < 2) return { rate: 0, steady: false };
  const h2 = history[n - 2], h3 = history[n - 1];
  const s1 = blobShift(h2.a, h3.a, T), s2 = blobShift(h3.a, B, T);
  if (s1 == null || s2 == null) return { rate: 0, steady: false };
  const r1 = s1 / (h3.t - h2.t) * DT, r2 = s2 / (tEnd - h3.t) * DT;
  return { rate: r2, steady: r1 > 0.08 && r2 > 0.08 && r2 / r1 > 0.5 && r2 / r1 < 2 };
}

// history: [{t, a}] analyses at checkpoints; vA, vB: v-fields DT steps apart at the end (vB at step tEnd).
export function classify(history, vA, vB, T, DT, tEnd) {
  const A = analyzeField(vA, T), B = analyzeField(vB, T);
  let act = 0, actN = 0;
  for (let i = 0; i < vA.length; i++) { const d = Math.abs(vB[i] - vA[i]); act += d; if (d > 0.03) actN++; }
  act /= vA.length;
  const activeFrac = actN / vA.length; // chaos is active everywhere; growing fronts only locally
  const motion = blobMotion(A, B, T);
  const drift = slowDrift(history, B, T, DT, tEnd);
  const first = history.find(h => h.a.highCount > 0) || history[0];
  const mid = history[Math.floor(history.length / 2)];
  const growth = Math.log((B.highCount + 1) / (first.a.highCount + 1));
  const turnover = Math.abs(B.count - A.count) / Math.max(1, Math.max(A.count, B.count));
  // late vs mid count change (still replicating at the end?)
  const lateGrowth = Math.log((B.highCount + 1) / (mid.a.highCount + 1));

  // Count of bright blobs at each checkpoint (from the first non-empty one) and the bright-area share.
  const series = [...history.slice(history.indexOf(first)).map(h => h.a), B];
  const counts = series.map(a => a.highCount);
  const monotone = counts.every((c, i) => !i || c >= counts[i - 1]);
  const fracs = history.slice(1).map(h => h.a.highFrac).concat(B.highFrac);
  // bright blobs that start small and keep swelling; not holes shrinking in a bright sea (fracs[0] already high)
  const swelling = fracs[0] < 0.35 && fracs.every((f, i) => !i || f > fracs[i - 1] * 1.03) && B.highFrac > fracs[0] * 1.4;

  let label;
  const spotty = B.compact > 0.55 && B.lineFrac < 0.3;
  // a division front sweeping the tile, not a whole field of spots splitting and dying (that stays chaos)
  const dividing = spotty && B.fgHigh && B.highFrac < 0.45 && turnover < 0.2 && activeFrac < 0.35 &&
    (growth > 1.6 || (monotone && counts[0] >= 1 && B.highCount >= 2 * counts[0] && B.highCount >= 4));
  if (B.range < 0.04 || B.count === 0 || B.std < 0.006) label = 'uniform';
  // gliders keep their number; wandering/splitting holes change it (turnover)
  else if ((spotty && motion > 0.6 && motion < 6 && turnover < 0.03 && activeFrac < 0.2 && growth < 1.2) ||
    (drift.steady && turnover < 0.03 && activeFrac < 0.2)) label = 'moving solitons'; // fast round blobs, or any shape drifting steadily
  // a few blobs whose bright area keeps growing at every checkpoint (bubbles); growing worms stay 'stripes'
  else if (B.highCount <= 4 && B.count <= 4 && B.lineFrac < 0.5 && counts[counts.length - 1] <= counts[0] + 1 && swelling) label = 'expanding';
  // spots still dividing at the end make the field active, so test this before chaos
  else if (dividing) label = 'self-replicating';
  else if (act > 0.006 && activeFrac > 0.12 && (turnover > 0.08 || act > 0.02 || activeFrac > 0.3)) label = 'chaos';
  else if (spotty) label = 'spots';
  else label = 'stripes';

  const c01 = x => Math.max(0, Math.min(1, x));
  const features = [
    c01(B.std * 4),
    c01(B.highFrac),
    c01(Math.log1p(B.count) / Math.log(400)),
    c01(Math.log1p(B.meanArea) / Math.log(4000)),
    c01(B.compact),
    c01(B.lineFrac),
    c01(Math.sqrt(act / 0.05)),
    c01(Math.max(motion, drift.steady ? drift.rate : 0) / 3),
    c01(growth / 5),
    c01(turnover * 3),
    B.fgHigh,
    c01(lateGrowth / 3 + 0.5),
    c01(activeFrac * 2),
  ];
  return { label, features, stats: { act, activeFrac, motion, drift: drift.steady ? drift.rate : 0, growth, turnover, count: B.count, highCount: B.highCount, compact: B.compact, lineFrac: B.lineFrac, highFrac: B.highFrac, std: B.std } };
}

export const GS_LABELS = ['uniform', 'spots', 'stripes', 'self-replicating', 'expanding', 'moving solitons', 'chaos'];
export const LABEL_COLORS = {
  uniform: '#5b6170', spots: '#4fb3bf', stripes: '#e2b85c', 'self-replicating': '#e2725b',
  'moving solitons': '#b48ce6', expanding: '#7fa8e8', chaos: '#e85d9a',
};
