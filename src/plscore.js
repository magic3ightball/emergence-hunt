// Particle Life structure scoring (CPU). Input: snapshots A,B (vec4 pos/vel per
// particle) DT sim-time apart, same particle order, so motion/rotation come from
// per-particle displacement.

const wrap = d => d - Math.round(d);

// Single-linkage clusters with link distance eps on the torus (grid hash).
export function clusters(data, off, N, eps) {
  const G = Math.floor(1 / eps), cell = new Map();
  const cx = new Int32Array(N), cy = new Int32Array(N);
  for (let i = 0; i < N; i++) {
    const x = data[(off + i) * 4], y = data[(off + i) * 4 + 1];
    cx[i] = Math.min(G - 1, (x * G) | 0); cy[i] = Math.min(G - 1, (y * G) | 0);
    const k = cy[i] * G + cx[i];
    (cell.get(k) || cell.set(k, []).get(k)).push(i);
  }
  const par = new Int32Array(N).map((_, i) => i);
  const find = i => { while (par[i] !== i) { par[i] = par[par[i]]; i = par[i]; } return i; };
  const e2 = eps * eps;
  for (let i = 0; i < N; i++) {
    const xi = data[(off + i) * 4], yi = data[(off + i) * 4 + 1];
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const l = cell.get(((cy[i] + dy + G) % G) * G + ((cx[i] + dx + G) % G));
      if (!l) continue;
      for (const j of l) {
        if (j <= i) continue;
        const ddx = wrap(data[(off + j) * 4] - xi), ddy = wrap(data[(off + j) * 4 + 1] - yi);
        if (ddx * ddx + ddy * ddy < e2) { const a = find(i), b = find(j); if (a !== b) par[a] = b; }
      }
    }
  }
  const groups = new Map();
  for (let i = 0; i < N; i++) { const r = find(i); (groups.get(r) || groups.set(r, []).get(r)).push(i); }
  return [...groups.values()];
}

function centroid(data, off, idx) { // circular mean
  let sx = 0, cx = 0, sy = 0, cy = 0;
  for (const i of idx) {
    const a = data[(off + i) * 4] * 2 * Math.PI, b = data[(off + i) * 4 + 1] * 2 * Math.PI;
    sx += Math.sin(a); cx += Math.cos(a); sy += Math.sin(b); cy += Math.cos(b);
  }
  const f = (s, c) => ((Math.atan2(s, c) / (2 * Math.PI)) + 1) % 1;
  return [f(sx, cx), f(sy, cy)];
}

export function scorePL(A, B, b, N, species, m, DT) {
  const off = b * N, MIN = 12;
  const all = clusters(B, off, N, 0.025);
  const cl = all.filter(c => c.length >= MIN);
  const inCl = cl.reduce((s, c) => s + c.length, 0);
  const clusteredFrac = inCl / N;
  const largestFrac = cl.length ? Math.max(...cl.map(c => c.length)) / N : 0;

  let speed = 0;
  for (let i = 0; i < N; i++) {
    const o = (off + i) * 4;
    speed += Math.hypot(wrap(B[o] - A[o]), wrap(B[o + 1] - A[o + 1]));
  }
  speed /= N * DT;

  let motion = 0, rot = 0, memb = 0, purity = 0, w = 0, mw = 0, persist = 0;
  const sizes = [];
  for (const c of cl) {
    const n = c.length; sizes.push(n);
    const cB = centroid(B, off, c), cA = centroid(A, off, c);
    const dcx = wrap(cB[0] - cA[0]), dcy = wrap(cB[1] - cA[1]);
    motion += n * Math.hypot(dcx, dcy) / DT;
    // rotation: mean signed angle change about the moving centroid, and coherence
    let dth = 0, R2 = 0, stay = 0;
    const rad = [];
    for (const i of c) {
      const o = (off + i) * 4;
      const bx = wrap(B[o] - cB[0]), by = wrap(B[o + 1] - cB[1]);
      const ax = wrap(A[o] - cA[0]), ay = wrap(A[o + 1] - cA[1]);
      const ra = Math.hypot(ax, ay), rb = Math.hypot(bx, by);
      if (ra > 0.004 && rb > 0.004 && ra < 0.2) dth += Math.atan2(ax * by - ay * bx, ax * bx + ay * by);
      if (ra < 0.2) stay++;
      R2 += rb * rb; rad.push(rb);
    }
    rot += n * Math.abs(dth / n) / DT;
    persist += stay;
    // species composition & radial layering
    const cnt = new Float32Array(m), rs = new Float32Array(m), rs2 = new Float32Array(m);
    c.forEach((i, k) => { const s = species[off + i]; cnt[s]++; rs[s] += rad[k]; rs2[s] += rad[k] * rad[k]; });
    purity += n * Math.max(...cnt) / n;
    const R = Math.sqrt(R2 / n) + 1e-6;
    const present = [...cnt.keys()].filter(s => cnt[s] >= Math.max(4, n * 0.08));
    if (present.length >= 2 && n >= 30) {
      const mr = present.map(s => rs[s] / cnt[s]);
      const outer = present[mr.indexOf(Math.max(...mr))];
      const mo = rs[outer] / cnt[outer], so = Math.sqrt(Math.max(0, rs2[outer] / cnt[outer] - mo * mo));
      const innerMean = present.filter(s => s !== outer).reduce((a, s) => a + rs[s], 0) /
        present.filter(s => s !== outer).reduce((a, s) => a + cnt[s], 0);
      const shell = Math.max(0, 1 - so / (mo + 1e-6));     // thin ring → 1
      memb += n * Math.max(0, (mo - innerMean) / R) * shell;
      mw += n;
    }
    w += n;
  }
  if (w) { motion /= w; rot /= w; purity /= w; persist /= w; }
  if (mw) memb /= mw;
  const meanSize = sizes.length ? inCl / sizes.length : 0;
  const sizeCV = sizes.length > 1 ? Math.sqrt(sizes.reduce((s, x) => s + (x - meanSize) ** 2, 0) / sizes.length) / meanSize : 0;

  const c01 = x => Math.max(0, Math.min(1, x));
  const features = [
    c01(clusteredFrac),
    c01(Math.log1p(cl.length) / Math.log(80)),
    c01(largestFrac),
    c01(speed / 0.4),
    c01(motion / 0.2),
    c01(rot / 3),
    c01(memb * 1.5),
    c01((purity - 1 / m) / (1 - 1 / m + 1e-6)),
    c01(sizeCV / 2),
    c01(Math.log1p(meanSize) / Math.log(N)),
  ];
  // interest: structured (not gas, not one blob) + dynamics + layering
  const structure = clusteredFrac > 0.15 && cl.length >= 1 ? Math.min(1, clusteredFrac * 1.4) * (largestFrac > 0.85 ? 0.4 : 1) : 0;
  const score = structure * (0.25 + 0.25 * c01(cl.length / 6)) + 0.6 * c01(motion / 0.15) * structure
    + 0.6 * c01(rot / 2) * structure + 0.8 * c01(memb * 1.5) * structure;
  let label = 'gas';
  if (structure > 0) {
    if (memb > 0.5) label = 'membranes';
    else if (rot > 1.2) label = 'rotators';
    else if (motion > 0.08) label = 'movers';
    else if (cl.length >= 4) label = 'clusters';
    else label = 'blobs';
  }
  return { features, score, label, stats: { clusters: cl.length, clusteredFrac, largestFrac, speed, motion, rot, memb, purity, persist } };
}

export const PL_LABELS = ['gas', 'blobs', 'clusters', 'movers', 'rotators', 'membranes'];
export const PL_COLORS = { gas: '#5b6170', blobs: '#4fb3bf', clusters: '#e2b85c', movers: '#b48ce6', rotators: '#e85d9a', membranes: '#73db85' };
