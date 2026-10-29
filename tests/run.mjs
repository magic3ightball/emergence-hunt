// node tests/run.mjs — checks the CPU analysis code on synthetic inputs.
import assert from 'node:assert/strict';
import { analyzeField, classify } from '../src/gsclassify.js';
import { scorePL } from '../src/plscore.js';

const T = 128;
const field = f => { const v = new Float32Array(T * T); for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) v[y * T + x] = f(x, y); return v; };
const spots = (ox = 0) => field((x, y) => {
  const dx = ((x - ox) % 16 + 16) % 16 - 8, dy = (y % 16) - 8;
  return dx * dx + dy * dy < 16 ? 0.35 : 0.02;
});
const stripes = field((x) => (x % 12 < 5 ? 0.3 : 0.02));
const flat = field(() => 0.0);
const hist = v => [{ t: 300, a: analyzeField(v, T) }, { t: 1000, a: analyzeField(v, T) }];
let n = 0;
const ok = (name, got, want) => { assert.equal(got, want, name); n++; console.log('ok', name, got); };

ok('uniform', classify(hist(flat), flat, flat, T, 200).label, 'uniform');
ok('static spots', classify(hist(spots()), spots(), spots(), T, 200).label, 'spots');
ok('stripes', classify(hist(stripes), stripes, stripes, T, 200).label, 'stripes');
ok('moving spots', classify(hist(spots()), spots(0), spots(2), T, 200).label, 'moving solitons');
const one = field((x, y) => ((x - 64) ** 2 + (y - 64) ** 2 < 16 ? 0.35 : 0.02));
ok('replicating spots', classify([{ t: 300, a: analyzeField(one, T) }, { t: 1000, a: analyzeField(one, T) }], spots(), spots(), T, 200).label, 'self-replicating');
const disc = r => field((x, y) => ((x - 64) ** 2 + (y - 64) ** 2 < r * r ? 0.35 : 0.02));
const H = (...vs) => vs.map((v, i) => ({ t: [300, 1500, 3500, 6000][i], a: analyzeField(v, T) }));
ok('growing blob', classify(H(disc(4), disc(7), disc(10), disc(13)), disc(17), disc(17), T, 200).label, 'expanding');
const holes = r => field((x, y) => ((x - 64) ** 2 + (y - 64) ** 2 < r * r ? 0.02 : 0.35));
ok('shrinking hole in a bright sea is not expanding', classify(H(holes(30), holes(26), holes(22), holes(18)), holes(14), holes(14), T, 200).label !== 'expanding', true);
const sea = field((x, y) => (((x % 16) - 8) ** 2 + ((y % 16) - 8) ** 2 < 16 ? 0.02 : 0.35));
ok('pattern that took over the tile (bright sea with holes) is not expanding', classify(H(disc(6), disc(12), disc(30), disc(50)), sea, sea, T, 200).label !== 'expanding', true);
// U-skate-like: a dark U (low v) in a bright blue-state sea, moving ~0.33 cells per 200 steps (Munafo's speed at Pearson's grid)
const U = (ox, arm = 8) => field((x, y) => {
  const dx = x - 40 - ox, dy = y - 64, r = Math.hypot(dx, dy);
  const inArc = Math.abs(r - 9) < 2.5 && dy <= 0;                           // closed end
  const inArm = dy > 0 && dy < arm && (Math.abs(dx - 9) < 2.5 || Math.abs(dx + 9) < 2.5);
  return inArc || inArm ? 0.05 : 0.29;
});
const at = t => U(Math.round(t / 200 * 0.33));
const UH = f => [300, 1500, 3500, 6000].map(t => ({ t, a: analyzeField(f(t), T) }));
const skate = classify(UH(at), at(9800), at(10000), T, 200, 10000);
ok('slow U-skate-like drift is a moving soliton', skate.label, 'moving solitons');
ok('static U is not moving', classify(UH(() => U(0)), U(0), U(0), T, 200, 10000).label !== 'moving solitons', true);
const settle = t => U(Math.round(12 * (1 - Math.exp(-t / 2500))));                  // decelerating: settling, not gliding
ok('settling U is not a glider', classify(UH(settle), settle(9800), settle(10000), T, 200, 10000).label !== 'moving solitons', true);
const hair = t => U(0, 8 + Math.round(t / 400));        // arms keep growing (hairpin)
ok('growing hairpin is not a glider', classify(UH(hair), hair(9800), hair(10000), T, 200, 10000).label !== 'moving solitons', true);
const creep = t => U(0, 8 + Math.round(t / 2000));      // arms lengthen ~1 cell / 2000 steps
ok('slowly lengthening arc is not a glider', classify(UH(creep), creep(9800), creep(10000), T, 200, 10000).label !== 'moving solitons', true);
const blobs = n => field((x, y) => { for (let i = 0; i < n; i++) { const cx = 16 + 32 * (i % 4), cy = 16 + 32 * ((i / 4) | 0); if ((x - cx) ** 2 + (y - cy) ** 2 < 16) return 0.35; } return 0.02; });
ok('slow division 1→2→4→8', classify(H(blobs(1), blobs(1), blobs(2), blobs(4)), blobs(8), blobs(8), T, 200).label, 'self-replicating');
const fewer = field((x, y) => (y < 112 ? spots(2)[y * T + x] : 0.02));
ok('moving but splitting/dying is not a glider', classify(hist(spots()), spots(0), fewer, T, 200).label !== 'moving solitons', true);
const a = analyzeField(field((x, y) => (((x % 16) - 8) ** 2 + ((y % 16) - 8) ** 2 < 16 ? 0.02 : 0.35)), T);
ok('negative spots use low phase', a.fgHigh, 0);
ok('negative spot count', a.count, 64);

// Particle Life: 4 tight clusters, one rotating
const N = 512, A = new Float32Array(N * 4), B = new Float32Array(N * 4), sp = new Uint32Array(N);
for (let i = 0; i < N; i++) {
  const c = i % 4, cx = 0.2 + 0.2 * c, cy = 0.5, ang = i * 0.37, r = 0.004 + (i % 7) * 0.0025;
  const rot = c === 0 ? 0.3 : 0;
  A[i * 4] = cx + r * Math.cos(ang); A[i * 4 + 1] = cy + r * Math.sin(ang);
  B[i * 4] = cx + r * Math.cos(ang + rot); B[i * 4 + 1] = cy + r * Math.sin(ang + rot);
  sp[i] = c % 2;
}
const s = scorePL(A, B, 0, N, sp, 2, 0.5);
ok('pl clusters', s.stats.clusters, 4);
assert.ok(s.stats.rot > 0.1, 'rotation detected'); n++; console.log('ok pl rotation', s.stats.rot.toFixed(3));
console.log(`${n} checks passed`);
