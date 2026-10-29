import { GrayScott, makeIC, packTiles, mapIC, tileThumb, ICS } from './gs.js';
import { analyzeField, classify, GS_LABELS, LABEL_COLORS } from './gsclassify.js';
import { ParticleLife, randomGenome, mutateGenome, plThumb } from './pl.js';
import { scorePL, PL_COLORS } from './plscore.js';
import { Gallery } from './gallery.js';
import { PEARSON, SOURCES } from './known.js';
import { randSeed } from './rng.js';

const $ = id => document.getElementById(id);
const status = t => ($('status').textContent = t);
// Yield without timers: setTimeout is throttled to ≥1s when the page is hidden.
const mc = new MessageChannel(), waiters = [];
mc.port1.onmessage = () => waiters.shift()?.();
const tick = () => new Promise(r => { waiters.push(r); mc.port2.postMessage(0); });
let gpuLive = true; // false once a GPU round trip proves throttled (pane not composited)
const pause = ms => (document.hidden || !gpuLive) ? tick() : new Promise(r => setTimeout(r, ms));

// ---------- GPU ----------
async function initGPU() {
  if (!navigator.gpu) throw new Error('WebGPU 없음');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const device = await adapter.requestDevice();
  device.lost.then(i => {
    // GPU process resets happen (seen with a backgrounded pane). Reload and resume running searches.
    status('GPU lost: ' + i.message + ' — 다시 시작합니다');
    try { sessionStorage.setItem('eh-resume', JSON.stringify({ scan: scanning, pl: plRunning, n: +(sessionStorage.getItem('eh-lost') || 0) + 1 }));
      sessionStorage.setItem('eh-lost', +(sessionStorage.getItem('eh-lost') || 0) + 1); } catch { }
    setTimeout(() => location.reload(), 500);
  });
  device.addEventListener('uncapturederror', e => { console.error(e.error.message); status('GPU 오류: ' + e.error.message.slice(0, 80)); });
  const mods = new Map();
  return { device, format: navigator.gpu.getPreferredCanvasFormat(),
    module: code => mods.get(code) || mods.set(code, device.createShaderModule({ code })).get(code) };
}
const settle = gpu => gpu.device.queue.onSubmittedWorkDone();
// A pane that isn't composited (even with visibilityState 'visible') gets WebGPU callbacks
// throttled to seconds, so grow the steps-per-submit when a round trip is slow.
const pacer = { n: {}, last: 0 };
function chunk(base, rest) {
  const now = performance.now(), dt = now - pacer.last; pacer.last = now;
  let n = pacer.n[base] || base;
  if (dt > 500 && dt < 120000) n = Math.min(n * 2, base * 40); else if (dt < 120) n = Math.max(base, n / 2);
  pacer.n[base] = n;
  return Math.max(1, Math.min(rest, Math.round(n)));
}

const gallery = new Gallery({ capacity: 40 });
let gpu;

// ---------- tabs ----------
document.querySelectorAll('nav button').forEach(b => b.onclick = () => showTab(b.dataset.tab));
function showTab(t) {
  document.querySelectorAll('nav button').forEach(b => b.classList.toggle('on', b.dataset.tab === t));
  document.querySelectorAll('main > section').forEach(s => s.classList.toggle('on', s.id === t));
}

// =====================================================================
// Gray-Scott
// =====================================================================
const MAP = { f0: 0.0, f1: 0.12, k0: 0.03, k1: 0.075 };
let gsMain = null;          // current sim on main canvas
let gsJob = 0;              // increments to cancel running loops
let scanning = false;
let scanPoints = [];        // {F,k,label,ic}
let scanCounter = 0;
try { scanPoints = JSON.parse(localStorage.getItem('eh-scan') || '[]'); scanCounter = scanPoints.length; } catch { }
const saveScan = () => { try { localStorage.setItem('eh-scan', JSON.stringify(scanPoints.slice(-6000))); } catch { } };

function setMainSim(opts) {
  gsMain?.destroy();
  gsMain = new GrayScott(gpu, { ...opts, canvas: $('gsCanvas') });
  $('gsOverlay').innerHTML = '';
  return gsMain;
}

async function runMap() {
  const job = ++gsJob; scanning = false; updateScanBtn();
  const sim = setMainSim({ mode: 1, W: 900, H: 1080 });
  Object.assign(sim.map, MAP); sim.writeUniform();
  sim.upload(mapIC(sim.W, sim.H, 12345));
  // Pearson / Munafo class markers
  const ov = $('gsOverlay');
  for (const p of PEARSON) {
    const x = (p.k - MAP.k0) / (MAP.k1 - MAP.k0) * 100, y = (1 - (p.F - MAP.f0) / (MAP.f1 - MAP.f0)) * 100;
    ov.insertAdjacentHTML('beforeend', `<div class="pt" style="left:${x}%;top:${y}%"></div><div class="pl" style="left:${x}%;top:${y}%">${p.id}</div>`);
  }
  ov.insertAdjacentHTML('beforeend', `<div class="tag" style="left:6px;bottom:6px">k ${MAP.k0} → ${MAP.k1} (가로) · F ${MAP.f0} → ${MAP.f1} (세로)</div>`);
  const total = 20000, t0 = performance.now();
  while (sim.steps < total && job === gsJob) {
    sim.run(chunk(100, total - sim.steps)); sim.render(); await settle(gpu);
    $('gsProg').style.width = (sim.steps / total * 100) + '%';
    const sps = sim.steps / ((performance.now() - t0) / 1000);
    $('gsInfo').textContent = `F,k 맵 ${sim.W}×${sim.H}, 스텝 ${sim.steps}/${total} (${sps | 0} 스텝/초). 흰 원 = Pearson/Munafo 클래스 위치.`;
  }
}

// Run a batch of tiles; returns results with classification.
async function runBatch(items, { cols, rows, T = 128, steps, job, label, filmstrip = false, stencil = 0 }) {
  const sim = setMainSim({ T, cols, rows, stencil });
  sim.setTileParams(items);
  sim.upload(packTiles(sim, items.map(it => it.icData || makeIC(it.ic, T, it.seed, it.F, it.k)))); // icData: custom {u,v} (tests)
  const DT = 200;
  const checks = [300, Math.round(steps * 0.15), Math.round(steps * 0.35), Math.round(steps * 0.6), steps - DT, steps];
  sim.allocSnaps(checks.length);
  const t0 = performance.now();
  let live = true; // wait for the GPU between chunks (progress UI) until a round trip turns out throttled
  for (let ci = 0; ci < checks.length; ci++) {
    while (sim.steps < checks[ci]) {
      if (job !== gsJob) return null;
      sim.run(Math.min(200, checks[ci] - sim.steps)); sim.render();
      if (live) {
        const t = performance.now(); await settle(gpu);
        if (performance.now() - t > 400) live = gpuLive = false;
        $('gsProg').style.width = (sim.steps / steps * 100) + '%';
        $('gsInfo').textContent = `${label}: 스텝 ${sim.steps}/${steps} (${(sim.steps / ((performance.now() - t0) / 1000)) | 0}/초)`;
      }
    }
    sim.snapshot(ci);
  }
  $('gsInfo').textContent = `${label}: GPU 마무리 대기…`;
  const snaps = await sim.readSnaps().catch(() => null); // rejected if the sim was replaced mid-map
  if (!snaps || job !== gsJob) return null;
  $('gsProg').style.width = '100%';
  $('gsInfo').textContent = `${label}: ${steps} 스텝, ${((performance.now() - t0) / 1000).toFixed(1)}초`;
  const A = snaps[checks.length - 2], B = snaps[checks.length - 1];
  return items.map((it, i) => {
    const hist = checks.slice(0, -2).map((t, k) => ({ t, a: analyzeField(sim.tileV(snaps[k], i), T) }));
    const vB = sim.tileV(B, i), vA = sim.tileV(A, i);
    const r = { ...it, ...classify(hist, vA, vB, T, DT, steps), thumb: tileThumb(vB, T), steps };
    if (filmstrip) r.film = snaps.map((s, k) => ({ t: checks[k], img: tileThumb(sim.tileV(s, i), T) })); // review/debug only
    return r;
  });
}

function tileOverlay(results, cols, rows, onClick) {
  const ov = $('gsOverlay'); ov.innerHTML = '';
  results.forEach((r, i) => {
    const x = (i % cols) / cols * 100, y = ((i / cols) | 0) / rows * 100;
    const el = document.createElement('div');
    el.className = 'cell';
    el.style.cssText = `left:${x}%;top:${y}%;width:${100 / cols}%;height:${100 / rows}%`;
    el.title = `F=${r.F.toFixed(4)} k=${r.k.toFixed(4)} ic=${r.ic}\n${r.label}\n` + Object.entries(r.stats).map(([k, v]) => `${k}=${(+v).toFixed(3)}`).join(' ');
    el.innerHTML = `<div class="tag" style="left:2px;top:2px;color:${LABEL_COLORS[r.label]}">${r.id ? r.id + ' ' : ''}${r.label}${r.isNew ? ' ★' : ''}</div>`;
    el.onclick = () => onClick(r);
    ov.appendChild(el);
  });
}

async function runBench() {
  const job = ++gsJob; scanning = false; updateScanBtn();
  const items = PEARSON.map(p => ({ ...p, ic: p.ic || 'square', seed: 1993 }));
  const cols = 5, rows = 4, steps = +$('gsSteps').value, T = 256;
  const res = await runBatch(items, { cols, rows, steps, job, T, label: 'Pearson 재현 (256² 타일 = Pearson 영역 크기)' });
  if (!res) return;
  tileOverlay(res, cols, rows, r => replayGS({ F: r.F, k: r.k, ic: r.ic, seed: r.seed, steps, T }));
  let ok = 0;
  const rowsHtml = res.map(r => {
    const m = r.expect.includes(r.label); ok += m;
    return `<tr><td>${r.id}</td><td>${r.F}</td><td>${r.k}</td><td title="${r.desc}">${r.desc.slice(0, 26)}</td><td>${r.ic}</td><td class="${m ? 'ok' : 'bad'}">${r.label}</td></tr>`;
  }).join('');
  $('gsBenchCard').style.display = '';
  $('gsBench').innerHTML = `<div class="small">일치 ${ok}/${res.length} (${steps} 스텝, Pearson 상수, 256² 주기 경계. 초기조건은 Pearson 중앙 사각형, Munafo가 다른 시작 상태를 적은 ν·ρ·σ만 그에 맞춤)</div>
    <table><tr><th></th><th>F</th><th>k</th><th>알려진 모습</th><th>초기</th><th>자동 분류</th></tr>${rowsHtml}</table>`;
  for (const r of res) {
    await gallery.addKnown({ id: 'known-gs-' + r.id, mode: 'gs', title: `Pearson ${r.id}`, label: r.label, features: r.features,
      thumb: r.thumb, params: { F: r.F, k: r.k, ic: r.ic, seed: r.seed, steps, T }, stats: r.stats, score: 0, desc: r.desc });
  }
  window.__eh.bench = res.map(r => ({ id: r.id, label: r.label, match: r.expect.includes(r.label), stats: r.stats }));
  status(`Pearson 재현 ${ok}/${res.length} 일치`);
}

// R2 low-discrepancy sequence over (F, k-offset) for an even, progressively finer scan.
function scanSample(n) {
  const g = 1.32471795724474602596, a1 = 1 / g, a2 = 1 / (g * g);
  const u = (0.5 + a1 * n) % 1, w = (0.5 + a2 * n) % 1;
  const F = 0.003 + u * (0.115 - 0.003);
  const kSN = Math.sqrt(F) / 2 - F;               // saddle-node line of the homogeneous state
  const k = Math.min(0.075, Math.max(0.03, kSN - 0.0065 + w * 0.012)); // patterns hug this line
  return { F, k };
}

// space-filling seeds survive in more of the plane, so weight them up
const IC_POOL = [...ICS, 'square', 'noise', 'noise', 'inverse', 'inverse', 'checker'];

async function scanLoop() {
  const job = ++gsJob;
  const cols = 10, rows = 10;
  let batchNo = 0;
  while (scanning && job === gsJob) {
    const odd = $('gsOddIC').checked;
    const items = [];
    const hits = scanPoints.filter(q => q.label !== 'uniform');
    for (let i = 0; i < cols * rows; i++) {
      // half global low-discrepancy coverage, half refinement around earlier non-uniform outcomes
      let p;
      if (hits.length > 10 && i % 2) {
        const h = hits[(Math.random() * hits.length) | 0];
        p = { F: Math.max(0.002, h.F + (Math.random() - 0.5) * 0.004), k: h.k + (Math.random() - 0.5) * 0.0016 };
      } else p = scanSample(scanCounter++);
      items.push({ ...p, ic: odd ? IC_POOL[(Math.random() * IC_POOL.length) | 0] : 'square', seed: randSeed() });
    }
    const steps = +$('gsSteps').value;
    const res = await runBatch(items, { cols, rows, steps, job, label: `스캔 판 ${++batchNo}` });
    if (!res) return;
    let added = 0;
    for (const r of res) {
      scanPoints.push({ F: +r.F.toFixed(5), k: +r.k.toFixed(5), label: r.label, ic: r.ic });
      if (r.label === 'uniform') continue;
      const it = await gallery.consider({ mode: 'gs', title: `F ${r.F.toFixed(4)} k ${r.k.toFixed(4)}`, label: r.label,
        features: r.features, thumb: r.thumb, params: { F: r.F, k: r.k, ic: r.ic, seed: r.seed, steps }, stats: r.stats,
        score: r.stats.act * 20 + (r.label === 'moving solitons' || r.label === 'self-replicating' ? 1 : 0) }, { minNovelty: 0.18 });
      if (it) { r.isNew = true; added++; }
    }
    saveScan(); drawScatter();
    tileOverlay(res, cols, rows, r => replayGS({ F: r.F, k: r.k, ic: r.ic, seed: r.seed, steps }));
    const counts = {}; res.forEach(r => counts[r.label] = (counts[r.label] || 0) + 1);
    status(`스캔 판 ${batchNo}: 갤러리 신규 ${added} · ` + Object.entries(counts).map(([k, v]) => `${k} ${v}`).join(', '));
    window.__eh.lastScan = { batchNo, added, counts };
    await pause(1500); // let the eye see the batch
  }
}

function updateScanBtn() { $('gsScanBtn').textContent = scanning ? '스캔 멈춤' : '자동 스캔 시작'; }
$('gsScanBtn').onclick = () => { scanning = !scanning; updateScanBtn(); if (scanning) scanLoop(); else gsJob++; };
$('gsMapBtn').onclick = runMap;
$('gsBenchBtn').onclick = runBench;

function drawScatter() {
  const cv = $('gsScatter'), g = cv.getContext('2d'), W = cv.width, H = cv.height, pad = 34;
  const kx = k => pad + (k - 0.03) / (0.075 - 0.03) * (W - pad - 8);
  const fy = F => H - pad - (F / 0.12) * (H - pad - 8);
  g.fillStyle = '#07080c'; g.fillRect(0, 0, W, H);
  g.strokeStyle = '#232837'; g.fillStyle = '#8a90a0'; g.font = '16px sans-serif';
  for (let k = 0.03; k <= 0.0751; k += 0.01) { g.beginPath(); g.moveTo(kx(k), 8); g.lineTo(kx(k), H - pad); g.stroke(); g.fillText(k.toFixed(2), kx(k) - 16, H - 10); }
  for (let F = 0; F <= 0.12; F += 0.02) { g.beginPath(); g.moveTo(pad, fy(F)); g.lineTo(W - 8, fy(F)); g.stroke(); g.fillText(F.toFixed(2), 0, fy(F) + 5); }
  g.strokeStyle = '#ffffff55'; g.setLineDash([4, 4]); g.beginPath();
  for (let F = 0.0005; F < 0.125; F += 0.001) { const k = Math.sqrt(F) / 2 - F; F < 0.001 ? g.moveTo(kx(k), fy(F)) : g.lineTo(kx(k), fy(F)); }
  g.stroke(); g.setLineDash([]);
  for (const p of scanPoints) { g.fillStyle = LABEL_COLORS[p.label] || '#fff'; g.fillRect(kx(p.k) - 2.5, fy(p.F) - 2.5, 5, 5); }
  g.font = 'bold 17px sans-serif';
  for (const p of PEARSON) {
    g.strokeStyle = '#fff'; g.lineWidth = 2; g.beginPath(); g.arc(kx(p.k), fy(p.F), 6, 0, 7); g.stroke(); g.lineWidth = 1;
    g.fillStyle = '#fff'; g.fillText(p.id, kx(p.k) + 8, fy(p.F) - 6);
  }
  g.fillStyle = '#8a90a0'; g.font = '15px sans-serif'; g.fillText('k →', W - 40, H - pad - 6); g.fillText('F ↑', pad + 4, 22);
  const counts = {}; scanPoints.forEach(p => counts[p.label] = (counts[p.label] || 0) + 1);
  $('gsScanStats').textContent = `스캔 ${scanPoints.length}점. 점선 = 균일상태 안장-마디 곡선 k=√F/2−F. ` + GS_LABELS.map(l => `${l} ${counts[l] || 0}`).join(' · ');
}
$('gsLegend').innerHTML = GS_LABELS.map(l => `<span><i style="background:${LABEL_COLORS[l]}"></i>${l}</span>`).join('');

// --- GS replay viewer ---
let gsView = null, gsViewJob = 0;
function replayGS(p) {
  const job = ++gsViewJob;
  gsView?.destroy();
  const VT = p.T || 128;
  gsView = new GrayScott(gpu, { T: VT, cols: 1, rows: 1, canvas: $('gsView') });
  gsView.setTileParams([{ F: p.F, k: p.k }]);
  gsView.upload(packTiles(gsView, [makeIC(p.ic, VT, p.seed, p.F, p.k)]));
  showTab('gs');
  const total = Math.max(p.steps || 10000, 30000);
  const loop = async () => {
    if (job !== gsViewJob) return;
    gsView.run(40); gsView.render(); await settle(gpu);
    $('gsViewInfo').textContent = `F=${p.F.toFixed(4)} k=${p.k.toFixed(4)} 초기조건 ${p.ic} seed ${p.seed} · 스텝 ${gsView.steps}` + (gsView.steps < (p.steps || 0) ? ` (저장 시점 ${p.steps})` : '');
    if (gsView.steps < total) setTimeout(loop, 16);
  };
  loop();
}

// =====================================================================
// Particle Life
// =====================================================================
let plMain = null, plRunning = false, plJob = 0, plBatchNo = 0;
const PL_N = 1024, PL_B = 16;
const PL_GRID = '<div style="position:absolute;inset:0;background-image:linear-gradient(#ffffff22 1px,transparent 1px),linear-gradient(90deg,#ffffff22 1px,transparent 1px);background-size:25% 25%"></div>';

async function plLoop() {
  const job = ++plJob;
  if (!plMain) plMain = new ParticleLife(gpu, { N: PL_N, B: PL_B, gcols: 4, canvas: $('plCanvas') });
  while (plRunning && job === plJob) {
    const parents = gallery.pool('pl').filter(i => !i.known).sort((a, b) => b.novelty - a.novelty).slice(0, 12);
    const genomes = [];
    for (let b = 0; b < PL_B; b++) {
      if ($('plMutate').checked && parents.length && b >= PL_B / 2) {
        const p = parents[(Math.random() * parents.length) | 0];
        genomes.push({ ...mutateGenome(p.params.genome, randSeed(), 0.3), parent: p.id });
      } else genomes.push(randomGenome(randSeed()));
    }
    plMain.load(genomes);
    $('plOverlay').innerHTML = PL_GRID;
    const warm = 1200, DTs = 25, t0 = performance.now();
    plBatchNo++;
    plMain.allocSnaps(2);
    let live = true;
    while (plMain.steps < warm) {
      if (job !== plJob) return;
      plMain.run(Math.min(50, warm - plMain.steps)); plMain.render();
      if (live) {
        const t = performance.now(); await settle(gpu);
        if (performance.now() - t > 400) live = gpuLive = false;
        $('plProg').style.width = (plMain.steps / warm * 100) + '%';
        $('plInfo').textContent = `판 ${plBatchNo}: 스텝 ${plMain.steps}/${warm} (${(plMain.steps / ((performance.now() - t0) / 1000)) | 0}/초 × 16 우주)`;
      }
    }
    plMain.snapshot(0);
    plMain.run(DTs); plMain.snapshot(1);
    const snaps = await plMain.readSnaps().catch(() => null);
    if (!snaps || job !== plJob) return;
    const [A, B] = snaps;
    $('plInfo').textContent = `판 ${plBatchNo}: ${warm + DTs} 스텝, ${((performance.now() - t0) / 1000).toFixed(1)}초`;
    const DT = DTs * plMain.cfg.dt;
    const res = genomes.map((g, b) => ({ g, b, ...scorePL(A, B, b, PL_N, plMain.species, g.m, DT) }));
    let added = 0;
    for (const r of res) {
      r.thumb = plThumb(B, r.b, PL_N, plMain.species);
      if (r.label === 'gas' || r.score < 0.25) continue;
      const it = await gallery.consider({ mode: 'pl', title: `${r.g.m}종 ${r.label}`, label: r.label, features: r.features,
        thumb: r.thumb, score: r.score, stats: r.stats, params: { genome: { m: r.g.m, mat: r.g.mat, seed: r.g.seed } } }, { minNovelty: 0.15 });
      if (it) { r.isNew = true; added++; }
    }
    // overlay labels
    const ov = $('plOverlay');
    ov.innerHTML = PL_GRID;
    res.forEach(r => {
      const x = (r.b % 4) * 25, y = ((r.b / 4) | 0) * 25;
      const el = document.createElement('div'); el.className = 'cell';
      el.style.cssText = `left:${x}%;top:${y}%;width:25%;height:25%`;
      el.innerHTML = `<div class="tag" style="left:3px;top:3px;color:${PL_COLORS[r.label]}">${r.label} ${r.score.toFixed(2)}${r.isNew ? ' ★' : ''}</div>`;
      el.onclick = () => replayPL({ genome: r.g });
      ov.appendChild(el);
    });
    $('plTable').innerHTML = '<table><tr><th>#</th><th>종</th><th>분류</th><th>점수</th><th>군집</th><th>이동</th><th>회전</th><th>막</th></tr>' +
      res.slice().sort((a, b) => b.score - a.score).map(r => `<tr><td>${r.b}</td><td>${r.g.m}</td><td style="color:${PL_COLORS[r.label]}">${r.label}${r.isNew ? ' ★' : ''}</td><td>${r.score.toFixed(2)}</td><td>${r.stats.clusters}</td><td>${r.stats.motion.toFixed(3)}</td><td>${r.stats.rot.toFixed(2)}</td><td>${r.stats.memb.toFixed(2)}</td></tr>`).join('') + '</table>';
    status(`PL 판 ${plBatchNo}: 갤러리 신규 ${added}`);
    window.__eh.lastPL = { batch: plBatchNo, added, labels: res.map(r => r.label), scores: res.map(r => +r.score.toFixed(2)) };
    // keep showing the batch a moment
    for (let i = 0; i < 20 && plRunning && job === plJob; i++) { plMain.run(4); plMain.render(); await pause(40); }
  }
}
$('plRunBtn').onclick = () => {
  plRunning = !plRunning; $('plRunBtn').textContent = plRunning ? '탐색 멈춤' : '랜덤 탐색 시작';
  if (plRunning) plLoop(); else plJob++;
};

let plView = null, plViewJob = 0;
function replayPL(p) {
  const job = ++plViewJob;
  if (!plView) { plView = new ParticleLife(gpu, { N: PL_N, B: 1, gcols: 1, canvas: $('plView') }); plView.cfg.psize = 0.006; plView.writeUniform(); }
  plView.load([p.genome]);
  showTab('pl');
  const m = p.genome.m;
  const matTxt = Array.from({ length: m }, (_, i) => p.genome.mat.slice(i * m, i * m + m).map(a => (a >= 0 ? '+' : '') + a.toFixed(2)).join(' ')).join('<br>');
  const loop = async () => {
    if (job !== plViewJob) return;
    plView.run(3); plView.render(); await settle(gpu);
    $('plViewInfo').innerHTML = `${m}종 · seed ${p.genome.seed} · 스텝 ${plView.steps}<div style="font-family:ui-monospace,monospace;font-size:10.5px;margin-top:4px">${matTxt}</div>`;
    setTimeout(loop, 16);
  };
  loop();
}

// =====================================================================
// Gallery UI
// =====================================================================
function renderGallery() {
  const mode = $('galMode').value, sort = $('galSort').value, showKnown = $('galKnown').checked;
  let items = gallery.items.filter(i => (mode === 'all' || i.mode === mode) && (showKnown || !i.known));
  items.sort((a, b) => sort === 'time' ? b.time - a.time : sort === 'score' ? (b.score || 0) - (a.score || 0) : b.novelty - a.novelty);
  $('galCount').textContent = `(${gallery.items.filter(i => !i.known).length})`;
  const grid = $('galGrid'); grid.innerHTML = '';
  for (const it of items) {
    const el = document.createElement('div');
    el.className = 'find' + (it.known ? ' known' : '');
    const col = (it.mode === 'gs' ? LABEL_COLORS : PL_COLORS)[it.label] || '#fff';
    el.innerHTML = `<img src="${it.thumb}" alt=""><div class="m"><div class="lab" style="color:${col}">${it.mode === 'gs' ? 'GS' : 'PL'} · ${it.label}</div>
      <div>${it.title}</div><div class="small">새로움 ${isFinite(it.novelty) ? it.novelty.toFixed(2) : '∞'}${it.nearest ? ' · 가까운: ' + it.nearest : ''}</div>
      ${it.params?.ic ? `<div class="small">초기조건 ${it.params.ic}</div>` : ''}</div>${it.known ? '' : '<button class="x" title="삭제">✕</button>'}`;
    el.onclick = e => {
      if (e.target.classList.contains('x')) { e.stopPropagation(); gallery.remove(it.id); return; }
      it.mode === 'gs' ? replayGS(it.params) : replayPL(it.params);
    };
    grid.appendChild(el);
  }
  if (!items.length) grid.innerHTML = '<div class="small">아직 발견이 없습니다. Gray-Scott 자동 스캔이나 Particle Life 탐색을 돌려 보세요.</div>';
}
gallery.on(renderGallery);
['galMode', 'galSort', 'galKnown'].forEach(id => $(id).onchange = renderGallery);
$('galExport').onclick = () => {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([gallery.export()], { type: 'application/json' }));
  a.download = 'emergence-hunt-gallery.json'; a.click();
};
$('galClear').onclick = () => { if (confirm('저장된 발견을 모두 지울까요? (Pearson 기준은 유지)')) gallery.clearFinds(); };
$('sources').innerHTML = '<b>출처</b><br>' + SOURCES.map(([t, u]) => `<a href="${u}" target="_blank" rel="noopener">${t}</a>`).join('<br>');

// =====================================================================
(async () => {
  try { gpu = await initGPU(); } catch (e) { status('WebGPU를 쓸 수 없습니다: ' + e.message); return; }
  await gallery.load();
  await gallery.prune('gs'); await gallery.prune('pl'); gallery.emit();
  drawScatter();
  status('준비됨');
  window.__eh = { gpu, gallery, runMap, runBench, replayGS, replayPL, runBatch, newJob: () => ++gsJob,
    startScan: () => { if (!scanning) $('gsScanBtn').click(); }, stopScan: () => { if (scanning) $('gsScanBtn').click(); },
    startPL: () => { if (!plRunning) $('plRunBtn').click(); }, stopPL: () => { if (plRunning) $('plRunBtn').click(); },
    scanPoints: () => scanPoints, showTab };
  let resume = null;
  try { resume = JSON.parse(sessionStorage.getItem('eh-resume') || 'null'); sessionStorage.removeItem('eh-resume'); } catch { }
  if (resume && resume.n <= 5 && (resume.scan || resume.pl)) {
    status(`GPU 복구 후 탐색 재개 (${resume.n}회째)`);
    if (resume.scan) window.__eh.startScan();
    if (resume.pl) { showTab('pl'); window.__eh.startPL(); }
  } else if (!gallery.items.some(i => i.known)) runBench(); else runMap();
})();
