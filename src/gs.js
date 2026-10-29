// Gray-Scott on WebGPU. One storage buffer holds a cols×rows grid of T×T tiles,
// each tile periodic with its own (F,k) — or, in map mode, one field with F,k
// varying across x/y (xmorphia-style map, clamped edges).
import { DU, DV } from './known.js';
import { rng } from './rng.js';

const WGSL = /* wgsl */ `
struct P { W:u32, H:u32, T:u32, mode:u32, cols:u32, stencil:u32, Du:f32, Dv:f32,
           f0:f32, f1:f32, k0:f32, k1:f32 };
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage, read> src: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2f>;
@group(0) @binding(3) var<storage, read> tp: array<vec2f>;

fn at(x:i32, y:i32) -> vec2f { return src[u32(y) * p.W + u32(x)]; }

@compute @workgroup_size(16, 16)
fn step(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= p.W || g.y >= p.H) { return; }
  let x = i32(g.x); let y = i32(g.y);
  var xl:i32; var xr:i32; var yu:i32; var yd:i32; var F:f32; var k:f32;
  if (p.mode == 0u) {
    let T = i32(p.T);
    let bx = (x / T) * T; let by = (y / T) * T;
    let lx = x - bx; let ly = y - by;
    xl = bx + (lx + T - 1) % T; xr = bx + (lx + 1) % T;
    yu = by + (ly + T - 1) % T; yd = by + (ly + 1) % T;
    let q = tp[u32(y / T) * p.cols + u32(x / T)];
    F = q.x; k = q.y;
  } else {
    xl = max(x - 1, 0); xr = min(x + 1, i32(p.W) - 1);
    yu = max(y - 1, 0); yd = min(y + 1, i32(p.H) - 1);
    let fx = f32(x) / f32(p.W - 1u);
    let fy = 1.0 - f32(y) / f32(p.H - 1u);
    k = mix(p.k0, p.k1, fx);
    F = mix(p.f0, p.f1, fy);
  }
  let c = at(x, y);
  var lap = at(xl, y) + at(xr, y) + at(x, yu) + at(x, yd) - 4.0 * c;
  if (p.stencil == 1u) { // isotropic 9-point: (4·edges + corners − 20·c) / 6
    let dg = at(xl, yu) + at(xr, yu) + at(xl, yd) + at(xr, yd);
    lap = (4.0 * (lap + 4.0 * c) + dg - 20.0 * c) / 6.0;
  }
  let uvv = c.x * c.y * c.y;
  let u = c.x + p.Du * lap.x - uvv + F * (1.0 - c.x);
  let v = c.y + p.Dv * lap.y + uvv - (F + k) * c.y;
  dst[g.y * p.W + g.x] = vec2f(clamp(u, 0.0, 1.5), clamp(v, 0.0, 1.5));
}

struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  var o: VO;
  let xy = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  o.pos = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2f(xy.x, 1.0 - xy.y);
  return o;
}
fn pal(t: f32) -> vec3f {
  // deep ink → teal → warm cream (v concentration)
  let a = vec3f(0.04, 0.05, 0.10); let b = vec3f(0.05, 0.45, 0.55);
  let c = vec3f(0.98, 0.86, 0.60); let d = vec3f(1.0, 0.98, 0.94);
  let s = clamp(t, 0.0, 1.0);
  if (s < 0.35) { return mix(a, b, s / 0.35); }
  if (s < 0.75) { return mix(b, c, (s - 0.35) / 0.4); }
  return mix(c, d, (s - 0.75) / 0.25);
}
@fragment fn fs(i: VO) -> @location(0) vec4f {
  let x = min(u32(i.uv.x * f32(p.W)), p.W - 1u);
  let y = min(u32(i.uv.y * f32(p.H)), p.H - 1u);
  let c = src[y * p.W + x];
  var col = pal(c.y * 2.6);
  if (p.mode == 0u && p.T >= 32u) {
    let lx = x % p.T; let ly = y % p.T;
    if (lx == 0u || ly == 0u) { col = col * 0.35; }
  }
  return vec4f(col, 1.0);
}`;

export class GrayScott {
  constructor(gpu, { T = 128, cols = 10, rows = 10, mode = 0, W, H, canvas, stencil = 0 }) {
    this.gpu = gpu;
    this.stencil = stencil; // 0 = 5-point Laplacian (Pearson), 1 = isotropic 9-point
    const dev = gpu.device;
    this.mode = mode;
    this.T = T; this.cols = cols; this.rows = rows;
    this.W = mode === 0 ? T * cols : W;
    this.H = mode === 0 ? T * rows : H;
    this.n = this.W * this.H;
    this.steps = 0;
    this.map = { f0: 0.0, f1: 0.1, k0: 0.03, k1: 0.07 };
    const size = this.n * 8;
    const S = GPUBufferUsage;
    this.buf = [0, 1].map(() => dev.createBuffer({ size, usage: S.STORAGE | S.COPY_SRC | S.COPY_DST }));
    this.ub = dev.createBuffer({ size: 48, usage: S.UNIFORM | S.COPY_DST });
    this.tpb = dev.createBuffer({ size: Math.max(16, cols * rows * 8), usage: S.STORAGE | S.COPY_DST });
    this.read = dev.createBuffer({ size, usage: S.MAP_READ | S.COPY_DST });
    const mod = gpu.module(WGSL);
    this.cp = dev.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'step' } });
    this.bg = [0, 1].map(i => dev.createBindGroup({
      layout: this.cp.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.ub } },
        { binding: 1, resource: { buffer: this.buf[i] } },
        { binding: 2, resource: { buffer: this.buf[1 - i] } },
        { binding: 3, resource: { buffer: this.tpb } },
      ],
    }));
    this.cur = 0;
    if (canvas) this.attach(canvas);
    this.writeUniform();
  }

  attach(canvas) {
    const dev = this.gpu.device;
    canvas.width = this.W; canvas.height = this.H;
    this.ctx = canvas.getContext('webgpu');
    this.ctx.configure({ device: dev, format: this.gpu.format, alphaMode: 'opaque' });
    const mod = this.gpu.module(WGSL);
    this.rp = dev.createRenderPipeline({
      layout: 'auto',
      vertex: { module: mod, entryPoint: 'vs' },
      fragment: { module: mod, entryPoint: 'fs', targets: [{ format: this.gpu.format }] },
      primitive: { topology: 'triangle-list' },
    });
    this.rbg = [0, 1].map(i => dev.createBindGroup({
      layout: this.rp.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.ub } },
        { binding: 1, resource: { buffer: this.buf[i] } },
      ],
    }));
  }

  writeUniform() {
    const a = new ArrayBuffer(48);
    const u = new Uint32Array(a), f = new Float32Array(a);
    u[0] = this.W; u[1] = this.H; u[2] = this.T; u[3] = this.mode; u[4] = this.cols; u[5] = this.stencil;
    f[6] = DU; f[7] = DV;
    f[8] = this.map.f0; f[9] = this.map.f1; f[10] = this.map.k0; f[11] = this.map.k1;
    this.gpu.device.queue.writeBuffer(this.ub, 0, a);
  }

  setTileParams(list) { // [{F,k}] row-major
    const d = new Float32Array(this.cols * this.rows * 2);
    list.forEach((p, i) => { d[2 * i] = p.F; d[2 * i + 1] = p.k; });
    this.gpu.device.queue.writeBuffer(this.tpb, 0, d);
    this.params = list;
  }

  upload(field) {
    this.gpu.device.queue.writeBuffer(this.buf[0], 0, field);
    this.cur = 0; this.steps = 0;
  }

  run(n) {
    const dev = this.gpu.device;
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.cp);
    const gx = Math.ceil(this.W / 16), gy = Math.ceil(this.H / 16);
    for (let i = 0; i < n; i++) {
      pass.setBindGroup(0, this.bg[this.cur]);
      pass.dispatchWorkgroups(gx, gy);
      this.cur = 1 - this.cur;
    }
    pass.end();
    dev.queue.submit([enc.finish()]);
    this.steps += n;
  }

  // Draw only from rAF: when the pane isn't composited, queued presents would stall the GPU queue.
  render() {
    if (!this.ctx || this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = 0; if (!this.dead) this.draw(); });
  }

  draw() {
    const dev = this.gpu.device;
    const enc = dev.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{
      view: this.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store',
      clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    pass.setPipeline(this.rp);
    pass.setBindGroup(0, this.rbg[this.cur]);
    pass.draw(3);
    pass.end();
    dev.queue.submit([enc.finish()]);
  }

  async readback() { // returns Float32Array [u,v,...] of whole field
    const dev = this.gpu.device;
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(this.buf[this.cur], 0, this.read, 0, this.n * 8);
    dev.queue.submit([enc.finish()]);
    await this.read.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.read.getMappedRange().slice(0));
    this.read.unmap();
    return out;
  }

  // Snapshots: copy the field into slot i of one MAP_READ buffer, map it once at the end.
  // (One callback per batch — WebGPU callbacks are throttled hard in a hidden pane.)
  allocSnaps(k) {
    this.snaps?.destroy();
    this.snaps = this.gpu.device.createBuffer({ size: this.n * 8 * k, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.nSnaps = k;
  }
  snapshot(slot) {
    const enc = this.gpu.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.buf[this.cur], 0, this.snaps, slot * this.n * 8, this.n * 8);
    this.gpu.device.queue.submit([enc.finish()]);
  }
  async readSnaps() {
    await this.snaps.mapAsync(GPUMapMode.READ);
    const all = this.snaps.getMappedRange();
    const out = [];
    for (let i = 0; i < this.nSnaps; i++) out.push(new Float32Array(all.slice(i * this.n * 8, (i + 1) * this.n * 8)));
    this.snaps.unmap();
    return out;
  }

  // v-channel of tile i as Float32Array T*T
  tileV(field, i) {
    const T = this.T, tx = i % this.cols, ty = (i / this.cols) | 0;
    const out = new Float32Array(T * T);
    for (let y = 0; y < T; y++) {
      const row = ((ty * T + y) * this.W + tx * T) * 2;
      for (let x = 0; x < T; x++) out[y * T + x] = field[row + 2 * x + 1];
    }
    return out;
  }

  destroy() { this.dead = true; this.snaps?.destroy(); [...this.buf, this.ub, this.tpb, this.read].forEach(b => b.destroy()); }
}

// ---------- initial conditions ----------
export const ICS = ['square', 'noise', 'dots', 'arc', 'ring', 'line', 'inverse', 'twin', 'checker', 'gradient'];

// Stable non-trivial ("blue") homogeneous state, if it exists for (F,k).
export function blueState(F, k) {
  const a = F + k, disc = F * F - 4 * F * a * a;
  if (!(disc > 0)) return null;
  const v = (F + Math.sqrt(disc)) / (2 * a);
  return { u: 1 - (a * v) / F, v };
}

// Fill a T×T tile (u,v interleaved) into dest at tile offset.
// Seeds use Pearson's (u,v)=(1/2,1/4) unless the blue state exists above it.
export function makeIC(kind, T, seed, F = 0, k = 0) {
  const bs = blueState(F, k);
  const SU = bs && bs.v > 0.25 ? bs.u : 0.5, SV = bs && bs.v > 0.25 ? bs.v : 0.25;
  const r = rng(seed);
  const u = new Float32Array(T * T).fill(1), v = new Float32Array(T * T);
  const set = (x, y, uu, vv) => {
    x = ((x % T) + T) % T; y = ((y % T) + T) % T;
    u[y * T + x] = uu; v[y * T + x] = vv;
  };
  const disk = (cx, cy, rad, uu = SU, vv = SV) => {
    for (let y = -rad; y <= rad; y++) for (let x = -rad; x <= rad; x++)
      if (x * x + y * y <= rad * rad) set(cx + x, cy + y, uu, vv);
  };
  const c = T >> 1, s = 10; // Pearson: 20×20 grid-point square
  switch (kind) {
    case 'square': // Pearson: central square u=1/2, v=1/4
      for (let y = c - s; y < c + s; y++) for (let x = c - s; x < c + s; x++) set(x, y, 0.5, 0.25);
      break;
    case 'noise':
      for (let i = 0; i < T * T; i++) { u[i] = 0.5 + r() * 0.5; v[i] = r() * 0.3; }
      break;
    case 'dots': {
      const n = 3 + ((r() * 10) | 0);
      for (let i = 0; i < n; i++) disk((r() * T) | 0, (r() * T) | 0, 2 + ((r() * 3) | 0));
      break;
    }
    case 'arc': { // U-shaped arc (seed for U-skate-like gliders)
      const R = s * 1.6, a0 = r() * Math.PI * 2;
      for (let t = 0; t < 1; t += 0.01) {
        const a = a0 + t * Math.PI;
        disk(Math.round(c + Math.cos(a) * R), Math.round(c + Math.sin(a) * R), 2);
      }
      break;
    }
    case 'ring': {
      const R = s * 2.2;
      for (let a = 0; a < Math.PI * 2; a += 0.02) disk(Math.round(c + Math.cos(a) * R), Math.round(c + Math.sin(a) * R), 1);
      break;
    }
    case 'line': {
      const y0 = (r() * T) | 0;
      for (let x = 0; x < T * 0.7; x++) for (let w = 0; w < 3; w++) set(x + (T * 0.15 | 0), y0 + w, SU, SV);
      break;
    }
    case 'inverse': // everything perturbed, a few holes
      for (let i = 0; i < T * T; i++) { u[i] = SU; v[i] = SV; }
      for (let i = 0; i < 6; i++) disk((r() * T) | 0, (r() * T) | 0, 3 + ((r() * 4) | 0), 1, 0);
      break;
    case 'twin':
      disk(c - s * 2, c, s); disk(c + s * 2, c + ((r() * 4) | 0), s);
      break;
    case 'checker': {
      const b = 8 + ((r() * 10) | 0);
      for (let y = 0; y < T; y++) for (let x = 0; x < T; x++)
        if ((((x / b) | 0) + ((y / b) | 0)) % 2 && r() < 0.5) set(x, y, SU, SV);
      break;
    }
    case 'gradient':
      for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) {
        const g = x / T;
        u[y * T + x] = 1 - (1 - SU) * g; v[y * T + x] = SV * g * (r() < 0.6 ? 1 : 0.3);
      }
      break;
  }
  // Pearson's 1% noise
  for (let i = 0; i < T * T; i++) { u[i] += (r() - 0.5) * 0.02; v[i] = Math.max(0, v[i] + (r() - 0.5) * 0.02); }
  return { u, v };
}

export function packTiles(sim, ics) { // ics: [{u,v}] per tile
  const f = new Float32Array(sim.n * 2);
  const T = sim.T;
  ics.forEach((ic, i) => {
    const tx = i % sim.cols, ty = (i / sim.cols) | 0;
    for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) {
      const o = ((ty * T + y) * sim.W + tx * T + x) * 2;
      f[o] = ic.u[y * T + x]; f[o + 1] = ic.v[y * T + x];
    }
  });
  return f;
}

// Map-mode field: uniform with scattered perturbation squares everywhere.
export function mapIC(W, H, seed) {
  const r = rng(seed), f = new Float32Array(W * H * 2);
  for (let i = 0; i < W * H; i++) { f[2 * i] = 1; f[2 * i + 1] = 0; }
  const n = (W * H) / 900;
  for (let i = 0; i < n; i++) {
    const cx = (r() * W) | 0, cy = (r() * H) | 0, s = 3 + ((r() * 4) | 0);
    for (let y = cy - s; y < cy + s; y++) for (let x = cx - s; x < cx + s; x++) {
      if (x < 0 || y < 0 || x >= W || y >= H) continue;
      const o = (y * W + x) * 2; f[o] = 0.5; f[o + 1] = 0.25;
    }
  }
  for (let i = 0; i < W * H; i++) f[2 * i + 1] = Math.max(0, f[2 * i + 1] + (r() - 0.5) * 0.02);
  return f;
}

export function tileThumb(v, T, size = 96) {
  const cv = document.createElement('canvas');
  cv.width = cv.height = size;
  const g = cv.getContext('2d');
  const img = g.createImageData(size, size);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const t = Math.min(1, v[((y * T / size) | 0) * T + ((x * T / size) | 0)] * 2.6);
    const [R, G, B] = palJS(t);
    const o = (y * size + x) * 4;
    img.data[o] = R; img.data[o + 1] = G; img.data[o + 2] = B; img.data[o + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  return cv.toDataURL('image/png');
}
function palJS(s) {
  const a = [0.04, 0.05, 0.10], b = [0.05, 0.45, 0.55], c = [0.98, 0.86, 0.60], d = [1, 0.98, 0.94];
  const m = (p, q, t) => p.map((x, i) => (x + (q[i] - x) * t) * 255);
  if (s < 0.35) return m(a, b, s / 0.35);
  if (s < 0.75) return m(b, c, (s - 0.35) / 0.4);
  return m(c, d, (s - 0.75) / 0.25);
}
