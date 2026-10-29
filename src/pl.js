// Particle Life on WebGPU: B independent universes × N particles on a unit torus.
// Force model after Tom Mohr's particle-life: repulsion below beta, then a
// triangular attraction/repulsion bump scaled by matrix a[si][sj].
import { rng } from './rng.js';

export const MAXM = 8;
const WG = 128;

const WGSL = /* wgsl */ `
struct P { N:u32, B:u32, gcols:u32, pad:u32, rmax:f32, beta:f32, fric:f32, dt:f32, force:f32, psize:f32, pad2:f32, pad3:f32 };
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage, read> pin: array<vec4f>;      // pos.xy vel.zw
@group(0) @binding(2) var<storage, read_write> pout: array<vec4f>;
@group(0) @binding(3) var<storage, read> sp: array<u32>;
@group(0) @binding(4) var<storage, read> mat: array<f32>;        // B*64

var<workgroup> tile: array<vec4f, ${WG}>;
var<workgroup> tsp: array<u32, ${WG}>;

fn wrap(d: vec2f) -> vec2f { return d - round(d); }
fn fr(r: f32, a: f32) -> f32 {
  if (r < p.beta) { return r / p.beta - 1.0; }
  if (r < 1.0) { return a * (1.0 - abs(2.0 * r - 1.0 - p.beta) / (1.0 - p.beta)); }
  return 0.0;
}

@compute @workgroup_size(${WG})
fn step(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) w: vec3u) {
  let i = g.x;                         // N is a multiple of WG so a workgroup stays in one universe
  let b = i / p.N;
  let me = pin[i];
  let si = sp[i];
  var acc = vec2f(0.0);
  let base = b * p.N;
  for (var t = 0u; t < p.N; t += ${WG}u) {
    tile[l.x] = pin[base + t + l.x];
    tsp[l.x] = sp[base + t + l.x];
    workgroupBarrier();
    for (var j = 0u; j < ${WG}u; j++) {
      let d = wrap(tile[j].xy - me.xy);
      let r = length(d);
      if (r > 0.0 && r < p.rmax) {
        acc += d / r * fr(r / p.rmax, mat[b * 64u + si * 8u + tsp[j]]);
      }
    }
    workgroupBarrier();
  }
  var v = me.zw * p.fric + acc * p.rmax * p.force * p.dt;
  var x = fract(me.xy + v * p.dt + 1.0);
  pout[i] = vec4f(x, v);
}

struct VO { @builtin(position) pos: vec4f, @location(0) c: vec3f, @location(1) q: vec2f };
var<private> COLS: array<vec3f, 8> = array<vec3f, 8>(vec3f(0.98,0.45,0.42), vec3f(0.36,0.78,0.85), vec3f(0.98,0.82,0.40), vec3f(0.62,0.52,0.95),
  vec3f(0.45,0.86,0.52), vec3f(0.97,0.55,0.80), vec3f(0.90,0.92,0.95), vec3f(0.98,0.62,0.25));
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  let b = ii / p.N;
  let cell = vec2f(f32(b % p.gcols), f32(b / p.gcols));
  let rows = f32((p.B + p.gcols - 1u) / p.gcols);
  let q = vec2f(f32(vi & 1u), f32((vi >> 1u) & 1u)) * 2.0 - 1.0;
  let x = pin[ii].xy;
  let px = (cell + x) / vec2f(f32(p.gcols), rows);
  let s = p.psize / vec2f(f32(p.gcols), rows);
  var o: VO;
  o.pos = vec4f((px.x * 2.0 - 1.0) + q.x * s.x, 1.0 - px.y * 2.0 + q.y * s.y, 0.0, 1.0);
  o.c = COLS[sp[ii] % 8u];
  o.q = q;
  return o;
}
@fragment fn fs(i: VO) -> @location(0) vec4f {
  let d = dot(i.q, i.q);
  if (d > 1.0) { discard; }
  let a = (1.0 - d) * 0.9;
  return vec4f(i.c * a, a);
}`;

export class ParticleLife {
  constructor(gpu, { N = 1024, B = 16, gcols = 4, canvas }) {
    this.gpu = gpu; this.N = N; this.B = B; this.gcols = gcols;
    this.cfg = { rmax: 0.1, beta: 0.3, halfLife: 0.04, dt: 0.02, force: 10, psize: 0.014 };
    const dev = gpu.device, S = GPUBufferUsage;
    const tot = N * B;
    this.pb = [0, 1].map(() => dev.createBuffer({ size: tot * 16, usage: S.STORAGE | S.COPY_SRC | S.COPY_DST }));
    this.spb = dev.createBuffer({ size: tot * 4, usage: S.STORAGE | S.COPY_DST });
    this.mb = dev.createBuffer({ size: B * 64 * 4, usage: S.STORAGE | S.COPY_DST });
    this.ub = dev.createBuffer({ size: 48, usage: S.UNIFORM | S.COPY_DST });
    this.read = dev.createBuffer({ size: tot * 16, usage: S.MAP_READ | S.COPY_DST });
    const mod = gpu.module(WGSL);
    this.cp = dev.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'step' } });
    this.bg = [0, 1].map(i => dev.createBindGroup({ layout: this.cp.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.ub } }, { binding: 1, resource: { buffer: this.pb[i] } },
      { binding: 2, resource: { buffer: this.pb[1 - i] } }, { binding: 3, resource: { buffer: this.spb } },
      { binding: 4, resource: { buffer: this.mb } }] }));
    this.cur = 0; this.steps = 0;
    this.species = new Uint32Array(tot);
    if (canvas) this.attach(canvas);
    this.writeUniform();
  }

  attach(canvas) {
    const dev = this.gpu.device;
    this.ctx = canvas.getContext('webgpu');
    this.ctx.configure({ device: dev, format: this.gpu.format, alphaMode: 'opaque' });
    const mod = this.gpu.module(WGSL);
    this.rp = dev.createRenderPipeline({ layout: 'auto',
      vertex: { module: mod, entryPoint: 'vs' },
      fragment: { module: mod, entryPoint: 'fs', targets: [{ format: this.gpu.format,
        blend: { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } } }] },
      primitive: { topology: 'triangle-strip' } });
    this.rbg = [0, 1].map(i => dev.createBindGroup({ layout: this.rp.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.ub } }, { binding: 1, resource: { buffer: this.pb[i] } },
      { binding: 3, resource: { buffer: this.spb } }] }));
  }

  writeUniform() {
    const c = this.cfg, a = new ArrayBuffer(48), u = new Uint32Array(a), f = new Float32Array(a);
    u[0] = this.N; u[1] = this.B; u[2] = this.gcols;
    f[4] = c.rmax; f[5] = c.beta; f[6] = Math.pow(0.5, c.dt / c.halfLife); f[7] = c.dt; f[8] = c.force; f[9] = c.psize;
    this.gpu.device.queue.writeBuffer(this.ub, 0, a);
  }

  // genomes: [{m, mat: Float32Array(m*m), seed}]
  load(genomes) {
    this.genomes = genomes;
    const N = this.N, tot = N * this.B;
    const pos = new Float32Array(tot * 4), mats = new Float32Array(this.B * 64);
    genomes.forEach((g, b) => {
      const r = rng(g.seed);
      for (let i = 0; i < N; i++) {
        const o = (b * N + i);
        pos[o * 4] = r(); pos[o * 4 + 1] = r();
        this.species[o] = Math.min(g.m - 1, (i * g.m / N) | 0);
      }
      for (let i = 0; i < g.m; i++) for (let j = 0; j < g.m; j++) mats[b * 64 + i * 8 + j] = g.mat[i * g.m + j];
    });
    const q = this.gpu.device.queue;
    q.writeBuffer(this.pb[0], 0, pos); q.writeBuffer(this.spb, 0, this.species); q.writeBuffer(this.mb, 0, mats);
    this.cur = 0; this.steps = 0;
  }

  run(n) {
    const dev = this.gpu.device, enc = dev.createCommandEncoder(), pass = enc.beginComputePass();
    pass.setPipeline(this.cp);
    const groups = (this.N * this.B) / WG;
    for (let i = 0; i < n; i++) { pass.setBindGroup(0, this.bg[this.cur]); pass.dispatchWorkgroups(groups); this.cur = 1 - this.cur; }
    pass.end(); dev.queue.submit([enc.finish()]); this.steps += n;
  }

  // Draw only from rAF: when the pane isn't composited, queued presents would stall the GPU queue.
  render() {
    if (!this.ctx || this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = 0; if (!this.dead) this.draw(); });
  }

  draw() {
    const dev = this.gpu.device, enc = dev.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(),
      loadOp: 'clear', storeOp: 'store', clearValue: { r: 0.02, g: 0.025, b: 0.04, a: 1 } }] });
    pass.setPipeline(this.rp); pass.setBindGroup(0, this.rbg[this.cur]);
    pass.draw(4, this.N * this.B); pass.end(); dev.queue.submit([enc.finish()]);
  }

  allocSnaps(k) {
    this.snaps?.destroy();
    this.snaps = this.gpu.device.createBuffer({ size: this.N * this.B * 16 * k, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.nSnaps = k;
  }
  snapshot(slot) {
    const sz = this.N * this.B * 16, enc = this.gpu.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.pb[this.cur], 0, this.snaps, slot * sz, sz);
    this.gpu.device.queue.submit([enc.finish()]);
  }
  async readSnaps() {
    const sz = this.N * this.B * 16;
    await this.snaps.mapAsync(GPUMapMode.READ);
    const all = this.snaps.getMappedRange(), out = [];
    for (let i = 0; i < this.nSnaps; i++) out.push(new Float32Array(all.slice(i * sz, (i + 1) * sz)));
    this.snaps.unmap();
    return out;
  }

  async readback() {
    const dev = this.gpu.device, enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(this.pb[this.cur], 0, this.read, 0, this.N * this.B * 16);
    dev.queue.submit([enc.finish()]);
    await this.read.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.read.getMappedRange().slice(0));
    this.read.unmap();
    return out;
  }

  destroy() { this.dead = true; this.snaps?.destroy(); [...this.pb, this.spb, this.mb, this.ub, this.read].forEach(b => b.destroy()); }
}

export function randomGenome(seed) {
  const r = rng(seed);
  const m = 3 + ((r() * 6) | 0); // 3..8 species
  const mat = new Float32Array(m * m);
  const style = r();
  for (let i = 0; i < m * m; i++) {
    let a = r() * 2 - 1;
    if (style < 0.3) a = Math.sign(a) * Math.pow(Math.abs(a), 0.5); // stronger
    mat[i] = a;
  }
  if (style > 0.8) for (let i = 0; i < m; i++) { // chain / snake motifs: self-attract + next-attract
    mat[i * m + i] = 0.6 + r() * 0.4; mat[i * m + ((i + 1) % m)] = 0.2 + r() * 0.5;
  }
  return { m, mat: Array.from(mat), seed: (r() * 2 ** 32) >>> 0 };
}

export function mutateGenome(g, seed, amt = 0.25) {
  const r = rng(seed);
  return { m: g.m, mat: g.mat.map(a => Math.max(-1, Math.min(1, a + (r() * 2 - 1) * amt))), seed: (r() * 2 ** 32) >>> 0 };
}

const PCOL = ['#fa736b', '#5cc7d9', '#fad166', '#9e85f2', '#73db85', '#f78ccc', '#e6ebf2', '#fa9e40'];
export function plThumb(data, b, N, species, size = 96) {
  const cv = document.createElement('canvas'); cv.width = cv.height = size;
  const g = cv.getContext('2d');
  g.fillStyle = '#05060a'; g.fillRect(0, 0, size, size);
  for (let i = 0; i < N; i++) {
    const o = (b * N + i) * 4;
    g.fillStyle = PCOL[species[b * N + i] % 8];
    g.fillRect(data[o] * size - 0.6, data[o + 1] * size - 0.6, 1.3, 1.3);
  }
  return cv.toDataURL('image/png');
}
