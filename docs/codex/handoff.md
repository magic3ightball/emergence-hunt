# Emergence Hunt — handoff

## Objective
Browser explorer for finding *new-looking* emergent patterns in two systems with no central census,
judging "new" by feature-space distance to a known baseline and to previously saved finds.

1. **Gray-Scott** reaction-diffusion (WebGPU compute): xmorphia-style F,k map, Pearson-class
   reproduction test, automated fine scan with unusual initial conditions, auto-classification
   into uniform / spots / stripes / self-replicating / moving solitons / chaos.
2. **Particle Life** (WebGPU, 16 universes × 1024 particles, Mohr force model): random + mutation
   search over attraction matrices, scored on clusters, cluster motion, rotation, membranes.
3. **Shared novelty gallery** (IndexedDB): feature vectors, novelty = mean distance to the 3 nearest
   items of the same mode (known refs + saved finds), keeps the 40 most novel per mode, thumbnails,
   click to replay from (params, seed).

## Run
- Static site, no build. Server entry `emergence-hunt` in `Claude_etc/.claude/launch.json`, port **8980**
  (`python3 serve.py 8980` in this folder — sends `Cache-Control: no-store`; plain `http.server`
  let the browser keep running stale ES modules after edits).
- `npm test` (= `node tests/run.mjs`): CPU analysis checks on synthetic fields/particles (9 checks).
- Debug hooks: `window.__eh` → `runBench()`, `runMap()`, `startScan()/stopScan()`, `startPL()/stopPL()`,
  `runBatch(items,{cols,rows,T,steps,job})` (get `job` from `newJob()`), `bench`, `lastScan`, `lastPL`, `gallery`.

## Deployment
- Vercel project `emergence-hunt` (scope magic3ightballs-projects): https://emergence-hunt.vercel.app
- Deploy from this folder: `npx vercel@latest deploy --prod --yes --scope magic3ightballs-projects`
  (`.vercelignore` excludes docs, tests, serve.py, .env*; delete any `.env.local` that vercel creates).
- Listed in the deployed-links hub (Claude_etc/deployed-links). The user wants every new build like this
  deployed and added there.

## Files
- `src/gs.js` — Gray-Scott sim (tiled periodic batch or gradient map), ICs, thumbnails.
- `src/gsclassify.js` — periodic connected components, features (13-d), label rules.
- `src/known.js` — Pearson/Munafo class coordinates + sources, Pearson constants (Du=0.2097, Dv=0.1049 at dx=1, dt=1).
- `src/pl.js` — Particle Life sim (workgroup-tiled O(N²) per universe), genomes, thumbnails.
- `src/plscore.js` — clustering (grid hash, link 0.025), motion/rotation/membrane/purity, 10-d features.
- `index.html` — UI plus the "작동 원리" tab (static explainer: inline SVG diagrams + cards for GS equations/scan/labels,
  PL force curve/labels, novelty rule). Keep it in sync when thresholds or features change.
- `src/gallery.js` — novelty archive + IndexedDB.  `src/main.js` — UI and loops.

## Verified (2026-10-20, Browser pane, mostly hidden/uncomposited)
- Pearson reproduction at 256² tiles (= Pearson's domain), 10 000 steps: **18/19** labels match the
  expected class (table in the GS tab, also stored as dashed "known" gallery refs). Miss: σ
  (F=0.110,k=0.0523) stays as a few static spots at 20k steps instead of bubbles. ζ dies from Pearson's
  square seed in our runs, so its benchmark uses a line seed (documented in `known.js`); ρ/ν/σ use the
  starting states Munafo describes (blue state with holes / few spots).
- F,k map mode (900×1080, 20k steps) shows the crescent with class markers; note that in a gradient map
  the blue uniform state invades neighbouring parameters, so the upper band (π, ρ, σ) is covered —
  use the tile benchmark, not the map, as the reproduction evidence.
- Auto scan: R2 low-discrepancy samples in a band k = √F/2 − F + [−0.0065, +0.0055] plus 50 % adaptive
  refinement around earlier non-uniform hits; ~40 % of tiles non-uniform. After 5 batches the GS
  archive held spots 12, stripes 12, chaos 12, self-replicating 3, moving solitons 1.
- Particle Life: batches of 16 run unattended (48 batches in one stretch); archive filled with
  membranes/cells, movers, rotators, clusters. Replay from gallery verified (genome + seed reproduce).
- Speed (pane visible): ~800–2800 GS steps/s for 0.3–1.6 M cells; PL batch of 16×1024 × 1225 steps ≈ 1–3 s.
- The scan's only "moving solitons" find (F 0.0245, k 0.0511, checker seed) was wandering holes in an
  active field (activeFrac 0.349), not a glider; rule tightened to activeFrac < 0.2 and the item removed.
  No genuine glider (U-skate) has been auto-found yet — they need specific seeds near F 0.062, k 0.0609.
- GPU device loss happened once after long background running ("valid external Instance reference no
  longer exists"); the page now reloads and resumes the running search (max 5 times per tab session).

## Classifier review (2026-10-20, later)
- Ran 1,200 scan tiles (R2 band, random seeds, 128², 10k steps) with `runBatch(..., {filmstrip:true})`, which returns
  thumbnails at every checkpoint, and reviewed self-replicating/solitons/spots/chaos/stripes by eye.
- Mislabels found and fixed in `gsclassify.js` (rule order: uniform → moving solitons → expanding → self-replicating → chaos → spots → stripes):
  - Spot division fronts still sweeping the tile were "chaos". Self-replicating is now tested before chaos when
    turnover < 0.2 and activeFrac < 0.35. Fields that keep splitting and dying everywhere (ε-like, activeFrac 0.4–0.6) stay chaos.
  - Slow division (1→2→4) was "spots". It now also counts as self-replicating when the count never drops across checkpoints and at least doubles (≥4).
  - The only "moving soliton" was holes splitting and regrouping (turnover 0.08). Gliders now need turnover < 0.03.
  - New label **expanding**: ≤4 bright blobs that start small (<35 % area) and keep swelling (bubbles, mostly F 0.08–0.11).
    Line-shaped growth (worms) stays "stripes". A pattern that took over the tile, leaving many holes (ι), is not expanding.
- Result on the same 1,200 tiles: 17 labels changed, and all 17 look right by eye.
  Counts: uniform 875, stripes 168, chaos 86, spots 42, self-replicating 19, expanding 10, moving solitons 0.
- Pearson bench is now 19/19, **but** σ and ρ now also accept "expanding" (bubbles still growing at 10k steps, as Munafo
  describes). So σ's earlier miss is resolved partly by widening its expectation.
- Feature vectors (13-d) are unchanged, so existing gallery items stay comparable. Old items keep their old labels.
- Tests: 14 synthetic checks (`npm test`), including growing blob, slow division, splitting-not-glider, shrinking hole, and takeover-with-holes.

## U-skate attempt (2026-10-27)
- Munafo ([arXiv:1501.01990](https://arxiv.org/abs/1501.01990)): the U-skate is a U of high u in the **blue state** (u≈0.42, v≈0.29),
  stable only for 0.06088 ≤ k ≤ 0.06098 at F=0.06, and it moves ~1 lu per 62,000 steps. At Pearson's grid that is ~0.33 cells
  per 200 steps. The page gives no seeding recipe; the paper says random rectangles on the blue state often produce Us.
- Planted ~1,500 seeds: U arcs (R 3–13, several thicknesses and spans), half disks, three-spot clusters, and random rectangles at 128² and 256² for up to 100k steps.
  Used F 0.06/0.062, k from 0.0608 to 0.0616 (down to 1e-6 steps near 0.0611), and both the 5-point and an isotropic 9-point Laplacian.
  **No U-skate formed.** In our discretization a U either shrinks to a single negaton (k ≤ 0.06110) or keeps lengthening
  into a hairpin (k ≥ 0.06111). The paper's "decays to a spot" edge sits ~0.00023 lower than ours, and the stable window between collapse and growth is missing.
  Random rectangles settle into static negatons. A three-spot cluster (spacing 10, radius 3) is stable and static.
- Classifier fix anyway: gliders like the U-skate were invisible to it (not round, and too slow for the last-200-step view).
  Added `slowDrift`, which measures blob shift over the last two checkpoint intervals. The blobs must keep count and area (±8 %, not tile-sized),
  and both intervals must show a steady rate ≥0.08 cells/200 steps. Tests: synthetic slow U → moving solitons; static,
  settling, hairpin, and slowly lengthening arcs → not. On 600 real scan tiles the first version gave 2 false hits (a slowly
  lengthening arc and a labyrinth). The ±8 % area rule and the tile-sized-blob exclusion removed them, and no other label changed. Pearson 19/19.
- Tools: `tests/uskate-lab.js` (browser-only: `negField`, `arc`, `spots`, `rects`, `sheet`). `runBatch` accepts `icData` (custom {u,v})
  and `stencil: 1` (9-point Laplacian, experiment only; the app uses 5-point).

## Gotchas
- **Hidden Browser pane**: `visibilityState` stays "visible" but rAF never fires and every WebGPU
  callback (`onSubmittedWorkDone`, `mapAsync`) is throttled to seconds/minutes. Hence: canvas draws only
  from rAF; each batch copies all checkpoint snapshots into one MAP_READ buffer and maps once; loops
  stop awaiting the GPU once a round trip takes >400 ms. Don't poll with `setTimeout` from the console;
  await `__eh.gpu.device.queue.onSubmittedWorkDone()` in a bounded loop instead.
- Archive eviction caps any label at 30 % once the archive is ≥75 % full (chaos snapshots always look
  "novel" from noise alone and otherwise took 26/40 slots).
- A find close to a Pearson reference (e.g. λ-like self-replication) is rejected as not new — by design.
- Feature vector length changes make old items incomparable (distance = ∞); clear finds after schema changes.
- Scan tiles are 128² (half Pearson's domain) for throughput; spirals (ξ) and some soliton behaviors
  need larger domains — benchmark uses 256².

## Next steps
- U-skate still not reproduced (see above). Next ideas: double precision / smaller dt (CPU or f64 emulation) to test whether
  the missing window is numerical, or start from Munafo's published pattern images. Review stripes vs. growing worms.
- Optional scan tile-size control (128/256), longer runs for F>0.05 (slow dynamics, σ).
- Particle Life: add parameter dimensions (beta, rmax, N) and a "known motifs" baseline.
