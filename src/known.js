// Known Gray-Scott regions used as the "known map" baseline.
// Coordinates: Pearson 1993 classes and Munafo's added classes, as listed on
// https://mrob.com/pub/comp/xmorphia/pearson-classes.html (fetched 2026-10-20),
// U-skate world from https://mrob.com/pub/comp/xmorphia/uskate-world.html.
// `expect` is the set of our auto-labels we consider a match for that class.

export const PEARSON = [
  { id: 'R', F: 0.014, k: 0.057, desc: 'uniform red (u≈1)', expect: ['uniform'] },
  { id: 'B', F: 0.050, k: 0.059, desc: 'uniform blue after brief pattern', expect: ['uniform', 'spots'] },
  { id: 'α', F: 0.010, k: 0.047, desc: 'chaos of wavelets / fledgling spirals', expect: ['chaos'] },
  { id: 'β', F: 0.014, k: 0.039, desc: 'chaos of shifting red/blue regions', expect: ['chaos'] },
  { id: 'γ', F: 0.022, k: 0.051, desc: 'worm-like stripes, ongoing instability', expect: ['stripes', 'chaos'] },
  { id: 'δ', F: 0.030, k: 0.055, desc: 'hexagonal negative spots (Turing)', expect: ['spots', 'stripes'] },
  { id: 'ε', F: 0.018, k: 0.055, desc: 'unstable spots that split, crowd, die', expect: ['self-replicating', 'chaos'] },
  { id: 'ζ', F: 0.022, k: 0.061, desc: 'like ε, stabler spots, subtle oscillation', ic: 'line', /* square seed dies here in our runs; a line seed shows the spots */ expect: ['self-replicating', 'spots', 'chaos'] },
  { id: 'η', F: 0.034, k: 0.063, desc: 'spots + worms settling to steady state', expect: ['stripes', 'spots', 'self-replicating'] },
  { id: 'θ', F: 0.030, k: 0.057, desc: 'spots → rings → connected stripes', expect: ['stripes'] },
  { id: 'ι', F: 0.046, k: 0.0594, desc: 'negative spots interacting like molecules', expect: ['spots', 'stripes'] },
  { id: 'κ', F: 0.050, k: 0.063, desc: 'meandering stripes / mazes, coral', expect: ['stripes'] },
  { id: 'λ', F: 0.026, k: 0.061, desc: 'solitons dividing by mitosis → hex grid', expect: ['self-replicating'] },
  { id: 'μ', F: 0.046, k: 0.065, desc: 'worms growing from ends → parallel stripes', expect: ['stripes'] },
  { id: 'ν', F: 0.054, k: 0.067, desc: 'inert solitons from shrinking blue areas (Munafo)', ic: 'inverse', expect: ['spots', 'uniform', 'moving solitons'] },
  { id: 'ξ', F: 0.014, k: 0.047, desc: 'sustained BZ-like spirals (Munafo)', expect: ['chaos'] },
  { id: 'π', F: 0.062, k: 0.0609, desc: 'U-skate world: stationary + moving solitons (Munafo)', expect: ['moving solitons', 'spots', 'stripes'] },
  { id: 'ρ', F: 0.090, k: 0.059, desc: 'mostly-blue + negatons → red soap bubbles (Munafo)', ic: 'inverse', /* bubbles may still be growing when the run ends */ expect: ['stripes', 'expanding'] },
  { id: 'σ', F: 0.110, k: 0.0523, desc: 'mostly-red + few spots → blue bubbles (Munafo)', ic: 'dots', /* bubbles may still be growing when the run ends */ expect: ['stripes', 'expanding'] },
];

export const SOURCES = [
  ['Pearson 1993, "Complex Patterns in a Simple System", Science 261:189', 'https://doi.org/10.1126/science.261.5118.189'],
  ['Munafo, Pearson classes (xmorphia)', 'https://mrob.com/pub/comp/xmorphia/pearson-classes.html'],
  ['Munafo, xmorphia F,k map', 'https://mrob.com/pub/comp/xmorphia/index.html'],
  ['Munafo, U-skate world', 'https://mrob.com/pub/comp/xmorphia/uskate-world.html'],
  ['Mohr, Particle Life (force model)', 'https://github.com/tom-mohr/particle-life-app'],
];

// Pearson's constants: Du=2e-5, Dv=1e-5 on a 2.5×2.5 domain at 256² (dx=2.5/256), dt=1.
// Rescaled to dx=1: D/dx² .
export const DU = 2e-5 / (2.5 / 256) ** 2; // ≈0.2097
export const DV = 1e-5 / (2.5 / 256) ** 2; // ≈0.1049
