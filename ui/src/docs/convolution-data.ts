import { P, signed } from '@/lib/palette'
// Inputs and drawing for the CNN essay. Nothing here convolves: presets are
// procedural input images, and the draw helpers only paint what Bend returns.

export const N = 28 // the input is N x N, like an MNIST digit
export type Kernel = number[] // nine weights, row-major

export const KERNELS = {
  identity: { label: 'identity', k: [0, 0, 0, 0, 1, 0, 0, 0, 0] },
  blur: { label: 'blur', k: [1, 2, 1, 2, 4, 2, 1, 2, 1].map(v => v / 16) },
  sharpen: { label: 'sharpen', k: [0, -1, 0, -1, 5, -1, 0, -1, 0] },
  sobelx: { label: 'Sobel x (vertical edges)', k: [-1, 0, 1, -2, 0, 2, -1, 0, 1] },
  sobely: { label: 'Sobel y (horizontal edges)', k: [-1, -2, -1, 0, 0, 0, 1, 2, 1] },
  outline: { label: 'outline (Laplacian)', k: [-1, -1, -1, -1, 8, -1, -1, -1, -1] },
  emboss: { label: 'emboss', k: [-2, -1, 0, -1, 1, 1, 0, 1, 2] },
} as const
export type KernelName = keyof typeof KERNELS | 'custom'
export const KERNEL_OPTIONS = (Object.keys(KERNELS) as (keyof typeof KERNELS)[]).map(v => [v, KERNELS[v].label] as const)

// Layer 1: eight hand-picked kernels (scaled so a clean edge scores about 1).
const sx = [-1, 0, 1, -2, 0, 2, -1, 0, 1], sy = [-1, -2, -1, 0, 0, 0, 1, 2, 1], d1 = [-2, -1, 0, -1, 0, 1, 0, 1, 2], d2 = [0, -1, -2, 1, 0, -1, 2, 1, 0]
const scale = (k: number[], s: number) => k.map(v => v / s)
export const LAYER1: { name: string; k: Kernel }[] = [
  { name: 'left edges', k: scale(sx, 4) }, { name: 'right edges', k: scale(sx, -4) },
  { name: 'top edges', k: scale(sy, 4) }, { name: 'bottom edges', k: scale(sy, -4) },
  { name: 'diagonal ↘', k: scale(d1, 4) }, { name: 'diagonal ↙', k: scale(d2, 4) },
  { name: 'outline', k: scale([-1, -1, -1, -1, 8, -1, -1, -1, -1], 4) }, { name: 'blur', k: scale([1, 2, 1, 2, 4, 2, 1, 2, 1], 16) },
]
// Layer 2 reads the four edge maps (left, right, top, bottom). A corner is
// where two edges meet, so each filter checks one arm of the corner on each
// map (the centre plus one step along that edge) and subtracts a bias: one
// edge alone stays below zero, both together pass the ReLU.
const zero = Array(9).fill(0)
const arm = (dx: number, dy: number) => { const k = [...zero]; k[4] = .5; k[(1 + dy) * 3 + 1 + dx] = .5; return k }
// h = +1: the top/bottom edge leaves the corner rightward; v = +1: the left/right edge leaves it downward
const corner = (h: number, v: number) => { const ks = [zero, zero, zero, zero]; ks[h > 0 ? 0 : 1] = arm(0, v); ks[v > 0 ? 2 : 3] = arm(h, 0); return ks }
export const LAYER2: { name: string; ks: Kernel[]; bias: number }[] = [
  { name: 'top-left corners', ks: corner(1, 1), bias: -1 },
  { name: 'top-right corners', ks: corner(-1, 1), bias: -1 },
  { name: 'bottom-left corners', ks: corner(1, -1), bias: -1 },
  { name: 'bottom-right corners', ks: corner(-1, -1), bias: -1 },
]

// -- procedural inputs (values in [0, 1], 1 = ink) ----------------------------

const segDist = (px: number, py: number, ax: number, ay: number, bx: number, by: number) => {
  const dx = bx - ax, dy = by - ay, t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy || 1)))
  return Math.hypot(px - ax - t * dx, py - ay - t * dy)
}
function strokes(paths: [number, number][][], width = 1.6) {
  const out = new Float32Array(N * N)
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    let d = 1e9
    for (const p of paths) for (let i = 1; i < p.length; i++) d = Math.min(d, segDist(x + .5, y + .5, ...p[i - 1], ...p[i]))
    out[y * N + x] = Math.max(0, Math.min(1, width + .5 - d))
  }
  return out
}
const arc = (cx: number, cy: number, r: number, a0: number, a1: number, n = 24) =>
  Array.from({ length: n + 1 }, (_, i) => { const a = (a0 + (a1 - a0) * i / n) * Math.PI / 180; return [cx + r * Math.cos(a), cy + r * Math.sin(a)] as [number, number] })

export const PRESETS = {
  digit: { label: 'a digit', make: () => strokes([[[8, 5], [19, 5], [12, 13]], arc(13.5, 18, 5.5, -95, 150)]) },
  box: { label: 'a box', make: () => { const o = new Float32Array(N * N); for (let y = 7; y < 21; y++) for (let x = 6; x < 22; x++) o[y * N + x] = 1; return o } },
  ring: { label: 'a ring', make: () => strokes([arc(14, 14, 8.5, 0, 360, 48)], 1.3) },
  checker: { label: 'checkerboard', make: () => Float32Array.from({ length: N * N }, (_, i) => ((Math.floor((i % N) / 7) + Math.floor(i / N / 7)) % 2)) },
  gradient: { label: 'gradient', make: () => Float32Array.from({ length: N * N }, (_, i) => ((i % N) + Math.floor(i / N)) / (2 * N - 2)) },
  blank: { label: 'blank (draw!)', make: () => new Float32Array(N * N) },
} as const
export type PresetName = keyof typeof PRESETS
export const PRESET_OPTIONS = (Object.keys(PRESETS) as PresetName[]).map(v => [v, PRESETS[v].label] as const)

// A soft round brush (radius in pixels). Erasing pulls values toward zero.
export function brush(img: Float32Array, cx: number, cy: number, erase: boolean, r = 1.4) {
  const out = img.slice()
  for (let y = Math.floor(cy - r - 1); y <= cy + r + 1; y++) for (let x = Math.floor(cx - r - 1); x <= cx + r + 1; x++) {
    if (x < 0 || y < 0 || x >= N || y >= N) continue
    const a = Math.max(0, Math.min(1, r + .5 - Math.hypot(x + .5 - cx, y + .5 - cy))), i = y * N + x
    out[i] = erase ? out[i] * (1 - a) : Math.max(out[i], a)
  }
  return out
}

// -- painting -------------------------------------------------------------------

const hex = (h: string) => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16))
const PAPER = hex(P.paper), INK = hex(P.ink4)
const RAMP = Array.from({ length: 65 }, (_, i) => { const t = i / 64; return `rgb(${PAPER.map((p, j) => Math.round(p + (INK[j] - p) * t)).join(',')})` })
export const gray = (v: number) => RAMP[Math.round(Math.max(0, Math.min(1, v)) * 64)]

export interface Box { x: number; y: number; s: number } // top-left and side, in CSS px
// Paints an n x n map. `mode` gray maps [0, 1] to paper..ink; signed maps
// [-1, 1] through the red/green diverging ramp. Values are divided by `range`.
export function paintMap(c: CanvasRenderingContext2D, data: ArrayLike<number>, n: number, box: Box, mode: 'gray' | 'signed', range = 1, grid = false) {
  const p = box.s / n
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const v = data[y * n + x] / range
    c.fillStyle = mode === 'gray' ? gray(v) : signed(v)
    // snap edges to device pixels so neighbouring cells do not leave seams
    const x0 = Math.round((box.x + x * p) * 2) / 2, y0 = Math.round((box.y + y * p) * 2) / 2
    c.fillRect(x0, y0, Math.round((box.x + (x + 1) * p) * 2) / 2 - x0, Math.round((box.y + (y + 1) * p) * 2) / 2 - y0)
  }
  if (grid && p >= 5) {
    c.strokeStyle = 'rgba(105,102,95,.12)'; c.lineWidth = 1; c.beginPath()
    for (let i = 1; i < n; i++) { const o = Math.round(box.x + i * p) + .5, q = Math.round(box.y + i * p) + .5; c.moveTo(o, box.y); c.lineTo(o, box.y + box.s); c.moveTo(box.x, q); c.lineTo(box.x + box.s, q) }
    c.stroke()
  }
  c.strokeStyle = P.bar; c.lineWidth = 1; c.strokeRect(Math.round(box.x) + .5, Math.round(box.y) + .5, Math.round(box.s) - 1, Math.round(box.s) - 1)
}
// Outlines cells [x, x+w) x [y, y+h) of an n x n map drawn in `box`.
export function outline(c: CanvasRenderingContext2D, box: Box, n: number, x: number, y: number, w: number, h: number, color: string = P.vio, width = 2) {
  const p = box.s / n
  c.strokeStyle = color; c.lineWidth = width
  c.strokeRect(box.x + x * p, box.y + y * p, w * p, h * p)
}
export const maxAbs = (a: ArrayLike<number>) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i])); return m || 1 }
export const fmt = (v: number, d = 2) => { const r = Number(v.toFixed(d)); return (r === 0 ? 0 : r).toFixed(d) }
