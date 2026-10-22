import { P, mono } from '@/lib/palette'
// Canvas drawing for the neural-networks essay. Nothing here computes the
// network: fields and weights arrive from Bend; this file only paints them.

export interface Box { x: number; y: number; s: number }
export type Ctx = CanvasRenderingContext2D
const PAPER = [242, 238, 231], RED = [196, 106, 96], GRN = [126, 154, 94]
export const toX = (b: Box, x: number) => b.x + (x + 1) / 2 * b.s
export const toY = (b: Box, y: number) => b.y + (1 - y) / 2 * b.s

// A field of signed values in [-1, 1] (n x n, top row first) as a smooth heatmap.
const scratch = typeof document !== 'undefined' ? document.createElement('canvas') : null
export function heat(c: Ctx, f: Float32Array, n: number, b: Box, strength = 0.6) {
  if (!scratch) return
  scratch.width = n; scratch.height = n
  const sc = scratch.getContext('2d')!, img = sc.createImageData(n, n)
  for (let i = 0; i < n * n; i++) {
    const v = Math.max(-1, Math.min(1, f[i])), k = Math.abs(v) * strength, to = v < 0 ? RED : GRN
    for (let ch = 0; ch < 3; ch++) img.data[i * 4 + ch] = Math.round(PAPER[ch] + (to[ch] - PAPER[ch]) * k)
    img.data[i * 4 + 3] = 255
  }
  sc.putImageData(img, 0, 0)
  c.save(); c.imageSmoothingEnabled = true; c.imageSmoothingQuality = 'high'
  c.drawImage(scratch, b.x, b.y, b.s, b.s); c.restore()
}

// The zero level set of a field, by marching squares over cell centres.
export function contour(c: Ctx, f: Float32Array, n: number, b: Box, color: string = P.ink4, width = 1.6) {
  const step = b.s / n, px = (i: number) => b.x + (i + 0.5) * step, py = (j: number) => b.y + (j + 0.5) * step
  c.beginPath()
  for (let j = 0; j < n - 1; j++) for (let i = 0; i < n - 1; i++) {
    const a = f[j * n + i], bb = f[j * n + i + 1], cc = f[(j + 1) * n + i + 1], d = f[(j + 1) * n + i]
    const pts: [number, number][] = []
    const edge = (v0: number, v1: number, x0: number, y0: number, x1: number, y1: number) => { if ((v0 < 0) !== (v1 < 0)) { const t = v0 / (v0 - v1); pts.push([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t]) } }
    edge(a, bb, px(i), py(j), px(i + 1), py(j)); edge(bb, cc, px(i + 1), py(j), px(i + 1), py(j + 1))
    edge(cc, d, px(i + 1), py(j + 1), px(i), py(j + 1)); edge(d, a, px(i), py(j + 1), px(i), py(j))
    for (let k = 0; k + 1 < pts.length; k += 2) { c.moveTo(pts[k][0], pts[k][1]); c.lineTo(pts[k + 1][0], pts[k + 1][1]) }
  }
  c.strokeStyle = color; c.lineWidth = width; c.lineCap = 'round'; c.stroke()
}

// Labelled points: (x, y, t) triples, t = 1 green, t = 0 red.
export function dots(c: Ctx, pts: Float32Array, b: Box, r = Math.max(2.2, b.s / 110)) {
  for (let i = 0; i < pts.length; i += 3) {
    c.beginPath(); c.arc(toX(b, pts[i]), toY(b, pts[i + 1]), r, 0, Math.PI * 2)
    c.fillStyle = pts[i + 2] > 0.5 ? P.grn : P.red; c.fill()
    c.lineWidth = 1; c.strokeStyle = 'rgba(77,74,68,.7)'; c.stroke() // ink-4, softened
  }
}

export function frame(c: Ctx, b: Box) {
  c.save(); c.strokeStyle = P.paper3; c.lineWidth = 1; c.setLineDash([2, 3])
  c.beginPath(); c.moveTo(b.x, toY(b, 0)); c.lineTo(b.x + b.s, toY(b, 0)); c.moveTo(toX(b, 0), b.y); c.lineTo(toX(b, 0), b.y + b.s); c.stroke()
  c.setLineDash([]); c.strokeStyle = P.bar; c.strokeRect(b.x + 0.5, b.y + 0.5, b.s - 1, b.s - 1); c.restore()
}

// Clip the line w1 x + w2 y + b = 0 to the square [-1, 1]^2.
export function lineEnds(w1: number, w2: number, b: number): [number, number, number, number] | null {
  const hits: [number, number][] = []
  if (Math.abs(w2) > 1e-9) for (const x of [-1, 1]) { const y = -(w1 * x + b) / w2; if (y >= -1 && y <= 1) hits.push([x, y]) }
  if (Math.abs(w1) > 1e-9) for (const y of [-1, 1]) { const x = -(w2 * y + b) / w1; if (x >= -1 && x <= 1) hits.push([x, y]) }
  if (hits.length < 2) return null
  let best: [number, number, number, number] = [...hits[0], ...hits[1]], far = -1
  for (const p of hits) for (const q of hits) { const d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2; if (d > far) { far = d; best = [p[0], p[1], q[0], q[1]] } }
  return best
}

export function text(c: Ctx, s: string, x: number, y: number, px: number, color: string = P.ink3, align: CanvasTextAlign = 'center', weight = 400) {
  c.font = mono(px, weight); c.fillStyle = color; c.textAlign = align; c.textBaseline = 'middle'; c.fillText(s, x, y)
}

export function arrow(c: Ctx, x0: number, y0: number, x1: number, y1: number, color: string, width = 1.5) {
  const a = Math.atan2(y1 - y0, x1 - x0), h = 4 + width * 2
  c.strokeStyle = color; c.fillStyle = color; c.lineWidth = width
  c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1 - Math.cos(a) * h * 0.6, y1 - Math.sin(a) * h * 0.6); c.stroke()
  c.beginPath(); c.moveTo(x1, y1); c.lineTo(x1 - h * Math.cos(a - 0.45), y1 - h * Math.sin(a - 0.45)); c.lineTo(x1 - h * Math.cos(a + 0.45), y1 - h * Math.sin(a + 0.45)); c.closePath(); c.fill()
}

// A weighted edge: green positive, red negative, thicker when larger.
export function edge(c: Ctx, x0: number, y0: number, x1: number, y1: number, w: number, scale = 1) {
  c.strokeStyle = w >= 0 ? P.grn : P.red; c.globalAlpha = Math.min(1, 0.25 + Math.abs(w) * 0.25)
  c.lineWidth = Math.min(5, 0.6 + Math.abs(w) * scale); c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1); c.stroke(); c.globalAlpha = 1
}

export function node(c: Ctx, x: number, y: number, r: number, label?: string, fill: string = P.paper) {
  c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.fillStyle = fill; c.fill(); c.lineWidth = 1.2; c.strokeStyle = P.ink2; c.stroke()
  if (label) text(c, label, x, y + 0.5, Math.max(9, r * 0.9), P.ink4, 'center', 600)
}

// A small curve plot of ys over [x0, x1], with the y range [lo, hi].
export function curve(c: Ctx, ys: ArrayLike<number>, x: number, y: number, w: number, h: number, lo: number, hi: number, color: string, width = 1.8) {
  c.beginPath()
  for (let i = 0; i < ys.length; i++) { const px = x + i / (ys.length - 1) * w, py = y + h - (ys[i] - lo) / (hi - lo) * h; if (i) c.lineTo(px, py); else c.moveTo(px, py) }
  c.strokeStyle = color; c.lineWidth = width; c.lineJoin = 'round'; c.stroke()
}
