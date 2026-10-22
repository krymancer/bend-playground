import { P, mono, signed } from '@/lib/palette'
// Worlds and canvas drawing for the reinforcement-learning essay. Bend computes
// every value drawn here (demos/qlearning/agent.bend); this file only paints.

export const FLOOR = 0, WALL = 1, PIT = 2, GOAL = 3
export interface World { kinds: number[]; start: number }
const parse = (rows: string[]): World => { const s = rows.join(''); return { kinds: [...s].map(c => ({ '#': WALL, X: PIT, G: GOAL } as Record<string, number>)[c] ?? FLOOR), start: s.indexOf('S') } }
export const PRESETS = {
  cliff: parse(['........', '..####..', '.....#..', '.###.#..', '...#....', '.#...##.', '........', 'SXXXXXXG']),
  maze: parse(['S..#....', '.#.#.##.', '.#...#..', '.####.#.', '......#.', '.###X.#.', '...#..#.', '#X.#X..G']),
  field: parse(['.......G', '........', '..X..X..', '........', '...XX...', '........', '.X....X.', 'S.......']),
} as const
export type Preset = keyof typeof PRESETS
export const copy = (w: World): World => ({ kinds: [...w.kinds], start: w.start })

export interface GridView {
  kinds: number[]; start: number
  value?: Float32Array; action?: Uint8Array; spread?: Float32Array
  path?: ArrayLike<number>; agent?: number; hover?: number; scale?: number
}

// Square grid at (x0, y0), `size` pixels wide.
export function drawGrid(c: CanvasRenderingContext2D, x0: number, y0: number, size: number, g: GridView) {
  const cell = size / 8, gap = Math.max(1, cell * .04), scale = g.scale ?? 1
  const cx = (i: number) => x0 + (i % 8 + .5) * cell, cy = (i: number) => y0 + ((i / 8 | 0) + .5) * cell
  for (let i = 0; i < 64; i++) {
    const x = x0 + (i % 8) * cell + gap / 2, y = y0 + (i / 8 | 0) * cell + gap / 2, s = cell - gap, k = g.kinds[i]
    if (k === WALL) { c.fillStyle = P.ink3; c.fillRect(x, y, s, s); continue }
    const known = g.value && (!g.spread || g.spread[i] > 1e-6)
    c.fillStyle = k === GOAL ? P.grn : k === PIT ? P.paper3 : known ? signed(g.value![i] / scale) : P.paper
    c.fillRect(x, y, s, s)
    if (k === PIT) {
      c.fillStyle = P.red; c.beginPath(); c.arc(cx(i), cy(i), cell * .3, 0, Math.PI * 2); c.fill()
      c.fillStyle = P.ink4; c.globalAlpha = .35; c.beginPath(); c.ellipse(cx(i), cy(i) + cell * .05, cell * .19, cell * .13, 0, 0, Math.PI * 2); c.fill(); c.globalAlpha = 1
    }
    if (k === GOAL) {
      const px = cx(i) - cell * .14, top = cy(i) - cell * .26
      c.strokeStyle = P.paper; c.lineWidth = Math.max(1.5, cell * .05); c.beginPath(); c.moveTo(px, top); c.lineTo(px, cy(i) + cell * .26); c.stroke()
      c.fillStyle = P.paper; c.beginPath(); c.moveTo(px, top); c.lineTo(px + cell * .32, top + cell * .1); c.lineTo(px, top + cell * .2); c.closePath(); c.fill()
    }
    if (k === FLOOR && known && g.action) arrow(c, cx(i), cy(i), cell * .2, g.action[i])
  }
  if (g.hover != null && g.hover >= 0) { c.strokeStyle = P.vio; c.lineWidth = 2; c.strokeRect(x0 + (g.hover % 8) * cell + 1, y0 + (g.hover / 8 | 0) * cell + 1, cell - 2, cell - 2) }
  c.strokeStyle = P.vio; c.lineWidth = Math.max(1.5, cell * .07); c.beginPath(); c.arc(cx(g.start), cy(g.start), cell * .32, 0, Math.PI * 2); c.stroke()
  if (g.path && g.path.length > 1) {
    c.strokeStyle = P.vio; c.globalAlpha = .6; c.lineWidth = Math.max(2, cell * .08); c.lineJoin = c.lineCap = 'round'; c.beginPath()
    for (let j = 0; j < g.path.length; j++) { const i = g.path[j]; if (j) c.lineTo(cx(i), cy(i)); else c.moveTo(cx(i), cy(i)) }
    c.stroke(); c.globalAlpha = 1
  }
  if (g.agent != null) { const i = g.agent; c.fillStyle = P.vio; c.strokeStyle = P.paper; c.lineWidth = 2; c.beginPath(); c.arc(cx(i), cy(i), cell * .22, 0, Math.PI * 2); c.fill(); c.stroke() }
}

// A chevron pointing up, right, down or left.
function arrow(c: CanvasRenderingContext2D, x: number, y: number, r: number, a: number) {
  const ang = [-Math.PI / 2, 0, Math.PI / 2, Math.PI][a] ?? 0
  c.save(); c.translate(x, y); c.rotate(ang); c.fillStyle = P.ink4; c.globalAlpha = .7
  c.beginPath(); c.moveTo(r, 0); c.lineTo(-r * .6, -r * .75); c.lineTo(-r * .25, 0); c.lineTo(-r * .6, r * .75); c.closePath(); c.fill(); c.restore()
}

// Grid on the left, a side panel on the right; stacked when narrow.
export function layout(width: number, height: number) {
  const pad = Math.max(10, Math.min(width, height) * .04)
  if (width < height) { const s = Math.min(width - pad * 2, height * .66); return { gx: (width - s) / 2, gy: pad, gs: s, side: { x: pad, y: pad * 2 + s, w: width - pad * 2, h: height - s - pad * 3 } } }
  const s = height - pad * 2
  return { gx: pad, gy: pad, gs: s, side: { x: pad * 2.5 + s, y: pad, w: width - s - pad * 3.5, h: height - pad * 2 } }
}
export function cellAt(width: number, height: number, px: number, py: number) {
  const { gx, gy, gs } = layout(width, height), x = Math.floor((px - gx) / gs * 8), y = Math.floor((py - gy) / gs * 8)
  return x >= 0 && x < 8 && y >= 0 && y < 8 ? y * 8 + x : -1
}

// Return per episode: faint dots, and their moving average in violet.
export function drawReturns(c: CanvasRenderingContext2D, r: { x: number; y: number; w: number; h: number }, raw: number[], smooth: number[]) {
  const lo = -2, hi = 1, top = r.y + 18, h = r.h - 36, n = Math.max(raw.length, 50)
  const X = (i: number) => r.x + (i + .5) / n * r.w, Y = (v: number) => top + (hi - Math.max(lo, Math.min(hi, v))) / (hi - lo) * h
  c.font = mono(11); c.fillStyle = P.ink2; c.textBaseline = 'alphabetic'; c.textAlign = 'left'; c.fillText('return per episode', r.x, r.y + 10)
  c.strokeStyle = P.ink1; c.lineWidth = 1; c.setLineDash([3, 3])
  for (const v of [1, 0, -1]) { c.beginPath(); c.moveTo(r.x, Y(v)); c.lineTo(r.x + r.w, Y(v)); c.stroke() }
  c.setLineDash([]); c.fillStyle = P.ink1; c.textAlign = 'right'
  for (const v of [1, 0, -1]) c.fillText(v > 0 ? '+1' : String(v), r.x + r.w, Y(v) - 3)
  c.fillStyle = P.ink2; c.globalAlpha = .45
  const step = Math.max(1, Math.floor(raw.length / 600))
  for (let i = 0; i < raw.length; i += step) { c.fillRect(X(i) - 1, Y(raw[i]) - 1, 2, 2) }
  c.globalAlpha = 1
  if (smooth.length > 1) {
    c.strokeStyle = P.vio; c.lineWidth = 2; c.beginPath()
    for (let i = 0; i < smooth.length; i += step) { if (i) c.lineTo(X(i), Y(smooth[i])); else c.moveTo(X(i), Y(smooth[i])) }
    c.stroke()
  }
  c.fillStyle = P.ink2; c.textAlign = 'left'; c.fillText(`episode ${raw.length}`, r.x, r.y + r.h - 4)
  c.textAlign = 'right'; c.fillStyle = P.vio; c.fillText('average', r.x + r.w, r.y + r.h - 4)
}

// Largest change in V per sweep. The Bellman backup is a contraction:
// each change is at most gamma times the one before.
export function drawDeltas(c: CanvasRenderingContext2D, r: { x: number; y: number; w: number; h: number }, deltas: number[]) {
  const top = r.y + 18, h = r.h - 36, n = Math.max(deltas.length, 20), hi = 1.1
  const Y = (d: number) => top + (1 - Math.min(d, hi) / hi) * h
  c.font = mono(11); c.fillStyle = P.ink2; c.textAlign = 'left'; c.textBaseline = 'alphabetic'; c.fillText('largest change per sweep', r.x, r.y + 10)
  c.strokeStyle = P.ink1; c.lineWidth = 1; c.setLineDash([3, 3])
  for (const v of [1, .5, 0]) { c.beginPath(); c.moveTo(r.x, Y(v)); c.lineTo(r.x + r.w, Y(v)); c.stroke() }
  c.setLineDash([]); c.fillStyle = P.ink1; c.textAlign = 'right'
  for (const v of [1, .5]) c.fillText(String(v), r.x + r.w, Y(v) - 3)
  const bw = r.w / n
  deltas.forEach((d, i) => { c.fillStyle = i === deltas.length - 1 ? P.vio : P.bar; const y = Y(d); c.fillRect(r.x + i * bw + bw * .15, y, bw * .7, Math.max(1, top + h - y)) })
  c.fillStyle = P.ink2; c.textAlign = 'left'; c.fillText(`sweep ${deltas.length}`, r.x, r.y + r.h - 4)
}

// A compact legend row for the world.
export function drawLegend(c: CanvasRenderingContext2D, r: { x: number; y: number; w: number; h: number }) {
  const items: [string, (x: number, y: number, s: number) => void][] = [
    ['start', (x, y, s) => { c.strokeStyle = P.vio; c.lineWidth = 2; c.beginPath(); c.arc(x + s / 2, y + s / 2, s * .4, 0, 7); c.stroke() }],
    ['goal  +1', (x, y, s) => { c.fillStyle = P.grn; c.fillRect(x, y, s, s) }],
    ['pit   −1', (x, y, s) => { c.fillStyle = P.red; c.beginPath(); c.arc(x + s / 2, y + s / 2, s * .42, 0, 7); c.fill() }],
    ['wall', (x, y, s) => { c.fillStyle = P.ink3; c.fillRect(x, y, s, s) }],
    ['step  −0.04', (x, y, s) => { c.fillStyle = P.paper; c.fillRect(x, y, s, s) }],
    ['agent', (x, y, s) => { c.fillStyle = P.vio; c.beginPath(); c.arc(x + s / 2, y + s / 2, s * .35, 0, 7); c.fill() }],
  ]
  const rowH = Math.min(26, r.h / (items.length + 1)), s = Math.min(14, rowH * .7)
  c.font = mono(12); c.textBaseline = 'middle'; c.textAlign = 'left'
  items.forEach(([label, icon], i) => { const y = r.y + i * rowH; icon(r.x, y, s); c.fillStyle = P.ink3; c.fillText(label, r.x + s + 10, y + s / 2) })
  c.textBaseline = 'alphabetic'
}
