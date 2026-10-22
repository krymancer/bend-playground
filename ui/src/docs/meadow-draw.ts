import { P, mono } from '@/lib/palette'
// Canvas drawing for the meadow essay. Nothing here simulates: it paints the
// typed arrays that meadow.worker.ts flattens out of Bend's World.
export const STRIDE = 8 // x, z, heading, energy, speed, metabolism, wool, flags
export const SIDE = 48
export type Tint = 'speed' | 'metab' | 'activity'
export interface Frame {
  grass: Float32Array; sheep: Float32Array; tick: number; time: number; population: number
  founders: number; births: number; old: number; starved: number; halted: boolean; food: number
  speed: number; metab: number; lambs: number; hungry: number; generation: number
  prints: [number, number][]; speeds?: number[]; metabs?: number[]; computeMs?: number
}

const rgb = (hex: string) => [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16))
const PAPER = rgb(P.paper2), GRASS = rgb(P.grn), DEEP = rgb(P.ink4)
const mix = (a: number[], b: number[], t: number) => a.map((v, i) => Math.round(v + (b[i] - v) * t))
const css = (c: number[]) => `rgb(${c[0]},${c[1]},${c[2]})`
export const ramp = (a: string, b: string, t: number) => css(mix(rgb(a), rgb(b), Math.max(0, Math.min(1, t))))

// Low -> wool white -> high, so the common middle reads as an ordinary sheep.
export const diverge = (lo: string, hi: string, t: number) => t < .5 ? ramp(lo, P.paper, t * 2) : ramp(P.paper, hi, t * 2 - 1)

// Grass ramp: bare paper -> meadow green, a touch darker at full biomass.
const LUT = Array.from({ length: 65 }, (_, i) => { const t = i / 64; return mix(mix(PAPER, GRASS, Math.min(1, t * 1.15)), DEEP, Math.max(0, t - .7) * .35) })
let tile: HTMLCanvasElement | null = null
export function drawField(c: CanvasRenderingContext2D, grass: Float32Array, x: number, y: number, size: number) {
  tile ??= Object.assign(document.createElement('canvas'), { width: SIDE, height: SIDE })
  const t = tile.getContext('2d')!, img = t.createImageData(SIDE, SIDE)
  for (let i = 0; i < SIDE * SIDE; i++) { const v = LUT[Math.round(Math.max(0, Math.min(4, grass[i])) * 16)]; img.data.set([v[0], v[1], v[2], 255], i * 4) }
  t.putImageData(img, 0, 0)
  c.imageSmoothingEnabled = false; c.drawImage(tile, x, y, size, size); c.imageSmoothingEnabled = true
  c.strokeStyle = P.bar; c.lineWidth = 1; c.strokeRect(x + .5, y + .5, size - 1, size - 1)
}

const ACT = [P.ink2, P.ora, P.blu, P.red] // wandering, seeking food, grazing, seeking a mate
export function sheepColor(s: Float32Array, o: number, tint: Tint) {
  if (tint === 'activity') return ACT[s[o + 7] & 3]
  if (tint === 'speed') return diverge(P.blu, P.ora, (s[o + 4] - .8) / .45)
  return diverge(P.vio, P.red, (s[o + 5] - .8) / .4)
}

export function drawFlock(c: CanvasRenderingContext2D, s: Float32Array, x: number, y: number, size: number, tint: Tint, scale = 1) {
  const cell = size / SIDE
  for (let o = 0; o < s.length; o += STRIDE) {
    const px = x + s[o] * cell, py = y + s[o + 1] * cell, h = s[o + 2], lamb = s[o + 7] & 4
    const r = cell * (lamb ? .42 : .62) * scale, dx = Math.sin(h), dy = Math.cos(h)
    c.save(); c.translate(px, py); c.rotate(Math.atan2(dy, dx))
    c.fillStyle = sheepColor(s, o, tint); c.strokeStyle = P.ink4; c.lineWidth = Math.max(.6, cell * .08)
    c.beginPath(); c.ellipse(0, 0, r * 1.25, r * .9, 0, 0, Math.PI * 2); c.fill(); c.stroke()
    c.fillStyle = P.ink4; c.beginPath(); c.arc(r * 1.25, 0, r * .45, 0, Math.PI * 2); c.fill()
    c.restore()
  }
}

// Fit a square meadow into a rect, centered.
export function square(w: number, h: number) { const size = Math.floor(Math.min(w, h)), x = Math.floor((w - size) / 2), y = Math.floor((h - size) / 2); return { x, y, size } }

export interface Sample { t: number; pop: number; food: number }
// Population (violet, left axis) and total food (green, right axis) over time.
export function drawHistory(c: CanvasRenderingContext2D, w: number, h: number, data: Sample[]) {
  const l = 46, r = 46, t = 14, b = 26, pw = w - l - r, ph = h - t - b
  c.font = mono(11); c.textBaseline = 'middle'
  if (data.length < 2) { c.fillStyle = P.ink1; c.textAlign = 'center'; c.fillText('press play in the meadow above', w / 2, h / 2); return }
  const t0 = data[0].t, t1 = Math.max(data.at(-1)!.t, t0 + 1), maxPop = Math.max(20, ...data.map(d => d.pop)) * 1.1
  const px = (v: number) => l + (v - t0) / (t1 - t0) * pw
  c.strokeStyle = P.line; c.lineWidth = 1
  for (let i = 0; i <= 4; i++) { const yy = t + ph * i / 4; c.beginPath(); c.moveTo(l, yy + .5); c.lineTo(l + pw, yy + .5); c.stroke() }
  const step = Math.max(1, Math.floor(data.length / pw))
  const line = (color: string, y: (d: Sample) => number) => { c.strokeStyle = color; c.lineWidth = 1.8; c.beginPath(); for (let i = 0; i < data.length; i += step) { const d = data[i]; i ? c.lineTo(px(d.t), y(d)) : c.moveTo(px(d.t), y(d)) } const d = data.at(-1)!; c.lineTo(px(d.t), y(d)); c.stroke() }
  line(P.grn, d => t + ph * (1 - d.food / 100))
  line(P.vio, d => t + ph * (1 - d.pop / maxPop))
  c.textAlign = 'right'; c.fillStyle = P.vio
  for (let i = 0; i <= 4; i += 2) c.fillText(String(Math.round(maxPop * (1 - i / 4))), l - 6, t + ph * i / 4)
  c.textAlign = 'left'; c.fillStyle = P.grn
  for (let i = 0; i <= 4; i += 2) c.fillText(`${100 - i * 25}%`, l + pw + 6, t + ph * i / 4)
  c.fillStyle = P.ink2; c.textAlign = 'center'; c.textBaseline = 'alphabetic'
  c.fillText(`${Math.round(t0)} s`, l + 16, h - 6); c.fillText(`${Math.round(t1)} s`, l + pw - 16, h - 6)
  c.fillStyle = P.vio; c.fillText('sheep', l + pw * .38, h - 6); c.fillStyle = P.grn; c.fillText('food', l + pw * .62, h - 6)
}

// Two histograms side by side: the founders (gray) and the living flock (violet), as shares.
export function drawTraits(c: CanvasRenderingContext2D, w: number, h: number, panels: { title: string; lo: number; hi: number; start?: number[]; now?: number[] }[]) {
  const gap = 24, pw = (w - gap * (panels.length - 1)) / panels.length
  panels.forEach((p, k) => {
    const x0 = k * (pw + gap), t = 22, b = 24, ph = h - t - b
    c.font = mono(11, 600); c.fillStyle = P.ink3; c.textAlign = 'center'; c.textBaseline = 'alphabetic'; c.fillText(p.title, x0 + pw / 2, 13)
    const share = (xs?: number[]) => { const n = xs?.reduce((a, v) => a + v, 0) || 1; return (xs ?? []).map(v => v / n) }
    const s = share(p.start), m = share(p.now), top = Math.max(.2, ...s, ...m), bins = 12, bw = pw / bins
    c.strokeStyle = P.line; c.beginPath(); c.moveTo(x0, t + ph + .5); c.lineTo(x0 + pw, t + ph + .5); c.stroke()
    for (let i = 0; i < bins; i++) {
      const bx = x0 + i * bw
      if (s[i]) { c.fillStyle = P.bar; c.fillRect(bx + 1, t + ph * (1 - s[i] / top), bw - 2, ph * s[i] / top) }
      if (m[i]) { c.fillStyle = P.vio; c.fillRect(bx + bw * .3, t + ph * (1 - m[i] / top), bw * .4, ph * m[i] / top) }
    }
    c.font = mono(10); c.fillStyle = P.ink2
    c.textAlign = 'left'; c.fillText(p.lo.toFixed(2), x0, h - 6); c.textAlign = 'right'; c.fillText(p.hi.toFixed(2), x0 + pw, h - 6)
  })
}

export interface Trace { t: number; e: number; act: number }
// Energy of one sheep over the last `span` seconds, with the 68 / 94 hysteresis band.
export function drawTrace(c: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, data: Trace[], span = 120) {
  const l = x + 30, pw = w - 34, t = y + 26, ph = h - 48
  const py = (e: number) => t + ph * (1 - (Math.max(40, e) - 40) / 60)
  c.fillStyle = 'rgba(139,131,181,.10)'; c.fillRect(l, py(94), pw, py(68) - py(94))
  c.font = mono(10); c.textBaseline = 'middle'; c.textAlign = 'right'; c.fillStyle = P.ink2
  for (const e of [40, 68, 94]) { c.fillText(String(e), l - 5, py(e)); c.strokeStyle = e > 40 ? P.vio : P.bar; c.setLineDash(e > 40 ? [3, 3] : []); c.beginPath(); c.moveTo(l, py(e) + .5); c.lineTo(l + pw, py(e) + .5); c.stroke() }
  c.setLineDash([])
  if (data.length < 2) return
  const t1 = Math.max(span, data.at(-1)!.t), t0 = t1 - span, px = (v: number) => l + (v - t0) / span * pw
  c.lineWidth = 2
  for (let i = 1; i < data.length; i++) {
    const a = data[i - 1], b = data[i]; if (b.t < t0) continue
    c.strokeStyle = ACT[b.act & 3]; c.beginPath(); c.moveTo(px(Math.max(t0, a.t)), py(a.e)); c.lineTo(px(b.t), py(b.e)); c.stroke()
  }
  c.lineWidth = 1; c.textAlign = 'center'; c.textBaseline = 'alphabetic'; c.fillStyle = P.ink2
  c.fillText(`${Math.round(t0)} s`, l + 18, y + h - 4); c.fillText(`${Math.round(t1)} s`, l + pw - 18, y + h - 4)
}

export const ACTIVITY = [['wandering', P.ink2], ['seeking food', P.ora], ['grazing', P.blu], ['seeking a mate', P.red]] as const
