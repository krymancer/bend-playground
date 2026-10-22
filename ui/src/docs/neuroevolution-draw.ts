import { P, mono, signed } from '@/lib/palette'
import { surface } from '@/lib/lab-canvas'

// Pure drawing for the neuroevolution essay. Every number here was computed
// by demos/neuroevolution/flock.bend; these functions only map it to pixels.

export interface Frame {
  ys: Float32Array; alive: Uint8Array; pipes: Float32Array; weights: Float32Array
  inputs: Float32Array; hidden: Float32Array; out: Float32Array
  leader: number; t: number; generation: number; living: number; pipesNow: number
  best: Float32Array; avg: Float32Array; champion: number; computeMs: number
}

// World units from flock.bend.
const WORLD = { w: 640, h: 400, ground: 380, bx: 140, r: 11, pw: 56, gap: 120 }

export function drawWorld(canvas: HTMLCanvasElement, f: Frame | null) {
  const { c, width, height } = surface(canvas), k = width / WORLD.w
  c.save(); c.scale(k, height / WORLD.h)
  c.fillStyle = P.paper3; c.fillRect(0, WORLD.ground, WORLD.w, WORLD.h - WORLD.ground)
  c.strokeStyle = P.ink1; c.lineWidth = 1 / k; c.beginPath(); c.moveTo(0, WORLD.ground); c.lineTo(WORLD.w, WORLD.ground); c.stroke()
  if (!f) { c.restore(); return }
  for (let i = 0; i < f.pipes.length; i += 2) {
    const x = f.pipes[i], top = f.pipes[i + 1], bottom = top + WORLD.gap
    c.fillStyle = P.grn
    c.fillRect(x, 0, WORLD.pw, top); c.fillRect(x, bottom, WORLD.pw, WORLD.ground - bottom)
    c.fillStyle = P.ink3; c.globalAlpha = .35
    c.fillRect(x - 3, top - 12, WORLD.pw + 6, 12); c.fillRect(x - 3, bottom, WORLD.pw + 6, 12)
    c.globalAlpha = 1
  }
  // What the leader senses: the next gap's top and bottom.
  const ly = f.ys[f.leader]
  const next = nextPipe(f)
  if (next && f.alive[f.leader]) {
    c.setLineDash([4, 4]); c.strokeStyle = P.vio; c.lineWidth = 1.2
    c.beginPath(); c.moveTo(WORLD.bx, ly); c.lineTo(next.x, next.top); c.moveTo(WORLD.bx, ly); c.lineTo(next.x, next.top + WORLD.gap); c.stroke()
    c.setLineDash([])
  }
  c.fillStyle = P.ink2; c.globalAlpha = .22
  for (let i = 0; i < f.ys.length; i++) if (f.alive[i] && i !== f.leader) { c.beginPath(); c.arc(WORLD.bx, f.ys[i], WORLD.r, 0, 7); c.fill() }
  c.globalAlpha = f.alive[f.leader] ? 1 : .35
  c.fillStyle = P.vio; c.strokeStyle = P.ink4; c.lineWidth = 1.5
  c.beginPath(); c.arc(WORLD.bx, ly, WORLD.r, 0, 7); c.fill(); c.stroke()
  c.fillStyle = P.paper; c.beginPath(); c.arc(WORLD.bx + 4, ly - 3, 3.2, 0, 7); c.fill()
  c.fillStyle = P.ink4; c.beginPath(); c.arc(WORLD.bx + 5, ly - 3, 1.4, 0, 7); c.fill()
  c.fillStyle = P.ora; c.beginPath(); c.moveTo(WORLD.bx + 9, ly - 1); c.lineTo(WORLD.bx + 16, ly + 2); c.lineTo(WORLD.bx + 9, ly + 4); c.fill()
  c.globalAlpha = 1
  c.restore()
  c.font = mono(Math.max(11, width / 48)); c.fillStyle = P.ink3; c.textBaseline = 'top'
  c.fillText(`gen ${f.generation}   ${f.living}/${f.ys.length} alive   ${f.pipesNow} pipes`, 10, 10)
}

function nextPipe(f: Frame) {
  for (let i = 0; i < f.pipes.length; i += 2) if (f.pipes[i] + WORLD.pw > WORLD.bx - WORLD.r) return { x: f.pipes[i], top: f.pipes[i + 1] }
  return null
}

// A layered network diagram. weights: per neuron [bias, w...] rows for each
// non-input layer; acts: activations per layer (inputs first).
export interface Net { sizes: number[]; inLabels: string[]; outLabels: string[]; weights?: Float32Array; acts?: Float32Array[]; pick?: number }
export function drawNet(canvas: HTMLCanvasElement, net: Net) {
  const { c, width, height } = surface(canvas), { sizes } = net
  const fs = Math.max(10, Math.min(13, width / 30)), r0 = Math.max(4, Math.min(10, height / 26))
  c.font = mono(fs)
  const text = (ls: string[], weight = 400) => { c.font = mono(fs, weight); return Math.max(0, ...ls.map(l => c.measureText(l).width)) }
  const padL = text(net.inLabels) + r0 + 14, padR = text(net.outLabels, 700) + r0 + 14, padY = fs * 1.4
  const xs = sizes.map((_, i) => padL + (width - padL - padR) * i / (sizes.length - 1))
  const ys = sizes.map(n => Array.from({ length: n }, (_, j) => padY + (height - 2 * padY) * (n === 1 ? .5 : j / (n - 1))))
  const r = Math.max(4, Math.min(10, (height - 2 * padY) / (Math.max(...sizes) * 2.6)))
  let at = 0
  for (let l = 1; l < sizes.length; l++) for (let j = 0; j < sizes[l]; j++) {
    at++ // skip bias
    for (let i = 0; i < sizes[l - 1]; i++) {
      const w = net.weights ? net.weights[at++] : 0
      c.strokeStyle = net.weights ? signed(w / 1.5, .9) : P.ink1
      c.lineWidth = net.weights ? .4 + Math.min(2.6, Math.abs(w) * 1.3) : .6
      c.beginPath(); c.moveTo(xs[l - 1], ys[l - 1][i]); c.lineTo(xs[l], ys[l][j]); c.stroke()
    }
  }
  c.font = mono(fs); c.textBaseline = 'middle'
  sizes.forEach((_, l) => ys[l].forEach((y, j) => {
    const a = net.acts?.[l]?.[j], lit = a === undefined ? 0 : Math.max(0, Math.min(1, Math.abs(a)))
    c.fillStyle = P.paper; c.beginPath(); c.arc(xs[l], y, r, 0, 7); c.fill()
    c.fillStyle = P.vio; c.globalAlpha = lit; c.beginPath(); c.arc(xs[l], y, r, 0, 7); c.fill(); c.globalAlpha = 1
    const chosen = l === sizes.length - 1 && net.pick === j
    c.strokeStyle = chosen ? P.vio : P.ink2; c.lineWidth = chosen ? 2.5 : 1
    c.beginPath(); c.arc(xs[l], y, r, 0, 7); c.stroke()
    if (l === 0) { c.textAlign = 'right'; c.fillStyle = P.ink3; c.fillText(net.inLabels[j] ?? '', xs[l] - r - 6, y) }
    if (l === sizes.length - 1) { c.textAlign = 'left'; c.fillStyle = chosen ? P.vio : P.ink3; c.font = mono(fs, chosen ? 700 : 400); c.fillText(net.outLabels[j] ?? '', xs[l] + r + 6, y); c.font = mono(fs) }
  }))
}

// Best and average pipes cleared, one point per finished generation.
export function drawHistory(canvas: HTMLCanvasElement, best: Float32Array, avg: Float32Array, cap: number) {
  const { c, width, height } = surface(canvas), fs = Math.max(10, Math.min(12, width / 32))
  const L = fs * 3.2, R = 10, T = fs * 2.4, B = fs * 2.2, w = width - L - R, h = height - T - B
  const n = Math.max(best.length, 10), X = (i: number) => L + w * (n <= 1 ? 0 : i / (n - 1)), Y = (v: number) => T + h * (1 - v / cap)
  c.font = mono(fs); c.textBaseline = 'middle'; c.textAlign = 'right'; c.lineWidth = 1
  for (const v of [0, 5, 10, 15, 20, 25].filter(v => v <= cap)) { c.strokeStyle = P.line; c.beginPath(); c.moveTo(L, Y(v)); c.lineTo(L + w, Y(v)); c.stroke(); c.fillStyle = P.ink1; c.fillText(String(v), L - 6, Y(v)) }
  c.textAlign = 'center'; c.textBaseline = 'top'; c.fillStyle = P.ink1
  c.fillText('generation', L + w / 2, T + h + fs * .7)
  c.textAlign = 'left'; c.fillText(String(0), L, T + h + fs * .7)
  c.textAlign = 'right'; c.fillText(String(n - 1), L + w, T + h + fs * .7)
  const line = (data: Float32Array, color: string, width: number) => {
    if (!data.length) return
    c.strokeStyle = color; c.lineWidth = width; c.lineJoin = 'round'; c.beginPath()
    data.forEach((v, i) => { const x = X(i), y = Y(Math.min(v, cap)); if (i) c.lineTo(x, y); else c.moveTo(x, y) }); c.stroke()
    if (data.length === 1) { c.fillStyle = color; c.beginPath(); c.arc(X(0), Y(data[0]), 2.5, 0, 7); c.fill() }
  }
  line(avg, P.ora, 1.6); line(best, P.vio, 2.2)
  c.textBaseline = 'middle'; c.textAlign = 'left'; c.font = mono(fs)
  c.fillStyle = P.vio; c.fillRect(L, fs * .9, 12, 3); c.fillStyle = P.ink3; c.fillText('best', L + 17, fs)
  c.fillStyle = P.ora; c.fillRect(L + fs * 6, fs * .9, 12, 3); c.fillStyle = P.ink3; c.fillText('average', L + fs * 6 + 17, fs)
  if (!best.length) { c.textAlign = 'center'; c.fillStyle = P.ink1; c.fillText('pipes cleared per generation appear here', L + w / 2, T + h / 2) }
}
