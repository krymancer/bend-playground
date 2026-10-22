import { surface } from '@/lib/lab-canvas'
import { P, mono, signed } from '@/lib/palette'
// Canvas drawings for the deepfakes essay. They only draw: every face, kernel,
// feature map and count comes from demos/deepfake/meso.bend via the worker.

type Ctx = CanvasRenderingContext2D
const rgb = (h: string) => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16))
const DARK = rgb(P.ink4), LIGHT = rgb(P.paper)
// Grayscale in the paper/ink family: 0 is ink, 1 is paper.
export function shade(v: number) {
  const t = Math.max(0, Math.min(1, v)), [r, g, b] = DARK.map((d, i) => Math.round(d + (LIGHT[i] - d) * t))
  return `rgb(${r},${g},${b})`
}
const VIO = rgb(P.vio), PAPER2 = rgb(P.paper2)
export function violet(v: number) {
  const t = Math.max(0, Math.min(1, v)), [r, g, b] = PAPER2.map((p, i) => Math.round(p + (VIO[i] - p) * t))
  return `rgb(${r},${g},${b})`
}

// n x n cells of values, painted by `color`.
export function cells(c: Ctx, values: ArrayLike<number>, offset: number, n: number, x: number, y: number, size: number, color: (v: number) => string) {
  const s = size / n
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) { c.fillStyle = color(values[offset + j * n + i]); c.fillRect(x + i * s, y + j * s, Math.ceil(s), Math.ceil(s)) }
}

function text(c: Ctx, t: string, x: number, y: number, px: number, color: string = P.ink2, align: CanvasTextAlign = 'center', weight = 400) {
  c.font = mono(px, weight); c.fillStyle = color; c.textAlign = align; c.textBaseline = 'middle'; c.fillText(t, x, y)
}
function arrow(c: Ctx, x0: number, y0: number, x1: number, y1: number, color: string = P.ink1) {
  const a = Math.atan2(y1 - y0, x1 - x0), h = 6
  c.strokeStyle = color; c.fillStyle = color; c.lineWidth = 1.2
  c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1 - Math.cos(a) * h, y1 - Math.sin(a) * h); c.stroke()
  c.beginPath(); c.moveTo(x1, y1); c.lineTo(x1 - Math.cos(a - .45) * h, y1 - Math.sin(a - .45) * h); c.lineTo(x1 - Math.cos(a + .45) * h, y1 - Math.sin(a + .45) * h); c.closePath(); c.fill()
}

// 1. The pipeline: video -> frames -> face crop -> CNN -> verdict ----------
export function drawPipeline(canvas: HTMLCanvasElement, patches: Float32Array | null, narrow: boolean) {
  const { c, width: W, height: H } = surface(canvas), n = 5
  const labels = ['video', 'frame', 'face crop', 'CNN', 'verdict']
  const s = narrow ? Math.min(W * .34, H / n * .62) : Math.min(W / n * .6, H * .58)
  const centers = Array.from({ length: n }, (_, i) => narrow ? { x: W * .36, y: H * (i + .5) / n } : { x: W * (i + .5) / n, y: H * .44 })
  const face = (k: number, x: number, y: number, size: number) => { if (patches) cells(c, patches, k * 256, 16, x, y, size, shade); else { c.fillStyle = P.paper3; c.fillRect(x, y, size, size) } }
  centers.forEach(({ x, y }, i) => {
    const l = x - s / 2, t = y - s / 2
    c.lineWidth = 1
    if (i === 0) {
      for (let f = 2; f >= 0; f--) { const o = f * s * .09; c.fillStyle = P.paper; c.fillRect(l + o, t - o + s * .12, s * .82, s * .7); c.strokeStyle = P.ink1; c.strokeRect(l + o, t - o + s * .12, s * .82, s * .7) }
      face(0, l + s * .22, t + s * .2, s * .54)
      c.fillStyle = P.ink3
      for (let h = 0; h < 5; h++) { c.fillRect(l + s * (.05 + h * .16), t + s * .14, s * .06, s * .04); c.fillRect(l + s * (.05 + h * .16), t + s * .76, s * .06, s * .04) }
    } else if (i === 1) {
      c.fillStyle = P.paper; c.fillRect(l, t + s * .1, s, s * .8); c.strokeStyle = P.ink1; c.strokeRect(l, t + s * .1, s, s * .8)
      face(1, l + s * .25, t + s * .17, s * .5)
      c.setLineDash([4, 3]); c.strokeStyle = P.vio; c.lineWidth = 1.6; c.strokeRect(l + s * .3, t + s * .22, s * .4, s * .44); c.setLineDash([])
    } else if (i === 2) {
      face(1, l + s * .1, t + s * .1, s * .8); c.strokeStyle = P.vio; c.lineWidth = 1.6; c.strokeRect(l + s * .1, t + s * .1, s * .8, s * .8)
    } else if (i === 3) {
      const hs = [.84, .66, .5, .36, .22]
      hs.forEach((h, k) => { const w = s * (.06 + k * .025), xx = l + s * (.02 + k * .2); c.fillStyle = k === 4 ? P.vio : P.paper3; c.fillRect(xx, y - s * h / 2, w, s * h); c.strokeStyle = P.ink2; c.lineWidth = 1; c.strokeRect(xx, y - s * h / 2, w, s * h) })
    } else {
      const pill = (yy: number, label: string, on: boolean) => {
        c.fillStyle = on ? P.vio : P.paper; c.strokeStyle = on ? P.vio : P.ink1; c.lineWidth = 1
        c.beginPath(); c.roundRect(l + s * .1, yy, s * .8, s * .28, s * .14); c.fill(); c.stroke()
        text(c, label, x, yy + s * .14, Math.max(10, s * .15), on ? P.paper : P.ink2, 'center', 600)
      }
      pill(t + s * .14, 'real', false); pill(t + s * .54, 'fake', true)
    }
    if (narrow) text(c, labels[i], x + s / 2 + 14, y, 12, P.ink3, 'left')
    else text(c, labels[i], x, y + s / 2 + 16, 12, P.ink3)
    if (i < n - 1) {
      const nx = centers[i + 1]
      if (narrow) arrow(c, x, y + s / 2 + 3, x, nx.y - s / 2 - 3)
      else arrow(c, x + s / 2 + 4, y, nx.x - s / 2 - 4, y)
    }
  })
}

// 2. Meso-4, the layer sizes from Afchar et al. (thesis Figure 8) -----------
const MESO = [
  { shape: '256×256×3', side: 256, ch: 3, op: 'conv 8×(3×3)+ReLU · BN · max-pool 2×2' },
  { shape: '128×128×8', side: 128, ch: 8, op: 'conv 8×(5×5)+ReLU · BN · max-pool 2×2' },
  { shape: '64×64×8', side: 64, ch: 8, op: 'conv 16×(5×5)+ReLU · BN · max-pool 2×2' },
  { shape: '32×32×16', side: 32, ch: 16, op: 'conv 16×(5×5)+ReLU · BN · max-pool 4×4' },
  { shape: '8×8×16', side: 8, ch: 16, op: 'flatten' },
  { shape: '1024', side: 0, ch: 1024, op: 'dropout 0.5 · dense 16' },
  { shape: '16', side: 0, ch: 16, op: 'dropout 0.5 · dense 1 · sigmoid' },
  { shape: '1', side: 0, ch: 1, op: '' },
]
export function drawMeso(canvas: HTMLCanvasElement) {
  const { c, width: W, height: H } = surface(canvas), n = MESO.length
  // A tensor is a slab: its height grows with the side, its thickness with the channels.
  const slab = (x: number, y: number, unit: number, i: number) => {
    const m = MESO[i], vec = m.side === 0
    const h = vec ? unit * (m.ch === 1024 ? 1 : m.ch === 16 ? .45 : .12) : unit * (.25 + .75 * Math.log2(m.side) / 8)
    const w = vec ? unit * .07 : unit * (.05 + .025 * Math.log2(m.ch) * 1.6), d = vec ? 0 : unit * .12
    c.lineWidth = 1; c.strokeStyle = P.ink2
    if (d) {
      c.fillStyle = P.paper3; c.beginPath(); c.moveTo(x - w / 2, y - h / 2); c.lineTo(x - w / 2 + d, y - h / 2 - d); c.lineTo(x + w / 2 + d, y - h / 2 - d); c.lineTo(x + w / 2 + d, y + h / 2 - d); c.lineTo(x + w / 2, y + h / 2); c.lineTo(x + w / 2, y - h / 2); c.closePath(); c.fill(); c.stroke()
    }
    c.fillStyle = i === n - 1 ? P.vio : vec ? P.bar : P.paper; c.fillRect(x - w / 2, y - h / 2, w, h); c.strokeRect(x - w / 2, y - h / 2, w, h)
  }
  {
    const row = (H - 14) / n, unit = row * .72, x = W * .1
    MESO.forEach((m, i) => {
      const y = 10 + row * (i + .5)
      slab(x, y, unit, i)
      text(c, m.shape, W * .19, y, 11, P.ink3, 'left', 600)
      if (m.op) {
        arrow(c, W * .4, y + row * .15, W * .4, y + row * .85)
        c.font = mono(10)
        if (c.measureText(m.op).width < W * .55) text(c, m.op, W * .43, y + row * .5, 10, P.ink2, 'left')
        else { const [head, ...rest] = m.op.split(' · '); text(c, head, W * .43, y + row * .5 - 6, 10, P.ink2, 'left'); text(c, rest.join(' · '), W * .43, y + row * .5 + 7, 10, P.ink2, 'left') }
      } else text(c, 'real or fake', W * .43, y, 11, P.vio, 'left', 600)
    })
  }
}

// 3. Sample faces with the live p(fake), and the artifact itself ------------
export function drawSamples(canvas: HTMLCanvasElement, patches: Float32Array | null, labels: Float32Array | null, preds: Float32Array | null, trace: Float32Array | null) {
  const { c, width: W, height: H } = surface(canvas), cols = 4, rows = 2
  const cw = W / cols, ch = H / rows, size = Math.min(cw * .78, ch - 58)
  for (let k = 0; k < 7; k++) {
    const x = (k % cols) * cw + (cw - size) / 2, y = Math.floor(k / cols) * ch + ch * .08
    if (k < 6) {
      if (patches) cells(c, patches, k * 256, 16, x, y, size, shade)
      const fake = labels?.[k] === 1, p = preds?.[k]
      text(c, fake ? 'fake' : 'real', x + size / 2, y + size + 12, 11, fake ? P.vio : P.ink3, 'center', 600)
      const by = y + size + 24
      c.fillStyle = P.paper3; c.fillRect(x, by, size, 5)
      if (p !== undefined) {
        const right = (p >= .5) === fake
        c.fillStyle = right ? P.grn : P.red; c.fillRect(x, by, size * p, 5)
        text(c, p.toFixed(2), x + size / 2, by + 14, 10, P.ink2)
      }
    } else if (trace) {
      let m = 1e-6; for (const v of trace) m = Math.max(m, Math.abs(v))
      cells(c, trace, 0, 16, x, y, size, v => signed(v / m))
      text(c, 'the artifact', x + size / 2, y + size + 12, 11, P.ink3, 'center', 600)
      text(c, `±${m.toFixed(3)}`, x + size / 2, y + size + 38, 10, P.ink2)
    }
  }
}

// 4. Training: accuracy curves and the held-out confusion matrix ------------
export interface Point { step: number; train: number; val: number }
export function drawTraining(canvas: HTMLCanvasElement, history: Point[], confusion: Float32Array | null, maxStep: number) {
  const { c, width: W, height: H } = surface(canvas)
  const plot = { x: 40, y: 34, w: W - 56, h: H * .4 }
  const lo = .3, X = (s: number) => plot.x + plot.w * s / maxStep, Y = (a: number) => plot.y + plot.h * (1 - (a - lo) / (1 - lo))
  c.lineWidth = 1
  for (const a of [.3, .5, .75, 1]) {
    c.strokeStyle = a === .5 ? P.ink1 : P.line; c.setLineDash(a === .5 ? [4, 4] : []); c.beginPath(); c.moveTo(plot.x, Y(a)); c.lineTo(plot.x + plot.w, Y(a)); c.stroke(); c.setLineDash([])
    text(c, `${Math.round(a * 100)}%`, plot.x - 6, Y(a), 10, P.ink2, 'right')
  }
  text(c, 'chance', plot.x + plot.w - 2, Y(.5) - 8, 10, P.ink1, 'right')
  for (let s = 0; s <= maxStep; s += maxStep / 4) text(c, String(s), X(s), plot.y + plot.h + 12, 10, P.ink2)
  text(c, 'training step', plot.x + plot.w / 2, plot.y + plot.h + 28, 10, P.ink2)
  const line = (key: 'train' | 'val', color: string, width: number) => {
    const pts = history.filter(p => Number.isFinite(p[key])); if (!pts.length) return
    c.strokeStyle = color; c.lineWidth = width; c.lineJoin = 'round'; c.beginPath()
    pts.forEach((p, i) => i ? c.lineTo(X(p.step), Y(Math.max(lo, p[key]))) : c.moveTo(X(p.step), Y(Math.max(lo, p[key])))); c.stroke()
  }
  line('train', P.ink2, 1.4); line('val', P.vio, 2.2)
  text(c, '— train', plot.x, 14, 11, P.ink2, 'left'); text(c, '— held-out', plot.x + 70, 14, 11, P.vio, 'left', 600)

  // Rows: the truth. Columns: the verdict. Fake is the positive class.
  const top = plot.y + plot.h + 72, side = Math.min(W * .5, H - top - 10), box = { x: (W - side) / 2, y: top, s: side }
  const s = box.s / 2, [tp, fp, fn, tn] = confusion ?? [0, 0, 0, 0]
  const grid = [[tn, fp], [fn, tp]], names = ['real', 'fake']
  text(c, 'held-out verdicts', box.x + box.s / 2, box.y - 28, 11, P.ink3, 'center', 600)
  names.forEach((nm, i) => { text(c, nm, box.x + s * (i + .5), box.y - 10, 10, P.ink2); text(c, nm, box.x - 6, box.y + s * (i + .5), 10, P.ink2, 'right') })
  grid.forEach((row, r) => {
    const total = Math.max(1, row[0] + row[1])
    row.forEach((v, k) => {
      const right = r === k, x = box.x + k * s, y = box.y + r * s
      c.fillStyle = right ? violet(v / total) : signed(-v / total * .8); c.fillRect(x + 1, y + 1, s - 2, s - 2)
      text(c, String(Math.round(v)), x + s / 2, y + s / 2 - 6, Math.max(12, s * .2), v / total > .6 ? P.paper : P.ink4, 'center', 600)
      text(c, right ? (r ? 'caught' : 'passed') : (r ? 'missed' : 'false alarm'), x + s / 2, y + s / 2 + s * .2, Math.min(9.5, s / 7), v / total > .6 ? P.paper : P.ink2)
    })
  })
  text(c, 'truth ↓', box.x - 6, box.y - 10, 10, P.ink1, 'right'); text(c, 'verdict →', box.x + box.s + 6, box.y - 10, 10, P.ink1, 'left')
}

// 5. Kernels and what they see ---------------------------------------------
export function drawKernels(canvas: HTMLCanvasElement, kernels: Float32Array | null, out: Float32Array | null, maps: Float32Array | null, narrow: boolean) {
  const { c, width: W, height: H } = surface(canvas), K = 4
  const left = narrow ? 8 : 112, colW = (W - left - 8) / K, lab = narrow ? 42 : 28
  const size = Math.min(colW * .78, (H - 3 * lab - 12) / 3), rowY = [0, 1, 2].map(r => 8 + lab + r * (size + lab))
  const rows = ['kernel (3×3)', 'on a real face', 'on a fake face']
  rows.forEach((r, i) => narrow ? text(c, r, W / 2, rowY[i] - 11, 11, P.ink3, 'center', 600) : text(c, r, 6, rowY[i] + size / 2, 11, P.ink3, 'left', 600))
  if (!kernels || !out || !maps) return
  let wmax = 1e-6, mmax = 1e-6
  for (const v of kernels) wmax = Math.max(wmax, Math.abs(v))
  for (const v of maps) mmax = Math.max(mmax, v)
  for (let k = 0; k < K; k++) {
    const x = left + k * colW + (colW - size) / 2
    cells(c, kernels, k * 9, 3, x, rowY[0], size, v => signed(v / wmax))
    c.strokeStyle = P.line; c.lineWidth = 1; c.strokeRect(x, rowY[0], size, size)
    const vote = out[k]
    text(c, `vote ${vote >= 0 ? '+' : '−'}${Math.abs(vote).toFixed(2)}`, x + size / 2, rowY[0] + size + (narrow ? 9 : 10), 10, vote > 0 ? P.vio : P.ink2, 'center', vote > 0 ? 600 : 400)
    for (let r = 0; r < 2; r++) { cells(c, maps, (k * 2 + r) * 196, 14, x, rowY[r + 1], size, v => violet(v / mmax)); c.strokeStyle = P.line; c.strokeRect(x, rowY[r + 1], size, size) }
  }
}
