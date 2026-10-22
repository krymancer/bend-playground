import { useCallback, useEffect, useRef, useState } from 'react'
import { Bars, Btn, Choice, Code, Controls, DocPage, Figure, Range, Section, Stage, Stat, Stats } from '@/components/Doc'
import { useWorker } from '@/lib/use-lab'
import { surface, usePaint } from '@/lib/lab-canvas'
import { P } from '@/lib/palette'
import { arrow, contour, curve, dots, edge, frame, heat, lineEnds, node, text, toX, toY, type Box } from './neural-draw'

// Every value drawn here comes from demos/neural/net.bend through neural.worker.ts.
const newWorker = () => new Worker(new URL('./neural.worker.ts', import.meta.url), { type: 'module' })
type Call = <T>(body: Record<string, unknown>) => Promise<T>
interface Unit { wx: number; wy: number; b: number; v: number }

// Send the latest request only; requests made while Bend is busy collapse into one.
function useLatest<T>(call: Call, body: Record<string, unknown>) {
  const [result, setResult] = useState<T | null>(null), busy = useRef(false), want = useRef<string | null>(null), key = JSON.stringify(body)
  useEffect(() => {
    want.current = key
    const run = () => {
      if (busy.current || want.current === null) return
      const next = want.current; want.current = null; busy.current = true
      call<T>(JSON.parse(next)).then(setResult, () => {}).finally(() => { busy.current = false; run() })
    }
    run()
  }, [key, call])
  return result
}

const signedField = (f: Float32Array, probability: boolean) => probability ? f.map(v => 2 * v - 1) : f
const fontPx = (w: number) => Math.max(9, Math.min(13, w / 55))

// ---------------------------------------------------------------- 1. neuron

const ACTS = [['0', 'sigmoid'], ['1', 'tanh'], ['2', 'step']] as const
type Act = typeof ACTS[number][0]
interface NeuronResult { field: Float32Array; points: Float32Array; hits: number; total: number; curve: Float32Array }

function NeuronFigure({ call }: { call: Call }) {
  const [w1, setW1] = useState(-1), [w2, setW2] = useState(1), [b, setB] = useState(0), [kind, setKind] = useState<Act>('0')
  const r = useLatest<NeuronResult>(call, { action: 'neuron', kind: Number(kind), w1, w2, b, res: 40, seed: 11 })
  const name = ACTS.find(a => a[0] === kind)![1]
  const ref = usePaint(canvas => {
    const { c, width: W, height: H } = surface(canvas), pad = Math.round(H * 0.04), fs = fontPx(W)
    const box: Box = { x: pad, y: pad, s: H - 2 * pad }
    if (r) heat(c, signedField(r.field, kind !== '1'), 40, box)
    frame(c, box)
    if (r) dots(c, r.points, box)
    const ends = lineEnds(w1, w2, b)
    if (ends) {
      const [x0, y0, x1, y1] = ends, mx = (x0 + x1) / 2, my = (y0 + y1) / 2, len = Math.hypot(w1, w2)
      c.save(); c.setLineDash([6, 4]); c.strokeStyle = P.ink4; c.lineWidth = 1.6
      c.beginPath(); c.moveTo(toX(box, x0), toY(box, y0)); c.lineTo(toX(box, x1), toY(box, y1)); c.stroke(); c.restore()
      arrow(c, toX(box, mx), toY(box, my), toX(box, mx + w1 / len * 0.3), toY(box, my + w2 / len * 0.3), P.vio, 2)
      text(c, 'w', toX(box, mx + w1 / len * 0.38), toY(box, my + w2 / len * 0.38), fs, P.vio, 'center', 600)
    }
    // Right: the neuron as a diagram, and its activation curve.
    const rx = box.x + box.s + pad * 1.5, rw = W - rx - pad
    const inX = rx + rw * 0.1, nX = rx + rw * 0.58, outX = rx + rw * 0.94, yx = H * 0.12, yy = H * 0.42, ny = H * 0.27, nr = Math.max(10, H * 0.06)
    edge(c, inX, yx, nX, ny, w1); edge(c, inX, yy, nX, ny, w2)
    const roomy = rw > 160
    if (roomy) text(c, `w1=${w1.toFixed(1)}`, inX + (nX - inX) * 0.42, yx + (ny - yx) * 0.42 - fs * 1.1, fs, P.ink3)
    if (roomy) text(c, `w2=${w2.toFixed(1)}`, inX + (nX - inX) * 0.42, yy + (ny - yy) * 0.42 + fs * 1.1, fs, P.ink3)
    node(c, inX, yx, nr * 0.8, 'x'); node(c, inX, yy, nr * 0.8, 'y'); node(c, nX, ny, nr, 'Σ')
    if (roomy) text(c, `b=${b.toFixed(2)}`, nX + nr * 0.6, ny - nr - fs * 0.9, fs, P.ink3, 'left')
    arrow(c, nX + nr + 2, ny, outX, ny, P.ink3, 1.3)
    if (roomy) text(c, name, (nX + outX) / 2 + nr / 2, ny + fs * 1.1, fs, P.ink2)
    const cy = H * 0.6, ch = H * 0.3, cx = rx + rw * 0.06, cw = rw * 0.88, lo = kind === '1' ? -1 : 0
    c.strokeStyle = P.bar; c.lineWidth = 1; c.beginPath(); c.moveTo(cx, cy + ch); c.lineTo(cx + cw, cy + ch); c.moveTo(cx + cw / 2, cy); c.lineTo(cx + cw / 2, cy + ch); c.stroke()
    if (r) curve(c, r.curve, cx, cy, cw, ch, lo, 1, P.vio)
    text(c, roomy ? 'z = w1·x + w2·y + b' : name, cx + cw / 2, cy + ch + fs * 1.2, fs, P.ink2)
    text(c, '−6', cx, cy + ch - fs * 0.7, fs * 0.85, P.ink1, 'left')
    text(c, '6', cx + cw, cy + ch - fs * 0.7, fs * 0.85, P.ink1, 'right')
  })
  return <Figure caption={<>The plane colored by the neuron’s output: <em>green</em> above the threshold, <span style={{ color: P.red }}>red</span> below. The dashed line is where <code>z = 0</code>; the arrow is the weight vector, always perpendicular to it.</>}>
    <Stage label="A single neuron"><canvas ref={ref} role="img" aria-label={`Plane colored by a ${name} neuron with weights ${w1}, ${w2} and bias ${b}; ${r ? r.hits : '?'} of 128 points on the correct side.`} /></Stage>
    <Controls>
      <Range label="w1" value={w1} min={-3} max={3} step={0.1} onChange={setW1} format={v => v.toFixed(1)} />
      <Range label="w2" value={w2} min={-3} max={3} step={0.1} onChange={setW2} format={v => v.toFixed(1)} />
      <Range label="b" value={b} min={-2} max={2} step={0.05} onChange={setB} format={v => v.toFixed(2)} />
      <Choice label="activation" value={kind} options={ACTS} onChange={setKind} />
    </Controls>
    <Stats><Stat label="correct side" value={r ? `${r.hits} / ${r.total}` : null} /><Stat label="line" value={`${w1.toFixed(1)}x ${w2 < 0 ? '−' : '+'} ${Math.abs(w2).toFixed(1)}y ${b < 0 ? '−' : '+'} ${Math.abs(b).toFixed(2)} = 0`} /></Stats>
  </Figure>
}

// ---------------------------------------------------------------- 2. layers

interface XorResult { hidden: Float32Array[]; field: Float32Array; points: Float32Array; acc: number }

function XorFigure({ call }: { call: Call }) {
  const [k, setK] = useState(4), r = useLatest<XorResult>(call, { action: 'xor', k, res: 40, seed: 5 })
  const ref = usePaint(canvas => {
    const { c, width: W, height: H } = surface(canvas), fs = fontPx(W), g = W * 0.035, lab = fs * 2.4
    const s = Math.min((W - 4 * g) / 3, H - lab - g), y = (H - s - lab) / 2, x0 = (W - 3 * s - 2 * g) / 2
    const labels = ['neuron A', 'neuron B', 'output']
    for (let i = 0; i < 3; i++) {
      const box: Box = { x: x0 + i * (s + g), y, s }
      if (r) {
        const f = i < 2 ? r.hidden[i] : signedField(r.field, true)
        heat(c, f, 40, box); contour(c, f, 40, box, i < 2 ? P.ink3 : P.ink4, i < 2 ? 1.2 : 1.8)
        dots(c, r.points, box, Math.max(2, s / 75))
      }
      frame(c, box)
      text(c, labels[i], box.x + s / 2, y + s + lab / 2, fs, i < 2 ? P.ink3 : P.ink4, 'center', 600)
      if (i < 2) text(c, i === 0 ? '+' : '→', box.x + s + g / 2, y + s / 2, fs * 1.3, P.ink2, 'center', 600)
    }
  })
  return <Figure caption={<>Two hidden <code>tanh</code> neurons each cut the plane once (left, middle). The output adds <code>A</code> and subtracts <code>B</code>, so only the band between the two lines stays <em>green</em> (right). Weights are set by hand, sharpness <code>k</code>.</>}>
    <Stage aspect="3 / 1.2" label="Two neurons solve XOR"><canvas ref={ref} role="img" aria-label="Three panels: neuron A, neuron B, and their combination, which separates the XOR pattern." /></Stage>
    <Controls><Range label="sharpness k" value={k} min={0.5} max={8} step={0.5} onChange={setK} format={v => v.toFixed(1)} /></Controls>
    <Stats><Stat label="XOR accuracy" value={r ? `${Math.round(r.acc * 100)}%` : null} /></Stats>
  </Figure>
}

const CAP_SIZES = [1, 2, 4, 8, 16], CAP_EPOCHS = 2000
function CapacityBars() {
  const { call } = useWorker(newWorker), [acc, setAcc] = useState<(number | null)[]>(CAP_SIZES.map(() => null))
  useEffect(() => {
    let alive = true
    ;(async () => {
      for (let i = 0; i < CAP_SIZES.length && alive; i++) {
        await call({ action: 'cap-reset', hidden: CAP_SIZES[i] })
        let r: { acc: number } = { acc: 0 }
        for (let e = 0; e < CAP_EPOCHS && alive; e += 50) { r = await call({ action: 'cap-train', epochs: 50 }); await new Promise(f => setTimeout(f, 4)) }
        if (alive) setAcc(a => a.map((v, j) => j === i ? r.acc : v))
      }
    })().catch(() => {})
    return () => { alive = false }
  }, [call])
  const best = acc.findIndex(a => a !== null && a >= 0.95)
  return <Bars max={100} title={<>Spiral accuracy after {CAP_EPOCHS} epochs</>} data={CAP_SIZES.map((h, i) => ({ label: `${h} hidden`, value: (acc[i] ?? 0) * 100, display: acc[i] === null ? '…' : `${Math.round(acc[i]! * 100)}%`, hi: i === best }))}
    caption={<>Trained live in Bend when this page loaded: same data, same seed, same learning rate; only the hidden layer changes. 50% is a coin flip.</>} />
}

// ---------------------------------------------------------------- 3. training

const DATASETS = [['0', 'two blobs'], ['1', 'xor'], ['2', 'circle'], ['3', 'spiral']] as const
type Kind = typeof DATASETS[number][0]
const PRESET: Record<Kind, [number, number]> = { '0': [2, 4], '1': [4, 4], '2': [6, 4], '3': [16, 5] }
const RATES = [0.03, 0.1, 0.3, 0.5, 1, 2, 3, 5], RES = 48, SMALL = 16, MAX_EPOCHS = 20000
interface Snap { epoch: number; loss: number; acc: number; c: number; units: Unit[]; field: Float32Array; hidden: Float32Array[]; points?: Float32Array; computeMs: number }

function drawTrainer(canvas: HTMLCanvasElement, snap: Snap | null, points: Float32Array | null, lines: boolean) {
  const { c, width: W, height: H } = surface(canvas), pad = Math.round(H * 0.04), fs = fontPx(W)
  const box: Box = { x: pad, y: pad, s: H - 2 * pad }
  if (snap) { const f = signedField(snap.field, true); heat(c, f, RES, box); contour(c, f, RES, box) }
  frame(c, box)
  if (snap && lines) {
    c.save(); c.beginPath(); c.rect(box.x, box.y, box.s, box.s); c.clip(); c.setLineDash([3, 4]); c.strokeStyle = P.vio; c.lineWidth = 1.2; c.globalAlpha = 0.8
    for (const u of snap.units) { const e = lineEnds(u.wx, u.wy, u.b); if (e) { c.beginPath(); c.moveTo(toX(box, e[0]), toY(box, e[1])); c.lineTo(toX(box, e[2]), toY(box, e[3])); c.stroke() } }
    c.restore()
  }
  if (points) dots(c, points, box)
  if (!snap) return
  // Right: inputs -> hidden neurons (each drawn as its own field) -> output.
  const rx = box.x + box.s + pad * 1.5, rw = W - rx - pad, n = snap.units.length, top = pad + fs * 1.6, avail = H - top - pad
  const m = Math.max(6, Math.min(rw * 0.3, avail / n - Math.max(2, avail * 0.012))), gap = (avail - n * m) / Math.max(1, n)
  const inX = rx + rw * 0.08, hX = rx + rw * 0.5, outX = rx + rw * 0.9, cy = (j: number) => top + gap / 2 + j * (m + gap) + m / 2
  const xs = [H * 0.38, H * 0.62], vmax = Math.max(1, ...snap.units.map(u => Math.abs(u.v)))
  snap.units.forEach((u, j) => {
    edge(c, inX, xs[0], hX - m / 2, cy(j), u.wx, Math.min(0.4, m / 50)); edge(c, inX, xs[1], hX - m / 2, cy(j), u.wy, Math.min(0.4, m / 50))
    edge(c, hX + m / 2, cy(j), outX, H / 2, u.v / vmax * 3, 0.6)
  })
  snap.units.forEach((_, j) => { const b: Box = { x: hX - m / 2, y: cy(j) - m / 2, s: m }; heat(c, snap.hidden[j], SMALL, b, 0.85); c.strokeStyle = P.ink2; c.lineWidth = 1; c.strokeRect(b.x + 0.5, b.y + 0.5, m - 1, m - 1) })
  const nr = Math.max(8, Math.min(16, rw * 0.07))
  node(c, inX, xs[0], nr, 'x'); node(c, inX, xs[1], nr, 'y'); node(c, outX, H / 2, nr, 'σ')
  text(c, 'input', inX, pad + fs * 0.5, fs * 0.9, P.ink2); text(c, `hidden ×${n}`, hX, pad + fs * 0.5, fs * 0.9, P.ink2); text(c, 'out', outX, pad + fs * 0.5, fs * 0.9, P.ink2)
}

function drawLoss(canvas: HTMLCanvasElement, hist: number[][]) {
  const { c, width: W, height: H } = surface(canvas), fs = fontPx(W) * 0.9, l = fs * 3.2, r = fs * 1.2, t = fs * 1.4, b = fs * 1.8
  const pw = W - l - r, ph = H - t - b, last = hist.at(-1)?.[0] ?? 0, span = Math.max(200, last), ymax = Math.max(1, ...hist.map(h => Math.min(h[1], 2)))
  c.strokeStyle = P.bar; c.lineWidth = 1; c.beginPath(); c.moveTo(l, t); c.lineTo(l, t + ph); c.lineTo(l + pw, t + ph); c.stroke()
  text(c, ymax.toFixed(1), l - fs * 0.5, t, fs, P.ink1, 'right'); text(c, '0', l - fs * 0.5, t + ph, fs, P.ink1, 'right')
  text(c, '0', l, t + ph + fs, fs, P.ink1, 'left'); text(c, `${span} epochs`, l + pw, t + ph + fs, fs, P.ink1, 'right')
  const line = (k: number, color: string, w: number) => {
    if (hist.length < 2) return
    c.beginPath(); hist.forEach((h, i) => { const x = l + h[0] / span * pw, y = t + ph - Math.min(h[k], ymax) / ymax * ph; if (i) c.lineTo(x, y); else c.moveTo(x, y) })
    c.strokeStyle = color; c.lineWidth = w; c.lineJoin = 'round'; c.stroke()
  }
  line(2, P.vio, 1.4); line(1, P.ink4, 1.8)
  text(c, 'loss', l + pw - fs * 9, t * 0.5, fs, P.ink4, 'right', 600); text(c, 'accuracy', l + pw, t * 0.5, fs, P.vio, 'right', 600)
}

function Trainer({ call }: { call: Call }) {
  const [kind, setKind] = useState<Kind>('2'), [hidden, setHidden] = useState(PRESET['2'][0]), [rate, setRate] = useState(PRESET['2'][1])
  const [init, setInit] = useState(7), [running, setRunning] = useState(false), [lines, setLines] = useState(false)
  const [snap, setSnap] = useState<Snap | null>(null), [points, setPoints] = useState<Float32Array | null>(null), [per, setPer] = useState(10)
  const hist = useRef<number[][]>([]), lr = useRef(RATES[rate]); lr.current = RATES[rate]
  const accept = useCallback((s: Snap) => {
    setSnap(s); const h = hist.current; h.push([s.epoch, s.loss, s.acc])
    if (h.length > 400) hist.current = h.filter((_, i) => i % 2 === 0 || i === h.length - 1)
  }, [])
  useEffect(() => {
    call<Snap>({ action: 'reset', kind: Number(kind), hidden, init, seed: 42, res: RES, small: SMALL }).then(s => { hist.current = []; setPoints(s.points!); accept(s) }, () => {})
  }, [call, accept, kind, hidden, init])
  useEffect(() => {
    if (!running) return
    // Start slowly so the first moves are visible, then speed up as learning slows,
    // within a per-call budget so each Bend call stays around 20 ms.
    let alive = true, budget = 40
    const tick = async () => {
      if (!alive) return
      try {
        const epoch = hist.current.at(-1)?.[0] ?? 0, epochs = Math.max(1, Math.min(budget, Math.ceil((epoch + 1) / 60)))
        const s = await call<Snap>({ action: 'train', epochs, lr: lr.current, res: RES, small: SMALL })
        if (!alive) return
        accept(s); setPer(epochs)
        if (epochs === budget) budget = s.computeMs < 20 ? Math.min(80, budget + 4) : Math.max(4, budget - 4)
        if (s.epoch >= MAX_EPOCHS) { setRunning(false); return }
      } catch { setRunning(false); return }
      requestAnimationFrame(tick)
    }
    tick(); return () => { alive = false }
  }, [running, call, accept])
  const step = () => call<Snap>({ action: 'train', epochs: 25, lr: lr.current, res: RES, small: SMALL }).then(accept, () => {})
  const pick = (k: Kind) => { setKind(k); setHidden(PRESET[k][0]); setRate(PRESET[k][1]) }
  const stage = usePaint(canvas => drawTrainer(canvas, snap, points, lines)), loss = usePaint(canvas => drawLoss(canvas, hist.current))
  const name = DATASETS.find(d => d[0] === kind)![1]
  return <Figure caption={<>Left: the network’s prediction over the plane, the black line where it says 50/50, and the 256 training points. Right: the network itself; each hidden neuron is drawn as its own little field, edges are <em>positive</em> or <span style={{ color: P.red }}>negative</span> weights. Below: the loss and accuracy as Bend trains.</>}>
    <Stage label="Training a neural network"><canvas ref={stage} role="img" aria-label={`Decision boundary of a ${hidden}-hidden-neuron network on the ${name} dataset at epoch ${snap?.epoch ?? 0}, accuracy ${snap ? Math.round(snap.acc * 100) : 0}%.`} /></Stage>
    <div style={{ marginTop: '.6em' }}><Stage aspect="16 / 3.2" label="Loss curve"><canvas ref={loss} role="img" aria-label={`Loss curve; current loss ${snap?.loss.toFixed(3) ?? 'unknown'}.`} /></Stage></div>
    <Controls>
      <Btn onClick={() => setRunning(r => !r)} pressed={running}>{running ? 'Pause' : 'Train'}</Btn>
      <Btn ghost onClick={step} disabled={running}>+25 epochs</Btn>
      <Btn ghost onClick={() => { setInit(i => i + 1) }} label="Reset with new random weights">Reset</Btn>
      <Btn ghost onClick={() => setLines(v => !v)} pressed={lines}>hidden lines</Btn>
    </Controls>
    <Controls>
      <Choice label="data" value={kind} options={DATASETS} onChange={pick} />
      <Range label="hidden" value={hidden} min={1} max={16} onChange={setHidden} />
      <Range label="learning rate" value={rate} min={0} max={RATES.length - 1} onChange={setRate} format={i => RATES[i]} />
    </Controls>
    <Stats>
      <Stat label="epoch" value={snap?.epoch} />
      <Stat label="loss" value={snap?.loss.toFixed(3)} />
      <Stat label="accuracy" value={snap ? `${Math.round(snap.acc * 100)}%` : null} />
      <Stat label="Bend" value={snap ? `${snap.computeMs.toFixed(0)} ms / ${running ? per : 25} epochs` : null} />
    </Stats>
  </Figure>
}

// ---------------------------------------------------------------- code excerpts

const NEURON = `
def sigmoid(z: F32) -> F32:
  (1.0 / (1.0 + F32.exp(F32.neg(z))) : F32)

def neuron(kind: Nat, w1: F32, w2: F32, b: F32, x: F32, y: F32) -> F32:
  act(kind,(w1 * x + w2 * y + b : F32))
`
const FORWARD = `
type Neuron is Data:
  Neuron{wx: F32, wy: F32, b: F32, v: F32}
type Net is Data:
  Net{units: List<&2, Neuron>, c: F32}

def hidden(units: List<&2, Neuron>, +x: F32, +y: F32) -> List<&2, F32>:
  match units:
    case []: []
    case Neuron{wx,wy,b,v} <> rest:
      F32.tanh((wx * x + wy * y + b : F32)) <> hidden(rest,x,y)

def logit(net: Net, +x: F32, +y: F32) -> F32:
  Net{+units,c} = net
  dot(units,hidden(units,x,y),c)   # c + sum of v * h
`
const BACKPROP = `
# d is dLoss/dz = q - t. Each hidden neuron gets d * v * tanh'(a).
def back(units: List<&2, Neuron>, hs: List<&2, F32>, +d: F32, +x: F32, +y: F32) -> List<&2, Neuron>:
  match units:
    case []: []
    case Neuron{wx,wy,b,v} <> rest:
      match hs:
        case []: []
        case +h <> hr:
          +a = (d * v * (1.0 - h * h) : F32)
          Neuron{(a * x : F32),(a * y : F32),a,(d * h : F32)} <> back(rest,hr,d,x,y)

def example(+net: Net, p: Pt) -> Grad:
  Net{+units,c} = net
  Pt{+x,+y,+t} = p
  +hs = hidden(units,x,y)
  +z = dot(units,hs,c)
  +q = sigmoid(z)
  +d = (q - t : F32)
  Grad{back(units,hs,d,x,y),d,loss(z,t),hit(q,t)}
`
const DESCENT = `
def train(epochs: Nat, fit: Fit, +batch: Batch, +lr: F32, +n: F32) -> Fit:
  match epochs:
    case 0n: fit
    case 1n+p:
      Fit{+net,l,a} = fit
      train(p,update(net,grad(net,batch),lr,n),batch,lr,n)
`
const PARALLEL = `
# A dataset is a balanced tree, so it splits into independent halves.
type Batch is Data:
  One{p: Pt}
  Two{l: Batch, r: Batch}

# The batch gradient is a sum over examples: split, recurse in parallel, add.
def grad(+net: Net, batch: Batch) -> Grad:
  match batch:
    case One{p}: example(net,p)
    case Two{l,r}:
      a b = grad(net,l) grad(net,r)
      plus(a,b)
`

// ---------------------------------------------------------------- page

export default function NeuralNetworks() {
  const { call, error } = useWorker(newWorker)
  return (
    <DocPage id="neural-networks" kicker="machine learning · 1" title="Neural networks" sub={<>a neuron <b>draws a line</b>, a layer <b>bends it</b>, and gradients <b>move it</b></>}
      tag={<>trained live in <b>Bend</b>, in your browser</>}>
      <p>A neural network is a function with knobs. You show it examples, measure
        how wrong it is, and turn every knob a little in the direction that makes
        it less wrong. Repeat a few thousand times. That is all of it.</p>
      <p>This page builds one from scratch: a single neuron, then a layer of them,
        then the training loop. Every figure is computed by a small Bend module,
        {' '}<code>net.bend</code>, running in a worker. The page only draws.</p>
      {error && <p className="note">Bend worker error: {error}</p>}

      <Section n={1} title={<>A neuron <b>WEIGHS</b>.</>}>
        <p>A neuron takes inputs, multiplies each by a <b>weight</b>, adds a
          {' '}<b>bias</b>, and squashes the total through an <b>activation</b> function.
          With two inputs <code>x</code> and <code>y</code>:</p>
        <p className="lead">out = act(w1·x + w2·y + b)</p>
        <p>The inside, <code>z = w1·x + w2·y + b</code>, is zero along a straight
          line. On one side <code>z</code> is positive, on the other negative. The
          activation only decides how sharply the output changes as you cross it:
          a <code>step</code> jumps, a <code>sigmoid</code> glides from 0 to 1,
          a <code>tanh</code> from −1 to 1.</p>
        <p>So a single neuron is a line with a soft edge. Drag the weights and try
          to put the two blobs on opposite sides.</p>
        <NeuronFigure call={call} />
        <p>The weights <code>(w1, w2)</code> set the line’s direction; the bias
          slides it. Bigger weights make the edge sharper without moving it. In
          Bend, the neuron is one line:</p>
        <Code file="demos/neural/net.bend">{NEURON}</Code>
        <p>Bend is typed and explicit: <code>(… : F32)</code> says the arithmetic
          is 32-bit float, and there is no <code>if</code>; branches are a
          {' '}<code>match</code>.</p>
      </Section>

      <Section n={2} title={<>Layers <b>BEND</b>.</>}>
        <p>Now take four clusters arranged like a checkerboard: top-right and
          bottom-left are green, the other two red. This is <b>XOR</b>, and no
          single line separates it: the best a lone neuron can do is put three
          of the four clusters on the right side.</p>
        <p>The fix is to stack neurons. A <b>hidden layer</b> of neurons each draws
          its own line; an output neuron then takes a weighted sum
          of <i>their</i> outputs. Two lines are enough for XOR: keep the band
          between them.</p>
        <XorFigure call={call} />
        <p>That is the whole trick of a neural network. Each hidden neuron
          contributes one soft edge, and the output mixes them. With enough
          edges, the boundary can curve around anything: a circle, a spiral, a
          face. The forward pass is two short functions:</p>
        <Code file="demos/neural/net.bend">{FORWARD}</Code>
        <p>A hidden neuron stores its two input weights, its bias, and
          {' '}<code>v</code>, the weight from it to the output. The <code>+</code> on
          {' '}<code>+x</code> marks a value read more than once: Bend is affine, so
          every value is used at most once unless you say otherwise.</p>
        <p>How many neurons does a spiral need? Fewer edges, fewer bends:</p>
        <CapacityBars />
      </Section>

      <Section n={3} title={<>Gradients <b>LEARN</b>.</>}>
        <p>Setting XOR’s weights by hand was easy. For a spiral with sixteen
          neurons there are 65 knobs; nobody tunes those by hand. We let the data
          do it.</p>
        <p>First, a number for “how wrong”. The output <code>q = sigmoid(z)</code>{' '}
          reads as the probability that a point is green. For a point with label
          {' '}<code>t</code> (1 green, 0 red), the <b>cross-entropy loss</b> is</p>
        <p className="lead">L = −[ t·log q + (1−t)·log(1−q) ]</p>
        <p>It is near zero when the network is confidently right and grows without
          bound when it is confidently wrong. Averaged over all points, it is one
          number to push down.</p>
        <p>Second, which way is down. The <b>gradient</b> is the derivative of
          {' '}<code>L</code> with respect to every weight: how much the loss changes if
          that weight moves a hair. The chain rule gives it in one sweep back from
          the output, called <b>backpropagation</b>:</p>
        <ul>
          <li>at the output: <code>∂L/∂z = q − t</code>. Call it <code>d</code>, the error.</li>
          <li>output weight of neuron j: <code>∂L/∂v = d · h</code>, where <code>h</code> is that neuron’s output.</li>
          <li>into neuron j: <code>d · v · (1 − h²)</code>, since tanh′ = 1 − tanh². Call it <code>a</code>.</li>
          <li>its input weights and bias: <code>a·x</code>, <code>a·y</code>, <code>a</code>.</li>
        </ul>
        <p>Each line reuses the one above it; that is why backprop is cheap. In Bend:</p>
        <Code file="demos/neural/net.bend">{BACKPROP}</Code>
        <p>Third, take a step. <b>Gradient descent</b> moves every weight against
          its gradient, averaged over the data: <code>w ← w − η · mean(∂L/∂w)</code>.
          The step size <code>η</code> is the <b>learning rate</b>. Too small and
          learning crawls; too large and it overshoots and the loss jumps around.</p>
        <Code file="demos/neural/net.bend">{DESCENT}</Code>
        <p className="lead">Press Train.</p>
        <Trainer call={call} />
        <p>Things to try. Set <b>xor</b> with <b>1</b> hidden neuron: it settles on one
          line and stalls near 75%. Give the <b>spiral</b> only 4 neurons and watch
          it run out of bends. Push the spiral’s learning rate to 5 and watch the loss curve
          wobble. Turn on <b>hidden lines</b> to see the edges each neuron
          contributes, then watch them rotate into place.</p>
        <p className="note">Hit Reset a few times on the spiral: different random starting
          weights find different solutions, and some take much longer. The loss
          surface is not a bowl.</p>
      </Section>

      <Section n={4} title={<>Batches <b>SPLIT</b>.</>}>
        <p>Look at what one epoch of training costs. The gradient of the average
          loss is the average of the per-example gradients. Each example’s gradient
          depends only on the current weights and that one example. So the sum
          over 256 points is 256 independent jobs, added together.</p>
        <p>That is the shape Bend is built for. Store the data as a balanced tree,
          and the gradient becomes a divide-and-conquer:</p>
        <Code file="demos/neural/net.bend">{PARALLEL}</Code>
        <p>The line <code>a b = grad(net,l) grad(net,r)</code> is a <b>parallel
          call</b>. It promises that the two halves are independent, which Bend
          can trust because it is pure and affine: no call can mutate what the other
          one reads. The two halves are equal-sized, so they finish together. Each
          half splits again, down to single examples: 256 leaves, 8 levels of
          forks, with nothing else to write. No threads, no locks, no reduction
          buffers.</p>
        <p>In this page Bend is compiled to JavaScript, which runs the forks one
          after another. Compile the same file natively and the forks spread over
          every CPU core; write <code>grad!(net, batch)</code> and they run on
          the GPU. The code does not change.</p>
        <p>Training a network is only one way to find good weights. On
          the <a href="#neuroevolution">neuroevolution</a> page, the same kind of
          network learns to fly with no gradients at all, by breeding. And
          on <a href="#cnn">convolutional nets</a>, neurons share weights across an
          image to see edges and shapes.</p>
      </Section>
    </DocPage>
  )
}
