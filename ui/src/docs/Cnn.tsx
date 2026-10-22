import { useCallback, useEffect, useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from 'react'
import { Bars, Btn, Choice, Code, Controls, DocPage, Figure, Range, Section, Stage, Stat, Stats } from '@/components/Doc'
import { useWorker } from '@/lib/use-lab'
import { usePaint, surface } from '@/lib/lab-canvas'
import { P, mono, signed } from '@/lib/palette'
import {
  KERNELS, KERNEL_OPTIONS, LAYER1, LAYER2, N, PRESETS, PRESET_OPTIONS, brush, fmt, gray, maxAbs, outline, paintMap,
  type Box, type Kernel, type KernelName, type PresetName,
} from './convolution-data'

type Call = <T>(body: Record<string, unknown>) => Promise<T>
interface Input { img: Float32Array; ver: number }
const C = N - 2, Q = C / 2, R = Q - 2, TN = 20

// Latest-wins requests: while Bend is busy, only the newest pending request
// runs next, so dragging the brush never queues up stale work.
function useLatest<T>(run: (() => Promise<T>) | null, deps: unknown[]) {
  const [value, setValue] = useState<T | null>(null), busy = useRef(false), next = useRef<(() => Promise<T>) | null>(null)
  useEffect(() => {
    if (!run) return
    const go = (r: () => Promise<T>) => { busy.current = true; r().then(setValue, () => {}).finally(() => { busy.current = false; const n = next.current; next.current = null; if (n) go(n) }) }
    if (busy.current) next.current = run; else go(run)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return value
}

// Maps a pointer to a cell of an n x n map drawn in `box`, or null outside it.
function cellAt(e: ReactPointerEvent<HTMLCanvasElement>, box: Box | null, n: number) {
  if (!box) return null
  const r = e.currentTarget.getBoundingClientRect(), x = Math.floor((e.clientX - r.left - box.x) / box.s * n), y = Math.floor((e.clientY - r.top - box.y) / box.s * n)
  return x >= 0 && y >= 0 && x < n && y < n ? { x, y } : null
}
// Canvas text; shrinks the font (down to 7px) until it fits in maxW.
const label = (c: CanvasRenderingContext2D, text: string, x: number, y: number, size: number, color: string = P.ink2, align: CanvasTextAlign = 'center', maxW = Infinity) => {
  c.font = mono(size)
  for (let s = size; s > 7 && c.measureText(text).width > maxW; s -= .5) c.font = mono(s - .5)
  c.fillStyle = color; c.textAlign = align; c.textBaseline = 'alphabetic'; c.fillText(text, x, y)
}
const arrow = (c: CanvasRenderingContext2D, x0: number, x1: number, y: number) => {
  c.strokeStyle = P.ink1; c.fillStyle = P.ink1; c.lineWidth = 1.2; c.beginPath(); c.moveTo(x0, y); c.lineTo(x1 - 5, y); c.stroke()
  c.beginPath(); c.moveTo(x1, y); c.lineTo(x1 - 6, y - 3.5); c.lineTo(x1 - 6, y + 3.5); c.fill()
}

// ---------------------------------------------------------------------------
// 1. The input: draw on it or pick a preset.

function InputFigure({ input, onChange }: { input: Input; onChange: (img: Float32Array) => void }) {
  const [preset, setPreset] = useState<PresetName>('digit'), [erase, setErase] = useState(false), [hover, setHover] = useState<{ x: number; y: number } | null>(null)
  const box = useRef<Box | null>(null), drawing = useRef(false), img = useRef(input.img); img.current = input.img
  const focus = hover ?? { x: 14, y: 9 }
  const ref = usePaint(canvas => {
    const { c, width, height } = surface(canvas), pad = Math.max(8, height * .06), s = Math.min(height - 2 * pad, width * .56)
    const b = box.current = { x: pad, y: (height - s) / 2, s }
    paintMap(c, input.img, N, b, 'gray', 1, true)
    const r = 2, x0 = Math.max(0, Math.min(N - 5, focus.x - r)), y0 = Math.max(0, Math.min(N - 5, focus.y - r))
    outline(c, b, N, x0, y0, 5, 5, P.vio, 1.5)
    if (hover) outline(c, b, N, hover.x, hover.y, 1, 1, P.ora, 1.5)
    // the same 5 x 5 patch, as the numbers Bend actually sees
    const left = b.x + s + pad * 1.6, cw = (width - left - pad) / 5, ch = Math.min(cw * .72, s / 5.6), top = b.y + (s - ch * 5) / 2 + ch * .3
    label(c, 'the numbers inside', left + cw * 2.5, top - ch * .45, Math.max(9, Math.min(13, cw * .2)), P.ink2)
    for (let j = 0; j < 5; j++) for (let i = 0; i < 5; i++) {
      const v = input.img[(y0 + j) * N + x0 + i], hit = hover && hover.x === x0 + i && hover.y === y0 + j
      c.fillStyle = gray(v * .55); c.fillRect(left + i * cw + 1, top + j * ch + 1, cw - 2, ch - 2)
      label(c, v.toFixed(2), left + (i + .5) * cw, top + (j + .66) * ch, Math.max(8, Math.min(14, cw * .24)), hit ? P.ora : v > .6 ? P.paper : P.ink4)
    }
  })
  const paint = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect(), b = box.current; if (!b) return
    const x = (e.clientX - r.left - b.x) / b.s * N, y = (e.clientY - r.top - b.y) / b.s * N
    if (x < -1 || y < -1 || x > N + 1 || y > N + 1) return
    img.current = brush(img.current, x, y, erase); onChange(img.current)
  }
  return <Figure caption={<>Your input: {N}×{N} = {N * N} numbers between 0 (paper) and 1 (ink). Draw with a mouse or finger; the violet square is read out on the right.</>}>
    <Stage aspect="16 / 9">
      <canvas ref={ref} role="img" aria-label={`A ${N} by ${N} grayscale image you can draw on, with a 5 by 5 patch of its pixel values printed beside it`} style={{ touchAction: 'none', cursor: 'crosshair' }}
        onPointerDown={e => { drawing.current = true; e.currentTarget.setPointerCapture(e.pointerId); paint(e) }}
        onPointerMove={e => { setHover(cellAt(e, box.current, N)); if (drawing.current) paint(e) }}
        onPointerUp={() => { drawing.current = false }} onPointerCancel={() => { drawing.current = false }} onPointerLeave={() => setHover(null)} />
    </Stage>
    <Controls>
      <Choice label="preset" value={preset} options={PRESET_OPTIONS} onChange={p => { setPreset(p); onChange(PRESETS[p].make()) }} />
      <Btn ghost pressed={!erase} onClick={() => setErase(false)}>draw</Btn>
      <Btn ghost pressed={erase} onClick={() => setErase(true)}>erase</Btn>
      <Btn ghost onClick={() => { setPreset('blank'); onChange(PRESETS.blank.make()) }}>clear</Btn>
    </Controls>
    <Stats><Stat label="pixels" value={N * N} />{hover && <Stat label={`pixel (${hover.x}, ${hover.y})`} value={input.img[hover.y * N + hover.x].toFixed(2)} />}</Stats>
  </Figure>
}

// ---------------------------------------------------------------------------
// 2. A kernel slides: one output pixel, step by step.

const mini: CSSProperties = { display: 'grid', gridTemplateColumns: 'repeat(3, 5.4ch)', gap: 2, fontSize: '.8em', fontVariantNumeric: 'tabular-nums' }
const cell = (bg: string, fg: string = P.ink4): CSSProperties => ({ background: bg, color: fg, textAlign: 'center', padding: '.15em 0', borderRadius: 2 })
function Grid3({ title, values, color }: { title: string; values: ArrayLike<number>; color: (v: number) => string }) {
  return <div><div style={{ fontSize: '.75em', color: P.ink2, textAlign: 'center' }}>{title}</div>
    <div style={mini}>{Array.from(values, (v, i) => <span key={i} style={cell(color(v))}>{fmt(v)}</span>)}</div></div>
}
// One editable weight. Text input so "-" or "0." can be typed mid-edit.
function Weight({ index, value, range, onChange }: { index: number; value: number; range: number; onChange: (v: number) => void }) {
  const [text, setText] = useState(String(Number(value.toFixed(3))))
  useEffect(() => { setText(t => (Number(t) === value ? t : String(Number(value.toFixed(3))))) }, [value])
  return <input type="text" inputMode="decimal" value={text} aria-label={`kernel weight row ${Math.floor(index / 3) + 1} column ${index % 3 + 1}`}
    onChange={e => { setText(e.target.value); const n = Number(e.target.value); if (e.target.value.trim() !== '' && Number.isFinite(n)) onChange(n) }}
    style={{ ...cell(signed(value / range * .6)), width: '100%', minWidth: 0, boxSizing: 'border-box', font: 'inherit', border: `1px solid ${P.line}`, padding: '.1em 0' }} />
}
const op: CSSProperties = { color: P.ink1, alignSelf: 'center', paddingTop: '1em' }

function SlideFigure({ call, input }: { call: Call; input: Input }) {
  const [name, setName] = useState<KernelName>('sobelx'), [k, setK] = useState<Kernel>([...KERNELS.sobelx.k])
  const [pos, setPos] = useState({ x: 15, y: 17 }), [playing, setPlaying] = useState(false)
  const boxes = useRef<{ a: Box; b: Box } | null>(null)
  const out = useLatest(() => call<{ conv: Float32Array; computeMs: number }>({ action: 'conv', img: input.img, ver: input.ver, kernel: k }), [input.ver, k.join()])
  const terms = useLatest(() => call<{ products: Float32Array }>({ action: 'products', img: input.img, ver: input.ver, kernel: k, x: pos.x, y: pos.y }), [input.ver, k.join(), pos.x, pos.y])
  useEffect(() => {
    if (!playing) return
    const t = setInterval(() => setPos(p => { const i = (p.y * C + p.x + 1) % (C * C); return { x: i % C, y: Math.floor(i / C) } }), 45)
    return () => clearInterval(t)
  }, [playing])
  const range = out ? maxAbs(out.conv) : 1
  const ref = usePaint(canvas => {
    const { c, width, height } = surface(canvas), pad = Math.max(10, height * .07), top = pad + 14, s = Math.min(height - top - pad, (width - 2 * pad) * .4)
    const gap = width - 2 * pad - 2 * s, a = { x: pad, y: top, s }, b = { x: pad + s + gap, y: top, s }
    boxes.current = { a, b }
    const fs = Math.max(10, Math.min(13, s / 14))
    label(c, `input ${N}×${N}`, a.x + s / 2, top - 6, fs); label(c, `feature map ${C}×${C}`, b.x + s / 2, top - 6, fs)
    paintMap(c, input.img, N, a, 'gray', 1, true)
    if (out) paintMap(c, out.conv, C, b, 'signed', range, true)
    outline(c, a, N, pos.x, pos.y, 3, 3, P.vio, 2)
    outline(c, b, C, pos.x, pos.y, 1, 1, P.vio, 2)
    // a fan from the window to its output pixel
    const pa = s / N, pb = s / C, wx = a.x + (pos.x + 1.5) * pa, wy = a.y + (pos.y + 1.5) * pa, ox = b.x + (pos.x + .5) * pb, oy = b.y + (pos.y + .5) * pb
    c.strokeStyle = 'rgba(139,131,181,.45)'; c.lineWidth = 1; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(a.x + (pos.x + 3) * pa, wy); c.lineTo(ox - pb / 2, oy); c.stroke(); c.setLineDash([])
    void wx
    label(c, '3×3 kernel', a.x + s + gap / 2, a.y + s / 2 - 8, fs, P.vio, 'center', gap - 8); arrow(c, a.x + s + 8, b.x - 8, a.y + s / 2)
  })
  const pick = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    const bs = boxes.current; if (!bs) return
    const p = cellAt(e, bs.b, C) ?? (() => { const q = cellAt(e, bs.a, N); return q && { x: Math.max(0, Math.min(C - 1, q.x - 1)), y: Math.max(0, Math.min(C - 1, q.y - 1)) } })()
    if (p) { setPlaying(false); setPos(p) }
  }
  const win = Array.from({ length: 9 }, (_, j) => input.img[(pos.y + Math.floor(j / 3)) * N + pos.x + (j % 3)])
  const sum = out?.conv[pos.y * C + pos.x]
  const kr = maxAbs(k)
  return <Figure caption={<>Hover or tap either image to move the window. Output pixel ({pos.x}, {pos.y}) is the sum of the nine products below. Green is positive, red negative.</>}>
    <Stage aspect="16 / 8">
      <canvas ref={ref} role="img" aria-label="The input image with a 3 by 3 window, and the feature map the kernel produces, with the matching output pixel outlined" style={{ cursor: 'crosshair' }} onPointerMove={pick} onPointerDown={pick} />
    </Stage>
    <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', alignItems: 'flex-start', gap: '.6em 1.2ch', marginTop: '.9em' }}>
      <Grid3 title="window" values={win} color={v => gray(.05 + v * .45)} />
      <span style={op}>×</span>
      <div><div style={{ fontSize: '.75em', color: P.ink2, textAlign: 'center' }}>kernel (edit me)</div>
        <div style={mini}>{k.map((v, i) => <Weight key={i} index={i} value={v} range={kr} onChange={n => { const next = [...k]; next[i] = n; setK(next); setName('custom') }} />)}</div></div>
      <span style={op}>=</span>
      <Grid3 title="products" values={terms?.products ?? new Float32Array(9)} color={v => signed(v / Math.max(1e-6, range) * 2)} />
      <span style={op}>→</span>
      <div><div style={{ fontSize: '.75em', color: P.ink2, textAlign: 'center' }}>sum</div>
        <div style={{ ...cell(signed((sum ?? 0) / range)), padding: '.15em 1ch', fontSize: '.8em', fontWeight: 600 }}>{sum === undefined ? '…' : fmt(sum)}</div></div>
    </div>
    <Controls>
      <Choice label="kernel" value={name} options={[...KERNEL_OPTIONS, ['custom', 'custom'] as const]} onChange={n => { setName(n); if (n !== 'custom') setK([...KERNELS[n].k]) }} />
      <Btn onClick={() => setPlaying(p => !p)} pressed={playing}>{playing ? 'pause' : 'slide'}</Btn>
    </Controls>
    <Stats><Stat label="output" value={`${C}×${C}`} /><Stat label="weights" value={9} /><Stat label="Bend" value={out ? `${out.computeMs.toFixed(1)} ms` : '…'} /></Stats>
  </Figure>
}

// ---------------------------------------------------------------------------
// 3. ReLU and max-pooling.

function PipeFigure({ call, input }: { call: Call; input: Input }) {
  const [name, setName] = useState<keyof typeof KERNELS>('sobelx'), [bias, setBias] = useState(0), [at, setAt] = useState({ x: 8, y: 8 })
  const boxes = useRef<Box[]>([])
  const out = useLatest(() => call<{ conv: Float32Array; relu: Float32Array; pool: Float32Array }>({ action: 'conv', img: input.img, ver: input.ver, kernel: KERNELS[name].k, bias }), [input.ver, name, bias])
  const range = out ? maxAbs(out.conv) : 1
  const ref = usePaint(canvas => {
    const { c, width, height } = surface(canvas), pad = 8, aw = Math.max(14, width * .035), fs = Math.max(10, Math.min(13, width / 40))
    const s = Math.min((width - 2 * pad - 3 * aw) / 4, height - 2 * fs - 22), top = (height - s) / 2 + fs * .2
    const bx = (i: number): Box => ({ x: pad + i * (s + aw) + (width - 2 * pad - 4 * s - 3 * aw) / 2, y: top, s })
    boxes.current = [0, 1, 2, 3].map(bx)
    const names = ['input', 'conv', 'ReLU', 'max-pool'], sizes = [N, C, C, Q]
    boxes.current.forEach((b, i) => { label(c, names[i], b.x + s / 2, top - 6, fs, P.ink3, 'center', s + aw); label(c, `${sizes[i]}×${sizes[i]}`, b.x + s / 2, top + s + fs + 4, fs, P.ink1) })
    for (let i = 0; i < 3; i++) arrow(c, boxes.current[i].x + s + 3, boxes.current[i + 1].x - 3, top + s / 2)
    paintMap(c, input.img, N, boxes.current[0], 'gray')
    if (out) { paintMap(c, out.conv, C, boxes.current[1], 'signed', range); paintMap(c, out.relu, C, boxes.current[2], 'gray', range); paintMap(c, out.pool, Q, boxes.current[3], 'gray', range) }
    // one pooled cell and everything that fed it
    outline(c, boxes.current[0], N, 2 * at.x, 2 * at.y, 4, 4, P.ora, 1.5)
    outline(c, boxes.current[1], C, 2 * at.x, 2 * at.y, 2, 2, P.ora, 1.5)
    outline(c, boxes.current[2], C, 2 * at.x, 2 * at.y, 2, 2, P.ora, 1.5)
    outline(c, boxes.current[3], Q, at.x, at.y, 1, 1, P.ora, 1.5)
  })
  const pick = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    const bs = boxes.current; if (bs.length < 4) return
    const p = cellAt(e, bs[3], Q), q = cellAt(e, bs[2], C) ?? cellAt(e, bs[1], C), r = cellAt(e, bs[0], N)
    const hit = p ?? (q && { x: q.x >> 1, y: q.y >> 1 }) ?? (r && { x: Math.min(Q - 1, Math.max(0, (r.x - 1) >> 1)), y: Math.min(Q - 1, Math.max(0, (r.y - 1) >> 1)) })
    if (hit) setAt(hit)
  }
  const four = out ? [0, 1, C, C + 1].map(o => out.relu[2 * at.y * C + 2 * at.x + o]) : []
  return <Figure caption={<>Hover any panel. The orange 2×2 block of the ReLU map collapses into one pooled pixel, which therefore sees a 4×4 patch of the input.</>}>
    <Stage aspect="16 / 6">
      <canvas ref={ref} role="img" aria-label="Pipeline: input 28 by 28, convolution 26 by 26, ReLU 26 by 26, max-pooling 13 by 13, with one pooled pixel and its sources outlined" style={{ cursor: 'crosshair' }} onPointerMove={pick} onPointerDown={pick} />
    </Stage>
    <Controls>
      <Choice label="kernel" value={name} options={KERNEL_OPTIONS} onChange={setName} />
      <Range label="bias" value={bias} min={-2} max={2} step={0.1} onChange={setBias} format={v => v.toFixed(1)} />
    </Controls>
    <Stats>{out && <Stat label={`pool (${at.x}, ${at.y})`} value={`max(${four.map(v => fmt(v)).join(', ')}) = ${fmt(out.pool[at.y * Q + at.x])}`} />}</Stats>
  </Figure>
}

// ---------------------------------------------------------------------------
// 4. A layer: eight kernels at once, then corners from edges.

const L1_SHORT = ['left', 'right', 'top', 'bottom', '↘ diag', '↙ diag', 'outline', 'blur'], L2_SHORT = ['┌', '┐', '└', '┘']
function LayerFigure({ call, input }: { call: Call; input: Input }) {
  const out = useLatest(() => call<{ first: Float32Array; second: Float32Array; computeMs: number }>({ action: 'layer', img: input.img, ver: input.ver, layer1: LAYER1.map(l => l.k), layer2: LAYER2.map(l => ({ ks: l.ks, bias: l.bias })) }), [input.ver])
  const ref = usePaint(canvas => {
    const { c, width, height } = surface(canvas), pad = 10, fs = Math.max(10, Math.min(13, width / 42)), lh = fs + 8, head = fs + 14
    const g = Math.max(4, width * .012), split = Math.max(18, width * .05)
    const s = Math.min((width - 2 * pad - split - 4 * g) / 6, (height - head - pad - 2 * lh - g) / 2)
    const total = 6 * s + 4 * g + split, x0 = (width - total) / 2, y0 = head + (height - head - pad - 2 * (s + lh) - g) / 2
    label(c, `layer 1 · 8 maps ${Q}×${Q}`, x0 + (4 * s + 3 * g) / 2, y0 - 8, fs, P.ink3, 'center', 4 * s + 3 * g)
    label(c, `layer 2 · corners ${R}×${R}`, x0 + 4 * s + 3 * g + split + s + g / 2, y0 - 8, fs, P.ink3, 'center', 2 * s + g + split * .6)
    const r1 = out ? maxAbs(out.first) : 1, r2 = out ? maxAbs(out.second) : 1
    for (let i = 0; i < 8; i++) {
      const b = { x: x0 + (i % 4) * (s + g), y: y0 + Math.floor(i / 4) * (s + lh + g), s }
      if (out) paintMap(c, out.first.subarray(i * Q * Q, (i + 1) * Q * Q), Q, b, 'gray', r1); else { c.fillStyle = P.paper; c.fillRect(b.x, b.y, s, s) }
      label(c, L1_SHORT[i], b.x + s / 2, b.y + s + fs + 3, fs, P.ink2, 'center', s + g)
    }
    for (let i = 0; i < 4; i++) {
      const b = { x: x0 + 4 * s + 3 * g + split + (i % 2) * (s + g), y: y0 + Math.floor(i / 2) * (s + lh + g), s }
      if (out) paintMap(c, out.second.subarray(i * R * R, (i + 1) * R * R), R, b, 'gray', r2); else { c.fillStyle = P.paper; c.fillRect(b.x, b.y, s, s) }
      label(c, L2_SHORT[i], b.x + s / 2, b.y + s + fs + 3, fs, P.vio)
    }
    arrow(c, x0 + 4 * s + 3 * g + 4, x0 + 4 * s + 3 * g + split - 4, y0 + s + lh / 2)
  })
  return <Figure caption={<>Layer 1 turns your image into eight pooled edge maps. Layer 2 reads the four edge maps at once; each of its filters lights up only where two edges meet. Draw a box (or pick the preset) to see the four corners separate.</>}>
    <Stage aspect="16 / 8.5">
      <canvas ref={ref} role="img" aria-label="Eight layer-one feature maps for edges, diagonals, outline and blur, and four layer-two maps that detect corners" />
    </Stage>
    <Stats><Stat label="layer 1 weights" value={`8 × 9`} /><Stat label="layer 2 weights" value={`4 × 4 × 9`} /><Stat label="Bend" value={out ? `${out.computeMs.toFixed(1)} ms` : '…'} /></Stats>
  </Figure>
}

// ---------------------------------------------------------------------------
// 5. Learning a kernel by gradient descent.

const TARGETS = ['sobelx', 'sobely', 'sharpen', 'emboss', 'outline', 'blur'] as const
type Target = typeof TARGETS[number]
const MAX_STEPS = 300
interface FitState { kernel: Float32Array; losses: number[]; step: number; learned?: Float32Array; wanted?: Float32Array; noise?: Float32Array; ms?: number }

function TrainFigure({ call, input }: { call: Call; input: Input }) {
  const [target, setTarget] = useState<Target>('sobelx'), [rate, setRate] = useState(0.05), [running, setRunning] = useState(false)
  const [st, setSt] = useState<FitState>({ kernel: new Float32Array(9), losses: [], step: 0 })
  const live = useRef({ running, rate, input, gen: 0 }); live.current = { ...live.current, running, rate, input }
  const reset = useCallback(async (t: Target) => {
    const gen = ++live.current.gen
    const r = await call<{ kernel: Float32Array; noise: Float32Array }>({ action: 'fit-reset', target: KERNELS[t].k, seed: 7 })
    if (gen !== live.current.gen) return
    // one zero-step fit reports both kernels on the reader's image
    const f = await call<{ kernel: Float32Array; learned: Float32Array; wanted: Float32Array }>({ action: 'fit', steps: 0, rate: 0, img: live.current.input.img, ver: live.current.input.ver })
    if (gen === live.current.gen) setSt({ kernel: r.kernel, losses: [], step: 0, noise: r.noise, learned: f.learned, wanted: f.wanted })
  }, [call])
  useEffect(() => { setRunning(false); reset(target) }, [target, reset])
  useEffect(() => {
    if (!running) return
    let stop = false
    const gen = live.current.gen
    const tick = async () => {
      if (stop || !live.current.running || gen !== live.current.gen) return
      const { input: i, rate: r } = live.current
      const f = await call<{ kernel: Float32Array; losses: Float32Array; step: number; learned: Float32Array; wanted: Float32Array; computeMs: number }>({ action: 'fit', steps: 2, rate: r, img: i.img, ver: i.ver })
      if (stop || gen !== live.current.gen) return
      const last = f.losses[f.losses.length - 1]
      setSt(s => ({ ...s, kernel: f.kernel, losses: [...s.losses, ...f.losses], step: f.step, learned: f.learned, wanted: f.wanted, ms: f.computeMs }))
      if (f.step >= MAX_STEPS || !Number.isFinite(last) || last > 1e4) { setRunning(false); return }
      setTimeout(() => requestAnimationFrame(tick), 30)
    }
    tick()
    return () => { stop = true }
  }, [running, call])
  // refresh both feature maps when the reader redraws while paused
  useEffect(() => {
    if (running || !st.noise) return
    call<{ learned: Float32Array; wanted: Float32Array }>({ action: 'fit', steps: 0, rate: 0, img: input.img, ver: input.ver }).then(f => setSt(s => ({ ...s, learned: f.learned, wanted: f.wanted })), () => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [input.ver])

  const want = KERNELS[target].k, last = st.losses[st.losses.length - 1]
  const diverged = last !== undefined && (!Number.isFinite(last) || last > 1e4)
  const ref = usePaint(canvas => {
    const { c, width, height } = surface(canvas), pad = 10, fs = Math.max(10, Math.min(13, width / 42)), g = Math.max(8, width * .025)
    const topH = height * .56, s = Math.min((width - 2 * pad - 4 * g) / 5, topH - 2 * fs - 14), y = fs + 12
    const x0 = (width - 5 * s - 4 * g) / 2, bx = (i: number): Box => ({ x: x0 + i * (s + g), y, s })
    const titles = ['noise', 'learned', 'target', 'learned ∗ you', 'target ∗ you']
    titles.forEach((t, i) => label(c, t, bx(i).x + s / 2, y - 6, fs, i === 1 || i === 3 ? P.vio : P.ink3, 'center', s + g * .8))
    if (st.noise) paintMap(c, st.noise, TN, bx(0), 'signed', 1)
    const kr = Math.max(maxAbs(want), 1)
    const drawK = (k: ArrayLike<number>, b: Box) => {
      paintMap(c, k, 3, b, 'signed', kr)
      for (let j = 0; j < 9; j++) label(c, fmt(k[j], 1), b.x + (j % 3 + .5) * s / 3, b.y + (Math.floor(j / 3) + .62) * s / 3, Math.min(13, s / 7.5), P.ink4, 'center', s / 3 - 3)
    }
    drawK(st.kernel, bx(1)); drawK(want, bx(2))
    const fr = st.wanted ? maxAbs(st.wanted) : 1
    if (st.learned) paintMap(c, st.learned, C, bx(3), 'signed', fr)
    if (st.wanted) paintMap(c, st.wanted, C, bx(4), 'signed', fr)
    // loss curve, log scale
    const lx = x0 + 34, lw = 5 * s + 4 * g - 34, ly = y + s + fs * 2 + 14, lh = height - ly - fs - 10
    c.strokeStyle = P.bar; c.lineWidth = 1; c.beginPath(); c.moveTo(lx, ly); c.lineTo(lx, ly + lh); c.lineTo(lx + lw, ly + lh); c.stroke()
    const lo = -6, hi = 2, yOf = (v: number) => ly + lh * (1 - (Math.log10(Math.max(v, 1e-12)) - lo) / (hi - lo))
    for (let e = lo; e <= hi; e += 2) { const yy = yOf(10 ** e); label(c, `1e${e}`, lx - 4, yy + 3, fs * .85, P.ink1, 'right'); c.strokeStyle = P.line; c.beginPath(); c.moveTo(lx + 1, yy); c.lineTo(lx + lw, yy); c.stroke() }
    label(c, 'loss', lx + 4, ly + fs, fs, P.ink2, 'left'); label(c, `step ${MAX_STEPS}`, lx + lw, ly + lh + fs + 4, fs * .85, P.ink1, 'right')
    if (st.losses.length) {
      c.save(); c.beginPath(); c.rect(lx, ly - 2, lw, lh + 2); c.clip()
      c.strokeStyle = diverged ? P.red : P.vio; c.lineWidth = 2; c.beginPath()
      st.losses.forEach((v, i) => { const xx = lx + lw * i / MAX_STEPS, yy = Number.isFinite(v) ? yOf(v) : ly; if (i) c.lineTo(xx, yy); else c.moveTo(xx, yy) })
      c.stroke(); c.restore()
    }
  })
  return <Figure caption={<>Gradient descent on the nine weights, starting from zero. It trains on random noise, then the learned kernel is applied to your drawing beside the real filter. Push the rate past about 3 and it diverges.</>}>
    <Stage aspect="16 / 9.5">
      <canvas ref={ref} role="img" aria-label={`Training a 3 by 3 kernel by gradient descent: the noise image, the learned and target kernels, both applied to your image, and the loss curve. Step ${st.step}.`} />
    </Stage>
    <Controls>
      <Choice label="target" value={target} options={TARGETS.map(t => [t, KERNELS[t].label] as const)} onChange={setTarget} />
      <Range label="rate" value={rate} min={0.01} max={3.5} step={0.01} onChange={setRate} format={v => v.toFixed(2)} />
      <Btn onClick={() => { if (st.step >= MAX_STEPS || diverged) { reset(target).then(() => setRunning(true)) } else setRunning(r => !r) }} pressed={running}>{running ? 'pause' : 'train'}</Btn>
      <Btn ghost onClick={() => { setRunning(false); reset(target) }}>reset</Btn>
    </Controls>
    <Stats><Stat label="step" value={st.step} /><Stat label="loss" value={last === undefined ? '—' : diverged ? 'diverged' : last.toExponential(2)} /><Stat label="Bend" value={st.ms ? `${st.ms.toFixed(1)} ms / 2 steps` : '…'} /></Stats>
  </Figure>
}

// ---------------------------------------------------------------------------

const TAPS = `
# One channel's 3x3 window at output (x, y): sum of input * weight.
# A kernel is its nine weights in reading order: weight j sits at row j / 3,
# column j % 3 of the window.
def taps(k: List<&2, F32>, +j: U32, +img: Img, +d: Nat, +w: U32, +x: U32, +y: U32, acc: F32) -> F32:
  match k:
    case Nil{}:
      acc
    case Con{+c, rest}:
      +v = at(d, img, ((y + j / 3) * w + x + j % 3 : U32))
      taps(rest, (j + 1 : U32), img, d, w, x, y, (acc + v * c : F32))
`
const RELU_POOL = `
def relu(m: Img) -> Img:
  match m:
    case Px{+v}:
      Px{F32.max(v, 0.0)}
    case Fork{lo, hi}:
      a b = relu(lo) relu(hi)
      Fork{a, b}

def pool_px(valid: Bool, +i: U32, +o: U32, +m: Img, +d: Nat, +w: U32) -> Img:
  match valid:
    case False{}:
      Px{0.0}
    case True{}:
      +x = (2 * (i % o) : U32)
      +y = (2 * (i / o) : U32)
      +top = F32.max(at(d, m, (y * w + x : U32)), at(d, m, (y * w + x + 1 : U32)))
      +low = F32.max(at(d, m, ((y + 1) * w + x : U32)), at(d, m, ((y + 1) * w + x + 1 : U32)))
      Px{F32.max(top, low)}
`
const WINDOW = `
# Sums the windows of every input channel, each with its own kernel.
def window(chans: List<&2, Img>, ks: List<&2, List<&2, F32>>, +d: Nat, +w: U32, +x: U32, +y: U32, acc: F32) -> F32:
  match chans:
    case Nil{}:
      acc
    case Con{c, cs}:
      match ks:
        case Nil{}:
          acc
        case Con{k, rest}:
          window(cs, rest, d, w, x, y, taps(k, 0, c, d, w, x, y, acc))
`
const GRAD = `
# dLoss/dw_j (up to 2/N): every error times the input pixel that met w_j.
def gsum(depth: Nat, +offset: U32, +err: Img, +img: Img, +d: Nat, +w: U32, +o: U32, +j: U32) -> F32:
  match depth:
    case 0n:
      +x = (offset % o : U32)
      +y = (offset / o : U32)
      (first(err) * at(d, img, ((y + j / 3) * w + x + j % 3 : U32)) : F32)
    case 1n+ +p:
      a b = gsum(p, offset, side(err, True{}), img, d, w, o, j) gsum(p, (offset + U32.shln(1, p) : U32), side(err, False{}), img, d, w, o, j)
      (a + b : F32)

# inside fit: one step of gradient descent
+err = diff(od, conv(od, 0, Con{img, Nil{}}, Con{k, Nil{}}, d, w, 0.0), target)
+loss = (sumsq(err) / n : F32)
next = update(k, 0, err, img, d, w, o, od, (rate * 2.0 / n : F32))
`
const CONV = `
# A valid 3x3 convolution: a w x w input becomes a (w-2) x (w-2) map with
# 2^depth leaves. The two halves of the output are computed in parallel.
def conv(depth: Nat, +offset: U32, +chans: List<&2, Img>, +ks: List<&2, List<&2, F32>>, +d: Nat, +w: U32, +bias: F32) -> Img:
  match depth:
    case 0n:
      +o = (w - 2 : U32)
      conv_px((offset < o * o : U32), offset, o, chans, ks, d, w, bias)
    case 1n+ +p:
      a b = conv(p, offset, chans, ks, d, w, bias) conv(p, (offset + U32.shln(1, p) : U32), chans, ks, d, w, bias)
      Fork{a, b}
`

const SPLIT = [1, 4, 16, 64, 1024].map(cores => ({ cores, rounds: Math.ceil(C * C / cores) }))

export default function Cnn() {
  const { call, error } = useWorker(() => new Worker(new URL('./convolution.worker.ts', import.meta.url), { type: 'module' }))
  const [input, setInput] = useState<Input>(() => ({ img: PRESETS.digit.make(), ver: 1 }))
  const change = useCallback((img: Float32Array) => setInput(i => ({ img, ver: i.ver + 1 })), [])
  return (
    <DocPage id="cnn" kicker="machine learning · essay" title="Convolutional networks" sub={<>how a network learns to <b>see</b>, one 3×3 window at a time</>}
      tag={<><b>kernels</b> · <b>feature maps</b> · <b>pooling</b> · <b>learning</b></>}>
      <p>A plain neural network looks at an image as one long list of numbers. It
        has no idea that pixel 30 sits right below pixel 2. A convolutional network
        keeps the grid, and asks the same small question everywhere: what does
        this neighbourhood look like?</p>
      <p>That one idea, a tiny pattern detector slid across the image, is behind
        nearly every vision model of the last decade. Below, every feature map is
        computed by <a href="https://bend-lang.com">Bend</a>, in your browser.</p>
      {error && <p className="note" role="alert">Bend worker error: {error}</p>}

      <Section n={1} title={<>An image is a <b>GRID</b> of numbers.</>}>
        <p>A grayscale image is a table of brightness values. Ours is {N}×{N},
          the size of an MNIST handwritten digit: {N * N} numbers between 0 and 1.
          There is nothing else in it. Every edge and stroke you see has to be
          recovered from these numbers.</p>
        <InputFigure input={input} onChange={change} />
        <p>Everything below reacts to this image. Draw something of your own.</p>
      </Section>

      <Section n={2} title={<>Kernels <b>SLIDE</b>.</>}>
        <p>A <b>kernel</b> is a 3×3 grid of weights. Lay it over a 3×3 window of
          the image, multiply each pixel by the weight on top of it, and add the
          nine products. That sum is one pixel of the output. Slide the window one
          step and repeat. The result is a new image, a <b>feature map</b>.</p>
        <p className="lead">out(x, y) = Σ k(i, j) · in(x+j, y+i)</p>
        <SlideFigure call={call} input={input} />
        <p>Pick <em>Sobel x</em>: its left column is negative and its right column
          positive, so it answers "how much brighter is the right side than the
          left?". It is zero on flat paper and flat ink, and strong on vertical
          edges, green where the stroke starts and red where it ends.
          A blur averages, sharpen exaggerates the centre against its neighbours,
          and identity copies the input.</p>
        <p>Two details. The window must fit inside the image, so a 28-wide input
          gives a 26-wide map; real networks often pad the border with zeros to
          keep the size. And strictly this is <i>cross-correlation</i>: textbook
          convolution flips the kernel first. Deep-learning libraries skip the flip,
          since a learned kernel can just as well learn the flipped weights.</p>
        <p>The same nine weights are reused at all 676 positions. A dense layer
          mapping 784 inputs to 676 outputs would need 529,984 weights; this
          needs nine, and it finds an edge wherever the edge is.</p>
        <Code file="demos/convolution/conv.bend">{TAPS}</Code>
        <p>Bend has no loops: <code>taps</code> walks the kernel's list of nine
          weights, and each step reads one pixel and multiplies it by one weight. The <code>+</code> marks
          the image and kernel as reusable, since every window reads them.</p>
      </Section>

      <Section n={3} title={<>Clip it, then <b>POOL</b> it.</>}>
        <p>Convolution is linear: scale the input and the output scales with it.
          Stack two convolutions and you get another convolution (two 3×3 kernels
          compose into one 5×5). Depth buys nothing until something bends the line.</p>
        <p>The <b>ReLU</b> does it with the simplest nonlinearity there is: keep
          positives, replace negatives with zero. Now a map says "there is an edge
          here", not "there is minus an edge here".</p>
        <p>Then <b>max-pooling</b> keeps the largest value of each 2×2 block and
          halves the resolution: 26×26 becomes 13×13. A small shift of the input
          barely changes the pooled map, and every pixel of the next layer looks at
          twice as much of the image.</p>
        <PipeFigure call={call} input={input} />
        <p>Try a negative bias: the conv output is shifted before the ReLU, so only
          strong responses survive. That threshold is how a neuron decides what
          counts as "enough edge".</p>
        <Code file="demos/convolution/conv.bend">{RELU_POOL}</Code>
      </Section>

      <Section n={4} title={<>Many kernels, many <b>MAPS</b>.</>}>
        <p>One kernel finds one kind of pattern. A convolutional <b>layer</b> runs
          many at once and stacks their outputs into channels. Here, eight kernels
          look for edges facing each way, two diagonals, outlines and blobs.</p>
        <p>The second layer is where it gets interesting. Its kernels are 3×3×4:
          each reads a window from <i>four</i> maps of the first layer at once,
          the four edge directions. A top-left corner is where a top edge heads
          right and a left edge heads down. So the kernel puts two weights of ½ on
          each of those arms, and a bias of −1. One edge alone sums to 1 and
          the ReLU erases it; both together sum to 2 and survive. Each pixel of
          this layer sees an 8×8 patch of the original image.</p>
        <LayerFigure call={call} input={input} />
        <p>Real networks repeat this a dozen times or more: edges become corners
          and curves, curves become eyes and wheels, those become faces and cars.
          We picked these kernels by hand. A trained network finds its own, and they
          often look much like the ones above.</p>
        <Code file="demos/convolution/conv.bend">{WINDOW}</Code>
      </Section>

      <Section n={5} title={<>Kernels are <b>LEARNED</b>.</>}>
        <p>Nobody writes Sobel into a modern network. The kernels are weights,
          and they are found the way every <a href="#neural-networks">neural network</a> finds
          its weights: by gradient descent on a loss.</p>
        <p>To watch it happen on nine weights, we hide a target kernel, show the
          network only its outputs, and start from zero. The loss is the mean
          squared difference between our feature map and the target's. Its gradient
          for weight j is itself a convolution: each output error times the input
          pixel that met weight j. Backpropagation through a convolution is
          more convolution.</p>
        <TrainFigure call={call} input={input} />
        <p>Why train on noise? On a smooth drawing neighbouring pixels move
          together, so the nine weights are tangled and plain gradient descent crawls.
          On random noise every pixel is independent, the loss surface is a round bowl,
          and each step takes the same fraction off the error: a straight line on a
          log scale. The kernel learned there still works on your drawing.</p>
        <Code file="demos/convolution/conv.bend">{GRAD}</Code>
      </Section>

      <Section n={6} title={<>Every pixel is <b>INDEPENDENT</b>.</>}>
        <p>Look at the formula again: output pixel (x, y) needs the input and the
          kernel, and nothing else. No output pixel waits on another. That is why
          GPUs eat convolutions, and why they fit Bend so well.</p>
        <p>Each feature map here is a binary tree of pixels. <code>conv</code> splits
          the output range in two and computes both halves with a <b>parallel
          call</b>, <code>a b = f(x) g(y)</code>. Ten splits give 1024 independent
          leaves, enough for the 676 pixels of a 26×26 map.
          <code> relu</code>, <code>pool</code>, the noise and the gradient sums split the same way.</p>
        <Code file="demos/convolution/conv.bend">{CONV}</Code>
        {/* --bar is not defined by the shared theme yet; set it here so the grey bars show */}
        <div style={{ ['--bar' as string]: P.bar }}><Bars title="window sums per core, one 26×26 map" data={SPLIT.map(({ cores, rounds }) => ({ label: `${cores} core${cores > 1 ? 's' : ''}`, value: rounds, hi: cores === 1024 }))}
          caption="Ideal counts from the split tree. On this page Bend runs on its JavaScript target, which is sequential; compiled natively, the same code runs on every CPU core, or on the GPU with f!(x)." /></div>
        <p>Both halves are the same size by construction, which is what Bend's
          scheduler asks for: it hands each task to a core once and never moves it.</p>
      </Section>

      <Section n={7} title={<>Where this <b>GOES</b>.</>}>
        <p>Everything a large vision model does is made of the pieces on this
          page: kernels, ReLUs, pooling, many channels, and gradient descent to set
          the weights. Only the counts change.</p>
        <p>My undergraduate thesis used CNNs like these to detect deepfaked faces:
          small, learned kernels tuned to the artifacts that face-swapping leaves behind.
          That story continues in <a href="#deepfakes">Detecting deepfakes</a>.</p>
      </Section>
    </DocPage>
  )
}
