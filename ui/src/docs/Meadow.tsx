import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import { Bars, Btn, Choice, Code, Controls, DocPage, Figure, Range, Section, Stage, Stat, Stats } from '@/components/Doc'
import { useWorker } from '@/lib/use-lab'
import { surface, usePaint } from '@/lib/lab-canvas'
import { P, mono } from '@/lib/palette'
import { ACTIVITY, SIDE, drawField, drawFlock, drawHistory, drawTraits, drawTrace, square, type Frame, type Sample, type Tint, type Trace } from './meadow-draw'

type Call = <T>(body: Record<string, unknown>) => Promise<T>

// Pause any figure that is scrolled out of view.
function useVisible<T extends Element>() {
  const ref = useRef<T>(null), [visible, setVisible] = useState(false)
  useEffect(() => {
    const el = ref.current; if (!el) return
    const o = new IntersectionObserver(([e]) => setVisible(e.isIntersecting), { threshold: .05 }); o.observe(el); return () => o.disconnect()
  }, [])
  return [ref, visible] as const
}

// One Bend World in the worker: create it when `spec` changes, then step it
// once per animation frame while playing and visible.
function useSim(call: Call, name: string, spec: Record<string, unknown>, ticks: number, running: boolean, opts: { extra?: boolean; init?: Record<string, unknown>; onFrame?: (f: Frame, fresh: boolean) => void } = {}) {
  const [frame, setFrame] = useState<Frame | null>(null), epoch = useRef(0), latest = useRef(opts); latest.current = opts
  const key = JSON.stringify(spec), tickRef = useRef(ticks); tickRef.current = ticks
  const apply = useCallback((f: Frame, e: number, fresh = false) => { if (e === epoch.current) { setFrame(f); latest.current.onFrame?.(f, fresh) } }, [])
  useEffect(() => {
    const e = ++epoch.current
    call<Frame>({ op: 'create', world: name, ...JSON.parse(key), ...latest.current.init }).then(f => apply(f, e, true)).catch(() => {})
  }, [call, name, key, apply])
  useEffect(() => {
    if (!running) return
    let alive = true, e = epoch.current
    const loop = async () => {
      while (alive) {
        if (e !== epoch.current) e = epoch.current
        const f = await call<Frame>({ op: 'step', world: name, ticks: tickRef.current, extra: latest.current.extra }).catch(() => null)
        if (!alive || !f) break
        apply(f, e)
        await new Promise(r => requestAnimationFrame(r))
      }
    }
    loop(); return () => { alive = false }
  }, [call, name, running, apply])
  const send = useCallback((body: Record<string, unknown>) => { const e = epoch.current; return call<Frame>({ world: name, ...body }).then(f => apply(f, e)).catch(() => {}) }, [call, name, apply])
  const restart = useCallback(() => { const e = ++epoch.current; call<Frame>({ op: 'create', world: name, ...JSON.parse(key), ...latest.current.init }).then(f => apply(f, e, true)).catch(() => {}) }, [call, name, key, apply])
  return { frame, send, restart }
}

// Drag on a meadow canvas to graze it bare (a 3-cell disk), one request at a time.
function useBrush(send: (b: Record<string, unknown>) => Promise<unknown>, area: (w: number, h: number) => { x: number; y: number; size: number }) {
  const busy = useRef(false), down = useRef(false)
  const paint = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect(), a = area(r.width, r.height)
    const x = (e.clientX - r.left - a.x) / a.size * SIDE, z = (e.clientY - r.top - a.y) / a.size * SIDE
    if (busy.current || x < -2 || z < -2 || x > SIDE + 2 || z > SIDE + 2) return
    busy.current = true; send({ op: 'paint', x, z, radius: 3, amount: -4 }).finally(() => { busy.current = false })
  }
  return {
    onPointerDown: (e: ReactPointerEvent<HTMLCanvasElement>) => { down.current = true; e.currentTarget.setPointerCapture(e.pointerId); paint(e) },
    onPointerMove: (e: ReactPointerEvent<HTMLCanvasElement>) => { if (down.current) paint(e) },
    onPointerUp: () => { down.current = false }, onPointerCancel: () => { down.current = false },
    style: { touchAction: 'none', cursor: 'crosshair' } as const,
  }
}

const pct = (food: number) => `${(food / (SIDE * SIDE * 4) * 100).toFixed(0)}%`
const fmtTime = (s: number) => s < 600 ? `${s.toFixed(0)} s` : `${Math.floor(s / 60)} min`
const hex = (n: number) => (n >>> 0).toString(16).padStart(8, '0')

// 1. Grass only.
function GrassFigure({ call }: { call: Call }) {
  const [box, visible] = useVisible<HTMLDivElement>(), [regrow, setRegrow] = useState(.015), [playing, setPlaying] = useState(true)
  const sim = useSim(call, 'grass', { seed: 7, population: 0, food: 4, metab: 1 }, 6, playing && visible, { init: { regrow } })
  useEffect(() => { sim.send({ op: 'tune', regrow, metab: 1 }) }, [regrow, sim.send])
  const area = (w: number, h: number) => square(w, h)
  const brush = useBrush(sim.send, area)
  const ref = usePaint(canvas => { const { c, width, height } = surface(canvas), a = area(width, height); if (sim.frame) drawField(c, sim.frame.grass, a.x, a.y, a.size) })
  const f = sim.frame
  return <Figure caption={<>The field starts nearly bare. Fertile patches fill first, then everything saturates at 4. Drag across it to graze it bare again. Each frame Bend advances 6 ticks (0.6 simulated seconds).</>}>
    <div ref={box} style={{ maxWidth: '26em', margin: '0 auto' }}>
      <Stage aspect="1 / 1"><canvas ref={ref} {...brush} role="img" aria-label="A 48 by 48 grass field; darker green cells hold more biomass. Drag to graze cells bare." /></Stage>
    </div>
    <Controls>
      <Btn onClick={() => setPlaying(p => !p)}>{playing ? 'Pause' : 'Play'}</Btn>
      <Btn ghost onClick={() => sim.restart()}>Reset field</Btn>
      <Range label="regrowth / s" value={regrow} min={0} max={.15} step={.005} onChange={setRegrow} format={v => v.toFixed(3)} />
    </Controls>
    <Stats><Stat label="time" value={f && fmtTime(f.time)} /><Stat label="food" value={f && pct(f.food)} /><Stat label="full in" value={regrow ? `~${Math.round(4 / (regrow * .85))} s` : 'never'} /></Stats>
  </Figure>
}

// 2. One sheep and its energy trace.
function LoneFigure({ call }: { call: Call }) {
  const [box, visible] = useVisible<HTMLDivElement>(), [seed, setSeed] = useState(1), [playing, setPlaying] = useState(true), trace = useRef<Trace[]>([])
  const onFrame = useCallback((f: Frame, fresh: boolean) => {
    if (fresh) trace.current = []
    const s = f.sheep
    if (s.length) { const last = trace.current.at(-1); if (!last || f.time > last.t) trace.current.push({ t: f.time, e: s[3], act: s[7] & 3 }); if (trace.current.length > 4000) trace.current.splice(0, 1000) }
  }, [])
  const sim = useSim(call, 'lone', { lone: true, seed, food: 6, regrow: .015 }, 3, playing && visible, { onFrame })
  useEffect(() => { if (sim.frame && sim.frame.tick > 0 && sim.frame.population === 0) setSeed(s => s + 1) }, [sim.frame])
  const area = (w: number, h: number) => { const wide = w > h * 1.5; return wide ? { x: 0, y: 0, size: h } : { x: Math.floor((w - h * .55) / 2), y: 0, size: Math.floor(h * .55) } }
  const brush = useBrush(sim.send, area)
  const ref = usePaint(canvas => {
    const { c, width, height } = surface(canvas), a = area(width, height), f = sim.frame
    if (!f) return
    drawField(c, f.grass, a.x, a.y, a.size); drawFlock(c, f.sheep, a.x, a.y, a.size, 'activity', 2.2)
    const wide = width > height * 1.5
    if (wide) drawTrace(c, a.size + 12, 0, width - a.size - 12, height, trace.current)
    else drawTrace(c, 0, a.size + 6, width, height - a.size - 6, trace.current)
    c.font = mono(11); c.textAlign = 'left'; c.textBaseline = 'alphabetic'
    let lx = wide ? a.size + 42 : 30
    if ((wide ? width - a.size : width) > 280) for (const [label, color] of ([['wander', P.ink2], ['seek food', P.ora], ['graze', P.blu]] as const)) { c.fillStyle = color; c.fillRect(lx, wide ? 9 : a.size + 15, 10, 3); c.fillStyle = P.ink2; c.fillText(label, lx + 14, wide ? 14 : a.size + 20); lx += c.measureText(label).width + 30 }
  })
  const f = sim.frame, e = f?.sheep.length ? f.sheep[3] : null, act = f?.sheep.length ? ACTIVITY[f.sheep[7] & 3][0] : 'gone'
  return <Figure caption={<>One sheep, alone. The line is its energy: orange while it walks to food, blue while it grazes, gray while it wanders. The shaded band is the gap between 68 and 94. Drag to clear the grass around it and the walks get longer. When it dies of old age, another one arrives.</>}>
    <div ref={box}><Stage aspect="2 / 1"><canvas ref={ref} {...brush} role="img" aria-label="A single sheep on a meadow, with a chart of its energy over the last two minutes rising while it grazes and falling while it wanders." /></Stage></div>
    <Controls>
      <Btn onClick={() => setPlaying(p => !p)}>{playing ? 'Pause' : 'Play'}</Btn>
      <Btn ghost onClick={() => sim.restart()}>Regrow field</Btn>
    </Controls>
    <Stats><Stat label="time" value={f && fmtTime(f.time)} /><Stat label="energy" value={e == null ? '—' : e.toFixed(1)} /><Stat label="doing" value={act} /><Stat label="food" value={f && pct(f.food)} /></Stats>
  </Figure>
}

const SPEEDS = [['1', '1 tick'], ['5', '5 ticks'], ['20', '20 ticks'], ['60', '60 ticks']] as const
const TINTS = [['speed', 'speed'], ['metab', 'metabolism'], ['activity', 'activity']] as const

// 3. The full meadow (its state also feeds the charts of section 4).
interface Main { sim: ReturnType<typeof useSim>; history: Sample[]; start: Frame | null; seed: number; setSeed: (f: (s: number) => number) => void; population: number; setPopulation: (n: number) => void; regrow: number; setRegrow: (n: number) => void; metab: number; setMetab: (n: number) => void; speed: '1' | '5' | '20' | '60'; setSpeed: (s: '1' | '5' | '20' | '60') => void; playing: boolean; setPlaying: (f: (p: boolean) => boolean) => void }
function useMain(call: Call): Main {
  const [seed, setSeed] = useState(42), [population, setPopulation] = useState(32), [regrow, setRegrow] = useState(.015), [metab, setMetab] = useState(1)
  const [speed, setSpeed] = useState<'1' | '5' | '20' | '60'>('20'), [playing, setPlaying] = useState(false)
  const history = useRef<Sample[]>([]), [start, setStart] = useState<Frame | null>(null)
  const onFrame = useCallback((f: Frame, fresh: boolean) => {
    if (fresh) { history.current = []; setStart(f) }
    const last = history.current.at(-1)
    if (!last || f.time - last.t >= 1) history.current.push({ t: f.time, pop: f.population, food: f.food / (SIDE * SIDE * 4) * 100 })
    if (history.current.length > 7200) history.current = history.current.filter((_, i) => i % 2 === 0)
  }, [])
  // The meadow keeps running while the reader scrolls down to its charts.
  const sim = useSim(call, 'meadow', { seed, population, food: 75 }, Number(speed), playing, { extra: true, onFrame, init: { regrow, metab } })
  useEffect(() => { sim.send({ op: 'tune', regrow, metab }) }, [regrow, metab, sim.send])
  return { sim, history: history.current, start, seed, setSeed, population, setPopulation, regrow, setRegrow, metab, setMetab, speed, setSpeed, playing, setPlaying }
}

function MeadowFigure({ main }: { main: Main }) {
  const [tint, setTint] = useState<Tint>('speed'), { sim } = main, f = sim.frame
  const area = (w: number, h: number) => square(w, h)
  const brush = useBrush(sim.send, area)
  const ref = usePaint(canvas => {
    const { c, width, height } = surface(canvas), a = area(width, height); if (!f) return
    drawField(c, f.grass, a.x, a.y, a.size); drawFlock(c, f.sheep, a.x, a.y, a.size, tint)
    if (f.halted) { c.fillStyle = 'rgba(233,228,219,.85)'; c.fillRect(a.x, a.y + a.size / 2 - 22, a.size, 44); c.fillStyle = P.red; c.font = mono(13, 600); c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText('1000-sheep ceiling: paused, not culled', a.x + a.size / 2, a.y + a.size / 2) }
    else if (f.tick > 0 && f.population === 0) { c.fillStyle = P.ink3; c.font = mono(13, 600); c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText('extinct: the grass keeps growing', a.x + a.size / 2, a.y + a.size / 2) }
  })
  const legend = tint === 'speed' ? <><span style={{ color: P.blu }}>slow 0.80</span> → white → <span style={{ color: P.ora }}>fast 1.25</span></> : tint === 'metab' ? <><span style={{ color: P.vio }}>frugal 0.80</span> → white → <span style={{ color: P.red }}>costly 1.20</span></> : <>{ACTIVITY.map(([l, c], i) => <span key={l} style={{ color: c }}>{i ? ' · ' : ''}{l}</span>)}</>
  return <Figure caption={<>Each blob is a sheep, colored by {tint === 'activity' ? 'what it is doing' : `its inherited ${tint === 'speed' ? 'speed' : 'metabolism'}`}: {legend}. Lambs are smaller. Drag to graze the field bare. Changing the seed or flock size restarts the meadow; the sliders apply on the next tick.</>}>
    <div style={{ maxWidth: '34em', margin: '0 auto' }}>
      <Stage aspect="1 / 1"><canvas ref={ref} {...brush} role="img" aria-label={`Top-down meadow with ${f?.population ?? 0} sheep on a 48 by 48 grass field.`} /></Stage>
    </div>
    <Controls>
      <Btn onClick={() => main.setPlaying(p => !p)} disabled={!!f?.halted}>{main.playing ? 'Pause' : 'Play'}</Btn>
      <Btn ghost onClick={() => sim.restart()}>Restart</Btn>
      <Btn ghost onClick={() => main.setSeed(s => s + 1)} label="Use the next seed">seed {main.seed} →</Btn>
      <Choice label="per frame" value={main.speed} options={SPEEDS} onChange={main.setSpeed} />
      <Choice label="color" value={tint} options={TINTS} onChange={setTint} />
    </Controls>
    <Controls>
      <Range label="sheep at start" value={main.population} min={2} max={200} step={2} onChange={main.setPopulation} />
      <Range label="regrowth" value={main.regrow} min={0} max={.06} step={.0025} onChange={main.setRegrow} format={v => v.toFixed(4)} />
      <Range label="metabolism ×" value={main.metab} min={.5} max={2} step={.05} onChange={main.setMetab} format={v => v.toFixed(2)} />
    </Controls>
    <Stats>
      <Stat label="time" value={f && fmtTime(f.time)} /><Stat label="sheep" value={f?.population} /><Stat label="lambs" value={f?.lambs} />
      <Stat label="births" value={f?.births} /><Stat label="deaths" value={f && <>{f.old + f.starved} <span style={{ fontWeight: 400, color: P.ink2 }}>({f.old} age / {f.starved} hunger)</span></>} />
      <Stat label="food" value={f && pct(f.food)} /><Stat label="mean speed" value={f?.population ? f.speed.toFixed(3) : '—'} /><Stat label="mean metabolism" value={f?.population ? f.metab.toFixed(3) : '—'} /><Stat label="generation" value={f?.generation} />
    </Stats>
  </Figure>
}

function HistoryFigure({ main }: { main: Main }) {
  const ref = usePaint(canvas => { const { c, width, height } = surface(canvas); drawHistory(c, width, height, main.history) })
  const f = main.sim.frame, ok = f ? f.founders + f.births - f.old - f.starved === f.population : null
  return <Figure caption={<>Population (violet, left axis) and food left in the meadow (green, right axis), sampled every simulated second from the meadow above.</>}>
    <Stage aspect="16 / 7"><canvas ref={ref} role="img" aria-label="Line chart of sheep population and total food over time, showing boom and bust cycles." /></Stage>
    <Stats>{f && <><Stat label="founders + births − deaths" value={`${f.founders} + ${f.births} − ${f.old + f.starved} = ${f.founders + f.births - f.old - f.starved}`} /><Stat label="living" value={<span style={{ color: ok ? P.grn : P.red }}>{f.population} {ok ? '✓' : '✗'}</span>} /></>}</Stats>
  </Figure>
}

function TraitFigure({ main }: { main: Main }) {
  const f = main.sim.frame, s = main.start
  const ref = usePaint(canvas => {
    const { c, width, height } = surface(canvas)
    drawTraits(c, width, height, [{ title: 'speed', lo: .8, hi: 1.25, start: s?.speeds, now: f?.speeds }, { title: 'metabolism', lo: .8, hi: 1.2, start: s?.metabs, now: f?.metabs }])
  })
  return <Figure caption={<>Share of the flock in each trait bin: the founders (wide gray bars) against the sheep alive now (narrow violet bars).</>}>
    <Stage aspect="16 / 6"><canvas ref={ref} role="img" aria-label="Histograms of speed and metabolism, founders versus the current flock." /></Stage>
    <Stats>
      <Stat label="speed" value={s && f ? `${s.speed.toFixed(3)} → ${f.population ? f.speed.toFixed(3) : '—'}` : null} />
      <Stat label="metabolism" value={s && f ? `${s.metab.toFixed(3)} → ${f.population ? f.metab.toFixed(3) : '—'}` : null} />
      <Stat label="generation" value={f?.generation} />
    </Stats>
  </Figure>
}

const growCode = `
# Regrowth touches every cell independently, so each fork is a parallel call.
def grow(f: Field, +rate: F32) -> Field:
  match f:
    case Void{}:
      Void{}
    case Cell{g, +k}:
      Cell{F32.min(4.0, (g + rate * k : F32)), k}
    case Fork{l, r}:
      a b = grow(l, rate) grow(r, rate)
      Fork{a, b}
`

const budgetCode = `
# Phase 1: age, basal metabolism, hunger hysteresis.
def older(s: Sheep, +burn: F32, +dt: F32) -> Sheep & F32:
  Sheep{..., +e, ..., +metab, ...} = s
  +spent = F32.min(e, (burn * metab : F32))
  +left = (e - spent : F32)
  +starving = (left < 68.0 : F32)
    || (hungry && (left <= 94.0 : F32))
  ...

# In walk: pay 0.65 x speed^2 per unit walked.
  +step = F32.sqrt((mx * mx + mz * mz : F32))
  +cost = (step * 0.65 * speed * speed : F32)
`

const breedCode = `
# Adult, fed, rested, and not hungry: ready to breed.
def ready(s: Sheep) -> Bool:
  Sheep{id, gen, x, z, h, age, life, e, cool, ...} = s
  (age >= 45.0 : F32) && (e >= 80.0 : F32) && (cool <= 0.0 : F32) && Bool.not(hungry)

def paid(s: Sheep) -> Sheep:      # each parent: -22 energy, 35 s cooldown
  Sheep{..., e, cool, ...} = s
  Sheep{..., (e - 22.0 : F32), 35.0, ...}

def inherit(+a: F32, +b: F32, +k: U32, salt: U32, +w: F32, lo: F32, hi: F32) -> F32:
  F32.clamp(((a + b) / 2.0 + jit(k, salt, w) : F32), lo, hi)
`

const keyCode = `
# A sheep's dice depend only on (seed, tick, id), never on visiting order.
def key(+seed: U32, +tick: U32, +id: U32) -> U32:
  R.hash((seed .^. R.hash((tick * 2246822519 + id * 3266489917 + 1 : U32)) : U32))

def rnd(+k: U32, salt: U32) -> F32:
  R.random((k + salt * 2654435769 : U32))
`

const splitCode = `
# Decisions only read the snapshot, so the flock splits into halves that
# think in parallel: 2^(depth+1) chunks in all.
def flock_think(d: Nat, hs: Halves, +grass: Field, +ms: List<&2, Mate>, ...) -> List<&2, Sheep>:
  match d:
    case 0n:
      Halves{a, na, b, nb} = hs
      List.append(&2, Sheep, think_all(a, grass, ms, ...), think_all(b, grass, ms, ...))
    case 1n+ +p:
      Halves{a, na, b, nb} = hs
      x y = flock_think(p, halve(a, na), grass, ms, ...) flock_think(p, halve(b, nb), grass, ms, ...)
      List.append(&2, Sheep, x, y)
`

const PHASES = [
  ['grow grass', 'field', 'field', 'yes: per cell'],
  ['age, metabolism, deaths', 'each sheep', 'each sheep', 'a fold'],
  ['decide and walk', 'field + ready list', 'each sheep', 'yes: per sheep'],
  ['graze', 'field', 'field + sheep', 'no: one bite at a time'],
  ['breed', 'flock', 'flock + lambs', 'no: one pairing at a time'],
] as const

export default function Meadow() {
  const { call, error } = useWorker(() => new Worker(new URL('./meadow.worker.ts', import.meta.url), { type: 'module' }))
  const main = useMain(call as Call)
  const prints = main.sim.frame?.prints ?? []
  return (
    <DocPage id="meadow" kicker="life & ecology" title="Sheep meadow" sub={<>sheep <b>graze</b>, <b>breed</b>, and <b>inherit</b>, computed in <b>Bend</b></>}
      tag={<>48 × 48 grass · energy budgets · blending inheritance · same seed, same meadow</>}>
      {error && <p className="note" role="alert" style={{ color: P.red }}>The Bend worker failed: {error}</p>}
      <p>A meadow, some sheep, and a few rules about energy. Nobody tells the
        flock how big to be. The number of sheep rises and falls on its own,
        and their traits drift as some sheep leave more lambs than others.</p>
      <p>This is a 2D retelling of my 3D <a href="https://github.com/krymancer/sheep-simulation-3d">sheep meadow</a>.
        It uses the same rules and the same constants. The 3D version runs them in TypeScript.
        Here every rule is written in Bend (<code>demos/meadow/ecology.bend</code>), compiled to
        JavaScript, and run in a worker. The page only draws what comes back.</p>

      <Section n={1} title={<>Grass <b>GROWS</b>.</>}>
        <p>The meadow is a 48 × 48 grid. Each cell holds between 0 and 4 units of
          biomass and has a <em>fertility</em> between 0.35 and 1.35. Fertility is a smooth
          sine pattern plus seeded noise, so some parts of the field are always lusher than others.</p>
        <p>Growth is linear, not logistic. Every second, each cell gains <code>0.015 × fertility</code> until
          it reaches 4. Even a cell grazed to zero grows back, because there are no roots to kill.
          A bare cell of average fertility (0.85) needs 4 / (0.015 × 0.85) ≈ 314 s to fill up.</p>
        <GrassFigure call={call as Call} />
        <p>In Bend the field is a binary tree over the cell index <code>z × 48 + x</code>, padded
          to 4096 leaves. The padding is <code>Void</code>, so it costs one node per empty subtree.
          No cell's growth depends on its neighbors, so the two halves of every fork can grow at the same time:</p>
        <Code file="demos/meadow/ecology.bend">{growCode}</Code>
        <p><code>a b = grow(l, rate) grow(r, rate)</code> is Bend's parallel call. In a native or
          GPU build the 2304 cells are spread over every core. This JavaScript build runs it
          sequentially, but the program is the same.</p>
      </Section>

      <Section n={2} title={<>Sheep <b>EAT</b>, and every step costs.</>}>
        <p className="lead">Energy in, energy out.</p>
        <p>A sheep is an energy budget with room for 100:</p>
        <ul>
          <li><b>Basal metabolism</b> costs <code>0.72 × inherited metabolism × global factor</code> per second, moving or not.</li>
          <li><b>Walking</b> costs <code>0.65 × speed²</code> per cell travelled. Fast legs are expensive: a sheep with speed 1.2 pays 44% more per cell than one with 1.0.</li>
          <li><b>Grazing</b> takes up to 0.8 biomass per second and turns it into 12 energy per unit, so at most 9.6 energy per second.</li>
        </ul>
        <Bars title="energy per second, a sheep with speed and metabolism 1.0" data={[
          { label: 'grazing', value: 9.6, display: '+9.6', hi: true },
          { label: 'basal', value: .72, display: '−0.72' },
          { label: 'walking to food', value: .65, display: '−0.65' },
          { label: 'wandering (60% pace)', value: .39, display: '−0.39' },
        ]} caption="Grazing earns over ten times what living costs, so a sheep only needs to eat for a short part of its day." />
        <p>Hunger works like a thermostat with two set points. A sheep starts looking for food
          when its energy drops <b>below 68</b> and keeps eating until it is <b>above 94</b>. With a single
          threshold, a sheep at 68 would take one bite, cross the line, walk away, drop back,
          and turn around again. The gap between the two numbers is called <em>hysteresis</em>. It makes
          every meal long and every walk long.</p>
        <p>To look for food, a sheep scans the 15 × 15 cells around it. It picks the cell with the best <code>biomass ÷ (1 + distance)</code>,
          skips cells with less than 0.12, and reconsiders about once a second. It walks to that
          cell and starts grazing when it is within half a cell.</p>
        <LoneFigure call={call as Call} />
        <Code file="demos/meadow/ecology.bend">{budgetCode}</Code>
        <p>There is no <code>if</code> in Bend. Each comparison returns a <code>Bool</code>, and every branch is a
          <code> match</code> on one. Values are used at most once unless marked <code>+</code>, which is why
          <code> +e</code> and <code>+metab</code> are marked: they are read twice.</p>
      </Section>

      <Section n={3} title={<>Sheep <b>BREED</b>, and lambs inherit.</>}>
        <p>Two adults (at least 45 s old) can make a lamb when they are within 1.9 cells
          of each other, both have at least 80 energy, and neither is hungry or resting. Each parent pays 22:
          20 goes into the lamb and 2 is lost. A birth costs the pair 44 and the lamb starts with 40.
          Both parents then rest for 35 s. A ready sheep that sees another ready sheep within 6 cells walks toward it.</p>
        <p>A lamb's speed, metabolism, and wool tint are the <b>average of its parents</b>, plus a small
          uniform mutation (±0.04, ±0.04, ±0.025), clamped to the founders' ranges. It lives
          240 to 360 s, unless it starves first.</p>
        <Code file="demos/meadow/ecology.bend">{breedCode}</Code>
        <p>Nothing in the code says "prefer a low metabolism". But a frugal sheep survives a famine a
          little more often, and survivors are the ones that breed. That is natural selection: no goal,
          just some sheep surviving and breeding more than others.</p>
        <MeadowFigure main={main} />
        <p className="note">What differs from the 3D original: 10 ticks per second instead of 20. There is no
          herding or personal-space push, because those need every sheep to check its neighbors every tick.
          A ready sheep walks to the <i>nearest</i> ready sheep. All sheep decide from the same snapshot
          of the field, and then their bites are applied one by one. The bookkeeping is the same, including the 1000-sheep ceiling:
          it pauses the meadow rather than quietly culling sheep.</p>
      </Section>

      <Section n={4} title={<><b>BOOM</b> and bust.</>}>
        <p>Press play above, choose 20 or 60 ticks per frame, and watch the two lines below.</p>
        <HistoryFigure main={main} />
        <p>The meadow starts 75% full, which is about 6900 units of grass, or 83,000 energy. That stock lets the flock grow far
          past what regrowth can feed. Regrowth adds about 2304 × 0.85 × 0.015 ≈ 29 biomass per second, which is
          ≈ 350 energy per second, while a sheep spends roughly 1 to 1.2. So the meadow can feed a few hundred sheep
          in the long run. Births lag behind food, though, because a lamb needs 45 s to mature. The flock overshoots,
          eats the meadow down, and crashes as sheep starve. The grass recovers while few sheep are left, and
          the cycle repeats. It is the predator–prey story, with grass as the prey.</p>
        <p>Now look at the traits.</p>
        <TraitFigure main={main} />
        <p>The founders' traits are drawn uniformly, so their histogram is flat. After a few generations
          the flock is a narrow spike, whatever the founders looked like. This comes from <b>blending</b>: a
          lamb is the average of two parents, which halves the variance every generation. A uniform mutation of ±w adds back w²/3:</p>
        <p className="quote">Var′ = Var / 2 + w² / 3<br />Var* = 2w² / 3,  σ* = w·√(2/3) ≈ 0.033</p>
        <p>That spread of 0.033, for w = 0.04, is what the meadow shows. This was Fleeming Jenkin's 1867 objection to Darwin:
          blending inheritance would wash out variation. Mendel's discrete genes turned out to be the answer.
          With so little variation to act on, selection is slow, and chance (which sheep happen to
          meet, which happen to starve) is as strong as the selection. I ran eight seeds for 3000 s. Mean metabolism fell in six of them, by 0.02 on
          average and up to 0.06, and rose slightly in two. Seed 42, the default, is one of the two, so try a few others.
          Speed moved both ways. A fast sheep reaches food and mates sooner, but it pays speed² for every cell.</p>
      </Section>

      <Section n={5} title={<>Same <b>SEED</b>, same meadow.</>}>
        <p>Every random choice in the meadow is a hash of the seed, the tick, the sheep's id, and a salt for the purpose.
          No sheep draws from a shared stream of random numbers, so the result does not depend on which
          sheep goes first, or on which core it runs on.</p>
        <Code file="demos/meadow/ecology.bend">{keyCode}</Code>
        <p>That is what makes it safe to split the flock. Decisions read a frozen snapshot: the field after
          regrowth, and a list of who is ready to breed and where. Each sheep's plan depends only on
          itself and that snapshot:</p>
        <Code file="demos/meadow/ecology.bend">{splitCode}</Code>
        <p>Two phases cannot be split. Grazing is a fold: when two sheep want the same cell, the first takes what it
          can and the second gets the rest, so no grass is eaten twice. Breeding is a fold too, because a sheep
          can pair at most once per tick.</p>
        <table>
          <thead><tr><th>phase</th><th>reads</th><th>writes</th><th>parallel?</th></tr></thead>
          <tbody>{PHASES.map(([p, r, w, par]) => <tr key={p} className={par.startsWith('yes') ? 'hi' : undefined}><td>{p}</td><td>{r}</td><td>{w}</td><td>{par}</td></tr>)}</tbody>
        </table>
        <p>The meadow above records a 32-bit fingerprint of every sheep and the whole field once per
          simulated minute. Restart with the same seed and settings, and the same fingerprints come back at any
          playback speed. Painting and the sliders are outside commands, so using them changes the fingerprints.</p>
        <Stats>{prints.length ? prints.slice(-4).map(([t, h]) => <Stat key={t} label={`${t / 600} min`} value={hex(h)} />) : <span>no fingerprints yet: let the meadow run for a simulated minute</span>}</Stats>
        <p><code>tests/meadow.test.mjs</code> checks the Bend module against an independent JavaScript
          reference. The field is checked cell by cell, bit for bit, through 400 ticks of regrowth. The test also checks that
          living = founders + births − deaths throughout a long run, that every tick conserves biomass
          (grown − eaten = change), that a birth costs exactly 4 energy, that the ceiling pauses
          instead of charging parents, and that splitting a run into batches changes nothing.</p>
        <p>The 3D original, with herding, an inspector, and live graphs,
          is <a href="https://github.com/krymancer/sheep-simulation-3d">krymancer/sheep-simulation-3d</a>.
          For a world with even fewer rules, try the <a href="#life">Game of Life</a>.</p>
      </Section>
    </DocPage>
  )
}
