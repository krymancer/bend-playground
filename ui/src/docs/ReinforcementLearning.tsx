import { useEffect, useRef, useState, type PointerEvent, type ReactNode } from 'react'
import { Bars, Btn, Choice, Code, Controls, DocPage, Figure, Range, Section, Stage, Stat, Stats } from '@/components/Doc'
import { surface, usePaint } from '@/lib/lab-canvas'
import { useWorker } from '@/lib/use-lab'
import { FLOOR, GOAL, PIT, PRESETS, WALL, cellAt, copy, drawDeltas, drawGrid, drawLegend, drawReturns, layout, type Preset, type World } from './qlearning-draw'

// Every number drawn here comes from demos/qlearning/agent.bend, run in a worker.
type Call = <T>(body: Record<string, unknown>) => Promise<T>
interface Status { value: Float32Array; action: Uint8Array; spread: Float32Array; route: Int16Array; episodes: number; steps: number }
interface Trained extends Status { returns: Float32Array; path: Int16Array }
interface Planned { value: Float32Array; action: Uint8Array; spread: Float32Array; route: Int16Array; sweeps: number; delta: number }
const COST = 0.04
const sleep = (ms: number) => new Promise(r => setTimeout(r, ms))
const fmt = (n: number) => n.toLocaleString('en-US')

function useNarrow() {
  const [narrow, setNarrow] = useState(() => typeof window !== 'undefined' && window.innerWidth < 560)
  useEffect(() => { const f = () => setNarrow(window.innerWidth < 560); window.addEventListener('resize', f); return () => window.removeEventListener('resize', f) }, [])
  return narrow
}
// Pause animations while their figure is off screen.
function useVisible<T extends HTMLElement>() {
  const ref = useRef<T>(null), visible = useRef(false)
  useEffect(() => {
    const el = ref.current; if (!el) return
    const o = new IntersectionObserver(([e]) => { visible.current = e.isIntersecting }); o.observe(el); return () => o.disconnect()
  }, [])
  return [ref, visible] as const
}
const outcome = (w: World, path: ArrayLike<number>) => { const k = w.kinds[path[path.length - 1]]; return k === GOAL ? 'reached the goal' : k === PIT ? 'fell in a pit' : 'ran out of time' }

const envCode = `
# Actions: 0 up, 1 right, 2 down, 3 left. Walls block.
def move(+world: World, +s: U32, +a: U32) -> U32:
  +t = target(s, a)
  Bool.pick(U32, U32.is_eq(cell(world, t), 1), s, t)

def reward(+k: U32, +cost: F32) -> F32:
  Bool.pick(F32, U32.is_eq(k, 3), 1.0, Bool.pick(F32, U32.is_eq(k, 2), (0.0 - 1.0 : F32), (0.0 - cost : F32)))
`
const learnCode = `
# Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
def learn(+old: F32, r: F32, future: F32, alpha: F32, gamma: F32) -> F32:
  (old + alpha * (r + gamma * future - old) : F32)

# Epsilon-greedy: with probability eps a random action, else the best one.
def act(+t: Table, +s: U32, +eps: F32, +seed: U32) -> U32:
  +explore = (R.random(seed) < eps : F32)
  Bool.pick(U32, explore, (R.hash((seed + 1 : U32)) % 4 : U32), greedy(t, s, R.hash((seed + 2 : U32))))
`
const episodeCode = `
def episode(fuel: Nat, alive: Bool, +t: Table, +world: World, +s: U32, ...) -> Episode:
  match fuel:
    case 0n: Episode{t, ret, steps, path}
    case 1n+p:
      match alive:
        case False{}: Episode{t, ret, steps, path}
        case True{}:
          +a = act(t, s, eps, seed)
          +next = move(world, s, a)
          +k = cell(world, next)
          +r = reward(k, cost)
          +done = terminal(k)
          +future = Bool.pick(F32, done, 0.0, best(t, next))
          +i = (s * 4 + a : U32)
          learned = put(8n, True{}, t, i, learn(get(8n, t, i), r, future, alpha, gamma))
          episode(p, Bool.not(done), learned, world, next, ...)
`
const sweepCode = `
# V(s) <- max_a [ r(s,a) + gamma * V(s') ]
def backup(+v: Table, +world: World, +s: U32, +gamma: F32, +cost: F32) -> F32:
  +a = F32.max(lookahead(v, world, s, 0, gamma, cost), lookahead(v, world, s, 1, gamma, cost))
  +b = F32.max(lookahead(v, world, s, 2, gamma, cost), lookahead(v, world, s, 3, gamma, cost))
  Bool.pick(F32, (cell(world, s) >= 1 : U32), 0.0, F32.max(a, b))

# Every cell reads only the old V, so the two halves of the grid are
# independent: a parallel call splits them, down to single cells.
def sweep(depth: Nat, +lo: U32, +v: Table, +world: World, +gamma: F32, +cost: F32) -> Table:
  match depth:
    case 0n: Leaf{backup(v, world, lo, gamma, cost)}
    case 1n+ +p:
      left right = sweep(p, lo, v, world, gamma, cost) sweep(p, (lo + (1 << p) : U32), v, world, gamma, cost)
      Node{left, right}
`

// 1. A random agent: Q-learning with alpha = 0 and eps = 1 never learns and always explores.
function RandomWalk({ call, world }: { call: Call; world: World }) {
  const narrow = useNarrow(), [box, visible] = useVisible<HTMLDivElement>()
  const [path, setPath] = useState<number[]>([]), [t, setT] = useState(0), [ret, setRet] = useState<number | null>(null), [n, setN] = useState(0)
  useEffect(() => {
    let alive = true
    ;(async () => {
      await call({ action: 'reset', key: 'random', ...world, seed: 7 })
      while (alive) {
        if (!visible.current) { await sleep(250); continue }
        const r = await call<Trained>({ action: 'train', key: 'random', n: 1, alpha: 0, gamma: .95, eps: 1, cost: COST })
        if (!alive) break
        const p = [...r.path]; setPath(p); setRet(null); setN(r.episodes)
        for (let i = 0; i < p.length && alive; i++) { setT(i); await sleep(p.length > 60 ? 35 : 70) }
        setRet(r.returns[0]); await sleep(1200)
      }
    })().catch(() => {})
    return () => { alive = false }
  }, [call, world, visible])
  const canvas = usePaint(cv => {
    const { c, width, height } = surface(cv), L = layout(width, height)
    drawGrid(c, L.gx, L.gy, L.gs, { ...world, path: path.slice(0, t + 1), agent: path.length ? path[t] : world.start })
    drawLegend(c, narrow ? { ...L.side, h: L.side.h } : { x: L.side.x, y: L.side.y + 6, w: L.side.w, h: Math.min(L.side.h, 190) })
  })
  const done = ret != null
  return <Figure caption="A random agent. It picks each of the four moves with equal odds and learns nothing. Most episodes end in a pit.">
    <div ref={box}><Stage aspect={narrow ? '3 / 4' : '16 / 10'}><canvas ref={canvas} role="img" aria-label="Gridworld with a randomly moving agent: walls, a row of pits along the bottom, and the goal in the bottom-right corner" /></Stage></div>
    <Stats><Stat label="episode" value={n || null} /><Stat label="steps" value={path.length ? t : null} /><Stat label="return" value={done ? ret!.toFixed(2) : '…'} /><Stat label="outcome" value={done ? outcome(world, path) : '…'} /></Stats>
  </Figure>
}

// 2 and 3. A live Q-learner. With `onEdit` the reader can paint the world.
const TOOLS = [['wall', 'wall'], ['pit', 'pit'], ['goal', 'goal'], ['start', 'start'], ['floor', 'floor']] as const
type Tool = typeof TOOLS[number][0]
function Learner({ call, id, world, onEdit, defaults }: { call: Call; id: string; world: World; onEdit?: (w: World) => void; defaults: { speed: number; alpha: number; gamma: number; eps: number } }) {
  const narrow = useNarrow(), [box, visible] = useVisible<HTMLDivElement>()
  const [running, setRunning] = useState(false), [speed, setSpeed] = useState(defaults.speed)
  const [alpha, setAlpha] = useState(defaults.alpha), [gamma, setGamma] = useState(defaults.gamma), [eps, setEps] = useState(defaults.eps), [seed, setSeed] = useState(1)
  const [status, setStatus] = useState<Status | null>(null), [path, setPath] = useState<Int16Array>(new Int16Array(0))
  const [watch, setWatch] = useState<{ route: Int16Array; t: number } | null>(null), [tool, setTool] = useState<Tool>('wall'), [hover, setHover] = useState(-1)
  const returns = useRef<number[]>([]), smooth = useRef<number[]>([]), epoch = useRef(0), params = useRef({ speed, alpha, gamma, eps })
  params.current = { speed, alpha, gamma, eps }
  const worldRef = useRef(world); worldRef.current = world

  const reset = async (s = seed) => {
    const e = ++epoch.current; setWatch(null)
    const r = await call<Status>({ action: 'reset', key: id, ...worldRef.current, seed: s })
    if (e !== epoch.current) return
    returns.current = []; smooth.current = []; setStatus(r); setPath(new Int16Array([worldRef.current.start]))
  }
  useEffect(() => { reset() }, []) // eslint-disable-line react-hooks/exhaustive-deps
  // Edits keep the Q table: the agent has to discover the change by acting.
  const first = useRef(true)
  useEffect(() => {
    if (first.current) { first.current = false; return }
    const e = epoch.current
    call<Status>({ action: 'world', key: id, ...world }).then(r => { if (e === epoch.current) setStatus(r) }).catch(() => {})
  }, [world, call, id])

  useEffect(() => {
    if (!running) return
    let alive = true
    ;(async () => {
      while (alive) {
        if (!visible.current) { await sleep(250); continue }
        const e = epoch.current, p = params.current
        const r = await call<Trained>({ action: 'train', key: id, n: p.speed, alpha: p.alpha, gamma: p.gamma, eps: p.eps, cost: COST })
        if (!alive || e !== epoch.current) continue
        for (const x of r.returns) { const last = smooth.current.at(-1); returns.current.push(x); smooth.current.push(last == null ? x : last + .05 * (x - last)) }
        setStatus(r); setPath(r.path)
        await sleep(p.speed === 1 ? 140 : 70)
      }
    })().catch(() => {})
    return () => { alive = false }
  }, [running, call, id, visible])

  useEffect(() => {
    if (!watch || watch.t >= watch.route.length - 1) return
    const h = setTimeout(() => setWatch(w => w && { ...w, t: w.t + 1 }), 170); return () => clearTimeout(h)
  }, [watch])
  const watchGreedy = () => { if (!status) return; setRunning(false); setWatch({ route: status.route, t: 0 }) }

  const canvas = usePaint(cv => {
    const { c, width, height } = surface(cv), L = layout(width, height)
    const trail = watch ? watch.route.slice(0, watch.t + 1) : path
    drawGrid(c, L.gx, L.gy, L.gs, { ...world, value: status?.value, action: status?.action, spread: status?.spread, path: trail, agent: trail.length ? trail[trail.length - 1] : world.start, hover: onEdit ? hover : -1 })
    drawReturns(c, L.side, returns.current, smooth.current)
  })

  const paint = useRef<number | null>(null)
  const edit = (i: number, begin: boolean) => {
    if (!onEdit || i < 0) return
    const w = copy(world), k = w.kinds[i]
    if (tool === 'start') { if (begin && k !== WALL) { w.start = i; if (k !== FLOOR) w.kinds[i] = FLOOR } else return }
    else {
      const want = { wall: WALL, pit: PIT, goal: GOAL, floor: FLOOR }[tool]
      if (begin) paint.current = k === want ? FLOOR : want
      if (paint.current == null || i === w.start || k === paint.current) return
      w.kinds[i] = paint.current
    }
    setWatch(null); onEdit(w); setRunning(true)
  }
  const at = (e: PointerEvent<HTMLCanvasElement>) => { const r = e.currentTarget.getBoundingClientRect(); return cellAt(r.width, r.height, e.clientX - r.left, e.clientY - r.top) }
  const handlers = onEdit ? {
    onPointerDown: (e: PointerEvent<HTMLCanvasElement>) => { e.currentTarget.setPointerCapture(e.pointerId); edit(at(e), true) },
    onPointerMove: (e: PointerEvent<HTMLCanvasElement>) => { const i = at(e); setHover(i); if (e.buttons) edit(i, false) },
    onPointerUp: () => { paint.current = null },
    onPointerLeave: () => setHover(-1),
  } : {}

  const recent = returns.current.slice(-50), avg = recent.length ? recent.reduce((a, b) => a + b, 0) / recent.length : null
  const route = status?.route, reaches = route && world.kinds[route[route.length - 1]] === GOAL
  return <Figure caption={onEdit
    ? <>Pick a tool, then click or drag on the grid. The agent keeps its Q table, so it must discover your change by bumping into it. Presets: {Object.keys(PRESETS).join(', ')}.</>
    : <>Each cell is tinted by its best Q value (green is good, red is bad) and shows the greedy move. The violet line is the latest training episode. The chart shows the return of every episode and its moving average.</>}>
    <div ref={box}><Stage aspect={narrow ? '3 / 4' : '16 / 10'}>
      <canvas ref={canvas} {...handlers} style={onEdit ? { cursor: 'pointer', touchAction: 'none' } : undefined} role="img"
        aria-label={onEdit ? 'Editable gridworld: click a cell to place the selected tool. Cell colors show learned values, arrows show the greedy policy.' : 'Gridworld being learned by Q-learning: cell colors show learned values, arrows show the greedy policy, and a chart shows the return of each episode.'} />
    </Stage></div>
    <Controls>
      <Btn onClick={() => { setWatch(null); setRunning(r => !r) }} pressed={running}>{running ? 'Pause' : 'Train'}</Btn>
      <Btn ghost onClick={() => { setRunning(false); reset() }}>Reset</Btn>
      <Btn ghost onClick={watchGreedy} disabled={!status}>Watch greedy episode</Btn>
      {onEdit && <>
        <Choice label="tool" value={tool} options={TOOLS} onChange={setTool} />
        <Choice<Preset | ''> label="preset" value="" options={[['', 'choose…'], ...Object.keys(PRESETS).map(k => [k, k] as const)] as [Preset | '', string][]} onChange={p => { if (p) { setWatch(null); onEdit(copy(PRESETS[p])); setRunning(true) } }} />
      </>}
      <Range label="episodes/tick" value={speed} min={1} max={50} onChange={setSpeed} />
      {!onEdit && <>
        <Range label="α" value={alpha} min={0.05} max={1} step={0.05} onChange={setAlpha} format={v => v.toFixed(2)} />
        <Range label="γ" value={gamma} min={0.5} max={0.99} step={0.01} onChange={setGamma} format={v => v.toFixed(2)} />
        <Range label="ε" value={eps} min={0} max={1} step={0.05} onChange={setEps} format={v => v.toFixed(2)} />
        <Range label="seed" value={seed} min={1} max={20} onChange={s => { setSeed(s); setRunning(false); reset(s) }} />
      </>}
    </Controls>
    <Stats>
      <Stat label="episodes" value={status ? fmt(status.episodes) : null} />
      <Stat label="steps" value={status ? fmt(status.steps) : null} />
      <Stat label="avg return (last 50)" value={avg == null ? null : avg.toFixed(2)} />
      <Stat label="greedy route" value={route ? reaches ? `${route.length - 1} steps to goal` : 'not there yet' : null} />
    </Stats>
  </Figure>
}

// 4. Value iteration on the reader's world, one sweep at a time.
function Planner({ call, world }: { call: Call; world: World }) {
  const narrow = useNarrow(), [box, visible] = useVisible<HTMLDivElement>()
  const [plan, setPlan] = useState<Planned | null>(null), [deltas, setDeltas] = useState<number[]>([]), [run, setRun] = useState(0), [auto, setAuto] = useState(true)
  useEffect(() => {
    let alive = true
    ;(async () => {
      const p = await call<Planned>({ action: 'plan-reset', key: 'plan', ...world, gamma: .95, cost: COST })
      if (!alive) return
      setPlan(p); setDeltas([])
      if (!auto) return
      let delta = Infinity
      while (alive && delta > 1e-4 && p.sweeps < 80) {
        if (!visible.current) { await sleep(250); continue }
        await sleep(260)
        const r = await call<Planned>({ action: 'sweep', key: 'plan', gamma: .95, cost: COST })
        if (!alive) return
        delta = r.delta; p.sweeps = r.sweeps; setPlan(r); setDeltas(d => [...d, r.delta])
      }
    })().catch(() => {})
    return () => { alive = false }
  }, [world, run, call, visible]) // eslint-disable-line react-hooks/exhaustive-deps
  const once = async () => {
    setAuto(false)
    const r = await call<Planned>({ action: 'sweep', key: 'plan', gamma: .95, cost: COST }); setPlan(r); setDeltas(d => [...d, r.delta])
  }
  const canvas = usePaint(cv => {
    const { c, width, height } = surface(cv), L = layout(width, height)
    drawGrid(c, L.gx, L.gy, L.gs, { ...world, value: plan?.value, action: plan && plan.sweeps ? plan.action : undefined, path: plan?.route })
    drawDeltas(c, L.side, deltas)
  })
  const route = plan?.route, reaches = route && world.kinds[route[route.length - 1]] === GOAL
  return <Figure caption="Value iteration on your world from section 3. Each sweep backs up all 64 cells from the previous V. Value spreads outward from the goal one cell per sweep; the violet line is the plan from the start.">
    <div ref={box}><Stage aspect={narrow ? '3 / 4' : '16 / 10'}><canvas ref={canvas} role="img" aria-label="Value iteration: cells tinted by their value V, arrows for the planned action, and a log-scale chart of the largest change per sweep" /></Stage></div>
    <Controls>
      <Btn onClick={() => { setAuto(true); setRun(r => r + 1) }}>Replay sweeps</Btn>
      <Btn ghost onClick={once} disabled={!plan}>One sweep</Btn>
    </Controls>
    <Stats>
      <Stat label="sweeps" value={plan ? plan.sweeps : null} />
      <Stat label="largest change" value={deltas.length ? deltas[deltas.length - 1].toFixed(3) : null} />
      <Stat label="planned route" value={route && plan?.sweeps ? reaches ? `${route.length - 1} steps to goal` : 'no route yet' : null} />
    </Stats>
  </Figure>
}

// Experience needed until each method's greedy route is the final, shortest one.
function Race({ call, world }: { call: Call; world: World }) {
  const [vi, setVi] = useState<{ sweeps: number; routeSweeps: number; routeLength: number; end: number } | null>(null)
  const [ql, setQl] = useState<{ steps: number; episodes: number; falls: number; done: boolean } | null>(null)
  useEffect(() => {
    let alive = true
    setVi(null); setQl(null)
    ;(async () => {
      await sleep(500); if (!alive) return
      const v = await call<{ sweeps: number; routeSweeps: number; routeLength: number; end: number }>({ action: 'solve', ...world, gamma: .95, cost: COST })
      if (!alive) return
      setVi(v)
      if (world.kinds[v.end] !== GOAL) return
      for (let fresh = true; alive; fresh = false) {
        const r = await call<{ steps: number; episodes: number; falls: number; routeLength: number; end: number }>({ action: 'race', key: 'race', fresh, ...world, seed: 1, n: 10, alpha: .5, gamma: .95, eps: .1, cost: COST })
        const done = r.routeLength === v.routeLength && world.kinds[r.end] === GOAL
        if (!alive) return
        setQl({ steps: r.steps, episodes: r.episodes, falls: r.falls, done })
        if (done || r.episodes >= 5000) return
        await sleep(0)
      }
    })().catch(() => {})
    return () => { alive = false }
  }, [world, call])
  if (vi && world.kinds[vi.end] !== GOAL) return <p className="note">Your world has no reachable goal right now, so there is nothing to race to. Open a path in section 3.</p>
  const data = vi ? [
    { label: 'value iteration: until its route is right', value: vi.routeSweeps * 256, display: fmt(vi.routeSweeps * 256), hi: true },
    { label: 'value iteration: until V converges', value: vi.sweeps * 256, display: fmt(vi.sweeps * 256) },
    { label: 'Q-learning: until its route is right', value: ql?.steps ?? 0, display: ql ? fmt(ql.steps) + (ql.done ? '' : '…') : '…' },
  ] : []
  return <>
    {vi && <Bars key={`${world.kinds.join('')}${world.start}`} title="transitions examined to find the shortest route" data={data}
      caption={<>Value iteration reads all 256 (cell, move) pairs per sweep from the model: {vi.routeSweeps} sweeps to the right route, {vi.sweeps} to converge. Q-learning (α 0.5, γ 0.95, ε 0.1, seed 1) needed {ql ? `${fmt(ql.steps)} real steps over ${fmt(ql.episodes)} episodes, ${fmt(ql.falls)} of which ended in a pit or timed out` : '…'}{ql && !ql.done ? ' so far' : ''}. Recomputed in Bend whenever you edit the world.</>} />}
    {!vi && <p className="note">Racing value iteration against Q-learning in Bend…</p>}
  </>
}

export default function ReinforcementLearning() {
  const { call, error } = useWorker(() => new Worker(new URL('./qlearning.worker.ts', import.meta.url), { type: 'module' }))
  const [world, setWorld] = useState<World>(() => copy(PRESETS.cliff))
  const [cliff] = useState(() => copy(PRESETS.cliff))
  // Children's effects run before this component's, so figures call the worker only once it exists.
  const [ready, setReady] = useState(false)
  useEffect(() => setReady(true), [])
  const later = (n: ReactNode, aspect = '16 / 10') => ready ? n : <Figure><Stage aspect={aspect}>{null}</Stage></Figure>
  return (
    <DocPage id="reinforcement-learning" title="Reinforcement learning"
      sub={<>an agent learns to act from <b>reward</b> alone</>}
      tag={<><b>Q-learning</b> · <b>value iteration</b> · computed in <b>Bend</b></>}>
      {error && <p className="note">Bend worker error: {error}</p>}
      <p>A neural network learning to read digits is shown the right answer
        for every example. Life is rarely that kind. A child learning to ride
        a bike, a program learning to play a game: nobody hands them the
        correct move. They try something and find out, later, whether it went
        well.</p>
      <p>Reinforcement learning is the study of learning from that thin
        signal. This page builds the classic algorithm, <b>Q-learning</b>,
        from scratch. Every step of every episode below is computed
        in <a href="https://bend-lang.com">Bend</a>, in your browser.</p>

      <Section n={1} title={<>Agents <b>ACT</b>.</>}>
        <p>The setting is a loop between an <b>agent</b> and
          an <b>environment</b>. The agent sees a state and picks an action.
          The environment answers with a reward, a single number, and the
          next state. Then it repeats.</p>
        <p className="lead">state → action → reward, next state → …</p>
        <p>Our environment is a gridworld: 64 cells, four moves (up, right,
          down, left). Walls and the edge block a move. Reaching
          the <em>goal</em> pays +1 and ends the episode. A <b>pit</b> costs −1
          and also ends it. Every other step costs 0.04, so wandering is
          not free. The agent wants a high <b>return</b>: the sum of rewards
          over an episode.</p>
        {later(<RandomWalk call={call} world={cliff} />)}
        <p>The shortest route runs along the edge of a row of pits, nine
          steps. A safer route exists, but it is longer. Nobody tells the
          agent any of this. It only ever sees cell numbers and rewards.</p>
        <p>The world itself is a few lines of Bend. Bend has no <code>if</code>:
          a branch is a <code>match</code>, or <code>Bool.pick</code> for a
          plain choice between two values.</p>
        <Code file="demos/qlearning/agent.bend">{envCode}</Code>
      </Section>

      <Section n={2} title={<>Remember what each <b>MOVE</b> is worth.</>}>
        <p>Q-learning keeps a table. For every state <code>s</code> and
          action <code>a</code>, <code>Q(s,a)</code> estimates the return the
          agent will collect if it takes <code>a</code> in <code>s</code> and
          plays well afterwards. Here that is 64 × 4 = 256 numbers, all zero
          at the start. If the table were right, acting well would be easy:
          in each cell, take the move with the largest Q.</p>
        <p>The table is filled in one step at a time. After taking <code>a</code> in <code>s</code>,
          the agent sees a reward <code>r</code> and lands
          in <code>s'</code>. Its best guess for what that move was worth is the
          reward plus whatever the best move from <code>s'</code> is worth:</p>
        <p className="quote">Q(s,a) ← Q(s,a) + α · (r + γ · max Q(s′,·) − Q(s,a))</p>
        <ul>
          <li><b>α</b>, the learning rate, is how far each update moves toward
            that new guess.</li>
          <li><b>γ</b>, the discount, makes a reward that comes later worth
            less. With γ = 0.95, the +1 ten steps away is worth 0.95¹⁰ ≈ 0.6
            today. That is why shorter routes win.</li>
          <li><b>ε</b> is curiosity. With probability ε the agent ignores its
            table and moves at random. Without that, it would repeat the first
            route that worked and never find a better one.</li>
        </ul>
        <p>The trick is that the guess uses the table itself. The goal's +1
          first reaches the cell next to it. Later episodes carry it back one
          cell further each time, until it reaches the start.</p>
        <Code file="demos/qlearning/agent.bend">{learnCode}</Code>
        {later(<Learner call={call} id="learn" world={cliff} defaults={{ speed: 2, alpha: .5, gamma: .95, eps: .1 }} />)}
        <p>Press <b>Train</b> and watch the green spread back from the goal.
          Within a few dozen episodes the arrows point along the pits, and the
          greedy route is the nine-step one.</p>
        <p>Notice that the average return stays below the greedy route's
          0.68. While training, the agent still explores with probability ε,
          and next to the pits a random step can kill it. Q-learning learns
          the value of acting <em>greedily</em> even while it behaves
          randomly some of the time. This is called learning off-policy. Raise
          ε to see more falls. Set γ low and the far-away goal stops
          mattering. Set α small and learning slows to a crawl.</p>
        <p>The Q table is a Bend <code>Table</code>, a binary tree with 256
          leaves. Values in Bend are immutable, so <code>put</code> rebuilds
          the eight nodes on the path to one leaf and shares the rest of the
          tree. An episode is a recursive function. Its <code>Nat</code> fuel
          caps the episode at 200 steps, which is also how Bend knows the
          function terminates:</p>
        <Code file="demos/qlearning/agent.bend (trimmed)">{episodeCode}</Code>
      </Section>

      <Section n={3} title={<>Change the <b>WORLD</b>.</>}>
        <p>The world is just data: a tree of 64 cell kinds handed to Bend
          on every call. Edit it. Wall off the pit-side route, dig a pit in
          the middle of the agent's favorite path, or move the goal.</p>
        {later(<Learner call={call} id="edit" world={world} onEdit={setWorld} defaults={{ speed: 10, alpha: .5, gamma: .95, eps: .1 }} />)}
        <p>The agent does not see your edit. Its table still says the old
          route is good, so it walks into the new wall, gets a little less
          reward than it expected, and corrects that one entry. The error then
          spreads back along the route the same way the goal did. Exploration
          is what finds the new way around. With ε at zero, an agent can stay
          stuck on a route that no longer exists.</p>
      </Section>

      <Section n={4} title={<>When you know the rules, <b>PLAN</b>.</>}>
        <p>Q-learning never looks at <code>move</code> or <code>reward</code>.
          It only lives through them. But we wrote those functions, so we know
          the model exactly. With a model, we do not have to walk anywhere.
          We can apply the same backup to every cell at once:</p>
        <p className="quote">V(s) ← max over a of [ r(s,a) + γ · V(s′) ]</p>
        <p>This is <b>value iteration</b>, Bellman's equation used as an
          update. One <em>sweep</em> rewrites all 64 values from the previous
          ones. Repeat until nothing changes.</p>
        <p>Inside a sweep, every cell reads only the old table and writes its
          own new value. No cell depends on another cell's new value. That is
          exactly what Bend's parallel call is for: <code>left right = sweep(…) sweep(…)</code> promises
          the two halves of the grid are independent. Each half splits again,
          down to 64 single-cell tasks. Compiled natively, the halves run on
          different cores or on the GPU. The JavaScript target on this page
          runs them one after the other.</p>
        <Code file="demos/qlearning/agent.bend">{sweepCode}</Code>
        {later(<Planner call={call} world={world} />)}
        <p>The backup is a <em>contraction</em>: each sweep's largest change
          is at most γ times the previous one, so value iteration always
          converges, and the plan it ends with is optimal. In this small,
          deterministic world it gets there exactly: once the wave from the
          goal has reached every cell, the next sweep changes nothing.</p>
        <p>How much work does each method need? Count the transitions each
          one looks at before its greedy route becomes the shortest one:</p>
        {ready && <Race call={call} world={world} />}
        <p>The counts are about the same size, and on some worlds Q-learning
          needs fewer. Value iteration backs up every (cell, move) pair, even
          ones no good route ever touches. Q-learning only updates the cells it
          actually visits. The difference is what each count <em>costs</em>. A
          backup is arithmetic on a model: a sweep takes a fraction of a
          millisecond, and every cell in it can run in parallel. A Q-learning
          step is an action taken in the world: a robot moving, a game being
          played. Some of those actions end in a pit.</p>
        <p>Planning also needs something Q-learning does not: the rules of
          the world. A robot does not know the exact physics of its own
          gripper. A game-playing agent may not have the game's code. When the
          model is missing, or too big to sweep, learning from experience is
          the only option.</p>
      </Section>

      <Section n={5} title={<>From tables to <b>NETWORKS</b>.</>}>
        <p>A table with 256 entries is fine for 64 cells. A game screen has
          more possible states than there are atoms in the universe, and most
          of them will never be seen twice. <b>Deep Q-learning</b> replaces
          the table with a <a href="#neural-networks">neural network</a> that
          takes a state and outputs four Q values. It trains on the same
          target, r + γ · max Q(s′,·). This is how DeepMind's 2015 agent
          learned Atari games from pixels.</p>
        <p>There is another way to train a game-playing network without
          answers. My <a href="https://github.com/krymancer/flappIA">flappIA</a> and <a href="https://github.com/krymancer/carcrush-js-ai">CarCrush</a> projects
          used <a href="#neuroevolution">neuroevolution</a>: breed a population
          of networks, keep the ones that fly furthest. That uses one number
          per <b>episode</b>, the fitness, and selection decides which
          networks were good. Q-learning gets a number every <b>step</b>,
          and decides which moves were good by passing value back from one
          state to the state before it. Evolution only learns how good a
          whole life was. Q-learning learns how good each move was.</p>
        <p className="lead">Reward, not answers.</p>
      </Section>
    </DocPage>
  )
}
