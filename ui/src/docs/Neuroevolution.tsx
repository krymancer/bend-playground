import { useCallback, useEffect, useRef, useState } from 'react'
import { Bars, Btn, Choice, Code, Controls, DocPage, Figure, Range, Section, Stage, Stat, Stats } from '@/components/Doc'
import { usePaint } from '@/lib/lab-canvas'
import { useWorker } from '@/lib/use-lab'
import { drawHistory, drawNet, drawWorld, type Frame } from './neuroevolution-draw'

const brainCode = `
type Neuron is Data:
  Neuron{bias: F32, w: List<&2, F32>}

type Brain is Data:
  Brain{hidden: List<&2, Neuron>, out: List<&2, Neuron>}

def fire(n: Neuron, +x: List<&2, F32>) -> F32:
  Neuron{bias, w} = n
  sigmoid(dot(w, x, bias))

def layer(ns: List<&2, Neuron>, +x: List<&2, F32>) -> List<&2, F32>:
  match ns:
    case []: []
    case n <> rest: fire(n, x) <> layer(rest, x)

def forward(brain: Brain, x: List<&2, F32>) -> List<&2, F32>:
  Brain{hidden, out} = brain
  layer(out, layer(hidden, x))
`

const stepCode = `
# Five inputs, normalized as in flappIA.
def senses(+y: F32, +vy: F32, +t: U32, +course: U32) -> List<&2, F32>:
  +k = next_pipe(t)
  +top = gap_top(k, course)
  [((pipe_x(k, t) - bird_x()) / width() : F32), (top / height() : F32),
   ((top + gap()) / height() : F32), (y / height() : F32), (vy / max_vel() : F32)]

# One frame for one bird: sense, think, flap or fall, collide.
def step(b: Bird, +t: U32, +course: U32) -> Bird:
  match b:
    case Bird{alive, +y, +vy, score, +brain}:
      match alive:
        case False{}: Bird{False{}, y, vy, score, brain}
        case True{}: fly(flaps(forward(brain, senses(y, vy, t, course))), y, vy, score, brain, t, course)
`

const selectCode = `
# flappIA weights each bird by score^3, then spins a roulette.
def weight(+score: U32) -> F32:
  +s = (U32.to_f32(score) / 100.0 : F32)
  (s * s * s : F32)

# The wheel is a tree of partial sums: one spin is a walk from the root.
def spin(d: Nat, p: Pool, +ticket: F32) -> Brain:
  match d:
    case 0n: ...
    case 1n+q:
      match p:
        case PLeaf{w, brain}: brain
        case PNode{w, +l, +r}:
          +left = (ticket < mass(l) : F32)
          spin(q, Bool.pick(Pool, left, l, r), Bool.pick(F32, left, ticket, (ticket - mass(l) : F32)))
`

const mutateCode = `
def gauss(+s: U32) -> F32:   # Box-Muller: a standard normal sample
  +u = (1.0 - R.random(s) : F32)
  +v = R.random(R.hash((s + 2246822519 : U32)))
  (F32.sqrt((0.0 - 2.0 * F32.log(u) : F32)) * F32.cos((6.2831855 * v : F32)) : F32)

# Each number moves by a gaussian step with probability \`rate\`.
def nudge(+x: F32, +rate: F32, +strength: F32, +s: U32) -> F32:
  Bool.pick(F32, (R.random(s) < rate : F32), (x + gauss(R.hash((s + 7 : U32))) * strength : F32), x)
`

const parallelCode = `
# One frame for the whole flock: the two halves are independent.
def advance(f: Flock, +t: U32, +course: U32) -> Flock:
  match f:
    case Leaf{b}: Leaf{step(b, t, course)}
    case Node{l, r}:
      a b = advance(l, t, course) advance(r, t, course)
      Node{a, b}

# The next generation, bred leaf by leaf; both halves in parallel.
def breed(+d: Nat, ..., +offset: U32) -> Flock:
  match d:
    case 0n: Leaf{newborn((offset < count : U32), elites, pool, ...)}
    case 1n+ +p:
      a b = breed(p, ..., offset) breed(p, ..., (offset + (1 << p) : U32))
      Node{a, b}
`

// Measured with flock.bend in node: seeds 1..20, 128 birds, strength 0.2,
// up to 40 generations, "solved" = the best bird flies all 2000 frames.
const RATES = [
  { label: '0', value: 0 }, { label: '0.02', value: 10 }, { label: '0.1', value: 15 },
  { label: '0.25', value: 20, hi: true }, { label: '0.5', value: 18 }, { label: '1.0', value: 20 },
].map(d => ({ ...d, display: `${d.value}/20` }))

const SPEEDS = [['1', '1×'], ['2', '2×'], ['4', '4×'], ['8', '8×'], ['16', '16×']] as const
type Speed = typeof SPEEDS[number][0]

function FlockFigure() {
  const { call, error } = useWorker(() => new Worker(new URL('./neuroevolution.worker.ts', import.meta.url), { type: 'module' }))
  const [frame, setFrame] = useState<Frame | null>(null)
  const [playing, setPlaying] = useState(() => !matchMedia('(prefers-reduced-motion: reduce)').matches)
  const [speed, setSpeed] = useState<Speed>('2'), [rate, setRate] = useState(0.25), [seed, setSeed] = useState(42)
  const [skipping, setSkipping] = useState(0), [visible, setVisible] = useState(false)
  const busy = useRef(false), box = useRef<HTMLDivElement>(null), skipLeft = useRef(0)

  const send = useCallback(async (body: Record<string, unknown>) => {
    busy.current = true
    try { setFrame(await call<Frame>(body)) } catch { /* surfaced through error */ } finally { busy.current = false }
  }, [call])

  useEffect(() => { send({ op: 'reset', seed: 42, rate: 0.25 }) }, [send])
  useEffect(() => {
    const el = box.current; if (!el) return
    const o = new IntersectionObserver(([e]) => setVisible(e.isIntersecting), { threshold: .1 }); o.observe(el); return () => o.disconnect()
  }, [])

  // The animation: each display frame asks Bend for `speed` more world frames.
  useEffect(() => {
    if (!playing || !visible || skipping) return
    let raf = 0
    const tick = () => { if (!busy.current) send({ op: 'step', frames: Number(speed) }); raf = requestAnimationFrame(tick) }
    raf = requestAnimationFrame(tick); return () => cancelAnimationFrame(raf)
  }, [playing, visible, skipping, speed, send])

  // Fast-forward: whole generations in Bend, one per call, nothing drawn but the chart.
  const skip = async (n: number) => {
    skipLeft.current = n; setSkipping(n)
    while (skipLeft.current > 0) {
      while (busy.current) await new Promise(r => setTimeout(r, 5))
      await send({ op: 'skip' }); skipLeft.current--; setSkipping(skipLeft.current)
    }
  }
  const reset = (s: number) => { skipLeft.current = 0; setSkipping(0); setSeed(s); send({ op: 'reset', seed: s, rate }) }

  const world = usePaint(c => drawWorld(c, frame))
  const brain = usePaint(c => drawNet(c, {
    sizes: [5, 8, 2], inLabels: ['dx', 'top', 'bottom', 'y', 'vy'], outLabels: ['no', 'FLAP'],
    weights: frame?.weights, acts: frame ? [frame.inputs, frame.hidden, frame.out] : undefined,
    pick: frame ? (frame.out[1] > frame.out[0] ? 1 : 0) : undefined,
  }))
  const chart = usePaint(c => drawHistory(c, frame?.best ?? new Float32Array(), frame?.avg ?? new Float32Array(), 25))
  const leaderOut = frame ? `${frame.out[0].toFixed(2)} / ${frame.out[1].toFixed(2)}` : null

  return <div ref={box}>
    <Figure caption={<>128 birds fly the same course. Each faint dot is one brain; the <b>violet</b> bird is the leader, and the dashed lines are what it sees. A generation ends when every bird crashes, or after 2,000 frames (25 pipes).</>}>
      <Stage aspect="16 / 10"><canvas ref={world} role="img" aria-label={frame ? `Generation ${frame.generation}: ${frame.living} of 128 birds alive after ${frame.pipesNow} pipes` : 'Flappy world loading'} /></Stage>
      <Controls>
        <Btn onClick={() => setPlaying(p => !p)} pressed={playing}>{playing ? 'pause' : 'play'}</Btn>
        <Choice label="speed" value={speed} options={SPEEDS} onChange={setSpeed} />
        <Btn ghost onClick={() => skip(1)} disabled={!!skipping}>skip 1 gen</Btn>
        <Btn ghost onClick={() => skip(10)} disabled={!!skipping}>{skipping ? `skipping… ${skipping}` : 'skip 10 gens'}</Btn>
      </Controls>
      <Controls>
        <Range label="mutation rate" value={rate} min={0} max={1} step={0.01} onChange={v => { setRate(v); call({ op: 'rate', rate: v }).catch(() => {}) }} format={v => v.toFixed(2)} />
        <Range label="seed" value={seed} min={1} max={99} onChange={reset} />
        <Btn ghost onClick={() => reset(seed)}>restart</Btn>
      </Controls>
      <Stats>
        <Stat label="generation" value={frame?.generation} />
        <Stat label="alive" value={frame ? `${frame.living}/128` : null} />
        <Stat label="best ever" value={frame ? `${Math.max(frame.champion, frame.pipesNow)} pipes` : null} />
        <Stat label="Bend" value={frame ? `${frame.computeMs.toFixed(1)} ms` : null} />
      </Stats>
      {error && <p className="note" role="alert">{error}</p>}
    </Figure>
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(17em, 1fr))', columnGap: '1.2em' }}>
      <Figure caption={<>The leader&apos;s brain, live. Edges are weights (<span style={{ color: 'var(--grn)' }}>green +</span>, <span style={{ color: 'var(--red)' }}>red −</span>); neurons light up with their activation. Outputs no / flap: <b>{leaderOut ?? '—'}</b></>}>
        <Stage aspect="16 / 10"><canvas ref={brain} role="img" aria-label="The leading bird's 5-8-2 neural network with weights and activations" /></Stage>
      </Figure>
      <Figure caption="Pipes cleared per generation: the best bird and the flock's average.">
        <Stage aspect="16 / 9"><canvas ref={chart} role="img" aria-label={frame?.best.length ? `Best pipes per generation: ${Array.from(frame.best).join(', ')}` : 'No finished generations yet'} /></Stage>
      </Figure>
    </div>
  </div>
}

function CarNet() {
  const ref = usePaint(c => drawNet(c, { sizes: [5, 8, 3], inLabels: ['my lane', 'clear 0', 'clear 1', 'clear 2', 'clear 3'], outLabels: ['LEFT', 'STAY', 'RIGHT'] }))
  return <Figure caption="CarCrush's brain: where am I, and how clear is each lane ahead? Then left, stay, or right.">
    <Stage aspect="16 / 9"><canvas ref={ref} role="img" aria-label="Diagram of a 5-8-3 network: lane and four lane clearances in, left, stay, right out" /></Stage>
  </Figure>
}

export default function Neuroevolution() {
  return (
    <DocPage id="neuroevolution" kicker="flappIA · CarCrush" title="Neuroevolution"
      sub={<>brains that are <b>bred</b>, not <b>trained</b></>}
      tag={<>a flock of tiny networks, computed in <b>Bend</b></>}>
      <p>Most neural networks learn by gradient descent: show an example, measure the
        error, nudge every weight downhill. That needs labels and a differentiable loss.
        A bird learning to fly has neither. Nobody knows the right flap for every frame.</p>
      <p>Neuroevolution skips the calculus. Make a crowd of random brains, let each one
        try, keep the ones that did best, copy them with small random changes, repeat.
        It is natural selection with a fitness function.</p>

      <Section n={1} title={<>Breed, don&apos;t <b>TRAIN</b>.</>}>
        <p className="lead">A genome is a list of numbers. A brain is what those numbers compute.</p>
        <p>Each bird in <a href="https://github.com/krymancer/flappIA">flappIA</a> carries a
          network with <b>5</b> inputs, <b>8</b> hidden neurons and <b>2</b> outputs. Every
          neuron is a bias plus one weight per input, squashed by a sigmoid. That is
          8×(5+1) + 2×(8+1) = <b>66</b> numbers. Those 66 numbers are the whole genome.</p>
        <Code file="demos/neuroevolution/flock.bend">{brainCode}</Code>
        <p>There is no backward pass, no learning rate, no loss. The network only ever
          runs forward. Learning happens between generations, to the population, not
          inside any one brain.</p>
      </Section>

      <Section n={2} title={<>A flock <b>FLIES</b>.</>}>
        <p>The world below is a simplified flappIA, written in Bend: gravity pulls each
          bird down, a flap sets its velocity upward, pipes scroll left, and touching a
          pipe, the floor or the ceiling ends that bird&apos;s run. Every frame, every
          living bird reads five numbers and asks its own brain whether to flap.</p>
        <ul>
          <li><b>dx</b>: horizontal distance to the next pipe</li>
          <li><b>top</b>, <b>bottom</b>: the edges of its gap</li>
          <li><b>y</b>, <b>vy</b>: the bird&apos;s height and velocity</li>
        </ul>
        <p>It flaps when output 1 beats output 0. Its score is the number of frames it
          survived.</p>
        <FlockFigure />
        <p>Generation 0 is pure noise: some birds flap constantly into the ceiling, most
          never flap at all. Within a few generations one lineage threads the first gap,
          and soon after the best bird flies the whole course. Each generation flies a new
          random course, so a brain has to generalize, not memorize. Skip ahead ten
          generations and watch the leader&apos;s brain as it nears a gap: the two outputs
          trade places right where it needs to flap.</p>
        <Code file="demos/neuroevolution/flock.bend">{stepCode}</Code>
      </Section>

      <Section n={3} title={<>Selection, then <b>MUTATION</b>.</>}>
        <p className="lead">The fit get more children. The children are slightly wrong copies.</p>
        <p>When the last bird crashes, flappIA weighs each one by its score cubed. Cubing
          sharpens the pressure: a bird that flew twice as far gets eight times the
          chance to be a parent. Then it spins a roulette wheel once per child. Here the
          wheel is a tree whose nodes hold partial sums, so a spin walks from the root to
          a leaf.</p>
        <Code file="demos/neuroevolution/flock.bend">{selectCode}</Code>
        <p>Each child is a copy of its one parent, and each of its 66 numbers has a
          chance, the <b>mutation rate</b>, of moving by a gaussian step of size 0.2. The
          best four brains also pass through untouched (elitism), so a good solution is
          never lost to bad luck.</p>
        <Code file="demos/neuroevolution/flock.bend">{mutateCode}</Code>
        <p>There is no crossover here, as in flappIA and CarCrush: one parent per child.
          Mixing two parents&apos; genes is the star of the <a href="#shakespeare">Shakespeare</a> page,
          where each gene is a letter and halves of good strings combine well. Halves of
          two networks rarely do, because hidden neurons in different brains need not mean
          the same thing.</p>
        <Bars title="seeds that fly the full course within 40 generations, by mutation rate" data={RATES} max={20}
          caption={<>Measured with flock.bend over seeds 1–20. Without mutation, selection can only pick among the first 128 random brains, and none can fly. Too little mutation and evolution crawls. flappIA&apos;s 0.25 solved every seed.</>} />
      </Section>

      <Section n={4} title={<>Same brain, new <b>BODY</b>.</>}>
        <p>The algorithm does not care what the brain controls. In <a href="https://github.com/krymancer/carcrush-js-ai">CarCrush</a> a
          car drives up a four-lane road while traffic comes down at it. Its network
          reads <b>5</b> numbers: its own lane, and how much clear road is ahead in each
          of the four lanes. <b>8</b> hidden neurons feed <b>3</b> actions: left, stay or
          right, whichever output is largest.</p>
        <CarNet />
        <p>Everything else is the flappIA loop: 1,000 cars, fitness raised to the third
          power, the 50 best brains carried over unchanged, every other child a mutated copy of one roulette pick.
          Fitness counts survival time plus a bonus of 100 for every car passed, so a
          brain that merely hides is beaten by one that dodges. <a href="https://krymancer.github.io/carcrush-js-ai/">Play it live</a>.</p>
      </Section>

      <Section n={5} title={<>Why <b>BEND</b>.</>}>
        <p className="lead">Every bird is independent. So is every child.</p>
        <p>Within a frame no bird affects another: they share the course and nothing else.
          Breeding is the same: each child needs the wheel and its own random seed. So the
          flock is a binary tree, and every pass over it is Bend&apos;s parallel call,
          two halves at once, all the way down.</p>
        <Code file="demos/neuroevolution/flock.bend">{parallelCode}</Code>
        <p>A line like <code>a b = advance(l, t, course) advance(r, t, course)</code> promises
          Bend the calls are independent and equally sized. Because the language is pure
          and affine, the first promise is checked for free; the balanced tree keeps the
          second. Compiled natively, the 128 birds spread over every core, or onto the GPU
          with a <code>!</code> on the call. In your browser the same code runs as plain JavaScript,
          sequentially, and still steps a whole flock in about a millisecond.</p>
        <p>Neuroevolution scales with the population, not with a gradient. Gradient
          training is the other road: see <a href="#neural-networks">neural networks</a> for
          the same kind of brain learning from labelled examples, one derivative at a time.</p>
      </Section>
    </DocPage>
  )
}
