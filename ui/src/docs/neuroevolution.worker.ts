import { array, serve, type Bend } from '@/lib/bend-worker'
// Holds the flock between calls. Bend does every frame, every forward pass and
// every generation; this file only walks the result to pack typed arrays.
const DEPTH = 7n, SIZE = 128, MAX_T = 2000, ELITE = 4, STRENGTH = 0.2

let flock: any, t = 0, generation = 0, seed = 42, rate = 0.25, course = 0, leader = 0
let best: number[] = [], avg: number[] = [], champion = 0

const leaves = (tree: any, out: any[] = []) => { if (tree.$ === 'Leaf') out.push(tree.bird); else { leaves(tree.left, out); leaves(tree.right, out) } return out }
const evolveSeed = (g: number) => (Math.imul(seed, 7919) + g) >>> 0

function reset(bend: Bend, s: number, r: number) {
  seed = s >>> 0; rate = r; generation = 0; t = 0; leader = 0; best = []; avg = []; champion = 0
  flock = bend.initial(DEPTH, seed, 0); course = bend.course(seed, 0)
}

// Close the generation: record its scores, then breed the next one in Bend.
function finish(bend: Bend) {
  const s = bend.stats(flock)
  best.push(s.best_pipes); avg.push(s.total_pipes / SIZE); champion = Math.max(champion, s.best_pipes)
  flock = bend.evolve(DEPTH, flock, ELITE, rate, STRENGTH, evolveSeed(generation))
  generation++; t = 0; leader = 0; course = bend.course(seed, generation)
}

function advance(bend: Bend, frames: number) {
  const r = bend.run(BigInt(Math.min(frames, MAX_T - t)), flock, t, course)
  flock = r.fst; t = r.snd
  return !bend.any_alive(flock) || t >= MAX_T
}

function frame(bend: Bend, ended = false) {
  const birds = leaves(flock), ys = new Float32Array(SIZE), alive = new Uint8Array(SIZE)
  birds.forEach((b, i) => { ys[i] = b.y; alive[i] = b.alive ? 1 : 0 })
  if (!alive[leader]) { const i = alive.indexOf(1); if (i >= 0) leader = i }
  const pipes = new Float32Array(array(bend.visible(t, course)).flatMap((p: any) => [p.x, p.top]))
  const lead = birds[leader], th = bend.thought(lead, t, course), brain = lead.brain
  // Weights row by row: [bias, w...] for 8 hidden neurons then 2 outputs.
  const weights = new Float32Array([...array(brain.hidden), ...array(brain.out)].flatMap((n: any) => [n.bias, ...array(n.w)]))
  const stats = bend.stats(flock)
  return {
    ys, alive, pipes, weights, leader, t, generation, ended,
    inputs: new Float32Array(array(th.inputs)), hidden: new Float32Array(array(th.hidden)), out: new Float32Array(array(th.out)),
    living: stats.alive, pipesNow: bend.next_pipe(t), best: Float32Array.from(best), avg: Float32Array.from(avg), champion,
  }
}

serve('neuroevolution', (d, bend) => {
  if (d.op === 'reset' || !flock) reset(bend, d.seed ?? seed, d.rate ?? rate)
  if (d.op === 'rate') rate = d.rate
  if (d.op === 'step') {
    if (advance(bend, d.frames)) { finish(bend); return frame(bend, true) }
  }
  if (d.op === 'skip') {
    // One generation per call, run in bounded chunks with nothing drawn.
    for (let done = false; !done;) done = advance(bend, 250)
    finish(bend)
  }
  return frame(bend)
})
