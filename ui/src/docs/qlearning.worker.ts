import { array, list, serve, type Bend } from '@/lib/bend-worker'
// Marshals requests to demos/qlearning/agent.bend. Every Q value, policy arrow,
// path, return and value-iteration sweep is computed by Bend; this worker only
// holds the Bend values between calls and flattens them into typed arrays.

const nil = { $: 'Nil' }
type Agent = { q: any; world: any; start: number; seed: number; episodes: number; steps: number; falls?: number }
const agents = new Map<string, Agent>()
const plans = new Map<string, { v: any; world: any; start: number; sweeps: number }>()

const tree = (kinds: number[], d = 6, lo = 0): any => d === 0 ? { $: 'Cell', kind: kinds[lo] } : { $: 'Split', l: tree(kinds, d - 1, lo), r: tree(kinds, d - 1, lo + (1 << (d - 1))) }
const mix = (seed: number, n: number) => (seed ^ Math.imul(n + 1, 0x9e3779b1)) >>> 0

function flat(spots: any[]) {
  const value = new Float32Array(64), action = new Uint8Array(64), spread = new Float32Array(64)
  spots.forEach((p, i) => { value[i] = p.value; action[i] = p.action; spread[i] = p.spread })
  return { value, action, spread }
}
const policy = (b: Bend, a: Agent) => flat(array(b.policy(64n, a.q, 0)))
const greedy = (b: Bend, a: Agent) => Int16Array.from(array(b.walk(64n, true, a.q, a.world, a.start, list([a.start]))))
const status = (b: Bend, a: Agent) => ({ ...policy(b, a), route: greedy(b, a), episodes: a.episodes, steps: a.steps })

function trainBatch(b: Bend, a: Agent, n: number, d: any) {
  const r = b.train(BigInt(n), a.q, a.world, a.start, d.alpha, d.gamma, d.eps, d.cost, mix(a.seed, a.episodes), nil, 0, nil)
  a.q = r.q; a.episodes += n; a.steps += r.steps
  return { returns: Float32Array.from(array(r.returns)), path: Int16Array.from(array(r.path)) }
}

serve('qlearning', (d, b) => {
  switch (d.action) {
    case 'reset': {
      const a = { q: b.zeros(8n), world: tree(d.kinds), start: d.start, seed: d.seed >>> 0, episodes: 0, steps: 0 }
      agents.set(d.key, a); return status(b, a)
    }
    case 'world': {
      const a = agents.get(d.key)!; a.world = tree(d.kinds); a.start = d.start; return status(b, a)
    }
    case 'train': {
      const a = agents.get(d.key)!, batch = trainBatch(b, a, d.n, d); return { ...batch, ...status(b, a) }
    }
    case 'plan-reset': {
      const p = { v: b.zeros(6n), world: tree(d.kinds), start: d.start, sweeps: 0 }; plans.set(d.key, p)
      return { ...flat(array(b.values(64n, p.v, p.world, 0, d.gamma, d.cost))), sweeps: 0, delta: Infinity, route: new Int16Array([p.start]) }
    }
    case 'sweep': {
      // One Bellman sweep; `sweep` splits the grid into independent halves.
      const p = plans.get(d.key)!, next = b.sweep(6n, 0, p.v, p.world, d.gamma, d.cost), delta = b.change(next, p.v)
      p.v = next; p.sweeps++
      return { ...flat(array(b.values(64n, p.v, p.world, 0, d.gamma, d.cost))), sweeps: p.sweeps, delta, route: Int16Array.from(array(b.plan(64n, true, p.v, p.world, p.start, d.gamma, d.cost, list([p.start])))) }
    }
    case 'solve': {
      // Value iteration to convergence; note the first sweep whose plan is already the final route.
      const world = tree(d.kinds), routes: number[][] = []
      let v = b.zeros(6n), sweeps = 0, delta = Infinity
      while (delta > 1e-4 && sweeps < 200) {
        const next = b.sweep(6n, 0, v, world, d.gamma, d.cost); delta = b.change(next, v); v = next; sweeps++
        routes.push(array(b.plan(64n, true, v, world, d.start, d.gamma, d.cost, list([d.start]))))
      }
      const final = routes[routes.length - 1], same = (r: number[]) => r.length === final.length && r.every((x, i) => x === final[i])
      let first = routes.length; while (first > 1 && same(routes[first - 2])) first--
      return { sweeps, routeSweeps: first, routeLength: final.length - 1, end: final[final.length - 1] }
    }
    case 'race': {
      // Q-learning from scratch, a chunk per call, until its greedy route is as short as the plan's.
      if (d.fresh) agents.set(d.key, { q: b.zeros(8n), world: tree(d.kinds), start: d.start, seed: d.seed >>> 0, episodes: 0, steps: 0 })
      const a = agents.get(d.key)!
      const { returns } = trainBatch(b, a, d.n, d)
      // An episode whose return is at most -1 ended in a pit (or ran out of time).
      a.falls = (a.falls ?? 0) + returns.filter(r => r <= -0.999).length
      const route = greedy(b, a)
      return { episodes: a.episodes, steps: a.steps, falls: a.falls, routeLength: route.length - 1, end: route[route.length - 1] }
    }
  }
  throw new Error('Unknown request ' + d.action)
})
