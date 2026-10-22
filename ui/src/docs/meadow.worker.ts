import { array, serve, type Bend } from '@/lib/bend-worker'
import { STRIDE } from './meadow-draw'
// Every rule of the meadow lives in demos/meadow/ecology.bend. This worker keeps
// one Bend World per figure, asks Bend to advance it, and flattens the result
// into typed arrays for drawing.
const worlds = new Map<string, any>()
const checkpoint = 600 // ticks between fingerprints (60 simulated seconds)
const prints = new Map<string, [number, number][]>()

function field(g: any): Float32Array {
  const out = new Float32Array(2304)
  const walk = (n: any, off: number, size: number) => {
    if (n.$ === 'Cell') out[off] = n.grass
    else if (n.$ === 'Fork') { walk(n.left, off, size / 2); walk(n.right, off + size / 2, size / 2) }
  }
  walk(g, 0, 4096)
  return out
}

// Per sheep: x, z, heading, energy, speed, metabolism, wool, flags (act | lamb << 2 | hungry << 3).
function flock(w: any): Float32Array {
  const xs = array(w.flock), out = new Float32Array(xs.length * STRIDE)
  xs.forEach((s: any, i: number) => out.set([s.x, s.z, s.heading, s.energy, s.speed, s.metab, s.wool, s.act | (s.age < 45 ? 4 : 0) | (s.hungry ? 8 : 0)], i * STRIDE))
  return out
}

function frame(name: string, bend: Bend, w: any, extra: boolean) {
  const census = bend.survey(w), n = census.n
  const out: Record<string, unknown> = {
    grass: field(w.grass), sheep: flock(w), tick: w.tick, time: w.tick / 10, population: n,
    founders: w.founders, births: w.births, old: w.old, starved: w.starved, halted: w.halted,
    food: bend.total(w.grass), speed: n ? census.speed / n : 0, metab: n ? census.metab / n : 0,
    lambs: census.lambs, hungry: census.hungry, generation: census.gen, prints: prints.get(name) ?? [],
  }
  if (extra) { out.speeds = array(bend.traits(w, 0)); out.metabs = array(bend.traits(w, 1)) }
  return out
}

serve('meadow', (d, bend) => {
  const name: string = d.world
  if (d.op === 'create') {
    const w = d.lone ? bend.lone(d.seed >>> 0, d.food, d.regrow) : bend.create(d.seed >>> 0, d.population, d.food, d.regrow, d.metab)
    worlds.set(name, w); prints.set(name, [])
    return frame(name, bend, w, true)
  }
  let w = worlds.get(name)
  if (!w) throw new Error(`No meadow named ${name}.`)
  if (d.op === 'paint') w = bend.paint(w, d.x, d.z, d.radius, d.amount)
  if (d.op === 'tune') w = bend.tune(w, d.regrow, d.metab)
  if (d.op === 'step') {
    // Advance up to d.ticks, stopping early once the time budget is spent, and
    // never stepping over a checkpoint so its fingerprint is exact.
    const start = performance.now(), list = prints.get(name)!
    let left = d.ticks as number
    while (left > 0 && !w.halted) {
      const chunk = Math.min(left, 5, checkpoint - (w.tick % checkpoint))
      w = bend.run(BigInt(chunk), w); left -= chunk
      if (w.tick % checkpoint === 0 && list.at(-1)?.[0] !== w.tick) list.push([w.tick, bend.fingerprint(w) >>> 0])
      if (performance.now() - start > (d.budget ?? 40)) break
    }
  }
  worlds.set(name, w)
  return frame(name, bend, w, d.op === 'step' ? !!d.extra : true)
})
