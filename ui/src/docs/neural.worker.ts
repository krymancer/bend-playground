import { array, serve, type Bend } from '@/lib/bend-worker'
// Marshals requests to demos/neural/net.bend. Every number drawn on the page
// (activations, predictions, gradients, updated weights) is computed by Bend;
// this worker only holds the Bend values between calls and flattens them.

const DEPTH = 8n, COUNT = 256
type Unit = { wx: number; wy: number; b: number; v: number }
let batch: any = null, fit: any = null, epoch = 0
let cap: { batch: any; fit: any } | null = null

// Sample a scalar field over [-1, 1]^2 one Bend row at a time (pixel centres, y up).
function field(n: number, row: (y: number, x0: number, dx: number) => any) {
  const out = new Float32Array(n * n), dx = 2 / n
  for (let j = 0; j < n; j++) out.set(array(row(1 - (j + 0.5) * dx, -1 + dx / 2, dx)), j * n)
  return out
}
const leaves = (b: any): any[] => { const out: any[] = [], stack = [b]; while (stack.length) { const t = stack.pop(); if (t.$ === 'One') out.push(t.p); else stack.push(t.r, t.l) } return out }
const points = (b: any) => { const ps = leaves(b), out = new Float32Array(ps.length * 3); ps.forEach((p, i) => { out[i * 3] = p.x; out[i * 3 + 1] = p.y; out[i * 3 + 2] = p.t }); return out }
const units = (net: any): Unit[] => array(net.units).map(({ wx, wy, b, v }: Unit) => ({ wx, wy, b, v }))
const unitFields = (bend: Bend, us: Unit[], n: number) => us.map(u => field(n, (y, x0, dx) => bend.neuron_row(BigInt(n), 1n, u.wx, u.wy, u.b, y, x0, dx)))

function snapshot(bend: Bend, res: number, small: number) {
  const us = units(fit.net)
  return { epoch, loss: fit.loss, acc: fit.acc, c: fit.net.c, units: us, field: field(res, (y, x0, dx) => bend.row(BigInt(res), fit.net, y, x0, dx)), hidden: unitFields(bend, us, small) }
}

serve('neural', (d, bend) => {
  switch (d.action) {
    case 'neuron': {
      const kind = BigInt(d.kind), n = d.res
      const own = bend.build(7n, 0n, 0, d.seed)
      return { field: field(n, (y, x0, dx) => bend.neuron_row(BigInt(n), kind, d.w1, d.w2, d.b, y, x0, dx)), points: points(own), hits: bend.perceptron(own, d.w1, d.w2, d.b), total: 128,
        curve: new Float32Array(array(bend.neuron_row(65n, kind, 6, 0, 0, 0, -1, 2 / 64))) }
    }
    case 'xor': {
      const net = bend.xor_net(d.k), data = bend.build(7n, 1n, 0, d.seed), n = d.res
      return { hidden: unitFields(bend, units(net), n), field: field(n, (y, x0, dx) => bend.row(BigInt(n), net, y, x0, dx)), points: points(data), acc: bend.score(net, data, 128).acc }
    }
    case 'reset': {
      batch = bend.build(DEPTH, BigInt(d.kind), 0, d.seed); epoch = 0
      fit = bend.score(bend.init(BigInt(d.hidden), d.init), batch, COUNT)
      return { ...snapshot(bend, d.res, d.small), points: points(batch) }
    }
    case 'train': {
      if (!fit) throw new Error('Reset the network first.')
      fit = bend.train(BigInt(d.epochs), fit, batch, d.lr, COUNT); epoch += d.epochs
      return snapshot(bend, d.res, d.small)
    }
    // Capacity experiment: the spiral with a given number of hidden neurons.
    case 'cap-reset': {
      const data = bend.build(DEPTH, 3n, 0, 42)
      cap = { batch: data, fit: bend.score(bend.init(BigInt(d.hidden), 7), data, COUNT) }
      return { acc: cap.fit.acc, loss: cap.fit.loss }
    }
    case 'cap-train': {
      if (!cap) throw new Error('Start the capacity run first.')
      cap.fit = bend.train(BigInt(d.epochs), cap.fit, cap.batch, 2, COUNT)
      return { acc: cap.fit.acc, loss: cap.fit.loss }
    }
  }
  throw new Error(`Unknown neural request: ${d.action}`)
})
