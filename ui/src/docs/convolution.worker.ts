import { array, list, serve, type Bend } from '@/lib/bend-worker'
// Marshals requests to demos/convolution/conv.bend. Convolution, ReLU,
// pooling, the window products, training noise and gradient descent all run
// in Bend; this worker only builds argument lists, keeps Bend values between
// calls, and flattens the result trees into typed arrays.

const N = 28, D = 10n            // 28 x 28 input, 1024-leaf tree
const C = N - 2, CD = 10n        // conv map 26 x 26
const Q = C / 2, QD = 8n         // pooled map 13 x 13
const R = Q - 2, RD = 7n         // layer-2 map 11 x 11
const T = 20, TD = 9n, TOD = 9n // training noise 20 x 20 -> 18 x 18

// Flattens the first n leaves of a Bend Img tree.
function flat(m: any, n: number) {
  const out = new Float32Array(n), stack = [m]; let i = 0
  while (stack.length && i < n) { const t = stack.pop(); if (t.$ === 'Px') out[i++] = t.v; else stack.push(t.hi, t.lo) }
  return out
}
const one = (x: any) => list([x])

let image: { ver: number; tree: any } | null = null
const input = (bend: Bend, d: { ver: number; img: Float32Array }) => {
  if (image?.ver !== d.ver) image = { ver: d.ver, tree: bend.from_list(D, list(Array.from(d.img))) }
  return image.tree
}

let fit: { noise: any; target: any; targetK: number[]; k: any; step: number } | null = null

serve('convolution', (d, bend) => {
  switch (d.action) {
    case 'conv': {                // image -> conv -> ReLU -> pool
      const conv = bend.conv(CD, 0, one(input(bend, d)), one(list(d.kernel)), D, N, d.bias ?? 0)
      const relu = bend.relu(conv)
      return { conv: flat(conv, C * C), relu: flat(relu, C * C), pool: flat(bend.pool(QD, 0, relu, CD, C), Q * Q) }
    }
    case 'products': {            // the nine terms of one window
      const ps = array(bend.products(list(d.kernel), 0, input(bend, d), D, N, d.x, d.y))
      return { products: Float32Array.from(ps) }
    }
    case 'layer': {               // 8 filters -> ReLU -> pool, then 4 corner filters over the edge maps
      const filter = (ks: number[][], bias: number) => ({ $: 'Filter', ks: list(ks.map(k => list(k))), bias })
      const l1 = bend.pool_all(bend.layer(list(d.layer1.map((k: number[]) => filter([k], 0))), one(input(bend, d)), D, N, CD), CD, C, QD)
      const a = array(l1), edges = list(a.slice(0, 4)) // layer 2 reads the four edge maps
      const l2 = bend.layer(list(d.layer2.map((f: { ks: number[][]; bias: number }) => filter(f.ks, f.bias))), edges, QD, Q, RD), b = array(l2), first = new Float32Array(a.length * Q * Q), second = new Float32Array(b.length * R * R)
      a.forEach((m, i) => first.set(flat(m, Q * Q), i * Q * Q)); b.forEach((m, i) => second.set(flat(m, R * R), i * R * R))
      return { first, second }
    }
    case 'fit-reset': {           // fresh noise, the target filter's answer on it, and a zero kernel
      const noise = bend.noise(TD, 0, T * T, d.seed >>> 0)
      fit = { noise, targetK: d.target, target: bend.conv(TOD, 0, one(noise), one(list(d.target)), TD, T, 0), k: list(Array(9).fill(0)), step: 0 }
      return { kernel: new Float32Array(9), losses: new Float32Array(0), step: 0, noise: flat(noise, T * T) }
    }
    case 'fit': {                 // `steps` gradient-descent updates, then both kernels applied to the reader's image
      if (!fit) throw new Error('Training was not reset.')
      const f = bend.fit(BigInt(d.steps), fit.noise, TD, T, TOD, fit.target, fit.k, d.rate, { $: 'Nil' })
      fit.k = f.kernel; fit.step += d.steps
      const img = input(bend, d), learned = bend.conv(CD, 0, one(img), one(fit.k), D, N, 0)
      const wanted = bend.conv(CD, 0, one(img), one(list(fit.targetK)), D, N, 0)
      return { kernel: Float32Array.from(array(fit.k)), losses: Float32Array.from(array(f.losses).reverse()), step: fit.step, learned: flat(learned, C * C), wanted: flat(wanted, C * C) }
    }
  }
  throw new Error(`Unknown action ${d.action}`)
})
