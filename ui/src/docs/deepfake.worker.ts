import { array, serve } from '@/lib/bend-worker'
// Marshals the toy detector in demos/deepfake/meso.bend. Every number below is
// computed by Bend; this worker only keeps the Bend values between calls and
// copies results into typed arrays for drawing.
const K = 4, SAMPLES = 6, TRAIN = 128, VAL = 128, LR = 0.03
const VAL_OFFSET = 100000, SLOTS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
let train: any, val: any, fit: any, samples: any[] = [], seed = 42, amp = 0.05

function snapshot(M: any) {
  const net = M.net_of(BigInt(K), fit.opt.ps), ks = array(net.ks)
  const kernels = new Float32Array(K * 9), out = new Float32Array(K), maps = new Float32Array(K * 2 * 196)
  ks.forEach((k: any, i: number) => {
    SLOTS.forEach((s, j) => { kernels[i * 9 + j] = k.w[s] })
    out[i] = k.v
    // Feature maps on the first real and the first fake sample.
    ;[0, 1].forEach(n => array(M.fmap(k.w, k.b, samples[n].ws)).forEach((x: number, j: number) => { maps[(i * 2 + n) * 196 + j] = x }))
  })
  const preds = new Float32Array(samples.map(ex => M.predict(net, ex)))
  const s = val ? M.score(BigInt(K), fit.opt, val) : null
  return { kernels, out, maps, preds, confusion: s ? new Float32Array([s.tp, s.fp, s.fn, s.tn]) : new Float32Array(4), valLoss: s ? s.loss / VAL : NaN }
}

serve('deepfake', (d, M) => {
  if (d.action === 'reset') {
    seed = d.seed; amp = d.amp; val = null
    train = M.build(7n, 0, seed, amp)
    fit = { $: 'Fit', opt: M.init(BigInt(K), seed), loss: NaN, acc: NaN }
    // Held-out faces, alternating real and fake (odd indices are fakes).
    samples = Array.from({ length: SAMPLES }, (_, i) => M.example_of(VAL_OFFSET + i, seed, amp))
    const patches = new Float32Array(SAMPLES * 256), trace = new Float32Array(array(M.trace(VAL_OFFSET + 1, seed, amp)))
    for (let i = 0; i < SAMPLES; i++) patches.set(array(M.patch(VAL_OFFSET + i, seed, amp)), i * 256)
    return { patches, trace, labels: new Float32Array(samples.map(ex => ex.t)) }
  }
  if (d.action === 'faces') {
    const patches = new Float32Array(d.count * 256)
    for (let i = 0; i < d.count; i++) patches.set(array(M.patch(VAL_OFFSET + i, d.seed, d.amp)), i * 256)
    return { patches }
  }
  // Split from reset to keep each call short.
  if (d.action === 'validation') { val = M.build(7n, VAL_OFFSET, seed, amp); return snapshot(M) }
  if (d.action === 'train') {
    fit = M.train(BigInt(d.steps), BigInt(K), fit, train, LR, TRAIN)
    return { loss: fit.loss, acc: fit.acc, ...snapshot(M) }
  }
  throw new Error(`unknown action ${d.action}`)
})
