import test from 'node:test';
import assert from 'node:assert/strict';
import M from '../demos/deepfake/meso.bend';

// An independent float64 reference of the toy detector: the face generator,
// 3x3 windows, conv -> ReLU -> mean pool -> logistic, its gradient, and Adam.
const array = xs => { const out = []; for (let x = xs; x.$ === 'Con'; x = x.tail) out.push(x.head); return out; };
const near = (a, b, e, what = '') => assert.ok(Math.abs(a - b) <= e, `${what} ${a} != ${b}`);
const hash = x => { x >>>= 0; const a = Math.imul((x ^ (x >>> 16)) >>> 0, 2146121005) >>> 0, b = Math.imul((a ^ (a >>> 15)) >>> 0, 2221713035) >>> 0; return (b ^ (b >>> 16)) >>> 0; };
const random = s => (hash(s >>> 0) >>> 8) / 16777216;
const clamp = x => Math.max(0, Math.min(1, x));
const N = 196;
function face(index, seed, amp) {
  const s = hash((index ^ hash(seed)) >>> 0), u = k => random((s + k) >>> 0), fake = index & 1;
  const cx = 8 + (u(1) - .5) * 2, cy = 8.4 + (u(2) - .5) * 1.6, rx = 4.6 + u(3) * 1.4, ry = 5.6 + u(4) * 1.4, skin = .62 + u(5) * .25, bg = .18 + u(6) * .3, gx = (u(7) - .5) * .03;
  const img = [];
  for (let y = 0; y < 16; y++) { const row = []; for (let x = 0; x < 16; x++) {
    const px = x + .5, py = y + .5, dx = (px - cx) / rx, dy = (py - cy) / ry, m = clamp((1 - (dx * dx + dy * dy)) * 4);
    const base = bg + gx * (px - 8), ey = py - (cy - .22 * ry);
    const eyes = clamp(1.6 - ((px - (cx - .42 * rx)) ** 2 + ey * ey)) + clamp(1.6 - ((px - (cx + .42 * rx)) ** 2 + ey * ey));
    const mx = (px - cx) / (.45 * rx), my = (py - (cy + .5 * ry)) / .7, mouth = clamp(1 - (mx * mx + my * my));
    const grain = (random(hash((s + 1000 + y * 16 + x) >>> 0)) - .5) * .08;
    row.push(base + m * (skin - .06 * dy - base) - .45 * m * eyes - .35 * m * mouth + grain + fake * amp * m * ((x + y) & 1 ? -1 : 1));
  } img.push(row); }
  return { img, t: fake };
}
const windows = img => { const ws = []; for (let y = 0; y < 14; y++) for (let x = 0; x < 14; x++) { const w = []; for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) w.push(img[y + j][x + i] - .5); ws.push(w); } return ws; };
const data = (offset, n, seed, amp) => Array.from({ length: n }, (_, i) => { const f = face(offset + i, seed, amp); return { ws: windows(f.img), t: f.t }; });
function init(K, seed) {
  let s = hash(seed); const p = [0];
  for (let k = 0; k < K; k++) { const w = Array.from({ length: 9 }, (_, j) => (2 * random((s + j) >>> 0) - 1) * .4), mu = w.reduce((a, b) => a + b) / 9; p.push(...w.map(x => x - mu), 0, (2 * random((s + 9) >>> 0) - 1) * .5); s = hash((s + 2654435769) >>> 0); }
  return p;
}
// Flat layout, as in meso.bend: c, then per kernel nine weights, bias, v.
function example(p, ex) {
  const K = (p.length - 1) / 11, g = Array(p.length).fill(0); let z = p[0]; const acc = [];
  for (let k = 0; k < K; k++) { const o = 1 + 11 * k; let s = 0, n = 0; const sum = Array(9).fill(0);
    for (const x of ex.ws) { let r = p[o + 9]; for (let j = 0; j < 9; j++) r += p[o + j] * x[j]; if (r > 0) { s += r; n++; for (let j = 0; j < 9; j++) sum[j] += x[j]; } }
    acc.push({ s, n, sum }); z += p[o + 10] * s / N; }
  const q = 1 / (1 + Math.exp(-z)), d = q - ex.t; g[0] = d;
  for (let k = 0; k < K; k++) { const o = 1 + 11 * k, a = d * p[o + 10] / N; for (let j = 0; j < 9; j++) g[o + j] = a * acc[k].sum[j]; g[o + 9] = a * acc[k].n; g[o + 10] = d * acc[k].s / N; }
  return { q, g, loss: Math.max(z, 0) - z * ex.t + Math.log(1 + Math.exp(-Math.abs(z))) };
}
function adamTrain(p, set, steps, lr) {
  let m = p.map(() => 0), v = p.map(() => 0); const losses = [];
  for (let t = 1; t <= steps; t++) {
    const g = p.map(() => 0); let loss = 0;
    for (const ex of set) { const r = example(p, ex); loss += r.loss; r.g.forEach((x, i) => g[i] += x / set.length); }
    losses.push(loss / set.length);
    m = m.map((x, i) => .9 * x + .1 * g[i]); v = v.map((x, i) => .999 * x + .001 * g[i] ** 2);
    p = p.map((x, i) => x - lr * (m[i] / (1 - .9 ** t)) / (Math.sqrt(v[i] / (1 - .999 ** t)) + 1e-8));
  }
  return { p, losses };
}
const flatOf = opt => array(opt.ps).map(q => q.p);
const fit = opt => ({ $: 'Fit', opt, loss: 0, acc: 0 });

test('Bend draws the same synthetic faces and windows as the reference', () => {
  for (const [index, seed, amp] of [[0, 42, .05], [5, 42, .05], [7, 1, .1], [12, 9, 0]]) {
    const bend = array(M.patch(index, seed, amp)), ref = face(index, seed, amp).img.flat();
    assert.equal(bend.length, 256);
    bend.forEach((x, i) => near(x, ref[i], 1e-5, `pixel ${i}`));
    const ex = M.example_of(index, seed, amp), ws = array(ex.ws), rw = windows(face(index, seed, amp).img);
    assert.equal(ws.length, N); assert.equal(ex.t, index & 1);
    ws.forEach((w, k) => ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'].forEach((key, j) => near(w[key], rw[k][j], 1e-5, `window ${k}`)));
  }
});

test('the artifact is only in fakes, and only inside the face', () => {
  const real = array(M.patch(4, 3, 0)), same = array(M.patch(4, 3, .1));
  assert.deepEqual(real, same, 'amplitude must not touch a real face');
  const clean = array(M.patch(5, 3, 0)), fake = array(M.patch(5, 3, .1));
  assert.ok(Math.abs(fake[0] - clean[0]) < 1e-6, 'corner pixel is background');
  assert.ok(Math.abs(fake[8 * 16 + 8] - clean[8 * 16 + 8]) > .05, 'center pixel carries the checkerboard');
});

test('initial kernels are zero-sum and match the reference initialisation', () => {
  const p = flatOf(M.init(4n, 42)), ref = init(4, 42);
  assert.equal(p.length, 45); p.forEach((x, i) => near(x, ref[i], 1e-6, `param ${i}`));
  for (let k = 0; k < 4; k++) near(p.slice(1 + 11 * k, 10 + 11 * k).reduce((a, b) => a + b), 0, 1e-5, 'kernel sum');
});

test('the Bend gradient of one example equals the reference gradient', () => {
  const opt = M.init(4n, 7), p = flatOf(opt), net = M.net_of(4n, opt.ps);
  for (const index of [2, 3]) {
    const g = M.example(net, M.example_of(index, 7, .05)), ref = example(p, data(index, 1, 7, .05)[0]);
    const bend = array(M.flat(g.net));
    bend.forEach((x, i) => near(x, ref.g[i], 1e-5 + 1e-4 * Math.abs(ref.g[i]), `grad ${i}`));
    near(g.loss, ref.loss, 1e-5, 'loss');
    near(M.predict(net, M.example_of(index, 7, .05)), ref.q, 1e-5, 'predict');
  }
});

test('Adam steps follow the reference trajectory', () => {
  const set = data(0, 16, 11, .05), batch = M.build(4n, 0, 11, .05);
  const ref = adamTrain(init(4, 11), set, 8, .03);
  let f = fit(M.init(4n, 11));
  for (let t = 0; t < 8; t++) { f = M.train(1n, 4n, f, batch, .03, 16); near(f.loss, ref.losses[t], 2e-4, `loss at step ${t}`); }
  flatOf(f.opt).forEach((x, i) => near(x, ref.p[i], 2e-3, `param ${i}`));
});

test('the toy detector learns the artifact and generalises to held-out faces', () => {
  const train = M.build(7n, 0, 42, .05), val = M.build(8n, 100000, 42, .05);
  let f = fit(M.init(4n, 42));
  f = M.train(200n, 4n, f, train, .03, 128);
  const s = M.score(4n, f.opt, val), acc = (s.tp + s.tn) / 256;
  assert.equal(s.tp + s.fp + s.fn + s.tn, 256);
  assert.ok(acc >= .9, `held-out accuracy ${acc}`);
  // Determinism: the same seed gives the same model.
  const again = M.train(200n, 4n, fit(M.init(4n, 42)), train, .03, 128);
  assert.deepEqual(flatOf(again.opt), flatOf(f.opt));
});

test('without the artifact there is nothing to learn', () => {
  const train = M.build(7n, 0, 42, 0), val = M.build(8n, 100000, 42, 0);
  const f = M.train(200n, 4n, fit(M.init(4n, 42)), train, .03, 128), s = M.score(4n, f.opt, val), acc = (s.tp + s.tn) / 256;
  assert.ok(acc < .65, `held-out accuracy ${acc} should stay near chance`);
});
