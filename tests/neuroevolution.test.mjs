import test from 'node:test';
import assert from 'node:assert/strict';
import B from '../demos/neuroevolution/flock.bend';

// An independent JavaScript model of flock.bend: same world, same hash, every
// F32 operation rounded with Math.fround just as Bend's F32 is.
const f = Math.fround;
const hash = x => { x >>>= 0; const a = Math.imul(x ^ (x >>> 16), 2146121005) >>> 0, b = Math.imul(a ^ (a >>> 15), 2221713035) >>> 0; return (b ^ (b >>> 16)) >>> 0; };
const W = { bx: 140, r: 11, g: f(0.45), lift: -7, maxv: 10, speed: 3, pw: 56, gap: 120, spacing: 230, x0: 480, ground: 380, h: 400, w: 640 };
const nextPipe = t => Math.floor(Math.max(W.bx - W.r - W.pw - W.x0 + W.speed * t, -1) / W.spacing) + 1;
const pipeX = (k, t) => W.x0 + k * W.spacing - W.speed * t;
const gapTop = (k, course) => 60 + hash((course + Math.imul(k, 2654435761)) >>> 0) % 141;
const sigmoid = x => f(1 / f(1 + f(Math.exp(f(-x)))));
const array = xs => { const out = []; for (let x = xs; x.$ === 'Con'; x = x.tail) out.push(x.head); return out; };
const list = a => a.reduceRight((tail, head) => ({ $: 'Con', head, tail }), { $: 'Nil' });
const layers = brain => [array(brain.hidden), array(brain.out)].map(l => l.map(n => ({ bias: n.bias, w: array(n.w) })));
const fire = (layer, x) => layer.map(n => sigmoid(n.w.reduce((acc, w, i) => f(acc + f(w * x[i])), n.bias)));
const senses = (y, vy, t, course) => { const k = nextPipe(t), top = gapTop(k, course); return [f((pipeX(k, t) - W.bx) / W.w), f(top / W.h), f((top + W.gap) / W.h), f(y / W.h), f(vy / W.maxv)]; };
const crashed = (y, t, course) => { const k = nextPipe(t), top = gapTop(k, course), px = pipeX(k, t); return (W.bx + W.r > px && (y - W.r < top || y + W.r > top + W.gap)) || y + W.r > W.ground || y - W.r < 0; };
const leaves = (t, out = []) => { if (t.$ === 'Leaf') out.push(t.bird); else { leaves(t.left, out); leaves(t.right, out); } return out; };
const near = (a, b, e = 1e-5) => assert.ok(Math.abs(a - b) < e, `${a} != ${b}`);

test('world geometry: next pipe, pipe positions and gaps match the reference', () => {
  for (const t of [0, 1, 10, 99, 100, 101, 177, 500, 1999]) {
    assert.equal(B.next_pipe(t), nextPipe(t), `next_pipe(${t})`);
    for (const k of [0, 1, 7]) { near(B.pipe_x(k, t), pipeX(k, t)); assert.equal(B.gap_top(k, 12345), gapTop(k, 12345)); }
  }
  // A pipe counts as cleared once its right edge is behind the bird's back.
  const t = 100; assert.ok(pipeX(0, t) + W.pw < W.bx - W.r + 1e-9 || nextPipe(t) === 0);
  const top = array(B.visible(300, 9)); assert.equal(top.length, 4); near(top[1].x, pipeX(nextPipe(300), 300));
});

test('forward pass matches an independent float64 implementation', () => {
  for (const seed of [1, 2, 3, 99]) {
    const brain = B.random_brain(seed), [hidden, out] = layers(brain);
    assert.equal(hidden.length, 8); assert.equal(out.length, 2);
    hidden.forEach(n => { assert.equal(n.w.length, 5); n.w.concat(n.bias).forEach(v => assert.ok(v >= -1 && v <= 1)); });
    out.forEach(n => assert.equal(n.w.length, 8));
    const x = [0.3, -0.2, 0.7, 0.5, -0.9];
    const ref = l => l.map(n => 1 / (1 + Math.exp(-(n.bias + n.w.reduce((s, w, i) => s + w * xs(l)[i], 0)))));
    let xs = () => x; const h = ref(hidden); xs = () => h; const o = ref(out);
    array(B.forward(brain, list(x))).forEach((v, i) => near(v, o[i]));
    const th = B.thought({ $: 'Bird', alive: true, y: 200, vy: 1, score: 0, brain }, 50, 7);
    const sx = senses(200, 1, 50, 7); array(th.inputs).forEach((v, i) => near(v, sx[i]));
    array(th.hidden).forEach((v, i) => near(v, fire(hidden, sx)[i]));
  }
});

test('a whole generation flies frame-for-frame like the reference simulation', () => {
  const depth = 4n, seed = 5, course = B.course(seed, 0);
  let flock = B.initial(depth, seed, 0);
  const birds = leaves(flock).map(b => ({ y: b.y, vy: b.vy, alive: true, score: 0, brain: layers(b.brain) }));
  let t = 0;
  for (; t < 2000 && birds.some(b => b.alive); t++) for (const b of birds) {
    if (!b.alive) continue;
    const [h, o] = b.brain, out = fire(o, fire(h, senses(b.y, b.vy, t, course))), flap = out[1] > out[0];
    b.vy = Math.min(f((flap ? W.lift : b.vy) + W.g), W.maxv); b.y = f(b.y + b.vy); b.score++;
    b.alive = !crashed(b.y, t + 1, course);
  }
  const { fst, snd } = B.run(5000n, flock, 0, course);
  assert.equal(snd, t);
  leaves(fst).forEach((b, i) => { assert.equal(b.score, birds[i].score, `bird ${i}`); assert.equal(b.alive, birds[i].alive); near(b.y, birds[i].y); });
  const s = B.stats(fst);
  assert.equal(s.best, Math.max(...birds.map(b => b.score)));
  assert.equal(s.total, birds.reduce((a, b) => a + b.score, 0));
  assert.equal(s.total_pipes, birds.reduce((a, b) => a + nextPipe(b.score), 0));
  assert.equal(s.alive, birds.filter(b => b.alive).length);
});

test('roulette selection walks the partial-sum tree like a linear cumulative scan', () => {
  const scores = [10, 250, 40, 0, 120, 75, 300, 5];
  const bird = (score, i) => ({ $: 'Leaf', bird: { $: 'Bird', alive: false, y: 0, vy: 0, score, brain: { $: 'Brain', hidden: list([{ $: 'Neuron', bias: i, w: list([]) }]), out: list([]) } } });
  const tree = (lo, hi) => hi - lo === 1 ? bird(scores[lo], lo) : { $: 'Node', left: tree(lo, (lo + hi) / 2), right: tree((lo + hi) / 2, hi) };
  const pool = B.wheel(tree(0, 8)), weights = scores.map(s => (s / 100) ** 3), total = weights.reduce((a, b) => a + b);
  near(B.mass(pool), total, 1e-4);
  for (let i = 0; i < 200; i++) {
    const ticket = f(total * i / 200);
    let k = 0, acc = 0; while (k < 7 && ticket >= acc + weights[k] - 1e-6) acc += weights[k++];
    const got = array(B.spin(3n, pool, ticket).hidden)[0].bias;
    if (Math.abs(ticket - acc) > 1e-4 && Math.abs(ticket - acc - weights[k]) > 1e-4) assert.equal(got, k, `ticket ${ticket}`);
  }
  assert.notEqual(array(B.spin(3n, pool, 0).hidden)[0].bias, 3, 'a zero-score bird is never picked');
});

test('mutation: rate 0 copies, rate 1 perturbs every number by a unit gaussian step', () => {
  const brain = B.random_brain(11), flat = b => layers(b).flat().flatMap(n => [n.bias, ...n.w]);
  assert.deepEqual(flat(B.mutate(brain, 0, 0.2, 3)), flat(brain));
  const steps = [];
  for (let s = 0; s < 60; s++) { const a = flat(brain), b = flat(B.mutate(brain, 1, 1, s)); a.forEach((v, i) => steps.push(b[i] - v)); }
  assert.ok(steps.every(d => d !== 0));
  const mean = steps.reduce((a, b) => a + b) / steps.length, sd = Math.sqrt(steps.reduce((a, b) => a + (b - mean) ** 2, 0) / steps.length);
  near(mean, 0, 0.06); near(sd, 1, 0.06);
  const half = flat(B.mutate(brain, 0.25, 0.2, 9)).filter((v, i) => v !== flat(brain)[i]).length;
  assert.ok(half > 66 * 0.1 && half < 66 * 0.45, `${half} of 66 genes mutated at rate 0.25`);
});

test('evolution keeps the elites unchanged and breeds children of the fittest', () => {
  const depth = 5n, course = B.course(3, 0), flown = B.run(3000n, B.initial(depth, 3, 0), 0, course).fst;
  const old = leaves(flown), ranked = [...old].sort((a, b) => b.score - a.score);
  const next = leaves(B.evolve(depth, flown, 4, 0.25, 0.2, 77));
  assert.equal(next.length, 32);
  next.forEach(b => { assert.equal(b.alive, true); assert.equal(b.score, 0); assert.equal(b.y, 190); });
  const key = b => JSON.stringify(layers(b.brain));
  for (let i = 0; i < 4; i++) assert.equal(key(next[i]), key(ranked[i]), `elite ${i}`);
});

test('the flock learns: best and average pipes rise over generations for a fixed seed', () => {
  const depth = 7n, seed = 42, history = [];
  let flock = B.initial(depth, seed, 0);
  for (let g = 0; g < 16; g++) {
    flock = B.run(2000n, flock, 0, B.course(seed, g)).fst;
    const s = B.stats(flock); history.push({ best: s.best_pipes, avg: s.total_pipes / 128 });
    flock = B.evolve(depth, flock, 4, 0.25, 0.2, (Math.imul(seed, 7919) + g) >>> 0);
  }
  const best = history.map(h => h.best), avg = history.map(h => h.avg);
  assert.ok(best[0] <= 1, `generation 0 is random: ${best[0]}`);
  assert.ok(Math.max(...best.slice(8)) >= 20, `late best ${best}`);
  const early = avg.slice(0, 4).reduce((a, b) => a + b) / 4, late = avg.slice(-4).reduce((a, b) => a + b) / 4;
  assert.ok(late > 5 * early && late > 1, `average ${early} -> ${late}`);
});
