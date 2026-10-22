# Neuroevolution after flappIA: a flock of birds, each flown by its own 5-8-2
# network, bred by fitness-weighted roulette selection plus gaussian mutation.
# The world is 640 x 400 units; pipes are a pure function of (index, course).
import Base
import ../probability/math.bend as R

# A neuron is a bias plus one weight per input; a brain is two layers.
type Neuron is Data:
  Neuron{bias: F32, w: List<&2, F32>}

type Brain is Data:
  Brain{hidden: List<&2, Neuron>, out: List<&2, Neuron>}

type Bird is Data:
  Bird{alive: Bool, y: F32, vy: F32, score: U32, brain: Brain}

# The population is a complete binary tree, so every pass over it splits in two.
type Flock is Data:
  Leaf{bird: Bird}
  Node{left: Flock, right: Flock}

type Thought is Data:
  Thought{inputs: List<&2, F32>, hidden: List<&2, F32>, out: List<&2, F32>}

type Pipe is Data:
  Pipe{x: F32, top: F32}

type Stats is Data:
  Stats{best: U32, total: U32, best_pipes: U32, total_pipes: U32, alive: U32}

# World constants.
def width() -> F32: 640.0
def height() -> F32: 400.0
def ground() -> F32: 380.0
def bird_x() -> F32: 140.0
def radius() -> F32: 11.0
def gravity() -> F32: 0.45
def lift() -> F32: (0.0 - 7.0 : F32)
def max_vel() -> F32: 10.0
def speed() -> F32: 3.0
def pipe_w() -> F32: 56.0
def gap() -> F32: 120.0
def spacing() -> F32: 230.0
def first_x() -> F32: 480.0
def start_y() -> F32: 190.0

# The network: sigmoid(bias + w . x) per neuron, layer by layer.
def sigmoid(x: F32) -> F32:
  (1.0 / (1.0 + F32.exp((0.0 - x : F32))) : F32)

def dot(w: List<&2, F32>, x: List<&2, F32>, acc: F32) -> F32:
  match w:
    case []: acc
    case a <> ws:
      match x:
        case []: acc
        case b <> xs: dot(ws, xs, (acc + a * b : F32))

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

# flappIA's rule: flap when output 1 beats output 0.
def flaps(out: List<&2, F32>) -> Bool:
  match out:
    case []: False{}
    case no <> rest:
      match rest:
        case []: False{}
        case yes <> _: (yes > no : F32)

# Pipe k sits at first_x + k * spacing and scrolls left by `speed` per frame.
def pipe_x(+k: U32, +t: U32) -> F32:
  (first_x() + U32.to_f32(k) * spacing() - speed() * U32.to_f32(t) : F32)

def gap_top(+k: U32, +course: U32) -> F32:
  (60.0 + U32.to_f32((R.hash((course + k * 2654435761 : U32)) % 141 : U32)) : F32)

# The first pipe whose right edge has not yet passed the bird; equivalently,
# how many pipes the bird has cleared at frame t.
def next_pipe(+t: U32) -> U32:
  +d = (bird_x() - radius() - pipe_w() - first_x() + speed() * U32.to_f32(t) : F32)
  F32.to_u32((F32.floor((F32.max(d, (0.0 - 1.0 : F32)) / spacing() : F32)) + 1.0 : F32))

# Five inputs, normalized as in flappIA: distance to the pipe, gap top,
# gap bottom, height, velocity.
def senses(+y: F32, +vy: F32, +t: U32, +course: U32) -> List<&2, F32>:
  +k = next_pipe(t)
  +top = gap_top(k, course)
  [((pipe_x(k, t) - bird_x()) / width() : F32), (top / height() : F32),
   ((top + gap()) / height() : F32), (y / height() : F32), (vy / max_vel() : F32)]

def crashed(+y: F32, +t: U32, +course: U32) -> Bool:
  +k = next_pipe(t)
  +top = gap_top(k, course)
  +inside = (bird_x() + radius() > pipe_x(k, t) : F32)
  +hits = (y - radius() < top : F32) || (y + radius() > top + gap() : F32)
  (inside && hits) || (y + radius() > ground() : F32) || (y - radius() < 0.0 : F32)

def fly(flap: Bool, +y: F32, vy: F32, score: U32, brain: Brain, +t: U32, +course: U32) -> Bird:
  +v = F32.min((Bool.pick(F32, flap, lift(), vy) + gravity() : F32), max_vel())
  +ny = (y + v : F32)
  Bird{Bool.not(crashed(ny, (t + 1 : U32), course)), ny, v, (score + 1 : U32), brain}

# One frame for one bird: sense, think, flap or fall, collide.
def step(b: Bird, +t: U32, +course: U32) -> Bird:
  match b:
    case Bird{alive, +y, +vy, score, +brain}:
      match alive:
        case False{}: Bird{False{}, y, vy, score, brain}
        case True{}: fly(flaps(forward(brain, senses(y, vy, t, course))), y, vy, score, brain, t, course)

# One frame for the whole flock: the two halves are independent.
def advance(f: Flock, +t: U32, +course: U32) -> Flock:
  match f:
    case Leaf{b}: Leaf{step(b, t, course)}
    case Node{l, r}:
      a b = advance(l, t, course) advance(r, t, course)
      Node{a, b}

def is_alive(b: Bird) -> Bool:
  Bird{alive, y, vy, score, brain} = b
  alive

def any_alive(f: Flock) -> Bool:
  match f:
    case Leaf{b}: is_alive(b)
    case Node{l, r}:
      a b = any_alive(l) any_alive(r)
      a || b

# Up to `fuel` frames, stopping as soon as every bird has crashed.
def until(fuel: Nat, live: Bool, f: Flock, +t: U32, +course: U32) -> Flock & U32:
  match fuel:
    case 0n: (f, t)
    case 1n+p:
      match live:
        case False{}: (f, t)
        case True{}:
          +g = advance(f, t, course)
          until(p, any_alive(g), g, (t + 1 : U32), course)

def run(fuel: Nat, +f: Flock, +t: U32, +course: U32) -> Flock & U32:
  until(fuel, any_alive(f), f, t, course)

# What the leader is thinking right now, for the brain diagram.
def thought(b: Bird, +t: U32, +course: U32) -> Thought:
  Bird{alive, y, vy, score, brain} = b
  Brain{hidden, out} = brain
  +x = senses(y, vy, t, course)
  +h = layer(hidden, x)
  Thought{x, h, layer(out, h)}

# The pipes on screen at frame t.
def pipes(n: Nat, +k: U32, +t: U32, +course: U32) -> List<&2, Pipe>:
  match n:
    case 0n: []
    case 1n+p: Pipe{pipe_x(k, t), gap_top(k, course)} <> pipes(p, (k + 1 : U32), t, course)

def visible(+t: U32, +course: U32) -> List<&2, Pipe>:
  +k = next_pipe(t)
  pipes(4n, (k - U32.min(k, 1) : U32), t, course)

# Statistics: score is frames survived, pipes are pipes cleared.
def bird_stats(b: Bird) -> Stats:
  Bird{alive, y, vy, +score, brain} = b
  +pipes = next_pipe(score)
  Stats{score, score, pipes, pipes, Bool.to_u32(alive)}

def merge(a: Stats, b: Stats) -> Stats:
  Stats{b1, t1, p1, q1, a1} = a
  Stats{b2, t2, p2, q2, a2} = b
  Stats{U32.max(b1, b2), (t1 + t2 : U32), U32.max(p1, p2), (q1 + q2 : U32), (a1 + a2 : U32)}

def stats(f: Flock) -> Stats:
  match f:
    case Leaf{b}: bird_stats(b)
    case Node{l, r}:
      a b = stats(l) stats(r)
      merge(a, b)

# A fresh brain: every weight and bias uniform in [-1, 1].
def uniform(+s: U32) -> F32:
  (2.0 * R.random(s) - 1.0 : F32)

def weights(n: Nat, +s: U32) -> List<&2, F32>:
  match n:
    case 0n: []
    case 1n+p: uniform(s) <> weights(p, R.hash((s + 1013904223 : U32)))

def neurons(n: Nat, +fan: Nat, +s: U32) -> List<&2, Neuron>:
  match n:
    case 0n: []
    case 1n+p: Neuron{uniform(s), weights(fan, R.hash((s + 374761393 : U32)))} <> neurons(p, fan, R.hash((s + 2654435769 : U32)))

def random_brain(+s: U32) -> Brain:
  Brain{neurons(8n, 5n, R.hash((s + 1 : U32))), neurons(2n, 8n, R.hash((s + 2 : U32)))}

def hatch(brain: Brain) -> Bird:
  Bird{True{}, start_y(), 0.0, 0, brain}

def initial(+d: Nat, +seed: U32, +offset: U32) -> Flock:
  match d:
    case 0n: Leaf{hatch(random_brain(R.hash((offset .^. R.hash(seed) : U32))))}
    case 1n+ +p:
      a b = initial(p, seed, offset) initial(p, seed, (offset + (1 << p) : U32))
      Node{a, b}

# Selection: flappIA weights each bird by score^3, then spins a roulette.
# The wheel is a tree of partial sums, so one spin is a walk from the root.
type Pool is Data:
  PLeaf{w: F32, brain: Brain}
  PNode{w: F32, left: Pool, right: Pool}

def weight(+score: U32) -> F32:
  +s = (U32.to_f32(score) / 100.0 : F32)
  (s * s * s : F32)

def mass(p: Pool) -> F32:
  match p:
    case PLeaf{w, brain}: w
    case PNode{w, l, r}: w

def join(+a: Pool, +b: Pool) -> Pool:
  PNode{(mass(a) + mass(b) : F32), a, b}

def slot(b: Bird) -> Pool:
  Bird{alive, y, vy, score, brain} = b
  PLeaf{weight(score), brain}

def wheel(f: Flock) -> Pool:
  match f:
    case Leaf{b}: slot(b)
    case Node{l, r}:
      a b = wheel(l) wheel(r)
      join(a, b)

def nobody() -> Brain:
  Brain{[], []}

def spin(d: Nat, p: Pool, +ticket: F32) -> Brain:
  match d:
    case 0n:
      match p:
        case PLeaf{w, brain}: brain
        case PNode{w, l, r}: nobody()
    case 1n+q:
      match p:
        case PLeaf{w, brain}: brain
        case PNode{w, +l, +r}:
          +left = (ticket < mass(l) : F32)
          spin(q, Bool.pick(Pool, left, l, r), Bool.pick(F32, left, ticket, (ticket - mass(l) : F32)))

# Mutation: each number moves by a gaussian step with probability `rate`.
def gauss(+s: U32) -> F32:
  +u = (1.0 - R.random(s) : F32)
  +v = R.random(R.hash((s + 2246822519 : U32)))
  (F32.sqrt((0.0 - 2.0 * F32.log(u) : F32)) * F32.cos((6.2831855 * v : F32)) : F32)

def nudge(+x: F32, +rate: F32, +strength: F32, +s: U32) -> F32:
  Bool.pick(F32, (R.random(s) < rate : F32), (x + gauss(R.hash((s + 7 : U32))) * strength : F32), x)

def mutate_all(xs: List<&2, F32>, +rate: F32, +strength: F32, +s: U32) -> List<&2, F32>:
  match xs:
    case []: []
    case x <> rest: nudge(x, rate, strength, s) <> mutate_all(rest, rate, strength, R.hash((s + 1013904223 : U32)))

def mutate_layer(ns: List<&2, Neuron>, +rate: F32, +strength: F32, +s: U32) -> List<&2, Neuron>:
  match ns:
    case []: []
    case n <> rest:
      Neuron{bias, w} = n
      Neuron{nudge(bias, rate, strength, s), mutate_all(w, rate, strength, R.hash((s + 3266489917 : U32)))} <> mutate_layer(rest, rate, strength, R.hash((s + 2654435769 : U32)))

def mutate(brain: Brain, +rate: F32, +strength: F32, +s: U32) -> Brain:
  Brain{hidden, out} = brain
  Brain{mutate_layer(hidden, rate, strength, R.hash((s + 11 : U32))), mutate_layer(out, rate, strength, R.hash((s + 13 : U32)))}

# Elitism: the best few brains are copied into the next generation unchanged.
type Ranked is Data:
  Ranked{score: U32, brain: Brain}

def ranked(f: Flock) -> List<&2, Ranked>:
  match f:
    case Leaf{b}:
      Bird{alive, y, vy, score, brain} = b
      [Ranked{score, brain}]
    case Node{l, r}:
      a b = ranked(l) ranked(r)
      List.append(&2, Ranked, a, b)

def ahead(a: Ranked, b: Ranked) -> Bool:
  Ranked{sa, ba} = a
  Ranked{sb, bb} = b
  (sa >= sb : U32)

def nth(n: Nat, xs: List<&2, Ranked>) -> Brain:
  match n:
    case 0n:
      match xs:
        case []: nobody()
        case Ranked{s, brain} <> rest: brain
    case 1n+q:
      match xs:
        case []: nobody()
        case r <> rest: nth(q, rest)

def newborn(elite: Bool, +elites: List<&2, Ranked>, +pool: Pool, +depth: Nat, +rate: F32, +strength: F32, +s: U32, +offset: U32) -> Bird:
  match elite:
    case True{}: hatch(nth(U32.to_nat(offset), elites))
    case False{}: hatch(mutate(spin(depth, pool, (R.random(s) * mass(pool) : F32)), rate, strength, R.hash((s + 99 : U32))))

# The next generation, bred leaf by leaf; both halves in parallel.
def breed(+d: Nat, +elites: List<&2, Ranked>, +count: U32, +pool: Pool, +depth: Nat, +rate: F32, +strength: F32, +seed: U32, +offset: U32) -> Flock:
  match d:
    case 0n: Leaf{newborn((offset < count : U32), elites, pool, depth, rate, strength, R.hash((offset .^. seed : U32)), offset)}
    case 1n+ +p:
      a b = breed(p, elites, count, pool, depth, rate, strength, seed, offset) breed(p, elites, count, pool, depth, rate, strength, seed, (offset + (1 << p) : U32))
      Node{a, b}

def evolve(+depth: Nat, +f: Flock, +elite: U32, rate: F32, strength: F32, seed: U32) -> Flock:
  +best = List.take(&2, Ranked, List.sort(~Ranked, ~(a => b => ahead(a, b)), ranked(f)), U32.to_nat(elite))
  breed(depth, best, elite, wheel(f), depth, rate, strength, R.hash(seed), 0)

# Each generation flies a fresh course.
def course(seed: U32, generation: U32) -> U32:
  R.hash((R.hash(seed) + generation * 2246822519 : U32))
