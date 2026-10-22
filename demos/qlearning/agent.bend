# Q-learning and value iteration on an 8x8 gridworld.
# Cells are numbered s = 8*y + x. Actions: 0 up, 1 right, 2 down, 3 left.
import Base
import ../probability/math.bend as R

# A complete binary tree of F32 leaves, read and written by index. The Q
# table (64 cells x 4 actions, depth 8) and the value table V (64 cells,
# depth 6) are both Tables.
type Table is Data:
  Leaf{v: F32}
  Node{l: Table, r: Table}

# The world, one kind per cell: 0 floor, 1 wall, 2 pit, 3 goal.
type World is Data:
  Cell{kind: U32}
  Split{l: World, r: World}

type Episode is Data:
  Episode{q: Table, ret: F32, steps: U32, path: List<&2, U32>}

type Batch is Data:
  Batch{q: Table, returns: List<&2, F32>, steps: U32, path: List<&2, U32>}

type Spot is Data:
  Spot{value: F32, action: U32, spread: F32}

# Tables

def zeros(depth: Nat) -> Table:
  match depth:
    case 0n: Leaf{0.0}
    case 1n+ +p: Node{zeros(p), zeros(p)}

def leaf(t: Table) -> F32:
  match t:
    case Leaf{v}: v
    case Node{l, r}: leaf(l)

def half(t: Table, left: Bool) -> Table:
  match t:
    case Leaf{v}: Leaf{v}
    case Node{l, r}: Bool.pick(Table, left, l, r)

def get(depth: Nat, t: Table, +i: U32) -> F32:
  match depth:
    case 0n: leaf(t)
    case 1n+ +p:
      +h = (1 << p : U32)
      get(p, half(t, (i < h : U32)), (i % h : U32))

# Rewrites one leaf; the untouched half is shared, not copied.
def put(depth: Nat, on: Bool, t: Table, +i: U32, +v: F32) -> Table:
  match depth:
    case 0n: Bool.pick(Table, on, Leaf{v}, t)
    case 1n+ +p:
      match on:
        case False{}: t
        case True{}:
          match t:
            case Leaf{x}: Leaf{x}
            case Node{l, r}:
              +h = (1 << p : U32)
              Node{put(p, (i < h : U32), l, (i % h : U32), v), put(p, (i >= h : U32), r, (i % h : U32), v)}

def change(a: Table, b: Table) -> F32:
  match a:
    case Leaf{x}:
      match b:
        case Leaf{y}: F32.abs((x - y : F32))
        case Node{l, r}: 0.0
    case Node{al, ar}:
      match b:
        case Leaf{y}: 0.0
        case Node{bl, br}:
          x y = change(al, bl) change(ar, br)
          F32.max(x, y)

# The world

def wleaf(w: World) -> U32:
  match w:
    case Cell{k}: k
    case Split{l, r}: wleaf(l)

def wside(w: World, left: Bool) -> World:
  match w:
    case Cell{k}: Cell{k}
    case Split{l, r}: Bool.pick(World, left, l, r)

def kind(depth: Nat, w: World, +i: U32) -> U32:
  match depth:
    case 0n: wleaf(w)
    case 1n+ +p:
      +h = (1 << p : U32)
      kind(p, wside(w, (i < h : U32)), (i % h : U32))

def cell(+world: World, +s: U32) -> U32:
  kind(6n, world, s)

# Where action a leads from s, before walls. The edge of the grid blocks.
def target(+s: U32, +a: U32) -> U32:
  +x = (s % 8 : U32)
  +y = (s / 8 : U32)
  +up = Bool.pick(U32, (y > 0 : U32), (s - 8 : U32), s)
  +right = Bool.pick(U32, (x < 7 : U32), (s + 1 : U32), s)
  +down = Bool.pick(U32, (y < 7 : U32), (s + 8 : U32), s)
  +left = Bool.pick(U32, (x > 0 : U32), (s - 1 : U32), s)
  Bool.pick(U32, (a < 2 : U32), Bool.pick(U32, U32.is_eq(a, 0), up, right), Bool.pick(U32, U32.is_eq(a, 2), down, left))

# The environment: walls block, pits and the goal end the episode.
def move(+world: World, +s: U32, +a: U32) -> U32:
  +t = target(s, a)
  Bool.pick(U32, U32.is_eq(cell(world, t), 1), s, t)

def reward(+k: U32, +cost: F32) -> F32:
  Bool.pick(F32, U32.is_eq(k, 3), 1.0, Bool.pick(F32, U32.is_eq(k, 2), (0.0 - 1.0 : F32), (0.0 - cost : F32)))

def terminal(+k: U32) -> Bool:
  (k >= 2 : U32)

# Q values

def q(+t: Table, +s: U32, +a: U32) -> F32:
  get(8n, t, (s * 4 + a : U32))

def best(+t: Table, +s: U32) -> F32:
  F32.max(F32.max(q(t, s, 0), q(t, s, 1)), F32.max(q(t, s, 2), q(t, s, 3)))

def worst(+t: Table, +s: U32) -> F32:
  F32.min(F32.min(q(t, s, 0), q(t, s, 1)), F32.min(q(t, s, 2), q(t, s, 3)))

# The best of four values, scanned from action o so ties break by o.
def top(+o: U32, +v0: F32, +v1: F32, +v2: F32, +v3: F32) -> U32:
  +a = (o % 4 : U32)
  +b = ((o + 1) % 4 : U32)
  +c = ((o + 2) % 4 : U32)
  +d = ((o + 3) % 4 : U32)
  +va = Bool.pick(F32, (a < 2 : U32), Bool.pick(F32, U32.is_eq(a, 0), v0, v1), Bool.pick(F32, U32.is_eq(a, 2), v2, v3))
  +vb = Bool.pick(F32, (b < 2 : U32), Bool.pick(F32, U32.is_eq(b, 0), v0, v1), Bool.pick(F32, U32.is_eq(b, 2), v2, v3))
  +vc = Bool.pick(F32, (c < 2 : U32), Bool.pick(F32, U32.is_eq(c, 0), v0, v1), Bool.pick(F32, U32.is_eq(c, 2), v2, v3))
  +vd = Bool.pick(F32, (d < 2 : U32), Bool.pick(F32, U32.is_eq(d, 0), v0, v1), Bool.pick(F32, U32.is_eq(d, 2), v2, v3))
  +ab = Bool.pick(U32, (vb > va : F32), b, a)
  +cd = Bool.pick(U32, (vd > vc : F32), d, c)
  Bool.pick(U32, (F32.max(vc, vd) > F32.max(va, vb) : F32), cd, ab)

def greedy(+t: Table, +s: U32, +o: U32) -> U32:
  top(o, q(t, s, 0), q(t, s, 1), q(t, s, 2), q(t, s, 3))

# Epsilon-greedy: with probability eps a random action, else the best one
# (ties broken at random, so an untrained agent wanders instead of freezing).
def act(+t: Table, +s: U32, +eps: F32, +seed: U32) -> U32:
  +explore = (R.random(seed) < eps : F32)
  Bool.pick(U32, explore, (R.hash((seed + 1 : U32)) % 4 : U32), greedy(t, s, R.hash((seed + 2 : U32))))

# Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
def learn(+old: F32, r: F32, future: F32, alpha: F32, gamma: F32) -> F32:
  (old + alpha * (r + gamma * future - old) : F32)

# One episode: act, observe reward and next state, update one entry of Q.
def episode(fuel: Nat, alive: Bool, +t: Table, +world: World, +s: U32, +alpha: F32, +gamma: F32, +eps: F32, +cost: F32, +seed: U32, ret: F32, steps: U32, path: List<&2, U32>) -> Episode:
  match fuel:
    case 0n: Episode{t, ret, steps, path}
    case 1n+p:
      match alive:
        case False{}: Episode{t, ret, steps, path}
        case True{}:
          +a = act(t, s, eps, seed)
          +next = move(world, s, a)
          +k = cell(world, next)
          +r = reward(k, cost)
          +done = terminal(k)
          +future = Bool.pick(F32, done, 0.0, best(t, next))
          +i = (s * 4 + a : U32)
          learned = put(8n, True{}, t, i, learn(get(8n, t, i), r, future, alpha, gamma))
          episode(p, Bool.not(done), learned, world, next, alpha, gamma, eps, cost, R.hash((seed + 2654435769 : U32)), (ret + r : F32), (steps + 1 : U32), next <> path)

def table(e: Episode) -> Table:
  Episode{t, r, n, p} = e
  t

def gain(e: Episode) -> F32:
  Episode{t, r, n, p} = e
  r

def length(e: Episode) -> U32:
  Episode{t, r, n, p} = e
  n

def trail(e: Episode) -> List<&2, U32>:
  Episode{t, r, n, p} = e
  p

# n episodes from the start cell. Returns the new table, every episode's
# return, the steps taken, and the last episode's path.
def train(n: Nat, +t: Table, +world: World, +start: U32, +alpha: F32, +gamma: F32, +eps: F32, +cost: F32, +seed: U32, returns: List<&2, F32>, steps: U32, path: List<&2, U32>) -> Batch:
  match n:
    case 0n: Batch{t, List.reverse(&2, F32, returns), steps, List.reverse(&2, U32, path)}
    case 1n+p:
      +e = episode(200n, True{}, t, world, start, alpha, gamma, eps, cost, seed, 0.0, 0, [start])
      train(p, table(e), world, start, alpha, gamma, eps, cost, R.hash((seed + 1013904223 : U32)), gain(e) <> returns, (steps + length(e) : U32), trail(e))

# What the page draws: per cell, max Q, the greedy action, and max - min Q.
def policy(n: Nat, +t: Table, +s: U32) -> List<&2, Spot>:
  match n:
    case 0n: []
    case 1n+p: Spot{best(t, s), greedy(t, s, 0), (best(t, s) - worst(t, s) : F32)} <> policy(p, t, (s + 1 : U32))

# Follow the greedy policy (no exploration) until the episode ends.
def walk(fuel: Nat, alive: Bool, +t: Table, +world: World, +s: U32, path: List<&2, U32>) -> List<&2, U32>:
  match fuel:
    case 0n: List.reverse(&2, U32, path)
    case 1n+p:
      match alive:
        case False{}: List.reverse(&2, U32, path)
        case True{}:
          +next = move(world, s, greedy(t, s, 0))
          walk(p, Bool.not(terminal(cell(world, next))), t, world, next, next <> path)

# Value iteration: the model is known, so back up every cell at once.
# V(s) <- max_a [ r(s,a) + gamma * V(s') ]

def lookahead(+v: Table, +world: World, +s: U32, +a: U32, +gamma: F32, +cost: F32) -> F32:
  +next = move(world, s, a)
  +k = cell(world, next)
  (reward(k, cost) + gamma * Bool.pick(F32, terminal(k), 0.0, get(6n, v, next)) : F32)

def backup(+v: Table, +world: World, +s: U32, +gamma: F32, +cost: F32) -> F32:
  +a = F32.max(lookahead(v, world, s, 0, gamma, cost), lookahead(v, world, s, 1, gamma, cost))
  +b = F32.max(lookahead(v, world, s, 2, gamma, cost), lookahead(v, world, s, 3, gamma, cost))
  Bool.pick(F32, (cell(world, s) >= 1 : U32), 0.0, F32.max(a, b))

# Every cell reads only the old V, so the two halves of the grid are
# independent: a parallel call splits them, down to single cells.
def sweep(depth: Nat, +lo: U32, +v: Table, +world: World, +gamma: F32, +cost: F32) -> Table:
  match depth:
    case 0n: Leaf{backup(v, world, lo, gamma, cost)}
    case 1n+ +p:
      left right = sweep(p, lo, v, world, gamma, cost) sweep(p, (lo + (1 << p) : U32), v, world, gamma, cost)
      Node{left, right}

def plan_action(+v: Table, +world: World, +s: U32, +gamma: F32, +cost: F32) -> U32:
  top(0, lookahead(v, world, s, 0, gamma, cost), lookahead(v, world, s, 1, gamma, cost), lookahead(v, world, s, 2, gamma, cost), lookahead(v, world, s, 3, gamma, cost))

def values(n: Nat, +v: Table, +world: World, +s: U32, +gamma: F32, +cost: F32) -> List<&2, Spot>:
  match n:
    case 0n: []
    case 1n+p: Spot{get(6n, v, s), plan_action(v, world, s, gamma, cost), 1.0} <> values(p, v, world, (s + 1 : U32), gamma, cost)

def plan(fuel: Nat, alive: Bool, +v: Table, +world: World, +s: U32, +gamma: F32, +cost: F32, path: List<&2, U32>) -> List<&2, U32>:
  match fuel:
    case 0n: List.reverse(&2, U32, path)
    case 1n+p:
      match alive:
        case False{}: List.reverse(&2, U32, path)
        case True{}:
          +next = move(world, s, plan_action(v, world, s, gamma, cost))
          plan(p, Bool.not(terminal(cell(world, next))), v, world, next, gamma, cost, next <> path)
