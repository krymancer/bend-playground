# Sheep meadow: a 48x48 grass field, and sheep that graze, walk, breed,
# inherit traits, and die. A 2D retelling of krymancer/sheep-simulation-3d.
# Every tick: grass grows (parallel over the field tree), sheep age and pay
# basal metabolism, decide from a snapshot (parallel over the flock), graze
# (sequential, so no blade of grass is eaten twice), then breed.
import Base
import ../probability/math.bend as R

# The field is a binary tree over cell indices z * 48 + x (2304 cells, padded
# to 4096 leaves). Padding subtrees are Void, so they cost one node.
type Field is Data:
  Void{}
  Cell{grass: F32, fert: F32}
  Fork{left: Field, right: Field}

# act: 0 wandering, 1 seeking food, 2 grazing, 3 seeking a mate.
# want: the bite a grazing sheep asks for this tick.
type Sheep is Data:
  Sheep{id: U32, gen: U32, x: F32, z: F32, heading: F32, age: F32, life: F32, energy: F32, cool: F32, speed: F32, metab: F32, wool: F32, hungry: Bool, target: U32, rethink: F32, act: U32, want: F32}

type World is Data:
  World{seed: U32, tick: U32, grass: Field, flock: List<&2, Sheep>, next: U32, founders: U32, births: U32, old: U32, starved: U32, halted: Bool, regrow: F32, metab: F32, eaten: F32}

def dt() -> F32:
  0.1

def none() -> U32:
  4294967295

# Randomness
# ----------
# Counter-based: a sheep's dice depend only on (seed, tick, id), never on the
# order sheep are visited. That is what makes the flock safe to split.

def key(+seed: U32, +tick: U32, +id: U32) -> U32:
  R.hash((seed .^. R.hash((tick * 2246822519 + id * 3266489917 + 1 : U32)) : U32))

def rnd(+k: U32, salt: U32) -> F32:
  R.random((k + salt * 2654435769 : U32))

def span(+k: U32, salt: U32, +lo: F32, hi: F32) -> F32:
  (lo + (hi - lo) * rnd(k, salt) : F32)

# Uniform in [-w, w).
def jit(+k: U32, salt: U32, +w: F32) -> F32:
  ((2.0 * rnd(k, salt) - 1.0) * w : F32)

# Grass
# -----

def soil(+seed: U32, +i: U32, +food: F32) -> Field:
  +x = U32.to_f32((i % 48 : U32))
  +z = U32.to_f32((i / 48 : U32))
  +k = R.hash((seed .^. R.hash((i * 2654435761 + 7 : U32)) : U32))
  +wave = (0.28 * F32.sin((x * 0.22 : F32)) * F32.cos((z * 0.19 : F32)) : F32)
  +fert = F32.clamp((0.85 + wave + jit(k, 0, 0.2) : F32), 0.35, 1.35)
  Cell{F32.clamp((4.0 * food / 100.0 * span(k, 1, 0.88, 1.12) : F32), 0.0, 4.0), fert}

def plant(d: Nat, live: Bool, +seed: U32, +food: F32, +off: U32) -> Field:
  match d:
    case 0n:
      match live:
        case True{}:
          soil(seed, off, food)
        case False{}:
          Void{}
    case 1n+ +p:
      match live:
        case False{}:
          Void{}
        case True{}:
          +mid = (off + U32.shln(1, p) : U32)
          a b = plant(p, True{}, seed, food, off) plant(p, (mid < 2304 : U32), seed, food, mid)
          Fork{a, b}

# Regrowth touches every cell independently, so each fork is a parallel call.
def grow(f: Field, +rate: F32) -> Field:
  match f:
    case Void{}:
      Void{}
    case Cell{g, +k}:
      Cell{F32.min(4.0, (g + rate * k : F32)), k}
    case Fork{l, r}:
      a b = grow(l, rate) grow(r, rate)
      Fork{a, b}

def total(f: Field) -> F32:
  match f:
    case Void{}:
      0.0
    case Cell{g, k}:
      g
    case Fork{l, r}:
      a b = total(l) total(r)
      (a + b : F32)

def biomass(d: Nat, f: Field, +i: U32) -> F32:
  match d:
    case 0n:
      match f:
        case Cell{g, k}:
          g
        case Void{}:
          0.0
        case Fork{l, r}:
          0.0
    case 1n+ +p:
      match f:
        case Fork{l, r}:
          +half = U32.shln(1, p)
          biomass(p, Bool.pick(Field, (i < half : U32), l, r), (i .&. (half - 1) : U32))
        case Cell{g, k}:
          g
        case Void{}:
          0.0

def rejoin(low: Bool, c: Field, l: Field, r: Field) -> Field:
  match low:
    case True{}:
      Fork{c, r}
    case False{}:
      Fork{l, c}

def rebite(low: Bool, l: Field, r: Field, res: Field & F32) -> Field & F32:
  (c, e) = res
  (rejoin(low, c, l, r), e)

# Eat up to `want` from cell i; returns the new field and what was eaten.
def bite(d: Nat, f: Field, +i: U32, +want: F32) -> Field & F32:
  match d:
    case 0n:
      match f:
        case Cell{+g, k}:
          +e = F32.max(0.0, F32.min(g, want))
          (Cell{(g - e : F32), k}, e)
        case Void{}:
          (Void{}, 0.0)
        case Fork{l, r}:
          (Fork{l, r}, 0.0)
    case 1n+ +p:
      match f:
        case Fork{+l, +r}:
          +half = U32.shln(1, p)
          +low = (i < half : U32)
          rebite(low, l, r, bite(p, Bool.pick(Field, low, l, r), (i .&. (half - 1) : U32), want))
        case Cell{g, k}:
          (Cell{g, k}, 0.0)
        case Void{}:
          (Void{}, 0.0)

# Add `amount` (negative grazes) to every cell whose center is in the disk.
def brush(d: Nat, f: Field, +off: U32, +cx: F32, +cz: F32, +rad: F32, +amount: F32) -> Field:
  match d:
    case 0n:
      match f:
        case Cell{+g, k}:
          +dx = (U32.to_f32((off % 48 : U32)) + 0.5 - cx : F32)
          +dz = (U32.to_f32((off / 48 : U32)) + 0.5 - cz : F32)
          Cell{Bool.pick(F32, (dx * dx + dz * dz <= rad * rad : F32), F32.clamp((g + amount : F32), 0.0, 4.0), g), k}
        case Void{}:
          Void{}
        case Fork{l, r}:
          Fork{l, r}
    case 1n+ +p:
      match f:
        case Fork{l, r}:
          a b = brush(p, l, off, cx, cz, rad, amount) brush(p, r, (off + U32.shln(1, p) : U32), cx, cz, rad, amount)
          Fork{a, b}
        case Cell{g, k}:
          Cell{g, k}
        case Void{}:
          Void{}

# Food search: the best biomass / (1 + distance) within 7 cells, scanning
# only the tree ranges that hold each row of the window.
type Best is Data:
  Best{score: F32, cell: U32}

def consider(b: Best, +g: F32, +i: U32, +sx: F32, +sz: F32) -> Best:
  Best{+score, cell} = b
  +dx = (U32.to_f32((i % 48 : U32)) + 0.5 - sx : F32)
  +dz = (U32.to_f32((i / 48 : U32)) + 0.5 - sz : F32)
  +v = (g / (1.0 + F32.sqrt((dx * dx + dz * dz : F32))) : F32)
  Bool.pick(Best, (g > 0.12 : F32) && (v > score : F32), Best{v, i}, Best{score, cell})

def scan(d: Nat, go: Bool, f: Field, +off: U32, +lo: U32, +hi: U32, +sx: F32, +sz: F32, best: Best) -> Best:
  match d:
    case 0n:
      match go:
        case False{}:
          best
        case True{}:
          match f:
            case Cell{g, k}:
              consider(best, g, off, sx, sz)
            case Void{}:
              best
            case Fork{l, r}:
              best
    case 1n+ +p:
      match go:
        case False{}:
          best
        case True{}:
          match f:
            case Fork{l, r}:
              +half = U32.shln(1, p)
              +mid = (off + half : U32)
              near = scan(p, (off <= hi : U32) && (lo < mid : U32), l, off, lo, hi, sx, sz, best)
              scan(p, (mid <= hi : U32) && (lo < mid + half : U32), r, mid, lo, hi, sx, sz, near)
            case Cell{g, k}:
              best
            case Void{}:
              best

def rows(n: Nat, +z: U32, +xlo: U32, +xhi: U32, +grass: Field, +sx: F32, +sz: F32, best: Best) -> Best:
  match n:
    case 0n:
      best
    case 1n+p:
      +row = (z * 48 : U32)
      rows(p, (z + 1 : U32), xlo, xhi, grass, sx, sz, scan(12n, True{}, grass, 0, (row + xlo : U32), (row + xhi : U32), sx, sz, best))

def best_cell(b: Best) -> U32:
  Best{s, c} = b
  c

def find_food(+grass: Field, +sx: F32, +sz: F32) -> U32:
  +fx = F32.to_u32(F32.floor(sx))
  +fz = F32.to_u32(F32.floor(sz))
  +zlo = (U32.max(fz, 7) - 7 : U32)
  +zhi = U32.min((fz + 7 : U32), 47)
  best_cell(rows(U32.to_nat((zhi - zlo + 1 : U32)), zlo, (U32.max(fx, 7) - 7 : U32), U32.min((fx + 7 : U32), 47), grass, sx, sz, Best{0.0, none()}))

# Sheep
# -----

def founder(+seed: U32, +i: U32) -> Sheep:
  +k = R.hash((seed .^. R.hash((i * 2654435761 + 77 : U32)) : U32))
  +c = (i % 4 : U32)
  +x = F32.clamp((14.0 + 17.0 * U32.to_f32((c % 2 : U32)) + jit(k, 0, 6.0) : F32), 1.0, 47.0)
  +z = F32.clamp((14.0 + 17.0 * U32.to_f32((c / 2 : U32)) + jit(k, 1, 6.0) : F32), 1.0, 47.0)
  Sheep{(i + 1 : U32), 0, x, z, jit(k, 2, 3.1415927), span(k, 3, 12.0, 160.0), span(k, 4, 240.0, 360.0), span(k, 5, 62.0, 92.0), span(k, 6, 2.0, 18.0), span(k, 7, 0.8, 1.25), span(k, 8, 0.8, 1.2), span(k, 9, 0.72, 1.0), False{}, none(), 0.0, 0, 0.0}

def founders(n: Nat, +seed: U32, +i: U32) -> List<&2, Sheep>:
  match n:
    case 0n:
      []
    case 1n+p:
      founder(seed, i) <> founders(p, seed, (i + 1 : U32))

def create(+seed: U32, +population: U32, +food: F32, regrow: F32, metab: F32) -> World:
  World{seed, 0, plant(12n, True{}, seed, food, 0), founders(U32.to_nat(population), seed, 0), (population + 1 : U32), population, 0, 0, 0, False{}, regrow, metab, 0.0}

def s_id(s: Sheep) -> U32:
  Sheep{id, gen, x, z, h, age, life, e, cool, speed, metab, wool, hungry, target, rethink, act, want} = s
  id

def s_x(s: Sheep) -> F32:
  Sheep{id, gen, x, z, h, age, life, e, cool, speed, metab, wool, hungry, target, rethink, act, want} = s
  x

def s_z(s: Sheep) -> F32:
  Sheep{id, gen, x, z, h, age, life, e, cool, speed, metab, wool, hungry, target, rethink, act, want} = s
  z

def s_energy(s: Sheep) -> F32:
  Sheep{id, gen, x, z, h, age, life, e, cool, speed, metab, wool, hungry, target, rethink, act, want} = s
  e

def s_age(s: Sheep) -> F32:
  Sheep{id, gen, x, z, h, age, life, e, cool, speed, metab, wool, hungry, target, rethink, act, want} = s
  age

def s_life(s: Sheep) -> F32:
  Sheep{id, gen, x, z, h, age, life, e, cool, speed, metab, wool, hungry, target, rethink, act, want} = s
  life

def s_act(s: Sheep) -> U32:
  Sheep{id, gen, x, z, h, age, life, e, cool, speed, metab, wool, hungry, target, rethink, act, want} = s
  act

# Adult, fed, rested, and not hungry: ready to breed.
def ready(s: Sheep) -> Bool:
  Sheep{id, gen, x, z, h, age, life, e, cool, speed, metab, wool, hungry, target, rethink, act, want} = s
  (age >= 45.0 : F32) && (e >= 80.0 : F32) && (cool <= 0.0 : F32) && Bool.not(hungry)

# Phase 1: age, basal metabolism, hunger hysteresis. Returns the energy burnt.
def older(s: Sheep, +burn: F32, +dt: F32) -> Sheep & F32:
  Sheep{id, gen, x, z, h, age, life, +e, cool, speed, +metab, wool, hungry, target, rethink, act, want} = s
  +spent = F32.min(e, (burn * metab : F32))
  +left = (e - spent : F32)
  +starving = (left < 68.0 : F32)
    || (hungry && (left <= 94.0 : F32))
  (Sheep{id, gen, x, z, h, (age + dt : F32), life, left, F32.max(0.0, (cool - dt : F32)), speed, metab, wool, starving, target, (rethink - dt : F32), act, 0.0}, spent)

type Cull is Data:
  Cull{alive: List<&2, Sheep>, old: U32, starved: U32, basal: F32, lost: F32}

def fate(starve: Bool, aged: Bool, +s: Sheep, acc: Cull) -> Cull:
  match starve:
    case True{}:
      Cull{alive, old, starved, basal, lost} = acc
      Cull{alive, old, (starved + 1 : U32), basal, lost}
    case False{}:
      match aged:
        case True{}:
          Cull{alive, old, starved, basal, lost} = acc
          Cull{alive, (old + 1 : U32), starved, basal, (lost + s_energy(s) : F32)}
        case False{}:
          Cull{alive, old, starved, basal, lost} = acc
          Cull{s <> alive, old, starved, basal, lost}

def tally(r: Sheep & F32, acc: Cull) -> Cull:
  (+s, spent) = r
  Cull{alive, old, starved, basal, lost} = acc
  fate((s_energy(s) <= 0.0 : F32), (s_age(s) >= s_life(s) : F32), s, Cull{alive, old, starved, (basal + spent : F32), lost})

# Starvation wins when both limits arrive on the same tick. Output reversed.
def cull(xs: List<&2, Sheep>, +burn: F32, +dt: F32, acc: Cull) -> Cull:
  match xs:
    case []:
      acc
    case s <> rest:
      cull(rest, burn, dt, tally(older(s, burn, dt), acc))

# Phase 2: a read-only snapshot of who is ready to breed, and where.
type Mate is Data:
  Mate{id: U32, x: F32, z: F32}

def offer(go: Bool, +s: Sheep, acc: List<&2, Mate>) -> List<&2, Mate>:
  match go:
    case True{}:
      Mate{s_id(s), s_x(s), s_z(s)} <> acc
    case False{}:
      acc

def mates(xs: List<&2, Sheep>, acc: List<&2, Mate>) -> List<&2, Mate>:
  match xs:
    case []:
      acc
    case +s <> rest:
      mates(rest, offer(ready(s), s, acc))

type Near is Data:
  Near{d2: F32, x: F32, z: F32, found: Bool}

def closer(m: Mate, +id: U32, +x: F32, +z: F32, best: Near) -> Near:
  Mate{mid, +mx, +mz} = m
  Near{+d2, bx, bz, found} = best
  +dx = (mx - x : F32)
  +dz = (mz - z : F32)
  +e2 = (dx * dx + dz * dz : F32)
  Bool.pick(Near, Bool.not(U32.is_eq(mid, id)) && (e2 < d2 : F32), Near{e2, mx, mz, True{}}, Near{d2, bx, bz, found})

def nearest(ms: List<&2, Mate>, +id: U32, +x: F32, +z: F32, best: Near) -> Near:
  match ms:
    case []:
      best
    case m <> rest:
      nearest(rest, id, x, z, closer(m, id, x, z, best))

# Phase 3: every sheep decides where to go, reading only the snapshot.
type Plan is Data:
  Plan{tx: F32, tz: F32, act: U32, target: U32, rethink: F32, heading: F32, want: F32}

def ahead(+x: F32, +z: F32, +h: F32, act: U32, target: U32, rethink: F32) -> Plan:
  Plan{(x + F32.sin(h) * 3.0 : F32), (z + F32.cos(h) * 3.0 : F32), act, target, rethink, h, 0.0}

def arrive(at: Bool, +cx: F32, +cz: F32, +t: U32, +r: F32, +h: F32, +e: F32, +dt: F32) -> Plan:
  match at:
    case True{}:
      Plan{cx, cz, 2, t, r, h, F32.min((0.8 * dt : F32), ((100.0 - e) / 12.0 : F32))}
    case False{}:
      Plan{cx, cz, 1, t, r, h, 0.0}

def aim(found: Bool, +t: U32, +r: F32, +s: Sheep, +dt: F32) -> Plan:
  match found:
    case True{}:
      Sheep{id, gen, +x, +z, h, age, life, e, cool, speed, metab, wool, hungry, target, rethink, act, want} = s
      +cx = (U32.to_f32((t % 48 : U32)) + 0.5 : F32)
      +cz = (U32.to_f32((t / 48 : U32)) + 0.5 : F32)
      +dx = (cx - x : F32)
      +dz = (cz - z : F32)
      arrive((F32.sqrt((dx * dx + dz * dz : F32)) < 0.5 : F32), cx, cz, t, r, h, e, dt)
    case False{}:
      Sheep{id, gen, x, z, h, age, life, e, cool, speed, metab, wool, hungry, target, rethink, act, want} = s
      ahead(x, z, h, 1, t, r)

def hunt(retarget: Bool, +s: Sheep, +grass: Field, +k: U32, +dt: F32) -> Plan:
  match retarget:
    case True{}:
      +t = find_food(grass, s_x(s), s_z(s))
      aim(Bool.not(U32.is_eq(t, none())), t, (0.75 + 0.5 * rnd(k, 1) : F32), s, dt)
    case False{}:
      Sheep{id, gen, x, z, h, age, life, e, cool, speed, metab, wool, hungry, target, rethink, act, want} = s
      aim(True{}, target, rethink, s, dt)

def wander(turn: Bool, +s: Sheep, +k: U32) -> Plan:
  match turn:
    case True{}:
      Sheep{id, gen, x, z, h, age, life, e, cool, speed, metab, wool, hungry, target, rethink, act, want} = s
      ahead(x, z, (h + jit(k, 2, 1.2) : F32), 0, target, span(k, 3, 1.5, 4.0))
    case False{}:
      Sheep{id, gen, x, z, h, age, life, e, cool, speed, metab, wool, hungry, target, rethink, act, want} = s
      ahead(x, z, h, 0, target, rethink)

def pursue(found: Bool, +mx: F32, +mz: F32, +target: U32, +rethink: F32, +h: F32, +s: Sheep, +k: U32) -> Plan:
  match found:
    case True{}:
      Plan{mx, mz, 3, target, rethink, h, 0.0}
    case False{}:
      wander((rethink <= 0.0 : F32), s, k)

def court(near: Near, +s: Sheep, +k: U32) -> Plan:
  match near:
    case Near{d2, mx, mz, found}:
      Sheep{id, gen, x, z, h, age, life, e, cool, speed, metab, wool, hungry, target, +rethink, act, want} = s
      pursue(found, mx, mz, target, rethink, h, s, k)

def flirt(go: Bool, +s: Sheep, +ms: List<&2, Mate>, +k: U32) -> Plan:
  match go:
    case True{}:
      court(nearest(ms, s_id(s), s_x(s), s_z(s), Near{36.0, 0.0, 0.0, False{}}), s, k)
    case False{}:
      Sheep{id, gen, x, z, h, age, life, e, cool, speed, metab, wool, hungry, target, +rethink, act, want} = s
      wander((rethink <= 0.0 : F32), s, k)

def decide(hungry: Bool, +s: Sheep, +grass: Field, +ms: List<&2, Mate>, +k: U32, +dt: F32) -> Plan:
  match hungry:
    case True{}:
      Sheep{id, gen, x, z, h, age, life, e, cool, speed, metab, wool, hun, +target, +rethink, act, want} = s
      hunt((rethink <= 0.0 : F32) || U32.is_eq(target, none()) || (biomass(12n, grass, target) < 0.02 : F32), s, grass, k, dt)
    case False{}:
      flirt(ready(s), s, ms, k)

# Walk toward the plan's target, nudged away from the fence; pay 0.65 x speed^2
# per unit walked. A grazing sheep stands still.
def walk(p: Plan, s: Sheep, +dt: F32) -> Sheep:
  Plan{+tx, +tz, +act, target, rethink, +h0, want} = p
  Sheep{id, gen, +x, +z, h, +age, life, +e, cool, +speed, metab, wool, hungry, t0, r0, a0, w0} = s
  +dx0 = (tx - x : F32)
  +dz0 = (tz - z : F32)
  +d0 = F32.sqrt((dx0 * dx0 + dz0 * dz0 : F32))
  +far = (d0 > 0.01 : F32)
  +ux = Bool.pick(F32, far, (dx0 / d0 : F32), dx0)
  +uz = Bool.pick(F32, far, (dz0 / d0 : F32), dz0)
  +dx = (ux + Bool.pick(F32, (x < 1.5 : F32), 2.0, 0.0) - Bool.pick(F32, (x > 46.5 : F32), 2.0, 0.0) : F32)
  +dz = (uz + Bool.pick(F32, (z < 1.5 : F32), 2.0, 0.0) - Bool.pick(F32, (z > 46.5 : F32), 2.0, 0.0) : F32)
  +d = F32.sqrt((dx * dx + dz * dz : F32))
  +moving = (d > 0.001 : F32) && Bool.not(U32.is_eq(act, 2))
  +v = (speed * Bool.pick(F32, (age < 45.0 : F32), 0.75, 1.0) * Bool.pick(F32, U32.is_eq(act, 0), 0.6, 1.0) * dt : F32)
  +nx = Bool.pick(F32, moving, F32.clamp((x + dx / d * v : F32), 0.25, 47.75), x)
  +nz = Bool.pick(F32, moving, F32.clamp((z + dz / d * v : F32), 0.25, 47.75), z)
  +mx = (nx - x : F32)
  +mz = (nz - z : F32)
  +step = F32.sqrt((mx * mx + mz * mz : F32))
  +cost = (step * 0.65 * speed * speed : F32)
  Sheep{id, gen, nx, nz, Bool.pick(F32, moving, F32.atan2(dx, dz), h0), age, life, F32.max(0.0, (e - cost : F32)), cool, speed, metab, wool, hungry, target, rethink, act, want}

def think(+s: Sheep, +grass: Field, +ms: List<&2, Mate>, +seed: U32, +tick: U32, +dt: F32) -> Sheep:
  Sheep{+id, gen, x, z, h, age, life, e, cool, speed, metab, wool, hungry, target, rethink, act, want} = s
  walk(decide(hungry, s, grass, ms, key(seed, tick, id), dt), s, dt)

def think_all(xs: List<&2, Sheep>, +grass: Field, +ms: List<&2, Mate>, +seed: U32, +tick: U32, +dt: F32) -> List<&2, Sheep>:
  match xs:
    case []:
      []
    case s <> rest:
      think(s, grass, ms, seed, tick, dt) <> think_all(rest, grass, ms, seed, tick, dt)

def split_con(s: Sheep, r: List<&2, Sheep> & List<&2, Sheep>) -> List<&2, Sheep> & List<&2, Sheep>:
  (a, b) = r
  (s <> a, b)

def split(n: Nat, xs: List<&2, Sheep>) -> List<&2, Sheep> & List<&2, Sheep>:
  match n:
    case 0n:
      ([], xs)
    case 1n+p:
      match xs:
        case []:
          ([], [])
        case s <> rest:
          split_con(s, split(p, rest))

# Decisions only read the snapshot, so the flock splits into halves that
# think in parallel: 2^(depth+1) chunks in all.
type Halves is Data:
  Halves{a: List<&2, Sheep>, na: U32, b: List<&2, Sheep>, nb: U32}

def halve_at(ab: List<&2, Sheep> & List<&2, Sheep>, na: U32, nb: U32) -> Halves:
  (a, b) = ab
  Halves{a, na, b, nb}

def halve(xs: List<&2, Sheep>, +n: U32) -> Halves:
  +h = (n / 2 : U32)
  halve_at(split(U32.to_nat(h), xs), h, (n - h : U32))

def flock_think(d: Nat, hs: Halves, +grass: Field, +ms: List<&2, Mate>, +seed: U32, +tick: U32, +dt: F32) -> List<&2, Sheep>:
  match d:
    case 0n:
      Halves{a, na, b, nb} = hs
      List.append(&2, Sheep, think_all(a, grass, ms, seed, tick, dt), think_all(b, grass, ms, seed, tick, dt))
    case 1n+ +p:
      Halves{a, na, b, nb} = hs
      x y = flock_think(p, halve(a, na), grass, ms, seed, tick, dt) flock_think(p, halve(b, nb), grass, ms, seed, tick, dt)
      List.append(&2, Sheep, x, y)

# Phase 4: grazing bites are applied one sheep at a time against the shared
# field (12 energy per unit of biomass); walkers that ran dry starve here.
type Pasture is Data:
  Pasture{grass: Field, flock: List<&2, Sheep>, eaten: F32, starved: U32, count: U32}

def fed(s: Sheep, r: Field & F32, acc: Pasture) -> Pasture:
  Sheep{id, gen, x, z, h, age, life, e, cool, speed, metab, wool, hungry, target, rethink, act, want} = s
  (g, +bit) = r
  Pasture{old, flock, eaten, starved, count} = acc
  Pasture{g, Sheep{id, gen, x, z, h, age, life, F32.min(100.0, (e + bit * 12.0 : F32)), cool, speed, metab, wool, hungry, target, rethink, act, 0.0} <> flock, (eaten + bit : F32), starved, (count + 1 : U32)}

def walked(starve: Bool, s: Sheep, acc: Pasture) -> Pasture:
  match starve:
    case True{}:
      Pasture{g, flock, eaten, starved, count} = acc
      Pasture{g, flock, eaten, (starved + 1 : U32), count}
    case False{}:
      Pasture{g, flock, eaten, starved, count} = acc
      Pasture{g, s <> flock, eaten, starved, (count + 1 : U32)}

def feed(grazing: Bool, +s: Sheep, acc: Pasture) -> Pasture:
  match grazing:
    case True{}:
      Sheep{id, gen, x, z, h, age, life, e, cool, speed, metab, wool, hungry, target, rethink, act, want} = s
      Pasture{g, flock, eaten, starved, count} = acc
      fed(s, bite(12n, g, target, want), Pasture{Void{}, flock, eaten, starved, count})
    case False{}:
      walked((s_energy(s) <= 0.0 : F32), s, acc)

def settle(xs: List<&2, Sheep>, acc: Pasture) -> Pasture:
  match xs:
    case []:
      acc
    case +s <> rest:
      settle(rest, feed(U32.is_eq(s_act(s), 2), s, acc))

# Phase 5: breeding. Each ready sheep pairs with the first later ready sheep
# within 1.9 cells. Both pay 22 energy; the lamb gets 40, 4 is overhead.
def paid(s: Sheep) -> Sheep:
  Sheep{id, gen, x, z, h, age, life, e, cool, speed, metab, wool, hungry, target, rethink, act, want} = s
  Sheep{id, gen, x, z, h, age, life, (e - 22.0 : F32), 35.0, speed, metab, wool, hungry, target, rethink, act, want}

def hits(xs: List<&2, Sheep>, +s: Sheep) -> Bool:
  match xs:
    case []:
      False{}
    case +h <> t:
      +dx = (s_x(h) - s_x(s) : F32)
      +dz = (s_z(h) - s_z(s) : F32)
      ready(h) && (dx * dx + dz * dz <= 3.61 : F32)

def find_con(h: Sheep, r: List<&2, Sheep> & Maybe<&2, Sheep>) -> List<&2, Sheep> & Maybe<&2, Sheep>:
  (rest, m) = r
  (h <> rest, m)

def find_mate(xs: List<&2, Sheep>, hit: Bool, +s: Sheep) -> List<&2, Sheep> & Maybe<&2, Sheep>:
  match xs:
    case []:
      ([], None{})
    case +h <> +t:
      match hit:
        case True{}:
          (paid(h) <> t, Some{h})
        case False{}:
          find_con(h, find_mate(t, hits(t, s), s))

def inherit(+a: F32, +b: F32, +k: U32, salt: U32, +w: F32, lo: F32, hi: F32) -> F32:
  F32.clamp(((a + b) / 2.0 + jit(k, salt, w) : F32), lo, hi)

def lamb(a: Sheep, b: Sheep, +id: U32, +seed: U32, +tick: U32) -> Sheep:
  Sheep{ia, ga, +xa, +za, ha, aa, la, ea, ca, +sa, +ma, +wa, hua, ta, ra, aca, wta} = a
  Sheep{ib, gb, +xb, +zb, hb, ab, lb, eb, cb, +sb, +mb, +wb, hub, tb, rb, acb, wtb} = b
  +k = key(seed, tick, (id .^. 2863311530 : U32))
  +x = F32.clamp(((xa + xb) / 2.0 + jit(k, 0, 0.8) : F32), 0.25, 47.75)
  +z = F32.clamp(((za + zb) / 2.0 + jit(k, 1, 0.8) : F32), 0.25, 47.75)
  Sheep{id, (U32.max(ga, gb) + 1 : U32), x, z, jit(k, 2, 3.1415927), 0.0, span(k, 3, 240.0, 360.0), 40.0, 35.0, inherit(sa, sb, k, 4, 0.04, 0.8, 1.25), inherit(ma, mb, k, 5, 0.04, 0.8, 1.2), inherit(wa, wb, k, 6, 0.025, 0.72, 1.0), False{}, none(), 0.0, 0, 0.0}

type Brood is Data:
  Brood{kept: List<&2, Sheep>, lambs: List<&2, Sheep>, next: U32, alive: U32, halted: Bool, born: U32}

type Turn is Data:
  Turn{rest: List<&2, Sheep>, brood: Brood}

def birth(full: Bool, +s: Sheep, +mate: Sheep, rest: List<&2, Sheep>, rest2: List<&2, Sheep>, +seed: U32, +tick: U32, acc: Brood) -> Turn:
  match full:
    case True{}:
      Brood{kept, lambs, next, alive, halted, born} = acc
      Turn{rest, Brood{s <> kept, lambs, next, alive, True{}, born}}
    case False{}:
      Brood{kept, lambs, +next, alive, halted, born} = acc
      Turn{rest2, Brood{paid(s) <> kept, lamb(s, mate, next, seed, tick) <> lambs, (next + 1 : U32), (alive + 1 : U32), halted, (born + 1 : U32)}}

def mated(m: Maybe<&2, Sheep>, +s: Sheep, rest: List<&2, Sheep>, rest2: List<&2, Sheep>, +seed: U32, +tick: U32, acc: Brood) -> Turn:
  match m:
    case None{}:
      Brood{kept, lambs, next, alive, halted, born} = acc
      Turn{rest, Brood{s <> kept, lambs, next, alive, halted, born}}
    case Some{mate}:
      Brood{kept, lambs, next, +alive, halted, born} = acc
      birth((alive >= 1000 : U32), s, mate, rest, rest2, seed, tick, Brood{kept, lambs, next, alive, halted, born})

def found(r: List<&2, Sheep> & Maybe<&2, Sheep>, +s: Sheep, rest: List<&2, Sheep>, +seed: U32, +tick: U32, acc: Brood) -> Turn:
  (rest2, m) = r
  mated(m, s, rest, rest2, seed, tick, acc)

def meet(go: Bool, +s: Sheep, +rest: List<&2, Sheep>, +seed: U32, +tick: U32, acc: Brood) -> Turn:
  match go:
    case True{}:
      found(find_mate(rest, hits(rest, s), s), s, rest, seed, tick, acc)
    case False{}:
      Brood{kept, lambs, next, alive, halted, born} = acc
      Turn{rest, Brood{s <> kept, lambs, next, alive, halted, born}}

def unhalted(b: Brood) -> Bool:
  Brood{kept, lambs, next, alive, halted, born} = b
  Bool.not(halted)

def breed(n: Nat, t: Turn, +seed: U32, +tick: U32) -> Brood:
  match n:
    case 0n:
      Turn{xs, acc} = t
      Brood{kept, lambs, next, alive, halted, born} = acc
      Brood{List.append(&2, Sheep, List.reverse(&2, Sheep, xs), kept), lambs, next, alive, halted, born}
    case 1n+p:
      match t:
        case Turn{xs, +acc}:
          match xs:
            case []:
              acc
            case +s <> rest:
              breed(p, meet(ready(s) && unhalted(acc), s, rest, seed, tick, acc), seed, tick)

# One tick
# --------

def bred(b: Brood, w: World) -> World:
  Brood{kept, lambs, next2, alive2, halted2, born} = b
  World{seed, t, g, flock, next, founders, births, old, starved, h, regrow, metab, ate} = w
  World{seed, t, g, List.append(&2, Sheep, List.reverse(&2, Sheep, kept), List.reverse(&2, Sheep, lambs)), next2, founders, (births + born : U32), old, starved, halted2, regrow, metab, ate}

def settled(p: Pasture, w: World) -> World:
  Pasture{g2, settled_rev, ate, starved3, +count} = p
  World{+seed, +t, g, flock, next, founders, births, old, starved, h, regrow, metab, eaten} = w
  bred(breed(U32.to_nat(count), Turn{List.reverse(&2, Sheep, settled_rev), Brood{[], [], next, count, False{}, 0}}, seed, t), World{seed, t, g2, [], 0, founders, births, old, (starved + starved3 : U32), h, regrow, metab, ate})

# The world passed along carries the grown field and an empty flock.
def culled(c: Cull, w: World) -> World:
  Cull{alive_rev, old2, starved2, basal, lost} = c
  World{+seed, +t, +g1, flock, next, founders, births, old, starved, h, regrow, metab, eaten} = w
  +alive = List.reverse(&2, Sheep, alive_rev)
  +n = U32.from_nat(List.length(&2, Sheep, alive))
  thought = flock_think(2n, halve(alive, n), g1, mates(alive, []), seed, t, dt())
  settled(settle(thought, Pasture{g1, [], 0.0, 0, 0}), World{seed, t, Void{}, [], next, founders, births, (old + old2 : U32), (starved + starved2 : U32), h, regrow, metab, eaten})

def advance(halted: Bool, w: World) -> World:
  match halted:
    case True{}:
      w
    case False{}:
      World{+seed, tick, grass, flock, next, founders, births, old, starved, h, +regrow, +metab, eaten} = w
      +step = dt()
      culled(cull(flock, (0.72 * metab * step : F32), step, Cull{[], 0, 0, 0.0, 0.0}), World{seed, (tick + 1 : U32), grow(grass, (regrow * step : F32)), [], next, founders, births, old, starved, h, regrow, metab, eaten})

def halted(w: World) -> Bool:
  World{seed, tick, grass, flock, next, founders, births, old, starved, h, regrow, metab, eaten} = w
  h

def tick(+w: World) -> World:
  advance(halted(w), w)

def run(n: Nat, w: World) -> World:
  match n:
    case 0n:
      w
    case 1n+p:
      run(p, tick(w))

# Commands from the reader
# ------------------------

def paint(w: World, +x: F32, +z: F32, +rad: F32, +amount: F32) -> World:
  World{seed, tick, grass, flock, next, founders, births, old, starved, h, regrow, metab, eaten} = w
  World{seed, tick, brush(12n, grass, 0, x, z, rad, amount), flock, next, founders, births, old, starved, h, regrow, metab, eaten}

def tune(w: World, regrow: F32, metab: F32) -> World:
  World{seed, tick, grass, flock, next, founders, births, old, starved, h, r, m, eaten} = w
  World{seed, tick, grass, flock, next, founders, births, old, starved, h, regrow, metab, eaten}

# Census
# ------

type Census is Data:
  Census{n: U32, speed: F32, metab: F32, wool: F32, energy: F32, lambs: U32, hungry: U32, gen: U32}

def count(c: Census, s: Sheep) -> Census:
  Census{n, cs, cm, cw, ce, cl, ch, cg} = c
  Sheep{id, gen, x, z, h, +age, life, e, cool, speed, metab, wool, +hungry, target, rethink, act, want} = s
  Census{(n + 1 : U32), (cs + speed : F32), (cm + metab : F32), (cw + wool : F32), (ce + e : F32), (cl + Bool.to_u32((age < 45.0 : F32)) : U32), (ch + Bool.to_u32(hungry) : U32), U32.max(cg, gen)}

def census(xs: List<&2, Sheep>, acc: Census) -> Census:
  match xs:
    case []:
      acc
    case s <> rest:
      census(rest, count(acc, s))

def trait(kind: U32, s: Sheep) -> F32:
  match kind:
    case 0:
      Sheep{id, gen, x, z, h, age, life, e, cool, speed, metab, wool, hungry, target, rethink, act, want} = s
      speed
    case _:
      Sheep{id, gen, x, z, h, age, life, e, cool, speed, metab, wool, hungry, target, rethink, act, want} = s
      metab

def in_bin(xs: List<&2, Sheep>, +kind: U32, +lo: F32, +width: F32, +bins: U32, +i: U32, acc: U32) -> U32:
  match xs:
    case []:
      acc
    case s <> rest:
      +b = U32.min(F32.to_u32(F32.max(0.0, ((trait(kind, s) - lo) / width : F32))), (bins - 1 : U32))
      in_bin(rest, kind, lo, width, bins, i, (acc + Bool.to_u32(U32.is_eq(b, i)) : U32))

# Counts per bin of a trait over [lo, lo + bins * width).
def histogram(n: Nat, +xs: List<&2, Sheep>, +kind: U32, +lo: F32, +width: F32, +bins: U32, +i: U32) -> List<&2, U32>:
  match n:
    case 0n:
      []
    case 1n+p:
      in_bin(xs, kind, lo, width, bins, i, 0) <> histogram(p, xs, kind, lo, width, bins, (i + 1 : U32))

def survey(w: World) -> Census:
  World{seed, tick, grass, flock, next, founders, births, old, starved, h, regrow, metab, eaten} = w
  census(flock, Census{0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0})

# 12 bins: speed over [0.8, 1.25], metabolism over [0.8, 1.2].
def traits(w: World, +kind: U32) -> List<&2, U32>:
  World{seed, tick, grass, flock, next, founders, births, old, starved, h, regrow, metab, eaten} = w
  histogram(12n, flock, kind, 0.8, Bool.pick(F32, U32.is_eq(kind, 0), 0.0375, 0.033333335), 12, 0)

# One adult sheep with a full life ahead, for watching a single energy budget.
def lone(+seed: U32, +food: F32, regrow: F32) -> World:
  World{seed, 0, plant(12n, True{}, seed, food, 0), [Sheep{1, 0, 24.0, 24.0, 0.6, 45.0, 360.0, 70.0, 35.0, 1.0, 1.0, 0.86, False{}, none(), 0.0, 0, 0.0}], 2, 1, 0, 0, 0, False{}, regrow, 1.0, 0.0}

def mix(+h: U32, v: F32) -> U32:
  R.hash((h .^. F32.bits(v) + 2654435769 : U32))

def marks(xs: List<&2, Sheep>, h: U32) -> U32:
  match xs:
    case []:
      h
    case s <> rest:
      Sheep{id, gen, x, z, hd, age, life, e, cool, speed, metab, wool, hungry, target, rethink, act, want} = s
      marks(rest, mix(mix(mix(h, x), z), e))

# A 32-bit digest of every sheep and the field: same seed, same digest.
def fingerprint(w: World) -> U32:
  World{seed, tick, grass, flock, next, founders, births, old, starved, h, regrow, metab, eaten} = w
  mix(marks(flock, R.hash(tick)), total(grass))
