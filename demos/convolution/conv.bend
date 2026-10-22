# A tiny convolutional network: 3x3 convolution over many channels, ReLU,
# 2x2 max-pooling, and gradient descent on a kernel's nine weights.
# Every image is a row-major Img tree padded to 2^depth leaves. Every output pixel
# is independent, so each output map is built by splitting its index range in
# two halves that run in parallel.
import Base

# An image, feature map or kernel: a persistent binary tree of F32 values.
type Img is Data:
  Px{v: F32}
  Fork{lo: Img, hi: Img}

# One output channel of a layer: a kernel per input channel, plus a bias.
type Filter is Data:
  Filter{ks: List<&2, List<&2, F32>>, bias: F32}

# A learned kernel, beside the loss before each step (most recent first).
type Fit is Data:
  Fit{kernel: List<&2, F32>, losses: List<&2, F32>}

# -- building and reading maps ------------------------------------------------

def leaves(xs: List<&2, F32>) -> List<&2, Img>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      Con{Px{h}, leaves(t)}

def blank(m: Img) -> Img:
  match m:
    case Px{v}:
      Px{0.0}
    case Fork{lo, hi}:
      Fork{blank(lo), blank(hi)}

# Joins neighbours: [a, b, c] becomes [Fork{a, b}, Fork{c, 0}].
def pairs(xs: List<&2, Img>) -> List<&2, Img>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+a, t}:
      match t:
        case Nil{}:
          Con{Fork{a, blank(a)}, Nil{}}
        case Con{b, rest}:
          Con{Fork{a, b}, pairs(rest)}

def head(xs: List<&2, Img>) -> Img:
  match xs:
    case Nil{}:
      Px{0.0}
    case Con{h, t}:
      h

def top(depth: Nat, xs: List<&2, Img>) -> Img:
  match depth:
    case 0n:
      head(xs)
    case 1n+p:
      top(p, pairs(xs))

# A row-major list of up to 2^depth values; missing values pad with zero.
def from_list(depth: Nat, xs: List<&2, F32>) -> Img:
  top(depth, leaves(xs))

def first(m: Img) -> F32:
  match m:
    case Px{v}:
      v
    case Fork{lo, hi}:
      first(lo)

def side(m: Img, low: Bool) -> Img:
  match m:
    case Px{v}:
      Px{v}
    case Fork{lo, hi}:
      Bool.pick(Img, low, lo, hi)

# Reads index i of a map with 2^depth leaves.
def at(depth: Nat, m: Img, +i: U32) -> F32:
  match depth:
    case 0n:
      first(m)
    case 1n+ +p:
      +half = U32.shln(1, p)
      at(p, side(m, (i < half : U32)), (i .&. (half - 1) : U32))

# -- convolution ---------------------------------------------------------------

# One channel's 3x3 window at output (x, y): sum of input * weight.
# A kernel is its nine weights in reading order: weight j sits at row j / 3,
# column j % 3 of the window.
def taps(k: List<&2, F32>, +j: U32, +img: Img, +d: Nat, +w: U32, +x: U32, +y: U32, acc: F32) -> F32:
  match k:
    case Nil{}:
      acc
    case Con{+c, rest}:
      +v = at(d, img, ((y + j / 3) * w + x + j % 3 : U32))
      taps(rest, (j + 1 : U32), img, d, w, x, y, (acc + v * c : F32))

# Sums the windows of every input channel, each with its own kernel.
def window(chans: List<&2, Img>, ks: List<&2, List<&2, F32>>, +d: Nat, +w: U32, +x: U32, +y: U32, acc: F32) -> F32:
  match chans:
    case Nil{}:
      acc
    case Con{c, cs}:
      match ks:
        case Nil{}:
          acc
        case Con{k, rest}:
          window(cs, rest, d, w, x, y, taps(k, 0, c, d, w, x, y, acc))

def conv_px(valid: Bool, +i: U32, +o: U32, +chans: List<&2, Img>, +ks: List<&2, List<&2, F32>>, +d: Nat, +w: U32, +bias: F32) -> Img:
  match valid:
    case False{}:
      Px{0.0}
    case True{}:
      Px{window(chans, ks, d, w, (i % o : U32), (i / o : U32), bias)}

# A valid 3x3 convolution: a w x w input becomes a (w-2) x (w-2) map with
# 2^depth leaves. The two halves of the output are computed in parallel.
def conv(depth: Nat, +offset: U32, +chans: List<&2, Img>, +ks: List<&2, List<&2, F32>>, +d: Nat, +w: U32, +bias: F32) -> Img:
  match depth:
    case 0n:
      +o = (w - 2 : U32)
      conv_px((offset < o * o : U32), offset, o, chans, ks, d, w, bias)
    case 1n+ +p:
      a b = conv(p, offset, chans, ks, d, w, bias) conv(p, (offset + U32.shln(1, p) : U32), chans, ks, d, w, bias)
      Fork{a, b}

# -- nonlinearity and pooling --------------------------------------------------

def relu(m: Img) -> Img:
  match m:
    case Px{+v}:
      Px{F32.max(v, 0.0)}
    case Fork{lo, hi}:
      a b = relu(lo) relu(hi)
      Fork{a, b}

def pool_px(valid: Bool, +i: U32, +o: U32, +m: Img, +d: Nat, +w: U32) -> Img:
  match valid:
    case False{}:
      Px{0.0}
    case True{}:
      +x = (2 * (i % o) : U32)
      +y = (2 * (i / o) : U32)
      +top = F32.max(at(d, m, (y * w + x : U32)), at(d, m, (y * w + x + 1 : U32)))
      +low = F32.max(at(d, m, ((y + 1) * w + x : U32)), at(d, m, ((y + 1) * w + x + 1 : U32)))
      Px{F32.max(top, low)}

# 2x2 max-pooling with stride 2: a w x w map becomes (w/2) x (w/2).
def pool(depth: Nat, +offset: U32, +m: Img, +d: Nat, +w: U32) -> Img:
  match depth:
    case 0n:
      +o = (w / 2 : U32)
      pool_px((offset < o * o : U32), offset, o, m, d, w)
    case 1n+ +p:
      a b = pool(p, offset, m, d, w) pool(p, (offset + U32.shln(1, p) : U32), m, d, w)
      Fork{a, b}

# -- a layer: many filters, many feature maps -----------------------------------

# Each filter sees every input channel; its map goes through ReLU. The head
# filter and the rest of the list are computed in parallel.
def layer(filters: List<&2, Filter>, +chans: List<&2, Img>, +d: Nat, +w: U32, +od: Nat) -> List<&2, Img>:
  match filters:
    case Nil{}:
      Nil{}
    case Con{f, rest}:
      Filter{ks, bias} = f
      a b = relu(conv(od, 0, chans, ks, d, w, bias)) layer(rest, chans, d, w, od)
      Con{a, b}

def pool_all(maps: List<&2, Img>, +d: Nat, +w: U32, +pd: Nat) -> List<&2, Img>:
  match maps:
    case Nil{}:
      Nil{}
    case Con{m, rest}:
      a b = pool(pd, 0, m, d, w) pool_all(rest, d, w, pd)
      Con{a, b}

# -- the step-by-step view of one output pixel ----------------------------------

# The nine input * weight products of one window, in reading order.
def products(k: List<&2, F32>, +j: U32, +img: Img, +d: Nat, +w: U32, +x: U32, +y: U32) -> List<&2, F32>:
  match k:
    case Nil{}:
      Nil{}
    case Con{+c, rest}:
      +v = at(d, img, ((y + j / 3) * w + x + j % 3 : U32))
      Con{(v * c : F32), products(rest, (j + 1 : U32), img, d, w, x, y)}

# -- learning a kernel ---------------------------------------------------------

def hash(+x: U32) -> U32:
  +a = ((x .^. (x >> 16n)) * 2146121005 : U32)
  +b = ((a .^. (a >> 15n)) * 2221713035 : U32)
  (b .^. (b >> 16n) : U32)

def noise_px(valid: Bool, +i: U32, +seed: U32) -> Img:
  match valid:
    case False{}:
      Px{0.0}
    case True{}:
      Px{(U32.to_f32((hash((i .^. hash(seed) : U32)) >> 8n : U32)) / 8388608.0 - 1.0 : F32)}

# A training image of n uniform values in [-1, 1): every weight sees an
# equally strong, uncorrelated signal, so plain gradient descent is well posed.
def noise(depth: Nat, +offset: U32, +n: U32, +seed: U32) -> Img:
  match depth:
    case 0n:
      noise_px((offset < n : U32), offset, seed)
    case 1n+ +p:
      a b = noise(p, offset, n, seed) noise(p, (offset + U32.shln(1, p) : U32), n, seed)
      Fork{a, b}

# err = prediction - target, leaf by leaf.
def diff(depth: Nat, +a: Img, +b: Img) -> Img:
  match depth:
    case 0n:
      Px{(first(a) - first(b) : F32)}
    case 1n+ +p:
      lo hi = diff(p, side(a, True{}), side(b, True{})) diff(p, side(a, False{}), side(b, False{}))
      Fork{lo, hi}

def sumsq(m: Img) -> F32:
  match m:
    case Px{+v}:
      (v * v : F32)
    case Fork{lo, hi}:
      a b = sumsq(lo) sumsq(hi)
      (a + b : F32)

# dLoss/dw_j (up to 2/N): every error times the input pixel that met w_j.
def gsum(depth: Nat, +offset: U32, +err: Img, +img: Img, +d: Nat, +w: U32, +o: U32, +j: U32) -> F32:
  match depth:
    case 0n:
      +x = (offset % o : U32)
      +y = (offset / o : U32)
      (first(err) * at(d, img, ((y + j / 3) * w + x + j % 3 : U32)) : F32)
    case 1n+ +p:
      a b = gsum(p, offset, side(err, True{}), img, d, w, o, j) gsum(p, (offset + U32.shln(1, p) : U32), side(err, False{}), img, d, w, o, j)
      (a + b : F32)

# The updated weights w_j - scale * gradient_j, in reading order.
def update(k: List<&2, F32>, +j: U32, +err: Img, +img: Img, +d: Nat, +w: U32, +o: U32, +od: Nat, +scale: F32) -> List<&2, F32>:
  match k:
    case Nil{}:
      Nil{}
    case Con{+c, rest}:
      +g = gsum(od, 0, err, img, d, w, o, j)
      Con{(c - scale * g : F32), update(rest, (j + 1 : U32), err, img, d, w, o, od, scale)}

# Gradient descent on the mean squared error between conv(img, k) and a
# target map: `steps` updates, returning the kernel and each step's loss.
def fit(steps: Nat, +img: Img, +d: Nat, +w: U32, +od: Nat, +target: Img, +k: List<&2, F32>, +rate: F32, losses: List<&2, F32>) -> Fit:
  match steps:
    case 0n:
      Fit{k, losses}
    case 1n+ +p:
      +o = (w - 2 : U32)
      +n = U32.to_f32((o * o : U32))
      +err = diff(od, conv(od, 0, Con{img, Nil{}}, Con{k, Nil{}}, d, w, 0.0), target)
      +loss = (sumsq(err) / n : F32)
      next = update(k, 0, err, img, d, w, o, od, (rate * 2.0 / n : F32))
      fit(p, img, d, w, od, target, next, rate, Con{loss, losses})
