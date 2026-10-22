# A 2 -> H -> 1 neural network trained by full-batch gradient descent.
# Hidden neurons use tanh, the output a sigmoid, the loss is cross-entropy.
import Base
import ../probability/math.bend as R

# One hidden neuron: input weights wx, wy, bias b, and its outgoing weight v.
type Neuron is Data:
  Neuron{wx: F32, wy: F32, b: F32, v: F32}
type Net is Data:
  Net{units: List<&2, Neuron>, c: F32}
# A labelled point, t is 0.0 or 1.0.
type Pt is Data:
  Pt{x: F32, y: F32, t: F32}
# A dataset is a balanced tree, so it splits into independent halves.
type Batch is Data:
  One{p: Pt}
  Two{l: Batch, r: Batch}
# A gradient has the shape of a net, plus the summed loss and hits.
type Grad is Data:
  Grad{units: List<&2, Neuron>, c: F32, loss: F32, hits: F32}
type Fit is Data:
  Fit{net: Net, loss: F32, acc: F32}

# 1. A single neuron -----------------------------------------------------

def sigmoid(z: F32) -> F32:
  (1.0 / (1.0 + F32.exp(F32.neg(z))) : F32)

# kind: 0 sigmoid, 1 tanh, 2 step
def act(kind: Nat, +z: F32) -> F32:
  match kind:
    case 0n: sigmoid(z)
    case 1n: F32.tanh(z)
    case _: Bool.pick(F32,(z >= 0.0 : F32),1.0,0.0)

def neuron(kind: Nat, w1: F32, w2: F32, b: F32, x: F32, y: F32) -> F32:
  act(kind,(w1 * x + w2 * y + b : F32))

def neuron_row(n: Nat, +kind: Nat, +w1: F32, +w2: F32, +b: F32, +y: F32, +x: F32, +dx: F32) -> List<&2, F32>:
  match n:
    case 0n: []
    case 1n+p: neuron(kind,w1,w2,b,x,y) <> neuron_row(p,kind,w1,w2,b,y,(x + dx : F32),dx)

# How many points a single neuron puts on the right side of its line.
def classify(p: Pt, +w1: F32, +w2: F32, +b: F32) -> F32:
  Pt{x,y,t} = p
  Bool.pick(F32,Bool.xor((w1 * x + w2 * y + b >= 0.0 : F32),(t >= 0.5 : F32)),0.0,1.0)

def perceptron(batch: Batch, +w1: F32, +w2: F32, +b: F32) -> F32:
  match batch:
    case One{p}: classify(p,w1,w2,b)
    case Two{l,r}:
      u v = perceptron(l,w1,w2,b) perceptron(r,w1,w2,b)
      (u + v : F32)

# 2. Forward pass ----------------------------------------------------------

def hidden(units: List<&2, Neuron>, +x: F32, +y: F32) -> List<&2, F32>:
  match units:
    case []: []
    case Neuron{wx,wy,b,v} <> rest:
      F32.tanh((wx * x + wy * y + b : F32)) <> hidden(rest,x,y)

def dot(units: List<&2, Neuron>, hs: List<&2, F32>, acc: F32) -> F32:
  match units:
    case []: acc
    case Neuron{wx,wy,b,v} <> rest:
      match hs:
        case []: acc
        case h <> hr: dot(rest,hr,(acc + v * h : F32))

def logit(net: Net, +x: F32, +y: F32) -> F32:
  Net{+units,c} = net
  dot(units,hidden(units,x,y),c)

def predict(net: Net, +x: F32, +y: F32) -> F32:
  sigmoid(logit(net,x,y))

def row(n: Nat, +net: Net, +y: F32, +x: F32, +dx: F32) -> List<&2, F32>:
  match n:
    case 0n: []
    case 1n+p: predict(net,x,y) <> row(p,net,y,(x + dx : F32),dx)

# Two tanh neurons draw parallel lines; the output keeps the band between them.
def xor_net(+k: F32) -> Net:
  Net{[Neuron{k,F32.neg(k),(k * 0.5 : F32),(2.0 * k : F32)},Neuron{k,F32.neg(k),F32.neg((k * 0.5 : F32)),F32.neg((2.0 * k : F32))}],F32.neg((2.0 * k : F32))}

# 3. Backpropagation -------------------------------------------------------

# Cross-entropy of sigmoid(z) against t, written so exp never overflows.
def loss(+z: F32, t: F32) -> F32:
  (F32.max(z,0.0) - z * t + F32.log((1.0 + F32.exp(F32.neg(F32.abs(z))) : F32)) : F32)

def hit(+q: F32, +t: F32) -> F32:
  Bool.pick(F32,Bool.xor((q >= 0.5 : F32),(t >= 0.5 : F32)),0.0,1.0)

# d is dLoss/dz = q - t. Each hidden neuron gets d * v * tanh'(a).
def back(units: List<&2, Neuron>, hs: List<&2, F32>, +d: F32, +x: F32, +y: F32) -> List<&2, Neuron>:
  match units:
    case []: []
    case Neuron{wx,wy,b,v} <> rest:
      match hs:
        case []: []
        case +h <> hr:
          +a = (d * v * (1.0 - h * h) : F32)
          Neuron{(a * x : F32),(a * y : F32),a,(d * h : F32)} <> back(rest,hr,d,x,y)

def example(+net: Net, p: Pt) -> Grad:
  Net{+units,c} = net
  Pt{+x,+y,+t} = p
  +hs = hidden(units,x,y)
  +z = dot(units,hs,c)
  +q = sigmoid(z)
  +d = (q - t : F32)
  Grad{back(units,hs,d,x,y),d,loss(z,t),hit(q,t)}

def add(a: List<&2, Neuron>, b: List<&2, Neuron>) -> List<&2, Neuron>:
  match a:
    case []: []
    case Neuron{ax,ay,ab,av} <> ar:
      match b:
        case []: []
        case Neuron{bx,by,bb,bv} <> br:
          Neuron{(ax + bx : F32),(ay + by : F32),(ab + bb : F32),(av + bv : F32)} <> add(ar,br)

def plus(a: Grad, b: Grad) -> Grad:
  Grad{ua,ca,la,ha} = a
  Grad{ub,cb,lb,hb} = b
  Grad{add(ua,ub),(ca + cb : F32),(la + lb : F32),(ha + hb : F32)}

# The batch gradient is a sum over examples: split, recurse in parallel, add.
def grad(+net: Net, batch: Batch) -> Grad:
  match batch:
    case One{p}: example(net,p)
    case Two{l,r}:
      a b = grad(net,l) grad(net,r)
      plus(a,b)

# 4. Gradient descent ------------------------------------------------------

def descend(units: List<&2, Neuron>, g: List<&2, Neuron>, +k: F32) -> List<&2, Neuron>:
  match units:
    case []: []
    case Neuron{wx,wy,b,v} <> rest:
      match g:
        case []: []
        case Neuron{gx,gy,gb,gv} <> gr:
          Neuron{(wx - k * gx : F32),(wy - k * gy : F32),(b - k * gb : F32),(v - k * gv : F32)} <> descend(rest,gr,k)

# One step: w <- w - (lr / n) * dL/dw, reporting the mean loss and accuracy.
def update(net: Net, g: Grad, +lr: F32, +n: F32) -> Fit:
  Net{units,c} = net
  Grad{gu,gc,l,h} = g
  +k = (lr / n : F32)
  Fit{Net{descend(units,gu,k),(c - k * gc : F32)},(l / n : F32),(h / n : F32)}

def train(epochs: Nat, fit: Fit, +batch: Batch, +lr: F32, +n: F32) -> Fit:
  match epochs:
    case 0n: fit
    case 1n+p:
      Fit{+net,l,a} = fit
      train(p,update(net,grad(net,batch),lr,n),batch,lr,n)

# Loss and accuracy of a net without changing it.
def summary(net: Net, g: Grad, +n: F32) -> Fit:
  Grad{gu,gc,l,h} = g
  Fit{net,(l / n : F32),(h / n : F32)}

def score(+net: Net, batch: Batch, +n: F32) -> Fit:
  summary(net,grad(net,batch),n)

# 5. Data and initialisation -----------------------------------------------

def unit(+s: U32, +k: U32) -> F32:
  R.random((s + k : U32))

# kind: 0 two blobs, 1 xor, 2 circle, 3 spiral. All points lie in [-1, 1]^2.
def shape(kind: Nat, +t: F32, +u: F32, +w: F32, +n1: F32, +n2: F32) -> Pt:
  match kind:
    case 0n:
      +s = (2.0 * t - 1.0 : F32)
      Pt{(0.4 * s + 0.35 * n1 : F32),(0.3 * s + 0.35 * n2 : F32),t}
    case 1n:
      +sx = Bool.pick(F32,(u < 0.5 : F32),F32.neg(1.0),1.0)
      +sy = Bool.pick(F32,(w < 0.5 : F32),F32.neg(1.0),1.0)
      Pt{(0.5 * sx + 0.25 * n1 : F32),(0.5 * sy + 0.25 * n2 : F32),Bool.pick(F32,F32.is_eq(sx,sy),1.0,0.0)}
    case 2n:
      +r = Bool.pick(F32,(t >= 0.5 : F32),(0.4 * u : F32),(0.6 + 0.35 * u : F32))
      +a = (6.2831853 * w : F32)
      Pt{(r * F32.cos(a) : F32),(r * F32.sin(a) : F32),t}
    case _:
      +r = (0.08 + 0.85 * u : F32)
      +a = (u * 9.424778 + t * 3.1415927 : F32)
      Pt{(r * F32.cos(a) + 0.04 * n1 : F32),(r * F32.sin(a) + 0.04 * n2 : F32),t}

def point(kind: Nat, +i: U32, seed: U32) -> Pt:
  +s = R.hash((i .^. R.hash(seed) : U32))
  +t = Bool.pick(F32,(unit(s,1) < 0.5 : F32),0.0,1.0)
  shape(kind,t,unit(s,6),unit(s,7),(unit(s,2) + unit(s,3) - 1.0 : F32),(unit(s,4) + unit(s,5) - 1.0 : F32))

def build(depth: Nat, +kind: Nat, +offset: U32, +seed: U32) -> Batch:
  match depth:
    case 0n: One{point(kind,offset,seed)}
    case 1n+ +p: Two{build(p,kind,offset,seed),build(p,kind,(offset + (1 << p) : U32),seed)}

def init_units(h: Nat, +seed: U32) -> List<&2, Neuron>:
  match h:
    case 0n: []
    case 1n+p:
      +a = (2.0 * unit(seed,0) - 1.0 : F32)
      +b = (2.0 * unit(seed,1) - 1.0 : F32)
      +c = (2.0 * unit(seed,2) - 1.0 : F32)
      +d = (2.0 * unit(seed,3) - 1.0 : F32)
      Neuron{(2.0 * a : F32),(2.0 * b : F32),(0.5 * c : F32),(0.5 * d : F32)} <> init_units(p,R.hash((seed + 2654435769 : U32)))

def init(h: Nat, seed: U32) -> Net:
  Net{init_units(h,seed),0.0}
