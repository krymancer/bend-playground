# A toy deepfake detector, trained in the browser.
#
# Synthetic 16x16 "faces" are drawn procedurally. A fake one carries a faint
# checkerboard inside the face, the trace that upsampling layers of image
# generators tend to leave. The detector is the smallest MesoNet-like model:
# K 3x3 convolution kernels, ReLU, global average pooling, and one logistic
# output. It learns by full-batch gradient descent with Adam.
import Base
import ../probability/math.bend as R

# 1. Types -------------------------------------------------------------------

# A 3x3 window of pixels, read row by row. Kernels are windows too.
type Win is Data:
  Win{a: F32, b: F32, c: F32, d: F32, e: F32, f: F32, g: F32, h: F32, i: F32}
# One kernel: weights w, bias b, and its weight v in the output neuron.
type Ker is Data:
  Ker{w: Win, b: F32, v: F32}
type Net is Data:
  Net{ks: List<&2, Ker>, c: F32}
# A training example: its 14x14 windows and its label (1.0 = fake).
type Ex is Data:
  Ex{ws: List<&2, Win>, t: F32}
# A dataset is a balanced tree, so it splits into independent halves.
type Batch is Data:
  One{ex: Ex}
  Two{l: Batch, r: Batch}
# What one kernel sees on one image: summed ReLU, active count, active windows.
type Acc is Data:
  Acc{s: F32, n: F32, g: Win}
# A gradient has the shape of a net, plus summed loss and confusion counts.
type Grad is Data:
  Grad{net: Net, loss: F32, tp: F32, fp: F32, fn: F32, tn: F32}
# Adam keeps a running mean m and mean square v next to every parameter p.
type Par is Data:
  Par{p: F32, m: F32, v: F32}
type Opt is Data:
  Opt{ps: List<&2, Par>, t: F32}
type Fit is Data:
  Fit{opt: Opt, loss: F32, acc: F32}
type Face is Data:
  Face{cx: F32, cy: F32, rx: F32, ry: F32, skin: F32, bg: F32, gx: F32, s: U32, fake: F32}

# 2. Synthetic faces -----------------------------------------------------------

def unit(+s: U32, k: U32) -> F32:
  R.random((s + k : U32))

# Odd indices are fakes, so every dataset is exactly balanced.
def face(+index: U32, +seed: U32) -> Face:
  +s = R.hash((index .^. R.hash(seed) : U32))
  Face{(8.0 + (unit(s,1) - 0.5) * 2.0 : F32),(8.4 + (unit(s,2) - 0.5) * 1.6 : F32),(4.6 + unit(s,3) * 1.4 : F32),(5.6 + unit(s,4) * 1.4 : F32),(0.62 + unit(s,5) * 0.25 : F32),(0.18 + unit(s,6) * 0.3 : F32),((unit(s,7) - 0.5) * 0.03 : F32),s,U32.to_f32(U32.and(index,1))}

def dist2(+x: F32, +y: F32) -> F32:
  (x * x + y * y : F32)

def bump(x: F32) -> F32:
  F32.clamp(x,0.0,1.0)

# Background, skin, two eyes, a mouth, sensor grain, and (if fake) the artifact.
def pixel(+f: Face, +amp: F32, +off: F32, +x: U32, +y: U32) -> F32:
  Face{+cx,+cy,+rx,+ry,+skin,+bg,+gx,+s,+fake} = f
  +px = (U32.to_f32(x) + 0.5 : F32)
  +py = (U32.to_f32(y) + 0.5 : F32)
  +dy = ((py - cy) / ry : F32)
  +m = bump(((1.0 - dist2(((px - cx) / rx : F32),dy)) * 4.0 : F32))
  +base = (bg + gx * (px - 8.0) : F32)
  +ey = (py - (cy - 0.22 * ry) : F32)
  +eyes = (bump((1.6 - dist2((px - (cx - 0.42 * rx) : F32),ey) : F32)) + bump((1.6 - dist2((px - (cx + 0.42 * rx) : F32),ey) : F32)) : F32)
  +mouth = bump((1.0 - dist2(((px - cx) / (0.45 * rx) : F32),((py - (cy + 0.5 * ry)) / 0.7 : F32)) : F32))
  +grain = ((R.random(R.hash((s + 1000 + y * 16 + x : U32))) - 0.5) * 0.08 : F32)
  +sign = Bool.pick(F32,U32.is_eq(U32.and((x + y : U32),1),0),1.0,F32.neg(1.0))
  (base + m * (skin - 0.06 * dy - base) - 0.45 * m * eyes - 0.35 * m * mouth + grain + fake * amp * m * sign - off : F32)

def row(n: Nat, +f: Face, +amp: F32, +off: F32, +x: U32, +y: U32) -> List<&2, F32>:
  match n:
    case 0n: []
    case 1n+p: pixel(f,amp,off,x,y) <> row(p,f,amp,off,(x + 1 : U32),y)

def rows(n: Nat, +f: Face, +amp: F32, +off: F32, +y: U32) -> List<&2, List<&2, F32>>:
  match n:
    case 0n: []
    case 1n+p: row(16n,f,amp,off,0,y) <> rows(p,f,amp,off,(y + 1 : U32))

# 256 raw pixels of one face, for drawing.
def patch(index: U32, seed: U32, amp: F32) -> List<&2, F32>:
  List.concat(&2,F32,rows(16n,face(index,seed),amp,0.0,0))

def minus(a: List<&2, F32>, b: List<&2, F32>) -> List<&2, F32>:
  match a b:
    case Con{x, ar} Con{y, br}: (x - y : F32) <> minus(ar,br)
    case _ _: []

# What the generator added to a face: the fake minus the same face without it.
def trace(+index: U32, +seed: U32, amp: F32) -> List<&2, F32>:
  minus(patch(index,seed,amp),patch(index,seed,0.0))

# 3. Sliding windows ---------------------------------------------------------

# Slide along three rows at once, carrying the last two columns. The first
# two steps only warm the carry up; every later step emits one window.
def emit(warm: Nat, w: Win, rest: List<&2, Win>) -> List<&2, Win>:
  match warm:
    case 0n: w <> rest
    case _: rest

def wrow(r0: List<&2, F32>, r1: List<&2, F32>, r2: List<&2, F32>, +warm: Nat, +a: F32, +b: F32, +d: F32, +e: F32, +g: F32, +h: F32) -> List<&2, Win>:
  match r0 r1 r2:
    case Con{+c, t0} Con{+f, t1} Con{+i, t2}:
      emit(warm,Win{a,b,c,d,e,f,g,h,i},wrow(t0,t1,t2,Nat.sub(warm,1n),b,c,e,f,h,i))
    case _ _ _: []

def emits(warm: Nat, ws: List<&2, Win>, rest: List<&2, Win>) -> List<&2, Win>:
  match warm:
    case 0n: List.append(&2,Win,ws,rest)
    case _: rest

# The same trick one level up: carry the last two rows.
def wrows(rs: List<&2, List<&2, F32>>, +warm: Nat, +p0: List<&2, F32>, +p1: List<&2, F32>) -> List<&2, Win>:
  match rs:
    case []: []
    case +r <> t: emits(warm,wrow(p0,p1,r,2n,0.0,0.0,0.0,0.0,0.0,0.0),wrows(t,Nat.sub(warm,1n),p1,r))

# Pixels are centered on 0.5 before they reach the network.
def labelled(+f: Face, ws: List<&2, Win>) -> Ex:
  Face{cx,cy,rx,ry,skin,bg,gx,s,fake} = f
  Ex{ws,fake}

def example_of(+index: U32, +seed: U32, +amp: F32) -> Ex:
  +f = face(index,seed)
  labelled(f,wrows(rows(16n,f,amp,0.5,0),2n,[],[]))

def build(depth: Nat, +offset: U32, +seed: U32, +amp: F32) -> Batch:
  match depth:
    case 0n: One{example_of(offset,seed,amp)}
    case 1n+ +p:
      a b = build(p,offset,seed,amp) build(p,(offset + (1 << p) : U32),seed,amp)
      Two{a,b}

# 4. Forward and backward pass -------------------------------------------------

def dot9(+w: Win, x: Win) -> F32:
  Win{wa,wb,wc,wd,we,wf,wg,wh,wi} = w
  Win{xa,xb,xc,xd,xe,xf,xg,xh,xi} = x
  (wa * xa + wb * xb + wc * xc + wd * xd + we * xe + wf * xf + wg * xg + wh * xh + wi * xi : F32)

# a + k * b, slot by slot.
def axpy(a: Win, +k: F32, b: Win) -> Win:
  Win{aa,ab,ac,ad,ae,af,ag,ah,ai} = a
  Win{ba,bb,bc,bd,be,bf,bg,bh,bi} = b
  Win{(aa + k * ba : F32),(ab + k * bb : F32),(ac + k * bc : F32),(ad + k * bd : F32),(ae + k * be : F32),(af + k * bf : F32),(ag + k * bg : F32),(ah + k * bh : F32),(ai + k * bi : F32)}

def zero() -> Win:
  Win{0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}

def sigmoid(z: F32) -> F32:
  (1.0 / (1.0 + F32.exp(F32.neg(z))) : F32)

# Slide one kernel over every window: ReLU, and remember which windows fired.
def scan(ws: List<&2, Win>, +w: Win, +b: F32, s: F32, n: F32, g: Win) -> Acc:
  match ws:
    case []: Acc{s,n,g}
    case +x <> rest:
      +z = (dot9(w,x) + b : F32)
      +on = Bool.pick(F32,(z > 0.0 : F32),1.0,0.0)
      scan(rest,w,b,(s + on * z : F32),(n + on : F32),axpy(g,on,x))

def accs(ks: List<&2, Ker>, +ws: List<&2, Win>) -> List<&2, Acc>:
  match ks:
    case []: []
    case Ker{w,b,v} <> rest: scan(ws,w,b,0.0,0.0,zero()) <> accs(rest,ws)

# Global average pooling (196 windows), then the output neuron.
def mix(ks: List<&2, Ker>, as: List<&2, Acc>, acc: F32) -> F32:
  match ks as:
    case Con{Ker{w,b,v}, kr} Con{Acc{s,n,g}, ar}: mix(kr,ar,(acc + v * s / 196.0 : F32))
    case _ _: acc

# The chain rule, per kernel: dz/dv = mean ReLU, dz/dw = v * mean active window.
def back(ks: List<&2, Ker>, as: List<&2, Acc>, +d: F32) -> List<&2, Ker>:
  match ks as:
    case Con{Ker{w,b,+v}, kr} Con{Acc{+s,n,g}, ar}:
      +k = (d * v / 196.0 : F32)
      Ker{axpy(zero(),k,g),(k * n : F32),(d * s / 196.0 : F32)} <> back(kr,ar,d)
    case _ _: []

# Cross-entropy of sigmoid(z) against t, written so exp never overflows.
def loss(+z: F32, t: F32) -> F32:
  (F32.max(z,0.0) - z * t + F32.log((1.0 + F32.exp(F32.neg(F32.abs(z))) : F32)) : F32)

def example(+net: Net, ex: Ex) -> Grad:
  Net{+ks,c} = net
  Ex{ws,+t} = ex
  +as = accs(ks,ws)
  +z = mix(ks,as,c)
  +q = sigmoid(z)
  +d = (q - t : F32)
  +p = Bool.pick(F32,(q >= 0.5 : F32),1.0,0.0)
  Grad{Net{back(ks,as,d),d},loss(z,t),(p * t : F32),(p * (1.0 - t) : F32),((1.0 - p) * t : F32),((1.0 - p) * (1.0 - t) : F32)}

def addk(a: List<&2, Ker>, b: List<&2, Ker>) -> List<&2, Ker>:
  match a b:
    case Con{Ker{aw,ab,av}, ar} Con{Ker{bw,bb,bv}, br}: Ker{axpy(aw,1.0,bw),(ab + bb : F32),(av + bv : F32)} <> addk(ar,br)
    case _ _: []

def plus(a: Grad, b: Grad) -> Grad:
  Grad{Net{ka,ca},la,pa,qa,ra,sa} = a
  Grad{Net{kb,cb},lb,pb,qb,rb,sb} = b
  Grad{Net{addk(ka,kb),(ca + cb : F32)},(la + lb : F32),(pa + pb : F32),(qa + qb : F32),(ra + rb : F32),(sa + sb : F32)}

# The batch gradient is a sum over examples: split, recurse in parallel, add.
def grad(+net: Net, batch: Batch) -> Grad:
  match batch:
    case One{ex}: example(net,ex)
    case Two{l,r}:
      a b = grad(net,l) grad(net,r)
      plus(a,b)

# The probability that one example is fake.
def predict(+net: Net, ex: Ex) -> F32:
  Net{+ks,c} = net
  Ex{ws,t} = ex
  sigmoid(mix(ks,accs(ks,ws),c))

# One kernel's ReLU feature map over a face (14x14 values), for drawing.
def fmap(+w: Win, +b: F32, ws: List<&2, Win>) -> List<&2, F32>:
  match ws:
    case []: []
    case x <> rest: F32.max((dot9(w,x) + b : F32),0.0) <> fmap(w,b,rest)

# 5. Flat parameters and Adam --------------------------------------------------

def wlist(w: Win, tail: List<&2, F32>) -> List<&2, F32>:
  Win{a,b,c,d,e,f,g,h,i} = w
  a <> b <> c <> d <> e <> f <> g <> h <> i <> tail

def kflat(ks: List<&2, Ker>) -> List<&2, F32>:
  match ks:
    case []: []
    case Ker{w,b,v} <> rest: wlist(w,b <> v <> kflat(rest))

# The flat order is c, then each kernel's nine weights, bias and v.
def flat(net: Net) -> List<&2, F32>:
  Net{ks,c} = net
  c <> kflat(ks)

# Read the flat list back, eleven numbers per kernel.
def kernels(n: Nat, ps: List<&2, F32>) -> List<&2, Ker>:
  match n ps:
    case 1n+p Con{a,Con{b,Con{c,Con{d,Con{e,Con{f,Con{g,Con{h,Con{i,Con{bb,Con{v,rest}}}}}}}}}}}:
      Ker{Win{a,b,c,d,e,f,g,h,i},bb,v} <> kernels(p,rest)
    case _ _: []

def values(ps: List<&2, Par>) -> List<&2, F32>:
  match ps:
    case []: []
    case Par{p,m,v} <> rest: p <> values(rest)

def unflat(k: Nat, ps: List<&2, F32>) -> Net:
  match ps:
    case c <> rest: Net{kernels(k,rest),c}
    case []: Net{[],0.0}

def net_of(k: Nat, ps: List<&2, Par>) -> Net:
  unflat(k,values(ps))

# k1, k2 undo the bias of the running means early on (1 / (1 - beta^t)).
def adam(ps: List<&2, Par>, gs: List<&2, F32>, +lr: F32, +n: F32, +k1: F32, +k2: F32) -> List<&2, Par>:
  match ps gs:
    case Con{Par{p,m,v}, pr} Con{g, gr}:
      +gn = (g / n : F32)
      +m2 = (0.9 * m + 0.1 * gn : F32)
      +v2 = (0.999 * v + 0.001 * gn * gn : F32)
      Par{(p - lr * m2 * k1 / (F32.sqrt((v2 * k2 : F32)) + 0.00000001) : F32),m2,v2} <> adam(pr,gr,lr,n,k1,k2)
    case _ _: []

# One Adam step from a batch gradient. Reports the loss and accuracy it saw.
def descend(ps: List<&2, Par>, +t: F32, g: Grad, +lr: F32, +n: F32) -> Fit:
  Grad{gn,l,tp,fp,fn,tn} = g
  +t1 = (t + 1.0 : F32)
  Fit{Opt{adam(ps,flat(gn),lr,n,(1.0 / (1.0 - F32.pow(0.9,t1)) : F32),(1.0 / (1.0 - F32.pow(0.999,t1)) : F32)),t1},(l / n : F32),((tp + tn) / n : F32)}

def step(+k: Nat, opt: Opt, +batch: Batch, +lr: F32, +n: F32) -> Fit:
  Opt{+ps,t} = opt
  descend(ps,t,grad(net_of(k,ps),batch),lr,n)

# n steps of full-batch training.
def train(steps: Nat, +k: Nat, fit: Fit, +batch: Batch, +lr: F32, +n: F32) -> Fit:
  match steps:
    case 0n: fit
    case 1n+p:
      Fit{opt,l,a} = fit
      train(p,k,step(k,opt,batch,lr,n),batch,lr,n)

# Loss and confusion counts of a net on a dataset, without training.
def score(k: Nat, opt: Opt, batch: Batch) -> Grad:
  Opt{ps,t} = opt
  grad(net_of(k,ps),batch)

# 6. Initialisation ------------------------------------------------------------

def centered(+s: U32) -> Win:
  +a = ((2.0 * unit(s,0) - 1.0) * 0.4 : F32)
  +b = ((2.0 * unit(s,1) - 1.0) * 0.4 : F32)
  +c = ((2.0 * unit(s,2) - 1.0) * 0.4 : F32)
  +d = ((2.0 * unit(s,3) - 1.0) * 0.4 : F32)
  +e = ((2.0 * unit(s,4) - 1.0) * 0.4 : F32)
  +f = ((2.0 * unit(s,5) - 1.0) * 0.4 : F32)
  +g = ((2.0 * unit(s,6) - 1.0) * 0.4 : F32)
  +h = ((2.0 * unit(s,7) - 1.0) * 0.4 : F32)
  +i = ((2.0 * unit(s,8) - 1.0) * 0.4 : F32)
  +mu = ((a + b + c + d + e + f + g + h + i) / 9.0 : F32)
  Win{(a - mu : F32),(b - mu : F32),(c - mu : F32),(d - mu : F32),(e - mu : F32),(f - mu : F32),(g - mu : F32),(h - mu : F32),(i - mu : F32)}

# Kernels start as random zero-sum filters, blind to plain brightness.
def init_ks(n: Nat, +s: U32) -> List<&2, Ker>:
  match n:
    case 0n: []
    case 1n+p: Ker{centered(s),0.0,((2.0 * unit(s,9) - 1.0) * 0.5 : F32)} <> init_ks(p,R.hash((s + 2654435769 : U32)))

def pars(ps: List<&2, F32>) -> List<&2, Par>:
  match ps:
    case []: []
    case p <> rest: Par{p,0.0,0.0} <> pars(rest)

def init(k: Nat, seed: U32) -> Opt:
  Opt{pars(flat(Net{init_ks(k,R.hash(seed)),0.0})),0.0}
