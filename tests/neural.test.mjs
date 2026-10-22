import test from 'node:test';
import assert from 'node:assert/strict';
import N from '../demos/neural/net.bend';
const list=a=>a.reduceRight((tail,head)=>({$:'Con',head,tail}),{$:'Nil'});
const array=xs=>{const out=[];for(let x=xs;x.$==='Con';x=x.tail)out.push(x.head);return out;};
const near=(a,b,e=1e-4,m='')=>assert.ok(Math.abs(a-b)<=e*Math.max(1,Math.abs(b)),`${m} ${a} != ${b}`);
const leaves=b=>b.$==='One'?[b.p]:[...leaves(b.l),...leaves(b.r)];
const net=(units,c)=>({$:'Net',units:list(units.map(([wx,wy,b,v])=>({$:'Neuron',wx,wy,b,v}))),c});
const weights=n=>array(n.units).map(u=>[u.wx,u.wy,u.b,u.v]);

// Independent reference: plain JS doubles, written from the math, not from net.bend.
const sig=z=>1/(1+Math.exp(-z));
function forward(W,c,x,y){const h=W.map(([a,b,d])=>Math.tanh(a*x+b*y+d));return {h,z:c+h.reduce((s,hj,j)=>s+W[j][3]*hj,0)};}
const bce=(z,t)=>{const q=sig(z);return -(t*Math.log(q)+(1-t)*Math.log(1-q));};
function refGrad(W,c,pts){const g=W.map(()=>[0,0,0,0]);let gc=0,L=0,hits=0;
 for(const {x,y,t} of pts){const {h,z}=forward(W,c,x,y),d=sig(z)-t;L+=bce(z,t);hits+=(sig(z)>=.5)===(t>=.5)?1:0;gc+=d;
  h.forEach((hj,j)=>{const a=d*W[j][3]*(1-hj*hj);g[j][0]+=a*x;g[j][1]+=a*y;g[j][2]+=a;g[j][3]+=d*hj;});}
 return {g,gc,L,hits};}
const W0=[[1.2,-.7,.3,.9],[-.4,1.5,-.2,-1.1],[.8,.6,.1,.5]],c0=-.15,N0=net(W0,c0);

test('a single neuron is a weighted sum, a bias and an activation',()=>{
 for(const [w1,w2,b,x,y] of [[1,2,.5,.3,-.4],[-2,.5,-.1,.9,.9],[0,0,0,.2,.2]]){const z=w1*x+w2*y+b;
  near(N.neuron(0n,w1,w2,b,x,y),sig(z));near(N.neuron(1n,w1,w2,b,x,y),Math.tanh(z));assert.equal(N.neuron(2n,w1,w2,b,x,y),z>=0?1:0);}
 const row=array(N.neuron_row(5n,0n,1,-1,0,.5,-1,.5));row.forEach((v,i)=>near(v,sig(-1+i*.5-.5)));
 const batch=N.build(6n,0n,0,3),pts=leaves(batch);
 assert.equal(N.perceptron(batch,1,1,0),pts.filter(p=>(p.x+p.y>=0)===(p.t>=.5)).length);
});

test('forward pass matches an independent reference',()=>{
 for(const [x,y] of [[0,0],[.5,-.3],[-.9,.8]]){const {z}=forward(W0,c0,x,y);near(N.logit(N0,x,y),z);near(N.predict(N0,x,y),sig(z));}
 const r=array(N.row(4n,N0,.25,-.75,.5));r.forEach((v,i)=>near(v,sig(forward(W0,c0,-.75+i*.5,.25).z)));
 for(const [x,y] of [[.5,.5],[-.5,-.5],[.5,-.5],[-.5,.5]])assert.equal(N.predict(N.xor_net(4),x,y)>=.5,x*y>0);
});

test('backprop gradient equals the reference and finite differences',()=>{
 const batch=N.build(5n,2n,0,9),pts=leaves(batch),G=N.grad(N0,batch),R=refGrad(W0,c0,pts),gu=weights(G);
 near(G.loss,R.L,1e-4,'loss');assert.equal(G.hits,R.hits);near(G.c,R.gc,1e-4,'dc');
 gu.forEach((g,j)=>g.forEach((v,i)=>near(v,R.g[j][i],1e-4,`d${j},${i}`)));
 const lossOf=(W,c)=>pts.reduce((s,{x,y,t})=>s+bce(forward(W,c,x,y).z,t),0),e=1e-5;
 for(let j=0;j<W0.length;j++)for(let i=0;i<4;i++){const P=W0.map(r=>[...r]),M=W0.map(r=>[...r]);P[j][i]+=e;M[j][i]-=e;near(gu[j][i],(lossOf(P,c0)-lossOf(M,c0))/(2*e),2e-3,`fd ${j},${i}`);}
 near(G.c,(lossOf(W0,c0+e)-lossOf(W0,c0-e))/(2*e),2e-3,'fd c');
});

test('one gradient-descent step moves every weight by -lr/n times its gradient',()=>{
 const batch=N.build(4n,1n,0,5),n=16,lr=.7,R=refGrad(W0,c0,leaves(batch));
 const fit=N.train(1n,N.score(N0,batch,n),batch,lr,n);
 weights(fit.net).forEach((w,j)=>w.forEach((v,i)=>near(v,W0[j][i]-lr/n*R.g[j][i])));near(fit.net.c,c0-lr/n*R.gc);
 near(fit.loss,R.L/n);near(fit.acc,R.hits/n);
});

test('datasets are seeded, labelled, inside the plane, and shaped as named',()=>{
 for(const kind of [0n,1n,2n,3n]){const pts=leaves(N.build(8n,kind,0,42));assert.equal(pts.length,256);
  assert.deepEqual(pts,leaves(N.build(8n,kind,0,42)));assert.notDeepEqual(pts,leaves(N.build(8n,kind,0,43)));
  for(const p of pts){assert.ok(p.t===0||p.t===1);assert.ok(Math.abs(p.x)<=1&&Math.abs(p.y)<=1);}
  const ones=pts.filter(p=>p.t===1).length;assert.ok(ones>90&&ones<166,`balanced ${ones}`);
  if(kind===1n)for(const p of pts)assert.equal(p.t,p.x*p.y>0?1:0);
  if(kind===2n)for(const p of pts)assert.equal(p.t,Math.hypot(p.x,p.y)<.5?1:0);}
});

function fit(kind,hidden,lr,epochs){const batch=N.build(8n,kind,0,42);let f=N.score(N.init(hidden,7),batch,256);for(let e=0;e<epochs;e+=50)f=N.train(50n,f,batch,lr,256);return f;}
test('gradient descent learns: loss falls, hidden layers solve what one line cannot',()=>{
 const start=N.score(N.init(4n,7),N.build(8n,1n,0,42),256).loss,xor=fit(1n,4n,1,1000);
 assert.ok(xor.loss<start/10,`xor loss ${start} -> ${xor.loss}`);assert.ok(xor.acc>=.98,`xor acc ${xor.acc}`);
 assert.ok(fit(1n,1n,1,1000).acc<.8,'one hidden neuron cannot solve xor');
 assert.ok(fit(2n,8n,1,1000).acc>=.98,'circle');
 const spiral=fit(3n,16n,2,3000);assert.ok(spiral.acc>=.95,`spiral acc ${spiral.acc}`);
 assert.ok(N.score(N.xor_net(4),N.build(8n,1n,0,42),256).acc>=.95,'hand-built xor net');
});
