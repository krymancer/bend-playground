import test from 'node:test';
import assert from 'node:assert/strict';
import C from '../demos/convolution/conv.bend';
// Independent float64 reference implementations; Bend computes in F32.
const list=a=>a.reduceRight((tail,head)=>({$:'Con',head,tail}),{$:'Nil'});
const array=xs=>{const out=[];for(let x=xs;x.$==='Con';x=x.tail)out.push(x.head);return out;};
const flat=(m,n)=>{const out=[],stack=[m];while(stack.length&&out.length<n){const t=stack.pop();if(t.$==='Px')out.push(t.v);else stack.push(t.hi,t.lo);}return out;};
const depth=n=>BigInt(Math.max(0,Math.ceil(Math.log2(n))));
const near=(a,b,e=1e-4)=>assert.ok(Math.abs(a-b)<=e*Math.max(1,Math.abs(b)),`${a} != ${b}`);
const close=(a,b,e)=>{assert.equal(a.length,b.length);a.forEach((v,i)=>near(v,b[i],e));};
let s=12345;const rand=()=>(s=(s*1103515245+12345)>>>0)/2**32;
const image=w=>Array.from({length:w*w},()=>rand());
const kernel=()=>Array.from({length:9},()=>rand()*2-1);
const refConv=(chans,ks,w,bias)=>{const o=w-2,out=[];for(let y=0;y<o;y++)for(let x=0;x<o;x++){let acc=bias;chans.forEach((c,ch)=>{for(let j=0;j<9;j++)acc+=c[(y+Math.floor(j/3))*w+x+j%3]*ks[ch][j];});out.push(acc);}return out;};
const refPool=(m,w)=>{const o=w/2,out=[];for(let y=0;y<o;y++)for(let x=0;x<o;x++){const i=2*y*w+2*x;out.push(Math.max(m[i],m[i+1],m[i+w],m[i+w+1]));}return out;};
const tree=(xs,n=xs.length)=>C.from_list(depth(n),list(xs));
const conv=(chans,ks,w,bias)=>{const o=w-2;return flat(C.conv(depth(o*o),0,list(chans.map(c=>tree(c))),list(ks.map(k=>list(k))),depth(w*w),w,bias),o*o);};

test('maps round-trip through from_list and at, padding with zeros',()=>{
 for(const n of [1,5,9,16,100,784]){const xs=image(Math.ceil(Math.sqrt(n))).slice(0,n),d=depth(n),m=C.from_list(d,list(xs));
  xs.forEach((v,i)=>near(C.at(d,m,i),v,1e-6));close(flat(m,n),xs,1e-6);
  if(2**Number(d)>n)assert.equal(C.at(d,m,n),0);}
});

test('single-channel 3x3 convolution matches a direct reference',()=>{
 for(const w of [3,4,7,13,28]){const img=image(w),k=kernel(),b=rand()-.5;close(conv([img],[k],w,b),refConv([img],[k],w,b),1e-4);}
 const img=image(10),id=[0,0,0,0,1,0,0,0,0],out=conv([img],[id],10,0);
 for(let y=0;y<8;y++)for(let x=0;x<8;x++)near(out[y*8+x],img[(y+1)*10+x+1],1e-6);
});

test('multi-channel convolution sums every channel with its own kernel',()=>{
 const w=13,chans=[image(w),image(w),image(w)],ks=[kernel(),kernel(),kernel()];
 close(conv(chans,ks,w,.25),refConv(chans,ks,w,.25),1e-4);
});

test('ReLU and 2x2 max-pooling match the reference pipeline 28 -> 26 -> 26 -> 13',()=>{
 const w=28,img=image(w),k=kernel();
 const c=C.conv(10n,0,list([tree(img)]),list([list(k)]),10n,w,-.1),r=C.relu(c),p=C.pool(8n,0,r,10n,26);
 const ref=refConv([img],[k],w,-.1).map(v=>Math.max(0,v));
 close(flat(r,676),ref,1e-4);close(flat(p,169),refPool(ref,26),1e-4);
 assert.ok(flat(r,676).every(v=>v>=0));
});

test('window products are the nine input*weight terms whose sum is the output',()=>{
 const w=12,img=image(w),k=kernel(),m=tree(img),out=conv([img],[k],w,0);
 for(const [x,y] of [[0,0],[3,7],[9,9]]){const ps=array(C.products(list(k),0,m,depth(w*w),w,x,y));
  assert.equal(ps.length,9);ps.forEach((p,j)=>near(p,img[(y+Math.floor(j/3))*w+x+j%3]*k[j],1e-5));near(ps.reduce((a,b)=>a+b,0),out[y*(w-2)+x],1e-4);}
});

test('gradient descent recovers a target kernel and the loss falls',()=>{
 const w=16,target=[-1,0,1,-2,0,2,-1,0,1],d=depth(w*w),od=depth(14*14),m=C.noise(d,0,w*w,7),img=flat(m,w*w);
 assert.ok(img.every(v=>v>=-1&&v<1));assert.ok(Math.abs(img.reduce((a,b)=>a+b,0)/img.length)<.15);assert.deepEqual(flat(C.noise(d,0,w*w,7),w*w),img);
 const goal=C.conv(od,0,list([m]),list([list(target)]),d,w,0);
 // one step against an analytic gradient
 const k0=Array(9).fill(0),rate=.3,fit1=C.fit(1n,m,d,w,od,goal,list(k0),rate,{$:'Nil'});
 const pred=refConv([img],[k0],w,0),tgt=refConv([img],[target],w,0),err=pred.map((v,i)=>v-tgt[i]),N=err.length;
 near(array(fit1.losses)[0],err.reduce((a,e)=>a+e*e,0)/N,1e-4);
 const grad=Array.from({length:9},(_,j)=>{let g=0;for(let y=0;y<14;y++)for(let x=0;x<14;x++)g+=err[y*14+x]*img[(y+Math.floor(j/3))*w+x+j%3];return 2*g/N;});
 close(array(fit1.kernel),k0.map((v,j)=>v-rate*grad[j]),1e-4);
 let k=list(k0),losses=[];
 for(let i=0;i<20;i++){const f=C.fit(10n,m,d,w,od,goal,k,rate,{$:'Nil'});k=f.kernel;losses.push(...array(f.losses).reverse());}
 assert.ok(losses.at(-1)<losses[0]*1e-3,`loss ${losses[0]} -> ${losses.at(-1)}`);
 close(array(k),target,1e-2);
 // too large a step diverges: the Hessian is about (2/3) I on uniform noise
 assert.ok(!(array(C.fit(30n,m,d,w,od,goal,list(k0),4,{$:'Nil'}).losses)[0]<1));
});

test('a layer applies every filter to every channel, then pools each map',()=>{
 const w=28,chans=[image(w),image(w)],fs=[[kernel(),kernel()],[kernel(),kernel()],[kernel(),kernel()]],bs=[0,-.5,.3];
 const filters=list(fs.map((ks,i)=>({$:'Filter',ks:list(ks.map(k=>list(k))),bias:bs[i]})));
 const maps=array(C.layer(filters,list(chans.map(c=>tree(c))),10n,w,10n)),pooled=array(C.pool_all(list(maps),10n,26,8n));
 assert.equal(maps.length,3);assert.equal(pooled.length,3);
 fs.forEach((ks,i)=>{const ref=refConv(chans,ks,w,bs[i]).map(v=>Math.max(0,v));close(flat(maps[i],676),ref,1e-4);close(flat(pooled[i],169),refPool(ref,26),1e-4);});
});

test('a 28x28 conv + ReLU + pool stays well inside a frame budget',()=>{
 const img=tree(image(28)),k=list([list(kernel())]),t=performance.now();
 for(let i=0;i<8;i++)C.pool(8n,0,C.relu(C.conv(10n,0,list([img]),k,10n,28,0)),10n,26);
 const ms=(performance.now()-t)/8;console.log(`conv+relu+pool 28x28: ${ms.toFixed(2)} ms`);assert.ok(ms<50);
});
