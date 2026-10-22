import test from 'node:test';
import assert from 'node:assert/strict';
import A from '../demos/qlearning/agent.bend';
// An independent JS reference for the gridworld, Q-learning and value iteration
// in demos/qlearning/agent.bend, rounded to F32 at every operation like Bend.
const f=Math.fround,nil={$:'Nil'},T=true;
const list=a=>a.reduceRight((tail,head)=>({$:'Con',head,tail}),nil);
const array=xs=>{const out=[];for(let x=xs;x.$==='Con';x=x.tail)out.push(x.head);return out;};
const near=(a,b,e=1e-5)=>assert.ok(Math.abs(a-b)<=e,`${a} != ${b}`);
const hash=x=>{const a=Math.imul((x^(x>>>16))>>>0,2146121005)>>>0,b=Math.imul((a^(a>>>15))>>>0,2221713035)>>>0;return (b^(b>>>16))>>>0;};
const random=s=>f(f(hash(s)>>>8)/16777216);
const CLIFF=['........','..####..','.....#..','.###.#..','...#....','.#...##.','........','SXXXXXXG'];
const parse=rows=>{const kinds=[...rows.join('')].map(c=>({'#':1,X:2,G:3}[c]??0));return {kinds,start:rows.join('').indexOf('S')};};
const tree=(kinds,d=6,lo=0)=>d===0?{$:'Cell',kind:kinds[lo]}:{$:'Split',l:tree(kinds,d-1,lo),r:tree(kinds,d-1,lo+(1<<(d-1)))};
const leaves=t=>t.$==='Leaf'?[t.v]:[...leaves(t.l),...leaves(t.r)];

function env(kinds){
  const target=(s,a)=>{const x=s%8,y=s/8|0;return [y>0?s-8:s,x<7?s+1:s,y<7?s+8:s,x>0?s-1:s][a];};
  const move=(s,a)=>{const t=target(s,a);return kinds[t]===1?s:t;};
  const reward=(k,cost)=>k===3?1:k===2?-1:f(0-cost);
  return {move,reward,terminal:k=>k>=2};
}
// Ties break by scanning from action o, keeping strictly larger values (as in Bend).
function top(o,v){
  const [a,b,c,d]=[0,1,2,3].map(i=>(o+i)%4),ab=v[b]>v[a]?b:a,cd=v[d]>v[c]?d:c;
  return Math.max(v[c],v[d])>Math.max(v[a],v[b])?cd:ab;
}
function reference(kinds,start,{alpha,gamma,eps,cost},seed,episodes){
  const {move,reward,terminal}=env(kinds),Q=new Float32Array(256),returns=[];let steps=0,path=[];
  for(let e=0;e<episodes;e++){
    let s=start,ret=0,sd=seed;path=[s];
    for(let n=0;n<200;n++){
      const row=[0,1,2,3].map(a=>Q[s*4+a]);
      const a=random(sd)<f(eps)?hash((sd+1)>>>0)%4:top(hash((sd+2)>>>0),row);
      const next=move(s,a),k=kinds[next],r=reward(k,cost),done=terminal(k);
      const future=done?0:Math.max(...[0,1,2,3].map(b=>Q[next*4+b])),old=Q[s*4+a];
      Q[s*4+a]=f(old+f(f(alpha)*f(f(f(r)+f(f(gamma)*future))-old)));
      ret=f(ret+r);steps++;s=next;path.push(s);sd=hash((sd+2654435769)>>>0);
      if(done)break;
    }
    returns.push(ret);seed=hash((seed+1013904223)>>>0);
  }
  return {Q,returns,steps,path};
}
function bfs(kinds,start){
  const {move}=env(kinds),dist=new Map([[start,0]]),queue=[start];
  while(queue.length){const s=queue.shift();if(kinds[s]===3)return dist.get(s);if(kinds[s]===2)continue;for(let a=0;a<4;a++){const t=move(s,a);if(!dist.has(t)){dist.set(t,dist.get(s)+1);queue.push(t);}}}
  return Infinity;
}
const P={alpha:.5,gamma:.95,eps:.1,cost:.04};
const train=(q,world,start,n,seed,p=P)=>A.train(BigInt(n),q,world,start,p.alpha,p.gamma,p.eps,p.cost,seed,nil,0,nil);

test('the environment: edges and walls block, pits and the goal end the episode',()=>{
  const {kinds}=parse(CLIFF),w=tree(kinds),{move}=env(kinds);
  for(let s=0;s<64;s++)for(let a=0;a<4;a++){if(kinds[s]!==1)assert.equal(A.move(w,s,a),move(s,a));}
  assert.equal(A.move(w,0,0),0);assert.equal(A.move(w,0,3),0);assert.equal(A.move(w,9,1),9,'wall at (2,1) blocks');
  near(A.reward(3,.04),1);near(A.reward(2,.04),-1);near(A.reward(0,.04),-.04);
  assert.equal(A.terminal(2),true);assert.equal(A.terminal(1),false);
});

test('the Q table is a tree: put changes one entry and shares the rest',()=>{
  let t=A.zeros(8n);const ref=new Float32Array(256);
  for(let i=0;i<300;i++){const k=hash(i)%256,v=random(i+7);t=A.put(8n,T,t,k,v);ref[k]=v;}
  assert.deepEqual(Float32Array.from(leaves(t)),ref);
  for(let k=0;k<256;k+=17)near(A.get(8n,t,k),ref[k],0);
});

test('Bend Q-learning matches the JS reference update for update, episode for episode',()=>{
  for(const seed of [1,42,777]){
    const {kinds,start}=parse(CLIFF),r=reference(kinds,start,P,seed,60),b=train(A.zeros(8n),tree(kinds),start,60,seed);
    assert.equal(b.steps,r.steps);
    array(b.returns).forEach((x,i)=>near(x,r.returns[i]));
    assert.deepEqual(array(b.path),r.path);
    Float32Array.from(leaves(b.q)).forEach((x,i)=>near(x,r.Q[i]));
  }
});

test('the update rule is Q + alpha (r + gamma max Q\' - Q)',()=>{
  near(A.learn(.2,-.04,.9,.5,.95),f(.2+.5*(-.04+.95*.9-.2)));near(A.learn(0,1,0,1,.9),1);
});

test('learning happens: returns rise and the greedy route becomes a shortest safe path',()=>{
  const {kinds,start}=parse(CLIFF),world=tree(kinds),best=bfs(kinds,start);
  assert.equal(best,9);
  let q=A.zeros(8n);const early=array(train(q,world,start,20,5).returns);
  for(let k=0;k<20;k++)q=train(q,world,start,25,hash(k+100)).q;
  const late=array(train(q,world,start,50,9,{...P,eps:0}).returns),mean=a=>a.reduce((x,y)=>x+y)/a.length;
  assert.ok(mean(late)>mean(early)+0.5,`${mean(early)} -> ${mean(late)}`);
  const route=array(A.walk(64n,T,q,world,start,list([start])));
  assert.equal(route.length-1,best);assert.equal(kinds[route.at(-1)],3);
  // deterministic with seeds
  assert.deepEqual(leaves(train(A.zeros(8n),world,start,30,3).q),leaves(train(A.zeros(8n),world,start,30,3).q));
});

test('value iteration: Bellman sweeps match the reference and converge to the shortest-path value',()=>{
  for(const rows of [CLIFF,['S......#','.####..#','.#..X...','.#.##.#.','...#..#.','##.#.##.','...X...X','.#...#.G']]){
    const {kinds,start}=parse(rows),world=tree(kinds),{move,reward,terminal}=env(kinds),g=.95,c=.04;
    let v=A.zeros(6n),ref=new Float32Array(64),sweeps=0;
    for(;sweeps<100;sweeps++){
      const next=A.sweep(6n,0,v,world,g,c),delta=A.change(next,v);
      ref=Float32Array.from(ref,(_,s)=>kinds[s]>=1?0:Math.max(...[0,1,2,3].map(a=>{const t=move(s,a);return f(reward(kinds[t],c)+f(g*(terminal(kinds[t])?0:ref[t])));})));
      Float32Array.from(leaves(next)).forEach((x,i)=>near(x,ref[i]));
      v=next;if(delta<1e-6)break;
    }
    assert.ok(sweeps<40,`converged in ${sweeps}`);
    const L=bfs(kinds,start);let expected=0;for(let i=0;i<L-1;i++)expected-=c*g**i;expected+=g**(L-1);
    near(A.get(6n,v,start),expected,1e-4);
    const route=array(A.plan(64n,T,v,world,start,g,c,list([start])));
    assert.equal(route.length-1,L);assert.equal(kinds[route.at(-1)],3);
  }
});
