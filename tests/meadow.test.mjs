import test from 'node:test';
import assert from 'node:assert/strict';
import M from '../demos/meadow/ecology.bend';
const list=a=>a.reduceRight((tail,head)=>({$:'Con',head,tail}),{$:'Nil'});
const array=xs=>{const out=[];for(let x=xs;x.$==='Con';x=x.tail)out.push(x.head);return out;};
const f=Math.fround,DT=f(.1),NONE=4294967295;
// Independent reference of the hash RNG and the field, mirroring F32 rounding.
const hash=x=>{x>>>=0;let a=Math.imul(x^(x>>>16),2146121005)>>>0;let b=Math.imul(a^(a>>>15),2221713035)>>>0;return (b^(b>>>16))>>>0;};
const random=s=>f(f(hash(s)>>>8)/16777216);
const rnd=(k,salt)=>random((k+Math.imul(salt,2654435769))>>>0);
const jit=(k,salt,w)=>f(f(f(2*rnd(k,salt))-1)*f(w));
const span=(k,salt,lo,hi)=>f(f(lo)+f(f(f(hi)-f(lo))*rnd(k,salt)));
const clamp=(v,lo,hi)=>Math.min(f(hi),Math.max(f(lo),v));
function soil(seed,i,food){
 const x=f(i%48),z=f(Math.floor(i/48)),k=hash((seed^hash((Math.imul(i,2654435761)+7)>>>0))>>>0);
 const wave=f(f(f(.28)*f(Math.sin(f(x*f(.22)))))*f(Math.cos(f(z*f(.19)))));
 const fert=clamp(f(f(f(.85)+wave)+jit(k,0,.2)),.35,1.35);
 return {grass:clamp(f(f(f(f(4*food)/100))*span(k,1,.88,1.12)),0,4),fert};
}
function cells(field){const out=[];const walk=(g,off,d)=>{if(g.$==='Cell')out[off]=g;else if(g.$==='Fork'){walk(g.left,off,d-1);walk(g.right,off+2**(d-1),d-1);}};walk(field,0,12);return out;}
const sum=a=>a.reduce((s,x)=>s+x,0);
const sheep=(o={})=>({$:'Sheep',id:1,gen:0,x:10,z:10,heading:0,age:100,life:300,energy:90,cool:0,speed:1,metab:1,wool:.9,hungry:false,target:NONE,rethink:1,act:0,want:0,...o});
const ledger=w=>{const n=array(w.flock).length;assert.equal(n,w.founders+w.births-w.old-w.starved,'living = founders + births - deaths');assert.equal(w.next,w.founders+w.births+1);};

test('the seeded field and its regrowth match an independent reference cell by cell',()=>{
 let w=M.create(7,0,60,.04,1),ref=Array.from({length:2304},(_,i)=>soil(7,i,60)),got=cells(w.grass);
 assert.equal(got.length,2304);assert.equal(got.filter(Boolean).length,2304);
 for(let i=0;i<2304;i++){assert.equal(got[i].fert,ref[i].fert,`fertility ${i}`);assert.equal(got[i].grass,ref[i].grass,`biomass ${i}`);assert.ok(ref[i].fert>=f(.35)&&ref[i].fert<=f(1.35));}
 const rate=f(.04*DT);
 for(let t=0;t<400;t++)for(const c of ref)c.grass=Math.min(4,f(c.grass+f(rate*c.fert)));
 w=M.run(400n,w);got=cells(w.grass);
 for(let i=0;i<2304;i++)assert.equal(got[i].grass,ref[i].grass,`regrown ${i}`);
 assert.ok(ref.some(c=>c.grass===4)&&ref.every(c=>c.grass<=4),'growth is capped at 4');
});

test('painting removes grass only inside the disk and never leaves [0, 4]',()=>{
 const w=M.create(3,0,80,.015,1),before=cells(w.grass),after=cells(M.paint(w,10,20,3,-9).grass);
 for(let i=0;i<2304;i++){const dx=i%48+.5-10,dz=Math.floor(i/48)+.5-20,inside=dx*dx+dz*dz<=9;assert.equal(after[i].grass,inside?0:before[i].grass);}
 assert.ok(cells(M.paint(w,30,30,5,9).grass).every(c=>c.grass>=0&&c.grass<=4));
});

test('population ledger, bounds, and inherited traits hold over a long seeded run',()=>{
 let w=M.create(42,32,75,.015,1),maxGen=0,sawBirth=false,sawStarve=false;
 assert.equal(array(w.flock).length,32);
 for(let chunk=0;chunk<60;chunk++){
  w=M.run(100n,w);ledger(w);
  const flock=array(w.flock),ids=flock.map(s=>s.id);
  assert.ok(ids.every((id,i)=>i===0||id>ids[i-1]),'flock stays in id order');
  for(const s of flock){
   assert.ok(s.energy>0&&s.energy<=100,`energy ${s.energy}`);assert.ok(s.age<s.life);
   assert.ok(s.speed>=f(.8)&&s.speed<=f(1.25));assert.ok(s.metab>=f(.8)&&s.metab<=f(1.2));assert.ok(s.wool>=f(.72)&&s.wool<=1);
   assert.ok(s.life>=240&&s.life<=360);// Hunger is set from the energy left after basal metabolism; one bite (<= 0.96) or one
   // breeding payment (22) plus a step of walking can move energy before the next check.
   if(s.hungry)assert.ok(s.energy<=94+.97);else assert.ok(s.age===0||s.energy>=68-22-.2,"newborns start at 40 and turn hungry next tick");
   maxGen=Math.max(maxGen,s.gen);
  }
  assert.ok(cells(w.grass).every(c=>c.grass>=0&&c.grass<=4));
  sawBirth||=w.births>0;sawStarve||=w.starved>0;
  const census=M.survey(w);assert.equal(census.n,flock.length);
  assert.equal(sum(array(M.traits(w,0))),flock.length);assert.equal(sum(array(M.traits(w,1))),flock.length);
 }
 assert.ok(sawBirth&&sawStarve&&maxGen>=5,'the run includes breeding, starvation, and several generations');
});

test('every tick conserves biomass: grown - eaten = change',()=>{
 let w=M.create(11,60,40,.015,1);
 for(let t=0;t<300;t++){
  const rate=f(.015*DT),grown=sum(cells(w.grass).map(c=>Math.min(4,f(c.grass+f(rate*c.fert)))));
  w=M.run(1n,w);const now=sum(cells(w.grass).map(c=>c.grass));
  assert.ok(Math.abs(grown-w.eaten-now)<2e-3,`tick ${t}: ${grown} - ${w.eaten} != ${now}`);
 }
});

test('grazing turns biomass into energy at 12 per unit, and a cell is never eaten twice',()=>{
 const w=M.create(5,0,100,.015,1),cell=20*48+20,g0=cells(w.grass)[cell].grass;
 const a=sheep({id:1,energy:50,act:2,target:cell,want:f(.08)}),b=sheep({id:2,energy:30,act:2,target:cell,want:f(.08)});
 let p=M.settle(list([a,b]),{$:'Pasture',grass:w.grass,flock:list([]),eaten:0,starved:0,count:0});
 let [b2,a2]=array(p.flock);
 assert.ok(Math.abs(a2.energy-(50+12*.08))<1e-4&&Math.abs(b2.energy-(30+12*.08))<1e-4);
 assert.ok(Math.abs(cells(p.grass)[cell].grass-(g0-2*.08))<1e-5);assert.ok(Math.abs(p.eaten-.16)<1e-6);
 const bare=M.paint(w,20.5,20.5,.4,f(.1)-g0);
 p=M.settle(list([a,b]),{$:'Pasture',grass:bare.grass,flock:list([]),eaten:0,starved:0,count:0});[b2,a2]=array(p.flock);
 assert.ok(Math.abs(p.eaten-.1)<1e-6,'two sheep share 0.1 biomass: first takes 0.08, second 0.02');
 assert.ok(Math.abs(a2.energy-50-12*.08)<1e-4&&Math.abs(b2.energy-30-12*.02)<1e-4);assert.equal(cells(p.grass)[cell].grass,0);
 const tired=sheep({id:3,energy:0,act:1});
 p=M.settle(list([tired]),{$:'Pasture',grass:w.grass,flock:list([]),eaten:0,starved:0,count:0});
 assert.equal(p.starved,1);assert.equal(p.count,0);
});

test('breeding: each parent pays 22, the lamb gets 40, traits are a mutated average',()=>{
 const a=sheep({id:4,x:10,z:10,speed:.9,metab:1.1,wool:.8,gen:2}),b=sheep({id:6,x:11,z:10.5,speed:1.2,metab:.85,wool:.95,gen:5}),far=sheep({id:9,x:30,z:30});
 const brood=(alive)=>({$:'Turn',rest:list([a,far,b]),brood:{$:'Brood',kept:list([]),lambs:list([]),next:20,alive,halted:false,born:0}});
 const r=M.breed(3n,brood(3),42,7),kept=array(r.kept).reverse(),[lamb]=array(r.lambs);
 assert.equal(r.born,1);assert.equal(r.next,21);assert.equal(r.alive,4);
 const pa=kept.find(s=>s.id===4),pb=kept.find(s=>s.id===6),pf=kept.find(s=>s.id===9);
 assert.equal(pa.energy,68);assert.equal(pb.energy,68);assert.equal(pa.cool,35);assert.equal(pb.cool,35);assert.equal(pf.energy,90);assert.equal(pf.cool,0);
 assert.equal(sum(kept.map(s=>s.energy))+lamb.energy,3*90-4,'4 energy is overhead');
 assert.equal(lamb.id,20);assert.equal(lamb.gen,6);assert.equal(lamb.energy,40);assert.equal(lamb.age,0);assert.equal(lamb.cool,35);
 assert.ok(Math.abs(lamb.speed-1.05)<=.04+1e-6);assert.ok(Math.abs(lamb.metab-.975)<=.04+1e-6);assert.ok(Math.abs(lamb.wool-.875)<=.025+1e-6);
 assert.ok(Math.abs(lamb.x-10.5)<=.8+1e-5&&Math.abs(lamb.z-10.25)<=.8+1e-5);
 const full=M.breed(3n,brood(1000),42,7);
 assert.equal(full.halted,true);assert.equal(full.born,0);assert.ok(array(full.kept).every(s=>s.energy===90),'the ceiling pauses instead of charging parents');
 const hungry=M.breed(2n,{$:'Turn',rest:list([sheep({id:1}),sheep({id:2,hungry:true})]),brood:{$:'Brood',kept:list([]),lambs:list([]),next:3,alive:2,halted:false,born:0}},1,1);
 assert.equal(hungry.born,0,'hungry sheep do not breed');
});

test('aging: starvation takes precedence over old age, and lost energy is reported',()=>{
 const c=M.cull(list([sheep({id:1,energy:.01,age:299.95}),sheep({id:2,energy:50,age:299.95}),sheep({id:3,energy:50,age:10})]),f(.072),DT,{$:'Cull',alive:list([]),old:0,starved:0,basal:0,lost:0});
 assert.equal(c.starved,1);assert.equal(c.old,1);assert.deepEqual(array(c.alive).map(s=>s.id),[3]);
 assert.ok(Math.abs(c.lost-(50-.072))<1e-4);assert.ok(Math.abs(c.basal-(.01+2*.072))<1e-5);
});

test('same seed, same meadow; batching ticks changes nothing; another seed differs',()=>{
 const a=M.run(500n,M.create(9,40,75,.015,1));
 let b=M.create(9,40,75,.015,1);for(let i=0;i<5;i++)b=M.run(100n,b);
 assert.equal(M.fingerprint(a),M.fingerprint(b));assert.deepEqual(array(a.flock),array(b.flock));assert.deepEqual(cells(a.grass),cells(b.grass));
 assert.notEqual(M.fingerprint(M.run(500n,M.create(10,40,75,.015,1))),M.fingerprint(a));
});
