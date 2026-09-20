import test from 'node:test';
import assert from 'node:assert/strict';
import P from '../demos/raytracer/path.bend';
import V from '../demos/raytracer/ray.bend';
import {config} from '../scripts/demo-lib.mjs';
const vec=(x,y,z)=>({$:'Vec',x,y,z});
const T=true,empty={$:'Empty'};
const dot=(a,b)=>a.x*b.x+a.y*b.y+a.z*b.z;
const close=(a,b,t=1e-5)=>assert.ok(Math.abs(a-b)<t,`${a} != ${b}`);
const balls=t=>t.$==='Ball'?[t]:t.$==='Branch'?[...balls(t.left),...balls(t.right)]:[];
const world=P.world(2,42);

test('seeded field has hundreds of spheres and all three material types',()=>{
  const all=balls(world);
  assert.ok(all.length>450&&all.length<484);
  assert.deepEqual(new Set(all.map(b=>b.kind)),new Set([0,1,2]));
  assert.deepEqual(world,P.world(2,42));
  assert.notDeepEqual(world,P.world(2,43));
  assert.deepEqual(P.world(1,42),empty);
  for(let i=0;i<1000;i++){const r=P.random(i);assert.ok(r>=0&&r<1);close(dot(P.random_unit(i),P.random_unit(i)),1,1e-5);}
});

test('spatial hierarchy agrees with brute-force sphere intersections',()=>{
  const all=balls(world);
  for(let i=0;i<80;i++){
    const origin=vec(13,2,3),target=vec(P.random(i)*22-11,0.2,P.random(i+100)*22-11),direction=V.unit(V.sub(target,origin));
    const actual=P.hit_tree(world,T,origin,direction,P.miss());
    const expected=Math.min(...all.map(b=>V.intersect(origin,direction,b.center,b.radius)));
    close(actual.distance,expected,1e-4);
  }
  assert.equal(P.box_hit(vec(0,0,0),vec(1,1,1),vec(2,.5,.5),vec(0,1,0),100), false);
  assert.equal(P.box_hit(vec(0,0,0),vec(1,1,1),vec(.5,2,.5),vec(0,-1,0),100), true);
});

test('surface normals face the ray from both outside and inside glass',()=>{
  const args=[vec(0,0,0),1,2,vec(1,1,1),0];
  const outside=P.ball_hit(...args,vec(0,0,-3),vec(0,0,1));
  close(outside.distance,2);assert.equal(outside.front,true);assert.deepEqual(outside.normal,vec(0,0,-1));
  const inside=P.ball_hit(...args,vec(0,0,0),vec(0,0,1));
  close(inside.distance,1);assert.equal(inside.front,false);close(inside.normal.z,-1);
});

test('dielectrics obey Snell’s law and total internal reflection',()=>{
  const normal=vec(0,1,0),direction=vec(.6,-.8,0);
  const refracted=P.refract(direction,normal,1/1.5,.8);
  close(refracted.x,.4);close(dot(refracted,refracted),1);
  close(P.reflectance(1,1.5),.04);
  const inside=vec(.9,-Math.sqrt(1-.81),0);
  const reflected=P.glass(inside,normal,1.5,42);
  close(reflected.x,.9);assert.ok(reflected.y>0);
});

test('matte scattering stays above the surface; polished metal reflects',()=>{
  for(let seed=0;seed<128;seed++){
    const b=P.scatter(0,vec(0,-1,0),vec(0,1,0),vec(.4,.5,.6),0,T,seed);
    assert.equal(b.$,'Bounce');assert.ok(b.direction.y>=0);close(dot(b.direction,b.direction),1,1e-4);
  }
  const metal=P.scatter(1,vec(.6,-.8,0),vec(0,1,0),vec(.7,.6,.5),0,T,42);
  close(metal.direction.x,.6);close(metal.direction.y,.8);assert.deepEqual(metal.attenuation,vec(.7,.6,.5));
});

test('escaped paths stop at the sky and exhausted paths return black',()=>{
  const sky=P.trace(1000000n,empty,vec(0,3,0),vec(0,1,0),42);close(sky.x,.5);close(sky.y,.7);close(sky.z,1);
  assert.deepEqual(P.trace(0n,empty,vec(0,3,0),vec(0,1,0),42),vec(0,0,0));
});

test('both path-traced scenes yield deterministic, distinct, finite RGB values',()=>{
  const pictures=[];
  for(const scene of [1,2]){
    const tree=P.world(scene,42),colors=[];
    for(let y=0;y<8;y++)for(let x=0;x<12;x++)colors.push(P.pixel(x,y,12,8,8n,tree,4,42));
    assert.ok(colors.every(p=>Number.isInteger(p)&&p>=0&&p<=0xffffff));assert.ok(new Set(colors).size>60);
    assert.equal(colors[54],P.pixel(6,4,12,8,8n,tree,4,42));pictures.push(colors);
  }
  assert.notDeepEqual(...pictures);
});

test('scene settings reach Bend and invalid settings are rejected',()=>{
  const c=config('raytracer',{scene:'weekend',samples:32,seed:7});
  assert.equal(c.env.RAY_SCENE,'2');assert.equal(c.env.RAY_SAMPLES,'32');assert.equal(c.env.RAY_SEED,'7');assert.equal(c.bounces,12);
  assert.equal(config('raytracer').samples,4);
  assert.throws(()=>config('raytracer',{scene:'unknown'}));
  assert.throws(()=>config('raytracer',{scene:'weekend',samples:0}));
});
