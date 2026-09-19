import { parseArgs } from 'node:util';
import { availableParallelism } from 'node:os';
import { mkdirSync,writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { config,execute,parseWords,output } from './demo-lib.mjs';

const {values}=parseArgs({options:{demo:{type:'string',default:'both'},width:{type:'string'},height:{type:'string'},bounces:{type:'string'},size:{type:'string'},steps:{type:'string'},seed:{type:'string'},pattern:{type:'string'}}});
const demos=values.demo==='both'?['raytracer','life']:[values.demo];
const results=[];
for(const demo of demos) {
  let reference;
  for(const backend of [{threads:1},{threads:Math.min(8,availableParallelism())},{gpu:true}]) {
    const c=config(demo,{...values,...backend});
    const result=execute(c,demo==='raytracer');
    const data=parseWords(result.lines[1]);
    if(!reference) reference=data;
    if(data.length!==reference.length) throw new Error(`${demo}: result lengths differ`);
    let maxDifference=0;
    for(let i=0;i<data.length;i++) {
      if(demo==='life' && data[i]!==reference[i]) throw new Error(`Life CPU/GPU mismatch at word ${i}`);
      if(demo==='raytracer') for(const shift of [0,8,16]) maxDifference=Math.max(maxDifference,Math.abs(((data[i]>>>shift)&255)-((reference[i]>>>shift)&255)));
    }
    if(maxDifference>2) throw new Error(`Ray CPU/GPU color difference ${maxDifference} exceeds 2/255.`);
    results.push({...result.metadata,maxChannelDifference:maxDifference});
    console.log(`${demo.padEnd(10)} ${result.metadata.backend.padEnd(18)} compute ${String(result.metadata.kernelMs).padStart(6)} ms | process ${String(result.metadata.wallMs).padStart(6)} ms | ${demo==='life'?'exact match':`max RGB difference ${maxDifference}/255`}`);
  }
}
mkdirSync(output,{recursive:true});
writeFileSync(join(output,'benchmarks.json'),JSON.stringify(results,null,2));
console.log('Saved output/benchmarks.json. Compute includes native Bend scheduling; process includes CUDA initialization and output. Single runs, not statistical benchmarks.');
