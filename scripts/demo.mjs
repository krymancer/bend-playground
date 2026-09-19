import { parseArgs } from 'node:util';
import { config,execute,save } from './demo-lib.mjs';

const {values,positionals}=parseArgs({allowPositionals:true,options:{gpu:{type:'boolean'},threads:{type:'string'},width:{type:'string'},height:{type:'string'},bounces:{type:'string'},size:{type:'string'},steps:{type:'string'},seed:{type:'string'},pattern:{type:'string'},digits:{type:'string'},help:{type:'boolean'}}});
if(values.help) console.log('npm run raytracer -- [--gpu] [--threads 8] [--width 512 --height 320 --bounces 3]\nnpm run life -- [--gpu] [--threads 8] [--size 128 --steps 120 --pattern random|glider|blinker --seed 42]\nnpm run cubes -- [--gpu] [--digits 3]');
else {
  try { const c=config(positionals[0],values); save(c,execute(c)); }
  catch(e) {console.error(e.message);process.exitCode=1;}
}
