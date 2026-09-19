import { existsSync, mkdirSync, mkdtempSync, rmSync, statSync, readdirSync, writeFileSync, renameSync } from 'node:fs';
import { execFileSync, spawnSync } from 'node:child_process';
import { resolve, join } from 'node:path';
import { deflateSync } from 'node:zlib';
import { setup, root } from './setup.mjs';

export const output = resolve(root, 'output');
function writeOutput(name,data){
  // Publish only complete artifacts, including when a viewer job is cancelled.
  const target=join(output,name),temporary=`${target}.${process.pid}.tmp`;
  writeFileSync(temporary,data);renameSync(temporary,target);
}
export function runtimeEnv() {
  const local = resolve(root, '.tools/cuda');
  const cuda = process.env.CUDA_HOME || (existsSync(`${local}/include/nvrtc.h`) ? local : '/usr/local/cuda');
  return {...process.env, CUDA_HOME:cuda, LD_LIBRARY_PATH:`${cuda}/lib64:${cuda}/lib:${process.env.LD_LIBRARY_PATH || ''}`};
}
function newest(path) {
  return Math.max(statSync(path).mtimeMs, ...readdirSync(path,{withFileTypes:true}).map(e=> e.isDirectory() ? newest(join(path,e.name)) : statSync(join(path,e.name)).mtimeMs));
}
export function buildDemo(demo, gpu) {
  setup();
  const env=runtimeEnv();
  if(gpu && !existsSync(`${env.CUDA_HOME}/include/nvrtc.h`)) throw new Error('CUDA NVRTC is missing. Run npm run setup:gpu first.');
  const binary=resolve(root,`build/${demo}-${gpu?'gpu':'cpu'}`);
  mkdirSync(resolve(root,'build'),{recursive:true});
  const modified=Math.max(newest(resolve(root,`demos/${demo}`)),newest(resolve(root,'demos/shared')), ...(demo==='cubes'?['accelerated.bend','precise.bend'].map(f=>statSync(resolve(root,f)).mtimeMs):[]));
  if(!existsSync(binary) || statSync(binary).mtimeMs<modified || (gpu && !existsSync(`${binary}.gpu`))) {
    console.log(`Building ${demo} for ${gpu?'CUDA':'CPU'}…`);
    // A deliberately absent CUDA_HOME prevents a CPU-only build from creating a CUDA kernel.
    const buildEnv=gpu?env:{...env,CUDA_HOME:resolve(root,'.tools/no-cuda')};
    // A cancelled compiler must not leave a partial binary in the build cache.
    const staging=mkdtempSync(resolve(root,`build/.${demo}-`)),candidate=join(staging,'program');
    try{
      execFileSync(process.execPath,['--stack-size=16000','.tools/bend/bend2/main.ts',`demos/${demo}/main.bend`,'-o',candidate],{cwd:root,env:buildEnv,stdio:'inherit'});
      if(gpu&&!existsSync(`${candidate}.gpu`)) throw new Error('No CUDA kernel was produced. Check your NVIDIA driver.');
      if(gpu)renameSync(`${candidate}.gpu`,`${binary}.gpu`);
      renameSync(candidate,binary);
    }finally{rmSync(staging,{recursive:true,force:true});}
  }
  return binary;
}

export function integer(value, name, allowZero=false) {
  if(!/^\d+$/.test(String(value)) || !Number.isSafeInteger(Number(value)) || Number(value)<(allowZero?0:1) || Number(value)>0xffffffff) throw new Error(`${name} must be ${allowZero?'a nonnegative':'a positive'} U32 integer.`);
  return Number(value);
}
export function config(demo, opts={}) {
  if(!['raytracer','life','cubes'].includes(demo)) throw new Error('Choose raytracer, life, or cubes.');
  const gpu=opts.gpu===true;
  const threads=integer(opts.threads??1,'threads');
  if(demo==='cubes') {
    const digits=integer(opts.digits??3,'digits');
    return {demo,gpu,threads,digits,env:{PI_DIGITS:String(digits)}};
  }
  if(demo==='raytracer') {
    const width=integer(opts.width??512,'width'),height=integer(opts.height??320,'height'), bounces=integer(opts.bounces??3,'bounces',true);
    if(width*height>0x40000000) throw new Error('The pixel tree would overflow its U32 indices.');
    return {demo,gpu,threads,width,height,bounces,env:{RAY_WIDTH:String(width),RAY_HEIGHT:String(height),RAY_BOUNCES:String(bounces),TREE_DEPTH:String(Math.ceil(Math.log2(width*height)))}};
  }
  const size=integer(opts.size??128,'size'), steps=integer(opts.steps??120,'steps',true),seed=integer(opts.seed??42,'seed',true);
  if(size<32 || (size & (size-1)) || size*size/32>0x40000000) throw new Error('Life size must be a power of two, at least 32, with word indices that fit U32.');
  const pattern=opts.pattern??'random';
  const kind={random:0,glider:1,blinker:2}[pattern];
  if(kind===undefined) throw new Error('Pattern must be random, glider, or blinker.');
  return {demo,gpu,threads,size,steps,seed,pattern,env:{LIFE_SIZE:String(size),LIFE_STEPS:String(steps),LIFE_SEED:String(seed),LIFE_PATTERN:String(kind),TREE_DEPTH:String(Math.log2(size*size/32))}};
}

export function execute(c, frames=true) {
  const binary=buildDemo(c.demo,c.gpu);
  const start=performance.now();
  const r=spawnSync(binary,c.gpu?['--gpu','1GB']:['--gpu','off','--threads',String(c.threads)],{
    cwd:root,env:{...runtimeEnv(),...c.env,DUMP:frames?'1':'0'},encoding:'utf8',maxBuffer:512*1024*1024,
  });
  const wallMs=performance.now()-start;
  if(r.error) throw r.error;
  if(r.status!==0) throw new Error(r.stderr||`Bend exited with ${r.status}`);
  const kernelMs=Number(r.stderr.match(/COMPUTE_MS (\d+)/)?.[1]);
  if(!Number.isFinite(kernelMs)) throw new Error('Bend did not report its compute time.');
  const lines=r.stdout.trim().split('\n');
  const metadata={...c,env:undefined,backend:c.gpu?'NVIDIA CUDA':`CPU · ${c.threads} thread${c.threads===1?'':'s'}`,kernelMs,wallMs:Math.round(wallMs),createdAt:new Date().toISOString()};
  return {metadata,lines};
}

export function parseWords(line) { return line.trim().split(/\s+/).filter(Boolean).map(Number); }

export function parseLife(c,result){
  const sparse=result.lines[0].startsWith('LIFE_SPARSE '),words=c.size*c.size/32;
  const frames=result.lines.slice(1).map(line=>{
    if(sparse&&!line.startsWith('S'))throw new Error('Missing sparse Life frame.');
    const values=parseWords(sparse?line.slice(1):line);
    if(sparse){
      if(values.length%2)throw new Error('Incomplete sparse Life frame.');
      for(let i=0;i<values.length;i+=2)if(!Number.isInteger(values[i])||values[i]<0||values[i]>=words||(i&&values[i]<=values[i-2])||!Number.isInteger(values[i+1])||values[i+1]<=0||values[i+1]>0xffffffff)throw new Error('Invalid sparse Life word.');
    }else if(values.length!==words)throw new Error('Incomplete Life replay.');
    return values;
  });
  if(frames.length!==c.steps+1)throw new Error('Incomplete Life replay.');
  return {...result.metadata,encoding:sparse?'sparse-words':'dense-words',frames};
}

export function parseCubes(result) {
  const total=result.lines[0].split(' ')[2];
  if(!/^\d+$/.test(total) || !Number.isSafeInteger(Number(total))) throw new Error('Invalid collision count.');
  const marker=result.lines.indexOf('MOTION');
  const parse=line=>{
    const values=parseWords(line);
    if(values.length!==8||!values.every(Number.isFinite)) throw new Error('Non-finite collision state; this request exceeds the current numerical precision.');
    const [count,x,y,v,w,time,u,wall]=values;
    return {count,x,y,v,w,time,u,wall:wall===1};
  };
  const frames=result.lines.slice(1,marker<0?undefined:marker).map(parse);
  const motion=marker<0?[]:result.lines.slice(marker+1).map(parse);
  if(frames[0]?.count!==0||frames.at(-1)?.count!==Number(total)||frames.some((f,i)=>!Number.isSafeInteger(f.count)||(i>0&&f.count<=frames[i-1].count))) throw new Error('Incomplete collision replay.');
  if(motion.length&&(motion[0].count!==0||motion[0].time!==0||motion.at(-1).count!==Number(total)||motion.some((f,i)=>!Number.isSafeInteger(f.count)||f.x < -0.0001||f.y<f.x-0.0001||(i>0&&(f.time<=motion[i-1].time||f.count<motion[i-1].count)))))throw new Error('Invalid physical-time replay.');
  return {...result.metadata,total,pi:total.length===1?total:`${total[0]}.${total.slice(1)}`,sampled:frames.length!==Number(total)+1,frames,motion};
}
function crc32(bytes) {
  let crc=0xffffffff;
  for(const b of bytes) { crc ^= b; for(let bit=0;bit<8;bit++) crc=(crc>>>1)^((crc&1)?0xedb88320:0); }
  return (crc^0xffffffff)>>>0;
}
function chunk(type,data) {
  const name=Buffer.from(type),length=Buffer.alloc(4),crc=Buffer.alloc(4);
  length.writeUInt32BE(data.length); crc.writeUInt32BE(crc32(Buffer.concat([name,data])));
  return Buffer.concat([length,name,data,crc]);
}
export function png(width,height,pixels) {
  const header=Buffer.alloc(13); header.writeUInt32BE(width,0); header.writeUInt32BE(height,4); header[8]=8; header[9]=2;
  const raw=Buffer.alloc(height*(width*3+1));
  for(let y=0;y<height;y++) for(let x=0;x<width;x++) {
    const p=pixels[y*width+x],at=y*(width*3+1)+1+x*3;
    raw[at]=(p>>>16)&255; raw[at+1]=(p>>>8)&255; raw[at+2]=p&255;
  }
  return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk('IHDR',header),chunk('IDAT',deflateSync(raw)),chunk('IEND',Buffer.alloc(0))]);
}
export function save(c,result) {
  mkdirSync(output,{recursive:true});
  if(c.demo==='raytracer') {
    const pixels=parseWords(result.lines[1]);
    if(pixels.length< c.width*c.height) throw new Error('Incomplete ray-tracer output.');
    writeOutput('raytracer.png',png(c.width,c.height,pixels));
    writeOutput('raytracer.json',JSON.stringify(result.metadata));
  } else if(c.demo==='cubes') {
    const replay=parseCubes(result);
    writeOutput('cubes.json',JSON.stringify(replay));
    console.log(`${replay.total} collisions -> pi = ${replay.pi}; ${replay.frames.length} ${replay.sampled?'sampled':'consecutive'} states.`);
  } else {
    writeOutput('life.json',JSON.stringify(parseLife(c,result)));
  }
  console.log(`${c.demo}: ${result.metadata.backend}; compute ${result.metadata.kernelMs} ms, process ${result.metadata.wallMs} ms.`);
  console.log(`Saved output/${c.demo==='raytracer'?'raytracer.png':`${c.demo}.json`}. View the experiments with npm run playground.`);
}
