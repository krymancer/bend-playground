import {drawCubes} from './cubes.js';
import {duration,nextTime} from './cube-clock.js';
const $=id=>document.getElementById(id);
const names=['raytracer','life','cubes','rubik'];
let rubik=null;
let demo=names.includes(location.hash.slice(1))?location.hash.slice(1):'raytracer';
let ray=null,life=null,cubes=null,frame=0,cursor=0,playing=false,last=0;
const loaded=new Set(),pending=new Map(),dirty=new Set(),versions=new Map();
let running=false,lifePixels=null,lifeWords=null;
const descriptions={
  rubik:['LIVE · 54 STICKERS', 'Turn the cube.', 'Every move updates the cube and graph together.'],
  raytracer:['4 SAMPLES / PIXEL', 'Trace some light.', 'Four spheres, one checkerboard, and a ray for every sample.'],
  life:['B3 / S23 · TOROIDAL GRID', 'Let it evolve.', 'Every cell follows the same rule. Patterns emerge from their neighbors.'],
  cubes:['ELASTIC COLLISIONS', 'Count the collisions.', 'Two blocks, a wall, and the digits of π. Every collision state is computed in Bend.']
};
async function read(name){const r=await fetch(`/output/${name}.json`,{cache:'no-cache'});if(r.status===404)return null;if(!r.ok)throw new Error(`Could not load ${name}.`);return r.json();}
function stats(data){
  $('compute').innerHTML=`${data?.kernelMs??'—'}<small> ms</small>`;
  $('process').innerHTML=`${data?.wallMs??'—'}<small> ms</small>`;
  $('backend-note').textContent=data?`${data.backend} · process time includes startup and output.`:'Run a demo to see its timing.';
}
function pause(){
  playing=false;
  $('play').textContent='▶ Play';$('play').setAttribute('aria-label','Play replay');
  $('cube-play').textContent='▶ Play';$('cube-play').setAttribute('aria-label','Play cube collisions');
}
function paintLife(){
  if(!life)return;
  const size=life.size,scale=size<=128?4:1,cv=$('life-canvas'),ctx=cv.getContext('2d');
  if(cv.width!==size*scale)cv.width=cv.height=size*scale;
  ctx.fillStyle='#101e1b';ctx.fillRect(0,0,cv.width,cv.height);ctx.fillStyle='#bbed91';
  let words=life.frames[frame];
  if(life.encoding==='sparse-words'){
    if(lifeWords?.length!==size*size/32)lifeWords=new Uint32Array(size*size/32);
    lifeWords.fill(0);for(let i=0;i<words.length;i+=2)lifeWords[words[i]]=words[i+1];
    words=lifeWords;
  }
  if(scale===1){
    if(lifePixels?.width!==size)lifePixels=ctx.createImageData(size,size);
    const pixels=lifePixels.data;
    for(let i=0;i<size*size;i++){const alive=(words[i>>>5]>>>(i&31))&1,p=i*4;pixels[p]=alive?187:16;pixels[p+1]=alive?237:30;pixels[p+2]=alive?145:27;pixels[p+3]=255;}
    ctx.putImageData(lifePixels,0,0);
  }else for(let y=0;y<size;y++)for(let x=0;x<size;x++)if((words[y*size/32+(x>>>5)]>>>(x&31))&1)ctx.fillRect(x*scale,y*scale,scale-1,scale-1);
  $('timeline').max=life.frames.length-1;$('timeline').value=frame;
  $('generation').textContent=`${frame} / ${life.frames.length-1}`;
}
const speed=v=>Math.abs(v)>=10000?v.toExponential(2):v.toFixed(2);
function paintCubes(){
  if(!cubes)return;
  const state=drawCubes($('cube-canvas'),cubes,cursor);
  if(!state)return;
  $('cube-timeline').max=duration(cubes);$('cube-timeline').value=cursor;
  $('cube-time').textContent=`${cursor.toFixed(2)} s`;
  $('cube-count').textContent=state.count.toLocaleString();
  $('cube-of').textContent=`of ${Number(cubes.total).toLocaleString()}`;
  $('cube-event').textContent=state.event;$('cube-event').classList.toggle('impact',state.impact);
  $('cube-v').textContent=speed(state.v);$('cube-w').textContent=speed(state.w);
  $('cube-canvas').setAttribute('aria-label',`Collision ${state.count} of ${cubes.total}. Simulation time ${cursor.toFixed(3)} seconds.`);
}
function syncControls(name){
  if(dirty.has(name))return;
  if(name==='raytracer'&&ray){const resolution=`${ray.width},${ray.height}`;if([...$('resolution').options].some(o=>o.value===resolution))$('resolution').value=resolution;$('bounces').value=ray.bounces;}
  if(name==='life'&&life){if([...$('size').options].some(o=>Number(o.value)===life.size))$('size').value=life.size;$('pattern').value=life.pattern;$('steps').value=life.steps;$('seed').value=life.seed;}
  if(name==='cubes'&&cubes)$('cube-digits').value=cubes.digits;
}
function switchDemo(name){
  demo=name;pause();
  document.querySelectorAll('.tab').forEach(b=>{const on=b.dataset.demo===name;b.classList.toggle('active',on);b.setAttribute('aria-selected',String(on));});
  for(const [key,ids] of Object.entries({raytracer:['ray-controls','ray-footer','ray-image'],life:['life-controls','life-footer','life-canvas'],cubes:['cube-controls','cube-footer','cube-replay-note','cube-canvas','cube-hud'],rubik:['rubik-controls','rubik-footer','rubik-view']}))for(const id of ids)$(id).hidden=key!==demo;
  $('stage').classList.toggle('cubes-stage',demo==='cubes');
  $('stage').classList.toggle('rubik-stage',demo==='rubik');
  document.querySelector('.preview').classList.toggle('rubik-preview',demo==='rubik');
  $('backend-label').hidden=$('native-run').hidden=demo==='rubik';
  rubik?.setActive(demo==='rubik');
  ['output-tag','control-title','description'].forEach((id,i)=>$(id).textContent=descriptions[demo][i]);
  $('run').innerHTML=demo==='raytracer'?'Render scene <span>↗</span>':demo==='life'?'Generate replay <span>↗</span>':'Calculate & animate <span>↗</span>';
  if(demo==='rubik'){$('empty').hidden=true;return;}
  const data={raytracer:ray,life,cubes}[demo];stats(data);$('empty').hidden=!!data;
  $('empty').textContent=pending.has(name)?'Loading saved output…':'No output yet. Run the experiment to begin.';
  if(demo==='life'){$('life-canvas').hidden=!life;if(data)$('output-tag').textContent=`${data.size} × ${data.size} · B3 / S23`;paintLife();}
  if(demo==='raytracer'){$('ray-image').hidden=!ray;if(data)$('output-tag').textContent=`${data.width} × ${data.height} · 4 SAMPLES / PIXEL`;}
  if(demo==='cubes'){
    $('cube-canvas').hidden=!cubes;$('cube-hud').hidden=!cubes;
    if(data){
      $('output-tag').textContent=`${data.digits} DIGITS · PHYSICAL TIME`;
      $('cube-result').textContent=data.pi;
      $('cube-total').textContent=`${Number(data.total).toLocaleString()} collisions`;
      $('cube-replay-note').textContent=data.sampled?'Physical-time motion sampled in Bend. The counter includes collisions too fast to see individually; slow playback to inspect the burst.':'Physical-time motion with every impact computed in Bend. Use slow motion or Step to inspect individual collisions.';
      if(data.digits>12)$('cube-replay-note').textContent+=' Above 12 digits, numerical accuracy is unverified.';
    }
    paintCubes();
  }
}
async function refresh(name,force=false){
  if(!force){if(pending.has(name))return pending.get(name);if(loaded.has(name))return;}
  const version=(versions.get(name)||0)+1;versions.set(name,version);
  const task=(async()=>{
    if(name==='rubik'){const {createRubik}=await import('./rubik.js');rubik=createRubik();rubik.setActive(demo==='rubik');loaded.add(name);return;}
    const data=await read(name);
    if(versions.get(name)!==version)return;
    if(name==='raytracer'){ray=data;if(ray)$('ray-image').src=`/output/raytracer.png?t=${encodeURIComponent(ray.createdAt)}`;}
    else if(name==='life'){life=data;frame=0;}
    else{cubes=data;cursor=0;}
    if(!loaded.has(name))syncControls(name);
    loaded.add(name);
  })();
  pending.set(name,task);
  try{await task;}finally{if(versions.get(name)===version){pending.delete(name);if(demo===name)switchDemo(name);}}
}
async function selectDemo(name){
  const loading=refresh(name);switchDemo(name);
  try{await loading;}catch(e){if(demo===name){$('status').textContent=e.message;$('status').classList.add('error');}}
}
document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>{history.replaceState(null,'',`#${b.dataset.demo}`);selectDemo(b.dataset.demo);}));
window.addEventListener('hashchange',()=>{if(names.includes(location.hash.slice(1)))selectDemo(location.hash.slice(1));});
for(const [name,id] of Object.entries({raytracer:'ray-controls',life:'life-controls',cubes:'cube-controls'}))$(id).addEventListener('input',()=>dirty.add(name));
$('play').addEventListener('click',()=>{if(!life)return;if(playing){pause();return;}if(frame===life.frames.length-1)frame=0;playing=true;last=0;$('play').textContent='Ⅱ Pause';$('play').setAttribute('aria-label','Pause replay');});
$('step').addEventListener('click',()=>{pause();if(life){frame=Math.min(frame+1,life.frames.length-1);paintLife();}});
$('timeline').addEventListener('input',()=>{pause();frame=Number($('timeline').value);paintLife();});
$('cube-play').addEventListener('click',()=>{if(!cubes)return;if(playing){pause();return;}if(cursor>=duration(cubes))cursor=0;playing=true;last=performance.now();$('cube-play').textContent='Ⅱ Pause';$('cube-play').setAttribute('aria-label','Pause cube collisions');});
$('cube-step').addEventListener('click',()=>{pause();if(cubes){cursor=nextTime(cubes,cursor);paintCubes();}});
$('cube-reset').addEventListener('click',()=>{pause();cursor=0;paintCubes();});
$('cube-timeline').addEventListener('input',()=>{pause();cursor=Number($('cube-timeline').value);paintCubes();});
function tick(now){
  if(playing&&demo==='life'&&life&&now-last>=1000/Number($('fps').value)){if(last!==0)frame=Math.min(frame+1,life.frames.length-1);last=now;paintLife();if(frame>=life.frames.length-1)pause();}
  if(playing&&demo==='cubes'&&cubes){cursor=Math.min(duration(cubes),cursor+Math.min(now-last,100)/1000*Number($('cube-speed').value));last=now;paintCubes();if(cursor>=duration(cubes))pause();}
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
new ResizeObserver(()=>{if(demo==='cubes'&&cubes)paintCubes();}).observe($('stage'));
$('run').addEventListener('click',async()=>{
  if(running||demo==='rubik')return;
  pause();const btn=$('run'),name=demo;running=true;btn.disabled=true;$('cancel').hidden=false;$('cancel').disabled=false;$('status').classList.remove('error');
  const start=performance.now();let phase='Computing';
  const showProgress=()=>{$('status').textContent=`${phase} ${name==='cubes'?'π blocks':name==='life'?'Life':'ray tracer'} · ${Math.floor((performance.now()-start)/1000)}s elapsed. ${phase==='Computing'?'Running the selected experiment.':''}`;};
  showProgress();const timer=setInterval(showProgress,1000);
  const backend=$('backend').value,body={demo:name,gpu:backend==='gpu',threads:backend==='cpu8'?8:1};
  if(name==='raytracer'){[body.width,body.height]=$('resolution').value.split(',').map(Number);body.bounces=$('bounces').value;}
  else if(name==='cubes')body.digits=$('cube-digits').value;
  else Object.assign(body,{size:$('size').value,steps:$('steps').value,seed:$('seed').value,pattern:$('pattern').value});
  try{
    const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const text=await r.text();if(!r.ok)throw new Error(text);
    phase='Loading';$('cancel').hidden=true;showProgress();
    await refresh(name,true);$('status').textContent=name==='raytracer'?'Render complete. Your PNG is ready.':'Replay ready. Press Play to watch.';
  }catch(e){$('status').textContent=e.message;$('status').classList.add('error');}finally{clearInterval(timer);running=false;btn.disabled=false;$('cancel').hidden=true;}
});
$('cancel').addEventListener('click',async()=>{
  $('cancel').disabled=true;
  try{const r=await fetch('/api/cancel',{method:'POST'});if(!r.ok)throw new Error(await r.text());}
  catch(e){$('status').textContent=e.message;$('status').classList.add('error');$('cancel').disabled=false;}
});
await selectDemo(demo);
