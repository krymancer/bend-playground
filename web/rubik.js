import Cube from '/rubik-engine.js';

const NS='http://www.w3.org/2000/svg';
const faces=['U','R','F','D','L','B'];
const colors=['#f0cf61','#e26964','#57b693','#edf1e9','#ed9c55','#619fd8'];
const $=id=>document.getElementById(id);
const list=values=>values.reduceRight((tail,head)=>({$:'Con',head,tail}),{$:'Nil'});
const array=values=>{const out=[];for(let x=values;x.$==='Con';x=x.tail)out.push(x.head);return out;};
const name=move=>faces[Math.floor(move/3)]+['','2','′'][move%3];
const node=(tag,attrs={},text)=>{const el=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,String(v));if(text!==undefined)el.textContent=text;return el;};
const vec=p=>[p.x-1,p.y-1,p.z-1];
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const add=(a,b)=>a.map((x,i)=>x+b[i]);
const scale=(a,k)=>a.map(x=>x*k);
const sites=Array.from({length:54},(_,i)=>Cube.site(i));
const normals=faces.map((_,i)=>vec(sites[i*9+4].normal));
const axes=faces.map((_,i)=>{
 const mid=vec(sites[i*9+4].position);
 return [vec(sites[i*9+5].position).map((x,j)=>x-mid[j]),vec(sites[i*9+7].position).map((x,j)=>x-mid[j])];
});
function rotate(v,axis,angle){
 const c=Math.cos(angle),s=Math.sin(angle),cross=[axis[1]*v[2]-axis[2]*v[1],axis[2]*v[0]-axis[0]*v[2],axis[0]*v[1]-axis[1]*v[0]];
 return v.map((x,i)=>x*c+cross[i]*s+axis[i]*dot(axis,v)*(1-c));
}
const graphPoints=sites.map((_,i)=>{
 const face=Math.floor(i/9),a=-Math.PI/2+face*Math.PI/3;
 return [250+Math.cos(a)*162+(i%3-1)*19,250+Math.sin(a)*162+(Math.floor(i%9/3)-1)*19];
});
function curve(i,j){
 const a=graphPoints[i],b=graphPoints[j],dx=b[0]-a[0],dy=b[1]-a[1];
 const bend=Math.floor(i/9)===Math.floor(j/9)?.32:.23;
 return {a,b,c:[(a[0]+b[0])/2-dy*bend,(a[1]+b[1])/2+dx*bend]};
}
const curvePath=({a,b,c})=>`M${a} Q${c} ${b}`;
const curveAt=({a,b,c},t)=>a.map((x,i)=>(1-t)**2*x+2*(1-t)*t*c[i]+t*t*b[i]);

export function createRubik(){
 const view=$('rubik-view'),cubeSvg=$('rubik-cube'),graphSvg=$('rubik-graph');
 let history=array(Cube.scramble(20n,42,6)),state=Cube.sequence(list(history),Cube.solved()),values=array(state);
 let queue=[],animation=null,paused=false,active=false,raf=0,last=0,selected=0,yaw=-.6,pitch=.46,drag=null,lastMs=0;
 const cubeGroup=node('g'),edges=node('g'),highlight=node('g'),dots=node('g');
 cubeSvg.append(node('text',{x:250,y:30,'text-anchor':'middle',class:'rubik-svg-label'},'3×3 CUBE'),cubeGroup,node('text',{x:250,y:477,'text-anchor':'middle',class:'rubik-svg-note'},'Drag to rotate the view'));
 graphSvg.append(node('text',{x:250,y:30,'text-anchor':'middle',class:'rubik-svg-label'},'54 STICKER POSITIONS'),edges,highlight,dots);
 const seen=new Set();
 for(let f=0;f<6;f++)for(let i=0;i<54;i++){
  const j=Cube.destination(i,f*3),key=[i,j].sort((a,b)=>a-b).join(':');
  if(i===j||seen.has(key))continue;seen.add(key);
  edges.append(node('path',{d:curvePath(curve(i,j)),fill:'none',stroke:'#607b72','stroke-opacity':.23,'stroke-width':1}));
 }
 const dotNodes=graphPoints.map((p,i)=>{
  const circle=node('circle',{cx:p[0],cy:p[1],r:7,stroke:'#0d1819','stroke-width':2,'data-position':i});
  circle.append(node('title',{},`${faces[Math.floor(i/9)]}${i%9+1}`));dots.append(circle);return circle;
 });
 for(let f=0;f<6;f++){
  const p=graphPoints[f*9+4],label=node('text',{x:p[0],y:p[1]-33,'text-anchor':'middle',fill:colors[f],class:'rubik-face-label',tabindex:0,role:'button','aria-label':`Turn ${faces[f]} clockwise`},faces[f]);
  label.addEventListener('click',e=>enqueue([f*3+(e.shiftKey?2:0)]));
  label.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();enqueue([f*3+(e.shiftKey?2:0)]);}});graphSvg.append(label);
 }
 const cubePolygons=[];
 // Dark cubie bodies also turn, so moving slices have visible interior faces.
 const geometry=[];
 for(let x=0;x<3;x++)for(let y=0;y<3;y++)for(let z=0;z<3;z++)for(let f=0;f<6;f++)geometry.push({p:[x-1,y-1,z-1],normal:normals[f],axes:axes[f],radius:.49,offset:.49,color:'#152022',sticker:-1,point:{$:'Point',x,y,z}});
 for(let i=0;i<54;i++)geometry.push({p:vec(sites[i].position),normal:vec(sites[i].normal),axes:axes[Math.floor(i/9)],radius:.43,offset:.505,sticker:i,point:sites[i].position});
 for(const g of geometry){const polygon=node('polygon',{'stroke-linejoin':'round','stroke-width':.8});cubeGroup.append(polygon);cubePolygons.push({g,polygon});}
 function camera(v){return rotate(rotate(v,[0,1,0],yaw),[1,0,0],pitch);}
 function render(progress=animation?.elapsed/animation?.duration||0){
  const eased=progress*progress*(3-2*progress),face=animation?Math.floor(animation.move/3):selected;
  const angle=animation?-Math.PI/2*([1,2,-1][animation.move%3])*eased:0;
  const projected=[];
  for(const {g,polygon} of cubePolygons){
   const moving=animation&&Cube.in_layer(face,g.point),spin=v=>moving?rotate(v,normals[face],angle):v;
   const n=camera(spin(g.normal));if(n[2]<=.001){polygon.setAttribute('display','none');continue;}polygon.removeAttribute('display');
   const center=add(g.p,scale(g.normal,g.offset));
   const points=[[-1,-1],[1,-1],[1,1],[-1,1]].map(([u,v])=>camera(spin(add(add(center,scale(g.axes[0],u*g.radius)),scale(g.axes[1],v*g.radius)))));
   polygon.setAttribute('points',points.map(p=>`${250+p[0]*79},${251-p[1]*79}`).join(' '));
   polygon.setAttribute('fill',g.sticker<0?g.color:colors[Math.floor(values[g.sticker]/9)]);
   polygon.setAttribute('stroke',g.sticker<0?'#0b1215':'#dce8df55');
   projected.push({polygon,z:camera(spin(center))[2]});
  }
  projected.sort((a,b)=>a.z-b.z).forEach(({polygon})=>cubeGroup.append(polygon));
  highlight.replaceChildren();
  const move=animation?.move??selected*3;
  for(let i=0;i<54;i++){
   const j=Cube.destination(i,move),moving=animation&&i!==j;
   if(i!==j)highlight.append(node('path',{d:curvePath(curve(i,j)),fill:'none',stroke:colors[face],'stroke-width':animation?2:1.5,'stroke-opacity':animation?0.8:0.45}));
   const p=moving?curveAt(curve(i,j),eased):graphPoints[i],circle=dotNodes[i];
   circle.setAttribute('cx',p[0]);circle.setAttribute('cy',p[1]);circle.setAttribute('fill',colors[Math.floor(values[i]/9)]);circle.setAttribute('r',moving?8:7);
  }
 }
 function ui(){
  const solved=Cube.is_solved(state);view.dataset.solved=String(solved);view.dataset.state=values.join(',');view.dataset.busy=String(!!animation||queue.length>0);
  $('rubik-state').textContent=animation?`${name(animation.move)} · ${paused?'paused':'turning'}`:solved?'Solved':'Scrambled';
  $('rubik-moves').textContent=`${history.length} moves in history`;
  $('rubik-history').textContent=history.map(name).join(' ')||'—';
  $('rubik-history').scrollTop=$('rubik-history').scrollHeight;
  $('rubik-timing').textContent=`Bend move: ${lastMs.toFixed(2)} ms · runs locally in your browser`;
  $('rubik-pause').textContent=paused?'▶ Resume':'Ⅱ Pause';$('rubik-pause').disabled=!animation&&!queue.length;
  $('rubik-undo').disabled=!history.length||!!animation||queue.length>0;
  $('rubik-unwind').disabled=!history.length||!!animation||queue.length>0;
  $('rubik-scramble').disabled=!!animation||queue.length>0;
 }
 function next(){
  if(animation||!queue.length)return;
  const task=queue.shift(),start=performance.now(),after=Cube.apply(state,task.move);lastMs=performance.now()-start;
  animation={...task,after,elapsed:0,duration:Number($('rubik-speed').value)};selected=Math.floor(task.move/3);ui();
 }
 function tick(now){
  raf=0;if(!active||paused)return;
  next();if(!animation)return;
  animation.elapsed=Math.min(animation.duration,animation.elapsed+(last?Math.min(now-last,100):0));last=now;
  render(animation.elapsed/animation.duration);
  if(animation.elapsed>=animation.duration){
   state=animation.after;values=array(state);if(animation.undo)history.pop();else history.push(animation.move);animation=null;render(0);ui();
  }
  if(animation||queue.length)raf=requestAnimationFrame(tick);
 }
 function wake(){if(active&&!paused&&!raf){last=0;raf=requestAnimationFrame(tick);}}
 function enqueue(moves,undo=false){queue.push(...moves.map(move=>({move,undo})));paused=false;ui();wake();}
 for(let f=0;f<6;f++)for(let turn=0;turn<3;turn++){
  const move=f*3+turn,b=document.createElement('button');b.type='button';b.textContent=name(move);b.dataset.move=String(move);b.setAttribute('aria-label',`Turn ${faces[f]} ${turn===0?'clockwise':turn===1?'180 degrees':'counterclockwise'}`);
  b.style.setProperty('--face-color',colors[f]);b.addEventListener('click',()=>enqueue([move]));
  const select=()=>{selected=f;if(!animation)render();};b.addEventListener('pointerenter',select);b.addEventListener('focus',select);$('rubik-turns').append(b);
 }
 $('rubik-reset').addEventListener('click',()=>{cancelAnimationFrame(raf);raf=0;queue=[];animation=null;history=[];state=Cube.solved();values=array(state);paused=false;render();ui();});
 $('rubik-undo').addEventListener('click',()=>{if(history.length&&!animation)enqueue([Cube.inverse(history.at(-1))],true);});
 $('rubik-unwind').addEventListener('click',()=>{if(history.length&&!animation)enqueue(array(Cube.undo_moves(list(history),{$:'Nil'})),true);});
 $('rubik-scramble').addEventListener('click',()=>{
  const n=Number($('rubik-length').value);
  if(!Number.isSafeInteger(n)||n<1){$('rubik-length').reportValidity();return;}
  const seed=crypto.getRandomValues(new Uint32Array(1))[0];enqueue(array(Cube.scramble(BigInt(n),seed,history.length?Math.floor(history.at(-1)/3):6)));
 });
 $('rubik-pause').addEventListener('click',()=>{paused=!paused;ui();wake();});
 cubeSvg.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY};cubeSvg.setPointerCapture(e.pointerId);});
 cubeSvg.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-drag.x)*.008;pitch=Math.max(-1.35,Math.min(1.35,pitch+(e.clientY-drag.y)*.008));drag={x:e.clientX,y:e.clientY};render();});
 const endDrag=()=>{drag=null;};cubeSvg.addEventListener('pointerup',endDrag);cubeSvg.addEventListener('pointercancel',endDrag);
 window.addEventListener('keydown',e=>{if(!active||e.repeat||e.ctrlKey||e.metaKey||e.altKey||/INPUT|SELECT|TEXTAREA|BUTTON/.test(e.target.tagName))return;const f=faces.indexOf(e.key.toUpperCase());if(f>=0){e.preventDefault();enqueue([f*3+(e.shiftKey?2:0)]);}});
 render();ui();
 return {setActive(value){active=value;if(!active){if(animation||queue.length)paused=true;cancelAnimationFrame(raf);raf=0;last=0;ui();}else wake();}};
}
