import {duration,stateAt} from './cube-clock.js';
const countFormat=new Intl.NumberFormat('en-US');
const speed=value=>Math.abs(value)>=10000?value.toExponential(2):value.toFixed(2);
const extents=new WeakMap();

function polygon(ctx,points,fill){ctx.beginPath();points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.fillStyle=fill;ctx.fill();}
function block(ctx,x,floor,size,color,label,velocity,labelOffset=34){
  const y=floor-size;
  ctx.fillStyle=color;ctx.fillRect(x,y,size,size);
  ctx.strokeStyle='#ffffff25';ctx.strokeRect(x+.5,y+.5,size-1,size-1);
  ctx.textAlign='center';ctx.fillStyle='#10201d';ctx.font='600 18px ui-monospace, monospace';ctx.fillText(label,x+size/2,y+size/2+6);
  ctx.fillStyle='#aec1b8';ctx.font='15px ui-monospace, monospace';ctx.fillText(`v = ${speed(velocity)}`,x+size/2,floor+labelOffset);
  if(Math.abs(velocity)>1e-6){const start=x+size/2,end=start+Math.sign(velocity)*40,ay=y-29;ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(start,ay);ctx.lineTo(end,ay);ctx.stroke();polygon(ctx,[[end,ay],[end-Math.sign(velocity)*7,ay-4],[end-Math.sign(velocity)*7,ay+4]],color);}
}

export function drawCubes(canvas,data,time){
  const ctx=canvas.getContext('2d'),width=1000,height=530;
  ctx.clearRect(0,0,width,height);ctx.fillStyle='#101b1d';ctx.fillRect(0,0,width,height);
  if(!data)return;
  const s=stateAt(data,time),{x,y}=s;
  let lo=0,hi=data.frames.length;
  while(lo+1<hi){const mid=(lo+hi)>>>1;if(data.frames[mid].count<=s.count)lo=mid;else hi=mid;}
  const index=lo,impact=!s.sampled&&s.count>0&&time-s.eventTime<.06;
  if(!extents.has(data))extents.set(data,Math.max(6,stateAt(data,duration(data)).y,...data.frames.map(f=>f.y)));
  const floor=350,wall=72,small=54,scale=(width-240)/extents.get(data);
  ctx.textAlign='left';ctx.fillStyle='#89a39b';ctx.font='12px ui-monospace, monospace';ctx.fillText('COLLISIONS',44,44);
  ctx.fillStyle='#e9efea';ctx.font=`500 ${s.count>=1e10?36:48}px ui-monospace, monospace`;ctx.fillText(countFormat.format(s.count),42,101);
  ctx.fillStyle='#88a39a';ctx.font='14px ui-monospace, monospace';ctx.fillText(`of ${countFormat.format(Number(data.total))}`,45,128);
  ctx.fillStyle='#bbed91';ctx.font='13px ui-monospace, monospace';
  const event=s.count===0?'HEAVY BLOCK MOVES LEFT':s.count===Number(data.total)?'COMPLETE · BLOCKS MOVE APART':impact?(s.wall?'IMPACT · SMALL BLOCK / WALL':'IMPACT · BLOCK / BLOCK'):'FRICTIONLESS MOTION';
  ctx.fillText(event,45,170);
  ctx.strokeStyle='#32483f';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(wall,floor);ctx.lineTo(960,floor);ctx.moveTo(wall,215);ctx.lineTo(wall,floor);ctx.stroke();
  ctx.strokeStyle=s.wall&&impact?'#bbed91':'#536d62';ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(wall,215);ctx.lineTo(wall,floor);ctx.stroke();
  ctx.lineWidth=1;ctx.strokeStyle='#354a42';for(let j=0;j<11;j++){ctx.beginPath();ctx.moveTo(wall-12,224+j*12);ctx.lineTo(wall,215+j*12);ctx.stroke();}
  for(let j=0;j<6;j++){ctx.beginPath();ctx.moveTo(wall+j*scale,floor);ctx.lineTo(wall+j*scale+8,floor+8);ctx.stroke();}
  block(ctx,wall+x*scale,floor,small,'#bbed91','1',s.v);
  const label=data.digits===1?'1':data.digits<=3?String(100**(data.digits-1)):`10^${2*(data.digits-1)}`;
  block(ctx,wall+y*scale+small,floor,data.digits===1?small:96,'#68b9cc',label,s.w,58);

  const cx=820,cy=123,r=75;
  ctx.strokeStyle='#345048';ctx.lineWidth=1;ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);ctx.moveTo(cx-r-10,cy);ctx.lineTo(cx+r+10,cy);ctx.moveTo(cx,cy-r-10);ctx.lineTo(cx,cy+r+10);ctx.stroke();
  ctx.strokeStyle='#bbed9155';ctx.beginPath();for(let j=Math.max(0,index-120);j<=index;j++){const f=data.frames[j],px=cx+f.w*r,py=cy-f.u*r;if(j===Math.max(0,index-120))ctx.moveTo(px,py);else ctx.lineTo(px,py);}ctx.stroke();
  ctx.fillStyle='#bbed91';ctx.beginPath();ctx.arc(cx+s.w*r,cy-s.u*r,5,0,Math.PI*2);ctx.fill();
  ctx.textAlign='center';ctx.fillStyle='#8eaaa3';ctx.font='11px ui-monospace, monospace';ctx.fillText('VELOCITY SPACE',cx,22);ctx.fillText('heavy velocity →',cx,cy+r+28);
  ctx.textAlign='left';ctx.fillStyle='#426056';ctx.fillRect(44,426,912,1);
  ctx.fillStyle='#8eaaa3';ctx.font='12px ui-monospace, monospace';ctx.fillText('PERFECTLY ELASTIC',45,463);ctx.fillText('NO FRICTION',360,463);ctx.fillText('BEND-COMPUTED STATES',670,463);
  ctx.fillStyle='#648078';ctx.font='12px ui-monospace, monospace';ctx.fillText(`t = ${time.toFixed(3)} s · Physical time · Block sizes are illustrative.`,45,495);
  return s;
}
