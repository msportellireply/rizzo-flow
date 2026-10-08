import {Simulation, PACES, WEATHER} from './simulation.js';
import {CityRenderer} from './renderer.js';
import {BLOCK, DIRECTIONS, streetName, outgoingFrame} from './network.js';

const $ = id => document.getElementById(id);
const sim = new Simulation();
let renderer;
try {renderer = new CityRenderer($('scene'),sim); $('loading').hidden=true;}
catch(error) {$('loading').textContent='This simulator needs WebGL. Enable hardware acceleration and reload.'; console.error(error);}
let running=false,configured=false,pending=false,generation=0,sequence=0,decisions=0,tokens=0,nextRequest=0,requestController=null;
let lastFrame=performance.now(),lastUi=0,frames=0,fpsSince=performance.now(),lastLogId=-1,lastChoice='',errorNotice='';
let mapFrame={east:0,north:0,scale:.7};
const clock=t=>`${String(Math.floor(t/60)).padStart(2,'0')}:${String(Math.floor(t%60)).padStart(2,'0')}`;
const labels={stop:'Stop',approach:'Approach',crawl:'Crawl',slow:'Slow',steady:'Steady',cruise:'Cruise'};
const attention={open_road:'Open road ahead',signal:'Watching the next signal',pedestrian:'Pedestrian crossing ahead',traffic:'Monitoring surrounding traffic',weather:'Adapting to road conditions',obstruction:'Watching the blocked lane',emergency:'Approaching emergency vehicle'};
for(const [key,label] of Object.entries(labels)){
  const row=document.createElement('div');row.className='prob-row';row.id=`prob-${key}`;
  row.innerHTML=`<span>${label}</span><div class="prob-track"><div class="prob-fill"></div></div><span>—</span>`;$('probabilities').appendChild(row);
}
function invalidate(){generation++;requestController?.abort();sim.lastDecisionAt=-100;nextRequest=0;}
function setRunning(value){
  if(value&&(!configured||!renderer||sim.halted||sim.arrived))return;
  running=value;invalidate();errorNotice='';
  $('start').innerHTML=value?'Ⅱ Pause driving <kbd>SPACE</kbd>':'▶ Start driving <kbd>SPACE</kbd>';
  $('car-state').textContent=value?'AUTONOMOUS':'PAUSED';
  $('inference-status').textContent=value?'CONNECTING':'PAUSED';
  if(value)sim.event('CLM driving engaged','jev');
}
function apiHeaders(){
  const key=$('api-key').value.trim();
  return {'Content-Type':'application/json',...(key?{Authorization:`Bearer ${key}`}:{})};
}
$('api-key').onchange=()=>{setRunning(false);connect();};
async function infer(){
  if(!running||pending||sim.arrived||performance.now()<nextRequest)return;
  pending=true;const gen=generation,epoch=sim.routeEpoch,observedAt=sim.time,state=sim.observe();
  const seq=++sequence;requestController=new AbortController();
  const timer=setTimeout(()=>requestController?.abort(),120000);
  $('inference-status').textContent='THINKING';
  try{
    const response=await fetch('/drive/api/decide',{method:'POST',headers:apiHeaders(),body:JSON.stringify({sequence:seq,state}),signal:requestController.signal});
    const result=await response.json();
    if(gen!==generation||!running)return;
    if(!response.ok)throw new Error(typeof result.detail==='string'?result.detail:'The driving observation was rejected.');
    if(result.sequence!==seq)throw new Error('Mismatched decision response.');
    tokens+=result.input_tokens;decisions++;
    $('latency').textContent=`${result.latency_ms} ms`;$('decisions').textContent=`${decisions} decisions`;$('tokens').textContent=`${tokens.toLocaleString()} tokens`;
    $('raw').textContent=JSON.stringify({observation:state,response:result},null,2);
    if(sim.time-observedAt>2||sim.routeEpoch!==epoch){
      sim.event('Discarded an outdated CLM decision','shield');
      $('inference-status').textContent='STALE';nextRequest=performance.now()+100;return;
    }
    sim.applyDecision(result.answers);
    const {pace,lane,attention:focus}=result.answers;
    $('decision').textContent=`${labels[pace.choice]}${lane.choice==='hold'?' · hold lane':` · move ${lane.choice}`}`;
    $('attention').textContent=attention[focus.choice]||focus.choice;
    $('decision-icon').textContent=pace.choice==='stop'?'Ⅱ':lane.choice==='left'?'↖':lane.choice==='right'?'↗':'↑';
    for(const key of Object.keys(PACES)){
      const row=$(`prob-${key}`),prob=pace.probabilities[key]||0;
      row.classList.toggle('selected',key===pace.choice);row.querySelector('.prob-fill').style.width=`${prob*100}%`;
      row.lastElementChild.textContent=`${Math.round(prob*100)}%`;
    }
    const choice=`${pace.choice}/${lane.choice}/${focus.choice}`;
    if(choice!==lastChoice){sim.event(`${labels[pace.choice]} · ${lane.choice==='hold'?'hold lane':`move ${lane.choice}`} · ${attention[focus.choice]}`,'jev');lastChoice=choice;}
    $('inference-status').textContent='LIVE';nextRequest=performance.now()+550;
  }catch(error){
    if(gen!==generation)return;
    setRunning(false);errorNotice=error.name==='AbortError'?'CLM request timed out. Drive paused; press Start to retry.':error.message;
    sim.event(errorNotice,'danger');$('inference-status').textContent='OFFLINE';
  }finally{clearTimeout(timer);pending=false;requestController=null;}
}
function reset(){
  setRunning(false);sim.reset();lastChoice='';decisions=0;tokens=0;lastLogId=-1;
  $('start').disabled=!configured;$('decision').textContent='Ready when you are';$('attention').textContent='Start a drive to see real model decisions.';
  $('decisions').textContent='0 decisions';$('tokens').textContent='0 tokens';$('latency').textContent='— ms';$('raw').textContent='No decisions yet.';
  for(const key of Object.keys(PACES)){const r=$(`prob-${key}`);r.classList.remove('selected');r.querySelector('.prob-fill').style.width='0%';r.lastElementChild.textContent='—';}
  $('night').checked=false;$('density').value=5;$('density-label').textContent='Moderate';$('shield').checked=true;
  $('route-preference').value='auto';
  setWeather('clear');$('inference-status').textContent='STANDBY';
}
function setWeather(value){sim.weather=value;invalidate();document.querySelectorAll('[data-weather]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.weather===value)));if(sim.time>0)sim.event(`Weather changed to ${WEATHER[value].label.toLowerCase()}`);}
$('start').onclick=()=>setRunning(!running);$('reset').onclick=reset;
$('camera').onclick=()=>{renderer.view=(renderer.view+1)%3;$('camera').querySelector('span').textContent=['Orbit view','Map view','Driver view'][renderer.view];};
document.querySelectorAll('[data-weather]').forEach(b=>b.onclick=()=>setWeather(b.dataset.weather));
document.querySelectorAll('[data-event]').forEach(b=>b.onclick=()=>{sim.inject(b.dataset.event);invalidate();});
$('night').onchange=e=>{sim.night=e.target.checked;invalidate();sim.event(sim.night?'Night driving enabled':'Daylight restored');};
$('shield').onchange=e=>{sim.shield=e.target.checked;sim.event(`Emergency brake assist ${sim.shield?'enabled':'disabled'}`);};
$('route-preference').onchange=e=>{sim.setRouteRequest(e.target.value);invalidate();};
function setDestination(i,j){
  if(!sim.setDestination(i,j)){errorNotice='Choose grid coordinates between -20 and 20.';return;}
  $('destination-east').value=i;$('destination-north').value=j;errorNotice='';invalidate();
  $('start').disabled=!configured||!renderer||sim.halted;
}
$('destination-form').onsubmit=e=>{e.preventDefault();setDestination(Number($('destination-east').value),Number($('destination-north').value));};
$('clear-destination').onclick=()=>{sim.clearDestination();invalidate();$('start').disabled=!configured||!renderer||sim.halted;};
$('route-map').onclick=e=>{
  const canvas=$('route-map'),rect=canvas.getBoundingClientRect();
  const x=(e.clientX-rect.left)*canvas.width/rect.width,y=(e.clientY-rect.top)*canvas.height/rect.height;
  setDestination(Math.round((mapFrame.east+(x-180)/mapFrame.scale)/BLOCK),Math.round((mapFrame.north-(y-120)/mapFrame.scale-100)/BLOCK));
};
$('density').oninput=e=>{sim.density=Number(e.target.value);$('density-label').textContent=sim.density<4?'Light':sim.density>6?'Heavy':'Moderate';
  const ordinary=sim.vehicles.filter(v=>sim.localActor(v).direction===0&&v.kind==='car');
  const excess=ordinary.length-(sim.density*2+1);
  if(excess>0){const remove=new Set(ordinary.sort((a,b)=>sim.localActor(b).s-sim.localActor(a).s).filter(v=>sim.localActor(v).s-sim.s>60).slice(0,excess).map(v=>v.id));sim.vehicles=sim.vehicles.filter(v=>!remove.has(v.id));}
  invalidate();};
$('info-button').onclick=()=>{$('about').showModal();if(running)setRunning(false);};$('close-about').onclick=()=>$('about').close();
document.addEventListener('keydown',e=>{
  if(['INPUT','BUTTON','SUMMARY','TEXTAREA','SELECT'].includes(document.activeElement.tagName)||$('about').open)return;
  if(e.code==='Space'){e.preventDefault();setRunning(!running);}
  if(e.code==='KeyC')$('camera').click();if(e.code==='KeyR')reset();
});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&running)setRunning(false);});
function ui(){
  if(renderer){const position=renderer.egoScreenPosition();const label=document.querySelector('.road-label');label.style.left=`${position.x}px`;label.style.top=`${position.y}px`;label.style.display=renderer.view===2?'none':'flex';}
  $('speed').textContent=Math.round(sim.speed*3.6);$('target').textContent=Math.round(sim.targetSpeed*3.6);
  $('speed-bar').style.width=`${Math.min(100,sim.speed*3.6/50*100)}%`;
  $('distance').innerHTML=sim.distance<1000?`${Math.round(sim.distance)} <small>m</small>`:`${(sim.distance/1000).toFixed(2)} <small>km</small>`;
  $('grip').innerHTML=`${Math.round(sim.environment.grip*100)}<small>%</small>`;
  const sig=sim.signal;$('signal-dot').className=`signal-dot ${sig.color}`;$('signal-distance').textContent=sig.distance<0?'Crossing':`${Math.round(sig.distance)} m`;
  $('place').textContent=`${DIRECTIONS[sim.frame.heading]} on ${streetName(sim.frame)}`;
  const route=sim.plan?.direction;
  $('route-action').textContent=sim.turn?`Turning ${sim.turn.direction}`:route?`${{left:'↰ Turn left',right:'↱ Turn right',straight:'↑ Continue straight'}[route]} · ${Math.max(0,Math.round(sig.distance))} m`:'CLM is choosing the next street';
  $('route-street').textContent=route?streetName(outgoingFrame(sim.frame,sig.center,route)):streetName(sim.frame);
  $('route-preference').value=sim.routeRequest;
  $('route-preference').disabled=!!sim.destination;
  $('clear-destination').disabled=!sim.destination;
  const goal=sim.destination;
  $('destination-status').textContent=goal?`${sim.arrived?'Arrived at':'Destination'} (${goal.east/BLOCK}, ${(goal.north-100)/BLOCK})${sim.arrived?'':` · ${Math.round(Math.hypot(sim.pose.east-goal.east,sim.pose.north-goal.north))} m away`}`:'No destination · free exploration';
  $('turn-count').textContent=`${sim.turns} TURNS`;
  $('passing-status').textContent=sim.passing?'Passing · waiting for safe clearance to return':`${sim.passes} passes completed · ${sim.lane===0?'left lane':'right lane'}`;
  drawMap();
  $('weather-pill').textContent=`${{clear:'☀',rain:'☂',fog:'≋',snow:'❄'}[sim.weather]} ${WEATHER[sim.weather].label}${sim.night?' · Night':''} · 50 km/h limit`;
  document.body.classList.toggle('night',sim.night);
  $('interventions').textContent=sim.interventions;$('violations').textContent=sim.redLights;$('collisions').textContent=sim.collisions;$('elapsed').textContent=clock(sim.time);
  const notice=errorNotice||(sim.halted?'Collision detected. Reset to drive again.':sim.arrived?'Destination reached. Choose another destination to continue.':running?sim.intervention:'');
  $('notice').hidden=!notice;$('notice').textContent=notice;$('notice').classList.toggle('error',!!errorNotice||sim.halted);
  if(sim.halted&&running){setRunning(false);$('start').disabled=true;$('car-state').textContent='STOPPED';}
  if(sim.arrived&&running){setRunning(false);$('start').disabled=true;$('car-state').textContent='ARRIVED';$('inference-status').textContent='ARRIVED';}
  if(sim.events[0]?.id!==lastLogId){
    lastLogId=sim.events[0]?.id;
    $('log').replaceChildren();
    if(!sim.events.length){const p=document.createElement('p');p.className='empty';p.textContent='Your drive starts here.';$('log').appendChild(p);}
    for(const e of sim.events.slice(0,7)){const row=document.createElement('div');row.className=`log-row ${e.type}`;const t=document.createElement('time');t.textContent=clock(e.time);const text=document.createElement('span');text.textContent=e.text;row.append(t,text);$('log').appendChild(row);}
  }
}
function drawMap(){
  const canvas=$('route-map'),ctx=canvas.getContext('2d'),ego=sim.pose,route=sim.destinationRoute();
  const points=[ego,...route,...(sim.destination?[sim.destination]:[])];
  const minE=Math.min(...points.map(p=>p.east)),maxE=Math.max(...points.map(p=>p.east));
  const minN=Math.min(...points.map(p=>p.north)),maxN=Math.max(...points.map(p=>p.north));
  const scale=Math.min(.7,300/(maxE-minE+60),180/(maxN-minN+60));
  mapFrame={east:(minE+maxE)/2,north:(minN+maxN)/2,scale};
  const px=e=>180+(e-mapFrame.east)*scale,py=n=>120-(n-mapFrame.north)*scale;
  ctx.clearRect(0,0,360,240);ctx.fillStyle='#e8eee2';ctx.fillRect(0,0,360,240);
  ctx.strokeStyle='#b6c9b8';ctx.lineWidth=Math.max(2,Math.min(12,16*scale));
  for(let i=Math.floor((mapFrame.east-180/scale)/BLOCK);i<=Math.ceil((mapFrame.east+180/scale)/BLOCK);i++){
    const x=px(i*BLOCK);ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,240);ctx.stroke();
  }
  for(let j=Math.floor((mapFrame.north-120/scale-100)/BLOCK);j<=Math.ceil((mapFrame.north+120/scale-100)/BLOCK);j++){
    const y=py(100+j*BLOCK);ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(360,y);ctx.stroke();
  }
  ctx.strokeStyle='#2c9161';ctx.lineWidth=4;ctx.beginPath();
  sim.trail.forEach((p,i)=>i?ctx.lineTo(px(p.east),py(p.north)):ctx.moveTo(px(p.east),py(p.north)));ctx.stroke();
  if(sim.destination){
    ctx.strokeStyle='#bd7b20';ctx.lineWidth=4;ctx.setLineDash([8,6]);ctx.beginPath();ctx.moveTo(px(ego.east),py(ego.north));
    route.forEach(p=>ctx.lineTo(px(p.east),py(p.north)));ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle=sim.arrived?'#2c9161':'#bd7b20';ctx.beginPath();ctx.arc(px(sim.destination.east),py(sim.destination.north),8,0,Math.PI*2);ctx.fill();
  }
  ctx.save();ctx.translate(px(ego.east),py(ego.north));ctx.rotate(ego.heading);ctx.fillStyle='#184d37';ctx.beginPath();ctx.moveTo(0,-10);ctx.lineTo(7,8);ctx.lineTo(0,5);ctx.lineTo(-7,8);ctx.closePath();ctx.fill();ctx.restore();
  ctx.fillStyle='#5b7765';ctx.font='18px sans-serif';ctx.fillText('N ↑',16,26);
  $('map-caption').textContent=sim.destination?'Gold: planned route · dot: destination':'Click a junction to set your destination';
}
function frame(now){
  const dt=Math.min((now-lastFrame)/1000,.05);lastFrame=now;
  // Local inference can take seconds: keep the observation current by pausing physics.
  if(running){if(!pending)for(let left=dt;left>0;left-=1/60)sim.step(Math.min(left,1/60));infer();}
  renderer?.render(dt);
  if(now-lastUi>100){ui();lastUi=now;}
  frames++;if(now-fpsSince>1000){$('fps').textContent=`${Math.round(frames*1000/(now-fpsSince))} FPS`;fpsSince=now;frames=0;}
  requestAnimationFrame(frame);
}
async function connect(){
  configured=false;$('start').disabled=true;errorNotice='';
  try{
    const response=await fetch('/drive/api/status',{headers:apiHeaders()});
    if(response.status===401)throw new Error('Enter the configured CLM API key.');
    if(!response.ok)throw new Error('Cannot reach the simulator server. Restart it and reload.');
    const status=await response.json();configured=status.configured;
    $('connection').textContent=configured?`CLM connected · ${status.model}`:'Model unavailable';
    if(!configured)errorNotice='Start the Qwen3-8B encoder and CLM server, then reload.';
  }catch(error){errorNotice=error.message;$('connection').textContent='Disconnected';}
  $('connection-dot').classList.toggle('ready',configured);$('start').disabled=!configured||!renderer;
}
connect();requestAnimationFrame(frame);
