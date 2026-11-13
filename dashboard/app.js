const $ = selector => document.querySelector(selector);
const api = async (path, options = {}) => {
  const response = await fetch(path, {headers: {'Content-Type': 'application/json'}, ...options});
  if (!response.ok) throw new Error((await response.text()) || `Request failed: ${response.status}`);
  return response.headers.get('content-type')?.includes('json') ? response.json() : response.text();
};
const state = {catalog:null, socket:null, running:false, paused:false, last:null, temperature:[], maxHistory:120};
const number = (value, digits=1) => Number.isFinite(Number(value)) ? Number(value).toLocaleString(undefined,{minimumFractionDigits:digits,maximumFractionDigits:digits}) : '—';

function toast(message){const element=$('#toast');element.textContent=message;element.classList.add('show');clearTimeout(toast.timer);toast.timer=setTimeout(()=>element.classList.remove('show'),3500)}
function setConnection(online){const badge=$('#connectionBadge');badge.textContent=online?'Backend connected':'Backend offline';badge.className=`connection ${online?'online':'offline'}`}
function setRunStatus(status){$('#runStatus').textContent=status;$('#runStatus').className=`status-chip ${status}`;state.running=status==='running'||status==='paused';state.paused=status==='paused';$('#pauseButton').disabled=!state.running;$('#resetButton').disabled=!state.running;$('#applyButton').disabled=!state.running;$('#compareButton').disabled=!state.running;$('#pauseButton').textContent=state.paused?'Resume':'Pause'}

async function initialize(){
  try{
    state.catalog=await api('/api/live/catalog');setConnection(true);
    const scenarios=state.catalog.scenarios;$('#scenarioSelect').innerHTML=Object.entries(scenarios).map(([key,value])=>`<option value="${key}">${value.label}</option>`).join('');
    $('#tempThreshold').textContent=`${state.catalog.thresholds.temperature_warning_C} °C`;$('#stressThreshold').textContent=`${state.catalog.thresholds.tensile_stress_warning_MPa} MPa`;
    $('#modelName').textContent=state.catalog.model.checkpoint;
    const validation=state.catalog.model.validation_metrics||{};
    const rmse=Number(validation.temperature_rmse_K);
    const rel=Number(validation.concentration_relative_l2);
    $('#modelMode').textContent=Number.isFinite(rmse)&&Number.isFinite(rel)
      ? `${state.catalog.model.training_mode} · held-out T RMSE ${rmse.toFixed(3)} K · concentration RelL2 ${(rel*100).toFixed(2)}%`
      : `${state.catalog.model.training_mode} checkpoint`;
    updateScenarioDescription();connectSocket();
  }catch(error){setConnection(false);toast(`Backend unavailable: ${error.message}`)}
}
function updateScenarioDescription(){const scenario=state.catalog?.scenarios[$('#scenarioSelect').value];if(!scenario)return;$('#scenarioDescription').textContent=scenario.description;$('#currentRange').value=scenario.base_c_rate;$('#coolingRange').value=scenario.cooling_factor;updateControlOutputs()}
function updateControlOutputs(){$('#currentOutput').textContent=$('#currentAuto').checked?'Scenario':`${number(+$('#currentRange').value,1)}C`;$('#coolingOutput').textContent=`${number(+$('#coolingRange').value,1)}×`;$('#currentRange').disabled=$('#currentAuto').checked}

function connectSocket(){
  if(state.socket&&state.socket.readyState<2)return;
  const protocol=location.protocol==='https:'?'wss':'ws';const socket=new WebSocket(`${protocol}://${location.host}/ws/live`);state.socket=socket;
  socket.onopen=()=>setConnection(true);socket.onclose=()=>{setConnection(false);setTimeout(connectSocket,1500)};socket.onerror=()=>setConnection(false);
  socket.onmessage=event=>{const payload=JSON.parse(event.data);if(payload.type==='connection'){setRunStatus(payload.state.status);return}if(payload.type==='session_complete'){setRunStatus('complete');toast('Software discharge session completed');return}if(payload.type==='telemetry')updateLive(payload)};
}

async function start(){try{const response=await api('/api/live/start',{method:'POST',body:JSON.stringify({scenario:$('#scenarioSelect').value,speed:+$('#speedSelect').value})});state.temperature=[];setRunStatus(response.status);$('#sessionId').textContent=response.session_id;$('#exportLink').classList.remove('disabled');$('#exportLink').removeAttribute('aria-disabled');$('#exportLink').href=`/api/live/sessions/${response.session_id}/csv`;toast('Software-only digital twin started')}catch(error){toast(error.message)}}
async function togglePause(){try{const result=await api(state.paused?'/api/live/resume':'/api/live/pause',{method:'POST'});setRunStatus(result.status)}catch(error){toast(error.message)}}
async function reset(){await api('/api/live/stop',{method:'POST'});await start()}
async function applyControl(){try{const body={speed:+$('#speedSelect').value,c_rate:$('#currentAuto').checked?null:+$('#currentRange').value,cooling_factor:+$('#coolingRange').value};await api('/api/live/control',{method:'POST',body:JSON.stringify(body)});toast('Intervention applied to the software plant')}catch(error){toast(error.message)}}

function updateLive(payload){
  state.last=payload;setRunStatus('running');$('#sessionId').textContent=payload.session_id;$('#exportLink').href=`/api/live/sessions/${payload.session_id}/csv`;$('#exportLink').classList.remove('disabled');
  const t=payload.telemetry,p=payload.prediction;
  $('#voltageValue').textContent=number(t.voltage_V,3);$('#currentValue').textContent=number(t.current_A,2);$('#socValue').textContent=number(t.soc_percent,1);$('#measuredTempValue').textContent=number(t.surface_temperature_C,1);$('#predictedTempValue').textContent=number(p.temperature_C,1);$('#stressValue').textContent=number(p.peak_tensile_stress_MPa,1);$('#radiusValue').textContent=number(p.peak_stress_radius_star,2);$('#latencyValue').textContent=number(p.inference_ms,1);$('#surfaceConcentration').textContent=`${number(p.surface_concentration_mol_m3,0)} mol m⁻³`;$('#elapsedTime').textContent=formatTime(payload.simulated_time_s);
  state.temperature.push({time:payload.simulated_time_s,measured:t.surface_temperature_C,predicted:p.temperature_C});if(state.temperature.length>state.maxHistory)state.temperature.shift();
  updateRisk(payload.risk,p.confidence);drawTemperature();drawProfile(p);updateRadialTable(p);updateForecast(payload.forecasts);window.dispatchEvent(new CustomEvent('twin-update',{detail:payload}));
}
function updateRisk(risk,confidence){$('#riskBadge').textContent=risk.level;$('#riskBadge').className=`risk ${risk.level}`;$('#riskScore').textContent=`${Math.round(confidence*100)}%`;$('#riskRecommendation').textContent=risk.recommendation;$('#riskReasons').innerHTML=(risk.reasons.length?risk.reasons:['No active alerts.']).map(reason=>`<li>${reason}</li>`).join('')}
function formatTime(seconds){const min=Math.floor(seconds/60),sec=Math.floor(seconds%60);return `${String(min).padStart(2,'0')}:${String(sec).padStart(2,'0')}`}

function canvasSetup(canvas){const rect=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);return{ctx,width:rect.width,height:rect.height}}
function axes(ctx,w,h,pad,min,max){ctx.strokeStyle='#34506d';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(pad,pad);ctx.lineTo(pad,h-pad);ctx.lineTo(w-pad,h-pad);ctx.stroke();ctx.fillStyle='#8fa5ba';ctx.font='11px system-ui';for(let i=0;i<4;i++){const v=min+(max-min)*i/3,y=h-pad-(h-2*pad)*i/3;ctx.fillText(number(v,1),4,y+4);ctx.strokeStyle='#1d3249';ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(w-pad,y);ctx.stroke()}}
function line(ctx,data,x,y,color){if(!data.length)return;ctx.strokeStyle=color;ctx.lineWidth=2.2;ctx.beginPath();data.forEach((point,index)=>{const px=x(point),py=y(point);index?ctx.lineTo(px,py):ctx.moveTo(px,py)});ctx.stroke()}
function drawTemperature(){const canvas=$('#temperatureChart'),{ctx,width:w,height:h}=canvasSetup(canvas),pad=42,data=state.temperature;ctx.clearRect(0,0,w,h);if(data.length<2){ctx.fillStyle='#8fa5ba';ctx.fillText('Waiting for live telemetry…',pad,h/2);return}const values=data.flatMap(d=>[d.measured,d.predicted]),min=Math.floor(Math.min(...values)-2),max=Math.ceil(Math.max(...values)+2),t0=data[0].time,t1=data.at(-1).time||t0+1,x=d=>pad+(d.time-t0)/(t1-t0||1)*(w-2*pad),y=d=>h-pad-(d-min)/(max-min||1)*(h-2*pad);axes(ctx,w,h,pad,min,max);line(ctx,data,x,d=>y(d.measured),'#22d3ee');line(ctx,data,x,d=>y(d.predicted),'#fb923c');$('#temperatureSummary').textContent=`Latest simulated temperature ${number(data.at(-1).measured,1)} °C; PINN estimate ${number(data.at(-1).predicted,1)} °C.`}
function drawProfile(p){const canvas=$('#profileChart'),{ctx,width:w,height:h}=canvasSetup(canvas),pad=42,r=p.radius_star,c=p.concentration_mol_m3,s=p.sigma_tangential_MPa;ctx.clearRect(0,0,w,h);if(!r?.length)return;const cmin=Math.min(...c),cmax=Math.max(...c),sabs=Math.max(...s.map(Math.abs),1),x=v=>pad+v*(w-2*pad),yc=v=>h-pad-(v-cmin)/(cmax-cmin||1)*(h-2*pad),ys=v=>h/2-v/sabs*(h-2*pad)/2;axes(ctx,w,h,pad,cmin,cmax);line(ctx,r.map((value,i)=>({x:value,y:c[i]})),d=>x(d.x),d=>yc(d.y),'#60a5fa');line(ctx,r.map((value,i)=>({x:value,y:s[i]})),d=>x(d.x),d=>ys(d.y),'#fb923c')}
function updateRadialTable(p){const step=Math.max(1,Math.floor(p.radius_star.length/7));$('#radialTable').innerHTML=p.radius_star.filter((_,i)=>i%step===0||i===p.radius_star.length-1).map((radius,displayIndex)=>{const i=Math.min(displayIndex*step,p.radius_star.length-1);return `<tr><td>${number(radius,3)}</td><td>${number(p.concentration_mol_m3[i],1)}</td><td>${number(p.sigma_radial_MPa[i],2)}</td><td>${number(p.sigma_tangential_MPa[i],2)}</td></tr>`}).join('')}
function updateForecast(items){const cards=$('#forecastCards');cards.innerHTML=items.map(item=>`<article class="forecast-card"><span>${item.horizon_s}s forecast</span><strong>${number(item.hybrid_temperature_C,1)} °C</strong><small>PINN T: ${number(item.pinn_temperature_C,1)} °C · stress: ${number(item.peak_stress_MPa,1)} MPa</small></article>`).join('')}
async function compare(){try{const options=await api('/api/live/compare?horizon_s=300');const best=Math.min(...options.map(o=>o.temperature_C));$('#forecastCards').innerHTML=options.map(option=>`<article class="forecast-card ${option.temperature_C===best?'recommended':''}"><span>${option.action}</span><strong>${number(option.temperature_C,1)} °C</strong><small>${number(option.c_rate,1)}C · cooling ${number(option.cooling_factor,1)}×</small><small>${option.basis}</small></article>`).join('')}catch(error){toast(error.message)}}

$('#scenarioSelect').addEventListener('change',updateScenarioDescription);$('#currentAuto').addEventListener('change',updateControlOutputs);$('#currentRange').addEventListener('input',updateControlOutputs);$('#coolingRange').addEventListener('input',updateControlOutputs);$('#startButton').addEventListener('click',start);$('#pauseButton').addEventListener('click',togglePause);$('#resetButton').addEventListener('click',reset);$('#applyButton').addEventListener('click',applyControl);$('#compareButton').addEventListener('click',compare);window.addEventListener('resize',()=>{drawTemperature();if(state.last)drawProfile(state.last.prediction)});
initialize();
