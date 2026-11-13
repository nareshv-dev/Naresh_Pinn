const canvas = document.querySelector('#twinCanvas');
const fallback = document.querySelector('#threeFallback');
const motionButton = document.querySelector('#motionButton');

async function start3D(){
  try{
    const THREE = await import('https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js');
    const renderer = new THREE.WebGLRenderer({canvas,antialias:true,alpha:true});
    renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(34,1,.1,100);camera.position.set(0,1.1,9.4);
    const world = new THREE.Group();scene.add(world);
    scene.add(new THREE.HemisphereLight(0xbfe9ff,0x08111c,2.2));const key=new THREE.DirectionalLight(0xffffff,3);key.position.set(4,6,5);scene.add(key);const rim=new THREE.PointLight(0x22d3ee,10,12);rim.position.set(-4,1,3);scene.add(rim);

    const cellGroup=new THREE.Group();cellGroup.position.x=-1.7;world.add(cellGroup);
    const cellMaterial=new THREE.MeshPhysicalMaterial({color:0x22d3ee,transparent:true,opacity:.42,roughness:.25,metalness:.1,transmission:.2,side:THREE.DoubleSide});
    const cell=new THREE.Mesh(new THREE.CylinderGeometry(1.05,1.05,3.7,64,1,true,0,Math.PI*1.65),cellMaterial);cellGroup.add(cell);
    const core=new THREE.Mesh(new THREE.CylinderGeometry(.66,.66,3.55,40),new THREE.MeshStandardMaterial({color:0x1d4ed8,roughness:.55,transparent:true,opacity:.75}));cellGroup.add(core);
    const top=new THREE.Mesh(new THREE.CylinderGeometry(.42,.42,.18,40),new THREE.MeshStandardMaterial({color:0x94a3b8,metalness:.7,roughness:.3}));top.position.y=1.94;cellGroup.add(top);
    for(let i=-1;i<=1;i++){const ring=new THREE.Mesh(new THREE.TorusGeometry(.82,.025,12,48),new THREE.MeshBasicMaterial({color:0x60a5fa,transparent:true,opacity:.5}));ring.rotation.x=Math.PI/2;ring.position.y=i*1.05;cellGroup.add(ring)}

    const particleGroup=new THREE.Group();particleGroup.position.x=1.75;world.add(particleGroup);
    const shellColors=[0x312e81,0x1d4ed8,0x0891b2,0x22d3ee];const shells=[];
    shellColors.forEach((color,index)=>{const radius=.42+index*.22;const mesh=new THREE.Mesh(new THREE.SphereGeometry(radius,42,28,0,Math.PI*1.6),new THREE.MeshPhysicalMaterial({color,transparent:true,opacity:index===3?.33:.22,roughness:.25,side:THREE.DoubleSide}));particleGroup.add(mesh);shells.push(mesh)});
    const stressRing=new THREE.Mesh(new THREE.TorusGeometry(1.12,.035,12,64),new THREE.MeshBasicMaterial({color:0xfb923c}));stressRing.rotation.x=Math.PI/2;particleGroup.add(stressRing);
    const centre=new THREE.Mesh(new THREE.SphereGeometry(.12,24,16),new THREE.MeshBasicMaterial({color:0xe0f2fe}));particleGroup.add(centre);

    let running=true,noMotion=matchMedia('(prefers-reduced-motion: reduce)').matches,dragging=false,lastX=0,lastY=0;
    const motionQuery=matchMedia('(prefers-reduced-motion: reduce)');motionQuery.addEventListener('change',event=>{noMotion=event.matches});
    motionButton.addEventListener('click',()=>{running=!running;motionButton.setAttribute('aria-pressed',String(!running));motionButton.setAttribute('aria-label',running?'Pause 3D rotation':'Resume 3D rotation')});
    canvas.addEventListener('pointerdown',event=>{dragging=true;lastX=event.clientX;lastY=event.clientY;canvas.setPointerCapture(event.pointerId)});canvas.addEventListener('pointerup',()=>dragging=false);canvas.addEventListener('pointermove',event=>{if(!dragging)return;world.rotation.y+=(event.clientX-lastX)*.008;world.rotation.x=Math.max(-.45,Math.min(.45,world.rotation.x+(event.clientY-lastY)*.006));lastX=event.clientX;lastY=event.clientY});
    const tempColor=new THREE.Color(),low=new THREE.Color(0x2563eb),mid=new THREE.Color(0x22d3ee),hot=new THREE.Color(0xef4444);
    window.addEventListener('twin-update',event=>{const data=event.detail,p=data.prediction,temp=data.telemetry.surface_temperature_C,t=Math.max(0,Math.min(1,(temp-20)/40));tempColor.copy(t<.5?low:mid).lerp(t<.5?mid:hot,t<.5?t*2:(t-.5)*2);cellMaterial.color.copy(tempColor);core.material.color.copy(tempColor).multiplyScalar(.7);const values=p.concentration_mol_m3,min=Math.min(...values),max=Math.max(...values);shells.forEach((shell,index)=>{const value=values[Math.round(index*(values.length-1)/(shells.length-1))],ratio=(value-min)/(max-min||1);shell.material.color.setHSL(.68-ratio*.2,.78,.43+.12*ratio)});const radius=.45+Math.max(.05,p.peak_stress_radius_star)*.72;stressRing.scale.setScalar(radius/1.12);stressRing.material.color.set(Math.abs(p.peak_tensile_stress_MPa)>60?0xef4444:0xfb923c)});
    function resize(){const rect=canvas.getBoundingClientRect();renderer.setSize(rect.width,rect.height,false);camera.aspect=rect.width/rect.height;camera.updateProjectionMatrix()}new ResizeObserver(resize).observe(canvas);resize();
    const clock=new THREE.Clock();function animate(){requestAnimationFrame(animate);const dt=Math.min(clock.getDelta(),.05);if(running&&!noMotion&&!dragging){world.rotation.y+=dt*.13;particleGroup.rotation.y-=dt*.16}renderer.render(scene,camera)}animate();
  }catch(error){canvas.hidden=true;fallback.hidden=false;fallback.textContent=`3D view unavailable (${error.message}). Live numerical values and accessible radial data remain available.`;console.error('Three.js initialization failed',error)}
}
start3D();
