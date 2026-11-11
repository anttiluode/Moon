(function(){
const NET=JSON.parse(document.getElementById('net').textContent);
const $=id=>document.getElementById(id);
const canvas=$('gl'), readEl=$('read');
const gl=canvas.getContext('webgl2',{antialias:false,alpha:false,preserveDrawingBuffer:true});
function fail(msg){const d=document.createElement('div');d.className='fail';d.textContent=msg;canvas.replaceWith(d);$('hint').hidden=true;readEl.hidden=true;}
if(!gl){fail('This page needs WebGL2, which this browser or device does not provide.');return;}
if(!gl.getExtension('EXT_color_buffer_float')&&!gl.getExtension('EXT_color_buffer_half_float')){fail('This GPU cannot render to float textures, which the bake step needs.');return;}

/* ---------- the compiler: weights -> GLSL ---------- */
const TERMS=[
 {id:'ss',out:0,src:['ss'],name:'Single scattering',note:'Exact: <code>exp(-tau_sun) x HG(g)</code>. Ten of the probes along the sun axis are the whole circuit.'},
 {id:'iso1',out:0,src:['iso1'],name:'Forward-peak remainder',note:'Light scattered into the forward peak still travels with the beam. Similarity scaling: <code>exp(-(1-g) tau) - exp(-tau)</code>.'},
 {id:'S2',out:0,src:['S2'],name:'Second scatter, by quadrature',note:'154 probes on 14 Lebedev directions: sunlit source at each probe, attenuated back to the point, summed.'},
 {id:'hi',out:0,src:['hi'],name:'Higher orders',note:'The same stencil with softer attenuation, multiplied by the build-up <code>(1-P)/P</code>, where P is the chance of escaping from here.'},
 {id:'gl',out:0,src:['gl1','gl2'],name:'Forward glow',note:'Second and third order light still remembers its direction: <code>HG(g^2)</code>, <code>HG(g^3)</code> with Poisson weights. This is the bright rim when the sun is behind.'},
 {id:'sky',out:1,src:null,name:'Sky light',note:'Directly visible sky plus a diffusion estimate: each exit direction weighted by <code>1/(tau* cos + 2/3)</code> and by how much sky that exit sees.'},
 {id:'gnd',out:2,src:null,name:'Ground light',note:'The same circuit read the other way: what is not sky is ground.'}
];
const f9=v=>{let s=(+v).toPrecision(9);if(!/[.e]/.test(s))s+='.';if(/e/.test(s)&&!/\./.test(s))s=s.replace('e','.e');return s;};
function compile(off){
  const Ls=NET.layers,nL=Ls.length,inputs=NET.inputs;
  const key=(l,k)=>l*100000+k;
  const unit=(l,k)=>Ls[l-1].units[k];
  const al=new Map();
  function alias(l,k){const K=key(l,k);if(al.has(K))return al.get(K);let r=[l,k];
    if(l>0){const u=unit(l,k),ks=Object.keys(u.w);if(u.act==='id'&&!('g' in u)&&u.b===0&&ks.length===1&&u.w[ks[0]]===1)r=alias(l-1,+ks[0]);}
    al.set(K,r);return r;}
  const vw=new Map();
  function view(l,k){const K=key(l,k);if(vw.has(K))return vw.get(K);let r=false;
    if(l===0)r=inputs[k]==='cs';else{const u=unit(l,k);for(const i in u.w)if(view(l-1,+i)){r=true;break;}if(!r&&u.g)for(const i in u.g)if(view(l-1,+i)){r=true;break;}}
    vw.set(K,r);return r;}
  const bake=[],march=[],bDone=new Map(),mDone=new Map(),front=[],fIdx=new Map();
  let used=0;
  function lin(terms,b,ref){ // -> list of chunks
    const parts=[];if(b!==0)parts.push(f9(b));
    for(const [i,v] of terms){parts.push((v===1?'':v===-1?'-':f9(v)+'*')+ref(i));used++;}
    if(!parts.length)parts.push('0.');
    const ch=[];for(let i=0;i<parts.length;i+=12)ch.push(parts.slice(i,i+12).join('+').replace(/\+-/g,'-'));return ch;}
  function emit(buf,name,u,l,ref){
    const A={id:z=>z,relu:z=>`max(${z},0.)`,exp:z=>`exp(${z})`,log:z=>`log(${z})`,recip:z=>`1./(${z})`};
    const r=i=>ref(...alias(l-1,+i));
    const ch=lin(Object.entries(u.w),u.b,r);
    buf.push(`float ${name}=${ch[0]};`);for(let i=1;i<ch.length;i++)buf.push(`${name}+=${ch[i]};`);
    if(u.act!=='id')buf.push(`${name}=${A[u.act](name)};`);
    if('g' in u){const gc=lin(Object.entries(u.g),u.c,r);if(gc.length===1)buf.push(`${name}*=(${gc[0]});`);else{buf.push(`float g${name}=${gc[0]};`);for(let i=1;i<gc.length;i++)buf.push(`g${name}+=${gc[i]};`);buf.push(`${name}*=g${name};`);}}
  }
  function refB(l,k){if(l===0)return `x[${k}]`;const K=key(l,k);if(bDone.has(K))return bDone.get(K);
    const name=`b${l}_${k}`;bDone.set(K,name);const u=unit(l,k);const tmp=[];emit(tmp,name,u,l,refB);for(const s of tmp)bake.push(s);return name;}
  function refM(l,k){if(l===0&&inputs[k]==='cs')return 'cs';
    if(!view(l,k)){const K=key(l,k);if(!fIdx.has(K)){fIdx.set(K,front.length);front.push(refB(l,k));}return `f${fIdx.get(K)}`;}
    const K=key(l,k);if(mDone.has(K))return mDone.get(K);
    const name=`m${l}_${k}`;mDone.set(K,name);const u=unit(l,k);const tmp=[];emit(tmp,name,u,l,refM);for(const s of tmp)march.push(s);return name;}
  const outs=[];
  Ls[nL-1].units.forEach((u,o)=>{
    const names=Ls[nL-2].units.map(q=>q.n);let scale=1;const dead=new Set();
    for(const t of TERMS)if(off.has(t.id)&&t.out===o){if(t.src)t.src.forEach(s=>dead.add(s));else scale=0;}
    const st=[],vt=[];
    for(const i in u.w){if(scale===0||dead.has(names[+i]))continue;const a=alias(nL-1,+i);(view(a[0],a[1])?vt:st).push([a,u.w[i]]);}
    const sref=p=>refB(p[0],p[1]),vref=p=>refM(p[0],p[1]);
    const sc=lin(st,scale?u.b:0,sref);const bn=`bo${o}`;bake.push(`float ${bn}=${sc[0]};`);for(let i=1;i<sc.length;i++)bake.push(`${bn}+=${sc[i]};`);
    const fi=front.length;front.push(bn);
    const vc=lin(vt,0,vref);outs.push(`f${fi}`+(vt.length?'+'+vc.join('+'):''));
  });
  if(front.length>8)throw new Error('more than 8 baked channels');
  while(front.length<8)front.push('0.');
  const marchFn=`vec3 net(vec4 t0,vec4 t1,float cs){\n`+[0,1,2,3].map(i=>`float f${i}=t0[${i}];`).join('')+[4,5,6,7].map(i=>`float f${i}=t1[${i-4}];`).join('')+'\n'+march.join('\n')+`\nreturn vec3(${outs.join(',\n ')});\n}`;
  return {bake:bake.join('\n'),front,marchFn,stats:{bakeUnits:bDone.size,marchUnits:mDone.size,weights:used,bakeLines:bake.length,marchLines:march.length,channels:fIdx.size+3}};
}

/* ---------- GLSL scaffolding ---------- */
const E=NET.edges,NC=E.length-1,MID=[],DE=[];for(let j=0;j<NC;j++){MID.push((E[j]+E[j+1])/2);DE.push(E[j+1]-E[j]);}
const PHI=1.6180339887,AXr=[[0,1,PHI],[0,-1,PHI],[1,PHI,0],[-1,PHI,0],[PHI,0,1],[PHI,0,-1]].map(a=>{const n=Math.hypot(...a);return a.map(v=>v/n);});
const VS=`#version 300 es
void main(){vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));gl_Position=vec4(p*2.-1.,0.,1.);}`;
const SCENE=`
const vec3 AX[6]=vec3[6](${AXr.map(a=>`vec3(${a.map(f9).join(',')})`).join(',')});
float sstep(float a,float b,float x){float t=clamp((x-a)/(b-a),0.,1.);return t*t*(3.-2.*t);}
float hull(vec3 p){return 1.-sstep(0.86,1.0,length(p));}
float dens(vec3 p){float r2=dot(p,p);if(r2>=1.)return 0.;float q=0.;for(int i=0;i<6;i++)q+=cos(11.0*dot(AX[i],p));
  return 12.0*(1.-sstep(0.86,1.0,sqrt(r2)))*(0.12+0.88*sstep(-0.3,1.8,q));}
const float MID[${NC}]=float[${NC}](${MID.map(f9).join(',')});
const float DE[${NC}]=float[${NC}](${DE.map(f9).join(',')});
const vec3 DIRS[14]=vec3[14](${NET.dirs.map(a=>`vec3(${a.map(f9).join(',')})`).join(',')});`;
const FS_A=`#version 300 es
precision highp float;uniform vec3 uSun;uniform float uZ,uRes;out vec4 o;${SCENE}
void main(){vec3 p=(vec3(gl_FragCoord.xy,uZ+.5)/uRes)*2.-1.;float t=0.;for(int j=0;j<${NC};j++)t+=DE[j]*dens(p+MID[j]*uSun);o=vec4(t);}`;
function fsBake(c){
  let fill='';
  for(const g of NET.groups){
    const f=q=>g.field==='dens'?`dens(${q})`:g.field==='hull'?`hull(${q})`:`texture(uBeam,(${q})*.5+.5).r`;
    if(g.frame==='world')fill+=`for(int i=0;i<14;i++)for(int j=0;j<${NC};j++){vec3 q=p+MID[j]*DIRS[i];x[${g.offset}+i*${NC}+j]=${f('q')};}\n`;
    else fill+=`for(int j=0;j<${NC};j++){vec3 q=p${g.frame==='sun'?'+':'-'}MID[j]*uSun;x[${g.offset}+j]=${f('q')};}\n`;
  }
  const S=NET.inputs.length-8;
  return `#version 300 es
precision highp float;precision highp sampler3D;uniform vec3 uSun;uniform float uZ,uRes;uniform sampler3D uBeam;
layout(location=0) out vec4 o0;layout(location=1) out vec4 o1;${SCENE}
void main(){vec3 p=(vec3(gl_FragCoord.xy,uZ+.5)/uRes)*2.-1.;
if(dot(p,p)>1.25){o0=vec4(0.);o1=vec4(0.);return;}
float x[${NET.inputs.length}];
${fill}x[${S}]=uSun.x;x[${S+1}]=uSun.y;x[${S+2}]=uSun.z;x[${S+3}]=0.;x[${S+4}]=dot(p,uSun);x[${S+5}]=p.x;x[${S+6}]=p.y;x[${S+7}]=p.z;
${c.bake}
o0=vec4(${c.front.slice(0,4).join(',')});o1=vec4(${c.front.slice(4,8).join(',')});}`;
}
function fsMarch(c){return `#version 300 es
precision highp float;precision highp sampler3D;
uniform sampler3D uT0,uT1;uniform vec3 uCam,uR,uU,uF,uSun,uSunCol,uSky,uGnd,uZen,uHor;uniform vec2 uRes;uniform float uTan;uniform int uDebug;
out vec4 o;${SCENE}
${c.marchFn}
vec3 bg(vec3 d){
  if(d.y<0.){float h=exp(d.y*30.);return mix(uGnd,uHor*.5,h);}
  vec3 c=mix(uHor,uZen,pow(d.y,.35));float m=max(dot(d,uSun),0.);
  return c+uSunCol*(.06*pow(m,24.)+.5*pow(m,400.)+8.*smoothstep(.99993,.99998,m));}
void main(){
  vec2 uv=(gl_FragCoord.xy/uRes)*2.-1.;vec3 d=normalize(uF+uv.x*uTan*uRes.x/uRes.y*uR+uv.y*uTan*uU);
  float b=dot(uCam,d),disc=b*b-(dot(uCam,uCam)-1.);vec3 L=vec3(0.);float Tv=1.;
  if(disc>0.){float sq=sqrt(disc),t0=-b-sq;const int N=64;float dt=2.*sq/float(N);
    float jit=uDebug>0?.5:fract(sin(dot(gl_FragCoord.xy,vec2(12.9898,78.233)))*43758.5453);float cs=dot(d,uSun);
    for(int i=0;i<N;i++){vec3 p=uCam+(t0+(float(i)+jit)*dt)*d;float rho=dens(p);if(rho<=0.)continue;
      float a=1.-exp(-rho*dt);vec3 uvw=p*.5+.5;vec3 J=net(texture(uT0,uvw),texture(uT1,uvw),cs);
      L+=Tv*a*J;Tv*=1.-a;if(Tv<.003)break;}}
  if(uDebug>0){float v=uDebug==1?L.x*3.:uDebug==2?L.y:uDebug==3?L.z:Tv;v=clamp(v,0.,.9999)*255.;o=vec4(floor(v)/255.,fract(v),0.,1.);return;}
  vec3 c=vec3(.95,.975,1.)*(uSunCol*L.x+uSky*L.y+uGnd*L.z)+Tv*bg(d);
  c=1.-exp(-c*1.15);o=vec4(pow(c,vec3(1./2.2)),1.);}`;}

function sh(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;}
const vsObj=sh(gl.VERTEX_SHADER,VS);
function prog(fs){const p=gl.createProgram();gl.attachShader(p,vsObj);gl.attachShader(p,sh(gl.FRAGMENT_SHADER,fs));gl.linkProgram(p);
  if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));const u={};const n=gl.getProgramParameter(p,gl.ACTIVE_UNIFORMS);
  for(let i=0;i<n;i++){const nm=gl.getActiveUniform(p,i).name;u[nm]=gl.getUniformLocation(p,nm);}return {p,u};}
const vao=gl.createVertexArray();gl.bindVertexArray(vao);
function tex3(res){const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_3D,t);gl.texStorage3D(gl.TEXTURE_3D,1,gl.RGBA16F,res,res,res);
  for(const [k,v] of [[gl.TEXTURE_MIN_FILTER,gl.LINEAR],[gl.TEXTURE_MAG_FILTER,gl.LINEAR],[gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE],[gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE],[gl.TEXTURE_WRAP_R,gl.CLAMP_TO_EDGE]])gl.texParameteri(gl.TEXTURE_3D,k,v);return t;}
const sets={};function getSet(res){if(!sets[res])sets[res]={res,A:tex3(res),B0:tex3(res),B1:tex3(res)};return sets[res];}
const fbo=gl.createFramebuffer();
let pA,pB,pM,cur=null,compiled=null;
const norm=v=>{const n=Math.hypot(...v);return v.map(x=>x/n);};
const st={sunAz:2.9,sunEl:0.14,camAz:0.0,camEl:0.10,mode:'sun',playing:false,off:new Set(),debug:0,fixedTan:0,lastBake:0};
const sunDir=()=>[Math.cos(st.sunEl)*Math.sin(st.sunAz),Math.sin(st.sunEl),Math.cos(st.sunEl)*Math.cos(st.sunAz)];
const camDir=()=>[Math.cos(st.camEl)*Math.sin(st.camAz),Math.sin(st.camEl),Math.cos(st.camEl)*Math.cos(st.camAz)];
let sunOverride=null,camOverride=null;
function bake(res){
  const S=getSet(res),s=sunOverride||sunDir();gl.bindFramebuffer(gl.FRAMEBUFFER,fbo);gl.viewport(0,0,res,res);
  gl.useProgram(pA.p);gl.uniform3fv(pA.u.uSun,s);gl.uniform1f(pA.u.uRes,res);gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
  gl.framebufferTextureLayer(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT1,null,0,0);
  for(let z=0;z<res;z++){gl.framebufferTextureLayer(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,S.A,0,z);gl.uniform1f(pA.u.uZ,z);gl.drawArrays(gl.TRIANGLES,0,3);}
  gl.useProgram(pB.p);gl.uniform3fv(pB.u.uSun,s);gl.uniform1f(pB.u.uRes,res);gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_3D,S.A);gl.uniform1i(pB.u.uBeam,0);
  gl.drawBuffers([gl.COLOR_ATTACHMENT0,gl.COLOR_ATTACHMENT1]);
  for(let z=0;z<res;z++){gl.framebufferTextureLayer(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,S.B0,0,z);gl.framebufferTextureLayer(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT1,S.B1,0,z);gl.uniform1f(pB.u.uZ,z);gl.drawArrays(gl.TRIANGLES,0,3);}
  gl.bindFramebuffer(gl.FRAMEBUFFER,null);cur=S;
}
const mix=(a,b,t)=>a.map((v,i)=>v+(b[i]-v)*t),ss=(a,b,x)=>{const t=Math.min(1,Math.max(0,(x-a)/(b-a)));return t*t*(3-2*t);};
function lights(s){const el=s[1],vis=ss(-0.10,0.03,el),day=ss(0,0.4,el),tw=ss(-0.25,0.05,el);
  const sun=mix([1,0.36,0.12],[1,0.95,0.88],day).map(v=>v*4.2*vis);
  const zen=mix([0.005,0.007,0.018],mix([0.035,0.045,0.14],[0.10,0.22,0.58],day),tw);
  const hor=mix([0.012,0.015,0.035],mix([0.62,0.27,0.11],[0.45,0.58,0.82],day),tw);
  const sky=mix(zen,hor,0.45);const alb=[0.06,0.055,0.05];
  const gnd=alb.map((a,i)=>a*(sun[i]*Math.max(el,0)+Math.PI*sky[i])/Math.PI);
  return {sun,zen,hor,sky,gnd};}
function draw(scale){
  const dpr=Math.min(window.devicePixelRatio||1,1.5)*scale;
  let w,h;if(st.debug){w=h=96;}else{w=Math.max(2,Math.round(canvas.clientWidth*dpr));h=Math.max(2,Math.round(canvas.clientHeight*dpr));}
  if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
  const s=sunOverride||sunDir(),c=camOverride||camDir(),f=c.map(v=>-v),r=norm([-f[2],0,f[0]]),u=[r[1]*f[2]-r[2]*f[1],r[2]*f[0]-r[0]*f[2],r[0]*f[1]-r[1]*f[0]];
  const Lt=lights(s);gl.viewport(0,0,w,h);gl.useProgram(pM.p);const U=pM.u;
  gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_3D,cur.B0);gl.uniform1i(U.uT0,0);gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_3D,cur.B1);gl.uniform1i(U.uT1,1);
  gl.uniform3fv(U.uCam,c.map(v=>v*3.4));gl.uniform3fv(U.uR,r);gl.uniform3fv(U.uU,u);gl.uniform3fv(U.uF,f);gl.uniform3fv(U.uSun,s);
  gl.uniform3fv(U.uSunCol,Lt.sun);gl.uniform3fv(U.uSky,Lt.sky);gl.uniform3fv(U.uGnd,Lt.gnd);gl.uniform3fv(U.uZen,Lt.zen);gl.uniform3fv(U.uHor,Lt.hor);
  gl.uniform2f(U.uRes,w,h);gl.uniform1f(U.uTan,st.debug?0.36:(w<h*1.2?0.46:0.40));gl.uniform1i(U.uDebug,st.debug);
  gl.drawArrays(gl.TRIANGLES,0,3);
}
const LO=40,HI=72;let dirtySun=true,dirtyView=true,idleT=0,fine=false,frames=0,fpsT=performance.now(),fps=0,raf=0;
function tick(now){raf=0;
  if(st.playing){st.sunAz+=0.012;st.sunEl=0.42*Math.sin(st.sunAz*0.5+1.2)+0.2;dirtySun=true;}
  const moving=dirtySun||dirtyView;
  if(dirtySun){bake(LO);fine=false;}
  if(moving){draw(st.playing||dragging?0.7:1);idleT=now;frames++;}
  else if(!fine&&now-idleT>140){bake(HI);draw(1);fine=true;}
  dirtySun=dirtyView=false;
  if(now-fpsT>500){fps=frames*1000/(now-fpsT);frames=0;fpsT=now;}
  if(compiled)readEl.textContent=(st.playing||dragging?fps.toFixed(0)+' fps · ':'')+'sun '+(st.sunEl*57.3).toFixed(0)+'° · '+compiled.stats.marchLines+' lines per step';
  if(st.playing||dragging||!fine)req();
}
function req(){if(!raf)raf=requestAnimationFrame(tick);}
function rebuild(){
  const t0=performance.now();compiled=compile(st.off);
  if(pB)gl.deleteProgram(pB.p);if(pM)gl.deleteProgram(pM.p);
  pB=prog(fsBake(compiled));pM=prog(fsMarch(compiled));
  const ms=performance.now()-t0,s=compiled.stats;
  $('stats').innerHTML=`Compiled in <b>${ms.toFixed(0)} ms</b>. Kept <b>${s.bakeUnits+s.marchUnits}</b> of 976 units and <b>${s.weights}</b> of 3,148 weights.<br>Baked per voxel when the sun moves: <b>${s.bakeUnits}</b> units into <b>${s.channels}</b> texture channels.<br>Left in the ray march: <b>${s.marchUnits}</b> units, <b>${s.marchLines}</b> lines of arithmetic, 2 texture reads per step.`;
  $('src').textContent=compiled.marchFn;dirtySun=true;req();
}
/* ---------- controls ---------- */
const circ=$('circ');
for(const t of TERMS){const l=document.createElement('label');l.innerHTML=`<input type="checkbox" id="t_${t.id}" checked><b>${t.name}</b><small>${t.note}</small>`;circ.appendChild(l);
  l.querySelector('input').addEventListener('change',e=>{e.target.checked?st.off.delete(t.id):st.off.add(t.id);try{rebuild();}catch(err){readEl.textContent='compile failed';console.error(err);}});}
let dragging=false,px=0,py=0;
canvas.addEventListener('pointerdown',e=>{dragging=true;px=e.clientX;py=e.clientY;canvas.setPointerCapture(e.pointerId);$('hint').hidden=true;setPlay(false);req();});
canvas.addEventListener('pointermove',e=>{if(!dragging)return;const dx=(e.clientX-px)/canvas.clientWidth,dy=(e.clientY-py)/canvas.clientHeight;px=e.clientX;py=e.clientY;
  if(st.mode==='sun'){st.sunAz+=dx*3.2;st.sunEl=Math.max(-0.45,Math.min(1.5,st.sunEl-dy*2.2));dirtySun=true;}
  else{st.camAz-=dx*3.2;st.camEl=Math.max(-0.6,Math.min(1.3,st.camEl+dy*2.2));dirtyView=true;}req();});
const up=()=>{if(dragging){dragging=false;dirtyView=true;req();}};canvas.addEventListener('pointerup',up);canvas.addEventListener('pointercancel',up);
canvas.tabIndex=0;canvas.addEventListener('keydown',e=>{const k={ArrowLeft:[-.08,0],ArrowRight:[.08,0],ArrowUp:[0,.06],ArrowDown:[0,-.06]}[e.key];if(!k)return;e.preventDefault();
  if(st.mode==='sun'){st.sunAz+=k[0];st.sunEl=Math.max(-0.45,Math.min(1.5,st.sunEl+k[1]));dirtySun=true;}else{st.camAz+=k[0];st.camEl=Math.max(-0.6,Math.min(1.3,st.camEl+k[1]));dirtyView=true;}req();});
function setPlay(v){st.playing=v;$('play').setAttribute('aria-pressed',v);$('play').textContent=v?'Stop the sweep':'Sweep the sun';if(v)req();}
$('play').addEventListener('click',()=>setPlay(!st.playing));
function setMode(m){st.mode=m;$('mSun').setAttribute('aria-pressed',m==='sun');$('mCam').setAttribute('aria-pressed',m==='cam');$('hint').textContent=m==='sun'?'Drag the sky to move the sun':'Drag to orbit the camera';}
$('mSun').addEventListener('click',()=>setMode('sun'));$('mCam').addEventListener('click',()=>setMode('cam'));
const PRE={rim:[2.9,0.14],noon:[0.7,1.05],sunset:[1.35,0.03],dusk:[2.2,-0.16]};
document.querySelectorAll('[data-pre]').forEach(b=>b.addEventListener('click',()=>{setPlay(false);[st.sunAz,st.sunEl]=PRE[b.dataset.pre];dirtySun=true;req();}));
new ResizeObserver(()=>{dirtyView=true;req();}).observe(canvas);
canvas.addEventListener('webglcontextlost',e=>e.preventDefault());
try{pA=prog(FS_A);rebuild();}catch(err){console.error(err);fail('The compiled shader did not build on this GPU: '+String(err.message||err).slice(0,200));return;}
/* test hook: render one transfer channel for a given camera and sun, 96x96, 16-bit packed */
window.__moon={channel(cam,sun,k,res){camOverride=norm(cam);sunOverride=norm(sun);st.debug=k;bake(res||HI);draw(1);const b=new Uint8Array(96*96*4);gl.readPixels(0,0,96,96,gl.RGBA,gl.UNSIGNED_BYTE,b);st.debug=0;camOverride=sunOverride=null;dirtySun=true;req();return Array.from(b);},stats:()=>compiled.stats};
})();
