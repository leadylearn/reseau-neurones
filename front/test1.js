/* ---------- utils ---------- */
const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
const sigmoid=z=>1/(1+Math.exp(-z));
const relu=z=>Math.max(0,z);
const reluPrime=z=>(z>0?1:0);
const tanh=z=>Math.tanh(z);
const tanhPrime=z=>1-Math.tanh(z)**2;

function randn(){
  let u=0,v=0; while(u===0)u=Math.random(); while(v===0)v=Math.random();
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
}
function bce(yhat,y){
  const eps=1e-9;
  return -(y*Math.log(yhat+eps) + (1-y)*Math.log(1-yhat+eps));
}

/* ---------- UI refs ---------- */
const svg=document.getElementById("svg");
const tooltip=document.getElementById("tooltip");
const logEl=document.getElementById("log");

const statusText=document.getElementById("statusText");
const statusDot=document.getElementById("statusDot");

const epochEl=document.getElementById("epoch");
const lrEl=document.getElementById("lr");
const actEl=document.getElementById("act");
const regEl=document.getElementById("reg");

const yhatOut=document.getElementById("yhat");
const lossOut=document.getElementById("loss");
const labelOut=document.getElementById("labelY");
const epochView=document.getElementById("epochView");

const layerCountEl=document.getElementById("layerCount");
const btnAdd=document.getElementById("btnAdd");
const btnRem=document.getElementById("btnRem");
const showKEl=document.getElementById("showK");
const newSizeEl=document.getElementById("newSize");
const inKEl=document.getElementById("inK");

function log(msg){
  const d=document.createElement("div");
  d.textContent=msg;
  logEl.appendChild(d);
  logEl.scrollTop=logEl.scrollHeight;
}
function setStatus(text,playing=false){
  statusText.textContent=text;
  statusDot.classList.toggle("playing",playing);
}

/* ============================================================
   MODEL (real math, fully-connected)
   ============================================================ */
const state={
  inputSize:2,
  inputVisualSize:64,
  hiddenLayers:[16,16,16],
  outputSize:1,
  act:"relu",
  showK:6,
  inputK:3,
  reg:"none"
};
const model={W:[],b:[],cache:{},grads:{}};

function layerSizes(){ return [state.inputSize, ...state.hiddenLayers, state.outputSize]; }

function initParams(){
  const sizes=layerSizes();
  model.W=[]; model.b=[];
  for(let l=0;l<sizes.length-1;l++){
    const inN=sizes[l], outN=sizes[l+1];
    const scale=Math.sqrt(2/(inN+outN));
    model.W.push(Array.from({length:outN},()=>Array.from({length:inN},()=>randn()*scale)));
    model.b.push(Array.from({length:outN},()=>0));
  }
  model.cache={}; model.grads={};
}
function actFn(z){ return (state.act==="tanh")?tanh(z):relu(z); }
function actPrime(z){ return (state.act==="tanh")?tanhPrime(z):reluPrime(z); }

function forward(x,y){
  const sizes=layerSizes();
  const a=[], z=[];
  a[0]=x.slice();
  for(let l=0;l<sizes.length-1;l++){
    const outN=sizes[l+1], inN=sizes[l];
    const zl=Array.from({length:outN},(_,i)=>{
      let s=model.b[l][i];
      for(let j=0;j<inN;j++) s+=model.W[l][i][j]*a[l][j];
      return s;
    });
    z[l+1]=zl;
    const isLast=(l===sizes.length-2);
    a[l+1]=isLast?zl.map(sigmoid):zl.map(actFn);
  }
  const yhat=a[a.length-1][0];
  const C=bce(yhat,y);
  model.cache={x,y,sizes,a,z,yhat,C};
  return {yhat,C};
}

function backward(){
  const {y,sizes,a,z,yhat}=model.cache;
  const delta=[];
  delta[sizes.length-1]=[yhat-y];

  for(let l=sizes.length-2;l>=1;l--){
    const n=sizes[l], nextN=sizes[l+1];
    delta[l]=Array.from({length:n},(_,i)=>{
      let s=0;
      for(let k=0;k<nextN;k++) s+=model.W[l][k][i]*delta[l+1][k];
      return s*actPrime(z[l][i]);
    });
  }

  const dW=[], db=[];
  for(let l=0;l<sizes.length-1;l++){
    const outN=sizes[l+1], inN=sizes[l];
    dW[l]=Array.from({length:outN},(_,i)=>Array.from({length:inN},(_,j)=>delta[l+1][i]*a[l][j]));
    db[l]=Array.from({length:outN},(_,i)=>delta[l+1][i]);
    if(state.reg==="l2"){
      const lambda=1e-3;
      for(let i=0;i<outN;i++) for(let j=0;j<inN;j++) dW[l][i][j]+=lambda*model.W[l][i][j];
    }
  }
  model.grads={delta,dW,db};
  return model.grads;
}

function applyUpdate(eta){
  const g=model.grads;
  if(!g||!g.dW) return;
  for(let l=0;l<model.W.length;l++){
    for(let i=0;i<model.W[l].length;i++){
      for(let j=0;j<model.W[l][i].length;j++){
        model.W[l][i][j]-=eta*g.dW[l][i][j];
      }
      model.b[l][i]-=eta*g.db[l][i];
    }
  }
}

/* ============================================================
   SVG VIS
   ============================================================ */
function E(tag,attrs={}){
  const el=document.createElementNS("http://www.w3.org/2000/svg",tag);
  for(const [k,v] of Object.entries(attrs)) el.setAttribute(k,v);
  return el;
}
const viz={nodes:new Map(),edges:new Map(),hoverEnabled:false};

function clearSvg(){
  while(svg.firstChild) svg.removeChild(svg.firstChild);
  viz.nodes.clear(); viz.edges.clear();
  tooltip.style.display="none";
}
function condensedIndices(n,K){
  if(n<=2*K) return Array.from({length:n},(_,i)=>i);
  const head=Array.from({length:K},(_,i)=>i);
  const tail=Array.from({length:K},(_,i)=>n-K+i);
  return [...head,null,...tail];
}
function addNode(id,x,y,label,layerIndex,neuronIndex,small=false){
  const g=E("g");
  const s=small?30:40;
  const rect=E("rect",{x:x-s/2,y:y-s/2,width:s,height:s,rx:12,ry:12,fill:"white",stroke:"#111827","stroke-width":"2"});
  const t=E("text",{x,y:y+6,"text-anchor":"middle","font-size":small?"12":"13","font-weight":"900",fill:"#111827"});
  t.textContent=label;
  g.appendChild(rect); g.appendChild(t);
  svg.appendChild(g);
  viz.nodes.set(id,{x,y,rect,text:t,group:g,layerIndex,neuronIndex,small});
}
function addEllipsis(x,y){
  const t=E("text",{x,y,"text-anchor":"middle","font-size":"22",fill:"rgba(107,114,128,1)"});
  t.textContent="…";
  svg.appendChild(t);
}
function addEdge(id,fromId,toId,l,outI,inJ){
  const a=viz.nodes.get(fromId), b=viz.nodes.get(toId);
  const padA=a.small?18:22, padB=b.small?18:22;
  const line=E("line",{x1:a.x+padA,y1:a.y,x2:b.x-padB,y2:b.y,stroke:"rgba(17,24,39,.16)","stroke-width":"2","stroke-linecap":"round"});
  const mx=(a.x+b.x)/2, my=(a.y+b.y)/2;

  const labelW=E("text",{x:mx,y:my-8,"text-anchor":"middle","font-size":"12",fill:"#111827"});
  const labelDW=E("text",{x:mx,y:my+14,"text-anchor":"middle","font-size":"12",fill:"#6b7280"});
  labelW.textContent=""; labelDW.textContent="";

  line.addEventListener("mousemove",(e)=>{
    if(!viz.hoverEnabled) return;
    const ed=viz.edges.get(id);
    tooltip.style.display="block";
    tooltip.textContent=`${ed.labelW.textContent}\n${ed.labelDW.textContent}`;
    let x=e.clientX+14,y=e.clientY+14;
    const pad=14,w=tooltip.offsetWidth,h=tooltip.offsetHeight;
    if(x+w+pad>window.innerWidth) x=e.clientX-w-14;
    if(y+h+pad>window.innerHeight) y=e.clientY-h-14;
    tooltip.style.left=x+"px"; tooltip.style.top=y+"px";
  });
  line.addEventListener("mouseleave",()=>tooltip.style.display="none");

  svg.insertBefore(line,svg.firstChild);
  svg.appendChild(labelW);
  svg.appendChild(labelDW);

  viz.edges.set(id,{id,fromId,toId,line,labelW,labelDW,mx,my,l,outI,inJ});
}

function colorForWeight(w){ return w>=0 ? "rgba(59,130,246,.50)" : "rgba(245,158,11,.50)"; }
function strokeWidthForWeight(w){ return 1.6 + Math.min(5.0, Math.abs(w)*3.2); }

function hideAllEdgeText(){
  for(const e of viz.edges.values()){
    e.labelW.textContent="";
    e.labelDW.textContent="";
    e.labelDW.setAttribute("fill","#6b7280");
  }
}

function updateEdgeAppearanceNoLabels(){
  hideAllEdgeText();
  for(const e of viz.edges.values()){
    const w=model.W[e.l]?.[e.outI]?.[e.inJ];
    if(typeof w==="number"){
      e.line.setAttribute("stroke",colorForWeight(w));
      e.line.setAttribute("stroke-width",strokeWidthForWeight(w));
      e.line.setAttribute("opacity","1");
    }
  }
}

function resetHighlights(){
  for(const n of viz.nodes.values()){
    n.rect.setAttribute("stroke","#111827");
    n.rect.setAttribute("stroke-width","2");
  }
  for(const e of viz.edges.values()){
    e.line.setAttribute("opacity","1");
  }
}

function glowNode(id){
  const n=viz.nodes.get(id);
  if(!n) return;
  n.rect.setAttribute("stroke","rgba(16,185,129,.95)");
  n.rect.setAttribute("stroke-width","4");
}
function dimAllEdges(){
  for(const e of viz.edges.values()) e.line.setAttribute("opacity","0.15");
}
function glowEdge(id){
  const e=viz.edges.get(id);
  if(!e) return;
  e.line.setAttribute("opacity","1");
}

function setOnlyEdgeLabels(edgeId, w, dw){
  hideAllEdgeText();
  const e=viz.edges.get(edgeId);
  if(!e) return;
  e.labelW.textContent=`w = ${w.toFixed(5)}`;
  if(dw===null || typeof dw==="undefined"){
    e.labelDW.textContent="";
  }else{
    e.labelDW.textContent=`dw = ${dw.toFixed(5)}`;
    e.labelDW.setAttribute("fill","rgba(16,185,129,.95)");
  }
}

function updateVisualNodeValues(){
  const c=model.cache;
  for(const id of viz.nodes.keys()){
    if(id.startsWith("vin_")||id.startsWith("vh_")) viz.nodes.get(id).text.textContent="0";
  }
  if(!c||!c.a) return;
  for(let h=0;h<state.hiddenLayers.length;h++){
    const aLayer=c.a[h+1];
    if(!aLayer) continue;
    for(const id of viz.nodes.keys()){
      if(id.startsWith(`vh_${h}_`)){
        const node=viz.nodes.get(id);
        const v=aLayer[node.neuronIndex];
        if(typeof v==="number") node.text.textContent=v.toFixed(2);
      }
    }
  }
  const out=c.a[c.a.length-1]?.[0];
  if(typeof out==="number") viz.nodes.get("vout_0").text.textContent=out.toFixed(2);
}

/* draw condensed layers + full connections between visible nodes */
function makeGraph(){
  clearSvg();
  viz.hoverEnabled=false;

  const VB_W=900, VB_H=620;
  const left=80, right=VB_W-80;
  const hiddenCount=state.hiddenLayers.length;
  const totalLayers=1+hiddenCount+1;
  const dx=(right-left)/(totalLayers-1);

  for(let l=0;l<totalLayers;l++){
    const x=left+dx*l;
    const t=E("text",{x,y:30,"text-anchor":"middle","font-size":"13",fill:"rgba(107,114,128,1)"});
    if(l===0) t.textContent=`INPUT (${state.inputVisualSize})`;
    else if(l===totalLayers-1) t.textContent=`OUTPUT (1)`;
    else t.textContent=`H${l} (${state.hiddenLayers[l-1]})`;
    svg.appendChild(t);
  }

  function columnHeight(indices,nodeStep,ellipsisStep){
    let h=0; for(const v of indices) h += (v===null)?ellipsisStep:nodeStep;
    return h;
  }
  function startYCentered(totalH){
    const topPad=80, usableH=VB_H-topPad-30;
    return topPad + Math.max(0,(usableH-totalH)/2);
  }

  // input (small condensed)
  {
    const x=left;
    const idx=condensedIndices(state.inputVisualSize, state.inputK);
    const totalH=columnHeight(idx,46,30);
    let y=startYCentered(totalH);
    idx.forEach(v=>{
      if(v===null){ addEllipsis(x,y+6); y+=30; }
      else { addNode(`vin_${v}`,x,y,"0",0,v,true); y+=46; }
    });
  }

  // hidden condensed
  for(let h=0;h<hiddenCount;h++){
    const x=left+dx*(h+1);
    const idx=condensedIndices(state.hiddenLayers[h], state.showK);
    const totalH=columnHeight(idx,46,30);
    let y=startYCentered(totalH);
    idx.forEach(v=>{
      if(v===null){ addEllipsis(x,y+6); y+=30; }
      else { addNode(`vh_${h}_${v}`,x,y,"0",h+1,v,true); y+=46; }
    });
  }

  // output
  addNode("vout_0", right, VB_H/2, "0", totalLayers-1, 0, false);

  // edges: visible-only fully connected
  const inputVisible=[...viz.nodes.keys()].filter(id=>id.startsWith("vin_")).map(id=>viz.nodes.get(id));
  const hiddenVisible=(h)=>[...viz.nodes.keys()].filter(id=>id.startsWith(`vh_${h}_`)).map(id=>viz.nodes.get(id));

  if(hiddenCount>0){
    const firstHiddenVisible=hiddenVisible(0);
    for(const toN of firstHiddenVisible){
      for(const fromN of inputVisible){
        const inJ_internal=fromN.neuronIndex % state.inputSize; // map visual -> true input idx
        addEdge(`e_in_${toN.neuronIndex}_${fromN.neuronIndex}`, `vin_${fromN.neuronIndex}`, `vh_0_${toN.neuronIndex}`, 0, toN.neuronIndex, inJ_internal);
      }
    }
  }

  for(let h=0;h<hiddenCount-1;h++){
    const fromV=hiddenVisible(h);
    const toV=hiddenVisible(h+1);
    const internalL=h+1;
    for(const toN of toV){
      for(const fromN of fromV){
        addEdge(`e_h_${h}_${toN.neuronIndex}_${fromN.neuronIndex}`, `vh_${h}_${fromN.neuronIndex}`, `vh_${h+1}_${toN.neuronIndex}`, internalL, toN.neuronIndex, fromN.neuronIndex);
      }
    }
  }

  if(hiddenCount>0){
    const lastH=hiddenCount-1;
    const fromV=hiddenVisible(lastH);
    const internalL=hiddenCount;
    for(const fromN of fromV){
      addEdge(`e_out_0_${fromN.neuronIndex}`, `vh_${lastH}_${fromN.neuronIndex}`, "vout_0", internalL, 0, fromN.neuronIndex);
    }
  }

  updateVisualNodeValues();
  updateEdgeAppearanceNoLabels();
}

/* ============================================================
   Backprop animation (one edge at a time)
   - shows only ONE weight label and ONE gradient label
   - highlights the current edge + nodes
   ============================================================ */
let steps=[], stepIndex=0, playing=false;

function makeToken(text,x,y){
  const g=E("g");
  const r=E("rect",{x:x-22,y:y-14,width:44,height:28,rx:10,ry:10,fill:"rgba(16,185,129,.95)"});
  const t=E("text",{x,y:y+6,"text-anchor":"middle","font-size":"12","font-weight":"900",fill:"#0b1220"});
  t.textContent=text;
  g.appendChild(r); g.appendChild(t);
  svg.appendChild(g);
  return {g,r,t};
}
function animateToken(token,x1,y1,x2,y2,ms=520){
  return new Promise(resolve=>{
    const start=performance.now();
    function tick(now){
      const u=clamp((now-start)/ms,0,1);
      const x=x1+(x2-x1)*u, y=y1+(y2-y1)*u;
      token.r.setAttribute("x",x-22);
      token.r.setAttribute("y",y-14);
      token.t.setAttribute("x",x);
      token.t.setAttribute("y",y+6);
      if(u<1) requestAnimationFrame(tick);
      else resolve();
    }
    requestAnimationFrame(tick);
  });
}
function removeToken(token){
  if(token?.g?.parentNode) token.g.parentNode.removeChild(token.g);
}

function buildBackpropSteps(){
  steps = [];
  stepIndex = 0;
  viz.hoverEnabled = false;
  tooltip.style.display = "none";

  const g = model.grads;
  const sizes = model.cache.sizes;
  const L = sizes.length - 1;
  const edgeList = [...viz.edges.values()];

  function edgesForLayer(l){
    return edgeList.filter(e => e.l === l);
  }

  // Intro
  steps.push(async () => {
    resetHighlights();
    updateEdgeAppearanceNoLabels();
    setStatus("backprop animation running…", true);
    log("Backprop animation (grouped by neuron).");
  });

  // Layer by layer (output → input)
  for (let l = L - 1; l >= 0; l--) {
    const layerEdges = edgesForLayer(l);

    // 🔑 GROUP BY SOURCE NEURON
    const edgesByFrom = {};
    for (const e of layerEdges) {
      if (!edgesByFrom[e.fromId]) {
        edgesByFrom[e.fromId] = [];
      }
      edgesByFrom[e.fromId].push(e);
    }

    // Deterministic order of neurons
    const fromIds = Object.keys(edgesByFrom).sort((a, b) => {
  const na = viz.nodes.get(a).neuronIndex;
  const nb = viz.nodes.get(b).neuronIndex;
  return na - nb;
});

    for (const fromId of fromIds) {
  const edges = edgesByFrom[fromId];

  // ✅ TRI DÉFINITIF ET CORRECT
  const edgesSorted = [...edges].sort((a, b) => {
    const na = viz.nodes.get(a.toId).neuronIndex;
    const nb = viz.nodes.get(b.toId).neuronIndex;
    return na - nb;
  });

  for (const e of edgesSorted) {
    steps.push(async () => {
      resetHighlights();
      dimAllEdges();
      glowNode(e.fromId);
      glowNode(e.toId);
      glowEdge(e.id);

      const w  = model.W[e.l][e.outI][e.inJ];
      const dw = g.dW[e.l][e.outI][e.inJ];

      setOnlyEdgeLabels(e.id, w, dw);

      const from = viz.nodes.get(e.fromId);
      const to   = viz.nodes.get(e.toId);

      const tok = makeToken(`dw=${dw.toFixed(3)}`, to.x, to.y);
      await animateToken(tok, to.x, to.y, from.x, from.y, 1500);
      removeToken(tok);
    });
  }
}


  }

  // End
  steps.push(async () => {
    resetHighlights();
    updateEdgeAppearanceNoLabels();
    viz.hoverEnabled = true;
    setStatus("done (hover edges)", false);
    log("Backprop animation done.");
  });
}


async function playAll(){
  if(playing) return;
  playing=true;
  while(stepIndex<steps.length){
    await steps[stepIndex++]();
    await new Promise(r=>setTimeout(r,120));
  }
  playing=false;
}


async function playAll(){
  if(playing) return;
  playing=true;
  while(stepIndex<steps.length){
    await steps[stepIndex++]();
    await new Promise(r=>setTimeout(r,120));
  }
  playing=false;
}

/* ============================================================
   Chart
   ============================================================ */
const ctx=document.getElementById("chart").getContext("2d");
let lossHistory=[];
function drawChart(){
  const W=ctx.canvas.width,H=ctx.canvas.height;
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle="rgba(17,24,39,.35)"; ctx.lineWidth=2;
  ctx.strokeRect(60,40,W-90,H-90);
  ctx.fillStyle="rgba(17,24,39,.85)"; ctx.font="16px system-ui";
  ctx.fillText("Training loss (BCE)",60,28);
  if(lossHistory.length<2){
    ctx.fillStyle="rgba(107,114,128,.9)"; ctx.font="13px system-ui";
    ctx.fillText("Upload an image to start.",80,80); return;
  }
  const ys=lossHistory;
  const yMax=Math.max(...ys,0.0001), yMin=Math.min(...ys,0);
  const x0=60,y0=40,w=W-90,h=H-90;
  const xTo=i=>x0+(i/(lossHistory.length-1))*w;
  const yTo=v=>(y0+h)-((v-yMin)/(yMax-yMin+1e-9))*h;
  ctx.strokeStyle="rgba(59,130,246,.95)"; ctx.lineWidth=3;
  ctx.beginPath(); ctx.moveTo(xTo(0),yTo(ys[0]));
  for(let i=1;i<ys.length;i++) ctx.lineTo(xTo(i),yTo(ys[i]));
  ctx.stroke();
}

/* ============================================================
   Upload -> training -> backprop animation auto-start
   ============================================================ */
const fileEl=document.getElementById("file");
const btnChoose=document.getElementById("btnChoose");
const preview=document.getElementById("preview");
const previewImg=document.getElementById("previewImg");
const dropZone=document.getElementById("dropZone");

btnChoose.addEventListener("click",()=>fileEl.click());
dropZone.addEventListener("dragover",(e)=>{e.preventDefault(); dropZone.style.borderColor="rgba(59,130,246,.55)";});
dropZone.addEventListener("dragleave",()=>{dropZone.style.borderColor="var(--border)";});
dropZone.addEventListener("drop",(e)=>{e.preventDefault(); dropZone.style.borderColor="var(--border)"; if(e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);});
fileEl.addEventListener("change",()=>{if(fileEl.files?.[0]) handleFile(fileEl.files[0]);});

function loadImage(file){
  return new Promise((resolve,reject)=>{
    const url=URL.createObjectURL(file);
    const img=new Image();
    img.onload=()=>{ URL.revokeObjectURL(url); resolve(img); };
    img.onerror=reject;
    img.src=url;
  });
}
async function imageToFeatures(file){
  const img=await loadImage(file);
  const c=document.createElement("canvas");
  const s=64; c.width=s; c.height=s;
  const cx=c.getContext("2d");
  cx.drawImage(img,0,0,s,s);
  const data=cx.getImageData(0,0,s,s).data;
  let mean=0,diff=0;
  for(let i=0;i<data.length;i+=4){
    const r=data[i],g=data[i+1],b=data[i+2];
    const lum=(0.2126*r+0.7152*g+0.0722*b);
    mean+=lum; diff+=(r-b);
  }
  const n=data.length/4;
  mean/=n; diff/=n;
  return [(mean/255)*2-1, clamp(diff/255,-1,1)];
}

async function handleFile(file) {
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    preview.style.display = "block";
    
    logEl.innerHTML = "";
    setStatus("Traitement de l'image...", true);
    log(`Fichier téléchargé : ${file.name}`);
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        console.log("Envoi de la requête au serveur...");
        
        const response = await fetch('http://localhost:8001/process-image', {
            method: 'POST',
            body: formData
        });
        
        console.log("Réponse reçue, statut:", response.status);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error("Erreur du serveur:", errorText);
            throw new Error(`Erreur HTTP: ${response.status} - ${errorText}`);
        }
        
        const result = await response.json();
        console.log("Réponse JSON parsée:", result);
        
        if (result.success) {
            setStatus("Image traitée avec succès !");
            log("L'image a été validée par le pipeline de traitement.");
            
            if (result.data && result.data.processing_steps) {
                log("Étapes de traitement effectuées :");
                result.data.processing_steps.forEach((step, index) => {
                    log(`  ${index + 1}. ${step}`);
                });
            }
            
            // ----- Entraînement et animation de rétropropagation -----
            // 1. Paramètres issus de l'interface
            state.reg = parseFloat(regEl.value) || 0;
            state.showK = Math.max(2, parseInt(showKEl.value || "6", 10));
            state.inputK = Math.max(1, parseInt(inKEl.value || "3", 10));
            epochView.textContent = epochEl.value;

            // 2. Réinitialisation des poids et redessiner le graphe
            initParams();
            makeGraph();

            // 3. Extraction des caractéristiques de l'image
            const x = await imageToFeatures(file);
            log(`Caractéristiques extraites : x=[${x.map(v => v.toFixed(3)).join(", ")}]`);

            // 4. Déterminer l'étiquette (backend ou aléatoire)
            const y = result.data && result.data.prediction !== undefined ? result.data.prediction : (Math.random() > 0.5 ? 1 : 0);
            labelOut.textContent = String(y);

            // 5. Boucle d'entraînement
            const EPOCHS = Math.max(1, parseInt(epochEl.value || "1", 10));
            const eta = parseFloat(lrEl.value) || 0.1;
            lossHistory = [];

            for (let e = 0; e < EPOCHS; e++) {
                const { yhat, C } = forward(x, y);
                backward();
                applyUpdate(eta);
                lossHistory.push(C);
                yhatOut.textContent = yhat.toFixed(4);
                lossOut.textContent = C.toFixed(4);
                updateVisualNodeValues();
            }
            drawChart();

            // 6. Dernière passe pour les gradients et animation
            forward(x, y);
            backward();
            log("Démarrage de l'animation de rétropropagation...");
            buildBackpropSteps();
            await playAll();
            log("Animation terminée !");

            // ----- Génération d'un fichier JSON avec les méta-données -----
            function serialize(obj){
                if(Array.isArray(obj)) return obj;
                if(obj && typeof obj === 'object'){
                    if('data' in obj) return obj.data; // petite matrice custom
                    if('toArray' in obj) return obj.toArray();
                }
                return obj;
            }
            const metadata = {
                image: file.name,
                processing: result.data,
                features: x,
                                weights: Object.fromEntries(Object.entries(state.params || {}).map(([k,v])=>[k, serialize(v)])),
                                gradients: Object.fromEntries(Object.entries(state.grads || {}).map(([k,v])=>[k, serialize(v)]))
            };
            const blob = new Blob([JSON.stringify(metadata, null, 2)], {type: 'application/json'});
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = `metadata_${file.name.replace(/\.[^/.]+$/, '')}.json`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(link.href);
            log("Fichier JSON téléchargé : " + link.download);
        } else {
            throw new Error(result.detail || "Erreur inconnue lors du traitement");
        }
    } catch (error) {
        console.error("Erreur complète:", error);
        setStatus("Erreur lors du traitement", false);
        log(`Erreur : ${error.message}`);
    }
}
/* Layer controls */
function refreshLayerCount(){
  layerCountEl.textContent=String(state.hiddenLayers.length);
  btnRem.style.opacity=(state.hiddenLayers.length===0)?0.4:1;
}
btnAdd.addEventListener("click",()=>{
  const n=Math.max(1,parseInt(newSizeEl.value||"16",10));
  state.hiddenLayers.push(n);
  initParams(); refreshLayerCount(); makeGraph();
  log("Added hidden layer. Weights reinitialized.");
});
btnRem.addEventListener("click",()=>{
  if(state.hiddenLayers.length===0) return;
  state.hiddenLayers.pop();
  initParams(); refreshLayerCount(); makeGraph();
  log("Removed hidden layer. Weights reinitialized.");
});
showKEl.addEventListener("change",()=>{state.showK=Math.max(2,parseInt(showKEl.value||"6",10)); makeGraph();});
inKEl.addEventListener("change",()=>{state.inputK=Math.max(1,parseInt(inKEl.value||"3",10)); makeGraph();});
actEl.addEventListener("change",()=>{state.act=actEl.value; makeGraph();});
regEl.addEventListener("change",()=>{state.reg=regEl.value;});
epochEl.addEventListener("change",()=>{epochView.textContent=epochEl.value;});

/* init */
initParams();
refreshLayerCount();
makeGraph();
drawChart();
setStatus("waiting for upload",false);
log("Ready. Upload an image to start.");