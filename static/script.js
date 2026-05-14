const drop=document.getElementById('dropZone'),fi=document.getElementById('fileInput');
let sel=null;
drop.onclick=()=>fi.click();
drop.ondragover=e=>{e.preventDefault();drop.classList.add('drag')};
drop.ondragleave=()=>drop.classList.remove('drag');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('drag');if(e.dataTransfer.files[0])load(e.dataTransfer.files[0])};
fi.onchange=e=>{if(e.target.files[0])load(e.target.files[0])};

function load(f){
  sel=f;
  const r=new FileReader();
  r.onload=ev=>{
    document.getElementById('prev-img').src=ev.target.result;
    document.getElementById('m-name').textContent=f.name;
    document.getElementById('m-size').textContent=(f.size/1024).toFixed(1)+' KB';
    document.getElementById('m-type').textContent=f.type||'unknown';
    document.getElementById('preview-section').classList.add('v');
    document.getElementById('result-section').classList.remove('v');
    document.getElementById('loading').classList.remove('v');
    
    // Clear heatmaps on new load
    document.getElementById('heatmap-img').src = '';
  };
  r.readAsDataURL(f);
}

document.getElementById('analyze-btn').onclick=async()=>{
  if(!sel)return;
  document.getElementById('analyze-btn').disabled=true;
  document.getElementById('loading').classList.add('v');
  document.getElementById('result-section').classList.remove('v');
  const fd=new FormData();fd.append('file',sel); // Backend should expect 'file'
  try{
    const res=await fetch('/detect',{method:'POST',body:fd});
    const d=await res.json();
    document.getElementById('loading').classList.remove('v');
    if(d.error){
      document.getElementById('loading').innerHTML=`<div class="err">ERROR: ${d.error}</div>`;
      document.getElementById('loading').classList.add('v');
    } else showResult(d);
  }catch(e){
    document.getElementById('loading').classList.remove('v');
    alert('Request failed: '+e.message);
  }finally{document.getElementById('analyze-btn').disabled=false}
};

function showResult(d){
  const v=document.getElementById('verdict');
  v.textContent=d.label;v.className='verdict '+d.label.toLowerCase();
  document.getElementById('conf-val').textContent=d.confidence+'%';
  document.getElementById('rv').textContent=d.real_score+'%';
  document.getElementById('fv').textContent=d.fake_score+'%';
  document.getElementById('ts').textContent=new Date().toISOString().replace('T',' ').slice(0,19);
  
  if(d.model_used) {
    document.getElementById('model-used-display').textContent = 'Model: ' + d.model_used + ' · Weights: Auto · For research use only';
  }

  // Handle heatmap
  if(d.heatmap_path) {
    // Add cache buster to bypass browser cache
    document.getElementById('heatmap-img').src = d.heatmap_path + '?t=' + new Date().getTime();
    document.getElementById('heatmap-container').classList.add('v');
  } else {
    document.getElementById('heatmap-container').classList.remove('v');
  }

  document.getElementById('result-section').classList.add('v');
  setTimeout(()=>{
    document.getElementById('rb').style.width=d.real_score+'%';
    document.getElementById('fb').style.width=d.fake_score+'%';
  },100);
}

document.getElementById('reset-btn').onclick=()=>{
  sel=null;fi.value='';
  ['preview-section','result-section','loading','heatmap-container'].forEach(id=>{
      const el = document.getElementById(id);
      if(el) el.classList.remove('v');
  });
  document.getElementById('rb').style.width='0%';
  document.getElementById('fb').style.width='0%';
  document.getElementById('heatmap-img').src='';
};
