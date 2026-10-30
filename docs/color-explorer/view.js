'use strict';
const model = globalThis.ZenColorModel;
const $ = id => document.getElementById(id);
const keys = ['media','layout','bits','placement','range','transfer','authority','action','target'];
const mb = n => (n/1e6).toFixed(2)+' MB';
for (const [key,p] of Object.entries(model.presets)) { const o=document.createElement('option');o.value=key;o.textContent=p.name;$('preset').append(o); }
function render(reset=false) {
  if (!$('controls').checkValidity()) return;
  const s=Object.fromEntries(keys.map(k=>[k,$(k).value]));s.bits=Number(s.bits);s.gainmap=$('gainmap').checked;
  const max=2**s.bits-1, previous=Number($('code').max);$('code').max=max;
  if(reset || previous!==max) $('code').value=Math.round(max/2);
  const v=model.sample(s.bits,s.placement,s.range,Number($('code').value));
  $('sample').textContent=`Code ${$('code').value} / ${max} → stored word ${v.stored} (0x${v.stored.toString(16).padStart(4,'0')}). Nominal luma: ${v.luma.toFixed(6)}. If treated as full-range codes: U16 ${v.full16}, nearest U8 ${v.narrow8}. ${s.range === "limited" ? `Explicit limited → full expansion, clipping excursions: U16 ${v.expanded16}.` : ""}`+(s.bits===8?` Widen U8 by replication: ${v.replicated8}; shift-only zero padding: ${v.zeroPadded8}. These are different contracts.`:' Storage shift is not full-range rescaling.');
  $('marker').setAttribute('cx',20+560*Math.min(1,Math.max(0,v.luma)));
  const b=model.storage(Number($('width').value),Number($('height').value),s.bits,s.layout,Number($('padding').value));
  $('storage').replaceChildren();
  const table=document.createElement('table');table.innerHTML='<thead><tr><th>Plane</th><th>Dimensions</th><th>Stride</th><th>Minimum span</th><th>Full rows</th></tr></thead>';
  for(const p of b.planes){const tr=document.createElement('tr');for(const val of [p.name,`${p.width} × ${p.height}`,`${p.stride} B`,mb(p.span),mb(p.allocation)]){const td=document.createElement('td');td.textContent=val;tr.append(td);}table.append(tr);}$('storage').append(table);
  const info=document.createElement('p');info.textContent=`Full-row allocation: ${mb(b.total)}; packed payload: ${mb(b.packed)}. A full RGB F32 intermediate would cost ${mb(b.f32rgb)}. Two RGB(A) F32 scratch rows cost ${mb(b.scratch)}. These are arithmetic sizes, not process peak-memory predictions. Retain strides with into_parts() when the consumer can use them.`;$('storage').append(info);
  $('current').textContent=`${s.bits}-bit ${s.layout}, ${s.transfer}; authority: ${s.authority}. Native sample placement and range must agree with these values.`;
  $('output').textContent=`Requested ${s.target}. Produce matching pixels and tags together. The actual container and display may impose more requirements.`;
  const result=model.evaluate(s);
  for(const [id,items] of [['decisions',result.decisions],['notes',result.notes]]) {$(id).replaceChildren();for(const text of items){const p=document.createElement('p');p.textContent=text;if(id==='notes')p.className='note';$(id).append(p);}}
}
function preset(){const p=model.presets[$('preset').value];for(const k of keys)$(k).value=p[k];$('gainmap').checked=p.gainmap;render(true);}
$('preset').addEventListener('change',preset);$('controls').addEventListener('input',()=>render());$('code').addEventListener('input',()=>render());$('controls').addEventListener('submit',e=>e.preventDefault());preset();
