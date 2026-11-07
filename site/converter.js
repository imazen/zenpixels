// The teaching model remains usable when the optional WASM build is unavailable.
let wasmConvert = null;
const targetFields = ['format', 'transfer', 'primaries', 'alphamode', 'range'];
for (const field of targetFields) {
  const select = document.getElementById('to-' + field);
  select.innerHTML = document.getElementById(field).innerHTML;
  select.addEventListener('change', renderConverter);
}
document.getElementById('to-format').value = 'Rgba8';
document.getElementById('to-transfer').value = 'Srgb';
document.getElementById('to-primaries').value = 'Bt709';
document.getElementById('to-alphamode').innerHTML = ['Straight', 'Premultiplied', 'Opaque'].map(v => `<option>${v}</option>`).join('');
document.getElementById('to-range').value = 'Full';
for (const id of ['sdr-white', 'hdr-peak', 'sdr-peak']) document.getElementById(id).addEventListener('input', renderConverter);

function converterDescriptor(cfg) {
  const f = FMAP[cfg.format];
  return { ...f, transfer: cfg.transfer, primaries: cfg.primaries, alphaMode: cfg.alphaMode,
    range: isColorish(f) && ['U8', 'U16'].includes(f.ct) ? cfg.range : 'Full' };
}
function pixelBytes(result) {
  // X is unspecified; choose zero on input and exclude it from comparisons.
  return result.chans.flatMap(c => c.bytes || Array(result.ct.bytes).fill(0));
}
function currentConversion() {
  const source = compute({ ...S, be: false });
  const target = { ...S, be: false, format: $('to-format').value, transfer: $('to-transfer').value,
    primaries: $('to-primaries').value, alphaMode: $('to-alphamode').value, range: $('to-range').value };
  const norms = Object.fromEntries(source.chans.filter(c => c.key !== 'X').map(c =>
    [c.key, unstore(source.f.ct, c.text, source.narrow && c.key !== 'A')]));
  const decoded = decodePixel(source.cfg, norms);
  const model = compute({ ...target, lin: decoded.lin, a: decoded.a });
  const st = stride(source.cfg, source.f);
  const one = pixelBytes(source);
  const bytes = Array(st * S.h).fill(0);
  for (let y = 0; y < S.h; y++) for (let x = 0; x < S.w; x++) {
    for (let c = 0; c < one.length; c++) bytes[y * st + x * one.length + c] = one[c];
  }
  return { source, model, request: { from: converterDescriptor(source.cfg), to: converterDescriptor(target),
    bytes, stride: st, width: S.w, rows: S.h, luma: S.luma,
    sdrWhite: $('sdr-white').value === '' ? null : Number($('sdr-white').value),
    hdrPeak: $('hdr-peak').value === '' ? null : Number($('hdr-peak').value),
    sdrPeak: $('sdr-peak').value === '' ? null : Number($('sdr-peak').value) } };
}
const hexBytes = bytes => bytes.map(x => hex2(x).toUpperCase()).join(' ');
function renderConverter() {
  const status = $('wasm-status');
  if (!wasmConvert) return;
  try {
    const { source, model, request } = currentConversion();
    const result = JSON.parse(wasmConvert(JSON.stringify(request)));
    window.lastWasmConversion = { request, result };
    $('wasm-plan').textContent = result.plan || '';
    $('wasm-bytes').textContent = result.bytes ? hexBytes(result.bytes) : result.error;
    $('model-bytes').textContent = hexBytes(pixelBytes(model));
    if (result.error) {
      status.textContent = 'Converter refused: ' + result.error;
      status.className = 'note'; status.dataset.state = 'refused';
      return;
    }
    const expected = pixelBytes(model);
    const different = result.bytes.reduce((n, byte, i) => {
      const c = Math.floor((i % expected.length) / model.ct.bytes);
      return n + (model.chans[c].key !== 'X' && byte !== expected[i % expected.length] ? 1 : 0);
    }, 0);
    const illustrative = source.f.generic || model.f.generic || source.isCmyk || model.isCmyk;
    status.dataset.state = different ? 'different' : 'match';
    status.className = 'note' + (different ? '' : ' info');
    status.textContent = different
      ? `Disagreement: ${different} of ${result.bytes.length} output bytes differ from the JavaScript model. Floating approximations, quantization and HDR anchoring can differ; this is not a correctness verdict.`
      : `Exact agreement on defined channel bytes (${result.bytes.length} output bytes; X padding excluded).`;
    if (illustrative) status.textContent += ' The native/CMYK model uses an illustrative convention.';
  } catch (e) {
    status.dataset.state = 'error'; status.textContent = 'Converter error: ' + e.message;
  }
}
// Called after every explorer edit, including edits to stored channel values.
window.renderConverter = renderConverter;
(async () => {
  try {
    const path = './wasm/converter.js';
    const available = await fetch(path);
    if (!available.ok) throw new Error('WASM assets have not been built');
    const module = await import(path);
    await module.default();
    wasmConvert = module.convert;
    $('wasm-status').dataset.loaded = 'true';
    renderConverter();
  } catch (e) {
    $('wasm-status').textContent = 'JavaScript model only. Optional WASM converter unavailable: ' + e.message;
    $('wasm-status').dataset.state = 'unavailable';
  }
})();
