/* Educational arithmetic; no codec, ICC parser, or browser/display emulation. */
(function (root) {
  'use strict';
  const presets = {
    jpeg: { name: 'JPEG → sRGB PNG', media: 'image', layout: 'rgb', bits: 8, placement: 'lsb', range: 'full', transfer: 'srgb', authority: 'icc', action: 'transcode', target: 'srgb', gainmap: false },
    p010: { name: 'P010 video → HDR display', media: 'video', layout: 'semiplanar', bits: 10, placement: 'msb', range: 'limited', transfer: 'pq', authority: 'cicp', action: 'display', target: 'pq', gainmap: false },
    av1: { name: '12-bit planar AV1 → sRGB', media: 'video', layout: 'planar', bits: 12, placement: 'lsb', range: 'limited', transfer: 'pq', authority: 'cicp', action: 'transcode', target: 'srgb', gainmap: false },
    gain: { name: 'Gain-map JPEG: scrub and rewrite', media: 'image', layout: 'rgb', bits: 8, placement: 'lsb', range: 'full', transfer: 'srgb', authority: 'icc', action: 'rewrite', target: 'srgb', gainmap: true },
    animation: { name: 'RGBA animation → PNG frames', media: 'animation', layout: 'rgba', bits: 16, placement: 'lsb', range: 'full', transfer: 'linear', authority: 'cicp', action: 'transcode', target: 'srgb', gainmap: false },
  };
  function sample(bits, placement, range, code) {
    if (![8, 10, 12, 16].includes(bits)) throw new Error('unsupported teaching depth');
    const max = 2 ** bits - 1, shift = bits === 8 || placement === 'lsb' ? 0 : 16 - bits;
    if (!Number.isInteger(code) || code < 0 || code > max) throw new Error('invalid code');
    const factor = 2 ** (bits - 8), low = range === 'limited' ? 16 * factor : 0;
    const high = range === 'limited' ? 235 * factor : max;
    const full16 = Math.round(code * 65535 / max);
    return { max, shift, stored: code * 2 ** shift, low, high,
      // Luma only: chroma's nominal upper code is 240, not 235.
      luma: (code - low) / (high - low), full16,
      narrow8: Math.round(code * 255 / max),
      replicated8: bits === 8 ? code * 257 : null,
      zeroPadded8: bits === 8 ? code * 256 : null };
  }
  function storage(width, height, bits, layout, padding) {
    if (![width, height, padding].every(Number.isSafeInteger) || width < 1 || height < 1 || padding < 0) throw new Error('invalid dimensions');
    const bytes = bits === 8 ? 1 : 2;
    const shapes = layout === 'planar' ? [['Y', width, height, 1], ['Cb', Math.ceil(width/2), Math.ceil(height/2), 1], ['Cr', Math.ceil(width/2), Math.ceil(height/2), 1]]
      : layout === 'semiplanar' ? [['Y', width, height, 1], ['CbCr', Math.ceil(width/2), Math.ceil(height/2), 2]]
      : [[layout.toUpperCase(), width, height, layout === 'rgba' ? 4 : 3]];
    // Round row padding to a sample boundary; packed row bytes remain exact.
    const planes = shapes.map(([name,w,h,c]) => { const row = w*c*bytes, stride = Math.ceil((row+padding)/bytes)*bytes;
      return {name, width:w, height:h, row, stride, span:(h-1)*stride+row, allocation:h*stride}; });
    return { planes, total: planes.reduce((n,p)=>n+p.allocation,0),
      packed: planes.reduce((n,p)=>n+p.row*p.height,0),
      f32rgb:width*height*12, scratch:width*(layout==='rgba'?4:3)*4*2 };
  }
  function evaluate(s) {
    const notes = [], decisions = [];
    const native = s.layout === 'planar' || s.layout === 'semiplanar';
    if (native) decisions.push('Decode/unpack range and YUV matrix explicitly before packed RGB conversion; retain chroma siting and subsampling.');
    if (s.bits === 10 || s.bits === 12) notes.push('Native codes in U16 are not full-range zenpixels RGB16. SampleEncoding records meaning; it does not convert samples.');
    if (s.authority === 'both') notes.push('BLOCKED: choose current ICC or CICP authority at decode. Keep both original blocks in provenance if needed.');
    if ((s.transfer === 'pq' || s.transfer === 'hlg') && s.target === 'srgb') notes.push('REQUIRES EXPLICIT POLICY: HDR → SDR tone mapping, source peak and output white; no hidden peak scan.');
    if (s.transfer === 'hlg') notes.push('HLG requires scene/display interpretation and viewing assumptions; a transfer curve alone is not a complete display transform.');
    if (s.transfer === 'linear') notes.push('Linear values need a luminance anchor. Do not conflate 203-nit relative white with raw PQ decode (1.0 = 10,000 nits).');
    if (s.action === 'rewrite') {
      decisions.push('Preserve compressed image samples; edit metadata semantically, then serialize headers and rebuild container references.');
      if (s.target !== s.transfer) notes.push('BLOCKED: metadata-only rewrite cannot change pixel color encoding by changing its tags.');
    } else if (s.action === 'transcode') decisions.push('Convert current pixels, then finalize matching context + output metadata. Owned finalization allocates one output image.');
    else decisions.push('Display result depends on OS compositor, browser/app, active output profile, HDR mode and current headroom. Capability is unknown here.');
    if (s.gainmap) notes.push('GAIN MAP: preserve its display-relevant metadata; after XMP serialization rebuild directory lengths and MPF offsets. Unsupported referenced edits must refuse, not copy stale offsets.');
    if (s.media === 'video') notes.push('VIDEO: PTS/DTS, frame order, track color changes, chroma siting, interlace and sequence MaxFALL need media-layer contracts. Audio is a separate track.');
    if (s.media === 'animation') notes.push('ANIMATION: duration, frame rectangles, blend/disposal and canvas color matter. A decoded frame is not automatically a composited canvas.');
    if (s.layout === 'rgba') notes.push('ALPHA: association is independent of transfer. Unassociate before nonlinear color operations; alpha is not transfer-encoded.');
    return {notes, decisions};
  }
  root.ZenColorModel = {presets, sample, storage, evaluate};
  if (typeof module !== 'undefined') module.exports = root.ZenColorModel;
})(globalThis);
