//! Classifies the HDR differences of benchmarks/jxl-cms-parity against exact math.
use jxl::api::{JxlCms, JxlCmsTransformer, JxlColorEncoding as Enc, JxlColorProfile as Prof, JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat, JxlPrimaries as P, JxlTransferFunction as T, JxlWhitePoint as W, ProcessingResult, RenderingIntent};
use lcms_cref as c;
use moxcms::*;
use std::os::raw::c_void;
use std::sync::Mutex;
use zenpixels::{Cicp, ColorProfileSource as Src, PixelFormat as Pf};
use zenpixels_convert::cms::{PluggableCms, RowTransformMut};
use zenpixels_convert::cms_moxcms::MoxCms as ZMox;
use zenpixels_convert::policy::ConvertOptions;

whereat::define_at_crate_info!();

// ---------- the benchmark's adapter, unchanged ----------
struct Adapter;
struct Tr { inner: Mutex<Box<dyn RowTransformMut>>, inc: usize, outc: usize }
fn fmt_of(ch: usize) -> Pf { match ch { 1 => Pf::GrayF32, 3 => Pf::RgbF32, 4 => Pf::CmykF32Inverted, _ => panic!() } }
fn is_srgb(p: &Prof) -> bool { matches!(p, Prof::Simple(Enc::RgbColorSpace { white_point: W::D65, primaries: P::SRGB, transfer_function: T::SRGB, .. })) }
impl JxlCms for Adapter {
    fn initialize_transforms(&self, n: usize, max: usize, input: Prof, output: Prof, _i: f32) -> jxl::api::Result<(usize, Vec<Box<dyn JxlCmsTransformer + Send + Sync>>)> {
        let err = |s: String| whereat::at!(jxl::api::Error::CmsError(s));
        let (inc, outc) = (input.channels(), output.channels());
        let si = input.try_as_icc().map(|c| c.into_owned()).unwrap_or_default();
        let di = output.try_as_icc().map(|c| c.into_owned()).unwrap_or_default();
        let srgb = Src::Cicp(Cicp::new(1, 13, 0, true));
        let opts = ConvertOptions::permissive().with_clip_out_of_gamut(false);
        let mut v: Vec<Box<dyn JxlCmsTransformer + Send + Sync>> = Vec::new();
        for _ in 0..n {
            let s = if is_srgb(&input) { srgb.clone() } else { Src::Icc(&si) };
            let d = if is_srgb(&output) { srgb.clone() } else { Src::Icc(&di) };
            let mut t = ZMox.build_source_transform(s, d, fmt_of(inc), fmt_of(outc), &opts).ok_or_else(|| err("declined".into()))?.map_err(|e| err(format!("{e:?}")))?;
            t.prepare(max as u32).map_err(|e| err(format!("{e:?}")))?;
            v.push(Box::new(Tr { inner: Mutex::new(t), inc, outc }));
        }
        Ok((outc, v))
    }
}
impl JxlCmsTransformer for Tr {
    fn do_transform(&mut self, input: &[f32], output: &mut [f32]) -> jxl::api::Result<()> {
        let w = (input.len() / self.inc) as u32;
        self.inner.get_mut().unwrap().try_transform_row(bytemuck::cast_slice(input), bytemuck::cast_slice_mut(output), w).map_err(|e| whereat::at!(jxl::api::Error::CmsError(format!("{e:?}"))))
    }
    fn do_transform_inplace(&mut self, inout: &mut [f32]) -> jxl::api::Result<()> {
        let copy = inout.to_vec();
        let mut out = vec![0f32; copy.len() / self.inc * self.outc];
        self.do_transform(&copy, &mut out)?;
        inout[..out.len()].copy_from_slice(&out);
        Ok(())
    }
}

// ---------- decode ----------
fn complete<S, O>(mut state: S, input: &mut &[u8], mut step: impl FnMut(S, &mut &[u8]) -> jxl::api::Result<ProcessingResult<O, S>>) -> jxl::api::Result<O> {
    for _ in 0..4096 {
        match step(state, input)? {
            ProcessingResult::Complete { result } => return Ok(result),
            ProcessingResult::NeedsMoreInput { fallback, .. } => state = fallback,
        }
    }
    panic!("stalled");
}
/// Decode with `cms`, or without one into `target` when the decoder can reach it natively.
fn decode(bytes: &[u8], cms: Option<Box<dyn JxlCms>>, target: Option<&Prof>) -> (usize, usize, Vec<f32>, Prof, f32) {
    let mut options = JxlDecoderOptions::default().with_adjust_orientation(false).with_render_spot_colors(false);
    if let Some(c) = cms { options = options.with_cms(c); }
    let mut input = bytes;
    let mut d = complete(JxlDecoder::new(options), &mut input, |d, i| d.process(i)).unwrap();
    let extras = d.basic_info().extra_channels.len();
    let intensity = d.basic_info().tone_mapping.intensity_target;
    let embedded = d.embedded_color_profile().clone();
    d.set_pixel_format(JxlPixelFormat { color_type: JxlColorType::Rgb, color_data_format: Some(JxlDataFormat::f32()), extra_channel_format: vec![None; extras] });
    if let Some(t) = target { d.set_output_color_profile(t.clone()).unwrap(); }
    let frame = complete(d, &mut input, |d, i| d.process(i)).unwrap();
    let (w, h) = frame.frame_header().size;
    let mut data = vec![0f32; w * h * 3];
    let mut b = [JxlOutputBuffer::new_with_stride(bytemuck::cast_slice_mut(&mut data), h, w * 12, w * 12)];
    complete(frame, &mut input, |d, i| d.process(i, &mut b)).unwrap();
    (w, h, data, embedded, intensity)
}

// ---------- readers on raw rows ----------
const NOCACHE_BPC_HIGHRES: u32 = 0x40 | 0x2000 | 0x400;
const fn flt_rgb() -> u32 { 1 << 22 | 4 << 16 | 3 << 3 | 4 }
fn c_lcms(src: &[u8], dst: &[u8], flags: u32, input: &[f32]) -> Vec<f32> {
    let n = input.len() / 3;
    let mut out = vec![0f32; n * 3];
    unsafe {
        let p = c::cmsOpenProfileFromMem(src.as_ptr() as *const c_void, src.len() as u32);
        let q = c::cmsOpenProfileFromMem(dst.as_ptr() as *const c_void, dst.len() as u32);
        assert!(!p.is_null() && !q.is_null());
        let t = c::cmsCreateTransform(p, flt_rgb(), q, flt_rgb(), 1, flags);
        assert!(!t.is_null());
        c::cmsDoTransform(t, input.as_ptr() as *const c_void, out.as_mut_ptr() as *mut c_void, n as u32);
        c::cmsDeleteTransform(t);
    }
    out
}
fn port_lcms(src: &[u8], dst: &[u8], flags: u32, input: &[f32]) -> Vec<f32> {
    let n = input.len() / 3;
    let mut out = vec![0f32; n * 3];
    let p = lcms::profile::open_profile_from_mem(None, src).unwrap();
    let q = lcms::profile::open_profile_from_mem(None, dst).unwrap();
    let mut ps = vec![p, q];
    let bpc = flags & 0x2000 != 0;
    let mut t = lcms::xform::create_extended_transform(None, &mut ps, &mut [bpc, bpc], &[1, 1], &[1.0, 1.0], None, 0, flt_rgb(), flt_rgb(), flags).unwrap();
    t.do_transform(bytemuck::cast_slice(input), bytemuck::cast_slice_mut(&mut out), n as u32);
    out
}
fn via_jxl_cms(cms: &dyn JxlCms, a: &Prof, b: &Prof, input: &[f32]) -> Vec<f32> {
    let n = input.len() / 3;
    let (ch, mut t) = cms.initialize_transforms(1, n, a.clone(), b.clone(), 255.0).unwrap();
    assert_eq!(ch, 3);
    let mut out = vec![0f32; n * 3];
    t[0].do_transform(input, &mut out).unwrap();
    out
}
fn mox_direct(s: &ColorProfile, d: &ColorProfile, o: TransformOptions, input: &[f32]) -> Vec<f32> {
    let t = s.create_transform_f32(Layout::Rgb, d, Layout::Rgb, o).unwrap();
    let mut out = vec![0f32; input.len()];
    t.transform(input, &mut out).unwrap();
    out
}

// ---------- exact math ----------
const PCS: [f64; 3] = [0.9642, 1.0, 0.8249];
fn lab_to_xyz(l: f64, a: f64, b: f64) -> [f64; 3] {
    let fy = (l + 16.0) / 116.0;
    let finv = |t: f64| if t > 6.0 / 29.0 { t * t * t } else { 3.0 * (6.0f64 / 29.0).powi(2) * (t - 4.0 / 29.0) };
    [PCS[0] * finv(fy + a / 500.0), PCS[1] * finv(fy), PCS[2] * finv(fy - b / 200.0)]
}
fn srgb_enc(v: f64) -> f64 { if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 } }
fn find_tag<'a>(icc: &'a [u8], sig: &[u8; 4]) -> Option<&'a [u8]> {
    let n = u32::from_be_bytes(icc[128..132].try_into().unwrap()) as usize;
    for i in 0..n {
        let e = &icc[132 + 12 * i..144 + 12 * i];
        if &e[..4] == sig {
            let (o, l) = (u32::from_be_bytes(e[4..8].try_into().unwrap()) as usize, u32::from_be_bytes(e[8..12].try_into().unwrap()) as usize);
            return Some(&icc[o..o + l]);
        }
    }
    None
}
/// mft1 evaluated as lcms2 does: input curves, tetrahedral CLUT, output curves.
struct Mft1 { g: usize, inp: Vec<f64>, clut: Vec<f64>, out: Vec<f64> }
impl Mft1 {
    fn parse(tag: &[u8]) -> Self {
        assert_eq!(&tag[..4], b"mft1");
        let g = tag[10] as usize;
        let inp: Vec<f64> = tag[48..48 + 768].iter().map(|&v| v as f64 / 255.0).collect();
        let n = g * g * g * 3;
        let clut: Vec<f64> = tag[48 + 768..48 + 768 + n].iter().map(|&v| v as f64 / 255.0).collect();
        let out: Vec<f64> = tag[48 + 768 + n..48 + 768 + n + 768].iter().map(|&v| v as f64 / 255.0).collect();
        Mft1 { g, inp, clut, out }
    }
    fn curve(t: &[f64], x: f64) -> f64 { let f = x.clamp(0.0, 1.0) * 255.0; let i = (f as usize).min(254); t[i] + (t[i + 1] - t[i]) * (f - i as f64) }
    fn eval_tri(&self, rgb: [f64; 3]) -> [f64; 3] {
        let g = self.g;
        let v = [0, 1, 2].map(|c| Self::curve(&self.inp[c * 256..(c + 1) * 256], rgb[c]) * (g - 1) as f64);
        let i = v.map(|x| (x as usize).min(g - 2));
        let (rx, ry, rz) = (v[0] - i[0] as f64, v[1] - i[1] as f64, v[2] - i[2] as f64);
        let node = |a: usize, b: usize, c: usize, o: usize| self.clut[(((i[0] + a) * g + (i[1] + b)) * g + (i[2] + c)) * 3 + o];
        let mut res = [0.0; 3];
        for o in 0..3 {
            let l = |a: f64, b: f64, t: f64| a + (b - a) * t;
            let x00 = l(node(0,0,0,o), node(1,0,0,o), rx);
            let x10 = l(node(0,1,0,o), node(1,1,0,o), rx);
            let x01 = l(node(0,0,1,o), node(1,0,1,o), rx);
            let x11 = l(node(0,1,1,o), node(1,1,1,o), rx);
            res[o] = Self::curve(&self.out[o * 256..(o + 1) * 256], l(l(x00, x10, ry), l(x01, x11, ry), rz));
        }
        res
    }
    fn eval(&self, rgb: [f64; 3]) -> [f64; 3] {
        let g = self.g;
        let v = [0, 1, 2].map(|c| Self::curve(&self.inp[c * 256..(c + 1) * 256], rgb[c]) * (g - 1) as f64);
        let i = v.map(|x| (x as usize).min(g - 2));
        let (rx, ry, rz) = (v[0] - i[0] as f64, v[1] - i[1] as f64, v[2] - i[2] as f64);
        let node = |a: usize, b: usize, c: usize, o: usize| self.clut[(((i[0] + a) * g + (i[1] + b)) * g + (i[2] + c)) * 3 + o];
        let mut res = [0.0; 3];
        for o in 0..3 {
            let c0 = node(0, 0, 0, o);
            let c111 = node(1, 1, 1, o);
            let (c1, c2, c3);
            if rx >= ry && ry >= rz { c1 = node(1,0,0,o) - c0; c2 = node(1,1,0,o) - node(1,0,0,o); c3 = c111 - node(1,1,0,o); }
            else if rx >= rz && rz >= ry { c1 = node(1,0,0,o) - c0; c2 = c111 - node(1,0,1,o); c3 = node(1,0,1,o) - node(1,0,0,o); }
            else if rz >= rx && rx >= ry { c1 = node(1,0,1,o) - node(0,0,1,o); c2 = c111 - node(1,0,1,o); c3 = node(0,0,1,o) - c0; }
            else if ry >= rx && rx >= rz { c1 = node(1,1,0,o) - node(0,1,0,o); c2 = node(0,1,0,o) - c0; c3 = c111 - node(1,1,0,o); }
            else if ry >= rz && rz >= rx { c1 = c111 - node(0,1,1,o); c2 = node(0,1,0,o) - c0; c3 = node(0,1,1,o) - node(0,1,0,o); }
            else { c1 = c111 - node(0,1,1,o); c2 = node(0,1,1,o) - node(0,0,1,o); c3 = node(0,0,1,o) - c0; }
            res[o] = Self::curve(&self.out[o * 256..(o + 1) * 256], c0 + c1 * rx + c2 * ry + c3 * rz);
        }
        res
    }
}
fn pq_eotf_nits(e: f64) -> f64 {
    let (m1, m2, c1, c2, c3) = (0.1593017578125, 78.84375, 0.8359375, 18.8515625, 18.6875);
    let p = e.max(0.0).powf(1.0 / m2);
    10000.0 * ((p - c1).max(0.0) / (c2 - c3 * p)).powf(1.0 / m1)
}
fn pq_inv(n: f64) -> f64 {
    let (m1, m2, c1, c2, c3) = (0.1593017578125, 78.84375, 0.8359375, 18.8515625, 18.6875);
    let y = (n / 10000.0).max(0.0).powf(m1);
    ((c1 + c2 * y) / (1.0 + c3 * y)).powf(m2)
}
fn hlg_scene(e: f64) -> f64 { let (a, b, c) = (0.17883277, 0.28466892, 0.55991073); if e <= 0.5 { e * e / 3.0 } else { (((e - c) / a).exp() + b) / 12.0 } }
const BT2100_Y: [f64; 3] = [0.2627, 0.678, 0.0593];
/// The profile's Rec. 2408 / HLG OOTF plus gamut map, unrounded, as tone_map_pixel computes its nodes.
fn tone_mapped_linear(pq: bool, rgb: [f64; 3]) -> [f64; 3] {
    let y = |v: [f64; 3]| v[0] * BT2100_Y[0] + v[1] * BT2100_Y[1] + v[2] * BT2100_Y[2];
    let mut l;
    if pq {
        l = rgb.map(|e| pq_eotf_nits(e) / 10000.0);
        let (src_hi, tgt_hi) = (10000.0, 250.0);
        let pm_hi = pq_inv(src_hi); let pm_lo = pq_inv(0.0); let range = pm_hi - pm_lo;
        let min_lum = (pq_inv(0.0) - pm_lo) / range;
        let max_lum = (pq_inv(tgt_hi) - pm_lo) / range;
        let ks = 1.5 * max_lum - 0.5;
        let lum = src_hi * y(l);
        let e1 = ((pq_inv(lum) - pm_lo) / range).min(1.0);
        let e2 = if e1 < ks { e1 } else {
            let t = (e1 - ks) / (1.0f64 - ks).max(1e-6);
            let (t2, t3) = (t * t, t * t * t);
            (2.0 * t3 - 3.0 * t2 + 1.0) * ks + (t3 - 2.0 * t2 + t) * (1.0 - ks) + (-2.0 * t3 + 3.0 * t2) * max_lum
        };
        let e3 = min_lum * (1.0 - e2).powi(4) + e2;
        let e4 = e3 * range + pm_lo;
        let nl = pq_eotf_nits(e4).clamp(0.0, tgt_hi);
        let ratio = nl / lum.max(1e-6);
        let mult = ratio * (src_hi / tgt_hi);
        l = if lum <= 1e-6 { [nl / tgt_hi; 3] } else { l.map(|v| v * mult) };
    } else {
        l = rgb.map(hlg_scene);
        let exp = 1.2f64 * 1.111f64.powf((80.0f64 / 1e3).log2()) - 1.0;
        if exp.abs() >= 0.1 && y(l) > 0.0 { let m = y(l).powf(exp); l = l.map(|v| v * m); }
    }
    // gamut_map, preserve_saturation 0.3
    let lum = y(l);
    let (mut gs, mut gl) = (0.0f64, 0.0f64);
    for &v in &l {
        let d = v - lum;
        let inv = if d == 0.0 { 1.0 } else { 1.0 / d };
        let over = v * inv;
        if d < 0.0 { gs = gs.max(over); }
        gl = gl.max(if d <= 0.0 { gs } else { over - inv });
    }
    let mix = (0.3 * (gs - gl) + gl).clamp(0.0, 1.0);
    l = l.map(|v| mix * (lum - v) + v);
    let mx = l[0].max(l[1]).max(l[2]).max(1.0);
    l.map(|v| v / mx)
}
/// libjxl's CMS path: the signal to display light relative to intensity_target, no tone map.
fn display_linear(pq: bool, intensity: f64, rgb: [f64; 3]) -> [f64; 3] {
    if pq { rgb.map(|e| pq_eotf_nits(e) / intensity) } else {
        let l = rgb.map(hlg_scene);
        let exp = 1.2f64 * 1.111f64.powf((intensity / 1e3).log2()) - 1.0;
        if exp.abs() < 0.1 { return l; }
        let m = (l[0] * BT2100_Y[0] + l[1] * BT2100_Y[1] + l[2] * BT2100_Y[2]).powf(exp);
        l.map(|v| v * m)
    }
}
struct Exact { lut: Mft1, pq: bool, intensity: f64, xyz_to_dst: [[f64; 3]; 3], bt2100_to_dst: [[f64; 3]; 3], src_to_xyz: [[f64; 3]; 3] }
fn mul3(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] { [0, 1, 2].map(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2]) }
impl Exact {
    fn enc(&self, xyz: [f64; 3]) -> [f64; 3] { mul3(&self.xyz_to_dst, xyz).map(|v| srgb_enc(v.clamp(0.0, 1.0))) }
    fn via_lut(&self, rgb: [f64; 3]) -> [f64; 3] {
        let o = self.lut.eval(rgb);
        self.enc(lab_to_xyz(o[0] * 100.0, o[1] * 255.0 - 128.0, o[2] * 255.0 - 128.0))
    }
    fn via_lut_tri(&self, rgb: [f64; 3]) -> [f64; 3] {
        let o = self.lut.eval_tri(rgb);
        self.enc(lab_to_xyz(o[0] * 100.0, o[1] * 255.0 - 128.0, o[2] * 255.0 - 128.0))
    }
    fn via_tone_map(&self, rgb: [f64; 3]) -> [f64; 3] {
        let l = tone_mapped_linear(self.pq, rgb);
        // tone_map_pixel writes Lab against its own white (0.964212, 1, 0.825188); the reader's PCS is 0.9642, 1, 0.8249.
        let xyz = mul3(&self.src_to_xyz, l);
        let (xn, zn) = (0.964212, 0.825188);
        let f = |t: f64| if t <= (6.0f64 / 29.0).powi(3) { t / (3.0 * (6.0f64 / 29.0).powi(2)) + 4.0 / 29.0 } else { t.cbrt() };
        let (fx, fy, fz) = (f(xyz[0] / xn), f(xyz[1]), f(xyz[2] / zn));
        self.enc(lab_to_xyz(116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)))
    }
    fn direct(&self, rgb: [f64; 3]) -> [f64; 3] {
        mul3(&self.bt2100_to_dst, display_linear(self.pq, self.intensity, rgb)).map(|v| srgb_enc(v.clamp(0.0, 1.0)))
    }
}
fn eval_all(input: &[f32], f: impl Fn([f64; 3]) -> [f64; 3]) -> Vec<f32> {
    input.chunks_exact(3).flat_map(|p| f([p[0] as f64, p[1] as f64, p[2] as f64]).map(|v| v as f32)).collect()
}

// ---------- comparison ----------
fn cmp(name: &str, input: &[f32], a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let (mut worst, mut at, mut sum, mut signed, mut o1, mut o2) = (0f64, 0usize, 0f64, 0f64, 0usize, 0usize);
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let d = (x.clamp(0.0, 1.0) as f64 - y.clamp(0.0, 1.0) as f64) * 255.0;
        sum += d.abs(); signed += d;
        if d.abs() > worst { worst = d.abs(); at = i / 3; }
        if d.abs() > 1.0 { o1 += 1 }
        if d.abs() > 2.0 { o2 += 1 }
    }
    let n = a.len() as f64;
    println!("  {name:<34} worst {worst:7.2} mean|d| {:7.3} mean {:+7.3} >1 {:6.2}% >2 {:6.2}%  worst at in {:.4?} -> {:.4?} vs {:.4?}", sum / n, signed / n, 100.0 * o1 as f64 / n, 100.0 * o2 as f64 / n, &input[at * 3..at * 3 + 3], &a[at * 3..at * 3 + 3], &b[at * 3..at * 3 + 3]);
}
fn read_pfm(path: &str) -> (usize, usize, Vec<f32>) {
    let d = std::fs::read(path).unwrap();
    let (mut i, mut nl) = (0, 0);
    while nl < 3 { if d[i] == 10 { nl += 1 } i += 1 }
    let hdr = String::from_utf8_lossy(&d[..i]).to_string();
    let mut t = hdr.split_whitespace().skip(1);
    let (w, h): (usize, usize) = (t.next().unwrap().parse().unwrap(), t.next().unwrap().parse().unwrap());
    let be = t.next().unwrap().parse::<f32>().unwrap() > 0.0;
    let v: Vec<f32> = d[i..].chunks_exact(4).map(|b| { let a: [u8; 4] = b.try_into().unwrap(); if be { f32::from_be_bytes(a) } else { f32::from_le_bytes(a) } }).collect();
    let mut o = vec![0f32; v.len()];
    for y in 0..h { o[y * w * 3..(y + 1) * w * 3].copy_from_slice(&v[(h - 1 - y) * w * 3..(h - y) * w * 3]); }
    (w, h, o)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (res, dj) = (&args[1], &args[2]);
    let m_opts = TransformOptions { allow_extended_range_rgb_xyz: true, rendering_intent: RenderingIntent2::Perceptual, barycentric_weight_scale: BarycentricWeightScale::High, ..TransformOptions::default() };
    for (file, pq, tf) in [("hdr_pq_test", true, T::PQ), ("hdr_hlg_test", false, T::HLG)] {
        let bytes = std::fs::read(format!("{res}/{file}.jxl")).unwrap();
        let src = Prof::Simple(Enc::RgbColorSpace { white_point: W::D65, primaries: P::BT2100, transfer_function: tf, rendering_intent: RenderingIntent::Relative });
        let (w, h, raw, embedded, intensity) = decode(&bytes, None, Some(&src));
        println!("==== {file}: {w}x{h}, intensity_target {intensity}, embedded {embedded:?}");
        let src_icc = src.try_as_icc().unwrap().into_owned();
        println!("  source ICC {} bytes, version {:02x}, pcs {:?}, tags A2B0 {} A2B1 {} rTRC {} cicp {}", src_icc.len(), src_icc[8], String::from_utf8_lossy(&src_icc[20..24]), find_tag(&src_icc, b"A2B0").is_some(), find_tag(&src_icc, b"A2B1").is_some(), find_tag(&src_icc, b"rTRC").is_some(), find_tag(&src_icc, b"cicp").is_some());
        for (tn, dst, djn) in [
            ("sRGB", Prof::Simple(Enc::srgb(false)), "SRG"),
            ("P3", Prof::Simple(Enc::RgbColorSpace { white_point: W::D65, primaries: P::P3, transfer_function: T::SRGB, rendering_intent: RenderingIntent::Relative }), "P3"),
        ] {
            let dst_icc = dst.try_as_icc().unwrap().into_owned();
            let sp = ColorProfile::new_from_slice(&src_icc).unwrap();
            let dp = ColorProfile::new_from_slice(&dst_icc).unwrap();
            let bt2100 = { let mut p = ColorProfile::new_bt2020(); let (r, g, b) = (p.red_colorant, p.green_colorant, p.blue_colorant);
                let s = [PCS[0] / (r.x + g.x + b.x), 1.0 / (r.y + g.y + b.y), PCS[2] / (r.z + g.z + b.z)];
                for c in [&mut p.red_colorant, &mut p.green_colorant, &mut p.blue_colorant] { c.x *= s[0]; c.y *= s[1]; c.z *= s[2]; } p };
            let ex = Exact {
                lut: Mft1::parse(find_tag(&src_icc, b"A2B0").unwrap()), pq, intensity: intensity as f64,
                xyz_to_dst: dp.colorant_matrix().inverse().v,
                bt2100_to_dst: dp.colorant_matrix().inverse().mat_mul(bt2100.colorant_matrix()).v,
                src_to_xyz: bt2100.colorant_matrix().v,
            };
            // At the table's own nodes, and at a few cell centers.
            {
                let mut nodes = Vec::new();
                for r in 0..9 { for g in 0..9 { for b in 0..9 { nodes.extend_from_slice(&[r as f32 / 8.0, g as f32 / 8.0, b as f32 / 8.0]); } } }
                let o = TransformOptions { interpolation_method: InterpolationMethod::Tetrahedral, prefer_fixed_point: false, allow_use_cicp_transfer: false, ..m_opts };
                let a = mox_direct(&sp, &dp, o, &nodes);
                let rl = eval_all(&nodes, |v| ex.via_lut(v));
                println!("--- {file} -> {tn}, table nodes only");
                cmp("moxcms tetra vs exact, at nodes", &nodes, &a, &rl);
                for (i, o2) in [([0.0f32, 0.8240, 0.8240]), ([0.5781, 0.6875, 0.6875]), ([0.8125, 0.8125, 0.8125]), ([0.0, 0.8125, 0.8125]), ([0.0, 0.875, 0.875])].iter().map(|v| (v, ())) {
                    let _ = o2;
                    let m = mox_direct(&sp, &dp, o, i);
                    let c = c_lcms(&src_icc, &dst_icc, 0x100, i);
                    let e = eval_all(i, |v| ex.via_lut(v));
                    let lab = ex.lut.eval([i[0] as f64, i[1] as f64, i[2] as f64]);
                    println!("    in {i:?}: moxcms {m:.4?} C {c:.4?} exact {e:.4?} table Lab {:.2} {:.2} {:.2}", lab[0] * 100.0, lab[1] * 255.0 - 128.0, lab[2] * 255.0 - 128.0);
                }
            }
            let mut grid = Vec::new();
            let n = 65usize;
            for r in 0..n { for g in 0..n { for b in 0..n { grid.extend_from_slice(&[r as f32 / (n - 1) as f32, g as f32 / (n - 1) as f32, b as f32 / (n - 1) as f32]); } } }
            for (set, input) in [("grid 65^3", &grid), ("real image rows", &raw)] {
                println!("--- {file} -> {tn}, {set}");
                let a = via_jxl_cms(&Adapter, &src, &dst, input);
                let m = via_jxl_cms(&jxl::api::MoxCms, &src, &dst, input);
                let l = c_lcms(&src_icc, &dst_icc, NOCACHE_BPC_HIGHRES, input);
                let l2 = via_jxl_cms(&jxl_cli::cms::Lcms2Cms, &src, &dst, input);
                let lp = port_lcms(&src_icc, &dst_icc, NOCACHE_BPC_HIGHRES, input);
                let rl = eval_all(input, |v| ex.via_lut(v));
                let rt = eval_all(input, |v| ex.via_tone_map(v));
                let rd = eval_all(input, |v| ex.direct(v));
                cmp("adapter vs decoder wrapper", input, &a, &m);
                cmp("adapter vs C lcms", input, &a, &l);
                cmp("decoder wrapper vs C lcms", input, &m, &l);
                cmp("Lcms2Cms vs C lcms", input, &l2, &l);
                cmp("lcms port vs C lcms", input, &lp, &l);
                cmp("C lcms vs exact mft1", input, &l, &rl);
                cmp("adapter vs exact mft1", input, &a, &rl);
                let rtri = eval_all(input, |v| ex.via_lut_tri(v));
                moxcms::BISECT_SKIP_BPC.store(true, std::sync::atomic::Ordering::Relaxed);
                let nb = mox_direct(&sp, &dp, TransformOptions { interpolation_method: InterpolationMethod::Tetrahedral, prefer_fixed_point: false, allow_use_cicp_transfer: false, ..m_opts }, input);
                moxcms::BISECT_SKIP_BPC.store(false, std::sync::atomic::Ordering::Relaxed);
                cmp("no V4 black point vs TRILINEAR mft1", input, &nb, &rtri);
                cmp("no V4 black point vs tetrahedral mft1", input, &nb, &rl);
                cmp("adapter vs TRILINEAR mft1", input, &a, &rtri);
                cmp("decoder wrapper vs TRILINEAR mft1", input, &m, &rtri);
                cmp("TRILINEAR mft1 vs tetrahedral mft1", input, &rtri, &rl);
                cmp("TRILINEAR mft1 vs analytic tone map", input, &rtri, &rt);
                cmp("decoder wrapper vs exact mft1", input, &m, &rl);
                cmp("exact mft1 vs analytic tone map", input, &rl, &rt);
                cmp("C lcms vs analytic tone map", input, &l, &rt);
                cmp("adapter vs analytic tone map", input, &a, &rt);
                cmp("adapter vs direct (no tone map)", input, &a, &rd);
                cmp("C lcms vs direct (no tone map)", input, &l, &rd);
                // Bisect wrapper -> adapter one option at a time, from the wrapper's own settings.
                let m0 = mox_direct(&sp, &dp, m_opts, input);
                cmp("mox M options vs wrapper", input, &m0, &m);
                let v = |name: &str, s: &ColorProfile, d: &ColorProfile, o: TransformOptions| {
                    let r = mox_direct(s, d, o, input);
                    cmp(&format!("{name} vs M options"), input, &r, &m0);
                    cmp(&format!("{name} vs exact mft1"), input, &r, &rl);
                };
                v("fixed=false", &sp, &dp, TransformOptions { prefer_fixed_point: false, ..m_opts });
                v("use_cicp=false", &sp, &dp, TransformOptions { allow_use_cicp_transfer: false, ..m_opts });
                v("extended=false", &sp, &dp, TransformOptions { allow_extended_range_rgb_xyz: false, ..m_opts });
                let (mut s0, mut d0) = (sp.clone(), dp.clone()); s0.cicp = None; d0.cicp = None;
                v("cicp cleared", &s0, &d0, m_opts);
                v("fixed=false,cicp cleared,no use", &s0, &d0, TransformOptions { prefer_fixed_point: false, allow_use_cicp_transfer: false, ..m_opts });
                // Interpolation: the wrapper leaves moxcms's default, trilinear; the adapter asks for tetrahedral.
                for (name, im) in [("M + Tetrahedral", InterpolationMethod::Tetrahedral), ("M + Pyramid", InterpolationMethod::Pyramid), ("M + Prism", InterpolationMethod::Prism)] {
                    let r = mox_direct(&sp, &dp, TransformOptions { interpolation_method: im, ..m_opts }, input);
                    cmp(&format!("{name} vs M options"), input, &r, &m0);
                    cmp(&format!("{name} vs exact mft1"), input, &r, &rl);
                    if name == "M + Tetrahedral" {
                        cmp("M + Tetrahedral vs adapter", input, &r, &a);
                        // The resample holds linear light, and the sRGB curve is applied afterwards, sign-preserving.
                        let mut lin = dp.clone();
                        lin.cicp = None;
                        for t in [&mut lin.red_trc, &mut lin.green_trc, &mut lin.blue_trc] { *t = Some(ToneReprCurve::Lut(Vec::new())); }
                        let r = mox_direct(&s0, &lin, TransformOptions { interpolation_method: im, prefer_fixed_point: false, allow_use_cicp_transfer: false, ..m_opts }, input);
                        let enc: Vec<f32> = r.iter().map(|&v| { let a = v.abs() as f64; (if a <= 0.003_130_8 { a * 12.92 } else { 1.055 * a.powf(1.0 / 2.4) - 0.055 }).copysign(v as f64) as f32 }).collect();
                        cmp("tetra, linear resample, curve after, vs exact mft1", input, &enc, &rl);
                        cmp("tetra, linear resample, curve after, vs tone map", input, &enc, &rt);
                    }
                }
                // PluggableCms handed CICP instead of the JXL ICC: matrix and analytic curve, no table.
                if let Some(cicp) = (if pq { Some(Cicp::new(9, 16, 0, true)) } else { Some(Cicp::new(9, 18, 0, true)) }) {
                    let opts = ConvertOptions::permissive().with_clip_out_of_gamut(false);
                    let d = if tn == "sRGB" { Src::Cicp(Cicp::new(1, 13, 0, true)) } else { Src::Icc(&dst_icc) };
                    let mut t = ZMox.build_source_transform(Src::Cicp(cicp), d, Pf::RgbF32, Pf::RgbF32, &opts).unwrap().unwrap();
                    let mut o = vec![0f32; input.len()];
                    t.try_transform_row(bytemuck::cast_slice(input), bytemuck::cast_slice_mut(&mut o), (input.len() / 3) as u32).unwrap();
                    cmp("PluggableCms given CICP vs direct", input, &o, &rd);
                }
                if set != "grid 65^3" {
                    let (_, _, full_a, _, _) = decode(&bytes, Some(Box::new(Adapter)), Some(&dst));
                    let (_, _, full_m, _, _) = decode(&bytes, Some(Box::new(jxl::api::MoxCms)), Some(&dst));
                    let (_, _, full_l, _, _) = decode(&bytes, Some(Box::new(jxl_cli::cms::Lcms2Cms)), Some(&dst));
                    cmp("FULL DECODE adapter vs wrapper", input, &full_a, &full_m);
                    cmp("FULL DECODE adapter vs Lcms2Cms", input, &full_a, &full_l);
                    cmp("FULL DECODE wrapper vs Lcms2Cms", input, &full_m, &full_l);
                    cmp("FULL DECODE adapter vs rows", input, &full_a, &a);
                    let (dw, dh, dj_img) = read_pfm(&format!("{dj}/{file}_{djn}.pfm"));
                    assert_eq!((dw, dh), (w, h));
                    cmp("djxl(skcms) vs direct (no tone map)", input, &dj_img, &rd);
                    cmp("djxl(skcms) vs exact mft1", input, &dj_img, &rl);
                    cmp("djxl(skcms) vs FULL adapter", input, &dj_img, &full_a);
                    cmp("djxl(skcms) vs FULL Lcms2Cms", input, &dj_img, &full_l);
                }
            }
        }
    }
}
use moxcms::RenderingIntent as RenderingIntent2;
