// Every MoxCms route against exact math, plus exact table evaluation for test3 and test1.
use lcms_cref as c;
use moxcms::{BarycentricWeightScale, ColorProfile, InterpolationMethod, Layout, RenderingIntent, ToneReprCurve, TransformOptions};
use std::os::raw::c_void;
use std::sync::atomic::Ordering::Relaxed;
use zenpixels::{Cicp, ColorProfileSource as S, PixelFormat as F};
use zenpixels_convert::ConvertOptions;
use zenpixels_convert::cms::PluggableCms;
use zenpixels_convert::cms_moxcms::MoxCms;

fn curve(t: &ToneReprCurve) -> impl Fn(f64) -> f64 + use<> {
    let t = t.clone();
    move |x: f64| match &t {
        ToneReprCurve::Parametric(p) => { let p: Vec<f64> = p.iter().map(|&v| v as f64).collect(); if x >= p[4] { (p[1] * x + p[2]).powf(p[0]) } else { p[3] * x } }
        ToneReprCurve::Lut(d) if d.len() == 1 => x.powf(d[0] as f64 / 256.0),
        ToneReprCurve::Lut(d) => { let f = x * (d.len() - 1) as f64; let i = (f as usize).min(d.len() - 2); (d[i] as f64 + (d[i + 1] as f64 - d[i] as f64) * (f - i as f64)) / 65535.0 }
    }
}
fn inverse(t: &ToneReprCurve) -> impl Fn(f64) -> f64 + use<> {
    let t = t.clone();
    move |y: f64| match &t {
        ToneReprCurve::Parametric(p) => { let p: Vec<f64> = p.iter().map(|&v| v as f64).collect(); if y >= p[3] * p[4] { (y.powf(1.0 / p[0]) - p[2]) / p[1] } else { y / p[3] } }
        ToneReprCurve::Lut(d) if d.len() == 1 => y.powf(256.0 / d[0] as f64),
        _ => panic!("no inverse"),
    }
}
fn pcs_white(mut p: ColorProfile) -> ColorProfile {
    let (r, g, b) = (p.red_colorant, p.green_colorant, p.blue_colorant);
    let s = [0.9642 / (r.x + g.x + b.x), 1.0 / (r.y + g.y + b.y), 0.8249 / (r.z + g.z + b.z)];
    for c in [&mut p.red_colorant, &mut p.green_colorant, &mut p.blue_colorant] { c.x *= s[0]; c.y *= s[1]; c.z *= s[2]; }
    p
}
fn adapter_srgb() -> ColorProfile {
    pcs_white(ColorProfile::new_from_cicp(moxcms::CicpProfile {
        color_primaries: moxcms::CicpColorPrimaries::Bt709,
        transfer_characteristics: moxcms::TransferCharacteristics::Srgb,
        matrix_coefficients: moxcms::MatrixCoefficients::Identity,
        full_range: true,
    }))
}
#[derive(Default)]
struct Stat { n: u64, sum: f64, wrong: u64, worst: f64, abs: f64 }
impl Stat {
    fn add(&mut self, got: f64, e: f64) { self.n += 1; let d = got - e; self.sum += d; self.abs += d.abs(); if d.abs() > self.worst { self.worst = d.abs() } if got != e.round() { self.wrong += 1 } }
    fn show(&self, name: &str) { println!("{name:<62} mean {:+.4}  mean|d| {:.4}  not-nearest {:6.2}%  worst {:.3}  n={}", self.sum / self.n as f64, self.abs / self.n as f64, 100.0 * self.wrong as f64 / self.n as f64, self.worst, self.n); }
}
struct Exact { m: [[f64; 3]; 3], lin: Vec<f64>, enc: Box<dyn Fn(f64) -> f64> }
impl Exact {
    fn new(s: &ColorProfile, t: &ColorProfile, depth: u32) -> Self {
        let dec = curve(s.red_trc.as_ref().unwrap());
        let n = 1usize << depth;
        Exact { m: t.colorant_matrix().inverse().mat_mul(s.colorant_matrix()).v, lin: (0..n).map(|i| dec(i as f64 / (n - 1) as f64)).collect(), enc: Box::new(inverse(t.red_trc.as_ref().unwrap())) }
    }
    fn unit(&self, c: usize, i: [usize; 3]) -> f64 { let v = self.m[c][0] * self.lin[i[0]] + self.m[c][1] * self.lin[i[1]] + self.m[c][2] * self.lin[i[2]]; (self.enc)(v.clamp(0.0, 1.0)) }
}
fn matrix_routes(dir: &str) {
    let f = |n: &str| std::fs::read(format!("{dir}/{n}")).unwrap();
    let (p3b, adobeb, bt2020b) = (f("DisplayP3Compat-v4.icc"), f("AdobeCompat-v2.icc"), f("Rec2020Compat-v4.icc"));
    let parse = |b: &[u8]| ColorProfile::new_from_slice(b).unwrap();
    let srgb_src = S::Cicp(Cicp::new(1, 13, 0, true));
    let srgb = adapter_srgb();
    let plain = pcs_white(ColorProfile::new_srgb());
    println!("adapter sRGB colorants vs new_srgb(): R {:?} {:?}, G {:?} {:?}, B {:?} {:?}", srgb.red_colorant, plain.red_colorant, srgb.green_colorant, plain.green_colorant, srgb.blue_colorant, plain.blue_colorant);
    let opts = ConvertOptions::permissive();
    let mut cube = Vec::with_capacity(3 << 24);
    for r in 0..=255u8 { for g in 0..=255u8 { for b in 0..=255u8 { cube.extend_from_slice(&[r, g, b]); } } }
    let mut h = 0x9E37_79B9u32;
    let mut next = || { h ^= h << 13; h ^= h >> 17; h ^= h << 5; h };
    let wide: Vec<u16> = (0..3 << 21).map(|_| (next() >> 16) as u16).collect();
    for (name, s, t, sp, tp) in [
        ("P3 -> sRGB", S::Icc(&p3b), srgb_src.clone(), parse(&p3b), srgb.clone()),
        ("sRGB -> P3", srgb_src.clone(), S::Icc(&p3b), srgb.clone(), parse(&p3b)),
        ("Adobe -> sRGB", S::Icc(&adobeb), srgb_src.clone(), parse(&adobeb), srgb.clone()),
        ("P3 -> BT.2020", S::Icc(&p3b), S::Icc(&bt2020b), parse(&p3b), parse(&bt2020b)),
        ("P3 -> Adobe", S::Icc(&p3b), S::Icc(&adobeb), parse(&p3b), parse(&adobeb)),
    ] {
        let x = Exact::new(&sp, &tp, 8);
        let x16 = Exact::new(&sp, &tp, 16);
        let mut o = vec![0u8; cube.len()];
        MoxCms.build_source_transform(s.clone(), t.clone(), F::Rgb8, F::Rgb8, &opts).unwrap().unwrap().try_transform_row(&cube, &mut o, 1 << 24).unwrap();
        let mut st = Stat::default();
        for (i, p) in cube.chunks_exact(3).zip(o.chunks_exact(3)) { for c in 0..3 { st.add(p[c] as f64, x.unit(c, [i[0] as usize, i[1] as usize, i[2] as usize]) * 255.0); } }
        st.show(&format!("{name} Rgb8->Rgb8, exhaustive"));
        let (mut wv, mut wi, mut wc) = (0f64, 0usize, 0usize);
        for (k, (i, p)) in cube.chunks_exact(3).zip(o.chunks_exact(3)).enumerate() { for c in 0..3 { let d = (p[c] as f64 - x.unit(c, [i[0] as usize, i[1] as usize, i[2] as usize]) * 255.0).abs(); if d > wv { wv = d; wi = k; wc = c; } } }
        let i = &cube[wi * 3..wi * 3 + 3];
        println!("    U8 worst: input {i:?} channel {wc}: got {:?}, exact code {:.3}", &o[wi * 3..wi * 3 + 3], x.unit(wc, [i[0] as usize, i[1] as usize, i[2] as usize]) * 255.0);
        let lin = sp.build_8bit_lin_table(&sp.red_trc).unwrap();
        let wt = (0..256).map(|v| (lin[v] as f64 - x.lin[v]).abs()).fold(0.0, f64::max);
        println!("    moxcms 8-bit linearize table vs f64 curve: worst {wt:.3e}");
        // Rgb8 -> Rgba8 takes the other route.
        let few = &cube[..3 << 22];
        let mut o2 = vec![0u8; (few.len() / 3) * 4];
        MoxCms.build_source_transform(s.clone(), t.clone(), F::Rgb8, F::Rgba8, &opts).unwrap().unwrap().try_transform_row(few, &mut o2, (few.len() / 3) as u32).unwrap();
        let mut st = Stat::default();
        let mut alpha_bad = 0u64;
        for (i, p) in few.chunks_exact(3).zip(o2.chunks_exact(4)) { for c in 0..3 { st.add(p[c] as f64, x.unit(c, [i[0] as usize, i[1] as usize, i[2] as usize]) * 255.0); } if p[3] != 255 { alpha_bad += 1 } }
        st.show(&format!("{name} Rgb8->Rgba8 (F32 route), alpha!=255: {alpha_bad}"));
        let n = wide.len() / 3;
        let mut ow = vec![0u16; wide.len()];
        MoxCms.build_source_transform(s.clone(), t.clone(), F::Rgb16, F::Rgb16, &opts).unwrap().unwrap().try_transform_row(bytemuck::cast_slice(&wide), bytemuck::cast_slice_mut(&mut ow), n as u32).unwrap();
        let mut st = Stat::default();
        for (i, p) in wide.chunks_exact(3).zip(ow.chunks_exact(3)) { for c in 0..3 { st.add(p[c] as f64, x16.unit(c, [i[0] as usize, i[1] as usize, i[2] as usize]) * 65535.0); } }
        st.show(&format!("{name} Rgb16->Rgb16, in 16-bit codes"));
        let inf: Vec<f32> = wide.iter().map(|&v| v as f32 / 65535.0).collect();
        let mut of = vec![0f32; inf.len()];
        MoxCms.build_source_transform(s.clone(), t.clone(), F::RgbF32, F::RgbF32, &opts).unwrap().unwrap().try_transform_row(bytemuck::cast_slice(&inf), bytemuck::cast_slice_mut(&mut of), n as u32).unwrap();
        let mut st = Stat::default();
        for (i, p) in wide.chunks_exact(3).zip(of.chunks_exact(3)) { for c in 0..3 { let (i0, i1, i2) = (i[0] as usize, i[1] as usize, i[2] as usize); st.add(p[c].clamp(0.0, 1.0) as f64 * 65535.0, x16.unit(c, [i0, i1, i2]) * 65535.0); } }
        st.show(&format!("{name} RgbF32, unrounded, in 16-bit codes"));
        // Where F32 is furthest from exact, and which is out of gamut.
        let (mut wv, mut wi, mut wc) = (0f64, 0usize, 0usize);
        let (mut neg_n, mut neg_abs, mut inside_abs, mut inside_n, mut inside_w) = (0u64, 0f64, 0f64, 0u64, 0f64);
        for (k, (i, p)) in wide.chunks_exact(3).zip(of.chunks_exact(3)).enumerate() {
            for c in 0..3 {
                let idx = [i[0] as usize, i[1] as usize, i[2] as usize];
                let e = x16.unit(c, idx);
                let d = (p[c].clamp(0.0, 1.0) as f64 - e) * 65535.0;
                if d.abs() > wv { wv = d.abs(); wi = k; wc = c; }
                let lin = x16.m[c][0] * x16.lin[idx[0]] + x16.m[c][1] * x16.lin[idx[1]] + x16.m[c][2] * x16.lin[idx[2]];
                if lin < 0.0 || lin > 1.0 { neg_n += 1; neg_abs += d.abs(); } else { inside_n += 1; inside_abs += d.abs(); inside_w = inside_w.max(d.abs()); }
            }
        }
        let i = &wide[wi * 3..wi * 3 + 3];
        let idx = [i[0] as usize, i[1] as usize, i[2] as usize];
        let lin = x16.m[wc][0] * x16.lin[idx[0]] + x16.m[wc][1] * x16.lin[idx[1]] + x16.m[wc][2] * x16.lin[idx[2]];
        println!("    F32 worst: input {i:?} channel {wc}: F32 raw {:?}, exact linear {lin:.6}, exact code {:.1}; out of range {neg_n} samples mean|d| {:.2}; in range {inside_n} samples mean|d| {:.3} worst {:.2}", &of[wi * 3..wi * 3 + 3], x16.unit(wc, idx) * 65535.0, neg_abs / neg_n.max(1) as f64, inside_abs / inside_n.max(1) as f64, inside_w);
    }
}

// ---- exact table evaluation ----
fn tag<'a>(icc: &'a [u8], sig: &[u8; 4]) -> &'a [u8] {
    let n = u32::from_be_bytes(icc[128..132].try_into().unwrap()) as usize;
    for i in 0..n {
        let e = &icc[132 + 12 * i..144 + 12 * i];
        if &e[..4] == sig { let (o, l) = (u32::from_be_bytes(e[4..8].try_into().unwrap()) as usize, u32::from_be_bytes(e[8..12].try_into().unwrap()) as usize); return &icc[o..o + l]; }
    }
    panic!("no tag")
}
struct Lut { trilinear: std::cell::Cell<bool>, ic: usize, g: usize, n_in: usize, n_out: usize, inp: Vec<f64>, clut: Vec<f64>, out: Vec<f64>, sixteen: bool }
impl Lut {
    fn parse(t: &[u8]) -> Self {
        let sixteen = &t[..4] == b"mft2";
        let (ic, oc, g) = (t[8] as usize, t[9] as usize, t[10] as usize);
        assert_eq!(oc, 3);
        let (n_in, n_out, start, sz) = if sixteen { (u16::from_be_bytes([t[48], t[49]]) as usize, u16::from_be_bytes([t[50], t[51]]) as usize, 52, 2) } else { (256, 256, 48, 1) };
        let rd = |o: usize, n: usize| -> Vec<f64> { (0..n).map(|i| if sz == 2 { u16::from_be_bytes([t[o + i * 2], t[o + i * 2 + 1]]) as f64 / 65535.0 } else { t[o + i] as f64 / 255.0 }).collect() };
        let nodes = g.pow(ic as u32) * 3;
        let inp = rd(start, n_in * ic);
        let clut = rd(start + n_in * ic * sz, nodes);
        let out = rd(start + (n_in * ic + nodes) * sz, n_out * 3);
        Lut { trilinear: std::cell::Cell::new(false), ic, g, n_in, n_out, inp, clut, out, sixteen }
    }
    fn c1(t: &[f64], x: f64) -> f64 { let f = x.clamp(0.0, 1.0) * (t.len() - 1) as f64; let i = (f as usize).min(t.len() - 2); t[i] + (t[i + 1] - t[i]) * (f - i as f64) }
    fn tetra(&self, base: usize, stride: [usize; 3], f: [f64; 3], o: usize) -> f64 {
        let n = |a: usize, b: usize, c: usize| self.clut[(base + a * stride[0] + b * stride[1] + c * stride[2]) * 3 + o];
        let (rx, ry, rz) = (f[0], f[1], f[2]);
        let c0 = n(0, 0, 0); let c111 = n(1, 1, 1);
        let (c1, c2, c3);
        if rx >= ry && ry >= rz { c1 = n(1,0,0) - c0; c2 = n(1,1,0) - n(1,0,0); c3 = c111 - n(1,1,0); }
        else if rx >= rz && rz >= ry { c1 = n(1,0,0) - c0; c2 = c111 - n(1,0,1); c3 = n(1,0,1) - n(1,0,0); }
        else if rz >= rx && rx >= ry { c1 = n(1,0,1) - n(0,0,1); c2 = c111 - n(1,0,1); c3 = n(0,0,1) - c0; }
        else if ry >= rx && rx >= rz { c1 = n(1,1,0) - n(0,1,0); c2 = n(0,1,0) - c0; c3 = c111 - n(1,1,0); }
        else if ry >= rz && rz >= rx { c1 = c111 - n(0,1,1); c2 = n(0,1,0) - c0; c3 = n(0,1,1) - n(0,1,0); }
        else { c1 = c111 - n(0,1,1); c2 = n(0,1,1) - n(0,0,1); c3 = n(0,0,1) - c0; }
        c0 + c1 * rx + c2 * ry + c3 * rz
    }
    /// Lab, as lcms2 does: 3D tetrahedral; 4D linear on the first input over tetrahedral on the rest.
    fn lab(&self, input: &[f64]) -> [f64; 3] {
        let g = self.g;
        let v: Vec<f64> = (0..self.ic).map(|c| Self::c1(&self.inp[c * self.n_in..(c + 1) * self.n_in], input[c]) * (g - 1) as f64).collect();
        let i: Vec<usize> = v.iter().map(|x| (*x as usize).min(g - 2)).collect();
        let fr: Vec<f64> = v.iter().zip(&i).map(|(x, i)| x - *i as f64).collect();
        let mut o = [0.0; 3];
        for ch in 0..3 {
            let raw = if self.trilinear.get() {
                let mut sum = 0.0;
                for corner in 0..1usize << self.ic {
                    let mut idx = 0usize;
                    let mut w = 1.0;
                    for a in 0..self.ic {
                        let up = corner >> a & 1;
                        idx = idx * g + i[a] + up;
                        w *= if up == 1 { fr[a] } else { 1.0 - fr[a] };
                    }
                    sum += w * self.clut[idx * 3 + ch];
                }
                sum
            } else if self.ic == 3 {
                self.tetra((i[0] * g + i[1]) * g + i[2], [g * g, g, 1], [fr[0], fr[1], fr[2]], ch)
            } else {
                let b = |k: usize| ((((i[0] + k) * g + i[1]) * g + i[2]) * g) + i[3];
                let a = self.tetra(b(0), [g * g, g, 1], [fr[1], fr[2], fr[3]], ch);
                let c = self.tetra(b(1), [g * g, g, 1], [fr[1], fr[2], fr[3]], ch);
                a + (c - a) * fr[0]
            };
            o[ch] = Self::c1(&self.out[ch * self.n_out..(ch + 1) * self.n_out], raw);
        }
        // Lab encodings: mft1 is L = v*100, a = v*255-128 in every version; mft2 in v2 is the legacy 65280 white.
        if self.sixteen { let k = 65535.0 / 65280.0; [o[0] * k * 100.0, o[1] * k * 255.0 - 128.0, o[2] * k * 255.0 - 128.0] } else { [o[0] * 100.0, o[1] * 255.0 - 128.0, o[2] * 255.0 - 128.0] }
    }
}
const PCS: [f64; 3] = [0.9642, 1.0, 0.8249];
fn lab_to_xyz(l: [f64; 3]) -> [f64; 3] {
    let fy = (l[0] + 16.0) / 116.0;
    let finv = |t: f64| if t > 6.0 / 29.0 { t * t * t } else { 3.0 * (6.0f64 / 29.0).powi(2) * (t - 4.0 / 29.0) };
    [PCS[0] * finv(fy + l[1] / 500.0), PCS[1] * finv(fy), PCS[2] * finv(fy - l[2] / 200.0)]
}
fn c_float(icc: &[u8], space: u32, ch: u32, intent: u32, input: &[f32], out: &mut [f32], n: u32) {
    const fn fmt(s: u32, c: u32) -> u32 { 1 << 22 | s << 16 | c << 3 | 4 }
    unsafe {
        let p = c::cmsOpenProfileFromMem(icc.as_ptr() as *const c_void, icc.len() as u32);
        let s = c::cmsCreate_sRGBProfile();
        let t = c::cmsCreateTransform(p, fmt(space, ch), s, fmt(4, 3), intent, 0x100);
        assert!(!t.is_null());
        c::cmsDoTransform(t, input.as_ptr() as *const c_void, out.as_mut_ptr() as *mut c_void, n);
        c::cmsDeleteTransform(t);
    }
}
fn table_cases(dir: &str) {
    let srgb = ColorProfile::new_srgb();
    let _ = &srgb;
    let srgb_w = adapter_srgb();
    let to_dst = srgb_w.colorant_matrix().inverse().v;
    let enc = |v: f64| { let v = v.clamp(0.0, 1.0); if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 } };
    let srgb_src = S::Cicp(Cicp::new(1, 13, 0, true));
    let opts = ConvertOptions::permissive();
    let mut h = 0x1234_5678u32;
    let mut next = || { h ^= h << 13; h ^= h >> 17; h ^= h << 5; h };
    for (file, sig, ch, from, fmt_space, intent_c, intent_m) in [("test3.icc", b"A2B0", 3usize, F::Rgb8, 4u32, 1u32, RenderingIntent::Perceptual), ("test1.icc", b"A2B1", 4usize, F::Cmyk8, 6, 1, RenderingIntent::RelativeColorimetric)] {
        let icc = std::fs::read(format!("{dir}/{file}")).unwrap();
        let t = tag(&icc, sig);
        let lut = Lut::parse(t);
        println!("{file}: {} grid {} inputs {} tables in {} out {}", String::from_utf8_lossy(&t[..4]), lut.g, lut.ic, lut.n_in, lut.n_out);
        let n = 1usize << 21;
        let input: Vec<u8> = (0..n * ch).map(|_| (next() >> 24) as u8).collect();
        let exact: Vec<f64> = input.chunks_exact(ch).flat_map(|p| {
            let v: Vec<f64> = p.iter().map(|&b| b as f64 / 255.0).collect();
            let xyz = lab_to_xyz(lut.lab(&v));
            [0, 1, 2].map(|i| enc(to_dst[i][0] * xyz[0] + to_dst[i][1] * xyz[1] + to_dst[i][2] * xyz[2]) * 255.0)
        }).collect();
        let row = |name: &str, got: &[f32]| { let mut st = Stat::default(); for (g, e) in got.iter().zip(&exact) { st.add(g.clamp(0.0, 1.0) as f64 * 255.0, *e); } st.show(name); };
        let mut o = vec![0u8; n * 3];
        MoxCms.build_source_transform(S::Icc(&icc), srgb_src.clone(), from, F::Rgb8, &opts).unwrap().unwrap().try_transform_row(&input, &mut o, n as u32).unwrap();
        let mut st = Stat::default();
        for (g, e) in o.iter().zip(&exact) { st.add(*g as f64, *e); }
        st.show(&format!("{file} MoxCms U8 (F32 route) vs exact table math"));
        let scale = if ch == 4 { 100.0 } else { 1.0 };
        let inc: Vec<f32> = input.iter().map(|&v| v as f32 / 255.0 * scale).collect();
        let mut rc = vec![0f32; n * 3];
        c_float(&icc, fmt_space, ch as u32, intent_c, &inc, &mut rc, n as u32);
        row(&format!("{file} C lcms2 F32 NOOPT vs exact table math"), &rc);
        let sp = ColorProfile::new_from_slice(&icc).unwrap();
        let inf: Vec<f32> = input.iter().map(|&v| v as f32 / 255.0).collect();
        let lay = if ch == 4 { Layout::Rgba } else { Layout::Rgb };
        for (skip, ext) in [(false, true), (true, true), (false, false), (true, false)] {
            moxcms::BISECT_SKIP_LAB_V2.store(skip, Relaxed);
            let o = TransformOptions { rendering_intent: intent_m, allow_use_cicp_transfer: false, barycentric_weight_scale: BarycentricWeightScale::High, interpolation_method: InterpolationMethod::Tetrahedral, prefer_fixed_point: false, allow_extended_range_rgb_xyz: ext, ..Default::default() };
            let tr = sp.create_transform_f32(lay, &srgb_w, Layout::Rgb, o).unwrap();
            let mut got = vec![0f32; n * 3];
            tr.transform(&inf, &mut got).unwrap();
            row(&format!("{file} moxcms F32 skip-v2-scale={skip} extended={ext} vs exact"), &got);
        }
        moxcms::BISECT_SKIP_LAB_V2.store(false, Relaxed);
        lut.trilinear.set(true);
        let tri: Vec<f64> = input.chunks_exact(ch).flat_map(|p| {
            let v: Vec<f64> = p.iter().map(|&b| b as f64 / 255.0).collect();
            let xyz = lab_to_xyz(lut.lab(&v));
            [0, 1, 2].map(|i| enc(to_dst[i][0] * xyz[0] + to_dst[i][1] * xyz[1] + to_dst[i][2] * xyz[2]) * 255.0)
        }).collect();
        lut.trilinear.set(false);
        for (skip, label) in [(false, "as released"), (true, "skip-v2-scale")] {
            moxcms::BISECT_SKIP_LAB_V2.store(skip, Relaxed);
            let o = TransformOptions { rendering_intent: intent_m, allow_use_cicp_transfer: false, barycentric_weight_scale: BarycentricWeightScale::High, interpolation_method: InterpolationMethod::Tetrahedral, prefer_fixed_point: false, allow_extended_range_rgb_xyz: true, ..Default::default() };
            let tr = sp.create_transform_f32(lay, &srgb_w, Layout::Rgb, o).unwrap();
            let mut got = vec![0f32; n * 3];
            tr.transform(&inf, &mut got).unwrap();
            let mut st = Stat::default();
            for (g, e) in got.iter().zip(&tri) { st.add(g.clamp(0.0, 1.0) as f64 * 255.0, *e); }
            st.show(&format!("{file} moxcms F32 {label} vs exact TRILINEAR table math"));
        }
        moxcms::BISECT_SKIP_LAB_V2.store(false, Relaxed);
    }
}
fn main() {
    let dir = std::env::args().nth(1).unwrap();
    table_cases(&dir);
    matrix_routes(&dir);
}
