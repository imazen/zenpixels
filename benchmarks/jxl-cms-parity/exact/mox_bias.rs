// moxcms U8 bias bisect. Reference is f64 math on the profiles' own curves and colorants.
use moxcms::*;
use std::sync::atomic::Ordering::Relaxed;

#[derive(Clone)]
enum C { Para(Vec<f64>), Gamma(f64), Lut(Vec<f64>) }
impl C {
    fn of(t: &ToneReprCurve) -> C {
        match t {
            ToneReprCurve::Parametric(p) => C::Para(p.iter().map(|&v| v as f64).collect()),
            ToneReprCurve::Lut(d) if d.len() == 1 => C::Gamma(d[0] as f64 / 256.0),
            ToneReprCurve::Lut(d) if d.is_empty() => C::Gamma(1.0),
            ToneReprCurve::Lut(d) => C::Lut(d.iter().map(|&v| v as f64 / 65535.0).collect()),
        }
    }
    fn dec(&self, x: f64) -> f64 {
        match self {
            C::Gamma(g) => x.powf(*g),
            C::Para(p) => match p.len() {
                1 => x.powf(p[0]),
                5 => if x >= p[4] { (p[1] * x + p[2]).powf(p[0]) } else { p[3] * x },
                n => panic!("para {n}"),
            },
            C::Lut(t) => { let f = x * (t.len() - 1) as f64; let i = (f as usize).min(t.len() - 2); t[i] + (t[i + 1] - t[i]) * (f - i as f64) }
        }
    }
    fn enc(&self, y: f64) -> f64 {
        match self {
            C::Gamma(g) => y.powf(1.0 / g),
            C::Para(p) => match p.len() {
                1 => y.powf(1.0 / p[0]),
                5 => { let k = p[3] * p[4]; if y >= k { (y.powf(1.0 / p[0]) - p[2]) / p[1] } else { y / p[3] } }
                n => panic!("para {n}"),
            },
            C::Lut(t) => { let i = t.partition_point(|&v| v < y).clamp(1, t.len() - 1); ((i - 1) as f64 + (y - t[i - 1]) / (t[i] - t[i - 1]).max(1e-12)) / (t.len() - 1) as f64 }
        }
    }
}
fn opts(fixed: bool) -> TransformOptions {
    TransformOptions { rendering_intent: RenderingIntent::RelativeColorimetric, allow_use_cicp_transfer: false,
        barycentric_weight_scale: BarycentricWeightScale::Low, interpolation_method: InterpolationMethod::Tetrahedral, prefer_fixed_point: fixed, ..Default::default() }
}
fn pcs_white(mut p: ColorProfile) -> ColorProfile {
    let (r, g, b) = (p.red_colorant, p.green_colorant, p.blue_colorant);
    let s = [0.9642 / (r.x + g.x + b.x), 1.0 / (r.y + g.y + b.y), 0.8249 / (r.z + g.z + b.z)];
    for c in [&mut p.red_colorant, &mut p.green_colorant, &mut p.blue_colorant] { c.x *= s[0]; c.y *= s[1]; c.z *= s[2]; }
    p
}
struct Exact { m: [[f64; 3]; 3], lin: Vec<f64>, enc: C, depth: u32 }
impl Exact {
    fn new(s: &ColorProfile, t: &ColorProfile, depth: u32) -> Self {
        let m = t.colorant_matrix().inverse().mat_mul(s.colorant_matrix()).v;
        let dec = C::of(s.red_trc.as_ref().unwrap());
        let n = 1usize << depth;
        let lin = (0..n).map(|i| dec.dec(i as f64 / (n - 1) as f64)).collect();
        Exact { m, lin, enc: C::of(t.red_trc.as_ref().unwrap()), depth }
    }
    fn code(&self, ch: usize, r: usize, g: usize, b: usize) -> f64 {
        let v = self.m[ch][0] * self.lin[r] + self.m[ch][1] * self.lin[g] + self.m[ch][2] * self.lin[b];
        self.enc.enc(v.clamp(0.0, 1.0)) * ((1u32 << self.depth) - 1) as f64
    }
}
#[derive(Default, Clone, Copy)]
struct Stat { n: u64, sum: f64, wrong: u64, worst: f64, hi: u64, lo: u64, ch: [f64; 3] }
impl Stat {
    fn add(&mut self, ch: usize, got: f64, e: f64) {
        self.n += 1; let d = got - e; self.sum += d; self.ch[ch] += d; if d.abs() > self.worst { self.worst = d.abs() }
        if got != e.round() { self.wrong += 1; if got < e.round() { self.lo += 1 } else { self.hi += 1 } }
    }
    fn show(&self, name: &str) {
        let k = 3.0 / self.n as f64;
        println!("{name:<52} mean {:+.4}  (R {:+.4} G {:+.4} B {:+.4})  not-nearest {:6.2}% (low {:.2}% high {:.2}%)  worst {:.3}  n={}",
            self.sum / self.n as f64, self.ch[0] * k, self.ch[1] * k, self.ch[2] * k, 100.0 * self.wrong as f64 / self.n as f64,
            100.0 * self.lo as f64 / self.n as f64, 100.0 * self.hi as f64 / self.n as f64, self.worst, self.n);
    }
}
fn cube() -> Vec<u8> { let mut v = Vec::with_capacity(3 << 24); for r in 0..=255u8 { for g in 0..=255u8 { for b in 0..=255u8 { v.extend_from_slice(&[r, g, b]); } } } v }

fn run_u8(name: &str, s: &ColorProfile, t: &ColorProfile, fixed: bool, input: &[u8]) -> Stat {
    let x = Exact::new(s, t, 8);
    let tr = s.create_transform_8bit(Layout::Rgb, t, Layout::Rgb, opts(fixed)).unwrap();
    let mut o = vec![0u8; input.len()];
    tr.transform(input, &mut o).unwrap();
    let mut st = Stat::default();
    for (i, p) in input.chunks_exact(3).zip(o.chunks_exact(3)) {
        for c in 0..3 { st.add(c, p[c] as f64, x.code(c, i[0] as usize, i[1] as usize, i[2] as usize)); }
    }
    st.show(name);
    st
}

/// Scalar model of moxcms's fixed-point executor, with each stage swappable. Returns whether baseline matched the library.
fn model(s: &ColorProfile, t: &ColorProfile, input: &[u8], lib: Option<&[u8]>) {
    let exact = Exact::new(s, t, 8);
    let lin32 = s.build_8bit_lin_table(&s.red_trc).unwrap();
    let m = t.colorant_matrix().inverse().mat_mul(s.colorant_matrix()).v;
    let g_lib: Box<[u8; 65536]> = t.build_gamma_table::<u8, 65536, 4096, 8>(&t.red_trc, false).unwrap();
    let enc = C::of(t.red_trc.as_ref().unwrap());
    let g_round: Vec<u8> = (0..4096).map(|i| (enc.enc(i as f64 / 4095.0) * 255.0).round() as u8).collect();
    let diff = (0..4096).filter(|&i| g_lib[i] != g_round[i]).count();
    let mean: f64 = (0..4096).map(|i| g_lib[i] as f64 - enc.enc(i as f64 / 4095.0) * 255.0).sum::<f64>() / 4096.0;
    let mean_r: f64 = (0..4096).map(|i| g_round[i] as f64 - enc.enc(i as f64 / 4095.0) * 255.0).sum::<f64>() / 4096.0;
    println!("  gamma table: {diff} of 4096 entries differ from nearest rounding; mean(entry - exact) lib {mean:+.4}, rounded {mean_r:+.4}");
    let l_q: Vec<i32> = lin32.iter().map(|&x| (x * 4095.0).round() as i32).collect();
    let mq: [[i32; 3]; 3] = m.map(|r| r.map(|v| ((v as f32) * 8192.0) as i16 as i32));
    let mq_r: [[i32; 3]; 3] = m.map(|r| r.map(|v| ((v as f32) * 8192.0).round() as i16 as i32));
    let row_sums: Vec<i32> = mq.iter().map(|r| r.iter().sum()).collect();
    let row_sums_r: Vec<i32> = mq_r.iter().map(|r| r.iter().sum()).collect();
    println!("  matrix Q13 row sums: truncated {row_sums:?}, rounded {row_sums_r:?}; 8192 = 1.0; f64 {:?}", m.map(|r| r.iter().sum::<f64>()));
    let mut base = Stat::default();
    let mut gr = Stat::default();
    let mut mr = Stat::default();
    let mut both = Stat::default();
    let mut ideal_idx = Stat::default();
    let mut mism = 0u64;
    for (k, i) in input.chunks_exact(3).enumerate() {
        let (r, g, b) = (i[0] as usize, i[1] as usize, i[2] as usize);
        for c in 0..3 {
            let e = exact.code(c, r, g, b);
            let idx = |q: &[[i32; 3]; 3]| ((l_q[r] * q[c][0] + l_q[g] * q[c][1] + l_q[b] * q[c][2] + 4096) >> 13).clamp(0, 4095) as usize;
            let a = g_lib[idx(&mq)];
            if let Some(l) = lib { if l[k * 3 + c] != a { mism += 1; } }
            base.add(c, a as f64, e);
            gr.add(c, g_round[idx(&mq)] as f64, e);
            mr.add(c, g_lib[idx(&mq_r)] as f64, e);
            both.add(c, g_round[idx(&mq_r)] as f64, e);
            // Exact linear value, table quantized only at the 4096-entry index.
            let v = exact.m[c][0] * exact.lin[r] + exact.m[c][1] * exact.lin[g] + exact.m[c][2] * exact.lin[b];
            let id = (v.clamp(0.0, 1.0) * 4095.0).round() as usize;
            ideal_idx.add(c, g_round[id] as f64, e);
        }
    }
    println!("  model vs library bytes: {mism} mismatches");
    base.show("  model baseline (Q12 lin, Q13 trunc, lib gamma)");
    gr.show("  model + nearest gamma entries");
    mr.show("  model + rounded matrix");
    both.show("  model + nearest gamma + rounded matrix");
    ideal_idx.show("  exact lin+matrix, 4096-entry nearest gamma only");
}

fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let f = |n: &str| ColorProfile::new_from_slice(&std::fs::read(format!("{dir}/{n}")).unwrap()).unwrap();
    let srgb_gen = ColorProfile::new_srgb();
    let srgb = pcs_white(srgb_gen.clone());
    let p3 = f("DisplayP3Compat-v4.icc");
    let adobe = f("AdobeCompat-v2.icc");
    let bt2020 = f("Rec2020Compat-v4.icc");
    let p3lut = f("DisplayP3Compat-v2-magic.icc");
    let input = cube();
    println!("moxcms 0.9.1 scratch copy, exhaustive 2^24 RGB");
    println!("== stage toggles in the library (P3 v4 -> sRGB generated, PCS white scaled) ==");
    let base = { let tr = p3.create_transform_8bit(Layout::Rgb, &srgb, Layout::Rgb, opts(true)).unwrap(); let mut o = vec![0u8; input.len()]; tr.transform(&input, &mut o).unwrap(); o };
    run_u8("default (fixed-point Q2.13 executor)", &p3, &srgb, true, &input);
    run_u8("prefer_fixed_point=false (float matrix, u8 gamma)", &p3, &srgb, false, &input);
    run_u8("default, target without PCS white scaling (Z .82521)", &p3, &srgb_gen, true, &input);
    moxcms::BISECT_ROUND_MATRIX.store(true, Relaxed);
    run_u8("+ matrix round in to_q2_13_i (rgbxyz.rs:225)", &p3, &srgb, true, &input);
    moxcms::BISECT_ROUND_MATRIX.store(false, Relaxed);
    moxcms::BISECT_ROUND_GAMMA.store(true, Relaxed);
    run_u8("+ gamma .round() in make_gamma_table_impl (gamma.rs:654)", &p3, &srgb, true, &input);
    run_u8("+ gamma .round(), prefer_fixed_point=false", &p3, &srgb, false, &input);
    moxcms::BISECT_ROUND_MATRIX.store(true, Relaxed);
    run_u8("+ gamma .round() + matrix round", &p3, &srgb, true, &input);
    moxcms::BISECT_ROUND_MATRIX.store(false, Relaxed);
    moxcms::BISECT_ROUND_GAMMA.store(false, Relaxed);
    println!("== scalar model, one stage swapped at a time ==");
    model(&p3, &srgb, &input, Some(&base));
    println!("== which targets carry it (default executor) ==");
    run_u8("P3 v4 -> sRGB generated (para, sRGB params)", &p3, &srgb, true, &input);
    run_u8("sRGB generated -> P3 v4 (para, sRGB params)", &srgb, &p3, true, &input);
    run_u8("P3 v4 -> Adobe (curv gamma, make_gamma_pow_table)", &p3, &adobe, true, &input);
    run_u8("P3 v4 -> BT.2020 v4 (para, rec709-like params)", &p3, &bt2020, true, &input);
    run_u8("Adobe -> P3 v2 magic (curv LUT, make_gamma_lut)", &adobe, &p3lut, true, &input);
    drop(input);
    lut_and_u16(&dir);
    // Sign of an out-of-gamut sample through the pure-gamma extended evaluator, and sRGB's toe.
    let o = TransformOptions { prefer_fixed_point: false, allow_extended_range_rgb_xyz: true, ..opts(false) };
    let t = p3.create_transform_f32(Layout::Rgb, &adobe, Layout::Rgb, o).unwrap();
    let mut out = [0f32; 3];
    t.transform(&[247.0 / 255.0, 1.0, 0.0], &mut out).unwrap();
    println!("F32 extended P3 -> Adobe of (247,255,0)/255: {out:?}; exact blue linear is negative");
}

pub fn lut_and_u16(dir: &str) {
    let f = |n: &str| ColorProfile::new_from_slice(&std::fs::read(format!("{dir}/{n}")).unwrap()).unwrap();
    let srgb = pcs_white(ColorProfile::new_srgb());
    let p3 = f("DisplayP3Compat-v4.icc");
    let adobe = f("AdobeCompat-v2.icc");
    // Generated targets.
    let cube = cube();
    moxcms::BISECT_ROUND_GAMMA.store(false, Relaxed);
    println!("== generated targets, default executor ==");
    run_u8("P3 v4 -> generated Display P3", &adobe, &pcs_white(ColorProfile::new_display_p3()), true, &cube);
    run_u8("Adobe -> generated BT.2020 (Bt709 TC)", &adobe, &pcs_white(ColorProfile::new_bt2020()), true, &cube);
    run_u8("Adobe -> generated Adobe RGB", &p3, &pcs_white(ColorProfile::new_adobe_rgb()), true, &cube);
    let n = 1usize << 21;
    let mut h = 0x9E37_79B9u32;
    let mut next = || { h ^= h << 13; h ^= h >> 17; h ^= h << 5; h };
    let rgb8: Vec<u8> = (0..n * 3).map(|_| (next() >> 24) as u8).collect();
    let cmyk8: Vec<u8> = (0..n * 4).map(|_| (next() >> 24) as u8).collect();
    let rgb16: Vec<u16> = (0..n * 3).map(|_| (next() >> 16) as u16).collect();
    println!("== LUT paths: 8-bit against moxcms's own F32, same options but float ==");
    let lut_case = |name: &str, s: &ColorProfile, lay: Layout, intent: RenderingIntent, input: &[u8], fixed: bool| {
        for round in [false, true] {
            moxcms::BISECT_ROUND_GAMMA.store(round, Relaxed);
            let mut o8 = opts(fixed); o8.rendering_intent = intent;
            let mut of = opts(false); of.rendering_intent = intent; of.barycentric_weight_scale = BarycentricWeightScale::High; of.allow_extended_range_rgb_xyz = false;
            let t8 = s.create_transform_8bit(lay, &srgb, Layout::Rgb, o8).unwrap();
            let tf = s.create_transform_f32(lay, &srgb, Layout::Rgb, of).unwrap();
            let px = input.len() / lay.channels();
            let inf: Vec<f32> = input.iter().map(|&v| v as f32 / 255.0).collect();
            let mut a = vec![0u8; px * 3]; let mut b = vec![0f32; px * 3];
            t8.transform(input, &mut a).unwrap(); tf.transform(&inf, &mut b).unwrap();
            let mut st = Stat::default();
            for (i, (x, y)) in a.iter().zip(&b).enumerate() { st.add(i % 3, *x as f64, (*y as f64).clamp(0.0, 1.0) * 255.0); }
            st.show(&format!("{name} gamma-round={round} fixed={fixed}"));
        }
    };
    for fixed in [true, false] {
        lut_case("test3 RGB mft1 Lab", &f("test3.icc"), Layout::Rgb, RenderingIntent::Perceptual, &rgb8, fixed);
        lut_case("test1 CMYK mft2 Lab", &f("test1.icc"), Layout::Rgba, RenderingIntent::RelativeColorimetric, &cmyk8, fixed);
    }
    println!("== U16 matrix path against f64, in 16-bit codes ==");
    for round in [false, true] {
        moxcms::BISECT_ROUND_GAMMA.store(round, Relaxed);
        for (name, s, t) in [("P3 -> sRGB", &p3, &srgb), ("sRGB -> P3", &srgb, &p3), ("P3 -> Adobe", &p3, &adobe)] {
            let x = Exact::new(s, t, 16);
            let tr = s.create_transform_16bit(Layout::Rgb, t, Layout::Rgb, opts(true)).unwrap();
            let mut o = vec![0u16; rgb16.len()];
            tr.transform(&rgb16, &mut o).unwrap();
            let mut st = Stat::default();
            for (i, p) in rgb16.chunks_exact(3).zip(o.chunks_exact(3)) { for c in 0..3 { st.add(c, p[c] as f64, x.code(c, i[0] as usize, i[1] as usize, i[2] as usize)); } }
            st.show(&format!("{name} gamma-round={round}"));
        }
    }
    moxcms::BISECT_ROUND_GAMMA.store(false, Relaxed);
}
