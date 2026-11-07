// Before/after through the public MoxCms adapter, comparing output bits on every route class.
use moxcms::{ColorProfile as P, ToneReprCurve as T};
#[derive(Clone)]
enum S { Icc(Vec<u8>), Cicp(u8, u8) }
macro_rules! conv { ($z:ident, $s:expr) => { match $s { S::Icc(i) => $z::ColorProfileSource::Icc(i.as_slice()), S::Cicp(p, t) => $z::ColorProfileSource::Cicp($z::Cicp::new(*p, *t, 0, true)) } } }
macro_rules! run { ($z:ident, $zc:ident, $s:expr, $t:expr, $from:ident, $to:ident, $clip:expr, $input:expr, $w:expr, $bytes:expr) => {{
    use $zc::cms::PluggableCms;
    let o = $zc::ConvertOptions::retain_channels().with_clip_out_of_gamut($clip);
    match $zc::cms_moxcms::MoxCms.build_source_transform(conv!($z, $s), conv!($z, $t), $z::PixelFormat::$from, $z::PixelFormat::$to, &o) {
        None => Err("declined".to_string()),
        Some(Err(e)) => Err(format!("refused: {}", format!("{e:?}").chars().take(60).collect::<String>())),
        Some(Ok(mut tr)) => { let mut d = vec![0u8; $bytes]; tr.try_transform_row($input, &mut d, $w as u32).map_err(|e| format!("{e:?}"))?; Ok(d) }
    }
}}}
type R = Result<Vec<u8>, String>;
fn f32grid() -> Vec<f32> {
    let mut v = Vec::new();
    for r in (0..=255).step_by(5) { for g in (0..=255).step_by(5) { for b in (0..=255).step_by(5) { v.extend_from_slice(&[r as f32 / 255., g as f32 / 255., b as f32 / 255.]); } } }
    let sp = [0f32, -0.0, 1e-30, 1e-20, 1e-12, 1e-6, 1e-3, 0.5, 1.0, 1.5, 8.0, -1e-6, -0.051, -0.5, -1.5, f32::MIN_POSITIVE, f32::from_bits(1)];
    for &a in &sp { for &b in &sp { for &c in &sp { v.extend_from_slice(&[a, b, c]); } } }
    v
}
fn u8grid() -> Vec<u8> { let mut v = Vec::new(); for r in (0..=255u8).step_by(5) { for g in (0..=255u8).step_by(5) { for b in (0..=255u8).step_by(5) { v.extend_from_slice(&[r, g, b]); } } } v }
fn u8full() -> Vec<u8> { let mut v = Vec::with_capacity(3 << 24); for r in 0..=255u8 { for g in 0..=255u8 { for b in 0..=255u8 { v.extend_from_slice(&[r, g, b]); } } } v }
fn report(name: &str, b: R, a: R, n: usize, sample: usize) -> (usize, usize) {
    match (&b, &a) {
        (Ok(x), Ok(y)) => { let d = x.chunks_exact(sample).zip(y.chunks_exact(sample)).filter(|(p, q)| p != q).count(); println!("{name:<54} {n:>9} samples, bits differ {d}"); (n, d) }
        (Err(x), Err(y)) if x == y => { println!("{name:<54} both refuse: {x}"); (0, 0) }
        _ => { println!("{name:<54} DIFFERENT OUTCOME before {:?} after {:?}", b.as_ref().map(|v| v.len()), a.as_ref().map(|v| v.len())); (0, 1) }
    }
}
fn main() {
    let icc = |p: P| S::Icc({ let mut p = p; p.cicp = None; p.encode().unwrap() });
    let gray = |g: f32| S::Icc(P::new_gray_with_gamma(g).encode().unwrap());
    let lut = S::Icc(std::fs::read(std::env::var("LUT_ICC").unwrap()).unwrap());
    let (pp, ad, srgb, p3, bt2020) = (icc(P::new_pro_photo_rgb()), icc(P::new_adobe_rgb()), icc(P::new_srgb()), icc(P::new_display_p3()), icc(P::new_bt2020()));
    let (g18, g22) = (gray(1.8), gray(2.2));
    let (csrgb, cg22, cg28, c428, cpq) = (S::Cicp(1, 13), S::Cicp(1, 4), S::Cicp(1, 5), S::Cicp(1, 17), S::Cicp(9, 16));
    let tabled = { let mut p = P::new_srgb(); p.cicp = None; let c = T::Lut((0..1024).map(|i| ((i as f64 / 1023.).powf(1.9) * 65535.) as u16).collect()); p.red_trc = Some(c.clone()); p.green_trc = Some(c.clone()); p.blue_trc = Some(c); S::Icc(p.encode().unwrap()) };
    let cmyk = S::Icc(std::fs::read(std::env::var("CMYK_ICC").unwrap()).unwrap());
    let g = f32grid(); let gb: &[u8] = bytemuck::cast_slice(&g); let w = g.len() / 3;
    let gray_in: Vec<f32> = g.iter().step_by(3).copied().collect(); let gray_b: &[u8] = bytemuck::cast_slice(&gray_in);
    let rgba: Vec<f32> = g.chunks_exact(3).flat_map(|c| [c[0], c[1], c[2], 0.75]).collect(); let rgba_b: &[u8] = bytemuck::cast_slice(&rgba);
    let (mut must_n, mut must_d) = (0, 0);
    println!("=== must stay bit-identical: F32 routes outside the fix");
    let f32_must: Vec<(&str, &S, &S)> = vec![
        ("P3 -> sRGB", &p3, &srgb), ("sRGB -> P3", &srgb, &p3), ("BT.2020 -> sRGB", &bt2020, &srgb), ("sRGB -> BT.2020", &srgb, &bt2020),
        ("P3 -> Adobe (target route, already wrapped)", &p3, &ad), ("sRGB -> ProPhoto (target route)", &srgb, &pp), ("tabled curve -> Adobe (target route)", &tabled, &ad), ("Adobe -> tabled curve (table path)", &ad, &tabled), ("ProPhoto -> tabled curve (table path)", &pp, &tabled),
        ("CICP sRGB -> P3", &csrgb, &p3), ("P3 -> CICP sRGB", &p3, &csrgb), ("CICP gamma 2.2 -> sRGB", &cg22, &srgb), ("CICP gamma 2.8 -> sRGB", &cg28, &srgb), ("CICP SMPTE 428 -> sRGB", &c428, &srgb), ("P3 -> CICP gamma 2.2", &p3, &cg22), ("P3 -> CICP gamma 2.8", &p3, &cg28), ("CICP PQ -> sRGB", &cpq, &srgb),
        ("LUT -> sRGB", &lut, &srgb), ("LUT -> P3", &lut, &p3), ("LUT -> CICP gamma 2.2", &lut, &cg22), ("sRGB -> LUT", &srgb, &lut), ("ProPhoto -> LUT (tables, no evaluator)", &pp, &lut), ("Adobe -> LUT (tables, no evaluator)", &ad, &lut),
    ];
    for clip in [true, false] { for (n, s, t) in &f32_must {
        let (b, a): (R, R) = ((|| run!(zb, zcb, *s, *t, RgbF32, RgbF32, clip, gb, w, gb.len()))(), (|| run!(za, zca, *s, *t, RgbF32, RgbF32, clip, gb, w, gb.len()))());
        let (x, y) = report(&format!("RgbF32 {n} clip={clip}"), b, a, w * 3, 4); must_n += x; must_d += y;
        let (b, a): (R, R) = ((|| run!(zb, zcb, *s, *t, RgbaF32, RgbaF32, clip, rgba_b, w, rgba_b.len()))(), (|| run!(za, zca, *s, *t, RgbaF32, RgbaF32, clip, rgba_b, w, rgba_b.len()))());
        let (x, y) = report(&format!("RgbaF32 {n} clip={clip}"), b, a, w * 4, 4); must_n += x; must_d += y;
    } }
    println!("=== must stay bit-identical: CMYK at F32");
    let cm: Vec<f32> = (0..200000u32).flat_map(|i| { let f = |k: u32| (i.wrapping_mul(k) >> 7 & 255) as f32 / 255.; [f(2654435761), f(40503), f(2246822519), f(3266489917)] }).collect(); let cm_b: &[u8] = bytemuck::cast_slice(&cm);
    for (n, t) in [("sRGB", &srgb), ("Adobe", &ad), ("ProPhoto", &pp)] { for clip in [true, false] {
        let (b, a): (R, R) = ((|| run!(zb, zcb, &cmyk, t, CmykF32, RgbF32, clip, cm_b, 200000, 200000 * 12))(), (|| run!(za, zca, &cmyk, t, CmykF32, RgbF32, clip, cm_b, 200000, 200000 * 12))());
        let (x, y) = report(&format!("CmykF32 -> {n} clip={clip}"), b, a, 600000, 4); must_n += x; must_d += y;
    } }
    println!("=== must stay bit-identical: every U8 and U16 route, pure-gamma pairs included");
    let u = u8grid(); let uw = u.len() / 3; let gu: Vec<u8> = (0..=255).collect();
    let u16g: Vec<u16> = u.iter().map(|&v| v as u16 * 257).collect(); let u16b: &[u8] = bytemuck::cast_slice(&u16g);
    let pairs: Vec<(&str, &S, &S)> = vec![("ProPhoto->sRGB", &pp, &srgb), ("ProPhoto->Adobe", &pp, &ad), ("Adobe->ProPhoto", &ad, &pp), ("Adobe->sRGB", &ad, &srgb), ("P3->Adobe", &p3, &ad), ("P3->sRGB", &p3, &srgb), ("LUT->sRGB", &lut, &srgb), ("LUT->Adobe", &lut, &ad), ("CICP sRGB->P3", &csrgb, &p3), ("CICP gamma 2.2->sRGB", &cg22, &srgb)];
    for (n, s, t) in &pairs {
        let (b, a): (R, R) = ((|| run!(zb, zcb, *s, *t, Rgb8, Rgb8, true, &u, uw, u.len()))(), (|| run!(za, zca, *s, *t, Rgb8, Rgb8, true, &u, uw, u.len()))());
        let (x, y) = report(&format!("Rgb8 {n}"), b, a, u.len(), 1); must_n += x; must_d += y;
        let (b, a): (R, R) = ((|| run!(zb, zcb, *s, *t, Rgb16, Rgb16, true, u16b, uw, u16b.len()))(), (|| run!(za, zca, *s, *t, Rgb16, Rgb16, true, u16b, uw, u16b.len()))());
        let (x, y) = report(&format!("Rgb16 {n}"), b, a, u16g.len(), 2); must_n += x; must_d += y;
    }
    for (n, s, t, tn) in [("Gray1.8->Gray2.2", &g18, &g22, 1usize), ("Gray2.2->Gray1.8", &g22, &g18, 1)] {
        let (b, a): (R, R) = ((|| run!(zb, zcb, s, t, Gray8, Gray8, true, &gu, 256, 256 * tn))(), (|| run!(za, zca, s, t, Gray8, Gray8, true, &gu, 256, 256 * tn))());
        let (x, y) = report(&format!("Gray8 {n}"), b, a, 256, 1); must_n += x; must_d += y;
    }
    for (n, s, t) in [("Gray1.8->sRGB", &g18, &srgb), ("Gray1.8->Adobe", &g18, &ad)] {
        let (b, a): (R, R) = ((|| run!(zb, zcb, s, t, Gray8, Rgb8, true, &gu, 256, 768))(), (|| run!(za, zca, s, t, Gray8, Rgb8, true, &gu, 256, 768))());
        let (x, y) = report(&format!("Gray8->Rgb8 {n}"), b, a, 768, 1); must_n += x; must_d += y;
    }
    for (n, s, t) in [("ProPhoto->Gray2.2", &pp, &g22), ("sRGB->Gray2.2", &srgb, &g22)] {
        let (b, a): (R, R) = ((|| run!(zb, zcb, s, t, Rgb8, Gray8, true, &u, uw, uw))(), (|| run!(za, zca, s, t, Rgb8, Gray8, true, &u, uw, uw))());
        let (x, y) = report(&format!("Rgb8->Gray8 {n}"), b, a, uw, 1); must_n += x; must_d += y;
    }
    let full = u8full(); let fw = full.len() / 3;
    for (n, s, t) in [("ProPhoto->Adobe", &pp, &ad), ("ProPhoto->sRGB", &pp, &srgb), ("P3->Adobe", &p3, &ad)] {
        let (b, a): (R, R) = ((|| run!(zb, zcb, s, t, Rgb8, Rgb8, true, &full, fw, full.len()))(), (|| run!(za, zca, s, t, Rgb8, Rgb8, true, &full, fw, full.len()))());
        let (x, y) = report(&format!("Rgb8 all 2^24 {n}"), b, a, full.len(), 1); must_n += x; must_d += y;
    }
    println!("MUST-STAY TOTAL: {must_n} samples, {must_d} differ");

    println!("=== fixed routes: how far did values with no bug in them move?");
    let srgb_gray_b = gray_b; let _ = srgb_gray_b;
    let valid: Vec<f32> = g.chunks_exact(3).filter(|c| c.iter().all(|&v| (1e-4..=8.0).contains(&v))).flatten().copied().collect();
    let vb: &[u8] = bytemuck::cast_slice(&valid); let vw = valid.len() / 3;
    for (n, s, t) in [("ProPhoto->sRGB", &pp, &srgb), ("ProPhoto->Adobe", &pp, &ad), ("Adobe->sRGB", &ad, &srgb), ("Adobe->ProPhoto", &ad, &pp)] { for clip in [true, false] {
        let (b, a): (R, R) = ((|| run!(zb, zcb, s, t, RgbF32, RgbF32, clip, vb, vw, vb.len()))(), (|| run!(za, zca, s, t, RgbF32, RgbF32, clip, vb, vw, vb.len()))());
        moved(&format!("{n} clip={clip}, inputs 1e-4..8"), b, a);
    } }
    let lut_valid: Vec<f32> = (0..=255).step_by(5).flat_map(|v: i32| { let v = v as f32 / 255.; [v, v, v] }).collect(); let lvb: &[u8] = bytemuck::cast_slice(&lut_valid);
    for clip in [true, false] {
        let (b, a): (R, R) = ((|| run!(zb, zcb, &lut, &ad, RgbF32, RgbF32, clip, gb, w, gb.len()))(), (|| run!(za, zca, &lut, &ad, RgbF32, RgbF32, clip, gb, w, gb.len()))());
        moved(&format!("sRGB LUT->Adobe (nothing out of gamut) clip={clip}"), b, a);
        let (b, a): (R, R) = ((|| run!(zb, zcb, &lut, &pp, RgbF32, RgbF32, clip, lvb, lut_valid.len() / 3, lvb.len()))(), (|| run!(za, zca, &lut, &pp, RgbF32, RgbF32, clip, lvb, lut_valid.len() / 3, lvb.len()))());
        moved(&format!("sRGB LUT->ProPhoto neutrals clip={clip}"), b, a);
    }
    let in_range: Vec<f32> = gray_in.iter().copied().filter(|v| (1e-4..=8.0).contains(v)).collect(); let irb: &[u8] = bytemuck::cast_slice(&in_range);
    for (n, s, t) in [("Gray1.8->Gray2.2", &g18, &g22), ("Gray2.2->Gray1.8", &g22, &g18)] {
        let (b, a): (R, R) = ((|| run!(zb, zcb, s, t, GrayF32, GrayF32, false, irb, in_range.len(), irb.len()))(), (|| run!(za, zca, s, t, GrayF32, GrayF32, false, irb, in_range.len(), irb.len()))());
        moved(&format!("{n}, inputs 1e-4..8"), b, a);
    }
}
fn moved(name: &str, b: R, a: R) {
    let (Ok(x), Ok(y)) = (&b, &a) else { println!("{name}: {:?} / {:?}", b.as_ref().err(), a.as_ref().err()); return; };
    let (x, y): (&[f32], &[f32]) = (bytemuck::cast_slice(x), bytemuck::cast_slice(y));
    let (mut d, mut worst_abs, mut worst_rel) = (0usize, 0f64, 0f64);
    for (p, q) in x.iter().zip(y) { if p.to_bits() != q.to_bits() { d += 1; let e = (*p as f64 - *q as f64).abs(); if e > worst_abs { worst_abs = e; } let r = e / (p.abs() as f64).max(1e-3); if r > worst_rel { worst_rel = r; } } }
    println!("{name:<54} {} samples, bits differ {d}, worst abs move {worst_abs:.3e} ({:.4} of an 8-bit code), worst rel {worst_rel:.3e}", x.len(), worst_abs * 255.);
}
