// The routes the wrapper can't fix or needs evidence for, run through zenpixels, plus the choice for table sources.
use moxcms::{ColorProfile as P, ToneReprCurve as T};
#[path = "../../zenpixels/zenpixels-convert/tests/common/synthetic_icc.rs"]
#[allow(dead_code)]
mod synthetic_icc;
macro_rules! f { ($z:ident, $zc:ident, $s:expr, $t:expr, $from:ident, $to:ident, $clip:expr, $input:expr, $w:expr, $n:expr) => {{
    use $zc::cms::PluggableCms;
    let o = $zc::ConvertOptions::retain_channels().with_clip_out_of_gamut($clip);
    match $zc::cms_moxcms::MoxCms.build_source_transform($s, $t, $z::PixelFormat::$from, $z::PixelFormat::$to, &o) {
        None => Err("declined".to_string()),
        Some(Err(e)) => Err(format!("refused: {}", format!("{e:?}").chars().take(90).collect::<String>())),
        Some(Ok(mut tr)) => { let mut d = vec![0f32; $n]; let i: &[f32] = $input; tr.try_transform_row(bytemuck::cast_slice(i), bytemuck::cast_slice_mut(&mut d), $w as u32).map_err(|e| format!("{e:?}"))?; Ok::<Vec<f32>, String>(d) }
    }
}}}
macro_rules! ic { ($z:ident, $i:expr) => { $z::ColorProfileSource::Icc($i) } }
macro_rules! pair { ($z:ident, $p:ident, $t:ident) => { $z::ColorProfileSource::PrimariesTransferPair { primaries: $z::ColorPrimaries::$p, transfer: $z::TransferFunction::$t } } }
fn narrow(curve: T) -> Vec<u8> {
    let mut p = P::new_srgb(); p.cicp = None; p.red_trc = Some(curve.clone()); p.green_trc = Some(curve.clone()); p.blue_trc = Some(curve);
    let w = [0.9642, 1.0, 0.8249];
    for c in [&mut p.red_colorant, &mut p.green_colorant, &mut p.blue_colorant] { c.x = c.x * 0.7 + w[0] * 0.1; c.y = c.y * 0.7 + w[1] * 0.1; c.z = c.z * 0.7 + w[2] * 0.1; }
    p.encode().unwrap()
}
fn srgb_dec(x: f64) -> f64 { if x < 0.04045 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) } }
#[derive(Default)] struct St { n: u64, sum: f64, worst: f64, bad: u64 }
impl St { fn add(&mut self, got: f64, e: f64) { let d = (got.clamp(0., 1.) - e.clamp(0., 1.)).abs() * 255.; self.n += 1; self.sum += d; if d > self.worst { self.worst = d; } if d > 2. { self.bad += 1; } }
    fn show(&self, n: &str) { println!("  {n:<46} mean |err| {:>8.4} codes, worst {:>8.3}, >2 codes {:>7} of {}", self.sum / self.n.max(1) as f64, self.worst, self.bad, self.n); } }
fn main() {
    let g18 = P::new_gray_with_gamma(1.8).encode().unwrap();
    let g22 = P::new_gray_with_gamma(2.2).encode().unwrap();
    let srgb = { let mut p = P::new_srgb(); p.cicp = None; p.encode().unwrap() };
    let adobe = { let mut p = P::new_adobe_rgb(); p.cicp = None; p.encode().unwrap() };
    let pp = { let mut p = P::new_pro_photo_rgb(); p.cicp = None; p.encode().unwrap() };
    let lut = std::fs::read(std::env::var("LUT_ICC").unwrap()).unwrap();
    let syn = synthetic_icc::synthetic_rgb_lut_icc();
    println!("=== LUT source into a gray target, and a gray source with a table");
    let px = [0.5f32, 0.5, 0.0];
    for (n, s) in [("srgb_perceptual.icc", &lut), ("synthetic LUT", &syn)] { for clip in [true, false] {
        let b = (|| f!(zb, zcb, ic!(zb, s), ic!(zb, &g22), RgbF32, GrayF32, clip, &px, 1, 1))();
        let a = (|| f!(za, zca, ic!(za, s), ic!(za, &g22), RgbF32, GrayF32, clip, &px, 1, 1))();
        println!("{n} -> Gray2.2 clip={clip}: before {b:?} after {a:?}");
    } }
    let gray_tabled = { let mut p = P::new_gray_with_gamma(1.8); p.lut_a_to_b_perceptual = P::new_from_slice(&lut).unwrap().lut_a_to_b_perceptual; p.encode() };
    println!("gray profile with a table encodes: {:?}", gray_tabled.as_ref().map(|v| v.len()).map_err(|e| format!("{e:?}")));
    if let Ok(g) = gray_tabled.as_ref() {
        println!("  parses: {:?}", P::new_from_slice(g).map(|p| (p.color_space == moxcms::DataColorSpace::Gray, p.lut_a_to_b_perceptual.is_some(), p.gray_trc.is_some())).map_err(|e| format!("{e:?}")));
        for t in [&srgb, &g22] { let to_gray = std::ptr::eq(t, &g22);
            if to_gray { let b = (|| f!(zb, zcb, ic!(zb, g), ic!(zb, t), GrayF32, GrayF32, true, &[0.5f32], 1, 1))(); let a = (|| f!(za, zca, ic!(za, g), ic!(za, t), GrayF32, GrayF32, true, &[0.5f32], 1, 1))(); println!("  tabled gray -> Gray2.2: before {b:?} after {a:?}"); }
            else { let b = (|| f!(zb, zcb, ic!(zb, g), ic!(zb, t), GrayF32, RgbF32, true, &[0.5f32], 1, 3))(); let a = (|| f!(za, zca, ic!(za, g), ic!(za, t), GrayF32, RgbF32, true, &[0.5f32], 1, 3))(); println!("  tabled gray -> sRGB: before {b:?} after {a:?}"); }
        }
    }
    println!("=== pure-gamma through primaries+transfer pairs (no ICC)");
    let px = [0.5f32, 0.5, 0.0];
    let b = (|| f!(zb, zcb, pair!(zb, AdobeRgb, Gamma22), ic!(zb, &srgb), RgbF32, RgbF32, true, &px, 1, 3))();
    let a = (|| f!(za, zca, pair!(za, AdobeRgb, Gamma22), ic!(za, &srgb), RgbF32, RgbF32, true, &px, 1, 3))();
    println!("pair Adobe/Gamma22 -> sRGB clip=true: before {b:?} after {a:?}");
    for x in [0f32, 1e-30, 1e-20] { let p = [x, 0.5, 0.5];
        let b = (|| f!(zb, zcb, pair!(zb, AdobeRgb, Gamma22), ic!(zb, &srgb), RgbF32, RgbF32, false, &p, 1, 3))();
        let a = (|| f!(za, zca, pair!(za, AdobeRgb, Gamma22), ic!(za, &srgb), RgbF32, RgbF32, false, &p, 1, 3))();
        println!("pair Adobe/Gamma22 -> sRGB clip=false {p:?}: before {b:?} after {a:?}");
    }
    println!("=== moved values: error against exact, before and after");
    let mut grid = Vec::new();
    for r in (0..=255).step_by(5) { for g in (0..=255).step_by(5) { for b in (0..=255).step_by(5) { grid.extend_from_slice(&[r as f32 / 255., g as f32 / 255., b as f32 / 255.]); } } }
    let w = grid.len() / 3;
    for (n, s, t, ts) in [("ProPhoto->Adobe", &pp, &adobe, false), ("ProPhoto->sRGB", &pp, &srgb, true), ("Adobe->sRGB", &adobe, &srgb, true), ("Adobe->ProPhoto", &adobe, &pp, false)] {
        let sp = P::new_from_slice(s).unwrap(); let tp = P::new_from_slice(t).unwrap();
        let m = tp.rgb_to_xyz_matrix().inverse().mat_mul(sp.rgb_to_xyz_matrix()).v;
        let gm = |p: &P| match p.red_trc.as_ref().unwrap() { T::Lut(l) => l[0] as f64 / 256., _ => 0. };
        let (gs, gt) = (gm(&sp), gm(&tp));
        let exact = |c: usize, px: &[f32]| { let l = [(px[0] as f64).powf(gs), (px[1] as f64).powf(gs), (px[2] as f64).powf(gs)]; let v = (m[c][0]*l[0]+m[c][1]*l[1]+m[c][2]*l[2]).clamp(0., 1.); if ts { if v < 0.0030412825601275209 { v * 12.92 } else { 1.0550107189475866 * v.powf(1. / 2.4) - 0.0550107189475866 } } else { v.powf(1. / gt) } };
        let b = (|| f!(zb, zcb, ic!(zb, s), ic!(zb, t), RgbF32, RgbF32, true, &grid, w, w * 3))().unwrap();
        let a = (|| f!(za, zca, ic!(za, s), ic!(za, t), RgbF32, RgbF32, true, &grid, w, w * 3))().unwrap();
        let (mut sb, mut sa) = (St::default(), St::default());
        for (i, px) in grid.chunks_exact(3).enumerate() { for c in 0..3 { let e = exact(c, px); sb.add(b[i * 3 + c] as f64, e); sa.add(a[i * 3 + c] as f64, e); } }
        println!("{n}, 52^3 grid, clip=true:"); sb.show("before"); sa.show("after");
    }
    println!("=== table source into a narrow pure-gamma target, options, against the sRGB matrix (the table's source)");
    let srgbp = P::new_from_slice(&srgb).unwrap();
    let opts = moxcms::TransformOptions { rendering_intent: moxcms::RenderingIntent::RelativeColorimetric, allow_use_cicp_transfer: false, barycentric_weight_scale: moxcms::BarycentricWeightScale::High, interpolation_method: moxcms::InterpolationMethod::Tetrahedral, prefer_fixed_point: false, allow_extended_range_rgb_xyz: true, ..Default::default() };
    let t22 = narrow(moxcms::curve_from_gamma(2.19921875)); let tlin = narrow(T::Lut(vec![]));
    let tp = P::new_from_slice(&t22).unwrap();
    let m = tp.rgb_to_xyz_matrix().inverse().mat_mul(srgbp.rgb_to_xyz_matrix()).v;
    for (name, src) in [("srgb_perceptual.icc", &lut), ("synthetic LUT", &syn)] {
        let sp = P::new_from_slice(src).unwrap();
        let exact = |c: usize, px: &[f32]| -> f64 { let l = [srgb_dec(px[0] as f64), srgb_dec(px[1] as f64), srgb_dec(px[2] as f64)]; let v = m[c][0]*l[0]+m[c][1]*l[1]+m[c][2]*l[2]; v.abs().powf(256. / 563.).copysign(v) };
        let b = (|| f!(zb, zcb, ic!(zb, src), ic!(zb, &t22), RgbF32, RgbF32, true, &grid, w, w * 3))().unwrap();
        let a = (|| f!(za, zca, ic!(za, src), ic!(za, &t22), RgbF32, RgbF32, true, &grid, w, w * 3))().unwrap();
        let opts = if sp.lut_a_to_b_colorimetric.is_none() { moxcms::TransformOptions { rendering_intent: moxcms::RenderingIntent::Perceptual, ..opts } } else { opts };
        let mut lin = vec![0f32; w * 3]; sp.create_transform_f32(moxcms::Layout::Rgb, &P::new_from_slice(&tlin).unwrap(), moxcms::Layout::Rgb, opts).unwrap().transform(&grid, &mut lin).unwrap();
        let mut tab = vec![0f32; w * 3]; sp.create_transform_f32(moxcms::Layout::Rgb, &tp, moxcms::Layout::Rgb, moxcms::TransformOptions { allow_extended_range_rgb_xyz: false, ..opts }).unwrap().transform(&grid, &mut tab).unwrap();
        // Splits: the whole grid, channels exactly in gamut in a pixel with nothing out of gamut, and the rest.
        for (split, keep) in [("all samples", 0), ("pixels entirely in gamut", 1), ("pixels with an out-of-gamut channel", 2)] {
            let (mut sb, mut sa, mut sl, mut st) = (St::default(), St::default(), St::default(), St::default());
            for (i, px) in grid.chunks_exact(3).enumerate() {
                let oog = (0..3).any(|c| { let e = exact(c, px); !(0.0..=1.0).contains(&e) });
                if (keep == 1 && oog) || (keep == 2 && !oog) { continue; }
                for c in 0..3 { let e = exact(c, px); sb.add(b[i * 3 + c] as f64, e); sa.add(a[i * 3 + c] as f64, e); let v = lin[i * 3 + c] as f64; sl.add(v.abs().powf(256. / 563.).copysign(v), e); st.add(tab[i * 3 + c] as f64, e); }
            }
            println!("{name}, {split}:"); sb.show("before"); sa.show("after: parametric + negative pass"); sl.show("option: interpolate linear, encode after"); st.show("option: extended range off (tables)");
        }
    }
}
