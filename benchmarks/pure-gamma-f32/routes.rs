// Every route class through zenpixels, before and after, against exact f64 math where it exists.
use moxcms::{ColorProfile as P, ToneReprCurve as T};
macro_rules! both { ($f:ident ! ($($a:tt)*)) => { (($f!(zb, zcb, $($a)*)), ($f!(za, zca, $($a)*))) } }
macro_rules! f32run { ($z:ident, $zc:ident, $s:expr, $t:expr, $from:ident, $to:ident, $clip:expr, $px:expr, $n:expr) => {{
    use $zc::cms::PluggableCms;
    let o = $zc::ConvertOptions::retain_channels().with_clip_out_of_gamut($clip);
    let r = $zc::cms_moxcms::MoxCms.build_source_transform($z::ColorProfileSource::Icc($s), $z::ColorProfileSource::Icc($t), $z::PixelFormat::$from, $z::PixelFormat::$to, &o);
    match r { None => Err("declined".to_string()), Some(Err(e)) => Err(format!("{e:?}").chars().take(70).collect()), Some(Ok(mut tr)) => {
        let mut d = vec![0f32; $n]; let src: &[f32] = $px;
        tr.try_transform_row(bytemuck::cast_slice(src), bytemuck::cast_slice_mut(&mut d), (src.len() / ($px.len() / 1).max(1)).max(1) as u32 * 1).map(|_| d).map_err(|e| format!("{e:?}").chars().take(70).collect::<String>())
    } }
}}}
macro_rules! u8run { ($z:ident, $zc:ident, $s:expr, $t:expr, $from:ident, $to:ident, $px:expr, $n:expr) => {{
    use $zc::cms::PluggableCms;
    let o = $zc::ConvertOptions::permissive();
    let r = $zc::cms_moxcms::MoxCms.build_source_transform($z::ColorProfileSource::Icc($s), $z::ColorProfileSource::Icc($t), $z::PixelFormat::$from, $z::PixelFormat::$to, &o);
    match r { None => Err("declined".to_string()), Some(Err(e)) => Err(format!("{e:?}").chars().take(70).collect()), Some(Ok(mut tr)) => {
        let mut d = vec![0u8; $n]; let src: &[u8] = $px;
        tr.try_transform_row(src, &mut d, 1).map(|_| d).map_err(|e| format!("{e:?}").chars().take(70).collect::<String>())
    } }
}}}
fn gray(g: f32) -> Vec<u8> { P::new_gray_with_gamma(g).encode().unwrap() }
fn no_cicp(mut p: P) -> Vec<u8> { p.cicp = None; p.encode().unwrap() }
fn gamma_rgb(mut p: P, g: f32) -> Vec<u8> { p.cicp = None; let c = moxcms::curve_from_gamma(g); p.red_trc = Some(c.clone()); p.green_trc = Some(c.clone()); p.blue_trc = Some(c); p.encode().unwrap() }
fn g(icc: &[u8]) -> f64 { let p = P::new_from_slice(icc).unwrap(); let c = if p.gray_trc.is_some() { p.gray_trc } else { p.red_trc }; match c.unwrap() { T::Lut(l) if l.len() == 1 => l[0] as f64 / 256., _ => f64::NAN } }
fn row(name: &str, px: String, b: Result<Vec<f32>, String>, a: Result<Vec<f32>, String>, exact: Option<Vec<f64>>) {
    let same = match (&b, &a) { (Ok(x), Ok(y)) => x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()), _ => false };
    println!("{name:<40} {px:<24} before {b:?}\n{:<65} after  {a:?}{}{}", "", if same { " (identical)" } else { "" }, exact.map(|e| format!("  exact {e:.5?}")).unwrap_or_default());
}
macro_rules! rgbrow { ($name:expr, $s:expr, $t:expr, $clip:expr, $px:expr) => {{
    let px: [f32; 3] = $px; let (b, a) = both!(f32run!($s, $t, RgbF32, RgbF32, $clip, &px, 3));
    let (ss, tt) = (P::new_from_slice($s).unwrap(), P::new_from_slice($t).unwrap());
    let m = tt.rgb_to_xyz_matrix().inverse().mat_mul(ss.rgb_to_xyz_matrix()).v;
    let (gs, gt) = (g($s), g($t));
    let l = px.map(|x| (x as f64).abs().powf(gs).copysign(x as f64));
    let e = (0..3).map(|c| { let v = m[c][0]*l[0]+m[c][1]*l[1]+m[c][2]*l[2]; let v = if $clip { v.clamp(0., 1.) } else { v }; if gt.is_nan() { let u = v.abs(); (if u < 0.0031308 { 12.92 * u } else { 1.055 * u.powf(1. / 2.4) - 0.055 }).copysign(v) } else { v.abs().powf(1. / gt).copysign(v) } }).collect();
    row(&format!("{} clip={}", $name, $clip), format!("{px:?}"), b, a, Some(e));
}}}
macro_rules! grayrow { ($name:expr, $s:expr, $t:expr, $to:ident, $tn:expr, $clip:expr, $x:expr) => {{
    let px = [$x as f32]; let (b, a) = both!(f32run!($s, $t, GrayF32, $to, $clip, &px, $tn));
    row(&format!("{} clip={}", $name, $clip), format!("{:?}", $x), b, a, None);
}}}
macro_rules! togray { ($name:expr, $s:expr, $t:expr, $clip:expr, $px:expr) => {{
    let px: [f32; 3] = $px; let (b, a) = both!(f32run!($s, $t, RgbF32, GrayF32, $clip, &px, 1));
    row(&format!("{} clip={}", $name, $clip), format!("{px:?}"), b, a, None);
}}}
macro_rules! u8row { ($name:expr, $s:expr, $t:expr, $from:ident, $to:ident, $tn:expr, $px:expr) => {{
    let px = $px; let (b, a) = both!(u8run!($s, $t, $from, $to, &px, $tn));
    println!("U8 {:<37} {:<24} before {b:?} after {a:?}", $name, format!("{px:?}"));
}}}
fn main() {
    let (pp, ad, srgb) = (no_cicp(P::new_pro_photo_rgb()), no_cicp(P::new_adobe_rgb()), no_cicp(P::new_srgb()));
    let (g18, g22, g26) = (gray(1.8), gray(2.2), gray(2.6));
    println!("=== RGB pure-gamma source");
    for clip in [true, false] {
        for px in [[0.5, 0.5, 0.0], [0.6, 0.0, 0.3], [0.0, 0.4, 0.4], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]] { rgbrow!("ProPhoto->sRGB", &pp, &srgb, clip, px); }
        for px in [[1e-30, 1e-30, 1e-30], [1e-20, 0.5, 0.5], [0.0, 0.0, 0.0], [-0.5, 0.5, 0.5]] { rgbrow!("Adobe->sRGB", &ad, &srgb, clip, px); }
        for px in [[0.5, 0.5, 0.0], [-0.5, 0.5, 0.5]] { rgbrow!("ProPhoto->Adobe", &pp, &ad, clip, px); }
    }
    println!("=== U8 routes");
    u8row!("ProPhoto->sRGB", &pp, &srgb, Rgb8, Rgb8, 3, [128u8, 128, 0]);
    u8row!("ProPhoto->Adobe", &pp, &ad, Rgb8, Rgb8, 3, [128u8, 128, 0]);
    u8row!("ProPhoto->Adobe", &pp, &ad, Rgb8, Rgb8, 3, [0u8, 0, 0]);
    u8row!("ProPhoto->Adobe", &pp, &ad, Rgb8, Rgb8, 3, [153u8, 0, 77]);
    u8row!("Adobe->ProPhoto", &ad, &pp, Rgb8, Rgb8, 3, [128u8, 128, 0]);
    u8row!("Gray1.8->Gray2.2", &g18, &g22, Gray8, Gray8, 1, [0u8]);
    u8row!("Gray1.8->Gray2.2", &g18, &g22, Gray8, Gray8, 1, [128u8]);
    u8row!("Gray1.8->sRGB", &g18, &srgb, Gray8, Rgb8, 3, [0u8]);
    u8row!("Gray1.8->sRGB", &g18, &srgb, Gray8, Rgb8, 3, [128u8]);
    u8row!("ProPhoto->Gray2.2", &pp, &g22, Rgb8, Gray8, 1, [128u8, 128, 0]);
    println!("=== Gray source");
    for clip in [true, false] {
        for x in [0.0f32, 1e-30, 1e-20, -0.5, 0.5] {
            grayrow!("Gray1.8->Gray2.2", &g18, &g22, GrayF32, 1, clip, x);
            grayrow!("Gray2.2->Gray2.6", &g22, &g26, GrayF32, 1, clip, x);
            grayrow!("Gray1.8->sRGB", &g18, &srgb, RgbF32, 3, clip, x);
            grayrow!("Gray1.8->Adobe", &g18, &ad, RgbF32, 3, clip, x);
        }
    }
    println!("=== RGB to gray target");
    let srgbs = srgb.clone();
    for clip in [true, false] {
        for px in [[0.5, 0.5, 0.0], [0.0, 0.0, 0.0], [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5]] {
            togray!("ProPhoto->Gray2.2", &pp, &g22, clip, px);
            togray!("sRGB->Gray2.2", &srgbs, &g22, clip, px);
        }
    }
    println!("=== LUT source to a narrow pure-gamma target");
    let lut = std::fs::read(std::env::var("LUT_ICC").unwrap()).unwrap();
    let narrow = |gm: f32, srgb_primaries_scale: f64| { let mut p = P::new_srgb(); p.cicp = None; let c = if gm == 0. { T::Lut(vec![]) } else { moxcms::curve_from_gamma(gm) }; p.red_trc = Some(c.clone()); p.green_trc = Some(c.clone()); p.blue_trc = Some(c); let w = [0.9642, 1.0, 0.8249]; let k = srgb_primaries_scale; for (col, base) in [(&mut p.red_colorant, 1.0/3.0), (&mut p.green_colorant, 1.0/3.0), (&mut p.blue_colorant, 1.0/3.0)] { col.x = col.x * k + w[0] * base * (1.0 - k); col.y = col.y * k + w[1] * base * (1.0 - k); col.z = col.z * k + w[2] * base * (1.0 - k); } p.encode().unwrap() };
    let (t22, t0) = (narrow(2.19921875, 0.7), narrow(0., 0.7));
    println!("LUT source parsed: a2b0 {} a2b1 {} red_trc {}", P::new_from_slice(&lut).unwrap().lut_a_to_b_perceptual.is_some(), P::new_from_slice(&lut).unwrap().lut_a_to_b_colorimetric.is_some(), P::new_from_slice(&lut).unwrap().red_trc.is_some());
    for clip in [true, false] {
        for px in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.5], [1.0, 1.0, 0.0], [0.8, 0.2, 0.1]] {
            let px: [f32; 3] = px;
            let (b, a) = both!(f32run!(&lut, &t22, RgbF32, RgbF32, clip, &px, 3));
            let (lb, _la) = both!(f32run!(&lut, &t0, RgbF32, RgbF32, false, &px, 3));
            let e = lb.clone().ok().map(|l: Vec<f32>| l.iter().map(|&v| { let v = v as f64; let v = if clip { v.clamp(0., 1.) } else { v }; v.abs().powf(256. / 563.).copysign(v) }).collect());
            row(&format!("sRGB LUT->narrow gamma clip={clip}"), format!("{px:?}"), b, a, e);
        }
    }
}
