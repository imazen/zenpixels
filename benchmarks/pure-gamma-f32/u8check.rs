// 8-bit before (main before either fix) and after this change, against exact f64 math.
use moxcms::{ColorProfile as P, ToneReprCurve as T};
macro_rules! run { ($z:ident, $zc:ident, $s:expr, $t:expr, $from:ident, $to:ident, $input:expr, $w:expr, $n:expr) => {{
    use $zc::cms::PluggableCms;
    let mut tr = $zc::cms_moxcms::MoxCms.build_source_transform($z::ColorProfileSource::Icc($s), $z::ColorProfileSource::Icc($t), $z::PixelFormat::$from, $z::PixelFormat::$to, &$zc::ConvertOptions::permissive()).unwrap().unwrap();
    let mut d = vec![0u8; $n]; tr.try_transform_row($input, &mut d, $w as u32).unwrap(); d
}}}
fn gm(p: &P) -> f64 { match p.gray_trc.as_ref().or(p.red_trc.as_ref()).unwrap() { T::Lut(l) => l[0] as f64 / 256., _ => f64::NAN } }
fn stats(name: &str, got: &[u8], exact: &[f64], other: &[u8]) {
    let (mut sum, mut worst, mut miss, mut over2) = (0f64, 0f64, 0u64, 0u64);
    for (g, e) in got.iter().zip(exact) { let d = (*g as f64 - e).abs(); sum += d; if d > worst { worst = d; } if *g as f64 != e.round() { miss += 1; } if d > 2.0 { over2 += 1; } }
    println!("{name:<44} mean |err| {:>8.4} worst {:>8.3} not-nearest {miss:>8} over-2-codes {over2:>8} of {}; differ from other side {}", sum / got.len() as f64, worst, got.len(), got.iter().zip(other).filter(|(a, b)| a != b).count());
}
fn cube(step: usize) -> Vec<u8> { let mut v = Vec::new(); for r in (0..=255).step_by(step) { for g in (0..=255).step_by(step) { for b in (0..=255).step_by(step) { v.extend_from_slice(&[r as u8, g as u8, b as u8]); } } } v }
fn main() {
    let no = |mut p: P| { p.cicp = None; p.encode().unwrap() };
    let (pp, ad, srgb, p3) = (no(P::new_pro_photo_rgb()), no(P::new_adobe_rgb()), no(P::new_srgb()), no(P::new_display_p3()));
    let g = |x: f32| P::new_gray_with_gamma(x).encode().unwrap();
    let (g18, g22, g26) = (g(1.8), g(2.2), g(2.6));
    let c = cube(5); let w = c.len() / 3;
    for (n, s, t) in [("ProPhoto->Adobe", &pp, &ad), ("Adobe->ProPhoto", &ad, &pp)] {
        let (sp, tp) = (P::new_from_slice(s).unwrap(), P::new_from_slice(t).unwrap());
        let m = tp.rgb_to_xyz_matrix().inverse().mat_mul(sp.rgb_to_xyz_matrix()).v; let (gs, gt) = (gm(&sp), gm(&tp));
        let exact: Vec<f64> = c.chunks_exact(3).flat_map(|p| { let l = [(p[0] as f64 / 255.).powf(gs), (p[1] as f64 / 255.).powf(gs), (p[2] as f64 / 255.).powf(gs)]; [0, 1, 2].map(|i| (m[i][0]*l[0]+m[i][1]*l[1]+m[i][2]*l[2]).clamp(0., 1.).powf(1. / gt) * 255.) }).collect();
        let (b, a) = (run!(zb, zcb, s, t, Rgb8, Rgb8, &c, w, c.len()), run!(za, zca, s, t, Rgb8, Rgb8, &c, w, c.len()));
        stats(&format!("{n} before"), &b, &exact, &a); stats(&format!("{n} after"), &a, &exact, &b);
    }
    let lv: Vec<u8> = (0..=255).collect();
    for (n, s, t) in [("Gray1.8->Gray2.2", &g18, &g22), ("Gray2.2->Gray2.6", &g22, &g26), ("Gray2.6->Gray1.8", &g26, &g18)] {
        let (gs, gt) = (gm(&P::new_from_slice(s).unwrap()), gm(&P::new_from_slice(t).unwrap()));
        let exact: Vec<f64> = lv.iter().map(|&v| (v as f64 / 255.).powf(gs).powf(1. / gt) * 255.).collect();
        let (b, a) = (run!(zb, zcb, s, t, Gray8, Gray8, &lv, 256, 256), run!(za, zca, s, t, Gray8, Gray8, &lv, 256, 256));
        stats(&format!("{n} before"), &b, &exact, &a); stats(&format!("{n} after"), &a, &exact, &b);
    }
    { let gs = gm(&P::new_from_slice(&g18).unwrap()); let gt = gm(&P::new_from_slice(&ad).unwrap());
      let exact: Vec<f64> = lv.iter().flat_map(|&v| [(v as f64 / 255.).powf(gs).powf(1. / gt) * 255.; 3]).collect();
      let (b, a) = (run!(zb, zcb, &g18, &ad, Gray8, Rgb8, &lv, 256, 768), run!(za, zca, &g18, &ad, Gray8, Rgb8, &lv, 256, 768));
      stats("Gray1.8->Adobe before", &b, &exact, &a); stats("Gray1.8->Adobe after", &a, &exact, &b); }
    { let sp = P::new_from_slice(&pp).unwrap(); let y = sp.colorant_matrix().v[1]; let (gs, gt) = (gm(&sp), gm(&P::new_from_slice(&g22).unwrap()));
      let exact: Vec<f64> = c.chunks_exact(3).map(|p| ((y[0]*(p[0] as f64 / 255.).powf(gs)+y[1]*(p[1] as f64 / 255.).powf(gs)+y[2]*(p[2] as f64 / 255.).powf(gs)).clamp(0., 1.)).powf(1. / gt) * 255.).collect();
      let (b, a) = (run!(zb, zcb, &pp, &g22, Rgb8, Gray8, &c, w, w), run!(za, zca, &pp, &g22, Rgb8, Gray8, &c, w, w));
      stats("ProPhoto->Gray2.2 before", &b, &exact, &a); stats("ProPhoto->Gray2.2 after", &a, &exact, &b); }
    println!("--- 8-bit routes that must not change");
    let lut = std::fs::read(std::env::var("LUT_ICC").unwrap()).unwrap();
    for (n, s, t) in [("ProPhoto->sRGB (matrix kernels)", &pp, &srgb), ("Adobe->sRGB (matrix kernels)", &ad, &srgb), ("P3->sRGB (matrix kernels)", &p3, &srgb), ("P3->Adobe (target route)", &p3, &ad), ("sRGB->Adobe (target route)", &srgb, &ad), ("LUT->sRGB", &lut, &srgb), ("sRGB->P3 (matrix or F32)", &srgb, &p3)] {
        let (b, a) = (run!(zb, zcb, s, t, Rgb8, Rgb8, &c, w, c.len()), run!(za, zca, s, t, Rgb8, Rgb8, &c, w, c.len()));
        println!("{n:<36} {} bytes, differ {}", c.len(), b.iter().zip(&a).filter(|(x, y)| x != y).count());
    }
    let (b, a) = (run!(zb, zcb, &lut, &ad, Rgb8, Rgb8, &c, w, c.len()), run!(za, zca, &lut, &ad, Rgb8, Rgb8, &c, w, c.len()));
    println!("LUT->Adobe (table route, parametric) {} bytes, differ {}, worst {}", c.len(), b.iter().zip(&a).filter(|(x, y)| x != y).count(), b.iter().zip(&a).map(|(x, y)| (*x as i32 - *y as i32).abs()).max().unwrap());
}
