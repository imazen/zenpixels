// Paired before/after of the F32 routes through the public MoxCms adapter.
use moxcms::ColorProfile as P;
use std::sync::Arc;
use zenbench::prelude::*;
fn rng(n: usize, seed: u32, f: impl Fn(u32) -> f32) -> Arc<Vec<f32>> {
    let mut s = seed;
    Arc::new((0..n).map(|_| { s ^= s << 13; s ^= s >> 17; s ^= s << 5; f(s) }).collect())
}
macro_rules! arm { ($g:ident, $name:literal, $z:ident, $zc:ident, $s:expr, $t:expr, $from:ident, $to:ident, $clip:expr, $input:expr, $w:expr, $n:expr) => {{
    use $zc::cms::PluggableCms;
    let (s, t, input) = ($s.clone(), $t.clone(), $input.clone());
    let n: usize = $n; let w: usize = $w;
    $g.bench($name, move |b| {
        let o = $zc::ConvertOptions::retain_channels().with_clip_out_of_gamut($clip);
        let mut tr = $zc::cms_moxcms::MoxCms.build_source_transform($z::ColorProfileSource::Icc(&s), $z::ColorProfileSource::Icc(&t), $z::PixelFormat::$from, $z::PixelFormat::$to, &o).unwrap().unwrap();
        let input = input.clone();
        let mut out = vec![0f32; n];
        b.iter(move || { tr.try_transform_row(bytemuck::cast_slice(black_box(&input[..])), bytemuck::cast_slice_mut(&mut out), w as u32).unwrap(); black_box(out[0]) })
    });
}}}
fn f32route(suite: &mut Suite) {
    let no = |mut p: P| { p.cicp = None; Arc::new(p.encode().unwrap()) };
    let (pp, ad, srgb, p3) = (no(P::new_pro_photo_rgb()), no(P::new_adobe_rgb()), no(P::new_srgb()), no(P::new_display_p3()));
    let g = |x: f32| Arc::new(P::new_gray_with_gamma(x).encode().unwrap());
    let (g18, g22) = (g(1.8), g(2.2));
    let lut = Arc::new(std::fs::read(std::env::var("LUT_ICC").unwrap()).unwrap());
    let unit = |v: u32| (v >> 8) as f32 / (1u32 << 24) as f32;
    for w in [64usize, 1024, 4096] {
        let rgb = rng(w * 3, 0x9E3779B9, unit);
        let gray = rng(w, 0x2545F491, unit);
        let zeros = rng(w * 3, 0x12345679, move |v: u32| if v % 10 == 0 { 0.0 } else { unit(v.rotate_left(9)) });
        macro_rules! rgbgroup { ($n:literal, $s:ident, $t:ident, $clip:expr) => {{
            let (rgb, s, t) = (rgb.clone(), $s.clone(), $t.clone());
            suite.group(format!("{}/w{w}", $n), move |g2| { g2.throughput(Throughput::Elements(w as u64)); g2.baseline("before");
                arm!(g2, "before", zb, zcb, s, t, RgbF32, RgbF32, $clip, rgb, w, w * 3);
                arm!(g2, "after", za, zca, s, t, RgbF32, RgbF32, $clip, rgb, w, w * 3); });
        }}}
        rgbgroup!("source/prophoto_srgb_clip", pp, srgb, true);
        rgbgroup!("source/prophoto_srgb_noclip", pp, srgb, false);
        rgbgroup!("source/adobe_srgb_clip", ad, srgb, true);
        rgbgroup!("both/prophoto_adobe_clip", pp, ad, true);
        rgbgroup!("both/adobe_prophoto_clip", ad, pp, true);
        rgbgroup!("table/lut_adobe_clip", lut, ad, true);
        rgbgroup!("table/lut_prophoto_noclip", lut, pp, false);
        { let (rgb, s, t) = (zeros.clone(), pp.clone(), srgb.clone());
          suite.group(format!("source/prophoto_srgb_10pct_zero/w{w}"), move |g2| { g2.throughput(Throughput::Elements(w as u64)); g2.baseline("before");
            arm!(g2, "before", zb, zcb, s, t, RgbF32, RgbF32, true, rgb, w, w * 3);
            arm!(g2, "after", za, zca, s, t, RgbF32, RgbF32, true, rgb, w, w * 3); }); }
        rgbgroup!("control/p3_srgb_clip", p3, srgb, true);
        rgbgroup!("control/p3_adobe_clip", p3, ad, true);
        macro_rules! graygroup { ($n:literal, $s:ident, $t:ident, $to:ident, $tn:expr) => {{
            let (gray, s, t) = (gray.clone(), $s.clone(), $t.clone());
            suite.group(format!("{}/w{w}", $n), move |g2| { g2.throughput(Throughput::Elements(w as u64)); g2.baseline("before");
                arm!(g2, "before", zb, zcb, s, t, GrayF32, $to, true, gray, w, w * $tn);
                arm!(g2, "after", za, zca, s, t, GrayF32, $to, true, gray, w, w * $tn); });
        }}}
        graygroup!("gray/g18_g22", g18, g22, GrayF32, 1);
        graygroup!("gray/g18_srgb", g18, srgb, RgbF32, 3);
        let (rgb2, s, t) = (rgb.clone(), srgb.clone(), g22.clone());
        suite.group(format!("gray/srgb_g22/w{w}"), move |g2| { g2.throughput(Throughput::Elements(w as u64)); g2.baseline("before");
            arm!(g2, "before", zb, zcb, s, t, RgbF32, GrayF32, true, rgb2, w, w);
            arm!(g2, "after", za, zca, s, t, RgbF32, GrayF32, true, rgb2, w, w); });
    }
}
zenbench::main!(f32route);
