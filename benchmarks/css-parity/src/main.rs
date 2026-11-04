//! zenpixels against CSS Color 4's formulas and matrices, in f64. Results: docs/css-color-parity.md.
mod matrices;
use linear_srgb::default as d;
use matrices::*;
use zenpixels::{ChannelLayout as L, ChannelType as T, Color, ColorPrimaries as P, PixelDescriptor as D, TransferFunction as Tf};
use zenpixels_convert::gamut::conversion_matrix;
use zenpixels_convert::{ColorConvertExt, ConvertOptions};

fn css_lin(c: f64) -> f64 { let a = c.abs(); if a <= 0.04045 { c / 12.92 } else { c.signum() * ((a + 0.055) / 1.055).powf(2.4) } }
fn css_enc(c: f64) -> f64 { let a = c.abs(); if a > 0.0031308 { c.signum() * (1.055 * a.powf(1.0 / 2.4) - 0.055) } else { 12.92 * c } }
fn floats(c: &Color) -> Vec<f32> { bytemuck::cast_slice::<u8, f32>(c.as_bytes()).to_vec() }
fn rgbf(desc: D, v: [f32; 3]) -> Color { Color::try_new(desc, bytemuck::bytes_of(&v)).unwrap() }
fn unclipped() -> ConvertOptions { ConvertOptions::permissive().with_clip_out_of_gamut(false) }
fn oklab_desc() -> D { D::new(T::F32, L::Oklab, None, Tf::Unknown) }

fn curves() {
    fn report(name: &str, xs: &[f32], got: &[f32], exact: fn(f64) -> f64) {
        let (mut worst, mut at, mut u8s, mut u16s) = (0f64, 0f32, 0u64, 0u64);
        for (x, g) in xs.iter().zip(got) {
            let e = exact(f64::from(*x));
            let diff = (f64::from(*g) - e).abs();
            if diff > worst { worst = diff; at = *x; }
            let code = |v: f64, m: f64| (v.clamp(0.0, 1.0) * m + 0.5) as i64;
            u8s += u64::from(code(f64::from(*g), 255.0) != code(e, 255.0));
            u16s += u64::from(code(f64::from(*g), 65535.0) != code(e, 65535.0));
        }
        println!("{name}: {} samples, worst {worst:.2e} at {at}; {u8s} differ as u8, {u16s} as u16", xs.len());
    }
    let grid: Vec<f32> = (0..=2_000_000).map(|i| i as f32 / 2e6).collect();
    let mut v = grid.clone(); d::srgb_to_linear_slice(&mut v); report("sRGB decode", &grid, &v, css_lin);
    let mut v = grid.clone(); d::linear_to_srgb_slice(&mut v); report("sRGB encode", &grid, &v, css_enc);
    let (mut worst, mut wrong) = (0f64, 0);
    for b in 0..=255u8 {
        let g = d::srgb_u8_to_linear(b);
        worst = worst.max((f64::from(g) - css_lin(f64::from(b) / 255.0)).abs());
        wrong += i32::from(d::linear_to_srgb_u8(g) != b);
    }
    println!("sRGB u8: worst decode difference {worst:.2e}; {wrong} of 256 fail to round trip");
    let opts = unclipped();
    let a98 = rgbf(D::RGBF32_LINEAR.with_transfer(Tf::Gamma22).with_primaries(P::AdobeRgb), [0.5; 3]).try_convert_to(D::RGBF32_LINEAR.with_primaries(P::AdobeRgb), &opts).unwrap();
    println!("a98 decode of 0.5: ours {}, CSS 0.5^(563/256) = {:.7}", floats(&a98)[0], 0.5f64.powf(563.0 / 256.0));
    let bt = rgbf(D::RGBF32_LINEAR.with_transfer(Tf::Bt709).with_primaries(P::Bt2020), [0.5; 3]).try_convert_to(D::RGBF32_LINEAR.with_primaries(P::Bt2020), &opts).unwrap();
    println!("rec2020 decode of 0.5: CSS 0.5^2.4 = {:.5}; our only BT.2020-family curve, Bt709, gives {:.5}", 0.5f64.powf(2.4), floats(&bt)[0]);
}

fn matrices_against_chromaticities() {
    let bt709 = [(0.64, 0.33), (0.30, 0.60), (0.15, 0.06)];
    let spaces = [
        ("sRGB", P::Bt709, bt709),
        ("Display P3", P::DisplayP3, [(0.68, 0.32), (0.265, 0.69), (0.15, 0.06)]),
        ("BT.2020", P::Bt2020, [(0.708, 0.292), (0.17, 0.797), (0.131, 0.046)]),
        ("Adobe RGB", P::AdobeRgb, [(0.64, 0.33), (0.21, 0.71), (0.15, 0.06)]),
    ];
    let derived = rgb_to_xyz(bt709);
    let mut worst = 0f64;
    for i in 0..3 { for j in 0..3 { worst = worst.max((derived[i][j] - CSS_SRGB_TO_XYZ[i][j]).abs()); } }
    println!("sRGB to XYZ, chromaticities against the spec's rationals: worst {worst:.2e}");
    let mut all = 0f64;
    for (fname, f, fp) in spaces { for (tname, t, tp) in spaces {
        if fname == tname { continue; }
        let exact = matmul(&inv(&rgb_to_xyz(tp)), &rgb_to_xyz(fp));
        let ours = conversion_matrix(f, t).unwrap();
        for i in 0..3 { for j in 0..3 { all = all.max((f64::from(ours[i][j]) - exact[i][j]).abs()); } }
    } }
    println!("Gamut matrices among 4 spaces, 12 pairs: worst element off by {all:.2e}");
}

fn css_oklab(rgb: [f64; 3]) -> [f64; 3] {
    let m1 = matmul(&CSS_XYZ_TO_LMS, &CSS_SRGB_TO_XYZ);
    mul(&CSS_LMS_TO_OKLAB, mul(&m1, rgb).map(f64::cbrt))
}
fn css_oklab_inverse(lab: [f64; 3]) -> [f64; 3] {
    let m1 = matmul(&CSS_XYZ_TO_LMS, &CSS_SRGB_TO_XYZ);
    mul(&inv(&m1), mul(&inv(&CSS_LMS_TO_OKLAB), lab).map(|c| c * c * c))
}
fn de(a: [f64; 3], b: [f64; 3]) -> f64 { (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt() }

fn oklab() {
    let opts = unclipped();
    for (name, c) in [("white", [1.0; 3]), ("red", [1.0, 0.0, 0.0]), ("green", [0.0, 1.0, 0.0]), ("blue", [0.0, 0.0, 1.0]), ("gray 18%", [0.18; 3])] {
        let e = css_oklab(c);
        let g = floats(&rgbf(D::RGBF32_LINEAR, c.map(|v| v as f32)).try_convert_to(oklab_desc(), &opts).unwrap());
        let ours = [f64::from(g[0]), f64::from(g[1]), f64::from(g[2])];
        println!("Oklab {name}: CSS [{:.6} {:.6} {:.6}], ours [{:.6} {:.6} {:.6}], dE_OK {:.1e}", e[0], e[1], e[2], ours[0], ours[1], ours[2], de(e, ours));
    }
    let (mut worst, mut at) = (0f64, [0u16; 3]);
    for r in (0..=255u16).step_by(5) { for g in (0..=255u16).step_by(5) { for b in (0..=255u16).step_by(5) {
        let lin = [r, g, b].map(|v| css_lin(f64::from(v) / 255.0));
        let e = css_oklab(lin);
        let got = floats(&rgbf(D::RGBF32_LINEAR, lin.map(|v| v as f32)).try_convert_to(oklab_desc(), &opts).unwrap());
        let d = de(e, [f64::from(got[0]), f64::from(got[1]), f64::from(got[2])]);
        if d > worst { worst = d; at = [r, g, b]; }
    } } }
    println!("Oklab, 52^3 sRGB grid: worst dE_OK {worst:.2e} at {at:?}");
    let w = css_oklab([1.0; 3]);
    let gray = floats(&rgbf(D::RGBF32_LINEAR, [0.5; 3]).try_convert_to(oklab_desc(), &opts).unwrap());
    println!("Oklab chroma of neutrals: CSS at white {:.1e}; ours at linear 0.5 {:.1e}", w[1].hypot(w[2]), f64::from(gray[1]).hypot(f64::from(gray[2])));
}

/// CSS Color 4's binary-search gamut mapping to sRGB, from its pseudocode. Not a browser.
fn css_gamut_map(linear: [f64; 3]) -> [f64; 3] {
    const JND: f64 = 0.02;
    const EPS: f64 = 0.0001;
    let enc = |c: [f64; 3]| c.map(css_enc);
    let clip = |c: [f64; 3]| enc(c).map(|v| v.clamp(0.0, 1.0));
    let lab_of_enc = |c: [f64; 3]| css_oklab(c.map(css_lin));
    let origin = css_oklab(linear);
    let (l, c0, h) = (origin[0], origin[1].hypot(origin[2]), origin[2].atan2(origin[1]));
    if l >= 1.0 { return [1.0; 3]; }
    if l <= 0.0 { return [0.0; 3]; }
    let in_gamut = |c: [f64; 3]| enc(c).iter().all(|v| (-EPS..=1.0 + EPS).contains(v));
    if in_gamut(linear) { return enc(linear); }
    let at = |c: f64| css_oklab_inverse([l, c * h.cos(), c * h.sin()]);
    let (mut min, mut max, mut min_in) = (0.0, c0, true);
    let mut clipped = clip(linear);
    if de(lab_of_enc(clipped), origin) < JND { return clipped; }
    while max - min > EPS {
        let chroma = (min + max) / 2.0;
        let current = at(chroma);
        if min_in && in_gamut(current) { min = chroma; continue; }
        clipped = clip(current);
        let e = de(lab_of_enc(clipped), css_oklab(current));
        if e < JND {
            if JND - e < EPS { return clipped; }
            min_in = false;
            min = chroma;
        } else { max = chroma; }
    }
    clipped
}

fn gamut_mapping() {
    let u8s = |c: [f64; 3]| c.map(|v| (v * 255.0 + 0.5) as u8);
    let hue = |l: [f64; 3]| l[2].atan2(l[1]).to_degrees();
    let cases = [
        ("display-p3 1 0 0", P::DisplayP3, [1.0f32, 0.0, 0.0]),
        ("display-p3 0 1 0", P::DisplayP3, [0.0, 1.0, 0.0]),
        ("display-p3 0 0 1", P::DisplayP3, [0.0, 0.0, 1.0]),
        ("rec2020 linear 1 0 0", P::Bt2020, [1.0, 0.0, 0.0]),
        ("rec2020 linear 0 1 0", P::Bt2020, [0.0, 1.0, 0.0]),
        ("rec2020 linear 0 0 1", P::Bt2020, [0.0, 0.0, 1.0]),
    ];
    for (name, p, v) in cases {
        let source = rgbf(D::RGBF32_LINEAR.with_primaries(p), v);
        let linear = floats(&source.try_convert_to(D::RGBF32_LINEAR, &unclipped()).unwrap());
        let linear = [f64::from(linear[0]), f64::from(linear[1]), f64::from(linear[2])];
        let origin = css_oklab(linear);
        let clip_opts = ConvertOptions::permissive();
        let ours = source.try_convert_to(D::RGB8_SRGB, &clip_opts).unwrap();
        let ob = ours.as_bytes();
        let ours_lab = css_oklab([ob[0], ob[1], ob[2]].map(|c| css_lin(f64::from(c) / 255.0)));
        let mapped = css_gamut_map(linear);
        let mapped_lab = css_oklab(mapped.map(css_lin));
        println!("{name}: ours {:?} hue {:+.1} dE_OK {:.3}; CSS map {:?} hue {:+.1} dE_OK {:.3}", [ob[0], ob[1], ob[2]], hue(ours_lab) - hue(origin), de(ours_lab, origin), u8s(mapped), hue(mapped_lab) - hue(origin), de(mapped_lab, origin));
    }
}

fn hdr() {
    let opts = unclipped();
    let bt2020 = D::RGBF32_LINEAR.with_primaries(P::Bt2020);
    let pq = bt2020.with_transfer(Tf::Pq);
    let hlg = bt2020.with_transfer(Tf::Hlg);
    let enc = |d: D, v: f32| floats(&rgbf(bt2020, [v; 3]).try_convert_to(d, &opts).unwrap())[0];
    println!("203 cd/m2 in PQ: {:.4} (CSS: 58%)", enc(pq, 203.0 / 10_000.0));
    println!("203 cd/m2 in HLG: {:.4} (CSS: 75%)", enc(hlg, (203.0f32 / 1_000.0).powf(1.0 / 1.2)));
    let w = Color::srgb8(255, 255, 255).try_convert_to(pq, &opts).unwrap();
    println!("sRGB white straight to PQ: {:.4}, which is {:.0} cd/m2 (CSS: 0.58, 203 cd/m2)", floats(&w)[0], 10_000.0 * zenpq(floats(&w)[0]));
    let linear = rgbf(bt2020, [1.0; 3]).try_convert_to(pq, &opts).unwrap();
    println!("linear 1.0 to PQ: {:.4} (CSS rec2100-linear 1.0: 0.58)", floats(&linear)[0]);
}
fn zenpq(v: f32) -> f32 { floats(&rgbf(D::RGBF32_LINEAR.with_transfer(Tf::Pq), [v; 3]).try_convert_to(D::RGBF32_LINEAR, &unclipped()).unwrap())[0] }

fn main() {
    curves();
    matrices_against_chromaticities();
    oklab();
    gamut_mapping();
    hdr();
}
