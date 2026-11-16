//! Third-party oracle (`palette`) and Safdar et al. (2017) verification for
//! `Oklch`, `Lch` (CIE 1976 D50), `Jzazbz`, `Jzczhz`, `Hsl`, and `Hsv`.

use palette::chromatic_adaptation::AdaptIntoUnclamped;
use palette::{
    FromColor, Hsl as PalHsl, Hsv as PalHsv, IntoColor, Lch as PalLch, Oklch as PalOklch, Srgb,
};
use zenpixels::hdr::DiffuseWhite;
use zenpixels::policy::{AlphaPolicy, ConvertOptions};
use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, Color, ColorPrimaries, PixelBuffer, PixelDescriptor,
    PixelFormat, TransferFunction,
};
use zenpixels_convert::{PixelBufferConvertExt, RowConverter};

const RGBF32_SRGB: PixelDescriptor = PixelDescriptor::RGBF32.with_transfer(TransferFunction::Srgb);

fn convert_f32_3(from: PixelDescriptor, to: PixelDescriptor, px: [f32; 3]) -> [f32; 3] {
    let mut conv = RowConverter::new(from, to).unwrap();
    let mut out = [0.0_f32; 3];
    conv.convert_row(
        bytemuck::cast_slice(&px),
        bytemuck::cast_slice_mut(&mut out),
        1,
    );
    out
}

#[test]
fn oklch_matches_palette_oracle_and_roundtrips() {
    let samples = [
        [0.0_f32, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5],
        [0.85, 0.15, 0.25],
        [0.10, 0.75, 0.35],
        [0.20, 0.30, 0.95],
    ];
    for rgb in samples {
        let got = convert_f32_3(RGBF32_SRGB, PixelDescriptor::OKLCHF32, rgb);
        let oracle: PalOklch<f32> = PalOklch::from_color(Srgb::new(rgb[0], rgb[1], rgb[2]));
        assert!(
            (got[0] - oracle.l).abs() < 5e-4,
            "Oklch L for {rgb:?}: got {got:?}, oracle ({}, {}, {})",
            oracle.l,
            oracle.chroma,
            oracle.hue.into_positive_degrees()
        );
        assert!(
            (got[1] - oracle.chroma).abs() < 5e-4,
            "Oklch C for {rgb:?}: got {got:?}, oracle {}",
            oracle.chroma
        );
        if oracle.chroma > 1e-4 {
            let dh = (got[2] - oracle.hue.into_positive_degrees()).abs();
            let dh = dh.min(360.0 - dh);
            assert!(
                dh < 0.05,
                "Oklch h for {rgb:?}: got {}, oracle {}",
                got[2],
                oracle.hue.into_positive_degrees()
            );
        }

        let back = convert_f32_3(PixelDescriptor::OKLCHF32, RGBF32_SRGB, got);
        for k in 0..3 {
            assert!(
                (back[k] - rgb[k]).abs() < 5e-4,
                "Oklch roundtrip {rgb:?} -> {got:?} -> {back:?}"
            );
        }
    }

    // Powerless / NaN hue at C == 0 produces clean neutral gray, never NaN.
    let nan_hue = [0.6_f32, 0.0, f32::NAN];
    let gray_rgb = convert_f32_3(
        PixelDescriptor::OKLCHF32,
        PixelDescriptor::RGBF32_LINEAR,
        nan_hue,
    );
    assert!(gray_rgb.iter().all(|v| v.is_finite()));
    assert!((gray_rgb[0] - gray_rgb[1]).abs() < 1e-4 && (gray_rgb[1] - gray_rgb[2]).abs() < 1e-4);
}

#[test]
fn cie_lch_d50_matches_palette_and_converts_directly_with_lab() {
    let samples = [
        [0.0_f32, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.8, 0.2, 0.1],
        [0.1, 0.7, 0.4],
        [0.2, 0.3, 0.9],
    ];
    for rgb in samples {
        let lch = convert_f32_3(RGBF32_SRGB, PixelDescriptor::LCHF32, rgb);
        let xyz_d65: palette::Xyz<palette::white_point::D65, f32> =
            Srgb::new(rgb[0], rgb[1], rgb[2]).into_color();
        let xyz_d50: palette::Xyz<palette::white_point::D50, f32> = xyz_d65.adapt_into_unclamped();
        let oracle: PalLch<palette::white_point::D50, f32> = PalLch::from_color(xyz_d50);
        assert!(
            (lch[0] - oracle.l).abs() < 0.05,
            "L* for {rgb:?}: got {lch:?}, oracle {}",
            oracle.l
        );
        assert!(
            (lch[1] - oracle.chroma).abs() < 0.08,
            "C* for {rgb:?}: got {lch:?}, oracle {}",
            oracle.chroma
        );
        if oracle.chroma > 0.5 {
            let dh = (lch[2] - oracle.hue.into_positive_degrees()).abs();
            let dh = dh.min(360.0 - dh);
            assert!(
                dh < 0.15,
                "h° for {rgb:?}: got {}, oracle {}",
                lch[2],
                oracle.hue.into_positive_degrees()
            );
        }

        let back = convert_f32_3(PixelDescriptor::LCHF32, RGBF32_SRGB, lch);
        for k in 0..3 {
            assert!(
                (back[k] - rgb[k]).abs() < 2e-4,
                "LCh roundtrip {rgb:?} -> {lch:?} -> {back:?}"
            );
        }

        // Direct LabF32 <-> LchF32 works without a CMS!
        let labf32 = PixelFormat::from_parts(ChannelType::F32, ChannelLayout::Lab, None)
            .unwrap()
            .descriptor();
        let lab = convert_f32_3(PixelDescriptor::LCHF32, labf32, lch);
        let lch_back = convert_f32_3(labf32, PixelDescriptor::LCHF32, lab);
        assert!((lch_back[0] - lch[0]).abs() < 1e-5);
        assert!((lch_back[1] - lch[1]).abs() < 1e-5);
    }
}

#[test]
fn jzazbz_and_jzczhz_cover_sdr_anchors_and_10000_nit_pq_hdr() {
    // 1. Exact black [0, 0, 0] -> [0, 0, 0].
    let black = convert_f32_3(
        PixelDescriptor::RGBF32_LINEAR,
        PixelDescriptor::JZAZBZF32,
        [0.0, 0.0, 0.0],
    );
    assert!(black[0].abs() < 1e-7 && black[1].abs() < 1e-7 && black[2].abs() < 1e-7);

    // 2. Default SDR 203 cd/m² D65 white -> Safdar et al. Jz ≈ 0.22206, chroma ≈ 0.
    let sdr_white = convert_f32_3(
        PixelDescriptor::RGBF32_LINEAR,
        PixelDescriptor::JZAZBZF32,
        [1.0, 1.0, 1.0],
    );
    assert!(
        (sdr_white[0] - 0.222_065).abs() < 5e-4,
        "203 nit Jz: {sdr_white:?}"
    );
    assert!(sdr_white[1].abs() < 5e-4 && sdr_white[2].abs() < 5e-4);

    // 3. Explicit ColorContext::diffuse_white (100 nits vs 203 nits) scales Jz lightness.
    let buf_100 = PixelBuffer::from_vec(
        bytemuck::cast_slice(&[1.0_f32, 1.0, 1.0]).to_vec(),
        1,
        1,
        PixelDescriptor::RGBF32_LINEAR,
    )
    .unwrap()
    .with_diffuse_white(DiffuseWhite::new(100.0));
    let out_100 = buf_100.convert_to(PixelDescriptor::JZAZBZF32).unwrap();
    let slice_100 = out_100.as_slice();
    let jz_100: &[f32] = bytemuck::cast_slice(slice_100.row(0));
    assert!(
        jz_100[0] < sdr_white[0] - 0.04,
        "100 nit Jz ({}) must be darker than 203 nit Jz ({})",
        jz_100[0],
        sdr_white[0]
    );

    // 4. Direct BT.2100 PQ HDR (up to 10,000 nits!) <-> Jzazbz <-> Jzczhz round-trip without tone-map clipping.
    let jz_bt2020 = PixelDescriptor::JZCZHZF32.with_primaries(ColorPrimaries::Bt2020);
    let hdr_pq_px = [0.75_f32, 0.52, 0.38]; // ~1,000+ nit chromatic HDR highlight
    let jzczhz = convert_f32_3(PixelDescriptor::RGBF32_BT2100_PQ, jz_bt2020, hdr_pq_px);
    assert!(jzczhz[0] > 0.3 && jzczhz[1] > 0.01);
    let pq_back = convert_f32_3(jz_bt2020, PixelDescriptor::RGBF32_BT2100_PQ, jzczhz);
    for k in 0..3 {
        assert!(
            (pq_back[k] - hdr_pq_px[k]).abs() < 2e-5,
            "PQ <-> Jzczhz HDR roundtrip: {hdr_pq_px:?} -> {jzczhz:?} -> {pq_back:?}"
        );
    }
}

#[test]
fn hsl_and_hsv_match_palette_across_float_integer_and_matte_composite() {
    let samples = [
        [0.0_f32, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.75, 0.35, 0.20],
        [0.15, 0.55, 0.85],
    ];
    for rgb in samples {
        let hsl = convert_f32_3(RGBF32_SRGB, PixelDescriptor::HSLF32_SRGB, rgb);
        let hsv = convert_f32_3(RGBF32_SRGB, PixelDescriptor::HSVF32_SRGB, rgb);
        let o_hsl: PalHsl<_, f32> = PalHsl::from_color(Srgb::new(rgb[0], rgb[1], rgb[2]));
        let o_hsv: PalHsv<_, f32> = PalHsv::from_color(Srgb::new(rgb[0], rgb[1], rgb[2]));

        assert!(
            (hsl[1] - o_hsl.saturation).abs() < 1e-5 && (hsl[2] - o_hsl.lightness).abs() < 1e-5
        );
        assert!((hsv[1] - o_hsv.saturation).abs() < 1e-5 && (hsv[2] - o_hsv.value).abs() < 1e-5);
        if o_hsl.saturation > 1e-4 {
            let dh = (hsl[0] - o_hsl.hue.into_positive_degrees()).abs();
            assert!(dh.min(360.0 - dh) < 1e-3);
        }
        if o_hsv.saturation > 1e-4 {
            let dh = (hsv[0] - o_hsv.hue.into_positive_degrees()).abs();
            assert!(dh.min(360.0 - dh) < 1e-3);
        }

        let back_hsl = convert_f32_3(PixelDescriptor::HSLF32_SRGB, RGBF32_SRGB, hsl);
        let back_hsv = convert_f32_3(PixelDescriptor::HSVF32_SRGB, RGBF32_SRGB, hsv);
        for k in 0..3 {
            assert!((back_hsl[k] - rgb[k]).abs() < 1e-5);
            assert!((back_hsv[k] - rgb[k]).abs() < 1e-5);
        }
    }

    // U16 HSV <-> RGB8_SRGB round-trips within 1 code.
    let hsv16 = PixelDescriptor::try_new(
        ChannelType::U16,
        ChannelLayout::Hsv,
        None,
        TransferFunction::Srgb,
    )
    .unwrap();
    let mut to_hsv16 = RowConverter::new(PixelDescriptor::RGB8_SRGB, hsv16).unwrap();
    let mut from_hsv16 = RowConverter::new(hsv16, PixelDescriptor::RGB8_SRGB).unwrap();
    let rgb8_in = [210u8, 65, 140, 20, 190, 240, 128, 128, 128];
    let mut mid_u16 = [0u8; 18];
    let mut rgb8_out = [0u8; 9];
    to_hsv16.convert_row(&rgb8_in, &mut mid_u16, 3);
    from_hsv16.convert_row(&mid_u16, &mut rgb8_out, 3);
    for (&a, &b) in rgb8_in.iter().zip(&rgb8_out) {
        assert!(
            a.abs_diff(b) <= 1,
            "HSV16 roundtrip: {rgb8_in:?} vs {rgb8_out:?}"
        );
    }

    // Matte compositing on polar HSLAF32_SRGB -> RGB8_SRGB composites in linear RGB, NOT on hue angle!
    // 50% translucent red (H = 0°, S = 1, L = 0.5, A = 0.5) over opaque white must stay pink-red (R > G == B),
    // never shifting hue toward cyan/yellow!
    let hsla_red = [0.0_f32, 1.0, 0.5, 0.5];
    let opts = ConvertOptions::permissive().with_alpha_policy(AlphaPolicy::CompositeOnto {
        background: Color::srgb8(255, 255, 255),
        reference_white: None,
    });
    let mut matte_conv = RowConverter::new_explicit(
        PixelDescriptor::HSLAF32_SRGB,
        PixelDescriptor::RGB8_SRGB,
        &opts,
    )
    .unwrap();
    let mut pink_rgb = [0u8; 3];
    matte_conv.convert_row(bytemuck::cast_slice(&hsla_red), &mut pink_rgb, 1);
    assert_eq!(pink_rgb[0], 255, "red over white R: {pink_rgb:?}");
    assert_eq!(
        pink_rgb[1], pink_rgb[2],
        "red over white G == B: {pink_rgb:?}"
    );
    assert!(
        pink_rgb[1] > 170 && pink_rgb[1] < 200,
        "linear-light pink G/B: {pink_rgb:?}"
    );

    // Premultiplied alpha is rejected on all polar layouts.
    assert!(
        PixelDescriptor::HSLAF32_SRGB
            .with_alpha(Some(AlphaMode::Premultiplied))
            .validate()
            .is_err()
    );
}
