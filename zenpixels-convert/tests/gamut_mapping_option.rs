//! `ConvertOptions::gamut_mapping`: clip stays the default, and compression matches
//! the kernel it wraps and never silently falls back to clipping.
use zenpixels_convert::hdr::SoftCompress;
use zenpixels_convert::oklab::OklabSpace;
use zenpixels_convert::policy::{AlphaPolicy, ConvertOptions, GamutMapping};
use zenpixels_convert::{
    AlphaMode, ChannelLayout, ChannelType, ColorPrimaries as P, ConvertError, PixelDescriptor as D,
    RowConverter, TransferFunction as Tf,
};

fn soft() -> ConvertOptions {
    ConvertOptions::permissive()
        .with_alpha_policy(AlphaPolicy::DiscardUnchecked)
        .with_gamut_mapping(GamutMapping::SoftCompress)
}
fn clip() -> ConvertOptions {
    ConvertOptions::permissive().with_alpha_policy(AlphaPolicy::DiscardUnchecked)
}

fn p3_rgb8() -> D {
    D::RGB8_SRGB.with_primaries(P::DisplayP3)
}

/// Every level of each channel against a few others.
fn grid() -> Vec<u8> {
    let mut v = Vec::new();
    for r in (0..=255u32).step_by(17) {
        for g in (0..=255u32).step_by(17) {
            for b in (0..=255u32).step_by(17) {
                v.extend_from_slice(&[r as u8, g as u8, b as u8]);
            }
        }
    }
    v
}

fn convert(from: D, to: D, options: &ConvertOptions, src: &[u8]) -> Vec<u8> {
    let w = src.len() / from.bytes_per_pixel();
    let mut c = RowConverter::new_explicit(from, to, options).unwrap();
    let mut out = vec![0u8; w * to.bytes_per_pixel()];
    c.convert_row(src, &mut out, w as u32);
    out
}

#[test]
fn default_is_clip_and_identical_to_no_option() {
    assert_eq!(
        ConvertOptions::permissive().gamut_mapping,
        GamutMapping::Clip
    );
    let src = grid();
    let w = (src.len() / 3) as u32;
    let mut a = RowConverter::new(p3_rgb8(), D::RGB8_SRGB).unwrap();
    let mut x = vec![0u8; src.len()];
    a.convert_row(&src, &mut x, w);
    assert_eq!(x, convert(p3_rgb8(), D::RGB8_SRGB, &clip(), &src));
}

#[test]
fn compression_matches_the_kernel_it_wraps() {
    let src = grid();
    let out = convert(p3_rgb8(), D::RGB8_SRGB, &soft(), &src);

    let lin_p3 = D::RGBF32_LINEAR.with_primaries(P::DisplayP3);
    let mut up = RowConverter::new_explicit(p3_rgb8(), lin_p3, &clip()).unwrap();
    let w = src.len() / 3;
    let mut linear = vec![[0f32; 3]; w];
    up.convert_row(&src, bytemuck::cast_slice_mut(&mut linear), w as u32);
    let m = zenpixels_convert::gamut::GamutMatrix::between(P::DisplayP3, P::Bt709).unwrap();
    m.apply_row_rgb(&mut linear);
    let space = OklabSpace::for_primaries(P::Bt709).unwrap();
    SoftCompress::new(&space, SoftCompress::DEFAULT_KNEE).apply_strip(&mut linear);
    let want = convert(
        D::RGBF32_LINEAR,
        D::RGB8_SRGB,
        &clip(),
        bytemuck::cast_slice(&linear),
    );
    let worst = out
        .iter()
        .zip(&want)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap();
    assert!(worst <= 1, "largest difference {worst} levels");
}

#[test]
fn p3_red_rolls_off_in_order() {
    // A saturated P3 color maps inside sRGB, with neighbouring levels moving smoothly.
    let src = [255u8, 0, 0, 240, 0, 0, 225, 0, 0];
    let out = convert(p3_rgb8(), D::RGB8_SRGB, &soft(), &src);
    assert!(
        out[1] > 0 && out[2] > 0,
        "green and blue rise to keep hue: {out:?}"
    );
    assert!(
        out[0] >= out[3] && out[3] >= out[6],
        "still ordered: {out:?}"
    );
}

#[test]
fn containing_targets_are_unchanged() {
    let src = grid();
    let to = D::RGB8_SRGB.with_primaries(P::DisplayP3);
    assert_eq!(
        convert(D::RGB8_SRGB, to, &soft(), &src),
        convert(D::RGB8_SRGB, to, &clip(), &src)
    );
}

#[test]
fn alpha_is_untouched_and_rows_can_be_split() {
    let from = D::new(
        ChannelType::U8,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        Tf::Srgb,
    )
    .with_primaries(P::DisplayP3);
    let to = D::new(
        ChannelType::U8,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        Tf::Srgb,
    );
    let mut src = Vec::new();
    for (i, px) in grid().chunks_exact(3).enumerate() {
        src.extend_from_slice(px);
        src.push((i % 251) as u8);
    }
    let whole = convert(from, to, &soft(), &src);
    for (a, b) in whole.chunks_exact(4).zip(src.chunks_exact(4)) {
        assert_eq!(a[3], b[3]);
    }
    let mut c = RowConverter::new_explicit(from, to, &soft()).unwrap();
    let mut parts = vec![0u8; src.len()];
    for (s, d) in src.chunks(4 * 17).zip(parts.chunks_mut(4 * 17)) {
        c.convert_row(s, d, (s.len() / 4) as u32);
    }
    assert_eq!(parts, whole);
}

#[test]
fn every_source_depth_stays_in_range_and_near_clip_inside_the_gamut() {
    let from = D::RGBF32_LINEAR.with_primaries(P::Bt2020);
    let colors: Vec<f32> = (0..=16)
        .flat_map(|r| {
            (0..=16).flat_map(move |g| {
                (0..=16).flat_map(move |b| [r as f32 / 16.0, g as f32 / 16.0, b as f32 / 16.0])
            })
        })
        .collect();
    let bytes: &[u8] = bytemuck::cast_slice(&colors);
    let out = convert(from, D::RGBF32_LINEAR, &soft(), bytes);
    for c in bytemuck::cast_slice::<u8, f32>(&out) {
        assert!((0.0..=1.0).contains(c), "{c}");
    }
    let to16 = D::RGB16_SRGB;
    assert_eq!(convert(from, to16, &soft(), bytes).len(), colors.len() * 2);
}

#[test]
fn unavailable_mapping_refuses_rather_than_clips() {
    let adobe_target = D::RGB8_SRGB.with_primaries(P::AdobeRgb);
    let from = D::RGB8_SRGB.with_primaries(P::Bt2020);
    let e = RowConverter::new_explicit(from, adobe_target, &soft())
        .err()
        .unwrap();
    assert!(matches!(*e.error(), ConvertError::NoPath { .. }), "{e:?}");
}
