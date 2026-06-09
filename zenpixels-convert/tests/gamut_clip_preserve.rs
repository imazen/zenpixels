//! End-to-end: `GamutClip::Preserve` flows through the real conversion
//! pipeline when narrowing Display P3 → sRGB.
//!
//! These are deliberately *behavioral* (not plan-introspection) tests: the
//! `ConvertStep` enum is crate-private, so the only honest proof that the snap
//! step actually runs is that it changes the pixels it must change (out-of-gamut
//! reds) and leaves alone the pixels it must not (in-gamut colors). If the
//! un-fusing in `insert_gamut_clip_steps` failed to fire, `Preserve` would equal
//! `PerChannel` for the out-of-gamut pixel and `differs_for_out_of_gamut_red`
//! would fail loudly.

use zenpixels_convert::{ColorPrimaries, ConvertOptions, GamutClip, PixelDescriptor, RowConverter};

/// Display P3 source: u8, sRGB transfer, P3 primaries.
fn p3_rgb8() -> PixelDescriptor {
    PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3)
}

/// sRGB destination: u8, sRGB transfer, BT.709 (sRGB) primaries.
fn srgb_rgb8() -> PixelDescriptor {
    PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::Bt709)
}

fn convert(px: [u8; 3], gamut_clip: GamutClip) -> [u8; 3] {
    let opts = ConvertOptions::permissive().with_gamut_clip(gamut_clip);
    let mut conv = RowConverter::new_explicit(p3_rgb8(), srgb_rgb8(), &opts).unwrap();
    let mut out = [0u8; 3];
    conv.convert_row(&px, &mut out, 1);
    out
}

/// A pure Display P3 red is outside the sRGB gamut. `Preserve` must produce a
/// different result than the per-channel hard clip — proof the snap step ran.
#[test]
fn differs_for_out_of_gamut_red() {
    let hard = convert([255, 0, 0], GamutClip::PerChannel);
    let soft = convert([255, 0, 0], GamutClip::Preserve);
    assert_ne!(
        hard, soft,
        "Preserve must change an out-of-gamut P3 red vs hard clip \
         (hard={hard:?} soft={soft:?})"
    );
    // The snap preserves lightness by trading saturation, so the secondary
    // channels rise above the hard clip's crushed zero.
    assert!(
        soft[1] >= hard[1] && soft[2] >= hard[2],
        "snap should lift the crushed channels, not deepen them \
         (hard={hard:?} soft={soft:?})"
    );
}

/// An in-gamut color (a muted P3 green that maps inside sRGB) must be identical
/// under both policies — `Preserve` is detection-based and never desaturates a
/// pixel that already fits.
#[test]
fn identical_for_in_gamut_color() {
    let muted = [100u8, 140, 90];
    let hard = convert(muted, GamutClip::PerChannel);
    let soft = convert(muted, GamutClip::Preserve);
    assert_eq!(
        hard, soft,
        "in-gamut pixel must be untouched by Preserve (hard={hard:?} soft={soft:?})"
    );
}

/// Same-primaries conversion: there is nothing out of gamut, so `Preserve` is a
/// no-op relative to `PerChannel` for every pixel.
#[test]
fn no_op_when_primaries_match() {
    let opts_hard = ConvertOptions::permissive();
    let opts_soft = ConvertOptions::permissive().with_gamut_clip(GamutClip::Preserve);
    let from = srgb_rgb8();
    let to = srgb_rgb8();
    let mut hc = RowConverter::new_explicit(from, to, &opts_hard).unwrap();
    let mut sc = RowConverter::new_explicit(from, to, &opts_soft).unwrap();
    for px in [[255u8, 0, 0], [10, 200, 30], [128, 128, 128]] {
        let mut a = [0u8; 3];
        let mut b = [0u8; 3];
        hc.convert_row(&px, &mut a, 1);
        sc.convert_row(&px, &mut b, 1);
        assert_eq!(a, b, "same-primaries Preserve must be a no-op for {px:?}");
    }
}

/// The RGBA path snaps color while passing alpha through untouched.
#[test]
fn rgba_preserves_alpha() {
    let from = PixelDescriptor::RGBA8_SRGB.with_primaries(ColorPrimaries::DisplayP3);
    let to = PixelDescriptor::RGBA8_SRGB.with_primaries(ColorPrimaries::Bt709);
    let opts = ConvertOptions::permissive().with_gamut_clip(GamutClip::Preserve);
    let mut conv = RowConverter::new_explicit(from, to, &opts).unwrap();
    let mut out = [0u8; 4];
    conv.convert_row(&[255, 0, 0, 137], &mut out, 1);
    assert_eq!(out[3], 137, "alpha must pass through unchanged: {out:?}");
    // and the color must have been snapped (differs from a crushed hard clip)
    let opts_hard = ConvertOptions::permissive();
    let mut hc = RowConverter::new_explicit(from, to, &opts_hard).unwrap();
    let mut hard = [0u8; 4];
    hc.convert_row(&[255, 0, 0, 137], &mut hard, 1);
    assert_ne!(&out[..3], &hard[..3], "RGBA color should be snapped");
}

/// The headline fix: two *different* bright P3 reds that the per-channel clip
/// crushes to the *same* flat sRGB red stay distinct under `Preserve` — this is
/// the washed-out-poppy regression (detail collapse) the snap exists to prevent.
#[test]
fn distinct_reds_survive_snap_but_collapse_under_hard_clip() {
    let a = [255u8, 0, 0];
    let b = [255u8, 40, 30];
    let hard_a = convert(a, GamutClip::PerChannel);
    let hard_b = convert(b, GamutClip::PerChannel);
    assert_eq!(
        hard_a, hard_b,
        "hard clip is expected to collapse these reds"
    );

    let soft_a = convert(a, GamutClip::Preserve);
    let soft_b = convert(b, GamutClip::Preserve);
    assert_ne!(
        soft_a, soft_b,
        "Preserve must keep distinct reds distinct (a={soft_a:?} b={soft_b:?})"
    );
}
