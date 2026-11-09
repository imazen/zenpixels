//! Transcoding should work when any format the encoder takes can be reached.
use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, ColorPrimaries as P, PixelDescriptor as D,
    TransferFunction as Tf,
};
use zenpixels_convert::policy::{AlphaPolicy, ConvertOptions};
use zenpixels_convert::{ConvertIntent, RowConverter};

fn sources() -> Vec<D> {
    let formats: [(ChannelType, ChannelLayout, Option<AlphaMode>); 14] = [
        (ChannelType::U8, ChannelLayout::Rgb, None),
        (
            ChannelType::U8,
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
        ),
        (
            ChannelType::U8,
            ChannelLayout::Bgra,
            Some(AlphaMode::Straight),
        ),
        (ChannelType::U8, ChannelLayout::Bgr, None),
        (ChannelType::U8, ChannelLayout::Gray, None),
        (
            ChannelType::U8,
            ChannelLayout::GrayAlpha,
            Some(AlphaMode::Straight),
        ),
        (ChannelType::U16, ChannelLayout::Rgb, None),
        (
            ChannelType::U16,
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
        ),
        (ChannelType::U16, ChannelLayout::Gray, None),
        (
            ChannelType::F16,
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
        ),
        (ChannelType::F32, ChannelLayout::Rgb, None),
        (
            ChannelType::F32,
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
        ),
        (
            ChannelType::F32,
            ChannelLayout::Rgba,
            Some(AlphaMode::Premultiplied),
        ),
        (ChannelType::F32, ChannelLayout::Gray, None),
    ];
    let mut out = Vec::new();
    for (ct, layout, alpha) in formats {
        for tf in [
            Tf::Srgb,
            Tf::Linear,
            Tf::Bt709,
            Tf::Gamma22,
            Tf::Pq,
            Tf::Hlg,
            Tf::Unknown,
        ] {
            for p in [P::Bt709, P::DisplayP3, P::Bt2020, P::AdobeRgb, P::Unknown] {
                if let Ok(d) = D::try_new(ct, layout, alpha, tf) {
                    out.push(d.with_primaries(p));
                }
            }
        }
    }
    out
}

fn encoders() -> [(&'static str, Vec<D>); 5] {
    let base = vec![D::RGB8_SRGB, D::RGBA8_SRGB, D::GRAY8_SRGB, D::BGRA8_SRGB];
    let mut deep = base.clone();
    deep.extend([
        D::RGB16_SRGB,
        D::RGBA16_SRGB,
        D::GRAY16_SRGB,
        D::RGBF32_LINEAR,
        D::RGBAF32_LINEAR,
        D::GRAYF32_LINEAR,
    ]);
    [
        ("8-bit RGB only", vec![D::RGB8_SRGB]),
        ("RGBA8 only", vec![D::RGBA8_SRGB]),
        ("8-bit", base),
        ("deep", deep),
        ("linear f32 only", vec![D::RGBAF32_LINEAR]),
    ]
}

/// PQ and HLG sources need a peak to reach SDR. A cheap
/// format that fails that way mustn't hide a dearer one that works: 80 of these 2,450 pairs
/// did before candidates were checked.
#[test]
fn best_match_never_picks_a_dead_format_over_a_live_one() {
    let options = ConvertOptions::permissive().with_alpha_policy(AlphaPolicy::DiscardUnchecked);
    let reaches = |from: D, to: D| RowConverter::new_explicit(from, to, &options).is_ok();
    let mut hidden = Vec::new();
    for src in sources() {
        for (name, supported) in encoders() {
            let target = ConvertIntent::Fastest.best_match(src, &supported).unwrap();
            if !reaches(src, target) && supported.iter().any(|&t| reaches(src, t)) {
                hidden.push(format!("{src:?} into {name}: chose {target:?}"));
            }
        }
    }
    assert!(hidden.is_empty(), "{}\n{}", hidden.len(), hidden.join("\n"));
}

#[test]
fn a_reachable_format_is_still_the_cheapest_one() {
    assert_eq!(
        ConvertIntent::Fastest.best_match(D::RGB8_SRGB, &[D::RGBAF32_LINEAR, D::RGB8_SRGB]),
        Some(D::RGB8_SRGB)
    );
    // PQ needs an anchor for linear and peaks for SDR. Its native format remains reachable.
    let pq = D::RGB16_SRGB
        .with_transfer(Tf::Pq)
        .with_primaries(P::Bt2020);
    assert_eq!(
        ConvertIntent::Fastest.best_match(pq, &[D::RGB8_SRGB, D::RGBF32_LINEAR, pq]),
        Some(pq)
    );
    // With nothing reachable, the cheapest comes back, so converting explains why.
    assert_eq!(
        ConvertIntent::Fastest.best_match(pq, &[D::RGB8_SRGB]),
        Some(D::RGB8_SRGB)
    );
}
