//! Transcoding should work when any format the encoder takes can be reached.
use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, ColorPrimaries as P, PixelDescriptor as D,
    TransferFunction as Tf,
};
use zenpixels_convert::policy::{AlphaPolicy, ConvertOptions};

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
    let options = ConvertOptions::retain_channels()
        .with_alpha_policy(zenpixels_convert::AlphaPolicy::DiscardIfOpaque)
        .with_depth_policy(zenpixels_convert::DepthPolicy::Round)
        .with_luma(Some(zenpixels_convert::LumaCoefficients::Bt709))
        .with_alpha_policy(AlphaPolicy::DiscardUnchecked);
    use zenpixels_convert::{CandidateTarget, ConversionSpec, ConvertPlan};
    let spec = ConversionSpec::new(options);
    let reaches =
        |from: D, to: D| ConvertPlan::build(from.encoding(), to.encoding(), &spec).is_ok();
    let mut hidden = Vec::new();
    for src in sources() {
        for (name, supported) in encoders() {
            let targets: Vec<_> = supported
                .iter()
                .map(|d| CandidateTarget::new(d.encoding()))
                .collect();
            match ConvertPlan::select(src.encoding(), &targets, &spec) {
                Ok(selected) => assert!(reaches(src, supported[selected.candidate_index()])),
                Err(error) if supported.iter().any(|&t| reaches(src, t)) => {
                    hidden.push(format!("{src:?} into {name}: {error}"));
                }
                Err(_) => {}
            }
        }
    }
    assert!(hidden.is_empty(), "{}\n{}", hidden.len(), hidden.join("\n"));
}

#[test]
fn the_identity_candidate_remains_available_when_other_candidates_refuse() {
    use zenpixels_convert::{CandidateTarget, ConversionSpec, ConvertPlan, PlanError};
    let pq = D::RGB16_SRGB
        .with_transfer(Tf::Pq)
        .with_primaries(P::Bt2020);
    let sdr = CandidateTarget::new(D::RGB8_SRGB.encoding());
    let native = CandidateTarget::new(pq.encoding());
    let spec = ConversionSpec::new(ConvertOptions::retain_channels());
    let selected = ConvertPlan::select(pq.encoding(), &[sdr, native], &spec).unwrap();
    assert_eq!(selected.candidate_index(), 1);
    assert!(selected.into_plan().is_identity());
    let error = ConvertPlan::select(pq.encoding(), &[sdr], &spec).unwrap_err();
    assert!(matches!(
        error.error(),
        PlanError::NoCandidate { index: 0, .. }
    ));
}
