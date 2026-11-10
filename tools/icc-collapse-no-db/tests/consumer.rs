#![forbid(unsafe_code)]
use std::sync::Arc;
use zenpixels::{Cicp, ColorContext, PixelBuffer, PixelDescriptor, PixelFormat, PixelSlice};
use zenpixels_convert::icc_profiles::{DISPLAY_P3_V4, SynthesizedIcc};
use zenpixels_convert::{PixelBufferLoadBearingExt, PixelSliceLoadBearingExt};

fn assert_reduction(context: ColorContext, expected: PixelFormat, keep_icc: bool) {
    let context = Arc::new(context);
    let row = [64u8, 64, 64, 255, 128, 128, 128, 255];
    let mut padded = Vec::new();
    for _ in 0..2 {
        padded.extend_from_slice(&row);
        padded.extend_from_slice(&[0xa5; 4]);
    }
    let pixels = PixelSlice::new(&padded, 2, 2, 12, PixelDescriptor::RGBA8_SRGB)
        .unwrap()
        .with_color_context(context.clone());
    let output = pixels.try_reduce_to_load_bearing_format().unwrap();
    let mut buffer = PixelBuffer::from_vec(row.repeat(2), 2, 2, PixelDescriptor::RGBA8_SRGB)
        .unwrap()
        .with_color_context(context.clone());
    buffer.reduce_to_load_bearing_format_in_place(true);
    for view in [output.as_slice(), buffer.as_slice()] {
        assert_eq!(view.descriptor().pixel_format(), expected);
        let actual_icc = view.color_context().and_then(|c| c.icc.as_deref());
        assert_eq!(
            actual_icc,
            if keep_icc {
                context.icc.as_deref()
            } else {
                None
            }
        );
        for y in 0..2 {
            let expected_row: &[u8] = if expected == PixelFormat::Gray8 {
                &[64, 128]
            } else {
                &[64, 64, 64, 128, 128, 128]
            };
            assert_eq!(view.row(y), expected_row);
        }
    }
}

#[test]
fn feature_isolation_proves_gray_database_is_absent() {
    for cicp in [Cicp::DISPLAY_P3, Cicp::new(9, 16, 0, true)] {
        assert_eq!(
            SynthesizedIcc::for_gray_cicp(cicp),
            SynthesizedIcc::NeedsCms
        );
    }
}

#[test]
fn conflicting_hint_preserves_valid_icc_and_rgb() {
    let mut context = ColorContext::from_icc(DISPLAY_P3_V4);
    context.cicp = Some(Cicp::new(12, 4, 0, true));
    assert_reduction(context, PixelFormat::Rgb8, true);
}

#[test]
fn qualified_rgb_stays_rgb_when_gray_synthesis_is_unavailable() {
    assert_reduction(
        ColorContext::from_icc(DISPLAY_P3_V4),
        PixelFormat::Rgb8,
        true,
    );
}

#[test]
fn unusable_icc_keeps_rgb_and_profile_with_or_without_cicp() {
    // A secondary hint cannot authorize replacing an unusable current ICC.
    // Opaque-alpha removal remains exact and is independent of color.
    for len in [0, 1, 8, 127, 128, 129] {
        for companion in [None, Some(Cicp::SRGB), Some(Cicp::DISPLAY_P3)] {
            let mut context = ColorContext::from_icc(vec![0u8; len]);
            context.cicp = companion;
            assert_reduction(context, PixelFormat::Rgb8, true);
        }
    }
}
