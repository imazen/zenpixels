//! BGR8 converts through RGB8. CMYK has no numeric conversion and must refuse.

use zenpixels::{
    ChannelLayout, ChannelType, ColorPrimaries as P, PixelDescriptor as D, TransferFunction as T,
};
use zenpixels_convert::policy::{AlphaPolicy, ConvertOptions};

const SRGB_BGR: D = D::new(ChannelType::U8, ChannelLayout::Bgr, None, T::Srgb);

#[test]
fn bgr8_swaps_channels_both_ways() {
    let bgr = [10u8, 20, 30, 40, 50, 60];
    let mut out = [0u8; 6];
    crate::test_conversion::new_worker(SRGB_BGR, D::RGB8_SRGB)
        .unwrap()
        .try_convert_row(&bgr, &mut out, 2)
        .unwrap();
    assert_eq!(out, [30, 20, 10, 60, 50, 40]);
    let mut back = [0u8; 6];
    crate::test_conversion::new_worker(D::RGB8_SRGB, SRGB_BGR)
        .unwrap()
        .try_convert_row(&out, &mut back, 2)
        .unwrap();
    assert_eq!(back, bgr);
    let mut rgba = [0u8; 8];
    crate::test_conversion::new_worker(SRGB_BGR, D::RGBA8_SRGB)
        .unwrap()
        .try_convert_row(&bgr, &mut rgba, 2)
        .unwrap();
    assert_eq!(rgba, [30, 20, 10, 255, 60, 50, 40, 255]);
}

/// Every conversion from BGR gives exactly what the swapped pixels give from RGB, and the
/// reverse, across depth, transfer, primaries and row splits.
#[test]
fn bgr8_matches_rgb8_through_every_route() {
    let rgb: Vec<u8> = (0..=255u8)
        .flat_map(|v| [v, v.wrapping_mul(7), 255 - v])
        .collect();
    let bgr: Vec<u8> = rgb
        .as_chunks::<3>()
        .0
        .iter()
        .flat_map(|p| [p[2], p[1], p[0]])
        .collect();
    let width = 256u32;
    let as_bgr = |d: D| {
        D::new(ChannelType::U8, ChannelLayout::Bgr, None, d.transfer()).with_primaries(d.primaries)
    };
    let sources = [
        D::RGB8_SRGB,
        D::RGB8_SRGB.with_transfer(T::Linear),
        D::RGB8_SRGB.with_transfer(T::Bt709),
        D::RGB8_SRGB.with_primaries(P::DisplayP3),
        D::RGB8_SRGB.with_primaries(P::Bt2020),
    ];
    let targets = [
        D::RGB8_SRGB,
        D::RGBA8_SRGB,
        D::BGRA8_SRGB,
        D::RGB16_SRGB,
        D::RGBF32_LINEAR,
        D::RGBAF32_LINEAR,
        D::GRAY8_SRGB,
        D::RGB8_SRGB.with_primaries(P::DisplayP3),
    ];
    let options = ConvertOptions::retain_channels()
        .with_alpha_policy(zenpixels_convert::AlphaPolicy::DiscardIfOpaque)
        .with_depth_policy(zenpixels_convert::DepthPolicy::Round)
        .with_luma(Some(zenpixels_convert::LumaCoefficients::Bt709))
        .with_alpha_policy(AlphaPolicy::DiscardUnchecked);
    for source in sources {
        for target in targets {
            let bytes = width as usize * target.bytes_per_pixel();
            let (mut from_rgb, mut from_bgr) = (vec![0u8; bytes], vec![0u8; bytes]);
            crate::test_conversion::explicit_worker(source, target, &options)
                .unwrap()
                .try_convert_row(&rgb, &mut from_rgb, width)
                .unwrap();
            let mut converter =
                crate::test_conversion::explicit_worker(as_bgr(source), target, &options).unwrap();
            converter
                .try_convert_row(&bgr, &mut from_bgr, width)
                .unwrap();
            assert_eq!(from_bgr, from_rgb, "{source:?} to {target:?}");
            // Split into rows of 17, which must not change anything.
            let mut split = vec![0u8; bytes];
            let (s3, d) = (17 * 3, 17 * target.bytes_per_pixel());
            for (src, dst) in bgr.chunks(s3).zip(split.chunks_mut(d)) {
                converter
                    .try_convert_row(src, dst, (src.len() / 3) as u32)
                    .unwrap();
            }
            assert_eq!(split, from_rgb, "{source:?} to {target:?} split");
        }
        // Into BGR from RGB, the swapped output of the RGB route.
        let target = as_bgr(D::RGB8_SRGB);
        let mut direct = vec![0u8; rgb.len()];
        crate::test_conversion::explicit_worker(source, D::RGB8_SRGB, &options)
            .unwrap()
            .try_convert_row(&rgb, &mut direct, width)
            .unwrap();
        let mut swapped = vec![0u8; rgb.len()];
        crate::test_conversion::explicit_worker(source, target, &options)
            .unwrap()
            .try_convert_row(&rgb, &mut swapped, width)
            .unwrap();
        let want: Vec<u8> = direct
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|p| [p[2], p[1], p[0]])
            .collect();
        assert_eq!(swapped, want, "{source:?} to BGR");
    }
}

#[test]
fn bgr8_changes_transfer_through_rgb() {
    let linear = SRGB_BGR.with_transfer(T::Linear);
    let mut out = [0u8; 3];
    crate::test_conversion::new_worker(linear, SRGB_BGR)
        .unwrap()
        .try_convert_row(&[8, 32, 147], &mut out, 1)
        .unwrap();
    assert_eq!(out[1], 99, "channel order stays: {out:?}");
    assert!(out[2] > out[1] && out[1] > out[0], "{out:?}");
    crate::test_conversion::new_worker(SRGB_BGR, SRGB_BGR)
        .unwrap()
        .try_convert_row(&[1, 2, 3], &mut out, 1)
        .unwrap();
    assert_eq!(out, [1, 2, 3]);
}

#[test]
fn cmyk16_has_no_numeric_conversion() {
    let cmyk16 = zenpixels::PixelFormat::CMYK16.descriptor();
    assert!(crate::test_conversion::new_worker(cmyk16, D::RGB16_SRGB).is_err());
    assert!(crate::test_conversion::new_worker(D::RGB16_SRGB, cmyk16).is_err());
}

#[path = "support/conversion.rs"]
mod test_conversion;
