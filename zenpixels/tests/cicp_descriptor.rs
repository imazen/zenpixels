use zenpixels::cicp::CicpDescriptorError;
use zenpixels::{AlphaMode, ChannelLayout, Cicp, ColorContext, PixelFormat, SignalRange};

#[test]
fn native_matrix_cannot_be_erased_by_descriptor_projection() {
    for matrix in 1..=255 {
        let raw = Cicp::new(9, 16, matrix, false);
        assert_eq!(
            raw.try_to_descriptor(PixelFormat::Rgb16),
            Err(CicpDescriptorError::NonIdentityMatrix(matrix))
        );
        // Failure to project must not prevent inspection or transport.
        assert_eq!(ColorContext::from_cicp(raw).cicp, Some(raw));
    }
}

#[test]
fn already_rgb_projection_preserves_range_and_padding_alpha() {
    for full in [false, true] {
        let rgb = Cicp::new(9, 16, 0, full);
        for format in [PixelFormat::Rgb16, PixelFormat::Rgba16, PixelFormat::Rgbx8] {
            let desc = rgb.try_to_descriptor(format).unwrap();
            assert_eq!(
                desc.signal_range,
                if full {
                    SignalRange::Full
                } else {
                    SignalRange::Narrow
                }
            );
            assert_eq!(desc.alpha(), format.default_alpha());
            assert_eq!(Cicp::from_descriptor(&desc), Some(rgb));
            desc.validate().unwrap();
        }
    }
    assert_eq!(
        Cicp::SRGB
            .try_to_descriptor(PixelFormat::Bgrx8)
            .unwrap()
            .alpha(),
        Some(AlphaMode::Undefined)
    );
}

#[test]
fn unrepresentable_codes_are_reported_without_replacing_them() {
    // XYZ is a defined code, but it cannot become an RGB enum by relabeling.
    for code in [0, 2, 10, 200] {
        let raw = Cicp::new(code, 13, 0, true);
        assert_eq!(
            raw.try_to_descriptor(PixelFormat::Rgb8),
            Err(CicpDescriptorError::UnmappedPrimaries(code))
        );
        assert_eq!(ColorContext::from_cicp(raw).cicp, Some(raw));
    }
    // SMPTE 240M must not silently become BT.709.
    for code in [0, 2, 7, 200] {
        let raw = Cicp::new(1, code, 0, true);
        assert_eq!(
            raw.try_to_descriptor(PixelFormat::Rgb8),
            Err(CicpDescriptorError::UnmappedTransfer(code))
        );
    }
}

#[test]
fn rgb_color_does_not_make_cmyk_or_oklab_samples_rgb() {
    for (format, layout) in [
        (PixelFormat::Cmyk8, ChannelLayout::Cmyk),
        (PixelFormat::OklabF32, ChannelLayout::Oklab),
    ] {
        assert_eq!(
            Cicp::SRGB.try_to_descriptor(format),
            Err(CicpDescriptorError::UnsupportedLayout(layout))
        );
    }
}
