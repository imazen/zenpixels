use std::sync::Arc;
use zenpixels::*;
use zenpixels_convert::icc_profiles::SynthesizedIcc;
use zenpixels_convert::{ConvertOptions, EncodeReady, OutputPlan, OutputProfile};

#[test]
fn origin_authority_table_at_both_output_boundaries() {
    let icc: Arc<[u8]> = zenpixels_convert::icc_profiles::DISPLAY_P3_V4.into();
    for authority in [ColorAuthority::Icc, ColorAuthority::Cicp] {
        for has_icc in [false, true] {
            for has_cicp in [false, true] {
                let origin = match (has_icc, has_cicp) {
                    // Deliberately contradictory: P3 ICC versus sRGB CICP.
                    (true, true) => ColorOrigin::from_icc_and_cicp(icc.clone(), Cicp::SRGB),
                    (true, false) => ColorOrigin::from_icc(icc.clone()),
                    (false, true) => ColorOrigin::from_cicp(Cicp::SRGB),
                    _ => ColorOrigin::assumed(),
                }
                .with_color_authority(authority);
                let buffer =
                    PixelBuffer::from_vec(vec![17, 51, 193], 1, 1, PixelDescriptor::RGB8_SRGB)
                        .unwrap();
                let plan = OutputPlan::new(
                    buffer.descriptor(),
                    None,
                    &origin,
                    OutputProfile::SameAsOrigin,
                    PixelFormat::Rgb8,
                    &ConvertOptions::permissive(),
                    None,
                )
                .unwrap();
                let ready = EncodeReady::finalize(
                    &buffer,
                    &origin,
                    OutputProfile::SameAsOrigin,
                    PixelFormat::Rgb8,
                    None,
                )
                .unwrap();
                assert_eq!(plan.metadata().icc, ready.metadata().icc);
                assert_eq!(plan.metadata().cicp, ready.metadata().cicp);
                // Independent expectations, rather than using the selector
                // under test as the oracle for its own routing.
                let selected_icc = has_icc && (authority == ColorAuthority::Icc || !has_cicp);
                assert_eq!(ready.metadata().icc.is_some(), selected_icc);
                assert_eq!(
                    ready.metadata().icc.as_deref(),
                    selected_icc.then_some(icc.as_ref())
                );
                assert_eq!(
                    ready.metadata().cicp,
                    (has_cicp && !selected_icc).then_some(Cicp::SRGB)
                );
                assert_eq!(plan.output_descriptor(), ready.pixels().descriptor());
                let context = ready.pixels().color_context().unwrap().clone();
                assert_eq!(context.icc, ready.metadata().icc);
                assert_eq!(context.cicp, ready.metadata().cicp);
                if !selected_icc {
                    assert_eq!(ready.pixels().row(0), [17, 51, 193]);
                }
                assert_eq!(buffer.as_slice().row(0), [17, 51, 193]);
            }
        }
    }
}

#[test]
fn profile_synthesis_does_not_decode_coded_matrix_or_range() {
    let rgb = Cicp::new(12, 13, 0, true);
    let SynthesizedIcc::Profile(reference) = SynthesizedIcc::for_cicp(rgb) else {
        panic!("P3 profile")
    };
    for coded in [
        rgb,
        Cicp::new(12, 13, 0, false),
        Cicp::new(12, 13, 9, false),
    ] {
        assert_eq!(
            ColorProfileSource::Cicp(coded).resolve().is_some(),
            coded == rgb
        );
        let SynthesizedIcc::Profile(bytes) = SynthesizedIcc::for_cicp(coded) else {
            panic!("color profile")
        };
        assert_eq!(bytes, reference);
        assert_eq!(Cicp::from_icc(&bytes), Some(rgb));
    }
}

#[test]
fn unsupported_authoritative_origin_never_falls_back_to_icc() {
    let descriptor = PixelDescriptor::RGB8_SRGB;
    let buffer = PixelBuffer::from_vec(vec![17, 51, 193], 1, 1, descriptor).unwrap();
    for bad in [
        Cicp::new(9, 18, 9, false),
        Cicp::new(1, 7, 0, true),
        Cicp::new(1, 0, 0, true),
    ] {
        let origin = ColorOrigin::from_icc_and_cicp(vec![1, 2, 3], bad)
            .with_color_authority(ColorAuthority::Cicp);
        assert!(
            OutputPlan::new(
                descriptor,
                None,
                &origin,
                OutputProfile::SameAsOrigin,
                PixelFormat::Rgb8,
                &ConvertOptions::permissive(),
                None
            )
            .is_err()
        );
        assert!(
            EncodeReady::finalize(
                &buffer,
                &origin,
                OutputProfile::SameAsOrigin,
                PixelFormat::Rgb8,
                None
            )
            .is_err()
        );
        assert_eq!(buffer.as_slice().row(0), [17, 51, 193]);
    }
}

#[derive(Default)]
struct RoutingSpy(std::sync::Mutex<Vec<bool>>);
impl zenpixels_convert::cms::PluggableCms for RoutingSpy {
    fn build_source_transform(
        &self,
        _: ColorProfileSource<'_>,
        target: ColorProfileSource<'_>,
        _: PixelFormat,
        _: PixelFormat,
        _: &ConvertOptions,
    ) -> Option<
        Result<
            Box<dyn zenpixels_convert::cms::RowTransformMut>,
            whereat::At<zenpixels_convert::cms::CmsPluginError>,
        >,
    > {
        self.0
            .lock()
            .unwrap()
            .push(matches!(target, ColorProfileSource::Icc(_)));
        None
    }
}
#[test]
fn unknown_authoritative_icc_reaches_cms_and_cannot_be_replaced_by_cicp() {
    let spy = RoutingSpy::default();
    let origin = ColorOrigin::from_icc_and_cicp(vec![1, 2, 3], Cicp::SRGB)
        .with_color_authority(ColorAuthority::Icc);
    let source = PixelBuffer::from_vec(vec![10, 20, 30], 1, 1, PixelDescriptor::RGB8_SRGB).unwrap();
    assert!(
        EncodeReady::finalize(
            &source,
            &origin,
            OutputProfile::SameAsOrigin,
            PixelFormat::Rgb8,
            Some(&spy)
        )
        .is_err()
    );
    let seen = spy.0.lock().unwrap();
    assert!(!seen.is_empty());
    assert!(seen.iter().all(|icc| *icc));
    assert_eq!(source.as_slice().row(0), [10, 20, 30]);
}

#[test]
fn opaque_and_metadata_only_icc_authority_table_reaches_both_boundaries() {
    let SynthesizedIcc::Profile(p3) = SynthesizedIcc::for_cicp(Cicp::DISPLAY_P3) else {
        panic!("qualified P3 export missing")
    };
    assert_eq!(
        icc::IccIdentification::from_icc(&p3).unwrap().valid_use,
        icc::IdentificationUse::MetadataOnly
    );
    for bytes in [b"opaque ICC declaration".as_slice(), p3.as_ref()] {
        for authority in [ColorAuthority::Icc, ColorAuthority::Cicp] {
            for has_icc in [false, true] {
                for has_cicp in [false, true] {
                    let origin = match (has_icc, has_cicp) {
                        (true, true) => ColorOrigin::from_icc_and_cicp(bytes, Cicp::SRGB),
                        (true, false) => ColorOrigin::from_icc(bytes),
                        (false, true) => ColorOrigin::from_cicp(Cicp::SRGB),
                        _ => ColorOrigin::assumed(),
                    }
                    .with_color_authority(authority);
                    let source =
                        PixelBuffer::from_vec(vec![17, 51, 193], 1, 1, PixelDescriptor::RGB8_SRGB)
                            .unwrap();
                    let spy = RoutingSpy::default();
                    let plan = OutputPlan::new(
                        source.descriptor(),
                        None,
                        &origin,
                        OutputProfile::SameAsOrigin,
                        PixelFormat::Rgb8,
                        &ConvertOptions::permissive(),
                        Some(&spy),
                    );
                    let ready = EncodeReady::finalize(
                        &source,
                        &origin,
                        OutputProfile::SameAsOrigin,
                        PixelFormat::Rgb8,
                        Some(&spy),
                    );
                    let selected_icc = has_icc && (authority == ColorAuthority::Icc || !has_cicp);
                    assert_eq!(plan.is_err(), selected_icc);
                    assert_eq!(ready.is_err(), selected_icc);
                    let routes = spy.0.lock().unwrap();
                    if selected_icc {
                        // Each boundary must offer the selected ICC to the CMS.
                        assert!(routes.len() >= 2);
                        assert!(routes.iter().all(|icc| *icc));
                    } else {
                        assert!(routes.is_empty());
                        let ready = ready.unwrap();
                        assert_eq!(ready.metadata().cicp, has_cicp.then_some(Cicp::SRGB));
                        assert_eq!(ready.metadata().icc, None);
                        assert_eq!(ready.pixels().row(0), [17, 51, 193]);
                    }
                    assert_eq!(source.as_slice().row(0), [17, 51, 193]);
                }
            }
        }
    }
}

#[test]
fn grayscale_reduction_preserves_unusable_icc_but_allows_exact_alpha_and_depth_reduction() {
    use zenpixels_convert::{PixelBufferLoadBearingExt, PixelSliceLoadBearingExt};

    for channel in [ChannelType::U8, ChannelType::U16] {
        for has_alpha in [false, true] {
            for force_alpha in [false, true] {
                for icc_len in [0, 8, 128] {
                    for cicp in [Cicp::SRGB, Cicp::DISPLAY_P3] {
                        let context =
                            Arc::new(ColorContext::from_icc(vec![0; icc_len]).with_cicp(cicp));
                        let descriptor = PixelDescriptor::new(
                            channel,
                            if has_alpha {
                                ChannelLayout::Rgba
                            } else {
                                ChannelLayout::Rgb
                            },
                            has_alpha.then_some(AlphaMode::Straight),
                            TransferFunction::Srgb,
                        )
                        .with_primaries(ColorPrimaries::Bt709);
                        let row_len = 2 * descriptor.bytes_per_pixel();
                        // Sample-aligned partial-pixel padding, absent on the final row.
                        let stride = row_len + 3 * channel.byte_size();
                        let mut bytes = vec![0xa5; stride + row_len];
                        let mut rgb = Vec::new();
                        let mut rgba = Vec::new();
                        for (pixel, value) in [64u8, 128, 17, 51].into_iter().enumerate() {
                            rgb.extend_from_slice(&[value; 3]);
                            rgba.extend_from_slice(&[value, value, value, 255]);
                            for c in 0..descriptor.channels() {
                                let sample = if c == 3 { 255 } else { value };
                                let index = (pixel / 2) * stride
                                    + (pixel % 2) * descriptor.bytes_per_pixel()
                                    + c * channel.byte_size();
                                match channel {
                                    ChannelType::U8 => bytes[index] = sample,
                                    ChannelType::U16 => bytes[index..index + 2]
                                        .copy_from_slice(&(u16::from(sample) * 257).to_ne_bytes()),
                                    _ => unreachable!(),
                                }
                            }
                        }
                        let original = bytes.clone();
                        let mut buffer = PixelBuffer::try_from_parts(PixelBufferParts::new(
                            bytes.into(),
                            2,
                            2,
                            stride,
                            descriptor,
                        ))
                        .unwrap()
                        .with_color_context(context.clone());
                        let report = buffer.as_slice().determine_load_bearing();
                        assert_eq!(report.uses_chroma, Some(false));
                        assert_eq!(report.uses_alpha, Some(false));
                        assert_eq!(report.uses_low_bits, Some(false));

                        let copied = buffer.as_slice().try_reduce_to_load_bearing_format();
                        if has_alpha || channel == ChannelType::U16 {
                            let copied = copied.unwrap();
                            assert_eq!(copied.descriptor(), PixelDescriptor::RGB8_SRGB);
                            assert_eq!(copied.as_contiguous_bytes().unwrap(), rgb);
                            assert!(Arc::ptr_eq(
                                copied.as_slice().color_context().unwrap(),
                                &context
                            ));
                        } else {
                            assert!(copied.is_none(), "already RGB8; no safe reduction remains");
                        }
                        assert_eq!(buffer.as_slice().as_strided_bytes(), original);

                        buffer.reduce_to_load_bearing_format_in_place(force_alpha);
                        let retain_alpha = has_alpha && !force_alpha;
                        let expected = if retain_alpha {
                            PixelDescriptor::RGBA8_SRGB.with_alpha(Some(AlphaMode::Opaque))
                        } else {
                            PixelDescriptor::RGB8_SRGB
                        };
                        assert_eq!(buffer.descriptor(), expected);
                        let expected_pixels = if retain_alpha { &rgba } else { &rgb };
                        let row = expected_pixels.len() / 2;
                        for (y, expected_row) in expected_pixels.chunks_exact(row).enumerate() {
                            assert_eq!(buffer.as_slice().row(y as u32), expected_row);
                        }
                        assert!(Arc::ptr_eq(
                            buffer.as_slice().color_context().unwrap(),
                            &context
                        ));
                    }
                }
            }
        }
    }
}
