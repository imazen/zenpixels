//! Narrow-range depth changes must refuse before full-range kernels or CMS run.
use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, ColorPrimaries, PixelBuffer, PixelDescriptor,
    SignalRange, TransferFunction,
};
use zenpixels_convert::{
    ConvertError, ConvertOptions, ConvertPlan, PixelBufferLoadBearingExt, PixelSliceLoadBearingExt,
    RowConverter,
};

fn assert_refused<T>(result: Result<T, whereat::At<ConvertError>>) {
    let Err(error) = result else {
        panic!("unsupported narrow depth change was accepted");
    };
    assert!(matches!(error.error(), ConvertError::NoPath { .. }));
    assert!(error.to_string().contains("narrow"));
}

#[test]
fn all_channel_type_changes_refuse_in_ordinary_and_explicit_plans() {
    let types = [
        ChannelType::U8,
        ChannelType::U16,
        ChannelType::F16,
        ChannelType::F32,
    ];
    for layout in [ChannelLayout::Gray, ChannelLayout::Rgb, ChannelLayout::Rgba] {
        for from_type in types {
            for to_type in types {
                if from_type == to_type {
                    continue;
                }
                let alpha = (layout == ChannelLayout::Rgba).then_some(AlphaMode::Straight);
                let from = PixelDescriptor::new(from_type, layout, alpha, TransferFunction::Srgb)
                    .with_signal_range(SignalRange::Narrow);
                let to = PixelDescriptor::new(to_type, layout, alpha, TransferFunction::Srgb)
                    .with_signal_range(SignalRange::Narrow);
                let options = ConvertOptions::permissive();
                assert_refused(ConvertPlan::new(from, to));
                assert_refused(ConvertPlan::new_explicit(from, to, &options));
                assert_refused(RowConverter::new(from, to));
                assert_refused(RowConverter::new_explicit(from, to, &options));
            }
        }
    }
}

#[test]
fn cms_cannot_bypass_range_guards() {
    use zenpixels::{ColorProfileSource, PixelFormat};
    use zenpixels_convert::cms::{CmsPluginError, PluggableCms, RowTransformMut};
    struct MustNotBuild;
    impl PluggableCms for MustNotBuild {
        fn build_source_transform(
            &self,
            _: ColorProfileSource<'_>,
            _: ColorProfileSource<'_>,
            _: PixelFormat,
            _: PixelFormat,
            _: &ConvertOptions,
        ) -> Option<Result<Box<dyn RowTransformMut>, whereat::At<CmsPluginError>>> {
            panic!("CMS setup must not run for unsupported range conversion");
        }
    }
    let narrow8 = PixelDescriptor::RGB8_SRGB.with_signal_range(SignalRange::Narrow);
    let narrow16 = PixelDescriptor::RGB16_SRGB
        .with_primaries(ColorPrimaries::DisplayP3)
        .with_signal_range(SignalRange::Narrow);
    for (from, to) in [
        (narrow8, narrow16),
        (narrow16, narrow8),
        (narrow8, narrow16.with_signal_range(SignalRange::Full)),
        (narrow16.with_signal_range(SignalRange::Full), narrow8),
    ] {
        assert_refused(RowConverter::new_explicit_with_cms(
            from,
            to,
            &ConvertOptions::permissive(),
            Some(&MustNotBuild),
        ));
    }
}

#[cfg(feature = "hdr-experimental")]
#[test]
fn hdr_constructor_refuses_narrow_depth_changes() {
    let narrow8 = PixelDescriptor::RGB8_SRGB
        .with_transfer(TransferFunction::Pq)
        .with_signal_range(SignalRange::Narrow);
    let narrow16 = PixelDescriptor::RGB16_SRGB
        .with_transfer(TransferFunction::Pq)
        .with_signal_range(SignalRange::Narrow);
    for (from, to) in [(narrow8, narrow16), (narrow16, narrow8)] {
        assert_refused(ConvertPlan::new_with_hdr_config(
            from,
            to,
            zenpixels_convert::HdrConfig::for_source_peak(1000.0),
        ));
    }
}

#[test]
fn full_range_widening_and_narrowing_remain_exact() {
    let input: Vec<u8> = (0..=255).collect();
    let mut wide = vec![0u16; 256];
    RowConverter::new(PixelDescriptor::GRAY8_SRGB, PixelDescriptor::GRAY16_SRGB)
        .unwrap()
        .convert_row(&input, bytemuck::cast_slice_mut(&mut wide), 256);
    for (q, &value) in wide.iter().enumerate() {
        assert_eq!(value, q as u16 * 257);
    }
    let all: Vec<u16> = (0..=65535).collect();
    let mut narrow = vec![0u8; all.len()];
    RowConverter::new(PixelDescriptor::GRAY16_SRGB, PixelDescriptor::GRAY8_SRGB)
        .unwrap()
        .convert_row(bytemuck::cast_slice(&all), &mut narrow, 65536);
    for (q, &value) in narrow.iter().enumerate() {
        assert_eq!(u32::from(value), (q as u32 + 128) / 257);
    }
}

#[test]
fn narrow_identity_and_same_depth_channel_addition_still_work() {
    let rgb = PixelDescriptor::RGB16_SRGB.with_signal_range(SignalRange::Narrow);
    let rgba = PixelDescriptor::RGBA16_SRGB.with_signal_range(SignalRange::Narrow);
    assert!(ConvertPlan::new(rgb, rgb).unwrap().is_identity());
    let input = [4096u16, 60160, 32768];
    let mut output = [0u16; 4];
    RowConverter::new(rgb, rgba).unwrap().convert_row(
        bytemuck::cast_slice(&input),
        bytemuck::cast_slice_mut(&mut output),
        1,
    );
    // Preserve color codes; alpha remains full-scale even for narrow color.
    assert_eq!(output, [4096, 60160, 32768, 65535]);
}

#[test]
fn replicated_narrow_gray_is_not_reduced_in_either_ownership_path() {
    let descriptor = PixelDescriptor::GRAY16_SRGB.with_signal_range(SignalRange::Narrow);
    let mut buffer = PixelBuffer::new(3, 1, descriptor);
    let values = [0x1010u16, 0xebeb, 0x8080];
    buffer
        .as_slice_mut()
        .row_mut(0)
        .copy_from_slice(bytemuck::cast_slice(&values));
    let pointer = buffer.as_slice().row(0).as_ptr();
    let stride = buffer.stride();
    let report = buffer.as_slice().determine_load_bearing();
    // The public measurement retains its byte-pattern meaning.
    assert_eq!(report.uses_low_bits, Some(false));
    assert_eq!(report.apply_to(&descriptor), descriptor);
    assert!(
        buffer
            .as_slice()
            .try_reduce_to_load_bearing_format()
            .is_none()
    );
    for force_alpha in [false, true] {
        buffer.reduce_to_load_bearing_format_in_place(force_alpha);
        assert_eq!(buffer.descriptor(), descriptor);
        assert_eq!(buffer.stride(), stride);
        assert_eq!(buffer.as_slice().row(0).as_ptr(), pointer);
        assert_eq!(buffer.as_slice().row(0), bytemuck::cast_slice(&values));
    }
    // The same byte pattern remains reducible under the full-range contract.
    let full = descriptor.with_signal_range(SignalRange::Full);
    assert_eq!(report.apply_to(&full).channel_type(), ChannelType::U8);
}

#[test]
fn narrow_depth_guard_keeps_independent_alpha_and_gray_reductions() {
    let descriptor = PixelDescriptor::RGBA16_SRGB.with_signal_range(SignalRange::Narrow);
    let mut buffer = PixelBuffer::new(2, 1, descriptor);
    let values = [
        0x1010u16, 0x1010, 0x1010, 65535, 0xebeb, 0xebeb, 0xebeb, 65535,
    ];
    buffer
        .as_slice_mut()
        .row_mut(0)
        .copy_from_slice(bytemuck::cast_slice(&values));
    let allocated = buffer
        .as_slice()
        .try_reduce_to_load_bearing_format()
        .unwrap();
    buffer.reduce_to_load_bearing_format_in_place(true);
    let target = PixelDescriptor::GRAY16_SRGB.with_signal_range(SignalRange::Narrow);
    for reduced in [&allocated, &buffer] {
        assert_eq!(reduced.descriptor(), target);
        assert_eq!(
            reduced.as_slice().row(0),
            bytemuck::cast_slice(&[0x1010u16, 0xebeb])
        );
    }
}
