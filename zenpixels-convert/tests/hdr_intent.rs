use std::sync::Arc;
#[cfg(feature = "bt2446a")]
use zenpixels::hdr::HdrPeak;
use zenpixels::hdr::{DiffuseWhite, Nits};
use zenpixels_convert::{
    ColorContext, ColorPrimaries, ConvertError, ConvertOptions, ConvertPlan, PixelBuffer,
    PixelBufferConvertExt, PixelDescriptor, PixelSlice, PixelSliceConvertExt, RowConverter,
};

fn linear() -> PixelDescriptor {
    PixelDescriptor::RGBF32_LINEAR.with_primaries(ColorPrimaries::Bt2020)
}
fn source(white: Option<DiffuseWhite>) -> PixelBuffer {
    let mut value = PixelBuffer::new(1, 1, linear());
    value
        .as_slice_mut()
        .row_mut(0)
        .copy_from_slice(bytemuck::cast_slice(&[1_f32; 3]));
    if let Some(white) = white {
        value =
            value.with_color_context(Arc::new(ColorContext::default().with_diffuse_white(white)));
    }
    value
}
fn codes(buffer: &PixelBuffer) -> Vec<u16> {
    buffer
        .as_slice()
        .row(0)
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| u16::from_ne_bytes(*v))
        .collect()
}
#[test]
fn every_entry_requires_an_anchor_then_agrees_at_203_nits() {
    let target = PixelDescriptor::RGB16_BT2100_PQ;
    assert!(matches!(
        source(None).convert_to(target).unwrap_err().error(),
        ConvertError::SdrWhiteRequired
    ));
    assert!(matches!(
        source(None)
            .as_slice()
            .quantize_to(target)
            .unwrap_err()
            .error(),
        ConvertError::SdrWhiteRequired
    ));
    assert!(matches!(
        RowConverter::new(linear(), target).unwrap_err().error(),
        ConvertError::SdrWhiteRequired
    ));
    let anchored = source(Some(DiffuseWhite::BT2408));
    let expected = anchored.convert_to(target).unwrap();
    assert_eq!(codes(&expected), vec![38055; 3]);
    assert_eq!(
        codes(&anchored.as_slice().quantize_to(target).unwrap()),
        codes(&expected)
    );
    let options = ConvertOptions::permissive().with_sdr_white(DiffuseWhite::BT2408);
    let mut converter = RowConverter::new_explicit(linear(), target, &options).unwrap();
    let mut output = [0_u16; 3];
    converter
        .try_convert_row(
            anchored.as_slice().row(0),
            bytemuck::cast_slice_mut(&mut output),
            1,
        )
        .unwrap();
    assert_eq!(output.as_slice(), codes(&expected));
    assert!(format!("{options:?}").contains("203"));
}
#[test]
fn explicit_white_overrides_context_and_survives_strided_conversion() {
    let pixels = [1_f32, 1., 1., -999., -999., -999., 0.5, 0.5, 0.5];
    let slice = PixelSlice::new(bytemuck::cast_slice(&pixels), 1, 2, 24, linear()).unwrap();
    let options = ConvertOptions::permissive().with_sdr_white(DiffuseWhite::new(100.));
    let converted = slice
        .convert_to_with_options(PixelDescriptor::RGB16_BT2100_PQ, &options)
        .unwrap();
    assert_eq!(converted.height(), 2);
    let inherited = source(Some(DiffuseWhite::BT2408))
        .convert_to_with_options(PixelDescriptor::RGB16_BT2100_PQ, &options)
        .unwrap();
    assert_eq!(codes(&inherited), codes(&converted));
    assert_eq!(
        inherited.color_context().unwrap().diffuse_white,
        Some(DiffuseWhite::new(100.))
    );
    assert_ne!(codes(&inherited), vec![38055; 3]);
}
#[test]
fn rows_need_a_peak_and_hlg_names_the_missing_ootf() {
    let from = PixelDescriptor::RGB16_BT2100_PQ;
    let to = PixelDescriptor::RGB8_SRGB;
    let options = ConvertOptions::permissive().with_sdr_peak(Nits::new(100.));
    assert!(matches!(
        ConvertPlan::new_explicit(from, to, &options)
            .unwrap_err()
            .error(),
        ConvertError::HdrPeakRequired
    ));
    assert!(matches!(
        ConvertPlan::new_explicit(PixelDescriptor::RGB16_BT2100_HLG, to, &options)
            .unwrap_err()
            .error(),
        ConvertError::HlgRenderingRequired
    ));
    assert!(matches!(
        ConvertPlan::new(linear(), PixelDescriptor::RGB16_BT2100_HLG)
            .unwrap_err()
            .error(),
        ConvertError::HlgRenderingRequired
    ));
}
#[cfg(feature = "bt2446a")]
#[test]
fn measured_default_matches_an_explicit_peak_and_declared_peak_refuses_if_absent() {
    let encoded = source(Some(DiffuseWhite::BT2408))
        .convert_to(PixelDescriptor::RGB16_BT2100_PQ)
        .unwrap();
    let target = PixelDescriptor::RGB8_SRGB;
    let options = ConvertOptions::permissive().with_sdr_peak(Nits::new(100.));
    let measured = encoded.convert_to_with_options(target, &options).unwrap();
    // Decode the actual quantized samples in absolute units, then supply that peak.
    let decoded = encoded
        .convert_to_with_options(linear(), &options.with_sdr_white(DiffuseWhite::new(10000.)))
        .unwrap();
    let peak = f32::from_ne_bytes(decoded.as_slice().row(0)[..4].try_into().unwrap()) * 10000.;
    let explicit = encoded
        .convert_to_with_options(
            target,
            &options.with_hdr_peak(HdrPeak::Nits(Nits::new(peak))),
        )
        .unwrap();
    assert_eq!(measured.as_slice().row(0), explicit.as_slice().row(0));
    assert!(matches!(
        encoded
            .convert_to_with_options(target, &options.with_hdr_peak(HdrPeak::Declared))
            .unwrap_err()
            .error(),
        ConvertError::HdrPeakRequired
    ));
}

#[test]
fn luminance_values_reject_invalid_units() {
    for value in [0., -0., -1., f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(Nits::try_new(value).is_none());
    }
    const PEAK: Nits = Nits::new(1000.);
    assert_eq!(PEAK.nits(), 1000.);
}

#[cfg(feature = "bt2446a")]
#[test]
fn peak_metadata_is_used_only_when_requested_and_options_override_it() {
    let encoded = source(Some(DiffuseWhite::BT2408))
        .convert_to(PixelDescriptor::RGB16_BT2100_PQ)
        .unwrap();
    let declared = encoded
        .convert_to(encoded.descriptor())
        .unwrap()
        .with_color_context(Arc::new(
            encoded
                .color_context()
                .unwrap()
                .as_ref()
                .clone()
                .with_hdr_peak(Nits::new(1000.)),
        ));
    let options = ConvertOptions::permissive().with_sdr_peak(Nits::new(100.));
    let to = PixelDescriptor::RGB8_SRGB;
    let bytes = |value: PixelBuffer| value.as_slice().row(0).to_vec();
    assert_eq!(
        bytes(encoded.convert_to_with_options(to, &options).unwrap()),
        bytes(declared.convert_to_with_options(to, &options).unwrap())
    );
    let expected = bytes(
        encoded
            .convert_to_with_options(to, &options.with_hdr_peak(HdrPeak::Nits(Nits::new(1000.))))
            .unwrap(),
    );
    assert_eq!(
        expected,
        bytes(
            declared
                .convert_to_with_options(to, &options.with_hdr_peak(HdrPeak::Declared))
                .unwrap()
        )
    );
    assert_eq!(
        expected,
        bytes(
            declared
                .convert_to_with_options(
                    to,
                    &options.with_hdr_peak(HdrPeak::Nits(Nits::new(1000.)))
                )
                .unwrap()
        )
    );
}

#[cfg(feature = "bt2446a")]
#[test]
fn black_frame_measurement_maps_to_black_without_inventing_a_declared_peak() {
    let black = PixelBuffer::new(2, 3, PixelDescriptor::RGB16_BT2100_PQ);
    let options = ConvertOptions::permissive().with_sdr_peak(Nits::new(100.));
    let result = black
        .convert_to_with_options(PixelDescriptor::RGB8_SRGB, &options)
        .unwrap();
    for row in 0..result.height() {
        assert_eq!(result.as_slice().row(row), &[0; 6]);
    }
    assert_eq!(result.color_context().unwrap().hdr_peak, None);
}

#[cfg(feature = "bt2446a")]
#[test]
fn legacy_hdr_constructor_does_not_bypass_sdr_white_or_hlg_refusal() {
    let options = zenpixels_convert::HdrConfig::default();
    assert!(matches!(
        ConvertPlan::new_with_hdr_config(
            PixelDescriptor::RGB8_SRGB,
            PixelDescriptor::RGB16_BT2100_PQ,
            options
        )
        .unwrap_err()
        .error(),
        ConvertError::SdrWhiteRequired
    ));
    assert!(matches!(
        ConvertPlan::new_with_hdr_config(
            PixelDescriptor::RGB8_SRGB,
            PixelDescriptor::RGB16_BT2100_HLG,
            options
        )
        .unwrap_err()
        .error(),
        ConvertError::HlgRenderingRequired
    ));
}
