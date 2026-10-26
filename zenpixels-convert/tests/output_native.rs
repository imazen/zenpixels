//! Qualify the combined native-sample and profile-aware output review APIs.
use std::sync::Arc;
use zenpixels::{ColorContext, ColorOrigin, PixelFormat, PixelSlice, PixelSliceMut};
use zenpixels_convert::{ConvertOptions, OutputPlan, OutputProfile};

#[test]
fn native_cmyk_identity_plan_preserves_float_bits_and_strides() {
    for format in [PixelFormat::CmykF32, PixelFormat::CmykF32Inverted] {
        let descriptor = format
            .descriptor()
            .with_primaries(zenpixels::ColorPrimaries::Unknown);
        // Opaque identity does not interpret the ICC or the sample bit patterns.
        // The real native ICC consumer is qualified separately in zencodec.
        let profile: Arc<[u8]> = Arc::from(&b"unchanged current CMYK profile"[..]);
        let context = Arc::new(ColorContext::from_icc(profile.clone()));
        let mut plan = OutputPlan::new(
            descriptor,
            Some(context.clone()),
            &ColorOrigin::from_icc(profile.clone()),
            OutputProfile::SameAsOrigin,
            format,
            &ConvertOptions::forbid_lossy(),
            None,
        )
        .unwrap();
        plan.prepare(2).unwrap();
        let source = [
            0u32,
            0x8000_0000,
            1,
            0x7fc0_1234,
            0x3f80_0000,
            0x3e80_0000,
            0x3f00_0000,
            0x3f40_0000,
            0xa5a5_a5a5,
            0xa5a5_a5a5,
            0xa5a5_a5a5,
            0xa5a5_a5a5,
            0x3f80_0000,
            1,
            0x8000_0000,
            0,
            0x7fc0_4321,
            0x3e80_0000,
            0x3f00_0000,
            0x3f40_0000,
        ];
        let mut destination = [0x5a5a_5a5au32; 24];
        let input = PixelSlice::new(bytemuck::cast_slice(&source), 2, 2, 48, descriptor)
            .unwrap()
            .with_color_context(context);
        let output = PixelSliceMut::new(
            bytemuck::cast_slice_mut(&mut destination),
            2,
            2,
            48,
            plan.output_descriptor(),
        )
        .unwrap();
        let converted = plan.convert_rows(input.clone(), output).unwrap();
        assert_eq!(
            converted.color_context().unwrap().icc,
            Some(profile.clone())
        );
        assert_eq!(plan.metadata().icc, Some(profile));
        assert_eq!(converted.row(0), input.row(0));
        assert_eq!(converted.row(1), input.row(1));
        assert_eq!(&destination[8..12], &[0x5a5a_5a5a; 4]);
        assert_eq!(&destination[20..24], &[0x5a5a_5a5a; 4]);
    }
}
