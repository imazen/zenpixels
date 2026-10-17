// Corrected output, CMS and storage contract regressions.
#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use zenpixels::*;
    use zenpixels_convert::cms::{CmsPluginError, PluggableCms, RowTransformMut};
    use zenpixels_convert::{RowConverter, finalize_for_output_with, output::OutputProfile};

    #[test]
    fn same_as_origin_converts_current_pixels() {
        let b = PixelBuffer::from_vec(vec![200, 50, 10], 1, 1, PixelDescriptor::RGB8_SRGB).unwrap();
        let origin = ColorOrigin::from_cicp(Cicp::DISPLAY_P3);
        let r = finalize_for_output_with(
            &b,
            &origin,
            OutputProfile::SameAsOrigin,
            PixelFormat::Rgb8,
            None,
        )
        .unwrap();
        assert_eq!(r.metadata().cicp, Some(Cicp::DISPLAY_P3));
        assert_eq!(r.pixels().descriptor().primaries, ColorPrimaries::DisplayP3);
        assert_ne!(r.pixels().row(0), [200, 50, 10]);
    }

    #[test]
    fn output_unassociates_premultiplied_alpha() {
        let d = PixelDescriptor::RGBA8_SRGB.with_alpha(Some(AlphaMode::Premultiplied));
        let b = PixelBuffer::from_vec(vec![64, 32, 16, 128], 1, 1, d).unwrap();
        let r = finalize_for_output_with(
            &b,
            &ColorOrigin::assumed(),
            OutputProfile::Named(Cicp::SRGB),
            PixelFormat::Rgba8,
            None,
        )
        .unwrap();
        assert_eq!(r.pixels().descriptor().alpha(), Some(AlphaMode::Straight));
        assert_eq!(r.pixels().row(0), [128, 64, 32, 128]);
    }
    #[test]
    fn output_refuses_unimplemented_range_change() {
        let b =
            PixelBuffer::from_vec(vec![255, 255, 255], 1, 1, PixelDescriptor::RGB8_SRGB).unwrap();
        let narrow = Cicp::new(1, 13, 0, false);
        assert!(
            finalize_for_output_with(
                &b,
                &ColorOrigin::assumed(),
                OutputProfile::Named(narrow),
                PixelFormat::Rgb8,
                None
            )
            .is_err()
        );
    }
    struct Fill;
    impl RowTransformMut for Fill {
        fn transform_row(&mut self, _s: &[u8], d: &mut [u8], _: u32) {
            d.fill(42)
        }
    }
    #[derive(Default)]
    struct Spy(Mutex<Vec<(bool, bool)>>);
    impl PluggableCms for Spy {
        fn build_source_transform(
            &self,
            s: ColorProfileSource<'_>,
            d: ColorProfileSource<'_>,
            _: PixelFormat,
            _: PixelFormat,
            _: &ConvertOptions,
        ) -> Option<Result<Box<dyn RowTransformMut>, whereat::At<CmsPluginError>>> {
            self.0.lock().unwrap().push((
                matches!(s, ColorProfileSource::Icc(_)),
                matches!(d, ColorProfileSource::Icc(_)),
            ));
            Some(Ok(Box::new(Fill)))
        }
    }
    #[test]
    fn output_cms_receives_actual_icc() {
        let cms = Spy::default();
        let icc: std::sync::Arc<[u8]> = zenpixels_convert::icc_profiles::DISPLAY_P3_V4.into();
        let b = PixelBuffer::from_vec(vec![200, 50, 10], 1, 1, PixelDescriptor::RGB8_SRGB)
            .unwrap()
            .with_icc(icc.clone());
        let r = finalize_for_output_with(
            &b,
            &ColorOrigin::from_icc(icc.clone()),
            OutputProfile::Icc(zenpixels_convert::icc_profiles::DISPLAY_P3_V2.into()),
            PixelFormat::Rgb8,
            Some(&cms),
        )
        .unwrap();
        assert_eq!(*cms.0.lock().unwrap(), vec![(true, true)]);
        assert_eq!(r.pixels().row(0), [42, 42, 42]);
    }
    fn custom_converter() -> RowConverter {
        RowConverter::new_explicit_with_cms(
            PixelDescriptor::RGB8_SRGB,
            PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3),
            &ConvertOptions::permissive(),
            Some(&Spy::default()),
        )
        .unwrap()
    }
    #[test]
    fn compose_refuses_to_discard_external_transform() {
        let mut a = custom_converter();
        let b = RowConverter::new(a.to_descriptor(), a.to_descriptor()).unwrap();
        assert!(a.compose(&b).is_none());
        let mut direct = [0; 3];
        a.convert_row(&[1, 2, 3], &mut direct, 1);
        assert_eq!(direct, [42, 42, 42]);
    }
    #[cfg(not(feature = "std"))]
    #[test]
    fn no_std_clone_refuses_to_discard_external_transform() {
        let mut a = custom_converter();
        assert!(a.try_clone().is_err());
        let mut direct = [0; 3];
        a.convert_row(&[1, 2, 3], &mut direct, 1);
        assert_eq!(direct, [42, 42, 42]);
    }
    #[test]
    fn reinterpret_rejects_invalid_alignment() {
        let data = [0u8; 8];
        let offset = (0..4)
            .find(|&i| !(data.as_ptr() as usize + i).is_multiple_of(4))
            .unwrap();
        let bytes = &data[offset..offset + 4];
        let s = PixelSlice::new(bytes, 1, 1, 4, PixelDescriptor::RGBA8_SRGB).unwrap();
        let f = PixelFormat::GrayF32.descriptor();
        assert!(PixelSlice::new(bytes, 1, 1, 4, f).is_err());
        assert!(s.reinterpret(f).is_err());
    }
    #[test]
    fn typed_reinterpret_rejects_wrong_type() {
        let mut data = [1u8, 2, 3, 4];
        let s = PixelSliceMut::<rgb::RGBA<u8>>::new_typed(&mut data, 1, 1, 1).unwrap();
        assert!(s.reinterpret(PixelDescriptor::BGRA8_SRGB).is_err());
    }
    #[test]
    fn layout_helper_preserves_color_and_alpha() {
        let mut data = [1u8, 2, 3, 128];
        let s = PixelSliceMut::<rgb::RGBA<u8>>::new_typed(&mut data, 1, 1, 1)
            .unwrap()
            .with_primaries(ColorPrimaries::DisplayP3)
            .with_transfer(TransferFunction::Linear)
            .with_alpha_mode(Some(AlphaMode::Premultiplied));
        let s = s.swap_to_bgra();
        assert_eq!(s.descriptor().primaries, ColorPrimaries::DisplayP3);
        assert_eq!(s.descriptor().transfer(), TransferFunction::Linear);
        assert_eq!(s.descriptor().alpha(), Some(AlphaMode::Premultiplied));
    }
}

#[test]
fn output_emits_only_the_selected_origin_authority() {
    use zenpixels::*;
    use zenpixels_convert::{OutputProfile, finalize_for_output_with};
    let b = PixelBuffer::from_vec(vec![100, 50, 20], 1, 1, PixelDescriptor::RGB8_SRGB).unwrap();
    let origin = ColorOrigin::from_icc_and_cicp(vec![1, 2, 3], Cicp::DISPLAY_P3)
        .with_color_authority(ColorAuthority::Cicp);
    let ready = finalize_for_output_with(
        &b,
        &origin,
        OutputProfile::SameAsOrigin,
        PixelFormat::Rgb8,
        None,
    )
    .unwrap();
    assert_eq!(ready.metadata().cicp, Some(Cicp::DISPLAY_P3));
    assert!(ready.metadata().icc.is_none());
}

#[test]
fn missing_origin_does_not_emit_untagged_p3() {
    use zenpixels::*;
    use zenpixels_convert::{OutputProfile, finalize_for_output_with};
    let b = PixelBuffer::from_vec(
        vec![100, 50, 20],
        1,
        1,
        PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3),
    )
    .unwrap();
    let ready = finalize_for_output_with(
        &b,
        &ColorOrigin::assumed(),
        OutputProfile::SameAsOrigin,
        PixelFormat::Rgb8,
        None,
    )
    .unwrap();
    assert_eq!(ready.metadata().cicp, Some(Cicp::DISPLAY_P3));
}
