//! Finalization must use current pixel interpretation and the requested profile.
use std::sync::{Arc, Mutex};
use zenpixels_convert::{
    Cicp, ColorContext, ColorOrigin, ColorProfileSource, PixelBuffer, PixelDescriptor, PixelFormat,
    TransferFunction,
    cms::{PluggableCms, RowTransform},
    finalize_for_output_with,
    output::OutputProfile,
    policy::ConvertOptions,
};

#[derive(Default)]
struct RecordingCms(Mutex<Vec<(Vec<u8>, Vec<u8>)>>);
struct Invert;
impl RowTransform for Invert {
    fn transform_row(&self, source: &[u8], target: &mut [u8], _: u32) {
        for (to, from) in target.iter_mut().zip(source) {
            *to = 255 - from;
        }
    }
}
impl PluggableCms for RecordingCms {
    fn build_source_transform(
        &self,
        _: ColorProfileSource<'_>,
        _: ColorProfileSource<'_>,
        _: PixelFormat,
        _: PixelFormat,
        _: &ConvertOptions,
    ) -> Option<
        Result<
            Box<dyn zenpixels_convert::cms::RowTransformMut>,
            whereat::At<zenpixels_convert::cms::CmsPluginError>,
        >,
    > {
        None
    }
    fn build_shared_source_transform(
        &self,
        source: ColorProfileSource<'_>,
        target: ColorProfileSource<'_>,
        _: PixelFormat,
        _: PixelFormat,
        _: &ConvertOptions,
    ) -> Option<Result<Arc<dyn RowTransform>, whereat::At<zenpixels_convert::cms::CmsPluginError>>>
    {
        if let (ColorProfileSource::Icc(a), ColorProfileSource::Icc(b)) = (source, target) {
            self.0.lock().unwrap().push((a.to_vec(), b.to_vec()));
            Some(Ok(Arc::new(Invert)))
        } else {
            None
        }
    }
}

#[test]
fn current_icc_and_target_icc_reach_the_plugin_even_when_descriptors_match() {
    let desc = PixelDescriptor::RGB8_SRGB
        .with_transfer(TransferFunction::Unknown)
        .with_primaries(zenpixels_convert::ColorPrimaries::Unknown);
    let current: Arc<[u8]> = Arc::from(&b"current-profile"[..]);
    let target: Arc<[u8]> = Arc::from(&b"target-profile"[..]);
    let buffer = PixelBuffer::from_vec(vec![7, 51, 193], 1, 1, desc)
        .unwrap()
        .with_color_context(Arc::new(ColorContext::from_icc(current.clone())));
    // Original file metadata is intentionally stale; current pixels win.
    let origin = ColorOrigin::from_icc(Arc::<[u8]>::from(&b"original-profile"[..]));
    let cms = RecordingCms::default();
    let output = finalize_for_output_with(
        &buffer,
        &origin,
        OutputProfile::Icc(target.clone()),
        PixelFormat::Rgb8,
        Some(&cms),
    )
    .unwrap();
    assert_eq!(
        *cms.0.lock().unwrap(),
        vec![(current.to_vec(), target.to_vec())]
    );
    assert_eq!(output.pixels().row(0), &[248, 204, 62]);
    assert_eq!(output.metadata().icc.as_deref(), Some(target.as_ref()));
    assert_eq!(
        output.pixels().color_context().unwrap().icc.as_deref(),
        Some(target.as_ref())
    );
}

#[test]
fn unsupported_icc_conversion_is_an_error_instead_of_a_relabel() {
    let desc = PixelDescriptor::RGB8_SRGB
        .with_transfer(TransferFunction::Unknown)
        .with_primaries(zenpixels_convert::ColorPrimaries::Unknown);
    let buffer = PixelBuffer::from_vec(vec![7, 51, 193], 1, 1, desc)
        .unwrap()
        .with_color_context(Arc::new(ColorContext::from_icc(Arc::<[u8]>::from(
            &b"source"[..],
        ))));
    assert!(
        finalize_for_output_with(
            &buffer,
            &ColorOrigin::assumed(),
            OutputProfile::Icc(Arc::from(&b"target"[..])),
            PixelFormat::Rgb8,
            None
        )
        .is_err()
    );
}

#[test]
fn same_as_origin_converts_current_linear_pixels_back_to_original_srgb() {
    let input = PixelBuffer::from_vec(
        0.5_f32.to_ne_bytes().repeat(3),
        1,
        1,
        PixelDescriptor::RGBF32_LINEAR,
    )
    .unwrap()
    .with_cicp(Cicp::new(1, 8, 0, true));
    let output = finalize_for_output_with(
        &input,
        &ColorOrigin::from_cicp(Cicp::SRGB),
        OutputProfile::SameAsOrigin,
        PixelFormat::Rgb8,
        None,
    )
    .unwrap();
    assert_eq!(output.pixels().row(0), &[188, 188, 188]);
    assert_eq!(output.pixels().descriptor(), PixelDescriptor::RGB8_SRGB);
    assert_eq!(
        output.pixels().color_context().unwrap().cicp,
        Some(Cicp::SRGB)
    );
}

#[test]
fn conflicting_current_cicp_is_not_silently_overruled_by_the_descriptor() {
    let input = PixelBuffer::from_vec(vec![127; 3], 1, 1, PixelDescriptor::RGB8_SRGB)
        .unwrap()
        .with_cicp(Cicp::new(1, 8, 0, true));
    assert!(
        finalize_for_output_with(
            &input,
            &ColorOrigin::assumed(),
            OutputProfile::Named(Cicp::SRGB),
            PixelFormat::Rgb8,
            None
        )
        .is_err()
    );
}

#[cfg(feature = "cms-moxcms")]
#[test]
fn real_icc_input_matches_direct_moxcms_and_keeps_straight_alpha() {
    use zenpixels_convert::{cms_moxcms::MoxCms, icc_profiles::ADOBE_RGB};
    let descriptor = PixelDescriptor::RGBA8_SRGB
        .with_transfer(TransferFunction::Unknown)
        .with_primaries(zenpixels_convert::ColorPrimaries::Unknown);
    let input = [191, 97, 53, 0, 47, 177, 121, 128, 157, 77, 193, 255];
    let buffer = PixelBuffer::from_vec(input.to_vec(), 3, 1, descriptor)
        .unwrap()
        .with_color_context(Arc::new(ColorContext::from_icc(ADOBE_RGB)));
    // This compares the finalizer's routing to a direct call, not a second
    // color-science oracle. The profile's exact TRC must reach the same CMS.
    let mut direct = MoxCms
        .build_source_transform(
            ColorProfileSource::Icc(ADOBE_RGB),
            ColorProfileSource::Cicp(Cicp::SRGB),
            PixelFormat::Rgba8,
            PixelFormat::Rgba8,
            &ConvertOptions::permissive(),
        )
        .unwrap()
        .unwrap();
    let mut expected = [0u8; 12];
    direct.transform_row(&input, &mut expected, 3);
    let ready = finalize_for_output_with(
        &buffer,
        &ColorOrigin::assumed(),
        OutputProfile::Named(Cicp::SRGB),
        PixelFormat::Rgba8,
        Some(&MoxCms),
    )
    .unwrap();
    assert_eq!(ready.pixels().row(0), expected);
    assert_ne!(expected, input);
    assert_eq!([expected[3], expected[7], expected[11]], [0, 128, 255]);
    assert_eq!(ready.pixels().descriptor(), PixelDescriptor::RGBA8_SRGB);
    assert_eq!(
        ready.pixels().color_context().unwrap().cicp,
        Some(Cicp::SRGB)
    );
}
