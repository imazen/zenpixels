use zenpixels::*;
use zenpixels_convert::PixelBufferConvertExt;

#[test]
fn invalid_in_place_target_refuses_before_mutation() {
    for storage in [
        ChannelType::U8,
        ChannelType::U16,
        ChannelType::F32,
        ChannelType::U64,
    ] {
        for (layout, target_layout) in [
            (ChannelLayout::Rgba, ChannelLayout::Rgb),
            (ChannelLayout::GrayAlpha, ChannelLayout::Gray),
        ] {
            for width in [0, 2] {
                let from = PixelDescriptor::new(
                    storage,
                    layout,
                    Some(AlphaMode::Opaque),
                    TransferFunction::Srgb,
                );
                let invalid =
                    PixelDescriptor::new(storage, target_layout, None, TransferFunction::Srgb)
                        .with_alpha(Some(AlphaMode::Straight));
                assert!(invalid.validate().is_err());
                let stride = (width as usize + 1) * from.bytes_per_pixel();
                let len = if width == 0 {
                    0
                } else {
                    stride + width as usize * from.bytes_per_pixel()
                };
                let mut buffer = PixelBuffer::try_from_parts(PixelBufferParts::new(
                    vec![0xff; len].into(),
                    width,
                    2,
                    stride,
                    from,
                ))
                .unwrap();
                let before = buffer.as_slice().as_strided_bytes().to_vec();
                let error = buffer.try_adapt_in_place(invalid).unwrap_err();
                assert!(matches!(
                    error.error(),
                    zenpixels_convert::ConvertError::Buffer(BufferError::InvalidDescriptor(_))
                ));
                assert_eq!(buffer.descriptor(), from);
                assert_eq!(buffer.as_slice().as_strided_bytes(), before);
            }
        }
    }
}

struct AcceptingCms(std::sync::atomic::AtomicUsize);
struct Fill;
impl zenpixels_convert::cms::RowTransformMut for Fill {
    fn transform_row(&mut self, _: &[u8], out: &mut [u8], _: u32) {
        out.fill(42);
    }
}
impl zenpixels_convert::cms::PluggableCms for AcceptingCms {
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
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(Ok(Box::new(Fill)))
    }
}
#[test]
fn bgr_icc_output_cannot_bypass_missing_luma_policy() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use zenpixels_convert::{ConvertError, OutputPlan, OutputProfile};
    let cms = AcceptingCms(AtomicUsize::new(0));
    let current = ColorContext::from_icc(Arc::<[u8]>::from(&b"current"[..]));
    let result = OutputPlan::new(
        PixelEncoding::new(
            PixelDescriptor::BGR8.with_transfer(TransferFunction::Srgb),
            Some(&current),
        ),
        &ColorOrigin::assumed(),
        OutputProfile::Icc(Arc::from(&b"target"[..])),
        PixelFormat::Gray8,
        &ConvertOptions::retain_channels(),
        Some(&cms),
    );
    assert!(matches!(
        result.err().unwrap().error(),
        ConvertError::RgbToGray
    ));
    assert_eq!(cms.0.load(Ordering::Relaxed), 0);
}

#[test]
fn bgr_cms_row_conversion_requires_luma_policy() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let cms = AcceptingCms(AtomicUsize::new(0));
    let from = PixelDescriptor::BGR8.with_transfer(TransferFunction::Srgb);
    let to = PixelDescriptor::GRAY8_SRGB.with_primaries(ColorPrimaries::DisplayP3);
    let result = zenpixels_convert::RowConverter::new_explicit_with_cms(
        from,
        to,
        &ConvertOptions::retain_channels(),
        Some(&cms),
    );
    assert!(matches!(
        result.err().unwrap().error(),
        zenpixels_convert::ConvertError::RgbToGray
    ));
    assert_eq!(cms.0.load(Ordering::Relaxed), 0);
}

#[test]
fn explicit_luma_still_allows_cms_dispatch() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    for from in [
        PixelDescriptor::RGB8_SRGB,
        PixelDescriptor::BGR8.with_transfer(TransferFunction::Srgb),
    ] {
        let cms = AcceptingCms(AtomicUsize::new(0));
        let to = PixelDescriptor::GRAY8_SRGB.with_primaries(ColorPrimaries::DisplayP3);
        let options = ConvertOptions::retain_channels()
            .with_luma(Some(zenpixels_convert::LumaCoefficients::Bt709));
        assert!(
            zenpixels_convert::RowConverter::new_explicit_with_cms(from, to, &options, Some(&cms))
                .is_ok()
        );
        assert_eq!(cms.0.load(Ordering::Relaxed), 1);
    }
}
