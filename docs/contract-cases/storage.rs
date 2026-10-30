// Review evidence: most tests assert known CURRENT bugs, not desired behavior.
// Run with scripts/check-contract-cases.py; intentionally outside the CI test suite.
#[cfg(test)]
mod tests {
    use zenpixels::*;
    #[test]
    fn adopted_minimal_extent_now_supports_owned_views() {
        let mut b = PixelBuffer::new(4, 1, PixelDescriptor::RGB8);
        b.transform_in_place(|p| {
            PixelSliceMut::new(p.bytes, 1, 2, 9, PixelDescriptor::RGB8).unwrap()
        });
        assert_eq!(b.stride(), 9);
        assert_eq!(b.as_slice().row(1).len(), 3);
    }
    #[test]
    fn empty_crop_positive_rows_are_empty() {
        let b = PixelBuffer::new(4, 4, PixelDescriptor::RGB8);
        let crop = b.crop_view(0, 0, 0, 2);
        assert!(crop.row(1).is_empty());
    }
    #[test]
    fn p3_does_not_contain_adobe_green_despite_predicate() {
        assert!(!ColorPrimaries::DisplayP3.contains(ColorPrimaries::AdobeRgb));
        let m = ColorPrimaries::AdobeRgb
            .gamut_matrix_to(ColorPrimaries::DisplayP3)
            .unwrap();
        eprintln!("Adobe green in P3: {:?}", [m[0][1], m[1][1], m[2][1]]);
        assert!(
            [m[0][1], m[1][1], m[2][1]]
                .iter()
                .any(|v| *v < -0.01 || *v > 1.01)
        );
    }
    #[test]
    fn cicp_to_descriptor_preserves_padding() {
        let d = Cicp::SRGB.try_to_descriptor(PixelFormat::Rgbx8).unwrap();
        assert_eq!(d.alpha, PixelFormat::Rgbx8.default_alpha());
    }
    #[test]
    fn named_pq_cicp_roundtrip_agrees() {
        let named = ColorProfileSource::Named(NamedProfile::Bt2020Pq);
        let cicp = ColorProfileSource::Cicp(NamedProfile::Bt2020Pq.to_cicp().unwrap());
        assert!(named.resolve().is_some());
        assert_eq!(cicp.resolve(), named.resolve());
    }
    #[test]
    fn contradictory_descriptors_are_rejected_at_boundaries() {
        let d = PixelDescriptor::RGBX8.with_alpha(Some(AlphaMode::Straight));
        assert_eq!(d.pixel_format(), PixelFormat::Rgbx8);
        assert!(d.has_alpha());
        assert!(PixelBuffer::try_new(1, 1, d).is_err());
        let d = PixelDescriptor::RGB8.with_alpha(Some(AlphaMode::Premultiplied));
        assert!(d.has_alpha());
        assert!(PixelBuffer::try_new(1, 1, d).is_err());
        assert!(!d.pixel_format().has_alpha_bytes());
    }
}

#[test]
fn orientation_preserves_color_context() {
    use std::sync::Arc;
    use zenpixels::{Cicp, ColorContext, Orientation, PixelBuffer, PixelDescriptor};
    use zenpixels_convert::orient::{apply_orientation, apply_orientation_in_place};
    let ctx = Arc::new(ColorContext::from_cicp(Cicp::DISPLAY_P3));
    let mut src = PixelBuffer::new(2, 3, PixelDescriptor::RGB8).with_color_context(ctx);
    let out = apply_orientation(src.as_slice(), Orientation::Rotate90);
    assert!(Arc::ptr_eq(out.color_context().unwrap(), src.color_context().unwrap()));
    apply_orientation_in_place(&mut src, Orientation::Rotate90).unwrap();
    assert!(src.color_context().is_some());
}
