use zenpixels::*;
use zenpixels_convert::{ConvertOptions, EncodeReady, OutputPlan, OutputProfile};

#[test]
fn borrowed_encoder_rows_avoid_pixel_allocation_and_match_owned_finalization() {
    for stride in [6, 10] {
        let source = vec![19u8; stride * 2];
        let buffer = PixelBuffer::try_from_parts(PixelBufferParts::new(
            source.into(),
            2,
            2,
            stride,
            PixelDescriptor::RGB8_SRGB,
        ))
        .unwrap();
        let origin = ColorOrigin::assumed();
        let target = OutputProfile::Named(Cicp::SRGB);
        let plan = OutputPlan::new(
            buffer.encoding(),
            &origin,
            target.clone(),
            PixelFormat::Rgb8,
            &ConvertOptions::retain_channels(),
            None,
        )
        .unwrap();
        let mut borrowed = None;
        let info = allocation_counter::measure(|| {
            borrowed = plan.borrow_rows(buffer.as_slice()).unwrap();
        });
        assert_eq!(info.bytes_total, 0);
        let rows = borrowed.unwrap();
        assert_eq!(
            rows.as_strided_bytes().as_ptr(),
            buffer.as_slice().as_strided_bytes().as_ptr()
        );
        assert_eq!(rows.stride(), stride);
        let mut owned = None;
        let info = allocation_counter::measure(|| {
            owned = Some(
                EncodeReady::finalize(&buffer, &origin, target, PixelFormat::Rgb8, None).unwrap(),
            );
        });
        assert!(info.bytes_total >= 12);
        let owned = owned.unwrap();
        assert_ne!(
            owned.pixels().as_strided_bytes().as_ptr(),
            rows.as_strided_bytes().as_ptr()
        );
        assert_eq!(owned.pixels().descriptor(), rows.descriptor());
        assert_eq!(owned.metadata().icc, plan.metadata().icc);
        assert_eq!(owned.metadata().cicp, plan.metadata().cicp);
        for y in 0..2 {
            assert_eq!(rows.row(y), owned.pixels().row(y));
        }
    }
}
