//! Tests for `finalize_for_output` — the atomic output preparation path.
//!
//! This is a critical function with zero prior test coverage.

use alloc::sync::Arc;

use zenpixels_convert::cms::{ColorManagement, RowTransform};
use zenpixels_convert::{
    Cicp, ColorOrigin, PixelBuffer, PixelDescriptor, PixelFormat, finalize_for_output,
    output::OutputProfile,
};

extern crate alloc;

// ---------------------------------------------------------------------------
// Stub CMS implementation
// ---------------------------------------------------------------------------

/// A no-op CMS that always fails `build_transform`.
struct NoopCms;

impl ColorManagement for NoopCms {
    type Error = &'static str;

    fn build_transform(
        &self,
        _src_icc: &[u8],
        _dst_icc: &[u8],
    ) -> Result<Box<dyn RowTransform>, Self::Error> {
        Err("no-op CMS: ICC transforms not supported")
    }

    fn identify_profile(&self, _icc: &[u8]) -> Option<Cicp> {
        None
    }
}

/// A CMS that copies src to dst (identity transform).
struct IdentityCms;

struct IdentityTransform;

impl RowTransform for IdentityTransform {
    fn transform_row(&self, src: &[u8], dst: &mut [u8], _width: u32) {
        let len = src.len().min(dst.len());
        dst[..len].copy_from_slice(&src[..len]);
    }
}

impl ColorManagement for IdentityCms {
    type Error = &'static str;

    fn build_transform(
        &self,
        _src_icc: &[u8],
        _dst_icc: &[u8],
    ) -> Result<Box<dyn RowTransform>, Self::Error> {
        Ok(Box::new(IdentityTransform))
    }

    fn identify_profile(&self, _icc: &[u8]) -> Option<Cicp> {
        None
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn make_rgb8_buffer(width: u32, height: u32, pixel: [u8; 3]) -> PixelBuffer {
    let mut data = Vec::new();
    for _ in 0..width * height {
        data.extend_from_slice(&pixel);
    }
    PixelBuffer::from_vec(data, width, height, PixelDescriptor::RGB8_SRGB).unwrap()
}

fn make_rgba8_buffer(width: u32, height: u32, pixel: [u8; 4]) -> PixelBuffer {
    let mut data = Vec::new();
    for _ in 0..width * height {
        data.extend_from_slice(&pixel);
    }
    PixelBuffer::from_vec(data, width, height, PixelDescriptor::RGBA8_SRGB).unwrap()
}

// ---------------------------------------------------------------------------
// SameAsOrigin tests
// ---------------------------------------------------------------------------

#[test]
fn same_as_origin_identity_preserves_pixels() {
    let buf = make_rgb8_buffer(2, 2, [100, 150, 200]);
    let origin = ColorOrigin::assumed();

    let ready = finalize_for_output(
        &buf,
        &origin,
        OutputProfile::SameAsOrigin,
        PixelFormat::Rgb8,
        &NoopCms,
    )
    .unwrap();

    let slice = ready.pixels();
    let row = slice.row(0);
    assert_eq!(&row[..3], &[100, 150, 200]);
    assert_eq!(&row[3..6], &[100, 150, 200]);
}

#[test]
fn same_as_origin_metadata_has_no_icc_no_cicp_for_assumed() {
    let buf = make_rgb8_buffer(1, 1, [128, 128, 128]);
    let origin = ColorOrigin::assumed();

    let ready = finalize_for_output(
        &buf,
        &origin,
        OutputProfile::SameAsOrigin,
        PixelFormat::Rgb8,
        &NoopCms,
    )
    .unwrap();

    let meta = ready.metadata();
    assert!(meta.icc.is_none());
    assert!(meta.cicp.is_none());
    assert!(meta.hdr.is_none());
}

#[test]
fn same_as_origin_preserves_cicp_from_origin() {
    let buf = make_rgb8_buffer(1, 1, [128, 128, 128]);
    let origin = ColorOrigin::from_cicp(Cicp::SRGB);

    let ready = finalize_for_output(
        &buf,
        &origin,
        OutputProfile::SameAsOrigin,
        PixelFormat::Rgb8,
        &NoopCms,
    )
    .unwrap();

    let meta = ready.metadata();
    assert_eq!(meta.cicp, Some(Cicp::SRGB));
}

#[test]
fn same_as_origin_with_format_change_converts() {
    // Source is RGB8, target is RGBA8 — must convert.
    let buf = make_rgb8_buffer(2, 1, [100, 150, 200]);
    let origin = ColorOrigin::assumed();

    let ready = finalize_for_output(
        &buf,
        &origin,
        OutputProfile::SameAsOrigin,
        PixelFormat::Rgba8,
        &NoopCms,
    )
    .unwrap();

    let slice = ready.pixels();
    let row = slice.row(0);
    // RGB → RGBA adds alpha=255
    assert_eq!(row[0], 100);
    assert_eq!(row[1], 150);
    assert_eq!(row[2], 200);
    assert_eq!(row[3], 255);
}

// ---------------------------------------------------------------------------
// Named profile tests
// ---------------------------------------------------------------------------

#[test]
fn named_srgb_produces_cicp_metadata() {
    let buf = make_rgb8_buffer(1, 1, [128, 128, 128]);
    let origin = ColorOrigin::assumed();

    let ready = finalize_for_output(
        &buf,
        &origin,
        OutputProfile::Named(Cicp::SRGB),
        PixelFormat::Rgb8,
        &NoopCms,
    )
    .unwrap();

    let meta = ready.metadata();
    assert!(meta.icc.is_none());
    assert_eq!(meta.cicp, Some(Cicp::SRGB));
}

#[test]
fn named_profile_converts_format() {
    // Source is RGBA8, target format is Rgb8 — must drop alpha.
    let buf = make_rgba8_buffer(2, 1, [100, 150, 200, 255]);
    let origin = ColorOrigin::assumed();

    let ready = finalize_for_output(
        &buf,
        &origin,
        OutputProfile::Named(Cicp::SRGB),
        PixelFormat::Rgb8,
        &NoopCms,
    )
    .unwrap();

    let slice = ready.pixels();
    let row = slice.row(0);
    assert_eq!(&row[..3], &[100, 150, 200]);
    assert_eq!(&row[3..6], &[100, 150, 200]);
}

// ---------------------------------------------------------------------------
// ICC profile tests
// ---------------------------------------------------------------------------

#[test]
fn icc_profile_output_embeds_icc_bytes() {
    let fake_icc: Arc<[u8]> = Arc::from(vec![0u8; 64].into_boxed_slice());
    let buf = make_rgb8_buffer(1, 1, [128, 128, 128]);
    let origin = ColorOrigin::assumed();

    let ready = finalize_for_output(
        &buf,
        &origin,
        OutputProfile::Icc(fake_icc.clone()),
        PixelFormat::Rgb8,
        &NoopCms,
    )
    .unwrap();

    let meta = ready.metadata();
    assert!(meta.icc.is_some());
    assert_eq!(meta.icc.as_ref().unwrap().len(), 64);
    assert!(meta.cicp.is_none());
}

#[test]
fn icc_with_source_icc_uses_cms() {
    use zenpixels_convert::ColorContext;

    let src_icc: Arc<[u8]> = Arc::from(vec![1u8; 32].into_boxed_slice());
    let dst_icc: Arc<[u8]> = Arc::from(vec![2u8; 32].into_boxed_slice());

    let buf = make_rgb8_buffer(2, 1, [100, 150, 200]);
    let buf = buf.with_color_context(Arc::new(ColorContext::from_icc(src_icc.to_vec())));
    let origin = ColorOrigin::from_icc(src_icc.to_vec());

    // IdentityCms copies src→dst, so pixels should be preserved.
    let ready = finalize_for_output(
        &buf,
        &origin,
        OutputProfile::Icc(dst_icc),
        PixelFormat::Rgb8,
        &IdentityCms,
    )
    .unwrap();

    let slice = ready.pixels();
    let row = slice.row(0);
    assert_eq!(&row[..3], &[100, 150, 200]);
}

#[test]
fn icc_without_source_icc_falls_through_to_row_converter() {
    // Source has no ICC profile, so CMS is not invoked. Falls through to RowConverter path.
    let dst_icc: Arc<[u8]> = Arc::from(vec![2u8; 32].into_boxed_slice());
    let buf = make_rgb8_buffer(2, 1, [100, 150, 200]);
    let origin = ColorOrigin::assumed();

    let ready = finalize_for_output(
        &buf,
        &origin,
        OutputProfile::Icc(dst_icc),
        PixelFormat::Rgb8,
        &NoopCms,
    )
    .unwrap();

    let slice = ready.pixels();
    let row = slice.row(0);
    assert_eq!(&row[..3], &[100, 150, 200]);
}

// ---------------------------------------------------------------------------
// EncodeReady API tests
// ---------------------------------------------------------------------------

#[test]
fn encode_ready_into_parts_works() {
    let buf = make_rgb8_buffer(2, 2, [50, 100, 150]);
    let origin = ColorOrigin::assumed();

    let ready = finalize_for_output(
        &buf,
        &origin,
        OutputProfile::SameAsOrigin,
        PixelFormat::Rgb8,
        &NoopCms,
    )
    .unwrap();

    let (pixels, metadata) = ready.into_parts();
    assert_eq!(pixels.width(), 2);
    assert_eq!(pixels.height(), 2);
    assert!(metadata.icc.is_none());
}

// ---------------------------------------------------------------------------
// Multi-row conversion
// ---------------------------------------------------------------------------

#[test]
fn multi_row_conversion_is_correct() {
    // 3×2 image, convert RGB8 → RGBA8
    let buf = make_rgb8_buffer(3, 2, [10, 20, 30]);
    let origin = ColorOrigin::assumed();

    let ready = finalize_for_output(
        &buf,
        &origin,
        OutputProfile::SameAsOrigin,
        PixelFormat::Rgba8,
        &NoopCms,
    )
    .unwrap();

    let slice = ready.pixels();
    for y in 0..2 {
        let row = slice.row(y);
        for x in 0..3 {
            let off = x as usize * 4;
            assert_eq!(row[off], 10, "row {y} pixel {x} R");
            assert_eq!(row[off + 1], 20, "row {y} pixel {x} G");
            assert_eq!(row[off + 2], 30, "row {y} pixel {x} B");
            assert_eq!(row[off + 3], 255, "row {y} pixel {x} A");
        }
    }
}
