//! Tests for `EncodeReady::finalize` — the atomic output preparation path.
//!
//! This is a critical function with zero prior test coverage.

use alloc::sync::Arc;
use core::sync::atomic::{AtomicU32, Ordering};

use zenpixels_convert::ColorProfileSource;
use zenpixels_convert::cms::{CmsPluginError, PluggableCms, RowTransformMut};
use zenpixels_convert::output::EncodeReady;
use zenpixels_convert::policy::ConvertOptions;

whereat::define_at_crate_info!();

type Built = Option<Result<Box<dyn RowTransformMut>, whereat::At<CmsPluginError>>>;

/// Only ICC reaches these mocks: anything else is the built-in path's.
fn involves_icc(src: &ColorProfileSource<'_>, dst: &ColorProfileSource<'_>) -> bool {
    matches!(src, ColorProfileSource::Icc(_)) || matches!(dst, ColorProfileSource::Icc(_))
}
use zenpixels_convert::{
    Cicp, ColorAuthority, ColorOrigin, ColorPrimaries, ConvertError, PixelBuffer, PixelDescriptor,
    PixelFormat, TransferFunction, output::OutputProfile,
};

extern crate alloc;

// ---------------------------------------------------------------------------
// Stub CMS implementation
// ---------------------------------------------------------------------------

/// A no-op CMS that always fails `build_transform`.
struct NoopCms;

impl PluggableCms for NoopCms {
    fn build_source_transform(
        &self,
        src: ColorProfileSource<'_>,
        dst: ColorProfileSource<'_>,
        _src_format: PixelFormat,
        _dst_format: PixelFormat,
        _options: &ConvertOptions,
    ) -> Built {
        involves_icc(&src, &dst).then(|| {
            Err(whereat::at!(CmsPluginError::msg(
                "no-op CMS: ICC transforms not supported"
            )))
        })
    }
}

/// A CMS that copies src to dst (identity transform).
struct IdentityCms;

struct IdentityTransform;

impl RowTransformMut for IdentityTransform {
    fn transform_row(&mut self, src: &[u8], dst: &mut [u8], _width: u32) {
        let len = src.len().min(dst.len());
        dst[..len].copy_from_slice(&src[..len]);
    }
}

impl PluggableCms for IdentityCms {
    fn build_source_transform(
        &self,
        src: ColorProfileSource<'_>,
        dst: ColorProfileSource<'_>,
        _src_format: PixelFormat,
        _dst_format: PixelFormat,
        _options: &ConvertOptions,
    ) -> Built {
        involves_icc(&src, &dst)
            .then(|| Ok(Box::new(IdentityTransform) as Box<dyn RowTransformMut>))
    }
}

/// A CMS that tracks which method was called (ICC vs CICP).
struct TrackingCms {
    icc_calls: AtomicU32,
    cicp_calls: AtomicU32,
}

impl TrackingCms {
    fn new() -> Self {
        Self {
            icc_calls: AtomicU32::new(0),
            cicp_calls: AtomicU32::new(0),
        }
    }
    fn icc_calls(&self) -> u32 {
        self.icc_calls.load(Ordering::Relaxed)
    }
    fn cicp_calls(&self) -> u32 {
        self.cicp_calls.load(Ordering::Relaxed)
    }
    fn total_calls(&self) -> u32 {
        self.icc_calls() + self.cicp_calls()
    }
}

impl PluggableCms for TrackingCms {
    fn build_source_transform(
        &self,
        src: ColorProfileSource<'_>,
        dst: ColorProfileSource<'_>,
        _src_format: PixelFormat,
        _dst_format: PixelFormat,
        _options: &ConvertOptions,
    ) -> Built {
        if !involves_icc(&src, &dst) {
            return None;
        }
        self.icc_calls.fetch_add(1, Ordering::Relaxed);
        Some(Ok(Box::new(IdentityTransform)))
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

/// A pixel already in `transfer` and BT.2020, as an HDR origin's buffer is.
fn make_hdr_rgb8_buffer(transfer: TransferFunction) -> PixelBuffer {
    let descriptor = PixelDescriptor::RGB8_SRGB
        .with_transfer(transfer)
        .with_primaries(ColorPrimaries::Bt2020);
    PixelBuffer::from_vec(vec![128; 3], 1, 1, descriptor).unwrap()
}

/// SDR pixels whose origin was HDR are refused: their white has no place on PQ.
#[test]
fn sdr_pixels_refuse_an_hdr_output_profile() {
    let buf = make_rgb8_buffer(1, 1, [128, 128, 128]);
    let origin = ColorOrigin::from_cicp(pq_cicp());
    let refused = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::SameAsOrigin,
        PixelFormat::Rgb8,
        Some(&NoopCms),
    );
    assert!(matches!(
        refused.err().map(|e| e.decompose().0),
        Some(ConvertError::NoPath { .. })
    ));
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

    let ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::SameAsOrigin,
        PixelFormat::Rgb8,
        Some(&NoopCms),
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

    let ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::SameAsOrigin,
        PixelFormat::Rgb8,
        Some(&NoopCms),
    )
    .unwrap();

    let meta = ready.metadata();
    assert!(meta.icc.is_none());
    assert!(meta.cicp.is_none());
}

#[test]
fn same_as_origin_preserves_cicp_from_origin() {
    let buf = make_rgb8_buffer(1, 1, [128, 128, 128]);
    let origin = ColorOrigin::from_cicp(Cicp::SRGB);

    let ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::SameAsOrigin,
        PixelFormat::Rgb8,
        Some(&NoopCms),
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

    let ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::SameAsOrigin,
        PixelFormat::Rgba8,
        Some(&NoopCms),
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

    let ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::Named(Cicp::SRGB),
        PixelFormat::Rgb8,
        Some(&NoopCms),
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

    let ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::Named(Cicp::SRGB),
        PixelFormat::Rgb8,
        Some(&NoopCms),
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

    let ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::Icc(fake_icc.clone()),
        PixelFormat::Rgb8,
        Some(&IdentityCms),
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
    let ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::Icc(dst_icc),
        PixelFormat::Rgb8,
        Some(&IdentityCms),
    )
    .unwrap();

    let slice = ready.pixels();
    let row = slice.row(0);
    assert_eq!(&row[..3], &[100, 150, 200]);
}

#[test]
fn icc_target_without_source_icc_asks_the_cms() {
    let dst_icc: Arc<[u8]> = Arc::from(vec![2u8; 32].into_boxed_slice());
    let buf = make_rgb8_buffer(2, 1, [100, 150, 200]);
    let origin = ColorOrigin::assumed();

    let ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::Icc(dst_icc),
        PixelFormat::Rgb8,
        Some(&IdentityCms),
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

    let ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::SameAsOrigin,
        PixelFormat::Rgb8,
        Some(&NoopCms),
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

    let ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::SameAsOrigin,
        PixelFormat::Rgba8,
        Some(&NoopCms),
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

// ---------------------------------------------------------------------------
// ColorAuthority decision table tests (12 rows)
// ---------------------------------------------------------------------------

/// A buffer decoded under the origin's ICC, as `SameAsOrigin` expects: only the profile says what its
/// samples mean, so converting to that profile is an identity.
fn in_origin_color() -> PixelBuffer {
    let unknown = PixelDescriptor::RGB8
        .with_transfer(zenpixels_convert::TransferFunction::Unknown)
        .with_primaries(zenpixels_convert::ColorPrimaries::Unknown);
    PixelBuffer::from_vec(vec![128, 128, 128], 1, 1, unknown)
        .unwrap()
        .with_color_context(Arc::new(zenpixels_convert::ColorContext::from_icc(
            fake_icc().to_vec(),
        )))
}

fn fake_icc() -> Arc<[u8]> {
    Arc::from(vec![0xFFu8; 32].into_boxed_slice())
}

fn p3_cicp() -> Cicp {
    Cicp::DISPLAY_P3
}

fn srgb_cicp() -> Cicp {
    Cicp::SRGB
}

/// Row 1: Icc authority, no cicp, no icc, SameAsOrigin → no transform
#[test]
fn authority_row01_icc_none_none_same() {
    let buf = make_rgb8_buffer(1, 1, [128, 128, 128]);
    let origin = ColorOrigin::assumed();
    let cms = TrackingCms::new();
    let ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::SameAsOrigin,
        PixelFormat::Rgb8,
        Some(&cms),
    )
    .unwrap();
    assert_eq!(cms.total_calls(), 0);
    assert!(ready.metadata().icc.is_none());
    assert!(ready.metadata().cicp.is_none());
}

/// Row 2: Icc authority, no cicp, sRGB icc, SameAsOrigin → no transform
#[test]
fn authority_row02_icc_none_srgb_same() {
    let buf = in_origin_color();
    let origin = ColorOrigin::from_icc(fake_icc().to_vec());
    let cms = TrackingCms::new();
    let ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::SameAsOrigin,
        PixelFormat::Rgb8,
        Some(&cms),
    )
    .unwrap();
    assert_eq!(cms.total_calls(), 0, "SameAsOrigin should not invoke CMS");
    assert!(ready.metadata().icc.is_some());
}

/// Row 3: Icc authority, no cicp, P3 icc, Icc(srgb) → CMS from ICC bytes
#[test]
fn authority_row03_icc_none_p3icc_to_srgb() {
    let buf = make_rgb8_buffer(1, 1, [128, 128, 128]);
    let origin = ColorOrigin::from_icc(fake_icc().to_vec());
    let cms = TrackingCms::new();
    let _ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::Icc(fake_icc()),
        PixelFormat::Rgb8,
        Some(&cms),
    )
    .unwrap();
    assert_eq!(cms.icc_calls(), 1, "should use ICC transform");
    assert_eq!(cms.cicp_calls(), 0, "should not use CICP");
}

/// Row 4: Icc authority, P3 cicp, no icc, Icc(srgb): the CMS is asked.
#[test]
fn authority_row04_icc_p3cicp_noicc_fallback() {
    let buf = make_rgb8_buffer(1, 1, [128, 128, 128]);
    let origin = ColorOrigin::from_cicp(p3_cicp()).with_color_authority(ColorAuthority::Icc);
    let cms = TrackingCms::new();
    let _ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::Icc(fake_icc()),
        PixelFormat::Rgb8,
        Some(&cms),
    )
    .unwrap();
    assert_eq!(
        cms.icc_calls(),
        1,
        "a target ICC needs a transform, so the CMS is asked"
    );
}

/// Row 5: Icc authority, sRGB cicp, sRGB icc, SameAsOrigin → no transform
#[test]
fn authority_row05_icc_srgb_srgb_same() {
    let buf = in_origin_color();
    let origin = ColorOrigin::from_icc_and_cicp(fake_icc().to_vec(), srgb_cicp())
        .with_color_authority(ColorAuthority::Icc);
    let cms = TrackingCms::new();
    let ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::SameAsOrigin,
        PixelFormat::Rgb8,
        Some(&cms),
    )
    .unwrap();
    assert_eq!(cms.total_calls(), 0, "SameAsOrigin should not invoke CMS");
    assert!(ready.metadata().icc.is_some());
    // Only the authority is signaled: a second signal could contradict it.
    assert_eq!(ready.metadata().cicp, None);
}

/// Row 6: Icc authority, P3 cicp, P3 icc, Icc(srgb) → CMS from ICC (ignore CICP)
#[test]
fn authority_row06_icc_p3cicp_p3icc_to_srgb() {
    let buf = make_rgb8_buffer(1, 1, [128, 128, 128]);
    let origin = ColorOrigin::from_icc_and_cicp(fake_icc().to_vec(), p3_cicp())
        .with_color_authority(ColorAuthority::Icc);
    let cms = TrackingCms::new();
    let _ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::Icc(fake_icc()),
        PixelFormat::Rgb8,
        Some(&cms),
    )
    .unwrap();
    assert_eq!(cms.icc_calls(), 1, "should use ICC (authoritative)");
    assert_eq!(cms.cicp_calls(), 0, "should not use CICP");
}

/// Row 7: Cicp authority, no cicp, no icc, SameAsOrigin → no transform
#[test]
fn authority_row07_cicp_none_none_same() {
    let buf = make_rgb8_buffer(1, 1, [128, 128, 128]);
    let origin = ColorOrigin::assumed().with_color_authority(ColorAuthority::Cicp);
    let cms = TrackingCms::new();
    let ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::SameAsOrigin,
        PixelFormat::Rgb8,
        Some(&cms),
    )
    .unwrap();
    assert_eq!(cms.total_calls(), 0);
    assert!(ready.metadata().icc.is_none());
}

/// Row 8: Cicp authority, sRGB cicp, any icc, SameAsOrigin → no transform
#[test]
fn authority_row08_cicp_srgb_anyicc_same() {
    let buf = make_rgb8_buffer(1, 1, [128, 128, 128]);
    let origin = ColorOrigin::from_icc_and_cicp(fake_icc().to_vec(), srgb_cicp())
        .with_color_authority(ColorAuthority::Cicp);
    let cms = TrackingCms::new();
    let ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::SameAsOrigin,
        PixelFormat::Rgb8,
        Some(&cms),
    )
    .unwrap();
    assert_eq!(cms.total_calls(), 0, "SameAsOrigin should not invoke CMS");
    assert_eq!(ready.metadata().cicp, Some(srgb_cicp()));
}

/// Row 9: Cicp authority, P3 cicp, no icc, Icc(srgb): the CMS is asked.
#[test]
fn authority_row09_cicp_p3_noicc_to_srgb() {
    let buf = make_rgb8_buffer(1, 1, [128, 128, 128]);
    let origin = ColorOrigin::from_cicp(p3_cicp());
    let cms = TrackingCms::new();
    let _ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::Icc(fake_icc()),
        PixelFormat::Rgb8,
        Some(&cms),
    )
    .unwrap();
    assert_eq!(
        cms.icc_calls(),
        1,
        "a target ICC needs a transform, so the CMS is asked"
    );
}

/// Row 10: Cicp authority, P3 cicp, P3 icc, Icc(srgb) → CMS uses ICC bytes
/// (build_cms_transform always uses ICC bytes when available, regardless of authority)
#[test]
fn authority_row10_cicp_p3_p3icc_to_srgb() {
    let buf = make_rgb8_buffer(1, 1, [128, 128, 128]);
    let origin = ColorOrigin::from_icc_and_cicp(fake_icc().to_vec(), p3_cicp());
    let cms = TrackingCms::new();
    let _ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::Icc(fake_icc()),
        PixelFormat::Rgb8,
        Some(&cms),
    )
    .unwrap();
    assert_eq!(
        cms.icc_calls(),
        1,
        "should use ICC bytes (only ICC path available)"
    );
}

/// Row 11: Cicp authority, no cicp, P3 icc, Icc(srgb) → fallback CMS from ICC
#[test]
fn authority_row11_cicp_nocicp_p3icc_fallback() {
    let buf = make_rgb8_buffer(1, 1, [128, 128, 128]);
    let origin =
        ColorOrigin::from_icc(fake_icc().to_vec()).with_color_authority(ColorAuthority::Cicp);
    let cms = TrackingCms::new();
    let _ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::Icc(fake_icc()),
        PixelFormat::Rgb8,
        Some(&cms),
    )
    .unwrap();
    assert_eq!(cms.icc_calls(), 1, "should fall back to ICC transform");
    assert_eq!(cms.cicp_calls(), 0, "should not use CICP (none available)");
}

/// Row 12: Icc authority, P3 cicp, P3 icc, SameAsOrigin → no transform
#[test]
fn authority_row12_icc_p3_p3icc_same() {
    let buf = in_origin_color();
    let origin = ColorOrigin::from_icc_and_cicp(fake_icc().to_vec(), p3_cicp())
        .with_color_authority(ColorAuthority::Icc);
    let cms = TrackingCms::new();
    let ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::SameAsOrigin,
        PixelFormat::Rgb8,
        Some(&cms),
    )
    .unwrap();
    assert_eq!(cms.total_calls(), 0, "SameAsOrigin should not invoke CMS");
    assert!(ready.metadata().icc.is_some());
    // Only the authority is signaled: a second signal could contradict it.
    assert_eq!(ready.metadata().cicp, None);
}

// ---------------------------------------------------------------------------
// Additional coverage: both-missing, error paths, Named target
// ---------------------------------------------------------------------------

/// A CMS that always fails — exercises error mapping paths.
struct FailingCms;

impl PluggableCms for FailingCms {
    fn build_source_transform(
        &self,
        src: ColorProfileSource<'_>,
        dst: ColorProfileSource<'_>,
        _src_format: PixelFormat,
        _dst_format: PixelFormat,
        _options: &ConvertOptions,
    ) -> Built {
        involves_icc(&src, &dst)
            .then(|| Err(whereat::at!(CmsPluginError::msg("deliberate ICC failure"))))
    }
}

/// Icc authority, no icc, no cicp, Icc(target): the CMS is asked.
#[test]
fn authority_icc_neither_field_to_icc_target() {
    let buf = make_rgb8_buffer(1, 1, [128, 128, 128]);
    let origin = ColorOrigin::assumed(); // Icc, no icc, no cicp
    let cms = TrackingCms::new();
    let ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::Icc(fake_icc()),
        PixelFormat::Rgb8,
        Some(&cms),
    )
    .unwrap();
    assert_eq!(
        cms.icc_calls(),
        1,
        "a target ICC needs a transform, so the CMS is asked"
    );
    // The mock is an identity, so pixels are preserved.
    let slice = ready.pixels();
    let row = slice.row(0);
    assert_eq!(&row[..3], &[128, 128, 128]);
}

/// Cicp authority, no icc, no cicp, Icc(target): the CMS is asked.
#[test]
fn authority_cicp_neither_field_to_icc_target() {
    let buf = make_rgb8_buffer(1, 1, [128, 128, 128]);
    let origin = ColorOrigin::assumed().with_color_authority(ColorAuthority::Cicp);
    let cms = TrackingCms::new();
    let _ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::Icc(fake_icc()),
        PixelFormat::Rgb8,
        Some(&cms),
    )
    .unwrap();
    assert_eq!(
        cms.icc_calls(),
        1,
        "a target ICC needs a transform, so the CMS is asked"
    );
}

/// CMS ICC transform error is propagated
#[test]
fn authority_icc_transform_error_propagated() {
    let buf = make_rgb8_buffer(1, 1, [128, 128, 128]);
    let origin = ColorOrigin::from_icc(fake_icc().to_vec());
    let result = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::Icc(fake_icc()),
        PixelFormat::Rgb8,
        Some(&FailingCms),
    );
    assert!(result.is_err(), "ICC transform failure should propagate");
}

/// CICP-only source to an ICC target: the transform is the CMS's, and its failure surfaces.
#[test]
fn cicp_source_to_icc_target_surfaces_cms_failure() {
    let buf = make_rgb8_buffer(1, 1, [128, 128, 128]);
    let origin = ColorOrigin::from_cicp(p3_cicp());
    let result = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::Icc(fake_icc()),
        PixelFormat::Rgb8,
        Some(&FailingCms),
    );
    assert!(
        result.is_err(),
        "a failed transform surfaces, rather than labeling unconverted pixels"
    );
}

/// Icc authority, no ICC bytes, CICP present, to an ICC target: same.
#[test]
fn icc_authority_without_icc_to_icc_target_surfaces_cms_failure() {
    let buf = make_rgb8_buffer(1, 1, [128, 128, 128]);
    let origin = ColorOrigin::from_cicp(p3_cicp()).with_color_authority(ColorAuthority::Icc);
    let result = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::Icc(fake_icc()),
        PixelFormat::Rgb8,
        Some(&FailingCms),
    );
    assert!(
        result.is_err(),
        "a failed transform surfaces, rather than labeling unconverted pixels"
    );
}

/// CMS ICC fallback error is propagated (Cicp authority, no cicp, icc present)
#[test]
fn authority_cicp_icc_fallback_error_propagated() {
    let buf = make_rgb8_buffer(1, 1, [128, 128, 128]);
    let origin =
        ColorOrigin::from_icc(fake_icc().to_vec()).with_color_authority(ColorAuthority::Cicp);
    let result = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::Icc(fake_icc()),
        PixelFormat::Rgb8,
        Some(&FailingCms),
    );
    assert!(result.is_err(), "ICC fallback failure should propagate");
}

/// Named target never invokes CMS regardless of authority/metadata
#[test]
fn named_target_never_uses_cms_with_icc_authority() {
    let buf = make_rgb8_buffer(1, 1, [128, 128, 128]);
    let origin = ColorOrigin::from_icc_and_cicp(fake_icc().to_vec(), p3_cicp())
        .with_color_authority(ColorAuthority::Icc);
    let cms = TrackingCms::new();
    let ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::Named(srgb_cicp()),
        PixelFormat::Rgb8,
        Some(&cms),
    )
    .unwrap();
    assert_eq!(cms.total_calls(), 0, "Named target uses matrices, not CMS");
    assert_eq!(ready.metadata().cicp, Some(srgb_cicp()));
    assert!(ready.metadata().icc.is_none());
}

/// Named target never invokes CMS regardless of authority/metadata (CICP authority)
#[test]
fn named_target_never_uses_cms_with_cicp_authority() {
    let buf = make_rgb8_buffer(1, 1, [128, 128, 128]);
    let origin = ColorOrigin::from_cicp(p3_cicp());
    let cms = TrackingCms::new();
    let _ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::Named(srgb_cicp()),
        PixelFormat::Rgb8,
        Some(&cms),
    )
    .unwrap();
    assert_eq!(cms.total_calls(), 0, "Named target uses matrices, not CMS");
}

// ---------------------------------------------------------------------------
// HDR passthrough tests
// ---------------------------------------------------------------------------

fn pq_cicp() -> Cicp {
    Cicp::BT2100_PQ
}

fn hlg_cicp() -> Cicp {
    Cicp::BT2100_HLG
}

/// The RGB form: these pixels aren't YCbCr, so a target claims no matrix for them.
fn rgb(cicp: Cicp) -> Cicp {
    Cicp::new(cicp.color_primaries, cicp.transfer_characteristics, 0, true)
}

/// HDR origin → SameAsOrigin → allowed (passthrough)
#[test]
fn hdr_pq_same_as_origin_allowed() {
    let buf = make_hdr_rgb8_buffer(TransferFunction::Pq);
    let origin = ColorOrigin::from_cicp(pq_cicp());
    let ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::SameAsOrigin,
        PixelFormat::Rgb8,
        Some(&NoopCms),
    )
    .unwrap();
    assert_eq!(ready.metadata().cicp, Some(rgb(pq_cicp())));
}

/// HDR origin → HDR Named(PQ) target → allowed
#[test]
fn hdr_pq_to_hdr_named_allowed() {
    let buf = make_hdr_rgb8_buffer(TransferFunction::Pq);
    let origin = ColorOrigin::from_cicp(pq_cicp());
    let ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::Named(rgb(pq_cicp())),
        PixelFormat::Rgb8,
        Some(&NoopCms),
    )
    .unwrap();
    assert_eq!(ready.metadata().cicp, Some(rgb(pq_cicp())));
}

/// HDR (HLG) origin → HDR Named(HLG) target → allowed
#[test]
fn hdr_hlg_to_hdr_named_allowed() {
    let buf = make_hdr_rgb8_buffer(TransferFunction::Hlg);
    let origin = ColorOrigin::from_cicp(hlg_cicp());
    let _ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::Named(rgb(hlg_cicp())),
        PixelFormat::Rgb8,
        Some(&NoopCms),
    )
    .unwrap();
}

/// SDR origin → SDR target → no HDR guard triggered
#[test]
fn sdr_to_sdr_no_hdr_guard() {
    let buf = make_rgb8_buffer(1, 1, [128, 128, 128]);
    let origin = ColorOrigin::from_cicp(p3_cicp()); // P3 is SDR
    let _ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::Named(srgb_cicp()),
        PixelFormat::Rgb8,
        Some(&NoopCms),
    )
    .unwrap();
}

/// SDR origin → ICC target → no HDR guard triggered
#[test]
fn sdr_to_icc_no_hdr_guard() {
    let buf = make_rgb8_buffer(1, 1, [128, 128, 128]);
    let origin = ColorOrigin::from_icc(fake_icc().to_vec());
    let _ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::Icc(fake_icc()),
        PixelFormat::Rgb8,
        Some(&IdentityCms),
    )
    .unwrap();
}

/// No color metadata → no HDR guard triggered
#[test]
fn no_metadata_no_hdr_guard() {
    let buf = make_rgb8_buffer(1, 1, [128, 128, 128]);
    let origin = ColorOrigin::assumed();
    let _ready = EncodeReady::finalize(
        &buf,
        &origin,
        OutputProfile::Icc(fake_icc()),
        PixelFormat::Rgb8,
        Some(&IdentityCms),
    )
    .unwrap();
}

// ---------------------------------------------------------------------------
// HDR contract pin (zenpixels#39 Rung 1)
// ---------------------------------------------------------------------------

/// A PQ (BT.2100) source finalized `SameAsOrigin` passes the origin CICP
/// through for container signaling.
#[test]
fn pq_source_same_as_origin_pins_cicp_passthrough() {
    let desc = PixelDescriptor::RGB16_BT2100_PQ;
    let px = vec![0u8; 2 * 2 * desc.bytes_per_pixel()];
    let buffer = PixelBuffer::from_vec(px, 2, 2, desc).unwrap();
    let origin = ColorOrigin::from_cicp(Cicp::new(9, 16, 0, true));

    let ready = zenpixels_convert::EncodeReady::finalize(
        &buffer,
        &origin,
        OutputProfile::SameAsOrigin,
        desc.pixel_format(),
        None,
    )
    .expect("fast-path finalize of an already-PQ buffer");

    let meta = ready.metadata();
    assert_eq!(meta.cicp, Some(Cicp::new(9, 16, 0, true)));
    assert!(meta.icc.is_none(), "no ICC was supplied at origin");
}
