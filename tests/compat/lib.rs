#![deny(deprecated)]

use zenpixels::{PixelFormat, PixelBuf};
#[cfg(test)]
use zenpixels_convert::{ConvertPolicy, RowPlan};

// Public buffer boundaries use the very same type from the two crates.
pub fn through_conversion_crate(buffer: PixelBuf) -> zenpixels_convert::PixelBuf {
    buffer
}

// The 0.5 CICP -> ColorSpace declaration boundary.
pub fn declare_rgb(
    cicp: zenpixels::Cicp,
    alpha: zenpixels::AlphaMode,
) -> Result<zenpixels::ColorSpace, zenpixels::ColorError> {
    zenpixels::ColorSpace::from_cicp(cicp, alpha)
}

#[allow(dead_code)]
struct SharedCms;
impl zenpixels_convert::CmsTransform for SharedCms {
    fn transform_row(
        &self,
        src: &[u8],
        dst: &mut [u8],
        width: u32,
    ) -> Result<(), whereat::At<zenpixels_convert::ConvertError>> {
        dst[..width as usize * 3].copy_from_slice(&src[..width as usize * 3]);
        Ok(())
    }
}

#[test]
fn open_backend_traits_keep_external_implementations() {
    use zenpixels_convert::CmsTransform;
    let mut dst = [0; 3];
    SharedCms.transform_row(&[10, 20, 30], &mut dst, 1).unwrap();
    assert_eq!(dst, [10, 20, 30]);
}

#[test]
fn storage_and_prepared_rows_share_one_public_type_graph() {
    let buffer = PixelBuf::from_bytes(vec![0, 127, 255], 1, 1, PixelFormat::RGB8_SRGB).unwrap();
    let ptr = buffer.as_view().row(0).as_ptr();
    let buf = through_conversion_crate(buffer);
    assert_eq!(buf.as_view().row(0).as_ptr(), ptr);
    let plan = RowPlan::build(
        buf.format(),
        &zenpixels::formats::RGBA8_SRGB,
        &ConvertPolicy::lossless(),
        None,
    )
    .unwrap();
    let mut row = [0; 4];
    plan.convert_row(buf.as_view().row(0), &mut row, 1)
        .unwrap();
    assert_eq!(row, [0, 127, 255, 255]);
    let widened = zenpixels_convert::ImagePlan::build(
        buf.format(),
        &zenpixels::formats::RGB16_SRGB,
        &ConvertPolicy::lossless(),
        zenpixels_convert::PeakSource::None,
    )
    .unwrap()
    .execute(&buf.as_view())
    .unwrap();
    assert_eq!(widened.format(), &PixelFormat::RGB16_SRGB);
}

#[cfg(feature = "std")]
#[test]
fn published_plan_auto_traits_hold() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<RowPlan>();
}

#[cfg(feature = "interop")]
#[test]
fn typed_interop_keeps_layout_and_stride() {
    let buf = PixelBuf::from_typed(
        vec![rgb::RGBA::new(1u8, 2, 3, 255); 2],
        2,
        1,
        PixelFormat::RGBA8_SRGB,
    )
    .unwrap();
    assert_eq!(buf.as_view().as_imgref::<rgb::RGBA<u8>>().unwrap().width(), 2);
    let vec = buf.into_vec::<rgb::RGBA<u8>>().unwrap();
    assert_eq!(vec.len(), 2);
}
