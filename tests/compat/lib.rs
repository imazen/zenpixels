#![deny(deprecated)]

use zenpixels::PixelBuffer;
#[cfg(test)]
use zenpixels::PixelDescriptor;
#[cfg(test)]
use zenpixels_convert::{ConvertPlan, PixelBufferConvertExt, RowConverter};

// Public buffer boundaries use the very same type from the two crates.
pub fn through_conversion_crate(buffer: PixelBuffer) -> zenpixels_convert::PixelBuffer {
    buffer
}

#[allow(dead_code)]
struct Shared;
impl zenpixels_convert::RowTransform for Shared {
    fn transform_row(&self, src: &[u8], dst: &mut [u8], width: u32) {
        dst[..width as usize * 3].copy_from_slice(&src[..width as usize * 3]);
    }
}
#[allow(dead_code)]
struct Stateful;
impl zenpixels_convert::RowTransformMut for Stateful {
    fn transform_row(&mut self, src: &[u8], dst: &mut [u8], width: u32) {
        dst[..width as usize * 3].copy_from_slice(&src[..width as usize * 3]);
    }
}

#[test]
fn open_backend_traits_keep_external_implementations() {
    use zenpixels_convert::{RowTransform, RowTransformMut};
    let mut dst = [0; 3];
    Shared
        .try_transform_row(&[10, 20, 30], &mut dst, 1)
        .unwrap();
    Stateful
        .try_transform_row(&[30, 20, 10], &mut dst, 1)
        .unwrap();
    assert_eq!(dst, [30, 20, 10]);
}

#[test]
fn storage_and_prepared_rows_share_one_public_type_graph() {
    let buffer =
        PixelBuffer::from_vec(vec![0, 127, 255], 1, 1, PixelDescriptor::RGB8_SRGB).unwrap();
    let ptr = buffer.as_slice().row(0).as_ptr();
    let parts = through_conversion_crate(buffer).into_parts();
    let restored = PixelBuffer::try_from_parts(parts)
        .map_err(|e| e.without_buffer())
        .unwrap();
    assert_eq!(restored.as_slice().row(0).as_ptr(), ptr);
    let mut worker = RowConverter::new(restored.descriptor(), PixelDescriptor::RGBA8_SRGB).unwrap();
    worker.prepare(1).unwrap();
    let mut row = [0; 4];
    worker
        .try_convert_row(restored.as_slice().row(0), &mut row, 1)
        .unwrap();
    assert_eq!(row, [0, 127, 255, 255]);
    let widened = restored.try_widen_to_u16().unwrap();
    assert_eq!(widened.descriptor(), PixelDescriptor::RGB16_SRGB);
    let proof =
        ConvertPlan::new_preserving_samples(restored.descriptor(), widened.descriptor()).unwrap();
    assert_eq!(proof.from(), PixelDescriptor::RGB8_SRGB);
    let identity = RowConverter::new(widened.descriptor(), widened.descriptor()).unwrap();
    assert!(identity.compose_preserving(&identity).is_some());
}

#[cfg(feature = "std")]
#[test]
fn published_worker_auto_traits_still_hold() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<RowConverter>();
}

#[cfg(feature = "interop")]
#[test]
fn typed_interop_keeps_layout_and_stride() {
    let typed = zenpixels::PixelBuffer::<rgb::RGBA<u8>>::from_pixels(
        vec![rgb::RGBA::new(1, 2, 3, 255); 2],
        2,
        1,
    )
    .unwrap()
    .with_descriptor(PixelDescriptor::RGBA8_SRGB);
    assert_eq!(typed.as_imgref().width(), 2);
    let packed: PixelBuffer<rgb::RGBA<u8>> = typed.into_contiguous();
    assert_eq!(
        packed
            .erase()
            .into_contiguous_pixels::<rgb::RGBA<u8>>()
            .unwrap()
            .len(),
        2
    );
}

#[cfg(feature = "experimental")]
#[test]
fn opt_in_estimation_and_hdr_names_are_identical() {
    use zenpixels_convert::{ComputeEnvironment, HdrConfig, ImageCharacteristics};
    let plan = ConvertPlan::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB).unwrap();
    let _ = plan.estimate_in(
        &ImageCharacteristics::new(10, 20, plan.from()),
        &ComputeEnvironment::new(),
    );
    let plan = ConvertPlan::new_with_hdr_config(
        PixelDescriptor::RGBF32_LINEAR,
        PixelDescriptor::RGB8_SRGB,
        HdrConfig::for_source_peak(1000.),
    )
    .unwrap();
    RowConverter::from_plan(plan).prepare(10).unwrap();
}

#[cfg(feature = "siblings")]
pub fn from_decoder(output: zencodec::decode::DecodeOutput) -> PixelBuffer {
    output.into_buffer()
}
#[cfg(feature = "siblings")]
pub fn pipeline_format(source: &dyn zenpipe::Source) -> zenpixels::PixelDescriptor {
    source.format()
}
