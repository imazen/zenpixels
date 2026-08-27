//! Allocation + parity gate for `PixelBufferHdrConvertExt::convert_to_sdr`
//! (imazen/zenpixels#69).
//!
//! `convert_to_sdr` auto-measures the source peak before tone-mapping. It
//! used to do that by materializing a full linear-F32 copy of the image
//! (`convert_to(lin_desc)` — 16 bytes/pixel, thrown away right after
//! `measure_max`). The measurement now streams row-by-row through one
//! reused scratch row, so the only full-image allocation left is the
//! output buffer itself.
//!
//! Two gates:
//!   - **allocation**: peak bytes allocated during `convert_to_sdr` stay
//!     well below the size of the old F32 intermediate;
//!   - **parity**: the output is byte-identical to the explicit
//!     "materialize → `measure_max` → `convert_to_with_hdr_config`"
//!     sequence it replaced.
//!
//! Links the `allocation-counter` global allocator, so this file is its
//! own test binary (see the `[[test]]` entry in Cargo.toml). Gated on
//! `hdr-experimental` (required-features).

#![cfg(feature = "hdr-experimental")]

use zenpixels::buffer::PixelBuffer;
use zenpixels::hdr::{ContentLightLevel, DiffuseWhite};
use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, ColorPrimaries, PixelDescriptor, TransferFunction,
};
use zenpixels_convert::hdr::{CllMeasure, LightLevelMethod};
use zenpixels_convert::{HdrConfig, PixelBufferConvertExt, PixelBufferHdrConvertExt};

const W: u32 = 256;
const H: u32 = 128;

fn pq_u16_bt2020_rgba() -> PixelDescriptor {
    PixelDescriptor::new_full(
        ChannelType::U16,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        TransferFunction::Pq,
        ColorPrimaries::Bt2020,
    )
}

/// Deterministic PQ RGBA16 gradient with a bright highlight band so the
/// measured peak is well above the 100-nit floor (the measurement has to
/// matter for the parity gate to be meaningful).
fn pq_source() -> PixelBuffer {
    let desc = pq_u16_bt2020_rgba();
    let mut data = Vec::with_capacity((W * H) as usize * 8);
    for y in 0..H {
        for x in 0..W {
            let t = x as f32 / (W - 1) as f32;
            // PQ code values: ~0.0–0.75 (≈ 0–1000 nits), highlight rows at 0.9.
            let base = if y % 32 < 4 { 0.9 } else { 0.75 * t };
            let r = (base * 65535.0) as u16;
            let g = ((base * 0.8) * 65535.0) as u16;
            let b = ((base * 0.6) * 65535.0) as u16;
            let a = if x % 7 == 0 { 0x8000 } else { 0xFFFF };
            for c in [r, g, b, a] {
                data.extend_from_slice(&c.to_ne_bytes());
            }
        }
    }
    PixelBuffer::from_vec(data, W, H, desc).expect("pq source")
}

/// The pre-#69 algorithm, spelled out: materialize the linear image, measure
/// it whole, then convert with the measured peak.
fn reference_convert_to_sdr(src: &PixelBuffer, target: PixelDescriptor) -> PixelBuffer {
    let lin_desc = PixelDescriptor::new_full(
        ChannelType::F32,
        ChannelLayout::Rgba,
        src.descriptor().alpha(),
        TransferFunction::Linear,
        src.descriptor().primaries,
    );
    let linear = src.convert_to(lin_desc).expect("materialized linear copy");
    let cll = ContentLightLevel::measure_max(
        linear.as_slice(),
        DiffuseWhite::BT2408,
        LightLevelMethod::MaxRgb,
    )
    .expect("linear f32 measurable");
    let peak = f32::from(cll.max_content_light_level).max(100.0);
    src.convert_to_with_hdr_config(target, HdrConfig::for_source_peak(peak))
        .expect("reference convert")
}

#[test]
fn convert_to_sdr_matches_materialized_measure_then_convert() {
    let src = pq_source();
    let target = PixelDescriptor::RGBA8_SRGB;

    let expected = reference_convert_to_sdr(&src, target);
    let actual = src.convert_to_sdr(target).expect("convert_to_sdr");

    assert_eq!(actual.descriptor(), expected.descriptor());
    assert_eq!(
        actual.copy_to_contiguous_bytes(),
        expected.copy_to_contiguous_bytes(),
        "convert_to_sdr must be byte-identical to the materialize-then-measure path"
    );
}

#[test]
fn convert_to_sdr_does_not_materialize_a_full_f32_intermediate() {
    let src = pq_source();
    let target = PixelDescriptor::RGBA8_SRGB;
    let out_bytes = (W * H) as usize * target.bytes_per_pixel();
    let f32_intermediate_bytes = (W * H) as usize * 16;

    // Warm any lazily-initialised globals (LUTs, dispatch caches) outside
    // the measured region so they cannot be mistaken for the intermediate.
    let _ = src.convert_to_sdr(target).expect("warm-up");

    let mut result = None;
    let info = allocation_counter::measure(|| {
        result = Some(src.convert_to_sdr(target).expect("convert_to_sdr"));
    });
    drop(result);

    // The output buffer is unavoidable (borrow-in / own-out signature). The
    // old code added a full F32 image on top of it (4× an RGBA8 output).
    // Row-scoped scratch is O(row), so the peak stays far below
    // `out + f32_intermediate`; the bound leaves 50 % headroom for
    // per-row scratch + plan machinery.
    let ceiling = (out_bytes + f32_intermediate_bytes / 2) as u64;
    assert!(
        info.bytes_max < ceiling,
        "convert_to_sdr peak allocation {} B >= {} B: a full-image F32 \
         intermediate ({} B) is being materialized again (output alone is {} B)",
        info.bytes_max,
        ceiling,
        f32_intermediate_bytes,
        out_bytes
    );
    // `result` is still alive here, so exactly the output buffer's bytes
    // remain allocated — anything more is a leaked intermediate.
    assert_eq!(
        info.bytes_current, out_bytes as i64,
        "bytes still live after convert_to_sdr must be exactly the output buffer"
    );
}
