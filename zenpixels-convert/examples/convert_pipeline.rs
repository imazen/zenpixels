//! Common `zenpixels-convert` usages, each self-checking.
//!
//! This is the "caller also has `zenpixels-convert`" story: on top of the
//! `zenpixels` description layer you now get transfer-function-aware
//! conversion, format negotiation, orientation, ICC synthesis, and encode
//! finalization. It walks the pixel lifecycle the crate is built around:
//!
//! ```text
//! Decode ──> Negotiate ──> Convert ──> Encode
//! ```
//!
//! Run the whole gallery:   `cargo run -p zenpixels-convert --example convert_pipeline`
//! Run it as tests:         `cargo test -p zenpixels-convert --examples`

use zenpixels::buffer::{PixelBuffer, PixelSlice, PixelSliceMut};
use zenpixels::cicp::Cicp;
use zenpixels::color::ColorOrigin;
use zenpixels::descriptor::{PixelDescriptor, PixelFormat, TransferFunction};
use zenpixels::hdr::ContentLightLevel;
use zenpixels::orientation::Orientation;

use zenpixels_convert::PixelSliceOrientationExt;
use zenpixels_convert::adapt::adapt_for_encode_cow;
use zenpixels_convert::converter::RowConverter;
use zenpixels_convert::ext::{PixelBufferConvertExt, TransferFunctionExt};
use zenpixels_convert::icc_profiles::{SynthesizedIcc, synthesize_icc_for_cicp};
use zenpixels_convert::output::OutputProfile;
use zenpixels_convert::{ConvertIntent, ConvertPlan, best_match, finalize_for_output_with};

type Fallible = Result<(), Box<dyn std::error::Error>>;

fn main() {
    convert_a_whole_buffer_one_shot().unwrap();
    convert_a_buffer_ergonomically().unwrap();
    linearize_for_gamma_correct_work().unwrap();
    widen_depth_and_add_alpha().unwrap();
    negotiate_an_encoder_format();
    adapt_pixels_before_encoding().unwrap();
    stream_rows_through_a_converter().unwrap();
    convert_a_strip_with_a_plan().unwrap();
    finalize_metadata_for_the_encoder().unwrap();
    apply_a_transfer_function_to_one_value();
    reorient_pixels();
    synthesize_then_read_back_an_icc();
    #[cfg(feature = "estimation-experimental")]
    estimate_conversion_resources().unwrap();
    carry_hdr_metadata();
    println!("all zenpixels-convert examples passed");
}

/// The lowest-friction one-shot: hold a `PixelBuffer`, convert it.
fn convert_a_whole_buffer_one_shot() -> Fallible {
    // 2x1 opaque RGB8 (red, green).
    let rgb = PixelBuffer::from_vec(vec![255, 0, 0, 0, 255, 0], 2, 1, PixelDescriptor::RGB8_SRGB)?;

    // `convert_to` keeps width/height/descriptor together on the result — no
    // separate bookkeeping. (The old `convert_buffer` free function returned a
    // bare `Vec<u8>` and dropped that geometry; it is deprecated.)
    let bgra = rgb.convert_to(PixelDescriptor::BGRA8_SRGB)?;
    // R and B are swapped and an opaque alpha byte is appended.
    assert_eq!(
        &bgra.as_slice().row(0)[..8],
        &[0, 0, 255, 255, 0, 255, 0, 255]
    );
    Ok(())
}

/// The ergonomic path when you hold a `PixelBuffer`: one call, metadata kept.
fn convert_a_buffer_ergonomically() -> Fallible {
    let src = PixelBuffer::new(4, 4, PixelDescriptor::RGBA8_SRGB);

    // `PixelBufferConvertExt::convert_to` preserves dims, stride, and color
    // context — and reads the source descriptor off the buffer for you.
    let f32_linear = src.convert_to(PixelDescriptor::RGBAF32_LINEAR)?;
    assert_eq!(f32_linear.descriptor().transfer(), TransferFunction::Linear);
    assert_eq!((f32_linear.width(), f32_linear.height()), (4, 4));
    Ok(())
}

/// Move to linear light for gamma-correct resize/blur, then back.
fn linearize_for_gamma_correct_work() -> Fallible {
    let srgb = PixelBuffer::new(8, 8, PixelDescriptor::RGB8_SRGB);

    // `linearize` picks the matching linear float descriptor automatically.
    let linear = srgb.linearize()?;
    assert!(linear.descriptor().is_linear());

    // ... resize / blur here ... then re-encode the transfer curve.
    let back = linear.delinearize(TransferFunction::Srgb)?;
    assert_eq!(back.descriptor().transfer(), TransferFunction::Srgb);
    Ok(())
}

/// Depth and alpha adjustments read as verbs on the buffer.
fn widen_depth_and_add_alpha() -> Fallible {
    let u8_rgb = PixelBuffer::new(2, 2, PixelDescriptor::RGB8_SRGB);

    let u16_rgb = u8_rgb.try_widen_to_u16()?;
    assert_eq!(u16_rgb.descriptor().bytes_per_pixel(), 6); // 3 channels * u16

    let rgba = u8_rgb.try_add_alpha()?; // appends an opaque alpha channel
    assert!(rgba.has_alpha());
    Ok(())
}

/// Pick the encoder-supported format that costs the least to reach.
fn negotiate_an_encoder_format() {
    // What a hypothetical encoder accepts.
    let encoder_supports = [PixelDescriptor::RGB8_SRGB, PixelDescriptor::GRAY8_SRGB];

    // Encoding => `Fastest`: get to a supported format with minimal work.
    let target = best_match(
        PixelDescriptor::RGBA8_SRGB,
        &encoder_supports,
        ConvertIntent::Fastest,
    );
    // Dropping an (opaque) alpha to RGB8 is cheaper than going to gray.
    assert_eq!(target, Some(PixelDescriptor::RGB8_SRGB));
}

/// Convert *only if needed* right before encoding — zero-copy on the fast path.
fn adapt_pixels_before_encoding() -> Fallible {
    let pixels = vec![0u8; 4 * 4 * 3];
    let stride = 4 * 3; // tightly packed
    let encoder_supports = [PixelDescriptor::RGB8_SRGB];

    let encode_pixels = adapt_for_encode_cow(
        &pixels,
        PixelDescriptor::RGB8_SRGB, // source already matches a supported format
        4,
        4,
        stride,
        &encoder_supports,
    )?;
    // No conversion was necessary, so the data is borrowed, not copied.
    assert!(encode_pixels.is_borrowed());
    assert_eq!(
        encode_pixels.as_slice().descriptor(),
        PixelDescriptor::RGB8_SRGB
    );

    // `PixelCow::as_slice()` gives an encoder-ready, stride-aware view for
    // either the borrowed fast path or an owned conversion result.
    let slice = encode_pixels.as_slice();
    assert_eq!((slice.width(), slice.rows()), (4, 4));
    Ok(())
}

/// Streaming: build the plan once, convert into a destination you own.
fn stream_rows_through_a_converter() -> Fallible {
    // RGBA8 -> BGRA8 is a pure R<->B swizzle.
    let mut conv = RowConverter::new(PixelDescriptor::RGBA8_SRGB, PixelDescriptor::BGRA8_SRGB)?;
    assert!(!conv.is_identity());

    let src = [10u8, 20, 30, 40, 50, 60, 70, 80]; // two RGBA pixels
    let mut dst = [0u8; 8];

    // `convert_slice_into` is the no-alloc primitive: each slice carries its own
    // stride, so neither is passed separately, and the descriptors are checked
    // against the plan. It supersedes `convert_rows(src, src_stride, dst,
    // dst_stride, width, rows)`, whose six positional args nothing type-checks.
    let src_slice = PixelSlice::new_contiguous(&src, 2, 1, PixelDescriptor::RGBA8_SRGB)?;
    let dst_slice = PixelSliceMut::new_contiguous(&mut dst, 2, 1, PixelDescriptor::BGRA8_SRGB)?;
    conv.convert_slice_into(src_slice, dst_slice)?;

    assert_eq!(dst, [30, 20, 10, 40, 70, 60, 50, 80]); // R and B swapped
    Ok(())
}

/// One-off plan execution: build the plan once, then convert row strips.
/// `ConvertPlan::convert_row` creates scratch per call; use `RowConverter`
/// below when repeated rows should reuse scratch.
fn convert_a_strip_with_a_plan() -> Fallible {
    let plan = ConvertPlan::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB)?;
    let (w, h) = (2u32, 2u32);
    let src = vec![9u8; (w * h) as usize * 3];
    let dst_bpp = PixelDescriptor::RGBA8_SRGB.bytes_per_pixel();
    let mut dst = vec![0u8; (w * h) as usize * dst_bpp];

    for y in 0..h as usize {
        let s = &src[y * w as usize * 3..][..w as usize * 3];
        let d = &mut dst[y * w as usize * dst_bpp..][..w as usize * dst_bpp];
        plan.convert_row(s, d, w);
    }
    // RGB8 -> RGBA8 fills an opaque alpha byte after each triple.
    assert_eq!(&dst[0..4], &[9, 9, 9, 255]);

    // The same conversion, bundled: wrap the (possibly strided) source as a
    // `PixelSlice` and let `RowConverter::convert_slice` size the output and
    // drive the row loop — no manual `alloc w*h*bpp` + `convert_row` scaffolding.
    let mut conv = RowConverter::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB)?;
    let src_slice =
        zenpixels::buffer::PixelSlice::new_contiguous(&src, w, h, PixelDescriptor::RGB8_SRGB)?;
    let out = conv.convert_slice(src_slice)?;
    assert_eq!(out.descriptor(), PixelDescriptor::RGBA8_SRGB);
    assert_eq!(&out.as_slice().row(0)[0..4], &[9, 9, 9, 255]);
    Ok(())
}

/// Atomically turn a processed buffer + its provenance into encoder-ready
/// pixels AND matching color metadata — the crate's recommended encode step.
///
/// This is the *color-correct* encode path: `finalize_for_output_with`
/// converts the pixels to the target profile and produces the exact ICC/CICP
/// to embed **together**, so they can never diverge (the classic "pixels are
/// Display P3 but the file says sRGB" bug). [`EncodeReady`] is the only way to
/// obtain both, and its `pixels()` come off a `PixelBuffer` — so, unlike
/// `Adapted::as_pixel_slice` above, they are always SIMD-aligned and never
/// fallible on alignment. Contrast with `adapt_for_encode`, which negotiates
/// pixel *format* only and leaves the color metadata to the caller (faster /
/// zero-copy when you are not changing color and manage the metadata yourself).
fn finalize_metadata_for_the_encoder() -> Fallible {
    let buffer = PixelBuffer::new(4, 4, PixelDescriptor::RGBA8_SRGB);
    let origin = ColorOrigin::from_cicp(Cicp::SRGB); // how the source declared color

    // `_with(..., None)` = no external CMS needed (sRGB in, sRGB out).
    let ready = finalize_for_output_with(
        &buffer,
        &origin,
        OutputProfile::SameAsOrigin,
        PixelFormat::Rgb8, // encoder wants RGB8
        None,
    )?;
    assert_eq!(
        ready.pixels().descriptor().pixel_format(),
        PixelFormat::Rgb8
    );
    assert_eq!(ready.metadata().cicp, Some(Cicp::SRGB));
    Ok(())
}

/// Apply a transfer curve to a single scalar (e.g. a lookup or unit test).
fn apply_a_transfer_function_to_one_value() {
    // sRGB midpoint 0.5 delinearizes to ~0.735 encoded.
    let encoded = TransferFunction::Srgb.delinearize(0.5);
    assert!((encoded - 0.735).abs() < 0.01);
    // ... and round-trips back.
    let linear = TransferFunction::Srgb.linearize(encoded);
    assert!((linear - 0.5).abs() < 1e-4);
}

/// Rotate pixels to match an EXIF orientation.
fn reorient_pixels() {
    let img = PixelBuffer::new(4, 2, PixelDescriptor::RGBA8_SRGB);
    let rotated = img.as_slice().apply_orientation(Orientation::Rotate90);
    // 90-degree rotation swaps the axes.
    assert_eq!((rotated.width(), rotated.height()), (2, 4));
}

/// Synthesize an ICC profile for a CICP signal, then read the signal back out.
fn synthesize_then_read_back_an_icc() {
    match synthesize_icc_for_cicp(Cicp::DISPLAY_P3) {
        SynthesizedIcc::Profile(bytes) => {
            // Recover the signal the way codecs do (cf. zencodec's color path):
            // a literal `cICP` tag if the profile embeds one, else fall back to
            // hash/matrix identification of a well-known profile.
            let recovered = zenpixels::icc::extract_cicp(&bytes)
                .or_else(|| zenpixels::icc::identify_common(&bytes).and_then(|id| id.to_cicp()));
            assert_eq!(recovered, Some(Cicp::DISPLAY_P3));
        }
        other => panic!("expected a bundled profile, got {other:?}"),
    }
}

/// Estimate memory + wall-time for a conversion before committing to it.
///
/// Requires `--features estimation-experimental` — the estimate surface is
/// experimental and off by default until a real scheduler consumer validates
/// its shape and its ±30 % accuracy contract.
#[cfg(feature = "estimation-experimental")]
fn estimate_conversion_resources() -> Fallible {
    let plan = ConvertPlan::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBAF32_LINEAR)?;
    let est = plan.estimate(1920, 1080);
    // A real conversion (not identity) reports a non-zero peak-memory estimate.
    assert!(est.peak_memory_bytes_est().unwrap_or(0) > 0);
    Ok(())
}

/// Build HDR signaling to hand to an encoder.
fn carry_hdr_metadata() {
    // HDR10 = PQ transfer + content light levels (MaxCLL / MaxFALL). Carry the
    // light-level and mastering-display structs directly; the transfer lives on
    // the descriptor. (`HdrMetadata` exists but is deprecated — see the report.)
    let cll = ContentLightLevel::new(1000, 400);
    assert_eq!(cll.max_content_light_level, 1000);

    let mastering = zenpixels::hdr::MasteringDisplay::HDR10_REFERENCE;
    assert!(mastering.max_luminance > 0.0);

    assert_eq!(
        PixelDescriptor::RGB16_BT2100_PQ.transfer(),
        TransferFunction::Pq
    );
}

#[test]
fn examples_run() {
    main();
}
