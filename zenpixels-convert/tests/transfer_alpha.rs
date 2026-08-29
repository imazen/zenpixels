//! Alpha must never pass through a transfer function.
//!
//! A transfer function maps **light** to **code value**. Alpha is a coverage
//! fraction, not light — pushing it through an EOTF/OETF is meaningless and
//! silently corrupts everything downstream (`StraightToPremul` then
//! multiplies the colour channels by the corrupted alpha).
//!
//! Every transfer kernel in `convert_kernels.rs` hands the whole flat row
//! (`width * channels` lanes) to a channel-agnostic SIMD EOTF/OETF — that
//! flat span is what makes them fast — and so each one must restore the
//! alpha lane afterwards, carrying it linearly across any depth change.
//! This suite pins that property on **every** transfer kernel, not one.
//!
//! Two deliberate choices in the fixtures, both load-bearing:
//!
//! * **Widths are not multiples of the vector width.** The kernels chunk 16
//!   lanes at a time (= 4 RGBA pixels), and `linear-srgb`'s SIMD slice
//!   entries have their own tails. A width that divides evenly cannot see a
//!   vector body diverging from its own scalar tail, so every case here is
//!   7 pixels: 28 RGBA lanes (one whole 16-lane chunk + a 12-lane remainder)
//!   and 14 GrayAlpha lanes (no whole chunk at all — pure tail).
//! * **Alpha is partially transparent.** 0.0 and 1.0 are fixed points of
//!   every transfer function in this crate, so an opaque fixture passes
//!   whether or not alpha is transferred. The pre-existing
//!   `*_opaque_alpha_preserved` tests are exactly that blind spot.

use zenpixels_convert::RowConverter;
use zenpixels_convert::policy::ConvertOptions;
use zenpixels_convert::{AlphaMode, ChannelLayout, ChannelType, PixelDescriptor, TransferFunction};

/// Partially-transparent u8 alphas, 7 pixels. Deliberately excludes 0 and
/// 255 — both are fixed points of every TF and witness nothing.
const ALPHA_U8: [u8; 7] = [1, 17, 64, 128, 191, 254, 200];

/// Colour ramp used for the RGB lanes; distinct from the alphas so a kernel
/// that mixed up the lane index is caught.
const COLOR_U8: [u8; 7] = [0, 32, 96, 128, 160, 224, 255];

const WIDTH: usize = 7;

fn rgba(ct: ChannelType, tf: TransferFunction) -> PixelDescriptor {
    PixelDescriptor::new(ct, ChannelLayout::Rgba, Some(AlphaMode::Straight), tf)
}

fn gray_alpha(ct: ChannelType, tf: TransferFunction) -> PixelDescriptor {
    PixelDescriptor::new(ct, ChannelLayout::GrayAlpha, Some(AlphaMode::Straight), tf)
}

/// RGBA8 source row: RGB from the colour ramp, A from the alpha ramp.
fn src_rgba8() -> Vec<u8> {
    let mut v = Vec::with_capacity(WIDTH * 4);
    for i in 0..WIDTH {
        v.extend_from_slice(&[COLOR_U8[i], COLOR_U8[i], COLOR_U8[i], ALPHA_U8[i]]);
    }
    v
}

/// RGBAF32 source row with the given per-pixel colour and the alpha ramp
/// carried linearly (`a / 255`).
fn src_rgba_f32(color: impl Fn(usize) -> f32) -> Vec<u8> {
    let mut v: Vec<f32> = Vec::with_capacity(WIDTH * 4);
    for (i, &a) in ALPHA_U8.iter().enumerate() {
        let c = color(i);
        v.extend_from_slice(&[c, c, c, f32::from(a) / 255.0]);
    }
    bytemuck::cast_slice(&v).to_vec()
}

/// GrayAlpha f32 source row — alpha lives at index 1, not 3.
fn src_gray_alpha_f32(color: impl Fn(usize) -> f32) -> Vec<u8> {
    let mut v: Vec<f32> = Vec::with_capacity(WIDTH * 2);
    for (i, &a) in ALPHA_U8.iter().enumerate() {
        v.extend_from_slice(&[color(i), f32::from(a) / 255.0]);
    }
    bytemuck::cast_slice(&v).to_vec()
}

/// Run one row and (when `__trace_ops` is on) assert the kernel under test
/// is the one that actually executed — so this suite provably covers every
/// kernel rather than whichever one the planner happened to pick.
fn run_row(
    from: PixelDescriptor,
    to: PixelDescriptor,
    opts: &ConvertOptions,
    src: &[u8],
    dst_bytes: usize,
    expect_step: &str,
) -> Vec<u8> {
    let mut conv = RowConverter::new_explicit(from, to, opts).unwrap();
    let mut dst = vec![0u8; dst_bytes];

    #[cfg(feature = "__trace_ops")]
    zenpixels_convert::__trace_ops::start_recording();

    conv.convert_row(src, &mut dst, WIDTH as u32);

    #[cfg(feature = "__trace_ops")]
    {
        let steps = zenpixels_convert::__trace_ops::stop_recording();
        assert!(
            steps.contains(&expect_step),
            "expected kernel {expect_step} to run for {from:?} -> {to:?}; got {steps:?}"
        );
    }
    #[cfg(not(feature = "__trace_ops"))]
    let _ = expect_step;

    dst
}

/// Assert the alpha lane of an f32 row equals the linearly-carried source
/// alpha exactly. Exact equality is correct here: a restored alpha is a copy
/// (or a plain `/255`), never the output of a curve.
fn assert_alpha_f32(dst: &[u8], channels: usize, alpha_idx: usize, expect: impl Fn(usize) -> f32) {
    let f: &[f32] = bytemuck::cast_slice(dst);
    for i in 0..WIDTH {
        let got = f[i * channels + alpha_idx];
        let want = expect(i);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "pixel {i}: alpha must be carried linearly, got {got} want {want} \
             (a transferred alpha would be ~{:.4})",
            linear_srgb::tf::srgb_to_linear(want)
        );
    }
}

/// The colour lanes must still be transferred — guards against a "fix" that
/// simply stops transferring anything.
fn assert_color_changed_f32(dst: &[u8], channels: usize, src_color: impl Fn(usize) -> f32) {
    let f: &[f32] = bytemuck::cast_slice(dst);
    let moved = (0..WIDTH).any(|i| {
        let c = src_color(i);
        c > 0.0 && c < 1.0 && (f[i * channels] - c).abs() > 1e-4
    });
    assert!(moved, "colour lanes were not transferred at all");
}

// ---------------------------------------------------------------------------
// u8 <-> f32 sRGB (depth conversion + transfer in one kernel)
// ---------------------------------------------------------------------------

#[test]
fn srgb_u8_to_linear_f32_carries_alpha() {
    let src = src_rgba8();
    let dst = run_row(
        rgba(ChannelType::U8, TransferFunction::Srgb),
        rgba(ChannelType::F32, TransferFunction::Linear),
        &ConvertOptions::permissive(),
        &src,
        WIDTH * 4 * 4,
        "SrgbU8ToLinearF32",
    );
    assert_alpha_f32(&dst, 4, 3, |i| f32::from(ALPHA_U8[i]) / 255.0);
    assert_color_changed_f32(&dst, 4, |i| f32::from(COLOR_U8[i]) / 255.0);
}

#[test]
fn linear_f32_to_srgb_u8_carries_alpha() {
    let src = src_rgba_f32(|i| f32::from(COLOR_U8[i]) / 255.0);
    let dst = run_row(
        rgba(ChannelType::F32, TransferFunction::Linear),
        rgba(ChannelType::U8, TransferFunction::Srgb),
        &ConvertOptions::permissive(),
        &src,
        WIDTH * 4,
        "LinearF32ToSrgbU8",
    );
    for i in 0..WIDTH {
        assert_eq!(
            dst[i * 4 + 3],
            ALPHA_U8[i],
            "pixel {i}: alpha must round-trip u8 unchanged"
        );
    }
}

// ---------------------------------------------------------------------------
// f32 <-> f32, same depth: sRGB / BT.709 / Gamma 2.2 / HLG (+ extended sRGB)
// ---------------------------------------------------------------------------

/// Table-driven over every same-depth f32 TF kernel. Each entry names the
/// kernel it must exercise, so adding a TF without an alpha restore fails
/// here rather than silently shipping.
#[test]
fn f32_transfer_kernels_carry_alpha() {
    let color = |i: usize| f32::from(COLOR_U8[i]) / 255.0;
    let src = src_rgba_f32(color);
    let clip = ConvertOptions::permissive();
    let noclip = ConvertOptions::permissive().with_clip_out_of_gamut(false);

    let cases: [(TransferFunction, &ConvertOptions, &str, &str); 5] = [
        (
            TransferFunction::Srgb,
            &clip,
            "SrgbF32ToLinearF32",
            "LinearF32ToSrgbF32",
        ),
        (
            TransferFunction::Srgb,
            &noclip,
            "SrgbF32ToLinearF32Extended",
            "LinearF32ToSrgbF32Extended",
        ),
        (
            TransferFunction::Bt709,
            &clip,
            "Bt709F32ToLinearF32",
            "LinearF32ToBt709F32",
        ),
        (
            TransferFunction::Gamma22,
            &clip,
            "Gamma22F32ToLinearF32",
            "LinearF32ToGamma22F32",
        ),
        (
            TransferFunction::Hlg,
            &clip,
            "HlgF32ToLinearF32",
            "LinearF32ToHlgF32",
        ),
    ];

    for (tf, opts, decode_step, encode_step) in cases {
        // Encoded -> linear (EOTF).
        let dst = run_row(
            rgba(ChannelType::F32, tf),
            rgba(ChannelType::F32, TransferFunction::Linear),
            opts,
            &src,
            WIDTH * 4 * 4,
            decode_step,
        );
        assert_alpha_f32(&dst, 4, 3, |i| f32::from(ALPHA_U8[i]) / 255.0);
        assert_color_changed_f32(&dst, 4, color);

        // Linear -> encoded (OETF).
        let dst = run_row(
            rgba(ChannelType::F32, TransferFunction::Linear),
            rgba(ChannelType::F32, tf),
            opts,
            &src,
            WIDTH * 4 * 4,
            encode_step,
        );
        assert_alpha_f32(&dst, 4, 3, |i| f32::from(ALPHA_U8[i]) / 255.0);
        assert_color_changed_f32(&dst, 4, color);
    }
}

// ---------------------------------------------------------------------------
// u16 <-> f32 HDR transfer kernels (HLG here; PQ already restored alpha and
// is pinned so the shared helper cannot regress it)
// ---------------------------------------------------------------------------

#[test]
fn hlg_u16_to_linear_f32_carries_alpha() {
    let mut v: Vec<u16> = Vec::with_capacity(WIDTH * 4);
    for i in 0..WIDTH {
        let c = u16::from(COLOR_U8[i]) * 257;
        let a = u16::from(ALPHA_U8[i]) * 257;
        v.extend_from_slice(&[c, c, c, a]);
    }
    let src: Vec<u8> = bytemuck::cast_slice(&v).to_vec();
    let dst = run_row(
        rgba(ChannelType::U16, TransferFunction::Hlg),
        rgba(ChannelType::F32, TransferFunction::Linear),
        &ConvertOptions::permissive(),
        &src,
        WIDTH * 4 * 4,
        "HlgU16ToLinearF32",
    );
    assert_alpha_f32(&dst, 4, 3, |i| {
        f32::from(u16::from(ALPHA_U8[i]) * 257) / 65535.0
    });
}

#[test]
fn linear_f32_to_hlg_u16_carries_alpha() {
    let src = src_rgba_f32(|i| f32::from(COLOR_U8[i]) / 255.0);
    let dst = run_row(
        rgba(ChannelType::F32, TransferFunction::Linear),
        rgba(ChannelType::U16, TransferFunction::Hlg),
        &ConvertOptions::permissive(),
        &src,
        WIDTH * 4 * 2,
        "LinearF32ToHlgU16",
    );
    let codes: &[u16] = bytemuck::cast_slice(&dst);
    for i in 0..WIDTH {
        let want = ((f32::from(ALPHA_U8[i]) / 255.0).clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        assert_eq!(
            codes[i * 4 + 3],
            want,
            "pixel {i}: alpha must be scaled linearly to u16, never HLG-encoded"
        );
    }
}

/// GrayAlpha through the **PQ** kernels — the one layout whose alpha the PQ
/// kernels used to get wrong.
///
/// They restored alpha under `if channels == 4`, which is true of RGBA/BGRA
/// but not of GrayAlpha (2 channels, alpha at index 1), so GrayAlpha alpha was
/// PQ-transferred like every other SDR kernel's. The planner reaches this:
/// `f32_tf_pair_steps` picks the step from the transfer function alone and
/// never consults the layout, so `GrayAlphaF32 PQ -> GrayAlphaF32 Linear`
/// dispatches `PqF32ToLinearF32` with `channels == 2`.
#[test]
fn gray_alpha_pq_carries_alpha() {
    let color = |i: usize| f32::from(COLOR_U8[i]) / 255.0;
    let src = src_gray_alpha_f32(color);
    let dst = run_row(
        gray_alpha(ChannelType::F32, TransferFunction::Pq),
        gray_alpha(ChannelType::F32, TransferFunction::Linear),
        &ConvertOptions::permissive(),
        &src,
        WIDTH * 2 * 4,
        "PqF32ToLinearF32",
    );
    assert_alpha_f32(&dst, 2, 1, |i| f32::from(ALPHA_U8[i]) / 255.0);
    assert_color_changed_f32(&dst, 2, color);
}

#[test]
fn pq_u16_round_trip_still_carries_alpha() {
    let mut v: Vec<u16> = Vec::with_capacity(WIDTH * 4);
    for i in 0..WIDTH {
        let c = u16::from(COLOR_U8[i]) * 257;
        let a = u16::from(ALPHA_U8[i]) * 257;
        v.extend_from_slice(&[c, c, c, a]);
    }
    let src: Vec<u8> = bytemuck::cast_slice(&v).to_vec();
    let dst = run_row(
        rgba(ChannelType::U16, TransferFunction::Pq),
        rgba(ChannelType::F32, TransferFunction::Linear),
        &ConvertOptions::permissive(),
        &src,
        WIDTH * 4 * 4,
        "PqU16ToLinearF32",
    );
    assert_alpha_f32(&dst, 4, 3, |i| {
        f32::from(u16::from(ALPHA_U8[i]) * 257) / 65535.0
    });
}

#[test]
fn pq_f32_still_carries_alpha() {
    let color = |i: usize| f32::from(COLOR_U8[i]) / 255.0;
    let src = src_rgba_f32(color);
    let dst = run_row(
        rgba(ChannelType::F32, TransferFunction::Pq),
        rgba(ChannelType::F32, TransferFunction::Linear),
        &ConvertOptions::permissive(),
        &src,
        WIDTH * 4 * 4,
        "PqF32ToLinearF32",
    );
    assert_alpha_f32(&dst, 4, 3, |i| f32::from(ALPHA_U8[i]) / 255.0);

    let dst = run_row(
        rgba(ChannelType::F32, TransferFunction::Linear),
        rgba(ChannelType::F32, TransferFunction::Pq),
        &ConvertOptions::permissive(),
        &src,
        WIDTH * 4 * 4,
        "LinearF32ToPqF32",
    );
    assert_alpha_f32(&dst, 4, 3, |i| f32::from(ALPHA_U8[i]) / 255.0);
}

// ---------------------------------------------------------------------------
// GrayAlpha: alpha is channel 1 of 2, not channel 3 of 4. A `channels == 4`
// test would miss this layout entirely.
// ---------------------------------------------------------------------------

#[test]
fn gray_alpha_f32_transfer_carries_alpha() {
    let color = |i: usize| f32::from(COLOR_U8[i]) / 255.0;
    let src = src_gray_alpha_f32(color);
    let dst = run_row(
        gray_alpha(ChannelType::F32, TransferFunction::Srgb),
        gray_alpha(ChannelType::F32, TransferFunction::Linear),
        &ConvertOptions::permissive(),
        &src,
        WIDTH * 2 * 4,
        "SrgbF32ToLinearF32",
    );
    assert_alpha_f32(&dst, 2, 1, |i| f32::from(ALPHA_U8[i]) / 255.0);
    assert_color_changed_f32(&dst, 2, color);
}

// ---------------------------------------------------------------------------
// Round trips: encode → linear → encode must return the alpha it started with
// ---------------------------------------------------------------------------

/// Every SDR transfer pairing, driven **both ways** through two separate
/// converters, must return the exact alpha it was handed.
///
/// Two converters rather than one: the planner peephole-cancels an
/// EOTF immediately followed by its own OETF (`can_cancel` in `convert.rs`),
/// so a single `RGBA8 sRGB -> RGBA8 sRGB` plan would collapse to `Identity`
/// and test nothing. Splitting the hop forces both kernels to run.
#[test]
fn alpha_survives_a_full_round_trip_through_every_sdr_pairing() {
    let clip = ConvertOptions::permissive();
    let noclip = ConvertOptions::permissive().with_clip_out_of_gamut(false);

    // f32 encoded <-> f32 linear, for each transfer function.
    let color = |i: usize| f32::from(COLOR_U8[i]) / 255.0;
    let src = src_rgba_f32(color);
    for (tf, opts, decode, encode) in [
        (
            TransferFunction::Srgb,
            &clip,
            "SrgbF32ToLinearF32",
            "LinearF32ToSrgbF32",
        ),
        (
            TransferFunction::Srgb,
            &noclip,
            "SrgbF32ToLinearF32Extended",
            "LinearF32ToSrgbF32Extended",
        ),
        (
            TransferFunction::Bt709,
            &clip,
            "Bt709F32ToLinearF32",
            "LinearF32ToBt709F32",
        ),
        (
            TransferFunction::Gamma22,
            &clip,
            "Gamma22F32ToLinearF32",
            "LinearF32ToGamma22F32",
        ),
        (
            TransferFunction::Hlg,
            &clip,
            "HlgF32ToLinearF32",
            "LinearF32ToHlgF32",
        ),
        (
            TransferFunction::Pq,
            &clip,
            "PqF32ToLinearF32",
            "LinearF32ToPqF32",
        ),
    ] {
        let lin = run_row(
            rgba(ChannelType::F32, tf),
            rgba(ChannelType::F32, TransferFunction::Linear),
            opts,
            &src,
            WIDTH * 4 * 4,
            decode,
        );
        let back = run_row(
            rgba(ChannelType::F32, TransferFunction::Linear),
            rgba(ChannelType::F32, tf),
            opts,
            &lin,
            WIDTH * 4 * 4,
            encode,
        );
        let out: &[f32] = bytemuck::cast_slice(&back);
        for i in 0..WIDTH {
            let want = f32::from(ALPHA_U8[i]) / 255.0;
            assert_eq!(
                out[i * 4 + 3].to_bits(),
                want.to_bits(),
                "{tf:?} round trip, pixel {i}: alpha changed ({} != {want})",
                out[i * 4 + 3]
            );
        }
    }
}

/// The u8 round trip: `RGBA8 sRGB -> RGBAF32 linear -> RGBA8 sRGB` must give
/// back the identical alpha **byte**. This is the pairing behind
/// `PixelBuffer::linearize()` and the `ConvertIntent::Blend` /
/// `ConvertIntent::LinearLight` plans.
#[test]
fn alpha_byte_survives_the_u8_srgb_round_trip() {
    let src = src_rgba8();
    let lin = run_row(
        rgba(ChannelType::U8, TransferFunction::Srgb),
        rgba(ChannelType::F32, TransferFunction::Linear),
        &ConvertOptions::permissive(),
        &src,
        WIDTH * 4 * 4,
        "SrgbU8ToLinearF32",
    );
    let back = run_row(
        rgba(ChannelType::F32, TransferFunction::Linear),
        rgba(ChannelType::U8, TransferFunction::Srgb),
        &ConvertOptions::permissive(),
        &lin,
        WIDTH * 4,
        "LinearF32ToSrgbU8",
    );
    for i in 0..WIDTH {
        assert_eq!(
            back[i * 4 + 3],
            ALPHA_U8[i],
            "pixel {i}: alpha byte changed across the u8 sRGB round trip"
        );
    }
}

/// The u16 HDR pairings, round-tripped the same way.
#[test]
fn alpha_survives_the_u16_hdr_round_trips() {
    let mut v: Vec<u16> = Vec::with_capacity(WIDTH * 4);
    for (i, &a) in ALPHA_U8.iter().enumerate() {
        let c = u16::from(COLOR_U8[i]) * 257;
        v.extend_from_slice(&[c, c, c, u16::from(a) * 257]);
    }
    let src: Vec<u8> = bytemuck::cast_slice(&v).to_vec();

    for (tf, decode, encode) in [
        (
            TransferFunction::Hlg,
            "HlgU16ToLinearF32",
            "LinearF32ToHlgU16",
        ),
        (TransferFunction::Pq, "PqU16ToLinearF32", "LinearF32ToPqU16"),
    ] {
        let lin = run_row(
            rgba(ChannelType::U16, tf),
            rgba(ChannelType::F32, TransferFunction::Linear),
            &ConvertOptions::permissive(),
            &src,
            WIDTH * 4 * 4,
            decode,
        );
        let back = run_row(
            rgba(ChannelType::F32, TransferFunction::Linear),
            rgba(ChannelType::U16, tf),
            &ConvertOptions::permissive(),
            &lin,
            WIDTH * 4 * 2,
            encode,
        );
        let codes: &[u16] = bytemuck::cast_slice(&back);
        for (i, &a) in ALPHA_U8.iter().enumerate() {
            assert_eq!(
                codes[i * 4 + 3],
                u16::from(a) * 257,
                "{tf:?} u16 round trip, pixel {i}: alpha code changed"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// End-to-end: the reported trigger, and the compounding premultiply path.
// ---------------------------------------------------------------------------

#[test]
fn linearize_rgba8_buffer_carries_alpha() {
    use zenpixels_convert::{PixelBuffer, PixelBufferConvertExt};

    let src = src_rgba8();
    let buf = PixelBuffer::from_vec(src, WIDTH as u32, 1, PixelDescriptor::RGBA8_SRGB).unwrap();
    let lin = buf.linearize().unwrap();
    let slice = lin.as_slice();
    let row: &[f32] = bytemuck::cast_slice(slice.row(0));
    for i in 0..WIDTH {
        let want = f32::from(ALPHA_U8[i]) / 255.0;
        assert_eq!(
            row[i * 4 + 3].to_bits(),
            want.to_bits(),
            "pixel {i}: linearize() must not push alpha through the sRGB EOTF"
        );
    }
}
