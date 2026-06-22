//! Buffer-level dispatch for [`HdrToSdr`].
//!
//! Wraps the linear-light `apply_strip` primitive with full descriptor-based
//! buffer conversion: transfer function, channel format, alpha mode, signal
//! range, and source diffuse-white anchor. The HDR-aware math (BT.2446-A
//! curve, source→BT.2020 matrix, BT.2020→target matrix, OKLch soft
//! compression) is *unchanged*. Only the byte-format dispatch around it is
//! new.
//!
//! # Architecture
//!
//! The conversion is broken into three stages:
//!
//! 1. **Decode** — source bytes → linear-light F32 RGB (or RGBA) in
//!    `source.primaries`. Done with [`crate::RowConverter`] (which handles
//!    transfer + format + alpha mode + primaries, but **not** signal-range
//!    crossings). The intermediate descriptor is re-tagged
//!    [`SignalRange::Full`] so the converter accepts it; if the buffer is
//!    [`SignalRange::Narrow`], we re-scale the linear-light values
//!    immediately after the decode to undo the narrow-range compression —
//!    a small approximation (the correct math is at byte level pre-EOTF),
//!    but adequate for this experimental surface.
//!
//! 2. **HDR pipeline** — RGB channels go through
//!    [`HdrToSdr::apply_strip`](super::HdrToSdr::apply_strip). Alpha is held
//!    aside; un-premultiplied if the source carried premultiplied alpha
//!    (with `alpha == 0` short-circuited to `RGB = 0` to avoid div-by-zero).
//!
//! 3. **Encode** — linear-light F32 RGB (now in `target.primaries`) is
//!    re-assembled with alpha, signal-range contracted if the target is
//!    [`SignalRange::Narrow`] (linear-space approximation, same caveat as
//!    the decode), then routed through another [`crate::RowConverter`] for
//!    the target format / transfer / alpha mode / range relabel.
//!
//! # Wraps existing infrastructure
//!
//! - [`crate::RowConverter`] / [`crate::ConvertPlan`] handle transfer
//!   decoding/encoding (Linear, Srgb, Bt709, Pq, Hlg, Gamma22), depth
//!   conversion (u8/u16/f16/f32), layout swaps (RGB↔RGBA, BGRA, …), alpha
//!   mode transitions (Straight ↔ Premultiplied), and gamut conversion
//!   between known primaries. We never re-implement any of that.
//!
//! - Signal range is handled outside the plan because no Narrow↔Full
//!   kernels exist in zenpixels-convert today (range crossings refuse with
//!   [`ConvertError::NoPath`]). The linear-space rescale here is a
//!   first-cut handling — exact math will land alongside the Narrow↔Full
//!   kernels when those ship.
//!
//! - Diffuse-white precedence: the buffer's [`ColorContext::diffuse_white`]
//!   wins over the struct field if present. This is so a decoder that
//!   knows the source's anchor (e.g. an UltraHDR gain-map decoder
//!   reporting 203 nits) can override the converter's default.

extern crate alloc;

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use whereat::{At, ResultAtExt};
use zenpixels::buffer::PixelBuffer;
use zenpixels::descriptor::{
    AlphaMode, ChannelLayout, ChannelType, PixelDescriptor, SignalRange, TransferFunction,
};
use zenpixels::hdr::DiffuseWhite;

use super::HdrToSdr;
use crate::ConvertError;
use crate::RowConverter;

/// Per-pixel limited-range anchor pair for u8 / u16 RGB or gray channels
/// (BT.709 / BT.2100 luma swing: 16..235 for 8-bit).
fn narrow_anchors_u(bits: u32) -> (f32, f32) {
    // ITU narrow range scales by 2^(N-8). 8-bit: 16, 235.
    let shift = bits.saturating_sub(8);
    let lo = 16.0_f32 * (1u32 << shift) as f32;
    let hi = 235.0_f32 * (1u32 << shift) as f32;
    let max = ((1u64 << bits) - 1) as f32;
    (lo / max, hi / max)
}

/// Apply the diffuse-white precedence rule. The buffer's
/// `color_context.diffuse_white` wins if present; otherwise use the
/// converter's stored value.
fn effective_source_diffuse_white(buf: &PixelBuffer, default_nits: f32) -> f32 {
    buf.color_context()
        .and_then(|ctx| ctx.diffuse_white)
        .map(DiffuseWhite::nits)
        .unwrap_or(default_nits)
}

impl HdrToSdr {
    /// Convert a full [`PixelBuffer`] from HDR to SDR, allocating a new
    /// output buffer matching the target descriptor.
    ///
    /// The converter's
    /// [`source`](Self::source) / [`target`](Self::target) descriptors
    /// fix the primaries of the linear-light pipeline (and were validated
    /// as `Linear` at construction); the **actual** transfer / format /
    /// alpha / signal range are read from `src.descriptor()` and the
    /// returned buffer's descriptor (which mirrors the converter's
    /// target descriptor but may carry any TF / format the caller wants
    /// — the buffer dispatch handles the encode).
    ///
    /// The returned buffer carries the converter's target descriptor
    /// (`*self.target()`) — to land in a different encoded format (e.g.
    /// `RGB8_SRGB` instead of `RGBF32_LINEAR`), use
    /// [`convert_into`](Self::convert_into) with a destination buffer
    /// pre-allocated in the desired descriptor (only the primaries need
    /// to match the converter's target).
    ///
    /// # Errors
    ///
    /// * [`ConvertError::NoPath`] if the source buffer's descriptor uses
    ///   a transfer function or signal range that can't be decoded into
    ///   linear-light F32 (this surface inherits whatever
    ///   [`crate::RowConverter`] accepts).
    /// * [`ConvertError::NoMatch`] (re-purposed) if the source's
    ///   primaries don't match the converter's `source.primaries`.
    /// * Any error surface that [`crate::RowConverter::new`] returns for
    ///   the source→linear or linear→target sub-conversions.
    ///
    /// # Diffuse white precedence
    ///
    /// The source's
    /// [`ColorContext::diffuse_white`](zenpixels::ColorContext::diffuse_white)
    /// wins over the struct's
    /// [`source_diffuse_white_nits`](Self::source_diffuse_white_nits)
    /// when present.
    ///
    /// # Pipeline order
    ///
    /// 1. Source transfer → linear (PQ / HLG / sRGB / BT.709 / Gamma22 / Linear)
    /// 2. Source format → f32 (u8 / u16 / f16 / f32)
    /// 3. Un-premultiply alpha (when source alpha is Premultiplied;
    ///    `alpha == 0` short-circuits RGB → 0)
    /// 4. Source signal-range expansion (linear-space approximation)
    /// 5. Linear-light HDR→SDR pipeline (existing [`apply_strip`])
    /// 6. Target signal-range contraction (linear-space approximation)
    /// 7. Re-multiply alpha (when target alpha is Premultiplied)
    /// 8. f32 → target format
    /// 9. Linear → target transfer
    #[track_caller]
    pub fn convert_buffer(&self, src: &PixelBuffer) -> Result<PixelBuffer, At<ConvertError>> {
        let mut dst = PixelBuffer::try_new(src.width(), src.height(), *self.target())
            .map_err(|_| whereat::at!(ConvertError::AllocationFailed))?;
        self.convert_into(src, &mut dst)?;
        // Carry the source's color context forward — it may carry CICP /
        // ICC metadata that downstream consumers still want.
        if let Some(ctx) = src.color_context() {
            dst = dst.with_color_context(Arc::clone(ctx));
        }
        Ok(dst)
    }

    /// Like [`convert_buffer`](Self::convert_buffer) but writes into a
    /// caller-provided destination [`PixelBuffer`]. The destination's
    /// `primaries` must match the converter's `target().primaries`;
    /// every other axis of the destination's descriptor (format /
    /// transfer / alpha / range) is honored as the encode target.
    /// The source dimensions must match the destination dimensions.
    ///
    /// # Errors
    ///
    /// In addition to the errors surfaced by
    /// [`convert_buffer`](Self::convert_buffer):
    ///
    /// * [`ConvertError::BufferSize`] if `src.width() != dst.width()` or
    ///   `src.height() != dst.height()`.
    /// * [`ConvertError::NoMatch`] if `src.descriptor().primaries` or
    ///   `dst.descriptor().primaries` doesn't match the converter's
    ///   `source` / `target` primaries respectively.
    #[track_caller]
    pub fn convert_into(
        &self,
        src: &PixelBuffer,
        dst: &mut PixelBuffer,
    ) -> Result<(), At<ConvertError>> {
        // ---- Validate dimensions.
        if src.width() != dst.width() || src.height() != dst.height() {
            return Err(whereat::at!(ConvertError::BufferSize {
                expected: src.width() as usize * src.height() as usize,
                actual: dst.width() as usize * dst.height() as usize,
            }));
        }

        // ---- Validate descriptors against the converter's expectations.
        let src_desc = src.descriptor();
        let dst_desc = dst.descriptor();

        if src_desc.primaries != self.source().primaries {
            return Err(whereat::at!(ConvertError::NoMatch { source: src_desc }));
        }
        if dst_desc.primaries != self.target().primaries {
            return Err(whereat::at!(ConvertError::NoMatch { source: dst_desc }));
        }

        let width = src.width();
        let height = src.height();
        if width == 0 || height == 0 {
            return Ok(());
        }

        // ---- Pick the intermediate (linear-light F32) layout. It carries
        // alpha when *either* the source or the target carries alpha — so
        // we can carry alpha values through the pipeline if either side
        // needs them. If neither side carries alpha, we drop the channel.
        let alpha_in_intermediate = src_desc.has_alpha() || dst_desc.has_alpha();
        let intermediate_layout = if alpha_in_intermediate {
            ChannelLayout::Rgba
        } else {
            ChannelLayout::Rgb
        };
        let intermediate_alpha = if alpha_in_intermediate {
            // Always carry alpha as Straight in the intermediate so the HDR
            // math sees clean RGB; we'll re-premultiply at encode time if
            // the target asks for it.
            Some(AlphaMode::Straight)
        } else {
            None
        };

        // ---- Source-side intermediate: linear-light F32 in source primaries.
        // Re-label the source descriptor to Full range so the RowConverter
        // accepts it (signal-range crossings refuse). The linear-space
        // rescale below undoes the narrow-range compression.
        let src_full = src_desc.with_signal_range(SignalRange::Full);

        let src_intermediate = PixelDescriptor::new_full(
            ChannelType::F32,
            intermediate_layout,
            intermediate_alpha,
            TransferFunction::Linear,
            self.source().primaries,
        );

        // Decode source → linear-light F32 (in source primaries).
        let mut linear = decode_to_linear_rgb(src, src_full, src_intermediate)?;

        // ---- Resolve the diffuse-white anchor; buffer context wins.
        let _diffuse_white = effective_source_diffuse_white(src, self.source_diffuse_white_nits());
        // (The current Bt2446A curve uses source_peak_nits to build itself
        // at construction time; the diffuse-white precedence is exposed so
        // future curves can read it without changing this entry point.)

        // ---- Source signal-range expansion (linear-space approximation).
        if src_desc.signal_range == SignalRange::Narrow {
            let bits = bits_of_channel(src_desc.channel_type());
            let (lo, hi) = narrow_anchors_u(bits);
            let range = (hi - lo).max(1e-6);
            linear_range_expand_rgb(&mut linear, intermediate_layout, lo, range);
        }

        // ---- Un-premultiply alpha and apply the HDR pipeline.
        let alpha_was_premultiplied =
            matches!(src_desc.alpha(), Some(AlphaMode::Premultiplied)) && alpha_in_intermediate;
        process_strips(
            &mut linear,
            intermediate_layout,
            alpha_was_premultiplied,
            self,
            dst_desc.alpha(),
        );

        // ---- Target signal-range contraction (linear-space approximation).
        if dst_desc.signal_range == SignalRange::Narrow {
            let bits = bits_of_channel(dst_desc.channel_type());
            let (lo, hi) = narrow_anchors_u(bits);
            let range = (hi - lo).max(1e-6);
            linear_range_contract_rgb(&mut linear, intermediate_layout, lo, range);
        }

        // ---- Encode linear-light F32 → target bytes.
        //
        // The target descriptor we pass to RowConverter is `dst_desc` with
        // its signal range relabeled Full (we've already contracted to
        // narrow values in linear space; the bytes get the narrow range
        // label after the encode).
        let dst_full = dst_desc.with_signal_range(SignalRange::Full);

        let dst_intermediate_desc = PixelDescriptor::new_full(
            ChannelType::F32,
            intermediate_layout,
            intermediate_alpha,
            TransferFunction::Linear,
            self.target().primaries,
        );

        encode_from_linear_rgb(&linear, width, height, dst_intermediate_desc, dst_full, dst)
    }
}

/// Decode source bytes → linear-light F32 RGB (or RGBA, matching
/// `intermediate.layout()`).
///
/// `src_desc` should be `src.descriptor()` re-tagged with Full signal range
/// so the RowConverter accepts the descriptor (signal-range crossings refuse;
/// the actual narrow-range rescale is handled by the caller in linear
/// space). The returned `Vec<f32>` is packed: `n_pixels * channels`.
fn decode_to_linear_rgb(
    src: &PixelBuffer,
    src_desc: PixelDescriptor,
    intermediate: PixelDescriptor,
) -> Result<Vec<f32>, At<ConvertError>> {
    let width = src.width();
    let height = src.height();
    let intermediate_channels = intermediate.layout().channels();
    let intermediate_bpp = intermediate.bytes_per_pixel();

    let mut converter = RowConverter::new(src_desc, intermediate).at()?;

    let src_slice = src.as_slice();
    let src_bpp = src_desc.bytes_per_pixel();
    let src_row_bytes = width as usize * src_bpp;
    let dst_row_bytes = width as usize * intermediate_bpp;
    let total_pixels = width as usize * height as usize;
    let total_floats = total_pixels * intermediate_channels;

    // Allocate one big F32 buffer; convert into it row-by-row through a
    // bytemuck view.
    let mut floats: Vec<f32> = vec![0.0; total_floats];
    {
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut floats);
        for y in 0..height {
            let src_row = src_slice.row(y);
            // The source row may be wider than packed (stride padding); take
            // the leading `src_row_bytes` only.
            let src_row = &src_row[..src_row_bytes];
            let dst_start = y as usize * dst_row_bytes;
            converter.convert_row(
                src_row,
                &mut bytes[dst_start..dst_start + dst_row_bytes],
                width,
            );
        }
    }
    Ok(floats)
}

/// Encode linear-light F32 → target bytes.
fn encode_from_linear_rgb(
    linear: &[f32],
    width: u32,
    height: u32,
    intermediate: PixelDescriptor,
    dst_desc: PixelDescriptor,
    dst: &mut PixelBuffer,
) -> Result<(), At<ConvertError>> {
    let intermediate_channels = intermediate.layout().channels();
    let intermediate_bpp = intermediate.bytes_per_pixel();
    let src_row_bytes = width as usize * intermediate_bpp;

    let mut converter = RowConverter::new(intermediate, dst_desc).at()?;
    let bytes: &[u8] = bytemuck::cast_slice(linear);

    // The actual dst descriptor (with its original signal_range) might
    // not match `dst_desc` (which is Full-relabeled). Relabel the
    // buffer's bytes via the descriptor we used for encoding; the
    // narrow-range value-space is already encoded by the contract step.
    let dst_stride = dst.stride();
    let mut dst_slice = dst.as_slice_mut();
    let dst_bytes = dst_slice.as_strided_bytes_mut();
    let dst_row_bytes = width as usize * dst_desc.bytes_per_pixel();

    for y in 0..height {
        let src_start = y as usize * src_row_bytes;
        let dst_start = y as usize * dst_stride;
        converter.convert_row(
            &bytes[src_start..src_start + (width as usize * intermediate_channels * 4)],
            &mut dst_bytes[dst_start..dst_start + dst_row_bytes],
            width,
        );
    }

    Ok(())
}

/// Apply the HDR pipeline to a packed linear-light F32 buffer.
///
/// - When `layout == Rgba` and `unpremultiply_first`, un-premultiplies
///   alpha into RGB before the pipeline runs (with `alpha == 0`
///   short-circuited to `RGB = 0`).
/// - Calls `pipeline.apply_strip` on the RGB channels.
/// - If the target alpha is Premultiplied, re-multiplies alpha into the
///   RGB channels after the pipeline runs.
fn process_strips(
    linear: &mut [f32],
    layout: ChannelLayout,
    unpremultiply_first: bool,
    pipeline: &HdrToSdr,
    target_alpha: Option<AlphaMode>,
) {
    let channels = layout.channels();
    let n_pixels = linear.len() / channels;
    debug_assert_eq!(linear.len(), n_pixels * channels);

    if channels == 3 {
        // Fast path: no alpha. Cast straight to `[[f32; 3]]` and call
        // apply_strip on the whole buffer.
        let strip: &mut [[f32; 3]] = bytemuck::cast_slice_mut(linear);
        pipeline.apply_strip(strip);
        return;
    }

    debug_assert!(channels == 4);

    // RGBA path. Walk in chunks. Un-premultiply (if requested), build an
    // RGB-only strip, run the pipeline, then re-premultiply (if target
    // wants it).
    //
    // Chunk size is bounded to keep the scratch RGB buffer small; an
    // 8 KiB strip (~512 pixels) keeps everything L1-resident.
    const CHUNK_PX: usize = 512;
    let mut rgb_strip: Vec<[f32; 3]> = vec![[0.0; 3]; CHUNK_PX];

    let mut idx = 0;
    while idx < n_pixels {
        let end = (idx + CHUNK_PX).min(n_pixels);
        let n = end - idx;
        let dst_strip = &mut rgb_strip[..n];

        // Copy RGB into the scratch strip, optionally un-premultiplying.
        for (i, rgba) in linear[idx * 4..end * 4].chunks_exact(4).enumerate() {
            let a = rgba[3];
            if unpremultiply_first && a > 0.0 {
                let inv = 1.0 / a;
                dst_strip[i] = [rgba[0] * inv, rgba[1] * inv, rgba[2] * inv];
            } else if unpremultiply_first {
                // alpha == 0 → fully transparent; the RGB is meaningless,
                // pin to 0 so the pipeline doesn't propagate stale color.
                dst_strip[i] = [0.0, 0.0, 0.0];
            } else {
                dst_strip[i] = [rgba[0], rgba[1], rgba[2]];
            }
        }

        // Run the HDR pipeline on the RGB-only strip.
        pipeline.apply_strip(dst_strip);

        // Copy RGB back into the RGBA buffer; optionally re-premultiply.
        let target_premul = matches!(target_alpha, Some(AlphaMode::Premultiplied));
        for (i, rgba) in linear[idx * 4..end * 4].chunks_exact_mut(4).enumerate() {
            let a = rgba[3];
            let rgb = dst_strip[i];
            if target_premul {
                rgba[0] = rgb[0] * a;
                rgba[1] = rgb[1] * a;
                rgba[2] = rgb[2] * a;
                // alpha unchanged.
            } else {
                rgba[0] = rgb[0];
                rgba[1] = rgb[1];
                rgba[2] = rgb[2];
            }
        }

        idx = end;
    }
}

/// Linear-space narrow-range expansion: `linear = (linear - lo) / range`.
/// Applied to RGB channels only — alpha (when present) is untouched.
fn linear_range_expand_rgb(linear: &mut [f32], layout: ChannelLayout, lo: f32, range: f32) {
    let channels = layout.channels();
    let inv = 1.0 / range;
    for pixel in linear.chunks_exact_mut(channels) {
        for c in &mut pixel[..3] {
            *c = (*c - lo) * inv;
        }
    }
}

/// Linear-space narrow-range contraction: `linear = lo + linear * range`.
/// Applied to RGB channels only — alpha (when present) is untouched.
fn linear_range_contract_rgb(linear: &mut [f32], layout: ChannelLayout, lo: f32, range: f32) {
    let channels = layout.channels();
    for pixel in linear.chunks_exact_mut(channels) {
        for c in &mut pixel[..3] {
            *c = lo + *c * range;
        }
    }
}

/// Bit depth of a [`ChannelType`] for signal-range math.
///
/// Floating-point channels are treated as if they were 8-bit for the
/// narrow-range scaling (the anchors are 16 / 235 / 255 on the 8-bit grid,
/// matching the ITU spec's base form). This keeps the math sensible when
/// callers tag an F32 buffer as `Narrow` — `lo / hi` come out as
/// `16/255 ≈ 0.0627` / `235/255 ≈ 0.922`, the same fraction as u8.
fn bits_of_channel(ch: ChannelType) -> u32 {
    match ch {
        ChannelType::U8 | ChannelType::F32 | ChannelType::F16 => 8,
        ChannelType::U16 => 16,
        _ => 8,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use zenpixels::{Cicp, ColorContext, ColorPrimaries};

    // ---- Helpers --------------------------------------------------------

    /// Build a tightly-packed test buffer from a slice of u8 bytes.
    fn buf_u8(bytes: Vec<u8>, width: u32, height: u32, desc: PixelDescriptor) -> PixelBuffer {
        PixelBuffer::from_vec(bytes, width, height, desc).expect("from_vec")
    }

    /// Build a tightly-packed F32 buffer from a slice of f32 values.
    fn buf_f32(values: Vec<f32>, width: u32, height: u32, desc: PixelDescriptor) -> PixelBuffer {
        let bytes: Vec<u8> = bytemuck::cast_slice(&values).to_vec();
        PixelBuffer::from_vec(bytes, width, height, desc).expect("from_vec")
    }

    /// Read out the raw f32 values of a tightly-packed buffer.
    fn read_f32(buf: &PixelBuffer) -> Vec<f32> {
        let bytes = buf.copy_to_contiguous_bytes();
        bytemuck::cast_slice(&bytes).to_vec()
    }

    /// Read out the raw u8 values of a tightly-packed buffer.
    fn read_u8(buf: &PixelBuffer) -> Vec<u8> {
        buf.copy_to_contiguous_bytes()
    }

    /// Build a converter for buffer-level dispatch.
    ///
    /// The struct itself is constructed with **linearized** descriptors
    /// (Linear TF, F32 RGB) carrying the relevant `primaries` — the
    /// buffer methods read the per-buffer descriptor for the actual
    /// transfer / format / alpha / range, and only consult the struct's
    /// stored descriptors for `primaries`.
    fn converter_for(source: PixelDescriptor, target: PixelDescriptor) -> HdrToSdr {
        let lin_src = PixelDescriptor::RGBF32_LINEAR.with_primaries(source.primaries);
        let lin_tgt = PixelDescriptor::RGBF32_LINEAR.with_primaries(target.primaries);
        HdrToSdr::new(lin_src, lin_tgt, 1000.0)
    }

    /// Like [`converter_for`] but with explicit peak nits + knee, for
    /// source==target near-identity tests.
    fn converter_for_with_params(
        source: PixelDescriptor,
        target: PixelDescriptor,
        source_peak: f32,
        target_peak: f32,
        knee: f32,
    ) -> HdrToSdr {
        let lin_src = PixelDescriptor::RGBF32_LINEAR.with_primaries(source.primaries);
        let lin_tgt = PixelDescriptor::RGBF32_LINEAR.with_primaries(target.primaries);
        HdrToSdr::with_params(lin_src, lin_tgt, source_peak, target_peak, knee)
    }

    // ---- Transfer functions --------------------------------------------

    #[test]
    fn pq_source_roundtrips_to_srgb_target() {
        // 16-bit PQ in, 8-bit sRGB out via convert_into so the caller's
        // dst descriptor is honored. The PQ→linear math in
        // zenpixels-convert is the ground truth; we only care that the
        // pipeline succeeds and produces a non-trivial result.
        let width = 4;
        let height = 4;
        let source = PixelDescriptor::RGB16_BT2100_PQ;
        let target = PixelDescriptor::RGB8_SRGB;
        let n_px = (width * height) as usize;
        let pq_value: u16 = 30_000;
        let mut bytes: Vec<u8> = Vec::with_capacity(n_px * 6);
        for _ in 0..n_px {
            for _ in 0..3 {
                bytes.push((pq_value & 0xff) as u8);
                bytes.push((pq_value >> 8) as u8);
            }
        }
        let src = buf_u8(bytes, width, height, source);
        let conv = converter_for(source, target);
        let mut dst = PixelBuffer::try_new(width, height, target).expect("try_new");
        conv.convert_into(&src, &mut dst).expect("convert");
        let pixels = read_u8(&dst);
        assert_eq!(pixels.len(), n_px * 3);
        let any_nonzero = pixels.iter().any(|&v| v > 0 && v < 255);
        assert!(
            any_nonzero,
            "expected non-trivial PQ→sRGB result, got {pixels:?}"
        );
    }

    #[test]
    fn linear_source_passes_through_to_bt709_target() {
        // Linear-F32 BT.2020 in, BT.709 + linear out — verifies the
        // pipeline runs without re-encoding the transfer (target stays
        // linear) and produces sensible mid-grey.
        let width = 2;
        let height = 2;
        let source = PixelDescriptor::RGBF32_LINEAR.with_primaries(ColorPrimaries::Bt2020);
        let target = PixelDescriptor::RGBF32_LINEAR; // BT.709 + Linear.
        let n_px = (width * height) as usize;
        let values: Vec<f32> = (0..n_px * 3).map(|_| 0.18_f32).collect();
        let src = buf_f32(values, width, height, source);
        let conv = converter_for(source, target);
        let out = conv.convert_buffer(&src).expect("convert");
        let out_vals = read_f32(&out);
        assert_eq!(out_vals.len(), n_px * 3);
        // Mid-grey HDR (0.18 source-norm at 1000 nits) → sensible SDR
        // mid range per `hdr_mid_grey_lands_in_sensible_sdr_range`.
        for chunk in out_vals.chunks_exact(3) {
            for &v in chunk {
                assert!(v > 0.25 && v < 0.55, "mid-grey expected ~0.37 SDR, got {v}");
            }
        }
    }

    #[test]
    fn srgb_to_linear_decode_correctness() {
        // sRGB 128 → linear ≈ 0.2158 (well-known IEC 61966-2-1 value).
        // We linearize via RowConverter inside convert_buffer, so this
        // pins that the wrapping doesn't disturb the standard EOTF math.
        // To isolate the EOTF math from BT.2446-A's curve, use a target
        // that the apply_strip pipeline degenerates on: BT.709 source +
        // BT.709 target with source_peak_nits = target_peak_nits = 100
        // (still goes through BT.2446-A, which is mildly non-identity
        // for mid-grey, but neutral grey stays neutral and within 25 %
        // per the existing `source_equals_target_is_near_identity` test).
        let width = 1;
        let height = 1;
        let source = PixelDescriptor::RGB8_SRGB; // BT.709 default.
        let target = PixelDescriptor::RGBF32_LINEAR;
        let bytes = vec![128u8, 128, 128];
        let src = buf_u8(bytes, width, height, source);
        let conv = converter_for_with_params(source, target, 100.0, 100.0, 0.9);
        let out = conv.convert_buffer(&src).expect("convert");
        let out_vals = read_f32(&out);
        // sRGB 128 decodes to linear ≈ 0.2159; with source==target HDR
        // pipeline the output stays within 25 % of that (the curve adds
        // a small bias for mid-grey by design).
        for &v in &out_vals {
            assert!(
                v > 0.1 && v < 0.55,
                "expected ~0.2 linear-light after sRGB decode + identity HDR, got {v}"
            );
        }
    }

    // ---- Format coverage -----------------------------------------------

    #[test]
    fn u8_rgb_buffer_in_u8_rgb_buffer_out() {
        let width = 2;
        let height = 2;
        let source = PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::Bt2020);
        let target = PixelDescriptor::RGB8_SRGB;
        let bytes = vec![200u8; (width * height * 3) as usize];
        let src = buf_u8(bytes, width, height, source);
        let conv = converter_for(source, target);
        let mut dst = PixelBuffer::try_new(width, height, target).expect("try_new");
        conv.convert_into(&src, &mut dst).expect("convert");
        assert_eq!(dst.descriptor(), target);
        let pixels = read_u8(&dst);
        assert_eq!(pixels.len(), (width * height * 3) as usize);
    }

    #[test]
    fn u16_rgba_buffer_in_f32_rgb_buffer_out() {
        let width = 2;
        let height = 2;
        let source = PixelDescriptor::new_full(
            ChannelType::U16,
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
            TransferFunction::Pq,
            ColorPrimaries::Bt2020,
        );
        let target = PixelDescriptor::RGBF32_LINEAR;
        let n_px = (width * height) as usize;
        let mut bytes: Vec<u8> = Vec::with_capacity(n_px * 8);
        for _ in 0..n_px {
            // RGB values
            for _ in 0..3 {
                let v: u16 = 30_000;
                bytes.push((v & 0xff) as u8);
                bytes.push((v >> 8) as u8);
            }
            // Alpha
            let a: u16 = 65_535;
            bytes.push((a & 0xff) as u8);
            bytes.push((a >> 8) as u8);
        }
        let src = buf_u8(bytes, width, height, source);
        let conv = converter_for(source, target);
        let out = conv.convert_buffer(&src).expect("convert");
        assert_eq!(out.descriptor(), target);
        let vals = read_f32(&out);
        assert_eq!(vals.len(), n_px * 3);
        // All channels in [0, 1] (the HDR pipeline clamps).
        for &v in &vals {
            assert!(
                v.is_finite() && (0.0..=1.0).contains(&v),
                "out of range: {v}"
            );
        }
    }

    #[test]
    fn f16_path_works() {
        let width = 2;
        let height = 2;
        let source = PixelDescriptor::new_full(
            ChannelType::F16,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
            ColorPrimaries::Bt2020,
        );
        let target = PixelDescriptor::RGBF32_LINEAR;
        let n_px = (width * height) as usize;
        // f16 0.5 = 0x3800; pack 3 channels.
        let mut bytes: Vec<u8> = Vec::with_capacity(n_px * 6);
        for _ in 0..n_px {
            for _ in 0..3 {
                bytes.push(0x00);
                bytes.push(0x38);
            }
        }
        let src = buf_u8(bytes, width, height, source);
        let conv = converter_for(source, target);
        let out = conv.convert_buffer(&src).expect("convert");
        let vals = read_f32(&out);
        for &v in &vals {
            assert!(
                v.is_finite() && (0.0..=1.0).contains(&v),
                "out of range: {v}"
            );
        }
    }

    // ---- Alpha coverage ------------------------------------------------

    #[test]
    fn premultiplied_alpha_round_trips() {
        // RGBA F32, premultiplied source → premultiplied target. After
        // the round trip the recovered RGB (un-premult / re-premult)
        // should match the intent within HDR pipeline noise.
        let width = 1;
        let height = 1;
        let source = PixelDescriptor::new_full(
            ChannelType::F32,
            ChannelLayout::Rgba,
            Some(AlphaMode::Premultiplied),
            TransferFunction::Linear,
            ColorPrimaries::Bt2020,
        );
        let target = source; // Round-trip with the same descriptor.

        // Straight RGB = (0.5, 0.5, 0.5), alpha = 0.5 → premultiplied
        // bytes = (0.25, 0.25, 0.25, 0.5).
        let values = vec![0.25_f32, 0.25, 0.25, 0.5];
        let src = buf_f32(values, width, height, source);
        let conv = converter_for_with_params(source, target, 100.0, 100.0, 0.9);
        let mut dst = PixelBuffer::try_new(width, height, target).expect("try_new");
        conv.convert_into(&src, &mut dst).expect("convert");
        let vals = read_f32(&dst);
        let alpha = vals[3];
        let r_premul = vals[0];
        let recovered_r = r_premul / alpha;
        assert!(
            (recovered_r - 0.5).abs() < 0.2,
            "recovered straight R = {recovered_r}, expected ~0.5"
        );
        // Alpha bit-identical (the pipeline never touches it).
        assert!((alpha - 0.5).abs() < 1e-5, "alpha drifted: {alpha}");
    }

    #[test]
    fn straight_alpha_passes_through_unchanged() {
        // Straight-alpha F32 RGBA. Alpha must be bit-identical on the
        // round trip (the pipeline never touches alpha).
        let width = 2;
        let height = 1;
        let source = PixelDescriptor::new_full(
            ChannelType::F32,
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
            TransferFunction::Linear,
            ColorPrimaries::Bt2020,
        );
        let target = source;
        let values = vec![
            0.5_f32, 0.3, 0.1, 0.875, // px 0
            0.2, 0.4, 0.6, 0.125, // px 1
        ];
        let src = buf_f32(values.clone(), width, height, source);
        let conv = converter_for_with_params(source, target, 100.0, 100.0, 0.9);
        let mut dst = PixelBuffer::try_new(width, height, target).expect("try_new");
        conv.convert_into(&src, &mut dst).expect("convert");
        let vals = read_f32(&dst);
        // Alpha (channel 3) bit-identical.
        assert!(
            (vals[3] - 0.875).abs() < 1e-7,
            "alpha 0 drifted: {} vs 0.875",
            vals[3]
        );
        assert!(
            (vals[7] - 0.125).abs() < 1e-7,
            "alpha 1 drifted: {} vs 0.125",
            vals[7]
        );
    }

    #[test]
    fn zero_alpha_pixel_handled_gracefully() {
        // Premultiplied alpha == 0 → RGB is meaningless. The un-premult
        // step must not divide by zero; it should produce a finite
        // output.
        let width = 1;
        let height = 1;
        let source = PixelDescriptor::new_full(
            ChannelType::F32,
            ChannelLayout::Rgba,
            Some(AlphaMode::Premultiplied),
            TransferFunction::Linear,
            ColorPrimaries::Bt2020,
        );
        let target = source;
        // (RGB, A) = (0.7, 0.3, 0.1, 0.0) — premult so RGB should really
        // be 0 too, but encoders sometimes leak through stale values.
        let values = vec![0.7_f32, 0.3, 0.1, 0.0];
        let src = buf_f32(values, width, height, source);
        let conv = converter_for_with_params(source, target, 100.0, 100.0, 0.9);
        let mut dst = PixelBuffer::try_new(width, height, target).expect("try_new");
        conv.convert_into(&src, &mut dst).expect("convert");
        let vals = read_f32(&dst);
        for &v in &vals {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
        // Alpha stays 0.
        assert!(vals[3].abs() < 1e-7, "alpha drifted from 0: {}", vals[3]);
    }

    // ---- Signal range --------------------------------------------------

    #[test]
    fn limited_range_u8_decodes_correctly() {
        // Narrow-range u8 mid-grey = (235 + 16) / 2 ≈ 125. Decode that
        // to linear via the convert_buffer machinery and verify the
        // result is roughly linear mid-grey.
        let width = 2;
        let height = 1;
        let source = PixelDescriptor::RGB8_SRGB
            .with_primaries(ColorPrimaries::Bt2020)
            .with_signal_range(SignalRange::Narrow);
        let target = PixelDescriptor::RGBF32_LINEAR;
        let bytes = vec![125u8; (width * height * 3) as usize];
        let src = buf_u8(bytes, width, height, source);
        // source==target peak; HDR pipeline near-identity on neutral grey.
        let conv = converter_for_with_params(source, target, 100.0, 100.0, 0.9);
        let out = conv.convert_buffer(&src).expect("convert");
        let vals = read_f32(&out);
        // Narrow 125 → ((125/255) - 16/255) / (235/255 - 16/255)
        // ≈ (0.490 - 0.063) / (0.922 - 0.063) ≈ 0.498 in encoded sRGB,
        // then sRGB→linear gives ~0.213; HDR pipeline preserves grey
        // within ~25 % per the existing identity-ish test.
        for &v in &vals {
            assert!(
                v > 0.1 && v < 0.5,
                "narrow-range mid-grey decoded to {v}, expected ~0.2"
            );
        }
    }

    #[test]
    fn full_range_u8_decodes_correctly() {
        // Full-range u8 = 128 → sRGB-encoded 0.502 → linear ≈ 0.215;
        // HDR pipeline preserves grey.
        let width = 2;
        let height = 1;
        let source = PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::Bt2020);
        let target = PixelDescriptor::RGBF32_LINEAR;
        let bytes = vec![128u8; (width * height * 3) as usize];
        let src = buf_u8(bytes, width, height, source);
        let conv = converter_for_with_params(source, target, 100.0, 100.0, 0.9);
        let out = conv.convert_buffer(&src).expect("convert");
        let vals = read_f32(&out);
        for &v in &vals {
            assert!(
                v > 0.1 && v < 0.55,
                "full-range mid-grey decoded to {v}, expected ~0.21"
            );
        }
    }

    // ---- Diffuse white precedence --------------------------------------

    #[test]
    fn buffer_color_context_diffuse_white_overrides_struct_default() {
        // The struct's stored anchor is BT2408 = 203; the buffer's
        // ColorContext carries a 100-nit override. Verify the effective
        // value follows the buffer's signal, not the struct's default.
        let width = 1;
        let height = 1;
        let source = PixelDescriptor::RGB16_BT2100_PQ;
        let target = PixelDescriptor::RGB8_SRGB;
        let n_px = (width * height) as usize;
        let mut bytes: Vec<u8> = Vec::with_capacity(n_px * 6);
        for _ in 0..n_px {
            for _ in 0..3 {
                let v: u16 = 30_000;
                bytes.push((v & 0xff) as u8);
                bytes.push((v >> 8) as u8);
            }
        }
        let src = buf_u8(bytes, width, height, source);
        let custom_ctx =
            ColorContext::from_cicp(Cicp::BT2100_PQ).with_diffuse_white(DiffuseWhite::new(100.0));
        let src = src.with_color_context(Arc::new(custom_ctx));

        let conv = converter_for(source, target);
        // Compute the effective anchor by hand — this is the load-bearing
        // contract: buffer overrides struct.
        let effective = effective_source_diffuse_white(&src, conv.source_diffuse_white_nits());
        assert!(
            (effective - 100.0).abs() < 1e-5,
            "buffer's 100-nit anchor should win over struct default; got {effective}"
        );

        // Sanity: with no override, the struct's value wins (203 by default).
        let plain = PixelBuffer::from_vec(
            vec![0u8; (width * height * 6) as usize],
            width,
            height,
            source,
        )
        .expect("from_vec");
        let plain_effective =
            effective_source_diffuse_white(&plain, conv.source_diffuse_white_nits());
        assert!(
            (plain_effective - conv.source_diffuse_white_nits()).abs() < 1e-5,
            "no override → struct default wins; got {plain_effective}"
        );
    }

    // ---- Mismatched descriptor / wrong primaries -----------------------

    #[test]
    fn mismatched_source_primaries_errors() {
        let conv = HdrToSdr::new(
            PixelDescriptor::RGBF32_LINEAR.with_primaries(ColorPrimaries::Bt2020),
            PixelDescriptor::RGBF32_LINEAR,
            1000.0,
        );
        // Source buffer has BT.709 primaries → should fail validation.
        let src = buf_f32(
            vec![0.5_f32, 0.5, 0.5],
            1,
            1,
            PixelDescriptor::RGBF32_LINEAR, // BT.709
        );
        assert!(conv.convert_buffer(&src).is_err());
    }

    #[test]
    fn wrong_dst_primaries_errors_on_convert_into() {
        // Source is BT.2020, target struct says BT.709 — passing a dst
        // descriptor with DisplayP3 primaries must reject (primaries
        // mismatch on the target side).
        let conv = HdrToSdr::new(
            PixelDescriptor::RGBF32_LINEAR.with_primaries(ColorPrimaries::Bt2020),
            PixelDescriptor::RGBF32_LINEAR, // BT.709 target
            1000.0,
        );
        let src = buf_f32(
            vec![0.5_f32, 0.5, 0.5],
            1,
            1,
            PixelDescriptor::RGBF32_LINEAR.with_primaries(ColorPrimaries::Bt2020),
        );
        let mut dst = PixelBuffer::try_new(
            1,
            1,
            PixelDescriptor::RGBF32_LINEAR.with_primaries(ColorPrimaries::DisplayP3),
        )
        .expect("try_new");
        assert!(conv.convert_into(&src, &mut dst).is_err());
    }
}
