//! Transfer-function-aware pixel conversion for zenpixels.
//!
//! This crate provides all the conversion logic that was split out of the
//! `zenpixels` interchange crate: row-level format conversion, gamut mapping,
//! codec format negotiation, and HDR tone mapping.
//!
//! # Re-exports
//!
//! All interchange types from `zenpixels` are re-exported at the crate root,
//! so downstream code can depend on `zenpixels-convert` alone.
//!
//! # Core concepts
//!
//! - **Format negotiation**: [`best_match`] picks the cheapest conversion
//!   target from a codec's supported formats for a given source descriptor.
//!
//! - **Row conversion**: [`RowConverter`] pre-computes a conversion plan and
//!   converts rows with no per-row allocation, using SIMD where available.
//!
//! - **Codec helpers**: [`adapt::adapt_for_encode`] negotiates format and converts
//!   pixel data in one call, returning `Cow::Borrowed` when the input
//!   already matches a supported format.
//!
//! - **Extension traits**: [`TransferFunctionExt`], [`ColorPrimariesExt`],
//!   and `PixelBufferConvertExt` add conversion methods to interchange types.
//!
//! # Codec compliance guide
//!
//! This section describes how to write a codec that integrates correctly with
//! the zenpixels ecosystem. A "codec" here means any decoder or encoder crate
//! that produces or consumes pixel data.
//!
//! ## Design principles
//!
//! 1. **Codecs own I/O; zenpixels-convert owns pixel math.** A codec reads
//!    and writes its container format. All pixel format conversion, transfer
//!    function application, gamut mapping, and alpha handling is done by
//!    `zenpixels-convert`. Codecs should not re-implement conversion logic.
//!
//! 2. **`PixelFormat` is byte layout; `PixelDescriptor` is full meaning.**
//!    `PixelFormat` describes the physical byte arrangement (channel count,
//!    order, depth). `PixelDescriptor` adds color interpretation: transfer
//!    function, primaries, alpha mode, signal range. Codecs register
//!    `PixelDescriptor` values because negotiation needs the full picture.
//!
//! 3. **No silent lossy conversions.** Every operation that destroys
//!    information (alpha removal, depth reduction, gamut clipping) requires
//!    an explicit policy via [`ConvertOptions`]. Codecs must not silently
//!    clamp, truncate, or discard data.
//!
//! 4. **Pixels and metadata travel together.** [`ColorContext`] rides on
//!    [`PixelBuffer`] via `Arc` so ICC/CICP metadata follows pixel data
//!    through the pipeline. [`finalize_for_output`] couples converted pixels
//!    with matching encoder metadata atomically.
//!
//! 5. **Provenance enables lossless round-trips.** The cost model tracks
//!    where data came from ([`Provenance`]). A JPEG u8 decoded to f32 for
//!    resize reports zero loss when converting back to u8, because the
//!    origin precision was u8 all along.
//!
//! ## The pixel lifecycle
//!
//! Every image processing pipeline follows this flow:
//!
//! ```text
//! ┌──────────┐    ┌───────────┐    ┌───────────┐    ┌──────────┐
//! │  Decode   │───>│ Negotiate │───>│  Convert  │───>│  Encode  │
//! │          │    │           │    │           │    │          │
//! │ Produces: │    │ Picks:    │    │ Uses:     │    │ Consumes:│
//! │ PixelBuf  │    │ best fmt  │    │ RowConv.  │    │ EncReady │
//! │ ColorCtx  │    │ from list │    │ per-row   │    │ metadata │
//! │ ColorOrig │    │           │    │           │    │          │
//! └──────────┘    └───────────┘    └───────────┘    └──────────┘
//! ```
//!
//! ### Step 1: Decode
//!
//! The decoder produces pixel data in one of its natively supported formats,
//! wraps it in a [`PixelBuffer`], and extracts color metadata from the file.
//!
//! ```rust,ignore
//! // Decode raw pixels
//! let pixels: Vec<u8> = my_codec_decode(&file_bytes)?;
//! let desc = PixelDescriptor::RGB8_SRGB;
//! let buffer = PixelBuffer::from_vec(pixels, width, height, desc)?;
//!
//! // Extract color metadata for CMS integration
//! let color_ctx = match (icc_chunk, cicp_chunk) {
//!     (Some(icc), Some(cicp)) =>
//!         Some(Arc::new(ColorContext::from_icc_and_cicp(icc, cicp))),
//!     (Some(icc), None) =>
//!         Some(Arc::new(ColorContext::from_icc(icc))),
//!     (None, Some(cicp)) =>
//!         Some(Arc::new(ColorContext::from_cicp(cicp))),
//!     (None, None) => None,
//! };
//!
//! // Track provenance for re-encoding decisions
//! let origin = ColorOrigin::from_icc_and_cicp(icc, cicp);
//! // or: ColorOrigin::from_icc(icc)
//! // or: ColorOrigin::from_cicp(cicp)
//! // or: ColorOrigin::from_gama_chrm()  // PNG gAMA+cHRM
//! // or: ColorOrigin::assumed()         // no color metadata in file
//! ```
//!
//! **Rules for decoders:**
//!
//! - Register only formats the decoder natively produces. Do not list formats
//!   that require internal conversion — let the caller convert via
//!   [`RowConverter`]. If your JPEG decoder outputs u8 sRGB only, register
//!   `RGB8_SRGB`, not `RGBF32_LINEAR`.
//!
//! - Extract all available color metadata. Both ICC and CICP can coexist
//!   (AVIF/HEIF containers carry both). Record all of it on [`ColorContext`].
//!
//! - Build a [`ColorOrigin`] that records *how* the file described its color,
//!   not what the pixels are. This is immutable and used only at encode time
//!   for provenance decisions (e.g., "re-embed the original ICC profile").
//!
//! - Set `effective_bits` correctly on `FormatEntry`. A 10-bit AVIF source
//!   decoded to u16 has `effective_bits = 10`, not 16. A JPEG decoded to f32
//!   with debiased dequantization has `effective_bits = 10`. Getting this
//!   wrong makes the cost model over- or under-value precision.
//!
//! - Set `can_overshoot = true` only when output values exceed `[0.0, 1.0]`.
//!   This is rare — only JPEG f32 decode with preserved IDCT ringing.
//!
//! ### Step 2: Negotiate
//!
//! Before encoding, the pipeline must pick a format the encoder accepts.
//! Negotiation uses the two-axis cost model (effort vs. loss) weighted by
//! [`ConvertIntent`].
//!
//! Three entry points, from simplest to most flexible:
//!
//! - **[`best_match`]**: Pass a source descriptor, a list of supported
//!   descriptors, and an intent. Good for simple encode paths.
//!
//! - **[`best_match_with`]**: Like `best_match`, but each candidate carries
//!   a consumer cost ([`FormatOption`]). Use this when the encoder has fast
//!   internal conversion paths (e.g., a JPEG encoder with a fused f32→u8+DCT
//!   kernel can advertise `RGBF32_LINEAR` with low consumer cost).
//!
//! - **[`negotiate`]**: Full control. Explicit [`Provenance`] (so the cost
//!   model knows the data's true origin) plus consumer costs. Use this in
//!   processing pipelines where data has been widened from a lower-precision
//!   source (e.g., JPEG u8 decoded to f32 for resize — provenance says "u8
//!   origin", so converting back to u8 reports zero loss).
//!
//! ```rust,ignore
//! // Simple: "what format should I encode to?"
//! let target = best_match(
//!     buffer.descriptor(),
//!     &encoder_supported,
//!     ConvertIntent::Fastest,
//! ).ok_or("no compatible format")?;
//!
//! // With provenance: "this f32 data came from u8 JPEG"
//! let provenance = Provenance::with_origin_depth(ChannelType::U8);
//! let target = negotiate(
//!     current_desc,
//!     provenance,
//!     options.iter().copied(),
//!     ConvertIntent::Fastest,
//! );
//! ```
//!
//! **Rules for negotiation:**
//!
//! - Use [`ConvertIntent::Fastest`] when encoding. The encoder knows what it
//!   wants; get there with minimal work.
//!
//! - Use [`ConvertIntent::LinearLight`] for resize, blur, anti-aliasing.
//!   These operations need linear light for gamma-correct results.
//!
//! - Use [`ConvertIntent::Blend`] for compositing. This ensures premultiplied
//!   alpha for correct Porter-Duff math.
//!
//! - Use [`ConvertIntent::Perceptual`] for sharpening, contrast, saturation.
//!   These are perceptual operations that work best in sRGB or Oklab space.
//!
//! - Track provenance when data has been widened. If you decoded a JPEG (u8)
//!   into f32 for processing, tell the cost model via
//!   `Provenance::with_origin_depth(ChannelType::U8)`. Otherwise it will
//!   penalize the f32→u8 conversion as lossy when it's actually a lossless
//!   round-trip.
//!
//! - If an operation genuinely expands the data's gamut (e.g., saturation
//!   boost in BT.2020 that pushes colors outside sRGB), call
//!   [`Provenance::invalidate_primaries`] with the current working primaries.
//!   Otherwise the cost model will incorrectly report gamut narrowing as
//!   lossless.
//!
//! ### Step 3: Convert
//!
//! Once a target format is chosen, convert pixel data row-by-row.
//!
//! ```rust,ignore
//! let converter = RowConverter::new(source_desc, target_desc)?;
//! for y in 0..height {
//!     converter.convert_row(src_row, dst_row, width);
//! }
//! ```
//!
//! Or use the convenience function that combines negotiation and conversion:
//!
//! ```rust,ignore
//! let adapted = adapt_for_encode(
//!     raw_bytes, descriptor, width, rows, stride,
//!     &encoder_supported,
//! )?;
//! // adapted.data is Cow::Borrowed if no conversion needed
//! ```
//!
//! **Rules for conversion:**
//!
//! - Use [`RowConverter`], not hand-rolled conversion. It handles transfer
//!   functions, gamut matrices, alpha mode changes, depth scaling, Oklab,
//!   and byte swizzle correctly. It pre-computes the plan so there is
//!   zero per-row overhead.
//!
//! - For policy-sensitive conversions (when you need to control what lossy
//!   operations are allowed), use [`adapt_for_encode_explicit`] with
//!   [`ConvertOptions`]. This validates policies *before* doing work and
//!   returns specific errors like [`ConvertError::AlphaNotOpaque`] or
//!   [`ConvertError::DepthReductionForbidden`].
//!
//! - The conversion system handles three tiers internally:
//!   (a) Direct SIMD kernels for common pairs (byte swizzle, depth shift,
//!   transfer LUTs).
//!   (b) Composed multi-step plans for less common pairs.
//!   (c) Hub path through linear sRGB f32 as a universal fallback.
//!
//! ### Step 4: Encode
//!
//! The encoder receives pixel data in a format it natively supports and
//! must embed correct color metadata.
//!
//! For the atomic path (recommended), use [`finalize_for_output`]:
//!
//! ```rust,ignore
//! let ready = finalize_for_output(
//!     &buffer,
//!     &color_origin,
//!     OutputProfile::SameAsOrigin,
//!     target_format,
//!     &cms,
//! )?;
//!
//! // Pixels and metadata are guaranteed to match
//! encoder.write_pixels(ready.pixels())?;
//! encoder.write_icc(ready.metadata().icc.as_deref())?;
//! encoder.write_cicp(ready.metadata().cicp)?;
//! ```
//!
//! **Rules for encoders:**
//!
//! - Register only formats the encoder natively accepts. If your JPEG encoder
//!   takes u8 sRGB, register `RGB8_SRGB`. Don't also list `RGBA8_SRGB`
//!   unless you actually handle RGBA natively (not just by stripping alpha
//!   internally). Let the conversion system handle format changes.
//!
//! - If you have fast internal conversion paths, advertise them via
//!   [`FormatOption::with_cost`]. Example: a JPEG encoder with a fused
//!   f32→DCT path can accept `RGBF32_LINEAR` at `ConversionCost::new(5, 0)`,
//!   so negotiation will route f32 data directly to the encoder instead of
//!   doing a redundant f32→u8 conversion first.
//!
//! - Use [`finalize_for_output`] to bundle pixels and metadata atomically.
//!   This prevents the most common color management bug: pixel values that
//!   don't match the embedded ICC/CICP.
//!
//! - Always embed color metadata when the format supports it. Check
//!   `CodecFormats::icc_encode` and `CodecFormats::cicp` for your codec's
//!   capabilities. Omitting color metadata causes browsers and OS viewers
//!   to assume sRGB, which corrupts Display P3 and HDR content.
//!
//! - The [`OutputProfile`] enum controls what gets embedded:
//!   - [`OutputProfile::SameAsOrigin`]: Re-embed the original ICC/CICP from
//!     the source file. Used for transcoding without color changes.
//!   - [`OutputProfile::Named`]: Use a well-known CICP profile (sRGB, P3,
//!     BT.2020). Uses hardcoded gamut matrices, no CMS needed.
//!   - [`OutputProfile::Icc`]: Use specific ICC profile bytes. Requires a
//!     [`ColorManagement`] implementation.
//!
//! ## Format registry
//!
//! Every codec must declare its capabilities in a `CodecFormats` struct,
//! typically as a `pub static`. This serves as the single source of truth
//! for what the codec can produce and consume. See the `pipeline::registry`
//! module (requires the `pipeline` feature) for the full format table and
//! examples for each codec.
//!
//! ```rust,ignore
//! pub static MY_CODEC: CodecFormats = CodecFormats {
//!     name: "mycodec",
//!     decode_outputs: &[
//!         FormatEntry::standard(PixelDescriptor::RGB8_SRGB),
//!         FormatEntry::standard(PixelDescriptor::RGBA8_SRGB),
//!     ],
//!     encode_inputs: &[
//!         FormatEntry::standard(PixelDescriptor::RGB8_SRGB),
//!     ],
//!     icc_decode: true,   // extracts ICC profiles
//!     icc_encode: true,   // embeds ICC profiles
//!     cicp: false,        // no CICP support
//! };
//! ```
//!
//! ## Cost model
//!
//! Format negotiation uses a two-axis cost model separating **effort**
//! (CPU work) from **loss** (information destroyed). These are independent:
//! a fast conversion can be very lossy (f32 HDR → u8 clamp), and a slow
//! conversion can be lossless (u8 sRGB → f32 linear).
//!
//! [`ConvertIntent`] controls how the axes are weighted:
//!
//! | Intent         | Effort weight | Loss weight | Use case |
//! |----------------|---------------|-------------|----------|
//! | `Fastest`      | 4x            | 1x          | Encoding |
//! | `LinearLight`  | 1x            | 4x          | Resize, blur |
//! | `Blend`        | 1x            | 4x          | Compositing |
//! | `Perceptual`   | 1x            | 3x          | Color grading |
//!
//! Cost components are additive: total = transfer_cost + depth_cost +
//! layout_cost + alpha_cost + primaries_cost + consumer_cost +
//! suitability_loss. The lowest-scoring candidate wins.
//!
//! ## CMS integration
//!
//! Named profile conversions (sRGB ↔ Display P3 ↔ BT.2020) use hardcoded
//! 3×3 gamut matrices and need no CMS backend. ICC-to-ICC transforms
//! require a [`ColorManagement`] implementation, which is a compile-time
//! feature (e.g., `cms-moxcms`, `cms-lcms2`).
//!
//! Codecs that handle ICC profiles must:
//! 1. Extract ICC bytes on decode and store them on [`ColorContext`].
//! 2. Record provenance on [`ColorOrigin`].
//! 3. On encode, let [`finalize_for_output`] handle the ICC transform
//!    (if the target profile differs from the source) or pass-through
//!    (if `SameAsOrigin`).
//!
//! ## Error handling
//!
//! [`ConvertError`] provides specific variants so codecs can handle each
//! failure mode:
//!
//! - [`ConvertError::NoMatch`] — no supported format works for this source.
//!   The codec's format list may be too restrictive.
//! - [`ConvertError::NoPath`] — no conversion kernel exists between formats.
//!   Unusual; most pairs are covered.
//! - [`ConvertError::AlphaNotOpaque`] — `DiscardIfOpaque` policy was set
//!   but the data has semi-transparent pixels.
//! - [`ConvertError::DepthReductionForbidden`] — `Forbid` policy prevents
//!   narrowing (e.g., f32→u8).
//! - [`ConvertError::AllocationFailed`] — buffer allocation failed (OOM).
//! - [`ConvertError::CmsError`] — CMS transform failed (invalid ICC profile,
//!   unsupported color space, etc.).
//!
//! Codecs should match on specific variants and return actionable errors
//! to callers. Do not flatten `ConvertError` into a generic string.
//!
//! ## Checklist
//!
//! - [ ] Declare `CodecFormats` with correct `effective_bits` and `can_overshoot`
//! - [ ] Decode: extract ICC + CICP → [`ColorContext`]
//! - [ ] Decode: record provenance → [`ColorOrigin`]
//! - [ ] Encode: negotiate via [`best_match`] or [`adapt::adapt_for_encode`]
//! - [ ] Encode: convert via [`RowConverter`] (not hand-rolled)
//! - [ ] Encode: embed metadata via [`finalize_for_output`]
//! - [ ] Encode: embed ICC/CICP when the format supports it
//! - [ ] Handle [`ConvertError`] variants specifically
//! - [ ] Test round-trip: native format → encode → decode = lossless
//! - [ ] Test negotiation: `best_match(my_format, my_supported, Fastest)` picks identity

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

whereat::define_at_crate_info!(path = "zenpixels-convert/");

// Re-export all interchange types from zenpixels.
pub use zenpixels::*;

// Conversion modules.
pub(crate) mod convert;
pub mod error;
pub(crate) mod negotiate;

pub mod adapt;
pub mod builtin_profiles;
pub mod cms;
#[allow(
    dead_code,
    unused_variables,
    clippy::needless_return,
    clippy::excessive_precision,
    clippy::derivable_impls
)]
pub(crate) mod cms_lite;
#[cfg(feature = "cms-moxcms")]
pub mod cms_moxcms;
pub mod converter;
pub mod ext;
#[allow(
    dead_code,
    unexpected_cfgs,
    unused_variables,
    clippy::needless_return,
    clippy::excessive_precision,
    clippy::derivable_impls
)]
pub(crate) mod fast_gamut;
pub mod gamut;
pub mod hdr;
pub mod icc_profiles;
pub mod oklab;
pub mod output;
#[cfg(feature = "pipeline")]
pub mod pipeline;

// Re-export key conversion types at crate root.
pub use adapt::adapt_for_encode_explicit;
pub use convert::{ConvertPlan, convert_row};
pub use converter::RowConverter;
pub use error::ConvertError;
pub use negotiate::{
    ConversionCost, ConvertIntent, FormatOption, Provenance, best_match, best_match_with,
    conversion_cost, conversion_cost_with_provenance, ideal_format, negotiate,
};
#[cfg(feature = "pipeline")]
pub use pipeline::{
    CodecFormats, ConversionPath, FormatEntry, LossBucket, MatrixStats, OpCategory, OpRequirement,
    PathEntry, QualityThreshold, generate_path_matrix, matrix_stats, optimal_path,
};

// Re-export extension traits.
#[cfg(feature = "rgb")]
pub use ext::PixelBufferConvertTypedExt;
pub use ext::{ColorPrimariesExt, PixelBufferConvertExt, TransferFunctionExt};

// Re-export gamut conversion utilities.
pub use gamut::{
    GamutMatrix, apply_matrix_f32, apply_matrix_row_f32, apply_matrix_row_rgba_f32,
    conversion_matrix,
};

// Re-export HDR types and tone mapping.
#[cfg(feature = "std")]
pub use hdr::exposure_tonemap;
pub use hdr::{
    ContentLightLevel, HdrMetadata, MasteringDisplay, reinhard_inverse, reinhard_tonemap,
};

// Re-export CMS traits, enums, and implementations.
#[allow(deprecated)]
pub use cms::{
    CmsPluginError, ColorManagement, ColorPriority, PluggableCms, RenderingIntent, RowTransform,
    RowTransformMut,
};
// TODO: pub use cms_lite::ZenCmsLite once benchmarked on aarch64.
#[cfg(feature = "cms-moxcms")]
pub use cms_moxcms::MoxCms;

// Re-export output types.
pub use output::finalize_for_output_with;
#[allow(deprecated)]
pub use output::{EncodeReady, OutputMetadata, OutputProfile, finalize_for_output};
