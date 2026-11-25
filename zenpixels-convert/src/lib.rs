//! # zenpixels-convert 0.5 — Pixel Conversion Engine
//!
//! Composable, `no_std + alloc` (disable default `std` feature), `#![forbid(unsafe_code)]`
//! pixel conversion, HDR tone mapping, gamut compression, ICC/CMS routing, gain-map
//! reconstruction, EXIF orientation, planar YCbCr/RGB/monochrome reconstruction, and codec
//! format negotiation for [`zenpixels`].
//!
//! Re-exports the entire [`zenpixels`] core surface (`pub use zenpixels::*;`) at
//! the crate root without shadowing a single module name.
//!
//! ## Which Entry Point Should I Use?
//!
//! - **Quick 8-bit sRGB:** [`to_rgb8`], [`to_gray8`] (both refuse unproven alpha
//!   discard via [`AlphaPolicy::DiscardIfOpaque`]), [`to_rgba8`], [`to_bgra8`].
//! - **One-shot image conversion:** [`convert_to_buf`] or [`convert_into`] (defaults to
//!   [`PeakSource::Measured`] only when HDR tone mapping needs a peak; use
//!   [`convert_to_buf_with_peak`] / [`convert_into_with_peak`] for explicit [`PeakSource`]).
//! - **Reusable image plan:** [`ImagePlan::build`] (accepts [`PeakSource`], works on
//!   [`PixelView`], [`PixelBuf`], and [`BandedBuf`]).
//! - **Streaming row loop (zero heap scratch):** [`RowPlan::build`] (accepts
//!   resolved `Option<Nits>` and executes rows via `convert_row`).
//! - **Planar YCbCr / GBR / RGB / monochrome reconstruction:**
//!   [`reconstruct_planar_to_buf`], [`reconstruct_planar_into`], [`reconstruct_planar_f32`].
//! - **Content compaction:** [`analyze_compaction`], [`compact_to_buf`], [`compact_in_place`].
//! - **Encoder output + CICP/ICC signaling:** [`adapt_for_encode_format`],
//!   [`adapt_for_encode`], or reusable [`OutputPlan::build`] with [`OutputSignaling`].
//!
//! [`ConvertPolicy::default()`] equals [`ConvertPolicy::lossless()`], a strict permission
//! preset that forbids lossy depth, gamut, range, and alpha changes. Opt in explicitly to
//! [`ConvertPolicy::permissive()`], [`ConvertPolicy::working_linear()`], or
//! [`ConvertPolicy::display_sdr`] when lossy conversion is desired.
//!
//! ## Quick Example
//!
//! ```rust
//! use zenpixels_convert::{ConvertPolicy, ImageSpec, PixelBuf, convert_to_buf, specs};
//!
//! let src = PixelBuf::from_typed(vec![255u8, 0, 0, 0, 255, 0], 2, 1, ImageSpec::RGB8_SRGB)?;
//! let linear = convert_to_buf(
//!     &src.as_view(),
//!     &specs::RGBF32_LINEAR,
//!     &ConvertPolicy::lossless(),
//! )?;
//! assert_eq!(linear.spec(), &specs::RGBF32_LINEAR);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Opt-in SIMD (`const` [`Accelerators`])
//!
//! Native pairs default to a compact scalar core with verified 0-bit parity
//! against SIMD kernels. Opt into SIMD globally via [`accel::enable_all`] /
//! [`accel::enable_web_sdr`] or per-policy via [`ConvertPolicy::with_accelerators`].

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

whereat::define_at_crate_info!(path = "zenpixels-convert/");

/// Full re-export of [`zenpixels`] core types (zero module name collisions!).
pub use zenpixels::*;

pub mod accel;
pub mod cms;
pub mod codes;
pub mod compact;
pub mod error;
pub(crate) mod f16_scalar;
#[cfg(feature = "gainmap")]
pub mod gainmap;
#[cfg(feature = "gainmap")]
pub(crate) mod gainmap_simd;
pub mod gamut;
#[cfg(feature = "hdr-transfers")]
pub(crate) mod hlg_render;
pub mod icc_profiles;
pub mod measure;
pub mod negotiate;
#[cfg(feature = "oklab")]
pub mod oklab;
pub mod orient;
pub mod output;
#[cfg(feature = "perceptual")]
pub mod perceptual;
pub mod plan;
pub mod planar;
pub mod policy;
pub mod scan;
pub(crate) mod strip;
pub(crate) mod strip_simd;
pub mod tonemap;
pub(crate) mod transfers;
pub mod ycbcr;

// Ergonomic top-level re-exports.
pub use accel::Accelerators;
#[cfg(feature = "cms-moxcms")]
pub use cms::MoxCms;
pub use cms::{CmsBackend, CmsTransform};
#[cfg(feature = "cms-lcms")]
pub use cms::{Lcms, LcmsUnoptimized};
pub use codes::{CodeMap, CodeMapError, CodeRole, CodesError, to_canonical};
pub use compact::{
    AlphaCompaction, CompactionReport, analyze_compaction, compact_in_place, compact_to_buf,
};
pub use enough::{Stop, StopReason, Unstoppable};
pub use error::ConvertError;
#[cfg(feature = "gainmap")]
pub use gainmap::{
    GainMapApplier, GainMapApplyConfig, GainMapDownsample, GainMapExtractConfig,
    GainMapOrientation, GainMapResample, GainMapRowExtractor, GainMapTarget,
    apply_gain_map_oriented, apply_gain_map_with, bake_dual_layer_orientation,
    bake_dual_layer_orientation_after, extract_gain_map, extract_gain_map_only,
};
pub use gamut::GamutMatrix;
pub use measure::{
    LightLevelHistogram, LightLevelMethod, measure_cll, measure_histogram, measure_max_smoothed,
    measure_peak,
};
pub use negotiate::{FormatCost, LossKind, Negotiate, NegotiatedFormat};
pub use orient::{
    apply_orientation, apply_orientation_banded, apply_orientation_banded_into,
    apply_orientation_into, apply_orientation_owned, bake_orientation, bake_orientation_banded,
    bake_orientation_owned,
};
pub use output::{
    EncodeReady, OutputPlan, OutputSignaling, adapt_for_encode, adapt_for_encode_format,
};
pub use plan::{
    ImagePlan, RowPlan, convert_into, convert_into_with_peak, convert_to_buf,
    convert_to_buf_with_peak, to_bgra8, to_gray8, to_rgb8, to_rgba8,
};
pub use planar::{
    ChromaUpsample, PlanarReconstructionOptions, reconstruct_planar_f32, reconstruct_planar_into,
    reconstruct_planar_to_buf,
};
pub use policy::{
    AlphaPolicy, ConvertPolicy, DepthPolicy, GamutPolicy, LumaCoefficients, LumaDomain, RangePolicy,
};
pub use scan::scan_facts;
#[cfg(feature = "bt2446a")]
pub use tonemap::Bt2446A;
#[cfg(feature = "oklab")]
pub use tonemap::SoftCompress;
pub use tonemap::{PreparedMapper, Requirements, StripMapper, ToneMapContext, ToneMapper};
pub use ycbcr::YcbcrMatrix;
