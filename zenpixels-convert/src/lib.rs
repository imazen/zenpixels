//! # zenpixels-convert 0.5 — Greenfield Pixel Conversion Engine
//!
//! Composable, `no_std + alloc`, `#![forbid(unsafe_code)]` pixel conversion,
//! HDR tone mapping, gamut compression, ICC/CMS routing, gain-map reconstruction,
//! EXIF orientation transpose, and codec format negotiation for [`zenpixels`].
//!
//! ## Collision-Free Namespace Design
//!
//! This crate re-exports the entire [`zenpixels`] core surface (`pub use zenpixels::*;`)
//! without shadowing a single module name (`zenpixels::hdr` remains pure metadata;
//! tone mapping and gamut compression live in [`tonemap`], content light measurement
//! lives in [`measure`], gain maps live in `gainmap` when the `gainmap` feature is enabled).
//!
//! ## Key Architecture Highlights
//!
//! - **64-pixel stack-allocated SoA L1-strip execution engine:** zero heap scratch
//!   allocation per row, composable micro-kernels covering full & bit-packed
//!   (`P010`, `RGB565`, `RGB10A2`) layouts, Narrow↔Full studio range, Bayer 4×4
//!   ordered dithering, exact linear-light gamut matrices, CIE XYZ D65 Oklab,
//!   BT.2100-3 HLG display rendering, ST 2084 PQ, and BT.2446A tone mapping.
//! - **Compile-time safe [`RowPlan`] vs [`ImagePlan`]:** `RowPlan::build` accepts
//!   only `Option<Nits>` (impossible to pass `PeakSource::Measured` to a single-row
//!   converter); `ImagePlan::build` accepts [`PeakSource`] and streams peak
//!   measurement in O(1) row memory across [`PixelView`], [`PixelBuf`], and [`BandedBuf`].
//! - **Dead-stripping `const` [`Accelerators`]:** link only the SIMD kernels you
//!   use (`SCALAR`, `LAYOUT`, `WEB_SDR`, `HDR`, `OKLAB`, `ALL`), with verified
//!   0-bit parity between `SCALAR` and `ALL`.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

whereat::define_at_crate_info!(path = "zenpixels-convert/");

/// Full re-export of [`zenpixels`] core types (zero module name collisions!).
pub use zenpixels::*;

pub mod accel;
pub mod cms;
pub mod codes;
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
pub mod load_bearing;
pub mod measure;
pub mod negotiate;
#[cfg(feature = "oklab")]
pub mod oklab;
pub mod orient;
pub mod output;
#[cfg(feature = "perceptual")]
pub mod perceptual;
pub mod plan;
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
pub use enough::{Stop, StopReason, Unstoppable};
pub use error::ConvertError;
#[cfg(feature = "gainmap")]
pub use gainmap::{
    GainMapApplier, GainMapApplyConfig, GainMapDownsample, GainMapExtractConfig,
    GainMapOrientation, GainMapResample, GainMapRowExtractor, GainMapTarget, apply_gain_map,
    apply_gain_map_oriented, apply_gain_map_view, apply_gain_map_with, bake_dual_layer_orientation,
    bake_dual_layer_orientation_after, extract_gain_map, extract_gain_map_only,
};
pub use gamut::GamutMatrix;
pub use load_bearing::{
    AlphaCompaction, LoadBearingReport, PixelBufLoadBearingExt, PixelViewLoadBearingExt,
};
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
    EncodeReady, OutputPlan, OutputProfile, adapt_for_encode, adapt_for_encode_format,
};
pub use plan::{
    ImagePlan, RowPlan, convert_into, convert_to_buf, to_bgra8, to_gray8, to_rgb8, to_rgba8,
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
pub use ycbcr::{
    ChromaUpsample, YcbcrMatrix, YcbcrOptions, ycbcr_to_rgb_f32, ycbcr_to_rgb_from_planar,
    ycbcr_to_rgb_into, ycbcr_to_rgb_with_options, ycbcr_to_rgb_with_options_into,
};
