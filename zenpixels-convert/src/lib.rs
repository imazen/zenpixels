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
//! lives in [`measure`], gain maps live in [`gainmap`]).
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
pub mod error;
pub(crate) mod f16_scalar;
pub mod gainmap;
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
pub use error::ConvertError;
pub use gainmap::{apply_gain_map, apply_gain_map_view, extract_gain_map};
pub use gamut::GamutMatrix;
pub use measure::{LightLevelHistogram, LightLevelMethod, measure_cll, measure_histogram};
pub use negotiate::{FormatCost, LossKind, Negotiate, NegotiatedFormat};
pub use orient::{apply_orientation, apply_orientation_into};
pub use output::{EncodeReady, OutputPlan, OutputProfile};
pub use plan::{ImagePlan, RowPlan, convert_buf, convert_into};
pub use policy::{
    AlphaPolicy, ConvertPolicy, DepthPolicy, GamutPolicy, LumaCoefficients, RangePolicy,
};
pub use scan::scan_facts;
#[cfg(feature = "bt2446a")]
pub use tonemap::Bt2446A;
#[cfg(feature = "oklab")]
pub use tonemap::SoftCompress;
pub use tonemap::{PreparedMapper, Requirements, StripMapper, ToneMapContext, ToneMapper};
pub use ycbcr::{YcbcrMatrix, ycbcr_to_rgb_f32};
