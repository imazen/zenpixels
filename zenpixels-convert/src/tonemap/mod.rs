//! HDR → SDR tone mapping and OKLch soft gamut compression.
//!
//! Distinct namespace from [`zenpixels::hdr`] (which holds pure HDR metadata
//! carriers `DiffuseWhite`, `Nits`, `PeakSource`, `ContentLightLevel`,
//! `MasteringDisplay`, `HlgRendering`), eliminating the 0.4 module collision.

#[cfg(feature = "bt2446a")]
pub mod bt2446a;
#[cfg(feature = "oklab")]
pub mod gamut_compress;
pub mod tone_mapper;

/// ITU-R BT.2020 NCL luminance weights (used by BT.2446A and measurement).
pub const BT2020_LR: f32 = 0.2627;
/// ITU-R BT.2020 NCL green luminance weight.
pub const BT2020_LG: f32 = 0.6780;
/// ITU-R BT.2020 NCL blue luminance weight.
pub const BT2020_LB: f32 = 0.0593;

#[cfg(feature = "bt2446a")]
pub use bt2446a::{Bt2446A, Bt2446APipelineManaged};
#[cfg(feature = "oklab")]
pub use gamut_compress::SoftCompress;
pub use tone_mapper::{PreparedMapper, Requirements, StripMapper, ToneMapContext, ToneMapper};
