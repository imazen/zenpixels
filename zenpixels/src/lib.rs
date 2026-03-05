//! Pixel format interchange types.
//!
//! Lightweight type definitions for describing pixel formats, color contexts,
//! and buffer layouts. No conversion logic — see `zenpixels-convert` for
//! transfer-function-aware conversion, gamut mapping, and codec negotiation.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod descriptor;
pub mod policy;

pub mod cicp;
pub mod color;
pub mod pixel_types;

pub mod buffer;

// Re-export key descriptor types at crate root for ergonomics.
pub use descriptor::{
    AlphaMode, ByteOrder, ChannelLayout, ChannelType, ColorModel, ColorPrimaries, PixelDescriptor,
    PixelFormat, SignalRange, TransferFunction,
};

// Re-export planar types when the `planar` feature is enabled.
#[cfg(feature = "planar")]
pub use descriptor::{
    MultiPlaneImage, Plane, PlaneDescriptor, PlaneLayout, PlaneMask, PlaneRelationship,
    PlaneSemantic, Subsampling, YuvMatrix,
};

// Re-export buffer types at crate root.
pub use buffer::{Bgrx, BufferError, Pixel, PixelBuffer, PixelSlice, PixelSliceMut, Rgbx};

// Re-export color types at crate root.
pub use cicp::Cicp;
pub use color::{ColorContext, ColorOrigin, ColorProfileSource, ColorProvenance, NamedProfile};

// Re-export GrayAlpha pixel types at crate root.
pub use pixel_types::{GrayAlpha8, GrayAlpha16, GrayAlphaF32};

pub use policy::{AlphaPolicy, ConvertOptions, DepthPolicy, GrayExpand, LumaCoefficients};
