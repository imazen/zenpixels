//! Pixel format interchange types for Rust image codecs.
//!
//! This crate provides the type system for describing pixel data: what the
//! bytes are ([`PixelFormat`]), what they mean ([`PixelDescriptor`]), and where
//! they live ([`PixelBuffer`], [`PixelSlice`], [`PixelSliceMut`]).
//!
//! No conversion logic lives here. For transfer-function-aware conversion,
//! gamut mapping, and codec format negotiation, see
//! [`zenpixels-convert`](https://docs.rs/zenpixels-convert).
//!
//! # Core types
//!
//! - [`PixelFormat`] — flat enum of byte layouts (`Rgb8`, `Rgba16`, `OklabF32`, etc.)
//! - [`PixelDescriptor`] — format + transfer function + primaries + alpha mode + signal range
//! - [`PixelBuffer`] — owned pixel storage with SIMD-aligned allocation
//! - [`PixelSlice`] / [`PixelSliceMut`] — borrowed views with stride support
//! - [`Pixel`] — trait mapping concrete types to their descriptor
//! - [`Cicp`] — ITU-T H.273 color signaling codes
//! - [`ColorContext`] — ICC profile bytes and/or CICP, `Arc`-shared
//! - [`ConvertOptions`] — policies for lossy operations (alpha removal, depth reduction)
//!
//! # Feature flags
//!
//! | Feature | What it enables |
//! |---------|----------------|
//! | `std` | Standard library (default; currently a no-op, everything is `no_std + alloc`) |
//! | `rgb` | [`Pixel`] impls for `rgb` crate types, typed `from_pixels()` constructors |
//! | `imgref` | `From<ImgRef>` / `From<ImgVec>` conversions (implies `rgb`) |
//! | `planar` | Multi-plane image types (YCbCr, Oklab, gain maps) |

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

whereat::define_at_crate_info!(path = "zenpixels/");

pub mod descriptor;
pub mod orientation;
#[cfg(feature = "planar")]
pub mod planar;
pub mod policy;

pub mod cicp;
pub mod color;
pub mod hdr;
#[cfg(feature = "icc")]
pub mod icc;
pub mod pixel_types;

pub mod buffer;

// Re-export orientation type at crate root.
pub use orientation::Orientation;

// Re-export key descriptor types at crate root for ergonomics.
pub use descriptor::{
    AlphaMode, ByteOrder, ChannelLayout, ChannelType, ColorModel, ColorPrimaries, PixelDescriptor,
    PixelFormat, SignalRange, TransferFunction,
};

// Re-export planar types when the `planar` feature is enabled.
#[cfg(feature = "planar")]
pub use planar::{
    MultiPlaneImage, Plane, PlaneDescriptor, PlaneLayout, PlaneMask, PlaneRelationship,
    PlaneSemantic, Subsampling, YuvMatrix,
};

// Re-export buffer types at crate root.
pub use buffer::{Bgrx, BufferError, Pixel, PixelBuffer, PixelSlice, PixelSliceMut, Rgbx};

// Re-export color types at crate root.
pub use cicp::Cicp;
pub use color::{
    ColorAuthority, ColorContext, ColorOrigin, ColorProfileSource, ColorProvenance, NamedProfile,
};

// Re-export HDR metadata types at crate root.
pub use hdr::{ContentLightLevel, MasteringDisplay};

// Re-export GrayAlpha pixel types at crate root.
pub use pixel_types::{GrayAlpha8, GrayAlpha16, GrayAlphaF32};

pub use policy::{AlphaPolicy, ConvertOptions, DepthPolicy, GrayExpand, LumaCoefficients};

// Re-export whereat types for error tracing.
pub use whereat::{At, ResultAtExt, at};
