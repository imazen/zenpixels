//! Pixel storage, image views and color descriptions for Rust image code.
//!
//! Start with three types:
//!
//! - [`PixelBuffer`] owns an image's samples and metadata.
//! - [`PixelSlice`] borrows an image or a rectangular subimage, keeping its stride.
//! - [`PixelDescriptor`] describes the samples: format, transfer, primaries, alpha and range.
//!
//! [`PixelSliceMut`] provides writable access. Borrowing or cropping does not copy pixels:
//!
//! ```
//! use zenpixels::{PixelBuffer, PixelDescriptor};
//!
//! let image = PixelBuffer::try_new(8, 6, PixelDescriptor::RGB8_SRGB)?;
//! let crop = image.crop_view(2, 1, 3, 2);
//! assert_eq!(crop.stride(), image.stride()); // Parent stride, in bytes.
//! assert_eq!(crop.row(0).len(), 9);          // Only the crop's active samples.
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Read the [pixel types guide](https://imazen.github.io/zenpixels/guide.html)
//! for decoded bytes, cropped views, typed pixels and native codes. The
//! [interactive explorer](https://imazen.github.io/zenpixels/) shows their memory layout.
//!
//! # Layout and meaning
//!
//! [`PixelFormat`] has common named formats and [`PixelFormat::Generic`] for native
//! combinations. Construct a format with [`PixelFormat::from_parts`]. A descriptor
//! adds color meaning; changing a declaration does not convert existing samples.
//! For conversion, see [zenpixels-convert](https://docs.rs/zenpixels-convert).
//!
//! [`ColorContext`] carries the current ICC/CICP and luminance declarations;
//! [`ColorOrigin`] retains original file claims. [`plane::Plane`] carries strided
//! samples; [`plane::PlanarSlice`] combines validated plane roles and interpretation.
//! [`sample::SampleEncoding`] describes native integer payloads, while
//! [`sample::ExactDepth`] is evidence about the current canonical code grid.
//!
//! # Allocation policy
//!
//! Width/height count pixels; erased byte-view strides count bytes. `row(y)` excludes
//! padding. A view can retain its parent's stride and omit final-row padding.
//! Row starts require sample alignment; typed views also respect the Rust type's alignment.
//!
//! [`PixelBuffer::try_new`] makes packed, zero-filled typed storage. Use
//! [`PixelBuffer::try_new_aligned`] for a 64-byte-aligned owner with padded rows.
//! Fallible image allocation returns located [`BufferError`] errors. `new` and
//! other documented convenience APIs panic on allocation failure. Some control,
//! cache and dependency allocations remain infallible; see the
//! [allocation audit](https://github.com/imazen/zenpixels/blob/main/docs/allocation-audit.md).
//! There is no general operation-level memory budget in this crate.
//!
//! # Feature flags
//!
//! | Feature | Effect |
//! |---------|--------|
//! | `std` | Default; omit for `no_std + alloc` |
//! | `icc` | Default; paired-fingerprint ICC recognition with scoped use permission |
//! | `rgb` | Concrete `rgb` pixel types |
//! | `imgref` | imgref adapters; enables `rgb` |
//! | `image` | image 0.25 interop (Rust 1.88+) |
//! | `hash` | Pixel hashing and configurable inclusions (Rust 1.89+) |
//! | `allocator_api` | Allocator-parameterized storage on supported compilers |
//! | `planar`, `serde` | Retained compatibility spellings; no API |
//!
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![cfg_attr(test, allow(deprecated))]

extern crate alloc;

whereat::define_at_crate_info!(path = "zenpixels/");

pub mod descriptor;
#[cfg(feature = "hash")]
mod hashing;
#[cfg(feature = "hash")]
pub use hashing::HashInclusions;
#[cfg(feature = "image")]
mod image_interop;
pub mod orientation;
pub mod plane;
pub mod policy;
pub mod precision;
pub mod sample;

pub mod cicp;
pub mod color;
pub mod hdr;
#[cfg(feature = "icc")]
pub mod icc;
pub mod pixel_types;
pub(crate) mod registry;

pub mod buffer;

// Re-export orientation type at crate root.
pub use orientation::Orientation;

// Re-export key descriptor types at crate root for ergonomics.
#[doc(inline)]
pub use descriptor::{
    AlphaMode, ChannelLayout, ChannelType, ColorModel, ColorPrimaries, PixelDescriptor,
    PixelFormat, SignalRange, TransferFunction,
};

// Re-export buffer and alignment types at crate root.
#[doc(inline)]
pub use buffer::{
    Bgrx, BufferError, ByteStorage, ByteStorageMut, FromPartsError, InPlacePixels, Pixel,
    PixelBuffer, PixelBufferParts, PixelCow, PixelSlice, PixelSliceMut, PixelStorage, Rgbx,
    StorageSample,
};
#[doc(inline)]
pub use plane::RowAlign;

// Re-export color types at crate root.
pub use cicp::Cicp;
#[doc(inline)]
pub use color::{
    AnyColor, CieXy, Color, ColorAuthority, ColorContext, ColorOrigin, ColorProfile,
    ColorProfileSource, ColorProvenance, GammaCurve, NamedProfile, ParametricColor,
    ParametricPrimaries, ParametricTransfer, PixelEncoding, RenderingIntent, WhitePoint,
};

// Re-export GrayAlpha pixel types at crate root.
pub use pixel_types::{GrayAlpha8, GrayAlpha16, GrayAlphaF32};

pub use policy::{
    AlphaPolicy, ConvertOptions, DepthPolicy, GamutMapping, GrayExpand, LumaCoefficients,
};

// Every fallible signature here names it.
pub use whereat::At;
