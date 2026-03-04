// ---------------------------------------------------------------------------
// ImgRef -> PixelSlice (zero-copy From impls) -- imgref feature only
// ---------------------------------------------------------------------------

#[cfg(feature = "imgref")]
use core::marker::PhantomData;

#[cfg(feature = "imgref")]
use imgref::ImgRef;

#[cfg(feature = "rgb")]
use rgb::alt::BGRA;
#[cfg(feature = "rgb")]
use rgb::{Gray, Rgb, Rgba};

use crate::descriptor::PixelDescriptor;

use super::{Pixel, PixelBuffer, PixelSlice, PixelSliceMut};

#[cfg(feature = "imgref")]
macro_rules! impl_from_imgref {
    ($pixel:ty, $descriptor:expr) => {
        impl<'a> From<ImgRef<'a, $pixel>> for PixelSlice<'a, $pixel> {
            fn from(img: ImgRef<'a, $pixel>) -> Self {
                let bytes: &[u8] = bytemuck::cast_slice(img.buf());
                let byte_stride = img.stride() * core::mem::size_of::<$pixel>();
                PixelSlice {
                    data: bytes,
                    width: img.width() as u32,
                    rows: img.height() as u32,
                    stride: byte_stride,
                    descriptor: $descriptor,

                    color: None,
                    _pixel: PhantomData,
                }
            }
        }
    };
}

// u8 types are conventionally sRGB, f32 types are conventionally linear.
// u16 types have no standard convention so use transfer-agnostic descriptors.
#[cfg(feature = "imgref")]
impl_from_imgref!(Rgb<u8>, PixelDescriptor::RGB8_SRGB);
#[cfg(feature = "imgref")]
impl_from_imgref!(Rgba<u8>, PixelDescriptor::RGBA8_SRGB);
#[cfg(feature = "imgref")]
impl_from_imgref!(Rgb<u16>, PixelDescriptor::RGB16);
#[cfg(feature = "imgref")]
impl_from_imgref!(Rgba<u16>, PixelDescriptor::RGBA16);
#[cfg(feature = "imgref")]
impl_from_imgref!(Rgb<f32>, PixelDescriptor::RGBF32_LINEAR);
#[cfg(feature = "imgref")]
impl_from_imgref!(Rgba<f32>, PixelDescriptor::RGBAF32_LINEAR);
#[cfg(feature = "imgref")]
impl_from_imgref!(Gray<u8>, PixelDescriptor::GRAY8_SRGB);
#[cfg(feature = "imgref")]
impl_from_imgref!(Gray<u16>, PixelDescriptor::GRAY16);
#[cfg(feature = "imgref")]
impl_from_imgref!(Gray<f32>, PixelDescriptor::GRAYF32_LINEAR);
#[cfg(feature = "imgref")]
impl_from_imgref!(BGRA<u8>, PixelDescriptor::BGRA8_SRGB);

// ---------------------------------------------------------------------------
// ImgRefMut -> PixelSliceMut (zero-copy From impls) -- imgref feature only
// ---------------------------------------------------------------------------

#[cfg(feature = "imgref")]
macro_rules! impl_from_imgref_mut {
    ($pixel:ty, $descriptor:expr) => {
        impl<'a> From<imgref::ImgRefMut<'a, $pixel>> for PixelSliceMut<'a, $pixel> {
            fn from(img: imgref::ImgRefMut<'a, $pixel>) -> Self {
                let width = img.width() as u32;
                let rows = img.height() as u32;
                let byte_stride = img.stride() * core::mem::size_of::<$pixel>();
                let buf = img.into_buf();
                let bytes: &mut [u8] = bytemuck::cast_slice_mut(buf);
                PixelSliceMut {
                    data: bytes,
                    width,
                    rows,
                    stride: byte_stride,
                    descriptor: $descriptor,

                    color: None,
                    _pixel: PhantomData,
                }
            }
        }
    };
}

#[cfg(feature = "imgref")]
impl_from_imgref_mut!(Rgb<u8>, PixelDescriptor::RGB8_SRGB);
#[cfg(feature = "imgref")]
impl_from_imgref_mut!(Rgba<u8>, PixelDescriptor::RGBA8_SRGB);
#[cfg(feature = "imgref")]
impl_from_imgref_mut!(Rgb<u16>, PixelDescriptor::RGB16);
#[cfg(feature = "imgref")]
impl_from_imgref_mut!(Rgba<u16>, PixelDescriptor::RGBA16);
#[cfg(feature = "imgref")]
impl_from_imgref_mut!(Rgb<f32>, PixelDescriptor::RGBF32_LINEAR);
#[cfg(feature = "imgref")]
impl_from_imgref_mut!(Rgba<f32>, PixelDescriptor::RGBAF32_LINEAR);
#[cfg(feature = "imgref")]
impl_from_imgref_mut!(Gray<u8>, PixelDescriptor::GRAY8_SRGB);
#[cfg(feature = "imgref")]
impl_from_imgref_mut!(Gray<u16>, PixelDescriptor::GRAY16);
#[cfg(feature = "imgref")]
impl_from_imgref_mut!(Gray<f32>, PixelDescriptor::GRAYF32_LINEAR);
#[cfg(feature = "imgref")]
impl_from_imgref_mut!(BGRA<u8>, PixelDescriptor::BGRA8_SRGB);

// ---------------------------------------------------------------------------
// Typed -> Erased blanket From impls (erase via From)
// ---------------------------------------------------------------------------

impl<'a, P: Pixel> From<PixelSlice<'a, P>> for PixelSlice<'a> {
    fn from(typed: PixelSlice<'a, P>) -> Self {
        typed.erase()
    }
}

impl<'a, P: Pixel> From<PixelSliceMut<'a, P>> for PixelSliceMut<'a> {
    fn from(typed: PixelSliceMut<'a, P>) -> Self {
        typed.erase()
    }
}

impl<P: Pixel> From<PixelBuffer<P>> for PixelBuffer {
    fn from(typed: PixelBuffer<P>) -> Self {
        typed.erase()
    }
}
