#[cfg(feature = "planar")]
use super::SampleEncoding;
#[cfg(feature = "planar")]
use crate::plane::{AnyPlane, AnyPlaneMut};
#[cfg(feature = "planar")]
use core::fmt;

/// Chroma resolution relative to the complete luma plane.
///
/// This is the one definition for the zen crates: `zencodec::frame`
/// re-exports it instead of defining its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ChromaSubsampling {
    /// One luma plane and no chroma planes (AV1 `I400`, HEVC 4:0:0).
    Mono,
    /// Equal horizontal and vertical resolution.
    Yuv444,
    /// Half horizontal resolution, rounded up at an odd edge.
    Yuv422,
    /// Half horizontal and vertical resolution, rounded up at odd edges.
    Yuv420,
    /// Equal horizontal resolution and half vertical resolution, rounded up at an odd edge (4:4:0).
    Yuv440,
}

impl ChromaSubsampling {
    /// Chroma plane size for a luma size, rounded up at odd edges; `None`
    /// for [`Mono`](Self::Mono).
    pub const fn chroma_size(self, width: u32, height: u32) -> Option<(u32, u32)> {
        match self {
            Self::Mono => None,
            Self::Yuv444 => Some((width, height)),
            Self::Yuv422 => Some((width.div_ceil(2), height)),
            Self::Yuv420 => Some((width.div_ceil(2), height.div_ceil(2))),
            Self::Yuv440 => Some((width, height.div_ceil(2))),
        }
    }
}

/// Declared chroma sample location. This is a fact, not a fallback policy:
/// `Unknown` is kept, never defaulted to `Center`. A conversion that needs
/// siting refuses `Unknown` on a subsampled axis or takes it from the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum ChromaSiting {
    /// The source did not establish a location.
    #[default]
    Unknown,
    /// Centered horizontally and vertically in the subsampled luma cell
    /// (JPEG style).
    Center,
    /// Horizontally co-sited with the left luma sample, vertically centered
    /// between rows (MPEG-2 style 4:2:0).
    Left,
    /// Co-sited with the top-left luma sample.
    TopLeft,
}

#[cfg(feature = "planar")]
#[derive(Clone, Copy, Debug)]
enum Samples<'a> {
    Bytes1(&'a [u8]),
    Words2(&'a [u16]),
    Words4(&'a [u32]),
}

#[cfg(feature = "planar")]
enum SamplesMut<'a> {
    Bytes1(&'a mut [u8]),
    Words2(&'a mut [u16]),
    Words4(&'a mut [u32]),
}

/// Integer codes of one component in a borrowed plane, without normalization
/// or color claims.
///
/// Code depth and placement come from [`crate::sample::SampleEncoding`];
/// padding bits are ignored on reads. The enclosing format supplies roles,
/// range, matrix, transfer and sample location. Nothing is copied or allocated.
///
/// The plane is a [`crate::plane::Plane`] of `u8`/`i8`, `u16`/`i16`, or
/// `u32`/`i32` samples. Unsigned encodings read via [`code`](Self::code),
/// [`code_u32`](Self::code_u32), [`read_codes`](Self::read_codes), and
/// [`read_codes_u32`](Self::read_codes_u32); signed two's-complement encodings
/// read via [`code_i32`](Self::code_i32) and
/// [`read_codes_i32`](Self::read_codes_i32). A
/// plane may hold several interleaved components per position, such as the
/// Cb/Cr plane of NV12 or P010, or interleaved RGB(A);
/// [`component`](Self::component) reads one of them:
///
/// ```
/// use zenpixels::{ChannelType, plane::{AnyPlane, PlaneRef}, sample::{CodePlane, SampleEncoding}};
/// // P010 chroma: one interleaved Cb/Cr plane, 10 high bits per sample.
/// let p010 = SampleEncoding::new(ChannelType::U16, 10, 6).unwrap();
/// let uv: [u16; 4] = [512 << 6, 100 << 6, 300 << 6, 1023 << 6]; // Cb Cr Cb Cr
/// let plane = AnyPlane::U16(PlaneRef::new(&uv[..], 4, 1, 4).unwrap());
/// let cb = CodePlane::component(plane, 2, 0, p010).unwrap();
/// let cr = CodePlane::component(plane, 2, 1, p010).unwrap();
/// assert_eq!((cb.code(1, 0), cr.code(1, 0)), (Some(300), Some(1023)));
/// ```
#[cfg(feature = "planar")]
#[derive(Clone, Copy, Debug)]
pub struct CodePlane<'a> {
    // Starts at this component's first sample, cast to bytes with alignment
    // guaranteed by the typed source slice; sample type comes from `encoding.storage()`.
    sample_bytes: &'a [u8],
    width: usize,
    height: usize,
    stride_samples: usize,
    step_samples: u32,
    encoding: SampleEncoding,
}

#[cfg(all(feature = "planar", target_pointer_width = "64"))]
const _: () = assert!(core::mem::size_of::<CodePlane<'static>>() == 48);

/// Codes cannot be read from or written to a plane as requested.
#[cfg(feature = "planar")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CodePlaneError {
    /// The plane's sample type differs from the sample encoding's storage,
    /// is not a supported 8/16/32-bit integer container (`U8`/`I8`, `U16`/`I16`,
    /// `U32`/`I32`), or an unsigned/signed call was made on the opposite
    /// signedness.
    StorageMismatch,
    /// The component index is not below the component count, or the plane's
    /// width does not hold whole positions.
    Components,
    /// A requested read or write extends beyond the active plane.
    OutsideActivePlane,
    /// A `u16` slice/scalar was passed for an encoding with `code_bits() > 16`;
    /// use [`CodePlane::code_u32`] / [`CodePlane::read_codes_u32`] or
    /// [`CodePlaneMut::set_code_u32`] / [`CodePlaneMut::write_codes_u32`].
    NarrowDestination,
    /// A code to be written does not fit within `encoding.code_bits()` (unsigned
    /// `0..=((1 << code_bits) - 1)` or signed two's-complement
    /// `-(1 << (code_bits - 1))..=((1 << (code_bits - 1)) - 1)`).
    CodeOutOfRange,
}

#[cfg(feature = "planar")]
impl fmt::Display for CodePlaneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::StorageMismatch => "code plane storage disagrees with sample encoding",
            Self::Components => "component index or plane width disagrees with the component count",
            Self::OutsideActivePlane => "code plane access extends beyond the active plane",
            Self::NarrowDestination => "code plane bit depth exceeds u16 width",
            Self::CodeOutOfRange => "code value exceeds sample encoding bit depth",
        })
    }
}
#[cfg(feature = "planar")]
impl core::error::Error for CodePlaneError {}

#[cfg(feature = "planar")]
impl<'a> CodePlane<'a> {
    /// Codes of a plane holding one component per position.
    ///
    /// # Errors
    ///
    /// [`CodePlaneError::StorageMismatch`] when the plane's samples differ from
    /// `encoding`'s storage or are not `U8`/`I8`, `U16`/`I16`, or `U32`/`I32`.
    pub fn new(plane: AnyPlane<'a>, encoding: SampleEncoding) -> Result<Self, CodePlaneError> {
        Self::component(plane, 1, 0, encoding)
    }

    /// Codes of component `index` in a plane holding `components`
    /// interleaved components per position. The plane's width must be a
    /// multiple of `components`; this view is `width / components` wide.
    ///
    /// # Errors
    ///
    /// [`CodePlaneError::Components`] for an index not below `components` or
    /// a width that is not a multiple of it, and
    /// [`CodePlaneError::StorageMismatch`] when the plane's samples differ from
    /// `encoding`'s storage or are not `U8`/`I8`, `U16`/`I16`, or `U32`/`I32`.
    pub fn component(
        plane: AnyPlane<'a>,
        components: usize,
        index: usize,
        encoding: SampleEncoding,
    ) -> Result<Self, CodePlaneError> {
        if index >= components || plane.width() % components != 0 {
            return Err(CodePlaneError::Components);
        }
        let Ok(step_samples) = u32::try_from(components) else {
            return Err(CodePlaneError::Components);
        };
        if plane.channel_type() != encoding.storage() {
            return Err(CodePlaneError::StorageMismatch);
        }
        let (width, height) = (plane.width() / components, plane.height());
        // A nonempty plane's span reaches past `index`; an empty one is never read.
        let skip = if width == 0 || height == 0 { 0 } else { index };
        let (sample_bytes, stride_samples) = match plane {
            AnyPlane::U8(p) => (&p.into_strided_samples()[skip..], p.stride()),
            AnyPlane::I8(p) => (
                bytemuck::cast_slice(&p.into_strided_samples()[skip..]),
                p.stride(),
            ),
            AnyPlane::U16(p) => (
                bytemuck::cast_slice(&p.into_strided_samples()[skip..]),
                p.stride(),
            ),
            AnyPlane::I16(p) => (
                bytemuck::cast_slice(&p.into_strided_samples()[skip..]),
                p.stride(),
            ),
            AnyPlane::U32(p) => (
                bytemuck::cast_slice(&p.into_strided_samples()[skip..]),
                p.stride(),
            ),
            AnyPlane::I32(p) => (
                bytemuck::cast_slice(&p.into_strided_samples()[skip..]),
                p.stride(),
            ),
            _ => return Err(CodePlaneError::StorageMismatch),
        };
        Ok(Self {
            sample_bytes,
            width,
            height,
            stride_samples,
            step_samples,
            encoding,
        })
    }

    #[inline]
    fn samples(self) -> Samples<'a> {
        match self.encoding.storage() {
            crate::ChannelType::U8 | crate::ChannelType::I8 => Samples::Bytes1(self.sample_bytes),
            crate::ChannelType::U16 | crate::ChannelType::I16 => {
                Samples::Words2(bytemuck::cast_slice(self.sample_bytes))
            }
            crate::ChannelType::U32 | crate::ChannelType::I32 => {
                Samples::Words4(bytemuck::cast_slice(self.sample_bytes))
            }
            _ => unreachable!("CodePlane only accepts 8/16/32-bit integer storage"),
        }
    }

    /// Active samples per row.
    pub const fn width(self) -> usize {
        self.width
    }
    /// Number of active rows.
    pub const fn height(self) -> usize {
        self.height
    }
    /// Storage width, code depth and bit placement; not a normalized pixel format.
    pub const fn encoding(self) -> SampleEncoding {
        self.encoding
    }

    #[inline]
    fn raw_code_u32(self, x: usize, y: usize) -> Option<u32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let step = self.step_samples as usize;
        let i = y * self.stride_samples + x * step;
        let sample = match self.samples() {
            Samples::Bytes1(s) => u32::from(s[i]),
            Samples::Words2(s) => u32::from(s[i]),
            Samples::Words4(s) => s[i],
        };
        Some((sample >> self.encoding.bit_shift()) & self.code_mask_u32())
    }

    /// Extract one unsigned `1..=16`-bit code, ignoring padding bits; returns
    /// `None` outside the active plane, when the encoding is signed (use
    /// [`code_i32`](Self::code_i32)), or when `self.encoding().code_bits() > 16`
    /// (use [`code_u32`](Self::code_u32) for `17..=32`-bit codes).
    pub fn code(self, x: usize, y: usize) -> Option<u16> {
        if self.encoding.is_signed() || self.encoding.code_bits() > 16 {
            return None;
        }
        self.raw_code_u32(x, y).map(|c| c as u16)
    }

    /// Extract one unsigned `1..=32`-bit code as `u32`, ignoring padding bits;
    /// returns `None` outside the active plane or when the encoding is signed
    /// (use [`code_i32`](Self::code_i32)).
    pub fn code_u32(self, x: usize, y: usize) -> Option<u32> {
        if self.encoding.is_signed() {
            return None;
        }
        self.raw_code_u32(x, y)
    }

    /// Extract one signed two's-complement `1..=32`-bit code sign-extended to
    /// `i32`, ignoring padding bits; returns `None` outside the active plane
    /// or when the encoding is unsigned (use [`code`](Self::code) /
    /// [`code_u32`](Self::code_u32)).
    pub fn code_i32(self, x: usize, y: usize) -> Option<i32> {
        if !self.encoding.is_signed() {
            return None;
        }
        let sign_shift = 32 - u32::from(self.encoding.code_bits());
        self.raw_code_u32(x, y)
            .map(|raw| ((raw << sign_shift) as i32) >> sign_shift)
    }

    /// Extract `codes.len()` consecutive unsigned `1..=16`-bit codes of row `y`,
    /// starting at column `x`.
    ///
    /// Padding bits are ignored, as in [`code`](Self::code). Returns
    /// [`CodePlaneError::StorageMismatch`] when the encoding is signed, and
    /// [`CodePlaneError::NarrowDestination`] when `self.encoding().code_bits() > 16`
    /// (use [`read_codes_u32`](Self::read_codes_u32) instead). The extent is
    /// checked before writing; on error `codes` is unchanged.
    pub fn read_codes(self, x: usize, y: usize, codes: &mut [u16]) -> Result<(), CodePlaneError> {
        if self.encoding.is_signed() {
            return Err(CodePlaneError::StorageMismatch);
        }
        if self.encoding.code_bits() > 16 {
            return Err(CodePlaneError::NarrowDestination);
        }
        if y >= self.height
            || x.checked_add(codes.len())
                .is_none_or(|end| end > self.width)
        {
            return Err(CodePlaneError::OutsideActivePlane);
        }
        let Some(last) = codes.len().checked_sub(1) else {
            return Ok(());
        };
        let step = self.step_samples as usize;
        // Construction proved the last active sample is addressable, so these
        // products of smaller coordinates cannot overflow.
        let start = y * self.stride_samples + x * step;
        let end = start + last * step + 1;
        let (shift, mask) = (self.encoding.bit_shift(), self.code_mask());
        match self.samples() {
            Samples::Bytes1(s) => {
                let samples = &s[start..end];
                if step == 1 {
                    for (code, sample) in codes.iter_mut().zip(samples) {
                        *code = (u16::from(*sample) >> shift) & mask;
                    }
                } else {
                    for (code, sample) in codes.iter_mut().zip(samples.iter().step_by(step)) {
                        *code = (u16::from(*sample) >> shift) & mask;
                    }
                }
            }
            Samples::Words2(s) => {
                let samples = &s[start..end];
                if step == 1 {
                    for (code, sample) in codes.iter_mut().zip(samples) {
                        *code = (*sample >> shift) & mask;
                    }
                } else {
                    for (code, sample) in codes.iter_mut().zip(samples.iter().step_by(step)) {
                        *code = (*sample >> shift) & mask;
                    }
                }
            }
            Samples::Words4(s) => {
                let mask32 = u32::from(mask);
                let samples = &s[start..end];
                if step == 1 {
                    for (code, sample) in codes.iter_mut().zip(samples) {
                        *code = ((*sample >> shift) & mask32) as u16;
                    }
                } else {
                    for (code, sample) in codes.iter_mut().zip(samples.iter().step_by(step)) {
                        *code = ((*sample >> shift) & mask32) as u16;
                    }
                }
            }
        }
        Ok(())
    }

    /// Extract `codes.len()` consecutive unsigned `1..=32`-bit codes of row `y`
    /// into `u32`, starting at column `x`.
    ///
    /// Padding bits are ignored, as in [`code_u32`](Self::code_u32). Works for
    /// `U8`, `U16`, and `U32` planes across all `1..=32` code depths, and returns
    /// [`CodePlaneError::StorageMismatch`] on signed planes. The extent is
    /// checked before writing; on error `codes` is unchanged.
    pub fn read_codes_u32(
        self,
        x: usize,
        y: usize,
        codes: &mut [u32],
    ) -> Result<(), CodePlaneError> {
        if self.encoding.is_signed() {
            return Err(CodePlaneError::StorageMismatch);
        }
        self.read_raw_u32(x, y, codes)
    }

    /// Extract `codes.len()` consecutive signed two's-complement `1..=32`-bit
    /// codes of row `y` sign-extended into `i32`, starting at column `x`.
    ///
    /// Padding bits are ignored, as in [`code_i32`](Self::code_i32). Works for
    /// `I8`, `I16`, and `I32` planes across all `1..=32` code depths, and returns
    /// [`CodePlaneError::StorageMismatch`] on unsigned planes. The extent is
    /// checked before writing; on error `codes` is unchanged.
    pub fn read_codes_i32(
        self,
        x: usize,
        y: usize,
        codes: &mut [i32],
    ) -> Result<(), CodePlaneError> {
        if !self.encoding.is_signed() {
            return Err(CodePlaneError::StorageMismatch);
        }
        let raw: &mut [u32] = bytemuck::cast_slice_mut(codes);
        self.read_raw_u32(x, y, raw)?;
        let sign_shift = 32 - u32::from(self.encoding.code_bits());
        if sign_shift != 0 {
            for code in codes.iter_mut() {
                *code = ((*code as u32) << sign_shift) as i32 >> sign_shift;
            }
        }
        Ok(())
    }

    fn read_raw_u32(self, x: usize, y: usize, codes: &mut [u32]) -> Result<(), CodePlaneError> {
        if y >= self.height
            || x.checked_add(codes.len())
                .is_none_or(|end| end > self.width)
        {
            return Err(CodePlaneError::OutsideActivePlane);
        }
        let Some(last) = codes.len().checked_sub(1) else {
            return Ok(());
        };
        let step = self.step_samples as usize;
        let start = y * self.stride_samples + x * step;
        let end = start + last * step + 1;
        let (shift, mask) = (self.encoding.bit_shift(), self.code_mask_u32());
        match self.samples() {
            Samples::Bytes1(s) => {
                let samples = &s[start..end];
                if step == 1 {
                    for (code, sample) in codes.iter_mut().zip(samples) {
                        *code = (u32::from(*sample) >> shift) & mask;
                    }
                } else {
                    for (code, sample) in codes.iter_mut().zip(samples.iter().step_by(step)) {
                        *code = (u32::from(*sample) >> shift) & mask;
                    }
                }
            }
            Samples::Words2(s) => {
                let samples = &s[start..end];
                if step == 1 {
                    for (code, sample) in codes.iter_mut().zip(samples) {
                        *code = (u32::from(*sample) >> shift) & mask;
                    }
                } else {
                    for (code, sample) in codes.iter_mut().zip(samples.iter().step_by(step)) {
                        *code = (u32::from(*sample) >> shift) & mask;
                    }
                }
            }
            Samples::Words4(s) => {
                let samples = &s[start..end];
                if step == 1 {
                    for (code, sample) in codes.iter_mut().zip(samples) {
                        *code = (*sample >> shift) & mask;
                    }
                } else {
                    for (code, sample) in codes.iter_mut().zip(samples.iter().step_by(step)) {
                        *code = (*sample >> shift) & mask;
                    }
                }
            }
        }
        Ok(())
    }

    const fn code_mask(self) -> u16 {
        ((1u32 << self.encoding.code_bits()) - 1) as u16
    }

    const fn code_mask_u32(self) -> u32 {
        u32::MAX >> (32 - self.encoding.code_bits() as u32)
    }
}

/// Writable integer codes of one component in a borrowed mutable plane, without
/// normalization or color claims.
///
/// By default, writing a code clears non-payload bits of the target sample word
/// to zero (`(raw & mask) << bit_shift`), satisfying [`SampleEncoding`]'s
/// zero-padding output rule even over unzeroed buffers. Chain
/// [`preserving_other_bits`](Self::preserving_other_bits) when packing multiple
/// bitfields into the same sample word (e.g. `R10G10B10A2` in `U32`).
///
/// Coordinates and code ranges are validated before mutating any sample; on
/// error, the backing plane is unchanged.
///
/// ```
/// use zenpixels::{ChannelType, plane::{AnyPlaneMut, PlaneMut}, sample::{CodePlaneMut, SampleEncoding}};
/// let p010 = SampleEncoding::new(ChannelType::U16, 10, 6).unwrap();
/// let mut uv = [0xFFFFu16; 4];
/// let plane = AnyPlaneMut::U16(PlaneMut::new(&mut uv[..], 4, 1, 4).unwrap());
/// let mut cb = CodePlaneMut::component(plane, 2, 0, p010).unwrap();
/// cb.write_codes(0, 0, &[512, 300]).unwrap();
/// assert_eq!(cb.code(0, 0), Some(512));
/// assert_eq!(cb.code(1, 0), Some(300));
/// // Cb slots have zero low padding; untouched Cr slots stay intact.
/// assert_eq!(uv, [512 << 6, 0xFFFF, 300 << 6, 0xFFFF]);
/// ```
#[cfg(feature = "planar")]
#[derive(Debug)]
pub struct CodePlaneMut<'a> {
    sample_bytes: &'a mut [u8],
    width: usize,
    height: usize,
    stride_samples: usize,
    step_samples: u32,
    encoding: SampleEncoding,
    preserve_bits: bool,
}

#[cfg(all(feature = "planar", target_pointer_width = "64"))]
const _: () = assert!(core::mem::size_of::<CodePlaneMut<'static>>() == 48);

#[cfg(feature = "planar")]
impl<'a> CodePlaneMut<'a> {
    /// Writable codes of a plane holding one component per position.
    ///
    /// # Errors
    ///
    /// [`CodePlaneError::StorageMismatch`] when the plane's samples differ from
    /// `encoding`'s storage or are not `U8`/`I8`, `U16`/`I16`, or `U32`/`I32`.
    pub fn new(plane: AnyPlaneMut<'a>, encoding: SampleEncoding) -> Result<Self, CodePlaneError> {
        Self::component(plane, 1, 0, encoding)
    }

    /// Writable codes of component `index` in a plane holding `components`
    /// interleaved components per position. The plane's width must be a
    /// multiple of `components`; this view is `width / components` wide.
    ///
    /// # Errors
    ///
    /// [`CodePlaneError::Components`] for an index not below `components` or
    /// a width that is not a multiple of it, and
    /// [`CodePlaneError::StorageMismatch`] when the plane's samples differ from
    /// `encoding`'s storage or are not `U8`/`I8`, `U16`/`I16`, or `U32`/`I32`.
    pub fn component(
        plane: AnyPlaneMut<'a>,
        components: usize,
        index: usize,
        encoding: SampleEncoding,
    ) -> Result<Self, CodePlaneError> {
        if index >= components || plane.width() % components != 0 {
            return Err(CodePlaneError::Components);
        }
        let Ok(step_samples) = u32::try_from(components) else {
            return Err(CodePlaneError::Components);
        };
        if plane.channel_type() != encoding.storage() {
            return Err(CodePlaneError::StorageMismatch);
        }
        let (width, height) = (plane.width() / components, plane.height());
        let skip = if width == 0 || height == 0 { 0 } else { index };
        let (sample_bytes, stride_samples): (&'a mut [u8], usize) = match plane {
            AnyPlaneMut::U8(p) => {
                let stride = p.stride();
                (&mut p.into_strided_samples_mut()[skip..], stride)
            }
            AnyPlaneMut::I8(p) => {
                let stride = p.stride();
                (
                    bytemuck::cast_slice_mut(&mut p.into_strided_samples_mut()[skip..]),
                    stride,
                )
            }
            AnyPlaneMut::U16(p) => {
                let stride = p.stride();
                (
                    bytemuck::cast_slice_mut(&mut p.into_strided_samples_mut()[skip..]),
                    stride,
                )
            }
            AnyPlaneMut::I16(p) => {
                let stride = p.stride();
                (
                    bytemuck::cast_slice_mut(&mut p.into_strided_samples_mut()[skip..]),
                    stride,
                )
            }
            AnyPlaneMut::U32(p) => {
                let stride = p.stride();
                (
                    bytemuck::cast_slice_mut(&mut p.into_strided_samples_mut()[skip..]),
                    stride,
                )
            }
            AnyPlaneMut::I32(p) => {
                let stride = p.stride();
                (
                    bytemuck::cast_slice_mut(&mut p.into_strided_samples_mut()[skip..]),
                    stride,
                )
            }
            _ => return Err(CodePlaneError::StorageMismatch),
        };
        Ok(Self {
            sample_bytes,
            width,
            height,
            stride_samples,
            step_samples,
            encoding,
            preserve_bits: false,
        })
    }

    /// Preserve bits outside `(mask << bit_shift)` in each written sample word
    /// instead of zeroing padding bits. Useful when packing multiple bitfields
    /// into a shared integer word.
    #[must_use]
    pub const fn preserving_other_bits(mut self) -> Self {
        self.preserve_bits = true;
        self
    }

    /// Active samples per row.
    pub const fn width(&self) -> usize {
        self.width
    }
    /// Number of active rows.
    pub const fn height(&self) -> usize {
        self.height
    }
    /// Storage width, code depth and bit placement.
    pub const fn encoding(&self) -> SampleEncoding {
        self.encoding
    }

    /// Borrow a read-only [`CodePlane`] view without copying or allocating.
    pub fn as_ref(&self) -> CodePlane<'_> {
        CodePlane {
            sample_bytes: self.sample_bytes,
            width: self.width,
            height: self.height,
            stride_samples: self.stride_samples,
            step_samples: self.step_samples,
            encoding: self.encoding,
        }
    }

    /// Extract one unsigned `1..=16`-bit code; see [`CodePlane::code`].
    pub fn code(&self, x: usize, y: usize) -> Option<u16> {
        self.as_ref().code(x, y)
    }

    /// Extract one unsigned `1..=32`-bit code as `u32`; see [`CodePlane::code_u32`].
    pub fn code_u32(&self, x: usize, y: usize) -> Option<u32> {
        self.as_ref().code_u32(x, y)
    }

    /// Extract one signed `1..=32`-bit code sign-extended to `i32`; see [`CodePlane::code_i32`].
    pub fn code_i32(&self, x: usize, y: usize) -> Option<i32> {
        self.as_ref().code_i32(x, y)
    }

    /// Write one unsigned `1..=16`-bit code at `(x, y)`.
    pub fn set_code(&mut self, x: usize, y: usize, code: u16) -> Result<(), CodePlaneError> {
        self.write_codes(x, y, core::slice::from_ref(&code))
    }

    /// Write one unsigned `1..=32`-bit code at `(x, y)`.
    pub fn set_code_u32(&mut self, x: usize, y: usize, code: u32) -> Result<(), CodePlaneError> {
        self.write_codes_u32(x, y, core::slice::from_ref(&code))
    }

    /// Write one signed two's-complement `1..=32`-bit code at `(x, y)`.
    pub fn set_code_i32(&mut self, x: usize, y: usize, code: i32) -> Result<(), CodePlaneError> {
        self.write_codes_i32(x, y, core::slice::from_ref(&code))
    }

    /// Write `codes.len()` consecutive unsigned `1..=16`-bit codes on row `y`,
    /// starting at column `x`.
    ///
    /// Returns [`CodePlaneError::StorageMismatch`] on signed planes,
    /// [`CodePlaneError::NarrowDestination`] when `code_bits() > 16` (use
    /// [`write_codes_u32`](Self::write_codes_u32)), [`CodePlaneError::OutsideActivePlane`]
    /// if the span exceeds the active row, and [`CodePlaneError::CodeOutOfRange`]
    /// if any code exceeds `(1 << code_bits) - 1`. Checked before mutating.
    pub fn write_codes(&mut self, x: usize, y: usize, codes: &[u16]) -> Result<(), CodePlaneError> {
        if self.encoding.is_signed() {
            return Err(CodePlaneError::StorageMismatch);
        }
        if self.encoding.code_bits() > 16 {
            return Err(CodePlaneError::NarrowDestination);
        }
        let Some((start, end)) = self.checked_span(x, y, codes.len())? else {
            return Ok(());
        };
        let max = self.code_mask_u32() as u16;
        if codes.iter().any(|&c| c > max) {
            return Err(CodePlaneError::CodeOutOfRange);
        }
        let step = self.step_samples as usize;
        let shift = self.encoding.bit_shift();
        let mask32 = self.code_mask_u32();
        let preserve = self.preserve_bits;
        match self.samples_mut() {
            SamplesMut::Bytes1(s) => {
                let clear = !((mask32 as u8) << shift);
                for (&c, dst) in codes.iter().zip(s[start..end].iter_mut().step_by(step)) {
                    let placed = (c as u8) << shift;
                    *dst = if preserve {
                        (*dst & clear) | placed
                    } else {
                        placed
                    };
                }
            }
            SamplesMut::Words2(s) => {
                let clear = !((mask32 as u16) << shift);
                for (&c, dst) in codes.iter().zip(s[start..end].iter_mut().step_by(step)) {
                    let placed = c << shift;
                    *dst = if preserve {
                        (*dst & clear) | placed
                    } else {
                        placed
                    };
                }
            }
            SamplesMut::Words4(s) => {
                let clear = !(mask32 << shift);
                for (&c, dst) in codes.iter().zip(s[start..end].iter_mut().step_by(step)) {
                    let placed = u32::from(c) << shift;
                    *dst = if preserve {
                        (*dst & clear) | placed
                    } else {
                        placed
                    };
                }
            }
        }
        Ok(())
    }

    /// Write `codes.len()` consecutive unsigned `1..=32`-bit codes on row `y`,
    /// starting at column `x`.
    ///
    /// Returns [`CodePlaneError::StorageMismatch`] on signed planes,
    /// [`CodePlaneError::OutsideActivePlane`] if the span exceeds the active row,
    /// and [`CodePlaneError::CodeOutOfRange`] if any code exceeds
    /// `(1 << code_bits) - 1`. Checked before mutating.
    pub fn write_codes_u32(
        &mut self,
        x: usize,
        y: usize,
        codes: &[u32],
    ) -> Result<(), CodePlaneError> {
        if self.encoding.is_signed() {
            return Err(CodePlaneError::StorageMismatch);
        }
        let Some((start, end)) = self.checked_span(x, y, codes.len())? else {
            return Ok(());
        };
        let max = self.code_mask_u32();
        if codes.iter().any(|&c| c > max) {
            return Err(CodePlaneError::CodeOutOfRange);
        }
        self.write_raw_unchecked(start, end, codes.iter().copied());
        Ok(())
    }

    /// Write `codes.len()` consecutive signed two's-complement `1..=32`-bit
    /// codes on row `y`, starting at column `x`.
    ///
    /// Returns [`CodePlaneError::StorageMismatch`] on unsigned planes,
    /// [`CodePlaneError::OutsideActivePlane`] if the span exceeds the active row,
    /// and [`CodePlaneError::CodeOutOfRange`] if any code lies outside
    /// `-(1 << (code_bits - 1))..=((1 << (code_bits - 1)) - 1)`. Checked before
    /// mutating.
    pub fn write_codes_i32(
        &mut self,
        x: usize,
        y: usize,
        codes: &[i32],
    ) -> Result<(), CodePlaneError> {
        if !self.encoding.is_signed() {
            return Err(CodePlaneError::StorageMismatch);
        }
        let Some((start, end)) = self.checked_span(x, y, codes.len())? else {
            return Ok(());
        };
        let bits = self.encoding.code_bits();
        if bits < 32 {
            let min = -(1i32 << (bits - 1));
            let max = (1i32 << (bits - 1)) - 1;
            if codes.iter().any(|&c| c < min || c > max) {
                return Err(CodePlaneError::CodeOutOfRange);
            }
        }
        let mask = self.code_mask_u32();
        self.write_raw_unchecked(start, end, codes.iter().map(|&c| (c as u32) & mask));
        Ok(())
    }

    fn checked_span(
        &self,
        x: usize,
        y: usize,
        len: usize,
    ) -> Result<Option<(usize, usize)>, CodePlaneError> {
        if y >= self.height || x.checked_add(len).is_none_or(|end| end > self.width) {
            return Err(CodePlaneError::OutsideActivePlane);
        }
        let Some(last) = len.checked_sub(1) else {
            return Ok(None);
        };
        let step = self.step_samples as usize;
        let start = y * self.stride_samples + x * step;
        let end = start + last * step + 1;
        Ok(Some((start, end)))
    }

    fn write_raw_unchecked(
        &mut self,
        start: usize,
        end: usize,
        raw_codes: impl Iterator<Item = u32>,
    ) {
        let step = self.step_samples as usize;
        let shift = self.encoding.bit_shift();
        let mask32 = self.code_mask_u32();
        let preserve = self.preserve_bits;
        match self.samples_mut() {
            SamplesMut::Bytes1(s) => {
                let clear = !((mask32 as u8) << shift);
                for (raw, dst) in raw_codes.zip(s[start..end].iter_mut().step_by(step)) {
                    let placed = (raw as u8) << shift;
                    *dst = if preserve {
                        (*dst & clear) | placed
                    } else {
                        placed
                    };
                }
            }
            SamplesMut::Words2(s) => {
                let clear = !((mask32 as u16) << shift);
                for (raw, dst) in raw_codes.zip(s[start..end].iter_mut().step_by(step)) {
                    let placed = (raw as u16) << shift;
                    *dst = if preserve {
                        (*dst & clear) | placed
                    } else {
                        placed
                    };
                }
            }
            SamplesMut::Words4(s) => {
                let clear = !(mask32 << shift);
                for (raw, dst) in raw_codes.zip(s[start..end].iter_mut().step_by(step)) {
                    let placed = raw << shift;
                    *dst = if preserve {
                        (*dst & clear) | placed
                    } else {
                        placed
                    };
                }
            }
        }
    }

    #[inline]
    fn samples_mut(&mut self) -> SamplesMut<'_> {
        match self.encoding.storage() {
            crate::ChannelType::U8 | crate::ChannelType::I8 => {
                SamplesMut::Bytes1(self.sample_bytes)
            }
            crate::ChannelType::U16 | crate::ChannelType::I16 => {
                SamplesMut::Words2(bytemuck::cast_slice_mut(self.sample_bytes))
            }
            crate::ChannelType::U32 | crate::ChannelType::I32 => {
                SamplesMut::Words4(bytemuck::cast_slice_mut(self.sample_bytes))
            }
            _ => unreachable!("CodePlaneMut only accepts 8/16/32-bit integer storage"),
        }
    }

    const fn code_mask_u32(&self) -> u32 {
        u32::MAX >> (32 - self.encoding.code_bits() as u32)
    }
}
