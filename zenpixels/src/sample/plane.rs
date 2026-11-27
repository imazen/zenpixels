//! Chroma siting, subsampling, and borrowed native integer [`CodePlane`].

use core::fmt;

use super::SampleEncoding;
use crate::buffer::{PlaneView, PlaneViewMut};
use crate::layout::SampleType;

/// Chroma resolution relative to the complete luma plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ChromaSubsampling {
    /// One luma plane and no chroma planes (AV1 `I400`, HEVC 4:0:0, monochrome JPEG).
    Mono,
    /// Equal horizontal and vertical resolution (4:4:4).
    Yuv444,
    /// Half horizontal resolution, rounded up at an odd edge (4:2:2).
    Yuv422,
    /// Half horizontal and vertical resolution, rounded up at an odd edge (4:2:0).
    Yuv420,
    /// Half vertical resolution, rounded up at an odd edge (4:4:0).
    Yuv440,
    /// Quarter horizontal resolution, rounded up at an odd edge (4:1:1).
    Yuv411,
}

impl ChromaSubsampling {
    /// Horizontal and vertical subsampling divisors `(hx, vy)`, or `None` for [`Mono`](Self::Mono).
    #[inline]
    #[must_use]
    pub const fn factors(self) -> Option<(u8, u8)> {
        match self {
            Self::Mono => None,
            Self::Yuv444 => Some((1, 1)),
            Self::Yuv422 => Some((2, 1)),
            Self::Yuv420 => Some((2, 2)),
            Self::Yuv440 => Some((1, 2)),
            Self::Yuv411 => Some((4, 1)),
        }
    }

    /// Chroma plane `(width, height)` for a luma size, rounded up at odd edges; `None` for [`Mono`](Self::Mono).
    #[inline]
    #[must_use]
    pub const fn chroma_size(self, width: u32, height: u32) -> Option<(u32, u32)> {
        match self.factors() {
            None => None,
            Some((hx, vy)) => Some((width.div_ceil(hx as u32), height.div_ceil(vy as u32))),
        }
    }
}

/// Declared chroma sample location relative to the luma sample grid.
///
/// This is a fact, not a fallback policy: [`Unknown`](Self::Unknown) is retained
/// rather than silently defaulting to [`Center`](Self::Center). A conversion that
/// requires siting either refuses `Unknown` on a subsampled axis or takes an
/// explicit caller fallback policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
#[doc(alias = "chroma_location")]
#[doc(alias = "chroma_position")]
#[doc(alias = "chroma_sample_position")]
pub enum ChromaSiting {
    /// The source did not declare a chroma sample location (AV1 `CSP_UNKNOWN`).
    #[default]
    Unknown,
    /// Centered horizontally and vertically in the subsampled luma cell (JPEG / H.265 type 1).
    Center,
    /// Horizontally co-sited with the left luma sample, vertically centered between rows
    /// (MPEG-2 / H.264 / AV1 `CSP_VERTICAL` / H.265 type 0).
    Left,
    /// Co-sited with the top-left luma sample (BT.2020 / AV1 `CSP_COLOCATED` / H.265 type 2).
    TopLeft,
    /// Horizontally centered, vertically co-sited with the top luma sample (H.265 type 3).
    Top,
    /// Co-sited with the bottom-left luma sample (H.265 type 4).
    BottomLeft,
    /// Horizontally centered, vertically co-sited with the bottom luma sample (H.265 type 5).
    Bottom,
}

impl ChromaSiting {
    /// Fractional `(dx, dy)` offset of chroma sample `(0, 0)` within its subsampling cell
    /// in **luma-pixel units** relative to the top-left luma sample center at `(0.0, 0.0)`.
    ///
    /// Returns `None` for [`Self::Unknown`] or [`ChromaSubsampling::Mono`].
    ///
    /// For [`ChromaSubsampling::Yuv420`] (`2×2`):
    /// - `TopLeft` → `(0.0, 0.0)`
    /// - `Left` → `(0.0, 0.5)`
    /// - `Center` → `(0.5, 0.5)`
    #[inline]
    #[must_use]
    pub const fn offset_in_luma_pixels(self, subsampling: ChromaSubsampling) -> Option<(f32, f32)> {
        match subsampling.factors() {
            Some((hx, vy)) => self.offset_in_luma_pixels_for_divisors(hx, vy),
            None => None,
        }
    }

    /// Fractional `(dx, dy)` offset of chroma sample `(0, 0)` within an arbitrary
    /// `horizontal_divisor × vertical_divisor` luma cell in **luma-pixel units** relative to
    /// the top-left luma sample center at `(0.0, 0.0)`, or `None` for [`Self::Unknown`].
    ///
    /// Divisors `<= 1` yield `0.0` along that axis.
    #[inline]
    #[must_use]
    pub const fn offset_in_luma_pixels_for_divisors(
        self,
        horizontal_divisor: u8,
        vertical_divisor: u8,
    ) -> Option<(f32, f32)> {
        let (hx, vy) = (horizontal_divisor, vertical_divisor);
        let cx = if hx <= 1 {
            0.0
        } else {
            (hx as f32 - 1.0) * 0.5
        };
        let cy = if vy <= 1 {
            0.0
        } else {
            (vy as f32 - 1.0) * 0.5
        };
        let by = if vy <= 1 { 0.0 } else { (vy - 1) as f32 };
        match self {
            Self::Unknown => None,
            Self::Center => Some((cx, cy)),
            Self::Left => Some((0.0, cy)),
            Self::TopLeft => Some((0.0, 0.0)),
            Self::Top => Some((cx, 0.0)),
            Self::BottomLeft => Some((0.0, by)),
            Self::Bottom => Some((cx, by)),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Words<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
}

/// Integer codes of one component in a borrowed plane, without normalization or color claims.
///
/// Code depth and bit placement come from [`SampleEncoding`]; padding bits are
/// ignored on read. Nothing is copied or allocated.
#[derive(Clone, Copy, Debug)]
#[doc(alias = "bit_plane")]
#[doc(alias = "raw_samples")]
pub struct CodePlane<'a> {
    words: Words<'a>,
    width: usize,
    height: usize,
    stride_words: usize,
    step_words: usize,
    encoding: SampleEncoding,
}

/// Codes cannot be read from a plane as requested.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CodePlaneError {
    /// The plane's sample type differs from the sample encoding's storage, or is not `U8`/`U16`.
    StorageMismatch,
    /// The component index is not below the component count, or the plane width/stride is invalid.
    Components,
    /// Buffer slice is too short or misaligned for the declared geometry.
    Geometry,
    /// A requested read or write extends beyond the active plane.
    OutsideActivePlane,
    /// A written code exceeds `(1 << code_bits) - 1`.
    CodeOutOfRange,
}

impl fmt::Display for CodePlaneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::StorageMismatch => "code plane storage disagrees with sample encoding",
            Self::Components => "component index or plane width disagrees with the component count",
            Self::Geometry => "plane dimensions, stride, or backing buffer length are invalid",
            Self::OutsideActivePlane => "code plane access extends beyond the active plane",
            Self::CodeOutOfRange => "code value exceeds the declared bit depth",
        })
    }
}

impl core::error::Error for CodePlaneError {}

/// Geometry and component selection for borrowing one component from an interleaved code plane.
///
/// Construct directly with struct literal syntax so all five fields are named at the call site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InterleavedCodeLayout {
    /// Active row width in storage words across all interleaved components (`width_pixels * component_count`).
    pub width_words: usize,
    /// Active plane height in rows.
    pub height: usize,
    /// Row stride in storage words (`>= width_words`).
    pub stride_words: usize,
    /// Number of interleaved components per element (e.g., `2` for `CbCr`).
    pub component_count: usize,
    /// Zero-based component index to extract (`0..component_count`).
    pub component_index: usize,
}

impl<'a> CodePlane<'a> {
    /// Borrow a single-component `u8` plane (`stride_words >= width_pixels`).
    pub fn from_u8(
        data: &'a [u8],
        width_pixels: usize,
        height: usize,
        stride_words: usize,
        encoding: SampleEncoding,
    ) -> Result<Self, CodePlaneError> {
        Self::component_from_u8(
            data,
            InterleavedCodeLayout {
                width_words: width_pixels,
                height,
                stride_words,
                component_count: 1,
                component_index: 0,
            },
            encoding,
        )
    }

    /// Borrow one component of an interleaved `u8` plane described by `layout`.
    pub fn component_from_u8(
        data: &'a [u8],
        layout: InterleavedCodeLayout,
        encoding: SampleEncoding,
    ) -> Result<Self, CodePlaneError> {
        let InterleavedCodeLayout {
            width_words,
            height,
            stride_words,
            component_count: components,
            component_index: index,
        } = layout;
        if encoding.storage() != SampleType::U8 {
            return Err(CodePlaneError::StorageMismatch);
        }
        if components == 0 || index >= components || width_words % components != 0 {
            return Err(CodePlaneError::Components);
        }
        if stride_words < width_words {
            return Err(CodePlaneError::Geometry);
        }
        let width = width_words / components;
        if width > 0 && height > 0 {
            let req = (height - 1)
                .checked_mul(stride_words)
                .and_then(|v| v.checked_add(width_words))
                .ok_or(CodePlaneError::Geometry)?;
            if data.len() < req {
                return Err(CodePlaneError::Geometry);
            }
        }
        let skip = if width == 0 || height == 0 { 0 } else { index };
        Ok(Self {
            words: Words::U8(&data[skip..]),
            width,
            height,
            stride_words,
            step_words: components,
            encoding,
        })
    }

    /// Borrow a single-component `u16` plane (`stride_words >= width_pixels`).
    pub fn from_u16(
        data: &'a [u16],
        width_pixels: usize,
        height: usize,
        stride_words: usize,
        encoding: SampleEncoding,
    ) -> Result<Self, CodePlaneError> {
        Self::component_from_u16(
            data,
            InterleavedCodeLayout {
                width_words: width_pixels,
                height,
                stride_words,
                component_count: 1,
                component_index: 0,
            },
            encoding,
        )
    }

    /// Borrow one component of an interleaved `u16` plane described by `layout`.
    pub fn component_from_u16(
        data: &'a [u16],
        layout: InterleavedCodeLayout,
        encoding: SampleEncoding,
    ) -> Result<Self, CodePlaneError> {
        let InterleavedCodeLayout {
            width_words,
            height,
            stride_words,
            component_count: components,
            component_index: index,
        } = layout;
        if encoding.storage() != SampleType::U16 {
            return Err(CodePlaneError::StorageMismatch);
        }
        if components == 0 || index >= components || width_words % components != 0 {
            return Err(CodePlaneError::Components);
        }
        if stride_words < width_words {
            return Err(CodePlaneError::Geometry);
        }
        let width = width_words / components;
        if width > 0 && height > 0 {
            let req = (height - 1)
                .checked_mul(stride_words)
                .and_then(|v| v.checked_add(width_words))
                .ok_or(CodePlaneError::Geometry)?;
            if data.len() < req {
                return Err(CodePlaneError::Geometry);
            }
        }
        let skip = if width == 0 || height == 0 { 0 } else { index };
        Ok(Self {
            words: Words::U16(&data[skip..]),
            width,
            height,
            stride_words,
            step_words: components,
            encoding,
        })
    }

    /// Borrow component `component_index` from a validated [`PlaneView<'a>`] zero-copy.
    pub fn from_plane_view(
        plane: PlaneView<'a>,
        component_index: usize,
        encoding: SampleEncoding,
    ) -> Result<Self, CodePlaneError> {
        let index = component_index;
        let spec = plane.format();
        if spec.packing.is_packed_word() {
            return Err(CodePlaneError::StorageMismatch);
        }
        let components = spec.components as usize;
        if components == 0 || index >= components {
            return Err(CodePlaneError::Components);
        }
        if spec.sample != encoding.storage() {
            return Err(CodePlaneError::StorageMismatch);
        }
        let width_words = (plane.width() as usize)
            .checked_mul(components)
            .ok_or(CodePlaneError::Geometry)?;
        let height = plane.height() as usize;
        match spec.sample {
            SampleType::U8 => Self::component_from_u8(
                plane.data(),
                InterleavedCodeLayout {
                    width_words,
                    height,
                    stride_words: plane.stride_bytes(),
                    component_count: components,
                    component_index: index,
                },
                encoding,
            ),
            SampleType::U16 => {
                if plane.stride_bytes() % 2 != 0 {
                    return Err(CodePlaneError::Geometry);
                }
                let words = bytemuck::try_cast_slice::<u8, u16>(plane.data())
                    .map_err(|_| CodePlaneError::Geometry)?;
                Self::component_from_u16(
                    words,
                    InterleavedCodeLayout {
                        width_words,
                        height,
                        stride_words: plane.stride_bytes() / 2,
                        component_count: components,
                        component_index: index,
                    },
                    encoding,
                )
            }
            _ => Err(CodePlaneError::StorageMismatch),
        }
    }

    /// Active samples per row.
    #[inline]
    #[must_use]
    pub const fn width(self) -> usize {
        self.width
    }

    /// Number of active rows.
    #[inline]
    #[must_use]
    pub const fn height(self) -> usize {
        self.height
    }

    /// Storage width, code depth, and bit placement.
    #[inline]
    #[must_use]
    pub const fn encoding(self) -> SampleEncoding {
        self.encoding
    }

    /// Extract one code, ignoring padding bits; outside the active plane returns `None`.
    #[must_use]
    pub fn code(self, x: usize, y: usize) -> Option<u16> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let i = y * self.stride_words + x * self.step_words;
        let word = match self.words {
            Words::U8(s) => u16::from(s[i]),
            Words::U16(s) => s[i],
        };
        Some((word >> self.encoding.bit_shift()) & self.code_mask())
    }

    /// Extract `codes.len()` consecutive codes of row `y` starting at column `x`.
    ///
    /// Padding bits are ignored. Validates bounds before writing; on error `codes` is unchanged.
    pub fn read_codes(self, x: usize, y: usize, codes: &mut [u16]) -> Result<(), CodePlaneError> {
        if y >= self.height
            || x.checked_add(codes.len())
                .is_none_or(|end| end > self.width)
        {
            return Err(CodePlaneError::OutsideActivePlane);
        }
        let Some(last) = codes.len().checked_sub(1) else {
            return Ok(());
        };
        let start = y * self.stride_words + x * self.step_words;
        let end = start + last * self.step_words + 1;
        let (shift, mask) = (self.encoding.bit_shift(), self.code_mask());
        match self.words {
            Words::U8(s) => {
                let words = &s[start..end];
                if self.step_words == 1 {
                    for (code, word) in codes.iter_mut().zip(words) {
                        *code = (u16::from(*word) >> shift) & mask;
                    }
                } else {
                    for (code, word) in codes.iter_mut().zip(words.iter().step_by(self.step_words))
                    {
                        *code = (u16::from(*word) >> shift) & mask;
                    }
                }
            }
            Words::U16(s) => {
                let words = &s[start..end];
                if self.step_words == 1 {
                    for (code, word) in codes.iter_mut().zip(words) {
                        *code = (*word >> shift) & mask;
                    }
                } else {
                    for (code, word) in codes.iter_mut().zip(words.iter().step_by(self.step_words))
                    {
                        *code = (*word >> shift) & mask;
                    }
                }
            }
        }
        Ok(())
    }

    const fn code_mask(self) -> u16 {
        ((1u32 << self.encoding.code_bits()) - 1) as u16
    }
}

#[derive(Debug)]
enum WordsMut<'a> {
    U8(&'a mut [u8]),
    U16(&'a mut [u16]),
}

/// Mutable view of one integer component in a borrowed plane, without normalization or color claims.
///
/// Code depth and bit placement come from [`SampleEncoding`]. By default, padding bits outside
/// `(mask << bit_shift)` are zeroed on write; call [`preserving_other_bits`](Self::preserving_other_bits)
/// to preserve surrounding bits when packing shared words.
#[derive(Debug)]
pub struct CodePlaneMut<'a> {
    words: WordsMut<'a>,
    width: usize,
    height: usize,
    stride_words: usize,
    step_words: usize,
    encoding: SampleEncoding,
    preserve_bits: bool,
}

impl<'a> CodePlaneMut<'a> {
    /// Borrow a single-component mutable `u8` plane (`stride_words >= width_pixels`).
    pub fn from_u8_mut(
        data: &'a mut [u8],
        width_pixels: usize,
        height: usize,
        stride_words: usize,
        encoding: SampleEncoding,
    ) -> Result<Self, CodePlaneError> {
        Self::component_from_u8_mut(
            data,
            InterleavedCodeLayout {
                width_words: width_pixels,
                height,
                stride_words,
                component_count: 1,
                component_index: 0,
            },
            encoding,
        )
    }

    /// Borrow one component of an interleaved mutable `u8` plane described by `layout`.
    pub fn component_from_u8_mut(
        data: &'a mut [u8],
        layout: InterleavedCodeLayout,
        encoding: SampleEncoding,
    ) -> Result<Self, CodePlaneError> {
        let InterleavedCodeLayout {
            width_words,
            height,
            stride_words,
            component_count: components,
            component_index: index,
        } = layout;
        if encoding.storage() != SampleType::U8 {
            return Err(CodePlaneError::StorageMismatch);
        }
        if components == 0 || index >= components || width_words % components != 0 {
            return Err(CodePlaneError::Components);
        }
        if stride_words < width_words {
            return Err(CodePlaneError::Geometry);
        }
        let width = width_words / components;
        if width > 0 && height > 0 {
            let req = (height - 1)
                .checked_mul(stride_words)
                .and_then(|v| v.checked_add(width_words))
                .ok_or(CodePlaneError::Geometry)?;
            if data.len() < req {
                return Err(CodePlaneError::Geometry);
            }
        }
        let skip = if width == 0 || height == 0 { 0 } else { index };
        Ok(Self {
            words: WordsMut::U8(&mut data[skip..]),
            width,
            height,
            stride_words,
            step_words: components,
            encoding,
            preserve_bits: false,
        })
    }

    /// Borrow a single-component mutable `u16` plane (`stride_words >= width_pixels`).
    pub fn from_u16_mut(
        data: &'a mut [u16],
        width_pixels: usize,
        height: usize,
        stride_words: usize,
        encoding: SampleEncoding,
    ) -> Result<Self, CodePlaneError> {
        Self::component_from_u16_mut(
            data,
            InterleavedCodeLayout {
                width_words: width_pixels,
                height,
                stride_words,
                component_count: 1,
                component_index: 0,
            },
            encoding,
        )
    }

    /// Borrow one component of an interleaved mutable `u16` plane described by `layout`.
    pub fn component_from_u16_mut(
        data: &'a mut [u16],
        layout: InterleavedCodeLayout,
        encoding: SampleEncoding,
    ) -> Result<Self, CodePlaneError> {
        let InterleavedCodeLayout {
            width_words,
            height,
            stride_words,
            component_count: components,
            component_index: index,
        } = layout;
        if encoding.storage() != SampleType::U16 {
            return Err(CodePlaneError::StorageMismatch);
        }
        if components == 0 || index >= components || width_words % components != 0 {
            return Err(CodePlaneError::Components);
        }
        if stride_words < width_words {
            return Err(CodePlaneError::Geometry);
        }
        let width = width_words / components;
        if width > 0 && height > 0 {
            let req = (height - 1)
                .checked_mul(stride_words)
                .and_then(|v| v.checked_add(width_words))
                .ok_or(CodePlaneError::Geometry)?;
            if data.len() < req {
                return Err(CodePlaneError::Geometry);
            }
        }
        let skip = if width == 0 || height == 0 { 0 } else { index };
        Ok(Self {
            words: WordsMut::U16(&mut data[skip..]),
            width,
            height,
            stride_words,
            step_words: components,
            encoding,
            preserve_bits: false,
        })
    }

    /// Borrow component `component_index` mutably from a validated [`PlaneViewMut<'a>`] zero-copy.
    pub fn from_plane_view_mut(
        plane: PlaneViewMut<'a>,
        component_index: usize,
        encoding: SampleEncoding,
    ) -> Result<Self, CodePlaneError> {
        let index = component_index;
        let spec = plane.format();
        if spec.packing.is_packed_word() {
            return Err(CodePlaneError::StorageMismatch);
        }
        let components = spec.components as usize;
        if components == 0 || index >= components {
            return Err(CodePlaneError::Components);
        }
        if spec.sample != encoding.storage() {
            return Err(CodePlaneError::StorageMismatch);
        }
        let width_words = (plane.width() as usize)
            .checked_mul(components)
            .ok_or(CodePlaneError::Geometry)?;
        let height = plane.height() as usize;
        let stride = plane.stride_bytes();
        let data = plane.into_data_mut();
        match spec.sample {
            SampleType::U8 => Self::component_from_u8_mut(
                data,
                InterleavedCodeLayout {
                    width_words,
                    height,
                    stride_words: stride,
                    component_count: components,
                    component_index: index,
                },
                encoding,
            ),
            SampleType::U16 => {
                if stride % 2 != 0 {
                    return Err(CodePlaneError::Geometry);
                }
                let words = bytemuck::try_cast_slice_mut::<u8, u16>(data)
                    .map_err(|_| CodePlaneError::Geometry)?;
                Self::component_from_u16_mut(
                    words,
                    InterleavedCodeLayout {
                        width_words,
                        height,
                        stride_words: stride / 2,
                        component_count: components,
                        component_index: index,
                    },
                    encoding,
                )
            }
            _ => Err(CodePlaneError::StorageMismatch),
        }
    }

    /// Preserve bits outside `(mask << bit_shift)` in each written sample word instead of
    /// zeroing padding bits.
    #[inline]
    #[must_use]
    pub const fn preserving_other_bits(mut self) -> Self {
        self.preserve_bits = true;
        self
    }

    /// Active samples per row.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Number of active rows.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Storage width, code depth, and bit placement.
    #[inline]
    #[must_use]
    pub const fn encoding(&self) -> SampleEncoding {
        self.encoding
    }

    /// Borrow as a read-only [`CodePlane`].
    #[inline]
    #[must_use]
    pub fn as_ref(&self) -> CodePlane<'_> {
        let words = match &self.words {
            WordsMut::U8(s) => Words::U8(s),
            WordsMut::U16(s) => Words::U16(s),
        };
        CodePlane {
            words,
            width: self.width,
            height: self.height,
            stride_words: self.stride_words,
            step_words: self.step_words,
            encoding: self.encoding,
        }
    }

    /// Extract one code, ignoring padding bits; outside the active plane returns `None`.
    #[inline]
    #[must_use]
    pub fn code(&self, x: usize, y: usize) -> Option<u16> {
        self.as_ref().code(x, y)
    }

    /// Extract `codes.len()` consecutive codes of row `y` starting at column `x`.
    #[inline]
    pub fn read_codes(&self, x: usize, y: usize, codes: &mut [u16]) -> Result<(), CodePlaneError> {
        self.as_ref().read_codes(x, y, codes)
    }

    /// Write one code at `(x, y)`. Validated before mutating.
    #[inline]
    pub fn set_code(&mut self, x: usize, y: usize, code: u16) -> Result<(), CodePlaneError> {
        self.write_codes(x, y, core::slice::from_ref(&code))
    }

    /// Write `codes.len()` consecutive codes on row `y` starting at column `x`.
    ///
    /// Validates bounds and code range (`<= (1 << code_bits) - 1`) before writing anything.
    pub fn write_codes(&mut self, x: usize, y: usize, codes: &[u16]) -> Result<(), CodePlaneError> {
        if y >= self.height
            || x.checked_add(codes.len())
                .is_none_or(|end| end > self.width)
        {
            return Err(CodePlaneError::OutsideActivePlane);
        }
        let mask = ((1u32 << self.encoding.code_bits()) - 1) as u16;
        if codes.iter().any(|&c| c > mask) {
            return Err(CodePlaneError::CodeOutOfRange);
        }
        let Some(last) = codes.len().checked_sub(1) else {
            return Ok(());
        };
        let start = y * self.stride_words + x * self.step_words;
        let end = start + last * self.step_words + 1;
        let shift = self.encoding.bit_shift();
        let preserve = self.preserve_bits;
        let step = self.step_words;
        match &mut self.words {
            WordsMut::U8(s) => {
                let clear = !((mask as u8) << shift);
                let words = &mut s[start..end];
                if step == 1 {
                    for (&c, dst) in codes.iter().zip(words.iter_mut()) {
                        let placed = (c as u8) << shift;
                        *dst = if preserve {
                            (*dst & clear) | placed
                        } else {
                            placed
                        };
                    }
                } else {
                    for (&c, dst) in codes.iter().zip(words.iter_mut().step_by(step)) {
                        let placed = (c as u8) << shift;
                        *dst = if preserve {
                            (*dst & clear) | placed
                        } else {
                            placed
                        };
                    }
                }
            }
            WordsMut::U16(s) => {
                let clear = !(mask << shift);
                let words = &mut s[start..end];
                if step == 1 {
                    for (&c, dst) in codes.iter().zip(words.iter_mut()) {
                        let placed = c << shift;
                        *dst = if preserve {
                            (*dst & clear) | placed
                        } else {
                            placed
                        };
                    }
                } else {
                    for (&c, dst) in codes.iter().zip(words.iter_mut().step_by(step)) {
                        let placed = c << shift;
                        *dst = if preserve {
                            (*dst & clear) | placed
                        } else {
                            placed
                        };
                    }
                }
            }
        }
        Ok(())
    }
}
