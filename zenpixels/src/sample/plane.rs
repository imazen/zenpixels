//! Chroma siting, subsampling, and borrowed native integer [`CodePlane`].

use core::fmt;

use super::SampleEncoding;
use crate::buffer::PlaneView;
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
    /// Fractional `(dx, dy)` offset of chroma sample `(0, 0)` within its `hx × vy` luma cell
    /// (in luma-pixel units relative to the top-left luma sample center at `(0.0, 0.0)`),
    /// or `None` for [`Self::Unknown`].
    ///
    /// For `hx = 2, vy = 2` (4:2:0):
    /// - `TopLeft` → `(0.0, 0.0)`
    /// - `Left` → `(0.0, 0.5)`
    /// - `Center` → `(0.5, 0.5)`
    #[inline]
    #[must_use]
    pub const fn offset_in_cell(self, hx: u8, vy: u8) -> Option<(f32, f32)> {
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
    /// A requested read extends beyond the active plane.
    OutsideActivePlane,
}

impl fmt::Display for CodePlaneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::StorageMismatch => "code plane storage disagrees with sample encoding",
            Self::Components => "component index or plane width disagrees with the component count",
            Self::Geometry => "plane dimensions, stride, or backing buffer length are invalid",
            Self::OutsideActivePlane => "code plane read extends beyond the active plane",
        })
    }
}

impl core::error::Error for CodePlaneError {}

impl<'a> CodePlane<'a> {
    /// Borrow a single-component `u8` plane (`stride_words >= width`).
    pub fn from_u8(
        data: &'a [u8],
        width: usize,
        height: usize,
        stride_words: usize,
        encoding: SampleEncoding,
    ) -> Result<Self, CodePlaneError> {
        Self::component_from_u8(data, width, height, stride_words, 1, 0, encoding)
    }

    /// Borrow component `index` of an interleaved `u8` plane (`width_words % components == 0`).
    pub fn component_from_u8(
        data: &'a [u8],
        width_words: usize,
        height: usize,
        stride_words: usize,
        components: usize,
        index: usize,
        encoding: SampleEncoding,
    ) -> Result<Self, CodePlaneError> {
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

    /// Borrow a single-component `u16` plane (`stride_words >= width`).
    pub fn from_u16(
        data: &'a [u16],
        width: usize,
        height: usize,
        stride_words: usize,
        encoding: SampleEncoding,
    ) -> Result<Self, CodePlaneError> {
        Self::component_from_u16(data, width, height, stride_words, 1, 0, encoding)
    }

    /// Borrow component `index` of an interleaved `u16` plane (`width_words % components == 0`).
    pub fn component_from_u16(
        data: &'a [u16],
        width_words: usize,
        height: usize,
        stride_words: usize,
        components: usize,
        index: usize,
        encoding: SampleEncoding,
    ) -> Result<Self, CodePlaneError> {
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

    /// Borrow component `index` from a validated [`PlaneView<'a>`] zero-copy.
    pub fn from_plane_view(
        plane: PlaneView<'a>,
        index: usize,
        encoding: SampleEncoding,
    ) -> Result<Self, CodePlaneError> {
        let spec = plane.spec();
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
                width_words,
                height,
                plane.stride(),
                components,
                index,
                encoding,
            ),
            SampleType::U16 => {
                if plane.stride() % 2 != 0 {
                    return Err(CodePlaneError::Geometry);
                }
                let words = bytemuck::try_cast_slice::<u8, u16>(plane.data())
                    .map_err(|_| CodePlaneError::Geometry)?;
                Self::component_from_u16(
                    words,
                    width_words,
                    height,
                    plane.stride() / 2,
                    components,
                    index,
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
