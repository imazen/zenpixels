use super::SampleEncoding;
use crate::ChannelType;
use core::fmt;

/// Chroma resolution relative to the complete luma plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChromaSubsampling {
    /// Equal horizontal and vertical resolution.
    Yuv444,
    /// Half horizontal resolution, rounded up at an odd edge.
    Yuv422,
    /// Half horizontal and vertical resolution, rounded up at odd edges.
    Yuv420,
}

/// Declared chroma sample location. This is a fact, not a fallback policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChromaSiting {
    /// The source did not establish a location.
    Unknown,
    /// Centered horizontally and vertically in the subsampled luma cell.
    Center,
    /// Horizontally co-sited, vertically centered.
    Left,
    /// Co-sited with the top-left luma sample.
    TopLeft,
}

#[derive(Clone, Copy, Debug)]
enum Words<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
}

/// One borrowed unsigned component plane, without normalization or color claims.
///
/// Typed words use native endianness. Code depth/placement come from
/// [`crate::sample::SampleEncoding`]; padding bits are ignored on reads. The
/// enclosing format supplies roles, range, matrix, transfer and sample location.
/// No pixel payload is copied or allocated. Final-row padding need not exist.
///
/// A component need not occupy consecutive words. The interleaved constructors
/// take the distance between neighboring samples of this component within a row,
/// so one type borrows planar, semi-planar and interleaved native layouts:
///
/// ```
/// use zenpixels::{ChannelType, sample::{CodePlane, SampleEncoding}};
/// // P010 chroma: one interleaved Cb/Cr plane, 10 high bits per little-endian word.
/// let p010 = SampleEncoding::new(ChannelType::U16, 10, 6).unwrap();
/// let uv: [u16; 4] = [512 << 6, 100 << 6, 300 << 6, 1023 << 6]; // Cb Cr Cb Cr
/// let cb = CodePlane::from_u16_interleaved(&uv, 2, 1, 8, 2, p010).unwrap();
/// let cr = CodePlane::from_u16_interleaved(&uv[1..], 2, 1, 8, 2, p010).unwrap();
/// assert_eq!((cb.code(1, 0), cr.code(1, 0)), (Some(300), Some(1023)));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct CodePlane<'a> {
    words: Words<'a>,
    width: usize,
    height: usize,
    stride_words: usize,
    step_words: usize,
    encoding: SampleEncoding,
}

/// A borrowed component plane does not fit its physical storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CodePlaneError {
    /// The sample encoding names a different storage word type.
    StorageMismatch,
    /// A byte stride does not hold a whole number of storage words.
    MisalignedStride,
    /// A nonempty plane's stride is shorter than its active row.
    ShortStride,
    /// Storage ends before the last active sample.
    ShortBuffer,
    /// Computing the last active sample overflows addressable storage.
    GeometryOverflow,
    /// An interleaved component's sample step is zero.
    ZeroStep,
    /// A requested read extends beyond the active plane.
    OutsideActivePlane,
}
impl fmt::Display for CodePlaneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::StorageMismatch => "code plane storage disagrees with sample encoding",
            Self::MisalignedStride => "code plane stride must hold whole words",
            Self::ShortStride => "code plane stride is shorter than an active row",
            Self::ShortBuffer => "code plane storage ends before the last active sample",
            Self::GeometryOverflow => "code plane geometry overflows addressable storage",
            Self::ZeroStep => "code plane sample step must be at least one word",
            Self::OutsideActivePlane => "code plane read extends beyond the active plane",
        })
    }
}
impl core::error::Error for CodePlaneError {}

impl<'a> CodePlane<'a> {
    /// Borrow U8 words with an explicit byte stride and sample encoding.
    pub fn from_u8(
        words: &'a [u8],
        width: usize,
        height: usize,
        stride_bytes: usize,
        encoding: SampleEncoding,
    ) -> Result<Self, CodePlaneError> {
        Self::new(Words::U8(words), width, height, stride_bytes, 1, encoding)
    }

    /// Borrow U16 words with an explicit byte stride and sample encoding.
    pub fn from_u16(
        words: &'a [u16],
        width: usize,
        height: usize,
        stride_bytes: usize,
        encoding: SampleEncoding,
    ) -> Result<Self, CodePlaneError> {
        Self::new(Words::U16(words), width, height, stride_bytes, 1, encoding)
    }

    /// Borrow one component of interleaved U8 words.
    ///
    /// `words` starts at this component's first sample; `step_words` is the
    /// distance to the same component of the next pixel. Rows are
    /// `stride_bytes` apart. Construction checks geometry without reading words.
    pub fn from_u8_interleaved(
        words: &'a [u8],
        width: usize,
        height: usize,
        stride_bytes: usize,
        step_words: usize,
        encoding: SampleEncoding,
    ) -> Result<Self, CodePlaneError> {
        Self::new(
            Words::U8(words),
            width,
            height,
            stride_bytes,
            step_words,
            encoding,
        )
    }

    /// Borrow one component of interleaved U16 words, such as P010 chroma.
    ///
    /// `words` starts at this component's first sample; `step_words` is the
    /// distance to the same component of the next pixel. Rows are
    /// `stride_bytes` apart. Construction checks geometry without reading words.
    pub fn from_u16_interleaved(
        words: &'a [u16],
        width: usize,
        height: usize,
        stride_bytes: usize,
        step_words: usize,
        encoding: SampleEncoding,
    ) -> Result<Self, CodePlaneError> {
        Self::new(
            Words::U16(words),
            width,
            height,
            stride_bytes,
            step_words,
            encoding,
        )
    }

    fn new(
        words: Words<'a>,
        width: usize,
        height: usize,
        stride_bytes: usize,
        step_words: usize,
        encoding: SampleEncoding,
    ) -> Result<Self, CodePlaneError> {
        let (storage, len, bytes) = match words {
            Words::U8(s) => (ChannelType::U8, s.len(), 1),
            Words::U16(s) => (ChannelType::U16, s.len(), 2),
        };
        if storage != encoding.storage() {
            return Err(CodePlaneError::StorageMismatch);
        }
        if step_words == 0 {
            return Err(CodePlaneError::ZeroStep);
        }
        if stride_bytes % bytes != 0 {
            return Err(CodePlaneError::MisalignedStride);
        }
        let stride_words = stride_bytes / bytes;
        if width != 0 && height != 0 {
            // One active row of this component spans (width - 1) steps + 1 word.
            let row_span = (width - 1)
                .checked_mul(step_words)
                .and_then(|v| v.checked_add(1))
                .ok_or(CodePlaneError::GeometryOverflow)?;
            if stride_words < row_span {
                return Err(CodePlaneError::ShortStride);
            }
            let needed = (height - 1)
                .checked_mul(stride_words)
                .and_then(|v| v.checked_add(row_span))
                .ok_or(CodePlaneError::GeometryOverflow)?;
            if needed > len {
                return Err(CodePlaneError::ShortBuffer);
            }
        }
        Ok(Self {
            words,
            width,
            height,
            stride_words,
            step_words,
            encoding,
        })
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

    /// Words between neighboring samples of this component in a row (1 = planar).
    pub const fn step_words(self) -> usize {
        self.step_words
    }

    /// Extract one code, ignoring padding bits; outside the active plane is None.
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

    /// Extract `codes.len()` consecutive codes of row `y`, starting at column `x`.
    ///
    /// Padding bits are ignored, as in [`code`](Self::code). The extent is checked
    /// before writing; on error `codes` is unchanged. This reads only the requested
    /// samples and allocates nothing.
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
        // Construction proved the last active sample is addressable, so these
        // products of smaller coordinates cannot overflow.
        let start = y * self.stride_words + x * self.step_words;
        let end = start + last * self.step_words + 1;
        let (shift, mask) = (self.encoding.bit_shift(), self.code_mask());
        match self.words {
            Words::U8(s) => {
                for (code, word) in codes
                    .iter_mut()
                    .zip(s[start..end].iter().step_by(self.step_words))
                {
                    *code = (u16::from(*word) >> shift) & mask;
                }
            }
            Words::U16(s) => {
                for (code, word) in codes
                    .iter_mut()
                    .zip(s[start..end].iter().step_by(self.step_words))
                {
                    *code = (*word >> shift) & mask;
                }
            }
        }
        Ok(())
    }

    const fn code_mask(self) -> u16 {
        ((1u32 << self.encoding.code_bits()) - 1) as u16
    }
}
