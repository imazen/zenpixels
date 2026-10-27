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
#[derive(Clone, Copy, Debug)]
pub struct CodePlane<'a> {
    words: Words<'a>,
    width: usize,
    height: usize,
    stride_words: usize,
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
}
impl fmt::Display for CodePlaneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::StorageMismatch => "code plane storage disagrees with sample encoding",
            Self::MisalignedStride => "code plane stride must hold whole words",
            Self::ShortStride => "code plane stride is shorter than an active row",
            Self::ShortBuffer => "code plane storage ends before the last active sample",
            Self::GeometryOverflow => "code plane geometry overflows addressable storage",
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
        Self::new(Words::U8(words), width, height, stride_bytes, encoding)
    }

    /// Borrow U16 words with an explicit byte stride and sample encoding.
    pub fn from_u16(
        words: &'a [u16],
        width: usize,
        height: usize,
        stride_bytes: usize,
        encoding: SampleEncoding,
    ) -> Result<Self, CodePlaneError> {
        Self::new(Words::U16(words), width, height, stride_bytes, encoding)
    }

    fn new(
        words: Words<'a>,
        width: usize,
        height: usize,
        stride_bytes: usize,
        encoding: SampleEncoding,
    ) -> Result<Self, CodePlaneError> {
        let (storage, len, bytes) = match words {
            Words::U8(s) => (ChannelType::U8, s.len(), 1),
            Words::U16(s) => (ChannelType::U16, s.len(), 2),
        };
        if storage != encoding.storage() {
            return Err(CodePlaneError::StorageMismatch);
        }
        if stride_bytes % bytes != 0 {
            return Err(CodePlaneError::MisalignedStride);
        }
        let stride_words = stride_bytes / bytes;
        if width != 0 && height != 0 {
            if stride_words < width {
                return Err(CodePlaneError::ShortStride);
            }
            let needed = (height - 1)
                .checked_mul(stride_words)
                .and_then(|v| v.checked_add(width))
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

    /// Extract one code, ignoring padding bits; outside the active plane is None.
    pub fn code(self, x: usize, y: usize) -> Option<u16> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let i = y * self.stride_words + x;
        let word = match self.words {
            Words::U8(s) => u16::from(s[i]),
            Words::U16(s) => s[i],
        };
        Some((word >> self.encoding.bit_shift()) & ((1u32 << self.encoding.code_bits()) - 1) as u16)
    }
}
