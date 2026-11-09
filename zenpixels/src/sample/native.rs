//! Borrowed native integer words, separate from canonical processing samples.
use crate::{ChannelLayout, ChannelType};
use core::fmt;

/// Interpretation of one integer component per native-endian storage word.
/// Signed codes use two's complement in the declared payload bits, not in padding.
/// This does not describe bit-packed multi-component words such as RGB565.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NativeEncoding {
    storage: ChannelType,
    bits: u8,
    shift: u8,
}
impl NativeEncoding {
    /// Validate payload placement. Floating point and non-integer storage refuse.
    pub const fn new(storage: ChannelType, bits: u8, shift: u8) -> Result<Self, NativeCodeError> {
        let width = match storage {
            ChannelType::U8 | ChannelType::I8 => 8,
            ChannelType::U16 | ChannelType::I16 => 16,
            ChannelType::U32 | ChannelType::I32 => 32,
            ChannelType::U64 | ChannelType::I64 => 64,
            _ => return Err(NativeCodeError::Encoding),
        };
        if bits == 0 || bits > width || shift > width - bits {
            return Err(NativeCodeError::Encoding);
        }
        Ok(Self {
            storage,
            bits,
            shift,
        })
    }
    /// Physical word type, independent of payload precision.
    pub const fn storage(self) -> ChannelType {
        self.storage
    }
    /// Payload width, excluding padding.
    pub const fn code_bits(self) -> u8 {
        self.bits
    }
    /// Low padding bits.
    pub const fn bit_shift(self) -> u8 {
        self.shift
    }
    fn signed(self) -> bool {
        matches!(
            self.storage,
            ChannelType::I8 | ChannelType::I16 | ChannelType::I32 | ChannelType::I64
        )
    }
    fn decode(self, bytes: &[u8]) -> i128 {
        let word = match bytes.len() {
            1 => u64::from(bytes[0]),
            2 => u64::from(u16::from_ne_bytes(bytes.try_into().unwrap())),
            4 => u64::from(u32::from_ne_bytes(bytes.try_into().unwrap())),
            8 => u64::from_ne_bytes(bytes.try_into().unwrap()),
            _ => unreachable!(),
        };
        let mask = u64::MAX >> (64 - self.bits);
        let code = (word >> self.shift) & mask;
        if self.signed() && code & (1u64 << (self.bits - 1)) != 0 {
            i128::from(code) - (1i128 << self.bits)
        } else {
            i128::from(code)
        }
    }
}

/// Invalid native-code interpretation or geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum NativeCodeError {
    /// Unsupported storage or payload outside its word.
    Encoding,
    /// Layout/channel count mismatch or mixed storage word types.
    Channels,
    /// Overflow, insufficient bytes or stride shorter than the active row.
    Geometry,
}
impl fmt::Display for NativeCodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid native code view: {self:?}")
    }
}
impl core::error::Error for NativeCodeError {}

/// Strided native integer components. Each component occupies a separate word.
///
/// Construction never scans or normalizes samples. Borrowed row bytes preserve
/// padding exactly; `code` ignores padding and sign-extends the payload. There is
/// deliberately no conversion to `PixelSlice`: canonicalization is explicit.
/// Words are native-endian and may be byte-unaligned. RGB565 and other external
/// packings must be unpacked before constructing this view.
#[derive(Clone, Copy, Debug)]
pub struct NativeCodeSlice<'a> {
    bytes: &'a [u8],
    width: u32,
    height: u32,
    stride: usize,
    layout: ChannelLayout,
    encodings: &'a [NativeEncoding],
    row_bytes: usize,
}
impl<'a> NativeCodeSlice<'a> {
    /// Borrow interleaved Gray, GrayAlpha, RGB or RGBA words, with byte stride.
    /// All components have the same storage type but can have different payloads.
    pub fn new(
        bytes: &'a [u8],
        width: u32,
        height: u32,
        stride: usize,
        layout: ChannelLayout,
        encodings: &'a [NativeEncoding],
    ) -> Result<Self, NativeCodeError> {
        let channels = match layout {
            ChannelLayout::Gray => 1,
            ChannelLayout::GrayAlpha => 2,
            ChannelLayout::Rgb => 3,
            ChannelLayout::Rgba => 4,
            _ => return Err(NativeCodeError::Channels),
        };
        if encodings.len() != channels
            || encodings.iter().any(|e| e.storage != encodings[0].storage)
        {
            return Err(NativeCodeError::Channels);
        }
        let row_bytes = (width as usize)
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(encodings[0].storage.byte_size()))
            .ok_or(NativeCodeError::Geometry)?;
        if width != 0 && height != 0 {
            let extent = (height as usize - 1)
                .checked_mul(stride)
                .and_then(|n| n.checked_add(row_bytes))
                .ok_or(NativeCodeError::Geometry)?;
            if stride < row_bytes || extent > bytes.len() {
                return Err(NativeCodeError::Geometry);
            }
        }
        Ok(Self {
            bytes,
            width,
            height,
            stride,
            layout,
            encodings,
            row_bytes,
        })
    }
    /// Active pixel width.
    pub const fn width(self) -> u32 {
        self.width
    }
    /// Active row count.
    pub const fn height(self) -> u32 {
        self.height
    }
    /// Component ordering, without a color-space claim.
    pub const fn layout(self) -> ChannelLayout {
        self.layout
    }
    /// Per-channel interpretation, in layout order.
    pub const fn encodings(self) -> &'a [NativeEncoding] {
        self.encodings
    }
    /// Exact active storage bytes, including sample padding but excluding row padding.
    pub fn row(self, y: u32) -> Option<&'a [u8]> {
        if y >= self.height {
            return None;
        }
        if self.row_bytes == 0 {
            return Some(&self.bytes[..0]);
        }
        let start = y as usize * self.stride;
        Some(&self.bytes[start..start + self.row_bytes])
    }
    /// Exact integer payload; i128 represents all supported unsigned and signed codes.
    pub fn code(self, x: u32, y: u32, channel: usize) -> Option<i128> {
        if x >= self.width {
            return None;
        }
        let e = *self.encodings.get(channel)?;
        let size = e.storage.byte_size();
        let start = (x as usize * self.encodings.len() + channel) * size;
        Some(e.decode(&self.row(y)?[start..start + size]))
    }
}
