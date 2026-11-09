//! Borrowed native integer words, separate from canonical processing samples.
use super::SampleEncoding;
use crate::{ChannelLayout, plane::PlaneRef};
use core::fmt;

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
    plane: PlaneRef<'a, u8>,
    width: u32,
    layout: ChannelLayout,
    encodings: &'a [SampleEncoding],
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
        encodings: &'a [SampleEncoding],
    ) -> Result<Self, NativeCodeError> {
        let channels = match layout {
            ChannelLayout::Gray => 1,
            ChannelLayout::GrayAlpha => 2,
            ChannelLayout::Rgb => 3,
            ChannelLayout::Rgba => 4,
            _ => return Err(NativeCodeError::Channels),
        };
        if encodings.len() != channels
            || encodings
                .iter()
                .any(|e| e.storage() != encodings[0].storage())
        {
            return Err(NativeCodeError::Channels);
        }
        let row_bytes = (width as usize)
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(encodings[0].storage().byte_size()))
            .ok_or(NativeCodeError::Geometry)?;
        let plane = PlaneRef::new(bytes, row_bytes, height as usize, stride)
            .map_err(|_| NativeCodeError::Geometry)?;
        Ok(Self {
            plane,
            width,
            layout,
            encodings,
        })
    }
    /// Active pixel width.
    pub const fn width(self) -> u32 {
        self.width
    }
    /// Active row count.
    pub fn height(self) -> u32 {
        self.plane.height() as u32
    }
    /// Component ordering, without a color-space claim.
    pub const fn layout(self) -> ChannelLayout {
        self.layout
    }
    /// Per-channel interpretation, in layout order.
    pub const fn encodings(self) -> &'a [SampleEncoding] {
        self.encodings
    }
    /// Exact active storage bytes, including sample padding but excluding row padding.
    pub fn row(self, y: u32) -> Option<&'a [u8]> {
        if y as usize >= self.plane.height() {
            return None;
        }
        let start = if self.plane.width() == 0 {
            0
        } else {
            y as usize * self.plane.stride()
        };
        Some(&self.plane.into_strided_samples()[start..start + self.plane.width()])
    }

    /// Exact integer payload; i128 represents all supported unsigned and signed codes.
    pub fn code(self, x: u32, y: u32, channel: usize) -> Option<i128> {
        if x >= self.width {
            return None;
        }
        let e = *self.encodings.get(channel)?;
        let size = e.storage().byte_size();
        let start = (x as usize * self.encodings.len() + channel) * size;
        Some(e.decode(&self.row(y)?[start..start + size]))
    }
}
