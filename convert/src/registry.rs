//! Codec format registry — static tables of what each codec can produce and consume.
//!
//! These tables are derived from the actual `supported_descriptors()` in each
//! zen* codec's zencodec-types integration. Extended (internal-API-only) formats
//! are listed separately where they exist.
//!
//! # Effective bits
//!
//! The `effective_bits` field tracks the actual precision of data within its
//! container type. Two u16 values may have different effective precision:
//! - PNG 16-bit: u16 with 16 effective bits (full range)
//! - AVIF 10-bit decoded to u16: 10 effective bits (top bits are replicated)
//! - Farbfeld: u16 with 16 effective bits
//!
//! This matters for provenance: converting a 10-bit-effective u16 to u8 loses
//! only 2 bits, not 8.

use crate::{ChannelLayout, ChannelType, ColorPrimaries, PixelDescriptor, TransferFunction};

/// A format a codec can produce (decode) or consume (encode).
#[derive(Clone, Copy, Debug)]
pub struct FormatEntry {
    /// The pixel descriptor for this format.
    pub descriptor: PixelDescriptor,

    /// Effective precision bits within the container type.
    ///
    /// Usually matches the container (u8=8, u16=16, f32=32), but can differ:
    /// - AVIF 10-bit source decoded to u8: effective_bits = 8 (precision lost)
    /// - JPEG f32 precise decode: effective_bits = 10 (debiased dequant)
    /// - PNG 1-bit gray decoded to u8: effective_bits = 8 (scaled to fill range)
    /// - Farbfeld u16: effective_bits = 16 (full range)
    pub effective_bits: u8,

    /// Whether output values can exceed the nominal range.
    ///
    /// JPEG f32 decode preserves IDCT ringing, producing values outside [0.0, 1.0].
    /// Most codecs clamp to nominal range.
    pub can_overshoot: bool,
}

impl FormatEntry {
    /// Create a format entry with standard precision (matches container type).
    const fn standard(descriptor: PixelDescriptor) -> Self {
        let effective_bits = match descriptor.channel_type() {
            ChannelType::U8 => 8,
            ChannelType::U16 => 16,
            ChannelType::F16 => 11,
            ChannelType::F32 => 32,
            _ => 0,
        };
        Self {
            descriptor,
            effective_bits,
            can_overshoot: false,
        }
    }

    /// Create a format entry with custom effective bits.
    const fn with_bits(descriptor: PixelDescriptor, effective_bits: u8) -> Self {
        Self {
            descriptor,
            effective_bits,
            can_overshoot: false,
        }
    }

    /// Create a format entry that can overshoot nominal range.
    const fn overshoot(descriptor: PixelDescriptor, effective_bits: u8) -> Self {
        Self {
            descriptor,
            effective_bits,
            can_overshoot: true,
        }
    }
}

/// Static description of a codec's format capabilities.
#[derive(Clone, Debug)]
pub struct CodecFormats {
    /// Codec name (e.g. "jpeg", "png").
    pub name: &'static str,
    /// Formats the decoder can produce (via zencodec `supported_descriptors`).
    pub decode_outputs: &'static [FormatEntry],
    /// Formats the encoder can accept (via zencodec `supported_descriptors`).
    pub encode_inputs: &'static [FormatEntry],
    /// Whether ICC profiles are extracted on decode.
    pub icc_decode: bool,
    /// Whether ICC profiles can be embedded on encode.
    pub icc_encode: bool,
    /// Whether CICP signaling is supported.
    pub cicp: bool,
}

// ═══════════════════════════════════════════════════════════════════════
// JPEG (zenjpeg)
// ═══════════════════════════════════════════════════════════════════════
//
// Decode: u8 sRGB only via zencodec API. Internal API also supports:
//   - SrgbF32: f32 sRGB gamma, unclamped (preserves IDCT ringing)
//   - LinearF32: f32 linear, unclamped
//   - SrgbF32Precise: f32 sRGB with debiased dequantization (~10-bit effective)
//   - LinearF32Precise: f32 linear with debiased dequantization
// All f32 decode paths can produce values outside [0.0, 1.0] (overshoot).
//
// Encode: accepts u8 sRGB, u16 sRGB, f32 linear. Internal conversion for
// non-native formats.

static JPEG_DECODE: &[FormatEntry] = &[
    FormatEntry::with_bits(PixelDescriptor::RGB8_SRGB, 8),
    FormatEntry::with_bits(PixelDescriptor::RGBA8_SRGB, 8),
    FormatEntry::with_bits(PixelDescriptor::GRAY8_SRGB, 8),
    FormatEntry::with_bits(PixelDescriptor::BGRA8_SRGB, 8),
];

static JPEG_ENCODE: &[FormatEntry] = &[
    FormatEntry::standard(PixelDescriptor::RGB8_SRGB),
    FormatEntry::standard(PixelDescriptor::RGBA8_SRGB),
    FormatEntry::standard(PixelDescriptor::GRAY8_SRGB),
    FormatEntry::standard(PixelDescriptor::BGRA8_SRGB),
    FormatEntry::standard(PixelDescriptor::RGB16_SRGB),
    FormatEntry::standard(PixelDescriptor::RGBA16_SRGB),
    FormatEntry::standard(PixelDescriptor::GRAY16_SRGB),
    FormatEntry::standard(PixelDescriptor::RGBF32_LINEAR),
    FormatEntry::standard(PixelDescriptor::RGBAF32_LINEAR),
    FormatEntry::standard(PixelDescriptor::GRAYF32_LINEAR),
];

/// Extended JPEG decode formats available via internal API (not zencodec).
///
/// These require using zenjpeg's `OutputTarget` directly rather than the
/// zencodec `DecoderConfig` interface. They provide higher quality decode
/// at the cost of API portability.
///
/// **Deblocking/debiased/XYB decode modes** produce f32 output where u8
/// would truncate precision — the extended precision from debiased
/// dequantization or XYB inverse transform cannot fit in 8 bits.
pub static JPEG_DECODE_EXTENDED: &[FormatEntry] = &[
    // SrgbF32: f32 sRGB gamma, unclamped integer IDCT, preserves ringing
    FormatEntry::overshoot(
        PixelDescriptor::new_full(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
            ColorPrimaries::Bt709,
        ),
        8,
    ),
    // LinearF32: f32 linear light, unclamped integer IDCT + sRGB→linear
    FormatEntry::overshoot(PixelDescriptor::RGBF32_LINEAR, 8),
    // SrgbF32Precise: f32 sRGB with Laplacian dequantization biases (~10-bit effective)
    FormatEntry::overshoot(
        PixelDescriptor::new_full(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
            ColorPrimaries::Bt709,
        ),
        10,
    ),
    // LinearF32Precise: f32 linear with Laplacian biases (~10-bit effective)
    FormatEntry::overshoot(PixelDescriptor::RGBF32_LINEAR, 10),
    // Grayscale variants
    FormatEntry::overshoot(PixelDescriptor::GRAYF32_LINEAR, 8),
];

pub static JPEG: CodecFormats = CodecFormats {
    name: "jpeg",
    decode_outputs: JPEG_DECODE,
    encode_inputs: JPEG_ENCODE,
    icc_decode: true,
    icc_encode: true,
    cicp: false,
};

// ═══════════════════════════════════════════════════════════════════════
// PNG (zenpng)
// ═══════════════════════════════════════════════════════════════════════
//
// Full format support: u8, u16, f32. Sub-8-bit (1/2/4-bit) sources are
// scaled up to fill u8 range.
//
// u16 uses full 16 bits. f32 output is clamped to [0.0, 1.0], no overshoot.

static PNG_FORMATS: &[FormatEntry] = &[
    FormatEntry::standard(PixelDescriptor::RGB8_SRGB),
    FormatEntry::standard(PixelDescriptor::RGBA8_SRGB),
    FormatEntry::standard(PixelDescriptor::GRAY8_SRGB),
    FormatEntry::standard(PixelDescriptor::GRAYA8_SRGB),
    FormatEntry::standard(PixelDescriptor::BGRA8_SRGB),
    FormatEntry::standard(PixelDescriptor::RGB16_SRGB),
    FormatEntry::standard(PixelDescriptor::RGBA16_SRGB),
    FormatEntry::standard(PixelDescriptor::GRAY16_SRGB),
    FormatEntry::standard(PixelDescriptor::GRAYA16_SRGB),
    FormatEntry::standard(PixelDescriptor::RGBF32_LINEAR),
    FormatEntry::standard(PixelDescriptor::RGBAF32_LINEAR),
    FormatEntry::standard(PixelDescriptor::GRAYF32_LINEAR),
    FormatEntry::standard(PixelDescriptor::GRAYAF32_LINEAR),
];

pub static PNG: CodecFormats = CodecFormats {
    name: "png",
    decode_outputs: PNG_FORMATS,
    encode_inputs: PNG_FORMATS,
    icc_decode: true,
    icc_encode: true,
    cicp: true,
};

// ═══════════════════════════════════════════════════════════════════════
// GIF (zengif)
// ═══════════════════════════════════════════════════════════════════════
//
// Always 8-bit indexed palette. Decode composites frames to RGBA8 natively.
// f32 decode is a conversion from RGBA8. Encode requires quantization.
// No ICC or CICP (GIF spec doesn't support color management).

static GIF_DECODE: &[FormatEntry] = &[
    FormatEntry::with_bits(PixelDescriptor::RGBA8_SRGB, 8),
    FormatEntry::with_bits(PixelDescriptor::RGB8_SRGB, 8),
    FormatEntry::with_bits(PixelDescriptor::GRAY8_SRGB, 8),
    FormatEntry::with_bits(PixelDescriptor::BGRA8_SRGB, 8),
    // f32 outputs are converted from 8-bit source — effective precision is 8 bits
    FormatEntry::with_bits(PixelDescriptor::RGBF32_LINEAR, 8),
    FormatEntry::with_bits(PixelDescriptor::RGBAF32_LINEAR, 8),
    FormatEntry::with_bits(PixelDescriptor::GRAYF32_LINEAR, 8),
];

static GIF_ENCODE: &[FormatEntry] = &[
    FormatEntry::standard(PixelDescriptor::RGBA8_SRGB),
    FormatEntry::standard(PixelDescriptor::RGB8_SRGB),
    FormatEntry::standard(PixelDescriptor::GRAY8_SRGB),
    FormatEntry::standard(PixelDescriptor::BGRA8_SRGB),
];

pub static GIF: CodecFormats = CodecFormats {
    name: "gif",
    decode_outputs: GIF_DECODE,
    encode_inputs: GIF_ENCODE,
    icc_decode: false,
    icc_encode: false,
    cicp: false,
};

// ═══════════════════════════════════════════════════════════════════════
// WebP (zenwebp)
// ═══════════════════════════════════════════════════════════════════════
//
// Only RGB8/RGBA8 via zencodec API. The encoder and decoder internally
// handle BGRA8, GRAY8, and f32 linear via conversion, but these are not
// in supported_descriptors(). ICC roundtrip supported.

static WEBP_FORMATS: &[FormatEntry] = &[
    FormatEntry::standard(PixelDescriptor::RGB8_SRGB),
    FormatEntry::standard(PixelDescriptor::RGBA8_SRGB),
];

pub static WEBP: CodecFormats = CodecFormats {
    name: "webp",
    decode_outputs: WEBP_FORMATS,
    encode_inputs: WEBP_FORMATS,
    icc_decode: true,
    icc_encode: true,
    cicp: false,
};

// ═══════════════════════════════════════════════════════════════════════
// AVIF (zenavif)
// ═══════════════════════════════════════════════════════════════════════
//
// Decode: AV1 supports 8/10/12-bit, but zenavif always outputs u8 sRGB
// via zencodec API. 10/12-bit sources are scaled to u16 internally then
// converted to u8. CICP decoded from container or AV1 config.
//
// The effective bits of the u8 output are 8 regardless of AV1 source depth
// (precision is lost in the u8 conversion).

static AVIF_FORMATS: &[FormatEntry] = &[
    FormatEntry::with_bits(PixelDescriptor::RGB8_SRGB, 8),
    FormatEntry::with_bits(PixelDescriptor::RGBA8_SRGB, 8),
];

pub static AVIF: CodecFormats = CodecFormats {
    name: "avif",
    decode_outputs: AVIF_FORMATS,
    encode_inputs: AVIF_FORMATS,
    icc_decode: true,
    icc_encode: false,
    cicp: true,
};

// ═══════════════════════════════════════════════════════════════════════
// JPEG XL (zenjxl)
// ═══════════════════════════════════════════════════════════════════════
//
// JXL supports arbitrary bit depths natively. The zencodec integration
// decodes to u8 by default, with f32 linear for HDR content.
// f32 output is clamped to [0.0, 1.0] at encode time.
// Full ICC roundtrip, CICP via ICC metadata.

static JXL_FORMATS: &[FormatEntry] = &[
    FormatEntry::standard(PixelDescriptor::RGB8_SRGB),
    FormatEntry::standard(PixelDescriptor::RGBA8_SRGB),
    FormatEntry::standard(PixelDescriptor::GRAY8_SRGB),
    FormatEntry::standard(PixelDescriptor::GRAYA8_SRGB),
    FormatEntry::standard(PixelDescriptor::BGRA8_SRGB),
    FormatEntry::standard(PixelDescriptor::RGBF32_LINEAR),
    FormatEntry::standard(PixelDescriptor::RGBAF32_LINEAR),
    FormatEntry::standard(PixelDescriptor::GRAYF32_LINEAR),
    FormatEntry::standard(PixelDescriptor::GRAYAF32_LINEAR),
];

pub static JXL: CodecFormats = CodecFormats {
    name: "jxl",
    decode_outputs: JXL_FORMATS,
    encode_inputs: JXL_FORMATS,
    icc_decode: true,
    icc_encode: true,
    cicp: true,
};

// ═══════════════════════════════════════════════════════════════════════
// BMP (zenbitmaps)
// ═══════════════════════════════════════════════════════════════════════
//
// Native format is BGR/BGRA. Supports 1-32 bit input but always outputs
// u8 via zencodec. No color management.

static BMP_FORMATS: &[FormatEntry] = &[
    FormatEntry::standard(PixelDescriptor::RGB8_SRGB),
    FormatEntry::standard(PixelDescriptor::RGBA8_SRGB),
    FormatEntry::standard(PixelDescriptor::BGRA8_SRGB),
];

pub static BMP: CodecFormats = CodecFormats {
    name: "bmp",
    decode_outputs: BMP_FORMATS,
    encode_inputs: BMP_FORMATS,
    icc_decode: false,
    icc_encode: false,
    cicp: false,
};

// ═══════════════════════════════════════════════════════════════════════
// Farbfeld (zenbitmaps)
// ═══════════════════════════════════════════════════════════════════════
//
// Always RGBA u16 big-endian on disk. Lossless format, 16 effective bits.
// Can output u8 and grayscale via conversion.

static FARBFELD_FORMATS: &[FormatEntry] = &[
    FormatEntry::standard(PixelDescriptor::RGBA16_SRGB),
    FormatEntry::with_bits(PixelDescriptor::RGBA8_SRGB, 8),
    FormatEntry::with_bits(PixelDescriptor::RGB8_SRGB, 8),
    FormatEntry::with_bits(PixelDescriptor::GRAY8_SRGB, 8),
];

pub static FARBFELD: CodecFormats = CodecFormats {
    name: "farbfeld",
    decode_outputs: FARBFELD_FORMATS,
    encode_inputs: FARBFELD_FORMATS,
    icc_decode: false,
    icc_encode: false,
    cicp: false,
};

// ═══════════════════════════════════════════════════════════════════════
// PNM (zenbitmaps)
// ═══════════════════════════════════════════════════════════════════════
//
// Covers PGM (P5), PPM (P6), PAM (P7), and PFM.
// Variable bit depth: P5/P6/P7 support maxval 1–65535.
// PFM is f32 [0.0, 1.0]. No color management.

static PNM_FORMATS: &[FormatEntry] = &[
    FormatEntry::standard(PixelDescriptor::RGB8_SRGB),
    FormatEntry::standard(PixelDescriptor::RGBA8_SRGB),
    FormatEntry::standard(PixelDescriptor::RGBA16_SRGB),
    FormatEntry::standard(PixelDescriptor::GRAY8_SRGB),
    FormatEntry::standard(PixelDescriptor::GRAYA8_SRGB),
    FormatEntry::standard(PixelDescriptor::BGRA8_SRGB),
    FormatEntry::standard(PixelDescriptor::RGBF32_LINEAR),
    FormatEntry::standard(PixelDescriptor::RGBAF32_LINEAR),
    FormatEntry::standard(PixelDescriptor::GRAYF32_LINEAR),
    FormatEntry::standard(PixelDescriptor::GRAYAF32_LINEAR),
];

pub static PNM: CodecFormats = CodecFormats {
    name: "pnm",
    decode_outputs: PNM_FORMATS,
    encode_inputs: PNM_FORMATS,
    icc_decode: false,
    icc_encode: false,
    cicp: false,
};

// ═══════════════════════════════════════════════════════════════════════
// All codecs
// ═══════════════════════════════════════════════════════════════════════

/// All registered codecs.
pub static ALL_CODECS: &[&CodecFormats] =
    &[&JPEG, &PNG, &GIF, &WEBP, &AVIF, &JXL, &BMP, &FARBFELD, &PNM];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_codecs_have_decode_and_encode() {
        for codec in ALL_CODECS {
            assert!(
                !codec.decode_outputs.is_empty(),
                "{} has no decode outputs",
                codec.name
            );
            assert!(
                !codec.encode_inputs.is_empty(),
                "{} has no encode inputs",
                codec.name
            );
        }
    }

    #[test]
    fn effective_bits_within_container() {
        for codec in ALL_CODECS {
            for entry in codec
                .decode_outputs
                .iter()
                .chain(codec.encode_inputs.iter())
            {
                let container_bits = match entry.descriptor.channel_type() {
                    ChannelType::U8 => 8,
                    ChannelType::U16 => 16,
                    ChannelType::F16 => 16,
                    ChannelType::F32 => 32,
                    _ => 0,
                };
                assert!(
                    entry.effective_bits <= container_bits,
                    "{}: effective_bits {} > container {} for {:?}",
                    codec.name,
                    entry.effective_bits,
                    container_bits,
                    entry.descriptor
                );
            }
        }
    }

    #[test]
    fn jpeg_extended_has_overshoot() {
        for entry in JPEG_DECODE_EXTENDED {
            assert!(
                entry.can_overshoot,
                "JPEG extended f32 decode should have overshoot"
            );
        }
    }

    #[test]
    fn codec_count() {
        assert_eq!(ALL_CODECS.len(), 9);
    }
}
