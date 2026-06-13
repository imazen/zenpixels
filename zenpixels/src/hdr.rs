//! HDR metadata types.
//!
//! Pure data types for HDR content description. These travel with pixel
//! data alongside [`Cicp`](crate::Cicp) and [`ColorContext`](crate::ColorContext).
//!
//! For tone mapping and HDR processing functions, see
//! [`zenpixels-convert::hdr`](https://docs.rs/zenpixels-convert/latest/zenpixels_convert/hdr/).

use crate::{PixelFormat, PixelSlice, TransferFunction};

/// The absolute luminance, in cd/m² (nits), that a relative-linear sample
/// value of `1.0` represents — the "diffuse white" (a.k.a. nominal diffuse
/// white / SDR reference white) anchor that bridges relative-linear pixel
/// data to absolute display light.
///
/// This is the single scalar the rest of the industry uses for that bridge:
/// OpenEXR's `whiteLuminance` ("nits of RGB (1,1,1)"), JPEG XL's
/// `intensity_target`, libheif's `ndwt` (nominal diffuse white), and
/// libplacebo's SDR-white constant. The cross-vendor default is
/// [`BT2408`](Self::BT2408) = 203 cd/m².
///
/// It is a *typed* anchor on purpose: HDR code mixes nits, PQ-encoded `[0,1]`,
/// log2 gain, and headroom ratios — passing a bare `f32` invites unit
/// confusion. Use [`DiffuseWhite::new`] / [`DiffuseWhite::nits`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DiffuseWhite(f32);

impl DiffuseWhite {
    /// ITU-R BT.2408 HDR reference white: **203 cd/m²**. The cross-industry
    /// default anchor for relative-linear HDR (matches Chrome `SDRWhiteLevel`,
    /// Skia skcms, CSS `rec2100-linear`, and libplacebo).
    pub const BT2408: Self = Self(203.0);

    /// An anchor of `nits` cd/m² (the luminance that relative-linear `1.0`
    /// represents).
    #[must_use]
    pub const fn new(nits: f32) -> Self {
        Self(nits)
    }

    /// The anchor in cd/m² (nits).
    #[must_use]
    pub const fn nits(self) -> f32 {
        self.0
    }
}

impl Default for DiffuseWhite {
    /// [`BT2408`](Self::BT2408) — 203 cd/m².
    fn default() -> Self {
        Self::BT2408
    }
}

/// Round nits to a CTA-861.3 `u16` code (saturating).
#[inline]
fn nits_to_u16(nits: f64) -> u16 {
    nits.round().clamp(0.0, 65535.0) as u16
}

/// Read one native-endian f32 sample from a pixel's bytes (no alignment
/// assumption, per the `PixelSlice` endianness contract).
#[inline]
fn sample_f32(bytes: &[u8], k: usize) -> f32 {
    f32::from_ne_bytes([
        bytes[4 * k],
        bytes[4 * k + 1],
        bytes[4 * k + 2],
        bytes[4 * k + 3],
    ])
}

/// HDR content light level metadata (CEA-861.3 / CTA-861-H).
///
/// Describes the peak brightness characteristics of HDR content.
/// Used by AVIF, JXL, PNG (cLLi chunk), and video containers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContentLightLevel {
    /// Maximum Content Light Level (MaxCLL) in cd/m² (nits).
    /// Peak luminance of any single pixel in the content.
    pub max_content_light_level: u16,
    /// Maximum Frame-Average Light Level (MaxFALL) in cd/m².
    /// Peak average luminance of any single frame.
    pub max_frame_average_light_level: u16,
}

impl ContentLightLevel {
    /// Create content light level metadata.
    pub const fn new(max_content_light_level: u16, max_frame_average_light_level: u16) -> Self {
        Self {
            max_content_light_level,
            max_frame_average_light_level,
        }
    }

    /// Measure MaxCLL / MaxFALL (CTA-861.3-A) from relative-linear RGB(A) f32
    /// pixels, with `white` anchoring the scale (sample `1.0` = `white` nits;
    /// [`DiffuseWhite::BT2408`] — 203 — is the convention).
    ///
    /// Semantics per CTA-861.3-A as PNG 3rd ed §11.3.2.8 imports it for stills
    /// (one still = one frame): **MaxCLL** is the brightest pixel's
    /// `max(R, G, B)` in cd/m², **MaxFALL** is the image's average of per-pixel
    /// `max(R, G, B)`. Negative/NaN samples clamp to 0; an alpha lane is
    /// ignored; strided rows are handled.
    ///
    /// Returns `None` if the descriptor is not relative-linear
    /// `RgbF32`/`RgbaF32` (the caller, having produced the linear buffer,
    /// knows its format). Zero-area input yields `Some(0, 0)`.
    #[must_use]
    pub fn measure(px: PixelSlice<'_>, white: DiffuseWhite) -> Option<Self> {
        let desc = px.descriptor();
        let channels = match desc.pixel_format() {
            PixelFormat::RgbF32 => 3,
            PixelFormat::RgbaF32 => 4,
            _ => return None,
        };
        if desc.transfer != TransferFunction::Linear {
            return None;
        }
        let w = px.width() as usize;
        let h = px.rows() as usize;
        if w == 0 || h == 0 {
            return Some(Self::new(0, 0));
        }
        let stride = px.stride();
        let bytes = px.as_strided_bytes();
        let row_len = w * channels * 4;
        let wn = f64::from(white.nits());

        let mut max_nits = 0.0f64;
        let mut sum_max_nits = 0.0f64;
        for row in 0..h {
            let row_bytes = &bytes[row * stride..row * stride + row_len];
            for pxl in row_bytes.chunks_exact(channels * 4) {
                // Fold from 0.0 so NaN and negative samples drop out.
                let m = 0.0f32
                    .max(sample_f32(pxl, 0))
                    .max(sample_f32(pxl, 1))
                    .max(sample_f32(pxl, 2));
                let nits = f64::from(m) * wn;
                max_nits = max_nits.max(nits);
                sum_max_nits += nits;
            }
        }
        let fall = sum_max_nits / (w as f64 * h as f64);
        Some(Self::new(nits_to_u16(max_nits), nits_to_u16(fall)))
    }
}

/// Mastering display color volume metadata (SMPTE ST 2086).
///
/// Describes the display on which the content was mastered, enabling
/// downstream displays to reproduce the creator's intent.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MasteringDisplay {
    /// RGB primaries of the mastering display in CIE 1931 xy coordinates.
    /// `[[rx, ry], [gx, gy], [bx, by]]`.
    pub primaries_xy: [[f32; 2]; 3],
    /// White point in CIE 1931 xy coordinates `[wx, wy]`.
    pub white_point_xy: [f32; 2],
    /// Maximum display luminance in cd/m².
    pub max_luminance: f32,
    /// Minimum display luminance in cd/m².
    pub min_luminance: f32,
}

impl MasteringDisplay {
    /// Create mastering display metadata from CIE 1931 xy coordinates and cd/m² luminances.
    pub const fn new(
        primaries_xy: [[f32; 2]; 3],
        white_point_xy: [f32; 2],
        max_luminance: f32,
        min_luminance: f32,
    ) -> Self {
        Self {
            primaries_xy,
            white_point_xy,
            max_luminance,
            min_luminance,
        }
    }

    /// BT.2020 primaries with D65 white point, 10000 nits peak (HDR10 reference).
    pub const HDR10_REFERENCE: Self = Self {
        primaries_xy: [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
        white_point_xy: [0.3127, 0.3290],
        max_luminance: 10000.0,
        min_luminance: 0.0001,
    };

    /// Display P3 primaries with D65 white point, 1000 nits.
    pub const DISPLAY_P3_1000: Self = Self {
        primaries_xy: [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]],
        white_point_xy: [0.3127, 0.3290],
        max_luminance: 1000.0,
        min_luminance: 0.0001,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PixelBuffer, PixelDescriptor};
    use alloc::vec::Vec;

    fn rgbf32(pixels: &[[f32; 3]], w: u32, h: u32) -> PixelBuffer {
        let mut data = Vec::with_capacity(pixels.len() * 12);
        for p in pixels {
            for c in p {
                data.extend_from_slice(&c.to_ne_bytes());
            }
        }
        PixelBuffer::from_vec(data, w, h, PixelDescriptor::RGBF32_LINEAR).unwrap()
    }

    #[test]
    fn diffuse_white_defaults_to_bt2408() {
        assert_eq!(DiffuseWhite::default(), DiffuseWhite::BT2408);
        assert_eq!(DiffuseWhite::BT2408.nits(), 203.0);
        assert_eq!(DiffuseWhite::new(100.0).nits(), 100.0);
    }

    #[test]
    fn measure_two_grays_cta_stills_semantics() {
        // [1.0, 2.0] @ 203: MaxCLL = 2·203 = 406; MaxFALL = avg(203, 406) = 304.5 → 305.
        let buf = rgbf32(&[[1.0; 3], [2.0; 3]], 2, 1);
        let cll = ContentLightLevel::measure(buf.as_slice(), DiffuseWhite::BT2408).unwrap();
        assert_eq!(cll.max_content_light_level, 406);
        assert_eq!(cll.max_frame_average_light_level, 305);
    }

    #[test]
    fn measure_clamps_nan_and_negative() {
        let buf = rgbf32(&[[-1.0, f32::NAN, 0.5]], 1, 1);
        let cll = ContentLightLevel::measure(buf.as_slice(), DiffuseWhite::BT2408).unwrap();
        // max(R,G,B) folds from 0.0 → 0.5 · 203 = 101.5 → 102.
        assert_eq!(cll.max_content_light_level, 102);
        assert_eq!(cll.max_frame_average_light_level, 102);
    }

    #[test]
    fn measure_ignores_alpha_and_custom_white() {
        let mut data = Vec::new();
        for c in [0.5f32, 0.5, 0.5, 7.0] {
            data.extend_from_slice(&c.to_ne_bytes());
        }
        let buf = PixelBuffer::from_vec(data, 1, 1, PixelDescriptor::RGBAF32_LINEAR).unwrap();
        // alpha 7.0 ignored; custom 100-nit white: 0.5 · 100 = 50.
        let cll = ContentLightLevel::measure(buf.as_slice(), DiffuseWhite::new(100.0)).unwrap();
        assert_eq!(cll.max_content_light_level, 50);
    }

    #[test]
    fn measure_rejects_non_linear_and_non_f32() {
        let u8buf =
            PixelBuffer::from_vec(alloc::vec![0u8; 3], 1, 1, PixelDescriptor::RGB8_SRGB).unwrap();
        assert!(ContentLightLevel::measure(u8buf.as_slice(), DiffuseWhite::BT2408).is_none());

        let nonlinear = PixelDescriptor::RGBF32_LINEAR.with_transfer(TransferFunction::Srgb);
        let mut data = Vec::new();
        for c in [0.5f32; 3] {
            data.extend_from_slice(&c.to_ne_bytes());
        }
        let buf = PixelBuffer::from_vec(data, 1, 1, nonlinear).unwrap();
        assert!(ContentLightLevel::measure(buf.as_slice(), DiffuseWhite::BT2408).is_none());
    }

    #[test]
    fn content_light_level_new() {
        let cll = ContentLightLevel::new(1000, 500);
        assert_eq!(cll.max_content_light_level, 1000);
        assert_eq!(cll.max_frame_average_light_level, 500);
    }

    #[test]
    fn content_light_level_default() {
        let cll = ContentLightLevel::default();
        assert_eq!(cll.max_content_light_level, 0);
        assert_eq!(cll.max_frame_average_light_level, 0);
    }

    #[test]
    fn mastering_display_new() {
        let md = MasteringDisplay::new(
            [[0.68, 0.32], [0.265, 0.69], [0.15, 0.06]],
            [0.3127, 0.329],
            1000.0,
            0.001,
        );
        assert_eq!(md.max_luminance, 1000.0);
        assert_eq!(md.min_luminance, 0.001);
    }

    #[test]
    fn mastering_display_constants() {
        assert_eq!(MasteringDisplay::HDR10_REFERENCE.max_luminance, 10000.0);
        assert_eq!(MasteringDisplay::DISPLAY_P3_1000.max_luminance, 1000.0);
    }

    #[test]
    fn content_light_level_clone_eq() {
        let a = ContentLightLevel::new(100, 50);
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    #[cfg(feature = "std")]
    fn content_light_level_hash() {
        use core::hash::{Hash, Hasher};
        let a = ContentLightLevel::new(100, 50);
        let b = a;
        let mut h1 = std::hash::DefaultHasher::new();
        a.hash(&mut h1);
        let mut h2 = std::hash::DefaultHasher::new();
        b.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }
}
