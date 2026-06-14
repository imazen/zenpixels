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

/// Round non-negative nits to a CTA-861.3 `u16` code (saturating).
///
/// `nits` is a luminance — always `≥ 0` at the call sites. Round-half-up is
/// then `(nits + 0.5)` truncated, and the float→int `as` cast saturates to
/// `[0, u16::MAX]` (mapping negatives and NaN to 0). Done by hand because
/// `f64::round` lives in `std` (libm) and this crate builds `no_std`.
#[inline]
fn nits_to_u16(nits: f64) -> u16 {
    (nits + 0.5) as u16
}

/// Reduce one row of `N`-channel f32 pixels to
/// `(max, sum)` of the per-pixel `max(R, G, B)`.
///
/// `N` is the channel count (3 = `Rgb`, 4 = `Rgba`); only the first three
/// lanes are read, so any alpha is ignored. Each channel is folded from `0.0`,
/// so `f32::max`'s non-NaN-propagating semantics drop NaN and negative samples.
/// `chunk` is reborrowed as a fixed-size `&[f32; N]` so the bounds checks fall
/// away and LLVM can vectorize the reduction. The sum accumulates in `f64`:
/// a 4K frame is ~8M pixels, beyond f32's precision for a running total.
#[inline]
fn row_max_sum<const N: usize>(row: &[f32]) -> (f32, f64) {
    let mut row_max = 0.0f32;
    let mut row_sum = 0.0f64;
    for chunk in row.chunks_exact(N) {
        // `chunks_exact(N)` yields exactly-`N` slices — the conversion is infallible.
        let px: &[f32; N] = chunk.try_into().unwrap();
        let m = 0.0f32.max(px[0]).max(px[1]).max(px[2]);
        row_max = row_max.max(m);
        row_sum += f64::from(m);
    }
    (row_max, row_sum)
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
    /// `RgbF32`/`RgbaF32`. This is deliberate, not a missing case: cd/m² is
    /// only defined in **linear light**, so a transfer function would have to
    /// be inverted first — and inverting one (PQ/HLG/sRGB → linear) is the
    /// conversion pipeline's job, which the foundational `zenpixels` crate has
    /// no dependency on. To measure an integer or non-linear HDR buffer,
    /// linearize it to `RgbaF32` first (`zenpixels_convert::convert_buffer`),
    /// then call this. Zero-area input yields `Some(0, 0)`.
    ///
    /// `RgbF32` and `RgbaF32` share one reduction (generic over the channel
    /// count); the inner loop reads whole f32s from the channel-aligned buffer
    /// so it vectorizes rather than decoding sample-by-sample.
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

        // Reduce in relative-linear units, then scale by the anchor once at the
        // end — ∑(mᵢ·w) = (∑mᵢ)·w, fewer multiplies for the same f64 result.
        let mut max_lin = 0.0f32;
        let mut sum_lin = 0.0f64;
        for row in 0..h {
            let row_bytes = &bytes[row * stride..row * stride + row_len];
            // f32 buffers are channel-aligned (the `PixelBuffer` alignment
            // invariant), and `row_len` is a multiple of 4, so this cast never
            // straddles a sample — and reading whole f32s lets the reduction
            // vectorize, unlike per-byte `from_ne_bytes`.
            let floats: &[f32] = bytemuck::cast_slice(row_bytes);
            let (row_max, row_sum) = if channels == 3 {
                row_max_sum::<3>(floats)
            } else {
                row_max_sum::<4>(floats)
            };
            max_lin = max_lin.max(row_max);
            sum_lin += row_sum;
        }
        let wn = f64::from(white.nits());
        let max_nits = f64::from(max_lin) * wn;
        let fall = sum_lin / (w as f64 * h as f64) * wn;
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
    fn measure_handles_stride_and_ignores_padding() {
        use crate::PixelSlice;
        // 2×2 RGB f32: 6 real f32/row, padded to 9 f32/row (36-byte stride, a
        // multiple of the 12-byte pixel). The padding holds a 1e9 sentinel — if
        // a row cast ever ran past `width*bpp`, MaxCLL would explode to ~2e11.
        let (w, h, row_floats) = (2u32, 2u32, 9usize);
        let mut data = alloc::vec![1.0e9f32; row_floats * h as usize];
        let pixels = [[0.5f32; 3], [1.0; 3], [2.0; 3], [0.25; 3]];
        for (i, p) in pixels.iter().enumerate() {
            let base = (i / w as usize) * row_floats + (i % w as usize) * 3;
            data[base..base + 3].copy_from_slice(p);
        }
        // `Vec<f32>` is f32-aligned, so the byte view satisfies the slice's
        // alignment contract; stride 40 is a multiple of the f32 size.
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        let px =
            PixelSlice::new(bytes, w, h, row_floats * 4, PixelDescriptor::RGBF32_LINEAR).unwrap();
        let cll = ContentLightLevel::measure(px, DiffuseWhite::BT2408).unwrap();
        // Peak max(R,G,B) = 2.0 → 406; FALL = avg(0.5,1,2,0.25)·203 = 190.3 → 190.
        assert_eq!(cll.max_content_light_level, 406);
        assert_eq!(cll.max_frame_average_light_level, 190);
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
