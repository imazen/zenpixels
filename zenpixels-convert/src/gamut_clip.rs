//! Hue- and lightness-preserving gamut mapping (Oklab snap-to-boundary).
//!
//! When a wide-gamut image is narrowed into a smaller destination
//! ([`RenderingIntent::RelativeColorimetric`](crate::cms::RenderingIntent)) and
//! encoded to 8-bit, the default is a per-channel hard clamp to `[0, 1]`. For
//! saturated out-of-gamut content — e.g. a Display P3 red poppy mapped to sRGB
//! — that flattens the over-the-line pixels to a solid primary and throws away
//! the lightness gradation (the petal shading), even though the detail was
//! present in the source.
//!
//! This module is the detail-preserving alternative, expressed as a small
//! [`GamutMapper`] **trait** so the same step plugs in regardless of which color
//! engine produced the destination RGB. Whether the conversion came from the
//! built-in fused gamut matrix (the [`ZenCmsLite`](crate::cms_lite) path that
//! handles P3↔sRGB in-house) or from an extended-range CMS (moxcms) transform,
//! both converge on **extended-range linear RGB in the destination primaries**,
//! and the mapper acts there — before the transfer-function encode quantizes to
//! the output type.
//!
//! Two built-in strategies ship:
//!
//! * [`PerChannelClip`] — the current behavior: independent per-channel clamp.
//!   Cheap, but distorts hue and crushes lightness for out-of-gamut colors.
//! * [`OklabSnap`] — **detection-based**: a pixel already inside `[0, 1]^3` is
//!   returned unchanged (no pointless desaturation of in-gamut colors); an
//!   out-of-gamut pixel keeps its Oklab lightness and hue while only its chroma
//!   is snapped to the destination gamut boundary (the most saturated
//!   representable color of that hue and lightness).

use alloc::boxed::Box;

use crate::gamut::GamutMatrix;
use crate::oklab::{lms_to_rgb_matrix, oklab_to_rgb, rgb_to_lms_matrix, rgb_to_oklab};
use crate::{ColorPrimaries, GamutClip};

/// In-gamut tolerance, ≈ half an 8-bit code value in the linear `[0, 1]` range.
/// A pixel within this of the cube is treated as representable (the final
/// clamp removes the sub-LSB overshoot).
const GAMUT_EPS: f32 = 1.0 / 512.0;

#[inline(always)]
fn in_gamut(rgb: &[f32; 3]) -> bool {
    rgb[0] >= -GAMUT_EPS
        && rgb[0] <= 1.0 + GAMUT_EPS
        && rgb[1] >= -GAMUT_EPS
        && rgb[1] <= 1.0 + GAMUT_EPS
        && rgb[2] >= -GAMUT_EPS
        && rgb[2] <= 1.0 + GAMUT_EPS
}

#[inline(always)]
fn clamp01(rgb: &mut [f32; 3]) {
    rgb[0] = rgb[0].clamp(0.0, 1.0);
    rgb[1] = rgb[1].clamp(0.0, 1.0);
    rgb[2] = rgb[2].clamp(0.0, 1.0);
}

/// A gamut-mapping strategy: bring extended-range linear RGB (already in the
/// destination primaries) into the destination's `[0, 1]^3` gamut.
///
/// Implementors operate in place on one pixel at a time via [`map_rgb`]; a
/// provided [`map_rgb_row`] applies that across a contiguous RGB row and may be
/// overridden with a vectorized version. The trait is the extension seam: a
/// caller can supply a custom mapper, and the conversion pipeline applies it at
/// the destination-linear seam shared by every color backend.
///
/// [`map_rgb`]: GamutMapper::map_rgb
/// [`map_rgb_row`]: GamutMapper::map_rgb_row
pub trait GamutMapper: Send + Sync {
    /// Map one extended-range linear destination-primaries RGB triple into
    /// `[0, 1]^3`, in place. The result must be within the unit cube.
    fn map_rgb(&self, rgb: &mut [f32; 3]);

    /// Apply [`map_rgb`](GamutMapper::map_rgb) across a contiguous interleaved
    /// RGB row (`len` a multiple of 3). The default loops per pixel; override
    /// for a SIMD path. A trailing partial group (malformed input) is ignored.
    fn map_rgb_row(&self, rgb: &mut [f32]) {
        for px in rgb.chunks_exact_mut(3) {
            let mut t = [px[0], px[1], px[2]];
            self.map_rgb(&mut t);
            px.copy_from_slice(&t);
        }
    }
}

/// Independent per-channel clamp to `[0, 1]` — the existing default behavior.
///
/// Fast and exact for in-gamut pixels, but for out-of-gamut colors it shifts
/// hue and crushes lightness (a P3 red flattens to flat sRGB red).
#[derive(Clone, Copy, Debug, Default)]
pub struct PerChannelClip;

impl GamutMapper for PerChannelClip {
    #[inline]
    fn map_rgb(&self, rgb: &mut [f32; 3]) {
        clamp01(rgb);
    }
}

/// Oklab snap-to-boundary: preserve lightness and hue, clip only chroma.
///
/// Holds the precomputed Oklab LMS matrices for one destination primaries set,
/// so the per-pixel hot path is the in-gamut test plus, for the minority of
/// out-of-gamut pixels, an Oklab round-trip and a short chroma bisection.
#[derive(Clone, Copy, Debug)]
pub struct OklabSnap {
    rgb_to_lms: GamutMatrix,
    lms_to_rgb: GamutMatrix,
}

impl OklabSnap {
    /// Build a snapper for the destination `primaries`. Returns `None` for
    /// primaries without a defined RGB↔XYZ matrix (e.g. `Unspecified`).
    #[must_use]
    pub fn new(primaries: ColorPrimaries) -> Option<Self> {
        Some(Self {
            rgb_to_lms: rgb_to_lms_matrix(primaries)?,
            lms_to_rgb: lms_to_rgb_matrix(primaries)?,
        })
    }
}

impl GamutMapper for OklabSnap {
    /// Snap one extended-range linear destination-primaries RGB triple into the
    /// destination gamut, preserving Oklab lightness and hue while clipping only
    /// chroma. In-gamut pixels are left unchanged. The result is within
    /// `[0, 1]^3`.
    #[inline]
    fn map_rgb(&self, rgb: &mut [f32; 3]) {
        if in_gamut(rgb) {
            clamp01(rgb);
            return;
        }
        let [l, a, b] = rgb_to_oklab(rgb[0], rgb[1], rgb[2], &self.rgb_to_lms);
        let c = (a * a + b * b).sqrt();
        if c < 1e-6 {
            // Achromatic but out of range (L below 0 or above 1): there is no
            // chroma to reduce, so the channel clamp is the right answer.
            clamp01(rgb);
            return;
        }
        let inv_c = 1.0 / c;
        let (ha, hb) = (a * inv_c, b * inv_c);
        // Bisect the largest chroma whose (L, hue) is still in gamut. The
        // boundary is monotone in chroma at fixed L+hue, so 18 steps resolve it
        // to ~`c / 2^18` (far below an 8-bit code value for any real chroma).
        let mut lo = 0.0f32;
        let mut hi = c;
        for _ in 0..18 {
            let mid = 0.5 * (lo + hi);
            let t = oklab_to_rgb(l, ha * mid, hb * mid, &self.lms_to_rgb);
            if in_gamut(&t) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let mut out = oklab_to_rgb(l, ha * lo, hb * lo, &self.lms_to_rgb);
        clamp01(&mut out);
        *rgb = out;
    }
}

/// Resolve a [`GamutClip`] policy into a boxed [`GamutMapper`] for the
/// destination `primaries`.
///
/// Returns `None` when no extra mapping step is needed — either the policy is
/// [`GamutClip::PerChannel`] (the transfer-function encode already clamps), or
/// [`GamutClip::Preserve`] was requested for primaries without a defined matrix
/// (falls back to the encode's per-channel clamp).
///
/// The [`GamutClip`] enum itself lives in `zenpixels` so it can be a field of
/// `ConvertOptions`; this resolver lives here because the snap needs the Oklab
/// matrices defined in this crate.
#[must_use]
pub fn mapper_for(clip: GamutClip, primaries: ColorPrimaries) -> Option<Box<dyn GamutMapper>> {
    match clip {
        GamutClip::PerChannel => None,
        GamutClip::Preserve => {
            OklabSnap::new(primaries).map(|s| Box::new(s) as Box<dyn GamutMapper>)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oklab::rgb_to_oklab;

    fn srgb() -> OklabSnap {
        OklabSnap::new(ColorPrimaries::Bt709).unwrap()
    }

    /// An in-gamut pixel must be returned bit-for-bit unchanged (no
    /// desaturation of colors that already fit).
    #[test]
    fn in_gamut_pixel_untouched() {
        let clip = srgb();
        for px in [
            [0.5, 0.2, 0.1],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.8, 0.05, 0.4],
        ] {
            let mut rgb = px;
            clip.map_rgb(&mut rgb);
            assert_eq!(rgb, px, "in-gamut pixel changed");
        }
    }

    /// A Display P3 red mapped to sRGB-linear is out of gamut (R ≈ 1.22,
    /// G/B slightly negative). Snap must bring it into [0,1] while keeping the
    /// Oklab lightness and hue, and only reducing chroma.
    #[test]
    fn out_of_gamut_red_preserves_lightness_and_hue() {
        let clip = srgb();
        // Display P3 (1,0,0) expressed in sRGB linear primaries.
        let src = [1.224_94, -0.042_06, -0.019_64];
        let m1 = rgb_to_lms_matrix(ColorPrimaries::Bt709).unwrap();
        let [l0, a0, b0] = rgb_to_oklab(src[0], src[1], src[2], &m1);
        let c0 = (a0 * a0 + b0 * b0).sqrt();

        let mut rgb = src;
        clip.map_rgb(&mut rgb);

        // now representable
        assert!(
            rgb.iter().all(|&v| (0.0..=1.0).contains(&v)),
            "not in gamut: {rgb:?}"
        );
        // it WAS out of gamut, so it must have moved
        assert!(rgb != src);

        let [l1, a1, b1] = rgb_to_oklab(rgb[0], rgb[1], rgb[2], &m1);
        let c1 = (a1 * a1 + b1 * b1).sqrt();
        // lightness preserved
        assert!((l1 - l0).abs() < 0.01, "lightness shifted: {l0} -> {l1}");
        // hue preserved (angle of (a,b))
        let h0 = b0.atan2(a0);
        let h1 = b1.atan2(a1);
        let mut dh = (h1 - h0).abs();
        if dh > core::f32::consts::PI {
            dh = 2.0 * core::f32::consts::PI - dh;
        }
        assert!(dh < 0.02, "hue shifted: {h0} -> {h1}");
        // chroma reduced to (just inside) the boundary
        assert!(c1 < c0, "chroma not reduced: {c0} -> {c1}");
    }

    /// Snap must beat a per-channel hard clamp on lightness fidelity for an
    /// out-of-gamut color: the hard clamp distorts lightness, snap keeps it.
    #[test]
    fn snap_keeps_more_lightness_than_hard_clip() {
        let snap = srgb();
        let m1 = rgb_to_lms_matrix(ColorPrimaries::Bt709).unwrap();
        let src = [1.30, 0.10, -0.05]; // bright out-of-gamut red-orange
        let l_src = rgb_to_oklab(src[0], src[1], src[2], &m1)[0];

        let mut hard = src;
        PerChannelClip.map_rgb(&mut hard);
        let l_hard = rgb_to_oklab(hard[0], hard[1], hard[2], &m1)[0];

        let mut soft = src;
        snap.map_rgb(&mut soft);
        let l_soft = rgb_to_oklab(soft[0], soft[1], soft[2], &m1)[0];

        assert!(
            (l_soft - l_src).abs() < (l_hard - l_src).abs(),
            "snap should preserve lightness better than hard clip: \
             src={l_src} hard={l_hard} soft={l_soft}"
        );
    }

    /// The row helper applies the per-pixel map across an interleaved RGB row.
    #[test]
    fn row_helper_matches_per_pixel() {
        let snap = srgb();
        let mut row = [1.224_94f32, -0.042_06, -0.019_64, 0.5, 0.2, 0.1];
        snap.map_rgb_row(&mut row);
        let mut p0 = [1.224_94f32, -0.042_06, -0.019_64];
        snap.map_rgb(&mut p0);
        assert_eq!(&row[0..3], &p0);
        assert_eq!(&row[3..6], &[0.5, 0.2, 0.1]); // in-gamut pixel untouched
    }

    /// The `GamutClip` policy resolves to the expected mapper presence.
    #[test]
    fn gamut_clip_enum_resolves() {
        assert!(mapper_for(GamutClip::PerChannel, ColorPrimaries::Bt709).is_none());
        assert!(mapper_for(GamutClip::Preserve, ColorPrimaries::Bt709).is_some());
        assert_eq!(GamutClip::default(), GamutClip::PerChannel);
    }
}
