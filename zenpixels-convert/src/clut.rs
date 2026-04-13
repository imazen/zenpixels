//! 3D Color Lookup Table with tetrahedral interpolation.
//!
//! Precomputes the entire u8→u8 gamut conversion (linearize → matrix → encode)
//! into a 33³×3 grid. Per-pixel cost is 5 table lookups + 4 lerps — no TRC
//! math, no matrix math.
//!
//! # Usage
//!
//! Built by [`Clut3d::build`] from a gamut matrix and TRC pair. Used
//! internally by [`ZenCmsLite`](super::cms_lite::ZenCmsLite) for u8 RGB
//! when a CLUT would be faster than the fused SIMD kernel.

use alloc::boxed::Box;
use alloc::vec;

/// Grid size for the 3D CLUT. 33 points = 32 intervals per axis.
/// 33³×3×4 = 431KB. Fits comfortably in L2 cache.
const GRID: usize = 33;

/// 3D Color Lookup Table for u8→u8 gamut conversion.
///
/// Stores `GRID³ × 3` f32 values in **linear light** after the gamut matrix.
/// Tetrahedral interpolation happens in linear space (where the function
/// is smooth), then the result is encoded to the destination TRC.
pub(crate) struct Clut3d {
    /// Flattened grid of linear-light values after matrix multiply.
    data: Box<[f32]>,
    /// TRC encode function applied after interpolation.
    encode: fn(f32) -> f32,
}

impl Clut3d {
    /// Build a CLUT from a linearize LUT, gamut matrix, and encode function.
    ///
    /// Evaluates `linearize → matrix → encode` at each of the 33³ grid points.
    /// Takes ~1ms on a modern CPU.
    /// Build a CLUT that stores **encoded** output values.
    ///
    /// Grid points sample the input u8 range at uniform f32 intervals.
    /// Each grid point evaluates linearize → matrix → encode at the exact
    /// f32 input value (not rounded to integer u8). This matches the
    /// interpolation scaling in `convert_pixel`.
    pub(crate) fn build(
        linearize: fn(f32) -> f32,
        matrix: &[[f32; 3]; 3],
        encode: fn(f32) -> f32,
    ) -> Self {
        let total = GRID * GRID * GRID * 3;
        let mut data = vec![0.0f32; total].into_boxed_slice();
        let inv = 1.0 / (GRID - 1) as f32;

        for ri in 0..GRID {
            let r_lin = linearize(ri as f32 * inv);
            for gi in 0..GRID {
                let g_lin = linearize(gi as f32 * inv);
                for bi in 0..GRID {
                    let b_lin = linearize(bi as f32 * inv);

                    let or = matrix[0][0] * r_lin + matrix[0][1] * g_lin + matrix[0][2] * b_lin;
                    let og = matrix[1][0] * r_lin + matrix[1][1] * g_lin + matrix[1][2] * b_lin;
                    let ob = matrix[2][0] * r_lin + matrix[2][1] * g_lin + matrix[2][2] * b_lin;

                    let idx = (ri * GRID * GRID + gi * GRID + bi) * 3;
                    data[idx] = or;
                    data[idx + 1] = og;
                    data[idx + 2] = ob;
                }
            }
        }

        Self { data, encode }
    }

    /// Look up a single output value at a grid point.
    #[inline(always)]
    fn fetch(&self, r: usize, g: usize, b: usize, ch: usize) -> f32 {
        self.data[(r * GRID * GRID + g * GRID + b) * 3 + ch]
    }

    /// Tetrahedral interpolation for one pixel, one channel.
    #[inline(always)]
    fn tetra_ch(
        &self,
        rx: f32,
        ry: f32,
        rz: f32,
        x: usize,
        y: usize,
        z: usize,
        xn: usize,
        yn: usize,
        zn: usize,
        ch: usize,
    ) -> f32 {
        let c0 = self.fetch(x, y, z, ch);
        let (c1, c2, c3);

        if rx >= ry {
            if ry >= rz {
                // rx >= ry >= rz
                c1 = self.fetch(xn, y, z, ch) - c0;
                c2 = self.fetch(xn, yn, z, ch) - self.fetch(xn, y, z, ch);
                c3 = self.fetch(xn, yn, zn, ch) - self.fetch(xn, yn, z, ch);
            } else if rx >= rz {
                // rx >= rz > ry
                c1 = self.fetch(xn, y, z, ch) - c0;
                c2 = self.fetch(xn, yn, zn, ch) - self.fetch(xn, y, zn, ch);
                c3 = self.fetch(xn, y, zn, ch) - self.fetch(xn, y, z, ch);
            } else {
                // rz > rx >= ry
                c1 = self.fetch(xn, y, zn, ch) - self.fetch(x, y, zn, ch);
                c2 = self.fetch(xn, yn, zn, ch) - self.fetch(xn, y, zn, ch);
                c3 = self.fetch(x, y, zn, ch) - c0;
            }
        } else if rx >= rz {
            // ry > rx >= rz
            c1 = self.fetch(xn, yn, z, ch) - self.fetch(x, yn, z, ch);
            c2 = self.fetch(x, yn, z, ch) - c0;
            c3 = self.fetch(xn, yn, zn, ch) - self.fetch(xn, yn, z, ch);
        } else if ry >= rz {
            // ry >= rz > rx
            c1 = self.fetch(xn, yn, zn, ch) - self.fetch(x, yn, zn, ch);
            c2 = self.fetch(x, yn, z, ch) - c0;
            c3 = self.fetch(x, yn, zn, ch) - self.fetch(x, yn, z, ch);
        } else {
            // rz > ry > rx
            c1 = self.fetch(xn, yn, zn, ch) - self.fetch(x, yn, zn, ch);
            c2 = self.fetch(x, yn, zn, ch) - self.fetch(x, y, zn, ch);
            c3 = self.fetch(x, y, zn, ch) - c0;
        }

        c0 + c1 * rx + c2 * ry + c3 * rz
    }

    /// Convert one u8 RGB pixel via tetrahedral interpolation in linear space,
    /// then encode to the destination TRC.
    #[inline(always)]
    fn convert_pixel(&self, r: u8, g: u8, b: u8) -> (u8, u8, u8) {
        const SCALE: f32 = (GRID - 1) as f32 / 255.0;

        let rf = r as f32 * SCALE;
        let gf = g as f32 * SCALE;
        let bf = b as f32 * SCALE;

        let ri = rf.floor() as usize;
        let gi = gf.floor() as usize;
        let bi = bf.floor() as usize;

        let rn = (ri + 1).min(GRID - 1);
        let gn = (gi + 1).min(GRID - 1);
        let bn = (bi + 1).min(GRID - 1);

        let rx = rf - ri as f32;
        let gy = gf - gi as f32;
        let bz = bf - bi as f32;

        // Interpolate in linear space (smooth, low error)
        let or = self.tetra_ch(rx, gy, bz, ri, gi, bi, rn, gn, bn, 0);
        let og = self.tetra_ch(rx, gy, bz, ri, gi, bi, rn, gn, bn, 1);
        let ob = self.tetra_ch(rx, gy, bz, ri, gi, bi, rn, gn, bn, 2);

        // Encode from linear to destination TRC, then quantize
        let enc = self.encode;
        (
            (enc(or) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
            (enc(og) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
            (enc(ob) * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
        )
    }

    /// Convert a row of u8 RGB pixels via the CLUT.
    pub(crate) fn convert_row_rgb(&self, src: &[u8], dst: &mut [u8]) {
        debug_assert_eq!(src.len() % 3, 0);
        debug_assert_eq!(src.len(), dst.len());
        for (s, d) in src.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
            let (r, g, b) = self.convert_pixel(s[0], s[1], s[2]);
            d[0] = r;
            d[1] = g;
            d[2] = b;
        }
    }

    /// Convert a row of u8 RGBA pixels via the CLUT. Alpha copied.
    pub(crate) fn convert_row_rgba(&self, src: &[u8], dst: &mut [u8]) {
        debug_assert_eq!(src.len() % 4, 0);
        debug_assert_eq!(src.len(), dst.len());
        for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
            let (r, g, b) = self.convert_pixel(s[0], s[1], s[2]);
            d[0] = r;
            d[1] = g;
            d[2] = b;
            d[3] = s[3];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_clut_preserves_values() {
        let identity_matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let clut = Clut3d::build(
            core::convert::identity,
            &identity_matrix,
            core::convert::identity,
        );

        // Every u8 value should roundtrip within ±1
        for r in (0..=255).step_by(17) {
            for g in (0..=255).step_by(17) {
                for b in (0..=255).step_by(17) {
                    let src = [r as u8, g as u8, b as u8];
                    let mut dst = [0u8; 3];
                    clut.convert_row_rgb(&src, &mut dst);
                    for ch in 0..3 {
                        assert!(
                            (src[ch] as i16 - dst[ch] as i16).unsigned_abs() <= 1,
                            "identity ({r},{g},{b}) ch{ch}: {} → {}",
                            src[ch],
                            dst[ch]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn white_and_black() {
        use crate::ColorPrimaries;
        let mat = ColorPrimaries::DisplayP3
            .gamut_matrix_to(ColorPrimaries::Bt709)
            .unwrap();
        let clut = Clut3d::build(
            linear_srgb::tf::srgb_to_linear,
            &mat,
            linear_srgb::tf::linear_to_srgb,
        );

        let mut dst = [0u8; 3];
        clut.convert_row_rgb(&[255, 255, 255], &mut dst);
        assert_eq!(dst, [255, 255, 255], "white");

        clut.convert_row_rgb(&[0, 0, 0], &mut dst);
        assert_eq!(dst, [0, 0, 0], "black");
    }

    #[test]
    fn matches_direct_conversion() {
        use crate::ColorPrimaries;
        let mat = ColorPrimaries::DisplayP3
            .gamut_matrix_to(ColorPrimaries::Bt709)
            .unwrap();
        let clut = Clut3d::build(
            linear_srgb::tf::srgb_to_linear,
            &mat,
            linear_srgb::tf::linear_to_srgb,
        );

        // Compare against direct scalar conversion
        let mut max_delta: u8 = 0;
        for r in (0..=255).step_by(5) {
            for g in (0..=255).step_by(5) {
                for b in (0..=255).step_by(5) {
                    let src = [r as u8, g as u8, b as u8];
                    let mut clut_dst = [0u8; 3];
                    let mut direct_dst = [0u8; 3];
                    clut.convert_row_rgb(&src, &mut clut_dst);

                    // Direct: linearize → matrix → encode → quantize
                    let rl = linear_srgb::tf::srgb_to_linear(src[0] as f32 / 255.0);
                    let gl = linear_srgb::tf::srgb_to_linear(src[1] as f32 / 255.0);
                    let bl = linear_srgb::tf::srgb_to_linear(src[2] as f32 / 255.0);
                    let or = mat[0][0] * rl + mat[0][1] * gl + mat[0][2] * bl;
                    let og = mat[1][0] * rl + mat[1][1] * gl + mat[1][2] * bl;
                    let ob = mat[2][0] * rl + mat[2][1] * gl + mat[2][2] * bl;
                    direct_dst[0] =
                        (linear_srgb::tf::linear_to_srgb(or) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                    direct_dst[1] =
                        (linear_srgb::tf::linear_to_srgb(og) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                    direct_dst[2] =
                        (linear_srgb::tf::linear_to_srgb(ob) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;

                    for ch in 0..3 {
                        let d = (clut_dst[ch] as i16 - direct_dst[ch] as i16).unsigned_abs() as u8;
                        if d > max_delta {
                            max_delta = d;
                        }
                    }
                }
            }
        }
        assert!(max_delta <= 1, "CLUT vs direct max delta: {max_delta}");
    }

    #[test]
    fn rgba_alpha_passthrough() {
        let mat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let clut = Clut3d::build(core::convert::identity, &mat, core::convert::identity);

        let src = [128u8, 64, 32, 200];
        let mut dst = [0u8; 4];
        clut.convert_row_rgba(&src, &mut dst);
        assert_eq!(dst[3], 200);
    }
}
