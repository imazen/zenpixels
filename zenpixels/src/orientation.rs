//! EXIF orientation as an element of the D4 dihedral group.
//!
//! Every image orientation decomposes into a rotation (0/90/180/270 degrees
//! clockwise) optionally followed by a horizontal flip. The 8 EXIF orientation
//! tag values (1-8) map one-to-one to the 8 elements of the D4 group.
//!
//! This module provides the canonical [`Orientation`] enum for the zen
//! ecosystem, replacing per-crate duplicates with a single shared type.

/// Image orientation — the 8-element D4 dihedral group.
///
/// Values 1-8 match [EXIF tag 274](https://www.exiv2.org/tags.html) exactly.
/// `#[repr(u8)]` means `o as u8` gives the EXIF value directly.
///
/// # Decomposition
///
/// ```text
/// | Variant    | Rotation | FlipH? | Swaps axes? |
/// |------------|----------|--------|-------------|
/// | Identity   | 0°       | no     | no          |
/// | FlipH      | 0°       | yes    | no          |
/// | Rotate180  | 180°     | no     | no          |
/// | FlipV      | 180°     | yes    | no          |
/// | Transpose  | 90° CW   | yes    | yes         |
/// | Rotate90   | 90° CW   | no     | yes         |
/// | Transverse | 270° CW  | yes    | yes         |
/// | Rotate270  | 270° CW  | no     | yes         |
/// ```
///
/// # D4 group operations
///
/// [`compose`](Orientation::compose) implements group multiplication (verified
/// against a full Cayley table). [`inverse`](Orientation::inverse) returns the
/// element that undoes the transformation.
///
/// ```
/// use zenpixels::Orientation;
///
/// let combined = Orientation::Rotate90.then(Orientation::FlipH);
/// assert_eq!(combined, Orientation::Transpose);
///
/// let roundtrip = Orientation::Rotate90.compose(Orientation::Rotate90.inverse());
/// assert_eq!(roundtrip, Orientation::Identity);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum Orientation {
    /// No transformation. EXIF 1.
    #[default]
    Identity = 1,
    /// Horizontal flip (mirror left-right). EXIF 2.
    FlipH = 2,
    /// 180° rotation. EXIF 3.
    Rotate180 = 3,
    /// Vertical flip (= Rotate180 + FlipH). EXIF 4.
    FlipV = 4,
    /// Transpose: reflect over main diagonal (= Rotate90 + FlipH). EXIF 5. Swaps axes.
    Transpose = 5,
    /// 90° clockwise rotation. EXIF 6. Swaps axes.
    Rotate90 = 6,
    /// Transverse: reflect over anti-diagonal (= Rotate270 + FlipH). EXIF 7. Swaps axes.
    Transverse = 7,
    /// 270° clockwise rotation (= 90° counter-clockwise). EXIF 8. Swaps axes.
    Rotate270 = 8,
}

impl Orientation {
    /// All 8 orientations in EXIF order (1-8).
    pub const ALL: [Orientation; 8] = [
        Self::Identity,
        Self::FlipH,
        Self::Rotate180,
        Self::FlipV,
        Self::Transpose,
        Self::Rotate90,
        Self::Transverse,
        Self::Rotate270,
    ];

    /// Create from EXIF orientation tag value (1-8).
    ///
    /// Returns `None` for values outside 1-8. Use `.unwrap_or_default()`
    /// if you want fallback to [`Identity`](Orientation::Identity).
    pub const fn from_exif(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Identity),
            2 => Some(Self::FlipH),
            3 => Some(Self::Rotate180),
            4 => Some(Self::FlipV),
            5 => Some(Self::Transpose),
            6 => Some(Self::Rotate90),
            7 => Some(Self::Transverse),
            8 => Some(Self::Rotate270),
            _ => None,
        }
    }

    /// Convert to EXIF orientation tag value (1-8).
    ///
    /// Equivalent to `self as u8` thanks to `#[repr(u8)]`.
    pub const fn to_exif(self) -> u8 {
        self as u8
    }

    /// Whether this is the identity (no-op) transformation.
    pub const fn is_identity(self) -> bool {
        matches!(self, Self::Identity)
    }

    /// Whether this orientation swaps width and height.
    ///
    /// True for orientations involving a 90° or 270° rotation (EXIF 5-8).
    pub const fn swaps_axes(self) -> bool {
        matches!(
            self,
            Self::Transpose | Self::Rotate90 | Self::Transverse | Self::Rotate270
        )
    }

    /// Whether this can be applied per-row without buffering the full image.
    ///
    /// Only [`Identity`](Self::Identity) and [`FlipH`](Self::FlipH) are
    /// row-local — all other transforms require access to multiple rows.
    pub const fn is_row_local(self) -> bool {
        matches!(self, Self::Identity | Self::FlipH)
    }

    /// Compose two orientations: apply `self` first, then `other`.
    ///
    /// This is D4 group multiplication, verified against the full Cayley table.
    pub const fn compose(self, other: Self) -> Self {
        let (r1, f1) = self.decompose();
        let (r2, f2) = other.decompose();
        if !f1 {
            Self::from_rotation_flip((r1 + r2) & 3, f2)
        } else {
            Self::from_rotation_flip(r1.wrapping_sub(r2) & 3, !f2)
        }
    }

    /// Alias for [`compose`](Self::compose). Reads naturally in chains:
    /// `Rotate90.then(FlipH)` = "apply Rotate90 first, then FlipH."
    pub const fn then(self, other: Self) -> Self {
        self.compose(other)
    }

    /// The inverse orientation: `self.compose(self.inverse()) == Identity`.
    pub const fn inverse(self) -> Self {
        let (r, f) = self.decompose();
        if !f {
            Self::from_rotation_flip((4 - r) & 3, false)
        } else {
            // Flips are self-inverse; rotation direction reverses under flip.
            self
        }
    }

    /// Compute output dimensions after applying this orientation.
    ///
    /// Returns `(width, height)` — swapped when [`swaps_axes`](Self::swaps_axes) is true.
    pub const fn output_dimensions(self, w: u32, h: u32) -> (u32, u32) {
        if self.swaps_axes() { (h, w) } else { (w, h) }
    }

    /// Forward-map a source pixel `(sx, sy)` to its destination position.
    ///
    /// `(w, h)` are the **source** (pre-orientation) dimensions.
    /// Returns `(dx, dy)` in the output coordinate space.
    pub const fn forward_map(self, sx: u32, sy: u32, w: u32, h: u32) -> (u32, u32) {
        match self {
            Self::Identity => (sx, sy),
            Self::FlipH => (w - 1 - sx, sy),
            Self::Rotate90 => (h - 1 - sy, sx),
            Self::Transpose => (sy, sx),
            Self::Rotate180 => (w - 1 - sx, h - 1 - sy),
            Self::FlipV => (sx, h - 1 - sy),
            Self::Rotate270 => (sy, w - 1 - sx),
            Self::Transverse => (h - 1 - sy, w - 1 - sx),
        }
    }

    /// Decompose into `(rotation_quarters, flip)`.
    ///
    /// `rotation_quarters` is 0-3 (number of 90° CW steps).
    /// `flip` is true if a horizontal flip follows the rotation.
    const fn decompose(self) -> (u8, bool) {
        match self {
            Self::Identity => (0, false),
            Self::FlipH => (0, true),
            Self::Rotate90 => (1, false),
            Self::Transpose => (1, true),
            Self::Rotate180 => (2, false),
            Self::FlipV => (2, true),
            Self::Rotate270 => (3, false),
            Self::Transverse => (3, true),
        }
    }

    /// Reconstruct from `(rotation_quarters & 3, flip)`.
    const fn from_rotation_flip(rotation: u8, flip: bool) -> Self {
        match (rotation & 3, flip) {
            (0, false) => Self::Identity,
            (0, true) => Self::FlipH,
            (1, false) => Self::Rotate90,
            (1, true) => Self::Transpose,
            (2, false) => Self::Rotate180,
            (2, true) => Self::FlipV,
            (3, false) => Self::Rotate270,
            (3, true) => Self::Transverse,
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exif_round_trip() {
        for v in 1..=8u8 {
            let o = Orientation::from_exif(v).unwrap();
            assert_eq!(o.to_exif(), v, "round-trip failed for EXIF {v}");
            assert_eq!(o as u8, v, "repr(u8) mismatch for EXIF {v}");
        }
    }

    #[test]
    fn exif_invalid() {
        assert!(Orientation::from_exif(0).is_none());
        assert!(Orientation::from_exif(9).is_none());
        assert!(Orientation::from_exif(255).is_none());
    }

    #[test]
    fn default_is_identity() {
        assert_eq!(Orientation::default(), Orientation::Identity);
    }

    #[test]
    fn identity_properties() {
        assert!(Orientation::Identity.is_identity());
        for &o in &Orientation::ALL[1..] {
            assert!(!o.is_identity());
        }
    }

    #[test]
    fn swaps_axes() {
        for &o in &Orientation::ALL {
            let expected = matches!(
                o,
                Orientation::Transpose
                    | Orientation::Rotate90
                    | Orientation::Transverse
                    | Orientation::Rotate270
            );
            assert_eq!(o.swaps_axes(), expected, "{o:?}");
        }
    }

    #[test]
    fn row_local() {
        assert!(Orientation::Identity.is_row_local());
        assert!(Orientation::FlipH.is_row_local());
        for &o in &Orientation::ALL[2..] {
            assert!(!o.is_row_local(), "{o:?} should not be row-local");
        }
    }

    #[test]
    fn output_dimensions() {
        for &o in &Orientation::ALL {
            let (dw, dh) = o.output_dimensions(100, 200);
            if o.swaps_axes() {
                assert_eq!((dw, dh), (200, 100), "{o:?}");
            } else {
                assert_eq!((dw, dh), (100, 200), "{o:?}");
            }
        }
    }

    #[test]
    fn all_array_matches_exif_order() {
        for (i, &o) in Orientation::ALL.iter().enumerate() {
            assert_eq!(o.to_exif(), (i + 1) as u8);
        }
    }

    // --- D4 group algebra ---

    /// Cayley table from zenjpeg's coeff_transform.rs, re-indexed.
    ///
    /// zenjpeg uses: None=0, FlipH=1, FlipV=2, Transpose=3, Rot90=4, Rot180=5, Rot270=6, Transverse=7
    #[test]
    fn cayley_table() {
        #[rustfmt::skip]
        const CAYLEY: [[usize; 8]; 8] = [
            [0,1,2,3,4,5,6,7], // None
            [1,0,5,6,7,2,3,4], // FlipH
            [2,5,0,4,3,1,7,6], // FlipV
            [3,4,6,0,1,7,2,5], // Transpose
            [4,3,7,2,5,6,0,1], // Rotate90
            [5,2,1,7,6,0,4,3], // Rotate180
            [6,7,3,1,0,4,5,2], // Rotate270
            [7,6,4,5,2,3,1,0], // Transverse
        ];

        // zenjpeg index order → Orientation
        let zj = [
            Orientation::Identity,   // 0
            Orientation::FlipH,      // 1
            Orientation::FlipV,      // 2
            Orientation::Transpose,  // 3
            Orientation::Rotate90,   // 4
            Orientation::Rotate180,  // 5
            Orientation::Rotate270,  // 6
            Orientation::Transverse, // 7
        ];

        for (i, row) in CAYLEY.iter().enumerate() {
            for (j, &expected_idx) in row.iter().enumerate() {
                let a = zj[i];
                let b = zj[j];
                let expected = zj[expected_idx];
                let got = a.compose(b);
                assert_eq!(
                    got, expected,
                    "Cayley: {a:?}.compose({b:?}) = {got:?}, expected {expected:?}"
                );
            }
        }
    }

    #[test]
    fn inverse_all() {
        for &o in &Orientation::ALL {
            let inv = o.inverse();
            assert_eq!(
                o.compose(inv),
                Orientation::Identity,
                "{o:?}.compose({inv:?}) should be Identity"
            );
            assert_eq!(
                inv.compose(o),
                Orientation::Identity,
                "{inv:?}.compose({o:?}) should be Identity"
            );
        }
    }

    #[test]
    fn associativity() {
        for &a in &Orientation::ALL {
            for &b in &Orientation::ALL {
                for &c in &Orientation::ALL {
                    let ab_c = a.compose(b).compose(c);
                    let a_bc = a.compose(b.compose(c));
                    assert_eq!(ab_c, a_bc, "({a:?}*{b:?})*{c:?} != {a:?}*({b:?}*{c:?})");
                }
            }
        }
    }

    #[test]
    fn identity_is_neutral() {
        let id = Orientation::Identity;
        for &o in &Orientation::ALL {
            assert_eq!(id.compose(o), o);
            assert_eq!(o.compose(id), o);
        }
    }

    #[test]
    fn then_is_compose() {
        for &a in &Orientation::ALL {
            for &b in &Orientation::ALL {
                assert_eq!(a.then(b), a.compose(b));
            }
        }
    }

    // --- Coordinate mapping ---

    #[test]
    fn forward_map_brute_force_4x3() {
        let (sw, sh) = (4u32, 3u32);
        for &o in &Orientation::ALL {
            let (dw, dh) = o.output_dimensions(sw, sh);
            // Every source pixel must land inside the output bounds.
            for sy in 0..sh {
                for sx in 0..sw {
                    let (dx, dy) = o.forward_map(sx, sy, sw, sh);
                    assert!(
                        dx < dw && dy < dh,
                        "{o:?}: ({sx},{sy}) mapped to ({dx},{dy}) outside {dw}x{dh}"
                    );
                }
            }
            // Forward map must be a bijection (every output pixel hit exactly once).
            let mut seen = alloc::vec![false; (dw * dh) as usize];
            for sy in 0..sh {
                for sx in 0..sw {
                    let (dx, dy) = o.forward_map(sx, sy, sw, sh);
                    let idx = (dy * dw + dx) as usize;
                    assert!(!seen[idx], "{o:?}: output ({dx},{dy}) hit twice");
                    seen[idx] = true;
                }
            }
            assert!(
                seen.iter().all(|&s| s),
                "{o:?}: not all output pixels covered"
            );
        }
    }

    #[test]
    fn forward_map_inverse_round_trip() {
        let (w, h) = (5u32, 3u32);
        for &o in &Orientation::ALL {
            let inv = o.inverse();
            let (dw, dh) = o.output_dimensions(w, h);
            for sy in 0..h {
                for sx in 0..w {
                    let (dx, dy) = o.forward_map(sx, sy, w, h);
                    let (rx, ry) = inv.forward_map(dx, dy, dw, dh);
                    assert_eq!(
                        (rx, ry),
                        (sx, sy),
                        "{o:?}: ({sx},{sy}) -> ({dx},{dy}) -> ({rx},{ry})"
                    );
                }
            }
        }
    }
}
