//! ITU-T H.273 Table 4 YCbCr / GBR / YCgCo matrix coefficients ([`YcbcrMatrix`]).
//!
//! For reconstructing multi-plane `PlanarView` buffers (YCbCr, GBR, planar RGB/RGBA,
//! or monochrome) into interleaved pixels, see [`crate::planar`]
//! ([`reconstruct_planar_to_buf`](crate::reconstruct_planar_to_buf),
//! [`reconstruct_planar_into`](crate::reconstruct_planar_into),
//! [`reconstruct_planar_f32`](crate::reconstruct_planar_f32)).

use zenpixels::Primaries;

/// YCbCr / GBR / YCgCo luma-chroma matrix standard (ITU-T H.273 Table 4).
///
/// Every distinct H.273 Table 4 non-constant-luminance code has its own variant so
/// [`from_cicp_code`](Self::from_cicp_code) ↔ [`to_cicp_code`](Self::to_cicp_code) is
/// 100% lossless (e.g. `5` = [`Bt470Bg`](Self::Bt470Bg) and `6` = [`Bt601`](Self::Bt601)
/// stay distinct on the wire while sharing coefficients via
/// [`same_coefficients`](Self::same_coefficients)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum YcbcrMatrix {
    /// Identity / GBR (`MC = 0`: `Y = G, Cb = B, Cr = R`, lossless RGB in planar form).
    Identity,
    /// ITU-R BT.709 (`MC = 1`: `Kr = 0.2126, Kb = 0.0722`).
    Bt709,
    /// FCC Title 47 CFR 73.682 (`MC = 4`: `Kr = 0.30, Kb = 0.11`).
    Fcc,
    /// ITU-R BT.470-6 System B, G (`MC = 5`: `Kr = 0.299, Kb = 0.114`).
    Bt470Bg,
    /// SMPTE 170M / ITU-R BT.601 (`MC = 6`: `Kr = 0.299, Kb = 0.114`).
    Bt601,
    /// SMPTE 240M (`MC = 7`: `Kr = 0.212, Kb = 0.087`).
    Smpte240M,
    /// YCgCo (`MC = 8`: `R = Y - Cg + Co, G = Y + Cg, B = Y - Cg - Co`).
    YCgCo,
    /// ITU-R BT.2020 NCL (`MC = 9`: `Kr = 0.2627, Kb = 0.0593`).
    Bt2020,
    /// Chromaticity-derived non-constant luminance (`MC = 12`: `Kr, Kb` derived from primaries).
    ChromaDerivedNcl,
}

impl YcbcrMatrix {
    /// Losslessly map an ITU-T H.273 Table 4 `matrix_coefficients` code to a [`YcbcrMatrix`].
    ///
    /// Recognizes `0` (`Identity` / GBR), `1` (`Bt709`), `4` (`Fcc`), `5` (`Bt470Bg`),
    /// `6` (`Bt601`), `7` (`Smpte240M`), `8` (`YCgCo`), `9` (`Bt2020`), and `12`
    /// (`ChromaDerivedNcl`). Returns `None` for unspecified (`2`), reserved (`3`), or
    /// non-NCL matrix codes.
    #[inline]
    #[must_use]
    pub const fn from_cicp_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Identity),
            1 => Some(Self::Bt709),
            4 => Some(Self::Fcc),
            5 => Some(Self::Bt470Bg),
            6 => Some(Self::Bt601),
            7 => Some(Self::Smpte240M),
            8 => Some(Self::YCgCo),
            9 => Some(Self::Bt2020),
            12 => Some(Self::ChromaDerivedNcl),
            _ => None,
        }
    }

    /// Exact ITU-T H.273 Table 4 `matrix_coefficients` code for this matrix.
    #[inline]
    #[must_use]
    pub const fn to_cicp_code(self) -> u8 {
        match self {
            Self::Identity => 0,
            Self::Bt709 => 1,
            Self::Fcc => 4,
            Self::Bt470Bg => 5,
            Self::Bt601 => 6,
            Self::Smpte240M => 7,
            Self::YCgCo => 8,
            Self::Bt2020 => 9,
            Self::ChromaDerivedNcl => 12,
        }
    }

    /// Whether `self` and `other` define identical luma-chroma reconstruction coefficients
    /// (e.g. `Bt470Bg` (`MC = 5`) and `Bt601` (`MC = 6`) share `Kr = 0.299, Kb = 0.114`).
    #[inline]
    #[must_use]
    pub const fn same_coefficients(self, other: Self) -> bool {
        matches!(
            (self, other),
            (Self::Identity, Self::Identity)
                | (Self::Bt709, Self::Bt709)
                | (Self::Fcc, Self::Fcc)
                | (Self::Bt470Bg | Self::Bt601, Self::Bt470Bg | Self::Bt601)
                | (Self::Smpte240M, Self::Smpte240M)
                | (Self::YCgCo, Self::YCgCo)
                | (Self::Bt2020, Self::Bt2020)
                | (Self::ChromaDerivedNcl, Self::ChromaDerivedNcl)
        )
    }

    /// Fixed `(Kr, Kb)` weights for standard NCL matrices (`None` for `Identity`, `YCgCo`,
    /// or `ChromaDerivedNcl`).
    #[inline]
    #[must_use]
    pub const fn kr_kb(self) -> Option<(f32, f32)> {
        match self {
            Self::Bt709 => Some((0.2126, 0.0722)),
            Self::Fcc => Some((0.30, 0.11)),
            Self::Bt470Bg | Self::Bt601 => Some((0.299, 0.114)),
            Self::Smpte240M => Some((0.212, 0.087)),
            Self::Bt2020 => Some((0.2627, 0.0593)),
            Self::Identity | Self::YCgCo | Self::ChromaDerivedNcl => None,
        }
    }

    /// Resolve `(Kr, Kb)` weights, deriving them from `primaries` when `self` is
    /// [`ChromaDerivedNcl`](Self::ChromaDerivedNcl).
    #[inline]
    #[must_use]
    pub fn kr_kb_for(self, primaries: Option<Primaries>) -> Option<(f32, f32)> {
        if self == Self::ChromaDerivedNcl {
            let p = primaries?;
            if p == Primaries::Unknown {
                return None;
            }
            let [kr, kg, kb] = p.luma_weights();
            if kg <= 1e-6 {
                return None;
            }
            return Some((kr, kb));
        }
        self.kr_kb()
    }
}
