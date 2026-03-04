use core::fmt;

use super::ChannelType;

// ---------------------------------------------------------------------------
// Chroma subsampling
// ---------------------------------------------------------------------------

/// Chroma subsampling ratio.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum Subsampling {
    /// 4:4:4 — no subsampling, full resolution chroma.
    #[default]
    S444 = 0,
    /// 4:2:2 — horizontal half resolution chroma.
    S422 = 1,
    /// 4:2:0 — both horizontal and vertical half resolution chroma.
    S420 = 2,
    /// 4:1:1 — quarter horizontal resolution chroma.
    S411 = 3,
}

impl Subsampling {
    /// Horizontal subsampling factor (1 = full, 2 = half, 4 = quarter).
    #[inline]
    #[allow(unreachable_patterns)]
    pub const fn h_factor(self) -> u8 {
        match self {
            Self::S444 => 1,
            Self::S422 | Self::S420 => 2,
            Self::S411 => 4,
            _ => 1,
        }
    }

    /// Vertical subsampling factor (1 = full, 2 = half).
    #[inline]
    #[allow(unreachable_patterns)]
    pub const fn v_factor(self) -> u8 {
        match self {
            Self::S420 => 2,
            _ => 1,
        }
    }

    /// Map horizontal and vertical subsampling factors to a named pattern.
    ///
    /// Returns `None` for factor combinations that don't match a standard
    /// subsampling pattern.
    pub const fn from_factors(h: u8, v: u8) -> Option<Self> {
        match (h, v) {
            (1, 1) => Some(Self::S444),
            (2, 1) => Some(Self::S422),
            (2, 2) => Some(Self::S420),
            (4, 1) => Some(Self::S411),
            _ => None,
        }
    }
}

impl fmt::Display for Subsampling {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::S444 => f.write_str("4:4:4"),
            Self::S422 => f.write_str("4:2:2"),
            Self::S420 => f.write_str("4:2:0"),
            Self::S411 => f.write_str("4:1:1"),
        }
    }
}

// ---------------------------------------------------------------------------
// YUV matrix coefficients
// ---------------------------------------------------------------------------

/// YCbCr matrix coefficients for luma/chroma conversion.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum YuvMatrix {
    /// Identity / not applicable (RGB, Gray, Oklab, etc.).
    #[default]
    Identity = 0,
    /// BT.601: Y = 0.299R + 0.587G + 0.114B (JPEG, WebP, SD video).
    Bt601 = 1,
    /// BT.709: Y = 0.2126R + 0.7152G + 0.0722B (AVIF, HEIC, HD video).
    Bt709 = 2,
    /// BT.2020: Y = 0.2627R + 0.6780G + 0.0593B (4K/8K HDR).
    Bt2020 = 3,
}

impl YuvMatrix {
    /// RGB to Y luma coefficients [Kr, Kg, Kb].
    #[allow(unreachable_patterns)]
    pub const fn rgb_to_y_coeffs(self) -> [f64; 3] {
        match self {
            Self::Identity => [1.0, 0.0, 0.0],
            Self::Bt601 => [0.299, 0.587, 0.114],
            Self::Bt709 => [0.2126, 0.7152, 0.0722],
            Self::Bt2020 => [0.2627, 0.6780, 0.0593],
            _ => [0.2126, 0.7152, 0.0722],
        }
    }

    /// Map a CICP `matrix_coefficients` code to a [`YuvMatrix`].
    pub const fn from_cicp(mc: u8) -> Option<Self> {
        match mc {
            0 => Some(Self::Identity),
            1 => Some(Self::Bt709),
            5 | 6 => Some(Self::Bt601),
            9 => Some(Self::Bt2020),
            _ => None,
        }
    }
}

impl fmt::Display for YuvMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identity => f.write_str("Identity"),
            Self::Bt601 => f.write_str("BT.601"),
            Self::Bt709 => f.write_str("BT.709"),
            Self::Bt2020 => f.write_str("BT.2020"),
        }
    }
}

// ---------------------------------------------------------------------------
// PlaneSemantic
// ---------------------------------------------------------------------------

/// Semantic label for a plane in a multi-plane image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum PlaneSemantic {
    /// Luma (Y) — brightness in YCbCr color spaces.
    Luma = 0,
    /// Chroma blue-difference (Cb / U).
    ChromaCb = 1,
    /// Chroma red-difference (Cr / V).
    ChromaCr = 2,
    /// Red channel.
    Red = 3,
    /// Green channel.
    Green = 4,
    /// Blue channel.
    Blue = 5,
    /// Alpha (transparency) channel.
    Alpha = 6,
    /// Depth map.
    Depth = 7,
    /// Gain map (e.g., Ultra HDR).
    GainMap = 8,
    /// Grayscale (single-channel luminance, not part of YCbCr).
    Gray = 9,
    /// Oklab lightness (L).
    OklabL = 10,
    /// Oklab green-red axis (a).
    OklabA = 11,
    /// Oklab blue-yellow axis (b).
    OklabB = 12,
}

impl PlaneSemantic {
    /// Whether this semantic represents a luminance-like channel.
    #[inline]
    pub const fn is_luminance(self) -> bool {
        matches!(self, Self::Luma | Self::Gray | Self::OklabL)
    }

    /// Whether this semantic represents a chroma channel.
    #[inline]
    pub const fn is_chroma(self) -> bool {
        matches!(
            self,
            Self::ChromaCb | Self::ChromaCr | Self::OklabA | Self::OklabB
        )
    }

    /// Whether this semantic is an RGB color channel.
    #[inline]
    pub const fn is_rgb(self) -> bool {
        matches!(self, Self::Red | Self::Green | Self::Blue)
    }

    /// Whether this semantic is the alpha channel.
    #[inline]
    pub const fn is_alpha(self) -> bool {
        matches!(self, Self::Alpha)
    }
}

impl fmt::Display for PlaneSemantic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Luma => f.write_str("Luma"),
            Self::ChromaCb => f.write_str("Cb"),
            Self::ChromaCr => f.write_str("Cr"),
            Self::Red => f.write_str("R"),
            Self::Green => f.write_str("G"),
            Self::Blue => f.write_str("B"),
            Self::Alpha => f.write_str("A"),
            Self::Depth => f.write_str("Depth"),
            Self::GainMap => f.write_str("GainMap"),
            Self::Gray => f.write_str("Gray"),
            Self::OklabL => f.write_str("Oklab.L"),
            Self::OklabA => f.write_str("Oklab.a"),
            Self::OklabB => f.write_str("Oklab.b"),
        }
    }
}

// ---------------------------------------------------------------------------
// PlaneDescriptor — per-plane metadata (no pixel data)
// ---------------------------------------------------------------------------

/// Metadata describing a single plane in a multi-plane image.
///
/// This is a pure descriptor — no pixel data, no heap allocation.
/// Subsample factors are explicit per-plane rather than inferred from
/// dimension ratios, because strip-based pipelines don't carry global
/// image dimensions.
///
/// # Examples
///
/// ```
/// use zenpixels::{PlaneDescriptor, PlaneSemantic, ChannelType};
///
/// let luma = PlaneDescriptor::new(PlaneSemantic::Luma, ChannelType::F32);
/// assert!(!luma.is_subsampled());
/// assert_eq!(luma.plane_width(1920), 1920);
///
/// let chroma = PlaneDescriptor::new(PlaneSemantic::ChromaCb, ChannelType::F32)
///     .with_subsampling(2, 2); // 4:2:0
/// assert!(chroma.is_subsampled());
/// assert_eq!(chroma.plane_width(1920), 960);
/// assert_eq!(chroma.plane_height(1080), 540);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlaneDescriptor {
    /// What this plane represents.
    pub semantic: PlaneSemantic,
    /// Storage type for each sample in this plane.
    pub channel_type: ChannelType,
    /// Horizontal subsampling factor (1 = full resolution, 2 = half, 4 = quarter).
    pub h_subsample: u8,
    /// Vertical subsampling factor (1 = full resolution, 2 = half).
    pub v_subsample: u8,
}

impl PlaneDescriptor {
    /// Create a full-resolution plane descriptor (no subsampling).
    #[inline]
    pub const fn new(semantic: PlaneSemantic, channel_type: ChannelType) -> Self {
        Self {
            semantic,
            channel_type,
            h_subsample: 1,
            v_subsample: 1,
        }
    }

    /// Builder: set subsampling factors.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `h` and `v` are non-zero powers of two.
    #[inline]
    pub const fn with_subsampling(mut self, h: u8, v: u8) -> Self {
        debug_assert!(
            h > 0 && h.is_power_of_two(),
            "h_subsample must be a power of 2"
        );
        debug_assert!(
            v > 0 && v.is_power_of_two(),
            "v_subsample must be a power of 2"
        );
        self.h_subsample = h;
        self.v_subsample = v;
        self
    }

    /// Compute the width of this plane given a reference (luma) width.
    ///
    /// Uses ceiling division so subsampled planes always cover the full image.
    #[inline]
    pub const fn plane_width(&self, ref_width: u32) -> u32 {
        ref_width.div_ceil(self.h_subsample as u32)
    }

    /// Compute the height of this plane given a reference (luma) height.
    ///
    /// Uses ceiling division so subsampled planes always cover the full image.
    #[inline]
    pub const fn plane_height(&self, ref_height: u32) -> u32 {
        ref_height.div_ceil(self.v_subsample as u32)
    }

    /// Whether this plane is subsampled (either axis).
    #[inline]
    pub const fn is_subsampled(&self) -> bool {
        self.h_subsample > 1 || self.v_subsample > 1
    }

    /// Bytes per sample in this plane.
    #[inline]
    pub const fn bytes_per_sample(&self) -> usize {
        self.channel_type.byte_size()
    }
}

impl fmt::Display for PlaneDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.semantic, self.channel_type)?;
        if self.is_subsampled() {
            write!(f, " (1/{}×1/{})", self.h_subsample, self.v_subsample)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PlaneMask — bitmask for plane selection
// ---------------------------------------------------------------------------

/// Bitmask selecting which planes to process in a multi-plane operation.
///
/// Supports up to 8 planes. Operations that only affect certain channels
/// (e.g., luma sharpening) use this to skip untouched planes.
///
/// # Examples
///
/// ```
/// use zenpixels::PlaneMask;
///
/// let mask = PlaneMask::LUMA.union(PlaneMask::ALPHA);
/// assert!(mask.includes(0));  // luma
/// assert!(!mask.includes(1)); // chroma Cb
/// assert!(mask.includes(3));  // alpha
/// assert_eq!(mask.count(), 2);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlaneMask {
    bits: u8,
}

impl PlaneMask {
    /// All planes (bits 0–7).
    pub const ALL: Self = Self { bits: 0xFF };
    /// No planes.
    pub const NONE: Self = Self { bits: 0 };
    /// Plane 0 only (luma / lightness / red / gray).
    pub const LUMA: Self = Self { bits: 0b0001 };
    /// Planes 1 and 2 (chroma Cb + Cr, or Oklab a + b).
    pub const CHROMA: Self = Self { bits: 0b0110 };
    /// Plane 3 (alpha).
    pub const ALPHA: Self = Self { bits: 0b1000 };

    /// Mask for a single plane by index (0–7).
    #[inline]
    pub const fn single(idx: usize) -> Self {
        debug_assert!(idx < 8, "PlaneMask supports at most 8 planes");
        Self {
            bits: 1u8 << (idx as u8),
        }
    }

    /// Whether the plane at `idx` is included in this mask.
    #[inline]
    pub const fn includes(&self, idx: usize) -> bool {
        if idx >= 8 {
            return false;
        }
        (self.bits >> (idx as u8)) & 1 != 0
    }

    /// Union of two masks.
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self {
            bits: self.bits | other.bits,
        }
    }

    /// Intersection of two masks.
    #[inline]
    pub const fn intersection(self, other: Self) -> Self {
        Self {
            bits: self.bits & other.bits,
        }
    }

    /// Number of planes selected.
    #[inline]
    pub const fn count(&self) -> u32 {
        self.bits.count_ones()
    }

    /// Whether no planes are selected.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.bits == 0
    }

    /// The raw bitmask value.
    #[inline]
    pub const fn bits(&self) -> u8 {
        self.bits
    }

    /// Construct from a raw bitmask value.
    #[inline]
    pub const fn from_bits(bits: u8) -> Self {
        Self { bits }
    }
}

impl fmt::Display for PlaneMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Self::ALL {
            return f.write_str("ALL");
        }
        if self.is_empty() {
            return f.write_str("NONE");
        }
        let mut first = true;
        for i in 0..8 {
            if self.includes(i) {
                if !first {
                    f.write_str("|")?;
                }
                write!(f, "{i}")?;
                first = false;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Plane — buffer + semantic (legacy, retained for backward compat)
// ---------------------------------------------------------------------------

/// A single plane with its semantic label.
///
/// Each plane is an independent [`PixelBuffer`](crate::buffer::PixelBuffer)
/// (opaque gray channel) that can be a different size from other planes
/// (e.g., subsampled chroma).
///
/// **Prefer [`PlaneLayout`] + separate buffers** for new code. This type
/// is retained for backward compatibility.
pub struct Plane {
    /// The pixel data for this plane.
    pub buffer: crate::buffer::PixelBuffer,
    /// What this plane represents.
    pub semantic: PlaneSemantic,
}

/// How planes in a multi-plane image relate to each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PlaneRelationship {
    /// Independent channels (e.g., split R, G, B).
    Independent,
    /// YCbCr with a specific matrix. Subsampling per-plane in [`PlaneDescriptor`].
    YCbCr {
        /// The YCbCr matrix coefficients.
        matrix: YuvMatrix,
    },
    /// Oklab perceptual color space (L/a/b). Fixed transform, no matrix parameter.
    Oklab,
    /// Gain map (base rendition + gain plane).
    GainMap,
}

// ---------------------------------------------------------------------------
// PlaneLayout — complete spatial layout descriptor
// ---------------------------------------------------------------------------

/// Complete layout of a multi-plane image.
///
/// Separates the *metadata* (how many planes, what they represent, their
/// subsampling) from any pixel buffers. Every consumer that needs to know
/// about planar organization works with `PlaneLayout`; only code that
/// allocates or reads pixels needs actual buffers.
///
/// # Examples
///
/// ```
/// use zenpixels::{PlaneLayout, ChannelType, PlaneSemantic};
///
/// let layout = PlaneLayout::ycbcr_420(ChannelType::U8);
/// assert!(layout.is_planar());
/// assert!(layout.is_ycbcr());
/// assert!(layout.has_subsampling());
/// assert_eq!(layout.plane_count(), 3);
///
/// // Check plane semantics
/// let planes = layout.planes();
/// assert_eq!(planes[0].semantic, PlaneSemantic::Luma);
/// assert_eq!(planes[1].semantic, PlaneSemantic::ChromaCb);
/// assert!(!planes[0].is_subsampled());
/// assert!(planes[1].is_subsampled());
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlaneLayout {
    /// Interleaved pixel data (channels packed per-pixel).
    Interleaved {
        /// Number of channels per pixel (e.g., 3 for RGB, 4 for RGBA).
        channels: u8,
    },
    /// Planar pixel data (one buffer per plane).
    Planar {
        /// Descriptor for each plane.
        planes: alloc::vec::Vec<PlaneDescriptor>,
        /// How the planes relate to each other.
        relationship: PlaneRelationship,
    },
}

impl PlaneLayout {
    // --- Factory methods ---

    /// YCbCr 4:4:4 (no subsampling).
    pub fn ycbcr_444(ct: ChannelType) -> Self {
        Self::Planar {
            planes: alloc::vec![
                PlaneDescriptor::new(PlaneSemantic::Luma, ct),
                PlaneDescriptor::new(PlaneSemantic::ChromaCb, ct),
                PlaneDescriptor::new(PlaneSemantic::ChromaCr, ct),
            ],
            relationship: PlaneRelationship::YCbCr {
                matrix: YuvMatrix::Bt601,
            },
        }
    }

    /// YCbCr 4:4:4 with a specific matrix.
    pub fn ycbcr_444_matrix(ct: ChannelType, matrix: YuvMatrix) -> Self {
        Self::Planar {
            planes: alloc::vec![
                PlaneDescriptor::new(PlaneSemantic::Luma, ct),
                PlaneDescriptor::new(PlaneSemantic::ChromaCb, ct),
                PlaneDescriptor::new(PlaneSemantic::ChromaCr, ct),
            ],
            relationship: PlaneRelationship::YCbCr { matrix },
        }
    }

    /// YCbCr 4:2:2 (horizontal half chroma).
    pub fn ycbcr_422(ct: ChannelType) -> Self {
        Self::Planar {
            planes: alloc::vec![
                PlaneDescriptor::new(PlaneSemantic::Luma, ct),
                PlaneDescriptor::new(PlaneSemantic::ChromaCb, ct).with_subsampling(2, 1),
                PlaneDescriptor::new(PlaneSemantic::ChromaCr, ct).with_subsampling(2, 1),
            ],
            relationship: PlaneRelationship::YCbCr {
                matrix: YuvMatrix::Bt601,
            },
        }
    }

    /// YCbCr 4:2:0 (half chroma in both axes).
    pub fn ycbcr_420(ct: ChannelType) -> Self {
        Self::Planar {
            planes: alloc::vec![
                PlaneDescriptor::new(PlaneSemantic::Luma, ct),
                PlaneDescriptor::new(PlaneSemantic::ChromaCb, ct).with_subsampling(2, 2),
                PlaneDescriptor::new(PlaneSemantic::ChromaCr, ct).with_subsampling(2, 2),
            ],
            relationship: PlaneRelationship::YCbCr {
                matrix: YuvMatrix::Bt601,
            },
        }
    }

    /// Planar RGB (3 independent planes, no subsampling).
    pub fn rgb(ct: ChannelType) -> Self {
        Self::Planar {
            planes: alloc::vec![
                PlaneDescriptor::new(PlaneSemantic::Red, ct),
                PlaneDescriptor::new(PlaneSemantic::Green, ct),
                PlaneDescriptor::new(PlaneSemantic::Blue, ct),
            ],
            relationship: PlaneRelationship::Independent,
        }
    }

    /// Planar RGBA (4 independent planes, no subsampling).
    pub fn rgba(ct: ChannelType) -> Self {
        Self::Planar {
            planes: alloc::vec![
                PlaneDescriptor::new(PlaneSemantic::Red, ct),
                PlaneDescriptor::new(PlaneSemantic::Green, ct),
                PlaneDescriptor::new(PlaneSemantic::Blue, ct),
                PlaneDescriptor::new(PlaneSemantic::Alpha, ct),
            ],
            relationship: PlaneRelationship::Independent,
        }
    }

    /// Oklab (L/a/b, 3 planes, no subsampling).
    pub fn oklab(ct: ChannelType) -> Self {
        Self::Planar {
            planes: alloc::vec![
                PlaneDescriptor::new(PlaneSemantic::OklabL, ct),
                PlaneDescriptor::new(PlaneSemantic::OklabA, ct),
                PlaneDescriptor::new(PlaneSemantic::OklabB, ct),
            ],
            relationship: PlaneRelationship::Oklab,
        }
    }

    /// Oklab with alpha (L/a/b/A, 4 planes, no subsampling).
    pub fn oklab_alpha(ct: ChannelType) -> Self {
        Self::Planar {
            planes: alloc::vec![
                PlaneDescriptor::new(PlaneSemantic::OklabL, ct),
                PlaneDescriptor::new(PlaneSemantic::OklabA, ct),
                PlaneDescriptor::new(PlaneSemantic::OklabB, ct),
                PlaneDescriptor::new(PlaneSemantic::Alpha, ct),
            ],
            relationship: PlaneRelationship::Oklab,
        }
    }

    /// Grayscale (single plane, no subsampling).
    pub fn gray(ct: ChannelType) -> Self {
        Self::Planar {
            planes: alloc::vec![PlaneDescriptor::new(PlaneSemantic::Gray, ct)],
            relationship: PlaneRelationship::Independent,
        }
    }

    // --- Queries ---

    /// Number of planes (or interleaved channels).
    #[inline]
    pub fn plane_count(&self) -> usize {
        match self {
            Self::Interleaved { channels } => *channels as usize,
            Self::Planar { planes, .. } => planes.len(),
        }
    }

    /// Plane descriptors. Empty slice for interleaved layout.
    #[inline]
    pub fn planes(&self) -> &[PlaneDescriptor] {
        match self {
            Self::Interleaved { .. } => &[],
            Self::Planar { planes, .. } => planes,
        }
    }

    /// Index of the luma/lightness plane, if any.
    pub fn luma_plane_index(&self) -> Option<usize> {
        match self {
            Self::Interleaved { .. } => None,
            Self::Planar { planes, .. } => planes.iter().position(|p| p.semantic.is_luminance()),
        }
    }

    /// Whether any plane is subsampled.
    pub fn has_subsampling(&self) -> bool {
        match self {
            Self::Interleaved { .. } => false,
            Self::Planar { planes, .. } => planes.iter().any(|p| p.is_subsampled()),
        }
    }

    /// Whether this is a YCbCr layout.
    pub fn is_ycbcr(&self) -> bool {
        matches!(
            self,
            Self::Planar {
                relationship: PlaneRelationship::YCbCr { .. },
                ..
            }
        )
    }

    /// Whether this is an Oklab layout.
    pub fn is_oklab(&self) -> bool {
        matches!(
            self,
            Self::Planar {
                relationship: PlaneRelationship::Oklab,
                ..
            }
        )
    }

    /// Whether this layout is planar (not interleaved).
    #[inline]
    pub fn is_planar(&self) -> bool {
        matches!(self, Self::Planar { .. })
    }

    /// The plane relationship, if planar.
    pub fn relationship(&self) -> Option<PlaneRelationship> {
        match self {
            Self::Interleaved { .. } => None,
            Self::Planar { relationship, .. } => Some(*relationship),
        }
    }

    /// Maximum vertical subsampling factor across all planes.
    ///
    /// Returns 1 for interleaved or non-subsampled layouts.
    pub fn max_v_subsample(&self) -> u8 {
        match self {
            Self::Interleaved { .. } => 1,
            Self::Planar { planes, .. } => planes.iter().map(|p| p.v_subsample).max().unwrap_or(1),
        }
    }

    /// Maximum horizontal subsampling factor across all planes.
    ///
    /// Returns 1 for interleaved or non-subsampled layouts.
    pub fn max_h_subsample(&self) -> u8 {
        match self {
            Self::Interleaved { .. } => 1,
            Self::Planar { planes, .. } => planes.iter().map(|p| p.h_subsample).max().unwrap_or(1),
        }
    }

    /// Build a [`PlaneMask`] from a predicate on plane semantics.
    ///
    /// Returns [`PlaneMask::NONE`] for interleaved layouts.
    pub fn mask_where(&self, f: impl Fn(PlaneSemantic) -> bool) -> PlaneMask {
        match self {
            Self::Interleaved { .. } => PlaneMask::NONE,
            Self::Planar { planes, .. } => {
                let mut bits = 0u8;
                for (i, p) in planes.iter().enumerate() {
                    if i < 8 && f(p.semantic) {
                        bits |= 1 << (i as u8);
                    }
                }
                PlaneMask::from_bits(bits)
            }
        }
    }

    /// Mask of luminance-like planes (Luma, Gray, OklabL).
    pub fn luma_mask(&self) -> PlaneMask {
        self.mask_where(|s| s.is_luminance())
    }

    /// Mask of chroma planes (Cb, Cr, OklabA, OklabB).
    pub fn chroma_mask(&self) -> PlaneMask {
        self.mask_where(|s| s.is_chroma())
    }

    /// Mask of alpha planes.
    pub fn alpha_mask(&self) -> PlaneMask {
        self.mask_where(|s| s.is_alpha())
    }
}

impl fmt::Display for PlaneLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Interleaved { channels } => write!(f, "Interleaved({channels}ch)"),
            Self::Planar {
                planes,
                relationship,
            } => {
                write!(f, "{relationship:?}[")?;
                for (i, p) in planes.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{p}")?;
                }
                f.write_str("]")
            }
        }
    }
}

/// A multi-plane image where each plane is an independent pixel buffer.
///
/// Combines a [`PlaneLayout`] (metadata describing plane count, semantics,
/// subsampling, and relationship) with actual pixel buffers. Each buffer
/// corresponds to one plane in the layout.
///
/// # Examples
///
/// ```
/// use zenpixels::{MultiPlaneImage, PlaneLayout, ChannelType, PixelBuffer, PixelDescriptor};
///
/// let layout = PlaneLayout::ycbcr_444(ChannelType::U8);
/// let y = PixelBuffer::new(1920, 1080, PixelDescriptor::GRAY8);
/// let cb = PixelBuffer::new(1920, 1080, PixelDescriptor::GRAY8);
/// let cr = PixelBuffer::new(1920, 1080, PixelDescriptor::GRAY8);
///
/// let img = MultiPlaneImage::new(layout, vec![y, cb, cr]);
/// assert_eq!(img.plane_count(), 3);
/// assert!(img.layout().is_ycbcr());
/// ```
pub struct MultiPlaneImage {
    layout: PlaneLayout,
    buffers: alloc::vec::Vec<crate::buffer::PixelBuffer>,
    origin: Option<alloc::sync::Arc<crate::color::ColorContext>>,
}

impl MultiPlaneImage {
    /// Create a new multi-plane image from a layout and corresponding buffers.
    ///
    /// # Panics
    ///
    /// Debug-asserts that the number of buffers matches the planar plane count.
    pub fn new(layout: PlaneLayout, buffers: alloc::vec::Vec<crate::buffer::PixelBuffer>) -> Self {
        debug_assert!(
            layout.is_planar(),
            "MultiPlaneImage requires a Planar layout"
        );
        debug_assert_eq!(
            layout.plane_count(),
            buffers.len(),
            "buffer count ({}) must match plane count ({})",
            buffers.len(),
            layout.plane_count(),
        );
        Self {
            layout,
            buffers,
            origin: None,
        }
    }

    /// Attach a color context.
    pub fn with_origin(mut self, ctx: alloc::sync::Arc<crate::color::ColorContext>) -> Self {
        self.origin = Some(ctx);
        self
    }

    /// The layout describing this image's plane organization.
    #[inline]
    pub fn layout(&self) -> &PlaneLayout {
        &self.layout
    }

    /// Number of planes.
    #[inline]
    pub fn plane_count(&self) -> usize {
        self.buffers.len()
    }

    /// Access a single buffer by index.
    #[inline]
    pub fn buffer(&self, idx: usize) -> Option<&crate::buffer::PixelBuffer> {
        self.buffers.get(idx)
    }

    /// Access a single buffer mutably by index.
    #[inline]
    pub fn buffer_mut(&mut self, idx: usize) -> Option<&mut crate::buffer::PixelBuffer> {
        self.buffers.get_mut(idx)
    }

    /// Access all buffers.
    #[inline]
    pub fn buffers(&self) -> &[crate::buffer::PixelBuffer] {
        &self.buffers
    }

    /// Access all buffers mutably.
    #[inline]
    pub fn buffers_mut(&mut self) -> &mut [crate::buffer::PixelBuffer] {
        &mut self.buffers
    }

    /// The optional color context shared across all planes.
    #[inline]
    pub fn origin(&self) -> Option<&alloc::sync::Arc<crate::color::ColorContext>> {
        self.origin.as_ref()
    }
}
