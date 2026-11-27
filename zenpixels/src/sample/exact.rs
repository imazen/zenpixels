//! Proven exact code-depth evidence for canonical `U8`/`U16` images.

use core::fmt;

use crate::buffer::PixelView;
use crate::color::{AlphaMode, ColorSpace, PixelFormat, SignalRange};
use crate::layout::{ChannelOrder, SampleType};

/// Proven exact code depth of the current samples of a canonical `U8`/`U16` image.
///
/// `color_bits = N` states that every current color sample `v` survives
/// re-quantization to `N`-bit codes: narrowing it to `N` bits and widening back
/// to storage depth reproduces `v` **exactly**, using the canonical rule for
/// `range`:
///
/// - **Full range:** nearest full-scale rescaling (`(q * max(M) + max(N)/2) / max(N)`).
///   Full-range grids nest **iff `base` divides `depth`** (because `2^base - 1`
///   divides `2^depth - 1`). For example, 8-bit full-range content in `U16` is
///   exact at 8 and 16 bits, **not** at 10 or 12 bits; 3-bit content in `U16`
///   does **not** embed in `U8` (`8 % 3 != 0`).
/// - **Narrow range:** power-of-two anchor scaling (`q << (M - N)`). Narrow
///   grids require at least 8 bits and nest monotonically (`depth >= base`).
/// - **Alpha:** always full scale, tracked independently in `alpha_bits`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExactDepth {
    storage: SampleType,
    range: SignalRange,
    color_bits: u8,
    alpha_bits: u8,
}

/// An exact-depth statement is impossible for its storage, range, or layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExactDepthError {
    /// Evidence is defined for canonical `U8` and `U16` full-packing integer storage only.
    UnsupportedStorage,
    /// Depth is zero or exceeds the storage width.
    InvalidBits,
    /// Narrow-range codes need at least eight bits.
    NarrowBelowEightBits,
    /// The evidence names a different storage type or signal range than the pixels.
    FormatMismatch,
    /// Scans cover gray, gray-alpha, RGB, BGR, RGBA, BGRA, RGBX, and BGRX layouts only.
    UnsupportedLayout,
}

impl fmt::Display for ExactDepthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::UnsupportedStorage => {
                "exact-depth evidence requires full-packing U8 or U16 storage"
            }
            Self::InvalidBits => "exact depth must be positive and fit its storage",
            Self::NarrowBelowEightBits => "narrow-range exact depth needs at least 8 bits",
            Self::FormatMismatch => "exact-depth evidence disagrees with the image spec",
            Self::UnsupportedLayout => {
                "exact-depth scans require Gray, GrayAlpha, RGB, BGR, RGBA, BGRA, RGBX, or BGRX"
            }
        })
    }
}

impl core::error::Error for ExactDepthError {}

impl ExactDepth {
    /// State that current color and alpha samples are exact at the given depths.
    ///
    /// Checked for consistency with `storage` and `range`; does not inspect pixel
    /// bytes. Use [`measure`](Self::measure) or [`holds_for`](Self::holds_for) to
    /// verify untrusted buffers.
    pub const fn new(
        storage: SampleType,
        range: SignalRange,
        color_bits: u8,
        alpha_bits: u8,
    ) -> Result<Self, ExactDepthError> {
        let storage_bits = match storage {
            SampleType::U8 => 8,
            SampleType::U16 => 16,
            _ => return Err(ExactDepthError::UnsupportedStorage),
        };
        if color_bits == 0
            || color_bits > storage_bits
            || alpha_bits == 0
            || alpha_bits > storage_bits
        {
            return Err(ExactDepthError::InvalidBits);
        }
        match range {
            SignalRange::Full => {}
            SignalRange::Narrow if color_bits >= 8 => {}
            SignalRange::Narrow => return Err(ExactDepthError::NarrowBelowEightBits),
        }
        Ok(Self {
            storage,
            range,
            color_bits,
            alpha_bits,
        })
    }

    /// Convenience for formats without an alpha channel: states that color channels are exact
    /// at `color_bits` (setting `alpha_bits` to full storage depth, matching [`measure`](Self::measure)).
    pub const fn new_opaque(
        storage: SampleType,
        range: SignalRange,
        color_bits: u8,
    ) -> Result<Self, ExactDepthError> {
        let storage_bits = match storage {
            SampleType::U8 => 8,
            SampleType::U16 => 16,
            _ => return Err(ExactDepthError::UnsupportedStorage),
        };
        Self::new(storage, range, color_bits, storage_bits)
    }

    /// Storage the statement applies to (`SampleType::U8` or `SampleType::U16`).
    #[inline]
    #[must_use]
    pub const fn storage(self) -> SampleType {
        self.storage
    }

    /// Signal range whose canonical rule defines the color grid.
    #[inline]
    #[must_use]
    pub const fn range(self) -> SignalRange {
        self.range
    }

    /// Smallest proven exact color depth (`1..=storage_bits`, or `8..=storage_bits` for narrow).
    #[inline]
    #[must_use]
    pub const fn color_bits(self) -> u8 {
        self.color_bits
    }

    /// Smallest proven exact alpha depth (`1..=storage_bits`, always full scale).
    #[inline]
    #[must_use]
    pub const fn alpha_bits(self) -> u8 {
        self.alpha_bits
    }

    /// Whether quantizing color to `bits` and widening back reproduces every sample.
    #[inline]
    #[must_use]
    pub const fn color_exact_at(self, bits: u8) -> bool {
        grid_nests(
            self.storage_bits(),
            self.color_bits,
            bits,
            matches!(self.range, SignalRange::Narrow),
        )
    }

    /// Whether quantizing alpha to `bits` and widening back reproduces every sample.
    ///
    /// Alpha is always full scale. `alpha_bits == 1` allows both transparent and
    /// opaque samples; it does not by itself prove an opaque image.
    #[inline]
    #[must_use]
    pub const fn alpha_exact_at(self, bits: u8) -> bool {
        grid_nests(self.storage_bits(), self.alpha_bits, bits, false)
    }

    /// Whether the `color_bits`-deep color codes remain recoverable at `bits` (weaker than exactness).
    #[inline]
    #[must_use]
    pub const fn color_codes_recoverable_at(self, bits: u8) -> bool {
        bits >= self.color_bits
    }

    /// Validate that this evidence is compatible with `format` without requiring a pixel buffer.
    pub const fn validate_format(self, format: &PixelFormat) -> Result<(), ExactDepthError> {
        let spec = format;
        if !spec.layout.packing.is_full() {
            return Err(ExactDepthError::UnsupportedStorage);
        }
        let Ok(roles) = Roles::of(spec) else {
            return match spec.layout.sample {
                SampleType::U8 | SampleType::U16 => Err(ExactDepthError::UnsupportedLayout),
                _ => Err(ExactDepthError::UnsupportedStorage),
            };
        };
        if !matches!(
            (self.storage, spec.layout.sample),
            (SampleType::U8, SampleType::U8) | (SampleType::U16, SampleType::U16)
        ) {
            return Err(ExactDepthError::FormatMismatch);
        }
        let spec_narrow = roles.narrow;
        let self_narrow = matches!(self.range, SignalRange::Narrow);
        if spec_narrow != self_narrow {
            return Err(ExactDepthError::FormatMismatch);
        }
        Ok(())
    }

    /// Measure the smallest depths whose grids hold every active sample in `view`.
    ///
    /// Color and alpha are measured separately; without real alpha, `alpha_bits`
    /// is the storage depth. Reads only active samples (never row padding or an
    /// `RGBX`/`BGRX` padding slot) and exits early once only storage depth remains.
    pub fn measure(view: &PixelView<'_>) -> Result<Self, ExactDepthError> {
        let spec = view.format();
        let roles = Roles::of(spec)?;
        let mut color = GridScan::new(roles.storage_bits, roles.narrow);
        let mut alpha = GridScan::new(roles.storage_bits, false);
        let has_alpha = roles.alpha.is_some();
        roles.for_each(view, |is_alpha, v| {
            if is_alpha {
                alpha.push(v);
            } else {
                color.push(v);
            }
            !(color.settled() && (!has_alpha || alpha.settled()))
        });
        let alpha_bits = if has_alpha {
            alpha.bits()
        } else {
            roles.storage_bits
        };
        let range = if roles.narrow {
            SignalRange::Narrow
        } else {
            SignalRange::Full
        };
        Self::new(spec.layout.sample, range, color.bits(), alpha_bits)
    }

    /// Whether every active sample of `view` lies on this statement's color & alpha grids.
    pub fn holds_for(self, view: &PixelView<'_>) -> Result<bool, ExactDepthError> {
        let spec = view.format();
        self.validate_format(spec)?;
        let roles = Roles::of(spec)?;
        let storage_bits = roles.storage_bits;
        Ok(roles.for_each(view, |is_alpha, v| {
            if is_alpha {
                on_full_grid(v, storage_bits, self.alpha_bits)
            } else if roles.narrow {
                on_narrow_grid(v, storage_bits, self.color_bits)
            } else {
                on_full_grid(v, storage_bits, self.color_bits)
            }
        }))
    }

    /// Intersect two `ExactDepth` claims that both describe the **same** sample set,
    /// returning the tighter proven depth (`min`) for matching storage & range.
    ///
    /// Taking `min` (rather than `gcd`) is intentionally sound under integer rounding:
    /// rounded non-divisor full-range grids can intersect outside `gcd` (for example,
    /// U16 value `64` lies on both the 10-bit and 12-bit grids, but not on the 2-bit grid).
    ///
    /// This is for merging independent evidence about the same buffer (e.g., caller policy
    /// facts plus buffer facts). To merge evidence from **disjoint** strips or regions of
    /// an image into a common containing grid, use [`ExactDepth::union_disjoint_regions`].
    #[must_use]
    pub const fn intersect_same_buffer(a: Self, b: Self) -> Option<Self> {
        if !matches!(
            (a.storage, b.storage),
            (SampleType::U8, SampleType::U8) | (SampleType::U16, SampleType::U16)
        ) {
            return None;
        }
        let a_narrow = matches!(a.range, SignalRange::Narrow);
        let b_narrow = matches!(b.range, SignalRange::Narrow);
        if a_narrow != b_narrow {
            return None;
        }
        let color_bits = if a.color_bits <= b.color_bits {
            a.color_bits
        } else {
            b.color_bits
        };
        let alpha_bits = if a.alpha_bits <= b.alpha_bits {
            a.alpha_bits
        } else {
            b.alpha_bits
        };
        Some(Self {
            storage: a.storage,
            range: a.range,
            color_bits,
            alpha_bits,
        })
    }

    /// Merge two `ExactDepth` measurements from **disjoint** strips or regions of the same
    /// image into the smallest common containing grid holding every sample from both regions.
    ///
    /// - **Narrow range:** monotonic `max(a, b)` (power-of-two anchor grids nest monotonically).
    /// - **Full range:** `lcm(a, b)` when `lcm(a, b) <= storage_bits`, otherwise `storage_bits`
    ///   (because full-range `d1`-bit and `d2`-bit grids embed in `d` iff both `d1` and `d2`
    ///   divide `d`, or `d == storage_bits`).
    ///
    /// Returns `None` if storage type or signal range disagree.
    #[must_use]
    pub const fn union_disjoint_regions(a: Self, b: Self) -> Option<Self> {
        if !matches!(
            (a.storage, b.storage),
            (SampleType::U8, SampleType::U8) | (SampleType::U16, SampleType::U16)
        ) {
            return None;
        }
        let a_narrow = matches!(a.range, SignalRange::Narrow);
        let b_narrow = matches!(b.range, SignalRange::Narrow);
        if a_narrow != b_narrow {
            return None;
        }
        let sb = a.storage_bits();
        let color_bits = if a_narrow {
            if a.color_bits >= b.color_bits {
                a.color_bits
            } else {
                b.color_bits
            }
        } else {
            full_grid_union_bits(sb, a.color_bits, b.color_bits)
        };
        let alpha_bits = full_grid_union_bits(sb, a.alpha_bits, b.alpha_bits);
        Some(Self {
            storage: a.storage,
            range: a.range,
            color_bits,
            alpha_bits,
        })
    }

    const fn storage_bits(self) -> u8 {
        match self.storage {
            SampleType::U8 => 8,
            _ => 16,
        }
    }
}

const fn gcd_u8(mut a: u8, mut b: u8) -> u8 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

const fn full_grid_union_bits(storage_bits: u8, a: u8, b: u8) -> u8 {
    if a >= storage_bits || b >= storage_bits {
        return storage_bits;
    }
    let g = gcd_u8(a, b);
    let lcm = (a as u16 / g as u16) * (b as u16);
    if lcm <= storage_bits as u16 {
        lcm as u8
    } else {
        storage_bits
    }
}

/// Whether the full-range `src_bits` integer code grid embeds **exact** inside the
/// `dst_bits` full-range grid (`2^src_bits - 1` divides `2^dst_bits - 1`, iff `src_bits`
/// divides `dst_bits`).
///
/// Examples:
/// - `2 → 8`: `8 % 2 == 0` → `true` (`255 / 3 = 85`)
/// - `4 → 8`: `8 % 4 == 0` → `true` (`255 / 15 = 17`)
/// - `3 → 8`: `8 % 3 != 0` → `false` (code `1/7` rounds to `36/255` ≠ `1/7`)
/// - `8 → 16`: `16 % 8 == 0` → `true` (`65535 / 255 = 257`)
/// - `10 → 12`: `12 % 10 != 0` → `false`
#[inline]
#[must_use]
pub const fn full_range_grid_nests(src_bits: u8, dst_bits: u8) -> bool {
    if src_bits == 0 || dst_bits == 0 {
        return false;
    }
    dst_bits >= src_bits && dst_bits % src_bits == 0
}

/// Whether the narrow-range `src_bits` integer code grid embeds **exact** inside the
/// `dst_bits` narrow-range grid (power-of-two scaling requires `src_bits >= 8` and `dst_bits >= src_bits`).
#[inline]
#[must_use]
pub const fn narrow_range_grid_nests(src_bits: u8, dst_bits: u8) -> bool {
    src_bits >= 8 && dst_bits >= src_bits
}

/// Whether the `base`-bit grid in `storage_bits` storage is a subset of the
/// `depth`-bit grid.
const fn grid_nests(storage_bits: u8, base: u8, depth: u8, narrow: bool) -> bool {
    if depth == 0 {
        false
    } else if depth >= storage_bits {
        true
    } else if narrow {
        depth >= base
    } else {
        depth % base == 0
    }
}

/// Whether `v` lies on the full-scale `bits`-deep grid of `storage_bits` storage.
const fn on_full_grid(v: u32, storage_bits: u8, bits: u8) -> bool {
    if bits >= storage_bits {
        return true;
    }
    let storage_max = (1u32 << storage_bits) - 1;
    let max = (1u32 << bits) - 1;
    let q = (v * max + storage_max / 2) / storage_max;
    (q * storage_max + max / 2) / max == v
}

/// Whether `v` lies on the narrow-range `bits`-deep grid of `storage_bits` storage.
const fn on_narrow_grid(v: u32, storage_bits: u8, bits: u8) -> bool {
    bits >= storage_bits || v & ((1u32 << (storage_bits - bits)) - 1) == 0
}

struct Roles {
    storage_bits: u8,
    narrow: bool,
    channels: usize,
    alpha: Option<usize>,
    padding: Option<usize>,
}

impl Roles {
    const fn of(spec: &PixelFormat) -> Result<Self, ExactDepthError> {
        if !spec.layout.packing.is_full() {
            return Err(ExactDepthError::UnsupportedStorage);
        }
        let storage_bits = match spec.layout.sample {
            SampleType::U8 => 8,
            SampleType::U16 => 16,
            _ => return Err(ExactDepthError::UnsupportedStorage),
        };
        let channels = spec.layout.order.channels();
        let (alpha, padding) = match spec.layout.order {
            ChannelOrder::Gray | ChannelOrder::Rgb | ChannelOrder::Bgr => (None, None),
            ChannelOrder::GrayAlpha | ChannelOrder::Rgba | ChannelOrder::Bgra => {
                if matches!(
                    spec.color.alpha(),
                    AlphaMode::Straight | AlphaMode::Premultiplied | AlphaMode::Opaque
                ) {
                    (Some(channels - 1), None)
                } else {
                    (None, Some(channels - 1))
                }
            }
            ChannelOrder::Rgbx | ChannelOrder::Bgrx => (None, Some(channels - 1)),
            _ => return Err(ExactDepthError::UnsupportedLayout),
        };
        let narrow = match spec.color {
            ColorSpace::Rgb { range, .. } | ColorSpace::Gray { range, .. } => {
                matches!(range, SignalRange::Narrow)
            }
            ColorSpace::Icc { .. } => false,
            _ => return Err(ExactDepthError::UnsupportedLayout),
        };
        Ok(Self {
            storage_bits,
            narrow,
            channels,
            alpha,
            padding,
        })
    }

    fn for_each(&self, view: &PixelView<'_>, mut visit: impl FnMut(bool, u32) -> bool) -> bool {
        let samples = view.width() as usize * self.channels;
        for y in 0..view.height() {
            let row = view.row(y);
            for i in 0..samples {
                let channel = i % self.channels;
                if Some(channel) == self.padding {
                    continue;
                }
                let v = if self.storage_bits == 8 {
                    u32::from(row[i])
                } else {
                    u32::from(u16::from_ne_bytes([row[2 * i], row[2 * i + 1]]))
                };
                if !visit(Some(channel) == self.alpha, v) {
                    return false;
                }
            }
        }
        true
    }
}

struct GridScan {
    storage_bits: u8,
    narrow: bool,
    candidates: u32,
    or: u32,
}

impl GridScan {
    fn new(storage_bits: u8, narrow: bool) -> Self {
        Self {
            storage_bits,
            narrow,
            candidates: (1u32 << (storage_bits - 1)) - 1,
            or: 0,
        }
    }

    fn push(&mut self, v: u32) {
        if self.narrow {
            self.or |= v;
            return;
        }
        let mut remaining = self.candidates;
        while remaining != 0 {
            let d = remaining.trailing_zeros() as u8 + 1;
            remaining &= remaining - 1;
            if !on_full_grid(v, self.storage_bits, d) {
                self.candidates &= !(1u32 << (d - 1));
            }
        }
    }

    fn settled(&self) -> bool {
        if self.narrow {
            self.or & 1 != 0
        } else {
            self.candidates == 0
        }
    }

    fn bits(&self) -> u8 {
        if self.narrow {
            let zeros = if self.or == 0 {
                16
            } else {
                self.or.trailing_zeros() as u8
            };
            self.storage_bits
                .saturating_sub(zeros)
                .clamp(8, self.storage_bits)
        } else if self.candidates == 0 {
            self.storage_bits
        } else {
            self.candidates.trailing_zeros() as u8 + 1
        }
    }
}
