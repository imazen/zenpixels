//! Comparison, accuracy lattice, semantic effect flags, and diagnostics derived
//! from executable operators.

use crate::PixelDescriptor;

/// Numerical envelope of an executed step, ordered from strongest (`Exact`) to
/// weakest (`BackendDefined`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum AccuracyContract {
    /// The operation moves or duplicates represented samples exactly.
    Exact = 0,
    /// The operation quantizes to its destination grid.
    Quantized = 1,
    /// Native floating-point color arithmetic.
    Computed = 2,
    /// The kernel uses a sampled LUT approximation of its function.
    Lookup = 3,
    /// Semantic change such as channel loss, matte or display rendering.
    Rendered = 4,
    /// External CMS backend owns the numerical contract.
    BackendDefined = 5,
}

/// Bitflags recording intentional semantic transformations along a compiled route.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SemanticEffects(u16);

#[cfg_attr(not(feature = "introspection"), allow(dead_code))]
impl SemanticEffects {
    /// No lossy or domain-changing semantic effects.
    pub const NONE: Self = Self(0);
    /// Out-of-range float values clamped to `[0, 1]`.
    pub const CLAMPED: Self = Self(1 << 0);
    /// Soft OKLch gamut chroma compression applied.
    pub const GAMUT_COMPRESSED: Self = Self(1 << 1);
    /// HDR-to-SDR tone mapping applied.
    pub const TONE_MAPPED: Self = Self(1 << 2);
    /// Alpha composited onto a solid background matte.
    pub const MATTE_APPLIED: Self = Self(1 << 3);
    /// Alpha channel discarded.
    pub const ALPHA_DISCARDED: Self = Self(1 << 4);
    /// RGB collapsed to single-channel luma.
    pub const LUMA_COLLAPSED: Self = Self(1 << 5);
    /// Bit depth narrowed (e.g. `U16 -> U8`, `F32 -> U8/U16/F16`).
    pub const DEPTH_NARROWED: Self = Self(1 << 6);
    /// Alpha association changed (`Straight <-> Premultiplied`).
    pub const ALPHA_REASSOCIATED: Self = Self(1 << 7);
    /// Explicit relabeling of an `Unknown` transfer function.
    pub const RELABELED_TRANSFER: Self = Self(1 << 8);

    /// Union of two effect sets.
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// True if `self` contains every flag in `other`.
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// True when no semantic effect flags are set.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Number of distinct semantic effect flags set.
    #[inline]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Severity penalty for default fidelity tie-breaking when no explicit
    /// [`ConvertIntent`](crate::ConvertIntent) is supplied, preventing destructive
    /// channel/depth loss from beating pure floating-point transfer math by slice order.
    #[inline]
    pub(crate) const fn fidelity_penalty(self) -> u32 {
        let mut p = 0u32;
        if self.contains(Self::LUMA_COLLAPSED) {
            p += 500;
        }
        if self.contains(Self::ALPHA_DISCARDED) {
            p += 400;
        }
        if self.contains(Self::TONE_MAPPED) {
            p += 250;
        }
        if self.contains(Self::DEPTH_NARROWED) {
            p += 150;
        }
        if self.contains(Self::MATTE_APPLIED) {
            p += 120;
        }
        if self.contains(Self::GAMUT_COMPRESSED) {
            p += 60;
        }
        if self.contains(Self::ALPHA_REASSOCIATED) {
            p += 25;
        }
        if self.contains(Self::CLAMPED) {
            p += 15;
        }
        if self.contains(Self::RELABELED_TRANSFER) {
            p += 5;
        }
        p
    }
}

/// Native dispatch selected or deferred by an operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum KernelTier {
    /// Ordinary scalar kernel.
    Scalar,
    /// x86-64 v3 matrix/LUT kernel.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    X86V3,
    /// The operator dispatches when called (possibly through a dependency).
    RuntimeDispatch,
    /// Backend-controlled path.
    Backend,
}

/// Work, accuracy, and semantic effects of one executable step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlannedStepInfo {
    pub(crate) operation: &'static str,
    pub(crate) from: PixelDescriptor,
    pub(crate) to: PixelDescriptor,
    pub(crate) tier: KernelTier,
    pub(crate) accuracy: AccuracyContract,
    pub(crate) semantic: SemanticEffects,
}

#[cfg(feature = "introspection")]
impl PlannedStepInfo {
    /// Operator name.
    pub const fn operation(&self) -> &'static str {
        self.operation
    }
    /// Actual input carrier.
    pub const fn from(&self) -> PixelDescriptor {
        self.from
    }
    /// Actual output carrier.
    pub const fn to(&self) -> PixelDescriptor {
        self.to
    }
    /// Dispatch category.
    pub const fn tier(&self) -> KernelTier {
        self.tier
    }
    /// Numerical guarantee category.
    pub const fn accuracy(&self) -> AccuracyContract {
        self.accuracy
    }
    /// Semantic effects introduced by this step.
    pub const fn semantic_effects(&self) -> SemanticEffects {
        self.semantic
    }
    /// Active interface bytes per pixel, excluding internal accesses.
    pub fn interface_bytes_per_pixel(&self) -> u32 {
        (self.from.bytes_per_pixel() + self.to.bytes_per_pixel()) as u32
    }
}

pub(crate) type TieKey = (u32, u16, u8, u64, u64, u64, u32, usize);

#[derive(Clone, Copy)]
pub(crate) struct RouteRank {
    pub(crate) effects: u8,
    pub(crate) accuracy: AccuracyContract,
    pub(crate) semantic: SemanticEffects,
    pub(crate) source_bits: u16,
    pub(crate) min_narrowed_bits: u16,
    pub(crate) traffic: u64,
    pub(crate) relative_compute: u64,
}

impl Default for RouteRank {
    fn default() -> Self {
        Self {
            effects: 0,
            accuracy: AccuracyContract::Exact,
            semantic: SemanticEffects::NONE,
            source_bits: 0,
            min_narrowed_bits: u16::MAX,
            traffic: 0,
            relative_compute: 0,
        }
    }
}

impl RouteRank {
    pub(crate) fn add(&mut self, step: PlannedStepInfo) {
        if self.source_bits == 0 {
            self.source_bits = channel_bits(step.from.channel_type());
        }
        let dst_bits = channel_bits(step.to.channel_type());
        if dst_bits < self.source_bits {
            self.min_narrowed_bits = self.min_narrowed_bits.min(dst_bits);
        }
        self.semantic = self.semantic.union(step.semantic);
        if step.operation == "Identity" || step.operation == "RelabelTransfer" {
            return;
        }
        if step.accuracy > self.accuracy {
            self.accuracy = step.accuracy;
        }
        let effect = match step.accuracy {
            AccuracyContract::Exact => 0,
            AccuracyContract::Quantized => 1,
            AccuracyContract::Computed => 2,
            AccuracyContract::Lookup => 2 | 4,
            AccuracyContract::Rendered => 8,
            AccuracyContract::BackendDefined => 16,
        };
        self.effects |= effect;
        self.traffic = self
            .traffic
            .saturating_add((step.from.bytes_per_pixel() + step.to.bytes_per_pixel()) as u64);
        let compute = match (step.accuracy, step.tier) {
            (AccuracyContract::Exact, _) => 1,
            (_, KernelTier::X86V3) => 2,
            (_, KernelTier::Scalar) => 4,
            (_, KernelTier::RuntimeDispatch) => 3,
            (_, KernelTier::Backend) => 8,
        };
        self.relative_compute = self.relative_compute.saturating_add(compute);
    }

    fn strict_subset_of(self, other: Self) -> bool {
        let a_bits = (u32::from(self.effects) << 16) | u32::from(self.semantic.0);
        let b_bits = (u32::from(other.effects) << 16) | u32::from(other.semantic.0);
        (a_bits & !b_bits) == 0
            && self.min_narrowed_bits >= other.min_narrowed_bits
            && (a_bits != b_bits || self.min_narrowed_bits > other.min_narrowed_bits)
    }

    fn same_mask(self, other: Self) -> bool {
        self.effects == other.effects
            && self.semantic == other.semantic
            && self.min_narrowed_bits == other.min_narrowed_bits
    }

    pub(crate) fn tie_key(self, preference: u32, scratch: u64, index: usize) -> TieKey {
        (
            self.semantic.fidelity_penalty(),
            u16::MAX - self.min_narrowed_bits,
            self.accuracy as u8,
            self.traffic,
            self.relative_compute,
            scratch,
            preference,
            index,
        )
    }
}

/// Bounded Pareto frontier retaining non-dominated routes and resolving incomparable
/// effect sets by semantic fidelity penalty before falling back to caller order.
pub(crate) struct RouteFrontier<T> {
    candidates: [Option<Ranked<T>>; 32],
    len: usize,
}

struct Ranked<T> {
    rank: RouteRank,
    preference: u32,
    scratch: u64,
    index: usize,
    value: T,
}

impl<T> RouteFrontier<T> {
    pub(crate) fn new() -> Self {
        Self {
            candidates: core::array::from_fn(|_| None),
            len: 0,
        }
    }

    pub(crate) fn offer(
        &mut self,
        rank: RouteRank,
        preference: u32,
        scratch: u64,
        index: usize,
        value: T,
    ) {
        for existing in self.candidates[..self.len].iter().flatten() {
            if existing.rank.strict_subset_of(rank) {
                return;
            }
            if existing.rank.same_mask(rank)
                && existing
                    .rank
                    .tie_key(existing.preference, existing.scratch, existing.index)
                    <= rank.tie_key(preference, scratch, index)
            {
                return;
            }
        }
        for slot in &mut self.candidates[..self.len] {
            if let Some(other) = slot
                && (rank.strict_subset_of(other.rank)
                    || (other.rank.same_mask(rank)
                        && rank.tie_key(preference, scratch, index)
                            < other
                                .rank
                                .tie_key(other.preference, other.scratch, other.index)))
            {
                *slot = None;
            }
        }
        let entry = Ranked {
            rank,
            preference,
            scratch,
            index,
            value,
        };
        if let Some(empty) = self.candidates[..self.len].iter_mut().find(|s| s.is_none()) {
            *empty = Some(entry);
        } else if self.len < self.candidates.len() {
            self.candidates[self.len] = Some(entry);
            self.len += 1;
        } else if let Some(worst) = self
            .candidates
            .iter_mut()
            .filter(|s| s.is_some())
            .max_by_key(|s| {
                let r = s.as_ref().unwrap();
                r.rank.tie_key(r.preference, r.scratch, r.index)
            })
            && entry
                .rank
                .tie_key(entry.preference, entry.scratch, entry.index)
                < {
                    let r = worst.as_ref().unwrap();
                    r.rank.tie_key(r.preference, r.scratch, r.index)
                }
        {
            *worst = Some(entry);
        }
    }

    pub(crate) fn finish(self) -> Option<T> {
        self.candidates
            .into_iter()
            .flatten()
            .min_by_key(|item| item.rank.tie_key(item.preference, item.scratch, item.index))
            .map(|item| item.value)
    }
}

/// Nominal precision bits for a channel type (for ordering, not bit-exact).
///
/// F16 has 10 mantissa bits (~3.3 decimal digits) — between U8 (8 bits) and
/// U16 (16 bits).
#[allow(unreachable_patterns)]
pub(crate) fn channel_bits(ct: crate::ChannelType) -> u16 {
    match ct {
        crate::ChannelType::U8 | crate::ChannelType::I8 => 8,
        crate::ChannelType::F16 => 11, // 10 mantissa + 1 implicit
        crate::ChannelType::U16 | crate::ChannelType::I16 => 16,
        crate::ChannelType::F32 | crate::ChannelType::U32 | crate::ChannelType::I32 => 32,
        crate::ChannelType::U64 | crate::ChannelType::I64 | crate::ChannelType::F64 => 64,
        _ => 0,
    }
}
