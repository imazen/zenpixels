//! Comparison and diagnostics derived from executable operators.

use crate::PixelDescriptor;

/// Numerical envelope of an executed step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum AccuracyContract {
    /// The operation moves or duplicates represented samples exactly.
    Exact,
    /// The operation quantizes to its destination grid.
    Quantized,
    /// Native color arithmetic.
    Computed,
    /// The kernel uses a sampled approximation of its function.
    Lookup,
    /// Semantic change such as channel loss, matte or rendering.
    Rendered,
    /// The backend owns the numerical contract.
    BackendDefined,
}

/// Native dispatch selected or deferred by an operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// Work and accuracy of one executable step.
#[derive(Clone, Copy, Debug)]
pub struct PlannedStepInfo {
    pub(crate) operation: &'static str,
    pub(crate) from: PixelDescriptor,
    pub(crate) to: PixelDescriptor,
    pub(crate) tier: KernelTier,
    pub(crate) accuracy: AccuracyContract,
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
    /// Active interface bytes per pixel, excluding internal accesses.
    pub fn interface_bytes_per_pixel(&self) -> u32 {
        (self.from.bytes_per_pixel() + self.to.bytes_per_pixel()) as u32
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct RouteRank {
    effects: u8,
    traffic: u64,
    relative_compute: u64,
}

impl RouteRank {
    pub(crate) fn add(&mut self, step: PlannedStepInfo) {
        if step.operation == "Identity" {
            return;
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
        self.effects != other.effects && self.effects & !other.effects == 0
    }

    fn cost(self, preference: u32, scratch: u64) -> (u64, u64, u64, u32) {
        (self.traffic, self.relative_compute, scratch, preference)
    }
}

// At most one winner per accuracy-effect set. The frontier retains
// incomparable sets until caller order can resolve the remaining tie.
pub(crate) struct RouteFrontier<T> {
    candidates: [Option<Ranked<T>>; 32],
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
        if self
            .candidates
            .iter()
            .flatten()
            .any(|item| item.rank.strict_subset_of(rank))
        {
            return;
        }
        for item in &mut self.candidates {
            if item
                .as_ref()
                .is_some_and(|other| rank.strict_subset_of(other.rank))
            {
                *item = None;
            }
        }
        let slot = &mut self.candidates[rank.effects as usize];
        if slot.as_ref().is_none_or(|other| {
            rank.cost(preference, scratch) < other.rank.cost(other.preference, other.scratch)
        }) {
            *slot = Some(Ranked {
                rank,
                preference,
                scratch,
                index,
                value,
            });
        }
    }

    pub(crate) fn finish(self) -> Option<T> {
        self.candidates
            .into_iter()
            .flatten()
            .min_by_key(|item| item.index)
            .map(|item| item.value)
    }
}
/// Nominal precision bits for a channel type (for ordering, not bit-exact).
///
/// F16 has 10 mantissa bits (~3.3 decimal digits) — between U8 (8 bits) and
/// U16 (16 bits).
#[allow(unreachable_patterns)] // non_exhaustive: future variants
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
