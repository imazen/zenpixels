//! Which conversions may drop what the source really carried, and who decides.
//!
//! Bits that arrived in the decoded image are not dropped unless someone asked for that by name.
//! Bits that were only ever storage can go without anyone asking. A direct conversion between two
//! named descriptors doesn't come here: naming both ends is consent. This module is for the places
//! where something *chooses* a format: encode adapters, pipeline working formats, decoder exits.
//!
//! It works on descriptors and stated facts and reads no samples. Only
//! [`ConversionPolicy::plan_chain`] allocates. See `docs/precision-policy-design.md`.
//!
//! ```
//! use zenpixels::precision::{ConversionOffer, ConversionPolicy, OutputChoice, SourcePrecision};
//! use zenpixels::{PixelDescriptor, SignalRange, TransferFunction};
//!
//! // An 8-bit source now held in f32. Going back to 8 bits loses nothing that arrived.
//! let eight = SourcePrecision::declared(8, TransferFunction::Srgb, SignalRange::Full).unwrap();
//! let policy = ConversionPolicy::new();
//! let offers = [ConversionOffer::new(PixelDescriptor::RGB8_SRGB)];
//! let record = policy.resolve_preset(&eight, OutputChoice::Hidden, &offers).unwrap();
//! assert!(record.effect.is_lossless());
//!
//! // A real 16-bit source, and nobody named the narrowing. Refused.
//! let sixteen = SourcePrecision::from_storage(PixelDescriptor::RGB16_SRGB);
//! assert!(policy.resolve_preset(&sixteen, OutputChoice::Hidden, &offers).is_err());
//! ```

use core::error::Error;
use core::fmt;

use alloc::vec::Vec;
use whereat::{At, ResultAtExt};

use crate::{ChannelType, PixelDescriptor, SignalRange, TransferFunction};

/// How much information the source really had, which no later edit changes.
///
/// This differs from the descriptor of the current samples: a JPEG held in `f32` linear still has
/// 8 significant bits after a resize. It also differs from
/// [`ExactDepth`](crate::sample::ExactDepth), which says where the current samples sit and which
/// any arithmetic ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourcePrecision {
    bits: u8,
    transfer: TransferFunction,
    range: SignalRange,
    extended: bool,
    alpha: AlphaContent,
}

/// What is known about a source's alpha channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AlphaContent {
    /// The source has no alpha channel, or a scan found every sample opaque.
    Opaque,
    /// A scan found a translucent sample, or the source declares it uses alpha.
    Translucent,
    /// The source has an alpha channel and nobody has looked. Treated as translucent.
    Unknown,
}

impl SourcePrecision {
    /// Nothing is known beyond the current samples, so all of their storage counts as real.
    ///
    /// Float samples may leave `[0, 1]`, so they count as extended range, as PQ and HLG do. Alpha
    /// is [`AlphaContent::Unknown`] when the descriptor has it.
    pub fn from_storage(descriptor: PixelDescriptor) -> Self {
        let storage = descriptor.channel_type();
        let transfer = descriptor.transfer();
        Self {
            bits: storage_capacity(storage),
            transfer,
            range: descriptor.signal_range,
            extended: storage.is_float() || is_hdr_transfer(transfer),
            alpha: if descriptor.may_have_transparency() {
                AlphaContent::Unknown
            } else {
                AlphaContent::Opaque
            },
        }
    }

    /// A decoder or a scan states the source's integer depth and how its codes were spaced.
    ///
    /// Pass the source's own transfer and range, not the current samples': a JPEG held in `f32`
    /// linear is `declared(8, Srgb, Full)`. Alpha starts [`AlphaContent::Opaque`]; use
    /// [`with_alpha`](Self::with_alpha) when the source has it.
    ///
    /// Only state what is known. A container's claim such as PNG `sBIT` isn't known until
    /// [`ExactDepth::holds_for`](crate::sample::ExactDepth::holds_for) has verified it, and that is
    /// the caller's choice because it costs a whole pass.
    ///
    /// `None` unless `bits` is between 1 and 64.
    pub const fn declared(
        bits: u8,
        transfer: TransferFunction,
        range: SignalRange,
    ) -> Option<Self> {
        if bits == 0 || bits > 64 {
            return None;
        }
        Some(Self {
            bits,
            transfer,
            range,
            extended: is_hdr_transfer(transfer),
            alpha: AlphaContent::Opaque,
        })
    }

    /// State what is known about the source's alpha channel.
    #[must_use]
    pub const fn with_alpha(mut self, alpha: AlphaContent) -> Self {
        self.alpha = alpha;
        self
    }

    /// What a composite of this source and `other` carries: the most of each.
    ///
    /// An 8-bit source on a real 16-bit canvas has 16 significant bits from then on.
    #[must_use]
    pub const fn joined_with(self, other: Self) -> Self {
        let deeper = if other.bits > self.bits { other } else { self };
        Self {
            extended: self.extended || other.extended,
            alpha: match (self.alpha, other.alpha) {
                (AlphaContent::Translucent, _) | (_, AlphaContent::Translucent) => {
                    AlphaContent::Translucent
                }
                (AlphaContent::Unknown, _) | (_, AlphaContent::Unknown) => AlphaContent::Unknown,
                _ => AlphaContent::Opaque,
            },
            ..deeper
        }
    }

    /// What is left of this source after a step through `offer`.
    fn after(mut self, offer: &ConversionOffer) -> Self {
        let effect = ConversionEffect::of(&self, offer);
        self.bits = effect.bits_kept;
        self.extended &= !effect.range_dropped;
        if !offer.output.has_alpha() {
            self.alpha = AlphaContent::Opaque;
        }
        self
    }

    /// Significant bits per sample. For a float source, its mantissa width.
    pub const fn significant_bits(&self) -> u8 {
        self.bits
    }
}

/// Who chose the output format. This decides whether a narrowing has consent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OutputChoice {
    /// The caller named an output format that isn't the source's own. Choosing it is consent to
    /// what it can't hold, and the record says what went.
    CallerNamed,
    /// A library chose: an encode adapter, a pipeline's working format, automatic format selection,
    /// or a decoder's exit. Also a caller's request for the source's own format, where the
    /// path that keeps the data is the one they meant.
    Hidden,
}

/// One format a conversion could produce, with what it costs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ConversionOffer {
    /// The format this route produces.
    pub output: PixelDescriptor,
    /// Alpha is composited onto a matte, so leaving it out of the output drops nothing.
    pub matte: bool,
    /// Peak bytes for this route, if known. Unknown is not zero: a memory limit refuses it.
    pub peak_bytes: Option<u64>,
}

impl ConversionOffer {
    /// An offer with no matte and unknown costs.
    pub const fn new(output: PixelDescriptor) -> Self {
        Self {
            output,
            matte: false,
            peak_bytes: None,
        }
    }

    /// Alpha is composited onto a matte on this route.
    #[must_use]
    pub const fn with_matte(mut self) -> Self {
        self.matte = true;
        self
    }

    /// State this route's peak bytes.
    #[must_use]
    pub const fn with_peak_bytes(mut self, bytes: u64) -> Self {
        self.peak_bytes = Some(bytes);
        self
    }
}

/// What a conversion to one output keeps of a source and what it drops.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct ConversionEffect {
    /// Significant bits that survive.
    pub bits_kept: u8,
    /// Significant bits that don't.
    pub bits_dropped: u8,
    /// The source may be translucent, and the output has no alpha and no matte.
    pub alpha_dropped: bool,
    /// The source may leave `[0, 1]` or is PQ or HLG, and the output can't carry that.
    pub range_dropped: bool,
}

impl ConversionEffect {
    /// What converting `source` to `offer` keeps and drops.
    pub fn of(source: &SourcePrecision, offer: &ConversionOffer) -> Self {
        let output = offer.output;
        let capacity = output_capacity(source, output);
        let bits_kept = source.bits.min(capacity);
        Self {
            bits_kept,
            bits_dropped: source.bits - bits_kept,
            alpha_dropped: source.alpha != AlphaContent::Opaque
                && !output.has_alpha()
                && !offer.matte,
            range_dropped: source.extended && !holds_extended_range(output),
        }
    }

    /// Whether nothing the source carried is lost, so the conversion raises no event.
    pub const fn is_lossless(self) -> bool {
        self.bits_dropped == 0 && !self.alpha_dropped && !self.range_dropped
    }
}

/// What a preset does when no resolver answers.
///
/// The one that drops bits says so in its name. It doesn't drop alpha or dynamic range: that takes
/// a resolver.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ConversionPreset {
    /// The first offer that loses nothing, in offer order. If none does, a format the caller named
    /// takes the one that keeps the most bits, and anything else refuses.
    #[default]
    KeepPrecision,
    /// The offer with the smallest known peak, even if it drops bits.
    DropPrecisionForMemory,
}

/// Who settled a conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DecidedBy {
    /// A [`ConversionResolver`] chose.
    Resolver,
    /// The preset chose.
    Preset,
}

/// The conversion that was chosen and anything it drops.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ConversionRecord {
    /// Index of the chosen offer.
    pub index: usize,
    /// The chosen format.
    pub output: PixelDescriptor,
    /// What it keeps and drops.
    pub effect: ConversionEffect,
    /// Who chose it.
    pub decided_by: DecidedBy,
}

/// A planned chain: a [`ConversionRecord`] per node, in order.
///
/// An effect counts what earlier steps dropped: a later, deeper format doesn't bring bits back.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ChainPlan {
    /// The choice at each node.
    pub steps: Vec<ConversionRecord>,
    /// The largest peak any step states, or `None` if a step states none.
    pub peak_bytes: Option<u64>,
}

/// A chain that can't be planned, and the node that stopped it.
#[derive(Debug)]
#[non_exhaustive]
pub struct ChainFailure<E> {
    /// Index of the node in the chain.
    pub node: usize,
    /// Why nothing at that node would do.
    pub failure: ConversionFailure<E>,
}

impl<E: fmt::Display> fmt::Display for ChainFailure<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "at chain node {}: {}", self.node, self.failure)
    }
}

impl<E: Error + 'static> Error for ChainFailure<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.failure)
    }
}

/// An expected refusal, with what the caller needs in order to change the request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConversionRefusal {
    /// Nothing was offered.
    NoOffers,
    /// Every offer keeps fewer bits than the required floor, or its peak is over the limit.
    NoneAdmitted {
        /// The bits the request requires, if it set a floor.
        min_bits: Option<u8>,
        /// The limit the request sets, if it set one.
        max_peak_bytes: Option<u64>,
    },
    /// Only a narrowing was left, and nobody named it.
    DropsPrecision {
        /// The source's significant bits.
        significant_bits: u8,
        /// The most any admitted offer keeps.
        best_kept: u8,
        /// The offer that keeps them.
        output: PixelDescriptor,
    },
    /// Every admitted offer drops alpha from a source that may use it, with no matte.
    DropsAlpha {
        /// The first such offer.
        output: PixelDescriptor,
    },
    /// Every admitted offer drops extended range or HDR.
    DropsRange {
        /// The first such offer.
        output: PixelDescriptor,
    },
    /// The depth that keeps the data doesn't fit, and a narrower one does. Both costs are here.
    MemoryOrPrecision {
        /// The peak-byte limit.
        limit: u64,
        /// Peak bytes of the cheapest offer that keeps the data.
        keep_bytes: u64,
        /// Peak bytes of the cheapest offer under the limit.
        narrow_bytes: u64,
        /// Bits that offer drops.
        narrow_drops: u8,
    },
    /// A preset needs a cost that no admitted offer states.
    CostUnknown,
    /// The resolver refused.
    ResolverRefused,
    /// The resolver chose an index it wasn't offered.
    InvalidChoice {
        /// The index it returned.
        index: usize,
    },
}

impl fmt::Display for ConversionRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoOffers => f.write_str("no conversion was offered"),
            Self::NoneAdmitted {
                min_bits,
                max_peak_bytes,
            } => write!(
                f,
                "no offer meets the requirements (minimum bits {min_bits:?}, peak-byte limit {max_peak_bytes:?})"
            ),
            Self::DropsPrecision {
                significant_bits,
                best_kept,
                output,
            } => write!(
                f,
                "the source has {significant_bits} significant bits and the best offer, {:?}, keeps {best_kept}; nobody named that narrowing",
                output.pixel_format()
            ),
            Self::DropsAlpha { output } => write!(
                f,
                "the source may be translucent and {:?} has no alpha or matte",
                output.pixel_format()
            ),
            Self::DropsRange { output } => write!(
                f,
                "the source has extended range and {:?} can't carry it",
                output.pixel_format()
            ),
            Self::MemoryOrPrecision {
                limit,
                keep_bytes,
                narrow_bytes,
                narrow_drops,
            } => write!(
                f,
                "keeping the data needs {keep_bytes} bytes against a limit of {limit}; {narrow_bytes} bytes fits but drops {narrow_drops} significant bits"
            ),
            Self::CostUnknown => f.write_str("the preset needs a cost no admitted offer states"),
            Self::ResolverRefused => f.write_str("the conversion resolver refused"),
            Self::InvalidChoice { index } => {
                write!(
                    f,
                    "the conversion resolver chose offer {index}, which it wasn't offered"
                )
            }
        }
    }
}

impl Error for ConversionRefusal {}

/// A conversion that didn't get decided.
#[derive(Debug)]
#[non_exhaustive]
pub enum ConversionFailure<E> {
    /// An expected refusal.
    Refused(ConversionRefusal),
    /// The resolver's own failure.
    Resolver(E),
}

impl<E: fmt::Display> fmt::Display for ConversionFailure<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(refusal) => fmt::Display::fmt(refusal, f),
            Self::Resolver(error) => fmt::Display::fmt(error, f),
        }
    }
}

impl<E: Error + 'static> Error for ConversionFailure<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Refused(refusal) => Some(refusal),
            Self::Resolver(error) => Some(error),
        }
    }
}

/// One admitted offer and what it would do to the source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConversionChoice<'a> {
    index: usize,
    offer: &'a ConversionOffer,
    effect: ConversionEffect,
}

impl<'a> ConversionChoice<'a> {
    /// The index to return in [`ConversionDecision::Choose`].
    pub const fn index(self) -> usize {
        self.index
    }

    /// The offer.
    pub const fn offer(self) -> &'a ConversionOffer {
        self.offer
    }

    /// What it keeps and drops.
    pub const fn effect(self) -> ConversionEffect {
        self.effect
    }
}

/// What a resolver is asked. Requirements have already removed the offers it doesn't see.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct ConversionQuestion<'a> {
    /// What the source really carried.
    pub source: &'a SourcePrecision,
    /// Who chose the output format.
    pub choice: OutputChoice,
    offers: &'a [ConversionOffer],
    policy: ConversionPolicy,
}

impl<'a> ConversionQuestion<'a> {
    /// A question over `offers`, for a caller that holds its own requirements.
    pub const fn new(
        source: &'a SourcePrecision,
        choice: OutputChoice,
        offers: &'a [ConversionOffer],
        policy: ConversionPolicy,
    ) -> Self {
        Self {
            source,
            choice,
            offers,
            policy,
        }
    }

    /// Offers that passed the requirements, in offer order.
    pub fn admitted(&self) -> impl Iterator<Item = ConversionChoice<'a>> + use<'a> {
        let source = *self.source;
        let policy = self.policy;
        self.offers
            .iter()
            .enumerate()
            .filter_map(move |(index, offer)| {
                let effect = ConversionEffect::of(&source, offer);
                policy.admits(offer, effect).then_some(ConversionChoice {
                    index,
                    offer,
                    effect,
                })
            })
    }
}

/// A resolver's answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConversionDecision {
    /// Use this admitted offer. This is a named opt-in: whatever it drops is allowed, and recorded.
    Choose(usize),
    /// Pass to the next resolver, or to the preset if there is none.
    Defer,
    /// Refuse the conversion.
    Refuse,
}

/// Chooses among what requirements left. Closures implement this.
///
/// A resolver can't bring back an offer a requirement removed, and it can defer, which makes
/// resolvers composable: each one answers or passes to the next.
pub trait ConversionResolver {
    /// The resolver's own failure, kept with its trace.
    type Error: Error + Send + Sync + 'static;
    /// Answer one question.
    fn resolve(
        &mut self,
        question: ConversionQuestion<'_>,
    ) -> Result<ConversionDecision, At<Self::Error>>;
}

impl<F, E> ConversionResolver for F
where
    F: FnMut(ConversionQuestion<'_>) -> Result<ConversionDecision, At<E>>,
    E: Error + Send + Sync + 'static,
{
    type Error = E;
    fn resolve(&mut self, question: ConversionQuestion<'_>) -> Result<ConversionDecision, At<E>> {
        self(question)
    }
}

/// Requirements and a preset. Requirements bind everyone; the preset answers if no resolver does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct ConversionPolicy {
    min_bits: Option<u8>,
    max_peak_bytes: Option<u64>,
    preset: ConversionPreset,
}

impl ConversionPolicy {
    /// No requirements, and [`ConversionPreset::KeepPrecision`].
    pub const fn new() -> Self {
        Self {
            min_bits: None,
            max_peak_bytes: None,
            preset: ConversionPreset::KeepPrecision,
        }
    }

    /// Remove any offer that keeps fewer than `bits` significant bits, or fewer than the source
    /// has if that is less. No resolver can bring it back.
    #[must_use]
    pub const fn with_min_bits(mut self, bits: u8) -> Self {
        self.min_bits = Some(bits);
        self
    }

    /// Remove any offer over `bytes`, or whose peak isn't known. No resolver can bring it back.
    #[must_use]
    pub const fn with_max_peak_bytes(mut self, bytes: u64) -> Self {
        self.max_peak_bytes = Some(bytes);
        self
    }

    /// What answers if no resolver does.
    #[must_use]
    pub const fn with_preset(mut self, preset: ConversionPreset) -> Self {
        self.preset = preset;
        self
    }

    fn admits(&self, offer: &ConversionOffer, effect: ConversionEffect) -> bool {
        let bits = match self.min_bits {
            Some(floor) => effect.bits_dropped == 0 || effect.bits_kept >= floor,
            None => true,
        };
        let memory = match self.max_peak_bytes {
            Some(limit) => offer.peak_bytes.is_some_and(|peak| peak <= limit),
            None => true,
        };
        bits && memory
    }

    /// Settle a conversion with the preset alone.
    ///
    /// # Errors
    ///
    /// [`ConversionRefusal`], when nothing admitted satisfies the preset.
    #[track_caller]
    pub fn resolve_preset(
        &self,
        source: &SourcePrecision,
        choice: OutputChoice,
        offers: &[ConversionOffer],
    ) -> Result<ConversionRecord, At<ConversionRefusal>> {
        let question = ConversionQuestion::new(source, choice, offers, *self);
        self.check_admitted(&question).at()?;
        self.apply_preset(&question).at()
    }

    /// Choose a format at every node of a chain, so that estimating and running a job read one
    /// answer.
    ///
    /// `nodes` holds each node's candidates; the last node's are the outputs, under `last`, and
    /// every other node is [`OutputChoice::Hidden`]. `start` is the format the chain receives.
    /// `resolver` is asked once per node, and an answer pins that node.
    ///
    /// Everywhere else, [`ConversionPreset::KeepPrecision`] uses only candidates that lose nothing,
    /// with the same consent as [`resolve_preset`](Self::resolve_preset) at the last node, and
    /// [`ConversionPreset::DropPrecisionForMemory`] uses any that keep alpha and range. Among the
    /// plans left, this takes the lowest peak, then the fewest format changes, then offer order.
    /// An unknown peak ranks behind any known one.
    ///
    /// # Errors
    ///
    /// [`ChainFailure`] naming the first node with nothing usable, and
    /// [`ConversionRefusal::NoOffers`] at node 0 for an empty chain.
    #[track_caller]
    pub fn plan_chain<R: ConversionResolver + ?Sized>(
        &self,
        source: &SourcePrecision,
        start: PixelDescriptor,
        nodes: &[&[ConversionOffer]],
        last: OutputChoice,
        resolver: &mut R,
    ) -> Result<ChainPlan, At<ChainFailure<R::Error>>> {
        if nodes.is_empty() {
            return Err(whereat::at!(ChainFailure {
                node: 0,
                failure: ConversionFailure::Refused(ConversionRefusal::NoOffers),
            }));
        }
        // The candidates each node may use. A node with one candidate has settled what later nodes
        // receive, so they're asked about that and not about the original source.
        let mut usable: Vec<Vec<Candidate<'_>>> = Vec::with_capacity(nodes.len());
        let mut received = *source;
        for (node, offers) in nodes.iter().enumerate() {
            let choice = if node + 1 == nodes.len() {
                last
            } else {
                OutputChoice::Hidden
            };
            let question = ConversionQuestion::new(&received, choice, offers, *self);
            let candidates = self
                .node_candidates(&question, offers, resolver)
                .map_err_at(|failure| ChainFailure { node, failure })?;
            if let [only] = candidates.as_slice() {
                received = received.after(only.offer);
            }
            usable.push(candidates);
        }

        let path = fewest_changes(start, &usable);
        let mut received = *source;
        let mut plan = ChainPlan {
            steps: Vec::with_capacity(path.len()),
            peak_bytes: Some(0),
        };
        let (mut range_lost, mut alpha_lost) = (false, false);
        for candidate in path {
            let effect = ConversionEffect::of(&received, candidate.offer);
            range_lost |= effect.range_dropped;
            alpha_lost |= effect.alpha_dropped;
            plan.steps.push(ConversionRecord {
                index: candidate.index,
                output: candidate.offer.output,
                effect: ConversionEffect {
                    bits_kept: effect.bits_kept,
                    bits_dropped: source.bits - effect.bits_kept,
                    range_dropped: range_lost,
                    alpha_dropped: alpha_lost,
                },
                decided_by: candidate.decided_by,
            });
            received = received.after(candidate.offer);
            plan.peak_bytes = match (plan.peak_bytes, candidate.offer.peak_bytes) {
                (Some(a), Some(b)) => Some(a.max(b)),
                _ => None,
            };
        }
        Ok(plan)
    }

    fn node_candidates<'o, R: ConversionResolver + ?Sized>(
        &self,
        question: &ConversionQuestion<'_>,
        offers: &'o [ConversionOffer],
        resolver: &mut R,
    ) -> Result<Vec<Candidate<'o>>, At<ConversionFailure<R::Error>>> {
        self.check_admitted(question)
            .map_err_at(ConversionFailure::Refused)?;
        let mut candidates = Vec::new();
        match resolver
            .resolve(*question)
            .at()
            .map_err_at(ConversionFailure::Resolver)?
        {
            ConversionDecision::Choose(index) => {
                if !question.admitted().any(|c| c.index == index) {
                    return Err(whereat::at!(ConversionFailure::Refused(
                        ConversionRefusal::InvalidChoice { index }
                    )));
                }
                candidates.push(Candidate {
                    index,
                    offer: &offers[index],
                    decided_by: DecidedBy::Resolver,
                });
            }
            ConversionDecision::Refuse => {
                return Err(whereat::at!(ConversionFailure::Refused(
                    ConversionRefusal::ResolverRefused
                )));
            }
            ConversionDecision::Defer => {
                candidates.extend(
                    question
                        .admitted()
                        .filter(|c| self.usable_in_chain(c.effect))
                        .map(|c| Candidate {
                            index: c.index,
                            offer: &offers[c.index],
                            decided_by: DecidedBy::Preset,
                        }),
                );
                if candidates.is_empty() {
                    // The single-node rule says why, or, for a caller's named output, what to take.
                    let record = self
                        .apply_preset(question)
                        .map_err_at(ConversionFailure::Refused)?;
                    candidates.push(Candidate {
                        index: record.index,
                        offer: &offers[record.index],
                        decided_by: DecidedBy::Preset,
                    });
                }
            }
        }
        Ok(candidates)
    }

    fn usable_in_chain(&self, effect: ConversionEffect) -> bool {
        match self.preset {
            ConversionPreset::KeepPrecision => effect.is_lossless(),
            ConversionPreset::DropPrecisionForMemory => {
                !effect.alpha_dropped && !effect.range_dropped
            }
        }
    }

    fn check_admitted(
        &self,
        question: &ConversionQuestion<'_>,
    ) -> Result<(), At<ConversionRefusal>> {
        if question.offers.is_empty() {
            return Err(whereat::at!(ConversionRefusal::NoOffers));
        }
        if question.admitted().next().is_none() {
            return Err(whereat::at!(ConversionRefusal::NoneAdmitted {
                min_bits: self.min_bits,
                max_peak_bytes: self.max_peak_bytes,
            }));
        }
        Ok(())
    }

    fn apply_preset(
        &self,
        question: &ConversionQuestion<'_>,
    ) -> Result<ConversionRecord, At<ConversionRefusal>> {
        let chosen = match self.preset {
            ConversionPreset::KeepPrecision => keep_precision(self, question)?,
            ConversionPreset::DropPrecisionForMemory => smallest_peak(question)?,
        };
        Ok(ConversionRecord {
            index: chosen.index,
            output: chosen.offer.output,
            effect: chosen.effect,
            decided_by: DecidedBy::Preset,
        })
    }
}

#[derive(Clone, Copy)]
struct Candidate<'a> {
    index: usize,
    offer: &'a ConversionOffer,
    decided_by: DecidedBy,
}

impl Candidate<'_> {
    fn peak(&self) -> u64 {
        self.offer.peak_bytes.unwrap_or(u64::MAX)
    }
}

/// The path with the lowest possible peak and, among those, the fewest format changes.
fn fewest_changes<'a>(start: PixelDescriptor, usable: &[Vec<Candidate<'a>>]) -> Vec<Candidate<'a>> {
    // The lowest peak any path can have is the worst node's best candidate.
    let ceiling = usable
        .iter()
        .map(|node| node.iter().map(Candidate::peak).min().unwrap_or(u64::MAX))
        .max()
        .unwrap_or(0);
    const UNREACHABLE: usize = usize::MAX;
    let changes = |from: PixelDescriptor, to: PixelDescriptor| {
        usize::from(from.pixel_format() != to.pixel_format() || from.transfer() != to.transfer())
    };

    // best[i][c]: changes needed to reach candidate c of node i, and the candidate before it.
    let mut best: Vec<Vec<(usize, usize)>> = Vec::with_capacity(usable.len());
    for (i, node) in usable.iter().enumerate() {
        let mut row = Vec::with_capacity(node.len());
        for candidate in node {
            let output = candidate.offer.output;
            let mut entry = (UNREACHABLE, 0);
            if candidate.peak() <= ceiling {
                if i == 0 {
                    entry = (changes(start, output), 0);
                } else {
                    for (p, before) in usable[i - 1].iter().enumerate() {
                        let reached: usize = best[i - 1][p].0;
                        if reached != UNREACHABLE {
                            let total = reached + changes(before.offer.output, output);
                            if total < entry.0 {
                                entry = (total, p);
                            }
                        }
                    }
                }
            }
            row.push(entry);
        }
        best.push(row);
    }

    let mut at = 0;
    if let Some(last) = best.last() {
        for (c, entry) in last.iter().enumerate() {
            if entry.0 < last[at].0 {
                at = c;
            }
        }
    }
    let mut path = Vec::with_capacity(usable.len());
    for i in (0..usable.len()).rev() {
        path.push(usable[i][at]);
        at = best[i][at].1;
    }
    path.reverse();
    path
}

fn keep_precision<'a>(
    policy: &ConversionPolicy,
    question: &ConversionQuestion<'a>,
) -> Result<ConversionChoice<'a>, At<ConversionRefusal>> {
    if let Some(keeps) = question.admitted().find(|c| c.effect.is_lossless()) {
        return Ok(keeps);
    }
    // Alpha isn't a rendition of the same image: what was under it shows. A named format doesn't
    // consent to that, so only offers that keep it are candidates, even for a caller.
    let Some(best) = best_without_alpha_loss(question) else {
        // Callers have checked that something was admitted.
        return Err(whereat::at!(match question.admitted().next() {
            Some(first) => ConversionRefusal::DropsAlpha {
                output: first.offer.output,
            },
            None => ConversionRefusal::NoOffers,
        }));
    };
    if question.choice == OutputChoice::CallerNamed {
        return Ok(best);
    }
    if let Some(refusal) = memory_refusal(policy, question, &best) {
        return Err(whereat::at!(refusal));
    }
    if best.effect.bits_dropped != 0 {
        return Err(whereat::at!(ConversionRefusal::DropsPrecision {
            significant_bits: question.source.bits,
            best_kept: best.effect.bits_kept,
            output: best.offer.output,
        }));
    }
    Err(whereat::at!(ConversionRefusal::DropsRange {
        output: best.offer.output,
    }))
}

/// The admitted offer that keeps the most, in offer order among equals.
fn best_without_alpha_loss<'a>(question: &ConversionQuestion<'a>) -> Option<ConversionChoice<'a>> {
    let mut best: Option<ConversionChoice<'a>> = None;
    for choice in question.admitted().filter(|c| !c.effect.alpha_dropped) {
        let better = best.is_none_or(|b| {
            let rank = |c: &ConversionChoice<'_>| (!c.effect.range_dropped, c.effect.bits_kept);
            rank(&choice) > rank(&b)
        });
        if better {
            best = Some(choice);
        }
    }
    best
}

/// When a limit removed what would have kept the data, the refusal names both costs.
fn memory_refusal(
    policy: &ConversionPolicy,
    question: &ConversionQuestion<'_>,
    best: &ConversionChoice<'_>,
) -> Option<ConversionRefusal> {
    let limit = policy.max_peak_bytes?;
    let source = question.source;
    let keep_bytes = question
        .offers
        .iter()
        .filter(|offer| ConversionEffect::of(source, offer).is_lossless())
        .filter_map(|offer| offer.peak_bytes)
        .filter(|&peak| peak > limit)
        .min()?;
    Some(ConversionRefusal::MemoryOrPrecision {
        limit,
        keep_bytes,
        narrow_bytes: best.offer.peak_bytes?,
        narrow_drops: best.effect.bits_dropped,
    })
}

fn smallest_peak<'a>(
    question: &ConversionQuestion<'a>,
) -> Result<ConversionChoice<'a>, At<ConversionRefusal>> {
    let mut best: Option<(u64, ConversionChoice<'a>)> = None;
    let mut blocked = None;
    for choice in question.admitted() {
        if choice.effect.alpha_dropped || choice.effect.range_dropped {
            blocked.get_or_insert(choice);
            continue;
        }
        if let Some(value) = choice.offer.peak_bytes {
            if best.is_none_or(|(smallest, _)| value < smallest) {
                best = Some((value, choice));
            }
        }
    }
    if let Some((_, choice)) = best {
        return Ok(choice);
    }
    match blocked {
        Some(choice) if choice.effect.alpha_dropped => {
            Err(whereat::at!(ConversionRefusal::DropsAlpha {
                output: choice.offer.output,
            }))
        }
        Some(choice) => Err(whereat::at!(ConversionRefusal::DropsRange {
            output: choice.offer.output,
        })),
        None => Err(whereat::at!(ConversionRefusal::CostUnknown)),
    }
}

const fn is_hdr_transfer(transfer: TransferFunction) -> bool {
    matches!(transfer, TransferFunction::Pq | TransferFunction::Hlg)
}

/// Whether the output can carry samples outside `[0, 1]` of SDR. Floats are unbounded. PQ and HLG
/// reach their peak in any storage. A linear or gamma integer stops at 1.0.
const fn holds_extended_range(output: PixelDescriptor) -> bool {
    output.channel_type().is_float()
        || is_hdr_transfer(output.transfer())
        || matches!(output.transfer(), TransferFunction::Unknown)
}

/// Integer codes a sample type holds exactly in `[0, 1]`. For floats, the mantissa including its
/// implicit bit: codes spaced `1 / (2^n - 1)` stay distinct near 1.0 up to that width.
#[allow(unreachable_patterns)]
const fn storage_capacity(storage: ChannelType) -> u8 {
    match storage {
        ChannelType::U8 | ChannelType::I8 => 8,
        ChannelType::U16 | ChannelType::I16 => 16,
        ChannelType::U32 | ChannelType::I32 => 32,
        ChannelType::U64 | ChannelType::I64 => 64,
        ChannelType::F16 => 11,
        ChannelType::F32 => 24,
        ChannelType::F64 => 53,
        _ => 0,
    }
}

/// How many of the source's bits `output` can hold, given how the source's codes were spaced.
///
/// Storage alone answers most cases. An integer output with a different curve doesn't: linear
/// storage crowds a gamma source's darks together, and a curve crowds a linear source's highlights.
/// The tables are the largest source depth whose codes all stay distinct. A test derives each
/// entry from the curve.
const fn output_capacity(source: &SourcePrecision, output: PixelDescriptor) -> u8 {
    let storage = output.channel_type();
    let capacity = storage_capacity(storage);
    if storage.is_float() {
        return capacity;
    }
    let to = output.transfer();
    let from = source.transfer;
    let curved = if matches!(to, TransferFunction::Linear)
        && !matches!(from, TransferFunction::Linear | TransferFunction::Unknown)
    {
        gamma_into_linear(from, capacity)
    } else if matches!(from, TransferFunction::Linear)
        && !matches!(to, TransferFunction::Linear | TransferFunction::Unknown)
    {
        linear_into_gamma(to, capacity)
    } else {
        capacity
    };
    // A full-range source doesn't fit on a narrow-range grid of the same width.
    if matches!(output.signal_range, SignalRange::Narrow)
        && matches!(source.range, SignalRange::Full)
    {
        curved.saturating_sub(1)
    } else {
        curved
    }
}

/// The deepest gamma-coded source that linear integer storage of `width` bits keeps distinct.
/// Governed by the darkest step, which is the curve's smallest.
#[allow(unreachable_patterns)]
const fn gamma_into_linear(from: TransferFunction, width: u8) -> u8 {
    let row = match from {
        TransferFunction::Srgb => [4, 12, 28, 60],
        TransferFunction::Bt709 => [5, 13, 29, 61],
        TransferFunction::Gamma22 => [3, 7, 14, 29],
        TransferFunction::Hlg => [3, 7, 15, 31],
        TransferFunction::Pq => [1, 3, 12, 20],
        _ => return width,
    };
    row[width_row(width)]
}

/// The deepest linear source that gamma integer storage of `width` bits keeps distinct. Governed
/// by the step at white, where these curves are flattest.
#[allow(unreachable_patterns)]
const fn linear_into_gamma(to: TransferFunction, width: u8) -> u8 {
    let row = match to {
        TransferFunction::Srgb | TransferFunction::Bt709 | TransferFunction::Gamma22 => {
            [6, 14, 30, 62]
        }
        TransferFunction::Hlg => [5, 13, 29, 61],
        TransferFunction::Pq => [4, 12, 28, 60],
        _ => return width,
    };
    row[width_row(width)]
}

const fn width_row(width: u8) -> usize {
    match width {
        8 => 0,
        16 => 1,
        32 => 2,
        _ => 3,
    }
}
