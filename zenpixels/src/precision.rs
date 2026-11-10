//! Shared precision facts and capacity accounting for codec format offers.
//!
//! This model describes declared depth, alpha and range. It does not prove
//! lossless processing, gamut preservation or pixel equality. Selection policy
//! and chain planning live in `zenpixels-convert::precision`.

use crate::{ChannelType, PixelDescriptor, SignalRange, TransferFunction};

/// How much information the source really had, which no later edit changes.
///
/// This differs from the descriptor of the current samples: an 8-bit JPEG
/// decoded into linear `f32` still has an 8-bit source origin. Resizing can
/// create values outside its original code grid; this record does not prove
/// that encoding the processed values in 8 bits preserves them. It differs from
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

    /// Update the declared capacity model after selecting an output.
    /// This does not establish pixel equality or account for image processing.
    pub fn after_conversion(mut self, offer: &ConversionOffer) -> Self {
        let effect = ConversionEffect::of(&self, offer);
        self.bits = effect.bits_kept;
        self.extended &= !effect.range_dropped;
        if !offer.output.has_alpha() {
            self.alpha = AlphaContent::Opaque;
        }
        self
    }

    /// Declared source capacity in bits. For a float source, its mantissa width.
    /// This is not measured precision or exact-depth evidence for processed values.
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
    /// Caller-reported peak bytes for this route, if known. Unknown is not zero:
    /// a memory limit refuses it. The caller defines the accounting scope;
    /// chain planning does not infer overlapping lifetimes or worker/cache costs.
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

    /// Whether the modeled bit capacity, alpha and range survive.
    /// This is not a pixel-equality or gamut-preservation guarantee.
    pub const fn preserves_modeled_capacity(self) -> bool {
        self.bits_dropped == 0 && !self.alpha_dropped && !self.range_dropped
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
