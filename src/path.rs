//! Conversion path solver — find the optimal (source, operation, output) path.
//!
//! Given a source format, an operation category, and an output format, the solver
//! finds the cheapest conversion chain that satisfies quality constraints.
//!
//! The algorithm:
//! 1. Get the operation's [`OpRequirement`]
//! 2. Generate candidate working formats
//! 3. For each candidate: compute source→working cost + suitability + working→output cost
//! 4. Filter by quality threshold
//! 5. Pick lowest total cost among qualifying paths

use alloc::vec::Vec;

use crate::PixelDescriptor;
use crate::negotiate::{
    ConversionCost, Provenance, conversion_cost_with_provenance, suitability_loss, weighted_score,
};
use crate::op_format::OpCategory;
use crate::registry::{CodecFormats, FormatEntry};

/// Perceptual loss buckets, calibrated against CIEDE2000 measurements.
///
/// Promoted from test-only type to public API for use with [`QualityThreshold`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LossBucket {
    /// ΔE < 0.5 — below just-noticeable difference. Model loss ≤ 10.
    Lossless,
    /// ΔE 0.5–2.0 — visible only in side-by-side comparison. Model loss 11–50.
    NearLossless,
    /// ΔE 2.0–5.0 — minor visible differences. Model loss 51–150.
    LowLoss,
    /// ΔE 5.0–15.0 — clearly visible quality difference. Model loss 151–400.
    Moderate,
    /// ΔE > 15.0 — severe quality degradation. Model loss > 400.
    High,
}

impl LossBucket {
    /// Classify a model loss value into a bucket.
    pub fn from_model_loss(loss: u16) -> Self {
        if loss <= 10 {
            Self::Lossless
        } else if loss <= 50 {
            Self::NearLossless
        } else if loss <= 150 {
            Self::LowLoss
        } else if loss <= 400 {
            Self::Moderate
        } else {
            Self::High
        }
    }

    /// Maximum model loss value for this bucket.
    pub fn max_loss(self) -> u16 {
        match self {
            Self::Lossless => 10,
            Self::NearLossless => 50,
            Self::LowLoss => 150,
            Self::Moderate => 400,
            Self::High => u16::MAX,
        }
    }
}

/// Quality threshold for path selection.
#[derive(Clone, Copy, Debug)]
pub enum QualityThreshold {
    /// Zero information loss (ULP-proven round-trip where available).
    Lossless,
    /// Below JND (ΔE < 0.5, model loss ≤ 10).
    SubPerceptual,
    /// Minimal visible loss (ΔE < 2.0, model loss ≤ 50).
    NearLossless,
    /// Fastest path within the given loss bucket.
    MaxBucket(LossBucket),
}

impl QualityThreshold {
    /// Maximum allowed total loss for this threshold.
    fn max_loss(self) -> u16 {
        match self {
            Self::Lossless => 0,
            Self::SubPerceptual => 10,
            Self::NearLossless => 50,
            Self::MaxBucket(bucket) => bucket.max_loss(),
        }
    }
}

/// A complete conversion path through the pipeline.
#[derive(Clone, Debug)]
pub struct ConversionPath {
    /// Source format (from decoder).
    pub source_format: PixelDescriptor,
    /// Convert source to this format before the operation.
    pub working_format: PixelDescriptor,
    /// Convert working format to this for the encoder.
    pub output_format: PixelDescriptor,
    /// Cost of source → working conversion.
    pub source_to_working: ConversionCost,
    /// Suitability loss of the working format for the operation.
    pub working_suitability: u16,
    /// Cost of working → output conversion.
    pub working_to_output: ConversionCost,
    /// Total weighted score (lower is better).
    pub total_score: u32,
    /// Total loss across all conversions + suitability.
    pub total_loss: u16,
    /// Whether this path is proven lossless by ULP test.
    pub proven_lossless: bool,
}

impl ConversionPath {
    /// The loss bucket this path falls into.
    pub fn loss_bucket(&self) -> LossBucket {
        LossBucket::from_model_loss(self.total_loss)
    }
}

/// Find the optimal conversion path for a (source, operation, output) triple.
///
/// Returns `None` if no path satisfies the quality threshold.
///
/// # Arguments
///
/// * `source` - Source pixel format (from decoder)
/// * `provenance` - Origin precision of the source data
/// * `operation` - What operation will be performed
/// * `output` - Target pixel format (for encoder)
/// * `threshold` - Maximum acceptable quality loss
pub fn optimal_path(
    source: PixelDescriptor,
    provenance: Provenance,
    operation: OpCategory,
    output: PixelDescriptor,
    threshold: QualityThreshold,
) -> Option<ConversionPath> {
    let intent = operation.to_intent();
    let candidates = operation.candidate_working_formats(source);
    let max_loss = threshold.max_loss();

    let mut best: Option<ConversionPath> = None;

    for working in candidates {
        let s2w = conversion_cost_with_provenance(source, working, provenance);
        let suit = suitability_loss(working, intent);
        let w2o = conversion_cost_with_provenance(
            working,
            output,
            provenance_after_operation(provenance, working),
        );

        let total_loss = s2w.loss.saturating_add(suit).saturating_add(w2o.loss);

        // Filter by threshold.
        if total_loss > max_loss {
            continue;
        }

        let total_effort = s2w.effort as u32 + w2o.effort as u32;
        let total_score = weighted_score(total_effort, total_loss as u32 + suit as u32, intent);

        let path = ConversionPath {
            source_format: source,
            working_format: working,
            output_format: output,
            source_to_working: s2w,
            working_suitability: suit,
            working_to_output: w2o,
            total_score,
            total_loss,
            proven_lossless: false,
        };

        match &best {
            Some(current) if path.total_score < current.total_score => best = Some(path),
            None => best = Some(path),
            _ => {}
        }
    }

    best
}

/// Compute provenance after the operation transforms data.
///
/// After an operation processes data in the working format, the provenance
/// origin depth is the *working format's* depth (the operation "consumes"
/// the original precision and produces new data at working precision).
fn provenance_after_operation(original: Provenance, working: PixelDescriptor) -> Provenance {
    Provenance::with_origin(working.channel_type(), original.origin_primaries)
}

/// An entry in the full path matrix.
#[derive(Clone, Debug)]
pub struct PathEntry {
    /// Source codec name.
    pub source_codec: &'static str,
    /// Source pixel format.
    pub source_format: PixelDescriptor,
    /// Effective bits of the source data.
    pub source_effective_bits: u8,
    /// Operation category.
    pub operation: OpCategory,
    /// Output codec name.
    pub output_codec: &'static str,
    /// Output pixel format.
    pub output_format: PixelDescriptor,
    /// The optimal conversion path (None if no path within threshold).
    pub path: Option<ConversionPath>,
}

/// Generate optimal paths for all (source, op, output) triples across codec pairs.
///
/// This produces the complete conversion matrix. With 9 codecs × ~5 avg formats ×
/// 13 ops × 9 codecs × ~3 avg formats ≈ ~15,000 triples. Most collapse to a
/// handful of distinct working formats.
pub fn generate_path_matrix(
    source_codecs: &[&CodecFormats],
    operations: &[OpCategory],
    output_codecs: &[&CodecFormats],
    threshold: QualityThreshold,
) -> Vec<PathEntry> {
    let mut entries = Vec::new();

    for source_codec in source_codecs {
        for source_entry in source_codec.decode_outputs {
            let provenance = provenance_from_entry(source_entry);

            for &operation in operations {
                for output_codec in output_codecs {
                    for output_entry in output_codec.encode_inputs {
                        let path = optimal_path(
                            source_entry.descriptor,
                            provenance,
                            operation,
                            output_entry.descriptor,
                            threshold,
                        );

                        entries.push(PathEntry {
                            source_codec: source_codec.name,
                            source_format: source_entry.descriptor,
                            source_effective_bits: source_entry.effective_bits,
                            operation,
                            output_codec: output_codec.name,
                            output_format: output_entry.descriptor,
                            path,
                        });
                    }
                }
            }
        }
    }

    entries
}

/// Derive provenance from a codec's format entry.
///
/// Uses `effective_bits` to determine the origin depth: if effective_bits ≤ 8,
/// origin is U8; if ≤ 16, origin is U16; otherwise F32.
fn provenance_from_entry(entry: &FormatEntry) -> Provenance {
    use crate::ChannelType;

    let origin_depth = if entry.effective_bits <= 8 {
        ChannelType::U8
    } else if entry.effective_bits <= 16 {
        ChannelType::U16
    } else {
        ChannelType::F32
    };
    Provenance::with_origin_depth(origin_depth)
}

/// Summary statistics for a path matrix.
#[derive(Clone, Debug, Default)]
pub struct MatrixStats {
    /// Total number of (source, op, output) triples evaluated.
    pub total_triples: usize,
    /// Number of triples with a valid path within threshold.
    pub paths_found: usize,
    /// Number of triples where no path met the threshold.
    pub no_path: usize,
    /// Distribution of paths by loss bucket.
    pub by_bucket: [usize; 5],
    /// Number of distinct working formats used across all paths.
    pub distinct_working_formats: usize,
}

/// Compute summary statistics for a path matrix.
pub fn matrix_stats(entries: &[PathEntry]) -> MatrixStats {
    use alloc::collections::BTreeSet;

    let mut stats = MatrixStats {
        total_triples: entries.len(),
        ..Default::default()
    };
    let mut working_formats = BTreeSet::new();

    for entry in entries {
        match &entry.path {
            Some(path) => {
                stats.paths_found += 1;
                let bucket_idx = match path.loss_bucket() {
                    LossBucket::Lossless => 0,
                    LossBucket::NearLossless => 1,
                    LossBucket::LowLoss => 2,
                    LossBucket::Moderate => 3,
                    LossBucket::High => 4,
                };
                stats.by_bucket[bucket_idx] += 1;

                // Encode working format as bytes for BTreeSet (PixelDescriptor doesn't impl Ord).
                let wf = path.working_format;
                let alpha_byte = match wf.alpha() {
                    None => 0u8,
                    Some(a) => a as u8,
                };
                let key = (
                    wf.channel_type() as u8,
                    wf.layout() as u8,
                    alpha_byte,
                    wf.transfer() as u8,
                    wf.primaries as u8,
                );
                working_formats.insert(key);
            }
            None => stats.no_path += 1,
        }
    }

    stats.distinct_working_formats = working_formats.len();
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry;
    use crate::{AlphaMode, ChannelType, TransferFunction};

    #[test]
    fn passthrough_identity_is_lossless() {
        let src = PixelDescriptor::RGB8_SRGB;
        let provenance = Provenance::from_source(src);
        let path = optimal_path(
            src,
            provenance,
            OpCategory::Passthrough,
            src,
            QualityThreshold::Lossless,
        );
        assert!(
            path.is_some(),
            "passthrough identity should always find a path"
        );
        let path = path.unwrap();
        assert_eq!(path.working_format, src);
        assert_eq!(path.total_loss, 0);
    }

    #[test]
    fn resize_sharp_uses_f32_linear() {
        let src = PixelDescriptor::RGB8_SRGB;
        let provenance = Provenance::from_source(src);
        let path = optimal_path(
            src,
            provenance,
            OpCategory::ResizeSharp,
            PixelDescriptor::RGB8_SRGB,
            QualityThreshold::MaxBucket(LossBucket::Moderate),
        );
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.working_format.channel_type(), ChannelType::F32);
        assert_eq!(path.working_format.transfer(), TransferFunction::Linear);
    }

    #[test]
    fn jpeg_to_jpeg_passthrough() {
        let src = PixelDescriptor::RGB8_SRGB;
        let provenance = Provenance::with_origin_depth(ChannelType::U8);
        let path = optimal_path(
            src,
            provenance,
            OpCategory::Passthrough,
            PixelDescriptor::RGB8_SRGB,
            QualityThreshold::Lossless,
        );
        assert!(path.is_some());
        assert_eq!(path.unwrap().total_loss, 0);
    }

    #[test]
    fn composite_uses_premultiplied() {
        let src = PixelDescriptor::RGBA8_SRGB;
        let provenance = Provenance::from_source(src);
        let path = optimal_path(
            src,
            provenance,
            OpCategory::Composite,
            PixelDescriptor::RGBA8_SRGB,
            QualityThreshold::MaxBucket(LossBucket::Moderate),
        );
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.working_format.alpha(), Some(AlphaMode::Premultiplied));
    }

    #[test]
    fn loss_bucket_classification() {
        assert_eq!(LossBucket::from_model_loss(0), LossBucket::Lossless);
        assert_eq!(LossBucket::from_model_loss(10), LossBucket::Lossless);
        assert_eq!(LossBucket::from_model_loss(11), LossBucket::NearLossless);
        assert_eq!(LossBucket::from_model_loss(50), LossBucket::NearLossless);
        assert_eq!(LossBucket::from_model_loss(51), LossBucket::LowLoss);
        assert_eq!(LossBucket::from_model_loss(150), LossBucket::LowLoss);
        assert_eq!(LossBucket::from_model_loss(151), LossBucket::Moderate);
        assert_eq!(LossBucket::from_model_loss(400), LossBucket::Moderate);
        assert_eq!(LossBucket::from_model_loss(401), LossBucket::High);
    }

    #[test]
    fn generate_jpeg_to_jpeg_matrix() {
        let ops = [
            OpCategory::Passthrough,
            OpCategory::ResizeGentle,
            OpCategory::ResizeSharp,
        ];
        let matrix = generate_path_matrix(
            &[&registry::JPEG],
            &ops,
            &[&registry::JPEG],
            QualityThreshold::MaxBucket(LossBucket::Moderate),
        );

        // JPEG decode has 4 formats × 3 ops × 10 encode formats = 120 triples
        assert!(!matrix.is_empty());

        let stats = matrix_stats(&matrix);
        assert!(stats.paths_found > 0, "should find at least some paths");
    }

    #[test]
    fn full_matrix_produces_results() {
        let all_ops = [
            OpCategory::Passthrough,
            OpCategory::ResizeGentle,
            OpCategory::ResizeSharp,
        ];
        let codecs: Vec<&CodecFormats> = registry::ALL_CODECS.iter().copied().collect();
        let matrix = generate_path_matrix(
            &codecs,
            &all_ops,
            &codecs,
            QualityThreshold::MaxBucket(LossBucket::High),
        );

        let stats = matrix_stats(&matrix);
        assert!(stats.total_triples > 100, "should have many triples");
        assert!(stats.paths_found > 0, "should find paths");
        // Most triples should have valid paths with High threshold
        assert!(
            stats.paths_found as f64 / stats.total_triples as f64 > 0.5,
            "most triples should have valid paths: {}/{}",
            stats.paths_found,
            stats.total_triples
        );
    }

    #[test]
    fn quality_threshold_filters_correctly() {
        let src = PixelDescriptor::RGBF32_LINEAR;
        let provenance = Provenance::with_origin_depth(ChannelType::F32);

        // Strict lossless threshold: f32→u8 should not qualify
        let lossless_path = optimal_path(
            src,
            provenance,
            OpCategory::Passthrough,
            PixelDescriptor::RGB8_SRGB,
            QualityThreshold::Lossless,
        );

        // Relaxed threshold: should find a path
        let relaxed_path = optimal_path(
            src,
            provenance,
            OpCategory::Passthrough,
            PixelDescriptor::RGB8_SRGB,
            QualityThreshold::MaxBucket(LossBucket::Moderate),
        );

        // f32 origin → u8 has loss, so lossless should fail but relaxed should work
        assert!(lossless_path.is_none(), "f32→u8 should not be lossless");
        assert!(
            relaxed_path.is_some(),
            "f32→u8 should work with relaxed threshold"
        );
    }
}
