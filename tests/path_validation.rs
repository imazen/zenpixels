//! Path validation tests — verify the path solver produces correct results.
//!
//! These tests check that:
//! 1. The solver returns paths for common (source, op, output) triples
//! 2. Working formats satisfy operation requirements
//! 3. Cost predictions are consistent with calibrated loss buckets
//! 4. The full matrix produces sensible results

use zenpixels::{
    generate_path_matrix, matrix_stats, optimal_path, CodecFormats, ConversionPath, LossBucket,
    OpCategory, Provenance, QualityThreshold,
};
use zenpixels::registry;
use zencodec_types::{AlphaMode, ChannelType, PixelDescriptor, TransferFunction};

// ═══════════════════════════════════════════════════════════════════════
// Common triples: JPEG → op → JPEG
// ═══════════════════════════════════════════════════════════════════════

fn jpeg_path(
    op: OpCategory,
    threshold: QualityThreshold,
) -> Option<ConversionPath> {
    optimal_path(
        PixelDescriptor::RGB8_SRGB,
        Provenance::with_origin_depth(ChannelType::U8),
        op,
        PixelDescriptor::RGB8_SRGB,
        threshold,
    )
}

#[test]
fn jpeg_passthrough_jpeg_is_lossless() {
    let path = jpeg_path(OpCategory::Passthrough, QualityThreshold::Lossless);
    assert!(path.is_some(), "JPEG passthrough should find a path");
    let path = path.unwrap();
    assert_eq!(path.total_loss, 0);
    assert_eq!(path.working_format, PixelDescriptor::RGB8_SRGB);
}

#[test]
fn jpeg_resize_gentle_jpeg() {
    let path = jpeg_path(
        OpCategory::ResizeGentle,
        QualityThreshold::MaxBucket(LossBucket::Moderate),
    );
    assert!(path.is_some(), "JPEG resize Mitchell should find a path");
    let path = path.unwrap();
    // Resize gentle prefers linear f32.
    assert_eq!(path.working_format.transfer, TransferFunction::Linear);
}

#[test]
fn jpeg_resize_sharp_jpeg() {
    let path = jpeg_path(
        OpCategory::ResizeSharp,
        QualityThreshold::MaxBucket(LossBucket::Moderate),
    );
    assert!(path.is_some(), "JPEG resize Lanczos should find a path");
    let path = path.unwrap();
    // Sharp resize requires f32 linear.
    assert_eq!(path.working_format.channel_type, ChannelType::F32);
    assert_eq!(path.working_format.transfer, TransferFunction::Linear);
}

#[test]
fn jpeg_blur_jpeg() {
    let path = jpeg_path(
        OpCategory::Blur,
        QualityThreshold::MaxBucket(LossBucket::Moderate),
    );
    assert!(path.is_some());
}

#[test]
fn jpeg_sharpen_jpeg() {
    let path = jpeg_path(
        OpCategory::Sharpen,
        QualityThreshold::MaxBucket(LossBucket::Moderate),
    );
    assert!(path.is_some());
}

#[test]
fn jpeg_color_matrix_jpeg() {
    let path = jpeg_path(
        OpCategory::ColorMatrix,
        QualityThreshold::MaxBucket(LossBucket::Moderate),
    );
    assert!(path.is_some(), "JPEG color matrix should find a path");
    let path = path.unwrap();
    // ColorMatrix works in sRGB u8 — should use source format directly.
    assert_eq!(path.working_format, PixelDescriptor::RGB8_SRGB);
}

// ═══════════════════════════════════════════════════════════════════════
// Common triples: PNG RGBA → op → WebP
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn png_rgba_composite_webp() {
    let path = optimal_path(
        PixelDescriptor::RGBA8_SRGB,
        Provenance::with_origin_depth(ChannelType::U8),
        OpCategory::Composite,
        PixelDescriptor::RGBA8_SRGB,
        QualityThreshold::MaxBucket(LossBucket::Moderate),
    );
    assert!(path.is_some(), "PNG RGBA composite should find a path");
    let path = path.unwrap();
    // Compositing requires premultiplied alpha in f32 linear.
    assert_eq!(path.working_format.alpha, AlphaMode::Premultiplied);
    assert_eq!(path.working_format.channel_type, ChannelType::F32);
    assert_eq!(path.working_format.transfer, TransferFunction::Linear);
}

#[test]
fn png_rgb_resize_webp() {
    let path = optimal_path(
        PixelDescriptor::RGB8_SRGB,
        Provenance::with_origin_depth(ChannelType::U8),
        OpCategory::ResizeGentle,
        PixelDescriptor::RGB8_SRGB,
        QualityThreshold::MaxBucket(LossBucket::Moderate),
    );
    assert!(path.is_some());
}

// ═══════════════════════════════════════════════════════════════════════
// Cross-codec paths
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn png16_passthrough_png16() {
    // 16-bit PNG passthrough should be lossless.
    let path = optimal_path(
        PixelDescriptor::RGB16_SRGB,
        Provenance::with_origin_depth(ChannelType::U16),
        OpCategory::Passthrough,
        PixelDescriptor::RGB16_SRGB,
        QualityThreshold::Lossless,
    );
    assert!(path.is_some());
    assert_eq!(path.unwrap().total_loss, 0);
}

#[test]
fn png16_to_jpeg_passthrough() {
    // 16-bit PNG → JPEG (u8): lossy (depth truncation).
    let path = optimal_path(
        PixelDescriptor::RGB16_SRGB,
        Provenance::with_origin_depth(ChannelType::U16),
        OpCategory::Passthrough,
        PixelDescriptor::RGB8_SRGB,
        QualityThreshold::Lossless,
    );
    assert!(path.is_none(), "u16→u8 should not be lossless");

    // But should work with relaxed threshold.
    let path = optimal_path(
        PixelDescriptor::RGB16_SRGB,
        Provenance::with_origin_depth(ChannelType::U16),
        OpCategory::Passthrough,
        PixelDescriptor::RGB8_SRGB,
        QualityThreshold::NearLossless,
    );
    assert!(path.is_some(), "u16→u8 should work with NearLossless threshold");
}

#[test]
fn jxl_f32_resize_jxl_f32() {
    // JXL f32 → resize → JXL f32: working in f32 linear is identity for source.
    let path = optimal_path(
        PixelDescriptor::RGBF32_LINEAR,
        Provenance::with_origin_depth(ChannelType::F32),
        OpCategory::ResizeSharp,
        PixelDescriptor::RGBF32_LINEAR,
        QualityThreshold::SubPerceptual,
    );
    assert!(path.is_some());
    let path = path.unwrap();
    // Source is already f32 linear, so working format should be f32 linear.
    assert_eq!(path.working_format, PixelDescriptor::RGBF32_LINEAR);
    // No conversion cost (source already matches working format).
    assert_eq!(path.source_to_working.effort, 0);
    assert_eq!(path.source_to_working.loss, 0);
}

// ═══════════════════════════════════════════════════════════════════════
// Working format satisfies operation requirements
// ═══════════════════════════════════════════════════════════════════════

fn assert_working_satisfies_op(path: &ConversionPath, op: OpCategory) {
    let req = op.requirement();

    if let Some(tf) = req.transfer {
        assert_eq!(
            path.working_format.transfer, tf,
            "working format transfer {:?} doesn't match requirement {:?} for {:?}",
            path.working_format.transfer, tf, op
        );
    }

    if req.requires_float {
        assert_eq!(
            path.working_format.channel_type,
            ChannelType::F32,
            "working format should be f32 for {:?}",
            op
        );
    }

    if let Some(alpha) = req.alpha {
        if path.working_format.layout.has_alpha() {
            assert_eq!(
                path.working_format.alpha, alpha,
                "working format alpha {:?} doesn't match requirement {:?} for {:?}",
                path.working_format.alpha, alpha, op
            );
        }
    }
}

#[test]
fn all_ops_satisfy_requirements_for_jpeg() {
    let ops = [
        OpCategory::Passthrough,
        OpCategory::ResizeGentle,
        OpCategory::ResizeSharp,
        OpCategory::Blur,
        OpCategory::Sharpen,
        OpCategory::ColorMatrix,
        OpCategory::Arithmetic,
    ];

    for &op in &ops {
        let path = jpeg_path(op, QualityThreshold::MaxBucket(LossBucket::High));
        if let Some(ref p) = path {
            assert_working_satisfies_op(p, op);
        }
    }
}

#[test]
fn composite_op_satisfies_for_rgba() {
    let path = optimal_path(
        PixelDescriptor::RGBA8_SRGB,
        Provenance::from_source(PixelDescriptor::RGBA8_SRGB),
        OpCategory::Composite,
        PixelDescriptor::RGBA8_SRGB,
        QualityThreshold::MaxBucket(LossBucket::High),
    );
    assert!(path.is_some());
    assert_working_satisfies_op(path.as_ref().unwrap(), OpCategory::Composite);
}

#[test]
fn tonemap_op_satisfies() {
    let path = optimal_path(
        PixelDescriptor::RGBF32_LINEAR,
        Provenance::from_source(PixelDescriptor::RGBF32_LINEAR),
        OpCategory::Tonemap,
        PixelDescriptor::RGB8_SRGB,
        QualityThreshold::MaxBucket(LossBucket::High),
    );
    assert!(path.is_some());
    assert_working_satisfies_op(path.as_ref().unwrap(), OpCategory::Tonemap);
}

// ═══════════════════════════════════════════════════════════════════════
// Provenance-aware cost predictions
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn u8_origin_roundtrip_is_lossless_passthrough() {
    // JPEG u8 → f32 (for resize) → u8 (for JPEG encode).
    // With u8 provenance, the f32→u8 step should be lossless.
    let path = optimal_path(
        PixelDescriptor::RGB8_SRGB,
        Provenance::with_origin_depth(ChannelType::U8),
        OpCategory::ResizeGentle,
        PixelDescriptor::RGB8_SRGB,
        QualityThreshold::MaxBucket(LossBucket::NearLossless),
    );
    assert!(path.is_some());
    let path = path.unwrap();
    // The working→output conversion should have 0 loss (u8 origin round-trip).
    assert_eq!(
        path.working_to_output.loss, 0,
        "u8 origin → f32 → u8 should report 0 loss, got {}",
        path.working_to_output.loss
    );
}

#[test]
fn f32_origin_to_u8_is_lossy() {
    // True f32 data → u8 should report loss.
    let path = optimal_path(
        PixelDescriptor::RGBF32_LINEAR,
        Provenance::with_origin_depth(ChannelType::F32),
        OpCategory::Passthrough,
        PixelDescriptor::RGB8_SRGB,
        QualityThreshold::MaxBucket(LossBucket::Moderate),
    );
    assert!(path.is_some());
    let path = path.unwrap();
    assert!(path.total_loss > 0, "f32→u8 with f32 origin should be lossy");
}

// ═══════════════════════════════════════════════════════════════════════
// Full matrix generation
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn full_matrix_with_all_codecs_and_three_ops() {
    let ops = [
        OpCategory::Passthrough,
        OpCategory::ResizeGentle,
        OpCategory::ResizeSharp,
    ];
    let codecs: Vec<&CodecFormats> = registry::ALL_CODECS.iter().copied().collect();
    let matrix = generate_path_matrix(
        &codecs,
        &ops,
        &codecs,
        QualityThreshold::MaxBucket(LossBucket::High),
    );
    let stats = matrix_stats(&matrix);

    // Sanity checks.
    assert!(stats.total_triples > 1000, "expected >1000 triples, got {}", stats.total_triples);
    assert!(stats.paths_found > 500, "expected >500 paths, got {}", stats.paths_found);

    // Most passthrough identity triples should be lossless.
    let passthrough_lossless = matrix
        .iter()
        .filter(|e| e.operation == OpCategory::Passthrough)
        .filter(|e| e.source_format == e.output_format)
        .filter(|e| e.path.as_ref().is_some_and(|p| p.total_loss == 0))
        .count();
    assert!(passthrough_lossless > 0, "some passthrough identities should be lossless");

    // Print summary for debugging.
    println!("Full matrix: {} triples, {} paths found, {} no path",
        stats.total_triples, stats.paths_found, stats.no_path);
    println!("  By bucket: Lossless={}, NearLossless={}, LowLoss={}, Moderate={}, High={}",
        stats.by_bucket[0], stats.by_bucket[1], stats.by_bucket[2],
        stats.by_bucket[3], stats.by_bucket[4]);
    println!("  Distinct working formats: {}", stats.distinct_working_formats);
}

#[test]
fn full_matrix_all_13_ops() {
    let all_ops = [
        OpCategory::Passthrough,
        OpCategory::ResizeGentle,
        OpCategory::ResizeSharp,
        OpCategory::Blur,
        OpCategory::Sharpen,
        OpCategory::OklabSharpen,
        OpCategory::Composite,
        OpCategory::OklabAdjust,
        OpCategory::ColorMatrix,
        OpCategory::Tonemap,
        OpCategory::IccTransform,
        OpCategory::Quantize,
        OpCategory::Arithmetic,
    ];

    // Use a subset of codecs for speed (JPEG, PNG, WebP cover the main cases).
    let codecs: Vec<&CodecFormats> = vec![&registry::JPEG, &registry::PNG, &registry::WEBP];
    let matrix = generate_path_matrix(
        &codecs,
        &all_ops,
        &codecs,
        QualityThreshold::MaxBucket(LossBucket::High),
    );
    let stats = matrix_stats(&matrix);

    println!("All ops matrix: {} triples, {} paths, {} no path",
        stats.total_triples, stats.paths_found, stats.no_path);

    // Every operation should find at least some valid paths.
    for &op in &all_ops {
        let op_paths = matrix
            .iter()
            .filter(|e| e.operation == op)
            .filter(|e| e.path.is_some())
            .count();
        assert!(
            op_paths > 0,
            "OpCategory {:?} should find at least one valid path",
            op
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Regression: loss bucket consistency
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn lossless_paths_have_zero_total_loss() {
    let codecs: Vec<&CodecFormats> = registry::ALL_CODECS.iter().copied().collect();
    let matrix = generate_path_matrix(
        &codecs,
        &[OpCategory::Passthrough],
        &codecs,
        QualityThreshold::Lossless,
    );

    for entry in &matrix {
        if let Some(ref path) = entry.path {
            assert_eq!(
                path.total_loss, 0,
                "path from {} {:?} → {} {:?} marked lossless but has loss {}",
                entry.source_codec,
                entry.source_format,
                entry.output_codec,
                entry.output_format,
                path.total_loss
            );
        }
    }
}

#[test]
fn sub_perceptual_paths_within_threshold() {
    let matrix = generate_path_matrix(
        &[&registry::JPEG, &registry::PNG],
        &[OpCategory::Passthrough, OpCategory::ResizeGentle],
        &[&registry::JPEG, &registry::PNG],
        QualityThreshold::SubPerceptual,
    );

    for entry in &matrix {
        if let Some(ref path) = entry.path {
            assert!(
                path.total_loss <= 10,
                "sub-perceptual path has loss {} > 10 for {:?} → {:?} → {:?}",
                path.total_loss,
                entry.source_format,
                entry.operation,
                entry.output_format
            );
        }
    }
}
