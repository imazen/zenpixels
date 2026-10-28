//! The exactness rule a codec plan reports (proposed zencodec
//! `CodedSamples::depth_exact`), checked against actual native narrowing.
//!
//! The rule uses only `ExactDepth`'s public queries. Every level of each evidence
//! grid, built by an independent integer widening and measured with
//! `ExactDepth::measure`, is narrowed to each coded depth with the codec-facing
//! `CodeMap` and widened back. The round trip must be the identity exactly when
//! the rule calls the coded depth exact. Full-range grids are not ordered by bit
//! count, and color and alpha evidence are judged separately.

use zenpixels::{
    ChannelType, PixelDescriptor, PixelSlice, SignalRange,
    sample::{ExactDepth, SampleEncoding},
};
use zenpixels_convert::codes::{CodeMap, CodeRole};

/// Proposed `CodedSamples::depth_exact`: coded color and, when coded, alpha.
fn depth_exact(evidence: ExactDepth, color_bits: u8, alpha_bits: Option<u8>) -> bool {
    evidence.color_exact_at(color_bits) && alpha_bits.is_none_or(|a| evidence.alpha_exact_at(a))
}

fn encoding(storage: ChannelType, bits: u8) -> SampleEncoding {
    SampleEncoding::new(storage, bits, 0).unwrap()
}

/// Every canonical level of the `bits`-deep grid, by integer widening.
fn levels(storage_bits: u8, bits: u8, narrow: bool) -> Vec<u16> {
    let storage_max = (1u64 << storage_bits) - 1;
    let max = (1u64 << bits) - 1;
    (0..=max)
        .map(|q| {
            if narrow {
                (q << (storage_bits - bits)) as u16
            } else {
                let (whole, rest) = (q * storage_max / max, q * storage_max % max);
                (whole + u64::from(2 * rest > max)) as u16
            }
        })
        .collect()
}

/// Count of levels that native narrowing to `coded` bits and widening back changes.
fn changed(
    storage: ChannelType,
    storage_bits: u8,
    values: &[u16],
    coded: u8,
    role: CodeRole,
    range: SignalRange,
) -> usize {
    let down = CodeMap::new(
        encoding(storage, storage_bits),
        encoding(storage, coded),
        role,
        range,
    )
    .unwrap();
    let up = CodeMap::new(
        encoding(storage, coded),
        encoding(storage, storage_bits),
        role,
        range,
    )
    .unwrap();
    values
        .iter()
        .filter(|&&v| up.code(down.code(v)) != v)
        .count()
}

/// Measured evidence of gray (or gray-alpha) samples.
fn measured(storage_bits: u8, narrow: bool, samples: &[u16], with_alpha: bool) -> ExactDepth {
    let (descriptor, bytes): (PixelDescriptor, Vec<u8>) = match (storage_bits, with_alpha) {
        (8, false) => (
            PixelDescriptor::GRAY8,
            samples.iter().map(|&v| v as u8).collect(),
        ),
        (8, true) => (
            PixelDescriptor::GRAYA8,
            samples.iter().map(|&v| v as u8).collect(),
        ),
        (_, false) => (
            PixelDescriptor::GRAY16,
            samples.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        ),
        (_, true) => (
            PixelDescriptor::GRAYA16,
            samples.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        ),
    };
    let descriptor = descriptor.with_signal_range(if narrow {
        SignalRange::Narrow
    } else {
        SignalRange::Full
    });
    let pixels = samples.len() / if with_alpha { 2 } else { 1 };
    let view = PixelSlice::new(&bytes, pixels as u32, 1, bytes.len(), descriptor).unwrap();
    ExactDepth::measure(&view).unwrap()
}

#[test]
fn coded_color_depth_is_exact_exactly_when_native_narrowing_round_trips() {
    for (storage, storage_bits) in [(ChannelType::U16, 16u8), (ChannelType::U8, 8)] {
        for range in [SignalRange::Full, SignalRange::Narrow] {
            let narrow = range == SignalRange::Narrow;
            let lowest = if narrow { 8 } else { 1 };
            for content in lowest..=storage_bits {
                let values = levels(storage_bits, content, narrow);
                let evidence = measured(storage_bits, narrow, &values, false);
                assert_eq!(evidence.color_bits(), content, "{storage:?} {range:?}");
                for coded in lowest..=storage_bits {
                    let moved = changed(
                        storage,
                        storage_bits,
                        &values,
                        coded,
                        CodeRole::Intensity,
                        range,
                    );
                    assert_eq!(
                        moved == 0,
                        depth_exact(evidence, coded, None),
                        "{storage:?} {range:?}: {content}-bit content coded at {coded} changes {moved}"
                    );
                }
            }
        }
    }
}

#[test]
fn coded_alpha_depth_follows_the_full_scale_rule_in_both_ranges() {
    for (storage, storage_bits) in [(ChannelType::U16, 16u8), (ChannelType::U8, 8)] {
        for range in [SignalRange::Full, SignalRange::Narrow] {
            let narrow = range == SignalRange::Narrow;
            for content in 1..=storage_bits {
                let alpha = levels(storage_bits, content, false);
                // Pair every alpha level with a black color sample of the range.
                let black = if narrow {
                    16u16 << (storage_bits - 8)
                } else {
                    0
                };
                let samples: Vec<u16> = alpha.iter().flat_map(|&a| [black, a]).collect();
                let evidence = measured(storage_bits, narrow, &samples, true);
                assert_eq!(evidence.alpha_bits(), content, "{storage:?} {range:?}");
                for coded in 1..=storage_bits {
                    let moved =
                        changed(storage, storage_bits, &alpha, coded, CodeRole::Alpha, range);
                    assert_eq!(
                        moved == 0,
                        evidence.alpha_exact_at(coded),
                        "alpha {content} coded at {coded}"
                    );
                }
            }
        }
    }
}

#[test]
fn more_bits_are_not_more_exact_and_color_and_alpha_decide_separately() {
    let u16_full = |content| levels(16, content, false);
    let intensity = |values: &[u16], coded| {
        changed(
            ChannelType::U16,
            16,
            values,
            coded,
            CodeRole::Intensity,
            SignalRange::Full,
        )
    };
    // Eight-bit content: exact at 8 and 16 bits only; 10 and 12 move most levels.
    let eight = u16_full(8);
    let e8 = measured(16, false, &eight, false);
    for (coded, moved) in [(8u8, 0usize), (10, 252), (12, 240), (16, 0)] {
        assert_eq!(intensity(&eight, coded), moved, "8-bit content at {coded}");
        assert_eq!(depth_exact(e8, coded, None), moved == 0);
    }
    // Ten-bit content is not exact at 12; four-bit content is exact at 8 and 12, not 10.
    assert_eq!(intensity(&u16_full(10), 12), 972);
    let four = u16_full(4);
    let e4 = measured(16, false, &four, false);
    for coded in [8u8, 12, 16] {
        assert_eq!(intensity(&four, coded), 0);
        assert!(depth_exact(e4, coded, None));
    }
    assert!(intensity(&four, 10) > 0 && !depth_exact(e4, 10, None));
    // Ten-bit color with eight-bit alpha: one coded depth for both is not exact.
    let ten = u16_full(10);
    let alpha8 = u16_full(8);
    let samples: Vec<u16> = ten
        .iter()
        .zip(alpha8.iter().cycle())
        .flat_map(|(&c, &a)| [c, a])
        .collect();
    let mixed = measured(16, false, &samples, true);
    assert_eq!((mixed.color_bits(), mixed.alpha_bits()), (10, 8));
    let alpha_moved = |coded| {
        changed(
            ChannelType::U16,
            16,
            &alpha8,
            coded,
            CodeRole::Alpha,
            SignalRange::Full,
        )
    };
    assert_eq!(intensity(&ten, 10), 0);
    assert_eq!(alpha_moved(10), 252);
    assert!(!depth_exact(mixed, 10, Some(10)));
    assert_eq!(alpha_moved(8), 0);
    assert!(depth_exact(mixed, 10, Some(8)));
    assert!(depth_exact(mixed, 10, Some(16)));
    // A plan that does not code alpha is judged on color alone.
    assert!(depth_exact(mixed, 10, None));
}
