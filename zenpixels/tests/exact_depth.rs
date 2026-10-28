//! `ExactDepth::measure` and `ExactDepth::holds_for` against an independent oracle.
//!
//! The oracle enumerates each grid: it widens every code of a depth with an
//! explicit quotient and remainder (full scale) or a shift (narrow range) and
//! marks the storage values it reaches. The scan instead narrows each sample and
//! widens it back, so the two agree only if both are right.

use zenpixels::{
    ChannelType, PixelBuffer, PixelDescriptor, PixelSlice, SignalRange,
    sample::{ExactDepth, ExactDepthError},
};

/// Storage values on the full-scale `bits`-deep grid.
fn full_grid(storage_bits: u32, bits: u32) -> Vec<bool> {
    let storage_max = (1u64 << storage_bits) - 1;
    let max = (1u64 << bits) - 1;
    let mut on = vec![false; 1 << storage_bits];
    for q in 0..=max {
        let (whole, rest) = (q * storage_max / max, q * storage_max % max);
        // Nearest; `max` is odd, so a remainder is never exactly half of it.
        on[(whole + u64::from(2 * rest > max)) as usize] = true;
    }
    on
}

/// Storage values on the narrow-range `bits`-deep grid: codes shifted up.
fn narrow_grid(storage_bits: u32, bits: u32) -> Vec<bool> {
    let mut on = vec![false; 1 << storage_bits];
    for q in 0..1u32 << bits {
        on[(q << (storage_bits - bits)) as usize] = true;
    }
    on
}

fn grid(storage_bits: u32, bits: u32, narrow: bool) -> Vec<bool> {
    if narrow {
        narrow_grid(storage_bits, bits)
    } else {
        full_grid(storage_bits, bits)
    }
}

fn depths(storage_bits: u32, narrow: bool) -> std::ops::RangeInclusive<u32> {
    if narrow {
        8..=storage_bits
    } else {
        1..=storage_bits
    }
}

fn storage(storage_bits: u32) -> ChannelType {
    if storage_bits == 8 {
        ChannelType::U8
    } else {
        ChannelType::U16
    }
}

fn range(narrow: bool) -> SignalRange {
    if narrow {
        SignalRange::Narrow
    } else {
        SignalRange::Full
    }
}

fn gray(storage_bits: u32, narrow: bool) -> PixelDescriptor {
    let d = if storage_bits == 8 {
        PixelDescriptor::GRAY8
    } else {
        PixelDescriptor::GRAY16
    };
    d.with_signal_range(range(narrow))
}

/// Native-endian bytes of `values` at the storage width.
fn bytes(storage_bits: u32, values: &[u32]) -> Vec<u8> {
    if storage_bits == 8 {
        values.iter().map(|&v| v as u8).collect()
    } else {
        values
            .iter()
            .flat_map(|&v| (v as u16).to_ne_bytes())
            .collect()
    }
}

fn claim(storage_bits: u32, narrow: bool, color: u32, alpha: u32) -> ExactDepth {
    ExactDepth::new(
        storage(storage_bits),
        range(narrow),
        color as u8,
        alpha as u8,
    )
    .unwrap()
}

#[test]
fn every_storage_value_holds_exactly_where_the_oracle_marks_it() {
    for storage_bits in [8u32, 16] {
        let all: Vec<u32> = (0..1u32 << storage_bits).collect();
        let data = bytes(storage_bits, &all);
        let bpp = storage_bits as usize / 8;
        for narrow in [false, true] {
            let descriptor = gray(storage_bits, narrow);
            for bits in depths(storage_bits, narrow) {
                let on = grid(storage_bits, bits, narrow);
                let evidence = claim(storage_bits, narrow, bits, storage_bits);
                for &v in &all {
                    let at = v as usize * bpp;
                    let one = PixelSlice::new(&data[at..at + bpp], 1, 1, bpp, descriptor).unwrap();
                    assert_eq!(
                        evidence.holds_for(&one),
                        Ok(on[v as usize]),
                        "{storage_bits}-bit narrow={narrow}: value {v} on the {bits}-bit grid"
                    );
                }
            }
        }
    }
}

#[test]
fn measure_finds_the_smallest_oracle_grid_for_values_and_complete_grids() {
    for storage_bits in [8u32, 16] {
        let bpp = storage_bits as usize / 8;
        for narrow in [false, true] {
            let descriptor = gray(storage_bits, narrow);
            let grids: Vec<(u32, Vec<bool>)> = depths(storage_bits, narrow)
                .map(|bits| (bits, grid(storage_bits, bits, narrow)))
                .collect();
            let smallest = |values: &[u32]| {
                grids
                    .iter()
                    .find(|(_, on)| values.iter().all(|&v| on[v as usize]))
                    .map(|(bits, _)| *bits)
                    .expect("the storage grid holds every value")
            };
            for v in 0..1u32 << storage_bits {
                let data = bytes(storage_bits, &[v]);
                let one = PixelSlice::new(&data, 1, 1, bpp, descriptor).unwrap();
                let measured = ExactDepth::measure(&one).unwrap();
                assert_eq!(
                    u32::from(measured.color_bits()),
                    smallest(&[v]),
                    "value {v}"
                );
                assert!(measured.holds_for(&one).unwrap());
            }
            for (bits, on) in &grids {
                let levels: Vec<u32> = (0..1u32 << storage_bits)
                    .filter(|&v| on[v as usize])
                    .collect();
                let data = bytes(storage_bits, &levels);
                let all =
                    PixelSlice::new(&data, levels.len() as u32, 1, data.len(), descriptor).unwrap();
                let measured = ExactDepth::measure(&all).unwrap();
                assert_eq!(u32::from(measured.color_bits()), smallest(&levels));
                assert_eq!(
                    u32::from(measured.color_bits()),
                    *bits,
                    "complete {bits}-bit grid"
                );
            }
        }
    }
}

#[test]
fn color_and_alpha_are_checked_against_their_own_grids() {
    // GrayAlpha U16: color on the 10-bit grid, alpha on the 8-bit grid.
    let ten = full_grid(16, 10);
    let eight = full_grid(16, 8);
    let color: Vec<u32> = (0..65536).filter(|&v| ten[v as usize]).step_by(7).collect();
    let alpha: Vec<u32> = (0..65536).filter(|&v| eight[v as usize]).collect();
    let pairs: Vec<u32> = color
        .iter()
        .zip(alpha.iter().cycle())
        .flat_map(|(&c, &a)| [c, a])
        .collect();
    let data = bytes(16, &pairs);
    let ga = PixelSlice::new(
        &data,
        (pairs.len() / 2) as u32,
        1,
        data.len(),
        PixelDescriptor::GRAYA16,
    )
    .unwrap();
    let measured = ExactDepth::measure(&ga).unwrap();
    assert_eq!((measured.color_bits(), measured.alpha_bits()), (10, 8));
    for (c, a, holds) in [
        (10, 8, true),
        (10, 16, true),
        (16, 8, true),
        (10, 4, false),
        (12, 8, false),
        (5, 8, false),
    ] {
        assert_eq!(
            claim(16, false, c, a).holds_for(&ga),
            Ok(holds),
            "color {c} alpha {a}"
        );
    }
    // Narrow color still has full-scale alpha: 255 << 8 is a narrow 8-bit code but
    // not a full-scale 8-bit alpha level (255 * 257 = 65535).
    let narrow_pixels = bytes(16, &[64 << 6, 255 * 257, 940 << 6, 255 << 8]);
    let narrow = PixelSlice::new(
        &narrow_pixels,
        2,
        1,
        8,
        PixelDescriptor::GRAYA16.with_signal_range(SignalRange::Narrow),
    )
    .unwrap();
    assert_eq!(claim(16, true, 10, 8).holds_for(&narrow), Ok(false));
    assert_eq!(claim(16, true, 10, 16).holds_for(&narrow), Ok(true));
    assert_eq!(ExactDepth::measure(&narrow).unwrap().alpha_bits(), 16);
    // BGRA keeps alpha last.
    let bgra = [17u8, 34, 51, 255, 68, 85, 102, 0];
    let bgra = PixelSlice::new(&bgra, 2, 1, 8, PixelDescriptor::BGRA8).unwrap();
    let measured = ExactDepth::measure(&bgra).unwrap();
    assert_eq!((measured.color_bits(), measured.alpha_bits()), (4, 1));
}

#[test]
fn padding_bytes_and_rgbx_padding_slots_are_never_read() {
    // Two 12-bit rows of three gray samples, stride five samples, padding 1 (on no
    // grid below 16 bits).
    let twelve = full_grid(16, 12);
    let codes: Vec<u32> = (0..65536)
        .filter(|&v| twelve[v as usize])
        .step_by(613)
        .take(6)
        .collect();
    let words = [
        codes[0], codes[1], codes[2], 1, 1, codes[3], codes[4], codes[5], 1, 1,
    ];
    let data = bytes(16, &words);
    let strided = PixelSlice::new(&data, 3, 2, 10, PixelDescriptor::GRAY16).unwrap();
    assert!(claim(16, false, 12, 16).holds_for(&strided).unwrap());
    assert_eq!(ExactDepth::measure(&strided).unwrap().color_bits(), 12);
    // RGBX: the fourth slot is undefined padding, not alpha.
    let rgbx = [17u8, 34, 51, 7, 68, 85, 102, 9];
    let rgbx = PixelSlice::new(&rgbx, 2, 1, 8, PixelDescriptor::RGBX8_SRGB).unwrap();
    let measured = ExactDepth::measure(&rgbx).unwrap();
    assert_eq!((measured.color_bits(), measured.alpha_bits()), (4, 8));
    assert!(claim(8, false, 4, 1).holds_for(&rgbx).unwrap());
}

#[test]
fn contradictory_samples_and_mismatched_statements_are_refused() {
    let ten = full_grid(16, 10);
    let codes: Vec<u32> = (0..65536)
        .filter(|&v| ten[v as usize])
        .step_by(37)
        .collect();
    let evidence = claim(16, false, 10, 16);
    let clean = bytes(16, &codes);
    let clean_view = PixelSlice::new(
        &clean,
        codes.len() as u32,
        1,
        clean.len(),
        PixelDescriptor::GRAY16,
    )
    .unwrap();
    assert_eq!(evidence.holds_for(&clean_view), Ok(true));
    // One sample one unit off the grid, at the first, middle and last position.
    for at in [0, codes.len() / 2, codes.len() - 1] {
        let mut edited = codes.clone();
        edited[at] ^= 1;
        assert!(!ten[edited[at] as usize]);
        let data = bytes(16, &edited);
        let view = PixelSlice::new(
            &data,
            edited.len() as u32,
            1,
            data.len(),
            PixelDescriptor::GRAY16,
        )
        .unwrap();
        assert_eq!(evidence.holds_for(&view), Ok(false), "edit at {at}");
        // A stale statement carried by the view is not consulted.
        let stale = view.with_exact_depth(evidence).unwrap();
        assert_eq!(evidence.holds_for(&stale), Ok(false));
        assert!(!ExactDepth::measure(&stale).unwrap().color_exact_at(10));
    }
    // Storage, range and layout must match what the statement covers.
    assert_eq!(
        claim(8, false, 8, 8).holds_for(&clean_view),
        Err(ExactDepthError::DescriptorMismatch)
    );
    assert_eq!(
        claim(16, true, 10, 16).holds_for(&clean_view),
        Err(ExactDepthError::DescriptorMismatch)
    );
    let cmyk = PixelSlice::new(&[0u8; 4], 1, 1, 4, PixelDescriptor::CMYK8).unwrap();
    assert_eq!(
        claim(8, false, 8, 8).holds_for(&cmyk),
        Err(ExactDepthError::UnsupportedLayout)
    );
    assert_eq!(
        ExactDepth::measure(&cmyk),
        Err(ExactDepthError::UnsupportedLayout)
    );
    // An owned buffer aligns its storage; a byte array need not be f32-aligned.
    let floats = PixelBuffer::new(1, 1, PixelDescriptor::RGBF32_LINEAR);
    let f32_view = floats.as_slice();
    assert_eq!(
        ExactDepth::measure(&f32_view),
        Err(ExactDepthError::UnsupportedStorage)
    );
    assert_eq!(
        evidence.holds_for(&f32_view),
        Err(ExactDepthError::UnsupportedStorage)
    );
}

#[test]
fn empty_views_hold_and_measure_exact_at_every_depth() {
    let data = bytes(16, &[1, 2, 3, 4]);
    let full = PixelSlice::new(&data, 2, 2, 4, PixelDescriptor::GRAY16).unwrap();
    let no_rows = full.sub_rows(1, 0);
    let no_width = PixelSlice::new(&[], 0, 3, 0, PixelDescriptor::GRAY16).unwrap();
    for empty in [&no_rows, &no_width] {
        assert_eq!(claim(16, false, 1, 1).holds_for(empty), Ok(true));
        let measured = ExactDepth::measure(empty).unwrap();
        assert!((1..=16).all(|bits| measured.color_exact_at(bits)));
    }
    let narrow = PixelSlice::new(&[], 0, 1, 0, gray(16, true)).unwrap();
    assert_eq!(ExactDepth::measure(&narrow).unwrap().color_bits(), 8);
}

#[test]
fn a_row_driver_finds_the_first_strip_that_breaks_a_declaration() {
    let ten = full_grid(16, 10);
    let codes: Vec<u32> = (0..65536).filter(|&v| ten[v as usize]).collect();
    let (width, height, strip) = (64u32, 16u32, 4u32);
    let mut words: Vec<u32> = (0..(width * height) as usize)
        .map(|i| codes[(i * 389) % codes.len()])
        .collect();
    // Row 9 (third strip) gains one sample off the 10-bit grid after "headers".
    let bad = (9 * width + 17) as usize;
    words[bad] ^= 1;
    assert!(!ten[words[bad] as usize]);
    let data = bytes(16, &words);
    let image = PixelSlice::new(
        &data,
        width,
        height,
        width as usize * 2,
        PixelDescriptor::GRAY16,
    )
    .unwrap();
    let declared = claim(16, false, 10, 16);
    let failing: Vec<u32> = (0..height / strip)
        .filter(|&s| {
            !declared
                .holds_for(&image.sub_rows(s * strip, strip))
                .unwrap()
        })
        .collect();
    assert_eq!(failing, [2]);
}
