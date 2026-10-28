//! Exhaustive checks of native code conversion against an independent oracle.
//!
//! The oracle rounds in f64 rather than with the kernel's integer formula. Every
//! code of every depth 1..=16 is visited; the full set runs in well under a second.

use zenpixels::{
    ChannelType, PixelDescriptor, PixelSlice, PixelSliceMut, SignalRange,
    sample::{CodePlane, ExactDepth, ExactDepthError, SampleEncoding},
};
use zenpixels_convert::codes::{CodeMap, CodeMapError, CodeRole, CodesError, to_canonical};

fn enc(bits: u8, shift: u8) -> SampleEncoding {
    SampleEncoding::new(ChannelType::U16, bits, shift).unwrap()
}
fn max(bits: u8) -> u32 {
    (1u32 << bits) - 1
}
/// Nearest full-scale rescaling by f64; no ties exist because max(N) is odd.
fn oracle_full(q: u32, n: u8, m: u8) -> u32 {
    (f64::from(q) * f64::from(max(m)) / f64::from(max(n))).round() as u32
}
/// Narrow range: exact power-of-two anchor scaling; narrowing rounds half up and clamps.
fn oracle_narrow(q: u32, n: u8, m: u8) -> u32 {
    if m >= n {
        q * (1 << (m - n))
    } else {
        let scaled = (f64::from(q) / f64::from(1u32 << (n - m)) + 0.5).floor() as u32;
        scaled.min(max(m))
    }
}
fn map(n: u8, m: u8, role: CodeRole, range: SignalRange) -> CodeMap {
    CodeMap::new(enc(n, 0), enc(m, 0), role, range).unwrap()
}

#[test]
fn full_scale_rules_match_the_oracle_for_every_depth_pair_and_code() {
    for n in 1..=16u8 {
        for m in 1..=16u8 {
            let intensity = map(n, m, CodeRole::Intensity, SignalRange::Full);
            // Alpha is full scale even inside a narrow-range color signal.
            let alpha = map(n, m, CodeRole::Alpha, SignalRange::Narrow);
            for q in 0..=max(n) {
                let want = oracle_full(q, n, m);
                assert_eq!(u32::from(intensity.code(q as u16)), want, "{n}->{m} q={q}");
                assert_eq!(
                    u32::from(alpha.code(q as u16)),
                    want,
                    "alpha {n}->{m} q={q}"
                );
            }
        }
    }
}

#[test]
fn narrow_rules_match_the_oracle_for_intensity_and_chroma() {
    for n in 8..=16u8 {
        for m in 8..=16u8 {
            for role in [CodeRole::Intensity, CodeRole::Chroma] {
                let rule = map(n, m, role, SignalRange::Narrow);
                for q in 0..=max(n) {
                    assert_eq!(
                        u32::from(rule.code(q as u16)),
                        oracle_narrow(q, n, m),
                        "{role:?} {n}->{m} q={q}"
                    );
                }
            }
        }
    }
}

#[test]
fn widening_then_narrowing_recovers_every_code() {
    for n in 1..=16u8 {
        for m in n..=16u8 {
            let mut cases = vec![
                (CodeRole::Intensity, SignalRange::Full),
                (CodeRole::Alpha, SignalRange::Full),
            ];
            if n >= 8 {
                cases.push((CodeRole::Intensity, SignalRange::Narrow));
                cases.push((CodeRole::Chroma, SignalRange::Narrow));
            }
            for (role, range) in cases {
                let up = map(n, m, role, range);
                let down = map(m, n, role, range);
                for q in 0..=max(n) as u16 {
                    assert_eq!(
                        down.code(up.code(q)),
                        q,
                        "{role:?}/{range} {n}->{m}->{n} q={q}"
                    );
                }
            }
        }
    }
}

/// The finding behind "higher depth is not automatically exact": canonical U16
/// samples holding N-bit content survive M-bit quantization exactly iff N | M.
#[test]
fn full_range_grids_nest_only_when_the_depth_divides() {
    let canonical = 16u8;
    for n in 1..16u8 {
        for m in 1..16u8 {
            let widen_n = map(n, canonical, CodeRole::Intensity, SignalRange::Full);
            let to_m = map(canonical, m, CodeRole::Intensity, SignalRange::Full);
            let from_m = map(m, canonical, CodeRole::Intensity, SignalRange::Full);
            let to_n = map(canonical, n, CodeRole::Intensity, SignalRange::Full);
            let mut changed = 0u32;
            for q in 0..=max(n) as u16 {
                let w = widen_n.code(q);
                let back = from_m.code(to_m.code(w));
                changed += u32::from(back != w);
                if m >= n {
                    // The original codes stay recoverable even when U16 values change.
                    assert_eq!(to_n.code(back), q, "codes {n} via {m}");
                }
            }
            assert_eq!(
                changed == 0,
                m.is_multiple_of(n),
                "grid({n}) in grid({m}): {changed} changed"
            );
        }
    }
    // Recorded magnitudes for the report (exact counts, not estimates).
    let count = |n: u8, m: u8| {
        let widen_n = map(n, 16, CodeRole::Intensity, SignalRange::Full);
        let to_m = map(16, m, CodeRole::Intensity, SignalRange::Full);
        let from_m = map(m, 16, CodeRole::Intensity, SignalRange::Full);
        (0..=max(n) as u16)
            .filter(|q| from_m.code(to_m.code(widen_n.code(*q))) != widen_n.code(*q))
            .count()
    };
    assert_eq!((count(8, 10), count(8, 12), count(10, 12)), (252, 240, 972));
}

#[test]
fn narrow_range_grids_nest_for_every_higher_depth() {
    for n in 8..16u8 {
        for m in n..16u8 {
            let widen_n = map(n, 16, CodeRole::Intensity, SignalRange::Narrow);
            let to_m = map(16, m, CodeRole::Intensity, SignalRange::Narrow);
            let from_m = map(m, 16, CodeRole::Intensity, SignalRange::Narrow);
            for q in 0..=max(n) as u16 {
                let w = widen_n.code(q);
                assert_eq!(from_m.code(to_m.code(w)), w, "narrow {n} in {m}");
            }
        }
    }
}

#[test]
fn narrow_anchors_scale_by_powers_of_two() {
    let anchors = |bits: u8| -> [u32; 5] {
        let k = 1u32 << (bits - 8);
        [16 * k, 235 * k, 128 * k, 240 * k, 16 * k]
    };
    for n in [8u8, 10, 12, 16] {
        for m in [8u8, 10, 12, 16] {
            let luma = map(n, m, CodeRole::Intensity, SignalRange::Narrow);
            let chroma = map(n, m, CodeRole::Chroma, SignalRange::Narrow);
            let [black, white, neutral, high, low] = anchors(n);
            let want = anchors(m);
            assert_eq!(u32::from(luma.code(black as u16)), want[0]);
            assert_eq!(u32::from(luma.code(white as u16)), want[1]);
            assert_eq!(u32::from(chroma.code(neutral as u16)), want[2]);
            assert_eq!(u32::from(chroma.code(high as u16)), want[3]);
            assert_eq!(u32::from(chroma.code(low as u16)), want[4]);
        }
    }
    // Full-scale widening would put narrow black at 16 * 257 = 4112, not 4096.
    assert_eq!(
        map(8, 16, CodeRole::Intensity, SignalRange::Narrow).code(16),
        4096
    );
    assert_eq!(
        map(8, 16, CodeRole::Intensity, SignalRange::Full).code(16),
        4112
    );
}

#[test]
fn narrow_quantization_clamps_super_white_at_the_top_code() {
    let rule = map(16, 10, CodeRole::Intensity, SignalRange::Narrow);
    let clamped = (0..=u16::MAX)
        .filter(|w| (u32::from(*w) + 32) >> 6 > 1023)
        .count();
    assert_eq!(clamped, 32);
    assert_eq!(rule.code(65535), 1023);
    assert_eq!(rule.code(65503), 1023);
    assert_eq!(rule.code(65502), 1023);
    assert_eq!(rule.code(32), 1); // half rounds up
    assert_eq!(rule.code(31), 0);
}

#[test]
fn placement_is_applied_after_conversion_and_padding_is_ignored() {
    let p010 = enc(10, 6);
    let canonical = enc(16, 0);
    let unpack = CodeMap::new(p010, canonical, CodeRole::Intensity, SignalRange::Full).unwrap();
    let repack = CodeMap::new(enc(10, 0), p010, CodeRole::Intensity, SignalRange::Full).unwrap();
    for q in 0..=1023u16 {
        for padding in [0u16, 1, 0x2a, 0x3f] {
            let word = (q << 6) | padding;
            assert_eq!(
                unpack.word(word),
                map(10, 16, CodeRole::Intensity, SignalRange::Full).code(q)
            );
        }
        assert_eq!(
            repack.word(q | 0xfc00),
            q << 6,
            "source padding above the code is ignored"
        );
        assert_eq!(repack.word(q), q << 6);
    }
    // Packing white is 65472; full-range normalization of white is 65535.
    assert_eq!(repack.word(1023), 65472);
    assert_eq!(unpack.word(65472), 65535);
    // Narrow range: zero-padded P010 words already equal canonical narrow U16.
    let narrow = CodeMap::new(p010, canonical, CodeRole::Intensity, SignalRange::Narrow).unwrap();
    for q in 0..=1023u16 {
        assert_eq!(narrow.word(q << 6), q << 6);
    }
}

#[test]
fn unexact_rules_refuse_at_construction() {
    assert_eq!(
        CodeMap::new(enc(10, 0), enc(8, 0), CodeRole::Chroma, SignalRange::Full),
        Err(CodeMapError::FullRangeChromaDepth {
            from_bits: 10,
            to_bits: 8
        })
    );
    // Same-depth full-range chroma only changes placement.
    let repack = CodeMap::new(enc(10, 0), enc(10, 6), CodeRole::Chroma, SignalRange::Full).unwrap();
    assert_eq!(repack.word(512), 512 << 6);
    assert_eq!(
        CodeMap::new(
            enc(4, 0),
            enc(10, 0),
            CodeRole::Intensity,
            SignalRange::Narrow
        ),
        Err(CodeMapError::NarrowBelowEightBits { bits: 4 })
    );
    let rule = map(10, 16, CodeRole::Intensity, SignalRange::Full);
    let mut out = [7u16; 3];
    assert_eq!(
        rule.map_words(&[1, 2], &mut out),
        Err(CodeMapError::LengthMismatch {
            source: 2,
            destination: 3
        })
    );
    assert_eq!(out, [7; 3]);
    rule.map_words(&[0, 1023, 512], &mut out).unwrap();
    assert_eq!(out, [0, 65535, 32800]);
}

// ---------------------------------------------------------------------------
// to_canonical: borrowed native components → canonical strided image
// ---------------------------------------------------------------------------

/// Run `to_canonical` into a U16 destination with three pixels of row padding
/// filled with a sentinel; return active samples and assert padding is intact.
fn canonical_u16(
    components: &[CodePlane<'_>],
    range: SignalRange,
    (x, y): (usize, usize),
    (width, rows): (usize, usize),
    descriptor: PixelDescriptor,
) -> Result<(Vec<u16>, ExactDepth), CodesError> {
    let channels = descriptor.channels();
    // Whole pixels: pixel slices refuse strides that are not pixel-aligned.
    let pad = 3 * channels;
    let stride = width * channels + pad;
    let mut words = vec![0xbeefu16; stride * rows];
    let dst = PixelSliceMut::new(
        bytemuck::cast_slice_mut(&mut words),
        width as u32,
        rows as u32,
        stride * 2,
        descriptor,
    )
    .unwrap();
    // Drop the returned view so the words can be inspected below.
    let result = to_canonical(components, range, x, y, dst)
        .map(|(_, exact)| exact)
        .map_err(|e| *e.error());
    let active: Vec<u16> = words
        .chunks_exact(stride)
        .flat_map(|r| {
            assert!(
                r[width * channels..].iter().all(|w| *w == 0xbeef),
                "padding changed"
            );
            r[..width * channels].iter().copied()
        })
        .collect();
    result.map(|exact| (active, exact))
}
fn full(storage: ChannelType, color: u8, alpha: u8) -> ExactDepth {
    ExactDepth::new(storage, SignalRange::Full, color, alpha).unwrap()
}

#[test]
fn interleaved_twelve_bit_rgb_widens_exactly_into_strided_rows() {
    let (w, h) = (7usize, 3usize);
    let src_stride_words = w * 3 + 2;
    let mut words = vec![0u16; src_stride_words * h];
    for (i, word) in words.iter_mut().enumerate() {
        *word = ((i * 997) % 4096) as u16 | 0xf000; // nonzero high padding
    }
    let twelve = enc(12, 0);
    let planes: Vec<CodePlane<'_>> = (0..3)
        .map(|c| {
            CodePlane::from_u16_interleaved(&words[c..], w, h, src_stride_words * 2, 3, twelve)
                .unwrap()
        })
        .collect();
    let (out, exact) = canonical_u16(
        &planes,
        SignalRange::Full,
        (0, 0),
        (w, h),
        PixelDescriptor::RGB16,
    )
    .unwrap();
    assert_eq!(exact, full(ChannelType::U16, 12, 16));
    // The scan independently finds the same depth in the written samples.
    let measured = ExactDepth::measure(
        &PixelSlice::new(
            bytemuck::cast_slice(&out),
            w as u32,
            h as u32,
            w * 6,
            PixelDescriptor::RGB16,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(measured, exact);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let q = u32::from(words[y * src_stride_words + x * 3 + c] & 0x0fff);
                assert_eq!(u32::from(out[(y * w + x) * 3 + c]), oracle_full(q, 12, 16));
            }
        }
    }
    // Crops and row partitions read the original planes and agree exactly.
    let whole = out;
    let (top, _) = canonical_u16(
        &planes,
        SignalRange::Full,
        (2, 0),
        (4, 2),
        PixelDescriptor::RGB16,
    )
    .unwrap();
    let (bottom, _) = canonical_u16(
        &planes,
        SignalRange::Full,
        (2, 2),
        (4, 1),
        PixelDescriptor::RGB16,
    )
    .unwrap();
    let expect: Vec<u16> = (0..3)
        .flat_map(|y| whole[(y * w + 2) * 3..(y * w + 6) * 3].to_vec())
        .collect();
    assert_eq!([top, bottom].concat(), expect);
}

#[test]
fn per_component_encodings_cover_ten_bit_color_with_sixteen_bit_alpha() {
    let color = [0u16, 1023, 512, 64];
    let alpha = [0u16, 65535, 32768, 1];
    let ten = enc(10, 0);
    let planes = [
        CodePlane::from_u16(&color[0..1], 1, 1, 2, ten).unwrap(),
        CodePlane::from_u16(&color[1..2], 1, 1, 2, ten).unwrap(),
        CodePlane::from_u16(&color[2..3], 1, 1, 2, ten).unwrap(),
        CodePlane::from_u16(&alpha[1..2], 1, 1, 2, enc(16, 0)).unwrap(),
    ];
    let (out, exact) = canonical_u16(
        &planes,
        SignalRange::Full,
        (0, 0),
        (1, 1),
        PixelDescriptor::RGBA16,
    )
    .unwrap();
    assert_eq!(out, [0, 65535, 32800, 65535]);
    assert_eq!(exact, full(ChannelType::U16, 10, 16));
    // Narrow-range color still gets full-scale alpha.
    let ten_alpha = [CodePlane::from_u16(&color[1..2], 1, 1, 2, ten).unwrap()];
    let narrow_color = [
        CodePlane::from_u16(&color[3..4], 1, 1, 2, ten).unwrap(),
        ten_alpha[0],
    ];
    let (out, exact) = canonical_u16(
        &narrow_color,
        SignalRange::Narrow,
        (0, 0),
        (1, 1),
        PixelDescriptor::GRAYA16.with_signal_range(SignalRange::Narrow),
    )
    .unwrap();
    assert_eq!(out, [64 << 6, 65535]);
    assert_eq!(
        exact,
        ExactDepth::new(ChannelType::U16, SignalRange::Narrow, 10, 10).unwrap()
    );
}

#[test]
fn p010_luma_with_zero_padding_is_already_canonical_narrow_u16() {
    let p010 = enc(10, 6);
    let luma: Vec<u16> = (0..1024u16).map(|q| q << 6).collect();
    let plane = CodePlane::from_u16(&luma, 1024, 1, 2048, p010).unwrap();
    let narrow = PixelDescriptor::GRAY16.with_signal_range(SignalRange::Narrow);
    let (out, exact) =
        canonical_u16(&[plane], SignalRange::Narrow, (0, 0), (1024, 1), narrow).unwrap();
    assert_eq!(
        out, luma,
        "a borrow would have sufficed when padding is known to be zero"
    );
    assert_eq!(exact.color_bits(), 10);
    // Full range is different: P010 white 65472 must become 65535.
    let (out, _) = canonical_u16(
        &[plane],
        SignalRange::Full,
        (0, 0),
        (1024, 1),
        PixelDescriptor::GRAY16,
    )
    .unwrap();
    assert_eq!(out[1023], 65535);
    assert_ne!(out, luma);
}

#[test]
fn eight_bit_codes_widen_by_257_and_copy_into_u8() {
    let codes: Vec<u8> = (0..=255).collect();
    let eight = SampleEncoding::new(ChannelType::U8, 8, 0).unwrap();
    let plane = CodePlane::from_u8(&codes, 256, 1, 256, eight).unwrap();
    let (out, exact) = canonical_u16(
        &[plane],
        SignalRange::Full,
        (0, 0),
        (256, 1),
        PixelDescriptor::GRAY16,
    )
    .unwrap();
    assert!(
        out.iter()
            .zip(&codes)
            .all(|(w, q)| *w == u16::from(*q) * 257)
    );
    assert_eq!(exact, full(ChannelType::U16, 8, 16));
    let mut bytes = vec![0u8; 256];
    let dst = PixelSliceMut::new(&mut bytes, 256, 1, 256, PixelDescriptor::GRAY8).unwrap();
    let (_, exact) = to_canonical(&[plane], SignalRange::Full, 0, 0, dst).unwrap();
    assert_eq!(exact, full(ChannelType::U8, 8, 8));
    assert_eq!(bytes, codes);
}

#[test]
fn refusals_precede_every_write() {
    let ten = enc(10, 0);
    let words = [1023u16; 8];
    let plane = CodePlane::from_u16(&words, 2, 2, 4, ten).unwrap();
    let rgb = [plane; 3];
    let mut bytes = vec![0x55u8; 2 * 2 * 3];
    let narrowing = PixelSliceMut::new(&mut bytes, 2, 2, 6, PixelDescriptor::RGB8_SRGB).unwrap();
    assert_eq!(
        *to_canonical(&rgb, SignalRange::Full, 0, 0, narrowing)
            .unwrap_err()
            .error(),
        CodesError::Narrowing {
            component: 0,
            bits: 10,
            destination_bits: 8
        }
    );
    assert!(bytes.iter().all(|b| *b == 0x55));
    assert!(matches!(
        canonical_u16(
            &rgb[..2],
            SignalRange::Full,
            (0, 0),
            (2, 2),
            PixelDescriptor::RGB16
        ),
        Err(CodesError::ComponentCount {
            expected: 3,
            actual: 2
        })
    ));
    assert!(matches!(
        canonical_u16(
            &rgb,
            SignalRange::Narrow,
            (0, 0),
            (2, 2),
            PixelDescriptor::RGB16
        ),
        Err(CodesError::RangeMismatch { .. })
    ));
    assert!(matches!(
        canonical_u16(
            &rgb,
            SignalRange::Full,
            (1, 0),
            (2, 2),
            PixelDescriptor::RGB16
        ),
        Err(CodesError::Extent {
            component: 0,
            origin: (1, 0),
            extent: (2, 2),
            plane: (2, 2)
        })
    ));
    let mut floats = vec![0f32; 4];
    let f32_dst = PixelSliceMut::new(
        bytemuck::cast_slice_mut(&mut floats),
        1,
        1,
        16,
        PixelDescriptor::RGBAF32_LINEAR,
    )
    .unwrap();
    assert!(matches!(
        *to_canonical(&[plane; 4], SignalRange::Full, 0, 0, f32_dst)
            .unwrap_err()
            .error(),
        CodesError::Destination { .. }
    ));
    assert!(floats.iter().all(|v| *v == 0.0));
}

// ---------------------------------------------------------------------------
// ExactDepth rule and the explicit scan
// ---------------------------------------------------------------------------

/// Brute force: is every sample of the `base` grid (in `storage` bits) unchanged
/// by quantization to `depth` bits and back? Independent of ExactDepth's rule.
fn grid_survives(storage: u8, base: u8, depth: u8, narrow: bool) -> bool {
    let (role, range) = if narrow {
        (CodeRole::Intensity, SignalRange::Narrow)
    } else {
        (CodeRole::Intensity, SignalRange::Full)
    };
    let widen = map(base, storage, role, range);
    let to = map(storage, depth, role, range);
    let back = map(depth, storage, role, range);
    (0..=max(base) as u16).all(|q| back.code(to.code(widen.code(q))) == widen.code(q))
}

#[test]
fn exact_depth_rule_matches_brute_force_grid_inclusion() {
    for (storage, storage_bits) in [(ChannelType::U8, 8u8), (ChannelType::U16, 16)] {
        for base in 1..=storage_bits {
            for depth in 1..=16u8 {
                let d = ExactDepth::new(storage, SignalRange::Full, base, base).unwrap();
                let want = grid_survives(storage_bits, base, depth.min(16), false);
                assert_eq!(
                    d.color_exact_at(depth),
                    want,
                    "full {storage_bits}: {base} at {depth}"
                );
                assert_eq!(
                    d.alpha_exact_at(depth),
                    want,
                    "alpha {storage_bits}: {base} at {depth}"
                );
                if base >= 8 && depth >= 8 {
                    let n =
                        ExactDepth::new(storage, SignalRange::Narrow, base, storage_bits).unwrap();
                    let want = grid_survives(storage_bits, base, depth, true);
                    assert_eq!(
                        n.color_exact_at(depth),
                        want,
                        "narrow {storage_bits}: {base} at {depth}"
                    );
                }
            }
        }
    }
    assert!(ExactDepth::new(ChannelType::F32, SignalRange::Full, 8, 8).is_err());
    assert!(ExactDepth::new(ChannelType::U16, SignalRange::Narrow, 6, 16).is_err());
    assert!(ExactDepth::new(ChannelType::U8, SignalRange::Full, 9, 8).is_err());
}

fn gray16(words: &[u16], narrow: bool) -> ExactDepth {
    let mut d = PixelDescriptor::GRAY16;
    if narrow {
        d = d.with_signal_range(SignalRange::Narrow);
    }
    let slice = PixelSlice::new(
        bytemuck::cast_slice(words),
        words.len() as u32,
        1,
        words.len() * 2,
        d,
    )
    .unwrap();
    ExactDepth::measure(&slice).unwrap()
}

#[test]
fn scan_finds_the_generating_depth_and_loses_it_after_one_edit() {
    for n in 1..=15u8 {
        let widen = map(n, 16, CodeRole::Intensity, SignalRange::Full);
        let mut words: Vec<u16> = (0..=max(n) as u16).map(|q| widen.code(q)).collect();
        assert_eq!(gray16(&words, false).color_bits(), n, "all {n}-bit levels");
        // One changed sample (as any filter could produce) ends the evidence.
        let i = words.len() / 2;
        // Flip the low bit: a 1-unit change cannot wrap onto another level, and
        // levels of grids up to 15 bits are at least 2 units apart.
        words[i] ^= 1;
        let after = gray16(&words, false);
        assert!(
            !after.color_exact_at(n),
            "edited {n}-bit content must not stay exact"
        );
    }
    // Narrow range uses the power-of-two grid.
    let narrow: Vec<u16> = (64..=940u16).map(|q| q << 6).collect();
    assert_eq!(gray16(&narrow, true).color_bits(), 10);
    assert_eq!(gray16(&[4096, 4097], true).color_bits(), 16);
    assert_eq!(gray16(&[0, 0], true).color_bits(), 8);
    // U8 storage: the 4-bit grid is 17·q.
    let u8s: Vec<u8> = (0..16u8).map(|q| q * 17).collect();
    let slice = PixelSlice::new(&u8s, 16, 1, 16, PixelDescriptor::GRAY8).unwrap();
    assert_eq!(
        ExactDepth::measure(&slice).unwrap(),
        full(ChannelType::U8, 4, 8)
    );
}

#[test]
fn scan_measures_alpha_separately_and_ignores_padding() {
    // RGBA16: color on the 12-bit grid, alpha only 0 or max (1-bit grid).
    let twelve = map(12, 16, CodeRole::Intensity, SignalRange::Full);
    let (w, h, stride_words) = (5usize, 2usize, 5 * 4 + 4); // one pixel of padding
    let mut words = vec![1u16; stride_words * h]; // padding 1 is on no coarse grid
    for y in 0..h {
        for x in 0..w {
            let px = &mut words[y * stride_words + x * 4..][..4];
            px[0] = twelve.code((x * 811 + y * 37) as u16 % 4096);
            px[1] = twelve.code((x * 13) as u16);
            px[2] = twelve.code(4095);
            px[3] = if x.is_multiple_of(2) { 65535 } else { 0 };
        }
    }
    let slice = PixelSlice::new(
        bytemuck::cast_slice(&words),
        w as u32,
        h as u32,
        stride_words * 2,
        PixelDescriptor::RGBA16,
    )
    .unwrap();
    assert_eq!(
        ExactDepth::measure(&slice).unwrap(),
        full(ChannelType::U16, 12, 1)
    );
    let floats = [0f32; 3];
    let f = PixelSlice::new(
        bytemuck::cast_slice(&floats),
        1,
        1,
        12,
        PixelDescriptor::RGBF32_LINEAR,
    )
    .unwrap();
    assert_eq!(
        ExactDepth::measure(&f).unwrap_err(),
        ExactDepthError::UnsupportedStorage
    );
}
