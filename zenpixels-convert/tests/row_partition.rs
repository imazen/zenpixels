//! A pixel's output must not depend on where it sits in a row or how the row is split.
use zenpixels::{
    AlphaMode, ChannelLayout as L, ChannelType as T, ColorPrimaries, PixelDescriptor as D,
    TransferFunction as Tf,
};
use zenpixels_convert::{AlphaPolicy, ConvertOptions, RowConverter};

const WIDTH: usize = 1031;

fn source_bytes(d: D) -> Vec<u8> {
    let n = WIDTH * d.layout().channels();
    let mut x = 0x2545_F491u32;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        x
    };
    let mut out = Vec::with_capacity(WIDTH * d.bytes_per_pixel());
    for _ in 0..n {
        match d.channel_type() {
            T::U8 => out.push(next() as u8),
            T::U16 => out.extend_from_slice(&(next() as u16).to_ne_bytes()),
            T::F32 => out.extend_from_slice(&((next() >> 8) as f32 / 16777215.0).to_ne_bytes()),
            T::F16 => {
                let v = half::f16::from_f32((next() >> 8) as f32 / 16777215.0);
                out.extend_from_slice(&v.to_bits().to_ne_bytes())
            }
            _ => unreachable!(),
        }
    }
    out
}

fn check(from: D, to: D, failures: &mut Vec<String>) {
    let options = ConvertOptions::permissive().with_alpha_policy(AlphaPolicy::DiscardUnchecked);
    let Ok(mut c) = RowConverter::new_explicit(from, to, &options) else {
        return;
    };
    let (sb, db) = (from.bytes_per_pixel(), to.bytes_per_pixel());
    let src = source_bytes(from);
    let mut whole = vec![0u8; WIDTH * db];
    c.convert_row(&src, &mut whole, WIDTH as u32);
    let mut single = vec![0u8; db];
    let mut differ = 0usize;
    let mut largest = 0u64;
    for i in 0..WIDTH {
        c.convert_row(&src[i * sb..(i + 1) * sb], &mut single, 1);
        let want = &whole[i * db..(i + 1) * db];
        if single != want {
            differ += 1;
            let size = to.channel_type().byte_size();
            for (a, b) in single.chunks(size).zip(want.chunks(size)) {
                let v = |s: &[u8]| s.iter().rev().fold(0u64, |n, &b| n << 8 | u64::from(b));
                largest = largest.max(v(a).abs_diff(v(b)));
            }
        }
    }
    if differ != 0 {
        failures.push(format!(
            "{:?}/{:?} -> {:?}/{:?}: {differ} of {WIDTH} pixels differ, largest {largest} codes",
            from.format,
            from.transfer(),
            to.format,
            to.transfer()
        ));
    }
}

#[test]
fn single_pixel_rows_match_a_whole_row() {
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
    let mut failures = Vec::new();
    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let tier = perm.to_string();
        let before = failures.len();
        for_each_conversion(|from, to| check(from, to, &mut failures));
        for f in &mut failures[before..] {
            f.push_str(&format!(" [{tier}]"));
        }
    });
    for f in failures.iter().take(40) {
        println!("{f}");
    }
    assert!(
        failures.is_empty(),
        "{} conversions depend on position",
        failures.len()
    );
}

fn for_each_conversion(mut run: impl FnMut(D, D)) {
    let tfs = [
        Tf::Srgb,
        Tf::Linear,
        Tf::Bt709,
        Tf::Pq,
        Tf::Hlg,
        Tf::Gamma22,
    ];
    let alpha = Some(AlphaMode::Straight);
    for (l, a) in [
        (L::Rgb, None),
        (L::Rgba, alpha),
        (L::Gray, None),
        (L::GrayAlpha, alpha),
    ] {
        for t1 in [T::U8, T::U16, T::F16, T::F32] {
            for t2 in [T::U8, T::U16, T::F16, T::F32] {
                for &f1 in &tfs {
                    for &f2 in &tfs {
                        run(D::new(t1, l, a, f1), D::new(t2, l, a, f2));
                    }
                }
            }
        }
    }
}

/// Output must not depend on the CPU: tests and goldens hold across machines.
#[test]
fn every_tier_gives_the_same_bytes() {
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
    use std::collections::BTreeMap;
    let policies = [
        AlphaPolicy::DiscardUnchecked,
        AlphaPolicy::CompositeOnto {
            r: 200,
            g: 100,
            b: 50,
            reference_white: None,
        },
    ];
    let mut first: BTreeMap<(String, String, usize), Vec<u8>> = BTreeMap::new();
    let mut failures = Vec::new();
    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let fill = first.is_empty();
        let mut record = |policy: usize, from: D, to: D| {
            let options = ConvertOptions::permissive().with_alpha_policy(policies[policy]);
            let Ok(mut c) = RowConverter::new_explicit(from, to, &options) else {
                return;
            };
            let mut row = vec![0u8; WIDTH * to.bytes_per_pixel()];
            c.convert_row(&source_bytes(from), &mut row, WIDTH as u32);
            let key = (format!("{from:?}"), format!("{to:?}"), policy);
            if fill {
                first.insert(key, row);
            } else if first.get(&key) != Some(&row) {
                failures.push(format!("{} -> {} at {}", key.0, key.1, perm.label));
            }
        };
        for_each_conversion(|from, to| {
            record(0, from, to);
            if from.layout().has_alpha() && !to.layout().has_alpha() {
                record(1, from, to);
            }
        });
        for_each_gamut_conversion(|from, to| record(0, from, to));
        for_each_other_conversion(|from, to| {
            record(0, from, to);
            if from.layout().has_alpha() && !to.layout().has_alpha() {
                record(1, from, to);
            }
        });
    });
    assert!(first.len() > 1000, "{} conversions", first.len());
    for f in failures.iter().take(20) {
        println!("{f}");
    }
    assert!(
        failures.is_empty(),
        "{} differ across tiers",
        failures.len()
    );
}

fn for_each_gamut_conversion(mut run: impl FnMut(D, D)) {
    let primaries = [
        ColorPrimaries::Bt709,
        ColorPrimaries::DisplayP3,
        ColorPrimaries::Bt2020,
    ];
    let alpha = Some(AlphaMode::Straight);
    for (l, a) in [(L::Rgb, None), (L::Rgba, alpha)] {
        for t in [T::U8, T::U16, T::F16, T::F32] {
            for tf in [Tf::Srgb, Tf::Linear, Tf::Pq] {
                for &p1 in &primaries {
                    for &p2 in &primaries {
                        run(
                            D::new(t, l, a, tf).with_primaries(p1),
                            D::new(t, l, a, tf).with_primaries(p2),
                        );
                    }
                }
            }
        }
    }
}

/// Oklab, layout changes and premultiplication.
fn for_each_other_conversion(mut run: impl FnMut(D, D)) {
    let s = Some(AlphaMode::Straight);
    let p = Some(AlphaMode::Premultiplied);
    let layouts = [
        (L::Rgb, None),
        (L::Rgba, s),
        (L::Gray, None),
        (L::GrayAlpha, s),
    ];
    for t in [T::U8, T::U16, T::F16, T::F32] {
        for tf in [Tf::Srgb, Tf::Linear] {
            for &(l1, a1) in &layouts {
                for &(l2, a2) in &layouts {
                    run(D::new(t, l1, a1, tf), D::new(t, l2, a2, tf));
                }
            }
            for (l, _) in [(L::Rgba, s), (L::GrayAlpha, s)] {
                run(D::new(t, l, s, tf), D::new(t, l, p, tf));
                run(D::new(t, l, p, tf), D::new(t, l, s, tf));
            }
        }
    }
    let f = T::F32;
    let (lin, un) = (Tf::Linear, Tf::Unknown);
    run(D::new(f, L::Rgb, None, lin), D::new(f, L::Oklab, None, un));
    run(D::new(f, L::Oklab, None, un), D::new(f, L::Rgb, None, lin));
    run(D::new(f, L::Rgba, s, lin), D::new(f, L::OklabA, s, un));
    run(D::new(f, L::OklabA, s, un), D::new(f, L::Rgba, s, lin));
}

/// The output of every conversion, hashed. A constant here means every platform CI runs on gives
/// the same bytes. A deliberate numerical change updates it; anything else is a bug.
#[test]
fn output_digest_matches_on_every_platform() {
    let options = ConvertOptions::permissive().with_alpha_policy(AlphaPolicy::DiscardUnchecked);
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut run = |from: D, to: D| {
        let Ok(mut c) = RowConverter::new_explicit(from, to, &options) else {
            return;
        };
        let mut row = vec![0u8; WIDTH * to.bytes_per_pixel()];
        c.convert_row(&source_bytes(from), &mut row, WIDTH as u32);
        // Samples, not bytes, so endianness can't matter for wide samples on a future target.
        for b in row {
            hash = (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for_each_conversion(&mut run);
    for_each_gamut_conversion(&mut run);
    for_each_other_conversion(&mut run);
    assert_eq!(hash, EXPECTED_DIGEST, "output changed: {hash:#018x}");
}

const EXPECTED_DIGEST: u64 = 0xa0b25fc4d0b3d965;

/// Rounding in f32 polynomials can reverse adjacent inputs, by at most one 16-bit code. Windows
/// of consecutive floats at the top of each range and around every branch.
#[test]
fn adjacent_inputs_reverse_by_at_most_one_code() {
    let options = ConvertOptions::permissive();
    let d = |t, tf| D::new(t, L::Gray, None, tf);
    let centers = [
        1.0f32,
        0.75,
        0.5,
        0.25,
        0.081242,
        1.0 / 12.0,
        0.04045,
        0.02,
        0.018053968,
        0.0031308,
        1e-4,
        1e-7,
    ];
    for tf in [Tf::Pq, Tf::Hlg, Tf::Bt709, Tf::Srgb, Tf::Gamma22] {
        for (from, to) in [(tf, Tf::Linear), (Tf::Linear, tf)] {
            let mut c =
                RowConverter::new_explicit(d(T::F32, from), d(T::U16, to), &options).unwrap();
            for center in centers {
                let start = center.to_bits() - (1 << 15);
                let input: Vec<f32> = (0..1u32 << 16).map(|i| f32::from_bits(start + i)).collect();
                let mut out = vec![0u16; input.len()];
                c.convert_row(
                    bytemuck::cast_slice(&input),
                    bytemuck::cast_slice_mut(&mut out),
                    input.len() as u32,
                );
                for w in out.windows(2) {
                    assert!(
                        u32::from(w[1]) + 1 >= u32::from(w[0]),
                        "{from:?} to {to:?} near {center}: {w:?}"
                    );
                }
            }
        }
    }
}
