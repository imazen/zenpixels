//! A pixel's output must not depend on where it sits in a row or how the row is split.
use zenpixels::{
    AlphaMode, ChannelLayout as L, ChannelType as T, PixelDescriptor as D, TransferFunction as Tf,
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

/// Not a contract: tiers may round differently. Prints how far they are apart.
#[test]
#[ignore = "measurement: run with --ignored --nocapture"]
fn measure_differences_between_tiers() {
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
    use std::collections::BTreeMap;
    let options = ConvertOptions::permissive().with_alpha_policy(AlphaPolicy::DiscardUnchecked);
    let mut rows: Vec<BTreeMap<(String, String, usize), Vec<u8>>> = Vec::new();
    let mut names = Vec::new();
    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        names.push(perm.to_string());
        let mut out = BTreeMap::new();
        for_each_conversion(|from, to| {
            if let Ok(mut c) = RowConverter::new_explicit(from, to, &options) {
                let mut row = vec![0u8; WIDTH * to.bytes_per_pixel()];
                c.convert_row(&source_bytes(from), &mut row, WIDTH as u32);
                out.insert(
                    (
                        format!("{from:?}"),
                        format!("{to:?}"),
                        match to.channel_type() {
                            T::U8 => 1,
                            T::U16 => 2,
                            _ => 0,
                        },
                    ),
                    row,
                );
            }
        });
        rows.push(out);
    });
    for (name, tier) in names.iter().zip(&rows) {
        let (mut conversions, mut differ, mut integer) = (0u32, 0u32, 0u32);
        let (mut samples, mut wrong, mut worst) = (0u64, 0u64, 0u64);
        let mut worst_at = String::new();
        for (k, row) in tier {
            conversions += 1;
            let base = &rows[0][k];
            if row == base {
                continue;
            }
            differ += 1;
            let size = k.2;
            if size == 0 {
                continue;
            }
            integer += 1;
            let v = |s: &[u8]| s.iter().rev().fold(0u64, |n, &b| n << 8 | u64::from(b));
            for (a, b) in row.chunks(size).zip(base.chunks(size)) {
                samples += 1;
                let d = v(a).abs_diff(v(b));
                wrong += u64::from(d != 0);
                if d > worst {
                    worst = d;
                    worst_at = format!("{} -> {}", k.0, k.1);
                }
            }
        }
        println!(
            "{name}: {differ} of {conversions} differ from {}; {integer} with integer output, {wrong} of {samples} samples, largest {worst} codes {worst_at}",
            names[0]
        );
    }
}
