//! Measure serialized curves against the existing independent H.273 reference.
use std::collections::BTreeMap;

#[path = "../../zenpixels-convert/src/icc_profiles/transfer_reference.rs"]
mod reference;

#[derive(Default)]
struct Metrics {
    profiles: usize,
    max_abs: f64,
    at: f64,
    zeros_with_nonzero_reference: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let folder = std::env::args_os()
        .nth(1)
        .ok_or("expected profile directory")?;
    let mut results = BTreeMap::<u8, Metrics>::new();
    for entry in std::fs::read_dir(folder)? {
        let path = entry?.path();
        if path.extension().is_none_or(|e| e != "icc") {
            continue;
        }
        let name = path.file_stem().unwrap().to_str().unwrap();
        let transfer: u8 = name.rsplit('-').next().unwrap().parse()?;
        let bytes = std::fs::read(&path)?;
        let signature = if &bytes[16..20] == b"GRAY" {
            b"kTRC"
        } else {
            b"rTRC"
        };
        let read =
            |offset| u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        let entry = (0..read(128))
            .map(|i| 132 + i * 12)
            .find(|&offset| &bytes[offset..offset + 4] == signature)
            .ok_or("missing TRC")?;
        let curve = &bytes[read(entry + 4)..][..read(entry + 8)];
        let metrics = results.entry(transfer).or_default();
        metrics.profiles += 1;
        for code in 0..=65535 {
            let x = f64::from(code) / 65535.0;
            let actual = reference::curve(curve, x);
            let expected = reference::analytic(transfer, x);
            // Log encodings have an interval-valued inverse at encoded zero.
            if code == 0 && matches!(transfer, 9 | 10) {
                let cutoff = if transfer == 9 {
                    0.01
                } else {
                    10_f64.sqrt() / 1000.0
                };
                assert!((0.0..=cutoff).contains(&actual));
                continue;
            }
            assert!(actual.is_finite() && expected.is_finite());
            let error = (actual - expected).abs();
            if error > metrics.max_abs {
                metrics.max_abs = error;
                metrics.at = x;
            }
            metrics.zeros_with_nonzero_reference += usize::from(actual == 0.0 && expected > 0.0);
        }
    }
    assert_eq!(results.values().map(|r| r.profiles).sum::<usize>(), 334);
    println!(
        "transfer\tprofiles\tmax_abs_linear_error\tat_encoded\tzero_with_nonzero_reference_cells"
    );
    for (transfer, m) in results {
        println!(
            "{transfer}\t{}\t{:.17e}\t{:.17e}\t{}",
            m.profiles, m.max_abs, m.at, m.zeros_with_nonzero_reference
        );
    }
    Ok(())
}
