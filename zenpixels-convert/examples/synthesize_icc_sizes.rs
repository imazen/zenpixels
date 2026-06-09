//! Brute-force every assigned CICP (primaries × transfer) through
//! [`synthesize_icc_for_cicp`] and report the synthesized ICC profile sizes.
//!
//! ```text
//! cargo run --example synthesize_icc_sizes
//! ```
//!
//! This always exercises the `cms-moxcms` path: a dev-dependency on this crate with
//! `cms-moxcms` enables it for every dev/example target via Cargo feature
//! unification, so the no-CMS `NeedsCms` outcome isn't reachable from the example
//! (it's covered by the unit tests' `#[cfg(not(feature = "cms-moxcms"))]` branch).
//!
//! Matrix coefficients are irrelevant to an RGB ICC and the full-range flag does
//! not change the encoded colorimetry, so the grid fixes both (matrix 0, limited
//! range). The point is the *byte footprint* of embedding a synthesized profile —
//! which matters most on tiny images, where a fixed ICC blob dominates the output.

use std::borrow::Cow;
use zenpixels_convert::Cicp;
use zenpixels_convert::icc_profiles::{SynthesizedIcc, synthesize_icc_for_cicp};

/// H.273 Table 2 — assigned colour primaries code points.
const PRIMARIES: &[(u8, &str)] = &[
    (1, "BT.709"),
    (4, "BT.470M"),
    (5, "BT.470BG"),
    (6, "BT.601"),
    (7, "SMPTE240M"),
    (8, "GenericFilm"),
    (9, "BT.2020"),
    (10, "SMPTE428/XYZ"),
    (11, "SMPTE431/DCI-P3"),
    (12, "SMPTE432/DisplayP3"),
    (22, "EBU3213"),
];

/// H.273 Table 3 — assigned transfer characteristics code points.
const TRANSFERS: &[(u8, &str)] = &[
    (1, "BT.709"),
    (4, "Gamma2.2"),
    (5, "Gamma2.8"),
    (6, "BT.601"),
    (7, "SMPTE240M"),
    (8, "Linear"),
    (9, "Log100"),
    (10, "Log100Sqrt10"),
    (11, "IEC61966-2-4"),
    (12, "BT.1361"),
    (13, "sRGB"),
    (14, "BT.2020-10bit"),
    (15, "BT.2020-12bit"),
    (16, "PQ/SMPTE2084"),
    (17, "SMPTE428"),
    (18, "HLG"),
];

fn main() {
    let mut sizes: Vec<(u8, u8, usize, &'static str)> = Vec::new();
    let (mut bundled, mut generated, mut not_needed, mut needs_cms, mut unsupported) =
        (0usize, 0usize, 0usize, 0usize, 0usize);

    println!(
        "{:>20} | {:>14} | {:>7} | outcome",
        "primaries", "transfer", "bytes"
    );
    println!("{}", "-".repeat(70));

    for &(p, pn) in PRIMARIES {
        for &(t, tn) in TRANSFERS {
            let cicp = Cicp::new(p, t, 0, false);
            let (bytes_col, outcome) = match synthesize_icc_for_cicp(cicp) {
                SynthesizedIcc::Profile(b) => {
                    let n = b.len();
                    let kind = match b {
                        Cow::Borrowed(_) => {
                            bundled += 1;
                            "bundled &'static"
                        }
                        Cow::Owned(_) => {
                            generated += 1;
                            "moxcms-generated"
                        }
                    };
                    sizes.push((p, t, n, kind));
                    (n.to_string(), format!("Profile ({kind})"))
                }
                SynthesizedIcc::NotNeeded => {
                    not_needed += 1;
                    (
                        "-".to_string(),
                        "NotNeeded (sRGB/BT.709 default)".to_string(),
                    )
                }
                SynthesizedIcc::NeedsCms => {
                    needs_cms += 1;
                    (
                        "-".to_string(),
                        "NeedsCms (build without cms-moxcms)".to_string(),
                    )
                }
                SynthesizedIcc::CmsUnsupported => {
                    unsupported += 1;
                    ("-".to_string(), "CmsUnsupported".to_string())
                }
                // SynthesizedIcc is #[non_exhaustive]; a future outcome embeds nothing.
                _ => ("-".to_string(), "unknown outcome".to_string()),
            };
            println!("{pn:>20} | {tn:>14} | {bytes_col:>7} | {outcome}");
        }
    }

    let n = sizes.len();
    let total: usize = sizes.iter().map(|s| s.2).sum();
    let combos = PRIMARIES.len() * TRANSFERS.len();
    println!(
        "\n=== summary ({combos} combos = {} primaries × {} transfers) ===",
        PRIMARIES.len(),
        TRANSFERS.len()
    );
    println!("Profile:        {n}  (bundled {bundled}, moxcms-generated {generated})");
    println!("NotNeeded:      {not_needed}");
    println!("NeedsCms:       {needs_cms}");
    println!("CmsUnsupported: {unsupported}");

    if n > 0 {
        let kb = |b: usize| b as f64 / 1024.0;
        let min = sizes.iter().map(|s| s.2).min().unwrap();
        let max = sizes.iter().map(|s| s.2).max().unwrap();
        println!(
            "\nICC size: min {} B ({:.2} KB), max {} B ({:.2} KB), mean {:.0} B ({:.2} KB)",
            min,
            kb(min),
            max,
            kb(max),
            total as f64 / n as f64,
            kb(total) / n as f64
        );
        let mut by_size = sizes.clone();
        by_size.sort_by_key(|s| std::cmp::Reverse(s.2));
        println!("Largest synthesized profiles:");
        for &(p, t, sz, kind) in by_size.iter().take(8) {
            let pn = PRIMARIES
                .iter()
                .find(|x| x.0 == p)
                .map(|x| x.1)
                .unwrap_or("?");
            let tn = TRANSFERS
                .iter()
                .find(|x| x.0 == t)
                .map(|x| x.1)
                .unwrap_or("?");
            println!("  {sz:>6} B ({:.2} KB)  {pn} / {tn}  [{kind}]", kb(sz));
        }
    }
}
