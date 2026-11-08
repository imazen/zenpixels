//! Exact qualified ICC synthesis versus uncached group decode and cached lookup.
#![forbid(unsafe_code)]
#[allow(dead_code, unused_imports)]
#[path = "../../zenpixels-convert/src/icc_profiles/canonical.rs"]
mod canonical;
#[allow(dead_code)]
#[path = "../../zenpixels-convert/src/icc_profiles/cicp_bundle_index.rs"]
mod index;
#[path = "../../zenpixels-convert/src/icc_profiles/synthesis.rs"]
mod synthesis;
use zenbench::prelude::*;
use zenpixels_convert::{Cicp, icc_profiles::SynthesizedIcc};
fn generate(gray: bool, p: u8, t: u8) -> Vec<u8> {
    if gray {
        synthesis::gray(p, t)
    } else {
        synthesis::rgb(p, t)
    }
    .expect("qualified recipe")
}
fn lookup(gray: bool, p: u8, t: u8) -> SynthesizedIcc {
    if gray {
        SynthesizedIcc::for_gray_cicp(Cicp::new(p, t, 0, true))
    } else {
        SynthesizedIcc::for_cicp(Cicp::new(p, t, 0, true))
    }
}
fn benches(suite: &mut Suite) {
    for (label, gray, p, t) in [
        ("rgb-p3-srgb", false, 12, 13),
        ("rgb-2020-bt2020", false, 9, 14),
        ("rgb-2020-pq", false, 9, 16),
        ("gray-p3-srgb", true, 12, 13),
        ("gray-pq", true, 9, 16),
        ("gray-log100", true, 1, 9),
    ] {
        let expected = generate(gray, p, t);
        let SynthesizedIcc::Profile(bundled) = lookup(gray, p, t) else {
            panic!("missing bundle");
        };
        assert_eq!(
            expected.as_slice(),
            bundled.as_ref(),
            "recipe/bundle mismatch: {label}"
        );
        let profiles = if gray {
            &index::GRAY_BUNDLE_PROFILES[..]
        } else {
            &index::BUNDLE_PROFILES[..]
        };
        let profile = profiles
            .iter()
            .find(|e| e.primaries == p && e.transfer == t)
            .unwrap();
        let group = &index::BUNDLE_GROUPS[profile.group_index];
        let block =
            &index::CICP_BUNDLE_LZ4[group.blob_offset..group.blob_offset + group.compressed_len];
        let size = group.decompressed_len;
        println!(
            "fixture {label}: profile={} group_compressed={} group_decoded={}",
            expected.len(),
            block.len(),
            size
        );
        suite.group(label, |g| {
            g.bench("qualified-synthesis", move |b| {
                b.iter(|| black_box(generate(gray, black_box(p), black_box(t))))
            });
            g.bench("uncached-group-decode", move |b| {
                b.iter(|| {
                    let mut out = vec![0u8; size];
                    let n = lz4_flex::block::decompress_into(black_box(block), &mut out).unwrap();
                    assert_eq!(n, size);
                    black_box(out)
                })
            });
            g.bench("cached-bundle-lookup", move |b| {
                b.iter(|| black_box(lookup(gray, black_box(p), black_box(t))))
            });
        });
    }
}
zenbench::main!(benches);
