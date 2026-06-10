//! Generate the bundled transfer-grouped ICC blob for full CICP coverage.
//!
//! Enumerates the full ITU-T H.273 assigned grid (11 colour-primaries × 16
//! transfer-characteristics = 176 combinations), synthesizes each profile via
//! moxcms exactly as `zenpixels_convert::cms_moxcms::icc_bytes_for_cicp` does at
//! runtime, deduplicates the resulting bytes, groups the unique profiles by
//! transfer code, LZ4-HC(12) compresses each group, and writes:
//!
//! * a concatenated blob → `zenpixels-convert/src/profiles/cicp_bundle.lz4`
//! * a generated Rust index module →
//!   `zenpixels-convert/src/icc_profiles/cicp_bundle_index.rs`
//!
//! The grouping is essential: LZ4's match window is 64 KiB, so clustering the
//! profiles that share an identical TRC/LUT payload (same transfer, varying
//! primaries) lets the compressor find the long cross-profile matches. The
//! decompressed group is sliced at runtime to recover each profile.
//!
//! Two combinations (`primaries=1` BT.709 with `transfer=1` BT.709, and
//! `primaries=1` with `transfer=13` sRGB) are the universally-assumed sRGB /
//! BT.709 default — `synthesize_icc_for_cicp` returns `NotNeeded` for them
//! before reaching the blob — so they are excluded from the bundle, leaving the
//! 174 covered combinations.
//!
//! Usage (run from the repo root so the relative output paths resolve):
//!   `cargo run -p icc-gen --release --bin cicp_bundle_gen [--no-write] [--out-dir <dir>]`
//!
//! `--no-write` prints the analysis (counts, sizes) without touching any files;
//! used by the byte-equality test to recompute the blob in memory and diff it
//! against the committed asset.

use std::collections::BTreeMap;
use std::path::PathBuf;

use lz4::block::{compress, CompressionMode};
use moxcms::{
    CicpColorPrimaries, CicpProfile, ColorProfile, MatrixCoefficients, TransferCharacteristics,
};

/// LZ4-HC compression level used for every group. 12 is the max
/// (`LZ4HC_CLEVEL_MAX`); the extra encode time is irrelevant for a build-time
/// generator and squeezes out the smallest blob.
const LZ4_HC_LEVEL: i32 = 12;

/// The assigned H.273 colour-primaries codes (everything that is neither
/// `Reserved` nor `Unspecified`). 11 entries.
const ASSIGNED_PRIMARIES: &[u8] = &[1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22];

/// The assigned H.273 transfer-characteristics codes (everything that is
/// neither `Reserved` nor `Unspecified`). 16 entries.
const ASSIGNED_TRANSFERS: &[u8] = &[1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18];

/// Synthesize ICC bytes for a raw CICP `(primaries, transfer)` exactly as the
/// runtime `icc_bytes_for_cicp` does, so the committed bytes match what a
/// `cms-moxcms` build would generate.
///
/// Returns `None` when moxcms cannot represent the combination (no populated
/// `red_trc`), mirroring the runtime `CmsUnsupported` outcome.
fn synth_icc(primaries: u8, transfer: u8) -> Option<Vec<u8>> {
    let color_primaries = CicpColorPrimaries::try_from(primaries).ok()?;
    let transfer_characteristics = TransferCharacteristics::try_from(transfer).ok()?;
    // Matrix coefficients don't affect an RGB ICC's colorimetry; default to
    // Identity exactly as the runtime path does.
    let matrix_coefficients =
        MatrixCoefficients::try_from(0u8).unwrap_or(MatrixCoefficients::Identity);

    let profile = ColorProfile::new_from_cicp(CicpProfile {
        color_primaries,
        transfer_characteristics,
        matrix_coefficients,
        // Range never changes an RGB ICC's bytes; fix it to full range so the
        // generated profile is deterministic and matches the runtime default
        // for CICP synthesis.
        full_range: true,
    });

    // A populated red_trc is moxcms's signal that synthesis was faithful (same
    // gate as the runtime path). No TRC ⇒ can't represent this CICP.
    profile.red_trc.as_ref()?;
    profile.encode().ok()
}

/// The sRGB / BT.709 default combinations that `synthesize_icc_for_cicp` short
/// circuits to `NotNeeded` before the blob is consulted. Excluded from the
/// bundle so the index never points at bytes that are never requested.
fn is_srgb_default(primaries: u8, transfer: u8) -> bool {
    matches!(primaries, 1 | 2) && matches!(transfer, 1 | 2 | 13)
}

/// One profile occurrence in the grid, before dedup.
struct GridEntry {
    primaries: u8,
    transfer: u8,
    bytes: Vec<u8>,
}

/// A built, transfer-grouped, compressed bundle plus its index metadata.
struct BuiltBundle {
    /// The concatenated LZ4-HC blocks (one per group), in group order.
    blob: Vec<u8>,
    /// Per-group records, in the order they appear in `blob`.
    groups: Vec<GroupRecord>,
    /// Per-`(primaries, transfer)` profile locator, sorted for determinism.
    profiles: Vec<ProfileRecord>,
    /// Diagnostics.
    total_combos: usize,
    profile_combos: usize,
    unique_profiles: usize,
    raw_unique_bytes: usize,
}

struct GroupRecord {
    transfer: u8,
    blob_offset: usize,
    compressed_len: usize,
    decompressed_len: usize,
    /// Number of distinct profiles packed into this group's decoded bytes.
    profile_count: usize,
}

struct ProfileRecord {
    primaries: u8,
    transfer: u8,
    group_index: usize,
    offset_in_group: usize,
    len: usize,
}

/// Build the bundle in memory: enumerate, synthesize, dedup, group by transfer,
/// compress each group. Deterministic ordering throughout.
fn build_bundle() -> BuiltBundle {
    // 1. Enumerate the full grid, synthesizing each combo. Deterministic order:
    //    primaries ascending, then transfer ascending.
    let mut total_combos = 0usize;
    let mut grid: Vec<GridEntry> = Vec::new();
    for &primaries in ASSIGNED_PRIMARIES {
        for &transfer in ASSIGNED_TRANSFERS {
            total_combos += 1;
            if is_srgb_default(primaries, transfer) {
                continue;
            }
            match synth_icc(primaries, transfer) {
                Some(bytes) => grid.push(GridEntry {
                    primaries,
                    transfer,
                    bytes,
                }),
                None => {
                    // moxcms couldn't represent this combo. Record it so the
                    // summary surfaces any unexpected gaps; it simply isn't in
                    // the bundle (runtime falls through to CmsUnsupported).
                    eprintln!(
                        "  note: ({primaries}, {transfer}) not representable by moxcms — skipped"
                    );
                }
            }
        }
    }
    let profile_combos = grid.len();

    // 2. Group by transfer, preserving primaries-ascending order within each
    //    group. Within a group the identical TRC/LUT payload clusters, which is
    //    what makes LZ4 (64 KiB window) collapse it.
    //
    //    Dedup is content-addressed *within* a group: if two primaries under
    //    the same transfer produce byte-identical profiles, they share one copy
    //    and both ProfileRecords point at the same offset/len. (Cross-group
    //    dedup isn't attempted — different transfers virtually never collide,
    //    and keeping groups self-contained keeps the runtime slice trivial.)
    let mut transfers_in_order: Vec<u8> = grid.iter().map(|e| e.transfer).collect();
    transfers_in_order.sort_unstable();
    transfers_in_order.dedup();

    let mut blob: Vec<u8> = Vec::new();
    let mut groups: Vec<GroupRecord> = Vec::new();
    let mut profiles: Vec<ProfileRecord> = Vec::new();
    let mut unique_profiles = 0usize;
    let mut raw_unique_bytes = 0usize;

    for (group_index, &transfer) in transfers_in_order.iter().enumerate() {
        // Collect this group's entries (primaries ascending).
        let mut entries: Vec<&GridEntry> =
            grid.iter().filter(|e| e.transfer == transfer).collect();
        entries.sort_by_key(|e| e.primaries);

        // Pack the group's decoded bytes, deduping identical profiles.
        let mut decoded: Vec<u8> = Vec::new();
        // Map profile bytes → offset within `decoded` (content-addressed dedup).
        let mut seen: BTreeMap<Vec<u8>, usize> = BTreeMap::new();
        let mut group_profile_count = 0usize;

        for e in &entries {
            let offset_in_group = if let Some(&off) = seen.get(&e.bytes) {
                off
            } else {
                let off = decoded.len();
                decoded.extend_from_slice(&e.bytes);
                seen.insert(e.bytes.clone(), off);
                unique_profiles += 1;
                raw_unique_bytes += e.bytes.len();
                group_profile_count += 1;
                off
            };
            profiles.push(ProfileRecord {
                primaries: e.primaries,
                transfer: e.transfer,
                group_index,
                offset_in_group,
                len: e.bytes.len(),
            });
        }

        // Compress the group as one raw LZ4-HC block (no size prefix; the
        // decompressed length lives in the index).
        let compressed = compress(
            &decoded,
            Some(CompressionMode::HIGHCOMPRESSION(LZ4_HC_LEVEL)),
            false,
        )
        .expect("LZ4-HC compression of ICC group failed");

        let blob_offset = blob.len();
        let compressed_len = compressed.len();
        let decompressed_len = decoded.len();
        blob.extend_from_slice(&compressed);

        groups.push(GroupRecord {
            transfer,
            blob_offset,
            compressed_len,
            decompressed_len,
            profile_count: group_profile_count,
        });
    }

    // Sort the profile locator by (primaries, transfer) for a deterministic,
    // binary-searchable table.
    profiles.sort_by(|a, b| (a.primaries, a.transfer).cmp(&(b.primaries, b.transfer)));

    BuiltBundle {
        blob,
        groups,
        profiles,
        total_combos,
        profile_combos,
        unique_profiles,
        raw_unique_bytes,
    }
}

/// Render the generated Rust index module.
fn render_index(b: &BuiltBundle) -> String {
    let mut s = String::new();
    s.push_str(
        "// @generated by `cargo run -p icc-gen --bin cicp_bundle_gen` — do not edit by hand.\n\
         //\n\
         // Index for the transfer-grouped, LZ4-HC compressed CICP ICC bundle\n\
         // (`../profiles/cicp_bundle.lz4`). Regenerate with the generator above\n\
         // after a moxcms version bump; the golden sha256 test will flag drift.\n\n",
    );

    // Group table.
    s.push_str("/// One LZ4-HC compressed group inside the bundle blob, keyed by\n");
    s.push_str("/// transfer-characteristics code. Each group's decoded bytes pack every\n");
    s.push_str("/// unique profile that shares that transfer.\n");
    s.push_str("#[derive(Clone, Copy)]\n");
    s.push_str("pub(crate) struct BundleGroup {\n");
    s.push_str("    /// H.273 transfer-characteristics code this group covers.\n");
    s.push_str("    pub transfer: u8,\n");
    s.push_str("    /// Byte offset of the group's LZ4 block within `CICP_BUNDLE_LZ4`.\n");
    s.push_str("    pub blob_offset: usize,\n");
    s.push_str("    /// Length of the group's LZ4 block.\n");
    s.push_str("    pub compressed_len: usize,\n");
    s.push_str("    /// Exact size of the group's decoded bytes (the `decompress_into` buffer).\n");
    s.push_str("    pub decompressed_len: usize,\n");
    s.push_str("}\n\n");

    // Profile table.
    s.push_str("/// Locator for one synthesized profile: which group holds it, and the\n");
    s.push_str("/// `[offset, offset + len)` slice of that group's *decoded* bytes.\n");
    s.push_str("#[derive(Clone, Copy)]\n");
    s.push_str("pub(crate) struct BundleProfile {\n");
    s.push_str("    /// H.273 colour-primaries code.\n");
    s.push_str("    pub primaries: u8,\n");
    s.push_str("    /// H.273 transfer-characteristics code.\n");
    s.push_str("    pub transfer: u8,\n");
    s.push_str("    /// Index into [`BUNDLE_GROUPS`].\n");
    s.push_str("    pub group_index: usize,\n");
    s.push_str("    /// Byte offset of this profile within the group's decoded bytes.\n");
    s.push_str("    pub offset_in_group: usize,\n");
    s.push_str("    /// Length of this profile in bytes.\n");
    s.push_str("    pub len: usize,\n");
    s.push_str("}\n\n");

    // The compressed blob, embedded via include_bytes.
    s.push_str("/// The concatenated LZ4-HC compressed groups. Embedded at compile time.\n");
    s.push_str(
        "pub(crate) static CICP_BUNDLE_LZ4: &[u8] = include_bytes!(\"../profiles/cicp_bundle.lz4\");\n\n",
    );

    s.push_str(&format!(
        "/// Number of transfer groups in the bundle.\n\
         pub(crate) const NUM_GROUPS: usize = {};\n\n",
        b.groups.len()
    ));

    // Group array.
    s.push_str("/// Group table, indexed by group index (also the order in the blob).\n");
    s.push_str(&format!(
        "pub(crate) static BUNDLE_GROUPS: [BundleGroup; NUM_GROUPS] = [\n"
    ));
    for g in &b.groups {
        s.push_str(&format!(
            "    BundleGroup {{ transfer: {}, blob_offset: {}, compressed_len: {}, decompressed_len: {} }}, // {} profile(s)\n",
            g.transfer, g.blob_offset, g.compressed_len, g.decompressed_len, g.profile_count
        ));
    }
    s.push_str("];\n\n");

    // Profile array.
    s.push_str("/// Profile locator table, sorted by `(primaries, transfer)` for binary search.\n");
    s.push_str(&format!(
        "pub(crate) static BUNDLE_PROFILES: [BundleProfile; {}] = [\n",
        b.profiles.len()
    ));
    for p in &b.profiles {
        s.push_str(&format!(
            "    BundleProfile {{ primaries: {}, transfer: {}, group_index: {}, offset_in_group: {}, len: {} }},\n",
            p.primaries, p.transfer, p.group_index, p.offset_in_group, p.len
        ));
    }
    s.push_str("];\n");

    s
}

struct Args {
    write: bool,
    out_convert_dir: PathBuf,
}

fn parse_args() -> Args {
    let mut write = true;
    // Default assumes the generator is run from the repo root.
    let mut out_convert_dir = PathBuf::from("zenpixels-convert");
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--no-write" => write = false,
            "--out-dir" => {
                out_convert_dir = PathBuf::from(
                    it.next()
                        .expect("--out-dir requires a path to the zenpixels-convert crate"),
                );
            }
            other => {
                eprintln!("unknown argument: {other}");
                eprintln!(
                    "usage: cargo run -p icc-gen --bin cicp_bundle_gen [--no-write] [--out-dir <zenpixels-convert dir>]"
                );
                std::process::exit(2);
            }
        }
    }
    Args {
        write,
        out_convert_dir,
    }
}

fn main() {
    let args = parse_args();
    let bundle = build_bundle();

    eprintln!("CICP→ICC bundle (transfer-grouped, LZ4-HC{LZ4_HC_LEVEL}):");
    eprintln!("  total grid combos:        {}", bundle.total_combos);
    eprintln!(
        "  profile-yielding combos:  {} (excludes {} sRGB-default NotNeeded)",
        bundle.profile_combos,
        bundle.total_combos - bundle.profile_combos
    );
    eprintln!("  unique profiles:          {}", bundle.unique_profiles);
    eprintln!("  groups (by transfer):     {}", bundle.groups.len());
    eprintln!(
        "  raw unique bytes:         {} ({:.1} KiB)",
        bundle.raw_unique_bytes,
        bundle.raw_unique_bytes as f64 / 1024.0
    );
    eprintln!(
        "  compressed blob:          {} ({:.1} KiB)",
        bundle.blob.len(),
        bundle.blob.len() as f64 / 1024.0
    );
    eprintln!(
        "  compression ratio:        {:.2}x",
        bundle.raw_unique_bytes as f64 / bundle.blob.len() as f64
    );

    if !args.write {
        eprintln!("(--no-write: not touching any files)");
        return;
    }

    let blob_path = args.out_convert_dir.join("src/profiles/cicp_bundle.lz4");
    let index_path = args
        .out_convert_dir
        .join("src/icc_profiles/cicp_bundle_index.rs");

    if let Some(parent) = blob_path.parent() {
        std::fs::create_dir_all(parent).expect("create profiles dir");
    }
    if let Some(parent) = index_path.parent() {
        std::fs::create_dir_all(parent).expect("create icc_profiles dir");
    }

    std::fs::write(&blob_path, &bundle.blob).expect("write blob");
    std::fs::write(&index_path, render_index(&bundle)).expect("write index");

    eprintln!("\nwrote:");
    eprintln!("  {} ({} bytes)", blob_path.display(), bundle.blob.len());
    eprintln!("  {}", index_path.display());
}
