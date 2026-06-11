//! Generate the bundled transfer-grouped ICC blob for full CICP coverage.
//!
//! Enumerates the full ITU-T H.273 assigned grid (11 colour-primaries × 16
//! transfer-characteristics = 176 combinations), synthesizes each profile via
//! moxcms exactly as `zenpixels_convert::cms_moxcms::icc_bytes_for_cicp` does at
//! runtime, deduplicates the resulting bytes, groups the unique profiles by
//! transfer code, LZ4 (`lz4_flex`) compresses each group, and writes:
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

use moxcms::{
    CicpColorPrimaries, CicpProfile, ColorProfile, MatrixCoefficients, TransferCharacteristics,
};

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
    let mut bytes = profile.encode().ok()?;
    normalize_creation_timestamp(&mut bytes);
    Some(bytes)
}

/// Zero the ICC header's creation date/time field (bytes 24..36).
///
/// moxcms's `encode()` stamps `ColorDateTime::now()` — the current wall-clock
/// time — into every generated profile (moxcms `writer.rs`). That makes the
/// bytes non-deterministic across regenerations and would make the committed
/// blob (and its golden sha256) unstable. The field is purely informational and
/// carries no colorimetry; zenpixels' own normalized ICC hash already treats
/// bytes 24..36 as metadata-to-ignore. Zeroing it yields a reproducible,
/// deterministic profile. The runtime roundtrip test masks the same field when
/// comparing against a fresh moxcms run.
fn normalize_creation_timestamp(icc: &mut [u8]) {
    if icc.len() >= 36 {
        icc[24..36].fill(0);
    }
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
    /// The concatenated LZ4 blocks (one per group), in group order.
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
        let mut entries: Vec<&GridEntry> = grid.iter().filter(|e| e.transfer == transfer).collect();
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

        // Compress the group as one raw LZ4 block (no size prefix; the
        // decompressed length lives in the index). Same pure-Rust `lz4_flex`
        // used at runtime — keeps the whole pipeline on one safe LZ4 impl and
        // off any C dependency. The raw block (no size prefix) is what
        // `lz4_flex::block::decompress_into` reads against the index length.
        let compressed = lz4_flex::block::compress(&decoded);

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
    profiles.sort_by_key(|p| (p.primaries, p.transfer));

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
         // Index for the transfer-grouped, LZ4-compressed CICP ICC bundle\n\
         // (`../profiles/cicp_bundle.lz4`). Regenerate with the generator above\n\
         // after a moxcms version bump; the golden sha256 test will flag drift.\n\n",
    );

    // Group table.
    s.push_str("/// One LZ4-compressed group inside the bundle blob, keyed by\n");
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
    s.push_str("/// The concatenated LZ4-compressed groups. Embedded at compile time.\n");
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
    s.push_str("pub(crate) static BUNDLE_GROUPS: [BundleGroup; NUM_GROUPS] = [\n");
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

    eprintln!("CICP→ICC bundle (transfer-grouped, LZ4 via lz4_flex):");
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The committed blob (`zenpixels-convert/src/profiles/cicp_bundle.lz4`)
    /// must be byte-identical to a fresh in-memory regeneration. This is the
    /// `cms-moxcms`-side guard the runtime crate can't do (it recomputes the
    /// encoder): if a moxcms version bump silently shifts the synthesized bytes,
    /// the recompressed blob diverges from the committed asset and this fails,
    /// forcing a deliberate regen + golden-hash update.
    ///
    /// x86_64-only: LZ4 compressed output is not arch-stable — `lz4_flex`'s
    /// match-finder emits different (equally valid) encodings on 32-bit
    /// targets (measured 2026-06-11: byte-identical 610,170-byte raw profile
    /// content on i686 and x86_64, but a 28,327-byte blob on i686 vs the
    /// committed x86_64 28,399). x86_64 is the canonical regen arch; every
    /// arch still pins the *decoded content* via
    /// [`bundled_profiles_match_fresh_synthesis`], which is the property the
    /// runtime actually relies on.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn regenerated_blob_matches_committed() {
        let committed = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../zenpixels-convert/src/profiles/cicp_bundle.lz4");
        let on_disk = std::fs::read(&committed)
            .unwrap_or_else(|e| panic!("cannot read committed blob {}: {e}", committed.display()));
        let fresh = build_bundle();
        assert_eq!(
            fresh.blob.len(),
            on_disk.len(),
            "regenerated blob length ({}) != committed ({}) — regenerate and update the golden sha256",
            fresh.blob.len(),
            on_disk.len()
        );
        assert!(
            fresh.blob == on_disk,
            "regenerated blob bytes differ from the committed asset — \
             moxcms drift or generator change; regenerate and update the golden sha256"
        );
    }

    /// Cross-arch content guard: the committed blob, decoded through the
    /// committed index, must yield byte-identical group content (and
    /// identical profile locators) to a fresh moxcms generation. Compressed
    /// bytes may legitimately differ per arch (see
    /// [`regenerated_blob_matches_committed`]); decoded content may not —
    /// that is the invariant the runtime depends on, and it runs on every
    /// target including 32-bit.
    #[test]
    fn committed_bundle_content_matches_fresh_synthesis() {
        // The generated runtime index is self-contained (struct defs,
        // group/profile tables, `include_bytes!` of the committed blob —
        // nested `include_bytes!` resolves relative to the included file,
        // so the committed asset loads exactly as shipped). Including it
        // here decodes the committed bundle without going through the
        // runtime crate's bundled-const priority ladder.
        mod committed {
            #![allow(dead_code)]
            include!("../../../../zenpixels-convert/src/icc_profiles/cicp_bundle_index.rs");
        }

        let fresh = build_bundle();
        assert_eq!(
            committed::BUNDLE_GROUPS.len(),
            fresh.groups.len(),
            "group count drift"
        );
        assert_eq!(
            committed::BUNDLE_PROFILES.len(),
            fresh.profiles.len(),
            "profile count drift"
        );

        for (gi, (cg, fg)) in committed::BUNDLE_GROUPS
            .iter()
            .zip(&fresh.groups)
            .enumerate()
        {
            assert_eq!(cg.transfer, fg.transfer, "group {gi} transfer drift");
            let c_block =
                &committed::CICP_BUNDLE_LZ4[cg.blob_offset..cg.blob_offset + cg.compressed_len];
            let f_block = &fresh.blob[fg.blob_offset..fg.blob_offset + fg.compressed_len];
            let c_raw = lz4_flex::block::decompress(c_block, cg.decompressed_len)
                .expect("committed group decompresses");
            let f_raw = lz4_flex::block::decompress(f_block, fg.decompressed_len)
                .expect("fresh group decompresses");
            assert!(
                c_raw == f_raw,
                "group {gi} (transfer {}): committed decoded content != fresh synthesis — \
                 moxcms drift or generator change; regenerate the bundle",
                cg.transfer
            );
        }

        for (cp, fp) in committed::BUNDLE_PROFILES.iter().zip(&fresh.profiles) {
            assert_eq!(
                (
                    cp.primaries,
                    cp.transfer,
                    cp.group_index,
                    cp.offset_in_group,
                    cp.len
                ),
                (
                    fp.primaries,
                    fp.transfer,
                    fp.group_index,
                    fp.offset_in_group,
                    fp.len
                ),
                "profile locator drift"
            );
        }
    }

    /// The grid enumeration must stay at the measured 176 combos / 174
    /// profile-yielding / 16 groups. A change here means the H.273 assigned set
    /// or the sRGB-default exclusion shifted — review before accepting.
    #[test]
    fn grid_shape_is_stable() {
        let b = build_bundle();
        assert_eq!(b.total_combos, 176, "expected 11×16 assigned grid");
        assert_eq!(
            b.profile_combos, 174,
            "expected 174 profile-yielding combos"
        );
        assert_eq!(b.unique_profiles, 174);
        assert_eq!(b.groups.len(), 16, "expected 16 transfer groups");
    }
}
