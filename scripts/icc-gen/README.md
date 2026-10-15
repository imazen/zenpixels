# icc-gen

Internal, unpublished tools, isolated here so their build-time dependencies
(`moxcms`, optional `lcms2`, `lz4`) don't leak into the shipped `zenpixels` /
`zenpixels-convert` crates. **Neither runs during a normal build** — they
regenerate checked-in assets on demand, and the consuming crates just
`include_*!` the committed output.

Three binaries:

| Binary | Regenerates | Recipe |
|--------|-------------|--------|
| `icc-gen` (default) | `zenpixels/src/icc/icc_table_{rgb,gray}.inc` — the ICC *identification* hash tables, from a corpus of ICC profiles | `just icc-gen` |
| `cicp_bundle_gen` | `zenpixels-convert/src/profiles/cicp_bundle.lz4` + `src/icc_profiles/cicp_bundle_index.rs` — the bundled CICP→ICC blob | `just cicp-bundle-gen` |
| `probe42` | nothing — diagnostic. Measures a profile's →sRGB identity error (moxcms RGB lattice, per rendering intent): the ground truth for "should `is_common_srgb` recognize this?" (zenpipe#42) | `cargo run -p icc-gen --release --bin probe42 -- <profile.icc>...` |

---

# `icc-gen` — ICC identification hash tables

Regenerates `zenpixels/src/icc/icc_table_rgb.inc` and `icc_table_gray.inc` from
a directory of ICC profiles.

## Usage

From the repo root:

```
just icc-gen
```

Or directly:

```
cargo run -p icc-gen --release -- <icc-cache-dir> [<bundled-dir> ...] <out-dir>
```

The default invocation uses:
- input: `~/.cache/zenpixels-icc/` + `zenpixels-convert/src/profiles/`
- output: `zenpixels/src/icc/`

## Optional `lcms2-crosscheck`

To cross-check profiles against Little CMS 2 (requires liblcms2 installed
on the system):

```
cargo run -p icc-gen --release --features lcms2-crosscheck -- ...
```

The default build uses only `moxcms` — no system libraries required.

## What it writes

Each RGB entry is `(hash, ColorPrimaries, TransferFunction, max_u16_err, intent_mask)`;
each gray entry is `(hash, TransferFunction, max_u16_err, intent_mask)`.

`intent_mask` is an empirical bitmask derived by running a test pixel ramp
through moxcms at each rendering intent and comparing against a synthetic
reference profile built from the identified `(primaries, transfer)`. The bits
are defined in `zenpixels::icc` (`INTENT_COLORIMETRIC_SAFE`,
`INTENT_PERCEPTUAL_SAFE`, `INTENT_SATURATION_SAFE`).

---

# `cicp_bundle_gen` — validated CICP→ICC bundle

Generates the bundled ICC profile blob used by
`zenpixels_convert::icc_profiles::{synthesize_icc_for_cicp, synthesize_gray_icc_for_cicp}`.
The assigned H.273 grid has 176 pairs: 159 yield RGB profiles, 16 XYZ-primary
pairs are refused for RGB, and sRGB (1,13) needs no profile. BT.709 (1,1)
has its own BT.709 curve. Gray has 175 mappings to 64 unique profiles,
including primaries 10: its equal-energy white and transfer are valid without
an RGB gamut matrix. Runtime lookup needs no CMS.

Before packing an RGB profile, the generator parses its rXYZ/gXYZ/bXYZ tags
and requires |determinant| > 0.01 and every colorant value within 0.001 of
an independent [H.273](https://www.itu.int/rec/T-REC-H.273) Table 2 computation,
Bradford-adapted to ICC D50. This computation does not call moxcms primaries
or adaptation helpers. Any failure aborts generation, except the explicitly
commented primaries-10 exclusion: its rank-1 matrix is reported and omitted.
Unexpected synthesis failures also abort rather than silently reducing coverage.

The separate `zenpixels-convert/tests/icc_colourimetry.rs` oracle parses the
shipped ICC bytes, independently solves the primaries/white equations, and
checks every RGB colorant and RGB/gray TRC. Transfer tolerances are absolute
linear-light errors: 0.0005 for SDR, 0.001 for PQ/HLG. At encoded zero,
logarithmic TC9/TC10 curves are checked against their valid inverse interval;
the pointwise tolerance still applies at every other sampled value. Gray
source whites are checked within 0.00002; their adaptation to the generator's
5003 K D50 approximation is checked within 0.001 of ICC D50.
These transfer tolerances do not promise relative precision in PQ's deep toe. HLG uses the encoded BT.2100 neutral
reference-display EOTF, with system gamma 1.2 (1000 cd/m²); it is not merely
the inverse scene OETF. ST 428's nominal 52.37/48 endpoint saturates at 1 in
the u16 `curv` LUT; CICP-native signaling avoids this ICC limitation.

## Usage

From the repo root:

```
just cicp-bundle-gen          # regenerate the blob + index
just cicp-bundle-dry-run      # print counts/sizes without writing
just cicp-bundle-update       # regenerate, then verify (golden sha256 + moxcms byte-equality)
```

Or directly:

```
cargo run -p icc-gen --release --bin cicp_bundle_gen [--no-write] [--out-dir <zenpixels-convert dir>]
```

Run it whenever you bump moxcms (the synthesized bytes can shift); the
`golden_blob_sha256_is_pinned` test catches an accidental regen, and the
`cms-moxcms`-gated `blob_decodes_byte_identical_to_moxcms` test detects a moxcms
version bump that quietly changes the canonical bytes. After a deliberate regen,
update the pinned hash in `zenpixels-convert/src/icc_profiles/cicp_bundle.rs`:

```
sha256sum zenpixels-convert/src/profiles/cicp_bundle.lz4
```

## What it writes

- **`zenpixels-convert/src/profiles/cicp_bundle.lz4`** — 16 LZ4 blocks (pure-Rust
  `lz4_flex`, one per transfer that yields profiles) concatenated into one asset
  (35,298 bytes with the pinned generator dependencies). Grouping by transfer is load-bearing: LZ4's match window is 64 KiB, so
  clustering the identical TRC/LUT payload (same transfer, varying primaries)
  is what lets the compressor collapse it.
- **`src/icc_profiles/cicp_bundle_index.rs`** — a generated module with the
  per-group table (`{transfer, blob_offset, compressed_len, decompressed_len}`)
  and the per-`(primaries, transfer)` profile locator
  (`{group_index, offset_in_group, len}`, sorted for binary search). The blob is
  embedded via `include_bytes!`; at runtime only the touched transfer group is
  LZ4-decoded (pure-Rust `lz4_flex`, `no_std`), cached, and sliced.

## Determinism

moxcms's `encode()` stamps the current wall-clock time into the ICC header's
creation-date field (bytes 24..36). The generator zeroes that field so the blob
is **reproducible** — regenerating without changing moxcms produces a
byte-identical asset (and a stable golden sha256). The field is purely
informational and carries no colorimetry.

## Ultra HDR corpus profile (2026-10-13)

The profile in `ultrahdr-conformance/valid/jpeg/awesome-gain-maps/rgba_uhdr.jpg`
was missing from the identification table. Extract it unchanged with
`exiftool -b -ICC_Profile <corpus-file> > <input-dir>/rgba_uhdr.icc`, then run
`just icc-hdr-profile-check <input-dir> <output-dir>`. Do not replace the full
table with this single-input output; copy its sorted entry into the table.

The generator produced normalized hash `0x11c9534ccea885a0`, Display P3 with
sRGB transfer, maximum u16 reference error 1, and `Safe::AnyIntent`. The run
used the optional Little CMS cross-check and reported no structural/empirical
intent disagreements. JPEG's adapter regression exercises the original corpus
file; no new binary fixture is committed here.
