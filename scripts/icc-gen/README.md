# icc-gen

Internal, unpublished tools, isolated here so their build-time dependencies
(`moxcms`, optional `lcms2`, `lz4`) don't leak into the shipped `zenpixels` /
`zenpixels-convert` crates. **Neither runs during a normal build** — they
regenerate checked-in assets on demand, and the consuming crates just
`include_*!` the committed output.

Two binaries:

| Binary | Regenerates | Recipe |
|--------|-------------|--------|
| `icc-gen` (default) | `zenpixels/src/icc/icc_table_{rgb,gray}.inc` — the ICC *identification* hash tables, from a corpus of ICC profiles | `just icc-gen` |
| `cicp_bundle_gen` | `zenpixels-convert/src/profiles/cicp_bundle.lz4` + `src/icc_profiles/cicp_bundle_index.rs` — the bundled, full-coverage CICP→ICC blob | `just cicp-bundle-gen` |

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

# `cicp_bundle_gen` — full-coverage CICP→ICC bundle

Generates the bundled ICC profile blob that backs
`zenpixels_convert::icc_profiles::synthesize_icc_for_cicp`, giving it a profile
for the **entire** assigned ITU-T H.273 grid (11 colour-primaries × 16
transfer-characteristics) **with no CMS at runtime**. This is what makes moxcms
a build-time generator only — a default (no-feature) build of `zenpixels-convert`
resolves any in-grid CICP from the committed blob, not from a CMS.

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

- **`zenpixels-convert/src/profiles/cicp_bundle.lz4`** — 16 LZ4-HC(12) blocks
  (one per transfer that yields profiles) concatenated into one asset (~27 KB).
  Grouping by transfer is load-bearing: LZ4's match window is 64 KiB, so
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
