# ICC original retention and new-object qualification

Verified 2026-11-09 against recognition tables at `09dc3659653c`, using
`icc-gen --inventory` (`8e120f75`) and per-object report output (`3334c5bd`).

## Original-byte coverage

All retained hashes have exact matching originals. Coverage uses the generator's
metadata-normalized FNV-1a fingerprint, independently checked on each selected
witness, plus a full-file SHA-256 digest. Profiles with similar descriptions
were never substituted for missing bytes.

| Table | Active | Deferred | Missing |
|---|---:|---:|---:|
| RGB | 271 | 174 | 0 |
| Gray | 141 | 12 | 0 |
| Total | 412 | 186 | 0 |

The live `s3://codec-corpus/icc-profiles/` prefix contains 1,224 ICC/ICM objects,
260,358,311 bytes. All 1,210 manifest entries exist, with 14 additional objects.
Every object was downloaded, size-checked and SHA-256 hashed. Source selection
prefers the archive where multiple originals share a normalized hash:

| Selected original source | Retained hashes covered |
|---|---:|
| Live ICC archive | 259 |
| Current checked-in profiles and decoded bundle | 120 |
| Exact profiles decoded from historical committed bundles | 217 |
| Embedded corpus profiles | 2 |

Thus the live R2 archive alone is incomplete. Adding current assets leaves
219 hashes whose exact originals were recovered from history/corpora. The historical bytes are retained in Git, rather than
recreated with today's generator. The eight inspected bundle/index revisions
are `210fd9c`, `2171b79`, `b6b12bb`, `b8374b9`, `75eb953`, `b12cbaf`,
`d6ac448` and `0d3d7d5`; decoding their existing LZ4 groups recovers all 217.

The two remaining originals are already embedded in corpora:

- `11c9534ccea885a0`: the 600-byte profile in Ultra HDR conformance fixture
  `valid/jpeg/awesome-gain-maps/rgba_uhdr.jpg`.
- `2dc2d3340af4d7c8`: the 520-byte Android Display P3/sRGB profile in the
  imazen-26 original mobile screenshot collection. PNG header range reads
  provided the original compressed ICC chunks; extraction preserves ICC bytes.

The retained audit receipt is `coverage-final.json` in
`~/.cache/zenpixels-icc/retention-audit/` on the audit host, r7900x. Its SHA-256
is `dfdd498c673409443aeaadf21bc792d1cf3c81b87c811ef9fb04365f6d145574`.
It records all 598 table rows, active/deferred state, selected original source,
size and full-file digest. The same directory retains the archive source map,
object listing, recovered originals and qualification log. Original sources
remain R2, committed Git assets and the existing corpora; this local receipt
does not replace those sources. No R2 objects or access settings were changed.

Ten object keys contain control characters. Listing through botocore's S3
paginator succeeded; the installed command-line listing tools failed on those
keys. Cached filenames use SHA-256 of the original key, and `sources.json`
preserves each exact key separately.

## The 14 objects absent from the manifest

Ran `icc-gen --report <new-objects> <separate-output>` with default features:
moxcms 0.9.2, lcms2 Rust bindings 6.2.0 and system Little CMS 2.17. Cargo.lock
SHA-256: `ffdf63a92951539bb369925d5987b2a7f6738f0b0fb2a925543495a918cd26b0`.
Small-input generated tables were kept separate from the committed full tables.

Each RGB transform uses the existing 320-pixel U16 ramp (gray, primary ramps
and mixed hues), output to sRGB, at relative colorimetric, perceptual and
saturation intents. Canonical qualification retains its existing gates:
256 U16 codes versus the canonical reference, 64 for intent differences,
and the independent-reader AND-gate. These sampled tolerances do not prove
bit-exact equivalence or cover every input.

| Normalized hash | Objects | Maximum moxcms/Little CMS difference, each intent | Canonical qualification |
|---|---:|---:|---|
| `079616e11678df00` | 2 | 245 | sRGB; passes existing gates |
| `6a669bf1e1bf97d6` | 1 | 247 | sRGB identification only; fails canonical gate |
| `9b2cc05c9ea02b2b` | 1 | 151 | Display P3/sRGB; passes existing gates |
| `8eb77ac64d443978` | 2 | 151 | Display P3/sRGB; passes existing gates |
| `832ffc20dff7d16a` | 1 | 151 | Display P3/sRGB; passes existing gates |
| `f46397a0d8f527b4` | 1 | 140 | Noncanonical monitor profile; no table admission |
| `bd57d5e632c34671` | 1 | 288 | Noncanonical monitor profile; no table admission |
| `f0bc773837572dbf` | 1 | 122 | Noncanonical monitor profile; no table admission |
| `6091121e62bfa8dc` | 2 | Not applicable | Lab input, outside the RGB/gray qualification |
| `6ebfe6e967b79c6d` | 2 | Not applicable | XYZ input, outside the RGB/gray qualification |

Six objects pass canonical qualification, one fails, three have no canonical
classification and four are outside this test's RGB input domain. The seven
classified objects already have identification-table entries; this run adds
no new recognition hashes.

The failing sRGB original has SHA-256
`c56e1685d888f5edb92fe07f2750f387f8fe8e91b32ff8fb0b56bfbbb9458353`.
Its measured moxcms canonical transform error is **298**, exceeding **256**.
The existing table incorrectly granted `Safe::AnyIntent`. Its regenerated
`Safe::IdOnly` value is applied without changing identification or any
threshold. The curve-fit error column remains 1: it measures a different
quantity and is not a complete-transform bound.

Validation passed: ICC unit tests, the generator's unit tests, scoped fmt and
Clippy. A direct public-API witness using this exact original confirms sRGB
recognition with `IdentificationUse::MetadataOnly` and rejects
`IccIdentification::is_common_srgb`, preserving the full CMS path.

Full historical tolerance requalification is still outstanding. Original-byte
coverage is complete; only the 14 manifest additions received the qualification
run described here. The private consolidated archive below now retains every
selected original. Updating the separate R2 archive/manifest remains maintenance
work; no R2 object or access setting was changed.

## Paired fingerprint verification

The MurmurHash3 migration rechecked all 598 selected originals against the
actual hash-enabled core/zenhash implementation and their full-file SHA-256
identities. Every `(FNV-1a64, MurmurHash3_x86_32)` pair matches the independent
upstream-derived mapping; no original is missing and no qualification or
intent permission changed in this migration. The Murmur seed is 3_242_157_231.

The extended receipt is `coverage-murmur3.json` beside `coverage-final.json`;
SHA-256 is
`09abe622e727cf6ee7bc92386f6b3232579b4d18b2e03ecfba7d2cbf541744a1`.
`murmur-verification.txt` records the 598/598 implementation check. This verifies
fingerprint coverage, not the still-separate historical tolerance requalification.


## Private consolidated witness archive

All 598 selected originals are copied to the private artifact archive at
`output/zenpixels/icc-originals-2026-11-09/`, totaling 1,657,355 profile bytes.
Each object is named `profiles/<full-file-sha256>.icc`. Its portable manifest
retains active/deferred state, both normalized fingerprints, full-file identity
and original source provenance. Paths resolve relative to the manifest, so
future revalidation does not depend on the audit host's scattered caches.

The manifest SHA-256 is
`6c9dfa82a4730d86cb73e260419ae5d0ac43584454cfc3a74c3f42d84fb143d0`.
Verification on the private archive passed for every object after transfer.
This retention change commits no original profile bytes to the public repository.

Use `just icc-retain-originals <receipt.json> <private-directory>` to construct
an archive, or omit the destination to verify its manifest. The tool checks
all source sizes/digests before copying, preserves links/provenance, and refuses
to replace conflicting retained objects or manifests. `just icc-retain-tool-check`
checks corruption, missing-source preflight and preservation of existing data.
This verifies retention only; fingerprint computation and tolerance qualification
remain their own checks.
