# U16 widening survey: lo==hi replication vs lo==0 shift, in reality

**Date:** 2026-06-10 · **Host:** lilith (WSL2, 7950X) · **Branch:** PR #30 (`push-qpwqxvsrmqqn` @ e2df5a47, survey commit f9d5ba2)
**Tool:** `benchmarks/u16_low_byte_survey.py` (cv2/tifffile/imageio decode; per-file classification of low-byte structure, alpha channel excluded from the verdict)

## Question

`scan::bit_replication_lossless_u16` detects "this 16-bit buffer is secretly
8-bit" via `lo == hi` (i.e. every sample is `v * 257`). Is that what
real-world widened 16-bit data actually looks like — or do producers emit
`v << 8` (low byte zero) instead?

## Answer

**Correct producers replicate; `lo == hi` is the right and only bit-exact
test.** `65535 = 255 × 257`, so correct full-scale widening
(`round(v·65535/255)`) *is* byte replication — there is no rounding
freedom. Every surveyed library that does a deliberate 8→16 conversion
produces exact `v * 257`. The `v << 8` and rawer patterns do exist in
reality, but none of them are bit-exact reducible (they sit at scale
256/257 of replicated, or 1/257 of it), so the predicate must keep
rejecting them.

## Producer matrix (each entry verified today, not recalled)

| producer | method | result |
|---|---|---|
| libpng `png_set_expand_16` | source read, `pngrtran.c` `png_do_expand_16` | **REPLICATED** — `dp[-2] = dp[-1] = *--sp` (in-place byte replication; comment derives `input * 257`) |
| ImageMagick 7 `convert -depth 16 png48:` | measured (all 256 values + noise) | **REPLICATED** — exact `v*257`, max dev 0 |
| Rust `image` 0.25.8 `DynamicImage::into_rgb16()` | measured (probe crate) | **REPLICATED** — exact `v*257` for all sampled values |
| ffmpeg CLI default `-i src8.png -pix_fmt rgb48be` | measured | **neither** — ≈`v << 8` ± 3 (deterministic per value; routed through a YUV intermediate). White → 65283, not 65535. Classifies TRUE16; undetectable by any exact test |
| naive `astype(uint16)` (numpy et al.) | measured control + found in the wild | **UNSCALED_8IN16** — raw 0..255 values, high byte zero |
| naive `v << 8` code / BT-spec video upshift (8→10: `d << 2`) | measured control / spec convention | **SHIFTED** — low k bits zero |

In-the-wild confirmation of the non-replicated patterns:
`/mnt/v/input/tiff-bench/photo-rgb16-deflate-hpred.tif` (21 MP photographic
TIFF) is **UNSCALED_8IN16** — `or_all = 0x00ff`, every sample < 256. A
hi==lo detector correctly refuses it (only 11.5% of samples pass), and
extracting low bytes would be a 257× brightness reinterpretation — exactly
the kind of value rewrite the load-bearing contract forbids.

## Local 16-bit file sweep

Swept: codec-corpus cache (`~/.cache/codec-corpus/v1`, full), tiff-bench,
geotiff-bench, gainmap-samples, datasets (16-bit PNG pre-filter via IHDR
sniff; all TIFFs attempted). 17 16-bit images decoded; f32 TIFFs and
exotic geo compressions skipped (43 skip/error lines, expected).

| class | count | what they are |
|---|---|---|
| TRUE16 | 16 | PngSuite 16bpc gradients, image-rs TIFF testsuite refs — synthetic true-16-bit by construction |
| UNSCALED_8IN16 | 1 | the tiff-bench photo above |
| REPLICATED / SHIFTED_8 | 0 | (none present locally outside generated controls) |

Raw rows (local sweep + tool-probe files): `u16_low_byte_survey_2026-06-10.tsv`.
Regenerate with `python3 benchmarks/u16_low_byte_survey.py <dirs> > out.tsv`
(classifier sanity-checked against ×257, <<8 and astype controls — all
three classify correctly; control rows included in the TSV).

Coverage note: no Photoshop-authored (15+1-bit internal) or
scanner-native 16-bit files were available locally; those remain
unmeasured. Photographic/scanner content is expected TRUE16 (sensor
noise occupies the low bits).

## Design conclusions (what this PR does about it)

1. `bit_replication_lossless_u16` stays `lo == hi` only — that is the
   exact inverse of every correct widening path, and the only pattern a
   bit-exact reduction may act on. Doc comment on the predicate now
   records this survey; `bit_repl_u16_rejects_real_world_inexact_widenings`
   pins rejection of the shifted / unscaled / ffmpeg-noise patterns.
2. `v << 8`, k-bit upshifts, and unscaled-8-in-16 are *reportable
   observations* but acting on them changes stored values (≈0.4% scale
   for shifts, 257× for unscaled). If a consumer ever wants
   "normalize sloppy widenings", that is an explicit opt-in conversion
   API (same stance as gamut narrowing), not a load-bearing reduction.
   No report field is added until such a consumer exists.
3. Cost note: if that consumer shows up, the structural classification
   (all-lo==0 / or-reduction trailing zeros) fuses into the existing
   single pass as one extra compare+OR per vector — no second pass
   needed. Not implemented now (YAGNI), so no perf claim is made.
