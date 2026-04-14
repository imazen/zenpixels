# icc-gen

Internal, unpublished tool that regenerates `zenpixels/src/icc/icc_table_rgb.inc`
and `icc_table_gray.inc` from a directory of ICC profiles.

Not part of the shipped crates — isolated here so its dependencies (`moxcms`,
optional `lcms2`) don't leak into `zenpixels` or `zenpixels-convert`.

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
