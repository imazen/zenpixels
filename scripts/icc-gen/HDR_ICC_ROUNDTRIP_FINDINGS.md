# HDR ICC round-trip validation (moxcms synth → lcms2 read-back)

Tool: `src/bin/hdr_roundtrip.rs` — run with `cargo run -p icc-gen --bin hdr_roundtrip`
(needs `liblcms2-dev` + the `cms-moxcms` feature, both already in this member's graph).
Date: 2026-06-10. moxcms 0.8.1, lcms2 6.1.1 / lcms2-sys 4.0.6, zenpixels-convert 0.2.12.

## Question

Do `synthesize_icc_for_cicp` HDR profiles (PQ / HLG / P3-PQ), whose EOTF is baked
as a **1024-entry `curv` LUT** (`build_trc_table(1024, pq_to_linear)`, output
u16-quantized `(y·65535+0.5).floor()`, sampled uniform in *code* space), reconstruct
the transfer faithfully when a real downstream CMS reads them back? Special concern:
the PQ **toe**, where the LUT may crush sub-1-nit detail.

## Method

For each CICP: synthesize the ICC (confirmed `SynthesizedIcc::Profile`, 6524 bytes,
rTRC = `curv` LUT, **1024 entries** — verified by tag walk). Build an lcms2 float
transform `ICC → XYZ PCS` (`RGB_FLT → XYZ_FLT`, `NO_OPTIMIZE | HIGHRES_PRECALC`,
relative-colorimetric). Feed a **dense log-spaced grey ramp** (R=G=B), 400 points
log-spaced 1e-4..0.06 in the toe + 600 uniform 0.06..1.0. For a grey input the PCS
**Y** is the decoded linear (all three channels share the same EOTF and x, so the
RGB→XYZ matrix sums the luminance weights back to 1.0). Compare against the analytic
EOTF — a **1:1 port of moxcms `pq_to_linear` / `hlg_to_linear`**, i.e. the exact
function baked into the LUT, so any divergence is LUT-reconstruction error, not a
formula mismatch. Normalize lcms2's Y by Y(white) to strip any constant gamut scale
(it came out 1.000000 anyway). Two extra data points: (a) moxcms reads its own bytes
back; (b) an **analytic curv-LUT model** (piecewise-linear interp over the same 1024
u16 samples) at 1024 / 2048 / 4096 entries to locate and size the error source.

## Results (lcms2 — authoritative)

| Profile | max_abs (where) | max_rel, ref_lin ≥ 1e-3 (where) | toe spot rel-err |
|---|---|---|---|
| **PQ** (BT.2020/2084) | 6.10e-5 @ x=0.992 (shoulder) | **0.93%** @ x=0.31 (mid) | x=0.10: **7.8%**, x=0.25: 1.7%, x≥0.5: ≤0.01% |
| **HLG** (BT.2020/B67) | 4.32e-5 @ x=0.959 (shoulder) | **0.93%** @ x=0.11 (mid) | x=0.05: 3.1%, x=0.10: 0.07%, x≥0.10: ≤0.07% |
| **P3-PQ** (P3/2084) | 6.10e-5 @ x=0.992 (shoulder) | **0.93%** @ x=0.31 (mid) | x=0.10: **7.8%**, x=0.25: 1.7%, x≥0.5: ≤0.01% |

`max_rel` over the *full* ramp hits ~100% for all three, but only at `ref_lin < 2e-5`
(PQ 7.6e-6, HLG 1e-9) — linear values **below one 16-bit code** (1.5e-5). That is the
output-quantization floor at near-black, not a reconstruction failure: there is no
real signal there even at 12-bit HDR depth.

**Source of the error (analytic LUT model):** the 1024-entry piecewise-linear model
predicts 0.53% at the same mid-tone x where lcms2 measures 0.93% (lcms2 adds its own
16-bit curve-eval grid on top → 6.1e-5 abs). Going to 2048/4096 entries does **not**
reduce the mid-tone relative error (0.50–0.63%) — it is bound by the **u16 output
quantization**, not by input sampling. So 1024 entries is already near the floor of
this `curv`+u16 design; a denser LUT is not the lever.

**moxcms self-readback** shows much larger error (12× rel, 0.29 abs) — expected: its
f32 transform is documented as ~14-bit and it applies a BT.2020→sRGB gamut matrix the
single-channel readout doesn't fully invert. It is the weaker second opinion by
construction; lcms2's float XYZ path is the clean measurement.

## VERDICT

**Acceptance bar:** white-normalized max relative error in linear < 1% at
perceptually-relevant magnitudes (`ref_lin ≥ 1e-3`, i.e. ≥ ~10 nits on a 10000-nit
PQ scale), with deep-toe (< one 16-bit code) quantization noise out of scope.

- **HLG — PASS, comfortably.** ≤ 0.93% above 1e-3, and already < 0.1% by x=0.10. The
  HLG toe is gentle (no 10000:1 range) and the LUT handles it cleanly. Safe to bundle.
- **PQ / P3-PQ — CONDITIONAL PASS.** Meets the bar for `ref_lin ≥ 1e-3` (0.93% mid).
  Above ~10 nits the reconstruction is faithful (≤ 0.01% from mid-grey up). **But** the
  deep PQ toe is genuinely soft: **~7.8% at x=0.10 (≈1 nit), ~1.7% at x=0.25**, because
  the 1024×u16 `curv` LUT cannot resolve PQ's sub-1-nit decade. This is *inherent to
  the LUT design*, not an lcms2 artifact (the analytic model agrees), and not fixable by
  more LUT entries (u16-output-bound).

### Recommendation

**Prefer CICP-native HDR signaling wherever the container supports it** — AVIF, JXL,
HEIC, and PNG (cICP chunk) all carry PQ/HLG code points directly, with zero LUT
round-trip. Route HDR color through the CICP carrier for those formats; do **not**
embed a synthesized HDR `curv`-LUT profile when a native code point is available.

**For the rare ICC-only HDR case** (PNG iCCP without cICP, WebP, JPEG — formats that
can carry an ICC but no CICP), the moxcms profile is *acceptable for HLG* and
*acceptable for PQ down to ~10 nits*, but **will visibly soften PQ sub-1-nit shadow
detail** (~1–8% in the bottom code-decade). Given the project's ZERO-TOLERANCE pixel
rule, the conservative call is:

- **Bundle HLG HDR profiles** if desired — they round-trip cleanly.
- **Exclude PQ / P3-PQ from a "lossless-grade" ICC bundle**, OR bundle them only with
  an explicit caveat that they are a *fallback for ICC-only containers* and that the
  PQ deep toe (< ~10 nits) reconstructs to single-digit-percent accuracy. The right
  primary path for PQ is always CICP-native.

The toe softness is a property of representing PQ as a finite u16 `curv` LUT, which is
the only option an ICC v4 `curv` tag offers for a non-parametric transfer — not a bug
in moxcms or lcms2.
