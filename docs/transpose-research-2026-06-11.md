# Big-transpose research — can `orient.rs` go faster? (2026-06-11)

Context: zenjpeg#150 follow-up. After landing `transpose_tiled::<BPP>` (imazen/zenpixels@0361f324,
RGB8 Rotate90 12MP 87.2 → 48.1 ms), the question is whether a better cache-local
algorithm exists. Three sourced web-research sweeps (blocking strategy SOTA, SIMD
microkernels for small elements, memory-path techniques) + local capability audit
of archmage/magetypes. Distilled here; estimates are labeled as estimates.

## Verdict

Yes — substantial headroom remains. The current tiled-gather path is the right
*macro* structure (explicit blocking, sequential destination writes — both
independently validated by the literature), but its **per-pixel scalar inner loop
is the bottleneck class**: the measured gap between "blocked + scalar inner loop"
and "blocked + in-register SIMD microkernel" is 3–8× across studies, and SIMD
transposes hold ~0.49 cycles/elem flat from cache-resident into multi-GB
DRAM-resident sizes (gudok.xyz/transpose — no crossover where scalar catches up).

Hardware ceiling on this box (Zen4 7950X, Chips and Cheese measurements): one core
reads ~50 GB/s; writes cap at 32 GB/s per CCD (16B/cycle IF link); NT stores reach
within 1% of that cap. 12MP RGB8 = 36 MB each way → bandwidth floor ~2 ms.
A tuned single-threaded kernel won't reach the floor, but landing in the 10–20 ms
range (vs 48 ms today) is consistent with what HPTT-class designs achieve
(92–102% of SAXPY bandwidth multithreaded; single-thread within ~1.2–2× of
single-core memcpy — the 2× is synthesis, not a single source).

## Measurement caveat discovered (affects current numbers)

`bench_orient` allocates the 36 MB destination **every iteration**. First-touch
soft faults cost ~0.9 µs/4KB page (measured, rahalkar.dev; ~3000 cycles/fault
Linux average), so a fresh 36 MB destination ≈ 9.2k pages ≈ **~8–12 ms of fault
time inside our 48.1 ms** (derived estimate, not measured on Zen4). zenjpeg's
real decode path reuses pooled buffers via `apply_orientation_into`, so it sees
kernel-only cost. **Step 0 of any further work: re-bench `_into` with a reused,
pre-touched destination** to separate kernel from allocator. Until then, treat
48.1 ms as an upper bound on the kernel.

## Ranked plan (gains are literature-derived estimates until measured here)

1. **SIMD microkernel for 3 bpp — expand→transpose→compress** (est. 2–3× on the
   kernel). The one production-proven RGB24 transpose shape (ermig1979/Simd
   library, SSE4.1/AVX2/AVX-512 variants): load 16 B per 4 RGB pixels (12 payload
   + 4 slop), `pshufb` 3→4 expand, transpose as 4-byte lanes, `pshufb` 4→3
   compress on store; scalar per-pixel tail peel so slop never crosses the buffer
   end. Nobody has published a direct 12-byte-group kernel — the expansion route
   is the settled shape. NEON gets RGB nearly free via `LD3/ST3` + `trn` cascade
   (libyuv pattern). Implementable in safe Rust today: value-mode intrinsics are
   safe inside `#[arcane]`/`#[rite]` on Rust 1.87+ (archmage MSRV is 1.89), with
   safe ref-based load/store wrappers via `import_intrinsics`
   (`safe_unaligned_simd`); zenwebp already ships `_mm_shuffle_epi8` kernels
   under `forbid(unsafe_code)` this way. Hand-tuned `_v3`/`_neon` tiers slot
   next to the `#[magetypes]` family via the `_<tier>` suffix + `incant!`.
   While here: 16×16-byte punpck cascade for 1 bpp and 8×8-epi16 for 2 bpp
   (settled art: Intel AP-528 1996 → pzemtsov → libyuv `TransposeWx8_SSSE3`).
2. **Scratch-tile staging** (fixes the critical-stride pathology; enables NT
   later). Gather the strided source tile into a small contiguous L1 scratch,
   then write destination rows from scratch. Chatterjee-Sen "half-copying" was
   the fastest non-exotic variant in the classic study and is stride-independent
   — our 2048² dip (1.59 vs 3.43 GiB/s at 1024²) is the textbook 4KB-aliasing
   pathology this kills (measured 2.5–6.5× penalties across three sources;
   Agner Fog: pad-or-tile recovers nearly all). Bonus: with staging, source
   reads can always walk ascending (apply the orientation's reflection in the
   scratch→dst phase), removing the negative-stride gather penalty visible in
   Rotate90-vs-Transpose (48.1 vs 39.6 ms).
3. **Non-temporal stores on the destination** (est. +1.2×, HPTT's measured
   average for exactly our β=0 out-of-place case). Requires staging (step 2)
   so writes are full 64-byte lines, ≤7 concurrent streams (Zen4 SOG WCB
   constraint; interleaved NT streams collapse >20× — ICPE'25). **Blocked on
   tooling**: `_mm_stream_*` is pointer-based and not wrapped by
   safe_unaligned_simd/archmage `import_intrinsics` — needs an upstream
   addition (user decision, different repo). Staging alone still pays without
   NT (full-line regular stores write-combine well).
4. **Software prefetch of the next tile** (+11–14% in TTC/gudok, Skylake; no
   Zen4 A/B published). `prefetchnta` semantics documented in Zen4 SOG (fills
   L2 marked quick-evict, skips L3 on eviction). Zen4's L2 prefetchers are
   sequential-only; the only stride prefetcher is per-IP at L1 — so SW prefetch
   plausibly retains value on the strided side. Low effort once staging exists;
   measure on this box. (Check whether `_mm_prefetch` is exposed safely.)
5. **Allocation-policy hardening (zenpixels crate)**: stride rule "64 B × odd
   multiple" for aligned buffer allocation dodges critical-stride aliasing
   structurally (gudok: 3.6–4.2× recovery at pathological sizes). Cheap,
   benefits every consumer, orthogonal to kernel work.
6. **Not now — threading**: single core is write-capped at 32 GB/s per CCD;
   2–4 threads across CCDs would saturate the socket, but codec callers own
   threading policy. Revisit only if a caller asks for it.

## What the research says NOT to do

- **Cache-oblivious recursion**: 2× slower than blocked+staging in the classic
  measurement; recursion-as-tile-order only pays at n ≥ 16384² — not image-sized.
- **Sequential-reads + scattered-writes direction**: measured worse (3.90 vs
  2.61 CPE naive-vs-reversed; store queue commits in order, RFOs double write
  traffic, prefetchers help reads not writes). Our write-sequential choice stands.
- **Bigger tiles**: 32-row tiles fit Zen4's 72-entry L1 dTLB at our strides;
  ≥128 rows would thrash TLB without huge pages.
- **Counting on THP**: Ubuntu defaults to `madvise`; plain malloc/Rust-default
  allocations get no huge pages. Buffer pooling (already supported via `_into`)
  is the reliable mitigation; MADV_HUGEPAGE arenas are caller-side policy.

## Expected end state (estimate, to be measured)

Steps 0–3 landed: 12MP RGB8 Rotate90 ~48 → ~10–20 ms single-threaded; the same
structure lifts 4 bpp (39 ms today; `f32x8::transpose_8x8` already exists in
magetypes for an AVX2-width microkernel upgrade) and 1/2 bpp. AVX-512 `vpermb`
(1 uop on Zen4, 2/clk on Zen5) is an optional `_v4x` tier on top — uniquely
cheap on AMD, marginal on Intel (3 uops, port-5-bound).

## Primary sources

- gudok.xyz/transpose — 1-byte progression: naive 3.90 → blocked 1.46 →
  +prefetch 1.35 → SWAR 0.74 → 256-bit SIMD 0.49 CPE; +40% from scratch+stream;
  DRAM-resident holds 0.49.
- HPTT (arXiv:1704.04374) — microkernel=SIMD-width, macro=4×, prefetch, NT
  +1.20×, 92–102% of SAXPY; TTC (arXiv:1603.02297) — loop-order spread 9.2×,
  explicit microkernel ≥+20% over autovectorized.
- Chatterjee & Sen, cache-efficient transposition — half-copy fastest, CO loses,
  TLB miss tables.
- ermig1979/Simd `SimdSse41Transform.cpp`/`SimdAvx2Transform.cpp` — the RGB24
  expand/transpose/compress kernels + rotate-90/270 via mirrored walks.
- libyuv rotate_gcc.cc / rotate_neon64.cc — production 8×8/16×8 byte kernels.
- AMD Zen4 SOG #57647 §2.13 (WCB: 8 streams, full-line rule, flush triggers);
  Chips and Cheese Zen4 part 2 (50 GB/s single-core read, 32 GB/s/CCD write,
  NT within 1%); ICPE'25 multi-striding (arXiv:2412.16001) — interleaved-NT
  collapse, +31–33% multi-stream reads.
- Agner Fog optimizing_cpp §9.10 — critical-stride tables (6×; padding fixes).
- Page faults: rahalkar.dev (~0.9 µs/page measured), Lemire (3.9 vs 20 GB/s
  first-touch 4K vs THP), ClickHouse prefault PR #6667.
- OpenCV matrix_transform.cpp — SIMD at sizeof 1/2/4/6 but NOT 3 (RGB24 scalar);
  libjpeg-turbo transupp.c — DCT-domain only, no spatial SIMD rotate: the
  3-byte SIMD transpose is a real gap in mainstream OSS.

Full agent reports with complete URL lists live in the session transcript
(2026-06-11); this file is the durable distillation.
