//! Microbench: does the `match ch_type { U8 => ...; U16 => ...; F32 => ...; F16 => ... }`
//! pattern used across convert_kernels.rs hurt autovec, compared to per-type
//! dedicated functions with `#[autoversion]` and `chunks_exact`?
//!
//! Kernel under test: `straight_to_premul` for U16 RGBA 4-channel. Picked
//! because:
//!   - Arithmetic-per-pixel: `channel * alpha + 32768) / 65535`, 4 independent lanes.
//!   - Not already delegated to garb (U8 RGBA and F32 RGBA are; U16 isn't).
//!   - Representative of the pattern used by the 13 swizzle/premul kernels.
//!
//! Four variants of the same math, same result:
//!   V0: current match-in-function kernel (scalar for-loop).
//!   V1: split-per-type function, `#[autoversion]`, plain indexed for-loop.
//!   V2: split-per-type function, `#[autoversion]`, `chunks_exact(16)` (4 pixels = 256 bits).
//!   V3: split-per-type function, `#[autoversion]`, fixed-size `[u16; 4]` per pixel.
//!
//! All four are correctness-verified against each other at a small size before
//! benching.
//!
//! Run: `cargo bench --bench bench_premul_u16`

use archmage::prelude::*;
use zenbench::prelude::*;

// ── V0: current production kernel (copied verbatim from convert_kernels.rs) ──

fn v0_match_in_fn(src: &[u8], dst: &mut [u8], width: usize, channels: usize) {
    let src16: &[u16] = bytemuck::cast_slice(&src[..width * channels * 2]);
    let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..width * channels * 2]);
    let alpha_idx = channels - 1;
    for i in 0..width {
        let base = i * channels;
        let a = src16[base + alpha_idx] as u32;
        for c in 0..alpha_idx {
            dst16[base + c] = ((src16[base + c] as u32 * a + 32768) / 65535) as u16;
        }
        dst16[base + alpha_idx] = src16[base + alpha_idx];
    }
}

// ── V1: split per-type, autoversion, plain loop ─────────────────────────────

#[autoversion]
fn v1_u16_rgba_plain(src: &[u16], dst: &mut [u16], width: usize) {
    for i in 0..width {
        let base = i * 4;
        let a = src[base + 3] as u32;
        dst[base] = ((src[base] as u32 * a + 32768) / 65535) as u16;
        dst[base + 1] = ((src[base + 1] as u32 * a + 32768) / 65535) as u16;
        dst[base + 2] = ((src[base + 2] as u32 * a + 32768) / 65535) as u16;
        dst[base + 3] = src[base + 3];
    }
}

fn v1_dispatch(src: &[u8], dst: &mut [u8], width: usize) {
    let n = width * 4 * 2;
    let src16: &[u16] = bytemuck::cast_slice(&src[..n]);
    let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..n]);
    v1_u16_rgba_plain(src16, dst16, width);
}

// ── V2: split per-type, autoversion, chunks_exact(16) = 4 pixels/chunk ──────

#[autoversion]
fn v2_u16_rgba_chunks(src: &[u16], dst: &mut [u16], width: usize) {
    let n = width * 4;
    let src = &src[..n];
    let dst = &mut dst[..n];

    // Process 4 pixels (16 u16 = 256 bits) at a time.
    let src_chunks = src.chunks_exact(16);
    let src_rem = src_chunks.remainder();
    let dst_chunks = dst.chunks_exact_mut(16);
    let rem_start = n - src_rem.len();

    for (s, d) in src_chunks.zip(dst_chunks) {
        // Explicit per-pixel unroll so LLVM sees independent lanes.
        for px in 0..4 {
            let base = px * 4;
            let a = s[base + 3] as u32;
            d[base] = ((s[base] as u32 * a + 32768) / 65535) as u16;
            d[base + 1] = ((s[base + 1] as u32 * a + 32768) / 65535) as u16;
            d[base + 2] = ((s[base + 2] as u32 * a + 32768) / 65535) as u16;
            d[base + 3] = s[base + 3];
        }
    }

    // Tail: process remaining pixels one at a time.
    let tail_src = &src[rem_start..];
    let tail_dst = &mut dst[rem_start..];
    for (s, d) in tail_src.chunks_exact(4).zip(tail_dst.chunks_exact_mut(4)) {
        let a = s[3] as u32;
        d[0] = ((s[0] as u32 * a + 32768) / 65535) as u16;
        d[1] = ((s[1] as u32 * a + 32768) / 65535) as u16;
        d[2] = ((s[2] as u32 * a + 32768) / 65535) as u16;
        d[3] = s[3];
    }
}

fn v2_dispatch(src: &[u8], dst: &mut [u8], width: usize) {
    let n = width * 4 * 2;
    let src16: &[u16] = bytemuck::cast_slice(&src[..n]);
    let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..n]);
    v2_u16_rgba_chunks(src16, dst16, width);
}

// ── V3: split per-type, autoversion, fixed-size [u16; 4] per pixel ──────────
//
// Per the archmage patterns: one `try_into::<&[T; N]>().unwrap()` at the
// function boundary gives LLVM a fixed-size array, enabling it to prove all
// indexing safe without bounds checks. Here applied per-pixel (N=4) which
// matches the RGBA unit of work.

#[autoversion]
fn v3_u16_rgba_fixedarr(src: &[u16], dst: &mut [u16], width: usize) {
    for i in 0..width {
        let base = i * 4;
        let s: &[u16; 4] = (&src[base..base + 4]).try_into().unwrap();
        let d: &mut [u16; 4] = (&mut dst[base..base + 4]).try_into().unwrap();
        let a = s[3] as u32;
        d[0] = ((s[0] as u32 * a + 32768) / 65535) as u16;
        d[1] = ((s[1] as u32 * a + 32768) / 65535) as u16;
        d[2] = ((s[2] as u32 * a + 32768) / 65535) as u16;
        d[3] = s[3];
    }
}

fn v3_dispatch(src: &[u8], dst: &mut [u8], width: usize) {
    let n = width * 4 * 2;
    let src16: &[u16] = bytemuck::cast_slice(&src[..n]);
    let dst16: &mut [u16] = bytemuck::cast_slice_mut(&mut dst[..n]);
    v3_u16_rgba_fixedarr(src16, dst16, width);
}

// ── Correctness check before benching ───────────────────────────────────────

fn verify_all_equal(width: usize) {
    let bytes = width * 4 * 2;
    let src: Vec<u8> = (0..bytes).map(|i| (i % 256) as u8).collect();
    let mut out_v0 = vec![0u8; bytes];
    let mut out_v1 = vec![0u8; bytes];
    let mut out_v2 = vec![0u8; bytes];
    let mut out_v3 = vec![0u8; bytes];

    v0_match_in_fn(&src, &mut out_v0, width, 4);
    v1_dispatch(&src, &mut out_v1, width);
    v2_dispatch(&src, &mut out_v2, width);
    v3_dispatch(&src, &mut out_v3, width);

    assert_eq!(out_v0, out_v1, "V1 disagrees with V0 at width {width}");
    assert_eq!(out_v0, out_v2, "V2 disagrees with V0 at width {width}");
    assert_eq!(out_v0, out_v3, "V3 disagrees with V0 at width {width}");
}

// ── Bench driver ────────────────────────────────────────────────────────────

fn main() {
    // Correctness first. Use a width that exercises the V2 tail (not a
    // multiple of 4) and one that doesn't.
    verify_all_equal(64); // 64 * 4 = 256 u16, exact multiple of 16
    verify_all_equal(67); // 67 * 4 = 268, not a multiple of 16 → tail
    verify_all_equal(1023); // stress tail again

    let sizes: &[(&str, usize)] = &[
        ("  256 px  (L1, 2 KB/2 KB)", 256),
        (" 4096 px  (L1/L2, 32 KB/32 KB)", 4096),
        ("1920x1080 frame (L3/memory)", 1920 * 1080),
    ];

    zenbench::run(|suite| {
        for (label, width) in sizes {
            let bytes = *width * 4 * 2;
            let src: Vec<u8> = (0..bytes).map(|i| (i % 256) as u8).collect();

            suite.group(format!("u16 RGBA straight_to_premul  {label}"), |g| {
                g.throughput(Throughput::Bytes(bytes as u64));

                let w = *width;
                {
                    let s = src.clone();
                    let mut dst = vec![0u8; bytes];
                    g.bench("V0 match-in-fn (current)", move |bench| {
                        bench.iter(|| {
                            v0_match_in_fn(&s, &mut dst, w, 4);
                            black_box(());
                        })
                    });
                }
                {
                    let s = src.clone();
                    let mut dst = vec![0u8; bytes];
                    g.bench("V1 split + autoversion", move |bench| {
                        bench.iter(|| {
                            v1_dispatch(&s, &mut dst, w);
                            black_box(());
                        })
                    });
                }
                {
                    let s = src.clone();
                    let mut dst = vec![0u8; bytes];
                    g.bench("V2 split + autoversion + chunks_exact(16)", move |bench| {
                        bench.iter(|| {
                            v2_dispatch(&s, &mut dst, w);
                            black_box(());
                        })
                    });
                }
                {
                    let s = src.clone();
                    let mut dst = vec![0u8; bytes];
                    g.bench("V3 split + autoversion + [u16; 4] fixed", move |bench| {
                        bench.iter(|| {
                            v3_dispatch(&s, &mut dst, w);
                            black_box(());
                        })
                    });
                }
            });
        }
    });
}
