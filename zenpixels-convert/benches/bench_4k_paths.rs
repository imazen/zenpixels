//! Full-frame 4K UHD (`3840 × 2160`, 8,294,400 pixels) regression benchmark suite
//! covering every `zenpixels-convert` conversion, gamut, matte, Oklab, perceptual,
//! `FloatFacts`, and orientation code path targeted by codegen compression, Cargo
//! feature gating, and const-generic capability masking.
//!
//! Each benchmark streams a full `3840 × 2160` frame row-by-row via
//! `RowConverter::convert_raw_rows` (or `PixelSliceConvertExt::oriented_into` for
//! 4K transpose/rotation), using a single shared page-aligned 4K buffer pair
//! (`src` at page offset `0`, `dst` at page offset `2048` to eliminate 4 KiB
//! cache-set aliasing while keeping peak RSS under ~300 MiB).
//!
//! # Running
//!
//! - Full suite (default active tier `v3` on x86-64, `neon` on aarch64):
//!   `cargo bench -p zenpixels-convert --bench bench_4k_paths`
//! - Filter a single family:
//!   `cargo bench -p zenpixels-convert --bench bench_4k_paths -- --group='^oklab/'`
//! - Select SIMD tier (`v3`, `v4x` [needs `--features avx512`], `baseline`, or `all`):
//!   `BENCH_4K_TIER=baseline cargo bench -p zenpixels-convert --bench bench_4k_paths -- --group='^gray_wide/'`
//! - Override rounds/time budget:
//!   `BENCH_4K_ROUNDS=8 BENCH_4K_MAX_MS=1000 cargo bench -p zenpixels-convert --bench bench_4k_paths`

use std::sync::{Arc, Mutex};
use std::time::Duration;

use zenbench::prelude::*;
use zenpixels::hdr::{DiffuseWhite, HlgRendering};
use zenpixels::sample::FloatFacts;
use zenpixels::{
    AlphaMode, ChannelLayout as L, ChannelType as T, Color, ColorPrimaries as P, LumaCoefficients,
    Orientation, PixelDescriptor as D, PixelSlice, PixelSliceMut, TransferFunction as Tf,
};
use zenpixels_convert::{
    AlphaPolicy, ConversionSpec, ConvertOptions, ConvertPlan, DepthPolicy, GamutPolicy,
    PixelSliceConvertExt,
};

const WIDTH: u32 = 3840;
const HEIGHT: u32 = 2160;
const PIXELS: usize = (WIDTH as usize) * (HEIGHT as usize);
const MAX_FRAME_BYTES: usize = PIXELS * 16; // 3840 * 2160 * 16 (RGBAF32) = 132,710,400 B

const UNIT_CLEAN: FloatFacts = FloatFacts::NO_NAN
    .union(FloatFacts::NON_NEGATIVE)
    .union(FloatFacts::AT_MOST_ONE);
const FINITE: FloatFacts = FloatFacts::NO_NAN.union(FloatFacts::NO_INFINITE);
const FINITE_NONNEG: FloatFacts = FINITE.union(FloatFacts::NON_NEGATIVE);
const FINITE_NONNEG_NOSUB: FloatFacts = FINITE_NONNEG.union(FloatFacts::NO_SUBNORMALS);
const FINITE_POS_NOSUB: FloatFacts = FINITE_NONNEG_NOSUB.union(FloatFacts::POSITIVE);

struct FrameScratch {
    src_buf: Vec<u8>,
    src_off: usize,
    dst_buf: Vec<u8>,
    dst_off: usize,
    last_fill_key: Option<(T, L, FillKind, usize)>,
}

impl FrameScratch {
    fn new() -> Self {
        let (src_off, src_buf) = placed(MAX_FRAME_BYTES, 0);
        let (dst_off, dst_buf) = placed(MAX_FRAME_BYTES, 2048);
        Self {
            src_buf,
            src_off,
            dst_buf,
            dst_off,
            last_fill_key: None,
        }
    }

    fn slices_mut(&mut self, src_bytes: usize, dst_bytes: usize) -> (&mut [u8], &mut [u8]) {
        let s = &mut self.src_buf[self.src_off..self.src_off + src_bytes];
        let d = &mut self.dst_buf[self.dst_off..self.dst_off + dst_bytes];
        (s, d)
    }
}

fn placed(len: usize, page_offset: usize) -> (usize, Vec<u8>) {
    let bytes = vec![0u8; len + 2 * 4096];
    let start = (4096 - bytes.as_ptr() as usize % 4096) % 4096 + page_offset;
    (start, bytes)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FillKind {
    UniformUnit,
    PositiveNoSub,
    NearBlackPq,
    NearBlackLinear,
    ExtendedSigned,
    Oklab,
    Oklch,
    Lch,
    Jzazbz,
    HslHsv,
}

fn fill_frame(scratch: &mut FrameScratch, desc: D, kind: FillKind) {
    let bytes = PIXELS * desc.bytes_per_pixel();
    let key = (desc.channel_type(), desc.layout(), kind, bytes);
    if scratch.last_fill_key == Some(key) {
        return;
    }
    let (src, _) = scratch.slices_mut(bytes, 16);
    let mut x = 0x2545_F491u32;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        x
    };
    let ch = desc.layout().channels();
    match desc.channel_type() {
        T::U8 => {
            for b in src.iter_mut() {
                *b = (next() >> 24) as u8;
            }
        }
        T::U16 => {
            for px in src.as_chunks_mut::<2>().0 {
                let v = (next() >> 16) as u16;
                *px = v.to_ne_bytes();
            }
        }
        T::F16 => {
            // Fill a 4096-entry lookup ring and tile across 4K to keep setup < 10 ms.
            let mut lut = [[0u8; 2]; 4096];
            for slot in &mut lut {
                let f = (next() >> 8) as f32 / 16_777_215.0;
                *slot = half::f16::from_f32(f).to_ne_bytes();
            }
            for (i, px) in src.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                *px = lut[i & 4095];
            }
        }
        T::F32 => {
            // Tile a 4096-pixel deterministic pattern so setup stays fast while
            // exercising diverse float exponents/branches across every 4K row.
            let mut tile = vec![0.0f32; 4096 * ch];
            for p in 0..4096 {
                let u = |n: u32| (n >> 8) as f32 / 16_777_215.0;
                let base = p * ch;
                match kind {
                    FillKind::UniformUnit => {
                        for c in 0..ch {
                            tile[base + c] = u(next());
                        }
                    }
                    FillKind::PositiveNoSub => {
                        for c in 0..ch {
                            tile[base + c] = 0.005 + 0.99 * u(next());
                        }
                    }
                    FillKind::NearBlackPq => {
                        for c in 0..ch {
                            tile[base + c] = 0.02 * u(next());
                        }
                    }
                    FillKind::NearBlackLinear => {
                        for c in 0..ch {
                            tile[base + c] = 1e-7 * u(next());
                        }
                    }
                    FillKind::ExtendedSigned => {
                        for c in 0..ch {
                            tile[base + c] = if c == 3 {
                                u(next())
                            } else {
                                -0.25 + 1.5 * u(next())
                            };
                        }
                    }
                    FillKind::Oklab => {
                        tile[base] = 0.05 + 0.9 * u(next());
                        tile[base + 1] = -0.3 + 0.6 * u(next());
                        tile[base + 2] = -0.3 + 0.6 * u(next());
                        if ch == 4 {
                            tile[base + 3] = u(next());
                        }
                    }
                    FillKind::Oklch => {
                        tile[base] = 0.05 + 0.9 * u(next());
                        tile[base + 1] = 0.3 * u(next());
                        tile[base + 2] = 360.0 * u(next());
                        if ch == 4 {
                            tile[base + 3] = u(next());
                        }
                    }
                    FillKind::Lch => {
                        tile[base] = 100.0 * u(next());
                        tile[base + 1] = 100.0 * u(next());
                        tile[base + 2] = 360.0 * u(next());
                        if ch == 4 {
                            tile[base + 3] = u(next());
                        }
                    }
                    FillKind::Jzazbz => {
                        tile[base] = 0.01 + 0.5 * u(next());
                        tile[base + 1] = -0.2 + 0.4 * u(next());
                        tile[base + 2] = -0.2 + 0.4 * u(next());
                        if ch == 4 {
                            tile[base + 3] = u(next());
                        }
                    }
                    FillKind::HslHsv => {
                        tile[base] = 360.0 * u(next());
                        tile[base + 1] = u(next());
                        tile[base + 2] = u(next());
                        if ch == 4 {
                            tile[base + 3] = u(next());
                        }
                    }
                }
            }
            let tile_bytes: &[u8] = bytemuck::cast_slice(&tile);
            let mut off = 0;
            while off < src.len() {
                let n = (src.len() - off).min(tile_bytes.len());
                src[off..off + n].copy_from_slice(&tile_bytes[..n]);
                off += n;
            }
        }
        other => panic!("unsupported bench channel type {other:?}"),
    }
    scratch.last_fill_key = Some(key);
}

fn active_tiers() -> Vec<&'static str> {
    let raw = std::env::var("BENCH_4K_TIER").unwrap_or_default();
    let mode = raw.trim().to_ascii_lowercase();
    #[cfg(target_arch = "x86_64")]
    {
        match mode.as_str() {
            "" | "v3" => vec!["v3"],
            "v4x" => {
                if !cfg!(feature = "avx512") {
                    panic!("BENCH_4K_TIER=v4x requires --features avx512");
                }
                vec!["v4x"]
            }
            "baseline" => vec!["baseline"],
            "all" => {
                if cfg!(feature = "avx512") {
                    vec!["v4x", "v3", "baseline"]
                } else {
                    vec!["v3", "baseline"]
                }
            }
            other => panic!("unknown BENCH_4K_TIER={other:?} (use v3, v4x, baseline, or all)"),
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        match mode.as_str() {
            "" | "neon" => vec!["neon"],
            "baseline" => vec!["baseline"],
            "all" => vec!["neon", "baseline"],
            other => panic!("unknown BENCH_4K_TIER={other:?} (use neon, baseline, or all)"),
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = mode;
        vec!["baseline"]
    }
}

fn set_tier(tier: &str) {
    use archmage::SimdToken;
    #[cfg(target_arch = "x86_64")]
    {
        let _ = archmage::X64V3Token::dangerously_disable_token_process_wide(tier == "baseline");
        let _ = archmage::X64V4Token::dangerously_disable_token_process_wide(tier != "v4x");
        let _ = archmage::X64V4xToken::dangerously_disable_token_process_wide(tier != "v4x");
        assert_eq!(
            archmage::X64V3Token::summon().is_none(),
            tier == "baseline",
            "tier {tier}"
        );
        if tier != "v4x" {
            assert!(archmage::X64V4Token::summon().is_none(), "tier {tier}");
            assert!(archmage::X64V4xToken::summon().is_none(), "tier {tier}");
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let _ = archmage::NeonToken::dangerously_disable_token_process_wide(tier == "baseline");
        assert_eq!(
            archmage::NeonToken::summon().is_none(),
            tier == "baseline",
            "tier {tier}"
        );
    }
    let _ = tier;
}

fn desc(c: T, l: L, a: Option<AlphaMode>, tf: Tf, p: P) -> D {
    D::new_full(c, l, a, tf, p)
}

fn base_opts() -> ConvertOptions {
    ConvertOptions::retain_channels()
        .with_alpha_policy(AlphaPolicy::DiscardUnchecked)
        .with_depth_policy(DepthPolicy::Round)
        .with_luma(Some(LumaCoefficients::Bt709))
        .with_sdr_white(DiffuseWhite::BT2408)
        .with_hlg_rendering(HlgRendering::REFERENCE_1000_NIT)
}

struct Case {
    group: &'static str,
    name: &'static str,
    from: D,
    to: D,
    options: ConvertOptions,
    facts: FloatFacts,
    fill: FillKind,
}

impl Case {
    fn new(group: &'static str, name: &'static str, from: D, to: D) -> Self {
        Self {
            group,
            name,
            from,
            to,
            options: base_opts(),
            facts: FloatFacts::NONE,
            fill: FillKind::UniformUnit,
        }
    }
    fn opts(mut self, options: ConvertOptions) -> Self {
        self.options = options;
        self
    }
    fn facts(mut self, facts: FloatFacts) -> Self {
        self.facts = facts;
        self
    }
    fn fill(mut self, fill: FillKind) -> Self {
        self.fill = fill;
        self
    }
}

fn all_cases() -> Vec<Case> {
    let s = Some(AlphaMode::Straight);
    let pm = Some(AlphaMode::Premultiplied);
    let n = None;
    let bt709 = P::Bt709;
    let p3 = P::DisplayP3;
    let bt2020 = P::Bt2020;
    let adobe = P::AdobeRgb;

    let matte_opts = base_opts().with_alpha_policy(AlphaPolicy::CompositeOnto {
        background: Color::srgb8(200, 100, 50),
        reference_white: None,
    });
    let trunc_opts = base_opts().with_depth_policy(DepthPolicy::Truncate);
    let ext_opts = base_opts().with_gamut_policy(GamutPolicy::PreserveExtended);
    let soft_opts = base_opts().with_gamut_policy(GamutPolicy::SoftCompress);
    let bt601_opts = base_opts().with_luma(Some(LumaCoefficients::Bt601));

    vec![
        // ── 1. layout_u8 ─────────────────────────────────────────────────────
        Case::new(
            "layout_u8",
            "rgba8_to_rgb8",
            desc(T::U8, L::Rgba, s, Tf::Srgb, bt709),
            desc(T::U8, L::Rgb, n, Tf::Srgb, bt709),
        ),
        Case::new(
            "layout_u8",
            "rgb8_to_rgba8",
            desc(T::U8, L::Rgb, n, Tf::Srgb, bt709),
            desc(T::U8, L::Rgba, s, Tf::Srgb, bt709),
        ),
        Case::new(
            "layout_u8",
            "rgb8_to_bgra8",
            desc(T::U8, L::Rgb, n, Tf::Srgb, bt709),
            desc(T::U8, L::Bgra, s, Tf::Srgb, bt709),
        ),
        Case::new(
            "layout_u8",
            "rgba8_to_bgra8",
            desc(T::U8, L::Rgba, s, Tf::Srgb, bt709),
            desc(T::U8, L::Bgra, s, Tf::Srgb, bt709),
        ),
        Case::new(
            "layout_u8",
            "rgb8_to_bgr8",
            desc(T::U8, L::Rgb, n, Tf::Srgb, bt709),
            desc(T::U8, L::Bgr, n, Tf::Srgb, bt709),
        ),
        // ── 2. layout_wide ───────────────────────────────────────────────────
        Case::new(
            "layout_wide",
            "rgba16_to_rgb16",
            desc(T::U16, L::Rgba, s, Tf::Srgb, bt709),
            desc(T::U16, L::Rgb, n, Tf::Srgb, bt709),
        ),
        Case::new(
            "layout_wide",
            "rgb16_to_rgba16",
            desc(T::U16, L::Rgb, n, Tf::Srgb, bt709),
            desc(T::U16, L::Rgba, s, Tf::Srgb, bt709),
        ),
        Case::new(
            "layout_wide",
            "rgbaf16_to_rgbf16",
            desc(T::F16, L::Rgba, s, Tf::Linear, bt709),
            desc(T::F16, L::Rgb, n, Tf::Linear, bt709),
        ),
        Case::new(
            "layout_wide",
            "rgbaf32_to_rgbf32",
            desc(T::F32, L::Rgba, s, Tf::Linear, bt709),
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
        ),
        Case::new(
            "layout_wide",
            "rgbf32_to_rgbaf32",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
            desc(T::F32, L::Rgba, s, Tf::Linear, bt709),
        ),
        // ── 3. gray_u8 ───────────────────────────────────────────────────────
        Case::new(
            "gray_u8",
            "rgb8_to_gray8_bt709",
            desc(T::U8, L::Rgb, n, Tf::Srgb, bt709),
            desc(T::U8, L::Gray, n, Tf::Srgb, bt709),
        ),
        Case::new(
            "gray_u8",
            "rgb8_to_gray8_bt601",
            desc(T::U8, L::Rgb, n, Tf::Srgb, bt709),
            desc(T::U8, L::Gray, n, Tf::Srgb, bt709),
        )
        .opts(bt601_opts),
        Case::new(
            "gray_u8",
            "rgba8_to_gray8_bt709",
            desc(T::U8, L::Rgba, s, Tf::Srgb, bt709),
            desc(T::U8, L::Gray, n, Tf::Srgb, bt709),
        ),
        Case::new(
            "gray_u8",
            "rgba8_to_gray8_bt601",
            desc(T::U8, L::Rgba, s, Tf::Srgb, bt709),
            desc(T::U8, L::Gray, n, Tf::Srgb, bt709),
        )
        .opts(bt601_opts),
        Case::new(
            "gray_u8",
            "rgba8_to_graya8",
            desc(T::U8, L::Rgba, s, Tf::Srgb, bt709),
            desc(T::U8, L::GrayAlpha, s, Tf::Srgb, bt709),
        ),
        Case::new(
            "gray_u8",
            "gray8_to_rgb8",
            desc(T::U8, L::Gray, n, Tf::Srgb, bt709),
            desc(T::U8, L::Rgb, n, Tf::Srgb, bt709),
        ),
        Case::new(
            "gray_u8",
            "gray8_to_rgba8",
            desc(T::U8, L::Gray, n, Tf::Srgb, bt709),
            desc(T::U8, L::Rgba, s, Tf::Srgb, bt709),
        ),
        Case::new(
            "gray_u8",
            "graya8_to_rgba8",
            desc(T::U8, L::GrayAlpha, s, Tf::Srgb, bt709),
            desc(T::U8, L::Rgba, s, Tf::Srgb, bt709),
        ),
        Case::new(
            "gray_u8",
            "graya8_to_rgb8",
            desc(T::U8, L::GrayAlpha, s, Tf::Srgb, bt709),
            desc(T::U8, L::Rgb, n, Tf::Srgb, bt709),
        ),
        Case::new(
            "gray_u8",
            "graya8_to_gray8",
            desc(T::U8, L::GrayAlpha, s, Tf::Srgb, bt709),
            desc(T::U8, L::Gray, n, Tf::Srgb, bt709),
        ),
        // ── 4. gray_wide ─────────────────────────────────────────────────────
        Case::new(
            "gray_wide",
            "rgb16_to_gray16",
            desc(T::U16, L::Rgb, n, Tf::Srgb, bt709),
            desc(T::U16, L::Gray, n, Tf::Srgb, bt709),
        ),
        Case::new(
            "gray_wide",
            "rgba16_to_gray16",
            desc(T::U16, L::Rgba, s, Tf::Srgb, bt709),
            desc(T::U16, L::Gray, n, Tf::Srgb, bt709),
        ),
        Case::new(
            "gray_wide",
            "rgbf16_to_grayf16",
            desc(T::F16, L::Rgb, n, Tf::Linear, bt709),
            desc(T::F16, L::Gray, n, Tf::Linear, bt709),
        ),
        Case::new(
            "gray_wide",
            "rgbf32_to_grayf32",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
            desc(T::F32, L::Gray, n, Tf::Linear, bt709),
        ),
        Case::new(
            "gray_wide",
            "rgbaf32_to_grayf32",
            desc(T::F32, L::Rgba, s, Tf::Linear, bt709),
            desc(T::F32, L::Gray, n, Tf::Linear, bt709),
        ),
        Case::new(
            "gray_wide",
            "gray16_to_rgb16",
            desc(T::U16, L::Gray, n, Tf::Srgb, bt709),
            desc(T::U16, L::Rgb, n, Tf::Srgb, bt709),
        ),
        Case::new(
            "gray_wide",
            "graya16_to_rgba16",
            desc(T::U16, L::GrayAlpha, s, Tf::Srgb, bt709),
            desc(T::U16, L::Rgba, s, Tf::Srgb, bt709),
        ),
        Case::new(
            "gray_wide",
            "grayaf32_to_rgbaf32",
            desc(T::F32, L::GrayAlpha, s, Tf::Linear, bt709),
            desc(T::F32, L::Rgba, s, Tf::Linear, bt709),
        ),
        // ── 5. depth (incl. FloatFacts & Truncate) ───────────────────────────
        Case::new(
            "depth",
            "rgb8_to_rgbf32_naive",
            desc(T::U8, L::Rgb, n, Tf::Linear, bt709),
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
        ),
        Case::new(
            "depth",
            "rgb16_to_rgbf32_naive",
            desc(T::U16, L::Rgb, n, Tf::Linear, bt709),
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
        ),
        Case::new(
            "depth",
            "rgbf16_to_rgbf32",
            desc(T::F16, L::Rgb, n, Tf::Linear, bt709),
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
        ),
        Case::new(
            "depth",
            "rgbf32_to_rgbf16",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
            desc(T::F16, L::Rgb, n, Tf::Linear, bt709),
        ),
        Case::new(
            "depth",
            "rgb16_to_rgb8_round",
            desc(T::U16, L::Rgb, n, Tf::Srgb, bt709),
            desc(T::U8, L::Rgb, n, Tf::Srgb, bt709),
        ),
        Case::new(
            "depth",
            "rgb16_to_rgb8_trunc",
            desc(T::U16, L::Rgb, n, Tf::Srgb, bt709),
            desc(T::U8, L::Rgb, n, Tf::Srgb, bt709),
        )
        .opts(trunc_opts),
        Case::new(
            "depth",
            "rgb8_to_rgb16",
            desc(T::U8, L::Rgb, n, Tf::Srgb, bt709),
            desc(T::U16, L::Rgb, n, Tf::Srgb, bt709),
        ),
        Case::new(
            "depth",
            "rgbf32_to_rgb8_round_untrusted",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
            desc(T::U8, L::Rgb, n, Tf::Linear, bt709),
        ),
        Case::new(
            "depth",
            "rgbf32_to_rgb8_round_no_nan",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
            desc(T::U8, L::Rgb, n, Tf::Linear, bt709),
        )
        .facts(FloatFacts::NO_NAN),
        Case::new(
            "depth",
            "rgbf32_to_rgb8_round_unit_clean",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
            desc(T::U8, L::Rgb, n, Tf::Linear, bt709),
        )
        .facts(UNIT_CLEAN),
        Case::new(
            "depth",
            "rgbf32_to_rgb8_trunc_untrusted",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
            desc(T::U8, L::Rgb, n, Tf::Linear, bt709),
        )
        .opts(trunc_opts),
        Case::new(
            "depth",
            "rgbf32_to_rgb8_trunc_unit_clean",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
            desc(T::U8, L::Rgb, n, Tf::Linear, bt709),
        )
        .opts(trunc_opts)
        .facts(UNIT_CLEAN),
        Case::new(
            "depth",
            "rgbf32_to_rgb16_round_untrusted",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
            desc(T::U16, L::Rgb, n, Tf::Linear, bt709),
        ),
        Case::new(
            "depth",
            "rgbf32_to_rgb16_round_no_nan",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
            desc(T::U16, L::Rgb, n, Tf::Linear, bt709),
        )
        .facts(FloatFacts::NO_NAN),
        Case::new(
            "depth",
            "rgbf32_to_rgb16_round_unit_clean",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
            desc(T::U16, L::Rgb, n, Tf::Linear, bt709),
        )
        .facts(UNIT_CLEAN),
        Case::new(
            "depth",
            "rgbf32_to_rgb16_trunc_untrusted",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
            desc(T::U16, L::Rgb, n, Tf::Linear, bt709),
        )
        .opts(trunc_opts),
        Case::new(
            "depth",
            "rgbf32_to_rgb16_trunc_unit_clean",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
            desc(T::U16, L::Rgb, n, Tf::Linear, bt709),
        )
        .opts(trunc_opts)
        .facts(UNIT_CLEAN),
        // ── 6. transfer_sdr ──────────────────────────────────────────────────
        Case::new(
            "transfer_sdr",
            "rgb8_srgb_to_rgbf32_lin",
            desc(T::U8, L::Rgb, n, Tf::Srgb, bt709),
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
        ),
        Case::new(
            "transfer_sdr",
            "rgbf32_lin_to_rgb8_srgb",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
            desc(T::U8, L::Rgb, n, Tf::Srgb, bt709),
        ),
        Case::new(
            "transfer_sdr",
            "rgba8_srgb_to_rgbaf32_lin",
            desc(T::U8, L::Rgba, s, Tf::Srgb, bt709),
            desc(T::F32, L::Rgba, s, Tf::Linear, bt709),
        ),
        Case::new(
            "transfer_sdr",
            "rgbaf32_lin_to_rgba8_srgb",
            desc(T::F32, L::Rgba, s, Tf::Linear, bt709),
            desc(T::U8, L::Rgba, s, Tf::Srgb, bt709),
        ),
        Case::new(
            "transfer_sdr",
            "rgbf32_srgb_to_lin_clip",
            desc(T::F32, L::Rgb, n, Tf::Srgb, bt709),
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
        ),
        Case::new(
            "transfer_sdr",
            "rgbf32_lin_to_srgb_clip",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
            desc(T::F32, L::Rgb, n, Tf::Srgb, bt709),
        ),
        Case::new(
            "transfer_sdr",
            "rgbf32_srgb_to_lin_ext",
            desc(T::F32, L::Rgb, n, Tf::Srgb, bt709),
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
        )
        .opts(ext_opts)
        .fill(FillKind::ExtendedSigned),
        Case::new(
            "transfer_sdr",
            "rgbf32_lin_to_srgb_ext",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
            desc(T::F32, L::Rgb, n, Tf::Srgb, bt709),
        )
        .opts(ext_opts)
        .fill(FillKind::ExtendedSigned),
        Case::new(
            "transfer_sdr",
            "rgbf32_bt709_to_lin",
            desc(T::F32, L::Rgb, n, Tf::Bt709, bt709),
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
        ),
        Case::new(
            "transfer_sdr",
            "rgbf32_lin_to_bt709",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
            desc(T::F32, L::Rgb, n, Tf::Bt709, bt709),
        ),
        Case::new(
            "transfer_sdr",
            "rgbf32_gamma22_to_lin_clip",
            desc(T::F32, L::Rgb, n, Tf::Gamma22, adobe),
            desc(T::F32, L::Rgb, n, Tf::Linear, adobe),
        ),
        Case::new(
            "transfer_sdr",
            "rgbf32_lin_to_gamma22_clip",
            desc(T::F32, L::Rgb, n, Tf::Linear, adobe),
            desc(T::F32, L::Rgb, n, Tf::Gamma22, adobe),
        ),
        Case::new(
            "transfer_sdr",
            "rgbf32_gamma22_to_lin_ext_untrusted",
            desc(T::F32, L::Rgb, n, Tf::Gamma22, adobe),
            desc(T::F32, L::Rgb, n, Tf::Linear, adobe),
        )
        .opts(ext_opts)
        .fill(FillKind::ExtendedSigned),
        Case::new(
            "transfer_sdr",
            "rgbf32_gamma22_to_lin_ext_unit_clean",
            desc(T::F32, L::Rgb, n, Tf::Gamma22, adobe),
            desc(T::F32, L::Rgb, n, Tf::Linear, adobe),
        )
        .opts(ext_opts)
        .facts(UNIT_CLEAN),
        Case::new(
            "transfer_sdr",
            "rgb16_srgb_to_rgb8_lin_lut",
            desc(T::U16, L::Rgb, n, Tf::Srgb, bt709),
            desc(T::U8, L::Rgb, n, Tf::Linear, bt709),
        ),
        // ── 7. transfer_hdr ──────────────────────────────────────────────────
        Case::new(
            "transfer_hdr",
            "rgb16_pq_to_rgbf32_lin",
            desc(T::U16, L::Rgb, n, Tf::Pq, bt2020),
            desc(T::F32, L::Rgb, n, Tf::Linear, bt2020),
        ),
        Case::new(
            "transfer_hdr",
            "rgbf32_lin_to_rgb16_pq",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt2020),
            desc(T::U16, L::Rgb, n, Tf::Pq, bt2020),
        ),
        Case::new(
            "transfer_hdr",
            "rgbf32_pq_to_lin_uniform",
            desc(T::F32, L::Rgb, n, Tf::Pq, bt2020),
            desc(T::F32, L::Rgb, n, Tf::Linear, bt2020),
        ),
        Case::new(
            "transfer_hdr",
            "rgbf32_lin_to_pq_uniform",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt2020),
            desc(T::F32, L::Rgb, n, Tf::Pq, bt2020),
        ),
        Case::new(
            "transfer_hdr",
            "rgbf32_pq_to_lin_near_black",
            desc(T::F32, L::Rgb, n, Tf::Pq, bt2020),
            desc(T::F32, L::Rgb, n, Tf::Linear, bt2020),
        )
        .fill(FillKind::NearBlackPq),
        Case::new(
            "transfer_hdr",
            "rgbf32_lin_to_pq_near_black",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt2020),
            desc(T::F32, L::Rgb, n, Tf::Pq, bt2020),
        )
        .fill(FillKind::NearBlackLinear),
        Case::new(
            "transfer_hdr",
            "rgb16_hlg_to_rgbf32_lin",
            desc(T::U16, L::Rgb, n, Tf::Hlg, bt2020),
            desc(T::F32, L::Rgb, n, Tf::Linear, bt2020),
        ),
        Case::new(
            "transfer_hdr",
            "rgbf32_lin_to_rgb16_hlg",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt2020),
            desc(T::U16, L::Rgb, n, Tf::Hlg, bt2020),
        ),
        Case::new(
            "transfer_hdr",
            "rgbf32_hlg_to_lin",
            desc(T::F32, L::Rgb, n, Tf::Hlg, bt2020),
            desc(T::F32, L::Rgb, n, Tf::Linear, bt2020),
        ),
        Case::new(
            "transfer_hdr",
            "rgbf32_lin_to_hlg",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt2020),
            desc(T::F32, L::Rgb, n, Tf::Hlg, bt2020),
        ),
        // ── 8. gamut (matlut + 3x3 + cross-TRC + soft_compress) ──────────────
        Case::new(
            "gamut",
            "p3_to_srgb_rgb8_matlut",
            desc(T::U8, L::Rgb, n, Tf::Srgb, p3),
            desc(T::U8, L::Rgb, n, Tf::Srgb, bt709),
        ),
        Case::new(
            "gamut",
            "p3_to_srgb_rgba8_matlut",
            desc(T::U8, L::Rgba, s, Tf::Srgb, p3),
            desc(T::U8, L::Rgba, s, Tf::Srgb, bt709),
        ),
        Case::new(
            "gamut",
            "p3_to_srgb_rgb16_matlut",
            desc(T::U16, L::Rgb, n, Tf::Srgb, p3),
            desc(T::U16, L::Rgb, n, Tf::Srgb, bt709),
        ),
        Case::new(
            "gamut",
            "p3_rgb8_to_srgb_lin_f32",
            desc(T::U8, L::Rgb, n, Tf::Srgb, p3),
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
        ),
        Case::new(
            "gamut",
            "p3_lin_f32_to_srgb_rgb8",
            desc(T::F32, L::Rgb, n, Tf::Linear, p3),
            desc(T::U8, L::Rgb, n, Tf::Srgb, bt709),
        ),
        Case::new(
            "gamut",
            "p3_to_srgb_lin_rgbf32",
            desc(T::F32, L::Rgb, n, Tf::Linear, p3),
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
        ),
        Case::new(
            "gamut",
            "p3_to_srgb_lin_rgbaf32",
            desc(T::F32, L::Rgba, s, Tf::Linear, p3),
            desc(T::F32, L::Rgba, s, Tf::Linear, bt709),
        ),
        Case::new(
            "gamut",
            "p3_to_srgb_srgb_rgbf32",
            desc(T::F32, L::Rgb, n, Tf::Srgb, p3),
            desc(T::F32, L::Rgb, n, Tf::Srgb, bt709),
        ),
        Case::new(
            "gamut",
            "p3_to_srgb_srgb_rgbaf32",
            desc(T::F32, L::Rgba, s, Tf::Srgb, p3),
            desc(T::F32, L::Rgba, s, Tf::Srgb, bt709),
        ),
        Case::new(
            "gamut",
            "bt2020_bt709_to_srgb_rgbf32",
            desc(T::F32, L::Rgb, n, Tf::Bt709, bt2020),
            desc(T::F32, L::Rgb, n, Tf::Srgb, bt709),
        ),
        Case::new(
            "gamut",
            "adobe_g22_to_srgb_rgbf32",
            desc(T::F32, L::Rgb, n, Tf::Gamma22, adobe),
            desc(T::F32, L::Rgb, n, Tf::Srgb, bt709),
        ),
        Case::new(
            "gamut",
            "p3_to_srgb_soft_compress_f32",
            desc(T::F32, L::Rgb, n, Tf::Linear, p3),
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
        )
        .opts(soft_opts),
        // ── 9. premul ────────────────────────────────────────────────────────
        Case::new(
            "premul",
            "rgba8_straight_to_premul",
            desc(T::U8, L::Rgba, s, Tf::Srgb, bt709),
            desc(T::U8, L::Rgba, pm, Tf::Srgb, bt709),
        ),
        Case::new(
            "premul",
            "rgba8_premul_to_straight",
            desc(T::U8, L::Rgba, pm, Tf::Srgb, bt709),
            desc(T::U8, L::Rgba, s, Tf::Srgb, bt709),
        ),
        Case::new(
            "premul",
            "rgba16_straight_to_premul",
            desc(T::U16, L::Rgba, s, Tf::Srgb, bt709),
            desc(T::U16, L::Rgba, pm, Tf::Srgb, bt709),
        ),
        Case::new(
            "premul",
            "rgba16_premul_to_straight",
            desc(T::U16, L::Rgba, pm, Tf::Srgb, bt709),
            desc(T::U16, L::Rgba, s, Tf::Srgb, bt709),
        ),
        Case::new(
            "premul",
            "rgbaf16_straight_to_premul",
            desc(T::F16, L::Rgba, s, Tf::Linear, bt709),
            desc(T::F16, L::Rgba, pm, Tf::Linear, bt709),
        ),
        Case::new(
            "premul",
            "rgbaf16_premul_to_straight",
            desc(T::F16, L::Rgba, pm, Tf::Linear, bt709),
            desc(T::F16, L::Rgba, s, Tf::Linear, bt709),
        ),
        Case::new(
            "premul",
            "rgbaf32_straight_to_premul_untrusted",
            desc(T::F32, L::Rgba, s, Tf::Linear, bt709),
            desc(T::F32, L::Rgba, pm, Tf::Linear, bt709),
        ),
        Case::new(
            "premul",
            "rgbaf32_straight_to_premul_unit_clean",
            desc(T::F32, L::Rgba, s, Tf::Linear, bt709),
            desc(T::F32, L::Rgba, pm, Tf::Linear, bt709),
        )
        .facts(UNIT_CLEAN),
        Case::new(
            "premul",
            "rgbaf32_premul_to_straight_untrusted",
            desc(T::F32, L::Rgba, pm, Tf::Linear, bt709),
            desc(T::F32, L::Rgba, s, Tf::Linear, bt709),
        ),
        Case::new(
            "premul",
            "rgbaf32_premul_to_straight_unit_clean",
            desc(T::F32, L::Rgba, pm, Tf::Linear, bt709),
            desc(T::F32, L::Rgba, s, Tf::Linear, bt709),
        )
        .facts(UNIT_CLEAN),
        Case::new(
            "premul",
            "graya8_straight_to_premul",
            desc(T::U8, L::GrayAlpha, s, Tf::Srgb, bt709),
            desc(T::U8, L::GrayAlpha, pm, Tf::Srgb, bt709),
        ),
        Case::new(
            "premul",
            "graya16_straight_to_premul",
            desc(T::U16, L::GrayAlpha, s, Tf::Srgb, bt709),
            desc(T::U16, L::GrayAlpha, pm, Tf::Srgb, bt709),
        ),
        Case::new(
            "premul",
            "grayaf16_straight_to_premul",
            desc(T::F16, L::GrayAlpha, s, Tf::Linear, bt709),
            desc(T::F16, L::GrayAlpha, pm, Tf::Linear, bt709),
        ),
        Case::new(
            "premul",
            "grayaf32_straight_to_premul",
            desc(T::F32, L::GrayAlpha, s, Tf::Linear, bt709),
            desc(T::F32, L::GrayAlpha, pm, Tf::Linear, bt709),
        ),
        // ── 10. matte ────────────────────────────────────────────────────────
        Case::new(
            "matte",
            "rgba8_srgb",
            desc(T::U8, L::Rgba, s, Tf::Srgb, bt709),
            desc(T::U8, L::Rgb, n, Tf::Srgb, bt709),
        )
        .opts(matte_opts),
        Case::new(
            "matte",
            "rgba8_linear",
            desc(T::U8, L::Rgba, s, Tf::Linear, bt709),
            desc(T::U8, L::Rgb, n, Tf::Linear, bt709),
        )
        .opts(matte_opts),
        Case::new(
            "matte",
            "rgba16_srgb",
            desc(T::U16, L::Rgba, s, Tf::Srgb, bt709),
            desc(T::U16, L::Rgb, n, Tf::Srgb, bt709),
        )
        .opts(matte_opts),
        Case::new(
            "matte",
            "rgbaf16_srgb",
            desc(T::F16, L::Rgba, s, Tf::Srgb, bt709),
            desc(T::F16, L::Rgb, n, Tf::Srgb, bt709),
        )
        .opts(matte_opts),
        Case::new(
            "matte",
            "rgbaf32_srgb",
            desc(T::F32, L::Rgba, s, Tf::Srgb, bt709),
            desc(T::F32, L::Rgb, n, Tf::Srgb, bt709),
        )
        .opts(matte_opts),
        Case::new(
            "matte",
            "rgbaf32_linear",
            desc(T::F32, L::Rgba, s, Tf::Linear, bt709),
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
        )
        .opts(matte_opts),
        Case::new(
            "matte",
            "rgbaf32_pq",
            desc(T::F32, L::Rgba, s, Tf::Pq, bt2020),
            desc(T::F32, L::Rgb, n, Tf::Pq, bt2020),
        )
        .opts(matte_opts),
        // ── 11. oklab (all FloatFacts regimes + 3ch/4ch) ─────────────────────
        Case::new(
            "oklab",
            "rgb_to_oklab_untrusted",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
            desc(T::F32, L::Oklab, n, Tf::Unknown, bt709),
        )
        .fill(FillKind::PositiveNoSub),
        Case::new(
            "oklab",
            "rgb_to_oklab_finite",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
            desc(T::F32, L::Oklab, n, Tf::Unknown, bt709),
        )
        .facts(FINITE)
        .fill(FillKind::PositiveNoSub),
        Case::new(
            "oklab",
            "rgb_to_oklab_finite_nonneg",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
            desc(T::F32, L::Oklab, n, Tf::Unknown, bt709),
        )
        .facts(FINITE_NONNEG)
        .fill(FillKind::PositiveNoSub),
        Case::new(
            "oklab",
            "rgb_to_oklab_finite_nonneg_nosub",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
            desc(T::F32, L::Oklab, n, Tf::Unknown, bt709),
        )
        .facts(FINITE_NONNEG_NOSUB)
        .fill(FillKind::PositiveNoSub),
        Case::new(
            "oklab",
            "rgb_to_oklab_finite_pos_nosub",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
            desc(T::F32, L::Oklab, n, Tf::Unknown, bt709),
        )
        .facts(FINITE_POS_NOSUB)
        .fill(FillKind::PositiveNoSub),
        Case::new(
            "oklab",
            "rgba_to_oklaba_untrusted",
            desc(T::F32, L::Rgba, s, Tf::Linear, bt709),
            desc(T::F32, L::OklabA, s, Tf::Unknown, bt709),
        )
        .fill(FillKind::PositiveNoSub),
        Case::new(
            "oklab",
            "rgba_to_oklaba_finite_nonneg_nosub",
            desc(T::F32, L::Rgba, s, Tf::Linear, bt709),
            desc(T::F32, L::OklabA, s, Tf::Unknown, bt709),
        )
        .facts(FINITE_NONNEG_NOSUB)
        .fill(FillKind::PositiveNoSub),
        Case::new(
            "oklab",
            "oklab_to_rgb",
            desc(T::F32, L::Oklab, n, Tf::Unknown, bt709),
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
        )
        .fill(FillKind::Oklab),
        Case::new(
            "oklab",
            "oklaba_to_rgba",
            desc(T::F32, L::OklabA, s, Tf::Unknown, bt709),
            desc(T::F32, L::Rgba, s, Tf::Linear, bt709),
        )
        .fill(FillKind::Oklab),
        // ── 12. perceptual ───────────────────────────────────────────────────
        Case::new(
            "perceptual",
            "oklab_to_oklch",
            desc(T::F32, L::Oklab, n, Tf::Unknown, bt709),
            desc(T::F32, L::Oklch, n, Tf::Unknown, bt709),
        )
        .fill(FillKind::Oklab),
        Case::new(
            "perceptual",
            "oklch_to_oklab",
            desc(T::F32, L::Oklch, n, Tf::Unknown, bt709),
            desc(T::F32, L::Oklab, n, Tf::Unknown, bt709),
        )
        .fill(FillKind::Oklch),
        Case::new(
            "perceptual",
            "lin_rgb_to_lch",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
            desc(T::F32, L::Lch, n, Tf::Unknown, bt709),
        ),
        Case::new(
            "perceptual",
            "lch_to_lin_rgb",
            desc(T::F32, L::Lch, n, Tf::Unknown, bt709),
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
        )
        .fill(FillKind::Lch),
        Case::new(
            "perceptual",
            "lin_rgb_to_jzazbz",
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
            desc(T::F32, L::Jzazbz, n, Tf::Unknown, bt709),
        ),
        Case::new(
            "perceptual",
            "jzazbz_to_lin_rgb",
            desc(T::F32, L::Jzazbz, n, Tf::Unknown, bt709),
            desc(T::F32, L::Rgb, n, Tf::Linear, bt709),
        )
        .fill(FillKind::Jzazbz),
        Case::new(
            "perceptual",
            "pq_rgb_to_jzazbz",
            desc(T::F32, L::Rgb, n, Tf::Pq, bt2020),
            desc(T::F32, L::Jzazbz, n, Tf::Unknown, bt709),
        ),
        Case::new(
            "perceptual",
            "srgb_rgb_to_hsl",
            desc(T::F32, L::Rgb, n, Tf::Srgb, bt709),
            desc(T::F32, L::Hsl, n, Tf::Srgb, bt709),
        ),
        Case::new(
            "perceptual",
            "hsl_to_srgb_rgb",
            desc(T::F32, L::Hsl, n, Tf::Srgb, bt709),
            desc(T::F32, L::Rgb, n, Tf::Srgb, bt709),
        )
        .fill(FillKind::HslHsv),
        Case::new(
            "perceptual",
            "srgb_rgb_to_hsv",
            desc(T::F32, L::Rgb, n, Tf::Srgb, bt709),
            desc(T::F32, L::Hsv, n, Tf::Srgb, bt709),
        ),
        Case::new(
            "perceptual",
            "hsv_to_srgb_rgb",
            desc(T::F32, L::Hsv, n, Tf::Srgb, bt709),
            desc(T::F32, L::Rgb, n, Tf::Srgb, bt709),
        )
        .fill(FillKind::HslHsv),
    ]
}

fn orient_cases() -> Vec<(&'static str, D)> {
    let s = Some(AlphaMode::Straight);
    let n = None;
    vec![
        (
            "rotate90_gray8_1bpp",
            desc(T::U8, L::Gray, n, Tf::Srgb, P::Bt709),
        ),
        (
            "rotate90_rgb8_3bpp",
            desc(T::U8, L::Rgb, n, Tf::Srgb, P::Bt709),
        ),
        (
            "rotate90_rgba8_4bpp",
            desc(T::U8, L::Rgba, s, Tf::Srgb, P::Bt709),
        ),
        (
            "rotate90_rgba16_8bpp",
            desc(T::U16, L::Rgba, s, Tf::Srgb, P::Bt709),
        ),
        (
            "rotate90_rgbf32_12bpp",
            desc(T::F32, L::Rgb, n, Tf::Linear, P::Bt709),
        ),
        (
            "rotate90_rgbaf32_16bpp",
            desc(T::F32, L::Rgba, s, Tf::Linear, P::Bt709),
        ),
    ]
}

fn main() {
    let tiers = active_tiers();
    let rounds: usize = std::env::var("BENCH_4K_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    let max_ms: u64 = std::env::var("BENCH_4K_MAX_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(900);

    let shared = Arc::new(Mutex::new(FrameScratch::new()));

    zenbench::run(|suite| {
        for case in all_cases() {
            let spec = ConversionSpec::new(case.options).with_float_facts(case.facts);
            let plan =
                ConvertPlan::build(case.from.into(), case.to.into(), &spec).unwrap_or_else(|e| {
                    panic!(
                        "bench_4k_paths: failed to plan {}/{}: {e:?}",
                        case.group, case.name
                    )
                });
            let src_stride = (WIDTH as usize) * case.from.bytes_per_pixel();
            let dst_stride = (WIDTH as usize) * case.to.bytes_per_pixel();
            let src_bytes = src_stride * (HEIGHT as usize);
            let dst_bytes = dst_stride * (HEIGHT as usize);
            let total_bytes = (src_bytes + dst_bytes) as u64;

            let group_id = format!("{}/{}", case.group, case.name);
            let tiers = tiers.clone();
            let shared = Arc::clone(&shared);
            suite.group(group_id, move |g| {
                g.throughput(Throughput::Bytes(total_bytes));
                g.config()
                    .auto_rounds(false)
                    .min_rounds(rounds)
                    .max_rounds(rounds)
                    .max_time(Duration::from_millis(max_ms));
                for &tier in &tiers {
                    let mut worker = plan
                        .prepare_shared(WIDTH)
                        .expect("prepare_shared 4K worker");
                    let shared = Arc::clone(&shared);
                    let from = case.from;
                    let fill = case.fill;
                    g.bench(tier, move |b| {
                        set_tier(tier);
                        let mut scratch = shared.lock().unwrap();
                        fill_frame(&mut scratch, from, fill);
                        let (src, dst) = scratch.slices_mut(src_bytes, dst_bytes);
                        // Warm up lazy tables (e.g., SrgbU8Encoder, SDR u16 LUTs) once.
                        worker
                            .convert_raw_rows(src, src_stride, dst, dst_stride, WIDTH, 1)
                            .unwrap();
                        b.iter(|| {
                            worker
                                .convert_raw_rows(src, src_stride, dst, dst_stride, WIDTH, HEIGHT)
                                .unwrap();
                            black_box(dst[dst_bytes - 1])
                        })
                    });
                }
            });
        }

        // ── 13. orient_4k ────────────────────────────────────────────────────
        for (name, format) in orient_cases() {
            let bpp = format.bytes_per_pixel();
            let src_stride = (WIDTH as usize) * bpp;
            let dst_stride = (HEIGHT as usize) * bpp; // Rotate90 swaps W and H
            let frame_bytes = PIXELS * bpp;
            let tiers = tiers.clone();
            let shared = Arc::clone(&shared);
            suite.group(format!("orient_4k/{name}"), move |g| {
                g.throughput(Throughput::Bytes((frame_bytes * 2) as u64));
                g.config()
                    .auto_rounds(false)
                    .min_rounds(rounds)
                    .max_rounds(rounds)
                    .max_time(Duration::from_millis(max_ms));
                for &tier in &tiers {
                    let shared = Arc::clone(&shared);
                    g.bench(tier, move |b| {
                        set_tier(tier);
                        let mut scratch = shared.lock().unwrap();
                        fill_frame(&mut scratch, format, FillKind::UniformUnit);
                        let (src, dst) = scratch.slices_mut(frame_bytes, frame_bytes);
                        b.iter(|| {
                            let s =
                                PixelSlice::new(src, WIDTH, HEIGHT, src_stride, format).unwrap();
                            let d =
                                PixelSliceMut::new(dst, HEIGHT, WIDTH, dst_stride, format).unwrap();
                            s.oriented_into(Orientation::Rotate90, d).unwrap();
                            black_box(dst[frame_bytes - 1])
                        })
                    });
                }
            });
        }

        set_tier(tiers[0]);
    });
}
