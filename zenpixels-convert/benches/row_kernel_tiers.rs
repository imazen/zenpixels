//! Speed of each `RowConverter` row kernel at each dispatch tier.
//!
//! Tells a kernel worth compiling per tier from one the baseline build already handles. A
//! kernel whose SIMD tier doesn't beat `baseline` costs binary size and buys nothing.
//!
//! `baseline` turns the runtime tiers off, so it runs what a plain function compiles to:
//! on x86-64 SSE2 with autovectorization, on aarch64 NEON with autovectorization.
//!
//! `--features avx512` adds the `v4x` tier that kernels opt into.
//!
//! Run with `RUSTFLAGS` unset: a `target-cpu` that guarantees a tier can't be turned off.
//!
//! Run: `cargo bench -p zenpixels-convert --bench row_kernel_tiers`
//! One family: `cargo bench -p zenpixels-convert --bench row_kernel_tiers -- --group='gray'`

use zenbench::prelude::*;
use zenpixels::{
    AlphaMode, ChannelLayout as L, ChannelType as T, LumaCoefficients, PixelDescriptor as D,
    PixelFormat, TransferFunction as Tf,
};
use zenpixels_convert::{AlphaPolicy, ConvertOptions, RowConverter};

#[cfg(target_arch = "x86_64")]
const TIERS: &[&str] = if cfg!(feature = "avx512") {
    &["v4x", "v3", "baseline"]
} else {
    &["v3", "baseline"]
};
#[cfg(target_arch = "aarch64")]
const TIERS: &[&str] = &["neon", "baseline"];
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const TIERS: &[&str] = &["baseline"];

fn set_tier(tier: &str) {
    #[cfg(target_arch = "x86_64")]
    {
        // In this order: changing a token also changes its descendants, so enabling V3
        // re-enables V4 and V4x.
        let _ = archmage::X64V3Token::dangerously_disable_token_process_wide(tier == "baseline");
        let _ = archmage::X64V4Token::dangerously_disable_token_process_wide(tier != "v4x");
        let _ = archmage::X64V4xToken::dangerously_disable_token_process_wide(tier != "v4x");
    }
    #[cfg(target_arch = "aarch64")]
    let _ = archmage::NeonToken::dangerously_disable_token_process_wide(tier == "baseline");
    let _ = tier;
}

fn d(c: T, l: L, a: Option<AlphaMode>, t: Tf) -> Option<D> {
    PixelFormat::from_parts(c, l, a).map(|_| D::new(c, l, a, t))
}

/// A buffer whose start sits `page_offset` bytes into a 4 KiB page, returned as (start, bytes).
///
/// Fixes the placement of source and destination: at an arbitrary distance apart, 4 KiB
/// aliasing and alignment moved identical kernels by 40% between runs.
fn placed(len: usize, page_offset: usize) -> (usize, Vec<u8>) {
    let bytes = vec![0u8; len + 2 * 4096];
    let start = (4096 - bytes.as_ptr() as usize % 4096) % 4096 + page_offset;
    (start, bytes)
}

/// Random samples, uniform in `[0, 1]` for floats. `near black` cases sit under the PQ
/// polynomials' floors, which is their worst case.
fn fill_source(bytes: &mut [u8], from: D, name: &str) {
    let scale = match (name.contains("near black"), from.transfer()) {
        (true, Tf::Linear) => 1e-7,
        (true, _) => 0.02,
        _ => 1.0,
    };
    let mut x = 0x2545_F491u32;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        x
    };
    match from.channel_type() {
        T::F32 => {
            for v in bytes.as_chunks_mut::<4>().0 {
                let f = (next() >> 8) as f32 / 16777215.0 * scale;
                *v = f.to_ne_bytes();
            }
        }
        T::F16 => {
            for v in bytes.as_chunks_mut::<2>().0 {
                *v = half::f16::from_f32((next() >> 8) as f32 / 16777215.0).to_ne_bytes();
            }
        }
        _ => bytes.fill_with(|| next() as u8),
    }
}

fn options(name: &str) -> ConvertOptions {
    let alpha = if name.starts_with("matte") {
        AlphaPolicy::CompositeOnto {
            r: 200,
            g: 100,
            b: 50,
        }
    } else {
        AlphaPolicy::DiscardUnchecked
    };
    ConvertOptions::permissive()
        .with_luma(Some(LumaCoefficients::Bt601))
        .with_alpha_policy(alpha)
}

/// Every kernel family, by name, from one descriptor to another.
fn cases() -> Vec<(String, Option<D>, Option<D>)> {
    let s = Some(AlphaMode::Straight);
    let p = Some(AlphaMode::Premultiplied);
    let (lin, srgb) = (Tf::Linear, Tf::Srgb);
    let mut cases = Vec::new();
    for (tn, t) in [("u16", T::U16), ("f16", T::F16), ("f32", T::F32)] {
        let mut c = |n: &str, f, to| cases.push((format!("{n} {tn}"), f, to));
        c("add alpha", d(t, L::Rgb, None, lin), d(t, L::Rgba, s, lin));
        c("drop alpha", d(t, L::Rgba, s, lin), d(t, L::Rgb, None, lin));
        c(
            "gray to rgb",
            d(t, L::Gray, None, lin),
            d(t, L::Rgb, None, lin),
        );
        c(
            "gray to rgba",
            d(t, L::Gray, None, lin),
            d(t, L::Rgba, s, lin),
        );
        c(
            "rgb to gray",
            d(t, L::Rgb, None, lin),
            d(t, L::Gray, None, lin),
        );
        c(
            "rgba to gray",
            d(t, L::Rgba, s, lin),
            d(t, L::Gray, None, lin),
        );
        c(
            "ga to rgba",
            d(t, L::GrayAlpha, s, lin),
            d(t, L::Rgba, s, lin),
        );
        c(
            "ga to rgb",
            d(t, L::GrayAlpha, s, lin),
            d(t, L::Rgb, None, lin),
        );
        c(
            "gray to ga",
            d(t, L::Gray, None, lin),
            d(t, L::GrayAlpha, s, lin),
        );
        c(
            "ga to gray",
            d(t, L::GrayAlpha, s, lin),
            d(t, L::Gray, None, lin),
        );
    }
    for (tn, t) in [("u8", T::U8), ("u16", T::U16), ("f32", T::F32)] {
        let tf = if t == T::F32 { lin } else { srgb };
        let mut c = |n: &str, f, to| cases.push((format!("{n} {tn}"), f, to));
        c(
            "premul ga",
            d(t, L::GrayAlpha, s, tf),
            d(t, L::GrayAlpha, p, tf),
        );
        c("premul rgba", d(t, L::Rgba, s, tf), d(t, L::Rgba, p, tf));
        c(
            "unpremul ga",
            d(t, L::GrayAlpha, p, tf),
            d(t, L::GrayAlpha, s, tf),
        );
        c("unpremul rgba", d(t, L::Rgba, p, tf), d(t, L::Rgba, s, tf));
    }
    for (tn, t) in [
        ("u8", T::U8),
        ("u16", T::U16),
        ("f16", T::F16),
        ("f32", T::F32),
    ] {
        for (fnm, tf) in [
            ("srgb", srgb),
            ("linear", lin),
            ("bt709", Tf::Bt709),
            ("pq", Tf::Pq),
            ("hlg", Tf::Hlg),
        ] {
            cases.push((
                format!("matte {tn} {fnm}"),
                d(t, L::Rgba, s, tf),
                d(t, L::Rgb, None, tf),
            ));
        }
    }
    let mut c = |n: &str, f, to| cases.push((n.to_string(), f, to));
    c(
        "rgb to gray u8",
        d(T::U8, L::Rgb, None, srgb),
        d(T::U8, L::Gray, None, srgb),
    );
    c(
        "rgba to gray u8",
        d(T::U8, L::Rgba, s, srgb),
        d(T::U8, L::Gray, None, srgb),
    );
    let (u, f) = (T::U16, T::F32);
    c(
        "hlg u16 to linear",
        d(u, L::Rgb, None, Tf::Hlg),
        d(f, L::Rgb, None, lin),
    );
    c(
        "linear to hlg u16",
        d(f, L::Rgb, None, lin),
        d(u, L::Rgb, None, Tf::Hlg),
    );
    c(
        "bt709 f32 to linear",
        d(f, L::Rgb, None, Tf::Bt709),
        d(f, L::Rgb, None, lin),
    );
    c(
        "linear to bt709 f32",
        d(f, L::Rgb, None, lin),
        d(f, L::Rgb, None, Tf::Bt709),
    );
    c(
        "hlg f32 to linear",
        d(f, L::Rgb, None, Tf::Hlg),
        d(f, L::Rgb, None, lin),
    );
    c(
        "linear to hlg f32",
        d(f, L::Rgb, None, lin),
        d(f, L::Rgb, None, Tf::Hlg),
    );
    c(
        "pq u16 to linear",
        d(u, L::Rgb, None, Tf::Pq),
        d(f, L::Rgb, None, lin),
    );
    c(
        "linear to pq u16",
        d(f, L::Rgb, None, lin),
        d(u, L::Rgb, None, Tf::Pq),
    );
    c(
        "pq f32 to linear",
        d(f, L::Rgb, None, Tf::Pq),
        d(f, L::Rgb, None, lin),
    );
    c(
        "linear to pq f32",
        d(f, L::Rgb, None, lin),
        d(f, L::Rgb, None, Tf::Pq),
    );
    c(
        "pq f32 to linear, near black",
        d(f, L::Rgb, None, Tf::Pq),
        d(f, L::Rgb, None, lin),
    );
    c(
        "linear to pq f32, near black",
        d(f, L::Rgb, None, lin),
        d(f, L::Rgb, None, Tf::Pq),
    );
    let un = Tf::Unknown;
    c(
        "rgb to oklab",
        d(f, L::Rgb, None, lin),
        d(f, L::Oklab, None, un),
    );
    c(
        "oklab to rgb",
        d(f, L::Oklab, None, un),
        d(f, L::Rgb, None, lin),
    );
    c(
        "rgba to oklaba",
        d(f, L::Rgba, s, lin),
        d(f, L::OklabA, s, un),
    );
    c(
        "oklaba to rgba",
        d(f, L::OklabA, s, un),
        d(f, L::Rgba, s, lin),
    );
    cases
}

fn main() {
    zenbench::run(|suite| {
        for (name, from, to) in cases() {
            let (Some(from), Some(to)) = (from, to) else {
                eprintln!("skip {name}: no such format");
                continue;
            };
            if let Err(e) = RowConverter::new_explicit(from, to, &options(&name)) {
                eprintln!("skip {name}: {e}");
                continue;
            }
            for width in [16usize, 1024] {
                let src_bytes = width * from.bytes_per_pixel();
                let dst_bytes = width * to.bytes_per_pixel();
                let name = name.clone();
                suite.group(format!("{name}, {width} px"), move |g| {
                    g.throughput(Throughput::Bytes(src_bytes as u64));
                    g.config().auto_rounds(false).min_rounds(20).max_rounds(20);
                    for &tier in TIERS {
                        let mut c = RowConverter::new_explicit(from, to, &options(&name)).unwrap();
                        let (s0, mut src) = placed(src_bytes, 0);
                        fill_source(&mut src[s0..s0 + src_bytes], from, &name);
                        let (d0, mut dst) = placed(dst_bytes, 2048);
                        g.bench(tier, move |b| {
                            set_tier(tier);
                            b.iter(|| {
                                set_tier(tier);
                                let s = &src[s0..s0 + src_bytes];
                                let o = &mut dst[d0..d0 + dst_bytes];
                                c.convert_row(s, o, width as u32);
                                black_box(o[0])
                            })
                        });
                    }
                });
            }
        }
        set_tier(TIERS[0]);
    });
}
