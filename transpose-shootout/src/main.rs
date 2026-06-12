//! Transpose shootout: 1/2/3/4-channel interleaved image transpose (and
//! Rotate90 where supported) across every known Rust implementation plus the
//! C++ Simd library (`--features cpp-simd`).
//!
//! Contestants per channel count:
//! * `zpc`   — zenpixels-convert `apply_orientation_into` (ours)
//! * `ft`    — fast_transpose 0.2.7 (default features: their SIMD on)
//! * `ejm`   — transpose 0.2.3 (ejmahler; pure transpose only)
//! * `zune`  — zune-imageprocs 0.5.1 (planar u8; u16/u32 reinterpret views
//!             for the 2/4-channel groups; pure transpose only)
//! * `simd++`— ermig1979/Simd `SimdTransformImage` (1/2/3/4 bpp, both ops)
//!
//! Fairness rules:
//! * Every contestant writes into a caller-provided, pre-touched destination
//!   (no allocation, no first-touch faults in the timed region).
//! * Tight strides everywhere (`stride == width * ch`) so the stride-less
//!   `transpose` crate competes on equal terms.
//! * Buffers are backed by `Vec<u32>` so the u16/u32 reinterpret views for
//!   zune are always aligned.
//! * Before benching, every contestant's output is verified against an
//!   independent naive double-loop oracle; fast_transpose's
//!   (FlipMode, FlopMode) combination and Simd's `SimdTransformType` value
//!   are *derived by probing* against that oracle, so a flag-semantics
//!   misread cannot silently bench the wrong operation. Mismatches abort.
//!
//! Run: `cargo run --release [--features cpp-simd] [-- --group="T 3ch 12MP"]`

use std::time::Duration;

use zenbench::prelude::*;
use zenpixels::{Orientation, PixelDescriptor, PixelSlice, PixelSliceMut};
use zenpixels_convert::orient::{__bench_apply_orientation_staged, apply_orientation_into};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
enum Op {
    Transpose,
    Rotate90,
}

impl Op {
    fn orientation(self) -> Orientation {
        match self {
            Op::Transpose => Orientation::Transpose,
            Op::Rotate90 => Orientation::Rotate90,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Op::Transpose => "T",
            Op::Rotate90 => "R90",
        }
    }
}

/// Independent oracle: naive per-pixel double loop, no shared code with any
/// contestant. dst dims are (h, w) for both ops.
fn oracle(src: &[u8], w: usize, h: usize, ch: usize, op: Op) -> Vec<u8> {
    let mut dst = vec![0u8; src.len()];
    let (ow, _oh) = (h, w);
    for y in 0..h {
        for x in 0..w {
            // Transpose: (x, y) -> (y, x). Rotate90 (CW, EXIF 6 semantics,
            // matches zenpixels forward_map (h-1-sy, sx)): (x, y) -> (h-1-y, x).
            let (dx, dy) = match op {
                Op::Transpose => (y, x),
                Op::Rotate90 => (h - 1 - y, x),
            };
            let s = (y * w + x) * ch;
            let d = (dy * ow + dx) * ch;
            dst[d..d + ch].copy_from_slice(&src[s..s + ch]);
        }
    }
    dst
}

/// u32-backed byte buffer => u16/u32 reinterpret views are always aligned
/// (zune's 2/4-channel entries cast to &[u16]/&[u32]; a bare Vec<u8> can be
/// misaligned for that). Used for sources AND destinations.
#[derive(Clone)]
struct Buf(Vec<u32>);
impl Buf {
    fn new(bytes: usize) -> Self {
        Buf(vec![0u32; bytes.div_ceil(4)])
    }
    /// Deterministic test pattern.
    fn pattern(bytes: usize) -> Self {
        let mut b = Self::new(bytes);
        for (i, v) in b.bytes_mut(bytes).iter_mut().enumerate() {
            *v = (i.wrapping_mul(31) % 251) as u8;
        }
        b
    }
    fn bytes(&self, len: usize) -> &[u8] {
        &bytemuck::cast_slice(&self.0)[..len]
    }
    fn bytes_mut(&mut self, len: usize) -> &mut [u8] {
        &mut bytemuck::cast_slice_mut(&mut self.0)[..len]
    }
}

fn pattern(bytes: usize) -> Buf {
    Buf::pattern(bytes)
}

fn descriptor(ch: usize) -> PixelDescriptor {
    match ch {
        1 => PixelDescriptor::GRAY8,
        2 => PixelDescriptor::GRAYA8,
        3 => PixelDescriptor::RGB8,
        4 => PixelDescriptor::RGBA8,
        _ => unreachable!(),
    }
}

// ── contestants ──────────────────────────────────────────────────────────────

fn run_zpc(src: &[u8], dst: &mut [u8], w: usize, h: usize, ch: usize, op: Op) {
    let desc = descriptor(ch);
    let s = PixelSlice::new(src, w as u32, h as u32, w * ch, desc).unwrap();
    let d = PixelSliceMut::new(dst, h as u32, w as u32, h * ch, desc).unwrap();
    apply_orientation_into(s, op.orientation(), d).unwrap();
}

/// Experimental staged micro-tile path (bpp 1..=4) from zenpixels-convert.
fn run_zpc_staged(src: &[u8], dst: &mut [u8], w: usize, h: usize, ch: usize, op: Op) {
    let desc = descriptor(ch);
    let s = PixelSlice::new(src, w as u32, h as u32, w * ch, desc).unwrap();
    let d = PixelSliceMut::new(dst, h as u32, w as u32, h * ch, desc).unwrap();
    __bench_apply_orientation_staged(s, op.orientation(), d).unwrap();
}

fn run_ejm(src: &[u8], dst: &mut [u8], w: usize, h: usize, ch: usize) {
    match ch {
        1 => transpose::transpose(src, dst, w, h),
        2 => transpose::transpose::<[u8; 2]>(
            bytemuck::cast_slice(src),
            bytemuck::cast_slice_mut(dst),
            w,
            h,
        ),
        3 => transpose::transpose::<[u8; 3]>(
            bytemuck::cast_slice(src),
            bytemuck::cast_slice_mut(dst),
            w,
            h,
        ),
        4 => transpose::transpose::<[u8; 4]>(
            bytemuck::cast_slice(src),
            bytemuck::cast_slice_mut(dst),
            w,
            h,
        ),
        _ => unreachable!(),
    }
}

fn run_zune(src: &[u8], dst: &mut [u8], w: usize, h: usize, ch: usize) {
    match ch {
        1 => zune_imageprocs::transpose::transpose_u8(src, dst, w, h),
        2 => zune_imageprocs::transpose::transpose_u16(
            bytemuck::cast_slice(src),
            bytemuck::cast_slice_mut(dst),
            w,
            h,
        ),
        4 => zune_imageprocs::transpose::transpose_u32(
            bytemuck::cast_slice(src),
            bytemuck::cast_slice_mut(dst),
            w,
            h,
        ),
        _ => unreachable!("zune has no 3-channel path"),
    }
}

fn run_ft(
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    ch: usize,
    modes: (fast_transpose::FlipMode, fast_transpose::FlopMode),
) {
    let (flip, flop) = modes;
    let r = match ch {
        1 => fast_transpose::transpose_plane(src, w, dst, h, w, h, flip, flop),
        2 => fast_transpose::transpose_plane_with_alpha(src, w * 2, dst, h * 2, w, h, flip, flop),
        3 => fast_transpose::transpose_rgb(src, w * 3, dst, h * 3, w, h, flip, flop),
        4 => fast_transpose::transpose_rgba(src, w * 4, dst, h * 4, w, h, flip, flop),
        _ => unreachable!(),
    };
    r.unwrap();
}

#[cfg(feature = "cpp-simd")]
mod cpp {
    unsafe extern "C" {
        // src/Simd/SimdLib.h: SimdTransformImage(src, srcStride, width, height,
        // pixelSize, transform, dst, dstStride). size_t == usize, enum == c_int.
        pub fn SimdTransformImage(
            src: *const u8,
            src_stride: usize,
            width: usize,
            height: usize,
            pixel_size: usize,
            transform: core::ffi::c_int,
            dst: *mut u8,
            dst_stride: usize,
        );
    }

    pub fn run(src: &[u8], dst: &mut [u8], w: usize, h: usize, ch: usize, transform: i32) {
        assert!(src.len() >= w * h * ch && dst.len() >= w * h * ch);
        // SAFETY: buffers cover w*h*ch bytes at tight strides; transform value
        // was probe-verified against the oracle before any benching.
        unsafe {
            SimdTransformImage(src.as_ptr(), w * ch, w, h, ch, transform, dst.as_mut_ptr(), h * ch);
        }
    }
}

// ── probe-derived mappings ───────────────────────────────────────────────────

/// Find the fast_transpose (flip, flop) combo whose output matches `op`'s
/// oracle, per channel count (probed independently in case semantics differ).
fn derive_ft_modes(
    ch: usize,
    op: Op,
) -> Option<(fast_transpose::FlipMode, fast_transpose::FlopMode)> {
    use fast_transpose::{FlipMode, FlopMode};
    let (w, h) = (17usize, 13usize);
    let n = w * h * ch;
    let src = pattern(n);
    let want = oracle(src.bytes(n), w, h, ch, op);
    for flip in [FlipMode::NoFlip, FlipMode::Flip] {
        for flop in [FlopMode::NoFlop, FlopMode::Flop] {
            let mut out = vec![0u8; n];
            run_ft(src.bytes(n), &mut out, w, h, ch, (flip, flop));
            if out == want {
                return Some((flip, flop));
            }
        }
    }
    None
}

#[cfg(feature = "cpp-simd")]
fn derive_simd_transform(ch: usize, op: Op) -> Option<i32> {
    let (w, h) = (17usize, 13usize);
    let n = w * h * ch;
    let src = pattern(n);
    let want = oracle(src.bytes(n), w, h, ch, op);
    // SimdTransformType has 8 values (rotate 0/90/180/270 × {plain, transposed}).
    for t in 0..8 {
        let mut out = vec![0u8; n];
        cpp::run(src.bytes(n), &mut out, w, h, ch, t);
        if out == want {
            return Some(t);
        }
    }
    None
}

/// Verify a contestant on probe geometry + a tile-boundary-crossing geometry.
/// `out` is a Buf (aligned) because zune's 2/4ch entries cast the dst too.
fn verify(name: &str, ch: usize, op: Op, mut f: impl FnMut(&[u8], &mut [u8], usize, usize)) {
    for (w, h) in [(17usize, 13usize), (67, 43)] {
        let n = w * h * ch;
        let src = pattern(n);
        let want = oracle(src.bytes(n), w, h, ch, op);
        let mut out = Buf::new(n);
        f(src.bytes(n), out.bytes_mut(n), w, h);
        assert_eq!(
            out.bytes(n),
            &want[..],
            "{name} {ch}ch {op:?} {w}x{h}: output mismatch vs oracle — refusing to bench"
        );
    }
}

// ── suite ────────────────────────────────────────────────────────────────────

// (label, w, h)
const SIZES: &[(&str, usize, usize)] = &[
    (" 256sq", 256, 256),
    ("1024sq", 1024, 1024),
    ("2048sq", 2048, 2048),
    ("12MP  ", 4000, 3000),
];

fn main() {
    // Derive + verify everything before any timing. Panics abort the run.
    let mut ft_modes = std::collections::HashMap::new();
    for ch in [1usize, 2, 3, 4] {
        for op in [Op::Transpose, Op::Rotate90] {
            verify("zpc", ch, op, |s, d, w, h| run_zpc(s, d, w, h, ch, op));
            verify("zpc-st", ch, op, |s, d, w, h| run_zpc_staged(s, d, w, h, ch, op));
            if let Some(m) = derive_ft_modes(ch, op) {
                verify("ft", ch, op, |s, d, w, h| run_ft(s, d, w, h, ch, m));
                ft_modes.insert((ch, op), m);
                eprintln!("ft {ch}ch {op:?}: {m:?}");
            } else {
                eprintln!("ft {ch}ch {op:?}: NO flag combo matches oracle — excluded");
            }
            if op == Op::Transpose {
                verify("ejm", ch, op, |s, d, w, h| run_ejm(s, d, w, h, ch));
                if ch != 3 {
                    verify("zune", ch, op, |s, d, w, h| run_zune(s, d, w, h, ch));
                }
            }
        }
    }
    #[cfg(feature = "cpp-simd")]
    let mut simd_transform = std::collections::HashMap::new();
    #[cfg(feature = "cpp-simd")]
    for ch in [1usize, 2, 3, 4] {
        for op in [Op::Transpose, Op::Rotate90] {
            if let Some(t) = derive_simd_transform(ch, op) {
                verify("simd++", ch, op, |s, d, w, h| cpp::run(s, d, w, h, ch, t));
                simd_transform.insert((ch, op), t);
                eprintln!("simd++ {ch}ch {op:?}: SimdTransformType = {t}");
            } else {
                eprintln!("simd++ {ch}ch {op:?}: no transform matches oracle — excluded");
            }
        }
    }
    eprintln!("all verifications passed; benching");

    // zenbench::run() doesn't parse --group= itself (that lives in its main!
    // macro), so wire the substring filter here.
    let group_filter: Option<String> =
        std::env::args().find_map(|a| a.strip_prefix("--group=").map(String::from));

    zenbench::run(|suite| {
        if let Some(f) = group_filter.clone() {
            suite.set_group_filter(f);
        }
        for op in [Op::Transpose, Op::Rotate90] {
            for ch in [1usize, 2, 3, 4] {
                for &(label, w, h) in SIZES {
                    let bytes = w * h * ch;
                    let src_data = pattern(bytes);
                    let ft_m = ft_modes.get(&(ch, op)).copied();
                    #[cfg(feature = "cpp-simd")]
                    let simd_t = simd_transform.get(&(ch, op)).copied();
                    suite.group(format!("{} {ch}ch {label}", op.label()), move |g| {
                        g.throughput(Throughput::Bytes(bytes as u64));
                        // Fast interleaved settings: zenbench's paired round-robin makes
                        // few-round deltas trustworthy; the decisions here are 2-5x.
                        g.config()
                            .warmup_time(Duration::from_millis(100))
                            .max_time(Duration::from_millis(1500))
                            .max_rounds(8)
                            .target_precision(0.10)
                            .max_wall_time(Duration::from_secs(10));
                        // Each bench owns a src copy + pre-touched dst (zenbench
                        // closures are 'static; copies keep per-bench memory
                        // layout identical instead of sharing via Arc).
                        let src = src_data.clone();
                        g.bench("zpc   ", move |b| {
                            let mut dst = Buf::new(bytes);
                            dst.bytes_mut(bytes).fill(1); // pre-fault
                            b.iter(|| {
                                run_zpc(black_box(src.bytes(bytes)), dst.bytes_mut(bytes), w, h, ch, op);
                                black_box(dst.bytes(1));
                            })
                        });
                        let src = src_data.clone();
                        g.bench("zpc-st", move |b| {
                            let mut dst = Buf::new(bytes);
                            dst.bytes_mut(bytes).fill(1);
                            b.iter(|| {
                                run_zpc_staged(black_box(src.bytes(bytes)), dst.bytes_mut(bytes), w, h, ch, op);
                                black_box(dst.bytes(1));
                            })
                        });
                        if let Some(m) = ft_m {
                            let src = src_data.clone();
                            g.bench("ft    ", move |b| {
                                let mut dst = Buf::new(bytes);
                                dst.bytes_mut(bytes).fill(1);
                                b.iter(|| {
                                    run_ft(black_box(src.bytes(bytes)), dst.bytes_mut(bytes), w, h, ch, m);
                                    black_box(dst.bytes(1));
                                })
                            });
                        }
                        if op == Op::Transpose {
                            let src = src_data.clone();
                            g.bench("ejm   ", move |b| {
                                let mut dst = Buf::new(bytes);
                                dst.bytes_mut(bytes).fill(1);
                                b.iter(|| {
                                    run_ejm(black_box(src.bytes(bytes)), dst.bytes_mut(bytes), w, h, ch);
                                    black_box(dst.bytes(1));
                                })
                            });
                            if ch != 3 {
                                let src = src_data.clone();
                                g.bench("zune  ", move |b| {
                                    let mut dst = Buf::new(bytes);
                                    dst.bytes_mut(bytes).fill(1);
                                    b.iter(|| {
                                        run_zune(black_box(src.bytes(bytes)), dst.bytes_mut(bytes), w, h, ch);
                                        black_box(dst.bytes(1));
                                    })
                                });
                            }
                        }
                        #[cfg(feature = "cpp-simd")]
                        if let Some(t) = simd_t {
                            let src = src_data.clone();
                            g.bench("simd++", move |b| {
                                let mut dst = Buf::new(bytes);
                                dst.bytes_mut(bytes).fill(1);
                                b.iter(|| {
                                    cpp::run(black_box(src.bytes(bytes)), dst.bytes_mut(bytes), w, h, ch, t);
                                    black_box(dst.bytes(1));
                                })
                            });
                        }
                    });
                }
            }
        }
    });
}
