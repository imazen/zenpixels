//! Does a row start's alignment change kernel speed, per SIMD tier?
//!
//! Every magetypes backend loads and stores through unaligned instructions,
//! so alignment never affects correctness; this measures what it costs. The
//! kernel is a strided in-place multiply-add over F32 RGBA rows, the access
//! pattern of the conversion kernels. Each tier is called directly with its
//! own token (no dispatch), so v3 and v4 are both measured on an AVX-512 CPU.
//!
//! Rows start 0, 4, 16 or 32 bytes past a 64-byte boundary; the stride is a
//! multiple of 64 bytes, so every row has the same offset. Sizes: L1-resident,
//! L2-resident and a 1920x1080 frame; 20 fixed rounds per group.
//!
//! Run: `cargo bench -p zenpixels-convert --features avx512 --bench bench_row_alignment`

use archmage::prelude::*;
use zenbench::prelude::*;

#[magetypes(define(f32x16), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn madd_row(token: Token, row: &mut [f32], gain: f32, bias: f32) {
    let g = f32x16::splat_t(token, gain);
    let b = f32x16::splat_t(token, bias);
    let (chunks, tail) = f32x16::partition_slice_mut_t(token, row);
    for chunk in chunks {
        (f32x16::load_t(token, chunk) * g + b).store(chunk);
    }
    for v in tail {
        *v = *v * gain + bias;
    }
}

/// F32 RGBA image whose rows all start `shift_bytes` past a 64-byte boundary.
struct Image {
    data: Vec<f32>,
    start: usize,
    stride: usize,
    width: usize,
    rows: usize,
}

impl Image {
    fn new(width: usize, rows: usize, shift_bytes: usize) -> Self {
        assert_eq!(shift_bytes % 4, 0);
        let row_floats = width * 4;
        let stride = row_floats.div_ceil(16) * 16 + 16; // multiple of 64 bytes, not a power of two
        let mut data = vec![0.5f32; stride * rows + 32];
        let misalign = (data.as_ptr() as usize % 64) / 4;
        let start = (16 - misalign) % 16 + shift_bytes / 4;
        assert_eq!((data.as_ptr() as usize + start * 4) % 64, shift_bytes);
        for y in 0..rows {
            for x in 0..row_floats {
                data[start + y * stride + x] = ((y * row_floats + x) % 1000) as f32 * 0.001;
            }
        }
        Self {
            data,
            start,
            stride,
            width,
            rows,
        }
    }

    fn run(&mut self, kernel: &impl Fn(&mut [f32])) {
        let n = self.width * 4;
        for y in 0..self.rows {
            let s = self.start + y * self.stride;
            kernel(&mut self.data[s..s + n]);
        }
    }
}

type RowKernel = Box<dyn Fn(&mut [f32]) + Send + Sync>;

fn tiers() -> Vec<(&'static str, RowKernel)> {
    let mut out: Vec<(&'static str, RowKernel)> = Vec::new();
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4Token::summon() {
        out.push((
            "v4 (AVX-512, 64-byte vectors)",
            Box::new(move |r| madd_row_v4(t, r, 0.9999, 1e-4)),
        ));
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        out.push((
            "v3 (AVX2, 32-byte vectors)",
            Box::new(move |r| madd_row_v3(t, r, 0.9999, 1e-4)),
        ));
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        out.push((
            "neon (16-byte vectors)",
            Box::new(move |r| madd_row_neon(t, r, 0.9999, 1e-4)),
        ));
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = Wasm128Token::summon() {
        out.push((
            "wasm128 (16-byte vectors)",
            Box::new(move |r| madd_row_wasm128(t, r, 0.9999, 1e-4)),
        ));
    }
    let t = ScalarToken::summon().unwrap();
    out.push((
        "scalar",
        Box::new(move |r| madd_row_scalar(t, r, 0.9999, 1e-4)),
    ));
    out
}

fn main() {
    // Same results at every offset and tier (single rounding per element, no
    // FMA contraction difference: `a * g + b` is two roundings on every tier).
    for (name, k) in tiers() {
        let mut want = Image::new(67, 3, 0);
        want.run(&k);
        for shift in [4, 16, 32] {
            let mut got = Image::new(67, 3, shift);
            got.run(&k);
            for y in 0..3 {
                let (a, b) = (want.start + y * want.stride, got.start + y * got.stride);
                assert_eq!(
                    want.data[a..a + 67 * 4],
                    got.data[b..b + 67 * 4],
                    "{name}: offset {shift} changed results"
                );
            }
        }
    }

    let sizes: &[(&str, usize, usize)] = &[
        ("L1: 64x8 px RGBA F32, 8 KB", 64, 8),
        ("L2: 256x64 px RGBA F32, 256 KB", 256, 64),
        ("memory: 1920x1080 px RGBA F32, 32 MB", 1920, 1080),
    ];

    zenbench::run(|suite| {
        for (tier, kernel) in tiers() {
            let kernel = std::sync::Arc::new(kernel);
            for &(label, w, h) in sizes {
                suite.group(format!("{tier} | {label}"), |g| {
                    g.throughput(Throughput::Bytes((w * h * 16) as u64));
                    g.config().auto_rounds(false).min_rounds(20).max_rounds(20);
                    for shift in [0usize, 4, 16, 32] {
                        let mut img = Image::new(w, h, shift);
                        let k = kernel.clone();
                        g.bench(format!("row start +{shift:>2} B"), move |bench| {
                            bench.iter(|| {
                                img.run(&*k);
                                black_box(img.data[img.start]);
                            })
                        });
                    }
                });
            }
        }
    });
}
