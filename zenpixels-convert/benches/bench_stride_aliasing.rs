//! Does a power-of-two row stride slow kernels that read one column across
//! several rows, and does adding 64 bytes to the stride fix it?
//!
//! Two experiments, each tier called directly with its own token:
//!
//! 1. Vertical K-tap filter (K = 2, 8, 12) over F32 rows. For each output row
//!    `y` and each 16-float column block, it reads that block from rows
//!    `y..y+K` (so K input rows are live together) and writes a tightly packed
//!    output row. Strides 1024, 2048, ... 32768 bytes are each compared with
//!    the same row width at stride + 64 bytes. The baseline arm of every group
//!    is the +64 stride, so zenbench's percent change is the power-of-two
//!    slowdown. Two image heights: L2-resident, and well past L3.
//! 2. Row in to row out (4K aliasing): `dst[y] = src[y] * g + b`, where `dst`
//!    lives in the same allocation as `src` at a byte distance that is a
//!    multiple of 4096 plus a chosen offset. Baseline is distance = multiple of
//!    4096 + 64.
//!
//! Before timing, every tier and every stride is checked for bit-identical
//! output against a plain scalar reference.
//!
//! Run: `cargo bench -p zenpixels-convert --features avx512 --bench bench_stride_aliasing -- --no-busy-gate`
//! `--group=NAME` filters groups (substring of the group name).

use archmage::prelude::*;
use zenbench::prelude::*;

const BLOCK: usize = 16; // floats per vector block (f32x16)

#[magetypes(define(f32x16), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn vfilter<const K: usize>(
    token: Token,
    src: &[f32],
    stride: usize,
    w: usize,
    taps: &[f32; K],
    dst: &mut [f32],
) {
    let tv: [f32x16; K] = core::array::from_fn(|k| f32x16::splat_t(token, taps[k]));
    for (y, out) in dst.chunks_exact_mut(w).enumerate() {
        let rows: [&[[f32; BLOCK]]; K] =
            core::array::from_fn(|k| src[(y + k) * stride..][..w].as_chunks::<BLOCK>().0);
        for (bi, o) in out.as_chunks_mut::<BLOCK>().0.iter_mut().enumerate() {
            let mut acc = f32x16::load_t(token, &rows[0][bi]) * tv[0];
            for (row, t) in rows.iter().zip(&tv).skip(1) {
                acc += f32x16::load_t(token, &row[bi]) * *t;
            }
            acc.store(o);
        }
    }
}

#[magetypes(define(f32x16), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn madd_row(token: Token, src: &[f32], dst: &mut [f32], gain: f32, bias: f32) {
    let g = f32x16::splat_t(token, gain);
    let b = f32x16::splat_t(token, bias);
    for (s, d) in src
        .as_chunks::<BLOCK>()
        .0
        .iter()
        .zip(dst.as_chunks_mut::<BLOCK>().0.iter_mut())
    {
        (f32x16::load_t(token, s) * g + b).store(d);
    }
}

type FilterKernel = Box<dyn Fn(&[f32], usize, usize, &mut [f32]) + Send + Sync>;
type RowKernel = Box<dyn Fn(&[f32], &mut [f32]) + Send + Sync>;

const GAIN: f32 = 0.9999;
const BIAS: f32 = 1e-4;

fn taps<const K: usize>() -> [f32; K] {
    core::array::from_fn(|k| ((k * 37 + 11) % 64) as f32 / 64.0)
}

/// One tier's kernels for the three tap counts plus the row kernel.
struct Tier {
    name: &'static str,
    k2: FilterKernel,
    k8: FilterKernel,
    k12: FilterKernel,
    row: RowKernel,
}

macro_rules! tier {
    ($name:expr, $t:expr, $vf:ident, $mr:ident) => {{
        let t = $t;
        Tier {
            name: $name,
            k2: Box::new(move |s, st, w, d| $vf::<2>(t, s, st, w, &taps::<2>(), d)),
            k8: Box::new(move |s, st, w, d| $vf::<8>(t, s, st, w, &taps::<8>(), d)),
            k12: Box::new(move |s, st, w, d| $vf::<12>(t, s, st, w, &taps::<12>(), d)),
            row: Box::new(move |s, d| $mr(t, s, d, GAIN, BIAS)),
        }
    }};
}

fn tiers() -> Vec<Tier> {
    let mut out = Vec::new();
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(t) = X64V4Token::summon() {
        out.push(tier!(
            "v4 (AVX-512, 64-byte vectors)",
            t,
            vfilter_v4,
            madd_row_v4
        ));
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        out.push(tier!(
            "v3 (AVX2, 32-byte vectors)",
            t,
            vfilter_v3,
            madd_row_v3
        ));
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        out.push(tier!(
            "neon (16-byte vectors)",
            t,
            vfilter_neon,
            madd_row_neon
        ));
    }
    #[cfg(target_arch = "wasm32")]
    if let Some(t) = Wasm128Token::summon() {
        out.push(tier!(
            "wasm128 (16-byte vectors)",
            t,
            vfilter_wasm128,
            madd_row_wasm128
        ));
    }
    let t = ScalarToken::summon().unwrap();
    out.push(tier!("scalar", t, vfilter_scalar, madd_row_scalar));
    out
}

/// Value of input element (y, x); the same for every stride and tier.
fn pixel(y: usize, x: usize) -> f32 {
    ((y * 131 + x * 7) % 1009) as f32 * 0.003 - 1.0
}

/// A zero-filled `Vec<f32>` plus the index of its first 64-byte-aligned float.
fn aligned_vec(floats: usize) -> (Vec<f32>, usize) {
    let v = vec![0.0f32; floats + 16];
    let start = (16 - (v.as_ptr() as usize % 64) / 4) % 16;
    assert_eq!((v.as_ptr() as usize + start * 4) % 64, 0);
    (v, start)
}

/// Input image: `rows` rows of `w` floats at a stride of `stride_bytes`,
/// every row starting 64-byte aligned.
struct Plane {
    data: Vec<f32>,
    start: usize,
    stride: usize, // floats
    w: usize,
}

impl Plane {
    fn new(w: usize, stride_bytes: usize, rows: usize) -> Self {
        assert_eq!(stride_bytes % 64, 0);
        assert!(w * 4 <= stride_bytes && w.is_multiple_of(BLOCK));
        let stride = stride_bytes / 4;
        let (mut data, start) = aligned_vec(stride * rows);
        for y in 0..rows {
            for x in 0..w {
                data[start + y * stride + x] = pixel(y, x);
            }
        }
        Self {
            data,
            start,
            stride,
            w,
        }
    }
    fn src(&self) -> &[f32] {
        &self.data[self.start..]
    }
}

/// Output buffer for a vertical filter: `rows - K + 1` tightly packed rows of
/// `w` floats, 64-byte aligned.
struct Out {
    data: Vec<f32>,
    start: usize,
    len: usize,
}

impl Out {
    fn new(w: usize, rows: usize) -> Self {
        let len = w * rows;
        let (data, start) = aligned_vec(len);
        Self { data, start, len }
    }
    fn slice(&mut self) -> &mut [f32] {
        &mut self.data[self.start..self.start + self.len]
    }
}

fn reference_vfilter<const K: usize>(w: usize, rows_out: usize) -> Vec<f32> {
    let tp = taps::<K>();
    let mut out = vec![0.0f32; w * rows_out];
    for y in 0..rows_out {
        for x in 0..w {
            let mut acc = pixel(y, x) * tp[0];
            for (k, t) in tp.iter().enumerate().skip(1) {
                acc += pixel(y + k, x) * t;
            }
            out[y * w + x] = acc;
        }
    }
    out
}

/// Strides in bytes: each power of two and its +64 partner.
const PAIRS: [usize; 6] = [1024, 2048, 4096, 8192, 16384, 32768];

fn check_vfilter(tiers: &[Tier]) {
    for tier in tiers {
        for (k, kern) in [(2usize, &tier.k2), (8, &tier.k8), (12, &tier.k12)] {
            for &p in &PAIRS {
                let w = p / 4;
                let rows_out = 11;
                let want = match k {
                    2 => reference_vfilter::<2>(w, rows_out),
                    8 => reference_vfilter::<8>(w, rows_out),
                    _ => reference_vfilter::<12>(w, rows_out),
                };
                for stride in [p, p + 64] {
                    let img = Plane::new(w, stride, rows_out + k - 1);
                    let mut out = Out::new(w, rows_out);
                    kern(img.src(), img.stride, w, out.slice());
                    assert!(
                        out.slice()
                            .iter()
                            .zip(&want)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{}: K={k} stride {stride} differs from the scalar reference",
                        tier.name
                    );
                }
            }
        }
    }
}

/// Source and destination rows in one allocation: `dst` starts `dist` bytes
/// after `src`.
struct RowPair {
    data: Vec<f32>,
    start: usize,
    stride: usize, // floats, both images
    w: usize,
    rows: usize,
    dst_off: usize, // floats from src start
}

impl RowPair {
    fn new(w: usize, stride_bytes: usize, rows: usize, dist_bytes: usize) -> Self {
        assert_eq!(dist_bytes % 64, 0);
        let stride = stride_bytes / 4;
        assert!(dist_bytes / 4 >= stride * rows, "images overlap");
        let dst_off = dist_bytes / 4;
        let (mut data, start) = aligned_vec(dst_off + stride * rows);
        for y in 0..rows {
            for x in 0..w {
                data[start + y * stride + x] = pixel(y, x);
            }
        }
        Self {
            data,
            start,
            stride,
            w,
            rows,
            dst_off,
        }
    }

    fn run(&mut self, kernel: &RowKernel) {
        let (a, b) = self.data[self.start..].split_at_mut(self.dst_off);
        for y in 0..self.rows {
            let s = y * self.stride;
            kernel(&a[s..s + self.w], &mut b[s..s + self.w]);
        }
    }

    fn dst_row(&self, y: usize) -> &[f32] {
        let s = self.start + self.dst_off + y * self.stride;
        &self.data[s..s + self.w]
    }
}

/// Smallest multiple of 4096 that holds `rows` rows of `stride_bytes`.
fn base_dist(stride_bytes: usize, rows: usize) -> usize {
    (stride_bytes * rows).div_ceil(4096) * 4096
}

const ROW_STRIDE: usize = 4160; // 1024 floats of data, not a power of two
const ROW_W: usize = 1024;
/// dst - src (mod 4096) for the row experiment; the first is the baseline.
const DISTS: [(&str, usize); 4] = [
    ("dst-src = 4096n + 64", 64),
    ("dst-src = 4096n", 0),
    ("dst-src = 4096n - 64", 4096 - 64),
    ("dst-src = 4096n + 2048", 2048),
];

fn check_rows(tiers: &[Tier]) {
    let rows = 5;
    for tier in tiers {
        for (_, d) in DISTS {
            let mut p = RowPair::new(ROW_W, ROW_STRIDE, rows, base_dist(ROW_STRIDE, rows) + d);
            p.run(&tier.row);
            for y in 0..rows {
                for x in 0..ROW_W {
                    let want = pixel(y, x) * GAIN + BIAS;
                    assert_eq!(
                        p.dst_row(y)[x].to_bits(),
                        want.to_bits(),
                        "{}: row kernel, dst-src +{d}",
                        tier.name
                    );
                }
            }
        }
    }
}

/// Images of the group being measured. Every group registers up front, but an
/// arm builds its images on first use and drops those of earlier groups, so
/// only one group's memory is live at a time.
enum State {
    Filter(Plane, Out),
    Rows(RowPair),
}

static LIVE: std::sync::Mutex<Vec<((usize, usize), State)>> = std::sync::Mutex::new(Vec::new());

#[derive(Clone, Copy)]
enum Kernel {
    Filter(&'static FilterKernel),
    Row(&'static RowKernel),
}

/// Registers one arm of group `gid`; `make` builds its images when first run.
fn arm(
    g: &mut BenchGroup,
    gid: usize,
    idx: usize,
    name: String,
    kernel: Kernel,
    make: impl Fn() -> State + Send + 'static,
) {
    g.bench(name, move |bench| {
        let mut live = LIVE.lock().unwrap();
        live.retain(|((g, _), _)| *g == gid);
        let pos = match live.iter().position(|(k, _)| *k == (gid, idx)) {
            Some(p) => p,
            None => {
                live.push(((gid, idx), make()));
                live.len() - 1
            }
        };
        let state = &mut live[pos].1;
        match (state, kernel) {
            (State::Filter(img, out), Kernel::Filter(kern)) => bench.iter(|| {
                kern(img.src(), img.stride, img.w, out.slice());
                black_box(out.data[out.start]);
            }),
            (State::Rows(pair), Kernel::Row(kern)) => bench.iter(|| {
                pair.run(kern);
                black_box(pair.data[pair.start + pair.dst_off]);
            }),
            _ => unreachable!(),
        }
    });
}

fn bench_filter(
    suite: &mut Suite,
    gid: &mut usize,
    tier: &'static Tier,
    k: usize,
    kern: &'static FilterKernel,
) {
    // Heights: L2 keeps about 128 KiB of input (at least K + 4 rows); memory
    // keeps 128 MiB, past both CCDs' L3 on the x86 host.
    for (size_label, l2) in [("L2", true), ("memory", false)] {
        for &p in &PAIRS {
            let w = p / 4;
            let in_rows = if l2 {
                (128 * 1024 / p).max(k + 4)
            } else {
                128 * 1024 * 1024 / p
            };
            let rows_out = in_rows - k + 1;
            let in_kib = in_rows * p / 1024;
            let label = format!(
                "{} | K={k} | row {w} floats | {size_label}: {in_rows} rows, {in_kib} KiB in",
                tier.name
            );
            *gid += 1;
            let gid = *gid;
            suite.group(label, |g| {
                g.throughput(Throughput::Bytes((rows_out * w * 4) as u64));
                g.config().auto_rounds(false).min_rounds(20).max_rounds(20);
                for (idx, stride) in [p + 64, p].into_iter().enumerate() {
                    let tag = if stride == p { "pow2" } else { "+64" };
                    arm(
                        g,
                        gid,
                        idx,
                        format!("stride {stride:>5} B ({tag})"),
                        Kernel::Filter(kern),
                        move || {
                            State::Filter(Plane::new(w, stride, in_rows), Out::new(w, rows_out))
                        },
                    );
                }
            });
        }
    }
}

fn bench_rows(suite: &mut Suite, gid: &mut usize, tier: &'static Tier) {
    for (size_label, rows) in [
        ("L1: 2 rows", 2usize),
        ("L2: 32 rows", 32),
        ("memory: 16384 rows", 16384),
    ] {
        let label = format!("{} | row-in row-out | {size_label}", tier.name);
        *gid += 1;
        let gid = *gid;
        suite.group(label, |g| {
            g.throughput(Throughput::Bytes((rows * ROW_W * 4) as u64));
            g.config().auto_rounds(false).min_rounds(20).max_rounds(20);
            for (idx, (name, d)) in DISTS.into_iter().enumerate() {
                arm(
                    g,
                    gid,
                    idx,
                    name.to_string(),
                    Kernel::Row(&tier.row),
                    move || {
                        State::Rows(RowPair::new(
                            ROW_W,
                            ROW_STRIDE,
                            rows,
                            base_dist(ROW_STRIDE, rows) + d,
                        ))
                    },
                );
            }
        });
    }
}

fn main() {
    let t = tiers();
    check_vfilter(&t);
    check_rows(&t);
    eprintln!("correctness: all tiers and strides bit-identical to the scalar reference");

    let t: &'static [Tier] = Box::leak(t.into_boxed_slice());
    zenbench::run(|suite| {
        let mut gid = 0;
        for tier in t {
            bench_rows(suite, &mut gid, tier);
            for (k, kern) in [(2usize, &tier.k2), (8, &tier.k8), (12, &tier.k12)] {
                bench_filter(suite, &mut gid, tier, k, kern);
            }
        }
    });
}
