//! End-to-end producer-SDR match for `zenpixels_convert::hdr::HdrToSdr`.
//!
//! Decodes ONE known imazen-26 sample twice — once as the SDR base (the
//! producer's reference) and once as HDR (full-headroom gain-map
//! reconstruction) — then pushes the HDR through `HdrToSdr` and compares
//! against the SDR ground truth via mean ΔE2000 (CIEDE2000) in CIE Lab.
//!
//! Pin: mean ΔE2000 < 5.0 on the BT.2446-A pipeline. The OLD buggy order
//! (BT.2020 → BT.709 matrix BEFORE the BT.2446-A curve, which uses BT.2020
//! luma weights) would push ΔE well above 5 on saturated content; the new
//! order should land near zentone's published BT.2446-A median of
//! ΔE2000 = 3.17 (see
//! `~/work/zen/zentone/benchmarks/hdr_tone_map_shootout_full_2026-06-20.md`).
//!
//! ## Decode path
//!
//! Uses the local `zenjpeg` (path dep at `../../zenjpeg/zenjpeg`) for the
//! one-call `decode_ultrahdr` (SDR base) + `decode_ultrahdr_hdr` (HDR
//! reconstruction) helpers. The workspace's `[patch.crates-io]` deliberately
//! omits a `zenpixels-convert` entry — that's what would create the
//! `zenpixels-convert → zenjpeg → zenanalyze → zenpixels-convert` cycle.
//! Instead the two-version graph (local 0.2.15 for the workspace member +
//! registry 0.2.14 for `zenanalyze`'s transitive need) coexists harmlessly,
//! and `cargo` requires `-p zenpixels-convert@0.2.15` to disambiguate the
//! test build.
//!
//! ## Gating
//!
//! Compiled only under the `__hdr-e2e-test` feature, which pulls
//! `zenjpeg` (with `decoder` + `ultrahdr`) and `anyhow` as test-only
//! deps. Runs a runtime presence check on the sample at
//! `/home/lilith/work/codec-corpus/imazen-26/...` — if absent (CI
//! runners, etc.) the test prints a `SKIP` line and exits without
//! panicking. **MUST be run on `lilith`'s machine before merging any
//! pipeline-touching change** (the corpus exists there).

#![cfg(feature = "__hdr-e2e-test")]

use std::path::Path;

use zenjpeg::ultrahdr::{HdrOutputFormat, decode_ultrahdr, decode_ultrahdr_hdr};
use zenpixels::{ChannelLayout, ChannelType, PixelBuffer, PixelDescriptor, TransferFunction};
use zenpixels_convert::PixelBufferConvertExt;
use zenpixels_convert::hdr::HdrToSdr;

/// Smallest known gain-mapped sample in imazen-26 (~4000x3000 JPEG, single
/// take, gain-map carries ~1000-nit HDR + sRGB BT.709 SDR base). Picked
/// for speed: still ~12 MP of decode work but the file is ~6 MB.
const SAMPLE_PATH: &str = "/home/lilith/work/codec-corpus/imazen-26/\
    1000-lilith-photos-general/\
    1064_general_castle-bridge-moat_montjuic-castle-barcelona_zfold7\
    _iso40-f1p7_20260315-175347_4000x3000.jpg";

/// Pin: the new (correct) pipeline must land mean ΔE2000 < this against
/// producer SDR. The OLD buggy pipeline pushed ΔE into 10+ territory on
/// saturated content; 5 is generous enough to absorb cell-to-cell
/// variance while still catching an order regression.
const DE_BUDGET_MEAN: f32 = 5.0;

/// Source peak for the test sample (it's a BT.2100 PQ gain-map carrying
/// ~1000-nit content). The shootout measures per-sample peak via
/// `ContentLightLevel`; we use a fixed value here to keep the test
/// self-contained (no zenpixels-convert hdr-measurement dep). The exact
/// peak value matters at the few-ΔE level but not at the 5-ΔE order-bug
/// detection level.
const SOURCE_PEAK_NITS: f32 = 1000.0;

/// Display-boost passed to `apply_gainmap`. `8.0+` produces near-full
/// gain-map reconstruction at PQ peak per the ultrahdr-core docs.
const DISPLAY_BOOST: f32 = 8.0;

#[test]
fn producer_sdr_match_on_a_real_imazen26_sample() {
    if !Path::new(SAMPLE_PATH).exists() {
        eprintln!(
            "SKIP: imazen-26 sample not present at {} — must run on `lilith`'s machine.",
            SAMPLE_PATH
        );
        return;
    }
    let bytes = std::fs::read(SAMPLE_PATH).expect("read sample bytes");

    // ---- SDR base — producer's ground truth (one-call helper).
    let sdr_buf = decode_ultrahdr(&bytes).expect("decode SDR base");
    let sdr_rgb = pixel_buffer_to_linear_rgb(&sdr_buf).expect("linearize SDR");

    // ---- HDR via codec's native gain-map reconstruction (full headroom).
    // `LinearFloat` = RGBA F32 linear-light in the SDR base's primaries
    // (typically BT.709), scale 1.0 = SDR diffuse white.
    let hdr_buf = decode_ultrahdr_hdr(&bytes, DISPLAY_BOOST, HdrOutputFormat::LinearFloat)
        .expect("decode HDR via gain-map");
    let hdr_src_primaries = hdr_buf.descriptor().primaries;
    let hdr_rgb = pixel_buffer_to_linear_rgb(&hdr_buf).expect("linearize HDR");

    assert_eq!(
        (sdr_rgb.width, sdr_rgb.height),
        (hdr_rgb.width, hdr_rgb.height),
        "SDR / HDR dimensions diverged"
    );

    // ---- Rescale HDR to source-peak-normalized (1.0 = SOURCE_PEAK_NITS).
    // The gain-map output uses 1.0 = SDR diffuse white = ~203 nits per
    // ultrahdr-core; HdrToSdr's source convention is 1.0 = source peak.
    let diffuse_white_nits = 203.0_f32;
    let max_pixel_value = (SOURCE_PEAK_NITS / diffuse_white_nits).max(1.0);
    let content_norm_scale = 1.0_f32 / max_pixel_value;

    let scratch: Vec<f32> = hdr_rgb.px.iter().map(|&v| v * content_norm_scale).collect();

    // ---- Build a PixelBuffer wrapping the source-normalized HDR data,
    // then drive the buffer-level dispatch (`convert_buffer`) for the
    // full HDR→SDR path.
    let source_desc = PixelDescriptor::new_full(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
        hdr_src_primaries,
    );
    let target_desc = PixelDescriptor::RGBF32_LINEAR;
    let scratch_bytes: Vec<u8> = bytemuck::cast_slice(&scratch).to_vec();
    let src_buf = PixelBuffer::from_vec(scratch_bytes, hdr_rgb.width, hdr_rgb.height, source_desc)
        .expect("from_vec src");

    let converter = HdrToSdr::new(source_desc, target_desc, SOURCE_PEAK_NITS);
    let out_buf = converter.convert_buffer(&src_buf).expect("convert_buffer");

    // ---- Materialize the converted buffer back to a tight RGB f32 vec.
    let candidate_rgb = pixel_buffer_to_linear_rgb(&out_buf).expect("linearize candidate");
    let candidate = LinearRgb {
        width: candidate_rgb.width,
        height: candidate_rgb.height,
        px: candidate_rgb.px,
    };

    // ---- Compute mean ΔE2000 against the producer SDR.
    let (sum_de, max_de) = compute_de2000(&sdr_rgb, &candidate);
    let n = sdr_rgb.pixels() as f64;
    let mean_de = (sum_de / n) as f32;

    eprintln!(
        "[hdr_producer_sdr_match] sample dims={}x{}, n={}, mean ΔE2000 = {:.4}, max ΔE2000 = {:.4}, source_primaries = {:?}",
        sdr_rgb.width, sdr_rgb.height, n as u64, mean_de, max_de, hdr_src_primaries,
    );

    assert!(
        mean_de < DE_BUDGET_MEAN,
        "mean ΔE2000 = {} exceeds budget {} — pipeline regression likely",
        mean_de,
        DE_BUDGET_MEAN
    );
}

// ============================================================================
// Helpers — LinearRgb buffer + decode-to-linear + ΔE2000
// ============================================================================

struct LinearRgb {
    width: u32,
    height: u32,
    px: Vec<f32>, // tightly packed, n_pixels * 3
}

impl LinearRgb {
    fn pixels(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }
}

/// Convert a PixelBuffer (any layout / transfer / channel type) to a
/// tightly packed linear-RGB f32 buffer using the buffer's stored
/// primaries. (Primaries are preserved on the way out; the candidate
/// produced by HdrToSdr lands in BT.709 and the imazen-26 sample's SDR
/// base is also BT.709, so the ΔE2000 comparison is apples-to-apples.)
fn pixel_buffer_to_linear_rgb(buf: &PixelBuffer) -> anyhow::Result<LinearRgb> {
    let src_desc = buf.descriptor();
    let target = PixelDescriptor::new_full(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
        src_desc.primaries,
    );

    let linear = if src_desc == target {
        copy_buffer_tight(buf)?
    } else {
        buf.convert_to(target)
            .map_err(|e| anyhow::anyhow!("convert_to linear RGB f32: {:?}", e.error()))?
    };

    let width = linear.width();
    let height = linear.height();
    let n_pix = width as usize * height as usize;
    let mut tight = vec![0.0f32; n_pix * 3];

    let slice = linear.as_slice();
    let stride = slice.stride();
    let bytes = slice.as_strided_bytes();
    let row_bytes = width as usize * 3 * 4;

    for y in 0..height as usize {
        let src = &bytes[y * stride..y * stride + row_bytes];
        let dst = &mut tight[y * width as usize * 3..(y + 1) * width as usize * 3];
        let src_f32: &[f32] = bytemuck::cast_slice(src);
        dst.copy_from_slice(src_f32);
    }

    Ok(LinearRgb {
        width,
        height,
        px: tight,
    })
}

fn copy_buffer_tight(buf: &PixelBuffer) -> anyhow::Result<PixelBuffer> {
    let desc = buf.descriptor();
    let width = buf.width();
    let height = buf.height();

    let row_bytes = width as usize * desc.channels() * desc.channel_type().byte_size();
    let total = row_bytes * height as usize;
    let mut tight = vec![0u8; total];

    let slice = buf.as_slice();
    let stride = slice.stride();
    let bytes = slice.as_strided_bytes();

    for y in 0..height as usize {
        tight[y * row_bytes..(y + 1) * row_bytes]
            .copy_from_slice(&bytes[y * stride..y * stride + row_bytes]);
    }

    PixelBuffer::from_vec(tight, width, height, desc)
        .map_err(|e| anyhow::anyhow!("from_vec tight: {:?}", e))
}

// ---------------- ΔE2000 (sequential, but fast enough for one sample) -------

fn compute_de2000(reference: &LinearRgb, candidate: &LinearRgb) -> (f64, f32) {
    let n_px = reference.pixels();
    debug_assert_eq!(n_px, candidate.pixels());
    let mut sum = 0.0_f64;
    let mut max = 0.0_f32;
    for i in 0..n_px {
        let r_lab = linear_rgb_to_lab([
            reference.px[i * 3],
            reference.px[i * 3 + 1],
            reference.px[i * 3 + 2],
        ]);
        let c_lab = linear_rgb_to_lab([
            candidate.px[i * 3],
            candidate.px[i * 3 + 1],
            candidate.px[i * 3 + 2],
        ]);
        let de = delta_e2000(r_lab, c_lab);
        sum += de as f64;
        if de > max {
            max = de;
        }
    }
    (sum, max)
}

fn linear_rgb_to_lab(rgb: [f32; 3]) -> [f32; 3] {
    let r = rgb[0].clamp(0.0, 1.0) as f64;
    let g = rgb[1].clamp(0.0, 1.0) as f64;
    let b = rgb[2].clamp(0.0, 1.0) as f64;

    // Linear BT.709 / sRGB → XYZ (D65).
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;

    let xn = 0.95047_f64;
    let yn = 1.0_f64;
    let zn = 1.08883_f64;

    fn f(t: f64) -> f64 {
        const DELTA: f64 = 6.0 / 29.0;
        if t > DELTA * DELTA * DELTA {
            t.cbrt()
        } else {
            t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
        }
    }

    let fx = f(x / xn);
    let fy = f(y / yn);
    let fz = f(z / zn);

    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let bb = 200.0 * (fy - fz);

    [l as f32, a as f32, bb as f32]
}

fn delta_e2000(lab1: [f32; 3], lab2: [f32; 3]) -> f32 {
    let (l1, a1, b1) = (lab1[0] as f64, lab1[1] as f64, lab1[2] as f64);
    let (l2, a2, b2) = (lab2[0] as f64, lab2[1] as f64, lab2[2] as f64);

    let kl = 1.0_f64;
    let kc = 1.0_f64;
    let kh = 1.0_f64;

    let c1 = (a1 * a1 + b1 * b1).sqrt();
    let c2 = (a2 * a2 + b2 * b2).sqrt();
    let c_bar = (c1 + c2) / 2.0;

    let c_bar7 = c_bar.powi(7);
    let g = 0.5 * (1.0 - (c_bar7 / (c_bar7 + 25.0_f64.powi(7))).sqrt());

    let a1p = (1.0 + g) * a1;
    let a2p = (1.0 + g) * a2;

    let c1p = (a1p * a1p + b1 * b1).sqrt();
    let c2p = (a2p * a2p + b2 * b2).sqrt();

    let h1p = if b1 == 0.0 && a1p == 0.0 {
        0.0
    } else {
        b1.atan2(a1p).to_degrees().rem_euclid(360.0)
    };
    let h2p = if b2 == 0.0 && a2p == 0.0 {
        0.0
    } else {
        b2.atan2(a2p).to_degrees().rem_euclid(360.0)
    };

    let dl_p = l2 - l1;
    let dc_p = c2p - c1p;

    let dhp = if c1p * c2p == 0.0 {
        0.0
    } else if (h2p - h1p).abs() <= 180.0 {
        h2p - h1p
    } else if h2p - h1p > 180.0 {
        h2p - h1p - 360.0
    } else {
        h2p - h1p + 360.0
    };
    let dh_p = 2.0 * (c1p * c2p).sqrt() * (dhp.to_radians() / 2.0).sin();

    let l_bar_p = (l1 + l2) / 2.0;
    let c_bar_p = (c1p + c2p) / 2.0;

    let h_bar_p = if c1p * c2p == 0.0 {
        h1p + h2p
    } else if (h1p - h2p).abs() <= 180.0 {
        (h1p + h2p) / 2.0
    } else if h1p + h2p < 360.0 {
        (h1p + h2p + 360.0) / 2.0
    } else {
        (h1p + h2p - 360.0) / 2.0
    };

    let t = 1.0 - 0.17 * ((h_bar_p - 30.0).to_radians()).cos()
        + 0.24 * (2.0 * h_bar_p.to_radians()).cos()
        + 0.32 * ((3.0 * h_bar_p + 6.0).to_radians()).cos()
        - 0.20 * ((4.0 * h_bar_p - 63.0).to_radians()).cos();

    let delta_theta = 30.0 * (-(((h_bar_p - 275.0) / 25.0).powi(2))).exp();
    let c_bar_p7 = c_bar_p.powi(7);
    let rc = 2.0 * (c_bar_p7 / (c_bar_p7 + 25.0_f64.powi(7))).sqrt();
    let sl = 1.0 + (0.015 * (l_bar_p - 50.0).powi(2)) / (20.0 + (l_bar_p - 50.0).powi(2)).sqrt();
    let sc = 1.0 + 0.045 * c_bar_p;
    let sh = 1.0 + 0.015 * c_bar_p * t;
    let rt = -((2.0 * delta_theta.to_radians()).sin()) * rc;

    let term_l = dl_p / (kl * sl);
    let term_c = dc_p / (kc * sc);
    let term_h = dh_p / (kh * sh);

    let de_squared = term_l * term_l + term_c * term_c + term_h * term_h + rt * term_c * term_h;
    de_squared.max(0.0).sqrt() as f32
}
