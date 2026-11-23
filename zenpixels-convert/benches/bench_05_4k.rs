//! 4K UHD (3840x2160) benchmark suite for `zenpixels-convert` 0.5 greenfield engine.
//!
//! Compares `Accelerators::ALL` (AVX2/SIMD) vs `Accelerators::SCALAR` across all
//! core pipeline families at full 4K UHD resolution (`8,294,400` px/frame), matching
//! the frame size and byte-throughput convention of `benchmarks/baseline_4k_2026-11-21.md`
//! for direct 0.5 vs 0.4 comparison.
//!
//! Run:
//!   cargo bench -p zenpixels-convert --all-features --bench bench_05_4k -- --no-busy-gate

use std::sync::{Arc, Mutex};
use std::time::Duration;
use zenbench::prelude::*;
use zenpixels_convert::{
    Accelerators, AlphaMode, AlphaPolicy, BitPacking, ChannelOrder, Color, ColorSpace,
    ConvertPolicy, DepthPolicy, DiffuseWhite, GamutPolicy, HlgMode, HlgRendering, ImageSpec,
    LumaCoefficients, Nits, Orientation, PerceptualSpace, PixelBuf, Primaries, RowPlan,
    SampleLayout, SampleType, SignalRange, Transfer, apply_orientation_into,
};

const W: u32 = 3840;
const H: u32 = 2160;
const PIXELS: usize = (W as usize) * (H as usize);

struct FrameScratch {
    src: Vec<u8>,
    dst: Vec<u8>,
}

impl FrameScratch {
    fn new() -> Self {
        Self {
            src: Vec::new(),
            dst: Vec::new(),
        }
    }

    fn prepare(&mut self, src_spec: &ImageSpec, dst_spec: &ImageSpec) -> (&[u8], &mut [u8]) {
        let sb = PIXELS * src_spec.layout.bytes_per_pixel();
        let db = PIXELS * dst_spec.layout.bytes_per_pixel();
        if self.src.len() < sb {
            self.src.resize(sb, 0);
        }
        if self.dst.len() < db {
            self.dst.resize(db, 0);
        }
        match src_spec.layout.sample {
            SampleType::U8 => {
                for (i, b) in self.src[..sb].iter_mut().enumerate() {
                    *b = ((i.wrapping_mul(37).wrapping_add(11)) % 251) as u8;
                }
            }
            SampleType::U16 => {
                for (i, chunk) in self.src[..sb].as_chunks_mut::<2>().0.iter_mut().enumerate() {
                    let v = ((i.wrapping_mul(9973).wrapping_add(12345)) & 0xFFFF) as u16;
                    *chunk = v.to_ne_bytes();
                }
            }
            SampleType::F16 => {
                for (i, chunk) in self.src[..sb].as_chunks_mut::<2>().0.iter_mut().enumerate() {
                    let f = ((i % 257) as f32) / 256.0;
                    let h = half::f16::from_f32(f).to_bits();
                    *chunk = h.to_ne_bytes();
                }
            }
            SampleType::F32 => {
                for (i, chunk) in self.src[..sb].as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    let f = ((i.wrapping_mul(37).wrapping_add(13)) % 1009) as f32 / 1008.0;
                    *chunk = f.to_ne_bytes();
                }
            }
            _ => {}
        }
        (&self.src[..sb], &mut self.dst[..db])
    }
}

struct Case {
    name: &'static str,
    src: ImageSpec,
    dst: ImageSpec,
    policy: ConvertPolicy,
    hdr_peak: Option<Nits>,
}

fn rgb(
    layout: SampleLayout,
    primaries: Primaries,
    transfer: Transfer,
    alpha: AlphaMode,
) -> ImageSpec {
    ImageSpec::new(
        layout,
        ColorSpace::Rgb {
            primaries,
            transfer,
            range: SignalRange::Full,
            alpha,
            white: None,
        },
    )
}

fn cases() -> Vec<Case> {
    let p_perm = ConvertPolicy::permissive();
    let p_trunc = ConvertPolicy::permissive().with_depth(DepthPolicy::Truncate);
    let p_dither = ConvertPolicy::permissive().with_depth(DepthPolicy::OrderedDither);
    let p_soft = ConvertPolicy::permissive().with_gamut(GamutPolicy::SoftCompress);
    let p_matte = ConvertPolicy::permissive().with_alpha(AlphaPolicy::CompositeOnto(Color::BLACK));
    let p_hdr_work = ConvertPolicy::working_linear();
    let p_hdr_sdr = ConvertPolicy::display_sdr(DiffuseWhite::BT2408, Nits::new(100.0));

    let rgbf32_srgb = rgb(
        SampleLayout::RGBF32,
        Primaries::Bt709,
        Transfer::Srgb,
        AlphaMode::None,
    );
    let rgbf32_bt709 = rgb(
        SampleLayout::RGBF32,
        Primaries::Bt709,
        Transfer::Bt709,
        AlphaMode::None,
    );
    let rgbf32_g22 = rgb(
        SampleLayout::RGBF32,
        Primaries::AdobeRgb,
        Transfer::Gamma22,
        AlphaMode::None,
    );
    let rgbf32_p3_srgb = rgb(
        SampleLayout::RGBF32,
        Primaries::DisplayP3,
        Transfer::Srgb,
        AlphaMode::None,
    );
    let rgbf32_p3_lin = rgb(
        SampleLayout::RGBF32,
        Primaries::DisplayP3,
        Transfer::Linear,
        AlphaMode::None,
    );
    let rgb8_p3 = rgb(
        SampleLayout::RGB8,
        Primaries::DisplayP3,
        Transfer::Srgb,
        AlphaMode::None,
    );
    let rgb16_p3 = rgb(
        SampleLayout::RGB16,
        Primaries::DisplayP3,
        Transfer::Srgb,
        AlphaMode::None,
    );
    let rgb16_pq = ImageSpec::rgb16_bt2100_pq();
    let rgb16_hlg = ImageSpec::new(
        SampleLayout::RGB16,
        ColorSpace::Rgb {
            primaries: Primaries::Bt2020,
            transfer: Transfer::Hlg(HlgMode::Display(HlgRendering::REFERENCE_1000_NIT)),
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: Some(DiffuseWhite::BT2408),
        },
    );
    let rgbf32_bt2020_lin = ImageSpec::new(
        SampleLayout::RGBF32,
        ColorSpace::Rgb {
            primaries: Primaries::Bt2020,
            transfer: Transfer::Linear,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: Some(DiffuseWhite::BT2408),
        },
    );
    let rgba8_premul = rgb(
        SampleLayout::RGBA8,
        Primaries::Bt709,
        Transfer::Srgb,
        AlphaMode::Premultiplied,
    );
    let oklab_f32 = ImageSpec::oklab_f32();
    let jzazbz_f32 = ImageSpec::new(
        SampleLayout::new(SampleType::F32, ChannelOrder::Perceptual3),
        ColorSpace::Perceptual {
            space: PerceptualSpace::Jzazbz,
            alpha: AlphaMode::None,
        },
    );
    let rgb8_narrow = ImageSpec::new(
        SampleLayout::RGB8,
        ColorSpace::Rgb {
            primaries: Primaries::Bt709,
            transfer: Transfer::Srgb,
            range: SignalRange::Narrow,
            alpha: AlphaMode::None,
            white: None,
        },
    );
    let rgb_p010 = ImageSpec::new(
        SampleLayout::RGB16.with_packing(BitPacking::P010),
        ColorSpace::Rgb {
            primaries: Primaries::Bt709,
            transfer: Transfer::Srgb,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: None,
        },
    );
    let rgb565 = ImageSpec::new(
        SampleLayout::RGB565,
        ColorSpace::Rgb {
            primaries: Primaries::Bt709,
            transfer: Transfer::Srgb,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: None,
        },
    );

    vec![
        // Layout U8 & Wide
        Case {
            name: "layout/rgba8_to_rgb8",
            src: ImageSpec::rgba8_srgb(),
            dst: ImageSpec::rgb8_srgb(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        Case {
            name: "layout/rgb8_to_bgra8",
            src: ImageSpec::rgb8_srgb(),
            dst: ImageSpec::bgra8_srgb(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        Case {
            name: "layout/rgba8_to_bgra8",
            src: ImageSpec::rgba8_srgb(),
            dst: ImageSpec::bgra8_srgb(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        Case {
            name: "layout/rgb8_to_bgr8",
            src: ImageSpec::rgb8_srgb(),
            dst: ImageSpec::bgr8_srgb(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        Case {
            name: "layout/rgba16_to_rgb16",
            src: ImageSpec::rgba16_srgb(),
            dst: ImageSpec::rgb16_srgb(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        Case {
            name: "layout/rgbaf32_to_rgbf32",
            src: ImageSpec::rgbaf32_linear(),
            dst: ImageSpec::rgbf32_linear(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        // Depth & BitPacking
        Case {
            name: "depth/rgb8_to_rgb16",
            src: ImageSpec::rgb8_srgb(),
            dst: ImageSpec::rgb16_srgb(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        Case {
            name: "depth/rgb16_to_rgb8_round",
            src: ImageSpec::rgb16_srgb(),
            dst: ImageSpec::rgb8_srgb(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        Case {
            name: "depth/rgb16_to_rgb8_trunc",
            src: ImageSpec::rgb16_srgb(),
            dst: ImageSpec::rgb8_srgb(),
            policy: p_trunc,
            hdr_peak: None,
        },
        Case {
            name: "depth/rgb16_to_rgb8_dither",
            src: ImageSpec::rgb16_srgb(),
            dst: ImageSpec::rgb8_srgb(),
            policy: p_dither,
            hdr_peak: None,
        },
        Case {
            name: "depth/rgbf16_to_rgbf32",
            src: ImageSpec::rgbf16_linear(),
            dst: ImageSpec::rgbf32_linear(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        Case {
            name: "depth/rgbf32_to_rgbf16",
            src: ImageSpec::rgbf32_linear(),
            dst: ImageSpec::rgbf16_linear(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        Case {
            name: "depth/rgbf32_to_rgb8_round",
            src: rgbf32_srgb.clone(),
            dst: ImageSpec::rgb8_srgb(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        Case {
            name: "depth/rgb8_narrow_to_full",
            src: rgb8_narrow,
            dst: ImageSpec::rgb8_srgb(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        Case {
            name: "depth/p010_to_rgb16",
            src: rgb_p010,
            dst: ImageSpec::rgb16_srgb(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        Case {
            name: "depth/rgb565_to_rgb8",
            src: rgb565,
            dst: ImageSpec::rgb8_srgb(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        // Transfer SDR
        Case {
            name: "transfer_sdr/rgb8_srgb_to_rgbf32_lin",
            src: ImageSpec::rgb8_srgb(),
            dst: ImageSpec::rgbf32_linear(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        Case {
            name: "transfer_sdr/rgbf32_lin_to_rgb8_srgb",
            src: ImageSpec::rgbf32_linear(),
            dst: ImageSpec::rgb8_srgb(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        Case {
            name: "transfer_sdr/rgbf32_srgb_to_lin",
            src: rgbf32_srgb.clone(),
            dst: ImageSpec::rgbf32_linear(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        Case {
            name: "transfer_sdr/rgbf32_lin_to_srgb",
            src: ImageSpec::rgbf32_linear(),
            dst: rgbf32_srgb.clone(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        Case {
            name: "transfer_sdr/rgbf32_bt709_to_lin",
            src: rgbf32_bt709,
            dst: ImageSpec::rgbf32_linear(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        Case {
            name: "transfer_sdr/rgbf32_gamma22_to_lin",
            src: rgbf32_g22,
            dst: ImageSpec::rgbf32_linear(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        // Gamut
        Case {
            name: "gamut/p3_to_srgb_rgb8",
            src: rgb8_p3,
            dst: ImageSpec::rgb8_srgb(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        Case {
            name: "gamut/p3_to_srgb_rgb16",
            src: rgb16_p3,
            dst: ImageSpec::rgb16_srgb(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        Case {
            name: "gamut/p3_to_srgb_lin_rgbf32",
            src: rgbf32_p3_lin,
            dst: ImageSpec::rgbf32_linear(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        Case {
            name: "gamut/p3_to_srgb_srgb_rgbf32",
            src: rgbf32_p3_srgb.clone(),
            dst: rgbf32_srgb.clone(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        Case {
            name: "gamut/p3_to_srgb_soft_compress_f32",
            src: rgbf32_p3_srgb,
            dst: rgbf32_srgb.clone(),
            policy: p_soft,
            hdr_peak: None,
        },
        // HDR & Tone Mapping
        Case {
            name: "transfer_hdr/rgb16_pq_to_rgbf32_lin",
            src: rgb16_pq.clone(),
            dst: rgbf32_bt2020_lin.clone(),
            policy: p_hdr_work.clone(),
            hdr_peak: None,
        },
        Case {
            name: "transfer_hdr/rgbf32_lin_to_rgb16_pq",
            src: rgbf32_bt2020_lin.clone(),
            dst: rgb16_pq.clone(),
            policy: p_hdr_work.clone(),
            hdr_peak: None,
        },
        Case {
            name: "transfer_hdr/rgb16_hlg_1000nit_to_lin",
            src: rgb16_hlg,
            dst: rgbf32_bt2020_lin,
            policy: p_hdr_work,
            hdr_peak: None,
        },
        Case {
            name: "transfer_hdr/pq_u16_to_srgb_u8_bt2446a",
            src: rgb16_pq,
            dst: ImageSpec::rgb8_srgb(),
            policy: p_hdr_sdr,
            hdr_peak: Some(Nits::new(1000.0)),
        },
        // Gray, Premul, Matte
        Case {
            name: "gray_alpha/rgb8_to_gray8_bt709",
            src: ImageSpec::rgb8_srgb(),
            dst: ImageSpec::gray8_srgb(),
            policy: p_perm.clone().with_luma(LumaCoefficients::Bt709),
            hdr_peak: None,
        },
        Case {
            name: "gray_alpha/rgba8_straight_to_premul",
            src: ImageSpec::rgba8_srgb(),
            dst: rgba8_premul.clone(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        Case {
            name: "gray_alpha/rgba8_premul_to_straight",
            src: rgba8_premul,
            dst: ImageSpec::rgba8_srgb(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        Case {
            name: "gray_alpha/rgba8_srgb_matte_black",
            src: ImageSpec::rgba8_srgb(),
            dst: ImageSpec::rgb8_srgb(),
            policy: p_matte,
            hdr_peak: None,
        },
        // Oklab & Perceptual
        Case {
            name: "oklab/rgb_lin_to_oklab_f32",
            src: ImageSpec::rgbf32_linear(),
            dst: oklab_f32.clone(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        Case {
            name: "oklab/oklab_to_rgb_lin_f32",
            src: oklab_f32.clone(),
            dst: ImageSpec::rgbf32_linear(),
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        Case {
            name: "oklab/rgb8_srgb_to_oklab_f32",
            src: ImageSpec::rgb8_srgb(),
            dst: oklab_f32,
            policy: p_perm.clone(),
            hdr_peak: None,
        },
        Case {
            name: "perceptual/lin_rgb_to_jzazbz_f32",
            src: ImageSpec::rgbf32_linear(),
            dst: jzazbz_f32,
            policy: p_perm,
            hdr_peak: None,
        },
    ]
}

fn main() {
    let shared = Arc::new(Mutex::new(FrameScratch::new()));

    zenbench::run(|suite| {
        for case in cases() {
            let src_stride = (W as usize) * case.src.layout.bytes_per_pixel();
            let dst_stride = (W as usize) * case.dst.layout.bytes_per_pixel();
            let total_bytes = ((src_stride + dst_stride) * (H as usize)) as u64;

            let plan_all = RowPlan::build(
                case.src.clone(),
                case.dst.clone(),
                &case.policy.clone().with_accelerators(Accelerators::ALL),
                case.hdr_peak,
            )
            .unwrap_or_else(|e| panic!("plan ALL failed for {}: {e:?}", case.name));

            let plan_scalar = RowPlan::build(
                case.src.clone(),
                case.dst.clone(),
                &case.policy.clone().with_accelerators(Accelerators::SCALAR),
                case.hdr_peak,
            )
            .unwrap_or_else(|e| panic!("plan SCALAR failed for {}: {e:?}", case.name));

            let shared = Arc::clone(&shared);
            let src_spec = case.src;
            let dst_spec = case.dst;
            suite.compare(case.name, move |g| {
                g.throughput(Throughput::Bytes(total_bytes));
                g.config()
                    .auto_rounds(false)
                    .min_rounds(6)
                    .max_rounds(6)
                    .max_time(Duration::from_millis(900));

                for (tier, plan) in [("ALL", plan_all), ("SCALAR", plan_scalar)] {
                    let shared = Arc::clone(&shared);
                    let src_spec = src_spec.clone();
                    let dst_spec = dst_spec.clone();
                    g.bench(tier, move |b| {
                        let mut scratch = shared.lock().unwrap();
                        let (src, dst) = scratch.prepare(&src_spec, &dst_spec);
                        // Warm up lazy tables once.
                        plan.convert_row(&src[..src_stride], &mut dst[..dst_stride], W, 0)
                            .unwrap();
                        b.iter(|| {
                            for y in 0..H {
                                let so = y as usize * src_stride;
                                let do_ = y as usize * dst_stride;
                                plan.convert_row(
                                    &src[so..so + src_stride],
                                    &mut dst[do_..do_ + dst_stride],
                                    W,
                                    y,
                                )
                                .unwrap();
                            }
                            black_box(dst[dst.len() - 1])
                        })
                    });
                }
            });
        }

        // Orient 4K Rotate90 across pixel widths
        for (name, spec) in [
            ("orient_4k/rotate90_gray8_1bpp", ImageSpec::gray8_srgb()),
            ("orient_4k/rotate90_rgb8_3bpp", ImageSpec::rgb8_srgb()),
            ("orient_4k/rotate90_rgba8_4bpp", ImageSpec::rgba8_srgb()),
            ("orient_4k/rotate90_rgba16_8bpp", ImageSpec::rgba16_srgb()),
            (
                "orient_4k/rotate90_rgbf32_12bpp",
                ImageSpec::rgbf32_linear(),
            ),
        ] {
            let bytes = (PIXELS * spec.layout.bytes_per_pixel() * 2) as u64;
            suite.group(name, move |g| {
                g.throughput(Throughput::Bytes(bytes));
                g.config()
                    .auto_rounds(false)
                    .min_rounds(6)
                    .max_rounds(6)
                    .max_time(Duration::from_millis(900));
                g.bench("ALL", move |b| {
                    let src = PixelBuf::new(W, H, spec.clone()).unwrap();
                    let mut dst = PixelBuf::new(H, W, spec.clone()).unwrap();
                    b.iter(|| {
                        apply_orientation_into(
                            &src.as_view(),
                            Orientation::Rotate90,
                            &mut dst.as_view_mut(),
                        )
                        .unwrap();
                        black_box(dst.as_view().row(0)[0])
                    })
                });
            });
        }
    });
}
