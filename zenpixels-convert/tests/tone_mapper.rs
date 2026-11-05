//! Injecting a tone mapper, and what the default is.
#![cfg(feature = "hdr-experimental")]
use std::sync::Arc;
use zenpixels::{ColorPrimaries as P, PixelDescriptor as D};
use zenpixels_convert::hdr::{
    Bt2446A, PreparedMapper, Requirements, StripMapper, ToneMapContext, ToneMapper,
};
use zenpixels_convert::{ConvertError, ConvertPlan, HdrConfig, RowConverter};

fn linear_2020() -> D {
    D::RGBF32_LINEAR.with_primaries(P::Bt2020)
}

fn ramp() -> Vec<f32> {
    (0..64 * 3).map(|i| i as f32 / (64.0 * 3.0 - 1.0)).collect()
}

fn run(plan: ConvertPlan, src: &[f32], width: u32) -> Vec<f32> {
    let mut out = vec![0f32; src.len()];
    RowConverter::from_plan(plan).convert_row(
        bytemuck::cast_slice(src),
        bytemuck::cast_slice_mut(&mut out),
        width,
    );
    out
}

/// Halves everything, so the output can't be mistaken for the default.
#[derive(Debug)]
struct Halve;
#[derive(Debug)]
struct HalveStrip;
impl StripMapper for HalveStrip {
    fn map_strip(&self, input: &[f32], output: &mut [f32]) {
        for (o, i) in output.iter_mut().zip(input) {
            *o = i / 2.0;
        }
    }
}
impl ToneMapper for Halve {
    fn requirements(&self) -> Requirements {
        Requirements::NONE
    }
    fn prepare(&self, ctx: &ToneMapContext<'_>) -> PreparedMapper {
        assert_eq!(ctx.source_peak_nits, 1000.0);
        PreparedMapper::Streaming(Box::new(HalveStrip))
    }
    fn name(&self) -> &'static str {
        "halve"
    }
}

#[test]
fn an_injected_mapper_runs_in_place_of_the_default() {
    let src = ramp();
    let plan = ConvertPlan::new_with_tone_mapper(
        linear_2020(),
        linear_2020(),
        HdrConfig::for_source_peak(1000.0),
        Arc::new(Halve),
    )
    .unwrap();
    let out = run(plan, &src, 64);
    for (o, i) in out.iter().zip(&src) {
        assert_eq!(*o, i / 2.0);
    }
}

#[test]
fn the_default_gives_the_bytes_of_the_curve_itself() {
    let src = ramp();
    let config = HdrConfig::for_source_peak(1000.0);
    let by_default = run(
        ConvertPlan::new_with_hdr_config(linear_2020(), linear_2020(), config).unwrap(),
        &src,
        64,
    );
    let injected = run(
        ConvertPlan::new_with_tone_mapper(
            linear_2020(),
            linear_2020(),
            config,
            Arc::new(Bt2446A::new(1000.0, 100.0)),
        )
        .unwrap(),
        &src,
        64,
    );
    assert_eq!(by_default, injected);
    let mut strip: Vec<[f32; 3]> = src.as_chunks::<3>().0.to_vec();
    Bt2446A::new(1000.0, 100.0).map_strip_simd(&mut strip);
    let direct: Vec<f32> = strip
        .as_flattened()
        .iter()
        .map(|v| v.clamp(0.0, 1.0))
        .collect();
    assert_eq!(by_default, direct);
}

/// A row's split must not show, and alpha passes through.
#[test]
fn rows_split_and_alpha_pass_through() {
    let d = D::RGBAF32_LINEAR.with_primaries(P::Bt2020);
    let src: Vec<f32> = (0..64)
        .flat_map(|i| {
            let v = i as f32 / 63.0;
            [v, v * 0.5, 1.0 - v, i as f32 / 100.0]
        })
        .collect();
    let plan =
        || ConvertPlan::new_with_hdr_config(d, d, HdrConfig::for_source_peak(1000.0)).unwrap();
    let whole = run(plan(), &src, 64);
    let mut converter = RowConverter::from_plan(plan());
    let mut parts = vec![0f32; src.len()];
    for (a, b) in src.chunks(7 * 4).zip(parts.chunks_mut(7 * 4)) {
        converter.convert_row(
            bytemuck::cast_slice(a),
            bytemuck::cast_slice_mut(b),
            (a.len() / 4) as u32,
        );
    }
    assert_eq!(whole, parts);
    for (o, i) in whole.chunks(4).zip(src.chunks(4)) {
        assert_eq!(o[3], i[3]);
    }
}

#[derive(Debug)]
struct NeedsEverything;
impl ToneMapper for NeedsEverything {
    fn requirements(&self) -> Requirements {
        let mut r = Requirements::NONE;
        r.full_image = true;
        r
    }
    fn prepare(&self, _: &ToneMapContext<'_>) -> PreparedMapper {
        unreachable!()
    }
    fn name(&self) -> &'static str {
        "needs-everything"
    }
}

#[test]
fn full_image_mappers_refuse_at_construction() {
    let error = ConvertPlan::new_with_tone_mapper(
        linear_2020(),
        linear_2020(),
        HdrConfig::for_source_peak(1000.0),
        Arc::new(NeedsEverything),
    )
    .unwrap_err();
    assert!(matches!(
        error.error(),
        ConvertError::FullImageMapperNotSupported {
            mapper_name: "needs-everything"
        }
    ));
}

/// Output is identical at every tier above scalar on this CPU. The scalar tier isn't, because
/// `pow_midp_unchecked`, `log2_midp_unchecked` and `exp2_midp_unchecked` round differently without FMA.
/// The bound is well under a 16-bit code. Closing it needs portable lanes for those three.
#[test]
fn tiers_agree_within_a_fraction_of_a_16_bit_code() {
    let mut src: Vec<[f32; 3]> = (0..67)
        .map(|i| {
            let v = i as f32 / 66.0;
            [v, v * v, 1.0 - v]
        })
        .collect();
    src.extend([[0.0; 3], [1.0; 3], [0.5, 0.0, 0.0], [1e-5; 3]]);
    let curve = Bt2446A::new(1000.0, 100.0);
    let reference = {
        let mut row = src.clone();
        curve.map_strip_simd(&mut row);
        row
    };
    let worst = std::sync::Mutex::new(0.0f32);
    let report = archmage::testing::for_each_token_permutation(
        archmage::testing::CompileTimePolicy::Warn,
        |perm| {
            let mut row = src.clone();
            curve.map_strip_simd(&mut row);
            let delta = row
                .as_flattened()
                .iter()
                .zip(reference.as_flattened())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("{perm}: {delta:e}");
            let mut w = worst.lock().unwrap();
            *w = w.max(delta);
        },
    );
    let worst = *worst.lock().unwrap();
    println!("worst across {report}: {worst:e}");
    assert!(worst < 1.0 / 65535.0 / 4.0, "{worst:e}");
}
