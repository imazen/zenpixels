//! HLG and BT.709 rows run the SIMD slices. They stay within rounding of per-sample math.
use linear_srgb::tf;
use zenpixels::{
    ChannelLayout as L, ChannelType as T, PixelDescriptor as D, TransferFunction as Tf,
};
use zenpixels_convert::{ConvertOptions, RowConverter};

fn f32_row(from: Tf, to: Tf, input: &[f32]) -> Vec<f32> {
    let d = |tf| D::new(T::F32, L::Gray, None, tf);
    let mut c = RowConverter::new_explicit(d(from), d(to), &ConvertOptions::permissive()).unwrap();
    let mut out = vec![0f32; input.len()];
    c.convert_row(
        bytemuck::cast_slice(input),
        bytemuck::cast_slice_mut(&mut out),
        input.len() as u32,
    );
    out
}

/// In range, out of range and extremes.
fn samples() -> Vec<f32> {
    let mut v: Vec<f32> = (0..=200_000).map(|i| i as f32 / 200_000.0).collect();
    v.extend((0..1000).map(|i| -(i as f32) / 1000.0));
    v.extend((0..1000).map(|i| 1.0 + i as f32 / 10.0));
    v.extend([1e-30, -1e-30, f32::MIN_POSITIVE, -1e6, f32::NEG_INFINITY]);
    v
}

fn check(name: &str, scalar: fn(f32) -> f32, from: Tf, to: Tf) {
    let input = samples();
    let row = f32_row(from, to, &input);
    for (x, got) in input.iter().zip(&row) {
        let want = scalar(*x);
        let close = (got - want).abs() <= 1e-7_f32.max(want.abs() * 1e-6);
        assert!(
            close || got.to_bits() == want.to_bits(),
            "{name}({x}): row {got}, per-sample {want}"
        );
    }
}

#[test]
fn hlg_rows_stay_within_rounding_of_per_sample_math() {
    check("hlg_to_linear", tf::hlg_to_linear, Tf::Hlg, Tf::Linear);
    check("linear_to_hlg", tf::linear_to_hlg, Tf::Linear, Tf::Hlg);
}

#[test]
fn bt709_rows_stay_within_rounding_of_per_sample_math() {
    check(
        "bt709_to_linear",
        tf::bt709_to_linear,
        Tf::Bt709,
        Tf::Linear,
    );
    check(
        "linear_to_bt709",
        tf::linear_to_bt709,
        Tf::Linear,
        Tf::Bt709,
    );
}

/// Encoded to `u16` and back through the row kernels, every code survives.
#[test]
fn hlg_u16_codes_round_trip() {
    let d = |t, tf| D::new(t, L::Gray, None, tf);
    let opts = ConvertOptions::permissive();
    let codes: Vec<u16> = (0..=u16::MAX).collect();
    let mut linear = vec![0f32; codes.len()];
    RowConverter::new_explicit(d(T::U16, Tf::Hlg), d(T::F32, Tf::Linear), &opts)
        .unwrap()
        .convert_row(
            bytemuck::cast_slice(&codes),
            bytemuck::cast_slice_mut(&mut linear),
            codes.len() as u32,
        );
    let mut back = vec![0u16; codes.len()];
    RowConverter::new_explicit(d(T::F32, Tf::Linear), d(T::U16, Tf::Hlg), &opts)
        .unwrap()
        .convert_row(
            bytemuck::cast_slice(&linear),
            bytemuck::cast_slice_mut(&mut back),
            codes.len() as u32,
        );
    let worst = codes.iter().zip(&back).map(|(a, b)| a.abs_diff(*b)).max();
    assert!(worst.unwrap() <= 1, "worst round trip {worst:?}");
}
