#![forbid(unsafe_code)]
// Every CICP transfer on the extended route at zero and tiny inputs: source curve into linear, linear into target curve.
use m0::{
    CicpColorPrimaries as Cp, CicpProfile, ColorProfile, Layout, MatrixCoefficients as Mc,
    TransferCharacteristics as T, TransformOptions,
};
fn with(tc: T) -> ColorProfile {
    let mut p = ColorProfile::new_srgb();
    p.cicp = Some(CicpProfile {
        color_primaries: Cp::Bt709,
        transfer_characteristics: tc,
        matrix_coefficients: Mc::Bt709,
        full_range: true,
    });
    p
}
fn main() {
    let o = TransformOptions {
        allow_extended_range_rgb_xyz: true,
        ..Default::default()
    };
    let tcs = [
        ("Bt709", T::Bt709),
        ("Bt470M(2.2)", T::Bt470M),
        ("Bt470Bg(2.8)", T::Bt470Bg),
        ("Smpte240", T::Smpte240),
        ("Log100", T::Log100),
        ("Log100sqrt10", T::Log100sqrt10),
        ("Iec61966", T::Iec61966),
        ("Bt1361", T::Bt1361),
        ("Srgb", T::Srgb),
        ("Smpte2084", T::Smpte2084),
        ("Smpte428", T::Smpte428),
        ("Hlg", T::Hlg),
    ];
    let lin = with(T::Linear);
    let inputs = [0f32, -0.5, -0.051, 1e-30, 1e-20, 1e-12, 0.5, 1.5];
    for (n, tc) in tcs {
        let p = with(tc);
        let dec = p
            .create_transform_f32(Layout::Rgb, &lin, Layout::Rgb, o)
            .unwrap();
        let enc = lin
            .create_transform_f32(Layout::Rgb, &p, Layout::Rgb, o)
            .unwrap();
        for x in inputs {
            let (mut a, mut b) = ([0f32; 3], [0f32; 3]);
            dec.transform(&[x; 3], &mut a).unwrap();
            enc.transform(&[x; 3], &mut b).unwrap();
            println!(
                "{n:>14} x={x:>9e} decode {:>13e}  encode {:>13e}",
                a[0], b[0]
            );
        }
    }
}
