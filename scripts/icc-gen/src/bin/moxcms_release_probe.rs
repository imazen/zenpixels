//! Small observations of the released CMS; output is not a correctness gate.
#![forbid(unsafe_code)]

fn main() {
    let source = moxcms::ColorProfile::new_pro_photo_rgb();
    let target = moxcms::ColorProfile::new_srgb();
    let transform = source
        .create_transform_f32(
            moxcms::Layout::Rgb,
            &target,
            moxcms::Layout::Rgb,
            moxcms::TransformOptions {
                allow_extended_range_rgb_xyz: true,
                ..Default::default()
            },
        )
        .unwrap();
    for input in [[0.0_f32; 3], [0.5, 0.5, 0.0], [0.5; 3]] {
        let mut output = [0.0; 3];
        transform.transform(&input, &mut output).unwrap();
        println!(
            "ProPhoto->sRGB {input:?}: raw={output:?}, clipped={:?}",
            output.map(|value| value.clamp(0.0, 1.0))
        );
    }
}
