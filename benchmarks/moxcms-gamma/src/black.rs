#![forbid(unsafe_code)]
// Does dirty_powf's underflow reach pixels? A black and a near-black pixel through the extended route.
macro_rules! run {
    ($m:ident, $name:expr, $s:expr, $t:expr) => {{
        let o = $m::TransformOptions {
            allow_extended_range_rgb_xyz: true,
            ..Default::default()
        };
        let tr = $s
            .create_transform_f32($m::Layout::Rgb, &$t, $m::Layout::Rgb, o)
            .unwrap();
        for src in [
            [0f32, 0., 0.],
            [0., 1., 0.],
            [1e-30, 1e-30, 1e-30],
            [1e-20, 1e-20, 1e-20],
            [1e-12, 1e-12, 1e-12],
        ] {
            let mut d = [0f32; 3];
            tr.transform(&src, &mut d).unwrap();
            println!("{:>30} {src:?} -> {d:?}", $name);
        }
    }};
}
fn main() {
    use m0::ColorProfile as P;
    run!(
        m0,
        "ProPhoto(1.8)->sRGB",
        P::new_pro_photo_rgb(),
        P::new_srgb()
    );
    run!(m0, "AdobeRGB(2.2)->sRGB", P::new_adobe_rgb(), P::new_srgb());
    run!(m0, "sRGB->AdobeRGB(2.2)", P::new_srgb(), P::new_adobe_rgb());
    run!(
        m0,
        "ProPhoto(1.8)->AdobeRGB(2.2)",
        P::new_pro_photo_rgb(),
        P::new_adobe_rgb()
    );
}
