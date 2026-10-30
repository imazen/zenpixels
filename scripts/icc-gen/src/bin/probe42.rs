//! One-off probe (zenpipe#42): measure ICC→sRGB identity error via moxcms
//! on an RGB lattice, per rendering intent. Ground truth for whether
//! `is_common_srgb` SHOULD recognize a profile (identity ⇒ skip-CMS safe).

use moxcms::{ColorProfile, Layout, RenderingIntent, TransformOptions};

fn main() {
    let srgb = ColorProfile::new_srgb();
    let steps: Vec<u16> = (0..=16u32).map(|i| (i * 65535 / 16) as u16).collect();
    let mut src: Vec<u16> = Vec::new();
    for &r in &steps {
        for &g in &steps {
            for &b in &steps {
                src.extend_from_slice(&[r, g, b]);
            }
        }
    }

    for path in std::env::args().skip(1) {
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(e) => {
                println!("{path}: read error {e}");
                continue;
            }
        };
        let profile = match ColorProfile::new_from_slice(&data) {
            Ok(p) => p,
            Err(e) => {
                println!("{path}: parse error {e:?}");
                continue;
            }
        };
        for (label, intent) in [
            ("perceptual", RenderingIntent::Perceptual),
            ("rel-colorimetric", RenderingIntent::RelativeColorimetric),
            ("saturation", RenderingIntent::Saturation),
        ] {
            let opts = TransformOptions {
                rendering_intent: intent,
                ..Default::default()
            };
            let transform =
                match profile.create_transform_16bit(Layout::Rgb, &srgb, Layout::Rgb, opts) {
                    Ok(t) => t,
                    Err(e) => {
                        println!("{path}: {label}: transform error {e:?}");
                        continue;
                    }
                };
            let mut dst = vec![0u16; src.len()];
            if let Err(e) = transform.transform(&src, &mut dst) {
                println!("{path}: {label}: exec error {e:?}");
                continue;
            }
            let max_err: u32 = src
                .iter()
                .zip(&dst)
                .map(|(a, b)| (*a as i64 - *b as i64).unsigned_abs() as u32)
                .max()
                .unwrap_or(0);
            println!(
                "{path}: {label}: max u16 err {max_err} (~{:.2} of u8 step)",
                max_err as f64 / 257.0
            );
        }
    }
}
