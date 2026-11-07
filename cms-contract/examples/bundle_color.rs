//! The existing bundle-color sample grid through the pinned pure-Rust LCMS port.
use lcms::{profile, virt, xform};

const RGB_F32: u32 = (1 << 22) | (4 << 16) | (3 << 3) | 4;
const XYZ_F64: u32 = (1 << 22) | (9 << 16) | (3 << 3);

fn transform(icc: &[u8], samples: &[[f32; 3]], intent: u32) -> Vec<[f64; 3]> {
    let mut profiles = vec![
        profile::open_profile_from_mem(None, icc).expect("RGB ICC profile"),
        virt::create_xyz_profile(None).expect("XYZ destination"),
    ];
    let adaptation = if intent == 3 { 0.0 } else { 1.0 };
    let mut transform = xform::create_extended_transform(
        None,
        &mut profiles,
        &mut [false; 2],
        &[intent; 2],
        &[adaptation; 2],
        None,
        0,
        RGB_F32,
        XYZ_F64,
        xform::FLAGS_NOOPTIMIZE | xform::FLAGS_NOCACHE,
    )
    .expect("RGB to XYZ transform");
    let mut output = vec![[0.0f64; 3]; samples.len()];
    transform.do_transform(
        bytemuck::cast_slice(samples),
        bytemuck::cast_slice_mut(&mut output),
        u32::try_from(samples.len()).unwrap(),
    );
    output
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err("expected old.icc new.icc".into());
    }
    let old = std::fs::read(&args[0])?;
    let new = std::fs::read(&args[1])?;
    let mut samples = Vec::new();
    for r in [0.0, 0.01, 0.18, 0.5, 1.0] {
        for g in [0.0, 0.01, 0.18, 0.5, 1.0] {
            for b in [0.0, 0.01, 0.18, 0.5, 1.0] {
                samples.push([r, g, b]);
            }
        }
    }
    samples.extend([
        [0.0001; 3],
        [0.001; 3],
        [0.1; 3],
        [0.75; 3],
        [-0.1, 0.5, 0.5],
        [1.1, 0.5, 0.5],
        [0.5, -0.1, 0.5],
        [0.5, 1.1, 0.5],
        [0.5, 0.5, -0.1],
        [0.5, 0.5, 1.1],
        [1.5; 3],
    ]);
    assert_eq!(samples.len(), 136);
    for intent in [1, 3] {
        let before = transform(&old, &samples, intent);
        let after = transform(&new, &samples, intent);
        for (index, ((input, a), b)) in samples.iter().zip(&before).zip(&after).enumerate() {
            print!("{intent}\t{index}");
            for value in input {
                print!("\t{value:.9}");
            }
            for value in a.iter().chain(b) {
                print!("\t{value:.17e}");
            }
            println!();
        }
    }
    Ok(())
}
