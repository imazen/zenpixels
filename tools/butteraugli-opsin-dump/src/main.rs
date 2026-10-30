//! Run imazen/butteraugli's `opsin_dynamics_image` on a deterministic 1024²
//! image and dump `in_{r,g,b,br,bg,bb}.f32` (the blurred planes from the
//! crate's own `blur_mirrored_5x5`, σ = 1.2, as `opsin_dynamics_image` uses)
//! plus `butteraugli_{0,1,2}.f32` for `tools/jzazbz-bench/oracle`.

use butteraugli::blur::{blur_mirrored_5x5, compute_separable5_weights};
use butteraugli::image::{BufferPool, Image3F, ImageF};
use butteraugli::opsin::opsin_dynamics_image;
use std::path::Path;

const W: usize = 1024;
const H: usize = 1024;
const INTENSITY: f32 = 80.0;

fn write_f32(path: &Path, v: impl Iterator<Item = f32>) {
    let bytes: Vec<u8> = v.flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap();
}

fn rows(img: &ImageF) -> impl Iterator<Item = f32> + '_ {
    (0..H).flat_map(move |y| img.row(y)[..W].iter().copied())
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: butteraugli-opsin-dump DIR");
    let dir = Path::new(&dir);
    std::fs::create_dir_all(dir).unwrap();
    // Deterministic content with structure (so the blur matters): smooth
    // ramps plus a hashed texture, linear [0, 1] with a few HDR and zero pixels.
    let mut s = 7u64;
    let mut next = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 40) as f32 / (1u64 << 24) as f32
    };
    let mut planes = [
        vec![0.0f32; W * H],
        vec![0.0f32; W * H],
        vec![0.0f32; W * H],
    ];
    for y in 0..H {
        for x in 0..W {
            let i = y * W + x;
            let base = [
                x as f32 / W as f32,
                y as f32 / H as f32,
                ((x ^ y) & 255) as f32 / 255.0,
            ];
            for c in 0..3 {
                let mut v = 0.5 * base[c] + 0.5 * next();
                if i % 997 == 0 {
                    v *= 40.0;
                }
                if i % 1009 == 5 {
                    v = 0.0;
                }
                planes[c][i] = v;
            }
        }
    }
    let pool = BufferPool::new();
    let img = Image3F::from_planes(
        ImageF::from_vec(planes[0].clone(), W, H),
        ImageF::from_vec(planes[1].clone(), W, H),
        ImageF::from_vec(planes[2].clone(), W, H),
    );
    let weights = compute_separable5_weights(1.2);
    let blurred: Vec<ImageF> = (0..3)
        .map(|c| blur_mirrored_5x5(img.plane(c), &weights, &pool))
        .collect();
    let xyb = opsin_dynamics_image(&img, INTENSITY, &pool);

    for (c, name) in ["r", "g", "b"].iter().enumerate() {
        write_f32(&dir.join(format!("in_{name}.f32")), rows(img.plane(c)));
        write_f32(&dir.join(format!("in_b{name}.f32")), rows(&blurred[c]));
        write_f32(
            &dir.join(format!("butteraugli_{c}.f32")),
            rows(xyb.plane(c)),
        );
    }
    std::fs::write(
        dir.join("meta.txt"),
        format!("n={}\nintensity_target={INTENSITY}\nsource=imazen/butteraugli@791d08d opsin_dynamics_image\n", W * H),
    )
    .unwrap();
    eprintln!("dumped {}x{} to {}", W, H, dir.display());
}
