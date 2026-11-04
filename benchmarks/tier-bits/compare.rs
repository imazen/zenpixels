//! Compares two dumps of f32 samples: count, largest ULP distance, differences after u16 rounding.
use std::fs;
const FN: [&str; 12] = ["srgb_to_linear", "linear_to_srgb", "srgb_to_linear_ext", "linear_to_srgb_ext", "gamma_to_linear", "linear_to_gamma", "bt709_to_linear", "linear_to_bt709", "pq_to_linear", "linear_to_pq", "hlg_to_linear", "linear_to_hlg"];
fn load(root: &str, build: &str, tier: &str, f: &str) -> Option<Vec<u32>> {
    let b = fs::read(format!("{root}/{build}/{tier}_{f}.bin")).ok()?;
    Some(b.chunks_exact(4).map(|c| u32::from_ne_bytes(c.try_into().unwrap())).collect())
}
fn code(u: u32) -> u32 { (f32::from_bits(u).clamp(0.0, 1.0) * 65535.0 + 0.5) as u32 }
fn main() {
    let root = std::env::args().nth(1).unwrap();
    let pairs: Vec<(String, String, String, String)> = std::env::args().skip(2).map(|p| { let v: Vec<&str> = p.split(':').collect(); (v[0].into(), v[1].into(), v[2].into(), v[3].into()) }).collect();
    for (a, ta, b, tb) in pairs {
        let (mut total, mut worst, mut codes, mut samples, mut same_fns, mut fns) = (0u64, 0u32, 0u64, 0u64, 0, 0);
        let mut rows = String::new();
        for f in FN {
            let (Some(x), Some(y)) = (load(&root, &a, &ta, f), load(&root, &b, &tb, f)) else { rows += &format!("   {f:<20} missing\n"); continue };
            assert_eq!(x.len(), y.len());
            let (mut n, mut mu, mut c) = (0u64, 0u32, 0u64);
            let mut wc = 0u32;
            for (p, q) in x.iter().zip(&y) {
                if p != q { n += 1; mu = mu.max(p.abs_diff(*q)); c += u64::from(code(*p) != code(*q)); wc = wc.max(code(*p).abs_diff(code(*q))); }
            }
            fns += 1; samples += x.len() as u64; same_fns += i32::from(n == 0);
            total += n; worst = worst.max(mu); codes += c;
            rows += &format!("   {f:<20}{n:>9}{mu:>11}{c:>7}{wc:>5}\n");
        }
        println!("== {a}/{ta} vs {b}/{tb}: {same_fns} of {fns} functions identical; {total} of {samples} differ, worst {worst} ULP, {codes} differ as u16\n{rows}");
    }
}
