//! Scratch: dump every slice at every tier, for comparison across builds and machines.
use linear_srgb::default as d;
use std::{fs, path::PathBuf};

fn set(tier: &str) {
    #[cfg(target_arch = "x86_64")]
    {
        let _ = archmage::X64V3Token::dangerously_disable_token_process_wide(tier == "scalar");
        let _ = archmage::X64V4Token::dangerously_disable_token_process_wide(tier != "v4x");
        let _ = archmage::X64V4xToken::dangerously_disable_token_process_wide(tier != "v4x");
    }
    #[cfg(target_arch = "aarch64")]
    let _ = archmage::NeonToken::dangerously_disable_token_process_wide(tier == "scalar");
    let _ = tier;
}

fn grid() -> Vec<f32> {
    let mut v: Vec<f32> = (0..=1_000_000).map(|i| i as f32 / 1e6).collect();
    v.extend((0..=200_000).map(|i| { let t = i as f32 / 2e5; t * t * t * t }));
    v
}

#[test]
fn dump() {
    let out: PathBuf = std::env::var("FMA_OUT").unwrap().into();
    fs::create_dir_all(&out).unwrap();
    let tiers: &[&str] = if cfg!(target_arch = "x86_64") { &["v4x", "v3", "scalar"] } else if cfg!(target_arch = "wasm32") { &["wasm128"] } else { &["neon", "scalar"] };
    let funcs: Vec<(&str, Box<dyn Fn(&mut [f32])>)> = vec![
        ("srgb_to_linear", Box::new(|v| d::srgb_to_linear_slice(v))),
        ("linear_to_srgb", Box::new(|v| d::linear_to_srgb_slice(v))),
        ("srgb_to_linear_ext", Box::new(|v| d::srgb_to_linear_extended_slice(v))),
        ("linear_to_srgb_ext", Box::new(|v| d::linear_to_srgb_extended_slice(v))),
        ("gamma_to_linear", Box::new(|v| d::gamma_to_linear_slice(v, 2.19921875))),
        ("linear_to_gamma", Box::new(|v| d::linear_to_gamma_slice(v, 2.19921875))),
        ("bt709_to_linear", Box::new(|v| d::bt709_to_linear_slice(v))),
        ("linear_to_bt709", Box::new(|v| d::linear_to_bt709_slice(v))),
        ("pq_to_linear", Box::new(|v| d::pq_to_linear_slice(v))),
        ("linear_to_pq", Box::new(|v| d::linear_to_pq_slice(v))),
        ("hlg_to_linear", Box::new(|v| d::hlg_to_linear_slice(v))),
        ("linear_to_hlg", Box::new(|v| d::linear_to_hlg_slice(v))),
    ];
    let g = grid();
    for &t in tiers {
        set(t);
        for (name, f) in &funcs {
            let mut v = g.clone();
            let start = std::time::Instant::now();
            f(&mut v);
            let ns = start.elapsed().as_secs_f64() * 1e9 / v.len() as f64;
            println!("TIME {t} {name} {ns:.2}");
            fs::write(out.join(format!("{t}_{name}.bin")), bytemuck::cast_slice::<f32, u8>(&v)).unwrap();
        }
    }
    set("all");
    }
