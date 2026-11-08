//! Write the exact build-time export recipe for independent qualification.
#![forbid(unsafe_code)]
#[path = "../../../../zenpixels-convert/src/icc_profiles/canonical.rs"]
mod canonical;
#[path = "../../../../zenpixels-convert/src/icc_profiles/synthesis.rs"]
mod synthesis;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::path::PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("expected fresh output directory")?,
    );
    std::fs::create_dir(&out)?;
    std::fs::create_dir(out.join("linear"))?;
    let mut count = 0;
    for p in [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22] {
        for (kind, synth) in [
            ("rgb", synthesis::rgb as fn(u8, u8) -> Option<Vec<u8>>),
            ("gray", synthesis::gray),
        ] {
            if kind == "rgb" && p == 10 {
                continue;
            }
            std::fs::write(
                out.join("linear").join(format!("{kind}-{p}.icc")),
                synth(p, 8).ok_or("linear recipe")?,
            )?;
            for t in [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] {
                if p == 1 && t == 13 {
                    continue;
                }
                let data = synth(p, t).ok_or_else(|| format!("missing {kind} ({p},{t})"))?;
                std::fs::write(out.join(format!("{kind}-{p}-{t}.icc")), data)?;
                count += 1;
            }
            assert!(synth(p, 18).is_none(), "targetless HLG must refuse");
        }
    }
    assert_eq!(count, 313);
    println!("{count} export mappings written");
    Ok(())
}
