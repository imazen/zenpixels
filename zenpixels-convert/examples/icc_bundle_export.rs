//! Export generated profiles for independent structural and reader checks.
#![forbid(unsafe_code)]
use zenpixels_convert::{Cicp, icc_profiles::SynthesizedIcc};

fn main() {
    let output = std::path::PathBuf::from(std::env::args_os().nth(1).expect("output directory"));
    std::fs::create_dir(&output).expect("fresh output directory");
    let mut count = 0;
    for primaries in [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22] {
        for transfer in [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] {
            let cicp = Cicp::new(primaries, transfer, 0, true);
            for (kind, profile) in [
                ("rgb", SynthesizedIcc::for_cicp(cicp)),
                ("gray", SynthesizedIcc::for_gray_cicp(cicp)),
            ] {
                match profile {
                    SynthesizedIcc::Profile(bytes) => {
                        std::fs::write(
                            output.join(format!("{kind}-{primaries}-{transfer}.icc")),
                            bytes,
                        )
                        .unwrap();
                        count += 1;
                    }
                    SynthesizedIcc::NotNeeded | SynthesizedIcc::CmsUnsupported => {}
                    _ => panic!("unhandled synthesis outcome"),
                }
            }
        }
    }
    assert_eq!(count, 313, "all supported export mappings must be present");
    println!("exported {count} profiles");
}
