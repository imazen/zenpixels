//! Export the exact public API bytes for native recognition/color qualification.
use zenpixels_convert::{
    Cicp,
    icc_profiles::{DISPLAY_P3_V4, SynthesizedIcc},
};
fn main() {
    let out = std::path::PathBuf::from(std::env::args_os().nth(1).expect("output directory"));
    std::fs::create_dir_all(&out).unwrap();
    for t in [13, 1] {
        let mut cicp = Cicp::DISPLAY_P3;
        cicp.transfer_characteristics = t;
        let SynthesizedIcc::Profile(bytes) = SynthesizedIcc::for_cicp(cicp) else {
            panic!("P3 synthesis refused")
        };
        std::fs::write(out.join(format!("p3-{t}.icc")), &bytes).unwrap();
        let array = format!(
            "// Exact current bundle export for CP12/TC{t}; qualified by tools/p3-export-oracle.\n&{:?}\n",
            &*bytes
        );
        std::fs::write(out.join(format!("p3-{t}.rs")), array).unwrap();
    }
    std::fs::write(out.join("compat.icc"), DISPLAY_P3_V4).unwrap();
}
