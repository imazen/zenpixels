//! Shared tests exercise the production LCMS backend alongside MoxCMS.
pub use zenpixels_convert::Lcms as LcmsPort;

/// Bytes of a qualified gamma 2.2 gray ICC profile.
pub fn gray_gamma22_icc() -> Vec<u8> {
    let zenpixels_convert::icc_profiles::SynthesizedIcc::Profile(bytes) =
        zenpixels_convert::icc_profiles::SynthesizedIcc::for_gray_cicp(zenpixels::Cicp::new(
            1, 4, 0, true,
        ))
    else {
        panic!("icc-db feature must synthesize qualified gray gamma 2.2 ICC");
    };
    bytes.to_vec()
}
