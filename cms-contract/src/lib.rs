//! Shared tests exercise the production LCMS backend alongside MoxCMS.
pub use zenpixels_convert::cms_lcms::Lcms as LcmsPort;

/// Bytes of a gamma 2.2 gray profile, written by the port.
pub fn gray_gamma22_icc() -> Vec<u8> {
    let curve = lcms::gamma::build_gamma(None, 2.2).expect("gamma curve");
    let mut profile = lcms::virt::create_gray_profile(None, None, Some(&curve)).expect("gray");
    lcms::profile::save_profile_to_mem(&mut profile).expect("saved profile")
}
