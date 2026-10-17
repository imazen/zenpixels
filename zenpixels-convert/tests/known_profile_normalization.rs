use std::sync::Arc;
use zenpixels_convert::{OutputProfile, icc_profiles::ADOBE_RGB};

#[test]
fn header_variants_use_the_existing_fingerprint_and_canonical_bytes() {
    let mut bytes = ADOBE_RGB.to_vec();
    for range in [4..8, 24..36, 40..44, 48..56, 80..100] {
        for byte in &mut bytes[range] {
            *byte ^= 0x5a;
        }
    }
    assert_eq!(
        zenpixels::icc::normalized_hash(&bytes),
        zenpixels::icc::normalized_hash(ADOBE_RGB)
    );
    let OutputProfile::Icc(result) = OutputProfile::Icc(bytes.into()).normalize_known_icc() else {
        panic!()
    };
    assert_eq!(result.as_ref(), ADOBE_RGB);
}

#[test]
fn changed_transform_and_intent_are_not_metadata_normalization() {
    for index in [64, ADOBE_RGB.len() - 1] {
        let mut bytes = ADOBE_RGB.to_vec();
        bytes[index] ^= 1;
        let original: Arc<[u8]> = bytes.into();
        let OutputProfile::Icc(result) = OutputProfile::Icc(original.clone()).normalize_known_icc()
        else {
            panic!()
        };
        assert!(Arc::ptr_eq(&original, &result));
    }
    let original: Arc<[u8]> = b"unknown profile".as_slice().into();
    let OutputProfile::Icc(result) = OutputProfile::Icc(original.clone()).normalize_known_icc()
    else {
        panic!()
    };
    assert!(Arc::ptr_eq(&original, &result));
}

#[test]
fn non_icc_targets_keep_their_meaning() {
    assert!(matches!(
        OutputProfile::SameAsOrigin.normalize_known_icc(),
        OutputProfile::SameAsOrigin
    ));
    assert!(matches!(
        OutputProfile::Named(zenpixels::Cicp::SRGB).normalize_known_icc(),
        OutputProfile::Named(zenpixels::Cicp::SRGB)
    ));
}
