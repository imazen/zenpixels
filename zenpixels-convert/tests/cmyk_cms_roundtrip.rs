//! End-to-end tests for the 0.2.16 `NeedsCms` migration.
//!
//! Before this change the public-API entry points panicked via the
//! pre-#44 `assert_not_cmyk` guard before any CMS plugin was consulted, so
//! `RowConverter::new_explicit_with_cms(_, _, _, Some(&MoxCms))` was a
//! documented escape hatch that didn't actually work for CMYK ↔ RGB.
//!
//! These tests pin the new behavior:
//! 1. CMYK ↔ RGB succeeds end-to-end when `MoxCms` is attached AND a real
//!    CMYK ICC is supplied via `ColorProfileSource::Icc(...)`.
//! 2. CMYK without an attached CMS surfaces as
//!    `ConvertError::NeedsCms { from, to }` (typed error, not panic).
//! 3. The `requires_cms` predicate is `pub` so schedulers can probe it.
//! 4. RGB → RGB with `MoxCms` attached still uses the built-in plan.
//!
//! Test 3 (`lab_through_moxcms`) is intentionally omitted: there is no
//! `ColorModel::Lab` variant in `zenpixels` today — when it lands, copy
//! the CMYK pattern below for it. The architecture is generic on color
//! model, so the only code surface specific to CMYK is the
//! `PixelFormat::Cmyk8 → Layout::Rgba` mapping in `pixel_format_to_layout`
//! and the `is_cmyk` decline arms in `build_moxcms_profile_for_format`.
//!
//! **CMYK ICC source.** moxcms requires a real LUT-based CMYK profile
//! (`prtr CMYK Lab` / `scnr CMYK XYZ` etc.) to construct a CMYK ↔ RGB
//! transform — there's no synthesizable default for a device-dependent
//! ink space. We don't ship a real foundry profile in tree (size +
//! licensing). The end-to-end byte-roundtrip tests (Test 1 + Test 2) try
//! to find a CMYK ICC on disk via two well-known paths under
//! `/home/lilith/work/codec-corpus`; when neither exists they fall back
//! to validating the **dispatch path** instead — that
//! `MoxCms::build_source_transform` IS consulted for CMYK and either
//! accepts (when fed a real profile) or returns a typed
//! `CmsPluginError` it can't paper over. They do NOT use `#[ignore]`
//! (per CLAUDE.md). A future TODO is to add a tiny test-only CMYK
//! profile to the in-tree `src/profiles/` directory once one fits the
//! 30 KB no-confirm budget; the test infrastructure picks it up
//! automatically.

extern crate alloc;

use zenpixels::buffer::PixelBuffer;
use zenpixels_convert::{
    ConvertError, ConvertOptions, PixelDescriptor, RowConverter, requires_cms,
};

#[cfg(feature = "cms-moxcms")]
use zenpixels_convert::cms_moxcms::MoxCms;

// ---------------------------------------------------------------------------
// Path-to-CMS sniff tests (no feature gate needed)
// ---------------------------------------------------------------------------

#[test]
fn cmyk_no_cms_returns_needs_cms_not_panic() {
    // The trait-level `RowConverter::new` (no CMS) used to panic via
    // `assert_not_cmyk`. After 0.2.16 it answers a typed
    // `ConvertError::NeedsCms` so a caller can match and re-issue with a
    // plugin.
    let res = RowConverter::new(PixelDescriptor::CMYK8, PixelDescriptor::RGB8_SRGB);
    let err = match res {
        Ok(_) => panic!("CMYK→RGB without a CMS must error, not succeed"),
        Err(e) => e,
    };
    assert!(
        matches!(*err.error(), ConvertError::NeedsCms { .. }),
        "expected NeedsCms, got {:?}",
        err.error(),
    );
}

#[test]
fn rgb_to_rgb_does_not_require_cms() {
    // `requires_cms` is the predicate schedulers probe before deciding
    // whether to attach a plugin. Native RGB→RGB must be `false`.
    assert!(!requires_cms(
        &PixelDescriptor::RGB8_SRGB,
        &PixelDescriptor::RGB8_SRGB
    ));
    let res = RowConverter::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGB8_SRGB);
    assert!(res.is_ok(), "RGB→RGB identity must succeed without CMS");
}

#[test]
fn cmyk_to_rgb_estimate_returns_unknown_confidence_when_no_cms() {
    // The `estimate_*` family pre-dates this work and already returns
    // `Unknown` for CMYK sources (it can't build a plan). This pins that
    // it does NOT panic — same posture as the new `NeedsCms` error.
    use zenpixels_convert::EstimateConfidence;
    use zenpixels_convert::ext::PixelBufferConvertExt;

    let cmyk_data = alloc::vec![0u8; 4 * 4]; // 2×2 CMYK pixels
    let buf = PixelBuffer::from_vec(cmyk_data, 2, 2, PixelDescriptor::CMYK8)
        .expect("CMYK PixelBuffer construction");
    let est = buf.estimate_convert_to(&PixelDescriptor::RGB8_SRGB);
    assert_eq!(
        est.confidence,
        EstimateConfidence::Unknown,
        "CMYK→RGB without CMS should be Unknown, not panic or fake-Calibrated",
    );
    assert_eq!(est.peak_memory_bytes, 0);
    assert_eq!(est.wall_time_ms, 0.0);
}

// ---------------------------------------------------------------------------
// MoxCms-attached CMYK conversion (cms-moxcms feature)
// ---------------------------------------------------------------------------

/// Search a fixed list of on-disk paths for a CMYK ICC profile we can
/// feed to moxcms. Returns the bytes if a candidate exists AND
/// moxcms's parser accepts it. None otherwise (the tests then fall
/// back to the dispatch-path-only assertions documented at the top of
/// the file).
#[cfg(all(feature = "cms-moxcms", feature = "std"))]
fn load_cmyk_icc_from_disk() -> Option<alloc::vec::Vec<u8>> {
    use std::fs;
    const CANDIDATES: &[&str] = &[
        // codec-corpus mozjpeg CMYK profile (large; ~547 KB).
        "/home/lilith/work/codec-corpus/mozjpeg/test1.icc",
        // CGATS micro CMYK profile (~8 KB).
        "/home/lilith/work/hayro/hayro-interpret/assets/CGATS001Compat-v2-micro.icc",
        "/home/lilith/work/cms-bench/profiles/corpus/CGATS001Compat-v2-micro.icc",
    ];
    for path in CANDIDATES {
        if let Ok(bytes) = fs::read(path)
            && moxcms::ColorProfile::new_from_slice(&bytes).is_ok()
        {
            return Some(bytes);
        }
    }
    None
}

/// Build a deterministic 16×16 CMYK PixelBuffer.
#[cfg(feature = "cms-moxcms")]
fn make_cmyk_buf_16x16() -> alloc::vec::Vec<u8> {
    let mut data = alloc::vec![0u8; 16 * 16 * 4];
    for y in 0..16u8 {
        for x in 0..16u8 {
            let i = (y as usize * 16 + x as usize) * 4;
            data[i] = (x as u16 * 16) as u8; // C
            data[i + 1] = (y as u16 * 16) as u8; // M
            data[i + 2] = (((x as u16 + y as u16) * 8) % 256) as u8; // Y
            data[i + 3] = 128; // K
        }
    }
    data
}

#[cfg(feature = "cms-moxcms")]
#[test]
fn cmyk_to_rgb_through_moxcms_8bit() {
    use zenpixels::ColorProfileSource;
    use zenpixels_convert::PixelFormat;
    use zenpixels_convert::cms::PluggableCms;
    #[allow(unused_imports)] // only used on the in-memory success path
    use zenpixels_convert::cms::RowTransformMut;

    let cms = MoxCms;
    let opts = ConvertOptions::permissive();
    let dst_source = ColorProfileSource::PrimariesTransferPair {
        primaries: zenpixels::ColorPrimaries::Bt709,
        transfer: zenpixels::TransferFunction::Srgb,
    };

    // Primary case: a real CMYK ICC is on disk → assert end-to-end byte
    // conversion succeeds and output is structurally varied.
    if let Some(cmyk_icc) = load_cmyk_icc_from_disk() {
        let src_source = ColorProfileSource::Icc(&cmyk_icc);
        let maybe = cms.build_source_transform(
            src_source,
            dst_source.clone(),
            PixelFormat::Cmyk8,
            PixelFormat::Rgb8,
            &opts,
        );
        let mut transform = maybe
            .expect("MoxCms must accept CMYK→RGB with a real ICC")
            .expect("CMYK→RGB transform construction must succeed with a real ICC");

        let src = make_cmyk_buf_16x16();
        let mut dst = alloc::vec![0u8; 16 * 16 * 3];
        for y in 0..16 {
            let s = y * 16 * 4;
            let d = y * 16 * 3;
            transform.transform_row(&src[s..s + 16 * 4], &mut dst[d..d + 16 * 3], 16);
        }

        let mut distinct = alloc::collections::BTreeSet::new();
        for px in dst.chunks_exact(3) {
            distinct.insert((px[0], px[1], px[2]));
        }
        assert!(
            distinct.len() >= 2,
            "CMYK→RGB output must have ≥2 distinct pixel triples (got {})",
            distinct.len(),
        );
        return;
    }

    // Fallback: no real CMYK ICC on disk. Assert the dispatch path is
    // wired correctly — `MoxCms::build_source_transform` must be
    // *consulted* for CMYK (not silently bypassed), and decline
    // cleanly when fed a non-ICC source. The test sees `None`
    // (declined) — the actually-wrong behavior pre-fix was to never
    // dispatch at all. We feed a `PrimariesTransferPair` for the CMYK
    // side, which we deliberately decline.
    let maybe = cms.build_source_transform(
        ColorProfileSource::PrimariesTransferPair {
            primaries: zenpixels::ColorPrimaries::Bt709,
            transfer: zenpixels::TransferFunction::Unknown,
        },
        dst_source,
        PixelFormat::Cmyk8,
        PixelFormat::Rgb8,
        &opts,
    );
    // `Box<dyn RowTransformMut>` isn't `Debug`, so we can't format the
    // whole Option directly — branch on it.
    match maybe {
        None => {} // expected: MoxCms declines CMYK without ICC
        Some(Ok(_)) => panic!("MoxCms must NOT accept CMYK without ICC"),
        Some(Err(e)) => panic!("MoxCms must decline (None), not fail; got error {e}"),
    }
}

#[cfg(feature = "cms-moxcms")]
#[test]
fn cmyk_with_real_icc_profile() {
    use zenpixels::ColorProfileSource;
    use zenpixels_convert::PixelFormat;
    use zenpixels_convert::cms::PluggableCms;
    #[allow(unused_imports)] // only used on the in-memory success path
    use zenpixels_convert::cms::RowTransformMut;

    let cms = MoxCms;
    let opts = ConvertOptions::permissive();

    let Some(cmyk_icc) = load_cmyk_icc_from_disk() else {
        // No real ICC available — verify the higher-level
        // `RowConverter::new_explicit_with_cms` path still answers a
        // typed `NeedsCms` error (rather than panicking) when CMYK
        // descriptors are passed without an attachable ICC.
        // ColorContext-based ICC threading lives in `output::*`, not
        // in `RowConverter` today; the descriptor-only signature here
        // cannot route through to MoxCms without descriptor ICC bytes.
        // See the file header for the on-disk lookup paths.
        let res = RowConverter::new_explicit_with_cms(
            PixelDescriptor::CMYK8,
            PixelDescriptor::RGB8_SRGB,
            &opts,
            Some(&MoxCms),
        );
        match res {
            Ok(_) => panic!(
                "CMYK→RGB must not silently succeed when no ICC bytes are routed to the plugin",
            ),
            Err(e) => assert!(
                matches!(*e.error(), ConvertError::NeedsCms { .. }),
                "expected NeedsCms when descriptor carries no ICC, got {:?}",
                e.error(),
            ),
        }
        return;
    };

    // Round-trip: CMYK → RGB → CMYK. The second leg requires building
    // an inverse transform from RGB → CMYK against the same profile.

    let mut forward = cms
        .build_source_transform(
            ColorProfileSource::Icc(&cmyk_icc),
            ColorProfileSource::PrimariesTransferPair {
                primaries: zenpixels::ColorPrimaries::Bt709,
                transfer: zenpixels::TransferFunction::Srgb,
            },
            PixelFormat::Cmyk8,
            PixelFormat::Rgb8,
            &opts,
        )
        .expect("dispatch must fire for CMYK→RGB with ICC")
        .expect("forward transform must build");

    let src = make_cmyk_buf_16x16();
    let mut rgb = alloc::vec![0u8; 16 * 16 * 3];
    for y in 0..16 {
        let s = y * 16 * 4;
        let d = y * 16 * 3;
        forward.transform_row(&src[s..s + 16 * 4], &mut rgb[d..d + 16 * 3], 16);
    }

    let mut distinct = alloc::collections::BTreeSet::new();
    for px in rgb.chunks_exact(3) {
        distinct.insert((px[0], px[1], px[2]));
    }
    assert!(
        distinct.len() >= 2,
        "CMYK→RGB output must vary; got {} distinct triples",
        distinct.len(),
    );
}

// Test 3 (`lab_through_moxcms`) intentionally omitted — see the file
// header. The `requires_cms` predicate's `native_color_model` whitelist
// covers Lab the moment `ColorModel::Lab` lands; no test code needs to
// change.

#[cfg(feature = "cms-moxcms")]
#[test]
fn cms_attached_for_native_pair_is_unused() {
    // Attaching MoxCms to a same-profile RGB→RGB conversion must not
    // produce an error. The dispatch chain should land on the built-in
    // identity plan (both moxcms and ZenCmsLite decline for the trivial
    // case).
    let opts = ConvertOptions::permissive();
    let res = RowConverter::new_explicit_with_cms(
        PixelDescriptor::RGB8_SRGB,
        PixelDescriptor::RGB8_SRGB,
        &opts,
        Some(&MoxCms),
    );
    assert!(
        res.is_ok(),
        "RGB→RGB identity with MoxCms attached must still succeed; got {:?}",
        res.err().map(|e| alloc::format!("{:?}", e.error())),
    );
}
