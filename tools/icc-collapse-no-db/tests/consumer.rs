#![forbid(unsafe_code)]
//! Standalone feature-isolation consumer tests (C01, C02, C03, C05).
//!
//! Built with `zenpixels` (`default-features = false, features = ["icc"]`) and
//! `zenpixels-convert` (`default-features = false, features = ["rgb"]`) — NO `icc-db`,
//! NO `bt2446a`, NO `hdr-transfers`, NO `gainmap`, NO `cms-*`, NO `std` on `zenpixels`!

use std::sync::Arc;
use zenpixels::{
    AlphaMode, Cicp, ColorSpace, PixelBuf, PixelFormat, PixelView, RenderingIntent, SampleLayout,
};
use zenpixels_convert::icc_profiles::{ADOBE_RGB, DISPLAY_P3_V4, SynthesizedIcc};
use zenpixels_convert::{
    AlphaCompaction, ConvertPolicy, DiffuseWhite, Negotiate, Nits, RowPlan, analyze_compaction,
    compact_in_place,
};

#[test]
fn c05_no_db_synthesis_distinguishes_compact_db_only_and_intrinsically_unsupported() {
    // 1. sRGB default -> NotNeeded even without icc-db.
    assert_eq!(
        SynthesizedIcc::for_cicp(Cicp::SRGB),
        SynthesizedIcc::NotNeeded
    );

    // 2. Qualified compact profiles (Display P3 / Rec.2020) -> Profile without icc-db!
    assert!(matches!(
        SynthesizedIcc::for_cicp(Cicp::DISPLAY_P3),
        SynthesizedIcc::Profile(_)
    ));
    assert!(matches!(
        SynthesizedIcc::for_cicp(Cicp::new(9, 1, 0, true)),
        SynthesizedIcc::Profile(_)
    ));

    // 3. Assigned DB-only pair (BT.2020 PQ) -> IccDbDisabled without icc-db!
    assert_eq!(
        SynthesizedIcc::for_cicp(Cicp::BT2100_PQ_YCBCR_FULL),
        SynthesizedIcc::IccDbDisabled
    );

    // 4. Intrinsically unsupported pairs (XYZ CP=10, HLG TC=18, reserved/unassigned TC=255, CP=99)
    //    MUST return CmsUnsupported even without icc-db (never IccDbDisabled)!
    assert_eq!(
        SynthesizedIcc::for_cicp(Cicp::new(10, 8, 0, true)),
        SynthesizedIcc::Unsupported
    );
    assert_eq!(
        SynthesizedIcc::for_cicp(Cicp::BT2100_HLG_YCBCR_FULL),
        SynthesizedIcc::Unsupported
    );
    assert_eq!(
        SynthesizedIcc::for_cicp(Cicp::new(1, 255, 0, true)),
        SynthesizedIcc::Unsupported
    );
    assert_eq!(
        SynthesizedIcc::for_cicp(Cicp::new(99, 13, 0, true)),
        SynthesizedIcc::Unsupported
    );

    // 5. Gray without icc-db: sRGB -> NotNeeded; assigned DB-only -> IccDbDisabled; reserved -> CmsUnsupported!
    assert_eq!(
        SynthesizedIcc::for_gray_cicp(Cicp::SRGB),
        SynthesizedIcc::NotNeeded
    );
    assert_eq!(
        SynthesizedIcc::for_gray_cicp(Cicp::DISPLAY_P3),
        SynthesizedIcc::IccDbDisabled
    );
    assert_eq!(
        SynthesizedIcc::for_gray_cicp(Cicp::new(9, 16, 0, true)),
        SynthesizedIcc::IccDbDisabled
    );
    assert_eq!(
        SynthesizedIcc::for_gray_cicp(Cicp::new(1, 255, 0, true)),
        SynthesizedIcc::Unsupported
    );
    assert_eq!(
        SynthesizedIcc::for_gray_cicp(Cicp::BT2100_HLG_YCBCR_FULL),
        SynthesizedIcc::Unsupported
    );
}

#[test]
fn c03_load_bearing_reduction_without_icc_db_preserves_icc_and_drops_opaque_alpha() {
    let p3_rgba = PixelFormat::new(
        SampleLayout::RGBA8,
        ColorSpace::Icc {
            profile: Arc::from(DISPLAY_P3_V4),
            color_channels: 3,
            alpha: AlphaMode::Straight,
            intent: RenderingIntent::RelativeColorimetric,
        },
    );
    let row = [64u8, 64, 64, 255, 128, 128, 128, 255];
    let mut padded = Vec::new();
    for _ in 0..2 {
        padded.extend_from_slice(&row);
        padded.extend_from_slice(&[0xa5; 4]);
    }
    let view = PixelView::new(&padded, 2, 2, 12, &p3_rgba).unwrap();
    let report = analyze_compaction(&view);
    assert_eq!(report.requires_alpha, Some(false));
    assert_eq!(report.requires_chroma, Some(false));
    // Must drop opaque alpha (RGBA8 -> RGB8) while keeping RGB ICC (cannot drop to Gray8 without Gray ICC!).
    let reduced_format = report.candidate_format(&p3_rgba);
    assert_eq!(reduced_format.layout, SampleLayout::RGB8);
    assert!(matches!(
        reduced_format.color,
        ColorSpace::Icc {
            color_channels: 3,
            alpha: AlphaMode::None,
            ..
        }
    ));

    let mut buf = PixelBuf::from_bytes(row.repeat(2), 2, 2, p3_rgba).unwrap();
    assert!(compact_in_place(&mut buf, AlphaCompaction::DropUnusedLanes).unwrap().is_compacted());
    assert_eq!(buf.format().layout, SampleLayout::RGB8);
    assert_eq!(buf.as_view().row(0), &[64, 64, 64, 128, 128, 128]);
    assert_eq!(buf.as_view().row(1), &[64, 64, 64, 128, 128, 128]);

    let _ = ADOBE_RGB;
}

#[test]
fn c03_b05_negotiate_without_bt2446a_never_elects_unbuildable_hdr_sdr_candidate() {
    let pq = PixelFormat::RGB16_BT2100_PQ;
    let srgb8 = PixelFormat::RGB8_SRGB;
    let pol = ConvertPolicy::display_sdr(DiffuseWhite::BT2408, Nits::new_positive(203.0).unwrap());
    assert!(
        RowPlan::build(&pq, &srgb8, &pol, Some(Nits::new_positive(1000.0).unwrap()),).is_err(),
        "without bt2446a or hdr-transfers, PQ->sRGB8 must fail cleanly"
    );
    assert!(
        Negotiate::select(&pq, &[srgb8], &pol).is_none(),
        "Negotiate::select must not offer an unbuildable PQ->sRGB8 candidate without bt2446a!"
    );
}
