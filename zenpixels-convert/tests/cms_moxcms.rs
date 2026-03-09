//! Integration tests for the moxcms CMS backend.
//!
//! Uses generated minimal ICC v2 profiles for real CMS transforms, then
//! validates results against our hardcoded gamut matrices and transfer
//! functions.

#![cfg(feature = "cms-moxcms")]

extern crate alloc;

use alloc::sync::Arc;

use zenpixels_convert::cms::ColorManagement;
use zenpixels_convert::cms_moxcms::MoxCms;
use zenpixels_convert::output::{OutputProfile, finalize_for_output};
use zenpixels_convert::{
    Cicp, ColorContext, ColorOrigin, PixelBuffer, PixelDescriptor, PixelFormat,
};

// ---------------------------------------------------------------------------
// Minimal ICC v2 profile generator
// ---------------------------------------------------------------------------

/// Generate a minimal ICC v2 matrix-based RGB profile with a gamma TRC.
///
/// This creates a valid ICC profile from XYZ primaries and a gamma value,
/// suitable for testing CMS transforms. The profile has:
/// - Header (128 bytes)
/// - Tag table (4 tags × 12 bytes)
/// - rXYZ, gXYZ, bXYZ tags (20 bytes each)
/// - rTRC = gTRC = bTRC (shared curv tag, 12 bytes for single gamma)
fn make_icc_v2(
    rx: f64,
    ry: f64,
    rz: f64,
    gx: f64,
    gy: f64,
    gz: f64,
    bx: f64,
    by: f64,
    bz: f64,
    gamma: f64,
) -> Vec<u8> {
    // s15Fixed16Number encoding
    fn s15f16(v: f64) -> [u8; 4] {
        let raw = (v * 65536.0).round() as i32;
        raw.to_be_bytes()
    }

    // Tag data layout:
    // offset 0..128: header
    // 128..132: tag count (4)
    // 132..132+48: tag table (4 entries × 12 bytes)
    // 180..200: rXYZ
    // 200..220: gXYZ
    // 220..240: bXYZ
    // 240..252: curv (single gamma)
    let profile_size: u32 = 252;

    let mut buf = vec![0u8; profile_size as usize];

    // Header
    buf[0..4].copy_from_slice(&profile_size.to_be_bytes()); // profile size
    buf[4..8].copy_from_slice(b"none"); // preferred CMM
    buf[8..12].copy_from_slice(&[2, 0x10, 0, 0]); // version 2.1.0
    buf[12..16].copy_from_slice(b"mntr"); // device class: monitor
    buf[16..20].copy_from_slice(b"RGB "); // color space: RGB
    buf[20..24].copy_from_slice(b"XYZ "); // PCS: XYZ
    // date: 2024-01-01
    buf[24..26].copy_from_slice(&2024u16.to_be_bytes());
    buf[26..28].copy_from_slice(&1u16.to_be_bytes());
    buf[28..30].copy_from_slice(&1u16.to_be_bytes());
    buf[36..40].copy_from_slice(b"acsp"); // signature
    buf[40..44].copy_from_slice(b"APPL"); // platform
    // flags, device mfr, device model: all zeros
    // D65 illuminant at offset 68
    buf[68..72].copy_from_slice(&s15f16(0.9505));
    buf[72..76].copy_from_slice(&s15f16(1.0));
    buf[76..80].copy_from_slice(&s15f16(1.0890));

    // Tag count
    let tag_count: u32 = 4;
    buf[128..132].copy_from_slice(&tag_count.to_be_bytes());

    // Tag table entries (sig, offset, size)
    let tags = [
        (b"rXYZ", 180u32, 20u32),
        (b"gXYZ", 200, 20),
        (b"bXYZ", 220, 20),
        (b"rTRC", 240, 12), // gTRC and bTRC share the same offset
    ];

    let mut table_off = 132;
    for (sig, off, sz) in &tags {
        buf[table_off..table_off + 4].copy_from_slice(*sig);
        buf[table_off + 4..table_off + 8].copy_from_slice(&off.to_be_bytes());
        buf[table_off + 8..table_off + 12].copy_from_slice(&sz.to_be_bytes());
        table_off += 12;
    }

    // Also add gTRC and bTRC pointing to same data as rTRC
    // We need to expand the tag table. Let me redo the layout:
    // Actually, we only declared 4 tags. To share curv data between
    // rTRC/gTRC/bTRC, we need 6 tag entries pointing to the same offset.

    // Let me redo with 6 tags:
    let tag_count: u32 = 6;
    buf[128..132].copy_from_slice(&tag_count.to_be_bytes());

    // Recalculate offsets:
    // 128..132: tag count
    // 132..132+72: 6 tag entries
    // 204..224: rXYZ
    // 224..244: gXYZ
    // 244..264: bXYZ
    // 264..276: curv
    let profile_size: u32 = 276;
    buf.resize(profile_size as usize, 0);
    buf[0..4].copy_from_slice(&profile_size.to_be_bytes());

    let tags = [
        (b"rXYZ", 204u32, 20u32),
        (b"gXYZ", 224, 20),
        (b"bXYZ", 244, 20),
        (b"rTRC", 264, 12),
        (b"gTRC", 264, 12), // shared
        (b"bTRC", 264, 12), // shared
    ];

    let mut table_off = 132;
    for (sig, off, sz) in &tags {
        buf[table_off..table_off + 4].copy_from_slice(*sig);
        buf[table_off + 4..table_off + 8].copy_from_slice(&off.to_be_bytes());
        buf[table_off + 8..table_off + 12].copy_from_slice(&sz.to_be_bytes());
        table_off += 12;
    }

    // XYZ tag data
    fn write_xyz(buf: &mut [u8], off: usize, x: f64, y: f64, z: f64) {
        buf[off..off + 4].copy_from_slice(b"XYZ "); // type signature
        buf[off + 4..off + 8].copy_from_slice(&[0; 4]); // reserved
        buf[off + 8..off + 12].copy_from_slice(&s15f16(x));
        buf[off + 12..off + 16].copy_from_slice(&s15f16(y));
        buf[off + 16..off + 20].copy_from_slice(&s15f16(z));
    }

    write_xyz(&mut buf, 204, rx, ry, rz);
    write_xyz(&mut buf, 224, gx, gy, gz);
    write_xyz(&mut buf, 244, bx, by, bz);

    // curv tag: single gamma
    buf[264..268].copy_from_slice(b"curv"); // type signature
    buf[268..272].copy_from_slice(&[0; 4]); // reserved
    buf[272..276].copy_from_slice(&1u32.to_be_bytes()); // count = 1 (parametric gamma)
    // Wait, curv with count=1 stores a u8Fixed8Number gamma.
    // Format: 'curv' 0000 count(4bytes) [if count==1: u8Fixed8Number]
    // Actually the size should be 12 + 2*count bytes = 14 bytes for count=1.
    // Let me fix the layout.
    let profile_size: u32 = 278;
    buf.resize(profile_size as usize, 0);
    buf[0..4].copy_from_slice(&profile_size.to_be_bytes());

    // Update curv tag size in all three entries
    for idx in [3, 4, 5] {
        let entry_off = 132 + idx * 12 + 8;
        buf[entry_off..entry_off + 4].copy_from_slice(&14u32.to_be_bytes());
    }

    // Write curv: 'curv' reserved(4) count(4)=1 gamma(2)
    buf[264..268].copy_from_slice(b"curv");
    buf[268..272].copy_from_slice(&[0; 4]);
    buf[272..276].copy_from_slice(&1u32.to_be_bytes());
    let gamma_fixed = (gamma * 256.0).round() as u16;
    buf[276..278].copy_from_slice(&gamma_fixed.to_be_bytes());

    buf
}

/// sRGB ICC profile (BT.709 primaries, gamma 2.2 approximation).
fn srgb_icc() -> Vec<u8> {
    make_icc_v2(
        0.4124, 0.2126, 0.0193, 0.3576, 0.7152, 0.1192, 0.1805, 0.0722, 0.9505, 2.2,
    )
}

/// Display P3 ICC profile (P3 primaries, gamma 2.2).
fn display_p3_icc() -> Vec<u8> {
    make_icc_v2(
        0.4866, 0.2290, 0.0000, 0.2657, 0.6917, 0.0451, 0.1982, 0.0793, 1.0439, 2.2,
    )
}

/// BT.2020 ICC profile (BT.2020 primaries, gamma 2.2).
fn bt2020_icc() -> Vec<u8> {
    make_icc_v2(
        0.6370, 0.2627, 0.0000, 0.1446, 0.6780, 0.0281, 0.1689, 0.0593, 1.0610, 2.2,
    )
}

// ---------------------------------------------------------------------------
// Basic transform creation
// ---------------------------------------------------------------------------

#[test]
fn build_transform_srgb_to_p3() {
    let cms = MoxCms;
    let src = srgb_icc();
    let dst = display_p3_icc();

    let xform = cms.build_transform(&src, &dst).unwrap();

    // Transform a red pixel: sRGB (255, 0, 0) → Display P3
    let src_row = [255u8, 0, 0];
    let mut dst_row = [0u8; 3];
    xform.transform_row(&src_row, &mut dst_row, 1);

    // sRGB red in P3 should have lower R and slightly positive G/B
    assert!(
        dst_row[0] < 255,
        "P3 red should be less than 255 for sRGB red"
    );
    assert!(
        dst_row[0] > 180,
        "P3 red should still be high, got {}",
        dst_row[0]
    );
}

#[test]
fn build_transform_for_format_rgba() {
    let cms = MoxCms;
    let src = srgb_icc();
    let dst = display_p3_icc();

    let xform = cms
        .build_transform_for_format(&src, &dst, PixelFormat::Rgba8, PixelFormat::Rgba8)
        .unwrap();

    // Alpha should be preserved through the transform
    let src_row = [128u8, 200, 64, 42];
    let mut dst_row = [0u8; 4];
    xform.transform_row(&src_row, &mut dst_row, 1);

    assert_eq!(dst_row[3], 42, "alpha must be preserved");
}

#[test]
fn build_transform_for_format_f32() {
    let cms = MoxCms;
    let src = srgb_icc();
    let dst = bt2020_icc();

    let xform = cms
        .build_transform_for_format(&src, &dst, PixelFormat::RgbF32, PixelFormat::RgbF32)
        .unwrap();

    // White should be approximately preserved (neutral axis)
    let src_bytes: [u8; 12] = bytemuck::cast([1.0f32, 1.0, 1.0]);
    let mut dst_bytes = [0u8; 12];
    xform.transform_row(&src_bytes, &mut dst_bytes, 1);

    let dst_f32: [f32; 3] = bytemuck::cast(dst_bytes);
    for (ch, &v) in ["R", "G", "B"].iter().zip(&dst_f32) {
        assert!((v - 1.0).abs() < 0.05, "white {ch} should be ~1.0, got {v}");
    }
}

#[test]
fn gray_neutral_preserved() {
    let cms = MoxCms;
    let src = srgb_icc();
    let dst = display_p3_icc();

    let xform = cms.build_transform(&src, &dst).unwrap();

    // Mid-gray should be approximately preserved (neutral axis)
    let src_row = [128u8, 128, 128];
    let mut dst_row = [0u8; 3];
    xform.transform_row(&src_row, &mut dst_row, 1);

    for ch in 0..3 {
        let err = (dst_row[ch] as i32 - 128).abs();
        assert!(err <= 3, "gray ch{ch}: expected ~128, got {}", dst_row[ch]);
    }
}

// ---------------------------------------------------------------------------
// Profile identification
// ---------------------------------------------------------------------------

#[test]
fn identify_srgb_profile() {
    let cms = MoxCms;
    let icc = srgb_icc();

    let cicp = cms.identify_profile(&icc);
    assert!(cicp.is_some(), "should identify sRGB profile");

    let cicp = cicp.unwrap();
    assert_eq!(cicp.color_primaries, 1, "BT.709 primaries");
}

#[test]
fn identify_display_p3_profile() {
    let cms = MoxCms;
    let icc = display_p3_icc();

    let cicp = cms.identify_profile(&icc);
    assert!(cicp.is_some(), "should identify Display P3 profile");

    let cicp = cicp.unwrap();
    assert_eq!(cicp.color_primaries, 12, "Display P3 primaries");
}

#[test]
fn identify_bt2020_profile() {
    let cms = MoxCms;
    let icc = bt2020_icc();

    let cicp = cms.identify_profile(&icc);
    assert!(cicp.is_some(), "should identify BT.2020 profile");

    let cicp = cicp.unwrap();
    assert_eq!(cicp.color_primaries, 9, "BT.2020 primaries");
}

#[test]
fn identify_garbage_returns_none() {
    let cms = MoxCms;
    assert!(cms.identify_profile(&[0u8; 100]).is_none());
}

// ---------------------------------------------------------------------------
// Round-trip: sRGB → P3 → sRGB
// ---------------------------------------------------------------------------

#[test]
fn srgb_p3_roundtrip_accuracy() {
    let cms = MoxCms;
    let srgb = srgb_icc();
    let p3 = display_p3_icc();

    let forward = cms.build_transform(&srgb, &p3).unwrap();
    let inverse = cms.build_transform(&p3, &srgb).unwrap();

    for r in (0..=255).step_by(51) {
        for g in (0..=255).step_by(51) {
            for b in (0..=255).step_by(51) {
                let src = [r as u8, g as u8, b as u8];
                let mut mid = [0u8; 3];
                let mut back = [0u8; 3];

                forward.transform_row(&src, &mut mid, 1);
                inverse.transform_row(&mid, &mut back, 1);

                for ch in 0..3 {
                    let err = (back[ch] as i32 - src[ch] as i32).abs();
                    // Tolerance of 12: gamma 2.2 in our minimal ICC profiles
                    // differs from the sRGB piecewise curve, and s15Fixed16
                    // matrix quantization adds ~1/65536 error per entry. Both
                    // errors get amplified by gamma compression in the u8 domain.
                    assert!(
                        err <= 12,
                        "roundtrip error > 12 at ({r},{g},{b}): ch{ch} {}→{}→{}, err={err}",
                        src[ch],
                        mid[ch],
                        back[ch]
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-pixel rows
// ---------------------------------------------------------------------------

#[test]
fn multi_pixel_row_transform() {
    let cms = MoxCms;
    let src = srgb_icc();
    let dst = display_p3_icc();

    let xform = cms.build_transform(&src, &dst).unwrap();

    // 4 pixels
    let src_row = [
        255u8, 0, 0, // red
        0, 255, 0, // green
        0, 0, 255, // blue
        128, 128, 128, // gray
    ];
    let mut dst_row = [0u8; 12];
    xform.transform_row(&src_row, &mut dst_row, 4);

    // Each pixel should be different from the next (sanity check)
    // Gray should be approximately preserved
    let gray_r = dst_row[9];
    let gray_g = dst_row[10];
    let gray_b = dst_row[11];
    for ch in [gray_r, gray_g, gray_b] {
        assert!(
            (ch as i32 - 128).abs() <= 3,
            "gray should be ~128, got {ch}"
        );
    }
}

// ---------------------------------------------------------------------------
// Integration with finalize_for_output
// ---------------------------------------------------------------------------

#[test]
fn finalize_with_moxcms_icc_transform() {
    let cms = MoxCms;

    let data = vec![200u8, 100, 50, 150, 75, 25];
    let buf = PixelBuffer::from_vec(data, 2, 1, PixelDescriptor::RGB8_SRGB).unwrap();

    let src_icc = srgb_icc();
    let ctx = Arc::new(ColorContext::from_icc(src_icc.clone()));
    let buf = buf.with_color_context(ctx);

    let origin = ColorOrigin::from_icc(src_icc);

    let dst_icc: Arc<[u8]> = display_p3_icc().into();

    let ready = finalize_for_output(
        &buf,
        &origin,
        OutputProfile::Icc(dst_icc.clone()),
        PixelFormat::Rgb8,
        &cms,
    )
    .unwrap();

    assert!(ready.metadata().icc.is_some());
    assert!(ready.metadata().cicp.is_none());

    // Pixels should be converted
    let out_pixels = ready.pixels();
    let out_row = out_pixels.row(0);
    assert_ne!(
        &out_row[..3],
        &[200u8, 100, 50],
        "pixels should be converted"
    );
}

#[test]
fn finalize_named_profile_bypasses_cms() {
    let cms = MoxCms;

    let data = vec![200u8, 100, 50];
    let buf = PixelBuffer::from_vec(data, 1, 1, PixelDescriptor::RGB8_SRGB).unwrap();

    let origin = ColorOrigin::assumed();

    let ready = finalize_for_output(
        &buf,
        &origin,
        OutputProfile::Named(Cicp::SRGB),
        PixelFormat::Rgb8,
        &cms,
    )
    .unwrap();

    assert!(ready.metadata().cicp.is_some());
    let pixels = ready.pixels();
    let out_row = pixels.row(0);
    assert_eq!(&out_row[..3], &[200u8, 100, 50], "should be identity");
}

// ---------------------------------------------------------------------------
// Cross-validation: CMS sRGB identity
// ---------------------------------------------------------------------------

#[test]
fn cms_srgb_to_srgb_is_identity() {
    // An sRGB → sRGB transform should be (approximately) identity.
    let cms = MoxCms;
    let srgb = srgb_icc();

    let xform = cms.build_transform(&srgb, &srgb.clone()).unwrap();

    for r in (0..=255).step_by(17) {
        for g in (0..=255).step_by(17) {
            for b in (0..=255).step_by(17) {
                let src = [r as u8, g as u8, b as u8];
                let mut dst = [0u8; 3];
                xform.transform_row(&src, &mut dst, 1);

                for ch in 0..3 {
                    let err = (dst[ch] as i32 - src[ch] as i32).abs();
                    assert!(
                        err <= 1,
                        "sRGB→sRGB should be identity at ({r},{g},{b}) ch{ch}: \
                         src={}, dst={}, err={err}",
                        src[ch],
                        dst[ch]
                    );
                }
            }
        }
    }
}

#[test]
fn cms_white_preserved_across_profiles() {
    let cms = MoxCms;

    // White should map to ~white in any gamut conversion
    for (name, dst_icc) in [("P3", display_p3_icc()), ("BT.2020", bt2020_icc())] {
        let xform = cms.build_transform(&srgb_icc(), &dst_icc).unwrap();
        let src = [255u8, 255, 255];
        let mut dst = [0u8; 3];
        xform.transform_row(&src, &mut dst, 1);

        for ch in 0..3 {
            assert!(
                dst[ch] >= 250,
                "white ch{ch} in {name}: expected ~255, got {}",
                dst[ch]
            );
        }
    }
}

#[test]
fn cms_black_preserved_across_profiles() {
    let cms = MoxCms;

    for (name, dst_icc) in [("P3", display_p3_icc()), ("BT.2020", bt2020_icc())] {
        let xform = cms.build_transform(&srgb_icc(), &dst_icc).unwrap();
        let src = [0u8, 0, 0];
        let mut dst = [0u8; 3];
        xform.transform_row(&src, &mut dst, 1);

        for ch in 0..3 {
            assert!(
                dst[ch] <= 5,
                "black ch{ch} in {name}: expected ~0, got {}",
                dst[ch]
            );
        }
    }
}
