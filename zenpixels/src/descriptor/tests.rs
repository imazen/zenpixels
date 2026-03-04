use core::mem::size_of;

use super::*;

#[test]
fn channel_type_byte_size() {
    assert_eq!(ChannelType::U8.byte_size(), 1);
    assert_eq!(ChannelType::U16.byte_size(), 2);
    assert_eq!(ChannelType::F16.byte_size(), 2);
    assert_eq!(ChannelType::F32.byte_size(), 4);
}

#[test]
fn descriptor_bytes_per_pixel() {
    assert_eq!(PixelDescriptor::RGB8.bytes_per_pixel(), 3);
    assert_eq!(PixelDescriptor::RGBA8.bytes_per_pixel(), 4);
    assert_eq!(PixelDescriptor::GRAY8.bytes_per_pixel(), 1);
    assert_eq!(PixelDescriptor::RGBAF32.bytes_per_pixel(), 16);
    assert_eq!(PixelDescriptor::GRAYA8.bytes_per_pixel(), 2);
}

#[test]
fn descriptor_has_alpha() {
    assert!(!PixelDescriptor::RGB8.has_alpha());
    assert!(PixelDescriptor::RGBA8.has_alpha());
    assert!(!PixelDescriptor::RGBX8.has_alpha());
    assert!(PixelDescriptor::GRAYA8.has_alpha());
}

#[test]
fn descriptor_is_grayscale() {
    assert!(PixelDescriptor::GRAY8.is_grayscale());
    assert!(PixelDescriptor::GRAYA8.is_grayscale());
    assert!(!PixelDescriptor::RGB8.is_grayscale());
}

#[test]
fn layout_compatible() {
    assert!(PixelDescriptor::RGB8_SRGB.layout_compatible(PixelDescriptor::RGB8));
    assert!(!PixelDescriptor::RGB8.layout_compatible(PixelDescriptor::RGBA8));
}

#[test]
fn pixel_format_descriptor_roundtrip() {
    let desc = PixelFormat::Rgba8.descriptor();
    assert_eq!(desc.layout(), ChannelLayout::Rgba);
    assert_eq!(desc.channel_type(), ChannelType::U8);
}

#[test]
fn pixel_format_enum_basics() {
    assert_eq!(PixelFormat::Rgb8.channels(), 3);
    assert_eq!(PixelFormat::Rgba8.channels(), 4);
    assert!(PixelFormat::Rgba8.has_alpha_bytes());
    assert!(!PixelFormat::Rgb8.has_alpha_bytes());
    assert_eq!(PixelFormat::RgbF32.bytes_per_pixel(), 12);
    assert_eq!(PixelFormat::RgbaF32.bytes_per_pixel(), 16);
    assert_eq!(PixelFormat::Gray8.channels(), 1);
    assert!(PixelFormat::Gray8.is_grayscale());
    assert!(!PixelFormat::Rgb8.is_grayscale());
    assert_eq!(PixelFormat::Bgra8.byte_order(), ByteOrder::Bgr);
    assert_eq!(PixelFormat::Rgb8.byte_order(), ByteOrder::Native);
}

#[test]
fn pixel_format_enum_size() {
    // Single-byte discriminant — much smaller than old 5-field struct.
    assert!(size_of::<PixelFormat>() <= 2);
}

#[test]
fn pixel_format_from_parts_roundtrip() {
    let fmt = PixelFormat::Rgba8;
    let rebuilt = PixelFormat::from_parts(fmt.channel_type(), fmt.layout(), fmt.default_alpha());
    assert_eq!(rebuilt, Some(fmt));

    let fmt2 = PixelFormat::Bgra8;
    let rebuilt2 =
        PixelFormat::from_parts(fmt2.channel_type(), fmt2.layout(), fmt2.default_alpha());
    assert_eq!(rebuilt2, Some(fmt2));

    let fmt3 = PixelFormat::Gray8;
    let rebuilt3 =
        PixelFormat::from_parts(fmt3.channel_type(), fmt3.layout(), fmt3.default_alpha());
    assert_eq!(rebuilt3, Some(fmt3));
}

#[test]
fn alpha_mode_semantics() {
    // None (Option) = no alpha channel
    assert!(!PixelDescriptor::RGB8.has_alpha());
    // Undefined = padding bytes, not real alpha
    assert!(!AlphaMode::Undefined.has_alpha());
    // Straight and Premultiplied = real alpha
    assert!(AlphaMode::Straight.has_alpha());
    assert!(AlphaMode::Premultiplied.has_alpha());
    assert!(AlphaMode::Opaque.has_alpha());
}

#[test]
fn color_primaries_containment() {
    assert!(ColorPrimaries::Bt2020.contains(ColorPrimaries::DisplayP3));
    assert!(ColorPrimaries::Bt2020.contains(ColorPrimaries::Bt709));
    assert!(ColorPrimaries::DisplayP3.contains(ColorPrimaries::Bt709));
    assert!(!ColorPrimaries::Bt709.contains(ColorPrimaries::DisplayP3));
    assert!(!ColorPrimaries::Unknown.contains(ColorPrimaries::Bt709));
}

#[test]
fn descriptor_size() {
    // PixelFormat (1 byte enum) + transfer (1) + alpha (2) + primaries (1) + signal_range (1) = ~6
    assert!(size_of::<PixelDescriptor>() <= 8);
}

#[test]
fn color_model_channels() {
    assert_eq!(ColorModel::Gray.color_channels(), 1);
    assert_eq!(ColorModel::Rgb.color_channels(), 3);
    assert_eq!(ColorModel::YCbCr.color_channels(), 3);
    assert_eq!(ColorModel::Oklab.color_channels(), 3);
}

#[cfg(feature = "planar")]
#[test]
fn subsampling_factors() {
    assert_eq!(Subsampling::S444.h_factor(), 1);
    assert_eq!(Subsampling::S444.v_factor(), 1);
    assert_eq!(Subsampling::S422.h_factor(), 2);
    assert_eq!(Subsampling::S422.v_factor(), 1);
    assert_eq!(Subsampling::S420.h_factor(), 2);
    assert_eq!(Subsampling::S420.v_factor(), 2);
    assert_eq!(Subsampling::S411.h_factor(), 4);
    assert_eq!(Subsampling::S411.v_factor(), 1);
}

#[cfg(feature = "planar")]
#[test]
fn yuv_matrix_cicp() {
    assert_eq!(YuvMatrix::from_cicp(1), Some(YuvMatrix::Bt709));
    assert_eq!(YuvMatrix::from_cicp(5), Some(YuvMatrix::Bt601));
    assert_eq!(YuvMatrix::from_cicp(9), Some(YuvMatrix::Bt2020));
    assert_eq!(YuvMatrix::from_cicp(99), None);
}

#[cfg(feature = "planar")]
#[test]
fn subsampling_from_factors() {
    assert_eq!(Subsampling::from_factors(1, 1), Some(Subsampling::S444));
    assert_eq!(Subsampling::from_factors(2, 1), Some(Subsampling::S422));
    assert_eq!(Subsampling::from_factors(2, 2), Some(Subsampling::S420));
    assert_eq!(Subsampling::from_factors(4, 1), Some(Subsampling::S411));
    assert_eq!(Subsampling::from_factors(3, 1), None);
    assert_eq!(Subsampling::from_factors(1, 2), None);
}

// --- PlaneSemantic tests ---

#[cfg(feature = "planar")]
#[test]
fn plane_semantic_classification() {
    // Luminance-like
    assert!(PlaneSemantic::Luma.is_luminance());
    assert!(PlaneSemantic::Gray.is_luminance());
    assert!(PlaneSemantic::OklabL.is_luminance());
    assert!(!PlaneSemantic::Red.is_luminance());

    // Chroma
    assert!(PlaneSemantic::ChromaCb.is_chroma());
    assert!(PlaneSemantic::ChromaCr.is_chroma());
    assert!(PlaneSemantic::OklabA.is_chroma());
    assert!(PlaneSemantic::OklabB.is_chroma());
    assert!(!PlaneSemantic::Luma.is_chroma());

    // RGB
    assert!(PlaneSemantic::Red.is_rgb());
    assert!(PlaneSemantic::Green.is_rgb());
    assert!(PlaneSemantic::Blue.is_rgb());
    assert!(!PlaneSemantic::Luma.is_rgb());

    // Alpha
    assert!(PlaneSemantic::Alpha.is_alpha());
    assert!(!PlaneSemantic::Luma.is_alpha());
    assert!(!PlaneSemantic::Depth.is_alpha());
}

#[cfg(feature = "planar")]
#[test]
fn plane_semantic_display() {
    assert_eq!(format!("{}", PlaneSemantic::Luma), "Luma");
    assert_eq!(format!("{}", PlaneSemantic::ChromaCb), "Cb");
    assert_eq!(format!("{}", PlaneSemantic::Gray), "Gray");
    assert_eq!(format!("{}", PlaneSemantic::OklabL), "Oklab.L");
}

// --- PlaneDescriptor tests ---

#[cfg(feature = "planar")]
#[test]
fn plane_descriptor_full_resolution() {
    let d = PlaneDescriptor::new(PlaneSemantic::Luma, ChannelType::F32);
    assert_eq!(d.semantic, PlaneSemantic::Luma);
    assert_eq!(d.channel_type, ChannelType::F32);
    assert!(!d.is_subsampled());
    assert_eq!(d.h_subsample, 1);
    assert_eq!(d.v_subsample, 1);
    assert_eq!(d.plane_width(1920), 1920);
    assert_eq!(d.plane_height(1080), 1080);
    assert_eq!(d.bytes_per_sample(), 4);
}

#[cfg(feature = "planar")]
#[test]
fn plane_descriptor_subsampled() {
    let d = PlaneDescriptor::new(PlaneSemantic::ChromaCb, ChannelType::U8).with_subsampling(2, 2);
    assert!(d.is_subsampled());
    assert_eq!(d.plane_width(1920), 960);
    assert_eq!(d.plane_height(1080), 540);
    // Odd dimensions use ceiling division
    assert_eq!(d.plane_width(1921), 961);
    assert_eq!(d.plane_height(1081), 541);
    assert_eq!(d.bytes_per_sample(), 1);
}

#[cfg(feature = "planar")]
#[test]
fn plane_descriptor_quarter_h() {
    let d = PlaneDescriptor::new(PlaneSemantic::ChromaCr, ChannelType::U16).with_subsampling(4, 1);
    assert!(d.is_subsampled());
    assert_eq!(d.plane_width(1920), 480);
    assert_eq!(d.plane_height(1080), 1080);
    assert_eq!(d.bytes_per_sample(), 2);
}

#[cfg(feature = "planar")]
#[test]
fn plane_descriptor_display() {
    let d = PlaneDescriptor::new(PlaneSemantic::Luma, ChannelType::F32);
    assert_eq!(format!("{d}"), "Luma:F32");

    let d = PlaneDescriptor::new(PlaneSemantic::ChromaCb, ChannelType::U8).with_subsampling(2, 2);
    assert_eq!(format!("{d}"), "Cb:U8 (1/2\u{00d7}1/2)");
}

#[cfg(feature = "planar")]
#[test]
fn plane_descriptor_size() {
    // Should be small — semantic (1) + channel_type (1) + h (1) + v (1) = 4
    assert!(size_of::<PlaneDescriptor>() <= 4);
}

// --- PlaneMask tests ---

#[cfg(feature = "planar")]
#[test]
fn plane_mask_constants() {
    assert!(PlaneMask::ALL.includes(0));
    assert!(PlaneMask::ALL.includes(7));
    assert!(!PlaneMask::NONE.includes(0));
    assert!(PlaneMask::NONE.is_empty());
    assert!(PlaneMask::LUMA.includes(0));
    assert!(!PlaneMask::LUMA.includes(1));
    assert!(PlaneMask::CHROMA.includes(1));
    assert!(PlaneMask::CHROMA.includes(2));
    assert!(!PlaneMask::CHROMA.includes(0));
    assert!(PlaneMask::ALPHA.includes(3));
    assert!(!PlaneMask::ALPHA.includes(0));
}

#[cfg(feature = "planar")]
#[test]
fn plane_mask_single() {
    let m = PlaneMask::single(5);
    assert!(m.includes(5));
    assert!(!m.includes(4));
    assert_eq!(m.count(), 1);
}

#[cfg(feature = "planar")]
#[test]
fn plane_mask_union_intersection() {
    let luma_alpha = PlaneMask::LUMA.union(PlaneMask::ALPHA);
    assert!(luma_alpha.includes(0));
    assert!(luma_alpha.includes(3));
    assert!(!luma_alpha.includes(1));
    assert_eq!(luma_alpha.count(), 2);

    let intersect = luma_alpha.intersection(PlaneMask::LUMA);
    assert!(intersect.includes(0));
    assert!(!intersect.includes(3));
    assert_eq!(intersect.count(), 1);
}

#[cfg(feature = "planar")]
#[test]
fn plane_mask_out_of_range() {
    assert!(!PlaneMask::ALL.includes(8));
    assert!(!PlaneMask::ALL.includes(100));
}

#[cfg(feature = "planar")]
#[test]
fn plane_mask_bits_roundtrip() {
    let m = PlaneMask::LUMA.union(PlaneMask::CHROMA);
    let bits = m.bits();
    assert_eq!(PlaneMask::from_bits(bits), m);
}

#[cfg(feature = "planar")]
#[test]
fn plane_mask_display() {
    assert_eq!(format!("{}", PlaneMask::ALL), "ALL");
    assert_eq!(format!("{}", PlaneMask::NONE), "NONE");
    assert_eq!(format!("{}", PlaneMask::LUMA), "0");
    assert_eq!(
        format!("{}", PlaneMask::LUMA.union(PlaneMask::ALPHA)),
        "0|3"
    );
}

// --- PlaneLayout tests ---

#[cfg(feature = "planar")]
#[test]
fn plane_layout_interleaved() {
    let layout = PlaneLayout::Interleaved { channels: 4 };
    assert!(!layout.is_planar());
    assert_eq!(layout.plane_count(), 4);
    assert!(layout.planes().is_empty());
    assert!(!layout.has_subsampling());
    assert!(!layout.is_ycbcr());
    assert!(!layout.is_oklab());
    assert_eq!(layout.luma_plane_index(), None);
}

#[cfg(feature = "planar")]
#[test]
fn plane_layout_ycbcr_444() {
    let layout = PlaneLayout::ycbcr_444(ChannelType::F32);
    assert!(layout.is_planar());
    assert!(layout.is_ycbcr());
    assert!(!layout.is_oklab());
    assert_eq!(layout.plane_count(), 3);
    assert!(!layout.has_subsampling());
    assert_eq!(layout.luma_plane_index(), Some(0));
    assert_eq!(layout.max_v_subsample(), 1);
    assert_eq!(layout.max_h_subsample(), 1);

    let planes = layout.planes();
    assert_eq!(planes[0].semantic, PlaneSemantic::Luma);
    assert_eq!(planes[1].semantic, PlaneSemantic::ChromaCb);
    assert_eq!(planes[2].semantic, PlaneSemantic::ChromaCr);
    assert_eq!(planes[0].channel_type, ChannelType::F32);
}

#[cfg(feature = "planar")]
#[test]
fn plane_layout_ycbcr_420() {
    let layout = PlaneLayout::ycbcr_420(ChannelType::U8);
    assert!(layout.is_planar());
    assert!(layout.is_ycbcr());
    assert!(layout.has_subsampling());
    assert_eq!(layout.max_v_subsample(), 2);
    assert_eq!(layout.max_h_subsample(), 2);

    let planes = layout.planes();
    assert!(!planes[0].is_subsampled());
    assert!(planes[1].is_subsampled());
    assert_eq!(planes[1].h_subsample, 2);
    assert_eq!(planes[1].v_subsample, 2);
}

#[cfg(feature = "planar")]
#[test]
fn plane_layout_ycbcr_422() {
    let layout = PlaneLayout::ycbcr_422(ChannelType::U8);
    assert!(layout.has_subsampling());
    assert_eq!(layout.max_v_subsample(), 1);
    assert_eq!(layout.max_h_subsample(), 2);

    let planes = layout.planes();
    assert_eq!(planes[1].h_subsample, 2);
    assert_eq!(planes[1].v_subsample, 1);
}

#[cfg(feature = "planar")]
#[test]
fn plane_layout_ycbcr_444_matrix() {
    let layout = PlaneLayout::ycbcr_444_matrix(ChannelType::U8, YuvMatrix::Bt709);
    assert_eq!(
        layout.relationship(),
        Some(PlaneRelationship::YCbCr {
            matrix: YuvMatrix::Bt709
        })
    );
}

#[cfg(feature = "planar")]
#[test]
fn plane_layout_oklab() {
    let layout = PlaneLayout::oklab(ChannelType::F32);
    assert!(layout.is_planar());
    assert!(layout.is_oklab());
    assert!(!layout.is_ycbcr());
    assert_eq!(layout.plane_count(), 3);
    assert!(!layout.has_subsampling());
    assert_eq!(layout.luma_plane_index(), Some(0));

    let planes = layout.planes();
    assert_eq!(planes[0].semantic, PlaneSemantic::OklabL);
    assert_eq!(planes[1].semantic, PlaneSemantic::OklabA);
    assert_eq!(planes[2].semantic, PlaneSemantic::OklabB);
    assert_eq!(planes[0].channel_type, ChannelType::F32);
}

#[cfg(feature = "planar")]
#[test]
fn plane_layout_oklab_alpha() {
    let layout = PlaneLayout::oklab_alpha(ChannelType::F32);
    assert!(layout.is_oklab());
    assert_eq!(layout.plane_count(), 4);
    assert_eq!(layout.planes()[3].semantic, PlaneSemantic::Alpha);
}

#[cfg(feature = "planar")]
#[test]
fn plane_layout_rgb() {
    let layout = PlaneLayout::rgb(ChannelType::U8);
    assert!(layout.is_planar());
    assert!(!layout.is_ycbcr());
    assert!(!layout.is_oklab());
    assert_eq!(layout.plane_count(), 3);
    assert_eq!(layout.relationship(), Some(PlaneRelationship::Independent));

    let planes = layout.planes();
    assert_eq!(planes[0].semantic, PlaneSemantic::Red);
    assert_eq!(planes[1].semantic, PlaneSemantic::Green);
    assert_eq!(planes[2].semantic, PlaneSemantic::Blue);
}

#[cfg(feature = "planar")]
#[test]
fn plane_layout_rgba() {
    let layout = PlaneLayout::rgba(ChannelType::U8);
    assert_eq!(layout.plane_count(), 4);
    assert_eq!(layout.planes()[3].semantic, PlaneSemantic::Alpha);
}

#[cfg(feature = "planar")]
#[test]
fn plane_layout_gray() {
    let layout = PlaneLayout::gray(ChannelType::U8);
    assert!(layout.is_planar());
    assert_eq!(layout.plane_count(), 1);
    assert_eq!(layout.planes()[0].semantic, PlaneSemantic::Gray);
    assert_eq!(layout.luma_plane_index(), Some(0));
}

#[cfg(feature = "planar")]
#[test]
fn plane_layout_display() {
    let layout = PlaneLayout::Interleaved { channels: 3 };
    assert_eq!(format!("{layout}"), "Interleaved(3ch)");

    let layout = PlaneLayout::oklab(ChannelType::F32);
    let s = format!("{layout}");
    assert!(s.starts_with("Oklab["), "got: {s}");
    assert!(s.contains("Oklab.L:F32"), "got: {s}");
}

// --- MultiPlaneImage tests ---

#[cfg(feature = "planar")]
#[test]
fn multi_plane_image_basic() {
    let layout = PlaneLayout::ycbcr_444(ChannelType::U8);
    let y = crate::buffer::PixelBuffer::new(64, 64, PixelDescriptor::GRAY8);
    let cb = crate::buffer::PixelBuffer::new(64, 64, PixelDescriptor::GRAY8);
    let cr = crate::buffer::PixelBuffer::new(64, 64, PixelDescriptor::GRAY8);

    let img = MultiPlaneImage::new(layout, alloc::vec![y, cb, cr]);
    assert_eq!(img.plane_count(), 3);
    assert!(img.layout().is_ycbcr());
    assert!(img.buffer(0).is_some());
    assert!(img.buffer(2).is_some());
    assert!(img.buffer(3).is_none());
    assert!(img.origin().is_none());
}

#[cfg(feature = "planar")]
#[test]
fn multi_plane_image_with_origin() {
    let layout = PlaneLayout::gray(ChannelType::U8);
    let buf = crate::buffer::PixelBuffer::new(32, 32, PixelDescriptor::GRAY8);

    let ctx = alloc::sync::Arc::new(crate::color::ColorContext::from_cicp(
        crate::cicp::Cicp::new(1, 13, 0, false),
    ));
    let img = MultiPlaneImage::new(layout, alloc::vec![buf]).with_origin(ctx.clone());
    assert!(img.origin().is_some());
}

// --- PlaneRelationship tests ---

#[cfg(feature = "planar")]
#[test]
fn plane_relationship_variants() {
    let r = PlaneRelationship::Oklab;
    assert_eq!(r, PlaneRelationship::Oklab);

    let r = PlaneRelationship::YCbCr {
        matrix: YuvMatrix::Bt709,
    };
    assert_ne!(r, PlaneRelationship::Independent);

    // Copy
    let r2 = r;
    assert_eq!(r, r2);
}

#[test]
fn reference_white_nits_values() {
    assert_eq!(TransferFunction::Pq.reference_white_nits(), 203.0);
    assert_eq!(TransferFunction::Srgb.reference_white_nits(), 1.0);
    assert_eq!(TransferFunction::Hlg.reference_white_nits(), 1.0);
    assert_eq!(TransferFunction::Linear.reference_white_nits(), 1.0);
    assert_eq!(TransferFunction::Unknown.reference_white_nits(), 1.0);
}

// --- PlaneLayout mask tests ---

#[cfg(feature = "planar")]
#[test]
fn mask_where_oklab() {
    let layout = PlaneLayout::oklab_alpha(ChannelType::F32);
    let luma = layout.luma_mask();
    assert!(luma.includes(0));
    assert!(!luma.includes(1));
    assert_eq!(luma.count(), 1);

    let chroma = layout.chroma_mask();
    assert!(chroma.includes(1));
    assert!(chroma.includes(2));
    assert!(!chroma.includes(0));
    assert_eq!(chroma.count(), 2);

    let alpha = layout.alpha_mask();
    assert!(alpha.includes(3));
    assert_eq!(alpha.count(), 1);
}

#[cfg(feature = "planar")]
#[test]
fn mask_where_ycbcr_420() {
    let layout = PlaneLayout::ycbcr_420(ChannelType::U8);
    let luma = layout.luma_mask();
    assert!(luma.includes(0));
    assert_eq!(luma.count(), 1);

    let chroma = layout.chroma_mask();
    assert!(chroma.includes(1));
    assert!(chroma.includes(2));
    assert_eq!(chroma.count(), 2);

    let alpha = layout.alpha_mask();
    assert!(alpha.is_empty());
}

#[cfg(feature = "planar")]
#[test]
fn mask_where_gray() {
    let layout = PlaneLayout::gray(ChannelType::U8);
    let luma = layout.luma_mask();
    assert!(luma.includes(0));
    assert_eq!(luma.count(), 1);
    assert!(layout.chroma_mask().is_empty());
    assert!(layout.alpha_mask().is_empty());
}

#[cfg(feature = "planar")]
#[test]
fn mask_where_interleaved_returns_none() {
    let layout = PlaneLayout::Interleaved { channels: 4 };
    assert!(layout.luma_mask().is_empty());
    assert!(layout.chroma_mask().is_empty());
    assert!(layout.alpha_mask().is_empty());
}
