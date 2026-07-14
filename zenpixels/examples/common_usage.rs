//! Common `zenpixels` usages, each self-checking.
//!
//! `zenpixels` is the *description* layer: it says what bytes are
//! ([`PixelFormat`]), what they mean ([`PixelDescriptor`]), and where they live
//! ([`PixelBuffer`] / [`PixelSlice`]). No conversion logic lives here — for that
//! reach for `zenpixels-convert` (see its `common_usage` example).
//!
//! Run the whole gallery:      `cargo run -p zenpixels --example common_usage`
//! Run it as tests:            `cargo test -p zenpixels --examples`
//!
//! [`PixelFormat`]: zenpixels::descriptor::PixelFormat
//! [`PixelDescriptor`]: zenpixels::descriptor::PixelDescriptor
//! [`PixelBuffer`]: zenpixels::buffer::PixelBuffer
//! [`PixelSlice`]: zenpixels::buffer::PixelSlice

use zenpixels::buffer::{Bgrx, PixelBuffer, PixelSlice, PixelSliceMut};
use zenpixels::cicp::Cicp;
use zenpixels::color::{ColorContext, NamedProfile};
use zenpixels::descriptor::{PixelDescriptor, PixelFormat, TransferFunction};
use zenpixels::orientation::Orientation;

fn main() {
    construct_owned_buffers();
    read_and_write_pixels();
    inspect_a_descriptor();
    crop_a_region();
    walk_strided_rows();
    swap_bgrx_to_rgbx_in_place();
    tag_color_metadata();
    describe_color_with_cicp();
    normalize_exif_orientation();
    println!("all zenpixels examples passed");
}

/// Build owned, zero-filled, SIMD-aligned pixel storage — three ways.
fn construct_owned_buffers() {
    // Use a `*_SRGB` preset so the color metadata (transfer = sRGB,
    // primaries = BT.709) travels with the pixels for free.
    let img = PixelBuffer::new(4, 2, PixelDescriptor::RGBA8_SRGB);
    assert_eq!((img.width(), img.height()), (4, 2));
    assert_eq!(img.descriptor().bytes_per_pixel(), 4);
    // stride is at least width * bytes-per-pixel; it may be padded for SIMD.
    assert!(img.stride() >= 4 * 4);

    // Wrap bytes you already have (e.g. straight from a decoder). The vec must
    // be tightly packed: `width * height * bytes_per_pixel`.
    let raw = vec![0u8; 4 * 2 * 3];
    let img =
        PixelBuffer::from_vec(raw, 4, 2, PixelDescriptor::RGB8_SRGB).expect("24 bytes == 4x2 RGB8");
    assert!(!img.has_alpha());

    // Untrusted dimensions? The fallible sibling returns an error instead of
    // panicking on a huge allocation.
    assert!(PixelBuffer::try_new(16, 16, PixelDescriptor::RGBA8).is_ok());
}

/// Read and write pixels through a mutable row view.
fn read_and_write_pixels() {
    let mut img = PixelBuffer::new(2, 1, PixelDescriptor::RGBA8_SRGB);
    {
        let mut view = img.as_slice_mut();
        let row = view.row_mut(0);
        row[0..4].copy_from_slice(&[255, 0, 0, 255]); // opaque red
        row[4..8].copy_from_slice(&[0, 255, 0, 255]); // opaque green
    }
    let view = img.as_slice();
    assert_eq!(&view.row(0)[0..4], &[255, 0, 0, 255]);
    assert_eq!(&view.row(0)[4..8], &[0, 255, 0, 255]);
}

/// Ask a descriptor what the bytes are, without touching the pixels.
fn inspect_a_descriptor() {
    let rgba = PixelFormat::Rgba8.descriptor();
    assert_eq!(rgba.channels(), 4);
    assert_eq!(rgba.bytes_per_pixel(), 4);
    assert!(rgba.has_alpha());
    assert!(!rgba.is_grayscale());
    assert!(!rgba.is_opaque()); // Rgba8's default alpha is Straight, not Opaque

    // Grayscale, no alpha channel => opaque by construction.
    assert!(PixelDescriptor::GRAY8.is_grayscale());
    assert!(PixelDescriptor::GRAY8.is_opaque());

    // Presets are just data — retag one axis with a `with_*` builder.
    let linear = PixelDescriptor::RGB16_SRGB.with_transfer(TransferFunction::Linear);
    assert_eq!(linear.transfer(), TransferFunction::Linear);
    assert!(linear.is_linear());
}

/// Crop a rectangle — borrowed (no copy) or owned.
fn crop_a_region() {
    let img = PixelBuffer::new(8, 8, PixelDescriptor::RGBA8_SRGB);

    // Borrow a sub-region: zero-copy, just an offset + the parent's stride.
    let view = img.crop_view(2, 2, 4, 4);
    assert_eq!((view.width(), view.rows()), (4, 4));

    // Take an owned, tightly-packed copy of the same region.
    let owned = img.crop_copy(2, 2, 4, 4);
    assert_eq!((owned.width(), owned.height()), (4, 4));
}

/// Strided rows are first-class: every row API excludes the padding.
fn walk_strided_rows() {
    let (width, rows) = (3u32, 2u32);
    let stride = 16usize; // 3*4 = 12 bytes of pixels + 4 bytes of padding per row
    let bytes = vec![0u8; stride * rows as usize];

    let slice = PixelSlice::new(&bytes, width, rows, stride, PixelDescriptor::RGBA8)
        .expect("stride >= width * bpp");
    assert_eq!(slice.stride(), 16);
    assert!(!slice.is_contiguous());
    for y in 0..slice.rows() {
        // `row` hands back exactly the pixel bytes — the 4 padding bytes are gone.
        assert_eq!(slice.row(y).len(), 3 * 4);
    }

    // For the common *packed* case, `new_tight` derives `width * bpp` for you —
    // no hand-computed stride (the single most-repeated line across the ecosystem).
    let packed = vec![0u8; width as usize * rows as usize * 4];
    let tight =
        PixelSlice::new_tight(&packed, width, rows, PixelDescriptor::RGBA8).expect("packed buffer");
    assert_eq!(tight.stride(), width as usize * 4);
    assert!(tight.is_contiguous());
}

/// BGRX -> RGBX is a pure channel swizzle: do it in place, no allocation.
fn swap_bgrx_to_rgbx_in_place() {
    // One BGRX pixel: b=10, g=20, r=30, x=40.
    let mut bytes = [10u8, 20, 30, 40];
    let slice = PixelSliceMut::new(&mut bytes, 1, 1, 4, PixelDescriptor::BGRX8).unwrap();
    let typed: PixelSliceMut<Bgrx> = slice.try_typed::<Bgrx>().expect("BGRX8 matches Bgrx");
    // swap_to_rgbx rewrites the bytes in place, then hands back an RGBX view.
    drop(typed.swap_to_rgbx());
    // Now stored as r=30, g=20, b=10, x=40.
    assert_eq!(bytes, [30, 20, 10, 40]);
}

/// Attach color metadata to a buffer with chained `with_*` builders.
fn tag_color_metadata() {
    // Start from a plain (untagged) RGBA8 buffer, then declare its color.
    let img = PixelBuffer::new(2, 2, PixelDescriptor::RGBA8)
        .with_primaries(zenpixels::descriptor::ColorPrimaries::DisplayP3)
        .with_transfer(TransferFunction::Srgb);
    assert_eq!(img.descriptor().transfer(), TransferFunction::Srgb);
    assert_eq!(
        img.descriptor().primaries,
        zenpixels::descriptor::ColorPrimaries::DisplayP3
    );

    // Or carry a full ICC / CICP context, `Arc`-shared so clones are cheap.
    let ctx = ColorContext::from_cicp(Cicp::DISPLAY_P3);
    assert!(!ctx.is_srgb());
    assert_eq!(ctx.transfer_function(), TransferFunction::Srgb);
}

/// `Cicp` is the compact ITU-T H.273 color signal; convert to/from descriptors.
fn describe_color_with_cicp() {
    // Named constants for the common signals.
    let srgb = Cicp::SRGB;
    assert_eq!(srgb.transfer_function_enum(), TransferFunction::Srgb);

    // Unpack a 4-byte CICP tuple straight from a container box (primaries,
    // transfer, matrix, full-range flag) — no `!= 0` papercut on the last byte.
    assert_eq!(Cicp::from_bytes([1, 13, 0, 1]), Cicp::SRGB);

    // A descriptor's color axes round-trip through CICP.
    let desc = Cicp::DISPLAY_P3.to_descriptor(PixelFormat::Rgba8);
    assert_eq!(
        desc.primaries,
        zenpixels::descriptor::ColorPrimaries::DisplayP3
    );
    assert_eq!(Cicp::from_descriptor(&desc), Some(Cicp::DISPLAY_P3));

    // Retag *only* the color axes of an existing descriptor from a decoded
    // CICP, keeping the format/type/alpha. Contrast with `to_descriptor`, which
    // builds a fresh descriptor from a `PixelFormat`.
    let hdr = PixelDescriptor::RGBA8_SRGB.with_color_from_cicp(Cicp::BT2100_PQ);
    assert_eq!(hdr.transfer(), TransferFunction::Pq);
    assert_eq!(hdr.primaries, zenpixels::descriptor::ColorPrimaries::Bt2020);

    // Named profiles bridge friendly names and CICP codes.
    assert_eq!(NamedProfile::Srgb.to_cicp(), Some(Cicp::SRGB));
    assert_eq!(
        NamedProfile::from_cicp(Cicp::BT2100_PQ),
        Some(NamedProfile::Bt2020Pq)
    );
}

/// Normalize an EXIF orientation tag to upright output dimensions.
fn normalize_exif_orientation() {
    // EXIF tag 6 = "rotate 90 CW": a 1920x1080 sensor frame displays as 1080x1920.
    let orient = Orientation::from_exif(6).expect("valid EXIF orientation");
    assert!(orient.swaps_axes());
    assert_eq!(orient.output_dimensions(1920, 1080), (1080, 1920));

    // Composing an orientation with its inverse is the identity.
    assert_eq!(orient.then(orient.inverse()), Orientation::Identity);
    assert!(Orientation::Identity.is_identity());
}

#[test]
fn examples_run() {
    main();
}
