//! A colour context whose CICP leaves primaries or transfer unspecified (H.273
//! code 2) agrees with a descriptor that is Unknown there: layout and depth
//! conversions run and keep the context; colour conversions still refuse.
use std::sync::Arc;
use zenpixels_convert::{
    Cicp, ColorContext, ColorPrimaries, ConvertError, PixelBuffer, PixelBufferConvertExt as _,
    PixelDescriptor, PixelFormat, TransferFunction,
};

fn unspecified() -> Arc<ColorContext> {
    Arc::new(ColorContext::from_cicp(Cicp::new(2, 2, 0, true)))
}

fn buffer(desc: PixelDescriptor) -> PixelBuffer {
    let (w, h) = (4u32, 2u32);
    let bytes = (0..w as usize * h as usize * desc.bytes_per_pixel())
        .map(|i| (i * 37) as u8)
        .collect();
    PixelBuffer::from_vec(bytes, w, h, desc)
        .unwrap()
        .with_color_context(unspecified())
}

fn unknown(desc: PixelDescriptor) -> PixelDescriptor {
    desc.with_transfer(TransferFunction::Unknown)
        .with_primaries(ColorPrimaries::Unknown)
}

#[test]
fn layout_changes_run_and_keep_the_context() {
    for (from, to) in [
        (PixelDescriptor::GRAY8_SRGB, PixelDescriptor::RGBA8_SRGB),
        (PixelDescriptor::GRAY8_SRGB, PixelDescriptor::RGB8_SRGB),
        (PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB),
        (PixelDescriptor::RGBA8_SRGB, PixelDescriptor::RGB8_SRGB),
    ] {
        let src = buffer(unknown(from));
        let out = src
            .convert_to(unknown(to))
            .unwrap_or_else(|e| panic!("{from:?} -> {to:?}: {e:?}"));
        assert_eq!(out.descriptor().pixel_format(), to.pixel_format());
        assert_eq!(
            out.as_slice().color_context().and_then(|c| c.cicp),
            Some(Cicp::new(2, 2, 0, true)),
            "the unspecified claim travels with the pixels"
        );
    }
    // Gray to RGB replicates the sample.
    let src = buffer(unknown(PixelDescriptor::GRAY8_SRGB));
    let out = src.convert_to(unknown(PixelDescriptor::RGB8_SRGB)).unwrap();
    let (g, rgb) = (
        src.as_slice().row(0).to_vec(),
        out.as_slice().row(0).to_vec(),
    );
    for (x, v) in g.iter().enumerate() {
        assert_eq!(&rgb[x * 3..x * 3 + 3], &[*v, *v, *v]);
    }
}

#[test]
fn an_unspecified_claim_converts_like_no_claim_at_all() {
    // Whatever the library does for an Unknown-labelled buffer with no
    // context, it does the same with an all-unspecified context: same
    // success or failure, same descriptor, same bytes.
    for from in [PixelDescriptor::GRAY8_SRGB, PixelDescriptor::RGB8_SRGB] {
        for to in [
            PixelDescriptor::RGBA8_SRGB,
            unknown(PixelDescriptor::RGBA8_SRGB),
            PixelDescriptor::RGB16_SRGB,
            PixelDescriptor::RGBF32_LINEAR,
        ] {
            let with = buffer(unknown(from));
            let without = PixelBuffer::from_vec(
                with.as_slice().contiguous_bytes().into_owned(),
                with.width(),
                with.height(),
                unknown(from),
            )
            .unwrap();
            let a = with
                .convert_to(to)
                .map(|b| (b.descriptor(), b.as_slice().contiguous_bytes().into_owned()));
            let b = without
                .convert_to(to)
                .map(|b| (b.descriptor(), b.as_slice().contiguous_bytes().into_owned()));
            assert_eq!(a.is_ok(), b.is_ok(), "{from:?} -> {to:?}");
            if let (Ok(a), Ok(b)) = (a, b) {
                assert_eq!(a, b, "{from:?} -> {to:?}");
            }
        }
    }
}

#[test]
fn a_descriptor_that_claims_more_than_the_context_is_refused() {
    // Bt709 primaries in the descriptor, unspecified in the context.
    let src = buffer(PixelDescriptor::RGB8_SRGB.with_transfer(TransferFunction::Unknown));
    let e = src
        .convert_to(PixelDescriptor::RGBA8_SRGB.with_transfer(TransferFunction::Unknown))
        .unwrap_err();
    // The context's CICP primaries code 2 (unspecified) has no
    // `ColorPrimaries`; the checked projection (#81) reports that instead
    // of the generic `NoPath`.
    assert!(
        matches!(
            e.error(),
            ConvertError::CicpDescriptor(zenpixels::cicp::CicpDescriptorError::UnmappedPrimaries(
                2
            ))
        ),
        "{e:?}"
    );
    // And a non-zero matrix on an RGB buffer is a mislabel too.
    let bad = buffer(unknown(PixelDescriptor::RGB8_SRGB))
        .with_color_context(Arc::new(ColorContext::from_cicp(Cicp::new(2, 2, 1, true))));
    assert!(
        bad.convert_to(unknown(PixelDescriptor::RGBA8_SRGB))
            .is_err()
    );
    let _ = PixelFormat::Rgb8;
}
