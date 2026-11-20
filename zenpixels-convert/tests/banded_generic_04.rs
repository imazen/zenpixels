//! Integration tests for root `BandLayout` / `BandedPixelBuffer` re-exports,
//! sealed `PackedSource` / `PackedSink` band-generic APIs (`RowConverter::convert`,
//! `OutputPlan::convert_into`, `zenpixels_convert::orient_into`), and metadata/evidence
//! preservation across single-buffer, slice, and multi-band carriers.

use std::sync::Arc;
use zenpixels::sample::{ExactDepth, FloatFacts};
use zenpixels::{
    BandLayout, BandedPixelBuffer, Cicp, ColorContext, ColorOrigin, DEFAULT_MAX_BAND_BYTES,
    Orientation, PixelBuffer, PixelDescriptor, PixelFormat, RowAlign, SignalRange,
};
use zenpixels_convert::{
    BandedPixelBufferConvertExt, ConversionSpec, ConvertOptions, ConvertPlan, OutputPlan,
    OutputProfile, OutputTarget, PackedSink, PackedSource, orient_into,
};

fn assert_carriers_match(
    src: impl PackedSource,
    dst: impl PackedSink,
    worker: &mut zenpixels_convert::RowConverter,
) {
    worker.convert(src, dst).unwrap();
}

#[test]
fn root_reexports_and_all_carrier_combinations_in_row_converter_and_output_plan() {
    let layout =
        BandLayout::new(19, 13, PixelDescriptor::RGB8_SRGB, DEFAULT_MAX_BAND_BYTES).unwrap();
    assert_eq!(layout.band_count(), 1);

    let w = 19u32;
    let h = 13u32;
    let src_desc = PixelDescriptor::RGB8_SRGB;
    let dst_desc = PixelDescriptor::RGBF32_LINEAR;

    let mut contig_src = PixelBuffer::try_new(w, h, src_desc).unwrap();
    let mut banded_src = BandedPixelBuffer::try_new(w, h, src_desc, u64::from(w) * 3 * 4).unwrap();
    assert!(banded_src.layout().band_count() >= 4);
    for y in 0..h {
        let mut s_row = contig_src.rows_mut(y, 1);
        for (i, b) in s_row.row_mut(0).iter_mut().enumerate() {
            *b = ((y as usize * 29 + i * 7) & 0xFF) as u8;
        }
        banded_src
            .row_mut(y)
            .unwrap()
            .copy_from_slice(contig_src.as_slice().row(y));
    }

    let spec = ConversionSpec::new(ConvertOptions::retain_channels());
    let plan = ConvertPlan::build(src_desc.into(), dst_desc.into(), &spec).unwrap();
    let expected_facts = plan.output_float_facts();
    assert!(expected_facts.contains(
        FloatFacts::NO_NAN
            | FloatFacts::NO_INFINITE
            | FloatFacts::NO_SUBNORMALS
            | FloatFacts::NON_NEGATIVE
            | FloatFacts::AT_MOST_ONE
    ));
    let mut worker = plan.prepare(w).unwrap();

    // Reference via concrete convert_rows.
    let mut ref_dst = PixelBuffer::try_new(w, h, dst_desc).unwrap();
    worker
        .convert_rows(contig_src.as_slice(), ref_dst.as_slice_mut())
        .unwrap();

    // 1. (&PixelBuffer, &mut PixelBuffer)
    let mut dst_buf = PixelBuffer::try_new(w, h, dst_desc)
        .unwrap()
        .with_color_context(Arc::new(ColorContext::from_icc(&b"stale"[..])));
    assert_carriers_match(&contig_src, &mut dst_buf, &mut worker);
    assert!(dst_buf.color_context().is_none(), "stale context cleared");
    assert_eq!(dst_buf.float_facts(), expected_facts);
    for y in 0..h {
        assert_eq!(dst_buf.as_slice().row(y), ref_dst.as_slice().row(y));
    }

    // 2. (PixelSlice, &mut BandedPixelBuffer)
    let mut dst_banded = BandedPixelBuffer::try_new(w, h, dst_desc, u64::from(w) * 12 * 3).unwrap();
    assert!(dst_banded.layout().band_count() >= 5);
    worker
        .convert(contig_src.as_slice(), &mut dst_banded)
        .unwrap();
    assert_eq!(dst_banded.float_facts(), expected_facts);
    for y in 0..h {
        assert_eq!(dst_banded.row(y).unwrap(), ref_dst.as_slice().row(y));
    }

    // 3. (&BandedPixelBuffer, &mut PixelBuffer)
    let mut dst_from_banded = PixelBuffer::try_new(w, h, dst_desc).unwrap();
    worker.convert(&banded_src, &mut dst_from_banded).unwrap();
    assert_eq!(dst_from_banded.float_facts(), expected_facts);
    for y in 0..h {
        assert_eq!(dst_from_banded.as_slice().row(y), ref_dst.as_slice().row(y));
    }

    // 4. (&BandedPixelBuffer, &mut BandedPixelBuffer) with mismatched band heights (4 vs 3 rows)
    let mut dst_mismatched =
        BandedPixelBuffer::try_new(w, h, dst_desc, u64::from(w) * 12 * 3).unwrap();
    worker.convert(&banded_src, &mut dst_mismatched).unwrap();
    assert_eq!(dst_mismatched.float_facts(), expected_facts);
    for y in 0..h {
        assert_eq!(dst_mismatched.row(y).unwrap(), ref_dst.as_slice().row(y));
    }

    // 5. (&PixelSlice, &mut PixelSliceMut)
    let mut raw_buf = PixelBuffer::try_new(w, h, dst_desc).unwrap();
    {
        let src_slice = contig_src.as_slice();
        let mut dst_slice = raw_buf.as_slice_mut();
        worker.convert(&src_slice, &mut dst_slice).unwrap();
    }
    for y in 0..h {
        assert_eq!(raw_buf.as_slice().row(y), ref_dst.as_slice().row(y));
    }

    // 6. OutputPlan::convert_into across (&BandedPixelBuffer -> &mut BandedPixelBuffer)
    //    and (&BandedPixelBuffer -> &mut PixelBuffer)
    let origin = ColorOrigin::assumed();
    let mut out_plan = OutputPlan::build(
        src_desc.encoding(),
        OutputTarget::new(
            OutputProfile::Named(Cicp::new(1, 8, 0, true)),
            PixelFormat::RgbF32,
        )
        .with_origin(&origin),
        &spec,
    )
    .unwrap();
    out_plan.prepare(w).unwrap();

    let mut out_banded =
        BandedPixelBuffer::try_new(w, h, out_plan.output_descriptor(), u64::from(w) * 12 * 5)
            .unwrap();
    out_plan.convert_into(&banded_src, &mut out_banded).unwrap();
    assert!(Arc::ptr_eq(
        out_banded.color_context().unwrap(),
        out_plan.output_color_context()
    ));
    assert_eq!(out_banded.float_facts(), out_plan.output_float_facts());
    for y in 0..h {
        assert_eq!(out_banded.row(y).unwrap(), ref_dst.as_slice().row(y));
    }
}

#[test]
fn orient_into_all_orientations_mixed_bands_aligned_stride_and_evidence() {
    let w = 17u32;
    let h = 11u32;
    let desc = PixelDescriptor::RGB16_SRGB;
    let exact8 = ExactDepth::new(desc.channel_type(), SignalRange::Full, 8, 1).unwrap();
    let ctx = Arc::new(ColorContext::from_icc(&b"p3-icc"[..]));

    let mut contig_src = PixelBuffer::try_new(w, h, desc)
        .unwrap()
        .with_color_context(Arc::clone(&ctx));
    let mut banded_src = BandedPixelBuffer::try_new_with_align(w, h, desc, 512, RowAlign::Cache64)
        .unwrap()
        .with_color_context(Arc::clone(&ctx));
    assert!(banded_src.layout().band_count() >= 3);

    for y in 0..h {
        let mut row = contig_src.rows_mut(y, 1);
        for (x, chunk) in row.row_mut(0).chunks_exact_mut(6).enumerate() {
            let r = (((y * 13 + x as u32) & 0xFF) as u16 * 257).to_ne_bytes();
            let g = (((y * 7 + x as u32 * 3) & 0xFF) as u16 * 257).to_ne_bytes();
            let b = (((y + x as u32 * 11) & 0xFF) as u16 * 257).to_ne_bytes();
            chunk[0..2].copy_from_slice(&r);
            chunk[2..4].copy_from_slice(&g);
            chunk[4..6].copy_from_slice(&b);
        }
        banded_src
            .row_mut(y)
            .unwrap()
            .copy_from_slice(contig_src.as_slice().row(y));
    }
    contig_src.set_exact_depth(Some(exact8)).unwrap();
    banded_src.set_exact_depth(Some(exact8)).unwrap();

    for orientation in Orientation::ALL {
        let (ow, oh) = orientation.output_dimensions(w, h);

        // Banded -> Banded (Cache64 aligned) via both free `orient_into` and `BandedPixelBufferConvertExt`
        let mut b_to_b =
            BandedPixelBuffer::try_new_with_align(ow, oh, desc, 384, RowAlign::Cache64).unwrap();
        orient_into(&banded_src, orientation, &mut b_to_b).unwrap();
        assert_eq!(b_to_b.exact_depth(), Some(exact8));
        assert_eq!(
            b_to_b.color_context().and_then(|c| c.icc()),
            Some(&b"p3-icc"[..])
        );

        let mut b_ext = BandedPixelBuffer::try_new(ow, oh, desc, 320).unwrap();
        banded_src.oriented_into(orientation, &mut b_ext).unwrap();
        assert_eq!(b_ext.exact_depth(), Some(exact8));

        // Contiguous -> Banded
        let mut c_to_b = BandedPixelBuffer::try_new(ow, oh, desc, 320).unwrap();
        orient_into(&contig_src, orientation, &mut c_to_b).unwrap();
        assert_eq!(c_to_b.exact_depth(), Some(exact8));

        // Banded -> Contiguous
        let mut b_to_c = PixelBuffer::try_new(ow, oh, desc).unwrap();
        orient_into(&banded_src, orientation, &mut b_to_c).unwrap();
        assert_eq!(b_to_c.exact_depth(), Some(exact8));
        assert_eq!(
            b_to_c.color_context().and_then(|c| c.icc()),
            Some(&b"p3-icc"[..])
        );

        // Verify pixel-exact against forward_map oracle!
        let contig_slice = contig_src.as_slice();
        let b_to_c_slice = b_to_c.as_slice();
        for sy in 0..h {
            for sx in 0..w {
                let (dx, dy) = orientation.forward_map(sx, sy, w, h);
                let s_px = &contig_slice.row(sy)[sx as usize * 6..(sx as usize + 1) * 6];
                let d_range = dx as usize * 6..(dx as usize + 1) * 6;
                assert_eq!(&b_to_b.row(dy).unwrap()[d_range.clone()], s_px);
                assert_eq!(&b_ext.row(dy).unwrap()[d_range.clone()], s_px);
                assert_eq!(&c_to_b.row(dy).unwrap()[d_range.clone()], s_px);
                assert_eq!(&b_to_c_slice.row(dy)[d_range], s_px);
            }
        }
    }

    // Also check FloatFacts preservation through orient_into across Single <-> Banded!
    let f_desc = PixelDescriptor::RGBF32_LINEAR;
    let facts = FloatFacts::NO_NAN
        | FloatFacts::NO_INFINITE
        | FloatFacts::NO_SUBNORMALS
        | FloatFacts::NON_NEGATIVE
        | FloatFacts::AT_MOST_ONE;
    let mut f_src = BandedPixelBuffer::try_new(9, 7, f_desc, 9 * 12 * 3)
        .unwrap()
        .with_float_facts(facts)
        .unwrap();
    assert_eq!(f_src.float_facts(), facts);
    let mut f_dst_buf = PixelBuffer::try_new(7, 9, f_desc).unwrap();
    orient_into(&f_src, Orientation::Rotate90, &mut f_dst_buf).unwrap();
    assert_eq!(f_dst_buf.float_facts(), facts);

    // Clearing stale ColorContext on orient_into when src has None:
    f_src.set_color_context(None);
    let mut f_dst_banded = BandedPixelBuffer::try_new(7, 9, f_desc, 7 * 12 * 3)
        .unwrap()
        .with_color_context(Arc::new(ColorContext::from_icc(&b"stale"[..])));
    orient_into(&f_src, Orientation::Rotate90, &mut f_dst_banded).unwrap();
    assert!(f_dst_banded.color_context().is_none());
    assert_eq!(f_dst_banded.float_facts(), facts);
}
