use std::sync::Arc;
use whereat::At;
use zenpixels::precision::AlphaContent;
use zenpixels::sample::ExactDepth;
use zenpixels::{AlphaMode, Orientation, ParametricColor, RowAlign};
use zenpixels_convert::{
    AlphaPolicy, BandedPixelBufferConvertExt, ByteBound, CandidateTarget, ChannelLayout,
    ChannelType, Cicp, ColorContext, ColorPrimaries, ColorProfileSource, ConversionSpec,
    ConvertError, ConvertOptions, ConvertPlan, DepthPolicy, EncodeReady, GamutPolicy,
    LoadBearingReport, OutputPlan, OutputProfile, OutputTarget, PixelBuffer, PixelBufferConvertExt,
    PixelDescriptor, PixelEncoding, PixelFormat, PixelSliceConvertExt, TransferFunction,
    WorkExtent,
    banded::BandedPixelBuffer,
    cms::{CmsPluginError, CmsRequest, CmsSupport, PluggableCms, RowTransform},
};

const RGB: PixelDescriptor = PixelDescriptor::RGB8_SRGB;
const RGBA: PixelDescriptor = PixelDescriptor::RGBA8_SRGB;

#[test]
fn negotiation_returns_an_executable_policy_admitted_plan() {
    let drop = CandidateTarget::new(RGB.into());
    let retain = CandidateTarget::new(RGBA.into());
    let options = ConvertOptions::retain_channels().with_alpha_policy(AlphaPolicy::DiscardIfOpaque);
    let spec = ConversionSpec::new(options);
    let selected = ConvertPlan::select(RGBA.into(), &[drop, retain], &spec).unwrap();
    assert_eq!(selected.candidate_index(), 1);
    let mut worker = selected.into_plan().prepare(1).unwrap();
    let mut dst = [0u8; 4];
    worker
        .try_convert_row(&[20, 10, 8, 6], &mut dst, 1)
        .unwrap();
    assert_eq!(dst, [20, 10, 8, 6]);
    let e = ConvertPlan::select(RGBA.into(), &[drop], &spec).unwrap_err();
    assert!(matches!(
        e.error(),
        ConvertError::NoCandidate { index: 0, cause }
            if matches!(cause.as_ref(), ConvertError::AlphaCheckRequired)
    ));
}

#[test]
fn bidirectional_unknown_transfer_refusal_and_explicit_relabel_composition() {
    let unknown = RGB.with_transfer(TransferFunction::Unknown);
    let spec = ConversionSpec::new(ConvertOptions::retain_channels());

    // Both Unknown -> Known and Known -> Unknown refuse without authority.
    assert!(matches!(
        ConvertPlan::build(unknown.into(), RGB.into(), &spec)
            .unwrap_err()
            .error(),
        ConvertError::UnknownTransferRelabelForbidden { .. }
    ));
    assert!(matches!(
        ConvertPlan::build(RGB.into(), unknown.into(), &spec)
            .unwrap_err()
            .error(),
        ConvertError::UnknownTransferRelabelForbidden { .. }
    ));

    // Verified CICP on source unlocks Unknown -> Known directly.
    let verified = ColorContext::from_cicp(Cicp::SRGB);
    let source = PixelEncoding::new(unknown, Some(&verified));
    let plan = ConvertPlan::build(source, RGB.into(), &spec).unwrap();
    assert!(plan.is_identity());
    assert_eq!(plan.output_encoding().descriptor(), RGB);

    // Explicit ConvertPlan::relabel_transfer bridges Unknown <-> Known and composes via then().
    let relabel = ConvertPlan::relabel_transfer(unknown, RGB).unwrap();
    assert!(relabel.is_identity());
    let to_lin =
        ConvertPlan::build(RGB.into(), PixelDescriptor::RGBF32_LINEAR.into(), &spec).unwrap();
    let composed = relabel.then(to_lin).unwrap();
    let mut worker = composed.prepare(1).unwrap();
    let mut out = [0.0f32; 3];
    worker
        .try_convert_row(&[255, 0, 0], bytemuck::cast_slice_mut(&mut out), 1)
        .unwrap();
    assert!((out[0] - 1.0).abs() < 1e-6 && out[1].abs() < 1e-6);

    // relabel_transfer rejects non-Unknown pairs and layout/primaries changes.
    assert!(
        ConvertPlan::relabel_transfer(RGB, RGB.with_transfer(TransferFunction::Bt709)).is_err()
    );
    assert!(ConvertPlan::relabel_transfer(unknown, RGBA).is_err());
}

#[test]
fn workspace_reports_exact_scratch_across_zero_one_two_and_three_operations() {
    use zenpixels_convert::Accelerators;

    let rgb16 = PixelDescriptor::RGB16_SRGB;
    let rgba16 = PixelDescriptor::RGBA16_SRGB;
    // Universal scalar executes every native path in a single pass with zero scratch.
    for (from, to) in [(RGB, RGB), (RGB, rgb16), (RGB, rgba16)] {
        let scalar_plan = ConvertPlan::build(
            from.into(),
            to.into(),
            &ConversionSpec::preserve_samples().with_accelerators(&Accelerators::SCALAR),
        )
        .unwrap();
        assert_eq!(
            scalar_plan
                .workspace(WorkExtent::new(5, u32::MAX))
                .unwrap()
                .worker_heap(),
            ByteBound::Exact(0)
        );
        let mut worker = None;
        let count = allocation_counter::measure(|| {
            worker = Some(scalar_plan.prepare(5).unwrap());
        });
        assert_eq!(count.bytes_total, 0);
        drop(worker);
    }

    // Multi-step SIMD pipeline reports exact scratch across 0, 1, 2, and 3 operations.
    let fast_exact = ConversionSpec::preserve_samples().with_accelerators(&Accelerators::ALL);
    for (from, to, expected) in [(RGB, RGB, 0), (RGB, rgb16, 0), (RGB, rgba16, 20)] {
        let plan = ConvertPlan::build(from.into(), to.into(), &fast_exact).unwrap();
        let needed = plan.workspace(WorkExtent::new(5, u32::MAX)).unwrap();
        assert_eq!(needed.worker_heap(), ByteBound::Exact(expected));
        let mut worker = None;
        let count = allocation_counter::measure(|| {
            worker = Some(plan.prepare(5).unwrap());
        });
        assert_eq!(count.bytes_total as u64, expected);
        drop(worker);
    }

    // Fused 1-step LUT + 3-step native parity after cold shared table warmup.
    let opts = ConvertOptions::retain_channels()
        .with_alpha_policy(AlphaPolicy::DiscardUnchecked)
        .with_depth_policy(DepthPolicy::Round)
        .with_gamut_policy(GamutPolicy::Clip);
    let spec = ConversionSpec::fast(opts);
    let p3_u8 = RGB.with_primaries(ColorPrimaries::DisplayP3);
    let lut_plan = ConvertPlan::build(p3_u8.into(), RGB.into(), &spec).unwrap();
    let _warm = lut_plan.prepare_shared(1).unwrap();
    let cloned_lut = lut_plan.try_clone().unwrap();
    let c_lut = allocation_counter::measure(|| {
        let _w = cloned_lut.prepare(256).unwrap();
    });
    assert_eq!(c_lut.bytes_total, 0);

    // Deduplicated shared LUT bytes when two stages reference the same static table.
    let back_plan = ConvertPlan::build(RGB.into(), p3_u8.into(), &spec).unwrap();
    let single_cold = lut_plan
        .workspace(WorkExtent::new(16, 1))
        .unwrap()
        .shared_cold_setup();
    let chained = lut_plan.then(back_plan).unwrap();
    assert_eq!(
        chained
            .workspace(WorkExtent::new(16, 1))
            .unwrap()
            .shared_cold_setup(),
        single_cold,
        "SharedLutMask must deduplicate shared LUT references across stages"
    );

    let bgr8 = PixelDescriptor::new(
        ChannelType::U8,
        ChannelLayout::Bgr,
        None,
        TransferFunction::Srgb,
    );
    let p3_rgba = PixelDescriptor::RGBAF32_LINEAR.with_primaries(ColorPrimaries::DisplayP3);
    let plan3 = ConvertPlan::build(bgr8.into(), p3_rgba.into(), &spec).unwrap();
    let _warm3 = plan3.prepare_shared(1).unwrap();
    let ByteBound::Exact(expected3) = plan3
        .workspace(WorkExtent::new(256, 1))
        .unwrap()
        .worker_heap()
    else {
        panic!("expected exact worker heap");
    };
    assert!(expected3 > 0);
    let c3 = allocation_counter::measure(|| {
        let _w = plan3.prepare(256).unwrap();
    });
    assert_eq!(c3.bytes_total as u64, expected3);
}

#[test]
fn band_zipper_conversion_precision_scan_and_opacity_proof_across_mismatched_bands() {
    let (w, h) = (83u32, 59u32);
    let dst_desc = PixelDescriptor::RGB16_SRGB.with_primaries(ColorPrimaries::DisplayP3);

    let mut src_contig = PixelBuffer::try_new(w, h, RGBA).unwrap();
    // 5-row source bands vs 7-row destination bands!
    let mut src_banded = BandedPixelBuffer::try_new(w, h, RGBA, (w as u64) * 4 * 5).unwrap();
    let mut dst_banded = BandedPixelBuffer::try_new(w, h, dst_desc, (w as u64) * 6 * 7).unwrap();
    let mut dst_contig = PixelBuffer::try_new(w, h, dst_desc).unwrap();

    for y in 0..h {
        let mut row = vec![0u8; (w as usize) * 4];
        for px in 0..(w as usize) {
            row[px * 4] = ((px * 7 + y as usize * 13) & 0xFF) as u8;
            row[px * 4 + 1] = ((px * 19 + y as usize * 3) & 0xFF) as u8;
            row[px * 4 + 2] = ((px * 31 + y as usize * 23) & 0xFF) as u8;
            row[px * 4 + 3] = 255;
        }
        src_contig.as_slice_mut().row_mut(y).copy_from_slice(&row);
        src_banded.row_mut(y).unwrap().copy_from_slice(&row);
    }

    src_banded.check_opaque().unwrap();
    let prec = src_banded
        .scan_source_precision(&enough::Unstoppable)
        .unwrap();
    assert_eq!(prec.alpha(), AlphaContent::Opaque);

    // Measured AlphaContent::Opaque unlocks AlphaPolicy::DiscardIfOpaque in ConvertPlan::build!
    let opts = ConvertOptions::retain_channels()
        .with_alpha_policy(AlphaPolicy::DiscardIfOpaque)
        .with_depth_policy(DepthPolicy::Forbid);
    let spec = ConversionSpec::new(opts).with_source_precision(prec);
    let plan = ConvertPlan::build(src_banded.encoding(), dst_desc.into(), &spec).unwrap();

    plan.prepare_shared(w)
        .unwrap()
        .convert_rows(src_contig.as_slice(), dst_contig.as_slice_mut())
        .unwrap();
    plan.prepare(w)
        .unwrap()
        .convert_bands(&src_banded, &mut dst_banded)
        .unwrap();

    for y in 0..h {
        assert_eq!(
            dst_banded.row(y).unwrap(),
            dst_contig.as_slice().row(y),
            "row {y}"
        );
    }

    // Final-pixel translucency in the last band is caught and rejected!
    src_banded.row_mut(h - 1).unwrap()[(w as usize - 1) * 4 + 3] = 254;
    assert_eq!(
        *src_banded.check_opaque().unwrap_err().error(),
        ConvertError::AlphaNotOpaque
    );
    let trans_prec = src_banded
        .scan_source_precision(&enough::Unstoppable)
        .unwrap();
    assert_eq!(trans_prec.alpha(), AlphaContent::Translucent);
    assert_eq!(
        *ConvertPlan::build(
            src_banded.encoding(),
            dst_desc.into(),
            &ConversionSpec::new(opts).with_source_precision(trans_prec)
        )
        .unwrap_err()
        .error(),
        ConvertError::AlphaNotOpaque
    );

    // Contradictory Opaque descriptor + Translucent SourcePrecision refuses!
    let rgbx_opaque = RGBA.with_alpha(Some(AlphaMode::Opaque));
    assert!(
        ConvertPlan::build(
            rgbx_opaque.into(),
            RGB.into(),
            &ConversionSpec::new(opts).with_source_precision(trans_prec),
        )
        .is_err()
    );

    let report = LoadBearingReport::scan_bands(&src_banded);
    assert_eq!(report.uses_alpha, Some(true));
    assert_eq!(report.uses_chroma, Some(true));
}

#[test]
fn banded_orientation_matches_contiguous_across_all_8_exif_orientations_and_preserves_metadata() {
    for desc in [
        PixelDescriptor::RGB8_SRGB,
        PixelDescriptor::RGBA8_SRGB,
        PixelDescriptor::RGB16_SRGB,
        PixelDescriptor::RGBF32_LINEAR,
    ] {
        let (w, h) = (73u32, 53u32);
        let bpp = desc.bytes_per_pixel();
        let mut contig = PixelBuffer::try_new(w, h, desc).unwrap();
        let mut banded = BandedPixelBuffer::try_new_with_align(
            w,
            h,
            desc,
            (w as u64) * (bpp as u64) * 11,
            RowAlign::Tight,
        )
        .unwrap();

        for y in 0..h {
            let mut row = vec![0u8; (w as usize) * bpp];
            for (i, b) in row.iter_mut().enumerate() {
                *b = ((y as usize * 97 + i * 37 + 11) & 0xFF) as u8;
            }
            contig.as_slice_mut().row_mut(y).copy_from_slice(&row);
            banded.row_mut(y).unwrap().copy_from_slice(&row);
        }
        let ctx = Arc::new(ColorContext::from_cicp(Cicp::SRGB));
        banded.set_color_context(Some(ctx.clone()));
        if desc == PixelDescriptor::RGB16_SRGB {
            let ed =
                ExactDepth::new(ChannelType::U16, zenpixels::SignalRange::Full, 16, 16).unwrap();
            banded.set_exact_depth(Some(ed)).unwrap();
        }

        for orient in Orientation::ALL {
            let want = contig.as_slice().oriented(orient).unwrap();
            let (ow, oh) = orient.output_dimensions(w, h);
            let mut got = BandedPixelBuffer::try_new_with_align(
                ow,
                oh,
                desc,
                (ow as u64) * (bpp as u64) * 9,
                RowAlign::Cache64,
            )
            .unwrap();
            assert!(got.layout().band_count() > 1);
            banded.oriented_into(orient, &mut got).unwrap();
            assert_eq!(got.color_context(), Some(&ctx));
            if desc == PixelDescriptor::RGB16_SRGB {
                assert!(got.exact_depth().is_some());
            }
            for y in 0..oh {
                assert_eq!(
                    got.row(y).unwrap(),
                    want.as_slice().row(y),
                    "{desc:?} {orient:?} row {y}"
                );
            }
        }
    }
}

#[test]
#[ignore = "allocates 2.16 GB (> i32::MAX) live gigapixel buffer; run explicitly with --ignored"]
fn live_over_one_billion_pixel_and_over_i32_max_banded_buffer_allocates_addresses_and_scans() {
    let w = 40_000u32;
    let h = 27_000u32;
    let desc = PixelDescriptor::new(
        ChannelType::U8,
        ChannelLayout::GrayAlpha,
        Some(AlphaMode::Straight),
        TransferFunction::Srgb,
    );
    let mut giga = BandedPixelBuffer::try_new(w, h, desc, 32 * 1024 * 1024)
        .expect("allocate 1.08-gigapixel / 2.16 GB BandedPixelBuffer");
    assert_eq!(giga.layout().pixel_count(), 1_080_000_000u64);
    let total_payload = giga.layout().pixel_count() * (desc.bytes_per_pixel() as u64);
    assert_eq!(total_payload, 2_160_000_000u64);
    assert!(total_payload > i32::MAX as u64);

    let boundary_y = giga.layout().band_rows();
    giga.row_mut(0).unwrap()[0..2].copy_from_slice(&[0x12, 0xFF]);
    giga.row_mut(0).unwrap()[2..4].copy_from_slice(&[0x34, 0x80]);
    giga.row_mut(boundary_y).unwrap()[19_999 * 2..19_999 * 2 + 2].copy_from_slice(&[0x77, 0xFF]);
    let last_x = (w - 1) as usize * 2;
    giga.row_mut(h - 1).unwrap()[last_x..last_x + 2].copy_from_slice(&[0xFE, 0xFF]);

    assert_eq!(&giga.row(0).unwrap()[0..2], &[0x12, 0xFF]);
    assert_eq!(
        &giga.row(boundary_y).unwrap()[19_999 * 2..19_999 * 2 + 2],
        &[0x77, 0xFF]
    );
    assert_eq!(&giga.row(h - 1).unwrap()[last_x..last_x + 2], &[0xFE, 0xFF]);
    // Settled on row 0 (8-bit non-replicated luma + translucent alpha 0x80)!
    let prec = giga.scan_source_precision(&enough::Unstoppable).unwrap();
    assert_eq!(prec.significant_bits(), 8);
    assert_eq!(prec.alpha(), AlphaContent::Translucent);
}

#[test]
fn typed_convert_rows_validates_source_context_and_clears_stale_destination_context() {
    let icc_a = Arc::new(ColorContext::from_cicp(Cicp::DISPLAY_P3));
    let stale_dst = Arc::new(ColorContext::from_cicp(Cicp::BT2100_PQ));
    let p3 = RGB.with_primaries(ColorPrimaries::DisplayP3);
    let src_buf = PixelBuffer::from_vec(vec![200, 50, 25], 1, 1, p3)
        .unwrap()
        .with_color_context(icc_a.clone());
    let mut dst_buf = PixelBuffer::from_vec(vec![1, 2, 3], 1, 1, RGB)
        .unwrap()
        .with_color_context(stale_dst);

    let spec = ConversionSpec::new(ConvertOptions::retain_channels());
    let plan = ConvertPlan::build(src_buf.encoding(), RGB.into(), &spec).unwrap();
    let mut worker = plan.prepare(1).unwrap();

    // Changing source context under a compiled worker refuses before mutating dst!
    let untagged = PixelBuffer::from_vec(vec![200, 50, 25], 1, 1, p3).unwrap();
    let err = worker
        .convert_rows(untagged.as_slice(), dst_buf.as_slice_mut())
        .unwrap_err();
    assert_eq!(*err.error(), ConvertError::SourceInterpretationChanged);
    assert_eq!(dst_buf.as_slice().row(0), [1, 2, 3]);

    // Valid tagged source replaces stale BT.2100 PQ destination context with sRGB!
    {
        let out_slice = worker
            .convert_rows(src_buf.as_slice(), dst_buf.as_slice_mut())
            .unwrap();
        assert_eq!(
            out_slice.color_context().and_then(|c| c.cicp()),
            Some(Cicp::SRGB),
        );
    }

    // Untagged plan (`output_encoding().color_context() == None`) clears stale destination context to None!
    let untagged_plan = ConvertPlan::build(p3.into(), RGB.into(), &spec).unwrap();
    assert!(untagged_plan.output_encoding().color_context().is_none());
    let mut untagged_worker = untagged_plan.prepare(1).unwrap();
    {
        let out_slice = untagged_worker
            .convert_rows(untagged.as_slice(), dst_buf.as_slice_mut())
            .unwrap();
        assert!(
            out_slice.color_context().is_none(),
            "stale destination ColorContext must be cleared when output encoding context is None"
        );
    }
}

#[test]
fn composition_preserves_zero_alpha_erasure_clamping_and_anchor_boundaries() {
    let spec = ConversionSpec::new(ConvertOptions::retain_channels());

    // 1. StraightToPremul -> PremulToStraight on F32 MUST NOT cancel: alpha == 0 erases RGB!
    let straight = PixelDescriptor::RGBAF32_LINEAR;
    let premul = straight.with_alpha(Some(AlphaMode::Premultiplied));
    let p_a = ConvertPlan::build(straight.into(), premul.into(), &spec).unwrap();
    let p_b = ConvertPlan::build(premul.into(), straight.into(), &spec).unwrap();
    let composed = p_a.then(p_b).unwrap();
    assert!(!composed.is_identity());
    let mut worker = composed.prepare(1).unwrap();
    let src_px = [1.0f32, 0.5, 0.25, 0.0];
    let mut dst_px = [9.0f32; 4];
    worker
        .try_convert_row(
            bytemuck::cast_slice(&src_px),
            bytemuck::cast_slice_mut(&mut dst_px),
            1,
        )
        .unwrap();
    assert_eq!(dst_px, [0.0, 0.0, 0.0, 0.0]);

    // 2. Clamped LinearF32 -> SrgbF32 -> LinearF32 MUST NOT cancel: > 1.0 clamps to 1.0!
    let lin = PixelDescriptor::RGBF32_LINEAR;
    let srgb_f32 = lin.with_transfer(TransferFunction::Srgb);
    let c_a = ConvertPlan::build(lin.into(), srgb_f32.into(), &spec).unwrap();
    let c_b = ConvertPlan::build(srgb_f32.into(), lin.into(), &spec).unwrap();
    let mut clamp_worker = c_a.then(c_b).unwrap().prepare(1).unwrap();
    let over = [1.5f32, -0.25, 0.5];
    let mut clamped = [0.0f32; 3];
    clamp_worker
        .try_convert_row(
            bytemuck::cast_slice(&over),
            bytemuck::cast_slice_mut(&mut clamped),
            1,
        )
        .unwrap();
    assert!((clamped[0] - 1.0).abs() < 1e-6);
    assert!(clamped[1].abs() < 1e-6);

    // 3. Mismatched diffuse-white anchors (203 nits vs 100 nits) refuse ConvertPlan::then!
    let w203 = ColorContext::default().with_diffuse_white(zenpixels::hdr::DiffuseWhite::BT2408);
    let w100 = ColorContext::default().with_diffuse_white(zenpixels::hdr::DiffuseWhite::new(100.0));
    let p203 = ConvertPlan::build(
        PixelEncoding::new(lin, Some(&w203)),
        PixelEncoding::new(lin, Some(&w203)),
        &spec,
    )
    .unwrap();
    let p100 = ConvertPlan::build(
        PixelEncoding::new(lin, Some(&w100)),
        PixelEncoding::new(lin, Some(&w100)),
        &spec,
    )
    .unwrap();
    assert_eq!(
        *p203.then(p100).unwrap_err().error(),
        ConvertError::IntermediateMismatch
    );
}

#[test]
fn owned_output_reuses_existing_pixels() {
    let mut image = PixelBuffer::from_vec(vec![1, 2, 3, 4, 5, 6], 2, 1, RGB).unwrap();
    image = image.with_cicp(Cicp::SRGB);
    let start = image.as_slice().row(0).as_ptr();
    let target = OutputTarget::new(OutputProfile::Named(Cicp::SRGB), PixelFormat::Rgb8);
    let plan = OutputPlan::build(
        image.encoding(),
        target,
        &ConversionSpec::preserve_samples(),
    )
    .unwrap();
    let output = EncodeReady::from_owned(image, plan).unwrap();
    assert_eq!(output.pixels().row(0).as_ptr(), start);
    assert_eq!(output.pixels().row(0), [1, 2, 3, 4, 5, 6]);
}

struct FloatCms;
struct PureCopy;
impl RowTransform for PureCopy {
    fn try_transform_row(
        &self,
        src: &[u8],
        dst: &mut [u8],
        _: u32,
    ) -> Result<(), At<CmsPluginError>> {
        dst.copy_from_slice(src);
        Ok(())
    }
    fn prepare(&self, _: u32) -> Result<(), At<CmsPluginError>> {
        Ok(())
    }
}
impl PluggableCms for FloatCms {
    fn build(&self, request: &CmsRequest<'_>) -> Result<CmsSupport, At<CmsPluginError>> {
        assert!(matches!(request.source(), ColorProfileSource::Icc(_)));
        assert!(matches!(request.target(), ColorProfileSource::Icc(_)));
        if request.from().pixel_format() != request.to().pixel_format() {
            return Ok(CmsSupport::Declined);
        }
        Ok(CmsSupport::Shared(Arc::new(PureCopy)))
    }
}

#[test]
fn opaque_icc_width_stage_retains_endpoint_precision_and_clears_stale_peak() {
    let unknown = RGB
        .with_primaries(ColorPrimaries::Unknown)
        .with_transfer(TransferFunction::Unknown);
    let wide = PixelDescriptor::RGB16_SRGB
        .with_primaries(ColorPrimaries::Unknown)
        .with_transfer(TransferFunction::Unknown);
    let from = ColorContext::from_icc(Arc::<[u8]>::from([1u8, 2, 3]))
        .with_hdr_peak(zenpixels::hdr::Nits::new(1000.0));
    let to = ColorContext::from_icc(Arc::<[u8]>::from([4u8, 5, 6]));
    let options = ConvertOptions::retain_channels().with_depth_policy(DepthPolicy::Forbid);
    let spec = ConversionSpec::new(options).with_cms(&FloatCms);
    let src = PixelEncoding::new(unknown, Some(&from));
    let plan = ConvertPlan::build(src, PixelEncoding::new(wide, Some(&to)), &spec).unwrap();
    assert!(!plan.is_identity());
    let mut worker = plan.prepare(1).unwrap();
    let mut dst = [0u16; 3];
    worker
        .try_convert_row(&[0, 127, 255], bytemuck::cast_slice_mut(&mut dst), 1)
        .unwrap();
    assert_eq!(dst, [0, 127 * 257, 65535]);

    // Distinct ICC clears stale peak; identical ICC preserves peak!
    let diff = ConvertPlan::build(src, PixelEncoding::new(unknown, Some(&to)), &spec).unwrap();
    assert!(
        diff.output_encoding()
            .color_context()
            .unwrap()
            .hdr_peak()
            .is_none()
    );
    let same = ConvertPlan::build(src, PixelEncoding::new(unknown, Some(&from)), &spec).unwrap();
    assert_eq!(
        same.output_encoding().color_context().unwrap().hdr_peak(),
        Some(zenpixels::hdr::Nits::new(1000.0))
    );
}

#[cfg(feature = "cms-moxcms")]
#[test]
fn real_moxcms_icc_u8_to_u16_and_aces_cg_parametric_through_convert_and_adapt() {
    use zenpixels_convert::MoxCms;
    use zenpixels_convert::icc_profiles::{ADOBE_RGB, DISPLAY_P3_V4};

    // Real AdobeRGB U8 -> DisplayP3 U16 under DepthPolicy::Forbid succeeds, while U16 -> U8 refuses!
    let src_ctx = ColorContext::from_icc(ADOBE_RGB);
    let dst_ctx = ColorContext::from_icc(DISPLAY_P3_V4);
    let opts = ConvertOptions::retain_channels().with_depth_policy(DepthPolicy::Forbid);
    let spec = ConversionSpec::new(opts).with_cms(&MoxCms);
    let plan = ConvertPlan::build(
        PixelEncoding::new(PixelDescriptor::RGB8, Some(&src_ctx)),
        PixelEncoding::new(PixelDescriptor::RGB16, Some(&dst_ctx)),
        &spec,
    )
    .unwrap();
    let mut w = plan.prepare(2).unwrap();
    let mut out = [0u16; 6];
    w.try_convert_row(
        &[200, 100, 50, 10, 20, 30],
        bytemuck::cast_slice_mut(&mut out),
        2,
    )
    .unwrap();
    assert!(out[0] > 0);
    assert_eq!(
        *ConvertPlan::build(
            PixelEncoding::new(PixelDescriptor::RGB16, Some(&src_ctx)),
            PixelEncoding::new(PixelDescriptor::RGB8, Some(&dst_ctx)),
            &spec,
        )
        .unwrap_err()
        .error(),
        ConvertError::DepthReductionForbidden
    );

    // Custom D60 ParametricColor::ACES_CG routes through BOTH convert_to AND adapt_for_encode!
    let sample = [0.5f32, 0.2, 0.1];
    let aces_buf = PixelBuffer::from_vec(
        bytemuck::cast_slice(&sample).to_vec(),
        1,
        1,
        PixelDescriptor::RGBF32_LINEAR.with_primaries(ColorPrimaries::Unknown),
    )
    .unwrap()
    .with_color_context(Arc::new(ColorContext::from_parametric(
        ParametricColor::ACES_CG,
    )));
    let via_convert = aces_buf.convert_to(PixelDescriptor::RGBF32_LINEAR).unwrap();
    let via_adapt = aces_buf
        .as_slice()
        .adapt_for_encode(
            &[PixelDescriptor::RGBF32_LINEAR],
            &ConvertOptions::retain_channels(),
        )
        .unwrap();
    let via_c_slice = via_convert.as_slice();
    let via_a_slice = via_adapt.as_slice();
    let c_rgb: &[f32] = bytemuck::cast_slice(via_c_slice.row(0));
    let a_rgb: &[f32] = bytemuck::cast_slice(via_a_slice.row(0));
    assert_eq!(c_rgb, a_rgb);
    assert!(
        (c_rgb[0] - sample[0]).abs() > 1e-3 || (c_rgb[1] - sample[1]).abs() > 1e-3,
        "ACES_CG primaries must transform through both convert_to and adapt_for_encode: {c_rgb:?}"
    );
}

#[cfg(feature = "hdr-experimental")]
#[test]
fn hdr_config_refuses_unresolved_opaque_profile_and_histogram_merges_checked() {
    use zenpixels_convert::HdrConfig;
    use zenpixels_convert::hdr::{CllMeasure, ContentLightLevel, LightLevelMethod};

    // Blocker #1 witness: explicit HDR rendering combined with an unresolved ICC/Parametric
    // profile refuses cleanly instead of ignoring ICC authority!
    let opaque_icc = ColorContext::from_icc(Arc::<[u8]>::from([1u8, 2, 3, 4]));
    let hdr = HdrConfig::for_source_peak(1000.0);
    let spec = ConversionSpec::new(ConvertOptions::retain_channels()).with_hdr(&hdr);
    let err = ConvertPlan::build(
        PixelEncoding::new(PixelDescriptor::RGBF32_LINEAR, Some(&opaque_icc)),
        RGB.into(),
        &spec,
    )
    .unwrap_err();
    assert_eq!(*err.error(), ConvertError::HdrWithOpaqueProfileUnsupported);

    // Banded CLL parity + checked LightLevelHistogram::try_merge up to 2^33 samples & overflow guard.
    let desc = PixelDescriptor::RGBF32_LINEAR;
    let mut banded = BandedPixelBuffer::try_new(13, 7, desc, 13 * 12 * 3).unwrap();
    let mut tight = PixelBuffer::try_new(13, 7, desc).unwrap();
    for y in 0..7 {
        for (i, s) in bytemuck::cast_slice_mut::<u8, f32>(banded.row_mut(y).unwrap())
            .iter_mut()
            .enumerate()
        {
            *s = (y + i as u32) as f32 / 8.0;
        }
        tight
            .as_slice_mut()
            .row_mut(y)
            .copy_from_slice(banded.row(y).unwrap());
    }
    let white = zenpixels::hdr::DiffuseWhite::BT2408;
    let method = LightLevelMethod::MaxRgb;
    assert_eq!(
        ContentLightLevel::measure_max_banded(&banded, white, method),
        ContentLightLevel::measure_max(tight.as_slice(), white, method)
    );
    let mut b = ContentLightLevel::measure_histogram_banded(&banded, white, method).unwrap();
    let t = ContentLightLevel::measure_histogram(tight.as_slice(), white, method).unwrap();
    assert_eq!(b.bins(), t.bins());
    assert_eq!(b.total_pixels(), t.total_pixels());

    // Method mismatch returns None without mutating `b`.
    let other_method = ContentLightLevel::measure_histogram(
        tight.as_slice(),
        white,
        LightLevelMethod::LuminanceBt2020,
    )
    .unwrap();
    assert!(b.try_merge(&other_method).is_none());
    assert_eq!(b.total_pixels(), t.total_pixels());

    // 2^33 sample accumulation (> 2 * u32::MAX) + u64 overflow guard!
    let px = [2.0f32, 1.0, 0.5];
    let one = PixelBuffer::from_vec(bytemuck::cast_slice(&px).to_vec(), 1, 1, desc).unwrap();
    let mut h33 = ContentLightLevel::measure_histogram(one.as_slice(), white, method).unwrap();
    for _ in 0..33 {
        let copy = h33.clone();
        h33.try_merge(&copy).unwrap();
    }
    assert_eq!(h33.total_pixels(), 1u64 << 33);
    // Double 31 more times: 30 succeed (2^63), 31st overflows u64::MAX cleanly returning None!
    for _ in 0..30 {
        let copy = h33.clone();
        h33.try_merge(&copy).unwrap();
    }
    assert_eq!(h33.total_pixels(), 1u64 << 63);
    let before_overflow = h33.total_pixels();
    let copy = h33.clone();
    assert!(h33.try_merge(&copy).is_none());
    assert_eq!(h33.total_pixels(), before_overflow);
}

#[test]
fn worker_budget_prunes_routes_and_enforces_limit_across_prepare_and_composition() {
    use zenpixels_convert::Accelerators;

    let source = RGB.encoding();
    let wide = PixelDescriptor::RGBA16_SRGB.encoding();
    let narrow = PixelDescriptor::RGB16_SRGB.encoding();
    let spec = ConversionSpec::preserve_samples()
        .with_accelerators(&Accelerators::ALL)
        .with_worker_limit(5, 0);
    let rejected = ConvertPlan::build(source, wide, &spec).unwrap_err();
    assert!(matches!(
        rejected.error(),
        ConvertError::WorkerBudgetExceeded {
            bytes: ByteBound::Exact(20),
            limit: 0
        }
    ));
    let targets = [CandidateTarget::new(wide), CandidateTarget::new(narrow)];
    let selected = ConvertPlan::select(source, &targets, &spec).unwrap();
    assert_eq!(selected.candidate_index(), 1);
    assert_eq!(
        selected
            .into_plan()
            .workspace(WorkExtent::new(5, 1))
            .unwrap()
            .worker_heap(),
        ByteBound::Exact(0)
    );

    // Admitted at width 5 under a 20-byte limit MUST refuse prepare(6) (which requires 24 bytes)!
    let spec_20 = ConversionSpec::preserve_samples()
        .with_accelerators(&Accelerators::ALL)
        .with_worker_limit(5, 20);
    let plan_20 = ConvertPlan::build(source, wide, &spec_20).unwrap();
    let mut worker_5 = plan_20.prepare_shared(5).unwrap();
    assert!(matches!(
        worker_5.prepare(6).unwrap_err().error(),
        ConvertError::WorkerBudgetExceeded {
            bytes: ByteBound::Exact(24),
            limit: 20
        }
    ));
    assert!(matches!(
        plan_20.try_clone().unwrap().prepare(6).unwrap_err().error(),
        ConvertError::WorkerBudgetExceeded {
            bytes: ByteBound::Exact(24),
            limit: 20
        }
    ));

    // Composed plans retain the tighter worker limit across then()!
    let next_id = ConvertPlan::build(wide, wide, &ConversionSpec::preserve_samples()).unwrap();
    let composed = plan_20.then(next_id).unwrap();
    assert!(matches!(
        composed.prepare(6).unwrap_err().error(),
        ConvertError::WorkerBudgetExceeded { limit: 20, .. }
    ));
}

#[cfg(feature = "introspection")]
#[test]
fn introspection_reports_steps_accuracy_and_semantic_effects() {
    use zenpixels_convert::{AccuracyContract, SemanticEffects};

    let opts = ConvertOptions::retain_channels()
        .with_alpha_policy(AlphaPolicy::DiscardUnchecked)
        .with_depth_policy(DepthPolicy::Round)
        .with_gamut_policy(GamutPolicy::SoftCompress);
    let p3_f32 = PixelDescriptor::RGBF32_LINEAR.with_primaries(ColorPrimaries::DisplayP3);
    let plan = ConvertPlan::build(p3_f32.into(), RGB.into(), &ConversionSpec::new(opts)).unwrap();
    let steps: Vec<_> = plan.steps().collect();
    assert!(!steps.is_empty());
    let mut all_effects = SemanticEffects::NONE;
    let mut worst = AccuracyContract::Exact;
    for step in &steps {
        all_effects = all_effects.union(step.semantic_effects());
        if step.accuracy() > worst {
            worst = step.accuracy();
        }
    }
    assert!(all_effects.contains(SemanticEffects::GAMUT_COMPRESSED));
    assert!(all_effects.contains(SemanticEffects::DEPTH_NARROWED));
    assert!(worst >= AccuracyContract::Rendered);

    // Explicit relabel_transfer reports SemanticEffects::RELABELED_TRANSFER!
    let unknown = RGB.with_transfer(TransferFunction::Unknown);
    let relabel = ConvertPlan::relabel_transfer(unknown, RGB).unwrap();
    let r_steps: Vec<_> = relabel.steps().collect();
    assert_eq!(r_steps.len(), 1);
    assert!(
        r_steps[0]
            .semantic_effects()
            .contains(SemanticEffects::RELABELED_TRANSFER)
    );
}
