use std::sync::Arc;
use whereat::At;
use zenpixels_convert::{
    AlphaPolicy, CandidateTarget, Cicp, ColorContext, ColorProfileSource, ConversionSpec,
    ConvertOptions, ConvertPlan, DepthPolicy, EncodeReady, LoadBearingReport, OutputPlan,
    OutputProfile, OutputTarget, PixelBuffer, PixelDescriptor, PixelEncoding, PixelFormat,
    PlanError, TransferFunction, WorkExtent,
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
    assert!(
        matches!(e.error(), PlanError::NoCandidate { index: 0, cause } if matches!(cause.error(), PlanError::NeedOpacityProof))
    );
}

#[test]
fn unknown_transfer_requires_an_explicit_label() {
    let unknown = RGB.with_transfer(TransferFunction::Unknown);
    let spec = ConversionSpec::new(ConvertOptions::retain_channels());
    assert!(matches!(
        ConvertPlan::build(unknown.into(), RGB.into(), &spec)
            .unwrap_err()
            .error(),
        PlanError::UnknownSourceTransfer
    ));
    let verified = ColorContext::from_cicp(Cicp::SRGB);
    let source = PixelEncoding::new(unknown, Some(&verified));
    let plan = ConvertPlan::build(source, RGB.into(), &spec).unwrap();
    assert!(plan.is_identity());
    assert_eq!(plan.output_encoding().descriptor(), RGB);
    let spec = spec.with_unknown_transfer_relabel();
    assert!(
        ConvertPlan::build(unknown.into(), RGB.into(), &spec)
            .unwrap()
            .is_identity()
    );
}

#[test]
fn workspace_reports_exact_scratch_for_zero_one_and_two_operations() {
    let rgb16 = PixelDescriptor::RGB16_SRGB;
    let rgba16 = PixelDescriptor::RGBA16_SRGB;
    for (from, to, expected) in [(RGB, RGB, 0), (RGB, rgb16, 0), (RGB, rgba16, 20)] {
        let plan = ConvertPlan::build(from.into(), to.into(), &ConversionSpec::preserve_samples())
            .unwrap();
        let needed = plan.workspace(WorkExtent::new(5, u32::MAX)).unwrap();
        assert_eq!(
            needed.worker_heap(),
            zenpixels_convert::ByteBound::Exact(expected)
        );
        let mut worker = None;
        let count = allocation_counter::measure(|| {
            worker = Some(plan.prepare(5).unwrap());
        });
        assert_eq!(count.bytes_total, expected);
        drop(worker);
    }
}

#[test]
fn band_conversion_and_scan_use_global_rows() {
    let mut source = BandedPixelBuffer::try_new(5, 7, RGB, 30).unwrap();
    let mut target = BandedPixelBuffer::try_new(5, 7, RGBA, 60).unwrap();
    for y in 0..7 {
        source.row_mut(y).unwrap().fill(y as u8);
    }
    source.row_mut(6).unwrap()[1] = 29;
    let plan =
        ConvertPlan::build(RGB.into(), RGBA.into(), &ConversionSpec::preserve_samples()).unwrap();
    plan.prepare(5)
        .unwrap()
        .convert_bands(&source, &mut target)
        .unwrap();
    for y in 0..7 {
        for (actual, src) in target
            .row(y)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .zip(source.row(y).unwrap().as_chunks::<3>().0.iter())
        {
            assert_eq!(*actual, [src[0], src[1], src[2], 255]);
        }
    }
    let report = LoadBearingReport::scan_bands(&target);
    assert_eq!(report.uses_alpha, Some(false));
    assert_eq!(report.uses_chroma, Some(true));
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
        .with_primaries(zenpixels_convert::ColorPrimaries::Unknown)
        .with_transfer(TransferFunction::Unknown);
    let wide = PixelDescriptor::RGB16_SRGB
        .with_primaries(zenpixels_convert::ColorPrimaries::Unknown)
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
    let plan = ConvertPlan::build(src, PixelEncoding::new(unknown, Some(&to)), &spec).unwrap();
    assert!(
        plan.output_encoding()
            .color_context()
            .unwrap()
            .hdr_peak()
            .is_none()
    );
}

#[cfg(feature = "hdr-experimental")]
#[test]
fn luminance_reduction_is_independent_of_band_height() {
    use zenpixels_convert::hdr::{CllMeasure, ContentLightLevel, LightLevelMethod};
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
    let b = ContentLightLevel::measure_histogram_banded(&banded, white, method).unwrap();
    let t = ContentLightLevel::measure_histogram(tight.as_slice(), white, method).unwrap();
    assert_eq!(b.bins(), t.bins());
    assert_eq!(b.total_pixels(), t.total_pixels());
}

#[test]
fn worker_budget_prunes_routes_using_reported_workspace() {
    use zenpixels_convert::ByteBound;
    let source = RGB.encoding();
    let wide = PixelDescriptor::RGBA16_SRGB.encoding();
    let narrow = PixelDescriptor::RGB16_SRGB.encoding();
    let spec = ConversionSpec::preserve_samples().with_worker_limit(5, 0);
    let rejected = ConvertPlan::build(source, wide, &spec).unwrap_err();
    assert!(matches!(
        rejected.error(),
        PlanError::WorkerBudgetExceeded {
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
}
