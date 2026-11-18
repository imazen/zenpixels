//! Explicit historical options for compatibility fixtures.

#![allow(dead_code)]

use whereat::At;
use zenpixels_convert::{
    AlphaPolicy, ColorOrigin, ConversionSpec, ConvertError, ConvertOptions, ConvertPlan,
    DepthPolicy, EncodeReady, LumaCoefficients, OutputPlan, OutputProfile, OutputTarget,
    PixelBuffer, PixelDescriptor, PixelEncoding, PixelFormat, PlanError, PluggableCms,
    RowConverter,
};

pub const fn legacy_options() -> ConvertOptions {
    ConvertOptions::retain_channels()
        .with_alpha_policy(AlphaPolicy::DiscardUnchecked)
        .with_depth_policy(DepthPolicy::Round)
        .with_luma(Some(LumaCoefficients::Bt709))
}

fn remap(err: At<PlanError>, from: PixelDescriptor, to: PixelDescriptor) -> At<ConvertError> {
    err.map_error(|error| match error {
        PlanError::Conversion(error) => error,
        PlanError::NeedOpacityProof => ConvertError::AlphaCheckRequired,
        PlanError::DepthLossForbidden => ConvertError::DepthReductionForbidden,
        PlanError::ExplicitSdrWhiteRequired => ConvertError::SdrWhiteRequired,
        _ => ConvertError::NoPath { from, to },
    })
}

fn with_cms<'a>(spec: ConversionSpec<'a>, cms: Option<&'a dyn PluggableCms>) -> ConversionSpec<'a> {
    match cms {
        Some(cms) => spec.with_cms(cms),
        None => spec,
    }
}

fn plan(
    from: PixelEncoding<'_>,
    to: PixelEncoding<'_>,
    spec: &ConversionSpec<'_>,
) -> Result<ConvertPlan, At<ConvertError>> {
    ConvertPlan::build(from, to, spec).map_err(|e| remap(e, from.descriptor(), to.descriptor()))
}

pub fn new_plan(
    from: PixelDescriptor,
    to: PixelDescriptor,
) -> Result<ConvertPlan, At<ConvertError>> {
    explicit_plan(from, to, &legacy_options())
}

pub fn explicit_plan(
    from: PixelDescriptor,
    to: PixelDescriptor,
    options: &ConvertOptions,
) -> Result<ConvertPlan, At<ConvertError>> {
    let spec = ConversionSpec::new(*options).with_unknown_transfer_relabel();
    plan(from.into(), to.into(), &spec)
}

pub fn exact_plan(
    from: PixelDescriptor,
    to: PixelDescriptor,
) -> Result<ConvertPlan, At<ConvertError>> {
    plan(from.into(), to.into(), &ConversionSpec::preserve_samples())
}

pub fn new_worker(
    from: PixelDescriptor,
    to: PixelDescriptor,
) -> Result<RowConverter, At<ConvertError>> {
    explicit_worker(from, to, &legacy_options())
}

pub fn explicit_worker(
    from: PixelDescriptor,
    to: PixelDescriptor,
    options: &ConvertOptions,
) -> Result<RowConverter, At<ConvertError>> {
    encoded_worker(from.into(), to.into(), options, None)
}

pub fn cms_worker(
    from: PixelDescriptor,
    to: PixelDescriptor,
    options: &ConvertOptions,
    cms: Option<&dyn PluggableCms>,
) -> Result<RowConverter, At<ConvertError>> {
    encoded_worker(from.into(), to.into(), options, cms)
}

pub fn encoded_worker(
    from: PixelEncoding<'_>,
    to: PixelEncoding<'_>,
    options: &ConvertOptions,
    cms: Option<&dyn PluggableCms>,
) -> Result<RowConverter, At<ConvertError>> {
    let spec = with_cms(
        ConversionSpec::new(*options).with_unknown_transfer_relabel(),
        cms,
    );
    plan(from, to, &spec).map(ConvertPlan::into_worker)
}

#[cfg(feature = "hdr-experimental")]
pub fn hdr_plan(
    from: PixelDescriptor,
    to: PixelDescriptor,
    hdr: zenpixels_convert::HdrConfig,
) -> Result<ConvertPlan, At<ConvertError>> {
    plan(
        from.into(),
        to.into(),
        &ConversionSpec::new(legacy_options()).with_hdr(&hdr),
    )
}

#[cfg(feature = "hdr-experimental")]
pub fn hdr_peak_plan(
    from: PixelDescriptor,
    to: PixelDescriptor,
    peak: f32,
) -> Result<ConvertPlan, At<ConvertError>> {
    hdr_plan(
        from,
        to,
        zenpixels_convert::HdrConfig::for_source_peak(peak),
    )
}

#[cfg(feature = "hdr-experimental")]
pub fn mapper_plan(
    from: PixelDescriptor,
    to: PixelDescriptor,
    hdr: zenpixels_convert::HdrConfig,
    mapper: std::sync::Arc<dyn zenpixels_convert::hdr::ToneMapper>,
) -> Result<ConvertPlan, At<ConvertError>> {
    plan(
        from.into(),
        to.into(),
        &ConversionSpec::new(legacy_options())
            .with_hdr(&hdr)
            .with_tone_mapper(&mapper),
    )
}

pub fn output_plan(
    source: PixelEncoding<'_>,
    origin: &ColorOrigin,
    profile: OutputProfile,
    format: PixelFormat,
    options: &ConvertOptions,
    cms: Option<&dyn PluggableCms>,
) -> Result<OutputPlan, At<ConvertError>> {
    let spec = with_cms(ConversionSpec::new(*options), cms);
    OutputPlan::build(
        source,
        OutputTarget::new(profile, format).with_origin(origin),
        &spec,
    )
}

pub fn output_descriptor_plan(
    source: PixelEncoding<'_>,
    origin: &ColorOrigin,
    profile: OutputProfile,
    output: PixelDescriptor,
    options: &ConvertOptions,
    cms: Option<&dyn PluggableCms>,
) -> Result<OutputPlan, At<ConvertError>> {
    let spec = with_cms(ConversionSpec::new(*options), cms);
    OutputPlan::build(
        source,
        OutputTarget::new(profile, output.pixel_format())
            .with_descriptor(output)
            .with_origin(origin),
        &spec,
    )
}

pub fn exact_output_plan(
    source: PixelEncoding<'_>,
    origin: &ColorOrigin,
    profile: OutputProfile,
    output: PixelDescriptor,
) -> Result<OutputPlan, At<ConvertError>> {
    OutputPlan::build(
        source,
        OutputTarget::new(profile, output.pixel_format())
            .with_descriptor(output)
            .with_origin(origin),
        &ConversionSpec::preserve_samples(),
    )
}

pub fn copy_output(
    source: &PixelBuffer,
    origin: &ColorOrigin,
    profile: OutputProfile,
    format: PixelFormat,
    cms: Option<&dyn PluggableCms>,
) -> Result<EncodeReady, At<ConvertError>> {
    let plan = output_plan(
        source.encoding(),
        origin,
        profile,
        format,
        &legacy_options(),
        cms,
    )?;
    EncodeReady::copy_from(source.as_slice(), plan)
}

pub fn worker(plan: ConvertPlan) -> RowConverter {
    plan.into_worker()
}
