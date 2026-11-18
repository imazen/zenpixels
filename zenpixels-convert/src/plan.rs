//! Compile color and sample conversions before creating a row worker.

use alloc::sync::Arc;
use core::fmt;

use crate::converter::RowRecipe;
use crate::precision::{ConversionEffect, ConversionOffer, OutputChoice, SourcePrecision};
use crate::{
    AlphaMode, AlphaPolicy, ColorContext, ColorModel, ConvertError, ConvertOptions, PixelEncoding,
    PluggableCms, RowConverter, TransferFunction,
};
use whereat::At;

/// Policy and backend selection for a conversion request.
#[derive(Clone, Copy)]
pub struct ConversionSpec<'a> {
    options: ConvertOptions,
    cms: Option<&'a dyn PluggableCms>,
    precision: Option<SourcePrecision>,
    choice: OutputChoice,
    preserves_samples: bool,
    relabel_unknown: bool,
    workspace_limit: Option<(u32, u64)>,
    #[cfg(feature = "hdr-experimental")]
    hdr: Option<&'a crate::HdrConfig>,
    #[cfg(feature = "hdr-experimental")]
    tone_mapper: Option<&'a Arc<dyn crate::hdr::ToneMapper>>,
}

impl<'a> ConversionSpec<'a> {
    /// Select explicit permissions. Discovery is side-effect free with respect to samples.
    pub const fn new(options: ConvertOptions) -> Self {
        Self {
            options,
            cms: None,
            precision: None,
            choice: OutputChoice::Hidden,
            preserves_samples: false,
            relabel_unknown: false,
            workspace_limit: None,
            #[cfg(feature = "hdr-experimental")]
            hdr: None,
            #[cfg(feature = "hdr-experimental")]
            tone_mapper: None,
        }
    }

    /// Request only reversible sample representation changes.
    pub const fn preserve_samples() -> Self {
        let mut spec = Self::new(ConvertOptions::retain_channels());
        spec.preserves_samples = true;
        spec
    }

    /// Provide a backend for profile-aware conversions.
    pub const fn with_cms(mut self, cms: &'a dyn PluggableCms) -> Self {
        self.cms = Some(cms);
        self
    }

    /// Add declared capacity facts for admission. These facts carry no pixel-equality proof.
    pub const fn with_source_precision(mut self, precision: SourcePrecision) -> Self {
        self.precision = Some(precision);
        self
    }

    /// Identify who chose the final output.
    pub const fn with_output_choice(mut self, choice: OutputChoice) -> Self {
        self.choice = choice;
        self
    }

    /// Allow compact unknown transfer to adopt a declared destination label.
    ///
    /// Use only when the caller knows the correct interpretation of those samples.
    pub const fn with_unknown_transfer_relabel(mut self) -> Self {
        self.relabel_unknown = true;
        self
    }

    /// Refuse routes whose fresh worker exceeds the byte cap at this width.
    ///
    /// An opaque backend is refused because its worker bytes cannot be bounded.
    /// This cap covers worker preparation. Recipe construction and shared tables
    /// have separate lifetimes, reported by [ConvertPlan::workspace].
    pub const fn with_worker_limit(mut self, max_width: u32, bytes: u64) -> Self {
        self.workspace_limit = Some((max_width, bytes));
        self
    }

    /// Request explicit HDR rendering, including linear HDR sources.
    #[cfg(feature = "hdr-experimental")]
    pub const fn with_hdr(mut self, hdr: &'a crate::HdrConfig) -> Self {
        self.hdr = Some(hdr);
        self
    }

    /// Select a streaming tone mapper for the explicit HDR rendering.
    #[cfg(feature = "hdr-experimental")]
    pub const fn with_tone_mapper(mut self, mapper: &'a Arc<dyn crate::hdr::ToneMapper>) -> Self {
        self.tone_mapper = Some(mapper);
        self
    }

    pub(crate) const fn options(&self) -> &ConvertOptions {
        &self.options
    }

    pub(crate) const fn cms(&self) -> Option<&'a dyn PluggableCms> {
        self.cms
    }

    pub(crate) const fn exact(&self) -> bool {
        self.preserves_samples
    }
}

/// Conversion refusal or backend construction failure.
#[derive(Debug)]
#[non_exhaustive]
pub enum PlanError {
    /// The source has no transfer suitable for the requested color work.
    UnknownSourceTransfer,
    /// Conditional alpha removal requires proof or explicit authorization.
    NeedOpacityProof,
    /// The requested endpoint reduces precision prohibited by policy.
    DepthLossForbidden,
    /// The relative and absolute light domains require a white anchor.
    ExplicitSdrWhiteRequired,
    /// The requested materialized boundary disagrees with the next input.
    IntermediateMismatch,
    /// The output would exceed the requested capacity-loss policy.
    CapacityLoss(ConversionEffect),
    /// A requested output anchor differs from the effective output.
    TargetLuminanceMismatch,
    /// The worker cannot be admitted under the requested cap.
    WorkerBudgetExceeded { bytes: ByteBound, limit: u64 },
    /// No candidates were supplied.
    NoCandidates,
    /// None of the candidates could be admitted. The nested error identifies a refusal.
    NoCandidate {
        index: usize,
        cause: alloc::boxed::Box<At<Self>>,
    },
    /// Geometry, conversion, or backend error.
    Conversion(ConvertError),
}

impl PlanError {
    pub(crate) fn into_conversion(
        self,
        from: crate::PixelDescriptor,
        to: crate::PixelDescriptor,
    ) -> ConvertError {
        match self {
            Self::Conversion(error) => error,
            Self::NoCandidate { cause, .. } => match cause.error() {
                Self::Conversion(error) => error.clone(),
                Self::NeedOpacityProof => ConvertError::AlphaCheckRequired,
                Self::DepthLossForbidden => ConvertError::DepthReductionForbidden,
                Self::ExplicitSdrWhiteRequired => ConvertError::SdrWhiteRequired,
                _ => ConvertError::NoPath { from, to },
            },
            Self::NeedOpacityProof => ConvertError::AlphaCheckRequired,
            Self::DepthLossForbidden => ConvertError::DepthReductionForbidden,
            Self::ExplicitSdrWhiteRequired => ConvertError::SdrWhiteRequired,
            _ => ConvertError::NoPath { from, to },
        }
    }

    fn from_conversion(error: ConvertError) -> Self {
        match error {
            ConvertError::AlphaCheckRequired => Self::NeedOpacityProof,
            ConvertError::DepthReductionForbidden => Self::DepthLossForbidden,
            ConvertError::SdrWhiteRequired => Self::ExplicitSdrWhiteRequired,
            error => Self::Conversion(error),
        }
    }
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownSourceTransfer => {
                f.write_str("the source transfer needs an explicit interpretation")
            }
            Self::NeedOpacityProof => f.write_str("alpha removal requires proof of opacity"),
            Self::DepthLossForbidden => f.write_str("the requested endpoint reduces depth"),
            Self::ExplicitSdrWhiteRequired => {
                f.write_str("the conversion requires an SDR-white anchor")
            }
            Self::IntermediateMismatch => {
                f.write_str("the materialized intermediate interpretation differs")
            }
            Self::CapacityLoss(effect) => {
                write!(f, "the requested output loses modeled capacity: {effect:?}")
            }
            Self::TargetLuminanceMismatch => {
                f.write_str("the output luminance does not match the target")
            }
            Self::WorkerBudgetExceeded { bytes, limit } => {
                write!(
                    f,
                    "worker requires {bytes:?} bytes under a {limit}-byte limit"
                )
            }
            Self::NoCandidates => f.write_str("no conversion targets were offered"),
            Self::NoCandidate { index, cause } => {
                write!(f, "candidate {index} was refused: {cause}")
            }
            Self::Conversion(error) => error.fmt(f),
        }
    }
}

impl core::error::Error for PlanError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Conversion(error) => Some(error),
            Self::NoCandidate { cause, .. } => Some(cause.as_ref()),
            _ => None,
        }
    }
}

/// Allocation or backend preparation failure.
#[derive(Debug)]
pub struct PrepareError(ConvertError);

impl fmt::Display for PrepareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl core::error::Error for PrepareError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        Some(&self.0)
    }
}

/// Shape of the work to be submitted. Pointwise workers reuse their workspace for all rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkExtent {
    width: u32,
    rows: u32,
}

impl WorkExtent {
    /// Shape in pixels.
    pub const fn new(width: u32, rows: u32) -> Self {
        Self { width, rows }
    }

    /// Number of pixels, including on 32-bit platforms.
    pub const fn pixel_count(self) -> u64 {
        self.width as u64 * self.rows as u64
    }
}

/// A statement about allocation requests. Allocator and OS overhead are outside these bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ByteBound {
    /// Exact requested payload for the stated ownership and lifecycle.
    Exact(u64),
    /// At most this many bytes under the stated conditions.
    AtMost(u64),
    /// The backend or asset does not expose its allocation requirements.
    Unknown,
}

/// Memory associated with one compiled plan and one fresh row worker.
#[derive(Debug, Clone, Copy)]
pub struct AllocationRequirements {
    pub(crate) native_worker: u64,
    pub(crate) worker_heap: ByteBound,
    pub(crate) resident_heap: ByteBound,
    pub(crate) shared_cold_setup: ByteBound,
}

impl AllocationRequirements {
    /// Bytes allocated by the converter for its own scratch and staging rows.
    pub const fn native_worker(&self) -> u64 {
        self.native_worker
    }
    /// Workspace, including backend allocations when known.
    pub const fn worker_heap(&self) -> ByteBound {
        self.worker_heap
    }
    /// Retained recipe heap, excluding borrowed interpretation and already shared assets.
    pub const fn resident_heap(&self) -> ByteBound {
        self.resident_heap
    }
    /// Shared allocations that may initialize during cold preparation.
    pub const fn shared_cold_setup(&self) -> ByteBound {
        self.shared_cold_setup
    }
}

/// A destination offered to the converter.
#[derive(Debug, Clone, Copy)]
pub struct CandidateTarget<'a> {
    target: PixelEncoding<'a>,
    preference: u32,
}

impl<'a> CandidateTarget<'a> {
    /// Offer the exact sample and color interpretation requested by the consumer.
    pub const fn new(target: PixelEncoding<'a>) -> Self {
        Self {
            target,
            preference: 0,
        }
    }
    /// Prefer smaller values among otherwise equivalent routes.
    pub const fn with_preference(mut self, preference: u32) -> Self {
        self.preference = preference;
        self
    }
}

/// An accepted candidate and its executable conversion.
#[derive(Debug)]
pub struct SelectedConversion {
    index: usize,
    plan: ConvertPlan,
}

impl SelectedConversion {
    /// Position in the offered candidate slice.
    pub const fn candidate_index(&self) -> usize {
        self.index
    }
    /// Reuse the plan without rebuilding the backend.
    pub fn into_plan(self) -> ConvertPlan {
        self.plan
    }
}

/// An executable conversion recipe, including backend operations and output interpretation.
pub struct ConvertPlan {
    recipe: RowRecipe,
    input: Option<Arc<ColorContext>>,
    output: Option<Arc<ColorContext>>,
}

impl fmt::Debug for ConvertPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("ConvertPlan");
        debug
            .field("input", &self.source_encoding())
            .field("output", &self.output_encoding())
            .field("is_identity", &self.is_identity());
        #[cfg(feature = "introspection")]
        debug.field("recipe", &self.recipe);
        debug.finish_non_exhaustive()
    }
}

impl ConvertPlan {
    /// Check authority and policy and compile the actual route.
    pub fn build(
        source: PixelEncoding<'_>,
        target: PixelEncoding<'_>,
        spec: &ConversionSpec<'_>,
    ) -> Result<Self, At<PlanError>> {
        let from = source.descriptor();
        let to = target.descriptor();
        let convert_error = |e: At<ConvertError>| e.map_error(PlanError::from_conversion);
        let src_profile = crate::output::current_profile_for(source).map_err(convert_error)?;
        let dst_profile = crate::output::current_profile_for(target).map_err(convert_error)?;
        let same_profile = crate::output::profiles_match(&src_profile, &dst_profile)
            || (from == to && source.color_context() == target.color_context());
        let mut options =
            crate::interpretation::resolve_options(source.color_context(), spec.options());
        if from.alpha() == Some(AlphaMode::Opaque)
            && options.alpha_policy() == AlphaPolicy::DiscardIfOpaque
        {
            options = options.with_alpha_policy(AlphaPolicy::DiscardUnchecked);
        }
        if from.transfer() == TransferFunction::Unknown
            && to.transfer() != TransferFunction::Unknown
            && matches!(from.color_model(), ColorModel::Rgb | ColorModel::Gray)
            && !matches!(
                src_profile,
                crate::ColorProfileSource::Icc(_) | crate::ColorProfileSource::Parametric(_)
            )
            && src_profile
                .resolve()
                .is_none_or(|(_, transfer)| transfer == TransferFunction::Unknown)
            && !spec.relabel_unknown
        {
            return Err(whereat::at!(PlanError::UnknownSourceTransfer));
        }

        if let Some(precision) = spec.precision {
            let mut offer = ConversionOffer::new(to);
            if matches!(options.alpha_policy(), AlphaPolicy::CompositeOnto { .. }) {
                offer = offer.with_matte();
            }
            let effect = ConversionEffect::of(&precision, &offer);
            if spec.choice == OutputChoice::Hidden && !effect.preserves_modeled_capacity() {
                return Err(whereat::at!(PlanError::CapacityLoss(effect)));
            }
        }

        let mut recipe =
            compile(source, target, same_profile, &options, spec).map_err(convert_error)?;
        // The row endpoints use the caller's carriers even if the kernel resolved
        // fields of a compact descriptor using attached profile evidence.
        recipe.from = from;
        recipe.to = to;
        if let Some((width, limit)) = spec.workspace_limit {
            let bytes = recipe
                .requirements(width)
                .map_err(convert_error)?
                .worker_heap();
            if !fits_worker_limit(bytes, limit) {
                return Err(whereat::at!(PlanError::WorkerBudgetExceeded {
                    bytes,
                    limit
                }));
            }
        }
        let output = resolve_output(source, target, same_profile, &options);
        #[cfg(feature = "hdr-experimental")]
        let output = if let Some(hdr) = spec.hdr
            && (matches!(
                from.transfer(),
                TransferFunction::Linear | TransferFunction::Pq
            )) {
            Some(
                output
                    .unwrap_or_default()
                    .with_diffuse_white(zenpixels::hdr::DiffuseWhite::new(hdr.target_peak_nits))
                    .with_hdr_peak(zenpixels::hdr::Nits::new(hdr.target_peak_nits)),
            )
        } else {
            output
        };
        if let Some(requested) = target.color_context()
            && (requested.diffuse_white().is_some_and(|white| {
                output.as_ref().and_then(ColorContext::diffuse_white) != Some(white)
            }) || requested
                .hdr_peak()
                .is_some_and(|peak| output.as_ref().and_then(ColorContext::hdr_peak) != Some(peak)))
        {
            return Err(whereat::at!(PlanError::TargetLuminanceMismatch));
        }
        Ok(Self {
            recipe,
            input: source.color_context().cloned().map(Arc::new),
            output: output.map(Arc::new),
        })
    }

    /// Select an admitted route and return its executable plan.
    pub fn select(
        source: PixelEncoding<'_>,
        targets: &[CandidateTarget<'_>],
        spec: &ConversionSpec<'_>,
    ) -> Result<SelectedConversion, At<PlanError>> {
        let mut admitted = crate::route::RouteFrontier::new();
        let mut refusal = None;
        for (index, candidate) in targets.iter().enumerate() {
            match Self::build(source, candidate.target, spec) {
                Ok(plan) => {
                    let scratch = spec
                        .workspace_limit
                        .map(|(width, _)| plan.workspace(WorkExtent::new(width, 1)))
                        .transpose()?
                        .map_or(0, |memory| memory.native_worker());
                    admitted.offer(
                        plan.recipe.rank(),
                        candidate.preference,
                        scratch,
                        index,
                        SelectedConversion { index, plan },
                    );
                }
                Err(e)
                    if matches!(
                        e.error(),
                        PlanError::Conversion(
                            ConvertError::AllocationFailed | ConvertError::CmsBackend(_)
                        )
                    ) =>
                {
                    return Err(e);
                }
                Err(e) => {
                    if refusal.is_none() {
                        refusal = Some((index, e));
                    }
                }
            }
        }
        if let Some(selected) = admitted.finish() {
            return Ok(selected);
        }
        if let Some((index, cause)) = refusal {
            return Err(whereat::at!(PlanError::NoCandidate {
                index,
                cause: alloc::boxed::Box::new(cause)
            }));
        }
        Err(whereat::at!(PlanError::NoCandidates))
    }

    /// Inspect the lowered native and backend operators in execution order.
    #[cfg(feature = "introspection")]
    pub fn steps(&self) -> impl Iterator<Item = crate::PlannedStepInfo> + '_ {
        self.recipe.steps()
    }

    /// Source interpretation required by this recipe.
    pub fn source_encoding(&self) -> PixelEncoding<'_> {
        PixelEncoding::new(self.recipe.from, self.input.as_deref())
    }

    /// Interpretation produced by this recipe, including any effective luminance metadata.
    pub fn output_encoding(&self) -> PixelEncoding<'_> {
        PixelEncoding::new(self.recipe.to, self.output.as_deref())
    }

    /// Whether the operation leaves active sample bytes unchanged.
    pub fn is_identity(&self) -> bool {
        !self.recipe.needs_pixel_work()
    }

    /// Compose with an explicitly materialized boundary, including its interpretation and rounding.
    pub fn then(self, next: Self) -> Result<Self, At<PlanError>> {
        if self.output_encoding() != next.source_encoding() {
            return Err(whereat::at!(PlanError::IntermediateMismatch));
        }
        Ok(Self {
            recipe: self.recipe.then(next.recipe),
            input: self.input,
            output: next.output,
        })
    }

    /// Report memory for the same layout used to prepare this recipe.
    pub fn workspace(&self, extent: WorkExtent) -> Result<AllocationRequirements, At<PlanError>> {
        self.recipe
            .requirements(extent.width)
            .map_err(|e| e.map_error(PlanError::from_conversion))
    }

    /// Build an independent worker and reserve scratch for rows at most this wide.
    pub fn prepare(self, max_width: u32) -> Result<RowConverter, At<PrepareError>> {
        let mut worker = self.into_worker();
        worker
            .prepare(max_width)
            .map_err(|e| e.map_error(PrepareError))?;
        Ok(worker)
    }

    /// Duplicate a reusable native or shared-backend plan.
    pub fn try_clone(&self) -> Result<Self, At<PlanError>> {
        Ok(Self {
            recipe: self
                .recipe
                .clone_for_worker()
                .map_err(|e| e.map_error(PlanError::from_conversion))?,
            input: self.input.clone(),
            output: self.output.clone(),
        })
    }

    /// Create a worker that grows scratch on demand. Prefer [Self::prepare] for bounded work.
    pub fn into_worker(self) -> RowConverter {
        let mut worker = self.recipe.into_worker(self.output);
        worker.set_source_context(self.input);
        worker
    }
}

fn compile(
    source: PixelEncoding<'_>,
    target: PixelEncoding<'_>,
    same_profile: bool,
    options: &ConvertOptions,
    spec: &ConversionSpec<'_>,
) -> Result<RowRecipe, At<ConvertError>> {
    let (mut from, to) = (source.descriptor(), target.descriptor());
    let hdr_requested = false;
    #[cfg(feature = "hdr-experimental")]
    let hdr_requested = hdr_requested || spec.hdr.is_some();
    let allow_identity = from == to
        && same_profile
        && (!crate::convert::needs_cms_for_color_model(&from, &to)
            || matches!(
                crate::output::current_profile_for(source)?,
                crate::ColorProfileSource::Icc(_)
            ));
    if allow_identity && !hdr_requested {
        crate::convert::validate_descriptors(from, to)?;
        return Ok(RowRecipe::identity(from, to));
    }
    if same_profile
        && let Some((primaries, transfer)) = crate::output::current_profile_for(source)?.resolve()
    {
        if from.primaries == crate::ColorPrimaries::Unknown {
            from.primaries = primaries;
        }
        if from.transfer() == TransferFunction::Unknown {
            from.transfer = transfer;
        }
    }
    if !hdr_requested
        && same_profile
        && from.transfer() == to.transfer()
        && from.primaries == to.primaries
        && from.signal_range == to.signal_range
        && (spec.exact()
            || (from.channel_type() == crate::ChannelType::U8
                && to.channel_type() == crate::ChannelType::U16))
        && let Some(recipe) = exact_path(from, to, spec.workspace_limit)
    {
        return Ok(recipe);
    }
    if spec.exact() {
        return Err(whereat::at!(ConvertError::NoPath { from, to }));
    }
    #[cfg(feature = "hdr-experimental")]
    if let Some(hdr) = spec.hdr {
        crate::convert::validate_descriptors(from, to)?;
        crate::convert::validate_luma_policy(from, to, options)?;
        // The special linear-HDR entry still applies the endpoint permissions.
        if from.has_alpha()
            && !to.has_alpha()
            && options.alpha_policy() != AlphaPolicy::DiscardUnchecked
        {
            return Err(whereat::at!(
                if options.alpha_policy() == AlphaPolicy::DiscardIfOpaque {
                    ConvertError::AlphaCheckRequired
                } else {
                    ConvertError::AlphaRemovalForbidden
                }
            ));
        }
        if options.depth_policy() == crate::DepthPolicy::Forbid
            && crate::route::channel_bits(from.channel_type())
                > crate::route::channel_bits(to.channel_type())
        {
            return Err(whereat::at!(ConvertError::DepthReductionForbidden));
        }
        let native = match spec.tone_mapper {
            Some(mapper) => {
                crate::NativePlan::new_with_tone_mapper(from, to, *hdr, Arc::clone(mapper))?
            }
            None => crate::NativePlan::new_with_hdr_config(from, to, *hdr)?,
        };
        let native = if let Some(white) =
            source.color_context().and_then(ColorContext::diffuse_white)
            && from.transfer() == TransferFunction::Linear
        {
            native.with_linear_input_white(white, hdr.source_peak_nits)
        } else {
            native
        };
        return Ok(RowRecipe::native(native));
    }
    if same_profile && from.transfer() == to.transfer() && from.primaries == to.primaries {
        return RowRecipe::new_explicit(from, to, options);
    }
    RowRecipe::new_with_encoding(source, target, options, spec.cms())
}

// The finite exact subgraph comprises the endpoints and the endpoint
// layout/depth cross-products. Existing native code proves each edge.
fn exact_path(
    from: crate::PixelDescriptor,
    to: crate::PixelDescriptor,
    limit: Option<(u32, u64)>,
) -> Option<RowRecipe> {
    #[derive(Clone, Copy)]
    struct State(crate::PixelDescriptor);

    fn edge(from: State, to: State) -> Option<RowRecipe> {
        crate::NativePlan::new_preserving_samples(from.0, to.0)
            .ok()
            .map(RowRecipe::native)
    }

    let (source, target) = (State(from), State(to));
    let mut frontier = crate::route::RouteFrontier::new();
    let mut excluded = crate::route::RouteFrontier::new();
    let mut consider = |recipe: RowRecipe, index| {
        let memory = limit.and_then(|(width, _)| recipe.requirements(width).ok());
        let scratch = memory.map_or(0, |memory| memory.native_worker());
        let is_excluded = limit.is_some_and(|(_, cap)| {
            memory.is_none_or(|memory| !fits_worker_limit(memory.worker_heap(), cap))
        });
        let routes = if is_excluded {
            &mut excluded
        } else {
            &mut frontier
        };
        routes.offer(recipe.rank(), 0, scratch, index, recipe);
    };
    if let Some(direct) = edge(source, target) {
        consider(direct, 0);
    }
    if from.layout() != to.layout() && from.channel_type() != to.channel_type() {
        for (i, (layout, depth)) in [(from, to.channel_type()), (to, from.channel_type())]
            .into_iter()
            .enumerate()
        {
            let middle = State(
                crate::PixelDescriptor::new(
                    depth,
                    layout.layout(),
                    layout.alpha(),
                    layout.transfer(),
                )
                .with_primaries(layout.primaries)
                .with_signal_range(layout.signal_range),
            );
            if let (Some(before), Some(after)) = (edge(source, middle), edge(middle, target)) {
                consider(before.then(after), i + 1);
            }
        }
    }
    frontier.finish().or_else(|| excluded.finish())
}

fn fits_worker_limit(bound: ByteBound, limit: u64) -> bool {
    matches!(bound, ByteBound::Exact(bytes) | ByteBound::AtMost(bytes) if bytes <= limit)
}

fn resolve_output(
    source: PixelEncoding<'_>,
    target: PixelEncoding<'_>,
    same_profile: bool,
    options: &ConvertOptions,
) -> Option<ColorContext> {
    let mut value = crate::interpretation::output_context_value(
        source.descriptor(),
        target.descriptor(),
        source.color_context(),
        options,
        false,
    );
    if let Some(target_context) = target.color_context() {
        let profile = target_context.profile().cloned();
        if !same_profile && source.descriptor() == target.descriptor() {
            value = None;
        }
        return Some(value.unwrap_or_default().with_profile(profile));
    }
    value
}
