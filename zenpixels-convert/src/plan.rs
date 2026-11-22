//! Compile color and sample conversions before creating a row worker.

use alloc::sync::Arc;
use core::fmt;

use crate::converter::RowRecipe;
use crate::precision::{
    AlphaContent, ConversionEffect, ConversionOffer, OutputChoice, SourcePrecision,
};
use crate::{
    AlphaMode, AlphaPolicy, ColorContext, ColorModel, ConvertError, ConvertIntent, ConvertOptions,
    PixelDescriptor, PixelEncoding, PluggableCms, Provenance, RowConverter, TransferFunction,
};
use whereat::At;

/// Policy, intent, and backend selection for a conversion request.
#[derive(Clone, Copy)]
pub struct ConversionSpec<'a> {
    options: ConvertOptions,
    cms: Option<&'a dyn PluggableCms>,
    precision: Option<SourcePrecision>,
    float_facts: zenpixels::sample::FloatFacts,
    choice: OutputChoice,
    intent: Option<ConvertIntent>,
    provenance: Option<Provenance>,
    preserves_samples: bool,
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
            float_facts: zenpixels::sample::FloatFacts::NONE,
            choice: OutputChoice::Hidden,
            intent: None,
            provenance: None,
            preserves_samples: false,
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

    /// Declare verified or trusted [`FloatFacts`](zenpixels::sample::FloatFacts) for
    /// floating-point input samples (`F16`, `F32`, `F64`).
    ///
    /// Plans compiled with non-empty facts specialize internal math kernels where sound,
    /// propagate proven output facts to [`ConvertPlan::output_float_facts`], and enforce
    /// `src.float_facts().contains(plan.input_float_facts())` in
    /// [`RowConverter::convert_rows`] and [`RowConverter::convert_bands`].
    /// Non-empty facts on integer sources fail compilation with
    /// [`ConvertError::FloatFactsUnsatisfied`].
    pub const fn with_float_facts(mut self, facts: zenpixels::sample::FloatFacts) -> Self {
        self.float_facts = facts;
        self
    }

    /// Declared floating-point input facts on this request.
    pub const fn float_facts(&self) -> zenpixels::sample::FloatFacts {
        self.float_facts
    }

    /// Identify who chose the final output.
    pub const fn with_output_choice(mut self, choice: OutputChoice) -> Self {
        self.choice = choice;
        self
    }

    /// Steer [`ConvertPlan::select`] by operational suitability (`Fastest`,
    /// `LinearLight`, `Blend`, or `Perceptual`).
    ///
    /// When omitted, [`ConvertPlan::select`] ranks admitted candidates by semantic
    /// fidelity first and execution cost second.
    pub const fn with_intent(mut self, intent: ConvertIntent) -> Self {
        self.intent = Some(intent);
        self
    }

    /// Supply true origin depth and primaries for [`ConvertPlan::select`] cost scoring.
    pub const fn with_provenance(mut self, provenance: Provenance) -> Self {
        self.provenance = Some(provenance);
        self
    }

    /// Refuse routes whose fresh worker exceeds the byte cap at this width.
    ///
    /// An opaque backend is refused because its worker bytes cannot be bounded.
    /// This cap covers worker preparation and carries forward onto prepared
    /// [`RowConverter`]s. Recipe construction and shared tables have separate
    /// lifetimes, reported by [`ConvertPlan::workspace`].
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

    /// Width in pixels.
    pub const fn width(self) -> u32 {
        self.width
    }

    /// Row count.
    pub const fn rows(self) -> u32 {
        self.rows
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

/// A destination offered to [`ConvertPlan::select`].
#[derive(Debug, Clone, Copy)]
pub struct CandidateTarget<'a> {
    target: PixelEncoding<'a>,
    preference: u32,
    consumer_cost: crate::negotiate::ConversionCost,
}

impl<'a> CandidateTarget<'a> {
    /// Offer the exact sample and color interpretation requested by the consumer.
    pub const fn new(target: PixelEncoding<'a>) -> Self {
        Self {
            target,
            preference: 0,
            consumer_cost: crate::negotiate::ConversionCost::ZERO,
        }
    }

    /// Prefer smaller values among otherwise equivalent routes.
    pub const fn with_preference(mut self, preference: u32) -> Self {
        self.preference = preference;
        self
    }

    /// Account for downstream conversion effort and loss inside the consumer.
    pub const fn with_consumer_cost(mut self, effort: u16, loss: u16) -> Self {
        self.consumer_cost = crate::negotiate::ConversionCost::new(effort, loss);
        self
    }

    /// Target encoding offered by this candidate.
    pub const fn encoding(&self) -> PixelEncoding<'a> {
        self.target
    }
}

impl<'a> From<PixelEncoding<'a>> for CandidateTarget<'a> {
    fn from(target: PixelEncoding<'a>) -> Self {
        Self::new(target)
    }
}

impl From<PixelDescriptor> for CandidateTarget<'static> {
    fn from(descriptor: PixelDescriptor) -> Self {
        Self::new(PixelEncoding::from(descriptor))
    }
}

impl From<&PixelDescriptor> for CandidateTarget<'static> {
    fn from(descriptor: &PixelDescriptor) -> Self {
        Self::new(PixelEncoding::from(*descriptor))
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

    /// Borrow the selected conversion plan.
    pub const fn plan(&self) -> &ConvertPlan {
        &self.plan
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
    worker_limit: Option<u64>,
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

fn profile_defines_transfer(profile: &crate::ColorProfileSource<'_>) -> bool {
    match profile {
        crate::ColorProfileSource::Icc(_) | crate::ColorProfileSource::Parametric(_) => true,
        _ => profile
            .resolve()
            .is_some_and(|(_, tf)| tf != TransferFunction::Unknown),
    }
}

pub(crate) struct ResolvedProfiles {
    same_profile: bool,
    src_unverified_tf: bool,
    dst_unverified_tf: bool,
    src_is_icc: bool,
    resolved_src: Option<(crate::ColorPrimaries, TransferFunction)>,
}

impl ResolvedProfiles {
    #[inline]
    fn from_descriptors(from: PixelDescriptor, to: PixelDescriptor) -> Self {
        Self {
            same_profile: from.primaries == to.primaries && from.transfer() == to.transfer(),
            src_unverified_tf: matches!(from.color_model(), ColorModel::Rgb | ColorModel::Gray)
                && from.transfer() == TransferFunction::Unknown,
            dst_unverified_tf: matches!(to.color_model(), ColorModel::Rgb | ColorModel::Gray)
                && to.transfer() == TransferFunction::Unknown,
            src_is_icc: false,
            resolved_src: None,
        }
    }
}

type PrepareProfilesFn =
    fn(PixelEncoding<'_>, PixelEncoding<'_>) -> Result<ResolvedProfiles, At<ConvertError>>;
type CompileExplicitFn = fn(
    PixelDescriptor,
    PixelDescriptor,
    &ConvertOptions,
    &crate::caps::PlannerHooks,
) -> Result<RowRecipe, At<ConvertError>>;
type CompileEncodingFn = fn(
    PixelEncoding<'_>,
    PixelEncoding<'_>,
    &ConvertOptions,
    Option<&dyn PluggableCms>,
    &crate::caps::PlannerHooks,
) -> Result<RowRecipe, At<ConvertError>>;

#[derive(Clone, Copy)]
pub(crate) struct ProfileHooks {
    prepare: Option<PrepareProfilesFn>,
    compile_explicit: CompileExplicitFn,
    compile_encoding: CompileEncodingFn,
}

pub(crate) const PROFILE_HOOKS: ProfileHooks = ProfileHooks {
    prepare: Some(prepare_with_profiles),
    compile_explicit: RowRecipe::new_explicit_hooks,
    compile_encoding: RowRecipe::new_with_encoding_hooks,
};

fn compile_descriptors_with_cms(
    source: PixelEncoding<'_>,
    target: PixelEncoding<'_>,
    options: &ConvertOptions,
    cms: Option<&dyn PluggableCms>,
    hooks: &crate::caps::PlannerHooks,
) -> Result<RowRecipe, At<ConvertError>> {
    RowRecipe::new_explicit_with_cms_hooks(
        source.descriptor(),
        target.descriptor(),
        options,
        cms,
        hooks,
    )
}

pub(crate) const DESCRIPTOR_CMS_HOOKS: ProfileHooks = ProfileHooks {
    prepare: None,
    compile_explicit: RowRecipe::new_explicit_hooks,
    compile_encoding: compile_descriptors_with_cms,
};

#[cfg(feature = "hdr-experimental")]
pub(crate) type ExplicitHdrFn = fn(
    PixelEncoding<'_>,
    PixelEncoding<'_>,
    bool,
    &ConvertOptions,
    &ConversionSpec<'_>,
    &crate::caps::PlannerHooks,
) -> Result<(RowRecipe, bool), At<ConvertError>>;

fn prepare_with_profiles(
    source: PixelEncoding<'_>,
    target: PixelEncoding<'_>,
) -> Result<ResolvedProfiles, At<ConvertError>> {
    let from = source.descriptor();
    let to = target.descriptor();
    let src_profile = crate::output::current_profile_for(source)?;
    let dst_profile = crate::output::current_profile_for(target)?;
    let same_profile = crate::output::profiles_match(&src_profile, &dst_profile)
        || (from == to && source.color_context() == target.color_context());
    let src_unverified_tf = matches!(from.color_model(), ColorModel::Rgb | ColorModel::Gray)
        && from.transfer() == TransferFunction::Unknown
        && !profile_defines_transfer(&src_profile);
    let dst_unverified_tf = matches!(to.color_model(), ColorModel::Rgb | ColorModel::Gray)
        && to.transfer() == TransferFunction::Unknown
        && !profile_defines_transfer(&dst_profile);
    let src_is_icc = matches!(src_profile, crate::ColorProfileSource::Icc(_));
    let resolved_src = if same_profile
        && (from.primaries == crate::ColorPrimaries::Unknown
            || from.transfer() == TransferFunction::Unknown)
    {
        src_profile.resolve()
    } else {
        None
    };
    Ok(ResolvedProfiles {
        same_profile,
        src_unverified_tf,
        dst_unverified_tf,
        src_is_icc,
        resolved_src,
    })
}

#[cfg(feature = "hdr-experimental")]
pub(crate) fn compile_explicit_hdr(
    source: PixelEncoding<'_>,
    target: PixelEncoding<'_>,
    same_profile: bool,
    options: &ConvertOptions,
    spec: &ConversionSpec<'_>,
    hooks: &crate::caps::PlannerHooks,
) -> Result<(RowRecipe, bool), At<ConvertError>> {
    let hdr = spec.hdr.expect("compile_explicit_hdr called with spec.hdr");
    let (from, to, effective_src_tf) = if hooks.profiles.is_some() {
        let src_profile = crate::output::current_profile_for(source)?;
        let dst_profile = crate::output::current_profile_for(target)?;
        if !same_profile && (src_profile.resolve().is_none() || dst_profile.resolve().is_none()) {
            return Err(whereat::at!(ConvertError::HdrWithOpaqueProfileUnsupported));
        }
        let from = src_profile
            .resolve()
            .map(|(p, t)| source.descriptor().with_primaries(p).with_transfer(t))
            .unwrap_or_else(|| source.descriptor());
        let to = dst_profile
            .resolve()
            .map(|(p, t)| target.descriptor().with_primaries(p).with_transfer(t))
            .unwrap_or_else(|| target.descriptor());
        let eff_tf = src_profile
            .resolve()
            .map_or(source.descriptor().transfer(), |(_, tf)| tf);
        (from, to, eff_tf)
    } else {
        let from = source.descriptor();
        let to = target.descriptor();
        (from, to, from.transfer())
    };
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
            crate::NativePlan::new_with_tone_mapper_unbound(from, to, *hdr, Arc::clone(mapper))?
        }
        None => crate::NativePlan::new_with_hdr_config_unbound(from, to, *hdr)?,
    };
    let native = if let Some(white) = source.color_context().and_then(ColorContext::diffuse_white)
        && from.transfer() == TransferFunction::Linear
    {
        native.with_linear_input_white(white, hdr.source_peak_nits)
    } else {
        native
    };
    let set_hdr_output_luminance = matches!(
        effective_src_tf,
        TransferFunction::Linear | TransferFunction::Pq
    );
    Ok((RowRecipe::native(native), set_hdr_output_luminance))
}

impl ConvertPlan {
    /// Check authority and policy and compile the actual route.
    #[track_caller]
    pub fn build(
        source: PixelEncoding<'_>,
        target: PixelEncoding<'_>,
        spec: &ConversionSpec<'_>,
    ) -> Result<Self, At<ConvertError>> {
        Self::build_with_caps::<{ crate::caps::ALL }>(source, target, spec)
    }

    /// Compile a route and bind only the kernel steps permitted by `CAPS`.
    ///
    /// Core planning is non-generic and shared across capability masks (via
    /// compile-time [`PlannerHooks`](crate::caps::PlannerHooks)); pure
    /// single-depth integer layout masks route through a dedicated minimal
    /// layout planner so depth/transfer/gamut/ICC planners dead-strip cleanly.
    #[inline(always)]
    #[track_caller]
    pub(crate) fn build_with_caps<const CAPS: u64>(
        source: PixelEncoding<'_>,
        target: PixelEncoding<'_>,
        spec: &ConversionSpec<'_>,
    ) -> Result<Self, At<ConvertError>> {
        let mut plan = if const { crate::caps::is_layout_only(CAPS) } {
            Self::build_layout_only(source, target, spec)?
        } else if source.color_context().is_none() && target.color_context().is_none() {
            Self::build_general(
                source,
                target,
                spec,
                &const { crate::caps::PlannerHooks::for_caps_descriptors::<CAPS>() },
            )?
        } else {
            Self::build_general(
                source,
                target,
                spec,
                &const { crate::caps::PlannerHooks::for_caps::<CAPS>() },
            )?
        };
        plan.recipe.bind::<CAPS>()?;
        Ok(plan)
    }

    #[track_caller]
    fn build_layout_only(
        source: PixelEncoding<'_>,
        target: PixelEncoding<'_>,
        spec: &ConversionSpec<'_>,
    ) -> Result<Self, At<ConvertError>> {
        let from = source.descriptor();
        let to = target.descriptor();
        if !spec.float_facts.is_empty() && !from.channel_type().is_float() {
            return Err(whereat::at!(ConvertError::FloatFactsUnsatisfied {
                required: spec.float_facts,
                actual: zenpixels::sample::FloatFacts::NONE,
            }));
        }
        // Without `caps::PROFILES`, reject any attached ICC/CICP/parametric profile
        // (unless the entire PixelEncoding is an identical native-model identity).
        let src_has_profile = source
            .color_context()
            .is_some_and(|c| c.as_profile_source().is_some());
        let dst_has_profile = target
            .color_context()
            .is_some_and(|c| c.as_profile_source().is_some());
        let exact_same_encoding = from == to && source.color_context() == target.color_context();
        if (src_has_profile || dst_has_profile) && !exact_same_encoding {
            return Err(whereat::at!(ConvertError::NeedsCms { from, to }));
        }
        let same_profile = exact_same_encoding
            || (from.primaries == to.primaries && from.transfer() == to.transfer());

        let mut options =
            crate::interpretation::resolve_options(source.color_context(), spec.options());
        if options.alpha_policy() == AlphaPolicy::DiscardIfOpaque
            && from.has_alpha()
            && !to.has_alpha()
        {
            if from.alpha() == Some(AlphaMode::Opaque)
                || spec
                    .precision
                    .is_some_and(|p| p.alpha() == AlphaContent::Opaque)
            {
                options = options.with_alpha_policy(AlphaPolicy::DiscardUnchecked);
            } else if spec
                .precision
                .is_some_and(|p| p.alpha() == AlphaContent::Translucent)
            {
                return Err(whereat::at!(ConvertError::AlphaNotOpaque));
            }
        }

        let src_unverified_tf = matches!(from.color_model(), ColorModel::Rgb | ColorModel::Gray)
            && from.transfer() == TransferFunction::Unknown;
        let dst_unverified_tf = matches!(to.color_model(), ColorModel::Rgb | ColorModel::Gray)
            && to.transfer() == TransferFunction::Unknown;
        if src_unverified_tf != dst_unverified_tf {
            return Err(whereat::at!(
                ConvertError::UnknownTransferRelabelForbidden { from, to }
            ));
        }

        if let Some(precision) = spec.precision {
            if from.alpha() == Some(AlphaMode::Opaque)
                && precision.alpha() == AlphaContent::Translucent
            {
                return Err(whereat::at!(ConvertError::NoPath { from, to }));
            }
            let mut offer = ConversionOffer::new(to);
            if matches!(options.alpha_policy(), AlphaPolicy::CompositeOnto { .. }) {
                offer = offer.with_matte();
            }
            let effect = ConversionEffect::of(&precision, &offer);
            if spec.choice == OutputChoice::Hidden && !effect.preserves_modeled_capacity() {
                return Err(whereat::at!(ConvertError::CapacityLoss(effect)));
            }
        }

        #[cfg(feature = "hdr-experimental")]
        if spec.hdr.is_some() {
            return Err(whereat::at!(ConvertError::NoPath { from, to }));
        }

        crate::convert::validate_descriptors(from, to)?;
        if crate::convert::needs_cms_for_color_model(&from, &to) {
            return Err(whereat::at!(ConvertError::NeedsCms { from, to }));
        }
        let recipe = if from == to {
            RowRecipe::identity(from, to)
        } else {
            let plan = crate::NativePlan::plan_layout_only(from, to, &options, spec.exact())?;
            RowRecipe::native(plan)
        };
        let worker_limit = spec.workspace_limit.map(|(_, limit)| limit);
        if let Some((width, limit)) = spec.workspace_limit {
            let bytes = recipe.requirements(width)?.worker_heap();
            if !fits_worker_limit(bytes, limit) {
                return Err(whereat::at!(ConvertError::WorkerBudgetExceeded {
                    bytes,
                    limit
                }));
            }
        }
        let output = resolve_output(source, target, same_profile, &options);
        if let Some(requested) = target.color_context()
            && (requested.diffuse_white().is_some_and(|white| {
                output.as_ref().and_then(ColorContext::diffuse_white) != Some(white)
            }) || requested
                .hdr_peak()
                .is_some_and(|peak| output.as_ref().and_then(ColorContext::hdr_peak) != Some(peak)))
        {
            return Err(whereat::at!(ConvertError::TargetLuminanceMismatch));
        }
        Ok(Self {
            recipe,
            input: source.color_context().cloned().map(Arc::new),
            output: output.map(Arc::new),
            worker_limit,
        })
    }

    #[track_caller]
    fn build_general(
        source: PixelEncoding<'_>,
        target: PixelEncoding<'_>,
        spec: &ConversionSpec<'_>,
        hooks: &crate::caps::PlannerHooks,
    ) -> Result<Self, At<ConvertError>> {
        let from = source.descriptor();
        let to = target.descriptor();
        if !spec.float_facts.is_empty() && !from.channel_type().is_float() {
            return Err(whereat::at!(ConvertError::FloatFactsUnsatisfied {
                required: spec.float_facts,
                actual: zenpixels::sample::FloatFacts::NONE,
            }));
        }
        let prof = if let Some(prepare) = hooks.profiles.and_then(|p| p.prepare) {
            prepare(source, target)?
        } else {
            let src_has_profile = source
                .color_context()
                .is_some_and(|c| c.as_profile_source().is_some());
            let dst_has_profile = target
                .color_context()
                .is_some_and(|c| c.as_profile_source().is_some());
            let exact_same = from == to && source.color_context() == target.color_context();
            if (src_has_profile || dst_has_profile) && !exact_same {
                return Err(whereat::at!(ConvertError::NeedsCms { from, to }));
            }
            let mut rp = ResolvedProfiles::from_descriptors(from, to);
            rp.same_profile = rp.same_profile || exact_same;
            rp
        };
        let same_profile = prof.same_profile;

        let mut options =
            crate::interpretation::resolve_options(source.color_context(), spec.options());
        if options.alpha_policy() == AlphaPolicy::DiscardIfOpaque
            && from.has_alpha()
            && !to.has_alpha()
        {
            if from.alpha() == Some(AlphaMode::Opaque)
                || spec
                    .precision
                    .is_some_and(|p| p.alpha() == AlphaContent::Opaque)
            {
                options = options.with_alpha_policy(AlphaPolicy::DiscardUnchecked);
            } else if spec
                .precision
                .is_some_and(|p| p.alpha() == AlphaContent::Translucent)
            {
                return Err(whereat::at!(ConvertError::AlphaNotOpaque));
            }
        }

        if prof.src_unverified_tf != prof.dst_unverified_tf {
            return Err(whereat::at!(
                ConvertError::UnknownTransferRelabelForbidden { from, to }
            ));
        }

        if let Some(precision) = spec.precision {
            if from.alpha() == Some(AlphaMode::Opaque)
                && precision.alpha() == AlphaContent::Translucent
            {
                return Err(whereat::at!(ConvertError::NoPath { from, to }));
            }
            let mut offer = ConversionOffer::new(to);
            if matches!(options.alpha_policy(), AlphaPolicy::CompositeOnto { .. }) {
                offer = offer.with_matte();
            }
            let effect = ConversionEffect::of(&precision, &offer);
            if spec.choice == OutputChoice::Hidden && !effect.preserves_modeled_capacity() {
                return Err(whereat::at!(ConvertError::CapacityLoss(effect)));
            }
        }

        let (mut recipe, hdr_output_luminance) =
            compile(source, target, &prof, &options, spec, hooks)?;
        // The row endpoints use the caller's carriers even if the kernel resolved
        // fields of a compact descriptor using attached profile evidence.
        recipe.from = from;
        recipe.to = to;
        recipe.apply_input_facts(spec.float_facts);
        let worker_limit = spec.workspace_limit.map(|(_, limit)| limit);
        if let Some((width, limit)) = spec.workspace_limit {
            let bytes = recipe.requirements(width)?.worker_heap();
            if !fits_worker_limit(bytes, limit) {
                return Err(whereat::at!(ConvertError::WorkerBudgetExceeded {
                    bytes,
                    limit
                }));
            }
        }
        let output = resolve_output(source, target, same_profile, &options);
        #[cfg(not(feature = "hdr-experimental"))]
        let _ = hdr_output_luminance;
        #[cfg(feature = "hdr-experimental")]
        let output = if hdr_output_luminance && let Some(hdr) = spec.hdr {
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
            return Err(whereat::at!(ConvertError::TargetLuminanceMismatch));
        }
        Ok(Self {
            recipe,
            input: source.color_context().cloned().map(Arc::new),
            output: output.map(Arc::new),
            worker_limit,
        })
    }

    /// Explicitly reinterpret sample bytes across an `Unknown` ↔ known transfer boundary
    /// without changing layout, depth, alpha, primaries, or signal range.
    ///
    /// Use this when the caller has out-of-band knowledge of the true transfer curve
    /// (and compose with [`ConvertPlan::then`] when further conversion is needed).
    /// Ordinary [`ConvertPlan::build`] calls refuse unverified `Unknown` ↔ known
    /// transfer changes with [`ConvertError::UnknownTransferRelabelForbidden`].
    pub fn relabel_transfer(
        from: PixelDescriptor,
        to: PixelDescriptor,
    ) -> Result<Self, At<ConvertError>> {
        crate::convert::validate_descriptors(from, to)?;
        let supported_sample = matches!(
            from.channel_type(),
            crate::ChannelType::U8
                | crate::ChannelType::U16
                | crate::ChannelType::F16
                | crate::ChannelType::F32
        );
        let only_transfer_differs = from.pixel_format() == to.pixel_format()
            && from.alpha() == to.alpha()
            && from.primaries == to.primaries
            && from.signal_range == to.signal_range;
        let uses_transfer = matches!(from.color_model(), ColorModel::Rgb | ColorModel::Gray);
        let bridges_unknown = from.transfer() != to.transfer()
            && (from.transfer() == TransferFunction::Unknown
                || to.transfer() == TransferFunction::Unknown);
        if !supported_sample || !only_transfer_differs || !uses_transfer || !bridges_unknown {
            return Err(whereat::at!(ConvertError::NoPath { from, to }));
        }
        Ok(Self {
            recipe: RowRecipe::relabel_transfer(from, to),
            input: None,
            output: None,
            worker_limit: None,
        })
    }

    /// Select an admitted route and return its executable plan.
    #[track_caller]
    pub fn select(
        source: PixelEncoding<'_>,
        targets: &[CandidateTarget<'_>],
        spec: &ConversionSpec<'_>,
    ) -> Result<SelectedConversion, At<ConvertError>> {
        Self::select_with_caps::<{ crate::caps::ALL }>(source, targets, spec)
    }

    #[inline(always)]
    #[track_caller]
    pub(crate) fn select_with_caps<const CAPS: u64>(
        source: PixelEncoding<'_>,
        targets: &[CandidateTarget<'_>],
        spec: &ConversionSpec<'_>,
    ) -> Result<SelectedConversion, At<ConvertError>> {
        Self::select_with_fn(source, targets, spec, Self::build_with_caps::<CAPS>)
    }

    #[track_caller]
    fn select_with_fn(
        source: PixelEncoding<'_>,
        targets: &[CandidateTarget<'_>],
        spec: &ConversionSpec<'_>,
        build_fn: fn(
            PixelEncoding<'_>,
            PixelEncoding<'_>,
            &ConversionSpec<'_>,
        ) -> Result<Self, At<ConvertError>>,
    ) -> Result<SelectedConversion, At<ConvertError>> {
        if targets.is_empty() {
            return Err(whereat::at!(ConvertError::EmptyFormatList));
        }
        let mut admitted = crate::route::RouteFrontier::new();
        let mut intent_best: Option<((u32, crate::route::TieKey), SelectedConversion)> = None;
        let mut refusal = None;
        let prov = spec
            .provenance
            .unwrap_or_else(|| Provenance::from_source(source.descriptor()));

        for (index, candidate) in targets.iter().enumerate() {
            match build_fn(source, candidate.target, spec) {
                Ok(plan) => {
                    let scratch = spec
                        .workspace_limit
                        .map(|(width, _)| plan.workspace(WorkExtent::new(width, 1)))
                        .transpose()?
                        .map_or(0, |memory| memory.native_worker());
                    if let Some(intent) = spec.intent {
                        let score = crate::negotiate::score_target(
                            source.descriptor(),
                            prov,
                            candidate.target.descriptor(),
                            candidate.consumer_cost,
                            intent,
                        );
                        let key = (
                            score,
                            plan.recipe
                                .rank()
                                .tie_key(candidate.preference, scratch, index),
                        );
                        if intent_best.as_ref().is_none_or(|(best, _)| key < *best) {
                            intent_best = Some((key, SelectedConversion { index, plan }));
                        }
                    } else {
                        let pref = candidate.preference.saturating_add(
                            u32::from(candidate.consumer_cost.loss) * 4
                                + u32::from(candidate.consumer_cost.effort),
                        );
                        admitted.offer(
                            plan.recipe.rank(),
                            pref,
                            scratch,
                            index,
                            SelectedConversion { index, plan },
                        );
                    }
                }
                Err(e)
                    if matches!(
                        e.error(),
                        ConvertError::AllocationFailed | ConvertError::CmsBackend(_)
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
        if let Some((_, selected)) = intent_best {
            return Ok(selected);
        }
        if let Some(selected) = admitted.finish() {
            return Ok(selected);
        }
        let (index, cause) = refusal.expect("non-empty targets had no admitted plan");
        Err(cause.map_error(|c| ConvertError::NoCandidate {
            index,
            cause: alloc::boxed::Box::new(c),
        }))
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

    /// Floating-point sample invariants required on input views (`FloatFacts::NONE` when unstated).
    pub const fn input_float_facts(&self) -> zenpixels::sample::FloatFacts {
        self.recipe.input_facts
    }

    /// Floating-point sample invariants proven for output views (`FloatFacts::NONE` on integer outputs).
    pub const fn output_float_facts(&self) -> zenpixels::sample::FloatFacts {
        self.recipe.output_facts
    }

    /// Whether the operation leaves active sample bytes unchanged.
    pub fn is_identity(&self) -> bool {
        !self.recipe.needs_pixel_work()
    }

    /// Compose with an explicitly materialized boundary, including its interpretation and rounding.
    pub fn then(self, next: Self) -> Result<Self, At<ConvertError>> {
        if self.output_encoding() != next.source_encoding() {
            return Err(whereat::at!(ConvertError::IntermediateMismatch));
        }
        if !self.output_float_facts().contains(next.input_float_facts()) {
            return Err(whereat::at!(ConvertError::FloatFactsUnsatisfied {
                required: next.input_float_facts(),
                actual: self.output_float_facts(),
            }));
        }
        let worker_limit = match (self.worker_limit, next.worker_limit) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        let input_facts = self.recipe.input_facts;
        let mut recipe = self.recipe.then(next.recipe);
        recipe.apply_input_facts(input_facts);
        Ok(Self {
            recipe,
            input: self.input,
            output: next.output,
            worker_limit,
        })
    }

    /// Report memory for the same layout used to prepare this recipe.
    pub fn workspace(
        &self,
        extent: WorkExtent,
    ) -> Result<AllocationRequirements, At<ConvertError>> {
        self.recipe.requirements(extent.width)
    }

    /// Build an independent worker and reserve scratch for rows at most this wide.
    pub fn prepare(self, max_width: u32) -> Result<RowConverter, At<ConvertError>> {
        let mut worker = self.into_worker();
        worker.prepare(max_width)?;
        Ok(worker)
    }

    /// Duplicate a reusable native or shared-backend plan and prepare an independent worker.
    pub fn prepare_shared(&self, max_width: u32) -> Result<RowConverter, At<ConvertError>> {
        self.try_clone()?.prepare(max_width)
    }

    /// Duplicate a reusable native or shared-backend plan.
    pub fn try_clone(&self) -> Result<Self, At<ConvertError>> {
        Ok(Self {
            recipe: self.recipe.clone_for_worker()?,
            input: self.input.clone(),
            output: self.output.clone(),
            worker_limit: self.worker_limit,
        })
    }

    pub(crate) fn into_worker(self) -> RowConverter {
        let mut worker = self.recipe.into_worker(self.output);
        worker.set_source_context(self.input);
        worker.set_worker_limit(self.worker_limit);
        worker
    }
}

fn compile(
    source: PixelEncoding<'_>,
    target: PixelEncoding<'_>,
    prof: &ResolvedProfiles,
    options: &ConvertOptions,
    spec: &ConversionSpec<'_>,
    hooks: &crate::caps::PlannerHooks,
) -> Result<(RowRecipe, bool), At<ConvertError>> {
    let (mut from, to) = (source.descriptor(), target.descriptor());
    let same_profile = prof.same_profile;
    let hdr_requested = false;
    #[cfg(feature = "hdr-experimental")]
    let hdr_requested = hdr_requested || spec.hdr.is_some();
    let allow_identity = from == to
        && same_profile
        && (!crate::convert::needs_cms_for_color_model(&from, &to) || prof.src_is_icc);
    if allow_identity && !hdr_requested {
        crate::convert::validate_descriptors(from, to)?;
        return Ok((RowRecipe::identity(from, to), false));
    }
    if let Some((primaries, transfer)) = prof.resolved_src {
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
        && let Some(recipe) = exact_path(from, to, spec.workspace_limit, hooks)
    {
        return Ok((recipe, false));
    }
    if spec.exact() {
        return Err(whereat::at!(ConvertError::NoPath { from, to }));
    }
    #[cfg(feature = "hdr-experimental")]
    if spec.hdr.is_some() {
        let Some(hdr_fn) = hooks.explicit_hdr else {
            return Err(whereat::at!(ConvertError::NoPath { from, to }));
        };
        return hdr_fn(source, target, same_profile, options, spec, hooks);
    }
    if same_profile && from.transfer() == to.transfer() && from.primaries == to.primaries {
        let recipe = if let Some(p) = hooks.profiles {
            (p.compile_explicit)(from, to, options, hooks)?
        } else {
            crate::convert::validate_descriptors(from, to)?;
            crate::convert::validate_luma_policy(from, to, options)?;
            RowRecipe::native(crate::NativePlan::new_explicit_core(
                from, to, options, hooks,
            )?)
        };
        return Ok((recipe, false));
    }
    if let Some(p) = hooks.profiles {
        return Ok((
            (p.compile_encoding)(source, target, options, spec.cms(), hooks)?,
            false,
        ));
    }
    // No `caps::PROFILES`: descriptor-only path (still supports native gamut
    // matrices, fused matlut, and SDR/HDR transfers without ICC/CMS dispatch).
    if spec.cms().is_some() && crate::convert::needs_cms_for_color_model(&from, &to) {
        return Err(whereat::at!(ConvertError::NeedsCms { from, to }));
    }
    crate::convert::validate_descriptors(from, to)?;
    crate::convert::validate_luma_policy(from, to, options)?;
    Ok((
        RowRecipe::native(crate::NativePlan::new_explicit_core(
            from, to, options, hooks,
        )?),
        false,
    ))
}

// The finite exact subgraph comprises the endpoints and the endpoint
// layout/depth cross-products. Existing native code proves each edge.
fn exact_path(
    from: PixelDescriptor,
    to: PixelDescriptor,
    limit: Option<(u32, u64)>,
    hooks: &crate::caps::PlannerHooks,
) -> Option<RowRecipe> {
    #[derive(Clone, Copy)]
    struct State(PixelDescriptor);

    let edge = |from: State, to: State| -> Option<RowRecipe> {
        crate::NativePlan::new_preserving_samples_core(from.0, to.0, hooks)
            .ok()
            .map(RowRecipe::native)
    };

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
                PixelDescriptor::new(depth, layout.layout(), layout.alpha(), layout.transfer())
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

pub(crate) fn fits_worker_limit(bound: ByteBound, limit: u64) -> bool {
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
