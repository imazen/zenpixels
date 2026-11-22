//! Pre-computed row converter.
//!
//! [`RowConverter`] wraps a [`crate::ConvertPlan`] with preallocated scratch
//! buffers for zero-allocation-per-row streaming conversion.

use crate::TransferFunction;
use alloc::boxed::Box;

use crate::convert::{ConvertScratch, NativePlan, convert_row_buffered, convert_row_prepared};
use crate::{ChannelLayout, ConvertError, PixelDescriptor};
use whereat::{At, ResultAtExt};

/// Prepared row converter with independent writable workspace.
pub struct RowConverter {
    recipe: RowRecipe,
    workspace: WorkerSpace,
    prepared_width: Option<u32>,
    worker_limit: Option<u64>,
    color: Option<alloc::sync::Arc<crate::ColorContext>>,
    source: Option<alloc::sync::Arc<crate::ColorContext>>,
}

#[derive(Debug)]
pub(crate) struct RowRecipe {
    pub(crate) from: PixelDescriptor,
    pub(crate) to: PixelDescriptor,
    stages: alloc::vec::Vec<Stage>,
    relabel_before: Option<(PixelDescriptor, PixelDescriptor)>,
    relabel_after: Option<(PixelDescriptor, PixelDescriptor)>,
    pub(crate) input_facts: zenpixels::sample::FloatFacts,
    pub(crate) output_facts: zenpixels::sample::FloatFacts,
}

#[derive(Debug)]
struct Stage {
    from: PixelDescriptor,
    to: PixelDescriptor,
    operation: Operation,
}

enum Operation {
    Native(NativePlan),
    Shared(alloc::sync::Arc<dyn crate::cms::RowTransform>),
    Owned(Box<dyn crate::cms::RowTransformMut>),
}

pub(crate) type HdrPivotFn = fn(
    PixelDescriptor,
    PixelDescriptor,
    TransferFunction,
    TransferFunction,
    crate::ColorProfileSource<'_>,
    crate::ColorProfileSource<'_>,
    bool,
    &crate::policy::ConvertOptions,
    Option<&dyn crate::cms::PluggableCms>,
    &crate::caps::PlannerHooks,
) -> Result<RowRecipe, At<ConvertError>>;

#[track_caller]
fn dispatch_cms_chain(
    source: crate::ColorProfileSource<'_>,
    target: crate::ColorProfileSource<'_>,
    from: PixelDescriptor,
    to: PixelDescriptor,
    options: &crate::policy::ConvertOptions,
    cms: Option<&dyn crate::cms::PluggableCms>,
    hooks: &crate::caps::PlannerHooks,
) -> Result<Option<Operation>, At<ConvertError>> {
    let map_error = |e: At<crate::cms::CmsPluginError>| {
        e.map_error(|error| ConvertError::CmsBackend(alloc::sync::Arc::new(error)))
    };
    use crate::cms::{CmsRequest, CmsSupport};
    if let Some(plugin) = cms {
        let request = CmsRequest::new(source.clone(), target.clone(), from, to, *options);
        match plugin.build(&request).map_err(map_error)? {
            CmsSupport::Shared(t) => return Ok(Some(Operation::Shared(t))),
            CmsSupport::Owned(t) => return Ok(Some(Operation::Owned(t))),
            CmsSupport::Declined => {}
        }
    }
    if let Some(result) = crate::cms_lite::ZenCmsLite::build_plan_with_hooks(
        source,
        target,
        from.pixel_format(),
        to.pixel_format(),
        options,
        hooks,
    ) {
        return whereat::at_crate!(result).map(|c| Some(Operation::Native(c)));
    }
    Ok(None)
}

impl core::fmt::Debug for Operation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Native(plan) => core::fmt::Debug::fmt(plan, f),
            Self::Shared(_) => f.write_str("SharedCms"),
            Self::Owned(_) => f.write_str("OwnedCms"),
        }
    }
}

impl RowRecipe {
    pub(crate) fn identity(from: PixelDescriptor, to: PixelDescriptor) -> Self {
        Self {
            from,
            to,
            stages: alloc::vec::Vec::new(),
            relabel_before: None,
            relabel_after: None,
            input_facts: zenpixels::sample::FloatFacts::NONE,
            output_facts: zenpixels::sample::FloatFacts::NONE,
        }
    }

    pub(crate) fn relabel_transfer(from: PixelDescriptor, to: PixelDescriptor) -> Self {
        Self {
            from,
            to,
            stages: alloc::vec::Vec::new(),
            relabel_before: Some((from, to)),
            relabel_after: None,
            input_facts: zenpixels::sample::FloatFacts::NONE,
            output_facts: zenpixels::sample::FloatFacts::NONE,
        }
    }

    pub(crate) fn native(plan: NativePlan) -> Self {
        let (from, to) = (plan.from(), plan.to());
        let (in_facts, out_facts) = (plan.input_facts(), plan.output_facts());
        let mut recipe = Self::single(from, to, Operation::Native(plan));
        recipe.input_facts = in_facts;
        recipe.output_facts = out_facts;
        recipe
    }

    fn single(from: PixelDescriptor, to: PixelDescriptor, operation: Operation) -> Self {
        Self {
            from,
            to,
            stages: alloc::vec![Stage {
                from,
                to,
                operation
            }],
            relabel_before: None,
            relabel_after: None,
            input_facts: zenpixels::sample::FloatFacts::NONE,
            output_facts: zenpixels::sample::FloatFacts::NONE,
        }
    }

    pub(crate) fn apply_input_facts(&mut self, initial: zenpixels::sample::FloatFacts) {
        use zenpixels::{AlphaMode, sample::FloatFacts};
        let initial = if self.from.channel_type().is_float() {
            initial
        } else {
            FloatFacts::NONE
        };
        self.input_facts = initial;
        let promotes_undefined = |(old, new): (PixelDescriptor, PixelDescriptor)| {
            old.alpha() == Some(AlphaMode::Undefined) && new.alpha() != Some(AlphaMode::Undefined)
        };
        if self.stages.is_empty() {
            let survives = self.from.channel_type() == self.to.channel_type()
                && self.to.channel_type().is_float()
                && !promotes_undefined((self.from, self.to))
                && !self.relabel_before.is_some_and(promotes_undefined)
                && !self.relabel_after.is_some_and(promotes_undefined);
            self.output_facts = if survives { initial } else { FloatFacts::NONE };
            return;
        }
        let mut cur = if self.relabel_before.is_some_and(promotes_undefined) {
            FloatFacts::NONE
        } else {
            initial
        };
        for stage in &mut self.stages {
            if promotes_undefined((stage.from, stage.to)) && stage.from == stage.to {
                cur = FloatFacts::NONE;
            }
            match &mut stage.operation {
                Operation::Native(plan) => {
                    plan.propagate_float_facts(cur);
                    cur = plan.output_facts();
                }
                Operation::Shared(_) | Operation::Owned(_) => {
                    cur = FloatFacts::NONE;
                }
            }
        }
        if self.relabel_after.is_some_and(promotes_undefined) || !self.to.channel_type().is_float()
        {
            cur = FloatFacts::NONE;
        }
        self.output_facts = cur;
    }

    pub(crate) fn then(mut self, mut next: Self) -> Self {
        debug_assert_eq!(self.to.bytes_per_pixel(), next.from.bytes_per_pixel());
        let initial = self.input_facts;
        let left_identity = self.is_identity_operation();
        self.to = next.to;
        if left_identity {
            self.stages.clear();
            if self.relabel_before.is_none() {
                self.relabel_before = next.relabel_before;
                self.relabel_after = next.relabel_after;
            } else {
                self.relabel_after = next
                    .relabel_after
                    .or(next.relabel_before)
                    .or(self.relabel_after);
            }
        } else {
            self.relabel_after = next
                .relabel_after
                .or(next.relabel_before)
                .or(self.relabel_after);
        }
        if !next.is_identity_operation() {
            if let Some(left) = self.stages.last_mut()
                && let Some(right) = next.stages.first()
                && let (Operation::Native(a), Operation::Native(b)) =
                    (&left.operation, &right.operation)
                && let Some(joined) = a.compose_preserving(b)
            {
                left.operation = Operation::Native(joined);
                left.to = right.to;
                self.stages.extend(next.stages.into_iter().skip(1));
            } else {
                self.stages.append(&mut next.stages);
            }
        }
        self.apply_input_facts(initial);
        self
    }

    fn is_identity_operation(&self) -> bool {
        self.stages.is_empty()
            || matches!(self.stages.as_slice(), [Stage { operation: Operation::Native(p), .. }] if p.is_identity())
    }

    #[inline(always)]
    pub(crate) fn bind<const CAPS: u64>(&mut self) -> Result<(), At<ConvertError>> {
        for stage in &mut self.stages {
            if let Operation::Native(plan) = &mut stage.operation {
                plan.bind::<CAPS>()?;
            }
        }
        Ok(())
    }

    pub(crate) fn needs_pixel_work(&self) -> bool {
        !self.is_identity_operation()
    }

    /// Create a converter from `from` to `to`.
    ///
    /// Alpha removal is unconditional, preserving the legacy constructor's
    /// behavior. Use `new_explicit` for policy control and `adapt::check_opaque`
    /// for an explicit opacity preflight.
    ///
    /// Cross-profile conversions (different primaries or transfer) go
    /// through the default CMS dispatch chain — see
    /// [`new_explicit_with_cms`](Self::new_explicit_with_cms) for details.
    /// Returns `Err` if no conversion path exists between the formats.
    #[track_caller]
    pub(crate) fn new(
        from: PixelDescriptor,
        to: PixelDescriptor,
    ) -> Result<Self, At<ConvertError>> {
        Self::new_explicit_with_cms(
            from,
            to,
            &crate::policy::ConvertOptions::retain_channels()
                .with_alpha_policy(crate::AlphaPolicy::DiscardIfOpaque)
                .with_depth_policy(crate::DepthPolicy::Round)
                .with_luma(Some(crate::LumaCoefficients::Bt709))
                .with_alpha_policy(crate::policy::AlphaPolicy::DiscardUnchecked),
            None,
        )
    }

    /// Create a converter with explicit policy options.
    ///
    /// Like [`new`](Self::new) but validates [`ConvertOptions`] policies
    /// (alpha removal, depth reduction, RGB→Gray). Cross-profile
    /// conversions go through the default CMS dispatch chain.
    ///
    /// [`ConvertOptions`]: crate::policy::ConvertOptions
    #[track_caller]
    pub(crate) fn new_explicit(
        from: PixelDescriptor,
        to: PixelDescriptor,
        options: &crate::policy::ConvertOptions,
    ) -> Result<Self, At<ConvertError>> {
        Self::new_explicit_with_cms_hooks(
            from,
            to,
            options,
            None,
            &crate::caps::PlannerHooks::ALL_HOOKS,
        )
    }

    pub(crate) fn new_explicit_hooks(
        from: PixelDescriptor,
        to: PixelDescriptor,
        options: &crate::policy::ConvertOptions,
        hooks: &crate::caps::PlannerHooks,
    ) -> Result<Self, At<ConvertError>> {
        Self::new_explicit_with_cms_hooks(from, to, options, None, hooks)
    }

    /// Create a converter that may delegate the color conversion to a
    /// [`PluggableCms`].
    ///
    /// When `cms` is `Some` and the source and destination have different
    /// primaries or transfer functions, the plugin is asked to supply a
    /// row transform for the full `(from, to)` pair. If it accepts, the
    /// plan becomes a single external-transform step; the built-in gamut
    /// matrix and matlut fast paths are bypassed for that conversion.
    /// If the plugin declines or there is no color work to do, behavior
    /// matches [`new_explicit`](Self::new_explicit).
    ///
    /// [`PluggableCms`]: crate::cms::PluggableCms
    #[track_caller]
    pub(crate) fn new_explicit_with_cms(
        from: PixelDescriptor,
        to: PixelDescriptor,
        options: &crate::policy::ConvertOptions,
        cms: Option<&dyn crate::cms::PluggableCms>,
    ) -> Result<Self, At<ConvertError>> {
        Self::new_explicit_with_cms_hooks(
            from,
            to,
            options,
            cms,
            &crate::caps::PlannerHooks::ALL_HOOKS,
        )
    }

    pub(crate) fn new_explicit_with_cms_hooks(
        from: PixelDescriptor,
        to: PixelDescriptor,
        options: &crate::policy::ConvertOptions,
        cms: Option<&dyn crate::cms::PluggableCms>,
        hooks: &crate::caps::PlannerHooks,
    ) -> Result<Self, At<ConvertError>> {
        use crate::policy::{AlphaPolicy, DepthPolicy};

        // Check before CMS setup: its profile/format interface cannot express
        // the missing narrow-range depth scaling either.
        crate::convert::validate_descriptors(from, to)?;
        crate::convert::validate_luma_policy(from, to, options)?;
        if from.transfer() != to.transfer()
            && matches!(
                (from.transfer(), to.transfer()),
                (TransferFunction::Pq | TransferFunction::Hlg, _)
                    | (_, TransferFunction::Pq | TransferFunction::Hlg)
            )
        {
            // The CMS interface has no luminance-intent parameter. Native HDR
            // plans must retain the explicit anchor and peak policy.
            return Ok(Self::native(NativePlan::new_explicit_core(
                from, to, options, hooks,
            )?));
        }

        if from.has_alpha()
            && !to.has_alpha()
            && options.alpha_policy() == AlphaPolicy::DiscardIfOpaque
        {
            return Err(whereat::at!(ConvertError::AlphaCheckRequired));
        }

        // Gamut mapping runs on the built-in plan. A plugin's transform would clip.
        if options.gamut_policy() == crate::policy::GamutPolicy::SoftCompress
            && crate::convert::soft_compress_applies(&from, &to)
        {
            return Ok(Self::native(NativePlan::new_explicit_core(
                from, to, options, hooks,
            )?));
        }

        // CMS dispatch chain. Fires when:
        //   - primaries differ (cross-gamut RGB↔RGB, where ZenCmsLite or
        //     moxcms supplies a matlut / 3x3+TF transform), OR
        //   - the (from, to) pair needs a CMS by `requires_cms`
        //     (CMYK / Lab / XYZ / future non-native models). Without this
        //     second arm the built-in plan path would error out with
        //     `NeedsCms` before any plugin was consulted, even when one
        //     was attached — the regression the 0.2.16 fix targets.
        //
        // Transfer-only changes inside the native RGB family stay on the
        // built-in plan path so `compose` can peephole them and options
        // like `clip_out_of_gamut` propagate through.
        //
        //   1. user-supplied plugin (if Some)
        //   2. ZenCmsLite (default) — named-profile matlut fast path
        //
        // Per-plugin semantics:
        //   - None → declined, try next plugin
        //   - Some(Ok(t)) → accepted, stop the chain
        //   - Some(Err(e)) → tried-and-failed, propagate the error and stop.
        //     Falling through to another backend could silently produce
        //     different output — surface the failure instead.
        let primaries_differ = from.primaries != to.primaries;
        let needs_cms_dispatch = crate::convert::needs_cms_for_color_model(&from, &to);
        // A plugin can only drop alpha. The built-in plan composites and applies the other policies.
        let plugin_may_drop_alpha = !(from.has_alpha() && !to.has_alpha())
            || options.alpha_policy() == AlphaPolicy::DiscardUnchecked;
        if (primaries_differ || needs_cms_dispatch)
            && plugin_may_drop_alpha
            && from.alpha() != Some(crate::AlphaMode::Premultiplied)
            && to.alpha() != Some(crate::AlphaMode::Premultiplied)
            && from.alpha() != Some(crate::AlphaMode::Undefined)
            && to.alpha() != Some(crate::AlphaMode::Undefined)
            && let Some(external) = dispatch_cms_chain(
                from.color_profile_source(),
                to.color_profile_source(),
                from,
                to,
                options,
                cms,
                hooks,
            )?
        {
            // Precision bits, not byte size — see convert.rs for rationale.
            let reduces_depth = crate::route::channel_bits(from.channel_type())
                > crate::route::channel_bits(to.channel_type());
            if reduces_depth && options.depth_policy() == DepthPolicy::Forbid {
                return Err(whereat::at!(ConvertError::DepthReductionForbidden));
            }

            return Ok(Self::single(from, to, external));
        }

        // Same profiles, or CMS chain declined — built-in plan.
        let plan = NativePlan::new_explicit_core(from, to, options, hooks).at()?;
        Ok(Self::native(plan))
    }

    /// Create a converter from borrowed [`PixelEncoding`](crate::PixelEncoding)
    /// pairs (`descriptor` + optional `ColorContext`), honoring authoritative
    /// ICC profiles, CICP agreement checks, and diffuse-white / HDR-peak anchors
    /// on `from`.
    pub(crate) fn new_with_encoding_hooks(
        from: crate::PixelEncoding<'_>,
        to: crate::PixelEncoding<'_>,
        options: &crate::policy::ConvertOptions,
        cms: Option<&dyn crate::cms::PluggableCms>,
        hooks: &crate::caps::PlannerHooks,
    ) -> Result<Self, At<ConvertError>> {
        let resolved = crate::interpretation::resolve_options(from.color_context(), options);
        let options = &resolved;
        let source_profile = crate::output::current_profile_for(from)?;
        let target_profile = crate::output::current_profile_for(to)?;
        let from_desc = from.descriptor();
        let to_desc = to.descriptor();
        if crate::route::channel_bits(from_desc.channel_type())
            > crate::route::channel_bits(to_desc.channel_type())
            && options.depth_policy() == crate::policy::DepthPolicy::Forbid
        {
            return Err(whereat::at!(ConvertError::DepthReductionForbidden));
        }
        let needs_profiles = !crate::output::profiles_match(&source_profile, &target_profile)
            && (matches!(
                source_profile,
                crate::ColorProfileSource::Icc(_) | crate::ColorProfileSource::Parametric(_)
            ) || matches!(
                target_profile,
                crate::ColorProfileSource::Icc(_) | crate::ColorProfileSource::Parametric(_)
            ));
        if needs_profiles {
            Self::new_with_sources_hooks(
                from_desc,
                to_desc,
                source_profile,
                target_profile,
                options,
                cms,
                hooks,
            )
        } else {
            Self::new_explicit_with_cms_hooks(from_desc, to_desc, options, cms, hooks)
        }
    }

    /// Build from actual current/target profiles, including opaque ICC bytes.
    #[cfg(test)]
    pub(crate) fn new_with_sources(
        from: PixelDescriptor,
        to: PixelDescriptor,
        source: crate::ColorProfileSource<'_>,
        target: crate::ColorProfileSource<'_>,
        options: &crate::policy::ConvertOptions,
        cms: Option<&dyn crate::cms::PluggableCms>,
    ) -> Result<Self, At<ConvertError>> {
        Self::new_with_sources_hooks(
            from,
            to,
            source,
            target,
            options,
            cms,
            &crate::caps::PlannerHooks::ALL_HOOKS,
        )
    }

    pub(crate) fn new_with_sources_hooks(
        from: PixelDescriptor,
        to: PixelDescriptor,
        source: crate::ColorProfileSource<'_>,
        target: crate::ColorProfileSource<'_>,
        options: &crate::policy::ConvertOptions,
        cms: Option<&dyn crate::cms::PluggableCms>,
        hooks: &crate::caps::PlannerHooks,
    ) -> Result<Self, At<ConvertError>> {
        let from = source
            .resolve()
            .map(|(p, t)| from.with_primaries(p).with_transfer(t))
            .unwrap_or(from);
        let to = target
            .resolve()
            .map(|(p, t)| to.with_primaries(p).with_transfer(t))
            .unwrap_or(to);
        crate::convert::validate_descriptors(from, to)?;
        crate::convert::validate_luma_policy(from, to, options)?;

        let profile_tf = |desc_tf: TransferFunction, p: &crate::ColorProfileSource<'_>| {
            if let crate::ColorProfileSource::Parametric(param) = p {
                match param.transfer {
                    zenpixels::ParametricTransfer::Linear => TransferFunction::Linear,
                    zenpixels::ParametricTransfer::Srgb => TransferFunction::Srgb,
                    zenpixels::ParametricTransfer::Bt709 => TransferFunction::Bt709,
                    zenpixels::ParametricTransfer::Pq => TransferFunction::Pq,
                    zenpixels::ParametricTransfer::Hlg => TransferFunction::Hlg,
                    zenpixels::ParametricTransfer::Dci
                    | zenpixels::ParametricTransfer::Gamma(_)
                    | zenpixels::ParametricTransfer::IccParametric(_) => TransferFunction::Gamma22,
                    _ => desc_tf,
                }
            } else {
                desc_tf
            }
        };
        let src_tf = profile_tf(from.transfer(), &source);
        let dst_tf = profile_tf(to.transfer(), &target);
        let hdr_crossing = !crate::output::profiles_match(&source, &target)
            && (matches!(src_tf, TransferFunction::Pq | TransferFunction::Hlg)
                || matches!(dst_tf, TransferFunction::Pq | TransferFunction::Hlg));
        if hdr_crossing && source.resolve().is_some() && target.resolve().is_some() {
            return Self::new_explicit_hooks(from, to, options, hooks);
        }

        // No ICC transform here can honor a mapping, and clipping silently would be wrong.
        if options.gamut_policy() == crate::policy::GamutPolicy::SoftCompress {
            return Err(whereat::at!(ConvertError::NoPath { from, to }));
        }
        // Plugins accept PixelFormat, which cannot express association or range.
        if (from.alpha() == Some(crate::AlphaMode::Undefined) && to.has_alpha())
            || (to.alpha() == Some(crate::AlphaMode::Undefined) && from.has_alpha())
            || from.signal_range != crate::SignalRange::Full
            || to.signal_range != crate::SignalRange::Full
        {
            return Err(whereat::at!(ConvertError::NoPath { from, to }));
        }
        let composites = from.has_alpha()
            && !to.has_alpha()
            && matches!(
                options.alpha_policy(),
                crate::AlphaPolicy::CompositeOnto { .. }
            );
        if from.has_alpha() && !to.has_alpha() && !composites {
            match options.alpha_policy() {
                crate::AlphaPolicy::DiscardUnchecked => {}
                crate::AlphaPolicy::DiscardIfOpaque => {
                    return Err(whereat::at!(ConvertError::AlphaCheckRequired));
                }
                _ => return Err(whereat::at!(ConvertError::AlphaRemovalForbidden)),
            }
        }

        if hdr_crossing {
            let Some(pivot) = hooks.hdr_pivot else {
                return Err(whereat::at!(ConvertError::NoPath { from, to }));
            };
            return pivot(
                from, to, src_tf, dst_tf, source, target, composites, options, cms, hooks,
            );
        }
        // A plugin takes one depth, straight alpha and no layout change that loses channels, so the
        // color change runs in straight F32 and the plan handles the rest on either side of it.
        let to_gray = matches!(to.layout(), ChannelLayout::Gray | ChannelLayout::GrayAlpha)
            && !matches!(
                from.layout(),
                ChannelLayout::Gray | ChannelLayout::GrayAlpha
            );
        if from.alpha() == Some(crate::AlphaMode::Premultiplied)
            || to.alpha() == Some(crate::AlphaMode::Premultiplied)
            || from.channel_type() != to.channel_type()
            || composites
            || to_gray
            || (options.depth_policy() == crate::DepthPolicy::Truncate
                && !to.channel_type().is_float())
        {
            let straight_float = |layout: ChannelLayout, color: PixelDescriptor| {
                let format = match layout {
                    ChannelLayout::Rgb => crate::PixelFormat::RgbF32,
                    ChannelLayout::Rgba | ChannelLayout::Bgra => crate::PixelFormat::RgbaF32,
                    ChannelLayout::Gray => crate::PixelFormat::GrayF32,
                    ChannelLayout::GrayAlpha => crate::PixelFormat::GrayAF32,
                    _ => return Err(whereat::at!(ConvertError::NoPath { from, to })),
                };
                Ok(format
                    .descriptor()
                    .with_primaries(color.primaries)
                    .with_transfer(color.transfer()))
            };
            let straight_from = straight_float(from.layout(), from)?;
            // Keep the channels until the color change is done.
            let middle_layout = if composites || to_gray {
                from.layout()
            } else {
                to.layout()
            };
            let straight_to = straight_float(middle_layout, to)?;
            if straight_from == from && straight_to == to {
                return Err(whereat::at!(ConvertError::NoPath { from, to }));
            }
            let final_options = if options.depth_policy() == crate::DepthPolicy::Forbid {
                if crate::route::channel_bits(from.channel_type())
                    > crate::route::channel_bits(to.channel_type())
                {
                    return Err(whereat::at!(ConvertError::DepthReductionForbidden));
                }
                options.with_depth_policy(crate::DepthPolicy::Round)
            } else {
                *options
            };
            let before = Self::new_explicit_hooks(from, straight_from, options, hooks)?;
            let color = Self::new_with_sources_hooks(
                straight_from,
                straight_to,
                source,
                target,
                options,
                cms,
                hooks,
            )?;
            let after = Self::new_explicit_hooks(straight_to, to, &final_options, hooks)?;
            return Ok(before.then(color).then(after));
        }
        if let Some(external) = dispatch_cms_chain(
            source.clone(),
            target.clone(),
            from,
            to,
            options,
            cms,
            hooks,
        )? {
            return Ok(Self::single(from, to, external));
        }
        // An unresolved ICC must never fall through to descriptor-only work.
        if source.resolve().is_none() || target.resolve().is_none() {
            return Err(whereat::at!(ConvertError::NeedsCms { from, to }));
        }
        Self::new_explicit_hooks(from, to, options, hooks)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_cms_only(
        from: PixelDescriptor,
        to: PixelDescriptor,
        source: crate::ColorProfileSource<'_>,
        target: crate::ColorProfileSource<'_>,
        options: &crate::policy::ConvertOptions,
        cms: Option<&dyn crate::cms::PluggableCms>,
        err_from: PixelDescriptor,
        err_to: PixelDescriptor,
        hooks: &crate::caps::PlannerHooks,
    ) -> Result<Self, At<ConvertError>> {
        if let Some(external) = dispatch_cms_chain(source, target, from, to, options, cms, hooks)? {
            return Ok(Self::single(from, to, external));
        }
        Err(whereat::at!(ConvertError::NeedsCms {
            from: err_from,
            to: err_to,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_sources_hdr_pivot_hooks(
        from: PixelDescriptor,
        to: PixelDescriptor,
        src_tf: TransferFunction,
        dst_tf: TransferFunction,
        source: crate::ColorProfileSource<'_>,
        target: crate::ColorProfileSource<'_>,
        composites: bool,
        options: &crate::policy::ConvertOptions,
        cms: Option<&dyn crate::cms::PluggableCms>,
        hooks: &crate::caps::PlannerHooks,
    ) -> Result<Self, At<ConvertError>> {
        let endpoint_reduction = crate::route::channel_bits(from.channel_type())
            > crate::route::channel_bits(to.channel_type());
        if endpoint_reduction && options.depth_policy() == crate::policy::DepthPolicy::Forbid {
            return Err(whereat::at!(ConvertError::DepthReductionForbidden));
        }

        let involves_hlg = src_tf == TransferFunction::Hlg || dst_tf == TransferFunction::Hlg;
        if involves_hlg && options.hlg_rendering().is_none() {
            return Err(whereat::at!(ConvertError::HlgRenderingRequired));
        }
        let dst_is_sdr = matches!(src_tf, TransferFunction::Pq | TransferFunction::Hlg)
            && !matches!(
                dst_tf,
                TransferFunction::Linear | TransferFunction::Pq | TransferFunction::Hlg
            );
        if dst_is_sdr {
            if src_tf == TransferFunction::Hlg && options.sdr_white().is_none() {
                return Err(whereat::at!(ConvertError::SdrWhiteRequired));
            }
            if !matches!(options.hdr_peak(), zenpixels::hdr::HdrPeak::Nits(_)) {
                return Err(whereat::at!(ConvertError::HdrPeakRequired));
            }
            if options.sdr_peak().is_none() {
                return Err(whereat::at!(ConvertError::SdrPeakRequired));
            }
            #[cfg(not(feature = "bt2446a"))]
            {
                return Err(whereat::at!(ConvertError::HdrFeatureRequired));
            }
        } else if options.sdr_white().is_none() {
            return Err(whereat::at!(ConvertError::SdrWhiteRequired));
        }

        let leg_opts = if !endpoint_reduction
            && options.depth_policy() == crate::policy::DepthPolicy::Forbid
        {
            options.with_depth_policy(crate::policy::DepthPolicy::Round)
        } else {
            *options
        };
        let straight_float = |layout: ChannelLayout, color: PixelDescriptor| {
            let format = match layout {
                ChannelLayout::Rgb => crate::PixelFormat::RgbF32,
                ChannelLayout::Rgba | ChannelLayout::Bgra => crate::PixelFormat::RgbaF32,
                ChannelLayout::Gray => crate::PixelFormat::GrayF32,
                ChannelLayout::GrayAlpha => crate::PixelFormat::GrayAF32,
                _ => return Err(whereat::at!(ConvertError::NoPath { from, to })),
            };
            Ok(format
                .descriptor()
                .with_primaries(color.primaries)
                .with_transfer(color.transfer()))
        };
        let to_gray = matches!(to.layout(), ChannelLayout::Gray | ChannelLayout::GrayAlpha)
            && !matches!(
                from.layout(),
                ChannelLayout::Gray | ChannelLayout::GrayAlpha
            );

        if source.resolve().is_some() {
            // Resolved source -> BT.2020 F32 handoff -> unresolved target via CMS -> `to`.
            let keep_alpha = if composites { false } else { to.has_alpha() };
            let handoff_fmt = if keep_alpha {
                crate::PixelFormat::RgbaF32
            } else {
                crate::PixelFormat::RgbF32
            };
            let (handoff_tf, handoff_cicp) = if dst_tf == TransferFunction::Pq {
                (TransferFunction::Pq, crate::Cicp::new(9, 16, 0, true))
            } else if matches!(dst_tf, TransferFunction::Linear | TransferFunction::Hlg) {
                (TransferFunction::Linear, crate::Cicp::new(9, 8, 0, true))
            } else {
                (TransferFunction::Bt709, crate::Cicp::new(9, 1, 0, true))
            };
            let handoff_desc = handoff_fmt
                .descriptor()
                .with_primaries(crate::ColorPrimaries::Bt2020)
                .with_transfer(handoff_tf);
            let dst_f32_layout = if to_gray {
                if keep_alpha {
                    ChannelLayout::Rgba
                } else {
                    ChannelLayout::Rgb
                }
            } else if keep_alpha {
                ChannelLayout::Rgba
            } else if matches!(to.layout(), ChannelLayout::Gray) {
                ChannelLayout::Gray
            } else {
                ChannelLayout::Rgb
            };
            let straight_to = straight_float(dst_f32_layout, to)?;
            let before = Self::new_explicit_hooks(
                from,
                handoff_desc,
                &leg_opts.with_gamut_policy(crate::GamutPolicy::PreserveExtended),
                hooks,
            )?;
            let cms_opts = if to.channel_type() == crate::ChannelType::F32 {
                leg_opts
            } else {
                leg_opts.with_gamut_policy(crate::GamutPolicy::Clip)
            };
            let color = Self::build_cms_only(
                handoff_desc,
                straight_to,
                crate::ColorProfileSource::Cicp(handoff_cicp),
                target,
                &cms_opts,
                cms,
                from,
                to,
                hooks,
            )?;
            let after = Self::new_explicit_hooks(straight_to, to, &leg_opts, hooks)?;
            return Ok(before.then(color).then(after));
        }

        // Unresolved source -> BT.2020 F32 handoff via CMS -> `target`.
        let straight_from = straight_float(from.layout(), from)?;
        let handoff_fmt = if from.has_alpha() {
            crate::PixelFormat::RgbaF32
        } else {
            crate::PixelFormat::RgbF32
        };
        let (handoff_tf, handoff_cicp) = if src_tf == TransferFunction::Pq {
            (TransferFunction::Pq, crate::Cicp::new(9, 16, 0, true))
        } else if src_tf == TransferFunction::Hlg && dst_is_sdr {
            (TransferFunction::Bt709, crate::Cicp::new(9, 1, 0, true))
        } else {
            (TransferFunction::Linear, crate::Cicp::new(9, 8, 0, true))
        };
        let handoff_desc = handoff_fmt
            .descriptor()
            .with_primaries(crate::ColorPrimaries::Bt2020)
            .with_transfer(handoff_tf);
        let unclipped = leg_opts.with_gamut_policy(crate::GamutPolicy::PreserveExtended);
        let before = Self::new_explicit_hooks(from, straight_from, &unclipped, hooks)?;
        let color = Self::build_cms_only(
            straight_from,
            handoff_desc,
            source,
            crate::ColorProfileSource::Cicp(handoff_cicp),
            &unclipped,
            cms,
            from,
            to,
            hooks,
        )?;
        let after = if target.resolve().is_some() {
            Self::new_explicit_hooks(handoff_desc, to, &leg_opts, hooks)?
        } else {
            Self::new_with_sources_hooks(
                handoff_desc,
                to,
                crate::ColorProfileSource::Cicp(handoff_cicp),
                target,
                &leg_opts,
                cms,
                hooks,
            )?
        };
        Ok(before.then(color).then(after))
    }

    pub(crate) fn into_worker(
        self,
        color: Option<alloc::sync::Arc<crate::ColorContext>>,
    ) -> RowConverter {
        RowConverter {
            recipe: self,
            workspace: WorkerSpace::default(),
            prepared_width: None,
            worker_limit: None,
            color,
            source: None,
        }
    }

    pub(crate) fn workspace(&self, width: u32) -> Result<WorkspaceLayout, At<ConvertError>> {
        let mut layout = WorkspaceLayout::default();
        crate::convert::row_bytes(self.from, width)?;
        crate::convert::row_bytes(self.to, width)?;
        for (i, stage) in self.stages.iter().enumerate() {
            if i + 1 < self.stages.len() {
                let slot = &mut layout.staging_words[i % 2];
                *slot = (*slot).max(crate::convert::row_bytes(stage.to, width)?.div_ceil(8));
            }
            match &stage.operation {
                Operation::Native(plan) => layout.native.merge(plan.workspace(width)?),
                Operation::Shared(_) | Operation::Owned(_) => layout.opaque = true,
            }
        }
        core::alloc::Layout::array::<u64>(layout.staging_words[0] + layout.staging_words[1])
            .map_err(|_| whereat::at!(ConvertError::InvalidWidth(width)))?;
        Ok(layout)
    }

    pub(crate) fn requirements(
        &self,
        width: u32,
    ) -> Result<crate::AllocationRequirements, At<ConvertError>> {
        use crate::ByteBound;
        use crate::convert::SharedLutMask;
        let layout = self.workspace(width)?;
        let native_worker = layout.bytes();
        let mut resident = (self.stages.capacity() * core::mem::size_of::<Stage>()) as u64;
        let mut shared = SharedLutMask::NONE;
        let mut opaque_resident = layout.opaque;
        let mut opaque_setup = layout.opaque;
        for stage in &self.stages {
            if let Operation::Native(native) = &stage.operation {
                let (heap, cold) = native.plan_allocations();
                resident += heap;
                match cold {
                    Some(mask) => shared = shared.union(mask),
                    None => opaque_setup = true,
                }
                opaque_resident |= native.has_mapper();
            }
        }
        let shared_bytes = shared.total_bytes();
        Ok(crate::AllocationRequirements {
            native_worker,
            worker_heap: if layout.opaque {
                ByteBound::Unknown
            } else {
                ByteBound::Exact(native_worker)
            },
            resident_heap: if opaque_resident {
                ByteBound::Unknown
            } else {
                ByteBound::Exact(resident)
            },
            shared_cold_setup: if opaque_setup {
                ByteBound::Unknown
            } else if shared_bytes == 0 {
                ByteBound::Exact(0)
            } else {
                ByteBound::AtMost(shared_bytes)
            },
        })
    }

    pub(crate) fn steps(&self) -> impl Iterator<Item = crate::route::PlannedStepInfo> + '_ {
        use crate::route::{AccuracyContract, KernelTier, PlannedStepInfo, SemanticEffects};
        let relabel_step = |(rf, rt)| PlannedStepInfo {
            operation: "RelabelTransfer",
            from: rf,
            to: rt,
            tier: KernelTier::Scalar,
            accuracy: AccuracyContract::Exact,
            semantic: SemanticEffects::RELABELED_TRANSFER,
        };
        let before = self.relabel_before.map(relabel_step);
        let after = self.relabel_after.map(relabel_step);
        let stage_steps = self.stages.iter().flat_map(|stage| {
            let (native, backend) = match &stage.operation {
                Operation::Native(plan) => (Some(plan.steps()), None),
                Operation::Shared(_) | Operation::Owned(_) => {
                    let mut semantic = SemanticEffects::NONE;
                    if stage.from.has_alpha() && !stage.to.has_alpha() {
                        semantic = semantic.union(SemanticEffects::ALPHA_DISCARDED);
                    }
                    if crate::route::channel_bits(stage.from.channel_type())
                        > crate::route::channel_bits(stage.to.channel_type())
                    {
                        semantic = semantic.union(SemanticEffects::DEPTH_NARROWED);
                    }
                    if !stage.to.channel_type().is_float() {
                        semantic = semantic.union(SemanticEffects::CLAMPED);
                    }
                    (
                        None,
                        Some(PlannedStepInfo {
                            operation: "Cms",
                            from: stage.from,
                            to: stage.to,
                            tier: KernelTier::Backend,
                            accuracy: AccuracyContract::BackendDefined,
                            semantic,
                        }),
                    )
                }
            };
            native.into_iter().flatten().chain(backend)
        });
        before.into_iter().chain(stage_steps).chain(after)
    }

    pub(crate) fn rank(&self) -> crate::route::RouteRank {
        let mut rank = crate::route::RouteRank::default();
        for step in self.steps() {
            rank.add(step);
        }
        rank
    }

    pub(crate) fn clone_for_worker(&self) -> Result<Self, At<ConvertError>> {
        let mut stages = alloc::vec::Vec::new();
        stages
            .try_reserve_exact(self.stages.len())
            .map_err(|_| whereat::at!(ConvertError::AllocationFailed))?;
        for stage in &self.stages {
            let operation = match &stage.operation {
                Operation::Native(plan) => Operation::Native(plan.clone()),
                Operation::Shared(t) => Operation::Shared(t.clone()),
                Operation::Owned(_) => {
                    return Err(whereat::at!(ConvertError::OwnedCmsConsumed));
                }
            };
            stages.push(Stage {
                operation,
                from: stage.from,
                to: stage.to,
            });
        }
        Ok(Self {
            from: self.from,
            to: self.to,
            stages,
            relabel_before: self.relabel_before,
            relabel_after: self.relabel_after,
            input_facts: self.input_facts,
            output_facts: self.output_facts,
        })
    }
}

#[derive(Default)]
pub(crate) struct WorkspaceLayout {
    staging_words: [usize; 2],
    native: crate::convert::NativeWorkspace,
    pub(crate) opaque: bool,
}

impl WorkspaceLayout {
    pub(crate) fn bytes(&self) -> u64 {
        (self.staging_words[0] as u64 + self.staging_words[1] as u64) * 8 + self.native.bytes()
    }
}

struct WorkerSpace {
    native: ConvertScratch,
    staging: alloc::vec::Vec<u64>,
    front_words: usize,
}

impl Default for WorkerSpace {
    fn default() -> Self {
        Self {
            native: ConvertScratch::new(),
            staging: alloc::vec::Vec::new(),
            front_words: 0,
        }
    }
}

impl WorkerSpace {
    fn reserve(&mut self, layout: &WorkspaceLayout, width: u32) -> Result<(), At<ConvertError>> {
        let words = layout.staging_words[0] + layout.staging_words[1];
        if self.staging.len() < words {
            self.staging = bytemuck::allocation::try_zeroed_vec(words)
                .map_err(|()| whereat::at!(ConvertError::AllocationFailed))?;
        }
        self.native.reserve(layout.native, width)?;
        self.front_words = layout.staging_words[0];
        Ok(())
    }
}

impl Stage {
    fn apply(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
        width: u32,
        scratch: &mut ConvertScratch,
        prepared: bool,
    ) -> Result<(), At<ConvertError>> {
        match &mut self.operation {
            Operation::Native(plan) if prepared => {
                convert_row_prepared(plan, src, dst, width, scratch);
                Ok(())
            }
            Operation::Native(plan) => convert_row_buffered(plan, src, dst, width, scratch),
            Operation::Shared(t) => t.try_transform_row(src, dst, width).map_err(cms_error),
            Operation::Owned(t) => t.try_transform_row(src, dst, width).map_err(cms_error),
        }
    }
}

fn cms_error(e: At<crate::cms::CmsPluginError>) -> At<ConvertError> {
    e.map_error(|error| ConvertError::CmsBackend(alloc::sync::Arc::new(error)))
}

impl core::fmt::Debug for RowConverter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RowConverter")
            .field("from", &self.recipe.from)
            .field("to", &self.recipe.to)
            .field("prepared_width", &self.prepared_width)
            .finish_non_exhaustive()
    }
}

impl RowConverter {
    pub(crate) fn set_source_context(
        &mut self,
        source: Option<alloc::sync::Arc<crate::ColorContext>>,
    ) {
        self.source = source;
    }

    pub(crate) fn set_worker_limit(&mut self, limit: Option<u64>) {
        self.worker_limit = limit;
    }

    pub(crate) fn from_plan(mut plan: NativePlan) -> Self {
        plan.bind_if_unbound_all()
            .expect("ZenCmsLite / NativePlan step binding under caps::ALL");
        RowRecipe::native(plan).into_worker(None)
    }

    pub(crate) fn new(
        from: PixelDescriptor,
        to: PixelDescriptor,
    ) -> Result<Self, At<ConvertError>> {
        let mut recipe = RowRecipe::new(from, to)?;
        recipe.bind::<{ crate::caps::ALL }>()?;
        Ok(recipe.into_worker(None))
    }

    pub(crate) fn new_explicit(
        from: PixelDescriptor,
        to: PixelDescriptor,
        options: &crate::ConvertOptions,
    ) -> Result<Self, At<ConvertError>> {
        let mut recipe = RowRecipe::new_explicit(from, to, options)?;
        recipe.bind::<{ crate::caps::ALL }>()?;
        Ok(recipe.into_worker(None))
    }

    #[cfg(test)]
    pub(crate) fn new_explicit_with_cms(
        from: PixelDescriptor,
        to: PixelDescriptor,
        options: &crate::ConvertOptions,
        cms: Option<&dyn crate::PluggableCms>,
    ) -> Result<Self, At<ConvertError>> {
        let mut recipe = RowRecipe::new_explicit_with_cms(from, to, options, cms)?;
        recipe.bind::<{ crate::caps::ALL }>()?;
        Ok(recipe.into_worker(None))
    }

    #[cfg(test)]
    pub(crate) fn new_with_sources(
        from: PixelDescriptor,
        to: PixelDescriptor,
        src: crate::ColorProfileSource<'_>,
        dst: crate::ColorProfileSource<'_>,
        options: &crate::ConvertOptions,
        cms: Option<&dyn crate::PluggableCms>,
    ) -> Result<Self, At<ConvertError>> {
        let mut recipe = RowRecipe::new_with_sources(from, to, src, dst, options, cms)?;
        recipe.bind::<{ crate::caps::ALL }>()?;
        Ok(recipe.into_worker(None))
    }

    #[cfg(test)]
    pub(crate) fn convert_row(&mut self, src: &[u8], dst: &mut [u8], width: u32) {
        self.try_convert_row(src, dst, width)
            .expect("test conversion failed");
    }

    /// Reserve the compiled workspace and initialize supported backend state.
    pub fn prepare(&mut self, max_width: u32) -> Result<(), At<ConvertError>> {
        // A failed preparer can partially change backend capacity.
        self.prepared_width = None;
        if let Some(limit) = self.worker_limit {
            let bytes = self.recipe.requirements(max_width)?.worker_heap();
            if !crate::plan::fits_worker_limit(bytes, limit) {
                return Err(whereat::at!(ConvertError::WorkerBudgetExceeded {
                    bytes,
                    limit
                }));
            }
        }
        let layout = self.recipe.workspace(max_width)?;
        self.workspace.reserve(&layout, max_width)?;
        for stage in &mut self.recipe.stages {
            match &mut stage.operation {
                Operation::Native(plan) => self.workspace.native.prepare(plan, max_width)?,
                Operation::Shared(t) => t.prepare(max_width).map_err(cms_error)?,
                Operation::Owned(t) => t.prepare(max_width).map_err(cms_error)?,
            }
        }
        self.prepared_width = Some(max_width);
        Ok(())
    }

    /// Validate row capacity, extents and sample alignment, then execute.
    /// Checks are constant-time and happen before destination writes. Backend
    /// failures may leave partial output; backend errors are propagated.
    pub fn try_convert_row(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
        width: u32,
    ) -> Result<(), At<ConvertError>> {
        if self.prepared_width.is_some_and(|capacity| width > capacity) {
            return Err(whereat::at!(ConvertError::InvalidWidth(width)));
        }
        let extent = |desc: PixelDescriptor,
                      len: usize,
                      ptr: *const u8|
         -> Result<usize, At<ConvertError>> {
            let expected = (width as usize)
                .checked_mul(desc.bytes_per_pixel())
                .ok_or_else(|| whereat::at!(ConvertError::InvalidWidth(width)))?;
            if len < expected {
                return Err(whereat::at!(ConvertError::BufferSize {
                    expected,
                    actual: len
                }));
            }
            if expected > 0 && !(ptr as usize).is_multiple_of(desc.min_alignment()) {
                return Err(whereat::at!(ConvertError::Buffer(
                    zenpixels::BufferError::AlignmentViolation
                )));
            }
            Ok(expected)
        };
        let slen = extent(self.recipe.from, src.len(), src.as_ptr())?;
        let dlen = extent(self.recipe.to, dst.len(), dst.as_ptr())?;
        self.execute(&src[..slen], &mut dst[..dlen], width)
    }

    fn execute(&mut self, src: &[u8], dst: &mut [u8], width: u32) -> Result<(), At<ConvertError>> {
        if width == 0 {
            return Ok(());
        }
        if self.recipe.stages.is_empty() {
            dst.copy_from_slice(src);
            return Ok(());
        }
        if self.prepared_width.is_none() {
            if let Some(limit) = self.worker_limit {
                let bytes = self.recipe.requirements(width)?.worker_heap();
                if !crate::plan::fits_worker_limit(bytes, limit) {
                    return Err(whereat::at!(ConvertError::WorkerBudgetExceeded {
                        bytes,
                        limit
                    }));
                }
            }
            if self.recipe.stages.len() > 1 {
                let layout = self.recipe.workspace(width)?;
                self.workspace.reserve(&layout, width)?;
            }
        }
        let stage_count = self.recipe.stages.len();
        let (front, back) = self
            .workspace
            .staging
            .split_at_mut(self.workspace.front_words);
        let a: &mut [u8] = bytemuck::cast_slice_mut(front);
        let b: &mut [u8] = bytemuck::cast_slice_mut(back);
        for (i, stage) in self.recipe.stages.iter_mut().enumerate() {
            let last = i + 1 == stage_count;
            let active_in = width as usize * stage.from.bytes_per_pixel();
            let active_out = width as usize * stage.to.bytes_per_pixel();
            let (input, output): (&[u8], &mut [u8]) = if i % 2 == 0 {
                let input = if i == 0 { src } else { &b[..active_in] };
                (
                    input,
                    if last {
                        &mut *dst
                    } else {
                        &mut a[..active_out]
                    },
                )
            } else {
                (
                    &a[..active_in],
                    if last {
                        &mut *dst
                    } else {
                        &mut b[..active_out]
                    },
                )
            };
            stage.apply(
                input,
                output,
                width,
                &mut self.workspace.native,
                self.prepared_width.is_some(),
            )?;
        }
        Ok(())
    }

    /// Clone the recipe for a fresh, independently prepared worker.
    pub fn try_clone(&self) -> Result<Self, At<ConvertError>> {
        let mut clone = self
            .recipe
            .clone_for_worker()?
            .into_worker(self.color.clone());
        clone.source = self.source.clone();
        clone.worker_limit = self.worker_limit;
        Ok(clone)
    }

    /// Convert typed strided [`PixelSlice`](zenpixels::PixelSlice) rows into `dst` and
    /// write the plan's output [`ColorContext`](crate::ColorContext) onto `dst`.
    ///
    /// Clears any stale [`ColorContext`](crate::ColorContext) on `dst` when the compiled
    /// output interpretation carries no profile or luminance context.
    #[track_caller]
    pub fn convert_rows<'a>(
        &mut self,
        src: zenpixels::PixelSlice<'_>,
        mut dst: zenpixels::PixelSliceMut<'a>,
    ) -> Result<zenpixels::PixelSliceMut<'a>, At<ConvertError>> {
        if src.descriptor() != self.recipe.from {
            return Err(whereat::at!(ConvertError::NoPath {
                from: src.descriptor(),
                to: self.recipe.to,
            }));
        }
        if src.encoding() != crate::PixelEncoding::new(self.recipe.from, self.source.as_deref()) {
            return Err(whereat::at!(ConvertError::SourceInterpretationChanged));
        }
        if !src.float_facts().contains(self.recipe.input_facts) {
            return Err(whereat::at!(ConvertError::FloatFactsUnsatisfied {
                required: self.recipe.input_facts,
                actual: src.float_facts(),
            }));
        }
        if dst.descriptor() != self.recipe.to {
            return Err(whereat::at!(ConvertError::NoPath {
                from: self.recipe.from,
                to: dst.descriptor(),
            }));
        }
        if src.width() != dst.width() || src.rows() != dst.rows() {
            return Err(whereat::at!(ConvertError::Buffer(
                zenpixels::BufferError::InvalidDimensions
            )));
        }
        let dst_stride = dst.stride();
        self.convert_raw_rows(
            src.as_strided_bytes(),
            src.stride(),
            dst.as_strided_bytes_mut(),
            dst_stride,
            src.width(),
            src.rows(),
        )?;
        dst.set_color_context(self.color.clone());
        Ok(dst)
    }

    /// Convert multiple rows from a raw strided byte buffer to a raw strided destination.
    ///
    /// Prefer [`convert_rows`](Self::convert_rows) when [`PixelSlice`](zenpixels::PixelSlice)
    /// carriers are available so endpoint encodings and output [`ColorContext`](crate::ColorContext)
    /// are verified and propagated automatically.
    #[track_caller]
    pub fn convert_raw_rows(
        &mut self,
        src: &[u8],
        src_stride: usize,
        dst: &mut [u8],
        dst_stride: usize,
        width: u32,
        rows: u32,
    ) -> Result<(), At<ConvertError>> {
        if width == 0 || rows == 0 {
            return Ok(());
        }
        for (len, stride, desc, ptr) in [
            (src.len(), src_stride, self.from_descriptor(), src.as_ptr()),
            (dst.len(), dst_stride, self.to_descriptor(), dst.as_ptr()),
        ] {
            let row_bytes = (width as usize)
                .checked_mul(desc.bytes_per_pixel())
                .ok_or_else(|| whereat::at!(ConvertError::InvalidWidth(width)))?;
            if stride < row_bytes {
                return Err(whereat::at!(ConvertError::Buffer(
                    zenpixels::BufferError::StrideTooSmall
                )));
            }
            let expected = (rows as usize - 1)
                .checked_mul(stride)
                .and_then(|v| v.checked_add(row_bytes))
                .ok_or_else(|| whereat::at!(ConvertError::InvalidWidth(width)))?;
            if expected > len {
                return Err(whereat::at!(ConvertError::BufferSize {
                    expected,
                    actual: len
                }));
            }
            if !(ptr as usize).is_multiple_of(desc.min_alignment())
                || (rows > 1 && !stride.is_multiple_of(desc.min_alignment()))
            {
                return Err(whereat::at!(ConvertError::Buffer(
                    zenpixels::BufferError::AlignmentViolation
                )));
            }
        }
        if self.prepared_width.is_some_and(|capacity| width > capacity) {
            return Err(whereat::at!(ConvertError::InvalidWidth(width)));
        }
        let src_bytes = width as usize * self.recipe.from.bytes_per_pixel();
        let dst_bytes = width as usize * self.recipe.to.bytes_per_pixel();
        for y in 0..rows as usize {
            let src_start = y * src_stride;
            let dst_start = y * dst_stride;
            self.execute(
                &src[src_start..src_start + src_bytes],
                &mut dst[dst_start..dst_start + dst_bytes],
                width,
            )?;
        }
        Ok(())
    }

    /// True if the conversion is a no-op (formats are identical).
    #[inline]
    #[must_use]
    pub fn is_identity(&self) -> bool {
        !self.recipe.needs_pixel_work()
    }

    /// Source pixel descriptor.
    pub fn from_descriptor(&self) -> PixelDescriptor {
        self.recipe.from
    }

    /// Effective destination pixel descriptor.
    pub fn to_descriptor(&self) -> PixelDescriptor {
        self.recipe.to
    }

    /// Floating-point sample invariants required on input views (`FloatFacts::NONE` when unstated).
    pub const fn input_float_facts(&self) -> zenpixels::sample::FloatFacts {
        self.recipe.input_facts
    }

    /// Floating-point sample invariants proven for output views (`FloatFacts::NONE` on integer outputs).
    pub const fn output_float_facts(&self) -> zenpixels::sample::FloatFacts {
        self.recipe.output_facts
    }

    #[inline]
    pub(crate) fn expected_source_encoding(&self) -> crate::PixelEncoding<'_> {
        crate::PixelEncoding::new(self.recipe.from, self.source.as_deref())
    }

    #[inline]
    pub(crate) const fn prepared_capacity(&self) -> Option<u32> {
        self.prepared_width
    }

    #[inline]
    pub(crate) fn output_color_arc(&self) -> Option<&alloc::sync::Arc<crate::ColorContext>> {
        self.color.as_ref()
    }

    /// Convert any [`PackedSource`](crate::PackedSource) (`PixelSlice`, `&PixelBuffer`, or
    /// `&BandedPixelBuffer`) into any [`PackedSink`](crate::PackedSink) (`PixelSliceMut`,
    /// `&mut PixelBuffer`, or `&mut BandedPixelBuffer`).
    ///
    /// Single-slice and single-buffer pairs compile down directly to the zero-overhead
    /// contiguous row loop; banded and mixed pairs stream across band intersections and
    /// stamp output [`ColorContext`](crate::ColorContext) and [`FloatFacts`](zenpixels::sample::FloatFacts)
    /// onto owned destinations.
    #[track_caller]
    #[inline]
    pub fn convert(
        &mut self,
        src: impl crate::PackedSource,
        dst: impl crate::PackedSink,
    ) -> Result<(), At<ConvertError>> {
        crate::packed::convert_packed(self, src, dst)
    }

    /// Stream rows across [`BandedPixelBuffer`](crate::banded::BandedPixelBuffer)s with
    /// identical or mismatched band heights using an O(bands) band zipper.
    pub fn convert_bands(
        &mut self,
        src: &crate::banded::BandedPixelBuffer,
        dst: &mut crate::banded::BandedPixelBuffer,
    ) -> Result<(), At<ConvertError>> {
        let extent = src.layout();
        if extent.width() != dst.layout().width() || extent.height() != dst.layout().height() {
            return Err(whereat::at!(ConvertError::Buffer(
                crate::BufferError::InvalidDimensions
            )));
        }
        if src.descriptor() != self.recipe.from {
            return Err(whereat::at!(ConvertError::NoPath {
                from: src.descriptor(),
                to: self.recipe.to,
            }));
        }
        if src.encoding() != crate::PixelEncoding::new(self.recipe.from, self.source.as_deref()) {
            return Err(whereat::at!(ConvertError::SourceInterpretationChanged));
        }
        if !src.float_facts().contains(self.recipe.input_facts) {
            return Err(whereat::at!(ConvertError::FloatFactsUnsatisfied {
                required: self.recipe.input_facts,
                actual: src.float_facts(),
            }));
        }
        if dst.descriptor() != self.recipe.to {
            return Err(whereat::at!(ConvertError::NoPath {
                from: self.recipe.from,
                to: dst.descriptor(),
            }));
        }
        let width = extent.width();
        if self.prepared_width.is_some_and(|cap| width > cap) {
            return Err(whereat::at!(ConvertError::InvalidWidth(width)));
        }
        if width > 0 && extent.height() > 0 {
            let mut src_iter = src.bands();
            let mut dst_iter = dst.bands_mut();
            let mut cur_src = src_iter.next().map(|b| (b, 0u32));
            let mut cur_dst = dst_iter.next().map(|b| (b, 0u32));
            let mut y = 0u32;
            while y < extent.height() {
                let (sb, sy) = cur_src.as_ref().expect("source bands cover height");
                let (db, dy) = cur_dst.as_mut().expect("destination bands cover height");
                let rem_s = sb.rows() - *sy;
                let rem_d = db.rows() - *dy;
                let chunk = rem_s.min(rem_d);
                let s_sub = sb.sub_rows(*sy, chunk);
                let mut d_sub = db.sub_rows_mut(*dy, chunk);
                let d_stride = d_sub.stride();
                self.convert_raw_rows(
                    s_sub.as_strided_bytes(),
                    s_sub.stride(),
                    d_sub.as_strided_bytes_mut(),
                    d_stride,
                    width,
                    chunk,
                )?;
                y += chunk;
                if *sy + chunk == sb.rows() {
                    cur_src = src_iter.next().map(|b| (b, 0));
                } else {
                    cur_src.as_mut().unwrap().1 += chunk;
                }
                if *dy + chunk == db.rows() {
                    cur_dst = dst_iter.next().map(|b| (b, 0));
                } else {
                    *dy += chunk;
                }
            }
        }
        dst.set_color_context(self.color.clone());
        let _ = dst.set_float_facts(self.recipe.output_facts);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::convert::NativePlan;
    use crate::policy::{AlphaPolicy, ConvertOptions, DepthPolicy};
    use crate::{AlphaMode, ChannelLayout, ChannelType, ConvertError, TransferFunction};

    /// Helper: build a RowConverter and convert a single pixel.
    fn convert_pixel(
        from: PixelDescriptor,
        to: PixelDescriptor,
        src: &[u8],
    ) -> alloc::vec::Vec<u8> {
        let mut conv =
            if from.transfer() == TransferFunction::Hlg || to.transfer() == TransferFunction::Hlg {
                RowConverter::from_plan(NativePlan::plan_native(from, to).unwrap())
            } else {
                RowConverter::new_explicit(
                    from,
                    to,
                    &ConvertOptions::retain_channels()
                        .with_alpha_policy(crate::AlphaPolicy::DiscardIfOpaque)
                        .with_depth_policy(crate::DepthPolicy::Round)
                        .with_luma(Some(crate::LumaCoefficients::Bt709))
                        .with_alpha_policy(AlphaPolicy::DiscardUnchecked)
                        .with_sdr_white(zenpixels::hdr::DiffuseWhite::new(10000.)),
                )
                .unwrap()
            };
        let dst_bpp = to.bytes_per_pixel();
        let mut dst = vec![0u8; dst_bpp];
        conv.convert_row(src, &mut dst, 1);
        dst
    }

    // -----------------------------------------------------------------------
    // Identity / no-op
    // -----------------------------------------------------------------------

    #[test]
    fn identity_conversion() {
        let desc = PixelDescriptor::RGB8_SRGB;
        let mut conv = RowConverter::new(desc, desc).unwrap();
        assert!(conv.is_identity());
        assert_eq!(conv.from_descriptor(), desc);
        assert_eq!(conv.to_descriptor(), desc);

        let src = [10u8, 20, 30, 40, 50, 60];
        let mut dst = [0u8; 6];
        conv.convert_row(&src, &mut dst, 2);
        assert_eq!(dst, src);
    }

    // -----------------------------------------------------------------------
    // Layout conversions
    // -----------------------------------------------------------------------

    #[test]
    fn rgb8_to_rgba8() {
        let dst = convert_pixel(
            PixelDescriptor::RGB8_SRGB,
            PixelDescriptor::RGBA8_SRGB,
            &[100, 150, 200],
        );
        assert_eq!(dst, [100, 150, 200, 255]);
    }

    #[test]
    fn rgba8_to_rgb8() {
        let dst = convert_pixel(
            PixelDescriptor::RGBA8_SRGB,
            PixelDescriptor::RGB8_SRGB,
            &[100, 150, 200, 128],
        );
        assert_eq!(dst, [100, 150, 200]);
    }

    #[test]
    fn bgra8_to_rgba8() {
        let dst = convert_pixel(
            PixelDescriptor::BGRA8_SRGB,
            PixelDescriptor::RGBA8_SRGB,
            &[200, 150, 100, 255], // BGRA
        );
        assert_eq!(dst, [100, 150, 200, 255]); // RGBA
    }

    #[test]
    fn rgba8_to_bgra8() {
        let dst = convert_pixel(
            PixelDescriptor::RGBA8_SRGB,
            PixelDescriptor::BGRA8_SRGB,
            &[100, 150, 200, 255],
        );
        assert_eq!(dst, [200, 150, 100, 255]);
    }

    #[test]
    fn rgb8_to_bgra8() {
        // RGB → RGBA → BGRA (two-step).
        let dst = convert_pixel(
            PixelDescriptor::RGB8_SRGB,
            PixelDescriptor::BGRA8_SRGB,
            &[100, 150, 200],
        );
        assert_eq!(dst, [200, 150, 100, 255]);
    }

    #[test]
    fn bgra8_to_rgb8() {
        // BGRA → RGBA → RGB (two-step).
        let dst = convert_pixel(
            PixelDescriptor::BGRA8_SRGB,
            PixelDescriptor::RGB8_SRGB,
            &[200, 150, 100, 255],
        );
        assert_eq!(dst, [100, 150, 200]);
    }

    #[test]
    fn gray8_to_rgb8() {
        let dst = convert_pixel(
            PixelDescriptor::GRAY8_SRGB,
            PixelDescriptor::RGB8_SRGB,
            &[128],
        );
        assert_eq!(dst, [128, 128, 128]);
    }

    #[test]
    fn gray8_to_rgba8() {
        let dst = convert_pixel(
            PixelDescriptor::GRAY8_SRGB,
            PixelDescriptor::RGBA8_SRGB,
            &[200],
        );
        assert_eq!(dst, [200, 200, 200, 255]);
    }

    #[test]
    fn gray8_to_bgra8() {
        // Gray → RGBA → BGRA (two-step).
        let dst = convert_pixel(
            PixelDescriptor::GRAY8_SRGB,
            PixelDescriptor::BGRA8_SRGB,
            &[128],
        );
        // Gray broadcasts to (128,128,128) then swizzles — all same so still (128,128,128,255).
        assert_eq!(dst, [128, 128, 128, 255]);
    }

    #[test]
    fn rgb8_to_gray8() {
        // BT.709 luma: (54*R + 183*G + 19*B + 128) >> 8
        let dst = convert_pixel(
            PixelDescriptor::RGB8_SRGB,
            PixelDescriptor::GRAY8_SRGB,
            &[255, 0, 0],
        );
        // (54*255 + 0 + 0 + 128) >> 8 = (13770 + 128) >> 8 = 13898 >> 8 = 54
        assert_eq!(dst, [54]);
    }

    #[test]
    fn rgba8_to_gray8() {
        let dst = convert_pixel(
            PixelDescriptor::RGBA8_SRGB,
            PixelDescriptor::GRAY8_SRGB,
            &[0, 255, 0, 255],
        );
        // (0 + 183*255 + 0 + 128) >> 8 = (46665 + 128) >> 8 = 46793 >> 8 = 182
        assert_eq!(dst, [182]);
    }

    #[test]
    fn bgra8_to_gray8() {
        // BGRA → RGBA → Gray (two-step).
        let dst = convert_pixel(
            PixelDescriptor::BGRA8_SRGB,
            PixelDescriptor::GRAY8_SRGB,
            &[0, 255, 0, 255], // BGRA: B=0, G=255, R=0
        );
        // After swizzle: RGBA = [0, 255, 0, 255], then gray = 182.
        assert_eq!(dst, [182]);
    }

    // -----------------------------------------------------------------------
    // GrayAlpha conversions
    // -----------------------------------------------------------------------

    #[test]
    fn gray8_to_grayalpha8() {
        let from = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::Gray,
            None,
            TransferFunction::Srgb,
        );
        let to = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::GrayAlpha,
            Some(AlphaMode::Straight),
            TransferFunction::Srgb,
        );
        let dst = convert_pixel(from, to, &[100]);
        assert_eq!(dst, [100, 255]);
    }

    #[test]
    fn grayalpha8_to_gray8() {
        let from = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::GrayAlpha,
            Some(AlphaMode::Straight),
            TransferFunction::Srgb,
        );
        let to = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::Gray,
            None,
            TransferFunction::Srgb,
        );
        let dst = convert_pixel(from, to, &[100, 200]);
        assert_eq!(dst, [100]); // Alpha dropped.
    }

    #[test]
    fn grayalpha8_to_rgba8() {
        let from = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::GrayAlpha,
            Some(AlphaMode::Straight),
            TransferFunction::Srgb,
        );
        let dst = convert_pixel(from, PixelDescriptor::RGBA8_SRGB, &[128, 200]);
        assert_eq!(dst, [128, 128, 128, 200]);
    }

    #[test]
    fn grayalpha8_to_rgb8() {
        let from = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::GrayAlpha,
            Some(AlphaMode::Straight),
            TransferFunction::Srgb,
        );
        let dst = convert_pixel(from, PixelDescriptor::RGB8_SRGB, &[128, 200]);
        assert_eq!(dst, [128, 128, 128]);
    }

    #[test]
    fn grayalpha8_to_bgra8() {
        // GrayAlpha → RGBA → BGRA (two-step).
        let from = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::GrayAlpha,
            Some(AlphaMode::Straight),
            TransferFunction::Srgb,
        );
        let dst = convert_pixel(from, PixelDescriptor::BGRA8_SRGB, &[128, 200]);
        // Gray broadcasts to (128,128,128,200) then BGRA swizzle — all same so (128,128,128,200).
        assert_eq!(dst, [128, 128, 128, 200]);
    }

    // -----------------------------------------------------------------------
    // Depth conversions
    // -----------------------------------------------------------------------

    #[test]
    fn u8_to_u16_roundtrip() {
        let u8_desc = PixelDescriptor::RGB8_SRGB;
        let u16_desc = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
        );
        let src = [0u8, 128, 255];
        let wide = convert_pixel(u8_desc, u16_desc, &src);
        let wide16: &[u16] = bytemuck::cast_slice(&wide);
        assert_eq!(wide16[0], 0); // 0 * 257 = 0
        assert_eq!(wide16[1], 128 * 257); // 32896
        assert_eq!(wide16[2], 255 * 257); // 65535

        // Narrow back to u8.
        let narrow = convert_pixel(u16_desc, u8_desc, &wide);
        assert_eq!(narrow, [0, 128, 255]);
    }

    #[test]
    fn naive_u8_to_f32() {
        // Non-sRGB transfer → naive path.
        let u8_desc = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        );
        let f32_desc = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        );
        let dst = convert_pixel(u8_desc, f32_desc, &[0, 128, 255]);
        let f: &[f32] = bytemuck::cast_slice(&dst);
        assert!((f[0] - 0.0).abs() < 1e-6);
        assert!((f[1] - 128.0 / 255.0).abs() < 1e-5);
        assert!((f[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn naive_f32_to_u8() {
        let f32_desc = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        );
        let u8_desc = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        );
        let src_f: [f32; 3] = [0.0, 0.5, 1.0];
        let src: &[u8] = bytemuck::cast_slice(&src_f);
        let dst = convert_pixel(f32_desc, u8_desc, src);
        assert_eq!(dst[0], 0);
        assert_eq!(dst[1], 128); // 0.5 * 255 + 0.5 = 128.0 → 128
        assert_eq!(dst[2], 255);
    }

    #[test]
    fn srgb_u8_to_linear_f32() {
        let dst = convert_pixel(
            PixelDescriptor::RGB8_SRGB,
            PixelDescriptor::new(
                ChannelType::F32,
                ChannelLayout::Rgb,
                None,
                TransferFunction::Linear,
            ),
            &[0, 128, 255],
        );
        let f: &[f32] = bytemuck::cast_slice(&dst);
        assert!((f[0] - 0.0).abs() < 1e-6);
        // sRGB 128/255 ≈ 0.2158 linear
        assert!((f[1] - 0.2158).abs() < 0.01);
        assert!((f[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn linear_f32_to_srgb_u8() {
        let f32_lin = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        );
        let src_f: [f32; 3] = [0.0, 0.5, 1.0];
        let src: &[u8] = bytemuck::cast_slice(&src_f);
        let dst = convert_pixel(f32_lin, PixelDescriptor::RGB8_SRGB, src);
        assert_eq!(dst[0], 0);
        // linear 0.5 ≈ sRGB 188
        assert!((dst[1] as i32 - 188).abs() <= 1);
        assert_eq!(dst[2], 255);
    }

    #[test]
    fn u16_to_f32_and_back() {
        let u16_desc = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
        );
        let f32_desc = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
        );
        let src16: [u16; 3] = [0, 32768, 65535];
        let src: &[u8] = bytemuck::cast_slice(&src16);
        let mid = convert_pixel(u16_desc, f32_desc, src);
        let f: &[f32] = bytemuck::cast_slice(&mid);
        assert!((f[0] - 0.0).abs() < 1e-6);
        assert!((f[1] - 0.5000076).abs() < 1e-4);
        assert!((f[2] - 1.0).abs() < 1e-6);

        // Round-trip back.
        let back = convert_pixel(f32_desc, u16_desc, &mid);
        let back16: &[u16] = bytemuck::cast_slice(&back);
        assert_eq!(back16[0], 0);
        assert!((back16[1] as i32 - 32768).abs() <= 1);
        assert_eq!(back16[2], 65535);
    }

    // -----------------------------------------------------------------------
    // HDR transfer function conversions
    // -----------------------------------------------------------------------

    #[test]
    fn pq_u16_to_linear_f32_roundtrip() {
        let pq_u16 = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Pq,
        );
        let lin_f32 = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        );
        let src16: [u16; 3] = [0, 32768, 65535];
        let src: &[u8] = bytemuck::cast_slice(&src16);
        let mid = convert_pixel(pq_u16, lin_f32, src);
        let f: &[f32] = bytemuck::cast_slice(&mid);
        assert_eq!(f[0], 0.0);
        assert!(f[1] > 0.0 && f[1] < 1.0);

        // Round-trip.
        let back = convert_pixel(lin_f32, pq_u16, &mid);
        let back16: &[u16] = bytemuck::cast_slice(&back);
        assert_eq!(back16[0], 0);
        assert!((back16[1] as i32 - 32768).abs() <= 2);
    }

    #[test]
    fn pq_f32_to_linear_f32_roundtrip() {
        let pq_f32 = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Pq,
        );
        let lin_f32 = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        );
        let src_f: [f32; 3] = [0.0, 0.5, 1.0];
        let src: &[u8] = bytemuck::cast_slice(&src_f);
        let mid = convert_pixel(pq_f32, lin_f32, src);
        let back = convert_pixel(lin_f32, pq_f32, &mid);
        let back_f: &[f32] = bytemuck::cast_slice(&back);
        for i in 0..3 {
            assert!(
                (back_f[i] - src_f[i]).abs() < 1e-4,
                "PQ F32 roundtrip ch{i}: {:.6} vs {:.6}",
                back_f[i],
                src_f[i]
            );
        }
    }

    #[test]
    fn hlg_u16_to_linear_f32_roundtrip() {
        let hlg_u16 = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Hlg,
        );
        let lin_f32 = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        );
        let src16: [u16; 3] = [0, 32768, 65535];
        let src: &[u8] = bytemuck::cast_slice(&src16);
        let mid = convert_pixel(hlg_u16, lin_f32, src);
        let f: &[f32] = bytemuck::cast_slice(&mid);
        assert_eq!(f[0], 0.0);
        assert!(f[1] > 0.0);

        let back = convert_pixel(lin_f32, hlg_u16, &mid);
        let back16: &[u16] = bytemuck::cast_slice(&back);
        assert_eq!(back16[0], 0);
        assert!((back16[1] as i32 - 32768).abs() <= 2);
    }

    #[test]
    fn hlg_f32_to_linear_f32_roundtrip() {
        let hlg_f32 = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Hlg,
        );
        let lin_f32 = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        );
        let src_f: [f32; 3] = [0.0, 0.5, 1.0];
        let src: &[u8] = bytemuck::cast_slice(&src_f);
        let mid = convert_pixel(hlg_f32, lin_f32, src);
        let back = convert_pixel(lin_f32, hlg_f32, &mid);
        let back_f: &[f32] = bytemuck::cast_slice(&back);
        for i in 0..3 {
            assert!(
                (back_f[i] - src_f[i]).abs() < 1e-4,
                "HLG F32 roundtrip ch{i}: {:.6} vs {:.6}",
                back_f[i],
                src_f[i]
            );
        }
    }

    #[test]
    fn hlg_pq_conversion_refuses() {
        // HLG↔PQ conflates scene-normalized HLG with absolute-nits PQ (no OOTF/
        // `Lw`) — a gross brightness error — so the planner refuses with `NoPath`,
        // the signal-range-refusal posture. HLG↔SDR and HLG↔Linear stay allowed
        // (endpoint-correct; only the OOTF mid-tone gamma is missing). Correct
        // HLG↔PQ is #45 S2's OOTF threading.
        let hlg_f32 = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Hlg,
        );
        let pq_f32 = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Pq,
        );
        match RowConverter::new(hlg_f32, pq_f32) {
            Err(e) => assert!(matches!(*e.error(), ConvertError::HlgRenderingRequired)),
            Ok(_) => panic!("HLG→PQ must refuse, not emit wrong pixels"),
        }
        // Both directions and the u16 form refuse.
        assert!(RowConverter::new(pq_f32, hlg_f32).is_err());
        assert!(
            RowConverter::new(
                PixelDescriptor::RGB16_BT2100_HLG,
                PixelDescriptor::RGB16_BT2100_PQ
            )
            .is_err()
        );
        for to in [PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBF32_LINEAR] {
            let error = RowConverter::new(hlg_f32, to).expect_err("HLG requires a display OOTF");
            assert!(matches!(error.error(), ConvertError::HlgRenderingRequired));
        }
    }

    #[test]
    #[cfg(feature = "bt2446a")]
    fn hdr_u16_to_sdr_u8_pq_requires_peak() {
        // Pre-0.2.16: PQ U16 → sRGB U8 silently routed through linear with no
        // tone-mapping (every HDR sample > SDR diffuse-white saturated to
        // 255). Now refuses loudly and points the caller at the HDR
        // constructors.
        let pq_u16 = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Pq,
        );
        let err = RowConverter::new(pq_u16, PixelDescriptor::RGB8_SRGB)
            .expect_err("plain HDR→SDR must refuse")
            .error()
            .clone();
        assert!(matches!(err, ConvertError::HdrPeakRequired));

        // The HDR-aware constructor accepts a source peak and produces a
        // valid sRGB pixel (BT.2446-A tone-map step inserted).
        let plan = NativePlan::new_with_hdr_peak(pq_u16, PixelDescriptor::RGB8_SRGB, 1000.0)
            .expect("HDR-aware plan should build");
        let mut conv = RowConverter::from_plan(plan);
        let src16: [u16; 3] = [32768, 32768, 32768];
        let src: &[u8] = bytemuck::cast_slice(&src16);
        let mut dst = [0u8; 3];
        conv.convert_row(src, &mut dst, 1);
        assert!(dst[0] > 0 && dst[0] < 255);
    }

    #[test]
    #[cfg(feature = "hdr-experimental")]
    fn hdr_u16_to_sdr_u8_hlg_requires_peak() {
        // HLG needs both peak information and an explicit display mapping.
        let hlg_u16 = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Hlg,
        );
        let err = RowConverter::new(hlg_u16, PixelDescriptor::RGB8_SRGB)
            .expect_err("plain HDR→SDR must refuse")
            .error()
            .clone();
        assert!(matches!(err, ConvertError::HlgRenderingRequired));

        assert!(
            NativePlan::new_with_hdr_peak(hlg_u16, PixelDescriptor::RGB8_SRGB, 1000.0).is_err(),
            "a peak alone does not provide HLG's display OOTF"
        );
    }

    #[test]
    fn straight_to_premul_u8() {
        let straight = PixelDescriptor::RGBA8_SRGB;
        let premul = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::Rgba,
            Some(AlphaMode::Premultiplied),
            TransferFunction::Srgb,
        );
        // 50% alpha: RGB channels halved.
        let dst = convert_pixel(straight, premul, &[200, 100, 50, 128]);
        // (200 * 128 + 128) / 255 = 100, (100 * 128 + 128) / 255 = 50, (50 * 128 + 128) / 255 = 25
        assert!((dst[0] as i32 - 100).abs() <= 1);
        assert!((dst[1] as i32 - 50).abs() <= 1);
        assert!((dst[2] as i32 - 25).abs() <= 1);
        assert_eq!(dst[3], 128);
    }

    #[test]
    fn premul_to_straight_u8() {
        let premul = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::Rgba,
            Some(AlphaMode::Premultiplied),
            TransferFunction::Srgb,
        );
        let straight = PixelDescriptor::RGBA8_SRGB;
        // Premul with alpha=128: channels are already halved.
        let dst = convert_pixel(premul, straight, &[100, 50, 25, 128]);
        // Unpremultiply: (100 * 255 + 64) / 128 = 199, etc.
        assert!((dst[0] as i32 - 200).abs() <= 1);
        assert!((dst[1] as i32 - 100).abs() <= 1);
        assert!((dst[2] as i32 - 50).abs() <= 1);
        assert_eq!(dst[3], 128);
    }

    #[test]
    fn premul_to_straight_zero_alpha() {
        let premul = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::Rgba,
            Some(AlphaMode::Premultiplied),
            TransferFunction::Srgb,
        );
        let straight = PixelDescriptor::RGBA8_SRGB;
        let dst = convert_pixel(premul, straight, &[0, 0, 0, 0]);
        assert_eq!(dst, [0, 0, 0, 0]);
    }

    #[test]
    fn straight_to_premul_f32() {
        let straight = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
            TransferFunction::Linear,
        );
        let premul = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgba,
            Some(AlphaMode::Premultiplied),
            TransferFunction::Linear,
        );
        let src_f: [f32; 4] = [1.0, 0.5, 0.25, 0.5];
        let src: &[u8] = bytemuck::cast_slice(&src_f);
        let dst = convert_pixel(straight, premul, src);
        let f: &[f32] = bytemuck::cast_slice(&dst);
        assert!((f[0] - 0.5).abs() < 1e-6);
        assert!((f[1] - 0.25).abs() < 1e-6);
        assert!((f[2] - 0.125).abs() < 1e-6);
        assert!((f[3] - 0.5).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Oklab conversions
    // -----------------------------------------------------------------------

    #[test]
    fn rgb8_to_oklabf32_does_not_panic() {
        let mut conv =
            RowConverter::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::OKLABF32).unwrap();
        assert!(!conv.is_identity());

        let src = [128u8, 64, 200];
        let mut dst = [0u8; 12];
        conv.convert_row(&src, &mut dst, 1);

        let oklab: [f32; 3] = bytemuck::cast(dst);
        assert!(
            oklab[0] >= 0.0 && oklab[0] <= 1.0,
            "L out of range: {}",
            oklab[0]
        );
    }

    #[test]
    fn oklabf32_roundtrip() {
        // Linear RGB F32 → Oklab F32 → Linear RGB F32.
        let lin = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        );
        let oklab = PixelDescriptor::OKLABF32;

        let src_f: [f32; 3] = [0.5, 0.3, 0.8];
        let src: &[u8] = bytemuck::cast_slice(&src_f);
        let mid = convert_pixel(lin, oklab, src);
        let back = convert_pixel(oklab, lin, &mid);
        let back_f: &[f32] = bytemuck::cast_slice(&back);
        for i in 0..3 {
            assert!(
                (back_f[i] - src_f[i]).abs() < 1e-4,
                "Oklab roundtrip ch{i}: {:.6} vs {:.6}",
                back_f[i],
                src_f[i]
            );
        }
    }

    #[test]
    fn oklabaf32_preserves_alpha() {
        let lin_rgba = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
            TransferFunction::Linear,
        );
        let oklaba = PixelDescriptor::OKLABAF32;
        let src_f: [f32; 4] = [0.5, 0.3, 0.8, 0.7];
        let src: &[u8] = bytemuck::cast_slice(&src_f);
        let mid = convert_pixel(lin_rgba, oklaba, src);
        let mid_f: &[f32] = bytemuck::cast_slice(&mid);
        assert!(
            (mid_f[3] - 0.7).abs() < 1e-6,
            "Alpha not preserved in Oklaba"
        );

        let back = convert_pixel(oklaba, lin_rgba, &mid);
        let back_f: &[f32] = bytemuck::cast_slice(&back);
        assert!(
            (back_f[3] - 0.7).abs() < 1e-6,
            "Alpha not preserved on round-trip"
        );
        for i in 0..3 {
            assert!(
                (back_f[i] - src_f[i]).abs() < 1e-4,
                "Oklaba roundtrip ch{i}: {:.6} vs {:.6}",
                back_f[i],
                src_f[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Multi-row conversion
    // -----------------------------------------------------------------------

    #[test]
    fn convert_rows_basic() {
        let mut conv =
            RowConverter::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB).unwrap();

        let src = [10u8, 20, 30, 40, 50, 60];
        let src_stride = 3; // tight, 1 pixel per row
        let mut dst = [0u8; 8]; // 2 rows × 4 bytes
        let dst_stride = 4;

        conv.convert_raw_rows(&src, src_stride, &mut dst, dst_stride, 1, 2)
            .unwrap();
        assert_eq!(&dst[0..4], &[10, 20, 30, 255]);
        assert_eq!(&dst[4..8], &[40, 50, 60, 255]);
    }

    #[test]
    fn convert_rows_buffer_too_small() {
        let mut conv =
            RowConverter::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB).unwrap();

        let src = [10u8, 20, 30];
        let mut dst = [0u8; 4];
        let err = conv
            .convert_raw_rows(&src, 3, &mut dst, 4, 1, 2)
            .unwrap_err();
        assert!(matches!(*err.error(), ConvertError::BufferSize { .. }));
    }

    // -----------------------------------------------------------------------
    // NativePlan::new_explicit policy checks
    // -----------------------------------------------------------------------

    #[test]
    fn new_explicit_alpha_forbid() {
        let from = PixelDescriptor::RGBA8_SRGB;
        let to = PixelDescriptor::RGB8_SRGB;
        let opts = ConvertOptions::retain_channels().with_depth_policy(DepthPolicy::Round);
        let err = NativePlan::new_explicit(from, to, &opts).unwrap_err();
        assert_eq!(*err.error(), ConvertError::AlphaRemovalForbidden);
    }

    /// Signal-range crossings refuse at plan construction in both
    /// directions: there are no Narrow↔Full kernels, and planning the other
    /// descriptor differences would relabel range-coded values without
    /// rescaling (lifted or crushed blacks). Same-range narrow plans —
    /// value-preserving steps — stay legal.
    #[test]
    fn signal_range_crossing_refuses_at_plan_time() {
        use zenpixels::SignalRange;
        let full = PixelDescriptor::RGB8_SRGB;
        let narrow = PixelDescriptor::RGB8_SRGB.with_signal_range(SignalRange::Narrow);

        let err = NativePlan::new(narrow, full).unwrap_err();
        assert!(matches!(*err.error(), ConvertError::NoPath { .. }));
        // The message must name the real blocker, not just print two
        // identical channel-type/layout pairs.
        assert!(
            err.error().to_string().contains("signal range"),
            "NoPath display should name the range crossing: {}",
            err.error()
        );

        let err = NativePlan::new(full, narrow).unwrap_err();
        assert!(matches!(*err.error(), ConvertError::NoPath { .. }));

        // Same-range narrow→narrow (layout-only change) still plans.
        let narrow_rgba = PixelDescriptor::RGBA8_SRGB.with_signal_range(SignalRange::Narrow);
        NativePlan::new(narrow, narrow_rgba).expect("same-range narrow plan must succeed");
    }

    #[test]
    fn new_explicit_depth_forbid() {
        let from = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
        );
        let to = PixelDescriptor::RGB8_SRGB;
        let opts =
            ConvertOptions::retain_channels().with_alpha_policy(AlphaPolicy::DiscardUnchecked);
        let err = NativePlan::new_explicit(from, to, &opts).unwrap_err();
        assert_eq!(*err.error(), ConvertError::DepthReductionForbidden);
    }

    #[test]
    fn new_explicit_rgb_to_gray_requires_luma() {
        let opts = ConvertOptions::retain_channels()
            .with_alpha_policy(AlphaPolicy::DiscardUnchecked)
            .with_depth_policy(DepthPolicy::Round);
        let err = NativePlan::new_explicit(
            PixelDescriptor::RGB8_SRGB,
            PixelDescriptor::GRAY8_SRGB,
            &opts,
        )
        .unwrap_err();
        assert_eq!(*err.error(), ConvertError::RgbToGray);
    }

    #[test]
    fn new_explicit_allows_when_policies_permit() {
        let opts = ConvertOptions::retain_channels()
            .with_alpha_policy(crate::AlphaPolicy::DiscardIfOpaque)
            .with_depth_policy(crate::DepthPolicy::Round)
            .with_luma(Some(crate::LumaCoefficients::Bt709))
            .with_alpha_policy(AlphaPolicy::DiscardUnchecked);
        let plan = NativePlan::new_explicit(
            PixelDescriptor::RGBA8_SRGB,
            PixelDescriptor::GRAY8_SRGB,
            &opts,
        )
        .unwrap();
        assert!(!plan.is_identity());
    }

    #[test]
    fn clip_out_of_gamut_false_preserves_negatives() {
        // P3 pure green → sRGB produces negative red (out of sRGB gamut).
        // With clip_out_of_gamut=false, the extended-range transfer must
        // preserve those negatives instead of clamping to zero.
        let p3 = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
        )
        .with_primaries(zenpixels::ColorPrimaries::DisplayP3);
        let srgb = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
        );

        let opts = ConvertOptions::retain_channels()
            .with_alpha_policy(crate::AlphaPolicy::DiscardIfOpaque)
            .with_depth_policy(crate::DepthPolicy::Round)
            .with_luma(Some(crate::LumaCoefficients::Bt709))
            .with_gamut_policy(crate::GamutPolicy::PreserveExtended);
        let mut conv = crate::RowConverter::new_explicit(p3, srgb, &opts).unwrap();

        let src: [f32; 3] = [0.0, 1.0, 0.0];
        let mut dst = [0.0f32; 3];
        conv.convert_row(
            bytemuck::cast_slice(&src),
            bytemuck::cast_slice_mut(&mut dst),
            1,
        );
        assert!(
            dst[0] < 0.0,
            "extended range should preserve negative red, got {}",
            dst[0]
        );
    }

    #[test]
    fn clip_out_of_gamut_true_clamps_negatives() {
        // Default clip_out_of_gamut=true clamps sRGB transfer to [0, 1].
        let p3 = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
        )
        .with_primaries(zenpixels::ColorPrimaries::DisplayP3);
        let srgb = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
        );

        let opts = ConvertOptions::retain_channels()
            .with_alpha_policy(crate::AlphaPolicy::DiscardIfOpaque)
            .with_depth_policy(crate::DepthPolicy::Round)
            .with_luma(Some(crate::LumaCoefficients::Bt709));
        assert!(opts.gamut_policy() == crate::GamutPolicy::Clip);
        let mut conv = crate::RowConverter::new_explicit(p3, srgb, &opts).unwrap();

        let src: [f32; 3] = [0.0, 1.0, 0.0];
        let mut dst = [0.0f32; 3];
        conv.convert_row(
            bytemuck::cast_slice(&src),
            bytemuck::cast_slice_mut(&mut dst),
            1,
        );
        assert!(
            dst[0] >= 0.0,
            "clamped path should not produce negatives, got {}",
            dst[0]
        );
    }

    // -----------------------------------------------------------------------
    // Pluggable CMS
    // -----------------------------------------------------------------------

    /// Mock CMS that paints every output pixel red. Used to verify that the
    /// plugin gets hooked into the plan and actually drives the row.
    struct PaintRedCms {
        accepted: core::sync::atomic::AtomicUsize,
    }

    struct PaintRedTransform;

    impl crate::cms::RowTransformMut for PaintRedTransform {
        fn try_transform_row(
            &mut self,
            _src: &[u8],
            dst: &mut [u8],
            width: u32,
        ) -> Result<(), whereat::At<crate::CmsPluginError>> {
            for px in dst.as_chunks_mut::<3>().0.iter_mut().take(width as usize) {
                px[0] = 255;
                px[1] = 0;
                px[2] = 0;
            }
            Ok(())
        }
    }

    impl crate::cms::PluggableCms for PaintRedCms {
        fn build(
            &self,
            request: &crate::cms::CmsRequest<'_>,
        ) -> Result<crate::cms::CmsSupport, whereat::At<crate::CmsPluginError>> {
            let build = || -> Option<
            Result<Box<dyn crate::cms::RowTransformMut>, whereat::At<crate::cms::CmsPluginError>>,
        > {
let _src = request.source();
let _dst = request.target();
let _src_format = request.from().pixel_format();
let _dst_format = request.to().pixel_format();
let _options = request.options();

            self.accepted
                .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            Some(Ok(Box::new(PaintRedTransform)))

};
            if let Some(worker) = build() {
                return worker.map(crate::cms::CmsSupport::Owned);
            }
            Ok(crate::cms::CmsSupport::Declined)
        }
    }

    #[test]
    fn composition_preserves_external_cms() {
        let p3 = PixelDescriptor::RGB8_SRGB.with_primaries(zenpixels::ColorPrimaries::DisplayP3);
        let srgb = PixelDescriptor::RGB8_SRGB;
        let cms = PaintRedCms {
            accepted: core::sync::atomic::AtomicUsize::new(0),
        };
        let spec = crate::ConversionSpec::new(ConvertOptions::retain_channels()).with_cms(&cms);
        let first = crate::ConvertPlan::build(p3.into(), srgb.into(), &spec).unwrap();
        let second =
            crate::ConvertPlan::build(first.output_encoding(), srgb.into(), &spec).unwrap();
        let mut worker = first.then(second).unwrap().into_worker();
        let mut output = [0; 3];
        worker.try_convert_row(&[1, 2, 3], &mut output, 1).unwrap();
        assert_eq!(output, [255, 0, 0]);
    }

    #[test]
    fn pluggable_cms_drives_row_when_profiles_differ() {
        // P3 RGB8 → sRGB RGB8: profiles differ, plugin must be asked and
        // its transform must be the one that runs.
        let p3 = PixelDescriptor::RGB8_SRGB.with_primaries(zenpixels::ColorPrimaries::DisplayP3);
        let srgb = PixelDescriptor::RGB8_SRGB;

        let cms = PaintRedCms {
            accepted: core::sync::atomic::AtomicUsize::new(0),
        };
        let opts = ConvertOptions::retain_channels()
            .with_alpha_policy(crate::AlphaPolicy::DiscardIfOpaque)
            .with_depth_policy(crate::DepthPolicy::Round)
            .with_luma(Some(crate::LumaCoefficients::Bt709));
        let mut conv =
            crate::RowConverter::new_explicit_with_cms(p3, srgb, &opts, Some(&cms)).unwrap();

        assert_eq!(cms.accepted.load(core::sync::atomic::Ordering::Relaxed), 1);

        let src = [10u8, 20, 30, 40, 50, 60];
        let mut dst = [0u8; 6];
        conv.convert_row(&src, &mut dst, 2);
        assert_eq!(dst, [255, 0, 0, 255, 0, 0]);
    }

    #[test]
    fn pluggable_cms_skipped_when_profiles_match() {
        // Same profile on both sides — plan is identity, plugin is never
        // consulted.
        let cms = PaintRedCms {
            accepted: core::sync::atomic::AtomicUsize::new(0),
        };
        let opts = ConvertOptions::retain_channels()
            .with_alpha_policy(crate::AlphaPolicy::DiscardIfOpaque)
            .with_depth_policy(crate::DepthPolicy::Round)
            .with_luma(Some(crate::LumaCoefficients::Bt709));
        let conv = crate::RowConverter::new_explicit_with_cms(
            PixelDescriptor::RGB8_SRGB,
            PixelDescriptor::RGB8_SRGB,
            &opts,
            Some(&cms),
        )
        .unwrap();
        assert!(conv.is_identity());
        assert_eq!(cms.accepted.load(core::sync::atomic::Ordering::Relaxed), 0);
    }

    #[test]
    fn pluggable_cms_shared_path_used_when_offered() {
        // Plugin offers a shareable stateless transform — RowConverter
        // must pick that path over the owned one.
        struct SharedPaintBlueCms;
        struct PaintBlueShared;

        impl crate::cms::RowTransform for PaintBlueShared {
            fn try_transform_row(
                &self,
                _src: &[u8],
                dst: &mut [u8],
                width: u32,
            ) -> Result<(), whereat::At<crate::CmsPluginError>> {
                for px in dst.as_chunks_mut::<3>().0.iter_mut().take(width as usize) {
                    px[0] = 0;
                    px[1] = 0;
                    px[2] = 255;
                }
                Ok(())
            }
        }

        impl crate::cms::PluggableCms for SharedPaintBlueCms {
            fn build(
                &self,
                request: &crate::cms::CmsRequest<'_>,
            ) -> Result<crate::cms::CmsSupport, whereat::At<crate::CmsPluginError>> {
                let build = || -> Option<
                Result<
                    alloc::sync::Arc<dyn crate::cms::RowTransform>,
                    whereat::At<crate::cms::CmsPluginError>,
                >,
            > {
let _src = request.source();
let _dst = request.target();
let _src_format = request.from().pixel_format();
let _dst_format = request.to().pixel_format();
let _options = request.options();

                Some(Ok(alloc::sync::Arc::new(PaintBlueShared)))

};
                if let Some(worker) = build() {
                    return worker.map(crate::cms::CmsSupport::Shared);
                }
                let build = || -> Option<
                Result<
                    Box<dyn crate::cms::RowTransformMut>,
                    whereat::At<crate::cms::CmsPluginError>,
                >,
            > {
let _src = request.source();
let _dst = request.target();
let _src_format = request.from().pixel_format();
let _dst_format = request.to().pixel_format();
let _options = request.options();

                panic!("owned path must not be used when shared is offered");

};
                if let Some(worker) = build() {
                    return worker.map(crate::cms::CmsSupport::Owned);
                }
                Ok(crate::cms::CmsSupport::Declined)
            }
        }

        let p3 = PixelDescriptor::RGB8_SRGB.with_primaries(zenpixels::ColorPrimaries::DisplayP3);
        let srgb = PixelDescriptor::RGB8_SRGB;
        let opts = ConvertOptions::retain_channels()
            .with_alpha_policy(crate::AlphaPolicy::DiscardIfOpaque)
            .with_depth_policy(crate::DepthPolicy::Round)
            .with_luma(Some(crate::LumaCoefficients::Bt709));
        let conv =
            crate::RowConverter::new_explicit_with_cms(p3, srgb, &opts, Some(&SharedPaintBlueCms))
                .unwrap();

        // Clone must carry the shared transform (via Arc).
        let mut conv2 = conv.try_clone().unwrap();
        let mut dst = [0u8; 3];
        conv2.convert_row(&[1, 2, 3], &mut dst, 1);
        assert_eq!(
            dst,
            [0, 0, 255],
            "cloned converter should inherit shared transform"
        );
    }

    /// Audit `T4` regression gate: a stateful `RowTransformMut`-only plugin
    /// must NEVER be silently dropped on `try_clone` (falling back to the
    /// built-in plan) — `try_clone` refuses honestly while keeping the
    /// original converter intact. Stateless plugins share via `RowTransform`.
    #[test]
    fn pluggable_cms_owned_path_refuses_clone() {
        const fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<crate::RowConverter>();
        assert_send_sync::<crate::OutputPlan>();

        // P3 → sRGB so the plugin is consulted; PaintRedCms only offers
        // the stateful `RowTransformMut` path — no `build_shared_*`.
        let p3 = PixelDescriptor::RGB8_SRGB.with_primaries(zenpixels::ColorPrimaries::DisplayP3);
        let srgb = PixelDescriptor::RGB8_SRGB;
        let cms = PaintRedCms {
            accepted: core::sync::atomic::AtomicUsize::new(0),
        };
        let opts = ConvertOptions::retain_channels()
            .with_alpha_policy(crate::AlphaPolicy::DiscardIfOpaque)
            .with_depth_policy(crate::DepthPolicy::Round)
            .with_luma(Some(crate::LumaCoefficients::Bt709));
        let mut conv =
            crate::RowConverter::new_explicit_with_cms(p3, srgb, &opts, Some(&cms)).unwrap();

        // Sanity: pre-clone, the plugin's `paint everything red` runs.
        let mut dst = [0u8; 6];
        conv.convert_row(&[10, 20, 30, 40, 50, 60], &mut dst, 2);
        assert_eq!(dst, [255, 0, 0, 255, 0, 0]);

        // Stateful worker refuses `try_clone` instead of silently dropping
        // the plugin or serializing threads behind a hidden mutex.
        assert!(conv.try_clone().is_err());

        // Original converter still runs the plugin cleanly.
        let mut dst2 = [0u8; 6];
        conv.convert_row(&[10, 20, 30, 40, 50, 60], &mut dst2, 2);
        assert_eq!(dst2, [255, 0, 0, 255, 0, 0]);
    }

    #[test]
    fn pluggable_cms_declines_falls_back_to_builtin() {
        // Plugin returns None — must fall back to the built-in gamut path.
        struct DeclineCms;
        impl crate::cms::PluggableCms for DeclineCms {
            fn build(
                &self,
                request: &crate::cms::CmsRequest<'_>,
            ) -> Result<crate::cms::CmsSupport, whereat::At<crate::CmsPluginError>> {
                let build = || -> Option<
                Result<
                    Box<dyn crate::cms::RowTransformMut>,
                    whereat::At<crate::cms::CmsPluginError>,
                >,
            > {
let _src = request.source();
let _dst = request.target();
let _src_format = request.from().pixel_format();
let _dst_format = request.to().pixel_format();
let _options = request.options();

                None

};
                if let Some(worker) = build() {
                    return worker.map(crate::cms::CmsSupport::Owned);
                }
                Ok(crate::cms::CmsSupport::Declined)
            }
        }

        let p3 = PixelDescriptor::RGB8_SRGB.with_primaries(zenpixels::ColorPrimaries::DisplayP3);
        let srgb = PixelDescriptor::RGB8_SRGB;
        let opts = ConvertOptions::retain_channels()
            .with_alpha_policy(crate::AlphaPolicy::DiscardIfOpaque)
            .with_depth_policy(crate::DepthPolicy::Round)
            .with_luma(Some(crate::LumaCoefficients::Bt709));
        let mut conv =
            crate::RowConverter::new_explicit_with_cms(p3, srgb, &opts, Some(&DeclineCms)).unwrap();

        // Built-in path should produce non-red output from grey source.
        let src = [128u8, 128, 128];
        let mut dst = [0u8; 3];
        conv.convert_row(&src, &mut dst, 1);
        // P3 grey → sRGB grey: R ≈ G ≈ B. Not the all-red sentinel the
        // plugin would have written, proving we took the built-in path.
        assert_ne!(dst, [255, 0, 0]);
    }

    // -----------------------------------------------------------------------
    // Plan accessors
    // -----------------------------------------------------------------------

    #[test]
    fn plan_accessors() {
        let from = PixelDescriptor::RGB8_SRGB;
        let to = PixelDescriptor::RGBA8_SRGB;
        let conv = RowConverter::new(from, to).unwrap();
        assert_eq!(conv.from_descriptor(), from);
        assert_eq!(conv.to_descriptor(), to);
        assert!(!conv.is_identity());
    }

    #[test]
    fn from_plan() {
        let plan =
            NativePlan::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB).unwrap();
        let conv = RowConverter::from_plan(plan);
        assert!(!conv.is_identity());
    }

    // -----------------------------------------------------------------------
    // Multi-pixel conversion
    // -----------------------------------------------------------------------

    #[test]
    fn convert_multiple_pixels() {
        let mut conv =
            RowConverter::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB).unwrap();
        let src = [10, 20, 30, 40, 50, 60, 70, 80, 90];
        let mut dst = [0u8; 12];
        conv.convert_row(&src, &mut dst, 3);
        assert_eq!(dst, [10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255]);
    }

    // -----------------------------------------------------------------------
    // Combined layout + depth conversions
    // -----------------------------------------------------------------------

    #[test]
    fn gray8_to_rgbaf32_linear() {
        // Gray U8 sRGB → RGBA F32 Linear: depth expands, then layout expands.
        let from = PixelDescriptor::GRAY8_SRGB;
        let to = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
            TransferFunction::Linear,
        );
        let dst = convert_pixel(from, to, &[128]);
        let f: &[f32] = bytemuck::cast_slice(&dst);
        // sRGB 128 ≈ linear 0.2158, broadcasted, alpha = 1.0.
        assert!((f[0] - 0.2158).abs() < 0.01);
        assert!((f[0] - f[1]).abs() < 1e-6);
        assert!((f[0] - f[2]).abs() < 1e-6);
        assert!((f[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rgba8_to_rgb16() {
        // Layout contracts (drop alpha), then depth expands.
        let from = PixelDescriptor::RGBA8_SRGB;
        let to = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
        );
        let dst = convert_pixel(from, to, &[100, 150, 200, 255]);
        let u16s: &[u16] = bytemuck::cast_slice(&dst);
        assert_eq!(u16s[0], 100 * 257);
        assert_eq!(u16s[1], 150 * 257);
        assert_eq!(u16s[2], 200 * 257);
    }

    // -----------------------------------------------------------------------
    // U16 and F32 layout conversion kernels
    // -----------------------------------------------------------------------

    #[test]
    fn gray_u16_to_rgb_u16() {
        let from = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Gray,
            None,
            TransferFunction::Srgb,
        );
        let to = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
        );
        let src16: [u16; 1] = [40000];
        let src: &[u8] = bytemuck::cast_slice(&src16);
        let dst = convert_pixel(from, to, src);
        let out: &[u16] = bytemuck::cast_slice(&dst);
        assert_eq!(out, [40000, 40000, 40000]);
    }

    #[test]
    fn gray_f32_to_rgba_f32() {
        let from = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Gray,
            None,
            TransferFunction::Linear,
        );
        let to = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
            TransferFunction::Linear,
        );
        let src_f: [f32; 1] = [0.6];
        let src: &[u8] = bytemuck::cast_slice(&src_f);
        let dst = convert_pixel(from, to, src);
        let out: &[f32] = bytemuck::cast_slice(&dst);
        assert!((out[0] - 0.6).abs() < 1e-6);
        assert!((out[1] - 0.6).abs() < 1e-6);
        assert!((out[2] - 0.6).abs() < 1e-6);
        assert!((out[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rgb_f32_to_rgba_f32() {
        let from = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        );
        let to = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
            TransferFunction::Linear,
        );
        let src_f: [f32; 3] = [0.2, 0.4, 0.8];
        let src: &[u8] = bytemuck::cast_slice(&src_f);
        let dst = convert_pixel(from, to, src);
        let out: &[f32] = bytemuck::cast_slice(&dst);
        assert!((out[0] - 0.2).abs() < 1e-6);
        assert!((out[1] - 0.4).abs() < 1e-6);
        assert!((out[2] - 0.8).abs() < 1e-6);
        assert!((out[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rgba_u16_to_rgb_u16() {
        let from = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
            TransferFunction::Srgb,
        );
        let to = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
        );
        let src16: [u16; 4] = [10000, 20000, 30000, 65535];
        let src: &[u8] = bytemuck::cast_slice(&src16);
        let dst = convert_pixel(from, to, src);
        let out: &[u16] = bytemuck::cast_slice(&dst);
        assert_eq!(out, [10000, 20000, 30000]);
    }

    #[test]
    fn gray_alpha_u16_to_rgba_u16() {
        let from = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::GrayAlpha,
            Some(AlphaMode::Straight),
            TransferFunction::Srgb,
        );
        let to = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
            TransferFunction::Srgb,
        );
        let src16: [u16; 2] = [50000, 32768];
        let src: &[u8] = bytemuck::cast_slice(&src16);
        let dst = convert_pixel(from, to, src);
        let out: &[u16] = bytemuck::cast_slice(&dst);
        assert_eq!(out, [50000, 50000, 50000, 32768]);
    }

    #[test]
    fn gray_alpha_f32_to_rgb_f32() {
        let from = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::GrayAlpha,
            Some(AlphaMode::Straight),
            TransferFunction::Linear,
        );
        let to = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        );
        let src_f: [f32; 2] = [0.75, 0.5];
        let src: &[u8] = bytemuck::cast_slice(&src_f);
        let dst = convert_pixel(from, to, src);
        let out: &[f32] = bytemuck::cast_slice(&dst);
        assert!((out[0] - 0.75).abs() < 1e-6);
        assert!((out[1] - 0.75).abs() < 1e-6);
        assert!((out[2] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn gray_u16_to_gray_alpha_u16() {
        let from = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Gray,
            None,
            TransferFunction::Srgb,
        );
        let to = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::GrayAlpha,
            Some(AlphaMode::Straight),
            TransferFunction::Srgb,
        );
        let src16: [u16; 1] = [12345];
        let src: &[u8] = bytemuck::cast_slice(&src16);
        let dst = convert_pixel(from, to, src);
        let out: &[u16] = bytemuck::cast_slice(&dst);
        assert_eq!(out, [12345, 65535]);
    }

    #[test]
    fn gray_alpha_f32_to_gray_f32() {
        let from = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::GrayAlpha,
            Some(AlphaMode::Straight),
            TransferFunction::Linear,
        );
        let to = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Gray,
            None,
            TransferFunction::Linear,
        );
        let src_f: [f32; 2] = [0.33, 0.9];
        let src: &[u8] = bytemuck::cast_slice(&src_f);
        let dst = convert_pixel(from, to, src);
        let out: &[f32] = bytemuck::cast_slice(&dst);
        assert!((out[0] - 0.33).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Transfer function roundtrips (ext.rs branches)
    // -----------------------------------------------------------------------

    #[test]
    fn bt709_linear_f32_roundtrip() {
        let values = [0.0f32, 0.1, 0.25, 0.5, 0.75, 1.0];
        for &v in &values {
            let linear = linear_srgb::default::bt709_to_linear(v);
            let back = linear_srgb::default::linear_to_bt709(linear);
            assert!(
                (back - v).abs() < 1e-5,
                "Bt709 roundtrip failed for {v}: linearize={linear}, delinearize={back}"
            );
        }
    }

    #[test]
    fn unknown_transfer_roundtrip() {
        let unknown = PixelDescriptor::RGB8.with_primaries(zenpixels::ColorPrimaries::Bt709);
        let plan = crate::ConvertPlan::relabel_transfer(unknown, PixelDescriptor::RGB8_SRGB)
            .expect("explicit unknown transfer relabel");
        assert!(plan.is_identity());
    }

    #[test]
    fn oklab_unknown_primaries_returns_error() {
        use crate::ColorPrimaries;
        let from = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        )
        .with_primaries(ColorPrimaries::Unknown);
        let to = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Oklab,
            None,
            TransferFunction::Linear,
        )
        .with_primaries(ColorPrimaries::Unknown);
        let result = RowConverter::new(from, to);
        assert!(result.is_err(), "Oklab with Unknown primaries should fail");
    }

    // -----------------------------------------------------------------------
    // Compose
    // -----------------------------------------------------------------------

    #[test]
    fn compose_roundtrip_cancels_to_identity() {
        let a =
            NativePlan::new(PixelDescriptor::RGBA8_SRGB, PixelDescriptor::RGBAF32_LINEAR).unwrap();
        let b =
            NativePlan::new(PixelDescriptor::RGBAF32_LINEAR, PixelDescriptor::RGBA8_SRGB).unwrap();
        let composed = a.compose(&b).unwrap();
        assert!(composed.is_identity(), "sRGB→linear→sRGB should cancel");
    }

    #[test]
    fn compose_chain_reduces_steps() {
        // RGBA8 sRGB → RGBAF32 linear → RGBAF32 sRGB
        // Steps: SrgbU8ToLinearF32 then LinearF32ToSrgbF32
        // The sRGB→linear step from plan A and linear→sRGB from plan B
        // should NOT cancel (different depth target), but composing avoids
        // an intermediate allocation.
        let a =
            NativePlan::new(PixelDescriptor::RGBA8_SRGB, PixelDescriptor::RGBAF32_LINEAR).unwrap();
        let srgb_f32 = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
            TransferFunction::Srgb,
        );
        let b = NativePlan::new(PixelDescriptor::RGBAF32_LINEAR, srgb_f32).unwrap();
        let composed = a.compose(&b).unwrap();
        assert!(!composed.is_identity());
        assert_eq!(composed.from(), PixelDescriptor::RGBA8_SRGB);
        assert_eq!(composed.to(), srgb_f32);
    }

    #[test]
    fn compose_incompatible_returns_none() {
        let a =
            NativePlan::new(PixelDescriptor::RGBA8_SRGB, PixelDescriptor::RGBAF32_LINEAR).unwrap();
        let b = NativePlan::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB).unwrap();
        assert!(a.compose(&b).is_none(), "RGBAF32_LINEAR != RGB8_SRGB");
    }

    #[test]
    fn compose_premul_roundtrip_cancels() {
        let premul = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgba,
            Some(AlphaMode::Premultiplied),
            TransferFunction::Linear,
        );
        let a = NativePlan::new(PixelDescriptor::RGBAF32_LINEAR, premul).unwrap();
        let b = NativePlan::new(premul, PixelDescriptor::RGBAF32_LINEAR).unwrap();
        let composed = a.compose(&b).unwrap();
        assert!(
            composed.is_identity(),
            "straight→premul→straight should cancel"
        );
    }

    // -----------------------------------------------------------------------
    // CMYK guard
    // -----------------------------------------------------------------------

    #[test]
    fn cmyk_returns_needs_cms_error_when_no_cms_attached() {
        let result = RowConverter::new(PixelDescriptor::CMYK8, PixelDescriptor::RGB8_SRGB);
        let err = match result {
            Ok(_) => panic!("CMYK→RGB without a CMS must error, not succeed"),
            Err(e) => e,
        };
        assert!(
            matches!(*err.error(), ConvertError::NeedsCms { .. }),
            "expected NeedsCms variant, got: {:?}",
            err.error(),
        );
    }
}

#[cfg(test)]
mod alpha_pipeline_contracts {
    use super::*;
    use crate::{
        AlphaMode, ColorProfileSource, PixelDescriptor, PixelFormat,
        cms::{CmsPluginError, PluggableCms, RowTransform},
    };
    use alloc::sync::Arc;
    struct Copy;
    impl RowTransform for Copy {
        fn try_transform_row(
            &self,
            src: &[u8],
            dst: &mut [u8],
            _: u32,
        ) -> Result<(), whereat::At<crate::CmsPluginError>> {
            dst.copy_from_slice(src);
            Ok(())
        }
        fn prepare(&self, _: u32) -> Result<(), At<CmsPluginError>> {
            Ok(())
        }
    }
    struct Cms;
    impl PluggableCms for Cms {
        fn build(
            &self,
            request: &crate::cms::CmsRequest<'_>,
        ) -> Result<crate::cms::CmsSupport, whereat::At<crate::CmsPluginError>> {
            let build = || -> Option<Result<Arc<dyn RowTransform>, At<CmsPluginError>>> {
                let _ = request.source();
                let _ = request.target();
                let from = request.from().pixel_format();
                let to = request.to().pixel_format();
                let _ = request.options();

                assert_eq!(from, PixelFormat::RgbaF32);
                assert_eq!(to, PixelFormat::RgbaF32);
                Some(Ok(Arc::new(Copy)))
            };
            if let Some(worker) = build() {
                return worker.map(crate::cms::CmsSupport::Shared);
            }
            let build =
                || -> Option<Result<Box<dyn crate::cms::RowTransformMut>, At<CmsPluginError>>> {
                    let _ = request.source();
                    let _ = request.target();
                    let _ = request.from().pixel_format();
                    let _ = request.to().pixel_format();
                    let _ = request.options();

                    None
                };
            if let Some(worker) = build() {
                return worker.map(crate::cms::CmsSupport::Owned);
            }
            Ok(crate::cms::CmsSupport::Declined)
        }
    }
    #[test]
    fn prepared_icc_alpha_is_float_rowwise_and_allocates_nothing() {
        let from = PixelDescriptor::RGBA8_SRGB.with_alpha(Some(AlphaMode::Premultiplied));
        let to = PixelDescriptor::RGBA8_SRGB;
        let mut converter = RowConverter::new_with_sources(
            from,
            to,
            ColorProfileSource::Icc(crate::icc_profiles::ADOBE_RGB),
            ColorProfileSource::Icc(crate::icc_profiles::ADOBE_RGB),
            &crate::ConvertOptions::retain_channels()
                .with_alpha_policy(crate::AlphaPolicy::DiscardIfOpaque)
                .with_depth_policy(crate::DepthPolicy::Round)
                .with_luma(Some(crate::LumaCoefficients::Bt709)),
            Some(&Cms),
        )
        .unwrap();
        converter.prepare(2).unwrap();
        let mut output = [0; 8];
        let allocations = allocation_counter::measure(|| {
            converter
                .try_convert_row(&[64, 32, 16, 128, 0, 0, 0, 0], &mut output, 2)
                .unwrap();
            converter.try_convert_row(&[], &mut [], 0).unwrap();
        });
        assert_eq!(allocations.count_total, 0);
        assert_eq!(output, [128, 64, 32, 128, 0, 0, 0, 0]);
        let before = output;
        assert!(converter.try_convert_row(&[0; 12], &mut output, 3).is_err());
        assert_eq!(before, output);
        let mut clone = converter.try_clone().unwrap();
        clone.prepare(2).unwrap();
        clone
            .try_convert_row(&[64, 32, 16, 128], &mut output, 1)
            .unwrap();
        assert_eq!(&output[..4], &[128, 64, 32, 128]);
    }
}
