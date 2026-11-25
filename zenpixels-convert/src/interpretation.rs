//! Shared resolution of conversion policy and the meaning of resulting samples.
//! Storage wrappers must use the same resolved options for pixels and metadata.
use alloc::sync::Arc;
use zenpixels::{ColorContext, ConvertOptions, PixelDescriptor};

pub(crate) fn resolve_options(
    current: Option<&ColorContext>,
    options: &ConvertOptions,
) -> ConvertOptions {
    let mut resolved = *options;
    if resolved.sdr_white().is_none()
        && let Some(white) = current.and_then(|c| c.diffuse_white())
    {
        resolved = resolved.with_sdr_white(white);
    }
    if resolved.hdr_peak() == zenpixels::hdr::HdrPeak::Declared
        && let Some(peak) = current.and_then(|c| c.hdr_peak())
    {
        resolved = resolved.with_hdr_peak(zenpixels::hdr::HdrPeak::Nits(peak));
    }
    if resolved.rendering_intent().is_none()
        && let Some(intent) = current.and_then(|c| c.rendering_intent())
    {
        resolved = resolved.with_rendering_intent(intent);
    }
    resolved
}

/// `options` must be the resolved options used to construct the conversion.
/// Profile-aware output finalization replaces only the selected profile fields.
pub(crate) fn output_context_value(
    from: PixelDescriptor,
    to: PixelDescriptor,
    current: Option<&ColorContext>,
    options: &ConvertOptions,
    tone_mapped: bool,
) -> Option<ColorContext> {
    let same_color = from.transfer() == to.transfer()
        && from.primaries == to.primaries
        && from.signal_range == to.signal_range;
    if same_color && !tone_mapped {
        return current.cloned();
    }
    let white = if tone_mapped || crate::convert::is_hdr_to_sdr(from.transfer(), to.transfer()) {
        options
            .sdr_peak()
            .map(|n| zenpixels::hdr::DiffuseWhite::new(n.nits()))
    } else {
        options
            .sdr_white()
            .or_else(|| current.and_then(|c| c.diffuse_white()))
    };
    if current.is_none() && white.is_none() {
        return None;
    }
    let mut next = ColorContext::default();
    if let Some(white) = white {
        next = next.with_diffuse_white(white);
    }
    if let (Some(p), Some(t)) = (to.primaries.to_cicp(), to.transfer().to_cicp()) {
        next = next.with_cicp(zenpixels::Cicp::new(
            p,
            t,
            0,
            to.signal_range == zenpixels::SignalRange::Full,
        ));
    }
    // A new transfer/gamut may change the signal peak. Never carry stale peak evidence.
    Some(next)
}

pub(crate) fn output_context(
    from: PixelDescriptor,
    to: PixelDescriptor,
    current: Option<&Arc<ColorContext>>,
    options: &ConvertOptions,
    tone_mapped: bool,
) -> Option<Arc<ColorContext>> {
    let same_color = from.transfer() == to.transfer()
        && from.primaries == to.primaries
        && from.signal_range == to.signal_range;
    if same_color && !tone_mapped {
        return current.cloned();
    }
    output_context_value(from, to, current.map(Arc::as_ref), options, tone_mapped).map(Arc::new)
}
