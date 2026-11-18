//! Prepared color conversion and its matching output signaling.
use super::{OutputMetadata, OutputProfile};
use crate::{
    Cicp, ColorContext, ColorOrigin, ColorPrimaries, ColorProfile, ConvertError, PixelDescriptor,
    PixelEncoding, PixelFormat, PixelSlice, PixelSliceMut, RowConverter, TransferFunction,
};
use alloc::sync::Arc;
use whereat::At;

/// Validate current color, target signaling and conversion before pixels arrive.
///
/// A transcode adapter can build this while a decoder is suspended, inspect
/// the output contract, prepare row capacity, and configure the destination
/// encoder before approving pixel work. This plans color conversion only:
/// native codec memory, output-carrier support and descriptive metadata remain
/// the adapter's responsibility.
///
/// Current context describes the samples. Origin is provenance used only by
/// `SameAsOrigin`. Every converted batch must still carry the exact source
/// descriptor and context used for preparation. No implicit alpha removal:
/// the supplied `ConvertOptions` controls permitted sample changes.
///
/// ```
/// use zenpixels_convert::{ColorOrigin, OutputPlan, OutputProfile,
///     Cicp, ConversionSpec, OutputTarget, PixelDescriptor, PixelSlice, PixelSliceMut};
/// let descriptor = PixelDescriptor::RGB8_SRGB;
/// let origin = ColorOrigin::assumed();
/// let mut plan = OutputPlan::build(
///     descriptor.encoding(),
///     OutputTarget::new(OutputProfile::Named(Cicp::SRGB), descriptor.pixel_format())
///         .with_origin(&origin),
///     &ConversionSpec::preserve_samples(),
/// )?;
/// plan.prepare(2)?;
/// // The adapter can validate plan.metadata() against its encoder now, and
/// // declare plan.output_color_context() as the rows' interpretation.
/// let input = [10, 20, 30, 40, 50, 60];
/// let mut output = [0; 6];
/// let source = PixelSlice::new(&input, 2, 1, 6, descriptor)?;
/// let destination = PixelSliceMut::new(&mut output, 2, 1, 6, plan.output_descriptor())?;
/// let converted = plan.convert_rows(source, destination)?;
/// assert_eq!(converted.row(0), &input);
/// assert_eq!(converted.color_context().unwrap().cicp(), Some(Cicp::SRGB));
/// assert!(std::sync::Arc::ptr_eq(converted.color_context().unwrap(), plan.output_color_context()));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct OutputPlan {
    source: PixelDescriptor,
    current: Option<ColorContext>,
    target: PixelDescriptor,
    context: Arc<ColorContext>,
    metadata: OutputMetadata,
    converter: Option<RowConverter>,
    prepared_width: Option<u32>,
}
/// Target signaling and packed carrier for an output plan.
#[derive(Clone, Debug)]
pub struct OutputTarget<'a> {
    profile: OutputProfile,
    layout: TargetLayout,
    origin: Option<&'a ColorOrigin>,
}

#[derive(Clone, Copy, Debug)]
enum TargetLayout {
    Format(PixelFormat),
    Required(PixelDescriptor),
}

impl<'a> OutputTarget<'a> {
    /// Choose a carrier, using the selected profile to complete its descriptor.
    pub fn new(profile: OutputProfile, format: PixelFormat) -> Self {
        Self {
            profile,
            layout: TargetLayout::Format(format),
            origin: None,
        }
    }

    /// Require the exact descriptor, including its alpha association.
    pub fn with_descriptor(mut self, descriptor: PixelDescriptor) -> Self {
        self.layout = TargetLayout::Required(descriptor);
        self
    }

    /// Supply the original profile for [OutputProfile::SameAsOrigin].
    pub fn with_origin(mut self, origin: &'a ColorOrigin) -> Self {
        self.origin = Some(origin);
        self
    }
}

impl core::fmt::Debug for OutputPlan {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OutputPlan")
            .field("source", &self.source)
            .field("target", &self.target)
            .field("prepared_width", &self.prepared_width)
            .finish_non_exhaustive()
    }
}
impl OutputPlan {
    /// Plan a conversion and derive matching encoder signaling.
    pub fn build(
        source: PixelEncoding<'_>,
        target: OutputTarget<'_>,
        spec: &crate::ConversionSpec<'_>,
    ) -> Result<Self, At<ConvertError>> {
        let OutputTarget {
            profile: target,
            layout,
            origin,
        } = target;
        let source_desc = source.descriptor();
        let current = source.color_context();
        let options = spec.options();
        let source_profile = super::current_profile_for(source)?;
        let target_profile = match &target {
            OutputProfile::SameAsOrigin => origin
                .and_then(super::origin_profile)
                .unwrap_or_else(|| source_profile.clone()),
            OutputProfile::Named(cicp) => crate::ColorProfileSource::Cicp(*cicp),
            OutputProfile::Icc(icc) => crate::ColorProfileSource::Icc(icc),
            OutputProfile::Parametric(p) => crate::ColorProfileSource::Parametric(*p),
        };
        if matches!(layout, TargetLayout::Format(_))
            && source_desc.alpha() == Some(crate::AlphaMode::Premultiplied)
            && matches!(source_profile, crate::ColorProfileSource::Icc(_))
            && super::profiles_match(&source_profile, &target_profile)
        {
            return Err(whereat::at!(ConvertError::NeedsCms {
                from: source_desc,
                to: source_desc.with_alpha(Some(crate::AlphaMode::Straight)),
            }));
        }
        let pixel_format = match layout {
            TargetLayout::Format(format) => format,
            TargetLayout::Required(descriptor) => descriptor.pixel_format(),
        };
        let inferred = super::descriptor_for_profile(pixel_format, &target_profile)?;
        let target_desc_full = match layout {
            TargetLayout::Format(_) => inferred,
            TargetLayout::Required(requested) => {
                requested
                    .validate()
                    .map_err(|e| whereat::at!(ConvertError::Buffer(e)))?;
                let profile = inferred.with_alpha(requested.alpha());
                if requested != profile {
                    return Err(whereat::at!(ConvertError::OutputInterpretationMismatch {
                        requested,
                        profile,
                    }));
                }
                requested
            }
        };
        crate::convert::validate_descriptors(source_desc, target_desc_full)?;
        // The opaque-profile CMS path cannot delegate required sample policy
        // to a plugin: a plugin may accept a transform while ignoring options.
        if crate::route::channel_bits(source_desc.channel_type())
            > crate::route::channel_bits(target_desc_full.channel_type())
            && options.depth_policy() == crate::DepthPolicy::Forbid
        {
            return Err(whereat::at!(ConvertError::DepthReductionForbidden));
        }
        crate::convert::validate_luma_policy(source_desc, target_desc_full, options)?;

        // Determine output metadata based on target profile.
        let metadata = match &target_profile {
            crate::ColorProfileSource::Icc(icc) => OutputMetadata {
                // Emit only the selected authority; OutputMetadata cannot tell a
                // destination codec how to prioritize contradictory ICC/CICP tags.
                icc: Some(match &target {
                    OutputProfile::Icc(bytes) => Arc::clone(bytes),
                    OutputProfile::SameAsOrigin => origin
                        .and_then(|origin| origin.icc.clone())
                        .or_else(|| current.and_then(|c| c.icc_arc().cloned()))
                        .unwrap_or_else(|| Arc::from(*icc)),
                    _ => Arc::from(*icc),
                }),
                cicp: None,
                parametric: None,
            },
            crate::ColorProfileSource::Cicp(cicp) => OutputMetadata {
                icc: None,
                cicp: Some(*cicp),
                parametric: None,
            },
            crate::ColorProfileSource::Parametric(parametric) => OutputMetadata {
                icc: None,
                cicp: None,
                parametric: Some(*parametric),
            },
            _ => {
                let (p, t) = target_profile.resolve().ok_or_else(|| {
                    whereat::at!(ConvertError::NeedsCms {
                        from: source_desc,
                        to: target_desc_full
                    })
                })?;
                if p == ColorPrimaries::Bt709 && t == TransferFunction::Srgb {
                    OutputMetadata {
                        icc: None,
                        cicp: None,
                        parametric: None,
                    }
                } else if let (Some(p), Some(t)) = (p.to_cicp(), t.to_cicp()) {
                    OutputMetadata {
                        icc: None,
                        cicp: Some(Cicp::new(
                            p,
                            t,
                            0,
                            target_desc_full.signal_range == crate::SignalRange::Full,
                        )),
                        parametric: None,
                    }
                } else {
                    return Err(whereat::at!(ConvertError::NeedsCms {
                        from: source_desc,
                        to: target_desc_full
                    }));
                }
            }
        };

        // The current context must describe the returned samples, including identity.
        let profile = if let Some(icc) = &metadata.icc {
            Some(ColorProfile::Icc(Arc::clone(icc)))
        } else if let Some(parametric) = metadata.parametric {
            Some(ColorProfile::Parametric(parametric))
        } else {
            metadata.cicp.map(ColorProfile::Cicp)
        };
        let target_context = ColorContext::default().with_profile(profile);
        let compiled = crate::ConvertPlan::build(
            source,
            PixelEncoding::new(target_desc_full, Some(&target_context)),
            spec,
        )?;
        let context = Arc::new(
            compiled
                .output_encoding()
                .color_context()
                .cloned()
                .unwrap_or_default(),
        );
        let converter = (!compiled.is_identity()).then(|| compiled.into_worker());
        Ok(Self {
            source: source_desc,
            current: current.cloned(),
            target: target_desc_full,
            context,
            metadata,
            converter,
            prepared_width: None,
        })
    }

    /// The descriptor the prepared transform produces, including color meaning.
    pub fn output_descriptor(&self) -> PixelDescriptor {
        self.target
    }

    /// Color signaling matching the produced samples. This is a planned
    /// contract, not proof that a particular codec has emitted these carriers.
    pub fn metadata(&self) -> &OutputMetadata {
        &self.metadata
    }

    /// Color context of the rows this plan produces, known before any row.
    ///
    /// Declare it to the destination encoder while the decoder is suspended.
    /// [`Self::convert_rows`] and [`Self::borrow_rows`] attach this same `Arc`
    /// to every batch. Do not rebuild it from the source's current context:
    /// after conversion that context describes other samples, and the plan
    /// can set fields the source lacks (PQ to linear output carries a
    /// 10,000 cd/m² diffuse white).
    pub fn output_color_context(&self) -> &Arc<ColorContext> {
        &self.context
    }

    /// Reserve scratch and initialize selected transform state for this width.
    /// Execution then refuses wider rows. Without this call, conversion may
    /// allocate scratch or initialize tables while running, preserving support
    /// for plugins without an explicit preparation hook.
    /// This admits conversion capacity only;
    /// destination and codec allocations remain outside it.
    pub fn prepare(&mut self, max_width: u32) -> Result<(), At<ConvertError>> {
        if max_width == 0 {
            return Err(whereat::at!(ConvertError::InvalidWidth(0)));
        }
        if let Some(converter) = &mut self.converter {
            converter.prepare(max_width)?;
        }
        self.prepared_width = Some(max_width);
        Ok(())
    }

    /// Borrow matching identity rows without allocating or copying sample data.
    ///
    /// `Some` means the prepared operation requires no sample conversion. The
    /// returned view carries the plan's output color context; keep its original
    /// owner alive, including any storage admission charge. `None` means sample
    /// conversion is required: use [`Self::convert_rows`] with an admitted
    /// destination. It does not authorize weakening the requested output.
    ///
    /// Changed source interpretation or a width beyond prepared capacity is an
    /// error even when this plan cannot borrow. This is an execution shortcut
    /// for adapters, not a codec capability promise or metadata privacy check.
    /// The returned descriptor also completes unknown primaries/transfer when
    /// the same authoritative ICC establishes them; sample bits stay unchanged.
    pub fn borrow_rows<'a>(
        &self,
        source: PixelSlice<'a>,
    ) -> Result<Option<PixelSlice<'a>>, At<ConvertError>> {
        self.validate_source(&source)?;
        Ok(self.converter.is_none().then(|| {
            source
                .with_descriptor(self.target)
                .with_color_context(self.context.clone())
        }))
    }

    fn validate_source(&self, source: &PixelSlice<'_>) -> Result<(), At<ConvertError>> {
        if self
            .prepared_width
            .is_some_and(|width| source.width() > width)
        {
            return Err(whereat::at!(ConvertError::InvalidWidth(source.width())));
        }
        let expected = PixelEncoding::new(self.source, self.current.as_ref());
        if source.encoding() != expected {
            return Err(whereat::at!(ConvertError::SourceInterpretationChanged));
        }
        Ok(())
    }

    /// Convert a strided batch and return its view with matching current color.
    /// Use the returned view when handing rows to the encoder: a view cannot
    /// update metadata on a separately owned backing buffer.
    ///
    /// Source interpretation, dimensions, descriptor and prepared width are
    /// checked before any write. A backend row failure may leave partial
    /// destination bytes; the caller must discard that batch. Row padding is
    /// untouched. This method does not publish bytes to an encoder or sink.
    pub fn convert_rows<'a>(
        &mut self,
        source: PixelSlice<'_>,
        mut destination: PixelSliceMut<'a>,
    ) -> Result<PixelSliceMut<'a>, At<ConvertError>> {
        self.validate_source(&source)?;
        if destination.descriptor() != self.target {
            return Err(whereat::at!(ConvertError::NoPath {
                from: self.target,
                to: destination.descriptor()
            }));
        }
        if source.width() != destination.width() || source.rows() != destination.rows() {
            return Err(whereat::at!(ConvertError::Buffer(
                zenpixels::BufferError::InvalidDimensions
            )));
        }
        for y in 0..source.rows() {
            if let Some(converter) = &mut self.converter {
                converter.try_convert_row(source.row(y), destination.row_mut(y), source.width())?;
            } else {
                destination.row_mut(y).copy_from_slice(source.row(y));
            }
        }
        Ok(destination.with_color_context(self.context.clone()))
    }

    pub(super) fn into_metadata(self) -> OutputMetadata {
        self.metadata
    }
}
