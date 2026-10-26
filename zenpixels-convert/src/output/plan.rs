//! Prepared color conversion and its matching output signaling.
use super::{OutputMetadata, OutputProfile};
use crate::{
    Cicp, ColorContext, ColorOrigin, ColorPrimaries, ConvertError, ConvertOptions, PixelDescriptor,
    PixelFormat, PixelSlice, PixelSliceMut, RowConverter, TransferFunction,
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
/// use zenpixels_convert::{ColorOrigin, ConvertOptions, OutputPlan, OutputProfile,
///     Cicp, PixelDescriptor, PixelFormat, PixelSlice, PixelSliceMut};
/// let descriptor = PixelDescriptor::RGB8_SRGB;
/// let mut plan = OutputPlan::new(descriptor, None, &ColorOrigin::assumed(),
///     OutputProfile::Named(Cicp::SRGB), PixelFormat::Rgb8,
///     &ConvertOptions::forbid_lossy(), None)?;
/// plan.prepare(2)?;
/// // The adapter can validate plan.metadata() against its encoder now.
/// let input = [10, 20, 30, 40, 50, 60];
/// let mut output = [0; 6];
/// let source = PixelSlice::new(&input, 2, 1, 6, descriptor)?;
/// let destination = PixelSliceMut::new(&mut output, 2, 1, 6, plan.output_descriptor())?;
/// let converted = plan.convert_rows(source, destination)?;
/// assert_eq!(converted.row(0), &input);
/// assert_eq!(converted.color_context().unwrap().cicp, Some(Cicp::SRGB));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct OutputPlan {
    source: PixelDescriptor,
    current: Option<Arc<ColorContext>>,
    target: PixelDescriptor,
    context: Arc<ColorContext>,
    metadata: OutputMetadata,
    converter: Option<RowConverter>,
    prepared_width: Option<u32>,
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
    /// Prepare profiles, matching output signaling and an actual transform.
    /// No image or row buffer is required. CMS construction can allocate its
    /// own state; this does not establish a whole-operation memory bound.
    #[allow(deprecated)]
    pub fn new(
        source_desc: PixelDescriptor,
        current: Option<Arc<ColorContext>>,
        origin: &ColorOrigin,
        target: OutputProfile,
        pixel_format: PixelFormat,
        options: &ConvertOptions,
        cms: Option<&dyn crate::cms::PluggableCms>,
    ) -> Result<Self, At<ConvertError>> {
        let source_profile = super::current_profile_for(source_desc, current.as_deref())?;
        let target_profile = match &target {
            OutputProfile::SameAsOrigin => {
                super::origin_profile(origin).unwrap_or_else(|| source_profile.clone())
            }
            OutputProfile::Named(cicp) => crate::ColorProfileSource::Cicp(*cicp),
            OutputProfile::Icc(icc) => crate::ColorProfileSource::Icc(icc),
        };
        let target_desc_full = super::descriptor_for_profile(pixel_format, &target_profile)?;
        crate::convert::validate_descriptors(source_desc, target_desc_full)?;
        // The opaque-profile CMS path cannot delegate required sample policy
        // to a plugin: a plugin may accept a transform while ignoring options.
        if crate::negotiate::channel_bits(source_desc.channel_type())
            > crate::negotiate::channel_bits(target_desc_full.channel_type())
            && options.depth_policy == crate::DepthPolicy::Forbid
        {
            return Err(whereat::at!(ConvertError::DepthReductionForbidden));
        }
        if matches!(
            source_desc.layout(),
            crate::ChannelLayout::Rgb | crate::ChannelLayout::Rgba | crate::ChannelLayout::Bgra
        ) && matches!(
            target_desc_full.layout(),
            crate::ChannelLayout::Gray | crate::ChannelLayout::GrayAlpha
        ) && options.luma.is_none()
        {
            return Err(whereat::at!(ConvertError::RgbToGray));
        }

        // Determine output metadata based on target profile.
        let metadata = match &target_profile {
            crate::ColorProfileSource::Icc(icc) => OutputMetadata {
                // Emit only the selected authority; OutputMetadata cannot tell a
                // destination codec how to prioritize contradictory ICC/CICP tags.
                icc: Some(match &target {
                    OutputProfile::Icc(bytes) => Arc::clone(bytes),
                    OutputProfile::SameAsOrigin => origin
                        .icc
                        .clone()
                        .or_else(|| current.as_ref().and_then(|c| c.icc.clone()))
                        .unwrap_or_else(|| Arc::from(*icc)),
                    _ => Arc::from(*icc),
                }),
                cicp: None,
                hdr: None,
            },
            crate::ColorProfileSource::Cicp(cicp) => OutputMetadata {
                icc: None,
                cicp: Some(*cicp),
                hdr: None,
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
                        hdr: None,
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
                        hdr: None,
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
        let mut context = crate::ColorContext::default();
        context.icc = metadata.icc.clone();
        context.cicp = metadata.cicp;
        context.diffuse_white = if source_desc.transfer() == TransferFunction::Pq
            && target_desc_full.transfer() == TransferFunction::Linear
        {
            Some(zenpixels::hdr::DiffuseWhite::new(10_000.0))
        } else {
            current.as_ref().and_then(|c| c.diffuse_white)
        };
        let context = Arc::new(context);

        let identity = source_desc.layout_compatible(target_desc_full)
            && super::descriptors_match(&source_desc, &target_desc_full)
            && super::profiles_match(&source_profile, &target_profile);
        // Adding opaque alpha changes storage, not the color coordinates. An
        // unchanged opaque ICC does not require a CMS round trip for this.
        // Keep this shortcut narrower than general same-profile conversion:
        // alpha removal/compositing, depth and color-model changes still use
        // their existing policy and backend paths.
        let alpha_addition = super::profiles_match(&source_profile, &target_profile)
            && source_desc.channel_type() == target_desc_full.channel_type()
            && source_desc.transfer == target_desc_full.transfer
            && source_desc.primaries == target_desc_full.primaries
            && source_desc.signal_range == target_desc_full.signal_range
            && source_desc.alpha().is_none()
            && target_desc_full.alpha() == Some(crate::AlphaMode::Straight)
            && matches!(
                (source_desc.layout(), target_desc_full.layout()),
                (crate::ChannelLayout::Rgb, crate::ChannelLayout::Rgba)
                    | (crate::ChannelLayout::Gray, crate::ChannelLayout::GrayAlpha)
            );
        let converter = if identity {
            None
        } else if alpha_addition {
            Some(crate::RowConverter::new_explicit(
                source_desc,
                target_desc_full,
                options,
            )?)
        } else {
            let needs_profiles = matches!(source_profile, crate::ColorProfileSource::Icc(_))
                || matches!(target_profile, crate::ColorProfileSource::Icc(_));
            let converter = if needs_profiles {
                crate::RowConverter::new_with_sources(
                    source_desc,
                    target_desc_full,
                    source_profile,
                    target_profile,
                    options,
                    cms,
                )?
            } else {
                crate::RowConverter::new_explicit_with_cms(
                    source_desc,
                    target_desc_full,
                    options,
                    cms,
                )?
            };
            Some(converter)
        };
        Ok(Self {
            source: source_desc,
            current,
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
    pub fn borrow_rows<'a>(
        &self,
        source: PixelSlice<'a>,
    ) -> Result<Option<PixelSlice<'a>>, At<ConvertError>> {
        self.validate_source(&source)?;
        Ok(self
            .converter
            .is_none()
            .then(|| source.with_color_context(self.context.clone())))
    }

    fn validate_source(&self, source: &PixelSlice<'_>) -> Result<(), At<ConvertError>> {
        if self
            .prepared_width
            .is_some_and(|width| source.width() > width)
        {
            return Err(whereat::at!(ConvertError::InvalidWidth(source.width())));
        }
        let context_matches = match (&self.current, source.color_context()) {
            (None, None) => true,
            (Some(a), Some(b)) => Arc::ptr_eq(a, b) || a == b,
            _ => false,
        };
        if source.descriptor() != self.source || !context_matches {
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

    pub(super) fn output_context(&self) -> Arc<ColorContext> {
        self.context.clone()
    }
}
