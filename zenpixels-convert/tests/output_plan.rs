//! Preparation before pixels, source binding and exact strided execution.
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use zenpixels_convert::{
    AlphaPolicy, Cicp, ColorContext, ColorOrigin, ColorProfileSource, ConvertError, ConvertOptions,
    OutputPlan, OutputProfile, PixelDescriptor, PixelEncoding, PixelFormat, PixelSlice,
    PixelSliceMut, RowConverter, TransferFunction,
    cms::{CmsPluginError, PluggableCms, RowTransform},
};

#[derive(Default)]
struct Spy {
    profiles: Mutex<Vec<(Vec<u8>, Vec<u8>)>>,
    rows: Arc<AtomicUsize>,
}
struct Invert(Arc<AtomicUsize>);
impl RowTransform for Invert {
    fn transform_row(&self, source: &[u8], target: &mut [u8], _: u32) {
        self.0.fetch_add(1, Ordering::SeqCst);
        for (a, b) in source.iter().zip(target) {
            *b = 255 - a;
        }
    }
    fn prepare(&self, _: u32) -> Result<(), whereat::At<CmsPluginError>> {
        Ok(())
    }
}
impl PluggableCms for Spy {
    fn build_source_transform(
        &self,
        _: ColorProfileSource<'_>,
        _: ColorProfileSource<'_>,
        _: PixelFormat,
        _: PixelFormat,
        _: &ConvertOptions,
    ) -> Option<Result<Box<dyn zenpixels_convert::cms::RowTransformMut>, whereat::At<CmsPluginError>>>
    {
        None
    }
    fn build_shared_source_transform(
        &self,
        source: ColorProfileSource<'_>,
        target: ColorProfileSource<'_>,
        _: PixelFormat,
        _: PixelFormat,
        _: &ConvertOptions,
    ) -> Option<Result<Arc<dyn RowTransform>, whereat::At<CmsPluginError>>> {
        let (ColorProfileSource::Icc(a), ColorProfileSource::Icc(b)) = (source, target) else {
            panic!("expected actual opaque profiles")
        };
        self.profiles.lock().unwrap().push((a.to_vec(), b.to_vec()));
        Some(Ok(Arc::new(Invert(self.rows.clone()))))
    }
}

fn opaque() -> PixelDescriptor {
    PixelDescriptor::RGB8_SRGB
        .with_primaries(zenpixels_convert::ColorPrimaries::Unknown)
        .with_transfer(TransferFunction::Unknown)
}

#[test]
fn transforms_are_prepared_before_rows_and_execute_with_matching_signaling() {
    let current = Arc::new(ColorContext::from_icc(Arc::<[u8]>::from(&b"current"[..])));
    let origin = ColorOrigin::from_icc(Arc::<[u8]>::from(&b"old-origin"[..]));
    let target: Arc<[u8]> = Arc::from(&b"target"[..]);
    let cms = Spy::default();
    let mut plan = OutputPlan::new(
        PixelEncoding::new(opaque(), Some(&current)),
        &origin,
        OutputProfile::Icc(target.clone()),
        PixelFormat::Rgb8,
        &ConvertOptions::retain_channels(),
        Some(&cms),
    )
    .unwrap();
    assert_eq!(
        *cms.profiles.lock().unwrap(),
        [(b"current".to_vec(), b"target".to_vec())]
    );
    assert_eq!(cms.rows.load(Ordering::SeqCst), 0);
    assert_eq!(plan.metadata().icc.as_ref(), Some(&target));
    plan.prepare(2).unwrap();
    assert_eq!(cms.rows.load(Ordering::SeqCst), 0);
    let input = [
        1, 2, 3, 4, 5, 6, 0xa5, 0xa5, 0xa5, 7, 8, 9, 10, 11, 12, 0xa5, 0xa5, 0xa5,
    ];
    let mut output = [0x51; 24];
    let target_desc = plan.output_descriptor();
    let allocations = allocation_counter::measure(|| {
        let source = PixelSlice::new(&input, 2, 2, 9, opaque())
            .unwrap()
            .with_color_context(current.clone());
        let destination = PixelSliceMut::new(&mut output, 2, 2, 12, target_desc).unwrap();
        let rows = plan.convert_rows(source, destination).unwrap();
        assert_eq!(rows.color_context().unwrap().icc_arc(), Some(&target));
        assert!(rows.color_context().unwrap().cicp().is_none());
    });
    assert_eq!(allocations.count_total, 0);
    assert_eq!(cms.rows.load(Ordering::SeqCst), 2);
    assert_eq!(
        output,
        [
            254, 253, 252, 251, 250, 249, 0x51, 0x51, 0x51, 0x51, 0x51, 0x51, 248, 247, 246, 245,
            244, 243, 0x51, 0x51, 0x51, 0x51, 0x51, 0x51
        ]
    );
    // Diagnostics must not disclose raw profiles.
    let text = format!("{plan:?}");
    assert!(!text.contains("old-origin"));
    assert!(!text.contains("99, 117, 114"));
}

#[test]
fn changed_interpretation_and_geometry_refuse_before_destination_writes() {
    let descriptor = PixelDescriptor::RGB8_SRGB;
    let current = Arc::new(ColorContext::from_cicp(Cicp::SRGB));
    let mut plan = OutputPlan::new(
        PixelEncoding::new(descriptor, Some(&current)),
        &ColorOrigin::assumed(),
        OutputProfile::Named(Cicp::SRGB),
        PixelFormat::Rgb8,
        &ConvertOptions::retain_channels(),
        None,
    )
    .unwrap();
    plan.prepare(2).unwrap();
    for context in [
        None,
        Some(Arc::new(ColorContext::from_icc(Arc::<[u8]>::from(
            &b"changed"[..],
        )))),
    ] {
        let mut source = PixelSlice::new(&[1; 6], 2, 1, 6, descriptor).unwrap();
        if let Some(context) = context {
            source = source.with_color_context(context);
        }
        let mut output = [0x51; 6];
        let error = plan
            .convert_rows(
                source,
                PixelSliceMut::new(&mut output, 2, 1, 6, descriptor).unwrap(),
            )
            .unwrap_err();
        assert_eq!(error.error(), &ConvertError::SourceInterpretationChanged);
        assert_eq!(output, [0x51; 6]);
    }
    for (width, target_width, target_desc) in [
        (3, 3, descriptor),
        (2, 1, descriptor),
        (
            2,
            2,
            PixelDescriptor::RGB8_SRGB.with_transfer(TransferFunction::Linear),
        ),
    ] {
        let source = PixelSlice::new(&[1; 9], width, 1, 9, descriptor)
            .unwrap()
            .with_color_context(current.clone());
        let mut output = [0x51; 9];
        assert!(
            plan.convert_rows(
                source,
                PixelSliceMut::new(&mut output, target_width, 1, 9, target_desc).unwrap()
            )
            .is_err()
        );
        assert_eq!(output, [0x51; 9]);
    }
    // Equal independently owned context is accepted, not pointer identity only.
    let equal = Arc::new((*current).clone());
    let source = PixelSlice::new(&[1, 2, 3, 4, 5, 6], 2, 1, 6, descriptor)
        .unwrap()
        .with_color_context(equal);
    let mut output = [0; 6];
    let rows = plan
        .convert_rows(
            source,
            PixelSliceMut::new(&mut output, 2, 1, 6, descriptor).unwrap(),
        )
        .unwrap();
    assert_eq!(rows.row(0), &[1, 2, 3, 4, 5, 6]);
    assert_eq!(rows.color_context().unwrap().cicp(), Some(Cicp::SRGB));
}

#[test]
fn required_sample_policies_cannot_be_waived_by_an_opaque_profile_plugin() {
    let current = Arc::new(ColorContext::from_icc(Arc::<[u8]>::from(&b"current"[..])));
    let target: Arc<[u8]> = Arc::from(&b"target"[..]);
    for (source, format, options, expected) in [
        (
            PixelDescriptor::RGBA8_SRGB,
            PixelFormat::Rgb8,
            ConvertOptions::retain_channels(),
            ConvertError::AlphaRemovalForbidden,
        ),
        (
            PixelDescriptor::RGBA8_SRGB,
            PixelFormat::Rgb8,
            ConvertOptions::permissive().with_alpha_policy(AlphaPolicy::DiscardIfOpaque),
            ConvertError::AlphaCheckRequired,
        ),
        (
            PixelDescriptor::RGB16_SRGB,
            PixelFormat::Rgb8,
            ConvertOptions::retain_channels(),
            ConvertError::DepthReductionForbidden,
        ),
        (
            PixelDescriptor::RGB8_SRGB,
            PixelFormat::Gray8,
            ConvertOptions::retain_channels(),
            ConvertError::RgbToGray,
        ),
    ] {
        let cms = Spy::default();
        let error = OutputPlan::new(
            PixelEncoding::new(source, Some(&current)),
            &ColorOrigin::assumed(),
            OutputProfile::Icc(target.clone()),
            format,
            &options,
            Some(&cms),
        )
        .unwrap_err();
        assert_eq!(error.error(), &expected);
        assert!(cms.profiles.lock().unwrap().is_empty());
    }
}

#[cfg(feature = "cms-moxcms")]
#[test]
fn native_icc_rows_match_a_separately_built_cms_worker_and_keep_strides() {
    use zenpixels_convert::MoxCms;
    let icc: Arc<[u8]> = zenpixels_convert::icc_profiles::DISPLAY_P3_V4.into();
    let current = Arc::new(ColorContext::from_icc(icc.clone()));
    let options = ConvertOptions::retain_channels();
    let width = 97;
    let rows = 5;
    let src_stride = width as usize * 3 + 18;
    let dst_stride = width as usize * 3 + 15;
    let input: Vec<u8> = (0..src_stride * rows as usize)
        .map(|i| (i * 73 + 17) as u8)
        .collect();
    let mut expected = vec![0x51; dst_stride * rows as usize];
    // This reference bypasses OutputPlan and finalize_for_output_with.
    let mut oracle = MoxCms
        .build_source_transform(
            ColorProfileSource::Icc(&icc),
            ColorProfileSource::Cicp(Cicp::SRGB),
            PixelFormat::Rgb8,
            PixelFormat::Rgb8,
            &options,
        )
        .unwrap()
        .unwrap();
    oracle.prepare(width).unwrap();
    for y in 0..rows as usize {
        oracle
            .try_transform_row(
                &input[y * src_stride..y * src_stride + width as usize * 3],
                &mut expected[y * dst_stride..y * dst_stride + width as usize * 3],
                width,
            )
            .unwrap();
    }
    let mut plan = OutputPlan::new(
        PixelEncoding::new(PixelDescriptor::RGB8_SRGB, Some(&current)),
        &ColorOrigin::from_cicp(Cicp::SRGB),
        OutputProfile::Named(Cicp::SRGB),
        PixelFormat::Rgb8,
        &options,
        Some(&MoxCms),
    )
    .unwrap();
    plan.prepare(width).unwrap();
    let mut output = vec![0x51; expected.len()];
    let source = PixelSlice::new(&input, width, rows, src_stride, PixelDescriptor::RGB8_SRGB)
        .unwrap()
        .with_color_context(current);
    let destination = PixelSliceMut::new(
        &mut output,
        width,
        rows,
        dst_stride,
        plan.output_descriptor(),
    )
    .unwrap();
    let converted = plan.convert_rows(source, destination).unwrap();
    assert_eq!(converted.color_context().unwrap().cicp(), Some(Cicp::SRGB));
    assert_eq!(plan.metadata().cicp, Some(Cicp::SRGB));
    assert!(plan.metadata().icc.is_none());
    assert_eq!(output, expected);
    assert_ne!(&output[..width as usize * 3], &input[..width as usize * 3]);
}

#[test]
fn output_color_context_is_declared_before_rows_and_carried_by_every_batch() {
    // A converting route declares the target's context, never the source's.
    let current = Arc::new(ColorContext::from_icc(Arc::<[u8]>::from(&b"current"[..])));
    let target: Arc<[u8]> = Arc::from(&b"target"[..]);
    let cms = Spy::default();
    let mut plan = OutputPlan::new(
        PixelEncoding::new(opaque(), Some(&current)),
        &ColorOrigin::assumed(),
        OutputProfile::Icc(target.clone()),
        PixelFormat::Rgb8,
        &ConvertOptions::retain_channels(),
        Some(&cms),
    )
    .unwrap();
    let declared = plan.output_color_context().clone();
    assert_eq!(declared.icc_arc(), Some(&target));
    assert_ne!(*declared, *current);
    let input = [1, 2, 3, 4, 5, 6];
    let mut output = [0; 6];
    let source = PixelSlice::new(&input, 2, 1, 6, opaque())
        .unwrap()
        .with_color_context(current.clone());
    let destination = PixelSliceMut::new(&mut output, 2, 1, 6, plan.output_descriptor()).unwrap();
    let rows = plan.convert_rows(source, destination).unwrap();
    assert!(Arc::ptr_eq(rows.color_context().unwrap(), &declared));

    // PQ decoding honors the source's explicit relative-linear white anchor.
    use zenpixels::hdr::DiffuseWhite;
    let pq = PixelDescriptor::new_full(
        zenpixels_convert::ChannelType::F32,
        zenpixels_convert::ChannelLayout::Rgb,
        None,
        TransferFunction::Pq,
        zenpixels_convert::ColorPrimaries::Bt2020,
    );
    let pq_context = Arc::new(
        ColorContext::from_cicp(Cicp::new(9, 16, 0, true))
            .with_diffuse_white(DiffuseWhite::new(203.0)),
    );
    let linear = OutputPlan::new(
        PixelEncoding::new(pq, Some(&pq_context)),
        &ColorOrigin::assumed(),
        OutputProfile::Named(Cicp::new(9, 8, 0, true)),
        PixelFormat::RgbF32,
        &ConvertOptions::permissive(),
        None,
    )
    .unwrap();
    assert_eq!(
        linear.output_descriptor().transfer(),
        TransferFunction::Linear
    );
    let declared = linear.output_color_context();
    assert_eq!(declared.cicp(), linear.metadata().cicp);
    assert_eq!(declared.icc_arc().cloned(), linear.metadata().icc);
    assert_eq!(declared.diffuse_white(), Some(DiffuseWhite::new(203.0)));
    assert_eq!(declared.diffuse_white(), pq_context.diffuse_white());
}

#[test]
fn row_converter_new_with_encoding_uses_current_profiles_and_anchors() {
    let src_ctx = ColorContext::from_icc(Arc::<[u8]>::from(&b"src-icc"[..]));
    let dst_ctx = ColorContext::from_icc(Arc::<[u8]>::from(&b"dst-icc"[..]));
    let cms = Spy::default();
    let mut conv = RowConverter::new_with_encoding(
        PixelEncoding::new(opaque(), Some(&src_ctx)),
        PixelEncoding::new(opaque(), Some(&dst_ctx)),
        &ConvertOptions::retain_channels(),
        Some(&cms),
    )
    .unwrap();
    assert_eq!(
        *cms.profiles.lock().unwrap(),
        [(b"src-icc".to_vec(), b"dst-icc".to_vec())]
    );
    let mut out = [0u8; 3];
    conv.try_convert_row(&[10, 20, 30], &mut out, 1).unwrap();
    assert_eq!(out, [245, 235, 225]);
}
