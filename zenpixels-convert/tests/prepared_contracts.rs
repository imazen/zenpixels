whereat::define_at_crate_info!();

use zenpixels_convert::{AlphaMode, ConvertPlan, PixelDescriptor, RowConverter, TransferFunction};

#[test]
fn preparation_moves_scratch_and_lut_cost_before_execution() {
    let from = PixelDescriptor::RGBA16_SRGB.with_transfer(TransferFunction::Linear);
    let to = PixelDescriptor::RGBA8_SRGB;
    let mut converter = RowConverter::new(from, to).unwrap();
    converter.prepare(257).unwrap();
    let src = vec![32768u16; 257 * 4];
    let mut dst = vec![0; 257 * 4];
    let allocations = allocation_counter::measure(|| {
        for width in [1, 257, 3, 256, 0, 257] {
            converter
                .try_convert_row(bytemuck::cast_slice(&src), &mut dst, width)
                .unwrap();
        }
    });
    assert_eq!(allocations.count_total, 0);
    let before = dst.clone();
    assert!(
        converter
            .try_convert_row(bytemuck::cast_slice(&src), &mut dst, 258)
            .is_err()
    );
    assert_eq!(dst, before);
}

#[test]
fn preparation_covers_multistep_scratch_and_independent_cms() {
    let from = PixelDescriptor::RGBA8_SRGB;
    let to =
        PixelDescriptor::RGBF32_LINEAR.with_primaries(zenpixels_convert::ColorPrimaries::DisplayP3);
    let mut a = RowConverter::new(from, to).unwrap();
    let mut b = RowConverter::new(from, to).unwrap();
    a.prepare(100).unwrap();
    b.prepare(100).unwrap();
    let src = vec![255u8; 400];
    let mut dst = vec![0f32; 300];
    let allocations = allocation_counter::measure(|| {
        for _ in 0..10 {
            a.try_convert_row(&src, bytemuck::cast_slice_mut(&mut dst), 100)
                .unwrap();
            b.try_convert_row(&src, bytemuck::cast_slice_mut(&mut dst), 100)
                .unwrap();
        }
    });
    assert_eq!(allocations.count_total, 0);
}

#[test]
fn row_extent_failure_precedes_all_writes_and_identity_respects_width() {
    let mut converter =
        RowConverter::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGB8_SRGB).unwrap();
    let mut dst = [99; 12];
    assert!(converter.try_convert_row(&[1; 2], &mut dst, 1).is_err());
    assert_eq!(dst, [99; 12]);
    converter.try_convert_row(&[1; 12], &mut dst, 1).unwrap();
    assert_eq!(dst, [1, 1, 1, 99, 99, 99, 99, 99, 99, 99, 99, 99]);
}

#[test]
fn explicit_composition_preserves_integer_stage() {
    let f = PixelDescriptor::RGBF32_LINEAR;
    let u = PixelDescriptor::RGB8_SRGB.with_transfer(TransferFunction::Linear);
    let a = RowConverter::new(f, u).unwrap();
    let b = RowConverter::new(u, f).unwrap();
    let mut optimized = a.compose(&b).unwrap();
    let mut preserved = a.compose_preserving(&b).unwrap();
    let src = [0.1234567f32; 3];
    let mut exact = [0f32; 3];
    let mut rounded = [0f32; 3];
    optimized
        .try_convert_row(
            bytemuck::cast_slice(&src),
            bytemuck::cast_slice_mut(&mut exact),
            1,
        )
        .unwrap();
    preserved
        .try_convert_row(
            bytemuck::cast_slice(&src),
            bytemuck::cast_slice_mut(&mut rounded),
            1,
        )
        .unwrap();
    assert_eq!(exact, src);
    assert_eq!(rounded, [31. / 255.; 3]);
}

#[test]
fn explicit_opacity_scan_rejects_nonfinite_and_out_of_range_alpha() {
    for alpha in [0.5, f32::NAN, f32::INFINITY, 1.01] {
        let values = [0., 0., 0., alpha];
        let view = zenpixels::PixelSlice::new(
            bytemuck::cast_slice(&values),
            1,
            1,
            16,
            PixelDescriptor::RGBAF32_LINEAR,
        )
        .unwrap();
        assert!(zenpixels_convert::adapt::check_opaque(view).is_err());
    }
}

#[test]
fn preservation_is_proven_or_refused_without_inspecting_samples() {
    assert!(
        ConvertPlan::new_preserving_samples(
            PixelDescriptor::RGB8_SRGB,
            PixelDescriptor::RGB16_SRGB
        )
        .is_ok()
    );
    assert!(
        ConvertPlan::new_preserving_samples(
            PixelDescriptor::RGB16_SRGB,
            PixelDescriptor::RGB8_SRGB
        )
        .is_err()
    );
    assert!(
        ConvertPlan::new_preserving_samples(
            PixelDescriptor::RGBF32_LINEAR,
            PixelDescriptor::RGB8_SRGB
        )
        .is_err()
    );
    assert!(
        ConvertPlan::new_preserving_samples(
            PixelDescriptor::RGBA8_SRGB,
            PixelDescriptor::RGBA8_SRGB.with_alpha(Some(AlphaMode::Opaque))
        )
        .is_err()
    );
}

#[derive(Debug)]
struct Sentinel;
impl std::fmt::Display for Sentinel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("sentinel row failure")
    }
}
impl std::error::Error for Sentinel {}
struct FailingWorker;
impl zenpixels_convert::cms::RowTransformMut for FailingWorker {
    fn transform_row(&mut self, _: &[u8], _: &mut [u8], _: u32) {
        panic!("fallible hook must be used")
    }
    fn try_transform_row(
        &mut self,
        _: &[u8],
        dst: &mut [u8],
        _: u32,
    ) -> Result<(), whereat::At<zenpixels_convert::cms::CmsPluginError>> {
        dst[0] = 42;
        Err(whereat::at!(zenpixels_convert::cms::CmsPluginError::new(
            Sentinel
        )))
    }
    fn prepare(
        &mut self,
        _: u32,
    ) -> Result<(), whereat::At<zenpixels_convert::cms::CmsPluginError>> {
        Ok(())
    }
}
struct FailingCms;
impl zenpixels_convert::cms::PluggableCms for FailingCms {
    fn build_source_transform(
        &self,
        _: zenpixels::ColorProfileSource<'_>,
        _: zenpixels::ColorProfileSource<'_>,
        _: zenpixels::PixelFormat,
        _: zenpixels::PixelFormat,
        _: &zenpixels::ConvertOptions,
    ) -> Option<
        Result<
            Box<dyn zenpixels_convert::cms::RowTransformMut>,
            whereat::At<zenpixels_convert::cms::CmsPluginError>,
        >,
    > {
        Some(Ok(Box::new(FailingWorker)))
    }
}
#[test]
fn backend_row_failure_preserves_concrete_error_and_never_returns_encode_ready() {
    use std::error::Error;
    let from = PixelDescriptor::RGB8_SRGB;
    let to = from.with_primaries(zenpixels::ColorPrimaries::DisplayP3);
    let mut converter = RowConverter::new_explicit_with_cms(
        from,
        to,
        &zenpixels::ConvertOptions::permissive(),
        Some(&FailingCms),
    )
    .unwrap();
    converter.prepare(1).unwrap();
    assert!(converter.try_clone().is_err());
    let mut out = [0; 3];
    let error = converter
        .try_convert_row(&[1, 2, 3], &mut out, 1)
        .unwrap_err();
    assert_eq!(out[0], 42); // explicitly permitted partial output after backend failure
    assert!(
        error
            .error()
            .source()
            .unwrap()
            .source()
            .unwrap()
            .downcast_ref::<Sentinel>()
            .is_some()
    );
    let buffer = zenpixels::PixelBuffer::from_vec(vec![1, 2, 3], 1, 1, from).unwrap();
    assert!(
        zenpixels_convert::finalize_for_output_with(
            &buffer,
            &zenpixels::ColorOrigin::assumed(),
            zenpixels_convert::OutputProfile::Named(zenpixels::Cicp::DISPLAY_P3),
            zenpixels::PixelFormat::Rgb8,
            Some(&FailingCms)
        )
        .is_err()
    );
}

#[cfg(feature = "hdr-experimental")]
#[test]
fn composed_hdr_preparation_retains_every_gamut_table() {
    use zenpixels_convert::{ColorPrimaries, HdrConfig};
    let rgb = PixelDescriptor::RGBF32_LINEAR;
    let p3 = rgb.with_primaries(ColorPrimaries::DisplayP3);
    let a = ConvertPlan::new_with_hdr_config(rgb, p3, HdrConfig::for_source_peak(1000.)).unwrap();
    let b = ConvertPlan::new_with_hdr_config(p3, rgb, HdrConfig::for_source_peak(500.)).unwrap();
    let mut worker = RowConverter::from_plan(a.compose_preserving(&b).unwrap());
    worker.prepare(3).unwrap();
    let src = [0.18f32; 9];
    let mut dst = [0f32; 9];
    let allocations = allocation_counter::measure(|| {
        for _ in 0..4 {
            worker
                .try_convert_row(
                    bytemuck::cast_slice(&src),
                    bytemuck::cast_slice_mut(&mut dst),
                    3,
                )
                .unwrap();
        }
    });
    assert_eq!(allocations.count_total, 0);
    assert!(dst.iter().all(|v| v.is_finite()));
}

#[cfg(feature = "hdr-experimental")]
#[test]
fn hdr_unassociates_before_nonlinear_mapping() {
    use zenpixels_convert::HdrConfig;
    let straight = PixelDescriptor::RGBAF32_LINEAR;
    let premul = straight.with_alpha(Some(AlphaMode::Premultiplied));
    let hdr = HdrConfig::for_source_peak(1000.);
    let mut a =
        RowConverter::from_plan(ConvertPlan::new_with_hdr_config(straight, straight, hdr).unwrap());
    let mut b =
        RowConverter::from_plan(ConvertPlan::new_with_hdr_config(premul, premul, hdr).unwrap());
    let src = [0.8f32, 0.4, 0.2, 0.5];
    let associated = [0.4f32, 0.2, 0.1, 0.5];
    let mut out_a = [0f32; 4];
    let mut out_b = [0f32; 4];
    a.try_convert_row(
        bytemuck::cast_slice(&src),
        bytemuck::cast_slice_mut(&mut out_a),
        1,
    )
    .unwrap();
    b.try_convert_row(
        bytemuck::cast_slice(&associated),
        bytemuck::cast_slice_mut(&mut out_b),
        1,
    )
    .unwrap();
    for c in 0..3 {
        assert!((out_a[c] * 0.5 - out_b[c]).abs() < 1e-6);
    }
    assert_eq!(out_b[3], 0.5);
    for invalid in [
        PixelDescriptor::GRAY8_SRGB.with_transfer(TransferFunction::Pq),
        straight.with_primaries(zenpixels_convert::ColorPrimaries::Unknown),
    ] {
        assert!(ConvertPlan::new_with_hdr_config(invalid, straight, hdr).is_err());
    }
}

#[test]
fn conversion_replaces_current_color_signaling_and_refuses_ambiguous_context() {
    use std::sync::Arc;
    use zenpixels::{Cicp, ColorContext, PixelBuffer};
    use zenpixels_convert::PixelBufferConvertExt;
    let buffer = PixelBuffer::from_vec(vec![128, 64, 32], 1, 1, PixelDescriptor::RGB8_SRGB)
        .unwrap()
        .with_color_context(Arc::new(ColorContext::from_cicp(Cicp::SRGB)));
    let linear = buffer.convert_to(PixelDescriptor::RGBF32_LINEAR).unwrap();
    assert_eq!(
        linear
            .color_context()
            .unwrap()
            .cicp
            .unwrap()
            .transfer_characteristics,
        8
    );
    assert!(linear.color_context().unwrap().icc.is_none());
    let ambiguous = buffer.with_color_context(Arc::new(
        ColorContext::from_cicp(Cicp::SRGB).with_icc(vec![1, 2, 3]),
    ));
    assert!(
        ambiguous
            .convert_to(PixelDescriptor::RGBF32_LINEAR)
            .is_err()
    );
}

#[cfg(feature = "hdr-experimental")]
#[test]
fn pq_and_explicit_linear_anchor_describe_the_same_luminance() {
    use std::sync::Arc;
    use zenpixels::{ColorContext, ColorPrimaries, PixelBuffer};
    use zenpixels_convert::{HdrConfig, PixelBufferHdrConvertExt, TransferFunctionExt};
    let linear = PixelDescriptor::RGBF32_LINEAR.with_primaries(ColorPrimaries::Bt2020);
    let pq = linear.with_transfer(TransferFunction::Pq);
    let peak = 1000.;
    let code = TransferFunction::Pq.delinearize(peak / 10_000.);
    let pq_buffer =
        PixelBuffer::from_vec(bytemuck::cast_slice(&[code; 3]).to_vec(), 1, 1, pq).unwrap();
    let relative = PixelBuffer::from_vec(
        bytemuck::cast_slice(&[peak / 203.; 3]).to_vec(),
        1,
        1,
        linear,
    )
    .unwrap()
    .with_color_context(Arc::new(
        ColorContext::default().with_diffuse_white(zenpixels::hdr::DiffuseWhite::BT2408),
    ));
    let config = HdrConfig::for_source_peak(peak);
    let a = pq_buffer
        .convert_to_with_hdr_config(linear, config)
        .unwrap();
    let b = relative.convert_to_with_hdr_config(linear, config).unwrap();
    let a_view = a.as_slice();
    let b_view = b.as_slice();
    let av: &[f32] = bytemuck::cast_slice(a_view.row(0));
    let bv: &[f32] = bytemuck::cast_slice(b_view.row(0));
    for (a, b) in av.iter().zip(bv) {
        assert!((a - b).abs() < 0.001, "{a} vs {b}");
    }
    assert_eq!(
        b.color_context().unwrap().diffuse_white.unwrap().nits(),
        100.
    );
}

#[test]
fn padding_is_not_alpha_and_opaque_requires_a_proof() {
    use zenpixels::{AlphaPolicy, ConvertOptions};
    for (from, src, expected) in [
        (
            PixelDescriptor::RGBX8.with_transfer(TransferFunction::Srgb),
            [10, 20, 30, 0],
            [10, 20, 30, 255],
        ),
        (
            PixelDescriptor::BGRX8.with_transfer(TransferFunction::Srgb),
            [30, 20, 10, 0],
            [10, 20, 30, 255],
        ),
    ] {
        let mut worker = RowConverter::new(from, PixelDescriptor::RGBA8_SRGB).unwrap();
        let mut dst = [0; 4];
        worker.try_convert_row(&src, &mut dst, 1).unwrap();
        assert_eq!(dst, expected);
    }
    let rgba = PixelDescriptor::RGBA8_SRGB;
    let padding = PixelDescriptor::RGBX8.with_transfer(TransferFunction::Srgb);
    assert!(RowConverter::new(rgba, rgba.with_alpha(Some(AlphaMode::Opaque))).is_err());
    assert!(RowConverter::new_explicit(rgba, padding, &ConvertOptions::forbid_lossy()).is_err());
    let options = ConvertOptions::permissive().with_alpha_policy(AlphaPolicy::CompositeOnto {
        r: 255,
        g: 255,
        b: 255,
    });
    let mut worker = RowConverter::new_explicit(rgba, padding, &options).unwrap();
    let mut dst = [0; 4];
    worker.try_convert_row(&[0, 0, 0, 0], &mut dst, 1).unwrap();
    assert_eq!(&dst[..3], &[255; 3]);
}

#[cfg(feature = "std")]
#[test]
fn row_converter_retains_published_send_sync_traits() {
    fn requires_send_sync<T: Send + Sync>() {}
    requires_send_sync::<RowConverter>();
}
