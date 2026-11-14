//! One trait? PluggableCms behind JxlCms, against lcms2 and the decoder's own moxcms wrapper.
#![forbid(unsafe_code)]
use jxl::api::{
    JxlCms, JxlCmsTransformer, JxlColorEncoding, JxlColorProfile, JxlColorType, JxlDataFormat,
    JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat, ProcessingResult,
};
use std::sync::Mutex;
use zenpixels::{ColorProfileSource, PixelFormat};
use zenpixels_convert::cms::{PluggableCms, RowTransformMut};
use zenpixels_convert::cms_moxcms::MoxCms;
use zenpixels_convert::policy::ConvertOptions;

whereat::define_at_crate_info!();
static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
struct Adapter<C>(C);
fn cicp_of(p: &JxlColorProfile) -> Option<zenpixels::Cicp> {
    use jxl::api::{JxlPrimaries as P, JxlTransferFunction as T, JxlWhitePoint as W};
    let JxlColorProfile::Simple(JxlColorEncoding::RgbColorSpace {
        white_point: W::D65,
        primaries,
        transfer_function,
        ..
    }) = p
    else {
        return None;
    };
    let pr = match primaries {
        P::SRGB => 1,
        P::P3 => 12,
        P::BT2100 => 9,
        _ => return None,
    };
    let tc = match transfer_function {
        T::SRGB => 13,
        T::PQ => 16,
        T::HLG => 18,
        T::Linear => 8,
        T::BT709 => 1,
        _ => return None,
    };
    Some(zenpixels::Cicp::new(pr, tc, 0, true))
}
fn has_hlg(profile: &JxlColorProfile) -> bool {
    use jxl::api::JxlTransferFunction;
    if let JxlColorProfile::Icc(bytes) = profile {
        return zenpixels::Cicp::from_icc(bytes)
            .is_some_and(|cicp| cicp.transfer_characteristics == 18);
    }
    matches!(
        profile,
        JxlColorProfile::Simple(
            JxlColorEncoding::RgbColorSpace {
                transfer_function: JxlTransferFunction::HLG,
                ..
            } | JxlColorEncoding::GrayscaleColorSpace {
                transfer_function: JxlTransferFunction::HLG,
                ..
            }
        )
    )
}
struct Tr {
    inner: Mutex<Box<dyn RowTransformMut>>,
    inc: usize,
    outc: usize,
}
fn fmt(ch: usize) -> PixelFormat {
    match ch {
        1 => PixelFormat::GrayF32,
        3 => PixelFormat::RgbF32,
        4 => PixelFormat::CMYKF32_INVERTED,
        _ => panic!("{ch}"),
    }
}
impl<C: PluggableCms> JxlCms for Adapter<C> {
    fn initialize_transforms(
        &self,
        n: usize,
        max: usize,
        input: JxlColorProfile,
        output: JxlColorProfile,
        _intensity: f32,
    ) -> jxl::api::Result<(usize, Vec<Box<dyn JxlCmsTransformer + Send + Sync>>)> {
        let err = |s: String| whereat::at!(jxl::api::Error::CmsError(s));
        if has_hlg(&input) || has_hlg(&output) {
            return Err(err(
                "HLG OOTF is not implemented by this PluggableCms adapter".into(),
            ));
        }
        let (inc, outc) = (input.channels(), output.channels());
        // Pass sRGB and PQ structured RGB as CICP; retain ICC for other profiles.
        let si_owned = input.try_as_icc().map(|c| c.into_owned());
        let di_owned = output.try_as_icc().map(|c| c.into_owned());
        let source = |p: &JxlColorProfile,
                      icc: &Option<Vec<u8>>|
         -> Option<(Option<zenpixels::Cicp>, Vec<u8>)> {
            let c = cicp_of(p).filter(|c| {
                (c.color_primaries == 1 && c.transfer_characteristics == 13)
                    || c.transfer_characteristics == 16
            });
            Some((c, icc.clone().unwrap_or_default()))
        };
        let (sc, si) = source(&input, &si_owned).ok_or_else(|| err("source".into()))?;
        let (dc, di) = source(&output, &di_owned).ok_or_else(|| err("dest".into()))?;
        let opts = ConvertOptions::permissive().with_clip_out_of_gamut(false);
        let mut v: Vec<Box<dyn JxlCmsTransformer + Send + Sync>> = Vec::new();
        for _ in 0..n {
            let mut t = self
                .0
                .build_source_transform(
                    sc.map(ColorProfileSource::Cicp)
                        .unwrap_or(ColorProfileSource::Icc(&si)),
                    dc.map(ColorProfileSource::Cicp)
                        .unwrap_or(ColorProfileSource::Icc(&di)),
                    fmt(inc),
                    fmt(outc),
                    &opts,
                )
                .ok_or_else(|| err("declined".into()))?
                .map_err(|e| err(format!("{e:?}")))?;
            t.prepare(max as u32).map_err(|e| err(format!("{e:?}")))?;
            v.push(Box::new(Tr {
                inner: Mutex::new(t),
                inc,
                outc,
            }));
        }
        Ok((outc, v))
    }
}
impl JxlCmsTransformer for Tr {
    fn do_transform(&mut self, input: &[f32], output: &mut [f32]) -> jxl::api::Result<()> {
        let w = (input.len() / self.inc) as u32;
        CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.inner
            .get_mut()
            .unwrap()
            .try_transform_row(
                bytemuck::cast_slice(input),
                bytemuck::cast_slice_mut(output),
                w,
            )
            .map_err(|e| whereat::at!(jxl::api::Error::CmsError(format!("{e:?}"))))
    }
    fn do_transform_inplace(&mut self, inout: &mut [f32]) -> jxl::api::Result<()> {
        let w = inout.len() / self.inc;
        let copy = inout.to_vec();
        let mut out = vec![0f32; w * self.outc];
        self.do_transform(&copy, &mut out)?;
        inout[..out.len()].copy_from_slice(&out);
        Ok(())
    }
}

fn complete<S, T>(
    mut state: S,
    input: &mut &[u8],
    mut step: impl FnMut(S, &mut &[u8]) -> jxl::api::Result<ProcessingResult<T, S>>,
) -> jxl::api::Result<T> {
    for _ in 0..4096 {
        match step(state, input)? {
            ProcessingResult::Complete { result } => return Ok(result),
            ProcessingResult::NeedsMoreInput { fallback, .. } => state = fallback,
        }
    }
    panic!("stalled");
}

fn decode(
    bytes: &[u8],
    cms: Box<dyn JxlCms>,
    target: &JxlColorProfile,
    ct: JxlColorType,
) -> Result<(usize, usize, Vec<f32>), String> {
    let input = bytes;
    let options = JxlDecoderOptions::default()
        .with_adjust_orientation(false)
        .with_render_spot_colors(false)
        .with_cms(cms);
    let inner = || -> jxl::api::Result<(usize, usize, Vec<f32>)> {
        let mut input = input;
        let mut d = complete(JxlDecoder::new(options), &mut input, |d, i| d.process(i))?;
        let extras = d.basic_info().extra_channels.len();
        d.set_pixel_format(JxlPixelFormat {
            color_type: ct,
            color_data_format: Some(JxlDataFormat::f32()),
            extra_channel_format: vec![None; extras],
        });
        d.set_output_color_profile(target.clone())?;
        let frame = complete(d, &mut input, |d, i| d.process(i))?;
        let (w, h) = frame.frame_header().size;
        let ch = ct.samples_per_pixel();
        let mut data = vec![0f32; w * h * ch];
        let mut buffers = [JxlOutputBuffer::new_with_stride(
            bytemuck::cast_slice_mut(&mut data),
            h,
            w * ch * 4,
            w * ch * 4,
        )];
        complete(frame, &mut input, |d, i| d.process(i, &mut buffers))?;
        drop(buffers);
        Ok((w, h, data))
    };
    inner().map_err(|e| format!("{:?}", e.error()).chars().take(400).collect())
}

fn cmp(a: &[f32], b: &[f32]) -> String {
    if a.len() != b.len() {
        return format!("len {} vs {}", a.len(), b.len());
    }
    let (mut worst, mut over1, mut over2) = (0f32, 0usize, 0usize);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (x.clamp(0.0, 1.0), y.clamp(0.0, 1.0));
        let d = (x - y).abs();
        worst = worst.max(d);
        if d * 255.0 > 1.0 {
            over1 += 1;
        }
        if d * 255.0 > 2.0 {
            over2 += 1;
        }
    }
    format!(
        "worst {:6.2} of 255, >1: {:5.2}%, >2: {:5.2}%",
        worst * 255.0,
        over1 as f64 * 100.0 / a.len() as f64,
        over2 as f64 * 100.0 / a.len() as f64
    )
}

fn cmyk_oracle(profile: Vec<u8>) {
    let dst = JxlColorProfile::Simple(JxlColorEncoding::srgb(false));
    let cases: [([f32; 4], [Option<f32>; 3]); 4] = [
        (
            [0.2863, 1.0, 0.6706, 1.0],
            [Some(0.11756), Some(0.73633), Some(0.72557)],
        ),
        (
            [0.2347, 0.9601, 0.5918, 0.9813],
            [Some(-0.0645), None, None],
        ),
        (
            [1.0, 1.0, 1.0, 0.0],
            [Some(0.13583), Some(0.12103), Some(0.12434)],
        ),
        ([1.0, 1.0, 1.0, 1.0], [Some(1.0), Some(1.0), Some(1.0)]),
    ];
    for (label, cms) in [
        ("decoder", Box::new(jxl::api::MoxCms) as Box<dyn JxlCms>),
        ("adapter", Box::new(Adapter(MoxCms))),
    ] {
        let (_, mut tr) = cms
            .initialize_transforms(
                1,
                1,
                JxlColorProfile::Icc(profile.clone()),
                dst.clone(),
                255.0,
            )
            .unwrap();
        for (i, want) in &cases {
            let mut o = [0f32; 3];
            tr[0].do_transform(i, &mut o).unwrap();
            let worst = want
                .iter()
                .zip(&o)
                .filter_map(|(w, g)| w.map(|w| (w - g).abs()))
                .fold(0f32, f32::max);
            println!(
                "ORACLE {label} {i:?} -> {o:.5?} worst vs lcms2 {:.5} ({:.2} of 255)",
                worst,
                worst * 255.0
            );
        }
    }
}

fn embedded_icc(bytes: &[u8]) -> Vec<u8> {
    let mut input = bytes;
    let d = complete(
        JxlDecoder::new(JxlDecoderOptions::default()),
        &mut input,
        |d, i| d.process(i),
    )
    .unwrap();
    match d.embedded_color_profile() {
        JxlColorProfile::Icc(v) => v.clone(),
        _ => panic!("no icc"),
    }
}

fn main() {
    if let Some(p) = std::env::var_os("CMYK_ORACLE") {
        let bytes = std::fs::read(p).unwrap();
        cmyk_oracle(embedded_icc(&bytes));
        return;
    }
    let mut files: Vec<String> = std::env::args().skip(1).collect();
    files.sort();
    let targets = [
        (
            "srgb",
            JxlColorProfile::Simple(JxlColorEncoding::srgb(false)),
        ),
        (
            "linear",
            JxlColorProfile::Simple(JxlColorEncoding::linear_srgb(false)),
        ),
        (
            "p3",
            JxlColorProfile::Simple(JxlColorEncoding::RgbColorSpace {
                white_point: jxl::api::JxlWhitePoint::D65,
                primaries: jxl::api::JxlPrimaries::P3,
                transfer_function: jxl::api::JxlTransferFunction::SRGB,
                rendering_intent: jxl::api::RenderingIntent::Relative,
            }),
        ),
    ];
    let ct = JxlColorType::Rgb;
    for f in files {
        let bytes = std::fs::read(&f).unwrap();
        let name = f.rsplit('/').next().unwrap();
        for (tn, t) in &targets {
            let l = decode(&bytes, Box::new(jxl_cli::cms::Lcms2Cms), t, ct);
            let m = decode(&bytes, Box::new(jxl::api::MoxCms), t, ct);
            CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
            let a = decode(&bytes, Box::new(Adapter(MoxCms)), t, ct);
            let calls = CALLS.load(std::sync::atomic::Ordering::Relaxed);
            if let (Ok(l), Ok(m), Ok(a)) = (&l, &m, &a)
                && (name.starts_with("hdr_") || name.starts_with("cmyk"))
            {
                for &p in &[
                    (l.1 / 2 * l.0 + l.0 / 2),
                    (l.1 / 4 * l.0 + l.0 / 4),
                    (3 * l.1 / 4 * l.0 + 3 * l.0 / 4),
                ] {
                    let at = |v: &[f32]| [v[p * 3], v[p * 3 + 1], v[p * 3 + 2]];
                    println!(
                        "SAMPLE {name} {tn} px{p}: lcms2 {:.4?} decoder {:.4?} adapter {:.4?}",
                        at(&l.2),
                        at(&m.2),
                        at(&a.2)
                    );
                }
            }
            match (&l, &m, &a) {
                (Ok(l), Ok(m), Ok(a)) => println!(
                    "{name}\t{tn}\tcalls {calls}\t{}x{}\tdecoder-moxcms vs lcms2: {}\tadapter vs lcms2: {}\tadapter vs decoder-moxcms: {}",
                    l.0,
                    l.1,
                    cmp(&m.2, &l.2),
                    cmp(&a.2, &l.2),
                    cmp(&a.2, &m.2)
                ),
                _ => println!(
                    "{name}\t{tn}\tlcms2 {:?}\tdecoder {:?}\tadapter {:?}",
                    l.as_ref().err(),
                    m.as_ref().err(),
                    a.as_ref().err()
                ),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    type Seen = Arc<Mutex<Vec<(Option<zenpixels::Cicp>, Option<zenpixels::Cicp>)>>>;
    struct Spy(Seen);
    impl PluggableCms for Spy {
        fn build_source_transform(
            &self,
            src: ColorProfileSource<'_>,
            dst: ColorProfileSource<'_>,
            _: PixelFormat,
            _: PixelFormat,
            _: &ConvertOptions,
        ) -> Option<
            Result<Box<dyn RowTransformMut>, whereat::At<zenpixels_convert::cms::CmsPluginError>>,
        > {
            let signal = |p| match p {
                ColorProfileSource::Cicp(c) => Some(c),
                _ => None,
            };
            self.0.lock().unwrap().push((signal(src), signal(dst)));
            Some(Err(whereat::at!(
                zenpixels_convert::cms::CmsPluginError::msg("recorded route")
            )))
        }
    }
    fn profile(tf: jxl::api::JxlTransferFunction) -> JxlColorProfile {
        JxlColorProfile::Simple(JxlColorEncoding::RgbColorSpace {
            white_point: jxl::api::JxlWhitePoint::D65,
            primaries: jxl::api::JxlPrimaries::BT2100,
            transfer_function: tf,
            rendering_intent: jxl::api::RenderingIntent::Relative,
        })
    }
    #[test]
    fn pq_reaches_plugin_as_cicp() {
        let seen = Seen::default();
        let adapter = Adapter(Spy(Arc::clone(&seen)));
        assert!(
            adapter
                .initialize_transforms(
                    1,
                    8,
                    profile(jxl::api::JxlTransferFunction::PQ),
                    JxlColorProfile::Simple(JxlColorEncoding::srgb(false)),
                    10000.0
                )
                .is_err()
        );
        assert_eq!(
            *seen.lock().unwrap(),
            vec![(
                Some(zenpixels::Cicp::new(9, 16, 0, true)),
                Some(zenpixels::Cicp::new(1, 13, 0, true))
            )]
        );
    }
    #[test]
    fn hlg_refuses_before_plugin_construction_in_both_directions() {
        let seen = Seen::default();
        let adapter = Adapter(Spy(Arc::clone(&seen)));
        let srgb = JxlColorProfile::Simple(JxlColorEncoding::srgb(false));
        let hlg = profile(jxl::api::JxlTransferFunction::HLG);
        for (source, target) in [(hlg.clone(), srgb.clone()), (srgb, hlg)] {
            let error = match adapter.initialize_transforms(1, 8, source, target, 1000.0) {
                Err(error) => error,
                Ok(_) => panic!("HLG must refuse until OOTF exists"),
            };
            assert!(format!("{error:?}").contains("HLG OOTF"));
        }
        assert!(seen.lock().unwrap().is_empty());
    }
    fn pfm(path: &std::path::Path) -> (usize, usize, Vec<f32>) {
        let bytes = std::fs::read(path).expect("caller-provided djxl reference PFM");
        let mut lines = bytes.splitn(4, |&b| b == b'\n');
        assert_eq!(lines.next().unwrap(), b"PF");
        let geometry = std::str::from_utf8(lines.next().unwrap()).unwrap();
        let mut dims = geometry
            .split_whitespace()
            .map(|v| v.parse::<usize>().unwrap());
        let (w, h) = (dims.next().unwrap(), dims.next().unwrap());
        let scale = std::str::from_utf8(lines.next().unwrap())
            .unwrap()
            .parse::<f32>()
            .unwrap();
        assert_eq!(scale.abs(), 1.0);
        let data = lines.next().unwrap();
        assert_eq!(data.len(), w * h * 3 * 4);
        let mut output = Vec::with_capacity(w * h * 3);
        for row in data.chunks_exact(w * 3 * 4).rev() {
            output.extend(row.as_chunks::<4>().0.iter().map(|&v| {
                if scale > 0.0 {
                    f32::from_be_bytes(v)
                } else {
                    f32::from_le_bytes(v)
                }
            }));
        }
        (w, h, output)
    }
    #[test]
    #[ignore = "caller supplies full JXL fixtures and frozen djxl PFM references"]
    fn pq_matches_djxl_references_and_hlg_returns_refusal() {
        let fixtures = std::path::PathBuf::from(
            std::env::var_os("JXL_CMS_FIXTURES").expect("JXL_CMS_FIXTURES"),
        );
        let references = std::path::PathBuf::from(
            std::env::var_os("JXL_CMS_REFERENCES").expect("JXL_CMS_REFERENCES"),
        );
        let bytes = std::fs::read(fixtures.join("hdr_pq_test.jxl")).unwrap();
        let targets = [
            (
                "SRG",
                JxlColorProfile::Simple(JxlColorEncoding::srgb(false)),
            ),
            (
                "P3",
                JxlColorProfile::Simple(JxlColorEncoding::RgbColorSpace {
                    white_point: jxl::api::JxlWhitePoint::D65,
                    primaries: jxl::api::JxlPrimaries::P3,
                    transfer_function: jxl::api::JxlTransferFunction::SRGB,
                    rendering_intent: jxl::api::RenderingIntent::Relative,
                }),
            ),
        ];
        for (name, target) in targets {
            let (w, h, actual) = decode(
                &bytes,
                Box::new(Adapter(MoxCms)),
                &target,
                JxlColorType::Rgb,
            )
            .unwrap();
            let (rw, rh, expected) = pfm(&references.join(format!("hdr_pq_test_{name}.pfm")));
            assert_eq!((w, h), (rw, rh));
            assert_eq!(actual.len(), expected.len());
            let worst = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| {
                    assert!(a.is_finite() && b.is_finite());
                    (a.clamp(0.0, 1.0) - b.clamp(0.0, 1.0)).abs() * 255.0
                })
                .fold(0.0f32, f32::max);
            println!(
                "PQ full decode vs frozen djxl {name}: {w}x{h}, worst {worst:.6} clamped 8-bit codes"
            );
            assert!(
                worst <= 0.1,
                "{name}: {worst} exceeds a tenth of one display code"
            );
            let hlg = std::fs::read(fixtures.join("hdr_hlg_test.jxl")).unwrap();
            let refusal =
                decode(&hlg, Box::new(Adapter(MoxCms)), &target, JxlColorType::Rgb).unwrap_err();
            assert!(refusal.contains("HLG OOTF"), "{refusal}");
        }
    }
}
