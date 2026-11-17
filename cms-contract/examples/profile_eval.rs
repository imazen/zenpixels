//! Evaluate exact ICC bytes, arbitrary float rows and a literal rendering intent.
#![forbid(unsafe_code)]
use lcms::{profile, types, xform};
use zenpixels::{ColorProfileSource, PixelFormat};
use zenpixels_convert::{
    cms::PluggableCms, cms_lcms::Lcms, cms_moxcms::MoxCms, policy::ConvertOptions,
};

fn format(icc: &[u8]) -> Result<(u32, PixelFormat, usize), Box<dyn std::error::Error>> {
    match icc.get(16..20) {
        Some(b"RGB ") => Ok(((1 << 22) | (4 << 16) | (3 << 3) | 4, PixelFormat::RgbF32, 3)),
        Some(b"GRAY") => Ok((
            (1 << 22) | (3 << 16) | (1 << 3) | 4,
            PixelFormat::GrayF32,
            1,
        )),
        Some(b"XYZ ") => Ok((
            (1 << 22) | (9 << 16) | (3 << 3) | 4,
            PixelFormat::from_parts(
                zenpixels::ChannelType::F32,
                zenpixels::ChannelLayout::Xyz,
                None,
            )
            .unwrap(),
            3,
        )),
        Some(b"Lab ") => Ok((
            (1 << 22) | (10 << 16) | (3 << 3) | 4,
            PixelFormat::from_parts(
                zenpixels::ChannelType::F32,
                zenpixels::ChannelLayout::Lab,
                None,
            )
            .unwrap(),
            3,
        )),
        Some(b"CMYK") => Ok((
            (1 << 22) | (6 << 16) | (4 << 3) | 4,
            PixelFormat::CMYKF32,
            4,
        )),
        _ => Err("only RGB/GRAY/XYZ/Lab/CMYK float fixtures supported".into()),
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 5 {
        return Err("expected SOURCE.icc DEST.icc INPUT.f32 OUTPUT.f32 INTENT|production".into());
    }
    let src = std::fs::read(&args[0])?;
    let dst = std::fs::read(&args[1])?;
    let (src_code, src_format, src_channels) = format(&src)?;
    let (dst_code, dst_format, dst_channels) = format(&dst)?;
    let mut input = std::fs::read(&args[2])?;
    if input.len() % (src_channels * 4) != 0 {
        return Err("partial input pixel".into());
    }
    let count = input.len() / (src_channels * 4);
    let mut output = vec![0u8; count * dst_channels * 4];
    let inverted = args[4].ends_with(":inverted");
    let mode = args[4].strip_suffix(":inverted").unwrap_or(&args[4]);
    let named = mode.ends_with(":named-srgb");
    let mode = mode.strip_suffix(":named-srgb").unwrap_or(mode);
    if matches!(mode, "production" | "mox") {
        let backend: &dyn PluggableCms = if mode == "mox" { &MoxCms } else { &Lcms };
        let mut transform = backend
            .build_source_transform(
                ColorProfileSource::Icc(&src),
                if named {
                    ColorProfileSource::Cicp(zenpixels::Cicp::SRGB)
                } else {
                    ColorProfileSource::Icc(&dst)
                },
                if inverted && src_format == PixelFormat::CMYKF32 {
                    PixelFormat::CMYKF32_INVERTED
                } else {
                    src_format
                },
                dst_format,
                &ConvertOptions::permissive().with_clip_out_of_gamut(false),
            )
            .ok_or("production transform declined")?
            .map_err(|e| format!("{e:?}"))?;
        transform
            .try_transform_row(&input, &mut output, u32::try_from(count)?)
            .map_err(|e| format!("{e:?}"))?;
    } else if let Some(intent) = args[4].strip_prefix("mox-raw:") {
        let intent: u32 = intent.parse()?;
        if intent > 3 {
            return Err("intent outside 0..3".into());
        }
        let mut source = moxcms::ColorProfile::new_from_slice(&src)?;
        let mut target = moxcms::ColorProfile::new_from_slice(&dst)?;
        source.cicp = None;
        target.cicp = None;
        let layout = |channels| match channels {
            1 => moxcms::Layout::Gray,
            3 => moxcms::Layout::Rgb,
            4 => moxcms::Layout::Rgba,
            _ => unreachable!(),
        };
        let transform = source.create_transform_f32(
            layout(src_channels),
            &target,
            layout(dst_channels),
            moxcms::TransformOptions {
                rendering_intent: moxcms::RenderingIntent::try_from(intent)?,
                allow_use_cicp_transfer: false,
                allow_extended_range_rgb_xyz: true,
                prefer_fixed_point: false,
                barycentric_weight_scale: moxcms::BarycentricWeightScale::High,
                ..Default::default()
            },
        )?;
        let floats: Vec<f32> = input
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_ne_bytes(*c))
            .collect();
        let mut result = vec![0.0f32; count * dst_channels];
        transform.transform(&floats, &mut result)?;
        for (out, value) in output.as_chunks_mut::<4>().0.iter_mut().zip(result) {
            out.copy_from_slice(&value.to_ne_bytes());
        }
    } else {
        let (intent, bpc) = match args[4].strip_suffix(":bpc") {
            Some(intent) => (intent, true),
            None => (args[4].as_str(), false),
        };
        let intent: u32 = intent.parse()?;
        if intent > 3 {
            return Err("intent outside 0..3".into());
        }
        // Little CMS represents floating ink in percent; the probe interface
        // uses normalized ICC ink, matching the production pixel API.
        if src_format == PixelFormat::CMYKF32 {
            for value in input.as_chunks_mut::<4>().0.iter_mut() {
                let scaled = f32::from_ne_bytes(*value) * 100.0;
                value.copy_from_slice(&scaled.to_ne_bytes());
            }
        }
        let mut profiles = [
            profile::open_profile_from_mem(None, &src).ok_or("source parse")?,
            profile::open_profile_from_mem(None, &dst).ok_or("destination parse")?,
        ];
        let adaptation = if intent == 3 { 0.0 } else { 1.0 };
        let mut transform = xform::create_extended_transform(
            None,
            &mut profiles,
            &mut [bpc; 2],
            &[intent; 2],
            &[adaptation; 2],
            None,
            0,
            src_code,
            dst_code,
            types::FLAGS_NOOPTIMIZE
                | types::FLAGS_NOCACHE
                | if bpc {
                    types::FLAGS_BLACKPOINTCOMPENSATION
                } else {
                    0
                },
        )
        .ok_or("literal-intent transform")?;
        transform.do_transform(&input, &mut output, u32::try_from(count)?);
        if dst_format == PixelFormat::CMYKF32 {
            for value in output.as_chunks_mut::<4>().0.iter_mut() {
                let normalized = f32::from_ne_bytes(*value) / 100.0;
                value.copy_from_slice(&normalized.to_ne_bytes());
            }
        }
    }
    if args[3] == "-" {
        use std::io::Write;
        std::io::stdout().write_all(&output)?;
    } else {
        std::fs::write(&args[3], output)?;
    }
    Ok(())
}
