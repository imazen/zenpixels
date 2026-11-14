#![forbid(unsafe_code)]

mod storage;

use serde_json::{Value, json};
use wasm_bindgen::prelude::*;
use zenpixels::policy::{AlphaPolicy, ConvertOptions, LumaCoefficients};
use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, ColorPrimaries, PixelBuffer, PixelDescriptor,
    PixelFormat, SignalRange, TransferFunction,
};
use zenpixels_convert::RowConverter;

fn field<'a>(v: &'a Value, key: &str) -> Result<&'a str, String> {
    v[key]
        .as_str()
        .ok_or_else(|| format!("Missing string: {key}"))
}
fn channel(v: &Value) -> Result<ChannelType, String> {
    match field(v, "ct")? {
        "U8" => Ok(ChannelType::U8),
        "U16" => Ok(ChannelType::U16),
        "F16" => Ok(ChannelType::F16),
        "F32" => Ok(ChannelType::F32),
        "I8" => Ok(ChannelType::I8),
        "I16" => Ok(ChannelType::I16),
        "U32" => Ok(ChannelType::U32),
        "I32" => Ok(ChannelType::I32),
        "U64" => Ok(ChannelType::U64),
        "I64" => Ok(ChannelType::I64),
        "F64" => Ok(ChannelType::F64),
        other => Err(format!("Unknown ct: {other}")),
    }
}
fn layout(v: &Value) -> Result<ChannelLayout, String> {
    match field(v, "layout")? {
        "Rgb" => Ok(ChannelLayout::Rgb),
        "Bgr" => Ok(ChannelLayout::Bgr),
        "Rgba" => Ok(ChannelLayout::Rgba),
        "Bgra" => Ok(ChannelLayout::Bgra),
        "Gray" => Ok(ChannelLayout::Gray),
        "GrayAlpha" => Ok(ChannelLayout::GrayAlpha),
        "Oklab" => Ok(ChannelLayout::Oklab),
        "OklabA" => Ok(ChannelLayout::OklabA),
        "Cmyk" => Ok(ChannelLayout::Cmyk),
        other => Err(format!("Unknown layout: {other}")),
    }
}
fn transfer(v: &Value) -> Result<TransferFunction, String> {
    match field(v, "transfer")? {
        "Srgb" => Ok(TransferFunction::Srgb),
        "Linear" => Ok(TransferFunction::Linear),
        "Bt709" => Ok(TransferFunction::Bt709),
        "Gamma22" => Ok(TransferFunction::Gamma22),
        "Pq" => Ok(TransferFunction::Pq),
        "Hlg" => Ok(TransferFunction::Hlg),
        "Unknown" => Ok(TransferFunction::Unknown),
        other => Err(format!("Unknown transfer: {other}")),
    }
}
fn primaries(v: &Value) -> Result<ColorPrimaries, String> {
    match field(v, "primaries")? {
        "Bt709" => Ok(ColorPrimaries::Bt709),
        "DisplayP3" => Ok(ColorPrimaries::DisplayP3),
        "Bt2020" => Ok(ColorPrimaries::Bt2020),
        "AdobeRgb" => Ok(ColorPrimaries::AdobeRgb),
        "Unknown" => Ok(ColorPrimaries::Unknown),
        other => Err(format!("Unknown primaries: {other}")),
    }
}
fn alpha(v: &Value) -> Result<AlphaMode, String> {
    match field(v, "alphaMode")? {
        "Undefined" => Ok(AlphaMode::Undefined),
        "Straight" => Ok(AlphaMode::Straight),
        "Premultiplied" => Ok(AlphaMode::Premultiplied),
        "Opaque" => Ok(AlphaMode::Opaque),
        other => Err(format!("Unknown alphaMode: {other}")),
    }
}
fn luma(v: &Value) -> Result<LumaCoefficients, String> {
    match field(v, "luma")? {
        "Bt709" => Ok(LumaCoefficients::Bt709),
        "Bt601" => Ok(LumaCoefficients::Bt601),
        "Bt2020" => Ok(LumaCoefficients::Bt2020),
        "DisplayP3" => Ok(LumaCoefficients::DisplayP3),
        other => Err(format!("Unknown luma: {other}")),
    }
}

fn descriptor(v: &Value) -> Result<PixelDescriptor, String> {
    let a = match field(v, "alphaKind")? {
        "alpha" => Some(alpha(v)?),
        "pad" => Some(AlphaMode::Undefined),
        "none" => None,
        _ => return Err("Unknown alpha kind".into()),
    };
    let format = match field(v, "id")? {
        "Cmyk8Inverted" => PixelFormat::CMYK8_INVERTED,
        "Cmyk16Inverted" => PixelFormat::CMYK16_INVERTED,
        "CmykF32Inverted" => PixelFormat::CMYKF32_INVERTED,
        _ => PixelFormat::from_parts(channel(v)?, layout(v)?, a)
            .ok_or("Unsupported channel/layout combination")?,
    };
    let mut d = format.descriptor().with_alpha(a);
    if matches!(
        d.layout(),
        ChannelLayout::Rgb
            | ChannelLayout::Bgr
            | ChannelLayout::Rgba
            | ChannelLayout::Bgra
            | ChannelLayout::Gray
            | ChannelLayout::GrayAlpha
    ) {
        d = d.with_transfer(transfer(v)?).with_primaries(primaries(v)?);
    }
    d = d.with_signal_range(match field(v, "range")? {
        "Full" => SignalRange::Full,
        "Narrow" => SignalRange::Narrow,
        _ => return Err("Unknown signal range".into()),
    });
    d.validate().map_err(|e| e.to_string())?;
    Ok(d)
}

fn integer(v: &Value, name: &str, max: u32) -> Result<u32, String> {
    let n = v[name]
        .as_u64()
        .ok_or_else(|| format!("Missing integer: {name}"))?;
    if n == 0 || n > u64::from(max) {
        return Err(format!("{name} must be in 1..={max}"));
    }
    Ok(n as u32)
}

fn run(v: &Value) -> Result<Value, String> {
    let from = descriptor(&v["from"])?;
    let to = descriptor(&v["to"])?;
    let width = integer(v, "width", 8)?;
    let rows = integer(v, "rows", 4)?;
    let stride = integer(v, "stride", 4096)? as usize;
    let source = v["bytes"].as_array().ok_or("Missing bytes array")?;
    let row_bytes = width as usize * from.bytes_per_pixel();
    if stride < row_bytes || source.len() != stride * rows as usize {
        return Err("Input bytes do not match dimensions/stride".into());
    }
    let bytes: Vec<u8> = source
        .iter()
        .map(|v| {
            v.as_u64()
                .filter(|n| *n <= 255)
                .map(|n| n as u8)
                .ok_or("Invalid byte".to_string())
        })
        .collect::<Result<_, _>>()?;
    // The view allocations guarantee native sample alignment. Copy only active rows.
    let mut input =
        PixelBuffer::try_new(width, rows, from).map_err(|e| format!("{:?}", e.error()))?;
    for y in 0..rows {
        input
            .as_slice_mut()
            .row_mut(y)
            .copy_from_slice(&bytes[y as usize * stride..y as usize * stride + row_bytes]);
    }
    let mut options = ConvertOptions::permissive()
        .with_alpha_policy(AlphaPolicy::DiscardUnchecked)
        .with_luma(Some(luma(v)?));
    for (key, destination) in [("sdrWhite", 0), ("hdrPeak", 1), ("sdrPeak", 2)] {
        if let Some(value) = v[key].as_f64() {
            let value = value as f32;
            if destination == 0 {
                let white = zenpixels::hdr::DiffuseWhite::try_new(value)
                    .ok_or_else(|| format!("{key} must be positive and finite"))?;
                options = options.with_sdr_white(white);
            } else {
                let nits = zenpixels::hdr::Nits::try_new(value)
                    .ok_or_else(|| format!("{key} must be positive and finite"))?;
                options = if destination == 1 {
                    options.with_hdr_peak(zenpixels::hdr::HdrPeak::Nits(nits))
                } else {
                    options.with_sdr_peak(nits)
                };
            }
        }
    }
    if let Some(h) = v.get("hlgRendering").filter(|h| !h.is_null()) {
        let number = |name: &str| {
            h[name]
                .as_f64()
                .map(|n| n as f32)
                .ok_or_else(|| format!("Missing HLG {name}"))
        };
        let peak = zenpixels::hdr::Nits::try_new(number("peak")?)
            .ok_or("HLG peak must be finite and positive")?;
        let rendering = zenpixels::hdr::HlgRendering::new(peak, number("black")?, number("gamma")?)
            .ok_or(
                "HLG black must be 0..=1% of peak; gamma must be 0.5..=3; values must be finite",
            )?;
        options = options.with_hlg_rendering(rendering);
    }
    let mut converter =
        RowConverter::new_explicit(from, to, &options).map_err(|e| format!("{:?}", e.error()))?;
    // Debug is the existing inspection surface; no new core API is needed.
    let plan = format!("{:#?}", converter.plan());
    let mut output =
        PixelBuffer::try_new(width, rows, to).map_err(|e| format!("{:?}", e.error()))?;
    converter
        .prepare(width)
        .map_err(|e| format!("{:?}", e.error()))?;
    for y in 0..rows {
        converter
            .try_convert_row(
                input.as_slice().row(y),
                output.as_slice_mut().row_mut(y),
                width,
            )
            .map_err(|e| format!("{:?}", e.error()))?;
    }
    Ok(
        json!({"bytes": output.copy_to_contiguous_bytes(), "plan": plan, "stride": width as usize * to.bytes_per_pixel()}),
    )
}

/// Browser-only JSON boundary for the unpublished explorer tool.
#[wasm_bindgen]
pub fn convert(request: &str) -> String {
    let result = serde_json::from_str(request)
        .map_err(|e| e.to_string())
        .and_then(|v| run(&v));
    match result {
        Ok(value) => value.to_string(),
        Err(error) => json!({"error": error}).to_string(),
    }
}

/// Browser-only live storage witnesses for the current crate APIs.
#[wasm_bindgen]
pub fn inspect_storage(request: &str) -> String {
    match serde_json::from_str(request)
        .map_err(|e| e.to_string())
        .and_then(|v| storage::inspect(&v))
    {
        Ok(value) => value.to_string(),
        Err(error) => json!({"error":error}).to_string(),
    }
}

/// Source-derived format facts used to check the teaching model for drift.
#[wasm_bindgen]
pub fn format_catalog() -> String {
    let mut formats = Vec::new();
    for channel in [
        ChannelType::U8,
        ChannelType::U16,
        ChannelType::F16,
        ChannelType::F32,
        ChannelType::I8,
        ChannelType::I16,
        ChannelType::U32,
        ChannelType::I32,
        ChannelType::U64,
        ChannelType::I64,
        ChannelType::F64,
    ] {
        for layout in [
            ChannelLayout::Gray,
            ChannelLayout::GrayAlpha,
            ChannelLayout::Rgb,
            ChannelLayout::Rgba,
        ] {
            let alpha = layout.has_alpha().then_some(AlphaMode::Straight);
            formats.push(PixelFormat::from_parts(channel, layout, alpha).unwrap());
        }
    }
    formats.extend([
        PixelFormat::Bgr8,
        PixelFormat::Bgra8,
        PixelFormat::Rgbx8,
        PixelFormat::Bgrx8,
        PixelFormat::OklabF32,
        PixelFormat::OklabaF32,
        PixelFormat::CMYK8,
        PixelFormat::CMYK8_INVERTED,
        PixelFormat::CMYK16,
        PixelFormat::CMYK16_INVERTED,
        PixelFormat::CMYKF32,
        PixelFormat::CMYKF32_INVERTED,
    ]);
    let entries: Vec<_> = formats
        .into_iter()
        .map(|f| {
            let d = f.descriptor();
            let is_cmyk = matches!(f.color_model(), zenpixels::ColorModel::Cmyk);
            let generic = matches!(f, PixelFormat::Generic(_)) && !is_cmyk;
            let id = match f {
                PixelFormat::CMYK8 => "Cmyk8".to_string(),
                PixelFormat::CMYK16 => "Cmyk16".to_string(),
                _ if is_cmyk || generic => f.name().to_string(),
                _ => format!("{f:?}"),
            };
            json!({"id":id, "ct":format!("{:?}", f.channel_type()),
            "layout":format!("{:?}", d.layout()), "bytes":d.bytes_per_pixel(),
            "generic":generic})
        })
        .collect();
    json!(entries).to_string()
}
