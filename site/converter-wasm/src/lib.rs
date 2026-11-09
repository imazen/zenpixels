#![forbid(unsafe_code)]

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
    let format = if field(v, "id")? == "CmykF32Inverted" {
        PixelFormat::CmykF32Inverted
    } else {
        PixelFormat::from_parts(channel(v)?, layout(v)?, a)
            .ok_or("Unsupported channel/layout combination")?
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
    d.validate().map_err(|e| format!("{e:?}"))?;
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
