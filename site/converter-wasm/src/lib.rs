#![forbid(unsafe_code)]

mod storage;

use serde_json::{Value, json};
use wasm_bindgen::prelude::*;
use zenpixels::{
    AlphaMode, ChannelOrder, ColorSpace, DiffuseWhite, HlgMode, HlgRendering, Nits, PixelBuf,
    PixelFormat, Primaries, SampleLayout, SampleType, SignalRange, Transfer,
};
use zenpixels_convert::{AlphaPolicy, ConvertPolicy, DepthPolicy, LumaCoefficients, RowPlan};

fn field<'a>(v: &'a Value, key: &str) -> Result<&'a str, String> {
    v[key]
        .as_str()
        .ok_or_else(|| format!("Missing string: {key}"))
}
fn sample(v: &Value) -> Result<SampleType, String> {
    match field(v, "ct")? {
        "U8" => Ok(SampleType::U8),
        "U16" => Ok(SampleType::U16),
        "F16" => Ok(SampleType::F16),
        "F32" => Ok(SampleType::F32),
        "I8" => Ok(SampleType::I8),
        "I16" => Ok(SampleType::I16),
        "U32" => Ok(SampleType::U32),
        "I32" => Ok(SampleType::I32),
        "U64" => Ok(SampleType::U64),
        "I64" => Ok(SampleType::I64),
        "F64" => Ok(SampleType::F64),
        other => Err(format!("Unknown ct: {other}")),
    }
}
fn order(v: &Value, pad: bool) -> Result<ChannelOrder, String> {
    match (field(v, "layout")?, pad) {
        ("Rgb", false) => Ok(ChannelOrder::Rgb),
        ("Bgr", false) => Ok(ChannelOrder::Bgr),
        ("Rgba", false) => Ok(ChannelOrder::Rgba),
        ("Rgba", true) | ("Rgbx", _) => Ok(ChannelOrder::Rgbx),
        ("Bgra", false) => Ok(ChannelOrder::Bgra),
        ("Bgra", true) | ("Bgrx", _) => Ok(ChannelOrder::Bgrx),
        ("Gray", false) => Ok(ChannelOrder::Gray),
        ("GrayAlpha", false) => Ok(ChannelOrder::GrayAlpha),
        ("Oklab", false) => Ok(ChannelOrder::Oklab),
        ("OklabA", false) => Ok(ChannelOrder::OklabA),
        ("Cmyk", false) => Ok(ChannelOrder::Cmyk),
        ("CmykInverted", false) => Ok(ChannelOrder::CmykInverted),
        (other, _) => Err(format!("Unknown layout: {other}")),
    }
}
fn transfer(v: &Value) -> Result<Transfer, String> {
    match field(v, "transfer")? {
        "Srgb" => Ok(Transfer::Srgb),
        "Linear" => Ok(Transfer::Linear),
        "Bt709" => Ok(Transfer::Bt709),
        "Gamma22" => Ok(Transfer::Gamma22),
        "AdobeRgb" => Ok(Transfer::AdobeRgb),
        "DciCinema" => Ok(Transfer::DciCinema),
        "Pq" => Ok(Transfer::Pq),
        "Hlg" => Ok(Transfer::Hlg(HlgMode::Scene)),
        "Unknown" => Ok(Transfer::Unknown),
        other => Err(format!("Unknown transfer: {other}")),
    }
}
fn primaries(v: &Value) -> Result<Primaries, String> {
    match field(v, "primaries")? {
        "Bt709" => Ok(Primaries::Bt709),
        "DisplayP3" => Ok(Primaries::DisplayP3),
        "Bt2020" => Ok(Primaries::Bt2020),
        "AdobeRgb" => Ok(Primaries::AdobeRgb),
        "DciP3" => Ok(Primaries::DciP3),
        "Unknown" => Ok(Primaries::Unknown),
        other => Err(format!("Unknown primaries: {other}")),
    }
}
fn alpha(v: &Value) -> Result<AlphaMode, String> {
    match field(v, "alphaMode")? {
        "Undefined" | "None" => Ok(AlphaMode::None),
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
        "DisplayP3" => Ok(LumaCoefficients::FromSourcePrimaries),
        other => Err(format!("Unknown luma: {other}")),
    }
}

fn spec_from_json(v: &Value) -> Result<PixelFormat, String> {
    let alpha_kind = field(v, "alphaKind")?;
    let is_pad = alpha_kind == "pad";
    let alpha_mode = match alpha_kind {
        "alpha" => alpha(v)?,
        "pad" | "none" => AlphaMode::None,
        _ => return Err("Unknown alpha kind".into()),
    };
    let ord = order(v, is_pad)?;
    let layout = SampleLayout::new(sample(v)?, ord);
    let range = match field(v, "range")? {
        "Full" => SignalRange::Full,
        "Narrow" => SignalRange::Narrow,
        _ => return Err("Unknown signal range".into()),
    };
    let color = match ord {
        ChannelOrder::Rgb
        | ChannelOrder::Bgr
        | ChannelOrder::Rgba
        | ChannelOrder::Bgra
        | ChannelOrder::Rgbx
        | ChannelOrder::Bgrx => ColorSpace::Rgb {
            primaries: primaries(v)?,
            transfer: transfer(v)?,
            range,
            alpha: alpha_mode,
            white: None,
        },
        ChannelOrder::Gray | ChannelOrder::GrayAlpha => ColorSpace::Gray {
            transfer: transfer(v)?,
            range,
            alpha: alpha_mode,
            white: None,
        },
        ChannelOrder::Oklab | ChannelOrder::OklabA => ColorSpace::Oklab { alpha: alpha_mode },
        ChannelOrder::Cmyk => ColorSpace::Cmyk {
            icc: None,
            alpha: alpha_mode,
            inverted: false,
        },
        ChannelOrder::CmykInverted => ColorSpace::Cmyk {
            icc: None,
            alpha: alpha_mode,
            inverted: true,
        },
        _ => return Err("Unsupported channel order".into()),
    };
    let spec = PixelFormat::new(layout, color);
    spec.validate().map_err(|e| e.to_string())?;
    Ok(spec)
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
    let from = spec_from_json(&v["from"])?;
    let to = spec_from_json(&v["to"])?;
    let width = integer(v, "width", 8)?;
    let rows = integer(v, "rows", 4)?;
    let stride = integer(v, "stride", 4096)? as usize;
    let source = v["bytes"].as_array().ok_or("Missing bytes array")?;
    let row_bytes = width as usize * from.layout.bytes_per_pixel();
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
    let mut input =
        PixelBuf::new(width, rows, from.clone()).map_err(|e| format!("{:?}", e.error()))?;
    for y in 0..rows {
        input
            .as_view_mut()
            .row_mut(y)
            .copy_from_slice(&bytes[y as usize * stride..y as usize * stride + row_bytes]);
    }
    let mut policy = ConvertPolicy::permissive()
        .with_depth(DepthPolicy::Round)
        .with_alpha(AlphaPolicy::DiscardUnchecked)
        .with_luma(luma(v)?);
    let mut hdr_peak: Option<Nits> = None;
    for (key, destination) in [("sdrWhite", 0), ("hdrPeak", 1), ("sdrPeak", 2)] {
        if let Some(value) = v[key].as_f64() {
            let value = value as f32;
            if destination == 0 {
                let white = DiffuseWhite::new(value)
                    .ok_or_else(|| format!("{key} must be positive and finite"))?;
                policy = policy.with_sdr_white(white);
            } else {
                let nits =
                    Nits::new(value).ok_or_else(|| format!("{key} must be positive and finite"))?;
                if destination == 1 {
                    hdr_peak = Some(nits);
                } else {
                    policy = policy.with_sdr_peak(nits);
                }
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
        let peak = Nits::new(number("peak")?).ok_or("HLG peak must be finite and positive")?;
        let rendering = HlgRendering::new(peak, number("black")?, number("gamma")?).ok_or(
            "HLG black must be 0..=1% of peak; gamma must be 0.5..=3; values must be finite",
        )?;
        policy = policy.with_hlg_rendering(rendering);
    }
    let row_plan = RowPlan::build(from, to.clone(), &policy, hdr_peak)
        .map_err(|e| format!("{:?}", e.error()))?;
    let plan = format!("{row_plan:#?}");
    let mut output =
        PixelBuf::new(width, rows, to.clone()).map_err(|e| format!("{:?}", e.error()))?;
    for y in 0..rows {
        row_plan
            .convert_row_at(
                y,
                input.as_view().row(y),
                output.as_view_mut().row_mut(y),
                width,
            )
            .map_err(|e| format!("{:?}", e.error()))?;
    }
    let out_stride = width as usize * to.layout.bytes_per_pixel();
    let mut out_bytes = Vec::with_capacity(out_stride * rows as usize);
    for y in 0..rows {
        out_bytes.extend_from_slice(output.as_view().row(y));
    }
    Ok(json!({
        "bytes": out_bytes,
        "plan": plan,
        "stride": out_stride
    }))
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
    let mut layouts = Vec::new();
    for st in [
        SampleType::U8,
        SampleType::U16,
        SampleType::F16,
        SampleType::F32,
        SampleType::I8,
        SampleType::I16,
        SampleType::U32,
        SampleType::I32,
        SampleType::U64,
        SampleType::I64,
        SampleType::F64,
    ] {
        for ord in [
            ChannelOrder::Gray,
            ChannelOrder::GrayAlpha,
            ChannelOrder::Rgb,
            ChannelOrder::Rgba,
        ] {
            layouts.push(SampleLayout::new(st, ord));
        }
    }
    layouts.extend([
        SampleLayout::BGR8,
        SampleLayout::BGRA8,
        SampleLayout::RGBX8,
        SampleLayout::BGRX8,
        SampleLayout::OKLABF32,
        SampleLayout::OKLABAF32,
        SampleLayout::CMYK8,
        SampleLayout::new(SampleType::U8, ChannelOrder::CmykInverted),
        SampleLayout::CMYK16,
        SampleLayout::new(SampleType::U16, ChannelOrder::CmykInverted),
        SampleLayout::CMYKF32,
        SampleLayout::new(SampleType::F32, ChannelOrder::CmykInverted),
    ]);
    let entries: Vec<_> = layouts
        .into_iter()
        .map(|l| {
            let id = format!("{:?}_{:?}", l.order, l.sample);
            json!({
                "id": id,
                "ct": format!("{:?}", l.sample),
                "layout": format!("{:?}", l.order),
                "bytes": l.bytes_per_pixel(),
                "generic": false
            })
        })
        .collect();
    json!(entries).to_string()
}
