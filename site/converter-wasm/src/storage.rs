//! Executable examples of the current storage views, not numerical conversions.
use serde_json::{Value, json};
use zenpixels::plane::{AnyPlane, PlaneRef, PlaneRole};
use zenpixels::sample::{CodePlane, NativeCodeSlice, SampleEncoding};
use zenpixels::{ChannelLayout, ChannelType};

pub(crate) fn inspect(v: &Value) -> Result<Value, String> {
    let case = v["case"].as_str().ok_or("Choose a storage example")?;
    let offset = v["offset"].as_u64().unwrap_or(0).min(1) as usize;
    let padding = v["padding"].as_bool().unwrap_or(true);
    if case == "u64" || case == "i64" {
        let storage = if case == "u64" {
            ChannelType::U64
        } else {
            ChannelType::I64
        };
        let enc = [SampleEncoding::new(storage, 64, 0).map_err(|e| e.to_string())?];
        let mut bytes = vec![0xA5; offset];
        bytes.extend_from_slice(&u64::MAX.to_ne_bytes());
        let native = NativeCodeSlice::new(&bytes[offset..], 1, 1, 8, ChannelLayout::Gray, &enc)
            .map_err(|e| e.to_string())?;
        return Ok(
            json!({"case":case, "role":"Gray", "bytes": &bytes[offset..],
            "nativeCodes":[[native.code(0,0,0).unwrap().to_string()]],
            "planeCodes":null, "planeStatus":"AnyPlane has U8/U16/F32 variants; CodePlane cannot read signed or wide integer words.",
            "nativeStatus":"Exact integer carriage, including byte-unaligned input. Numerical conversion is a separate capability.",
            "width":1,"rows":1,"stride":8,"offset":offset}),
        );
    }
    let (components, bits, shifts, role, layout) = match case {
        "p010" => (
            PlaneRole::CbCr.components(),
            vec![10, 10],
            vec![6, 6],
            "CbCr",
            None,
        ),
        "crcb" => (
            PlaneRole::CrCb.components(),
            vec![10, 10],
            vec![6, 6],
            "CrCb",
            None,
        ),
        "rgba10a1" => (
            4,
            vec![10, 10, 10, 1],
            vec![0; 4],
            "RGBA",
            Some(ChannelLayout::Rgba),
        ),
        _ => return Err("Unknown storage example".into()),
    };
    let enc: Vec<_> = bits
        .iter()
        .zip(&shifts)
        .map(|(&b, &s)| SampleEncoding::new(ChannelType::U16, b, s).unwrap())
        .collect();
    let width = 2;
    let rows = 2;
    let stride = width * components + 2;
    let mut words = vec![0xA5A5; stride + width * components];
    for y in 0..rows {
        for x in 0..width {
            for c in 0..components {
                let mask = (1u16 << bits[c]) - 1;
                let code = if x == 0 {
                    mask
                } else {
                    (y as u16 + c as u16) & mask
                };
                let outside = if padding { !(mask << shifts[c]) } else { 0 };
                words[y * stride + x * components + c] = (code << shifts[c]) | outside;
            }
        }
    }
    let plane = AnyPlane::U16(
        PlaneRef::new(&words[..], width * components, rows, stride).map_err(|e| e.to_string())?,
    );
    let readers: Vec<_> = enc
        .iter()
        .enumerate()
        .map(|(c, &e)| CodePlane::component(plane, components, c, e).unwrap())
        .collect();
    let mut plane_codes = Vec::new();
    for y in 0..rows {
        for x in 0..width {
            plane_codes.push(
                readers
                    .iter()
                    .map(|p| p.code(x, y).unwrap().to_string())
                    .collect::<Vec<_>>(),
            );
        }
    }
    let mut bytes = vec![0xA5; offset];
    for word in &words {
        bytes.extend_from_slice(&word.to_ne_bytes());
    }
    let native_codes = if let Some(layout) = layout {
        let native = NativeCodeSlice::new(
            &bytes[offset..],
            width as u32,
            rows as u32,
            stride * 2,
            layout,
            &enc,
        )
        .map_err(|e| e.to_string())?;
        let mut codes = Vec::new();
        for y in 0..rows {
            for x in 0..width {
                codes.push(
                    (0..components)
                        .map(|c| native.code(x as u32, y as u32, c).unwrap().to_string())
                        .collect::<Vec<_>>(),
                );
            }
        }
        Some(codes)
    } else {
        None
    };
    Ok(
        json!({"case":case,"role":role,"bytes":&bytes[offset..], "width":width,"rows":rows,"stride":stride*2,"offset":offset,
        "planeCodes":plane_codes,"nativeCodes":native_codes,
        "planeStatus":"CodePlane::component already supports interleaved components and a different SampleEncoding for each. This reads the original typed U16 plane.",
        "nativeStatus":if layout.is_some() { "Same codes via the byte-backed RGB/gray adapter. An odd byte offset is supported here; the typed plane needs aligned storage." } else { "No CbCr/CrCb layout in NativeCodeSlice. Do not relabel chroma as GrayAlpha to force it through." }}),
    )
}
