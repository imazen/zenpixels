//! Executable examples of existing typed component views.
use serde_json::{Value, json};
use zenpixels::ChannelType;
use zenpixels::plane::{AnyPlane, PlaneRef, PlaneRole};
use zenpixels::sample::{CodePlane, SampleEncoding};

pub(crate) fn inspect(v: &Value) -> Result<Value, String> {
    let case = v["case"].as_str().ok_or("Choose a storage example")?;
    let padding = v["padding"].as_bool().unwrap_or(true);
    let (components, bits, shifts, role) = match case {
        "p010" => (
            PlaneRole::CbCr.components(),
            vec![10, 10],
            vec![6, 6],
            "CbCr",
        ),
        "crcb" => (
            PlaneRole::CrCb.components(),
            vec![10, 10],
            vec![6, 6],
            "CrCb",
        ),
        "rgba10a1" => (4, vec![10, 10, 10, 1], vec![0; 4], "RGBA"),
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
    let bytes: Vec<_> = words.iter().flat_map(|word| word.to_ne_bytes()).collect();
    Ok(
        json!({"case":case,"role":role,"bytes":bytes,"width":width,"rows":rows,"stride":stride*2,
        "planeCodes":plane_codes,
        "planeStatus":"CodePlane::component reads interleaved components with a different SampleEncoding for each. The storage is an aligned typed U16 plane."}),
    )
}
