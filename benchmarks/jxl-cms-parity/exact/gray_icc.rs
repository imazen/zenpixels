// Is load_bearing's GRAY-class ICC accepted by readers on a gray PNG?
use lcms_cref as c;
use std::os::raw::c_void;
use std::sync::Arc;
use zenpixels::{Cicp, ColorContext, PixelBuffer, PixelDescriptor};
use zenpixels_convert::PixelBufferLoadBearingExt;
use zenpixels_convert::icc_profiles::SynthesizedIcc;

fn analytic(t: u8, e: f64) -> Option<f64> {
    let (a, b, c0) = (0.17883277, 0.28466892, 0.55991073);
    Some(match t {
        1 | 6 | 14 | 15 => if e < 0.081 { e / 4.5 } else { ((e + 0.099) / 1.099).powf(1.0 / 0.45) },
        13 => if e <= 0.04045 { e / 12.92 } else { ((e + 0.055) / 1.055).powf(2.4) },
        8 => e,
        4 => e.powf(2.2),
        5 => e.powf(2.8),
        16 => { let p = e.powf(1.0 / 78.84375); ((p - 0.8359375).max(0.0) / (18.8515625 - 18.6875 * p)).powf(1.0 / 0.1593017578125) }
        18 => if e <= 0.5 { e * e / 3.0 } else { (((e - c0) / a).exp() + b) / 12.0 },
        _ => return None,
    })
}
fn srgb_enc(v: f64) -> f64 { if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 } }

/// Gray8 ramp through C lcms2 to sRGB float. Returns (accepted, worst R-G-B spread, black, white, ramp).
fn lcms_gray(icc: &[u8]) -> Option<(f32, [f32; 3], [f32; 3], Vec<f32>)> {
    let gray8 = 3 << 16 | 1 << 3 | 1u32;
    let rgbf = 1 << 22 | 4 << 16 | 3 << 3 | 4u32;
    unsafe {
        let p = c::cmsOpenProfileFromMem(icc.as_ptr() as *const c_void, icc.len() as u32);
        if p.is_null() { return None; }
        let s = c::cmsCreate_sRGBProfile();
        let t = c::cmsCreateTransform(p, gray8, s, rgbf, 1, 0x100);
        if t.is_null() { return None; }
        let input: Vec<u8> = (0..=255).collect();
        let mut out = vec![0f32; 768];
        c::cmsDoTransform(t, input.as_ptr() as *const c_void, out.as_mut_ptr() as *mut c_void, 256);
        c::cmsDeleteTransform(t);
        let spread = out.chunks_exact(3).map(|p| (p[0] - p[1]).abs().max((p[1] - p[2]).abs())).fold(0f32, f32::max);
        Some((spread, [out[0], out[1], out[2]], [out[765], out[766], out[767]], out))
    }
}

fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let mut profiles = 0;
    let (mut not_needed, mut unsupported, mut class_ok, mut lcms_ok, mut png_ok) = (0, 0, 0, 0, 0);
    let mut worst_spread = 0f32;
    let mut per_t: std::collections::BTreeMap<u8, (f64, f64, usize)> = Default::default();
    let ramp: Vec<u8> = (0..=255).collect();
    let mut failures = Vec::new();
    for p in 0u8..=23 {
        for t in 0u8..=19 {
            let cicp = Cicp::new(p, t, 0, true);
            let icc = match SynthesizedIcc::for_gray_cicp(cicp) {
                SynthesizedIcc::Profile(b) => b.into_owned(),
                SynthesizedIcc::NotNeeded => { not_needed += 1; continue; }
                _ => { unsupported += 1; continue; }
            };
            profiles += 1;
            if &icc[12..16] == b"mntr" && &icc[16..20] == b"GRAY" && &icc[20..24] == b"XYZ " && &icc[36..40] == b"acsp" { class_ok += 1; } else { failures.push(format!("{p}/{t} header {:?}", String::from_utf8_lossy(&icc[12..24]))); }
            // Encoded with the png crate (simplified: not through zenpng's writer), read back the same way.
            let mut bytes = Vec::new();
            {
                let mut info = png::Info::with_size(256, 1);
                info.color_type = png::ColorType::Grayscale;
                info.bit_depth = png::BitDepth::Eight;
                info.icc_profile = Some(icc.clone().into());
                let enc = png::Encoder::with_info(&mut bytes, info).unwrap();
                let mut w = enc.write_header().unwrap();
                w.write_image_data(&ramp).unwrap();
            }
            let path = format!("{dir}/gray_p{p}_t{t}.png");
            std::fs::write(&path, &bytes).unwrap();
            let dec = png::Decoder::new(std::io::Cursor::new(&bytes));
            let reader = dec.read_info().unwrap();
            let info = reader.info();
            assert_eq!(info.color_type, png::ColorType::Grayscale);
            let got = info.icc_profile.as_ref().map(|c| c.to_vec()).unwrap_or_default();
            if got == icc { png_ok += 1; } else { failures.push(format!("{p}/{t} iCCP bytes differ ({} vs {})", got.len(), icc.len())); }
            match lcms_gray(&got) {
                Some((spread, black, white, out)) => {
                    lcms_ok += 1;
                    worst_spread = worst_spread.max(spread);
                    let e = per_t.entry(t).or_insert((0.0, 0.0, 0));
                    e.2 += 1;
                    let bw = black.iter().map(|v| v.abs()).fold(0f32, f32::max) as f64;
                    let wb = white.iter().map(|v| (v - 1.0).abs()).fold(0f32, f32::max) as f64;
                    e.1 = e.1.max(bw.max(wb));
                    if let Some(_) = analytic(t, 0.5) {
                        for i in 0..256 {
                            let want = srgb_enc(analytic(t, i as f64 / 255.0).unwrap().clamp(0.0, 1.0)) * 255.0;
                            e.0 = e.0.max((out[i * 3 + 1] as f64 * 255.0 - want).abs());
                        }
                    }
                }
                None => failures.push(format!("{p}/{t} refused by lcms")),
            }
        }
    }
    println!("grid 24x20: {profiles} gray profiles, {not_needed} NotNeeded, {unsupported} unsupported");
    println!("header mntr/GRAY/XYZ/acsp: {class_ok}/{profiles}; iCCP round trip through png crate: {png_ok}/{profiles}; C lcms2 opened and built a transform: {lcms_ok}/{profiles}; worst R/G/B spread {worst_spread:.6}");
    for (t, (w, end, n)) in &per_t { println!("  transfer {t:>2}: {n} profiles, worst vs analytic {w:.3} of 255 (0 = not compared), worst black/white {end:.6}"); }
    for f in &failures { println!("  FAIL {f}"); }

    println!("== the load_bearing path: RGB ICC context, R==G==B, reduced in place ==");
    for (p, t) in [(12u8, 13u8), (9, 16), (9, 18), (9, 1), (1, 8), (12, 16), (1, 13), (1, 1)] {
        let cicp = Cicp::new(p, t, 0, true);
        let rgb_icc = match SynthesizedIcc::for_cicp(cicp) { SynthesizedIcc::Profile(b) => b.into_owned(), o => { println!("  {p}/{t}: RGB synth {o:?}"); continue; } };
        let mut buf = PixelBuffer::try_new(256, 1, PixelDescriptor::RGB8_SRGB).unwrap();
        { let mut s = buf.as_slice_mut(); s.row_mut(0).chunks_exact_mut(3).enumerate().for_each(|(i, px)| px.fill(i as u8)); }
        let mut ctx = ColorContext::from_icc(rgb_icc.clone());
        ctx.cicp = Some(cicp);
        let mut buf = buf.with_color_context(Arc::new(ctx));
        let before = buf.descriptor();
        buf.reduce_to_load_bearing_format_in_place(false);
        let ctx = buf.color_context().cloned();
        let icc = ctx.as_ref().and_then(|c| c.icc.clone());
        println!("  {p}/{t}: {:?}/{:?} -> {:?}/{:?}, ICC {:?}, cicp kept {}", before.layout(), before.channel_type(), buf.descriptor().layout(), buf.descriptor().channel_type(), icc.as_ref().map(|i| String::from_utf8_lossy(&i[16..20]).to_string()), ctx.as_ref().and_then(|c| c.cicp).is_some());
        if let Some(icc) = icc { let (s, b, w, _) = lcms_gray(&icc).expect("lcms refused"); println!("      C lcms: spread {s:.6}, black {b:?}, white {w:?}"); }
    }
    // A profile with no derivable CICP keeps RGB, so an RGB ICC never lands on a gray image.
    let mut buf = PixelBuffer::try_new(4, 1, PixelDescriptor::RGB8_SRGB).unwrap().with_color_context(Arc::new(ColorContext::from_icc(zenpixels_convert::icc_profiles::ADOBE_RGB.to_vec())));
    buf.reduce_to_load_bearing_format_in_place(false);
    println!("  Adobe RGB ICC, all-zero pixels: {:?}, ICC class {:?}", buf.descriptor().layout(), buf.color_context().and_then(|c| c.icc.clone()).map(|i| String::from_utf8_lossy(&i[16..20]).to_string()));
}
