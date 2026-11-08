//! Build-time profile definitions, shared by generation and reproducibility tests.
//! Runtime export uses the committed bundle; independent math lives in the tests.
extern crate alloc;
use alloc::{format, vec, vec::Vec};
use moxcms::{
    Chromaticity, CicpColorPrimaries, CicpProfile, ColorPrimaries, ColorProfile, LocalizableString,
    Matrix3d, MatrixCoefficients, ProfileText, ToneReprCurve, TransferCharacteristics, XyY, Xyzd,
    adapt_to_d50_d,
};

const PCS_D50: Xyzd = Xyzd {
    x: 0.9642,
    y: 1.0,
    z: 0.8249,
};

fn white(p: u8) -> Option<(&'static str, XyY)> {
    let (name, x, y) = match p {
        1 | 5 | 6 | 7 | 9 | 12 | 22 => ("D65", 0.3127, 0.3290),
        4 | 8 => ("C", 0.310, 0.316),
        10 => ("E", 1.0 / 3.0, 1.0 / 3.0),
        11 => ("DCI", 0.314, 0.351),
        _ => return None,
    };
    Some((name, XyY { x, y, yb: 1.0 }))
}

fn primaries(p: u8) -> Option<ColorPrimaries> {
    let xy = match p {
        1 => [[0.640, 0.330], [0.300, 0.600], [0.150, 0.060]],
        4 => [[0.670, 0.330], [0.210, 0.710], [0.140, 0.080]],
        5 => [[0.640, 0.330], [0.290, 0.600], [0.150, 0.060]],
        6 | 7 => [[0.630, 0.340], [0.310, 0.595], [0.155, 0.070]],
        8 => [[0.681, 0.319], [0.243, 0.692], [0.145, 0.049]],
        9 => [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
        11 | 12 => [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]],
        22 => [[0.630, 0.340], [0.295, 0.605], [0.155, 0.077]],
        _ => return None,
    };
    let [red, green, blue] = xy.map(|[x, y]| Chromaticity { x, y });
    Some(ColorPrimaries { red, green, blue })
}

fn cicp(p: u8, t: u8) -> Option<CicpProfile> {
    if t == 18 {
        return None;
    }
    Some(CicpProfile {
        color_primaries: CicpColorPrimaries::try_from(p).ok()?,
        transfer_characteristics: TransferCharacteristics::try_from(t).ok()?,
        matrix_coefficients: MatrixCoefficients::Identity,
        full_range: true,
    })
}

fn curve(t: u8, inherited: &ToneReprCurve) -> ToneReprCurve {
    match t {
        // ICC paraType-0 stores its exponent as s15Fixed16 instead of u8Fixed8.
        4 => ToneReprCurve::Parametric(vec![2.2]),
        5 => ToneReprCurve::Parametric(vec![2.8]),
        // Normalize ICC device white to peak white; lumi records 52.37 cd/m².
        // Multiplication by 52.37/48 recovers H.273's 48-nit normalization.
        17 => ToneReprCurve::Parametric(vec![2.6]),
        9 | 10 => {
            let ToneReprCurve::Lut(samples) = inherited else {
                unreachable!("log table")
            };
            let mut samples = samples.clone();
            // Zero has an interval-valued inverse. Its upper endpoint matches
            // the right-hand limit, avoiding a spurious ramp across positive codes.
            let upper = if t == 9 { 0.01 } else { 10_f64.sqrt() / 1000.0 };
            samples[0] = (upper * 65535.0).round() as u16;
            ToneReprCurve::Lut(samples)
        }
        _ => inherited.clone(),
    }
}

pub(crate) fn rgb(p: u8, t: u8) -> Option<Vec<u8>> {
    let cicp = cicp(p, t)?;
    let mut profile = ColorProfile::new_from_cicp(cicp);
    let trc = curve(t, profile.red_trc.as_ref()?);
    profile.update_rgb_colorimetry(white(p)?.1, primaries(p)?);
    profile.cicp = Some(cicp);
    profile.red_trc = Some(trc.clone());
    profile.green_trc = Some(trc.clone());
    profile.blue_trc = Some(trc);
    if t == 17 {
        profile.luminance = Some(Xyzd {
            x: 0.0,
            y: 52.37,
            z: 0.0,
        });
    }
    encode(profile)
}

pub(crate) fn gray(p: u8, t: u8) -> Option<Vec<u8>> {
    let rgb = ColorProfile::new_from_cicp(cicp(p, t)?);
    let (name, white) = white(p)?;
    let mut gray = ColorProfile::new_gray_with_gamma(1.0);
    gray.gray_trc = Some(curve(t, rgb.red_trc.as_ref()?));
    gray.white_point = PCS_D50;
    gray.media_white_point = Some(PCS_D50);
    gray.chromatic_adaptation = Some(adapt_to_d50_d(Matrix3d::IDENTITY, white));
    if t == 17 {
        gray.luminance = Some(Xyzd {
            x: 0.0,
            y: 52.37,
            z: 0.0,
        });
    }
    gray.description = Some(ProfileText::Localizable(vec![LocalizableString::new(
        "en".into(),
        "US".into(),
        format!("Gray H.273 TC{t} {name} white"),
    )]));
    encode(gray)
}

fn encode(profile: ColorProfile) -> Option<Vec<u8>> {
    let mut bytes = profile.encode().ok()?;
    // ICC.1:2022 explicitly specifies signed PCSXYZ and the CICP tag.
    bytes[8..12].copy_from_slice(&[4, 0x40, 0, 0]);
    super::canonical::canonicalize(bytes)
}
