//! CSS color parsing, as far as a single [`Color`] goes.
//!
//! Resolves hex, named colors, `rgb()`, `hsl()`, `hwb()`, `color()`, `lab()`, `lch()`,
//! `oklab()` and `oklch()`. Spaces with no descriptor of their own convert at parse time
//! to linear BT.2020 `f32`, which keeps out-of-gamut values, so a color can't be serialized
//! back as written. `none` reads as 0, which only differs for interpolation.
//!
//! Not parsed, because they need more than the string: `currentcolor`, `light-dark()`,
//! `color-mix()`, relative colors, `calc()`, `@color-profile` and `device-cmyk()`.
//!
//! See `docs/css-color-parity.md`.

use core::fmt;

use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, Color, ColorPrimaries, PixelDescriptor, TransferFunction,
};

/// Why a string isn't a color this parser can return.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CssColorError {
    /// Not valid CSS color syntax, or a name that isn't a color.
    Invalid,
    /// Valid CSS, but it needs context that a string doesn't carry.
    Unsupported,
}

impl fmt::Display for CssColorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Invalid => "not a CSS color",
            Self::Unsupported => "this CSS color form needs context a string doesn't carry",
        })
    }
}

impl core::error::Error for CssColorError {}

type M3 = [[f64; 3]; 3];

/// XYZ D65 to linear BT.2020.
const XYZ_TO_BT2020: M3 = [
    [1.716651187971268, -0.355670783776392, -0.253366281373660],
    [-0.666684351832489, 1.616481236634939, 0.015768545813911],
    [0.017639857445311, -0.042770613257809, 0.942103121235474],
];
/// Bradford, D50 to D65, from CSS Color 4's sample code.
const D50_TO_D65: M3 = [
    [0.955473421488075, -0.02309845494876471, 0.06325924320057072],
    [
        -0.0283697093338637,
        1.0099953980813041,
        0.021041441191917323,
    ],
    [
        0.012314014864481998,
        -0.020507649298898964,
        1.330365926242124,
    ],
];
/// Linear ProPhoto to XYZ D50, from CSS Color 4's sample code.
const PROPHOTO_TO_XYZ_D50: M3 = [
    [0.7977666449006423, 0.13518129740053308, 0.0313477341283922],
    [0.2880748288194013, 0.711835234241873, 0.00008993693872564],
    [0.0, 0.0, 0.8251046025104602],
];
/// D50 white, as CSS Color 4 states it.
const D50: [f64; 3] = [0.3457 / 0.3585, 1.0, (1.0 - 0.3457 - 0.3585) / 0.3585];
/// OKLab to cube-root LMS, then LMS to XYZ D65, from CSS Color 4's sample code.
const LMS_CBRT_FROM_OKLAB: M3 = [
    [1.0, 0.3963377773761749, 0.2158037573099136],
    [1.0, -0.1055613458156586, -0.0638541728258133],
    [1.0, -0.0894841775298119, -1.2914855480194092],
];
const XYZ_FROM_LMS: M3 = [
    [1.2268798758459243, -0.5578149944602171, 0.2813910456659647],
    [-0.0405757452148619, 1.112286803280317, -0.0717110580655164],
    [-0.0763729366746601, -0.4214933324022432, 1.5869240198367816],
];

fn mul(m: &M3, v: [f64; 3]) -> [f64; 3] {
    core::array::from_fn(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

/// `f32` RGB or RGBA with straight alpha.
fn float_color(
    transfer: TransferFunction,
    primaries: ColorPrimaries,
    rgb: [f64; 3],
    alpha: f64,
) -> Result<Color, CssColorError> {
    let [r, g, b] = rgb.map(|v| v as f32);
    let opaque = alpha >= 1.0;
    let descriptor = PixelDescriptor::new(
        ChannelType::F32,
        if opaque {
            ChannelLayout::Rgb
        } else {
            ChannelLayout::Rgba
        },
        (!opaque).then_some(AlphaMode::Straight),
        transfer,
    )
    .with_primaries(primaries);
    let pixel = [r, g, b, alpha as f32];
    let n = if opaque { 12 } else { 16 };
    Color::try_new(descriptor, &bytemuck::bytes_of(&pixel)[..n]).map_err(|_| CssColorError::Invalid)
}

fn xyz_d65_color(xyz: [f64; 3], alpha: f64) -> Result<Color, CssColorError> {
    float_color(
        TransferFunction::Linear,
        ColorPrimaries::Bt2020,
        mul(&XYZ_TO_BT2020, xyz),
        alpha,
    )
}

fn u8_color(r: u8, g: u8, b: u8, a: u8) -> Result<Color, CssColorError> {
    if a == 255 {
        return Ok(Color::srgb8(r, g, b));
    }
    Ok(Color::srgba8(r, g, b, a))
}

/// One argument of a color function.
#[derive(Clone, Copy)]
enum Arg {
    Number(f64),
    Percent(f64),
    /// Degrees.
    Angle(f64),
    None,
}

impl Arg {
    /// With a percentage's 100% at `full`. `none` is 0.
    fn value(self, full: f64) -> Result<f64, CssColorError> {
        match self {
            Self::Number(v) => Ok(v),
            Self::Percent(v) => Ok(v / 100.0 * full),
            Self::None => Ok(0.0),
            Self::Angle(_) => Err(CssColorError::Invalid),
        }
    }

    /// A hue in degrees. A bare number is degrees, and percentages aren't hues.
    fn hue(self) -> Result<f64, CssColorError> {
        match self {
            Self::Number(v) | Self::Angle(v) => Ok(v),
            Self::None => Ok(0.0),
            Self::Percent(_) => Err(CssColorError::Invalid),
        }
    }
}

fn parse_arg(token: &str) -> Result<Arg, CssColorError> {
    if token == "none" {
        return Ok(Arg::None);
    }
    let number = |text: &str| -> Result<f64, CssColorError> {
        // Rust's float syntax also takes `inf` and `nan`, which CSS doesn't.
        let digits = text.bytes().any(|c| c.is_ascii_digit());
        let letters = text.bytes().all(|c| !c.is_ascii_alphabetic() || c == b'e');
        if !digits || !letters {
            return Err(CssColorError::Invalid);
        }
        let value: f64 = text.parse().map_err(|_| CssColorError::Invalid)?;
        value
            .is_finite()
            .then_some(value)
            .ok_or(CssColorError::Invalid)
    };
    if let Some(text) = token.strip_suffix('%') {
        return Ok(Arg::Percent(number(text)?));
    }
    for (unit, to_degrees) in [
        ("grad", 0.9),
        ("turn", 360.0),
        ("rad", 180.0 / core::f64::consts::PI),
        ("deg", 1.0),
    ] {
        if let Some(text) = token.strip_suffix(unit) {
            return Ok(Arg::Angle(number(text)? * to_degrees));
        }
    }
    Ok(Arg::Number(number(token)?))
}

/// Components, and alpha clamped to `0..=1`, from the text inside the parentheses.
fn parse_args(
    inner: &str,
    count: usize,
    legacy_commas: bool,
) -> Result<([Arg; 4], f64), CssColorError> {
    let (components, alpha_text) = if inner.contains(',') {
        if !legacy_commas || inner.contains('/') {
            return Err(CssColorError::Invalid);
        }
        (inner, None)
    } else {
        match inner.split_once('/') {
            Some((c, a)) => (c, Some(a.trim())),
            None => (inner, None),
        }
    };
    let mut args = [Arg::None; 4];
    let mut n = 0;
    let mut alpha_arg = None;
    let separators: &[char] = if inner.contains(',') {
        &[',']
    } else {
        &[' ', '\t', '\n']
    };
    for token in components
        .split(separators)
        .map(str::trim)
        .filter(|t| !t.is_empty())
    {
        if n == count {
            // Only the comma form puts alpha in the list.
            if !inner.contains(',') || alpha_arg.is_some() {
                return Err(CssColorError::Invalid);
            }
            alpha_arg = Some(parse_arg(token)?);
            continue;
        }
        args[n] = parse_arg(token)?;
        n += 1;
    }
    if n != count {
        return Err(CssColorError::Invalid);
    }
    if inner.contains(',') && args[..count].iter().any(|a| matches!(a, Arg::None)) {
        return Err(CssColorError::Invalid);
    }
    if let Some(text) = alpha_text {
        alpha_arg = Some(parse_arg(text)?);
    }
    let alpha = match alpha_arg {
        Some(a) => a.value(1.0)?.clamp(0.0, 1.0),
        None => 1.0,
    };
    Ok((args, alpha))
}

fn parse_hex(hex: &str) -> Result<Color, CssColorError> {
    if !hex.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(CssColorError::Invalid);
    }
    let digit = |i: usize| -> u8 { (hex.as_bytes()[i] as char).to_digit(16).unwrap_or(0) as u8 };
    let short = |i: usize| digit(i) * 17;
    let long = |i: usize| digit(i) * 16 + digit(i + 1);
    match hex.len() {
        3 => u8_color(short(0), short(1), short(2), 255),
        4 => u8_color(short(0), short(1), short(2), short(3)),
        6 => u8_color(long(0), long(2), long(4), 255),
        8 => u8_color(long(0), long(2), long(4), long(6)),
        _ => Err(CssColorError::Invalid),
    }
}

fn parse_rgb(inner: &str) -> Result<Color, CssColorError> {
    let (args, alpha) = parse_args(inner, 3, true)?;
    // Numbers run to 255. Values outside the range clamp at parse time.
    let channels: [f64; 3] = [
        args[0].value(255.0)?.clamp(0.0, 255.0),
        args[1].value(255.0)?.clamp(0.0, 255.0),
        args[2].value(255.0)?.clamp(0.0, 255.0),
    ];
    let whole = channels.iter().all(|&v| libm::trunc(v) == v);
    let alpha8 = alpha * 255.0;
    if whole && libm::trunc(alpha8) == alpha8 {
        let [r, g, b] = channels.map(|v| v as u8);
        return u8_color(r, g, b, alpha8 as u8);
    }
    float_color(
        TransferFunction::Srgb,
        ColorPrimaries::Bt709,
        channels.map(|v| v / 255.0),
        alpha,
    )
}

fn parse_hsl(inner: &str) -> Result<Color, CssColorError> {
    let (args, alpha) = parse_args(inner, 3, true)?;
    let h = wrap(args[0].hue()?, 360.0);
    let s = (args[1].value(100.0)? / 100.0).max(0.0);
    let l = args[2].value(100.0)? / 100.0;
    let a = s * l.min(1.0 - l);
    let f = |n: f64| {
        let k = wrap(n + h / 30.0, 12.0);
        l - a * (k - 3.0).min(9.0 - k).clamp(-1.0, 1.0)
    };
    float_color(
        TransferFunction::Srgb,
        ColorPrimaries::Bt709,
        [f(0.0), f(8.0), f(4.0)],
        alpha,
    )
}

fn parse_hwb(inner: &str) -> Result<Color, CssColorError> {
    let (args, alpha) = parse_args(inner, 3, false)?;
    let h = wrap(args[0].hue()?, 360.0);
    let w = args[1].value(100.0)? / 100.0;
    let b = args[2].value(100.0)? / 100.0;
    let rgb = if w + b >= 1.0 {
        [w / (w + b); 3]
    } else {
        let f = |n: f64| {
            let k = wrap(n + h / 30.0, 12.0);
            let pure = 0.5 - 0.5 * (k - 3.0).min(9.0 - k).clamp(-1.0, 1.0);
            pure * (1.0 - w - b) + w
        };
        [f(0.0), f(8.0), f(4.0)]
    };
    float_color(TransferFunction::Srgb, ColorPrimaries::Bt709, rgb, alpha)
}

fn lab_to_xyz_d65(l: f64, a: f64, b: f64) -> [f64; 3] {
    const KAPPA: f64 = 24389.0 / 27.0;
    const EPSILON: f64 = 216.0 / 24389.0;
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;
    let cube = |f: f64| f * f * f;
    let x = if cube(fx) > EPSILON {
        cube(fx)
    } else {
        (116.0 * fx - 16.0) / KAPPA
    };
    let y = if l > KAPPA * EPSILON {
        cube(fy)
    } else {
        l / KAPPA
    };
    let z = if cube(fz) > EPSILON {
        cube(fz)
    } else {
        (116.0 * fz - 16.0) / KAPPA
    };
    mul(&D50_TO_D65, [x * D50[0], y * D50[1], z * D50[2]])
}

fn oklab_to_xyz_d65(l: f64, a: f64, b: f64) -> [f64; 3] {
    let lms = mul(&LMS_CBRT_FROM_OKLAB, [l, a, b]).map(|v| v * v * v);
    mul(&XYZ_FROM_LMS, lms)
}

fn polar(c: f64, hue: f64) -> (f64, f64) {
    let radians = hue * core::f64::consts::PI / 180.0;
    (c * libm::cos(radians), c * libm::sin(radians))
}

fn parse_lab(name: &str, inner: &str) -> Result<Color, CssColorError> {
    let (args, alpha) = parse_args(inner, 3, false)?;
    let xyz = match name {
        "lab" => lab_to_xyz_d65(
            args[0].value(100.0)?.clamp(0.0, 100.0),
            args[1].value(125.0)?,
            args[2].value(125.0)?,
        ),
        "lch" => {
            let (a, b) = polar(args[1].value(150.0)?.max(0.0), args[2].hue()?);
            lab_to_xyz_d65(args[0].value(100.0)?.clamp(0.0, 100.0), a, b)
        }
        "oklab" => oklab_to_xyz_d65(
            args[0].value(1.0)?.clamp(0.0, 1.0),
            args[1].value(0.4)?,
            args[2].value(0.4)?,
        ),
        _ => {
            let (a, b) = polar(args[1].value(0.4)?.max(0.0), args[2].hue()?);
            oklab_to_xyz_d65(args[0].value(1.0)?.clamp(0.0, 1.0), a, b)
        }
    };
    xyz_d65_color(xyz, alpha)
}

/// `v` into `0..modulus`, with `libm` for `no_std`.
fn wrap(v: f64, modulus: f64) -> f64 {
    let r = libm::fmod(v, modulus);
    if r < 0.0 { r + modulus } else { r }
}

fn signed_pow(v: f64, exponent: f64) -> f64 {
    libm::copysign(libm::pow(libm::fabs(v), exponent), v)
}

fn parse_color_function(inner: &str) -> Result<Color, CssColorError> {
    let inner = inner.trim_start();
    let split = inner
        .find(|c: char| c.is_ascii_whitespace())
        .ok_or(CssColorError::Invalid)?;
    let (space, rest) = inner.split_at(split);
    let (args, alpha) = parse_args(rest, 3, false)?;
    let v = [
        args[0].value(1.0)?,
        args[1].value(1.0)?,
        args[2].value(1.0)?,
    ];
    use {ColorPrimaries as P, TransferFunction as T};
    match space {
        "srgb" => float_color(T::Srgb, P::Bt709, v, alpha),
        "srgb-linear" => float_color(T::Linear, P::Bt709, v, alpha),
        "display-p3" => float_color(T::Srgb, P::DisplayP3, v, alpha),
        "display-p3-linear" => float_color(T::Linear, P::DisplayP3, v, alpha),
        "a98-rgb" => float_color(T::Gamma22, P::AdobeRgb, v, alpha),
        "rec2100-pq" => float_color(T::Pq, P::Bt2020, v, alpha),
        "rec2100-hlg" => float_color(T::Hlg, P::Bt2020, v, alpha),
        // CSS puts 1.0 at 203 cd/m², and our linear puts it at PQ's peak.
        "rec2100-linear" => {
            float_color(T::Linear, P::Bt2020, v.map(|c| c * 203.0 / 10_000.0), alpha)
        }
        // Gamma 2.4 has no transfer function here.
        "rec2020" => float_color(T::Linear, P::Bt2020, v.map(|c| signed_pow(c, 2.4)), alpha),
        "prophoto-rgb" => {
            let linear = v.map(|c| {
                if libm::fabs(c) <= 16.0 / 512.0 {
                    c / 16.0
                } else {
                    signed_pow(c, 1.8)
                }
            });
            xyz_d65_color(mul(&D50_TO_D65, mul(&PROPHOTO_TO_XYZ_D50, linear)), alpha)
        }
        "xyz" | "xyz-d65" => xyz_d65_color(v, alpha),
        "xyz-d50" => xyz_d65_color(mul(&D50_TO_D65, v), alpha),
        _ => Err(CssColorError::Invalid),
    }
}

pub(crate) fn parse(css: &str) -> Result<Color, CssColorError> {
    let text = css.trim().to_ascii_lowercase();
    if let Some(hex) = text.strip_prefix('#') {
        return parse_hex(hex);
    }
    if let Some(open) = text.find('(') {
        let inner = text[open + 1..]
            .strip_suffix(')')
            .ok_or(CssColorError::Invalid)?;
        let name = text[..open].trim_end();
        return match name {
            "rgb" | "rgba" => parse_rgb(inner),
            "hsl" | "hsla" => parse_hsl(inner),
            "hwb" => parse_hwb(inner),
            "lab" | "lch" | "oklab" | "oklch" => parse_lab(name, inner),
            "color" => parse_color_function(inner),
            "light-dark" | "color-mix" | "contrast-color" | "device-cmyk" | "calc" | "ictcp"
            | "jzazbz" | "jzczhz" => Err(CssColorError::Unsupported),
            _ => Err(CssColorError::Invalid),
        };
    }
    match text.as_str() {
        "transparent" => u8_color(0, 0, 0, 0),
        "currentcolor" | "canvas" | "canvastext" => Err(CssColorError::Unsupported),
        name => named(name)
            .map(|[r, g, b]| Color::srgb8(r, g, b))
            .ok_or(CssColorError::Invalid),
    }
}

fn named(name: &str) -> Option<[u8; 3]> {
    NAMED
        .binary_search_by_key(&name, |entry| entry.0)
        .ok()
        .map(|i| NAMED[i].1)
}

/// CSS named colors, sorted by name for the binary search.
#[rustfmt::skip]
const NAMED: &[(&str, [u8; 3])] = &[
    ("aliceblue", [240, 248, 255]),
    ("antiquewhite", [250, 235, 215]),
    ("aqua", [0, 255, 255]),
    ("aquamarine", [127, 255, 212]),
    ("azure", [240, 255, 255]),
    ("beige", [245, 245, 220]),
    ("bisque", [255, 228, 196]),
    ("black", [0, 0, 0]),
    ("blanchedalmond", [255, 235, 205]),
    ("blue", [0, 0, 255]),
    ("blueviolet", [138, 43, 226]),
    ("brown", [165, 42, 42]),
    ("burlywood", [222, 184, 135]),
    ("cadetblue", [95, 158, 160]),
    ("chartreuse", [127, 255, 0]),
    ("chocolate", [210, 105, 30]),
    ("coral", [255, 127, 80]),
    ("cornflowerblue", [100, 149, 237]),
    ("cornsilk", [255, 248, 220]),
    ("crimson", [220, 20, 60]),
    ("cyan", [0, 255, 255]),
    ("darkblue", [0, 0, 139]),
    ("darkcyan", [0, 139, 139]),
    ("darkgoldenrod", [184, 134, 11]),
    ("darkgray", [169, 169, 169]),
    ("darkgreen", [0, 100, 0]),
    ("darkgrey", [169, 169, 169]),
    ("darkkhaki", [189, 183, 107]),
    ("darkmagenta", [139, 0, 139]),
    ("darkolivegreen", [85, 107, 47]),
    ("darkorange", [255, 140, 0]),
    ("darkorchid", [153, 50, 204]),
    ("darkred", [139, 0, 0]),
    ("darksalmon", [233, 150, 122]),
    ("darkseagreen", [143, 188, 143]),
    ("darkslateblue", [72, 61, 139]),
    ("darkslategray", [47, 79, 79]),
    ("darkslategrey", [47, 79, 79]),
    ("darkturquoise", [0, 206, 209]),
    ("darkviolet", [148, 0, 211]),
    ("deeppink", [255, 20, 147]),
    ("deepskyblue", [0, 191, 255]),
    ("dimgray", [105, 105, 105]),
    ("dimgrey", [105, 105, 105]),
    ("dodgerblue", [30, 144, 255]),
    ("firebrick", [178, 34, 34]),
    ("floralwhite", [255, 250, 240]),
    ("forestgreen", [34, 139, 34]),
    ("fuchsia", [255, 0, 255]),
    ("gainsboro", [220, 220, 220]),
    ("ghostwhite", [248, 248, 255]),
    ("gold", [255, 215, 0]),
    ("goldenrod", [218, 165, 32]),
    ("gray", [128, 128, 128]),
    ("green", [0, 128, 0]),
    ("greenyellow", [173, 255, 47]),
    ("grey", [128, 128, 128]),
    ("honeydew", [240, 255, 240]),
    ("hotpink", [255, 105, 180]),
    ("indianred", [205, 92, 92]),
    ("indigo", [75, 0, 130]),
    ("ivory", [255, 255, 240]),
    ("khaki", [240, 230, 140]),
    ("lavender", [230, 230, 250]),
    ("lavenderblush", [255, 240, 245]),
    ("lawngreen", [124, 252, 0]),
    ("lemonchiffon", [255, 250, 205]),
    ("lightblue", [173, 216, 230]),
    ("lightcoral", [240, 128, 128]),
    ("lightcyan", [224, 255, 255]),
    ("lightgoldenrodyellow", [250, 250, 210]),
    ("lightgray", [211, 211, 211]),
    ("lightgreen", [144, 238, 144]),
    ("lightgrey", [211, 211, 211]),
    ("lightpink", [255, 182, 193]),
    ("lightsalmon", [255, 160, 122]),
    ("lightseagreen", [32, 178, 170]),
    ("lightskyblue", [135, 206, 250]),
    ("lightslategray", [119, 136, 153]),
    ("lightslategrey", [119, 136, 153]),
    ("lightsteelblue", [176, 196, 222]),
    ("lightyellow", [255, 255, 224]),
    ("lime", [0, 255, 0]),
    ("limegreen", [50, 205, 50]),
    ("linen", [250, 240, 230]),
    ("magenta", [255, 0, 255]),
    ("maroon", [128, 0, 0]),
    ("mediumaquamarine", [102, 205, 170]),
    ("mediumblue", [0, 0, 205]),
    ("mediumorchid", [186, 85, 211]),
    ("mediumpurple", [147, 112, 219]),
    ("mediumseagreen", [60, 179, 113]),
    ("mediumslateblue", [123, 104, 238]),
    ("mediumspringgreen", [0, 250, 154]),
    ("mediumturquoise", [72, 209, 204]),
    ("mediumvioletred", [199, 21, 133]),
    ("midnightblue", [25, 25, 112]),
    ("mintcream", [245, 255, 250]),
    ("mistyrose", [255, 228, 225]),
    ("moccasin", [255, 228, 181]),
    ("navajowhite", [255, 222, 173]),
    ("navy", [0, 0, 128]),
    ("oldlace", [253, 245, 230]),
    ("olive", [128, 128, 0]),
    ("olivedrab", [107, 142, 35]),
    ("orange", [255, 165, 0]),
    ("orangered", [255, 69, 0]),
    ("orchid", [218, 112, 214]),
    ("palegoldenrod", [238, 232, 170]),
    ("palegreen", [152, 251, 152]),
    ("paleturquoise", [175, 238, 238]),
    ("palevioletred", [219, 112, 147]),
    ("papayawhip", [255, 239, 213]),
    ("peachpuff", [255, 218, 185]),
    ("peru", [205, 133, 63]),
    ("pink", [255, 192, 203]),
    ("plum", [221, 160, 221]),
    ("powderblue", [176, 224, 230]),
    ("purple", [128, 0, 128]),
    ("rebeccapurple", [102, 51, 153]),
    ("red", [255, 0, 0]),
    ("rosybrown", [188, 143, 143]),
    ("royalblue", [65, 105, 225]),
    ("saddlebrown", [139, 69, 19]),
    ("salmon", [250, 128, 114]),
    ("sandybrown", [244, 164, 96]),
    ("seagreen", [46, 139, 87]),
    ("seashell", [255, 245, 238]),
    ("sienna", [160, 82, 45]),
    ("silver", [192, 192, 192]),
    ("skyblue", [135, 206, 235]),
    ("slateblue", [106, 90, 205]),
    ("slategray", [112, 128, 144]),
    ("slategrey", [112, 128, 144]),
    ("snow", [255, 250, 250]),
    ("springgreen", [0, 255, 127]),
    ("steelblue", [70, 130, 180]),
    ("tan", [210, 180, 140]),
    ("teal", [0, 128, 128]),
    ("thistle", [216, 191, 216]),
    ("tomato", [255, 99, 71]),
    ("turquoise", [64, 224, 208]),
    ("violet", [238, 130, 238]),
    ("wheat", [245, 222, 179]),
    ("white", [255, 255, 255]),
    ("whitesmoke", [245, 245, 245]),
    ("yellow", [255, 255, 0]),
    ("yellowgreen", [154, 205, 50]),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_name_table_is_sorted() {
        assert!(NAMED.windows(2).all(|w| w[0].0 < w[1].0));
    }
}
