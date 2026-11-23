#![allow(clippy::excessive_precision)]
//! Closed-form perceptual & cylindrical color conversions (`Oklch`, `CieLab`,
//! `CieLch`, `Jzazbz`, `Jzczhz`, `Hsl`, `Hsv`).

use core::f32::consts::PI;
use zenpixels::{PerceptualSpace, Primaries};

use crate::gamut::GamutMatrix;
use crate::oklab::OklabSpace;

const RAD_TO_DEG: f32 = 180.0 / PI;
const DEG_TO_RAD: f32 = PI / 180.0;

// D65 reference white point in CIE XYZ (Y = 1.0).
const D65_X: f32 = 0.95047;
const D65_Y: f32 = 1.00000;
const D65_Z: f32 = 1.08883;

/// Convert linear RGB (in `primaries`) to a [`PerceptualSpace`] triple.
#[must_use]
pub fn linear_rgb_to_perceptual(
    rgb: [f32; 3],
    primaries: Primaries,
    space: PerceptualSpace,
    white_nits: f32,
) -> [f32; 3] {
    match space {
        PerceptualSpace::Oklch => {
            let ok = OklabSpace::for_primaries(primaries)
                .unwrap_or_else(|| OklabSpace::for_primaries(Primaries::Bt709).unwrap())
                .from_rgb(rgb);
            cartesian_to_polar(ok)
        }
        PerceptualSpace::CieLab => linear_rgb_to_cielab(rgb, primaries),
        PerceptualSpace::CieLch => cartesian_to_polar(linear_rgb_to_cielab(rgb, primaries)),
        PerceptualSpace::Jzazbz => linear_rgb_to_jzazbz(rgb, primaries, white_nits),
        PerceptualSpace::Jzczhz => {
            cartesian_to_polar(linear_rgb_to_jzazbz(rgb, primaries, white_nits))
        }
        PerceptualSpace::Hsl => {
            let srgb = [
                linear_srgb::precise::linear_to_srgb_extended(rgb[0]),
                linear_srgb::precise::linear_to_srgb_extended(rgb[1]),
                linear_srgb::precise::linear_to_srgb_extended(rgb[2]),
            ];
            rgb_to_hsl(srgb)
        }
        PerceptualSpace::Hsv => {
            let srgb = [
                linear_srgb::precise::linear_to_srgb_extended(rgb[0]),
                linear_srgb::precise::linear_to_srgb_extended(rgb[1]),
                linear_srgb::precise::linear_to_srgb_extended(rgb[2]),
            ];
            rgb_to_hsv(srgb)
        }
        _ => rgb,
    }
}

/// Convert a [`PerceptualSpace`] triple back to linear RGB (in `primaries`).
#[must_use]
pub fn perceptual_to_linear_rgb(
    v: [f32; 3],
    primaries: Primaries,
    space: PerceptualSpace,
    white_nits: f32,
) -> [f32; 3] {
    match space {
        PerceptualSpace::Oklch => {
            let lab = polar_to_cartesian(v);
            OklabSpace::for_primaries(primaries)
                .unwrap_or_else(|| OklabSpace::for_primaries(Primaries::Bt709).unwrap())
                .to_rgb(lab)
        }
        PerceptualSpace::CieLab => cielab_to_linear_rgb(v, primaries),
        PerceptualSpace::CieLch => cielab_to_linear_rgb(polar_to_cartesian(v), primaries),
        PerceptualSpace::Jzazbz => jzazbz_to_linear_rgb(v, primaries, white_nits),
        PerceptualSpace::Jzczhz => {
            jzazbz_to_linear_rgb(polar_to_cartesian(v), primaries, white_nits)
        }
        PerceptualSpace::Hsl => {
            let srgb = hsl_to_rgb(v);
            [
                linear_srgb::precise::srgb_to_linear_extended(srgb[0]),
                linear_srgb::precise::srgb_to_linear_extended(srgb[1]),
                linear_srgb::precise::srgb_to_linear_extended(srgb[2]),
            ]
        }
        PerceptualSpace::Hsv => {
            let srgb = hsv_to_rgb(v);
            [
                linear_srgb::precise::srgb_to_linear_extended(srgb[0]),
                linear_srgb::precise::srgb_to_linear_extended(srgb[1]),
                linear_srgb::precise::srgb_to_linear_extended(srgb[2]),
            ]
        }
        _ => v,
    }
}

#[inline]
pub fn cartesian_to_polar([l, a, b]: [f32; 3]) -> [f32; 3] {
    let c = libm::hypotf(a, b);
    let mut h = libm::atan2f(b, a) * RAD_TO_DEG;
    if h < 0.0 {
        h += 360.0;
    }
    [l, c, h]
}

#[inline]
pub fn polar_to_cartesian([l, c, h]: [f32; 3]) -> [f32; 3] {
    let rad = h * DEG_TO_RAD;
    [l, c * libm::cosf(rad), c * libm::sinf(rad)]
}

fn lab_f(t: f32) -> f32 {
    const EPSILON: f32 = 216.0 / 24389.0;
    const KAPPA: f32 = 24389.0 / 27.0;
    if t > EPSILON {
        libm::cbrtf(t)
    } else {
        (KAPPA * t + 16.0) / 116.0
    }
}

fn lab_f_inv(f: f32) -> f32 {
    const DELTA: f32 = 6.0 / 29.0;
    const KAPPA: f32 = 24389.0 / 27.0;
    if f > DELTA {
        f * f * f
    } else {
        (116.0 * f - 16.0) / KAPPA
    }
}

fn linear_rgb_to_cielab(rgb: [f32; 3], primaries: Primaries) -> [f32; 3] {
    let to_xyz = GamutMatrix::to_xyz(primaries)
        .unwrap_or_else(|| GamutMatrix::to_xyz(Primaries::Bt709).unwrap());
    let [x, y, z] = to_xyz.apply(rgb);
    let fx = lab_f(x / D65_X);
    let fy = lab_f(y / D65_Y);
    let fz = lab_f(z / D65_Z);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

fn cielab_to_linear_rgb([l, a, b]: [f32; 3], primaries: Primaries) -> [f32; 3] {
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;
    let xyz = [
        lab_f_inv(fx) * D65_X,
        lab_f_inv(fy) * D65_Y,
        lab_f_inv(fz) * D65_Z,
    ];
    let from_xyz = GamutMatrix::from_xyz(primaries)
        .unwrap_or_else(|| GamutMatrix::from_xyz(Primaries::Bt709).unwrap());
    from_xyz.apply(xyz)
}

// Safdar et al. (2017) Jzazbz constants.
const JZ_B: f64 = 1.15;
const JZ_G: f64 = 0.66;
const JZ_C1: f64 = 3424.0 / 4096.0;
const JZ_C2: f64 = 2413.0 / 128.0;
const JZ_C3: f64 = 2392.0 / 128.0;
const JZ_N: f64 = 2610.0 / 16384.0;
const JZ_P: f64 = 1.7 * 2523.0 / 32.0;
const JZ_D: f64 = -0.56;
const JZ_D0: f64 = 1.6295499532821566e-11;

fn jz_pq_encode(v: f64) -> f64 {
    let vn = libm::pow((v / 10_000.0).max(0.0), JZ_N);
    libm::pow((JZ_C1 + JZ_C2 * vn) / (1.0 + JZ_C3 * vn), JZ_P)
}

fn jz_pq_decode(v: f64) -> f64 {
    let vp = libm::pow(v.max(0.0), 1.0 / JZ_P);
    let num = (vp - JZ_C1).max(0.0);
    let den = (JZ_C2 - JZ_C3 * vp).max(1e-30);
    10_000.0 * libm::pow(num / den, 1.0 / JZ_N)
}

fn linear_rgb_to_jzazbz(rgb: [f32; 3], primaries: Primaries, white_nits: f32) -> [f32; 3] {
    let to_xyz = GamutMatrix::to_xyz(primaries)
        .unwrap_or_else(|| GamutMatrix::to_xyz(Primaries::Bt709).unwrap());
    let [x0, y0, z0] = to_xyz.apply(rgb);
    let s = f64::from(white_nits);
    let (x, y, z) = (f64::from(x0) * s, f64::from(y0) * s, f64::from(z0) * s);
    let xp = JZ_B * x - (JZ_B - 1.0) * z;
    let yp = JZ_G * y - (JZ_G - 1.0) * x;
    let l = 0.41478972 * xp + 0.579999 * yp + 0.0146480 * z;
    let m = -0.2015100 * xp + 1.120649 * yp + 0.0531008 * z;
    let sc = -0.0166008 * xp + 0.264800 * yp + 0.6684799 * z;
    let lp = jz_pq_encode(l);
    let mp = jz_pq_encode(m);
    let sp = jz_pq_encode(sc);
    let iz = 0.5 * (lp + mp);
    let az = 3.524000 * lp - 4.066708 * mp + 0.542708 * sp;
    let bz = 0.199076 * lp + 1.096799 * mp - 1.295875 * sp;
    let jz = ((1.0 + JZ_D) * iz) / (1.0 + JZ_D * iz) - JZ_D0;
    [jz as f32, az as f32, bz as f32]
}

fn jzazbz_to_linear_rgb([jz, az, bz]: [f32; 3], primaries: Primaries, white_nits: f32) -> [f32; 3] {
    let jz64 = f64::from(jz) + JZ_D0;
    let iz = jz64 / (1.0 + JZ_D - JZ_D * jz64);
    let a64 = f64::from(az);
    let b64 = f64::from(bz);
    let lp = iz + 0.1386050432715393 * a64 + 0.0580473161561189 * b64;
    let mp = iz - 0.1386050432715393 * a64 - 0.0580473161561189 * b64;
    let sp = iz - 0.0960192420263190 * a64 - 0.8118918960560390 * b64;
    let l = jz_pq_decode(lp);
    let m = jz_pq_decode(mp);
    let s = jz_pq_decode(sp);
    let xp = 1.9242264357876067 * l - 1.0047923125953657 * m + 0.0376514040306180 * s;
    let yp = 0.3503167620949991 * l + 0.7264811939316552 * m - 0.0653844229480850 * s;
    let z = -0.0909828109828475 * l - 0.3127282905230739 * m + 1.5227665613052603 * s;
    let x = (xp + (JZ_B - 1.0) * z) / JZ_B;
    let y = (yp + (JZ_G - 1.0) * x) / JZ_G;
    let inv = 1.0 / f64::from(white_nits).max(1e-6);
    let from_xyz = GamutMatrix::from_xyz(primaries)
        .unwrap_or_else(|| GamutMatrix::from_xyz(Primaries::Bt709).unwrap());
    from_xyz.apply([(x * inv) as f32, (y * inv) as f32, (z * inv) as f32])
}

fn rgb_to_hsl([r, g, b]: [f32; 3]) -> [f32; 3] {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = 0.5 * (max + min);
    let d = max - min;
    if d <= 1e-7 {
        return [0.0, 0.0, l];
    }
    let s = if l > 0.5 {
        d / (2.0 - max - min).max(1e-7)
    } else {
        d / (max + min).max(1e-7)
    };
    let mut h = if max == r {
        (g - b) / d + if g < b { 6.0 } else { 0.0 }
    } else if max == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    h *= 60.0;
    [h, s, l]
}

fn hsl_to_rgb([h, s, l]: [f32; 3]) -> [f32; 3] {
    if s <= 1e-7 {
        return [l, l, l];
    }
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let hk = ((h % 360.0) + 360.0) % 360.0 / 360.0;
    let hue = |mut t: f32| -> f32 {
        if t < 0.0 {
            t += 1.0;
        }
        if t > 1.0 {
            t -= 1.0;
        }
        if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        }
    };
    [hue(hk + 1.0 / 3.0), hue(hk), hue(hk - 1.0 / 3.0)]
}

fn rgb_to_hsv([r, g, b]: [f32; 3]) -> [f32; 3] {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let s = if max <= 1e-7 { 0.0 } else { d / max };
    if d <= 1e-7 {
        return [0.0, s, max];
    }
    let mut h = if max == r {
        (g - b) / d + if g < b { 6.0 } else { 0.0 }
    } else if max == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    h *= 60.0;
    [h, s, max]
}

fn hsv_to_rgb([h, s, v]: [f32; 3]) -> [f32; 3] {
    if s <= 1e-7 {
        return [v, v, v];
    }
    let hh = (((h % 360.0) + 360.0) % 360.0) / 60.0;
    let i = hh as i32;
    let f = hh - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match i % 6 {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}
