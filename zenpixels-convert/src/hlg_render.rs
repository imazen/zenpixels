//! BT.2100-3 Table 5 display rendering, resolved by the plan.
//! OOTF structure follows zentone f32e091c's hlg module (same ownership).
//! F64 evaluation and analytic curves avoid a second approximation error budget.
use crate::ChannelLayout;
use zenpixels::hdr::HlgRendering;

fn decode(v: f64) -> f64 {
    let a = 0.17883277;
    let b = 1.0 - 4.0 * a;
    let c = 0.5 - a * libm::log(4.0 * a);
    if v.abs() <= 0.5 {
        v * v.abs() / 3.0
    } else {
        v.signum() * (libm::exp((v.abs() - c) / a) + b) / 12.0
    }
}
fn encode(v: f64) -> f64 {
    let a = 0.17883277;
    let b = 1.0 - 4.0 * a;
    let c = 0.5 - a * libm::log(4.0 * a);
    if v.abs() <= 1.0 / 12.0 {
        v.signum() * libm::sqrt(3.0 * v.abs())
    } else {
        v.signum() * (a * libm::log(12.0 * v.abs() - b) + c)
    }
}
// Extended RGB uses signed transfer curves and |Y| in the common luminance
// multiplier. Zero luminance maps to zero; nonfinite RGB maps to RGB NaN.
// Alpha is copied bit-for-bit. No per-row reduction or allocation occurs.
#[derive(Clone, Debug)]
pub(crate) struct PreparedHlg {
    gamma: f64,
    beta: f64,
    scale: f64,
    pub(super) inverse: bool,
    clip: bool,
}
impl PreparedHlg {
    pub(super) fn new(rendering: HlgRendering, inverse: bool, white: f32, clip: bool) -> Self {
        let gamma = f64::from(rendering.system_gamma());
        let peak = f64::from(rendering.peak().nits());
        let beta =
            libm::sqrt(3.0 * libm::pow(f64::from(rendering.black_nits()) / peak, 1.0 / gamma));
        Self {
            gamma,
            beta,
            scale: peak / f64::from(white),
            inverse,
            clip,
        }
    }
}
pub(super) fn row(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    layout: ChannelLayout,
    prepared: &PreparedHlg,
) {
    let channels = if layout == ChannelLayout::Rgba { 4 } else { 3 };
    let PreparedHlg {
        gamma,
        beta,
        scale,
        inverse,
        clip,
    } = *prepared;
    for (s, d) in src[..width * channels * 4]
        .chunks_exact(channels * 4)
        .zip(dst[..width * channels * 4].chunks_exact_mut(channels * 4))
    {
        d.copy_from_slice(s);
        let mut rgb = [0.0; 3];
        for (i, v) in rgb.iter_mut().enumerate() {
            *v = f64::from(f32::from_ne_bytes(s[i * 4..i * 4 + 4].try_into().unwrap()));
        }
        if rgb.iter().any(|v| !v.is_finite()) {
            rgb = [f64::NAN; 3];
        } else {
            if inverse {
                for v in &mut rgb {
                    *v /= scale;
                }
                let y = 0.2627 * rgb[0] + 0.6780 * rgb[1] + 0.0593 * rgb[2];
                let gain = if y == 0.0 {
                    0.0
                } else {
                    libm::pow(y.abs(), (1.0 - gamma) / gamma)
                };
                for v in &mut rgb {
                    *v = (encode(*v * gain) - beta) / (1.0 - beta);
                    if clip {
                        *v = v.clamp(0.0, 1.0);
                    }
                }
            } else {
                for v in &mut rgb {
                    if clip {
                        *v = v.clamp(0.0, 1.0);
                    }
                    *v = decode((1.0 - beta) * *v + beta);
                }
                let y = 0.2627 * rgb[0] + 0.6780 * rgb[1] + 0.0593 * rgb[2];
                let gain = if y == 0.0 {
                    0.0
                } else {
                    scale * libm::pow(y.abs(), gamma - 1.0)
                };
                for v in &mut rgb {
                    *v *= gain;
                }
            }
        }
        for (i, v) in rgb.iter().enumerate() {
            d[i * 4..i * 4 + 4].copy_from_slice(&(*v as f32).to_ne_bytes());
        }
    }
}
