// Independent ICC TRC evaluator and H.273 reference used by read-back tests
// and the profile-candidate diagnostic. Normalized [0,1] inputs only.
fn u32be(b: &[u8]) -> usize {
    u32::from_be_bytes(b[..4].try_into().unwrap()) as usize
}
fn fixed(b: &[u8]) -> f64 {
    f64::from(i32::from_be_bytes(b[..4].try_into().unwrap())) / 65536.0
}
pub(super) fn curve(trc: &[u8], x: f64) -> f64 {
    match &trc[..4] {
        b"para" => {
            let kind = u16::from_be_bytes(trc[8..10].try_into().unwrap());
            let v: Vec<_> = (0..(trc.len() - 12) / 4)
                .map(|i| fixed(&trc[12 + 4 * i..]))
                .collect();
            match kind {
                0 => x.powf(v[0]),
                1 => {
                    if x >= -v[2] / v[1] {
                        (v[1] * x + v[2]).powf(v[0])
                    } else {
                        0.0
                    }
                }
                2 => {
                    if x >= -v[2] / v[1] {
                        (v[1] * x + v[2]).powf(v[0]) + v[3]
                    } else {
                        v[3]
                    }
                }
                3 => {
                    if x >= v[4] {
                        (v[1] * x + v[2]).powf(v[0])
                    } else {
                        v[3] * x
                    }
                }
                4 => {
                    if x >= v[4] {
                        (v[1] * x + v[2]).powf(v[0]) + v[5]
                    } else {
                        v[3] * x + v[6]
                    }
                }
                _ => panic!("unknown para function {kind}"),
            }
        }
        b"curv" => {
            let n = u32be(&trc[8..]);
            let sample = |i: usize| {
                f64::from(u16::from_be_bytes(
                    trc[12 + 2 * i..14 + 2 * i].try_into().unwrap(),
                ))
            };
            match n {
                0 => x,
                1 => x.powf(sample(0) / 256.0),
                _ => {
                    let position = x * (n - 1) as f64;
                    let i = (position.floor() as usize).min(n - 2);
                    (sample(i) + (sample(i + 1) - sample(i)) * (position - i as f64)) / 65535.0
                }
            }
        }
        other => panic!("unsupported TRC {other:?}"),
    }
}
fn inverse_oetf(x: f64, slope: f64, alpha: f64, beta: f64, power: f64) -> f64 {
    if x < slope * beta {
        x / slope
    } else {
        ((x + alpha - 1.0) / alpha).powf(power)
    }
}
pub(super) fn analytic(t: u8, x: f64) -> f64 {
    let bt709 = || inverse_oetf(x, 4.5, 1.099296826809442, 0.018053968510807, 1.0 / 0.45);
    match t {
        1 | 6 | 11 | 12 | 14 | 15 => bt709(), // 11/12 agree on [0,1].
        4 => x.powf(2.2),
        5 => x.powf(2.8),
        7 => inverse_oetf(x, 4.0, 1.111572195921731, 0.022821585529445, 1.0 / 0.45),
        8 => x,
        9 => 10_f64.powf(2.0 * (x - 1.0)),
        10 => 10_f64.powf(2.5 * (x - 1.0)),
        13 => inverse_oetf(x, 12.92, 1.0550107189475866, 0.003_041_282_560_127_521, 2.4),
        16 => {
            let e = x.powf(32.0 / 2523.0);
            ((e - 107.0 / 128.0).max(0.0) / (2413.0 / 128.0 - 2392.0 / 128.0 * e))
                .powf(8192.0 / 1305.0)
        }
        // A u16 curv LUT saturates at 1; ST428's nominal endpoint is 52.37/48.
        17 => (x.powf(2.6) * 52.37 / 48.0).min(1.0),
        18 => {
            let scene = if x <= 0.5 {
                x * x / 3.0
            } else {
                (((x - 0.55991073) / 0.17883277).exp() + 0.28466892) / 12.0
            };
            // The shipped HLG ICC encodes BT.2100 reference-display neutral
            // EOTF (1000 cd/m², system gamma 1.2), including its OOTF.
            scene.powf(1.2)
        }
        _ => panic!("unassigned transfer {t}"),
    }
}
