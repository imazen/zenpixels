//! Small table-only profiles built in memory, so tests need no file on disk.
//!
//! It has one tag: an `A2B0` `mft2` table with 9 grid points per channel, mapping ink to Lab with a
//! smooth made-up ink model. It has no `A2B1`, which also exercises the fallback from relative
//! colorimetric to `A2B0`. No ink is exactly Lab white.

/// Ink at a node of the grid, as legacy 16-bit Lab.
fn node_lab(c: f64, m: f64, y: f64, k: f64) -> [u16; 3] {
    let lit = 1.0 - k;
    let l = 100.0 * lit * (1.0 - 0.3 * c - 0.4 * m - 0.1 * y);
    let a = 60.0 * lit * (m - c);
    let b = 70.0 * lit * (y - 0.5 * (c + m));
    [
        (l * 652.8).round() as u16,
        ((a + 128.0) * 256.0).round() as u16,
        ((b + 128.0) * 256.0).round() as u16,
    ]
}

const GRID: u32 = 9;

/// Builds the profile. Grid order is C slowest, K fastest, as `mft2` specifies.
pub fn synthetic_cmyk_icc() -> Vec<u8> {
    let mut lut = Vec::new();
    lut.extend_from_slice(b"mft2");
    lut.extend_from_slice(&[0; 4]);
    lut.extend_from_slice(&[4, 3, GRID as u8, 0]);
    for row in 0..3 {
        for column in 0..3 {
            let one: i32 = if row == column { 0x10000 } else { 0 };
            lut.extend_from_slice(&one.to_be_bytes());
        }
    }
    lut.extend_from_slice(&2u16.to_be_bytes());
    lut.extend_from_slice(&2u16.to_be_bytes());
    let ramp = [0u16, 0xFFFF];
    for _ in 0..4 {
        ramp.iter()
            .for_each(|v| lut.extend_from_slice(&v.to_be_bytes()));
    }
    for node in 0..GRID.pow(4) {
        let ink = |place: u32| f64::from(node / GRID.pow(place) % GRID) / f64::from(GRID - 1);
        for v in node_lab(ink(3), ink(2), ink(1), ink(0)) {
            lut.extend_from_slice(&v.to_be_bytes());
        }
    }
    for _ in 0..3 {
        ramp.iter()
            .for_each(|v| lut.extend_from_slice(&v.to_be_bytes()));
    }

    const TAG_START: usize = 128 + 4 + 12;
    let size = TAG_START + lut.len();
    let mut icc = vec![0u8; 128];
    icc[0..4].copy_from_slice(&(size as u32).to_be_bytes());
    icc[8..12].copy_from_slice(&0x0240_0000u32.to_be_bytes());
    icc[12..16].copy_from_slice(b"prtr");
    icc[16..20].copy_from_slice(b"CMYK");
    icc[20..24].copy_from_slice(b"Lab ");
    icc[36..40].copy_from_slice(b"acsp");
    for (i, v) in [0x0000_F6D6u32, 0x0001_0000, 0x0000_D32D]
        .iter()
        .enumerate()
    {
        icc[68 + i * 4..72 + i * 4].copy_from_slice(&v.to_be_bytes());
    }
    icc.extend_from_slice(&1u32.to_be_bytes());
    icc.extend_from_slice(b"A2B0");
    icc.extend_from_slice(&(TAG_START as u32).to_be_bytes());
    icc.extend_from_slice(&(lut.len() as u32).to_be_bytes());
    icc.extend_from_slice(&lut);
    icc
}

const fn lab_f(t: f64, cube_root: f64) -> f64 {
    if t > 216.0 / 24389.0 {
        cube_root
    } else {
        (24389.0 / 27.0 * t + 16.0) / 116.0
    }
}

/// An RGB profile with only a table: an `A2B0` `mft2` that maps sRGB values to Lab, so converting it to
/// sRGB is close to an identity.
pub fn synthetic_rgb_lut_icc() -> Vec<u8> {
    const NODES: u32 = 17;
    // sRGB's colorants, adapted to the PCS white.
    const M: [[f64; 3]; 3] = [
        [0.4360, 0.3851, 0.1431],
        [0.2225, 0.7169, 0.0606],
        [0.0139, 0.0971, 0.7139],
    ];
    const WHITE: [f64; 3] = [0.9642, 1.0, 0.8249];
    let decode = |v: f64| {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    let mut lut = Vec::new();
    lut.extend_from_slice(b"mft2");
    lut.extend_from_slice(&[0; 4]);
    lut.extend_from_slice(&[3, 3, NODES as u8, 0]);
    for row in 0..3 {
        for column in 0..3 {
            let one: i32 = if row == column { 0x10000 } else { 0 };
            lut.extend_from_slice(&one.to_be_bytes());
        }
    }
    lut.extend_from_slice(&2u16.to_be_bytes());
    lut.extend_from_slice(&2u16.to_be_bytes());
    let ramp = [0u16, 0xFFFF];
    for _ in 0..3 {
        ramp.iter()
            .for_each(|v| lut.extend_from_slice(&v.to_be_bytes()));
    }
    for node in 0..NODES.pow(3) {
        let channel =
            |place: u32| decode(f64::from(node / NODES.pow(place) % NODES) / f64::from(NODES - 1));
        let rgb = [channel(2), channel(1), channel(0)];
        let xyz: Vec<f64> = (0..3)
            .map(|i| {
                let v = M[i][0] * rgb[0] + M[i][1] * rgb[1] + M[i][2] * rgb[2];
                let t = v / WHITE[i];
                lab_f(t, t.cbrt())
            })
            .collect();
        let l = 116.0 * xyz[1] - 16.0;
        let a = 500.0 * (xyz[0] - xyz[1]);
        let b = 200.0 * (xyz[1] - xyz[2]);
        for v in [l * 652.8, (a + 128.0) * 256.0, (b + 128.0) * 256.0] {
            lut.extend_from_slice(&(v.round().clamp(0.0, 65535.0) as u16).to_be_bytes());
        }
    }
    for _ in 0..3 {
        ramp.iter()
            .for_each(|v| lut.extend_from_slice(&v.to_be_bytes()));
    }

    const TAG_START: usize = 128 + 4 + 12;
    let size = TAG_START + lut.len();
    let mut icc = vec![0u8; 128];
    icc[0..4].copy_from_slice(&(size as u32).to_be_bytes());
    icc[8..12].copy_from_slice(&0x0240_0000u32.to_be_bytes());
    icc[12..16].copy_from_slice(b"spac");
    icc[16..20].copy_from_slice(b"RGB ");
    icc[20..24].copy_from_slice(b"Lab ");
    icc[36..40].copy_from_slice(b"acsp");
    for (i, v) in [0x0000_F6D6u32, 0x0001_0000, 0x0000_D32D]
        .iter()
        .enumerate()
    {
        icc[68 + i * 4..72 + i * 4].copy_from_slice(&v.to_be_bytes());
    }
    icc.extend_from_slice(&1u32.to_be_bytes());
    icc.extend_from_slice(b"A2B0");
    icc.extend_from_slice(&(TAG_START as u32).to_be_bytes());
    icc.extend_from_slice(&(lut.len() as u32).to_be_bytes());
    icc.extend_from_slice(&lut);
    icc
}
