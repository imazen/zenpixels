//! Generate ICC hash table .inc files for zenpixels::icc.
//!
//! Scans ICC profiles, computes normalized FNV-1a hashes, identifies primaries
//! and TRC (measuring max u16 error against all reference EOTFs for all 65536
//! values), deduplicates by normalized hash, and writes Rust include files.
//!
//! Usage:
//!   rustc -O scripts/gen_icc_tables.rs -o /tmp/gen_icc_tables
//!   ICC_PROFILES_DIR=/tmp/icc-upload /tmp/gen_icc_tables
//!
//! Output goes to zenpixels/src/icc/icc_table_{rgb,gray}.inc relative to
//! the script's location, or to --out-dir if specified.

use std::collections::BTreeMap;
use std::path::PathBuf;

// ── Normalized FNV-1a hash ───────────────────────────────────────────────

fn fnv1a_64_normalized(data: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut hash = OFFSET;

    // Phase 1: first 100 bytes — zero metadata fields.
    let header_len = data.len().min(100);
    let mut i = 0;
    while i < header_len {
        let b = if (i >= 4 && i < 8)
            || (i >= 24 && i < 36)
            || (i >= 40 && i < 44)
            || (i >= 48 && i < 56)
            || (i >= 80)
        { 0u8 } else { data[i] };
        hash ^= b as u64;
        hash = hash.wrapping_mul(PRIME);
        i += 1;
    }

    // Phase 2: remaining bytes — straight hash.
    while i < data.len() {
        hash ^= data[i] as u64;
        hash = hash.wrapping_mul(PRIME);
        i += 1;
    }
    hash
}

// ── Reference EOTFs ──────────────────────────────────────────────────────

fn srgb_eotf(v: f64) -> f64 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}
fn bt709_eotf(v: f64) -> f64 {
    if v < 0.081 { v / 4.5 } else { ((v + 0.099) / 1.099).powf(1.0 / 0.45) }
}
fn gamma22_eotf(v: f64) -> f64 { v.powf(2.19921875) }
fn gamma18_eotf(v: f64) -> f64 { v.powf(1.8) }
fn pq_eotf(v: f64) -> f64 {
    const M1: f64 = 0.1593017578125;
    const M2: f64 = 78.84375;
    const C1: f64 = 0.8359375;
    const C2: f64 = 18.8515625;
    const C3: f64 = 18.6875;
    let vp = v.powf(1.0 / M2);
    let num = (vp - C1).max(0.0);
    let den = C2 - C3 * vp;
    if den <= 0.0 { 0.0 } else { (num / den).powf(1.0 / M1) }
}
fn hlg_ootf_inv(v: f64) -> f64 {
    // HLG OETF inverse (signal → scene linear)
    if v <= 0.5 { (v * v) / 3.0 } else {
        const A: f64 = 0.17883277;
        const B: f64 = 1.0 - 4.0 * A;
        const C: f64 = 0.5 - A * (4.0 * A - B) / (4.0 * A);
        (((v - C) / A).exp() + B) / 12.0
    }
}

#[allow(dead_code)]
struct RefTrc { name: &'static str, cp_name: &'static str, tf_name: &'static str, eotf: fn(f64) -> f64 }
const REFERENCE_TRCS: &[RefTrc] = &[
    RefTrc { name: "sRGB",     cp_name: "Bt709",    tf_name: "Srgb",    eotf: srgb_eotf },
    RefTrc { name: "BT.709",   cp_name: "Bt709",    tf_name: "Bt709",   eotf: bt709_eotf },
    RefTrc { name: "gamma2.2", cp_name: "AdobeRgb", tf_name: "Gamma22", eotf: gamma22_eotf },
    RefTrc { name: "gamma1.8", cp_name: "ProPhoto",  tf_name: "Gamma18", eotf: gamma18_eotf },
    RefTrc { name: "PQ",       cp_name: "Bt2020",   tf_name: "Pq",      eotf: pq_eotf },
    RefTrc { name: "HLG",      cp_name: "Bt2020",   tf_name: "Hlg",     eotf: hlg_ootf_inv },
];

// ── ICC TRC parsing ──────────────────────────────────────────────────────

enum Trc { Para(Vec<f64>), Lut(Vec<u16>), Gamma(f64) }

fn eval_para(p: &[f64], x: f64) -> f64 {
    match p.len() {
        1 => x.powf(p[0]),
        3 => { let (g, a, b) = (p[0], p[1], p[2]); if x >= -b / a { (a * x + b).powf(g) } else { 0.0 } }
        5 => { let (g, a, b, c, d) = (p[0], p[1], p[2], p[3], p[4]); if x >= d { (a * x + b).powf(g) } else { c * x } }
        7 => { let (g, a, b, c, d, e, f) = (p[0], p[1], p[2], p[3], p[4], p[5], p[6]); if x >= d { (a * x + b).powf(g) + e } else { c * x + f } }
        _ => x,
    }
}

fn eval_trc(t: &Trc, x: f64) -> f64 {
    match t {
        Trc::Para(p) => eval_para(p, x),
        Trc::Gamma(g) => x.powf(*g),
        Trc::Lut(l) => {
            let p = x * (l.len() - 1) as f64;
            let i = p.floor() as usize;
            let f = p - i as f64;
            if i >= l.len() - 1 { l[l.len() - 1] as f64 / 65535.0 }
            else { let a = l[i] as f64 / 65535.0; let b = l[i + 1] as f64 / 65535.0; a + f * (b - a) }
        }
    }
}

fn parse_trc(d: &[u8], o: usize) -> Option<Trc> {
    if o + 12 > d.len() { return None; }
    match &d[o..o + 4] {
        b"para" => {
            let ft = u16::from_be_bytes([d[o + 8], d[o + 9]]);
            let n = match ft { 0 => 1, 1 => 3, 2 => 4, 3 => 5, 4 => 7, _ => return None };
            let mut p = Vec::new();
            for i in 0..n {
                let q = o + 12 + i * 4;
                if q + 4 > d.len() { return None; }
                p.push(i32::from_be_bytes([d[q], d[q+1], d[q+2], d[q+3]]) as f64 / 65536.0);
            }
            Some(Trc::Para(p))
        }
        b"curv" => {
            let c = u32::from_be_bytes([d[o+8], d[o+9], d[o+10], d[o+11]]) as usize;
            if c == 0 { Some(Trc::Gamma(1.0)) }
            else if c == 1 { Some(Trc::Gamma(u16::from_be_bytes([d[o+12], d[o+13]]) as f64 / 256.0)) }
            else {
                let mut l = Vec::with_capacity(c);
                for i in 0..c { let q = o + 12 + i * 2; if q + 2 > d.len() { break; } l.push(u16::from_be_bytes([d[q], d[q+1]])); }
                Some(Trc::Lut(l))
            }
        }
        _ => None,
    }
}

fn find_tag(d: &[u8], s: &[u8; 4]) -> Option<usize> {
    if d.len() < 132 { return None; }
    let tc = u32::from_be_bytes([d[128], d[129], d[130], d[131]]) as usize;
    for i in 0..tc.min(200) {
        let b = 132 + i * 12;
        if b + 12 > d.len() { break; }
        if &d[b..b + 4] == s {
            return Some(u32::from_be_bytes([d[b+4], d[b+5], d[b+6], d[b+7]]) as usize);
        }
    }
    None
}

fn max_u16_err(trc: &Trc, eotf: fn(f64) -> f64) -> u32 {
    let mut mx = 0u32;
    for i in 0..=65535u16 {
        let x = i as f64 / 65535.0;
        let a = (eval_trc(trc, x) * 65535.0).round() as i64;
        let b = (eotf(x) * 65535.0).round() as i64;
        let d = (a - b).unsigned_abs() as u32;
        if d > mx { mx = d; }
    }
    mx
}

// ── Primaries identification ─────────────────────────────────────────────

struct KP { rust_name: &'static str, rx: f64, ry: f64, gx: f64, gy: f64, bx: f64, by: f64 }
const KNOWN_P: &[KP] = &[
    KP { rust_name: "Bt709",    rx: 0.4361, ry: 0.2225, gx: 0.3851, gy: 0.7169, bx: 0.1431, by: 0.0606 },
    KP { rust_name: "DisplayP3", rx: 0.5151, ry: 0.2412, gx: 0.2919, gy: 0.6922, bx: 0.1572, by: 0.0666 },
    KP { rust_name: "Bt2020",   rx: 0.6734, ry: 0.2790, gx: 0.1656, gy: 0.6753, bx: 0.1251, by: 0.0456 },
    KP { rust_name: "AdobeRgb", rx: 0.6097, ry: 0.3111, gx: 0.2053, gy: 0.6257, bx: 0.1492, by: 0.0632 },
    KP { rust_name: "ProPhoto", rx: 0.7977, ry: 0.2880, gx: 0.1352, gy: 0.7119, bx: 0.0313, by: 0.0001 },
];

fn identify_primaries(data: &[u8]) -> Option<&'static str> {
    if data.len() < 132 { return None; }
    let tc = u32::from_be_bytes([data[128], data[129], data[130], data[131]]) as usize;
    let (mut r, mut g, mut b) = ((0.0f64, 0.0f64), (0.0, 0.0), (0.0, 0.0));
    for i in 0..tc.min(200) {
        let base = 132 + i * 12;
        if base + 12 > data.len() { break; }
        let sig = &data[base..base+4];
        let off = u32::from_be_bytes([data[base+4], data[base+5], data[base+6], data[base+7]]) as usize;
        if off + 20 > data.len() { continue; }
        let rd = |o: usize| (
            i32::from_be_bytes([data[o+8], data[o+9], data[o+10], data[o+11]]) as f64 / 65536.0,
            i32::from_be_bytes([data[o+12], data[o+13], data[o+14], data[o+15]]) as f64 / 65536.0,
        );
        match sig { b"rXYZ" => r = rd(off), b"gXYZ" => g = rd(off), b"bXYZ" => b = rd(off), _ => {} }
    }
    for k in KNOWN_P {
        const T: f64 = 0.003;
        if (r.0 - k.rx).abs() < T && (r.1 - k.ry).abs() < T
            && (g.0 - k.gx).abs() < T && (g.1 - k.gy).abs() < T
            && (b.0 - k.bx).abs() < T && (b.1 - k.by).abs() < T
        { return Some(k.rust_name); }
    }
    None
}

// ── TRC identification ───────────────────────────────────────────────────

/// Returns (tf_rust_name, max_u16_error) for the best-matching reference.
fn identify_trc(data: &[u8], trc_tag: &[u8; 4]) -> Option<(&'static str, u32)> {
    let trc_off = find_tag(data, trc_tag)?;
    let trc = parse_trc(data, trc_off)?;

    let mut best: Option<(&str, u32)> = None;
    for r in REFERENCE_TRCS {
        let err = max_u16_err(&trc, r.eotf);
        if best.is_none() || err < best.unwrap().1 {
            best = Some((r.tf_name, err));
        }
    }
    best
}

// ── .inc formatting ──────────────────────────────────────────────────────

fn fmt_hash(h: u64) -> String {
    let s = format!("{h:016x}");
    format!("0x{}_{}_{}_{}", &s[0..4], &s[4..8], &s[8..12], &s[12..16])
}

fn clean_desc(fname: &str) -> String {
    fname.replace(".icc", "").replace('_', " ")
        .chars().take(50).collect()
}

// ── Main ─────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    // Parse arguments: all args before the last are input dirs, last is output dir.
    // Special case: if only one arg, it's the input dir (output defaults).
    // If no args, use defaults for both.
    let (input_dirs, out_dir) = if args.len() >= 2 {
        let out = PathBuf::from(args.last().unwrap());
        let inputs: Vec<PathBuf> = args[..args.len() - 1].iter().map(PathBuf::from).collect();
        (inputs, out)
    } else if args.len() == 1 {
        let input = PathBuf::from(&args[0]);
        let exe = std::env::current_exe().unwrap_or_default();
        let out = exe.parent().unwrap_or(&PathBuf::from(".")).join("../zenpixels/src/icc");
        (vec![input], out)
    } else if let Ok(d) = std::env::var("ICC_PROFILES_DIR") {
        let exe = std::env::current_exe().unwrap_or_default();
        let out = exe.parent().unwrap_or(&PathBuf::from(".")).join("../zenpixels/src/icc");
        (vec![PathBuf::from(d)], out)
    } else {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        let cache = PathBuf::from(home).join(".cache/zencodec-icc");
        let exe = std::env::current_exe().unwrap_or_default();
        let out = exe.parent().unwrap_or(&PathBuf::from(".")).join("../zenpixels/src/icc");
        (vec![cache], out)
    };

    // Always include the bundled profiles from zenpixels-convert.
    // Locate them relative to the output directory.
    let bundled_dir = out_dir.join("../../zenpixels-convert/src/profiles");
    let mut all_dirs = input_dirs.clone();
    if bundled_dir.exists() && !all_dirs.iter().any(|d| d == &bundled_dir) {
        all_dirs.push(bundled_dir.clone());
        eprintln!("Auto-including bundled profiles: {}", bundled_dir.display());
    }

    for dir in &all_dirs {
        if !dir.exists() {
            eprintln!("Directory not found: {}", dir.display());
            eprintln!("Usage: gen_icc_tables <dir1> [dir2 ...] <out-dir>");
            std::process::exit(1);
        }
    }

    // norm_hash → (cp_rust_name, tf_rust_name, max_err, description)
    let mut rgb: BTreeMap<u64, (&str, &str, u32, String)> = BTreeMap::new();
    let mut gray: BTreeMap<u64, (&str, u32, String)> = BTreeMap::new();
    let mut skipped = 0u32;
    let mut scanned = 0u32;

    let mut entries: Vec<PathBuf> = Vec::new();
    for dir in &all_dirs {
        if let Ok(rd) = std::fs::read_dir(dir) {
            entries.extend(
                rd.filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| matches!(p.extension().and_then(|e| e.to_str()), Some("icc" | "icm")))
            );
        }
    }
    entries.sort();

    for path in &entries {
        let data = match std::fs::read(path) { Ok(d) => d, Err(_) => continue };
        if data.len() < 132 { continue; }
        scanned += 1;

        let fname = path.file_name().unwrap().to_string_lossy().to_string();
        let cs = &data[16..20];
        let norm_hash = fnv1a_64_normalized(&data);

        if cs == b"RGB " {
            let Some(cp_name) = identify_primaries(&data) else { skipped += 1; continue; };
            let Some((tf_name, err)) = identify_trc(&data, b"rTRC") else { skipped += 1; continue; };
            if err > 56 { skipped += 1; continue; }

            rgb.entry(norm_hash)
                .and_modify(|e| e.2 = e.2.max(err))
                .or_insert((cp_name, tf_name, err, fname));
        } else if cs == b"GRAY" {
            // Try kTRC first, then gTRC
            let trc = identify_trc(&data, b"kTRC").or_else(|| identify_trc(&data, b"gTRC"));
            let Some((tf_name, err)) = trc else { skipped += 1; continue; };
            if err > 56 { skipped += 1; continue; }

            gray.entry(norm_hash)
                .and_modify(|e| e.1 = e.1.max(err))
                .or_insert((tf_name, err, fname));
        }
    }

    // ── Write RGB .inc ───────────────────────────────────────────────

    let rgb_path = out_dir.join("icc_table_rgb.inc");
    let mut rgb_out = String::from("&[\n");
    for (&h, (cp, tf, err, desc)) in &rgb {
        rgb_out += &format!(
            "    ({}, CP::{}, TF::{}, {:>2}),  // {}\n",
            fmt_hash(h), cp, tf, err, clean_desc(desc)
        );
    }
    rgb_out += "]\n";
    std::fs::write(&rgb_path, &rgb_out).unwrap();

    // ── Write gray .inc ──────────────────────────────────────────────

    let gray_path = out_dir.join("icc_table_gray.inc");
    let mut gray_out = String::from("&[\n");
    for (&h, (tf, err, desc)) in &gray {
        gray_out += &format!(
            "    ({}, TF::{}, {:>2}),  // {}\n",
            fmt_hash(h), tf, err, clean_desc(desc)
        );
    }
    gray_out += "]\n";
    std::fs::write(&gray_path, &gray_out).unwrap();

    // ── Summary ──────────────────────────────────────────────────────

    eprintln!("Scanned: {scanned} profiles");
    eprintln!("Skipped: {skipped} (unknown primaries or TRC error >56)");
    eprintln!("RGB:  {} entries → {}", rgb.len(), rgb_path.display());
    eprintln!("Gray: {} entries → {}", gray.len(), gray_path.display());

    // Count by type
    let mut combos: BTreeMap<String, usize> = BTreeMap::new();
    for (cp, tf, _, _) in rgb.values() {
        *combos.entry(format!("{cp}+{tf}")).or_default() += 1;
    }
    eprintln!("\nRGB by type:");
    let mut sorted_combos: Vec<_> = combos.iter().collect();
    sorted_combos.sort_by(|a, b| b.1.cmp(a.1));
    for (k, v) in sorted_combos {
        eprintln!("  {k}: {v}");
    }
}
