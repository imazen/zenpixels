//! Demonstrate `GamutClip::Preserve` vs the per-channel hard clip when
//! narrowing a Display P3 image into sRGB.
//!
//! Input is a binary PPM (`P6`) assumed to be Display P3 with the sRGB transfer
//! function — exactly what the `heic` decoder emits for an iPhone photo, e.g.:
//!
//! ```text
//! # in the heic repo:
//! cargo run --example decode -- /mnt/v/heic/IMG_6189.HEIC poppies_p3.ppm
//! # then here:
//! cargo run --example gamut_clip_demo -- poppies_p3.ppm /mnt/v/output/poppies
//! ```
//!
//! Writes two sRGB PPMs (`<prefix>_hardclip.ppm`, `<prefix>_preserve.ppm`) you
//! can compare side by side, and prints clipping statistics: how much of the
//! image was actually out of the sRGB gamut, and how much detail the per-channel
//! clip collapses that the snap recovers.

use std::fs;
use std::io::Write;

use zenpixels_convert::{
    ColorPrimaries, ConvertOptions, GamutClip, PixelDescriptor, RowConverter,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <input_p3.ppm> <output_prefix>", args[0]);
        std::process::exit(1);
    }
    let input = &args[1];
    let prefix = &args[2];

    let (w, h, p3) = read_ppm(input).unwrap_or_else(|e| {
        eprintln!("Failed to read {input}: {e}");
        std::process::exit(1);
    });
    let n = (w as usize) * (h as usize);
    println!("Input: {w}x{h} ({n} px), assumed Display P3 / sRGB transfer");

    let from = PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3);
    let to = PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::Bt709);

    let hard = convert(&p3, w, h, from, to, GamutClip::PerChannel);
    let soft = convert(&p3, w, h, from, to, GamutClip::Preserve);

    // Statistics.
    let mut out_of_gamut = 0usize; // pixels the snap had to move
    let mut hard_pinned = 0usize; // hard-clip pixels with a channel pinned to 255
    let mut collapse_pairs = 0usize; // hard pixels that are a pure 255-primary
    for i in 0..n {
        let (hp, sp) = (&hard[i * 3..i * 3 + 3], &soft[i * 3..i * 3 + 3]);
        if hp != sp {
            out_of_gamut += 1;
        }
        if hp.contains(&255) && hp != sp {
            hard_pinned += 1;
        }
        // A hard-clipped pixel pinned at 255 in one channel and crushed to 0 in
        // another is exactly the lost-detail case the snap recovers.
        if hp.contains(&255) && hp.contains(&0) && hp != sp {
            collapse_pairs += 1;
        }
    }
    let pct = |c: usize| 100.0 * c as f64 / n as f64;
    println!("Out of sRGB gamut (snap moved):     {out_of_gamut:>10} px ({:.2}%)", pct(out_of_gamut));
    println!("Hard-clip pinned a channel to 255:  {hard_pinned:>10} px ({:.2}%)", pct(hard_pinned));
    println!("Detail-collapse (255 & 0 crushed):  {collapse_pairs:>10} px ({:.2}%)", pct(collapse_pairs));

    write_ppm(&format!("{prefix}_hardclip.ppm"), w, h, &hard).unwrap();
    write_ppm(&format!("{prefix}_preserve.ppm"), w, h, &soft).unwrap();
    println!("Wrote {prefix}_hardclip.ppm and {prefix}_preserve.ppm (both sRGB)");
}

fn convert(
    p3: &[u8],
    w: u32,
    h: u32,
    from: PixelDescriptor,
    to: PixelDescriptor,
    gamut_clip: GamutClip,
) -> Vec<u8> {
    let opts = ConvertOptions::permissive().with_gamut_clip(gamut_clip);
    let mut conv = RowConverter::new_explicit(from, to, &opts).unwrap();
    let mut out = vec![0u8; p3.len()];
    // Convert the whole image as one wide row.
    conv.convert_row(p3, &mut out, w * h);
    out
}

/// Minimal binary PPM (`P6`) reader.
fn read_ppm(path: &str) -> Result<(u32, u32, Vec<u8>), String> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    let mut pos = 0;
    let mut tokens = Vec::new();
    // Read magic + 3 header integers, skipping whitespace and # comments.
    while tokens.len() < 4 && pos < bytes.len() {
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos < bytes.len() && bytes[pos] == b'#' {
            while pos < bytes.len() && bytes[pos] != b'\n' {
                pos += 1;
            }
            continue;
        }
        let start = pos;
        while pos < bytes.len() && !bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        tokens.push(String::from_utf8_lossy(&bytes[start..pos]).into_owned());
    }
    if tokens.len() < 4 || tokens[0] != "P6" {
        return Err("not a binary P6 PPM".into());
    }
    let w: u32 = tokens[1].parse().map_err(|_| "bad width")?;
    let h: u32 = tokens[2].parse().map_err(|_| "bad height")?;
    let maxval: u32 = tokens[3].parse().map_err(|_| "bad maxval")?;
    if maxval != 255 {
        return Err("only 8-bit (maxval 255) PPM supported".into());
    }
    pos += 1; // single whitespace after maxval before pixel data
    let need = (w as usize) * (h as usize) * 3;
    if bytes.len() < pos + need {
        return Err("truncated pixel data".into());
    }
    Ok((w, h, bytes[pos..pos + need].to_vec()))
}

fn write_ppm(path: &str, w: u32, h: u32, rgb: &[u8]) -> Result<(), String> {
    let mut f = fs::File::create(path).map_err(|e| e.to_string())?;
    write!(f, "P6\n{w} {h}\n255\n").map_err(|e| e.to_string())?;
    f.write_all(rgb).map_err(|e| e.to_string())?;
    Ok(())
}
