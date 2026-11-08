//! Compare explicit H.273 colorimetry from both CMS constructors with unchanged TRCs.
use lcms::{gamma, profile, types, virt};
use std::path::Path;

#[allow(dead_code)]
mod reference {
    include!("../../scripts/icc-gen/src/bin/cicp_bundle_gen/colourimetry.rs");
    pub fn chromaticities(p: u8) -> ([[f64; 2]; 3], [f64; 2]) {
        h273_chromaticities(p)
    }
    pub fn colorants(p: u8) -> [[f64; 3]; 3] {
        expected_h273_colorants(p)
    }
}
#[path = "../../zenpixels-convert/src/icc_profiles/canonical.rs"]
#[allow(dead_code)]
mod canonical;

fn tag(bytes: &[u8], signature: &[u8; 4]) -> Option<Vec<u8>> {
    let read = |offset: usize| -> Option<usize> {
        Some(u32::from_be_bytes(bytes.get(offset..offset + 4)?.try_into().ok()?) as usize)
    };
    for i in 0..read(128)? {
        let entry = 132 + 12 * i;
        if bytes.get(entry..entry + 4)? == signature {
            let start = read(entry + 4)?;
            return Some(
                bytes
                    .get(start..start.checked_add(read(entry + 8)?)?)?
                    .to_vec(),
            );
        }
    }
    None
}

fn candidate(source: &[u8], primaries: u8, mox_exact: bool) -> Vec<u8> {
    let (xy, white) = reference::chromaticities(primaries);
    let xy_y = |[x, y]: [f64; 2]| types::CiexyY { x, y, Y: 1.0 };
    let primaries = types::CiexyYTriple {
        Red: xy_y(xy[0]),
        Green: xy_y(xy[1]),
        Blue: xy_y(xy[2]),
    };
    // Only colorimetry comes from this constructor. Original serialized curves
    // replace the placeholders verbatim, so this comparison cannot hide a TRC edit.
    let placeholder = gamma::build_gamma(None, 1.0).expect("identity curve");
    let mut built = if mox_exact {
        let chromaticity = |[x, y]: [f64; 2]| moxcms::Chromaticity {
            x: x as f32,
            y: y as f32,
        };
        let mut built = moxcms::ColorProfile::new_from_slice(source).expect("source profile");
        built.update_rgb_colorimetry(
            moxcms::XyY {
                x: white[0],
                y: white[1],
                yb: 1.0,
            },
            moxcms::ColorPrimaries {
                red: chromaticity(xy[0]),
                green: chromaticity(xy[1]),
                blue: chromaticity(xy[2]),
            },
        );
        profile::open_profile_from_mem(None, &built.encode().expect("MoxCMS writer"))
            .expect("controlled MoxCMS colorimetry")
    } else {
        virt::create_rgb_profile(
            None,
            Some(&xy_y(white)),
            Some(&primaries),
            Some([&placeholder, &placeholder, &placeholder]),
        )
        .expect("LCMS RGB constructor")
    };
    for signature in [*b"rTRC", *b"gTRC", *b"bTRC", *b"cicp"] {
        let data = tag(source, &signature).expect("controlled source tag");
        assert!(profile::cms_write_raw_tag(
            &mut built,
            u32::from_be_bytes(signature),
            &data
        ));
    }
    built.rendering_intent = u32::from_be_bytes(source[64..68].try_into().unwrap());
    let bytes =
        canonical::canonicalize(profile::save_profile_to_mem(&mut built).expect("ICC writer"))
            .expect("canonical ICC structure");
    for signature in [*b"rTRC", *b"gTRC", *b"bTRC", *b"cicp"] {
        assert_eq!(
            tag(source, &signature),
            tag(&bytes, &signature),
            "{signature:?}"
        );
    }
    bytes
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if !(2..=3).contains(&args.len()) {
        return Err("expected source profiles, fresh output directory, optional mox-exact".into());
    }
    let mox_exact = match args.get(2).and_then(|v| v.to_str()) {
        None => false,
        Some("mox-exact") => true,
        _ => return Err("unknown candidate constructor".into()),
    };
    let input = Path::new(&args[0]);
    let output = Path::new(&args[1]);
    std::fs::create_dir(output)?;
    use std::io::Write;
    let mut reference = std::fs::File::create(output.join("reference.tsv"))?;
    writeln!(
        reference,
        "primaries\twhite_x\twhite_y\twhite_z\tr_x\tr_y\tr_z\tg_x\tg_y\tg_z\tb_x\tb_y\tb_z"
    )?;
    for p in [1, 4, 5, 6, 7, 8, 9, 11, 12, 22] {
        let (_, [x, y]) = reference::chromaticities(p);
        write!(reference, "{p}")?;
        for value in [x / y, 1.0, (1.0 - x - y) / y]
            .into_iter()
            .chain(reference::colorants(p).into_iter().flatten())
        {
            write!(reference, "\t{value:.17e}")?;
        }
        writeln!(reference)?;
    }
    let mut files: Vec<_> = std::fs::read_dir(input)?
        .map(|e| e.unwrap().path())
        .collect();
    files.sort();
    let (mut rgb, mut gray) = (0, 0);
    for path in files {
        let name = path.file_name().unwrap().to_str().unwrap();
        if name.starts_with("rgb-") && name.ends_with(".icc") {
            let p: u8 = name.split('-').nth(1).unwrap().parse()?;
            let source = std::fs::read(&path)?;
            let data = candidate(&source, p, mox_exact);
            std::fs::write(output.join(name), &data)?;
            rgb += 1;
        } else if name.starts_with("gray-") && name.ends_with(".icc") {
            std::fs::copy(&path, output.join(name))?;
            gray += 1;
        }
    }
    assert_eq!((rgb, gray), (159, 175));
    println!(
        "{} colorimetry: {rgb} RGB profiles; {gray} gray controls unchanged; transfer/CICP tags exact",
        if mox_exact {
            "MoxCMS exact-H273"
        } else {
            "LCMS exact-H273"
        }
    );
    Ok(())
}
