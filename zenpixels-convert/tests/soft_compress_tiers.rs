//! SoftCompress uses no fused operations, so every SIMD tier gives identical bits.
use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use zenpixels_convert::ColorPrimaries as P;
use zenpixels_convert::gamut::GamutMatrix;
use zenpixels_convert::hdr::SoftCompress;
use zenpixels_convert::oklab::OklabSpace;

fn bits(rows: &[[f32; 3]]) -> Vec<u32> {
    rows.iter().flatten().map(|v| v.to_bits()).collect()
}

#[test]
fn every_tier_gives_identical_bits() {
    for target in [P::Bt709, P::DisplayP3] {
        let space = OklabSpace::for_primaries(target).unwrap();
        let sc = SoftCompress::new(&space, SoftCompress::DEFAULT_KNEE);
        let m = GamutMatrix::between(P::Bt2020, target).unwrap();
        let mut input = Vec::new();
        for r in 0..=16 {
            for g in 0..=16 {
                for b in 0..=16 {
                    input.push(m.apply([r as f32 / 16.0, g as f32 / 16.0, b as f32 / 16.0]));
                }
            }
        }
        let lab: Vec<[f32; 3]> = input.iter().map(|&c| space.from_rgb(c)).collect();
        let l: Vec<f32> = lab.iter().map(|v| v[0]).collect();

        let mut reference: Option<(Vec<u32>, Vec<u32>, Vec<u32>)> = None;
        let mut tiers = Vec::new();
        let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            tiers.push(perm.to_string());
            let mut rgb = input.clone();
            sc.apply_strip(&mut rgb);
            let mut a: Vec<f32> = lab.iter().map(|v| v[1]).collect();
            let mut b: Vec<f32> = lab.iter().map(|v| v[2]).collect();
            sc.compress_oklab_planes(&l, &mut a, &mut b).unwrap();
            let got = (
                bits(&rgb),
                a.iter().map(|v| v.to_bits()).collect(),
                b.iter().map(|v| v.to_bits()).collect(),
            );
            match &reference {
                None => reference = Some(got),
                Some(want) => assert!(*want == got, "{perm} differs on {target:?}"),
            }
        });
        println!("{target:?}: {} permutations: {tiers:?}", tiers.len());
    }
}
