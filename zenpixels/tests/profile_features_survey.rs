//! Survey profile features across the local ICC cache.
//! Run with: cargo test -p zenpixels --test profile_features_survey -- --ignored --nocapture

#[cfg(feature = "icc")]
#[test]
#[ignore] // only run when explicitly requested
fn survey_corpus_features() {
    let cache = std::env::var("ICC_CACHE").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}/.cache/zenpixels-icc")
    });
    let mut safe = 0;
    let mut unsafe_lab = 0;
    let mut unsafe_lut = 0;
    let mut unsafe_chad = 0;
    let mut no_matrix = 0;
    let mut total = 0;
    let entries = match std::fs::read_dir(&cache) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if !matches!(
            path.extension().and_then(|s| s.to_str()),
            Some("icc" | "icm")
        ) {
            continue;
        }
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(_) => continue,
        };
        total += 1;
        if let Some(feat) = zenpixels::icc::inspect_profile(&data) {
            if feat.is_safe_matrix_shaper() {
                safe += 1;
            } else if !feat.has_matrix_shaper {
                no_matrix += 1;
            } else if feat.pcs_is_lab {
                unsafe_lab += 1;
            } else if feat.has_a2b0
                || feat.has_a2b1
                || feat.has_a2b2
                || feat.has_b2a0
                || feat.has_b2a1
                || feat.has_b2a2
            {
                unsafe_lut += 1;
            } else if feat.has_chad && !feat.chad_is_bradford {
                unsafe_chad += 1;
            }
        }
    }
    eprintln!("\n=== ICC Profile Features Survey (cache: {cache}) ===");
    eprintln!("Total profiles:                  {total}");
    eprintln!("Safe matrix-shaper:              {safe}");
    eprintln!("Has matrix tags + LUTs:          {unsafe_lut}");
    eprintln!("Lab PCS:                         {unsafe_lab}");
    eprintln!("Non-Bradford chad:               {unsafe_chad}");
    eprintln!("No matrix-shaper tags:           {no_matrix}");
}
