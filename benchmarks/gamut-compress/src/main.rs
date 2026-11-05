//! Can image gamut compression be cheap enough to offer? Compares the existing
//! boundary-table `SoftCompress` with a two-step mapper: an interior gate in
//! linear RGB, then a direct solve for the chroma scale at the boundary.
#![forbid(unsafe_code)]
mod kernels;
#[path = "../../gamut-survey/src/matrices.rs"]
mod matrices;
#[path = "../../gamut-survey/src/methods.rs"]
#[allow(dead_code)]
mod methods;

use kernels::*;
use matrices::*;
use methods::*;
use std::io::Write;
use zenpixels::{ColorPrimaries as P, PixelDescriptor as D};
use zenpixels_convert::{RowConverter, hdr::SoftCompress, oklab::OklabSpace, policy::ConvertOptions};

const KNEE: f32 = 0.96;

fn source_matrix(g: usize) -> M {
    let p = if g == 0 {
        [(0.68, 0.32), (0.265, 0.69), (0.15, 0.06)]
    } else {
        [(0.708, 0.292), (0.17, 0.797), (0.131, 0.046)]
    };
    matmul(&inv(&CSS_SRGB_TO_XYZ), &rgb_to_xyz(p))
}

fn params(iters: u32, tau: f32, eps: f32) -> Params {
    let space = OklabSpace::for_primaries(P::Bt709).unwrap();
    Params {
        m1: *space.rgb_to_lms().rows(),
        m1_inv: *space.lms_to_rgb().rows(),
        knee: KNEE,
        tau,
        eps,
        iters,
        luma: [0.2126729, 0.7151522, 0.072175],
    }
}

fn soft() -> SoftCompress {
    SoftCompress::new(&OklabSpace::for_primaries(P::Bt709).unwrap(), KNEE)
}

/// Exact chroma boundary along constant L and hue, by bisection in f64.
fn exact_cmax(l: f64, a: f64, b: f64) -> f64 {
    let c = a.hypot(b);
    if c < 1e-12 {
        return 1.0;
    }
    let (ca, sa) = (a / c, b / c);
    let (mut lo, mut hi) = (0.0, 2.0);
    for _ in 0..60 {
        let mid = (lo + hi) / 2.0;
        if ing(unlab([l, mid * ca, mid * sa])) {
            lo = mid
        } else {
            hi = mid
        }
    }
    lo
}

/// The same knee applied to the exact boundary: the reference for the fast solve.
fn exact_map(rgb: [f64; 3], knee: f64) -> [f64; 3] {
    let [l, a, b] = lab(rgb);
    if l <= 0.0 || l >= 1.0 {
        return unlab([l, 0.0, 0.0]);
    }
    let c = a.hypot(b);
    if c < 1e-12 {
        return rgb;
    }
    let cmax = exact_cmax(l, a, b);
    let k = knee * cmax;
    if c <= k {
        return rgb;
    }
    let range = cmax - k;
    if range <= 0.0 {
        return unlab([l, 0.0, 0.0]);
    }
    let excess = c - k;
    let s = (k + range * excess / (excess + range)) / c;
    unlab([l, a * s, b * s])
}

/// Boundary points at `knee`: the tightest tau for each eps.
fn derive_gate() -> Vec<(f64, f64)> {
    let epss = [0.002, 0.005, 0.01, 0.02, 0.05];
    let mut tau = vec![f64::NEG_INFINITY; epss.len()];
    for li in 1..400 {
        let l = li as f64 / 400.0;
        for hi in 0..1440 {
            let (sh, ch) = (hi as f64 / 1440.0 * core::f64::consts::TAU).sin_cos();
            let cmax = exact_cmax(l, ch, sh);
            // Tested past the knee too: nothing further out may pass either.
            for k in [KNEE as f64 - 1e-3, 0.98, 1.0] {
                let p = unlab([l, k * cmax * ch, k * cmax * sh]);
                let mx = p[0].max(p[1]).max(p[2]);
                let mn = p[0].min(p[1]).min(p[2]);
                if mx <= 0.0 {
                    continue;
                }
                for (i, e) in epss.iter().enumerate() {
                    if mx < 1.0 - e {
                        tau[i] = tau[i].max(mn / mx);
                    }
                }
            }
        }
    }
    epss.iter().copied().zip(tau).collect()
}

type Eval = Box<dyn Fn(&mut [[f32; 3]]) + Send + Sync>;

fn methods(tau: f32, eps: f32) -> Vec<(&'static str, Eval)> {
    let s = soft();
    let mut out: Vec<(&'static str, Eval)> = vec![
        ("clip", Box::new(|v| clip_strip(v))),
        ("soft", Box::new(move |v| s.apply_strip(v))),
    ];
    for (name, n) in [("solve-3", 3)] {
        let p = params(n, tau, eps);
        out.push((name, Box::new(move |v| solve_strip(&p, v))));
    }
    let p = params(0, tau, eps);
    out.push(("line-rgb", Box::new(move |v| line_strip(&p, v))));
    out
}

fn eval(f: &Eval, input: &[[f64; 3]]) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let mut v: Vec<[f32; 3]> = input.iter().map(|x| x.map(|c| c as f32)).collect();
    f(&mut v);
    let raw: Vec<[f64; 3]> = v.iter().map(|x| x.map(|c| c as f64)).collect();
    let bounded = raw.iter().map(|x| clip(*x)).collect();
    (raw, bounded)
}

fn quality(tau: f32, eps: f32) {
    let mut prim = std::fs::File::create("results/primaries.csv").unwrap();
    let mut sweep = std::fs::File::create("results/sweep.csv").unwrap();
    let mut ramps = std::fs::File::create("results/ramps.csv").unwrap();
    writeln!(prim, "gamut,method,primary,hue_deg,de,r8,g8,b8").unwrap();
    writeln!(sweep, "gamut,method,n,in_gamut,in_gamut_changed,in_gamut_max_de,mean_de,max_de,mean_hue,max_hue,raw_outside,max_excursion,de_vs_exact_mean,de_vs_exact_max,de_vs_exact_inside_mean").unwrap();
    writeln!(ramps, "gamut,method,steps,reversals,max_reverse,ramps_with_reversal,max_adjacent_de").unwrap();
    let ms = methods(tau, eps);
    for g in 0..2 {
        let gn = ["P3", "BT2020"][g];
        let m = source_matrix(g);
        for (name, f) in &ms {
            for (i, pn) in ["red", "green", "blue"].iter().enumerate() {
                let src = mul(&m, core::array::from_fn(|j| if i == j { 1.0 } else { 0.0 }));
                let (_, b) = eval(f, &[src]);
                let (o, a) = (lab(src), lab(b[0]));
                let q = b[0].map(|x| (enc(x).clamp(0.0, 1.0) * 255.0).round() as u8);
                writeln!(prim, "{gn},{name},{pn},{:.3},{:.5},{},{},{}", hue_delta(o, a), de(o, a), q[0], q[1], q[2]).unwrap();
            }
            // Same 17^3 grid in the source's encoded-as-linear cube as the survey.
            let cube: Vec<[f64; 3]> = (0..17)
                .flat_map(|r| {
                    (0..17).flat_map(move |g| {
                        (0..17).map(move |b| [r as f64 / 16.0, g as f64 / 16.0, b as f64 / 16.0])
                    })
                })
                .collect();
            let input: Vec<[f64; 3]> = cube.iter().map(|&c| mul(&m, c)).collect();
            let (raw, out) = eval(f, &input);
            let (mut inside, mut changed, mut imax) = (0u32, 0u32, 0.0f64);
            let (mut sum, mut max, mut hs, mut hm, mut hn) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0u32);
            let (mut outside, mut exc) = (0u32, 0.0f64);
            let (mut xs, mut xm, mut xi, mut xin) = (0.0f64, 0.0f64, 0.0f64, 0u32);
            for i in 0..input.len() {
                let o = lab(input[i]);
                let a = lab(out[i]);
                let e = de(o, a);
                sum += e;
                max = max.max(e);
                if chroma(o) > 1e-4 && chroma(a) > 1e-4 {
                    let h = hue_delta(o, a).abs();
                    hs += h;
                    hm = hm.max(h);
                    hn += 1;
                }
                let is_in = input[i].iter().all(|v| (-1e-12..=1.0 + 1e-12).contains(v));
                let e2 = de(lab(clip(exact_map(input[i], KNEE as f64))), a);
                xs += e2;
                xm = xm.max(e2);
                if is_in {
                    inside += 1;
                    imax = imax.max(e);
                    if e > 1e-6 {
                        changed += 1
                    }
                    xi += e2;
                    xin += 1;
                }
                let x = raw[i].iter().map(|&v| (-v).max(v - 1.0).max(0.0)).fold(0.0, f64::max);
                if x > 1e-7 {
                    outside += 1
                }
                exc = exc.max(x);
            }
            let n = input.len() as f64;
            writeln!(sweep, "{gn},{name},{},{inside},{changed},{imax:.8},{:.6},{max:.6},{:.4},{hm:.4},{outside},{exc:.6},{:.7},{xm:.6},{:.8}", input.len(), sum / n, hs / hn as f64, xs / n, xi / xin as f64).unwrap();
            // Same 288 ramps as the survey: constant L and hue out to the source boundary.
            let mi = inv(&m);
            let (mut rev, mut maxrev, mut withrev, mut maxstep, mut steps) = (0u64, 0.0f64, 0u32, 0.0f64, 0u64);
            for ll in [0.2, 0.4, 0.6, 0.8] {
                for h in (0..360).step_by(5) {
                    let (sh, ch) = (h as f64).to_radians().sin_cos();
                    let (mut lo, mut hi) = (0.0, 1.0);
                    for _ in 0..44 {
                        let c = (lo + hi) / 2.0;
                        if ing(mul(&mi, unlab([ll, c * ch, c * sh]))) {
                            lo = c
                        } else {
                            hi = c
                        }
                    }
                    let src: Vec<_> = (0..1024)
                        .map(|i| {
                            let c = lo * i as f64 / 1023.0;
                            unlab([ll, c * ch, c * sh])
                        })
                        .collect();
                    let (_, o) = eval(f, &src);
                    let mut prev = lab(o[0]);
                    let mut has = false;
                    for x in o.iter().skip(1) {
                        let l = lab(*x);
                        let r = chroma(prev) - chroma(l);
                        if r > 1e-7 {
                            rev += 1;
                            has = true
                        }
                        maxrev = maxrev.max(r);
                        maxstep = maxstep.max(de(prev, l));
                        steps += 1;
                        prev = l;
                    }
                    if has {
                        withrev += 1
                    }
                }
            }
            writeln!(ramps, "{gn},{name},{steps},{rev},{maxrev:.8},{withrev},{maxstep:.7}").unwrap();
        }
    }
    // The gate must be conservative: it may never pass a pixel the map would change.
    let p = params(4, tau, eps);
    let (mut passed, mut wrong, mut worst) = (0u64, 0u64, 0.0f64);
    let n = 257;
    for r in 0..n {
        for g in 0..n {
            for b in 0..n {
                let px = [r as f32 / 256.0, g as f32 / 256.0, b as f32 / 256.0];
                if gate_passes(&p, &px) {
                    passed += 1;
                    let x = px.map(|v| v as f64);
                    let e = de(lab(x), lab(exact_map(x, KNEE as f64)));
                    if e > 1e-9 {
                        wrong += 1;
                        worst = worst.max(e);
                    }
                }
            }
        }
    }
    println!("gate conservative check tau={tau} eps={eps}: {n}^3 grid, {passed} passed, {wrong} would have changed, worst dE {worst:.8}");
}

struct Img {
    name: String,
    /// Linear sRGB-primaries RGB, unclipped, from Display P3 8-bit sRGB-curve input.
    rgb: Vec<[f32; 3]>,
    p3_bytes: Vec<u8>,
}

fn load(path: &std::path::Path, m: &M, max_rows: usize) -> Img {
    let mut d = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()));
    d.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut r = d.read_info().unwrap();
    let icc = r.info().icc_profile.as_ref().map(|c| c.to_vec());
    let id = icc.as_deref().and_then(zenpixels::icc::IccIdentification::from_icc);
    let mut buf = vec![0u8; r.output_buffer_size().unwrap()];
    let info = r.next_frame(&mut buf).unwrap();
    let ch = info.color_type.samples();
    let w = info.width as usize;
    // A centered 2048-wide band: enough real content, bounded RAM.
    let (x0, cw) = (w.saturating_sub(2048) / 2, w.min(2048));
    let rows = (info.height as usize).min(max_rows);
    let y0 = (info.height as usize - rows) / 2;
    let lut: Vec<f32> = (0..256).map(|i| dec(i as f64 / 255.0) as f32).collect();
    let m32: [[f32; 3]; 3] = m.map(|r| r.map(|v| v as f32));
    let mut rgb = Vec::with_capacity(cw * rows);
    let mut bytes = Vec::with_capacity(cw * rows * 3);
    for y in y0..y0 + rows {
        let row = &buf[y * info.line_size..][..info.line_size];
        for x in x0..x0 + cw {
            let p = &row[x * ch..x * ch + 3];
            bytes.extend_from_slice(p);
            let l = [lut[p[0] as usize], lut[p[1] as usize], lut[p[2] as usize]];
            rgb.push(core::array::from_fn(|i| {
                m32[i][0] * l[0] + m32[i][1] * l[1] + m32[i][2] * l[2]
            }));
        }
    }
    let name: String = path.file_name().unwrap().to_string_lossy().split('_').take(3).collect::<Vec<_>>().join("_");
    println!("{name}: {cw}x{rows}, ch={ch}, icc={:?}", id.map(|i| (i.primaries, i.transfer)));
    Img { name, rgb, p3_bytes: bytes }
}

fn images(dir: &str, m: &M) -> Vec<Img> {
    let mut p: Vec<_> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|e| e == "png")).collect();
    p.sort();
    p.iter().map(|p| load(p, m, 1024)).collect()
}

fn real(dir: &str, tau: f32, eps: f32) {
    let m = source_matrix(0);
    let imgs = images(dir, &m);
    let mut f = std::fs::File::create("results/real.csv").unwrap();
    writeln!(f, "image,pixels,out_of_srgb_pct,gate_pass_pct,chunks_all_interior_pct,clip_mean_de,clip_max_de,solve2_mean_de,solve2_max_de,solve2_changed_pct,table_changed_pct,solve2_vs_table_mean_de,solve2_vs_table_max_de,clip_vs_solve2_pct_gt_0.02").unwrap();
    let ps = params(2, tau, eps);
    let s = soft();
    for img in &imgs {
        let n = img.rgb.len() as f64;
        let out = img.rgb.iter().filter(|p| p.iter().any(|&v| !(-1e-6..=1.0 + 1e-6).contains(&v))).count() as f64;
        let pass = img.rgb.iter().filter(|p| gate_passes(&ps, p)).count() as f64;
        let chunks = img.rgb.chunks_exact(8).filter(|c| c.iter().all(|p| gate_passes(&ps, p))).count() as f64;
        let mut a = img.rgb.clone();
        solve_strip(&ps, &mut a);
        let mut b = img.rgb.clone();
        s.apply_strip(&mut b);
        let to = |p: &[f32; 3]| lab(clip(p.map(|v| v as f64)));
        let (mut cm, mut cx, mut sm, mut sx, mut sc, mut tc, mut vm, mut vx, mut big) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0u64, 0u64, 0.0f64, 0.0f64, 0u64);
        for i in 0..img.rgb.len() {
            let o = lab(img.rgb[i].map(|v| v as f64));
            let (la, lb, lc) = (to(&a[i]), to(&b[i]), to(&img.rgb[i]));
            let (ec, es) = (de(o, lc), de(o, la));
            cm += ec;
            cx = cx.max(ec);
            sm += es;
            sx = sx.max(es);
            if a[i] != img.rgb[i] {
                sc += 1
            }
            if de(o, lb) > 1e-4 {
                tc += 1
            }
            let v = de(la, lb);
            vm += v;
            vx = vx.max(v);
            if de(lc, la) > 0.02 {
                big += 1
            }
        }
        writeln!(f, "{},{},{:.3},{:.2},{:.2},{:.6},{:.4},{:.6},{:.4},{:.3},{:.3},{:.6},{:.4},{:.4}", img.name, img.rgb.len(), out / n * 100.0, pass / n * 100.0, chunks * 8.0 / n * 100.0, cm / n, cx, sm / n, sx, sc as f64 / n * 100.0, tc as f64 / n * 100.0, vm / n, vx, big as f64 / n * 100.0).unwrap();
    }
}

fn scrambled(g: usize) -> Vec<[f32; 3]> {
    let m = source_matrix(g);
    (0..1024)
        .map(|i| {
            let c = [((i * 313 + 17) % 1024) as f64 / 1023.0, ((i * 619 + 71) % 1024) as f64 / 1023.0, ((i * 109 + 337) % 1024) as f64 / 1023.0];
            mul(&m, c).map(|v| v as f32)
        })
        .collect()
}

fn opts() -> ConvertOptions {
    ConvertOptions::permissive().with_clip_out_of_gamut(false)
}

fn bench(dir: &str, tau: f32, eps: f32) {
    let m = source_matrix(0);
    let imgs = images(dir, &m);
    let result = zenbench::run(|suite| {
        let mut sets: Vec<(String, Vec<[f32; 3]>)> = vec![("scrambled-P3".into(), scrambled(0)), ("scrambled-BT2020".into(), scrambled(1))];
        for i in &imgs {
            sets.push((i.name.clone(), i.rgb[..(1 << 18).min(i.rgb.len())].to_vec()));
        }
        for (set, data) in sets {
            suite.group(format!("kernel-{set}"), |group| {
                let cfg = group.config();
                cfg.max_rounds = 30;
                cfg.min_rounds = 20;
                cfg.auto_rounds = false;
                cfg.warmup_time = std::time::Duration::from_millis(100);
                cfg.max_time = std::time::Duration::from_secs(2);
                cfg.max_wall_time = std::time::Duration::from_secs(20);
                cfg.min_iterations = 1;
                cfg.max_iterations = 16;
                group.throughput(zenbench::Throughput::Elements(data.len() as u64));
                for (name, f) in methods(tau, eps) {
                    let src = data.clone();
                    let mut dst = data.clone();
                    group.bench(name, move |b| {
                        b.iter(|| {
                            dst.copy_from_slice(std::hint::black_box(&src));
                            f(&mut dst);
                            std::hint::black_box(&dst);
                        })
                    });
                }
            });
        }
        // Whole P3 u8 to sRGB u8 row: today's fused clip against gate-and-solve.
        for img in &imgs {
            suite.group(format!("u8-row-{}", img.name), |group| {
                let cfg = group.config();
                cfg.max_rounds = 30;
                cfg.min_rounds = 20;
                cfg.auto_rounds = false;
                cfg.warmup_time = std::time::Duration::from_millis(100);
                cfg.max_time = std::time::Duration::from_secs(2);
                cfg.max_iterations = 16;
                let w = 2048usize.min(img.p3_bytes.len() / 3);
                let rows = (img.p3_bytes.len() / 3 / w).min(128);
                group.throughput(zenbench::Throughput::Elements((w * rows) as u64));
                let from = D::RGB8_SRGB.with_primaries(P::DisplayP3);
                let to = D::RGB8_SRGB;
                let src = img.p3_bytes[..w * rows * 3].to_vec();
                let mut c = RowConverter::new(from, to).unwrap();
                let s2 = src.clone();
                let mut dst = vec![0u8; w * 3];
                group.bench("native-clip", move |b| {
                    b.iter(|| {
                        for row in s2.chunks_exact(w * 3) {
                            c.convert_row(row, &mut dst, w as u32);
                            std::hint::black_box(&dst);
                        }
                    })
                });
                let mut opt = RowConverter::new_explicit(
                    from,
                    to,
                    &ConvertOptions::permissive().with_gamut_mapping(zenpixels::GamutMapping::SoftCompress),
                )
                .unwrap();
                let s2 = src.clone();
                let mut dst = vec![0u8; w * 3];
                group.bench("option-soft", move |b| {
                    b.iter(|| {
                        for row in s2.chunks_exact(w * 3) {
                            opt.convert_row(row, &mut dst, w as u32);
                            std::hint::black_box(&dst);
                        }
                    })
                });
            });
        }
    });
    std::fs::write(std::env::var("BENCH_JSON").unwrap(), serde_json::to_string_pretty(&result).unwrap()).unwrap();
    println!("{}", result.to_markdown());
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    std::fs::create_dir_all("results").unwrap();
    let (tau, eps): (f32, f32) = (a.get(3).and_then(|v| v.parse().ok()).unwrap_or(0.0), a.get(4).map(|v| v.parse().unwrap()).unwrap_or(0.01));
    match a.get(1).map(|s| s.as_str()) {
        Some("css-golden") => {
            // Reference values for the crate's map_to_gamut test: the f64 port that
            // agreed with Color.js within 1.55e-14 in the gamut survey.
            let m = source_matrix(0);
            let m2 = source_matrix(1);
            for (name, mat, c) in [
                ("p3 red", &m, [1.0, 0.0, 0.0]),
                ("p3 green", &m, [0.0, 1.0, 0.0]),
                ("p3 blue", &m, [0.0, 0.0, 1.0]),
                ("p3 mid", &m, [0.9, 0.2, 0.3]),
                ("2020 red", &m2, [1.0, 0.0, 0.0]),
                ("2020 green", &m2, [0.0, 1.0, 0.0]),
                ("2020 blue", &m2, [0.0, 0.0, 1.0]),
                ("2020 mid", &m2, [0.2, 0.8, 0.6]),
            ] {
                let e = css(mul(mat, c));
                println!("{name}: {:.6} {:.6} {:.6}", e[0], e[1], e[2]);
            }
        }
        Some("worst") => {
            let m = source_matrix(0);
            let p = params(3, tau, eps);
            let mut rows = Vec::new();
            for r in 0..17 { for g in 0..17 { for b in 0..17 {
                let c = [r as f64 / 16.0, g as f64 / 16.0, b as f64 / 16.0];
                let x = mul(&m, c);
                let mut v = [x.map(|q| q as f32)];
                solve_strip(&p, &mut v);
                let o = clip(v[0].map(|q| q as f64));
                let e = de(lab(o), lab(clip(exact_map(x, KNEE as f64))));
                rows.push((e, c, x, o));
            }}}
            rows.sort_by(|a, b| b.0.total_cmp(&a.0));
            for (e, c, x, o) in &rows[..6] {
                let [l, a, b] = lab(*x);
                let h = a.hypot(b);
                // First crossing along chroma, by marching out from gray.
                let mut first = 0.0;
                for i in 1..200_000 {
                    let t = i as f64 / 100_000.0;
                    if !ing(unlab([l, a / h * t, b / h * t])) { first = t; break }
                }
                println!("de {e:.5} src {c:?} L {l:.5} C {h:.5} bisect {:.5} first_crossing {first:.5} solve_out {:?}", exact_cmax(l, a, b), o.map(|v| (v * 1e4).round() / 1e4));
            }
        }
        Some("planar") => {
            let m = source_matrix(0);
            let imgs = images(&a[2], &m);
            let space = OklabSpace::for_primaries(P::Bt709).unwrap();
            let result = zenbench::run(|suite| {
                for img in &imgs {
                    let n = (1usize << 18).min(img.rgb.len());
                    let (mut l, mut oa, mut ob) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
                    for i in 0..n {
                        let [x, y, z] = space.from_rgb(img.rgb[i]);
                        l[i] = x;
                        oa[i] = y;
                        ob[i] = z;
                    }
                    suite.group(format!("planar-{}", img.name), |group| {
                        let cfg = group.config();
                        cfg.max_rounds = 30;
                        cfg.min_rounds = 20;
                        cfg.auto_rounds = false;
                        cfg.max_iterations = 16;
                        group.throughput(zenbench::Throughput::Elements(n as u64));
                        let sc = soft();
                        let (mut wa, mut wb) = (oa.clone(), ob.clone());
                        group.bench("solve", move |b| {
                            b.iter(|| {
                                wa.copy_from_slice(&oa);
                                wb.copy_from_slice(&ob);
                                sc.compress_oklab_planes(&l, &mut wa, &mut wb).unwrap();
                                std::hint::black_box(&wa);
                            })
                        });
                    });
                }
            });
            std::fs::write(&a[3], serde_json::to_string_pretty(&result).unwrap()).unwrap();
        }
        Some("gate-all") => {
            // Boundary points for each target and knee: the tightest tau at eps 0.05.
            let targets: [(&str, [(f64, f64); 3]); 3] = [
                ("Bt709", [(0.64, 0.33), (0.30, 0.60), (0.15, 0.06)]),
                ("DisplayP3", [(0.68, 0.32), (0.265, 0.69), (0.15, 0.06)]),
                ("Bt2020", [(0.708, 0.292), (0.17, 0.797), (0.131, 0.046)]),
            ];
            for (name, prim) in targets {
                let to = inv(&matmul(&inv(&CSS_SRGB_TO_XYZ), &rgb_to_xyz(prim)));
                let eval = |lab_v: [f64; 3]| mul(&to, unlab(lab_v));
                for knee in [0.90f64, 0.96] {
                    let mut tau = f64::NEG_INFINITY;
                    for li in 1..400 {
                        let l = li as f64 / 400.0;
                        for hi in 0..1440 {
                            let (sh, ch) = (hi as f64 / 1440.0 * core::f64::consts::TAU).sin_cos();
                            let (mut lo, mut hi2) = (0.0, 2.0);
                            for _ in 0..60 {
                                let mid = (lo + hi2) / 2.0;
                                if ing(eval([l, mid * ch, mid * sh])) { lo = mid } else { hi2 = mid }
                            }
                            for k in [knee - 1e-3, knee + 0.02, 1.0] {
                                let p = eval([l, k * lo * ch, k * lo * sh]);
                                let mx = p[0].max(p[1]).max(p[2]);
                                let mn = p[0].min(p[1]).min(p[2]);
                                if mx > 0.0 && mx < 0.95 {
                                    tau = tau.max(mn / mx);
                                }
                            }
                        }
                    }
                    println!("{name} knee {knee}: tau {tau:.6}");
                }
            }
        }
        Some("gate") => {
            for (e, t) in derive_gate() {
                println!("eps={e} tau={t:.6}");
            }
        }
        Some("quality") => quality(tau, eps),
        Some("real") => real(&a[2], tau, eps),
        Some("bench") => bench(&a[2], tau, eps),
        _ => panic!("gate | quality _ tau eps | real dir tau eps | bench dir tau eps"),
    }
}
