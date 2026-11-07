use zenbench::prelude::*;
fn resize(n: usize) -> Vec<u16> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).unwrap();
    v.resize(n, 0);
    v
}
fn macro_(n: usize) -> Vec<u16> {
    vec![0u16; n]
}
fn zeroed(n: usize) -> Vec<u16> {
    bytemuck::allocation::try_zeroed_vec::<u16>(n).unwrap()
}
fn fill(mut v: Vec<u16>) -> Vec<u16> {
    for x in v.iter_mut() {
        *x = 3;
    }
    v
}
fn main() {
    zenbench::run(|suite| {
        for (label, bytes) in [
            ("4 KiB", 4usize << 10),
            ("64 KiB", 64 << 10),
            ("1 MiB", 1 << 20),
            ("16 MiB", 16 << 20),
            ("64 MiB", 64 << 20),
            ("256 MiB", 256 << 20),
        ] {
            let n = bytes / 2;
            suite.group(format!("alloc only {label}"), |g| {
                g.config().auto_rounds(false).min_rounds(20).max_rounds(20);
                g.bench("try_reserve+resize", move |b| {
                    b.iter(|| black_box(resize(n)))
                });
                g.bench("vec![0; n]", move |b| b.iter(|| black_box(macro_(n))));
                g.bench("try_zeroed_vec", move |b| b.iter(|| black_box(zeroed(n))));
            });
            suite.group(format!("alloc then write {label}"), |g| {
                g.config().auto_rounds(false).min_rounds(20).max_rounds(20);
                g.bench("try_reserve+resize", move |b| {
                    b.iter(|| black_box(fill(resize(n))))
                });
                g.bench("vec![0; n]", move |b| b.iter(|| black_box(fill(macro_(n)))));
                g.bench("try_zeroed_vec", move |b| {
                    b.iter(|| black_box(fill(zeroed(n))))
                });
            });
        }
    });
}
