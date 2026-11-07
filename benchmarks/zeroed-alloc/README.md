# Zeroed allocation: `resize` against `calloc` (2026-11-04)

`vec![0; n]` and `bytemuck::allocation::try_zeroed_vec` go through `alloc_zeroed`, which can take
untouched zero pages from the OS. `try_reserve_exact` plus `resize` writes every byte. `try_zeroed_vec`
is also fallible and needs no `unsafe` in this crate, so `PixelStorage` and `AlignedSamples` use it.

Host r7900x (Zen 4), Ubuntu GLIBC 2.43-2ubuntu2.4, rustc 1.99.1, `u16` samples, 20 fixed rounds per
group through zenbench 0.1.11. Not measured on Zen 5.

```sh
cd benchmarks/zeroed-alloc && cargo bench --bench z -- --no-busy-gate
```

```text
  alloc only 4 KiB  20 rounds × 23K calls
  ├─ try_reserve+resize   48.4 ±1.2ns  [47.4–49.5]ns
  ├─ vec![0; n]           45.2 ±1.1ns  [-7.8%–-5.7%]
  ╰─ try_zeroed_vec       44.2 ±0.8ns  [-10.1%–-8.1%]
  alloc then write 4 KiB  20 rounds × 11K calls
  ├─ try_reserve+resize   93.2 ±1.3ns  [91.9–95.0]ns
  ├─ vec![0; n]           87.0 ±0.8ns  [-6.8%–-5.6%]
  ╰─ try_zeroed_vec       88.3 ±1.1ns  [-6.0%–-4.3%]
  alloc only 64 KiB  20 rounds × 2K calls
  ├─ try_reserve+resize   415 ±1ns  [412–417]ns
  ├─ vec![0; n]           408 ±7ns  [-1.7%–-0.9%]   [1]
  ╰─ try_zeroed_vec       409 ±6ns  [-1.7%–-1.2%]
  alloc then write 64 KiB  20 rounds × 876 calls
  ├─ try_reserve+resize   1.2 ±0.0µs  [1.2–1.2]µs
  ├─ vec![0; n]           1.2 ±0.0µs  [-0.4%–+0.0%]   [1]
  ╰─ try_zeroed_vec       1.2 ±0.0µs  [-0.8%–-0.5%]   [2] [3]
  alloc only 1 MiB  20 rounds × 130 calls
  ├─ try_reserve+resize   7.5 ±0.2µs  [7.4–7.6]µs
  ├─ vec![0; n]           7.6 ±0.2µs  [-0.5%–+0.8%]   [1]
  ╰─ try_zeroed_vec       7.5 ±0.1µs  [-0.8%–+0.7%]   [2]
  alloc then write 1 MiB  20 rounds × 50 calls
  ├─ try_reserve+resize   19.5 ±0.2µs  [19.4–19.6]µs
  ├─ vec![0; n]           19.4 ±0.1µs  [-1.4%–+0.4%]    [1]
  ╰─ try_zeroed_vec       19.4 ±0.5µs  [-1.5%–+0.3%]    [2]
  alloc only 16 MiB  20 rounds × 43 calls
  ├─ try_reserve+resize   115.7 ±1.1µs  [115.1–116.2]µs
  ├─ vec![0; n]           114.5 ±2.4µs  [-1.7%–-0.2%]      [1]
  ╰─ try_zeroed_vec       115.7 ±1.8µs  [-0.6%–+0.6%]      [2]
  alloc then write 16 MiB  20 rounds × 16 calls
  ├─ try_reserve+resize   310.3 ±2.0µs  [309.3–311.4]µs
  ├─ vec![0; n]           308.8 ±3.0µs  [-1.0%–-0.0%]      [1]
  ╰─ try_zeroed_vec       308.7 ±4.9µs  [-1.2%–+0.1%]      [2]
  alloc only 64 MiB  20 rounds × 175 calls
  ├─ try_reserve+resize   21.3 ±0.2ms  [21.2–21.4]ms
  ├─ vec![0; n]            0.0 ±0.0ms  [-100.3%–-99.7%]
  ╰─ try_zeroed_vec        0.0 ±0.0ms  [-100.3%–-99.7%]
  alloc then write 64 MiB  20 rounds × 1 calls
  ├─ try_reserve+resize   24.2 ±0.6ms  [23.9–24.6]ms
  ├─ vec![0; n]           21.8 ±0.2ms  [-9.4%–-8.1%]
  ╰─ try_zeroed_vec       21.9 ±0.2ms  [-9.8%–-8.3%]
  alloc only 256 MiB  10 rounds × 141 calls
  ├─ try_reserve+resize   92.7 ±0.7ms  [92.3–93.0]ms
  ├─ vec![0; n]            0.0 ±0.0ms  [-100.4%–-99.6%]   [1]
  ╰─ try_zeroed_vec        0.0 ±0.0ms  [-100.4%–-99.6%]   [2]
  alloc then write 256 MiB  20 rounds × 1 calls
  ├─ try_reserve+resize   104.4 ±2.3ms  [103.6–105.2]ms
  ├─ vec![0; n]            94.7 ±0.9ms  [-10.2%–-8.5%]
  ╰─ try_zeroed_vec        93.4 ±2.1ms  [-11.4%–-9.6%]
```

- **Under 32 MiB** all three are within about 2%, except about 6 to 10% at 4 KiB, in favor of the zeroed
  forms. This loop frees and reallocates, which raises glibc's mmap threshold to its 32 MiB maximum,
  so `calloc` reuses heap memory and has to clear it. A first allocation of 1 to 32 MiB in a fresh
  process would get fresh pages instead; that case isn't measured here.
- **At 64 MiB and up** allocation alone goes from 21 ms (64 MiB) and 93 ms (256 MiB) to under the
  reporting resolution. When every byte is then written, about 8 to 11% remains, because the OS still
  zeroes each page when it is first touched.
- **`try_zeroed_vec` against `vec![0; n]`:** within 1% in every group but one, where it's 2% slower
  at 4 KiB write-through, so there's no speed reason for a separate infallible path.
