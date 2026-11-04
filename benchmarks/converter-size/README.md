# What `RowConverter` costs in binary size (2026-11-04)

`src/bin/base.rs` allocates a `PixelBuffer`. `src/bin/conv.rs` adds one RGBA8 to RGB8 `RowConverter`.
Both use fat LTO, one codegen unit, `panic = "abort"`, and `zenpixels-convert` with
`default-features = false, features = ["rgb"]`. Host r7900x, rustc 1.99.1, tree `8de69899`. Sizes are
stripped. Numbers are size only: no speed was measured here.

The scripts expect `nm -C -S --radix=d --size-sort`, `objdump -d --no-show-raw-insn -C` and
`llvm-dwarfdump --debug-line` output in `~/tmp/csa/`.

## Sections (x86_64, opt-level 3)

| Section | base | conv | added |
|---|---:|---:|---:|
| `.text` | 225,149 | 661,913 | 436,764 |
| `.rodata` | 20,988 | 67,352 | 46,364 |
| `.eh_frame` + `.eh_frame_hdr` | 19,484 | 40,700 | 21,216 |
| `.rela.dyn` | 8,424 | 20,592 | 12,168 |
| `.data.rel.ro` | 5,864 | 17,768 | 11,904 |
| Total | 293,018 | 821,501 | 528,483 |

Of the added `.rodata`, about 3.5 KB is printable strings and 5 KB is named LUTs. The rest wasn't
identified.

## What keeps the code alive

`by_source_and_dominators.py` builds a call graph from direct `call`, `jmp`, `j*` and `lea` targets and
computes dominators from `main`. Retained size is what would disappear if a function did. It accounts for
437,872 of the 438,123 bytes of new code, so the graph is close to complete.

```text
437,872  conv::main
385,577    RowConverter::try_convert_row
382,452      convert_row_buffered
377,858        apply_step_u8                  own code 6,691; dispatches 75 kernels
 29,545    ConvertPlan::new_explicit          includes ConvertPlan::new, 21,841
```

Kernels under `apply_step_u8`, by family. Assignment is by symbol name, so it's approximate. Shared
helpers count with whichever family alone reaches them.

| Family | Retained | Kernels |
|---|---:|---:|
| Gray | 64,593 | 9 |
| Matte compositing | 61,350 | 1 entry, monomorphized per transfer for `f32` and `f16` |
| Gamut matrix and LUT | 57,061 | 6 |
| sRGB and linear transfer | 32,365 | 10 |
| HLG | 29,889 | 4 |
| PQ | 28,178 | 6 |
| Oklab | 22,479 | 5 |
| Premultiply | 20,713 | 2 |
| Add or drop alpha, swizzle | 19,776 | 4 |
| `u16` and `u8` depth | 11,628 | 10 |
| BT.709 | 9,972 | 2 |
| Other | 5,576 | 9 |
| `f16` | 4,415 | 4 |
| Gamma 2.2 | 3,172 | 3 |

## By SIMD tier in the symbol name (zen crates)

| Tier | Bytes | Symbols |
|---|---:|---:|
| No tier in the name, mostly dispatchers with an inlined scalar body | 254,464 | 89 |
| `v3` | 104,021 | 86 |
| `v4` | 42,686 | 49 |
| `scalar` | 14,757 | 6 |

## Inlined source, from the line tables

`core` iterators and `f32` methods account for about 136 KB, `magetypes` for 54 KB (of which
`impls/scalar.rs` and `nostd_math.rs` are 52 KB), and `linear-srgb` for 36 KB. That is what scalar
fallback loops look like once inlined, and it points the same way as the matrix below.

`cargo llvm-lines` on the library shows no generic explosion: 141,692 lines across 2,552 copies, and the
largest single item is 3%.

## Build matrix

```text
o3                                 base=  293120 conv=  816016 added=  522896
o2                                 base=  292176 conv=  835672 added=  543496
os                                 base=  286776 conv=  688000 added=  401224
oz                                 base=  289128 conv=  676912 added=  387784
o3_cpu_v3                          base=  293440 conv=  767880 added=  474440
o3_cpu_v4                          base=  293440 conv=  749112 added=  455672
os_cpu_v3                          base=  287496 conv=  510120 added=  222624
wasm_o3                            base=   54201 conv=  362619 added=  308418
wasm_o3_simd                       base=   54683 conv=  354042 added=  299359
wasm_os_simd                       base=   54028 conv=  324307 added=  270279
wasm_oz_simd                       base=   57937 conv=  313932 added=  255995
```

- `opt-level = "s"` removes 122 KB of the added size. Dropping the scalar tier with
  `-Ctarget-cpu=x86-64-v3` removes 48 KB at opt-level 3. Together they remove 300 KB, far more than the
  sum, which points at autovectorized scalar loops at opt-level 3. `target-cpu` is not a general option,
  because users get runtime dispatch.
- wasm: `wasm_oz_simd` after `wasm-opt -Oz` is 252,479 bytes against a 38,600 byte base (214 KB added),
  and 85,274 against 16,913 after `gzip -9` (68 KB added). wasm has one SIMD tier and no `std` backtrace.

## After the tier rule (2026-11-04)

The numbers above are from before row kernels were limited to the tiers that pay for them. Opt-level 3
now adds 319,592 bytes, not 522,784. `../row_kernel_tiers_2026-11-04.md` has the rule, speeds and size
per build.
