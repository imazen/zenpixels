# Precision policy: what we keep by default, and who decides

Proposal, 2026-11-04, revised 2026-11-05. No policy or solver code lands until the current PR stack merges. Signatures go
through a PR for review. The measurements are in
[precision-audit-2026-11-04.md](precision-audit-2026-11-04.md).

## The rule

Bits that arrived in the decoded image don't get dropped unless someone asked for that by name.
Bits that were only ever storage can go without anyone asking.

| Situation | Default |
|---|---|
| 8-bit source, any working format, 8-bit output | Allowed, no event. Going through `u16`, `i16` or `f32` linear and back to 8 lost nothing that arrived |
| 8-bit content in `u16` storage, 8-bit output | Allowed, no event, once the content is known to be 8-bit |
| Real 16-bit source, output can hold it | Keep it |
| Real 16-bit source, output format chosen automatically | Pick a format that holds it |
| Real 16-bit source, conversion to fewer bits that the caller didn't name | Refuse |
| 8-bit source onto a `u16` canvas | Allowed. Output depth follows the canvas |
| Real 16-bit source onto an 8-bit canvas | Refuse unless allowed by name, because the canvas decides the output |
| Translucent source, output has no alpha, no matte | Refuse. Same category as depth |
| Peak memory over the limit at the depth that keeps the data | Refuse, naming both costs, unless a resolver chooses |

A direct `RowConverter::new(RGB16, RGB8)` already names its target, and it has no provenance to consult, so
it stays as it is. Naming a conversion is consent. The rule binds wherever a format is *chosen*: `adapt_for_encode` (41 call sites in 5 repos),
`best_match`, `negotiate`, zenpipe, zencodecs and the codec command-line tools. Those call sites run
`DiscardUnchecked` today, and only 2 sites in the tree set any depth policy.

## Which conversions are hidden

| Who chose | Example | Default |
|---|---|---|
| The caller named both ends | `RowConverter::new(RGB16, RGB8)`, `convert_to` | Do it, with no event |
| The caller named an output format and the encoder can hold the source | 16-bit PNG source to PNG | Keep it |
| The caller named an output format and the encoder can't hold the source | 16-bit PNG source to JPEG | Allowed: choosing the format is consent. The result records what was dropped |
| The same format in and out, and only one path keeps the data | 12-bit JPEG to JPEG | Take the path that keeps it. If the caller's request rules that out, refuse |
| The library chose the output format | Automatic format selection | Pick one that holds the source, or refuse |
| Something chose between decode and encode | `adapt_for_encode`, a zenpipe working format, a decoder asked for its first supported descriptor | Never below significant bits without a resolver or a named opt-in |

The last two rows are the ones a caller can't see, and they're what the policy exists for. `zencodecs` and `zenpipe` transcoding sit
on all of them at once, so that's where the choice has to become visible: in the encode result and its trace, with bits kept and
bits dropped. A narrowing nobody named must never be silent, even when it's allowed.

## "JPEG" doesn't name a depth

A format's capability isn't one number. It varies by codec and by path. JPEG in zenjpeg, as read from source at `762fa712` on
2026-11-05 (not run):

| Path | Depth |
|---|---|
| The format | 8 and 12-bit DCT; lossless SOF3 at 2 to 16 bits |
| zenjpeg decode (`extended-jpeg`, on by default) | 12-bit DCT and lossless. `OutputTarget::Native16` gives native codes, 0 to 4095 for 12-bit, not canonical U16. `sample_precision()` reports the source |
| zenjpeg's zencodec decoder | Since zenjpeg `a8ebcd0b` (2026-11-05): offers and honors `RGB16_SRGB`, `RGBA16_SRGB` and `GRAY16_SRGB` on one-shot, push and streaming; a frame above 8 bits gets U16 unless the preference list offers only U8. Matches libjpeg-turbo `djpeg` on 6 files. Before that, `U8` and `F32` only, and 12-bit input rounded to 8 |
| Coefficient transcoding (lossless transforms) | Keeps 12-bit, written as SOF1 |
| Pixel encoding and lossy recompression | 8-bit only |

So a capability table needs at least: what the decoder delivers, what the encoder accepts, and what a coefficient path keeps. Today a
Before zenjpeg `a8ebcd0b`, a 12-bit JPEG through `zencodecs` with a `U8` preference dropped 4 bits at the decoder with nothing but a metadata field to show it; with an only-U8 list it still does. This
is the hidden case in the table above. The fix has two parts, both after the stack merges: zenjpeg's zencodec decoder offers a canonical
`U16` exit, scaled to white at 65535 by `zenpixels_convert::codes`; and a decoder that delivers fewer bits than the source
reports that as a narrowing.

## Why two facts

- **Exact bits** (`ExactDepth`, exists): the samples currently sit on an N-bit grid. Any arithmetic ends this.
- **Significant bits** (new): how much information the source really had. A resize doesn't change it, and it
  sets the floor above. `Provenance.origin_depth` is a `ChannelType`, so it can't say 10 bits. It becomes bits.

Decoders declare what they know. A scan, behind a flag, measures the rest, because it costs a whole pass: `uses_low_bits`
detects 8 in 16 today, and `ExactDepth::holds_for` can test other depths. A format's claim such as PNG `sBIT` counts only
once a scan has verified it.

## Who decides

1. **Requirements first.** A floor on bits and a ceiling on memory remove offers before anyone is asked. This is the
   `PolicySet` rule from zencodec: a callback can't weaken a requirement.
2. **A resolver chooses among what's left.** Each offer states output format, peak bytes, estimated time, significant bits
   kept and dropped, and whether alpha survives.
3. **A preset answers if no resolver does.** Keep precision is the default. Prefer memory and prefer speed exist, by name.
4. **The result records the choice**, including anything dropped.

### Two levels that nest

zenpixels-convert carries the conversion-only form, for callers with no codec:

```rust,ignore
pub trait ConversionResolver {
    type Error: Error + Send + Sync + 'static;
    fn resolve(&mut self, question: ConversionQuestion<'_>) -> Result<ConversionDecision, At<Self::Error>>;
}
```

zencodec's `PolicyResolver` already has `Question::ImageConversion` with `ImageConversionOffer`, which states the output,
`admitted_bytes` and `workspace_upper_bound` but not bits kept. The offer gains those facts, using types from `zenpixels` core so
that zencodec needs no conversion dependency. A zencodec resolver can delegate the question to a `ConversionResolver`, and a
closure works at either level. `Decision::UsePreset` already ends the chain in zencodec. The conversion form adds `Defer`, which
is what makes resolvers composable: each one answers or passes to the next.

Whole-buffer conversions take `Option<&dyn Pulse>`, as zencodec's operations do, and check it between rows. Row kernels stay
free of it. Failures keep `whereat` traces, which `At<ConvertError>` already does.

## Validation of the old `pipeline` solver

`pipeline::optimal_path` was never adopted: no use outside zenpixels on any fetched branch of zenpipe, zencodec, zenjpeg, zenpng,
zenwebp, zenavif, zenjxl, zenfilters, zenresize or imageflow, and none in a GitHub code search of the org. It was removed
in `759a0e68`. Run against the rule above at `43a2c27`:

| Scenario | Its answer | The rule |
|---|---|---|
| 8-bit, resize, back to 8 | Refused at `Lossless`, loss 10 elsewhere | Allowed |
| Real 16-bit, resize, to 8 | Loss 10, bucket `Lossless` | Refuse |
| Real 16-bit, no operation, to 8 | Loss 10, bucket `Lossless` | Refuse |
| 8-bit onto a 16-bit canvas | Refused at `SubPerceptual`, `NearLossless` at best | Allowed |
| Real 16-bit, resize, to 16 | Loss 5, `Lossless` | Allowed |

It can't tell the first two rows apart, and it has no memory or time in its score. `provenance_after_operation` resets the origin to
the working depth, which erases exactly the fact the rule needs. Its 20 tests check internal consistency, so none of them caught this.
Worth keeping: the 13 operation categories in `op_format.rs` and the candidate working formats per category.

## The solver

A pass over the job's chain with two or three candidate formats per node. Each candidate's memory comes from the estimator,
which stops assuming `RGBA8`. Compile and estimate read the same result.

Measured for `zenresize` at 4096x3072 to 1024x768, on noise, on Zen 4: RGB8 linear 23.1 ms, RGBA8 linear 24.8, RGBF32 22.5, RGBAF32 24.8, RGB16
63.0, RGBA16 66.4. A whole frame at that size is 50 MB in RGBA8, 101 in RGBA16 and 201 in RGBAF32. Constants that land in source need
the four-axis sweep.

## Command-line tools use the same rule

| Tool | Today |
|---|---|
| `cjxl-rs` | Keeps 16-bit PNG input |
| `cavif` (`ravif`) | Since cavif-rs `0aa806ca` (2026-11-05): 16-bit input goes to 10-bit planes; `--depth=8` narrows, with a notice unless `--quiet`. Before that it narrowed to `RGBA8` with no notice |
| `zenpng` | Refuses 16-bit to convert, and its compare goes through `to_rgba8` |
| `zjpeg` | JPEG-to-JPEG operations, no pixel conversion |
| `zencodecs`, `zenpipe` | Whatever the library does |

## One CMS trait

`PluggableCms` is the trait. A prototype adapter, `impl JxlCms` over it, decodes real JPEG XL files; `benchmarks/jxl-cms-parity` has the
tables. Against lcms2 on RGB and gray the worst difference is 0.17 of a code. On CMYK it is 0.26 against pinned lcms2 values, the same as
the decoder's own wrapper. On HDR, moxcms and lcms2 differ by far more than the two moxcms paths do, so neither is a reference there.

| Need | `JxlCms` (zenjxl-decoder) | `PluggableCms` |
|---|---|---|
| Profiles | ICC, or a JXL encoding | ICC, CICP, named, primaries + transfer |
| Samples | `f32` only, interleaved | Any `PixelFormat`, as bytes |
| Channels | May differ, CMYK to RGB. Returns the count | Through layouts |
| Threads | `n` transformers, each with a pixel limit | One owned per call, or a shared `Arc`, with `prepare(max_width)` |
| In place | Yes | No |
| Luminance | `intensity_target` | None |
| Ink | 0 is maximum ink | `CmykF32Inverted` is that convention; `CmykF32` is 1 at maximum ink |
| Errors | `whereat` | `whereat` |

Remaining gaps, none blocking:

- `RowTransformMut` isn't `Sync`. The adapter holds it in a `Mutex`.
- There's no intensity target. Both JXL wrappers ignore it too, so it has no measured effect.

The decoder stays a jxl-rs fork with its own trait and no zenpixels dependency, so the adapter belongs in zenjxl, which has both. It lands
after this stack: zenjxl-decoder caps archmage below what this stack needs. zenjxl sets no CMS on the decoder today.

`cms-contract/` holds every backend to the same behavior. It found four defects in `MoxCms`, fixed on #113. The lcms port is a
reference only for matrix profiles for now, since it diverges from C on table profiles (imazen/lcms#1).

## Tone mapping

There is one algorithm: `Bt2446A`, the winner of zentone's 2026-06-22 shootout. `ToneMapper` injects another. `ConvertPlan::new_with_hdr_config`
takes a peak and `new_with_tone_mapper` takes the mapper.

Streaming doesn't work for PQ, because the peak needs every row. `zencodecs` holds PQ rows until `finish`, measures the brightest channel, and
reports `OutputTiming::AT_FINISH`. HLG has a fixed reference peak and streams. Declared MaxCLL and mastering peaks don't drive the render: all of
the shootout's top results measure, and the flicker notes recorded a declared 24,285 nits for content measuring 1,401. No metadata variant
was scored, so this isn't a measured comparison against metadata.

Two consequences: a flat 203-nit image renders as white, since 203 is its peak, and 203 nits at a 1000-nit peak is 171 of 255, lower than
BT.2408 would give.

Size, for `zenpixels-convert` alone, stripped with thin LTO, against a binary that already does a non-HDR conversion: `hdr-experimental` on and
unused adds 14,480 B at opt-3 or 8,144 B at opt-s. The machinery with an injected mapper adds 24,920 B or 16,496 B. With the `bt2446a` feature and
`Bt2446A` planned it adds 88,616 B or 100,888 B, so leaving that feature off saves 63,696 B or 84,392 B.
