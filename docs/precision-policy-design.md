# Precision policy: what we keep by default, and who decides

Proposal, 2026-11-04, revised 2026-11-05 after implementing it in `zenpixels::precision`. Signatures go through a PR for
review. The measurements are in [precision-audit-2026-11-04.md](precision-audit-2026-11-04.md).

## The rule

Bits that arrived in the decoded image don't get dropped unless someone asked for that by name.
Bits that were only ever storage can go without anyone asking.

| Situation | Default |
|---|---|
| 8-bit source, any working format that holds it, 8-bit output | Allowed, no event. Going through `u16`, `i16` or `f32` linear and back to 8 lost nothing that arrived. `u8` linear doesn't hold it: see "What a format holds" |
| 8-bit content in `u16` storage, 8-bit output | Allowed, no event, once the content is known to be 8-bit |
| Real 16-bit source, output can hold it | Keep it |
| Real 16-bit source, output format chosen automatically | Pick a format that holds it |
| Real 16-bit source, conversion to fewer bits that the caller didn't name | Refuse |
| 8-bit source onto a `u16` canvas | Allowed. Output depth follows the canvas |
| Real 16-bit source onto an 8-bit canvas | Refuse unless allowed by name, because the canvas decides the output |
| Translucent source, output has no alpha, no matte | Refuse, even for a format the caller named, unless a resolver chooses it |
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
| The caller named an output format and the encoder can't hold the source | 16-bit PNG source to JPEG | Allowed: choosing the format is consent to lost bits and range. The result records what was dropped. Not to lost alpha: what was under it shows, so the result isn't a rendition of the same image |
| The same format in and out, and only one path keeps the data | 12-bit JPEG to JPEG | Take the path that keeps it. If the caller's request rules that out, refuse |
| The library chose the output format | Automatic format selection | Pick one that holds the source, or refuse |
| Something chose between decode and encode | `adapt_for_encode`, a zenpipe working format, a decoder asked for its first supported descriptor | Never below significant bits without a resolver or a named opt-in |

The last two rows are the ones a caller can't see, and they're what the policy exists for. `zencodecs` and `zenpipe` transcoding sit
on all of them at once, so that's where the choice has to become visible: in the encode result and its trace, with bits kept and
bits dropped. A narrowing nobody named must never be silent, even when it's allowed.

In code, `OutputChoice` has two values. `CallerNamed` is the third row. `Hidden` is the last three, which turned out to need the
same rule, and the same-format row, because a caller asking for the source's own format meant the path that keeps it. The
first row doesn't reach the resolver, and the second is `CallerNamed` with an offer that holds the source.

## "JPEG" doesn't name a depth

A format's capability isn't one number. It varies by codec and by path. JPEG in zenjpeg, as read from source at `762fa712` on
2026-11-05 (not run):

| Path | Depth |
|---|---|
| The format | 8 and 12-bit DCT; lossless SOF3 at 2 to 16 bits |
| zenjpeg decode (`extended-jpeg`, on by default) | 12-bit DCT and lossless. `OutputTarget::Native16` gives native codes, 0 to 4095 for 12-bit, not canonical U16. `sample_precision()` reports the source |
| zenjpeg's zencodec decoder | `U8` and `F32` descriptors only. 12-bit input rounds to 8 when `U8` is preferred, with `bit_depth` 12 in `ImageInfo`. A comment there still says the parser rejects 12-bit |
| Coefficient transcoding (lossless transforms) | Keeps 12-bit, written as SOF1 |
| Pixel encoding and lossy recompression | 8-bit only |

So a capability table needs at least: what the decoder delivers, what the encoder accepts, and what a coefficient path keeps. Today a
12-bit JPEG through `zencodecs` with a `U8` preference drops 4 bits at the decoder with nothing but a metadata field to show it. This
is the hidden case in the table above. The fix has two parts, both after the stack merges: zenjpeg's zencodec decoder offers a canonical
`U16` exit, scaled to white at 65535 by `zenpixels_convert::codes`; and a decoder that delivers fewer bits than the source
reports that as a narrowing.

## Why two facts

- **Exact bits** (`ExactDepth`, exists): the samples currently sit on an N-bit grid. Any arithmetic ends this.
- **Significant bits** (`SourcePrecision`, new): how much information the source really had. A resize doesn't change it, and
  it sets the floor above. A composite carries the deeper of its inputs (`joined_with`).

`Provenance.origin_depth` stays a `ChannelType`, so it can't say 10 bits. It remains what it was, a cost hint for `negotiate`; the
rule reads `SourcePrecision`. Merging them would change `negotiate`'s scores, and nothing in the rule needs that.

Decoders declare what they know. `scan_source_precision` measures the rest, as the caller's choice, because it costs a whole pass
over the samples. A format's claim such as PNG `sBIT` counts only once a scan has verified it. A declaration names the source's own
transfer and range, not the current samples': a JPEG held in `f32` linear is `declared(8, Srgb, Full)`.

## What a format holds

A format's capacity isn't its storage width. Integer output with another curve holds less, because codes that were distinct land on
one value. These are the deepest sources whose codes all stay at least one storage step apart, and a test derives each from the curve:

| Source curve into linear integer | `u8` | `u16` | `u32` | `u64` |
|---|---|---|---|---|
| sRGB | 4 | 12 | 28 | 60 |
| BT.709 | 5 | 13 | 29 | 61 |
| Gamma 2.2 | 3 | 7 | 14 | 29 |
| HLG | 3 | 7 | 15 | 31 |
| PQ | 1 | 3 | 12 | 20 |

| Linear source into curved integer | `u8` | `u16` | `u32` | `u64` |
|---|---|---|---|---|
| sRGB, BT.709, Gamma 2.2 | 6 | 14 | 30 | 62 |
| HLG | 5 | 13 | 29 | 61 |
| PQ | 4 | 12 | 28 | 60 |

Run through the real kernels: 8-bit sRGB comes back whole through `u16` linear and `f32`, and loses codes through `u8` linear. 16-bit
sRGB comes back whole through `f32`, and loses codes through `u16` linear. 12-bit codes in `u16` come back whole through `u16` linear.

A float holds its mantissa, 11 bits for `f16`, 24 for `f32` and 53 for `f64`, whatever the curve. A full-range source holds one bit
less on a narrow-range grid of the same width. Between two curves that aren't linear, such as sRGB and BT.709, storage width is the
answer used: this doesn't derive a table per pair.

Dynamic range is counted apart from bits. A PQ or HLG source, or a float source nobody has said otherwise about, needs a float, PQ or
HLG output.

## Who decides

1. **Requirements first.** A floor on bits and a ceiling on memory remove offers before anyone is asked. This is the
   `PolicySet` rule from zencodec: a callback can't weaken a requirement.
2. **A resolver chooses among what's left.** Each offer states output format, peak bytes, estimated time, significant bits
   kept and dropped, and whether alpha survives.
3. **A preset answers if no resolver does.** `KeepPrecision` is the default. `DropPrecisionForMemory` exists, by name, and drops
   bits only: alpha and range take a resolver. There's no speed preset, because nothing states a time: see "The solver".
4. **The result records the choice**, including anything dropped.

### Two levels that nest

`zenpixels::precision` carries the conversion-only form, for callers with no codec. It sits in the core crate rather than
zenpixels-convert, because zencodec has to reach it without a conversion dependency. It decides from descriptors and stated facts,
so it needs no conversion math:

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

The resolver takes no stop token, because it decides without reading samples. The one long operation is the scan,
`PixelSliceConvertExt::scan_source_precision`, in zenpixels-convert: the core crate takes no new dependency, and its dependency guard says so.
It takes `&dyn enough::Stop` and checks it between rows, and a `&dyn how_far::Pulse` upcasts to it. Whole-buffer conversions don't change here: a caller that wants
cancellation owns the row loop over `RowConverter`, as zenpipe does. Row kernels stay free of it. Refusals and resolver failures keep
`whereat` traces, and a chain's failure names its node.

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

`ConversionPolicy::plan_chain` is a pass over the job's chain with two or three candidate formats per node. Each candidate's memory comes
from the estimator, which stops assuming `RGBA8`. Compile and estimate read the same result. It takes the lowest peak any plan can have,
then the fewest format changes among plans at that peak, then offer order. Bits dropped early stay dropped: a later, deeper format doesn't
bring them back.

The chain's owner supplies the candidates, because only it knows which kernels exist. So the 13 categories in `op_format.rs` don't come back
as public types, and zenpipe's own nodes name what they can run in.

Time isn't in the score. An offer has no time field and there's no speed preset, because no caller can state one yet: the figures below come
from one sweep point, and constants that land in source need the four-axis sweep.

Measured for `zenresize` at 4096x3072 to 1024x768, on noise, on Zen 4: RGB8 linear 23.1 ms, RGBA8 linear 24.8, RGBF32 22.5, RGBAF32 24.8, RGB16
63.0, RGBA16 66.4. A whole frame at that size is 50 MB in RGBA8, 101 in RGBA16 and 201 in RGBAF32. Constants that land in source need
the four-axis sweep.

## Command-line tools use the same rule

| Tool | Today |
|---|---|
| `cjxl-rs` | Keeps 16-bit PNG input |
| `cavif` (`ravif`) | Narrows `RGB16`, `RGBA16`, `GRAY16` and `GRAYA16` input to `RGBA8` with no notice |
| `zenpng` | Refuses 16-bit to convert, and its compare goes through `to_rgba8` |
| `zjpeg` | JPEG-to-JPEG operations, no pixel conversion |
| `zencodecs`, `zenpipe` | `zenpipe`'s graph plans through the rule on `feat/precision-policy`. `zencodecs` is whatever the library does |

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
