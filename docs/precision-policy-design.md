# Precision policy: what we keep by default, and who decides

Proposal, 2026-11-04. Signatures go through a PR for review. The measurements are in
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
it stays as it is. The rule binds wherever a format is *chosen*: `adapt_for_encode` (41 call sites in 5 repos),
`best_match`, `negotiate`, zenpipe, zencodecs and the codec command-line tools. Those call sites run
`DiscardUnchecked` today, and only 2 sites in the tree set any depth policy.

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
| `cavif` (`ravif`) | Narrows `RGB16`, `RGBA16`, `GRAY16` and `GRAYA16` input to `RGBA8` with no notice |
| `zenpng` | Refuses 16-bit to convert, and its compare goes through `to_rgba8` |
| `zjpeg` | JPEG-to-JPEG operations, no pixel conversion |
| `zencodecs`, `zenpipe` | Whatever the library does |

## One CMS trait

| Need | `JxlCms` (zenjxl-decoder) | `PluggableCms` |
|---|---|---|
| Profiles | ICC, or a JXL encoding | ICC, CICP, named, primaries + transfer |
| Samples | `f32` only, interleaved | Any `PixelFormat`, as bytes |
| Channels | May differ, CMYK to RGB. Returns the count | Through layouts |
| Threads | `n` transformers, each with a pixel limit | One owned per call, or a shared `Arc`, with `prepare(max_width)` |
| In place | Yes | No |
| Luminance | `intensity_target` | None |
| Ink | 0 is maximum ink | Not stated |
| Errors | `whereat` | `whereat` |

`PluggableCms` covers the larger case. Three gaps stand between it and JXL: an intensity target, a stated ink convention, and gray and
CMYK coverage proven in tests. In-place and `n` transformers fit in an adapter. The decoder stays a jxl-rs fork with its own trait
and no zenpixels dependency, so the adapter belongs in zenjxl, which has both. There is then one trait to implement across our crates.

Two findings: zenjxl sets no CMS on the decoder at all, and zenjxl-decoder's moxcms wrapper sets `allow_use_cicp_transfer` true and ignores
`intensity_target`, while zenpixels-convert's sets it false. Those two can disagree on the same profile.

## Tone mapping

There is one algorithm: `Bt2446A`, the winner of zentone's 2026-06-22 shootout. It reaches zencodecs through
`ConvertPlan::new_with_hdr_config`. `ToneMapper`, with `Requirements`, `ToneMapContext` and `StripMapper`, is on the unmerged
`feat/tone-map-two-phase` branch, which #61 closed in favor of.
