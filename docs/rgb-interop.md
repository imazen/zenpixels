# zenpixels and the `rgb` crate: 0.8.9x / 1.0 interop study

Written 2026-10-31 against zenpixels `main` (`873f1e7`), `release/0.2`,
`release/0.3.1` and `feat/0.3.1-extensibility` (the rgb integration is the same
on all four), and rust-rgb `main` (`7915bd4`, 2026-02-15) plus its tags
`v0.8.53` and `v0.8.54`. Observations about releases and issues are dated;
re-check them with the commands at the end before acting.

## Where rgb is going

- **There is no rgb 0.9 release.** The redesign that was called "0.9" ships in
  the 0.8 line as 0.8.9x, so one `Rgb` type stays shared across the ecosystem.
  Cargo will not unify a prerelease like `0.8.90-alpha` with plain `0.8`
  requirements, which is why the numbers jumped (comment in rust-rgb's
  `Cargo.toml`; [issue 113](https://github.com/kornelski/rust-rgb/issues/113)).
- **Published (crates.io, checked 2026-10-31):** stable `0.8.53` (2026-02-25);
  prerelease `0.8.92-rc.1` (2026-02-15). The `v0.8.54` tag (2026-08-29) and
  `main` (`version = "0.8.92"`) are not published.
- **1.0** is 0.8.9x minus the deprecated items, with old and new versions
  sharing types through the semver trick (one version re-exports the other's
  types). The maintainer's stated rule is staged migration (deprecate, let
  users move, then drop compatibility), never a flag-day break
  ([issue 89](https://github.com/kornelski/rust-rgb/issues/89),
  [issue 39](https://github.com/kornelski/rust-rgb/issues/39)). No date;
  issues 39 and [131](https://github.com/kornelski/rust-rgb/issues/131) are open.
- Our request to publish 0.8.92 and 0.8.54
  ([issue 158](https://github.com/kornelski/rust-rgb/issues/158), 2026-04-16)
  has no reply.
- **Soundness ([issue 160](https://github.com/kornelski/rust-rgb/issues/160),
  2026-07-04):** with the default `as-bytes` feature, 0.8.53 implements `Pod`
  for `Rgba<T, A>` with any `T` and `A`; `Rgba<u8, u16>` has padding, so that
  impl is unsound. `main` implements `Pod` only for `Rgba<T>` (`A = T`).
- **The 0.8.9x design:** named fields everywhere (`Gray { v }`,
  `GrayA { v, a }`); a separate alpha type parameter (`Rgba<T, A = T>`);
  traits `HetPixel`, `Pixel`, `HasAlpha`, `GainAlpha`, `ArrayLike` in
  `rgb::prelude`; byte casts through bytemuck only; `Bgr`, `Argb`, `Abgr`,
  `Grb`, `Rgbw`, `GrayA44` always available; MSRV 1.85 (0.8.53: 1.64).
- **Scope ([issue 142](https://github.com/kornelski/rust-rgb/issues/142)):**
  whole images are out of scope (single pixels or slices at most). Primaries
  and white-point definitions "could be okay"; shared YUV metadata enums are
  needed; HDR is "a hairy problem". Nothing was decided.

## How zenpixels uses rgb today

- `zenpixels` feature `rgb`: `Pixel` impls for `rgb::{Rgb, Rgba, Gray}` over
  `u8`/`u16`/`f32` and `rgb::alt::BGRA<u8>` (`zenpixels/src/buffer.rs:110-158`),
  typed `PixelBuffer<P>`/`PixelSlice<P>`, `from_pixels`, and `imgref`
  conversions. Dependency: `rgb = { version = "0.8", default-features = false,
  features = ["bytemuck"] }`.
- `zenpixels-convert` feature `rgb`: `PixelBufferConvertTypedExt::to_rgb8`,
  `to_rgba8`, `to_gray8`, `to_bgra8` return `PixelBuffer<rgb::…>`
  (`zenpixels-convert/src/ext.rs:200-209`). These are public signatures that
  name rgb types.
- Our own pixel structs: `GrayAlpha8/16/F32 { v, a }`, `Rgbx`, `Bgrx`.
  `GrayAlpha*` exist because rgb 0.8.53's `GrayAlpha` is a tuple struct.
  42 Rust files in 11 zen checkouts use them (jxl-encoder 16; zenpng,
  zenextras, zenjxl, zentiff, zensim).
- Every `Cargo.lock` under `~/work` that contains rgb (248) resolves `0.8.53`.
  No manifest enables `unstable-experimental` or requires an rgb prerelease.

## Mismatches

1. **`rgb::Gray` changes type in 1.0.** In 0.8.53 and 0.8.92, `rgb::Gray` is
   the tuple struct `Gray_v08(T)`; the named-field `Gray_v09 { v }` exists
   beside it. In 1.0, `Gray` becomes the named-field struct. So our
   `impl Pixel for Gray<…>` and `to_gray8() -> PixelBuffer<rgb::Gray<u8>>`
   will name a different type, and today nothing covers `Gray_v09`. The same
   swap already happens on 0.8.53 if any crate in the build enables rgb's
   `unstable-experimental` feature (none of ours does). zencodecs reads gray
   pixels through `.value()` (`zencodecs/src/decode.rs:799`), which exists on
   both structs, so that call site survives the swap.
2. **Duplicate gray+alpha types.** rgb's `GrayA<T, A = T> { v, a }` has the
   same fields and layout as our `GrayAlpha*`. We cannot add
   `impl Pixel for GrayA<u8>` against 0.8.53: there `GrayA` is `Pod` only
   when no crate in the build enables rgb's default `as-bytes` feature
   (`bytemuck_impl` is compiled only under `not(feature = "as-bytes")`, and
   `as_bytes.rs` has no `GrayA` impl). Tag `v0.8.54` (`9100e3c`) and `main`
   fix this; neither is published.
3. **Two traits named `Pixel`.** `rgb::prelude::*` exports rgb's `Pixel`
   (generic component access); zenpixels exports its `Pixel` (a type tagged
   with a `PixelDescriptor`). Glob-importing both is an ambiguity error at the
   first use of `Pixel`; explicit imports are fine.
4. **Legacy path and loose version.** We spell BGRA as `rgb::alt::BGRA`. The
   `alt` module is legacy and leaves in 1.0; `rgb::Bgra` is the same type
   (`alt.rs`: `pub use crate::formats::bgra::Bgra as BGRA`). The requirement
   `"0.8"` admits versions without root-level `Rgb`/`Bgra` (0.8.38 has only
   `RGB`) and without the `bytemuck` feature. The lowest unyanked version with
   both is 0.8.48 (0.8.40 to 0.8.47 are yanked); we only build against 0.8.53.
5. **Coverage differs.** rgb has `Bgr`, `Argb`, `Abgr`, `Grb`, `Rgbw`,
   `GrayA44` and mixed-type alpha, none of which has a zenpixels format.
   zenpixels has padded RGBX/BGRX (our own types), F16, Oklab, CMYK and
   `PixelFormat::Generic`, none of which has an rgb type. `Rgb<half::f16>`
   would work with `half`'s bytemuck support, but needs a new dependency and
   has no consumer.
6. **Meaning.** rgb is color-agnostic: "sRGB is generally assumed" for `RGB8`,
   native byte order for `RGB16`, and nothing about premultiplied alpha. Our
   typed impls attach `TransferFunction::Unknown` but assert BT.709 primaries,
   full range and straight alpha (`PixelDescriptor::RGB8` etc.). That is
   consistent, but stricter than rgb and not stated on the impls. Byte order
   matches: both are native.
7. **rgb's `From` conversions bypass our alpha policy.** `From<Rgba<T, A>> for
   Rgb<T>` (and the BGR/GRB/gray equivalents) drops alpha without compositing,
   although rgb's own `HasAlpha` docs warn against that. `ConvertOptions` /
   `AlphaPolicy` never see a user's `.into()`. rgb has no bit-depth conversions
   (its issue 46 is open); zenpixels-convert covers those.
8. **A public dependency on rgb 0.8.** `zenpixels` and `zenpixels-convert`
   name rgb 0.8 types in public impls and signatures. Moving them to rgb 1.0 is
   a breaking change for us unless 1.0's semver trick makes the 0.8 and 1.0
   types identical. How it does that, especially for `Gray`, is unknown until
   1.0 ships.

## Recommendations

Doc or internal changes (no public API change):

- Spell BGRA as `rgb::Bgra` in `buffer.rs` and `ext.rs`. It is the same type,
  so signatures do not change.
- Raise the rgb requirement in both crates to `0.8.53`, the only version we
  build against (global rule: full versions).
- Document what each typed impl assumes (transfer, primaries, range, alpha,
  native byte order), as rgb does on its aliases.
- Add `#[doc(alias = "RGBA8")]`-style aliases (and `Luma`, `Mono`) to the
  matching `PixelFormat` variants for search.

Needs a release of rgb, then a PR assigned to lilith (new trait impls are
public API):

- When 0.8.54 or 0.8.92 is published: `impl Pixel for Gray_v09<u8/u16/f32>`
  and `GrayA<u8/u16/f32>`, with the rgb floor raised to that version. Keep
  `GrayAlpha*`: 42 files use them and removal would be a break. Do not depend
  on `0.8.92-rc.1`; Cargo will not unify it with the 0.8.53 that every zen
  lockfile uses.

Decisions for the owner:

- **Our `Pixel` name.** 0.3.1 is the cheap window to rename it. Recommendation:
  keep it. The clash needs two glob imports, and a rename touches every typed
  consumer.
- **Typed conversions naming rgb types** (`to_gray8() -> PixelBuffer<rgb::Gray<u8>>`).
  Recommendation: leave them until rgb 1.0 shows how `Gray` is shared, then
  decide whether `to_gray8` follows 1.0's `Gray`.
- **The semver trick for our own 0.2 → 0.3.1.** `release-plan-0.2.17-and-0.3.1.md`
  §5.5 notes that a graph mixing 0.2 and 0.3 resolves two zenpixels crates
  whose types do not unify. rgb's answer to the same problem is a final old
  version that depends on the new one and re-exports every unchanged type
  (here: leaf enums such as `AlphaMode`, `ColorPrimaries`, `SignalRange`,
  `Cicp`, `Orientation`; not `PixelFormat`, `PixelDescriptor` or the buffers,
  which changed). The owner chose one core version per connected pipeline;
  this is a fallback if mixed graphs turn out to be common.
- **Upstream contribution.** kornelski is open to shared primaries and YUV
  metadata enums; ours (`ColorPrimaries`, `Cicp`) already cover that. Posting
  in rust-rgb needs the owner's explicit approval of the text.

Not worth adopting: mixed-type alpha, `HetPixel`/`map_colors`, per-pixel
arithmetic (rgb's ops also apply to alpha), `Rgbw`, `Grb`, `GrayA44`.
zenpixels works on whole rows in runtime formats, and none of these has a
consumer.

## Re-checking

```sh
# Published versions and yanks
curl -s -H 'User-Agent: zenpixels-rgb-study (github.com/imazen/zenpixels)' \
  'https://crates.io/api/v1/crates/rgb/versions?per_page=100' \
  | python3 -c 'import json,sys; [print(v["num"], v["created_at"][:10], "YANKED" if v["yanked"] else "") for v in json.load(sys.stdin)["versions"][:12]]'
# rgb source (tags, Pod gating, Gray aliases)
git clone https://github.com/kornelski/rust-rgb ~/tmp/rgb-study/rust-rgb
git -C ~/tmp/rgb-study/rust-rgb show v0.8.53:src/lib.rs | grep -n 'as-bytes\|unstable-experimental'
# rgb versions resolved across our checkouts
rg -l --glob '!**/target/**' -g Cargo.lock 'name = "rgb"' ~/work \
  | xargs awk '/^name = "rgb"$/{getline; print $NF}' | sort | uniq -c
# GrayAlpha* users outside zenpixels
rg -l --glob '!**/target/**' --glob '!**/.jj/**' --glob '!zen/zenpixels*/**' -t rust \
  'GrayAlpha(8|16|F32)\b' ~/work
```
