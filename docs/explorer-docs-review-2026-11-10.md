# Explorer and documentation accuracy pass

Source baseline: merged main `76005fc9`. The owner requested a readable,
accurate guide and an explorer that fits its viewport. A PixelSlice must be
explained as a borrowed image/subimage view retaining parent stride.

## Verified problems and corrections

- The published page expanded to 1111 pixels at a 360-pixel viewport with
  advanced panels open and RgbaF64 selected. Intrinsic grid tracks, wide byte
  groups and control minima widened the document. Containers now shrink;
  complete wide byte groups and tables scroll locally instead of being hidden.
- The glossary required whole-pixel byte strides. Erased byte views actually
  require sample alignment; a crop retains its parent's stride. The guide
  demonstrates an eight-byte RGB8 pitch, a short final row and a cropped view's
  original pointer/stride. Typed constructors retain their own alignment rules.
- Copied Rust omitted Unknown declarations and non-default generic alpha,
  allowing format defaults to disagree with the selected controls. It now emits
  those choices explicitly. The browser exports the actual snippets for Rust
  compilation/execution, including the two regression cases.
- RGB565 was described as requiring a prior separate unpack. Existing
  CodePlane views can select its three shared-word bitfields. The live Rust
  example extracts them and runs to_canonical directly into RGB8.
- Library/README text described a flat format enum, universal aligned owners,
  a different allocation strategy and unqualified allocation-free operations.
  Current source uses named/generic formats, packed default rows, explicit
  aligned owners and scoped fallibility. The panicking constructors call the
  fallible ones; the old speed explanation was unsupported.
- Historical review-branch links and closed issues looked current. The dated
  review now uses main links and the current precision successor, 154. The
  recommendation model is explicitly historical. Unresolved original carriers
  are distinguished from ICC-first current ColorContext selection.

## Learning path

The [guide](guide.md) starts with PixelBuffer, PixelSlice and PixelDescriptor.
It wraps decoded samples, borrows/crops an image without copying, explains
stride, writes through a mutable view, then introduces actual conversion.
Concrete Rust pixels, native codes and generic wide storage come later.
The source remains one Markdown document rendered by rustdoc for the site.

Both READMEs and crate-level documentation point to that path. They keep
ownership/color declarations separate from conversion and leave detailed
contracts in [the reference](design-reference.md). No additional foundational
type or convenience API is introduced.

## Checks and limits

The WASM format catalog uses actual PixelFormat construction and descriptor
facts. Browser checks compare all displayed entries against it. The source
factory list covers this explorer's formats; this is not reflection over every
future enum variant.

The browser checks all displayed format/alignment settings and route pairs,
plus concrete swizzle, strided U16, padding, native extraction, RGB565 and HDR
policy witnesses. Main document width is checked at 320, 360, 390, 768, 1024 and
1440 pixels; linked teaching pages are checked at 320, 390, 768 and 1440.
Missing WASM retains the teaching model. No existing expectation was weakened.

Guide/README code and copied explorer code compile and execute through
`just guide-check`. These checks establish API usage and their assertions;
they do not qualify the JavaScript model as an independent pixel oracle.
Current source/metadata and numerical limitations remain in the shared reference.

Reproduce with `just site-build`, `just site-check URL`, `just guide-check PATH`
and `node --test docs/color-explorer/model.test.cjs`. The Pages workflow runs
the browser-to-Rust example check. Screenshots are bounded, opt-in artifacts;
generated WASM, screenshots and exported example packets are not committed.
