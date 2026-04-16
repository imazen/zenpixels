## Public API surface — YAGNI is the rule, not a suggestion

**Every public item in zenpixels is forever. Treat each `pub` as a load-bearing external commitment.**

zenpixels is the foundational crate of the zen ecosystem — zencodec, zenpipe, imageflow, and every codec depend on it directly or transitively. Once a type, function, method, field, enum variant, trait method, or feature flag is `pub`, removing or renaming it requires a 0.3.0 bump (see below), and that break ripples through every downstream. We have deliberately chosen to avoid 0.3.0 indefinitely. The practical consequence: **public items added speculatively cannot be taken back.**

### Rules for new public API

1. **No `pub` without a concrete current consumer in this repo or a sibling zen crate.** "Someone might want this" is not a reason. If the hypothetical caller shows up later, they can file a request and we can add it *then* — adding is cheap, removing is impossible.
2. **Default to `pub(crate)`.** Promote to `pub` only after a concrete caller is merging the code that uses it.
3. **Builder-style APIs over struct literals** wherever possible. `StructName::new()` + `.with_foo()` lets us add fields without the struct becoming a perma-frozen shape. Avoid public structs with public fields unless they are genuinely trivial data bags (e.g., `Rect`, `Size`).
4. **Enums should be `#[non_exhaustive]` from the first release** unless we are *absolutely* sure the variant set is fixed (e.g., `Matrix::Bt601 | Bt709 | Bt2020` — the color-space matrix set is standardized; never will be a new one there). When in doubt: `#[non_exhaustive]`.
5. **Traits should be sealed** (`pub trait Foo: private::Sealed`) unless external implementation is a required feature. Every `impl Foo for MyType` by an external crate turns a supertrait bound addition into a breaking change.
6. **Feature flags are part of the API.** Once shipped, removing a feature flag is a break (see tolerated-break #4 below — feature removal is *tolerated* only when nobody sets it). Don't add feature flags speculatively.
7. **Re-exports are commitments too.** `pub use` of an internal or third-party type pins us to that name and that dependency version. Re-export only items downstream code will reach for frequently.
8. **"Helper" / "convenience" / "shortcut" methods get the strictest scrutiny.** These are the items most commonly added speculatively and most commonly unused. If a user can construct the behavior in 2-3 lines with existing API, do not add a helper. Examples of things to NOT add: `foo_or_default()`, `foo_with_options_fallback()`, `is_probably_bar()`.

### Before adding any `pub` item, you must be able to answer

- Who calls this *today* (file:line references)?
- If removed tomorrow, what concrete code breaks?
- Would a caller who wanted this synthesize it from existing API? If yes, reject the addition.

If you cannot answer with a concrete, verified current consumer: the item stays `pub(crate)` or is not added at all.

### Before every release

Audit `src/lib.rs` for items added since the last version. For each new `pub`, verify a consumer exists. If a consumer cannot be found (including `builtin_profiles`-style "positioned for future use" modules), either:
- Find a real caller and migrate them (simultaneously), OR
- Demote to `pub(crate)` before publishing.

The `builtin_profiles` module (currently in `[Unreleased]`) is a live example: it exposes 8 public items (enum + const + 6 functions) via `pub mod builtin_profiles` and has zero consumers outside its own tests. Before the next publish, demote to `pub(crate)` or find a real caller. Catching this *before* it ships is the whole point.

---

## 0.2.x versioning policy — "tolerated technical breaks"

**Do NOT version-gate on `cargo semver-checks` output alone for this repo in 0.2.x.**

We deliberately ship certain narrow semver-breaking changes as `0.2.N` bumps rather than forcing a `0.3.0` release. A 0.3.0 break is expensive — it ripples through every zen* sibling (zencodec, zenpipe, imageflow) and forces downstream dependency-graph churn. When the breakage is technical but unlikely to bite real callers, we accept it.

### Tolerated breaks

Changes that may ship inside a `0.2.N` patch release:

1. **Adding `#[non_exhaustive]` to structs / enums** that nobody constructs via struct-literal in practice (verified by grepping `~/work/zen/`). Breaks theoretical external callers; no known victims.
2. **Adding fields to a non-`#[non_exhaustive]` struct**, provided all in-tree callers are migrated to builder / `::new` / preset constructors simultaneously. External struct-literal construction breaks.
3. **Adding auto-trait supertraits** (e.g., `Send + Sync` to a trait that was `Send`-only) when every in-tree impl already satisfies them. `cargo semver-checks` flags this as a major break; we accept it.
4. **Removing a Cargo feature** when the feature was either experimental (no release had it in default) or has been folded into the default behavior (e.g., `zencms-lite` was dropped when its functionality became unconditional via `OnceBox`).
5. **Losing auto-trait impls as a mechanical consequence** of new private fields (e.g., `RowConverter` lost `Sync` because it now holds `Box<dyn RowTransformMut: Send>`). Acceptable when the type's primary use is `&mut self` and cross-thread shared access was never meaningful.
6. **Dropping a derive like `Debug`** when the type's fields are not meaningfully Debug-able and no callers rely on the default derive. Replace with a curated manual impl or a `_opaque: "<redacted>"` placeholder if needed.

### Not tolerated in 0.2.N

These REQUIRE a 0.3.0 bump:
- Removing public items (types, functions, methods, fields, variants)
- Changing method signatures (argument types, return types, generic bounds that affect callers)
- Renaming public items
- Changing semantic behavior silently (e.g., default flag values)
- Any change that requires known callers to modify source to keep compiling

### Workflow for tolerated breaks

1. Before shipping, grep `~/work/zen/` for external struct-literal / trait-impl / feature-name uses. Document the audit in the commit message.
2. Update all in-tree callers in the same commit (or sibling repo PR) so sibling crates keep building.
3. CHANGELOG entry must be under **`#### Changed (BREAKING, tolerated in 0.2.x)`** heading so readers can see the technical semver status.
4. `cargo semver-checks` failures listed under tolerated-break categories above are acceptable; failures outside them block the release.
5. Ship. If real external breakage is reported, the next release bumps to 0.3.0 and we move the change out of the tolerated bucket.

### Verifying before a release

When preparing to publish:
- Run `cargo semver-checks --workspace` and inspect every failure.
- For each failure, categorize: **tolerated** (matches the list above, document in commit) or **not tolerated** (requires 0.3.0 bump, stop the release).
- If *any* failure is not tolerated, bump to 0.3.0 and cut a proper breaking release.

See also: `CHANGELOG.md` — the 0.3.0 queue section accumulates non-tolerated breaks for batched release.
