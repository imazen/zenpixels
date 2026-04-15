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
