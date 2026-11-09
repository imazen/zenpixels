# ICC authority during grayscale collapse

A context CICP hint or embedded tag cannot authorize replacing an ICC's
transform. Equal stored R/G/B samples alone do not prove that another
profile's grayscale interpretation is equivalent. The shared collapse planner
now requires `MatrixTrcSubstitution` identification and agreement with any
explicit hint. Otherwise RGB and the original ICC remain intact; independent
alpha or depth reductions may still proceed.

The established fallback for truncated metadata is retained: a buffer shorter
than the 128-byte ICC header cannot contain a valid ICC transform, and may
still use its explicit CICP hint. Every existing assertion and fixture stays
unchanged. The separate stricter malformed-input candidate is preserved;
this fix does not require adopting that policy.

The original function fails both new regressions: a real metadata-only PQ
export and a valid Display-P3 ICC accompanied by a contradictory gamma-2.2
hint are replaced with GRAY ICCs. It passes the other 61 controls. The fixed
function passes all 63. New regressions exercise strided rows and in-place
reduction, exact samples and original ICC retention. Existing qualified-profile,
CICP-only and malformed-input positive controls remain unchanged.

An independent C LCMS 2.20 witness at code 128 shows the consequence of the
contradictory-hint bug: source P3 linear Y is 0.21585828065872192; the incorrect
gamma-2.2 gray replacement gives 0.2195202112197876 (difference
0.003661930561065674). The source and output hashes and original failures are
retained in the qualification evidence. No pixel tolerance is relaxed.

`just icc-collapse-check` also tests an isolated normal dependency with no
`icc-db` or CMS backend. Its positive control requires `NeedsCms` from gray
synthesis; the four consumer controls pass. This isolation matters: the main
crate's self dev-dependency previously re-enabled defaults during unit tests.
The established self dev-dependency configuration is retained. An isolated
experiment disables its defaults: 60 focused controls and strict Clippy pass,
but the full library run is 429 passed / 10 failed / 1 existing ignore. Each
failure requests a bundled profile while the database is absent; the result
is `NeedsCms`. These assertions are preserved, not relaxed or skipped. The
alternative source and failing receipts remain available separately. The
standalone consumer instead tests the supported normal dependency with no
std, database or CMS backend enabled.
The focused CMS workflow runs the authority and isolated consumer controls on
Linux x64/ARM, Windows x64/ARM, macOS ARM and i686.

Local gates: 63 focused controls, 4 genuinely isolated consumer controls,
strict all-target converter Clippy and strict consumer Clippy pass. The final
restored-source controls record 14 seconds / 0.48 GiB and 5 seconds / 0.31 GiB
peak RSS; these are correctness checks, not performance comparisons.

The isolated unit experiment also exposed three test helpers whose callers
already require icc-db. Their definitions now use that same feature gate;
no test is skipped and no assertion changes. Its strict all-target Clippy
passes (13 seconds / 0.35 GiB); the 10 full-run failures remain explicit.
