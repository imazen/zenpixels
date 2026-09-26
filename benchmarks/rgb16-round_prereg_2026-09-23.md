# RGB16 narrowing preregistration — 2026-09-23

This is a deterministic code correctness study. No human labels, sample selection, bootstrap CI, or random seeds apply.

## Inputs and decision rule

- Base: `main@origin` at workspace creation (`e56f626b`). The committed source tree is the input; its commit id is the content hash.
- Enumerate every `u16` code 0 through 65535 through public `RowConverter` paths, for RGB, RGBA, Gray and GrayAlpha, including a strided row.
- The oracle for same-transfer, same-primaries `u16 -> u8` is `(v * 255 + 32767) / 65535` in `u32` arithmetic. No tolerance is allowed.
- Compare every available archmage tier with scalar byte for byte. Count mismatches by route and tier.
- For real transfer changes, compare against a separately evaluated high precision composite transfer oracle where available. State any oracle limitation explicitly.
- Run the existing zenbench kernel before and after at 64², 256², 1024², and 4096²; report timing and fit of `T = alpha + beta * pixels`. Label contention based on machine load.
- Accept the fix only if same-transfer paths have zero errors for all codes, strided rows agree, all tiers agree, and the workspace test and clippy gates pass.
