# Review evidence

Resolve the artifact root from `MOXCMS_REVIEW_ARTIFACTS`; the producing host retained
this run at `/mnt/v/output/zenpixels/moxcms-review-20261106/`. The SHA256SUMS.json
manifest identifies every retained raw file. Artifact paths are observations
from 2026-11-06, not a promise that a checkout contains them.

The original confirmation population is rejected and retained under
gamma-review-data. Complete evaluator disassembly and 27 Callgrind traces are
under gamma-review-instructions-v2. The timing binary SHA-256 is
`d5d2a9cf0719a31b6f4b75bd771dfbe21ce9c7d164e65cbc6d2112e8b32484c9`.


The qualified ARM population is in `arm/cell-study`: `study.json` records the
predeclared 36 commands, source binary hash and each evidence-object hash.
Rejected ARM populations remain in `arm/data` and `arm/data-unpolled`.
Full assembly is in `gamma-review-instructions-v2/before.asm`, `branch.asm`,
and `guarded.asm`. Reconstruct the CSV with `summarize-review-cells.py`.
