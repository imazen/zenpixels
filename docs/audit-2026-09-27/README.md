# Migration inventory snapshot

Generated with [audit-pixel-migration.py](../../scripts/audit-pixel-migration.py):

```sh
python scripts/audit-pixel-migration.py \
  --root /home/lilith/work \
  --out /tmp/zenpixels-migration-audit-2026-09-27
```

This is **lexical discovery, not Rust type resolution or a build matrix**.
See the [reviewed migration ledger](../migration-examples-and-audit-0.3.1.md#reviewed-consumer-ledger)
for dispositions supported by reading receivers and surrounding code.

- `scope.json`: searched roots, exclusions and patterns.
- `summary.json`: counts for the complete scan, including alternate checkouts.
- `repos.tsv`: counts by nearest repository; incidental nested repositories can
  have matches without being discovery roots.
- `primary-manifests.tsv`: 149 declarations in primary-classified checkouts,
  including dev dependencies, aliases and workspace declarations. These are not
  149 distinct dependent crates and are not published dependency evidence.
- `boundary-candidates.tsv`: 611 source lines near public signatures/re-exports,
  including zenpixels itself. The heuristic includes `pub(crate)` and unrelated
  same-name types; it can miss trait signatures, inferred values and aliases.
- `errors.json`: three TOML parse failures in alternate zensim checkouts. Other
  discovered source in those repositories remains scanned.

The full `hits.tsv`, `primary-focused.tsv` and original `rg-hits.jsonl` are in
the output directory. They can be regenerated; the large raw inventory is not
checked in. The focused file contains 14,227 candidate lines across 746 files.
The broad 168,127-line count includes generic `.clone()` and `.compose()` calls,
comments, tests, duplicate checkouts and unrelated types. It is emphatically not
a count of zenpixels migration sites.

“Primary” is a filesystem heuristic, not a declaration of branch authority.
For example `_zensim-pu-panel` and `zensim-featcompress` are duplicated work.
No symlinks are followed. Generated/build/dependency trees listed in `scope.json`
are excluded. Arbitrarily renamed re-exports and generated code require compiler
checks after migration; regex alone cannot prove that every boundary was found.

Snapshot taken 2026-09-27 against local working trees; source line numbers may
move. No dependent repositories were modified by this audit.
