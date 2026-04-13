# zenpixels justfile

icc_cache := env("HOME") / ".cache/zenpixels-icc"
icc_out := "zenpixels/src/icc"
r2_bucket := "codec-corpus"
r2_prefix := "icc-profiles/"

# Run all checks (fmt, clippy, test)
ci: fmt clippy test

# Format
fmt:
    cargo fmt --check

# Clippy
clippy:
    cargo clippy --workspace -- -D warnings

# Test all packages
test:
    cargo test --workspace

# ── ICC profile table management ──────────────────────────────────────

# Fetch ICC profiles from R2 to local cache
icc-fetch:
    #!/usr/bin/env bash
    set -euo pipefail
    : "${R2_ACCOUNT_ID:?Set R2_ACCOUNT_ID}"
    : "${R2_ACCESS_KEY_ID:?Set R2_ACCESS_KEY_ID}"
    : "${R2_SECRET_ACCESS_KEY:?Set R2_SECRET_ACCESS_KEY}"
    mkdir -p "{{icc_cache}}"
    ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
    export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
    echo "Syncing ICC profiles from R2 → {{icc_cache}} ..."
    aws s3 sync "s3://{{r2_bucket}}/{{r2_prefix}}" "{{icc_cache}}/" \
        --endpoint-url "$ENDPOINT" --no-progress
    echo "Done: $(find "{{icc_cache}}" -name '*.icc' | wc -l) profiles"

# Upload ICC profiles to R2 from a local directory
icc-upload dir:
    #!/usr/bin/env bash
    set -euo pipefail
    : "${R2_ACCOUNT_ID:?Set R2_ACCOUNT_ID}"
    : "${R2_ACCESS_KEY_ID:?Set R2_ACCESS_KEY_ID}"
    : "${R2_SECRET_ACCESS_KEY:?Set R2_SECRET_ACCESS_KEY}"
    ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
    export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
    echo "Uploading ICC profiles from {{dir}} → R2 ..."
    aws s3 sync "{{dir}}/" "s3://{{r2_bucket}}/{{r2_prefix}}" \
        --endpoint-url "$ENDPOINT" --no-progress
    echo "Done."

# Regenerate .inc table files from ICC profile cache
icc-gen: icc-build-gen
    /tmp/zenpixels-gen-icc-tables "{{icc_cache}}" "{{icc_out}}"

# Build the table generator
icc-build-gen:
    rustc -O scripts/gen_icc_tables.rs -o /tmp/zenpixels-gen-icc-tables

# Full pipeline: fetch profiles, regenerate tables, test
icc-update: icc-fetch icc-gen test
    @echo "ICC tables updated and tests pass."

# Show what the generator would produce without writing (dry run)
icc-dry-run: icc-build-gen
    /tmp/zenpixels-gen-icc-tables "{{icc_cache}}" /tmp/zenpixels-icc-dry-run
    @echo "--- RGB ---"
    @head -5 /tmp/zenpixels-icc-dry-run/icc_table_rgb.inc
    @echo "..."
    @tail -3 /tmp/zenpixels-icc-dry-run/icc_table_rgb.inc
    @echo "--- Gray ---"
    @cat /tmp/zenpixels-icc-dry-run/icc_table_gray.inc
