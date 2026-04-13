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

# Fetch ICC profiles from R2 to local cache using the manifest
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
    # Download manifest first
    aws s3 cp "s3://{{r2_bucket}}/{{r2_prefix}}MANIFEST.txt" "{{icc_cache}}/MANIFEST.txt" \
        --endpoint-url "$ENDPOINT" --no-progress 2>/dev/null || true
    # Sync all profiles
    echo "Syncing ICC profiles from R2 → {{icc_cache}} ..."
    aws s3 sync "s3://{{r2_bucket}}/{{r2_prefix}}" "{{icc_cache}}/" \
        --endpoint-url "$ENDPOINT" --no-progress
    TOTAL=$(find "{{icc_cache}}" -name '*.icc' -o -name '*.icm' | wc -l)
    echo "Done: $TOTAL profiles in cache"
    # Verify against manifest if present
    if [ -f "{{icc_cache}}/MANIFEST.txt" ]; then
        EXPECTED=$(wc -l < "{{icc_cache}}/MANIFEST.txt")
        echo "Manifest expects $EXPECTED, have $TOTAL"
        if [ "$TOTAL" -lt "$EXPECTED" ]; then
            echo "WARNING: fewer profiles than manifest — some may be missing"
        fi
    fi

# Upload ICC profiles to R2 from a local directory and update manifest
icc-upload dir:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Uploading ICC profiles from {{dir}} → R2 ({{r2_bucket}}/{{r2_prefix}})..."
    for f in "{{dir}}"/*.icc "{{dir}}"/*.icm; do
        [ -f "$f" ] || continue
        name=$(basename "$f")
        npx wrangler r2 object put "{{r2_bucket}}/{{r2_prefix}}$name" --file "$f" --content-type application/octet-stream 2>/dev/null
        echo "  uploaded: $name"
    done
    # Rebuild manifest: list all objects in the prefix (handles spaces in names)
    echo "Rebuilding manifest..."
    ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
        aws s3 ls "s3://{{r2_bucket}}/{{r2_prefix}}" --endpoint-url "$ENDPOINT" 2>/dev/null \
        | awk '{ for (i=4; i<=NF; i++) printf "%s%s", $i, (i==NF?"\n":" ") }' \
        | grep -E '\.(icc|icm)$' | sort > /tmp/icc-manifest-update.txt
    npx wrangler r2 object put "{{r2_bucket}}/{{r2_prefix}}MANIFEST.txt" \
        --file /tmp/icc-manifest-update.txt --content-type text/plain 2>/dev/null
    echo "Done. $(wc -l < /tmp/icc-manifest-update.txt) profiles in manifest."

# Ensure well-known ICC profile collections are in the local cache
icc-ensure-sources:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "{{icc_cache}}"
    # Compact-ICC-Profiles (saucecontrol)
    COMPACT="/tmp/compact-icc-profiles"
    if [ ! -d "$COMPACT/profiles" ]; then
        echo "Cloning Compact-ICC-Profiles..."
        git clone --depth 1 https://github.com/saucecontrol/Compact-ICC-Profiles.git "$COMPACT"
    fi
    cp -n "$COMPACT"/profiles/*.icc "{{icc_cache}}/" 2>/dev/null || true
    # moxcms test profiles (awxkee)
    MOXCMS="/tmp/moxcms-profiles"
    if [ ! -d "$MOXCMS/assets" ]; then
        echo "Cloning moxcms assets..."
        git clone --depth 1 --filter=blob:none --sparse https://github.com/awxkee/moxcms.git "$MOXCMS"
        git -C "$MOXCMS" sparse-checkout set assets
    fi
    cp -n "$MOXCMS"/assets/*.icc "$MOXCMS"/assets/*.icm "{{icc_cache}}/" 2>/dev/null || true
    echo "ICC cache: $(find {{icc_cache}} -name '*.icc' -o -name '*.icm' | wc -l) profiles"

# Regenerate .inc table files from ICC profile cache + bundled profiles
icc-gen: icc-build-gen icc-ensure-sources
    /tmp/zenpixels-gen-icc-tables "{{icc_cache}}" "zenpixels-convert/src/profiles" "{{icc_out}}"

# Build the table generator
icc-build-gen:
    rustc -O scripts/gen_icc_tables.rs -o /tmp/zenpixels-gen-icc-tables

# Full pipeline: fetch profiles, regenerate tables, test
icc-update: icc-fetch icc-gen test
    @echo "ICC tables updated and tests pass."

# Show what the generator would produce without writing (dry run)
icc-dry-run: icc-build-gen
    /tmp/zenpixels-gen-icc-tables "{{icc_cache}}" "zenpixels-convert/src/profiles" /tmp/zenpixels-icc-dry-run
    @echo "--- RGB ---"
    @head -5 /tmp/zenpixels-icc-dry-run/icc_table_rgb.inc
    @echo "..."
    @tail -3 /tmp/zenpixels-icc-dry-run/icc_table_rgb.inc
    @echo "--- Gray ---"
    @cat /tmp/zenpixels-icc-dry-run/icc_table_gray.inc
