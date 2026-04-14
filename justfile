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

# Build the table generator (unpublished workspace crate)
icc-build-gen:
    cargo build -p icc-gen --release

# Regenerate .inc table files from ICC profile cache + bundled profiles
icc-gen: icc-build-gen icc-ensure-sources
    ./target/release/icc-gen "{{icc_cache}}" "zenpixels-convert/src/profiles" "{{icc_out}}"

# Full pipeline: fetch profiles, regenerate tables, test
icc-update: icc-fetch icc-gen test
    @echo "ICC tables updated and tests pass."

# Show what the generator would produce without writing (dry run)
icc-dry-run: icc-build-gen
    mkdir -p /tmp/zenpixels-icc-dry-run
    ./target/release/icc-gen "{{icc_cache}}" "zenpixels-convert/src/profiles" /tmp/zenpixels-icc-dry-run
    @echo "--- RGB ---"
    @head -5 /tmp/zenpixels-icc-dry-run/icc_table_rgb.inc
    @echo "..."
    @tail -3 /tmp/zenpixels-icc-dry-run/icc_table_rgb.inc
    @echo "--- Gray ---"
    @cat /tmp/zenpixels-icc-dry-run/icc_table_gray.inc

# ── Build-time diagnostics ────────────────────────────────────────────
# Cold release wall-time, three runs. Reports min/avg.
build-bench n="3" args="":
    #!/usr/bin/env bash
    set -euo pipefail
    times=()
    for i in $(seq 1 {{n}}); do
        cargo clean -q
        t=$(/usr/bin/time -f "%e" cargo build --release {{args}} -q 2>&1 | tail -1)
        echo "  run $i: ${t}s"
        times+=("$t")
    done
    python3 -c "import sys; t=[float(x) for x in sys.argv[1:]]; print(f'min={min(t):.2f}s  avg={sum(t)/len(t):.2f}s  n={len(t)}')" "${times[@]}"

# Generate cargo build --timings HTML and print top units by duration.
build-timings args="":
    #!/usr/bin/env bash
    set -euo pipefail
    cargo clean -q
    cargo build --release --timings {{args}} 2>&1 | tail -2
    awk '/const UNIT_DATA = \[/,/^\];$/' target/cargo-timings/cargo-timing.html \
        | python3 -c "
    import sys, json
    text = sys.stdin.read().replace('const UNIT_DATA = ', '').rstrip().rstrip(';')
    data = sorted(json.loads(text), key=lambda u: -u['duration'])
    print(f'{\"crate\":36} {\"target\":18} {\"start\":>6} {\"end\":>6} {\"dur\":>6}')
    for u in data[:20]:
        tgt = (u.get('target') or 'lib').strip()[:17]
        print(f'{u[\"name\"][:35]:36} {tgt:18} {u[\"start\"]:6.2f} {u[\"start\"]+u[\"duration\"]:6.2f} {u[\"duration\"]:6.2f}')
    print(f'Total units: {len(data)}, total CPU-sec: {sum(u[\"duration\"] for u in data):.1f}')
    "
    echo "Open target/cargo-timings/cargo-timing.html for full Gantt."

# Per-crate .text contribution to the cdylib bench (uses cargo-bloat).
build-bloat args="":
    cargo bloat --release -p cdylib-link-bench --crates -n 15 {{args}}

# Top symbols in the cdylib bench.
build-bloat-symbols args="":
    cargo bloat --release -p cdylib-link-bench -n 25 {{args}}

# Cold cdylib build/link wall-time, three runs.
build-cdylib-bench n="3" args="":
    just build-bench {{n}} "-p cdylib-link-bench {{args}}"
