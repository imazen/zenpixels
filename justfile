# zenpixels justfile

icc_cache := env("HOME") / ".cache/zenpixels-icc"
icc_out := "zenpixels/src/icc"
r2_bucket := "codec-corpus"
r2_prefix := "icc-profiles/"

# Run all checks (fmt, clippy, test)
ci: fmt clippy test

# The rustdoc-JSON nightly for the public-API snapshots is PINNED (keep in
# sync with the `api-doc` job in .github/workflows/ci.yml). An unpinned
# tracking nightly churns the rendered surface with zero repo changes: the
# 2026-08-26 nightly started rendering `PixelBuffer<P>`'s (unconditional,
# PhantomData-only) `Freeze` impl as conditional on Linux x86_64 while the
# same nightly on macOS aarch64 did not, so the committed snapshots could
# not be current on both sides at once. The harness auto-installs the pin
# via rustup. Bump deliberately, regenerating the snapshots in the same
# commit.
apidoc_toolchain := "nightly-2026-07-13"

# Format + regenerate the public-API surface snapshots (docs/public-api/).
# The snapshot runner lives in the workspace-excluded apidoc/ package, so it
# is never built or run by plain `cargo test` or any CI job.
fmt:
    cargo fmt --check
    ZEN_API_DOC_TOOLCHAIN={{apidoc_toolchain}} cargo test --manifest-path apidoc/Cargo.toml

# Regenerate the public-API surface snapshots only
api-doc:
    ZEN_API_DOC_TOOLCHAIN={{apidoc_toolchain}} cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current
api-doc-check:
    ZEN_API_DOC=check ZEN_API_DOC_TOOLCHAIN={{apidoc_toolchain}} cargo test --manifest-path apidoc/Cargo.toml

# Clippy
clippy:
    cargo clippy --workspace -- -D warnings

# Test all packages
test:
    cargo test --workspace

# Native TIFF/IMIJ sample carriage and explicit conversion refusal.
native-samples-check:
    cargo test -p zenpixels --test parts
    cargo test -p zenpixels -p zenpixels-convert --test native_samples --features zenpixels-convert/hdr-experimental,zenpixels-convert/cms-moxcms
    cargo clippy -p zenpixels -p zenpixels-convert --lib --test native_samples --features zenpixels-convert/hdr-experimental,zenpixels-convert/cms-moxcms -- -D warnings

# Full source coverage after extending the physical format enum.
native-samples-suite:
    cargo test -p zenpixels -p zenpixels-convert --lib --tests --features zenpixels-convert/hdr-experimental,zenpixels-convert/cms-moxcms
    cargo check -p zenpixels --no-default-features --target wasm32-unknown-unknown
    just api-doc

# Compare production fused scan predicates with forced scalar on macOS ARM64.
arm-scan-tiers-macos:
    mkdir -p "$HOME/tmp"
    CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4 TMPDIR="$HOME/tmp" nice -n 19 cargo bench -p zenpixels-convert --features __bench_scan --bench kernel_tiers -- --cpu-policy=unrestricted --format=llm > "$HOME/tmp/zenpixels-scan-tiers.log" 2>&1

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

# ── CICP→ICC bundle (full-coverage compressed blob) ───────────────────
#
# Regenerates the bundled, transfer-grouped LZ4 ICC blob (pure-Rust lz4_flex)
# that gives `synthesize_icc_for_cicp` full H.273-grid coverage with no CMS.
# moxcms is used here (build-time) only; nothing in the normal build runs this.
# Run after a moxcms version bump (the golden sha256 test flags drift), then
# update the pinned hash in zenpixels-convert/src/icc_profiles/cicp_bundle.rs.

# Build the CICP bundle generator (unpublished workspace bin)
cicp-bundle-build:
    cargo build -p icc-gen --release --bin cicp_bundle_gen

# Regenerate src/profiles/cicp_bundle.lz4 + the generated index module
cicp-bundle-gen: cicp-bundle-build
    ./target/release/cicp_bundle_gen
    @echo "Regenerated. If the bytes changed, update the golden sha256 in"
    @echo "  zenpixels-convert/src/icc_profiles/cicp_bundle.rs"
    @echo "  (sha256sum zenpixels-convert/src/profiles/cicp_bundle.lz4)"

# Show what the generator would produce without writing any files (dry run)
cicp-bundle-dry-run: cicp-bundle-build
    ./target/release/cicp_bundle_gen --no-write

# Regenerate the bundle and confirm it round-trips byte-for-byte against moxcms
cicp-bundle-update: cicp-bundle-gen
    cargo test -p zenpixels-convert --features cms-moxcms cicp_bundle
    cargo test -p icc-gen --release --bin cicp_bundle_gen
    @echo "CICP bundle regenerated and verified (golden sha256 + moxcms byte-equality)."

# Recheck the Ultra HDR corpus profile using both moxcms and Little CMS.
icc-hdr-profile-check input output:
    cargo run -p icc-gen --features lcms2-crosscheck --bin icc-gen -- --report {{input}} {{output}}
    cargo test -p zenpixels --features icc icc::

# Check compatibility after admitting the current archmage/magetypes beta.
beta-cohort-check:
    cargo fmt --check
    cargo clippy --locked --workspace --lib -- -D warnings
    cargo test --locked --workspace --lib --no-fail-fast

# Prepared execution must preserve ordinary conversion pixels and row allocation contracts.
prepared-contracts:
    cargo test -p zenpixels-convert --test prepared_contracts --features std,cms-moxcms,icc-db --locked
    cargo clippy -p zenpixels-convert --lib --test prepared_contracts --features std,cms-moxcms,icc-db --locked -- -D warnings

prepared-sync export:
    test ! -e "{{export}}/.git"
    test ! -e "{{export}}/.jj"
    cp justfile "{{export}}/justfile"
    cp zenpixels-convert/src/convert.rs zenpixels-convert/src/converter.rs zenpixels-convert/src/output.rs zenpixels-convert/src/error.rs zenpixels-convert/src/lib.rs "{{export}}/zenpixels-convert/src/"
    cp zenpixels-convert/tests/prepared_contracts.rs "{{export}}/zenpixels-convert/tests/"

# Combine native sample carriage with the existing profile-aware output planner.
native-output-sync export:
    test ! -e "{{export}}/.git"
    test ! -e "{{export}}/.jj"
    cp Cargo.toml justfile "{{export}}/"
    cp zenpixels/Cargo.toml "{{export}}/zenpixels/"
    cp zenpixels-convert/Cargo.toml "{{export}}/zenpixels-convert/"
    rsync -a zenpixels/src/ "{{export}}/zenpixels/src/"
    rsync -a zenpixels/tests/ "{{export}}/zenpixels/tests/"
    rsync -a zenpixels-convert/src/ "{{export}}/zenpixels-convert/src/"
    rsync -a zenpixels-convert/tests/ "{{export}}/zenpixels-convert/tests/"

native-output-check:
    cargo fmt -p zenpixels -p zenpixels-convert --check
    just output-plan-check
    just native-samples-check

output-borrow-compat:
    cargo test -p zenpixels-convert --doc --features std,cms-moxcms,icc-db --locked
    cargo check -p zenpixels-convert --no-default-features --target wasm32-unknown-unknown --locked
    RUSTDOCFLAGS="-D warnings" cargo doc -p zenpixels-convert --no-deps --features std,cms-moxcms,icc-db --locked
    just api-doc

# Output planning, existing atomic conveniences and actual ICC/backend contracts.
output-plan-check:
    cargo test -p zenpixels-convert --test output_native --test output_plan --test output_current_color --test output_contracts --test output_finalize --test unspecified_cicp_context --test prepared_contracts --features std,cms-moxcms,icc-db --locked
    cargo clippy -p zenpixels-convert --lib --test output_native --test output_plan --features std,cms-moxcms,icc-db --locked -- -D warnings

# Native ICC consumer regression: format-only alpha addition needs no CMS.
output-alpha-check:
    cargo fmt -p zenpixels-convert
    just output-plan-check
    cargo test -p zenpixels-convert --lib --features std,cms-moxcms,icc-db --locked
    cargo check -p zenpixels-convert --no-default-features --target wasm32-unknown-unknown --locked

# Float ICC precision, selected range behavior and explicit backend refusals.
output-float-check:
    cargo fmt -p zenpixels-convert
    cargo test -p zenpixels-convert --test cms_moxcms --features std,cms-moxcms,icc-db --locked
    just output-alpha-check
    cargo clippy -p zenpixels-convert --test cms_moxcms --features std,cms-moxcms,icc-db --locked -- -D warnings

# Run in an isolated export; deliberately changes only its dependency lock.
output-float-backend-check version:
    test ! -e .git
    test ! -e .jj
    cargo update -p moxcms --precise {{version}}
    cargo test -p zenpixels-convert --test cms_moxcms --features std,cms-moxcms,icc-db --locked
    cargo clippy -p zenpixels-convert --lib --test cms_moxcms --features std,cms-moxcms,icc-db --locked -- -D warnings

output-plan-sync export: (prepared-sync export)
    mkdir -p "{{export}}/zenpixels-convert/src/output"
    cp zenpixels-convert/src/output/plan.rs "{{export}}/zenpixels-convert/src/output/"
    cp zenpixels-convert/tests/output_plan.rs "{{export}}/zenpixels-convert/tests/"
