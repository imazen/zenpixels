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
# One behavioral contract for every CMS backend. A directory of ICC profiles adds a comparison on each.
cms-contract PROFILES='':
    CMS_CONTRACT_PROFILES={{PROFILES}} cargo test --manifest-path cms-contract/Cargo.toml {{ if PROFILES == "" { "" } else { "-- --include-ignored --nocapture" } }}

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

# Native borrowed code-plane conversion, default and freestanding compilation.
ycbcr-check:
    cargo test -p zenpixels --test code_planes
    cargo test -p zenpixels-convert --test ycbcr_borrowed
    cargo test -p zenpixels-convert --no-default-features --test ycbcr_borrowed
    cargo clippy -p zenpixels -p zenpixels-convert --lib --test code_planes --test ycbcr_borrowed -- -D warnings
    cargo check -p zenpixels -p zenpixels-convert --no-default-features --target wasm32-unknown-unknown

# Structural size_of/align_of of pixel views at BASE vs this tree (not heap or RSS).
layout-probe BASE='86d0e2d30406a006bf8332f142029b7f58353086':
    benchmarks/layout-probe/run.sh {{BASE}}

# Native TIFF/IMIJ sample carriage and explicit conversion refusal.
native-samples-check:
    cargo test -p zenpixels --test parts
    cargo test -p zenpixels -p zenpixels-convert --test native_samples --features zenpixels-convert/bt2446a,zenpixels-convert/cms-moxcms
    cargo clippy -p zenpixels -p zenpixels-convert --lib --test native_samples --features zenpixels-convert/bt2446a,zenpixels-convert/cms-moxcms -- -D warnings

# Full source coverage after extending the physical format enum.
native-samples-suite:
    cargo test -p zenpixels -p zenpixels-convert --lib --tests --features zenpixels-convert/bt2446a,zenpixels-convert/cms-moxcms
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

# Compatibility alias for existing callers.
beta-cohort-check: stable-cohort-check

# Validate the released SIMD/CMS cohort without changing pixel expectations.
stable-cohort-check:
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
    cp scripts/icc-gen/src/main.rs "{{export}}/scripts/icc-gen/src/"

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
    cargo test -p zenpixels-convert --test output_exact --test output_native --test output_plan --test output_descriptor --test output_current_color --test output_contracts --test output_finalize --test unspecified_cicp_context --test prepared_contracts --features std,cms-moxcms,icc-db --locked
    cargo clippy -p zenpixels-convert --lib --test output_exact --test output_native --test output_plan --test output_descriptor --features std,cms-moxcms,icc-db --locked -- -D warnings

output-descriptor-check:
    cargo fmt -p zenpixels-convert
    just output-plan-check
    just output-borrow-compat

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

# Include the ICC generator: workspace feature unification affects its options.
output-float-workspace-check:
    cargo fmt -p zenpixels -p zenpixels-convert -p icc-gen
    cargo test --workspace --locked
    cargo clippy --workspace --all-targets --locked -- -D warnings
    just output-float-generator-check

output-float-generator-check:
    cargo clippy -p icc-gen --all-targets --locked -- -D warnings

output-plan-sync export: (prepared-sync export)
    mkdir -p "{{export}}/zenpixels-convert/src/output"
    cp zenpixels-convert/src/output/plan.rs "{{export}}/zenpixels-convert/src/output/"
    cp zenpixels-convert/tests/output_plan.rs "{{export}}/zenpixels-convert/tests/"

# Unclipped transfer requests, legacy in-range parity and exact alpha/padding.
unclipped-transfer-check:
    cargo test -p zenpixels-convert --lib kernel_tests::transfer_alpha --features __trace_ops --locked
    cargo test -p zenpixels-convert --test unclipped_transfer_contract --test gamma22_transfer --features __trace_ops --locked
    cargo clippy -p zenpixels-convert --lib --test unclipped_transfer_contract --features __trace_ops --locked -- -D warnings

# Qualify the 0.3 retirement merged with current output and precision contracts.
release-reconcile-check:
    cargo check --workspace --all-targets --all-features --locked
    cargo test --workspace --all-features --locked
    cargo clippy --workspace --all-targets --all-features --locked -- -D warnings

# Still-image IQA contract: no codec or metric implementation dependencies.
iqa-check:
    cargo fmt --manifest-path zenpsy-api/Cargo.toml --check
    cargo test --locked --manifest-path zenpsy-api/Cargo.toml --no-default-features
    cargo test --locked --manifest-path zenpsy-api/Cargo.toml --features std
    cargo clippy --locked --manifest-path zenpsy-api/Cargo.toml --all-targets --all-features -- -D warnings
    RUSTDOCFLAGS="-D warnings" cargo doc --locked --manifest-path zenpsy-api/Cargo.toml --no-deps

iqa-portability-check:
    cargo +1.88.0 check --locked --manifest-path zenpsy-api/Cargo.toml --no-default-features
    cargo check --locked --manifest-path zenpsy-api/Cargo.toml --no-default-features --target wasm32-unknown-unknown
    cargo check --locked --manifest-path zenpsy-api/Cargo.toml --no-default-features --target thumbv7em-none-eabihf

# Run at the repository root to retain pixel workspace inheritance in cross.
iqa-cross-check toolchain='':
    cross {{toolchain}} test --locked --manifest-path zenpsy-api/Cargo.toml --target i686-unknown-linux-gnu

# Zeroed storage plus the existing row/alignment constructor contracts.
zeroed-alloc-check:
    cargo test --locked -p zenpixels --lib
    cargo check --locked -p zenpixels --no-default-features
    cargo clippy --locked -p zenpixels --lib -- -D warnings

# Build the optional browser converter and assemble the GitHub Pages artifact.
site-build:
    ./site/build.sh

# Browser integration checks; PLAYWRIGHT_MODULE points at an installed Playwright.
site-check URL='http://127.0.0.1:3116':
    node site/smoke.cjs {{URL}}

# Explicit HDR units, existing kernel goldens and public API contracts.
hdr-intent-check:
    cargo test --locked --workspace --all-features
    cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
    cargo check --locked -p zenpixels-convert --no-default-features --features bt2446a
    ZEN_API_DOC_TOOLCHAIN={{apidoc_toolchain}} cargo test --manifest-path apidoc/Cargo.toml

# Frozen, unpublished gamma diagnostics; prepare.py reconstructs dependencies.
moxcms-gamma-check:
    cargo fmt --manifest-path benchmarks/moxcms-gamma/Cargo.toml -p moxcms-gamma-harness --check
    cargo check --locked --manifest-path benchmarks/moxcms-gamma/Cargo.toml --all-targets
    cargo run --locked --manifest-path benchmarks/moxcms-gamma/Cargo.toml --bin probe
    cargo run --locked --manifest-path benchmarks/moxcms-gamma/Cargo.toml --bin black

# Borrowed planar views, layout validation and typed-vector docs.
planar-api-check:
    cargo test --locked -p zenpixels --all-features
    cargo clippy --locked -p zenpixels --all-targets --all-features -- -D warnings
    cargo check --locked -p zenpixels --no-default-features
    ZEN_API_DOC_TOOLCHAIN={{apidoc_toolchain}} cargo test --manifest-path apidoc/Cargo.toml

# Add fresh bundle hashes without dropping recognition of older profile bytes.
icc-bundle-identification output:
    cargo run --locked -p icc-gen --release --bin icc-gen -- zenpixels-convert/src/profiles {{output}}
    python3 scripts/icc-gen/merge_bundle_tables.py {{output}} zenpixels/src/icc

# Requires pkg-config, libpng development headers and Little CMS 2.
icc-bundle-reader-check output:
    cargo run --locked -p zenpixels-convert --example icc_bundle_export -- "{{output}}"
    cc -Wall -Wextra -Werror scripts/icc-gen/validation/png_icc_reader.c -o "{{output}}/reader" $(pkg-config --cflags --libs libpng lcms2)
    python3 scripts/icc-gen/validation/check_bundle.py "{{output}}/reader" "{{output}}" 313

# Exact upstream review sources; caller supplies the host resource wrapper.
moxcms-review-build:
    cargo check --locked --manifest-path benchmarks/moxcms-gamma/Cargo.toml --features review-revisions --all-targets
    cargo build --locked --manifest-path benchmarks/moxcms-gamma/Cargo.toml --release --features review-revisions --bin icount

moxcms-review-instructions output:
    python3 benchmarks/moxcms-gamma/review-evidence.py "{{output}}"

moxcms-review-bench data:
    ZENBENCH_DATA_DIR="{{data}}" cargo bench --locked --manifest-path benchmarks/moxcms-gamma/Cargo.toml --features review-revisions --bench underflow -- --mode=confirm --processes=5 --timeout=1200 --workload-id=moxcms-review-84104443-45b011cd-4ab190f0-v1 --format=llm

# Predeclared cell matrix: unchanged admission; confidence adjusted across 36 cells.
moxcms-review-cells binary output:
    python3 benchmarks/moxcms-gamma/review-cells.py "{{binary}}" "{{output}}"

# Validate all retained process gates and export review timing comparisons.
moxcms-review-summary study output:
    python3 benchmarks/moxcms-gamma/summarize-review-cells.py "{{study}}" "{{output}}"

# Read-only source checkouts; output must be a new directory.
codec-cohort-prepare codec output core pixels:
    python3 benchmarks/codec-cohort/prepare.py "{{codec}}" "{{output}}" "{{core}}" "{{pixels}}"

codec-cohort-resolve manifest:
    cargo metadata --format-version 1 --manifest-path "{{manifest}}"

# Caller supplies the frozen experimental dependencies; never edits them.
jxl-cms-prepare pixels decoder output:
    python3 benchmarks/jxl-cms-parity/prepare.py "{{pixels}}" "{{decoder}}" "{{output}}"

jxl-cms-adapter-check manifest:
    cargo test --locked --manifest-path "{{manifest}}" --bin jxl-cms-parity-review
    cargo clippy --locked --manifest-path "{{manifest}}" --bin jxl-cms-parity-review --no-deps -- -D warnings

# Explicit fixture admission; this fails if either input directory is absent.
jxl-cms-reference-check manifest fixtures references:
    JXL_CMS_FIXTURES="{{fixtures}}" JXL_CMS_REFERENCES="{{references}}" cargo test --locked --manifest-path "{{manifest}}" --bin jxl-cms-parity-review pq_matches_djxl_references_and_hlg_returns_refusal -- --ignored --nocapture

jxl-cms-render-references djxl fixtures output:
    mkdir "{{output}}"
    "{{djxl}}" --version
    "{{djxl}}" "{{fixtures}}/hdr_pq_test.jxl" "{{output}}/hdr_pq_test_SRG.pfm" --num_threads=4 --color_space=RGB_D65_SRG_Rel_SRG
    "{{djxl}}" "{{fixtures}}/hdr_pq_test.jxl" "{{output}}/hdr_pq_test_P3.pfm" --num_threads=4 --color_space=RGB_D65_DCI_Rel_SRG

# Exact public-transform zero-channel comparison, separate from timing.
moxcms-real-pixel-check:
    cargo run --locked --manifest-path benchmarks/moxcms-gamma/Cargo.toml --features review-revisions --bin black

pow-comparison-check:
    cargo fmt --manifest-path benchmarks/pow-comparison/Cargo.toml -p gamma-pow-comparison --check
    cargo clippy --locked --manifest-path benchmarks/pow-comparison/Cargo.toml --all-targets -- -D warnings

pow-comparison output:
    cargo run --locked --release --manifest-path benchmarks/pow-comparison/Cargo.toml > "{{output}}"

moxcms-optimized-check:
    cargo run --locked --release --manifest-path benchmarks/moxcms-gamma/Cargo.toml --features optimized --bin optimized_check

moxcms-optimized-instructions output:
    python3 benchmarks/moxcms-gamma/optimized-evidence.py "{{output}}"

moxcms-optimized-cells binary output:
    python3 benchmarks/moxcms-gamma/optimized-cells.py "{{binary}}" "{{output}}"

moxcms-optimized-summary study output:
    python3 benchmarks/moxcms-gamma/summarize-optimized-cells.py "{{study}}" "{{output}}"

moxcms-outlined-build:
    cargo fmt --manifest-path benchmarks/moxcms-gamma/Cargo.toml -p moxcms-gamma-harness --check
    cargo clippy --locked --manifest-path benchmarks/moxcms-gamma/Cargo.toml --features outlined --bench underflow --bin optimized_check --bin icount --no-deps -- -D warnings
    cargo run --locked --release --manifest-path benchmarks/moxcms-gamma/Cargo.toml --features outlined --bin optimized_check
    cargo build --locked --release --manifest-path benchmarks/moxcms-gamma/Cargo.toml --features outlined --bin icount --bench underflow
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc cargo build --locked --release --manifest-path benchmarks/moxcms-gamma/Cargo.toml --features outlined --bench underflow --target aarch64-unknown-linux-gnu

moxcms-outlined-cells binary output:
    python3 benchmarks/moxcms-gamma/outlined-cells.py "{{binary}}" "{{output}}"

moxcms-outlined-instructions output:
    python3 benchmarks/moxcms-gamma/outlined-evidence.py "{{output}}"

moxcms-final-build:
    cargo fmt --manifest-path benchmarks/moxcms-gamma/Cargo.toml -p moxcms-gamma-harness --check
    cargo clippy --locked --manifest-path benchmarks/moxcms-gamma/Cargo.toml --features final-review --bench underflow --bin optimized_check --bin icount --no-deps -- -D warnings
    cargo run --locked --release --manifest-path benchmarks/moxcms-gamma/Cargo.toml --features final-review --bin optimized_check
    cargo build --locked --release --manifest-path benchmarks/moxcms-gamma/Cargo.toml --features final-review --bin icount --bench underflow
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc cargo build --locked --release --manifest-path benchmarks/moxcms-gamma/Cargo.toml --features final-review --bench underflow --target aarch64-unknown-linux-gnu

moxcms-final-instructions output:
    python3 benchmarks/moxcms-gamma/final-evidence.py "{{output}}"

moxcms-final-cells binary output:
    python3 benchmarks/moxcms-gamma/final-cells.py "{{binary}}" "{{output}}"

moxcms-final-summary study output:
    python3 benchmarks/moxcms-gamma/summarize-optimized-cells.py "{{study}}" "{{output}}" --allow-partial

moxcms-final-report study comparisons output:
    python3 benchmarks/moxcms-gamma/report-final.py "{{study}}" "{{comparisons}}" "{{output}}"

# Observe the released CMS on the original exact-zero regression inputs.
moxcms-release-probe:
    cargo run --locked -p icc-gen --bin moxcms_release_probe

# Compare a proposed bundle without replacing committed data or expectations.
icc-bundle-compare proposed output:
    python3 scripts/icc-gen/validation/compare_bundles.py zenpixels-convert {{proposed}} {{output}}

# Independent old/new decoded-color characterization; no fixture updates.
icc-bundle-color-compare old proposed output:
    cc -Wall -Wextra -Werror scripts/icc-gen/validation/compare_color.c -o "{{output}}-comparator" $(pkg-config --cflags --libs lcms2)
    python3 scripts/icc-gen/validation/compare_bundle_color.py "{{output}}-comparator" "{{old}}" "{{proposed}}" "{{output}}"

# Same grid and output schema through the already integrated Rust LCMS port.
icc-bundle-rust-color-compare old proposed output:
    cargo build --locked --manifest-path cms-contract/Cargo.toml --examples
    python3 scripts/icc-gen/validation/compare_bundle_color.py "${CARGO_TARGET_DIR:-cms-contract/target}/debug/examples/bundle_color" "{{old}}" "{{proposed}}" "{{output}}"

cms-rust-cmyk-floats icc:
    cargo run --locked --manifest-path cms-contract/Cargo.toml --example cmyk_floats -- "{{icc}}"

cms-rust-cmyk-oracle-check icc oracle:
    cargo run --locked --manifest-path cms-contract/Cargo.toml --example cmyk_floats -- "{{icc}}" "{{oracle}}"

# Print selected versions and beta ceilings without copying local paths.
inspect-cohort metadata output:
    python3 scripts/inspect-cohort.py "{{metadata}}" "{{output}}"

# Caller must already hold the isolated repository lock; prepares without pushing.
prepare-stable-cap repo base owner *manifests:
    python3 scripts/prepare-stable-cap.py "{{repo}}" "{{base}}" "{{owner}}" {{manifests}}

# Isolated roots and resource wrapper must already be owned by the caller.
resolve-stable-cohorts inventory output:
    python3 scripts/resolve-stable-cohorts.py "{{inventory}}" "{{output}}"

check-stable-cohorts plan output owner:
    python3 scripts/check-stable-cohorts.py "{{plan}}" "{{output}}" "{{owner}}"

stable-standalone-check:
    cargo check --locked --manifest-path tools/jzazbz-bench/Cargo.toml --all-targets
    cargo test --locked --manifest-path cms-contract/Cargo.toml
    cargo check --locked --manifest-path site/converter-wasm/Cargo.toml --target wasm32-unknown-unknown

# Production LCMS contract, with MoxCMS independently exercised by cms-contract.
cms-lcms-check:
    cargo check --locked -p zenpixels-convert --lib --no-default-features --features cms-lcms
    cargo test --locked -p zenpixels-convert --features cms-lcms --test cms_lcms_contracts
    cargo test --locked --manifest-path cms-contract/Cargo.toml
    cargo clippy --locked -p zenpixels-convert --features cms-lcms --lib --test cms_lcms_contracts -- -D warnings

# Construct independent colorimetry while preserving every curve/CICP payload.
icc-lcms-candidates source output:
    cargo run --locked --manifest-path cms-contract/Cargo.toml --example profile_candidates -- "{{source}}" "{{output}}"

icc-profile-triage reference output *candidates:
    python3 scripts/icc-gen/validation/triage_profiles.py "{{reference}}" "{{output}}" {{candidates}}

icc-profile-curves profiles:
    cargo run --locked --manifest-path cms-contract/Cargo.toml --example profile_curves -- "{{profiles}}"

# Characterize alternate PQ encodings before selecting export definitions.
icc-pq-precision rgb gray output:
    cargo run --locked --manifest-path cms-contract/Cargo.toml --example pq_precision -- "{{rgb}}" "{{gray}}" "{{output}}"

# Read-only source/bundle inspection; requires the system liblz4 decoder.
icc-bundle-definition-audit revision="":
    python3 scripts/icc-gen/validation/inspect_bundle_definition.py {{revision}}

# Equal-input control: MoxCMS constructor with the same exact H.273 coordinates.
icc-mox-exact-candidates source output:
    cargo run --locked --manifest-path cms-contract/Cargo.toml --example profile_candidates -- "{{source}}" "{{output}}" mox-exact

# External reference profile characterization; no fixture or threshold changes.
icc-pq-external profile:
    cargo run --locked --manifest-path cms-contract/Cargo.toml --example pq_external_profile -- "{{profile}}"

# ICC-only and native-cICP capability witnesses; requires Playwright browsers.
icc-pq-png-readers original floating external output *readers:
    node scripts/icc-gen/validation/pq_png_readers.cjs "{{original}}" "{{floating}}" "{{external}}" "{{output}}" {{readers}}

# Export recipe bytes for independent numeric/container qualification.
icc-export-candidates output:
    cargo run --locked -p icc-gen --bin export_candidates -- "{{output}}"

icc-profile-eval source dest input output intent="production":
    cargo run --locked --manifest-path cms-contract/Cargo.toml --example profile_eval -- "{{source}}" "{{dest}}" "{{input}}" "{{output}}" "{{intent}}"

# Scalar only: source candidates, hashes and bounded worst-case witnesses survive.
icc-export-qualify profiles c_reader rust_reader output phase *extra:
    python3 scripts/icc-gen/validation/qualify_exports.py "{{profiles}}" "{{c_reader}}" "{{rust_reader}}" "{{output}}" --phase "{{phase}}" {{extra}}

# Qualify runtime exports and recipe source without changing expectations.
icc-export-runtime-check:
    cargo test --locked -p zenpixels-convert --lib --features cms-moxcms,cms-lcms,icc-db exported_pq

icc-authority-check:
    cargo test --locked -p zenpixels-convert --lib --features cms-moxcms,cms-lcms,icc-db load_bearing

icc-export-source-check:
    cargo fmt -p zenpixels -p zenpixels-convert -p icc-gen --check
    cargo clippy --workspace --all-targets --features cms-moxcms,cms-lcms,icc-db --locked -- -D warnings

icc-export-workspace-check:
    cargo test --workspace --features cms-moxcms,cms-lcms,icc-db --locked

# Test-only CMYK probes. Use a host resource wrapper and isolated CARGO_TARGET_DIR.
cmyk-probes-build:
    cargo build --locked --manifest-path cms-contract/Cargo.toml --example profile_eval --example cmyk_layout_contract

cmyk-probes-check:
    cargo fmt --manifest-path cms-contract/Cargo.toml --check
    cargo clippy --locked --manifest-path cms-contract/Cargo.toml --all-targets -- -D warnings
    cargo test --locked --manifest-path cms-contract/Cargo.toml

# Frozen C LCMS reference source directory with include/ and src/ children.
cmyk-c-oracle-build reference binary:
    test ! -e "{{binary}}"
    cc -O2 -I "{{reference}}/include" scripts/icc-gen/validation/profile_eval.c "{{reference}}"/src/*.c -lm -lpthread -o "{{binary}}"

cmyk-native-compare evaluator source input reference rgb output:
    PYTHONDONTWRITEBYTECODE=1 python3 cms-contract/examples/native_cmyk_compare.py --evaluator "{{evaluator}}" --source "{{source}}" --input "{{input}}" --reference "{{reference}}" --rgb-profile "{{rgb}}" --output "{{output}}"

# Qualified recipe generation versus fresh group allocation/decode and cached lookup.
icc-synthesis-build:
    cargo bench --locked --manifest-path cms-contract/Cargo.toml --bench icc_synthesis --no-run

icc-synthesis-plan binary cpu:
    taskset -c {{cpu}} "{{binary}}" --cpu-policy=auto --cpu-set={{cpu}} --mode=confirm --processes=3 --rounds=12 --workload-id=icc-qualified-af19-synthesis-v1 --dry-run

icc-synthesis-measure binary cpu data:
    ZENBENCH_DATA_DIR="{{data}}" taskset -c {{cpu}} "{{binary}}" --cpu-policy=auto --cpu-set={{cpu}} --mode=confirm --processes=3 --rounds=12 --workload-id=icc-qualified-af19-synthesis-v1 --timeout=600 --format=llm

# Container cpuset supplies the enforced process boundary required by zenbench.
# Host libraries match the compiler host; the reference image is supplied by immutable ID.
icc-synthesis-container binary image cpu data *options:
    docker run --rm --network none --pid host --workdir /results --memory 2g --memory-swap 2g --cpuset-cpus {{cpu}} --user "$(id -u):$(id -g)" --mount type=bind,src={{binary}},dst=/bench,readonly --mount type=bind,src={{data}},dst=/results --mount type=bind,src=/lib/x86_64-linux-gnu,dst=/lib/x86_64-linux-gnu,readonly --mount type=bind,src="${HOME}/.cache/zenbench/admission-v1",dst=/coord --env ZENBENCH_COORDINATION_DIR=/coord --env TMPDIR=/results --env ZENBENCH_DATA_DIR=/results/evidence --entrypoint /usr/bin/nice {{image}} -n19 /bench --cpu-policy=auto --cpu-set={{cpu}} --mode=confirm --processes=3 --rounds=12 --workload-id=icc-qualified-af19-synthesis-v1 --timeout=600 --format=llm {{options}}

icc-u16-rounding-probe:
    cargo run --locked --release --manifest-path cms-contract/Cargo.toml --example u16_rounding_probe

# Qualified ICC substitution and a genuine consumer without icc-db.
icc-collapse-check:
    cargo test --locked -p zenpixels-convert --lib load_bearing::
    cargo test --locked --manifest-path tools/icc-collapse-no-db/Cargo.toml
    cargo clippy --locked --manifest-path tools/icc-collapse-no-db/Cargo.toml --all-targets -- -D warnings

# Consolidated CMS stack gates; run with the shared resource wrapper.
main-cms-integration-check:
    cargo clippy --workspace --all-targets --features cms-moxcms,cms-lcms,icc-db --locked -- -D warnings
    cargo test --locked --manifest-path cms-contract/Cargo.toml
    cargo test --locked --manifest-path tools/icc-collapse-no-db/Cargo.toml
    cargo clippy --locked --manifest-path tools/icc-collapse-no-db/Cargo.toml --all-targets -- -D warnings

main-hdr-focused-check:
    cargo test -p zenpixels-convert --features bt2446a,cms-lcms --test hdr_intent --test precision_policy --test precision_rule_oracle

main-hdr-integration-check:
    cargo test --workspace --features bt2446a,cms-moxcms,cms-lcms,icc-db --locked
    just main-hdr-final-check

main-hdr-final-check:
    cargo clippy --workspace --all-targets --features bt2446a,cms-moxcms,cms-lcms,icc-db --locked -- -D warnings
    cargo check -p zenpixels-convert --no-default-features --features bt2446a --locked
    cargo test --manifest-path cms-contract/Cargo.toml
    cargo test --manifest-path tools/icc-collapse-no-db/Cargo.toml
    cargo check --manifest-path site/converter-wasm/Cargo.toml --target wasm32-unknown-unknown

main-byte-stride-check:
    cargo test --locked -p zenpixels --all-features --test byte_stride --test parts --test image_interop
    cargo test --locked -p zenpixels-convert --features bt2446a,cms-lcms --test byte_stride
    cargo clippy --locked -p zenpixels -p zenpixels-convert --all-targets --all-features -- -D warnings

# Keep the browser converter on its committed standalone dependency graph.
explorer-boundary-check:
    cargo fmt --manifest-path site/converter-wasm/Cargo.toml --check
    cargo clippy --manifest-path site/converter-wasm/Cargo.toml --locked --target wasm32-unknown-unknown -- -D warnings

main-api-cleanup-check:
    cargo test --locked --workspace --all-features
    cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
    just api-doc

# Native word interpretation and exact metadata contracts.
native-code-check:
    cargo test -p zenpixels -p zenpixels-convert --test native_codes --test native_canonical

# Explicit display rendering and existing refusal gates.
hlg-rendering-check:
    cargo test -p zenpixels-convert --test hlg_rendering --test hdr_intent
    cargo test -p zenpixels-convert --features bt2446a --test hlg_rendering --test hdr_intent

# Native libultrahdr recognition and independent C LCMS color qualification.
# Invoke just through run-heavy; OUT must not exist.
p3-export-check SOURCE OUT:
    tools/p3-export-oracle/check.sh {{SOURCE}} {{OUT}}

# Complete approved contract gate; release avoids debug exhaustive-grid overhead.
approved-contracts-check:
    cargo test -p zenpixels-convert --release --features __trace_ops --test plan_validation
    cargo test --workspace --release --lib --tests
    cargo test --workspace --release --doc
    cargo test -p zenpixels-convert --release --all-features --test hlg_rendering --test hdr_intent --test matte_row_kernels --test native_canonical
    cargo test -p zenpixels --no-default-features --test native_codes
    cargo clippy -p zenpixels -p zenpixels-convert --all-targets -- -D warnings

# Shared sample interpretation and native carriage capability boundaries.
sample-design-check:
    cargo test -p zenpixels -p zenpixels-convert --test sample_encoding --test code_planes --test native_codes --test native_canonical --test codes_exhaustive

# Regressions for the single-contract conversion design.
design-check:
    cargo test -p zenpixels -p zenpixels-convert --test design_invariants

# Planner rewrites, complete operation contracts, and explicit HDR policy.
plan-design-check:
    cargo test -p zenpixels-convert --release --features __trace_ops,bt2446a --test plan_validation --test design_invariants --test matte_row_kernels --test hlg_rendering

# Pixel/context parity across allocation strategies and profile finalization.
context-design-check:
    cargo test -p zenpixels-convert --release --test design_invariants --test output_descriptor --test output_native --test hdr_intent

# Local algorithm/payload consistency; magetypes primitive convergence is separate.
local-numerics-check:
    cargo test -p zenpixels-convert --release --features hdr-experimental --test design_invariants --test cll_measure

# Additional feature and generator checks for the breaking-release consolidation.
design-feature-check:
    cargo test -p zenpixels-convert --release --all-features --lib --test design_invariants --test cll_measure
    cargo test -p icc-gen --release --bin icc-gen
    cargo clippy -p zenpixels -p zenpixels-convert -p icc-gen --all-targets --all-features -- -D warnings
