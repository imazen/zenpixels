#!/usr/bin/env python3
"""Reproducible ripgrep candidate inventory; does not claim Rust type resolution.

Usage: python scripts/audit-pixel-migration.py --root ~/work --out /tmp/pixel-audit
Ignored files are included. Generated dependency/build trees are excluded below.
Alternate checkouts and archives are retained and separately classified.
"""
import argparse
import collections
import csv
import functools
import json
import re
import subprocess
import tomllib
from pathlib import Path

EXCLUDED = ["target", "target-*", ".git", ".jj", "node_modules", "vendor", "registry", ".venv", "venv", ".cache", ".cargo-home", "cargo_home"]
TOKENS = (
    r"\b(?:zenpixels|zenpixels_convert|PixelBuffer|PixelSlice|PixelSliceMut|PixelCow|"
    r"PixelDescriptor|InPlacePixels|RowConverter|ConvertPlan|PluggableCms|RowTransformMut|"
    r"RowTransform|ColorManagement|ColorContext|ColorOrigin|DiffuseWhite|MultiPlaneImage|"
    r"PlaneDescriptor|OutputProfile|EncodeReady|PixelBufferConvert\w*)\b"
)
RULES = {
    "conversion": r"\b(?:RowConverter|ConvertPlan|convert_row|convert_rows|convert_buffer|adapt_for_encode\w*|try_adapt_in_place|finalize_for_output\w*|quantize_to)\b|\.(?:convert_to|convert_into|into_converted|convert_in_place|convert_to_sdr|linearize|delinearize)\s*\(",
    "storage_candidate": r"\b(?:InPlacePixels|PixelBufferLayout)\b|\.(?:into_vec|from_vec|reinterpret|transform_in_place|into_contiguous_pixels)\s*\(|PixelBuffer\s*::\s*from_vec",
    "color": r"\b(?:ColorContext|ColorOrigin|ColorProfileSource|OutputProfile|EncodeReady|ColorAuthority)\b|\.(?:with_color_context|color_context|with_transfer|with_primaries|with_alpha_mode|with_signal_range|with_descriptor|with_icc|with_cicp)\s*\(",
    "cms": r"\b(?:PluggableCms|RowTransformMut|RowTransform|ColorManagement|build_source_transform|build_shared_source_transform|transform_row)\b",
    "hdr": r"\b(?:DiffuseWhite|CllMeasure|HdrConfig|ContentLightLevel|measure_max|measure_robust|quantize_to|new_with_hdr_peak|new_with_hdr_config)\b|\.(?:with_diffuse_white|with_pq_anchor|convert_to_sdr)\s*\(",
    "planar": r"\b(?:MultiPlaneImage|PlaneDescriptor|PlaneLayout)\b|\.(?:buffers_mut|buffer_mut)\s*\(",
    "streaming_candidate": r"\b(?:next_batch|encode_from|push_rows|DecodeRowSink|StreamingDecode)\b|\.(?:row|row_mut|rows_mut)\s*\(|impl(?:\s*<[^>]*>)?\s+(?:Source|Sink)\s+for",
    "policy_cleanup": r"\b(?:forbid_lossy|ByteOrder|requires_cms|REC2020_V4|ADOBE_RGB_V4|PROPHOTO_V4|icc_profile_for_primaries|apply_orientation\w*)\b",
    "composition_candidate": r"\.compose\s*\(|\.clone\s*\(|\.is_identity\s*\(",
}
MATCH = "|".join([TOKENS, *RULES.values()])
TYPES = re.compile(TOKENS)
CLASSIFIERS = {k: re.compile(v) for k, v in RULES.items()}


def rg(extra, roots):
    cmd = ["rg", "--hidden", "--no-ignore", "--color", "never"]
    for name in EXCLUDED:
        cmd += ["-g", f"!**/{name}/**"]
    cmd += extra + [str(p) for p in roots]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    return result.stdout


def tsv(path, columns, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, columns, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    errors = []

    @functools.lru_cache(None)
    def repo_for(directory):
        p = Path(directory)
        for candidate in [p, *p.parents]:
            if (candidate / ".git").exists() or (candidate / ".jj").is_dir():
                return candidate
            if candidate == root:
                break
        # Standalone consumer experiments without VCS: nearest Cargo manifest.
        for candidate in [p, *p.parents]:
            if (candidate / "Cargo.toml").is_file():
                return candidate
            if candidate == root:
                break
        return p

    def classification(repo):
        rel = repo.relative_to(root)
        if any(p in {"pre-filter", "zen-arm-src", "codec-artifacts", "imagers-research"}
               or p.startswith("zensim-validation-") for p in rel.parts):
            return "snapshot/research"
        if any(p in {"retired", "archived", "archive", "_dbg", "references", "reference"} for p in rel.parts):
            return "archive/reference"
        if (repo / ".git").is_file() or ".claude" in rel.parts or "--" in repo.name:
            return "alternate-checkout"
        if not (repo / ".git").exists() and not (repo / ".jj").exists():
            return "unversioned"
        return "primary"

    manifests = sorted(rg(["-l", "-g", "Cargo.toml", r"\bzenpixels(?:-convert)?\b"], [root]).splitlines())
    records = []
    repos = set()
    for file in manifests:
        p = Path(file)
        try:
            data = tomllib.loads(p.read_text())
        except (ValueError, OSError) as e:
            errors.append({"file": file, "error": str(e)})
            continue
        repo = repo_for(str(p.parent))

        def visit(table, path=()):
            for key, value in table.items():
                if isinstance(value, dict):
                    if key in {"dependencies", "dev-dependencies", "build-dependencies"}:
                        for alias, spec in value.items():
                            package = spec.get("package", alias) if isinstance(spec, dict) else alias
                            if package not in {"zenpixels", "zenpixels-convert"}:
                                continue
                            repos.add(repo)
                            records.append({"repo": str(repo.relative_to(root)), "class": classification(repo),
                                "manifest": str(p.relative_to(root)), "consumer": data.get("package", {}).get("name", "<workspace>"),
                                "table": ".".join((*path, key)), "alias": alias, "dependency": package,
                                "spec": json.dumps(spec, sort_keys=True)})
                    visit(value, (*path, key))
        visit(data)
        if data.get("package", {}).get("name") in {"zenpixels", "zenpixels-convert"}:
            repos.add(repo)

    direct_files = sorted(rg(["-l", "-g", "*.rs", r"\b(?:zenpixels|zenpixels_convert)\b"], [root]).splitlines())
    for file in direct_files:
        repos.add(repo_for(str(Path(file).parent)))
    # A parent repository covers nested repositories too; assign each hit to its
    # actual nearest root afterward, while avoiding duplicate scanning.
    scan_roots = []
    for repo in sorted(repos, key=lambda p: (len(p.parts), str(p))):
        if not any(repo.is_relative_to(p) for p in scan_roots):
            scan_roots.append(repo)
    (out / "scope.json").write_text(json.dumps({"root": str(root), "excluded_directory_names": EXCLUDED,
        "manifest_matches": len(manifests), "direct_reference_files": len(direct_files),
        "repos": [str(p.relative_to(root)) for p in sorted(repos)], "rules": RULES,
        "warning": "Regex candidates, not resolved Rust symbols. Includes comments/tests and unrelated same-name methods."
    }, indent=2) + "\n")

    hits = []
    raw = rg(["--json", "-g", "*.rs", MATCH], scan_roots)
    (out / "rg-hits.jsonl").write_text(raw)
    lines_cache = {}
    relevance_cache = {}
    for line in raw.splitlines():
        event = json.loads(line)
        if event["type"] != "match":
            continue
        d = event["data"]
        p = Path(d["path"]["text"])
        repo = repo_for(str(p.parent))
        n = d["line_number"]
        text = d["lines"]["text"].strip()
        categories = [key for key, pattern in CLASSIFIERS.items() if pattern.search(text)]
        if TYPES.search(text):
            categories.append("type_reference")
        if p not in lines_cache:
            lines_cache[p] = p.read_text(errors="replace").splitlines()
        src = lines_cache[p]
        if p not in relevance_cache:
            relevance_cache[p] = "explicit-type-reference" if any(TYPES.search(s) for s in src) else "receiver-unresolved"
        file_relevance = relevance_cache[p]
        start = n - 1
        while start > max(0, n - 17):
            prev = src[start - 1]
            if "{" in prev or ";" in prev or "}" in prev:
                break
            start -= 1
        context = " ".join(src[start:n])
        boundary = bool(TYPES.search(text) and re.search(r"\bpub(?:\([^)]*\))?\s+(?:async\s+)?(?:fn|use|type|struct|enum|trait|\w+\s*:)", context))
        if boundary:
            categories.append("public_boundary_candidate")
        # The focused export filters on visible pixel context in the file;
        # even there a clone/compose receiver is still unresolved.
        role = "test/example/bench" if any(x in p.parts for x in ("tests", "examples", "benches", "benchmarks", "fuzz")) else "source"
        if text.startswith(("//", "*")):
            role += "/comment"
        hits.append({"repo": str(repo.relative_to(root)), "class": classification(repo),
            "file": str(p.relative_to(root)), "line": n, "role": role,
            "relevance": file_relevance, "categories": ",".join(categories), "text": text})

    hits.sort(key=lambda h: (h["repo"], h["file"], h["line"]))
    tsv(out / "manifests.tsv", ["repo", "class", "manifest", "consumer", "table", "alias", "dependency", "spec"], records)
    columns = ["repo", "class", "file", "line", "role", "relevance", "categories", "text"]
    tsv(out / "hits.tsv", columns, hits)
    tsv(out / "primary-focused.tsv", columns,
        [h for h in hits if h["class"] == "primary" and h["relevance"] == "explicit-type-reference"])
    groups = collections.defaultdict(list)
    for h in hits:
        groups[h["repo"]].append(h)
    summaries = []
    for name, group in sorted(groups.items()):
        counts = collections.Counter(c for h in group for c in h["categories"].split(",") if c)
        summaries.append({"repo": name, "class": group[0]["class"], "files": len({h["file"] for h in group}),
            "matched_lines": len(group), **counts})
    tsv(out / "repos.tsv", ["repo", "class", "files", "matched_lines", "type_reference", "public_boundary_candidate", *RULES], summaries)
    (out / "errors.json").write_text(json.dumps(errors, indent=2) + "\n")
    summary = {"manifest_matches": len(manifests), "dependency_declarations": len(records),
        "direct_reference_files": len(direct_files), "scoped_repos": len(repos),
        "matched_repos": len(summaries), "matched_files": len(lines_cache), "matched_lines": len(hits),
        "repo_classes": dict(collections.Counter(s["class"] for s in summaries)), "parse_errors": len(errors)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("Full candidate inventory:", out)


if __name__ == "__main__":
    main()
