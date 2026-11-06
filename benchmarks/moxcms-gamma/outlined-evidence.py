#!/usr/bin/env python3
"""Build frozen optimized variants with assembly and collect row-scoped Callgrind counts."""
import argparse, hashlib, json, os, subprocess
from pathlib import Path

REVISIONS = {
    "before": "841044439077ec1ec67a71b2db979ab51d304fcf",
    "branch": "45b011cdd045351eea689d250df0e7cc55af84dd",
    "guarded": "4ab190f0e76f95b332a04ce059b51c893e2f5d79",
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    manifest = root / "Cargo.toml"
    target = Path(os.environ["CARGO_TARGET_DIR"])
    receipt = {"revisions": REVISIONS, "rustc": subprocess.check_output(["rustc", "-Vv"], text=True),
               "rustflags": os.environ.get("RUSTFLAGS", ""), "commands": []}
    command = ["cargo", "build", "--locked", "--manifest-path", str(manifest), "--release",
               "--features", "outlined", "--bin", "icount", "--message-format=json"]
    receipt["commands"].append(command)
    with (output / "build.log").open("w") as log:
        built = subprocess.run(command, cwd=root, stdout=subprocess.PIPE, stderr=log, text=True, check=True)
    (output / "cargo-artifacts.jsonl").write_text(built.stdout)
    artifacts = [json.loads(line) for line in built.stdout.splitlines()]
    identities = dict(REVISIONS)
    for name in ["opt_pixel", "opt_cold"]:
        variant = root / "variants" / name.replace("_", "-")
        identities[name] = str(variant)
        receipt[name] = json.loads((variant / "experiment-source.json").read_text())
    for name, revision in identities.items():
        matches = [item for item in artifacts if item.get("reason") == "compiler-artifact"
                   and revision in item.get("package_id", "") and item.get("target", {}).get("name") == "moxcms"]
        assert len(matches) == 1, (name, matches)
        library = next(Path(path) for path in matches[0]["filenames"] if path.endswith(".rlib"))
        command = ["objdump", "-Cd", "--no-show-raw-insn", str(library)]
        receipt["commands"].append(command)
        receipt.setdefault(name, {}).update({"library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                         "package_id": matches[0]["package_id"]})
        # Preserve only the complete evaluator function; the archive contains many unrelated objects.
        disassembler = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
        selected, fallback, capturing, capturing_fallback = [], [], False, False
        for line in disassembler.stdout:
            if line.rstrip().endswith(">:"):
                capturing = ("ToneCurveEvaluatorPureGamma as " in line and "::evaluate_tristimulus>" in line)
                capturing_fallback = ("ToneCurveEvaluatorPureGamma" in line and "::fallback_rgb>" in line)
            if capturing:
                selected.append(line)
            if capturing_fallback:
                fallback.append(line)
        assert disassembler.wait() == 0
        assert selected, name
        (output / f"{name}.asm").write_text("".join(selected))
        if fallback:
            (output / f"{name}-fallback.asm").write_text("".join(fallback))
    binary = target / "release/icount"
    receipt["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    counts = []
    for workload, negative in [("eval_encode", 0), ("eval_encode", 10), ("eval_encode", 50),
                               ("eval_encode", 100), ("eval_decode", 0), ("eval_decode", 50),
                               ("p3_adobe_f32", 0), ("srgb_adobe_f32", 0), ("p3_adobe_u8", 0), ("cicp22_bt2020",0), ("cicp28_bt2020",0),
                               ("prophoto_srgb_zero10",0), ("prophoto_srgb_zero100",0)]:
        for arm in identities:
            label = f"{workload}-{negative}-{arm}"
            command = ["valgrind", "--tool=callgrind", f"--callgrind-out-file={output/label}.callgrind",
                       "--toggle-collect=*run_work*", "--branch-sim=yes", str(binary), arm,
                       workload, str(negative), "4096"]
            receipt["commands"].append(command)
            with (output / f"{label}.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            text = (output / f"{label}.callgrind").read_text()
            events = next(line.split()[1:] for line in text.splitlines() if line.startswith("events:"))
            values = next(line.split()[1:] for line in text.splitlines() if line.startswith("summary:"))
            counts.append({"workload": workload, "negative_percent": negative, "arm": arm,
                           "pixels": 4096, **dict(zip(events, map(int, values)))})
            (output / "counts.json").write_text(json.dumps(counts, indent=2)+"\n")
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"counts": len(counts), "binary_sha256": receipt["binary_sha256"]}))

if __name__ == "__main__":
    main()
