#!/usr/bin/env python3
"""Qualify frozen zencodec source against one committed zenpixels cohort."""
import argparse
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
import tomllib


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def snapshot(root, output):
    revision = git(root, "rev-parse", "HEAD").decode().strip()
    if git(root, "status", "--porcelain", "--untracked-files=no").strip():
        raise RuntimeError(f"tracked changes present in {root}; commit before qualifying")
    output.mkdir()
    with tarfile.open(fileobj=io.BytesIO(git(root, "archive", revision))) as archive:
        archive.extractall(output, filter="data")
    return revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("consumer", type=Path, help="clean zencodec checkout")
    parser.add_argument("output", type=Path, help="fresh durable run directory")
    args = parser.parse_args()
    core = Path(__file__).resolve().parents[2]
    consumer = args.consumer.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "preparing", "commands": []}

    def save():
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    def run(command, label, json_output=False):
        receipt["commands"].append(command)
        receipt["stage"] = label
        receipt["status"] = "running"
        save()
        with (output / f"{label}.log").open("wb") as log:
            if json_output:
                with (output / f"{label}.json").open("wb") as data:
                    completed = subprocess.run(command, stdout=data, stderr=log)
            else:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"{label} exited {completed.returncode}; see {output / (label + '.log')}")

    try:
        receipt["zenpixels"] = snapshot(core, output / "zenpixels")
        receipt["zencodec"] = snapshot(consumer, output / "zencodec")
        core_copy = output / "zenpixels"
        consumer_copy = output / "zencodec"
        dependencies = tomllib.loads((core_copy / "Cargo.toml").read_text())["patch"]["crates-io"]
        # Preserve the exact core numerical dependencies; Cargo does not inherit
        # patches from path dependencies or parent workspaces.
        patches = "[patch.crates-io]\n"
        for name, value in dependencies.items():
            fields = ", ".join(f"{k} = {json.dumps(v)}" for k, v in value.items())
            patches += f"{name} = {{ {fields} }}\n"
        for name in ["zenpixels", "zenpixels-convert"]:
            patches += f"{name} = {{ path = {json.dumps(str(core_copy / name))} }}\n"
        for path in [consumer_copy / "Cargo.toml", consumer_copy / "zencodec-render/Cargo.toml"]:
            text = path.read_text()
            text, count = re.subn(r"(?ms)^\[patch\.crates-io\]\n.*?(?=^\[|\Z)", lambda _: patches + "\n", text)
            if count != 1:
                raise RuntimeError(f"expected one patch table: {path}")
            path.write_text(text)
        receipt["effective_patches"] = patches
        for name, manifest in [("codec", consumer_copy / "Cargo.toml"), ("render", consumer_copy / "zencodec-render/Cargo.toml")]:
            # Resolve separately: the render crate is a standalone workspace.
            command = ["cargo", "metadata", "--manifest-path", str(manifest), "--format-version", "1"]
            run(command, name + "-metadata", json_output=True)
            metadata = json.loads((output / (name + "-metadata.json")).read_text())
            for package in ["zenpixels", "zenpixels-convert"]:
                matches = [p for p in metadata["packages"] if p["name"] == package]
                if not matches and name == "codec" and package == "zenpixels-convert":
                    continue
                if len(matches) != 1 or Path(matches[0]["manifest_path"]).parent != core_copy / package:
                    raise RuntimeError(f"{name}: expected exactly one selected {package} from the cohort")
            command = ["cargo", "test", "--manifest-path", str(manifest), "--all-targets"]
            if name == "codec":
                command += ["--workspace", "--features", "std"]
            run(command, name + "-tests")
        receipt["status"] = "passed"
        save()
        print(f"Cohort passed: {output / 'receipt.json'}")
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = str(error)
        save()
        raise


if __name__ == "__main__":
    main()
