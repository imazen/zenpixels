#!/usr/bin/env python3
"""Run every frozen workload once as its own five-process evidence cell.

The confidence is Bonferroni-adjusted across all 45 cells; each cell's own
comparison family is adjusted by zenbench. Rejected cells are never retried.
"""
import argparse, hashlib, json, subprocess
from pathlib import Path
from datetime import datetime, timezone

def groups():
    result = []
    for width in [64, 1024, 4096]:
        for direction in ["encode", "decode"]:
            for zero, negative in [(0, 0), (0, 50), (10, 0), (100, 0)]:
                result.append(f"eval/{direction}/zero{zero}/neg{negative}/w{width}")
        for transform in ["cicp_gamma22_to_bt2020", "cicp_smpte428_to_bt2020",
                          "cicp_gamma28_to_bt2020", "adobe_to_prophoto_extended"]:
            result.append(f"transform/{transform}/w{width}")
    for width in [64,1024,4096]:
        for zero in [0,10,100]:
            result.append(f"transform/prophoto_to_srgb/zero{zero}/w{width}")
    assert len(result) == 45 and len(set(result)) == 45
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    binary = args.binary.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    names = groups()
    confidence = 1.0 - 0.05 / len(names)
    manifest = {"protocol": "moxcms-outlined-cells-v1", "family_confidence": 0.95,
                "cell_confidence": confidence, "processes": 5, "rounds": 30,
                "margin_percent": 1.0, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "cells": []}
    for index, name in enumerate(names):
        cell = output / f"cell-{index:02}"
        command = [str(binary), "--mode=evidence", "--processes=5", "--rounds=30", "--timeout=120",
                   "--workload-id=moxcms-outlined-zero-exponent-v1",
                   f"--scope=neoverse-n1/moxcms-outlined-cells-v1/{name}", f"--group={name}",
                   f"--confidence={confidence:.16f}", "--margin=1", "--format=llm"]
        manifest["cells"].append({"group": name, "output": str(cell), "command": command, "status": "planned"})
    record = output / "study.json"
    def save():
        record.write_text(json.dumps(manifest, indent=2)+"\n")
    save()
    if args.plan_only:
        print(record)
        return
    import os
    for index, cell in enumerate(manifest["cells"]):
        cell["status"] = "running"
        cell["started_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        directory = Path(cell["output"])
        environment = dict(os.environ, ZENBENCH_DATA_DIR=str(directory))
        print(f"cell {index+1}/45 starting {cell['group']}", flush=True)
        with (output / f"cell-{index:02}.log").open("w") as log:
            result = subprocess.run(cell["command"], env=environment, stdout=log, stderr=subprocess.STDOUT)
        cell["exit_code"] = result.returncode
        attempts = list(directory.glob("processes/*/experiment.json"))
        if len(attempts) == 1:
            experiment = json.loads(attempts[0].read_text())
            cell["status"] = experiment["status"]
            cell["evidence_sha256"] = experiment.get("evidence_sha256")
        else:
            cell["status"] = "failed-no-unique-manifest"
        cell["completed_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        print(f"cell {index+1}/45 {cell['status']} exit={result.returncode}", flush=True)
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    qualified = sum(cell["status"] == "complete" for cell in manifest["cells"])
    print(f"qualified cells: {qualified}/45; all failures retained", flush=True)
    raise SystemExit(0 if qualified == 45 else 2)

if __name__ == "__main__":
    main()
