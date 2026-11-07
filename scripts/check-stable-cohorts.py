#!/usr/bin/env python3
"""Run an explicit serialized qualification plan in owned isolated roots.

Usage: check-stable-cohorts.py PLAN_JSON OUTPUT_DIRECTORY OWNER
Each plan row contains repo (absolute path), label, and commands (argv lists).
The caller supplies the resource wrapper. Receipts record failures and keep
subsequent independent roots running; no command is silently skipped.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    output = Path(sys.argv[2]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    owner = sys.argv[3]
    receipts = []
    for row in plan:
        repo = Path(row["repo"])
        marker = repo / ".workongoing"
        if not marker.exists() or owner not in marker.read_text():
            raise RuntimeError(f"repository ownership missing: {repo}")
        env = dict(os.environ, CARGO_TARGET_DIR=str(output.parent / "targets" / row["label"]))
        receipt = dict(label=row["label"], stages=[])
        receipt["source"] = subprocess.check_output(
            ["jj", "log", "-r", "@-", "--no-graph", "-T", "commit_id"], cwd=repo, text=True)
        receipts.append(receipt)
        for index, command in enumerate(row["commands"]):
            marker.write_text(datetime.now(timezone.utc).isoformat()
                              + f" {owner} stable qualification {row['label']} stage {index}\n")
            stage = dict(command=command, started_utc=datetime.now(timezone.utc).isoformat())
            receipt["stages"].append(stage)
            (output / "results.json").write_text(json.dumps(receipts, indent=2) + "\n")
            with (output / f"{row['label']}-{index}.log").open("w") as log:
                process = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=log)
                stage["pid"] = process.pid
                (output / "results.json").write_text(json.dumps(receipts, indent=2) + "\n")
                stage["rc"] = process.wait()
            stage["ended_utc"] = datetime.now(timezone.utc).isoformat()
            (output / "results.json").write_text(json.dumps(receipts, indent=2) + "\n")
            print(json.dumps(dict(label=row["label"], stage=stage)), flush=True)
    if any(stage["rc"] for receipt in receipts for stage in receipt["stages"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
