#!/usr/bin/env python3
"""Resolve prepared isolated Cargo roots, retaining every diagnostic receipt.

Usage: resolve-stable-cohorts.py ROOTS_JSON OUTPUT_DIRECTORY
Run through the shared resource wrapper. This edits only generated lockfiles;
it neither changes manifests nor makes a build or numerical qualification claim.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    inventory = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    roots = json.loads(inventory.read_text())
    receipts = []
    for row in roots:
        name = row["repo"]
        repo = inventory.parent / name
        marker = repo / ".workongoing"
        if not marker.exists() or "codex-116" not in marker.read_text():
            raise RuntimeError(f"ownership missing: {name}")
        marker.write_text(datetime.now(timezone.utc).isoformat()
                          + " codex-116 isolated stable dependency resolution\n")
        env = dict(os.environ, CARGO_TARGET_DIR=str(output / "targets" / name))
        log_path = output / f"{name}-resolution.log"
        candidate = subprocess.check_output(
            ["jj", "log", "-r", "@-", "--no-graph", "-T", "commit_id"], cwd=repo, text=True)
        result = dict(repo=name, prepared_candidate=row["candidate"], candidate=candidate, stages=[])
        with log_path.open("w") as log:
            def run(args, suffix):
                dest = output / f"{name}-{suffix}.json"
                log.write("COMMAND " + json.dumps(args) + "\n")
                log.flush()
                with dest.open("w") as stdout:
                    proc = subprocess.run(args, cwd=repo, env=env,
                                          stdout=stdout, stderr=log)
                result["stages"].append(dict(command=args, rc=proc.returncode,
                                               stdout=dest.name))
                return proc.returncode, dest

            rc, metadata = run(["cargo", "metadata", "--format-version", "1"], "initial")
            if rc == 0:
                # One update can move several packages. Re-read identities after
                # each update instead of targeting an already removed package ID.
                for family in ["archmage", "archmage-macros", "magetypes"]:
                    while True:
                        packages = json.loads(metadata.read_text())["packages"]
                        versions = sorted({p["version"] for p in packages
                                           if p["name"] == family and p["version"].startswith("0.9.")
                                           and p["version"] != "0.9.31"
                                           and (p["source"] or "").startswith("registry+")})
                        if not versions:
                            break
                        version = versions[0]
                        rc, _ = run(["cargo", "update", "-p", f"{family}@{version}",
                                     "--precise", "0.9.31"], f"update-{family}-{version}")
                        if rc:
                            break
                        rc, metadata = run(["cargo", "metadata", "--format-version", "1"],
                                           f"after-{family}-{version}")
                        if rc:
                            break
                    if rc:
                        break
                if rc == 0:
                    rc, metadata = run(["cargo", "metadata", "--locked", "--format-version", "1"],
                                       "locked")
                if rc == 0:
                    packages = json.loads(metadata.read_text())["packages"]
                    result["families"] = {family: sorted({p["version"] for p in packages
                                                          if p["name"] == family})
                                          for family in ["archmage", "archmage-macros", "magetypes", "moxcms"]}
                    result["beta_ceilings"] = [dict(package=p["name"], dependency=d["name"], req=d["req"])
                                                for p in packages for d in p["dependencies"]
                                                if d["name"] in ["archmage", "magetypes"]
                                                and ("<=0.9.31-beta" in d["req"] or "=0.9.31-beta" in d["req"])]
            result["rc"] = rc
        receipts.append(result)
        (output / "resolution-results.json").write_text(json.dumps(receipts, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
