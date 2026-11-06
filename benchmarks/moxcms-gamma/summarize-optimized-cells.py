#!/usr/bin/env python3
"""Validate retained optimized cells and export their process-level comparisons."""
import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--allow-partial",action="store_true",help="export only qualified cells and report all failures")
    args = parser.parse_args()
    study = json.loads((args.study / "study.json").read_text())
    assert len(study["cells"]) == 45
    rows = []
    rejected=[]
    counts = defaultdict(Counter)
    for index, cell in enumerate(study["cells"]):
        directory = args.study / f"cell-{index:02}"
        experiments = list(directory.glob("processes/*/experiment.json"))
        assert len(experiments) == 1
        experiment = json.loads(experiments[0].read_text())
        assert cell["status"] == experiment["status"]
        if experiment["status"] != "complete":
            rejected.append({"group":cell["group"],"status":experiment["status"]})
            assert args.allow_partial, rejected
            continue
        digest = experiment["evidence_sha256"]
        data = json.loads(gzip.decompress((directory / "objects" / f"{digest}.json.gz").read_bytes()))
        assert not data["unreliable"] and not data["incomplete"]
        assert data["execution"]["binary_sha256"] == study["binary_sha256"]
        assert len(data["replicates"]) == 5
        for process in data["replicates"]:
            assert not process["unreliable"] and not process["incomplete"]
            for group in process["comparisons"]:
                assert group["completed_rounds"] == 30
                assert all(sample["gate_clean"] for sample in group["samples"])
        means = {bench["name"]: bench["summary"]["mean"] / 1000
                 for bench in data["comparisons"][0]["benchmarks"]}
        for comparison in data["process_comparisons"]:
            assert comparison["baseline_processes"] == comparison["candidate_processes"] == 5
            assert comparison["family_confidence"] == study["cell_confidence"]
            name = comparison["candidate"]
            counts[name][comparison["verdict"]] += 1
            rows.append({"group": cell["group"], "candidate": name,
                         "baseline": comparison["baseline"],
                         "baseline_mean_us": means[comparison["baseline"]], "candidate_mean_us": means[name],
                         **{key: comparison[key] for key in
                            ["estimate_pct", "lower_pct", "upper_pct", "verdict"]},
                         "evidence_sha256": digest})
    with args.csv.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"qualified":len(study["cells"])-len(rejected),"required":45,"rejected":rejected,"verdicts":counts}, indent=2))


if __name__ == "__main__":
    main()
