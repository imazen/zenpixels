#!/usr/bin/env python3
"""Extract both backend summaries without filtering flagged measurements."""

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate", type=Path)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    with gzip.open(args.aggregate, "rt") as source:
        result = json.load(source)
    if len(result.get("replicates", [])) != 3:
        raise ValueError("expected the complete three-process aggregate")
    rows = []
    for group in result["comparisons"]:
        backends = {item["name"]: item["summary"] for item in group["benchmarks"]}
        if set(backends) != {"rapidhash_v3", "xxh3_64"}:
            raise ValueError(f"unexpected backends: {group['group_name']}")
        rapid, xxh = backends["rapidhash_v3"], backends["xxh3_64"]
        rows.append([
            group["group_name"], group["throughput"]["Bytes"],
            rapid["median"], xxh["median"], rapid["mean"], xxh["mean"],
            xxh["median"] / rapid["median"],
        ])
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow([
        "group", "sample_bytes", "rapid_median_ns", "xxh3_median_ns",
        "rapid_mean_ns", "xxh3_mean_ns", "xxh3_over_rapid_median",
    ])
    writer.writerows(rows)
    payload = output.getvalue().encode()
    if args.csv.exists():
        if args.csv.read_bytes() != payload:
            raise FileExistsError(f"preserve existing data before replacing {args.csv}")
    else:
        args.csv.write_bytes(payload)
    samples = [sample for replicate in result["replicates"]
               for group in replicate["comparisons"] for sample in group["samples"]]
    print(json.dumps({
        "aggregate": args.aggregate.name,
        "csv_sha256": hashlib.sha256(payload).hexdigest(),
        "groups": len(rows), "processes": len(result["replicates"]),
        "unreliable": result["unreliable"], "incomplete": result["incomplete"],
        "clean_rounds": sum(sample["gate_clean"] for sample in samples),
        "total_rounds": len(samples),
        "xxh3_lower_descriptive_median": sum(row[-1] < 1 for row in rows),
        "execution": result["execution"],
    }, indent=2))


if __name__ == "__main__":
    main()
