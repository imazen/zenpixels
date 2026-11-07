#!/usr/bin/env python3
"""Summarize resolved Cargo metadata without copying local filesystem paths.

Usage: inspect-cohort.py METADATA_JSON OUTPUT_JSON
This records selected identities; it does not claim build or pixel parity.
"""

import json
from pathlib import Path
import sys


def main():
    metadata = json.loads(Path(sys.argv[1]).read_text())
    families = {name: [] for name in ["archmage", "archmage-macros", "magetypes",
                "moxcms", "zenpixels", "zenpixels-convert", "zencodec", "garb", "linear-srgb"]}
    caps = []
    for package in metadata["packages"]:
        if package["name"] in families:
            families[package["name"]].append({"version": package["version"],
                                               "source": package["source"]})
        for dependency in package["dependencies"]:
            if dependency["name"] in ["archmage", "magetypes"]:
                requirement = dependency["req"]
                if "<=0.9.31-beta" in requirement or requirement.startswith("=0.9.31-beta"):
                    caps.append({"package": package["name"], "source": package["source"],
                                 "dependency": dependency["name"], "requirement": requirement})
    result = {"selected_families": families, "temporary_beta_ceilings": caps,
              "qualification": "resolved identities only; no compile or numeric claim"}
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
