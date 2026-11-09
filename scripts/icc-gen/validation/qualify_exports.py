#!/usr/bin/env python3
"""Scalar qualification of exact export ICCs; no diagnostic image/map arrays saved.

Requires NumPy and independently built C LCMS2.20 and Rust profile_eval tools.
Keeps source profiles, scalar results, output hashes and bounded worst witnesses.
Each run uses a new output directory. A failed cell is recorded and fails the run.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pq(values):
    x = np.clip(values.astype(np.float64), 0, 1) ** (32 / 2523)
    return (np.maximum(x - 107 / 128, 0) / (2413 / 128 - 2392 / 128 * x)) ** (8192 / 1305)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", type=Path)
    parser.add_argument("c_reader", type=Path)
    parser.add_argument("rust_reader", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--phase", choices=["pq", "sdr"], required=True)
    parser.add_argument("--primaries", type=int, help="Explicit bounded pilot, not full-grid coverage")
    parser.add_argument("--reuse-inputs", type=Path, help="Recheck an original failing receipt's immutable float inputs")
    args = parser.parse_args()
    args.output.mkdir()
    report = {"phase": args.phase, "primaries_filter": args.primaries,
              "numpy": np.__version__, "pq_domain": "[0,1], constant end segments clamp encoded excursions",
              "zero_gate": "All neutral/active U16 ramps; mixed-grid losses remain reported and bounded by linear error, including float32 PCS cancellation",
              "reader_sha256": {
                  "c": sha(args.c_reader.read_bytes()), "rust": sha(args.rust_reader.read_bytes())},
              "limits": {"linear_absolute": 0.0005, "c_rust_absolute": 0.000001}, "cells": []}
    failures = []
    cache = {}

    def samples(channels, dense):
        key = channels, dense
        if key in cache:
            return cache[key]
        if args.reuse_inputs:
            if not dense:
                raise ValueError("reuse-inputs requires the explicit PQ phase")
            source = args.reuse_inputs / f"input-{channels}-dense.f32"
            inverse = args.reuse_inputs / f"linear-{channels}-dense.f32"
            values = np.frombuffer(source.read_bytes(), dtype=np.float32).reshape(-1, channels)
            linear = np.frombuffer(inverse.read_bytes(), dtype=np.float32).reshape(-1, channels)
            cache[key] = values, linear, source, inverse
            return cache[key]
        if dense:
            ramp = np.arange(65536, dtype=np.float64).astype(np.float32) / np.float32(65535)
            neutral = np.repeat(ramp[:, None], channels, axis=1)
            parts = [neutral]
            if channels == 3:
                for channel in range(3):
                    colored = np.zeros_like(neutral)
                    colored[:, channel] = ramp
                    parts.append(colored)
            values = np.concatenate(parts)
            if channels == 3:
                grid = [-0.1, 0, 1/65535, 0.001, 0.01, 0.18, 0.5, 1, 1.1]
                values = np.concatenate([values, np.asarray(list(itertools.product(grid, repeat=3)), dtype=np.float32)])
        else:
            grid = [-0.1, 0, 1 / 65535, 0.0001, 0.01, 0.18, 0.5, 0.75, 1, 1.1]
            values = np.asarray(list(itertools.product(grid, repeat=channels)), dtype=np.float32)
        source = args.output / f"input-{channels}-{'dense' if dense else 'sparse'}.f32"
        source.write_bytes(values.tobytes())
        inverse = args.output / f"linear-{channels}-{'dense' if dense else 'sparse'}.f32"
        linear = pq(values).astype(np.float32)
        inverse.write_bytes(linear.tobytes())
        cache[key] = values, linear, source, inverse
        return cache[key]

    def evaluate(engine, src, dst, input_path, intent):
        tool = args.c_reader if engine == "c" else args.rust_reader
        proc = subprocess.run([str(tool), str(src), str(dst), str(input_path), "-", str(intent)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode:
            raise RuntimeError(f"{engine} exit {proc.returncode}: {proc.stderr.decode(errors='replace')}")
        result = np.frombuffer(proc.stdout, dtype=np.float32)
        if not np.all(np.isfinite(result)):
            raise RuntimeError(f"{engine} non-finite output")
        return result, sha(proc.stdout)

    profiles = sorted(args.profiles.glob("*.icc"))
    assert len(profiles) == 313, f"need complete 313-profile recipe, got {len(profiles)}"
    for path in profiles:
        kind, p, transfer = path.stem.split("-"); p, transfer = int(p), int(transfer)
        if (transfer == 16) != (args.phase == "pq"):
            continue
        if args.primaries is not None and p != args.primaries:
            continue
        channels = 1 if kind == "gray" else 3
        values, linear_values, source_input, inverse_input = samples(channels, args.phase == "pq")
        target = args.profiles / "linear" / f"{kind}-{p}.icc"
        for direction in (["forward", "reverse"] if args.phase == "pq" else ["forward"]):
            src, dst = (path, target) if direction == "forward" else (target, path)
            input_path = source_input if direction == "forward" else inverse_input
            for intent in ([0, 1, 2, 3] if args.phase == "pq" else [1]):
                cell = {"profile": path.name, "profile_sha256": sha(path.read_bytes()),
                        "direction": direction, "intent": intent, "pixels": len(values)}
                try:
                    c, c_hash = evaluate("c", src, dst, input_path, intent)
                    rust, rust_hash = evaluate("rust", src, dst, input_path, intent)
                    assert c.size == rust.size == values.size
                    difference = np.abs(c.astype(np.float64) - rust.astype(np.float64))
                    index = int(np.argmax(difference))
                    cell.update(c_sha256=c_hash, rust_sha256=rust_hash,
                                different_channels=int(np.count_nonzero(c != rust)),
                                max_c_rust_error=float(difference[index]),
                                worst_oracle={"pixel": index // channels, "channel": index % channels,
                                              "input": values[index // channels].tolist(),
                                              "c": float(c[index]), "rust": float(rust[index])})
                    if cell["max_c_rust_error"] > report["limits"]["c_rust_absolute"]:
                        failures.append([path.name, direction, intent, "C/Rust difference"])
                    if args.phase == "pq":
                        cell["analytic"] = {}
                        for engine, actual in [("c", c), ("rust", rust)]:
                            expected = linear_values.astype(np.float64).ravel()
                            actual_linear = actual.astype(np.float64) if direction == "forward" else pq(actual)
                            errors = np.abs(actual_linear - expected)
                            worst = int(np.argmax(errors))
                            a = {"max_linear_error": float(errors[worst]), "worst_pixel": worst // channels,
                                 "worst_channel": worst % channels,
                                 "positive_to_zero": int(np.count_nonzero((actual == 0) & (values.ravel() > 0))),
                                 "max_encoded_error": float(np.max(np.abs(actual.astype(np.float64) - np.clip(values.ravel(), 0, 1)))) if direction == "reverse" else None}
                            ramp_elements = 65536 * channels * (4 if channels == 3 else 1)
                            a["positive_to_zero_ramps"] = int(np.count_nonzero((actual[:ramp_elements] == 0) & (values.ravel()[:ramp_elements] > 0)))
                            cell["analytic"][engine] = a
                            # The relative float path is the production contract.
                            # Other intents are measured, including bounded legacy absolute.
                            if intent == 1 and (a["max_linear_error"] > report["limits"]["linear_absolute"] or a["positive_to_zero_ramps"]):
                                failures.append([path.name, direction, intent, engine, "analytic precision/zero loss"])
                            if intent == 1:
                                shaped = actual.reshape(-1, channels)
                                monotonic = bool(np.all(np.diff(shaped[:65536], axis=0) >= 0))
                                if channels == 3:
                                    monotonic &= all(np.all(np.diff(shaped[(i+1)*65536:(i+2)*65536, i]) >= 0) for i in range(3))
                                a["active_ramps_monotonic"] = monotonic
                                if not monotonic:
                                    failures.append([path.name, direction, intent, engine, "nonmonotonic active ramp"])
                        if intent == 1:
                            prod, prod_hash = evaluate("rust", src, dst, input_path, "production")
                            cell["production_sha256"] = prod_hash
                            cell["production_diff_channels"] = int(np.count_nonzero(prod != rust))
                            if cell["production_diff_channels"]:
                                failures.append([path.name, direction, intent, "production/direct mismatch"])
                except Exception as error:
                    cell["error"] = str(error)
                    failures.append([path.name, direction, intent, str(error)])
                report["cells"].append(cell)
                report["failures"] = failures
                (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(cell), flush=True)
    print(f"{len(report['cells'])} cells; {len(failures)} failures", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
