#!/usr/bin/env python3
"""Turn bench_stride_aliasing output into markdown tables.

Usage: stride_aliasing_tables.py LOG [LOG2 ...] [--rows ROWLOG ...]
Each log is one full run; the tables put the runs side by side.
Cells are zenbench's 95% CI of the power-of-two stride's time against the +64
stride (positive = the power of two is slower).
"""
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
HDR = re.compile(r"^\s{2}(?P<name>\S.*?) {2,}20 rounds")
ARM = re.compile(r"^\s*[├╰]─ (?P<arm>.*?)\s{2,}(?P<mean>[\d.]+) ±[\d.]+(?P<unit>\S+)\s+\[(?P<ci>[^\]]+)\]")
PCT = re.compile(r"^([+-]?[\d.]+)%[–](.+)%$")


def parse(path):
    groups = {}
    cur = None
    for line in open(path, encoding="utf-8"):
        line = ANSI.sub("", line.rstrip("\n"))
        m = HDR.match(line)
        if m:
            cur = m.group("name")
            groups[cur] = []
            continue
        m = ARM.match(line)
        if m and cur:
            groups[cur].append((m.group("arm").strip(), m.group("mean"), m.group("unit"), m.group("ci")))
    return groups


def ci_bounds(ci):
    m = re.match(r"^([+-]?[\d.]+)%[–]([+-]?[\d.]+)%$", ci)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def fmt(ci):
    return short(ci)


def filter_tables(groups):
    out = {}
    for name, arms in groups.items():
        m = re.match(r"^(?P<tier>.*?) \| K=(?P<k>\d+) \| row (?P<w>\d+) floats \| out@(?P<o>\d+) \| (?P<size>L2|memory):", name)
        if not m or len(arms) != 2:
            continue
        key = (m["tier"], m["o"])
        out.setdefault(key, {})[(int(m["w"]) * 4, int(m["k"]), m["size"])] = arms[1][3]
    return out


def row_tables(groups):
    out = {}
    for name, arms in groups.items():
        m = re.match(r"^(?P<tier>.*?) \| row-in row-out \| (?P<size>.*)$", name)
        if m:
            out[(m["tier"], m["size"])] = arms
    return out


def short(ci):
    b = ci_bounds(ci)
    if not b:
        return ci
    return f"{b[0]:+.1f}..{b[1]:+.1f}"


def mid(ci):
    b = ci_bounds(ci)
    return None if not b else (b[0] + b[1]) / 2


def render(paths, row_paths=None):
    runs = [parse(p) for p in paths]
    fts = [filter_tables(g) for g in runs]
    lines = []
    unstable = 0
    cells_total = 0
    keys = sorted(fts[0], key=lambda kv: (kv[1] != "2048", kv[0]))
    for tier, o in keys:
        for k in (2, 8, 12):
            lines.append(f"\n#### {tier}, K={k}, out@{o}\n")
            hdr = "| Stride B |"
            for size in ("L2", "memory"):
                for r in range(len(runs)):
                    hdr += f" {size} r{r + 1} |"
            lines.append(hdr)
            lines.append("|---|" + "---|" * (2 * len(runs)))
            for p in sorted({key[0] for key in fts[0][(tier, o)]}):
                row = f"| {p} |"
                for size in ("L2", "memory"):
                    cis = [ft[(tier, o)].get((p, k, size), "") for ft in fts]
                    ms = [mid(c) for c in cis]
                    cells_total += 1
                    flag = ""
                    if all(m is not None for m in ms) and max(ms) - min(ms) > 5:
                        flag = " *"
                        unstable += 1
                    for n, c in enumerate(cis):
                        row += f" {short(c)}{flag if n == len(cis) - 1 else ''} |"
                lines.append(row)
    lines.append(f"\n`*` marks {unstable} of {cells_total} (stride, K, size) cells whose two runs' interval midpoints differ by more than 5 percentage points.")
    rruns = [parse(p) for p in (row_paths or paths)]
    lines.append("\n#### Row in, row out: each placement's time against `dst-src = 4096n + 64`\n")
    rts = [row_tables(g) for g in rruns]
    names = [a[0] for a in list(rts[0].values())[0][1:]]
    lines.append("| Tier, size, run | baseline (+64) | " + " | ".join(names) + " |")
    lines.append("|---|---|" + "---|" * len(names))
    for key in sorted(rts[0]):
        for r, rt in enumerate(rts):
            base = f"{rt[key][0][1]} {rt[key][0][2]}"
            cells = [short(rt[key][a][3]) for a in range(1, len(names) + 1)]
            lines.append(f"| {key[0]}, {key[1]}, run {r + 1} | {base} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    args = sys.argv[1:]
    rows = None
    if "--rows" in args:
        i = args.index("--rows")
        args, rows = args[:i], args[i + 1:]
    print(render(args, rows))
