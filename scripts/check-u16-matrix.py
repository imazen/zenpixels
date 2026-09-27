#!/usr/bin/env python3
"""Exhaustive arithmetic evidence for docs/u16-contract-matrix.md.

Checks proposed numeric contracts, not production raw-sample API support.
No dependencies and no image scans in library code are introduced by this script.
"""
from fractions import Fraction
from pathlib import Path


def nearest_positive(value):
    whole, remainder = divmod(value.numerator, value.denominator)
    return whole + (2 * remainder >= value.denominator)


def rescale(q, n, m):
    maximum = (1 << n) - 1
    return (q * ((1 << m) - 1) + maximum // 2) // maximum


def replicate(q, n, m):
    value, bits = q, n
    while bits < m:
        value = (value << n) | q
        bits += n
    return value >> (bits - m)


def main():
    depths = (8, 10, 12, 16)
    checked = 0
    for n in depths:
        for m in depths:
            previous = -1
            for q in range(1 << n):
                value = rescale(q, n, m)
                oracle = nearest_positive(Fraction(q * ((1 << m) - 1), (1 << n) - 1))
                assert value == oracle
                assert previous <= value <= (1 << m) - 1
                previous = value
                if m >= n:
                    assert rescale(value, m, n) == q
                checked += 1
            assert rescale(0, n, m) == 0
            assert rescale((1 << n) - 1, n, m) == (1 << m) - 1

    # Every supported code and every legal padding shift. A padding-bit read
    # policy masks all unused bits, including deliberately nonzero padding.
    for n in depths:
        mask = (1 << n) - 1
        for shift in range(17 - n):
            padding = 65535 ^ (mask << shift)
            for q in range(1 << n):
                word = q << shift
                assert (word >> shift) & mask == q
                assert ((word | padding) >> shift) & mask == q

    lines = [
        "| Expansion | Replication differs from nearest | Largest difference | First differing input: replicated / nearest |",
        "|---|---:|---:|---|",
    ]
    for n in depths:
        for m in depths:
            if m <= n:
                continue
            differences = [(q, replicate(q, n, m), rescale(q, n, m))
                           for q in range(1 << n)
                           if replicate(q, n, m) != rescale(q, n, m)]
            error = max((abs(r - exact) for _, r, exact in differences), default=0)
            assert error <= 1
            first = "none" if not differences else "{}: {} / {}".format(*differences[0])
            lines.append(f"| {n}→{m} | {len(differences)} / {1 << n} | {error} | {first} |")

    for value in range(65536):
        t = min(value + 128, 65535)
        assert (t - (t >> 8)) >> 8 == nearest_positive(Fraction(value, 257))
        assert (((value + 128) // 257) * 257 == value) == (value % 257 == 0)
    for n in (8, 10, 12):
        for q in range(1 << n):
            assert rescale(q, n, 8) == rescale(rescale(q, n, 16), 16, 8)
        midpoint = 1 << (n - 1)
        # Chroma must scale around its offset, preserving neutral exactly.
        neutral = Fraction((midpoint - midpoint) * 65535, (1 << n) - 1) + 32768
        assert neutral == 32768
        assert rescale(midpoint, n, 16) != 32768
        for anchor in (16, 128, 235, 240):
            assert (anchor << (n - 8)) << (16 - n) == anchor << 8

    # Byte reversibility is insufficient for a narrow-domain reduction.
    assert 0x10 * 257 == 0x1010
    assert Fraction(0x1010 - 4096, 56064) != Fraction(0x10 - 16, 219)
    for q in range(256):
        assert Fraction((q << 8) - 4096, 56064) == Fraction(q - 16, 219)

    matrix = Path(__file__).resolve().parents[1] / "docs/u16-contract-matrix.md"
    table = "\n".join(lines)
    assert table in matrix.read_text(), "Replication table differs; expected:\n" + table
    print(table)
    print(f"PASS: {checked} full-range mappings; packing, inverse widening, "
          "replication, U16 narrowing, neutral chroma and narrow anchors checked.")


if __name__ == "__main__":
    main()
