#!/usr/bin/env python3
# Do RGB samples reconstructed from 10-bit BT.709 YCbCr land on the 10-bit grid once stored as
# canonical full-range U16? Float recipe as zenpixels-convert ycbcr.rs (limited range, 4:4:4),
# then nearest U16 quantization of each in-range RGB value.
import random
def F(n, m, q):
    a, b = (1 << n) - 1, (1 << m) - 1
    return (q * b + a // 2) // a
def on10(v): return F(10, 16, F(16, 10, v)) == v
kr, kb = 0.2126, 0.0722; kg = 1 - kr - kb
random.seed(7)
total = on = 0
for _ in range(200000):
    y, cb, cr = random.randint(64, 940), random.randint(64, 960), random.randint(64, 960)
    l = (y - 64) / 876.0; u = (cb - 512) / 896.0; v = (cr - 512) / 896.0
    for c in (l + 2 * (1 - kr) * v, l - 2 * kb * (1 - kb) / kg * u - 2 * kr * (1 - kr) / kg * v, l + 2 * (1 - kb) * u):
        if 0.0 <= c <= 1.0:
            total += 1; on += on10(int(c * 65535 + 0.5))
print(f"in-range RGB samples {total}, on the 10-bit grid {on} ({100*on/total:.2f}%); a random U16 value is on it with p={1024/65536:.4f}")
# Neutral gray (cb = cr = 512) maps Y to R=G=B=(y-64)/876: on the 10-bit grid only when (y-64)*1023/876 is integral-ish.
gray_on = sum(on10(int((y - 64) / 876 * 65535 + 0.5)) for y in range(64, 941))
print(f"neutral gray codes 64..940: {gray_on} of 877 on the 10-bit grid")
