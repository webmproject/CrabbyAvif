#!/usr/bin/env python3

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generates kernels.rs: straight-line 1D inverse transforms (AV1 spec
7.13.2) with zero propagation.

Usage: python3 src/dsp/itx/gen_kernels.py > src/dsp/itx/kernels.rs

The butterfly networks below are transcriptions of the spec pseudo code
(the same as the reference loops in itx.rs). They are executed
symbolically: every array entry is either the constant 0 or an SSA
variable, so that

* all loops, indices and angles are resolved at generation time, and
* for a variant with only the first `nz` inputs non-zero, every operation
  whose result is known to be zero is dropped and products with a zero
  operand are omitted.

Dropping operations on zeros is exact: B() of two zeros gives
Round2(0, 12) = 0, H() of two zeros gives clamp(0) = 0. Clamps of
non-zero values are always kept.

Products are computed in i64 (as in the reference code), so the kernels
are exact for any input that fits the spec's Dequant range.
"""

import sys

COS128 = [
    4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973, 3948, 3920, 3889, 3857, 3822,
    3784, 3745, 3703, 3659, 3612, 3564, 3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967,
    2896, 2824, 2751, 2675, 2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019, 1931, 1842, 1751, 1660,
    1567, 1474, 1380, 1285, 1189, 1092, 995, 897, 799, 700, 601, 501, 401, 301, 201, 101, 0,
]


def cos128(angle):
    a = angle & 255
    if a <= 64:
        return COS128[a]
    if a <= 128:
        return -COS128[128 - a]
    if a <= 192:
        return -COS128[a - 128]
    return COS128[256 - a]


def sin128(angle):
    return cos128(angle - 64)


def brev(bits, x):
    r = 0
    for i in range(bits):
        r |= ((x >> i) & 1) << (bits - 1 - i)
    return r


ZERO = None


# Lane kernels (column pass) use i32 products. Their inputs and all
# Hadamard outputs are clamped to colClampRange <= 18 bits, so |v| <=
# 2^17; the generator tracks a magnitude bound of every value and checks
# that no 32-bit sum can overflow.
LANE_IN_BOUND = 1 << 17
I32_MAX = (1 << 31) - 1


class Net:
    """Symbolic transform state. Entries are None (zero) or a var name.

    `lanes`: emit code over `[i32; L]` lane arrays with i32 products
    instead of scalar code with i64 products."""

    def __init__(self, n, nz, lanes=False):
        self.n = n
        self.lanes = lanes
        self.lines = []
        self.count = 0
        self.bound = {}
        self.t = [f"i{i}" if i < nz else ZERO for i in range(n)]
        for i in range(nz):
            self.lines.append(f"let i{i} = t[{i}];")
            self.bound[f"i{i}"] = LANE_IN_BOUND

    def var(self, expr, bound=0):
        name = f"v{self.count}"
        self.count += 1
        self.lines.append(f"let {name} = {expr};")
        self.bound[name] = bound
        return name

    def perm(self, idx):
        copy = list(self.t)
        self.t = [copy[idx(i)] for i in range(self.n)]

    def B(self, a, b, angle, flip):
        ta, tb = self.t[a], self.t[b]
        c, s = cos128(angle), sin128(angle)

        def lin(terms):
            terms = [(v, k) for v, k in terms if v is not ZERO and k != 0]
            if not terms:
                return ZERO
            if self.lanes:
                total = sum(self.bound[v] * abs(k) for v, k in terms) + 2048
                assert total <= I32_MAX, "lane kernel may overflow"
                bound = (total >> 12) + 1
                if len(terms) == 1:
                    (v, k), = terms
                    return self.var(f"lmul1(&{v}, {k})", bound)
                (v0, k0), (v1, k1) = terms
                return self.var(f"lmul2(&{v0}, {k0}, &{v1}, {k1})", bound)
            expr = ""
            for v, k in terms:
                if expr == "":
                    expr = f"{v} as i64 * {k}" if k > 0 else f"{v} as i64 * -{-k}"
                else:
                    expr += f" + {v} as i64 * {k}" if k > 0 else f" - {v} as i64 * {-k}"
            return self.var(f"r12({expr})")

        x = lin([(ta, c), (tb, -s)])
        y = lin([(ta, s), (tb, c)])
        if flip:
            x, y = y, x
        self.t[a], self.t[b] = x, y

    def H(self, a, b, flip):
        if flip:
            a, b = b, a
        x, y = self.t[a], self.t[b]
        if x is ZERO and y is ZERO:
            return
        if self.lanes:
            B = LANE_IN_BOUND
            for v in (x, y):
                if v is not ZERO:
                    assert self.bound[v] * 2 <= I32_MAX
            if y is ZERO:
                sa = self.var(f"lclamp({x}, r)", B)
                self.t[a], self.t[b] = sa, sa
            elif x is ZERO:
                self.t[a] = self.var(f"lclamp({y}, r)", B)
                self.t[b] = self.var(f"lclamp(lneg({y}), r)", B)
            else:
                self.t[a] = self.var(f"ladd(&{x}, &{y}, r)", B)
                self.t[b] = self.var(f"lsub(&{x}, &{y}, r)", B)
            return
        if y is ZERO:
            sa = self.var(f"r.apply({x})")
            self.t[a], self.t[b] = sa, sa
        elif x is ZERO:
            self.t[a] = self.var(f"r.apply({y})")
            self.t[b] = self.var(f"r.apply(-{y})")
        else:
            self.t[a] = self.var(f"r.apply({x} + {y})")
            self.t[b] = self.var(f"r.apply({x} - {y})")

    def neg(self, i):
        if self.t[i] is not ZERO:
            v = self.t[i]
            if self.lanes:
                self.t[i] = self.var(f"lneg({v})", self.bound[v])
            else:
                self.t[i] = self.var(f"-{v}")

    def finish(self):
        zero = "[0; L]" if self.lanes else "0"
        for i, v in enumerate(self.t):
            self.lines.append(f"t[{i}] = {v if v is not ZERO else zero};")
        return self.lines


def dct(N, nz, lanes=False):
    n0 = 1 << N
    s = Net(n0, nz, lanes)
    B, H = s.B, s.H
    s.perm(lambda i: brev(N, i))
    if N == 6:
        for i in range(16):
            B(32 + i, 63 - i, 63 - 4 * brev(4, i), False)
    if N >= 5:
        for i in range(8):
            B(16 + i, 31 - i, 6 + (brev(3, 7 - i) << 3), False)
    if N == 6:
        for i in range(16):
            H(32 + i * 2, 33 + i * 2, i & 1)
    if N >= 4:
        for i in range(4):
            B(8 + i, 15 - i, 12 + (brev(2, 3 - i) << 4), False)
    if N >= 5:
        for i in range(8):
            H(16 + 2 * i, 17 + 2 * i, i & 1)
    if N == 6:
        for i in range(4):
            for j in range(2):
                B(62 - i * 4 - j, 33 + i * 4 + j, 60 - 16 * brev(2, i) + 64 * j, True)
    if N >= 3:
        for i in range(2):
            B(4 + i, 7 - i, 56 - 32 * i, False)
    if N >= 4:
        for i in range(4):
            H(8 + 2 * i, 9 + 2 * i, i & 1)
    if N >= 5:
        for i in range(2):
            for j in range(2):
                B(30 - 4 * i - j, 17 + 4 * i + j, 24 + (j << 6) + ((1 - i) << 5), True)
    if N == 6:
        for i in range(8):
            for j in range(2):
                H(32 + i * 4 + j, 35 + i * 4 - j, i & 1)
    for i in range(2):
        B(2 * i, 2 * i + 1, 32 + 16 * i, i == 0)
    if N >= 3:
        for i in range(2):
            H(4 + 2 * i, 5 + 2 * i, i != 0)
    if N >= 4:
        for i in range(2):
            B(14 - i, 9 + i, 48 + 64 * i, True)
    if N >= 5:
        for i in range(4):
            for j in range(2):
                H(16 + 4 * i + j, 19 + 4 * i - j, i & 1)
    if N == 6:
        for i in range(2):
            for j in range(4):
                B(61 - i * 8 - j, 34 + i * 8 + j, 56 - i * 32 + (j >> 1) * 64, True)
    for i in range(2):
        H(i, 3 - i, False)
    if N >= 3:
        B(6, 5, 32, True)
    if N >= 4:
        for i in range(2):
            for j in range(2):
                H(8 + 4 * i + j, 11 + 4 * i - j, i != 0)
    if N >= 5:
        for i in range(4):
            B(29 - i, 18 + i, 48 + (i >> 1) * 64, True)
    if N == 6:
        for i in range(4):
            for j in range(4):
                H(32 + 8 * i + j, 39 + 8 * i - j, i & 1)
    if N >= 3:
        for i in range(4):
            H(i, 7 - i, False)
    if N >= 4:
        for i in range(2):
            B(13 - i, 10 + i, 32, True)
    if N >= 5:
        for i in range(2):
            for j in range(4):
                H(16 + i * 8 + j, 23 + i * 8 - j, i != 0)
    if N == 6:
        for i in range(8):
            B(59 - i, 36 + i, 48 if i < 4 else 112, True)
    if N >= 4:
        for i in range(8):
            H(i, 15 - i, False)
    if N >= 5:
        for i in range(4):
            B(27 - i, 20 + i, 32, True)
    if N == 6:
        for i in range(8):
            H(32 + i, 47 - i, False)
            H(48 + i, 63 - i, True)
    if N >= 5:
        for i in range(16):
            H(i, 31 - i, False)
    if N == 6:
        for i in range(8):
            B(55 - i, 40 + i, 32, True)
    if N == 6:
        for i in range(32):
            H(i, 63 - i, False)
    return s.finish()


def adst_in(s):
    n0 = s.n
    s.perm(lambda i: i - 1 if i & 1 else n0 - i - 1)


def adst_out(s):
    n0 = s.n
    log2 = n0.bit_length() - 1

    def idx(i):
        a = (i >> 3) & 1
        b = ((i >> 2) & 1) ^ ((i >> 3) & 1)
        c = ((i >> 1) & 1) ^ ((i >> 2) & 1)
        d = (i & 1) ^ ((i >> 1) & 1)
        return ((d << 3) | (c << 2) | (b << 1) | a) >> (4 - log2)

    s.perm(idx)
    for i in range(1, n0, 2):
        s.neg(i)


def adst8(nz, lanes=False):
    s = Net(8, nz, lanes)
    B, H = s.B, s.H
    adst_in(s)
    for i in range(4):
        B(2 * i, 2 * i + 1, 60 - 16 * i, True)
    for i in range(4):
        H(i, 4 + i, False)
    for i in range(2):
        B(4 + 3 * i, 5 + i, 48 - 32 * i, True)
    for i in range(2):
        for j in range(2):
            H(4 * j + i, 2 + 4 * j + i, False)
    for i in range(2):
        B(2 + 4 * i, 3 + 4 * i, 32, True)
    adst_out(s)
    return s.finish()


def adst16(nz, lanes=False):
    s = Net(16, nz, lanes)
    B, H = s.B, s.H
    adst_in(s)
    for i in range(8):
        B(2 * i, 2 * i + 1, 62 - 8 * i, True)
    for i in range(8):
        H(i, 8 + i, False)
    for i in range(2):
        B(8 + 2 * i, 9 + 2 * i, 56 - 32 * i, True)
        B(13 + 2 * i, 12 + 2 * i, 8 + 32 * i, True)
    for i in range(4):
        for j in range(2):
            H(8 * j + i, 4 + 8 * j + i, False)
    for i in range(2):
        for j in range(2):
            B(4 + 8 * j + 3 * i, 5 + 8 * j + i, 48 - 32 * i, True)
    for i in range(2):
        for j in range(4):
            H(4 * j + i, 2 + 4 * j + i, False)
    for i in range(4):
        B(2 + 4 * i, 3 + 4 * i, 32, True)
    adst_out(s)
    return s.finish()


def emit(name, doc, lines, out, test_only=False):
    uses_r = any("r.apply" in l for l in lines)
    out.append(f"/// {doc}")
    if test_only:
        out.append("#[cfg(test)]")
    out.append("#[rustfmt::skip]")
    out.append(f"pub(super) fn {name}(t: &mut [i32; 64], {'r' if uses_r else '_r'}: Clamp) {{")
    for l in lines:
        out.append("    " + l)
    out.append("}")
    out.append("")


def emit_lanes(name, doc, n, lines, out):
    uses_r = any(", r)" in l for l in lines)
    out.append(f"/// {doc} Lane version (column pass, i32 arithmetic).")
    out.append("#[rustfmt::skip]")
    out.append(f"pub(super) fn {name}<const L: usize>(t: &mut [[i32; L]], {'r' if uses_r else '_r'}: Clamp) {{")
    out.append(f"    let t: &mut [[i32; L]; {n}] = (&mut t[..{n}]).try_into().unwrap();")
    for l in lines:
        out.append("    " + l)
    out.append("}")
    out.append("")


def main():
    out = []
    out.extend([
        "// Copyright 2026 Google LLC",
        "//",
        "// Licensed under the Apache License, Version 2.0 (the \"License\");",
        "// you may not use this file except in compliance with the License.",
        "// You may obtain a copy of the License at",
        "//",
        "//     http://www.apache.org/licenses/LICENSE-2.0",
        "//",
        "// Unless required by applicable law or agreed to in writing, software",
        "// distributed under the License is distributed on an \"AS IS\" BASIS,",
        "// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
        "// See the License for the specific language governing permissions and",
        "// limitations under the License.",
        "",
    ])
    out.append("//! Generated by gen_kernels.py; do not edit.")
    out.append("//!")
    out.append("//! Straight-line 1D inverse transforms (spec 7.13.2). `*_nzK`")
    out.append("//! variants assume that only the first `K` inputs are non-zero.")
    out.append("//! Each kernel reads `t[..K]` and writes `t[..n]`.")
    out.append("")
    out.append("#![allow(clippy::all, unused_variables)]")
    out.append("")
    out.append("use super::{ladd, lclamp, lmul1, lmul2, lneg, lsub, r12, Clamp};")
    out.append("")
    dct_variants = {2: [4], 3: [8, 4], 4: [16, 8, 4], 5: [32, 16, 8, 4], 6: [64, 32, 16, 8, 4]}
    for N, nzs in dct_variants.items():
        for nz in nzs:
            n = 1 << N
            # AV1 never codes DCT64 inputs 32..63; the dense version is
            # only used by tests.
            doc = f"Inverse DCT{n} with inputs `{nz}..` zero." if nz < n else f"Inverse DCT{n}."
            emit(f"idct{n}_nz{nz}", doc, dct(N, nz), out, test_only=(nz == 64))
    for nz in [8, 4]:
        emit(f"iadst8_nz{nz}", f"Inverse ADST8 with inputs `{nz}..` zero.", adst8(nz), out)
    for nz in [16, 8, 4]:
        emit(f"iadst16_nz{nz}", f"Inverse ADST16 with inputs `{nz}..` zero.", adst16(nz), out)
    for N, nzs in dct_variants.items():
        for nz in nzs:
            n = 1 << N
            if nz == 64:
                continue
            emit_lanes(f"idct{n}_nz{nz}_l", f"Inverse DCT{n} with inputs `{nz}..` zero.", n,
                       dct(N, nz, True), out)
    for nz in [8, 4]:
        emit_lanes(f"iadst8_nz{nz}_l", f"Inverse ADST8 with inputs `{nz}..` zero.", 8,
                   adst8(nz, True), out)
    for nz in [16, 8, 4]:
        emit_lanes(f"iadst16_nz{nz}_l", f"Inverse ADST16 with inputs `{nz}..` zero.", 16,
                   adst16(nz, True), out)
    sys.stdout.write("\n".join(out) + "\n")


main()
