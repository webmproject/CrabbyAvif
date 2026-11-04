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

Usage (from the crate root):
  python3 src/decoder/av1/dsp/itx/gen_kernels.py
    > src/decoder/av1/dsp/itx/kernels.rs
  python3 src/decoder/av1/dsp/itx/gen_kernels.py --simd
    > src/decoder/av1/dsp/simd/itx/kernels.rs

With `--simd` the kernels are emitted generic over the lane vector type
of the `std::simd` inverse transforms (`dsp::simd::itx::TxLanes`); the
magnitude checks of the lane kernels apply to them too, so they are exact
with `i32` lanes for inputs and clamping ranges of at most 18 bits.

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
from fractions import Fraction

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
    instead of scalar code with i64 products. `lanes == "simd"`: emit
    code generic over a `TxLanes` vector type."""

    def __init__(self, n, nz, lanes=False, in_map=None):
        self.n = n
        self.nz = nz
        self.lanes = lanes
        self.in_map = in_map
        self.lines = []
        self.count = 0
        self.bound = {}
        self.loaded = set()
        self.def_line = {}
        self.origin = {}
        self.pending_barrier = False
        self.t = [f"i{i}" if i < nz else ZERO for i in range(n)]
        for i in range(nz):
            self.bound[f"i{i}"] = LANE_IN_BOUND

    def barrier(self):
        if self.lanes == "simd" and self.loaded and (self.nz > 2 or self.n > 32):
            self.pending_barrier = True

    def use(self, v):
        if v is not ZERO and v.startswith("i") and v not in self.loaded:
            if self.pending_barrier:
                self.lines.append("let t = std::hint::black_box(&mut *t);")
                self.pending_barrier = False
            self.loaded.add(v)
            idx = int(v[1:])
            src_idx = self.in_map(idx) if self.in_map else idx
            self.lines.append(f"let {v} = t[{src_idx}];")

    def var(self, expr, bound=0):
        name = f"v{self.count}"
        self.count += 1
        self.def_line[name] = len(self.lines)
        self.lines.append(f"let {name} = {expr};")
        self.bound[name] = bound
        return name

    def mark_h0(self, v):
        orig = self.origin.get(v, v)
        if orig in self.def_line:
            idx = self.def_line[orig]
            self.lines[idx] = self.lines[idx].replace(".m1(", ".m1_h0(").replace(".m2(", ".m2_h0(")

    def perm(self, idx):
        copy = list(self.t)
        self.t = [copy[idx(i)] for i in range(self.n)]

    def B(self, a, b, angle, flip):
        ta, tb = self.t[a], self.t[b]
        self.use(ta)
        self.use(tb)
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
                    if self.lanes == "simd":
                        return self.var(f"{v}.m1({k})", bound)
                    return self.var(f"lmul1(&{v}, {k})", bound)
                (v0, k0), (v1, k1) = terms
                if self.lanes == "simd":
                    return self.var(f"{v0}.m2({k0}, {v1}, {k1})", bound)
                return self.var(f"lmul2(&{v0}, {k0}, &{v1}, {k1})", bound)
            expr = ""
            for v, k in terms:
                if expr == "":
                    expr = f"{v} as i64 * {k}" if k > 0 else f"{v} as i64 * -{-k}"
                else:
                    expr += f" + {v} as i64 * {k}" if k > 0 else f" - {v} as i64 * {-k}"
            return self.var(f"r12({expr})")

        if tb is ZERO and c == s:
            x = lin([(ta, c)])
            self.t[a], self.t[b] = x, x
            return
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
        self.use(x)
        self.use(y)
        if self.lanes:
            B = LANE_IN_BOUND
            for v in (x, y):
                if v is not ZERO:
                    assert self.bound[v] * 2 <= I32_MAX
            if self.lanes == "simd":
                if y is ZERO:
                    self.mark_h0(x)
                    sa = self.var(f"r.apply({x})", B)
                    self.origin[sa] = self.origin.get(x, x)
                    self.t[a], self.t[b] = sa, sa
                elif x is ZERO:
                    self.mark_h0(y)
                    sa = self.var(f"r.apply({y})", B)
                    self.origin[sa] = self.origin.get(y, y)
                    self.t[a] = sa
                    self.t[b] = self.var(f"r.apply(-{y})", B)
                else:
                    self.t[a] = self.var(f"r.apply({x} + {y})", B)
                    self.t[b] = self.var(f"r.apply({x} - {y})", B)
                return
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
            self.use(v)
            if self.lanes == "simd":
                self.t[i] = self.var(f"-{v}", self.bound[v])
            elif self.lanes:
                self.t[i] = self.var(f"lneg({v})", self.bound[v])
            else:
                self.t[i] = self.var(f"-{v}")

    def finish(self):
        zero = {"simd": "V::splat(0)", True: "[0; L]", False: "0"}[self.lanes]
        for v in self.t:
            self.use(v)
        for i, v in enumerate(self.t):
            self.lines.append(f"t[{i}] = {v if v is not ZERO else zero};")
        return self.lines


def dct16_body(s):
    B, H = s.B, s.H
    for e in range(2):
        for k in range(2 * e, 2 * e + 2):
            B(8 + k, 15 - k, 12 + (brev(2, 3 - k) << 4), False)
        for m in (e, 3 - e):
            H(8 + 2 * m, 9 + 2 * m, m & 1)
        B(14 - e, 9 + e, 48 + 64 * e, True)
    for i in range(2):
        for j in range(2):
            H(8 + 4 * i + j, 11 + 4 * i - j, i != 0)
    for i in range(2):
        B(13 - i, 10 + i, 32, True)
    s.barrier()
    for i in range(2):
        B(4 + i, 7 - i, 56 - 32 * i, False)
    for i in range(2):
        H(4 + 2 * i, 5 + 2 * i, i != 0)
    B(6, 5, 32, True)
    for i in range(2):
        B(2 * i, 2 * i + 1, 32 + 16 * i, i == 0)
    for i in range(2):
        H(i, 3 - i, False)
    for i in range(4):
        H(i, 7 - i, False)
    for i in range(8):
        H(i, 15 - i, False)


def dct(N, nz, lanes=False, net=Net):
    n0 = 1 << N
    s = net(n0, nz, lanes)
    B, H = s.B, s.H
    s.perm(lambda i: brev(N, i))
    if N == 6:
        for h in range(2):
            for q in range(2 * h, 2 * h + 2):
                if q > 0:
                    s.barrier()
                for e in range(2 * q, 2 * q + 2):
                    for k in range(2 * e, 2 * e + 2):
                        B(32 + k, 63 - k, 63 - 4 * brev(4, k), False)
                    for m in (e, 15 - e):
                        H(32 + m * 2, 33 + m * 2, m & 1)
                    B(62 - q * 4 - (e & 1), 33 + q * 4 + (e & 1), 60 - 16 * brev(2, q) + 64 * (e & 1), True)
                for m in (q, 7 - q):
                    for j in range(2):
                        H(32 + m * 4 + j, 35 + m * 4 - j, m & 1)
                for j in range(2 * (q & 1), 2 * (q & 1) + 2):
                    i = q >> 1
                    B(61 - i * 8 - j, 34 + i * 8 + j, 56 - i * 32 + (j >> 1) * 64, True)
            for m in (h, 3 - h):
                for j in range(4):
                    H(32 + 8 * m + j, 39 + 8 * m - j, m & 1)
            for i in range(4 * h, 4 * h + 4):
                B(59 - i, 36 + i, 48 if i < 4 else 112, True)
        for i in range(8):
            H(32 + i, 47 - i, False)
            H(48 + i, 63 - i, True)
        for i in range(8):
            B(55 - i, 40 + i, 32, True)
        s.barrier()
    if N >= 5:
        for q in range(2):
            if q > 0:
                s.barrier()
            for e in range(2 * q, 2 * q + 2):
                for k in range(2 * e, 2 * e + 2):
                    B(16 + k, 31 - k, 6 + (brev(3, 7 - k) << 3), False)
                for m in (e, 7 - e):
                    H(16 + 2 * m, 17 + 2 * m, m & 1)
                B(30 - 4 * q - (e & 1), 17 + 4 * q + (e & 1), 24 + ((e & 1) << 6) + ((1 - q) << 5), True)
            for m in (q, 3 - q):
                for j in range(2):
                    H(16 + 4 * m + j, 19 + 4 * m - j, m & 1)
            for i in range(2 * q, 2 * q + 2):
                B(29 - i, 18 + i, 48 + (i >> 1) * 64, True)
        for i in range(2):
            for j in range(4):
                H(16 + i * 8 + j, 23 + i * 8 - j, i != 0)
        for i in range(4):
            B(27 - i, 20 + i, 32, True)
        s.barrier()
    if N >= 4:
        dct16_body(s)
    else:
        if N >= 3:
            for i in range(2):
                B(4 + i, 7 - i, 56 - 32 * i, False)
            for i in range(2):
                H(4 + 2 * i, 5 + 2 * i, i != 0)
            B(6, 5, 32, True)
        for i in range(2):
            B(2 * i, 2 * i + 1, 32 + 16 * i, i == 0)
        for i in range(2):
            H(i, 3 - i, False)
        if N >= 3:
            for i in range(4):
                H(i, 7 - i, False)
    if N >= 5:
        for i in range(16):
            H(i, 31 - i, False)
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


def adst8(nz, lanes=False, net=Net):
    s = net(8, nz, lanes)
    B, H = s.B, s.H
    adst_in(s)
    for k in range(2):
        B(2 * k, 2 * k + 1, 60 - 16 * k, True)
        B(4 + 2 * k, 5 + 2 * k, 60 - 16 * (2 + k), True)
        H(2 * k, 4 + 2 * k, False)
        H(2 * k + 1, 5 + 2 * k, False)
    for i in range(2):
        B(4 + 3 * i, 5 + i, 48 - 32 * i, True)
    for j in range(2):
        for i in range(2):
            H(4 * j + i, 2 + 4 * j + i, False)
        B(2 + 4 * j, 3 + 4 * j, 32, True)
    adst_out(s)
    return s.finish()


def adst16(nz, lanes=False, net=Net):
    s = net(16, nz, lanes)
    B, H = s.B, s.H
    adst_in(s)
    for k in range(4):
        B(2 * k, 2 * k + 1, 62 - 8 * k, True)
        B(8 + 2 * k, 9 + 2 * k, 62 - 8 * (4 + k), True)
        H(2 * k, 8 + 2 * k, False)
        H(2 * k + 1, 8 + 2 * k + 1, False)
        if k < 2:
            B(8 + 2 * k, 9 + 2 * k, 56 - 32 * k, True)
        else:
            i = k - 2
            B(13 + 2 * i, 12 + 2 * i, 8 + 32 * i, True)
    for j in range(2):
        for i in range(4):
            H(8 * j + i, 4 + 8 * j + i, False)
        for i in range(2):
            B(4 + 8 * j + 3 * i, 5 + 8 * j + i, 48 - 32 * i, True)
    for j in range(4):
        for i in range(2):
            H(4 * j + i, 2 + 4 * j + i, False)
        B(2 + 4 * j, 3 + 4 * j, 32, True)
    adst_out(s)
    return s.finish()


class BoundNet:
    """Exact magnitude analysis of a network (same interface as `Net`).

    Every entry is None (zero) or `(c, e)`: the value is `sum(c[k] * x_k)`
    of the inputs `x_k` plus a rounding error of magnitude at most `e`
    (`Round2(s, 12)` is `s / 4096` plus an error in `(-1/2, 1/2]`). Clamps
    are assumed inactive. `finish` returns the largest `|c[k]|` and `e`
    over all the intermediate and output values."""

    def __init__(self, n, nz, lanes=False, in_map=None):
        self.n = n
        self.t = [(tuple(Fraction(int(i == k)) for k in range(n)), Fraction(0))
                  if i < nz else ZERO for i in range(n)]
        self.max_c = Fraction(0)
        self.max_e = Fraction(0)

    def note(self, v):
        self.max_c = max(self.max_c, max(abs(c) for c in v[0]))
        self.max_e = max(self.max_e, v[1])
        return v

    def barrier(self):
        pass

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
            coef = tuple(sum(k * v[0][i] for v, k in terms) / 4096 for i in range(self.n))
            err = sum(abs(k) * v[1] for v, k in terms) / 4096 + Fraction(1, 2)
            return self.note((coef, err))

        if tb is ZERO and c == s:
            x = lin([(ta, c)])
            self.t[a], self.t[b] = x, x
            return
        x = lin([(ta, c), (tb, -s)])
        y = lin([(ta, s), (tb, c)])
        if flip:
            x, y = y, x
        self.t[a], self.t[b] = x, y

    def H(self, a, b, flip):
        if flip:
            a, b = b, a
        x, y = self.t[a], self.t[b]

        def add(p, q, sign):
            if p is ZERO:
                return (tuple(sign * c for c in q[0]), q[1])
            if q is ZERO:
                return p
            return self.note((tuple(i + sign * j for i, j in zip(p[0], q[0])), p[1] + q[1]))

        if x is ZERO and y is ZERO:
            return
        self.t[a], self.t[b] = add(x, y, 1), add(x, y, -1)

    def neg(self, i):
        if self.t[i] is not ZERO:
            c, e = self.t[i]
            self.t[i] = (tuple(-x for x in c), e)

    def finish(self):
        return self.max_c, self.max_e


# `S16_ROW_L1_MAX` in simd/itx/large.rs relies on these bounds: every
# value of the DCT / ADST networks is `sum(c_k * x_k)` with `|c_k| <= 1`
# plus at most `S16_ROW_ERR_MAX` of accumulated rounding error.
S16_ROW_ERR_MAX = 63


def check_s16_row_bound():
    """Checks the bounds of `S16_ROW_ERR_MAX` for all (dense) networks."""
    nets = [(f"DCT{1 << N}", lambda N=N: dct(N, 1 << N, net=BoundNet)) for N in range(2, 7)]
    nets += [("ADST8", lambda: adst8(8, net=BoundNet)), ("ADST16", lambda: adst16(16, net=BoundNet))]
    for name, run in nets:
        max_c, max_e = run()
        assert max_c <= 1, f"{name}: coefficient {float(max_c)} > 1"
        assert max_e <= S16_ROW_ERR_MAX, f"{name}: rounding error {float(max_e)} > {S16_ROW_ERR_MAX}"


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


def emit_simd(name, doc, n, lines, out, inline_never=False):
    uses_r = any("r.apply" in l or ", r)" in l for l in lines)
    out.append(f"/// {doc}")
    if inline_never:
        out.append("#[inline(never)]")
    out.append("#[rustfmt::skip]")
    out.append(f"pub(super) fn {name}<V: TxLanes>(t: &mut [V], {'r' if uses_r else '_r'}: Clamp<V>) {{")
    out.append(f"    let t: &mut [V; {n}] = (&mut t[..{n}]).try_into().unwrap();")
    for l in lines:
        out.append("    " + l)
    out.append("}")
    out.append("")


def emit_simd_odd(name, n, lines, out):
    uses_r = any("r.apply" in l for l in lines)
    out.append("#[inline(never)]")
    out.append("#[rustfmt::skip]")
    out.append(f"fn {name}<V: TxLanes>(t: &mut [V; {n}], {'r' if uses_r else '_r'}: Clamp<V>) {{")
    for l in lines:
        out.append("    " + l)
    out.append("}")
    out.append("")


def dct32_simd(nz, out):
    s = Net(32, nz, "simd")
    B, H = s.B, s.H
    s.perm(lambda i: brev(5, i))
    for q in range(2):
        for e in range(2 * q, 2 * q + 2):
            for k in range(2 * e, 2 * e + 2):
                B(16 + k, 31 - k, 6 + (brev(3, 7 - k) << 3), False)
            for m in (e, 7 - e):
                H(16 + 2 * m, 17 + 2 * m, m & 1)
            B(30 - 4 * q - (e & 1), 17 + 4 * q + (e & 1), 24 + ((e & 1) << 6) + ((1 - q) << 5), True)
        for m in (q, 3 - q):
            for j in range(2):
                H(16 + 4 * m + j, 19 + 4 * m - j, m & 1)
        for i in range(2 * q, 2 * q + 2):
            B(29 - i, 18 + i, 48 + (i >> 1) * 64, True)
    for i in range(2):
        for j in range(4):
            H(16 + i * 8 + j, 23 + i * 8 - j, i != 0)
    for i in range(4):
        B(27 - i, 20 + i, 32, True)
    for i in range(16):
        s.use(s.t[16 + i])
    for i in range(16):
        v = s.t[16 + i]
        s.lines.append(f"t[{16 + i}] = {v if v is not ZERO else 'V::splat(0)'};")
    emit_simd_odd(f"idct32_odd_nz{nz}", 32, s.lines, out)

    s_even = Net(16, nz // 2, "simd", in_map=lambda i: 2 * i if i < 8 else 2 * (i - 8) + 1)
    s_even.perm(lambda i: brev(4, i))
    dct16_body(s_even)
    for i in range(16):
        s_even.use(s_even.t[i])
    for i in range(16):
        a = s_even.t[i]
        b = f"b{i}"
        s_even.lines.append(f"let {b} = t[{31 - i}];")
        s_even.lines.append(f"t[{i}] = r.apply({a} + {b});")
        s_even.lines.append(f"t[{31 - i}] = r.apply({a} - {b});")

    lines = []
    if nz == 32:
        for i in range(8, 16):
            lines.append(f"let e{i} = t[{2 * i}];")
    lines.append(f"idct32_odd_nz{nz}(t, r);")
    if nz == 32:
        for i in range(8, 16):
            lines.append(f"t[{2 * (i - 8) + 1}] = e{i};")
    lines.extend(s_even.lines)
    doc = f"Inverse DCT32 with inputs `{nz}..` zero." if nz < 32 else "Inverse DCT32."
    emit_simd(f"idct32_nz{nz}", doc, 32, lines, out)


def dct64_simd(nz, out):
    for h in range(2):
        s = Net(64, nz, "simd")
        B, H = s.B, s.H
        s.perm(lambda i: brev(6, i))
        for q in range(2 * h, 2 * h + 2):
            for e in range(2 * q, 2 * q + 2):
                for k in range(2 * e, 2 * e + 2):
                    B(32 + k, 63 - k, 63 - 4 * brev(4, k), False)
                for m in (e, 15 - e):
                    H(32 + m * 2, 33 + m * 2, m & 1)
                B(62 - q * 4 - (e & 1), 33 + q * 4 + (e & 1), 60 - 16 * brev(2, q) + 64 * (e & 1), True)
            for m in (q, 7 - q):
                for j in range(2):
                    H(32 + m * 4 + j, 35 + m * 4 - j, m & 1)
            for j in range(2 * (q & 1), 2 * (q & 1) + 2):
                i = q >> 1
                B(61 - i * 8 - j, 34 + i * 8 + j, 56 - i * 32 + (j >> 1) * 64, True)
        for m in (h, 3 - h):
            for j in range(4):
                H(32 + 8 * m + j, 39 + 8 * m - j, m & 1)
        for i in range(4 * h, 4 * h + 4):
            B(59 - i, 36 + i, 48 if i < 4 else 112, True)
        idxs = [8 * m + j for m in (h, 3 - h) for j in range(8)]
        for idx in idxs:
            s.use(s.t[32 + idx])
        for idx in idxs:
            v = s.t[32 + idx]
            s.lines.append(f"t[{32 + idx}] = {v if v is not ZERO else 'V::splat(0)'};")
        emit_simd_odd(f"idct64_odd{h}_nz{nz}", 64, s.lines, out)

    s_tail = Net(32, 32, "simd")
    s_tail.loaded = {f"i{i}" for i in range(32)}
    s_tail.t = [f"t[{32 + i}]" for i in range(32)]
    for i in range(32):
        s_tail.bound[f"t[{32 + i}]"] = LANE_IN_BOUND
    for i in range(4):
        s_tail.H(i, 15 - i, False)
        s_tail.H(16 + i, 31 - i, True)
        s_tail.B(16 + i, 15 - i, 32, True)
        for idx in (i, 15 - i, 16 + i, 31 - i):
            s_tail.lines.append(f"t[{32 + idx}] = {s_tail.t[idx]};")
        s_tail.H(7 - i, 8 + i, False)
        s_tail.H(23 - i, 24 + i, True)
        s_tail.B(23 - i, 8 + i, 32, True)
        for idx in (7 - i, 8 + i, 23 - i, 24 + i):
            s_tail.lines.append(f"t[{32 + idx}] = {s_tail.t[idx]};")

    lines = [
        f"idct64_odd0_nz{nz}(t, r);",
        f"idct64_odd1_nz{nz}(t, r);",
    ] + s_tail.lines + [
        f"t[{i}] = t[{2 * i}];" for i in range(1, nz // 2)
    ]
    even_nz = max(2, nz // 2)
    lines.append(f"idct32_nz{even_nz}(t, r);")
    lines.append("for i in 0..32 {")
    lines.append("    let a = t[i];")
    lines.append("    let b = t[63 - i];")
    lines.append("    t[i] = r.apply(a + b);")
    lines.append("    t[63 - i] = r.apply(a - b);")
    lines.append("}")
    doc = f"Inverse DCT64 with inputs `{nz}..` zero."
    emit_simd(f"idct64_nz{nz}", doc, 64, lines, out)


def main_simd():
    check_s16_row_bound()
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
    out.append("//! Generated by src/decoder/av1/dsp/itx/gen_kernels.py --simd; do not edit.")
    out.append("//!")
    out.append("//! Straight-line 1D inverse transforms (spec 7.13.2) over lane")
    out.append("//! vectors (one transform per lane). `*_nzK` variants assume that")
    out.append("//! only the first `K` inputs are non-zero. Each kernel reads")
    out.append("//! `t[..K]` and writes `t[..n]`.")
    out.append("")
    out.append("#![allow(clippy::all, unused_variables)]")
    out.append("")
    out.append("use super::{Clamp, TxLanes};")
    out.append("")
    dct_variants = {2: [4, 2], 3: [8, 4, 2], 4: [16, 8, 4, 2]}
    for N, nzs in dct_variants.items():
        for nz in nzs:
            n = 1 << N
            doc = f"Inverse DCT{n} with inputs `{nz}..` zero." if nz < n else f"Inverse DCT{n}."
            emit_simd(f"idct{n}_nz{nz}", doc, n, dct(N, nz, "simd"), out)
    for nz in [32, 16, 8, 4]:
        dct32_simd(nz, out)
    emit_simd("idct32_nz2", "Inverse DCT32 with inputs `2..` zero.", 32, dct(5, 2, "simd"), out)
    for nz in [32, 16, 8, 4]:
        dct64_simd(nz, out)
    emit_simd("idct64_nz2", "Inverse DCT64 with inputs `2..` zero.", 64, dct(6, 2, "simd"), out)
    for nz in [8, 4, 2, 1]:
        doc = f"Inverse ADST8 with inputs `{nz}..` zero." if nz < 8 else "Inverse ADST8."
        emit_simd(f"iadst8_nz{nz}", doc, 8, adst8(nz, "simd"), out)
    for nz in [16, 8, 4, 2, 1]:
        doc = f"Inverse ADST16 with inputs `{nz}..` zero." if nz < 16 else "Inverse ADST16."
        emit_simd(f"iadst16_nz{nz}", doc, 16, adst16(nz, "simd"), out)
    sys.stdout.write("\n".join(out) + "\n")


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


if "--simd" in sys.argv[1:]:
    main_simd()
else:
    main()
