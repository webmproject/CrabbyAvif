// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! SIMD inverse transforms (see `dsp::itx` for the contract).
//!
//! Same 2D process as the portable [`crate::decoder::av1::dsp::itx::inv_txfm_add`],
//! with every 1D transform run on lane vectors, one transform per lane:
//!
//! * Row pass: `RN` rows at a time (one row per lane). Coefficient tiles
//!   of `RN x RN` are transposed on load and on store, so the row-major
//!   layout of the intermediate buffer is the same as in the portable
//!   code. Identity rows are element-wise and need no transposes.
//! * Column pass: `CN` adjacent columns at a time (one column per lane),
//!   followed by the residual add with clipping.
//!
//! The 1D kernels are generated
//! (`src/decoder/av1/dsp/itx/gen_kernels.py --simd`) and are generic over
//! the lane vector type ([`TxLanes`]).
//!
//! # Arithmetic
//!
//! `i32` lanes are exact when the inputs and the intermediate clamping
//! range have at most 18 bits (the generator checks every product sum).
//! This holds for every column pass (`colClampRange <= 18`) and for row
//! passes with `bd <= 10` (`rowClampRange = bd + 8`, and Dequant is
//! clamped to the same range by the decoder). 12-bit row passes use
//! [`Wide`] lanes, which split the products. `std::simd` integer
//! arithmetic wraps, so even values outside these ranges never panic.

use std::any::Any;
use std::ops::{Add, Neg, Sub};
use std::simd::prelude::*;
use std::simd::SimdElement;

use super::{load_i32, store_clip};
use crate::decoder::av1::dsp::itx::{
    eob_bounds, flip_lr, flip_ud, round12, round2, tx1d_kinds, Clamp as ScalarClamp, Tx1d,
};
#[cfg(test)]
use crate::decoder::av1::dsp::itx::{TRANSFORM_ROW_SHIFT, TX_HEIGHT_LOG2, TX_WIDTH_LOG2};
use crate::decoder::av1::pixel::Pixel;
use crate::decoder::av1::types::{TxSize, TxType};

mod kernels;
mod large;

/// `a.map(f)`, always inlined: `array::map` over SIMD vectors is not
/// inlined on aarch64 (its `try_map` closure is outlined, sending the
/// whole array through the stack and a call).
#[inline(always)]
pub(super) fn amap<A: Copy, B: Copy, const N: usize>(a: [A; N], f: impl Fn(A) -> B) -> [B; N] {
    let mut o = [f(a[0]); N];
    for i in 1..N {
        o[i] = f(a[i]);
    }
    o
}

/// Lane vector type of the 1D kernels.
pub(crate) trait TxLanes:
    Copy + Add<Output = Self> + Sub<Output = Self> + Neg<Output = Self>
{
    /// True if the lanes saturate to the only clamping range they are
    /// used with (16 bits), making [`Clamp::apply`] a no-op.
    const SATURATING_16: bool = false;

    fn splat(v: i32) -> Self;
    /// Lane-wise (wrapping) product with a constant.
    fn mul(self, k: i32) -> Self;
    /// Lane-wise `Round2(x, n)` (`n` may be 0).
    fn rnd(self, n: u32) -> Self;
    fn max(self, o: Self) -> Self;
    fn min(self, o: Self) -> Self;

    /// `Round2(a * ka, 12)`.
    #[inline(always)]
    fn m1(self, ka: i32) -> Self {
        self.mul(ka).rnd(12)
    }

    /// Same result as [`Self::m1`]; only a codegen hint (see
    /// `dot_rnd12_wrap`'s `FORCE_CORR`) for products whose result feeds a
    /// saturating / clamping operation directly (e.g. a single-input
    /// `H(x, 0)` into another `B()` butterfly, or the rect2 pre-scaling).
    #[inline(always)]
    fn m1_h0(self, ka: i32) -> Self {
        self.m1(ka)
    }

    /// `Round2(a * ka + b * kb, 12)`.
    #[inline(always)]
    fn m2(self, ka: i32, b: Self, kb: i32) -> Self {
        (self.mul(ka) + b.mul(kb)).rnd(12)
    }

    /// Same result as [`Self::m2`]; only a codegen hint, see
    /// [`Self::m1_h0`].
    #[inline(always)]
    fn m2_h0(self, ka: i32, b: Self, kb: i32) -> Self {
        self.m2(ka, b, kb)
    }

    /// `Round2(a * ka + b * kb, 12)` narrowed modulo 2^16 on 16-bit lanes
    /// (matching dav1d's `ITX_MUL2X_PACK` with `flags & 2 != 0` on non-conformant
    /// inputs; identical to [`Self::m2`] on conformant inputs).
    #[inline(always)]
    fn m2_wrap(self, ka: i32, b: Self, kb: i32) -> Self {
        self.m2(ka, b, kb)
    }

    /// `Round2(sum(a[i] * k[i]), 12)`.
    #[inline(always)]
    fn dot_rnd12<const K: usize>(a: [Self; K], k: [i32; K]) -> Self {
        let mut s = a[0].mul(k[0]);
        for i in 1..K {
            s = s + a[i].mul(k[i]);
        }
        s.rnd(12)
    }
}

macro_rules! impl_tx_lanes {
    ($t:ty) => {
        impl<const N: usize> TxLanes for Simd<$t, N> {
            #[inline(always)]
            fn splat(v: i32) -> Self {
                Simd::splat(v as $t)
            }
            #[inline(always)]
            fn mul(self, k: i32) -> Self {
                self * Simd::splat(k as $t)
            }
            #[inline(always)]
            fn rnd(self, n: u32) -> Self {
                (self + Simd::splat((1 << n) >> 1)) >> Simd::splat(n as $t)
            }
            #[inline(always)]
            fn max(self, o: Self) -> Self {
                self.simd_max(o)
            }
            #[inline(always)]
            fn min(self, o: Self) -> Self {
                self.simd_min(o)
            }
        }
    };
}

impl_tx_lanes!(i32);

/// `i32` lanes for values of up to 22 bits (12-bit row passes), whose
/// products with the transform constants (up to 14 bits) do not fit in
/// 32 bits. `Round2(sum(a[i] * k[i]), 12)` is computed exactly by
/// splitting every `a = 256 * ah + al` (`0 <= al < 256`):
/// `sum = 256 * H + L` with `H = sum(ah * k)` and `L = sum(al * k)`,
/// and with `H = 16 * q + r` (`0 <= r < 16`) the result is
/// `q + ((256 * r + L + 2048) >> 12)`. Every term fits in 32 bits.
#[derive(Copy, Clone)]
struct Wide<const N: usize>(Simd<i32, N>);

impl<const N: usize> Add for Wide<N> {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Wide(self.0 + o.0)
    }
}

impl<const N: usize> Sub for Wide<N> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Wide(self.0 - o.0)
    }
}

impl<const N: usize> Neg for Wide<N> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Wide(-self.0)
    }
}

impl<const N: usize> TxLanes for Wide<N> {
    #[inline(always)]
    fn splat(v: i32) -> Self {
        Wide(Simd::splat(v))
    }
    /// Only used with small constants (identity32: 4).
    #[inline(always)]
    fn mul(self, k: i32) -> Self {
        Wide(self.0 * Simd::splat(k))
    }
    #[inline(always)]
    fn rnd(self, n: u32) -> Self {
        Wide(self.0.rnd(n))
    }
    #[inline(always)]
    fn max(self, o: Self) -> Self {
        Wide(self.0.simd_max(o.0))
    }
    #[inline(always)]
    fn min(self, o: Self) -> Self {
        Wide(self.0.simd_min(o.0))
    }
    #[inline(always)]
    fn m1(self, ka: i32) -> Self {
        Self::dot_rnd12([self], [ka])
    }
    #[inline(always)]
    fn m2(self, ka: i32, b: Self, kb: i32) -> Self {
        Self::dot_rnd12([self, b], [ka, kb])
    }
    #[inline(always)]
    fn dot_rnd12<const K: usize>(a: [Self; K], k: [i32; K]) -> Self {
        #[cfg(target_arch = "aarch64")]
        if K <= 4 && k.iter().all(|&x| (-4096..=16384).contains(&x)) {
            // Widening `i32 -> i64` products (`smull` / `smlal` + `rshrn #12`),
            // avoiding negative constants like `dot_rnd12_wrap` so LLVM keeps
            // `smull`. Exact for 25-bit inputs (`|sum| < 2^40`, shifted `< 2^28`).
            let mut sum = Simd::<i64, N>::splat(0);
            let mut corr = Simd::<i32, N>::splat(0);
            for (a, k) in a.into_iter().zip(k) {
                let k = if k < 0 {
                    corr += a.0;
                    k + 4096
                } else {
                    k
                };
                sum += a.0.cast::<i64>() * Simd::splat(k as i64);
            }
            return Wide(((sum + Simd::splat(2048)) >> 12).cast::<i32>() - corr);
        }
        let mut hi = Simd::splat(0);
        let mut lo = Simd::splat(2048);
        for i in 0..K {
            let kv = Simd::splat(k[i]);
            hi += (a[i].0 >> 8) * kv;
            lo += (a[i].0 & Simd::splat(255)) * kv;
        }
        let q = hi >> 4;
        let r = (hi & Simd::splat(15)) << 8;
        Wide(q + ((r + lo) >> 12))
    }
}

/// `i16` lanes for transforms whose intermediate clamping range is 16
/// bits: all 8-bit passes and 10-bit column passes (`rowClampRange =
/// bd + 8`, `colClampRange = max(bd + 6, 16)`).
///
/// Every value stored by the spec's butterflies and Hadamard steps then
/// fits in 16 bits: the Hadamard outputs are clamped to 16 bits, which is
/// exactly a saturating add / subtract, and conformance requires the
/// 2-input `B` butterfly outputs `Round2(x, 12)` to be representable with
/// `r = 16` bits (7.13.2.1). The products are computed widened to `i32`
/// (inputs of 16 bits times constants of at most 14 bits, at most 4 terms,
/// fit) and narrowed with saturation (on aarch64, 1- and 2-term `B`
/// products with constants `|k| <= 4096` are narrowed modulo 2^16 instead,
/// see `dot_rnd12_wrap`; `ADST4`'s `dot_rnd12` saturates because 7.13.2.6
/// only bounds the unrounded 4-term sums to `r + 12` bits `< 2^27`, which
/// allows `Round2(x, 12) == 32768` on conformant streams). Negations
/// saturate (`-(-32768)` gives 32767 instead of 32768, which is the same
/// after the clamp / `Round2` that follows every negated output).
///
/// For non-conformant streams the results may differ from the spec (but
/// all arithmetic is wrapping or saturating, so it never panics).
/// Identity transforms, whose outputs can exceed 16 bits, are not run on
/// these lanes (see `row_pass_identity` and `col_pass16`).
#[derive(Copy, Clone)]
pub(crate) struct S16<const N: usize>(Simd<i16, N>);

impl<const N: usize> Add for S16<N> {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        S16(self.0.saturating_add(o.0))
    }
}

impl<const N: usize> Sub for S16<N> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        S16(self.0.saturating_sub(o.0))
    }
}

impl<const N: usize> Neg for S16<N> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        S16(Simd::<i16, N>::splat(0).saturating_sub(self.0))
    }
}

/// `Round2(x, 12)` narrowed to `i16` with saturation.
///
/// On aarch64 this compiles to `add` + `sqshrn #12` (the `add` usually
/// folds into a preceding `mla`). `x + 2^11` cannot overflow for the
/// products of `i16` lanes and 12-bit constants this is used for.
#[inline(always)]
fn narrow_rnd12<const N: usize>(x: Simd<i32, N>) -> Simd<i16, N> {
    ((x + Simd::splat(1 << 11)) >> 12)
        .simd_clamp(Simd::splat(i16::MIN as i32), Simd::splat(i16::MAX as i32))
        .cast()
}

/// `Round2(sum(a[i] * k[i]), 12)` of `i16` lanes for `|k[i]| <= 4096`,
/// narrowed to 16 bits *modulo 2^16*: exact whenever the result fits in
/// 16 bits, which it does for conformant streams (see `S16`); otherwise
/// it wraps. Returns `None` if some `|k[i]| > 4096` (or off aarch64).
///
/// On aarch64 this is `smull` / `smlal` + `rshrn #12` (the same as the
/// `sqrshrn` sequence without saturation), plus one `sub` per negative
/// constant: a negative `k` is applied as `a * (k + 4096) - (a << 12)`,
/// i.e. `Round2(.., 12) - a`. Negative constants are avoided because LLVM
/// shrinks them to the low 28 bits that the result depends on, which
/// defeats `smull`.
///
/// With `FORCE_CORR`, if all constants are non-negative the first one is
/// applied the same way (`a * (k + 4096)`, then `- a`), so that there is
/// always one `sub`. `Round2(s + a * 4096, 12) == Round2(s, 12) + a`
/// exactly, so the result is bit-identical to `FORCE_CORR = false`; only
/// the instruction selection on aarch64 changes.
#[inline(always)]
fn dot_rnd12_wrap<const N: usize, const K: usize, const FORCE_CORR: bool>(
    a: [Simd<i16, N>; K],
    k: [i32; K],
) -> Option<Simd<i16, N>> {
    if !cfg!(target_arch = "aarch64") || k.iter().any(|k| k.abs() > 4096) {
        return None;
    }
    // `|sum| <= K * 2^27 < 2^31` for the `K <= 4` used here.
    debug_assert!(K <= 8);
    let all_pos = FORCE_CORR && k.iter().all(|&x| x >= 0);
    let mut sum = Simd::<i32, N>::splat(0);
    let mut corr = Simd::<i16, N>::splat(0);
    for (i, (&a, &k)) in a.iter().zip(&k).enumerate() {
        let k = if k < 0 || (all_pos && i == 0) {
            corr += a;
            k + 4096
        } else {
            k
        };
        sum += a.cast::<i32>() * Simd::splat(k);
    }
    Some(((sum + Simd::splat(1 << 11)) >> 12).cast::<i16>() - corr)
}

/// `sum(ab[2 * i + e] * k[e])` over `e = 0, 1`: lane `i` is the sum of the
/// products of the interleaved pair `ab[2 * i], ab[2 * i + 1]` with
/// `ka, kb`. Compiles to `pmaddwd` on x86 (exact for `|k| < 2^15`, which
/// holds for all transform constants).
#[inline(always)]
fn pmadd8(ab: Simd<i16, 8>, ka: i32, kb: i32) -> Simd<i32, 4> {
    let k = Simd::<i32, 8>::from_array([ka, kb, ka, kb, ka, kb, ka, kb]);
    let p = ab.cast::<i32>() * k;
    simd_swizzle!(p, [0, 2, 4, 6]) + simd_swizzle!(p, [1, 3, 5, 7])
}

/// [`pmadd8`] of 16 lanes (`vpmaddwd` on ymm).
#[inline(always)]
fn pmadd16(ab: Simd<i16, 16>, ka: i32, kb: i32) -> Simd<i32, 8> {
    let k = Simd::<i32, 16>::from_array(std::array::from_fn(|i| if i % 2 == 0 { ka } else { kb }));
    let p = ab.cast::<i32>() * k;
    simd_swizzle!(p, [0, 2, 4, 6, 8, 10, 12, 14]) + simd_swizzle!(p, [1, 3, 5, 7, 9, 11, 13, 15])
}

/// `Round2(sum(a[i] * k[i]), 12)` of `i16` lanes narrowed with saturation
/// (or modulo 2^16 when `WRAP` is true, matching dav1d's `ITX_MUL2X_PACK`
/// with `flags & 2 != 0`), computed with interleaves + `pmaddwd` + `psrad` +
/// `packssdw` (the sequence of dav1d's x86 `ITX_MULSUB_2W`), instead of
/// widening every input to `i32` and `pmulld`. Pairs of inputs share one
/// `pmaddwd`; an odd input is paired with a vector of ones times the rounding
/// constant. Every lane of `pmaddwd` is exact: `|a| <= 2^15` and `|k| <= 2^14`.
/// Returns `None` for unsupported lane counts.
#[inline(always)]
fn dot_rnd12_pmadd<const N: usize, const K: usize, const WRAP: bool>(
    a: [Simd<i16, N>; K],
    k: [i32; K],
) -> Option<Simd<i16, N>> {
    if !cfg!(target_arch = "x86_64") || !matches!(N, 4 | 8 | 16) {
        return None;
    }
    if K == 3 {
        let k1a = k[1] / 2;
        return dot_rnd12_pmadd::<N, 4, WRAP>(
            [a[0], a[1], a[1], a[2]],
            [k[0], k1a, k[1] - k1a, k[2]],
        );
    }
    let ones = Simd::<i16, N>::splat(1);
    // Pairs (a[2j], a[2j + 1]) with constants, plus the rounding term.
    let pairs = K.div_ceil(2);
    let pair = |j: usize| -> (Simd<i16, N>, i32, Simd<i16, N>, i32) {
        if 2 * j + 1 < K {
            (a[2 * j], k[2 * j], a[2 * j + 1], k[2 * j + 1])
        } else {
            (a[2 * j], k[2 * j], ones, 1 << 11)
        }
    };
    let rnd_in_pairs = K % 2 == 1;
    if N == 16 {
        let mut lo = Simd::<i32, 8>::splat(if rnd_in_pairs { 0 } else { 1 << 11 });
        let mut hi = lo;
        for j in 0..pairs {
            let (x, kx, y, ky) = pair(j);
            let (x, y) = (x.resize::<16>(0), y.resize::<16>(0));
            // In-lane interleaves (`vpunpck{l,h}wd` on ymm).
            let l = simd_swizzle!(
                x,
                y,
                [0, 16, 1, 17, 2, 18, 3, 19, 8, 24, 9, 25, 10, 26, 11, 27]
            );
            let h = simd_swizzle!(
                x,
                y,
                [4, 20, 5, 21, 6, 22, 7, 23, 12, 28, 13, 29, 14, 30, 15, 31]
            );
            lo += pmadd16(l, kx, ky);
            hi += pmadd16(h, kx, ky);
        }
        let (lo, hi) = if WRAP {
            ((lo >> 12).cast::<i16>(), (hi >> 12).cast::<i16>())
        } else {
            (sat16(lo >> 12), sat16(hi >> 12))
        };
        // In-lane `vpackssdw` order.
        let r = simd_swizzle!(
            lo,
            hi,
            [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15]
        );
        return Some(r.resize::<N>(0));
    }
    if N == 8 {
        let mut lo = Simd::<i32, 4>::splat(if rnd_in_pairs { 0 } else { 1 << 11 });
        let mut hi = lo;
        for j in 0..pairs {
            let (x, kx, y, ky) = pair(j);
            let (l, h) = x.resize::<8>(0).interleave(y.resize::<8>(0));
            lo += pmadd8(l, kx, ky);
            hi += pmadd8(h, kx, ky);
        }
        let (lo, hi) = if WRAP {
            ((lo >> 12).cast::<i16>(), (hi >> 12).cast::<i16>())
        } else {
            (sat16(lo >> 12), sat16(hi >> 12))
        };
        let r = simd_swizzle!(lo, hi, [0, 1, 2, 3, 4, 5, 6, 7]);
        return Some(r.resize::<N>(0));
    }
    let mut s = Simd::<i32, 4>::splat(if rnd_in_pairs { 0 } else { 1 << 11 });
    for j in 0..pairs {
        let (x, kx, y, ky) = pair(j);
        let xy = simd_swizzle!(x.resize::<4>(0), y.resize::<4>(0), [0, 4, 1, 5, 2, 6, 3, 7]);
        s += pmadd8(xy, kx, ky);
    }
    let r = if WRAP { (s >> 12).cast::<i16>() } else { sat16(s >> 12) };
    Some(r.resize::<N>(0))
}

/// Narrows `i32` lanes to `i16` with saturation (`packssdw`).
#[inline(always)]
pub(super) fn sat16<const N: usize>(x: Simd<i32, N>) -> Simd<i16, N> {
    x.simd_clamp(Simd::splat(i16::MIN as i32), Simd::splat(i16::MAX as i32))
        .cast()
}

/// `Round2(a * k, 12)` of `i16` lanes, saturated to 16 bits (wrapping on
/// aarch64 for `|k| <= 4096`, see `dot_rnd12_wrap`).
///
/// On x86_64, `k` is split as `k' + c * 4096` with `|k'| <= 2047` so
/// `k16 = 16 * k'` fits in `i16` and `Round2(a * k', 12) =
/// (a * k16 + 2^15) >> 16 = pmulhw(a, k16) + (pmullw(a, k16) >> 15)`
/// in 4 instructions (avoiding `dot_rnd12_pmadd` where LLVM constant-folds
/// `1 * 2048` and falls back to `pmuludq`).
#[inline(always)]
fn mul_rnd12<const N: usize, const FORCE_CORR: bool>(a: Simd<i16, N>, k: i32) -> Simd<i16, N> {
    if let Some(r) = dot_rnd12_wrap::<N, 1, FORCE_CORR>([a], [k]) {
        return r;
    }
    if cfg!(target_arch = "x86_64") {
        let c = (k + 2048) >> 12;
        let kp = k - c * 4096;
        if (-2047..=2047).contains(&kp) {
            let k16 = (kp * 16) as i16;
            let hi = ((a.cast::<i32>() * Simd::splat(k16 as i32)) >> 16).cast::<i16>();
            let lo = a * Simd::splat(k16);
            let mut r = hi - (lo >> 15);
            if c > 0 {
                for _ in 0..c {
                    r = r.saturating_add(a);
                }
            } else if c < 0 {
                for _ in 0..-c {
                    r = r.saturating_sub(a);
                }
            }
            return r;
        }
    }
    if let Some(r) = dot_rnd12_pmadd::<N, 1, false>([a], [k]) {
        return r;
    }
    narrow_rnd12(a.cast::<i32>() * Simd::splat(k))
}

/// `Round2(v, n)` of `i16` lanes without intermediate overflow.
#[inline(always)]
fn rnd16<const N: usize>(v: Simd<i16, N>, n: u32) -> Simd<i16, N> {
    #[cfg(target_arch = "aarch64")]
    if matches!(n, 1..=4) {
        // `Round2(v, n) = ((v >> (n - 1)) + 1) >> 1`, where the halving
        // add compiles to `shadd` (after `sshr` for `n > 1`) on aarch64.
        let t = (v >> Simd::splat(n as i16 - 1)).cast::<i32>();
        return ((t + Simd::splat(1)) >> 1).cast();
    }
    if n == 0 {
        v
    } else if n >= 2 {
        // `v >> (n - 1)` is in `[-16384, 16383]`, so `+ 1` cannot overflow.
        ((v >> Simd::splat(n as i16 - 1)) + Simd::splat(1)) >> Simd::splat(1)
    } else {
        (v >> Simd::splat(n as i16)) + ((v >> Simd::splat(n as i16 - 1)) & Simd::splat(1))
    }
}

impl<const N: usize> TxLanes for S16<N> {
    const SATURATING_16: bool = true;

    #[inline(always)]
    fn splat(v: i32) -> Self {
        S16(Simd::splat(v as i16))
    }
    /// Wrapping; not used by the kernels.
    #[inline(always)]
    fn mul(self, k: i32) -> Self {
        S16(self.0 * Simd::splat(k as i16))
    }
    #[inline(always)]
    fn rnd(self, n: u32) -> Self {
        S16(rnd16(self.0, n))
    }
    #[inline(always)]
    fn max(self, o: Self) -> Self {
        S16(self.0.simd_max(o.0))
    }
    #[inline(always)]
    fn min(self, o: Self) -> Self {
        S16(self.0.simd_min(o.0))
    }
    #[inline(always)]
    fn m1(self, ka: i32) -> Self {
        S16(mul_rnd12::<N, false>(self.0, ka))
    }
    #[inline(always)]
    fn m1_h0(self, ka: i32) -> Self {
        S16(mul_rnd12::<N, true>(self.0, ka))
    }
    #[inline(always)]
    fn m2(self, ka: i32, b: Self, kb: i32) -> Self {
        if let Some(r) = dot_rnd12_wrap::<N, 2, false>([self.0, b.0], [ka, kb]) {
            return S16(r);
        }
        if let Some(r) = dot_rnd12_pmadd::<N, 2, false>([self.0, b.0], [ka, kb]) {
            return S16(r);
        }
        S16(narrow_rnd12(
            self.0.cast::<i32>() * Simd::splat(ka) + b.0.cast::<i32>() * Simd::splat(kb),
        ))
    }
    #[inline(always)]
    fn m2_h0(self, ka: i32, b: Self, kb: i32) -> Self {
        if let Some(r) = dot_rnd12_wrap::<N, 2, true>([self.0, b.0], [ka, kb]) {
            return S16(r);
        }
        self.m2(ka, b, kb)
    }
    #[inline(always)]
    fn m2_wrap(self, ka: i32, b: Self, kb: i32) -> Self {
        if let Some(r) = dot_rnd12_wrap::<N, 2, false>([self.0, b.0], [ka, kb]) {
            return S16(r);
        }
        if let Some(r) = dot_rnd12_pmadd::<N, 2, true>([self.0, b.0], [ka, kb]) {
            return S16(r);
        }
        self.m2(ka, b, kb)
    }
    #[inline(always)]
    fn dot_rnd12<const K: usize>(a: [Self; K], k: [i32; K]) -> Self {
        if let Some(r) = dot_rnd12_pmadd::<N, K, false>(amap(a, |a| a.0), k) {
            return S16(r);
        }
        let mut s = a[0].0.cast::<i32>() * Simd::splat(k[0]);
        for i in 1..K {
            s += a[i].0.cast::<i32>() * Simd::splat(k[i]);
        }
        S16(narrow_rnd12(s))
    }
}

impl<const N: usize> S16<N> {
    /// `Round2(self, 4)` for residual addition to a `BD <= 10` pixel.
    /// For `v >= 32760`, `2047` and `2048` both saturate `d + r` to
    /// `(1 << BD) - 1 <= 1023`.
    #[inline(always)]
    fn rnd4_add(self) -> Simd<i16, N> {
        if cfg!(target_arch = "x86_64") {
            self.0.saturating_add(Simd::splat(8)) >> 4
        } else {
            self.rnd(4).0
        }
    }
}

/// Lane vectors of the row pass: `N` lanes loaded from / stored to `i32`.
trait RowLanes<const N: usize>: TxLanes {
    /// True for [`Wide`] lanes (split products), which [`row_pass`]
    /// replaces with plain `i32` lanes when [`rows_fit_i32`].
    const SPLIT: bool = false;
    fn from_i32(v: Simd<i32, N>) -> Self;
    fn to_i32(self) -> Simd<i32, N>;
    fn bitor(self, o: Self) -> Self;
    fn last_nz_lane(self) -> usize;
    #[inline(always)]
    fn load_row(src: &[i32; N]) -> Self {
        Self::from_i32(Simd::from_array(*src))
    }
    /// Narrows to `i16` (the value must fit).
    #[inline(always)]
    fn to_i16(self) -> Simd<i16, N> {
        self.to_i32().cast()
    }
    /// Converts an `N x N` tile of rows to columns (lane `l` of output
    /// `j` is `rows[l][j]`).
    #[inline(always)]
    fn load_tile(rows: [Simd<i32, N>; N]) -> [Self; N] {
        amap(transpose_e(rows), Self::from_i32)
    }
    /// Transposes an `N x N` tile of already-loaded rows to columns.
    #[inline(always)]
    fn transpose_tile(rows: [Self; N]) -> [Self; N] {
        amap(transpose_e(amap(rows, Self::to_i32)), Self::from_i32)
    }
}

impl<const N: usize> RowLanes<N> for Simd<i32, N> {
    #[inline(always)]
    fn from_i32(v: Simd<i32, N>) -> Self {
        v
    }
    #[inline(always)]
    fn to_i32(self) -> Simd<i32, N> {
        self
    }
    #[inline(always)]
    fn bitor(self, o: Self) -> Self {
        self | o
    }
    #[inline(always)]
    fn last_nz_lane(self) -> usize {
        last_nz_lane(self)
    }
}

impl<const N: usize> RowLanes<N> for Wide<N> {
    const SPLIT: bool = true;
    #[inline(always)]
    fn from_i32(v: Simd<i32, N>) -> Self {
        Wide(v)
    }
    #[inline(always)]
    fn to_i32(self) -> Simd<i32, N> {
        self.0
    }
    #[inline(always)]
    fn bitor(self, o: Self) -> Self {
        Wide(self.0 | o.0)
    }
    #[inline(always)]
    fn last_nz_lane(self) -> usize {
        last_nz_lane(self.0)
    }
}

impl<const N: usize> RowLanes<N> for S16<N> {
    /// Dequant has at most `bd + 8 = 16` bits for 8-bit streams.
    #[inline(always)]
    fn from_i32(v: Simd<i32, N>) -> Self {
        S16(v.cast())
    }
    #[inline(always)]
    fn to_i32(self) -> Simd<i32, N> {
        self.0.cast()
    }
    #[inline(always)]
    fn bitor(self, o: Self) -> Self {
        S16(self.0 | o.0)
    }
    #[inline(always)]
    fn last_nz_lane(self) -> usize {
        last_nz_lane16(self.0)
    }
    #[inline(always)]
    fn to_i16(self) -> Simd<i16, N> {
        self.0
    }
    #[inline(always)]
    fn load_tile(rows: [Simd<i32, N>; N]) -> [Self; N] {
        amap(transpose_e(amap(rows, |r| r.cast::<i16>())), S16)
    }
    #[inline(always)]
    fn transpose_tile(rows: [Self; N]) -> [Self; N] {
        amap(transpose_e(amap(rows, |r| r.0)), S16)
    }
}

/// Signed clamping range of `r` bits, splatted.
#[derive(Copy, Clone)]
pub(crate) struct Clamp<V> {
    lo: V,
    hi: V,
}

impl<V: TxLanes> Clamp<V> {
    #[inline(always)]
    fn bits(r: u32) -> Self {
        debug_assert!(!V::SATURATING_16 || r == 16);
        Clamp {
            lo: V::splat(-(1 << (r - 1))),
            hi: V::splat((1 << (r - 1)) - 1),
        }
    }

    #[inline(always)]
    pub(crate) fn apply(self, v: V) -> V {
        if V::SATURATING_16 {
            return v;
        }
        v.max(self.lo).min(self.hi)
    }
}

/// A 1D kernel transforming the first `n` entries in place.
type Kern<V> = fn(&mut [V], Clamp<V>);

/// Selects the 1D kernel for a transform kind and length `1 << log2n`,
/// given that only the first `nz` inputs (`1 <= nz <= 32`) can be
/// non-zero (see the portable `select_1d`).
#[inline(always)]
fn select<V: TxLanes>(kind: Tx1d, log2n: usize, nz: usize) -> Kern<V> {
    use kernels::*;
    // 0 for nz <= 1, 1 for 2, 2 for 3..=4, 3 for 5..=8, 4 for 9..=16, 5 for 17..=32.
    let tier = (usize::BITS - nz.saturating_sub(1).leading_zeros()) as usize;
    match (kind, log2n) {
        (Tx1d::Adst, 2) => [iadst4_nz1::<V>, iadst4::<V>][tier.min(1)],
        (Tx1d::Adst, 3) => [
            iadst8_nz1::<V>,
            iadst8_nz2::<V>,
            iadst8_nz4::<V>,
            iadst8_nz8::<V>,
        ][tier.min(3)],
        (Tx1d::Adst, 4) => [
            iadst16_nz1::<V>,
            iadst16_nz2::<V>,
            iadst16_nz4::<V>,
            iadst16_nz8::<V>,
            iadst16_nz16::<V>,
        ][tier.min(4)],
        (Tx1d::Identity, 2) => iidentity::<V, 4>,
        (Tx1d::Identity, 3) => iidentity::<V, 8>,
        (Tx1d::Identity, 4) => iidentity::<V, 16>,
        (Tx1d::Identity, 5) => iidentity::<V, 32>,
        (kind, n) => {
            debug_assert!(
                kind == Tx1d::Dct,
                "illegal {kind:?} transform of length {}",
                1 << n
            );
            match n {
                2 => [idct4_nz2::<V>, idct4_nz2::<V>, idct4_nz4::<V>][tier.min(2)],
                3 => [
                    idct8_nz2::<V>,
                    idct8_nz2::<V>,
                    idct8_nz4::<V>,
                    idct8_nz8::<V>,
                ][tier.min(3)],
                4 => [
                    idct16_nz2::<V>,
                    idct16_nz2::<V>,
                    idct16_nz4::<V>,
                    idct16_nz8::<V>,
                    idct16_nz16::<V>,
                ][tier.min(4)],
                5 => [
                    idct32_nz2::<V>,
                    idct32_nz2::<V>,
                    idct32_nz4::<V>,
                    idct32_nz8::<V>,
                    idct32_nz16::<V>,
                    idct32_nz32::<V>,
                ][tier.min(5)],
                _ => [
                    idct64_nz2::<V>,
                    idct64_nz2::<V>,
                    idct64_nz4::<V>,
                    idct64_nz8::<V>,
                    idct64_nz16::<V>,
                    idct64_nz32::<V>,
                ][tier.min(5)],
            }
        }
    }
}

/// Inverse ADST4 with inputs `1..` zero.
#[inline(always)]
fn iadst4_nz1<V: TxLanes>(t: &mut [V], _r: Clamp<V>) {
    let t: &mut [V; 4] = (&mut t[..4]).try_into().unwrap();
    let i0 = t[0];
    t[0] = i0.m1(1321);
    t[1] = i0.m1(2482);
    t[2] = i0.m1(3344);
    t[3] = i0.m1(3803);
}

/// Inverse ADST4 (7.13.2.6), with the sums of the spec expanded into one
/// linear combination of the inputs per output. With `i32` lanes and
/// inputs of at most 18 bits every sum fits in 32 bits (the absolute
/// coefficients add up to 10950 < 2^14).
fn iadst4<V: TxLanes>(t: &mut [V], _r: Clamp<V>) {
    let t: &mut [V; 4] = (&mut t[..4]).try_into().unwrap();
    let i = *t;
    t[0] = V::dot_rnd12(i, [1321, 3344, 3803, 2482]);
    t[1] = V::dot_rnd12(i, [2482, 3344, -1321, -3803]);
    t[2] = V::dot_rnd12([i[0], i[2], i[3]], [3344, -3344, 3344]);
    t[3] = V::dot_rnd12(i, [3803, -3344, 2482, -1321]);
}

/// Inverse identity transform of length `N` of one value (7.13.2.11 -
/// 7.13.2.14).
#[inline(always)]
fn identity<V: TxLanes, const N: usize>(v: V) -> V {
    match N {
        4 => v.m1(5793),
        8 => v + v,
        16 => v.m1(11586),
        _ => v.mul(4),
    }
}

/// Inverse identity transforms (7.13.2.11 - 7.13.2.14).
fn iidentity<V: TxLanes, const N: usize>(t: &mut [V], _r: Clamp<V>) {
    for v in &mut t[..N] {
        *v = identity::<V, N>(*v);
    }
}

/// Transposes an `N x N` matrix of `i32` (`N` a power of two) with
/// `log2(N)` rounds of interleaving.
#[inline(always)]
fn transpose<const N: usize>(mut m: [Simd<i32, N>; N]) -> [Simd<i32, N>; N] {
    let mut k = 1;
    while k < N {
        let mut o = m;
        for i in 0..N / 2 {
            let (lo, hi) = m[i].interleave(m[i + N / 2]);
            o[2 * i] = lo;
            o[2 * i + 1] = hi;
        }
        m = o;
        k *= 2;
    }
    m
}

/// Adds the saturated 16-bit residual `r16` to `d` with saturating
/// `i16` addition and clipping to `[0, 2^BD - 1]` (exact for arbitrary
/// `i32` residuals narrowed via [`sat16`], since pixels have `<= 12` bits).
#[inline(always)]
fn add_row_sat16<T: Pixel, const N: usize, const BD: u32>(d: &mut [T; N], r16: Simd<i16, N>) {
    debug_assert!(BD <= 12);
    let max = Simd::splat((1 << BD) - 1);
    let any: &mut dyn Any = d;
    if let Some(d) = any.downcast_mut::<[u8; N]>() {
        let v = Simd::from_array(*d).cast::<i16>().saturating_add(r16);
        *d = v.simd_clamp(Simd::splat(0), max).cast::<u8>().to_array();
    } else if let Some(d) = any.downcast_mut::<[u16; N]>() {
        let v = Simd::from_array(*d).cast::<i16>().saturating_add(r16);
        *d = v.simd_clamp(Simd::splat(0), max).cast::<u16>().to_array();
    } else {
        // Not reached (pixels are u8 or u16).
        let v = load_i32::<T, N>(d) + r16.cast::<i32>();
        store_clip(d, v, BD);
    }
}

/// Compile-time `Transform_Row_Shift` from `(W, H)`.
#[inline(always)]
pub(super) const fn row_shift(w: usize, h: usize) -> u32 {
    match (w, h) {
        (4, 4) | (4, 8) | (8, 4) => 0,
        (8, 32) | (32, 8) | (16, 16) | (16, 64) | (64, 16) | (32, 32) | (64, 64) => 2,
        _ => 1,
    }
}

/// Compile-time `|log2(W) - log2(H)| == 1`.
#[inline(always)]
pub(super) const fn rect2(w: usize, h: usize) -> bool {
    w * 2 == h || h * 2 == w
}

/// Per transform size and type entry point: `fn(dst, stride, coeffs,
/// eob_type)` with `eob_type = eob | (tx_type as usize) << 16`, `eob >= 1`,
/// non-lossless (see [`itx_table`]). Four integer arguments, so that the
/// call passes everything in registers.
pub type ItxFn<T> = fn(&mut [T], usize, &mut [i32], usize);

/// [`ItxFn`] of each `[tx_size as usize][tx_type as usize]` for
/// [`inv_txfm_add_bd`]'s contract (`eob >= 1`, not lossless; `bd == BD`),
/// so that callers can skip [`inv_txfm_add_bd`]'s dispatch.
pub fn itx_table<T: Pixel, const BD: u32>() -> &'static [[ItxFn<T>; 16]; 19] {
    // One monomorphized entry per size (and per transform type for the
    // sizes of at most 8x8), so that the size and type dependent
    // parameters are compile-time constants: one table lookup + indirect
    // call, like dav1d's per (size, type) functions.
    const {
        &[
            entries!(T, BD, 4, 4),
            entries!(T, BD, 8, 8),
            entries!(T, BD, 16, 16),
            entries!(T, BD, 32, 32),
            entries!(T, BD, 64, 64),
            entries!(T, BD, 4, 8),
            entries!(T, BD, 8, 4),
            entries!(T, BD, 8, 16),
            entries!(T, BD, 16, 8),
            entries!(T, BD, 16, 32),
            entries!(T, BD, 32, 16),
            entries!(T, BD, 32, 64),
            entries!(T, BD, 64, 32),
            entries!(T, BD, 4, 16),
            entries!(T, BD, 16, 4),
            entries!(T, BD, 8, 32),
            entries!(T, BD, 32, 8),
            entries!(T, BD, 16, 64),
            entries!(T, BD, 64, 16),
        ]
    }
}

/// See [`crate::decoder::av1::dsp::itx::inv_txfm_add`].
#[allow(clippy::too_many_arguments)]
pub fn inv_txfm_add_bd<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    coeffs: &mut [i32],
    tx_size: TxSize,
    tx_type: TxType,
    lossless: bool,
    eob: usize,
    bd: u32,
) {
    // The decoder uses `u8` pixels iff the bit depth is 8. The tests also
    // run `u16` pixels at bit depth 8, which every kernel supports.
    debug_assert!(
        !(std::mem::size_of::<T>() == 1 && BD > 8)
            && (cfg!(test) || !(std::mem::size_of::<T>() == 2 && BD == 8))
    );
    debug_assert_eq!(bd, BD);
    if eob == 0 {
        return;
    }
    if lossless {
        debug_assert_eq!(tx_size, TxSize::Tx4x4);
        inv_wht4x4_add_bd::<T, BD>(dst, stride, coeffs, tx_type);
        return;
    }
    itx_table::<T, BD>()[tx_size as usize][tx_type as usize](
        dst,
        stride,
        coeffs,
        eob | (tx_type as usize) << 16,
    )
}

/// The 16 [`ItxFn`]s of a `W x H` transform size, indexed by `TxType`.
/// For the sizes of at most 8x8 with 16-bit intermediates, the four
/// DCT / ADST combinations (about 98% of all such blocks; the identity
/// and flipped types are rare) get per type monomorphized entries
/// ([`typed_entry`]), but only for the pixel type of the bit depth
/// (`u8` for 8 bits, `u16` else) to bound the code size. Everything else
/// uses [`size_entry`].
macro_rules! entries {
    ($T:ty, $BD:expr, $W:expr, $H:expr) => {{
        let mut e = [size_entry::<$T, $BD, $W, $H> as ItxFn<$T>; 16];
        if $W <= 8 && $H <= 8 && $BD <= 10 && (std::mem::size_of::<$T>() == 1) == ($BD == 8) {
            e[0] = typed_entry::<$T, $BD, $W, $H, 0>;
            e[1] = typed_entry::<$T, $BD, $W, $H, 1>;
            e[2] = typed_entry::<$T, $BD, $W, $H, 2>;
            e[3] = typed_entry::<$T, $BD, $W, $H, 3>;
        }
        e
    }};
}
use entries;

/// [`size_entry`] for the transform type `TxType::ALL[TT]` (the type
/// bits of `eob_type` are ignored).
fn typed_entry<T: Pixel, const BD: u32, const W: usize, const H: usize, const TT: u8>(
    dst: &mut [T],
    stride: usize,
    coeffs: &mut [i32],
    eob_type: usize,
) {
    size_entry_inl::<T, BD, W, H>(
        dst,
        stride,
        coeffs,
        TxType::ALL[TT as usize],
        eob_type & 0xffff,
    )
}

/// [`inv_txfm_add_bd`] for one `W x H` transform size (`eob >= 1`, not
/// lossless; see [`ItxFn`] for `eob_type`).
fn size_entry<T: Pixel, const BD: u32, const W: usize, const H: usize>(
    dst: &mut [T],
    stride: usize,
    coeffs: &mut [i32],
    eob_type: usize,
) {
    let tx_type = TxType::ALL[(eob_type >> 16) & 15];
    size_entry_inl::<T, BD, W, H>(dst, stride, coeffs, tx_type, eob_type & 0xffff)
}

/// Body of [`size_entry`], inlined into [`typed_entry`].
#[inline(always)]
fn size_entry_inl<T: Pixel, const BD: u32, const W: usize, const H: usize>(
    dst: &mut [T],
    stride: usize,
    coeffs: &mut [i32],
    tx_type: TxType,
    eob: usize,
) {
    let (log2w, log2h) = const { (W.ilog2() as usize, H.ilog2() as usize) };
    let (tw, th) = const { (if W < 32 { W } else { 32 }, if H < 32 { H } else { 32 }) };
    let row_shift = const { row_shift(W, H) };
    let rect2 = const { rect2(W, H) };
    let col_range = const {
        if BD + 6 > 16 {
            BD + 6
        } else {
            16
        }
    };
    let stride = stride & 0xffff_ffff;
    let coeffs = &mut coeffs[..tw * th];
    assert!(stride >= W && dst.len() >= (H - 1) * stride + W);

    if eob == 1
        && matches!(
            tx_type,
            TxType::DCT_DCT | TxType::IDTX | TxType::V_DCT | TxType::H_DCT
        )
    {
        add_dc_only::<T, BD, W, H>(dst, stride, coeffs, tx_type);
        return;
    }

    let (col_kind, row_kind) = tx1d_kinds(tx_type);
    let (eob_rows, eob_cols) =
        if eob == 1 { (1, 1) } else { eob_bounds(eob, tw, th, row_kind, col_kind) };
    let p = Params {
        w: W,
        h: H,
        tw,
        th,
        log2w,
        log2h,
        rect2,
        row_shift,
        row_kind,
        col_kind,
        col_range,
        flip_ud: H <= 16 && flip_ud(tx_type),
        flip_lr: W <= 16 && flip_lr(tx_type),
        bd: BD,
        eob_rows,
        eob_cols,
    };
    if BD <= 10 {
        // 16-bit intermediates (see `S16`).
        transform_add16_dispatch::<T, BD>(dst, stride, coeffs, &p);
        return;
    }
    if W <= 32 {
        // tw == w: row transforms in place in the coefficient buffer.
        transform_add_bd::<T, BD>(dst, stride, coeffs, &p);
    } else {
        // Only the first eob_rows <= th <= 32 rows can be non-zero.
        let mut buf = [0i32; 64 * 32];
        buf[..eob_rows * tw].copy_from_slice(&coeffs[..eob_rows * tw]);
        transform_add_bd::<T, BD>(dst, stride, &mut buf[..W * th], &p);
    }
    // Only the first eob_rows rows (stride tw == w, or untouched for
    // W == 64) can be non-zero: the input coefficients and the row
    // outputs of the in-place row pass.
    coeffs[..eob_rows * tw].fill(0);
}

/// Transforms with only `Dequant[0][0]` non-zero (`eob == 1`) of the
/// types whose output is then a constant (`DCT_DCT`), a constant row
/// (`H_DCT`), a constant column (`V_DCT`) or a single sample (`IDTX`).
/// Leaves `coeffs[0]` zero.
#[inline(always)]
fn add_dc_only<T: Pixel, const BD: u32, const W: usize, const H: usize>(
    dst: &mut [T],
    stride: usize,
    coeffs: &mut [i32],
    tx_type: TxType,
) {
    let (log2w, log2h) = const { (W.ilog2() as usize, H.ilog2() as usize) };
    let row_shift = const { row_shift(W, H) };
    let col_range = const {
        if BD + 6 > 16 {
            BD + 6
        } else {
            16
        }
    };
    let row_clamp = ScalarClamp::bits(BD + 8);
    let col_clamp = ScalarClamp::bits(col_range);
    let mut dc = coeffs[0];
    coeffs[0] = 0;
    if const { rect2(W, H) } {
        dc = round12(dc as i64 * 2896);
    }
    dc = if matches!(tx_type, TxType::DCT_DCT | TxType::H_DCT) {
        row_clamp.apply(round12(dc as i64 * 2896))
    } else {
        match log2w {
            2 => round12(dc as i64 * 5793),
            3 => dc * 2,
            4 => round12(dc as i64 * 11586),
            _ => dc * 4,
        }
    };
    dc = col_clamp.apply(round2(dc, row_shift));
    dc = if matches!(tx_type, TxType::DCT_DCT | TxType::V_DCT) {
        col_clamp.apply(round12(dc as i64 * 2896))
    } else {
        match log2h {
            2 => round12(dc as i64 * 5793),
            3 => dc * 2,
            4 => round12(dc as i64 * 11586),
            _ => dc * 4,
        }
    };
    let dc = round2(dc, 4);
    if dc == 0 {
        return;
    }
    let max = (1 << BD) - 1;
    match tx_type {
        TxType::IDTX => {
            dst[0] = T::from_i32((dst[0].to_i32() + dc).clamp(0, max));
        }
        TxType::V_DCT => {
            visit_rows::<T, 1, H>(dst, stride, |_, [d]| {
                *d = T::from_i32((d.to_i32() + dc).clamp(0, max));
            });
        }
        TxType::H_DCT => {
            if BD <= 10 {
                match W {
                    4 => add_dc16::<T, 4, W, 1, BD>(dst, stride, dc),
                    8 => add_dc16::<T, 8, W, 1, BD>(dst, stride, dc),
                    _ => add_dc16::<T, 16, W, 1, BD>(dst, stride, dc),
                }
            } else {
                match W {
                    4 => add_dc_bd::<T, 4, W, 1, BD>(dst, stride, dc),
                    _ => add_dc_bd::<T, 8, W, 1, BD>(dst, stride, dc),
                }
            }
        }
        _ => {
            if BD <= 10 {
                match W {
                    4 => add_dc16::<T, 4, W, H, BD>(dst, stride, dc),
                    8 => add_dc16::<T, 8, W, H, BD>(dst, stride, dc),
                    _ => add_dc16::<T, 16, W, H, BD>(dst, stride, dc),
                }
            } else {
                match W {
                    4 => add_dc_bd::<T, 4, W, H, BD>(dst, stride, dc),
                    _ => add_dc_bd::<T, 8, W, H, BD>(dst, stride, dc),
                }
            }
        }
    }
}

/// Parameters of a (non-lossless) 2D inverse transform.
struct Params {
    w: usize,
    h: usize,
    tw: usize,
    th: usize,
    log2w: usize,
    log2h: usize,
    rect2: bool,
    row_shift: u32,
    row_kind: Tx1d,
    col_kind: Tx1d,
    col_range: u32,
    flip_ud: bool,
    flip_lr: bool,
    bd: u32,
    eob_rows: usize,
    eob_cols: usize,
}

/// 2D transform core, same buffer contract as the portable
/// `transform_add`: on entry `buf[i * tw + j]` holds `Dequant[i][j]`;
/// `buf.len() == w * th`.
fn transform_add_bd<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    buf: &mut [i32],
    p: &Params,
) {
    // 32-bit lane count for widths >= 8: 8 with AVX2 (one 256-bit ymm),
    // 4 otherwise (one 128-bit NEON register, avoiding register spills in
    // 16/32/64-point kernels).
    const LW: usize = if cfg!(target_feature = "avx2") { 8 } else { 4 };
    // Rows: split products when rowClampRange exceeds 18 bits (12 bit).
    let nz_rows = if BD > 10 {
        row_pass_w::<Wide<4>, Wide<LW>, LW>(buf, p)
    } else {
        row_pass_w::<Simd<i32, 4>, Simd<i32, LW>, LW>(buf, p)
    };
    if nz_rows == 0 {
        return;
    }
    match (p.w, p.h) {
        (4, 4) => col_pass_bd::<T, 4, 4, BD>(dst, stride, buf, p, nz_rows),
        (4, 8) => col_pass_bd::<T, 4, 8, BD>(dst, stride, buf, p, nz_rows),
        (4, _) => col_pass_bd::<T, 4, 16, BD>(dst, stride, buf, p, nz_rows),
        (_, 4) => col_pass_bd::<T, LW, 4, BD>(dst, stride, buf, p, nz_rows),
        (_, 8) => col_pass_bd::<T, LW, 8, BD>(dst, stride, buf, p, nz_rows),
        (_, 16) => col_pass_bd::<T, LW, 16, BD>(dst, stride, buf, p, nz_rows),
        (_, 32) => col_pass_bd::<T, LW, 32, BD>(dst, stride, buf, p, nz_rows),
        _ => col_pass_bd::<T, LW, 64, BD>(dst, stride, buf, p, nz_rows),
    }
}

/// Row pass dispatch on the transform width, with `V4` / `VL` the 4 / `LW`
/// lane row vector types.
fn row_pass_w<V4: RowLanes<4>, VL: RowLanes<LW>, const LW: usize>(
    buf: &mut [i32],
    p: &Params,
) -> usize {
    if p.row_kind == Tx1d::Identity {
        return if p.w == 4 {
            row_pass_identity::<V4, 4>(buf, p)
        } else {
            row_pass_identity::<VL, LW>(buf, p)
        };
    }
    match (p.w, p.th) {
        (4, _) => row_pass::<V4, 4, 4>(buf, p),
        (8, 4) => row_pass::<V4, 4, 8>(buf, p),
        (8, _) => row_pass::<VL, LW, 8>(buf, p),
        (16, 4) => row_pass::<V4, 4, 16>(buf, p),
        (16, _) => row_pass::<VL, LW, 16>(buf, p),
        (32, _) => row_pass::<VL, LW, 32>(buf, p),
        _ => row_pass::<VL, LW, 64>(buf, p),
    }
}

/// Row transforms of width `W`, `N` rows at a time (`N <= min(tw, th)`),
/// in place: row `i` of the output is stored at `buf[i * w..]`. Returns
/// one more than the index of the last non-zero input row (0 if all are
/// zero); output rows from there on are not written.
fn row_pass<V: RowLanes<N>, const N: usize, const W: usize>(buf: &mut [i32], p: &Params) -> usize {
    debug_assert_eq!(W, p.w);
    let (w, tw) = (W, p.tw);
    let zero = Simd::<i32, N>::splat(0);
    let mut t = [V::splat(0); W];
    let mut nz_rows = 0;
    let col_tiles = p.eob_cols.div_ceil(N);
    // Groups are processed bottom-up so that writing rows (stride w)
    // never clobbers unread input rows (stride tw <= w).
    for g in (0..p.eob_rows.div_ceil(N)).rev() {
        let i0 = g * N;
        // t[j] lane l = Dequant[i0 + l][j]. `any` lane l is non-zero if
        // row i0 + l is; `nz` is one more than the last non-zero column.
        let mut any = zero;
        let mut nz = 0;
        for j0 in (0..col_tiles).map(|k| k * N) {
            let rows: [Simd<i32, N>; N] =
                std::array::from_fn(|l| Simd::from_slice(&buf[(i0 + l) * tw + j0..]));
            let cols_nz = rows.iter().fold(zero, |a, &v| a | v);
            if cols_nz == zero {
                t[j0..j0 + N].fill(V::splat(0));
                continue;
            }
            nz = j0 + last_nz_lane(cols_nz);
            for (d, &s) in t[j0..j0 + N].iter_mut().zip(&transpose(rows)) {
                any |= s;
                *d = V::from_i32(s);
            }
        }
        if nz == 0 {
            // Rows at or beyond nz_rows are never read.
            if tw != w && nz_rows != 0 {
                buf[i0 * w..(i0 + N) * w].fill(0);
            }
            continue;
        }
        if nz_rows == 0 {
            // The highest lane of `any` that is non-zero.
            nz_rows = i0 + last_nz_lane(any);
        }
        if V::SPLIT && rows_fit_i32(&t[..nz]) {
            // 12 bits: plain `i32` lanes are exact for these rows (see
            // `rows_fit_i32`), and much cheaper than the split products.
            let mut t32 = [zero; W];
            for (d, s) in t32[..nz].iter_mut().zip(&t[..nz]) {
                *d = s.to_i32();
            }
            row_group_tail::<Simd<i32, N>, N, W>(&mut t32, nz, i0, buf, p);
        } else {
            // Earlier groups transformed in place may have written `t[col_tiles * N..]`.
            let k_in = nz.next_power_of_two().max(4).min(W);
            if col_tiles * N < k_in {
                t[col_tiles * N..k_in].fill(V::splat(0));
            }
            row_group_tail::<V, N, W>(&mut t, nz, i0, buf, p);
        }
    }
    nz_rows
}

/// Largest per-row sum of absolute input coefficients for which a 12-bit
/// row transform on plain `i32` lanes is exact: every intermediate value
/// then has at most `S16_ROW_L1_MAX`'s bound of `sum(|x_k|) + 63 < 2^17`,
/// i.e. 18 bits, for which the generator checked that no `i32` product
/// sum overflows, and the 20-bit row clamp never applies (so the results
/// equal those of [`Wide`] lanes).
const I32_ROW_L1_MAX: u32 = (1 << 17) - 128;

/// Whether every lane (row) of `t` has an absolute sum of at most
/// [`I32_ROW_L1_MAX`] (each term is capped at `2^20` first, so the sum
/// cannot overflow and `i32::MIN` is rejected).
#[inline(always)]
fn rows_fit_i32<V: RowLanes<N>, const N: usize>(t: &[V]) -> bool {
    let cap = Simd::<u32, N>::splat(1 << 20);
    let mut l1 = Simd::<u32, N>::splat(0);
    for v in t {
        l1 += v.to_i32().abs().cast::<u32>().simd_min(cap);
    }
    l1.reduce_max() <= I32_ROW_L1_MAX
}

/// Rest of [`row_pass`] for one group of `N` rows held in `t` (`t[j]`
/// lane `l` = input `j` of row `i0 + l`, `nz >= 1` one more than the last
/// non-zero input): the 1D transforms and the transposed store of the
/// outputs to `buf` (stride `W`).
#[inline(always)]
fn row_group_tail<V: RowLanes<N>, const N: usize, const W: usize>(
    t: &mut [V; W],
    nz: usize,
    i0: usize,
    buf: &mut [i32],
    p: &Params,
) {
    let w = W;
    let row_clamp = Clamp::<V>::bits(p.bd + 8);
    let col_clamp = Clamp::<V>::bits(p.col_range);
    if nz == 1 && p.row_kind == Tx1d::Dct {
        // DC only: every DCT output is clamp(Round2(in * cos128(32),
        // 12)) (see the portable `dct_dc`).
        let mut v = t[0];
        if p.rect2 {
            v = v.m1(2896);
        }
        let v = col_clamp
            .apply(row_clamp.apply(v.m1(2896)).rnd(p.row_shift))
            .to_i32();
        for (l, &x) in v.as_array().iter().enumerate() {
            buf[(i0 + l) * w..(i0 + l + 1) * w].fill(x);
        }
        return;
    }
    if p.rect2 {
        for v in &mut t[..nz] {
            *v = v.m1(2896);
        }
    }
    select::<V>(p.row_kind, p.log2w, nz)(t, row_clamp);
    match p.row_shift {
        0 => store_row_tiles::<V, N, W, 0>(t, i0, buf, col_clamp),
        1 => store_row_tiles::<V, N, W, 1>(t, i0, buf, col_clamp),
        _ => store_row_tiles::<V, N, W, 2>(t, i0, buf, col_clamp),
    }
}

/// Rounds (`RSHIFT`), clamps, transposes and stores the `W` row outputs
/// of `t` for rows `i0..i0 + N` into `buf` (stride `W`).
#[inline(always)]
fn store_row_tiles<V: RowLanes<N>, const N: usize, const W: usize, const RSHIFT: u32>(
    t: &[V; W],
    i0: usize,
    buf: &mut [i32],
    col_clamp: Clamp<V>,
) {
    for j0 in (0..W / N).map(|k| k * N) {
        let ts: &[V; N] = (&t[j0..j0 + N]).try_into().unwrap();
        let mut o = [Simd::<i32, N>::splat(0); N];
        for k in 0..N {
            let v = if RSHIFT == 0 { ts[k] } else { ts[k].rnd(RSHIFT) };
            o[k] = col_clamp.apply(v).to_i32();
        }
        for (l, r) in transpose(o).iter().enumerate() {
            r.copy_to_slice(&mut buf[(i0 + l) * W + j0..]);
        }
    }
}

/// Identity row transforms (element-wise, `N` columns at a time; `w`
/// is at most 32 so `tw == w`). Same contract as [`row_pass`].
fn row_pass_identity<V: RowLanes<N>, const N: usize>(buf: &mut [i32], p: &Params) -> usize {
    let w = p.w;
    let col_clamp = Clamp::<V>::bits(p.col_range);
    let zero = Simd::<i32, N>::splat(0);
    let mut nz_rows = 0;
    for (i, row) in buf.chunks_exact_mut(w).take(p.eob_rows).enumerate() {
        let mut any = zero;
        for chunk in row.chunks_exact_mut(N) {
            let s = Simd::from_slice(chunk);
            any |= s;
            let mut v = V::from_i32(s);
            if p.rect2 {
                v = v.m1(2896);
            }
            v = match p.log2w {
                2 => identity::<V, 4>(v),
                3 => identity::<V, 8>(v),
                4 => identity::<V, 16>(v),
                _ => identity::<V, 32>(v),
            };
            let v = match p.row_shift {
                0 => v,
                1 => v.rnd(1),
                _ => v.rnd(2),
            };
            col_clamp.apply(v).to_i32().copy_to_slice(chunk);
        }
        if any != zero {
            nz_rows = i + 1;
        }
    }
    nz_rows
}

/// Column transforms of height `H`, `N` columns at a time, with the
/// final Round2(T[i], colShift = 4), added to the frame with the flips
/// of 7.12.3 step 3. Rows at or beyond `nz_rows` of `buf` are zero.
fn col_pass_bd<T: Pixel, const N: usize, const H: usize, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    buf: &[i32],
    p: &Params,
    nz_rows: usize,
) {
    debug_assert_eq!(H, p.h);
    let (w, h) = (p.w, H);
    let clamp = Clamp::<Simd<i32, N>>::bits(p.col_range);
    if nz_rows == 1 && p.col_kind == Tx1d::Dct {
        // Every column is constant.
        if N == 4 && w >= 8 {
            let clamp4 = Clamp::<Simd<i32, 4>>::bits(p.col_range);
            for j0 in (0..w / 8).map(|k| k * 8) {
                let r0 = sat16(
                    clamp4
                        .apply(Simd::<i32, 4>::from_slice(&buf[j0..]).m1(2896))
                        .rnd(4),
                );
                let r1 = sat16(
                    clamp4
                        .apply(Simd::<i32, 4>::from_slice(&buf[j0 + 4..]).m1(2896))
                        .rnd(4),
                );
                let mut r = simd_swizzle!(r0, r1, [0, 1, 2, 3, 4, 5, 6, 7]);
                let x0 = if p.flip_lr {
                    r = r.reverse();
                    w - j0 - 8
                } else {
                    j0
                };
                visit_rows::<T, 8, H>(&mut dst[x0..], stride, |_, d| {
                    add_row_sat16::<T, 8, BD>(d, r);
                });
            }
            return;
        }
        for j0 in (0..w / N).map(|k| k * N) {
            let mut r = sat16(clamp.apply(Simd::from_slice(&buf[j0..]).m1(2896)).rnd(4));
            let x0 = if p.flip_lr {
                r = r.reverse();
                w - j0 - N
            } else {
                j0
            };
            visit_rows::<T, N, H>(&mut dst[x0..], stride, |_, d| {
                add_row_sat16::<T, N, BD>(d, r);
            });
        }
        return;
    }
    let k = if p.col_kind == Tx1d::Identity {
        h
    } else {
        nz_rows.next_power_of_two().max(4).min(h)
    };
    if N == 4 && w >= 8 {
        let clamp4 = Clamp::<Simd<i32, 4>>::bits(p.col_range);
        let kernel4 = select::<Simd<i32, 4>>(p.col_kind, p.log2h, nz_rows);
        let mut c0 = [Simd::<i32, 4>::splat(0); H];
        let mut c1 = [Simd::<i32, 4>::splat(0); H];
        for j0 in (0..w / 8).map(|k| k * 8) {
            for i in 0..nz_rows {
                let row = &buf[i * w + j0..][..8];
                c0[i] = Simd::from_slice(&row[..4]);
                c1[i] = Simd::from_slice(&row[4..]);
            }
            c0[nz_rows..k].fill(Simd::splat(0));
            c1[nz_rows..k].fill(Simd::splat(0));
            kernel4(&mut c0, clamp4);
            kernel4(&mut c1, clamp4);
            let x0 = if p.flip_lr { w - j0 - 8 } else { j0 };
            if !p.flip_lr && !p.flip_ud {
                visit_rows::<T, 8, H>(&mut dst[x0..], stride, |y, d| {
                    let r = simd_swizzle!(
                        sat16(c0[y].rnd(4)),
                        sat16(c1[y].rnd(4)),
                        [0, 1, 2, 3, 4, 5, 6, 7]
                    );
                    add_row_sat16::<T, 8, BD>(d, r);
                });
            } else {
                visit_rows::<T, 8, H>(&mut dst[x0..], stride, |y, d| {
                    let yi = if p.flip_ud { h - 1 - y } else { y };
                    let r = simd_swizzle!(
                        sat16(c0[yi].rnd(4)),
                        sat16(c1[yi].rnd(4)),
                        [0, 1, 2, 3, 4, 5, 6, 7]
                    );
                    add_row_sat16::<T, 8, BD>(d, if p.flip_lr { r.reverse() } else { r });
                });
            }
        }
        return;
    }
    let kernel = select::<Simd<i32, N>>(p.col_kind, p.log2h, nz_rows);
    let mut c = [Simd::splat(0); H];
    for j0 in (0..w / N).map(|k| k * N) {
        for (i, v) in c[..nz_rows].iter_mut().enumerate() {
            *v = Simd::from_slice(&buf[i * w + j0..]);
        }
        // The kernel overwrites all h rows.
        c[nz_rows..k].fill(Simd::splat(0));
        kernel(&mut c, clamp);
        let x0 = if p.flip_lr { w - j0 - N } else { j0 };
        if !p.flip_lr && !p.flip_ud {
            visit_rows::<T, N, H>(&mut dst[x0..], stride, |y, d| {
                add_row_sat16::<T, N, BD>(d, sat16(c[y].rnd(4)));
            });
        } else {
            visit_rows::<T, N, H>(&mut dst[x0..], stride, |y, d| {
                let yi = if p.flip_ud { h - 1 - y } else { y };
                let r = sat16(c[yi].rnd(4));
                add_row_sat16::<T, N, BD>(d, if p.flip_lr { r.reverse() } else { r });
            });
        }
    }
}

// ---------------------------------------------------------------------------
// 16-bit intermediate path (bd <= 10)
// ---------------------------------------------------------------------------

/// Transposes an `N x N` matrix (`N` a power of two) with `log2(N)`
/// rounds of interleaving.
#[inline(always)]
fn transpose_e<E: SimdElement, const N: usize>(mut m: [Simd<E, N>; N]) -> [Simd<E, N>; N] {
    let mut k = 1;
    while k < N {
        let mut o = m;
        for i in 0..N / 2 {
            let (lo, hi) = m[i].interleave(m[i + N / 2]);
            o[2 * i] = lo;
            o[2 * i + 1] = hi;
        }
        m = o;
        k *= 2;
    }
    m
}

/// One more than the index of the last non-zero lane of `v` (0 if all
/// are zero). Avoids `to_bitmask`, which is slow on aarch64.
#[inline(always)]
fn last_nz_lane<const N: usize>(v: Simd<i32, N>) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        let m = v.simd_ne(Simd::splat(0)).to_bitmask();
        return (64 - m.leading_zeros()) as usize;
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let idx = Simd::from_array(std::array::from_fn(|i| i as i32 + 1));
        v.simd_ne(Simd::splat(0))
            .select(idx, Simd::splat(0))
            .reduce_max() as usize
    }
}

/// One more than the index of the last non-zero lane of `v` (0 if all
/// are zero).
#[inline(always)]
pub(super) fn last_nz_lane16<const N: usize>(v: Simd<i16, N>) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        let m = v.simd_ne(Simd::splat(0)).to_bitmask();
        return (64 - m.leading_zeros()) as usize;
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let idx = Simd::from_array(std::array::from_fn(|i| i as i16 + 1));
        v.simd_ne(Simd::splat(0))
            .select(idx, Simd::splat(0))
            .reduce_max() as usize
    }
}

/// Stages 2..3 of the 8-point inverse DCT.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn idct8_tail<V: TxLanes>(
    v0: V,
    v1: V,
    v2: V,
    v3: V,
    v4: V,
    v5: V,
    v6: V,
    v7: V,
    r: Clamp<V>,
) -> [V; 8] {
    let v8 = r.apply(v0 + v2);
    let v9 = r.apply(v0 - v2);
    let v10 = r.apply(v1 + v3);
    let v11 = r.apply(v1 - v3);
    let v12 = r.apply(v5 + v7);
    let v13 = r.apply(v5 - v7);
    let v14 = r.apply(v4 + v6);
    let v15 = r.apply(v4 - v6);
    let v16 = v11.m2(2896, v9, -2896);
    let v17 = v11.m2(2896, v9, 2896);
    [
        r.apply(v12 + v10),
        r.apply(v14 + v17),
        r.apply(v15 + v16),
        r.apply(v13 + v8),
        r.apply(v13 - v8),
        r.apply(v15 - v16),
        r.apply(v14 - v17),
        r.apply(v12 - v10),
    ]
}

/// Stages 2..4 of the 8-point inverse ADST.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn iadst8_tail<V: TxLanes>(
    v0: V,
    v1: V,
    v2: V,
    v3: V,
    v4: V,
    v5: V,
    v6: V,
    v7: V,
    r: Clamp<V>,
) -> [V; 8] {
    let v8 = r.apply(v1 + v5);
    let v9 = r.apply(v1 - v5);
    let v10 = r.apply(v0 + v4);
    let v11 = r.apply(v0 - v4);
    let v12 = r.apply(v3 + v7);
    let v13 = r.apply(v3 - v7);
    let v14 = r.apply(v2 + v6);
    let v15 = r.apply(v2 - v6);
    let v16 = v9.m2_wrap(1567, v11, -3784);
    let v17 = v9.m2_wrap(3784, v11, 1567);
    let v18 = v15.m2_wrap(3784, v13, -1567);
    let v19 = v15.m2_wrap(1567, v13, 3784);
    let v20 = r.apply(v8 + v12);
    let v21 = r.apply(v8 - v12);
    let v22 = r.apply(v17 + v18);
    let v23 = r.apply(v17 - v18);
    let v24 = r.apply(v10 + v14);
    let v25 = r.apply(v10 - v14);
    let v26 = r.apply(v16 + v19);
    let v27 = r.apply(v16 - v19);
    let v28 = v21.m2(2896, v25, -2896);
    let v29 = v21.m2(2896, v25, 2896);
    let v30 = v23.m2(2896, v27, -2896);
    let v31 = v23.m2(2896, v27, 2896);
    [v20, -v22, v31, -v29, v28, -v30, v26, -v24]
}

/// Runs the DCT or ADST of length `N` (4 or 8) on `t` (`L` lanes),
/// given whether all inputs beyond the first 4 are zero (`le4`). Inlined
/// on register arrays so small transforms keep all lanes in registers.
// Not inlined with debug assertions (unoptimized and sanitizer builds),
// where every inlined copy keeps its own stack slots and the frames of
// the callers would exceed the default 2 MiB thread stack.
#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
fn dct_adst_small<V: TxLanes, const N: usize>(kind: Tx1d, le4: bool, t: &mut [V; N], c: Clamp<V>) {
    if N == 4 {
        let [i0, i1, i2, i3] = [t[0], t[1], t[2], t[3]];
        let o = if kind == Tx1d::Adst {
            let i = [i0, i1, i2, i3];
            [
                V::dot_rnd12(i, [1321, 3344, 3803, 2482]),
                V::dot_rnd12(i, [2482, 3344, -1321, -3803]),
                V::dot_rnd12([i0, i2, i3], [3344, -3344, 3344]),
                V::dot_rnd12(i, [3803, -3344, 2482, -1321]),
            ]
        } else {
            let v0 = i0.m2(2896, i2, -2896);
            let v1 = i0.m2(2896, i2, 2896);
            let v2 = i1.m2(1567, i3, -3784);
            let v3 = i1.m2(3784, i3, 1567);
            [
                c.apply(v1 + v3),
                c.apply(v0 + v2),
                c.apply(v0 - v2),
                c.apply(v1 - v3),
            ]
        };
        t.copy_from_slice(&o);
        return;
    }
    let [i0, i1, i2, i3] = [t[0], t[1], t[2], t[3]];
    let o = match (kind == Tx1d::Adst, le4) {
        (false, true) => {
            let v4 = i0.m1(2896);
            idct8_tail(
                i1.m1(799),
                i1.m1(4017),
                i3.m1(-2276),
                i3.m1(3406),
                v4,
                v4,
                i2.m1(1567),
                i2.m1(3784),
                c,
            )
        }
        (false, false) => {
            let [i4, i5, i6, i7] = [t[4], t[5], t[6], t[7]];
            idct8_tail(
                i1.m2_wrap(799, i7, -4017),
                i1.m2_wrap(4017, i7, 799),
                i5.m2_wrap(3406, i3, -2276),
                i5.m2_wrap(2276, i3, 3406),
                i0.m2(2896, i4, -2896),
                i0.m2(2896, i4, 2896),
                i2.m2(1567, i6, -3784),
                i2.m2(3784, i6, 1567),
                c,
            )
        }
        (true, true) => iadst8_tail(
            i0.m1(-4076),
            i0.m1(401),
            i2.m1(-3612),
            i2.m1(1931),
            i3.m1(3166),
            i3.m1(2598),
            i1.m1(3920),
            i1.m1(1189),
            c,
        ),
        (true, false) => {
            let [i4, i5, i6, i7] = [t[4], t[5], t[6], t[7]];
            iadst8_tail(
                i7.m2_wrap(401, i0, -4076),
                i7.m2_wrap(4076, i0, 401),
                i5.m2_wrap(1931, i2, -3612),
                i5.m2_wrap(3612, i2, 1931),
                i3.m2_wrap(3166, i4, -2598),
                i3.m2_wrap(2598, i4, 3166),
                i1.m2_wrap(3920, i6, -1189),
                i1.m2_wrap(1189, i6, 3920),
                c,
            )
        }
    };
    t.copy_from_slice(&o);
}

/// Fast path of [`transform_add_square`] for 8x8 blocks when `eob`
/// guarantees all non-zero coefficients lie in the top-left 4x4
/// (`eob_rows <= 4 && eob_cols <= 4`). Loads/zeroes only the 4x4
/// sub-block and transforms the 4 active rows on 4-lane vectors.
#[inline(always)]
fn transform_add_8x8_sub4<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    coeffs: &mut [i32],
    p: &Params,
) {
    debug_assert!(p.w == 8 && p.h == 8 && p.eob_rows <= 4 && p.eob_cols <= 4);
    let coeffs = &mut coeffs[..64];
    let rows: [Simd<i32, 4>; 4] = [
        Simd::from_slice(&coeffs[..4]),
        Simd::from_slice(&coeffs[8..12]),
        Simd::from_slice(&coeffs[16..20]),
        Simd::from_slice(&coeffs[24..28]),
    ];
    for i in 0..4 {
        coeffs[i * 8..][..4].fill(0);
    }
    // 8-bit Dequant has 16 bits; 10-bit rows accepted by `rows_fit_s16`
    // (checked on the saturated values) are exact on 16-bit lanes.
    let tile16 = transpose_e(amap(
        rows,
        |r| if BD == 8 { r.cast::<i16>() } else { sat16(r) },
    ));
    let t: [Simd<i16, 4>; 8] = if BD == 8 || large::rows_fit_s16(tile16) {
        let tile = amap(tile16, S16);
        let mut t = [S16::<4>::splat(0); 8];
        t[..4].copy_from_slice(&tile);
        if p.row_kind != Tx1d::Identity {
            dct_adst_small(p.row_kind, true, &mut t, Clamp::bits(16));
            t = amap(t, |v| v.rnd(1));
        }
        amap(t, |v| v.0)
    } else {
        sub4_8x8_rows32::<BD>(rows, p.row_kind)
    };
    let lo = transpose_e([t[0], t[1], t[2], t[3]]);
    let hi = transpose_e([t[4], t[5], t[6], t[7]]);
    let mut c = [S16::<8>::splat(0); 8];
    for i in 0..4 {
        c[i] = S16(simd_swizzle!(lo[i], hi[i], [0, 1, 2, 3, 4, 5, 6, 7]));
    }
    if p.col_kind == Tx1d::Identity {
        // Rows `i < 4` go to row `i`, or to row `7 - i` (i.e. rows 4..8
        // upside down) with `flip_ud`.
        let r = amap([0, 1, 2, 3], |i| col_identity_rnd4::<8, 8>(c[i].0));
        let base = if p.flip_ud { 4 * stride } else { 0 };
        add_rows16::<T, 8, 4, BD>(&mut dst[base..], stride, p, &r);
        return;
    }
    dct_adst_small(p.col_kind, true, &mut c, Clamp::bits(16));
    let r = amap(c, |v| v.rnd4_add());
    add_rows16::<T, 8, 8, BD>(dst, stride, p, &r);
}

/// 32-bit lane row pass fallback for [`transform_add_8x8_sub4`] when a
/// 10-bit row exceeds [`large::S16_ROW_L1_MAX`].
#[cold]
#[inline(never)]
fn sub4_8x8_rows32<const BD: u32>(rows: [Simd<i32, 4>; 4], row_kind: Tx1d) -> [Simd<i16, 4>; 8] {
    let tile = transpose_e(rows);
    let mut t = [Simd::<i32, 4>::splat(0); 8];
    t[..4].copy_from_slice(&tile);
    if row_kind == Tx1d::Identity {
        for v in &mut t[..4] {
            *v = identity::<_, 8>(*v);
        }
    } else {
        dct_adst_small(row_kind, true, &mut t, Clamp::bits(BD + 8));
    }
    let col_clamp = Clamp::<Simd<i32, 4>>::bits(16);
    amap(t, |v| col_clamp.apply(v.rnd(1)).cast())
}

/// `[Round2(lo, 12) | Round2(hi, 12)]` narrowed with saturation
/// (`psrad` + `packssdw`).
#[inline(always)]
fn pack_rnd12(lo: Simd<i32, 4>, hi: Simd<i32, 4>) -> Simd<i16, 8> {
    let r = Simd::splat(1 << 11);
    simd_swizzle!(
        sat16((lo + r) >> 12),
        sat16((hi + r) >> 12),
        [0, 1, 2, 3, 4, 5, 6, 7]
    )
}

/// 4-point DCT or ADST of 4 vectors of 4 lanes held as `a = [vec 0 |
/// vec 1]`, `b = [vec 2 | vec 3]`, returning the transposed outputs
/// `[out 0 | out 1]`, `[out 2 | out 3]` for ADST and `[out 0 | out 1]`,
/// `[out 3 | out 2]` for DCT (`out k` has lane `i` = output `k` of
/// `vec i`), with `pmaddwd` on horizontal pairs. Leaving `[out 3 | out 2]`
/// unswapped in DCT avoids a 64-bit half-swap shuffle between the row
/// and column passes; [`transform_add_4x4_packed`] accounts for the
/// swapped pair after the column rounding shift.
#[inline(always)]
fn dct_adst4_packed(adst: bool, a: Simd<i16, 8>, b: Simd<i16, 8>) -> [Simd<i16, 8>; 2] {
    if adst {
        // Horizontal pairs (in0, in1) and (in2, in3) of each vector.
        let q01 = simd_swizzle!(a, b, [0, 1, 4, 5, 8, 9, 12, 13]);
        let q23 = simd_swizzle!(a, b, [2, 3, 6, 7, 10, 11, 14, 15]);
        let p0 = pmadd8(q01, 1321, 3344);
        let o0 = p0 + pmadd8(q23, 3803, 2482);
        let o1 = pmadd8(q01, 2482, 3344) + pmadd8(q23, -1321, -3803);
        // 3344 * in0 = (4665 * in0 + 3344 * in1) - (1321 * in0 + 3344 * in1).
        let o2 = (pmadd8(q01, 4665, 3344) - p0) + pmadd8(q23, -3344, 3344);
        let o3 = pmadd8(q01, 3803, -3344) + pmadd8(q23, 2482, -1321);
        [pack_rnd12(o0, o1), pack_rnd12(o2, o3)]
    } else {
        // Horizontal pairs (in0, in2) and (in1, in3) of each vector.
        let p02 = simd_swizzle!(a, b, [0, 2, 4, 6, 8, 10, 12, 14]);
        let p13 = simd_swizzle!(a, b, [1, 3, 5, 7, 9, 11, 13, 15]);
        let u = pack_rnd12(pmadd8(p02, 2896, 2896), pmadd8(p02, 2896, -2896));
        let v = pack_rnd12(pmadd8(p13, 3784, 1567), pmadd8(p13, 1567, -3784));
        // u = [t0 | t1], v = [t3 | t2]: outputs [t0 + t3 | t1 + t2] =
        // [out 0 | out 1] and [t0 - t3 | t1 - t2] = [out 3 | out 2].
        [u.saturating_add(v), u.saturating_sub(v)]
    }
}

/// 8-bit and 10-bit 4x4 DCT / ADST transforms without flips on x86,
/// with the whole block in two 128-bit registers (`[row 0 | row 1]`,
/// `[row 2 | row 3]`) instead of four half-empty 4-lane vectors. Returns
/// false (doing nothing) for other types.
#[inline(always)]
fn transform_add_4x4_packed<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    coeffs: &mut [i32],
    p: &Params,
) -> bool {
    if !cfg!(target_arch = "x86_64")
        || BD > 10
        || p.flip_lr
        || p.flip_ud
        || p.row_kind == Tx1d::Identity
        || p.col_kind == Tx1d::Identity
    {
        return false;
    }
    let c: &mut [i32; 16] = (&mut coeffs[..16]).try_into().unwrap();
    // Dequant has 16 bits for 8-bit streams.
    let a = sat16(Simd::<i32, 8>::from_slice(&c[..8]));
    let b = sat16(Simd::<i32, 8>::from_slice(&c[8..]));
    if BD == 10 {
        // 10-bit rows are exact on 16-bit lanes if their absolute sums
        // are at most `S16_ROW_L1_MAX` (`large::rows_fit_s16`), which holds
        // if every coefficient is at most a quarter of it (coefficients
        // beyond 16 bits saturate above that).
        let zero = Simd::<i16, 8>::splat(0);
        let abs_a = a.simd_max(zero.saturating_sub(a));
        let abs_b = b.simd_max(zero.saturating_sub(b));
        if !abs_a
            .simd_max(abs_b)
            .simd_le(Simd::splat(large::S16_ROW_L1_MAX / 4))
            .all()
        {
            transform_add_4x4_fallback::<T, BD>(dst, stride, coeffs, p.row_kind, p.col_kind);
            return true;
        }
    }
    *c = [0; 16];
    // Rows (rowShift = 0, 16-bit row clamp = saturation).
    let [o01, o23] = dct_adst4_packed(p.row_kind == Tx1d::Adst, a, b);
    // Columns.
    let [r01, r23] = dct_adst4_packed(p.col_kind == Tx1d::Adst, o01, o23);
    let (r01, r23) = (S16(r01).rnd4_add(), S16(r23).rnd4_add());
    let (r2, r3) = if p.col_kind == Tx1d::Adst {
        (r23.resize::<4>(0), simd_swizzle!(r23, [4, 5, 6, 7]))
    } else {
        (simd_swizzle!(r23, [4, 5, 6, 7]), r23.resize::<4>(0))
    };
    let mut rows = [r01.resize::<4>(0), simd_swizzle!(r01, [4, 5, 6, 7]), r2, r3];
    if p.row_kind == Tx1d::Dct {
        rows = amap(rows, |r| simd_swizzle!(r, [0, 1, 3, 2]));
    }
    visit_rows::<T, 4, 4>(dst, stride, |y, d| add_row16::<T, 4, BD>(d, rows[y]));
    true
}

/// Cold fallback for [`transform_add_4x4_packed`] when a 10-bit
/// coefficient exceeds `S16_ROW_L1_MAX / 4`.
#[cold]
#[inline(never)]
fn transform_add_4x4_fallback<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    coeffs: &mut [i32],
    row_kind: Tx1d,
    col_kind: Tx1d,
) {
    let coeffs = &mut coeffs[..16];
    let rows: [Simd<i32, 4>; 4] = [
        Simd::from_slice(&coeffs[..4]),
        Simd::from_slice(&coeffs[4..8]),
        Simd::from_slice(&coeffs[8..12]),
        Simd::from_slice(&coeffs[12..16]),
    ];
    coeffs.fill(0);
    let t16 = transpose_e(amap(rows, sat16));
    let t = if large::rows_fit_s16(t16) {
        let mut t = amap(t16, S16);
        dct_adst_small(row_kind, true, &mut t, Clamp::bits(16));
        amap(t, |v| v.0)
    } else {
        square_rows32::<BD, 4>(rows, row_kind)
    };
    let mut c = amap(transpose_e(t), S16);
    dct_adst_small(col_kind, true, &mut c, Clamp::bits(16));
    let r = amap(c, |v| v.rnd4_add());
    visit_rows::<T, 4, 4>(dst, stride, |y, d| add_row16::<T, 4, BD>(d, r[y]));
}

/// Transposes a 4x4 block of `i16` held as `[row 0 | row 1]`, `[row 2 |
/// row 3]` into `[col 0 | col 1]`, `[col 2 | col 3]` (two `zip1` / `zip2`
/// pairs on NEON).
#[inline(always)]
fn transpose4x4_packed(a: Simd<i16, 8>, b: Simd<i16, 8>) -> [Simd<i16, 8>; 2] {
    let (z0, z1) = a.interleave(b);
    let (c01, c23) = z0.interleave(z1);
    [c01, c23]
}

/// `Round2(x, 12)` of the two 4-lane `B` butterfly sums narrowed *modulo
/// 2^16* and packed as `[lo | hi]` (`rshrn` + `rshrn2` on NEON). Exact
/// whenever the results fit in 16 bits, which 7.13.2.1 guarantees for
/// conformant `DCT4` streams (see `S16`; for 10-bit rows
/// [`transform_add_4x4_neon`] checks it).
#[inline(always)]
fn narrow2_rnd12(lo: Simd<i32, 4>, hi: Simd<i32, 4>) -> Simd<i16, 8> {
    let r = Simd::splat(1 << 11);
    simd_swizzle!(
        ((lo + r) >> 12).cast::<i16>(),
        ((hi + r) >> 12).cast::<i16>(),
        [0, 1, 2, 3, 4, 5, 6, 7]
    )
}

/// 4-point DCT or ADST (spec 7.13.2.3 / 7.13.2.6) of 4 transforms held as
/// `x01 = [in 0 | in 1]`, `x23 = [in 2 | in 3]` (lane `i` of `in j` is
/// input `j` of transform `i`), returning `[out 0 | out 1]`, `[out 2 |
/// out 3]`. Every product is a `smull` / `smlal{,2}` by an element of the
/// constant vector `k` (`ADST4_K` or `DCT4_K`), so that the constants
/// live in one register instead of one `dup` each. The `i32` sums are
/// exact for 16-bit inputs (the absolute constants of an output add up
/// to at most 10950 < 2^16). `ADST4` narrows with saturation (`pack_rnd12`)
/// because 7.13.2.6 bounds `o0..o3` to `< 2^27` before `Round2(.., 12)`,
/// allowing `Round2(o, 12) == 32768` on conformant streams.
#[inline(always)]
fn dct_adst4_neon(
    adst: bool,
    x01: Simd<i16, 8>,
    x23: Simd<i16, 8>,
    k: Simd<i16, 8>,
) -> [Simd<i16, 8>; 2] {
    macro_rules! kl {
        ($i:literal) => {
            simd_swizzle!(k, [$i; 4]).cast::<i32>()
        };
    }
    let x0 = simd_swizzle!(x01, [0, 1, 2, 3]).cast::<i32>();
    let x1 = simd_swizzle!(x01, [4, 5, 6, 7]).cast::<i32>();
    let x2 = simd_swizzle!(x23, [0, 1, 2, 3]).cast::<i32>();
    let x3 = simd_swizzle!(x23, [4, 5, 6, 7]).cast::<i32>();
    if adst {
        // k = [1321, 2482, 3344, 3803, -1321, -2482, -3344, -3803].
        let o0 = x0 * kl!(0) + x1 * kl!(2) + x2 * kl!(3) + x3 * kl!(1);
        let o1 = x0 * kl!(1) + x1 * kl!(2) + x2 * kl!(4) + x3 * kl!(7);
        let o2 = x0 * kl!(2) + x2 * kl!(6) + x3 * kl!(2);
        let o3 = x0 * kl!(3) + x1 * kl!(6) + x2 * kl!(1) + x3 * kl!(4);
        [pack_rnd12(o0, o1), pack_rnd12(o2, o3)]
    } else {
        // k = [2896, -2896, 1567, 3784, -3784, 0, 0, 0].
        let a0 = x0 * kl!(0) + x2 * kl!(0);
        let a1 = x0 * kl!(0) + x2 * kl!(1);
        let b3 = x1 * kl!(3) + x3 * kl!(2);
        let b2 = x1 * kl!(2) + x3 * kl!(4);
        let a01 = narrow2_rnd12(a0, a1);
        let b32 = narrow2_rnd12(b3, b2);
        // [t0 + t3 | t1 + t2] and [t0 - t3 | t1 - t2] = [out 3 | out 2].
        let d = a01.saturating_sub(b32);
        [
            a01.saturating_add(b32),
            simd_swizzle!(d, [4, 5, 6, 7, 0, 1, 2, 3]),
        ]
    }
}

/// Constants of [`dct_adst4_neon`].
const ADST4_K: [i16; 8] = [1321, 2482, 3344, 3803, -1321, -2482, -3344, -3803];
const DCT4_K: [i16; 8] = [2896, -2896, 1567, 3784, -3784, 0, 0, 0];

/// Adds the residual rows `r = [row a | row b]` to the two 4-pixel rows
/// `da` and `db` with clipping to `BD` bits.
#[inline(always)]
fn add_rows2_4<T: Pixel, const BD: u32>(da: &mut [T; 4], db: &mut [T; 4], r: Simd<i16, 8>) {
    let max = Simd::splat((1 << BD) - 1);
    let (aa, ab): (&mut dyn Any, &mut dyn Any) = (&mut *da, &mut *db);
    if let (Some(da), Some(db)) = (aa.downcast_mut::<[u16; 4]>(), ab.downcast_mut::<[u16; 4]>()) {
        let d: Simd<u16, 8> = simd_swizzle!(
            Simd::from_array(*da),
            Simd::from_array(*db),
            [0, 1, 2, 3, 4, 5, 6, 7]
        );
        let v = (d.cast::<i16>() + r)
            .simd_clamp(Simd::splat(0), max)
            .cast::<u16>();
        *da = simd_swizzle!(v, [0, 1, 2, 3]).to_array();
        *db = simd_swizzle!(v, [4, 5, 6, 7]).to_array();
    } else if let (Some(da), Some(db)) =
        (aa.downcast_mut::<[u8; 4]>(), ab.downcast_mut::<[u8; 4]>())
    {
        let d: Simd<u8, 8> = simd_swizzle!(
            Simd::from_array(*da),
            Simd::from_array(*db),
            [0, 1, 2, 3, 4, 5, 6, 7]
        );
        let v = (d.cast::<i16>() + r)
            .simd_clamp(Simd::splat(0), max)
            .cast::<u8>();
        *da = simd_swizzle!(v, [0, 1, 2, 3]).to_array();
        *db = simd_swizzle!(v, [4, 5, 6, 7]).to_array();
    } else {
        // Not reached (pixels are u8 or u16).
        add_row16::<T, 4, BD>(da, simd_swizzle!(r, [0, 1, 2, 3]));
        add_row16::<T, 4, BD>(db, simd_swizzle!(r, [4, 5, 6, 7]));
    }
}

/// 8-bit and 10-bit 4x4 DCT / ADST transforms without flips on aarch64,
/// with the block in two 128-bit registers (`[row 0 | row 1]`, `[row 2 |
/// row 3]`) and the transform constants in one register per kind (see
/// [`dct_adst4_neon`]). Returns false (doing nothing) for other types.
#[inline(always)]
fn transform_add_4x4_neon<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    coeffs: &mut [i32],
    p: &Params,
) -> bool {
    if !cfg!(target_arch = "aarch64")
        || BD > 10
        || p.flip_lr
        || p.flip_ud
        || p.row_kind == Tx1d::Identity
        || p.col_kind == Tx1d::Identity
    {
        return false;
    }
    transform_add_4x4_neon_impl::<T, BD>(dst, stride, coeffs, p.row_kind, p.col_kind);
    true
}

/// Body of [`transform_add_4x4_neon`] (any architecture, for the tests):
/// `row_kind` and `col_kind` are DCT or ADST.
#[inline(always)]
fn transform_add_4x4_neon_impl<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    coeffs: &mut [i32],
    row_kind: Tx1d,
    col_kind: Tx1d,
) {
    // A 4x4 block always has 16 coefficients.
    let Some(c) = coeffs.first_chunk_mut::<16>() else {
        return;
    };
    // Dequant has 16 bits for 8-bit streams.
    let a = sat16(Simd::<i32, 8>::from_slice(&c[..8]));
    let b = sat16(Simd::<i32, 8>::from_slice(&c[8..]));
    if BD == 10 {
        // 10-bit rows are exact on 16-bit lanes if their absolute sums
        // are at most `S16_ROW_L1_MAX` (`large::rows_fit_s16`), which holds
        // if every coefficient is at most a quarter of it (coefficients
        // beyond 16 bits saturate above that). Then every row output
        // (at most the absolute sum, all constants being below 4096)
        // fits in 16 bits too. `|x| <= L` is `(x + L) as u16 <= 2 * L`.
        const L: i16 = large::S16_ROW_L1_MAX / 4;
        let bias = |v: Simd<i16, 8>| (v + Simd::splat(L)).cast::<u16>();
        if bias(a).simd_max(bias(b)).reduce_max() > 2 * L as u16 {
            transform_add_4x4_fallback::<T, BD>(dst, stride, coeffs, row_kind, col_kind);
            return;
        }
    }
    *c = [0; 16];
    let (row_adst, col_adst) = (row_kind == Tx1d::Adst, col_kind == Tx1d::Adst);
    // Opaque to the optimizer, so that the products use the constants
    // as vector elements instead of materializing one splat per constant.
    let kr = std::hint::black_box(Simd::from_array(if row_adst { ADST4_K } else { DCT4_K }));
    let kc = if row_adst == col_adst {
        kr
    } else {
        std::hint::black_box(Simd::from_array(if col_adst { ADST4_K } else { DCT4_K }))
    };
    // Rows (rowShift = 0, 16-bit row clamp = saturation): out k lane i is
    // Residual[i][k].
    let [t01, t23] = transpose4x4_packed(a, b);
    let [o01, o23] = dct_adst4_neon(row_adst, t01, t23, kr);
    // Columns: lane j of in i is Residual[i][j]; out k lane j is row k.
    let [c01, c23] = transpose4x4_packed(o01, o23);
    let [r01, r23] = dct_adst4_neon(col_adst, c01, c23, kc);
    let (r01, r23) = (S16(r01).rnd4_add(), S16(r23).rnd4_add());
    let stride = stride & 0xffff_ffff;
    assert!(stride >= 4);
    let dst = &mut dst[..3 * stride + 4];
    let (top, bot) = dst.split_at_mut(2 * stride);
    let (d0, d1) = top.split_at_mut(stride);
    let (d2, d3) = bot.split_at_mut(stride);
    // Every row has at least 4 pixels (`stride >= 4`).
    let (Some(d0), Some(d1), Some(d2), Some(d3)) = (
        d0.first_chunk_mut::<4>(),
        d1.first_chunk_mut::<4>(),
        d2.first_chunk_mut::<4>(),
        d3.first_chunk_mut::<4>(),
    ) else {
        return;
    };
    add_rows2_4::<T, BD>(d0, d1, r01);
    add_rows2_4::<T, BD>(d2, d3, r23);
}

/// `N x N` transforms (`N` = 4 or 8, `BD` = 8 or 10) with all data in
/// registers: one row (column) per lane, transposes between the passes.
/// 10-bit rows are transformed on `i32` lanes (18-bit clamping range).
#[inline(always)]
fn transform_add_square<T: Pixel, const BD: u32, const N: usize>(
    dst: &mut [T],
    stride: usize,
    coeffs: &mut [i32],
    p: &Params,
) {
    debug_assert!(p.w == N && p.h == N && p.bd == BD && BD <= 10 && coeffs.len() == N * N);
    if N == 4 && transform_add_4x4_packed::<T, BD>(dst, stride, coeffs, p) {
        return;
    }
    if N == 4 && transform_add_4x4_neon::<T, BD>(dst, stride, coeffs, p) {
        return;
    }
    if N == 8 && p.eob_rows <= 4 && p.eob_cols <= 4 {
        transform_add_8x8_sub4::<T, BD>(dst, stride, coeffs, p);
        return;
    }
    let coeffs = &mut coeffs[..N * N];
    let mut rows = [Simd::<i32, N>::splat(0); N];
    for i in 0..N {
        rows[i] = Simd::from_slice(&coeffs[i * N..][..N]);
    }
    coeffs.fill(0);
    let rshift = const { row_shift(N, N) };
    let rows_le4 = N == 4
        || p.eob_rows <= 4
        || ((rows[N / 2] | rows[N / 2 + 1]) | (rows[N - 2] | rows[N - 1])) == Simd::splat(0);
    // Row transforms: t[j] lane i = Residual[i][j] (clamped to 16 bits).
    let t: [Simd<i16, N>; N] = if BD == 8 {
        // Dequant has 16 bits for 8-bit streams.
        square_rows16::<N>(transpose_e(amap(rows, |r| r.cast::<i16>())), p, rshift)
    } else {
        let t16 = transpose_e(amap(rows, sat16));
        if large::rows_fit_s16(t16) {
            // 16-bit lanes are exact for these rows (see `rows_fit_s16`).
            square_rows16::<N>(t16, p, rshift)
        } else {
            square_rows32::<BD, N>(rows, p.row_kind)
        }
    };
    // c[i] lane j = Residual[i][j]; rows at or beyond nz_rows are zero.
    let mut c = amap(transpose_e(t), S16);
    if p.col_kind == Tx1d::Identity {
        for i in 0..p.eob_rows.min(N) {
            let y = if p.flip_ud { N - 1 - i } else { i };
            let v = col_identity_rnd4::<N, N>(c[i].0);
            add_row16_at::<T, N, BD>(dst, y * stride, if p.flip_lr { v.reverse() } else { v });
        }
        return;
    }
    dct_adst_small(p.col_kind, rows_le4, &mut c, Clamp::bits(16));
    let r = amap(c, |v| v.rnd4_add());
    add_rows16::<T, N, N, BD>(dst, stride, p, &r);
}

/// 32-bit lane row pass fallback for [`transform_add_square`] when a
/// 10-bit row exceeds [`large::S16_ROW_L1_MAX`].
#[cold]
#[inline(never)]
fn square_rows32<const BD: u32, const N: usize>(
    rows: [Simd<i32, N>; N],
    row_kind: Tx1d,
) -> [Simd<i16, N>; N] {
    let rshift = const { row_shift(N, N) };
    let mut t = transpose_e(rows);
    let cols_le4 = N == 4 || ((t[N / 2] | t[N / 2 + 1]) | (t[N - 2] | t[N - 1])) == Simd::splat(0);
    if row_kind == Tx1d::Identity {
        t = amap(t, identity::<_, N>);
    } else {
        dct_adst_small(row_kind, cols_le4, &mut t, Clamp::bits(BD + 8));
    }
    let col_clamp = Clamp::<Simd<i32, N>>::bits(16);
    amap(t, |v| col_clamp.apply(v.rnd(rshift)).cast())
}

/// Row transforms of [`transform_add_square`] on 16-bit lanes: `t16[j]`
/// lane `i` is `Dequant[i][j]`; returns `t[j]` lane `i` = Residual[i][j].
/// Exact for 8-bit streams and for 10-bit rows accepted by
/// [`large::rows_fit_s16`] (whose identity outputs `Round2(x * 5793, 12)`
/// (4x4) and `x` (8x8) then saturate exactly like the clamp to the
/// 16-bit column range).
#[inline(always)]
fn square_rows16<const N: usize>(
    t16: [Simd<i16, N>; N],
    p: &Params,
    rshift: u32,
) -> [Simd<i16, N>; N] {
    let cols_le4 = N == 4
        || p.eob_cols <= 4
        || ((t16[N / 2] | t16[N / 2 + 1]) | (t16[N - 2] | t16[N - 1])) == Simd::splat(0);
    let mut t = amap(t16, S16);
    if p.row_kind == Tx1d::Identity {
        // 4x4: Round2(x * 5793, 12) clamped to 16 bits (rowShift =
        // 0). 8x8: Round2(2 * x, rowShift = 1) = x.
        if N == 4 {
            t = amap(t, |v| v.m1(5793));
        }
    } else {
        dct_adst_small(p.row_kind, cols_le4, &mut t, Clamp::bits(16));
        t = amap(t, |v| v.rnd(rshift));
    }
    amap(t, |v| v.0)
}

/// 8x4 and 4x8 transforms (`BD` = 8 or 10) with all data in registers
/// (the transposes are done as 8x8 ones with zero padding).
#[inline(always)]
fn transform_add_rect<T: Pixel, const BD: u32, const W: usize, const H: usize>(
    dst: &mut [T],
    stride: usize,
    coeffs: &mut [i32],
    p: &Params,
) {
    debug_assert!(p.w == W && p.h == H && p.bd == BD && BD <= 10 && p.rect2 && p.row_shift == 0);
    let coeffs = &mut coeffs[..W * H];
    let mut rows = [Simd::<i32, W>::splat(0); H];
    for i in 0..H {
        rows[i] = Simd::from_slice(&coeffs[i * W..][..W]);
    }
    coeffs.fill(0);
    let rows_le4 = H == 4
        || p.eob_rows <= 4
        || ((rows[H / 2] | rows[H / 2 + 1]) | (rows[H - 2] | rows[H - 1])) == Simd::splat(0);
    let z16 = Simd::<i16, 8>::splat(0);
    // Row transforms on 16-bit lanes (see `rect_rows16`).
    let t_cols: [Simd<i16, H>; W] = if BD == 8 {
        let mut m = [z16; 8];
        for i in 0..H {
            m[i] = rows[i].cast::<i16>().resize::<8>(0);
        }
        rect_rows16::<W, H>(transpose_e(m), p)
    } else {
        let mut m16 = [z16; 8];
        for i in 0..H {
            m16[i] = sat16(rows[i]).resize::<8>(0);
        }
        let m16 = transpose_e(m16);
        if large::rows_fit_s16(m16[..W].iter().copied()) {
            rect_rows16::<W, H>(m16, p)
        } else {
            rect_rows32::<BD, W, H>(rows, p.row_kind)
        }
    };
    let mut m = [z16; 8];
    for j in 0..W {
        m[j] = t_cols[j].resize::<8>(0);
    }
    // c[i] lane j = Residual[i][j].
    let m = transpose_e(m);
    let mut c = [S16::<W>::splat(0); H];
    for i in 0..H {
        c[i] = S16(m[i].resize::<W>(0));
    }
    if p.col_kind == Tx1d::Identity {
        for i in 0..p.eob_rows.min(H) {
            let y = if p.flip_ud { H - 1 - i } else { i };
            let v = col_identity_rnd4::<W, H>(c[i].0);
            add_row16_at::<T, W, BD>(dst, y * stride, if p.flip_lr { v.reverse() } else { v });
        }
        return;
    }
    dct_adst_small(p.col_kind, rows_le4, &mut c, Clamp::bits(16));
    let r = amap(c, |v| v.rnd4_add());
    add_rows16::<T, W, H, BD>(dst, stride, p, &r);
}

/// Row transforms of [`transform_add_rect`] on 16-bit lanes from `m[j]`
/// lane `i` = Dequant[i][j]: exact for 8-bit streams and for 10-bit rows
/// accepted by `rows_fit_s16` (the identity outputs then saturate like
/// the clamp to 16 bits, rowShift being 0). A function rather than a
/// closure so that it is inlined on aarch64.
#[inline(always)]
fn rect_rows16<const W: usize, const H: usize>(
    m: [Simd<i16, 8>; 8],
    p: &Params,
) -> [Simd<i16, H>; W] {
    let z16 = Simd::<i16, 8>::splat(0);
    let cols_le4 = W == 4 || p.eob_cols <= 4 || ((m[4] | m[5]) | (m[6] | m[7])) == z16;
    // t[j] lane i = Round2(Dequant[i][j] * 2896, 12) (rect2).
    let mut t = [S16::<H>::splat(0); W];
    for j in 0..W {
        t[j] = S16(m[j].resize::<H>(0)).m1_h0(2896);
    }
    if p.row_kind == Tx1d::Identity {
        // rowShift = 0, so the clamp to 16 bits is the saturation.
        t = if W == 8 { amap(t, |v| v + v) } else { amap(t, |v| v.m1(5793)) };
    } else {
        dct_adst_small(p.row_kind, cols_le4, &mut t, Clamp::bits(16));
    }
    amap(t, |v| v.0)
}

/// 32-bit lane row pass fallback for [`transform_add_rect`] when a
/// 10-bit row exceeds [`large::S16_ROW_L1_MAX`].
#[cold]
#[inline(never)]
fn rect_rows32<const BD: u32, const W: usize, const H: usize>(
    rows: [Simd<i32, W>; H],
    row_kind: Tx1d,
) -> [Simd<i16, H>; W] {
    let z32 = Simd::<i32, 8>::splat(0);
    let mut m = [z32; 8];
    for i in 0..H {
        m[i] = rows[i].resize::<8>(0);
    }
    let m = transpose_e(m);
    let cols_le4 = W == 4 || ((m[4] | m[5]) | (m[6] | m[7])) == z32;
    let mut t = [Simd::<i32, H>::splat(0); W];
    for j in 0..W {
        t[j] = m[j].resize::<H>(0).m1(2896);
    }
    if row_kind == Tx1d::Identity {
        t = amap(t, identity::<_, W>);
    } else {
        dct_adst_small(row_kind, cols_le4, &mut t, Clamp::bits(BD + 8));
    }
    let col_clamp = Clamp::<Simd<i32, H>>::bits(16);
    amap(t, |v| col_clamp.apply(v).cast())
}

/// Dispatches [`large::transform_add16`] on the transform size (generic
/// parameters after the size: `TH = min(H, 32)`, row / column pass lane
/// counts, then the sizes of its column buffers). Inlined into
/// [`size_entry`], where the size is a constant.
#[inline(always)]
fn transform_add16_dispatch<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    coeffs: &mut [i32],
    p: &Params,
) {
    // Column lanes for widths >= 16: one 256-bit vector of 16 `i16` with
    // AVX2, else 8 (two 128-bit NEON registers per vector would double
    // the register pressure of the 32-point kernels).
    const CW: usize = if cfg!(target_feature = "avx2") { 16 } else { 8 };
    match (p.w, p.h) {
        (4, 4) => return transform_add_square::<T, BD, 4>(dst, stride, coeffs, p),
        (8, 8) => return transform_add_square::<T, BD, 8>(dst, stride, coeffs, p),
        (8, 4) => return transform_add_rect::<T, BD, 8, 4>(dst, stride, coeffs, p),
        (4, 8) => return transform_add_rect::<T, BD, 4, 8>(dst, stride, coeffs, p),
        _ => {}
    }
    let f = match (p.w, p.h) {
        (4, 16) => large::transform_add16::<T, BD, 4, 16, 16, 4, 4, { 64 / 4 }, { 16 / 4 }>,
        (8, 16) => large::transform_add16::<T, BD, 8, 16, 16, 8, 8, { 128 / 8 }, { 32 / 8 }>,
        (8, 32) => large::transform_add16::<T, BD, 8, 32, 32, 8, 8, { 256 / 8 }, { 32 / 8 }>,
        (16, 4) => large::transform_add16::<T, BD, 16, 4, 4, 4, CW, { 64 / CW }, { 64 / CW }>,
        (16, 8) => large::transform_add16::<T, BD, 16, 8, 8, 8, CW, { 128 / CW }, { 64 / CW }>,
        (16, 16) => large::transform_add16::<T, BD, 16, 16, 16, 8, CW, { 256 / CW }, { 64 / CW }>,
        (16, 32) => large::transform_add16::<T, BD, 16, 32, 32, 8, CW, { 512 / CW }, { 64 / CW }>,
        (16, 64) => large::transform_add16::<T, BD, 16, 64, 32, 8, CW, { 512 / CW }, { 64 / CW }>,
        (32, 8) => large::transform_add16::<T, BD, 32, 8, 8, 8, CW, { 256 / CW }, { 128 / CW }>,
        (32, 16) => large::transform_add16::<T, BD, 32, 16, 16, 8, CW, { 512 / CW }, { 128 / CW }>,
        (32, 32) => large::transform_add16::<T, BD, 32, 32, 32, 8, CW, { 1024 / CW }, { 128 / CW }>,
        (32, 64) => large::transform_add16::<T, BD, 32, 64, 32, 8, CW, { 1024 / CW }, { 128 / CW }>,
        (64, 16) => large::transform_add16::<T, BD, 64, 16, 16, 8, CW, { 1024 / CW }, { 256 / CW }>,
        (64, 32) => large::transform_add16::<T, BD, 64, 32, 32, 8, CW, { 2048 / CW }, { 256 / CW }>,
        _ => large::transform_add16::<T, BD, 64, 64, 32, 8, CW, { 2048 / CW }, { 256 / CW }>,
    };
    f(dst, stride, coeffs, p)
}

/// `Round2(identity_H(v), 4)` (the column identity transform and the
/// final rounding).
#[inline(always)]
fn col_identity_rnd4<const N: usize, const H: usize>(v: Simd<i16, N>) -> Simd<i16, N> {
    if H == 8 {
        return rnd16(v, 3);
    }
    if H == 32 {
        return rnd16(v, 2);
    }
    let k: i16 = if H == 4 { 5793 } else { 11586 };
    if cfg!(target_arch = "x86_64") {
        let hi = ((v.cast::<i32>() * Simd::splat(k as i32)) >> 16).cast::<i16>();
        let lo = (v * Simd::splat(k)).cast::<u16>();
        return hi + (((lo >> 11) + Simd::splat(17)) >> 5).cast::<i16>();
    }
    let x = v.cast::<i32>();
    ((x * Simd::splat(k as i32) + Simd::splat(34816)) >> 16).cast()
}

/// Adds the residual `r` to `dst[off..off + N]` with clipping to `BD`
/// bits (`|r| <= 2^11`, so the sum cannot overflow).
///
/// Caller (`inv_txfm_add_bd`) has verified `(h - 1) * stride + w <=
/// dst.len()` and `stride >= w`, so the bounds check never fails. Loops
/// over rows should prefer `add_rows16` or `chunks_mut(stride)`.
#[inline(always)]
fn add_row16_at<T: Pixel, const N: usize, const BD: u32>(
    dst: &mut [T],
    off: usize,
    r: Simd<i16, N>,
) {
    let d: &mut [T; N] = (&mut dst[off..off + N]).try_into().unwrap();
    add_row16::<T, N, BD>(d, r);
}

/// Adds the residual `r` to `d` with clipping to `BD` bits (`|r| <=
/// 2^11`, so the sum cannot overflow).
#[inline(always)]
fn add_row16<T: Pixel, const N: usize, const BD: u32>(d: &mut [T; N], r: Simd<i16, N>) {
    let max = Simd::splat((1 << BD) - 1);
    let any: &mut dyn Any = d;
    if let Some(d) = any.downcast_mut::<[u8; N]>() {
        let v = Simd::from_array(*d).cast::<i16>() + r;
        *d = v.simd_clamp(Simd::splat(0), max).cast::<u8>().to_array();
    } else if let Some(d) = any.downcast_mut::<[u16; N]>() {
        let v = Simd::from_array(*d).cast::<i16>() + r;
        *d = v.simd_clamp(Simd::splat(0), max).cast::<u16>().to_array();
    } else {
        // Not reached (pixels are u8 or u16).
        for (o, &x) in d.iter_mut().zip(r.as_array()) {
            *o = T::from_i32((o.to_i32() + x as i32).clamp(0, (1 << BD) - 1));
        }
    }
}

/// Calls `f(y, row)` for the `N` pixels at `dst[y * stride..]`, `y in
/// 0..H`, top to bottom. Masks `stride` to 32 bits and slices `dst` up
/// front so `y * stride + N <= dst.len()` is proved without per-row
/// bounds checks.
#[inline(always)]
fn visit_rows<T, const N: usize, const H: usize>(
    dst: &mut [T],
    stride: usize,
    mut f: impl FnMut(usize, &mut [T; N]),
) {
    let stride = stride & 0xffff_ffff;
    assert!(stride >= N);
    if H < 4 {
        let dst = &mut dst[..(H - 1) * stride + N];
        for y in 0..H {
            let base = y * stride;
            let d: &mut [T; N] = (&mut dst[base..base + N]).try_into().unwrap();
            f(y, d);
        }
        return;
    }
    const { assert!(H < 4 || H.is_multiple_of(4)) };
    let s2 = stride * 2;
    let s3 = stride * 3;
    let s4 = stride * 4;
    let head_len = (H / 4 - 1) * s4;
    let dst = &mut dst[..head_len + s3 + N];
    let (head, tail) = dst.split_at_mut(head_len);
    for (y4, win) in head.chunks_exact_mut(s4).enumerate() {
        let y = y4 * 4;
        f(y, (&mut win[..N]).try_into().unwrap());
        f(y + 1, (&mut win[stride..stride + N]).try_into().unwrap());
        f(y + 2, (&mut win[s2..s2 + N]).try_into().unwrap());
        f(y + 3, (&mut win[s3..s3 + N]).try_into().unwrap());
    }
    let y = H - 4;
    f(y, (&mut tail[..N]).try_into().unwrap());
    f(y + 1, (&mut tail[stride..stride + N]).try_into().unwrap());
    f(y + 2, (&mut tail[s2..s2 + N]).try_into().unwrap());
    f(y + 3, (&mut tail[s3..s3 + N]).try_into().unwrap());
}

/// Adds `H` residual rows of width `N` to the left edge of `dst`,
/// applying `p.flip_lr` and `p.flip_ud`. Hoists the flip checks outside
/// the row loop so the common non-flipped path has no per-row branch.
#[inline(always)]
fn add_rows16<T: Pixel, const N: usize, const H: usize, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    p: &Params,
    rows: &[Simd<i16, N>; H],
) {
    if !p.flip_lr && !p.flip_ud {
        visit_rows::<T, N, H>(dst, stride, |y, d| add_row16::<T, N, BD>(d, rows[y]));
    } else {
        visit_rows::<T, N, H>(dst, stride, |y, d| {
            let r = rows[if p.flip_ud { H - 1 - y } else { y }];
            add_row16::<T, N, BD>(d, if p.flip_lr { r.reverse() } else { r });
        });
    }
}

/// Adds the saturated 16-bit residual rows `r = [row a | row b]` to the
/// two 4-pixel rows `da` and `db` with saturating `i16` addition and
/// clipping to `BD` bits (exact for arbitrary residuals narrowed via
/// [`sat16`]).
#[inline(always)]
fn add_rows2_4_sat<T: Pixel, const BD: u32>(da: &mut [T; 4], db: &mut [T; 4], r: Simd<i16, 8>) {
    let max = Simd::splat((1 << BD) - 1);
    let (aa, ab): (&mut dyn Any, &mut dyn Any) = (&mut *da, &mut *db);
    if let (Some(da), Some(db)) = (aa.downcast_mut::<[u16; 4]>(), ab.downcast_mut::<[u16; 4]>()) {
        let d: Simd<u16, 8> = simd_swizzle!(
            Simd::from_array(*da),
            Simd::from_array(*db),
            [0, 1, 2, 3, 4, 5, 6, 7]
        );
        let v = d
            .cast::<i16>()
            .saturating_add(r)
            .simd_clamp(Simd::splat(0), max)
            .cast::<u16>();
        *da = simd_swizzle!(v, [0, 1, 2, 3]).to_array();
        *db = simd_swizzle!(v, [4, 5, 6, 7]).to_array();
    } else if let (Some(da), Some(db)) =
        (aa.downcast_mut::<[u8; 4]>(), ab.downcast_mut::<[u8; 4]>())
    {
        let d: Simd<u8, 8> = simd_swizzle!(
            Simd::from_array(*da),
            Simd::from_array(*db),
            [0, 1, 2, 3, 4, 5, 6, 7]
        );
        let v = d
            .cast::<i16>()
            .saturating_add(r)
            .simd_clamp(Simd::splat(0), max)
            .cast::<u8>();
        *da = simd_swizzle!(v, [0, 1, 2, 3]).to_array();
        *db = simd_swizzle!(v, [4, 5, 6, 7]).to_array();
    } else {
        // Not reached (pixels are u8 or u16).
        add_row_sat16::<T, 4, BD>(da, simd_swizzle!(r, [0, 1, 2, 3]));
        add_row_sat16::<T, 4, BD>(db, simd_swizzle!(r, [4, 5, 6, 7]));
    }
}

/// Calls `f(row0, row1)` on each adjacent pair of 4-pixel rows in `0..H`
/// (`H >= 4` a multiple of 4) without per-row bounds checks.
#[inline(always)]
fn visit_row_pairs_4<T, const H: usize>(
    dst: &mut [T],
    stride: usize,
    mut f: impl FnMut(&mut [T; 4], &mut [T; 4]),
) {
    if H < 4 {
        return;
    }
    const { assert!(H < 4 || H.is_multiple_of(4)) };
    let stride = stride & 0xffff_ffff;
    assert!(stride >= 4);
    let s2 = stride * 2;
    let head_len = (H / 2 - 1) * s2;
    let dst = &mut dst[..head_len + stride + 4];
    let (head, tail) = dst.split_at_mut(head_len);
    for win in head.chunks_exact_mut(s2) {
        let (a, b) = win.split_at_mut(stride);
        f(a.first_chunk_mut().unwrap(), b.first_chunk_mut().unwrap());
    }
    let (a, b) = tail.split_at_mut(stride);
    f(a.first_chunk_mut().unwrap(), b.first_chunk_mut().unwrap());
}

/// Adds the constant residual `dc` (`|dc| <= 2^11`) to a `W x H` block
/// with clipping to `BD` bits, `N` pixels at a time (`N <= W`, `W` a
/// multiple of `N`).
#[inline(always)]
fn add_dc16<T: Pixel, const N: usize, const W: usize, const H: usize, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    dc: i32,
) {
    if W == 4 && H >= 4 {
        let r8 = Simd::<i16, 8>::splat(dc as i16);
        if BD > 8 {
            visit_row_pairs_4::<T, H>(dst, stride, |da, db| add_rows2_4::<T, BD>(da, db, r8));
            return;
        }
        let a8 = Simd::<u8, 8>::splat(dc.unsigned_abs().min(255) as u8);
        if dc >= 0 {
            visit_row_pairs_4::<T, H>(dst, stride, |da, db| {
                let (aa, ab): (&mut dyn Any, &mut dyn Any) = (&mut *da, &mut *db);
                if let (Some(da), Some(db)) =
                    (aa.downcast_mut::<[u8; 4]>(), ab.downcast_mut::<[u8; 4]>())
                {
                    let d: Simd<u8, 8> = simd_swizzle!(
                        Simd::from_array(*da),
                        Simd::from_array(*db),
                        [0, 1, 2, 3, 4, 5, 6, 7]
                    );
                    let v = d.saturating_add(a8);
                    *da = simd_swizzle!(v, [0, 1, 2, 3]).to_array();
                    *db = simd_swizzle!(v, [4, 5, 6, 7]).to_array();
                } else {
                    add_rows2_4::<T, BD>(da, db, r8);
                }
            });
        } else {
            visit_row_pairs_4::<T, H>(dst, stride, |da, db| {
                let (aa, ab): (&mut dyn Any, &mut dyn Any) = (&mut *da, &mut *db);
                if let (Some(da), Some(db)) =
                    (aa.downcast_mut::<[u8; 4]>(), ab.downcast_mut::<[u8; 4]>())
                {
                    let d: Simd<u8, 8> = simd_swizzle!(
                        Simd::from_array(*da),
                        Simd::from_array(*db),
                        [0, 1, 2, 3, 4, 5, 6, 7]
                    );
                    let v = d.saturating_sub(a8);
                    *da = simd_swizzle!(v, [0, 1, 2, 3]).to_array();
                    *db = simd_swizzle!(v, [4, 5, 6, 7]).to_array();
                } else {
                    add_rows2_4::<T, BD>(da, db, r8);
                }
            });
        }
        return;
    }
    let r16 = Simd::<i16, N>::splat(dc as i16);
    if BD > 8 {
        visit_rows::<T, W, H>(dst, stride, |_, row| {
            for x in (0..W / N).map(|k| k * N) {
                let d: &mut [T; N] = (&mut row[x..x + N]).try_into().unwrap();
                add_row16::<T, N, BD>(d, r16);
            }
        });
        return;
    }
    let a8 = Simd::<u8, N>::splat(dc.unsigned_abs().min(255) as u8);
    if dc >= 0 {
        visit_rows::<T, W, H>(dst, stride, |_, row| {
            for x in (0..W / N).map(|k| k * N) {
                let d: &mut [T; N] = (&mut row[x..x + N]).try_into().unwrap();
                let any: &mut dyn Any = d;
                if let Some(d) = any.downcast_mut::<[u8; N]>() {
                    *d = Simd::from_array(*d).saturating_add(a8).to_array();
                } else {
                    add_row16::<T, N, BD>(d, r16);
                }
            }
        });
    } else {
        visit_rows::<T, W, H>(dst, stride, |_, row| {
            for x in (0..W / N).map(|k| k * N) {
                let d: &mut [T; N] = (&mut row[x..x + N]).try_into().unwrap();
                let any: &mut dyn Any = d;
                if let Some(d) = any.downcast_mut::<[u8; N]>() {
                    *d = Simd::from_array(*d).saturating_sub(a8).to_array();
                } else {
                    add_row16::<T, N, BD>(d, r16);
                }
            }
        });
    }
}

/// Adds a constant residual to a `W x H` block with clipping (`N <= W`).
#[inline(always)]
fn add_dc_bd<T: Pixel, const N: usize, const W: usize, const H: usize, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    dc: i32,
) {
    let dc16 = dc.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    if W == 4 && H >= 4 {
        let r8 = Simd::<i16, 8>::splat(dc16);
        visit_row_pairs_4::<T, H>(dst, stride, |da, db| add_rows2_4_sat::<T, BD>(da, db, r8));
        return;
    }
    let r16 = Simd::<i16, N>::splat(dc16);
    visit_rows::<T, W, H>(dst, stride, |_, row| {
        for x in (0..W / N).map(|k| k * N) {
            let d: &mut [T; N] = (&mut row[x..x + N]).try_into().unwrap();
            add_row_sat16::<T, N, BD>(d, r16);
        }
    });
}

/// Inverse Walsh-Hadamard transform (7.13.2.10) of 4 vectors.
#[inline(always)]
fn iwht4<const SHIFT: i32>(t: [i32x4; 4]) -> [i32x4; 4] {
    let mut a = if SHIFT == 0 { t[0] } else { t[0] >> SHIFT };
    let mut c = if SHIFT == 0 { t[1] } else { t[1] >> SHIFT };
    let mut d = if SHIFT == 0 { t[2] } else { t[2] >> SHIFT };
    let mut b = if SHIFT == 0 { t[3] } else { t[3] >> SHIFT };
    a += c;
    d -= b;
    let e = (a - d) >> 1;
    b = e - b;
    c = e - c;
    a -= b;
    d += c;
    [a, b, c, d]
}

/// Lossless 4x4 Walsh-Hadamard reconstruction (7.13.3 with Lossless = 1).
/// Leaves `coeffs[..16]` zero.
#[inline(always)]
fn inv_wht4x4_add_bd<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    coeffs: &mut [i32],
    tx_type: TxType,
) {
    let col_range = const {
        if BD + 6 > 16 {
            BD + 6
        } else {
            16
        }
    };
    let clamp = Clamp::<i32x4>::bits(col_range);
    let c: &mut [i32; 16] = (&mut coeffs[..16]).try_into().unwrap();
    let rows = [
        i32x4::from_slice(&c[..4]),
        i32x4::from_slice(&c[4..8]),
        i32x4::from_slice(&c[8..12]),
        i32x4::from_slice(&c[12..16]),
    ];
    *c = [0; 16];
    // Row transforms with one row per lane: o[j] lane i = Residual[i][j].
    let o = amap(iwht4::<2>(transpose(rows)), |v| clamp.apply(v));
    // Column transforms with one column per lane: r[i] lane j.
    let r = iwht4::<0>(transpose(o));
    if tx_type == TxType::DCT_DCT || (!flip_ud(tx_type) && !flip_lr(tx_type)) {
        let r01 = simd_swizzle!(sat16(r[0]), sat16(r[1]), [0, 1, 2, 3, 4, 5, 6, 7]);
        let r23 = simd_swizzle!(sat16(r[2]), sat16(r[3]), [0, 1, 2, 3, 4, 5, 6, 7]);
        let stride = stride & 0xffff_ffff;
        assert!(stride >= 4);
        let dst = &mut dst[..3 * stride + 4];
        let (top, bot) = dst.split_at_mut(2 * stride);
        let (d0, d1) = top.split_at_mut(stride);
        let (d2, d3) = bot.split_at_mut(stride);
        let (Some(d0), Some(d1), Some(d2), Some(d3)) = (
            d0.first_chunk_mut::<4>(),
            d1.first_chunk_mut::<4>(),
            d2.first_chunk_mut::<4>(),
            d3.first_chunk_mut::<4>(),
        ) else {
            return;
        };
        add_rows2_4_sat::<T, BD>(d0, d1, r01);
        add_rows2_4_sat::<T, BD>(d2, d3, r23);
        return;
    }
    let (fud, flr) = (flip_ud(tx_type), flip_lr(tx_type));
    visit_rows::<T, 4, 4>(dst, stride, |y, d| {
        let v = r[if fud { 3 - y } else { y }];
        add_row_sat16::<T, 4, BD>(d, sat16(if flr { v.reverse() } else { v }));
    });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::av1::dsp::itx as portable;

    /// Small deterministic PRNG (xorshift64*).
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        /// Uniform in `[lo, hi]`.
        fn range(&mut self, lo: i64, hi: i64) -> i64 {
            lo + (self.next() % ((hi - lo + 1) as u64)) as i64
        }
    }

    #[test]
    fn transpose_works() {
        let m4: [i32x4; 4] =
            std::array::from_fn(|i| i32x4::from_array(std::array::from_fn(|j| (i * 4 + j) as i32)));
        let t4 = transpose(m4);
        let m8: [i32x8; 8] =
            std::array::from_fn(|i| i32x8::from_array(std::array::from_fn(|j| (i * 8 + j) as i32)));
        let t8 = transpose(m8);
        for i in 0..8 {
            for j in 0..8 {
                assert_eq!(t8[j][i], m8[i][j]);
                if i < 4 && j < 4 {
                    assert_eq!(t4[j][i], m4[i][j]);
                }
            }
        }
    }

    /// `Wide` products equal 64-bit products for values of up to 22 bits.
    #[test]
    fn wide_dot_exact() {
        let mut rng = Rng(77);
        let lim = 1i64 << 21;
        for _ in 0..20000 {
            let v = |rng: &mut Rng| match rng.range(0, 4) {
                0 => -lim,
                1 => lim - 1,
                2 => rng.range(-300, 300),
                _ => rng.range(-lim, lim - 1),
            };
            let a: [i64; 4] = std::array::from_fn(|_| v(&mut rng));
            let k: [i64; 4] = std::array::from_fn(|_| rng.range(-4096, 4096));
            let k = if rng.range(0, 1) == 0 { k } else { [1321, 3344, 3803, 2482] };
            let want4 = (a.iter().zip(&k).map(|(x, y)| x * y).sum::<i64>() + 2048) >> 12;
            let want2 = (a[0] * k[0] + a[1] * k[1] + 2048) >> 12;
            let w = |x: i64| Wide::<4>(Simd::splat(x as i32));
            let got4 = Wide::dot_rnd12(a.map(w), k.map(|x| x as i32)).0[0];
            let got2 = w(a[0]).m2(k[0] as i32, w(a[1]), k[1] as i32).0[0];
            assert_eq!(got4 as i64, want4, "{a:?} {k:?}");
            assert_eq!(got2 as i64, want2, "{a:?} {k:?}");
        }
    }

    /// Transform types whose 1D components are defined for this size
    /// (the spec only allows a subset for 32 and 64 point sizes).
    fn types_for(tx_size: TxSize) -> Vec<TxType> {
        let w = 1 << TX_WIDTH_LOG2[tx_size.idx()];
        let h = 1 << TX_HEIGHT_LOG2[tx_size.idx()];
        TxType::ALL
            .iter()
            .copied()
            .filter(|&t| {
                let (c, r) = tx1d_kinds(t);
                let ok = |k: Tx1d, n: usize| match k {
                    Tx1d::Dct => true,
                    Tx1d::Adst => n <= 16,
                    Tx1d::Identity => n <= 32,
                };
                ok(c, h) && ok(r, w)
            })
            .collect()
    }

    /// Coefficient blocks: 0 DC only, 1 sparse (in a sub-block tier),
    /// 2 dense, 3 full Dequant range (non-conformant), 4 zero, 5 sparse
    /// at the range limits, 6 first row / column only, 7 rows at the
    /// 10-bit 16-bit row lane limit.
    fn gen_coeffs(rng: &mut Rng, dist: u32, tw: usize, th: usize, bd: u32) -> (Vec<i32>, usize) {
        let max = (1i64 << (7 + bd)) - 1;
        let min = -(1i64 << (7 + bd));
        let mut c = vec![0i32; tw * th];
        let mut eob = tw * th;
        let extreme = |rng: &mut Rng| match rng.range(0, 3) {
            0 => min,
            1 => max,
            _ => rng.range(min, max),
        } as i32;
        match dist {
            0 => {
                c[0] = rng.range(min, max) as i32;
                eob = 1;
            }
            1 => {
                // Exercise the sub-block eob tiers (4x4, 8x8, 16x16, full).
                let (sub, tier_eob) = match rng.range(0, 3) {
                    0 => (4usize, 4usize.min(tw * th)),
                    1 => (8, 36.min(tw * th)),
                    2 => (16, 136.min(tw * th)),
                    _ => (32, tw * th),
                };
                let (mr, mc) = (th.min(sub), tw.min(sub));
                // For MROW / MCOL scans, ensure eob covers rows < mr and cols < mc.
                eob = tier_eob
                    .max((mr - 1) * tw + mc)
                    .max((mc - 1) * th + mr)
                    .min(tw * th);
                let n = rng.range(1, 8);
                let amp = 1i64 << (bd + rng.range(0, 6) as u32);
                for _ in 0..n {
                    let i = (rng.range(0, mr as i64 - 1) * rng.range(0, 3) / 3) as usize;
                    let j = (rng.range(0, mc as i64 - 1) * rng.range(0, 3) / 3) as usize;
                    c[i * tw + j] = rng.range(-amp, amp).clamp(min, max) as i32;
                }
            }
            2 => {
                let amp = 1i64 << (bd as i64 + rng.range(-2, 3)) as u32;
                for v in &mut c {
                    *v = rng.range(-amp, amp) as i32;
                }
            }
            3 => {
                for v in &mut c {
                    *v = extreme(rng);
                }
            }
            4 => {
                eob = match rng.range(0, 2) {
                    0 => 2,
                    1 => 10.min(tw * th),
                    _ => tw * th,
                };
            }
            5 => {
                for _ in 0..rng.range(1, 6) {
                    let i = rng.range(0, th as i64 - 1) as usize;
                    let j = rng.range(0, tw as i64 - 1) as usize;
                    c[i * tw + j] = extreme(rng);
                }
            }
            6 => {
                let row = rng.range(0, 1) == 0;
                for k in 0..if row { tw } else { th } {
                    if rng.range(0, 1) == 0 {
                        let idx = if row { k } else { k * tw };
                        c[idx] = rng.range(min / 8, max / 8) as i32;
                    }
                }
            }
            _ => {
                // One or two rows whose absolute sums are at (or just
                // below) the limit of the 16-bit row lanes of 10-bit
                // streams (`large::S16_ROW_L1_MAX`).
                for _ in 0..rng.range(1, 2) {
                    let i = rng.range(0, th as i64 - 1) as usize;
                    let mut left = (large::S16_ROW_L1_MAX as i64
                        - rng.range(0, 2) * rng.range(0, 64))
                    .min(max);
                    let k = rng.range(1, tw.min(8) as i64);
                    for n in 0..k {
                        let j = rng.range(0, tw as i64 - 1) as usize;
                        let a = if n == k - 1 { left } else { rng.range(0, left) };
                        left -= a;
                        let v = c[i * tw + j] as i64 + if rng.range(0, 1) == 0 { a } else { -a };
                        c[i * tw + j] = v.clamp(min, max) as i32;
                    }
                }
            }
        }
        (c, eob)
    }

    /// 10-bit rows at the limit of the 16-bit row lanes match the
    /// portable transforms.
    #[test]
    fn s16_row_limit_10bit() {
        let mut rng = Rng(0x0123_4567_89ab_cdef);
        let (mut ok, mut n) = (0, 0);
        for &tx_size in TxSize::ALL {
            for tx_type in types_for(tx_size) {
                for _ in 0..24 {
                    ok += run_case::<u16, 10>(&mut rng, tx_size, tx_type, false, 7, 10) as usize;
                    n += 1;
                }
            }
        }
        assert!(ok as f64 >= n as f64 * 0.5, "{ok}/{n} conformant");
    }

    #[test]
    fn eob_bounds_cover_all_scans() {
        for &tx_size in TxSize::ALL {
            let w = 1usize << TX_WIDTH_LOG2[tx_size.idx()];
            let h = 1usize << TX_HEIGHT_LOG2[tx_size.idx()];
            assert_eq!(row_shift(w, h), TRANSFORM_ROW_SHIFT[tx_size.idx()] as u32);
            assert_eq!(rect2(w, h), w.ilog2().abs_diff(h.ilog2()) == 1);
            let (tw, th) = (w.min(32), h.min(32));
            for tx_type in types_for(tx_size) {
                let (col_kind, row_kind) = tx1d_kinds(tx_type);
                if (row_kind == Tx1d::Identity) != (col_kind == Tx1d::Identity) && tw.max(th) > 16 {
                    continue;
                }
                let scan = crate::decoder::av1::decode::block::get_scan(tx_size, tx_type);
                assert_eq!(scan.len(), tw * th);
                for eob in 1..=tw * th {
                    let (er, ec) = eob_bounds(eob, tw, th, row_kind, col_kind);
                    for &pos in &scan[..eob] {
                        let r = pos as usize / tw;
                        let c = pos as usize % tw;
                        assert!(
                            r < er && c < ec,
                            "{tx_size:?} {tx_type:?} eob={eob}: ({r},{c}) not in ({er},{ec})"
                        );
                    }
                }
            }
        }
    }

    fn run_case<T: Pixel, const BD: u32>(
        rng: &mut Rng,
        tx_size: TxSize,
        tx_type: TxType,
        lossless: bool,
        dist: u32,
        bd: u32,
    ) -> bool {
        let w = 1usize << TX_WIDTH_LOG2[tx_size.idx()];
        let h = 1usize << TX_HEIGHT_LOG2[tx_size.idx()];
        let (tw, th) = (w.min(32), h.min(32));
        let (coeffs, eob) = gen_coeffs(rng, dist, tw, th, bd);
        let stride = w + rng.range(0, 5) as usize;
        let pix_max = (1i64 << bd) - 1;
        let frame: Vec<T> = (0..stride * h)
            .map(|_| T::from_i32(rng.range(0, pix_max) as i32))
            .collect();
        let mut want = frame.clone();
        portable::conformance::take_violated();
        let mut c_want = coeffs.clone();
        portable::inv_txfm_add(
            &mut want,
            stride,
            &mut c_want,
            tx_size,
            tx_type,
            lossless,
            eob,
            bd,
        );
        let mut got = frame;
        let mut c_got = coeffs.clone();
        inv_txfm_add_bd::<T, BD>(
            &mut got, stride, &mut c_got, tx_size, tx_type, lossless, eob, bd,
        );
        // The 16-bit intermediates of bd <= 10 (`S16`) only match the spec
        // for conformant inputs. Non-conformant ones must not panic.
        let conformant = !portable::conformance::take_violated();
        if bd > 10 || conformant {
            assert!(
                got == want,
                "mismatch: {tx_size:?} {tx_type:?} lossless={lossless} dist {dist} bd={bd}"
            );
        }
        // Both leave the coefficients zero.
        assert!(c_want.iter().chain(&c_got).all(|&c| c == 0));
        conformant
    }

    fn run_all<T: Pixel, const BD: u32>(bd: u32, seed: u64) {
        let mut rng = Rng(seed);
        // Cases compared / run, per coefficient distribution.
        let mut stats = [(0usize, 0usize); 7];
        for &tx_size in TxSize::ALL {
            let w = 1usize << TX_WIDTH_LOG2[tx_size.idx()];
            let h = 1usize << TX_HEIGHT_LOG2[tx_size.idx()];
            let iters = if w * h >= 2048 { 4 } else { 12 };
            for tx_type in types_for(tx_size) {
                for dist in 0..7 {
                    for _ in 0..iters {
                        let ok = run_case::<T, BD>(&mut rng, tx_size, tx_type, false, dist, bd);
                        let s = &mut stats[dist as usize];
                        s.0 += ok as usize;
                        s.1 += 1;
                    }
                }
            }
        }
        for &tx_type in TxType::ALL {
            for dist in 0..7 {
                for _ in 0..20 {
                    run_case::<T, BD>(&mut rng, TxSize::Tx4x4, tx_type, true, dist, bd);
                }
            }
        }
        // Only the extreme value distributions (3: full range, 5: sparse
        // at the limits) are often non-conformant.
        for (dist, &(ok, n)) in stats.iter().enumerate() {
            let min = match dist {
                3 => 0.0,
                5 => 0.5,
                _ => 0.9,
            };
            assert!(
                bd > 10 || ok as f64 >= n as f64 * min,
                "dist {dist}: {ok}/{n} conformant"
            );
        }
    }

    /// Lossless 8-bit blocks whose residual overflows 16 bits.
    #[test]
    fn lossless_large_residual_matches_portable() {
        fn run<T: Pixel>() {
            let coeffs: Vec<i32> = (0..16)
                .map(|i| if i % 4 < 2 { 32767 } else { -32768 })
                .collect();
            for px in [0, 2, 128, 255] {
                let frame = vec![T::from_i32(px); 4 * 4];
                let mut want = frame.clone();
                let mut c_want = coeffs.clone();
                portable::inv_txfm_add(
                    &mut want,
                    4,
                    &mut c_want,
                    TxSize::Tx4x4,
                    TxType::DCT_DCT,
                    true,
                    16,
                    8,
                );
                let mut got = frame;
                let mut c_got = coeffs.clone();
                inv_txfm_add_bd::<T, 8>(
                    &mut got,
                    4,
                    &mut c_got,
                    TxSize::Tx4x4,
                    TxType::DCT_DCT,
                    true,
                    16,
                    8,
                );
                assert!(got == want, "pixel {px}");
            }
        }
        run::<u8>();
        run::<u16>();
    }

    #[test]
    fn matches_portable_8bit() {
        run_all::<u8, 8>(8, 0x1234_5678_9abc_def1);
    }

    #[test]
    fn matches_portable_10bit() {
        run_all::<u16, 10>(10, 0x0fed_cba9_8765_4321);
    }

    #[test]
    fn matches_portable_12bit() {
        run_all::<u16, 12>(12, 0x5555_aaaa_3333_cccc);
    }

    #[test]
    fn matches_portable_8bit_in_u16() {
        run_all::<u16, 8>(8, 0x7777_1111_2222_9999);
    }

    /// The aarch64 4x4 path (run on every architecture) matches the
    /// portable transforms for the DCT / ADST types.
    #[test]
    fn neon_4x4_matches_portable() {
        fn run<T: Pixel, const BD: u32>(seed: u64) {
            let mut rng = Rng(seed);
            for tx_type in [
                TxType::DCT_DCT,
                TxType::ADST_DCT,
                TxType::DCT_ADST,
                TxType::ADST_ADST,
            ] {
                let (col_kind, row_kind) = tx1d_kinds(tx_type);
                for dist in 0..8 {
                    for _ in 0..200 {
                        let (coeffs, eob) = gen_coeffs(&mut rng, dist, 4, 4, BD);
                        let stride = 4 + rng.range(0, 5) as usize;
                        let pix_max = (1i64 << BD) - 1;
                        let frame: Vec<T> = (0..stride * 4)
                            .map(|_| T::from_i32(rng.range(0, pix_max) as i32))
                            .collect();
                        let mut want = frame.clone();
                        portable::conformance::take_violated();
                        let mut c_want = coeffs.clone();
                        portable::inv_txfm_add(
                            &mut want,
                            stride,
                            &mut c_want,
                            TxSize::Tx4x4,
                            tx_type,
                            false,
                            eob,
                            BD,
                        );
                        let conformant = !portable::conformance::take_violated();
                        let mut got = frame;
                        let mut c_got = coeffs.clone();
                        transform_add_4x4_neon_impl::<T, BD>(
                            &mut got, stride, &mut c_got, row_kind, col_kind,
                        );
                        if conformant {
                            assert!(got == want, "mismatch: {tx_type:?} dist {dist} bd={BD}");
                        }
                        assert!(c_got.iter().all(|&c| c == 0));
                    }
                }
            }
        }
        run::<u8, 8>(0x1357_9bdf_2468_ace0);
        run::<u16, 10>(0x0246_8ace_1357_9bdf);
    }

    /// Conformant `ADST4` inputs whose unrounded 4-term sum `x` lies in
    /// `[2^27 - 2048, 2^27 - 1]` satisfy the `(r + 12)`-bit bound of spec
    /// 7.13.2.6 while producing `Round2(x, 12) == 32768`, which must
    /// saturate to `32767` rather than wrap to `-32768` on Arm.
    #[test]
    fn adst4_round2_32768_matches_portable() {
        fn run<T: Pixel, const BD: u32>() {
            // Row [1952, 872, 32767, 1656]: x0 = 134_217_653 < 2^27, Round2(x0, 12) = 32768.
            // Row [32767, -1230, 1230, -1845]: x3 = 134_216_126 < 2^27, Round2(x3, 12) = 32768.
            for row in [[1952, 872, 32767, 1656], [32767, -1230, 1230, -1845]] {
                for tx_size in [TxSize::Tx4x4, TxSize::Tx4x8, TxSize::Tx4x16] {
                    let w = 1usize << TX_WIDTH_LOG2[tx_size.idx()];
                    let h = 1usize << TX_HEIGHT_LOG2[tx_size.idx()];
                    for tx_type in [
                        TxType::ADST_ADST,
                        TxType::DCT_ADST,
                        TxType::ADST_DCT,
                        TxType::FLIPADST_ADST,
                    ] {
                        let mut coeffs = vec![0i32; w * h];
                        coeffs[..4].copy_from_slice(&row);
                        let eob = w * h;
                        let frame = vec![T::from_i32(1 << (BD - 1)); w * h];
                        let mut want = frame.clone();
                        portable::conformance::take_violated();
                        let mut c_want = coeffs.clone();
                        portable::inv_txfm_add(
                            &mut want,
                            w,
                            &mut c_want,
                            tx_size,
                            tx_type,
                            false,
                            eob,
                            BD,
                        );
                        assert!(!portable::conformance::take_violated());
                        let mut got = frame.clone();
                        let mut c_got = coeffs.clone();
                        inv_txfm_add_bd::<T, BD>(
                            &mut got, w, &mut c_got, tx_size, tx_type, false, eob, BD,
                        );
                        assert_eq!(got, want, "mismatch: {tx_size:?} {tx_type:?} bd={BD}");
                        if tx_size == TxSize::Tx4x4 && !flip_lr(tx_type) && !flip_ud(tx_type) {
                            let (col_kind, row_kind) = tx1d_kinds(tx_type);
                            let mut got_neon = frame;
                            let mut c_neon = coeffs;
                            transform_add_4x4_neon_impl::<T, BD>(
                                &mut got_neon,
                                w,
                                &mut c_neon,
                                row_kind,
                                col_kind,
                            );
                            assert_eq!(
                                got_neon, want,
                                "neon mismatch: {tx_size:?} {tx_type:?} bd={BD}"
                            );
                        }
                    }
                }
            }
        }
        run::<u8, 8>();
        run::<u16, 10>();
    }

    /// Sparse blocks with `nz_rows` and `nz_cols` in `1..=4` exercise the
    /// `_nz1` and `_nz2` kernel tiers across every `(TxSize, TxType)` and
    /// bit depth.
    #[test]
    fn sparse_nz1_nz2_matches_portable() {
        fn run<T: Pixel, const BD: u32>(seed: u64) {
            let mut rng = Rng(seed);
            for &tx_size in TxSize::ALL {
                let w = 1usize << TX_WIDTH_LOG2[tx_size.idx()];
                let h = 1usize << TX_HEIGHT_LOG2[tx_size.idx()];
                let (tw, th) = (w.min(32), h.min(32));
                for tx_type in types_for(tx_size) {
                    for mr in [1usize, 2, 3, 4] {
                        for mc in [1usize, 2, 3, 4] {
                            for eob_tier in [1usize, 2, 3, 4, 16, tw * th] {
                                let eob = eob_tier
                                    .max((mr - 1) * tw + mc)
                                    .max((mc - 1) * th + mr)
                                    .min(tw * th);
                                let amp = 1i64 << BD;
                                let mut coeffs = vec![0i32; tw * th];
                                for r in 0..mr {
                                    for c in 0..mc {
                                        coeffs[r * tw + c] = rng.range(-amp, amp) as i32;
                                    }
                                }
                                if coeffs[..mr * tw].iter().all(|&x| x == 0) {
                                    coeffs[(mr - 1) * tw + (mc - 1)] = 17;
                                }
                                let stride = w + rng.range(0, 3) as usize;
                                let pix_max = (1i64 << BD) - 1;
                                let frame: Vec<T> = (0..stride * h)
                                    .map(|_| T::from_i32(rng.range(0, pix_max) as i32))
                                    .collect();
                                let mut want = frame.clone();
                                portable::conformance::take_violated();
                                let mut c_want = coeffs.clone();
                                portable::inv_txfm_add(
                                    &mut want,
                                    stride,
                                    &mut c_want,
                                    tx_size,
                                    tx_type,
                                    false,
                                    eob,
                                    BD,
                                );
                                let conformant = !portable::conformance::take_violated();
                                let mut got = frame;
                                let mut c_got = coeffs;
                                inv_txfm_add_bd::<T, BD>(
                                    &mut got, stride, &mut c_got, tx_size, tx_type, false, eob, BD,
                                );
                                if BD > 10 || conformant {
                                    assert!(
                                        got == want,
                                        "mismatch: {tx_size:?} {tx_type:?} mr={mr} mc={mc} eob={eob} bd={BD}"
                                    );
                                }
                                assert!(c_got.iter().all(|&c| c == 0));
                            }
                        }
                    }
                }
            }
        }
        run::<u8, 8>(0x2468_ace0_1357_9bdf);
        run::<u16, 10>(0x1357_9bdf_0246_8ace);
        run::<u16, 12>(0x9bdf_1357_ace0_2468);
    }

    /// Multi-row-group MCOL scans (`eob_cols` in `9..=12`, so `col_tiles * 4 = 12 < 16`)
    /// with large 12-bit rows that take the `Wide` in-place row path.
    #[test]
    fn mcol_partial_col_tiles_12bit_matches_portable() {
        let mut rng = Rng(0xc01_711e_12b1_7000);
        for &tx_size in &[TxSize::Tx16x8, TxSize::Tx16x16] {
            let w = 1usize << TX_WIDTH_LOG2[tx_size.idx()];
            let h = 1usize << TX_HEIGHT_LOG2[tx_size.idx()];
            for tx_type in [TxType::H_DCT, TxType::H_ADST, TxType::H_FLIPADST] {
                for mc in 9..=12 {
                    let eob = (mc - 1) * h + h;
                    let mut coeffs = vec![0i32; w * h];
                    for r in 0..h {
                        for c in 0..mc {
                            coeffs[r * w + c] = rng.range(-300_000, 300_000) as i32;
                        }
                    }
                    let frame: Vec<u16> = (0..w * h).map(|_| rng.range(0, 4095) as u16).collect();
                    let mut want = frame.clone();
                    let mut c_want = coeffs.clone();
                    portable::inv_txfm_add(
                        &mut want,
                        w,
                        &mut c_want,
                        tx_size,
                        tx_type,
                        false,
                        eob,
                        12,
                    );
                    let mut got = frame;
                    let mut c_got = coeffs;
                    inv_txfm_add_bd::<u16, 12>(
                        &mut got, w, &mut c_got, tx_size, tx_type, false, eob, 12,
                    );
                    assert!(got == want, "mismatch: {tx_size:?} {tx_type:?} mc={mc}");
                    assert!(c_got.iter().all(|&c| c == 0));
                }
            }
        }
    }

    /// Rough timing (`cargo test --release --lib
    /// bench_simd_itx -- --ignored --nocapture`).
    #[test]
    #[ignore]
    fn bench_simd_itx() {
        fn run<T: Pixel, const BD: u32>(bd: u32) {
            use TxSize::*;
            use TxType::*;
            let mut rng = Rng(9);
            let cases = [
                (Tx4x4, DCT_DCT),
                (Tx4x4, ADST_ADST),
                (Tx4x4, IDTX),
                (Tx8x8, DCT_DCT),
                (Tx8x8, ADST_DCT),
                (Tx4x8, DCT_DCT),
                (Tx8x16, DCT_DCT),
                (Tx16x16, DCT_DCT),
                (Tx16x16, ADST_ADST),
                (Tx32x32, DCT_DCT),
                (Tx64x64, DCT_DCT),
            ];
            let mut dst = vec![T::from_i32(1 << (bd - 1)); 64 * 64];
            let mut total = 0.0;
            // ITX_BENCH_ONLY=<substring of "{tx_size:?} {tx_type:?}">.
            let only = std::env::var("ITX_BENCH_ONLY").unwrap_or_default();
            for (tx_size, tx_type) in cases {
                if !format!("{tx_size:?} {tx_type:?}").contains(&only) {
                    continue;
                }
                let w = 1usize << TX_WIDTH_LOG2[tx_size.idx()];
                let h = 1usize << TX_HEIGHT_LOG2[tx_size.idx()];
                let (tw, th) = (w.min(32), h.min(32));
                for dist in [1, 2] {
                    let blocks: Vec<(Vec<i32>, usize)> = (0..64)
                        .map(|_| gen_coeffs(&mut rng, dist, tw, th, bd))
                        .collect();
                    let mut c = vec![0i32; tw * th];
                    let iters = 4_000_000 / (w * h) * if only.is_empty() { 1 } else { 20 };
                    let start = std::time::Instant::now();
                    for k in 0..iters {
                        let (ref blk, eob) = blocks[k % 64];
                        c.copy_from_slice(blk);
                        inv_txfm_add_bd::<T, BD>(
                            &mut dst, 64, &mut c, tx_size, tx_type, false, eob, bd,
                        );
                    }
                    let ns = start.elapsed().as_nanos() as f64 / iters as f64;
                    total += ns / (w * h) as f64;
                    eprintln!("bd{bd} {tx_size:?} {tx_type:?} dist {dist}: {ns:.0} ns/block");
                }
            }
            eprintln!("bd{bd} sum of ns/pixel: {total:.2}");
        }
        run::<u8, 8>(8);
        run::<u16, 10>(10);
    }
}
