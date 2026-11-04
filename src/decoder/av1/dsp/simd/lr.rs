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

//! SIMD loop restoration filters (see `dsp::lr` for the contracts).
//!
//! The kernels process `N` columns at a time. Samples, Wiener
//! intermediates and the self guided `A` weights fit in `i16` lanes;
//! products are formed as `i16 x i16 -> i32` (`madd`, which maps to
//! `pmaddwd` on x86) and accumulated in `i32` lanes.
//!
//! Row buffers are sized to a multiple of `N` so that only loads from
//! the source (and stores to the destination) need tail handling: the
//! last chunk of a row loads past-the-end source lanes as 0 and only
//! stores its first `w % N` lanes. Lanes past the end of a row are
//! computed but never used.

use std::simd::prelude::*;

use crate::decoder::av1::dsp::lr::{
    check_rect, one_over_n, wiener_coefficients, wiener_rounding, FILTER_BITS, LR_BORDER, LR_MAX_W,
    SGRPROJ_MTABLE_BITS, SGRPROJ_PRJ_BITS, SGRPROJ_RECIP_BITS, SGRPROJ_RST_BITS, SGRPROJ_SGR_BITS,
    SGR_PARAMS, SGR_SCALE,
};
use crate::decoder::av1::pixel::Pixel;

/// Number of lanes (columns) processed at once.
const N: usize = 8;
type I16s = Simd<i16, N>;
type I32s = Simd<i32, N>;
type U32s = Simd<u32, N>;

/// Lane indices interleaving two `N` lane vectors.
#[cfg(not(target_arch = "aarch64"))]
const INTERLEAVE: [usize; 2 * N] = {
    let mut t = [0; 2 * N];
    let mut i = 0;
    while i < 2 * N {
        t[i] = i / 2 + (i % 2) * N;
        i += 1;
    }
    t
};
/// Even (`EVEN[i] = 2 * i`) and odd lane indices of a `2 * N` vector.
#[cfg(not(target_arch = "aarch64"))]
const EVEN: [usize; N] = {
    let mut t = [0; N];
    let mut i = 0;
    while i < N {
        t[i] = 2 * i;
        i += 1;
    }
    t
};
#[cfg(not(target_arch = "aarch64"))]
const ODD: [usize; N] = {
    let mut t = [0; N];
    let mut i = 0;
    while i < N {
        t[i] = 2 * i + 1;
        i += 1;
    }
    t
};

/// `a * fa + b * fb` per lane, with exact `i32` products and sum.
///
/// Written as interleave / widen / multiply / pairwise add, which LLVM
/// lowers to `pmaddwd` on x86.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn madd(a: I16s, b: I16s, fa: I16s, fb: I16s) -> I32s {
    let x: Simd<i16, { 2 * N }> = simd_swizzle!(a, b, INTERLEAVE);
    let f: Simd<i16, { 2 * N }> = simd_swizzle!(fa, fb, INTERLEAVE);
    let p = x.cast::<i32>() * f.cast::<i32>();
    let even: I32s = simd_swizzle!(p, EVEN);
    let odd: I32s = simd_swizzle!(p, ODD);
    even + odd
}

/// `a * fa + b * fb` per lane, with exact `i32` products and sum.
///
/// On aarch64 there is no `pmaddwd`; two widening multiplies lower to
/// `smull` / `smlal` (the interleave form costs extra `zip` / `uzp`).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn madd(a: I16s, b: I16s, fa: I16s, fb: I16s) -> I32s {
    mul_wide(a, fa) + mul_wide(b, fb)
}

/// `a * b` per lane, with exact `i32` products.
#[inline(always)]
fn mul_wide(a: I16s, b: I16s) -> I32s {
    a.cast::<i32>() * b.cast::<i32>()
}

/// Loads `N` pixels from `src[..N]` into `i16` lanes.
#[inline(always)]
fn load_px<T: Pixel>(src: &[T]) -> I16s {
    if T::MAX_BITDEPTH == 8 {
        let mut a = [0u8; N];
        for (d, s) in a.iter_mut().zip(&src[..N]) {
            *d = s.to_i32() as u8;
        }
        Simd::<u8, N>::from_array(a).cast::<i16>()
    } else {
        let mut a = [0i16; N];
        for (d, s) in a.iter_mut().zip(&src[..N]) {
            *d = s.to_i32() as i16;
        }
        I16s::from_array(a)
    }
}

/// Like [`load_px`], but lanes past the end of `src` are 0.
#[inline(never)]
fn load_px_or_zero<T: Pixel>(src: &[T]) -> I16s {
    let mut a = [0i16; N];
    for (d, s) in a.iter_mut().zip(src) {
        *d = s.to_i32() as i16;
    }
    I16s::from_array(a)
}

/// Stores `N` (`8`) clamped pixel values `v` to `dst[..N]`.
#[inline(always)]
fn store_px8<T: Pixel>(dst: &mut [T], v: I16s) {
    for (d, s) in dst[..N].iter_mut().zip(v.to_array()) {
        *d = T::from_i32(s as i32);
    }
}

/// Like [`store_px8`], for a tail slice `dst` shorter than `N`.
#[inline(never)]
fn store_px8_tail<T: Pixel>(dst: &mut [T], v: I16s) {
    for (d, s) in dst.iter_mut().zip(v.to_array()) {
        *d = T::from_i32(s as i32);
    }
}

/// Number of box sum columns to compute when `ncols` are needed from
/// columns `col0..` of the padded source rows `0..h + 2 * LR_BORDER`:
/// `ncols` rounded up to a multiple of 8 (`N`) if every row is long
/// enough for full-chunk SIMD loads (see
/// [`crate::decoder::av1::dsp::lr::LR_SRC_SLACK`]), so that the scalar
/// tail loads are skipped. The extra columns only feed lanes that are
/// never used.
#[inline(always)]
fn box_cols<T>(src: &[&[T]], h: usize, col0: usize, ncols: usize) -> usize {
    let full = ncols.next_multiple_of(N);
    if src[..h + 2 * LR_BORDER]
        .iter()
        .all(|r| r.len() >= col0 + full)
    {
        full
    } else {
        ncols
    }
}

/// `v` narrowed to `i16` lanes with signed saturation (`packssdw`).
#[inline(always)]
fn narrow_sat(v: I32s) -> I16s {
    v.simd_max(I32s::splat(i16::MIN as i32))
        .simd_min(I32s::splat(i16::MAX as i32))
        .cast::<i16>()
}

/// `Clip3(0, max, v)` as `i16` lanes, for `max < 2^15`. Saturating to
/// `i16` first does not change the result.
#[inline(always)]
fn clip_px(v: I32s, max: I16s) -> I16s {
    narrow_sat(v).simd_max(I16s::splat(0)).simd_min(max)
}

/// Length of a widened source row of the Wiener filter: `w + 6`
/// samples plus room for the full chunk loads of the last chunk.
const WIDE_LEN: usize = LR_MAX_W + 2 * N;

/// Widens `src` to `i16` into `dst[..src.len()]`.
#[inline(always)]
fn widen_row<T: Pixel>(src: &[T], dst: &mut [i16]) {
    let mut c = 0;
    while c + N <= src.len() {
        load_px(&src[c..]).copy_to_slice(&mut dst[c..c + N]);
        c += N;
    }
    for (d, s) in dst[c..].iter_mut().zip(&src[c..]) {
        *d = s.to_i32() as i16;
    }
}

/// Length of a raw `A`/`B` row: rectangle columns `-1..=w` rounded up
/// to whole chunks (this also covers the 3-wide neighbourhood loads of
/// the last output chunk, which end at `LR_MAX_W + 2`).
const AB_LEN: usize = (LR_MAX_W + 2).next_multiple_of(N);
/// Length of a box sum row: `AB_LEN` plus one chunk of read slack.
const SUM_LEN: usize = AB_LEN + N;

/// One row of horizontally weighted `A`/`B` values for rectangle columns
/// `0..w`, `x[j] = k0 * (raw[j] + raw[j + 2]) + k1 * raw[j + 1]`:
/// 5, 6 (r = 2 pass), 3, 4 or 4, 4 (r = 1 pass). All `A` sums are at
/// most `32 * 256`.
struct WeightedRow {
    a: [i16; LR_MAX_W],
    b: [i32; LR_MAX_W],
}

impl WeightedRow {
    const fn new() -> Self {
        Self {
            a: [0; LR_MAX_W],
            b: [0; LR_MAX_W],
        }
    }
}

/// Vertical box sums (`sum` <= `5 * 4095 = 20475` in `u16`, `sq` in `i32`).
struct WideBoxSums {
    sum: [u16; SUM_LEN],
    sq: [i32; SUM_LEN],
}

impl WideBoxSums {
    const fn new() -> Self {
        Self {
            sum: [0; SUM_LEN],
            sq: [0; SUM_LEN],
        }
    }
}

struct WideAbRow {
    a: [i16; AB_LEN],
    b: [i32; AB_LEN],
}

impl WideAbRow {
    const fn new() -> Self {
        Self {
            a: [0; AB_LEN],
            b: [0; AB_LEN],
        }
    }
}

/// Per-thread scratch buffers of the 8-lane (12-bit) Wiener and self guided
/// filters, allocated lazily once per thread so 12-bit restoration units do
/// not zero ~24 KB on the stack per unit.
struct WideScratch {
    wide: [i16; WIDE_LEN],
    ring: [[i16; LR_MAX_W]; 8],
    bs3: WideBoxSums,
    bs5: WideBoxSums,
    ab: WideAbRow,
    p0: [WeightedRow; 2],
    p343: [WeightedRow; 3],
}

impl WideScratch {
    #[cold]
    #[inline(never)]
    fn new_boxed() -> Box<Self> {
        Box::new(Self {
            wide: [0; WIDE_LEN],
            ring: [[0; LR_MAX_W]; 8],
            bs3: WideBoxSums::new(),
            bs5: WideBoxSums::new(),
            ab: WideAbRow::new(),
            p0: [WeightedRow::new(), WeightedRow::new()],
            p343: [WeightedRow::new(), WeightedRow::new(), WeightedRow::new()],
        })
    }
}

thread_local! {
    static WIDE_SCRATCH: std::cell::Cell<Option<Box<WideScratch>>> =
        const { std::cell::Cell::new(None) };
}

struct WideScratchGuard(Option<Box<WideScratch>>);

impl WideScratchGuard {
    #[inline(always)]
    fn take() -> Self {
        let s = WIDE_SCRATCH.try_with(std::cell::Cell::take).ok().flatten();
        Self(Some(s.unwrap_or_else(WideScratch::new_boxed)))
    }
}

impl std::ops::Deref for WideScratchGuard {
    type Target = WideScratch;
    #[inline(always)]
    fn deref(&self) -> &WideScratch {
        self.0.as_ref().unwrap()
    }
}

impl std::ops::DerefMut for WideScratchGuard {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut WideScratch {
        self.0.as_mut().unwrap()
    }
}

impl Drop for WideScratchGuard {
    #[inline(always)]
    fn drop(&mut self) {
        let s = self.0.take();
        let _ = WIDE_SCRATCH.try_with(|c| c.set(s));
    }
}

/// Wiener filter, same contract as [`crate::decoder::av1::dsp::lr::wiener_filter`].
///
/// The 7-tap filters are symmetric: the horizontal pass adds the sample
/// pairs (at most `2 * 4095`, fits `i16`) before multiplying; the
/// vertical pass multiplies both samples of a pair with one `madd`. The
/// horizontal pass output is clamped to
/// `[-offset, limit - offset] = [-2^(bd+3-round0), 3 * 2^(bd+3-round0) - 1]`
/// (at most `[-8192, 24575]`), so the ring of intermediate rows is `i16`.
#[allow(clippy::too_many_arguments)]
pub fn wiener_filter<T: Pixel>(
    dst: &mut [T],
    dst_stride: usize,
    src: &[&[T]],
    w: usize,
    h: usize,
    coeffs: &[[i32; 3]; 2],
    bd: u32,
) {
    check_rect(w, h);
    let vfilter = wiener_coefficients(&coeffs[0]);
    let hfilter = wiener_coefficients(&coeffs[1]);
    let (round0, round1) = wiener_rounding(bd);
    let offset = 1i32 << (bd + FILTER_BITS - round0 - 1);
    let limit = (1i32 << (bd + 1 + FILTER_BITS - round0)) - 1;
    let max_px = (1i32 << bd) - 1;

    // Taps are in [-23, 218].
    let hf: [I16s; 4] = [
        I16s::splat(hfilter[0] as i16),
        I16s::splat(hfilter[1] as i16),
        I16s::splat(hfilter[2] as i16),
        I16s::splat(hfilter[3] as i16),
    ];
    let vf: [I16s; 4] = [
        I16s::splat(vfilter[0] as i16),
        I16s::splat(vfilter[1] as i16),
        I16s::splat(vfilter[2] as i16),
        I16s::splat(vfilter[3] as i16),
    ];
    let zero = I16s::splat(0);
    let rnd0 = I32s::splat(1 << (round0 - 1));
    let lo0 = I16s::splat(-offset as i16);
    let hi0 = I16s::splat((limit - offset) as i16);
    let rnd1 = I32s::splat(1 << (round1 - 1));
    let max_px = I16s::splat(max_px as i16);
    let h5 = coeffs[1][0] == 0;
    let v5 = coeffs[0][0] == 0;

    let mut scratch = WideScratchGuard::take();
    let WideScratch { wide, ring, .. } = &mut *scratch;

    let mut hpass = |r: usize, out: &mut [i16; LR_MAX_W]| {
        let slen = w + 2 * LR_BORDER;
        widen_row(&src[r][..slen], wide);
        zero.copy_to_slice(&mut wide[slen..slen + N]);
        let mut v0 = I16s::from_slice(&wide[0..N]);
        for c in (0..w).step_by(N) {
            let v1 = I16s::from_slice(&wide[c + N..c + 2 * N]);
            let p1 = simd_swizzle!(v0, v1, [1, 2, 3, 4, 5, 6, 7, 8]);
            let p2 = simd_swizzle!(v0, v1, [2, 3, 4, 5, 6, 7, 8, 9]);
            let p3 = simd_swizzle!(v0, v1, [3, 4, 5, 6, 7, 8, 9, 10]);
            let p4 = simd_swizzle!(v0, v1, [4, 5, 6, 7, 8, 9, 10, 11]);
            let p5 = simd_swizzle!(v0, v1, [5, 6, 7, 8, 9, 10, 11, 12]);
            let s = if h5 {
                madd(p1 + p5, p2 + p4, hf[1], hf[2]) + madd(p3, zero, hf[3], zero)
            } else {
                let p0 = v0;
                let p6 = simd_swizzle!(v0, v1, [6, 7, 8, 9, 10, 11, 12, 13]);
                madd(p0 + p6, p1 + p5, hf[0], hf[1]) + madd(p2 + p4, p3, hf[2], hf[3])
            };
            v0 = v1;
            let v = narrow_sat((s + rnd0) >> round0 as i32)
                .simd_max(lo0)
                .simd_min(hi0);
            v.copy_to_slice(&mut out[c..c + N]);
        }
    };

    let chunks = (w / N).min(LR_MAX_W / N);
    let c_tail = chunks * N;
    if v5 {
        for r in 1..5 {
            hpass(r, &mut ring[r & 7]);
        }
        for r in 0..h {
            hpass(r + 5, &mut ring[(r + 5) & 7]);
            let r1 = &ring[(r + 1) & 7];
            let r2 = &ring[(r + 2) & 7];
            let r3 = &ring[(r + 3) & 7];
            let r4 = &ring[(r + 4) & 7];
            let r5 = &ring[(r + 5) & 7];
            let vstep = |c: usize| {
                let s = madd(
                    I16s::from_slice(&r1[c..]),
                    I16s::from_slice(&r5[c..]),
                    vf[1],
                    vf[1],
                ) + madd(
                    I16s::from_slice(&r2[c..]),
                    I16s::from_slice(&r4[c..]),
                    vf[2],
                    vf[2],
                ) + madd(I16s::from_slice(&r3[c..]), zero, vf[3], zero);
                clip_px((s + rnd1) >> round1 as i32, max_px)
            };
            let out = &mut dst[r * dst_stride..r * dst_stride + w];
            let out_chunks = &mut out[..c_tail];
            for k in 0..chunks {
                let c = k * N;
                store_px8(&mut out_chunks[c..], vstep(c));
            }
            if c_tail < w {
                store_px8_tail(&mut out[c_tail..w], vstep(c_tail));
            }
        }
    } else {
        for (r, slot) in ring.iter_mut().enumerate().take(6) {
            hpass(r, slot);
        }
        for r in 0..h {
            hpass(r + 6, &mut ring[(r + 6) & 7]);
            let r0 = &ring[r & 7];
            let r1 = &ring[(r + 1) & 7];
            let r2 = &ring[(r + 2) & 7];
            let r3 = &ring[(r + 3) & 7];
            let r4 = &ring[(r + 4) & 7];
            let r5 = &ring[(r + 5) & 7];
            let r6 = &ring[(r + 6) & 7];
            let vstep = |c: usize| {
                let s = madd(
                    I16s::from_slice(&r0[c..]),
                    I16s::from_slice(&r6[c..]),
                    vf[0],
                    vf[0],
                ) + madd(
                    I16s::from_slice(&r1[c..]),
                    I16s::from_slice(&r5[c..]),
                    vf[1],
                    vf[1],
                ) + madd(
                    I16s::from_slice(&r2[c..]),
                    I16s::from_slice(&r4[c..]),
                    vf[2],
                    vf[2],
                ) + madd(I16s::from_slice(&r3[c..]), zero, vf[3], zero);
                clip_px((s + rnd1) >> round1 as i32, max_px)
            };
            let out = &mut dst[r * dst_stride..r * dst_stride + w];
            let out_chunks = &mut out[..c_tail];
            for k in 0..chunks {
                let c = k * N;
                store_px8(&mut out_chunks[c..], vstep(c));
            }
            if c_tail < w {
                store_px8_tail(&mut out[c_tail..w], vstep(c_tail));
            }
        }
    }
}

/// `x * K` for a small constant `K`, with shifts and adds.
#[inline(always)]
fn mul_const<const K: i32>(x: I32s) -> I32s {
    match K {
        3 => (x << 1) + x,
        4 => x << 2,
        5 => (x << 2) + x,
        6 => (x << 2) + (x << 1),
        9 => (x << 3) + x,
        #[cfg(not(target_arch = "aarch64"))]
        25 => (x << 4) + (x << 3) + x,
        _ => x * I32s::splat(K),
    }
}

/// `SGR_A2[z]` (see `dsp::lr`) for `z <= 255`, computed as in its
/// definition, `((z << 8) + z / 2) / (z + 1)`, with an `f32` division.
///
/// The exact quotient `q` is below 256 and its fractional part is 0 or
/// in `[1 / 256, 255 / 256]`; the `f32` quotient and the subtraction of
/// `255 / 512` below are within `2^-16` of exact, so
/// `r = q - 255 / 512` is within `0.499` of `floor(q)`. Adding
/// `1.5 * 2^23` (the result is in `[2^23, 2^24)`, where the `f32` spacing
/// is 1) rounds `r` to the nearest integer, which is then read from the
/// low mantissa bits (a float to int `cast` is not vectorized because of
/// its saturating semantics). The formula gives 0 for z = 0 and 255 for
/// z = 255 instead of 1 and 256: add 1 there.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline(always)]
fn sgr_a2(z: I32s) -> I32s {
    const MAGIC: f32 = (3 << 22) as f32;
    let num = ((z << 8) + (z >> 1)).cast::<f32>();
    let q = num / (z + I32s::splat(1)).cast::<f32>();
    let r = (q - Simd::splat(255.0 / 512.0)) + Simd::splat(MAGIC);
    let q = (r.to_bits() - Simd::splat(MAGIC.to_bits())).cast::<i32>();
    let (zero, one) = (I32s::splat(0), I32s::splat(1));
    let fix = (z.simd_eq(zero) | z.simd_eq(I32s::splat(255))).select(one, zero);
    q + fix
}

#[inline(always)]
fn vsum_init3_w<T: Pixel, const TAIL: bool>(
    src: &[&[T]],
    row0: usize,
    col0: usize,
    ncols: usize,
    bs: &mut WideBoxSums,
) {
    let r0 = &src[row0][col0..col0 + ncols];
    let r1 = &src[row0 + 1][col0..col0 + ncols];
    let r2 = &src[row0 + 2][col0..col0 + ncols];
    let mut c = 0;
    while c + N <= ncols {
        let (p0, p1, p2) = (load_px(&r0[c..]), load_px(&r1[c..]), load_px(&r2[c..]));
        let sum = (p0 + p1 + p2).cast::<u16>();
        let sq = madd(p0, p1, p0, p1) + mul_wide(p2, p2);
        sum.copy_to_slice(&mut bs.sum[c..c + N]);
        sq.copy_to_slice(&mut bs.sq[c..c + N]);
        c += N;
    }
    if TAIL && c < ncols {
        let (p0, p1, p2) = (
            load_px_or_zero(&r0[c..]),
            load_px_or_zero(&r1[c..]),
            load_px_or_zero(&r2[c..]),
        );
        let sum = (p0 + p1 + p2).cast::<u16>();
        let sq = madd(p0, p1, p0, p1) + mul_wide(p2, p2);
        sum.copy_to_slice(&mut bs.sum[c..c + N]);
        sq.copy_to_slice(&mut bs.sq[c..c + N]);
    }
}

#[inline(always)]
fn vsum_slide1_w<T: Pixel, const TAIL: bool>(
    src: &[&[T]],
    row_old: usize,
    row_new: usize,
    col0: usize,
    ncols: usize,
    bs: &mut WideBoxSums,
) {
    let ro = &src[row_old][col0..col0 + ncols];
    let rn = &src[row_new][col0..col0 + ncols];
    let mut c = 0;
    while c + N <= ncols {
        let (po, pn) = (load_px(&ro[c..]), load_px(&rn[c..]));
        let diff = pn - po;
        let sum = Simd::<u16, N>::from_slice(&bs.sum[c..c + N]) + diff.cast::<u16>();
        let sq = I32s::from_slice(&bs.sq[c..c + N]) + mul_wide(pn + po, diff);
        sum.copy_to_slice(&mut bs.sum[c..c + N]);
        sq.copy_to_slice(&mut bs.sq[c..c + N]);
        c += N;
    }
    if TAIL && c < ncols {
        let (po, pn) = (load_px_or_zero(&ro[c..]), load_px_or_zero(&rn[c..]));
        let diff = pn - po;
        let sum = Simd::<u16, N>::from_slice(&bs.sum[c..c + N]) + diff.cast::<u16>();
        let sq = I32s::from_slice(&bs.sq[c..c + N]) + mul_wide(pn + po, diff);
        sum.copy_to_slice(&mut bs.sum[c..c + N]);
        sq.copy_to_slice(&mut bs.sq[c..c + N]);
    }
}

#[inline(always)]
fn vsum_add2_w<T: Pixel, const TAIL: bool>(
    src: &[&[T]],
    row_a: usize,
    row_b: usize,
    ncols: usize,
    bs3: &WideBoxSums,
    bs5: &mut WideBoxSums,
) {
    let ra = &src[row_a][..ncols];
    let rb = &src[row_b][..ncols];
    let mut c = 0;
    while c + N <= ncols {
        let (pa, pb) = (load_px(&ra[c..]), load_px(&rb[c..]));
        let s3 = Simd::<u16, N>::from_slice(&bs3.sum[c..c + N]);
        let q3 = I32s::from_slice(&bs3.sq[c..c + N]);
        (s3 + (pa + pb).cast::<u16>()).copy_to_slice(&mut bs5.sum[c..c + N]);
        (q3 + madd(pa, pb, pa, pb)).copy_to_slice(&mut bs5.sq[c..c + N]);
        c += N;
    }
    if TAIL && c < ncols {
        let (pa, pb) = (load_px_or_zero(&ra[c..]), load_px_or_zero(&rb[c..]));
        let s3 = Simd::<u16, N>::from_slice(&bs3.sum[c..c + N]);
        let q3 = I32s::from_slice(&bs3.sq[c..c + N]);
        (s3 + (pa + pb).cast::<u16>()).copy_to_slice(&mut bs5.sum[c..c + N]);
        (q3 + madd(pa, pb, pa, pb)).copy_to_slice(&mut bs5.sq[c..c + N]);
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn vsum_slide1_and_5_w<T: Pixel, const TAIL: bool>(
    src: &[&[T]],
    row_old: usize,
    row_new1: usize,
    row_new2: usize,
    ncols: usize,
    bs3: &mut WideBoxSums,
    bs5: &mut WideBoxSums,
) {
    let ro = &src[row_old][..ncols];
    let rn1 = &src[row_new1][..ncols];
    let rn2 = &src[row_new2][..ncols];
    let mut c = 0;
    while c + N <= ncols {
        let (po, p1, p2) = (load_px(&ro[c..]), load_px(&rn1[c..]), load_px(&rn2[c..]));
        let s3 = Simd::<u16, N>::from_slice(&bs3.sum[c..c + N]);
        let q3 = I32s::from_slice(&bs3.sq[c..c + N]);
        let diff = p1 - po;
        (s3 + diff.cast::<u16>()).copy_to_slice(&mut bs3.sum[c..c + N]);
        (q3 + mul_wide(p1 + po, diff)).copy_to_slice(&mut bs3.sq[c..c + N]);
        (s3 + (p1 + p2).cast::<u16>()).copy_to_slice(&mut bs5.sum[c..c + N]);
        (q3 + madd(p1, p2, p1, p2)).copy_to_slice(&mut bs5.sq[c..c + N]);
        c += N;
    }
    if TAIL && c < ncols {
        let (po, p1, p2) = (
            load_px_or_zero(&ro[c..]),
            load_px_or_zero(&rn1[c..]),
            load_px_or_zero(&rn2[c..]),
        );
        let s3 = Simd::<u16, N>::from_slice(&bs3.sum[c..c + N]);
        let q3 = I32s::from_slice(&bs3.sq[c..c + N]);
        let diff = p1 - po;
        (s3 + diff.cast::<u16>()).copy_to_slice(&mut bs3.sum[c..c + N]);
        (q3 + mul_wide(p1 + po, diff)).copy_to_slice(&mut bs3.sq[c..c + N]);
        (s3 + (p1 + p2).cast::<u16>()).copy_to_slice(&mut bs5.sum[c..c + N]);
        (q3 + madd(p1, p2, p1, p2)).copy_to_slice(&mut bs5.sq[c..c + N]);
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn vsum_slide2_w<T: Pixel, const TAIL: bool>(
    src: &[&[T]],
    row_old0: usize,
    row_old1: usize,
    row_new0: usize,
    row_new1: usize,
    ncols: usize,
    bs: &mut WideBoxSums,
) {
    let ro0 = &src[row_old0][..ncols];
    let ro1 = &src[row_old1][..ncols];
    let rn0 = &src[row_new0][..ncols];
    let rn1 = &src[row_new1][..ncols];
    let mut c = 0;
    while c + N <= ncols {
        let (po0, po1) = (load_px(&ro0[c..]), load_px(&ro1[c..]));
        let (pn0, pn1) = (load_px(&rn0[c..]), load_px(&rn1[c..]));
        let (d0, d1) = (pn0 - po0, pn1 - po1);
        let sum = Simd::<u16, N>::from_slice(&bs.sum[c..c + N]) + (d0 + d1).cast::<u16>();
        let sq = I32s::from_slice(&bs.sq[c..c + N]) + madd(pn0 + po0, pn1 + po1, d0, d1);
        sum.copy_to_slice(&mut bs.sum[c..c + N]);
        sq.copy_to_slice(&mut bs.sq[c..c + N]);
        c += N;
    }
    if TAIL && c < ncols {
        let (po0, po1) = (load_px_or_zero(&ro0[c..]), load_px_or_zero(&ro1[c..]));
        let (pn0, pn1) = (load_px_or_zero(&rn0[c..]), load_px_or_zero(&rn1[c..]));
        let (d0, d1) = (pn0 - po0, pn1 - po1);
        let sum = Simd::<u16, N>::from_slice(&bs.sum[c..c + N]) + (d0 + d1).cast::<u16>();
        let sq = I32s::from_slice(&bs.sq[c..c + N]) + madd(pn0 + po0, pn1 + po1, d0, d1);
        sum.copy_to_slice(&mut bs.sum[c..c + N]);
        sq.copy_to_slice(&mut bs.sq[c..c + N]);
    }
}

/// Computes one raw row of `X = 256 - A` and `B` arrays (spec 7.17.3) into
/// `ab` (`ab.a = x`, `ab.b = b2`) from the vertical box sums `(sums, sqs)`
/// with radius `R` and scale `s`.
///
/// Ranges: box sums `b <= 25 * 4095` and `a <= 25 * 4095^2` (i32);
/// `d = Round2(b, bd - 8) <= 6398` (i16). All products fit in 32 bits:
/// * `p * s` once `p` is clamped to `pmax`, the smallest value giving
///   `z >= 255` (larger `z` are clamped to 255 by the spec): `p * s +
///   2^19 < 255 * 2^20 + s`, which also gives `z <= 255` directly.
/// * `(256 - a2) * b * one_over_n <= 255 * 25 * 4095 * 164` (`R = 2`) or
///   `255 * 9 * 4095 * 455` (`R = 1`), both below `2^32 - 2^11`, so
///   these are computed modulo 2^32 in `u32` lanes.
#[inline(always)]
fn ab_from_vsum_w<const R: usize>(
    sums: &[u16],
    sqs: &[i32],
    w: usize,
    s: u32,
    bd: u32,
    ab: &mut WideAbRow,
) {
    let steps = (w + 2).div_ceil(N);
    assert!(
        w <= LR_MAX_W
            && steps * N <= AB_LEN
            && sums.len() >= SUM_LEN - 1
            && sqs.len() >= SUM_LEN - 1
    );
    let sums = &sums[..steps * N + 4];
    let sqs = &sqs[..steps * N + 4];
    let sh_a = 2 * (bd - 8) as i32;
    let rnd_a = I32s::splat((1 << sh_a) >> 1);
    let sh_b = (bd - 8) as i32;
    let rnd_b = I32s::splat((1 << sh_b) >> 1);
    let pmax = I32s::splat(
        ((255u32 << SGRPROJ_MTABLE_BITS) - (1 << (SGRPROJ_MTABLE_BITS - 1))).div_ceil(s) as i32,
    );
    let sv = I32s::splat(s as i32);
    let one_over_n = U32s::splat(one_over_n(R));
    let rnd_r = U32s::splat(1 << (SGRPROJ_RECIP_BITS - 1));

    let step_bp = |j: usize| -> (I32s, I32s) {
        let sum_win = &sums[j..j + N + 4];
        let sq_win = &sqs[j..j + N + 4];
        let s0 = Simd::<u16, N>::from_slice(&sum_win[..N]);
        let s1 = Simd::<u16, N>::from_slice(&sum_win[1..1 + N]);
        let s2 = Simd::<u16, N>::from_slice(&sum_win[2..2 + N]);
        let bsum: I32s = if R == 1 {
            (s0 + s1).cast::<i32>() + s2.cast::<i32>()
        } else {
            let s3 = Simd::<u16, N>::from_slice(&sum_win[3..3 + N]);
            let s4 = Simd::<u16, N>::from_slice(&sum_win[4..4 + N]);
            (s0 + s1).cast::<i32>() + (s2 + s3).cast::<i32>() + s4.cast::<i32>()
        };
        let q0 = I32s::from_slice(&sq_win[..N]);
        let q1 = I32s::from_slice(&sq_win[1..1 + N]);
        let q2 = I32s::from_slice(&sq_win[2..2 + N]);
        let asum = if R == 1 {
            (q0 + q1) + q2
        } else {
            let q3 = I32s::from_slice(&sq_win[3..3 + N]);
            let q4 = I32s::from_slice(&sq_win[4..4 + N]);
            (q0 + q1) + (q2 + q3) + q4
        };
        let a = (asum + rnd_a) >> sh_a;
        let dd = ((bsum + rnd_b) >> sh_b).cast::<i16>();
        let an = if R == 1 { mul_const::<9>(a) } else { mul_const::<25>(a) };
        (bsum, an - mul_wide(dd, dd))
    };

    #[cfg(target_arch = "aarch64")]
    {
        let z_from_p = |p: I32s| -> Simd<u8, N> {
            let y16 = ((p.simd_min(pmax) * sv) >> 16)
                .cast::<i16>()
                .simd_max(I16s::splat(0));
            ((y16.cast::<u16>() + Simd::splat(8)) >> 4).cast::<u8>()
        };
        let store_ab = |oa: &mut [i16], ob: &mut [i32], x_u16: Simd<u16, N>, bsum: I32s| {
            let b2 = x_u16.cast::<u32>() * bsum.cast::<u32>() * one_over_n;
            x_u16.cast::<i16>().copy_to_slice(oa);
            ((b2 + rnd_r) >> SGRPROJ_RECIP_BITS)
                .cast::<i32>()
                .copy_to_slice(ob);
        };
        let pairs = steps / 2;
        for ((pair, oa), ob) in ab.a[..pairs * 2 * N]
            .chunks_exact_mut(2 * N)
            .enumerate()
            .zip(ab.b[..pairs * 2 * N].chunks_exact_mut(2 * N))
        {
            let j = pair * (2 * N);
            let (bsum0, p0) = step_bp(j);
            let (bsum1, p1) = step_bp(j + N);
            let z0 = z_from_p(p0);
            let z1 = z_from_p(p1);
            let z16: Simd<u8, 16> = simd_swizzle!(
                z0,
                z1,
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            );
            let x16 = narrow::sgr_x_u8(z16);
            let x0: Simd<u8, N> = simd_swizzle!(x16, [0, 1, 2, 3, 4, 5, 6, 7]);
            let x1: Simd<u8, N> = simd_swizzle!(x16, [8, 9, 10, 11, 12, 13, 14, 15]);
            store_ab(&mut oa[..N], &mut ob[..N], x0.cast::<u16>(), bsum0);
            store_ab(&mut oa[N..], &mut ob[N..], x1.cast::<u16>(), bsum1);
        }
        if steps & 1 != 0 {
            let j = pairs * (2 * N);
            let (bsum, p) = step_bp(j);
            let z0 = z_from_p(p);
            let x16 = narrow::sgr_x_u8(z0.resize::<16>(0));
            let x0: Simd<u8, N> = simd_swizzle!(x16, [0, 1, 2, 3, 4, 5, 6, 7]);
            store_ab(
                &mut ab.a[j..j + N],
                &mut ab.b[j..j + N],
                x0.cast::<u16>(),
                bsum,
            );
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for k in 0..steps {
        let j = k * N;
        let (bsum, p) = step_bp(j);
        let p = (p & !(p >> 31)).simd_min(pmax);
        let rnd_z = I32s::splat(1 << (SGRPROJ_MTABLE_BITS - 1));
        let z = (p * sv + rnd_z) >> SGRPROJ_MTABLE_BITS as i32;
        let x = (I32s::splat(1 << SGRPROJ_SGR_BITS) - sgr_a2(z)).cast::<i16>();
        let b2 = x.cast::<u16>().cast::<u32>() * bsum.cast::<u32>() * one_over_n;
        x.copy_to_slice(&mut ab.a[j..j + N]);
        ((b2 + rnd_r) >> SGRPROJ_RECIP_BITS)
            .cast::<i32>()
            .copy_to_slice(&mut ab.b[j..j + N]);
    }
}

/// Weights the raw `X`/`B` row in `ab` horizontally (see
/// [`WeightedRow`]) into `out` for columns `0..w`.
#[inline(always)]
fn weigh_row_w<const K0: i32, const K1: i32>(ab: &WideAbRow, w: usize, out: &mut WeightedRow) {
    assert!(w <= LR_MAX_W);
    const { assert!((K0 == 5 && K1 == 6) || (K0 == 3 && K1 == 4)) };
    for c in (0..w).step_by(N) {
        let a_win = &ab.a[c..c + N + 2];
        let b_win = &ab.b[c..c + N + 2];
        let a0 = I16s::from_slice(&a_win[..N]);
        let a1 = I16s::from_slice(&a_win[1..1 + N]);
        let a2 = I16s::from_slice(&a_win[2..2 + N]);
        let b0 = I32s::from_slice(&b_win[..N]);
        let b1 = I32s::from_slice(&b_win[1..1 + N]);
        let b2 = I32s::from_slice(&b_win[2..2 + N]);
        let sa02 = a0 + a2;
        let sb02 = b0 + b2;
        let sa = sa02 + a1;
        let sb = sb02 + b1;
        let (wa, wb) = if K0 == 5 {
            ((sa << 2) + sa + a1, (sb << 2) + sb + b1)
        } else {
            ((sa << 2) - sa02, (sb << 2) - sb02)
        };
        wa.copy_to_slice(&mut out.a[c..c + N]);
        wb.copy_to_slice(&mut out.b[c..c + N]);
    }
}

/// Fused horizontal weighting for the 12-bit `r = 1` pass: accumulates `(4, 4)`
/// into `tm_accum` (`p343_prev + m444_cur`) and writes `(3, 4)` into `out343`.
#[inline(always)]
fn weigh_r1_w(ab: &WideAbRow, w: usize, tm_accum: &mut WeightedRow, out343: &mut WeightedRow) {
    assert!(w <= LR_MAX_W);
    for c in (0..w).step_by(N) {
        let a_win = &ab.a[c..c + N + 2];
        let b_win = &ab.b[c..c + N + 2];
        let a0 = I16s::from_slice(&a_win[..N]);
        let a1 = I16s::from_slice(&a_win[1..1 + N]);
        let a2 = I16s::from_slice(&a_win[2..2 + N]);
        let b0 = I32s::from_slice(&b_win[..N]);
        let b1 = I32s::from_slice(&b_win[1..1 + N]);
        let b2 = I32s::from_slice(&b_win[2..2 + N]);
        let sa02 = a0 + a2;
        let sb02 = b0 + b2;
        let sa444 = (sa02 + a1) << 2;
        let sa343 = sa444 - sa02;
        let sb444 = (sb02 + b1) << 2;
        let sb343 = sb444 - sb02;
        (I16s::from_slice(&tm_accum.a[c..c + N]) + sa444).copy_to_slice(&mut tm_accum.a[c..c + N]);
        (I32s::from_slice(&tm_accum.b[c..c + N]) + sb444).copy_to_slice(&mut tm_accum.b[c..c + N]);
        sa343.copy_to_slice(&mut out343.a[c..c + N]);
        sb343.copy_to_slice(&mut out343.b[c..c + N]);
    }
}

#[inline(always)]
fn two_mut_w<T>(a: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
    if i < j {
        let (l, r) = a.split_at_mut(j);
        (&mut l[i], &mut r[0])
    } else {
        let (l, r) = a.split_at_mut(i);
        (&mut r[0], &mut l[j])
    }
}

/// `f - (px << 4) = Round2(b - x * px, shift)` of the box filter output.
#[inline(always)]
fn box_out(x: I16s, b: I32s, px: I16s, shift: u32) -> I32s {
    (b - mul_wide(x, px) + I32s::splat(1 << (shift - 1))) >> shift as i32
}

const WIDE_CHUNKS: usize = LR_MAX_W / N;

struct WideChunkWeights {
    w0: [i16; WIDE_CHUNKS],
    w2: [i16; WIDE_CHUNKS],
}

#[cfg_attr(target_arch = "aarch64", inline(never))]
#[cfg_attr(not(target_arch = "aarch64"), inline(always))]
fn sgr_finish_row_w<T: Pixel, const P0: u8, const P1: bool>(
    px_row: &[T],
    out: &mut [T],
    rows: [&WeightedRow; 4],
    wts: &WideChunkWeights,
    max_px: I16s,
) {
    let w = out.len();
    assert!(w <= LR_MAX_W && px_row.len() >= w);
    let [prev, next, tm, d] = rows;
    let shift = SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS;
    let rnd = I32s::splat(1 << (shift - 1));
    let ld = |r: &WeightedRow, c: usize| {
        (
            I16s::from_slice(&r.a[c..c + N]),
            I32s::from_slice(&r.b[c..c + N]),
        )
    };
    let step = |px: I16s, k: usize| -> I16s {
        let c = k * N;
        let f0 = match P0 {
            1 => {
                let (na, nb) = ld(next, c);
                box_out(na, nb, px, SGRPROJ_SGR_BITS + 4 - SGRPROJ_RST_BITS)
            }
            _ => {
                let (pa, pb) = ld(prev, c);
                let (na, nb) = ld(next, c);
                box_out(
                    pa + na,
                    pb + nb,
                    px,
                    SGRPROJ_SGR_BITS + 5 - SGRPROJ_RST_BITS,
                )
            }
        };
        let (tma, tmb) = ld(tm, c);
        let (da, db) = ld(d, c);
        let f1 = box_out(
            tma + da,
            tmb + db,
            px,
            SGRPROJ_SGR_BITS + 5 - SGRPROJ_RST_BITS,
        );
        let (w0, w2) = (I32s::splat(wts.w0[k] as i32), I32s::splat(wts.w2[k] as i32));
        let v = match (P0 != 0, P1) {
            (true, true) => f0 * w0 + f1 * w2 + rnd,
            (true, false) => f0 * w0 + rnd,
            _ => f1 * w2 + rnd,
        };
        // Even at bd = 12, |delta| <= 11229 and px in [0, 4095], so `px + delta`
        // stays strictly inside i16 without saturating narrowing.
        let delta = (v >> shift as i32).cast::<i16>();
        (px + delta).simd_max(I16s::splat(0)).simd_min(max_px)
    };
    let chunks = (w / N).min(WIDE_CHUNKS);
    let px_chunks = &px_row[..chunks * N];
    let out_chunks = &mut out[..chunks * N];
    for (k, (px_s, out_s)) in px_chunks
        .chunks_exact(N)
        .zip(out_chunks.chunks_exact_mut(N))
        .enumerate()
    {
        store_px8(out_s, step(load_px(px_s), k));
    }
    let (k, c) = (chunks, chunks * N);
    if c < w && k < WIDE_CHUNKS {
        store_px8_tail(&mut out[c..w], step(load_px_or_zero(&px_row[c..w]), k));
    }
}

#[allow(clippy::too_many_arguments)]
fn sgr_filter_both_w_impl<T: Pixel, const TAIL: bool>(
    dst: &mut [T],
    dst_stride: usize,
    src: &[&[T]],
    w: usize,
    h: usize,
    ncols: usize,
    s0: u32,
    s1: u32,
    bd: u32,
    wts: &WideChunkWeights,
) {
    assert!(w <= LR_MAX_W);
    let max_px = I16s::splat(((1i32 << bd) - 1) as i16);
    let mut scratch = WideScratchGuard::take();
    let WideScratch {
        bs3,
        bs5,
        ab,
        p0,
        p343,
        ..
    } = &mut *scratch;
    let mut cur0 = 0usize;

    vsum_init3_w::<T, TAIL>(src, 1, 0, ncols, bs3);
    ab_from_vsum_w::<1>(&bs3.sum[1..], &bs3.sq[1..], w, s1, bd, ab);
    weigh_row_w::<3, 4>(ab, w, &mut p343[0]);
    vsum_add2_w::<T, TAIL>(src, 0, 4, ncols, bs3, bs5);
    ab_from_vsum_w::<2>(&bs5.sum, &bs5.sq, w, s0, bd, ab);
    weigh_row_w::<5, 6>(ab, w, &mut p0[0]);

    vsum_slide1_w::<T, TAIL>(src, 1, 4, 0, ncols, bs3);
    ab_from_vsum_w::<1>(&bs3.sum[1..], &bs3.sq[1..], w, s1, bd, ab);
    {
        let (tm0, out1) = two_mut_w(p343, 0, 1);
        weigh_r1_w(ab, w, tm0, out1);
    }

    let mut t_idx = 0usize;
    let mut m_idx = 1usize;
    let mut d_idx = 2usize;
    for i in 0..h {
        let odd = i & 1 == 1;
        if !odd {
            vsum_slide1_and_5_w::<T, TAIL>(src, i + 2, i + 5, i + 6, ncols, bs3, bs5);
            cur0 ^= 1;
            ab_from_vsum_w::<2>(&bs5.sum, &bs5.sq, w, s0, bd, ab);
            weigh_row_w::<5, 6>(ab, w, &mut p0[cur0]);
        } else {
            vsum_slide1_w::<T, TAIL>(src, i + 2, i + 5, 0, ncols, bs3);
        }
        ab_from_vsum_w::<1>(&bs3.sum[1..], &bs3.sq[1..], w, s1, bd, ab);
        {
            let (tm_next, out_d) = two_mut_w(p343, m_idx, d_idx);
            weigh_r1_w(ab, w, tm_next, out_d);
        }

        let (prev, next) = (&p0[cur0 ^ 1], &p0[cur0]);
        let (tm, d) = (&p343[t_idx], &p343[d_idx]);
        (t_idx, m_idx, d_idx) = (m_idx, d_idx, t_idx);

        let px_row = &src[i + LR_BORDER][LR_BORDER..LR_BORDER + w];
        let out = &mut dst[i * dst_stride..i * dst_stride + w];
        let rows = [prev, next, tm, d];
        if odd {
            sgr_finish_row_w::<T, 1, true>(px_row, out, rows, wts, max_px);
        } else {
            sgr_finish_row_w::<T, 2, true>(px_row, out, rows, wts, max_px);
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(target_arch = "aarch64", inline(never))]
fn sgr_filter_both_w<T: Pixel>(
    dst: &mut [T],
    dst_stride: usize,
    src: &[&[T]],
    w: usize,
    h: usize,
    ncols: usize,
    s0: u32,
    s1: u32,
    bd: u32,
    wts: &WideChunkWeights,
) {
    sgr_filter_both_w_impl::<T, false>(dst, dst_stride, src, w, h, ncols, s0, s1, bd, wts);
}

#[allow(clippy::too_many_arguments)]
#[cold]
#[inline(never)]
fn sgr_filter_both_w_tail<T: Pixel>(
    dst: &mut [T],
    dst_stride: usize,
    src: &[&[T]],
    w: usize,
    h: usize,
    ncols: usize,
    s0: u32,
    s1: u32,
    bd: u32,
    wts: &WideChunkWeights,
) {
    sgr_filter_both_w_impl::<T, true>(dst, dst_stride, src, w, h, ncols, s0, s1, bd, wts);
}

#[allow(clippy::too_many_arguments)]
fn sgr_filter_r1_w_impl<T: Pixel, const TAIL: bool>(
    dst: &mut [T],
    dst_stride: usize,
    src: &[&[T]],
    w: usize,
    h: usize,
    ncols: usize,
    s1: u32,
    bd: u32,
    wts: &WideChunkWeights,
) {
    assert!(w <= LR_MAX_W);
    let col0 = LR_BORDER - 2;
    let max_px = I16s::splat(((1i32 << bd) - 1) as i16);
    let mut scratch = WideScratchGuard::take();
    let WideScratch { bs3, ab, p343, .. } = &mut *scratch;

    vsum_init3_w::<T, TAIL>(src, 1, col0, ncols, bs3);
    ab_from_vsum_w::<1>(&bs3.sum, &bs3.sq, w, s1, bd, ab);
    weigh_row_w::<3, 4>(ab, w, &mut p343[0]);

    vsum_slide1_w::<T, TAIL>(src, 1, 4, col0, ncols, bs3);
    ab_from_vsum_w::<1>(&bs3.sum, &bs3.sq, w, s1, bd, ab);
    {
        let (tm0, out1) = two_mut_w(p343, 0, 1);
        weigh_r1_w(ab, w, tm0, out1);
    }

    let mut t_idx = 0usize;
    let mut m_idx = 1usize;
    let mut d_idx = 2usize;
    for i in 0..h {
        vsum_slide1_w::<T, TAIL>(src, i + 2, i + 5, col0, ncols, bs3);
        ab_from_vsum_w::<1>(&bs3.sum, &bs3.sq, w, s1, bd, ab);
        {
            let (tm_next, out_d) = two_mut_w(p343, m_idx, d_idx);
            weigh_r1_w(ab, w, tm_next, out_d);
        }
        let (tm, d) = (&p343[t_idx], &p343[d_idx]);
        (t_idx, m_idx, d_idx) = (m_idx, d_idx, t_idx);

        let px_row = &src[i + LR_BORDER][LR_BORDER..LR_BORDER + w];
        let out = &mut dst[i * dst_stride..i * dst_stride + w];
        sgr_finish_row_w::<T, 0, true>(px_row, out, [tm, tm, tm, d], wts, max_px);
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(target_arch = "aarch64", inline(never))]
fn sgr_filter_r1_w<T: Pixel>(
    dst: &mut [T],
    dst_stride: usize,
    src: &[&[T]],
    w: usize,
    h: usize,
    ncols: usize,
    s1: u32,
    bd: u32,
    wts: &WideChunkWeights,
) {
    sgr_filter_r1_w_impl::<T, false>(dst, dst_stride, src, w, h, ncols, s1, bd, wts);
}

#[allow(clippy::too_many_arguments)]
#[cold]
#[inline(never)]
fn sgr_filter_r1_w_tail<T: Pixel>(
    dst: &mut [T],
    dst_stride: usize,
    src: &[&[T]],
    w: usize,
    h: usize,
    ncols: usize,
    s1: u32,
    bd: u32,
    wts: &WideChunkWeights,
) {
    sgr_filter_r1_w_impl::<T, true>(dst, dst_stride, src, w, h, ncols, s1, bd, wts);
}

#[allow(clippy::too_many_arguments)]
fn sgr_filter_r2_w_impl<T: Pixel, const TAIL: bool>(
    dst: &mut [T],
    dst_stride: usize,
    src: &[&[T]],
    w: usize,
    h: usize,
    ncols: usize,
    s0: u32,
    bd: u32,
    wts: &WideChunkWeights,
) {
    assert!(w <= LR_MAX_W);
    let max_px = I16s::splat(((1i32 << bd) - 1) as i16);
    let mut scratch = WideScratchGuard::take();
    let WideScratch {
        bs3, bs5, ab, p0, ..
    } = &mut *scratch;
    let mut cur0 = 0usize;

    vsum_init3_w::<T, TAIL>(src, 1, 0, ncols, bs3);
    vsum_add2_w::<T, TAIL>(src, 0, 4, ncols, bs3, bs5);
    ab_from_vsum_w::<2>(&bs5.sum, &bs5.sq, w, s0, bd, ab);
    weigh_row_w::<5, 6>(ab, w, &mut p0[0]);

    for i in 0..h {
        let odd = i & 1 == 1;
        if !odd {
            vsum_slide2_w::<T, TAIL>(src, i, i + 1, i + 5, i + 6, ncols, bs5);
            cur0 ^= 1;
            ab_from_vsum_w::<2>(&bs5.sum, &bs5.sq, w, s0, bd, ab);
            weigh_row_w::<5, 6>(ab, w, &mut p0[cur0]);
        }
        let (prev, next) = (&p0[cur0 ^ 1], &p0[cur0]);

        let px_row = &src[i + LR_BORDER][LR_BORDER..LR_BORDER + w];
        let out = &mut dst[i * dst_stride..i * dst_stride + w];
        let rows = [prev, next, next, next];
        if odd {
            sgr_finish_row_w::<T, 1, false>(px_row, out, rows, wts, max_px);
        } else {
            sgr_finish_row_w::<T, 2, false>(px_row, out, rows, wts, max_px);
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(target_arch = "aarch64", inline(never))]
fn sgr_filter_r2_w<T: Pixel>(
    dst: &mut [T],
    dst_stride: usize,
    src: &[&[T]],
    w: usize,
    h: usize,
    ncols: usize,
    s0: u32,
    bd: u32,
    wts: &WideChunkWeights,
) {
    sgr_filter_r2_w_impl::<T, false>(dst, dst_stride, src, w, h, ncols, s0, bd, wts);
}

#[allow(clippy::too_many_arguments)]
#[cold]
#[inline(never)]
fn sgr_filter_r2_w_tail<T: Pixel>(
    dst: &mut [T],
    dst_stride: usize,
    src: &[&[T]],
    w: usize,
    h: usize,
    ncols: usize,
    s0: u32,
    bd: u32,
    wts: &WideChunkWeights,
) {
    sgr_filter_r2_w_impl::<T, true>(dst, dst_stride, src, w, h, ncols, s0, bd, wts);
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn sgr_filter_units_w<T: Pixel>(
    dst: &mut [T],
    dst_stride: usize,
    src: &[&[T]],
    w: usize,
    h: usize,
    set: usize,
    units: &[(usize, [i32; 2])],
    bd: u32,
) {
    check_rect(w, h);
    let r0 = SGR_PARAMS[set][0] as usize;
    let r1 = SGR_PARAMS[set][2] as usize;
    let [s0, s1] = SGR_SCALE[set];
    let mut wts = WideChunkWeights {
        w0: [0; WIDE_CHUNKS],
        w2: [0; WIDE_CHUNKS],
    };
    let mut k = 0;
    let mut total = 0;
    for (i, &(uw, xqd)) in units.iter().enumerate() {
        debug_assert!(i + 1 == units.len() || uw % N == 0);
        let w2_raw = (1 << SGRPROJ_PRJ_BITS) - xqd[0] - xqd[1];
        let w0 = if r0 != 0 { xqd[0] } else { 0 };
        let w2 = if r1 != 0 { w2_raw } else { 0 };
        debug_assert!(w0.unsigned_abs() <= 1024 && w2.unsigned_abs() <= 1024);
        let (w0, w2) = (w0 as i16, w2 as i16);
        let end = (k + uw.div_ceil(N)).min(WIDE_CHUNKS);
        while k < end {
            wts.w0[k] = w0;
            wts.w2[k] = w2;
            k += 1;
        }
        total += uw;
    }
    assert_eq!(total, w);

    match (r0 != 0, r1 != 0) {
        (true, true) => {
            let ncols = box_cols(src, h, 0, w + 6);
            if ncols & 7 == 0 {
                sgr_filter_both_w(dst, dst_stride, src, w, h, ncols, s0, s1, bd, &wts);
            } else {
                sgr_filter_both_w_tail(dst, dst_stride, src, w, h, ncols, s0, s1, bd, &wts);
            }
        }
        (false, true) => {
            let ncols = box_cols(src, h, LR_BORDER - 2, w + 4);
            if ncols & 7 == 0 {
                sgr_filter_r1_w(dst, dst_stride, src, w, h, ncols, s1, bd, &wts);
            } else {
                sgr_filter_r1_w_tail(dst, dst_stride, src, w, h, ncols, s1, bd, &wts);
            }
        }
        (true, false) => {
            let ncols = box_cols(src, h, 0, w + 6);
            if ncols & 7 == 0 {
                sgr_filter_r2_w(dst, dst_stride, src, w, h, ncols, s0, bd, &wts);
            } else {
                sgr_filter_r2_w_tail(dst, dst_stride, src, w, h, ncols, s0, bd, &wts);
            }
        }
        (false, false) => {}
    }
}

/// Self guided filter, same contract as [`crate::decoder::av1::dsp::lr::sgr_filter`].
///
/// Vertical box sums are maintained incrementally by sliding a 3-row window
/// (and sharing the inner 3 rows with the 5-row pass when both passes are
/// active). Horizontal weighting fuses `(3, 4)` and `(4, 4)` for `r = 1`.
#[allow(clippy::too_many_arguments)]
pub fn sgr_filter<T: Pixel>(
    dst: &mut [T],
    dst_stride: usize,
    src: &[&[T]],
    w: usize,
    h: usize,
    set: usize,
    xqd: [i32; 2],
    bd: u32,
) {
    sgr_filter_units_w(dst, dst_stride, src, w, h, set, &[(w, xqd)], bd);
}

// ---------------------------------------------------------------------------
mod narrow {
    use super::*;
    use std::cell::Cell;
    const N: usize = 16;
    const LR_CHUNKS: usize = LR_MAX_W / N;
    type I16s = Simd<i16, N>;
    type U16s = Simd<u16, N>;
    type I32s = Simd<i32, N>;
    type U32s = Simd<u32, N>;
    // 4-lane types of the non-aarch64 tail of `ab_from_vsum`.
    #[cfg(not(target_arch = "aarch64"))]
    type I16x4 = Simd<i16, 4>;
    #[cfg(not(target_arch = "aarch64"))]
    type U16x4 = Simd<u16, 4>;
    #[cfg(not(target_arch = "aarch64"))]
    type I32x4 = Simd<i32, 4>;
    #[cfg(not(target_arch = "aarch64"))]
    type U32x4 = Simd<u32, 4>;
    type RingRow = [I16s; LR_CHUNKS];

    /// Lane indices interleaving two `N` lane vectors.
    #[cfg(not(target_arch = "aarch64"))]
    const INTERLEAVE: [usize; 2 * N] = {
        let mut t = [0; 2 * N];
        let mut i = 0;
        while i < 2 * N {
            t[i] = i / 2 + (i % 2) * N;
            i += 1;
        }
        t
    };
    /// Even (`EVEN[i] = 2 * i`) and odd lane indices of a `2 * N` vector.
    #[cfg(not(target_arch = "aarch64"))]
    const EVEN: [usize; N] = {
        let mut t = [0; N];
        let mut i = 0;
        while i < N {
            t[i] = 2 * i;
            i += 1;
        }
        t
    };
    #[cfg(not(target_arch = "aarch64"))]
    const ODD: [usize; N] = {
        let mut t = [0; N];
        let mut i = 0;
        while i < N {
            t[i] = 2 * i + 1;
            i += 1;
        }
        t
    };

    /// `a * fa + b * fb` per lane, with exact `i32` products and sum.
    ///
    /// Written as interleave / widen / multiply / pairwise add, which LLVM
    /// lowers to `pmaddwd` on x86.
    #[cfg(not(target_arch = "aarch64"))]
    #[inline(always)]
    fn madd(a: I16s, b: I16s, fa: I16s, fb: I16s) -> I32s {
        let x: Simd<i16, { 2 * N }> = simd_swizzle!(a, b, INTERLEAVE);
        let f: Simd<i16, { 2 * N }> = simd_swizzle!(fa, fb, INTERLEAVE);
        let p = x.cast::<i32>() * f.cast::<i32>();
        let even: I32s = simd_swizzle!(p, EVEN);
        let odd: I32s = simd_swizzle!(p, ODD);
        even + odd
    }

    /// `a * fa + b * fb` per lane, with exact `i32` products and sum.
    ///
    /// On aarch64 there is no `pmaddwd`; two widening multiplies lower to
    /// `smull` / `smlal` (the interleave form costs extra `zip` / `uzp`).
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn madd(a: I16s, b: I16s, fa: I16s, fb: I16s) -> I32s {
        mul_wide(a, fa) + mul_wide(b, fb)
    }

    /// `a * b` per lane, with exact `i32` products.
    #[inline(always)]
    fn mul_wide(a: I16s, b: I16s) -> I32s {
        a.cast::<i32>() * b.cast::<i32>()
    }

    /// Lane order of the [`madd_p`] results on x86: lane `i` holds element
    /// `Q[i]` (`Q` swaps the second and third groups of 4 lanes and is its
    /// own inverse). This is the order in which `punpck{l,h}wd` + `pmaddwd`
    /// on 256-bit vectors leave the sums (they work within 128-bit lanes),
    /// and `packssdw` of the two halves restores the natural order, so the
    /// `vperm2i128` / `vpermq` fix-ups of [`madd`] + [`narrow_sat`] go away.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    const Q: [usize; N] = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15];
    /// In-lane interleave of two `N` lane vectors: `punpcklwd` then
    /// `punpckhwd` of every 128-bit lane.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    const INTERLEAVE_P: [usize; 2 * N] = {
        let mut t = [0; 2 * N];
        let mut i = 0;
        while i < 2 * N {
            let e = Q[i / 2];
            t[i] = if i % 2 == 0 { e } else { e + N };
            i += 1;
        }
        t
    };

    /// `a * fa + b * fb` per lane (exact), for splat factors `fa` and `fb`,
    /// with the result lanes in the order [`Q`] on 256-bit x86 (natural order
    /// on 128-bit SSE2 and aarch64). Only combine the result with other `madd_p`
    /// results and splats, and narrow it with [`narrow_sat_p`].
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    fn madd_p(a: I16s, b: I16s, fa: I16s, fb: I16s) -> I32s {
        let x: Simd<i16, { 2 * N }> = simd_swizzle!(a, b, INTERLEAVE_P);
        // `fa` and `fb` are splats, so `INTERLEAVE` and `INTERLEAVE_P` are
        // equal, and `INTERLEAVE` lets LLVM fold the two splats into a
        // single 32-bit pair broadcast.
        let f: Simd<i16, { 2 * N }> = simd_swizzle!(fa, fb, INTERLEAVE);
        let p = x.cast::<i32>() * f.cast::<i32>();
        let even: I32s = simd_swizzle!(p, EVEN);
        let odd: I32s = simd_swizzle!(p, ODD);
        even + odd
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    #[inline(always)]
    fn madd_p(a: I16s, b: I16s, fa: I16s, fb: I16s) -> I32s {
        madd(a, b, fa, fb)
    }

    /// [`narrow_sat`] of a vector in [`madd_p`] order, giving natural order.
    /// On 256-bit x86 the reorder is a `<4 x u64>` `[0, 2, 1, 3]` shuffle (`vpermq`)
    /// after `narrow_sat`, which keeps `narrow_sat`'s `clamp + trunc` intact
    /// for `packssdw` selection and then cancels its trailing `vpermq`.
    #[inline(always)]
    fn narrow_sat_p(v: I32s) -> I16s {
        let n = narrow_sat(v);
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let n = {
            let q = Simd::<u64, 4>::from_ne_bytes(n.to_ne_bytes());
            let q: Simd<u64, 4> = simd_swizzle!(q, [0, 2, 1, 3]);
            I16s::from_ne_bytes(q.to_ne_bytes())
        };
        n
    }

    /// Loads `N` pixels from `src[..N]` into `i16` lanes.
    #[inline(always)]
    fn load_px<T: Pixel>(src: &[T]) -> I16s {
        if T::MAX_BITDEPTH == 8 {
            let mut a = [0u8; N];
            for (d, s) in a.iter_mut().zip(&src[..N]) {
                *d = s.to_i32() as u8;
            }
            Simd::<u8, N>::from_array(a).cast::<i16>()
        } else {
            let mut a = [0i16; N];
            for (d, s) in a.iter_mut().zip(&src[..N]) {
                *d = s.to_i32() as i16;
            }
            I16s::from_array(a)
        }
    }

    /// Like [`load_px`], but lanes past the end of `src` are 0.
    ///
    /// Kept `#[inline(never)]` (without `#[cold]`, which would cause LLVM
    /// HotColdSplitting to outline closures into a stack-frame capture struct)
    /// so LLVM's loop strength reduction does not hoist pointer induction
    /// variables from the tail loop into hot outer SIMD loops.
    #[inline(never)]
    fn load_px_or_zero<T: Pixel>(src: &[T]) -> I16s {
        let mut a = [0i16; N];
        for (d, s) in a.iter_mut().zip(src) {
            *d = s.to_i32() as i16;
        }
        I16s::from_array(a)
    }

    /// Stores `N` clamped pixel values `v` to `dst[..N]`.
    #[inline(always)]
    fn store_px<T: Pixel>(dst: &mut [T], v: I16s) {
        for (d, s) in dst[..N].iter_mut().zip(v.to_array()) {
            *d = T::from_i32(s as i32);
        }
    }

    /// Like [`store_px`], for a tail slice `dst` shorter than `N`.
    #[inline(never)]
    fn store_px_tail<T: Pixel>(dst: &mut [T], v: I16s) {
        for (d, s) in dst.iter_mut().zip(v.to_array()) {
            *d = T::from_i32(s as i32);
        }
    }

    /// `v` narrowed to `i16` lanes with signed saturation (`packssdw`).
    #[inline(always)]
    fn narrow_sat(v: I32s) -> I16s {
        v.simd_max(I32s::splat(i16::MIN as i32))
            .simd_min(I32s::splat(i16::MAX as i32))
            .cast::<i16>()
    }

    /// 4-lane [`narrow_sat`].
    #[cfg(not(target_arch = "aarch64"))]
    #[inline(always)]
    fn narrow_sat4(v: I32x4) -> I16x4 {
        v.simd_max(I32x4::splat(i16::MIN as i32))
            .simd_min(I32x4::splat(i16::MAX as i32))
            .cast::<i16>()
    }

    /// `Clip3(0, max, v)` as `i16` lanes, for `max < 2^15`. Saturating to
    /// `i16` first does not change the result.
    #[cfg(not(target_arch = "aarch64"))]
    #[inline(always)]
    fn clip_px(v: I32s, max: I16s) -> I16s {
        narrow_sat(v).simd_max(I16s::splat(0)).simd_min(max)
    }

    /// [`clip_px`] of a vector in [`madd_p`] order, giving natural order.
    #[cfg(not(target_arch = "aarch64"))]
    #[inline(always)]
    fn clip_px_p(v: I32s, max: I16s) -> I16s {
        narrow_sat_p(v).simd_max(I16s::splat(0)).simd_min(max)
    }

    /// Loads the 7 overlapping `N`-lane windows of a short (`< N + 6`) tail
    /// slice `src_tail`, zero-filling lanes past `src_tail.len()`.
    #[inline(never)]
    fn load_wiener_tail<T: Pixel>(src_tail: &[T]) -> [I16s; 7] {
        let mut buf = [0i16; N + 6];
        for (d, s) in buf.iter_mut().zip(src_tail) {
            *d = s.to_i32() as i16;
        }
        std::array::from_fn(|t| I16s::from_slice(&buf[t..t + N]))
    }

    /// Per-thread row buffers of the narrow kernels, zeroed once per thread and
    /// reused across restoration units (a fresh zeroed set per unit costs a
    /// 6 KB / 16 KB memset). Every row element a kernel reads for columns `0..w`
    /// is written earlier in the same call, so stale contents are never used.
    struct Scratch {
        /// Wiener intermediate rows (see [`wiener_filter`]).
        ring: [RingRow; 8],
        /// Self guided weighted rows: `r = 2` pass (5, 6).
        p0: [WeightedRow; 2],
        /// Self guided weighted rows: `r = 1` pass (accumulated `3, 4` + `4, 4`).
        p343: [WeightedRow; 3],
        /// Self guided vertical box sums (3 and 5 rows) and raw `A`/`B` row.
        bs3: BoxSums,
        bs5: BoxSums,
        ab: AbRow,
    }

    impl Scratch {
        #[cold]
        #[inline(never)]
        fn new_boxed() -> Box<Self> {
            const R: WeightedRow = WeightedRow::new();
            Box::new(Self {
                ring: [[I16s::from_array([0; N]); LR_CHUNKS]; 8],
                p0: [R; 2],
                p343: [R; 3],
                bs3: BoxSums::new(),
                bs5: BoxSums::new(),
                ab: AbRow::new(),
            })
        }
    }

    thread_local! {
        static SCRATCH: Cell<Option<Box<Scratch>>> = const { Cell::new(None) };
    }

    /// This thread's [`Scratch`], taken out of [`SCRATCH`] (or newly allocated)
    /// and put back on drop.
    struct ScratchGuard(Option<Box<Scratch>>);

    impl ScratchGuard {
        #[inline(always)]
        fn take() -> Self {
            let s = SCRATCH.try_with(Cell::take).ok().flatten();
            Self(Some(s.unwrap_or_else(Scratch::new_boxed)))
        }
    }

    impl std::ops::Deref for ScratchGuard {
        type Target = Scratch;
        #[inline(always)]
        fn deref(&self) -> &Scratch {
            self.0.as_ref().unwrap()
        }
    }

    impl std::ops::DerefMut for ScratchGuard {
        #[inline(always)]
        fn deref_mut(&mut self) -> &mut Scratch {
            self.0.as_mut().unwrap()
        }
    }

    impl Drop for ScratchGuard {
        #[inline(always)]
        fn drop(&mut self) {
            let s = self.0.take();
            let _ = SCRATCH.try_with(|c| c.set(s));
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    #[inline(always)]
    fn ld_row(row: &RingRow, k: usize) -> I16s {
        row[k]
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn load_u8x16<T: Pixel>(src: &[T]) -> Simd<u8, 16> {
        let mut a = [0u8; 16];
        for (d, s) in a.iter_mut().zip(&src[..16]) {
            *d = s.to_i32() as u8;
        }
        Simd::from_array(a)
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn add_u8_i16(a: Simd<u8, 16>, b: Simd<u8, 16>) -> I16s {
        (a.cast::<u16>() + b.cast::<u16>()).cast::<i16>()
    }

    /// Horizontal pass of [`wiener_filter`] for one padded source row `src_row`
    /// of length `w + 2 * LR_BORDER`.
    ///
    /// 8-bit uses 16-bit arithmetic (as dav1d): with `f3' = f3 - 128`,
    /// `m = f0 (p0 + p6) + f1 (p1 + p5) + f2 (p2 + p4) + f3' p3` equals
    /// `sum_t f_t (s_t - 2 p3)` (`f3' = -2 (f0 + f1 + f2)`), where
    /// `s_t - 2 p3` lies in `[-2 p3, 510 - 2 p3]`. With the tap ranges
    /// `f0 in [-5, 10]`, `f1 in [-23, 8]`, `f2 in [-17, 46]` this gives
    /// `|m| <= 510 * (10 + 8 + 46) = 32640`, so `m` is exact in wrapping
    /// `i16` arithmetic. The filter sum is `S = m + 128 p3`, and the
    /// stored value is `clamp(Round2(S, 3), -offset, limit - offset) - mid
    /// = clamp((S + 4 - 2^14) >> 3, -2^12, 2^12 - 1)` (offset = mid =
    /// 2048, limit = 8191). `S + 4 - 2^14` is formed with one saturating
    /// add of `m` and `128 p3 + 4 - 2^14` (in `[-16380, 16264]`): it is
    /// exact when the result is not clamped, and saturation to
    /// `[-2^15, 2^15 - 1]` gives exactly the clamp bounds after `>> 3`.
    /// 8-bit rows are stored plus [`WIENER8_BIAS`] (see
    /// [`wiener_vpass_row_u8`]).
    #[inline(always)]
    fn wiener_hpass_row<T: Pixel, const BD: u32, const TAP5: bool>(
        src_row: &[T],
        w: usize,
        hf: &[I16s; 4],
        out: &mut RingRow,
    ) {
        const ROUND0: u32 = 3;
        let chunks = (w / N).min(LR_CHUNKS);
        let c_tail = chunks * N;
        assert!(
            w <= LR_MAX_W
                && chunks <= LR_CHUNKS
                && src_row.len() >= w + 6
                && src_row.len() >= c_tail + 6
        );
        let src_chunks = &src_row[..c_tail + 6];
        if BD == 8 {
            let bias8 = I16s::splat(4 - (1 << 14));
            #[cfg(target_arch = "aarch64")]
            if T::MAX_BITDEPTH == 8 {
                for (k, out_slot) in out[..chunks].iter_mut().enumerate() {
                    let win = &src_chunks[k * N..k * N + N + 6];
                    let u1 = load_u8x16(&win[1..]);
                    let u2 = load_u8x16(&win[2..]);
                    let u3 = load_u8x16(&win[3..]);
                    let u4 = load_u8x16(&win[4..]);
                    let u5 = load_u8x16(&win[5..]);
                    let p3 = u3.cast::<u16>().cast::<i16>();
                    let m = if TAP5 {
                        add_u8_i16(u1, u5) * hf[1] + add_u8_i16(u2, u4) * hf[2] + p3 * hf[3]
                    } else {
                        let u0 = load_u8x16(&win[0..]);
                        let u6 = load_u8x16(&win[6..]);
                        add_u8_i16(u0, u6) * hf[0]
                            + add_u8_i16(u1, u5) * hf[1]
                            + add_u8_i16(u2, u4) * hf[2]
                            + p3 * hf[3]
                    };
                    let v = m.saturating_add((p3 << 7) + bias8) >> 3;
                    *out_slot = v + I16s::splat(WIENER8_BIAS);
                }
                if c_tail < w && chunks < LR_CHUNKS {
                    let [p0, p1, p2, p3, p4, p5, p6] = load_wiener_tail(&src_row[c_tail..]);
                    let m = if TAP5 {
                        (p1 + p5) * hf[1] + (p2 + p4) * hf[2] + p3 * hf[3]
                    } else {
                        (p0 + p6) * hf[0] + (p1 + p5) * hf[1] + (p2 + p4) * hf[2] + p3 * hf[3]
                    };
                    let v = m.saturating_add((p3 << 7) + bias8) >> 3;
                    out[chunks] = v + I16s::splat(WIENER8_BIAS);
                }
                return;
            }
            for (k, out_slot) in out[..chunks].iter_mut().enumerate() {
                let c = k * N;
                let win = &src_chunks[c..c + N + 6];
                let p1 = load_px(&win[1..]);
                let p2 = load_px(&win[2..]);
                let p3 = load_px(&win[3..]);
                let p4 = load_px(&win[4..]);
                let p5 = load_px(&win[5..]);
                let m = if TAP5 {
                    (p1 + p5) * hf[1] + (p2 + p4) * hf[2] + p3 * hf[3]
                } else {
                    let p0 = load_px(&win[0..]);
                    let p6 = load_px(&win[6..]);
                    (p0 + p6) * hf[0] + (p1 + p5) * hf[1] + (p2 + p4) * hf[2] + p3 * hf[3]
                };
                let v = m.saturating_add((p3 << 7) + bias8) >> 3;
                *out_slot = v + I16s::splat(WIENER8_BIAS);
            }
            if c_tail < w && chunks < LR_CHUNKS {
                let [p0, p1, p2, p3, p4, p5, p6] = load_wiener_tail(&src_row[c_tail..]);
                let m = if TAP5 {
                    (p1 + p5) * hf[1] + (p2 + p4) * hf[2] + p3 * hf[3]
                } else {
                    (p0 + p6) * hf[0] + (p1 + p5) * hf[1] + (p2 + p4) * hf[2] + p3 * hf[3]
                };
                let v = m.saturating_add((p3 << 7) + bias8) >> 3;
                out[chunks] = v + I16s::splat(WIENER8_BIAS);
            }
            return;
        }
        let offset = 1i32 << (BD + FILTER_BITS - ROUND0 - 1);
        let limit = (1i32 << (BD + 1 + FILTER_BITS - ROUND0)) - 1;
        let mid = (limit + 1) / 2 - offset;
        // Rounding minus the midpoint, a multiple of 4 (`mid << ROUND0`).
        let rnd0_s = (1 << (ROUND0 - 1)) - (mid << ROUND0);
        debug_assert_eq!(rnd0_s % 4, 0);
        let rnd0 = I32s::splat(rnd0_s);
        // For the 5-tap filter on x86, the rounding is folded into the second `madd`
        // as `4 * (rnd0 / 4)` (exact).
        #[cfg(not(target_arch = "aarch64"))]
        let four = I16s::splat(4);
        #[cfg(not(target_arch = "aarch64"))]
        let rnd0_q = I16s::splat((rnd0_s / 4) as i16);
        const {
            assert!(BD == 8 || BD == 10);
        }
        let hsum = |p0: I16s, p1: I16s, p2: I16s, p3: I16s, p4: I16s, p5: I16s, p6: I16s| -> I32s {
            #[cfg(target_arch = "aarch64")]
            {
                if TAP5 {
                    rnd0 + madd_p(p1 + p5, p2 + p4, hf[1], hf[2]) + mul_wide(p3, hf[3])
                } else {
                    rnd0 + madd_p(p0 + p6, p1 + p5, hf[0], hf[1])
                        + madd_p(p2 + p4, p3, hf[2], hf[3])
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                if TAP5 {
                    madd_p(p1 + p5, p2 + p4, hf[1], hf[2]) + madd_p(p3, four, hf[3], rnd0_q)
                } else {
                    madd_p(p0 + p6, p1 + p5, hf[0], hf[1])
                        + madd_p(p2 + p4, p3, hf[2], hf[3])
                        + rnd0
                }
            }
        };
        // For BD == 10, the clamp range `[-offset - mid, limit - offset - mid]`
        // is `[-2^14, 2^14 - 1]`, which is `[-2^15, 2^15 - 1] >> 1`, so
        // saturating `s >> (ROUND0 - 1)` to `i16` (`packssdw`) and shifting right
        // by 1 (`psraw $1`) performs both `>> ROUND0` and the clamp.
        for (k, out_slot) in out[..chunks].iter_mut().enumerate() {
            let c = k * N;
            let win = &src_chunks[c..c + N + 6];
            let p1 = load_px(&win[1..]);
            let p2 = load_px(&win[2..]);
            let p3 = load_px(&win[3..]);
            let p4 = load_px(&win[4..]);
            let p5 = load_px(&win[5..]);
            let (p0, p6) = if TAP5 { (p1, p5) } else { (load_px(&win[0..]), load_px(&win[6..])) };
            let s = hsum(p0, p1, p2, p3, p4, p5, p6);
            *out_slot = narrow_sat_p(s >> (ROUND0 as i32 - 1)) >> 1;
        }
        if c_tail < w && chunks < LR_CHUNKS {
            let [p0, p1, p2, p3, p4, p5, p6] = load_wiener_tail(&src_row[c_tail..]);
            let s = hsum(p0, p1, p2, p3, p4, p5, p6);
            out[chunks] = narrow_sat_p(s >> (ROUND0 as i32 - 1)) >> 1;
        }
    }

    /// Bias added to the 8-bit horizontal pass outputs: the midpoint
    /// `c = 2048` plus 8, so that with the vertical taps (summing to 128)
    /// the vertical pass sum includes `128 c + 2^10`, i.e. the add back of
    /// the midpoint and the rounding of `Round2(., 11)` (as dav1d). The
    /// biased values are in `[-2040, 6151]`, so pair sums fit `i16`.
    const WIENER8_BIAS: i16 = 2048 + 8;

    /// Stores `N` (`16`) `u8` pixels `v` to `dst[..N]`.
    #[inline(always)]
    fn store_px16_u8<T: Pixel>(dst: &mut [T], v: Simd<u8, N>) {
        for (d, s) in dst[..N].iter_mut().zip(v.to_array()) {
            *d = T::from_i32(s as i32);
        }
    }

    /// Like [`store_px16_u8`], for a tail slice `dst` shorter than `N`.
    #[inline(never)]
    fn store_px16_u8_tail<T: Pixel>(dst: &mut [T], v: Simd<u8, N>) {
        for (d, &s) in dst.iter_mut().zip(v.as_array()) {
            *d = T::from_i32(s as i32);
        }
    }

    /// `[a, b]` concatenation of two 16 lane vectors.
    #[cfg(not(target_arch = "aarch64"))]
    const CONCAT32: [usize; 32] = {
        let mut t = [0; 32];
        let mut i = 0;
        while i < 32 {
            t[i] = i;
            i += 1;
        }
        t
    };

    /// Stores 32 pixels `v` (`u8` lanes) to `dst[..32]`.
    #[cfg(not(target_arch = "aarch64"))]
    #[inline(always)]
    fn store_px32<T: Pixel>(dst: &mut [T], v: Simd<u8, 32>) {
        for (d, s) in dst[..32].iter_mut().zip(v.to_array()) {
            *d = T::from_i32(s as i32);
        }
    }

    /// Vertical pass of the 8-bit Wiener filter for one output row `out`
    /// (`w` pixels) from the 7 ring rows `rows` (biased by
    /// [`WIENER8_BIAS`]), with the taps `vf32` scaled by 32: the sum is then
    /// `2^16` times the spec sum before `Round2(., 11)`, so the pixel is
    /// `clamp(sum >> 16, 0, 255)` (dav1d's `packuswb` / `psrlw` layout; here
    /// `psrad` + `packssdw` + `packuswb`).
    #[inline(always)]
    fn wiener_vpass_row_u8<T: Pixel, const TAP5: bool>(
        rows: [&RingRow; 7],
        vf32: &[I16s; 4],
        out: &mut [T],
        w: usize,
    ) {
        assert!(w <= LR_MAX_W && out.len() >= w);
        let [r0, r1, r2, r3, r4, r5, r6] = rows;
        // On aarch64, 16 pixels per iteration (`shrn #16` + `sqxtun` + 16-byte `str q`)
        // keeps all 7 ring row loads, 4 tap vectors, and accumulators within the 32
        // NEON registers, and avoids both the x86 `vf32[0] == 0` dummy multiply and
        // per-iteration slice bounds checks.
        #[cfg(target_arch = "aarch64")]
        {
            let chunks = (w / N).min(LR_CHUNKS);
            let eval = |v0: I16s,
                        v1: I16s,
                        v2: I16s,
                        v3: I16s,
                        v4: I16s,
                        v5: I16s,
                        v6: I16s|
             -> Simd<u8, N> {
                let s = if TAP5 {
                    madd_p(v1 + v5, v2 + v4, vf32[1], vf32[2]) + mul_wide(v3, vf32[3])
                } else {
                    madd_p(v0 + v6, v1 + v5, vf32[0], vf32[1])
                        + madd_p(v2 + v4, v3, vf32[2], vf32[3])
                };
                // `s >> 16` is bounded in `[-926, 925]`, strictly inside `i16`.
                (s >> 16)
                    .cast::<i16>()
                    .simd_clamp(I16s::splat(0), I16s::splat(255))
                    .cast::<u8>()
            };
            if TAP5 {
                for (((((&v1, &v2), &v3), &v4), &v5), o) in r1[..chunks]
                    .iter()
                    .zip(&r2[..chunks])
                    .zip(&r3[..chunks])
                    .zip(&r4[..chunks])
                    .zip(&r5[..chunks])
                    .zip(out[..chunks * N].chunks_exact_mut(N))
                {
                    store_px16_u8(o, eval(v1, v1, v2, v3, v4, v5, v5));
                }
            } else {
                for (((((((&v0, &v1), &v2), &v3), &v4), &v5), &v6), o) in r0[..chunks]
                    .iter()
                    .zip(&r1[..chunks])
                    .zip(&r2[..chunks])
                    .zip(&r3[..chunks])
                    .zip(&r4[..chunks])
                    .zip(&r5[..chunks])
                    .zip(&r6[..chunks])
                    .zip(out[..chunks * N].chunks_exact_mut(N))
                {
                    store_px16_u8(o, eval(v0, v1, v2, v3, v4, v5, v6));
                }
            }
            let c = chunks * N;
            if c < w && chunks < LR_CHUNKS {
                let u = eval(
                    r0[chunks], r1[chunks], r2[chunks], r3[chunks], r4[chunks], r5[chunks],
                    r6[chunks],
                );
                store_px16_u8_tail(&mut out[c..w], u);
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            // `sum >> 16` of chunk `k` as `i16` lanes (natural order).
            let px = |k: usize| -> I16s {
                let s = if TAP5 {
                    // The 5-tap filter has `vf32[0] == 0` (a runtime value, so that
                    // LLVM keeps a plain `pmaddwd` instead of a widening multiply).
                    madd_p(r1[k] + r5[k], r2[k] + r4[k], vf32[1], vf32[2])
                        + madd_p(r3[k], r3[k], vf32[3], vf32[0])
                } else {
                    madd_p(r0[k] + r6[k], r1[k] + r5[k], vf32[0], vf32[1])
                        + madd_p(r2[k] + r4[k], r3[k], vf32[2], vf32[3])
                };
                narrow_sat_p(s >> 16)
            };
            let zero = I16s::splat(0);
            let to_u8 = |a: I16s, b: I16s| -> Simd<u8, 32> {
                let v: Simd<i16, 32> = simd_swizzle!(a, b, CONCAT32);
                v.simd_clamp(Simd::splat(0), Simd::splat(255)).cast::<u8>()
            };
            let pairs = w / (2 * N);
            for (j, o) in out[..pairs * 2 * N].chunks_exact_mut(2 * N).enumerate() {
                store_px32(o, to_u8(px(2 * j), px(2 * j + 1)));
            }
            let c = pairs * 2 * N;
            if c < w {
                let k = 2 * pairs;
                let b = if c + N < w { px(k + 1) } else { zero };
                store_px32_tail(&mut out[c..w], to_u8(px(k), b));
            }
        }
    }

    /// Like [`store_px32`], for a tail slice `dst` shorter than 32.
    #[cfg(not(target_arch = "aarch64"))]
    #[inline(never)]
    fn store_px32_tail<T: Pixel>(dst: &mut [T], v: Simd<u8, 32>) {
        for (d, &s) in dst.iter_mut().zip(v.as_array()) {
            *d = T::from_i32(s as i32);
        }
    }

    /// Wiener filter for 8 and 10-bit, same contract as
    /// [`crate::decoder::av1::dsp::lr::wiener_filter`].
    ///
    /// The 7-tap filters are symmetric, so both passes add the sample pairs
    /// before multiplying. The horizontal pass output is clamped to
    /// `[-offset, limit - offset]`, a range of `limit + 1 = 2^(bd + 5)`
    /// values (round0 = 3), and stored minus its midpoint
    /// `c = (limit + 1) / 2 - offset` as `i16` in `[-2^(bd + 4), 2^(bd + 4))`,
    /// so that the pair sums of the vertical pass (`[-2^15, 2^15 - 2]` for
    /// 10-bit) still fit `i16`. As the vertical taps sum to 128, the
    /// vertical pass adds back `128 * c`.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn wiener_filter<T: Pixel, const BD: u32>(
        dst: &mut [T],
        dst_stride: usize,
        src: &[&[T]],
        w: usize,
        h: usize,
        coeffs: &[[i32; 3]; 2],
    ) {
        check_rect(w, h);
        const { assert!(BD == 8 || BD == 10) };
        let vfilter = wiener_coefficients(&coeffs[0]);
        let hfilter = wiener_coefficients(&coeffs[1]);
        // Constant shifts (immediate operands) for 8 and 10-bit.
        const ROUND0: u32 = 3;
        const ROUND1: u32 = 11;
        debug_assert_eq!(wiener_rounding(BD), (ROUND0, ROUND1));
        let offset = 1i32 << (BD + FILTER_BITS - ROUND0 - 1);
        let limit = (1i32 << (BD + 1 + FILTER_BITS - ROUND0)) - 1;
        let mid = (limit + 1) / 2 - offset;
        let max_px = (1i32 << BD) - 1;

        if BD == 8 {
            debug_assert!(coeffs[1][0] >= -5 && coeffs[1][0] <= 10);
            debug_assert!(coeffs[1][1] >= -23 && coeffs[1][1] <= 8);
            debug_assert!(coeffs[1][2] >= -17 && coeffs[1][2] <= 46);
        }
        // Taps are in [-23, 218]; 8-bit horizontal pass uses f3' = f3 - 128.
        let hf: [I16s; 4] = [
            I16s::splat(hfilter[0] as i16),
            I16s::splat(hfilter[1] as i16),
            I16s::splat(hfilter[2] as i16),
            I16s::splat((if BD == 8 { hfilter[3] - 128 } else { hfilter[3] }) as i16),
        ];
        let vf: [I16s; 4] = [
            I16s::splat(vfilter[0] as i16),
            I16s::splat(vfilter[1] as i16),
            I16s::splat(vfilter[2] as i16),
            I16s::splat(vfilter[3] as i16),
        ];
        // 8-bit vertical taps scaled by 32 (`|32 f| <= 32 * 218 < 2^15`).
        let vf32: [I16s; 4] = std::array::from_fn(|t| I16s::splat((vfilter[t] * 32) as i16));
        let rnd1_s = (1 << (ROUND1 - 1)) + (mid << FILTER_BITS);
        #[cfg(not(target_arch = "aarch64"))]
        let rnd1 = I32s::splat(rnd1_s);
        // `rnd1` is a multiple of 256 (`mid` is even), for the 5-tap fold on x86.
        debug_assert_eq!(rnd1_s % 256, 0);
        #[cfg(not(target_arch = "aarch64"))]
        let k256 = I16s::splat(256);
        #[cfg(not(target_arch = "aarch64"))]
        let rnd1_q = I16s::splat((rnd1_s / 256) as i16);
        #[cfg(target_arch = "aarch64")]
        let mid_bias = I32s::splat(mid << FILTER_BITS);
        let max_px = I16s::splat(max_px as i16);
        let swidth = w + 2 * LR_BORDER;
        let h5 = coeffs[1][0] == 0;
        // Power-of-two ring of 8 intermediate rows (`r & 7` indexing avoids
        // modulo-7 / modulo-5 divisions).
        let mut scratch = ScratchGuard::take();
        let ring = &mut scratch.ring;

        // 5-tap vertical fast path (all chroma Wiener filters and many luma units
        // have coeffs[0][0] == 0): only h + 4 rows of the horizontal pass are needed.
        #[inline(always)]
        fn run_v5<T: Pixel, const BD: u32, const H5: bool>(
            dst: &mut [T],
            dst_stride: usize,
            src: &[&[T]],
            w: usize,
            h: usize,
            swidth: usize,
            hf: &[I16s; 4],
            vf: &[I16s; 4],
            vf32: &[I16s; 4],
            #[cfg(not(target_arch = "aarch64"))] k256: I16s,
            #[cfg(not(target_arch = "aarch64"))] rnd1_q: I16s,
            #[cfg(target_arch = "aarch64")] mid_bias: I32s,
            max_px: I16s,
            ring: &mut [RingRow; 8],
        ) {
            const ROUND1: u32 = 11;
            for r in 1..5 {
                wiener_hpass_row::<T, BD, H5>(&src[r][..swidth], w, hf, &mut ring[r & 7]);
            }
            for r in 0..h {
                wiener_hpass_row::<T, BD, H5>(&src[r + 5][..swidth], w, hf, &mut ring[(r + 5) & 7]);
                let r1 = &ring[(r + 1) & 7];
                let r2 = &ring[(r + 2) & 7];
                let r3 = &ring[(r + 3) & 7];
                let r4 = &ring[(r + 4) & 7];
                let r5 = &ring[(r + 5) & 7];
                let out = &mut dst[r * dst_stride..r * dst_stride + w];
                if BD == 8 {
                    wiener_vpass_row_u8::<T, true>([r1, r1, r2, r3, r4, r5, r5], vf32, out, w);
                    continue;
                }
                #[cfg(target_arch = "aarch64")]
                {
                    // For BD == 10, `(S + 1024) >> 11` lies in `[-1952, 2976]`, strictly inside
                    // `i16`: starting at `mid_bias` lowers the whole sum to `smlal`/`smlal2`,
                    // and `((s + 1024) >> 11).cast::<i16>()` lowers to `rshrn #11` + `rshrn2 #11`.
                    let rnd_11 = I32s::splat(1 << (ROUND1 - 1));
                    let chunks = (w / N).min(LR_CHUNKS);
                    let eval = |v1: I16s, v2: I16s, v3: I16s, v4: I16s, v5: I16s| -> I16s {
                        let s =
                            mid_bias + madd_p(v1 + v5, v2 + v4, vf[1], vf[2]) + mul_wide(v3, vf[3]);
                        ((s + rnd_11) >> ROUND1 as i32)
                            .cast::<i16>()
                            .simd_max(I16s::splat(0))
                            .simd_min(max_px)
                    };
                    for (((((&v1, &v2), &v3), &v4), &v5), o) in r1[..chunks]
                        .iter()
                        .zip(&r2[..chunks])
                        .zip(&r3[..chunks])
                        .zip(&r4[..chunks])
                        .zip(&r5[..chunks])
                        .zip(out[..chunks * N].chunks_exact_mut(N))
                    {
                        store_px(o, eval(v1, v2, v3, v4, v5));
                    }
                    let c = chunks * N;
                    if c < w && chunks < LR_CHUNKS {
                        let v = eval(r1[chunks], r2[chunks], r3[chunks], r4[chunks], r5[chunks]);
                        store_px_tail(&mut out[c..], v);
                    }
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    // Rounding folded into the second `madd` as 256 * (rnd1 / 256).
                    let vsum = |k: usize| {
                        madd_p(
                            ld_row(r1, k) + ld_row(r5, k),
                            ld_row(r2, k) + ld_row(r4, k),
                            vf[1],
                            vf[2],
                        ) + madd_p(ld_row(r3, k), k256, vf[3], rnd1_q)
                    };
                    for k in 0..w / N {
                        let c = k * N;
                        let v = clip_px_p(vsum(k) >> ROUND1 as i32, max_px);
                        store_px(&mut out[c..], v);
                    }
                    let (k, c) = (w / N, w / N * N);
                    if c < w {
                        let v = clip_px_p(vsum(k) >> ROUND1 as i32, max_px);
                        store_px_tail(&mut out[c..], v);
                    }
                }
            }
        }

        #[inline(always)]
        fn run_v7<T: Pixel, const BD: u32, const H5: bool>(
            dst: &mut [T],
            dst_stride: usize,
            src: &[&[T]],
            w: usize,
            h: usize,
            swidth: usize,
            hf: &[I16s; 4],
            vf: &[I16s; 4],
            vf32: &[I16s; 4],
            #[cfg(not(target_arch = "aarch64"))] rnd1: I32s,
            #[cfg(target_arch = "aarch64")] mid_bias: I32s,
            max_px: I16s,
            ring: &mut [RingRow; 8],
        ) {
            const ROUND1: u32 = 11;
            for (r, slot) in ring.iter_mut().enumerate().take(6) {
                wiener_hpass_row::<T, BD, H5>(&src[r][..swidth], w, hf, slot);
            }
            for r in 0..h {
                wiener_hpass_row::<T, BD, H5>(&src[r + 6][..swidth], w, hf, &mut ring[(r + 6) & 7]);
                let [r0, r1, r2, r3, r4, r5, r6]: [&RingRow; 7] =
                    std::array::from_fn(|t| &ring[(r + t) & 7]);
                let out = &mut dst[r * dst_stride..r * dst_stride + w];
                if BD == 8 {
                    wiener_vpass_row_u8::<T, false>([r0, r1, r2, r3, r4, r5, r6], vf32, out, w);
                    continue;
                }
                #[cfg(target_arch = "aarch64")]
                {
                    let rnd_11 = I32s::splat(1 << (ROUND1 - 1));
                    let chunks = (w / N).min(LR_CHUNKS);
                    let eval = |v0: I16s,
                                v1: I16s,
                                v2: I16s,
                                v3: I16s,
                                v4: I16s,
                                v5: I16s,
                                v6: I16s|
                     -> I16s {
                        let s = mid_bias
                            + madd_p(v0 + v6, v1 + v5, vf[0], vf[1])
                            + madd_p(v2 + v4, v3, vf[2], vf[3]);
                        ((s + rnd_11) >> ROUND1 as i32)
                            .cast::<i16>()
                            .simd_max(I16s::splat(0))
                            .simd_min(max_px)
                    };
                    for (((((((&v0, &v1), &v2), &v3), &v4), &v5), &v6), o) in r0[..chunks]
                        .iter()
                        .zip(&r1[..chunks])
                        .zip(&r2[..chunks])
                        .zip(&r3[..chunks])
                        .zip(&r4[..chunks])
                        .zip(&r5[..chunks])
                        .zip(&r6[..chunks])
                        .zip(out[..chunks * N].chunks_exact_mut(N))
                    {
                        store_px(o, eval(v0, v1, v2, v3, v4, v5, v6));
                    }
                    let c = chunks * N;
                    if c < w && chunks < LR_CHUNKS {
                        let v = eval(
                            r0[chunks], r1[chunks], r2[chunks], r3[chunks], r4[chunks], r5[chunks],
                            r6[chunks],
                        );
                        store_px_tail(&mut out[c..], v);
                    }
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    let vsum = |k: usize| {
                        madd_p(
                            ld_row(r0, k) + ld_row(r6, k),
                            ld_row(r1, k) + ld_row(r5, k),
                            vf[0],
                            vf[1],
                        ) + madd_p(ld_row(r2, k) + ld_row(r4, k), ld_row(r3, k), vf[2], vf[3])
                            + rnd1
                    };
                    for k in 0..w / N {
                        let c = k * N;
                        let v = clip_px_p(vsum(k) >> ROUND1 as i32, max_px);
                        store_px(&mut out[c..], v);
                    }
                    let (k, c) = (w / N, w / N * N);
                    if c < w {
                        let v = clip_px_p(vsum(k) >> ROUND1 as i32, max_px);
                        store_px_tail(&mut out[c..], v);
                    }
                }
            }
        }

        match (coeffs[0][0] == 0, h5) {
            (true, true) => run_v5::<T, BD, true>(
                dst,
                dst_stride,
                src,
                w,
                h,
                swidth,
                &hf,
                &vf,
                &vf32,
                #[cfg(not(target_arch = "aarch64"))]
                k256,
                #[cfg(not(target_arch = "aarch64"))]
                rnd1_q,
                #[cfg(target_arch = "aarch64")]
                mid_bias,
                max_px,
                ring,
            ),
            (true, false) => run_v5::<T, BD, false>(
                dst,
                dst_stride,
                src,
                w,
                h,
                swidth,
                &hf,
                &vf,
                &vf32,
                #[cfg(not(target_arch = "aarch64"))]
                k256,
                #[cfg(not(target_arch = "aarch64"))]
                rnd1_q,
                #[cfg(target_arch = "aarch64")]
                mid_bias,
                max_px,
                ring,
            ),
            (false, true) => run_v7::<T, BD, true>(
                dst,
                dst_stride,
                src,
                w,
                h,
                swidth,
                &hf,
                &vf,
                &vf32,
                #[cfg(not(target_arch = "aarch64"))]
                rnd1,
                #[cfg(target_arch = "aarch64")]
                mid_bias,
                max_px,
                ring,
            ),
            (false, false) => run_v7::<T, BD, false>(
                dst,
                dst_stride,
                src,
                w,
                h,
                swidth,
                &hf,
                &vf,
                &vf32,
                #[cfg(not(target_arch = "aarch64"))]
                rnd1,
                #[cfg(target_arch = "aarch64")]
                mid_bias,
                max_px,
                ring,
            ),
        }
    }

    /// Length of a raw `A`/`B` row: rectangle columns `-1..=w` rounded up
    /// to whole chunks (this also covers the 3-wide neighbourhood loads of
    /// the last output chunk, which end at `LR_MAX_W + 2`).
    const AB_LEN: usize = (LR_MAX_W + 2).next_multiple_of(N);
    /// Length of a box sum row: `AB_LEN` plus `N` extra lanes so 5-wide
    /// window loads in [`ab_from_vsum`] (including `&bs3.sum[1..]`) stay in
    /// bounds up to `j = LR_MAX_W`.
    const SUM_LEN: usize = AB_LEN + N;

    /// Vertical box sums of `c` and `c * c` for one row of windows.
    struct BoxSums {
        sum: [i16; SUM_LEN],
        sq: [i32; SUM_LEN],
    }

    impl BoxSums {
        const fn new() -> Self {
            Self {
                sum: [0; SUM_LEN],
                sq: [0; SUM_LEN],
            }
        }
    }

    /// One raw row of `X = 256 - A` and `B` (spec 7.17.3) for rectangle columns
    /// `-1..=w` (index `j = col + 1`). On aarch64 `x = 256 - A2(z)` in `0..=255`
    /// is stored directly as `u8`, halving L1D traffic and enabling 16-lane `ldr q`
    /// + `uaddl`/`uaddw` horizontal weighting.
    struct AbRow {
        #[cfg(target_arch = "aarch64")]
        a: [u8; AB_LEN],
        #[cfg(not(target_arch = "aarch64"))]
        a: [i16; AB_LEN],
        b: [i32; AB_LEN],
    }

    impl AbRow {
        const fn new() -> Self {
            Self {
                a: [0; AB_LEN],
                b: [0; AB_LEN],
            }
        }
    }

    /// One row of horizontally weighted `A`/`B` values for rectangle columns
    /// `0..w`, `x[j] = k0 * (raw[j] + raw[j + 2]) + k1 * raw[j + 1]`:
    /// 5, 6 (r = 2 pass), 3, 4 or 4, 4 (r = 1 pass). All `A` sums are at
    /// most `16 * 256`.
    struct WeightedRow {
        a: [I16s; LR_CHUNKS],
        b: [I32s; LR_CHUNKS],
    }

    impl WeightedRow {
        const fn new() -> Self {
            Self {
                a: [I16s::from_array([0; N]); LR_CHUNKS],
                b: [I32s::from_array([0; N]); LR_CHUNKS],
            }
        }

        #[inline(always)]
        fn ld_a(&self, k: usize) -> I16s {
            self.a[k]
        }

        #[inline(always)]
        fn ld_b(&self, k: usize) -> I32s {
            self.b[k]
        }
    }

    /// `x * K` for a small constant `K`, with shifts and adds.
    #[inline(always)]
    fn mul_const<const K: i32>(x: I32s) -> I32s {
        match K {
            3 => (x << 1) + x,
            4 => x << 2,
            5 => (x << 2) + x,
            6 => (x << 2) + (x << 1),
            9 => (x << 3) + x,
            #[cfg(not(target_arch = "aarch64"))]
            25 => (x << 4) + (x << 3) + x,
            _ => x * I32s::splat(K),
        }
    }

    /// 4-lane [`mul_const`].
    #[cfg(not(target_arch = "aarch64"))]
    #[inline(always)]
    fn mul_const4<const K: i32>(x: I32x4) -> I32x4 {
        match K {
            9 => (x << 3) + x,
            25 => (x << 4) + (x << 3) + x,
            _ => x * I32x4::splat(K),
        }
    }

    /// `X[z] = 256 - SGR_A2[z]`, in `[0, 255]` and non-increasing.
    #[cfg(target_arch = "aarch64")]
    const SGR_X: [u8; 256] = {
        let mut t = [0u8; 256];
        let mut z = 0;
        while z < 256 {
            t[z] = (256 - crate::decoder::av1::dsp::lr::SGR_A2[z]) as u8;
            z += 1;
        }
        t
    };
    /// Number of leading `SGR_X` entries found with table lookups.
    #[cfg(target_arch = "aarch64")]
    const SGR_X_TBL: usize = 48;
    /// `SGR_X[z]` for `z >= SGR_X_TBL` equals the number of `k` with
    /// `z <= SGR_X_LAST[k]`, where `SGR_X_LAST[v - 1]` is the last index
    /// with `SGR_X[z] >= v`.
    #[cfg(target_arch = "aarch64")]
    const SGR_X_STEPS: usize = SGR_X[SGR_X_TBL] as usize;
    #[cfg(target_arch = "aarch64")]
    const SGR_X_LAST: [u8; SGR_X_STEPS] = {
        let mut t = [0u8; SGR_X_STEPS];
        let mut v = 1;
        while v <= SGR_X_STEPS {
            let mut z = SGR_X_TBL;
            while z < 255 && SGR_X[z + 1] as usize >= v {
                z += 1;
            }
            t[v - 1] = z as u8;
            v += 1;
        }
        t
    };
    /// The table part of `SGR_X`, biased by `-SGR_X_STEPS`, in 16-entry chunks.
    /// For `z < SGR_X_TBL`, all `SGR_X_STEPS` step comparisons (`z <= last`) are
    /// true and add `SGR_X_STEPS` back; for `z >= SGR_X_TBL`, `swizzle_dyn`
    /// returns 0 and only the active steps contribute.
    #[cfg(target_arch = "aarch64")]
    const SGR_X_LUT: [[u8; 16]; SGR_X_TBL / 16] = {
        let mut t = [[0u8; 16]; SGR_X_TBL / 16];
        let mut z = 0;
        while z < SGR_X_TBL {
            t[z / 16][z % 16] = SGR_X[z] - SGR_X_STEPS as u8;
            z += 1;
        }
        t
    };

    /// `256 - SGR_A2[z]` for `z <= 255`: on aarch64 (dav1d's `sgr_x_by_x`),
    /// the first `SGR_X_TBL` entries use 16-byte `tbl` lookups (`swizzle_dyn`,
    /// where an out-of-range index gives 0) and the rest count steps.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub(super) fn sgr_x_u8(z8: Simd<u8, N>) -> Simd<u8, N> {
        const { assert!(SGR_X_LUT.len() == 3 && SGR_X_STEPS == 5) };
        let s16 = Simd::splat(16);
        let z16 = z8 - s16;
        let t0 = Simd::<u8, N>::from_array(SGR_X_LUT[0]).swizzle_dyn(z8);
        let t1 = Simd::<u8, N>::from_array(SGR_X_LUT[1]).swizzle_dyn(z16);
        let t2 = Simd::<u8, N>::from_array(SGR_X_LUT[2]).swizzle_dyn(z16 - s16);
        // `to_simd()` is -1 (0xff) when true, 0 when false; summing the 5 masks
        // in a balanced tree and subtracting adds the count of active steps.
        let m = |k: usize| {
            z8.simd_le(Simd::splat(SGR_X_LAST[k]))
                .to_simd()
                .cast::<u8>()
        };
        let neg_steps = (m(0) + m(1)) + (m(2) + m(3)) + m(4);
        ((t0 | t1) | t2) - neg_steps
    }

    #[cfg(all(test, not(target_arch = "aarch64")))]
    #[inline(always)]
    fn sgr_x_u16(z: I32s) -> I16s {
        let z16 = z.cast::<i16>().cast::<u16>();
        let d = z16 + U16s::splat(1);
        let s0 = (d << 7) - U16s::splat(1);
        let s1 = s0.saturating_add(s0);
        let s2 = s1.saturating_add(s1);
        let s3 = s2.saturating_add(s2);
        let s4 = s3.saturating_add(s3);
        let s5 = s4.saturating_add(s4);
        let s6 = s5.saturating_add(s5);
        let mut acc = (U16s::splat(256) + (d >> 1)) << 7;
        for s in [s6, s5, s4, s3, s2, s1, s0] {
            acc = acc.simd_min(acc - s);
        }
        let q = (acc & U16s::splat(127)) + U16s::splat(255).saturating_sub(s0 - z16) - (d >> 8);
        q.cast::<i16>()
    }

    #[cfg(test)]
    pub(super) fn sgr_x(z: I32s) -> I32s {
        #[cfg(target_arch = "aarch64")]
        {
            sgr_x_u8(z.cast()).cast()
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            sgr_x_u16(z).cast()
        }
    }

    #[cfg(test)]
    pub(super) fn sgr_x_from_p(p: I32s, s: u32) -> I32s {
        XFromP::new(s)
            .x::<10>(p.simd_max(I32s::splat(-65536)))
            .cast()
    }

    /// Initializes `bs` with the vertical sum of `c` and `c * c` over the 3
    /// rows starting at `row0` (`col0 .. col0 + ncols`).
    #[inline(always)]
    fn vsum_init3<T: Pixel, const TAIL: bool>(
        src: &[&[T]],
        row0: usize,
        col0: usize,
        ncols: usize,
        bs: &mut BoxSums,
    ) {
        let r0 = &src[row0][col0..col0 + ncols];
        let r1 = &src[row0 + 1][col0..col0 + ncols];
        let r2 = &src[row0 + 2][col0..col0 + ncols];
        let mut c = 0;
        while c + N <= ncols {
            let (p0, p1, p2) = (load_px(&r0[c..]), load_px(&r1[c..]), load_px(&r2[c..]));
            let sum = p0 + p1 + p2;
            let sq = madd(p0, p1, p0, p1) + mul_wide(p2, p2);
            sum.copy_to_slice(&mut bs.sum[c..c + N]);
            sq.copy_to_slice(&mut bs.sq[c..c + N]);
            c += N;
        }
        if c + super::N <= ncols {
            let (p0, p1, p2) = (
                super::load_px(&r0[c..]),
                super::load_px(&r1[c..]),
                super::load_px(&r2[c..]),
            );
            let sum = p0 + p1 + p2;
            let sq = super::madd(p0, p1, p0, p1) + super::mul_wide(p2, p2);
            sum.copy_to_slice(&mut bs.sum[c..c + super::N]);
            sq.copy_to_slice(&mut bs.sq[c..c + super::N]);
            c += super::N;
        }
        if TAIL && c < ncols {
            let (p0, p1, p2) = (
                load_px_or_zero(&r0[c..]),
                load_px_or_zero(&r1[c..]),
                load_px_or_zero(&r2[c..]),
            );
            let sum = p0 + p1 + p2;
            let sq = madd(p0, p1, p0, p1) + mul_wide(p2, p2);
            sum.copy_to_slice(&mut bs.sum[c..c + N]);
            sq.copy_to_slice(&mut bs.sq[c..c + N]);
        }
    }

    /// Slides a vertical sum `bs` by 1 row: subtracts `row_old` and adds
    /// `row_new` using `p_new^2 - p_old^2 = (p_new + p_old) * (p_new - p_old)`.
    #[inline(always)]
    fn vsum_slide1<T: Pixel, const TAIL: bool>(
        src: &[&[T]],
        row_old: usize,
        row_new: usize,
        col0: usize,
        ncols: usize,
        bs: &mut BoxSums,
    ) {
        let ro = &src[row_old][col0..col0 + ncols];
        let rn = &src[row_new][col0..col0 + ncols];
        let mut c = 0;
        while c + N <= ncols {
            let (po, pn) = (load_px(&ro[c..]), load_px(&rn[c..]));
            let diff = pn - po;
            let sum = I16s::from_slice(&bs.sum[c..c + N]) + diff;
            let sq = I32s::from_slice(&bs.sq[c..c + N]) + mul_wide(pn + po, diff);
            sum.copy_to_slice(&mut bs.sum[c..c + N]);
            sq.copy_to_slice(&mut bs.sq[c..c + N]);
            c += N;
        }
        if c + super::N <= ncols {
            let (po, pn) = (super::load_px(&ro[c..]), super::load_px(&rn[c..]));
            let diff = pn - po;
            let sum = super::I16s::from_slice(&bs.sum[c..c + super::N]) + diff;
            let sq =
                super::I32s::from_slice(&bs.sq[c..c + super::N]) + super::mul_wide(pn + po, diff);
            sum.copy_to_slice(&mut bs.sum[c..c + super::N]);
            sq.copy_to_slice(&mut bs.sq[c..c + super::N]);
            c += super::N;
        }
        if TAIL && c < ncols {
            let (po, pn) = (load_px_or_zero(&ro[c..]), load_px_or_zero(&rn[c..]));
            let diff = pn - po;
            let sum = I16s::from_slice(&bs.sum[c..c + N]) + diff;
            let sq = I32s::from_slice(&bs.sq[c..c + N]) + mul_wide(pn + po, diff);
            sum.copy_to_slice(&mut bs.sum[c..c + N]);
            sq.copy_to_slice(&mut bs.sq[c..c + N]);
        }
    }

    /// Simultaneously forms the 5-row vertical sum `bs5 = bs3 + row_new1 + row_new2`
    /// and slides `bs3` by 1 row (`bs3 += row_new1 - row_old`).
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn vsum_slide1_and_5<T: Pixel, const TAIL: bool>(
        src: &[&[T]],
        row_old: usize,
        row_new1: usize,
        row_new2: usize,
        ncols: usize,
        bs3: &mut BoxSums,
        bs5: &mut BoxSums,
    ) {
        let ro = &src[row_old][..ncols];
        let rn1 = &src[row_new1][..ncols];
        let rn2 = &src[row_new2][..ncols];
        let mut c = 0;
        while c + N <= ncols {
            let (po, p1, p2) = (load_px(&ro[c..]), load_px(&rn1[c..]), load_px(&rn2[c..]));
            let s3 = I16s::from_slice(&bs3.sum[c..c + N]);
            let q3 = I32s::from_slice(&bs3.sq[c..c + N]);
            let diff = p1 - po;
            (s3 + diff).copy_to_slice(&mut bs3.sum[c..c + N]);
            (q3 + mul_wide(p1 + po, diff)).copy_to_slice(&mut bs3.sq[c..c + N]);
            (s3 + p1 + p2).copy_to_slice(&mut bs5.sum[c..c + N]);
            (q3 + madd(p1, p2, p1, p2)).copy_to_slice(&mut bs5.sq[c..c + N]);
            c += N;
        }
        if c + super::N <= ncols {
            let (po, p1, p2) = (
                super::load_px(&ro[c..]),
                super::load_px(&rn1[c..]),
                super::load_px(&rn2[c..]),
            );
            let s3 = super::I16s::from_slice(&bs3.sum[c..c + super::N]);
            let q3 = super::I32s::from_slice(&bs3.sq[c..c + super::N]);
            let diff = p1 - po;
            (s3 + diff).copy_to_slice(&mut bs3.sum[c..c + super::N]);
            (q3 + super::mul_wide(p1 + po, diff)).copy_to_slice(&mut bs3.sq[c..c + super::N]);
            (s3 + p1 + p2).copy_to_slice(&mut bs5.sum[c..c + super::N]);
            (q3 + super::madd(p1, p2, p1, p2)).copy_to_slice(&mut bs5.sq[c..c + super::N]);
            c += super::N;
        }
        if TAIL && c < ncols {
            let (po, p1, p2) = (
                load_px_or_zero(&ro[c..]),
                load_px_or_zero(&rn1[c..]),
                load_px_or_zero(&rn2[c..]),
            );
            let s3 = I16s::from_slice(&bs3.sum[c..c + N]);
            let q3 = I32s::from_slice(&bs3.sq[c..c + N]);
            let diff = p1 - po;
            (s3 + diff).copy_to_slice(&mut bs3.sum[c..c + N]);
            (q3 + mul_wide(p1 + po, diff)).copy_to_slice(&mut bs3.sq[c..c + N]);
            (s3 + p1 + p2).copy_to_slice(&mut bs5.sum[c..c + N]);
            (q3 + madd(p1, p2, p1, p2)).copy_to_slice(&mut bs5.sq[c..c + N]);
        }
    }

    /// Adds two outer source rows `row_a` and `row_b` to `bs3` into `bs5`.
    #[inline(always)]
    fn vsum_add2<T: Pixel, const TAIL: bool>(
        src: &[&[T]],
        row_a: usize,
        row_b: usize,
        ncols: usize,
        bs3: &BoxSums,
        bs5: &mut BoxSums,
    ) {
        let ra = &src[row_a][..ncols];
        let rb = &src[row_b][..ncols];
        let mut c = 0;
        while c + N <= ncols {
            let (pa, pb) = (load_px(&ra[c..]), load_px(&rb[c..]));
            let s3 = I16s::from_slice(&bs3.sum[c..c + N]);
            let q3 = I32s::from_slice(&bs3.sq[c..c + N]);
            (s3 + pa + pb).copy_to_slice(&mut bs5.sum[c..c + N]);
            (q3 + madd(pa, pb, pa, pb)).copy_to_slice(&mut bs5.sq[c..c + N]);
            c += N;
        }
        if c + super::N <= ncols {
            let (pa, pb) = (super::load_px(&ra[c..]), super::load_px(&rb[c..]));
            let s3 = super::I16s::from_slice(&bs3.sum[c..c + super::N]);
            let q3 = super::I32s::from_slice(&bs3.sq[c..c + super::N]);
            (s3 + pa + pb).copy_to_slice(&mut bs5.sum[c..c + super::N]);
            (q3 + super::madd(pa, pb, pa, pb)).copy_to_slice(&mut bs5.sq[c..c + super::N]);
            c += super::N;
        }
        if TAIL && c < ncols {
            let (pa, pb) = (load_px_or_zero(&ra[c..]), load_px_or_zero(&rb[c..]));
            let s3 = I16s::from_slice(&bs3.sum[c..c + N]);
            let q3 = I32s::from_slice(&bs3.sq[c..c + N]);
            (s3 + pa + pb).copy_to_slice(&mut bs5.sum[c..c + N]);
            (q3 + madd(pa, pb, pa, pb)).copy_to_slice(&mut bs5.sq[c..c + N]);
        }
    }

    /// Slides a 5-row vertical sum `bs` by 2 rows: subtracts `row_old0`, `row_old1`
    /// and adds `row_new0`, `row_new1`.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn vsum_slide2<T: Pixel, const TAIL: bool>(
        src: &[&[T]],
        row_old0: usize,
        row_old1: usize,
        row_new0: usize,
        row_new1: usize,
        ncols: usize,
        bs: &mut BoxSums,
    ) {
        let ro0 = &src[row_old0][..ncols];
        let ro1 = &src[row_old1][..ncols];
        let rn0 = &src[row_new0][..ncols];
        let rn1 = &src[row_new1][..ncols];
        let mut c = 0;
        while c + N <= ncols {
            let (po0, po1) = (load_px(&ro0[c..]), load_px(&ro1[c..]));
            let (pn0, pn1) = (load_px(&rn0[c..]), load_px(&rn1[c..]));
            let (d0, d1) = (pn0 - po0, pn1 - po1);
            let sum = I16s::from_slice(&bs.sum[c..c + N]) + d0 + d1;
            let sq = I32s::from_slice(&bs.sq[c..c + N]) + madd(pn0 + po0, pn1 + po1, d0, d1);
            sum.copy_to_slice(&mut bs.sum[c..c + N]);
            sq.copy_to_slice(&mut bs.sq[c..c + N]);
            c += N;
        }
        if c + super::N <= ncols {
            let (po0, po1) = (super::load_px(&ro0[c..]), super::load_px(&ro1[c..]));
            let (pn0, pn1) = (super::load_px(&rn0[c..]), super::load_px(&rn1[c..]));
            let (d0, d1) = (pn0 - po0, pn1 - po1);
            let sum = super::I16s::from_slice(&bs.sum[c..c + super::N]) + d0 + d1;
            let sq = super::I32s::from_slice(&bs.sq[c..c + super::N])
                + super::madd(pn0 + po0, pn1 + po1, d0, d1);
            sum.copy_to_slice(&mut bs.sum[c..c + super::N]);
            sq.copy_to_slice(&mut bs.sq[c..c + super::N]);
            c += super::N;
        }
        if TAIL && c < ncols {
            let (po0, po1) = (load_px_or_zero(&ro0[c..]), load_px_or_zero(&ro1[c..]));
            let (pn0, pn1) = (load_px_or_zero(&rn0[c..]), load_px_or_zero(&rn1[c..]));
            let (d0, d1) = (pn0 - po0, pn1 - po1);
            let sum = I16s::from_slice(&bs.sum[c..c + N]) + d0 + d1;
            let sq = I32s::from_slice(&bs.sq[c..c + N]) + madd(pn0 + po0, pn1 + po1, d0, d1);
            sum.copy_to_slice(&mut bs.sum[c..c + N]);
            sq.copy_to_slice(&mut bs.sq[c..c + N]);
        }
    }

    /// `x = 256 - A2(z)` of spec 7.17.3 from `p = a * n - d * d` (any sign)
    /// and the scale `s`, with `z = min(255, Round2(max(p, 0) * s, 20))`.
    #[derive(Clone, Copy)]
    struct XFromP {
        /// Smallest `p` giving `z >= 255`.
        #[cfg(target_arch = "aarch64")]
        pmax: I32s,
        #[cfg(target_arch = "aarch64")]
        sv: I32s,
        /// `s` in `0..4096`.
        #[cfg(not(target_arch = "aarch64"))]
        sv16: I16s,
        /// `2 * s` in `0..8192`, so `(pl * s) >> 15 == mul_hi_i16(pl, 2 * s)`.
        #[cfg(not(target_arch = "aarch64"))]
        sv2: I16s,
        /// `8160 / s + 1`: clamping `ph` to `ph_max` keeps `ph * s` in `i16`
        /// while still reaching `d = z + 1 >= 256`.
        #[cfg(not(target_arch = "aarch64"))]
        ph_max: I16s,
        /// 4-lane versions of `sv16`, `sv2` and `ph_max` for [`Self::x4`].
        #[cfg(not(target_arch = "aarch64"))]
        sv16_4: I16x4,
        #[cfg(not(target_arch = "aarch64"))]
        sv2_4: I16x4,
        #[cfg(not(target_arch = "aarch64"))]
        ph_max4: I16x4,
    }

    impl XFromP {
        #[inline(always)]
        fn new(s: u32) -> Self {
            debug_assert!(s > 0 && s < 1 << 12);
            #[cfg(not(target_arch = "aarch64"))]
            let ph_max = (8160 / s + 1) as i16;
            #[cfg(target_arch = "aarch64")]
            {
                let pmax = ((255u32 << SGRPROJ_MTABLE_BITS) - (1 << (SGRPROJ_MTABLE_BITS - 1)))
                    .div_ceil(s);
                Self {
                    pmax: I32s::splat(pmax as i32),
                    sv: I32s::splat(s as i32),
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                Self {
                    sv16: I16s::splat(s as i16),
                    sv2: I16s::splat((s << 1) as i16),
                    ph_max: I16s::splat(ph_max),
                    sv16_4: I16x4::splat(s as i16),
                    sv2_4: I16x4::splat((s << 1) as i16),
                    ph_max4: I16x4::splat(ph_max),
                }
            }
        }

        /// On aarch64: `(x + 2^19) >> 20 == ((x >> 16) + 8) >> 4`, so `p * s`
        /// narrows in two steps (`shrn #16` to `i16`, clamping `< 0` in `i16`
        /// only when `BD != 8` since `p >= 0` at 8-bit, then `rshrn #4` to `u8`)
        /// before the 16-byte table lookup ([`sgr_x_u8`]).
        #[cfg(target_arch = "aarch64")]
        #[inline(always)]
        fn x_u8<const BD: u32>(self, p: I32s) -> Simd<u8, N> {
            let y16 = ((p.simd_min(self.pmax) * self.sv) >> 16).cast::<i16>();
            let y16 = if BD == 8 { y16 } else { y16.simd_max(I16s::splat(0)) };
            let z8 = ((y16.cast::<u16>() + U16s::splat(8)) >> 4).cast::<u8>();
            sgr_x_u8(z8)
        }

        #[cfg(target_arch = "aarch64")]
        #[inline(always)]
        fn x<const BD: u32>(self, p: I32s) -> I16s {
            self.x_u8::<BD>(p).cast::<u16>().cast::<i16>()
        }

        /// Without `pmulld`, `pminsd` and `pshufb` (SSE2):
        ///
        /// * With `p = ph * 2^15 + pl` (`ph = p >> 15` signed, `pl` in
        ///   `[0, 2^15)`), `d = z + 1 = ((ph * s + ((pl * s) >> 15) + 48) >> 5)`
        ///   is evaluated entirely in `i16` lanes (`I16s`): clamping `ph` to
        ///   `[-1, ph_max]` keeps `ph * s` in `i16`, and `(pl * s) >> 15` is
        ///   a single signed high-multiply `pmulhw(pl, 2 * s)`.
        /// * Clamping `d16` to `[1, 256]` and converting to `f32`, one
        ///   Newton-Raphson step from the IEEE-754 bit approximation
        ///   `r0 = from_bits(0x80f0_0100 - d.to_bits()) ≈ 16 / d` computes
        ///   `q = r0 * (32 - d * r0) ≈ 256 / d`, which rounds to the exact
        ///   `SGR_X[z]` for all `0 <= z <= 254` (and `1` at `z = 255`, where
        ///   subtracting `d16 >> 8` yields `0`) without any `divps`.
        #[cfg(not(target_arch = "aarch64"))]
        #[inline(always)]
        fn x<const BD: u32>(self, p: I32s) -> I16s {
            const MAGIC: f32 = (3 << 22) as f32;
            const RECIP_BIAS: u32 = 0x80f0_0100;
            let ph = narrow_sat(p >> 15)
                .simd_max(I16s::splat(-1))
                .simd_min(self.ph_max);
            let pl = narrow_sat(p & I32s::splat(0x7fff));
            let pl_hi = ((pl.cast::<i32>() * self.sv2.cast::<i32>()) >> 16).cast::<i16>();
            let d16 = ((ph * self.sv16 + pl_hi + I16s::splat(48)) >> 5)
                .simd_max(I16s::splat(1))
                .simd_min(I16s::splat(256));
            let d = d16.cast::<f32>();
            let r0 = Simd::<f32, N>::from_bits(U32s::splat(RECIP_BIAS) - d.to_bits());
            let q = r0 * (Simd::splat(32.0) - d * r0) + Simd::splat(MAGIC);
            narrow_sat(q.to_bits().cast::<i32>() - I32s::splat(MAGIC.to_bits() as i32)) - (d16 >> 8)
        }

        /// 4-lane version of [`Self::x`], operating on a single 128-bit `f32x4` /
        /// `i32x4` register for the 2-column tail (`w..w + 2`) of [`ab_from_vsum`].
        #[cfg(not(target_arch = "aarch64"))]
        #[inline(always)]
        fn x4(self, p: I32x4) -> I16x4 {
            const MAGIC: f32 = (3 << 22) as f32;
            const RECIP_BIAS: u32 = 0x80f0_0100;
            let ph = narrow_sat4(p >> 15)
                .simd_max(I16x4::splat(-1))
                .simd_min(self.ph_max4);
            let pl = narrow_sat4(p & I32x4::splat(0x7fff));
            let pl_hi = ((pl.cast::<i32>() * self.sv2_4.cast::<i32>()) >> 16).cast::<i16>();
            let d16 = ((ph * self.sv16_4 + pl_hi + I16x4::splat(48)) >> 5)
                .simd_max(I16x4::splat(1))
                .simd_min(I16x4::splat(256));
            let d = d16.cast::<f32>();
            let r0 = Simd::<f32, 4>::from_bits(U32x4::splat(RECIP_BIAS) - d.to_bits());
            let q = r0 * (Simd::splat(32.0) - d * r0) + Simd::splat(MAGIC);
            narrow_sat4(q.to_bits().cast::<i32>() - I32x4::splat(MAGIC.to_bits() as i32))
                - (d16 >> 8)
        }
    }

    /// Computes one raw row of `X = 256 - A` and `B` arrays (`ab.a`, `ab.b`) from
    /// the vertical box sums `(sums, sqs)` with radius `R` and scale `s` (spec 7.17.3).
    ///
    /// Storing `x = 256 - a2` instead of `a2` lets [`box_out`] compute `f - u`
    /// (`u = px << 4`) directly as `Round2(b - x * px, shift)`.
    #[inline(always)]
    fn ab_from_vsum<const BD: u32, const R: usize>(
        sums: &[i16],
        sqs: &[i32],
        w: usize,
        s: u32,
        ab: &mut AbRow,
    ) {
        const { assert!(R == 1 || R == 2) };
        assert!(w <= LR_MAX_W && sums.len() >= SUM_LEN - 1 && sqs.len() >= SUM_LEN - 1);
        let sums = &sums[..SUM_LEN - 1];
        let sqs = &sqs[..SUM_LEN - 1];
        let xc = XFromP::new(s);
        #[cfg(target_arch = "aarch64")]
        {
            let steps = (w + 2).div_ceil(N);
            assert!(steps * N <= AB_LEN);
            let sums = &sums[..steps * N + 4];
            let sqs = &sqs[..steps * N + 4];
            for ((k, oa), ob) in ab.a[..steps * N]
                .chunks_exact_mut(N)
                .enumerate()
                .zip(ab.b[..steps * N].chunks_exact_mut(N))
            {
                let j = k * N;
                let sum_win = &sums[j..j + N + 4];
                let sq_win = &sqs[j..j + N + 4];
                let s0 = I16s::from_slice(&sum_win[0..N]);
                let s1 = I16s::from_slice(&sum_win[1..1 + N]);
                let s2 = I16s::from_slice(&sum_win[2..2 + N]);
                let bsum = if R == 1 {
                    (s0 + s1) + s2
                } else {
                    let s3 = I16s::from_slice(&sum_win[3..3 + N]);
                    let s4 = I16s::from_slice(&sum_win[4..4 + N]);
                    (s0 + s1) + (s2 + s3) + s4
                };
                let q0 = I32s::from_slice(&sq_win[0..N]);
                let q1 = I32s::from_slice(&sq_win[1..1 + N]);
                let q2 = I32s::from_slice(&sq_win[2..2 + N]);
                let asum = if R == 1 {
                    (q0 + q1) + q2
                } else {
                    let q3 = I32s::from_slice(&sq_win[3..3 + N]);
                    let q4 = I32s::from_slice(&sq_win[4..4 + N]);
                    (q0 + q1) + (q2 + q3) + q4
                };

                let (a, dd) = if BD == 8 {
                    (asum, bsum)
                } else {
                    const SH_A: i32 = 4;
                    const SH_B: i16 = 2;
                    (
                        (asum + I32s::splat(1 << (SH_A - 1))) >> SH_A,
                        (bsum + I16s::splat(1 << (SH_B - 1))) >> SH_B,
                    )
                };
                let an = if R == 1 { mul_const::<9>(a) } else { mul_const::<25>(a) };
                let p = an - mul_wide(dd, dd);
                let x8 = xc.x_u8::<BD>(p);
                let x = x8.cast::<u16>().cast::<i16>();
                let b2 = if R == 2 {
                    const { assert!(one_over_n(2) == 41 * 4) };
                    const SH: i32 = SGRPROJ_RECIP_BITS as i32 - 2;
                    (mul_wide(x * I16s::splat(41), bsum) + I32s::splat(1 << (SH - 1))) >> SH
                } else {
                    const { assert!(one_over_n(1) == 91 * 5 && BD <= 10) };
                    const SH: u32 = SGRPROJ_RECIP_BITS;
                    let x91 = (x.cast::<u16>() * U16s::splat(91)).cast::<u32>();
                    let b5 = (bsum.cast::<u16>() * U16s::splat(5)).cast::<u32>();
                    ((x91 * b5 + U32s::splat(1 << (SH - 1))) >> SH).cast::<i32>()
                };
                x8.copy_to_slice(oa);
                b2.copy_to_slice(ob);
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let d = 2 * R + 1;
            let mut j = 0;
            while j + 4 < w + 2 {
                let sum_win = &sums[j..j + N + 4];
                let sq_win = &sqs[j..j + N + 4];
                let mut bsum = I16s::from_slice(&sum_win[..N]);
                let mut asum = I32s::from_slice(&sq_win[..N]);
                for k in 1..d {
                    bsum += I16s::from_slice(&sum_win[k..k + N]);
                    asum += I32s::from_slice(&sq_win[k..k + N]);
                }
                let (a, dd) = if BD == 8 {
                    (asum, bsum)
                } else {
                    const SH_A: i32 = 4;
                    const SH_B: i16 = 2;
                    (
                        (asum + I32s::splat(1 << (SH_A - 1))) >> SH_A,
                        (bsum + I16s::splat(1 << (SH_B - 1))) >> SH_B,
                    )
                };
                let an = if R == 1 {
                    mul_const::<9>(a)
                } else if BD == 8 {
                    mul_const::<25>(a)
                } else {
                    // `16 * a = (asum + 8) & !15`; keeping `16 * a` as a bitwise mask
                    // prevents LLVM from folding `16 * a + 9 * a` into `a * 25` (which
                    // SSE2 lowers to `pmuludq` + shuffles instead of shifts and adds).
                    ((asum + I32s::splat(8)) & I32s::splat(-16)) + mul_const::<9>(a)
                };
                let p = an - mul_wide(dd, dd);
                let x = xc.x::<BD>(p);
                let b2 = if R == 2 {
                    const { assert!(one_over_n(2) == 41 * 4) };
                    const SH: i32 = SGRPROJ_RECIP_BITS as i32 - 2;
                    (mul_wide(x * I16s::splat(41), bsum) + I32s::splat(1 << (SH - 1))) >> SH
                } else {
                    const { assert!(one_over_n(1) == 91 * 5 && BD <= 10) };
                    const SH: u32 = SGRPROJ_RECIP_BITS;
                    let x91 = (x.cast::<u16>() * U16s::splat(91)).cast::<u32>();
                    let b5 = (bsum.cast::<u16>() * U16s::splat(5)).cast::<u32>();
                    ((x91 * b5 + U32s::splat(1 << (SH - 1))) >> SH).cast::<i32>()
                };
                x.copy_to_slice(&mut ab.a[j..j + N]);
                b2.copy_to_slice(&mut ab.b[j..j + N]);
                j += N;
            }
            if j < w + 2 {
                let sum_win = &sums[j..j + 4 + 4];
                let sq_win = &sqs[j..j + 4 + 4];
                let mut bsum = I16x4::from_slice(&sum_win[..4]);
                let mut asum = I32x4::from_slice(&sq_win[..4]);
                for k in 1..d {
                    bsum += I16x4::from_slice(&sum_win[k..k + 4]);
                    asum += I32x4::from_slice(&sq_win[k..k + 4]);
                }
                let (a, dd) = if BD == 8 {
                    (asum, bsum)
                } else {
                    const SH_A: i32 = 4;
                    const SH_B: i16 = 2;
                    (
                        (asum + I32x4::splat(1 << (SH_A - 1))) >> SH_A,
                        (bsum + I16x4::splat(1 << (SH_B - 1))) >> SH_B,
                    )
                };
                let an = if R == 1 {
                    mul_const4::<9>(a)
                } else if BD == 8 {
                    mul_const4::<25>(a)
                } else {
                    ((asum + I32x4::splat(8)) & I32x4::splat(-16)) + mul_const4::<9>(a)
                };
                let p = an - dd.cast::<i32>() * dd.cast::<i32>();
                let x = xc.x4(p);
                let b2 = if R == 2 {
                    const SH: i32 = SGRPROJ_RECIP_BITS as i32 - 2;
                    ((x * I16x4::splat(41)).cast::<i32>() * bsum.cast::<i32>()
                        + I32x4::splat(1 << (SH - 1)))
                        >> SH
                } else {
                    const SH: u32 = SGRPROJ_RECIP_BITS;
                    let x91 = (x.cast::<u16>() * U16x4::splat(91)).cast::<u32>();
                    let b5 = (bsum.cast::<u16>() * U16x4::splat(5)).cast::<u32>();
                    ((x91 * b5 + U32x4::splat(1 << (SH - 1))) >> SH).cast::<i32>()
                };
                x.copy_to_slice(&mut ab.a[j..j + 4]);
                b2.copy_to_slice(&mut ab.b[j..j + 4]);
            }
        }
    }

    /// Weights the raw `A`/`B` row in `ab` horizontally (see [`WeightedRow`])
    /// into `out` for columns `0..w`.
    #[inline(always)]
    fn weigh_row<const K0: i32, const K1: i32>(ab: &AbRow, w: usize, out: &mut WeightedRow) {
        assert!(w <= LR_MAX_W);
        const { assert!((K0 == 5 && K1 == 6) || (K0 == 3 && K1 == 4)) };
        #[cfg(target_arch = "aarch64")]
        {
            let chunks = w.div_ceil(N).min(LR_CHUNKS);
            let aw = &ab.a[..chunks * N + 2];
            let bw = &ab.b[..chunks * N + 2];
            for ((k, oa), ob) in out.a[..chunks]
                .iter_mut()
                .enumerate()
                .zip(&mut out.b[..chunks])
            {
                let c = k * N;
                let awin = &aw[c..c + N + 2];
                let bwin = &bw[c..c + N + 2];
                let a0 = Simd::<u8, N>::from_slice(&awin[0..N]);
                let a1_8 = Simd::<u8, N>::from_slice(&awin[1..1 + N]);
                let a2 = Simd::<u8, N>::from_slice(&awin[2..2 + N]);
                let sa02_u = a0.cast::<u16>() + a2.cast::<u16>();
                let sa_u = sa02_u + a1_8.cast::<u16>();
                let b0 = I32s::from_slice(&bwin[0..N]);
                let b1 = I32s::from_slice(&bwin[1..1 + N]);
                let b2 = I32s::from_slice(&bwin[2..2 + N]);
                let sb02 = b0 + b2;
                let sb = sb02 + b1;
                let (wa, wb) = if K0 == 5 {
                    (
                        ((sa_u << 2) + sa_u + a1_8.cast::<u16>()).cast::<i16>(),
                        (sb << 2) + sb + b1,
                    )
                } else {
                    (((sa_u << 2) - sa02_u).cast::<i16>(), (sb << 2) - sb02)
                };
                *oa = wa;
                *ob = wb;
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        for (k, c) in (0..w).step_by(N).enumerate() {
            let awin = &ab.a[c..c + N + 2];
            let bwin = &ab.b[c..c + N + 2];
            let a0 = I16s::from_slice(&awin[0..N]);
            let a1 = I16s::from_slice(&awin[1..1 + N]);
            let a2 = I16s::from_slice(&awin[2..2 + N]);
            let b0 = I32s::from_slice(&bwin[0..N]);
            let b1 = I32s::from_slice(&bwin[1..1 + N]);
            let b2 = I32s::from_slice(&bwin[2..2 + N]);
            let sa02 = a0 + a2;
            let sb02 = b0 + b2;
            let sa = sa02 + a1;
            let sb = sb02 + b1;
            let (wa, wb) = if K0 == 5 {
                ((sa << 2) + sa + a1, (sb << 2) + sb + b1)
            } else {
                ((sa << 2) - sa02, (sb << 2) - sb02)
            };
            out.a[k] = wa;
            out.b[k] = wb;
        }
    }

    /// Fused horizontal weighting for the `r = 1` pass: accumulates the `(4, 4)`
    /// weights of the current row directly into the previous row's `(3, 4)`
    /// weights in `tm_accum` (`tm = p343_prev + m444_cur`), and writes the current
    /// row's `(3, 4)` weights into `out343`.
    #[inline(always)]
    fn weigh_r1(ab: &AbRow, w: usize, tm_accum: &mut WeightedRow, out343: &mut WeightedRow) {
        assert!(w <= LR_MAX_W);
        #[cfg(target_arch = "aarch64")]
        {
            let chunks = w.div_ceil(N).min(LR_CHUNKS);
            let aw = &ab.a[..chunks * N + 2];
            let bw = &ab.b[..chunks * N + 2];
            for ((((k, tma), tmb), oa), ob) in tm_accum.a[..chunks]
                .iter_mut()
                .enumerate()
                .zip(&mut tm_accum.b[..chunks])
                .zip(&mut out343.a[..chunks])
                .zip(&mut out343.b[..chunks])
            {
                let c = k * N;
                let awin = &aw[c..c + N + 2];
                let bwin = &bw[c..c + N + 2];
                let a0 = Simd::<u8, N>::from_slice(&awin[0..N]);
                let a1_8 = Simd::<u8, N>::from_slice(&awin[1..1 + N]);
                let a2 = Simd::<u8, N>::from_slice(&awin[2..2 + N]);
                let sa02_u = a0.cast::<u16>() + a2.cast::<u16>();
                let sa_u = sa02_u + a1_8.cast::<u16>();
                let sa444 = (sa_u << 2).cast::<i16>();
                let sa343 = sa444 - sa02_u.cast::<i16>();

                let b0 = I32s::from_slice(&bwin[0..N]);
                let b1 = I32s::from_slice(&bwin[1..1 + N]);
                let b2 = I32s::from_slice(&bwin[2..2 + N]);
                let sb02 = b0 + b2;
                let sb444 = (sb02 + b1) << 2;
                let sb343 = sb444 - sb02;
                *tma += sa444;
                *tmb += sb444;
                *oa = sa343;
                *ob = sb343;
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        for (k, c) in (0..w).step_by(N).enumerate() {
            let awin = &ab.a[c..c + N + 2];
            let bwin = &ab.b[c..c + N + 2];
            let a0 = I16s::from_slice(&awin[0..N]);
            let a1 = I16s::from_slice(&awin[1..1 + N]);
            let a2 = I16s::from_slice(&awin[2..2 + N]);
            let b0 = I32s::from_slice(&bwin[0..N]);
            let b1 = I32s::from_slice(&bwin[1..1 + N]);
            let b2 = I32s::from_slice(&bwin[2..2 + N]);
            let sa02 = a0 + a2;
            let sb02 = b0 + b2;
            let sa444 = (sa02 + a1) << 2;
            let sa343 = sa444 - sa02;
            let sb444 = (sb02 + b1) << 2;
            let sb343 = sb444 - sb02;
            tm_accum.a[k] += sa444;
            tm_accum.b[k] += sb444;
            out343.a[k] = sa343;
            out343.b[k] = sb343;
        }
    }

    /// Mutable references `(a[i], a[j])` for distinct indices `i != j`.
    #[inline(always)]
    fn two_mut<T>(a: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
        if i < j {
            let (l, r) = a.split_at_mut(j);
            (&mut l[i], &mut r[0])
        } else {
            let (l, r) = a.split_at_mut(i);
            (&mut r[0], &mut l[j])
        }
    }

    /// `f - (px << 4) = Round2(b - x * px, shift)` of the box filter output
    /// (since `a = (256 << (shift - 4)) - x`, `(a * px) >> shift` contributes
    /// `px << 4` plus `(-x * px) >> shift`).
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn box_out(x: I16s, b: I32s, px: I16s, shift: u32) -> I32s {
        (b - mul_wide(x, px) + I32s::splat(1 << (shift - 1))) >> shift as i32
    }

    #[cfg(not(target_arch = "aarch64"))]
    #[inline(always)]
    fn box_out(x: I16s, b: I32s, px: I16s, shift: u32) -> I16s {
        narrow_sat((b - mul_wide(x, px) + I32s::splat(1 << (shift - 1))) >> shift as i32)
    }

    /// Filters one output row `out` from its source pixels `px_row` and the
    /// weighted rows `[prev, next, tm, d]` (see [`sgr_filter_both`]).
    ///
    /// `P0` selects the `r = 2` pass: 0 = off, 1 = odd row (`next` only),
    /// 2 = even row (`prev + next`); `P1` enables the `r = 1` pass (`tm + d`,
    /// where `tm = t + m`). `w0` / `w2` are the projection weights of the
    /// `r = 2` / `r = 1` passes (the weight of a disabled pass is 0 and is
    /// not read). Const parameters keep each loop body in one basic block so
    /// widening multiplies select cleanly, and outlining on aarch64 gives the
    /// projection loop dedicated NEON registers without spills.
    #[cfg_attr(target_arch = "aarch64", inline(never))]
    #[cfg_attr(not(target_arch = "aarch64"), inline(always))]
    fn sgr_finish_row<T: Pixel, const BD: u32, const P0: u8, const P1: bool>(
        px_row: &[T],
        out: &mut [T],
        rows: [&WeightedRow; 4],
        wts: &ChunkWeights,
    ) {
        const { assert!(BD == 8 || BD == 10) };
        let w = out.len();
        assert!(w <= LR_MAX_W && px_row.len() >= w);
        let [prev, next, tm, d] = rows;
        let shift = SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS;
        let rnd = I32s::splat(1 << (shift - 1));
        #[cfg(not(target_arch = "aarch64"))]
        let rnd16 = I16s::splat(1 << (shift - 1));
        #[cfg(not(target_arch = "aarch64"))]
        let one = I16s::splat(1);
        let max_px = I16s::splat(((1i32 << BD) - 1) as i16);
        // Since w0 + w1 + w2 == 2^7, w1 * (px << 4) + w0 * f0 + w2 * f1
        // equals (px << 11) + w0 * (f0 - (px << 4)) + w2 * (f1 - (px << 4)).
        let step = |px: I16s, k: usize| -> I16s {
            let f0 = match P0 {
                1 => box_out(
                    next.ld_a(k),
                    next.ld_b(k),
                    px,
                    SGRPROJ_SGR_BITS + 4 - SGRPROJ_RST_BITS,
                ),
                _ => box_out(
                    prev.ld_a(k) + next.ld_a(k),
                    prev.ld_b(k) + next.ld_b(k),
                    px,
                    SGRPROJ_SGR_BITS + 5 - SGRPROJ_RST_BITS,
                ),
            };
            #[cfg(target_arch = "aarch64")]
            let f1 = box_out(
                tm.ld_a(k) + d.ld_a(k),
                tm.ld_b(k) + d.ld_b(k),
                px,
                SGRPROJ_SGR_BITS + 5 - SGRPROJ_RST_BITS,
            );
            #[cfg(not(target_arch = "aarch64"))]
            let f1 = {
                // For BD <= 10, `|(b - x * px + 256) >> 8| <= 32728 < i16::MAX`, so
                // packing before the final `>> 1` is exact and blocks LLVM's InstCombine
                // from pushing `madd_p`'s `INTERLEAVE` shuffle through `narrow_sat` into i32.
                const SH1: i32 = (SGRPROJ_SGR_BITS + 5 - SGRPROJ_RST_BITS) as i32;
                let x1 = tm.ld_a(k) + d.ld_a(k);
                let b1 = tm.ld_b(k) + d.ld_b(k);
                narrow_sat((b1 - mul_wide(x1, px) + I32s::splat(1 << (SH1 - 1))) >> (SH1 - 1)) >> 1
            };
            #[cfg(target_arch = "aarch64")]
            let v = {
                // Keep `f0, f1` in `I32s` (`srshr #8/#9`) and project in `I32s` (`mul` +
                // `mla` -> `rshrn #11`), avoiding LLVM's 4-way `v4i16` split of `I16s`.
                let (w0, w2) = (I32s::splat(wts.w0[k] as i32), I32s::splat(wts.w2[k] as i32));
                match (P0 != 0, P1) {
                    (true, true) => f0 * w0 + f1 * w2 + rnd,
                    (true, false) => f0 * w0 + rnd,
                    _ => f1 * w2 + rnd,
                }
            };
            #[cfg(not(target_arch = "aarch64"))]
            let v = {
                let (w0, w2) = (I16s::splat(wts.w0[k]), I16s::splat(wts.w2[k]));
                match (P0 != 0, P1) {
                    (true, true) => madd_p(f0, f1, w0, w2) + rnd,
                    (true, false) => madd_p(f0, rnd16, w0, one),
                    _ => madd_p(f1, rnd16, w2, one),
                }
            };
            // With w0 in [-96, 31], w2 in [-62, 256], and f0, f1 in [-16368, 16384],
            // `v >> 11` lies in [-2816, 2816], strictly inside i16.
            #[cfg(target_arch = "aarch64")]
            let delta = (v >> shift as i32).cast::<i16>();
            #[cfg(not(target_arch = "aarch64"))]
            let delta = narrow_sat_p(v >> shift as i32);
            px + delta
        };
        let chunks = (w / N).min(LR_CHUNKS);
        let px_chunks = &px_row[..chunks * N];
        let out_chunks = &mut out[..chunks * N];
        if BD == 8 && T::MAX_BITDEPTH == 8 {
            for (k, (px_s, out_s)) in px_chunks
                .chunks_exact(N)
                .zip(out_chunks.chunks_exact_mut(N))
                .enumerate()
            {
                let u = step(load_px(px_s), k)
                    .simd_clamp(I16s::splat(0), I16s::splat(255))
                    .cast::<u8>();
                store_px16_u8(out_s, u);
            }
            let (k, c) = (chunks, chunks * N);
            if c < w && k < LR_CHUNKS {
                let u = step(load_px_or_zero(&px_row[c..w]), k)
                    .simd_clamp(I16s::splat(0), I16s::splat(255))
                    .cast::<u8>();
                store_px16_u8_tail(&mut out[c..w], u);
            }
        } else {
            for (k, (px_s, out_s)) in px_chunks
                .chunks_exact(N)
                .zip(out_chunks.chunks_exact_mut(N))
                .enumerate()
            {
                let u = step(load_px(px_s), k)
                    .simd_max(I16s::splat(0))
                    .simd_min(max_px);
                store_px(out_s, u);
            }
            let (k, c) = (chunks, chunks * N);
            if c < w && k < LR_CHUNKS {
                let u = step(load_px_or_zero(&px_row[c..w]), k)
                    .simd_max(I16s::splat(0))
                    .simd_min(max_px);
                store_px_tail(&mut out[c..w], u);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn sgr_filter_both_impl<T: Pixel, const BD: u32, const TAIL: bool>(
        dst: &mut [T],
        dst_stride: usize,
        src: &[&[T]],
        w: usize,
        h: usize,
        ncols: usize,
        s0: u32,
        s1: u32,
        wts: &ChunkWeights,
    ) {
        assert!(w <= LR_MAX_W);
        let mut scratch = ScratchGuard::take();
        let Scratch {
            p0,
            p343,
            bs3,
            bs5,
            ab,
            ..
        } = &mut *scratch;
        let mut cur0 = 0;

        // Row -1: r1 uses padded rows 1..=3, r2 uses padded rows 0..=4.
        vsum_init3::<T, TAIL>(src, 1, 0, ncols, bs3);
        ab_from_vsum::<BD, 1>(&bs3.sum[1..], &bs3.sq[1..], w, s1, ab);
        weigh_row::<3, 4>(ab, w, &mut p343[0]);
        vsum_add2::<T, TAIL>(src, 0, 4, ncols, bs3, bs5);
        ab_from_vsum::<BD, 2>(&bs5.sum, &bs5.sq, w, s0, ab);
        weigh_row::<5, 6>(ab, w, &mut p0[0]);

        // Row 0: r1 slides from padded rows 1..=3 to 2..=4.
        vsum_slide1::<T, TAIL>(src, 1, 4, 0, ncols, bs3);
        ab_from_vsum::<BD, 1>(&bs3.sum[1..], &bs3.sq[1..], w, s1, ab);
        {
            let (tm0, out1) = two_mut(p343, 0, 1);
            weigh_r1(ab, w, tm0, out1);
        }

        let mut t_idx = 0usize;
        let mut m_idx = 1usize;
        let mut d_idx = 2usize;
        for i in 0..h {
            let odd = i & 1 == 1;
            if !odd {
                // Row i + 1 (odd): r2 uses padded rows i+2..=i+6, r1 slides to i+3..=i+5.
                vsum_slide1_and_5::<T, TAIL>(src, i + 2, i + 5, i + 6, ncols, bs3, bs5);
                cur0 ^= 1;
                ab_from_vsum::<BD, 2>(&bs5.sum, &bs5.sq, w, s0, ab);
                weigh_row::<5, 6>(ab, w, &mut p0[cur0]);
            } else {
                // Row i + 1 (even): r1 slides from padded rows i+2..=i+4 to i+3..=i+5.
                vsum_slide1::<T, TAIL>(src, i + 2, i + 5, 0, ncols, bs3);
            }
            ab_from_vsum::<BD, 1>(&bs3.sum[1..], &bs3.sq[1..], w, s1, ab);
            {
                let (tm_next, out_d) = two_mut(p343, m_idx, d_idx);
                weigh_r1(ab, w, tm_next, out_d);
            }

            let (prev, next) = (&p0[cur0 ^ 1], &p0[cur0]);
            let (tm, d) = (&p343[t_idx], &p343[d_idx]);
            (t_idx, m_idx, d_idx) = (m_idx, d_idx, t_idx);

            let px_row = &src[i + LR_BORDER][LR_BORDER..LR_BORDER + w];
            let out = &mut dst[i * dst_stride..i * dst_stride + w];
            let rows = [prev, next, tm, d];
            if odd {
                sgr_finish_row::<T, BD, 1, true>(px_row, out, rows, wts);
            } else {
                sgr_finish_row::<T, BD, 2, true>(px_row, out, rows, wts);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(target_arch = "aarch64", inline(never))]
    fn sgr_filter_both<T: Pixel, const BD: u32>(
        dst: &mut [T],
        dst_stride: usize,
        src: &[&[T]],
        w: usize,
        h: usize,
        ncols: usize,
        s0: u32,
        s1: u32,
        wts: &ChunkWeights,
    ) {
        sgr_filter_both_impl::<T, BD, false>(dst, dst_stride, src, w, h, ncols, s0, s1, wts);
    }

    #[allow(clippy::too_many_arguments)]
    #[cold]
    #[inline(never)]
    fn sgr_filter_both_tail<T: Pixel, const BD: u32>(
        dst: &mut [T],
        dst_stride: usize,
        src: &[&[T]],
        w: usize,
        h: usize,
        ncols: usize,
        s0: u32,
        s1: u32,
        wts: &ChunkWeights,
    ) {
        sgr_filter_both_impl::<T, BD, true>(dst, dst_stride, src, w, h, ncols, s0, s1, wts);
    }

    #[allow(clippy::too_many_arguments)]
    fn sgr_filter_r1_impl<T: Pixel, const BD: u32, const TAIL: bool>(
        dst: &mut [T],
        dst_stride: usize,
        src: &[&[T]],
        w: usize,
        h: usize,
        ncols: usize,
        s1: u32,
        wts: &ChunkWeights,
    ) {
        assert!(w <= LR_MAX_W);
        let col0 = LR_BORDER - 2;
        let mut scratch = ScratchGuard::take();
        let Scratch { p343, bs3, ab, .. } = &mut *scratch;

        vsum_init3::<T, TAIL>(src, 1, col0, ncols, bs3);
        ab_from_vsum::<BD, 1>(&bs3.sum, &bs3.sq, w, s1, ab);
        weigh_row::<3, 4>(ab, w, &mut p343[0]);

        vsum_slide1::<T, TAIL>(src, 1, 4, col0, ncols, bs3);
        ab_from_vsum::<BD, 1>(&bs3.sum, &bs3.sq, w, s1, ab);
        {
            let (tm0, out1) = two_mut(p343, 0, 1);
            weigh_r1(ab, w, tm0, out1);
        }

        let mut t_idx = 0usize;
        let mut m_idx = 1usize;
        let mut d_idx = 2usize;
        for i in 0..h {
            vsum_slide1::<T, TAIL>(src, i + 2, i + 5, col0, ncols, bs3);
            ab_from_vsum::<BD, 1>(&bs3.sum, &bs3.sq, w, s1, ab);
            {
                let (tm_next, out_d) = two_mut(p343, m_idx, d_idx);
                weigh_r1(ab, w, tm_next, out_d);
            }

            let (tm, d) = (&p343[t_idx], &p343[d_idx]);
            (t_idx, m_idx, d_idx) = (m_idx, d_idx, t_idx);

            let px_row = &src[i + LR_BORDER][LR_BORDER..LR_BORDER + w];
            let out = &mut dst[i * dst_stride..i * dst_stride + w];
            sgr_finish_row::<T, BD, 0, true>(px_row, out, [tm, tm, tm, d], wts);
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(target_arch = "aarch64", inline(never))]
    fn sgr_filter_r1<T: Pixel, const BD: u32>(
        dst: &mut [T],
        dst_stride: usize,
        src: &[&[T]],
        w: usize,
        h: usize,
        ncols: usize,
        s1: u32,
        wts: &ChunkWeights,
    ) {
        sgr_filter_r1_impl::<T, BD, false>(dst, dst_stride, src, w, h, ncols, s1, wts);
    }

    #[allow(clippy::too_many_arguments)]
    #[cold]
    #[inline(never)]
    fn sgr_filter_r1_tail<T: Pixel, const BD: u32>(
        dst: &mut [T],
        dst_stride: usize,
        src: &[&[T]],
        w: usize,
        h: usize,
        ncols: usize,
        s1: u32,
        wts: &ChunkWeights,
    ) {
        sgr_filter_r1_impl::<T, BD, true>(dst, dst_stride, src, w, h, ncols, s1, wts);
    }

    #[allow(clippy::too_many_arguments)]
    fn sgr_filter_r2_impl<T: Pixel, const BD: u32, const TAIL: bool>(
        dst: &mut [T],
        dst_stride: usize,
        src: &[&[T]],
        w: usize,
        h: usize,
        ncols: usize,
        s0: u32,
        wts: &ChunkWeights,
    ) {
        assert!(w <= LR_MAX_W);
        let mut scratch = ScratchGuard::take();
        let Scratch {
            p0, bs3, bs5, ab, ..
        } = &mut *scratch;
        let mut cur0 = 0;

        vsum_init3::<T, TAIL>(src, 1, 0, ncols, bs3);
        vsum_add2::<T, TAIL>(src, 0, 4, ncols, bs3, bs5);
        ab_from_vsum::<BD, 2>(&bs5.sum, &bs5.sq, w, s0, ab);
        weigh_row::<5, 6>(ab, w, &mut p0[0]);

        for i in 0..h {
            let odd = i & 1 == 1;
            if !odd {
                vsum_slide2::<T, TAIL>(src, i, i + 1, i + 5, i + 6, ncols, bs5);
                cur0 ^= 1;
                ab_from_vsum::<BD, 2>(&bs5.sum, &bs5.sq, w, s0, ab);
                weigh_row::<5, 6>(ab, w, &mut p0[cur0]);
            }
            let (prev, next) = (&p0[cur0 ^ 1], &p0[cur0]);

            let px_row = &src[i + LR_BORDER][LR_BORDER..LR_BORDER + w];
            let out = &mut dst[i * dst_stride..i * dst_stride + w];
            let rows = [prev, next, next, next];
            if odd {
                sgr_finish_row::<T, BD, 1, false>(px_row, out, rows, wts);
            } else {
                sgr_finish_row::<T, BD, 2, false>(px_row, out, rows, wts);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(target_arch = "aarch64", inline(never))]
    fn sgr_filter_r2<T: Pixel, const BD: u32>(
        dst: &mut [T],
        dst_stride: usize,
        src: &[&[T]],
        w: usize,
        h: usize,
        ncols: usize,
        s0: u32,
        wts: &ChunkWeights,
    ) {
        sgr_filter_r2_impl::<T, BD, false>(dst, dst_stride, src, w, h, ncols, s0, wts);
    }

    #[allow(clippy::too_many_arguments)]
    #[cold]
    #[inline(never)]
    fn sgr_filter_r2_tail<T: Pixel, const BD: u32>(
        dst: &mut [T],
        dst_stride: usize,
        src: &[&[T]],
        w: usize,
        h: usize,
        ncols: usize,
        s0: u32,
        wts: &ChunkWeights,
    ) {
        sgr_filter_r2_impl::<T, BD, true>(dst, dst_stride, src, w, h, ncols, s0, wts);
    }

    /// Self guided filter, same contract as [`crate::decoder::av1::dsp::lr::sgr_filter`].
    ///
    /// Vertical box sums are maintained incrementally by sliding a 3-row window
    /// (`p_new^2 - p_old^2 = (p_new + p_old) * (p_new - p_old)`) and sharing the
    /// inner 3 rows with the 5-row pass on odd rows. The `A`/`B` rows are weighted
    /// horizontally once per row (5, 6 for the r = 2 pass; fused 3, 4 and 4, 4 for
    /// the r = 1 pass), so the 3x3 weighted sums of spec 7.17.3 become sums of 1
    /// to 3 weighted rows.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn sgr_filter<T: Pixel, const BD: u32>(
        dst: &mut [T],
        dst_stride: usize,
        src: &[&[T]],
        w: usize,
        h: usize,
        set: usize,
        xqd: [i32; 2],
    ) {
        sgr_filter_units::<T, BD>(dst, dst_stride, src, w, h, set, &[(w, xqd)]);
    }

    /// Per output chunk projection weights `(w0, w2)` (see [`sgr_finish_row`]).
    /// Stored as scalar `i16`s (96 B total vs 1.5 KB of `I16s` vectors) to avoid
    /// a per-call `memset` and `SpecFill` on `sgr_filter_units`; NEON `ld1r`
    /// splats directly from memory.
    pub(super) struct ChunkWeights {
        w0: [i16; LR_CHUNKS],
        w2: [i16; LR_CHUNKS],
    }

    /// [`sgr_filter`] for a run of horizontally adjacent restoration units
    /// with the same `set`: `units` lists the width and `xqd` of each unit
    /// from left to right, `w` is their total width, and every width but the
    /// last is a multiple of 16. The box sums and the `A` / `B` arrays do
    /// not depend on `xqd`, so they are computed once for the whole run
    /// (exactly as for each unit: they only depend on the source samples
    /// within 3 columns of each output column); only the final projection
    /// uses the weights of the unit of each output column.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn sgr_filter_units<T: Pixel, const BD: u32>(
        dst: &mut [T],
        dst_stride: usize,
        src: &[&[T]],
        w: usize,
        h: usize,
        set: usize,
        units: &[(usize, [i32; 2])],
    ) {
        check_rect(w, h);
        const { assert!(BD == 8 || BD == 10) };
        let r0 = SGR_PARAMS[set][0] as usize;
        let r1 = SGR_PARAMS[set][2] as usize;
        let [s0, s1] = SGR_SCALE[set];
        let mut wts = ChunkWeights {
            w0: [0; LR_CHUNKS],
            w2: [0; LR_CHUNKS],
        };
        let mut k = 0;
        let mut total = 0;
        for (i, &(uw, xqd)) in units.iter().enumerate() {
            debug_assert!(i + 1 == units.len() || uw % N == 0);
            let w2_raw = (1 << SGRPROJ_PRJ_BITS) - xqd[0] - xqd[1];
            let w0 = if r0 != 0 { xqd[0] } else { 0 };
            let w2 = if r1 != 0 { w2_raw } else { 0 };
            debug_assert!(w0.unsigned_abs() <= 1024 && w2.unsigned_abs() <= 1024);
            let (w0, w2) = (w0 as i16, w2 as i16);
            let end = (k + uw.div_ceil(N)).min(LR_CHUNKS);
            while k < end {
                wts.w0[k] = w0;
                wts.w2[k] = w2;
                k += 1;
            }
            total += uw;
        }
        assert_eq!(total, w);

        match (r0 != 0, r1 != 0) {
            (true, true) => {
                let ncols = box_cols(src, h, 0, w + 6);
                if ncols & 7 == 0 {
                    sgr_filter_both::<T, BD>(dst, dst_stride, src, w, h, ncols, s0, s1, &wts);
                } else {
                    sgr_filter_both_tail::<T, BD>(dst, dst_stride, src, w, h, ncols, s0, s1, &wts);
                }
            }
            (false, true) => {
                let ncols = box_cols(src, h, LR_BORDER - 2, w + 4);
                if ncols & 7 == 0 {
                    sgr_filter_r1::<T, BD>(dst, dst_stride, src, w, h, ncols, s1, &wts);
                } else {
                    sgr_filter_r1_tail::<T, BD>(dst, dst_stride, src, w, h, ncols, s1, &wts);
                }
            }
            (true, false) => {
                let ncols = box_cols(src, h, 0, w + 6);
                if ncols & 7 == 0 {
                    sgr_filter_r2::<T, BD>(dst, dst_stride, src, w, h, ncols, s0, &wts);
                } else {
                    sgr_filter_r2_tail::<T, BD>(dst, dst_stride, src, w, h, ncols, s0, &wts);
                }
            }
            (false, false) => {}
        }
    }

    // ---------------------------------------------------------------------------
}

// Bit depth specialized entry points (selected by `Dsp::simd(bd)`).
// `BD` is 8, 10 or 12 and always equals the runtime `bd` argument; the
// pixel / intermediate ranges of `BD` may be relied upon.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn wiener_filter_bd<T: Pixel, const BD: u32>(
    dst: &mut [T],
    dst_stride: usize,
    src: &[&[T]],
    w: usize,
    h: usize,
    coeffs: &[[i32; 3]; 2],
    bd: u32,
) {
    debug_assert_eq!(bd, BD);
    // The decoder uses `u8` pixels iff the bit depth is 8. The tests also
    // run `u16` pixels at bit depth 8, which every kernel supports.
    debug_assert!(cfg!(test) || (T::MAX_BITDEPTH == 8) == (BD == 8));
    match BD {
        8 => narrow::wiener_filter::<T, 8>(dst, dst_stride, src, w, h, coeffs),
        10 => narrow::wiener_filter::<T, 10>(dst, dst_stride, src, w, h, coeffs),
        _ => wiener_filter(dst, dst_stride, src, w, h, coeffs, BD),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn sgr_filter_bd<T: Pixel, const BD: u32>(
    dst: &mut [T],
    dst_stride: usize,
    src: &[&[T]],
    w: usize,
    h: usize,
    set: usize,
    xqd: [i32; 2],
    bd: u32,
) {
    debug_assert_eq!(bd, BD);
    // The decoder uses `u8` pixels iff the bit depth is 8. The tests also
    // run `u16` pixels at bit depth 8, which every kernel supports.
    debug_assert!(cfg!(test) || (T::MAX_BITDEPTH == 8) == (BD == 8));
    match BD {
        8 => narrow::sgr_filter::<T, 8>(dst, dst_stride, src, w, h, set, xqd),
        10 => narrow::sgr_filter::<T, 10>(dst, dst_stride, src, w, h, set, xqd),
        _ => sgr_filter(dst, dst_stride, src, w, h, set, xqd, BD),
    }
}

/// Self guided filter of a run of adjacent restoration units with the
/// same parameter set (see [`narrow::sgr_filter_units`]): `units` holds
/// the width and `xqd` of each unit, `w` their total width (at most
/// `LR_MAX_W`), and every width but the last is a multiple of 16.
/// Equivalent to [`sgr_filter_bd`] on each unit.
#[allow(clippy::too_many_arguments)]
pub fn sgr_filter_units<T: Pixel>(
    dst: &mut [T],
    dst_stride: usize,
    src: &[&[T]],
    w: usize,
    h: usize,
    set: usize,
    units: &[(usize, [i32; 2])],
    bd: u32,
) {
    match bd {
        8 if T::MAX_BITDEPTH == 8 => {
            narrow::sgr_filter_units::<T, 8>(dst, dst_stride, src, w, h, set, units)
        }
        10 if T::MAX_BITDEPTH != 8 => {
            narrow::sgr_filter_units::<T, 10>(dst, dst_stride, src, w, h, set, units)
        }
        _ => sgr_filter_units_w(dst, dst_stride, src, w, h, set, units, bd),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::av1::dsp::lr::{
        self as portable, LR_MAX_H, SGRPROJ_XQD_MAX, SGRPROJ_XQD_MIN, SGR_A2, WIENER_TAPS_MAX,
        WIENER_TAPS_MIN,
    };

    /// The rows of a padded source buffer with row stride `stride`.
    fn rows_of<T>(buf: &[T], stride: usize) -> Vec<&[T]> {
        buf.chunks(stride).collect()
    }

    /// Small deterministic PRNG (xorshift64*).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }
        fn range(&mut self, lo: i64, hi: i64) -> i64 {
            lo + (self.next() % (hi - lo + 1) as u64) as i64
        }
    }

    /// Random padded source: uniform, extremes only (maximum variance),
    /// smooth with small noise, or a mix of 0 / max / one flat value.
    fn make_src<T: Pixel>(rng: &mut Rng, len: usize, bd: u32) -> Vec<T> {
        let max = (1i64 << bd) - 1;
        let mode = rng.range(0, 3);
        let base = rng.range(0, max);
        let amp = rng.range(0, 64);
        (0..len)
            .map(|_| {
                let v = match mode {
                    0 => rng.range(0, max),
                    1 => [0, max][rng.range(0, 1) as usize],
                    2 => (base + rng.range(-amp, amp)).clamp(0, max),
                    _ => [0, max, base][rng.range(0, 2) as usize],
                };
                T::from_i32(v as i32)
            })
            .collect()
    }

    /// Sizes: every width in 1..=40 and around multiples of `N`, the
    /// largest rectangle, and random ones.
    fn sizes(rng: &mut Rng, n_random: usize) -> Vec<(usize, usize)> {
        let mut v: Vec<(usize, usize)> = (1..=40).map(|w| (w, rng.range(1, 6) as usize)).collect();
        for w in [
            LR_MAX_W - 9,
            LR_MAX_W - 8,
            LR_MAX_W - 7,
            LR_MAX_W - 1,
            LR_MAX_W,
        ] {
            v.push((w, rng.range(1, 3) as usize));
        }
        v.push((LR_MAX_W, LR_MAX_H));
        v.push((1, LR_MAX_H));
        for _ in 0..n_random {
            v.push((
                rng.range(1, LR_MAX_W as i64) as usize,
                rng.range(1, LR_MAX_H as i64) as usize,
            ));
        }
        v
    }

    /// Runs `f` (SIMD kernel or portable kernel, selected by the bool) on
    /// a random source of every size and compares the outputs, including
    /// the destination padding that must stay untouched.
    fn compare<T: Pixel>(
        rng: &mut Rng,
        bd: u32,
        n_random: usize,
        what: &str,
        f: &dyn Fn(bool, &mut [T], usize, &[&[T]], usize, usize),
    ) {
        for (w, h) in sizes(rng, n_random) {
            let stride = w + 2 * LR_BORDER + rng.range(0, 5) as usize;
            // The buffer ends right after the last padded sample, or
            // `LR_SRC_SLACK` samples later (full-vector loads at row ends).
            let slack = [0, portable::LR_SRC_SLACK][rng.range(0, 1) as usize];
            let len = stride * (h + 2 * LR_BORDER - 1) + w + 2 * LR_BORDER + slack;
            let src = make_src::<T>(rng, len, bd);
            let rows: Vec<&[T]> = (0..h + 2 * LR_BORDER)
                .map(|r| &src[r * stride..(r * stride + w + 2 * LR_BORDER + slack).min(len)])
                .collect();
            let dst_stride = w + rng.range(0, 3) as usize;
            let dst_len = dst_stride * (h - 1) + w;
            let fill = T::from_i32(rng.range(0, (1 << bd) - 1) as i32);
            let mut got = vec![fill; dst_len];
            let mut want = vec![fill; dst_len];
            f(true, &mut got, dst_stride, &rows, w, h);
            f(false, &mut want, dst_stride, &rows, w, h);
            assert!(got == want, "{what} bd={bd} {w}x{h}");
        }
    }

    fn run_wiener<T: Pixel>(bd: u32, seed: u64) {
        let mut rng = Rng(seed);
        for _ in 0..12 {
            let mut coeffs = [[0i32; 3]; 2];
            for c in coeffs.iter_mut() {
                for (k, t) in c.iter_mut().enumerate() {
                    *t = match rng.range(0, 3) {
                        0 => WIENER_TAPS_MIN[k],
                        1 => WIENER_TAPS_MAX[k],
                        _ => rng.range(WIENER_TAPS_MIN[k] as i64, WIENER_TAPS_MAX[k] as i64) as i32,
                    };
                }
                if rng.range(0, 3) == 0 {
                    c[0] = 0; // chroma style 5-tap filter
                }
            }
            let what = format!("wiener {coeffs:?}");
            compare::<T>(&mut rng, bd, 8, &what, &|simd, d, ds, s, w, h| {
                if simd {
                    wiener_filter(d, ds, s, w, h, &coeffs, bd)
                } else {
                    portable::wiener_filter(d, ds, s, w, h, &coeffs, bd)
                }
            });
        }
    }

    fn run_wiener_bd<T: Pixel, const BD: u32>(seed: u64) {
        let mut rng = Rng(seed);
        for iter in 0..12 {
            let mut coeffs = [[0i32; 3]; 2];
            for c in coeffs.iter_mut() {
                for (k, t) in c.iter_mut().enumerate() {
                    *t = match rng.range(0, 3) {
                        0 => WIENER_TAPS_MIN[k],
                        1 => WIENER_TAPS_MAX[k],
                        _ => rng.range(WIENER_TAPS_MIN[k] as i64, WIENER_TAPS_MAX[k] as i64) as i32,
                    };
                }
                if rng.range(0, 3) == 0 {
                    c[0] = 0; // chroma style 5-tap filter
                }
            }
            if iter < 4 {
                if iter & 1 != 0 {
                    coeffs[0][0] = 0;
                }
                if iter & 2 != 0 {
                    coeffs[1][0] = 0;
                }
            }
            let what = format!("wiener_bd {coeffs:?}");
            compare::<T>(&mut rng, BD, 8, &what, &|simd, d, ds, s, w, h| {
                if simd {
                    wiener_filter_bd::<T, BD>(d, ds, s, w, h, &coeffs, BD)
                } else {
                    portable::wiener_filter(d, ds, s, w, h, &coeffs, BD)
                }
            });
        }
    }

    #[test]
    fn wiener_matches_portable_8bit() {
        run_wiener::<u8>(8, 0x1111_5678_9abc_def1);
    }

    #[test]
    fn wiener_matches_portable_10bit() {
        run_wiener::<u16>(10, 0x2222_5678_9abc_def1);
    }

    #[test]
    fn wiener_matches_portable_12bit() {
        run_wiener::<u16>(12, 0x3333_5678_9abc_def1);
    }

    #[test]
    fn wiener_bd_matches_portable_8bit() {
        run_wiener_bd::<u8, 8>(0x1111_5678_9abc_def1);
    }

    /// Every combination of extreme horizontal taps on 0 / 255 sources,
    /// which reach the bounds of the 16-bit 8-bit horizontal pass.
    #[test]
    fn wiener_bd_8bit_extreme_taps() {
        let mut rng = Rng(0x9999_5678_9abc_def1);
        let (w, h) = (64, 8);
        let stride = w + 2 * LR_BORDER;
        for k in 0..8 {
            let pick =
                |i: usize| if k >> i & 1 == 0 { WIENER_TAPS_MIN[i] } else { WIENER_TAPS_MAX[i] };
            let coeffs = [[3, -7, 15], [pick(0), pick(1), pick(2)]];
            for mode in 0..3 {
                let src: Vec<u8> = (0..stride * (h + 2 * LR_BORDER))
                    .map(|_| match mode {
                        0 => [0, 255][rng.range(0, 1) as usize],
                        1 => [0, 255, 128, 1, 254][rng.range(0, 4) as usize],
                        _ => rng.range(0, 255) as u8,
                    })
                    .collect();
                let mut got = vec![0u8; w * h];
                let mut want = vec![0u8; w * h];
                wiener_filter_bd::<u8, 8>(&mut got, w, &rows_of(&src, stride), w, h, &coeffs, 8);
                portable::wiener_filter(&mut want, w, &rows_of(&src, stride), w, h, &coeffs, 8);
                assert!(got == want, "{coeffs:?} mode={mode}");
            }
        }
    }

    #[test]
    fn wiener_bd_matches_portable_10bit() {
        run_wiener_bd::<u16, 10>(0x2222_5678_9abc_def1);
    }

    #[test]
    fn wiener_bd_matches_portable_12bit() {
        run_wiener_bd::<u16, 12>(0x3333_5678_9abc_def1);
    }

    fn random_xqd(rng: &mut Rng, set: usize) -> [i32; 2] {
        let mut xqd = [0i32; 2];
        for (k, x) in xqd.iter_mut().enumerate() {
            *x = match rng.range(0, 3) {
                0 => SGRPROJ_XQD_MIN[k],
                1 => SGRPROJ_XQD_MAX[k],
                _ => rng.range(SGRPROJ_XQD_MIN[k] as i64, SGRPROJ_XQD_MAX[k] as i64) as i32,
            };
        }
        // As in the bitstream: a disabled pass forces its weight.
        if SGR_PARAMS[set][0] == 0 {
            xqd[0] = 0;
        }
        if SGR_PARAMS[set][2] == 0 {
            xqd[1] = (1 << SGRPROJ_PRJ_BITS) - xqd[0];
        }
        xqd
    }

    fn run_sgr<T: Pixel>(bd: u32, seed: u64) {
        let mut rng = Rng(seed);
        for set in 0..SGR_PARAMS.len() {
            for _ in 0..2 {
                let xqd = random_xqd(&mut rng, set);
                let what = format!("sgr set={set} xqd={xqd:?}");
                compare::<T>(&mut rng, bd, 4, &what, &|simd, d, ds, s, w, h| {
                    if simd {
                        sgr_filter(d, ds, s, w, h, set, xqd, bd)
                    } else {
                        portable::sgr_filter(d, ds, s, w, h, set, xqd, bd)
                    }
                });
            }
        }
    }

    fn run_sgr_bd<T: Pixel, const BD: u32>(seed: u64) {
        let mut rng = Rng(seed);
        for set in 0..SGR_PARAMS.len() {
            for _ in 0..2 {
                let xqd = random_xqd(&mut rng, set);
                let what = format!("sgr_bd set={set} xqd={xqd:?}");
                compare::<T>(&mut rng, BD, 4, &what, &|simd, d, ds, s, w, h| {
                    if simd {
                        sgr_filter_bd::<T, BD>(d, ds, s, w, h, set, xqd, BD)
                    } else {
                        portable::sgr_filter(d, ds, s, w, h, set, xqd, BD)
                    }
                });
            }
        }
    }

    #[test]
    fn sgr_matches_portable_8bit() {
        run_sgr::<u8>(8, 0x4444_5678_9abc_def1);
    }

    #[test]
    fn sgr_matches_portable_10bit() {
        run_sgr::<u16>(10, 0x5555_5678_9abc_def1);
    }

    #[test]
    fn sgr_matches_portable_12bit() {
        run_sgr::<u16>(12, 0x6666_5678_9abc_def1);
    }

    /// Patterns maximising the variance (p * s) and b, at every bit depth.
    fn run_sgr_extremes<T: Pixel, const BD: u32>() {
        let (w, h) = (LR_MAX_W, 16);
        let stride = w + 2 * LR_BORDER;
        let rows = h + 2 * LR_BORDER;
        let max = (1 << BD) - 1;
        let patterns: [&dyn Fn(usize, usize) -> i32; 5] = [
            &|_, _| max,
            &|x, y| if (x + y) & 1 == 0 { 0 } else { max },
            &|x, _| if x & 1 == 0 { 0 } else { max },
            &|x, y| if (x / 3 + y / 2) & 1 == 0 { 0 } else { max },
            &|x, y| if (x / 2 + y / 3) & 1 == 0 { max / 2 } else { max },
        ];
        for pat in patterns {
            let src: Vec<T> = (0..stride * rows)
                .map(|k| T::from_i32(pat(k % stride, k / stride)))
                .collect();
            for set in 0..SGR_PARAMS.len() {
                for xqd in [[-96, 95], [31, -32], [0, 0], [-96, -32]] {
                    let mut got = vec![T::default(); w * h];
                    let mut got_bd = vec![T::default(); w * h];
                    let mut want = vec![T::default(); w * h];
                    sgr_filter(&mut got, w, &rows_of(&src, stride), w, h, set, xqd, BD);
                    sgr_filter_bd::<T, BD>(
                        &mut got_bd,
                        w,
                        &rows_of(&src, stride),
                        w,
                        h,
                        set,
                        xqd,
                        BD,
                    );
                    portable::sgr_filter(&mut want, w, &rows_of(&src, stride), w, h, set, xqd, BD);
                    assert!(got == want, "bd={BD} set={set} xqd={xqd:?}");
                    assert!(got_bd == want, "bd_spec bd={BD} set={set} xqd={xqd:?}");
                }
            }
        }
    }

    #[test]
    fn sgr_bd_matches_portable_8bit() {
        run_sgr_bd::<u8, 8>(0x4444_5678_9abc_def1);
    }

    /// [`sgr_filter_units`] on a run of units equals [`sgr_filter_bd`] on
    /// each unit.
    fn run_sgr_units<T: Pixel, const BD: u32>(seed: u64) {
        let mut rng = Rng(seed);
        for iter in 0..24 {
            let set = iter % SGR_PARAMS.len();
            let n = 1 + rng.range(0, 5) as usize;
            let unit = [16, 32, 64, 128][rng.range(0, 3) as usize];
            let mut units: Vec<(usize, [i32; 2])> =
                (0..n).map(|_| (unit, random_xqd(&mut rng, set))).collect();
            let last = rng.range(1, unit as i64 * 3 / 2) as usize;
            units.last_mut().unwrap().0 = last;
            let w: usize = units.iter().map(|u| u.0).sum();
            if w > LR_MAX_W {
                continue;
            }
            let h = 1 + rng.range(0, 63) as usize;
            let stride = w + 2 * LR_BORDER;
            let src = make_src::<T>(
                &mut rng,
                stride * (h + 2 * LR_BORDER) + portable::LR_SRC_SLACK,
                BD,
            );
            let rows = rows_of(&src, stride);
            let rows = &rows[..h + 2 * LR_BORDER];
            let mut got = vec![T::default(); w * h];
            let mut want = vec![T::default(); w * h];
            sgr_filter_units(&mut got, w, rows, w, h, set, &units, BD);
            let mut x = 0;
            for &(uw, xqd) in &units {
                let urows: Vec<&[T]> = rows.iter().map(|r| &r[x..]).collect();
                sgr_filter_bd::<T, BD>(&mut want[x..], w, &urows, uw, h, set, xqd, BD);
                x += uw;
            }
            assert!(got == want, "bd={BD} set={set} units={units:?} h={h}");
        }
    }

    #[test]
    fn sgr_units_match_per_unit() {
        run_sgr_units::<u8, 8>(0x7777_5678_9abc_def1);
        run_sgr_units::<u16, 10>(0x8888_5678_9abc_def1);
        run_sgr_units::<u16, 12>(0x9999_5678_9abc_def1);
    }

    #[test]
    fn sgr_bd_matches_portable_10bit() {
        run_sgr_bd::<u16, 10>(0x5555_5678_9abc_def1);
    }

    #[test]
    fn sgr_bd_matches_portable_12bit() {
        run_sgr_bd::<u16, 12>(0x6666_5678_9abc_def1);
    }

    #[test]
    fn sgr_extremes() {
        run_sgr_extremes::<u8, 8>();
        run_sgr_extremes::<u16, 10>();
        run_sgr_extremes::<u16, 12>();
    }

    /// Micro benchmark of the bit depth specialized kernels on 64 row
    /// stripes of a 384 wide unit:
    /// `cargo test --release lr_bench -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn lr_bench() {
        fn run<T: Pixel, const BD: u32>(name: &str) {
            let (w, h) = (LR_MAX_W, 64);
            let stride = w + 2 * LR_BORDER;
            let mut rng = Rng(0x7777_1234);
            let src: Vec<T> = (0..stride * (h + 2 * LR_BORDER))
                .map(|k| {
                    let (x, y) = ((k % stride) as i64, (k / stride) as i64);
                    let v = (x * 3 + y * 5) % (1 << BD) + rng.range(-8, 8);
                    T::from_i32(v.clamp(0, (1 << BD) - 1) as i32)
                })
                .collect();
            let mut dst = vec![T::default(); w * h];
            // Minimum over trials of the mean time of `iters` calls.
            let time = |f: &mut dyn FnMut()| {
                (0..7)
                    .map(|_| {
                        let t = std::time::Instant::now();
                        for _ in 0..100 {
                            f();
                        }
                        t.elapsed().as_secs_f64() / 100.0
                    })
                    .fold(f64::MAX, f64::min)
            };
            let mut tw = [0f64; 2];
            for (k, coeffs) in [[[3, -7, 15], [-4, -12, 30]], [[0, -7, 15], [0, -12, 30]]]
                .iter()
                .enumerate()
            {
                tw[k] = time(&mut || {
                    wiener_filter_bd::<T, BD>(&mut dst, w, &rows_of(&src, stride), w, h, coeffs, BD)
                });
            }
            let mut ts = [0f64; 3];
            for (k, set) in [0usize, 10, 14].into_iter().enumerate() {
                ts[k] = time(&mut || {
                    sgr_filter_bd::<T, BD>(
                        &mut dst,
                        w,
                        &rows_of(&src, stride),
                        w,
                        h,
                        set,
                        [-32, 31],
                        BD,
                    )
                });
            }
            println!(
                "{name}: wiener7 {:.2} us, wiener5 {:.2} us, sgr both {:.1} us, r1 {:.1} us, r2 {:.1} us",
                tw[0] * 1e6,
                tw[1] * 1e6,
                ts[0] * 1e6,
                ts[1] * 1e6,
                ts[2] * 1e6
            );
            std::hint::black_box(&dst);
        }
        run::<u8, 8>("8-bit");
        run::<u16, 10>("10-bit");
    }

    #[test]
    fn sgr_x_exhaustive() {
        for z in 0..=255u32 {
            let want = 256 - SGR_A2[z as usize] as i32;
            assert_eq!(
                narrow::sgr_x(Simd::splat(z as i32)),
                Simd::splat(want),
                "z={z}"
            );
        }
        // Mixed lanes, so that each lane's lookup is independent.
        let z: Simd<i32, 16> = Simd::from_array(std::array::from_fn(|k| (k * 17) as i32));
        let want = Simd::from_array(std::array::from_fn(|k| 256 - SGR_A2[k * 17] as i32));
        assert_eq!(narrow::sgr_x(z), want);
    }

    #[test]
    fn sgr_a2_exhaustive() {
        for z in 0..=255u32 {
            assert_eq!(
                sgr_a2(I32s::splat(z as i32)),
                I32s::splat(SGR_A2[z as usize] as i32),
                "z={z}"
            );
        }
    }

    /// `x` from `p` (any sign, `|p| < 2^26`) equals the spec's for every
    /// scale: all `p` near the `z` steps are checked, the rest sparsely.
    #[test]
    fn sgr_x_from_p() {
        let max_p: i64 = 1 << 26;
        for &s in SGR_SCALE.iter().flatten().filter(|&&s| s != 0) {
            let want = |p: i64| -> i32 {
                let z = ((p.max(0) * s as i64 + (1 << (SGRPROJ_MTABLE_BITS - 1)))
                    >> SGRPROJ_MTABLE_BITS)
                    .min(255);
                256 - SGR_A2[z as usize] as i32
            };
            let mut ps: Vec<i64> = (-max_p + 1..max_p).step_by(4099).collect();
            ps.extend([-max_p + 1, -1, 0, 1, max_p - 1]);
            // Around every step of z (and x).
            for z in 0..=256i64 {
                let p0 = ((z << SGRPROJ_MTABLE_BITS) - (1 << (SGRPROJ_MTABLE_BITS - 1))) / s as i64;
                ps.extend(p0 - 3..p0 + 4);
            }
            for chunk in ps.chunks(16) {
                let mut a = [0i32; 16];
                for (d, &p) in a.iter_mut().zip(chunk) {
                    *d = p.clamp(-max_p + 1, max_p - 1) as i32;
                }
                let got = narrow::sgr_x_from_p(Simd::from_array(a), s);
                for (k, &p) in a.iter().enumerate() {
                    assert_eq!(got[k], want(p as i64), "s={s} p={p}");
                }
            }
        }
    }

    /// The clamped 32-bit `z` of `box_ab_row` equals the 64-bit one of
    /// the portable kernel for every scale and every reachable `p`
    /// (`p <= 25 * 25 * 2^16`; values around the clamp are checked
    /// exhaustively, the rest sparsely).
    #[test]
    fn sgr_z_clamp() {
        let max_p = 25 * 25 * 65536u32;
        for &s in SGR_SCALE.iter().flatten().filter(|&&s| s != 0) {
            let pmax =
                ((255u32 << SGRPROJ_MTABLE_BITS) - (1 << (SGRPROJ_MTABLE_BITS - 1))).div_ceil(s);
            let near = pmax.saturating_sub(100_000)..pmax + 100_000;
            for p in near.chain((0..=max_p).step_by(997)).chain([max_p]) {
                let want = ((p as u64 * s as u64 + (1 << (SGRPROJ_MTABLE_BITS - 1)))
                    >> SGRPROJ_MTABLE_BITS)
                    .min(255) as u32;
                let got =
                    (p.min(pmax) * s + (1 << (SGRPROJ_MTABLE_BITS - 1))) >> SGRPROJ_MTABLE_BITS;
                assert_eq!(got, want, "s={s} p={p}");
            }
        }
    }
}
