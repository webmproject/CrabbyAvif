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

//! SIMD CDEF kernels (see `dsp::cdef` for the contracts).

use std::simd::prelude::*;

use crate::decoder::av1::dsp::cdef::{
    damping_adj, pad_block, tap_offset, CdefKernels, CDEF_BORDER, CDEF_HAVE_ALL, CDEF_PAD_STRIDE,
    CDEF_PRI_TAPS, FAST_MAX_STRENGTH,
};
use crate::decoder::av1::pixel::Pixel;

/// Per-block parameters of [`filter_block`].
///
/// The taps are held as windows of the (sliced) source, one per tap
/// position: window `win` of tap offset `off` starts at sample `(2 +
/// dy, 2 + dx)` of the bordered block and holds the `(H - 1) * src_stride
/// + W` samples covering all loads of that tap (see [`load_i16`]). Only
/// building the windows is bounds checked (once per block), not the
/// loads.
struct Params<'a, S, const L: usize> {
    pri_str: Simd<u16, L>,
    sec_str: Simd<u16, L>,
    pri_adj: u16,
    sec_adj: u16,
    pri_taps: [Simd<i16, L>; 2],
    /// The center samples.
    center: &'a [S],
    /// Primary tap `k` (direction `dir`) at `+off` / `-off`.
    pri: [[&'a [S]; 2]; 2],
    /// Secondary tap `k` (directions `dir -/+ 2`) at `+off` / `-off`.
    sec: [[[&'a [S]; 2]; 2]; 2],
}

impl<'a, S: Pixel, const L: usize> Params<'a, S, L> {
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn new<const W: usize, const H: usize, const PRI: bool, const SEC: bool>(
        src: &'a [S],
        src_stride: usize,
        pri_str: i32,
        sec_str: i32,
        damping: i32,
        dir: usize,
        tap_set: usize,
    ) -> Self {
        let len = (H - 1) * src_stride + W;
        let base = CDEF_BORDER * src_stride + CDEF_BORDER;
        let win = |off: isize| -> &'a [S] {
            let start = (base as isize + off) as usize;
            &src[start..start + len]
        };
        let pair = |d: usize, k: usize| {
            let off = tap_offset(d, k, src_stride);
            [win(off), win(-off)]
        };
        let center = win(0);
        let (s0, s1) = ((dir + 6) & 7, (dir + 2) & 7);
        Self {
            pri_str: if PRI { Simd::splat(pri_str as u16) } else { Simd::splat(0) },
            sec_str: if SEC { Simd::splat(sec_str as u16) } else { Simd::splat(0) },
            pri_adj: if PRI { damping_adj(pri_str, damping) as u16 } else { 0 },
            sec_adj: if SEC { damping_adj(sec_str, damping) as u16 } else { 0 },
            pri_taps: [
                Simd::splat(CDEF_PRI_TAPS[tap_set][0] as i16),
                Simd::splat(CDEF_PRI_TAPS[tap_set][1] as i16),
            ],
            center,
            pri: if PRI { [pair(dir, 0), pair(dir, 1)] } else { [[center; 2]; 2] },
            sec: if SEC {
                [[pair(s0, 0), pair(s1, 0)], [pair(s0, 1), pair(s1, 1)]]
            } else {
                [[[center; 2]; 2]; 2]
            },
        }
    }
}

/// The 16-bit lane path of [`cdef_filter`] (kept out of line so that the
/// dispatcher stays small). `src` holds `u16` samples with
/// [`CDEF_VERY_LARGE`](crate::decoder::av1::dsp::cdef::CDEF_VERY_LARGE) for unavailable
/// ones (`SENT`), or pixels (all available, `!SENT`).
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn filter_i16<T: Pixel, S: Pixel, const SENT: bool>(
    dst: &mut [T],
    dst_stride: usize,
    src: &[S],
    src_stride: usize,
    w: usize,
    h: usize,
    pri_str: i32,
    sec_str: i32,
    damping: i32,
    dir: usize,
    tap_set: usize,
) {
    macro_rules! size {
        ($w:literal, $h:literal) => {
            match (pri_str != 0, sec_str != 0) {
                (true, true) => filter_block::<T, S, $w, $h, true, true, SENT, LANES>(
                    dst, dst_stride, src, src_stride, pri_str, sec_str, damping, dir, tap_set,
                ),
                (true, false) => filter_block::<T, S, $w, $h, true, false, SENT, LANES>(
                    dst, dst_stride, src, src_stride, pri_str, sec_str, damping, dir, tap_set,
                ),
                _ => filter_block::<T, S, $w, $h, false, true, SENT, LANES>(
                    dst, dst_stride, src, src_stride, pri_str, sec_str, damping, dir, tap_set,
                ),
            }
        };
    }
    match (w, h) {
        (8, 8) => size!(8, 8),
        (8, 4) => size!(8, 4),
        (4, 8) => size!(4, 8),
        _ => size!(4, 4),
    }
}

/// The lanes of the 16-bit path: 16 (two rows of an 8 wide block or
/// four rows of a 4 wide one) with AVX2, 8 otherwise.
const LANES: usize =
    if cfg!(all(target_arch = "x86_64", target_feature = "avx2")) { 16 } else { 8 };

/// Loads the `L` lanes of rows `i..i + L / W` (`W` samples each) of a
/// tap window (see [`Params`]), `pos = i * src_stride`.
#[inline(always)]
fn load_i16<const W: usize, const L: usize, S: Pixel>(
    win: &[S],
    pos: usize,
    src_stride: usize,
) -> Simd<u16, L> {
    if L == 8 {
        Simd::from_slice(load8_i16::<W, S>(win, pos, src_stride).as_array())
    } else {
        let a = load8_i16::<W, S>(win, pos, src_stride);
        let b = load8_i16::<W, S>(win, pos + 8 / W * src_stride, src_stride);
        let v: u16x16 = simd_swizzle!(a, b, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        Simd::from_slice(v.as_array())
    }
}

/// Loads the 8 lanes of rows `i..i + 8 / W` of a tap window.
#[inline(always)]
fn load8_i16<const W: usize, S: Pixel>(win: &[S], pos: usize, src_stride: usize) -> u16x8 {
    if W == 8 {
        u16x8::from_array(load_u16::<S, 8>(&win[pos..]))
    } else {
        let a = u16x4::from_array(load_u16::<S, 4>(&win[pos..]));
        let b = u16x4::from_array(load_u16::<S, 4>(&win[pos + src_stride..]));
        simd_swizzle!(a, b, [0, 1, 2, 3, 4, 5, 6, 7])
    }
}

/// Loads `N` samples as `u16`.
#[inline(always)]
fn load_u16<S: Pixel, const N: usize>(src: &[S]) -> [u16; N] {
    let s: &[S; N] = src[..N].try_into().unwrap();
    let mut v = [0u16; N];
    for (d, &p) in v.iter_mut().zip(s) {
        *d = p.to_u16();
    }
    v
}

/// Spec `constrain(d, strength, damping)` lane-wise, with `d = p - x` and
/// the damping adjustment `adj` precomputed (`strength >= 0`, `|d| <= 30000`).
///
/// `sign(d) * Min(|d|, t)` with `t = Max(0, strength - (|d| >> adj))`
/// is computed as `Clip3(-t, t, d)`.
#[inline(always)]
fn constrain_i16<const L: usize>(
    d: Simd<i16, L>,
    strength: Simd<u16, L>,
    adj: u16,
) -> Simd<i16, L> {
    let ad = d.abs().cast::<u16>();
    let t = strength
        .saturating_sub(ad >> Simd::splat(adj))
        .cast::<i16>();
    d.simd_max(-t).simd_min(t)
}

/// constrain() of a symmetric tap pair (+off, -off) of [`filter_lanes`],
/// also updating `min_d` / `max_d` when `CLAMP`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tap_pair<S: Pixel, const W: usize, const L: usize, const CLAMP: bool, const MASK: bool>(
    win: [&[S]; 2],
    pos: usize,
    src_stride: usize,
    x: Simd<i16, L>,
    strength: Simd<u16, L>,
    adj: u16,
    min_d: &mut Simd<i16, L>,
    max_d: &mut Simd<i16, L>,
) -> Simd<i16, L> {
    let d0 = load_i16::<W, L, S>(win[0], pos, src_stride).cast::<i16>() - x;
    let d1 = load_i16::<W, L, S>(win[1], pos, src_stride).cast::<i16>() - x;
    if CLAMP {
        *min_d = min_d.simd_min(d0).simd_min(d1);
        if MASK {
            let (z, c4000) = (Simd::splat(0), Simd::splat(0x4000));
            let m0 = d0.simd_ge(c4000).select(z, d0);
            let m1 = d1.simd_ge(c4000).select(z, d1);
            *max_d = max_d.simd_max(m0).simd_max(m1);
        } else {
            *max_d = max_d.simd_max(d0).simd_max(d1);
        }
    }
    constrain_i16(d0, strength, adj) + constrain_i16(d1, strength, adj)
}

/// Filters the `L` lanes of rows `i..i + L / W` (`pos = i * src_stride`,
/// see [`filter_block`]), returning `(filtered, max_d)`.
#[inline(always)]
fn filter_lanes<
    S: Pixel,
    const W: usize,
    const L: usize,
    const PRI: bool,
    const SEC: bool,
    const MASK: bool,
>(
    pos: usize,
    src_stride: usize,
    k: &Params<S, L>,
) -> (Simd<u16, L>, Simd<i16, L>) {
    let x = load_i16::<W, L, S>(k.center, pos, src_stride).cast::<i16>();
    let z = Simd::splat(0);
    let mut min_d = z;
    let mut max_d = z;
    macro_rules! pair {
        ($win:expr, $strength:expr, $adj:expr) => {
            if PRI && SEC {
                tap_pair::<S, W, L, true, MASK>(
                    $win, pos, src_stride, x, $strength, $adj, &mut min_d, &mut max_d,
                )
            } else {
                tap_pair::<S, W, L, false, MASK>(
                    $win, pos, src_stride, x, $strength, $adj, &mut min_d, &mut max_d,
                )
            }
        };
    }
    // The taps sharing a weight are summed before the multiplication.
    let mut sum = z;
    if PRI {
        let s0 = pair!(k.pri[0], k.pri_str, k.pri_adj);
        let s1 = pair!(k.pri[1], k.pri_str, k.pri_adj);
        sum += k.pri_taps[0] * s0 + k.pri_taps[1] * s1;
    }
    if SEC {
        let s0 =
            pair!(k.sec[0][0], k.sec_str, k.sec_adj) + pair!(k.sec[0][1], k.sec_str, k.sec_adj);
        let s1 =
            pair!(k.sec[1][0], k.sec_str, k.sec_adj) + pair!(k.sec[1][1], k.sec_str, k.sec_adj);
        sum += (s0 << 1) + s1;
    }
    let mut d = (Simd::splat(8) + sum + (sum >> 15)) >> 4;
    if PRI && SEC {
        d = d.simd_max(min_d).simd_min(max_d);
    }
    ((x + d).cast::<u16>(), max_d)
}

#[inline(always)]
fn store_i16<T: Pixel, const W: usize, const L: usize>(
    dst: &mut [T],
    dst_stride: usize,
    i: usize,
    v: Simd<u16, L>,
) {
    let v = v.to_array();
    for r in 0..L / W {
        let start = (i + r) * dst_stride;
        let row: &mut [T; W] = (&mut dst[start..start + W]).try_into().unwrap();
        for j in 0..W {
            row[j] = T::from_i32(v[r * W + j] as i32);
        }
    }
}

/// Cold fallback when an unavailable ([`CDEF_VERY_LARGE`]) tap reached `max`
/// during a `PRI && SEC` block.
#[allow(clippy::too_many_arguments)]
#[cold]
#[inline(never)]
fn filter_block_masked<T: Pixel, S: Pixel, const W: usize, const H: usize, const L: usize>(
    dst: &mut [T],
    dst_stride: usize,
    src: &[S],
    src_stride: usize,
    pri_str: i32,
    sec_str: i32,
    damping: i32,
    dir: usize,
    tap_set: usize,
) {
    let k = Params::<S, L>::new::<W, H, true, true>(
        src, src_stride, pri_str, sec_str, damping, dir, tap_set,
    );
    for i in (0..H).step_by(L / W) {
        let (v, _) = filter_lanes::<S, W, L, true, true, true>(i * src_stride, src_stride, &k);
        store_i16::<T, W, L>(dst, dst_stride, i, v);
    }
}

/// Spec 7.15.3 for `W x H` blocks (`W`, `H` in {4, 8}) in 16-bit lanes,
/// valid under the same conditions as the portable fast path
/// (`damping <= 13`, strengths below [`FAST_MAX_STRENGTH`]), see
/// `dsp::cdef` for why that is exact:
///
/// * Taps are loaded directly from `src` in symmetric pairs without an
///   intermediate stack copy.
/// * constrain() of an unavailable ([`CDEF_VERY_LARGE`]) tap is 0 and
///   such a tap never lowers `min`, so it only matters for `max`.
/// * `max` is first computed without masking; valid samples are below
///   4096 (`< 0x4000`), so `all_max` has a lane `>= 0x4000` iff one of the
///   taps was unavailable. Only then (near frame edges) is the block
///   recomputed with unavailable taps masked out.
/// * With a single non-zero strength the final clamp is a no-op and is
///   skipped (see `dsp::cdef::filter_fast`).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn filter_block<
    T: Pixel,
    S: Pixel,
    const W: usize,
    const H: usize,
    const PRI: bool,
    const SEC: bool,
    const SENT: bool,
    const L: usize,
>(
    dst: &mut [T],
    dst_stride: usize,
    src: &[S],
    src_stride: usize,
    pri_str: i32,
    sec_str: i32,
    damping: i32,
    dir: usize,
    tap_set: usize,
) {
    assert!(src_stride >= W + 2 * CDEF_BORDER && src_stride <= 0xffff_ffff);
    assert!(dst_stride <= 0xffff_ffff);
    let src = &src[..(H + 2 * CDEF_BORDER - 1) * src_stride + W + 2 * CDEF_BORDER];
    let dst = &mut dst[..(H - 1) * dst_stride + W];
    let k = Params::<S, L>::new::<W, H, PRI, SEC>(
        src, src_stride, pri_str, sec_str, damping, dir, tap_set,
    );
    let mut all_max = Simd::<i16, L>::splat(0);
    for i in (0..H).step_by(L / W) {
        let (v, max) = filter_lanes::<S, W, L, PRI, SEC, false>(i * src_stride, src_stride, &k);
        if SENT && PRI && SEC {
            all_max = all_max.simd_max(max);
        }
        store_i16::<T, W, L>(dst, dst_stride, i, v);
    }
    if SENT && PRI && SEC && all_max.simd_ge(Simd::splat(0x4000)).any() {
        filter_block_masked::<T, S, W, H, L>(
            dst, dst_stride, src, src_stride, pri_str, sec_str, damping, dir, tap_set,
        );
    }
}

/// Eight 16-bit lanes: one row of an 8 wide block or two rows of a 4
/// wide one.
macro_rules! cdef_impl {
    ($BD:literal) => {
/// `dsp::cdef::cdef_filter` with `std::simd`.
///
/// Spec-conformant parameters (and everything else the portable 16-bit
/// fast path accepts) use [`filter_block`]; any other input is passed to
/// the portable kernel, so that the output is identical for all inputs.
#[allow(clippy::too_many_arguments)]
pub(super) fn cdef_filter<T: Pixel>(dst: &mut [T], dst_stride: usize, src: &[u16], src_stride: usize, w: usize, h: usize, pri_str: i32, sec_str: i32, damping: i32, dir: usize) {
    let fast = damping <= 13
        && (0..FAST_MAX_STRENGTH).contains(&pri_str)
        && (0..FAST_MAX_STRENGTH).contains(&sec_str)
        && (pri_str != 0 || sec_str != 0)
        && matches!((w, h), (8, 8) | (8, 4) | (4, 8) | (4, 4));
    if !fast {
        crate::decoder::av1::dsp::cdef::cdef_filter(dst, dst_stride, src, src_stride, w, h, pri_str, sec_str, damping, dir, BD);
        return;
    }
    let tap_set = ((pri_str >> (BD - 8)) & 1) as usize;
    filter_i16::<T, u16, true>(dst, dst_stride, src, src_stride, w, h, pri_str, sec_str, damping, dir, tap_set);
}

/// `dsp::cdef::CdefKernels::filter` with `std::simd`: blocks whose border
/// is inside the frame are filtered straight from the pixel source (8-bit
/// conformant parameters on `u8` lanes, like dav1d's "edged" kernels,
/// the rest on 16-bit lanes); blocks at the frame border get a padded
/// copy with sentinels for [`cdef_filter`].
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn filter_px<T: Pixel>(dst: &mut [T], dst_stride: usize, src: &[T], src_stride: usize, w: usize, h: usize, pri_str: i32, sec_str: i32, damping: i32, dir: usize, edges: u8) {
    let fast = edges == CDEF_HAVE_ALL
        && damping <= 13
        && (0..FAST_MAX_STRENGTH).contains(&pri_str)
        && (0..FAST_MAX_STRENGTH).contains(&sec_str)
        && (pri_str != 0 || sec_str != 0)
        && matches!((w, h), (8, 8) | (8, 4) | (4, 8) | (4, 4));
    if !fast {
        filter_padded(dst, dst_stride, src, src_stride, w, h, pri_str, sec_str, damping, dir, edges);
        return;
    }
    let tap_set = ((pri_str >> (BD - 8)) & 1) as usize;
    if BD == 8 && std::mem::size_of::<T>() == 1 && damping <= 7 && pri_str < 16 && sec_str < 16 {
        match (w, h) {
            (8, 8) => u8_path::filter_8x8_dir(dst, dst_stride, src, src_stride, pri_str, sec_str, damping, dir, tap_set),
            (8, 4) => u8_path::filter::<T, 8, 4>(dst, dst_stride, src, src_stride, pri_str, sec_str, damping, dir, tap_set),
            (4, 8) => u8_path::filter::<T, 4, 8>(dst, dst_stride, src, src_stride, pri_str, sec_str, damping, dir, tap_set),
            _ => u8_path::filter_4x4_dir(dst, dst_stride, src, src_stride, pri_str, sec_str, damping, dir, tap_set),
        }
        return;
    }
    filter_i16::<T, T, false>(dst, dst_stride, src, src_stride, w, h, pri_str, sec_str, damping, dir, tap_set);
}

/// [`filter_px`] of a block at the frame border (or with parameters
/// outside the fast paths): padded copy with sentinels (`w`, `h` <= 8).
#[allow(clippy::too_many_arguments)]
#[cold]
#[inline(never)]
fn filter_padded<T: Pixel>(dst: &mut [T], dst_stride: usize, src: &[T], src_stride: usize, w: usize, h: usize, pri_str: i32, sec_str: i32, damping: i32, dir: usize, edges: u8) {
    let mut tmp = [0u16; CDEF_PAD_STRIDE * CDEF_PAD_STRIDE];
    pad_block(&mut tmp, src, src_stride, w, h, edges);
    cdef_filter(dst, dst_stride, &tmp, CDEF_PAD_STRIDE, w, h, pri_str, sec_str, damping, dir);
}

/// Loads row `pos..pos + 8` of the direction search block as
/// `(px >> shift) - 128`.
#[inline(always)]
fn dir_row<T: Pixel>(src: &[T], pos: usize, shift: i16) -> i16x8 {
    let row: &[T; 8] = src[pos..pos + 8].try_into().unwrap();
    if BD == 8 {
        let mut v = [0u8; 8];
        for (d, &p) in v.iter_mut().zip(row) {
            *d = p.to_i32() as u8;
        }
        u8x8::from_array(v).cast::<i16>() - i16x8::splat(128)
    } else {
        let mut v = [0i16; 8];
        for (d, &p) in v.iter_mut().zip(row) {
            *d = p.to_i32() as i16;
        }
        (i16x8::from_array(v) >> shift) - i16x8::splat(128)
    }
}

/// Lane sums of two vectors: `[a0 + a1, a2 + a3, a4 + a5, a6 + a7,
/// b0 + b1, b2 + b3, b4 + b5, b6 + b7]` (NEON `addp.8h`).
#[inline(always)]
fn addp(a: i16x8, b: i16x8) -> i16x8 {
    let (even, odd) = a.deinterleave(b);
    even + odd
}

/// Window `8 - S..16 - S` of the concatenation of two vectors (NEON
/// `ext`): with `[0, x]` the low half of `x` shifted towards higher
/// lanes by `S` (zero filled), with `[x, 0]` the high half.
struct Ext<const S: usize>;
impl<const S: usize> std::simd::Swizzle<8> for Ext<S> {
    const INDEX: [usize; 8] = {
        let mut r = [0; 8];
        let mut j = 0;
        while j < 8 {
            r[j] = 8 - S + j;
            j += 1;
        }
        r
    };
}

/// Adds `x`, whose lanes `N..8` are 0, shifted towards higher lanes by
/// `S` to the 16-lane accumulator `a` (low half, high half).
#[inline(always)]
fn acc<const S: usize, const N: usize>(a: &mut [i16x8; 2], x: i16x8) {
    use std::simd::Swizzle;
    let z = i16x8::splat(0);
    a[0] += Ext::<S>::concat_swizzle(z, x);
    if S + N > 8 {
        a[1] += Ext::<S>::concat_swizzle(x, z);
    }
}

/// Widening square of 4 `i16` lanes (NEON `smull.4s` / `smull2.4s`).
#[inline(always)]
fn sq4(v: i16x4) -> i32x4 {
    let v: i32x4 = v.cast();
    v * v
}

/// `sum_k p[k]^2 * w[k]` folded to 4 lanes for directions 0 and 4
/// (15 sums: lane 7 of `p[1]` is 0, and weights are symmetric).
#[inline(always)]
fn weighted_diag(p: [i16x8; 2]) -> i32x4 {
    let p0_lo = simd_swizzle!(p[0], [0, 1, 2, 3]);
    let p0_hi = simd_swizzle!(p[0], [4, 5, 6, 7]);
    let p1_hi_rev = simd_swizzle!(p[1], [6, 5, 4, 3]);
    let p1_lo_rev = simd_swizzle!(p[1], [2, 1, 0, 7]);
    (sq4(p0_lo) + sq4(p1_hi_rev)) * i32x4::from_array([840, 420, 280, 210])
        + (sq4(p0_hi) + sq4(p1_lo_rev)) * i32x4::from_array([168, 140, 120, 105])
}

/// `sum_k p[k]^2 * w[k]` folded to 4 lanes for the odd directions (11
/// sums: lanes `3..8` of `p[1]` are 0, and weights are symmetric).
#[inline(always)]
fn weighted_odd(p: [i16x8; 2]) -> i32x4 {
    let p0_lo = simd_swizzle!(p[0], [0, 1, 2, 3]);
    let p0_hi = simd_swizzle!(p[0], [4, 5, 6, 7]);
    let p1_lo_rev = simd_swizzle!(p[1], [2, 1, 0, 3]);
    (sq4(p0_lo) + sq4(p1_lo_rev)) * i32x4::from_array([420, 210, 140, 105])
        + sq4(p0_hi) * i32x4::splat(105)
}

/// `sum_k p[k]^2 * 105` folded to 4 lanes for directions 2 and 6.
#[inline(always)]
fn weighted_hv(p: i16x8) -> i32x4 {
    let lo = simd_swizzle!(p, [0, 1, 2, 3]);
    let hi = simd_swizzle!(p, [4, 5, 6, 7]);
    (sq4(lo) + sq4(hi)) * i32x4::splat(105)
}

/// `[v[0].reduce_sum(), .., v[7].reduce_sum()]` as `(cost[0..4],
/// cost[4..8])`, pairing each direction `d` with `d + 4` in the same
/// lane.
#[inline(always)]
fn hsum4x8(v: [i32x4; 8]) -> (i32x4, i32x4) {
    let pairs = |a: i32x4, b: i32x4| {
        let (even, odd) = a.deinterleave(b);
        even + odd
    };
    let lo = pairs(pairs(v[0], v[1]), pairs(v[2], v[3]));
    let hi = pairs(pairs(v[4], v[5]), pairs(v[6], v[7]));
    (lo, hi)
}

/// `dsp::cdef::cdef_direction` with `std::simd`: spec 7.15.2, returns
/// `(yDir, var)` for the 8x8 block whose top-left sample is `src[0]`.
///
/// The partial sums `partial[d]` are 16-lane vectors to which every row
/// (or pair sums of it) is added shifted by a row dependent number of
/// lanes; directions 3 and 4 are accumulated mirrored like the portable
/// kernel (`p3r[k] = partial[3][10 - k]`, `p4r[k] = partial[4][14 - k]`),
/// which leaves their costs unchanged.
///
/// Exactness: every partial sum has at most 8 terms in `-128..128`, so
/// it fits in `i16`. Squares and costs are computed in `i32`: a cost is
/// at most `840 * 128^2 * 64 < 2^30` (each `partial[d][k]` with `n`
/// terms has weight `840 / n`).
pub(super) fn cdef_direction<T: Pixel>(src: &[T], stride: usize) -> (usize, i32) {
    assert!(stride <= 0x0fff_ffff);
    let src = &src[..7 * stride + 8];
    let shift = (BD - 8) as i16;
    let x: [i16x8; 8] = std::array::from_fn(|i| dir_row(src, i * stride, shift));
    let z0 = i16x8::splat(0);

    // Rows 2k and 2k + 1 have the same offsets for directions 5 and 7:
    // their sum is added once.
    let y = [x[0] + x[1], x[2] + x[3], x[4] + x[5], x[6] + x[7]];
    let p6 = (y[0] + y[1]) + (y[2] + y[3]); // partial[6][j]
    let c6 = weighted_hv(p6);
    let mut p5 = [y[3], z0];
    acc::<3, 8>(&mut p5, y[0]); // partial[5][3 - i / 2 + j]
    acc::<2, 8>(&mut p5, y[1]);
    acc::<1, 8>(&mut p5, y[2]);
    let mut p7 = [y[0], z0];
    acc::<1, 8>(&mut p7, y[1]); // partial[7][i / 2 + j]
    acc::<2, 8>(&mut p7, y[2]);
    acc::<3, 8>(&mut p7, y[3]);
    let (c5, c7) = (weighted_odd(p5), weighted_odd(p7));

    // Horizontal pair sums of rows 2k and 2k + 1 (low and high 4 lanes).
    // Rows i and i + 4 have disjoint 4-lane windows at offsets i and i + 4
    // in p1 (and 7 - i and 3 - i in p3r), so packing [row_i, row_{i+4}]
    // into 8-lane vectors reduces p1 and p3r to 4-step accumulations.
    let s = [addp(x[0], x[1]), addp(x[2], x[3]), addp(x[4], x[5]), addp(x[6], x[7])];
    let p2 = addp(addp(s[0], s[1]), addp(s[2], s[3])); // partial[2][i]
    let c2 = weighted_hv(p2);
    let q = [
        simd_swizzle!(s[0], s[2], [0, 1, 2, 3, 8, 9, 10, 11]),
        simd_swizzle!(s[0], s[2], [4, 5, 6, 7, 12, 13, 14, 15]),
        simd_swizzle!(s[1], s[3], [0, 1, 2, 3, 8, 9, 10, 11]),
        simd_swizzle!(s[1], s[3], [4, 5, 6, 7, 12, 13, 14, 15]),
    ];
    let qr = [
        simd_swizzle!(q[0], [4, 5, 6, 7, 0, 1, 2, 3]),
        simd_swizzle!(q[1], [4, 5, 6, 7, 0, 1, 2, 3]),
        simd_swizzle!(q[2], [4, 5, 6, 7, 0, 1, 2, 3]),
        simd_swizzle!(q[3], [4, 5, 6, 7, 0, 1, 2, 3]),
    ];
    let mut p1 = [q[0], z0];
    acc::<1, 8>(&mut p1, q[1]);
    acc::<2, 8>(&mut p1, q[2]);
    acc::<3, 8>(&mut p1, q[3]);
    let mut p3r = [qr[3], z0];
    acc::<3, 8>(&mut p3r, qr[0]);
    acc::<2, 8>(&mut p3r, qr[1]);
    acc::<1, 8>(&mut p3r, qr[2]);
    let (c1, c3) = (weighted_odd(p1), weighted_odd(p3r));

    let mut p0 = [x[0], z0];
    let mut p4r = [x[7], z0];
    acc::<7, 8>(&mut p4r, x[0]);
    macro_rules! diag_row {
        ($i:literal) => {{
            acc::<$i, 8>(&mut p0, x[$i]); // partial[0][i + j]
            acc::<{ 7 - $i }, 8>(&mut p4r, x[$i]); // partial[4][7 + i - j]
        }};
    }
    diag_row!(1);
    diag_row!(2);
    diag_row!(3);
    diag_row!(4);
    diag_row!(5);
    diag_row!(6);
    acc::<7, 8>(&mut p0, x[7]);
    let (c0, c4) = (weighted_diag(p0), weighted_diag(p4r));

    let (lo, hi) = hsum4x8([c0, c1, c2, c3, c4, c5, c6, c7]);
    // Each lane k holds (cost[k], cost[k + 4]). For the winning lane,
    // max_lh[k] is best_cost and min_lh[k] is cost[y_dir ^ 4]. Packing
    // (d << 24) | var (< 2^20) into u32 and taking reduce_min picks the
    // first direction of maximal cost alongside its variance.
    let max_lh = lo.simd_max(hi);
    let min_lh = lo.simd_min(hi);
    let best = i32x4::splat(max_lh.reduce_max());
    let var = ((max_lh - min_lh).cast::<u32>() >> 10)
        | u32x4::from_array([0 << 24, 1 << 24, 2 << 24, 3 << 24]);
    let m_lo = lo.simd_eq(best).select(var, u32x4::splat(u32::MAX));
    let m_hi = hi.simd_eq(best).select(var | u32x4::splat(4 << 24), u32x4::splat(u32::MAX));
    let r = m_lo.simd_min(m_hi).reduce_min();
    ((r >> 24) as usize, (r & 0xff_ffff) as i32)
}


    };
}
mod bd8 {
    use super::*;
    const BD: u32 = 8;
    cdef_impl!(8);
}
mod bd10 {
    use super::*;
    const BD: u32 = 10;
    cdef_impl!(10);
}
mod bd12 {
    use super::*;
    const BD: u32 = 12;
    cdef_impl!(12);
}

/// 8-bit filter on 16 `u8` lanes (two rows of an 8 wide block or four
/// rows of a 4 wide one) for blocks whose border is entirely available,
/// like dav1d's "edged" 8bpc NEON kernels.
///
/// Exactness, for 8-bit samples, `damping <= 7` and strengths `< 16`:
///
/// * `|p - x|` is computed as `max - min` in `u8`, shifted by the
///   damping adjustment (`<= damping <= 7`, so no shift wraps), and
///   `Max(0, strength - (|d| >> adj))` is a saturating `u8` subtraction;
///   the constrained magnitude is `<= strength < 16`, so it and the sums
///   of up to four of them fit in `i8`.
/// * The weighted sum, the rounding and the clamp are done in `i16`:
///   `|sum| <= 15 * 2 * (4 + 2) + 15 * 4 * (2 + 1) < 32768`.
/// * Without sentinels the spec's `min` / `max` are simply the minimum
///   / maximum over the center and all taps.
mod u8_path {
    use super::*;
    use crate::decoder::av1::dsp::cdef::CDEF_DIRECTIONS;

    /// Length of the local `u8` copy of a block and its border:
    /// `(H + 4)` rows of `stride::<W>()` samples, plus slack for the
    /// windows of the last rows.
    const BLK: usize = 12 * 16 + 16;

    /// Length of a tap window (see `simd::cdef::Window`): enough for all
    /// loads of a block, which end at `16 * 7 + 8` (W = 8) and `8 * 4 +
    /// 32` (W = 4).
    const WIN: usize = 120;

    /// Row stride of the local copy (dav1d's tmp stride).
    const fn stride(w: usize) -> usize {
        if w == 8 {
            16
        } else {
            8
        }
    }

    /// Window of the local copy at tap offset `off` from the block's
    /// top-left sample. The start is at most `4 * stride + 4` and `4 *
    /// 16 + 4 + WIN <= BLK` (the `min` only lets the compiler see that).
    #[inline(always)]
    fn window<const W: usize>(blk: &[u8; BLK], off: isize) -> &[u8; WIN] {
        let start = (((2 * stride(W) + 2) as isize + off) as usize).min(BLK - WIN);
        blk[start..start + WIN].try_into().unwrap()
    }

    /// Lanes of rows `i..i + 16 / W` of a window.
    #[inline(always)]
    fn load<const W: usize>(win: &[u8; WIN], i: usize) -> u8x16 {
        let s = stride(W);
        let pos = i * s;
        if W == 8 {
            let a = u8x8::from_slice(&win[pos..pos + 8]);
            let b = u8x8::from_slice(&win[pos + s..pos + s + 8]);
            simd_swizzle!(a, b, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        } else {
            let v = u8x32::from_slice(&win[pos..pos + 32]);
            simd_swizzle!(
                v,
                [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]
            )
        }
    }

    /// Spec `constrain(p - x, strength, damping)` with the shift `sh =
    /// adj` precomputed (see the module documentation).
    #[inline(always)]
    fn constrain(x: u8x16, p: u8x16, strength: u8x16, sh: u8x16) -> i8x16 {
        let mn = x.simd_min(p);
        let ad = x.simd_max(p) - mn;
        let m = strength.saturating_sub(ad >> sh).simd_min(ad).cast::<i8>();
        let neg = if cfg!(target_arch = "x86_64") { mn.simd_eq(p) } else { x.simd_gt(p) };
        neg.select(-m, m)
    }

    /// 32-lane `constrain` on `[x, x]` and `[a, b]` (for AVX2).
    #[inline(always)]
    fn constrain32(xx: u8x32, ab: u8x32, strength: u8x32, sh: u8x32) -> i8x32 {
        let mn = xx.simd_min(ab);
        let ad = xx.simd_max(ab) - mn;
        let m = strength.saturating_sub(ad >> sh).simd_min(ad).cast::<i8>();
        let neg = if cfg!(target_arch = "x86_64") { mn.simd_eq(ab) } else { xx.simd_gt(ab) };
        neg.select(-m, m)
    }

    /// Loads `N` 8-bit samples of a pixel row as bytes.
    #[inline(always)]
    fn load_px<T: Pixel, const N: usize>(src: &[T]) -> [u8; N] {
        let s: &[T; N] = src[..N].try_into().unwrap();
        let mut v = [0u8; N];
        for (d, &p) in v.iter_mut().zip(s) {
            *d = p.to_i32() as u8;
        }
        v
    }

    /// Filters a `W x H` block of 8-bit pixels whose border is entirely
    /// inside the frame (`src` as for `CdefKernels::filter`, with
    /// [`CDEF_SRC_SLACK`] readable samples after the block). Requires
    /// `BD == 8`, `damping <= 7`, strengths in `0..16`, not both 0.
    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    pub(super) fn filter<T: Pixel, const W: usize, const H: usize>(
        dst: &mut [T],
        dst_stride: usize,
        src: &[T],
        src_stride: usize,
        pri_str: i32,
        sec_str: i32,
        damping: i32,
        dir: usize,
        tap_set: usize,
    ) {
        assert!(src_stride <= 0xffff_ffff);
        let s = stride(W);
        let src = &src[..(H + 2 * CDEF_BORDER - 1) * src_stride + s];
        let mut blk = [0u8; BLK];
        for r in 0..H + 2 * CDEF_BORDER {
            // Every row is written with a single store, which contains
            // each of the later loads from it (8 bytes at offsets <= 4
            // for W = 8, 4 bytes at offsets <= 4 for W = 4), so that
            // they can be store-forwarded.
            let row = &src[r * src_stride..];
            if W == 8 {
                blk[r * s..r * s + 16].copy_from_slice(&load_px::<T, 16>(row));
            } else {
                blk[r * s..r * s + 8].copy_from_slice(&load_px::<T, 8>(row));
            }
        }
        match (pri_str != 0, sec_str != 0) {
            (true, true) => run::<T, W, H, true, true>(
                dst, dst_stride, &blk, pri_str, sec_str, damping, dir, tap_set,
            ),
            (true, false) => run::<T, W, H, true, false>(
                dst, dst_stride, &blk, pri_str, sec_str, damping, dir, tap_set,
            ),
            _ => run::<T, W, H, false, true>(
                dst, dst_stride, &blk, pri_str, sec_str, damping, dir, tap_set,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn run<T: Pixel, const W: usize, const H: usize, const PRI: bool, const SEC: bool>(
        dst: &mut [T],
        dst_stride: usize,
        blk: &[u8; BLK],
        pri_str: i32,
        sec_str: i32,
        damping: i32,
        dir: usize,
        tap_set: usize,
    ) {
        assert!(dst_stride <= 0xffff_ffff);
        let dst = &mut dst[..(H - 1) * dst_stride + W];
        let s = stride(W);
        let w = |o: isize| window::<W>(blk, o);
        let off = |d: usize, k: usize| tap_offset(d, k, s);
        let center = w(0);
        let k = K8::new::<PRI, SEC>(pri_str, sec_str, damping, tap_set);
        for i in [0, 16 / W] {
            let x = load::<W>(center, i);
            let tap = |d: usize, t: usize, sign: isize| load::<W>(w(sign * off(d, t)), i);
            let v = compute::<PRI, SEC>(x, dir, &k, tap);
            store::<T, W>(dst, dst_stride, i, v);
        }
    }

    /// Stores the 16 lanes of `v` to rows `i..i + 16 / W` of `dst`.
    #[inline(always)]
    fn store<T: Pixel, const W: usize>(dst: &mut [T], dst_stride: usize, i: usize, v: u8x16) {
        if W == 4 {
            let rows = [
                simd_swizzle!(v, [0, 1, 2, 3, 0, 1, 2, 3]),
                simd_swizzle!(v, [4, 5, 6, 7, 4, 5, 6, 7]),
                simd_swizzle!(v, [8, 9, 10, 11, 8, 9, 10, 11]),
                simd_swizzle!(v, [12, 13, 14, 15, 12, 13, 14, 15]),
            ];
            for (r, rv) in rows.into_iter().enumerate() {
                let a = rv.cast::<u16>().to_array();
                let start = (i + r) * dst_stride;
                let row: &mut [T; 4] = (&mut dst[start..start + 4]).try_into().unwrap();
                for j in 0..4 {
                    row[j] = T::from_i32(a[j] as i32);
                }
            }
            return;
        }
        let v = v.to_array();
        for r in 0..16 / W {
            let start = (i + r) * dst_stride;
            let row: &mut [T; W] = (&mut dst[start..start + W]).try_into().unwrap();
            for j in 0..W {
                row[j] = T::from_i32(v[r * W + j] as i32);
            }
        }
    }

    /// Per-block parameters, as lane vectors.
    struct K8 {
        pri_th: u8x16,
        sec_th: u8x16,
        pri_sh: u8x16,
        sec_sh: u8x16,
        /// All-ones (`-1`) when `tap_set == 0` (`CDEF_PRI_TAPS = [4, 2]`),
        /// `0` when `tap_set == 1` (`[3, 3]`).
        even_mask: i8x16,
    }

    impl K8 {
        #[inline(always)]
        fn new<const PRI: bool, const SEC: bool>(
            pri_str: i32,
            sec_str: i32,
            damping: i32,
            tap_set: usize,
        ) -> Self {
            let z = u8x16::splat(0);
            let zi = i8x16::splat(0);
            let (pri_th, pri_sh, even_mask) = if PRI {
                (
                    u8x16::splat(pri_str as u8),
                    u8x16::splat(damping_adj(pri_str, damping) as u8),
                    i8x16::splat(((tap_set & 1) as i8) - 1),
                )
            } else {
                (z, z, zi)
            };
            let (sec_th, sec_sh) = if SEC {
                (
                    u8x16::splat(sec_str as u8),
                    u8x16::splat(damping_adj(sec_str, damping) as u8),
                )
            } else {
                (z, z)
            };
            K8 {
                pri_th,
                sec_th,
                pri_sh,
                sec_sh,
                even_mask,
            }
        }
    }

    /// Filters 16 lanes: `x` are the samples and `tap(d, k, sign)` loads
    /// the tap vector at `sign * CDEF_DIRECTIONS[d][k]`.
    ///
    /// For `pri_str < 16` and `sec_str < 16`, each constrained difference
    /// is in `-15..=15`. Using `CDEF_PRI_TAPS = [[4, 2], [3, 3]]` and
    /// `CDEF_SEC_TAPS = [2, 1]`, the weighted sum is decomposed as
    /// `2 * e + f` with `e = (p0 + p1) + c0` (`|e| <= 120`) and
    /// `f = (p0 + p1 + ((p0 - p1) & even_mask)) + c1` (`|f| <= 120`),
    /// both fitting in `i8x16` before a single widening to `i16x16`.
    #[inline(always)]
    fn compute<const PRI: bool, const SEC: bool>(
        x: u8x16,
        dir: usize,
        k: &K8,
        tap: impl Fn(usize, usize, isize) -> u8x16,
    ) -> u8x16 {
        let clamp = PRI && SEC;
        let mut min = x;
        let mut max = x;
        let (e, f) = if cfg!(target_feature = "avx2") {
            let dup = |v: u8x16| -> u8x32 {
                simd_swizzle!(
                    v,
                    v,
                    [
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6,
                        7, 8, 9, 10, 11, 12, 13, 14, 15
                    ]
                )
            };
            let fold_i8 = |v: i8x32| -> i8x16 {
                let lo: i8x16 =
                    simd_swizzle!(v, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
                let hi: i8x16 = simd_swizzle!(
                    v,
                    [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
                );
                lo + hi
            };
            let xx = dup(x);
            let mut min32 = xx;
            let mut max32 = xx;
            let mut pair32 = |d: usize, t: usize, th: u8x32, sh: u8x32| -> i8x32 {
                let a = tap(d, t, 1);
                let b = tap(d, t, -1);
                let ab: u8x32 = simd_swizzle!(
                    a,
                    b,
                    [
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
                    ]
                );
                if clamp {
                    min32 = min32.simd_min(ab);
                    max32 = max32.simd_max(ab);
                }
                constrain32(xx, ab, th, sh)
            };
            let (mut e32, mut f32) = (i8x32::splat(0), i8x32::splat(0));
            if PRI {
                let (th, sh) = (dup(k.pri_th), dup(k.pri_sh));
                let mask = dup(k.even_mask.cast::<u8>()).cast::<i8>();
                let p0 = pair32(dir, 0, th, sh);
                let p1 = pair32(dir, 1, th, sh);
                let p01 = p0 + p1;
                e32 = p01;
                f32 = p01 + ((p0 - p1) & mask);
            }
            if SEC {
                let (th, sh) = (dup(k.sec_th), dup(k.sec_sh));
                let (s0, s1) = ((dir + 6) & 7, (dir + 2) & 7);
                let c0 = pair32(s0, 0, th, sh) + pair32(s1, 0, th, sh);
                let c1 = pair32(s0, 1, th, sh) + pair32(s1, 1, th, sh);
                e32 = if PRI { e32 + c0 } else { c0 };
                f32 = if PRI { f32 + c1 } else { c1 };
            }
            if clamp {
                let min_lo: u8x16 = simd_swizzle!(
                    min32,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                );
                let min_hi: u8x16 = simd_swizzle!(
                    min32,
                    [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
                );
                let max_lo: u8x16 = simd_swizzle!(
                    max32,
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                );
                let max_hi: u8x16 = simd_swizzle!(
                    max32,
                    [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
                );
                min = min_lo.simd_min(min_hi);
                max = max_lo.simd_max(max_hi);
            }
            (fold_i8(e32), fold_i8(f32))
        } else {
            let mut pair = |d: usize, t: usize, th: u8x16, sh: u8x16| -> i8x16 {
                let a = tap(d, t, 1);
                let b = tap(d, t, -1);
                if clamp {
                    min = min.simd_min(a).simd_min(b);
                    max = max.simd_max(a).simd_max(b);
                }
                constrain(x, a, th, sh) + constrain(x, b, th, sh)
            };
            let (mut e, mut f) = (i8x16::splat(0), i8x16::splat(0));
            if PRI {
                let p0 = pair(dir, 0, k.pri_th, k.pri_sh);
                let p1 = pair(dir, 1, k.pri_th, k.pri_sh);
                let p01 = p0 + p1;
                e = p01;
                f = p01 + ((p0 - p1) & k.even_mask);
            }
            if SEC {
                let (s0, s1) = ((dir + 6) & 7, (dir + 2) & 7);
                let c0 = pair(s0, 0, k.sec_th, k.sec_sh) + pair(s1, 0, k.sec_th, k.sec_sh);
                let c1 = pair(s0, 1, k.sec_th, k.sec_sh) + pair(s1, 1, k.sec_th, k.sec_sh);
                e = if PRI { e + c0 } else { c0 };
                f = if PRI { f + c1 } else { c1 };
            }
            (e, f)
        };
        let sum = (e.cast::<i16>() << 1) + f.cast::<i16>();
        let off = (i16x16::splat(8) + sum + (sum >> 15)) >> 4;
        if clamp {
            let v = (x.cast::<i16>() + off)
                .simd_max(i16x16::splat(0))
                .simd_min(i16x16::splat(255))
                .cast::<u8>();
            v.simd_max(min).simd_min(max)
        } else {
            // Within the range of the taps, i.e. 0..=255.
            x + off.cast::<u8>()
        }
    }

    /// AVX2 forms of the 8x8 and 4x4 kernels (256-bit vectors, whose
    /// byte shuffles, packs and unpacks work within 128-bit halves).
    ///
    /// Compared to [`compute`], `constrain` needs no sign select: with
    /// `a = p -sat x` and `b = x -sat p` one of them is 0 and the other
    /// is `|p - x|`, so `sign(p - x) * Min(|p - x|, t)` is `Min(t, a) -
    /// Min(t, b)`. On 32 lanes the weighted sum `2 * e + f` (see
    /// [`compute`]) is formed in 16-bit lanes from the in-lane byte
    /// interleaves of `e` and `f` with zero (`e << 8`, `f << 8`), and the
    /// 16-bit offsets are packed back to bytes in the original lane
    /// order. Blocks of direction 0 are first tested for being flat
    /// (see [`flat`]), which is common in synthetic content.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    pub(super) mod avx2 {
        use super::*;

        /// Per-block parameters on `N` lanes (see [`K8`]).
        struct Kn<const N: usize> {
            pri_th: Simd<u8, N>,
            sec_th: Simd<u8, N>,
            /// The damping shifts (as 16-bit lane shifts) and the masks
            /// `0xff >> shift` that make them byte shifts.
            pri_sh: Sh<N>,
            sec_sh: Sh<N>,
            even_mask: Simd<i8, N>,
        }

        /// A byte shift right by `sh` as a 16-bit shift and a mask.
        #[derive(Clone, Copy)]
        struct Sh<const N: usize> {
            sh: u16,
            mask: Simd<u8, N>,
        }

        impl<const N: usize> Sh<N> {
            #[inline(always)]
            fn new(sh: u32) -> Self {
                Sh {
                    sh: sh as u16,
                    mask: Simd::splat((0xffu32 >> sh) as u8),
                }
            }
        }

        impl<const N: usize> Kn<N> {
            #[inline(always)]
            fn new<const PRI: bool, const SEC: bool>(
                pri_str: i32,
                sec_str: i32,
                damping: i32,
                tap_set: usize,
            ) -> Self {
                let z = Simd::splat(0);
                let (pri_th, pri_sh, even_mask) = if PRI {
                    (
                        Simd::splat(pri_str as u8),
                        Sh::new(damping_adj(pri_str, damping)),
                        Simd::splat(((tap_set & 1) as i8) - 1),
                    )
                } else {
                    (z, Sh::new(0), Simd::splat(0))
                };
                let (sec_th, sec_sh) = if SEC {
                    (
                        Simd::splat(sec_str as u8),
                        Sh::new(damping_adj(sec_str, damping)),
                    )
                } else {
                    (z, Sh::new(0))
                };
                Kn {
                    pri_th,
                    sec_th,
                    pri_sh,
                    sec_sh,
                    even_mask,
                }
            }
        }

        /// Spec `constrain(p - x, strength, damping)` (see the module
        /// documentation and [`super::constrain`]).
        #[inline(always)]
        fn constrain<const N: usize>(
            x: Simd<u8, N>,
            p: Simd<u8, N>,
            th: Simd<u8, N>,
            sh: Sh<N>,
        ) -> Simd<i8, N> {
            let a = p.saturating_sub(x);
            let b = x.saturating_sub(p);
            let t = th.saturating_sub(shr(a | b, sh));
            (t.simd_min(a) - t.simd_min(b)).cast::<i8>()
        }

        /// `v >> sh` (bytes) as a 16-bit shift and a mask.
        #[inline(always)]
        fn shr<const N: usize>(v: Simd<u8, N>, sh: Sh<N>) -> Simd<u8, N> {
            if N == 32 {
                let w =
                    u16x16::from_ne_bytes(Simd::<u8, 32>::from_slice(v.as_array()).to_ne_bytes());
                Simd::from_slice(&(w >> sh.sh).to_ne_bytes().to_array()) & sh.mask
            } else {
                let w =
                    u16x8::from_ne_bytes(Simd::<u8, 16>::from_slice(v.as_array()).to_ne_bytes());
                Simd::from_slice(&(w >> sh.sh).to_ne_bytes().to_array()) & sh.mask
            }
        }

        /// In-lane byte interleave with zero: `[0, v[16 * b + i + H]]`
        /// pairs (`i` in `0..8`) for every 16-byte block `b` (`H = 8` for
        /// the high halves), i.e. `punpck{l,h}bw(0, v)`.
        struct Unpack<const N: usize, const H: usize>;
        impl<const N: usize, const H: usize> std::simd::Swizzle<N> for Unpack<N, H> {
            const INDEX: [usize; N] = {
                let mut r = [0; N];
                let mut j = 0;
                while j < N {
                    // Odd bytes (the high byte of each word) take v.
                    if j % 2 == 1 {
                        r[j] = N + (j / 16) * 16 + H + (j % 16) / 2;
                    }
                    j += 1;
                }
                r
            };
        }

        /// Inverse of [`Unpack`] on the low bytes of the words (as
        /// `packsswb(lo, hi)`): bytes `16 * b..16 * b + 8` from the low
        /// halves (first vector), the others from the high halves.
        struct Pack<const N: usize>;
        impl<const N: usize> std::simd::Swizzle<N> for Pack<N> {
            const INDEX: [usize; N] = {
                let mut r = [0; N];
                let mut j = 0;
                while j < N {
                    let (b, i) = (j / 16, j % 16);
                    r[j] = if i < 8 { 16 * b + 2 * i } else { N + 16 * b + 2 * (i - 8) };
                    j += 1;
                }
                r
            };
        }

        /// The rounded offset `(8 + sum - (sum < 0)) >> 4` of `sum = 2 *
        /// e + f` (`|sum| <= 360`, see [`compute`]) for 32 lanes.
        #[inline(always)]
        fn offset32(e: i8x32, f: i8x32) -> i8x32 {
            use std::simd::Swizzle;
            let z = i8x32::splat(0);
            let half = |lo: bool| -> i16x16 {
                let (we, wf) = if lo {
                    (
                        Unpack::<32, 0>::concat_swizzle(z, e),
                        Unpack::<32, 0>::concat_swizzle(z, f),
                    )
                } else {
                    (
                        Unpack::<32, 8>::concat_swizzle(z, e),
                        Unpack::<32, 8>::concat_swizzle(z, f),
                    )
                };
                let sum = (i16x16::from_ne_bytes(we.to_ne_bytes()) >> 7)
                    + (i16x16::from_ne_bytes(wf.to_ne_bytes()) >> 8);
                (sum + i16x16::splat(8) + (sum >> 15)) >> 4
            };
            let (lo, hi) = (half(true), half(false));
            i8x32::from_ne_bytes(Pack::<32>::concat_swizzle(
                lo.to_ne_bytes(),
                hi.to_ne_bytes(),
            ))
        }

        /// [`offset32`] for 16 lanes, in one 16-lane `i16` vector.
        #[inline(always)]
        fn offset16(e: i8x16, f: i8x16) -> i8x16 {
            let sum = (e.cast::<i16>() << 1) + f.cast::<i16>();
            ((sum + i16x16::splat(8) + (sum >> 15)) >> 4).cast::<i8>()
        }

        /// `x + off`, clamped to `min..=max` when `CLAMP` (otherwise the
        /// sum is within the range of the taps).
        #[inline(always)]
        fn apply<const N: usize, const CLAMP: bool>(
            x: Simd<u8, N>,
            off: Simd<i8, N>,
            min: Simd<u8, N>,
            max: Simd<u8, N>,
        ) -> Simd<u8, N> {
            if CLAMP {
                let bias = Simd::splat(0x80u8);
                // min <= x <= max, so clamping the saturated sum is exact.
                let v = ((x ^ bias).cast::<i8>().saturating_add(off)).cast::<u8>() ^ bias;
                v.simd_max(min).simd_min(max)
            } else {
                x + off.cast::<u8>()
            }
        }

        /// Filters 32 lanes, one tap per vector (see [`super::compute`]).
        #[inline(always)]
        fn compute32<const PRI: bool, const SEC: bool>(
            x: u8x32,
            dir: usize,
            k: &Kn<32>,
            tap: impl Fn(usize, usize, isize) -> u8x32,
        ) -> u8x32 {
            let clamp = PRI && SEC;
            let mut min = x;
            let mut max = x;
            let mut pair = |d: usize, t: usize, th: u8x32, sh: Sh<32>| -> i8x32 {
                let a = tap(d, t, 1);
                let b = tap(d, t, -1);
                if clamp {
                    min = min.simd_min(a).simd_min(b);
                    max = max.simd_max(a).simd_max(b);
                }
                constrain(x, a, th, sh) + constrain(x, b, th, sh)
            };
            let (mut e, mut f) = (i8x32::splat(0), i8x32::splat(0));
            if PRI {
                let p0 = pair(dir, 0, k.pri_th, k.pri_sh);
                let p1 = pair(dir, 1, k.pri_th, k.pri_sh);
                let p01 = p0 + p1;
                e = p01;
                f = p01 + ((p0 - p1) & k.even_mask);
            }
            if SEC {
                let (s0, s1) = ((dir + 6) & 7, (dir + 2) & 7);
                let c0 = pair(s0, 0, k.sec_th, k.sec_sh) + pair(s1, 0, k.sec_th, k.sec_sh);
                let c1 = pair(s0, 1, k.sec_th, k.sec_sh) + pair(s1, 1, k.sec_th, k.sec_sh);
                e = if PRI { e + c0 } else { c0 };
                f = if PRI { f + c1 } else { c1 };
            }
            let off = offset32(e, f);
            if clamp {
                apply::<32, true>(x, off, min, max)
            } else {
                apply::<32, false>(x, off, min, max)
            }
        }

        /// [`super::filter_8x8`] on 32 lanes (4 rows): the 12 rows of the
        /// block and its border are held as `y[j] = [row j | row j + 2]`
        /// with each row as `[columns 0..8, columns 4..12]`, so that rows
        /// `r..r + 4` of a tap are two 64-bit shifts (by the column) and
        /// one in-lane dword shuffle of `y[r + 2 + dy]` and `y[r + 3 +
        /// dy]`.
        #[allow(clippy::too_many_arguments)]
        #[inline(never)]
        pub(in super::super) fn filter_8x8<
            T: Pixel,
            const DIR: usize,
            const PRI: bool,
            const SEC: bool,
        >(
            dst: &mut [T],
            dst_stride: usize,
            src: &[T],
            src_stride: usize,
            pri_str: i32,
            sec_str: i32,
            damping: i32,
            tap_set: usize,
        ) {
            assert!(src_stride <= 0xffff_ffff && dst_stride <= 0xffff_ffff);
            // As u32 (zero extended), so that the compiler sees that no
            // offset computation wraps and all bounds checks are implied
            // by the two slicings.
            let (src_stride, dst_stride) = (src_stride as u32 as usize, dst_stride as u32 as usize);
            let src = &src[..11 * src_stride + 16];
            let dst = &mut dst[..7 * dst_stride + 8];
            let k = Kn::<32>::new::<PRI, SEC>(pri_str, sec_str, damping, tap_set);
            let rows: [u8x16; 12] = std::array::from_fn(|r| {
                let pos = r * src_stride;
                u8x16::from_array(load_px::<T, 16>(&src[pos..pos + 16]))
            });
            if DIR == 0 && flat::<12>(&rows) {
                store_rows::<T, 8, 32>(dst, dst_stride, 0, u8x32::splat(rows[2][2]));
                store_rows::<T, 8, 32>(dst, dst_stride, 4, u8x32::splat(rows[2][2]));
                return;
            }
            let y: [u64x4; 10] = std::array::from_fn(|j| {
                u64x4::from_ne_bytes(simd_swizzle!(
                    rows[j],
                    rows[j + 2],
                    [
                        0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21,
                        22, 23, 20, 21, 22, 23, 24, 25, 26, 27
                    ]
                ))
            });
            // Rows r..r + 4 (unrolled, so that all indices are constants).
            macro_rules! group {
                ($r:literal) => {{
                    let at = |dy: isize, dx: isize| -> u8x32 {
                        let j = ($r + 2 + dy) as usize;
                        let sh = ((2 + dx) as u64) * 8;
                        let a = u32x8::from_ne_bytes((y[j] >> sh).to_ne_bytes());
                        let b = u32x8::from_ne_bytes((y[j + 1] >> sh).to_ne_bytes());
                        simd_swizzle!(a, b, [0, 2, 8, 10, 4, 6, 12, 14]).to_ne_bytes()
                    };
                    let tap = |d: usize, t: usize, sign: isize| {
                        let [dy, dx] = CDEF_DIRECTIONS[d][t];
                        at(sign * dy, sign * dx)
                    };
                    let v = compute32::<PRI, SEC>(at(0, 0), DIR, &k, tap);
                    store_rows::<T, 8, 32>(dst, dst_stride, $r, v);
                }};
            }
            group!(0);
            group!(4);
        }

        /// Filters 16 lanes with the taps in symmetric pairs `[+tap |
        /// -tap]` on 32 lanes (`tap2(d, k)`), folded at the end.
        #[inline(always)]
        fn compute_pairs<const PRI: bool, const SEC: bool>(
            x: u8x16,
            dir: usize,
            k: &Kn<32>,
            tap2: impl Fn(usize, usize) -> u8x32,
        ) -> u8x16 {
            let clamp = PRI && SEC;
            let xx: u8x32 = simd_swizzle!(
                x,
                x,
                [
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                    22, 23, 24, 25, 26, 27, 28, 29, 30, 31
                ]
            );
            let mut min = xx;
            let mut max = xx;
            let mut pair = |d: usize, t: usize, th: u8x32, sh: Sh<32>| -> i8x32 {
                let p = tap2(d, t);
                if clamp {
                    min = min.simd_min(p);
                    max = max.simd_max(p);
                }
                constrain(xx, p, th, sh)
            };
            let (mut e, mut f) = (i8x32::splat(0), i8x32::splat(0));
            if PRI {
                let p0 = pair(dir, 0, k.pri_th, k.pri_sh);
                let p1 = pair(dir, 1, k.pri_th, k.pri_sh);
                let p01 = p0 + p1;
                e = p01;
                f = p01 + ((p0 - p1) & k.even_mask);
            }
            if SEC {
                let (s0, s1) = ((dir + 6) & 7, (dir + 2) & 7);
                let c0 = pair(s0, 0, k.sec_th, k.sec_sh) + pair(s1, 0, k.sec_th, k.sec_sh);
                let c1 = pair(s0, 1, k.sec_th, k.sec_sh) + pair(s1, 1, k.sec_th, k.sec_sh);
                e = if PRI { e + c0 } else { c0 };
                f = if PRI { f + c1 } else { c1 };
            }
            fn halves<E: std::simd::SimdElement>(v: Simd<E, 32>) -> (Simd<E, 16>, Simd<E, 16>) {
                (
                    simd_swizzle!(v, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
                    simd_swizzle!(
                        v,
                        [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
                    ),
                )
            }
            let ((e0, e1), (f0, f1)) = (halves(e), halves(f));
            let off = offset16(e0 + e1, f0 + f1);
            if clamp {
                let ((n0, n1), (m0, m1)) = (halves(min), halves(max));
                apply::<16, true>(x, off, n0.simd_min(n1), m0.simd_max(m1))
            } else {
                apply::<16, false>(x, off, x, x)
            }
        }

        /// [`super::filter_4x4`] with the taps in symmetric pairs on 32
        /// lanes: rows `r, r + 1` of the block and its border are held as
        /// `pairs[r]` (`u64x2`), and `[+tap | -tap]` is two 128-bit inserts
        /// of row pairs, two variable 64-bit shifts (by the columns) and
        /// one in-lane dword shuffle.
        #[allow(clippy::too_many_arguments)]
        #[inline(never)]
        pub(in super::super) fn filter_4x4<
            T: Pixel,
            const DIR: usize,
            const PRI: bool,
            const SEC: bool,
        >(
            dst: &mut [T],
            dst_stride: usize,
            src: &[T],
            src_stride: usize,
            pri_str: i32,
            sec_str: i32,
            damping: i32,
            tap_set: usize,
        ) {
            assert!(src_stride <= 0xffff_ffff && dst_stride <= 0xffff_ffff);
            // See filter_8x8.
            let (src_stride, dst_stride) = (src_stride as u32 as usize, dst_stride as u32 as usize);
            let src = &src[..7 * src_stride + 8];
            let dst = &mut dst[..3 * dst_stride + 4];
            let rows: [u8x8; 8] = std::array::from_fn(|r| {
                let pos = r * src_stride;
                u8x8::from_array(load_px::<T, 8>(&src[pos..pos + 8]))
            });
            if DIR == 0 {
                let v = u8x8::splat(rows[0][0]);
                let mut acc = u8x8::splat(0);
                for r in &rows {
                    acc |= *r ^ v;
                }
                if acc.reduce_or() == 0 {
                    store_rows::<T, 4, 16>(dst, dst_stride, 0, u8x16::splat(rows[0][0]));
                    return;
                }
            }
            let pairs: [u64x2; 7] = std::array::from_fn(|r| {
                u64x2::from_ne_bytes(simd_swizzle!(
                    rows[r],
                    rows[r + 1],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                ))
            });
            let k = Kn::<32>::new::<PRI, SEC>(pri_str, sec_str, damping, tap_set);
            let x = {
                let a = u32x4::from_ne_bytes((pairs[2] >> 16).to_ne_bytes());
                let b = u32x4::from_ne_bytes((pairs[4] >> 16).to_ne_bytes());
                simd_swizzle!(a, b, [0, 2, 4, 6]).to_ne_bytes()
            };
            let tap2 = |d: usize, t: usize| -> u8x32 {
                let [dy, dx] = CDEF_DIRECTIONS[d][t];
                let (sp, sm) = (((2 + dx) * 8) as u64, ((2 - dx) * 8) as u64);
                let sh = u64x4::from_array([sp, sp, sm, sm]);
                let (r0, r1) = ((2 + dy) as usize, (2 - dy) as usize);
                let a: u64x4 = simd_swizzle!(pairs[r0], pairs[r1], [0, 1, 2, 3]) >> sh;
                let b: u64x4 = simd_swizzle!(pairs[r0 + 2], pairs[r1 + 2], [0, 1, 2, 3]) >> sh;
                let (a, b) = (
                    u32x8::from_ne_bytes(a.to_ne_bytes()),
                    u32x8::from_ne_bytes(b.to_ne_bytes()),
                );
                simd_swizzle!(a, b, [0, 2, 8, 10, 4, 6, 12, 14]).to_ne_bytes()
            };
            let v = compute_pairs::<PRI, SEC>(x, DIR, &k, tap2);
            store_rows::<T, 4, 16>(dst, dst_stride, 0, v);
        }

        /// Whether the first 12 samples of all `R` rows (a block and its
        /// border) are equal. Then every tap equals its center sample, so
        /// that all constrained differences are 0 and the block is left
        /// unchanged. Flat blocks have all direction costs equal, so they
        /// get direction 0, and only those kernels test this.
        #[inline(always)]
        fn flat<const R: usize>(rows: &[u8x16; R]) -> bool {
            let v = u8x16::splat(rows[0][0]);
            let mut acc = u8x16::splat(0);
            for r in rows {
                acc |= *r ^ v;
            }
            let m = u8x16::from_array([
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0,
            ]);
            (acc & m).reduce_or() == 0
        }

        /// Stores the `N` lanes of `v` to rows `i..i + N / W` of `dst`.
        #[inline(always)]
        fn store_rows<T: Pixel, const W: usize, const N: usize>(
            dst: &mut [T],
            dst_stride: usize,
            i: usize,
            v: Simd<u8, N>,
        ) {
            let v = v.to_array();
            for r in 0..N / W {
                let start = (i + r) * dst_stride;
                debug_assert!(start + W <= dst.len());
                let row: &mut [T; W] = (&mut dst[start..start + W]).try_into().unwrap();
                for j in 0..W {
                    row[j] = T::from_i32(v[r * W + j] as i32);
                }
            }
        }
    }

    /// A 4x4 block without a tmp copy: its 8 rows (with the border) are
    /// loaded once and every tap vector is one table lookup on two row
    /// pairs. `DIR` is the (constant) direction. Same contract as
    /// [`filter`].
    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    pub(super) fn filter_4x4<T: Pixel, const DIR: usize, const PRI: bool, const SEC: bool>(
        dst: &mut [T],
        dst_stride: usize,
        src: &[T],
        src_stride: usize,
        pri_str: i32,
        sec_str: i32,
        damping: i32,
        tap_set: usize,
    ) {
        assert!(src_stride <= 0xffff_ffff && dst_stride <= 0xffff_ffff);
        let src = &src[..7 * src_stride + 8];
        let dst = &mut dst[..3 * dst_stride + 4];
        let rows: [u8x8; 8] =
            std::array::from_fn(|r| u8x8::from_array(load_px::<T, 8>(&src[r * src_stride..])));
        // A flat block (and border) is left unchanged (see `flat_8x8`).
        if DIR == 0 {
            let v = u8x8::splat(rows[0][0]);
            if rows
                .iter()
                .fold(u8x8::splat(0), |acc, &r| acc | (r ^ v))
                .reduce_or()
                == 0
            {
                store::<T, 4>(dst, dst_stride, 0, u8x16::splat(rows[0][0]));
                return;
            }
        }
        // pairs[r] = rows r and r + 1 as two u64 lanes.
        let pairs: [u64x2; 7] = std::array::from_fn(|r| {
            u64x2::from_ne_bytes(simd_swizzle!(
                rows[r],
                rows[r + 1],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            ))
        });
        // Rows 2 + dy..6 + dy, columns 2 + dx..6 + dx. All arguments are
        // constants after inlining.
        let at = |dy: isize, dx: isize| {
            let sh = ((2 + dx) as u64) * 8;
            let a = u32x4::from_ne_bytes((pairs[(2 + dy) as usize] >> sh).to_ne_bytes());
            let b = u32x4::from_ne_bytes((pairs[(4 + dy) as usize] >> sh).to_ne_bytes());
            simd_swizzle!(a, b, [0, 2, 4, 6]).to_ne_bytes()
        };
        let tap = |d: usize, k: usize, sign: isize| {
            let [dy, dx] = CDEF_DIRECTIONS[d][k];
            at(sign * dy, sign * dx)
        };
        let k = K8::new::<PRI, SEC>(pri_str, sec_str, damping, tap_set);
        let v = compute::<PRI, SEC>(at(0, 0), DIR, &k, tap);
        store::<T, 4>(dst, dst_stride, 0, v);
    }

    /// Calls `$f::<T, DIR, PRI, SEC>` for the runtime direction (`dir <
    /// 8`; only `dir & 3` matters without primary taps) and strengths.
    macro_rules! dispatch_dir {
        ($f:ident, $pri_str:expr, $sec_str:expr, $dir:expr, ($($arg:expr),*)) => {{
            macro_rules! call {
                ($pri:literal, $sec:literal, $d:literal) => {
                    $f::<T, $d, $pri, $sec>($($arg),*)
                };
            }
            match ($pri_str != 0, $sec_str != 0, $dir & 7) {
                (false, _, 0 | 4) => call!(false, true, 0),
                (false, _, 1 | 5) => call!(false, true, 1),
                (false, _, 2 | 6) => call!(false, true, 2),
                (false, _, _) => call!(false, true, 3),
                (true, false, 0) => call!(true, false, 0),
                (true, false, 1) => call!(true, false, 1),
                (true, false, 2) => call!(true, false, 2),
                (true, false, 3) => call!(true, false, 3),
                (true, false, 4) => call!(true, false, 4),
                (true, false, 5) => call!(true, false, 5),
                (true, false, 6) => call!(true, false, 6),
                (true, false, _) => call!(true, false, 7),
                (true, true, 0) => call!(true, true, 0),
                (true, true, 1) => call!(true, true, 1),
                (true, true, 2) => call!(true, true, 2),
                (true, true, 3) => call!(true, true, 3),
                (true, true, 4) => call!(true, true, 4),
                (true, true, 5) => call!(true, true, 5),
                (true, true, 6) => call!(true, true, 6),
                (true, true, _) => call!(true, true, 7),
            }
        }};
    }

    /// [`filter_4x4`] for a runtime direction (`dir < 8`); the strengths
    /// must not both be 0.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub(super) fn filter_4x4_dir<T: Pixel>(
        dst: &mut [T],
        dst_stride: usize,
        src: &[T],
        src_stride: usize,
        pri_str: i32,
        sec_str: i32,
        damping: i32,
        dir: usize,
        tap_set: usize,
    ) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        use avx2::filter_4x4;
        dispatch_dir!(
            filter_4x4,
            pri_str,
            sec_str,
            dir,
            (dst, dst_stride, src, src_stride, pri_str, sec_str, damping, tap_set)
        )
    }

    /// An 8x8 block without a tmp copy (see [`filter_4x4`]): the 12 rows
    /// of 12 samples are held as 16 byte vectors and the lanes of rows
    /// `i, i + 1` of a tap are one lookup on two of them.
    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    pub(super) fn filter_8x8<T: Pixel, const DIR: usize, const PRI: bool, const SEC: bool>(
        dst: &mut [T],
        dst_stride: usize,
        src: &[T],
        src_stride: usize,
        pri_str: i32,
        sec_str: i32,
        damping: i32,
        tap_set: usize,
    ) {
        assert!(src_stride <= 0xffff_ffff && dst_stride <= 0xffff_ffff);
        let src = &src[..11 * src_stride + 16];
        let dst = &mut dst[..7 * dst_stride + 8];
        if DIR == 0 && flat_8x8(src, src_stride) {
            let v = u8x16::splat(src[2 * src_stride + 2].to_i32() as u8);
            for i in [0, 2, 4, 6] {
                store::<T, 8>(dst, dst_stride, i, v);
            }
            return;
        }
        let k = K8::new::<PRI, SEC>(pri_str, sec_str, damping, tap_set);
        // Row r as [columns 0..8, columns 4..12].
        macro_rules! load_row {
            ($r:literal) => {{
                let v = u8x16::from_array(load_px::<T, 16>(&src[$r * src_stride..]));
                u64x2::from_ne_bytes(simd_swizzle!(
                    v,
                    [0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 9, 10, 11]
                ))
            }};
        }
        macro_rules! row_pair {
            ($i:literal, $r0:ident, $r1:ident, $r2:ident, $r3:ident, $r4:ident, $r5:ident) => {{
                let win = [$r0, $r1, $r2, $r3, $r4, $r5];
                let at = |dy: isize, dx: isize| {
                    let r = (2 + dy) as usize;
                    let sh = ((2 + dx) as u64) * 8;
                    let a = u32x4::from_ne_bytes((win[r] >> sh).to_ne_bytes());
                    let b = u32x4::from_ne_bytes((win[r + 1] >> sh).to_ne_bytes());
                    simd_swizzle!(a, b, [0, 2, 4, 6]).to_ne_bytes()
                };
                let tap = |d: usize, k: usize, sign: isize| {
                    let [dy, dx] = CDEF_DIRECTIONS[d][k];
                    at(sign * dy, sign * dx)
                };
                let v = compute::<PRI, SEC>(at(0, 0), DIR, &k, tap);
                store::<T, 8>(dst, dst_stride, $i, v);
            }};
        }
        let (r0, r1, r2, r3, r4, r5) = (
            load_row!(0),
            load_row!(1),
            load_row!(2),
            load_row!(3),
            load_row!(4),
            load_row!(5),
        );
        row_pair!(0, r0, r1, r2, r3, r4, r5);
        let (r6, r7) = (load_row!(6), load_row!(7));
        row_pair!(2, r2, r3, r4, r5, r6, r7);
        let (r8, r9) = (load_row!(8), load_row!(9));
        row_pair!(4, r4, r5, r6, r7, r8, r9);
        let (r10, r11) = (load_row!(10), load_row!(11));
        row_pair!(6, r6, r7, r8, r9, r10, r11);
    }

    /// Whether the 12 x 12 samples of an 8x8 block and its border (`src`
    /// as for [`filter_8x8`]) are all equal. Then every tap equals its
    /// center sample, all constrained differences are 0 and the block is
    /// left unchanged. Flat blocks have all direction costs equal, so
    /// they get direction 0, and only those kernels test this (it is
    /// common in synthetic content and alpha planes).
    #[inline(always)]
    fn flat_8x8<T: Pixel>(src: &[T], src_stride: usize) -> bool {
        let v = u8x16::splat(src[0].to_i32() as u8);
        let mut acc = u8x16::splat(0);
        for r in 0..12 {
            acc |= u8x16::from_array(load_px::<T, 16>(&src[r * src_stride..])) ^ v;
        }
        let m = u8x16::from_array([
            255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0,
        ]);
        (acc & m).reduce_or() == 0
    }

    /// [`filter_8x8`] for a runtime direction (`dir < 8`); the strengths
    /// must not both be 0.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub(super) fn filter_8x8_dir<T: Pixel>(
        dst: &mut [T],
        dst_stride: usize,
        src: &[T],
        src_stride: usize,
        pri_str: i32,
        sec_str: i32,
        damping: i32,
        dir: usize,
        tap_set: usize,
    ) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        use avx2::filter_8x8;
        dispatch_dir!(
            filter_8x8,
            pri_str,
            sec_str,
            dir,
            (dst, dst_stride, src, src_stride, pri_str, sec_str, damping, tap_set)
        )
    }
}
// ---------------------------------------------------------------------------
// Bit depth specialized entry points (selected by `Dsp::simd(bd)`).
// `BD` is 8, 10 or 12 and always equals the runtime `bd` argument; the
// pixel / intermediate ranges of `BD` may be relied upon.
// ---------------------------------------------------------------------------

pub fn cdef_direction_bd<T: Pixel, const BD: u32>(
    src: &[T],
    stride: usize,
    bd: u32,
) -> (usize, i32) {
    debug_assert_eq!(bd, BD);
    match BD {
        8 => bd8::cdef_direction(src, stride),
        10 => bd10::cdef_direction(src, stride),
        _ => bd12::cdef_direction(src, stride),
    }
}

/// The `std::simd` [`CdefKernels`] for bit depth `BD`.
pub struct SimdCdef<const BD: u32>;

impl<T: Pixel, const BD: u32> CdefKernels<T> for SimdCdef<BD> {
    #[inline(always)]
    fn direction(src: &[T], stride: usize, bd: u32) -> (usize, i32) {
        cdef_direction_bd::<T, BD>(src, stride, bd)
    }

    #[inline(always)]
    fn filter(
        dst: &mut [T],
        dst_stride: usize,
        src: &[T],
        src_stride: usize,
        w: usize,
        h: usize,
        pri: i32,
        sec: i32,
        damping: i32,
        dir: usize,
        edges: u8,
        bd: u32,
    ) {
        debug_assert_eq!(bd, BD);
        match BD {
            8 => bd8::filter_px(
                dst, dst_stride, src, src_stride, w, h, pri, sec, damping, dir, edges,
            ),
            10 => bd10::filter_px(
                dst, dst_stride, src, src_stride, w, h, pri, sec, damping, dir, edges,
            ),
            _ => bd12::filter_px(
                dst, dst_stride, src, src_stride, w, h, pri, sec, damping, dir, edges,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::av1::dsp::cdef as portable;
    use crate::decoder::av1::dsp::cdef::CDEF_VERY_LARGE;

    fn simd_cdef_filter<T: Pixel>(
        dst: &mut [T],
        dst_stride: usize,
        src: &[u16],
        src_stride: usize,
        w: usize,
        h: usize,
        pri_str: i32,
        sec_str: i32,
        damping: i32,
        dir: usize,
        bd: u32,
    ) {
        match bd {
            8 => super::bd8::cdef_filter(
                dst, dst_stride, src, src_stride, w, h, pri_str, sec_str, damping, dir,
            ),
            10 => super::bd10::cdef_filter(
                dst, dst_stride, src, src_stride, w, h, pri_str, sec_str, damping, dir,
            ),
            _ => super::bd12::cdef_filter(
                dst, dst_stride, src, src_stride, w, h, pri_str, sec_str, damping, dir,
            ),
        }
    }

    fn simd_cdef_direction<T: Pixel>(src: &[T], stride: usize, bd: u32) -> (usize, i32) {
        match bd {
            8 => super::bd8::cdef_direction(src, stride),
            10 => super::bd10::cdef_direction(src, stride),
            _ => super::bd12::cdef_direction(src, stride),
        }
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
        fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % (hi - lo + 1) as u64) as i32
        }
    }

    /// Random padded source for a `w x h` block: noise around a base,
    /// sometimes an edge, and unavailable rows / columns / holes in the
    /// border.
    fn gen_src(rng: &mut Rng, w: usize, h: usize, stride: usize, bd: u32) -> Vec<u16> {
        let maxv = (1 << bd) - 1;
        let (fw, fh) = (w + 4, h + 4);
        let base = rng.range(0, maxv);
        let noise = [0, 2, 10, 50, maxv][rng.range(0, 4) as usize] << (bd - 8).min(2);
        let edge = if rng.range(0, 1) == 0 { 0 } else { rng.range(-maxv, maxv) };
        let ex = rng.range(0, fw as i32) as usize;
        let (top, left) = (rng.range(0, 3) == 0, rng.range(0, 3) == 0);
        let (bottom, right) = (rng.range(0, 3) == 0, rng.range(0, 3) == 0);
        let holes = rng.range(0, 3) == 0;
        // Extra garbage after the last used column of every row.
        let mut src = vec![0u16; stride * fh];
        for v in src.iter_mut() {
            *v = rng.range(0, 65535) as u16;
        }
        for y in 0..fh {
            for x in 0..fw {
                let inside = (2..2 + h).contains(&y) && (2..2 + w).contains(&x);
                let unavail = !inside
                    && ((top && y < 2)
                        || (bottom && y >= 2 + h)
                        || (left && x < 2)
                        || (right && x >= 2 + w)
                        || (holes && rng.range(0, 3) == 0));
                let e = if x + y / 2 >= ex { edge } else { 0 };
                src[y * stride + x] = if unavail {
                    CDEF_VERY_LARGE
                } else {
                    (base + e + rng.range(-noise, noise)).clamp(0, maxv) as u16
                };
            }
        }
        src
    }

    fn filter_random<T: Pixel>(bd: u32, seed: u64) {
        let mut rng = Rng(seed);
        let cs = bd - 8;
        for (w, h) in [(8usize, 8usize), (4, 8), (8, 4), (4, 4)] {
            for iter in 0..20000 {
                let stride = w + 4 + rng.range(0, 5) as usize;
                let src = gen_src(&mut rng, w, h, stride, bd);
                let dir = rng.range(0, 7) as usize;
                let (pri, sec, damping) = match rng.range(0, 4) {
                    // Conformant: pri-only / sec-only / both / none.
                    0 => (
                        rng.range(0, 15) << cs,
                        0,
                        rng.range(3, 6) + cs as i32 - rng.range(0, 1),
                    ),
                    1 => (
                        0,
                        [1, 2, 4][rng.range(0, 2) as usize] << cs,
                        rng.range(3, 6) + cs as i32 - rng.range(0, 1),
                    ),
                    2 => (
                        rng.range(0, 15) << cs,
                        [0, 1, 2, 4][rng.range(0, 3) as usize] << cs,
                        rng.range(3, 6) + cs as i32 - rng.range(0, 1),
                    ),
                    // Luma strengths scaled by the variance (any value).
                    3 => (
                        rng.range(0, 15 << cs),
                        [0, 1, 2, 4][rng.range(0, 3) as usize] << cs,
                        rng.range(2, 9) + cs as i32,
                    ),
                    // Non-conformant (fast path limits, portable fallback).
                    _ => match rng.range(0, 2) {
                        0 => (rng.range(0, 1023), rng.range(0, 1023), rng.range(0, 13)),
                        1 => (rng.range(0, 1100), rng.range(0, 1100), rng.range(0, 15)),
                        _ => (rng.range(0, 5000), rng.range(0, 5000), rng.range(0, 20)),
                    },
                };
                let dst_stride = w + rng.range(0, 3) as usize;
                let init: Vec<T> = (0..dst_stride * h)
                    .map(|_| T::from_i32(rng.range(0, (1 << bd) - 1)))
                    .collect();
                let mut want = init.clone();
                let mut got = init;
                portable::cdef_filter(
                    &mut want, dst_stride, &src, stride, w, h, pri, sec, damping, dir, bd,
                );
                simd_cdef_filter(
                    &mut got, dst_stride, &src, stride, w, h, pri, sec, damping, dir, bd,
                );
                assert_eq!(
                    got, want,
                    "iter {iter} bd {bd} {w}x{h} pri {pri} sec {sec} damping {damping} dir {dir}"
                );
            }
        }
    }

    /// Times `f` on every source of `srcs` (ns per call).
    fn time<T: Pixel>(srcs: &[Vec<u16>], f: impl Fn(&mut [T], &[u16])) -> f64 {
        let mut dst = vec![T::default(); 8 * 8];
        let reps = 3000;
        let mut t = std::time::Instant::now();
        // The first reps warm up caches and the clock frequency.
        for rep in 0..reps + 50 {
            if rep == 50 {
                t = std::time::Instant::now();
            }
            for s in srcs {
                f(&mut dst, std::hint::black_box(s));
                std::hint::black_box(&mut dst);
            }
        }
        t.elapsed().as_nanos() as f64 / (reps * srcs.len()) as f64
    }

    /// Wall time of the portable and SIMD filters: `cargo test --release
    /// --lib -- --ignored --nocapture
    /// simd::cdef::tests::bench`
    #[test]
    #[ignore]
    fn bench_filter() {
        fn run<T: Pixel>(bd: u32) {
            let cs = bd - 8;
            let mut rng = Rng(7);
            // Warm up the clock frequency.
            let t = std::time::Instant::now();
            while t.elapsed().as_millis() < 300 {
                std::hint::black_box(gen_src(&mut rng, 8, 8, 12, bd));
            }
            for (w, h) in [(8usize, 8usize), (4, 8), (4, 4)] {
                let stride = w + 4;
                // Interior blocks (all of the border available), as most
                // blocks of a frame.
                let srcs: Vec<_> = (0..64)
                    .map(|_| {
                        let mut s = gen_src(&mut rng, w, h, stride, bd);
                        for v in s.iter_mut() {
                            *v = (*v).min((1 << bd) - 1);
                        }
                        s
                    })
                    .collect();
                for (name, pri, sec) in [
                    ("pri", 6 << cs, 0),
                    ("sec", 0, 2 << cs),
                    ("both", 6 << cs, 2 << cs),
                ] {
                    let damping = 5 + cs as i32;
                    let p = time::<T>(&srcs, |d, s| {
                        portable::cdef_filter(d, 8, s, stride, w, h, pri, sec, damping, 3, bd)
                    });
                    let q = time::<T>(&srcs, |d, s| {
                        simd_cdef_filter(d, 8, s, stride, w, h, pri, sec, damping, 3, bd)
                    });
                    // damping 8 disables the 8-bit u8 path (same cost
                    // otherwise).
                    let r = time::<T>(&srcs, |d, s| {
                        simd_cdef_filter(d, 8, s, stride, w, h, pri, sec, 8 + cs as i32, 3, bd)
                    });
                    println!("filter bd {bd} {w}x{h} {name}: portable {p:.1} ns, simd {q:.1} ns ({:.2}x), i16 path {r:.1} ns", p / q);
                }
            }
        }
        run::<u8>(8);
        run::<u16>(10);
    }

    /// Wall time of the 8-bit `u8` lane kernels of [`SimdCdef::filter`]
    /// (interior blocks), new AVX2 forms against the 16-lane ones:
    /// `RUSTFLAGS="-C target-cpu=x86-64-v3" cargo test --release
    /// --lib -- --ignored --nocapture bench_px`
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[test]
    #[ignore]
    fn bench_px() {
        use super::u8_path;
        let mut rng = Rng(9);
        let stride = 32;
        let t = std::time::Instant::now();
        while t.elapsed().as_millis() < 300 {
            std::hint::black_box(gen_src(&mut rng, 8, 8, 12, 8));
        }
        type K = fn(&mut [u8], usize, &[u8], usize, i32, i32, i32, usize);
        let time = |srcs: &[Vec<u8>], f: K, pri: i32, sec: i32| -> f64 {
            let mut dst = vec![0u8; 8 * 8];
            let reps = 3000;
            let mut t = std::time::Instant::now();
            for rep in 0..reps + 50 {
                if rep == 50 {
                    t = std::time::Instant::now();
                }
                for s in srcs {
                    f(&mut dst, 8, std::hint::black_box(s), stride, pri, sec, 5, 0);
                    std::hint::black_box(&mut dst);
                }
            }
            t.elapsed().as_nanos() as f64 / (reps * srcs.len()) as f64
        };
        for (w, h) in [(8usize, 8usize), (4, 4)] {
            let srcs: Vec<Vec<u8>> = (0..64)
                .map(|_| {
                    let mut s: Vec<u8> = gen_src(&mut rng, w, h, stride, 8)
                        .iter()
                        .map(|&v| v.min(255) as u8)
                        .collect();
                    s.resize(s.len() + 16, 0);
                    s
                })
                .collect();
            let cases: [(&str, i32, i32, K, K); 3] = if w == 8 {
                [
                    (
                        "pri",
                        6,
                        0,
                        u8_path::filter_8x8::<u8, 1, true, false>,
                        u8_path::avx2::filter_8x8::<u8, 1, true, false>,
                    ),
                    (
                        "sec",
                        0,
                        2,
                        u8_path::filter_8x8::<u8, 1, false, true>,
                        u8_path::avx2::filter_8x8::<u8, 1, false, true>,
                    ),
                    (
                        "both",
                        6,
                        2,
                        u8_path::filter_8x8::<u8, 1, true, true>,
                        u8_path::avx2::filter_8x8::<u8, 1, true, true>,
                    ),
                ]
            } else {
                [
                    (
                        "pri",
                        6,
                        0,
                        u8_path::filter_4x4::<u8, 1, true, false>,
                        u8_path::avx2::filter_4x4::<u8, 1, true, false>,
                    ),
                    (
                        "sec",
                        0,
                        2,
                        u8_path::filter_4x4::<u8, 1, false, true>,
                        u8_path::avx2::filter_4x4::<u8, 1, false, true>,
                    ),
                    (
                        "both",
                        6,
                        2,
                        u8_path::filter_4x4::<u8, 1, true, true>,
                        u8_path::avx2::filter_4x4::<u8, 1, true, true>,
                    ),
                ]
            };
            for (name, pri, sec, old, new) in cases {
                let (mut o, mut n) = (f64::MAX, f64::MAX);
                for _ in 0..5 {
                    o = o.min(time(&srcs, old, pri, sec));
                    n = n.min(time(&srcs, new, pri, sec));
                }
                println!(
                    "u8 {w}x{h} {name}: old {o:.1} ns, new {n:.1} ns ({:.2}x)",
                    o / n
                );
            }
        }
    }

    #[test]
    fn filter_vs_portable() {
        filter_random::<u8>(8, 1);
        filter_random::<u16>(8, 2);
        filter_random::<u16>(10, 3);
        filter_random::<u16>(12, 4);
    }

    /// The pixel source kernels ([`SimdCdef`] against [`PortableCdef`]):
    /// the 8-bit `u8` lane path (all parameters it accepts and the limits
    /// around them, extreme sample values), the 16-bit lane path and the
    /// padded path of blocks at the frame border, with garbage in the
    /// unavailable border.
    #[test]
    fn filter_px_vs_portable() {
        use crate::decoder::av1::dsp::cdef::{
            CdefKernels, PortableCdef, CDEF_HAVE_ALL, CDEF_SRC_SLACK,
        };
        fn run<T: Pixel, const BD: u32>(seed: u64) {
            let mut rng = Rng(seed);
            let cs = BD - 8;
            let maxv = (1 << BD) - 1;
            for (w, h) in [(8usize, 8usize), (4, 8), (8, 4), (4, 4)] {
                for iter in 0..20000 {
                    let stride = w + 4 + rng.range(0, 5) as usize;
                    let mut src = gen_src(&mut rng, w, h, stride, BD);
                    src.resize(src.len() + CDEF_SRC_SLACK, 0);
                    let kind = rng.range(0, 3);
                    for v in src.iter_mut() {
                        *v = match kind {
                            0 => (*v).min(maxv as u16),
                            1 => [0, maxv][rng.range(0, 1) as usize] as u16,
                            2 => [0, 1, maxv - 1, maxv][rng.range(0, 3) as usize] as u16,
                            _ => rng.range(0, maxv) as u16,
                        };
                    }
                    let src: Vec<T> = src.iter().map(|&v| T::from_i32(v as i32)).collect();
                    let edges =
                        if rng.range(0, 2) != 0 { CDEF_HAVE_ALL } else { rng.range(0, 15) as u8 };
                    let (pri, sec, damping) = match rng.range(0, 3) {
                        0 => (
                            rng.range(0, 16) << cs,
                            [0, 1, 2, 4, 8, 15, 16][rng.range(0, 6) as usize] << cs,
                            rng.range(2, 8) + cs as i32,
                        ),
                        1 => (
                            rng.range(0, 15 << cs),
                            [0, 1, 2, 4][rng.range(0, 3) as usize] << cs,
                            rng.range(2, 9) + cs as i32,
                        ),
                        2 => (rng.range(0, 1023), rng.range(0, 1023), rng.range(0, 13)),
                        _ => (rng.range(0, 5000), rng.range(0, 5000), rng.range(0, 20)),
                    };
                    if pri == 0 && sec == 0 {
                        continue;
                    }
                    let dir = rng.range(0, 7) as usize;
                    let dst_stride = w + rng.range(0, 3) as usize;
                    let init: Vec<T> = (0..dst_stride * h)
                        .map(|_| T::from_i32(rng.range(0, maxv)))
                        .collect();
                    let mut want = init.clone();
                    let mut got = init;
                    PortableCdef::filter(
                        &mut want, dst_stride, &src, stride, w, h, pri, sec, damping, dir, edges,
                        BD,
                    );
                    SimdCdef::<BD>::filter(
                        &mut got, dst_stride, &src, stride, w, h, pri, sec, damping, dir, edges, BD,
                    );
                    assert_eq!(got, want, "iter {iter} bd {BD} {w}x{h} pri {pri} sec {sec} damping {damping} dir {dir} edges {edges}");
                }
            }
        }
        run::<u8, 8>(31);
        run::<u16, 8>(32);
        run::<u16, 10>(33);
        run::<u16, 12>(34);
    }

    /// Random 8x8 block (as in the portable tests): noise, oriented
    /// stripes, extremes or a flat area with small noise.
    fn gen_block<T: Pixel>(rng: &mut Rng, bd: u32, stride: usize) -> Vec<T> {
        let maxv = (1 << bd) - 1;
        let kind = rng.range(0, 3);
        let (a, c) = (rng.range(-4, 4), rng.range(-4, 4));
        let amp = rng.range(0, maxv / 2);
        let base = rng.range(0, maxv);
        let mut buf = vec![T::default(); stride * 7 + 8];
        for i in 0..8 {
            for j in 0..8 {
                let ph = (a * i as i32 + c * j as i32).rem_euclid(4);
                let v = match kind {
                    0 => rng.range(0, maxv),
                    1 => (base + if ph < 2 { amp } else { -amp }).clamp(0, maxv),
                    2 => [0, maxv][rng.range(0, 1) as usize],
                    _ => (base + rng.range(-8, 8)).clamp(0, maxv),
                };
                buf[i * stride + j] = T::from_i32(v);
            }
        }
        buf
    }

    fn direction_random<T: Pixel>(bd: u32, seed: u64) {
        let mut rng = Rng(seed);
        let mut dirs = [0usize; 8];
        for _ in 0..50000 {
            let stride = 8 + rng.range(0, 8) as usize;
            let buf = gen_block::<T>(&mut rng, bd, stride);
            let got = simd_cdef_direction(&buf, stride, bd);
            assert_eq!(
                got,
                portable::cdef_direction(&buf, stride, bd),
                "bd {bd} {buf:?}"
            );
            dirs[got.0] += 1;
        }
        assert!(dirs.iter().all(|&d| d > 100), "{dirs:?}");
        // Constant blocks (all costs 0) and the extreme values.
        for v in [0, (1 << bd) - 1] {
            let buf = vec![T::from_i32(v); 64];
            assert_eq!(
                simd_cdef_direction(&buf, 8, bd),
                portable::cdef_direction(&buf, 8, bd)
            );
        }
    }

    #[test]
    fn direction_vs_portable() {
        direction_random::<u8>(8, 21);
        direction_random::<u16>(8, 22);
        direction_random::<u16>(10, 23);
        direction_random::<u16>(12, 24);
    }

    /// Wall time of the portable and SIMD direction search (see
    /// [`bench_filter`]).
    #[test]
    #[ignore]
    fn bench_direction() {
        fn run<T: Pixel>(bd: u32) {
            let mut rng = Rng(8);
            let blocks: Vec<_> = (0..512).map(|_| gen_block::<T>(&mut rng, bd, 8)).collect();
            let time = |f: fn(&[T], usize, u32) -> (usize, i32)| {
                let reps = 200;
                let mut t = std::time::Instant::now();
                for rep in 0..reps + 50 {
                    if rep == 50 {
                        t = std::time::Instant::now();
                    }
                    for b in &blocks {
                        std::hint::black_box(f(std::hint::black_box(b), 8, bd));
                    }
                }
                t.elapsed().as_nanos() as f64 / (reps * blocks.len()) as f64
            };
            let p = time(portable::cdef_direction::<T>);
            let q = time(simd_cdef_direction::<T>);
            println!(
                "direction bd {bd}: portable {p:.1} ns, simd {q:.1} ns ({:.2}x)",
                p / q
            );
        }
        run::<u8>(8);
        run::<u16>(10);
    }
}
