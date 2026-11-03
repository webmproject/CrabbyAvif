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

//! SIMD intra predictors (see `dsp::ipred` for the contracts).
//!
//! Every kernel is dispatched on the block width to a version with a
//! compile time width `W` (and vector width `N`), so that all row loops
//! have a fixed trip count and operate on whole `Simd` vectors.

use std::any::Any;
use std::simd::prelude::*;
use std::simd::SimdElement;

use crate::decoder::av1::dsp::ipred::{
    edge_filter_strength, filter_corner, sm_weights, use_upsample, IntraEdges, IntraPredParams,
    ANGLE_STEP, D67_PRED, DC_PRED, DR_INTRA_DERIVATIVE, EDGE_BUF_LEN, EDGE_PAD, INTRA_FILTER_TAPS,
    MAX_CFL_AC, MAX_EDGE, MAX_UPSAMPLE_PX, MODE_TO_ANGLE, PAETH_PRED, SMOOTH_H_PRED, SMOOTH_PRED,
    SMOOTH_V_PRED, STRENGTH_LUT, V_PRED,
};
use crate::decoder::av1::pixel::Pixel;

/// Holder of the SIMD predictor table (used by the tests to compare
/// individual kernels).
#[cfg(test)]
struct SimdKernels<T, const BD: u32>(std::marker::PhantomData<T>);

#[cfg(test)]
impl<T: Pixel, const BD: u32> SimdKernels<T, BD> {
    /// A constant, so that `predict_intra` passes a static table instead
    /// of building one per call.
    const TABLE: crate::decoder::av1::dsp::ipred::IpredKernels<T> =
        crate::decoder::av1::dsp::ipred::IpredKernels {
            dc: ipred_dc::<T, BD>,
            dc_top: ipred_dc_top::<T, BD>,
            dc_left: ipred_dc_left::<T, BD>,
            dc_128: ipred_dc_128::<T, BD>,
            v: ipred_v::<T, BD>,
            h: ipred_h::<T, BD>,
            paeth: ipred_paeth::<T, BD>,
            smooth: ipred_smooth::<T, BD>,
            smooth_v: ipred_smooth_v::<T, BD>,
            smooth_h: ipred_smooth_h::<T, BD>,
            z1: ipred_z1::<T, BD>,
            z2: ipred_z2::<T, BD>,
            z3: ipred_z3::<T, BD>,
            filter: ipred_filter::<T, BD>,
        };
}

/// `dsp::ipred::predict_intra` using the SIMD predictors.
pub fn predict_intra<T: Pixel>(
    dst: &mut [T],
    stride: usize,
    edges: &IntraEdges<T>,
    p: &IntraPredParams,
) {
    match p.bd {
        8 => predict_intra_bd::<T, 8>(dst, stride, edges, p),
        10 => predict_intra_bd::<T, 10>(dst, stride, edges, p),
        _ => predict_intra_bd::<T, 12>(dst, stride, edges, p),
    }
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

/// Calls `$f::<T, W, N>(args)` for the runtime width `$w` (4..=64), with
/// `N = min(W, 16)` the vector width used for rows of `W` samples.
macro_rules! by_width {
    ($w:expr, $f:ident::<$($t:ty),+>($($a:expr),* $(,)?)) => {
        match $w {
            4 => $f::<$($t),+, 4, 4>($($a),*),
            8 => $f::<$($t),+, 8, 8>($($a),*),
            16 => $f::<$($t),+, 16, 16>($($a),*),
            32 => $f::<$($t),+, 32, 16>($($a),*),
            64 => $f::<$($t),+, 64, 16>($($a),*),
            w => panic!("invalid block width {w}"),
        }
    };
}

/// Bounds-checks `dst` for a `W x h` block (`h` in 1..=64) with row
/// stride `stride`, returning the bounded slice and masked `stride` so
/// row offsets `i * stride` for `i < h` cannot overflow 64-bit `usize`.
#[inline(always)]
fn check_block<T, const W: usize>(dst: &mut [T], stride: usize, h: usize) -> (&mut [T], usize) {
    let stride = stride & 0xffff_ffff;
    assert!(stride >= W && (1..=64).contains(&h));
    (&mut dst[..(h - 1) * stride + W], stride)
}

/// Bounds-checks `dst` for a `W x h` block where `h` is a multiple of 4
/// in `4..=64`, returning `(dst, stride, h4)` with `h4 = h / 4`.
#[inline(always)]
fn check_block4<T, const W: usize>(
    dst: &mut [T],
    stride: usize,
    h: usize,
) -> (&mut [T], usize, usize) {
    let stride = (stride & 0xffff_ffff).max(W);
    let h4 = h >> 2;
    assert!((1..=16).contains(&h4));
    let s3 = stride * 3;
    let s4 = stride * 4;
    (&mut dst[..(h4 - 1) * s4 + s3 + W], stride, h4)
}

/// Splits a window of at least `3 * stride + W` samples (`stride >= W`)
/// into 4 consecutive `W`-sample rows separated by `stride`.
#[inline(always)]
fn split_rows4<T, const W: usize>(
    win: &mut [T],
    stride: usize,
) -> (&mut [T; W], &mut [T; W], &mut [T; W], &mut [T; W]) {
    let (r0, win) = win.split_at_mut(stride);
    let (r1, win) = win.split_at_mut(stride);
    let (r2, r3) = win.split_at_mut(stride);
    (
        (&mut r0[..W]).try_into().unwrap(),
        (&mut r1[..W]).try_into().unwrap(),
        (&mut r2[..W]).try_into().unwrap(),
        (&mut r3[..W]).try_into().unwrap(),
    )
}

/// Calls `f(i4, r0, r1, r2, r3)` for each group of 4 consecutive `W`-sample
/// rows (`i4 in 0..h / 4`, `h` a multiple of 4 in `4..=64`). Slices `dst`
/// into `head` (`chunks_exact_mut(stride * 4)`) and `tail` (`stride * 3 + W`)
/// up front so the loop has no `imul` or per-iteration bounds checks.
#[inline(always)]
fn visit_row4<T, const W: usize>(
    dst: &mut [T],
    stride: usize,
    h: usize,
    mut f: impl FnMut(usize, &mut [T; W], &mut [T; W], &mut [T; W], &mut [T; W]),
) {
    let (dst, stride, h4) = check_block4::<T, W>(dst, stride, h);
    let s4 = stride * 4;
    let head_len = (h4 - 1) * s4;
    let (head, tail) = dst.split_at_mut(head_len);
    for (i4, win) in head.chunks_exact_mut(s4).enumerate() {
        let (r0, r1, r2, r3) = split_rows4::<T, W>(win, stride);
        f(i4, r0, r1, r2, r3);
    }
    let (r0, r1, r2, r3) = split_rows4::<T, W>(tail, stride);
    f(h4 - 1, r0, r1, r2, r3);
}

/// Returns the `W`-sample row starting at `off` in `dst`.
#[inline(always)]
fn row_mut<T, const W: usize>(dst: &mut [T], off: usize) -> &mut [T; W] {
    (&mut dst[off..off + W]).try_into().unwrap()
}

/// Returns 4 consecutive `W`-sample rows starting at row `4 * i` in `dst`
/// (`i < h4`, `stride >= W` checked by [`check_block4`]).
#[inline(always)]
fn row4_mut<T, const W: usize>(
    dst: &mut [T],
    i: usize,
    stride: usize,
) -> (&mut [T; W], &mut [T; W], &mut [T; W], &mut [T; W]) {
    let s3 = stride * 3;
    let base = i * (stride * 4);
    split_rows4::<T, W>(&mut dst[base..base + s3 + W], stride)
}

/// Fills a `W x h` block with `v` (`h` a multiple of 4 in `4..=64`).
#[inline(always)]
fn fill<T: Pixel, const W: usize>(dst: &mut [T], stride: usize, h: usize, v: T) {
    let src = [v; W];
    visit_row4::<T, W>(dst, stride, h, |_, r0, r1, r2, r3| {
        *r0 = src;
        *r1 = src;
        *r2 = src;
        *r3 = src;
    });
}

/// Loads `N` samples as `u16` lanes.
///
/// The sample array is downcast to its concrete type (resolved at
/// compile time) so that this is a plain vector load and widening.
#[inline(always)]
fn load_u16<T: Pixel, const N: usize>(src: &[T]) -> Simd<u16, N> {
    let src: &[T; N] = src[..N].try_into().unwrap();
    let any: &dyn Any = src;
    if let Some(s) = any.downcast_ref::<[u8; N]>() {
        Simd::from_array(*s).cast()
    } else if let Some(s) = any.downcast_ref::<[u16; N]>() {
        Simd::from_array(*s)
    } else {
        Simd::from_array(std::array::from_fn(|i| src[i].to_u16()))
    }
}

/// Loads `N` samples as `i16` lanes.
#[inline(always)]
fn load_i16<T: Pixel, const N: usize>(src: &[T]) -> Simd<i16, N> {
    load_u16::<T, N>(src).cast()
}

/// Stores `N` `u16` lanes (already in the pixel range) to `dst[..N]`.
#[inline(always)]
fn store_u16<T: Pixel, const N: usize>(dst: &mut [T], v: Simd<u16, N>) {
    let dst: &mut [T; N] = (&mut dst[..N]).try_into().unwrap();
    let any: &mut dyn Any = dst;
    if let Some(d) = any.downcast_mut::<[u8; N]>() {
        *d = v.cast::<u8>().to_array();
    } else if let Some(d) = any.downcast_mut::<[u16; N]>() {
        *d = v.to_array();
    } else {
        for (d, &x) in dst.iter_mut().zip(v.as_array()) {
            *d = T::from_i32(x as i32);
        }
    }
}

/// Stores `N` `i16` lanes (already in the pixel range).
#[inline(always)]
fn store_i16<T: Pixel, const N: usize>(dst: &mut [T], v: Simd<i16, N>) {
    store_u16(dst, v.cast::<u16>());
}

/// Stores `N` `i32` lanes (already in the pixel range).
#[inline(always)]
fn store_i32<T: Pixel, const N: usize>(dst: &mut [T], v: Simd<i32, N>) {
    store_u16(dst, v.cast::<u16>());
}

/// Sum of `e[..n]` (`n` a multiple of `N` in `4..=64`).
#[inline(always)]
fn edge_sum<T: Pixel, const BD: u32, const N: usize>(e: &[T], n: usize) -> u32 {
    if BD <= 10 {
        // At most 64 * 1023 = 65472, which fits in u16 without overflow.
        let mut acc = Simd::<u16, N>::splat(0);
        for c in e[..n].as_chunks::<N>().0 {
            acc += load_u16::<T, N>(c);
        }
        acc.reduce_sum() as u32
    } else {
        let mut acc = Simd::<u32, N>::splat(0);
        for c in e[..n].as_chunks::<N>().0 {
            acc += load_u16::<T, N>(c).cast::<u32>();
        }
        acc.reduce_sum()
    }
}

/// [`edge_sum`] for a compile-time edge length `M` (4..=64).
#[inline(always)]
fn edge_sum_c<T: Pixel, const BD: u32, const M: usize>(e: &[T]) -> u32 {
    match M {
        4 => edge_sum::<T, BD, 4>(e, 4),
        8 => edge_sum::<T, BD, 8>(e, 8),
        16 => edge_sum::<T, BD, 16>(e, 16),
        32 => edge_sum::<T, BD, 16>(e, 32),
        _ => edge_sum::<T, BD, 16>(e, 64),
    }
}

/// [`edge_sum`] for a runtime edge length `n` (4..=64).
#[inline(always)]
fn edge_sum_n<T: Pixel, const BD: u32>(e: &[T], n: usize) -> u32 {
    match n {
        4 => edge_sum::<T, BD, 4>(e, 4),
        8 => edge_sum::<T, BD, 8>(e, 8),
        16 => edge_sum::<T, BD, 16>(e, 16),
        32 => edge_sum::<T, BD, 16>(e, 32),
        _ => edge_sum::<T, BD, 16>(e, 64),
    }
}

// ---------------------------------------------------------------------------
// DC kernels.
// ---------------------------------------------------------------------------

/// Calls `$f::<T.., W, H>(args)` for the runtime block size `$w x $h`
/// (powers of two in 4..=64, aspect ratio at most 4), through one jump
/// table on `log2(w) - 2 + 5 * (log2(h) - 2)`.
macro_rules! by_size {
    ($w:expr, $h:expr, $f:ident::<$($t:ty),+>($($a:expr),* $(,)?)) => {
        match ($w.trailing_zeros() - 2) + 5 * ($h.trailing_zeros() - 2) {
            0 => $f::<$($t),+, 4, 4>($($a),*),
            1 => $f::<$($t),+, 8, 4>($($a),*),
            2 => $f::<$($t),+, 16, 4>($($a),*),
            5 => $f::<$($t),+, 4, 8>($($a),*),
            6 => $f::<$($t),+, 8, 8>($($a),*),
            7 => $f::<$($t),+, 16, 8>($($a),*),
            8 => $f::<$($t),+, 32, 8>($($a),*),
            10 => $f::<$($t),+, 4, 16>($($a),*),
            11 => $f::<$($t),+, 8, 16>($($a),*),
            12 => $f::<$($t),+, 16, 16>($($a),*),
            13 => $f::<$($t),+, 32, 16>($($a),*),
            14 => $f::<$($t),+, 64, 16>($($a),*),
            16 => $f::<$($t),+, 8, 32>($($a),*),
            17 => $f::<$($t),+, 16, 32>($($a),*),
            18 => $f::<$($t),+, 32, 32>($($a),*),
            19 => $f::<$($t),+, 64, 32>($($a),*),
            22 => $f::<$($t),+, 16, 64>($($a),*),
            23 => $f::<$($t),+, 32, 64>($($a),*),
            24 => $f::<$($t),+, 64, 64>($($a),*),
            _ => panic!("invalid block size {}x{}", $w, $h),
        }
    };
}

/// `dsp::ipred::ipred_dc`.
#[inline]
pub fn ipred_dc<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    w: usize,
    h: usize,
) {
    let v = by_size!(w, h, dc_both_block::<T, BD>(above, left));
    by_width!(w, dc_fill::<T>(dst, stride, h, v));
}

#[inline(always)]
fn dc_edge_val<T: Pixel, const BD: u32>(edge: &[T], n: usize) -> T {
    let sum = edge_sum_n::<T, BD>(edge, n);
    let avg = (sum + (n as u32 >> 1)) >> n.trailing_zeros();
    T::from_i32(avg as i32)
}

#[inline(always)]
fn dc_fill<T: Pixel, const W: usize, const N: usize>(dst: &mut [T], stride: usize, h: usize, v: T) {
    fill::<T, W>(dst, stride, h, v);
}

/// Computes the DC value of a `W x H` block when both `above` and `left`
/// are available, with the edge sums and division specialized for `W x H`.
/// For 8/10-bit blocks up to 16x16, both edges are summed in a single
/// `u16` vector of width `max(W, H)` before one horizontal reduction.
#[inline(always)]
fn dc_both_block<T: Pixel, const BD: u32, const W: usize, const H: usize>(
    above: &[T],
    left: &[T],
) -> T {
    let sum =
        if BD <= 10 && W <= 16 && H <= 16 {
            // Each lane holds at most 2 * 1023, so the 4/8/16-lane reduction
            // (at most 32 * 1023 = 32736) fits u16.
            match (W, H) {
                (4, 4) => (load_u16::<T, 4>(above) + load_u16::<T, 4>(left)).reduce_sum() as u32,
                (8, 4) => (load_u16::<T, 8>(above) + load_u16::<T, 4>(left).resize::<8>(0))
                    .reduce_sum() as u32,
                (4, 8) => (load_u16::<T, 4>(above).resize::<8>(0) + load_u16::<T, 8>(left))
                    .reduce_sum() as u32,
                (8, 8) => (load_u16::<T, 8>(above) + load_u16::<T, 8>(left)).reduce_sum() as u32,
                (16, 4) => (load_u16::<T, 16>(above) + load_u16::<T, 4>(left).resize::<16>(0))
                    .reduce_sum() as u32,
                (4, 16) => (load_u16::<T, 4>(above).resize::<16>(0) + load_u16::<T, 16>(left))
                    .reduce_sum() as u32,
                (16, 8) => (load_u16::<T, 16>(above) + load_u16::<T, 8>(left).resize::<16>(0))
                    .reduce_sum() as u32,
                (8, 16) => (load_u16::<T, 8>(above).resize::<16>(0) + load_u16::<T, 16>(left))
                    .reduce_sum() as u32,
                _ => (load_u16::<T, 16>(above) + load_u16::<T, 16>(left)).reduce_sum() as u32,
            }
        } else {
            edge_sum_c::<T, BD, W>(above) + edge_sum_c::<T, BD, H>(left)
        };
    T::from_i32(dc_div::<W, H>(sum) as i32)
}

/// `(sum + (W + H) / 2) / (W + H)` for `W + H` = 2^k, 3 * 2^k or 5 * 2^k.
#[inline(always)]
fn dc_div<const W: usize, const H: usize>(sum: u32) -> u32 {
    let n = (W + H) as u32;
    let k = n.trailing_zeros();
    let x = (sum + (n >> 1)) >> k;
    match n >> k {
        1 => x,
        3 => ((x as u64 * 0xAAAA_AAAB) >> 33) as u32,
        _ => ((x as u64 * 0xCCCC_CCCD) >> 34) as u32,
    }
}

/// `dsp::ipred::ipred_dc_top`.
#[inline]
pub fn ipred_dc_top<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    w: usize,
    h: usize,
) {
    let v = dc_edge_val::<T, BD>(above, w);
    by_width!(w, dc_fill::<T>(dst, stride, h, v));
}

/// `dsp::ipred::ipred_dc_left`.
#[inline]
pub fn ipred_dc_left<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    left: &[T],
    w: usize,
    h: usize,
) {
    let v = dc_edge_val::<T, BD>(left, h);
    by_width!(w, dc_fill::<T>(dst, stride, h, v));
}

/// `dsp::ipred::ipred_dc_128`.
#[inline]
pub fn ipred_dc_128<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    w: usize,
    h: usize,
    bd: u32,
) {
    by_width!(w, dc_fill::<T>(dst, stride, h, T::from_i32(1 << (bd - 1))));
}

// ---------------------------------------------------------------------------
// V / H kernels.
// ---------------------------------------------------------------------------

/// `dsp::ipred::ipred_v`.
#[inline]
pub fn ipred_v<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    w: usize,
    h: usize,
) {
    by_width!(w, v_w::<T>(dst, stride, above, h));
}

#[inline(always)]
fn v_w<T: Pixel, const W: usize, const N: usize>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    h: usize,
) {
    let src: [T; W] = above[..W].try_into().unwrap();
    visit_row4::<T, W>(dst, stride, h, |_, r0, r1, r2, r3| {
        *r0 = src;
        *r1 = src;
        *r2 = src;
        *r3 = src;
    });
}

/// `dsp::ipred::ipred_h`.
#[inline]
pub fn ipred_h<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    left: &[T],
    w: usize,
    h: usize,
) {
    by_width!(w, h_w::<T>(dst, stride, left, h));
}

#[inline(always)]
fn h_w<T: Pixel, const W: usize, const N: usize>(
    dst: &mut [T],
    stride: usize,
    left: &[T],
    h: usize,
) {
    let h4 = h >> 2;
    assert!((1..=16).contains(&h4));
    let left4 = left[..h4 * 4].as_chunks::<4>().0;
    visit_row4::<T, W>(dst, stride, h, |i, r0, r1, r2, r3| {
        let [l0, l1, l2, l3] = left4[i];
        *r0 = [l0; W];
        *r1 = [l1; W];
        *r2 = [l2; W];
        *r3 = [l3; W];
    });
}

// ---------------------------------------------------------------------------
// Paeth kernel.
// ---------------------------------------------------------------------------

/// `dsp::ipred::ipred_paeth`.
#[inline(never)]
pub fn ipred_paeth<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    top_left: T,
    above: &[T],
    left: &[T],
    w: usize,
    h: usize,
) {
    by_width!(w, paeth_w::<T>(dst, stride, top_left, above, left, h));
}

/// Paeth on `i16` lanes: all intermediate values (`a + l - 2 * tl` at
/// most) are within +-8190 for 12-bit samples.
#[inline(always)]
fn paeth_w<T: Pixel, const W: usize, const N: usize>(
    dst: &mut [T],
    stride: usize,
    top_left: T,
    above: &[T],
    left: &[T],
    h: usize,
) {
    let (dst, stride, h4) = check_block4::<T, W>(dst, stride, h);
    let tl_s = top_left.to_i32() as i16;
    let tl = Simd::<i16, N>::splat(tl_s);
    let above = &above[..W];
    let left4 = left[..h4 * 4].as_chunks::<4>().0;
    // Column chunks outside, rows inside: `a`, `a - tl`, and `|a - tl|`
    // are row-invariant and loaded/computed once per column chunk.
    for x in (0..W).step_by(N) {
        let a = load_i16::<T, N>(&above[x..]);
        let a_tl = a - tl;
        let p_left = a_tl.abs();
        let dst_x = &mut dst[x..];
        visit_row4::<T, N>(dst_x, stride, h, |i4, r0, r1, r2, r3| {
            let l4 = &left4[i4];
            for (r, &l_pix) in [r0, r1, r2, r3].into_iter().zip(l4) {
                let l_s = l_pix.to_i32() as i16;
                let l_tl = l_s - tl_s;
                let lv = Simd::splat(l_s);
                let p_top = Simd::splat(l_tl.abs());
                let p_top_left = (a_tl + Simd::splat(l_tl)).abs();
                let use_left = p_left.simd_le(p_top) & p_left.simd_le(p_top_left);
                let use_above = p_top.simd_le(p_top_left);
                store_i16(r, use_left.select(lv, use_above.select(a, tl)));
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Smooth kernels.
//
// u8 samples use u16 lanes: `w * a + (256 - w) * b` is at most 256 * 255.
// Other samples use i32 lanes.
// ---------------------------------------------------------------------------

/// Loads `N` smooth weights.
#[inline(always)]
fn weights<E: SimdElement + From<u8>, const N: usize>(w: &[u8]) -> Simd<E, N> {
    Simd::from_array(std::array::from_fn(|i| E::from(w[i])))
}

/// `dsp::ipred::ipred_smooth`.
#[inline(never)]
pub fn ipred_smooth<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    w: usize,
    h: usize,
) {
    if BD == 8 {
        if w == 4 {
            return smooth_packed::<T, 4, 4>(dst, stride, above, left, h);
        }
        if cfg!(target_feature = "avx2") && w == 8 {
            return smooth_packed::<T, 8, 2>(dst, stride, above, left, h);
        }
    }
    by_width!(w, smooth_w::<T, BD>(dst, stride, above, left, h));
}

/// 8-bit [`smooth_w`] for narrow blocks (`W` 4 or 8): `R = 16 / W` rows
/// are computed in one 16-lane vector (lane `j` is row `j / W`, column
/// `j % W`), with the same `u16` arithmetic as [`smooth_w`].
#[inline(always)]
fn smooth_packed<T: Pixel, const W: usize, const R: usize>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    h: usize,
) {
    const { assert!(W * R == 16) };
    let (dst, stride, h4) = check_block4::<T, W>(dst, stride, h);
    let h = h4 * 4;
    let wx = &sm_weights(W)[..W];
    let wy = sm_weights(h)[..h].as_chunks::<R>().0;
    let left_c = left[..h].as_chunks::<R>().0;
    let above = &above[..W];
    let bottom = left[h - 1].to_u16();
    let right = above[W - 1].to_u16();
    type V = Simd<u16, 16>;
    let a_diff = V::from_array(std::array::from_fn(|j| above[j % W].to_u16())) - V::splat(bottom);
    let wxv = V::from_array(std::array::from_fn(|j| wx[j % W] as u16));
    let hbase = (V::splat(256) - wxv) * V::splat(right) + V::splat(128);
    let vbase = V::splat(bottom * 256 + 128);
    let one = V::splat(1);
    for (g, (wy_r, left_r)) in wy.iter().zip(left_c).enumerate() {
        let i0 = g * R;
        let wyv = V::from_array(std::array::from_fn(|j| wy_r[j / W] as u16));
        let lv = V::from_array(std::array::from_fn(|j| left_r[j / W].to_u16()));
        // (256 - wy) * bottom + wy * a == 256 * bottom + wy * (a - bottom)
        // (mod 2^16, and the true value fits u16).
        let vert = wyv * a_diff + vbase;
        let horz = wxv * lv + hbase;
        let p = ((vert >> one) + (horz >> one) + (vert & horz & one)) >> 8;
        store_rows_u8::<T, W>(dst, stride, i0, p.cast::<i16>());
    }
}

#[inline(always)]
fn smooth_w<T: Pixel, const BD: u32, const W: usize, const N: usize>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    h: usize,
) {
    let (dst, stride, h4) = check_block4::<T, W>(dst, stride, h);
    let h = h4 * 4;
    let wx = &sm_weights(W)[..W];
    let wy4 = sm_weights(h)[..h].as_chunks::<4>().0;
    let left4 = left[..h].as_chunks::<4>().0;
    let bottom = left[h - 1].to_i32();
    let right = above[W - 1].to_i32();
    let above = &above[..W];
    // Column chunks outside, rows inside: the per column vectors are
    // loaded once per chunk.
    for x in (0..W).step_by(N) {
        let dst_x = &mut dst[x..];
        if BD == 8 {
            // Add 128 to each of vert and horz (256 total == 1 << 8) so
            // both stay <= 256 * 255 + 128 = 65408 (fits u16), then
            // (vert + horz) >> 9 == ((vert >> 1) + (horz >> 1) + (vert & horz & 1)) >> 8.
            let one = Simd::<u16, N>::splat(1);
            let wxv = weights::<u16, N>(&wx[x..]);
            let a_diff = load_u16::<T, N>(&above[x..]) - Simd::splat(bottom as u16);
            let vbase = Simd::<u16, N>::splat(bottom as u16 * 256 + 128);
            let hbase = (Simd::splat(256) - wxv) * Simd::splat(right as u16) + Simd::splat(128);
            visit_row4::<T, N>(dst_x, stride, h, |i4, r0, r1, r2, r3| {
                let (wy_r, l_r) = (&wy4[i4], &left4[i4]);
                for ((r, &wy_i), &l_i) in [r0, r1, r2, r3].into_iter().zip(wy_r).zip(l_r) {
                    let vert = Simd::splat(wy_i as u16) * a_diff + vbase;
                    let horz = wxv * Simd::splat(l_i.to_u16()) + hbase;
                    let avg = (vert >> one) + (horz >> one) + (vert & horz & one);
                    store_u16(r, avg >> 8);
                }
            });
        } else {
            // wy * a + (256 - wy) * bottom + wx * l + (256 - wx) * right
            // == wy * (a - bottom) + wx * (l - right) + 256 * (bottom + right)
            // where all factors fit in i16 (16 x 16 -> 32 bit products).
            let bias = Simd::<i32, N>::splat(256 * (bottom + right) + 256);
            let wxv = weights::<i16, N>(&wx[x..]).cast::<i32>();
            let a = (load_i16::<T, N>(&above[x..]) - Simd::splat(bottom as i16)).cast::<i32>();
            visit_row4::<T, N>(dst_x, stride, h, |i4, r0, r1, r2, r3| {
                let (wy_r, l_r) = (&wy4[i4], &left4[i4]);
                for ((r, &wy_i), &l_i) in [r0, r1, r2, r3].into_iter().zip(wy_r).zip(l_r) {
                    let wyv = Simd::<i16, N>::splat(wy_i as i16).cast::<i32>();
                    let l = Simd::<i16, N>::splat((l_i.to_i32() - right) as i16).cast::<i32>();
                    store_i32(r, (wyv * a + wxv * l + bias) >> 9);
                }
            });
        }
    }
}

/// `dsp::ipred::ipred_smooth_v`.
#[inline(never)]
pub fn ipred_smooth_v<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    w: usize,
    h: usize,
) {
    if BD == 8 {
        if w == 4 {
            return smooth_hv_packed::<T, 4, 4, true>(dst, stride, above, left, h);
        }
        if cfg!(target_feature = "avx2") && w == 8 {
            return smooth_hv_packed::<T, 8, 2, true>(dst, stride, above, left, h);
        }
    }
    by_width!(w, smooth_v_w::<T, BD>(dst, stride, above, left, h));
}

/// 8-bit [`smooth_v_w`] (`V`) or [`smooth_h_w`] (`!V`) for narrow blocks
/// (`W` 4 or 8), `R = 16 / W` rows per 16-lane vector as in
/// [`smooth_packed`].
#[inline(always)]
fn smooth_hv_packed<T: Pixel, const W: usize, const R: usize, const V: bool>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    h: usize,
) {
    const { assert!(W * R == 16) };
    let (dst, stride, h4) = check_block4::<T, W>(dst, stride, h);
    let h = h4 * 4;
    let wx = &sm_weights(W)[..W];
    let wy = sm_weights(h)[..h].as_chunks::<R>().0;
    let left_c = left[..h].as_chunks::<R>().0;
    let above = &above[..W];
    type U = Simd<u16, 16>;
    // V: wy * (a - bottom) + 256 * bottom + 128; H: wx * (l - right) +
    // 256 * right + 128 (mod 2^16; the true values fit u16).
    let (edge, base) = if V {
        let bottom = left[h - 1].to_u16();
        (
            U::from_array(std::array::from_fn(|j| above[j % W].to_u16())) - U::splat(bottom),
            U::splat(bottom * 256 + 128),
        )
    } else {
        let right = above[W - 1].to_u16();
        (
            U::from_array(std::array::from_fn(|j| wx[j % W] as u16)),
            U::splat(right * 256 + 128),
        )
    };
    let right = U::splat(above[W - 1].to_u16());
    for (g, (wy_r, left_r)) in wy.iter().zip(left_c).enumerate() {
        let i0 = g * R;
        let p = if V {
            let wyv = U::from_array(std::array::from_fn(|j| wy_r[j / W] as u16));
            (wyv * edge + base) >> 8
        } else {
            let lv = U::from_array(std::array::from_fn(|j| left_r[j / W].to_u16()));
            (edge * (lv - right) + base) >> 8
        };
        store_rows_u8::<T, W>(dst, stride, i0, p.cast::<i16>());
    }
}

#[inline(always)]
fn smooth_v_w<T: Pixel, const BD: u32, const W: usize, const N: usize>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    h: usize,
) {
    let (dst, stride, h4) = check_block4::<T, W>(dst, stride, h);
    let h = h4 * 4;
    let wy4 = sm_weights(h)[..h].as_chunks::<4>().0;
    let bottom = left[h - 1].to_i32();
    let above = &above[..W];
    for x in (0..W).step_by(N) {
        let dst_x = &mut dst[x..];
        if BD == 8 {
            // wy * (a - bottom) + 256 * bottom + 128 (mod 2^16) fits in u16.
            let a_diff = load_u16::<T, N>(&above[x..]) - Simd::splat(bottom as u16);
            let base = Simd::<u16, N>::splat(bottom as u16 * 256 + 128);
            visit_row4::<T, N>(dst_x, stride, h, |i4, r0, r1, r2, r3| {
                let wy_r = &wy4[i4];
                for (r, &wy_i) in [r0, r1, r2, r3].into_iter().zip(wy_r) {
                    let wyv = Simd::splat(wy_i as u16);
                    store_u16(r, (wyv * a_diff + base) >> 8);
                }
            });
        } else {
            // wy * (a - bottom) + 256 * bottom with 16 x 16 bit products.
            let a = (load_i16::<T, N>(&above[x..]) - Simd::splat(bottom as i16)).cast::<i32>();
            let bias = Simd::splat(256 * bottom + 128);
            visit_row4::<T, N>(dst_x, stride, h, |i4, r0, r1, r2, r3| {
                let wy_r = &wy4[i4];
                for (r, &wy_i) in [r0, r1, r2, r3].into_iter().zip(wy_r) {
                    let wyv = Simd::<i16, N>::splat(wy_i as i16).cast::<i32>();
                    store_i32(r, (wyv * a + bias) >> 8);
                }
            });
        }
    }
}

/// `dsp::ipred::ipred_smooth_h`.
#[inline(never)]
pub fn ipred_smooth_h<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    w: usize,
    h: usize,
) {
    if BD == 8 {
        if w == 4 {
            return smooth_hv_packed::<T, 4, 4, false>(dst, stride, above, left, h);
        }
        if cfg!(target_feature = "avx2") && w == 8 {
            return smooth_hv_packed::<T, 8, 2, false>(dst, stride, above, left, h);
        }
    }
    by_width!(w, smooth_h_w::<T, BD>(dst, stride, above, left, h));
}

#[inline(always)]
fn smooth_h_w<T: Pixel, const BD: u32, const W: usize, const N: usize>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    h: usize,
) {
    let (dst, stride, h4) = check_block4::<T, W>(dst, stride, h);
    let h = h4 * 4;
    let wx = sm_weights(W);
    let right = above[W - 1].to_i32();
    let left4 = left[..h].as_chunks::<4>().0;
    for x in (0..W).step_by(N) {
        let dst_x = &mut dst[x..];
        if BD == 8 {
            // wx * l + (256 - wx) * right + 128 fits in u16.
            let w = weights::<u16, N>(&wx[x..]);
            let b = (Simd::splat(256) - w) * Simd::splat(right as u16) + Simd::splat(128);
            visit_row4::<T, N>(dst_x, stride, h, |i4, r0, r1, r2, r3| {
                let l_r = &left4[i4];
                for (r, &l_i) in [r0, r1, r2, r3].into_iter().zip(l_r) {
                    let l = Simd::splat(l_i.to_u16());
                    store_u16(r, (w * l + b) >> 8);
                }
            });
        } else {
            // wx * (l - right) + 256 * right with 16 x 16 bit products.
            let w = weights::<i16, N>(&wx[x..]).cast::<i32>();
            let bias = Simd::splat(256 * right + 128);
            visit_row4::<T, N>(dst_x, stride, h, |i4, r0, r1, r2, r3| {
                let l_r = &left4[i4];
                for (r, &l_i) in [r0, r1, r2, r3].into_iter().zip(l_r) {
                    let l = Simd::<i16, N>::splat((l_i.to_i32() - right) as i16).cast::<i32>();
                    store_i32(r, (w * l + bias) >> 8);
                }
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Directional kernels.
// ---------------------------------------------------------------------------

/// Lane type of the directional predictors: `i16` for u8 samples (the
/// 2-tap interpolation `a + (((b - a) * s + 16) >> 5)` fits in 16 bits for
/// 8-bit samples), `u16` (with a 16 x 16 -> 32 bit product split in high
/// and low halves) otherwise.
trait Lane: SimdElement {
    fn load<T: Pixel, const N: usize>(src: &[T]) -> Simd<Self, N>;
    fn store<T: Pixel, const N: usize>(dst: &mut [T], v: Simd<Self, N>);
    fn splat<const N: usize>(v: i32) -> Simd<Self, N>;
    /// `Round2(a * (32 - s) + b * s, 5)` (`interp` of `dsp::ipred`).
    fn interp<const N: usize>(
        a: Simd<Self, N>,
        b: Simd<Self, N>,
        s: Simd<Self, N>,
    ) -> Simd<Self, N>;
    /// Lanes `k < m` from `a`, the others from `b`.
    fn select_lt<const N: usize>(m: usize, a: Simd<Self, N>, b: Simd<Self, N>) -> Simd<Self, N>;
    fn concat4(a: Simd<Self, 4>, b: Simd<Self, 4>) -> Simd<Self, 8>;
}

impl Lane for i16 {
    #[inline(always)]
    fn load<T: Pixel, const N: usize>(src: &[T]) -> Simd<i16, N> {
        load_i16(src)
    }
    #[inline(always)]
    fn store<T: Pixel, const N: usize>(dst: &mut [T], v: Simd<i16, N>) {
        store_i16(dst, v)
    }
    #[inline(always)]
    fn splat<const N: usize>(v: i32) -> Simd<i16, N> {
        Simd::splat(v as i16)
    }
    #[inline(always)]
    fn interp<const N: usize>(a: Simd<i16, N>, b: Simd<i16, N>, s: Simd<i16, N>) -> Simd<i16, N> {
        a + (((b - a) * s + Simd::splat(16)) >> 5)
    }
    #[inline(always)]
    fn select_lt<const N: usize>(m: usize, a: Simd<i16, N>, b: Simd<i16, N>) -> Simd<i16, N> {
        let idx = Simd::<i16, N>::from_array(std::array::from_fn(|k| k as i16));
        idx.simd_lt(Simd::splat(m as i16)).select(a, b)
    }
    #[inline(always)]
    fn concat4(a: Simd<i16, 4>, b: Simd<i16, 4>) -> Simd<i16, 8> {
        simd_swizzle!(a, b, [0, 1, 2, 3, 4, 5, 6, 7])
    }
}

impl Lane for u16 {
    #[inline(always)]
    fn load<T: Pixel, const N: usize>(src: &[T]) -> Simd<u16, N> {
        load_u16(src)
    }
    #[inline(always)]
    fn store<T: Pixel, const N: usize>(dst: &mut [T], v: Simd<u16, N>) {
        store_u16(dst, v)
    }
    #[inline(always)]
    fn splat<const N: usize>(v: i32) -> Simd<u16, N> {
        Simd::splat(v as u16)
    }
    #[inline(always)]
    fn interp<const N: usize>(a: Simd<u16, N>, b: Simd<u16, N>, s: Simd<u16, N>) -> Simd<u16, N> {
        // (b - a) * s needs 18 bits for 12-bit samples: split b - a into
        // 64 * hi + lo with lo in 0..64, then
        // ((b - a) * s + 16) >> 5 == 2 * s * hi + ((s * lo + 16) >> 5)
        // where both products fit in 16 bits.
        let d = b.cast::<i16>() - a.cast::<i16>();
        let s = s.cast::<i16>();
        let hi = (d >> 6) * (s + s);
        let lo = ((d & Simd::splat(63)) * s + Simd::splat(16)) >> 5;
        (a.cast::<i16>() + hi + lo).cast::<u16>()
    }
    #[inline(always)]
    fn select_lt<const N: usize>(m: usize, a: Simd<u16, N>, b: Simd<u16, N>) -> Simd<u16, N> {
        let idx = Simd::<u16, N>::from_array(std::array::from_fn(|k| k as u16));
        idx.simd_lt(Simd::splat(m as u16)).select(a, b)
    }
    #[inline(always)]
    fn concat4(a: Simd<u16, 4>, b: Simd<u16, 4>) -> Simd<u16, 8> {
        simd_swizzle!(a, b, [0, 1, 2, 3, 4, 5, 6, 7])
    }
}

/// `interp(src[k << u], src[(k << u) + 1], s)` for lanes `k = 0..N`,
/// with `u = upsample`. Reads `src[..N + 1]` (`src[..2 * N]` when
/// upsampled).
#[inline(always)]
fn interp_run<T: Pixel, E: Lane, const N: usize>(
    src: &[T],
    upsample: bool,
    s: Simd<E, N>,
) -> Simd<E, N> {
    if upsample {
        let src = &src[..2 * N];
        let (a, b) = E::load::<T, N>(src).deinterleave(E::load::<T, N>(&src[N..]));
        E::interp(a, b, s)
    } else {
        let src = &src[..N + 1];
        E::interp(E::load::<T, N>(src), E::load::<T, N>(&src[1..]), s)
    }
}

/// Whether `Simd::<u8, 16>::swizzle_dyn` lowers to a fast hardware table
/// lookup instruction (NEON `tbl`, WASM `i8x16.swizzle`).
///
/// Not on x86: `swizzle_dyn` picks its implementation with
/// `cfg(target_feature = "ssse3")` inside the standard library, which is
/// evaluated when the (prebuilt, baseline SSE2) standard library is
/// compiled, not in this crate. So even with `-C target-cpu=x86-64-v3`
/// it is scalarized across 16 lanes (16 extracts, loads and inserts),
/// and the `i16` vector kernels (`z1_w`, `z2_k`, `z3_k`) are much faster.
const HAS_FAST_SWIZZLE: bool = cfg!(any(target_arch = "aarch64", target_feature = "simd128"));

/// `dsp::ipred::ipred_z1`.
pub fn ipred_z1<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    w: usize,
    h: usize,
    dx: i32,
    upsample: bool,
) {
    // 8-bit blocks up to 8x16: per lane table lookups (maxBaseX <= 30).
    if HAS_FAST_SWIZZLE
        && is_u8::<T>()
        && w <= 8
        && h <= 16
        && (!upsample || w + h <= 16)
        && above.len() >= 32
    {
        match (w, upsample) {
            (4, false) => z1_tbl::<T, 4, 0>(dst, stride, above, h, dx),
            (4, true) => z1_tbl::<T, 4, 1>(dst, stride, above, h, dx),
            (_, false) => z1_tbl::<T, 8, 0>(dst, stride, above, h, dx),
            (_, true) => z1_tbl::<T, 8, 1>(dst, stride, above, h, dx),
        }
        return;
    }
    ipred_z1_large::<T, BD>(dst, stride, above, w, h, dx, upsample);
}

#[inline(never)]
fn ipred_z1_large<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    w: usize,
    h: usize,
    dx: i32,
    upsample: bool,
) {
    // Separate instances with and without upsampling (z1_w is inlined).
    match (BD <= 10, upsample) {
        (true, false) => by_width!(w, z1_w::<T, i16>(dst, stride, above, h, dx, false)),
        (true, true) => by_width!(w, z1_w::<T, i16>(dst, stride, above, h, dx, true)),
        (false, false) => by_width!(w, z1_w::<T, u16>(dst, stride, above, h, dx, false)),
        (false, true) => by_width!(w, z1_w::<T, u16>(dst, stride, above, h, dx, true)),
    }
}

#[inline(always)]
fn z1_w<T: Pixel, E: Lane, const W: usize, const N: usize>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    h: usize,
    dx: i32,
    upsample: bool,
) {
    let (dst, stride, h4) = check_block4::<T, W>(dst, stride, h);
    let h = h4 * 4;
    let u = upsample as usize;
    let max_base_x = (W + h - 1) << u;
    let edge_val = above[max_base_x];
    let edge_v = E::splat::<N>(edge_val.to_i32());
    let len = if upsample { 2 * N } else { N + 1 };
    let mut padded;
    let above = if W == 64 && above.len() < max_base_x + len {
        padded = [edge_val; MAX_EDGE + 16];
        padded[..MAX_EDGE].copy_from_slice(&above[..MAX_EDGE]);
        &padded[..max_base_x + len]
    } else {
        &above[..max_base_x + len]
    };
    if W <= 8 {
        for i in 0..h {
            let row = row_mut::<T, N>(dst, i * stride);
            let idx = (i as i32 + 1) * dx;
            let base0 = (idx >> (6 - u)) as usize;
            let shift = E::splat::<N>(((idx << u) >> 1) & 0x1f);
            let b = base0.min(max_base_x);
            let m = ((max_base_x as i32 - base0 as i32 + (1 << u) - 1) >> u).clamp(0, N as i32)
                as usize;
            let v = interp_run::<T, E, N>(&above[b..], upsample, shift);
            E::store(row, E::select_lt(m, v, edge_v));
        }
        return;
    }
    let row_span = ((W - N) << u) + len;
    for i4 in 0..h4 {
        let (r0, r1, r2, r3) = row4_mut::<T, W>(dst, i4, stride);
        for (k, row) in [r0, r1, r2, r3].into_iter().enumerate() {
            let i = i4 * 4 + k;
            let idx = (i as i32 + 1) * dx;
            let base0 = (idx >> (6 - u)) as usize;
            if base0 >= max_base_x {
                for d in row.as_chunks_mut::<N>().0 {
                    E::store(d, edge_v);
                }
                continue;
            }
            let shift = E::splat::<N>(((idx << u) >> 1) & 0x1f);
            // Number of columns with base < maxBaseX: only those read the
            // edge (up to above[maxBaseX]), the others are edge_val.
            let n = ((max_base_x - base0 + (1 << u) - 1) >> u).min(W);
            if n == W {
                let win = &above[base0..base0 + row_span];
                for (x, d) in (0..W).step_by(N).zip(row.as_chunks_mut::<N>().0.iter_mut()) {
                    let v = interp_run::<T, E, N>(&win[x << u..], upsample, shift);
                    E::store(d, v);
                }
                continue;
            }
            for (x, d) in (0..W).step_by(N).zip(row.as_chunks_mut::<N>().0.iter_mut()) {
                if x >= n {
                    E::store(d, edge_v);
                    continue;
                }
                let b = (base0 + (x << u)).min(max_base_x);
                let v = interp_run::<T, E, N>(&above[b..], upsample, shift);
                E::store(
                    d,
                    if n < x + N { E::select_lt(n - x, v, edge_v) } else { v },
                );
            }
        }
    }
}

/// Transposes the `K x K` block held in `v` (`v[c][r]` -> `v[r][c]`).
///
/// Each round interleaves vector `k` with vector `k + K / 2`, which
/// rotates the (vector index, lane index) bit string by one bit; after
/// log2(K) rounds rows and columns are swapped.
#[inline(always)]
fn transpose<E: SimdElement, const K: usize>(mut v: [Simd<E, K>; K]) -> [Simd<E, K>; K] {
    let mut n = 1;
    while n < K {
        let old = v;
        for k in 0..K / 2 {
            let (lo, hi) = old[k].interleave(old[k + K / 2]);
            v[2 * k] = lo;
            v[2 * k + 1] = hi;
        }
        n *= 2;
    }
    v
}

/// `dsp::ipred::ipred_z2`.
#[allow(clippy::too_many_arguments)]
pub fn ipred_z2<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    w: usize,
    h: usize,
    dx: i32,
    dy: i32,
    upsample_above: bool,
    upsample_left: bool,
) {
    // 8-bit blocks up to 8x16: per lane table lookups (see z2_tbl).
    // The decoder only upsamples for w + h <= 16.
    if HAS_FAST_SWIZZLE && is_u8::<T>() && w <= 8 && h <= 16 && (!upsample_left || h <= 8) {
        macro_rules! tbl {
            ($ua:literal, $ul:literal) => {
                if w == 4 {
                    z2_tbl::<T, 4, $ua, $ul>(dst, stride, above, left, h, dx, dy)
                } else {
                    z2_tbl::<T, 8, $ua, $ul>(dst, stride, above, left, h, dx, dy)
                }
            };
        }
        match (upsample_above, upsample_left) {
            (false, false) => tbl!(0, 0),
            (true, false) => tbl!(1, 0),
            (false, true) => tbl!(0, 1),
            (true, true) => tbl!(1, 1),
        }
        return;
    }
    // Up to 10-bit samples in u16: 4x4 tiles, two rows per 8-lane vector.
    if use_z2_small16::<T, BD>(w, h, upsample_left) {
        let above: &[T; EDGE_BUF_LEN] = above[..EDGE_BUF_LEN].try_into().unwrap();
        let left: &[T; EDGE_BUF_LEN] = left[..EDGE_BUF_LEN].try_into().unwrap();
        z2_small16::<T, EDGE_BUF_LEN>(
            dst,
            stride,
            above,
            left,
            w,
            h,
            dx,
            dy,
            upsample_above,
            upsample_left,
        );
        return;
    }
    if BD <= 10 && !upsample_above && !upsample_left && w >= 8 && h >= 8 {
        let above: &[T; EDGE_BUF_LEN] = above[..EDGE_BUF_LEN].try_into().unwrap();
        let left: &[T; EDGE_BUF_LEN] = left[..EDGE_BUF_LEN].try_into().unwrap();
        match w {
            8 => z2_big::<T, 8>(dst, stride, above, left, h, dx, dy),
            16 => z2_big::<T, 16>(dst, stride, above, left, h, dx, dy),
            32 => z2_big::<T, 32>(dst, stride, above, left, h, dx, dy),
            _ => z2_big::<T, 64>(dst, stride, above, left, h, dx, dy),
        }
        return;
    }
    ipred_z2_large::<T, BD>(
        dst,
        stride,
        above,
        left,
        w,
        h,
        dx,
        dy,
        upsample_above,
        upsample_left,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn ipred_z2_large<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    w: usize,
    h: usize,
    dx: i32,
    dy: i32,
    upsample_above: bool,
    upsample_left: bool,
) {
    // In AV1 (spec 7.11.2.10), upsampling is only enabled when w + h <= 16
    // (so w <= 8 and h <= 8). Specialize (false, false) across all widths,
    // and specialize upsampled combinations only for w <= 8 (using a single
    // runtime-flag fallback for synthetic test calls with w >= 16).
    macro_rules! go {
        ($e:ty) => {
            if w <= 8 && h <= 16 && (!upsample_left || h <= 8) {
                // Small blocks without table lookups: branch free tiles.
                match (w, h) {
                    (4, _) => z2_small::<T, $e, 4>(
                        dst,
                        stride,
                        above,
                        left,
                        h,
                        dx,
                        dy,
                        upsample_above,
                        upsample_left,
                    ),
                    (_, 4) => z2_8x4::<T, $e>(
                        dst,
                        stride,
                        above,
                        left,
                        dx,
                        dy,
                        upsample_above,
                        upsample_left,
                    ),
                    _ => z2_small::<T, $e, 8>(
                        dst,
                        stride,
                        above,
                        left,
                        h,
                        dx,
                        dy,
                        upsample_above,
                        upsample_left,
                    ),
                }
            } else if !upsample_above && !upsample_left {
                by_width!(
                    w,
                    z2_k::<T, $e>(dst, stride, above, left, h, dx, dy, false, false)
                )
            } else {
                by_width!(
                    w,
                    z2_w_fallback::<T, $e>(
                        dst,
                        stride,
                        above,
                        left,
                        h,
                        dx,
                        dy,
                        upsample_above,
                        upsample_left
                    )
                )
            }
        };
    }
    if BD <= 10 {
        go!(i16)
    } else {
        go!(u16)
    }
}

/// Whether `T` is `u8` (resolved at compile time).
#[inline(always)]
fn is_u8<T: Pixel>() -> bool {
    std::any::TypeId::of::<T>() == std::any::TypeId::of::<u8>()
}

/// Loads 16 samples as `u8` lanes (`T` must be `u8` for exact values).
#[inline(always)]
fn load_u8x16<T: Pixel>(src: &[T]) -> Simd<u8, 16> {
    let src: &[T; 16] = src[..16].try_into().unwrap();
    let any: &dyn Any = src;
    match any.downcast_ref::<[u8; 16]>() {
        Some(s) => Simd::from_array(*s),
        None => Simd::from_array(std::array::from_fn(|i| src[i].to_i32() as u8)),
    }
}

/// `table[idx]` for a 32 entry table held in two vectors (0 for indices
/// past the table, like `tbl`).
#[inline(always)]
fn tbl32(t: &[Simd<u8, 16>; 2], idx: Simd<u8, 16>) -> Simd<u8, 16> {
    t[0].swizzle_dyn(idx) | t[1].swizzle_dyn(idx - Simd::splat(16))
}

/// z2 for 8-bit blocks of width `W` (4 or 8) and height up to 16 with
/// upsampling `UA` / `UL`, 16 / W rows per vector.
///
/// Every lane computes both its above and its left edge prediction
/// with table lookups of the edges (spec indices -2..30, which covers
/// every index the block can use: up to (8 << 1) + 1 when upsampled
/// (h <= 8 if `UL`), 16 + 1 otherwise) and selects one. Lanes that do not use
/// an edge may compute out of range (wrapped) indices into it; the
/// lookups then return arbitrary table entries or 0, which are not
/// selected. Positions (i16): |(i + 1) * dx| <= 16 * 1023 and
/// |(j + 1) * dy| <= 8 * 1023.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn z2_tbl<T: Pixel, const W: usize, const UA: i16, const UL: i16>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    h: usize,
    dx: i32,
    dy: i32,
) {
    const L: usize = 16;
    const { assert!(EDGE_PAD >= 2 && EDGE_PAD + 30 <= EDGE_BUF_LEN && (W == 4 || W == 8)) };
    debug_assert!(h <= 16 && h % (L / W) == 0);
    let (dst, stride, h4) = check_block4::<T, W>(dst, stride, h);
    // Table entry k is spec index k - 2. Pre-bias positions by 2 << (6 - U)
    // so the shifted base indices already include the +2 table offset.
    let ta = [
        load_u8x16::<T>(&above[EDGE_PAD - 2..]),
        load_u8x16::<T>(&above[EDGE_PAD + 14..]),
    ];
    let tl = [
        load_u8x16::<T>(&left[EDGE_PAD - 2..]),
        load_u8x16::<T>(&left[EDGE_PAD + 14..]),
    ];
    let lane = Simd::<i16, L>::from_array(std::array::from_fn(|k| k as i16));
    let j = lane & Simd::splat(W as i16 - 1);
    let r = lane >> Simd::splat(W.trailing_zeros() as i16);
    let mut idx_x =
        (j << 6) - (r + Simd::splat(1)) * Simd::splat(dx as i16) + Simd::splat(2 << (6 - UA));
    let mut idx_y =
        (r << 6) - (j + Simd::splat(1)) * Simd::splat(dy as i16) + Simd::splat(2 << (6 - UL));
    let x_step = Simd::splat((L / W) as i16 * dx as i16);
    let y_step = Simd::splat(((L / W) as i16) << 6);
    // Left positions advance by 64 per row: the shift is constant.
    let shift_y = if UL == 1 { idx_y } else { idx_y >> 1 } & Simd::splat(31);
    for g in 0..(4 / (L / W)) * h4 {
        let i0 = g * (L / W);
        let base_x = idx_x >> (6 - UA);
        let base_y = idx_y >> (6 - UL);
        let shift_x = if UA == 1 { idx_x } else { idx_x >> 1 } & Simd::splat(31);
        let pa = interp_tbl(&ta, base_x, shift_x);
        let pl = interp_tbl(&tl, base_y, shift_y);
        store_rows_u8::<T, W>(
            dst,
            stride,
            i0,
            base_x.simd_ge(Simd::splat(2 - (1 << UA))).select(pa, pl),
        );
        idx_x -= x_step;
        idx_y += y_step;
    }
}

/// `interp(t[idx], t[idx + 1], s)` of 32 entry `u8` tables (indices
/// wrap to `u8`).
#[inline(always)]
fn interp_tbl(t: &[Simd<u8, 16>; 2], idx: Simd<i16, 16>, s: Simd<i16, 16>) -> Simd<i16, 16> {
    let i = idx.cast::<u8>();
    let a = tbl32(t, i).cast::<i16>();
    let b = tbl32(t, i + Simd::splat(1)).cast::<i16>();
    a + (((b - a) * s + Simd::splat(16)) >> 5)
}

#[inline(always)]
fn store_slice_u8<T: Pixel, const W: usize>(row: &mut [T; W], s: &[u8]) {
    let any: &mut dyn Any = row;
    if let Some(d) = any.downcast_mut::<[u8; W]>() {
        d.copy_from_slice(&s[..W]);
    } else if let Some(d) = any.downcast_mut::<[u16; W]>() {
        *d = std::array::from_fn(|k| s[k] as u16);
    } else {
        for (d, &x) in row.iter_mut().zip(&s[..W]) {
            *d = T::from_i32(x as i32);
        }
    }
}

/// Stores the 16 `u8` lanes of `v` (pixel values) to rows `i0..i0 + 16 / W`
/// of `dst` (`W` is 4 or 8, `stride >= W` checked by [`check_block4`]).
#[inline(always)]
fn store_rows_u8<T: Pixel, const W: usize>(
    dst: &mut [T],
    stride: usize,
    i0: usize,
    v: Simd<i16, 16>,
) {
    let v = v.cast::<u8>().to_array();
    if W == 4 {
        let (r0, r1, r2, r3) = row4_mut::<T, W>(dst, i0 >> 2, stride);
        store_slice_u8(r0, &v[..4]);
        store_slice_u8(r1, &v[4..8]);
        store_slice_u8(r2, &v[8..12]);
        store_slice_u8(r3, &v[12..16]);
    } else {
        let base = (i0 & 63) * stride;
        let win = &mut dst[base..base + stride + W];
        let (r0, r1) = win.split_at_mut(stride);
        store_slice_u8::<T, W>((&mut r0[..W]).try_into().unwrap(), &v[..W]);
        store_slice_u8::<T, W>(r1.try_into().unwrap(), &v[W..2 * W]);
    }
}

/// z1 for 8-bit blocks of width `W` (4 or 8) and height up to 16 with
/// upsampling `U` (`maxBaseX = (W + h - 1) << U <= 30`): lane (i, j)
/// interpolates at `base = ((i + 1) * dx >> (6 - U)) + (j << U)` from
/// table lookups of `above[..32]`, or is `above[maxBaseX]` if `base >=
/// maxBaseX` (selected on the i16 position, so wrapped table indices of
/// such lanes do not matter).
#[inline(always)]
fn z1_tbl<T: Pixel, const W: usize, const U: i16>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    h: usize,
    dx: i32,
) {
    const L: usize = 16;
    const { assert!(W == 4 || W == 8) };
    debug_assert!(h <= 16 && h % (L / W) == 0 && ((W + h - 1) << U) <= 30);
    let (dst, stride, h4) = check_block4::<T, W>(dst, stride, h);
    let h = h4 * 4;
    let t = [load_u8x16::<T>(above), load_u8x16::<T>(&above[16..])];
    let max_base_x = ((W + h - 1) << U) as i16;
    let edge = Simd::<i16, L>::splat(above[max_base_x as usize].to_i32() as i16);
    let lane = Simd::<i16, L>::from_array(std::array::from_fn(|k| k as i16));
    let j = (lane & Simd::splat(W as i16 - 1)) << U;
    let r = lane >> Simd::splat(W.trailing_zeros() as i16);
    // Positions (i + 1) * dx <= 16 * 1023 fit in i16.
    let mut idx = (r + Simd::splat(1)) * Simd::splat(dx as i16);
    let step = Simd::splat((L / W) as i16 * dx as i16);
    for g in 0..(4 / (L / W)) * h4 {
        let i0 = g * (L / W);
        let base = (idx >> (6 - U)) + j;
        let shift = if U == 1 { idx } else { idx >> 1 } & Simd::splat(31);
        let v = base
            .simd_ge(Simd::splat(max_base_x))
            .select(edge, interp_tbl(&t, base, shift));
        store_rows_u8::<T, W>(dst, stride, i0, v);
        idx += step;
    }
}

/// z3 for 8-bit blocks of width `W` (4 or 8) and height up to 16 with
/// upsampling `U`, whose reads stay within `left[..32]` (checked by the
/// caller): lane (i, j) interpolates at `base = ((j + 1) * dy >> (6 - U))
/// + (i << U)` with a per column shift, from table lookups.
#[inline(always)]
fn z3_tbl<T: Pixel, const W: usize, const U: i16>(
    dst: &mut [T],
    stride: usize,
    left: &[T],
    h: usize,
    dy: i32,
) {
    const L: usize = 16;
    const { assert!(W == 4 || W == 8) };
    debug_assert!(h <= 16 && h % (L / W) == 0);
    let (dst, stride, h4) = check_block4::<T, W>(dst, stride, h);
    let t = [load_u8x16::<T>(left), load_u8x16::<T>(&left[16..])];
    let lane = Simd::<i16, L>::from_array(std::array::from_fn(|k| k as i16));
    let j = lane & Simd::splat(W as i16 - 1);
    let r = lane >> Simd::splat(W.trailing_zeros() as i16);
    let idx = (j + Simd::splat(1)) * Simd::splat(dy as i16);
    let mut base = (idx >> (6 - U)) + (r << U);
    let step = Simd::splat(((L / W) as i16) << U);
    let shift = if U == 1 { idx } else { idx >> 1 } & Simd::splat(31);
    for g in 0..(4 / (L / W)) * h4 {
        let i0 = g * (L / W);
        store_rows_u8::<T, W>(dst, stride, i0, interp_tbl(&t, base, shift));
        base += step;
    }
}

/// Selects the tile size of [`z2_w`]: 8 when both dimensions allow it.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn z2_k<T: Pixel, E: Lane, const W: usize, const N: usize>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    h: usize,
    dx: i32,
    dy: i32,
    ua: bool,
    ul: bool,
) {
    if W >= 8 && h >= 8 {
        z2_w::<T, E, W, N, 8>(dst, stride, above, left, h, dx, dy, ua, ul);
    } else {
        z2_w::<T, E, W, N, 4>(dst, stride, above, left, h, dx, dy, ua, ul);
    }
}

/// Non-inlined fallback for synthetic tests calling `ipred_z2` with
/// upsampling enabled on blocks with `W >= 16` or `h > 8`.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn z2_w_fallback<T: Pixel, E: Lane, const W: usize, const N: usize>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    h: usize,
    dx: i32,
    dy: i32,
    ua: bool,
    ul: bool,
) {
    z2_w::<T, E, W, N, 4>(dst, stride, above, left, h, dx, dy, ua, ul);
}

/// z2 for blocks of width `K` (4 or 8) and height `K` or `2 * K`
/// (`h <= 16`, `h <= 8` if `ul`), by `K x K` tiles without branches on
/// the left/above boundary: every tile computes the left edge prediction
/// (one vector per tile column, transposed) and the above edge prediction
/// (one vector per tile row) and selects per row.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn z2_small<T: Pixel, E: Lane, const K: usize>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    h: usize,
    dx: i32,
    dy: i32,
    upsample_above: bool,
    upsample_left: bool,
) {
    const { assert!(K <= 8 && EDGE_PAD == 16 && EDGE_PAD + 16 + 2 * 8 <= EDGE_BUF_LEN) };
    debug_assert!(h % K == 0 && h <= 16 && (!upsample_left || h <= 8));
    let (dst, stride, h4) = check_block4::<T, K>(dst, stride, h);
    let h = h4 * 4;
    let above: &[T; EDGE_BUF_LEN] = above[..EDGE_BUF_LEN].try_into().unwrap();
    let left: &[T; EDGE_BUF_LEN] = left[..EDGE_BUF_LEN].try_into().unwrap();
    let ua = upsample_above as i32;
    let ul = upsample_left as i32;
    let pad = EDGE_PAD as i32;
    let b0: [usize; K] = std::array::from_fn(|k| {
        let idx = -(k as i32 + 1) * dy;
        ((idx >> (6 - ul)).max(-pad) + pad) as usize
    });
    let sl: [Simd<E, K>; K] = std::array::from_fn(|k| {
        let idx = -(k as i32 + 1) * dy;
        E::splat(((idx << ul) >> 1) & 0x1f)
    });
    for r in (0..h / K).map(|t| t * K) {
        let cols: [Simd<E, K>; K] = std::array::from_fn(|k| {
            interp_run::<T, E, K>(&left[b0[k] + (r << ul)..], upsample_left, sl[k])
        });
        let lt_rows = transpose(cols);
        for c4 in 0..K / 4 {
            let (r0, r1, r2, r3) = row4_mut::<T, K>(dst, (r >> 2) + c4, stride);
            for (k4, row) in [r0, r1, r2, r3].into_iter().enumerate() {
                let k = c4 * 4 + k4;
                let i = r + k;
                let t = (i as i32 + 1) * dx;
                // Columns j < j0 = ceil((t - 64) / 64) use the left edge.
                let m = (((t - 1) >> 6).clamp(0, K as i32)) as usize;
                let idx = -t;
                let b = ((idx >> (6 - ua)).max(-pad) + pad) as usize;
                let at = interp_run::<T, E, K>(
                    &above[b..],
                    upsample_above,
                    E::splat(((idx << ua) >> 1) & 0x1f),
                );
                E::store(row, E::select_lt(m, lt_rows[k], at));
            }
        }
    }
}

/// Buffer index of the spec edge index `b` for the run loads of the z2
/// tile kernels (wrapped to the buffer by [`z2_pair4`]). Every run read
/// by a selected lane starts at a spec index in `-EDGE_PAD..32`; runs
/// starting below `-EDGE_PAD` have no selected lane (see [`z2_tiles`]),
/// so it does not matter which (in bounds) samples they load.
#[inline(always)]
fn z2_pos(b: i32) -> usize {
    (b + EDGE_PAD as i32) as usize
}

/// Edge buffer length of the small z2 blocks of [`predict_directional`]
/// (samples up to 10 bits, `w <= 8`, `h <= 16`): spec indices
/// `-EDGE_PAD..24`, enough for [`fill_edge`] / [`upsample_edge8`] of
/// `w <= 8`, `h <= 16` samples and the wrapped run loads of [`z2_pair4`].
const Z2_SMALL_LEN: usize = 32 + 8;

/// The interpolation inputs of 4 consecutive output lanes whose first
/// sample is `buf[pos]`: `(buf[pos + k], buf[pos + k + 1])`, or with
/// upsampling (`U`) `(buf[pos + 2k], buf[pos + 2k + 1])`. `pos` is
/// wrapped to `0..32`, which leaves every run read by a selected lane
/// (`pos` in `8..=27`) unchanged.
#[inline(always)]
fn z2_pair4<T: Pixel, const U: bool, const L: usize>(
    buf: &[T; L],
    pos: usize,
) -> (Simd<i16, 4>, Simd<i16, 4>) {
    const { assert!(L >= 32 + 8) };
    let pos = pos & 31;
    if U {
        let v = load_i16::<T, 8>(&buf[pos..]);
        (
            simd_swizzle!(v, [0, 2, 4, 6]),
            simd_swizzle!(v, [1, 3, 5, 7]),
        )
    } else {
        (
            load_i16::<T, 4>(&buf[pos..]),
            load_i16::<T, 4>(&buf[pos + 1..]),
        )
    }
}

/// `Round2(a * (32 - s) + b * s, 5)` on `i16` lanes (exact for samples
/// of up to 10 bits: `|b - a| * s <= 1023 * 31`).
#[inline(always)]
fn interp_i16x8(a: i16x8, b: i16x8, s: i16x8) -> i16x8 {
    a + (((b - a) * s + Simd::splat(16)) >> 5)
}

/// Interpolates two pairs of 4-lane runs (rows or columns `k`, `k + 1`).
#[inline(always)]
fn interp_runs(
    p: (Simd<i16, 4>, Simd<i16, 4>),
    q: (Simd<i16, 4>, Simd<i16, 4>),
    s: i16x8,
) -> i16x8 {
    interp_i16x8(
        simd_swizzle!(p.0, q.0, [0, 1, 2, 3, 4, 5, 6, 7]),
        simd_swizzle!(p.1, q.1, [0, 1, 2, 3, 4, 5, 6, 7]),
        s,
    )
}

/// Inputs of one 4x4 tile of [`z2_tiles`]: buffer positions of the above
/// runs of its 4 rows, `idx = (j << 6) - (i + 1) * dx` of its row pairs,
/// tile row/column offset, and whether any lane uses the above / left edge.
struct Z2Tile {
    pa: [usize; 4],
    idx: [i16x8; 2],
    ru: i32,
    c: usize,
    need_a: bool,
    need_l: bool,
}

/// Transposes 4 columns `(c02, c13)` of 4 `i16` rows into row pairs
/// `(l01, l23)` using two 16-bit interleaves and two 32-bit interleaves.
#[inline(always)]
fn z2_transpose4(c02: i16x8, c13: i16x8) -> (i16x8, i16x8) {
    let (lo, hi) = c02.interleave(c13);
    let (r01, r23) = Simd::<i32, 4>::from_ne_bytes(lo.to_ne_bytes())
        .interleave(Simd::<i32, 4>::from_ne_bytes(hi.to_ne_bytes()));
    (
        i16x8::from_ne_bytes(r01.to_ne_bytes()),
        i16x8::from_ne_bytes(r23.to_ne_bytes()),
    )
}

/// One 4x4 tile of [`z2_tiles`]: rows `(0, 1)` and `(2, 3)` of the tile.
#[inline(always)]
fn z2_tile<T: Pixel, const UA: bool, const UL: bool, const L: usize>(
    above: &[T; L],
    left: &[T; L],
    dy: i32,
    dyv: i16x8,
    t: &Z2Tile,
) -> [i16x8; 2] {
    let a = if t.need_a {
        let run = |k: usize| z2_pair4::<T, UA, L>(above, t.pa[k]);
        let ua = i16x8::splat(UA as i16);
        let m = i16x8::splat(31);
        let sa0 = ((t.idx[0] << ua) >> 1) & m;
        let sa1 = ((t.idx[1] << ua) >> 1) & m;
        [
            interp_runs(run(0), run(1), sa0),
            interp_runs(run(2), run(3), sa1),
        ]
    } else {
        [i16x8::splat(0); 2]
    };
    if !t.need_l {
        return a;
    }
    let ul = UL as i32;
    let c = t.c as i32;
    let run = |k: i32| z2_pair4::<T, UL, L>(left, z2_pos(t.ru + ((-(c + k) * dy) >> (6 - ul))));
    let u02 = (if t.c == 0 { Z2_COL_02 } else { Z2_COL_46 }) * dyv;
    let u13 = u02 + dyv;
    let c02 = interp_runs(run(1), run(3), z2_shift(u02, UL));
    let c13 = interp_runs(run(2), run(4), z2_shift(u13, UL));
    let (l01, l23) = z2_transpose4(c02, c13);
    if !t.need_a {
        return [l01, l23];
    }
    let m = i16x8::splat(-64);
    [
        t.idx[0].simd_ge(m).select(a[0], l01),
        t.idx[1].simd_ge(m).select(a[1], l23),
    ]
}

/// Lane `k` of a row pair vector is column `k & 3` of row `k >> 2`:
/// `(k & 3) << 6` and `(k >> 2) + 1`.
const Z2_LANE_J64: i16x8 = i16x8::from_array([0, 64, 128, 192, 0, 64, 128, 192]);
const Z2_LANE_I1: i16x8 = i16x8::from_array([1, 1, 1, 1, 2, 2, 2, 2]);
const Z2_COL_02: i16x8 = i16x8::from_array([1, 1, 1, 1, 3, 3, 3, 3]);
const Z2_COL_46: i16x8 = i16x8::from_array([5, 5, 5, 5, 7, 7, 7, 7]);

/// `((-t << u) >> 1) & 31`: the interpolation shift at position `-t`
/// (plus any multiple of 64).
#[inline(always)]
fn z2_shift(t: i16x8, u: bool) -> i16x8 {
    ((-t << i16x8::splat(u as i16)) >> 1) & i16x8::splat(31)
}

/// Stores the row pair vectors `v` to 4 rows of 4 samples.
#[inline(always)]
fn z2_store4<T: Pixel>(rows: [&mut [T]; 4], v: [i16x8; 2]) {
    let [r0, r1, r2, r3] = rows;
    store_i16::<T, 4>(r0, simd_swizzle!(v[0], [0, 1, 2, 3]));
    store_i16::<T, 4>(r1, simd_swizzle!(v[0], [4, 5, 6, 7]));
    store_i16::<T, 4>(r2, simd_swizzle!(v[1], [0, 1, 2, 3]));
    store_i16::<T, 4>(r3, simd_swizzle!(v[1], [4, 5, 6, 7]));
}

/// Table of up to 24 `u16` edge samples in 3 byte vectors for
/// [`z2_tbl16`]: spec indices from `-1` (from `-2` when upsampled, `U`).
/// The vectors are built from loads at the addresses of the 16 byte
/// stores of [`fill_edge`] / [`upsample_edge`] (and a 1 sample load of the
/// corner), so that store to load forwarding works right after the edge
/// preparation. Only the first `n` vectors are loaded.
#[inline(always)]
fn z2_tbl_load<T: Pixel, const U: bool, const L: usize>(buf: &[T; L], n: usize) -> [u8x16; 3] {
    let base = if U { EDGE_PAD - 1 } else { EDGE_PAD };
    let corner = u16x8::splat(buf[base - 1].to_u16());
    let q0 = load_u16::<T, 8>(&buf[base..]);
    let q1 = if n > 1 { load_u16::<T, 8>(&buf[base + 8..]) } else { u16x8::splat(0) };
    let q2 = if n > 2 { load_u16::<T, 8>(&buf[base + 16..]) } else { u16x8::splat(0) };
    let ext = |a: u16x8, b: u16x8| -> u8x16 {
        simd_swizzle!(a, b, [7, 8, 9, 10, 11, 12, 13, 14]).to_ne_bytes()
    };
    [ext(corner, q0), ext(q0, q1), ext(q1, q2)]
}

/// `(t[e], t[e + 1])` for the `u16` table `t` of [`z2_tbl_load`] (its
/// first `n` vectors) and every lane's element index `e`. Lanes whose
/// index is outside the table get arbitrary values.
#[inline(always)]
fn z2_tbl_lookup(t: &[u8x16; 3], n: usize, e: i16x8) -> (i16x8, i16x8) {
    // Byte indices 2e and 2e + 1 of every u16 lane (little endian).
    let bi = (e * i16x8::splat(514) + i16x8::splat(256)).cast::<u16>();
    let look = |b: u16x8| {
        let b = b.to_ne_bytes();
        let mut v = t[0].swizzle_dyn(b);
        if n > 1 {
            v |= t[1].swizzle_dyn(b - u8x16::splat(16));
        }
        if n > 2 {
            v |= t[2].swizzle_dyn(b - u8x16::splat(32));
        }
        i16x8::from_ne_bytes(v)
    };
    (look(bi), look(bi + u16x8::splat(0x0202)))
}

/// z2 for samples of up to 10 bits and blocks of `W x H` (`W` 4 or 8,
/// `H` 4, 8 or 16, at most 8 when `UL`) with per lane table lookups
/// (NEON `tbl`) of the edges held in registers, 8 lanes (two rows of 4 or
/// one row of 8) at a time. Every lane computes its above and its left
/// prediction and selects one (`idx >= -64`, see [`z2_tiles`]). The
/// tables cover spec indices `-1..23` (`-2..22` upsampled), which
/// includes every index a selected lane reads (up to `2 * W - 1` above
/// with upsampling, `H` on the left).
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn z2_tbl16<
    T: Pixel,
    const W: usize,
    const H: usize,
    const UA: bool,
    const UL: bool,
    const L: usize,
>(
    dst: &mut [T],
    stride: usize,
    above: &[T; L],
    left: &[T; L],
    dx: i32,
    dy: i32,
) {
    const { assert!((W == 4 || W == 8) && (H == 4 || H == 8 || H == 16) && (!UL || H <= 8)) };
    let (dst, stride, _) = check_block4::<T, W>(dst, stride, H);
    // Table vectors needed: above indices up to W (2 * W - 1 + 1 when
    // upsampled) from -1 (-2), left indices up to H (2 * H) likewise.
    let need_a = ((W - 1) * 64) as i32 - dx >= -64;
    let need_l = -((H as i32) * dx) < -64;
    let na = if UA {
        if W == 4 {
            2
        } else {
            3
        }
    } else if W == 4 {
        1
    } else {
        2
    };
    let nl = if UL {
        if H == 4 {
            2
        } else {
            3
        }
    } else if H == 4 {
        1
    } else if H == 8 {
        2
    } else {
        3
    };
    let ta = if need_a { z2_tbl_load::<T, UA, L>(above, na) } else { [u8x16::splat(0); 3] };
    let tl = if need_l { z2_tbl_load::<T, UL, L>(left, nl) } else { [u8x16::splat(0); 3] };
    let lane = i16x8::from_array([0, 1, 2, 3, 4, 5, 6, 7]);
    let j = lane & i16x8::splat(W as i16 - 1);
    let r = lane >> i16x8::splat(W.trailing_zeros() as i16);
    let (ua, ul) = (UA as i16, UL as i16);
    let rows = 8 / W;
    let mut idx_x = (j << 6) - (r + i16x8::splat(1)) * i16x8::splat(dx as i16);
    let idx_y = (r << 6) - (j + i16x8::splat(1)) * i16x8::splat(dy as i16);
    // The left shift does not depend on the row, and by advances by rows << ul.
    let sy = ((idx_y << i16x8::splat(ul)) >> 1) & i16x8::splat(31);
    let mut by = (idx_y >> i16x8::splat(6 - ul)) + i16x8::splat(1 + ul);
    let step_x = i16x8::splat((rows as i32 * dx) as i16);
    let step_by = i16x8::splat((rows as i16) << ul);
    for g in 0..H / rows {
        let i = g * rows;
        let g_need_a = ((W - 1) * 64) as i32 - (i as i32 + 1) * dx >= -64;
        let g_need_l = -(((i + rows) as i32) * dx) < -64;
        let v = if !g_need_l {
            let bx = (idx_x >> i16x8::splat(6 - ua)) + i16x8::splat(1 + ua);
            let sx = ((idx_x << i16x8::splat(ua)) >> 1) & i16x8::splat(31);
            let (a0, a1) = z2_tbl_lookup(&ta, na, bx);
            interp_i16x8(a0, a1, sx)
        } else if !g_need_a {
            let (l0, l1) = z2_tbl_lookup(&tl, nl, by);
            interp_i16x8(l0, l1, sy)
        } else {
            let bx = (idx_x >> i16x8::splat(6 - ua)) + i16x8::splat(1 + ua);
            let sx = ((idx_x << i16x8::splat(ua)) >> 1) & i16x8::splat(31);
            let (a0, a1) = z2_tbl_lookup(&ta, na, bx);
            let (l0, l1) = z2_tbl_lookup(&tl, nl, by);
            idx_x
                .simd_ge(i16x8::splat(-64))
                .select(interp_i16x8(a0, a1, sx), interp_i16x8(l0, l1, sy))
        };
        if W == 4 {
            let base = (i & 15) * stride;
            let win = &mut dst[base..base + stride + 4];
            let (r0, r1) = win.split_at_mut(stride);
            store_i16::<T, 4>(r0, simd_swizzle!(v, [0, 1, 2, 3]));
            store_i16::<T, 4>(r1, simd_swizzle!(v, [4, 5, 6, 7]));
        } else {
            store_i16::<T, 8>(&mut dst[(i & 15) * stride..], v);
        }
        idx_x -= step_x;
        by += step_by;
    }
}

/// Register-edge specialization of [`z2_tbl16`] for non-upsampled blocks
/// with `W <= 8` and `H <= 8`: filters the edges and builds the NEON `tbl`
/// tables directly in registers (see [`z2_small_edges`] for `packed`).
#[inline(never)]
fn z2_tbl16_reg<T: Pixel, const W: usize, const H: usize>(
    dst: &mut [T],
    stride: usize,
    edges: &IntraEdges<T>,
    packed: u64,
) {
    const { assert!((W == 4 || W == 8) && (H == 4 || H == 8)) };
    let (c, va, vl, dx, dy) = z2_small_edges::<T, W, H>(edges, packed);
    let (ca, cl) = (c, c);
    let (dst, stride, _) = check_block4::<T, W>(dst, stride, H);
    let ext = |a: u16x8, b: u16x8| -> u8x16 {
        simd_swizzle!(a, b, [7, 8, 9, 10, 11, 12, 13, 14]).to_ne_bytes()
    };
    let na = if W == 4 { 1 } else { 2 };
    let nl = if H == 4 { 1 } else { 2 };
    let ta = [
        ext(ca, va),
        if W == 8 { ext(va, ca) } else { u8x16::splat(0) },
        u8x16::splat(0),
    ];
    let tl = [
        ext(cl, vl),
        if H == 8 { ext(vl, cl) } else { u8x16::splat(0) },
        u8x16::splat(0),
    ];
    let lane = i16x8::from_array([0, 1, 2, 3, 4, 5, 6, 7]);
    let j = lane & i16x8::splat(W as i16 - 1);
    let r = lane >> i16x8::splat(W.trailing_zeros() as i16);
    let rows = 8 / W;
    let mut idx_x = (j << 6) - (r + i16x8::splat(1)) * i16x8::splat(dx as i16);
    let idx_y = (r << 6) - (j + i16x8::splat(1)) * i16x8::splat(dy as i16);
    let sy = (idx_y >> 1) & i16x8::splat(31);
    let mut by = (idx_y >> 6) + i16x8::splat(1);
    let step_x = i16x8::splat((rows as i32 * dx) as i16);
    let step_by = i16x8::splat(rows as i16);
    for g in 0..H / rows {
        let i = g * rows;
        let g_need_a = ((W - 1) * 64) as i32 - (i as i32 + 1) * dx >= -64;
        let g_need_l = -(((i + rows) as i32) * dx) < -64;
        let v = if !g_need_l {
            let bx = (idx_x >> 6) + i16x8::splat(1);
            let sx = (idx_x >> 1) & i16x8::splat(31);
            let (a0, a1) = z2_tbl_lookup(&ta, na, bx);
            interp_i16x8(a0, a1, sx)
        } else if !g_need_a {
            let (l0, l1) = z2_tbl_lookup(&tl, nl, by);
            interp_i16x8(l0, l1, sy)
        } else {
            let bx = (idx_x >> 6) + i16x8::splat(1);
            let sx = (idx_x >> 1) & i16x8::splat(31);
            let (a0, a1) = z2_tbl_lookup(&ta, na, bx);
            let (l0, l1) = z2_tbl_lookup(&tl, nl, by);
            idx_x
                .simd_ge(i16x8::splat(-64))
                .select(interp_i16x8(a0, a1, sx), interp_i16x8(l0, l1, sy))
        };
        if W == 4 {
            let base = (i & 15) * stride;
            let win = &mut dst[base..base + stride + 4];
            let (r0, r1) = win.split_at_mut(stride);
            store_i16::<T, 4>(r0, simd_swizzle!(v, [0, 1, 2, 3]));
            store_i16::<T, 4>(r1, simd_swizzle!(v, [4, 5, 6, 7]));
        } else {
            store_i16::<T, 8>(&mut dst[(i & 15) * stride..], v);
        }
        idx_x -= step_x;
        by += step_by;
    }
}

/// Pair of 4-lane runs from `buf[0..16]` (`buf[0..8]` = corner,
/// `buf[8..16]` = edge[0..8]) at `(pos & 7) + OFF` (`OFF` is `0` or `4`).
#[inline(always)]
fn z2_pair4_reg<const OFF: usize>(buf: &[i16; 16], pos: usize) -> (Simd<i16, 4>, Simd<i16, 4>) {
    let p = (pos & 7) + OFF;
    (
        Simd::from_array(buf[p..p + 4].try_into().unwrap()),
        Simd::from_array(buf[p + 1..p + 5].try_into().unwrap()),
    )
}

/// Left-edge prediction `(l01, l23)` of a 4x4 tile at tile row `R` (`0` or
/// `4`) and tile column `C` (`0` or `4`).
#[inline(always)]
fn z2_left4_reg<const R: usize, const C: usize>(
    lbuf: &[i16; 16],
    dy: i32,
    dyv: i16x8,
) -> [i16x8; 2] {
    let c = C as i32;
    let run = |k: i32| z2_pair4_reg::<R>(lbuf, ((-(c + k) * dy) >> 6) as usize);
    let u02 = (if C == 0 { Z2_COL_02 } else { Z2_COL_46 }) * dyv;
    let u13 = u02 + dyv;
    let m = i16x8::splat(31);
    let c02 = interp_runs(run(1), run(3), (-u02 >> 1) & m);
    let c13 = interp_runs(run(2), run(4), (-u13 >> 1) & m);
    let (l01, l23) = z2_transpose4(c02, c13);
    [l01, l23]
}

/// One 4-row strip (`R` is `0` or `4`) of [`z2_reg_x86`]: shares the
/// above-edge row indices `pa` and shifts `sa` across `C = 0` and `C = 4`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn z2_strip_x86<T: Pixel, const W: usize, const R: usize>(
    rows: (&mut [T; W], &mut [T; W], &mut [T; W], &mut [T; W]),
    abuf: &[i16; 16],
    lbuf: &[i16; 16],
    dx: i32,
    dy: i32,
    dyv: i16x8,
    t01: i16x8,
    t23: i16x8,
) {
    let r = R as i32;
    let (r0, r1, r2, r3) = rows;
    let idx0 = [Z2_LANE_J64 - t01, Z2_LANE_J64 - t23];
    let pos = |k: i32| ((-(r + k) * dx) >> 6) as usize;
    let m31 = i16x8::splat(31);
    let th = i16x8::splat(-64);
    let need_a0 = 3 * 64 - (r + 1) * dx >= -64;
    let need_l0 = -(r + 4) * dx < -64;
    if W == 4 {
        let v0 = if !need_l0 {
            let run = |p: usize| z2_pair4_reg::<0>(abuf, p);
            [
                interp_runs(run(pos(1)), run(pos(2)), (idx0[0] >> 1) & m31),
                interp_runs(run(pos(3)), run(pos(4)), (idx0[1] >> 1) & m31),
            ]
        } else if !need_a0 {
            z2_left4_reg::<R, 0>(lbuf, dy, dyv)
        } else {
            let run = |p: usize| z2_pair4_reg::<0>(abuf, p);
            let a0 = [
                interp_runs(run(pos(1)), run(pos(2)), (idx0[0] >> 1) & m31),
                interp_runs(run(pos(3)), run(pos(4)), (idx0[1] >> 1) & m31),
            ];
            let l0 = z2_left4_reg::<R, 0>(lbuf, dy, dyv);
            [
                idx0[0].simd_ge(th).select(a0[0], l0[0]),
                idx0[1].simd_ge(th).select(a0[1], l0[1]),
            ]
        };
        z2_store4::<T>([r0, r1, r2, r3], v0);
        return;
    }
    let need_a1 = 7 * 64 - (r + 1) * dx >= -64;
    let need_l1 = 4 * 64 - (r + 4) * dx < -64;
    let j4 = Z2_LANE_J64 + i16x8::splat(256);
    let idx1 = [j4 - t01, j4 - t23];
    let (pa, sa) = if need_a1 {
        (
            [pos(1), pos(2), pos(3), pos(4)],
            [(idx0[0] >> 1) & m31, (idx0[1] >> 1) & m31],
        )
    } else {
        ([0; 4], [i16x8::splat(0); 2])
    };
    let v0 = if !need_l0 {
        let run = |p: usize| z2_pair4_reg::<0>(abuf, p);
        [
            interp_runs(run(pa[0]), run(pa[1]), sa[0]),
            interp_runs(run(pa[2]), run(pa[3]), sa[1]),
        ]
    } else if !need_a0 {
        z2_left4_reg::<R, 0>(lbuf, dy, dyv)
    } else {
        let run = |p: usize| z2_pair4_reg::<0>(abuf, p);
        let a0 = [
            interp_runs(run(pa[0]), run(pa[1]), sa[0]),
            interp_runs(run(pa[2]), run(pa[3]), sa[1]),
        ];
        let l0 = z2_left4_reg::<R, 0>(lbuf, dy, dyv);
        [
            idx0[0].simd_ge(th).select(a0[0], l0[0]),
            idx0[1].simd_ge(th).select(a0[1], l0[1]),
        ]
    };
    let v1 = if !need_l1 {
        let run = |p: usize| z2_pair4_reg::<4>(abuf, p);
        [
            interp_runs(run(pa[0]), run(pa[1]), sa[0]),
            interp_runs(run(pa[2]), run(pa[3]), sa[1]),
        ]
    } else if !need_a1 {
        z2_left4_reg::<R, 4>(lbuf, dy, dyv)
    } else {
        let run = |p: usize| z2_pair4_reg::<4>(abuf, p);
        let a1 = [
            interp_runs(run(pa[0]), run(pa[1]), sa[0]),
            interp_runs(run(pa[2]), run(pa[3]), sa[1]),
        ];
        let l1 = z2_left4_reg::<R, 4>(lbuf, dy, dyv);
        [
            idx1[0].simd_ge(th).select(a1[0], l1[0]),
            idx1[1].simd_ge(th).select(a1[1], l1[1]),
        ]
    };
    store_i16::<T, 8>(r0, simd_swizzle!(v0[0], v1[0], [0, 1, 2, 3, 8, 9, 10, 11]));
    store_i16::<T, 8>(
        r1,
        simd_swizzle!(v0[0], v1[0], [4, 5, 6, 7, 12, 13, 14, 15]),
    );
    store_i16::<T, 8>(r2, simd_swizzle!(v0[1], v1[1], [0, 1, 2, 3, 8, 9, 10, 11]));
    store_i16::<T, 8>(
        r3,
        simd_swizzle!(v0[1], v1[1], [4, 5, 6, 7, 12, 13, 14, 15]),
    );
}

/// Specialization of [`z2_4x4`] / [`z2_tiles`] for non-upsampled blocks
/// with `W` in `{4, 8}` and `H` in `{4, 8}`, filtering the edges itself
/// (see [`z2_small_edges`] for `packed`). All 4 arguments are passed in
/// integer registers.
#[inline(never)]
fn z2_reg_x86<T: Pixel, const W: usize, const H: usize>(
    dst: &mut [T],
    stride: usize,
    edges: &IntraEdges<T>,
    packed: u64,
) {
    const { assert!((W == 4 || W == 8) && (H == 4 || H == 8)) };
    let (c, va, vl, dx, dy) = z2_small_edges::<T, W, H>(edges, packed);
    let abuf: [i16; 16] = simd_swizzle!(
        c,
        va,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    )
    .cast::<i16>()
    .to_array();
    let lbuf: [i16; 16] = simd_swizzle!(
        c,
        vl,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    )
    .cast::<i16>()
    .to_array();
    let (abuf, lbuf) = (&abuf, &lbuf);
    let (dst, stride, _) = check_block4::<T, W>(dst, stride, H);
    let (dxv, dyv) = (i16x8::splat(dx as i16), i16x8::splat(dy as i16));
    let t01 = Z2_LANE_I1 * dxv;
    let t23 = t01 + dxv + dxv;
    z2_strip_x86::<T, W, 0>(
        row4_mut::<T, W>(dst, 0, stride),
        abuf,
        lbuf,
        dx,
        dy,
        dyv,
        t01,
        t23,
    );
    if H == 8 {
        let t45 = t23 + dxv + dxv;
        let t67 = t45 + dxv + dxv;
        z2_strip_x86::<T, W, 4>(
            row4_mut::<T, W>(dst, 1, stride),
            abuf,
            lbuf,
            dx,
            dy,
            dyv,
            t45,
            t67,
        );
    }
}

/// Packs the scalar parameters of a small register z2 block for
/// [`z2_small_edges`]: `dx`, `dy` (at most 1023), the edge filter sizes
/// `sz_a`, `sz_l` (at most 9) and strengths `s_a`, `s_l` (at most 3).
#[inline(always)]
fn z2_small_pack(dx: i32, dy: i32, sz_a: i32, sz_l: i32, s_a: u32, s_l: u32) -> u64 {
    (dx as u64 & 0xffff)
        | ((dy as u64 & 0xffff) << 16)
        | ((sz_a as u64 & 15) << 32)
        | ((sz_l as u64 & 15) << 36)
        | ((s_a as u64 & 3) << 40)
        | ((s_l as u64 & 3) << 42)
}

/// Loads and filters the edges of a non-upsampled `W x H` z2 block (`W`,
/// `H` 4 or 8; see [`z2_small_pack`]): returns the splatted corner, the
/// above and left edges (zero if not read by the block) and `dx`, `dy`.
#[inline(always)]
fn z2_small_edges<T: Pixel, const W: usize, const H: usize>(
    edges: &IntraEdges<T>,
    packed: u64,
) -> (u16x8, u16x8, u16x8, i32, i32) {
    let dx = (packed & 0xffff) as i32;
    let dy = ((packed >> 16) & 0xffff) as i32;
    let (sz_a, sz_l) = (((packed >> 32) & 15) as i32, ((packed >> 36) & 15) as i32);
    let (s_a, s_l) = (((packed >> 40) & 3) as u32, ((packed >> 42) & 3) as u32);
    // Lane (i, j) selects the above edge iff (j << 6) - (i + 1) * dx >= -64.
    let need_a = dx <= (W as i32) * 64;
    let need_l = (H as i32) * dx > 64;
    // Edges of 4 samples are loaded as 4 lanes: the caller stores the
    // replicated samples past them separately, and an 8-lane load
    // spanning both stores would stall on store forwarding.
    let c = u16x8::splat(edges.top_left.to_u16());
    let va = if need_a {
        filter_edge8_s12(&edges.above, c, sz_a, s_a, W == 4)
    } else {
        u16x8::splat(0)
    };
    let vl = if need_l {
        filter_edge8_s12(&edges.left, c, sz_l, s_l, H == 4)
    } else {
        u16x8::splat(0)
    };
    (c, va, vl, dx, dy)
}

/// `[w1, w2]` of the edge filter kernels `[0, w1, w2, w1, 0]` of strengths
/// 1 and 2 (entry 0 is the identity, entry 3 is unused).
static EDGE_W12: [[u16x8; 2]; 4] = {
    const fn s(w: u16) -> u16x8 {
        u16x8::from_array([w; 8])
    }
    [[s(0), s(16)], [s(4), s(8)], [s(5), s(6)], [s(5), s(6)]]
};

/// `LANE_LT[k]` lane `l` is all ones iff `l < k`.
static LANE_LT: [u16x8; 16] = {
    let mut t = [u16x8::from_array([0; 8]); 16];
    let mut k = 0;
    while k < 16 {
        let mut a = [0u16; 8];
        let mut l = 0;
        while l < 8 {
            if l < k {
                a[l] = u16::MAX;
            }
            l += 1;
        }
        t[k] = u16x8::from_array(a);
        k += 1;
    }
    t
};

/// [`filter_edge8_vec`] for the strengths of blocks with `w + h <= 16`
/// (at most 2, see `STRENGTH_LUT`), whose kernels have no outer taps:
/// one mask load and two neighbour vectors instead of four.
#[inline(always)]
fn filter_edge8_s12<T: Pixel>(
    src: &[T; MAX_EDGE],
    prev: u16x8,
    sz: i32,
    strength: u32,
    four: bool,
) -> u16x8 {
    const M: usize = 8;
    debug_assert!(sz <= M as i32 + 1 && (!four || sz <= 5) && strength <= 2);
    let raw = if four { load_u16::<T, 4>(src).resize::<M>(0) } else { load_u16::<T, M>(src) };
    if strength == 0 || sz <= 1 {
        return raw;
    }
    // Lanes below sz - 1 are filtered; the inputs from there on repeat the
    // last sample.
    let m = LANE_LT[(sz - 1) as usize & 15];
    let last = u16x8::splat(src[(sz as usize - 2) & 7].to_u16());
    let cur = (raw & m) | (last & !m);
    let t1 = simd_swizzle!(prev, cur, [7, 8, 9, 10, 11, 12, 13, 14]);
    let t3 = simd_swizzle!(cur, last, [1, 2, 3, 4, 5, 6, 7, 8]);
    let [w1, w2] = EDGE_W12[strength as usize & 3];
    let f = ((t1 + t3) * w1 + cur * w2 + u16x8::splat(8)) >> 4;
    (f & m) | (raw & !m)
}

/// Dispatches a non-upsampled `w <= 8, h <= 8` 10-bit z2 block to
/// [`z2_tbl16_reg`] (NEON) or [`z2_reg_x86`] (x86), which load and filter
/// the edges themselves (see [`z2_small_pack`] for `packed`).
#[inline(always)]
fn z2_small16_reg<T: Pixel>(
    dst: &mut [T],
    stride: usize,
    edges: &IntraEdges<T>,
    w: usize,
    h: usize,
    packed: u64,
) {
    if HAS_FAST_SWIZZLE {
        match (w, h) {
            (4, 4) => z2_tbl16_reg::<T, 4, 4>(dst, stride, edges, packed),
            (4, _) => z2_tbl16_reg::<T, 4, 8>(dst, stride, edges, packed),
            (_, 4) => z2_tbl16_reg::<T, 8, 4>(dst, stride, edges, packed),
            _ => z2_tbl16_reg::<T, 8, 8>(dst, stride, edges, packed),
        }
    } else {
        match (w, h) {
            (4, 4) => z2_reg_x86::<T, 4, 4>(dst, stride, edges, packed),
            (4, _) => z2_reg_x86::<T, 4, 8>(dst, stride, edges, packed),
            (_, 4) => z2_reg_x86::<T, 8, 4>(dst, stride, edges, packed),
            _ => z2_reg_x86::<T, 8, 8>(dst, stride, edges, packed),
        }
    }
}

/// Whether [`z2_small16`] handles a z2 block.
#[inline(always)]
fn use_z2_small16<T: Pixel, const BD: u32>(w: usize, h: usize, upsample_left: bool) -> bool {
    BD <= 10 && !is_u8::<T>() && w <= 8 && h <= 16 && (!upsample_left || h <= 8)
}

/// [`z2_4x4`] / [`z2_tiles`] for runtime upsampling flags.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn z2_small16<T: Pixel, const L: usize>(
    dst: &mut [T],
    stride: usize,
    above: &[T; L],
    left: &[T; L],
    w: usize,
    h: usize,
    dx: i32,
    dy: i32,
    upsample_above: bool,
    upsample_left: bool,
) {
    macro_rules! tiles {
        ($ua:literal, $ul:literal) => {
            if HAS_FAST_SWIZZLE {
                match (w, h) {
                    (4, 4) => z2_tbl16::<T, 4, 4, $ua, $ul, L>(dst, stride, above, left, dx, dy),
                    (4, 8) => z2_tbl16::<T, 4, 8, $ua, $ul, L>(dst, stride, above, left, dx, dy),
                    (8, 4) => z2_tbl16::<T, 8, 4, $ua, $ul, L>(dst, stride, above, left, dx, dy),
                    (8, 8) => z2_tbl16::<T, 8, 8, $ua, $ul, L>(dst, stride, above, left, dx, dy),
                    // h == 16: never upsampled on the left.
                    (4, _) => z2_tbl16::<T, 4, 16, $ua, false, L>(dst, stride, above, left, dx, dy),
                    _ => z2_tbl16::<T, 8, 16, $ua, false, L>(dst, stride, above, left, dx, dy),
                }
            } else if w == 4 && h == 4 {
                z2_4x4::<T, $ua, $ul, L>(dst, stride, above, left, dx, dy)
            } else {
                z2_tiles::<T, $ua, $ul, L>(dst, stride, above, left, w, h, dx, dy)
            }
        };
    }
    match (upsample_above, upsample_left) {
        (false, false) => tiles!(false, false),
        (true, false) => tiles!(true, false),
        (false, true) => tiles!(false, true),
        (true, true) => tiles!(true, true),
    }
}

/// z2 of a 4x4 block (see [`z2_tiles`]).
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn z2_4x4<T: Pixel, const UA: bool, const UL: bool, const L: usize>(
    dst: &mut [T],
    stride: usize,
    above: &[T; L],
    left: &[T; L],
    dx: i32,
    dy: i32,
) {
    let (dst, stride, _) = check_block4::<T, 4>(dst, stride, 4);
    let ua = UA as i32;
    let (dxv, dyv) = (i16x8::splat(dx as i16), i16x8::splat(dy as i16));
    let t01 = Z2_LANE_I1 * dxv;
    let t23 = t01 + dxv + dxv;
    let pa = |k: i32| z2_pos((-k * dx) >> (6 - ua));
    let tile = Z2Tile {
        pa: [pa(1), pa(2), pa(3), pa(4)],
        idx: [Z2_LANE_J64 - t01, Z2_LANE_J64 - t23],
        ru: 0,
        c: 0,
        need_a: 3 * 64 - dx >= -64,
        need_l: -4 * dx < -64,
    };
    let v = z2_tile::<T, UA, UL, L>(above, left, dy, dyv, &tile);
    let (r0, r1, r2, r3) = row4_mut::<T, 4>(dst, 0, stride);
    z2_store4::<T>([r0, r1, r2, r3], v);
}

/// z2 for samples of up to 10 bits and blocks of width `w` (4 or 8) and
/// height `h` (4, 8 or 16; at most 8 if `UL`), with upsampling `UA` /
/// `UL`, by 4x4 tiles of two 8-lane vectors (two rows each).
///
/// In a tile, the above edge prediction of every row is one contiguous
/// run (packed by row pairs) and the left edge prediction of every
/// column is one contiguous run with a constant shift (packed by column
/// pairs, then transposed); each lane selects one of them. Lane `(i, j)`
/// uses the above edge iff `base >= -(1 << UA)`, i.e. `idx = (j << 6) -
/// (i + 1) * dx >= -64`. Tiles in which no lane uses one of the edges
/// skip it. A run starting below spec index `-EDGE_PAD` has no lane that
/// selects it (`h <= 16`, `w <= 8`, and the decoder's left reads are at
/// spec indices `>= -(1 << UL)`), so its position may wrap
/// ([`z2_pos`]).
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn z2_tiles<T: Pixel, const UA: bool, const UL: bool, const L: usize>(
    dst: &mut [T],
    stride: usize,
    above: &[T; L],
    left: &[T; L],
    w: usize,
    h: usize,
    dx: i32,
    dy: i32,
) {
    if w == 4 {
        z2_tiles_w::<T, 4, UA, UL, L>(dst, stride, above, left, h, dx, dy);
    } else {
        z2_tiles_w::<T, 8, UA, UL, L>(dst, stride, above, left, h, dx, dy);
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn z2_tiles_w<T: Pixel, const W: usize, const UA: bool, const UL: bool, const L: usize>(
    dst: &mut [T],
    stride: usize,
    above: &[T; L],
    left: &[T; L],
    h: usize,
    dx: i32,
    dy: i32,
) {
    const { assert!(EDGE_PAD == 16 && (W == 4 || W == 8) && L >= 32 + 8) };
    debug_assert!(h % 4 == 0 && h <= 16 && (!UL || h <= 8));
    let (dst, stride, h4) = check_block4::<T, W>(dst, stride, h);
    let (ua, ul) = (UA as i32, UL as i32);
    let (dxv, dyv) = (i16x8::splat(dx as i16), i16x8::splat(dy as i16));
    let mut t01 = Z2_LANE_I1 * dxv;
    for i4 in 0..h4 {
        let r = (4 * i4) as i32;
        let t23 = t01 + dxv + dxv;
        let ab = |k: i32| (-(r + k + 1) * dx) >> (6 - ua);
        let abs = [ab(0), ab(1), ab(2), ab(3)];
        let (r0, r1, r2, r3) = row4_mut::<T, W>(dst, i4, stride);
        let ru = r << ul;
        let tile0 = Z2Tile {
            pa: [
                z2_pos(abs[0]),
                z2_pos(abs[1]),
                z2_pos(abs[2]),
                z2_pos(abs[3]),
            ],
            idx: [Z2_LANE_J64 - t01, Z2_LANE_J64 - t23],
            ru,
            c: 0,
            need_a: 3 * 64 - (r + 1) * dx >= -64,
            need_l: -(r + 4) * dx < -64,
        };
        let v0 = z2_tile::<T, UA, UL, L>(above, left, dy, dyv, &tile0);
        if W == 4 {
            z2_store4::<T>([r0, r1, r2, r3], v0);
        } else {
            let cu = 4 << ua;
            let j4 = Z2_LANE_J64 + i16x8::splat(256);
            let tile1 = Z2Tile {
                pa: [
                    z2_pos(cu + abs[0]),
                    z2_pos(cu + abs[1]),
                    z2_pos(cu + abs[2]),
                    z2_pos(cu + abs[3]),
                ],
                idx: [j4 - t01, j4 - t23],
                ru,
                c: 4,
                need_a: 7 * 64 - (r + 1) * dx >= -64,
                need_l: 4 * 64 - (r + 4) * dx < -64,
            };
            let v1 = z2_tile::<T, UA, UL, L>(above, left, dy, dyv, &tile1);
            store_i16::<T, 8>(r0, simd_swizzle!(v0[0], v1[0], [0, 1, 2, 3, 8, 9, 10, 11]));
            store_i16::<T, 8>(
                r1,
                simd_swizzle!(v0[0], v1[0], [4, 5, 6, 7, 12, 13, 14, 15]),
            );
            store_i16::<T, 8>(r2, simd_swizzle!(v0[1], v1[1], [0, 1, 2, 3, 8, 9, 10, 11]));
            store_i16::<T, 8>(
                r3,
                simd_swizzle!(v0[1], v1[1], [4, 5, 6, 7, 12, 13, 14, 15]),
            );
        }
        t01 = t23 + dxv + dxv;
    }
}

/// Loads 8 samples at `buf[pos & 127..]` and the 8 following ones by
/// one, as `i16` lanes (see [`z2_pos`] for the wrapping).
#[inline(always)]
fn z2_pair8<T: Pixel>(buf: &[T; EDGE_BUF_LEN], pos: usize) -> (i16x8, i16x8) {
    const { assert!(128 + 9 <= EDGE_BUF_LEN) };
    let pos = pos & 127;
    (
        load_i16::<T, 8>(&buf[pos..]),
        load_i16::<T, 8>(&buf[pos + 1..]),
    )
}

/// Left edge prediction of the 8x8 tile at rows `r..r + 8`, columns
/// `c..c + 8` of [`z2_big`] (transposed column runs).
#[inline(always)]
fn z2_left8<T: Pixel>(left: &[T; EDGE_BUF_LEN], r: i32, c: usize, dy: i32) -> [i16x8; 8] {
    let v0 = -((c as i32 + 1) * dy);
    let col = |k: i32| {
        let v = v0 - k * dy;
        let (a, b) = z2_pair8::<T>(left, z2_pos(r + (v >> 6)));
        interp_i16x8(a, b, i16x8::splat(((v >> 1) & 31) as i16))
    };
    transpose([
        col(0),
        col(1),
        col(2),
        col(3),
        col(4),
        col(5),
        col(6),
        col(7),
    ])
}

/// z2 without upsampling for samples of up to 10 bits and blocks of
/// width `W` (8..=64) and height `h` (8..=64, a multiple of 8), by 8x8
/// tiles of 8-lane rows.
///
/// Row `i` uses the left edge for columns `j < m(i) = ((i + 1) * dx - 1)
/// >> 6` (clamped to `0..=W`), which does not decrease with `i`. In a
/// group of 8 rows, tiles left of `m(r)` only use the left edge: every
/// column is one contiguous run of the left edge with a constant shift,
/// interpolated as one vector and transposed. Tiles right of `m(r + 7)`
/// only use the above edge: every row is one contiguous run of it with
/// a constant shift. The tiles between compute both and select per lane
/// (`idx = (j << 6) - (i + 1) * dx >= -64`). Run positions wrap into the
/// buffer ([`z2_pos`]): a run starting below spec index `-EDGE_PAD`
/// (or above the 128 samples) has no lane selecting it.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn z2_big<T: Pixel, const W: usize>(
    dst: &mut [T],
    stride: usize,
    above: &[T; EDGE_BUF_LEN],
    left: &[T; EDGE_BUF_LEN],
    h: usize,
    dx: i32,
    dy: i32,
) {
    const { assert!(W % 8 == 0 && W <= 64 && 128 + 17 <= EDGE_BUF_LEN) };
    debug_assert!(h % 8 == 0 && (8..=64).contains(&h));
    let (dst, stride, h4) = check_block4::<T, W>(dst, stride, h);
    let lane64 = i16x8::from_array([0, 64, 128, 192, 256, 320, 384, 448]);
    let m = |i: i32| ((((i + 1) * dx - 1) >> 6).clamp(0, W as i32)) as usize;
    let interp = |p: (i16x8, i16x8), s: i16x8| interp_i16x8(p.0, p.1, s);
    for i8 in 0..h4 / 2 {
        let r = i8 * 8;
        let ri = r as i32;
        let (j_min, j_max) = (m(ri), m(ri + 7));
        let base = r * stride;
        let win = &mut dst[base..base + stride * 7 + W];
        let mut c = 0;
        // Tiles using only the left edge.
        while c + 8 <= j_min {
            let v = z2_left8::<T>(left, ri, c, dy);
            let (top, bot) = win.split_at_mut(stride * 4);
            let (r0, r1, r2, r3) = split_rows4::<T, W>(top, stride);
            let (r4, r5, r6, r7) = split_rows4::<T, W>(bot, stride);
            for (row, &vk) in [r0, r1, r2, r3, r4, r5, r6, r7].into_iter().zip(v.iter()) {
                store_i16::<T, 8>(&mut row[c..], vk);
            }
            c += 8;
        }
        // Tiles crossing the boundary.
        while c < j_max {
            let v = z2_left8::<T>(left, ri, c, dy);
            let c64 = lane64 + i16x8::splat((c * 64) as i16);
            let (top, bot) = win.split_at_mut(stride * 4);
            let (r0, r1, r2, r3) = split_rows4::<T, W>(top, stride);
            let (r4, r5, r6, r7) = split_rows4::<T, W>(bot, stride);
            for (k, (row, &vk)) in [r0, r1, r2, r3, r4, r5, r6, r7]
                .into_iter()
                .zip(v.iter())
                .enumerate()
            {
                let tk = (ri + k as i32 + 1) * dx;
                let sa = i16x8::splat(((-tk >> 1) & 31) as i16);
                let a = interp(
                    z2_pair8::<T>(above, z2_pos(((c * 64) as i32 - tk) >> 6)),
                    sa,
                );
                // (j << 6) <= 4032: larger t only select the left edge.
                let idx = c64 - i16x8::splat(tk.min(8191) as i16);
                store_i16::<T, 8>(&mut row[c..], idx.simd_ge(i16x8::splat(-64)).select(a, vk));
            }
            c += 8;
        }
        // Tiles using only the above edge: each row k is a contiguous run
        // starting at spec index ((-tk) >> 6) + c_above with constant shift.
        if c < W {
            let c_above = c;
            let (top, bot) = win.split_at_mut(stride * 4);
            let (r0, r1, r2, r3) = split_rows4::<T, W>(top, stride);
            let (r4, r5, r6, r7) = split_rows4::<T, W>(bot, stride);
            for (k, row) in [r0, r1, r2, r3, r4, r5, r6, r7].into_iter().enumerate() {
                let tk = (ri + k as i32 + 1) * dx;
                let b0 = z2_pos((-tk) >> 6);
                let s8 = i16x8::splat(((-tk >> 1) & 31) as i16);
                let s16 = Simd::<i16, 16>::splat(((-tk >> 1) & 31) as i16);
                let mut col = c_above;
                while col + 16 <= W {
                    let pos = b0.wrapping_add(col) & 127;
                    let a = load_i16::<T, 16>(&above[pos..]);
                    let b = load_i16::<T, 16>(&above[pos + 1..]);
                    let v = a + (((b - a) * s16 + Simd::splat(16)) >> 5);
                    store_i16::<T, 16>(&mut row[col..], v);
                    col += 16;
                }
                if col < W {
                    let v = interp(z2_pair8::<T>(above, b0.wrapping_add(col)), s8);
                    store_i16::<T, 8>(&mut row[col..], v);
                }
            }
        }
    }
}

#[inline(always)]
fn z2_8x4_left_tile<T: Pixel, E: Lane>(
    left: &[T; EDGE_BUF_LEN],
    c: usize,
    dy: i32,
    upsample_left: bool,
) -> [Simd<E, 4>; 4] {
    let ul = upsample_left as i32;
    let pad = EDGE_PAD as i32;
    let mut cols = [E::splat::<4>(0); 4];
    for (k, col) in cols.iter_mut().enumerate() {
        let idx = -((c + k + 1) as i32) * dy;
        let b = ((idx >> (6 - ul)).max(-pad) + pad) as usize;
        let s = E::splat(((idx << ul) >> 1) & 0x1f);
        *col = interp_run::<T, E, 4>(&left[b..], upsample_left, s);
    }
    transpose(cols)
}

/// z2 for 8x4 blocks: interpolates the 4-row left columns in 4-lane
/// tiles and each 8-pixel row of the above edge as a single 8-lane
/// vector. Columns 4..8 only read the left edge when `4 * dx > 256`
/// (`dx > 64`).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn z2_8x4<T: Pixel, E: Lane>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    dx: i32,
    dy: i32,
    upsample_above: bool,
    upsample_left: bool,
) {
    let (dst, stride, _) = check_block4::<T, 8>(dst, stride, 4);
    let above: &[T; EDGE_BUF_LEN] = above[..EDGE_BUF_LEN].try_into().unwrap();
    let left: &[T; EDGE_BUF_LEN] = left[..EDGE_BUF_LEN].try_into().unwrap();
    let ua = upsample_above as i32;
    let pad = EDGE_PAD as i32;
    let lt0 = z2_8x4_left_tile::<T, E>(left, 0, dy, upsample_left);
    let lt1 = if dx > 64 {
        z2_8x4_left_tile::<T, E>(left, 4, dy, upsample_left)
    } else {
        [E::splat::<4>(0); 4]
    };
    let (r0, r1, r2, r3) = row4_mut::<T, 8>(dst, 0, stride);
    for (i, row) in [r0, r1, r2, r3].into_iter().enumerate() {
        let lt = E::concat4(lt0[i], lt1[i]);
        let t = (i as i32 + 1) * dx;
        let m = (((t - 1) >> 6).clamp(0, 8)) as usize;
        let idx = -t;
        let b = ((idx >> (6 - ua)).max(-pad) + pad) as usize;
        let at = interp_run::<T, E, 8>(
            &above[b..],
            upsample_above,
            E::splat(((idx << ua) >> 1) & 0x1f),
        );
        E::store(row, E::select_lt(m, lt, at));
    }
}

/// z2 by rows of `K x K` tiles.
///
/// In row `i` the columns `j < j0(i)` use the left edge and the others
/// the above edge; `j0` does not decrease with `i`. Along a row the
/// above edge is a contiguous run (constant shift), so above samples
/// are computed per row, in vectors of `N` where no row of the tile
/// uses the left edge. Along a column the left edge is a contiguous run
/// (constant shift), so left samples are computed per tile column and
/// transposed.
///
/// In boundary tiles crossing the diagonal, every tile column computes
/// its left-edge vector and every tile row computes its above-edge
/// vector without branching, with edge positions clamped to spec index
/// `-16` (`EDGE_PAD`) so unused lanes stay in bounds.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn z2_w<T: Pixel, E: Lane, const W: usize, const N: usize, const K: usize>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    h: usize,
    dx: i32,
    dy: i32,
    upsample_above: bool,
    upsample_left: bool,
) {
    const { assert!(EDGE_PAD >= 2 * K) };
    let (dst, stride) = check_block::<T, W>(dst, stride, h);
    let above: &[T; EDGE_BUF_LEN] = above[..EDGE_BUF_LEN].try_into().unwrap();
    let left: &[T; EDGE_BUF_LEN] = left[..EDGE_BUF_LEN].try_into().unwrap();
    let ua = upsample_above as i32;
    let ul = upsample_left as i32;
    let pad = EDGE_PAD as i32;
    for r in (0..h / K).map(|t| t * K) {
        let j_min = (((r as i32 + 1) * dx - 1) >> 6).clamp(0, W as i32) as usize;
        let j_max = (((r + K) as i32 * dx - 1) >> 6).clamp(0, W as i32) as usize;
        // Tiles completely inside the left-edge region (c + K <= j_min).
        let c_full = j_min & !(K - 1);
        for c in (0..c_full).step_by(K) {
            let mut cols = [E::splat::<K>(0); K];
            for (k, col) in cols.iter_mut().enumerate() {
                let idx = ((r as i32) << 6) - ((c + k + 1) as i32) * dy;
                let b = ((idx >> (6 - ul)) + pad) as usize;
                let s = E::splat(((idx << ul) >> 1) & 0x1f);
                *col = interp_run::<T, E, K>(&left[b..], upsample_left, s);
            }
            for (k, lt) in transpose(cols).into_iter().enumerate() {
                E::store(row_mut::<T, K>(dst, (r + k) * stride + c), lt);
            }
        }
        // Boundary tiles crossing the left/above diagonal (c_full..x0).
        let x0 = j_max.next_multiple_of(K);
        for c in (c_full..x0).step_by(K) {
            let mut cols = [E::splat::<K>(0); K];
            for (k, col) in cols.iter_mut().enumerate() {
                let idx = ((r as i32) << 6) - ((c + k + 1) as i32) * dy;
                let b = ((idx >> (6 - ul)).max(-pad) + pad) as usize;
                let s = E::splat(((idx << ul) >> 1) & 0x1f);
                *col = interp_run::<T, E, K>(&left[b..], upsample_left, s);
            }
            for (k, lt) in transpose(cols).into_iter().enumerate() {
                let i = r + k;
                let t = (i as i32 + 1) * dx;
                let m = (((t - 1) >> 6) - c as i32).clamp(0, K as i32) as usize;
                let idx = (c as i32) * 64 - t;
                let b = ((idx >> (6 - ua)).max(-pad) + pad) as usize;
                let s = E::splat(((idx << ua) >> 1) & 0x1f);
                let at = interp_run::<T, E, K>(&above[b..], upsample_above, s);
                E::store(
                    row_mut::<T, K>(dst, i * stride + c),
                    E::select_lt(m, lt, at),
                );
            }
        }
        if x0 >= W {
            continue;
        }
        // Columns from x0 on use the above edge in every row of the tile.
        for i in r..r + K {
            let row = row_mut::<T, W>(dst, i * stride);
            let idx0 = -(i as i32 + 1) * dx;
            let s = E::splat(((idx0 << ua) >> 1) & 0x1f);
            let b0 = ((idx0 >> (6 - ua)) + pad) as usize;
            let mut x = x0;
            if K == 4 && N >= 8 && (x & 4) != 0 && x < W {
                let v = interp_run::<T, E, 4>(
                    &above[b0.wrapping_add(x << ua)..],
                    upsample_above,
                    E::splat(((idx0 << ua) >> 1) & 0x1f),
                );
                E::store(&mut row[x..x + 4], v);
                x += 4;
            }
            if N == 16 && (x & 8) != 0 && x < W {
                let v = interp_run::<T, E, 8>(
                    &above[b0.wrapping_add(x << ua)..],
                    upsample_above,
                    E::splat(((idx0 << ua) >> 1) & 0x1f),
                );
                E::store(&mut row[x..x + 8], v);
                x += 8;
            }
            while x < W {
                let v =
                    interp_run::<T, E, N>(&above[b0.wrapping_add(x << ua)..], upsample_above, s);
                E::store(&mut row[x..x + N], v);
                x += N;
            }
        }
    }
}

/// `dsp::ipred::ipred_z3`.
pub fn ipred_z3<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    left: &[T],
    w: usize,
    h: usize,
    dy: i32,
    upsample: bool,
) {
    // 8-bit blocks up to 8x16 whose edge reads stay within 32 samples:
    // per lane table lookups.
    let u = upsample as usize;
    if HAS_FAST_SWIZZLE
        && is_u8::<T>()
        && w <= 8
        && h <= 16
        && ((w as i32 * dy) >> (6 - u)) + ((h as i32 - 1) << u) + 1 < 32
        && left.len() >= 32
    {
        match (w, upsample) {
            (4, false) => z3_tbl::<T, 4, 0>(dst, stride, left, h, dy),
            (4, true) => z3_tbl::<T, 4, 1>(dst, stride, left, h, dy),
            (_, false) => z3_tbl::<T, 8, 0>(dst, stride, left, h, dy),
            (_, true) => z3_tbl::<T, 8, 1>(dst, stride, left, h, dy),
        }
        return;
    }
    ipred_z3_large::<T, BD>(dst, stride, left, w, h, dy, upsample);
}

#[inline(never)]
fn ipred_z3_large<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    left: &[T],
    w: usize,
    h: usize,
    dy: i32,
    upsample: bool,
) {
    // Separate instances with and without upsampling (z3_k and z3_w are
    // inlined).
    match (BD <= 10, upsample) {
        (true, false) => by_width!(w, z3_k::<T, i16>(dst, stride, left, h, dy, false)),
        (true, true) => by_width!(w, z3_k::<T, i16>(dst, stride, left, h, dy, true)),
        (false, false) => by_width!(w, z3_k::<T, u16>(dst, stride, left, h, dy, false)),
        (false, true) => by_width!(w, z3_k::<T, u16>(dst, stride, left, h, dy, true)),
    }
}

/// Selects the tile size of [`z3_w`]: 8 when both dimensions allow it.
#[inline(always)]
fn z3_k<T: Pixel, E: Lane, const W: usize, const N: usize>(
    dst: &mut [T],
    stride: usize,
    left: &[T],
    h: usize,
    dy: i32,
    upsample: bool,
) {
    if W >= 8 && h >= 8 {
        z3_w::<T, E, W, 8>(dst, stride, left, h, dy, upsample);
    } else {
        z3_w::<T, E, W, 4>(dst, stride, left, h, dy, upsample);
    }
}

/// z3 by `K x K` tiles: each column of a tile is a contiguous run of the
/// left edge (base advances by one sample per row, constant shift),
/// interpolated as one vector; the tile is then transposed. Reads
/// exactly the samples the scalar kernel reads.
#[inline(always)]
fn z3_w<T: Pixel, E: Lane, const W: usize, const K: usize>(
    dst: &mut [T],
    stride: usize,
    left: &[T],
    h: usize,
    dy: i32,
    upsample: bool,
) {
    let (dst, stride, h4) = check_block4::<T, W>(dst, stride, h);
    let h = h4 * 4;
    let u = upsample as usize;
    for c in (0..W / K).map(|t| t * K) {
        let dst_c = &mut dst[c..];
        for r in (0..h / K).map(|t| t * K) {
            // Rows r..r + K of column c + k: left[base0 + (i << u)] and
            // the next sample.
            let mut cols = [E::splat::<K>(0); K];
            for (k, col) in cols.iter_mut().enumerate() {
                let idx = (c + k + 1) as i32 * dy;
                let b = (idx >> (6 - u)) as usize + (r << u);
                let s = E::splat(((idx << u) >> 1) & 0x1f);
                *col = interp_run::<T, E, K>(&left[b..], upsample, s);
            }
            let rows = transpose(cols);
            for c4 in 0..K / 4 {
                let (r0, r1, r2, r3) = row4_mut::<T, K>(dst_c, (r >> 2) + c4, stride);
                for (k4, row) in [r0, r1, r2, r3].into_iter().enumerate() {
                    E::store(row, rows[c4 * 4 + k4]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Recursive filter intra.
// ---------------------------------------------------------------------------

/// `INTRA_FILTER_TAPS` transposed: `[mode][t][k]` is tap `t` of output
/// `k`.
static FILTER_TAP_VECTORS: [[[i16; 8]; 7]; 5] = {
    let mut v = [[[0i16; 8]; 7]; 5];
    let mut m = 0;
    while m < 5 {
        let mut k = 0;
        while k < 8 {
            let mut t = 0;
            while t < 7 {
                v[m][t][k] = INTRA_FILTER_TAPS[m][k][t] as i16;
                t += 1;
            }
            k += 1;
        }
        m += 1;
    }
    v
};

/// `dsp::ipred::ipred_filter`.
///
/// Each 4x2 block is one 8-lane vector (lane `k` is output `k`, i.e.
/// row `k >> 2`, column `k & 3`): the 7-tap filter is 7 multiply-adds of
/// a tap vector by a broadcast input sample. Samples up to 10 bits use
/// i16 lanes (the positive taps of an output sum to at most 28, the
/// negative ones to at least -12, so sums are within +-28 * 1023), 12-bit
/// samples i32 lanes with 16 x 16 bit products.
///
/// All inputs of a block are lanes of earlier output vectors (the block
/// above, the block above-left and the block to the left), so they are
/// broadcast from registers (multiply-add by element) instead of going
/// through memory: the dependency chain between blocks stays short.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn ipred_filter<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    top_left: T,
    above: &[T],
    left: &[T],
    w: usize,
    h: usize,
    mode: usize,
    bd: u32,
) {
    debug_assert!(w <= 32 && h <= 32 && bd == BD);
    match w {
        4 => filter_w::<T, BD, 1>(dst, stride, top_left, above, left, h, mode),
        8 => filter_w::<T, BD, 2>(dst, stride, top_left, above, left, h, mode),
        16 => filter_w::<T, BD, 4>(dst, stride, top_left, above, left, h, mode),
        32 => filter_w::<T, BD, 8>(dst, stride, top_left, above, left, h, mode),
        w => panic!("invalid filter intra width {w}"),
    }
}

/// Stores the 8 lanes of a 4x2 filter intra subblock to `d0` (row 0)
/// and `d1` (row 1), narrowing once.
#[inline(always)]
fn store_filter_4x2<T: Pixel>(d0: &mut [T; 4], d1: &mut [T; 4], out: Simd<i16, 8>) {
    let any0: &mut dyn Any = d0;
    if let Some(d0) = any0.downcast_mut::<[u8; 4]>() {
        let any1: &mut dyn Any = d1;
        let d1 = any1.downcast_mut::<[u8; 4]>().unwrap();
        let u = out.cast::<u8>().to_array();
        *d0 = u[..4].try_into().unwrap();
        *d1 = u[4..8].try_into().unwrap();
    } else {
        store_i16::<T, 4>(d0, simd_swizzle!(out, [0, 1, 2, 3]));
        store_i16::<T, 4>(d1, simd_swizzle!(out, [4, 5, 6, 7]));
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn filter_w<T: Pixel, const BD: u32, const WB: usize>(
    dst: &mut [T],
    stride: usize,
    top_left: T,
    above: &[T],
    left: &[T],
    h: usize,
    mode: usize,
) {
    let w = 4 * WB;
    let stride = stride & 0xffff_ffff;
    assert!(stride >= w && (2..=32).contains(&h));
    let dst = &mut dst[..(h - 1) * stride + w];
    let above = &above[..w];
    let left4 = left[..h].as_chunks::<2>().0;
    let tap = FILTER_TAP_VECTORS[mode].map(Simd::<i16, 8>::from_array);
    // prev[b]: output vector of block b of the previous strip; lanes 4..8
    // (its bottom row) are the above inputs of block b.With compile-time
    // WB, prev stays in SIMD registers across row strips.
    let mut prev: [Simd<i16, 8>; WB] = std::array::from_fn(|b| {
        let a = load_i16::<T, 4>(&above[4 * b..]);
        simd_swizzle!(a, [0, 1, 2, 3, 0, 1, 2, 3])
    });
    // Broadcast of lane K.
    macro_rules! lane {
        ($v:expr, $k:literal) => {
            simd_swizzle!($v, [$k; 8])
        };
    }
    let max = (1i32 << BD) - 1;
    let mut tl_row = Simd::<i16, 8>::splat(top_left.to_i32() as i16);
    for (i2, &[l0, l1]) in left4.iter().enumerate() {
        let r0 = 2 * i2 * stride;
        let r1 = r0 + stride;
        let mut tl = tl_row;
        let mut p5 = Simd::<i16, 8>::splat(l0.to_i32() as i16);
        let mut p6 = Simd::<i16, 8>::splat(l1.to_i32() as i16);
        tl_row = p6;
        for (b, pv) in prev.iter_mut().enumerate() {
            let p = *pv;
            let p4 = lane!(p, 7);
            // Only `p5` and `p6` (outputs of the block to the left) are on
            // the dependency chain between blocks: the other five terms are
            // summed first, the chain terms last.
            let out = if BD <= 10 {
                let base = Simd::<i16, 8>::splat(8)
                    + tap[0] * tl
                    + tap[1] * lane!(p, 4)
                    + tap[2] * lane!(p, 5)
                    + tap[3] * lane!(p, 6)
                    + tap[4] * p4;
                let acc = base + tap[5] * p5 + tap[6] * p6;
                (acc >> 4).simd_clamp(Simd::splat(0), Simd::splat(max as i16))
            } else {
                let ins = [tl, lane!(p, 4), lane!(p, 5), lane!(p, 6), p4, p5, p6];
                let mut acc = Simd::<i32, 8>::splat(8);
                for (&t, &v) in tap.iter().zip(&ins) {
                    acc += t.cast::<i32>() * v.cast::<i32>();
                }
                (acc >> 4)
                    .simd_clamp(Simd::splat(0), Simd::splat(max))
                    .cast::<i16>()
            };
            let (before, after) = dst.split_at_mut(r1);
            store_filter_4x2(
                (&mut before[r0 + 4 * b..r0 + 4 * b + 4])
                    .try_into()
                    .unwrap(),
                (&mut after[4 * b..4 * b + 4]).try_into().unwrap(),
                out,
            );
            tl = p4;
            p5 = lane!(out, 3);
            p6 = lane!(out, 7);
            *pv = out;
        }
    }
}

// ---------------------------------------------------------------------------
// Chroma from luma.
// ---------------------------------------------------------------------------

/// `dsp::ipred::cfl_ac`.
#[allow(clippy::too_many_arguments)]
pub fn cfl_ac<T: Pixel, const BD: u32>(
    ac: &mut [i16],
    luma: &[T],
    stride: usize,
    w: usize,
    h: usize,
    wv: usize,
    hv: usize,
    ss_x: usize,
    ss_y: usize,
) {
    // The decoder uses `u8` pixels iff the bit depth is 8. The tests also
    // run `u16` pixels at bit depth 8, which every kernel supports.
    debug_assert!(
        !(std::mem::size_of::<T>() == 1 && BD > 8)
            && (cfg!(test) || !(std::mem::size_of::<T>() == 2 && BD == 8))
    );
    debug_assert!((1..=w).contains(&wv) && (1..=h).contains(&hv) && w * h <= MAX_CFL_AC);
    macro_rules! go {
        ($sx:literal, $sy:literal) => {
            match w {
                4 => cfl_ac_w::<T, BD, $sx, $sy, 4, 4>(ac, luma, stride, h, wv, hv),
                8 => cfl_ac_w::<T, BD, $sx, $sy, 8, 8>(ac, luma, stride, h, wv, hv),
                16 => cfl_ac_w::<T, BD, $sx, $sy, 16, 16>(ac, luma, stride, h, wv, hv),
                32 => cfl_ac_w::<T, BD, $sx, $sy, 32, 16>(ac, luma, stride, h, wv, hv),
                w => panic!("invalid cfl width {w}"),
            }
        };
    }
    match (ss_x, ss_y) {
        (1, 1) => go!(1, 1),
        (1, 0) => go!(1, 0),
        (0, 0) => go!(0, 0),
        _ => go!(0, 1),
    }
}

/// `src[2 * k] + src[2 * k + 1]` for lanes `k = 0..K` (reads
/// `src[..2 * K]`).
///
/// For `u8` samples the byte pairs are loaded as `u16` lanes and summed
/// as `(p & 0xff) + (p >> 8)` (independent of the byte order): one plain
/// load and three cheap lane ops (x86 `pand` / `psrlw` / `paddw`, NEON
/// `bic` / `usra`), instead of widening two vectors and deinterleaving
/// them.
#[inline(always)]
fn pair_sum<T: Pixel, const K: usize>(src: &[T]) -> Simd<u16, K> {
    let p: &[[T; 2]; K] = src.as_chunks::<2>().0[..K].try_into().unwrap();
    let any: &dyn Any = p;
    if let Some(p) = any.downcast_ref::<[[u8; 2]; K]>() {
        let w = Simd::<u16, K>::from_array(std::array::from_fn(|k| u16::from_ne_bytes(p[k])));
        (w & Simd::splat(0xff)) + (w >> 8)
    } else {
        let (e, o) = load_u16::<T, K>(src).deinterleave(load_u16::<T, K>(&src[K..]));
        e + o
    }
}

/// `K` subsampled luma sums (scaled to 3 fractional bits) starting at
/// output column 0 of luma rows `r0` / `r1` (`r1` is only read if
/// `SY == 1`). `u16` lanes hold up to `8 * 4095`.
#[inline(always)]
fn cfl_ac_vec<T: Pixel, const SX: usize, const SY: usize, const K: usize>(
    r0: &[T],
    r1: &[T],
) -> Simd<u16, K> {
    let row = |r: &[T]| if SX == 1 { pair_sum::<T, K>(r) } else { load_u16::<T, K>(r) };
    let mut s = row(r0);
    if SY == 1 {
        s += row(r1);
    }
    s << Simd::splat((3 - SX - SY) as u16)
}

#[inline(never)]
fn cfl_ac_w<
    T: Pixel,
    const BD: u32,
    const SX: usize,
    const SY: usize,
    const W: usize,
    const K: usize,
>(
    ac: &mut [i16],
    luma: &[T],
    stride: usize,
    h: usize,
    wv: usize,
    hv: usize,
) {
    const { assert!(W == K || W == 2 * K) };
    debug_assert!((1..=32).contains(&h) && (1..=h).contains(&hv));
    let ac = &mut ac[..W * h];
    let stride = stride & 0xffff_ffff;
    // 8-bit values are at most 8 * 255, so a u16 lane can sum the 32
    // values of one column chunk over all rows (at most 32 * 2040).
    let small = BD == 8 || is_u8::<T>();
    let mut acc0 = Simd::<u16, K>::splat(0);
    let mut acc1 = Simd::<u16, K>::splat(0);
    let mut acc = Simd::<u32, K>::splat(0);
    if wv == W {
        let w_luma = W << SX;
        debug_assert!(stride >= w_luma && (1..=32).contains(&hv));
        // Panic-free in release builds: the clamp and the bounds-checked
        // slices below keep every access in `luma`.
        let stride = stride.max(w_luma);
        let luma = &luma[..((hv << SY) - 1) * stride + w_luma];
        for (i, row) in ac.as_chunks_mut::<W>().0[..hv].iter_mut().enumerate() {
            let (r0, r1) = if SY == 0 {
                let r = &luma[(i & 31) * stride..][..w_luma];
                (r, r)
            } else {
                let base = (i & 31) * (stride * 2);
                let win = &luma[base..base + stride + w_luma];
                let (a, b) = win.split_at(stride);
                (&a[..w_luma], b)
            };
            let v0 = cfl_ac_vec::<T, SX, SY, K>(r0, r1);
            if small {
                acc0 += v0;
            } else {
                acc += v0.cast::<u32>();
            }
            row[..K].copy_from_slice(v0.cast::<i16>().as_array());
            if W == 2 * K {
                let v1 = cfl_ac_vec::<T, SX, SY, K>(&r0[K << SX..], &r1[K << SX..]);
                if small {
                    acc1 += v1;
                } else {
                    acc += v1.cast::<u32>();
                }
                row[K..2 * K].copy_from_slice(v1.cast::<i16>().as_array());
            }
        }
    } else {
        // Right frame edge: output columns computed with whole vectors
        // where possible, the rest per sample.
        let wfull = wv / K * K;
        for (i, row) in ac.as_chunks_mut::<W>().0[..hv].iter_mut().enumerate() {
            let r0 = &luma[(i << SY) * stride..];
            let r1 = &luma[((i << SY) + SY) * stride..];
            if wfull >= K {
                let v0 = cfl_ac_vec::<T, SX, SY, K>(r0, r1);
                row[..K].copy_from_slice(v0.cast::<i16>().as_array());
            }
            for (j, d) in row[..wv].iter_mut().enumerate().skip(wfull) {
                let x = j << SX;
                let mut t = r0[x].to_i32();
                if SX == 1 {
                    t += r0[x + 1].to_i32();
                }
                if SY == 1 {
                    t += r1[x].to_i32();
                    if SX == 1 {
                        t += r1[x + 1].to_i32();
                    }
                }
                *d = (t << (3 - SX - SY)) as i16;
            }
            let last = row[wv - 1];
            row[wv..].fill(last);
        }
    }
    for i in hv..h {
        ac.copy_within((hv - 1) * W..hv * W, i * W);
    }
    let sum = if wv == W && hv == h {
        if small {
            (acc0.cast::<u32>() + acc1.cast::<u32>()).reduce_sum()
        } else {
            acc.reduce_sum()
        }
    } else {
        ac.iter().map(|&v| v as u32).sum()
    };
    let avg =
        Simd::<i16, K>::splat(((sum + ((W * h) as u32 >> 1)) >> (W * h).trailing_zeros()) as i16);
    for c in ac.as_chunks_mut::<K>().0 {
        *c = (Simd::from_array(*c) - avg).to_array();
    }
}

/// `dsp::ipred::cfl_pred`.
pub fn cfl_pred<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    ac: &[i16],
    w: usize,
    h: usize,
    alpha: i32,
    bd: u32,
) {
    // The decoder uses `u8` pixels iff the bit depth is 8. The tests also
    // run `u16` pixels at bit depth 8, which every kernel supports.
    debug_assert!(
        !(std::mem::size_of::<T>() == 1 && BD > 8)
            && (cfg!(test) || !(std::mem::size_of::<T>() == 2 && BD == 8))
    );
    debug_assert!((-16..=16).contains(&alpha) && bd == BD);
    match w {
        4 => cfl_pred_w::<T, BD, 4, 4>(dst, stride, ac, h, alpha),
        8 => cfl_pred_w::<T, BD, 8, 8>(dst, stride, ac, h, alpha),
        16 => cfl_pred_w::<T, BD, 16, 16>(dst, stride, ac, h, alpha),
        32 => cfl_pred_w::<T, BD, 32, 16>(dst, stride, ac, h, alpha),
        w => panic!("invalid cfl width {w}"),
    }
}

/// `Round2Signed(alpha * ac, 6)` is `(p + (p < 0 ? -1 : 0) + 32) >> 6`
/// with `p = alpha * ac`. For 8-bit `|p| <= 16 * 8 * 255` fits in `i16`.
/// For 10/12-bit (`|ac| <= 8 * 4095 = 32760`), signing `a` by `alpha` and
/// splitting `a = 64 * (a >> 6) + (a & 63)` keeps the entire computation
/// in `i16` lanes.
#[inline(always)]
fn cfl_pred_w<T: Pixel, const BD: u32, const W: usize, const K: usize>(
    dst: &mut [T],
    stride: usize,
    ac: &[i16],
    h: usize,
    alpha: i32,
) {
    let (dst, stride, h4) = check_block4::<T, W>(dst, stride, h);
    let max = Simd::<i16, K>::splat(((1i32 << BD) - 1) as i16);
    let zero = Simd::<i16, K>::splat(0);
    let al = Simd::<i16, K>::splat(alpha.abs() as i16);
    let al_signed = Simd::<i16, K>::splat(alpha as i16);
    let neg = Simd::<i16, K>::splat((alpha >> 31) as i16);
    let c32 = Simd::<i16, K>::splat(32);
    let c63 = Simd::<i16, K>::splat(63);
    let ac4 = ac[..W * h4 * 4].as_chunks::<W>().0.as_chunks::<4>().0;
    visit_row4::<T, W>(dst, stride, h, |i4, r0, r1, r2, r3| {
        let acr4 = &ac4[i4];
        for (row, acr) in [r0, r1, r2, r3].into_iter().zip(acr4) {
            for (d, a) in row
                .as_chunks_mut::<K>()
                .0
                .iter_mut()
                .zip(acr.as_chunks::<K>().0)
            {
                let a = Simd::<i16, K>::from_array(*a);
                let dc = load_i16::<T, K>(d);
                let p = if BD == 8 {
                    let p = a * al_signed;
                    (p + (p >> 15) + c32) >> 6
                } else {
                    let a = (a ^ neg) - neg;
                    (a >> 6) * al + (((a & c63) * al + (a >> 15) + c32) >> 6)
                };
                store_i16(d, (dc + p).simd_clamp(zero, max));
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Bit depth specialized entry points (selected by `Dsp::simd(bd)`).
// `BD` is 8, 10 or 12 and always equals the runtime `bd` argument; the
// pixel / intermediate ranges of `BD` may be relied upon.
// ---------------------------------------------------------------------------

pub fn predict_intra_bd<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    edges: &IntraEdges<T>,
    p: &IntraPredParams,
) {
    // The decoder uses `u8` pixels iff the bit depth is 8. The tests also
    // run `u16` pixels at bit depth 8, which every kernel supports.
    debug_assert!(
        !(std::mem::size_of::<T>() == 1 && BD > 8)
            && (cfg!(test) || !(std::mem::size_of::<T>() == 2 && BD == 8))
    );
    debug_assert_eq!(p.bd, BD);
    // Directional modes first, as a tail call without a stack frame (the
    // other modes need many registers for their inlined kernels).
    if p.filter_intra_mode.is_none() && (V_PRED..=D67_PRED).contains(&p.mode) {
        let p_angle = MODE_TO_ANGLE[p.mode as usize] + p.angle_delta * ANGLE_STEP;
        if p_angle != 90 && p_angle != 180 {
            predict_directional::<T, BD>(dst, stride, edges, p, p_angle);
            return;
        }
    }
    predict_intra_other::<T, BD>(dst, stride, edges, p);
}

/// [`predict_intra_bd`] for the non-directional modes and pAngle 90 / 180.
#[inline(never)]
fn predict_intra_other<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    edges: &IntraEdges<T>,
    p: &IntraPredParams,
) {
    let (w, h) = (p.w, p.h);
    debug_assert!(w.is_power_of_two() && (4..=64).contains(&w));
    debug_assert!(h.is_power_of_two() && (4..=64).contains(&h));
    debug_assert!(stride >= w && dst.len() >= (h - 1) * stride + w);
    let above = &edges.above[..];
    let left = &edges.left[..];
    // Same dispatch as `dsp::ipred::predict_intra_with`, with direct
    // calls instead of a kernel table. Heavy modes are `#[inline(never)]`
    // so fast DC / V / H modes avoid saving vector/callee-saved registers.
    if let Some(fm) = p.filter_intra_mode {
        ipred_filter::<T, BD>(
            dst,
            stride,
            edges.top_left,
            above,
            left,
            w,
            h,
            fm as usize,
            BD,
        );
        return;
    }
    match p.mode {
        DC_PRED => {
            let v = match (p.have_left, p.have_above) {
                (true, true) => by_size!(w, h, dc_both_block::<T, BD>(above, left)),
                (true, false) => dc_edge_val::<T, BD>(left, h),
                (false, true) => dc_edge_val::<T, BD>(above, w),
                (false, false) => T::from_i32(1 << (BD - 1)),
            };
            by_width!(w, dc_fill::<T>(dst, stride, h, v));
        }
        V_PRED..=D67_PRED => {
            let p_angle = MODE_TO_ANGLE[p.mode as usize] + p.angle_delta * ANGLE_STEP;
            if p_angle == 90 {
                ipred_v::<T, BD>(dst, stride, above, w, h);
            } else if p_angle == 180 {
                ipred_h::<T, BD>(dst, stride, left, w, h);
            } else {
                predict_directional::<T, BD>(dst, stride, edges, p, p_angle);
            }
        }
        SMOOTH_PRED => ipred_smooth::<T, BD>(dst, stride, above, left, w, h),
        SMOOTH_V_PRED => ipred_smooth_v::<T, BD>(dst, stride, above, left, w, h),
        SMOOTH_H_PRED => ipred_smooth_h::<T, BD>(dst, stride, above, left, w, h),
        PAETH_PRED => ipred_paeth::<T, BD>(dst, stride, edges.top_left, above, left, w, h),
        m => panic!("invalid intra mode {m}"),
    }
}

/// Prepares one directional edge buffer (`above` or `left`) of up to 8 samples.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn prepare_edge8<T: Pixel, const BD: u32, const L: usize>(
    buf: &mut [T; L],
    src: &[T; MAX_EDGE],
    corner: T,
    w: usize,
    h: usize,
    filter: bool,
    ft: bool,
    have: bool,
    delta: i32,
    len: i32,
    up_px: usize,
) -> bool {
    if !filter {
        fill_edge8::<T, L>(buf, src, corner, 0, 0);
        return false;
    }
    let up = use_upsample(w, h, ft, delta);
    if up {
        debug_assert!(!have || edge_filter_strength(w, h, ft, delta) == 0);
        upsample_edge8::<T, BD, L>(buf, corner, src, up_px);
    } else {
        let strength = if have { edge_filter_strength(w, h, ft, delta) } else { 0 };
        fill_edge8::<T, L>(buf, src, corner, len + 1, strength);
    }
    up
}

/// Prepares one directional edge buffer (`above` or `left`).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn prepare_edge<T: Pixel, const BD: u32, const L: usize>(
    buf: &mut [T; L],
    src: &[T; MAX_EDGE],
    corner: T,
    w: usize,
    h: usize,
    filter: bool,
    ft: bool,
    have: bool,
    delta: i32,
    len: i32,
    up_px: usize,
) -> bool {
    let up = filter && use_upsample(w, h, ft, delta);
    if up {
        // The edge filter strength is 0 (and n <= 16, so up_px <= 8 and no
        // corner filter) whenever upsampling is selected.
        debug_assert!(!have || edge_filter_strength(w, h, ft, delta) == 0);
        upsample_edge8::<T, BD, L>(buf, corner, src, up_px);
    } else {
        let strength = if filter && have { edge_filter_strength(w, h, ft, delta) } else { 0 };
        if up_px <= 8 {
            fill_edge8::<T, L>(buf, src, corner, len + 1, strength);
        } else {
            fill_edge::<T, L>(buf, src, corner, up_px, len + 1, strength);
        }
    }
    up
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn predict_z1_buffered<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    edges: &IntraEdges<T>,
    w: usize,
    h: usize,
    dx: i32,
    max_w: i32,
    up: bool,
    strength: u32,
) {
    let n = w + h;
    let mut above = [T::default(); EDGE_BUF_LEN];
    if up {
        if n <= 8 {
            upsample_edge8::<T, BD, EDGE_BUF_LEN>(&mut above, edges.top_left, &edges.above, n);
        } else {
            upsample_edge::<T, BD, EDGE_BUF_LEN>(&mut above, edges.top_left, &edges.above, n);
        }
    } else if n <= 8 {
        fill_edge8::<T, EDGE_BUF_LEN>(
            &mut above,
            &edges.above,
            edges.top_left,
            max_w + h as i32 + 1,
            strength,
        );
    } else {
        fill_edge::<T, EDGE_BUF_LEN>(
            &mut above,
            &edges.above,
            edges.top_left,
            n,
            max_w + h as i32 + 1,
            strength,
        );
    }
    ipred_z1::<T, BD>(dst, stride, &above[EDGE_PAD..], w, h, dx, up);
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn predict_z3_buffered<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    edges: &IntraEdges<T>,
    w: usize,
    h: usize,
    dy: i32,
    max_h: i32,
    up: bool,
    strength: u32,
) {
    let n = w + h;
    let mut left = [T::default(); EDGE_BUF_LEN];
    if up {
        if n <= 8 {
            upsample_edge8::<T, BD, EDGE_BUF_LEN>(&mut left, edges.top_left, &edges.left, n);
        } else {
            upsample_edge::<T, BD, EDGE_BUF_LEN>(&mut left, edges.top_left, &edges.left, n);
        }
    } else if n <= 8 {
        fill_edge8::<T, EDGE_BUF_LEN>(
            &mut left,
            &edges.left,
            edges.top_left,
            max_h + w as i32 + 1,
            strength,
        );
    } else {
        fill_edge::<T, EDGE_BUF_LEN>(
            &mut left,
            &edges.left,
            edges.top_left,
            n,
            max_h + w as i32 + 1,
            strength,
        );
    }
    ipred_z3::<T, BD>(dst, stride, &left[EDGE_PAD..], w, h, dy, up);
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn predict_z2_buffered<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    edges: &IntraEdges<T>,
    p: &IntraPredParams,
    p_angle: i32,
    dx: i32,
    dy: i32,
    need_a: bool,
    need_l: bool,
) {
    let (w, h) = (p.w, p.h);
    let n = w + h;
    let filter = p.enable_intra_edge_filter;
    let ft = p.smooth_filter_type;
    let max_w = (w as i32).min(p.max_x - p.x + 1);
    let max_h = (h as i32).min(p.max_y - p.y + 1);
    let corner = if filter && n >= 24 {
        T::from_i32(filter_corner(edges.left[0], edges.top_left, edges.above[0]))
    } else {
        edges.top_left
    };
    if use_z2_small16::<T, BD>(w, h, false) {
        let mut above = [T::default(); Z2_SMALL_LEN];
        let mut left = [T::default(); Z2_SMALL_LEN];
        let ua = if need_a {
            prepare_edge8::<T, BD, Z2_SMALL_LEN>(
                &mut above,
                &edges.above,
                corner,
                w,
                h,
                filter,
                ft,
                p.have_above,
                p_angle - 90,
                max_w,
                w,
            )
        } else {
            false
        };
        let ul = if !need_l {
            false
        } else if h <= 8 {
            prepare_edge8::<T, BD, Z2_SMALL_LEN>(
                &mut left,
                &edges.left,
                corner,
                w,
                h,
                filter,
                ft,
                p.have_left,
                p_angle - 180,
                max_h,
                h,
            )
        } else {
            let strength = if filter && p.have_left {
                edge_filter_strength(w, h, ft, p_angle - 180)
            } else {
                0
            };
            fill_edge16::<T, Z2_SMALL_LEN>(&mut left, &edges.left, corner, max_h + 1, strength);
            false
        };
        z2_small16::<T, Z2_SMALL_LEN>(dst, stride, &above, &left, w, h, dx, dy, ua, ul);
        return;
    }
    let mut above = [T::default(); EDGE_BUF_LEN];
    let mut left = [T::default(); EDGE_BUF_LEN];
    let ua = if need_a {
        prepare_edge::<T, BD, EDGE_BUF_LEN>(
            &mut above,
            &edges.above,
            corner,
            w,
            h,
            filter,
            ft,
            p.have_above,
            p_angle - 90,
            max_w,
            w,
        )
    } else {
        false
    };
    let ul = if need_l {
        prepare_edge::<T, BD, EDGE_BUF_LEN>(
            &mut left,
            &edges.left,
            corner,
            w,
            h,
            filter,
            ft,
            p.have_left,
            p_angle - 180,
            max_h,
            h,
        )
    } else {
        false
    };
    ipred_z2::<T, BD>(dst, stride, &above, &left, w, h, dx, dy, ua, ul);
}

/// [`DR_INTRA_DERIVATIVE`] padded to 128 entries, indexed by angles masked
/// to 7 bits (no bounds checks).
const DR_DERIV128: [i32; 128] = {
    let mut t = [0; 128];
    let mut i = 0;
    while i < DR_INTRA_DERIVATIVE.len() {
        t[i] = DR_INTRA_DERIVATIVE[i];
        i += 1;
    }
    t
};

/// `dsp::ipred::predict_directional` with SIMD edge filtering. Small
/// non-upsampled 10-bit z2 blocks (the most common directional blocks)
/// go straight to their register kernels, which filter the edges
/// themselves, as a tail call without a stack frame; every other block
/// goes to [`predict_directional_other`].
#[inline(never)]
fn predict_directional<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    edges: &IntraEdges<T>,
    p: &IntraPredParams,
    p_angle: i32,
) {
    let (w, h) = (p.w, p.h);
    if BD <= 10 && !is_u8::<T>() && w <= 8 && h <= 8 && p_angle > 90 && p_angle < 180 {
        let n = w + h;
        let filter = p.enable_intra_edge_filter;
        let ft = p.smooth_filter_type;
        let da = (p_angle - 90) as usize & 127;
        let dl = (180 - p_angle) as usize & 127;
        let dx = DR_DERIV128[dl];
        let dy = DR_DERIV128[da];
        // Lane (i, j) selects the above edge iff (j << 6) - (i + 1) * dx >= -64.
        let need_a = dx <= (w as i32) * 64;
        let need_l = (h as i32) * dx > 64;
        let can_up = filter && n <= if ft { 8 } else { 16 };
        if !(can_up && ((need_a && da < 40) || (need_l && dl < 40))) {
            // w + h <= 16 < 24, so the corner filter is never active here.
            // n <= 16: strength class 0 or 1.
            let lut = &STRENGTH_LUT[ft as usize][(n - 1) >> 3 & 1];
            let max_w = (w as i32).min(p.max_x - p.x + 1);
            let max_h = (h as i32).min(p.max_y - p.y + 1);
            let s_a = if filter && p.have_above { lut[da] as u32 } else { 0 };
            let s_l = if filter && p.have_left { lut[dl] as u32 } else { 0 };
            let packed = z2_small_pack(dx, dy, max_w + 1, max_h + 1, s_a, s_l);
            z2_small16_reg::<T>(dst, stride, edges, w, h, packed);
            return;
        }
    }
    predict_directional_other::<T, BD>(dst, stride, edges, p, p_angle);
}

/// [`predict_directional`] for every block but the small register z2
/// ones. Only the edge buffer(s) read by the selected kernel are prepared.
#[inline(never)]
fn predict_directional_other<T: Pixel, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    edges: &IntraEdges<T>,
    p: &IntraPredParams,
    p_angle: i32,
) {
    let (w, h) = (p.w, p.h);
    let n = w + h;
    let filter = p.enable_intra_edge_filter;
    let ft = p.smooth_filter_type;
    let up_max = if ft { 8 } else { 16 };
    let lut = &STRENGTH_LUT[ft as usize][((n.wrapping_sub(1)) >> 3).min(4)];
    let max_w = (w as i32).min(p.max_x - p.x + 1);
    let max_h = (h as i32).min(p.max_y - p.y + 1);
    if p_angle < 90 {
        let d = (90 - p_angle) as usize & 127;
        let dx = DR_DERIV128[p_angle as usize & 127];
        let up = filter && d < 40 && n <= up_max;
        let strength = if !up && filter && p.have_above { lut[d] as u32 } else { 0 };
        if !up && strength == 0 {
            ipred_z1::<T, BD>(dst, stride, &edges.above, w, h, dx, false);
        } else {
            predict_z1_buffered::<T, BD>(dst, stride, edges, w, h, dx, max_w, up, strength);
        }
        return;
    }
    if p_angle > 180 {
        let d = (p_angle - 180) as usize & 127;
        let dy = DR_DERIV128[(270 - p_angle) as usize & 127];
        let up = filter && d < 40 && n <= up_max;
        let strength = if !up && filter && p.have_left { lut[d] as u32 } else { 0 };
        if !up && strength == 0 {
            ipred_z3::<T, BD>(dst, stride, &edges.left, w, h, dy, false);
        } else {
            predict_z3_buffered::<T, BD>(dst, stride, edges, w, h, dy, max_h, up, strength);
        }
        return;
    }
    let da = (p_angle - 90) as usize & 127;
    let dl = (180 - p_angle) as usize & 127;
    let dx = DR_DERIV128[dl];
    let dy = DR_DERIV128[da];
    // Lane (i, j) selects the above edge iff (j << 6) - (i + 1) * dx >= -64.
    let need_a = dx <= (w as i32) * 64;
    let need_l = (h as i32) * dx > 64;
    predict_z2_buffered::<T, BD>(dst, stride, edges, p, p_angle, dx, dy, need_a, need_l);
}

const EDGE_FILTER_W: [[u16; 3]; 4] = [[0, 0, 16], [0, 4, 8], [0, 5, 6], [2, 4, 4]];

/// Filters up to 8 edge samples in a `u16x8` register (`prev` is the
/// splatted corner sample).
#[inline(always)]
fn filter_edge8_vec<T: Pixel>(src: &[T; MAX_EDGE], prev: u16x8, sz: i32, strength: u32) -> u16x8 {
    const M: usize = 8;
    debug_assert!(sz <= M as i32 + 1);
    let raw = load_u16::<T, M>(src);
    if strength == 0 || sz <= 1 {
        return raw;
    }
    let lane = i16x8::from_array([0, 1, 2, 3, 4, 5, 6, 7]);
    let last = u16x8::splat(src[(sz as usize - 2) & 7].to_u16());
    let cur = lane.simd_gt(i16x8::splat(sz as i16 - 2)).select(last, raw);
    let [w0, w1, w2] = EDGE_FILTER_W[strength as usize & 3].map(u16x8::splat);
    let t0 = simd_swizzle!(prev, cur, [6, 7, 8, 9, 10, 11, 12, 13]);
    let t1 = simd_swizzle!(prev, cur, [7, 8, 9, 10, 11, 12, 13, 14]);
    let t3 = simd_swizzle!(cur, last, [1, 2, 3, 4, 5, 6, 7, 8]);
    let t4 = simd_swizzle!(cur, last, [2, 3, 4, 5, 6, 7, 8, 9]);
    let f = ((t0 + t4) * w0 + (t1 + t3) * w1 + cur * w2 + u16x8::splat(8)) >> 4;
    lane.simd_lt(i16x8::splat(sz as i16 - 1)).select(f, raw)
}

/// 8-lane specialization of [`fill_edge`] for `n <= 8` (`sz <= 9`).
#[inline(always)]
fn fill_edge8<T: Pixel, const L: usize>(
    buf: &mut [T; L],
    src: &[T; MAX_EDGE],
    corner: T,
    sz: i32,
    strength: u32,
) {
    const M: usize = 8;
    const { assert!(EDGE_PAD >= M && EDGE_PAD + M <= L) };
    let prev = u16x8::splat(corner.to_u16());
    store_u16::<T, M>(&mut buf[EDGE_PAD - M..], prev);
    store_u16::<T, M>(
        &mut buf[EDGE_PAD..],
        filter_edge8_vec(src, prev, sz, strength),
    );
}

/// 16-lane specialization of [`fill_edge`] for `n <= 16` (`sz <= 17`).
#[inline(always)]
fn fill_edge16<T: Pixel, const L: usize>(
    buf: &mut [T; L],
    src: &[T; MAX_EDGE],
    corner: T,
    sz: i32,
    strength: u32,
) {
    const N: usize = 16;
    const { assert!(EDGE_PAD >= 8 && EDGE_PAD + N <= L) };
    debug_assert!(sz <= N as i32 + 1);
    store_u16::<T, 8>(&mut buf[EDGE_PAD - 8..], u16x8::splat(corner.to_u16()));
    if strength == 0 || sz <= 1 {
        buf[EDGE_PAD..EDGE_PAD + N].copy_from_slice(&src[..N]);
        return;
    }
    let sz_u = sz as usize;
    if sz_u <= 9 {
        fill_edge8::<T, L>(buf, src, corner, sz, strength);
        buf[EDGE_PAD + 8..EDGE_PAD + N].copy_from_slice(&src[8..N]);
        return;
    }
    let lane = Simd::<i16, N>::from_array(std::array::from_fn(|l| l as i16));
    let last = Simd::<u16, N>::splat(src[sz_u - 2].to_u16());
    let prev0 = Simd::<u16, N>::splat(corner.to_u16());
    let raw0 = load_u16::<T, N>(src);
    let cur0 = lane.simd_gt(Simd::splat(sz as i16 - 2)).select(last, raw0);
    let [w0, w1, w2] = EDGE_FILTER_W[strength as usize & 3].map(Simd::<u16, N>::splat);
    let r = Simd::<u16, N>::splat(8);
    let t0 = simd_swizzle!(
        prev0,
        cur0,
        [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    );
    let t1 = simd_swizzle!(
        prev0,
        cur0,
        [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    );
    let t3 = simd_swizzle!(
        cur0,
        last,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    );
    let t4 = simd_swizzle!(
        cur0,
        last,
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    );
    let f = ((t0 + t4) * w0 + (t1 + t3) * w1 + cur0 * w2 + r) >> 4;
    let v = lane.simd_lt(Simd::splat(sz as i16 - 1)).select(f, raw0);
    store_u16::<T, N>(&mut buf[EDGE_PAD..], v);
}

/// 8-lane specialization of [`upsample_edge`] for `num_px <= 8`.
#[inline(always)]
fn upsample_edge8<T: Pixel, const BD: u32, const L: usize>(
    buf: &mut [T; L],
    corner: T,
    src: &[T; MAX_EDGE],
    num_px: usize,
) {
    const N: usize = 8;
    const { assert!(EDGE_PAD + 2 * N <= L) };
    debug_assert!((1..=N).contains(&num_px));
    let lane = i16x8::from_array([0, 1, 2, 3, 4, 5, 6, 7]);
    let last = i16x8::splat(src[num_px - 1].to_i32() as i16);
    let c = i16x8::splat(corner.to_i32() as i16);
    let s = lane
        .simd_ge(i16x8::splat(num_px as i16))
        .select(last, load_i16::<T, N>(src));
    let d0 = simd_swizzle!(c, s, [6, 7, 8, 9, 10, 11, 12, 13]);
    let d1 = simd_swizzle!(c, s, [7, 8, 9, 10, 11, 12, 13, 14]);
    let d3 = simd_swizzle!(s, last, [1, 2, 3, 4, 5, 6, 7, 8]);
    let max = i16x8::splat(((1i32 << BD) - 1) as i16);
    let filt = if BD <= 10 {
        ((i16x8::splat(9) * (d1 + s) - d0 - d3 + i16x8::splat(8)) >> 4)
            .simd_clamp(i16x8::splat(0), max)
    } else {
        let w = |v: i16x8| v.cast::<i32>();
        ((Simd::splat(9) * (w(d1) + w(s)) - w(d0) - w(d3) + Simd::splat(8)) >> 4)
            .simd_clamp(Simd::splat(0), Simd::splat((1 << BD) - 1))
            .cast::<i16>()
    };
    let (lo, hi) = filt.interleave(s);
    buf[EDGE_PAD - 2] = corner;
    store_i16::<T, N>(&mut buf[EDGE_PAD - 1..], lo);
    store_i16::<T, N>(&mut buf[EDGE_PAD - 1 + N..], hi);
}

/// Fills the directional edge buffer `buf` (spec `edge[k]`, i.e.
/// `AboveRow[k - 1]` / `LeftCol[k - 1]`, is `buf[EDGE_PAD - 1 + k]`)
/// from `corner` (`edge[0]`) and `src[..n]`, applying the intra edge
/// filter (`dsp::ipred::filter_edge`) with `strength` to `edge[1..sz]`.
///
/// Works on `u16` lanes (the 16-weight sums are at most `16 * 4095`)
/// loaded from `src` only: the shifted tap vectors are built from
/// neighbouring vectors in registers, and lanes past the filtered range
/// keep the unfiltered samples. Samples of `src` past `n` (up to the
/// next multiple of 16) are copied too; the kernels never read them.
#[inline(always)]
fn fill_edge<T: Pixel, const L: usize>(
    buf: &mut [T; L],
    src: &[T; MAX_EDGE],
    corner: T,
    n: usize,
    sz: i32,
    strength: u32,
) {
    // 16 lanes: for u8 one 16 byte store per chunk, which the 16 byte
    // table loads of the small block kernels can forward from.
    const N: usize = 16;
    const { assert!(EDGE_PAD >= 8 && EDGE_PAD + N <= L && MAX_EDGE % N == 0) };
    debug_assert!(
        n <= MAX_EDGE && n % 4 == 0 && sz <= n as i32 + 1 && EDGE_PAD + n.next_multiple_of(N) <= L
    );
    let prev = u16x8::splat(corner.to_u16());
    store_u16::<T, 8>(&mut buf[EDGE_PAD - 8..], prev);
    if strength == 0 || sz <= 1 {
        let mut k = 0;
        while k < n {
            buf[EDGE_PAD + k..EDGE_PAD + k + N].copy_from_slice(&src[k..k + N]);
            k += N;
        }
        return;
    }
    let sz_u = sz as usize;
    if sz_u <= 9 {
        // Every filtered sample is in the first 8 lanes (blocks up to 8
        // samples along the edge): filter 8 lanes only, the samples from
        // index 8 on are copied unchanged.
        const M: usize = 8;
        store_u16::<T, M>(
            &mut buf[EDGE_PAD..],
            filter_edge8_vec(src, prev, sz, strength),
        );
        buf[EDGE_PAD + M..EDGE_PAD + N].copy_from_slice(&src[M..N]);
        let mut k = N;
        while k < n {
            buf[EDGE_PAD + k..EDGE_PAD + k + N].copy_from_slice(&src[k..k + N]);
            k += N;
        }
        return;
    }
    let lane = Simd::<i16, N>::from_array(std::array::from_fn(|l| l as i16));
    let last = Simd::<u16, N>::splat(src[sz_u - 2].to_u16());
    let prev0 = Simd::<u16, N>::splat(corner.to_u16());
    let raw0 = load_u16::<T, N>(src);
    let cur0 = lane.simd_gt(Simd::splat(sz as i16 - 2)).select(last, raw0);
    if sz_u <= N + 1 {
        // Every filtered sample is in the first 16 lanes, and all lanes at
        // index >= 16 are clamped to `last`.
        let [w0, w1, w2] = EDGE_FILTER_W[strength as usize & 3].map(Simd::<u16, N>::splat);
        let r = Simd::<u16, N>::splat(8);
        let t0 = simd_swizzle!(
            prev0,
            cur0,
            [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        );
        let t1 = simd_swizzle!(
            prev0,
            cur0,
            [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        );
        let t3 = simd_swizzle!(
            cur0,
            last,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
        let t4 = simd_swizzle!(
            cur0,
            last,
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        );
        let f = ((t0 + t4) * w0 + (t1 + t3) * w1 + cur0 * w2 + r) >> 4;
        let v = lane.simd_lt(Simd::splat(sz as i16 - 1)).select(f, raw0);
        store_u16::<T, N>(&mut buf[EDGE_PAD..], v);
        let mut k = N;
        while k < n {
            buf[EDGE_PAD + k..EDGE_PAD + k + N].copy_from_slice(&src[k..k + N]);
            k += N;
        }
        return;
    }
    fill_edge_large::<T, L>(buf, src, n, sz, strength, prev0, raw0, cur0, last);
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn fill_edge_large<T: Pixel, const L: usize>(
    buf: &mut [T; L],
    src: &[T; MAX_EDGE],
    n: usize,
    sz: i32,
    strength: u32,
    mut prev: Simd<u16, 16>,
    mut raw: Simd<u16, 16>,
    mut cur: Simd<u16, 16>,
    last: Simd<u16, 16>,
) {
    const N: usize = 16;
    let sz_u = sz as usize;
    let lane = Simd::<i16, N>::from_array(std::array::from_fn(|l| l as i16));
    let [w0, w1, w2] = EDGE_FILTER_W[strength as usize & 3].map(Simd::<u16, N>::splat);
    let r = Simd::<u16, N>::splat(8);
    let mut k = 0;
    while k < sz_u - 1 {
        let (next_raw, next) = if k + 2 * N > MAX_EDGE {
            (last, last)
        } else {
            let nr = load_u16::<T, N>(&src[k + N..]);
            let nc = (lane + Simd::splat((k + N) as i16))
                .simd_gt(Simd::splat(sz as i16 - 2))
                .select(last, nr);
            (nr, nc)
        };
        let t0 = simd_swizzle!(
            prev,
            cur,
            [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        );
        let t1 = simd_swizzle!(
            prev,
            cur,
            [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        );
        let t3 = simd_swizzle!(
            cur,
            next,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
        let t4 = simd_swizzle!(
            cur,
            next,
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        );
        let f = ((t0 + t4) * w0 + (t1 + t3) * w1 + cur * w2 + r) >> 4;
        let v = (lane + Simd::splat(k as i16))
            .simd_lt(Simd::splat(sz as i16 - 1))
            .select(f, raw);
        store_u16::<T, N>(&mut buf[EDGE_PAD + k..], v);
        (prev, cur, raw) = (cur, next, next_raw);
        k += N;
    }
    while k < n {
        buf[EDGE_PAD + k..EDGE_PAD + k + N].copy_from_slice(&src[k..k + N]);
        k += N;
    }
}

/// `dsp::ipred::upsample_edge` of the unfiltered edge `corner`,
/// `src[..num_px]` (upsampling is only selected when the edge filter
/// strength is 0, see the `upsample_implies_no_filter` test) into the
/// directional edge buffer `buf`. Writes spec indices
/// `-2..=2 * numPx - 2` (and some more that the kernels never read).
#[inline(always)]
fn upsample_edge<T: Pixel, const BD: u32, const L: usize>(
    buf: &mut [T; L],
    corner: T,
    src: &[T; MAX_EDGE],
    num_px: usize,
) {
    const N: usize = MAX_UPSAMPLE_PX;
    const { assert!(EDGE_PAD + 2 * N <= L && N <= MAX_EDGE) };
    debug_assert!((1..=N).contains(&num_px));
    let lane = Simd::<i16, N>::from_array(std::array::from_fn(|l| l as i16));
    let last = Simd::<i16, N>::splat(src[num_px - 1].to_i32() as i16);
    let c = Simd::<i16, N>::splat(corner.to_i32() as i16);
    // s[i] = src[min(i, numPx - 1)], s[-1] = s[-2] = corner; tap k of
    // output i is s[i - 2 + k] (dup[i + k] of the spec).
    let s = lane
        .simd_ge(Simd::splat(num_px as i16))
        .select(last, load_i16::<T, N>(src));
    let d0 = simd_swizzle!(
        c,
        s,
        [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    );
    let d1 = simd_swizzle!(
        c,
        s,
        [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    );
    let d3 = simd_swizzle!(
        s,
        last,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    );
    let max = Simd::splat(((1i32 << BD) - 1) as i16);
    let filt = if BD <= 10 {
        // 9 * (d1 + s) is at most 18 * 1023.
        ((Simd::splat(9) * (d1 + s) - d0 - d3 + Simd::splat(8)) >> 4)
            .simd_clamp(Simd::splat(0), max)
    } else {
        let w = |v: Simd<i16, N>| v.cast::<i32>();
        ((Simd::splat(9) * (w(d1) + w(s)) - w(d0) - w(d3) + Simd::splat(8)) >> 4)
            .simd_clamp(Simd::splat(0), Simd::splat((1 << BD) - 1))
            .cast::<i16>()
    };
    // Spec index 2 * i - 1 is filt[i], 2 * i is s[i]; -2 is the corner.
    let (lo, hi) = filt.interleave(s);
    buf[EDGE_PAD - 2] = corner;
    store_i16::<T, N>(&mut buf[EDGE_PAD - 1..], lo);
    store_i16::<T, N>(&mut buf[EDGE_PAD - 1 + N..], hi);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::av1::dsp::ipred::IpredKernels;

    /// Small deterministic PRNG (xorshift64*).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
        }
        fn below(&mut self, n: u32) -> u32 {
            self.next() % n
        }
    }

    /// All (w, h) transform dimensions used by AV1.
    fn sizes() -> Vec<(usize, usize)> {
        let mut v = Vec::new();
        for lw in 2..=6i32 {
            for lh in 2..=6i32 {
                if (lw - lh).abs() <= 2 {
                    v.push((1usize << lw, 1usize << lh));
                }
            }
        }
        v
    }

    /// Random edge samples of one of several flavours.
    fn gen<T: Pixel>(rng: &mut Rng, bd: u32, out: &mut [T]) {
        let max = (1u32 << bd) - 1;
        let flavour = rng.below(4);
        let mut prev = rng.below(max + 1) as i32;
        for o in out.iter_mut() {
            prev = match flavour {
                0 => rng.below(max + 1) as i32,
                1 => [0, max as i32][rng.below(2) as usize],
                2 => (prev + rng.below(33) as i32 - 16).clamp(0, max as i32),
                _ => (prev + rng.below(3) as i32 - 1).clamp(0, max as i32),
            };
            *o = T::from_i32(prev);
        }
    }

    /// Edge buffers for one test case.
    struct Case<T: Pixel> {
        tl: T,
        above: [T; EDGE_BUF_LEN],
        left: [T; EDGE_BUF_LEN],
    }

    impl<T: Pixel> Case<T> {
        fn new(rng: &mut Rng, bd: u32) -> Self {
            let mut c = Self {
                tl: T::default(),
                above: [T::default(); EDGE_BUF_LEN],
                left: [T::default(); EDGE_BUF_LEN],
            };
            gen(rng, bd, &mut c.above);
            gen(rng, bd, &mut c.left);
            c.tl = c.above[EDGE_PAD - 1];
            c
        }
        /// `IntraEdges::above` / `left` view (spec index 0 first).
        fn a(&self) -> &[T] {
            &self.above[EDGE_PAD..EDGE_PAD + MAX_EDGE]
        }
        fn l(&self) -> &[T] {
            &self.left[EDGE_PAD..EDGE_PAD + MAX_EDGE]
        }
    }

    /// Runs `f` with the scalar and the SIMD table on a `w x h` block with
    /// a random stride and compares the outputs (including that nothing
    /// outside the block is written).

    fn get_table<T: Pixel>(bd: u32) -> &'static IpredKernels<T> {
        match bd {
            8 => &SimdKernels::<T, 8>::TABLE,
            10 => &SimdKernels::<T, 10>::TABLE,
            _ => &SimdKernels::<T, 12>::TABLE,
        }
    }
    fn cmp<T: Pixel>(
        rng: &mut Rng,
        bd: u32,
        w: usize,
        h: usize,
        what: &str,
        f: impl Fn(&IpredKernels<T>, &mut [T], usize),
    ) {
        let stride = w + rng.below(9) as usize;
        let sentinel = T::from_i32(0x5a);
        let mut d0 = vec![sentinel; stride * (h - 1) + w];
        let mut d1 = d0.clone();
        f(&IpredKernels::SCALAR, &mut d0, stride);
        f(get_table::<T>(bd), &mut d1, stride);
        assert!(d0 == d1, "{what} mismatch {w}x{h}");
    }

    fn run<T: Pixel>(bd: u32, seed: u64, trials: usize) {
        let mut rng = Rng(seed);
        for (w, h) in sizes() {
            run_directional::<T>(&mut rng, bd, w, h, trials / 10);
            for _ in 0..trials {
                let c = Case::<T>::new(&mut rng, bd);
                let r = &mut rng;
                cmp(r, bd, w, h, "dc", |k, d, s| {
                    (k.dc)(d, s, c.a(), c.l(), w, h)
                });
                cmp(r, bd, w, h, "dc_top", |k, d, s| {
                    (k.dc_top)(d, s, c.a(), w, h)
                });
                cmp(r, bd, w, h, "dc_left", |k, d, s| {
                    (k.dc_left)(d, s, c.l(), w, h)
                });
                cmp::<T>(r, bd, w, h, "dc_128", |k, d, s| (k.dc_128)(d, s, w, h, bd));
                cmp(r, bd, w, h, "v", |k, d, s| (k.v)(d, s, c.a(), w, h));
                cmp(r, bd, w, h, "h", |k, d, s| (k.h)(d, s, c.l(), w, h));
                cmp(r, bd, w, h, "paeth", |k, d, s| {
                    (k.paeth)(d, s, c.tl, c.a(), c.l(), w, h)
                });
                cmp(r, bd, w, h, "smooth", |k, d, s| {
                    (k.smooth)(d, s, c.a(), c.l(), w, h)
                });
                cmp(r, bd, w, h, "smooth_v", |k, d, s| {
                    (k.smooth_v)(d, s, c.a(), c.l(), w, h)
                });
                cmp(r, bd, w, h, "smooth_h", |k, d, s| {
                    (k.smooth_h)(d, s, c.a(), c.l(), w, h)
                });
                // Filter intra is only used for blocks up to 32x32.
                if w <= 32 && h <= 32 {
                    for mode in 0..5 {
                        cmp(r, bd, w, h, "filter", |k, d, s| {
                            (k.filter)(d, s, c.tl, c.a(), c.l(), w, h, mode, bd)
                        });
                    }
                }
            }
        }
    }

    /// Directional kernels, for every angle and upsampling choice the
    /// decoder can produce (and more).
    fn run_directional<T: Pixel>(rng: &mut Rng, bd: u32, w: usize, h: usize, trials: usize) {
        // Upsampling is only selected for w + h <= 16 (spec 7.11.2.10).
        let ups: &[bool] = if w + h <= 16 { &[false, true] } else { &[false] };
        for p_angle in 1..90 {
            let dx = DR_INTRA_DERIVATIVE[p_angle];
            if dx == 0 {
                continue;
            }
            for &up in ups {
                for _ in 0..trials {
                    let c = Case::<T>::new(rng, bd);
                    let a = &c.above[EDGE_PAD..];
                    cmp(rng, bd, w, h, "z1", |k, d, s| (k.z1)(d, s, a, w, h, dx, up));
                }
            }
        }
        // z2 reads spec indices >= -(1 << upsample) for every angle and
        // upsampling combination (see dsp::ipred z2_left_index_bounds).
        for p_angle in 91..180 {
            let dx = DR_INTRA_DERIVATIVE[180 - p_angle];
            let dy = DR_INTRA_DERIVATIVE[p_angle - 90];
            if dx == 0 || dy == 0 {
                continue;
            }
            for (ua, ul) in [(false, false), (true, false), (false, true), (true, true)] {
                for _ in 0..trials {
                    let c = Case::<T>::new(rng, bd);
                    cmp(rng, bd, w, h, "z2", |k, d, s| {
                        (k.z2)(d, s, &c.above, &c.left, w, h, dx, dy, ua, ul)
                    });
                }
            }
        }
        // z3 angles whose reads stay inside the edge buffer (the decoder
        // uses 183..=212).
        for p_angle in 181..270 {
            let dy = DR_INTRA_DERIVATIVE[270 - p_angle];
            if dy == 0 {
                continue;
            }
            for &up in ups {
                let u = up as usize;
                if ((w as i32 * dy) >> (6 - u)) as usize + (h << u) >= MAX_EDGE + 16 {
                    continue;
                }
                for _ in 0..trials {
                    let c = Case::<T>::new(rng, bd);
                    let l = &c.left[EDGE_PAD..];
                    cmp(rng, bd, w, h, "z3", |k, d, s| (k.z3)(d, s, l, w, h, dy, up));
                }
            }
        }
    }

    /// Whole `predict_intra` (dispatch, edge filtering and upsampling,
    /// kernels) against the portable one, with random parameters.
    fn run_predict<T: Pixel, const BD: u32>(seed: u64, trials: usize) {
        let mut rng = Rng(seed);
        let mut e = IntraEdges::<T>::default();
        for (w, h) in sizes() {
            for mode in 0u8..=12 {
                for _ in 0..trials {
                    let c = Case::<T>::new(&mut rng, BD);
                    e.top_left = c.tl;
                    e.above.copy_from_slice(c.a());
                    e.left.copy_from_slice(c.l());
                    let flags = rng.below(16);
                    let filter = mode == 0 && w <= 32 && h <= 32 && rng.below(2) == 0;
                    let x = 4 * rng.below(64) as i32;
                    let y = 4 * rng.below(64) as i32;
                    let p = IntraPredParams {
                        mode,
                        angle_delta: rng.below(7) as i32 - 3,
                        filter_intra_mode: filter.then(|| rng.below(5) as u8),
                        have_left: flags & 1 != 0,
                        have_above: flags & 2 != 0,
                        enable_intra_edge_filter: flags & 4 != 0,
                        smooth_filter_type: flags & 8 != 0,
                        x,
                        y,
                        max_x: x + rng.below(2 * w as u32 + 2) as i32,
                        max_y: y + rng.below(2 * h as u32 + 2) as i32,
                        w,
                        h,
                        bd: BD,
                    };
                    let stride = w + rng.below(9) as usize;
                    let mut d0 = vec![T::from_i32(0x5a); stride * (h - 1) + w];
                    let mut d1 = d0.clone();
                    crate::decoder::av1::dsp::ipred::predict_intra(&mut d0, stride, &e, &p);
                    predict_intra_bd::<T, BD>(&mut d1, stride, &e, &p);
                    assert!(d0 == d1, "predict_intra mismatch {p:?}");
                    // Edge samples past edge_needs() must not matter.
                    let (na, nl) = crate::decoder::av1::dsp::ipred::edge_needs(&p);
                    let mut e2 = e.clone();
                    let max = (1u32 << BD) - 1;
                    for v in e2.above[na..].iter_mut().chain(&mut e2.left[nl..]) {
                        *v = T::from_i32(rng.below(max + 1) as i32);
                    }
                    let mut d2 = d0.clone();
                    let mut d3 = d0.clone();
                    crate::decoder::av1::dsp::ipred::predict_intra(&mut d2, stride, &e2, &p);
                    predict_intra_bd::<T, BD>(&mut d3, stride, &e2, &p);
                    assert!(d0 == d2 && d0 == d3, "edge_needs too small for {p:?}");
                }
            }
        }
    }

    #[test]
    fn predict_intra_matches_portable() {
        run_predict::<u8, 8>(0x5eed_0001, 100);
        run_predict::<u16, 8>(0x5eed_0002, 20);
        run_predict::<u16, 10>(0x5eed_0003, 100);
        run_predict::<u16, 12>(0x5eed_0004, 100);
    }

    #[test]
    fn matches_scalar_8bit() {
        run::<u8>(8, 0x1234_5678_9abc_def1, 200);
    }

    #[test]
    fn matches_scalar_8bit_in_u16() {
        run::<u16>(8, 0x0f0f_1234_5555_aaaa, 50);
    }

    #[test]
    fn matches_scalar_10bit() {
        run::<u16>(10, 0xdead_beef_cafe_f00d, 200);
    }

    #[test]
    fn matches_scalar_12bit() {
        run::<u16>(12, 0x0123_4567_89ab_cdef, 200);
    }

    /// Runs `f0` (portable) and `f1` (SIMD) on a `w x h` block with a
    /// random stride and compares the outputs (including that nothing
    /// outside the block is written).
    fn cmp2<T: Pixel>(
        rng: &mut Rng,
        w: usize,
        h: usize,
        what: &str,
        f0: impl Fn(&mut [T], usize),
        f1: impl Fn(&mut [T], usize),
    ) {
        let stride = w + rng.below(9) as usize;
        let mut d0 = vec![T::from_i32(0x5a); stride * (h - 1) + w];
        let mut d1 = d0.clone();
        f0(&mut d0, stride);
        f1(&mut d1, stride);
        assert!(d0 == d1, "{what} mismatch {w}x{h}");
    }

    /// The table lookup kernels of small 8-bit blocks, with the
    /// preconditions under which `ipred_z1` / `ipred_z2` / `ipred_z3`
    /// select them. They are only dispatched where `swizzle_dyn` is fast
    /// (not on x86), so they are called directly here to be tested on
    /// every host.
    #[test]
    fn tbl_kernels_match_scalar() {
        use crate::decoder::av1::dsp::ipred as p;
        let mut rng = Rng(0x7ab1_e5ed_0bad_cafe);
        for (w, h) in sizes() {
            if w > 8 || h > 16 {
                continue;
            }
            let ups: &[bool] = if w + h <= 16 { &[false, true] } else { &[false] };
            for _ in 0..8 {
                let c = Case::<u8>::new(&mut rng, 8);
                let a = &c.above[EDGE_PAD..];
                let l = &c.left[EDGE_PAD..];
                for p_angle in 1..90 {
                    let dx = DR_INTRA_DERIVATIVE[p_angle];
                    if dx == 0 {
                        continue;
                    }
                    for &up in ups {
                        cmp2(
                            &mut rng,
                            w,
                            h,
                            "z1_tbl",
                            |d, s| p::ipred_z1(d, s, a, w, h, dx, up),
                            |d, s| match (w, up) {
                                (4, false) => z1_tbl::<u8, 4, 0>(d, s, a, h, dx),
                                (4, true) => z1_tbl::<u8, 4, 1>(d, s, a, h, dx),
                                (_, false) => z1_tbl::<u8, 8, 0>(d, s, a, h, dx),
                                (_, true) => z1_tbl::<u8, 8, 1>(d, s, a, h, dx),
                            },
                        );
                    }
                }
                for p_angle in 91..180 {
                    let dx = DR_INTRA_DERIVATIVE[180 - p_angle];
                    let dy = DR_INTRA_DERIVATIVE[p_angle - 90];
                    if dx == 0 || dy == 0 {
                        continue;
                    }
                    for (ua, ul) in [(false, false), (true, false), (false, true), (true, true)] {
                        if ul && h > 8 {
                            continue;
                        }
                        let (ab, lb) = (&c.above[..], &c.left[..]);
                        let f0 = |d: &mut [u8], s| p::ipred_z2(d, s, ab, lb, w, h, dx, dy, ua, ul);
                        cmp2(&mut rng, w, h, "z2_tbl", f0, |d, s| match (w, ua, ul) {
                            (4, false, false) => z2_tbl::<u8, 4, 0, 0>(d, s, ab, lb, h, dx, dy),
                            (4, true, false) => z2_tbl::<u8, 4, 1, 0>(d, s, ab, lb, h, dx, dy),
                            (4, false, true) => z2_tbl::<u8, 4, 0, 1>(d, s, ab, lb, h, dx, dy),
                            (4, true, true) => z2_tbl::<u8, 4, 1, 1>(d, s, ab, lb, h, dx, dy),
                            (_, false, false) => z2_tbl::<u8, 8, 0, 0>(d, s, ab, lb, h, dx, dy),
                            (_, true, false) => z2_tbl::<u8, 8, 1, 0>(d, s, ab, lb, h, dx, dy),
                            (_, false, true) => z2_tbl::<u8, 8, 0, 1>(d, s, ab, lb, h, dx, dy),
                            (_, true, true) => z2_tbl::<u8, 8, 1, 1>(d, s, ab, lb, h, dx, dy),
                        });
                    }
                }
                for p_angle in 181..270 {
                    let dy = DR_INTRA_DERIVATIVE[270 - p_angle];
                    if dy == 0 {
                        continue;
                    }
                    for &up in ups {
                        let u = up as usize;
                        if ((w as i32 * dy) >> (6 - u)) + ((h as i32 - 1) << u) + 1 >= 32 {
                            continue;
                        }
                        cmp2(
                            &mut rng,
                            w,
                            h,
                            "z3_tbl",
                            |d, s| p::ipred_z3(d, s, l, w, h, dy, up),
                            |d, s| match (w, up) {
                                (4, false) => z3_tbl::<u8, 4, 0>(d, s, l, h, dy),
                                (4, true) => z3_tbl::<u8, 4, 1>(d, s, l, h, dy),
                                (_, false) => z3_tbl::<u8, 8, 0>(d, s, l, h, dy),
                                (_, true) => z3_tbl::<u8, 8, 1>(d, s, l, h, dy),
                            },
                        );
                    }
                }
            }
        }
    }

    /// The register z2 kernels of [`z2_small16_reg`] ([`z2_tbl16_reg`] is
    /// only dispatched where `swizzle_dyn` is fast, [`z2_reg_x86`]
    /// elsewhere), called directly so that both are tested on every host,
    /// against the portable `predict_intra`.
    #[test]
    fn z2_reg_kernels_match_portable() {
        fn run<const BD: u32>(seed: u64, trials: usize) {
            let mut rng = Rng(seed);
            let mut e = IntraEdges::<u16>::default();
            let mut tested = 0;
            for (w, h) in [(4usize, 4usize), (4, 8), (8, 4), (8, 8)] {
                for mode in V_PRED..=D67_PRED {
                    for angle_delta in -3..=3 {
                        let p_angle = MODE_TO_ANGLE[mode as usize] + angle_delta * ANGLE_STEP;
                        if p_angle <= 90 || p_angle >= 180 {
                            continue;
                        }
                        for _ in 0..trials {
                            let c = Case::<u16>::new(&mut rng, BD);
                            e.top_left = c.tl;
                            e.above.copy_from_slice(c.a());
                            e.left.copy_from_slice(c.l());
                            let flags = rng.below(16);
                            let x = 4 * rng.below(64) as i32;
                            let y = 4 * rng.below(64) as i32;
                            let p = IntraPredParams {
                                mode,
                                angle_delta,
                                filter_intra_mode: None,
                                have_left: flags & 1 != 0,
                                have_above: flags & 2 != 0,
                                enable_intra_edge_filter: flags & 4 != 0,
                                smooth_filter_type: flags & 8 != 0,
                                x,
                                y,
                                max_x: x + rng.below(2 * w as u32 + 2) as i32,
                                max_y: y + rng.below(2 * h as u32 + 2) as i32,
                                w,
                                h,
                                bd: BD,
                            };
                            // Same selection and parameters as
                            // predict_directional.
                            let n = w + h;
                            let filter = p.enable_intra_edge_filter;
                            let ft = p.smooth_filter_type;
                            let da = (p_angle - 90) as usize;
                            let dl = (180 - p_angle) as usize;
                            let dx = DR_DERIV128[dl];
                            let dy = DR_DERIV128[da];
                            let need_a = dx <= (w as i32) * 64;
                            let need_l = (h as i32) * dx > 64;
                            let can_up = filter && n <= if ft { 8 } else { 16 };
                            if can_up && ((need_a && da < 40) || (need_l && dl < 40)) {
                                continue;
                            }
                            let lut = &STRENGTH_LUT[ft as usize][(n - 1) >> 3 & 1];
                            let max_w = (w as i32).min(p.max_x - p.x + 1);
                            let max_h = (h as i32).min(p.max_y - p.y + 1);
                            let s_a = if filter && p.have_above { lut[da] as u32 } else { 0 };
                            let s_l = if filter && p.have_left { lut[dl] as u32 } else { 0 };
                            let packed = z2_small_pack(dx, dy, max_w + 1, max_h + 1, s_a, s_l);
                            let stride = w + rng.below(9) as usize;
                            let mut d0 = vec![0x5au16; stride * (h - 1) + w];
                            let mut d1 = d0.clone();
                            let mut d2 = d0.clone();
                            crate::decoder::av1::dsp::ipred::predict_intra(&mut d0, stride, &e, &p);
                            match (w, h) {
                                (4, 4) => {
                                    z2_tbl16_reg::<u16, 4, 4>(&mut d1, stride, &e, packed);
                                    z2_reg_x86::<u16, 4, 4>(&mut d2, stride, &e, packed);
                                }
                                (4, _) => {
                                    z2_tbl16_reg::<u16, 4, 8>(&mut d1, stride, &e, packed);
                                    z2_reg_x86::<u16, 4, 8>(&mut d2, stride, &e, packed);
                                }
                                (_, 4) => {
                                    z2_tbl16_reg::<u16, 8, 4>(&mut d1, stride, &e, packed);
                                    z2_reg_x86::<u16, 8, 4>(&mut d2, stride, &e, packed);
                                }
                                _ => {
                                    z2_tbl16_reg::<u16, 8, 8>(&mut d1, stride, &e, packed);
                                    z2_reg_x86::<u16, 8, 8>(&mut d2, stride, &e, packed);
                                }
                            }
                            assert!(d0 == d1, "z2_tbl16_reg mismatch {p:?}");
                            assert!(d0 == d2, "z2_reg_x86 mismatch {p:?}");
                            tested += 1;
                        }
                    }
                }
            }
            assert!(tested > 0);
        }
        run::<10>(0x2e2_1610_5eed_0002, 40);
        run::<8>(0x2e2_1610_5eed_0003, 10);
    }

    /// The 10-bit table lookup z2 kernel (only dispatched where
    /// `swizzle_dyn` is fast) and the 4x4 tile kernels, called directly so
    /// that both are tested on every host.
    #[test]
    fn z2_small16_kernels_match_scalar() {
        use crate::decoder::av1::dsp::ipred as p;
        let mut rng = Rng(0x2e2_1610_5eed_0001);
        for bd in [8u32, 10] {
            for (w, h) in sizes() {
                if w > 8 || h > 16 {
                    continue;
                }
                for _ in 0..6 {
                    let c = Case::<u16>::new(&mut rng, bd);
                    let (ab, lb) = (&c.above, &c.left);
                    for p_angle in 91..180 {
                        let dx = DR_INTRA_DERIVATIVE[180 - p_angle];
                        let dy = DR_INTRA_DERIVATIVE[p_angle - 90];
                        if dx == 0 || dy == 0 {
                            continue;
                        }
                        for (ua, ul) in [(false, false), (true, false), (false, true), (true, true)]
                        {
                            if ul && h > 8 {
                                continue;
                            }
                            let f0 = |d: &mut [u16], s| {
                                p::ipred_z2(d, s, &ab[..], &lb[..], w, h, dx, dy, ua, ul)
                            };
                            macro_rules! tbl {
                                ($w:literal, $h:literal, $ua:literal, $ul:literal) => {
                                    |d: &mut [u16], s| {
                                        z2_tbl16::<u16, $w, $h, $ua, $ul, EDGE_BUF_LEN>(
                                            d, s, ab, lb, dx, dy,
                                        )
                                    }
                                };
                            }
                            macro_rules! by_ups {
                                ($w:literal, $h:literal) => {
                                    match (ua, ul) {
                                        (false, false) => cmp2(
                                            &mut rng,
                                            w,
                                            h,
                                            "z2_tbl16",
                                            f0,
                                            tbl!($w, $h, false, false),
                                        ),
                                        (true, false) => cmp2(
                                            &mut rng,
                                            w,
                                            h,
                                            "z2_tbl16",
                                            f0,
                                            tbl!($w, $h, true, false),
                                        ),
                                        (false, true) => cmp2(
                                            &mut rng,
                                            w,
                                            h,
                                            "z2_tbl16",
                                            f0,
                                            tbl!($w, $h, false, true),
                                        ),
                                        (true, true) => cmp2(
                                            &mut rng,
                                            w,
                                            h,
                                            "z2_tbl16",
                                            f0,
                                            tbl!($w, $h, true, true),
                                        ),
                                    }
                                };
                            }
                            match (w, h) {
                                (4, 4) => by_ups!(4, 4),
                                (4, 8) => by_ups!(4, 8),
                                (8, 4) => by_ups!(8, 4),
                                (8, 8) => by_ups!(8, 8),
                                (4, _) if !ul => {
                                    if ua {
                                        cmp2(
                                            &mut rng,
                                            w,
                                            h,
                                            "z2_tbl16",
                                            f0,
                                            tbl!(4, 16, true, false),
                                        )
                                    } else {
                                        cmp2(
                                            &mut rng,
                                            w,
                                            h,
                                            "z2_tbl16",
                                            f0,
                                            tbl!(4, 16, false, false),
                                        )
                                    }
                                }
                                (_, _) if !ul => {
                                    if ua {
                                        cmp2(
                                            &mut rng,
                                            w,
                                            h,
                                            "z2_tbl16",
                                            f0,
                                            tbl!(8, 16, true, false),
                                        )
                                    } else {
                                        cmp2(
                                            &mut rng,
                                            w,
                                            h,
                                            "z2_tbl16",
                                            f0,
                                            tbl!(8, 16, false, false),
                                        )
                                    }
                                }
                                _ => {}
                            }
                            let f1 = |d: &mut [u16], s| {
                                z2_small16::<u16, EDGE_BUF_LEN>(d, s, ab, lb, w, h, dx, dy, ua, ul)
                            };
                            cmp2(&mut rng, w, h, "z2_small16", f0, f1);
                            let f2 = |d: &mut [u16], s| match (ua, ul) {
                                (false, false) => z2_tiles::<u16, false, false, EDGE_BUF_LEN>(
                                    d, s, ab, lb, w, h, dx, dy,
                                ),
                                (true, false) => z2_tiles::<u16, true, false, EDGE_BUF_LEN>(
                                    d, s, ab, lb, w, h, dx, dy,
                                ),
                                (false, true) => z2_tiles::<u16, false, true, EDGE_BUF_LEN>(
                                    d, s, ab, lb, w, h, dx, dy,
                                ),
                                (true, true) => z2_tiles::<u16, true, true, EDGE_BUF_LEN>(
                                    d, s, ab, lb, w, h, dx, dy,
                                ),
                            };
                            cmp2(&mut rng, w, h, "z2_tiles", f0, f2);
                        }
                    }
                }
            }
        }
    }

    /// `cfl_ac` + `cfl_pred` against the portable kernels for every
    /// size, subsampling and valid width / height.
    fn run_cfl<T: Pixel, const BD: u32>(seed: u64, trials: usize) {
        let mut rng = Rng(seed);
        let max = (1i32 << BD) - 1;
        const LS: usize = 72;
        let mut luma = vec![T::default(); LS * LS];
        for (w, h) in sizes() {
            if w > 32 || h > 32 {
                continue;
            }
            for (sx, sy) in [(0usize, 0usize), (1, 0), (1, 1), (0, 1)] {
                for _ in 0..trials {
                    gen(&mut rng, BD, &mut luma);
                    let wv = if rng.below(2) == 0 { w } else { 1 + rng.below(w as u32) as usize };
                    let hv = if rng.below(2) == 0 { h } else { 1 + rng.below(h as u32) as usize };
                    // Only the rows / columns the kernel may read.
                    let rows = ((hv - 1) << sy) + sy + 1;
                    let len = (rows - 1) * LS + ((wv - 1) << sx) + sx + 1;
                    let src = &luma[..len];
                    let mut ac0 = [0x7777i16; MAX_CFL_AC];
                    let mut ac1 = ac0;
                    crate::decoder::av1::dsp::ipred::cfl_ac(
                        &mut ac0, src, LS, w, h, wv, hv, sx, sy,
                    );
                    cfl_ac::<T, BD>(&mut ac1, src, LS, w, h, wv, hv, sx, sy);
                    assert!(
                        ac0 == ac1,
                        "cfl_ac mismatch {w}x{h} ss {sx},{sy} wv {wv} hv {hv}"
                    );
                    let alpha = rng.below(33) as i32 - 16;
                    let dc = T::from_i32(
                        [0, max, rng.below(max as u32 + 1) as i32][rng.below(3) as usize],
                    );
                    let stride = w + rng.below(9) as usize;
                    let mut d0 = vec![dc; stride * (h - 1) + w];
                    for (k, d) in d0.iter_mut().enumerate() {
                        if k % stride >= w {
                            *d = T::from_i32(0x5a);
                        }
                    }
                    let mut d1 = d0.clone();
                    crate::decoder::av1::dsp::ipred::cfl_pred(
                        &mut d0, stride, &ac0, w, h, alpha, BD,
                    );
                    cfl_pred::<T, BD>(&mut d1, stride, &ac1, w, h, alpha, BD);
                    assert!(d0 == d1, "cfl_pred mismatch {w}x{h} alpha {alpha}");
                }
            }
        }
    }

    #[test]
    fn cfl_matches_scalar() {
        run_cfl::<u8, 8>(0xcf1_0001, 300);
        run_cfl::<u16, 8>(0xcf1_0002, 50);
        run_cfl::<u16, 10>(0xcf1_0003, 300);
        run_cfl::<u16, 12>(0xcf1_0004, 300);
    }

    /// Times every kernel (scalar vs SIMD) over all block sizes:
    /// `cargo test --release bench_ipred -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_ipred() {
        bench::<u8>(8);
        bench::<u16>(10);
    }

    fn bench<T: Pixel>(bd: u32) {
        use std::hint::black_box;
        let mut rng = Rng(42);
        let c = Case::<T>::new(&mut rng, bd);
        let mut dst = vec![T::default(); 64 * 64];
        // IPRED_SIZES=WxH,WxH,... replaces the block sizes of the "all" set.
        let all: Vec<(usize, usize)> = match std::env::var("IPRED_SIZES") {
            Ok(s) => s
                .split(',')
                .map(|wh| {
                    let (w, h) = wh.split_once('x').unwrap();
                    (w.parse().unwrap(), h.parse().unwrap())
                })
                .collect(),
            Err(_) => sizes(),
        };
        let small: Vec<_> = all.iter().copied().filter(|&(w, h)| w * h <= 64).collect();
        // IPRED_BENCH=<prefix> restricts the run to some kernels.
        let filter = std::env::var("IPRED_BENCH").unwrap_or_default();
        let mut time = |name: &str, f: &dyn Fn(&IpredKernels<T>, &mut [T], usize, usize)| {
            if !name.starts_with(filter.as_str()) {
                return;
            }
            let mut t = [[f64::MAX; 2]; 2];
            for (si, (sz, reps)) in [(&all, 5000), (&small, 100000)].into_iter().enumerate() {
                for _ in 0..3 {
                    for (ti, k) in [&IpredKernels::SCALAR, get_table::<T>(bd)]
                        .iter()
                        .enumerate()
                    {
                        let start = std::time::Instant::now();
                        for _ in 0..reps {
                            for &(w, h) in sz {
                                f(black_box(k), black_box(&mut dst), w, h);
                            }
                        }
                        t[si][ti] = t[si][ti].min(start.elapsed().as_secs_f64());
                    }
                }
            }
            println!(
                "{bd:>2}-bit {name:<9} all: scalar {:6.2} ms simd {:6.2} ms x{:.2} | <=64px: scalar {:6.2} ms simd {:6.2} ms x{:.2}",
                t[0][0] * 1e3,
                t[0][1] * 1e3,
                t[0][0] / t[0][1],
                t[1][0] * 1e3,
                t[1][1] * 1e3,
                t[1][0] / t[1][1]
            );
        };
        time("dc", &|k, d, w, h| (k.dc)(d, 64, c.a(), c.l(), w, h));
        time("dc_top", &|k, d, w, h| (k.dc_top)(d, 64, c.a(), w, h));
        time("dc_left", &|k, d, w, h| (k.dc_left)(d, 64, c.l(), w, h));
        time("dc_128", &|k, d, w, h| (k.dc_128)(d, 64, w, h, bd));
        time("v", &|k, d, w, h| (k.v)(d, 64, c.a(), w, h));
        time("h", &|k, d, w, h| (k.h)(d, 64, c.l(), w, h));
        time("paeth", &|k, d, w, h| {
            (k.paeth)(d, 64, c.tl, c.a(), c.l(), w, h)
        });
        time("smooth", &|k, d, w, h| {
            (k.smooth)(d, 64, c.a(), c.l(), w, h)
        });
        time("smooth_v", &|k, d, w, h| {
            (k.smooth_v)(d, 64, c.a(), c.l(), w, h)
        });
        time("smooth_h", &|k, d, w, h| {
            (k.smooth_h)(d, 64, c.a(), c.l(), w, h)
        });
        time("filter", &|k, d, w, h| {
            (k.filter)(d, 64, c.tl, c.a(), c.l(), w.min(32), h.min(32), 3, bd)
        });
        // A steep (dx = 90) and a shallow (dx = 3) angle.
        time("z1 45-9", &|k, d, w, h| {
            (k.z1)(d, 64, &c.above[EDGE_PAD..], w, h, 90, false)
        });
        time("z1 87", &|k, d, w, h| {
            (k.z1)(d, 64, &c.above[EDGE_PAD..], w, h, 3, false)
        });
        time("z1 up", &|k, d, w, h| {
            (k.z1)(d, 64, &c.above[EDGE_PAD..], w.min(8), h.min(8), 27, true)
        });
        time("z2 135", &|k, d, w, h| {
            (k.z2)(d, 64, &c.above, &c.left, w, h, 64, 64, false, false)
        });
        time("z2 113", &|k, d, w, h| {
            (k.z2)(d, 64, &c.above, &c.left, w, h, 27, 151, false, false)
        });
        time("z2 157", &|k, d, w, h| {
            (k.z2)(d, 64, &c.above, &c.left, w, h, 151, 27, false, false)
        });
        time("z2 up", &|k, d, w, h| {
            (k.z2)(
                d,
                64,
                &c.above,
                &c.left,
                w.min(8),
                h.min(8),
                80,
                51,
                true,
                true,
            )
        });
        time("z3 203", &|k, d, w, h| {
            (k.z3)(d, 64, &c.left[EDGE_PAD..], w, h, 40, false)
        });
        time("z3 183", &|k, d, w, h| {
            (k.z3)(d, 64, &c.left[EDGE_PAD..], w, h, 3, false)
        });
        time("z3 up", &|k, d, w, h| {
            (k.z3)(d, 64, &c.left[EDGE_PAD..], w.min(8), h.min(8), 27, true)
        });
        // CFL (4:2:0) and the whole predict_intra (directional with edge
        // filter), not part of the kernel table.
        let mut luma = vec![T::default(); 128 * 128];
        gen(&mut rng, bd, &mut luma);
        let mut ac = [0i16; MAX_CFL_AC];
        let mut e = IntraEdges::<T>::default();
        e.above.copy_from_slice(c.a());
        e.left.copy_from_slice(c.l());
        let mut time2 = |name: &str,
                         f: &mut dyn FnMut(
            bool,
            &mut [T],
            &mut [i16; MAX_CFL_AC],
            usize,
            usize,
        )| {
            if !name.starts_with(filter.as_str()) {
                return;
            }
            let mut t = [[f64::MAX; 2]; 2];
            for (si, (sz, reps)) in [(&all, 5000), (&small, 100000)].into_iter().enumerate() {
                for _ in 0..3 {
                    for simd in [false, true] {
                        let start = std::time::Instant::now();
                        for _ in 0..reps {
                            for &(w, h) in sz {
                                f(black_box(simd), black_box(&mut dst), &mut ac, w, h);
                            }
                        }
                        t[si][simd as usize] =
                            t[si][simd as usize].min(start.elapsed().as_secs_f64());
                    }
                }
            }
            println!(
                "{bd:>2}-bit {name:<9} all: scalar {:6.2} ms simd {:6.2} ms x{:.2} | <=64px: scalar {:6.2} ms simd {:6.2} ms x{:.2}",
                t[0][0] * 1e3,
                t[0][1] * 1e3,
                t[0][0] / t[0][1],
                t[1][0] * 1e3,
                t[1][1] * 1e3,
                t[1][0] / t[1][1]
            );
        };
        time2("cfl_ac", &mut |simd, _d, ac, w, h| {
            let (w, h) = (w.min(32), h.min(32));
            if simd {
                match bd {
                    8 => cfl_ac::<T, 8>(ac, &luma, 128, w, h, w, h, 1, 1),
                    _ => cfl_ac::<T, 10>(ac, &luma, 128, w, h, w, h, 1, 1),
                }
            } else {
                crate::decoder::av1::dsp::ipred::cfl_ac(ac, &luma, 128, w, h, w, h, 1, 1)
            }
        });
        time2("cfl_pred", &mut |simd, d, ac, w, h| {
            let (w, h) = (w.min(32), h.min(32));
            if simd {
                match bd {
                    8 => cfl_pred::<T, 8>(d, 64, ac, w, h, -7, bd),
                    _ => cfl_pred::<T, 10>(d, 64, ac, w, h, -7, bd),
                }
            } else {
                crate::decoder::av1::dsp::ipred::cfl_pred(d, 64, ac, w, h, -7, bd)
            }
        });
        for (name, mode, delta) in [
            ("pi z1", 3u8, 1),
            ("pi z2", 4, 0),
            ("pi z3", 7, -2),
            ("pi dc", 0, 0),
        ] {
            time2(name, &mut |simd, d, _ac, w, h| {
                let p = IntraPredParams {
                    mode,
                    angle_delta: delta,
                    have_left: true,
                    have_above: true,
                    enable_intra_edge_filter: true,
                    x: 64,
                    y: 64,
                    max_x: 1000,
                    max_y: 1000,
                    w,
                    h,
                    bd,
                    ..Default::default()
                };
                if simd {
                    predict_intra(d, 64, &e, &p)
                } else {
                    crate::decoder::av1::dsp::ipred::predict_intra(d, 64, &e, &p)
                }
            });
        }
    }
}
