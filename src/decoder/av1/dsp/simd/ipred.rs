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
    SMOOTH_V_PRED, V_PRED,
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
    let stride = stride & 0xffff_ffff;
    let h4 = h >> 2;
    assert!(stride >= W && (1..=16).contains(&h4));
    (&mut dst[..(h4 * 4 - 1) * stride + W], stride, h4)
}

/// Returns the `W`-sample row starting at `off` in `dst`.
#[inline(always)]
fn row_mut<T, const W: usize>(dst: &mut [T], off: usize) -> &mut [T; W] {
    (&mut dst[off..off + W]).try_into().unwrap()
}

/// Fills a `W x h` block with `v` (`h` a multiple of 4 in `4..=64`).
#[inline(always)]
fn fill<T: Pixel, const W: usize>(dst: &mut [T], stride: usize, h: usize, v: T) {
    let (dst, stride, h4) = check_block4::<T, W>(dst, stride, h);
    let src = [v; W];
    let s2 = stride * 2;
    let s3 = stride * 3;
    let s4 = stride * 4;
    for i in 0..h4 {
        let base = i * s4;
        *row_mut::<T, W>(dst, base) = src;
        *row_mut::<T, W>(dst, base + stride) = src;
        *row_mut::<T, W>(dst, base + s2) = src;
        *row_mut::<T, W>(dst, base + s3) = src;
    }
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

/// Sum of `e[..n]` (`n` a multiple of `N`).
#[inline(always)]
fn edge_sum<T: Pixel, const BD: u32, const N: usize>(e: &[T], n: usize) -> u32 {
    if BD <= 10 {
        let mut acc = Simd::<u16, N>::splat(0);
        for c in e[..n].as_chunks::<N>().0 {
            acc += load_u16::<T, N>(c);
        }
        acc.cast::<u32>().reduce_sum()
    } else {
        let mut acc = Simd::<u32, N>::splat(0);
        for c in e[..n].as_chunks::<N>().0 {
            acc += load_u16::<T, N>(c).cast::<u32>();
        }
        acc.reduce_sum()
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
    let v = dc_both_val::<T, BD>(above, left, w, h);
    by_width!(w, dc_fill::<T>(dst, stride, h, v));
}

#[inline(always)]
fn dc_both_val<T: Pixel, const BD: u32>(above: &[T], left: &[T], w: usize, h: usize) -> T {
    let sum = edge_sum_n::<T, BD>(above, w) + edge_sum_n::<T, BD>(left, h);
    // w + h is 2^k, 3 * 2^k or 5 * 2^k: shift, then divide by 3 or 5 with
    // an exact multiply (valid for any u32 numerator).
    let n = (w + h) as u32;
    let k = n.trailing_zeros();
    let x = (sum + (n >> 1)) >> k;
    let avg = match n >> k {
        1 => x,
        3 => ((x as u64 * 0xAAAA_AAAB) >> 33) as u32,
        _ => ((x as u64 * 0xCCCC_CCCD) >> 34) as u32,
    };
    T::from_i32(avg as i32)
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

/// DC prediction of a `W x H` block (all variants), with the edge sums,
/// the division and the fill specialized for the block size. 8-bit
/// blocks up to 16x16 add both edges in one vector (lanes past `W` /
/// `H` zeroed) before a single horizontal reduction.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn dc_block<T: Pixel, const BD: u32, const W: usize, const H: usize>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    have_left: bool,
    have_above: bool,
) {
    let v = match (have_left, have_above) {
        (true, true) => {
            if BD == 8 && W <= 16 && H <= 16 {
                // Each lane holds at most 2 * 255, so the 16-lane
                // reduction (at most 8160) fits u16.
                const L: usize = 16;
                let lane = Simd::<u16, L>::from_array(std::array::from_fn(|k| k as u16));
                let a = load_u16::<T, L>(above);
                let l = load_u16::<T, L>(left);
                let a = lane
                    .simd_lt(Simd::splat(W as u16))
                    .select(a, Simd::splat(0));
                let l = lane
                    .simd_lt(Simd::splat(H as u16))
                    .select(l, Simd::splat(0));
                let sum = (a + l).reduce_sum() as u32;
                T::from_i32(dc_div::<W, H>(sum) as i32)
            } else {
                dc_both_val::<T, BD>(above, left, W, H)
            }
        }
        (true, false) => dc_edge_val::<T, BD>(left, H),
        (false, true) => dc_edge_val::<T, BD>(above, W),
        (false, false) => T::from_i32(1 << (BD - 1)),
    };
    fill::<T, W>(dst, stride, H, v);
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
    let (dst, stride, h4) = check_block4::<T, W>(dst, stride, h);
    let src: [T; W] = above[..W].try_into().unwrap();
    let s2 = stride * 2;
    let s3 = stride * 3;
    let s4 = stride * 4;
    for i in 0..h4 {
        let base = i * s4;
        *row_mut::<T, W>(dst, base) = src;
        *row_mut::<T, W>(dst, base + stride) = src;
        *row_mut::<T, W>(dst, base + s2) = src;
        *row_mut::<T, W>(dst, base + s3) = src;
    }
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
    let (dst, stride, h4) = check_block4::<T, W>(dst, stride, h);
    let left4 = left[..h4 * 4].as_chunks::<4>().0;
    let s2 = stride * 2;
    let s3 = stride * 3;
    let s4 = stride * 4;
    for (i, &[l0, l1, l2, l3]) in left4.iter().enumerate() {
        let base = i * s4;
        *row_mut::<T, W>(dst, base) = [l0; W];
        *row_mut::<T, W>(dst, base + stride) = [l1; W];
        *row_mut::<T, W>(dst, base + s2) = [l2; W];
        *row_mut::<T, W>(dst, base + s3) = [l3; W];
    }
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
    let (dst, stride) = check_block::<T, W>(dst, stride, h);
    let tl_s = top_left.to_i32() as i16;
    let tl = Simd::<i16, N>::splat(tl_s);
    let above = &above[..W];
    let left = &left[..h];
    // Column chunks outside, rows inside: `a`, `a - tl`, and `|a - tl|`
    // are row-invariant and loaded/computed once per column chunk.
    for x in (0..W).step_by(N) {
        let a = load_i16::<T, N>(&above[x..]);
        let a_tl = a - tl;
        let p_left = a_tl.abs();
        for (i, &l) in left.iter().enumerate() {
            let l_s = l.to_i32() as i16;
            let l_tl = l_s - tl_s;
            let lv = Simd::splat(l_s);
            let p_top = Simd::splat(l_tl.abs());
            let p_top_left = (a_tl + Simd::splat(l_tl)).abs();
            let use_left = p_left.simd_le(p_top) & p_left.simd_le(p_top_left);
            let use_above = p_top.simd_le(p_top_left);
            store_i16(
                row_mut::<T, N>(dst, i * stride + x),
                use_left.select(lv, use_above.select(a, tl)),
            );
        }
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
        match w {
            4 => return smooth_packed::<T, 4, 4>(dst, stride, above, left, h),
            8 => return smooth_packed::<T, 8, 2>(dst, stride, above, left, h),
            _ => {}
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
    let wy = &sm_weights(h)[..h];
    let left = &left[..h];
    let above = &above[..W];
    let bottom = left[h - 1].to_u16();
    let right = above[W - 1].to_u16();
    type V = Simd<u16, 16>;
    let one = V::splat(1);
    let a = V::from_array(std::array::from_fn(|j| above[j % W].to_u16()));
    let wxv = V::from_array(std::array::from_fn(|j| wx[j % W] as u16));
    let hbase = (V::splat(256) - wxv) * V::splat(right) + V::splat(128);
    let vbase = V::splat(bottom * 256 + 128);
    for (g, (l, wyg)) in left.chunks_exact(R).zip(wy.chunks_exact(R)).enumerate() {
        let wyv = V::from_array(std::array::from_fn(|j| wyg[j / W] as u16));
        let lv = V::from_array(std::array::from_fn(|j| l[j / W].to_u16()));
        // (256 - wy) * bottom + wy * a == 256 * bottom + wy * (a - bottom)
        // (mod 2^16, and the true value fits u16).
        let vert = wyv * (a - V::splat(bottom)) + vbase;
        let horz = wxv * lv + hbase;
        let p = ((vert >> one) + (horz >> one) + (vert & horz & one)) >> 8;
        for r in 0..R {
            let v = Simd::<u16, W>::from_slice(&p.as_array()[r * W..r * W + W]);
            store_u16(row_mut::<T, W>(dst, (g * R + r) * stride), v);
        }
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
    let (dst, stride) = check_block::<T, W>(dst, stride, h);
    let wx = &sm_weights(W)[..W];
    let wy = &sm_weights(h)[..h];
    let bottom = left[h - 1].to_i32();
    let right = above[W - 1].to_i32();
    let above = &above[..W];
    let left = &left[..h];
    // Column chunks outside, rows inside: the per column vectors are
    // loaded once per chunk.
    for x in (0..W).step_by(N) {
        if BD == 8 {
            // vert = wy * a + (256 - wy) * bottom + 128 and
            // horz = wx * l + (256 - wx) * right + 128 both fit in u16
            // (<= 256 * 255 + 128 = 65408), and
            // (vert + horz) >> 9 == ((vert >> 1) + (horz >> 1) + (vert & horz & 1)) >> 8.
            let one = Simd::<u16, N>::splat(1);
            let wxv = weights::<u16, N>(&wx[x..]);
            let a = load_u16::<T, N>(&above[x..]);
            let hbase = (Simd::splat(256) - wxv) * Simd::splat(right as u16) + Simd::splat(128);
            for (i, (&l, &wyi)) in left.iter().zip(wy).enumerate() {
                let vert = Simd::splat(wyi as u16) * a
                    + Simd::splat((256 - wyi as u16) * bottom as u16 + 128);
                let horz = wxv * Simd::splat(l.to_u16()) + hbase;
                let p = (vert >> one) + (horz >> one) + (vert & horz & one);
                store_u16(row_mut::<T, N>(dst, i * stride + x), p >> 8);
            }
        } else {
            // wy * a + (256 - wy) * bottom + wx * l + (256 - wx) * right
            // == wy * (a - bottom) + wx * (l - right) + 256 * (bottom + right)
            // where all factors fit in i16 (16 x 16 -> 32 bit products).
            let bias = Simd::<i32, N>::splat(256 * (bottom + right) + 256);
            let wxv = weights::<i16, N>(&wx[x..]).cast::<i32>();
            let a = (load_i16::<T, N>(&above[x..]) - Simd::splat(bottom as i16)).cast::<i32>();
            for (i, (&l, &wyi)) in left.iter().zip(wy).enumerate() {
                let wyv = Simd::<i16, N>::splat(wyi as i16).cast::<i32>();
                let l = Simd::<i16, N>::splat((l.to_i32() - right) as i16).cast::<i32>();
                store_i32(
                    row_mut::<T, N>(dst, i * stride + x),
                    (wyv * a + wxv * l + bias) >> 9,
                );
            }
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
        match w {
            4 => return smooth_hv_packed::<T, 4, 4, true>(dst, stride, above, left, h),
            8 => return smooth_hv_packed::<T, 8, 2, true>(dst, stride, above, left, h),
            _ => {}
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
    let wy = &sm_weights(h)[..h];
    let left = &left[..h];
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
    for (g, (l, wyg)) in left.chunks_exact(R).zip(wy.chunks_exact(R)).enumerate() {
        let p = if V {
            let wyv = U::from_array(std::array::from_fn(|j| wyg[j / W] as u16));
            (wyv * edge + base) >> 8
        } else {
            let lv = U::from_array(std::array::from_fn(|j| l[j / W].to_u16()));
            (edge * (lv - right) + base) >> 8
        };
        for r in 0..R {
            let v = Simd::<u16, W>::from_slice(&p.as_array()[r * W..r * W + W]);
            store_u16(row_mut::<T, W>(dst, (g * R + r) * stride), v);
        }
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
    let (dst, stride) = check_block::<T, W>(dst, stride, h);
    let wy = sm_weights(h);
    let bottom = left[h - 1].to_i32();
    let above = &above[..W];
    for x in (0..W).step_by(N) {
        if BD == 8 {
            // wy * a + (256 - wy) * bottom + 128 fits in u16.
            let a = load_u16::<T, N>(&above[x..]);
            for (i, &wyi) in wy.iter().enumerate() {
                let wyv = Simd::splat(wyi as u16);
                let base = Simd::splat((256 - wyi as u16) * bottom as u16 + 128);
                store_u16(row_mut::<T, N>(dst, i * stride + x), (wyv * a + base) >> 8);
            }
        } else {
            // wy * (a - bottom) + 256 * bottom with 16 x 16 bit products.
            let a = (load_i16::<T, N>(&above[x..]) - Simd::splat(bottom as i16)).cast::<i32>();
            let bias = Simd::splat(256 * bottom + 128);
            for (i, &wyi) in wy.iter().enumerate() {
                let wyv = Simd::<i16, N>::splat(wyi as i16).cast::<i32>();
                store_i32(row_mut::<T, N>(dst, i * stride + x), (wyv * a + bias) >> 8);
            }
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
        match w {
            4 => return smooth_hv_packed::<T, 4, 4, false>(dst, stride, above, left, h),
            8 => return smooth_hv_packed::<T, 8, 2, false>(dst, stride, above, left, h),
            _ => {}
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
    let (dst, stride) = check_block::<T, W>(dst, stride, h);
    let wx = sm_weights(W);
    let right = above[W - 1].to_i32();
    let left = &left[..h];
    for x in (0..W).step_by(N) {
        if BD == 8 {
            // wx * l + (256 - wx) * right + 128 fits in u16.
            let w = weights::<u16, N>(&wx[x..]);
            let b = (Simd::splat(256) - w) * Simd::splat(right as u16) + Simd::splat(128);
            for (i, &l) in left.iter().enumerate() {
                let l = Simd::splat(l.to_u16());
                store_u16(row_mut::<T, N>(dst, i * stride + x), (w * l + b) >> 8);
            }
        } else {
            // wx * (l - right) + 256 * right with 16 x 16 bit products.
            let w = weights::<i16, N>(&wx[x..]).cast::<i32>();
            let bias = Simd::splat(256 * right + 128);
            for (i, &l) in left.iter().enumerate() {
                let l = Simd::<i16, N>::splat((l.to_i32() - right) as i16).cast::<i32>();
                store_i32(row_mut::<T, N>(dst, i * stride + x), (w * l + bias) >> 8);
            }
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
        let (a, b) = E::load::<T, N>(src).deinterleave(E::load::<T, N>(&src[N..]));
        E::interp(a, b, s)
    } else {
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
    let (dst, stride) = check_block::<T, W>(dst, stride, h);
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
    for i in 0..h {
        let row = row_mut::<T, W>(dst, i * stride);
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
            for (x, d) in (0..W).step_by(N).zip(row.as_chunks_mut::<N>().0.iter_mut()) {
                let v = interp_run::<T, E, N>(&above[base0 + (x << u)..], upsample, shift);
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
    let (dst, stride) = check_block::<T, W>(dst, stride, h);
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
    for i0 in (0..h).step_by(L / W) {
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

/// Stores the 16 `u8` lanes of `v` (pixel values) to rows `i0..i0 + 16 / W`
/// of `dst` (`T` is `u8`).
#[inline(always)]
fn store_rows_u8<T: Pixel, const W: usize>(
    dst: &mut [T],
    stride: usize,
    i0: usize,
    v: Simd<i16, 16>,
) {
    let v = v.cast::<u8>().to_array();
    for k in 0..16 / W {
        let row: &mut dyn Any = row_mut::<T, W>(dst, (i0 + k) * stride);
        if let Some(row) = row.downcast_mut::<[u8; W]>() {
            row.copy_from_slice(&v[k * W..(k + 1) * W]);
        }
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
    let (dst, stride) = check_block::<T, W>(dst, stride, h);
    let t = [load_u8x16::<T>(above), load_u8x16::<T>(&above[16..])];
    let max_base_x = ((W + h - 1) << U) as i16;
    let edge = Simd::<i16, L>::splat(above[max_base_x as usize].to_i32() as i16);
    let lane = Simd::<i16, L>::from_array(std::array::from_fn(|k| k as i16));
    let j = (lane & Simd::splat(W as i16 - 1)) << U;
    let r = lane >> Simd::splat(W.trailing_zeros() as i16);
    // Positions (i + 1) * dx <= 16 * 1023 fit in i16.
    let mut idx = (r + Simd::splat(1)) * Simd::splat(dx as i16);
    let step = Simd::splat((L / W) as i16 * dx as i16);
    for i0 in (0..h).step_by(L / W) {
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
    let (dst, stride) = check_block::<T, W>(dst, stride, h);
    let t = [load_u8x16::<T>(left), load_u8x16::<T>(&left[16..])];
    let lane = Simd::<i16, L>::from_array(std::array::from_fn(|k| k as i16));
    let j = lane & Simd::splat(W as i16 - 1);
    let r = lane >> Simd::splat(W.trailing_zeros() as i16);
    let idx = (j + Simd::splat(1)) * Simd::splat(dy as i16);
    let mut base = (idx >> (6 - U)) + (r << U);
    let step = Simd::splat(((L / W) as i16) << U);
    let shift = if U == 1 { idx } else { idx >> 1 } & Simd::splat(31);
    for i0 in (0..h).step_by(L / W) {
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
    let (dst, stride) = check_block::<T, K>(dst, stride, h);
    let above: &[T; EDGE_BUF_LEN] = above[..EDGE_BUF_LEN].try_into().unwrap();
    let left: &[T; EDGE_BUF_LEN] = left[..EDGE_BUF_LEN].try_into().unwrap();
    let ua = upsample_above as i32;
    let ul = upsample_left as i32;
    let pad = EDGE_PAD as i32;
    for r in (0..h / K).map(|t| t * K) {
        let mut cols = [E::splat::<K>(0); K];
        for (k, col) in cols.iter_mut().enumerate() {
            let idx = -(k as i32 + 1) * dy;
            let b = ((idx >> (6 - ul)).max(-pad) + pad) as usize + (r << ul);
            let s = E::splat(((idx << ul) >> 1) & 0x1f);
            *col = interp_run::<T, E, K>(&left[b..], upsample_left, s);
        }
        for (k, lt) in transpose(cols).into_iter().enumerate() {
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
            E::store(row_mut::<T, K>(dst, i * stride), E::select_lt(m, lt, at));
        }
    }
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
    let (dst, stride) = check_block::<T, 8>(dst, stride, 4);
    let above: &[T; EDGE_BUF_LEN] = above[..EDGE_BUF_LEN].try_into().unwrap();
    let left: &[T; EDGE_BUF_LEN] = left[..EDGE_BUF_LEN].try_into().unwrap();
    let ua = upsample_above as i32;
    let ul = upsample_left as i32;
    let pad = EDGE_PAD as i32;
    let left_tile = |c: usize| {
        let mut cols = [E::splat::<4>(0); 4];
        for (k, col) in cols.iter_mut().enumerate() {
            let idx = -((c + k + 1) as i32) * dy;
            let b = ((idx >> (6 - ul)).max(-pad) + pad) as usize;
            let s = E::splat(((idx << ul) >> 1) & 0x1f);
            *col = interp_run::<T, E, 4>(&left[b..], upsample_left, s);
        }
        transpose(cols)
    };
    let lt0 = left_tile(0);
    let lt1 = if dx > 64 { left_tile(4) } else { [E::splat::<4>(0); 4] };
    for i in 0..4 {
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
        E::store(row_mut::<T, 8>(dst, i * stride), E::select_lt(m, lt, at));
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
            let s = ((idx0 << ua) >> 1) & 0x1f;
            let base = |x: usize| (((x as i32 * 64 + idx0) >> (6 - ua)) + pad) as usize;
            let mut x = x0;
            if K == 4 && N >= 8 && (x & 4) != 0 && x < W {
                let v = interp_run::<T, E, 4>(&above[base(x)..], upsample_above, E::splat(s));
                E::store(&mut row[x..x + 4], v);
                x += 4;
            }
            if N == 16 && (x & 8) != 0 && x < W {
                let v = interp_run::<T, E, 8>(&above[base(x)..], upsample_above, E::splat(s));
                E::store(&mut row[x..x + 8], v);
                x += 8;
            }
            while x < W {
                let v = interp_run::<T, E, N>(&above[base(x)..], upsample_above, E::splat(s));
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
    let (dst, stride) = check_block::<T, W>(dst, stride, h);
    let u = upsample as usize;
    for c in (0..W / K).map(|t| t * K) {
        let mut base0 = [0usize; K];
        let mut shift = [E::splat::<K>(0); K];
        for k in 0..K {
            let idx = (c + k + 1) as i32 * dy;
            base0[k] = (idx >> (6 - u)) as usize;
            shift[k] = E::splat(((idx << u) >> 1) & 0x1f);
        }
        for r in (0..h / K).map(|t| t * K) {
            // Rows r..r + K of column c + k: left[base0 + (i << u)] and
            // the next sample.
            let mut cols = [E::splat::<K>(0); K];
            for k in 0..K {
                cols[k] = interp_run::<T, E, K>(&left[base0[k] + (r << u)..], upsample, shift[k]);
            }
            for (i, v) in transpose(cols).into_iter().enumerate() {
                E::store(row_mut::<T, K>(dst, (r + i) * stride + c), v);
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
    let p: &[[T; 2]; K] = src[..2 * K].as_chunks::<2>().0.try_into().unwrap();
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
        let luma = &luma[..((hv << SY) - 1) * stride + (W << SX)];
        for i in 0..hv {
            let r0 = &luma[(i << SY) * stride..];
            let r1 = &luma[((i << SY) + SY) * stride..];
            let row: &mut [i16; W] = (&mut ac[i * W..(i + 1) * W]).try_into().unwrap();
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
        for i in 0..hv {
            let r0 = &luma[(i << SY) * stride..];
            let r1 = &luma[((i << SY) + SY) * stride..];
            let row: &mut [i16; W] = (&mut ac[i * W..(i + 1) * W]).try_into().unwrap();
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
#[inline(always)]
fn cfl_pred_w<T: Pixel, const BD: u32, const W: usize, const K: usize>(
    dst: &mut [T],
    stride: usize,
    ac: &[i16],
    h: usize,
    alpha: i32,
) {
    let (dst, stride) = check_block::<T, W>(dst, stride, h);
    let max = (1i32 << BD) - 1;
    for (i, acr) in ac[..W * h].as_chunks::<W>().0.iter().enumerate() {
        let row = row_mut::<T, W>(dst, i * stride);
        for (d, a) in row
            .as_chunks_mut::<K>()
            .0
            .iter_mut()
            .zip(acr.as_chunks::<K>().0)
        {
            let a = Simd::<i16, K>::from_array(*a);
            let dc = load_i16::<T, K>(d);
            let v = if BD == 8 {
                let p = a * Simd::splat(alpha as i16);
                let p = (p + (p >> 15) + Simd::splat(32)) >> 6;
                (dc + p).simd_clamp(Simd::splat(0), Simd::splat(max as i16))
            } else {
                let p = a.cast::<i32>() * Simd::splat(alpha);
                let p = (p + (p >> 31) + Simd::splat(32)) >> 6;
                (dc.cast::<i32>() + p)
                    .simd_clamp(Simd::splat(0), Simd::splat(max))
                    .cast::<i16>()
            };
            store_i16(d, v);
        }
    }
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
    debug_assert_eq!(p.bd, BD);
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
            by_size!(
                w,
                h,
                dc_block::<T, BD>(dst, stride, above, left, p.have_left, p.have_above)
            );
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

/// Prepares one directional edge buffer (`above` or `left`).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn prepare_edge<T: Pixel, const BD: u32>(
    buf: &mut [T; EDGE_BUF_LEN],
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
        // The edge filter strength is 0 (and n < 24, so no corner
        // filter) whenever upsampling is selected.
        debug_assert!(!have || edge_filter_strength(w, h, ft, delta) == 0);
        upsample_edge::<T, BD>(buf, corner, src, up_px);
    } else {
        let strength = if filter && have { edge_filter_strength(w, h, ft, delta) } else { 0 };
        fill_edge::<T>(buf, src, corner, up_px, len + 1, strength);
    }
    up
}

/// `dsp::ipred::predict_directional` with SIMD edge filtering. Only the
/// edge buffer(s) read by the selected kernel are prepared.
#[inline(never)]
fn predict_directional<T: Pixel, const BD: u32>(
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
    let max_w = (w as i32).min(p.max_x - p.x + 1);
    let max_h = (h as i32).min(p.max_y - p.y + 1);
    if p_angle < 90 {
        let dx = DR_INTRA_DERIVATIVE[p_angle as usize];
        let delta = p_angle - 90;
        let up = filter && use_upsample(w, h, ft, delta);
        let strength = if !up && filter && p.have_above {
            edge_filter_strength(w, h, ft, delta)
        } else {
            0
        };
        if !up && strength == 0 {
            ipred_z1::<T, BD>(dst, stride, &edges.above, w, h, dx, false);
            return;
        }
        let mut above = [T::default(); EDGE_BUF_LEN];
        if up {
            upsample_edge::<T, BD>(&mut above, edges.top_left, &edges.above, n);
        } else {
            fill_edge::<T>(
                &mut above,
                &edges.above,
                edges.top_left,
                n,
                max_w + h as i32 + 1,
                strength,
            );
        }
        ipred_z1::<T, BD>(dst, stride, &above[EDGE_PAD..], w, h, dx, up);
        return;
    }
    if p_angle > 180 {
        let dy = DR_INTRA_DERIVATIVE[(270 - p_angle) as usize];
        let delta = p_angle - 180;
        let up = filter && use_upsample(w, h, ft, delta);
        let strength = if !up && filter && p.have_left {
            edge_filter_strength(w, h, ft, delta)
        } else {
            0
        };
        if !up && strength == 0 {
            ipred_z3::<T, BD>(dst, stride, &edges.left, w, h, dy, false);
            return;
        }
        let mut left = [T::default(); EDGE_BUF_LEN];
        if up {
            upsample_edge::<T, BD>(&mut left, edges.top_left, &edges.left, n);
        } else {
            fill_edge::<T>(
                &mut left,
                &edges.left,
                edges.top_left,
                n,
                max_h + w as i32 + 1,
                strength,
            );
        }
        ipred_z3::<T, BD>(dst, stride, &left[EDGE_PAD..], w, h, dy, up);
        return;
    }
    let corner = if filter && n >= 24 {
        T::from_i32(filter_corner(edges.left[0], edges.top_left, edges.above[0]))
    } else {
        edges.top_left
    };
    let mut above = [T::default(); EDGE_BUF_LEN];
    let mut left = [T::default(); EDGE_BUF_LEN];
    let ua = prepare_edge::<T, BD>(
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
    );
    let ul = prepare_edge::<T, BD>(
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
    );
    let dx = DR_INTRA_DERIVATIVE[(180 - p_angle) as usize];
    let dy = DR_INTRA_DERIVATIVE[(p_angle - 90) as usize];
    ipred_z2::<T, BD>(dst, stride, &above, &left, w, h, dx, dy, ua, ul);
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
fn fill_edge<T: Pixel>(
    buf: &mut [T; EDGE_BUF_LEN],
    src: &[T; MAX_EDGE],
    corner: T,
    n: usize,
    sz: i32,
    strength: u32,
) {
    // 16 lanes: for u8 one 16 byte store per chunk, which the 16 byte
    // table loads of the small block kernels can forward from.
    const N: usize = 16;
    const { assert!(EDGE_PAD + MAX_EDGE <= EDGE_BUF_LEN && MAX_EDGE % N == 0) };
    debug_assert!(n <= MAX_EDGE && n % 4 == 0 && sz <= n as i32 + 1);
    buf[EDGE_PAD - 1] = corner;
    let mut k = 0;
    if strength != 0 && sz > 1 {
        let sz = sz as usize;
        // Spec edge[i] for i = 1..sz is filtered from edge[i - 2 ..= i + 2]
        // clamped to 0..sz. In src indices (src[j] = edge[j + 1]): out[j]
        // for j < sz - 1 from s[j - 2 ..= j + 2] with s[-2] = s[-1] =
        // corner and s[j] = src[sz - 2] for j > sz - 2.
        let lane = Simd::<u16, N>::from_array(std::array::from_fn(|l| l as u16));
        let last = Simd::<u16, N>::splat(src[sz - 2].to_u16());
        // (raw src[k..k + N], s[k..k + N]).
        let load = |k: usize| {
            if k + N > MAX_EDGE {
                // Only reached when every lane is past sz - 2.
                return (last, last);
            }
            let raw = load_u16::<T, N>(&src[k..]);
            (
                raw,
                (lane + Simd::splat(k as u16))
                    .simd_gt(Simd::splat((sz - 2) as u16))
                    .select(last, raw),
            )
        };
        let r = Simd::<u16, N>::splat(8);
        let mut prev = Simd::<u16, N>::splat(corner.to_u16());
        let (mut raw, mut cur) = load(0);
        while k < sz - 1 {
            let (next_raw, next) = load(k + N);
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
            let f = match strength {
                1 => ((t1 + t3) * Simd::splat(4) + cur * Simd::splat(8) + r) >> 4,
                2 => ((t1 + t3) * Simd::splat(5) + cur * Simd::splat(6) + r) >> 4,
                _ => {
                    let t0 = simd_swizzle!(
                        prev,
                        cur,
                        [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
                    );
                    let t4 = simd_swizzle!(
                        cur,
                        next,
                        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
                    );
                    ((t0 + t4) * Simd::splat(2) + (t1 + cur + t3) * Simd::splat(4) + r) >> 4
                }
            };
            let v = (lane + Simd::splat(k as u16))
                .simd_lt(Simd::splat((sz - 1) as u16))
                .select(f, raw);
            store_u16::<T, N>(&mut buf[EDGE_PAD + k..], v);
            (prev, cur, raw) = (cur, next, next_raw);
            k += N;
        }
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
fn upsample_edge<T: Pixel, const BD: u32>(
    buf: &mut [T; EDGE_BUF_LEN],
    corner: T,
    src: &[T; MAX_EDGE],
    num_px: usize,
) {
    const N: usize = MAX_UPSAMPLE_PX;
    const { assert!(EDGE_PAD + 2 * N <= EDGE_BUF_LEN && N <= MAX_EDGE) };
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
