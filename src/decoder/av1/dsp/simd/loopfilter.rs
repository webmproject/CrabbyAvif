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

//! SIMD deblocking filters (see `dsp::loopfilter` for the contracts).
//!
//! Eight lines across an edge are filtered at once, one line per lane.
//! All arithmetic is done in 16-bit lanes, which is exact for every bit
//! depth up to 12:
//!
//! * the filter mask and narrow filter values stay within `i16` (at most
//!   `3 * 4095 + 2047` in magnitude);
//! * the wide filter sums are positive and below `16 * 4095 + 8 < 65536`,
//!   so they are computed with wrapping `u16` arithmetic, where the
//!   running sums (`out[i + 1] = out[i] - a - b + c + d`) are exact.
//!
//! Every filter path is computed for all lanes and the results are
//! blended with the per lane masks, skipping paths that no lane takes.

use std::simd::prelude::*;
use std::simd::Swizzle;

use crate::decoder::av1::dsp::loopfilter::{self as portable, FilterLimits};
use crate::decoder::av1::pixel::Pixel;

/// Lines filtered together (one per lane).
const L: usize = 8;
type V = Simd<i16, L>;
type U = Simd<u16, L>;
type M = Mask<i16, L>;

#[inline(always)]
fn valid16<const LINES: usize>() -> M {
    if LINES == 8 {
        M::splat(true)
    } else {
        M::from_array([true, true, true, true, false, false, false, false])
    }
}

/// Bit depth scaled thresholds (see `dsp::loopfilter::Thresholds`),
/// splatted.
struct Th {
    limit: V,
    blimit: V,
    thresh: V,
}

impl Th {
    #[inline(always)]
    fn new<const BD: u32>(limits: FilterLimits) -> Self {
        let shift = BD - 8;
        let s = |v: i32| V::splat((v << shift) as i16);
        Th {
            limit: s(limits.limit),
            blimit: s(limits.blimit),
            thresh: s(limits.thresh),
        }
    }
}

/// Indices of NEON `trn1` / `trn2` (`HI`) on groups of `G` lanes.
const fn trn_index<const N: usize>(g: usize, hi: bool) -> [usize; N] {
    let mut r = [0; N];
    let mut i = 0;
    while i < N {
        let blk = i / g;
        let w = i % g;
        let h = hi as usize;
        r[i] = if blk % 2 == 0 { (blk + h) * g + w } else { N + (blk - 1 + h) * g + w };
        i += 1;
    }
    r
}

struct Trn16<const G: usize, const HI: bool>;
impl<const G: usize, const HI: bool> Swizzle<L> for Trn16<G, HI> {
    const INDEX: [usize; L] = trn_index::<L>(G, HI);
}

#[inline(always)]
fn trn16<const G: usize>(a: V, b: V) -> (V, V) {
    (
        Trn16::<G, false>::concat_swizzle(a, b),
        Trn16::<G, true>::concat_swizzle(a, b),
    )
}

/// Loads `src[..N]` (N <= L) into the first lanes, zeroing the others.
#[inline(always)]
fn load<T: Pixel, const N: usize>(src: &[T]) -> V {
    let src: &[T; N] = src[..N].try_into().unwrap();
    V::from_array(std::array::from_fn(|i| {
        if i < N {
            src[i].to_i32() as i16
        } else {
            0
        }
    }))
}

/// Stores the first `N` lanes of `v` to `dst[..N]`. `v` must hold valid
/// pixel values.
#[inline(always)]
fn store<T: Pixel, const N: usize>(dst: &mut [T], v: V) {
    let dst: &mut [T; N] = (&mut dst[..N]).try_into().unwrap();
    let v = v.to_array();
    for (d, &x) in dst.iter_mut().zip(&v) {
        *d = T::from_i32(x as i32);
    }
}

/// Transposes an 8x8 matrix held as 8 row vectors (three rounds of the
/// perfect shuffle: each one rotates the 6 index bits by one).
#[inline(always)]
fn transpose(mut v: [V; 8]) -> [V; 8] {
    for _ in 0..3 {
        let mut w = [V::splat(0); 8];
        for j in 0..4 {
            (w[2 * j], w[2 * j + 1]) = v[j].interleave(v[j + 4]);
        }
        v = w;
    }
    v
}

/// Transposes the 4x4 matrices held in each 4-lane half of 4 vectors:
/// row `k` (in lanes `0..4`) and row `k + 4` (in lanes `4..8`) of `v[k]`
/// become column `col` (rows `0..8` in lanes `0..8`) of the output `col`.
#[inline(always)]
fn transpose4(v: [V; 4]) -> [V; 4] {
    let (t0, t1) = trn16::<1>(v[0], v[1]);
    let (t2, t3) = trn16::<1>(v[2], v[3]);
    let (c0, c2) = trn16::<2>(t0, t2);
    let (c1, c3) = trn16::<2>(t1, t3);
    [c0, c1, c2, c3]
}

#[inline(always)]
fn load4_16<T: Pixel>(src: &[T]) -> u64 {
    let src: &[T; 4] = src[..4].try_into().unwrap();
    let v = Simd::<i16, 4>::from_array(std::array::from_fn(|i| src[i].to_i32() as i16));
    u64::from_ne_bytes(v.to_ne_bytes().to_array())
}

#[inline(always)]
fn store4_16<T: Pixel>(dst: &mut [T], v: u64) {
    let dst: &mut [T; 4] = (&mut dst[..4]).try_into().unwrap();
    let a = Simd::<i16, 4>::from_ne_bytes(Simd::from_array(v.to_ne_bytes())).to_array();
    for i in 0..4 {
        dst[i] = T::from_i32(a[i] as i32);
    }
}

#[inline(always)]
fn load_row4_16<T: Pixel, const LINES: usize>(win: &[T], pos: usize, stride: usize) -> V {
    let r0 = load4_16(&win[pos..]);
    let r1 = if LINES == 8 { load4_16(&win[pos + 4 * stride..]) } else { 0 };
    V::from_ne_bytes(Simd::<u64, 2>::from_array([r0, r1]).to_ne_bytes())
}

/// Loads 4 columns for `LINES` (4 or 8) rows starting at `pos` and
/// transposes them into 4 column vectors.
#[inline(always)]
fn load_cols4<T: Pixel, const LINES: usize>(buf: &[T], pos: usize, stride: usize) -> [V; 4] {
    let stride = stride & 0x0fff_ffff;
    let win = &buf[pos..][..(LINES - 1) * stride + 4];
    transpose4([
        load_row4_16::<T, LINES>(win, 0, stride),
        load_row4_16::<T, LINES>(win, stride, stride),
        load_row4_16::<T, LINES>(win, 2 * stride, stride),
        load_row4_16::<T, LINES>(win, 3 * stride, stride),
    ])
}

/// Transposes 4 column vectors and stores 4 columns for `LINES` (4 or 8)
/// rows starting at `pos`.
#[inline(always)]
fn store_cols4<T: Pixel, const LINES: usize>(
    buf: &mut [T],
    pos: usize,
    stride: usize,
    cols: [V; 4],
) {
    let stride = stride & 0x0fff_ffff;
    let win = &mut buf[pos..][..(LINES - 1) * stride + 4];
    let v = transpose4(cols);
    for (k, &vk) in v.iter().enumerate() {
        let [r0, r1] = Simd::<u64, 2>::from_ne_bytes(vk.to_ne_bytes()).to_array();
        store4_16(&mut win[k * stride..], r0);
        if LINES == 8 {
            store4_16(&mut win[(k + 4) * stride..], r1);
        }
    }
}

#[inline(always)]
fn load_cols8<T: Pixel, const LINES: usize, const K: usize>(
    buf: &[T],
    pos: usize,
    stride: usize,
) -> [V; 8] {
    let stride = stride & 0x0fff_ffff;
    let win = &buf[pos..][..(LINES - 1) * stride + K];
    let mut v = [V::splat(0); 8];
    for (k, vk) in v[..LINES].iter_mut().enumerate() {
        *vk = load::<T, K>(&win[k * stride..]);
    }
    transpose(v)
}

#[inline(always)]
fn store_cols8<T: Pixel, const LINES: usize, const K: usize>(
    buf: &mut [T],
    pos: usize,
    stride: usize,
    cols: [V; 8],
) {
    let stride = stride & 0x0fff_ffff;
    let win = &mut buf[pos..][..(LINES - 1) * stride + K];
    let v = transpose(cols);
    for (k, &vk) in v[..LINES].iter().enumerate() {
        store::<T, K>(&mut win[k * stride..], vk);
    }
}

#[inline(always)]
fn abd16(a: V, b: V) -> V {
    a.simd_max(b) - a.simd_min(b)
}

/// Filters the lines held in `s` (`s[7 + k]` holds the samples at signed
/// offset `k` from `q0`, see `dsp::loopfilter::filter_line`) for the
/// lanes in `valid`. Returns the modified radius `m` (`0`, `1`, `2`, `3`
/// or `6`): columns `7 - m..7 + m` of `s` were modified.
///
/// For `SIZE == 16`, `s[3..11]` must be populated on entry, and
/// `load_outer` is called to load `([p6, p5, p4], [q4, q5, q6])` only
/// when at least one lane is flat.
#[inline(always)]
fn filter<const BD: u32, const SIZE: usize>(
    s: &mut [V; 14],
    valid: M,
    th: &Th,
    load_outer: impl FnOnce() -> ([V; 3], [V; 3]),
) -> usize {
    let [_, _, _, p3, p2, p1, p0, q0, q1, q2, q3, _, _, _] = *s;
    let flat_th = V::splat(1 << (BD - 8));

    // 7.14.6.2 filter mask.
    let inner = abd16(p1, p0).simd_max(abd16(q1, q0));
    let mut m_lim = inner;
    if SIZE >= 6 {
        m_lim = m_lim.simd_max(abd16(p2, p1)).simd_max(abd16(q2, q1));
    }
    if SIZE >= 8 {
        m_lim = m_lim.simd_max(abd16(p3, p2)).simd_max(abd16(q3, q2));
    }
    let e = (abd16(p0, q0) << 1) + (abd16(p1, q1) >> 1);
    let filt = valid & m_lim.simd_le(th.limit) & e.simd_le(th.blimit);
    if !filt.any() {
        return 0;
    }

    let flat = if SIZE >= 6 {
        let mut f = inner.simd_max(abd16(p2, p0)).simd_max(abd16(q2, q0));
        if SIZE >= 8 {
            f = f.simd_max(abd16(p3, p0)).simd_max(abd16(q3, q0));
        }
        filt & f.simd_le(flat_th)
    } else {
        M::splat(false)
    };

    // 7.14.6.3 narrow filter, for the lanes that are filtered but not
    // flat.
    let narrow = filt & !flat;
    if narrow.any() {
        let lo = V::splat(-(0x80 << (BD - 8)));
        let hi = V::splat((0x80 << (BD - 8)) - 1);
        let maxv = V::splat((1 << BD) - 1);
        let c = |v: V| v.simd_max(lo).simd_min(hi);
        let p_clamp = |v: V| v.simd_max(V::splat(0)).simd_min(maxv);
        let hev = inner.simd_gt(th.thresh);
        let base = hev.select(c(p1 - q1), V::splat(0));
        let f = c(base + V::splat(3) * (q0 - p0));
        let f1 = (f + V::splat(4)).simd_min(hi) >> 3;
        let f2 = (f + V::splat(3)).simd_min(hi) >> 3;
        s[7] = narrow.select(p_clamp(q0 - f1), q0);
        s[6] = narrow.select(p_clamp(p0 + f2), p0);
        let outer = narrow & !hev;
        let f = (f1 + V::splat(1)) >> 1;
        s[8] = outer.select(p_clamp(q1 - f), q1);
        s[5] = outer.select(p_clamp(p1 + f), p1);
    }

    // 7.14.6.4 wide filters.
    if SIZE >= 6 && flat.any() {
        let k = |v: u16| U::splat(v);
        let blend =
            |orig: V, m: M, sum: U, shift: u16| m.select((sum >> k(shift)).cast::<i16>(), orig);
        if SIZE == 6 {
            let (p2, p1, p0, q0, q1, q2) = (
                p2.cast::<u16>(),
                p1.cast::<u16>(),
                p0.cast::<u16>(),
                q0.cast::<u16>(),
                q1.cast::<u16>(),
                q2.cast::<u16>(),
            );
            let p0q0 = p0 + q0 + k(2);
            s[5] = blend(s[5], flat, k(3) * p2 + k(2) * (p1 + p0) + q0 + k(4), 3);
            s[6] = blend(s[6], flat, p2 + k(2) * (p1 + p0q0) + q1, 3);
            s[7] = blend(s[7], flat, p1 + k(2) * (p0q0 + q1) + q2, 3);
            s[8] = blend(s[8], flat, p0 + k(2) * (q0 + q1) + k(3) * q2 + k(4), 3);
            return 2;
        }
        let flat2 = if SIZE == 16 {
            let ([p6, p5, p4], [q4, q5, q6]) = load_outer();
            s[1] = p5;
            s[2] = p4;
            s[11] = q4;
            s[12] = q5;
            let mut f = abd16(p4, p0).simd_max(abd16(q4, q0));
            f = f.simd_max(abd16(p5, p0)).simd_max(abd16(q5, q0));
            f = f.simd_max(abd16(p6, p0)).simd_max(abd16(q6, q0));
            let flat2 = flat & f.simd_le(flat_th);
            if flat2.any() {
                let (p6, p5, p4, p3, p2, p1, p0) = (
                    p6.cast::<u16>(),
                    p5.cast::<u16>(),
                    p4.cast::<u16>(),
                    p3.cast::<u16>(),
                    p2.cast::<u16>(),
                    p1.cast::<u16>(),
                    p0.cast::<u16>(),
                );
                let (q0, q1, q2, q3, q4, q5, q6) = (
                    q0.cast::<u16>(),
                    q1.cast::<u16>(),
                    q2.cast::<u16>(),
                    q3.cast::<u16>(),
                    q4.cast::<u16>(),
                    q5.cast::<u16>(),
                    q6.cast::<u16>(),
                );
                let mut t = k(7) * p6 + k(2) * (p5 + p4) + p3 + p2 + p1 + p0 + q0 + k(8);
                s[1] = blend(s[1], flat2, t, 4);
                t = t - p6 - p6 + p3 + q1;
                s[2] = blend(s[2], flat2, t, 4);
                t = t - p6 - p5 + p2 + q2;
                s[3] = blend(s[3], flat2, t, 4);
                t = t - p6 - p4 + p1 + q3;
                s[4] = blend(s[4], flat2, t, 4);
                t = t - p6 - p3 + p0 + q4;
                s[5] = blend(s[5], flat2, t, 4);
                t = t - p6 - p2 + q0 + q5;
                s[6] = blend(s[6], flat2, t, 4);
                t = t - p6 - p1 + q1 + q6;
                s[7] = blend(s[7], flat2, t, 4);
                t = t - p5 - p0 + q2 + q6;
                s[8] = blend(s[8], flat2, t, 4);
                t = t - p4 - q0 + q3 + q6;
                s[9] = blend(s[9], flat2, t, 4);
                t = t - p3 - q1 + q4 + q6;
                s[10] = blend(s[10], flat2, t, 4);
                t = t - p2 - q2 + q5 + q6;
                s[11] = blend(s[11], flat2, t, 4);
                t = t - p1 - q3 + q6 + q6;
                s[12] = blend(s[12], flat2, t, 4);
            }
            flat2
        } else {
            M::splat(false)
        };
        let flat8 = flat & !flat2;
        if flat8.any() {
            let (p3, p2, p1, p0, q0, q1, q2, q3) = (
                p3.cast::<u16>(),
                p2.cast::<u16>(),
                p1.cast::<u16>(),
                p0.cast::<u16>(),
                q0.cast::<u16>(),
                q1.cast::<u16>(),
                q2.cast::<u16>(),
                q3.cast::<u16>(),
            );
            let mut t = k(3) * p3 + k(2) * p2 + p1 + p0 + q0 + k(4);
            s[4] = blend(s[4], flat8, t, 3);
            t = t - p3 - p2 + p1 + q1;
            s[5] = blend(s[5], flat8, t, 3);
            t = t - p3 - p1 + p0 + q2;
            s[6] = blend(s[6], flat8, t, 3);
            t = t - p3 - p0 + q0 + q3;
            s[7] = blend(s[7], flat8, t, 3);
            t = t - p2 - q0 + q1 + q3;
            s[8] = blend(s[8], flat8, t, 3);
            t = t - p1 - q1 + q2 + q3;
            s[9] = blend(s[9], flat8, t, 3);
        }
        return if SIZE == 16 && flat2.any() { 6 } else { 3 };
    }
    2
}

/// Filters across a vertical edge, see
/// [`crate::decoder::av1::dsp::loopfilter::filter_vertical_edge`].
#[allow(clippy::too_many_arguments)]
pub fn filter_vertical_edge<T: Pixel>(
    buf: &mut [T],
    stride: usize,
    x: usize,
    y: usize,
    len: usize,
    filter_size: u32,
    limits: FilterLimits,
    bd: u32,
) {
    match bd {
        8 => vertical_bd::<T, 8>(buf, stride, x, y, len, filter_size, limits),
        10 => vertical_bd::<T, 10>(buf, stride, x, y, len, filter_size, limits),
        12 => vertical_bd::<T, 12>(buf, stride, x, y, len, filter_size, limits),
        _ => panic!("invalid bit depth {bd}"),
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn vertical_bd<T: Pixel, const BD: u32>(
    buf: &mut [T],
    stride: usize,
    x: usize,
    y: usize,
    len: usize,
    filter_size: u32,
    limits: FilterLimits,
) {
    let th = Th::new::<BD>(limits);
    let start = y * stride + x;
    let n = len & !3;
    match filter_size {
        4 => vertical::<T, BD, 4>(buf, start, stride, n, &th),
        6 => vertical::<T, BD, 6>(buf, start, stride, n, &th),
        8 => vertical::<T, BD, 8>(buf, start, stride, n, &th),
        16 => vertical::<T, BD, 16>(buf, start, stride, n, &th),
        _ => panic!("invalid loop filter size {filter_size}"),
    }
    if n < len {
        portable::filter_vertical_edge(buf, stride, x, y + n, len - n, filter_size, limits, BD);
    }
}

#[inline(always)]
fn vchunk16<T: Pixel, const BD: u32, const SIZE: usize, const LINES: usize>(
    buf: &mut [T],
    rows: usize,
    stride: usize,
    th: &Th,
) {
    let mut s = [V::splat(0); 14];
    if SIZE == 4 {
        s[5..9].copy_from_slice(&load_cols4::<T, LINES>(buf, rows - 2, stride));
    } else if SIZE == 6 {
        let v = load_cols8::<T, LINES, 6>(buf, rows - 3, stride);
        s[4..10].copy_from_slice(&v[..6]);
    } else {
        // SIZE == 8 or SIZE == 16: load the inner 8 columns (-4..4) first.
        s[3..11].copy_from_slice(&load_cols8::<T, LINES, 8>(buf, rows - 4, stride));
    }
    let m = filter::<BD, SIZE>(&mut s, valid16::<LINES>(), th, || {
        let lo = load_cols4::<T, LINES>(buf, rows - 7, stride);
        let hi = load_cols4::<T, LINES>(buf, rows + 3, stride);
        ([lo[0], lo[1], lo[2]], [hi[1], hi[2], hi[3]])
    });
    if m == 0 {
        return;
    }
    let z = V::splat(0);
    if m <= 2 {
        store_cols4::<T, LINES>(buf, rows - 2, stride, [s[5], s[6], s[7], s[8]]);
    } else if SIZE >= 8 && m == 3 {
        store_cols8::<T, LINES, 6>(
            buf,
            rows - 3,
            stride,
            [s[4], s[5], s[6], s[7], s[8], s[9], z, z],
        );
    } else if SIZE == 16 {
        store_cols4::<T, LINES>(buf, rows - 6, stride, [s[1], s[2], s[3], s[4]]);
        store_cols8::<T, LINES, 8>(
            buf,
            rows - 2,
            stride,
            [s[5], s[6], s[7], s[8], s[9], s[10], s[11], s[12]],
        );
    }
}

/// Vertical edge: rows of up to 8 samples on each side of the edge are
/// loaded for 8 (or 4) lines and transposed so that each vector holds one
/// column.
#[inline(always)]
fn vertical<T: Pixel, const BD: u32, const SIZE: usize>(
    buf: &mut [T],
    start: usize,
    stride: usize,
    len: usize,
    th: &Th,
) {
    let mut i = 0;
    while i + 8 <= len {
        vchunk16::<T, BD, SIZE, 8>(buf, start + i * stride, stride, th);
        i += 8;
    }
    if i < len {
        vchunk16::<T, BD, SIZE, 4>(buf, start + i * stride, stride, th);
    }
}

/// Filters across a horizontal edge, see
/// [`crate::decoder::av1::dsp::loopfilter::filter_horizontal_edge`].
#[allow(clippy::too_many_arguments)]
pub fn filter_horizontal_edge<T: Pixel>(
    buf: &mut [T],
    stride: usize,
    x: usize,
    y: usize,
    len: usize,
    filter_size: u32,
    limits: FilterLimits,
    bd: u32,
) {
    match bd {
        8 => horizontal_bd::<T, 8>(buf, stride, x, y, len, filter_size, limits),
        10 => horizontal_bd::<T, 10>(buf, stride, x, y, len, filter_size, limits),
        12 => horizontal_bd::<T, 12>(buf, stride, x, y, len, filter_size, limits),
        _ => panic!("invalid bit depth {bd}"),
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn horizontal_bd<T: Pixel, const BD: u32>(
    buf: &mut [T],
    stride: usize,
    x: usize,
    y: usize,
    len: usize,
    filter_size: u32,
    limits: FilterLimits,
) {
    let th = Th::new::<BD>(limits);
    let start = y * stride + x;
    // The decoder only filters multiples of 4 lines; any other remainder
    // is left to the portable kernel.
    let n = len & !3;
    match filter_size {
        4 => horizontal::<T, BD, 4>(buf, start, stride, n, &th),
        6 => horizontal::<T, BD, 6>(buf, start, stride, n, &th),
        8 => horizontal::<T, BD, 8>(buf, start, stride, n, &th),
        16 => horizontal::<T, BD, 16>(buf, start, stride, n, &th),
        _ => panic!("invalid loop filter size {filter_size}"),
    }
    if n < len {
        portable::filter_horizontal_edge(buf, stride, x + n, y, len - n, filter_size, limits, BD);
    }
}

#[inline(always)]
fn hchunk16<T: Pixel, const BD: u32, const SIZE: usize, const LINES: usize>(
    buf: &mut [T],
    pos: usize,
    stride: usize,
    th: &Th,
) {
    let stride = stride & 0x0fff_ffff;
    let mut s = [V::splat(0); 14];
    if SIZE == 4 {
        let win = &buf[pos - 2 * stride..][..3 * stride + LINES];
        for k in 0..4 {
            s[5 + k] = load::<T, LINES>(&win[k * stride..]);
        }
    } else if SIZE == 6 {
        let win = &buf[pos - 3 * stride..][..5 * stride + LINES];
        for k in 0..6 {
            s[4 + k] = load::<T, LINES>(&win[k * stride..]);
        }
    } else {
        let win = &buf[pos - 4 * stride..][..7 * stride + LINES];
        for k in 0..8 {
            s[3 + k] = load::<T, LINES>(&win[k * stride..]);
        }
    }
    let m = filter::<BD, SIZE>(&mut s, valid16::<LINES>(), th, || {
        let win = &buf[pos - 7 * stride..][..13 * stride + LINES];
        (
            [
                load::<T, LINES>(&win[0..]),
                load::<T, LINES>(&win[stride..]),
                load::<T, LINES>(&win[2 * stride..]),
            ],
            [
                load::<T, LINES>(&win[11 * stride..]),
                load::<T, LINES>(&win[12 * stride..]),
                load::<T, LINES>(&win[13 * stride..]),
            ],
        )
    });
    if m == 0 {
        return;
    }
    if m == 2 {
        let win = &mut buf[pos - 2 * stride..][..3 * stride + LINES];
        for k in 0..4 {
            store::<T, LINES>(&mut win[k * stride..], s[5 + k]);
        }
    } else if SIZE >= 8 && m == 3 {
        let win = &mut buf[pos - 3 * stride..][..5 * stride + LINES];
        for k in 0..6 {
            store::<T, LINES>(&mut win[k * stride..], s[4 + k]);
        }
    } else if SIZE == 16 {
        let win = &mut buf[pos - 6 * stride..][..11 * stride + LINES];
        for k in 0..12 {
            store::<T, LINES>(&mut win[k * stride..], s[1 + k]);
        }
    }
}

/// Horizontal edge: each row is loaded as one vector of 8 (or 4) columns.
#[inline(always)]
fn horizontal<T: Pixel, const BD: u32, const SIZE: usize>(
    buf: &mut [T],
    start: usize,
    stride: usize,
    len: usize,
    th: &Th,
) {
    let mut j = 0;
    while j + 8 <= len {
        hchunk16::<T, BD, SIZE, 8>(buf, start + j, stride, th);
        j += 8;
    }
    if j < len {
        hchunk16::<T, BD, SIZE, 4>(buf, start + j, stride, th);
    }
}

/// 8-bit kernels on `u8` lanes, like dav1d's 8bpc NEON loop filters: up
/// to 16 lines per vector (16 for runs of at least 16 lines, else 8).
///
/// Exactness (8-bit samples, limits of `dsp::loopfilter::filter_limits`,
/// i.e. `limit <= 63`, `blimit <= 193`, `thresh <= 3`, checked by the
/// caller):
///
/// * absolute differences are `max - min` in `u8`; `2 * |p0 - q0| +
///   (|p1 - q1| >> 1)` is computed with saturating additions, which
///   leaves its comparison with `blimit < 255` unchanged;
/// * the narrow filter's `Clip3(-128, 127, ..)` of 8-bit signed values,
///   including `filter + 3 * (qs0 - ps0)`, is exact in saturating `i8`
///   arithmetic (`base + d + d + d` with `d = qs0.saturating_sub(ps0)`);
/// * the wide filter sums are below `16 * 255 + 8` and computed in `u16`.
mod lanes8 {
    use super::*;

    type B<const N: usize> = Simd<u8, N>;
    type MB<const N: usize> = Mask<i8, N>;

    #[inline(always)]
    fn valid8<const N: usize, const LINES: usize>() -> MB<N> {
        if LINES == N {
            MB::splat(true)
        } else {
            MB::from_array(std::array::from_fn(|i| i < LINES))
        }
    }

    /// Thresholds (splatted, or per lane for the group kernels).
    pub(super) struct Th8<const N: usize> {
        pub(super) limit: B<N>,
        pub(super) blimit: B<N>,
        pub(super) thresh: B<N>,
    }

    impl<const N: usize> Th8<N> {
        #[inline(always)]
        pub(super) fn new(l: FilterLimits) -> Self {
            Th8 {
                limit: B::splat(l.limit as u8),
                blimit: B::splat(l.blimit as u8),
                thresh: B::splat(l.thresh as u8),
            }
        }
    }

    /// Whether the `u8` kernels are exact for these limits (see the
    /// module documentation).
    #[inline(always)]
    pub(super) fn supported(l: FilterLimits) -> bool {
        (0..=63).contains(&l.limit) && (0..=254).contains(&l.blimit) && (0..=63).contains(&l.thresh)
    }

    #[inline(always)]
    fn abd<const N: usize>(a: B<N>, b: B<N>) -> B<N> {
        a.simd_max(b) - a.simd_min(b)
    }

    /// `s[7 + k]` holds the samples at signed offset `k` from `q0` (as in
    /// the 16-bit `filter`). Returns the modified radius `m` (`0`, `1`,
    /// `2`, `3` or `6`).
    ///
    /// `SIZE` is the widest filter of the lanes. `wide` marks the lanes
    /// whose own size is above 4 (their filter mask includes the outer
    /// taps and they may take the flat filters) and `w16` the lanes of
    /// size 16 (which may take the 16 tap filter); both are all true
    /// when every lane has size `SIZE`.
    #[inline(always)]
    pub(super) fn filter<const N: usize, const SIZE: usize>(
        s: &mut [B<N>; 14],
        valid: MB<N>,
        wide: MB<N>,
        w16: MB<N>,
        th: &Th8<N>,
        load_outer: impl FnOnce() -> ([B<N>; 3], [B<N>; 3]),
    ) -> usize {
        let [_, _, _, p3, p2, p1, p0, q0, q1, q2, q3, _, _, _] = *s;
        let one = B::<N>::splat(1);
        // max(|p1 - p0|, |q1 - q0|): shared by the mask, hev and flat.
        let inner = abd(p1, p0).simd_max(abd(q1, q0));
        let mut m = inner;
        if SIZE >= 6 {
            let mut ext = abd(p2, p1).simd_max(abd(q2, q1));
            if SIZE >= 8 {
                ext = ext.simd_max(abd(p3, p2)).simd_max(abd(q3, q2));
            }
            m = m.simd_max(wide.select(ext, B::splat(0)));
        }
        let ad0 = abd(p0, q0);
        let e = ad0.saturating_add(ad0).saturating_add(abd(p1, q1) >> one);
        let filt = valid & m.simd_le(th.limit) & e.simd_le(th.blimit);
        if !filt.any() {
            return 0;
        }
        let flat = if SIZE >= 6 {
            let mut f = inner.simd_max(abd(p2, p0)).simd_max(abd(q2, q0));
            if SIZE >= 8 {
                f = f.simd_max(abd(p3, p0)).simd_max(abd(q3, q0));
            }
            filt & wide & f.simd_le(one)
        } else {
            MB::splat(false)
        };

        let narrow = filt & !flat;
        if narrow.any() {
            type S<const N: usize> = Simd<i8, N>;
            let hev = inner.simd_gt(th.thresh);
            let k80 = B::<N>::splat(0x80);
            let sg = |v: B<N>| (v ^ k80).cast::<i8>();
            let us = |v: S<N>| v.cast::<u8>() ^ k80;
            let (ps1, ps0, qs0, qs1) = (sg(p1), sg(p0), sg(q0), sg(q1));
            let base = hev.select(ps1.saturating_sub(qs1), S::splat(0));
            let d = qs0.saturating_sub(ps0);
            let f = base.saturating_add(d).saturating_add(d).saturating_add(d);
            let f1 = f.saturating_add(S::splat(4)) >> S::splat(3);
            let f2 = f.saturating_add(S::splat(3)) >> S::splat(3);
            s[7] = narrow.select(us(qs0.saturating_sub(f1)), q0);
            s[6] = narrow.select(us(ps0.saturating_add(f2)), p0);
            let outer = narrow & !hev;
            let f3 = (f1 + S::splat(1)) >> S::splat(1);
            s[8] = outer.select(us(qs1.saturating_sub(f3)), q1);
            s[5] = outer.select(us(ps1.saturating_add(f3)), p1);
        }

        if SIZE >= 6 && flat.any() {
            type U<const N: usize> = Simd<u16, N>;
            let k = |v: u16| U::<N>::splat(v);
            let blend = |orig: B<N>, m: MB<N>, sum: U<N>, shift: u16| {
                m.select((sum >> k(shift)).cast::<u8>(), orig)
            };
            let add2 = |a: B<N>, b: B<N>| a.cast::<u16>() + b.cast::<u16>();
            if SIZE == 6 {
                let (p2, p1, p0, q0, q1, q2) = (
                    p2.cast::<u16>(),
                    p1.cast::<u16>(),
                    p0.cast::<u16>(),
                    q0.cast::<u16>(),
                    q1.cast::<u16>(),
                    q2.cast::<u16>(),
                );
                let p0q0 = p0 + q0 + k(2);
                s[5] = blend(s[5], flat, k(3) * p2 + k(2) * (p1 + p0) + q0 + k(4), 3);
                s[6] = blend(s[6], flat, p2 + k(2) * (p1 + p0q0) + q1, 3);
                s[7] = blend(s[7], flat, p1 + k(2) * (p0q0 + q1) + q2, 3);
                s[8] = blend(s[8], flat, p0 + k(2) * (q0 + q1) + k(3) * q2 + k(4), 3);
                return 2;
            }
            let flat2 = if SIZE == 16 && (flat & w16).any() {
                let ([p6, p5, p4], [q4, q5, q6]) = load_outer();
                s[1] = p5;
                s[2] = p4;
                s[11] = q4;
                s[12] = q5;
                let mut f = abd(p4, p0).simd_max(abd(q4, q0));
                f = f.simd_max(abd(p5, p0)).simd_max(abd(q5, q0));
                f = f.simd_max(abd(p6, p0)).simd_max(abd(q6, q0));
                let flat2 = flat & w16 & f.simd_le(one);
                if flat2.any() && HALVES && N == 16 {
                    let v = [p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6];
                    let lo = flat16_h(widen_all::<N, false, 14>(&v));
                    let hi = flat16_h(widen_all::<N, true, 14>(&v));
                    for i in 0..12 {
                        s[1 + i] = flat2.select(from16(pack16(lo[i], hi[i])), s[1 + i]);
                    }
                } else if flat2.any() {
                    let mut t = k(7) * p6.cast::<u16>()
                        + k(2) * add2(p5, p4)
                        + add2(p3, p2)
                        + add2(p1, p0)
                        + q0.cast::<u16>()
                        + k(8);
                    s[1] = blend(s[1], flat2, t, 4);
                    t = t + add2(p3, q1) - add2(p6, p6);
                    s[2] = blend(s[2], flat2, t, 4);
                    t = t + add2(p2, q2) - add2(p6, p5);
                    s[3] = blend(s[3], flat2, t, 4);
                    t = t + add2(p1, q3) - add2(p6, p4);
                    s[4] = blend(s[4], flat2, t, 4);
                    t = t + add2(p0, q4) - add2(p6, p3);
                    s[5] = blend(s[5], flat2, t, 4);
                    t = t + add2(q0, q5) - add2(p6, p2);
                    s[6] = blend(s[6], flat2, t, 4);
                    t = t + add2(q1, q6) - add2(p6, p1);
                    s[7] = blend(s[7], flat2, t, 4);
                    t = t + add2(q2, q6) - add2(p5, p0);
                    s[8] = blend(s[8], flat2, t, 4);
                    t = t + add2(q3, q6) - add2(p4, q0);
                    s[9] = blend(s[9], flat2, t, 4);
                    t = t + add2(q4, q6) - add2(p3, q1);
                    s[10] = blend(s[10], flat2, t, 4);
                    t = t + add2(q5, q6) - add2(p2, q2);
                    s[11] = blend(s[11], flat2, t, 4);
                    t = t + add2(q6, q6) - add2(p1, q3);
                    s[12] = blend(s[12], flat2, t, 4);
                }
                flat2
            } else {
                MB::splat(false)
            };
            let flat8 = flat & !flat2;
            if flat8.any() && HALVES && N == 16 {
                let v = [p3, p2, p1, p0, q0, q1, q2, q3];
                let lo = flat8_h(widen_all::<N, false, 8>(&v));
                let hi = flat8_h(widen_all::<N, true, 8>(&v));
                for i in 0..6 {
                    s[4 + i] = flat8.select(from16(pack16(lo[i], hi[i])), s[4 + i]);
                }
            } else if flat8.any() {
                let (p3, p2, p1, p0, q0, q1, q2, q3) = (
                    p3.cast::<u16>(),
                    p2.cast::<u16>(),
                    p1.cast::<u16>(),
                    p0.cast::<u16>(),
                    q0.cast::<u16>(),
                    q1.cast::<u16>(),
                    q2.cast::<u16>(),
                    q3.cast::<u16>(),
                );
                let mut t = k(3) * p3 + k(2) * p2 + p1 + p0 + q0 + k(4);
                s[4] = blend(s[4], flat8, t, 3);
                t = t - p3 - p2 + p1 + q1;
                s[5] = blend(s[5], flat8, t, 3);
                t = t - p3 - p1 + p0 + q2;
                s[6] = blend(s[6], flat8, t, 3);
                t = t - p3 - p0 + q0 + q3;
                s[7] = blend(s[7], flat8, t, 3);
                t = t - p2 - q0 + q1 + q3;
                s[8] = blend(s[8], flat8, t, 3);
                t = t - p1 - q1 + q2 + q3;
                s[9] = blend(s[9], flat8, t, 3);
            }
            return if SIZE == 16 && flat2.any() { 6 } else { 3 };
        }
        2
    }

    /// Whether the 8 and 16 tap flat filters of 16 lane vectors are
    /// evaluated on 8 lane halves: on x86 (16 SSE2 registers) a full width
    /// `u16` evaluation keeps twice as many vectors live and spills.
    const HALVES: bool = cfg!(not(target_arch = "aarch64"));

    type H = Simd<u16, 8>;

    /// `v` as a 16 lane vector (only used when `N == 16`).
    #[inline(always)]
    fn to16<const N: usize>(v: B<N>) -> Simd<u8, 16> {
        Simd::from_slice(&v.as_array()[..16])
    }

    /// Inverse of [`to16`] (only used when `N == 16`).
    #[inline(always)]
    fn from16<const N: usize>(v: Simd<u8, 16>) -> B<N> {
        let mut a = [0u8; N];
        a[..16].copy_from_slice(v.as_array());
        B::from_array(a)
    }

    /// The low (`HI == false`) or high 8 lanes of `v`, widened.
    #[inline(always)]
    fn widen<const HI: bool>(v: Simd<u8, 16>) -> H {
        if HI {
            simd_swizzle!(v, [8, 9, 10, 11, 12, 13, 14, 15]).cast()
        } else {
            simd_swizzle!(v, [0, 1, 2, 3, 4, 5, 6, 7]).cast()
        }
    }

    /// [`widen`] of every vector of `v` (only used when `N == 16`).
    #[inline(always)]
    fn widen_all<const N: usize, const HI: bool, const K: usize>(v: &[B<N>; K]) -> [H; K] {
        let mut r = [H::splat(0); K];
        for i in 0..K {
            r[i] = widen::<HI>(to16(v[i]));
        }
        r
    }

    /// Packs two halves (all lanes `< 256`).
    #[inline(always)]
    fn pack16(lo: H, hi: H) -> Simd<u8, 16> {
        let (a, b): (Simd<u8, 8>, Simd<u8, 8>) = (lo.cast(), hi.cast());
        simd_swizzle!(a, b, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    }

    /// The 8 tap flat filter outputs `p2..=q2` from `p3..=q3`.
    #[inline(always)]
    fn flat8_h(v: [H; 8]) -> [H; 6] {
        let [p3, p2, p1, p0, q0, q1, q2, q3] = v;
        let k = H::splat;
        let mut t = k(3) * p3 + k(2) * p2 + p1 + p0 + q0 + k(4);
        let o0 = t >> k(3);
        t = t - p3 - p2 + p1 + q1;
        let o1 = t >> k(3);
        t = t - p3 - p1 + p0 + q2;
        let o2 = t >> k(3);
        t = t - p3 - p0 + q0 + q3;
        let o3 = t >> k(3);
        t = t - p2 - q0 + q1 + q3;
        let o4 = t >> k(3);
        t = t - p1 - q1 + q2 + q3;
        [o0, o1, o2, o3, o4, t >> k(3)]
    }

    /// The 16 tap flat filter outputs `p5..=q5` from `p6..=q6`.
    #[inline(always)]
    fn flat16_h(v: [H; 14]) -> [H; 12] {
        let [p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6] = v;
        let k = H::splat;
        let mut o = [k(0); 12];
        let mut t = k(7) * p6 + k(2) * (p5 + p4) + p3 + p2 + p1 + p0 + q0 + k(8);
        o[0] = t >> k(4);
        t = t + p3 + q1 - p6 - p6;
        o[1] = t >> k(4);
        t = t + p2 + q2 - p6 - p5;
        o[2] = t >> k(4);
        t = t + p1 + q3 - p6 - p4;
        o[3] = t >> k(4);
        t = t + p0 + q4 - p6 - p3;
        o[4] = t >> k(4);
        t = t + q0 + q5 - p6 - p2;
        o[5] = t >> k(4);
        t = t + q1 + q6 - p6 - p1;
        o[6] = t >> k(4);
        t = t + q2 + q6 - p5 - p0;
        o[7] = t >> k(4);
        t = t + q3 + q6 - p4 - q0;
        o[8] = t >> k(4);
        t = t + q4 + q6 - p3 - q1;
        o[9] = t >> k(4);
        t = t + q5 + q6 - p2 - q2;
        o[10] = t >> k(4);
        t = t + q6 + q6 - p1 - q3;
        o[11] = t >> k(4);
        o
    }

    struct Trn<const G: usize, const HI: bool>;
    impl<const N: usize, const G: usize, const HI: bool> Swizzle<N> for Trn<G, HI> {
        const INDEX: [usize; N] = trn_index::<N>(G, HI);
    }

    #[inline(always)]
    fn trn<const N: usize, const G: usize>(a: B<N>, b: B<N>) -> (B<N>, B<N>) {
        (
            Trn::<G, false>::concat_swizzle(a, b),
            Trn::<G, true>::concat_swizzle(a, b),
        )
    }

    /// Transposes the 4x4 byte matrices held in each 4 byte quarter of 4
    /// vectors: row `4 * g + k` in bytes `4 * g..4 * g + 4` of `v[k]`
    /// becomes column `col` (rows `0..N` in order) of output `col`.
    #[inline(always)]
    fn transpose4<const N: usize>(v: [B<N>; 4]) -> [B<N>; 4] {
        let (t0, t1) = trn::<N, 1>(v[0], v[1]);
        let (t2, t3) = trn::<N, 1>(v[2], v[3]);
        let (c0, c2) = trn::<N, 2>(t0, t2);
        let (c1, c3) = trn::<N, 2>(t1, t3);
        [c0, c1, c2, c3]
    }

    /// Transposes the 8x8 byte matrices held in each 8 byte half of 8
    /// vectors (for `N = 16`, two independent ones): row `k` of a matrix
    /// in vector `k` becomes column `k`.
    #[inline(always)]
    fn transpose8<const N: usize>(v: [B<N>; 8]) -> [B<N>; 8] {
        let (t0, t1) = trn::<N, 1>(v[0], v[1]);
        let (t2, t3) = trn::<N, 1>(v[2], v[3]);
        let (t4, t5) = trn::<N, 1>(v[4], v[5]);
        let (t6, t7) = trn::<N, 1>(v[6], v[7]);
        let (u0, u2) = trn::<N, 2>(t0, t2);
        let (u1, u3) = trn::<N, 2>(t1, t3);
        let (u4, u6) = trn::<N, 2>(t4, t6);
        let (u5, u7) = trn::<N, 2>(t5, t7);
        let (c0, c4) = trn::<N, 4>(u0, u4);
        let (c1, c5) = trn::<N, 4>(u1, u5);
        let (c2, c6) = trn::<N, 4>(u2, u6);
        let (c3, c7) = trn::<N, 4>(u3, u7);
        [c0, c1, c2, c3, c4, c5, c6, c7]
    }

    #[inline(always)]
    fn load4<T: Pixel>(src: &[T]) -> u32 {
        let src: &[T; 4] = src[..4].try_into().unwrap();
        u32::from_ne_bytes(std::array::from_fn(|i| src[i].to_i32() as u8))
    }

    #[inline(always)]
    fn store4<T: Pixel>(dst: &mut [T], v: u32) {
        let b = v.to_ne_bytes();
        let dst: &mut [T; 4] = (&mut dst[..4]).try_into().unwrap();
        for i in 0..4 {
            dst[i] = T::from_i32(b[i] as i32);
        }
    }

    #[inline(always)]
    fn load_row4_8<T: Pixel, const N: usize, const LINES: usize>(
        win: &[T],
        pos: usize,
        stride: usize,
    ) -> B<N> {
        let r0 = load4(&win[pos..]);
        let r1 = if LINES >= 8 { load4(&win[pos + 4 * stride..]) } else { 0 };
        if N == 8 {
            B::from_slice(
                Simd::<u32, 2>::from_array([r0, r1])
                    .to_ne_bytes()
                    .as_array(),
            )
        } else {
            let r2 = if LINES == 16 { load4(&win[pos + 8 * stride..]) } else { 0 };
            let r3 = if LINES == 16 { load4(&win[pos + 12 * stride..]) } else { 0 };
            B::from_slice(
                Simd::<u32, 4>::from_array([r0, r1, r2, r3])
                    .to_ne_bytes()
                    .as_array(),
            )
        }
    }

    /// Loads 4 columns for `LINES` (4, 8 or 16) lines starting at `pos`
    /// and transposes them via [`transpose4`].
    #[inline(always)]
    fn load_cols4<T: Pixel, const N: usize, const LINES: usize>(
        buf: &[T],
        pos: usize,
        stride: usize,
    ) -> [B<N>; 4] {
        let stride = stride & 0x0fff_ffff;
        let win = &buf[pos..][..(LINES - 1) * stride + 4];
        transpose4::<N>([
            load_row4_8::<T, N, LINES>(win, 0, stride),
            load_row4_8::<T, N, LINES>(win, stride, stride),
            load_row4_8::<T, N, LINES>(win, 2 * stride, stride),
            load_row4_8::<T, N, LINES>(win, 3 * stride, stride),
        ])
    }

    /// Transposes 4 column vectors via [`transpose4`] and stores 4 columns
    /// for `LINES` (4, 8 or 16) lines starting at `pos`.
    #[inline(always)]
    fn store_cols4<T: Pixel, const N: usize, const LINES: usize>(
        buf: &mut [T],
        pos: usize,
        stride: usize,
        cols: [B<N>; 4],
    ) {
        let stride = stride & 0x0fff_ffff;
        let win = &mut buf[pos..][..(LINES - 1) * stride + 4];
        let v = transpose4::<N>(cols);
        for (k, vk) in v.iter().enumerate() {
            if N == 8 {
                let [r0, r1] =
                    Simd::<u32, 2>::from_ne_bytes(u8x8::from_slice(vk.as_array())).to_array();
                store4(&mut win[k * stride..], r0);
                if LINES >= 8 {
                    store4(&mut win[(k + 4) * stride..], r1);
                }
            } else {
                let [r0, r1, r2, r3] =
                    Simd::<u32, 4>::from_ne_bytes(u8x16::from_slice(vk.as_array())).to_array();
                store4(&mut win[k * stride..], r0);
                if LINES >= 8 {
                    store4(&mut win[(k + 4) * stride..], r1);
                }
                if LINES == 16 {
                    store4(&mut win[(k + 8) * stride..], r2);
                    store4(&mut win[(k + 12) * stride..], r3);
                }
            }
        }
    }

    /// Loads `K <= 8` samples of a line into the first lanes of 8.
    #[inline(always)]
    fn load8<T: Pixel, const K: usize>(src: &[T]) -> u64 {
        let src: &[T; K] = src[..K].try_into().unwrap();
        u64::from_ne_bytes(std::array::from_fn(|i| {
            if i < K {
                src[i].to_i32() as u8
            } else {
                0
            }
        }))
    }

    /// Stores the first `K` lanes of `v` to `dst[..K]`.
    #[inline(always)]
    fn store8<T: Pixel, const K: usize>(dst: &mut [T], v: u64) {
        let b = v.to_ne_bytes();
        let dst: &mut [T; K] = (&mut dst[..K]).try_into().unwrap();
        for i in 0..K {
            dst[i] = T::from_i32(b[i] as i32);
        }
    }

    /// Loads the samples at columns `0..K` relative to `pos` of `LINES`
    /// lines starting at `pos` and transposes them (see [`transpose8`]).
    #[inline(always)]
    fn load_cols<T: Pixel, const N: usize, const LINES: usize, const K: usize>(
        buf: &[T],
        pos: usize,
        stride: usize,
    ) -> [B<N>; 8] {
        let stride = stride & 0x0fff_ffff;
        let win = &buf[pos..][..(LINES - 1) * stride + K];
        let mut v = [B::<N>::splat(0); 8];
        let rows = if LINES < 8 { LINES } else { 8 };
        for (k, vk) in v[..rows].iter_mut().enumerate() {
            let r0 = load8::<T, K>(&win[k * stride..]);
            if N == 8 {
                *vk = B::from_slice(&r0.to_ne_bytes());
            } else {
                let r1 = if LINES == 16 { load8::<T, K>(&win[(k + 8) * stride..]) } else { 0 };
                *vk = B::from_slice(
                    Simd::<u64, 2>::from_array([r0, r1])
                        .to_ne_bytes()
                        .as_array(),
                );
            }
        }
        transpose8::<N>(v)
    }

    /// Inverse of [`load_cols`] for the first `K` columns of each line.
    #[inline(always)]
    fn store_cols<T: Pixel, const N: usize, const LINES: usize, const K: usize>(
        buf: &mut [T],
        pos: usize,
        stride: usize,
        cols: [B<N>; 8],
    ) {
        let stride = stride & 0x0fff_ffff;
        let win = &mut buf[pos..][..(LINES - 1) * stride + K];
        let v = transpose8::<N>(cols);
        let rows = if LINES < 8 { LINES } else { 8 };
        for (k, vk) in v[..rows].iter().enumerate() {
            if N == 8 {
                let r0 = u64::from_ne_bytes(vk.to_array()[..8].try_into().unwrap());
                store8::<T, K>(&mut win[k * stride..], r0);
            } else {
                let [r0, r1] =
                    Simd::<u64, 2>::from_ne_bytes(u8x16::from_slice(vk.as_array())).to_array();
                store8::<T, K>(&mut win[k * stride..], r0);
                if LINES == 16 {
                    store8::<T, K>(&mut win[(k + 8) * stride..], r1);
                }
            }
        }
    }

    /// `LINES` lines (4, 8 or 16) of a vertical edge, starting at the `q0`
    /// sample `pos`.
    #[inline(always)]
    fn vchunk<T: Pixel, const N: usize, const LINES: usize, const SIZE: usize>(
        buf: &mut [T],
        pos: usize,
        stride: usize,
        valid: MB<N>,
        th: &Th8<N>,
    ) {
        let mut s = [B::<N>::splat(0); 14];
        if SIZE == 4 {
            s[5..9].copy_from_slice(&load_cols4::<T, N, LINES>(buf, pos - 2, stride));
        } else if SIZE == 6 {
            let v = load_cols::<T, N, LINES, 6>(buf, pos - 3, stride);
            s[4..10].copy_from_slice(&v[..6]);
        } else {
            // SIZE == 8 or SIZE == 16: load the inner 8 columns (-4..4) first.
            let v = load_cols::<T, N, LINES, 8>(buf, pos - 4, stride);
            s[3..11].copy_from_slice(&v);
        }
        let all = MB::splat(true);
        let m = filter::<N, SIZE>(&mut s, valid, all, all, th, || {
            let lo = load_cols4::<T, N, LINES>(buf, pos - 7, stride);
            let hi = load_cols4::<T, N, LINES>(buf, pos + 3, stride);
            ([lo[0], lo[1], lo[2]], [hi[1], hi[2], hi[3]])
        });
        if m == 0 {
            return;
        }
        let z = B::<N>::splat(0);
        if m <= 2 {
            store_cols4::<T, N, LINES>(buf, pos - 2, stride, [s[5], s[6], s[7], s[8]]);
        } else if SIZE >= 8 && m == 3 {
            store_cols::<T, N, LINES, 6>(
                buf,
                pos - 3,
                stride,
                [s[4], s[5], s[6], s[7], s[8], s[9], z, z],
            );
        } else if SIZE == 16 {
            store_cols4::<T, N, LINES>(buf, pos - 6, stride, [s[1], s[2], s[3], s[4]]);
            store_cols::<T, N, LINES, 8>(
                buf,
                pos - 2,
                stride,
                [s[5], s[6], s[7], s[8], s[9], s[10], s[11], s[12]],
            );
        }
    }

    /// Loads `LINES` (4, 8 or 16) samples of a row into the first lanes.
    #[inline(always)]
    fn load_row<T: Pixel, const N: usize, const LINES: usize>(src: &[T]) -> B<N> {
        let src: &[T; LINES] = src[..LINES].try_into().unwrap();
        B::from_array(std::array::from_fn(|i| {
            if i < LINES {
                src[i].to_i32() as u8
            } else {
                0
            }
        }))
    }

    #[inline(always)]
    fn store_row<T: Pixel, const N: usize, const LINES: usize>(dst: &mut [T], v: B<N>) {
        let v = v.to_array();
        let dst: &mut [T; LINES] = (&mut dst[..LINES]).try_into().unwrap();
        for i in 0..LINES {
            dst[i] = T::from_i32(v[i] as i32);
        }
    }

    /// `LINES` (4, 8 or 16) columns of a horizontal edge, starting at the
    /// `q0` sample `pos`.
    #[inline(always)]
    fn hchunk<T: Pixel, const N: usize, const LINES: usize, const SIZE: usize>(
        buf: &mut [T],
        pos: usize,
        stride: usize,
        valid: MB<N>,
        th: &Th8<N>,
    ) {
        let stride = stride & 0x0fff_ffff;
        let mut s = [B::<N>::splat(0); 14];
        if SIZE == 4 {
            let win = &buf[pos - 2 * stride..][..3 * stride + LINES];
            for k in 0..4 {
                s[5 + k] = load_row::<T, N, LINES>(&win[k * stride..]);
            }
        } else if SIZE == 6 {
            let win = &buf[pos - 3 * stride..][..5 * stride + LINES];
            for k in 0..6 {
                s[4 + k] = load_row::<T, N, LINES>(&win[k * stride..]);
            }
        } else {
            let win = &buf[pos - 4 * stride..][..7 * stride + LINES];
            for k in 0..8 {
                s[3 + k] = load_row::<T, N, LINES>(&win[k * stride..]);
            }
        }
        let all = MB::splat(true);
        let m = filter::<N, SIZE>(&mut s, valid, all, all, th, || {
            let win = &buf[pos - 7 * stride..][..13 * stride + LINES];
            (
                [
                    load_row::<T, N, LINES>(&win[0..]),
                    load_row::<T, N, LINES>(&win[stride..]),
                    load_row::<T, N, LINES>(&win[2 * stride..]),
                ],
                [
                    load_row::<T, N, LINES>(&win[11 * stride..]),
                    load_row::<T, N, LINES>(&win[12 * stride..]),
                    load_row::<T, N, LINES>(&win[13 * stride..]),
                ],
            )
        });
        if m == 0 {
            return;
        }
        if m == 2 {
            let win = &mut buf[pos - 2 * stride..][..3 * stride + LINES];
            for k in 0..4 {
                store_row::<T, N, LINES>(&mut win[k * stride..], s[5 + k]);
            }
        } else if SIZE >= 8 && m == 3 {
            let win = &mut buf[pos - 3 * stride..][..5 * stride + LINES];
            for k in 0..6 {
                store_row::<T, N, LINES>(&mut win[k * stride..], s[4 + k]);
            }
        } else if SIZE == 16 {
            let win = &mut buf[pos - 6 * stride..][..11 * stride + LINES];
            for k in 0..12 {
                store_row::<T, N, LINES>(&mut win[k * stride..], s[1 + k]);
            }
        }
    }

    /// `len` (a multiple of 4) lines of a vertical edge whose first `q0`
    /// sample is `start`.
    #[inline(always)]
    pub(super) fn vertical<T: Pixel, const SIZE: usize>(
        buf: &mut [T],
        start: usize,
        stride: usize,
        len: usize,
        l: FilterLimits,
    ) {
        let mut i = 0;
        if len >= 16 {
            let t16 = Th8::<16>::new(l);
            while i + 16 <= len {
                vchunk::<T, 16, 16, SIZE>(
                    buf,
                    start + i * stride,
                    stride,
                    valid8::<16, 16>(),
                    &t16,
                );
                i += 16;
            }
        }
        if i < len {
            let t8 = Th8::<8>::new(l);
            if i + 8 <= len {
                vchunk::<T, 8, 8, SIZE>(buf, start + i * stride, stride, valid8::<8, 8>(), &t8);
                i += 8;
            }
            if i < len {
                vchunk::<T, 8, 4, SIZE>(buf, start + i * stride, stride, valid8::<8, 4>(), &t8);
            }
        }
    }

    /// `len` (a multiple of 4) columns of a horizontal edge whose first
    /// `q0` sample is `start`.
    #[inline(always)]
    pub(super) fn horizontal<T: Pixel, const SIZE: usize>(
        buf: &mut [T],
        start: usize,
        stride: usize,
        len: usize,
        l: FilterLimits,
    ) {
        let mut j = 0;
        if len >= 16 {
            let t16 = Th8::<16>::new(l);
            while j + 16 <= len {
                hchunk::<T, 16, 16, SIZE>(buf, start + j, stride, valid8::<16, 16>(), &t16);
                j += 16;
            }
        }
        if j < len {
            let t8 = Th8::<8>::new(l);
            if j + 8 <= len {
                hchunk::<T, 8, 8, SIZE>(buf, start + j, stride, valid8::<8, 8>(), &t8);
                j += 8;
            }
            if j < len {
                hchunk::<T, 8, 4, SIZE>(buf, start + j, stride, valid8::<8, 4>(), &t8);
            }
        }
    }

    #[cfg(test)]
    #[test]
    fn transpose8_halves() {
        let v: [u8x16; 8] =
            std::array::from_fn(|r| u8x16::from_array(std::array::from_fn(|c| (r * 16 + c) as u8)));
        let t = transpose8::<16>(v);
        for (c, col) in t.iter().enumerate() {
            for r in 0..8 {
                assert_eq!(col[r], (r * 16 + c) as u8);
                assert_eq!(col[8 + r], (r * 16 + 8 + c) as u8);
            }
        }
        let v: [u8x8; 8] =
            std::array::from_fn(|r| u8x8::from_array(std::array::from_fn(|c| (r * 8 + c) as u8)));
        let t = transpose8::<8>(v);
        for (c, col) in t.iter().enumerate() {
            for r in 0..8 {
                assert_eq!(col[r], (r * 8 + c) as u8);
            }
        }
        let v4: [u8x16; 4] =
            std::array::from_fn(|r| u8x16::from_array(std::array::from_fn(|c| (r * 16 + c) as u8)));
        let t4 = transpose4::<16>(v4);
        for (c, col) in t4.iter().enumerate() {
            for g in 0..4 {
                for r in 0..4 {
                    assert_eq!(col[4 * g + r], (r * 16 + 4 * g + c) as u8);
                }
            }
        }
    }
}

/// 8-bit whole edge kernels on groups of [`grp::N`] lines (16, or 32 with
/// AVX2), like dav1d's `lpf_{h,v}_sb_*`: every group of `N / 4` segments
/// is filtered at once with per lane sizes and limits, running the
/// widest filter of the group (lanes of smaller size are masked to their
/// own filter). Shuffles are 128-bit lane local (`punpck*` on x86,
/// `zip1` / `zip2` on NEON); with `N == 32` lines `0..16` are in the low
/// 128-bit half and lines `16..32` in the high half of every vector.
mod grp {
    use super::lanes8::{filter, Th8};
    use super::*;

    /// Lines per group.
    pub(super) const N: usize =
        if cfg!(all(target_arch = "x86_64", target_feature = "avx2")) { 32 } else { 16 };
    /// 4-line segments per group.
    pub(super) const GS: usize = N / 4;
    /// 8-byte (64-bit) lanes per vector.
    const N8: usize = N / 8;
    type B = Simd<u8, N>;
    type MB = Mask<i8, N>;

    /// Indices of a 128-bit lane local interleave of `e`-byte elements
    /// (x86 `punpckl*` / `punpckh*` with `hi`, NEON `zip1` / `zip2`).
    const fn zip_index(e: usize, hi: bool) -> [usize; N] {
        let mut r = [0; N];
        let mut i = 0;
        while i < N {
            let (blk, j) = (i / 16, i % 16);
            let (k, w) = (j / e, j % e);
            let el = if hi { 8 / e } else { 0 } + k / 2;
            let idx = blk * 16 + el * e + w;
            r[i] = if k % 2 == 0 { idx } else { N + idx };
            i += 1;
        }
        r
    }

    struct Zip<const E: usize, const HI: bool>;
    impl<const E: usize, const HI: bool> Swizzle<N> for Zip<E, HI> {
        const INDEX: [usize; N] = zip_index(E, HI);
    }

    #[inline(always)]
    fn zip<const E: usize>(a: B, b: B) -> (B, B) {
        (
            Zip::<E, false>::concat_swizzle(a, b),
            Zip::<E, true>::concat_swizzle(a, b),
        )
    }

    #[inline(always)]
    fn zlo<const E: usize>(a: B, b: B) -> B {
        Zip::<E, false>::concat_swizzle(a, b)
    }

    /// Lane `i` holds segment `i / 4` (replicated via two `zip1` / `punpckl`
    /// steps, without a constant table load).
    #[inline(always)]
    fn expand(a: [u8; GS]) -> B {
        let mut u = [0u32; GS];
        u[0] = u32::from_ne_bytes([a[0], a[1], a[2], a[3]]);
        if N == 32 {
            u[4] = u32::from_ne_bytes([a[4], a[5], a[6], a[7]]);
        }
        let v = Simd::<u32, GS>::from_array(u).to_ne_bytes();
        let t = zlo::<1>(v, v);
        zlo::<2>(t, t)
    }

    #[inline(always)]
    fn ld<T: Pixel, const K: usize>(src: &[T]) -> [u8; K] {
        let src: &[T; K] = src[..K].try_into().unwrap();
        std::array::from_fn(|i| src[i].to_i32() as u8)
    }

    #[inline(always)]
    fn ld4<T: Pixel>(src: &[T]) -> u32 {
        u32::from_ne_bytes(ld::<T, 4>(src))
    }

    #[inline(always)]
    fn ld8<T: Pixel>(src: &[T]) -> u64 {
        u64::from_ne_bytes(ld::<T, 8>(src))
    }

    #[inline(always)]
    fn st<T: Pixel, const K: usize>(dst: &mut [T], v: &[u8]) {
        let dst: &mut [T; K] = (&mut dst[..K]).try_into().unwrap();
        for i in 0..K {
            dst[i] = T::from_i32(v[i] as i32);
        }
    }

    /// `[e(0), e(1), .., e(7)]` for the expression `e(r)`, unrolled.
    macro_rules! arr8 {
        ($r:ident => $e:expr) => {
            arr8!(@ $r => $e; 0 1 2 3 4 5 6 7)
        };
        (@ $r:ident => $e:expr; $($i:literal)*) => {
            [$({
                let $r: usize = $i;
                $e
            }),*]
        };
    }

    /// Runs the block `b(r)` for `r` in `0..16`, unrolled.
    macro_rules! for16 {
        ($r:ident => $b:block) => {
            for16!(@ $r => $b; 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
        };
        (@ $r:ident => $b:block; $($i:literal)*) => {
            $({
                let $r: usize = $i;
                $b
            })*
        };
    }

    /// Packs 4 samples of lines `d, d + 4, d + 8, d + 12` (and `16 + ..`
    /// in the high half when `N == 32`) of `win`.
    #[inline(always)]
    fn line4<T: Pixel>(win: &[T], d: usize, stride: usize) -> B {
        let mut a = [0u32; GS];
        a[0] = ld4(&win[d * stride..]);
        a[1] = ld4(&win[(d + 4) * stride..]);
        a[2] = ld4(&win[(d + 8) * stride..]);
        a[3] = ld4(&win[(d + 12) * stride..]);
        if N == 32 {
            a[4] = ld4(&win[(d + 16) * stride..]);
            a[5] = ld4(&win[(d + 20) * stride..]);
            a[6] = ld4(&win[(d + 24) * stride..]);
            a[7] = ld4(&win[(d + 28) * stride..]);
        }
        Simd::<u32, GS>::from_array(a).to_ne_bytes()
    }

    /// Packs 8 samples of lines `d, d + 8` (and `16 + d, 24 + d` in the
    /// high half when `N == 32`) of `win`.
    #[inline(always)]
    fn line8<T: Pixel>(win: &[T], d: usize, stride: usize) -> B {
        let mut a = [0u64; N8];
        a[0] = ld8(&win[d * stride..]);
        a[1] = ld8(&win[(d + 8) * stride..]);
        if N == 32 {
            a[2] = ld8(&win[(d + 16) * stride..]);
            a[3] = ld8(&win[(d + 24) * stride..]);
        }
        Simd::<u64, N8>::from_array(a).to_ne_bytes()
    }

    /// Columns `0..4` of the `N` lines starting at `pos`.
    #[inline(always)]
    fn load_cols4<T: Pixel>(buf: &[T], pos: usize, stride: usize) -> [B; 4] {
        let stride = stride & 0x0fff_ffff;
        let win = &buf[pos..][..(N - 1) * stride + 4];
        let v = [
            line4(win, 0, stride),
            line4(win, 1, stride),
            line4(win, 2, stride),
            line4(win, 3, stride),
        ];
        let (t0, t1) = zip::<1>(v[0], v[1]);
        let (t2, t3) = zip::<1>(v[2], v[3]);
        let (u0, u1) = zip::<2>(t0, t2);
        let (u2, u3) = zip::<2>(t1, t3);
        let (w0, w1) = zip::<4>(u0, u1);
        let (w2, w3) = zip::<4>(u2, u3);
        let (k0, k1) = zip::<8>(w0, w2);
        let (k2, k3) = zip::<8>(w1, w3);
        [k0, k1, k2, k3]
    }

    /// Columns `0..8` of the `N` lines starting at `pos`.
    #[inline(always)]
    fn load_cols8<T: Pixel>(buf: &[T], pos: usize, stride: usize) -> [B; 8] {
        let stride = stride & 0x0fff_ffff;
        let win = &buf[pos..][..(N - 1) * stride + 8];
        let v: [B; 8] = arr8!(d => line8(win, d, stride));
        let (a0, b0) = zip::<1>(v[0], v[1]);
        let (a1, b1) = zip::<1>(v[2], v[3]);
        let (a2, b2) = zip::<1>(v[4], v[5]);
        let (a3, b3) = zip::<1>(v[6], v[7]);
        let (c0, c1) = zip::<2>(a0, a1);
        let (c2, c3) = zip::<2>(a2, a3);
        let (c4, c5) = zip::<2>(b0, b1);
        let (c6, c7) = zip::<2>(b2, b3);
        let (d0, d1) = zip::<4>(c0, c2);
        let (d2, d3) = zip::<4>(c1, c3);
        let (d4, d5) = zip::<4>(c4, c6);
        let (d6, d7) = zip::<4>(c5, c7);
        let (k0, k1) = zip::<8>(d0, d4);
        let (k2, k3) = zip::<8>(d1, d5);
        let (k4, k5) = zip::<8>(d2, d6);
        let (k6, k7) = zip::<8>(d3, d7);
        [k0, k1, k2, k3, k4, k5, k6, k7]
    }

    /// Stores 4 columns to the `N` lines starting at `pos`.
    #[inline(always)]
    fn store_cols4<T: Pixel>(buf: &mut [T], pos: usize, stride: usize, c: [B; 4]) {
        let stride = stride & 0x0fff_ffff;
        let win = &mut buf[pos..][..(N - 1) * stride + 4];
        // [r: c0 c1] for rows 0..8 / 8..16
        let (a0, a1) = zip::<1>(c[0], c[1]);
        let (b0, b1) = zip::<1>(c[2], c[3]);
        // [r: c0..c3] for rows 8y + 4z .. + 4
        let (r0, r1) = zip::<2>(a0, b0);
        let (r2, r3) = zip::<2>(a1, b1);
        // Row r is 32-bit lane r % 4 of a[r / 4] (and lane 4 + r % 4 for
        // line 16 + r).
        let a = [r0, r1, r2, r3].map(|v| Simd::<u32, GS>::from_ne_bytes(v));
        for16!(r => {
            let (k, d) = (r / 4, r % 4);
            st4(&mut win[r * stride..], a[k][d]);
            if N == 32 {
                st4(&mut win[(16 + r) * stride..], a[k][4 + d]);
            }
        });
    }

    /// Stores the 4 bytes of `x` (in memory order) as samples.
    #[inline(always)]
    fn st4<T: Pixel>(dst: &mut [T], x: u32) {
        let b = x.to_ne_bytes();
        let dst: &mut [T; 4] = (&mut dst[..4]).try_into().unwrap();
        for i in 0..4 {
            dst[i] = T::from_i32(b[i] as i32);
        }
    }

    /// Stores the 8 bytes of `x` (in memory order) as samples.
    #[inline(always)]
    fn st8<T: Pixel>(dst: &mut [T], x: u64) {
        let b = x.to_ne_bytes();
        let dst: &mut [T; 8] = (&mut dst[..8]).try_into().unwrap();
        for i in 0..8 {
            dst[i] = T::from_i32(b[i] as i32);
        }
    }

    /// Stores 8 columns to the `N` lines starting at `pos`.
    #[inline(always)]
    fn store_cols8<T: Pixel>(buf: &mut [T], pos: usize, stride: usize, c: &[B; 8]) {
        let stride = stride & 0x0fff_ffff;
        let win = &mut buf[pos..][..(N - 1) * stride + 8];
        // s1[2i + y]: [r: c2i c2i+1] for rows 8y..8y+8
        let mut s1 = [B::splat(0); 8];
        for i in 0..4 {
            (s1[2 * i], s1[2 * i + 1]) = zip::<1>(c[2 * i], c[2 * i + 1]);
        }
        // s2[(2j + y) * 2 + z]: [r: c4j..c4j+3] for rows 8y + 4z..+4
        let mut s2 = [B::splat(0); 8];
        for j in 0..2 {
            for y in 0..2 {
                let i = (2 * j + y) * 2;
                (s2[i], s2[i + 1]) = zip::<2>(s1[4 * j + y], s1[4 * j + 2 + y]);
            }
        }
        // a[(2y + z) * 2 + w]: rows 8y + 4z + 2w + (0, 1), as 64-bit lanes
        // (lanes 2 and 3 for lines 16 + ..).
        let mut a = [Simd::<u64, N8>::splat(0); 8];
        for y in 0..2 {
            for z in 0..2 {
                let (lo, hi) = zip::<4>(s2[2 * y + z], s2[4 + 2 * y + z]);
                a[(2 * y + z) * 2] = Simd::<u64, N8>::from_ne_bytes(lo);
                a[(2 * y + z) * 2 + 1] = Simd::<u64, N8>::from_ne_bytes(hi);
            }
        }
        for16!(r => {
            let (k, q) = (r / 2, r % 2);
            st8(&mut win[r * stride..], a[k][q]);
            if N == 32 {
                st8(&mut win[(16 + r) * stride..], a[k][2 + q]);
            }
        });
    }

    /// Per lane masks and thresholds of a group with segment sizes `sz`
    /// and levels `lv`.
    #[inline(always)]
    fn masks_th<const SIZE: usize>(
        sz: [u8; GS],
        lv: [u8; GS],
        lut: &[FilterLimits; 64],
    ) -> (MB, MB, MB, Th8<N>) {
        let sv = expand(sz);
        let valid = sv.simd_ne(B::splat(0));
        let wide = if SIZE >= 6 { sv.simd_gt(B::splat(4)) } else { MB::splat(false) };
        let w16 = if SIZE == 16 { sv.simd_eq(B::splat(16)) } else { MB::splat(false) };
        let th = if lv == [lv[0]; GS] {
            let f = lut[(lv[0] & 63) as usize];
            Th8 {
                limit: B::splat(f.limit as u8),
                blimit: B::splat(f.blimit as u8),
                thresh: B::splat(f.thresh as u8),
            }
        } else {
            let f: [FilterLimits; GS] = std::array::from_fn(|i| lut[(lv[i] & 63) as usize]);
            Th8 {
                limit: expand(std::array::from_fn(|i| f[i].limit as u8)),
                blimit: expand(std::array::from_fn(|i| f[i].blimit as u8)),
                thresh: expand(std::array::from_fn(|i| f[i].thresh as u8)),
            }
        };
        (valid, wide, w16, th)
    }

    /// One group of a vertical edge whose first `q0` sample is `pos`.
    #[inline(never)]
    fn vgroup<T: Pixel, const SIZE: usize>(
        buf: &mut [T],
        pos: usize,
        stride: usize,
        sz: [u8; GS],
        lv: [u8; GS],
        lut: &[FilterLimits; 64],
    ) {
        let (valid, wide, w16, th) = masks_th::<SIZE>(sz, lv, lut);
        let mut s = [B::splat(0); 14];
        if SIZE == 4 {
            s[5..9].copy_from_slice(&load_cols4(buf, pos - 2, stride));
        } else {
            s[3..11].copy_from_slice(&load_cols8(buf, pos - 4, stride));
        }
        let m = filter::<N, SIZE>(&mut s, valid, wide, w16, &th, || {
            let lo = load_cols4(buf, pos - 7, stride);
            let hi = load_cols4(buf, pos + 3, stride);
            ([lo[0], lo[1], lo[2]], [hi[1], hi[2], hi[3]])
        });
        if m == 0 {
            return;
        }
        if m <= 2 {
            store_cols4(buf, pos - 2, stride, [s[5], s[6], s[7], s[8]]);
        } else if SIZE >= 8 && m == 3 {
            store_cols8(buf, pos - 4, stride, s[3..11].try_into().unwrap());
        } else if SIZE == 16 {
            store_cols4(buf, pos - 6, stride, [s[1], s[2], s[3], s[4]]);
            store_cols8(buf, pos - 2, stride, s[5..13].try_into().unwrap());
        }
    }

    /// One group of a horizontal edge whose first `q0` sample is `pos`.
    #[inline(never)]
    fn hgroup<T: Pixel, const SIZE: usize>(
        buf: &mut [T],
        pos: usize,
        stride: usize,
        sz: [u8; GS],
        lv: [u8; GS],
        lut: &[FilterLimits; 64],
    ) {
        let (valid, wide, w16, th) = masks_th::<SIZE>(sz, lv, lut);
        let stride = stride & 0x0fff_ffff;
        let row = |win: &[T], k: usize| B::from_array(ld::<T, N>(&win[k * stride..]));
        let mut s = [B::splat(0); 14];
        if SIZE == 4 {
            let win = &buf[pos - 2 * stride..][..3 * stride + N];
            for k in 0..4 {
                s[5 + k] = row(win, k);
            }
        } else if SIZE == 6 {
            let win = &buf[pos - 3 * stride..][..5 * stride + N];
            for k in 0..6 {
                s[4 + k] = row(win, k);
            }
        } else {
            let win = &buf[pos - 4 * stride..][..7 * stride + N];
            for k in 0..8 {
                s[3 + k] = row(win, k);
            }
        }
        let m = filter::<N, SIZE>(&mut s, valid, wide, w16, &th, || {
            let win = &buf[pos - 7 * stride..][..13 * stride + N];
            (
                [row(win, 0), row(win, 1), row(win, 2)],
                [row(win, 11), row(win, 12), row(win, 13)],
            )
        });
        if m == 0 {
            return;
        }
        if m <= 2 {
            let win = &mut buf[pos - 2 * stride..][..3 * stride + N];
            for k in 0..4 {
                st::<T, N>(&mut win[k * stride..], &s[5 + k].to_array());
            }
        } else if SIZE >= 8 && m == 3 {
            let win = &mut buf[pos - 3 * stride..][..5 * stride + N];
            for k in 0..6 {
                st::<T, N>(&mut win[k * stride..], &s[4 + k].to_array());
            }
        } else if SIZE == 16 {
            let win = &mut buf[pos - 6 * stride..][..11 * stride + N];
            for k in 0..12 {
                st::<T, N>(&mut win[k * stride..], &s[1 + k].to_array());
            }
        }
    }

    /// Filters the whole groups of a vertical (`VERT`) or horizontal
    /// edge; returns the number of segments handled.
    #[inline(always)]
    pub(super) fn edge<T: Pixel, const VERT: bool>(
        buf: &mut [T],
        stride: usize,
        x: usize,
        y: usize,
        sizes: &[u8],
        levels: &[u8],
        lut: &[FilterLimits; 64],
    ) -> usize {
        let full = sizes.len() / GS * GS;
        let mut g = 0;
        while g < full {
            let sz: [u8; GS] = sizes[g..g + GS].try_into().unwrap();
            if sz != [0; GS] {
                let lv: [u8; GS] = levels[g..g + GS].try_into().unwrap();
                let maxs = sz.iter().fold(0, |a, &b| a.max(b));
                if VERT {
                    let pos = (y + 4 * g) * stride + x;
                    match maxs {
                        4 => vgroup::<T, 4>(buf, pos, stride, sz, lv, lut),
                        6 => vgroup::<T, 6>(buf, pos, stride, sz, lv, lut),
                        8 => vgroup::<T, 8>(buf, pos, stride, sz, lv, lut),
                        _ => vgroup::<T, 16>(buf, pos, stride, sz, lv, lut),
                    }
                } else {
                    let pos = y * stride + x + 4 * g;
                    match maxs {
                        4 => hgroup::<T, 4>(buf, pos, stride, sz, lv, lut),
                        6 => hgroup::<T, 6>(buf, pos, stride, sz, lv, lut),
                        8 => hgroup::<T, 8>(buf, pos, stride, sz, lv, lut),
                        _ => hgroup::<T, 16>(buf, pos, stride, sz, lv, lut),
                    }
                }
            }
            g += GS;
        }
        full
    }

    #[cfg(test)]
    #[test]
    fn transposes() {
        let w = 40;
        let buf: Vec<u8> = (0..w * N).map(|i| (i * 7 + i / w) as u8).collect();
        let c8 = load_cols8(&buf, 3, w);
        let c4 = load_cols4(&buf, 5, w);
        for r in 0..N {
            for c in 0..8 {
                assert_eq!(c8[c][r], buf[r * w + 3 + c], "cols8 r {r} c {c}");
            }
            for c in 0..4 {
                assert_eq!(c4[c][r], buf[r * w + 5 + c], "cols4 r {r} c {c}");
            }
        }
        let mut out = vec![0u8; w * N];
        store_cols8(&mut out, 3, w, &c8);
        store_cols4(&mut out, 20, w, c4);
        for r in 0..N {
            assert_eq!(out[r * w + 3..r * w + 11], buf[r * w + 3..r * w + 11]);
            assert_eq!(out[r * w + 20..r * w + 24], buf[r * w + 5..r * w + 9]);
        }
    }
}

/// [`filter_vertical_edge`] / [`filter_horizontal_edge`] for 8-bit with
/// the `u8` lane kernels where they apply.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn edge_bd8<T: Pixel, const VERTICAL: bool>(
    buf: &mut [T],
    stride: usize,
    x: usize,
    y: usize,
    len: usize,
    filter_size: u32,
    limits: FilterLimits,
) {
    let n = len & !3;
    if !lanes8::supported(limits) || !matches!(filter_size, 4 | 6 | 8 | 16) {
        if VERTICAL {
            vertical_bd::<T, 8>(buf, stride, x, y, len, filter_size, limits);
        } else {
            horizontal_bd::<T, 8>(buf, stride, x, y, len, filter_size, limits);
        }
        return;
    }
    let start = y * stride + x;
    if VERTICAL {
        match filter_size {
            4 => lanes8::vertical::<T, 4>(buf, start, stride, n, limits),
            6 => lanes8::vertical::<T, 6>(buf, start, stride, n, limits),
            8 => lanes8::vertical::<T, 8>(buf, start, stride, n, limits),
            _ => lanes8::vertical::<T, 16>(buf, start, stride, n, limits),
        }
        if n < len {
            portable::filter_vertical_edge(buf, stride, x, y + n, len - n, filter_size, limits, 8);
        }
    } else {
        match filter_size {
            4 => lanes8::horizontal::<T, 4>(buf, start, stride, n, limits),
            6 => lanes8::horizontal::<T, 6>(buf, start, stride, n, limits),
            8 => lanes8::horizontal::<T, 8>(buf, start, stride, n, limits),
            _ => lanes8::horizontal::<T, 16>(buf, start, stride, n, limits),
        }
        if n < len {
            portable::filter_horizontal_edge(
                buf,
                stride,
                x + n,
                y,
                len - n,
                filter_size,
                limits,
                8,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Bit depth specialized entry points (selected by `Dsp::simd(bd)`).
// `BD` is 8, 10 or 12 and always equals the runtime `bd` argument; the
// pixel / intermediate ranges of `BD` may be relied upon.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn filter_vertical_edge_bd<T: Pixel, const BD: u32>(
    buf: &mut [T],
    stride: usize,
    x: usize,
    y: usize,
    len: usize,
    filter_size: u32,
    limits: FilterLimits,
    bd: u32,
) {
    debug_assert_eq!(bd, BD);
    // The decoder uses `u8` pixels iff the bit depth is 8. The tests also
    // run `u16` pixels at bit depth 8, which every kernel supports.
    debug_assert!(cfg!(test) || (T::MAX_BITDEPTH == 8) == (BD == 8));
    if BD == 8 {
        return edge_bd8::<T, true>(buf, stride, x, y, len, filter_size, limits);
    }
    vertical_bd::<T, BD>(buf, stride, x, y, len, filter_size, limits);
}

#[allow(clippy::too_many_arguments)]
pub fn filter_horizontal_edge_bd<T: Pixel, const BD: u32>(
    buf: &mut [T],
    stride: usize,
    x: usize,
    y: usize,
    len: usize,
    filter_size: u32,
    limits: FilterLimits,
    bd: u32,
) {
    debug_assert_eq!(bd, BD);
    // The decoder uses `u8` pixels iff the bit depth is 8. The tests also
    // run `u16` pixels at bit depth 8, which every kernel supports.
    debug_assert!(cfg!(test) || (T::MAX_BITDEPTH == 8) == (BD == 8));
    if BD == 8 {
        return edge_bd8::<T, false>(buf, stride, x, y, len, filter_size, limits);
    }
    horizontal_bd::<T, BD>(buf, stride, x, y, len, filter_size, limits);
}

/// Tail of [`filter_v_edge_bd`] (and all of it when `BD != 8`, so it is
/// not `#[cold]`: it is the whole edge loop of high bit depth streams).
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn filter_v_edge_tail_bd<T: Pixel, const BD: u32>(
    buf: &mut [T],
    stride: usize,
    x: usize,
    y: usize,
    sizes: &[u8],
    levels: &[u8],
    lut: &[FilterLimits; 64],
    mut i: usize,
) {
    while i < sizes.len() {
        let s = sizes[i];
        if s == 0 {
            i += 1;
            continue;
        }
        let l = levels[i];
        let mut j = i + 1;
        while j < sizes.len() && sizes[j] == s && levels[j] == l {
            j += 1;
        }
        let start = (y + 4 * i) * stride + x;
        let len = 4 * (j - i);
        let limits = lut[(l & 63) as usize];
        if BD == 8 {
            match s {
                4 => lanes8::vertical::<T, 4>(buf, start, stride, len, limits),
                6 => lanes8::vertical::<T, 6>(buf, start, stride, len, limits),
                8 => lanes8::vertical::<T, 8>(buf, start, stride, len, limits),
                _ => lanes8::vertical::<T, 16>(buf, start, stride, len, limits),
            }
        } else {
            let th = Th::new::<BD>(limits);
            match s {
                4 => vertical::<T, BD, 4>(buf, start, stride, len, &th),
                6 => vertical::<T, BD, 6>(buf, start, stride, len, &th),
                8 => vertical::<T, BD, 8>(buf, start, stride, len, &th),
                _ => vertical::<T, BD, 16>(buf, start, stride, len, &th),
            }
        }
        i = j;
    }
}

/// Whole edge with per segment sizes and levels, see
/// [`crate::decoder::av1::dsp::loopfilter::filter_v_edge`]. Runs of segments with
/// equal parameters are filtered with one call of the run kernel (a
/// kernel with per-lane parameters was measured slower on the device:
/// mixed groups pay for the widest filter in all lanes).
#[allow(clippy::too_many_arguments)]
pub fn filter_v_edge_bd<T: Pixel, const BD: u32>(
    buf: &mut [T],
    stride: usize,
    x: usize,
    y: usize,
    sizes: &[u8],
    levels: &[u8],
    lut: &[FilterLimits; 64],
    bd: u32,
) {
    debug_assert_eq!(bd, BD);
    // The decoder uses `u8` pixels iff the bit depth is 8. The tests also
    // run `u16` pixels at bit depth 8, which every kernel supports.
    debug_assert!(cfg!(test) || (T::MAX_BITDEPTH == 8) == (BD == 8));
    assert_eq!(sizes.len(), levels.len());
    // 8-bit: whole groups of segments with the group kernels, the rest
    // (fewer than a group, at the end of the frame) run by run.
    let i = if BD == 8 {
        grp::edge::<T, true>(buf, stride, x, y, sizes, levels, lut)
    } else {
        0
    };
    if i < sizes.len() {
        filter_v_edge_tail_bd::<T, BD>(buf, stride, x, y, sizes, levels, lut, i);
    }
}

/// Tail of [`filter_h_edge_bd`] (and all of it when `BD != 8`, so it is
/// not `#[cold]`: it is the whole edge loop of high bit depth streams).
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn filter_h_edge_tail_bd<T: Pixel, const BD: u32>(
    buf: &mut [T],
    stride: usize,
    x: usize,
    y: usize,
    sizes: &[u8],
    levels: &[u8],
    lut: &[FilterLimits; 64],
    mut i: usize,
) {
    while i < sizes.len() {
        let s = sizes[i];
        if s == 0 {
            i += 1;
            continue;
        }
        let l = levels[i];
        let mut j = i + 1;
        while j < sizes.len() && sizes[j] == s && levels[j] == l {
            j += 1;
        }
        let start = y * stride + x + 4 * i;
        let len = 4 * (j - i);
        let limits = lut[(l & 63) as usize];
        if BD == 8 {
            match s {
                4 => lanes8::horizontal::<T, 4>(buf, start, stride, len, limits),
                6 => lanes8::horizontal::<T, 6>(buf, start, stride, len, limits),
                8 => lanes8::horizontal::<T, 8>(buf, start, stride, len, limits),
                _ => lanes8::horizontal::<T, 16>(buf, start, stride, len, limits),
            }
        } else {
            let th = Th::new::<BD>(limits);
            match s {
                4 => horizontal::<T, BD, 4>(buf, start, stride, len, &th),
                6 => horizontal::<T, BD, 6>(buf, start, stride, len, &th),
                8 => horizontal::<T, BD, 8>(buf, start, stride, len, &th),
                _ => horizontal::<T, BD, 16>(buf, start, stride, len, &th),
            }
        }
        i = j;
    }
}

/// Whole edge with per segment sizes and levels, see
/// [`crate::decoder::av1::dsp::loopfilter::filter_h_edge`].
#[allow(clippy::too_many_arguments)]
pub fn filter_h_edge_bd<T: Pixel, const BD: u32>(
    buf: &mut [T],
    stride: usize,
    x: usize,
    y: usize,
    sizes: &[u8],
    levels: &[u8],
    lut: &[FilterLimits; 64],
    bd: u32,
) {
    debug_assert_eq!(bd, BD);
    // The decoder uses `u8` pixels iff the bit depth is 8. The tests also
    // run `u16` pixels at bit depth 8, which every kernel supports.
    debug_assert!(cfg!(test) || (T::MAX_BITDEPTH == 8) == (BD == 8));
    assert_eq!(sizes.len(), levels.len());
    let i = if BD == 8 {
        grp::edge::<T, false>(buf, stride, x, y, sizes, levels, lut)
    } else {
        0
    };
    if i < sizes.len() {
        filter_h_edge_tail_bd::<T, BD>(buf, stride, x, y, sizes, levels, lut, i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// A line profile across an edge that hits all filter paths: flat
    /// regions, small noise and steps of various heights.
    fn gen_line(rng: &mut Rng, bd: u32, out: &mut [i32]) {
        let maxv = (1 << bd) - 1;
        let sc = 1 << (bd - 8);
        let base = rng.range(0, maxv);
        let step = match rng.range(0, 3) {
            0 => 0,
            1 => rng.range(-2, 2) * sc,
            2 => rng.range(-20, 20) * sc,
            _ => rng.range(-maxv, maxv),
        };
        let noise = match rng.range(0, 4) {
            0 => 0,
            1 => sc,
            2 => 3 * sc,
            3 => 40 * sc,
            _ => maxv,
        };
        let half = out.len() / 2;
        for (k, v) in out.iter_mut().enumerate() {
            let s = if k >= half { step } else { 0 };
            *v = (base + s + rng.range(-noise, noise)).clamp(0, maxv);
        }
    }

    fn run<T: Pixel>(bd: u32, seed: u64, iters: usize) {
        let mut rng = Rng(seed);
        const E: usize = 8; // edge position (row or column)
                            // changed[size][d]: samples changed at distance d from the edge.
        let mut changed = [[0usize; E + 1]; 4];
        for iter in 0..iters {
            let si = rng.range(0, 3) as usize;
            let size = [4u32, 6, 8, 16][si];
            let vertical = rng.range(0, 1) == 0;
            // The decoder passes multiples of 4; others take the fallback.
            let len =
                if rng.range(0, 9) == 0 { rng.range(1, 40) } else { 4 * rng.range(1, 10) } as usize;
            let along0 = rng.range(0, 5) as usize;
            let w = if vertical {
                2 * E + rng.range(0, 5) as usize
            } else {
                along0 + len + rng.range(0, 5) as usize
            };
            let h = if vertical {
                along0 + len + rng.range(0, 3) as usize
            } else {
                2 * E + rng.range(0, 3) as usize
            };
            let limits = portable::filter_limits(rng.range(1, 63), rng.range(0, 7));
            // Lines perpendicular to the edge, a new profile every few
            // lines (so that neighbouring lanes take different paths).
            let mut data = vec![0i32; w * h];
            let mut line = [0i32; 2 * E];
            for a in 0..if vertical { h } else { w } {
                if a == 0 || rng.range(0, 2) == 0 {
                    gen_line(&mut rng, bd, &mut line);
                }
                for (k, &v) in line.iter().enumerate() {
                    if vertical {
                        data[a * w + k] = v;
                    } else {
                        data[k * w + a] = v;
                    }
                }
            }
            let mut want: Vec<T> = data.iter().map(|&v| T::from_i32(v)).collect();
            let mut got = want.clone();
            let orig = want.clone();
            let mut got8 = want.clone();
            if vertical {
                portable::filter_vertical_edge(&mut want, w, E, along0, len, size, limits, bd);
                filter_vertical_edge(&mut got, w, E, along0, len, size, limits, bd);
                if bd == 8 {
                    filter_vertical_edge_bd::<T, 8>(&mut got8, w, E, along0, len, size, limits, bd);
                }
            } else {
                portable::filter_horizontal_edge(&mut want, w, along0, E, len, size, limits, bd);
                filter_horizontal_edge(&mut got, w, along0, E, len, size, limits, bd);
                if bd == 8 {
                    filter_horizontal_edge_bd::<T, 8>(
                        &mut got8, w, along0, E, len, size, limits, bd,
                    );
                }
            }
            assert_eq!(
                got, want,
                "iter {iter} bd {bd} size {size} vertical {vertical} len {len}"
            );
            if bd == 8 {
                assert_eq!(
                    got8, want,
                    "u8 lanes: iter {iter} size {size} vertical {vertical} len {len}"
                );
            }
            for (i, (a, b)) in want.iter().zip(&orig).enumerate() {
                if a != b {
                    let k = if vertical { i % w } else { i / w };
                    changed[si][if k >= E { k - E + 1 } else { E - k }] += 1;
                }
            }
        }
        // All filter paths must have been taken: the narrow filter
        // (distance 1, and 2 without hev), the 6 tap (distance 2), 8 tap
        // (3) and 16 tap (6) wide filters.
        for (si, d) in [(0, 2), (1, 2), (2, 3), (3, 3), (3, 6)] {
            assert!(changed[si][d] > iters / 100, "bd {bd}: {changed:?}");
        }
        for (c, m) in changed.iter().zip([2, 2, 3, 6]) {
            assert!(c[m + 1..].iter().all(|&n| n == 0), "bd {bd}: {changed:?}");
        }
    }

    /// Whole edges with random per segment sizes (of one plane class,
    /// with gaps) and levels against the portable kernel.
    fn edge_run<T: Pixel, const BD: u32>(seed: u64, iters: usize) {
        let mut rng = Rng(seed);
        const E: usize = 8;
        for iter in 0..iters {
            let chroma = rng.range(0, 3) == 0;
            let vertical = rng.range(0, 1) == 0;
            let nseg = rng.range(1, 20) as usize;
            let mut sizes: Vec<u8> = (0..nseg)
                .map(|_| {
                    if chroma {
                        [0u8, 4, 6][rng.range(0, 2) as usize]
                    } else {
                        [0u8, 4, 8, 16][rng.range(0, 3) as usize]
                    }
                })
                .collect();
            if rng.range(0, 2) == 0 {
                // Long uniform runs.
                let s = sizes[0];
                sizes.iter_mut().for_each(|v| *v = s);
            }
            let sharp = rng.range(0, 7);
            let mut lut = [FilterLimits::default(); 64];
            for (l, v) in lut.iter_mut().enumerate().skip(1) {
                *v = portable::filter_limits(l as i32, sharp);
            }
            let lvl0 = rng.range(1, 63) as u8;
            let levels: Vec<u8> = (0..nseg)
                .map(|_| if rng.range(0, 1) == 0 { lvl0 } else { rng.range(1, 63) as u8 })
                .collect();
            let len = 4 * nseg;
            let along0 = rng.range(0, 5) as usize;
            let w = if vertical {
                2 * E + rng.range(0, 5) as usize
            } else {
                along0 + len + rng.range(0, 5) as usize
            };
            let h = if vertical {
                along0 + len + rng.range(0, 3) as usize
            } else {
                2 * E + rng.range(0, 3) as usize
            };
            let mut data = vec![0i32; w * h];
            let mut line = [0i32; 2 * E];
            for a in 0..if vertical { h } else { w } {
                if a == 0 || rng.range(0, 2) == 0 {
                    gen_line(&mut rng, BD, &mut line);
                }
                for (k, &v) in line.iter().enumerate() {
                    if vertical {
                        data[a * w + k] = v;
                    } else {
                        data[k * w + a] = v;
                    }
                }
            }
            let mut want: Vec<T> = data.iter().map(|&v| T::from_i32(v)).collect();
            let mut got = want.clone();
            if vertical {
                portable::filter_v_edge(&mut want, w, E, along0, &sizes, &levels, &lut, BD);
                filter_v_edge_bd::<T, BD>(&mut got, w, E, along0, &sizes, &levels, &lut, BD);
            } else {
                portable::filter_h_edge(&mut want, w, along0, E, &sizes, &levels, &lut, BD);
                filter_h_edge_bd::<T, BD>(&mut got, w, along0, E, &sizes, &levels, &lut, BD);
            }
            assert_eq!(
                got, want,
                "iter {iter} vertical {vertical} sizes {sizes:?} levels {levels:?}"
            );
        }
    }

    #[test]
    fn edge_matches_portable() {
        edge_run::<u8, 8>(0x5151_7272_9393_a4a4, 40_000);
        edge_run::<u16, 8>(0x1111_2222_3333_4444, 10_000);
        edge_run::<u16, 10>(0x7777_8888_9999_aaaa, 10_000);
        edge_run::<u16, 12>(0xd1b5_4a32_d192_ed03, 10_000);
    }

    #[test]
    fn matches_portable_8bit() {
        run::<u8>(8, 0x1234_5678_9abc_def1, 100_000);
        run::<u16>(8, 0x0bad_cafe_dead_beef, 30_000);
    }

    #[test]
    fn matches_portable_10bit() {
        run::<u16>(10, 0x9e37_79b9_7f4a_7c15, 100_000);
    }

    #[test]
    fn matches_portable_12bit() {
        run::<u16>(12, 0xd1b5_4a32_d192_ed03, 100_000);
    }

    /// Wall time of the whole edge kernels (`filter_{v,h}_edge_bd`, the
    /// ones the decoder calls) on a frame with a mix of segment sizes and
    /// content that takes all filter paths: `cargo test --release
    /// --lib -- --ignored --nocapture bench_edges`.
    #[test]
    #[ignore]
    fn bench_edges() {
        fn time<T: Pixel, const BD: u32>() {
            let (w, h) = (1024usize, 512usize);
            let stride = w + 64;
            let mut rng = Rng(11);
            let mut src = vec![T::default(); stride * h];
            // 8x8 blocks with a DC offset (random steps of the sizes in
            // `gen_line` between neighbours) and per block noise, so that
            // vertical and horizontal edges see the same statistics.
            let maxv = (1i32 << BD) - 1;
            let sc = 1i32 << (BD - 8);
            let (bw, bh) = (w / 8, h / 8);
            let mut dc = vec![0i32; bw * bh];
            let mut noise = vec![0i32; bw * bh];
            for by in 0..bh {
                for bx in 0..bw {
                    let mut step = || match rng.range(0, 3) {
                        0 => 0,
                        1 => rng.range(-2, 2) * sc,
                        2 => rng.range(-20, 20) * sc,
                        _ => rng.range(-maxv / 4, maxv / 4),
                    };
                    let left = if bx > 0 { dc[by * bw + bx - 1] + step() } else { 128 * sc };
                    let up = if by > 0 { dc[(by - 1) * bw + bx] + step() } else { left };
                    dc[by * bw + bx] = ((left + up) / 2).clamp(0, maxv);
                    noise[by * bw + bx] = [0, sc, 3 * sc, 40 * sc][rng.range(0, 3) as usize];
                }
            }
            for y in 0..h {
                for x in 0..w {
                    let b = (y / 8) * bw + x / 8;
                    let n = noise[b];
                    src[y * stride + x] = T::from_i32((dc[b] + rng.range(-n, n)).clamp(0, maxv));
                }
            }
            let lut: [FilterLimits; 64] =
                std::array::from_fn(|l| portable::filter_limits(l.max(1) as i32, 0));
            // Per segment sizes: luma-like (mostly 4 / 8 / 16) and
            // chroma-like (4 / 6); a constant level, as without delta LF.
            let mut make = |n: usize, chroma: bool| -> (Vec<u8>, Vec<u8>) {
                let sizes = (0..n)
                    .map(|_| match (chroma, rng.range(0, 9)) {
                        (_, 0) => 0,
                        (true, 1..=4) => 4,
                        (true, _) => 6,
                        (false, 1..=3) => 4,
                        (false, 4..=6) => 8,
                        (false, _) => 16,
                    })
                    .collect();
                (sizes, vec![32u8; n])
            };
            let segs_v = (h - 16) / 4;
            let segs_h = (w - 32) / 4;
            for chroma in [false, true] {
                let v_params: Vec<_> = (0..w / 8).map(|_| make(segs_v, chroma)).collect();
                let h_params: Vec<_> = (0..h / 8).map(|_| make(segs_h, chroma)).collect();
                let reps = 30;
                let mut best_v = f64::MAX;
                let mut best_h = f64::MAX;
                for _ in 0..reps {
                    let mut buf = src.clone();
                    let t = std::time::Instant::now();
                    for (i, (s, l)) in v_params.iter().enumerate().skip(2).take(w / 8 - 4) {
                        filter_v_edge_bd::<T, BD>(&mut buf, stride, 8 * i, 8, s, l, &lut, BD);
                    }
                    best_v = best_v.min(t.elapsed().as_secs_f64());
                    let t = std::time::Instant::now();
                    for (i, (s, l)) in h_params.iter().enumerate().skip(2).take(h / 8 - 4) {
                        filter_h_edge_bd::<T, BD>(&mut buf, stride, 16, 8 * i, s, l, &lut, BD);
                    }
                    best_h = best_h.min(t.elapsed().as_secs_f64());
                    std::hint::black_box(&buf);
                }
                println!(
                    "bd {BD} {}: vertical {:.1} us, horizontal {:.1} us",
                    if chroma { "chroma" } else { "luma  " },
                    best_v * 1e6,
                    best_h * 1e6
                );
            }
        }
        time::<u8, 8>();
        time::<u16, 10>();
    }

    /// Wall time of the portable and SIMD kernels on a noisy frame
    /// (`cargo test --release -- --ignored
    /// --nocapture bench`).
    #[test]
    #[ignore]
    fn bench() {
        fn time<T: Pixel>(bd: u32) {
            let (w, h) = (1024usize, 256usize);
            let mut rng = Rng(7);
            let sc = 1 << (bd - 8);
            let src: Vec<T> = (0..w * h)
                .map(|i| T::from_i32(((i % 64) as i32 + rng.range(0, 3)) * sc))
                .collect();
            let limits = portable::filter_limits(40, 0);
            type F<T> = fn(&mut [T], usize, usize, usize, usize, u32, FilterLimits, u32);
            let mut kernels: Vec<(&str, F<T>, F<T>)> = vec![
                (
                    "portable",
                    portable::filter_vertical_edge::<T>,
                    portable::filter_horizontal_edge::<T>,
                ),
                (
                    "simd",
                    filter_vertical_edge::<T>,
                    filter_horizontal_edge::<T>,
                ),
            ];
            if bd == 8 {
                kernels.push((
                    "u8 lanes",
                    filter_vertical_edge_bd::<T, 8>,
                    filter_horizontal_edge_bd::<T, 8>,
                ));
            }
            for (name, v, hz) in kernels {
                for size in [4u32, 6, 8, 16] {
                    let mut buf = src.clone();
                    let t = std::time::Instant::now();
                    for _ in 0..20 {
                        for x in (16..w - 16).step_by(16) {
                            v(&mut buf, w, x, 8, h - 16, size, limits, bd);
                        }
                    }
                    let tv = t.elapsed();
                    let t = std::time::Instant::now();
                    for _ in 0..20 {
                        for y in (16..h - 16).step_by(16) {
                            hz(&mut buf, w, 8, y, w - 16, size, limits, bd);
                        }
                    }
                    println!(
                        "bd {bd} {name:8} size {size:2}: vertical {tv:?} horizontal {:?}",
                        t.elapsed()
                    );
                }
            }
        }
        time::<u8>(8);
        time::<u16>(10);
    }

    #[test]
    fn transpose_8x8() {
        let v: [V; 8] =
            std::array::from_fn(|r| V::from_array(std::array::from_fn(|c| (r * 8 + c) as i16)));
        let t = transpose(v);
        for (c, col) in t.iter().enumerate() {
            for r in 0..8 {
                assert_eq!(col[r], (r * 8 + c) as i16);
            }
        }
        let v4: [V; 4] =
            std::array::from_fn(|r| V::from_array(std::array::from_fn(|c| (r * 8 + c) as i16)));
        let t4 = transpose4(v4);
        for (c, col) in t4.iter().enumerate() {
            for r in 0..4 {
                assert_eq!(col[r], (r * 8 + c) as i16);
                assert_eq!(col[4 + r], (r * 8 + 4 + c) as i16);
            }
        }
    }

    /// Extreme values: flat lines at 0 / max and maximal steps.
    #[test]
    fn extremes() {
        for bd in [8u32, 10, 12] {
            let maxv = (1 << bd) - 1;
            for size in [4u32, 6, 8, 16] {
                for lvl in [1, 63] {
                    for pattern in 0..4 {
                        let limits = portable::filter_limits(lvl, 0);
                        let w = 16;
                        let mut data = vec![0u16; w * 16];
                        for (i, v) in data.iter_mut().enumerate() {
                            let (r, c) = (i / w, i % w);
                            *v = match pattern {
                                0 => maxv,
                                1 => {
                                    if c >= 8 {
                                        maxv
                                    } else {
                                        0
                                    }
                                }
                                2 => {
                                    if r >= 8 {
                                        0
                                    } else {
                                        maxv
                                    }
                                }
                                _ => {
                                    if (r + c) % 2 == 0 {
                                        maxv
                                    } else {
                                        0
                                    }
                                }
                            } as u16;
                        }
                        let (mut a, mut b) = (data.clone(), data.clone());
                        portable::filter_vertical_edge(&mut a, w, 8, 0, 16, size, limits, bd);
                        filter_vertical_edge(&mut b, w, 8, 0, 16, size, limits, bd);
                        assert_eq!(a, b);
                        if bd == 8 {
                            let mut c = data.clone();
                            filter_vertical_edge_bd::<u16, 8>(
                                &mut c, w, 8, 0, 16, size, limits, bd,
                            );
                            assert_eq!(a, c);
                            let mut c: Vec<u8> = data.iter().map(|&v| v as u8).collect();
                            filter_vertical_edge_bd::<u8, 8>(&mut c, w, 8, 0, 16, size, limits, bd);
                            assert!(a.iter().zip(&c).all(|(&a, &c)| a == c as u16));
                        }
                        let (mut a, mut b) = (data.clone(), data.clone());
                        portable::filter_horizontal_edge(&mut a, w, 0, 8, 16, size, limits, bd);
                        filter_horizontal_edge(&mut b, w, 0, 8, 16, size, limits, bd);
                        assert_eq!(a, b);
                        if bd == 8 {
                            let mut c: Vec<u8> = data.iter().map(|&v| v as u8).collect();
                            filter_horizontal_edge_bd::<u8, 8>(
                                &mut c, w, 0, 8, 16, size, limits, bd,
                            );
                            assert!(a.iter().zip(&c).all(|(&a, &c)| a == c as u16));
                        }
                    }
                }
            }
        }
    }
}
