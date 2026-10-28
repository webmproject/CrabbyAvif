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

//! Deblocking loop filter sample filtering (spec 7.14.4 and 7.14.6).
//!
//! OWNER: lf/cdef worker. Contract below must be kept.
//!
//! The caller performs edge selection, filter size and level derivation
//! (spec 7.14.1 - 7.14.3 and 7.14.5). This module filters samples.
//!
//! # Filter sizes
//!
//! The spec's sample filtering process (7.14.6) takes `plane` and
//! `filterSize` as inputs, but `plane` only matters when the spec
//! `filterSize` is 8 for a chroma plane: then `filterLen` is 6 and the 8
//! tap wide filter is replaced by the 6 tap chroma variant (`n = 2`,
//! `n2 = 1`). This module folds `plane` into the filter size, i.e. the
//! kernel `filter_size` is:
//!
//! | spec plane | spec filterSize | kernel `filter_size` |
//! |------------|-----------------|----------------------|
//! | any        | 4               | 4                    |
//! | > 0        | 8               | 6                    |
//! | 0          | 8               | 8                    |
//! | 0          | 16              | 16                   |
//!
//! [`kernel_filter_size`] performs this mapping.

use crate::decoder::av1::pixel::Pixel;

/// Filter thresholds for one filter level (spec 7.14.4 outputs, 8-bit
/// scale; the bit depth shift is applied inside the filter functions).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FilterLimits {
    pub limit: i32,
    pub blimit: i32,
    pub thresh: i32,
}

/// Spec 7.14.4 adaptive filter strength process: computes the limits for
/// filter level `lvl` (1..=63) and `loop_filter_sharpness`.
pub const fn filter_limits(lvl: i32, sharpness: i32) -> FilterLimits {
    let shift = if sharpness > 4 {
        2
    } else if sharpness > 0 {
        1
    } else {
        0
    };
    let shifted = lvl >> shift;
    let limit = if sharpness > 0 {
        let hi = 9 - sharpness;
        if shifted < 1 {
            1
        } else if shifted > hi {
            hi
        } else {
            shifted
        }
    } else if shifted > 1 {
        shifted
    } else {
        1
    };
    FilterLimits {
        limit,
        blimit: 2 * (lvl + 2) + limit,
        thresh: lvl >> 4,
    }
}

/// Maps the spec's (`plane`, `filterSize`) pair (outputs of 7.14.3) to the
/// kernel `filter_size` accepted by [`filter_vertical_edge`] and
/// [`filter_horizontal_edge`] (see the module documentation).
#[inline]
pub fn kernel_filter_size(plane: usize, spec_filter_size: u32) -> u32 {
    if plane > 0 && spec_filter_size >= 8 {
        6
    } else {
        spec_filter_size
    }
}

/// Filters across a *vertical* edge (samples are modified horizontally).
/// The edge lies between columns `x - 1` and `x` of `buf` (row stride
/// `stride`), and `len` consecutive rows starting at row `y` are filtered
/// (spec 7.14.6 invoked once per row with dx = 1, dy = 0).
///
/// `filter_size` is the spec filterSize (4, 6, 8 or 16). The function
/// must only touch samples within `filter_size / 2` (and read within
/// `filter_size / 2 + 1`, i.e. up to 8 for size 16) columns of the edge.
/// `plane` is only needed where the spec distinguishes luma and chroma.
///
/// Note: 6 denotes the spec's chroma filterSize 8 (see
/// [`kernel_filter_size`]); with this convention no `plane` argument is
/// needed.
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
    filter_edge(buf, y * stride + x, stride, 1, len, filter_size, limits, bd);
}

/// Filters across a *horizontal* edge between rows `y - 1` and `y`, for
/// `len` consecutive columns starting at column `x` (dx = 0, dy = 1).
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
    filter_edge(buf, y * stride + x, 1, stride, len, filter_size, limits, bd);
}

/// Filters one whole vertical edge made of `sizes.len()` segments of 4
/// rows: the edge lies between columns `x - 1` and `x`, segment `i` is
/// rows `y + 4i..y + 4i + 4`. `sizes[i]` is the kernel filter size of
/// segment `i` (0: not filtered, else 4, 6, 8 or 16 as for
/// [`filter_vertical_edge`]) and `levels[i]` its filter level (1..=63,
/// an index into `lut`, which holds `filter_limits(level, sharpness)`).
/// `sizes` and `levels` have the same length.
#[allow(clippy::too_many_arguments)]
pub fn filter_v_edge<T: Pixel>(
    buf: &mut [T],
    stride: usize,
    x: usize,
    y: usize,
    sizes: &[u8],
    levels: &[u8],
    lut: &[FilterLimits; 64],
    bd: u32,
) {
    for_runs(sizes, levels, |i, n, size, level| {
        filter_vertical_edge(buf, stride, x, y + 4 * i, 4 * n, size, lut[level], bd)
    });
}

/// Filters one whole horizontal edge made of `sizes.len()` segments of 4
/// columns: the edge lies between rows `y - 1` and `y`, segment `i` is
/// columns `x + 4i..x + 4i + 4` (see [`filter_v_edge`]).
#[allow(clippy::too_many_arguments)]
pub fn filter_h_edge<T: Pixel>(
    buf: &mut [T],
    stride: usize,
    x: usize,
    y: usize,
    sizes: &[u8],
    levels: &[u8],
    lut: &[FilterLimits; 64],
    bd: u32,
) {
    for_runs(sizes, levels, |i, n, size, level| {
        filter_horizontal_edge(buf, stride, x + 4 * i, y, 4 * n, size, lut[level], bd)
    });
}

/// Calls `f(first, count, size, level)` for every maximal run of
/// filtered segments with equal size and level.
#[inline]
pub(crate) fn for_runs(sizes: &[u8], levels: &[u8], mut f: impl FnMut(usize, usize, u32, usize)) {
    assert_eq!(sizes.len(), levels.len());
    let mut i = 0;
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
        f(i, j - i, s as u32, l as usize);
        i = j;
    }
}

/// Bit depth scaled thresholds (spec 7.14.6.2 `*Bd` variables) and pixel
/// constants of the narrow filter.
#[derive(Clone, Copy)]
struct Thresholds {
    limit: i32,
    blimit: i32,
    thresh: i32,
    /// `thresholdBd` of the flatness tests.
    flat: i32,
    /// `0x80 << (bd - 8)`: offset between pixels and signed values.
    off: i32,
    /// Signed clamping range of the narrow filter: `[-off, off - 1]`.
    lo: i32,
    hi: i32,
}

/// Filters `len` lines across one edge. `start` is the index of the first
/// `q0` sample, `along` the index step between successive lines and
/// `across` the index step perpendicular to the edge (towards `q1`).
#[allow(clippy::too_many_arguments)]
#[inline]
fn filter_edge<T: Pixel>(
    buf: &mut [T],
    start: usize,
    along: usize,
    across: usize,
    len: usize,
    filter_size: u32,
    limits: FilterLimits,
    bd: u32,
) {
    let shift = bd - 8;
    let off = 0x80 << shift;
    let th = Thresholds {
        limit: limits.limit << shift,
        blimit: limits.blimit << shift,
        thresh: limits.thresh << shift,
        flat: 1 << shift,
        off,
        lo: -off,
        hi: off - 1,
    };
    match filter_size {
        4 => filter_lines::<T, 4>(buf, start, along, across, len, &th),
        6 => filter_lines::<T, 6>(buf, start, along, across, len, &th),
        8 => filter_lines::<T, 8>(buf, start, along, across, len, &th),
        16 => filter_lines::<T, 16>(buf, start, along, across, len, &th),
        _ => panic!("invalid loop filter size {filter_size}"),
    }
}

/// Number of samples read on each side of the edge for a kernel size.
const fn reach(size: usize) -> usize {
    match size {
        4 => 2,
        6 => 3,
        8 => 4,
        _ => 7,
    }
}

/// Number of samples modified on each side of the edge for a kernel size.
const fn modified(size: usize) -> usize {
    match size {
        4 | 6 => 2,
        8 => 3,
        _ => 6,
    }
}

/// Lines processed together when the samples of a line are strided
/// (horizontal edges): rows are then read / written as contiguous runs.
const GROUP: usize = 4;

#[inline(always)]
fn filter_lines<T: Pixel, const SIZE: usize>(
    buf: &mut [T],
    start: usize,
    along: usize,
    across: usize,
    len: usize,
    th: &Thresholds,
) {
    let r = reach(SIZE);
    let m = modified(SIZE);
    if across == 1 {
        // Vertical edge: each line is a contiguous run of 2 * r samples.
        for i in 0..len {
            let pos = start + i * along;
            let seg = &mut buf[pos - r..pos + r];
            let mut s = [0i32; 14];
            for (d, v) in s[7 - r..7 + r].iter_mut().zip(seg.iter()) {
                *d = v.to_i32();
            }
            if filter_line::<SIZE>(&mut s, th) {
                for (d, &v) in seg[r - m..r + m].iter_mut().zip(&s[7 - m..7 + m]) {
                    *d = T::from_i32(v);
                }
            }
        }
    } else {
        // Horizontal edge (along == 1 for the public entry points): load
        // GROUP columns at once, row by row.
        let mut j0 = 0;
        while j0 < len {
            let n = GROUP.min(len - j0);
            let mut s = [[0i32; 14]; GROUP];
            if n == GROUP && along == 1 {
                // Common case: GROUP adjacent columns, one contiguous run
                // per row.
                let base = start + j0 - r * across;
                for k in 0..2 * r {
                    let row: &[T; GROUP] = buf[base + k * across..][..GROUP].try_into().unwrap();
                    for (line, v) in s.iter_mut().zip(row) {
                        line[7 - r + k] = v.to_i32();
                    }
                }
                let mut any = false;
                for line in &mut s {
                    any |= filter_line::<SIZE>(line, th);
                }
                if any {
                    for k in r - m..r + m {
                        let row: &mut [T; GROUP] =
                            (&mut buf[base + k * across..][..GROUP]).try_into().unwrap();
                        for (v, line) in row.iter_mut().zip(&s) {
                            *v = T::from_i32(line[7 - r + k]);
                        }
                    }
                }
                j0 += n;
                continue;
            }
            for k in 0..2 * r {
                let row = start + j0 * along + k * across - r * across;
                for (c, line) in s[..n].iter_mut().enumerate() {
                    line[7 - r + k] = buf[row + c * along].to_i32();
                }
            }
            let mut any = false;
            for line in &mut s[..n] {
                any |= filter_line::<SIZE>(line, th);
            }
            if any {
                for k in r - m..r + m {
                    let row = start + j0 * along + k * across - r * across;
                    for (c, line) in s[..n].iter().enumerate() {
                        buf[row + c * along] = T::from_i32(line[7 - r + k]);
                    }
                }
            }
            j0 += n;
        }
    }
}

/// Spec 7.14.6.1 - 7.14.6.4 for one line held in `s`, where `s[7 + k]`
/// holds the sample at signed offset `k` (`-7..=6`) from `q0`:
/// `q_i = s[7 + i]`, `p_i = s[6 - i]`. Filters `s` in place and returns
/// whether it was modified (i.e. the filter mask passed).
#[inline(always)]
fn filter_line<const SIZE: usize>(s: &mut [i32; 14], th: &Thresholds) -> bool {
    let p = |i: usize| s[6 - i];
    let q = |i: usize| s[7 + i];

    // 7.14.6.2 filter mask process. Here filterLen == SIZE.
    let (p0, p1, q0, q1) = (p(0), p(1), q(0), q(1));
    let ad_p1p0 = (p1 - p0).abs();
    let ad_q1q0 = (q1 - q0).abs();
    let mut mask = (ad_p1p0 > th.limit)
        | (ad_q1q0 > th.limit)
        | ((p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 > th.blimit);
    if SIZE >= 6 {
        mask |= ((p(2) - p1).abs() > th.limit) | ((q(2) - q1).abs() > th.limit);
    }
    if SIZE >= 8 {
        mask |= ((p(3) - p(2)).abs() > th.limit) | ((q(3) - q(2)).abs() > th.limit);
    }
    if mask {
        return false;
    }

    let flat = SIZE >= 6 && {
        let mut m = (ad_p1p0 > th.flat)
            | (ad_q1q0 > th.flat)
            | ((p(2) - p0).abs() > th.flat)
            | ((q(2) - q0).abs() > th.flat);
        if SIZE >= 8 {
            m |= ((p(3) - p0).abs() > th.flat) | ((q(3) - q0).abs() > th.flat);
        }
        !m
    };
    if !flat {
        let hev = (ad_p1p0 > th.thresh) | (ad_q1q0 > th.thresh);
        narrow_filter(s, hev, th);
        return true;
    }

    let flat2 = SIZE >= 16 && {
        let mut m = false;
        for i in 4..7 {
            m |= ((p(i) - p0).abs() > th.flat) | ((q(i) - q0).abs() > th.flat);
        }
        !m
    };
    match (SIZE, flat2) {
        (6, _) => wide_filter6(s),
        (_, false) => wide_filter8(s),
        (_, true) => wide_filter16(s),
    }
    true
}

/// Spec 7.14.6.3 narrow filter process on `s` (see [`filter_line`]).
#[inline(always)]
fn narrow_filter(s: &mut [i32; 14], hev: bool, th: &Thresholds) {
    let c = |v: i32| v.max(th.lo).min(th.hi);
    let off = th.off;
    let (ps1, ps0, qs0, qs1) = (s[5] - off, s[6] - off, s[7] - off, s[8] - off);
    let base = if hev { c(ps1 - qs1) } else { 0 };
    let filter = c(base + 3 * (qs0 - ps0));
    let filter1 = c(filter + 4) >> 3;
    let filter2 = c(filter + 3) >> 3;
    s[7] = c(qs0 - filter1) + off;
    s[6] = c(ps0 + filter2) + off;
    if !hev {
        let f = (filter1 + 1) >> 1;
        s[8] = c(qs1 - f) + off;
        s[5] = c(ps1 + f) + off;
    }
}

/// Spec 7.14.6.4 wide filter process, written out per `(n, n2,
/// log2Size)`, in place on `s` (see [`filter_line`]). Each output is
/// `Round2(sum_j tap_j * s[clamp(i + j, -(n + 1), n)], log2Size)` with
/// the spec taps folded into the coefficients.
///
/// `n = 2, n2 = 1, log2Size = 3` (chroma filterSize 8).
#[inline(always)]
fn wide_filter6(s: &mut [i32; 14]) {
    let [p2, p1, p0, q0, q1, q2] = [s[4], s[5], s[6], s[7], s[8], s[9]];
    s[5] = (3 * p2 + 2 * p1 + 2 * p0 + q0 + 4) >> 3;
    s[6] = (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3;
    s[7] = (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3;
    s[8] = (p0 + 2 * q0 + 2 * q1 + 3 * q2 + 4) >> 3;
}

/// Wide filter with `n = 3, n2 = 0, log2Size = 3` (see [`wide_filter6`]).
#[inline(always)]
fn wide_filter8(s: &mut [i32; 14]) {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = [s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10]];
    s[4] = (3 * p3 + 2 * p2 + p1 + p0 + q0 + 4) >> 3;
    s[5] = (2 * p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4) >> 3;
    s[6] = (p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4) >> 3;
    s[7] = (p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4) >> 3;
    s[8] = (p1 + p0 + q0 + 2 * q1 + q2 + 2 * q3 + 4) >> 3;
    s[9] = (p0 + q0 + q1 + 2 * q2 + 3 * q3 + 4) >> 3;
}

/// Wide filter with `n = 6, n2 = 1, log2Size = 4` (see [`wide_filter6`]).
#[inline(always)]
fn wide_filter16(s: &mut [i32; 14]) {
    let [p6, p5, p4, p3, p2, p1, p0] = [s[0], s[1], s[2], s[3], s[4], s[5], s[6]];
    let [q0, q1, q2, q3, q4, q5, q6] = [s[7], s[8], s[9], s[10], s[11], s[12], s[13]];
    s[1] = (7 * p6 + 2 * p5 + 2 * p4 + p3 + p2 + p1 + p0 + q0 + 8) >> 4;
    s[2] = (5 * p6 + 2 * p5 + 2 * p4 + 2 * p3 + p2 + p1 + p0 + q0 + q1 + 8) >> 4;
    s[3] = (4 * p6 + p5 + 2 * p4 + 2 * p3 + 2 * p2 + p1 + p0 + q0 + q1 + q2 + 8) >> 4;
    s[4] = (3 * p6 + p5 + p4 + 2 * p3 + 2 * p2 + 2 * p1 + p0 + q0 + q1 + q2 + q3 + 8) >> 4;
    s[5] = (2 * p6 + p5 + p4 + p3 + 2 * p2 + 2 * p1 + 2 * p0 + q0 + q1 + q2 + q3 + q4 + 8) >> 4;
    s[6] = (p6 + p5 + p4 + p3 + p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + q2 + q3 + q4 + q5 + 8) >> 4;
    s[7] = (p5 + p4 + p3 + p2 + p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + q3 + q4 + q5 + q6 + 8) >> 4;
    s[8] = (p4 + p3 + p2 + p1 + p0 + 2 * q0 + 2 * q1 + 2 * q2 + q3 + q4 + q5 + 2 * q6 + 8) >> 4;
    s[9] = (p3 + p2 + p1 + p0 + q0 + 2 * q1 + 2 * q2 + 2 * q3 + q4 + q5 + 3 * q6 + 8) >> 4;
    s[10] = (p2 + p1 + p0 + q0 + q1 + 2 * q2 + 2 * q3 + 2 * q4 + q5 + 4 * q6 + 8) >> 4;
    s[11] = (p1 + p0 + q0 + q1 + q2 + 2 * q3 + 2 * q4 + 2 * q5 + 5 * q6 + 8) >> 4;
    s[12] = (p0 + q0 + q1 + q2 + q3 + 2 * q4 + 2 * q5 + 7 * q6 + 8) >> 4;
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

    /// Direct transcription of spec 7.14.6 operating on a frame of i32.
    mod spec {
        pub struct Frame {
            pub w: usize,
            pub data: Vec<i32>,
            pub bd: u32,
        }
        impl Frame {
            fn at(&self, y: isize, x: isize) -> i32 {
                self.data[y as usize * self.w + x as usize]
            }
            fn set(&mut self, y: isize, x: isize, v: i32) {
                self.data[y as usize * self.w + x as usize] = v;
            }
        }

        fn clip3(lo: i32, hi: i32, v: i32) -> i32 {
            v.clamp(lo, hi)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn sample_filter(
            f: &mut Frame,
            x: isize,
            y: isize,
            plane: usize,
            limit: i32,
            blimit: i32,
            thresh: i32,
            dx: isize,
            dy: isize,
            filter_size: u32,
        ) -> usize {
            let bd = f.bd;
            let s = |k: isize| f.at(y + dy * k, x + dx * k);
            let (q0, q1, q2, q3) = (s(0), s(1), s(2), s(3));
            let (p0, p1, p2, p3) = (s(-1), s(-2), s(-3), s(-4));
            let mut hev = 0;
            let thresh_bd = thresh << (bd - 8);
            hev |= ((p1 - p0).abs() > thresh_bd) as i32;
            hev |= ((q1 - q0).abs() > thresh_bd) as i32;
            let filter_len = if filter_size == 4 {
                4
            } else if plane != 0 {
                6
            } else if filter_size == 8 {
                8
            } else {
                16
            };
            let limit_bd = limit << (bd - 8);
            let blimit_bd = blimit << (bd - 8);
            let mut mask = 0;
            mask |= ((p1 - p0).abs() > limit_bd) as i32;
            mask |= ((q1 - q0).abs() > limit_bd) as i32;
            mask |= ((p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 > blimit_bd) as i32;
            if filter_len >= 6 {
                mask |= ((p2 - p1).abs() > limit_bd) as i32;
                mask |= ((q2 - q1).abs() > limit_bd) as i32;
            }
            if filter_len >= 8 {
                mask |= ((p3 - p2).abs() > limit_bd) as i32;
                mask |= ((q3 - q2).abs() > limit_bd) as i32;
            }
            let filter_mask = mask == 0;
            let threshold_bd = 1 << (bd - 8);
            let mut flat_mask = false;
            if filter_size >= 8 {
                let mut mask = 0;
                mask |= ((p1 - p0).abs() > threshold_bd) as i32;
                mask |= ((q1 - q0).abs() > threshold_bd) as i32;
                mask |= ((p2 - p0).abs() > threshold_bd) as i32;
                mask |= ((q2 - q0).abs() > threshold_bd) as i32;
                if filter_len >= 8 {
                    mask |= ((p3 - p0).abs() > threshold_bd) as i32;
                    mask |= ((q3 - q0).abs() > threshold_bd) as i32;
                }
                flat_mask = mask == 0;
            }
            let mut flat_mask2 = false;
            if filter_size >= 16 {
                let (q4, q5, q6) = (s(4), s(5), s(6));
                let (p4, p5, p6) = (s(-5), s(-6), s(-7));
                let mut mask = 0;
                mask |= ((p6 - p0).abs() > threshold_bd) as i32;
                mask |= ((q6 - q0).abs() > threshold_bd) as i32;
                mask |= ((p5 - p0).abs() > threshold_bd) as i32;
                mask |= ((q5 - q0).abs() > threshold_bd) as i32;
                mask |= ((p4 - p0).abs() > threshold_bd) as i32;
                mask |= ((q4 - q0).abs() > threshold_bd) as i32;
                flat_mask2 = mask == 0;
            }
            // Returns the taken path (for coverage checks).
            if !filter_mask {
                0
            } else if filter_size == 4 || !flat_mask {
                narrow(f, x, y, hev != 0, dx, dy);
                1 + hev as usize
            } else if filter_size == 8 || !flat_mask2 {
                wide(f, x, y, plane, dx, dy, 3);
                3
            } else {
                wide(f, x, y, plane, dx, dy, 4);
                4
            }
        }

        fn narrow(f: &mut Frame, x: isize, y: isize, hev: bool, dx: isize, dy: isize) {
            let bd = f.bd;
            let c = |v: i32| clip3(-(1 << (bd - 1)), (1 << (bd - 1)) - 1, v);
            let q0 = f.at(y, x);
            let q1 = f.at(y + dy, x + dx);
            let p0 = f.at(y - dy, x - dx);
            let p1 = f.at(y - dy * 2, x - dx * 2);
            let o = 0x80 << (bd - 8);
            let (ps1, ps0, qs0, qs1) = (p1 - o, p0 - o, q0 - o, q1 - o);
            let mut filter = if hev { c(ps1 - qs1) } else { 0 };
            filter = c(filter + 3 * (qs0 - ps0));
            let filter1 = c(filter + 4) >> 3;
            let filter2 = c(filter + 3) >> 3;
            f.set(y, x, c(qs0 - filter1) + o);
            f.set(y - dy, x - dx, c(ps0 + filter2) + o);
            if !hev {
                let filter = (filter1 + 1) >> 1;
                f.set(y + dy, x + dx, c(qs1 - filter) + o);
                f.set(y - dy * 2, x - dx * 2, c(ps1 + filter) + o);
            }
        }

        fn wide(f: &mut Frame, x: isize, y: isize, plane: usize, dx: isize, dy: isize, log2: u32) {
            let n: isize = if log2 == 4 {
                6
            } else if plane == 0 {
                3
            } else {
                2
            };
            let n2: isize = if log2 == 3 && plane == 0 { 0 } else { 1 };
            let mut out = vec![0; 2 * n as usize];
            for i in -n..n {
                let mut t = 0;
                for j in -n..=n {
                    let p = (i + j).clamp(-(n + 1), n);
                    let tap = if j.abs() <= n2 { 2 } else { 1 };
                    t += f.at(y + p * dy, x + p * dx) * tap;
                }
                out[(i + n) as usize] = (t + (1 << (log2 - 1))) >> log2;
            }
            for i in -n..n {
                f.set(y + i * dy, x + i * dx, out[(i + n) as usize]);
            }
        }
    }

    #[test]
    fn limits() {
        // Spot checks against hand computed values.
        assert_eq!(
            filter_limits(10, 0),
            FilterLimits {
                limit: 10,
                blimit: 34,
                thresh: 0
            }
        );
        assert_eq!(
            filter_limits(63, 0),
            FilterLimits {
                limit: 63,
                blimit: 193,
                thresh: 3
            }
        );
        assert_eq!(
            filter_limits(63, 7),
            FilterLimits {
                limit: 2,
                blimit: 132,
                thresh: 3
            }
        );
        assert_eq!(
            filter_limits(20, 3),
            FilterLimits {
                limit: 6,
                blimit: 50,
                thresh: 1
            }
        );
        assert_eq!(
            filter_limits(1, 5),
            FilterLimits {
                limit: 1,
                blimit: 7,
                thresh: 0
            }
        );
    }

    /// Generates a line profile across an edge that hits all filter paths:
    /// flat regions, small noise and steps of various heights.
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
        let noise = match rng.range(0, 3) {
            0 => 0,
            1 => sc,
            2 => 3 * sc,
            _ => 40 * sc,
        };
        let half = out.len() / 2;
        for (k, v) in out.iter_mut().enumerate() {
            let s = if k >= half { step } else { 0 };
            *v = (base + s + rng.range(-noise, noise)).clamp(0, maxv);
        }
    }

    fn run_random<T: Pixel>(bd: u32, seed: u64) {
        let mut rng = Rng(seed);
        const W: usize = 24;
        const H: usize = 24;
        const E: usize = 8; // edge position
                            // paths[kind][path]: coverage of the spec paths per filter kind.
        let mut paths = [[0usize; 5]; 5];
        for iter in 0..20000 {
            let kind = rng.range(0, 4) as usize;
            let (plane, spec_size) = match kind {
                0 => (0, 4),
                1 => (0, 8),
                2 => (0, 16),
                3 => (1, 4),
                _ => (2, 8),
            };
            let vertical = rng.range(0, 1) == 0;
            let lvl = rng.range(1, 63);
            let sharp = rng.range(0, 7);
            let limits = filter_limits(lvl, sharp);
            let len = rng.range(1, 8) as usize;
            let mut frame = spec::Frame {
                w: W,
                data: vec![0; W * H],
                bd,
            };
            // Fill with lines perpendicular to the edge.
            let mut line = [0i32; W];
            for a in 0..H {
                gen_line(&mut rng, bd, &mut line[..2 * E]);
                for (k, &v) in line[..2 * E].iter().enumerate() {
                    if vertical {
                        frame.data[a * W + k] = v;
                    } else {
                        frame.data[k * W + a] = v;
                    }
                }
            }
            let mut buf: Vec<T> = frame.data.iter().map(|&v| T::from_i32(v)).collect();
            let start = 4usize;
            let ks = kernel_filter_size(plane, spec_size);
            if vertical {
                filter_vertical_edge(&mut buf, W, E, start, len, ks, limits, bd);
                for r in start..start + len {
                    paths[kind][spec::sample_filter(
                        &mut frame,
                        E as isize,
                        r as isize,
                        plane,
                        limits.limit,
                        limits.blimit,
                        limits.thresh,
                        1,
                        0,
                        spec_size,
                    )] += 1;
                }
            } else {
                filter_horizontal_edge(&mut buf, W, start, E, len, ks, limits, bd);
                for c in start..start + len {
                    paths[kind][spec::sample_filter(
                        &mut frame,
                        c as isize,
                        E as isize,
                        plane,
                        limits.limit,
                        limits.blimit,
                        limits.thresh,
                        0,
                        1,
                        spec_size,
                    )] += 1;
                }
            }
            let got: Vec<i32> = buf.iter().map(|v| v.to_i32()).collect();
            assert_eq!(
                got, frame.data,
                "mismatch iter {iter} bd {bd} plane {plane} size {spec_size} vertical {vertical}"
            );
        }
        // Every reachable path must have been exercised.
        let reachable: [&[usize]; 5] = [
            &[0, 1, 2],
            &[0, 1, 2, 3],
            &[0, 1, 2, 3, 4],
            &[0, 1, 2],
            &[0, 1, 2, 3],
        ];
        for (kind, r) in reachable.iter().enumerate() {
            for &path in r.iter() {
                assert!(
                    paths[kind][path] > 50,
                    "bd {bd} kind {kind} path {path}: {paths:?}"
                );
            }
        }
    }

    #[test]
    fn random_vs_spec_8bit() {
        run_random::<u8>(8, 0x1234_5678_9abc_def1);
        run_random::<u16>(8, 0x0bad_cafe_dead_beef);
    }

    #[test]
    fn random_vs_spec_10bit() {
        run_random::<u16>(10, 0x9e37_79b9_7f4a_7c15);
    }

    #[test]
    fn random_vs_spec_12bit() {
        run_random::<u16>(12, 0xd1b5_4a32_d192_ed03);
    }

    /// Checks that each filter size modifies exactly the expected number of
    /// samples on each side of the edge (and never more).
    #[test]
    fn touches_only_allowed_samples() {
        let mut rng = Rng(42);
        for &size in &[4u32, 6, 8, 16] {
            let half = (size / 2) as usize;
            let mut changed_max = 0;
            for _ in 0..5000 {
                let mut line = [0i32; 16];
                gen_line(&mut rng, 8, &mut line);
                let mut buf: Vec<u8> = line.iter().map(|&v| v as u8).collect();
                let orig = buf.clone();
                let limits = filter_limits(rng.range(1, 63), rng.range(0, 7));
                filter_vertical_edge(&mut buf, 16, 8, 0, 1, size, limits, 8);
                for k in 0..16 {
                    if buf[k] != orig[k] {
                        let d = if k >= 8 { k - 8 + 1 } else { 8 - k };
                        changed_max = changed_max.max(d);
                    }
                }
            }
            let expect = match size {
                4 => 2,
                6 => 2,
                8 => 3,
                _ => 6,
            };
            assert_eq!(changed_max, expect, "size {size}");
            assert!(changed_max <= half);
        }
    }
}
