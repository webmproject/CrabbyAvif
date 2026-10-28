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

//! CDEF direction search and filter (spec 7.15.2 and 7.15.3).
//!
//! OWNER: lf/cdef worker. Contract below must be kept.
//!
//! The caller implements 7.15 / 7.15.1 (block iteration, skip check,
//! strength / damping / direction adjustment) and prepares a padded
//! source buffer.
//!
//! # Unavailable samples
//!
//! The spec's `cdef_get_at` returns `CdefAvailable == 0` for samples
//! outside the filter region, and such taps contribute nothing to `sum`,
//! `max` or `min`. Here the caller stores [`CDEF_VERY_LARGE`] in those
//! positions of the padded source and [`cdef_filter`] skips every tap
//! equal to the sentinel. As valid samples are at most 4095 the sentinel
//! can never be confused with a real sample, so this is exactly the
//! spec's behaviour for any input.
//!
//! (The fast path relies on this: with `damping <= 13` and strengths
//! below 25905, which includes all spec-conformant values `threshold <=
//! 240` and `damping <= 10`, `constrain(VERY_LARGE - x)` is always 0 since
//! `|diff| >= 30000 - 4095 >= 1 << 14` and `(1 << 14) >> (damping -
//! FloorLog2(threshold)) > threshold`; and the sentinel never lowers
//! `min`, so only `max` needs masking. Other parameters use a generic
//! path that skips sentinel taps explicitly.)
//!
//! # Performance
//!
//! The fast path ([`cdef_filter`] → `filter_fast`) works on 8 lanes of
//! `i16` (one row of an 8 wide block or two rows of a 4 wide one), branch
//! free, with separate primary-only / secondary-only / both variants, so
//! that the compiler vectorizes it even for baseline x86-64 (SSE2).

use crate::decoder::av1::pixel::{floor_log2, Pixel};

/// Sentinel for unavailable source samples (outside the frame, i.e.
/// `is_inside_filter_region` false). Larger than any valid 12-bit sample.
pub const CDEF_VERY_LARGE: u16 = 30000;

/// Border (in samples) around a block in the padded CDEF source buffer.
pub const CDEF_BORDER: usize = 2;

/// Block edge flags: the [`CDEF_BORDER`] columns / rows on that side of a
/// block are inside the frame (spec `CdefAvailable`), see [`CdefUnit`].
pub const CDEF_HAVE_LEFT: u8 = 1;
/// See [`CDEF_HAVE_LEFT`].
pub const CDEF_HAVE_RIGHT: u8 = 2;
/// See [`CDEF_HAVE_LEFT`].
pub const CDEF_HAVE_TOP: u8 = 4;
/// See [`CDEF_HAVE_LEFT`].
pub const CDEF_HAVE_BOTTOM: u8 = 8;
/// All borders of a block are available.
pub const CDEF_HAVE_ALL: u8 = 15;

/// Spec `Cdef_Uv_Dir[subsampling_x][subsampling_y][yDir]`: chroma
/// direction for a luma direction (used by the caller, 7.15.1 step 11).
pub const CDEF_UV_DIR: [[[usize; 8]; 2]; 2] = [
    [[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 2, 2, 3, 4, 6, 0]],
    [[7, 0, 2, 4, 5, 6, 6, 6], [0, 1, 2, 3, 4, 5, 6, 7]],
];

/// Spec `Cdef_Directions[dir][k]` = `[dy, dx]` of tap `k`.
pub(crate) const CDEF_DIRECTIONS: [[[isize; 2]; 2]; 8] = [
    [[-1, 1], [-2, 2]],
    [[0, 1], [-1, 2]],
    [[0, 1], [0, 2]],
    [[0, 1], [1, 2]],
    [[1, 1], [2, 2]],
    [[1, 0], [2, 1]],
    [[1, 0], [2, 0]],
    [[1, 0], [2, -1]],
];

/// Spec `Cdef_Pri_Taps`.
pub(crate) const CDEF_PRI_TAPS: [[i32; 2]; 2] = [[4, 2], [3, 3]];

/// Spec `Cdef_Sec_Taps`.
pub(crate) const CDEF_SEC_TAPS: [[i32; 2]; 2] = [[2, 1], [2, 1]];

/// Spec `Div_Table` (7.15.2).
const DIV_TABLE: [i32; 9] = [0, 840, 420, 280, 210, 168, 140, 120, 105];

/// Spec 7.15.2 partial sums `partial[d]` of the direction search, with
/// directions 3 and 4 stored mirrored (they run against the column
/// index): `p3r[k] = partial[3][10 - k]`, `p4r[k] = partial[4][14 - k]`.
/// Odd directions only use indices 0..=10.
struct Partials {
    p0: [i32; 15],
    p1: [i32; 11],
    p2: [i32; 8],
    p3r: [i32; 11],
    p4r: [i32; 15],
    p5: [i32; 11],
    p6: [i32; 8],
    p7: [i32; 11],
}

impl Partials {
    /// Accumulates row `I` (values `x[j] = (px >> (bd - 8)) - 128`).
    #[inline(always)]
    fn add_row<const I: usize>(&mut self, x: [i32; 8]) {
        let pairs: [i32; 4] = std::array::from_fn(|k| x[2 * k] + x[2 * k + 1]);
        self.p2[I] = x.iter().sum();
        for j in 0..8 {
            self.p0[I + j] += x[j]; // partial[0][i + j]
            self.p4r[7 - I + j] += x[j]; // partial[4][7 + i - j]
            self.p5[3 - I / 2 + j] += x[j]; // partial[5][3 - i / 2 + j]
            self.p6[j] += x[j]; // partial[6][j]
            self.p7[I / 2 + j] += x[j]; // partial[7][i / 2 + j]
        }
        for k in 0..4 {
            self.p1[I + k] += pairs[k]; // partial[1][i + j / 2]
            self.p3r[7 - I + k] += pairs[k]; // partial[3][3 + i - j / 2]
        }
    }
}

/// Spec 7.15.2: returns `(yDir, var)` for the 8x8 luma block whose top-left
/// sample is `src[0]` (row stride `stride`). All 64 samples are valid.
pub fn cdef_direction<T: Pixel>(src: &[T], stride: usize, bd: u32) -> (usize, i32) {
    // The partial sums of the spec are accumulated row by row as vector
    // additions at row dependent offsets. Directions 3 and 4 run "against"
    // the column index and are accumulated mirrored:
    // p3r[k] = partial[3][10 - k], p4r[k] = partial[4][14 - k].
    let shift = bd - 8;
    let mut p = Partials {
        p0: [0; 15],
        p1: [0; 11],
        p2: [0; 8],
        p3r: [0; 11],
        p4r: [0; 15],
        p5: [0; 11],
        p6: [0; 8],
        p7: [0; 11],
    };
    // One call per row with a constant row index, so that all offsets are
    // compile-time constants and the inner loops vectorize.
    macro_rules! rows {
        ($($i:literal)*) => {$(
            let row: &[T; 8] = src[$i * stride..$i * stride + 8].try_into().unwrap();
            p.add_row::<$i>(row.map(|px| (px.to_i32() >> shift) - 128));
        )*};
    }
    rows!(0 1 2 3 4 5 6 7);
    let Partials {
        p0,
        p1,
        p2,
        p3r,
        p4r,
        p5,
        p6,
        p7,
    } = p;
    let sq = |v: i32| v * v;
    let mut cost = [0i32; 8];
    cost[2] = p2.iter().map(|&v| sq(v)).sum::<i32>() * DIV_TABLE[8];
    cost[6] = p6.iter().map(|&v| sq(v)).sum::<i32>() * DIV_TABLE[8];
    // Directions 0 and 4: cost is symmetric under the mirroring.
    let diag = |p: &[i32; 15]| {
        let mut c = sq(p[7]) * DIV_TABLE[8];
        for i in 0..7 {
            c += (sq(p[i]) + sq(p[14 - i])) * DIV_TABLE[i + 1];
        }
        c
    };
    cost[0] = diag(&p0);
    cost[4] = diag(&p4r);
    // Odd directions (also symmetric under mirroring).
    let odd = |p: &[i32; 11]| {
        let mut c = 0;
        for &v in &p[3..8] {
            c += sq(v);
        }
        c *= DIV_TABLE[8];
        for j in 0..3 {
            c += (sq(p[j]) + sq(p[10 - j])) * DIV_TABLE[2 * j + 2];
        }
        c
    };
    cost[1] = odd(&p1);
    cost[3] = odd(&p3r);
    cost[5] = odd(&p5);
    cost[7] = odd(&p7);
    let mut best_cost = 0;
    let mut y_dir = 0;
    for (i, &c) in cost.iter().enumerate() {
        if c > best_cost {
            best_cost = c;
            y_dir = i;
        }
    }
    (y_dir, (best_cost - cost[(y_dir + 4) & 7]) >> 10)
}

/// Spec `constrain(diff, threshold, damping)` with the damping adjustment
/// `adj = Max(0, damping - FloorLog2(threshold))` precomputed.
///
/// No special case is needed for `threshold == 0`: then
/// `threshold - (|diff| >> adj) <= 0` and the result is 0 as in the spec.
/// (`threshold` must not be negative.)
#[inline(always)]
fn constrain(diff: i32, threshold: i32, adj: u32) -> i32 {
    let ad = diff.abs();
    let v = (threshold - (ad >> adj)).max(0).min(ad);
    if diff < 0 {
        -v
    } else {
        v
    }
}

/// `Max(0, damping - FloorLog2(threshold))` (0 if `threshold` is 0; the
/// value is unused then).
#[inline]
pub(crate) fn damping_adj(threshold: i32, damping: i32) -> u32 {
    if threshold <= 0 {
        0
    } else {
        (damping - floor_log2(threshold as u32) as i32).max(0) as u32
    }
}

/// Spec 7.15.3 for one block of `w x h` samples (8x8, 4x8, 8x4 or 4x4).
///
/// * `src`: pre-CDEF samples with a [`CDEF_BORDER`] border on all sides;
///   the block's top-left sample is at `src[CDEF_BORDER * src_stride +
///   CDEF_BORDER]`. Unavailable samples hold [`CDEF_VERY_LARGE`] and must
///   be treated as `CdefAvailable == 0`.
/// * `dst`: output, `dst[0]` = top-left sample of the block.
/// * `pri_str`, `sec_str`, `damping`, `dir`: already adjusted exactly as the
///   inputs of spec 7.15.3 (i.e. after steps 1-6 / 9-12 of 7.15.1, including
///   the `<< coeffShift`).
#[allow(clippy::too_many_arguments)]
pub fn cdef_filter<T: Pixel>(
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
    cdef_filter_src(
        dst, dst_stride, src, src_stride, w, h, pri_str, sec_str, damping, dir, bd,
    );
}

/// [`cdef_filter`] for a source of any sample type: `u16` with
/// [`CDEF_VERY_LARGE`] for unavailable samples, or pixels (all samples
/// available).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn cdef_filter_src<T: Pixel, S: Pixel>(
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
    bd: u32,
) {
    let coeff_shift = bd - 8;
    let tap_set = ((pri_str >> coeff_shift) & 1) as usize;
    let k = Kernel {
        pri_str,
        sec_str,
        pri_adj: damping_adj(pri_str, damping),
        sec_adj: damping_adj(sec_str, damping),
        pri_taps: CDEF_PRI_TAPS[tap_set],
        dir,
        pri_off: [
            tap_offset(dir, 0, src_stride),
            tap_offset(dir, 1, src_stride),
        ],
        sec_off: [
            [
                tap_offset((dir + 6) & 7, 0, src_stride),
                tap_offset((dir + 2) & 7, 0, src_stride),
            ],
            [
                tap_offset((dir + 6) & 7, 1, src_stride),
                tap_offset((dir + 2) & 7, 1, src_stride),
            ],
        ],
    };

    // Fast paths rely on constrain() of a sentinel tap being 0, which
    // holds whenever damping <= 13 and the strengths are below 25905 (see
    // the module documentation), and on 16-bit arithmetic not overflowing
    // (strengths below FAST_MAX_STRENGTH). Conformant streams use damping
    // <= 10 and strengths <= 240. Anything else takes the generic path.
    let fast = damping <= 13
        && (0..FAST_MAX_STRENGTH).contains(&pri_str)
        && (0..FAST_MAX_STRENGTH).contains(&sec_str);
    if !fast {
        filter_generic(dst, dst_stride, src, src_stride, w, h, &k);
        return;
    }
    macro_rules! fast {
        ($w:literal, $h:literal) => {
            match (pri_str != 0, sec_str != 0) {
                (true, true) => {
                    filter_fast::<T, S, $w, $h, true, true>(dst, dst_stride, src, src_stride, &k)
                }
                (true, false) => {
                    filter_fast::<T, S, $w, $h, true, false>(dst, dst_stride, src, src_stride, &k)
                }
                (false, true) => {
                    filter_fast::<T, S, $w, $h, false, true>(dst, dst_stride, src, src_stride, &k)
                }
                (false, false) => copy_block(dst, dst_stride, src, src_stride, $w, $h),
            }
        };
    }
    match (w, h) {
        (8, 8) => fast!(8, 8),
        (8, 4) => fast!(8, 4),
        (4, 8) => fast!(4, 8),
        (4, 4) => fast!(4, 4),
        _ => filter_generic(dst, dst_stride, src, src_stride, w, h, &k),
    }
}

/// Both strengths 0: every constrain() is 0 and the output is the input
/// (the caller normally skips such blocks).
fn copy_block<T: Pixel, S: Pixel>(
    dst: &mut [T],
    dst_stride: usize,
    src: &[S],
    src_stride: usize,
    w: usize,
    h: usize,
) {
    for i in 0..h {
        let s = (i + CDEF_BORDER) * src_stride + CDEF_BORDER;
        for (d, &v) in dst[i * dst_stride..i * dst_stride + w]
            .iter_mut()
            .zip(&src[s..s + w])
        {
            *d = T::from_i32(v.to_i32());
        }
    }
}

/// Strength limit (exclusive) of the 16-bit fast path.
pub(crate) const FAST_MAX_STRENGTH: i32 = 1024;

/// Per-block filter parameters shared by the kernels.
struct Kernel {
    pri_str: i32,
    sec_str: i32,
    pri_adj: u32,
    sec_adj: u32,
    pri_taps: [i32; 2],
    dir: usize,
    /// Index offsets of primary tap `k` (direction `dir`).
    pri_off: [isize; 2],
    /// Index offsets of secondary tap `k` for directions `dir - 2`, `dir + 2`.
    sec_off: [[isize; 2]; 2],
}

/// Signed index offset of tap `k` of direction `d` for row stride `stride`.
#[inline]
pub(crate) fn tap_offset(d: usize, k: usize, stride: usize) -> isize {
    let [dy, dx] = CDEF_DIRECTIONS[d][k];
    dy * stride as isize + dx
}

/// Number of 16-bit lanes processed together by [`filter_fast`] (one
/// 128-bit vector): one row of an 8 wide block or two rows of a 4 wide one.
const LANES: usize = 8;

/// A vector of [`LANES`] samples or intermediate values.
type Lanes = [i16; LANES];

/// Row stride of the local copy of a block and its border in
/// [`filter_fast`] (`>= 8 + 2 * CDEF_BORDER`).
const LOCAL_STRIDE: usize = 12;

/// Length of the window of the local copy read through one tap offset:
/// rows `0..8`, columns `0..8` relative to the tap position.
const TAP_WINDOW: usize = 7 * LOCAL_STRIDE + 8;

/// Window of the local block copy `blk` (see [`filter_fast`]) seen
/// through tap `t` of direction `d` (`sign` = 1 or -1, or 0 for the
/// center), i.e. starting at that offset from the block's top-left sample.
/// The start is in `0..=4 * LOCAL_STRIDE + 4`, so the window fits.
#[inline(always)]
fn tap_window(
    blk: &[i16; LOCAL_STRIDE * LOCAL_STRIDE],
    d: usize,
    t: usize,
    sign: isize,
) -> &[i16; TAP_WINDOW] {
    let [dy, dx] = CDEF_DIRECTIONS[d][t];
    let s = LOCAL_STRIDE as isize;
    let b = CDEF_BORDER as isize;
    let start = ((b + sign * dy) * s + b + sign * dx) as usize;
    blk[start..start + TAP_WINDOW].try_into().unwrap()
}

/// Loads the lanes for rows `i..i + LANES / W` of a `W` wide block from
/// the tap window `t` (see [`filter_fast`]).
#[inline(always)]
fn load<const W: usize>(t: &[i16; TAP_WINDOW], i: usize) -> Lanes {
    let mut v = [0i16; LANES];
    for (r, chunk) in v.chunks_exact_mut(W).enumerate() {
        let start = (i + r) * LOCAL_STRIDE;
        chunk.copy_from_slice(&t[start..start + W]);
    }
    v
}

/// Accumulates `weight * constrain(p - x, strength, adj)` into `sum` (and
/// updates `min` / `max` when `CLAMP`, masking unavailable taps for `max`
/// when `MASK`), lane-wise.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tap<const CLAMP: bool, const MASK: bool>(
    sum: &mut Lanes,
    min: &mut Lanes,
    max: &mut Lanes,
    x: &Lanes,
    p: Lanes,
    strength: i16,
    adj: u32,
    weight: i16,
) {
    const SENTINEL: i16 = CDEF_VERY_LARGE as i16;
    // Written branch free (sign masks instead of abs() / conditional
    // negation) so that the loop vectorizes.
    for j in 0..LANES {
        let d = p[j] - x[j];
        let sign = d >> 15; // 0 or -1
        let ad = (d ^ sign) - sign;
        let v = (strength - (ad >> adj)).max(0).min(ad);
        sum[j] += weight * ((v ^ sign) - sign);
        if CLAMP {
            min[j] = min[j].min(p[j]);
            max[j] = max[j].max(if MASK { p[j] & -((p[j] != SENTINEL) as i16) } else { p[j] });
        }
    }
}

/// Optimised spec 7.15.3 for `W x H` blocks (`W`, `H` in {4, 8}), valid
/// when `damping <= 13` and both strengths are below
/// [`FAST_MAX_STRENGTH`]. Arithmetic is done in 16 bits, lane parallel, so
/// that the compiler can vectorize it even for baseline x86-64 (SSE2 has
/// 16-bit min / max / multiply but no 32-bit ones).
///
/// * The block and its border are first copied into a local array with a
///   constant stride; every tap then reads through a fixed size window of
///   it, so that all further indexing is checked at compile time.
/// * constrain() of an unavailable ([`CDEF_VERY_LARGE`]) tap is 0 (module
///   documentation), and such a tap never lowers `min`; so unavailable
///   taps only need masking for `max`.
/// * When only one of the strengths is non-zero, the final clamp to
///   `[min, max]` is a no-op and is skipped (as in dav1d): the tap weights
///   of either filter sum to 12, and `|constrain(d)| <= |d|` with the sign
///   of `d`, so `sum <= 12 * (max - x)` and `sum >= -12 * (x - min)`, which
///   gives `x + ((8 + sum - (sum < 0)) >> 4)` within `[min, max]`.
/// * No 16-bit overflow: `|p - x| <= 30000`, and `|sum| <= 12 * (pri_str +
///   sec_str) < 24576`.
#[inline(always)]
fn filter_fast<
    T: Pixel,
    Src: Pixel,
    const W: usize,
    const H: usize,
    const PRI: bool,
    const SEC: bool,
>(
    dst: &mut [T],
    dst_stride: usize,
    src: &[Src],
    src_stride: usize,
    k: &Kernel,
) {
    const S: usize = LOCAL_STRIDE;

    // Local copy: rows 0..H + 4, columns 0..W + 4 (samples < 4096 and the
    // sentinel 30000 fit in i16).
    let mut blk = [0i16; S * S];
    let mut sentinels = 0u16;
    for r in 0..H + 2 * CDEF_BORDER {
        let s = &src[r * src_stride..r * src_stride + W + 2 * CDEF_BORDER];
        for (d, &v) in blk[r * S..r * S + W + 2 * CDEF_BORDER].iter_mut().zip(s) {
            let v = v.to_u16();
            *d = v as i16;
            sentinels |= (v == CDEF_VERY_LARGE) as u16;
        }
    }
    // Window of the local copy seen through the tap at offset (dy, dx)
    // from the block's top-left sample. Its start is in 0..=4 * S + 4 and
    // start + TAP_WINDOW <= S * S.
    let center = tap_window(&blk, 0, 0, 0);
    let dir = k.dir;
    let pri_win = [
        [tap_window(&blk, dir, 0, 1), tap_window(&blk, dir, 0, -1)],
        [tap_window(&blk, dir, 1, 1), tap_window(&blk, dir, 1, -1)],
    ];
    let (s0, s1) = ((dir + 6) & 7, (dir + 2) & 7);
    let sec_win = [
        [
            tap_window(&blk, s0, 0, 1),
            tap_window(&blk, s0, 0, -1),
            tap_window(&blk, s1, 0, 1),
            tap_window(&blk, s1, 0, -1),
        ],
        [
            tap_window(&blk, s0, 1, 1),
            tap_window(&blk, s0, 1, -1),
            tap_window(&blk, s1, 1, 1),
            tap_window(&blk, s1, 1, -1),
        ],
    ];

    let windows = Windows {
        center,
        pri: pri_win,
        sec: sec_win,
    };
    // Unavailable samples only occur near frame edges; elsewhere `max`
    // needs no masking.
    if PRI && SEC && sentinels != 0 {
        filter_rows::<T, W, H, PRI, SEC, true>(dst, dst_stride, &windows, k);
    } else {
        filter_rows::<T, W, H, PRI, SEC, false>(dst, dst_stride, &windows, k);
    }
}

/// Tap windows of a block (see [`filter_fast`]): `pri[k]` for the primary
/// taps `+k` / `-k`, `sec[k]` for the secondary taps `+k` / `-k` of
/// directions `dir - 2` and `dir + 2`.
struct Windows<'a> {
    center: &'a [i16; TAP_WINDOW],
    pri: [[&'a [i16; TAP_WINDOW]; 2]; 2],
    sec: [[&'a [i16; TAP_WINDOW]; 4]; 2],
}

/// Row loop of [`filter_fast`].
#[inline(always)]
fn filter_rows<
    T: Pixel,
    const W: usize,
    const H: usize,
    const PRI: bool,
    const SEC: bool,
    const MASK: bool,
>(
    dst: &mut [T],
    dst_stride: usize,
    win: &Windows,
    k: &Kernel,
) {
    let clamp = PRI && SEC;
    let rows = LANES / W;
    let (pri_str, sec_str) = (k.pri_str as i16, k.sec_str as i16);
    let pri_taps = k.pri_taps.map(|t| t as i16);
    for i in (0..H).step_by(rows) {
        let x = load::<W>(win.center, i);
        let mut max = x;
        let mut min = x;
        let mut sum = [0i16; LANES];
        for t in 0..2 {
            if PRI {
                for w in win.pri[t] {
                    let p = load::<W>(w, i);
                    if clamp {
                        tap::<true, MASK>(
                            &mut sum,
                            &mut min,
                            &mut max,
                            &x,
                            p,
                            pri_str,
                            k.pri_adj,
                            pri_taps[t],
                        );
                    } else {
                        tap::<false, false>(
                            &mut sum,
                            &mut min,
                            &mut max,
                            &x,
                            p,
                            pri_str,
                            k.pri_adj,
                            pri_taps[t],
                        );
                    }
                }
            }
            if SEC {
                let weight = CDEF_SEC_TAPS[0][t] as i16;
                for w in win.sec[t] {
                    let p = load::<W>(w, i);
                    if clamp {
                        tap::<true, MASK>(
                            &mut sum, &mut min, &mut max, &x, p, sec_str, k.sec_adj, weight,
                        );
                    } else {
                        tap::<false, false>(
                            &mut sum, &mut min, &mut max, &x, p, sec_str, k.sec_adj, weight,
                        );
                    }
                }
            }
        }
        let mut out = [0i16; LANES];
        for j in 0..LANES {
            let s = sum[j];
            let mut v = x[j] + ((8 + s + (s >> 15)) >> 4);
            if clamp {
                v = v.max(min[j]).min(max[j]);
            }
            out[j] = v;
        }
        for (r, chunk) in out.chunks_exact(W).enumerate() {
            let start = (i + r) * dst_stride;
            let row: &mut [T; W] = (&mut dst[start..start + W]).try_into().unwrap();
            for (d, &v) in row.iter_mut().zip(chunk) {
                *d = T::from_i32(v as i32);
            }
        }
    }
}

/// Spec 7.15.3 for any parameters, skipping every [`CDEF_VERY_LARGE`] tap.
#[allow(clippy::too_many_arguments)]
fn filter_generic<T: Pixel, S: Pixel>(
    dst: &mut [T],
    dst_stride: usize,
    src: &[S],
    src_stride: usize,
    w: usize,
    h: usize,
    k: &Kernel,
) {
    let constrain = |diff: i32, threshold: i32, adj: u32| {
        if threshold <= 0 {
            0
        } else {
            constrain(diff, threshold, adj)
        }
    };
    for i in 0..h {
        let src_row = (i + CDEF_BORDER) * src_stride + CDEF_BORDER;
        let dst_row = &mut dst[i * dst_stride..i * dst_stride + w];
        for (j, out) in dst_row.iter_mut().enumerate() {
            let pos = (src_row + j) as isize;
            let x = src[pos as usize].to_i32();
            let mut sum = 0;
            let mut max = x;
            let mut min = x;
            for t in 0..2 {
                for sign in [-1isize, 1] {
                    let p = src[(pos + sign * k.pri_off[t]) as usize].to_u16();
                    if p != CDEF_VERY_LARGE {
                        let p = p as i32;
                        sum += k.pri_taps[t] * constrain(p - x, k.pri_str, k.pri_adj);
                        max = max.max(p);
                        min = min.min(p);
                    }
                    for &o in &k.sec_off[t] {
                        let s = src[(pos + sign * o) as usize].to_u16();
                        if s != CDEF_VERY_LARGE {
                            let s = s as i32;
                            sum += CDEF_SEC_TAPS[0][t] * constrain(s - x, k.sec_str, k.sec_adj);
                            max = max.max(s);
                            min = min.min(s);
                        }
                    }
                }
            }
            let v = x + ((8 + sum - (sum < 0) as i32) >> 4);
            *out = T::from_i32(v.clamp(min, max));
        }
    }
}

/// Row stride of the padded copy of a block made by [`pad_block`].
pub const CDEF_PAD_STRIDE: usize = 8 + 2 * CDEF_BORDER;

/// Copies the `(h + 4) x (w + 4)` source of a `w x h` block (`w`, `h` <=
/// 8; `src` as for [`CdefKernels::filter`]) into `tmp` (row stride
/// [`CDEF_PAD_STRIDE`]), with [`CDEF_VERY_LARGE`] for the samples that
/// are outside the frame according to `edges`. This is the source format
/// of [`cdef_filter`].
#[inline]
pub fn pad_block<T: Pixel>(
    tmp: &mut [u16; CDEF_PAD_STRIDE * CDEF_PAD_STRIDE],
    src: &[T],
    src_stride: usize,
    w: usize,
    h: usize,
    edges: u8,
) {
    const S: usize = CDEF_PAD_STRIDE;
    const B: usize = CDEF_BORDER;
    let r0 = if edges & CDEF_HAVE_TOP != 0 { 0 } else { B };
    let r1 = if edges & CDEF_HAVE_BOTTOM != 0 { h + 2 * B } else { h + B };
    let c0 = if edges & CDEF_HAVE_LEFT != 0 { 0 } else { B };
    let c1 = if edges & CDEF_HAVE_RIGHT != 0 { w + 2 * B } else { w + B };
    tmp.fill(CDEF_VERY_LARGE);
    for r in r0..r1 {
        let s = &src[r * src_stride + c0..r * src_stride + c1];
        for (d, v) in tmp[r * S + c0..r * S + c1].iter_mut().zip(s) {
            *d = v.to_u16();
        }
    }
}

/// Pre-CDEF source and destination of one plane of a [`CdefUnit`].
pub struct CdefPlane<'a, T> {
    /// Destination; the unit's top-left sample is `dst[dst_base]`.
    pub dst: &'a mut [T],
    pub dst_stride: usize,
    pub dst_base: usize,
    /// Pre-CDEF samples of the unit with a border of [`CDEF_BORDER`]
    /// samples on all sides; the unit's top-left sample is
    /// `src[src_base]`. Border samples outside the frame (see
    /// `plane_w` / `plane_h`) may hold any value. At least
    /// [`CDEF_SRC_SLACK`] samples must follow the bottom right border
    /// sample of every block.
    pub src: &'a [T],
    pub src_stride: usize,
    pub src_base: usize,
    /// Plane position of the unit's top-left sample, and the size of the
    /// plane area inside the frame (spec `is_inside_filter_region`), which
    /// gives the [`CDEF_HAVE_LEFT`] etc. flags of each block.
    pub x0: usize,
    pub y0: usize,
    pub plane_w: usize,
    pub plane_h: usize,
}

/// Samples that may be read after the last border sample of a block in
/// [`CdefPlane::src`] (vector loads of whole rows).
pub const CDEF_SRC_SLACK: usize = 16;

/// One CDEF unit (spec 7.15: the 64x64 luma samples sharing one
/// `cdef_idx`) of up to three planes, see [`cdef_unit`].
pub struct CdefUnit<'a, T> {
    /// Luma (needed when luma is filtered or a primary strength is used,
    /// for the direction search) and chroma (needed when chroma is
    /// filtered).
    pub planes: [Option<CdefPlane<'a, T>>; 3],
    /// Bit `c` of `noskip[r]`: the 8x8 luma block of row `r`, column `c`
    /// of the unit is filtered.
    pub noskip: [u8; 8],
    /// Spec 7.15.1 strengths (after `<< coeffShift`), and the luma damping
    /// (chroma uses `damping - 1`).
    pub y_pri: i32,
    pub y_sec: i32,
    pub uv_pri: i32,
    pub uv_sec: i32,
    pub damping: i32,
    pub ss_x: usize,
    pub ss_y: usize,
    pub bd: u32,
}

/// Block kernels of [`cdef_unit`].
pub trait CdefKernels<T: Pixel> {
    /// [`cdef_direction`].
    fn direction(src: &[T], stride: usize, bd: u32) -> (usize, i32);

    /// [`cdef_filter`] of a block whose source holds pixels: `src[0]` is
    /// the top-left border sample (the block's top-left sample is at
    /// `CDEF_BORDER * (src_stride + 1)`), `edges` tells which borders are
    /// inside the frame (other border samples may hold any value).
    #[allow(clippy::too_many_arguments)]
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
    );
}

/// Spec 7.15.1 adjustment of the luma primary strength by the variance.
#[inline(always)]
fn adjust_strength(pri: i32, var: i32) -> i32 {
    if var == 0 {
        return 0;
    }
    let i = if (var >> 6) != 0 { floor_log2((var >> 6) as u32).min(12) as i32 } else { 0 };
    (pri * (4 + i) + 8) >> 4
}

/// Filters one block of `p` at unit offset `(bx, by)`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn unit_block<T: Pixel, K: CdefKernels<T>>(
    p: &mut CdefPlane<'_, T>,
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    pri: i32,
    sec: i32,
    damping: i32,
    dir: usize,
    bd: u32,
) {
    let (x, y) = (p.x0 + bx, p.y0 + by);
    let edges = (x > 0) as u8 * CDEF_HAVE_LEFT
        | (x + w < p.plane_w) as u8 * CDEF_HAVE_RIGHT
        | (y > 0) as u8 * CDEF_HAVE_TOP
        | (y + h < p.plane_h) as u8 * CDEF_HAVE_BOTTOM;
    let s = p.src_base + by * p.src_stride + bx - CDEF_BORDER * (p.src_stride + 1);
    let d = p.dst_base + by * p.dst_stride + bx;
    K::filter(
        &mut p.dst[d..],
        p.dst_stride,
        &p.src[s..],
        p.src_stride,
        w,
        h,
        pri,
        sec,
        damping,
        dir,
        edges,
        bd,
    );
}

/// Spec 7.15 / 7.15.1 for the 8x8 blocks of one [`CdefUnit`]: direction
/// search, strength adjustment and the filter of every plane.
pub fn cdef_unit<T: Pixel, K: CdefKernels<T>>(u: &mut CdefUnit<'_, T>) {
    let (y_pri, y_sec, uv_pri, uv_sec) = (u.y_pri, u.y_sec, u.uv_pri, u.uv_sec);
    let (damping, bd, ss_x, ss_y) = (u.damping, u.bd, u.ss_x, u.ss_y);
    let need_dir = y_pri != 0 || uv_pri != 0;
    let y_on = y_pri != 0 || y_sec != 0;
    let uv_on = uv_pri != 0 || uv_sec != 0;
    let uv_dirs = &CDEF_UV_DIR[ss_x & 1][ss_y & 1];
    let (uv_w, uv_h) = (8 >> ss_x, 8 >> ss_y);
    let noskip = u.noskip;
    let [py, pu, pv] = &mut u.planes;
    for (r, &row) in noskip.iter().enumerate() {
        let mut mask = row;
        while mask != 0 {
            let c = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let (bx, by) = (8 * c, 8 * r);
            let (y_dir, var) = match py {
                Some(p) if need_dir => K::direction(
                    &p.src[p.src_base + by * p.src_stride + bx..],
                    p.src_stride,
                    bd,
                ),
                _ => (0, 0),
            };
            if y_on {
                if let Some(p) = py.as_mut() {
                    let pri = if y_pri != 0 { adjust_strength(y_pri, var) } else { 0 };
                    if pri != 0 || y_sec != 0 {
                        let dir = if y_pri == 0 { 0 } else { y_dir };
                        unit_block::<T, K>(p, bx, by, 8, 8, pri, y_sec, damping, dir, bd);
                    }
                }
            }
            if uv_on {
                let dir = if uv_pri == 0 { 0 } else { uv_dirs[y_dir] };
                let (cx, cy) = (bx >> ss_x, by >> ss_y);
                for p in [&mut *pu, &mut *pv].into_iter().flatten() {
                    unit_block::<T, K>(p, cx, cy, uv_w, uv_h, uv_pri, uv_sec, damping - 1, dir, bd);
                }
            }
        }
    }
}

/// Portable [`CdefKernels`].
pub struct PortableCdef;

impl<T: Pixel> CdefKernels<T> for PortableCdef {
    #[inline(always)]
    fn direction(src: &[T], stride: usize, bd: u32) -> (usize, i32) {
        cdef_direction(src, stride, bd)
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
        if edges == CDEF_HAVE_ALL {
            cdef_filter_src(
                dst, dst_stride, src, src_stride, w, h, pri, sec, damping, dir, bd,
            );
        } else {
            let mut tmp = [0u16; CDEF_PAD_STRIDE * CDEF_PAD_STRIDE];
            pad_block(&mut tmp, src, src_stride, w, h, edges);
            cdef_filter(
                dst,
                dst_stride,
                &tmp,
                CDEF_PAD_STRIDE,
                w,
                h,
                pri,
                sec,
                damping,
                dir,
                bd,
            );
        }
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
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

    /// Spec 7.15.2 transcription on an 8x8 block of i32.
    fn spec_direction(blk: &[[i32; 8]; 8], bd: u32) -> (usize, i32) {
        let mut cost = [0i64; 8];
        let mut partial = [[0i64; 15]; 8];
        let mut best_cost = 0;
        let mut y_dir = 0;
        for i in 0..8 {
            for j in 0..8 {
                let x = ((blk[i][j] >> (bd - 8)) - 128) as i64;
                partial[0][i + j] += x;
                partial[1][i + j / 2] += x;
                partial[2][i] += x;
                partial[3][3 + i - j / 2] += x;
                partial[4][7 + i - j] += x;
                partial[5][3 - i / 2 + j] += x;
                partial[6][j] += x;
                partial[7][i / 2 + j] += x;
            }
        }
        let dt: [i64; 9] = [0, 840, 420, 280, 210, 168, 140, 120, 105];
        for i in 0..8 {
            cost[2] += partial[2][i] * partial[2][i];
            cost[6] += partial[6][i] * partial[6][i];
        }
        cost[2] *= dt[8];
        cost[6] *= dt[8];
        for i in 0..7 {
            cost[0] += (partial[0][i] * partial[0][i] + partial[0][14 - i] * partial[0][14 - i])
                * dt[i + 1];
            cost[4] += (partial[4][i] * partial[4][i] + partial[4][14 - i] * partial[4][14 - i])
                * dt[i + 1];
        }
        cost[0] += partial[0][7] * partial[0][7] * dt[8];
        cost[4] += partial[4][7] * partial[4][7] * dt[8];
        let mut i = 1;
        while i < 8 {
            for j in 0..5 {
                cost[i] += partial[i][3 + j] * partial[i][3 + j];
            }
            cost[i] *= dt[8];
            for j in 0..3 {
                cost[i] += (partial[i][j] * partial[i][j]
                    + partial[i][10 - j] * partial[i][10 - j])
                    * dt[2 * j + 2];
            }
            i += 2;
        }
        for (i, &c) in cost.iter().enumerate() {
            if c > best_cost {
                best_cost = c;
                y_dir = i;
            }
        }
        (y_dir, ((best_cost - cost[(y_dir + 4) & 7]) >> 10) as i32)
    }

    fn gen_block(rng: &mut Rng, bd: u32) -> [[i32; 8]; 8] {
        let maxv = (1 << bd) - 1;
        let mut b = [[0; 8]; 8];
        match rng.range(0, 3) {
            0 => {
                for row in b.iter_mut() {
                    for v in row.iter_mut() {
                        *v = rng.range(0, maxv);
                    }
                }
            }
            1 => {
                // Oriented stripes.
                let (a, c) = (rng.range(-4, 4), rng.range(-4, 4));
                let amp = rng.range(0, maxv / 2);
                let base = rng.range(0, maxv);
                for (i, row) in b.iter_mut().enumerate() {
                    for (j, v) in row.iter_mut().enumerate() {
                        let ph = (a * i as i32 + c * j as i32).rem_euclid(4);
                        *v = (base + if ph < 2 { amp } else { -amp }).clamp(0, maxv);
                    }
                }
            }
            2 => {
                // Extremes.
                for row in b.iter_mut() {
                    for v in row.iter_mut() {
                        *v = if rng.range(0, 1) == 0 { 0 } else { maxv };
                    }
                }
            }
            _ => {
                let base = rng.range(0, maxv);
                for row in b.iter_mut() {
                    for v in row.iter_mut() {
                        *v = (base + rng.range(-8, 8)).clamp(0, maxv);
                    }
                }
            }
        }
        b
    }

    fn direction_random<T: Pixel>(bd: u32, seed: u64) {
        let mut rng = Rng(seed);
        let mut dirs = [0usize; 8];
        for _ in 0..20000 {
            let blk = gen_block(&mut rng, bd);
            let stride = 11;
            let mut buf = vec![T::default(); stride * 8];
            for i in 0..8 {
                for j in 0..8 {
                    buf[i * stride + j] = T::from_i32(blk[i][j]);
                }
            }
            let got = cdef_direction(&buf, stride, bd);
            assert_eq!(got, spec_direction(&blk, bd), "bd {bd} {blk:?}");
            dirs[got.0] += 1;
        }
        assert!(dirs.iter().all(|&d| d > 100), "{dirs:?}");
    }

    #[test]
    fn direction_vs_spec() {
        direction_random::<u8>(8, 1);
        direction_random::<u16>(10, 2);
        direction_random::<u16>(12, 3);
    }

    /// Spec 7.15.3 transcription. `frame` / `avail` are (h + 4) x (w + 4)
    /// with the block at offset (2, 2).
    #[allow(clippy::too_many_arguments)]
    fn spec_filter(
        frame: &[Vec<i32>],
        avail: &[Vec<bool>],
        w: usize,
        h: usize,
        pri_str: i32,
        sec_str: i32,
        damping: i32,
        dir: usize,
        bd: u32,
    ) -> Vec<Vec<i32>> {
        let coeff_shift = bd - 8;
        let constrain = |diff: i32, threshold: i32, damping: i32| -> i32 {
            if threshold == 0 {
                return 0;
            }
            let damping_adj = (damping - floor_log2(threshold as u32) as i32).max(0);
            let sign = if diff < 0 { -1 } else { 1 };
            sign * (threshold - (diff.abs() >> damping_adj)).clamp(0, diff.abs())
        };
        let get_at = |i: usize, j: usize, dir: usize, k: usize, sign: isize| -> (bool, i32) {
            let y = (2 + i) as isize + sign * CDEF_DIRECTIONS[dir][k][0];
            let x = (2 + j) as isize + sign * CDEF_DIRECTIONS[dir][k][1];
            if avail[y as usize][x as usize] {
                (true, frame[y as usize][x as usize])
            } else {
                (false, 0)
            }
        };
        let mut out = vec![vec![0; w]; h];
        for i in 0..h {
            for j in 0..w {
                let mut sum = 0;
                let x = frame[2 + i][2 + j];
                let mut max = x;
                let mut min = x;
                for k in 0..2 {
                    for sign in [-1isize, 1] {
                        let (av, p) = get_at(i, j, dir, k, sign);
                        if av {
                            sum += CDEF_PRI_TAPS[((pri_str >> coeff_shift) & 1) as usize][k]
                                * constrain(p - x, pri_str, damping);
                            max = max.max(p);
                            min = min.min(p);
                        }
                        for dir_off in [-2isize, 2] {
                            let d = ((dir as isize + dir_off) & 7) as usize;
                            let (av, s) = get_at(i, j, d, k, sign);
                            if av {
                                sum += CDEF_SEC_TAPS[((pri_str >> coeff_shift) & 1) as usize][k]
                                    * constrain(s - x, sec_str, damping);
                                max = max.max(s);
                                min = min.min(s);
                            }
                        }
                    }
                }
                out[i][j] = (x + ((8 + sum - (sum < 0) as i32) >> 4)).clamp(min, max);
            }
        }
        out
    }

    /// `extreme`: also use non-conformant strengths / damping (exercising
    /// the 16-bit fast path limits and the generic fallback).
    fn filter_random<T: Pixel>(bd: u32, seed: u64, extreme: bool) {
        let mut rng = Rng(seed);
        let maxv = (1 << bd) - 1;
        let coeff_shift = bd - 8;
        let sizes = [(8usize, 8usize), (4, 8), (8, 4), (4, 4)];
        let mut changed = 0usize;
        for iter in 0..20000 {
            let (w, h) = sizes[rng.range(0, 3) as usize];
            let (fw, fh) = (w + 4, h + 4);
            // Content: noise around a base, sometimes with an edge.
            let base = rng.range(0, maxv);
            let noise = [0, 2, 10, 50, maxv][rng.range(0, 4) as usize] << coeff_shift.min(2);
            let edge = if rng.range(0, 1) == 0 { 0 } else { rng.range(-maxv, maxv) };
            let ex = rng.range(0, fw as i32) as usize;
            let mut frame = vec![vec![0; fw]; fh];
            for (y, row) in frame.iter_mut().enumerate() {
                for (x, v) in row.iter_mut().enumerate() {
                    let e = if x + y / 2 >= ex { edge } else { 0 };
                    *v = (base + e + rng.range(-noise, noise)).clamp(0, maxv);
                }
            }
            // Availability: frame-boundary like rows / columns plus, for
            // generality, random isolated holes in the border.
            let (top, left) = (rng.range(0, 3) == 0, rng.range(0, 3) == 0);
            let (bottom, right) = (rng.range(0, 3) == 0, rng.range(0, 3) == 0);
            let holes = rng.range(0, 2) == 0;
            let mut avail = vec![vec![true; fw]; fh];
            for (y, row) in avail.iter_mut().enumerate() {
                for (x, a) in row.iter_mut().enumerate() {
                    let inside = (2..2 + h).contains(&y) && (2..2 + w).contains(&x);
                    if inside {
                        continue;
                    }
                    if (top && y < 2)
                        || (bottom && y >= 2 + h)
                        || (left && x < 2)
                        || (right && x >= 2 + w)
                        || (holes && rng.range(0, 3) == 0)
                    {
                        *a = false;
                    }
                }
            }
            // Strengths as derived by 7.15.1.
            let chroma = rng.range(0, 1) == 1;
            let cdef_damping = rng.range(3, 6);
            let mut pri = rng.range(0, 15) << coeff_shift;
            let sec_raw = rng.range(0, 3);
            let sec = (if sec_raw == 3 { 4 } else { sec_raw }) << coeff_shift;
            let damping = if chroma {
                cdef_damping + coeff_shift as i32 - 1
            } else {
                let var = [0, rng.range(0, 63), rng.range(64, 1 << 20)][rng.range(0, 2) as usize];
                let var_str =
                    if (var >> 6) != 0 { floor_log2((var >> 6) as u32).min(12) } else { 0 };
                pri = if var != 0 { (pri * (4 + var_str as i32) + 8) >> 4 } else { 0 };
                cdef_damping + coeff_shift as i32
            };
            let dir = rng.range(0, 7) as usize;
            let (pri, sec, damping) = if extreme {
                match rng.range(0, 2) {
                    0 => (rng.range(0, 1023), rng.range(0, 1023), rng.range(2, 13)),
                    1 => (rng.range(0, 1100), rng.range(0, 1100), rng.range(2, 13)),
                    _ => (rng.range(0, 5000), rng.range(0, 5000), rng.range(2, 20)),
                }
            } else {
                (pri, sec, damping)
            };

            let mut src = vec![0u16; fw * fh];
            for y in 0..fh {
                for x in 0..fw {
                    src[y * fw + x] =
                        if avail[y][x] { frame[y][x] as u16 } else { CDEF_VERY_LARGE };
                }
            }
            let dst_stride = 9;
            let mut dst = vec![T::default(); dst_stride * h];
            cdef_filter(
                &mut dst, dst_stride, &src, fw, w, h, pri, sec, damping, dir, bd,
            );
            let want = spec_filter(&frame, &avail, w, h, pri, sec, damping, dir, bd);
            for i in 0..h {
                for j in 0..w {
                    assert_eq!(
                        dst[i * dst_stride + j].to_i32(),
                        want[i][j],
                        "iter {iter} bd {bd} {w}x{h} ({i},{j}) pri {pri} sec {sec} \
                         damping {damping} dir {dir}"
                    );
                    changed += (want[i][j] != frame[2 + i][2 + j]) as usize;
                }
            }
        }
        // Make sure the filter actually did something on many samples.
        assert!(changed > 100_000, "changed {changed}");
    }

    #[test]
    fn filter_vs_spec_8bit() {
        filter_random::<u8>(8, 11, false);
        filter_random::<u16>(8, 12, false);
    }

    #[test]
    fn filter_vs_spec_10bit() {
        filter_random::<u16>(10, 13, false);
    }

    #[test]
    fn filter_vs_spec_12bit() {
        filter_random::<u16>(12, 14, false);
    }

    #[test]
    fn filter_vs_spec_extreme_params() {
        filter_random::<u8>(8, 15, true);
        filter_random::<u16>(10, 16, true);
        filter_random::<u16>(12, 17, true);
    }

    /// [`CdefKernels::filter`] of [`PortableCdef`] (pixel source, edge
    /// flags) against the spec, with garbage in the unavailable border.
    fn kernel_random<T: Pixel>(bd: u32, seed: u64) {
        let mut rng = Rng(seed);
        let maxv = (1 << bd) - 1;
        let cs = bd - 8;
        let sizes = [(8usize, 8usize), (4, 8), (8, 4), (4, 4)];
        for iter in 0..20000 {
            let (w, h) = sizes[rng.range(0, 3) as usize];
            let (fw, fh) = (w + 4, h + 4);
            let base = rng.range(0, maxv);
            let noise = [0, 2, 10, 50, maxv][rng.range(0, 4) as usize] << cs.min(2);
            let edges = if rng.range(0, 1) == 0 { CDEF_HAVE_ALL } else { rng.range(0, 15) as u8 };
            let mut frame = vec![vec![0; fw]; fh];
            let mut avail = vec![vec![true; fw]; fh];
            for y in 0..fh {
                for x in 0..fw {
                    frame[y][x] = (base + rng.range(-noise, noise)).clamp(0, maxv);
                    avail[y][x] = (y >= 2 || edges & CDEF_HAVE_TOP != 0)
                        && (y < h + 2 || edges & CDEF_HAVE_BOTTOM != 0)
                        && (x >= 2 || edges & CDEF_HAVE_LEFT != 0)
                        && (x < w + 2 || edges & CDEF_HAVE_RIGHT != 0);
                }
            }
            let (pri, sec, damping) = match rng.range(0, 3) {
                0 => (rng.range(0, 15) << cs, 0, rng.range(2, 6) + cs as i32),
                1 => (
                    0,
                    [1, 2, 4][rng.range(0, 2) as usize] << cs,
                    rng.range(2, 6) + cs as i32,
                ),
                2 => (
                    rng.range(0, 15 << cs),
                    [0, 1, 2, 4][rng.range(0, 3) as usize] << cs,
                    rng.range(2, 6) + cs as i32,
                ),
                _ => (rng.range(0, 1100), rng.range(0, 1100), rng.range(2, 15)),
            };
            if pri == 0 && sec == 0 {
                continue;
            }
            let dir = rng.range(0, 7) as usize;
            let stride = fw + rng.range(0, 5) as usize;
            let mut src = vec![T::default(); stride * fh + CDEF_SRC_SLACK];
            for (i, v) in src.iter_mut().enumerate() {
                let (y, x) = (i / stride, i % stride);
                *v = T::from_i32(if y < fh && x < fw && avail[y][x] {
                    frame[y][x]
                } else {
                    rng.range(0, maxv)
                });
            }
            let dst_stride = 9;
            let mut dst = vec![T::default(); dst_stride * h];
            PortableCdef::filter(
                &mut dst, dst_stride, &src, stride, w, h, pri, sec, damping, dir, edges, bd,
            );
            let want = spec_filter(&frame, &avail, w, h, pri, sec, damping, dir, bd);
            for i in 0..h {
                for j in 0..w {
                    assert_eq!(
                        dst[i * dst_stride + j].to_i32(),
                        want[i][j],
                        "iter {iter} bd {bd} {w}x{h} ({i},{j}) pri {pri} sec {sec} damping {damping} dir {dir} edges {edges}"
                    );
                }
            }
        }
    }

    #[test]
    fn kernel_vs_spec() {
        kernel_random::<u8>(8, 41);
        kernel_random::<u16>(10, 42);
        kernel_random::<u16>(12, 43);
    }

    /// The sentinel argument for SIMD versions: with spec-conformant
    /// strengths the constrain() of an unavailable tap is always 0.
    #[test]
    fn sentinel_constrain_is_zero() {
        for bd in [8u32, 10, 12] {
            let cs = bd - 8;
            for x in [0, (1 << bd) - 1] {
                let diff = CDEF_VERY_LARGE as i32 - x;
                for damping in 2..=(6 + cs as i32) {
                    for thr in 1..=(15 << cs) {
                        let adj = damping_adj(thr, damping);
                        assert_eq!(constrain(diff, thr, adj), 0);
                    }
                }
            }
        }
    }
}
