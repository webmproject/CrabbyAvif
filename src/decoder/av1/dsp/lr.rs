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

//! Loop restoration filters: Wiener (spec 7.17.4, 7.17.5) and self guided
//! (spec 7.17.2, 7.17.3).
//!
//! OWNER: lr worker. Contract below must be kept.
//!
//! The caller implements 7.17 / 7.17.1 (unit and stripe iteration) and
//! 7.17.6 (get_source_sample, i.e. stripe boundary handling and frame edge
//! clamping) by preparing a padded source buffer for a rectangle that lies
//! entirely within one restoration unit and one stripe.
//!
//! # Implementation notes
//!
//! Both filters work row by row with small ring buffers so that the stack
//! footprint stays around 20 KiB for the largest rectangle
//! ([`LR_MAX_W`] x [`LR_MAX_H`]) and no heap allocation is needed.
//!
//! * Wiener: the horizontal pass produces one row of the spec's
//!   `intermediate` array at a time into a 7 row ring; the vertical pass
//!   consumes 7 consecutive ring rows.
//! * Self guided: the `A`/`B` arrays of 7.17.3 are produced one row at a
//!   time (2 row ring for the r = 2 pass, which only needs odd rows, and a
//!   3 row ring for the r = 1 pass) and filtered/combined immediately.
//!
//! # Border requirements
//!
//! [`LR_BORDER`] = 3 is sufficient for both filters:
//! * Wiener reads rows `-3..h+3` and columns `-3..w+3`.
//! * The r = 2 box filter computes `A`/`B` for odd rows in `-1..=h` and
//!   columns `-1..=w`, each reading a 5x5 neighbourhood, i.e. rows `-3..h+3`
//!   and columns `-3..w+3`. The r = 1 pass needs only `-2..h+2`.
//!
//! # Row parity
//!
//! The r = 2 self guided pass treats rows differently depending on their
//! parity relative to the top of the filtered block (spec 7.17.3). In the
//! spec every block starts at an even plane row, so the rectangle passed to
//! [`sgr_filter`] must start at an even row of the plane (stripe starts and
//! 4x4/2x2 block rows always are).

use crate::decoder::av1::pixel::{round2, Pixel};

/// Border (in samples) around the rectangle in the padded source buffer.
pub const LR_BORDER: usize = 3;

/// Maximum rectangle width accepted by the filters (largest restoration
/// unit is 256 samples wide, the last unit of a row may be up to 1.5x).
pub const LR_MAX_W: usize = 384;

/// Maximum rectangle height accepted by the filters (one 64 row stripe).
pub const LR_MAX_H: usize = 64;

/// Number of bits used in Wiener filter coefficients.
pub const FILTER_BITS: u32 = 7;
/// Number of bits needed to specify a self guided filter set.
pub const SGRPROJ_PARAMS_BITS: u32 = 4;
/// Precision bits of the self guided projection weights.
pub const SGRPROJ_PRJ_BITS: u32 = 7;
/// Restoration precision bits generated higher than source before
/// projection.
pub const SGRPROJ_RST_BITS: u32 = 4;
/// Precision of the mtable division table.
pub const SGRPROJ_MTABLE_BITS: u32 = 20;
/// Precision of the division by n table.
pub const SGRPROJ_RECIP_BITS: u32 = 12;
/// Internal precision bits for the self guided core.
pub const SGRPROJ_SGR_BITS: u32 = 8;

/// `Sgr_Params[set] = { r0, eps0, r1, eps1 }` (spec 7.17.3).
pub const SGR_PARAMS: [[u32; 4]; 1 << SGRPROJ_PARAMS_BITS] = [
    [2, 12, 1, 4],
    [2, 15, 1, 6],
    [2, 18, 1, 8],
    [2, 21, 1, 9],
    [2, 24, 1, 10],
    [2, 29, 1, 11],
    [2, 36, 1, 12],
    [2, 45, 1, 13],
    [2, 56, 1, 14],
    [2, 68, 1, 15],
    [0, 0, 1, 5],
    [0, 0, 1, 8],
    [0, 0, 1, 11],
    [0, 0, 1, 14],
    [2, 30, 0, 0],
    [2, 75, 0, 0],
];

/// Lower bounds of the coded Wiener taps (spec 5.11.58 semantics).
pub const WIENER_TAPS_MIN: [i32; 3] = [-5, -23, -17];
/// Upper bounds of the coded Wiener taps.
pub const WIENER_TAPS_MAX: [i32; 3] = [10, 8, 46];
/// Subexponential parameter of the coded Wiener taps.
pub const WIENER_TAPS_K: [u32; 3] = [1, 2, 3];
/// Reference values of the Wiener taps at the start of a tile.
pub const WIENER_TAPS_MID: [i32; 3] = [3, -7, 15];
/// Lower bounds of the self guided projection weights.
pub const SGRPROJ_XQD_MIN: [i32; 2] = [-96, -32];
/// Upper bounds of the self guided projection weights.
pub const SGRPROJ_XQD_MAX: [i32; 2] = [31, 95];
/// Reference values of the projection weights at the start of a tile.
pub const SGRPROJ_XQD_MID: [i32; 2] = [-32, 31];

/// Per pass constants of the box filter derived from [`SGR_PARAMS`]:
/// `s = ((1 << SGRPROJ_MTABLE_BITS) + n2e / 2) / n2e` with
/// `n2e = n * n * eps` (0 when the pass is disabled).
pub const SGR_SCALE: [[u32; 2]; 1 << SGRPROJ_PARAMS_BITS] = {
    let mut t = [[0u32; 2]; 1 << SGRPROJ_PARAMS_BITS];
    let mut set = 0;
    while set < t.len() {
        let mut pass = 0;
        while pass < 2 {
            let r = SGR_PARAMS[set][pass * 2];
            let eps = SGR_PARAMS[set][pass * 2 + 1];
            if r != 0 {
                let n = (2 * r + 1) * (2 * r + 1);
                let n2e = n * n * eps;
                t[set][pass] = ((1 << SGRPROJ_MTABLE_BITS) + n2e / 2) / n2e;
            }
            pass += 1;
        }
        set += 1;
    }
    t
};

/// `a2` of spec 7.17.3 as a function of `z` (clamped to 255):
/// 1 for z = 0, 256 for z >= 255, otherwise
/// `((z << SGRPROJ_SGR_BITS) + z / 2) / (z + 1)`.
pub const SGR_A2: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut z = 0;
    while z < 256 {
        t[z as usize] = if z >= 255 {
            256
        } else if z == 0 {
            1
        } else {
            ((z << SGRPROJ_SGR_BITS) + z / 2) / (z + 1)
        };
        z += 1;
    }
    t
};

/// `oneOverN` of spec 7.17.3 for a box of radius `r` (1 or 2).
#[inline(always)]
pub(crate) const fn one_over_n(r: usize) -> u32 {
    let n = ((2 * r + 1) * (2 * r + 1)) as u32;
    ((1 << SGRPROJ_RECIP_BITS) + n / 2) / n
}

/// Spec 7.17.5: expands the 3 coded taps into the symmetric 7-tap filter
/// with unit DC gain (taps sum to 128).
#[inline]
pub fn wiener_coefficients(coeff: &[i32; 3]) -> [i32; 7] {
    let mut filter = [0i32; 7];
    filter[3] = 128;
    for (i, &c) in coeff.iter().enumerate() {
        filter[i] = c;
        filter[6 - i] = c;
        filter[3] -= 2 * c;
    }
    filter
}

/// `(InterRound0, InterRound1)` of spec 7.11.3.2 with `isCompound = 0`.
#[inline(always)]
pub(crate) fn wiener_rounding(bd: u32) -> (u32, u32) {
    if bd == 12 {
        (5, 9)
    } else {
        (3, 11)
    }
}

#[inline(always)]
pub(crate) fn check_rect(w: usize, h: usize) {
    assert!(
        (1..=LR_MAX_W).contains(&w) && (1..=LR_MAX_H).contains(&h),
        "lr: invalid rectangle {w}x{h}"
    );
}

/// Wiener filter of a `w x h` rectangle.
///
/// * `src`: samples equal to `get_source_sample(plane, x, y)` for the
///   rectangle extended by [`LR_BORDER`] on every side; the rectangle's
///   top-left sample is at `src[LR_BORDER * src_stride + LR_BORDER]`.
/// * `coeffs`: `LrWiener[plane][unitRow][unitCol]`, i.e. `coeffs[0]` holds
///   the 3 coded vertical taps and `coeffs[1]` the 3 horizontal taps (spec
///   7.17.5 derives the full 7-tap filters from them).
/// * `dst`: output, `dst[0]` = top-left sample of the rectangle.
///
/// `w` must be in `1..=LR_MAX_W` and `h` in `1..=LR_MAX_H`.
#[allow(clippy::too_many_arguments)]
pub fn wiener_filter<T: Pixel>(
    dst: &mut [T],
    dst_stride: usize,
    src: &[T],
    src_stride: usize,
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

    // Horizontal pass for intermediate row `r` (padded source row `r`).
    let hpass = |r: usize, out: &mut [i32; LR_MAX_W]| {
        let row = &src[r * src_stride..r * src_stride + w + 2 * LR_BORDER];
        for (c, o) in out[..w].iter_mut().enumerate() {
            let taps = &row[c..c + 7];
            let s: i32 = taps
                .iter()
                .zip(hfilter.iter())
                .map(|(p, &f)| f * p.to_i32())
                .sum();
            *o = round2(s, round0).clamp(-offset, limit - offset);
        }
    };

    // Ring of 7 intermediate rows; row r lives in slot r % 7.
    let mut ring = [[0i32; LR_MAX_W]; 7];
    for (r, slot) in ring.iter_mut().enumerate().take(6) {
        hpass(r, slot);
    }
    for r in 0..h {
        hpass(r + 6, &mut ring[(r + 6) % 7]);
        let rows: [&[i32; LR_MAX_W]; 7] = std::array::from_fn(|t| &ring[(r + t) % 7]);
        let out = &mut dst[r * dst_stride..r * dst_stride + w];
        for (c, o) in out.iter_mut().enumerate() {
            let s: i32 = (0..7).map(|t| vfilter[t] * rows[t][c]).sum();
            *o = T::from_i32(round2(s, round1).clamp(0, max_px));
        }
    }
}

/// One row of the `A` (`a`) and `B` (`b`) arrays of spec 7.17.3, covering
/// rectangle columns `-1..=w` (stored at index `j + 1`).
struct AbRow {
    a: [u32; LR_MAX_W + 2],
    b: [i32; LR_MAX_W + 2],
}

impl AbRow {
    const fn new() -> Self {
        Self {
            a: [0; LR_MAX_W + 2],
            b: [0; LR_MAX_W + 2],
        }
    }
}

/// Computes row `i` (rectangle row, `-1..=h`) of the `A`/`B` arrays of the
/// box filter with radius `r` and scale `s` (spec 7.17.3).
#[allow(clippy::too_many_arguments)]
fn box_ab_row<T: Pixel>(
    src: &[T],
    src_stride: usize,
    i: isize,
    w: usize,
    r: usize,
    s: u32,
    bd: u32,
    out: &mut AbRow,
) {
    const COLS: usize = LR_MAX_W + 2 + 4;
    let n = ((2 * r + 1) * (2 * r + 1)) as u32;
    let one_over_n = one_over_n(r) as u64;
    let ncols = w + 2 + 2 * r;
    // Leftmost padded column used: rectangle column -1 - r.
    let col0 = LR_BORDER - 1 - r;

    // Vertical sums of c and c * c over the 2r+1 rows around row i.
    let mut vsum = [0u32; COLS];
    let mut vsq = [0u32; COLS];
    for dy in -(r as isize)..=(r as isize) {
        let pr = (i + dy + LR_BORDER as isize) as usize;
        let row = &src[pr * src_stride + col0..pr * src_stride + col0 + ncols];
        for ((sum, sq), p) in vsum.iter_mut().zip(vsq.iter_mut()).zip(row) {
            let c = p.to_i32() as u32;
            *sum += c;
            *sq += c * c;
        }
    }

    // Horizontal sliding window of width 2r+1.
    let d = 2 * r + 1;
    let mut bsum: u32 = vsum[..d - 1].iter().sum();
    let mut asum: u32 = vsq[..d - 1].iter().sum();
    let sh_a = 2 * (bd - 8);
    let sh_b = bd - 8;
    for j in 0..w + 2 {
        bsum += vsum[j + d - 1];
        asum += vsq[j + d - 1];

        let a = (asum + ((1 << sh_a) >> 1)) >> sh_a;
        let dd = (bsum + ((1 << sh_b) >> 1)) >> sh_b;
        let p = (a * n).saturating_sub(dd * dd);
        let z = (p as u64 * s as u64 + (1 << (SGRPROJ_MTABLE_BITS - 1))) >> SGRPROJ_MTABLE_BITS;
        let a2 = SGR_A2[z.min(255) as usize];
        let b2 = ((1 << SGRPROJ_SGR_BITS) - a2) as u64 * bsum as u64 * one_over_n;
        out.a[j] = a2;
        out.b[j] = ((b2 + (1 << (SGRPROJ_RECIP_BITS - 1))) >> SGRPROJ_RECIP_BITS) as i32;

        bsum -= vsum[j];
        asum -= vsq[j];
    }
}

/// Output of the r = 2 box filter (pass 0) for rectangle row `i`.
///
/// `prev`/`next` hold `A`/`B` rows `i - 1` and `i + 1` (even `i`), or
/// `next` holds row `i` (odd `i`).
fn box_filter_r2_row<T: Pixel>(
    px: &[T],
    prev: &AbRow,
    next: &AbRow,
    odd: bool,
    out: &mut [i32; LR_MAX_W],
) {
    let w = px.len();
    if odd {
        let shift = SGRPROJ_SGR_BITS + 4 - SGRPROJ_RST_BITS;
        for (j, o) in out[..w].iter_mut().enumerate() {
            let a = 5 * (next.a[j] + next.a[j + 2]) + 6 * next.a[j + 1];
            let b = 5 * (next.b[j] + next.b[j + 2]) + 6 * next.b[j + 1];
            *o = round2(a as i32 * px[j].to_i32() + b, shift);
        }
    } else {
        let shift = SGRPROJ_SGR_BITS + 5 - SGRPROJ_RST_BITS;
        for (j, o) in out[..w].iter_mut().enumerate() {
            let a = 5 * (prev.a[j] + prev.a[j + 2] + next.a[j] + next.a[j + 2])
                + 6 * (prev.a[j + 1] + next.a[j + 1]);
            let b = 5 * (prev.b[j] + prev.b[j + 2] + next.b[j] + next.b[j + 2])
                + 6 * (prev.b[j + 1] + next.b[j + 1]);
            *o = round2(a as i32 * px[j].to_i32() + b, shift);
        }
    }
}

/// Output of the r = 1 box filter (pass 1) for one rectangle row, given
/// `A`/`B` rows above (`t`), at (`m`) and below (`d`) it.
fn box_filter_r1_row<T: Pixel>(
    px: &[T],
    t: &AbRow,
    m: &AbRow,
    d: &AbRow,
    out: &mut [i32; LR_MAX_W],
) {
    let w = px.len();
    let shift = SGRPROJ_SGR_BITS + 5 - SGRPROJ_RST_BITS;
    for (j, o) in out[..w].iter_mut().enumerate() {
        let a = 3 * (t.a[j] + t.a[j + 2] + d.a[j] + d.a[j + 2])
            + 4 * (t.a[j + 1] + d.a[j + 1] + m.a[j] + m.a[j + 2] + m.a[j + 1]);
        let b = 3 * (t.b[j] + t.b[j + 2] + d.b[j] + d.b[j + 2])
            + 4 * (t.b[j + 1] + d.b[j + 1] + m.b[j] + m.b[j + 2] + m.b[j + 1]);
        *o = round2(a as i32 * px[j].to_i32() + b, shift);
    }
}

/// Self guided filter of a `w x h` rectangle (same buffer conventions as
/// [`wiener_filter`]). `set` is `LrSgrSet` and `xqd` is `LrSgrXqd` for the
/// unit.
///
/// `w` must be in `1..=LR_MAX_W` and `h` in `1..=LR_MAX_H`; the rectangle
/// must start at an even row of the plane (see the module documentation).
#[allow(clippy::too_many_arguments)]
pub fn sgr_filter<T: Pixel>(
    dst: &mut [T],
    dst_stride: usize,
    src: &[T],
    src_stride: usize,
    w: usize,
    h: usize,
    set: usize,
    xqd: [i32; 2],
    bd: u32,
) {
    check_rect(w, h);
    let r0 = SGR_PARAMS[set][0] as usize;
    let r1 = SGR_PARAMS[set][2] as usize;
    let [s0, s1] = SGR_SCALE[set];
    let w0 = xqd[0];
    let w1 = xqd[1];
    let w2 = (1 << SGRPROJ_PRJ_BITS) - w0 - w1;
    let max_px = (1i32 << bd) - 1;

    // Pass 0 (r = 2): A/B only for odd rows; ab0[cur0] holds the most
    // recent odd row, ab0[cur0 ^ 1] the one before it.
    let mut ab0 = [AbRow::new(), AbRow::new()];
    let mut cur0 = 1;
    // Pass 1 (r = 1): rows -1..=h, row k in slot (k + 1) % 3.
    let mut ab1 = [AbRow::new(), AbRow::new(), AbRow::new()];

    if r0 != 0 {
        box_ab_row(src, src_stride, -1, w, r0, s0, bd, &mut ab0[0]);
        box_ab_row(src, src_stride, 1, w, r0, s0, bd, &mut ab0[1]);
    }
    if r1 != 0 {
        box_ab_row(src, src_stride, -1, w, r1, s1, bd, &mut ab1[0]);
        box_ab_row(src, src_stride, 0, w, r1, s1, bd, &mut ab1[1]);
    }

    let mut flt0 = [0i32; LR_MAX_W];
    let mut flt1 = [0i32; LR_MAX_W];
    for i in 0..h {
        let pr = (i + LR_BORDER) * src_stride + LR_BORDER;
        let px = &src[pr..pr + w];

        if r0 != 0 {
            let odd = i & 1 == 1;
            if !odd && i >= 2 {
                cur0 ^= 1;
                box_ab_row(
                    src,
                    src_stride,
                    i as isize + 1,
                    w,
                    r0,
                    s0,
                    bd,
                    &mut ab0[cur0],
                );
            }
            box_filter_r2_row(px, &ab0[cur0 ^ 1], &ab0[cur0], odd, &mut flt0);
        }
        if r1 != 0 {
            box_ab_row(
                src,
                src_stride,
                i as isize + 1,
                w,
                r1,
                s1,
                bd,
                &mut ab1[(i + 2) % 3],
            );
            box_filter_r1_row(
                px,
                &ab1[i % 3],
                &ab1[(i + 1) % 3],
                &ab1[(i + 2) % 3],
                &mut flt1,
            );
        }

        let out = &mut dst[i * dst_stride..i * dst_stride + w];
        for (j, o) in out.iter_mut().enumerate() {
            let u = px[j].to_i32() << SGRPROJ_RST_BITS;
            let f0 = if r0 != 0 { flt0[j] } else { u };
            let f1 = if r1 != 0 { flt1[j] } else { u };
            let v = w1 * u + w0 * f0 + w2 * f1;
            let s = round2(v, SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS);
            *o = T::from_i32(s.clamp(0, max_px));
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
        fn range(&mut self, lo: i64, hi: i64) -> i64 {
            lo + (self.next() % (hi - lo + 1) as u64) as i64
        }
    }

    fn r2(x: i64, n: u32) -> i64 {
        if n == 0 {
            x
        } else {
            (x + (1 << (n - 1))) >> n
        }
    }

    /// Padded source accessed with spec coordinates relative to the
    /// rectangle (get_source_sample equivalent).
    struct Src<'a, T> {
        buf: &'a [T],
        stride: usize,
    }
    impl<T: Pixel> Src<'_, T> {
        fn get(&self, x: i64, y: i64) -> i64 {
            let b = LR_BORDER as i64;
            assert!(x >= -b && y >= -b, "read outside border");
            self.buf[((y + b) as usize) * self.stride + (x + b) as usize].to_i32() as i64
        }
    }

    /// Direct transcription of spec 7.17.4 / 7.17.5 (64-bit arithmetic).
    fn spec_wiener<T: Pixel>(
        src: &Src<T>,
        w: usize,
        h: usize,
        coeffs: &[[i32; 3]; 2],
        bd: u32,
    ) -> Vec<i64> {
        let coef = |c: &[i32; 3]| {
            let mut filter = [0i64; 7];
            filter[3] = 128;
            for i in 0..3 {
                filter[i] = c[i] as i64;
                filter[6 - i] = c[i] as i64;
                filter[3] -= 2 * c[i] as i64;
            }
            filter
        };
        let vfilter = coef(&coeffs[0]);
        let hfilter = coef(&coeffs[1]);
        let round0 = if bd == 12 { 5 } else { 3 };
        let round1 = if bd == 12 { 9 } else { 11 };
        let offset = 1i64 << (bd + FILTER_BITS - round0 - 1);
        let limit = (1i64 << (bd + 1 + FILTER_BITS - round0)) - 1;
        let mut inter = vec![vec![0i64; w]; h + 6];
        for r in 0..h + 6 {
            for c in 0..w {
                let mut s = 0;
                for t in 0..7 {
                    s += hfilter[t] * src.get(c as i64 + t as i64 - 3, r as i64 - 3);
                }
                let v = r2(s, round0);
                inter[r][c] = v.clamp(-offset, limit - offset);
            }
        }
        let mut out = vec![0; w * h];
        for r in 0..h {
            for c in 0..w {
                let mut s = 0;
                for t in 0..7 {
                    s += vfilter[t] * inter[r + t][c];
                }
                out[r * w + c] = r2(s, round1).clamp(0, (1 << bd) - 1);
            }
        }
        out
    }

    /// Direct transcription of spec 7.17.3 (64-bit arithmetic).
    fn spec_box<T: Pixel>(
        src: &Src<T>,
        w: usize,
        h: usize,
        set: usize,
        pass: usize,
        bd: u32,
    ) -> Option<Vec<Vec<i64>>> {
        let r = SGR_PARAMS[set][pass * 2] as i64;
        if r == 0 {
            return None;
        }
        let eps = SGR_PARAMS[set][pass * 2 + 1] as i64;
        let (wi, hi) = (w as i64, h as i64);
        let mut aa = vec![vec![0i64; w + 2]; h + 2];
        let mut bb = vec![vec![0i64; w + 2]; h + 2];
        let n = (2 * r + 1) * (2 * r + 1);
        let n2e = n * n * eps;
        let s = ((1 << SGRPROJ_MTABLE_BITS) + n2e / 2) / n2e;
        for i in -1..hi + 1 {
            for j in -1..wi + 1 {
                let mut a = 0;
                let mut b = 0;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let c = src.get(j + dx, i + dy);
                        a += c * c;
                        b += c;
                    }
                }
                a = r2(a, 2 * (bd - 8));
                let d = r2(b, bd - 8);
                let p = (a * n - d * d).max(0);
                let z = r2(p * s, SGRPROJ_MTABLE_BITS);
                let a2 = if z >= 255 {
                    256
                } else if z == 0 {
                    1
                } else {
                    ((z << SGRPROJ_SGR_BITS) + (z / 2)) / (z + 1)
                };
                let one_over_n = ((1 << SGRPROJ_RECIP_BITS) + (n / 2)) / n;
                let b2 = ((1 << SGRPROJ_SGR_BITS) - a2) * b * one_over_n;
                aa[(i + 1) as usize][(j + 1) as usize] = a2;
                bb[(i + 1) as usize][(j + 1) as usize] = r2(b2, SGRPROJ_RECIP_BITS);
            }
        }
        let mut f = vec![vec![0i64; w]; h];
        for i in 0..hi {
            let shift = if pass == 0 && (i & 1) == 1 { 4 } else { 5 };
            for j in 0..wi {
                let mut a = 0;
                let mut b = 0;
                for dy in -1..=1i64 {
                    for dx in -1..=1i64 {
                        let weight = if pass == 0 {
                            if ((i + dy) & 1) != 0 {
                                if dx == 0 {
                                    6
                                } else {
                                    5
                                }
                            } else {
                                0
                            }
                        } else if dx == 0 || dy == 0 {
                            4
                        } else {
                            3
                        };
                        a += weight * aa[(i + dy + 1) as usize][(j + dx + 1) as usize];
                        b += weight * bb[(i + dy + 1) as usize][(j + dx + 1) as usize];
                    }
                }
                let v = a * src.get(j, i) + b;
                f[i as usize][j as usize] = r2(v, SGRPROJ_SGR_BITS + shift - SGRPROJ_RST_BITS);
            }
        }
        Some(f)
    }

    /// Direct transcription of spec 7.17.2.
    fn spec_sgr<T: Pixel>(
        src: &Src<T>,
        w: usize,
        h: usize,
        set: usize,
        xqd: [i32; 2],
        bd: u32,
    ) -> Vec<i64> {
        let flt0 = spec_box(src, w, h, set, 0, bd);
        let flt1 = spec_box(src, w, h, set, 1, bd);
        let w0 = xqd[0] as i64;
        let w1 = xqd[1] as i64;
        let w2 = (1 << SGRPROJ_PRJ_BITS) - w0 - w1;
        let mut out = vec![0; w * h];
        for i in 0..h {
            for j in 0..w {
                let u = src.get(j as i64, i as i64) << SGRPROJ_RST_BITS;
                let mut v = w1 * u;
                v += w0 * flt0.as_ref().map_or(u, |f| f[i][j]);
                v += w2 * flt1.as_ref().map_or(u, |f| f[i][j]);
                let s = r2(v, SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS);
                out[i * w + j] = s.clamp(0, (1 << bd) - 1);
            }
        }
        out
    }

    /// Random padded source. `mode` selects the distribution: uniform,
    /// extremes only (maximum variance), or smooth with small noise.
    fn make_src<T: Pixel>(rng: &mut Rng, stride: usize, rows: usize, bd: u32) -> Vec<T> {
        let max = (1i64 << bd) - 1;
        let mode = rng.range(0, 3);
        let base = rng.range(0, max);
        let amp = rng.range(0, 64);
        (0..stride * rows)
            .map(|_| {
                let v = match mode {
                    0 => rng.range(0, max),
                    1 => {
                        if rng.next() & 1 == 0 {
                            0
                        } else {
                            max
                        }
                    }
                    2 => (base + rng.range(-amp, amp)).clamp(0, max),
                    _ => [0, max, base][rng.range(0, 2) as usize],
                };
                T::from_i32(v as i32)
            })
            .collect()
    }

    fn random_size(rng: &mut Rng) -> (usize, usize) {
        match rng.range(0, 4) {
            0 => (LR_MAX_W, LR_MAX_H),
            1 => (rng.range(1, 8) as usize, rng.range(1, 8) as usize),
            _ => (
                rng.range(1, LR_MAX_W as i64) as usize,
                rng.range(1, LR_MAX_H as i64) as usize,
            ),
        }
    }

    fn run_wiener<T: Pixel>(bd: u32, iters: usize, seed: u64) {
        let mut rng = Rng(seed);
        for _ in 0..iters {
            let (w, h) = random_size(&mut rng);
            let stride = w + 2 * LR_BORDER + rng.range(0, 5) as usize;
            let src = make_src::<T>(&mut rng, stride, h + 2 * LR_BORDER, bd);
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
            let dst_stride = w + rng.range(0, 3) as usize;
            let mut dst = vec![T::default(); dst_stride * h];
            wiener_filter(&mut dst, dst_stride, &src, stride, w, h, &coeffs, bd);
            let want = spec_wiener(&Src { buf: &src, stride }, w, h, &coeffs, bd);
            for i in 0..h {
                for j in 0..w {
                    assert_eq!(
                        dst[i * dst_stride + j].to_i32() as i64,
                        want[i * w + j],
                        "wiener bd={bd} {w}x{h} coeffs={coeffs:?} at ({j},{i})"
                    );
                }
            }
        }
    }

    fn run_sgr<T: Pixel>(bd: u32, iters_per_set: usize, seed: u64) {
        let mut rng = Rng(seed);
        for set in 0..SGR_PARAMS.len() {
            for _ in 0..iters_per_set {
                let (w, h) = random_size(&mut rng);
                let stride = w + 2 * LR_BORDER + rng.range(0, 5) as usize;
                let src = make_src::<T>(&mut rng, stride, h + 2 * LR_BORDER, bd);
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
                let dst_stride = w + rng.range(0, 3) as usize;
                let mut dst = vec![T::default(); dst_stride * h];
                sgr_filter(&mut dst, dst_stride, &src, stride, w, h, set, xqd, bd);
                let want = spec_sgr(&Src { buf: &src, stride }, w, h, set, xqd, bd);
                for i in 0..h {
                    for j in 0..w {
                        assert_eq!(
                            dst[i * dst_stride + j].to_i32() as i64,
                            want[i * w + j],
                            "sgr bd={bd} set={set} {w}x{h} xqd={xqd:?} at ({j},{i})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn wiener_coefficients_sum_to_128() {
        let f = wiener_coefficients(&[3, -7, 15]);
        assert_eq!(f, [3, -7, 15, 106, 15, -7, 3]);
        assert_eq!(f.iter().sum::<i32>(), 128);
    }

    #[test]
    fn sgr_tables() {
        // dav1d_sgr_params (src/tables.c).
        const DAV1D_SGR_PARAMS: [[u32; 2]; 16] = [
            [140, 3236],
            [112, 2158],
            [93, 1618],
            [80, 1438],
            [70, 1295],
            [58, 1177],
            [47, 1079],
            [37, 996],
            [30, 925],
            [25, 863],
            [0, 2589],
            [0, 1618],
            [0, 1177],
            [0, 925],
            [56, 0],
            [22, 0],
        ];
        assert_eq!(SGR_SCALE, DAV1D_SGR_PARAMS);
        assert_eq!(one_over_n(2), 164);
        assert_eq!(one_over_n(1), 455);
        // dav1d_sgr_x_by_x[z] == 256 - a2(z): run lengths of each value.
        let runs: [(u32, usize); 22] = [
            (255, 1),
            (128, 1),
            (85, 1),
            (64, 1),
            (51, 1),
            (43, 1),
            (37, 1),
            (32, 1),
            (28, 1),
            (26, 1),
            (23, 1),
            (21, 1),
            (20, 1),
            (18, 1),
            (17, 1),
            (16, 1),
            (15, 1),
            (14, 1),
            (13, 2),
            (12, 2),
            (11, 2),
            (10, 2),
        ];
        let mut z = 0;
        for &(v, n) in runs.iter() {
            for _ in 0..n {
                assert_eq!(256 - SGR_A2[z], v, "z={z}");
                z += 1;
            }
        }
        let tail: [(u32, usize); 10] = [
            (9, 4),
            (8, 4),
            (7, 5),
            (6, 7),
            (5, 10),
            (4, 17),
            (3, 29),
            (2, 68),
            (1, 85),
            (0, 1),
        ];
        for &(v, n) in tail.iter() {
            for _ in 0..n {
                assert_eq!(256 - SGR_A2[z], v, "z={z}");
                z += 1;
            }
        }
        assert_eq!(z, 256);
    }

    #[test]
    fn wiener_matches_spec_8bit() {
        run_wiener::<u8>(8, 150, 0x1234_5678_9abc_def1);
    }

    #[test]
    fn wiener_matches_spec_10bit() {
        run_wiener::<u16>(10, 150, 0x2234_5678_9abc_def1);
    }

    #[test]
    fn wiener_matches_spec_12bit() {
        run_wiener::<u16>(12, 150, 0x3234_5678_9abc_def1);
    }

    #[test]
    fn sgr_matches_spec_8bit() {
        run_sgr::<u8>(8, 8, 0x4234_5678_9abc_def1);
    }

    #[test]
    fn sgr_matches_spec_10bit() {
        run_sgr::<u16>(10, 8, 0x5234_5678_9abc_def1);
    }

    #[test]
    fn sgr_matches_spec_12bit() {
        run_sgr::<u16>(12, 8, 0x6234_5678_9abc_def1);
    }

    /// Worst case for the a/b computation: alternating 0 / max samples
    /// maximise the variance (p * s) and full-scale flat areas maximise b.
    #[test]
    fn sgr_extremes_12bit() {
        let (w, h) = (LR_MAX_W, LR_MAX_H);
        let stride = w + 2 * LR_BORDER;
        let rows = h + 2 * LR_BORDER;
        let max = 4095u16;
        let patterns: [Box<dyn Fn(usize, usize) -> u16>; 4] = [
            Box::new(|_, _| max),
            Box::new(move |x, y| if (x + y) & 1 == 0 { 0 } else { max }),
            Box::new(move |x, _| if x & 1 == 0 { 0 } else { max }),
            Box::new(move |x, y| if (x / 3 + y / 2) & 1 == 0 { 0 } else { max }),
        ];
        for pat in patterns.iter() {
            let src: Vec<u16> = (0..stride * rows)
                .map(|k| pat(k % stride, k / stride))
                .collect();
            for set in 0..SGR_PARAMS.len() {
                for xqd in [[-96, 95], [31, -32], [0, 0], [-96, -32]] {
                    let mut dst = vec![0u16; w * h];
                    sgr_filter(&mut dst, w, &src, stride, w, h, set, xqd, 12);
                    let want = spec_sgr(&Src { buf: &src, stride }, w, h, set, xqd, 12);
                    for (k, (&d, &e)) in dst.iter().zip(want.iter()).enumerate() {
                        assert_eq!(d as i64, e, "set={set} xqd={xqd:?} at {k}");
                    }
                }
            }
        }
    }
}
