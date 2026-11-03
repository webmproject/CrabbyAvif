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

//! Generic 2D inverse transforms with 16-bit intermediates (bd <= 10)
//! for the sizes without a dedicated in-register path in the parent
//! module: everything with a 16, 32 or 64 point dimension.
//!
//! Rows are transformed `RN` at a time (one row per lane) from the
//! coefficient buffer into an `i16` buffer of the row outputs (tiles
//! transposed in registers), then columns `CN` at a time, followed by
//! the residual add.

use std::simd::prelude::*;

use super::{
    add_row16, add_row16_at, col_identity_rnd4, identity, kernels, last_nz_lane, rect2, row_shift,
    select, transpose_e, visit_rows, Clamp, Params, RowLanes, TxLanes, S16,
};
use crate::decoder::av1::dsp::itx::Tx1d;
use crate::decoder::av1::pixel::Pixel;

/// Row pass outputs (16-bit) of `rows` rows, stored by groups of `CN`
/// columns: entry `g * rows + i` holds row `i`, columns `g * CN..(g + 1) *
/// CN`. The `H` (or `rows`) entries of a group are the column pass
/// vectors of those columns, so the column kernels run in place.
/// Entries not written by the row pass are zero.
pub(super) struct Cols<'a, const CN: usize> {
    v: &'a mut [S16<CN>],
    rows: usize,
}

impl<const CN: usize> Cols<'_, CN> {
    /// Stores the `M` columns `j0..j0 + M` of row `i` (`M <= CN`, `j0` a
    /// multiple of `M`).
    #[inline(always)]
    fn put<const M: usize>(&mut self, i: usize, j0: usize, x: Simd<i16, M>) {
        let e = self.v[(j0 / CN) * self.rows + i].0.as_mut_array();
        e[j0 % CN..][..M].copy_from_slice(x.as_array());
    }

    /// Sets all `w` columns of row `i` to `x`.
    #[inline(always)]
    fn fill_row(&mut self, i: usize, w: usize, x: i16) {
        for g in 0..w / CN {
            self.v[g * self.rows + i] = S16(Simd::splat(x));
        }
    }
}

/// 2D transform of a `W x H` block with 16-bit intermediates.
/// `coeffs[i * tw + j]` holds `Dequant[i][j]` (`coeffs.len() == tw *
/// th`). Rows are transformed `RN` at a time, columns `CN` at a time.
/// `LV = W * th / CN` and `LV4 = W * 4 / CN` are the [`Cols`] buffer
/// sizes of the full and the 4-row paths. Leaves `coeffs` all zero.
pub(super) fn transform_add16<
    T: Pixel,
    const BD: u32,
    const W: usize,
    const H: usize,
    const RN: usize,
    const CN: usize,
    const LV: usize,
    const LV4: usize,
>(
    dst: &mut [T],
    stride: usize,
    coeffs: &mut [i32],
    p: &Params,
) {
    debug_assert!(W == p.w && H == p.h && LV * CN == W * p.th && LV4 * CN == W * 4 && BD == p.bd);
    if p.row_kind != Tx1d::Identity && p.eob_rows <= 4 && p.eob_cols <= 4 {
        let mut buf = [S16::<CN>::splat(0); LV4];
        let mut cols = Cols {
            v: &mut buf,
            rows: 4,
        };
        let nz_rows = if BD == 8 {
            row_pass16_sub4::<S16<4>, BD, W, H, CN>(coeffs, &mut cols, p)
        } else {
            row_pass16_sub4::<Simd<i32, 4>, BD, W, H, CN>(coeffs, &mut cols, p)
        };
        if nz_rows != 0 {
            col_pass16::<T, CN, W, H, BD>(dst, stride, &mut cols, p, nz_rows);
        }
        return;
    }
    let mut buf = [S16::<CN>::splat(0); LV];
    let mut cols = Cols {
        v: &mut buf,
        rows: H.min(32),
    };
    let nz_rows = if p.row_kind == Tx1d::Identity {
        row_pass_identity16::<RN, W, H, CN>(coeffs, &mut cols, p)
    } else if BD == 8 && CN == 16 && H >= 16 && W >= 16 && p.eob_rows > 8 {
        // AVX2: 16 rows per 256-bit vector.
        row_pass16_x2::<W, H, CN>(coeffs, &mut cols, p)
    } else if BD == 8 {
        row_pass16::<S16<RN>, BD, RN, W, H, CN>(coeffs, &mut cols, p)
    } else {
        row_pass16::<Simd<i32, RN>, BD, RN, W, H, CN>(coeffs, &mut cols, p)
    };
    if nz_rows == 0 {
        return;
    }
    col_pass16::<T, CN, W, H, BD>(dst, stride, &mut cols, p, nz_rows);
}

/// Fast path of [`row_pass16`] when `eob` guarantees all non-zero
/// coefficients lie in the top-left 4x4 (`eob_rows <= 4 && eob_cols <= 4`).
/// Transforms at most 4 rows on 4-lane vectors using the `nz4` kernel.
fn row_pass16_sub4<
    V: RowLanes<4>,
    const BD: u32,
    const W: usize,
    const H: usize,
    const CN: usize,
>(
    src: &mut [i32],
    buf: &mut Cols<CN>,
    p: &Params,
) -> usize {
    use kernels::*;
    let tw = W.min(32);
    let th = H.min(32);
    debug_assert!(tw == p.tw && th == p.th && p.eob_rows <= 4 && p.eob_cols <= 4);
    let src = &mut src[..tw * th];
    let rows: [Simd<i32, 4>; 4] = std::array::from_fn(|l| Simd::from_slice(&src[l * tw..][..4]));
    let zero = Simd::<i32, 4>::splat(0);
    let col_or = (rows[0] | rows[1]) | (rows[2] | rows[3]);
    if col_or == zero {
        return 0;
    }
    for l in 0..4 {
        src[l * tw..][..4].fill(0);
    }
    let nz_rows = if p.eob_rows == 1 {
        1
    } else {
        rows.iter().rposition(|&r| r != zero).map_or(0, |l| l + 1)
    };
    let row_clamp = Clamp::<V>::bits(BD + 8);
    let col_clamp = Clamp::<V>::bits(16);
    let tile = V::load_tile(rows);
    let rshift = const { row_shift(W, H) };
    let is_rect2 = const { rect2(W, H) };
    if p.row_kind == Tx1d::Dct && (col_or[1] | col_or[2] | col_or[3]) == 0 {
        let mut v = tile[0];
        if is_rect2 {
            v = v.m1(2896);
        }
        let v = col_clamp
            .apply(row_clamp.apply(v.m1(2896)).rnd(rshift))
            .to_i16();
        for (l, &x) in v.as_array()[..nz_rows].iter().enumerate() {
            buf.fill_row(l, W, x);
        }
        return nz_rows;
    }
    let mut t = [V::splat(0); W];
    for (d, mut s) in t[..4].iter_mut().zip(tile) {
        if is_rect2 {
            s = s.m1(2896);
        }
        *d = s;
    }
    match (W, p.row_kind == Tx1d::Adst) {
        (4, false) => idct4_nz4(&mut t, row_clamp),
        (4, true) => super::iadst4(&mut t, row_clamp),
        (8, false) => idct8_nz4(&mut t, row_clamp),
        (8, true) => iadst8_nz4(&mut t, row_clamp),
        (16, false) => idct16_nz4(&mut t, row_clamp),
        (16, true) => iadst16_nz4(&mut t, row_clamp),
        (32, _) => idct32_nz4(&mut t, row_clamp),
        _ => idct64_nz4(&mut t, row_clamp),
    }
    for j0 in (0..W / 4).map(|k| k * 4) {
        let o: [Simd<i16, 4>; 4] =
            std::array::from_fn(|k| col_clamp.apply(t[j0 + k].rnd(rshift)).to_i16());
        for (l, &r) in transpose_e(o).iter().enumerate() {
            buf.put(l, j0, r);
        }
    }
    nz_rows
}

/// Row transforms of width `W`, `N` rows at a time, from `src` (stride
/// `tw`) to `buf` (stride `W`), including the clamp to colClampRange
/// (16 bits). Returns one more than the index of the last non-zero
/// input row (0 if all are zero); output rows from there on are not
/// written. Leaves `src` all zero.
fn row_pass16<
    V: RowLanes<N>,
    const BD: u32,
    const N: usize,
    const W: usize,
    const H: usize,
    const CN: usize,
>(
    src: &mut [i32],
    buf: &mut Cols<CN>,
    p: &Params,
) -> usize {
    let tw = W.min(32);
    let th = H.min(32);
    debug_assert!(tw == p.tw && th == p.th);
    let src = &mut src[..tw * th];
    let row_clamp = Clamp::<V>::bits(BD + 8);
    let col_clamp = Clamp::<V>::bits(16);
    let rshift = const { row_shift(W, H) };
    let is_rect2 = const { rect2(W, H) };
    let log2w = const { W.ilog2() as usize };
    let zero = Simd::<i32, N>::splat(0);
    let mut t = [V::splat(0); W];
    let mut nz_rows = 0;
    let col_tiles = p.eob_cols.div_ceil(N);
    assert!(col_tiles * N <= tw);
    for g in (0..p.eob_rows.div_ceil(N)).rev() {
        let i0 = g * N;
        // The N input rows (stride `tw`).
        let blk = &mut src[i0 * tw..(i0 + N) * tw];
        // t[j] lane l = Dequant[i0 + l][j]; `nz` is one more than the
        // last non-zero column.
        let mut nz = 0;
        // Lane-wise OR of the rows (only needed until nz_rows is known).
        let mut rows_or = zero;
        for j0 in (0..col_tiles).map(|k| k * N) {
            let rows: [Simd<i32, N>; N] =
                std::array::from_fn(|l| Simd::from_slice(&blk[l * tw + j0..][..N]));
            let last = last_nz_lane(rows.iter().fold(zero, |a, &v| a | v));
            if last == 0 {
                t[j0..j0 + N].fill(V::splat(0));
                continue;
            }
            nz = j0 + last;
            for l in 0..N {
                blk[l * tw + j0..][..N].fill(0);
            }
            for (d, s) in t[j0..j0 + N].iter_mut().zip(V::load_tile(rows)) {
                // Lane l of the OR is non-zero if row i0 + l is.
                rows_or |= s.to_i32();
                *d = s;
            }
        }
        if nz == 0 {
            // The outputs of zero rows are zero (`buf` starts zeroed).
            continue;
        }
        if nz_rows == 0 {
            nz_rows = i0 + last_nz_lane(rows_or);
        }
        if nz == 1 && p.row_kind == Tx1d::Dct {
            // DC only: every DCT output is clamp(Round2(in * cos128(32),
            // 12)) (see the portable `dct_dc`).
            let mut v = t[0];
            if is_rect2 {
                v = v.m1(2896);
            }
            let v = col_clamp
                .apply(row_clamp.apply(v.m1(2896)).rnd(rshift))
                .to_i16();
            for (l, &x) in v.as_array().iter().enumerate() {
                buf.fill_row(i0 + l, W, x);
            }
            continue;
        }
        if is_rect2 {
            for v in &mut t[..nz] {
                *v = v.m1(2896);
            }
        }
        select::<V>(p.row_kind, log2w, nz)(&mut t, row_clamp);
        for j0 in (0..W / N).map(|k| k * N) {
            let o: [Simd<i16, N>; N] =
                std::array::from_fn(|k| col_clamp.apply(t[j0 + k].rnd(rshift)).to_i16());
            for (l, &r) in transpose_e(o).iter().enumerate() {
                buf.put(i0 + l, j0, r);
            }
        }
    }
    nz_rows
}

/// One more than the index of the last non-zero lane of `v` (0 if all
/// are zero).
#[inline(always)]
fn last_nz_lane16<const N: usize>(v: Simd<i16, N>) -> usize {
    let idx = Simd::from_array(std::array::from_fn(|i| i as i16 + 1));
    v.simd_ne(Simd::splat(0))
        .select(idx, Simd::splat(0))
        .reduce_max() as usize
}

/// Transposes the two 8x8 matrices held in the low and the high 8 lanes
/// of `m` (lane `l` of `m[i]` is entry `(i, l)` of the low matrix, lane
/// `8 + l` entry `(i, l)` of the high one), with in-lane interleaves only
/// (`vpunpck{l,h}wd` on ymm, no lane crossing).
#[inline(always)]
fn transpose8x2(mut m: [Simd<i16, 16>; 8]) -> [Simd<i16, 16>; 8] {
    for _ in 0..3 {
        let mut o = m;
        for i in 0..4 {
            let (a, b) = (m[i], m[i + 4]);
            o[2 * i] = simd_swizzle!(
                a,
                b,
                [0, 16, 1, 17, 2, 18, 3, 19, 8, 24, 9, 25, 10, 26, 11, 27]
            );
            o[2 * i + 1] = simd_swizzle!(
                a,
                b,
                [4, 20, 5, 21, 6, 22, 7, 23, 12, 28, 13, 29, 14, 30, 15, 31]
            );
        }
        m = o;
    }
    m
}

/// [`row_pass16`] for 8-bit streams with 16 rows per vector (`S16<16>`,
/// one 256-bit register with AVX2): the low 8 lanes hold rows `i0..i0 +
/// 8` and the high 8 lanes rows `i0 + 8..i0 + 16`, so the tile
/// transposes are two 8x8 ones side by side ([`transpose8x2`]). Requires
/// `W >= 16` and `H >= 16`. Same contract as [`row_pass16`].
fn row_pass16_x2<const W: usize, const H: usize, const CN: usize>(
    src: &mut [i32],
    buf: &mut Cols<CN>,
    p: &Params,
) -> usize {
    type V = S16<16>;
    let tw = W.min(32);
    let th = H.min(32);
    debug_assert!(tw == p.tw && th == p.th && th >= 16 && tw >= 16);
    let src = &mut src[..tw * th];
    // rowClampRange = colClampRange = 16 bits: the lanes saturate.
    let clamp = Clamp::<V>::bits(16);
    let rshift = const { row_shift(W, H) };
    let is_rect2 = const { rect2(W, H) };
    let log2w = const { W.ilog2() as usize };
    let zero = Simd::<i32, 8>::splat(0);
    let mut t = [V::splat(0); W];
    let mut nz_rows = 0;
    let col_tiles = p.eob_cols.div_ceil(8);
    assert!(col_tiles * 8 <= tw);
    for g in (0..p.eob_rows.div_ceil(16)).rev() {
        let i0 = g * 16;
        // The 16 input rows (stride `tw`).
        let blk = &mut src[i0 * tw..(i0 + 16) * tw];
        let mut nz = 0;
        // Lane `l` is non-zero if row `i0 + l` is.
        let mut rows_or = Simd::<i16, 16>::splat(0);
        for j0 in (0..col_tiles).map(|k| k * 8) {
            let rows: [Simd<i32, 8>; 16] =
                std::array::from_fn(|l| Simd::from_slice(&blk[l * tw + j0..][..8]));
            let last = last_nz_lane(rows.iter().fold(zero, |a, &v| a | v));
            if last == 0 {
                t[j0..j0 + 8].fill(V::splat(0));
                continue;
            }
            nz = j0 + last;
            for l in 0..16 {
                blk[l * tw + j0..][..8].fill(0);
            }
            // Dequant has 16 bits for 8-bit streams, so the saturating
            // narrowing (`vpackssdw` + `vpermq`) is exact.
            let m: [Simd<i16, 16>; 8] = std::array::from_fn(|l| {
                let r: Simd<i32, 16> = simd_swizzle!(
                    rows[l],
                    rows[l + 8],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                );
                super::sat16(r)
            });
            for (d, s) in t[j0..j0 + 8].iter_mut().zip(transpose8x2(m)) {
                rows_or |= s;
                *d = S16(s);
            }
        }
        if nz == 0 {
            // The outputs of zero rows are zero (`buf` starts zeroed).
            continue;
        }
        if nz_rows == 0 {
            nz_rows = i0 + last_nz_lane16(rows_or);
        }
        if nz == 1 && p.row_kind == Tx1d::Dct {
            // DC only: every DCT output is clamp(Round2(in * cos128(32),
            // 12)) (see the portable `dct_dc`).
            let mut v = t[0];
            if is_rect2 {
                v = v.m1(2896);
            }
            let v = v.m1(2896).rnd(rshift).0;
            for (l, &x) in v.as_array().iter().enumerate() {
                buf.fill_row(i0 + l, W, x);
            }
            continue;
        }
        if is_rect2 {
            for v in &mut t[..nz] {
                *v = v.m1(2896);
            }
        }
        select::<V>(p.row_kind, log2w, nz)(&mut t, clamp);
        for j0 in (0..W / 8).map(|k| k * 8) {
            let o: [Simd<i16, 16>; 8] = std::array::from_fn(|k| t[j0 + k].rnd(rshift).0);
            for (l, r) in transpose8x2(o).iter().enumerate() {
                let (lo, hi) = (
                    r.resize::<8>(0),
                    simd_swizzle!(*r, [8, 9, 10, 11, 12, 13, 14, 15]),
                );
                buf.put(i0 + l, j0, lo);
                buf.put(i0 + l + 8, j0, hi);
            }
        }
    }
    nz_rows
}

/// Identity row transforms (element-wise, `N` columns at a time, in
/// `i32` since the outputs can exceed 16 bits before the clamp; `w` is
/// at most 32 so `tw == w`). Only the active columns `0..w_active`
/// (`p.eob_cols` rounded up to 8) are transformed; columns beyond that
/// are zero in `src` and skipped by [`col_pass16`].
fn row_pass_identity16<const N: usize, const W: usize, const H: usize, const CN: usize>(
    src: &mut [i32],
    buf: &mut Cols<CN>,
    p: &Params,
) -> usize {
    // Identity transforms are at most 32 long.
    let (w, th) = (W.min(32), H.min(32));
    debug_assert!(w == p.w && th == p.th);
    let src = &mut src[..w * th];
    let col_clamp = Clamp::<Simd<i32, N>>::bits(16);
    let rshift = const { row_shift(W, H) };
    let is_rect2 = const { rect2(W, H) };
    let zero = Simd::<i32, N>::splat(0);
    let w_active = p.eob_cols.next_multiple_of(8).min(w);
    let mut nz_rows = 0;
    for i in (0..p.eob_rows).rev() {
        let row = &mut src[i * w..i * w + w_active];
        let mut any = zero;
        for (k, s) in row.chunks_exact_mut(N).enumerate() {
            let x = Simd::from_slice(s);
            if x == zero {
                // Zero outputs (`buf` starts zeroed).
                continue;
            }
            s.fill(0);
            any |= x;
            let mut v = x;
            if is_rect2 {
                v = v.m1(2896);
            }
            v = identity::<_, W>(v);
            buf.put(i, k * N, col_clamp.apply(v.rnd(rshift)).cast::<i16>());
        }
        if any != zero && nz_rows == 0 {
            nz_rows = i + 1;
        }
    }
    nz_rows
}

/// Column transforms of a `W x H` block, `N` columns at a time, with
/// the final Round2(T[i], colShift = 4), added to the frame with the
/// flips of 7.12.3 step 3. `buf` holds the row pass outputs for rows
/// below `nz_rows`; the other rows are zero. The column kernels run in
/// place in `buf` when it holds all `H` rows.
fn col_pass16<T: Pixel, const N: usize, const W: usize, const H: usize, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    buf: &mut Cols<N>,
    p: &Params,
    nz_rows: usize,
) {
    let rows = buf.rows;
    debug_assert!(nz_rows <= rows && rows.is_power_of_two() && rows >= 4);
    let x0 = |j0: usize| if p.flip_lr { W - j0 - N } else { j0 };
    let flip = |r: Simd<i16, N>| if p.flip_lr { r.reverse() } else { r };
    let row = |i: usize| if p.flip_ud { H - 1 - i } else { i };
    let w_active = if p.row_kind == Tx1d::Identity {
        debug_assert!(!p.flip_lr);
        // Columns beyond the transformed ones are zero in `buf`.
        p.eob_cols.next_multiple_of(N).min(W)
    } else {
        W
    };
    if nz_rows == 1 && p.col_kind == Tx1d::Dct {
        // Every column is constant.
        for j0 in (0..w_active / N).map(|k| k * N) {
            let r = flip(buf.v[(j0 / N) * rows].m1(2896).rnd(4).0);
            visit_rows::<T, N, H>(&mut dst[x0(j0)..], stride, |_, d| {
                add_row16::<T, N, BD>(d, r)
            });
        }
        return;
    }
    if p.col_kind == Tx1d::Identity {
        // Zero rows have zero outputs.
        for i in 0..nz_rows {
            let y_off = row(i) * stride;
            for j0 in (0..w_active / N).map(|k| k * N) {
                let v = buf.v[(j0 / N) * rows + i].0;
                let r = flip(col_identity_rnd4::<N, H>(v));
                add_row16_at::<T, N, BD>(dst, y_off + x0(j0), r);
            }
        }
        return;
    }
    let log2h = const { H.ilog2() as usize };
    // The kernel reads the first `next_power_of_two(nz_rows).max(4) <=
    // rows` entries (rows at or beyond nz_rows are zero) and overwrites
    // all `H`.
    let kernel = select::<S16<N>>(p.col_kind, log2h, nz_rows);
    let clamp = Clamp::<S16<N>>::bits(16);
    let mut tmp = [S16::<N>::splat(0); H];
    for j0 in (0..w_active / N).map(|k| k * N) {
        let g = &mut buf.v[(j0 / N) * rows..][..rows];
        let c: &mut [S16<N>] = if rows == H {
            g
        } else {
            tmp[..rows].copy_from_slice(g);
            &mut tmp
        };
        kernel(c, clamp);
        let c: &[S16<N>; H] = (&*c).try_into().unwrap();
        let x = x0(j0);
        if !p.flip_lr && !p.flip_ud {
            visit_rows::<T, N, H>(&mut dst[x..], stride, |y, d| {
                add_row16::<T, N, BD>(d, c[y].rnd(4).0)
            });
        } else {
            visit_rows::<T, N, H>(&mut dst[x..], stride, |y, d| {
                let v = c[if p.flip_ud { H - 1 - y } else { y }].rnd(4).0;
                add_row16::<T, N, BD>(d, if p.flip_lr { v.reverse() } else { v })
            });
        }
    }
}
