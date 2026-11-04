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
    add_row16, amap, col_identity_rnd4, dct_adst_small, iadst4, iadst4_nz1, identity, kernels,
    last_nz_lane, last_nz_lane16, rect2, row_shift, select, transpose_e, visit_rows, Clamp, Kern,
    Params, RowLanes, TxLanes, S16,
};
use crate::decoder::av1::dsp::itx::Tx1d;
use crate::decoder::av1::pixel::Pixel;

/// Row pass outputs (16-bit) of `ROWS` rows, stored by groups of `CN`
/// columns: entry `g * ROWS + i` holds row `i`, columns `g * CN..(g + 1) *
/// CN`. The `H` (or `ROWS`) entries of a group are the column pass
/// vectors of those columns, so the column kernels run in place.
/// Entries not written by the row pass are zero.
pub(super) struct Cols<'a, const CN: usize, const ROWS: usize> {
    v: &'a mut [S16<CN>],
}

impl<const CN: usize, const ROWS: usize> Cols<'_, CN, ROWS> {
    /// Stores the `M` columns `j0..j0 + M` of row `i` (`M <= CN`, `j0` a
    /// multiple of `M`).
    #[inline(always)]
    fn put<const M: usize>(&mut self, i: usize, j0: usize, x: Simd<i16, M>) {
        if M == CN {
            self.v[(j0 / CN) * ROWS + i] = S16(x.resize::<CN>(0));
            return;
        }
        let e = self.v[(j0 / CN) * ROWS + i].0.as_mut_array();
        e[j0 % CN..][..M].copy_from_slice(x.as_array());
    }

    /// Sets all `w` columns of row `i` to `x`.
    #[inline(always)]
    fn fill_row(&mut self, i: usize, w: usize, x: i16) {
        for g in 0..w / CN {
            self.v[g * ROWS + i] = S16(Simd::splat(x));
        }
    }
}

/// 2D transform of a `W x H` block with 16-bit intermediates.
/// `coeffs[i * tw + j]` holds `Dequant[i][j]` (`coeffs.len() == tw *
/// th`, `TH == th == min(H, 32)`). Rows are transformed `RN` at a time,
/// columns `CN` at a time. `LV = W * TH / CN` and `LV4 = W * 4 / CN`
/// are the [`Cols`] buffer sizes of the full and 4-row paths. Leaves
/// `coeffs` all zero.
pub(super) fn transform_add16<
    T: Pixel,
    const BD: u32,
    const W: usize,
    const H: usize,
    const TH: usize,
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
    debug_assert!(
        W == p.w && H == p.h && TH == p.th && LV * CN == W * TH && LV4 * CN == W * 4 && BD == p.bd
    );
    if p.row_kind != Tx1d::Identity && p.eob_rows <= 4 {
        let mut buf = [S16::<CN>::splat(0); LV4];
        let mut cols = Cols::<CN, 4> { v: &mut buf };
        let nz_rows = if p.eob_cols <= 4 {
            if BD == 8 {
                row_pass16_sub4::<S16<4>, BD, W, H, CN>(coeffs, &mut cols, p)
            } else {
                row_pass16_sub4::<Simd<i32, 4>, BD, W, H, CN>(coeffs, &mut cols, p)
            }
        } else if BD == 8 {
            row_pass16::<S16<4>, BD, 4, W, H, CN, 4>(coeffs, &mut cols, p)
        } else {
            row_pass16_hbd::<BD, 4, W, H, CN, 4>(coeffs, &mut cols, p)
        };
        if nz_rows != 0 {
            col_pass16::<T, CN, W, H, 4, BD, true>(dst, stride, &mut cols, p, nz_rows);
        }
        return;
    }
    let mut buf = [S16::<CN>::splat(0); LV];
    let mut cols = Cols::<CN, TH> { v: &mut buf };
    let nz_rows = if p.row_kind == Tx1d::Identity {
        row_pass_identity16::<RN, W, H, CN, TH>(coeffs, &mut cols, p)
    } else if BD == 8 && CN == 16 && H >= 16 && W >= 16 && p.eob_rows > 8 {
        // AVX2: 16 rows per 256-bit vector.
        row_pass16_x2::<W, H, CN, TH>(coeffs, &mut cols, p)
    } else if BD == 8 {
        row_pass16::<S16<RN>, BD, RN, W, H, CN, TH>(coeffs, &mut cols, p)
    } else {
        row_pass16_hbd::<BD, RN, W, H, CN, TH>(coeffs, &mut cols, p)
    };
    if nz_rows == 0 {
        return;
    }
    col_pass16::<T, CN, W, H, TH, BD, false>(dst, stride, &mut cols, p, nz_rows);
}

/// Fast path of [`row_pass16`] when `eob` guarantees all non-zero
/// coefficients lie in the top-left 4x4 (`eob_rows <= 4 && eob_cols <= 4`).
/// Transforms at most 4 rows on 4-lane vectors using the `nz4` kernel.
#[inline(never)]
fn row_pass16_sub4<
    V: RowLanes<4>,
    const BD: u32,
    const W: usize,
    const H: usize,
    const CN: usize,
>(
    src: &mut [i32],
    buf: &mut Cols<CN, 4>,
    p: &Params,
) -> usize {
    let tw = W.min(32);
    let th = H.min(32);
    debug_assert!(tw == p.tw && th == p.th && p.eob_rows <= 4 && p.eob_cols <= 4);
    let src = &mut src[..tw * th];
    assert!(buf.v.len() == (W / CN) * 4);
    let rows: [Simd<i32, 4>; 4] = [
        Simd::from_slice(&src[..4]),
        Simd::from_slice(&src[tw..][..4]),
        Simd::from_slice(&src[2 * tw..][..4]),
        Simd::from_slice(&src[3 * tw..][..4]),
    ];
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
    let nz = if p.eob_cols == 1 { 1 } else { last_nz_lane(col_or) };
    if BD > 8 {
        // 10 bits: 16-bit lanes are exact if `rows_fit_s16`.
        let tile16 = transpose_e(amap(rows, super::sat16));
        if rows_fit_s16(tile16) {
            let tile16 = amap(tile16, S16);
            if p.row_kind == Tx1d::Dct && nz == 1 {
                row_dc::<S16<4>, 4, W, H, CN, 4>(tile16[0], 0, buf, 16);
            } else {
                sub4_tail::<S16<4>, W, H, CN>(tile16, nz, buf, p, 16);
            }
        } else {
            row_pass16_sub4_32::<BD, W, H, CN>(rows, nz, buf, p);
        }
        return nz_rows;
    }
    let tile = V::load_tile(rows);
    if p.row_kind == Tx1d::Dct && nz == 1 {
        row_dc::<V, 4, W, H, CN, 4>(tile[0], 0, buf, BD + 8);
        return nz_rows;
    }
    sub4_tail::<V, W, H, CN>(tile, nz, buf, p, BD + 8);
    nz_rows
}

/// 32-bit lane fallback for [`row_pass16_sub4`] when a 10-bit row
/// exceeds [`S16_ROW_L1_MAX`].
#[cold]
#[inline(never)]
fn row_pass16_sub4_32<const BD: u32, const W: usize, const H: usize, const CN: usize>(
    rows: [Simd<i32, 4>; 4],
    nz: usize,
    buf: &mut Cols<CN, 4>,
    p: &Params,
) {
    let tile = <Simd<i32, 4> as RowLanes<4>>::load_tile(rows);
    if p.row_kind == Tx1d::Dct && nz == 1 {
        row_dc::<Simd<i32, 4>, 4, W, H, CN, 4>(tile[0], 0, buf, BD + 8);
    } else {
        sub4_tail::<Simd<i32, 4>, W, H, CN>(tile, nz, buf, p, BD + 8);
    }
}

/// Rounds and clamps `M` adjacent row transform outputs `t[j0..j0 + M]`
/// to 16-bit column inputs without going through `array::from_fn`.
#[inline(always)]
fn rnd_tile<U: RowLanes<N>, const N: usize, const M: usize>(
    t: &[U],
    j0: usize,
    col_clamp: Clamp<U>,
    rshift: u32,
) -> [Simd<i16, M>; M] {
    let ts: &[U; M] = (&t[j0..j0 + M]).try_into().unwrap();
    let mut o = [Simd::<i16, M>::splat(0); M];
    for k in 0..M {
        o[k] = col_clamp.apply(ts[k].rnd(rshift)).to_i16().resize::<M>(0);
    }
    o
}

/// Rest of [`row_pass16_sub4`]: the `nz <= 4` row transforms of the (at
/// most) 4 rows in `tile` (`tile[j]` lane `l` = input `j` of row `l`) with
/// intermediate clamping to `row_bits`, stored to `buf`.
#[inline(always)]
fn sub4_tail<U: RowLanes<4>, const W: usize, const H: usize, const CN: usize>(
    tile: [U; 4],
    nz: usize,
    buf: &mut Cols<CN, 4>,
    p: &Params,
    row_bits: u32,
) {
    let row_clamp = Clamp::<U>::bits(row_bits);
    let col_clamp = Clamp::<U>::bits(16);
    let rshift = const { row_shift(W, H) };
    let is_rect2 = const { rect2(W, H) };
    let mut t = [U::splat(0); W];
    for (d, mut s) in t[..4].iter_mut().zip(tile) {
        if is_rect2 {
            s = s.m1_h0(2896);
        }
        *d = s;
    }
    if W == 4 {
        run_small::<U, 4>(
            p.row_kind,
            nz.min(4),
            t.first_chunk_mut::<4>().unwrap(),
            row_clamp,
        );
    } else if W == 8 {
        run_small::<U, 8>(
            p.row_kind,
            nz.min(4),
            t.first_chunk_mut::<8>().unwrap(),
            row_clamp,
        );
    } else {
        select::<U>(p.row_kind, const { W.ilog2() as usize }, nz)(&mut t, row_clamp);
    }
    if CN == 8 && W >= 8 {
        for j0 in (0..W / 8).map(|k| k * 8) {
            let o0 = rnd_tile::<U, 4, 4>(&t, j0, col_clamp, rshift);
            let o1 = rnd_tile::<U, 4, 4>(&t, j0 + 4, col_clamp, rshift);
            let (t0, t1) = (transpose_e(o0), transpose_e(o1));
            for l in 0..4 {
                buf.put(l, j0, simd_swizzle!(t0[l], t1[l], [0, 1, 2, 3, 4, 5, 6, 7]));
            }
        }
    } else {
        for j0 in (0..W / 4).map(|k| k * 4) {
            let o = rnd_tile::<U, 4, 4>(&t, j0, col_clamp, rshift);
            for (l, &r) in transpose_e(o).iter().enumerate() {
                buf.put(l, j0, r);
            }
        }
    }
}

/// Row transforms of width `W`, `N` rows at a time, from `src` (stride
/// `tw`) to `buf` (stride `W`), including the clamp to colClampRange
/// (16 bits). Returns one more than the index of the last non-zero
/// input row (0 if all are zero); output rows from there on are not
/// written. Leaves `src` all zero.
#[inline(never)]
fn row_pass16<
    V: RowLanes<N>,
    const BD: u32,
    const N: usize,
    const W: usize,
    const H: usize,
    const CN: usize,
    const ROWS: usize,
>(
    src: &mut [i32],
    buf: &mut Cols<CN, ROWS>,
    p: &Params,
) -> usize {
    let tw = W.min(32);
    let th = H.min(32);
    debug_assert!(tw == p.tw && th == p.th && ROWS <= th);
    let src = &mut src[..tw * th];
    assert!(buf.v.len() == (W / CN) * ROWS);
    let zero = V::splat(0);
    let mut t = [zero; W];
    let mut nz_rows = 0;
    let row_tiles = if ROWS == N { 1 } else { p.eob_rows.div_ceil(N) };
    let col_tiles = p.eob_cols.div_ceil(N);
    assert!(row_tiles <= ROWS / N && col_tiles <= tw / N);
    for g in (0..row_tiles).rev() {
        let i0 = g * N;
        // The N input rows (stride `tw`).
        let blk = &mut src[i0 * tw..][..N * tw];
        // t[j] lane l = Dequant[i0 + l][j]; `nz` is one more than the
        // last non-zero column.
        let mut nz = 0;
        // Lane-wise OR of the rows (only needed until nz_rows is known).
        let mut rows_or = zero;
        for k in 0..col_tiles {
            let j0 = k * N;
            let mut rows = [zero; N];
            let mut row_or = zero;
            for l in 0..N {
                let r = V::load_row((&blk[l * tw + j0..][..N]).try_into().unwrap());
                row_or = row_or.bitor(r);
                rows[l] = r;
            }
            let last = row_or.last_nz_lane();
            if last == 0 {
                t[j0..][..N].fill(zero);
                continue;
            }
            nz = j0 + last;
            for l in 0..N {
                blk[l * tw + j0..][..N].fill(0);
            }
            for (d, s) in t[j0..][..N].iter_mut().zip(V::transpose_tile(rows)) {
                // Lane l of the OR is non-zero if row i0 + l is.
                rows_or = rows_or.bitor(s);
                *d = s;
            }
        }
        if nz == 0 {
            // The outputs of zero rows are zero (`buf` starts zeroed).
            continue;
        }
        if nz_rows == 0 {
            nz_rows = i0 + rows_or.last_nz_lane();
        }
        if nz == 1 && p.row_kind == Tx1d::Dct {
            row_dc::<V, N, W, H, CN, ROWS>(t[0], i0, buf, BD + 8);
            continue;
        }
        row_tail::<V, N, W, H, CN, ROWS>(&mut t, nz, i0, buf, p, BD + 8);
    }
    nz_rows
}

/// [`row_pass16`] for 10-bit streams. The rows are loaded narrowed to 16
/// bits with saturation and each group of `N` rows is transformed on
/// [`S16`] lanes if [`rows_fit_s16`] (all rows of real streams but a
/// handful), else reloaded and transformed on `i32` lanes.
#[inline(never)]
fn row_pass16_hbd<
    const BD: u32,
    const N: usize,
    const W: usize,
    const H: usize,
    const CN: usize,
    const ROWS: usize,
>(
    src: &mut [i32],
    buf: &mut Cols<CN, ROWS>,
    p: &Params,
) -> usize {
    let tw = W.min(32);
    let th = H.min(32);
    debug_assert!(tw == p.tw && th == p.th && ROWS <= th && BD == 10);
    let src = &mut src[..tw * th];
    assert!(buf.v.len() == (W / CN) * ROWS);
    let zero = Simd::<i16, N>::splat(0);
    let mut t = [S16::<N>::splat(0); W];
    let mut nz_rows = 0;
    let row_tiles = if ROWS == N { 1 } else { p.eob_rows.div_ceil(N) };
    let col_tiles = p.eob_cols.div_ceil(N);
    assert!(row_tiles <= ROWS / N && col_tiles <= tw / N && tw / N <= 32);
    for g in (0..row_tiles).rev() {
        let i0 = g * N;
        // The N input rows (stride `tw`).
        let blk = &mut src[i0 * tw..][..N * tw];
        // t[j] lane l = sat16(Dequant[i0 + l][j]); `nz` is one more than
        // the last non-zero column; bit k of `nz_tiles` is set if column
        // tile k is non-zero.
        let mut nz = 0;
        let mut nz_tiles = 0u32;
        // Lane-wise OR of the rows (only needed until nz_rows is known).
        let mut rows_or = zero;
        for k in 0..col_tiles {
            let j0 = k * N;
            let mut rows = [zero; N];
            let mut row_or = zero;
            for l in 0..N {
                let r = super::sat16(Simd::from_slice(&blk[l * tw + j0..][..N]));
                row_or |= r;
                rows[l] = r;
            }
            let last = last_nz_lane16(row_or);
            if last == 0 {
                t[j0..][..N].fill(S16(zero));
                continue;
            }
            nz = j0 + last;
            nz_tiles |= 1 << k;
            for (d, s) in t[j0..][..N].iter_mut().zip(transpose_e(rows)) {
                // Lane l of the OR is non-zero if row i0 + l is.
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
        let dc_only = nz == 1 && p.row_kind == Tx1d::Dct;
        if rows_fit_s16(t[..nz.min(W)].iter().map(|v| v.0)) {
            // 16-bit lanes are exact for these rows (see `rows_fit_s16`).
            if dc_only {
                row_dc::<S16<N>, N, W, H, CN, ROWS>(t[0], i0, buf, 16);
            } else {
                row_tail::<S16<N>, N, W, H, CN, ROWS>(&mut t, nz, i0, buf, p, 16);
            }
        } else {
            row_pass16_hbd_row32::<BD, N, W, H, CN, ROWS>(
                blk, col_tiles, nz_tiles, nz, i0, dc_only, buf, p,
            );
        }
        for k in (0..col_tiles).filter(|k| nz_tiles & (1 << k) != 0) {
            for l in 0..N {
                blk[l * tw + k * N..][..N].fill(0);
            }
        }
    }
    nz_rows
}

/// 32-bit lane fallback for one row group of [`row_pass16_hbd`] when a
/// 10-bit row exceeds [`S16_ROW_L1_MAX`].
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn row_pass16_hbd_row32<
    const BD: u32,
    const N: usize,
    const W: usize,
    const H: usize,
    const CN: usize,
    const ROWS: usize,
>(
    blk: &[i32],
    col_tiles: usize,
    nz_tiles: u32,
    nz: usize,
    i0: usize,
    dc_only: bool,
    buf: &mut Cols<CN, ROWS>,
    p: &Params,
) {
    let tw = W.min(32);
    let mut t32 = [Simd::<i32, N>::splat(0); W];
    for k in (0..col_tiles).filter(|k| nz_tiles & (1 << k) != 0) {
        let j0 = k * N;
        let mut rows = [Simd::<i32, N>::splat(0); N];
        for l in 0..N {
            rows[l] = Simd::from_slice(&blk[l * tw + j0..][..N]);
        }
        t32[j0..][..N].copy_from_slice(&transpose_e(rows));
    }
    if dc_only {
        row_dc::<Simd<i32, N>, N, W, H, CN, ROWS>(t32[0], i0, buf, BD + 8);
    } else {
        row_tail::<Simd<i32, N>, N, W, H, CN, ROWS>(&mut t32, nz, i0, buf, p, BD + 8);
    }
}

/// DC-only row transforms of a group of `N` rows (lane `l` of `v` is the
/// DC input of row `i0 + l`): every DCT output is clamp(Round2(in *
/// cos128(32), 12)) (see the portable `dct_dc`).
#[inline(always)]
fn row_dc<
    U: RowLanes<N>,
    const N: usize,
    const W: usize,
    const H: usize,
    const CN: usize,
    const ROWS: usize,
>(
    mut v: U,
    i0: usize,
    buf: &mut Cols<CN, ROWS>,
    row_bits: u32,
) {
    let row_clamp = Clamp::<U>::bits(row_bits);
    let col_clamp = Clamp::<U>::bits(16);
    if const { rect2(W, H) } {
        v = v.m1(2896);
    }
    let v = col_clamp
        .apply(row_clamp.apply(v.m1(2896)).rnd(const { row_shift(W, H) }))
        .to_i16();
    for (l, &x) in v.as_array().iter().enumerate() {
        buf.fill_row(i0 + l, W, x);
    }
}

/// Largest per-row sum of absolute (rect2-unscaled) input coefficients
/// for which a row transform on saturating 16-bit lanes ([`S16`]) is
/// exact. Every intermediate value `v` of the AV1 1D DCT / ADST networks
/// (up to 64 points) is a linear combination `sum(c_k * x_k)` of the
/// inputs with all `|c_k| <= 1`, plus at most 63 of accumulated rounding
/// error (bounds computed exactly from the butterfly networks and asserted
/// by `check_s16_row_bound` in `gen_kernels.py` when it generates the SIMD
/// kernels; the largest are `|c_k| < 0.99992` and an error below 62.9,
/// both for DCT64), so `|v| <= sum(|x_k|) + 63`. With this limit no
/// value exceeds 16 bits: no lane saturates and no clamp to the 18-bit
/// row range of 10-bit streams applies, so the results equal those of
/// the `i32` lanes. The rect2 pre-scaling never increases `|x_k|`, and
/// the identity transforms of length 4 and 8 (`Round2(x * 5793, 12)`,
/// `2 * x`) followed by the clamp to the 16-bit column range give the
/// same as their saturating 16-bit versions.
pub(super) const S16_ROW_L1_MAX: i16 = i16::MAX - 127;

/// Whether every lane (row) of `t` has an absolute sum of at most
/// [`S16_ROW_L1_MAX`], where `t[j]` lane `l` is input `j` of row `l`
/// narrowed to 16 bits with saturation. The sums saturate too, so a row
/// with an input beyond 16 bits (which saturates to at least 32767) or a
/// larger sum is rejected.
#[inline(always)]
pub(super) fn rows_fit_s16<const N: usize>(t: impl IntoIterator<Item = Simd<i16, N>>) -> bool {
    let zero = Simd::<i16, N>::splat(0);
    let mut l1 = zero;
    for v in t {
        l1 = l1.saturating_add(v.simd_max(zero.saturating_sub(v)));
    }
    #[cfg(target_arch = "aarch64")]
    {
        return l1.reduce_max() <= S16_ROW_L1_MAX;
    }
    #[cfg(not(target_arch = "aarch64"))]
    l1.simd_le(Simd::splat(S16_ROW_L1_MAX)).all()
}

/// Rest of [`row_pass16`] for one group of `N` rows held in `t` (`t[j]`
/// lane `l` = input `j` of row `i0 + l`, `nz` one more than the last
/// non-zero input): rect2 scaling, the 1D transform with intermediate
/// clamping to `row_bits`, rounding and the transposed store to `buf`.
#[inline(always)]
fn row_tail<
    U: RowLanes<N>,
    const N: usize,
    const W: usize,
    const H: usize,
    const CN: usize,
    const ROWS: usize,
>(
    t: &mut [U; W],
    nz: usize,
    i0: usize,
    buf: &mut Cols<CN, ROWS>,
    p: &Params,
    row_bits: u32,
) {
    let row_clamp = Clamp::<U>::bits(row_bits);
    let col_clamp = Clamp::<U>::bits(16);
    let rshift = const { row_shift(W, H) };
    let is_rect2 = const { rect2(W, H) };
    let log2w = const { W.ilog2() as usize };
    if is_rect2 {
        for v in &mut t[..nz.min(W)] {
            *v = v.m1_h0(2896);
        }
    }
    select::<U>(p.row_kind, log2w, nz)(t, row_clamp);
    if N == 4 && CN == 8 && W >= 8 {
        for j0 in (0..W / 8).map(|k| k * 8) {
            let o0 = rnd_tile::<U, N, 4>(t, j0, col_clamp, rshift);
            let o1 = rnd_tile::<U, N, 4>(t, j0 + 4, col_clamp, rshift);
            let (t0, t1) = (transpose_e(o0), transpose_e(o1));
            for l in 0..4 {
                buf.put(
                    i0 + l,
                    j0,
                    simd_swizzle!(t0[l], t1[l], [0, 1, 2, 3, 4, 5, 6, 7]),
                );
            }
        }
    } else {
        for j0 in (0..W / N).map(|k| k * N) {
            let o = rnd_tile::<U, N, N>(t, j0, col_clamp, rshift);
            for (l, &r) in transpose_e(o).iter().enumerate() {
                buf.put(i0 + l, j0, r);
            }
        }
    }
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
fn row_pass16_x2<const W: usize, const H: usize, const CN: usize, const ROWS: usize>(
    src: &mut [i32],
    buf: &mut Cols<CN, ROWS>,
    p: &Params,
) -> usize {
    type V = S16<16>;
    let tw = W.min(32);
    let th = H.min(32);
    debug_assert!(tw == p.tw && th == p.th && th >= 16 && tw >= 16 && ROWS == th);
    let src = &mut src[..tw * th];
    assert!(buf.v.len() == (W / CN) * ROWS);
    // rowClampRange = colClampRange = 16 bits: the lanes saturate.
    let clamp = Clamp::<V>::bits(16);
    let rshift = const { row_shift(W, H) };
    let is_rect2 = const { rect2(W, H) };
    let log2w = const { W.ilog2() as usize };
    let zero = Simd::<i32, 8>::splat(0);
    let mut t = [V::splat(0); W];
    let mut nz_rows = 0;
    let row_tiles = p.eob_rows.div_ceil(16);
    let col_tiles = p.eob_cols.div_ceil(8);
    assert!(row_tiles <= ROWS / 16 && col_tiles <= tw / 8);
    for g in (0..row_tiles).rev() {
        let i0 = g * 16;
        // The 16 input rows (stride `tw`).
        let blk = &mut src[i0 * tw..][..16 * tw];
        let mut nz = 0;
        // Lane `l` is non-zero if row `i0 + l` is.
        let mut rows_or = Simd::<i16, 16>::splat(0);
        for k in 0..col_tiles {
            let j0 = k * 8;
            let mut rows = [zero; 16];
            let mut row_or = zero;
            for l in 0..16 {
                let r = Simd::from_slice(&blk[l * tw + j0..][..8]);
                row_or |= r;
                rows[l] = r;
            }
            let last = last_nz_lane(row_or);
            if last == 0 {
                t[j0..][..8].fill(V::splat(0));
                continue;
            }
            nz = j0 + last;
            for l in 0..16 {
                blk[l * tw + j0..][..8].fill(0);
            }
            // Dequant has 16 bits for 8-bit streams, so the saturating
            // narrowing (`vpackssdw` + `vpermq`) is exact.
            let mut m = [Simd::<i16, 16>::splat(0); 8];
            for l in 0..8 {
                let r: Simd<i32, 16> = simd_swizzle!(
                    rows[l],
                    rows[l + 8],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                );
                m[l] = super::sat16(r);
            }
            for (d, s) in t[j0..][..8].iter_mut().zip(transpose8x2(m)) {
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
            for v in &mut t[..nz.min(W)] {
                *v = v.m1_h0(2896);
            }
        }
        select::<V>(p.row_kind, log2w, nz)(&mut t, clamp);
        for j0 in (0..W / 8).map(|k| k * 8) {
            let ts: &[V; 8] = (&t[j0..j0 + 8]).try_into().unwrap();
            let mut o = [Simd::<i16, 16>::splat(0); 8];
            for k in 0..8 {
                o[k] = ts[k].rnd(rshift).0;
            }
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
fn row_pass_identity16<
    const N: usize,
    const W: usize,
    const H: usize,
    const CN: usize,
    const ROWS: usize,
>(
    src: &mut [i32],
    buf: &mut Cols<CN, ROWS>,
    p: &Params,
) -> usize {
    // Identity transforms are at most 32 long.
    let (w, th) = (W.min(32), H.min(32));
    debug_assert!(w == p.w && th == p.th && ROWS == th);
    let src = &mut src[..w * th];
    assert!(buf.v.len() == (W / CN) * ROWS);
    let col_clamp = Clamp::<Simd<i32, N>>::bits(16);
    let rshift = const { row_shift(W, H) };
    let is_rect2 = const { rect2(W, H) };
    let zero = Simd::<i32, N>::splat(0);
    let w_active = p.eob_cols.next_multiple_of(8).min(w);
    let mut nz_rows = 0;
    for i in (0..p.eob_rows.min(ROWS)).rev() {
        let row = &mut src[i * w..][..w_active];
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

/// Inline 4-point or 8-point 1D transform on a register array `t`,
/// given that only the first `nz` (`1 <= nz <= N`) inputs can be non-zero.
// Not inlined with debug assertions (unoptimized and sanitizer builds),
// where every inlined copy keeps its own stack slots and the frames of
// the callers would exceed the default 2 MiB thread stack.
#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
fn run_small<V: TxLanes, const N: usize>(kind: Tx1d, nz: usize, t: &mut [V; N], clamp: Clamp<V>) {
    use kernels::*;
    debug_assert!(N == 4 || N == 8);
    if N == 4 {
        if kind == Tx1d::Adst {
            if nz <= 1 {
                iadst4_nz1(t, clamp);
            } else {
                iadst4(t, clamp);
            }
        } else if nz <= 2 {
            idct4_nz2(t, clamp);
        } else {
            dct_adst_small(Tx1d::Dct, true, t, clamp);
        }
        return;
    }
    if kind == Tx1d::Adst {
        match nz {
            0 | 1 => iadst8_nz1(t, clamp),
            2 => iadst8_nz2(t, clamp),
            3 | 4 => dct_adst_small(Tx1d::Adst, true, t, clamp),
            _ => dct_adst_small(Tx1d::Adst, false, t, clamp),
        }
    } else {
        match nz {
            0..=2 => idct8_nz2(t, clamp),
            3 | 4 => dct_adst_small(Tx1d::Dct, true, t, clamp),
            _ => dct_adst_small(Tx1d::Dct, false, t, clamp),
        }
    }
}

/// Runs the column transform of height `H` on `c`: inline on register
/// arrays for `H <= 8`, and via the pre-selected `kernel` for `H >= 16`.
#[inline(always)]
fn run_col<V: TxLanes, const H: usize>(
    kind: Tx1d,
    nz: usize,
    kernel: Kern<V>,
    c: &mut [V; H],
    clamp: Clamp<V>,
) {
    if H == 4 {
        run_small::<V, 4>(kind, nz, c.first_chunk_mut::<4>().unwrap(), clamp);
    } else if H == 8 {
        run_small::<V, 8>(kind, nz, c.first_chunk_mut::<8>().unwrap(), clamp);
    } else {
        kernel(c, clamp);
    }
}

/// Concatenates the first 8 lanes of `a` and `b` into a 16-lane vector.
#[inline(always)]
fn concat8<const N: usize>(a: Simd<i16, N>, b: Simd<i16, N>) -> Simd<i16, 16> {
    let a8 = a.resize::<8>(0);
    let b8 = b.resize::<8>(0);
    simd_swizzle!(
        a8,
        b8,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    )
}

/// Adds one `N`-column group `c` at column `j0` to `dst`.
#[inline(always)]
fn add_cols16_one<T: Pixel, const N: usize, const W: usize, const H: usize, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    j0: usize,
    c: &[S16<N>; H],
    flip_lr: bool,
    flip_ud: bool,
) {
    let x = if flip_lr { W - j0 - N } else { j0 };
    if !flip_lr && !flip_ud {
        visit_rows::<T, N, H>(&mut dst[x..], stride, |y, d| {
            add_row16::<T, N, BD>(d, c[y].rnd4_add())
        });
    } else {
        visit_rows::<T, N, H>(&mut dst[x..], stride, |y, d| {
            let v = c[if flip_ud { H - 1 - y } else { y }].rnd4_add();
            add_row16::<T, N, BD>(d, if flip_lr { v.reverse() } else { v })
        });
    }
}

/// Adds two adjacent 8-column groups `c0` and `c1` at column `j0` (16
/// columns total) to `dst` in a single `visit_rows::<T, 16, H>` pass.
#[inline(always)]
fn add_cols16_pair<T: Pixel, const N: usize, const W: usize, const H: usize, const BD: u32>(
    dst: &mut [T],
    stride: usize,
    j0: usize,
    c0: &[S16<N>; H],
    c1: &[S16<N>; H],
    flip_lr: bool,
    flip_ud: bool,
) {
    let x = if flip_lr { W - j0 - 16 } else { j0 };
    if !flip_lr && !flip_ud {
        visit_rows::<T, 16, H>(&mut dst[x..], stride, |y, d| {
            add_row16::<T, 16, BD>(d, concat8(c0[y].rnd4_add(), c1[y].rnd4_add()))
        });
    } else {
        visit_rows::<T, 16, H>(&mut dst[x..], stride, |y, d| {
            let yi = if flip_ud { H - 1 - y } else { y };
            let (r0, r1) = (c0[yi].rnd4_add(), c1[yi].rnd4_add());
            let r = if flip_lr { concat8(r1.reverse(), r0.reverse()) } else { concat8(r0, r1) };
            add_row16::<T, 16, BD>(d, r)
        });
    }
}

/// Column transforms of a `W x H` block, `N` columns at a time, with
/// the final Round2(T[i], colShift = 4), added to the frame with the
/// flips of 7.12.3 step 3. `buf` holds the row pass outputs for rows
/// below `nz_rows`; the other rows are zero. The column kernels run in
/// place in `buf` when it holds all `H` rows.
fn col_pass16<
    T: Pixel,
    const N: usize,
    const W: usize,
    const H: usize,
    const ROWS: usize,
    const BD: u32,
    const SUB4: bool,
>(
    dst: &mut [T],
    stride: usize,
    buf: &mut Cols<N, ROWS>,
    p: &Params,
    nz_rows: usize,
) {
    assert!(nz_rows <= ROWS && ROWS.is_power_of_two() && ROWS >= 4);
    assert!(buf.v.len() == (W / N) * ROWS);
    let flip_lr = W <= 16 && p.flip_lr;
    let flip_ud = H <= 16 && p.flip_ud;
    let flip = |r: Simd<i16, N>| if flip_lr { r.reverse() } else { r };
    let row = |i: usize| if flip_ud { H - 1 - i } else { i };
    let w_active = if !SUB4 && W <= 32 && p.row_kind == Tx1d::Identity {
        debug_assert!(!flip_lr);
        // Columns beyond the transformed ones are zero in `buf`.
        p.eob_cols.next_multiple_of(N).min(W)
    } else {
        W
    };
    if nz_rows == 1 && (H > 32 || p.col_kind == Tx1d::Dct) {
        // Every column is constant (untransformed columns in `buf` are zero).
        let r: [Simd<i16, N>; 8] = std::array::from_fn(|g| {
            if g < W / N {
                let src_g = if flip_lr { W / N - 1 - g } else { g };
                flip(buf.v[src_g * ROWS].m1(2896).rnd4_add())
            } else {
                Simd::splat(0)
            }
        });
        if N == 8 && W >= 16 {
            let r16: [Simd<i16, 16>; 4] =
                std::array::from_fn(|g2| concat8(r[2 * g2], r[2 * g2 + 1]));
            visit_rows::<T, W, H>(dst, stride, |_, row| {
                for g2 in 0..W / 16 {
                    let d: &mut [T; 16] = (&mut row[g2 * 16..(g2 + 1) * 16]).try_into().unwrap();
                    add_row16::<T, 16, BD>(d, r16[g2]);
                }
            });
            return;
        }
        visit_rows::<T, W, H>(dst, stride, |_, row| {
            for g in 0..W / N {
                let d: &mut [T; N] = (&mut row[g * N..(g + 1) * N]).try_into().unwrap();
                add_row16::<T, N, BD>(d, r[g]);
            }
        });
        return;
    }
    if H <= 32 && p.col_kind == Tx1d::Identity {
        // Zero rows have zero outputs (and untransformed columns in `buf` are zero).
        let stride = stride & 0xffff_ffff;
        let dst = &mut dst[..(H - 1) * stride + W];
        for i in 0..nz_rows {
            let y = row(i);
            let d_row: &mut [T; W] = (&mut dst[y * stride..][..W]).try_into().unwrap();
            for g in 0..W / N {
                let src_g = if flip_lr { W / N - 1 - g } else { g };
                let v = buf.v[src_g * ROWS + i].0;
                let r = flip(col_identity_rnd4::<N, H>(v));
                let d: &mut [T; N] = (&mut d_row[g * N..(g + 1) * N]).try_into().unwrap();
                add_row16::<T, N, BD>(d, r);
            }
        }
        return;
    }
    let log2h = const { H.ilog2() as usize };
    let nz = nz_rows.min(ROWS);
    // The kernel reads at most the first `nz` (rounded up to its tier <=
    // ROWS; rows at or beyond nz_rows are zero) entries and overwrites all
    // `H`.
    let kernel = select::<S16<N>>(p.col_kind, log2h, nz);
    let clamp = Clamp::<S16<N>>::bits(16);
    if ROWS == H {
        if N == 8 && W >= 16 && w_active >= 16 {
            for j0 in (0..w_active / 16).map(|k| k * 16) {
                let pair = &mut buf.v[(j0 / 8) * H..][..2 * H];
                let (c0, c1) = pair.split_at_mut(H);
                let c0: &mut [S16<N>; H] = c0.try_into().unwrap();
                let c1: &mut [S16<N>; H] = c1.try_into().unwrap();
                run_col::<S16<N>, H>(p.col_kind, nz, kernel, c0, clamp);
                run_col::<S16<N>, H>(p.col_kind, nz, kernel, c1, clamp);
                add_cols16_pair::<T, N, W, H, BD>(dst, stride, j0, c0, c1, flip_lr, flip_ud);
            }
            if w_active % 16 != 0 {
                let j0 = w_active - 8;
                let c: &mut [S16<N>; H] = (&mut buf.v[(j0 / 8) * H..][..H]).try_into().unwrap();
                run_col::<S16<N>, H>(p.col_kind, nz, kernel, c, clamp);
                add_cols16_one::<T, N, W, H, BD>(dst, stride, j0, c, flip_lr, flip_ud);
            }
            return;
        }
        for j0 in (0..w_active / N).map(|k| k * N) {
            let c: &mut [S16<N>; H] = (&mut buf.v[(j0 / N) * H..][..H]).try_into().unwrap();
            run_col::<S16<N>, H>(p.col_kind, nz, kernel, c, clamp);
            add_cols16_one::<T, N, W, H, BD>(dst, stride, j0, c, flip_lr, flip_ud);
        }
    } else {
        if N == 8 && W >= 16 && w_active >= 16 {
            let mut tmp0 = [S16::<N>::splat(0); H];
            let mut tmp1 = [S16::<N>::splat(0); H];
            for j0 in (0..w_active / 16).map(|k| k * 16) {
                tmp0[..ROWS].copy_from_slice(&buf.v[(j0 / 8) * ROWS..][..ROWS]);
                tmp1[..ROWS].copy_from_slice(&buf.v[(j0 / 8 + 1) * ROWS..][..ROWS]);
                run_col::<S16<N>, H>(p.col_kind, nz, kernel, &mut tmp0, clamp);
                run_col::<S16<N>, H>(p.col_kind, nz, kernel, &mut tmp1, clamp);
                add_cols16_pair::<T, N, W, H, BD>(dst, stride, j0, &tmp0, &tmp1, flip_lr, flip_ud);
            }
            if w_active % 16 != 0 {
                let j0 = w_active - 8;
                tmp0[..ROWS].copy_from_slice(&buf.v[(j0 / 8) * ROWS..][..ROWS]);
                run_col::<S16<N>, H>(p.col_kind, nz, kernel, &mut tmp0, clamp);
                add_cols16_one::<T, N, W, H, BD>(dst, stride, j0, &tmp0, flip_lr, flip_ud);
            }
            return;
        }
        let mut tmp = [S16::<N>::splat(0); H];
        for j0 in (0..w_active / N).map(|k| k * N) {
            let g = &buf.v[(j0 / N) * ROWS..][..ROWS];
            tmp[..ROWS].copy_from_slice(g);
            run_col::<S16<N>, H>(p.col_kind, nz, kernel, &mut tmp, clamp);
            add_cols16_one::<T, N, W, H, BD>(dst, stride, j0, &tmp, flip_lr, flip_ud);
        }
    }
}
