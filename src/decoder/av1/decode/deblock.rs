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

//! Deblocking loop filter edge selection (spec 7.14.1 - 7.14.5) for one
//! superblock row. Sample filtering is done by `dsp::loopfilter`.

use std::simd::prelude::*;

use crate::decoder::av1::dsp::loopfilter::{filter_limits, kernel_filter_size, FilterLimits};
use crate::decoder::av1::pixel::Pixel;
use crate::decoder::av1::tables::block::{TX_HEIGHT, TX_WIDTH};
use crate::decoder::av1::types::*;

use super::buffers::{MiWin, PlaneWin};
use super::state::FrameCtx;

/// `lf_code(tx)` for every transform size (padded to 32 for branchless
/// indexing).
const LF_CODE: [u8; 32] = {
    let mut t = [0u8; 32];
    let mut i = 0;
    while i < TX_WIDTH.len() {
        t[i] = (((TX_WIDTH[i] >> 2) - 1) | (((TX_HEIGHT[i] >> 2) - 1) << 4)) as u8;
        i += 1;
    }
    t
};

/// The value stored in `MiWin::lf_tx` for a transform of size `tx`: its
/// width in 4x4 units minus one (`0, 1, 3, 7` or `15`) in the low nibble
/// and its height in 4x4 units minus one in the high nibble. A 4x4
/// column `c` is the first of its transform iff `c & (code & 15) == 0`
/// (transforms are aligned to their size), likewise for rows.
#[inline(always)]
pub fn lf_code(tx: u8) -> u8 {
    LF_CODE[(tx as usize) & 31]
}

#[inline(always)]
fn tx_w4(code: u8) -> u8 {
    (code & 15) + 1
}

#[inline(always)]
fn tx_h4(code: u8) -> u8 {
    (code >> 4) + 1
}

/// Precomputed `filter_limits(lvl, sharpness)` table for `sharpness` in `0..8`
/// and `lvl` in `0..64` (`lvl == 0` is unused and zeroed).
const LIMITS_LUT: [[FilterLimits; 64]; 8] = {
    let mut lut = [[FilterLimits {
        limit: 0,
        blimit: 0,
        thresh: 0,
    }; 64]; 8];
    let mut s = 0;
    while s < 8 {
        let mut lvl = 1;
        while lvl < 64 {
            lut[s][lvl] = filter_limits(lvl as i32, s as i32);
            lvl += 1;
        }
        s += 1;
    }
    lut
};

type V16 = Simd<u8, 16>;

/// Lane `i` is `i`.
const LANES: V16 = V16::from_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);

/// 16 `lf_tx` codes from `row[i..]` (the row has at least 16 entries of
/// slack past every used column).
#[inline(always)]
fn ld16(row: &[u8], i: usize) -> V16 {
    V16::from_slice(&row[i..i + 16])
}

/// The codes of the 4x4 columns left of the 16 columns `cur` from
/// `row[i..]` (column `i - 1` is only read if `i > 0`; lane 0 is a
/// don't care otherwise).
#[inline(always)]
fn ld16_left(row: &[u8], i: usize, cur: V16) -> V16 {
    if i > 0 {
        ld16(row, i - 1)
    } else {
        simd_swizzle!(cur, [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    }
}

/// Kernel sizes of the edges in front of 16 consecutive 4x4 positions
/// along a line (columns for vertical edges, rows for horizontal ones):
/// `cur` / `prev` are the extents minus one (`0, 1, 3, 7, 15`) across the
/// edge of the transforms at and before each position, `pos` the
/// position (low 4 bits) and `valid` the positions with an edge
/// candidate. Luma: `min(a4, b4, 4) * 4`, chroma: `4` if a transform is
/// one 4x4 wide, else `6` (spec 7.14.3 / 7.14.4).
#[inline(always)]
fn edge_sizes<const CHROMA: bool>(cur: V16, prev: V16, pos: V16, valid: Mask<i8, 16>) -> V16 {
    let start = (pos & cur).simd_eq(V16::splat(0)) & valid;
    let m = cur.simd_min(prev);
    let sz = if CHROMA {
        m.simd_eq(V16::splat(0))
            .select(V16::splat(4), V16::splat(6))
    } else {
        (m.simd_min(V16::splat(3)) + V16::splat(1)) << V16::splat(2)
    };
    start.select(sz, V16::splat(0))
}

/// Spec 7.14.5: filter level for the block at (row, col).
fn filter_level(
    f: &FrameCtx,
    mi: &MiWin,
    row: usize,
    col: usize,
    plane: usize,
    pass: usize,
) -> i32 {
    let fh = f.fh;
    let m = mi.get(row, col);
    let i = if plane == 0 { pass } else { plane + 1 };
    let delta_lf = if fh.delta_lf_present {
        let dlf = mi.delta_lf_at(row, col);
        (if fh.delta_lf_multi { dlf[i] } else { dlf[0] }) as i32
    } else {
        0
    };
    let base = (delta_lf + fh.lf.level[i] as i32).clamp(0, MAX_LOOP_FILTER);
    let mut lvl = base;
    let seg = m.segment_id as usize;
    let feature = SEG_LVL_ALT_LF_Y_V + i;
    if fh.seg.enabled && fh.seg.feature_enabled[seg][feature] {
        lvl = (fh.seg.feature_data[seg][feature] + lvl).clamp(0, MAX_LOOP_FILTER);
    }
    if fh.lf.delta_enabled {
        let n_shift = lvl >> 5;
        // All blocks of an intra frame use ref INTRA_FRAME.
        lvl += (fh.lf.ref_deltas[0] as i32) << n_shift;
        lvl = lvl.clamp(0, MAX_LOOP_FILTER);
    }
    lvl
}

/// Deblocks the mode-info rows `mi_row0..mi_row1` (one superblock row) of
/// all planes, vertical edges first, then horizontal edges. `grid` is
/// scratch space. Requires `mi_row1 - mi_row0 <= 32` (at most one 128x128
/// superblock row); larger ranges are only partially deblocked.
#[allow(clippy::too_many_arguments)]
pub fn deblock_rows<T: Pixel>(
    f: &FrameCtx,
    mi: &MiWin,
    planes: &mut [PlaneWin<T>],
    mi_row0: usize,
    mi_row1: usize,
    cols: (usize, usize),
    dsp: &crate::decoder::av1::dsp::Dsp<T>,
    grid: &mut Vec<u8>,
) {
    let fh = f.fh;
    if fh.lf.level[0] == 0 && fh.lf.level[1] == 0 {
        return;
    }
    let limits_lut = &LIMITS_LUT[(fh.lf.sharpness as usize) & 7];
    let mi_row1 = mi_row1.min(f.mi_rows);
    let col_end = cols.1.min(f.mi_cols);
    // Edges at or beyond the frame width / height are not filtered.
    let col_lim = col_end.min(fh.frame_width.div_ceil(MI_SIZE));
    let row_lim = mi_row1.min(fh.frame_height.div_ceil(MI_SIZE));
    let seg_lf = fh.seg.enabled
        && fh.seg.feature_enabled.iter().any(|f| {
            f[SEG_LVL_ALT_LF_Y_V..SEG_LVL_ALT_LF_Y_V + 4]
                .iter()
                .any(|&b| b)
        });
    if !seg_lf && !fh.delta_lf_present {
        deblock_rows_const(
            f, mi, planes, mi_row0, col_lim, row_lim, cols.0, dsp, limits_lut, grid,
        );
        return;
    }
    deblock_rows_var(
        f, mi, planes, mi_row0, col_lim, row_lim, cols.0, dsp, limits_lut, grid,
    );
}

#[allow(clippy::too_many_arguments)]
#[cold]
#[inline(never)]
fn deblock_rows_var<T: Pixel>(
    f: &FrameCtx,
    mi: &MiWin,
    planes: &mut [PlaneWin<T>],
    mi_row0: usize,
    col_lim: usize,
    row_lim: usize,
    col0: usize,
    dsp: &crate::decoder::av1::dsp::Dsp<T>,
    limits_lut: &[FilterLimits; 64],
    grid: &mut Vec<u8>,
) {
    let fh = f.fh;
    for plane in 0..f.num_planes {
        if plane > 0 && fh.lf.level[1 + plane] == 0 {
            continue;
        }
        let (sx, sy) = f.sub(plane);
        let pt = plane.min(1);
        let pl = &mut planes[plane];
        let stride = pl.stride;
        let y0 = pl.y0;
        let row_step = 1 << sy;
        let col_step = 1 << sx;
        let max_fs: u32 = if plane == 0 { 16 } else { 8 };
        // Level of the edge at luma mi (r, c) whose other side is (pr, pc).
        let lvl_at = |pass: usize, r: usize, c: usize, pr: usize, pc: usize| -> i32 {
            let l = filter_level(f, mi, r, c, plane, pass);
            if l == 0 {
                filter_level(f, mi, pr, pc, plane, pass)
            } else {
                l
            }
        };
        let n_c = col_lim.saturating_sub(col0).div_ceil(col_step);
        let n_r = row_lim.saturating_sub(mi_row0).div_ceil(row_step);
        if n_c == 0 || n_r == 0 {
            continue;
        }
        // One superblock row: at most 32 4x4 rows (see `deblock_rows`).
        debug_assert!(n_r <= 32);
        let n_r = n_r.min(32);
        let nv = 32 * n_c;
        let needed = 2 * nv + 3 * n_c;
        if grid.len() < needed {
            grid.resize(needed, 0);
        }
        let (vs, rest) = grid[..needed].split_at_mut(nv);
        let (vl, rest) = rest.split_at_mut(nv);
        let (col_has, rest) = rest.split_at_mut(n_c);
        let (hs, hl) = rest.split_at_mut(n_c);
        col_has.fill(0);
        let r4_0 = mi_row0 >> sy;
        let c4_0 = col0 >> sx;
        let yp0 = r4_0 * MI_SIZE - y0;
        let xp0 = c4_0 * MI_SIZE;
        for ri in 0..n_r {
            let row = mi_row0 + ri * row_step;
            let trow = mi.lf_tx_row(pt, r4_0 + ri);
            let mut prev_tw4 = if c4_0 > 0 { tx_w4(trow[c4_0 - 1]) } else { 0 };
            let mut ci = 0;
            while ci < n_c {
                let c4 = c4_0 + ci;
                let tx = trow[c4];
                let tw4 = tx_w4(tx);
                debug_assert_eq!(c4 & (tw4 as usize - 1), 0);
                if c4 != 0 {
                    let size = ((prev_tw4.min(tw4) as u32) << 2).min(max_fs);
                    let (r, c) = (row | sy, (c4 << sx) | sx);
                    let lvl = lvl_at(0, r, c, r, c - col_step);
                    if lvl != 0 {
                        let col_s = &mut vs[ci * 32..ci * 32 + 32];
                        let col_l = &mut vl[ci * 32..ci * 32 + 32];
                        if col_has[ci] == 0 {
                            col_s.copy_from_slice(&[0u8; 32]);
                            col_has[ci] = 1;
                        }
                        col_s[ri] = kernel_filter_size(plane, size) as u8;
                        col_l[ri] = lvl as u8;
                    }
                }
                prev_tw4 = tw4;
                ci += tw4 as usize;
            }
        }
        for ci in 0..n_c {
            if col_has[ci] != 0 {
                let xp = xp0 + ci * MI_SIZE;
                let col = ci * 32..ci * 32 + n_r;
                (dsp.lf_v_edge)(
                    pl.data_mut(),
                    stride,
                    xp,
                    yp0,
                    &vs[col.clone()],
                    &vl[col],
                    limits_lut,
                    f.bd,
                );
            }
        }
        // Horizontal edges.
        for ri in 0..n_r {
            let row = mi_row0 + ri * row_step;
            if row == 0 {
                continue;
            }
            let r4 = r4_0 + ri;
            let trow = mi.lf_tx_row(pt, r4);
            let prow = mi.lf_tx_row(pt, r4 - 1);
            hs.fill(0);
            let mut any = false;
            let mut ci = 0;
            while ci < n_c {
                let c4 = c4_0 + ci;
                let tx = trow[c4];
                let th4 = tx_h4(tx);
                if r4 & (th4 as usize - 1) == 0 {
                    let size = ((tx_h4(prow[c4]).min(th4) as u32) << 2).min(max_fs);
                    let (r, c) = (row | sy, (c4 << sx) | sx);
                    let lvl = lvl_at(1, r, c, r - row_step, c);
                    if lvl != 0 {
                        hs[ci] = kernel_filter_size(plane, size) as u8;
                        hl[ci] = lvl as u8;
                        any = true;
                    }
                    ci += 1;
                } else {
                    ci += tx_w4(tx) as usize;
                }
            }
            if any {
                (dsp.lf_h_edge)(
                    pl.data_mut(),
                    stride,
                    xp0,
                    r4 * MI_SIZE - y0,
                    hs,
                    hl,
                    limits_lut,
                    f.bd,
                );
            }
        }
    }
}

/// Vertical edge kernel sizes of the 4x4 rows `r4_0..r4_0 + n_r` and
/// columns `c4_0..c4_0 + n_c` of plane type `pt`, row major into `vs`
/// (`ncp`, a multiple of 16, per row). `any[ci]` is non-zero for the
/// columns with any edge.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn v_edge_sizes<const CHROMA: bool>(
    mi: &MiWin,
    pt: usize,
    r4_0: usize,
    n_r: usize,
    c4_0: usize,
    n_c: usize,
    ncp: usize,
    vs: &mut [u8],
    any: &mut [u8],
) {
    let k15 = V16::splat(15);
    let pos = (LANES + V16::splat((c4_0 & 15) as u8)) & k15;
    let any = &mut any[..ncp];
    any.fill(0);
    let all = Mask::<i8, 16>::splat(true);
    let last = LANES.simd_lt(V16::splat((n_c - (ncp - 16)) as u8));
    for ri in 0..n_r {
        let row = mi.lf_tx_row(pt, r4_0 + ri);
        let out = &mut vs[ri * ncp..][..ncp];
        let mut ci = 0;
        while ci < ncp {
            let c = c4_0 + ci;
            let cur = ld16(row, c);
            let prev = ld16_left(row, c, cur) & k15;
            let mut valid = if ci + 16 <= n_c { all } else { last };
            if c == 0 {
                // No edge left of the first column of the plane.
                valid &= LANES.simd_ne(V16::splat(0));
            }
            let sz = edge_sizes::<CHROMA>(cur & k15, prev, pos, valid);
            sz.copy_to_slice(&mut out[ci..ci + 16]);
            let a = &mut any[ci..ci + 16];
            (V16::from_slice(a) | sz).copy_to_slice(a);
            ci += 16;
        }
    }
}

/// Filters the vertical edges found by [`v_edge_sizes`] of the planes
/// `pls` (`(plane, level)`, level 0: skipped; columns from `xp0`, rows
/// from 4x4 row `r4_0`).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn v_edges_filter<T: Pixel>(
    planes: &mut [PlaneWin<T>],
    pls: &[(usize, u8)],
    dsp: &crate::decoder::av1::dsp::Dsp<T>,
    xp0: usize,
    r4_0: usize,
    n_r: usize,
    ncp: usize,
    vs: &[u8],
    any: &[u8],
    limits_lut: &[FilterLimits; 64],
    bd: u32,
) {
    let n_r = n_r.min(32);
    let mut c16 = 0;
    while c16 < ncp {
        let mut m = V16::from_slice(&any[c16..c16 + 16])
            .simd_ne(V16::splat(0))
            .to_bitmask();
        while m != 0 {
            let ci = c16 + m.trailing_zeros() as usize;
            m &= m - 1;
            let mut col = [0u8; 32];
            for (ri, s) in col[..n_r].iter_mut().enumerate() {
                *s = vs[ri * ncp + ci];
            }
            for &(p, lvl) in pls {
                if lvl == 0 {
                    continue;
                }
                let vl = [lvl; 32];
                let pl = &mut planes[p];
                let (stride, yp0) = (pl.stride, r4_0 * MI_SIZE - pl.y0);
                (dsp.lf_v_edge)(
                    pl.data_mut(),
                    stride,
                    xp0 + ci * MI_SIZE,
                    yp0,
                    &col[..n_r],
                    &vl[..n_r],
                    limits_lut,
                    bd,
                );
            }
        }
        c16 += 16;
    }
}

/// Horizontal edge kernel sizes of the 4x4 row `r4` (> 0) of plane type
/// `pt`, columns `c4_0..c4_0 + n_c`, into `hs` (`ncp` entries). Returns
/// whether there is any edge.
#[inline(always)]
fn h_edge_sizes<const CHROMA: bool>(
    mi: &MiWin,
    pt: usize,
    r4: usize,
    c4_0: usize,
    n_c: usize,
    ncp: usize,
    hs: &mut [u8],
) -> bool {
    let trow = mi.lf_tx_row(pt, r4);
    let prow = mi.lf_tx_row(pt, r4 - 1);
    let pos = V16::splat((r4 & 15) as u8);
    let k4 = V16::splat(4);
    let hs = &mut hs[..ncp];
    let mut acc = V16::splat(0);
    let last = LANES.simd_lt(V16::splat((n_c - (ncp - 16)) as u8));
    let mut ci = 0;
    while ci < ncp {
        let c = c4_0 + ci;
        let cur = ld16(trow, c) >> k4;
        let prev = ld16(prow, c) >> k4;
        let valid = if ci + 16 <= n_c { Mask::splat(true) } else { last };
        let sz = edge_sizes::<CHROMA>(cur, prev, pos, valid);
        sz.copy_to_slice(&mut hs[ci..ci + 16]);
        acc |= sz;
        ci += 16;
    }
    acc.simd_ne(V16::splat(0)).any()
}

/// Fast path when no segment has a loop filter feature (`SEG_LVL_ALT_LF_*`)
/// and delta LF is disabled: filter levels
/// are constant per (plane, pass), and U and V share the same chroma
/// transform edge sizes. The edge sizes are computed for 16 positions at
/// a time from the `lf_code`s.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn deblock_rows_const<T: Pixel>(
    f: &FrameCtx,
    mi: &MiWin,
    planes: &mut [PlaneWin<T>],
    mi_row0: usize,
    col_lim: usize,
    row_lim: usize,
    col0: usize,
    dsp: &crate::decoder::av1::dsp::Dsp<T>,
    limits_lut: &[FilterLimits; 64],
    grid: &mut Vec<u8>,
) {
    let fh = f.fh;
    let const_lvl = |plane: usize, pass: usize| -> u8 {
        let i = if plane == 0 { pass } else { plane + 1 };
        if plane > 0 && fh.lf.level[i] == 0 {
            return 0;
        }
        let mut l = (fh.lf.level[i] as i32).clamp(0, MAX_LOOP_FILTER);
        if fh.lf.delta_enabled {
            l = (l + ((fh.lf.ref_deltas[0] as i32) << (l >> 5))).clamp(0, MAX_LOOP_FILTER);
        }
        l as u8
    };

    // Luma (plane 0).
    let n_c = col_lim.saturating_sub(col0);
    let n_r = row_lim.saturating_sub(mi_row0);
    if n_c > 0 && n_r > 0 {
        debug_assert!(n_r <= 32);
        let n_r = n_r.min(32);
        let lvl_v = const_lvl(0, 0);
        let lvl_h = const_lvl(0, 1);
        let ncp = n_c.next_multiple_of(16);
        let needed = (n_r + 3) * ncp;
        if grid.len() < needed {
            grid.resize(needed, 0);
        }
        let (vs, rest) = grid[..needed].split_at_mut(n_r * ncp);
        let (any, rest) = rest.split_at_mut(ncp);
        let (hs, hl) = rest.split_at_mut(ncp);
        let xp0 = col0 * MI_SIZE;
        if lvl_v != 0 {
            v_edge_sizes::<false>(mi, 0, mi_row0, n_r, col0, n_c, ncp, vs, any);
            v_edges_filter(
                planes,
                &[(0, lvl_v)],
                dsp,
                xp0,
                mi_row0,
                n_r,
                ncp,
                vs,
                any,
                limits_lut,
                f.bd,
            );
        }
        if lvl_h != 0 {
            hl.fill(lvl_h);
            let pl = &mut planes[0];
            let (stride, y0) = (pl.stride, pl.y0);
            for ri in 0..n_r {
                let r4 = mi_row0 + ri;
                if r4 == 0 {
                    continue;
                }
                if h_edge_sizes::<false>(mi, 0, r4, col0, n_c, ncp, hs) {
                    (dsp.lf_h_edge)(
                        pl.data_mut(),
                        stride,
                        xp0,
                        r4 * MI_SIZE - y0,
                        &hs[..n_c],
                        &hl[..n_c],
                        limits_lut,
                        f.bd,
                    );
                }
            }
        }
    }

    // Chroma (planes 1 and 2 share edge sizes).
    if f.num_planes <= 1 {
        return;
    }
    let lvl_u = const_lvl(1, 0);
    let lvl_v = const_lvl(2, 0);
    if lvl_u == 0 && lvl_v == 0 {
        return;
    }
    let (sx, sy) = (f.ss_x, f.ss_y);
    let n_c = col_lim.saturating_sub(col0).div_ceil(1 << sx);
    let n_r = row_lim.saturating_sub(mi_row0).div_ceil(1 << sy);
    if n_c == 0 || n_r == 0 {
        return;
    }
    debug_assert!(n_r <= 32);
    let n_r = n_r.min(32);
    let ncp = n_c.next_multiple_of(16);
    let needed = (n_r + 4) * ncp;
    if grid.len() < needed {
        grid.resize(needed, 0);
    }
    let (vs, rest) = grid[..needed].split_at_mut(n_r * ncp);
    let (any, rest) = rest.split_at_mut(ncp);
    let (hs, rest) = rest.split_at_mut(ncp);
    let (hl_u, hl_v) = rest.split_at_mut(ncp);
    let r4_0 = mi_row0 >> sy;
    let c4_0 = col0 >> sx;
    let xp0 = c4_0 * MI_SIZE;
    v_edge_sizes::<true>(mi, 1, r4_0, n_r, c4_0, n_c, ncp, vs, any);
    v_edges_filter(
        planes,
        &[(1, lvl_u), (2, lvl_v)],
        dsp,
        xp0,
        r4_0,
        n_r,
        ncp,
        vs,
        any,
        limits_lut,
        f.bd,
    );
    hl_u.fill(lvl_u);
    hl_v.fill(lvl_v);
    for ri in 0..n_r {
        let r4 = r4_0 + ri;
        if r4 == 0 || !h_edge_sizes::<true>(mi, 1, r4, c4_0, n_c, ncp, hs) {
            continue;
        }
        let yp = r4 * MI_SIZE;
        for (plane, lvl, hl) in [(1usize, lvl_u, &*hl_u), (2usize, lvl_v, &*hl_v)] {
            if lvl != 0 {
                let pl = &mut planes[plane];
                let (stride, y) = (pl.stride, yp - pl.y0);
                (dsp.lf_h_edge)(
                    pl.data_mut(),
                    stride,
                    xp0,
                    y,
                    &hs[..n_c],
                    &hl[..n_c],
                    limits_lut,
                    f.bd,
                );
            }
        }
    }
}
