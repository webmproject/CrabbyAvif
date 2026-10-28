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

use crate::decoder::av1::dsp::loopfilter::{filter_limits, kernel_filter_size, FilterLimits};
use crate::decoder::av1::pixel::Pixel;
use crate::decoder::av1::tables::block::{TX_HEIGHT, TX_WIDTH};
use crate::decoder::av1::types::*;

use super::buffers::{MiWin, PlaneWin};
use super::state::FrameCtx;

/// Transform widths and heights in 4x4 units.
const TX_W4: [u8; TX_WIDTH.len()] = {
    let mut t = [0u8; TX_WIDTH.len()];
    let mut i = 0;
    while i < TX_WIDTH.len() {
        t[i] = (TX_WIDTH[i] >> 2) as u8;
        i += 1;
    }
    t
};

const TX_H4: [u8; TX_HEIGHT.len()] = {
    let mut t = [0u8; TX_HEIGHT.len()];
    let mut i = 0;
    while i < TX_HEIGHT.len() {
        t[i] = (TX_HEIGHT[i] >> 2) as u8;
        i += 1;
    }
    t
};

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

#[inline(always)]
fn ksize_luma(a4: u8, b4: u8) -> u8 {
    a4.min(b4).min(4) << 2
}

#[inline(always)]
fn ksize_chroma(a4: u8, b4: u8) -> u8 {
    if a4.min(b4) == 1 {
        4
    } else {
        6
    }
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
/// scratch space.
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
    if !fh.seg.enabled && !fh.delta_lf_present {
        deblock_rows_const(
            f, mi, planes, mi_row0, col_lim, row_lim, cols.0, dsp, limits_lut, grid,
        );
        return;
    }
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
        let n_c = col_lim.saturating_sub(cols.0).div_ceil(col_step);
        let n_r = row_lim.saturating_sub(mi_row0).div_ceil(row_step);
        if n_c == 0 || n_r == 0 {
            continue;
        }
        let nv = n_r * n_c;
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
        let c4_0 = cols.0 >> sx;
        let yp0 = r4_0 * MI_SIZE - y0;
        let xp0 = c4_0 * MI_SIZE;
        for ri in 0..n_r {
            let row = mi_row0 + ri * row_step;
            let trow = mi.lf_tx_row(pt, r4_0 + ri);
            let mut prev_tx = if c4_0 > 0 { trow[c4_0 - 1] } else { 0 };
            let mut ci = 0;
            while ci < n_c {
                let c4 = c4_0 + ci;
                let tx = trow[c4];
                let tw4 = TX_W4[tx as usize] as usize;
                debug_assert_eq!(c4 & (tw4 - 1), 0);
                if c4 != 0 {
                    let size = (TX_WIDTH[prev_tx as usize] as u32)
                        .min((tw4 * 4) as u32)
                        .min(max_fs);
                    let (r, c) = (row | sy, (c4 << sx) | sx);
                    let lvl = lvl_at(0, r, c, r, c - col_step);
                    if lvl != 0 {
                        if col_has[ci] == 0 {
                            vs[ci * n_r..(ci + 1) * n_r].fill(0);
                            col_has[ci] = 1;
                        }
                        vs[ci * n_r + ri] = kernel_filter_size(plane, size) as u8;
                        vl[ci * n_r + ri] = lvl as u8;
                    }
                }
                prev_tx = tx;
                ci += tw4;
            }
        }
        for ci in 0..n_c {
            if col_has[ci] != 0 {
                let xp = xp0 + ci * MI_SIZE;
                let col = ci * n_r..(ci + 1) * n_r;
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
                let th4 = TX_H4[tx as usize] as usize;
                if r4 & (th4 - 1) == 0 {
                    let size = (TX_HEIGHT[prow[c4] as usize] as u32)
                        .min((th4 * 4) as u32)
                        .min(max_fs);
                    let (r, c) = (row | sy, (c4 << sx) | sx);
                    let lvl = lvl_at(1, r, c, r - row_step, c);
                    if lvl != 0 {
                        hs[ci] = kernel_filter_size(plane, size) as u8;
                        hl[ci] = lvl as u8;
                        any = true;
                    }
                    ci += 1;
                } else {
                    ci += TX_W4[tx as usize] as usize;
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

/// Fast path when segmentation and delta LF are disabled: filter levels
/// are constant per (plane, pass), and U and V share the same chroma
/// transform edge sizes.
#[allow(clippy::too_many_arguments)]
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
        let lvl_v = const_lvl(0, 0);
        let lvl_h = const_lvl(0, 1);
        let nv = n_r * n_c;
        let needed = nv + 3 * n_c;
        if grid.len() < needed {
            grid.resize(needed, 0);
        }
        let (vs, rest) = grid[..needed].split_at_mut(nv);
        let (col_has, rest) = rest.split_at_mut(n_c);
        let (hs, hl) = rest.split_at_mut(n_c);
        let pl = &mut planes[0];
        let stride = pl.stride;
        let y0 = pl.y0;
        let yp0 = mi_row0 * MI_SIZE - y0;
        let xp0 = col0 * MI_SIZE;
        if lvl_v != 0 {
            col_has.fill(0);
            for ri in 0..n_r {
                let trow = mi.lf_tx_row(0, mi_row0 + ri);
                let mut prev_tw4 = if col0 > 0 { TX_W4[trow[col0 - 1] as usize] } else { 0 };
                let mut ci = 0;
                while ci < n_c {
                    let c4 = col0 + ci;
                    let tw4 = TX_W4[trow[c4] as usize];
                    debug_assert_eq!(c4 & (tw4 as usize - 1), 0);
                    if c4 != 0 {
                        if col_has[ci] == 0 {
                            vs[ci * n_r..(ci + 1) * n_r].fill(0);
                            col_has[ci] = 1;
                        }
                        vs[ci * n_r + ri] = ksize_luma(prev_tw4, tw4);
                    }
                    prev_tw4 = tw4;
                    ci += tw4 as usize;
                }
            }
            let vl = [lvl_v; 32];
            for ci in 0..n_c {
                if col_has[ci] != 0 {
                    let col = ci * n_r..(ci + 1) * n_r;
                    (dsp.lf_v_edge)(
                        pl.data_mut(),
                        stride,
                        xp0 + ci * MI_SIZE,
                        yp0,
                        &vs[col],
                        &vl[..n_r],
                        limits_lut,
                        f.bd,
                    );
                }
            }
        }
        if lvl_h != 0 {
            hl.fill(lvl_h);
            for ri in 0..n_r {
                let r4 = mi_row0 + ri;
                if r4 == 0 {
                    continue;
                }
                let trow = mi.lf_tx_row(0, r4);
                let prow = mi.lf_tx_row(0, r4 - 1);
                hs.fill(0);
                let mut any = false;
                let mut ci = 0;
                while ci < n_c {
                    let c4 = col0 + ci;
                    let tx = trow[c4] as usize;
                    let tw4 = TX_W4[tx] as usize;
                    let th4 = TX_H4[tx];
                    if r4 & (th4 as usize - 1) == 0 {
                        let end = (ci + tw4).min(n_c);
                        while ci < end {
                            let c4 = col0 + ci;
                            let ptx = prow[c4] as usize;
                            let ptw4 = TX_W4[ptx] as usize;
                            let pth4 = TX_H4[ptx];
                            let p_end = (ci + ptw4 - (c4 & (ptw4 - 1))).min(end);
                            hs[ci..p_end].fill(ksize_luma(pth4, th4));
                            ci = p_end;
                        }
                        any = true;
                    } else {
                        ci += tw4;
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
    let nv = n_r * n_c;
    let needed = nv + 4 * n_c;
    if grid.len() < needed {
        grid.resize(needed, 0);
    }
    let (vs, rest) = grid[..needed].split_at_mut(nv);
    let (col_has, rest) = rest.split_at_mut(n_c);
    let (hs, rest) = rest.split_at_mut(n_c);
    let (hl_u, hl_v) = rest.split_at_mut(n_c);
    col_has.fill(0);
    if lvl_u != 0 {
        hl_u.fill(lvl_u);
    }
    if lvl_v != 0 {
        hl_v.fill(lvl_v);
    }
    let r4_0 = mi_row0 >> sy;
    let c4_0 = col0 >> sx;
    let xp0 = c4_0 * MI_SIZE;
    for ri in 0..n_r {
        let trow = mi.lf_tx_row(1, r4_0 + ri);
        let mut prev_tw4 = if c4_0 > 0 { TX_W4[trow[c4_0 - 1] as usize] } else { 0 };
        let mut ci = 0;
        while ci < n_c {
            let c4 = c4_0 + ci;
            let tw4 = TX_W4[trow[c4] as usize];
            debug_assert_eq!(c4 & (tw4 as usize - 1), 0);
            if c4 != 0 {
                if col_has[ci] == 0 {
                    vs[ci * n_r..(ci + 1) * n_r].fill(0);
                    col_has[ci] = 1;
                }
                vs[ci * n_r + ri] = ksize_chroma(prev_tw4, tw4);
            }
            prev_tw4 = tw4;
            ci += tw4 as usize;
        }
    }
    for (plane, lvl) in [(1usize, lvl_u), (2usize, lvl_v)] {
        if lvl == 0 {
            continue;
        }
        let vl = [lvl; 32];
        let pl = &mut planes[plane];
        let stride = pl.stride;
        let yp0 = r4_0 * MI_SIZE - pl.y0;
        for ci in 0..n_c {
            if col_has[ci] != 0 {
                let col = ci * n_r..(ci + 1) * n_r;
                (dsp.lf_v_edge)(
                    pl.data_mut(),
                    stride,
                    xp0 + ci * MI_SIZE,
                    yp0,
                    &vs[col],
                    &vl[..n_r],
                    limits_lut,
                    f.bd,
                );
            }
        }
    }
    for ri in 0..n_r {
        let r4 = r4_0 + ri;
        if r4 == 0 {
            continue;
        }
        let trow = mi.lf_tx_row(1, r4);
        let prow = mi.lf_tx_row(1, r4 - 1);
        hs.fill(0);
        let mut any = false;
        let mut ci = 0;
        while ci < n_c {
            let c4 = c4_0 + ci;
            let tx = trow[c4] as usize;
            let tw4 = TX_W4[tx] as usize;
            let th4 = TX_H4[tx];
            if r4 & (th4 as usize - 1) == 0 {
                let end = (ci + tw4).min(n_c);
                while ci < end {
                    let c4 = c4_0 + ci;
                    let ptx = prow[c4] as usize;
                    let ptw4 = TX_W4[ptx] as usize;
                    let pth4 = TX_H4[ptx];
                    let p_end = (ci + ptw4 - (c4 & (ptw4 - 1))).min(end);
                    hs[ci..p_end].fill(ksize_chroma(pth4, th4));
                    ci = p_end;
                }
                any = true;
            } else {
                ci += tw4;
            }
        }
        if any {
            let yp = r4 * MI_SIZE;
            if lvl_u != 0 {
                let pl = &mut planes[1];
                let (stride, y) = (pl.stride, yp - pl.y0);
                (dsp.lf_h_edge)(pl.data_mut(), stride, xp0, y, hs, hl_u, limits_lut, f.bd);
            }
            if lvl_v != 0 {
                let pl = &mut planes[2];
                let (stride, y) = (pl.stride, yp - pl.y0);
                (dsp.lf_h_edge)(pl.data_mut(), stride, xp0, y, hs, hl_v, limits_lut, f.bd);
            }
        }
    }
}
