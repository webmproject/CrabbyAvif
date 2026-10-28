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

//! Reconstruction helpers of the superblock decoder: intra edge
//! preparation (spec 7.11.2.1), palette (7.11.4), chroma from luma
//! (7.11.5) and inverse transform dispatch (7.12.3).

use crate::decoder::av1::dsp::ipred::{ANGLE_STEP, MAX_EDGE, MODE_TO_ANGLE};
use crate::decoder::av1::pixel::{round2, Pixel};
use crate::decoder::av1::types::*;

use super::block::{txh, txw, SbDecoder};

impl<'a, 'b, T: Pixel> SbDecoder<'a, 'b, T> {
    /// Reads a reconstructed (pre loop filter) sample of the current frame.
    /// Rows above the current superblock row come from the saved line.
    #[inline(always)]
    fn px(&self, plane: usize, x: usize, y: usize) -> T {
        let sb_top = self.sb_y >> if plane > 0 { self.f.ss_y } else { 0 };
        if y < sb_top && !self.s.above_line[plane].is_empty() {
            debug_assert_eq!(y + 1, sb_top);
            self.s.above_line[plane][x - self.s.planes[plane].x0]
        } else {
            self.s.planes[plane].get(x, y)
        }
    }

    /// Spec 7.11.2.8 `get_filter_type`.
    fn get_filter_type(&self, plane: usize) -> bool {
        let b = &self.b;
        let is_smooth = |r: usize, c: usize| -> bool {
            let mi = self.s.mi.get(r, c);
            let mode = if plane == 0 { mi.y_mode } else { mi.uv_mode };
            mode.wrapping_sub(IntraMode::SMOOTH_PRED as u8) < 3
        };
        if plane == 0 {
            // (mi_row - 1, mi_col) and (mi_row, mi_col - 1) are the block's
            // above / left mode info, already loaded.
            let sm = |m: u8| m.wrapping_sub(IntraMode::SMOOTH_PRED as u8) < 3;
            return (b.avail_u && sm(b.above_mi.y_mode)) || (b.avail_l && sm(b.left_mi.y_mode));
        }
        let (sx, sy) = (self.f.ss_x, self.f.ss_y);
        if b.avail_u_chroma && is_smooth((b.mi_row & !sy) - 1, b.mi_col | sx) {
            return true;
        }
        b.avail_l_chroma && is_smooth(b.mi_row | sy, (b.mi_col & !sx) - 1)
    }

    /// Computes the intra prediction setup of the current block
    /// (`s.ipred`, the parts of spec 7.11.2 that are the same for all
    /// transform blocks of a plane type). Called once per intra block
    /// before its transform blocks are predicted.
    pub(super) fn setup_intra(&mut self) {
        let b = &self.b;
        if b.palette_size_y == 0 {
            let y_tx = if b.lossless { TxSize::Tx4x4 } else { b.tx_size };
            let fi = if b.use_filter_intra { Some(b.filter_intra_mode) } else { None };
            let (mode, delta) = (b.y_mode, b.angle_delta_y);
            self.intra_setup(0, mode, delta, fi, y_tx);
        }
        let b = &self.b;
        if b.has_chroma && b.palette_size_uv == 0 {
            let uv_tx =
                if b.lossless { TxSize::Tx4x4 } else { self.f.uv_tx_size[b.bsize as usize] };
            let mode =
                if b.uv_mode == IntraMode::UV_CFL_PRED { IntraMode::DC_PRED } else { b.uv_mode };
            let delta = b.angle_delta_uv;
            self.intra_setup(1, mode, delta, None, uv_tx);
        }
    }

    /// Writes the per-block fields of `s.ipred[plane != 0]` in place (the
    /// per-frame ones are set by `IntraSetup::new`).
    #[inline(always)]
    fn intra_setup(
        &mut self,
        plane: usize,
        mode: IntraMode,
        angle_delta: i32,
        filter_intra_mode: Option<u8>,
        tx_sz: TxSize,
    ) {
        let f = self.f;
        let w = txw(tx_sz);
        let h = txh(tx_sz);
        let use_filter = filter_intra_mode.is_some();
        // Edge samples read with both neighbors available (DC_PRED reads
        // only the available ones).
        let (na, nl, need_filter) = if use_filter {
            (w, h, false)
        } else {
            match mode {
                IntraMode::V_PRED
                | IntraMode::H_PRED
                | IntraMode::D45_PRED
                | IntraMode::D135_PRED
                | IntraMode::D113_PRED
                | IntraMode::D157_PRED
                | IntraMode::D203_PRED
                | IntraMode::D67_PRED => {
                    let p_angle = MODE_TO_ANGLE[mode as usize] + angle_delta * ANGLE_STEP;
                    match p_angle {
                        90 => (w, 0, false),
                        180 => (0, h, false),
                        a if a < 90 => (w + h, 0, true),
                        a if a < 180 => (w, h, true),
                        _ => (0, w + h, true),
                    }
                }
                _ => (w, h, false),
            }
        };
        let smooth_filter_type =
            need_filter && f.seq.enable_intra_edge_filter && self.get_filter_type(plane);
        let su = &mut self.s.ipred[(plane != 0) as usize];
        su.p.mode = mode as u8;
        su.p.angle_delta = angle_delta;
        su.p.filter_intra_mode = filter_intra_mode;
        su.p.smooth_filter_type = smooth_filter_type;
        su.p.w = w;
        su.p.h = h;
        su.na = na;
        su.nl = nl;
        su.dc = !use_filter && mode == IntraMode::DC_PRED;
        su.need_top_left = use_filter || need_filter || mode == IntraMode::PAETH_PRED;
    }

    /// Spec 7.11.2 for one transform block, using the block's
    /// `setup_intra`. Only the edge samples the selected predictor reads
    /// are derived (`ipred::edge_needs`, as dav1d's prepare_intra_edges).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn predict_intra(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        have_left: bool,
        have_above: bool,
        have_above_right: bool,
        have_below_left: bool,
    ) {
        let pt = (plane != 0) as usize;
        let sb_top = self.sb_y >> (self.f.ss_y & pt);
        let s = &mut *self.s;
        let su = &mut s.ipred[pt];
        let (w, h) = (su.p.w, su.p.h);
        let (na, nl) = if su.dc {
            (
                if have_above { su.na } else { 0 },
                if have_left { su.nl } else { 0 },
            )
        } else {
            (su.na, su.nl)
        };
        let need_top_left = su.need_top_left;
        let max_x = su.p.max_x as usize;
        let max_y = su.p.max_y as usize;
        let bd = su.p.bd;
        su.p.x = x as i32;
        su.p.y = y as i32;
        su.p.have_left = have_left;
        su.p.have_above = have_above;
        let e = &mut s.edges;
        let pl = &mut s.planes[plane];
        let stride = pl.stride & 0x3_ffff;
        let col_off = x - pl.x0;
        let row_off = (y - pl.y0) * stride;
        let dst_off = row_off + col_off;
        let full_stride = pl.stride;
        let data = pl.data_mut();
        if have_above && have_left {
            // Common case (both neighbors available): `above[0]` is the
            // top-left sample, `above[1..]` row y - 1 from column x.
            let above: &[T] = if y == sb_top && !s.above_line[plane].is_empty() {
                &s.above_line[plane][col_off - 1..]
            } else {
                let a_off = row_off - stride + col_off - 1;
                &data[a_off..a_off + stride]
            };
            if need_top_left {
                e.top_left = above[0];
            }
            if na != 0 {
                // Samples x..=min(max_x, x + n - 1) are read, the rest
                // replicates the last one.
                let m = (max_x + 1 - x).min(w << have_above_right as u32).min(na);
                copy_above_edge(&mut e.above, &above[1..], m);
                if m < na {
                    fill_edge(&mut e.above, m, na, above[m]);
                }
            }
            if nl != 0 {
                let m = (max_y + 1 - y).min(h << have_below_left as u32).min(nl);
                gather_left_edge(&mut e.left, data, dst_off - 1, stride, m);
                if m < nl {
                    let v = e.left[m - 1];
                    fill_edge(&mut e.left, m, nl, v);
                }
            }
            (self.dsp.predict_intra)(&mut data[dst_off..], full_stride, e, &su.p);
            return;
        }
        // Row y - 1: the saved unfiltered line at the top of the
        // superblock row, the current window otherwise (see `px`).
        let above_row: &[T] = if !have_above || (na == 0 && have_left && !need_top_left) {
            &[]
        } else if y - 1 < sb_top && !s.above_line[plane].is_empty() {
            debug_assert_eq!(y, sb_top);
            &s.above_line[plane]
        } else {
            let a_off = row_off - stride;
            &data[a_off..a_off + stride]
        };
        if na == 0 {
        } else if !have_above {
            let v = if have_left { data[dst_off - 1] } else { T::from_i32((1 << (bd - 1)) - 1) };
            fill_edge(&mut e.above, 0, na, v);
        } else {
            let above_limit = max_x.min(x + if have_above_right { 2 * w } else { w } - 1);
            let row = &above_row[col_off..];
            // Samples x..=above_limit are read, the rest replicates the
            // last one.
            let m = (above_limit + 1 - x).min(na);
            copy_above_edge(&mut e.above, row, m);
            if m < na {
                fill_edge(&mut e.above, m, na, row[m - 1]);
            }
        }
        if nl == 0 {
        } else if !have_left {
            let v = if have_above { above_row[col_off] } else { T::from_i32((1 << (bd - 1)) + 1) };
            fill_edge(&mut e.left, 0, nl, v);
        } else {
            let left_limit = max_y.min(y + if have_below_left { 2 * h } else { h } - 1);
            let m = (left_limit + 1 - y).min(nl);
            gather_left_edge(&mut e.left, data, dst_off - 1, stride, m);
            if m < nl {
                let v = e.left[m - 1];
                fill_edge(&mut e.left, m, nl, v);
            }
        }
        if need_top_left {
            e.top_left = if have_above && have_left {
                above_row[col_off - 1]
            } else if have_above {
                above_row[col_off]
            } else if have_left {
                data[dst_off - 1]
            } else {
                T::from_i32(1 << (bd - 1))
            };
        }
        (self.dsp.predict_intra)(&mut data[dst_off..], full_stride, e, &su.p);
    }

    /// Spec 7.11.4.
    pub(super) fn predict_palette(
        &mut self,
        plane: usize,
        start_x: usize,
        start_y: usize,
        x: usize,
        y: usize,
        tx_sz: TxSize,
    ) {
        let w = txw(tx_sz);
        let h = txh(tx_sz);
        let palette = match plane {
            0 => self.b.palette_y,
            1 => self.b.palette_u,
            _ => self.b.palette_v,
        };
        let map = if plane == 0 { &self.sc.color_map_y } else { &self.sc.color_map_uv };
        let pl = &mut self.s.planes[plane];
        let stride = pl.stride;
        let mut dst = pl.idx(start_x, start_y);
        let mx = x * 4;
        let my = y * 4;
        for mrow in &map[my..my + h] {
            let row = &mut pl.data_mut()[dst..dst + w];
            for (d, &idx) in row.iter_mut().zip(&mrow[mx..mx + w]) {
                *d = T::from_i32(palette[(idx & 7) as usize] as i32);
            }
            dst += stride;
        }
    }

    /// Spec 7.11.5.
    #[inline(never)]
    pub(super) fn predict_cfl(
        &mut self,
        plane: usize,
        start_x: usize,
        start_y: usize,
        tx_sz: TxSize,
    ) {
        let alpha = if plane == 1 { self.b.cfl_alpha_u } else { self.b.cfl_alpha_v };
        if alpha == 0 {
            return;
        }
        let f = self.f;
        let w = txw(tx_sz);
        let h = txh(tx_sz);
        let s = &mut *self.s;
        if plane == 1 || self.b.cfl_alpha_u == 0 {
            let (sx, sy) = (f.ss_x, f.ss_y);
            let max_ly = self.b.max_luma_h - (1 << sy);
            let max_lx = self.b.max_luma_w - (1 << sx);
            let (lx0, ly0) = (start_x << sx, start_y << sy);
            // Columns j < wv (rows i < hv) read luma column lx0 + (j << sx)
            // <= max_lx (row ly0 + (i << sy) <= max_ly); the remaining ones
            // are clamped to max_lx (max_ly), i.e. replicate the last one.
            let wv = if lx0 > max_lx { 0 } else { (((max_lx - lx0) >> sx) + 1).min(w) };
            let hv = if ly0 > max_ly { 0 } else { (((max_ly - ly0) >> sy) + 1).min(h) };
            let ac = &mut s.cfl_ac[..w * h];
            let luma = &s.planes[0];
            if wv > 0 && hv > 0 {
                (self.dsp.cfl_ac)(ac, luma.slice(lx0, ly0), luma.stride, w, h, wv, hv, sx, sy);
            } else {
                cfl_ac_corrupt(ac, luma, start_x, start_y, w, h, sx, sy, max_lx, max_ly);
            }
        }
        let ac = &s.cfl_ac[..w * h];
        let pl = &mut s.planes[plane];
        let stride = pl.stride;
        (self.dsp.cfl_pred)(
            pl.slice_mut(start_x, start_y),
            stride,
            ac,
            w,
            h,
            alpha,
            f.bd,
        );
    }

    /// Spec 7.12.3 steps 2-3 (coefficients are already dequantized).
    pub(super) fn reconstruct(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        tx_sz: TxSize,
        eob: usize,
    ) {
        let tw = txw(tx_sz).min(32);
        let th = txh(tx_sz).min(32);
        let pl = &mut self.s.planes[plane];
        let stride = pl.stride;
        // Also leaves the coefficients zero for the next block.
        (self.dsp.inv_txfm_add)(
            pl.slice_mut(x, y),
            stride,
            &mut self.sc.coeffs[..tw * th],
            tx_sz,
            self.plane_tx_type,
            self.b.lossless,
            eob,
            self.f.bd,
        );
    }

    pub(super) fn clear_coeffs(&mut self, tx_sz: TxSize) {
        let tw = txw(tx_sz).min(32);
        let th = txh(tx_sz).min(32);
        self.sc.coeffs[..tw * th].fill(0);
    }
}

/// Only reachable with corrupt streams (the clamped luma position lies before the block).
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn cfl_ac_corrupt<T: Pixel>(
    ac: &mut [i16],
    luma: &super::buffers::PlaneWin<T>,
    start_x: usize,
    start_y: usize,
    w: usize,
    h: usize,
    sx: usize,
    sy: usize,
    max_lx: usize,
    max_ly: usize,
) {
    let mut sum = 0i32;
    for i in 0..h {
        let ly = ((start_y + i) << sy).min(max_ly);
        for j in 0..w {
            let lx = ((start_x + j) << sx).min(max_lx);
            let mut t = 0;
            for dy in 0..=sy {
                for dx in 0..=sx {
                    t += luma.get(lx + dx, ly + dy).to_i32();
                }
            }
            let v = t << (3 - sx - sy);
            ac[i * w + j] = v as i16;
            sum += v;
        }
    }
    let avg = round2(sum, w.trailing_zeros() + h.trailing_zeros());
    for a in ac.iter_mut() {
        *a = (*a as i32 - avg) as i16;
    }
}

/// Copies `row[..m]` into `dst[..m]` using power-of-two vector copies
/// (samples past `m` when `m <= 16` are overwritten by `fill_edge` or unused;
/// `row` always has at least 64 padding samples past `max_x`).
#[inline(always)]
fn copy_above_edge<T: Pixel>(dst: &mut [T; MAX_EDGE], row: &[T], m: usize) {
    debug_assert!((1..=MAX_EDGE).contains(&m));
    if m <= 4 {
        dst[..4].copy_from_slice(&row[..4]);
        return;
    }
    if m <= 8 {
        dst[..8].copy_from_slice(&row[..8]);
        return;
    }
    if m <= 16 {
        dst[..16].copy_from_slice(&row[..16]);
        return;
    }
    let src = &row[..m];
    let dst = &mut dst[..m];
    if m <= 32 {
        dst[..16].copy_from_slice(&src[..16]);
        dst[m - 16..].copy_from_slice(&src[m - 16..]);
    } else if m <= 64 {
        dst[..32].copy_from_slice(&src[..32]);
        dst[m - 32..].copy_from_slice(&src[m - 32..]);
    } else {
        dst[..64].copy_from_slice(&src[..64]);
        dst[m - 64..].copy_from_slice(&src[m - 64..]);
    }
}

#[inline(always)]
fn gather4<T: Pixel>(dst: &mut [T], data: &[T], base: usize, s1: usize, s2: usize, s3: usize) {
    let col = &data[base..base + s3 + 1];
    dst[..4].copy_from_slice(&[col[0], col[s1], col[s2], col[s3]]);
}

#[inline(always)]
fn gather8<T: Pixel>(
    dst: &mut [T],
    data: &[T],
    base: usize,
    s1: usize,
    s2: usize,
    s3: usize,
    s4: usize,
) {
    let col = &data[base..base + s4 + s3 + 1];
    let col1 = &col[s4..];
    dst[0..4].copy_from_slice(&[col[0], col[s1], col[s2], col[s3]]);
    dst[4..8].copy_from_slice(&[col1[0], col1[s1], col1[s2], col1[s3]]);
}

#[inline(always)]
fn gather16<T: Pixel>(
    dst: &mut [T],
    data: &[T],
    base: usize,
    s1: usize,
    s2: usize,
    s3: usize,
    s4: usize,
) {
    let col = &data[base..base + s4 * 3 + s3 + 1];
    let col1 = &col[s4..];
    let col2 = &col[s4 * 2..];
    let col3 = &col[s4 * 3..];
    dst[0..4].copy_from_slice(&[col[0], col[s1], col[s2], col[s3]]);
    dst[4..8].copy_from_slice(&[col1[0], col1[s1], col1[s2], col1[s3]]);
    dst[8..12].copy_from_slice(&[col2[0], col2[s1], col2[s2], col2[s3]]);
    dst[12..16].copy_from_slice(&[col3[0], col3[s1], col3[s2], col3[s3]]);
}

/// Gathers `m` strided left-edge samples starting at `data[base]`
/// (`m` is always a multiple of 4 in `4..=MAX_EDGE`).
#[inline(always)]
fn gather_left_edge<T: Pixel>(
    dst: &mut [T; MAX_EDGE],
    data: &[T],
    base: usize,
    stride: usize,
    m: usize,
) {
    debug_assert!((4..=MAX_EDGE).contains(&m) && m & 3 == 0);
    let s1 = stride & 0x3_ffff;
    let m = m.min(MAX_EDGE);
    let s2 = s1 * 2;
    let s3 = s1 * 3;
    if m <= 4 {
        gather4(dst, data, base, s1, s2, s3);
        return;
    }
    let s4 = s1 * 4;
    if m <= 8 {
        gather8(dst, data, base, s1, s2, s3, s4);
        return;
    }
    if m == 16 {
        gather16(dst, data, base, s1, s2, s3, s4);
        return;
    }
    if m < 16 {
        gather8(dst, data, base, s1, s2, s3, s4);
        gather8(&mut dst[m - 8..], data, base + (m - 8) * s1, s1, s2, s3, s4);
        return;
    }
    let dst = &mut dst[..m];
    let mut i = 0;
    let mut off = base;
    let s16 = s1 * 16;
    while i + 16 <= m {
        gather16(&mut dst[i..i + 16], data, off, s1, s2, s3, s4);
        i += 16;
        off += s16;
    }
    if i < m {
        gather16(
            &mut dst[m - 16..m],
            data,
            base + (m - 16) * s1,
            s1,
            s2,
            s3,
            s4,
        );
    }
}

/// Sets `edge[from..to]` to `v`, using fixed-size stores so LLVM does not
/// emit a `memset` call (samples past `to` when `to - from <= 16` are unused).
#[inline(always)]
fn fill_edge<T: Pixel>(edge: &mut [T; MAX_EDGE], from: usize, to: usize, v: T) {
    debug_assert!(from < to && to <= MAX_EDGE);
    let len = to - from;
    if len <= 16 {
        if let Some(dst) = edge.get_mut(from..from + 16) {
            dst.copy_from_slice(&[v; 16]);
        } else {
            let dst = &mut edge[from..to];
            if len >= 8 {
                dst[..8].copy_from_slice(&[v; 8]);
                dst[len - 8..].copy_from_slice(&[v; 8]);
            } else if len >= 4 {
                dst[..4].copy_from_slice(&[v; 4]);
                dst[len - 4..].copy_from_slice(&[v; 4]);
            } else if len > 0 {
                dst[0] = v;
                dst[len >> 1] = v;
                dst[len - 1] = v;
            }
        }
        return;
    }
    let dst = &mut edge[from..to];
    if len <= 32 {
        let c = [v; 16];
        dst[..16].copy_from_slice(&c);
        dst[len - 16..].copy_from_slice(&c);
    } else if len <= 64 {
        let c = [v; 32];
        dst[..32].copy_from_slice(&c);
        dst[len - 32..].copy_from_slice(&c);
    } else {
        let c = [v; 64];
        dst[..64].copy_from_slice(&c);
        dst[len - 64..].copy_from_slice(&c);
    }
}
