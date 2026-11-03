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

use crate::decoder::av1::dsp::ipred::MAX_EDGE;
use crate::decoder::av1::pixel::{round2, Pixel};
use crate::decoder::av1::types::*;

use super::block::{txh, txw, SbDecoder};

/// Bound of a sample offset into a plane window (`PlaneWin::data`), used
/// as a mask so that LLVM knows the offset arithmetic below cannot
/// overflow (and drops the matching checks). A window is at most a whole
/// frame of `stride` x `rows` samples, with `stride` and `rows` (multiples
/// of the superblock size) at most 65536 since AV1 frame dimensions are at
/// most 65536, so every offset is below 2^32 and the mask never changes
/// it. (It is a no-op on 32-bit targets.)
const OFF_MASK: usize = 0xffff_ffff;

/// Mask of `stride`, `x - x0` and `y - y0` (all at most 65536, see
/// `OFF_MASK`).
const DIM_MASK: usize = 0x3_ffff;

// The cold or rarely used out-of-line helpers of this file take an unused
// `const DUMMY: usize` generic, always instantiated as `::<0>`. It has no
// semantic effect; like the ones in `msac.rs` it only changes where the
// function is monomorphized (codegen unit placement), which measurably
// changed the generated code. Keep it unless benchmarks show otherwise.

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
    #[inline(always)]
    fn get_filter_type(&self, plane: usize) -> bool {
        let b = &self.b;
        let sm = |m: u8| m.wrapping_sub(IntraMode::SMOOTH_PRED as u8) < 3;
        if plane == 0 {
            // (mi_row - 1, mi_col) and (mi_row, mi_col - 1) are the block's
            // above / left mode info, already loaded.
            return (b.avail_u && sm(b.above_mi.y_mode)) || (b.avail_l && sm(b.left_mi.y_mode));
        }
        let (sx, sy) = (self.f.ss_x, self.f.ss_y);
        (b.avail_u_chroma && sm(self.s.mi.get((b.mi_row & !sy) - 1, b.mi_col | sx).uv_mode))
            || (b.avail_l_chroma && sm(self.s.mi.get(b.mi_row | sy, (b.mi_col & !sx) - 1).uv_mode))
    }

    /// Computes the intra prediction setup of the current block
    /// (`s.ipred`, the parts of spec 7.11.2 that are the same for all
    /// transform blocks of a plane type). Called once per intra block
    /// before its transform blocks are predicted.
    #[inline(always)]
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
    /// per-frame ones are set by `IntraSetup::new`, and `above_line_y` by
    /// `FrameState::set_sb_y`).
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
        // only the available ones). Since |angle_delta * 3| <= 9, only
        // V_PRED (90) and H_PRED (180) can cross or land on 90/180.
        let (na, nl, need_filter) = if use_filter {
            (w, h, false)
        } else {
            match mode {
                IntraMode::V_PRED => {
                    if angle_delta == 0 {
                        (w, 0, false)
                    } else if angle_delta < 0 {
                        (w + h, 0, true)
                    } else {
                        (w, h, true)
                    }
                }
                IntraMode::H_PRED => {
                    if angle_delta == 0 {
                        (0, h, false)
                    } else if angle_delta < 0 {
                        (w, h, true)
                    } else {
                        (0, w + h, true)
                    }
                }
                IntraMode::D45_PRED | IntraMode::D67_PRED => (w + h, 0, true),
                IntraMode::D135_PRED | IntraMode::D113_PRED | IntraMode::D157_PRED => (w, h, true),
                IntraMode::D203_PRED => (0, w + h, true),
                _ => (w, h, false),
            }
        };
        let smooth_filter_type =
            need_filter && f.seq.enable_intra_edge_filter && self.get_filter_type(plane);
        let pt = (plane != 0) as usize;
        let su = &mut self.s.ipred[pt];
        su.p.mode = mode as u8;
        su.p.angle_delta = angle_delta;
        su.p.filter_intra_mode = filter_intra_mode;
        su.p.smooth_filter_type = smooth_filter_type;
        su.p.w = w;
        su.p.h = h;
        su.na = na;
        su.nl = nl;
        su.na_lim = [w, na.min(w << 1)];
        su.nl_lim = [h, nl.min(h << 1)];
        su.dc = !use_filter && mode == IntraMode::DC_PRED;
        su.need_top_left = use_filter || need_filter || mode == IntraMode::PAETH_PRED;
    }

    /// Spec 7.11.2 for one transform block, using the block's
    /// `setup_intra`. Only the edge samples the selected predictor reads
    /// are derived (`ipred::edge_needs`, as dav1d's prepare_intra_edges).
    #[inline(always)]
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
        if !have_above || !have_left {
            let flags = (have_left as u32)
                | ((have_above as u32) << 1)
                | ((have_above_right as u32) << 2)
                | ((have_below_left as u32) << 3);
            self.predict_intra_border::<0>(plane, x, y, flags);
            return;
        }
        let plane = plane.min(2);
        let pt = (plane != 0) as usize;
        let s = &mut *self.s;
        let su = &mut s.ipred[pt];
        su.p.x = x as i32;
        su.p.y = y as i32;
        let e = &mut s.edges;
        let pl = &mut s.planes[plane];
        let stride = pl.stride & DIM_MASK;
        let col_off = (x - pl.x0) & DIM_MASK;
        let row_off = ((y - pl.y0) & DIM_MASK) * stride;
        debug_assert!(pl.stride <= DIM_MASK && x - pl.x0 <= DIM_MASK && y - pl.y0 <= DIM_MASK);
        debug_assert!(row_off + col_off <= OFF_MASK);
        let left_off = (row_off + col_off).wrapping_sub(1) & OFF_MASK;
        let dst_off = left_off + 1;
        let data = pl.data_mut();
        // Common case (both neighbors available): `above[0]` is the
        // top-left sample, `above[1..]` row y - 1 from column x.
        let na = su.na;
        if na != 0 {
            let m_above = (su.max_x_p1 - x).min(su.na_lim[have_above_right as usize]);
            let (src, base) = if y != su.above_line_y {
                (&data[..], left_off.wrapping_sub(stride) & OFF_MASK)
            } else {
                (&s.above_line[plane][..], col_off.wrapping_sub(1) & OFF_MASK)
            };
            if m_above <= 16 && base + 17 <= src.len() {
                let win = &src[base..base + 17];
                e.top_left = win[0];
                e.above[..16].copy_from_slice(&win[1..17]);
                if m_above < na {
                    let v = win[m_above];
                    if na <= m_above + 8 {
                        e.above[m_above..m_above + 8].copy_from_slice(&[v; 8]);
                    } else {
                        fill_edge::<T, 0>(&mut e.above, m_above, na, v);
                    }
                }
            } else {
                copy_above_edge_large::<T, 0>(e, src, base, m_above, na);
            }
        } else if su.need_top_left {
            e.top_left = if y != su.above_line_y {
                data[left_off.wrapping_sub(stride) & OFF_MASK]
            } else {
                s.above_line[plane][col_off.wrapping_sub(1) & OFF_MASK]
            };
        }
        let nl = su.nl;
        if nl != 0 {
            let m = (su.max_y_p1 - y).min(su.nl_lim[have_below_left as usize]);
            if m == 4 {
                let v = gather4(&mut e.left, data, left_off, stride);
                if 4 < nl {
                    if nl <= 12 {
                        e.left[4..12].copy_from_slice(&[v; 8]);
                    } else {
                        fill_edge::<T, 0>(&mut e.left, 4, nl, v);
                    }
                }
            } else if m == 8 {
                let v = gather8(&mut e.left, data, left_off, stride);
                if 8 < nl {
                    if nl <= 16 {
                        e.left[8..16].copy_from_slice(&[v; 8]);
                    } else {
                        fill_edge::<T, 0>(&mut e.left, 8, nl, v);
                    }
                }
            } else {
                gather_left_edge_large::<T, 0>(&mut e.left, &data[left_off..], stride, m, nl);
            }
        }
        (self.dsp.predict_intra)(&mut data[dst_off..], stride, e, &su.p);
    }

    /// `predict_intra` for a transform block without a left or an above
    /// neighbor (`flags`: have_left, have_above, have_above_right,
    /// have_below_left in bits 0..4).
    #[cold]
    #[inline(never)]
    fn predict_intra_border<const DUMMY: usize>(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        flags: u32,
    ) {
        let have_left = (flags & 1) != 0;
        let have_above = (flags & 2) != 0;
        let have_above_right = (flags & 4) != 0;
        let have_below_left = (flags & 8) != 0;
        debug_assert!(!(have_above && have_left));
        let plane = plane.min(2);
        let pt = (plane != 0) as usize;
        let s = &mut *self.s;
        let su = &mut s.ipred[pt];
        let (na, nl) = if su.dc {
            (
                if have_above { su.na } else { 0 },
                if have_left { su.nl } else { 0 },
            )
        } else {
            (su.na, su.nl)
        };
        let need_top_left = su.need_top_left;
        let bd = su.p.bd;
        su.p.x = x as i32;
        su.p.y = y as i32;
        su.p.have_left = have_left;
        su.p.have_above = have_above;
        let e = &mut s.edges;
        let pl = &mut s.planes[plane];
        let stride = pl.stride & DIM_MASK;
        let col_off = (x - pl.x0) & DIM_MASK;
        let row_off = ((y - pl.y0) & DIM_MASK) * stride;
        let dst_off = row_off + col_off;
        let data = pl.data_mut();
        // Row y - 1: the saved unfiltered line at the top of the
        // superblock row, the current window otherwise (see `px`).
        let above_row: &[T] = if !have_above || (na == 0 && have_left && !need_top_left) {
            &[]
        } else if y != su.above_line_y {
            let a_off = row_off - stride;
            &data[a_off..]
        } else {
            &s.above_line[plane]
        };
        if na == 0 {
        } else if !have_above {
            let v = if have_left { data[dst_off - 1] } else { T::from_i32((1 << (bd - 1)) - 1) };
            if na <= 16 {
                e.above[..16].copy_from_slice(&[v; 16]);
            } else {
                fill_edge::<T, 0>(&mut e.above, 0, na, v);
            }
        } else {
            let m = (su.max_x_p1 - x)
                .min(su.na_lim[have_above_right as usize])
                .min(MAX_EDGE);
            if m <= 16 && col_off + 16 <= above_row.len() {
                let win = &above_row[col_off..col_off + 16];
                e.above[..16].copy_from_slice(win);
                if m < na {
                    let v = win[m - 1];
                    if na <= m + 8 {
                        e.above[m..m + 8].copy_from_slice(&[v; 8]);
                    } else {
                        fill_edge::<T, 0>(&mut e.above, m, na, v);
                    }
                }
            } else {
                let row = &above_row[col_off..col_off + m];
                copy_above_edge(&mut e.above, row, m);
                if m < na {
                    fill_edge::<T, 0>(&mut e.above, m, na, row[m - 1]);
                }
            }
        }
        if nl == 0 {
        } else if !have_left {
            let v = if have_above { above_row[col_off] } else { T::from_i32((1 << (bd - 1)) + 1) };
            if nl <= 16 {
                e.left[..16].copy_from_slice(&[v; 16]);
            } else {
                fill_edge::<T, 0>(&mut e.left, 0, nl, v);
            }
        } else {
            let m = (su.max_y_p1 - y).min(su.nl_lim[have_below_left as usize]);
            let left_off = dst_off - 1;
            if m == 4 {
                let v = gather4(&mut e.left, data, left_off, stride);
                if 4 < nl {
                    if nl <= 12 {
                        e.left[4..12].copy_from_slice(&[v; 8]);
                    } else {
                        fill_edge::<T, 0>(&mut e.left, 4, nl, v);
                    }
                }
            } else if m == 8 {
                let v = gather8(&mut e.left, data, left_off, stride);
                if 8 < nl {
                    if nl <= 16 {
                        e.left[8..16].copy_from_slice(&[v; 8]);
                    } else {
                        fill_edge::<T, 0>(&mut e.left, 8, nl, v);
                    }
                }
            } else {
                gather_left_edge_large::<T, 0>(&mut e.left, &data[left_off..], stride, m, nl);
            }
        }
        if need_top_left {
            // `predict_intra` handles the case with both neighbors.
            e.top_left = if have_above {
                above_row[col_off]
            } else if have_left {
                data[dst_off - 1]
            } else {
                T::from_i32(1 << (bd - 1))
            };
        }
        (self.dsp.predict_intra)(&mut data[dst_off..], stride, e, &su.p);
        su.p.have_left = true;
        su.p.have_above = true;
    }

    /// Spec 7.11.4.
    #[cold]
    #[inline(never)]
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
        let raw_pal = match plane {
            0 => self.b.palette_y,
            1 => self.b.palette_u,
            _ => self.b.palette_v,
        };
        let palette = raw_pal.map(|c| T::from_i32(c as i32));
        let map = if plane == 0 { &self.sc.color_map_y } else { &self.sc.color_map_uv };
        let pl = &mut self.s.planes[plane.min(2)];
        let stride = pl.stride;
        let data = pl.slice_mut(start_x, start_y);
        let mx = x * 4;
        let my = y * 4;
        let mut dst = 0;
        for mrow in &map[my..my + h] {
            let row = &mut data[dst..dst + w];
            for (d, &idx) in row.iter_mut().zip(&mrow[mx..mx + w]) {
                *d = palette[(idx & 7) as usize];
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
                cfl_ac_corrupt::<T, 0>(ac, luma, start_x, start_y, w, h, sx, sy, max_lx, max_ly);
            }
        }
        let ac = &s.cfl_ac[..w * h];
        let pl = &mut s.planes[plane.min(2)];
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
    #[inline(always)]
    pub(super) fn reconstruct(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        tx_sz: TxSize,
        eob: usize,
    ) {
        let pl = &mut self.s.planes[plane.min(2)];
        let stride = pl.stride;
        // Also leaves the coefficients zero for the next block.
        if let (Some(tbl), false) = (self.dsp.itx, self.b.lossless) {
            let ty = self.plane_tx_type as usize;
            let f = tbl[(tx_sz as usize).min(18)][ty & 15];
            f(
                pl.slice_mut(x, y),
                stride,
                &mut self.sc.coeffs,
                eob | (ty << 16),
            );
            return;
        }
        (self.dsp.inv_txfm_add)(
            pl.slice_mut(x, y),
            stride,
            &mut self.sc.coeffs,
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
fn cfl_ac_corrupt<T: Pixel, const DUMMY: usize>(
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

#[inline(never)]
fn copy_above_edge_large<T: Pixel, const DUMMY: usize>(
    e: &mut crate::decoder::av1::dsp::ipred::IntraEdges<T>,
    src: &[T],
    base: usize,
    m: usize,
    na: usize,
) {
    if m <= 32 && base + 33 <= src.len() {
        let win = &src[base..base + 33];
        e.top_left = win[0];
        e.above[..32].copy_from_slice(&win[1..33]);
        if m < na {
            fill_edge::<T, 0>(&mut e.above, m, na, win[m]);
        }
        return;
    }
    if m <= 64 && base + 65 <= src.len() {
        let win = &src[base..base + 65];
        e.top_left = win[0];
        e.above[..64].copy_from_slice(&win[1..65]);
        if m < na {
            fill_edge::<T, 0>(&mut e.above, m, na, win[m]);
        }
        return;
    }
    let m = m.min(MAX_EDGE);
    let above = &src[base..base + m + 1];
    e.top_left = above[0];
    copy_above_edge(&mut e.above, &above[1..], m);
    if m < na {
        fill_edge::<T, 0>(&mut e.above, m, na, above[m]);
    }
}

/// Copies `src` (of length `m` in `4..=MAX_EDGE`, multiple of 4) into `dst`
/// using power-of-two vector copies without internal bounds checks.
#[inline(always)]
fn copy_above_edge<T: Pixel>(dst: &mut [T; MAX_EDGE], src: &[T], m: usize) {
    let m = m.min(MAX_EDGE);
    let src = &src[..m];
    let dst = &mut dst[..m];
    if m <= 16 {
        if m >= 8 {
            dst[..8].copy_from_slice(&src[..8]);
            dst[m - 8..].copy_from_slice(&src[m - 8..]);
        } else if m >= 4 {
            dst[..4].copy_from_slice(&src[..4]);
        }
    } else if m <= 32 {
        if m == 32 {
            dst[..32].copy_from_slice(&src[..32]);
        } else {
            dst[..16].copy_from_slice(&src[..16]);
            dst[m - 16..].copy_from_slice(&src[m - 16..]);
        }
    } else if m <= 64 {
        if m == 64 {
            dst[..64].copy_from_slice(&src[..64]);
        } else {
            dst[..32].copy_from_slice(&src[..32]);
            dst[m - 32..].copy_from_slice(&src[m - 32..]);
        }
    } else if m == 128 {
        dst[..128].copy_from_slice(&src[..128]);
    } else {
        dst[..64].copy_from_slice(&src[..64]);
        dst[m - 64..].copy_from_slice(&src[m - 64..]);
    }
}

#[inline(always)]
fn gather4<T: Pixel>(dst: &mut [T], data: &[T], left_off: usize, s1: usize) -> T {
    debug_assert!(left_off <= OFF_MASK);
    let left_off = left_off & OFF_MASK;
    let s1 = s1 & DIM_MASK;
    let s3 = s1 * 3;
    let col = &data[left_off..left_off + s3 + 1];
    let v = col[s3];
    dst[..4].copy_from_slice(&[col[0], col[s1], col[s1 * 2], v]);
    v
}

#[inline(always)]
fn gather8<T: Pixel>(dst: &mut [T], data: &[T], left_off: usize, s1: usize) -> T {
    debug_assert!(left_off <= OFF_MASK);
    let left_off = left_off & OFF_MASK;
    let s1 = s1 & DIM_MASK;
    let s3 = s1 * 3;
    let s4 = s1 * 4;
    let col = &data[left_off..left_off + s4 + s3 + 1];
    dst[0..4].copy_from_slice(&[col[0], col[s1], col[s1 * 2], col[s3]]);
    let col4 = &col[s4..];
    let v = col4[s3];
    dst[4..8].copy_from_slice(&[col4[0], col4[s1], col4[s1 * 2], v]);
    v
}

#[inline(always)]
fn gather16<T: Pixel>(dst: &mut [T], data: &[T], s1: usize) {
    let s1 = s1 & DIM_MASK;
    let s3 = s1 * 3;
    let s4 = s1 * 4;
    let s8 = s1 * 8;
    let s12 = s8 + s4;
    let col = &data[..s12 + s3 + 1];
    dst[0..4].copy_from_slice(&[col[0], col[s1], col[s1 * 2], col[s3]]);
    let col4 = &col[s4..];
    dst[4..8].copy_from_slice(&[col4[0], col4[s1], col4[s1 * 2], col4[s3]]);
    let col8 = &col[s8..];
    dst[8..12].copy_from_slice(&[col8[0], col8[s1], col8[s1 * 2], col8[s3]]);
    let col12 = &col[s12..];
    dst[12..16].copy_from_slice(&[col12[0], col12[s1], col12[s1 * 2], col12[s3]]);
}

#[inline(never)]
fn gather_left_edge_large<T: Pixel, const DUMMY: usize>(
    dst: &mut [T; MAX_EDGE],
    data: &[T],
    s1: usize,
    m: usize,
    nl: usize,
) {
    let m = (m & !3).min(MAX_EDGE);
    let s1 = s1 & DIM_MASK;
    if m == 16 {
        gather16(dst, data, s1);
    } else if m == 32 {
        let s16 = s1 * 16;
        let col = &data[..s1 * 31 + 1];
        gather16(&mut dst[0..16], col, s1);
        gather16(&mut dst[16..32], &col[s16..], s1);
    } else if m <= 8 {
        if m == 8 {
            gather8(dst, data, 0, s1);
        } else if m == 4 {
            gather4(dst, data, 0, s1);
        }
    } else if m == 12 {
        let s8 = s1 * 8;
        let col = &data[..s1 * 11 + 1];
        gather8(&mut dst[0..8], col, 0, s1);
        gather4(&mut dst[8..12], col, s8, s1);
    } else {
        let s8 = s1 * 8;
        let s16 = s1 * 16;
        let dst_slice = &mut dst[..m];
        let mut i = 0;
        let mut off = 0;
        while i + 16 <= m {
            gather16(&mut dst_slice[i..i + 16], &data[off..], s1);
            i += 16;
            off += s16;
        }
        if (m & 8) != 0 {
            gather8(&mut dst_slice[i..i + 8], data, off, s1);
            i += 8;
            off += s8;
        }
        if (m & 4) != 0 {
            gather4(&mut dst_slice[i..i + 4], data, off, s1);
        }
    }
    if m < nl {
        let v = dst[(m - 1) & (MAX_EDGE - 1)];
        fill_edge::<T, 0>(dst, m, nl, v);
    }
}

/// Sets `edge[from..to]` to `v`, using fixed-size stores so LLVM does not
/// emit a `memset` call.
#[inline(never)]
fn fill_edge<T: Pixel, const DUMMY: usize>(edge: &mut [T; MAX_EDGE], from: usize, to: usize, v: T) {
    debug_assert!(from < to && to <= MAX_EDGE);
    let to = to.min(MAX_EDGE);
    let from = from.min(to);
    let dst = &mut edge[from..to];
    let len = dst.len();
    if len <= 16 {
        if len >= 8 {
            dst[..8].copy_from_slice(&[v; 8]);
            dst[len - 8..].copy_from_slice(&[v; 8]);
        } else if len >= 4 {
            dst[..4].copy_from_slice(&[v; 4]);
        }
        return;
    }
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
