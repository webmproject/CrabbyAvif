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

//! Intra block copy: motion vector prediction (spec 7.10.2, restricted to
//! the intra frame case), MV parsing (5.11.26, 5.11.31, 5.11.32) and the
//! block copy prediction (7.11.3 with the bilinear filter, no scaling).

use crate::decoder::av1::error::{Error, Result};
use crate::decoder::av1::pixel::{clip_pixel, round2, round2_signed, Pixel};
use crate::decoder::av1::tables::block::{
    BLOCK_HEIGHT, BLOCK_WIDTH, NUM_4X4_BLOCKS_HIGH, NUM_4X4_BLOCKS_WIDE,
};
use crate::decoder::av1::types::*;

use super::block::{plane_residual_size, SbDecoder};

const MAX_REF_MV_STACK_SIZE: usize = 8;
const REF_CAT_LEVEL: u32 = 640;
const MV_BORDER: i32 = 128;
const INTRABC_DELAY_PIXELS: i32 = 256;
const INTRABC_DELAY_SB64: i64 = 4;

/// Bilinear entry of Subpel_Filters (index 3).
const BILINEAR: [[i32; 8]; 16] = {
    let mut f = [[0i32; 8]; 16];
    let mut i = 0;
    while i < 16 {
        f[i][3] = 128 - 8 * i as i32;
        f[i][4] = 8 * i as i32;
        i += 1;
    }
    f
};

struct MvStack {
    num: usize,
    mvs: [[i32; 2]; MAX_REF_MV_STACK_SIZE],
    weights: [u32; MAX_REF_MV_STACK_SIZE],
    found_match: bool,
}

/// Spec 7.10.2.10 with force_integer_mv = 1.
fn lower_mv_precision(mv: &mut [i32; 2]) {
    for v in mv.iter_mut() {
        let a = v.abs();
        let a_int = (a + 3) >> 3;
        *v = if *v > 0 { a_int << 3 } else { -(a_int << 3) };
    }
}

impl<'a, 'b, T: Pixel> SbDecoder<'a, 'b, T> {
    fn mv_inside(&self, r: isize, c: isize) -> bool {
        c >= self.t.mi_col_start as isize
            && c < self.t.mi_col_end as isize
            && r >= self.t.mi_row_start as isize
            && r < self.t.mi_row_end as isize
    }

    /// Spec 7.10.2.7 / 7.10.2.8 for the intra frame case (single
    /// reference INTRA_FRAME, candidates must be intra block copy blocks).
    fn add_ref_mv_candidate(&self, st: &mut MvStack, r: usize, c: usize, weight: u32) {
        let mi = self.s.mi.get(r, c);
        if !mi.is_inter {
            return;
        }
        let mv = self.s.mi.mv_at(r, c);
        let mut cand = [mv[0] as i32, mv[1] as i32];
        lower_mv_precision(&mut cand);
        st.found_match = true;
        for i in 0..st.num {
            if st.mvs[i] == cand {
                st.weights[i] += weight;
                return;
            }
        }
        if st.num < MAX_REF_MV_STACK_SIZE {
            st.mvs[st.num] = cand;
            st.weights[st.num] = weight;
            st.num += 1;
        }
    }

    fn scan_row(&self, st: &mut MvStack, delta_row: isize) {
        let b = &self.b;
        let bw4 = b.bw4;
        let end4 = bw4.min(self.f.mi_cols - b.mi_col).min(16);
        let mut delta_col = 0isize;
        let use_step16 = bw4 >= 16;
        let mut delta_row = delta_row;
        if delta_row.abs() > 1 {
            delta_row += (b.mi_row & 1) as isize;
            delta_col = 1 - (b.mi_col & 1) as isize;
        }
        let mut i = 0;
        while i < end4 {
            let mv_row = b.mi_row as isize + delta_row;
            let mv_col = b.mi_col as isize + delta_col + i as isize;
            if !self.mv_inside(mv_row, mv_col) {
                break;
            }
            let cand = self.s.mi.get(mv_row as usize, mv_col as usize);
            let mut len = bw4.min(NUM_4X4_BLOCKS_WIDE[cand.bsize as usize] as usize);
            if delta_row.abs() > 1 {
                len = len.max(2);
            }
            if use_step16 {
                len = len.max(4);
            }
            self.add_ref_mv_candidate(st, mv_row as usize, mv_col as usize, len as u32 * 2);
            i += len;
        }
    }

    fn scan_col(&self, st: &mut MvStack, delta_col: isize) {
        let b = &self.b;
        let bh4 = b.bh4;
        let end4 = bh4.min(self.f.mi_rows - b.mi_row).min(16);
        let mut delta_row = 0isize;
        let use_step16 = bh4 >= 16;
        let mut delta_col = delta_col;
        if delta_col.abs() > 1 {
            delta_row = 1 - (b.mi_row & 1) as isize;
            delta_col += (b.mi_col & 1) as isize;
        }
        let mut i = 0;
        while i < end4 {
            let mv_row = b.mi_row as isize + delta_row + i as isize;
            let mv_col = b.mi_col as isize + delta_col;
            if !self.mv_inside(mv_row, mv_col) {
                break;
            }
            let cand = self.s.mi.get(mv_row as usize, mv_col as usize);
            let mut len = bh4.min(NUM_4X4_BLOCKS_HIGH[cand.bsize as usize] as usize);
            if delta_col.abs() > 1 {
                len = len.max(2);
            }
            if use_step16 {
                len = len.max(4);
            }
            self.add_ref_mv_candidate(st, mv_row as usize, mv_col as usize, len as u32 * 2);
            i += len;
        }
    }

    fn scan_point(&self, st: &mut MvStack, delta_row: isize, delta_col: isize) {
        let b = &self.b;
        let mv_row = b.mi_row as isize + delta_row;
        let mv_col = b.mi_col as isize + delta_col;
        if !self.mv_inside(mv_row, mv_col) {
            return;
        }
        // "has been written for this frame": use the luma BlockDecoded flags
        // relative to the current superblock.
        let mask = (self.f.sb4 - 1) as isize;
        let sb_r = b.mi_row as isize & !mask;
        let sb_c = b.mi_col as isize & !mask;
        let rel_r = mv_row - sb_r;
        let rel_c = mv_col - sb_c;
        let decoded = if rel_r < -1
            || rel_c < -1
            || rel_r > self.f.sb4 as isize
            || rel_c > self.f.sb4 as isize
        {
            // Outside the tracked area: rows above are decoded, anything else
            // to the right / below is not.
            rel_r < 0 && rel_c <= self.f.sb4 as isize
        } else {
            self.sc.block_decoded[0][(rel_r + 1) as usize][(rel_c + 1) as usize]
        };
        if decoded {
            self.add_ref_mv_candidate(st, mv_row as usize, mv_col as usize, 4);
        }
    }

    /// find_mv_stack(0) for an intra block copy block; returns
    /// (RefStackMv[0][0], RefStackMv[1][0]).
    fn find_mv_stack_intrabc(&self) -> ([i32; 2], [i32; 2]) {
        let b = &self.b;
        let (bw4, bh4) = (b.bw4, b.bh4);
        let mut st = MvStack {
            num: 0,
            mvs: [[0; 2]; MAX_REF_MV_STACK_SIZE],
            weights: [0; 8],
            found_match: false,
        };
        self.scan_row(&mut st, -1);
        st.found_match = false;
        self.scan_col(&mut st, -1);
        st.found_match = false;
        if bw4.max(bh4) <= 16 {
            self.scan_point(&mut st, -1, bw4 as isize);
        }
        let num_nearest = st.num;
        for w in st.weights[..num_nearest].iter_mut() {
            *w += REF_CAT_LEVEL;
        }
        self.scan_point(&mut st, -1, -1);
        self.scan_row(&mut st, -3);
        self.scan_col(&mut st, -3);
        if bh4 > 1 {
            self.scan_row(&mut st, -5);
        }
        if bw4 > 1 {
            self.scan_col(&mut st, -5);
        }
        sort_stack(&mut st, 0, num_nearest);
        let n = st.num;
        sort_stack(&mut st, num_nearest, n);
        // Extra search: no candidates can be added for intra frames, only
        // the (zero) global motion vectors fill up to 2 entries.
        for idx in st.num..2 {
            st.mvs[idx] = [0, 0];
        }
        // Context and clamping.
        let bw = BLOCK_WIDTH[b.bsize as usize] as i32;
        let bh = BLOCK_HEIGHT[b.bsize as usize] as i32;
        for idx in 0..st.num {
            st.mvs[idx][0] = self.clamp_mv_row(st.mvs[idx][0], MV_BORDER + bh * 8);
            st.mvs[idx][1] = self.clamp_mv_col(st.mvs[idx][1], MV_BORDER + bw * 8);
        }
        (st.mvs[0], st.mvs[1])
    }

    fn clamp_mv_row(&self, mv: i32, border: i32) -> i32 {
        let b = &self.b;
        let to_top = -((b.mi_row * MI_SIZE * 8) as i32);
        let to_bottom =
            ((self.f.mi_rows as i32 - b.bh4 as i32 - b.mi_row as i32) * MI_SIZE as i32) * 8;
        mv.clamp(to_top - border, to_bottom + border)
    }

    fn clamp_mv_col(&self, mv: i32, border: i32) -> i32 {
        let b = &self.b;
        let to_left = -((b.mi_col * MI_SIZE * 8) as i32);
        let to_right =
            ((self.f.mi_cols as i32 - b.bw4 as i32 - b.mi_col as i32) * MI_SIZE as i32) * 8;
        mv.clamp(to_left - border, to_right + border)
    }

    /// find_mv_stack(0) + assign_mv(0) for an intra block copy block.
    #[inline(never)]
    pub(super) fn intrabc_mode_info(&mut self) -> Result<()> {
        let (s0, s1) = self.find_mv_stack_intrabc();
        let mut pred = s0;
        if pred == [0, 0] {
            pred = s1;
        }
        if pred == [0, 0] {
            let sb4 = self.f.sb4;
            if (self.b.mi_row as isize - sb4 as isize) < self.t.mi_row_start as isize {
                pred[0] = 0;
                pred[1] = -((sb4 * MI_SIZE) as i32 + INTRABC_DELAY_PIXELS) * 8;
            } else {
                pred[0] = -((sb4 * MI_SIZE * 8) as i32);
                pred[1] = 0;
            }
        }
        // read_mv(0) with MvCtx = MV_INTRABC_CONTEXT.
        let joint = self.t.msac.symbol(&mut self.t.cdf.mv_joint);
        let mut diff = [0i32; 2];
        // MV_JOINT_ZERO = 0, HNZVZ = 1, HZVNZ = 2, HNZVNZ = 3
        if joint == 2 || joint == 3 {
            diff[0] = self.read_mv_component(0);
        }
        if joint == 1 || joint == 3 {
            diff[1] = self.read_mv_component(1);
        }
        self.b.mv = [pred[0] + diff[0], pred[1] + diff[1]];
        if !self.is_mv_valid() {
            return Err(Error::Invalid("intra block copy motion vector"));
        }
        Ok(())
    }

    /// Spec `is_mv_valid(0)` (6.10.25) for an intra block copy block: the
    /// vector must be integer, point inside the tile, at already decoded
    /// superblocks at least `INTRABC_DELAY_SB64` away (wavefront
    /// constraint). Conformant streams always satisfy it; checking it keeps
    /// the prediction inside the decoded area.
    fn is_mv_valid(&self) -> bool {
        let b = &self.b;
        let t = &self.t;
        let mv = b.mv;
        if mv[0].abs() >= 1 << 14 || mv[1].abs() >= 1 << 14 {
            return false;
        }
        if (mv[0] & 7) != 0 || (mv[1] & 7) != 0 {
            return false;
        }
        let bw = BLOCK_WIDTH[b.bsize as usize] as i64;
        let bh = BLOCK_HEIGHT[b.bsize as usize] as i64;
        let mi = MI_SIZE as i64;
        let delta_row = (mv[0] >> 3) as i64;
        let delta_col = (mv[1] >> 3) as i64;
        let mut src_top = b.mi_row as i64 * mi + delta_row;
        let mut src_left = b.mi_col as i64 * mi + delta_col;
        let src_bottom = src_top + bh;
        let src_right = src_left + bw;
        if b.has_chroma {
            if bw < 8 && self.f.ss_x != 0 {
                src_left -= 4;
            }
            if bh < 8 && self.f.ss_y != 0 {
                src_top -= 4;
            }
        }
        if src_top < t.mi_row_start as i64 * mi
            || src_left < t.mi_col_start as i64 * mi
            || src_bottom > t.mi_row_end as i64 * mi
            || src_right > t.mi_col_end as i64 * mi
        {
            return false;
        }
        let sb128 = self.f.sb4 == 32;
        let sb_h = (self.f.sb4 * MI_SIZE) as i64;
        let active_sb_row = (b.mi_row as i64 * mi) / sb_h;
        let active_sb64_col = (b.mi_col as i64 * mi) >> 6;
        let src_sb_row = (src_bottom - 1) / sb_h;
        let src_sb64_col = (src_right - 1) >> 6;
        let total_sb64_per_row = ((t.mi_col_end as i64 - t.mi_col_start as i64 - 1) >> 4) + 1;
        let active_sb64 = active_sb_row * total_sb64_per_row + active_sb64_col;
        let src_sb64 = src_sb_row * total_sb64_per_row + src_sb64_col;
        if src_sb64 >= active_sb64 - INTRABC_DELAY_SB64 {
            return false;
        }
        let gradient = 1 + INTRABC_DELAY_SB64 + sb128 as i64;
        let wf_offset = gradient * (active_sb_row - src_sb_row);
        if src_sb_row > active_sb_row
            || src_sb64_col >= active_sb64_col - INTRABC_DELAY_SB64 + wf_offset
        {
            return false;
        }
        true
    }

    fn read_mv_component(&mut self, comp: usize) -> i32 {
        let t = &mut *self.t;
        let sign = t.msac.bool_adapt(&mut t.cdf.mv_sign[comp]);
        let class = t.msac.symbol(&mut t.cdf.mv_class[comp]);
        let mag = if class == 0 {
            let bit = t.msac.bool_adapt(&mut t.cdf.mv_class0_bit[comp]) as i32;
            // force_integer_mv: fr = 3, hp = 1
            ((bit << 3) | (3 << 1) | 1) + 1
        } else {
            let mut d = 0i32;
            for i in 0..class {
                d |= (t.msac.bool_adapt(&mut t.cdf.mv_bit[comp][i]) as i32) << i;
            }
            let mag = 2 << (class + 2); // CLASS0_SIZE << (class + 2)
            mag + ((d << 3) | (3 << 1) | 1) + 1
        };
        if sign {
            -mag
        } else {
            mag
        }
    }

    /// compute_prediction() for an intra block copy block.
    ///
    /// The vector is integer (`is_mv_valid`), so in luma and in
    /// unsubsampled chroma the 8-tap bilinear filter of spec 7.11.3.4 is
    /// at phase 0 (a copy); subsampled chroma can be at phase 8 (half
    /// pel). Only taps 3 and 4 of `BILINEAR` are non zero, so the
    /// prediction reads the clamped positions (x, y), (x + 1, y),
    /// (x, y + 1) and (x + 1, y + 1) only.
    #[inline(never)]
    pub(super) fn predict_intrabc(&mut self) -> Result<()> {
        if !self.recon {
            return Ok(());
        }
        let f = self.f;
        let b = &self.b;
        let planes = if b.has_chroma { 3 } else { 1 };
        let bd = f.bd;
        let round0 = if bd == 12 { 5 } else { 3 };
        let round1 = if bd == 12 { 9 } else { 11 };
        let (bsize, mi_row, mi_col, mv) = (b.bsize, b.mi_row, b.mi_col, b.mv);
        for plane in 0..planes {
            let (sx, sy) = f.sub(plane);
            let plane_sz = plane_residual_size(bsize, sx, sy);
            let w = NUM_4X4_BLOCKS_WIDE[plane_sz as usize] as usize * 4;
            let h = NUM_4X4_BLOCKS_HIGH[plane_sz as usize] as usize * 4;
            let base_x = (mi_col >> sx) * MI_SIZE;
            let base_y = (mi_row >> sy) * MI_SIZE;
            // Motion vector scaling with no scaling (7.11.3.3).
            let pos_x = ((base_x as i32) << 4) + ((2 * mv[1]) >> sx);
            let pos_y = ((base_y as i32) << 4) + ((2 * mv[0]) >> sy);
            let start_x = pos_x * 64 + 32;
            let start_y = pos_y * 64 + 32;
            let last_x = (((f.mi_cols * MI_SIZE) + sx) >> sx) as i32 - 1;
            let last_y = (((f.mi_rows * MI_SIZE) + sy) >> sy) as i32 - 1;
            let fx = ((start_x >> 6) & 15) as usize;
            let fy = ((start_y >> 6) & 15) as usize;
            let ix = start_x >> 10;
            let iy = start_y >> 10;
            let pl = &mut self.s.planes[plane];
            let min_x = pl.x0 as i32;
            let min_y = pl.y0 as i32;
            let inside =
                ix >= min_x && iy >= min_y && ix + w as i32 <= last_x && iy + h as i32 <= last_y;
            let stride = pl.stride;
            let mut dst = pl.idx(base_x, base_y);
            if fx == 0 && fy == 0 && inside {
                // Phase 0 in both directions: a plain copy (the source
                // area is decoded and does not overlap the block).
                let mut src = pl.idx(ix as usize, iy as usize);
                for _ in 0..h {
                    pl.data_mut().copy_within(src..src + w, dst);
                    src += stride;
                    dst += stride;
                }
                continue;
            }
            let (f0x, f1x) = (BILINEAR[fx][3], BILINEAR[fx][4]);
            let (f0y, f1y) = (BILINEAR[fy][3], BILINEAR[fy][4]);
            // Blocks are up to 128x128.
            let x0 = pl.x0;
            let mut xs = [0u16; 129];
            for (c, x) in xs[..w + 1].iter_mut().enumerate() {
                *x = ((ix + c as i32).clamp(min_x, last_x) as usize - x0) as u16;
            }
            // Horizontally filtered rows y and y + 1 of the current
            // output row (rows of `inter` in the spec).
            let mut buf = [[0i32; 128]; 2];
            let hfilter = |pl: &crate::decoder::av1::decode::buffers::PlaneWin<T>,
                           r: usize,
                           out: &mut [i32; 128]| {
                let yy = (iy + r as i32).clamp(min_y, last_y) as usize;
                let row = pl.row(yy);
                for c in 0..w {
                    let s =
                        f0x * row[xs[c] as usize].to_i32() + f1x * row[xs[c + 1] as usize].to_i32();
                    out[c] = round2(s, round0);
                }
            };
            hfilter(pl, 0, &mut buf[0]);
            for r in 0..h {
                let (prev_i, cur_i) = (r & 1, (r + 1) & 1);
                hfilter(pl, r + 1, &mut buf[cur_i]);
                let prev = &buf[prev_i][..w];
                let cur = &buf[cur_i][..w];
                let row = &mut pl.data_mut()[dst..dst + w];
                for c in 0..w {
                    let s = f0y * prev[c] + f1y * cur[c];
                    row[c] = clip_pixel(round2(s, round1), bd);
                }
                dst += stride;
            }
        }
        let _ = round2_signed(0, 0);
        Ok(())
    }
}

fn sort_stack(st: &mut MvStack, start: usize, end: usize) {
    let mut end = end;
    while end > start {
        let mut new_end = start;
        for idx in (start + 1)..end {
            if st.weights[idx - 1] < st.weights[idx] {
                st.weights.swap(idx - 1, idx);
                st.mvs.swap(idx - 1, idx);
                new_end = idx;
            }
        }
        end = new_end;
    }
}
