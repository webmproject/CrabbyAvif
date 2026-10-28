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

//! Superblock / partition / mode info parsing (spec 5.11.4 - 5.11.38
//! except residual and coefficients), split out of block.rs.

use super::*;
use crate::decoder::av1::decode::buffers::MiInfo;

const SPLIT_TX_BY_DEPTH: [[TxSize; 5]; TxSize::COUNT] = {
    let mut t = [[TxSize::Tx4x4; 5]; TxSize::COUNT];
    let mut i = 0;
    while i < TxSize::COUNT {
        let mut cur = TxSize::ALL[i];
        let mut d = 0;
        while d < 5 {
            t[i][d] = cur;
            cur = SPLIT_TX_SIZE[cur as usize];
            d += 1;
        }
        i += 1;
    }
    t
};

#[inline(always)]
fn zero_ctx2(a: &mut [u8], b: &mut [u8], pos: usize, len: usize) {
    #[inline(always)]
    fn z<const N: usize>(a: &mut [u8], b: &mut [u8], pos: usize) {
        a[pos..pos + N].fill(0);
        b[pos..pos + N].fill(0);
    }
    match len {
        1 => z::<1>(a, b, pos),
        2 => z::<2>(a, b, pos),
        4 => z::<4>(a, b, pos),
        8 => z::<8>(a, b, pos),
        16 => z::<16>(a, b, pos),
        _ => z::<32>(a, b, pos),
    }
}

#[inline(always)]
fn zero_ctx4(a1: &mut [u8], b1: &mut [u8], a2: &mut [u8], b2: &mut [u8], pos: usize, len: usize) {
    #[inline(always)]
    fn z<const N: usize>(a1: &mut [u8], b1: &mut [u8], a2: &mut [u8], b2: &mut [u8], pos: usize) {
        a1[pos..pos + N].fill(0);
        b1[pos..pos + N].fill(0);
        a2[pos..pos + N].fill(0);
        b2[pos..pos + N].fill(0);
    }
    match len {
        1 => z::<1>(a1, b1, a2, b2, pos),
        2 => z::<2>(a1, b1, a2, b2, pos),
        4 => z::<4>(a1, b1, a2, b2, pos),
        8 => z::<8>(a1, b1, a2, b2, pos),
        16 => z::<16>(a1, b1, a2, b2, pos),
        _ => z::<32>(a1, b1, a2, b2, pos),
    }
}

/// Sets `w` (1, 2, 4, 8, 16 or 32) entries from column `c0` in rows
/// `r0..r_end` of a wrapped 2D array. `fill` on a constant-length
/// subslice packs `val` into one register and broadcasts it (copying
/// from a `[val; N]` array made LLVM store `MiInfo` byte by byte).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn fill_mi_rows<T: Copy>(
    data: &mut [T],
    stride: usize,
    row_mask: usize,
    r0: usize,
    r_end: usize,
    c0: usize,
    w: usize,
    val: T,
) {
    #[inline(always)]
    fn rows<T: Copy, const N: usize>(
        data: &mut [T],
        stride: usize,
        row_mask: usize,
        r0: usize,
        r_end: usize,
        c0: usize,
        val: T,
    ) {
        for row in r0..r_end {
            let i = (row & row_mask) * stride + c0;
            data[i..i + N].fill(val);
        }
    }
    match w {
        1 => rows::<T, 1>(data, stride, row_mask, r0, r_end, c0, val),
        2 => rows::<T, 2>(data, stride, row_mask, r0, r_end, c0, val),
        4 => rows::<T, 4>(data, stride, row_mask, r0, r_end, c0, val),
        8 => rows::<T, 8>(data, stride, row_mask, r0, r_end, c0, val),
        16 => rows::<T, 16>(data, stride, row_mask, r0, r_end, c0, val),
        _ => rows::<T, 32>(data, stride, row_mask, r0, r_end, c0, val),
    }
}

/// `fill_mi_rows` of the mode info and a second array (`LoopfilterTxSizes`
/// or `Mvs`, same layout) in one pass.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn fill_mi_rows2<U: Copy>(
    data: &mut [MiInfo],
    aux: &mut [U],
    stride: usize,
    row_mask: usize,
    r0: usize,
    r_end: usize,
    c0: usize,
    w: usize,
    info: MiInfo,
    aux_val: U,
) {
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn rows<U: Copy, const N: usize>(
        data: &mut [MiInfo],
        aux: &mut [U],
        stride: usize,
        row_mask: usize,
        r0: usize,
        r_end: usize,
        c0: usize,
        info: MiInfo,
        aux_val: U,
    ) {
        for row in r0..r_end {
            let i = (row & row_mask) * stride + c0;
            data[i..i + N].fill(info);
            aux[i..i + N].fill(aux_val);
        }
    }
    match w {
        1 => rows::<U, 1>(data, aux, stride, row_mask, r0, r_end, c0, info, aux_val),
        2 => rows::<U, 2>(data, aux, stride, row_mask, r0, r_end, c0, info, aux_val),
        4 => rows::<U, 4>(data, aux, stride, row_mask, r0, r_end, c0, info, aux_val),
        8 => rows::<U, 8>(data, aux, stride, row_mask, r0, r_end, c0, info, aux_val),
        16 => rows::<U, 16>(data, aux, stride, row_mask, r0, r_end, c0, info, aux_val),
        _ => rows::<U, 32>(data, aux, stride, row_mask, r0, r_end, c0, info, aux_val),
    }
}

impl<'a, 'b, T: Pixel> SbDecoder<'a, 'b, T> {
    /// Decodes the superblock at (r, c) (in 4x4 units): clear_cdef,
    /// clear_block_decoded_flags, read_lr and decode_partition.
    pub fn decode_sb(&mut self, r: usize, c: usize) -> Result<()> {
        // clear_cdef
        self.s.set_cdef_idx(r, c, -1);
        if self.f.sb128 {
            self.s.set_cdef_idx(r, c + 16, -1);
            self.s.set_cdef_idx(r + 16, c, -1);
            self.s.set_cdef_idx(r + 16, c + 16, -1);
        }
        self.clear_block_decoded_flags(r, c);
        self.read_deltas = self.f.fh.delta_q_present;
        self.read_lr(r, c)?;
        self.decode_partition(r, c, self.f.sb_size)
    }

    pub(crate) fn clear_block_decoded_flags(&mut self, r: usize, c: usize) {
        let sb4 = self.f.sb4;
        for plane in 0..self.f.num_planes {
            let (sx, sy) = self.f.sub(plane);
            let sb_w4 = (self.t.mi_col_end - c) >> sx;
            let sb_h4 = (self.t.mi_row_end - r) >> sy;
            let w = (sb4 >> sx) + 2;
            let h = (sb4 >> sy) + 2;
            let n0 = (sb_w4 + 1).min(w);
            let h_in = sb_h4.min(h - 2);
            let bd = &mut self.sc.block_decoded[plane];
            bd[0] = BlockDecodedRow((1u64 << n0) - 1);
            bd[1..=h_in].fill(BlockDecodedRow(1));
            bd[1 + h_in..h].fill(BlockDecodedRow(0));
        }
    }

    pub(crate) fn read_lr(&mut self, r: usize, c: usize) -> Result<()> {
        let fh = self.f.fh;
        if fh.allow_intrabc {
            return Ok(());
        }
        let w = self.f.sb4;
        let h = self.f.sb4;
        for plane in 0..self.f.num_planes {
            if fh.lr.frame_restoration_type[plane] == RestorationType::None {
                continue;
            }
            let (sx, sy) = self.f.sub(plane);
            let lp = &self.s.lr[plane];
            let unit_size = lp.unit_size;
            let unit_rows = lp.unit_rows;
            let unit_cols = lp.unit_cols;
            let unit_row_start = (r * (MI_SIZE >> sy)).div_ceil(unit_size);
            let unit_row_end = unit_rows.min(((r + h) * (MI_SIZE >> sy)).div_ceil(unit_size));
            let (numerator, denominator) = if fh.use_superres {
                (
                    (MI_SIZE >> sx) * fh.superres_denom as usize,
                    unit_size * SUPERRES_NUM as usize,
                )
            } else {
                (MI_SIZE >> sx, unit_size)
            };
            let unit_col_start = (c * numerator).div_ceil(denominator);
            let unit_col_end = unit_cols.min(((c + w) * numerator).div_ceil(denominator));
            for unit_row in unit_row_start..unit_row_end {
                for unit_col in unit_col_start..unit_col_end {
                    self.read_lr_unit(plane, unit_row, unit_col);
                }
            }
        }
        Ok(())
    }

    pub(crate) fn read_lr_unit(&mut self, plane: usize, unit_row: usize, unit_col: usize) {
        use crate::decoder::av1::dsp::lr::{SGRPROJ_PARAMS_BITS, SGRPROJ_PRJ_BITS};
        use crate::decoder::av1::tables::misc::{
            SGRPROJ_XQD_MAX, SGRPROJ_XQD_MIN, WIENER_TAPS_K, WIENER_TAPS_MAX, WIENER_TAPS_MIN,
        };
        const SGRPROJ_PRJ_SUBEXP_K: u32 = 4;
        let frame_type = self.f.fh.lr.frame_restoration_type[plane];
        let typ = match frame_type {
            RestorationType::Wiener => {
                if self.bool_cdf(|c| &mut c.use_wiener) {
                    RestorationType::Wiener
                } else {
                    RestorationType::None
                }
            }
            RestorationType::Sgrproj => {
                if self.bool_cdf(|c| &mut c.use_sgrproj) {
                    RestorationType::Sgrproj
                } else {
                    RestorationType::None
                }
            }
            _ => match self.sym(|c| &mut c.restoration_type) {
                0 => RestorationType::None,
                1 => RestorationType::Wiener,
                _ => RestorationType::Sgrproj,
            },
        };
        let mut unit = crate::decoder::av1::decode::state::LrUnit {
            typ,
            ..Default::default()
        };
        match typ {
            RestorationType::Wiener => {
                for pass in 0..2 {
                    let first = if plane > 0 { 1 } else { 0 };
                    for j in first..3 {
                        let min = WIENER_TAPS_MIN[j] as i32;
                        let max = WIENER_TAPS_MAX[j] as i32;
                        let k = WIENER_TAPS_K[j] as u32;
                        let v = self.decode_signed_subexp_with_ref_bool(
                            min,
                            max + 1,
                            k,
                            self.t.ref_lr_wiener[plane][pass][j],
                        );
                        unit.wiener[pass][j] = v;
                        self.t.ref_lr_wiener[plane][pass][j] = v;
                    }
                }
            }
            RestorationType::Sgrproj => {
                let set = self.lit(SGRPROJ_PARAMS_BITS) as usize;
                unit.sgr_set = set as u8;
                for i in 0..2 {
                    let radius = sgr_radius(set, i);
                    let min = SGRPROJ_XQD_MIN[i] as i32;
                    let max = SGRPROJ_XQD_MAX[i] as i32;
                    let v = if radius != 0 {
                        self.decode_signed_subexp_with_ref_bool(
                            min,
                            max + 1,
                            SGRPROJ_PRJ_SUBEXP_K,
                            self.t.ref_sgr_xqd[plane][i],
                        )
                    } else if i == 1 {
                        ((1 << SGRPROJ_PRJ_BITS) - self.t.ref_sgr_xqd[plane][0]).clamp(min, max)
                    } else {
                        0
                    };
                    unit.sgr_xqd[i] = v;
                    self.t.ref_sgr_xqd[plane][i] = v;
                }
            }
            _ => {}
        }
        let lp = &mut self.s.lr[plane];
        let idx = unit_row * lp.unit_cols + unit_col;
        lp.units[idx] = unit;
    }

    pub(crate) fn decode_signed_subexp_with_ref_bool(
        &mut self,
        low: i32,
        high: i32,
        k: u32,
        r: i32,
    ) -> i32 {
        let x = self.decode_unsigned_subexp_with_ref_bool(high - low, k, r - low);
        x + low
    }

    pub(crate) fn decode_unsigned_subexp_with_ref_bool(&mut self, mx: i32, k: u32, r: i32) -> i32 {
        let v = self.decode_subexp_bool(mx, k);
        if (r << 1) <= mx {
            inverse_recenter(r, v)
        } else {
            mx - 1 - inverse_recenter(mx - 1 - r, v)
        }
    }

    pub(crate) fn decode_subexp_bool(&mut self, num_syms: i32, k: u32) -> i32 {
        let mut i = 0;
        let mut mk = 0;
        loop {
            let b2 = if i > 0 { k + i - 1 } else { k };
            let a = 1 << b2;
            if num_syms <= mk + 3 * a {
                return self.t.msac.ns((num_syms - mk) as u32) as i32 + mk;
            } else if self.lit(1) != 0 {
                i += 1;
                mk += a;
            } else {
                return self.lit(b2) as i32 + mk;
            }
        }
    }

    pub(crate) fn decode_partition(&mut self, r: usize, c: usize, bsize: BlockSize) -> Result<()> {
        let f = self.f;
        if r >= f.mi_rows || c >= f.mi_cols {
            return Ok(());
        }
        if bsize == BlockSize::B4x4 {
            return self.decode_block(r, c, BlockSize::B4x4);
        }
        let avail_u = r > self.t.mi_row_start;
        let avail_l = c > self.t.mi_col_start;
        let num4x4 = bw4(bsize);
        let half = num4x4 >> 1;
        let has_rows = r + half < f.mi_rows;
        let has_cols = c + half < f.mi_cols;
        let bsl = MI_WIDTH_LOG2[bsize as usize];
        let above = avail_u && MI_WIDTH_LOG2[self.s.mi.get(r - 1, c).bsize as usize] < bsl;
        let left = avail_l && MI_HEIGHT_LOG2[self.s.mi.get(r, c - 1).bsize as usize] < bsl;
        let ctx = left as usize * 2 + above as usize;
        let partition = if has_rows && has_cols {
            let p = match bsl {
                1 => self.sym(|cd| &mut cd.partition_w8[ctx]),
                2 => self.sym(|cd| &mut cd.partition_w16[ctx]),
                3 => self.sym(|cd| &mut cd.partition_w32[ctx]),
                4 => self.sym(|cd| &mut cd.partition_w64[ctx]),
                _ => self.sym(|cd| &mut cd.partition_w128[ctx]),
            };
            Partition::from_u8(p as u8)
        } else if has_cols || has_rows {
            let pcdf: &[u16] = match bsl {
                2 => &cdf!(self).partition_w16[ctx],
                3 => &cdf!(self).partition_w32[ctx],
                4 => &cdf!(self).partition_w64[ctx],
                _ => &cdf!(self).partition_w128[ctx],
            };
            let p = |i: Partition| -> u32 {
                let i = i as usize;
                pcdf[i] as u32 - if i > 0 { pcdf[i - 1] as u32 } else { 0 }
            };
            use Partition::*;
            let psum = if has_cols {
                // split_or_horz
                let mut s = p(Vert) + p(Split) + p(HorzA) + p(VertA) + p(VertB);
                if bsize != BlockSize::B128x128 {
                    s += p(Vert4);
                }
                s
            } else {
                // split_or_vert
                let mut s = p(Horz) + p(Split) + p(HorzA) + p(HorzB) + p(VertA);
                if bsize != BlockSize::B128x128 {
                    s += p(Horz4);
                }
                s
            };
            let mut tmp = [(32768 - psum) as u16, 32768, 0];
            let split = self.t.msac.symbol(&mut tmp) != 0;
            if split {
                Split
            } else if has_cols {
                Horz
            } else {
                Vert
            }
        } else {
            Partition::Split
        };
        let sub_size = PARTITION_SUBSIZE[partition as usize][bsize as usize]
            .ok_or(Error::Invalid("partition"))?;
        // Conformance (5.11.4): get_plane_residual_size(subSize, 1) is
        // not BLOCK_INVALID (e.g. 4:2:2 B4x16). Else chroma areas would be
        // left unwritten.
        if !f.sub_valid[sub_size as usize] {
            return Err(Error::Invalid("subsampled size"));
        }
        use Partition::*;
        match partition {
            None => self.decode_block(r, c, sub_size)?,
            Horz => {
                self.decode_block(r, c, sub_size)?;
                if has_rows {
                    self.decode_block(r + half, c, sub_size)?;
                }
            }
            Vert => {
                self.decode_block(r, c, sub_size)?;
                if has_cols {
                    self.decode_block(r, c + half, sub_size)?;
                }
            }
            Split => {
                self.decode_partition(r, c, sub_size)?;
                self.decode_partition(r, c + half, sub_size)?;
                self.decode_partition(r + half, c, sub_size)?;
                self.decode_partition(r + half, c + half, sub_size)?;
            }
            HorzA => {
                let split_size = PARTITION_SUBSIZE[Partition::Split as usize][bsize as usize]
                    .unwrap_or(BlockSize::B4x4);
                self.decode_block(r, c, split_size)?;
                self.decode_block(r, c + half, split_size)?;
                self.decode_block(r + half, c, sub_size)?;
            }
            HorzB => {
                let split_size = PARTITION_SUBSIZE[Partition::Split as usize][bsize as usize]
                    .unwrap_or(BlockSize::B4x4);
                self.decode_block(r, c, sub_size)?;
                self.decode_block(r + half, c, split_size)?;
                self.decode_block(r + half, c + half, split_size)?;
            }
            VertA => {
                let split_size = PARTITION_SUBSIZE[Partition::Split as usize][bsize as usize]
                    .unwrap_or(BlockSize::B4x4);
                self.decode_block(r, c, split_size)?;
                self.decode_block(r + half, c, split_size)?;
                self.decode_block(r, c + half, sub_size)?;
            }
            VertB => {
                let split_size = PARTITION_SUBSIZE[Partition::Split as usize][bsize as usize]
                    .unwrap_or(BlockSize::B4x4);
                self.decode_block(r, c, sub_size)?;
                self.decode_block(r, c + half, split_size)?;
                self.decode_block(r + half, c + half, split_size)?;
            }
            Horz4 => {
                let quarter = half >> 1;
                for i in 0..4 {
                    let rr = r + quarter * i;
                    if i < 3 || rr < f.mi_rows {
                        self.decode_block(rr, c, sub_size)?;
                    }
                }
            }
            Vert4 => {
                let quarter = half >> 1;
                for i in 0..4 {
                    let cc = c + quarter * i;
                    if i < 3 || cc < f.mi_cols {
                        self.decode_block(r, cc, sub_size)?;
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn decode_block(&mut self, r: usize, c: usize, bsize: BlockSize) -> Result<()> {
        let f = self.f;
        let bw = bw4(bsize);
        let bh = bh4(bsize);
        let has_chroma = if (bh == 1 && f.ss_y == 1 && (r & 1) == 0)
            || (bw == 1 && f.ss_x == 1 && (c & 1) == 0)
        {
            false
        } else {
            f.num_planes > 1
        };
        let avail_u = r > self.t.mi_row_start;
        let avail_l = c > self.t.mi_col_start;
        let (avail_u_chroma, avail_l_chroma) = if has_chroma {
            (
                if f.ss_y == 1 && bh == 1 { r > self.t.mi_row_start + 1 } else { avail_u },
                if f.ss_x == 1 && bw == 1 { c > self.t.mi_col_start + 1 } else { avail_l },
            )
        } else {
            (false, false)
        };
        self.b.mi_row = r;
        self.b.mi_col = c;
        self.b.bsize = bsize;
        self.b.bw4 = bw;
        self.b.bh4 = bh;
        self.b.has_chroma = has_chroma;
        self.b.avail_u = avail_u;
        self.b.avail_l = avail_l;
        self.b.avail_u_chroma = avail_u_chroma;
        self.b.avail_l_chroma = avail_l_chroma;
        self.b.above_mi = if avail_u { *self.s.mi.get(r - 1, c) } else { MiInfo::default() };
        self.b.left_mi = if avail_l { *self.s.mi.get(r, c - 1) } else { MiInfo::default() };
        self.b.uv_mode = IntraMode::DC_PRED;
        self.b.angle_delta_y = 0;
        self.b.angle_delta_uv = 0;
        self.b.use_filter_intra = false;
        self.b.mv = [0; 2];
        self.intra_frame_mode_info()?;
        if (self.b.palette_size_y | self.b.palette_size_uv) > 0 {
            self.palette_tokens();
        }
        self.read_block_tx_size();
        if self.b.skip {
            self.reset_block_context();
        }
        self.store_mode_info();
        if self.b.is_inter {
            self.predict_intrabc()?;
        } else if self.recon {
            self.setup_intra();
        }
        self.residual()?;
        Ok(())
    }

    pub(crate) fn reset_block_context(&mut self) {
        let f = self.f;
        let b = &self.b;
        let x4 = b.mi_col - self.s.above.col0;
        let ly = b.mi_row & f.sb_mask_y[0];
        let [al0, al1, al2] = &mut self.s.above.level;
        let [ad0, ad1, ad2] = &mut self.s.above.dc;
        let [ll0, ll1, ll2] = &mut self.s.left.level;
        let [ld0, ld1, ld2] = &mut self.s.left.dc;
        zero_ctx2(al0, ad0, x4, b.bw4);
        zero_ctx2(ll0, ld0, ly, b.bh4);
        if b.has_chroma {
            let (sx, sy) = (f.ss_x, f.ss_y);
            let ux4 = x4 >> sx;
            let uw4 = (b.bw4 + sx) >> sx;
            let uly = (b.mi_row >> sy) & f.sb_mask_y[1];
            let uh4 = (b.bh4 + sy) >> sy;
            zero_ctx4(al1, ad1, al2, ad2, ux4, uw4);
            zero_ctx4(ll1, ld1, ll2, ld2, uly, uh4);
        }
    }

    /// Writes the per-4x4 mode info of the current block.
    pub(crate) fn store_mode_info(&mut self) {
        let f = self.f;
        let b = &self.b;
        let r_end = (b.mi_row + b.bh4).min(f.mi_rows);
        let tx_size = b.tx_size as u8;
        let info = MiInfo {
            bsize: b.bsize as u8,
            y_mode: b.y_mode as u8,
            uv_mode: b.uv_mode as u8,
            skip: b.skip,
            is_inter: b.is_inter,
            segment_id: b.segment_id as u8,
            inter_tx_size: tx_size,
            palette_sizes: (b.palette_size_y as u8) | ((b.palette_size_uv as u8) << 4),
        };
        let uniform_tx = !self.var_tx();
        let mi = &mut self.s.mi;
        let col0 = mi.col0;
        let mc0 = b.mi_col - col0;
        if !mi.lf_tx[0].is_empty() {
            let y_tx = if b.lossless { TxSize::Tx4x4 as u8 } else { tx_size };
            if uniform_tx {
                fill_mi_rows2(
                    &mut mi.data,
                    &mut mi.lf_tx[0],
                    mi.stride,
                    mi.row_mask,
                    b.mi_row,
                    r_end,
                    mc0,
                    b.bw4,
                    info,
                    y_tx,
                );
            } else {
                let mc_end = (b.mi_col + b.bw4).min(f.mi_cols) - col0;
                for row in b.mi_row..r_end {
                    for m in &mut mi.row_mut(row)[mc0..mc_end] {
                        *m = MiInfo {
                            inter_tx_size: m.inter_tx_size,
                            ..info
                        };
                    }
                }
                fill_mi_rows(
                    &mut mi.lf_tx[0],
                    mi.stride,
                    mi.row_mask,
                    b.mi_row,
                    r_end,
                    mc0,
                    b.bw4,
                    y_tx,
                );
            }
            if b.has_chroma && !mi.lf_tx[1].is_empty() {
                let uv_tx = if b.lossless {
                    TxSize::Tx4x4 as u8
                } else {
                    f.uv_tx_size[b.bsize as usize] as u8
                };
                let (sx, sy) = (f.ss_x, f.ss_y);
                let r4_0 = b.mi_row >> sy;
                let r4_end = (r4_0 + ((b.bh4 + sy) >> sy)).min((f.mi_rows + sy) >> sy);
                let c4_0 = (b.mi_col >> sx) - (col0 >> sx);
                let uv_w = (b.bw4 + sx) >> sx;
                fill_mi_rows(
                    &mut mi.lf_tx[1],
                    mi.lf_stride[1],
                    mi.lf_mask[1],
                    r4_0,
                    r4_end,
                    c4_0,
                    uv_w,
                    uv_tx,
                );
            }
            if !mi.delta_lf.is_empty() {
                let dlf = [
                    self.t.delta_lf[0] as i8,
                    self.t.delta_lf[1] as i8,
                    self.t.delta_lf[2] as i8,
                    self.t.delta_lf[3] as i8,
                ];
                fill_mi_rows(
                    &mut mi.delta_lf,
                    mi.stride,
                    mi.row_mask,
                    b.mi_row,
                    r_end,
                    mc0,
                    b.bw4,
                    dlf,
                );
            }
        } else if uniform_tx {
            if !mi.mv.is_empty() {
                let mv = [b.mv[0] as i16, b.mv[1] as i16];
                fill_mi_rows2(
                    &mut mi.data,
                    &mut mi.mv,
                    mi.stride,
                    mi.row_mask,
                    b.mi_row,
                    r_end,
                    mc0,
                    b.bw4,
                    info,
                    mv,
                );
            } else {
                fill_mi_rows(
                    &mut mi.data,
                    mi.stride,
                    mi.row_mask,
                    b.mi_row,
                    r_end,
                    mc0,
                    b.bw4,
                    info,
                );
            }
        } else {
            let mc_end = (b.mi_col + b.bw4).min(f.mi_cols) - col0;
            for row in b.mi_row..r_end {
                for m in &mut mi.row_mut(row)[mc0..mc_end] {
                    *m = MiInfo {
                        inter_tx_size: m.inter_tx_size,
                        ..info
                    };
                }
            }
            if !mi.mv.is_empty() {
                let mv = [b.mv[0] as i16, b.mv[1] as i16];
                fill_mi_rows(
                    &mut mi.mv,
                    mi.stride,
                    mi.row_mask,
                    b.mi_row,
                    r_end,
                    mc0,
                    b.bw4,
                    mv,
                );
            }
        }
        if (b.palette_size_y | b.palette_size_uv) > 0 {
            self.store_palette_context(r_end);
        }
    }

    #[inline(never)]
    fn store_palette_context(&mut self, r_end: usize) {
        let f = self.f;
        let b = &self.b;
        let c_end = (b.mi_col + b.bw4).min(f.mi_cols);
        let acol0 = self.s.above.col0;
        for col in b.mi_col..c_end {
            self.s.above.palette[0][col - acol0] = b.palette_y;
            self.s.above.palette[1][col - acol0] = b.palette_u;
        }
        let mask = f.sb4 - 1;
        for row in b.mi_row..r_end {
            self.s.left.palette[0][row & mask] = b.palette_y;
            self.s.left.palette[1][row & mask] = b.palette_u;
        }
    }

    #[inline(always)]
    pub(crate) fn seg_feature_active(&self, feature: usize) -> bool {
        let seg = &self.f.fh.seg;
        seg.enabled && seg.feature_enabled[self.b.segment_id][feature]
    }

    pub(crate) fn intra_frame_mode_info(&mut self) -> Result<()> {
        let fh = self.f.fh;
        self.b.skip = false;
        if fh.seg.seg_id_pre_skip {
            self.intra_segment_id()?;
        }
        // read_skip
        if fh.seg.seg_id_pre_skip && self.seg_feature_active(SEG_LVL_SKIP) {
            self.b.skip = true;
        } else {
            let ctx = self.b.above_mi.skip as usize + self.b.left_mi.skip as usize;
            self.b.skip = self.bool_cdf(|c| &mut c.skip[ctx]);
        }
        if !fh.seg.seg_id_pre_skip {
            self.intra_segment_id()?;
        }
        self.read_cdef();
        if self.read_deltas {
            if self.b.bsize != self.f.sb_size || !self.b.skip {
                self.read_delta_qindex();
                self.read_delta_lf();
            }
            self.read_deltas = false;
        }
        self.b.use_intrabc = fh.allow_intrabc && self.bool_cdf(|c| &mut c.intrabc);
        if self.b.use_intrabc {
            self.b.is_inter = true;
            self.b.y_mode = IntraMode::DC_PRED;
            self.b.uv_mode = IntraMode::DC_PRED;
            self.b.palette_size_y = 0;
            self.b.palette_size_uv = 0;
            self.intrabc_mode_info()?;
        } else {
            self.b.is_inter = false;
            let actx = INTRA_MODE_CONTEXT[self.b.above_mi.y_mode as usize] as usize;
            let lctx = INTRA_MODE_CONTEXT[self.b.left_mi.y_mode as usize] as usize;
            let y_mode = self.sym(|cd| &mut cd.intra_frame_y_mode[actx][lctx]);
            self.b.y_mode = IntraMode::from_u8(y_mode as u8);
            self.b.angle_delta_y = self.intra_angle_info(self.b.y_mode);
            let bs = self.b.bsize as usize;
            if self.b.has_chroma {
                let cfl_allowed = if self.b.lossless {
                    self.f.uv_residual_size[bs] == BlockSize::B4x4
                } else {
                    (BLOCK_WIDTH[bs] | BLOCK_HEIGHT[bs]) < 64
                };
                let uv = if cfl_allowed {
                    self.sym(|cd| &mut cd.uv_mode_cfl_allowed[y_mode])
                } else {
                    self.sym(|cd| &mut cd.uv_mode_cfl_not_allowed[y_mode])
                };
                self.b.uv_mode = IntraMode::from_u8(uv as u8);
                if self.b.uv_mode == IntraMode::UV_CFL_PRED {
                    self.read_cfl_alphas();
                }
                self.b.angle_delta_uv = self.intra_angle_info(self.b.uv_mode);
            }
            self.b.palette_size_y = 0;
            self.b.palette_size_uv = 0;
            if fh.allow_screen_content_tools
                && self.b.bsize >= BlockSize::B8x8
                && (BLOCK_WIDTH[bs] | BLOCK_HEIGHT[bs]) < 128
            {
                self.palette_mode_info();
            }
            self.filter_intra_mode_info();
        }
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn intra_segment_id(&mut self) -> Result<()> {
        if self.f.fh.seg.enabled {
            self.read_segment_id()?;
        } else {
            self.b.segment_id = 0;
        }
        self.b.lossless = self.f.fh.lossless_array[self.b.segment_id];
        Ok(())
    }

    pub(crate) fn read_segment_id(&mut self) -> Result<()> {
        let (r, c) = (self.b.mi_row, self.b.mi_col);
        let (au, al) = (self.b.avail_u, self.b.avail_l);
        let prev_ul: i32 =
            if au && al { self.s.mi.get(r - 1, c - 1).segment_id as i32 } else { -1 };
        let prev_u: i32 = if au { self.b.above_mi.segment_id as i32 } else { -1 };
        let prev_l: i32 = if al { self.b.left_mi.segment_id as i32 } else { -1 };
        let pred = if prev_u == -1 {
            if prev_l == -1 {
                0
            } else {
                prev_l
            }
        } else if prev_l == -1 {
            prev_u
        } else if prev_ul == prev_u {
            prev_u
        } else {
            prev_l
        };
        if self.b.skip {
            self.b.segment_id = pred as usize;
        } else {
            let ctx = if prev_ul < 0 {
                0
            } else if prev_ul == prev_u && prev_ul == prev_l {
                2
            } else if prev_ul == prev_u || prev_ul == prev_l || prev_u == prev_l {
                1
            } else {
                0
            };
            let v = self.sym(|cd| &mut cd.segment_id[ctx]) as i32;
            let max = self.f.fh.seg.last_active_seg_id as i32 + 1;
            let id = neg_deinterleave(v, pred, max);
            // Conformance: 0 <= segment_id <= LastActiveSegId (6.10.8).
            if !(0..max).contains(&id) {
                return Err(Error::Invalid("segment_id"));
            }
            self.b.segment_id = id as usize;
        }
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn read_cdef(&mut self) {
        let fh = self.f.fh;
        if self.b.skip || fh.coded_lossless || !self.f.seq.enable_cdef || fh.allow_intrabc {
            return;
        }
        let r = self.b.mi_row & !15;
        let c = self.b.mi_col & !15;
        if self.s.cdef_idx_at(r, c) == -1 {
            let v = self.lit(fh.cdef.bits as u32) as i8;
            let w4 = self.b.bw4;
            let h4 = self.b.bh4;
            let mut i = r;
            while i < r + h4 {
                let mut j = c;
                while j < c + w4 {
                    self.s.set_cdef_idx(i, j, v);
                    j += 16;
                }
                i += 16;
            }
        }
    }

    #[inline(never)]
    pub(crate) fn read_delta_qindex(&mut self) {
        let mut abs = self.sym(|c| &mut c.delta_q) as i32;
        if abs == DELTA_Q_SMALL as i32 {
            let rem_bits = self.lit(3) + 1;
            let abs_bits = self.lit(rem_bits) as i32;
            abs = abs_bits + (1 << rem_bits) + 1;
        }
        if abs != 0 {
            let sign = self.lit(1) != 0;
            let reduced = if sign { -abs } else { abs };
            self.t.current_qindex =
                (self.t.current_qindex + (reduced << self.f.fh.delta_q_res)).clamp(1, 255);
        }
    }

    #[inline(never)]
    pub(crate) fn read_delta_lf(&mut self) {
        let fh = self.f.fh;
        if fh.delta_lf_present {
            let count = if fh.delta_lf_multi {
                if self.f.num_planes > 1 {
                    FRAME_LF_COUNT
                } else {
                    FRAME_LF_COUNT - 2
                }
            } else {
                1
            };
            for i in 0..count {
                let mut abs = if fh.delta_lf_multi {
                    self.sym(|c| &mut c.delta_lf_multi[i])
                } else {
                    self.sym(|c| &mut c.delta_lf)
                } as i32;
                if abs == DELTA_LF_SMALL as i32 {
                    let n = self.lit(3) + 1;
                    let bits = self.lit(n) as i32;
                    abs = bits + (1 << n) + 1;
                }
                if abs != 0 {
                    let sign = self.lit(1) != 0;
                    let reduced = if sign { -abs } else { abs };
                    self.t.delta_lf[i] = (self.t.delta_lf[i] + (reduced << fh.delta_lf_res))
                        .clamp(-MAX_LOOP_FILTER, MAX_LOOP_FILTER);
                }
            }
        }
    }

    #[inline(always)]
    pub(crate) fn intra_angle_info(&mut self, mode: IntraMode) -> i32 {
        let m = (mode as usize).wrapping_sub(IntraMode::V_PRED as usize);
        if self.b.bsize >= BlockSize::B8x8 && m < 8 {
            self.sym(|c| &mut c.angle_delta[m]) as i32 - 3
        } else {
            0
        }
    }

    pub(crate) fn read_cfl_alphas(&mut self) {
        let signs = self.sym(|c| &mut c.cfl_sign) as i32;
        let sign_u = (signs + 1) / 3;
        let sign_v = (signs + 1) % 3;
        // CFL_SIGN_ZERO = 0, CFL_SIGN_NEG = 1, CFL_SIGN_POS = 2
        if sign_u != 0 {
            let ctx = ((sign_u - 1) * 3 + sign_v) as usize;
            let a = self.sym(|c| &mut c.cfl_alpha[ctx]) as i32 + 1;
            self.b.cfl_alpha_u = if sign_u == 1 { -a } else { a };
        } else {
            self.b.cfl_alpha_u = 0;
        }
        if sign_v != 0 {
            let ctx = ((sign_v - 1) * 3 + sign_u) as usize;
            let a = self.sym(|c| &mut c.cfl_alpha[ctx]) as i32 + 1;
            self.b.cfl_alpha_v = if sign_v == 1 { -a } else { a };
        } else {
            self.b.cfl_alpha_v = 0;
        }
    }

    #[inline(always)]
    pub(crate) fn filter_intra_mode_info(&mut self) {
        let bs = self.b.bsize as usize;
        if self.b.y_mode == IntraMode::DC_PRED
            && self.f.seq.enable_filter_intra
            && self.b.palette_size_y == 0
            && (BLOCK_WIDTH[bs] | BLOCK_HEIGHT[bs]) < 64
        {
            self.b.use_filter_intra = self.bool_cdf(|c| &mut c.filter_intra[bs]);
            if self.b.use_filter_intra {
                self.b.filter_intra_mode = self.sym(|c| &mut c.filter_intra_mode) as u8;
            }
        }
    }

    pub(crate) fn get_palette_cache(&self, plane: usize, cache: &mut [u16; 16]) -> usize {
        let (r, c) = (self.b.mi_row, self.b.mi_col);
        let above_n = if (r & 15) != 0 { self.b.above_mi.palette_size(plane) } else { 0 };
        let left_n = self.b.left_mi.palette_size(plane);
        let above = &self.s.above.palette[plane][c - self.s.above.col0];
        let left = &self.s.left.palette[plane][r & (self.f.sb4 - 1)];
        let (mut ai, mut li, mut n) = (0, 0, 0);
        while ai < above_n && li < left_n {
            let ac = above[ai];
            let lc = left[li];
            if lc < ac {
                if n == 0 || lc != cache[n - 1] {
                    cache[n] = lc;
                    n += 1;
                }
                li += 1;
            } else {
                if n == 0 || ac != cache[n - 1] {
                    cache[n] = ac;
                    n += 1;
                }
                ai += 1;
                if lc == ac {
                    li += 1;
                }
            }
        }
        while ai < above_n {
            let v = above[ai];
            ai += 1;
            if n == 0 || v != cache[n - 1] {
                cache[n] = v;
                n += 1;
            }
        }
        while li < left_n {
            let v = left[li];
            li += 1;
            if n == 0 || v != cache[n - 1] {
                cache[n] = v;
                n += 1;
            }
        }
        n
    }

    pub(crate) fn read_palette_colors(
        &mut self,
        plane: usize,
        size: usize,
        delta_offset: i32,
    ) -> [u16; 8] {
        let bd = self.f.bd;
        let mut colors = [0u16; 8];
        let mut cache = [0u16; 16];
        let cache_n = self.get_palette_cache(plane, &mut cache);
        let mut idx = 0;
        let mut i = 0;
        while i < cache_n && idx < size {
            if self.lit(1) != 0 {
                colors[idx] = cache[i];
                idx += 1;
            }
            i += 1;
        }
        if idx < size {
            colors[idx] = self.lit(bd) as u16;
            idx += 1;
        }
        let mut palette_bits = 0;
        if idx < size {
            let min_bits = bd - 3;
            palette_bits = min_bits + self.lit(2);
        }
        while idx < size {
            let delta = self.lit(palette_bits) as i32 + delta_offset;
            let v = (colors[idx - 1] as i32 + delta).clamp(0, (1 << bd) - 1);
            colors[idx] = v as u16;
            let range = (1 << bd) - v - delta_offset;
            palette_bits = palette_bits.min(ceil_log2(range.max(0) as u32));
            idx += 1;
        }
        colors[..size].sort_unstable();
        colors
    }

    #[inline(never)]
    pub(crate) fn palette_mode_info(&mut self) {
        let bs = self.b.bsize as usize;
        let bsize_ctx = MI_WIDTH_LOG2[bs] as usize + MI_HEIGHT_LOG2[bs] as usize - 2;
        if self.b.y_mode == IntraMode::DC_PRED {
            let ctx = (self.b.above_mi.palette_size(0) > 0) as usize
                + (self.b.left_mi.palette_size(0) > 0) as usize;
            if self.bool_cdf(|cd| &mut cd.palette_y_mode[bsize_ctx][ctx]) {
                let size = self.sym(|cd| &mut cd.palette_y_size[bsize_ctx]) + 2;
                self.b.palette_size_y = size;
                self.b.palette_y = self.read_palette_colors(0, size, 1);
            }
        }
        if self.b.has_chroma && self.b.uv_mode == IntraMode::DC_PRED {
            let ctx = (self.b.palette_size_y > 0) as usize;
            if self.bool_cdf(|cd| &mut cd.palette_uv_mode[ctx]) {
                let size = self.sym(|cd| &mut cd.palette_uv_size[bsize_ctx]) + 2;
                self.b.palette_size_uv = size;
                self.b.palette_u = self.read_palette_colors(1, size, 0);
                // V colors
                let bd = self.f.bd;
                if self.lit(1) != 0 {
                    let min_bits = bd - 4;
                    let max_val = 1i32 << bd;
                    let bits = min_bits + self.lit(2);
                    self.b.palette_v[0] = self.lit(bd) as u16;
                    for idx in 1..size {
                        let mut delta = self.lit(bits) as i32;
                        if delta != 0 && self.lit(1) != 0 {
                            delta = -delta;
                        }
                        let mut val = self.b.palette_v[idx - 1] as i32 + delta;
                        if val < 0 {
                            val += max_val;
                        }
                        if val >= max_val {
                            val -= max_val;
                        }
                        self.b.palette_v[idx] = val.clamp(0, (1 << bd) - 1) as u16;
                    }
                } else {
                    for idx in 0..size {
                        self.b.palette_v[idx] = self.lit(bd) as u16;
                    }
                }
            }
        }
    }

    #[inline(never)]
    pub(crate) fn palette_tokens(&mut self) {
        let f = self.f;
        let bs = self.b.bsize as usize;
        let mut block_h = BLOCK_HEIGHT[bs] as usize;
        let mut block_w = BLOCK_WIDTH[bs] as usize;
        let mut onscreen_h = block_h.min((f.mi_rows - self.b.mi_row) * MI_SIZE);
        let mut onscreen_w = block_w.min((f.mi_cols - self.b.mi_col) * MI_SIZE);
        if self.b.palette_size_y > 0 {
            let n = self.b.palette_size_y;
            Self::read_color_map(
                self.t,
                &mut self.sc.color_map_y,
                n,
                block_w,
                block_h,
                onscreen_w,
                onscreen_h,
                false,
            );
        }
        if self.b.palette_size_uv > 0 {
            let n = self.b.palette_size_uv;
            block_h >>= f.ss_y;
            block_w >>= f.ss_x;
            onscreen_h >>= f.ss_y;
            onscreen_w >>= f.ss_x;
            if block_w < 4 {
                block_w += 2;
                onscreen_w += 2;
            }
            if block_h < 4 {
                block_h += 2;
                onscreen_h += 2;
            }
            Self::read_color_map(
                self.t,
                &mut self.sc.color_map_uv,
                n,
                block_w,
                block_h,
                onscreen_w,
                onscreen_h,
                true,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read_color_map(
        t: &mut TileState<'b>,
        map: &mut [[u8; 64]; 64],
        n: usize,
        block_w: usize,
        block_h: usize,
        onscreen_w: usize,
        onscreen_h: usize,
        uv: bool,
    ) {
        map[0][0] = t.msac.ns(n as u32) as u8;
        let (m, c) = (&mut t.msac, &mut t.cdf);
        let (w, h) = (onscreen_w, onscreen_h);
        // One monomorphized wavefront loop per palette size (the CDF set
        // and the symbol size are fixed for the block).
        match (uv, n) {
            (false, 2) => color_map_n(m, &mut c.palette_size_2_y_color, map, w, h),
            (false, 3) => color_map_n(m, &mut c.palette_size_3_y_color, map, w, h),
            (false, 4) => color_map_n(m, &mut c.palette_size_4_y_color, map, w, h),
            (false, 5) => color_map_n(m, &mut c.palette_size_5_y_color, map, w, h),
            (false, 6) => color_map_n(m, &mut c.palette_size_6_y_color, map, w, h),
            (false, 7) => color_map_n(m, &mut c.palette_size_7_y_color, map, w, h),
            (false, _) => color_map_n(m, &mut c.palette_size_8_y_color, map, w, h),
            (true, 2) => color_map_n(m, &mut c.palette_size_2_uv_color, map, w, h),
            (true, 3) => color_map_n(m, &mut c.palette_size_3_uv_color, map, w, h),
            (true, 4) => color_map_n(m, &mut c.palette_size_4_uv_color, map, w, h),
            (true, 5) => color_map_n(m, &mut c.palette_size_5_uv_color, map, w, h),
            (true, 6) => color_map_n(m, &mut c.palette_size_6_uv_color, map, w, h),
            (true, 7) => color_map_n(m, &mut c.palette_size_7_uv_color, map, w, h),
            (true, _) => color_map_n(m, &mut c.palette_size_8_uv_color, map, w, h),
        }
        for row in map.iter_mut().take(onscreen_h) {
            let v = row[onscreen_w - 1];
            for x in row.iter_mut().take(block_w).skip(onscreen_w) {
                *x = v;
            }
        }
        for i in onscreen_h..block_h {
            map[i] = map[onscreen_h - 1];
        }
    }

    /// True when the transform sizes of the block are read with
    /// `read_var_tx_size` (else `InterTxSizes` equals `TxSize` over the
    /// whole block).
    #[inline(always)]
    fn var_tx(&self) -> bool {
        self.b.is_inter
            && !self.b.skip
            && !self.b.lossless
            && self.b.bsize > BlockSize::B4x4
            && self.f.fh.tx_mode == TxMode::Select
    }

    #[inline(always)]
    pub(crate) fn read_block_tx_size(&mut self) {
        if self.var_tx() {
            let bs = self.b.bsize;
            let (r0, c0) = (self.b.mi_row, self.b.mi_col);
            let max_tx = MAX_TX_SIZE_RECT[bs as usize];
            let tw4 = txw(max_tx) / 4;
            let th4 = txh(max_tx) / 4;
            let mut row = r0;
            while row < r0 + self.b.bh4 {
                let mut col = c0;
                while col < c0 + self.b.bw4 {
                    self.read_var_tx_size(row, col, max_tx, 0);
                    col += tw4;
                }
                row += th4;
            }
        } else {
            // InterTxSizes = TxSize is written by store_mode_info.
            self.read_tx_size(!self.b.skip || !self.b.is_inter);
        }
    }

    #[inline(always)]
    pub(crate) fn read_tx_size(&mut self, allow_select: bool) {
        if self.b.lossless {
            self.b.tx_size = TxSize::Tx4x4;
            return;
        }
        let bs = self.b.bsize as usize;
        let max_rect = MAX_TX_SIZE_RECT[bs];
        self.b.tx_size = max_rect;
        if self.b.bsize > BlockSize::B4x4 && allow_select && self.f.fh.tx_mode == TxMode::Select {
            let max_depth = MAX_TX_DEPTH[bs] as usize;
            let ctx = self.tx_depth_ctx(max_rect);
            let depth = match max_depth {
                4 => self.sym(|c| &mut c.tx_64x64[ctx]),
                3 => self.sym(|c| &mut c.tx_32x32[ctx]),
                2 => self.sym(|c| &mut c.tx_16x16[ctx]),
                _ => self.sym(|c| &mut c.tx_8x8[ctx]),
            };
            self.b.tx_size = SPLIT_TX_BY_DEPTH[max_rect as usize][depth];
        }
    }

    #[inline(always)]
    pub(crate) fn get_above_tx_width(&self, row: usize, col: usize) -> usize {
        if row == self.b.mi_row && !self.b.avail_u {
            return 64;
        }
        let a = self.s.mi.get(row - 1, col);
        if row == self.b.mi_row && a.skip && a.is_inter {
            return BLOCK_WIDTH[a.bsize as usize] as usize;
        }
        txw(TxSize::from_u8(a.inter_tx_size))
    }

    #[inline(always)]
    pub(crate) fn get_left_tx_height(&self, row: usize, col: usize) -> usize {
        if col == self.b.mi_col && !self.b.avail_l {
            return 64;
        }
        let l = self.s.mi.get(row, col - 1);
        if col == self.b.mi_col && l.skip && l.is_inter {
            return BLOCK_HEIGHT[l.bsize as usize] as usize;
        }
        txh(TxSize::from_u8(l.inter_tx_size))
    }

    #[inline(always)]
    pub(crate) fn tx_depth_ctx(&self, max_rect: TxSize) -> usize {
        let max_w = txw(max_rect);
        let max_h = txh(max_rect);
        let above_w = if self.b.avail_u {
            let a = self.b.above_mi;
            if a.is_inter {
                BLOCK_WIDTH[a.bsize as usize] as usize
            } else {
                TX_WIDTH[a.inter_tx_size as usize] as usize
            }
        } else {
            0
        };
        let left_h = if self.b.avail_l {
            let l = self.b.left_mi;
            if l.is_inter {
                BLOCK_HEIGHT[l.bsize as usize] as usize
            } else {
                TX_HEIGHT[l.inter_tx_size as usize] as usize
            }
        } else {
            0
        };
        (above_w >= max_w) as usize + (left_h >= max_h) as usize
    }

    pub(crate) fn read_var_tx_size(&mut self, row: usize, col: usize, tx_sz: TxSize, depth: usize) {
        let f = self.f;
        if row >= f.mi_rows || col >= f.mi_cols {
            return;
        }
        let split = if tx_sz == TxSize::Tx4x4 || depth == MAX_VARTX_DEPTH {
            false
        } else {
            let above = self.get_above_tx_width(row, col) < txw(tx_sz);
            let left = self.get_left_tx_height(row, col) < txh(tx_sz);
            let bs = self.b.bsize as usize;
            let size = 64.min((BLOCK_WIDTH[bs] as usize).max(BLOCK_HEIGHT[bs] as usize));
            let max_tx_sz = find_tx_size(size, size) as usize;
            let sqr_up = TX_SIZE_SQR_UP[tx_sz as usize] as usize;
            let ctx = (sqr_up != max_tx_sz) as usize * 3
                + (4 - max_tx_sz) * 6
                + above as usize
                + left as usize;
            self.bool_cdf(|c| &mut c.txfm_split[ctx])
        };
        let w4 = txw(tx_sz) / 4;
        let h4 = txh(tx_sz) / 4;
        if split {
            let sub = SPLIT_TX_SIZE[tx_sz as usize];
            let step_w = txw(sub) / 4;
            let step_h = txh(sub) / 4;
            let mut i = 0;
            while i < h4 {
                let mut j = 0;
                while j < w4 {
                    self.read_var_tx_size(row + i, col + j, sub, depth + 1);
                    j += step_w;
                }
                i += step_h;
            }
        } else {
            for i in 0..h4 {
                for j in 0..w4 {
                    if row + i < f.mi_rows && col + j < f.mi_cols {
                        self.s.mi.get_mut(row + i, col + j).inter_tx_size = tx_sz as u8;
                    }
                }
            }
            self.b.tx_size = tx_sz;
        }
    }
}

/// Spec `inverse_recenter`.
pub fn inverse_recenter(r: i32, v: i32) -> i32 {
    if v > 2 * r {
        v
    } else if v & 1 != 0 {
        r - ((v + 1) >> 1)
    } else {
        r + (v >> 1)
    }
}

/// Spec `neg_deinterleave`.
pub(crate) fn neg_deinterleave(diff: i32, r: i32, max: i32) -> i32 {
    if r == 0 {
        return diff;
    }
    if r >= max - 1 {
        return max - diff - 1;
    }
    if 2 * r < max {
        if diff <= 2 * r {
            if diff & 1 != 0 {
                return r + ((diff + 1) >> 1);
            } else {
                return r - (diff >> 1);
            }
        }
        diff
    } else {
        if diff <= 2 * (max - r - 1) {
            if diff & 1 != 0 {
                return r + ((diff + 1) >> 1);
            } else {
                return r - (diff >> 1);
            }
        }
        max - (diff + 1)
    }
}

/// The palette color index wavefront of spec `get_color_map` /
/// `palette_tokens` for one block (`map[0][0]` is already read) with
/// `N - 1` colors: the CDF set and the symbol size are constants.
#[inline(always)]
fn color_map_n<const N: usize>(
    msac: &mut crate::decoder::av1::msac::SymbolDecoder<'_>,
    cdfs: &mut [[u16; N]; 5],
    map: &mut [[u8; 64]; 64],
    w: usize,
    h: usize,
) {
    for i in 1..(h + w - 1) {
        let j_max = i.min(w - 1);
        let j_min = (i + 1).saturating_sub(h);
        for j in (j_min..=j_max).rev() {
            let r = i - j;
            let (ctx, order) = color_order::<N>(map, r, j);
            let idx = msac.symbol_n(&mut cdfs[ctx]);
            map[r][j] = order[idx];
        }
    }
}

/// Spec `get_palette_color_context` for a palette of `N - 1` colors,
/// returning (ctx, ColorOrder): the three neighbors (left and top
/// score 2, top-left 1) sorted by score with ties in color order, then the
/// other colors in increasing order. Derived directly from the neighbor
/// equalities like dav1d's `order_palette` (the sorted scores are
/// [2] -> ctx 0, [5] -> 4, [4, 1] -> 3, [3, 2] -> 2, [2, 2, 1] -> 1).
#[inline(always)]
fn color_order<const N: usize>(map: &[[u8; 64]; 64], r: usize, c: usize) -> (usize, [u8; 9]) {
    let mut order = [0u8; 9];
    let (ctx, k, mask) = if c == 0 {
        let t = map[r - 1][c];
        order[0] = t;
        (0, 1, 1u32 << (t & 7))
    } else if r == 0 {
        let l = map[r][c - 1];
        order[0] = l;
        (0, 1, 1u32 << (l & 7))
    } else {
        let l = map[r][c - 1];
        let t = map[r - 1][c];
        let tl = map[r - 1][c - 1];
        let m = (1u32 << (l & 7)) | (1u32 << (t & 7)) | (1u32 << (tl & 7));
        if t == l {
            order[0] = t;
            if t == tl {
                (4, 1, m)
            } else {
                order[1] = tl;
                (3, 2, m)
            }
        } else if t == tl || l == tl {
            order[0] = tl;
            order[1] = if t == tl { l } else { t };
            (2, 2, m)
        } else {
            order[0] = t.min(l);
            order[1] = t.max(l);
            order[2] = tl;
            (1, 3, m)
        }
    };
    // The remaining colors (all neighbors are < N - 1) in increasing order.
    let mut k = k;
    for v in 0..(N - 1) as u8 {
        order[k] = v;
        k += ((mask >> v) & 1 == 0) as usize;
        k = k.min(8);
    }
    (ctx, order)
}

/// Spec `get_palette_color_context`: returns (ctx, ColorOrder) (reference
/// for [`color_order`]).
#[cfg(test)]
pub(crate) fn palette_color_context(
    map: &[[u8; 64]; 64],
    r: usize,
    c: usize,
    n: usize,
) -> (usize, [u8; 8]) {
    let mut scores = [0u32; PALETTE_COLORS];
    let mut order = [0u8, 1, 2, 3, 4, 5, 6, 7];
    if c > 0 {
        scores[map[r][c - 1] as usize] += 2;
    }
    if r > 0 && c > 0 {
        scores[map[r - 1][c - 1] as usize] += 1;
    }
    if r > 0 {
        scores[map[r - 1][c] as usize] += 2;
    }
    for i in 0..PALETTE_NUM_NEIGHBORS {
        let mut max_score = scores[i];
        let mut max_idx = i;
        for j in (i + 1)..n {
            if scores[j] > max_score {
                max_score = scores[j];
                max_idx = j;
            }
        }
        if max_idx != i {
            let max_score = scores[max_idx];
            let max_order = order[max_idx];
            for k in ((i + 1)..=max_idx).rev() {
                scores[k] = scores[k - 1];
                order[k] = order[k - 1];
            }
            scores[i] = max_score;
            order[i] = max_order;
        }
    }
    let mut hash = 0u32;
    for i in 0..PALETTE_NUM_NEIGHBORS {
        hash += scores[i] * PALETTE_COLOR_HASH_MULTIPLIERS[i] as u32;
    }
    let ctx = PALETTE_COLOR_CONTEXT[hash as usize];
    (ctx.max(0) as usize, order)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check<const N: usize>() {
        let n = N - 1;
        let mut map = [[0u8; 64]; 64];
        for t in 0..n as u8 {
            for l in 0..n as u8 {
                for tl in 0..n as u8 {
                    map[0][1] = t;
                    map[1][0] = l;
                    map[0][0] = tl;
                    for (r, c) in [(1, 1), (1, 0), (0, 1)] {
                        let (ctx, order) = color_order::<N>(&map, r, c);
                        let (ctx_ref, order_ref) = palette_color_context(&map, r, c, n);
                        assert_eq!(ctx, ctx_ref, "n {n} t {t} l {l} tl {tl} at {r},{c}");
                        assert_eq!(
                            order[..n],
                            order_ref[..n],
                            "n {n} t {t} l {l} tl {tl} at {r},{c}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn color_order_matches_spec() {
        check::<3>();
        check::<4>();
        check::<5>();
        check::<6>();
        check::<7>();
        check::<8>();
        check::<9>();
    }
}
