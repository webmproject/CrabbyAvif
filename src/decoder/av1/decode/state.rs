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

//! Frame / tile level decoding state shared by the superblock decoder.

use crate::decoder::av1::dsp::lr::SGR_PARAMS;
use crate::decoder::av1::headers::{FrameHeader, RestorationType, SequenceHeader};
use crate::decoder::av1::msac::SymbolDecoder;
use crate::decoder::av1::pixel::Pixel;
use crate::decoder::av1::tables::block::{MAX_TX_SIZE_RECT, SUBSAMPLED_SIZE, TX_HEIGHT, TX_WIDTH};
use crate::decoder::av1::tables::cdf::CdfContext;
use crate::decoder::av1::tables::misc::{SGRPROJ_XQD_MID, WIENER_TAPS_MID};
use crate::decoder::av1::types::*;

use super::buffers::{MiWin, PlaneWin};

/// Immutable per-frame parameters derived from the headers.
#[derive(Clone, Copy)]
pub struct FrameCtx<'a> {
    pub seq: &'a SequenceHeader,
    pub fh: &'a FrameHeader,
    pub bd: u32,
    pub ss_x: usize,
    pub ss_y: usize,
    pub num_planes: usize,
    pub sb128: bool,
    pub sb_size: BlockSize,
    /// Superblock size in 4x4 units (16 or 32).
    pub sb4: usize,
    /// log2 of the superblock size in 4x4 units (4 or 5).
    pub sb_shift: usize,
    /// Superblock size in luma samples (64 or 128).
    pub sb_px: usize,
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub sb_cols: usize,
    pub sb_rows: usize,
    /// `(mi_cols * MI_SIZE) >> sx` for luma (`0`) and chroma (`1`).
    pub max_x: [usize; 2],
    /// `(mi_rows * MI_SIZE) >> sy` for luma (`0`) and chroma (`1`).
    pub max_y: [usize; 2],
    /// `(sb4 - 1) >> sx` for luma (`0`) and chroma (`1`).
    pub sb_mask_x: [usize; 2],
    /// `(sb4 - 1) >> sy` for luma (`0`) and chroma (`1`).
    pub sb_mask_y: [usize; 2],
    /// Whether `SUBSAMPLED_SIZE[bsize][ss_x][ss_y]` is valid (`Some`).
    pub sub_valid: [bool; BlockSize::COUNT],
    /// `plane_residual_size(bsize, ss_x, ss_y)` precomputed for chroma.
    pub uv_residual_size: [BlockSize; BlockSize::COUNT],
    /// `get_tx_size(1, ...)` precomputed for each `bsize` at `(ss_x, ss_y)`.
    pub uv_tx_size: [TxSize; BlockSize::COUNT],
}

const SUB_TABLES: (
    [[[bool; BlockSize::COUNT]; 2]; 2],
    [[[BlockSize; BlockSize::COUNT]; 2]; 2],
    [[[TxSize; BlockSize::COUNT]; 2]; 2],
) = {
    let mut sub_valid = [[[false; BlockSize::COUNT]; 2]; 2];
    let mut uv_res = [[[BlockSize::B4x4; BlockSize::COUNT]; 2]; 2];
    let mut uv_tx = [[[TxSize::Tx4x4; BlockSize::COUNT]; 2]; 2];
    let mut sx = 0;
    while sx < 2 {
        let mut sy = 0;
        while sy < 2 {
            let mut i = 0;
            while i < BlockSize::COUNT {
                let (valid, res_sz) = match SUBSAMPLED_SIZE[i][sx][sy] {
                    Some(sz) => (true, sz),
                    None => (false, BlockSize::B4x4),
                };
                sub_valid[sx][sy][i] = valid;
                uv_res[sx][sy][i] = res_sz;
                let tx = MAX_TX_SIZE_RECT[res_sz as usize];
                let w = TX_WIDTH[tx as usize];
                let h = TX_HEIGHT[tx as usize];
                uv_tx[sx][sy][i] = if w == 64 || h == 64 {
                    if w == 16 {
                        TxSize::Tx16x32
                    } else if h == 16 {
                        TxSize::Tx32x16
                    } else {
                        TxSize::Tx32x32
                    }
                } else {
                    tx
                };
                i += 1;
            }
            sy += 1;
        }
        sx += 1;
    }
    (sub_valid, uv_res, uv_tx)
};

impl<'a> FrameCtx<'a> {
    pub fn new(seq: &'a SequenceHeader, fh: &'a FrameHeader) -> Self {
        let sb128 = seq.use_128x128_superblock;
        let sb4 = if sb128 { 32 } else { 16 };
        let ss_x = (seq.color.subsampling_x as usize) & 1;
        let ss_y = (seq.color.subsampling_y as usize) & 1;
        let max_x0 = fh.mi_cols * MI_SIZE;
        let max_y0 = fh.mi_rows * MI_SIZE;
        let sb_mask = sb4 - 1;
        Self {
            seq,
            fh,
            bd: seq.color.bit_depth,
            ss_x,
            ss_y,
            num_planes: seq.color.num_planes(),
            sb128,
            sb_size: if sb128 { BlockSize::B128x128 } else { BlockSize::B64x64 },
            sb4,
            sb_shift: if sb128 { 5 } else { 4 },
            sb_px: sb4 * 4,
            mi_cols: fh.mi_cols,
            mi_rows: fh.mi_rows,
            sb_cols: fh.mi_cols.div_ceil(sb4),
            sb_rows: fh.mi_rows.div_ceil(sb4),
            max_x: [max_x0, max_x0 >> ss_x],
            max_y: [max_y0, max_y0 >> ss_y],
            sb_mask_x: [sb_mask, sb_mask >> ss_x],
            sb_mask_y: [sb_mask, sb_mask >> ss_y],
            sub_valid: SUB_TABLES.0[ss_x][ss_y],
            uv_residual_size: SUB_TABLES.1[ss_x][ss_y],
            uv_tx_size: SUB_TABLES.2[ss_x][ss_y],
        }
    }

    #[inline(always)]
    pub fn sub(&self, plane: usize) -> (usize, usize) {
        let c = (plane != 0) as usize;
        (self.ss_x & c, self.ss_y & c)
    }
}

/// State of one tile that persists across superblock rows. Tiles of the
/// same tile row are decoded interleaved, one superblock row at a time.
pub struct TileState<'a> {
    pub msac: SymbolDecoder<'a>,
    pub cdf: Box<CdfContext>,
    pub mi_row_start: usize,
    pub mi_row_end: usize,
    pub mi_col_start: usize,
    pub mi_col_end: usize,
    pub current_qindex: i32,
    pub delta_lf: [i32; FRAME_LF_COUNT],
    pub ref_sgr_xqd: [[i32; 2]; 3],
    pub ref_lr_wiener: [[[i32; 3]; 2]; 3],
}

impl<'a> TileState<'a> {
    pub fn new(
        data: &'a [u8],
        fh: &FrameHeader,
        seq: &SequenceHeader,
        rows: (usize, usize),
        cols: (usize, usize),
    ) -> Self {
        Self::with_cdf(data, fh, seq, rows, cols, None)
    }

    pub fn with_cdf(
        data: &'a [u8],
        fh: &FrameHeader,
        seq: &SequenceHeader,
        rows: (usize, usize),
        cols: (usize, usize),
        cdf: Option<Box<CdfContext>>,
    ) -> Self {
        const INIT_WIENER: [[[i32; 3]; 2]; 3] = [[[
            WIENER_TAPS_MID[0] as i32,
            WIENER_TAPS_MID[1] as i32,
            WIENER_TAPS_MID[2] as i32,
        ]; 2]; 3];
        let mut cdf = cdf.unwrap_or_else(|| Box::new(CdfContext::new(0)));
        cdf.init_for_frame(fh, seq);
        Self {
            msac: SymbolDecoder::new(data, fh.disable_cdf_update),
            cdf,
            mi_row_start: rows.0,
            mi_row_end: rows.1,
            mi_col_start: cols.0,
            mi_col_end: cols.1,
            current_qindex: fh.quant.base_q_idx as i32,
            delta_lf: [0; FRAME_LF_COUNT],
            ref_sgr_xqd: [[SGRPROJ_XQD_MID[0] as i32, SGRPROJ_XQD_MID[1] as i32]; 3],
            ref_lr_wiener: INIT_WIENER,
        }
    }

    pub fn stub_with_cdf(
        rows: (usize, usize),
        cols: (usize, usize),
        cdf: Option<Box<CdfContext>>,
    ) -> Self {
        const INIT_WIENER: [[[i32; 3]; 2]; 3] = [[[
            WIENER_TAPS_MID[0] as i32,
            WIENER_TAPS_MID[1] as i32,
            WIENER_TAPS_MID[2] as i32,
        ]; 2]; 3];
        Self {
            msac: SymbolDecoder::new(&[], true),
            cdf: cdf.unwrap_or_else(|| Box::new(CdfContext::new(0))),
            mi_row_start: rows.0,
            mi_row_end: rows.1,
            mi_col_start: cols.0,
            mi_col_end: cols.1,
            current_qindex: 0,
            delta_lf: [0; FRAME_LF_COUNT],
            ref_sgr_xqd: [[SGRPROJ_XQD_MID[0] as i32, SGRPROJ_XQD_MID[1] as i32]; 3],
            ref_lr_wiener: INIT_WIENER,
        }
    }

    pub fn stub(rows: (usize, usize), cols: (usize, usize)) -> Self {
        Self::stub_with_cdf(rows, cols, None)
    }
}

/// Loop restoration parameters of one restoration unit.
#[derive(Clone, Copy, Default, Debug)]
pub struct LrUnit {
    pub typ: RestorationType,
    pub wiener: [[i32; 3]; 2],
    pub sgr_set: u8,
    pub sgr_xqd: [i32; 2],
}

/// Loop restoration units of one plane (the whole frame; units are at
/// least 32x32 so this is small).
#[derive(Clone, Default)]
pub struct LrPlane {
    pub unit_size: usize,
    pub unit_rows: usize,
    pub unit_cols: usize,
    pub units: Vec<LrUnit>,
}

/// Spec `count_units_in_frame`.
pub fn count_units_in_frame(unit_size: usize, frame_size: usize) -> usize {
    ((frame_size + (unit_size >> 1)) / unit_size).max(1)
}

impl LrPlane {
    pub fn new(fh: &FrameHeader, plane: usize, ss_x: usize, ss_y: usize) -> Self {
        let mut lp = Self::default();
        lp.reset(fh, plane, ss_x, ss_y);
        lp
    }

    pub fn reset(&mut self, fh: &FrameHeader, plane: usize, ss_x: usize, ss_y: usize) {
        if fh.lr.frame_restoration_type[plane] == RestorationType::None {
            self.unit_size = 0;
            self.unit_rows = 0;
            self.unit_cols = 0;
            self.units.clear();
            return;
        }
        let unit_size = fh.lr.loop_restoration_size[plane];
        let (sx, sy) = if plane == 0 { (0, 0) } else { (ss_x, ss_y) };
        let unit_rows = count_units_in_frame(unit_size, (fh.frame_height + sy) >> sy);
        let unit_cols = count_units_in_frame(unit_size, (fh.upscaled_width + sx) >> sx);
        let n = unit_rows * unit_cols;
        let pl = self.units.len().min(n);
        self.units[..pl].fill(LrUnit::default());
        self.units.resize(n, LrUnit::default());
        self.unit_size = unit_size;
        self.unit_rows = unit_rows;
        self.unit_cols = unit_cols;
    }
}

/// Sgr_Params radius of pass `i` for set `set`.
pub fn sgr_radius(set: usize, pass: usize) -> i32 {
    SGR_PARAMS[set][pass * 2] as i32
}

/// Above context arrays (indexed by 4x4 column of the plane relative to `col0 >> sx`).
#[derive(Default)]
pub struct AboveCtx {
    /// Starting luma 4x4 column of this context buffer (0 for full frame, `mi_col_start` for a
    /// tile).
    pub col0: usize,
    pub level: [Vec<u8>; 3],
    pub dc: [Vec<u8>; 3],
    /// Palette colors of the last decoded block at each luma 4x4 column
    /// (plane 0: Y, plane 1: U).
    pub palette: [Vec<[u16; 8]>; 2],
}

impl AboveCtx {
    pub fn reset(&mut self, col0: usize, above_w: usize, np: usize, screen_content: bool) {
        self.col0 = col0;
        for a in self.level[..np].iter_mut().chain(self.dc[..np].iter_mut()) {
            let pl = a.len().min(above_w);
            a[..pl].fill(0);
            a.resize(above_w, 0);
        }
        if screen_content {
            for p in self.palette.iter_mut() {
                p.resize(above_w, [0; 8]);
            }
        }
    }
}

/// Left context arrays (indexed by 4x4 row within the superblock).
pub struct LeftCtx {
    pub level: [[u8; 32]; 3],
    pub dc: [[u8; 32]; 3],
    pub palette: [[[u16; 8]; 32]; 2],
}

impl Default for LeftCtx {
    fn default() -> Self {
        Self::new()
    }
}

impl LeftCtx {
    pub fn new() -> Self {
        Self {
            level: [[0; 32]; 3],
            dc: [[0; 32]; 3],
            palette: [[[0; 8]; 32]; 2],
        }
    }
    pub fn clear(&mut self) {
        self.level = [[0; 32]; 3];
        self.dc = [[0; 32]; 3];
    }
}

/// Intra prediction setup of the current block for one plane type (luma,
/// chroma), computed once per block by `SbDecoder::setup_intra`: the
/// `IntraPredParams` fields that are the same for all its transform
/// blocks and which edge samples the predictor reads.
#[derive(Clone, Copy, Default)]
pub struct IntraSetup {
    /// `x`, `y`, `have_left` and `have_above` are set per transform block.
    pub p: crate::decoder::av1::dsp::ipred::IntraPredParams,
    /// `f.max_x[pt]` (`p.max_x as usize + 1`).
    pub max_x_p1: usize,
    /// `f.max_y[pt]` (`p.max_y as usize + 1`).
    pub max_y_p1: usize,
    /// `[w, na.min(w << 1)]` indexed by `have_above_right as usize`.
    pub na_lim: [usize; 2],
    /// `[h, nl.min(h << 1)]` indexed by `have_below_left as usize`.
    pub nl_lim: [usize; 2],
    /// Above / left edge samples read when both neighbors are available.
    pub na: usize,
    pub nl: usize,
    /// `sb_y >> sy` when `!above_line[plane].is_empty()`, else `usize::MAX`.
    pub above_line_y: usize,
    /// Plain DC_PRED: an unavailable edge is not read.
    pub dc: bool,
    /// The top-left sample is read.
    pub need_top_left: bool,
}

impl IntraSetup {
    /// Setup with the per-frame fields of plane type `pt` (0 luma, 1
    /// chroma) set; `SbDecoder::setup_intra` writes the per-block ones.
    pub fn new(f: &FrameCtx, pt: usize) -> Self {
        let mut s = Self::default();
        s.p.have_left = true;
        s.p.have_above = true;
        s.p.enable_intra_edge_filter = f.seq.enable_intra_edge_filter;
        s.p.max_x = f.max_x[pt] as i32 - 1;
        s.p.max_y = f.max_y[pt] as i32 - 1;
        s.p.bd = f.bd;
        s.max_x_p1 = f.max_x[pt];
        s.max_y_p1 = f.max_y[pt];
        s.above_line_y = usize::MAX;
        s
    }
}

/// Mutable frame-wide (or tile-local) state written by the superblock decoder.
pub struct FrameState<T: Pixel> {
    /// Reconstructed (and later in-place filtered) pixels.
    pub planes: [PlaneWin<T>; 3],
    /// Unfiltered last sample row of the previous superblock row (used as
    /// the above edge for intra prediction at the top of a superblock).
    pub above_line: [Vec<T>; 3],
    pub mi: MiWin,
    pub above: AboveCtx,
    pub left: LeftCtx,
    /// `cdef_idx` per 64x64 block of the whole frame (or tile).
    pub cdef_idx: Vec<i8>,
    pub cdef_stride: usize,
    pub cdef_row0: usize,
    pub cdef_col0: usize,
    pub lr: [LrPlane; 3],
    /// Chroma from luma AC buffer of one transform block (reused so that
    /// it is not zeroed per block).
    pub cfl_ac: [i16; crate::decoder::av1::dsp::ipred::MAX_CFL_AC],
    /// Intra prediction edge buffer of one transform block (reused so that
    /// it is not zeroed per block).
    pub edges: crate::decoder::av1::dsp::ipred::IntraEdges<T>,
    /// Intra prediction setup of the current block (luma, chroma).
    pub ipred: [IntraSetup; 2],
}

impl<T: Pixel> Default for FrameState<T> {
    fn default() -> Self {
        Self {
            planes: Default::default(),
            above_line: Default::default(),
            mi: MiWin::default(),
            above: AboveCtx::default(),
            left: LeftCtx::new(),
            cdef_idx: Vec::new(),
            cdef_stride: 0,
            cdef_row0: 0,
            cdef_col0: 0,
            lr: Default::default(),
            cfl_ac: [0; crate::decoder::av1::dsp::ipred::MAX_CFL_AC],
            edges: Default::default(),
            ipred: Default::default(),
        }
    }
}

impl<T: Pixel> FrameState<T> {
    #[inline(always)]
    pub fn set_sb_y(&mut self, f: &FrameCtx, sb_y: usize) {
        if self.above_line[0].is_empty() {
            self.ipred[0].above_line_y = usize::MAX;
            self.ipred[1].above_line_y = usize::MAX;
        } else {
            self.ipred[0].above_line_y = sb_y;
            self.ipred[1].above_line_y = sb_y >> f.ss_y;
        }
    }

    #[inline(always)]
    pub fn cdef_idx_at(&self, mi_row: usize, mi_col: usize) -> i8 {
        let r = (mi_row >> 4) - self.cdef_row0;
        let c = (mi_col >> 4) - self.cdef_col0;
        self.cdef_idx[r * self.cdef_stride + c]
    }
    #[inline(always)]
    pub fn set_cdef_idx(&mut self, mi_row: usize, mi_col: usize, v: i8) {
        let r = (mi_row >> 4) - self.cdef_row0;
        let c = (mi_col >> 4) - self.cdef_col0;
        let i = r * self.cdef_stride + c;
        if i < self.cdef_idx.len() {
            self.cdef_idx[i] = v;
        }
    }
}
