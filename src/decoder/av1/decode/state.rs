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

impl<'a> FrameCtx<'a> {
    pub fn new(seq: &'a SequenceHeader, fh: &'a FrameHeader) -> Self {
        let sb128 = seq.use_128x128_superblock;
        let sb4 = if sb128 { 32 } else { 16 };
        let ss_x = seq.color.subsampling_x as usize;
        let ss_y = seq.color.subsampling_y as usize;
        let mut sub_valid = [false; BlockSize::COUNT];
        let mut uv_residual_size = [BlockSize::B4x4; BlockSize::COUNT];
        let mut uv_tx_size = [TxSize::Tx4x4; BlockSize::COUNT];
        for i in 0..BlockSize::COUNT {
            let sub = SUBSAMPLED_SIZE[i][ss_x][ss_y];
            sub_valid[i] = sub.is_some();
            let res_sz = sub.unwrap_or(BlockSize::B4x4);
            uv_residual_size[i] = res_sz;
            let uv_tx = MAX_TX_SIZE_RECT[res_sz as usize];
            let w = TX_WIDTH[uv_tx as usize];
            let h = TX_HEIGHT[uv_tx as usize];
            uv_tx_size[i] = if w == 64 || h == 64 {
                if w == 16 {
                    TxSize::Tx16x32
                } else if h == 16 {
                    TxSize::Tx32x16
                } else {
                    TxSize::Tx32x32
                }
            } else {
                uv_tx
            };
        }
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
            sub_valid,
            uv_residual_size,
            uv_tx_size,
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
        rows: (usize, usize),
        cols: (usize, usize),
    ) -> Self {
        let mut ref_lr_wiener = [[[0; 3]; 2]; 3];
        for p in ref_lr_wiener.iter_mut() {
            for pass in p.iter_mut() {
                for (i, v) in pass.iter_mut().enumerate() {
                    *v = WIENER_TAPS_MID[i] as i32;
                }
            }
        }
        Self {
            msac: SymbolDecoder::new(data, fh.disable_cdf_update),
            cdf: Box::new(CdfContext::new(fh.quant.base_q_idx)),
            mi_row_start: rows.0,
            mi_row_end: rows.1,
            mi_col_start: cols.0,
            mi_col_end: cols.1,
            current_qindex: fh.quant.base_q_idx as i32,
            delta_lf: [0; FRAME_LF_COUNT],
            ref_sgr_xqd: [[SGRPROJ_XQD_MID[0] as i32, SGRPROJ_XQD_MID[1] as i32]; 3],
            ref_lr_wiener,
        }
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
        if fh.lr.frame_restoration_type[plane] == RestorationType::None {
            return Self::default();
        }
        let unit_size = fh.lr.loop_restoration_size[plane];
        let (sx, sy) = if plane == 0 { (0, 0) } else { (ss_x, ss_y) };
        let unit_rows = count_units_in_frame(unit_size, (fh.frame_height + sy) >> sy);
        let unit_cols = count_units_in_frame(unit_size, (fh.upscaled_width + sx) >> sx);
        Self {
            unit_size,
            unit_rows,
            unit_cols,
            units: vec![LrUnit::default(); unit_rows * unit_cols],
        }
    }
}

/// Sgr_Params radius of pass `i` for set `set`.
pub fn sgr_radius(set: usize, pass: usize) -> i32 {
    SGR_PARAMS[set][pass * 2] as i32
}

/// Above context arrays (indexed by 4x4 column of the plane relative to `col0 >> sx`).
pub struct AboveCtx {
    /// Starting luma 4x4 column of this context buffer (0 for full frame, `mi_col_start` for a tile).
    pub col0: usize,
    pub level: [Vec<u8>; 3],
    pub dc: [Vec<u8>; 3],
    /// Palette colors of the last decoded block at each luma 4x4 column
    /// (plane 0: Y, plane 1: U).
    pub palette: [Vec<[u16; 8]>; 2],
}

/// Left context arrays (indexed by 4x4 row within the superblock).
pub struct LeftCtx {
    pub level: [[u8; 32]; 3],
    pub dc: [[u8; 32]; 3],
    pub palette: [[[u16; 8]; 32]; 2],
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
    /// Above / left edge samples read when both neighbors are available.
    pub na: usize,
    pub nl: usize,
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
        s.p.enable_intra_edge_filter = f.seq.enable_intra_edge_filter;
        s.p.max_x = f.max_x[pt] as i32 - 1;
        s.p.max_y = f.max_y[pt] as i32 - 1;
        s.p.bd = f.bd;
        s
    }
}

/// Mutable frame-wide (or tile-local) state written by the superblock decoder.
pub struct FrameState<T: Pixel> {
    /// Reconstructed (and later in-place filtered) pixels.
    pub planes: Vec<PlaneWin<T>>,
    /// Unfiltered last sample row of the previous superblock row (used as
    /// the above edge for intra prediction at the top of a superblock).
    pub above_line: Vec<Vec<T>>,
    /// Plane row index the `above_line` corresponds to (+1 = first row of
    /// the current superblock row).
    pub above_line_y: [usize; 3],
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
    pub cfl_ac: Box<[i16; crate::decoder::av1::dsp::ipred::MAX_CFL_AC]>,
    /// Intra prediction edge buffer of one transform block (reused so that
    /// it is not zeroed per block).
    pub edges: crate::decoder::av1::dsp::ipred::IntraEdges<T>,
    /// Intra prediction setup of the current block (luma, chroma).
    pub ipred: [IntraSetup; 2],
}

impl<T: Pixel> FrameState<T> {
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
