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

//! Superblock decoding: partition / mode info / residual parsing (spec
//! 5.11) interleaved with reconstruction (spec 7.11.2, 7.11.4, 7.11.5,
//! 7.12.3).

use crate::decoder::av1::error::{Error, Result};
use crate::decoder::av1::headers::{RestorationType, TxMode};
use crate::decoder::av1::pixel::{ceil_log2, Pixel};
use crate::decoder::av1::tables::block::*;
use crate::decoder::av1::tables::cdf::CdfContext;
use crate::decoder::av1::types::*;

use super::state::{sgr_radius, FrameCtx, FrameState, TileState};

const MAX_VARTX_DEPTH: usize = 2;
/// Padding (columns) of the coefficient `levels` array rows for the 1D
/// transform classes (the 2D class is unpadded, see `coef_tokens`).
const LEVELS_PAD: usize = 4;
const INTRABC_DELAY_PIXELS: i32 = 256;

#[inline(always)]
pub fn bw4(b: BlockSize) -> usize {
    NUM_4X4_BLOCKS_WIDE[b as usize] as usize
}
#[inline(always)]
pub fn bh4(b: BlockSize) -> usize {
    NUM_4X4_BLOCKS_HIGH[b as usize] as usize
}
#[inline(always)]
pub fn txw(t: TxSize) -> usize {
    TX_WIDTH[t as usize] as usize
}
#[inline(always)]
pub fn txh(t: TxSize) -> usize {
    TX_HEIGHT[t as usize] as usize
}

/// Spec `get_plane_residual_size`.
#[inline(always)]
pub fn plane_residual_size(b: BlockSize, sx: usize, sy: usize) -> BlockSize {
    SUBSAMPLED_SIZE[b as usize][sx][sy].unwrap_or(BlockSize::B4x4)
}

const FIND_TX_SIZE_TABLE: [[TxSize; 5]; 5] = [
    [
        TxSize::Tx4x4,
        TxSize::Tx4x8,
        TxSize::Tx4x16,
        TxSize::Tx4x4,
        TxSize::Tx4x4,
    ],
    [
        TxSize::Tx8x4,
        TxSize::Tx8x8,
        TxSize::Tx8x16,
        TxSize::Tx8x32,
        TxSize::Tx4x4,
    ],
    [
        TxSize::Tx16x4,
        TxSize::Tx16x8,
        TxSize::Tx16x16,
        TxSize::Tx16x32,
        TxSize::Tx16x64,
    ],
    [
        TxSize::Tx4x4,
        TxSize::Tx32x8,
        TxSize::Tx32x16,
        TxSize::Tx32x32,
        TxSize::Tx32x64,
    ],
    [
        TxSize::Tx4x4,
        TxSize::Tx4x4,
        TxSize::Tx64x16,
        TxSize::Tx64x32,
        TxSize::Tx64x64,
    ],
];

/// Spec `find_tx_size`.
#[inline(always)]
pub fn find_tx_size(w: usize, h: usize) -> TxSize {
    let wi = ((w.trailing_zeros() as usize).wrapping_sub(2)).min(4);
    let hi = ((h.trailing_zeros() as usize).wrapping_sub(2)).min(4);
    FIND_TX_SIZE_TABLE[wi][hi]
}

/// Spec `get_tx_class`: 0 = 2D, 1 = horizontal, 2 = vertical.
#[inline(always)]
pub fn tx_class(t: TxType) -> usize {
    use TxType::*;
    match t {
        V_DCT | V_ADST | V_FLIPADST => 2,
        H_DCT | H_ADST | H_FLIPADST => 1,
        _ => 0,
    }
}

const TX_SET_DCTONLY: usize = 0;
const TX_SET_INTRA_1: usize = 1;
const TX_SET_INTRA_2: usize = 2;
const TX_SET_INTER_1: usize = 1;
const TX_SET_INTER_2: usize = 2;
const TX_SET_INTER_3: usize = 3;

/// Per block state (the spec's block level variables).
#[derive(Clone)]
pub struct Block {
    pub mi_row: usize,
    pub mi_col: usize,
    pub bsize: BlockSize,
    pub bw4: usize,
    pub bh4: usize,
    pub has_chroma: bool,
    pub avail_u: bool,
    pub avail_l: bool,
    pub avail_u_chroma: bool,
    pub avail_l_chroma: bool,
    pub above_mi: super::buffers::MiInfo,
    pub left_mi: super::buffers::MiInfo,
    pub skip: bool,
    pub segment_id: usize,
    pub lossless: bool,
    pub is_inter: bool,
    pub use_intrabc: bool,
    pub y_mode: IntraMode,
    pub uv_mode: IntraMode,
    pub angle_delta_y: i32,
    pub angle_delta_uv: i32,
    pub use_filter_intra: bool,
    pub filter_intra_mode: u8,
    pub cfl_alpha_u: i32,
    pub cfl_alpha_v: i32,
    pub palette_size_y: usize,
    pub palette_size_uv: usize,
    pub palette_y: [u16; 8],
    pub palette_u: [u16; 8],
    pub palette_v: [u16; 8],
    pub tx_size: TxSize,
    pub mv: [i32; 2],
    pub max_luma_w: usize,
    pub max_luma_h: usize,
}

impl Default for Block {
    fn default() -> Self {
        Self {
            mi_row: 0,
            mi_col: 0,
            bsize: BlockSize::B4x4,
            bw4: 1,
            bh4: 1,
            has_chroma: false,
            avail_u: false,
            avail_l: false,
            avail_u_chroma: false,
            avail_l_chroma: false,
            above_mi: super::buffers::MiInfo::default(),
            left_mi: super::buffers::MiInfo::default(),
            skip: false,
            segment_id: 0,
            lossless: false,
            is_inter: false,
            use_intrabc: false,
            y_mode: IntraMode::DC_PRED,
            uv_mode: IntraMode::DC_PRED,
            angle_delta_y: 0,
            angle_delta_uv: 0,
            use_filter_intra: false,
            filter_intra_mode: 0,
            cfl_alpha_u: 0,
            cfl_alpha_v: 0,
            palette_size_y: 0,
            palette_size_uv: 0,
            palette_y: [0; 8],
            palette_u: [0; 8],
            palette_v: [0; 8],
            tx_size: TxSize::Tx4x4,
            mv: [0; 2],
            max_luma_w: 0,
            max_luma_h: 0,
        }
    }
}

/// One row of `BlockDecoded` flags stored as a `u64` bitmask (bit `x + 1` is column `x`).
#[derive(Clone, Copy, Default)]
#[repr(transparent)]
pub struct BlockDecodedRow(pub u64);

impl std::ops::Index<usize> for BlockDecodedRow {
    type Output = bool;
    #[inline(always)]
    fn index(&self, idx: usize) -> &bool {
        if ((self.0 >> idx) & 1) != 0 {
            &true
        } else {
            &false
        }
    }
}

/// Scratch buffers reused across blocks.
#[derive(Clone)]
pub struct Scratch {
    pub color_map_y: Box<[[u8; 64]; 64]>,
    pub color_map_uv: Box<[[u8; 64]; 64]>,
    /// Dequantized coefficients for the inverse transform.
    pub coeffs: Box<[i32; 1024]>,
    /// Transform types per 4x4 of the current block (`TxTypes`), indexed
    /// relative to the block's top-left (32x32 4x4 blocks max).
    pub tx_types: Box<[[TxType; 32]; 32]>,
    /// `BlockDecoded[plane][y + 1]` bitmask (bit `x + 1` is column `x`).
    pub block_decoded: [[BlockDecodedRow; 34]; 3],
    /// Padded 2D copy of the coefficient levels for context derivation.
    pub levels: Box<[u8; 2200]>,
    /// Non-zero AC tokens of the current transform block in reverse scan
    /// order, as `level << 16 | pos` (see `coef_tokens`).
    pub nz: Box<[u32; 1024]>,
}

impl Scratch {
    pub fn new() -> Self {
        Self {
            color_map_y: Box::new([[0; 64]; 64]),
            color_map_uv: Box::new([[0; 64]; 64]),
            coeffs: Box::new([0; 1024]),
            tx_types: Box::new([[TxType::DCT_DCT; 32]; 32]),
            block_decoded: [[BlockDecodedRow(0); 34]; 3],
            levels: Box::new([0; 2200]),
            nz: Box::new([0; 1024]),
        }
    }
}

/// Decodes superblocks of one tile.
pub struct SbDecoder<'a, 'b, T: Pixel> {
    pub f: &'a FrameCtx<'a>,
    pub t: &'a mut TileState<'b>,
    pub s: &'a mut FrameState<T>,
    pub sc: &'a mut Scratch,
    /// DSP kernels.
    pub dsp: &'a crate::decoder::av1::dsp::Dsp<T>,
    pub b: Block,
    /// When false, the superblock is parsed but not reconstructed (it is
    /// outside the region that has to be decoded).
    pub recon: bool,
    /// Top row (luma samples) of the current superblock row.
    pub sb_y: usize,
    /// Spec `ReadDeltas`.
    pub read_deltas: bool,
    /// Spec `PlaneTxType` of the last transform block with coefficients.
    pub plane_tx_type: TxType,
}

macro_rules! cdf {
    ($self:ident) => {
        $self.t.cdf
    };
}

/// Constants of one plane type's transform block loop in `residual`.
#[derive(Clone, Copy)]
struct TxLoop {
    /// Intra prediction (not palette) per transform block.
    predict: bool,
    /// Palette prediction per transform block.
    palette: bool,
    /// Chroma from luma (chroma only).
    cfl: bool,
    avail_l: bool,
    avail_u: bool,
    /// Coefficients are coded (`!skip`).
    coefs: bool,
    recon: bool,
}

impl<'a, 'b, T: Pixel> SbDecoder<'a, 'b, T> {
    #[inline(always)]
    fn sym<const N: usize>(&mut self, sel: impl FnOnce(&mut CdfContext) -> &mut [u16; N]) -> usize {
        let cdf = sel(&mut self.t.cdf);
        self.t.msac.symbol_n(cdf)
    }

    #[inline(always)]
    fn bool_cdf(&mut self, sel: impl FnOnce(&mut CdfContext) -> &mut [u16; 3]) -> bool {
        let cdf = sel(&mut self.t.cdf);
        self.t.msac.bool_adapt(cdf)
    }

    #[inline(always)]
    fn lit(&mut self, n: u32) -> u32 {
        self.t.msac.literal(n)
    }

    // ------------------------------------------------------------------
    // Residual
    // ------------------------------------------------------------------

    /// Spec `get_tx_size(plane, txSz)`.
    #[inline(always)]
    fn get_tx_size(&self, plane: usize, tx_sz: TxSize) -> TxSize {
        if plane == 0 {
            tx_sz
        } else {
            self.f.uv_tx_size[self.b.bsize as usize]
        }
    }

    fn residual(&mut self) -> Result<()> {
        let f = self.f;
        let b = &self.b;
        let bw = b.bw4;
        let bh = b.bh4;
        let width_chunks = 1.max(bw >> 4);
        let height_chunks = 1.max(bh >> 4);
        let mi_size_chunk = if (bw | bh) >= 32 { BlockSize::B64x64 } else { b.bsize };
        let y_tx = if b.lossless { TxSize::Tx4x4 } else { b.tx_size };
        let y_step_x = txw(y_tx) >> 2;
        let y_step_y = txh(y_tx) >> 2;
        let y_n4w = bw.min(16);
        let y_n4h = bh.min(16);
        let y_base_x = b.mi_col * MI_SIZE;
        let y_base_y = b.mi_row * MI_SIZE;
        let tree = b.is_inter && !b.lossless;
        let has_chroma = b.has_chroma;
        let cfl = has_chroma && b.uv_mode == IntraMode::UV_CFL_PRED;
        // Block constants of the transform block loops, kept in locals
        // (the callees take `&mut self`, so `self.b` fields would be
        // reloaded after every call).
        let pred = !b.is_inter && self.recon;
        let lc = TxLoop {
            predict: pred && b.palette_size_y == 0,
            palette: pred && b.palette_size_y > 0,
            cfl: false,
            avail_l: b.avail_l,
            avail_u: b.avail_u,
            coefs: !b.skip,
            recon: self.recon,
        };
        let cc = TxLoop {
            predict: pred && b.palette_size_uv == 0,
            palette: pred && b.palette_size_uv > 0,
            cfl,
            avail_l: b.avail_l_chroma,
            avail_u: b.avail_u_chroma,
            ..lc
        };

        let (sx, sy) = (f.ss_x, f.ss_y);
        let (uv_tx, uv_step_x, uv_step_y, uv_n4w, uv_n4h, uv_base_x, uv_base_y) = if has_chroma {
            let tx = if b.lossless { TxSize::Tx4x4 } else { f.uv_tx_size[b.bsize as usize] };
            let plane_sz = f.uv_residual_size[mi_size_chunk as usize];
            (
                tx,
                txw(tx) >> 2,
                txh(tx) >> 2,
                bw4(plane_sz),
                bh4(plane_sz),
                (b.mi_col >> sx) * MI_SIZE,
                (b.mi_row >> sy) * MI_SIZE,
            )
        } else {
            (TxSize::Tx4x4, 1, 1, 0, 0, 0, 0)
        };

        for chunk_y in 0..height_chunks {
            for chunk_x in 0..width_chunks {
                let ox0 = chunk_x << 4;
                let oy0 = chunk_y << 4;
                if tree {
                    self.transform_tree(
                        y_base_x + ox0 * MI_SIZE,
                        y_base_y + oy0 * MI_SIZE,
                        y_n4w * 4,
                        y_n4h * 4,
                    )?;
                } else {
                    let mut y = 0;
                    while y < y_n4h {
                        let mut x = 0;
                        while x < y_n4w {
                            self.transform_block::<false>(
                                &lc,
                                0,
                                y_base_x,
                                y_base_y,
                                y_tx,
                                x + ox0,
                                y + oy0,
                            )?;
                            x += y_step_x;
                        }
                        y += y_step_y;
                    }
                    if cfl {
                        let sw = y_step_x * 4;
                        let sh = y_step_y * 4;
                        self.b.max_luma_w = y_base_x
                            + ((f.max_x[0] - y_base_x + sw - 1) & !(sw - 1)).min(y_n4w * 4);
                        self.b.max_luma_h = y_base_y
                            + ((f.max_y[0] - y_base_y + sh - 1) & !(sh - 1)).min(y_n4h * 4);
                    }
                }
                if has_chroma {
                    let ox = ox0 >> sx;
                    let oy = oy0 >> sy;
                    for plane in 1..3 {
                        let mut y = 0;
                        while y < uv_n4h {
                            let mut x = 0;
                            while x < uv_n4w {
                                self.transform_block::<true>(
                                    &cc,
                                    plane,
                                    uv_base_x,
                                    uv_base_y,
                                    uv_tx,
                                    x + ox,
                                    y + oy,
                                )?;
                                x += uv_step_x;
                            }
                            y += uv_step_y;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn transform_tree(&mut self, start_x: usize, start_y: usize, w: usize, h: usize) -> Result<()> {
        let f = self.f;
        if start_x >= f.max_x[0] || start_y >= f.max_y[0] {
            return Ok(());
        }
        let row = start_y >> 2;
        let col = start_x >> 2;
        let luma_tx = TxSize::from_u8(self.s.mi.get(row, col).inter_tx_size);
        if w <= txw(luma_tx) && h <= txh(luma_tx) {
            let tx_sz = find_tx_size(w, h);
            // Inter (intra block copy) blocks: no prediction per
            // transform block.
            let c = TxLoop {
                predict: false,
                palette: false,
                cfl: false,
                avail_l: false,
                avail_u: false,
                coefs: !self.b.skip,
                recon: self.recon,
            };
            self.transform_block::<false>(&c, 0, start_x, start_y, tx_sz, 0, 0)
        } else if w > h {
            self.transform_tree(start_x, start_y, w / 2, h)?;
            self.transform_tree(start_x + w / 2, start_y, w / 2, h)
        } else if w < h {
            self.transform_tree(start_x, start_y, w, h / 2)?;
            self.transform_tree(start_x, start_y + h / 2, w, h / 2)
        } else {
            self.transform_tree(start_x, start_y, w / 2, h / 2)?;
            self.transform_tree(start_x + w / 2, start_y, w / 2, h / 2)?;
            self.transform_tree(start_x, start_y + h / 2, w / 2, h / 2)?;
            self.transform_tree(start_x + w / 2, start_y + h / 2, w / 2, h / 2)
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn transform_block<const CHROMA: bool>(
        &mut self,
        c: &TxLoop,
        plane: usize,
        base_x: usize,
        base_y: usize,
        tx_sz: TxSize,
        x: usize,
        y: usize,
    ) -> Result<()> {
        let f = self.f;
        let pt = CHROMA as usize;
        let start_x = base_x + 4 * x;
        let start_y = base_y + 4 * y;
        if start_x >= f.max_x[pt] || start_y >= f.max_y[pt] {
            return Ok(());
        }
        let sub_x = (start_x >> 2) & f.sb_mask_x[pt];
        let sub_y = (start_y >> 2) & f.sb_mask_y[pt];
        let bx = sub_x + 1;
        let by = sub_y + 1;
        let step_x = txw(tx_sz) >> 2;
        let step_y = txh(tx_sz) >> 2;
        if c.predict {
            let have_left = c.avail_l || x > 0;
            let have_above = c.avail_u || y > 0;
            let bd =
                if !CHROMA { &self.sc.block_decoded[0] } else { &self.sc.block_decoded[plane] };
            let have_above_right = ((bd[sub_y].0 >> (bx + step_x)) & 1) != 0;
            let have_below_left = ((bd[by + step_y].0 >> sub_x) & 1) != 0;
            self.predict_intra(
                plane,
                start_x,
                start_y,
                have_left,
                have_above,
                have_above_right,
                have_below_left,
            );
            if CHROMA && c.cfl {
                self.predict_cfl(plane, start_x, start_y, tx_sz);
            }
        } else if c.palette {
            self.predict_palette(plane, start_x, start_y, x, y, tx_sz);
        }
        if c.coefs {
            let eob = self.coeffs(plane, start_x, start_y, tx_sz)?;
            if eob > 0 {
                if c.recon {
                    self.reconstruct(plane, start_x, start_y, tx_sz, eob);
                } else {
                    self.clear_coeffs(tx_sz);
                }
            }
        }
        // BlockDecoded.
        let mask = ((1u64 << step_x) - 1) << bx;
        let bd = if !CHROMA {
            &mut self.sc.block_decoded[0]
        } else {
            &mut self.sc.block_decoded[plane]
        };
        match step_y {
            1 => bd[by].0 |= mask,
            2 => {
                bd[by].0 |= mask;
                bd[by + 1].0 |= mask;
            }
            4 => {
                for r in &mut bd[by..by + 4] {
                    r.0 |= mask;
                }
            }
            8 => {
                for r in &mut bd[by..by + 8] {
                    r.0 |= mask;
                }
            }
            _ => {
                for r in &mut bd[by..by + 16] {
                    r.0 |= mask;
                }
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Transform type
    // ------------------------------------------------------------------

    fn get_tx_set(&self, tx_sz: TxSize) -> usize {
        let sqr = TX_SIZE_SQR[tx_sz as usize];
        let sqr_up = TX_SIZE_SQR_UP[tx_sz as usize];
        if sqr_up > TxSize::Tx32x32 {
            return TX_SET_DCTONLY;
        }
        if self.b.is_inter {
            if self.f.fh.reduced_tx_set || sqr_up == TxSize::Tx32x32 {
                TX_SET_INTER_3
            } else if sqr == TxSize::Tx16x16 {
                TX_SET_INTER_2
            } else {
                TX_SET_INTER_1
            }
        } else if sqr_up == TxSize::Tx32x32 {
            TX_SET_DCTONLY
        } else if self.f.fh.reduced_tx_set || sqr == TxSize::Tx16x16 {
            TX_SET_INTRA_2
        } else {
            TX_SET_INTRA_1
        }
    }

    fn is_tx_type_in_set(&self, set: usize, t: TxType) -> bool {
        if self.b.is_inter {
            TX_TYPE_IN_SET_INTER[set][t as usize]
        } else {
            TX_TYPE_IN_SET_INTRA[set][t as usize]
        }
    }

    /// Reads `transform_type` for a luma transform block at (x4, y4)
    /// (absolute 4x4 units), stores it in the block's TxTypes when needed
    /// for inter chroma derivation, and returns it.
    fn transform_type(&mut self, x4: usize, y4: usize, tx_sz: TxSize) -> TxType {
        let set = self.get_tx_set(tx_sz);
        let fh = self.f.fh;
        let qidx = if fh.seg.enabled {
            fh.get_qindex(true, self.b.segment_id, self.t.current_qindex)
        } else {
            fh.quant.base_q_idx as i32
        };
        let tx_type = if set > 0 && qidx > 0 {
            let sqr = TX_SIZE_SQR[tx_sz as usize] as usize;
            let msac = &mut self.t.msac;
            let cdf = &mut *self.t.cdf;
            if self.b.is_inter {
                match set {
                    TX_SET_INTER_1 => {
                        let v = msac.symbol_n(&mut cdf.inter_tx_type_set1[sqr]);
                        TX_TYPE_INTER_INV_SET1[v]
                    }
                    TX_SET_INTER_2 => {
                        let v = msac.symbol_n(&mut cdf.inter_tx_type_set2);
                        TX_TYPE_INTER_INV_SET2[v]
                    }
                    _ => {
                        let v = msac.symbol_n(&mut cdf.inter_tx_type_set3[sqr]);
                        TX_TYPE_INTER_INV_SET3[v]
                    }
                }
            } else {
                let intra_dir = if self.b.use_filter_intra {
                    FILTER_INTRA_MODE_TO_INTRA_DIR[self.b.filter_intra_mode as usize] as usize
                } else {
                    self.b.y_mode as usize
                };
                if set == TX_SET_INTRA_1 {
                    let v = msac.symbol_n(&mut cdf.intra_tx_type_set1[sqr][intra_dir]);
                    TX_TYPE_INTRA_INV_SET1[v]
                } else {
                    let v = msac.symbol_n(&mut cdf.intra_tx_type_set2[sqr][intra_dir]);
                    TX_TYPE_INTRA_INV_SET2[v]
                }
            }
        } else {
            TxType::DCT_DCT
        };
        if self.b.is_inter {
            self.set_tx_types(x4, y4, tx_sz, tx_type);
        }
        tx_type
    }

    #[inline(never)]
    fn set_tx_types(&mut self, x4: usize, y4: usize, tx_sz: TxSize, t: TxType) {
        let bx = x4 - self.b.mi_col;
        let by = y4 - self.b.mi_row;
        let bx_end = (bx + (txw(tx_sz) >> 2)).min(32);
        let by_end = (by + (txh(tx_sz) >> 2)).min(32);
        if bx >= bx_end || by >= by_end {
            return;
        }
        for row in &mut self.sc.tx_types[by..by_end] {
            row[bx..bx_end].fill(t);
        }
    }

    /// Spec `compute_tx_type(plane, txSz, blockX, blockY)`.
    fn compute_tx_type(
        &self,
        plane: usize,
        tx_sz: TxSize,
        block_x: usize,
        block_y: usize,
    ) -> TxType {
        let sqr_up = TX_SIZE_SQR_UP[tx_sz as usize];
        if self.b.lossless || sqr_up > TxSize::Tx32x32 {
            return TxType::DCT_DCT;
        }
        let set = self.get_tx_set(tx_sz);
        if plane == 0 {
            return self.sc.tx_types[block_y - self.b.mi_row][block_x - self.b.mi_col];
        }
        let t = if self.b.is_inter {
            let x4 = self.b.mi_col.max(block_x << self.f.ss_x);
            let y4 = self.b.mi_row.max(block_y << self.f.ss_y);
            self.sc.tx_types[y4 - self.b.mi_row][x4 - self.b.mi_col]
        } else {
            MODE_TO_TXFM[self.b.uv_mode as usize]
        };
        if !self.is_tx_type_in_set(set, t) {
            TxType::DCT_DCT
        } else {
            t
        }
    }

    // ------------------------------------------------------------------
    // Coefficients
    // ------------------------------------------------------------------

    fn coeffs(
        &mut self,
        plane: usize,
        start_x: usize,
        start_y: usize,
        tx_sz: TxSize,
    ) -> Result<usize> {
        let f = self.f;
        let x4 = start_x >> 2;
        let y4 = start_y >> 2;
        let w4 = txw(tx_sz) >> 2;
        let h4 = txh(tx_sz) >> 2;
        let ts = tx_sz as usize;
        let tx_sz_ctx = (TX_SIZE_SQR[ts] as usize + TX_SIZE_SQR_UP[ts] as usize + 1) >> 1;
        let (sx, sy) = f.sub(plane);
        let max_x4 = f.mi_cols >> sx;
        let max_y4 = f.mi_rows >> sy;
        let left_mask = (f.sb4 >> sy) - 1;

        // Above / left context ranges of this transform block (the parts
        // inside the frame). A transform block never crosses the
        // superblock, so the left range does not wrap.
        let kx = w4.min(max_x4.saturating_sub(x4));
        let ky = h4.min(max_y4.saturating_sub(y4));
        let ly = y4 & left_mask;
        debug_assert!(ly + ky <= left_mask + 1);
        let ax0 = x4 - (self.s.above.col0 >> sx);
        let ax1 = ax0 + kx;
        let (ly0, ly1) = (ly, (ly + ky).min(self.s.left.level[plane].len()));

        // all_zero context
        let all_zero_ctx = if plane == 0 {
            if self.b.bw4 == w4 && self.b.bh4 == h4 {
                0
            } else {
                let top = or_slice(&self.s.above.level[0][ax0..ax1]) as usize;
                let left = or_slice(&self.s.left.level[0][ly0..ly1]) as usize;
                if (top | left) == 0 {
                    1
                } else if top == 0 || left == 0 {
                    2 + ((top | left) > 3) as usize
                } else if (top | left) <= 3 {
                    4
                } else if top.min(left) <= 3 {
                    5
                } else {
                    6
                }
            }
        } else {
            let bsize = plane_residual_size(self.b.bsize, sx, sy);
            let bw = BLOCK_WIDTH[bsize as usize] as usize;
            let bh = BLOCK_HEIGHT[bsize as usize] as usize;
            let above = any_nz2(
                &self.s.above.level[plane][ax0..ax1],
                &self.s.above.dc[plane][ax0..ax1],
            );
            let left = any_nz2(
                &self.s.left.level[plane][ly0..ly1],
                &self.s.left.dc[plane][ly0..ly1],
            );
            let mut ctx = above as usize + left as usize + 7;
            if bw * bh > (w4 * h4) << 4 {
                ctx += 3;
            }
            ctx
        };

        let all_zero = self
            .t
            .msac
            .bool_adapt(&mut self.t.cdf.txb_skip[tx_sz_ctx][all_zero_ctx]);
        if all_zero {
            if plane == 0 && self.b.is_inter {
                self.set_tx_types(x4, y4, tx_sz, TxType::DCT_DCT);
            }
            fill_ctx2(
                &mut self.s.above.level[plane],
                &mut self.s.above.dc[plane],
                ax0,
                w4,
                0,
                0,
            );
            fill_ctx2(
                &mut self.s.left.level[plane],
                &mut self.s.left.dc[plane],
                ly,
                h4,
                0,
                0,
            );
            return Ok(0);
        }
        self.coeffs_nz(
            plane, x4, y4, w4, h4, tx_sz, ts, tx_sz_ctx, ax0, ax1, ly, ly0, ly1,
        )
    }

    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn coeffs_nz(
        &mut self,
        plane: usize,
        x4: usize,
        y4: usize,
        w4: usize,
        h4: usize,
        tx_sz: TxSize,
        ts: usize,
        tx_sz_ctx: usize,
        ax0: usize,
        ax1: usize,
        ly: usize,
        ly0: usize,
        ly1: usize,
    ) -> Result<usize> {
        let f = self.f;
        let ptype = (plane > 0) as usize;
        let tx_type = if plane == 0 {
            let t = self.transform_type(x4, y4, tx_sz);
            if self.b.lossless {
                TxType::DCT_DCT
            } else {
                t
            }
        } else {
            self.compute_tx_type(plane, tx_sz, x4, y4)
        };
        self.plane_tx_type = tx_type;
        let tx_cls = tx_class(tx_type);
        let eob_multisize =
            (TX_WIDTH_LOG2[ts] as usize).min(5) + (TX_HEIGHT_LOG2[ts] as usize).min(5) - 4;
        let eob_ctx = (tx_cls != 0) as usize;
        let msac = &mut self.t.msac;
        let cdf = &mut *self.t.cdf;
        let eob_bin = match eob_multisize {
            0 => msac.symbol_n(&mut cdf.eob_pt_16[ptype][eob_ctx]),
            1 => msac.symbol_n(&mut cdf.eob_pt_32[ptype][eob_ctx]),
            2 => msac.symbol_n(&mut cdf.eob_pt_64[ptype][eob_ctx]),
            3 => msac.symbol_n(&mut cdf.eob_pt_128[ptype][eob_ctx]),
            4 => msac.symbol_n(&mut cdf.eob_pt_256[ptype][eob_ctx]),
            5 => msac.symbol_n(&mut cdf.eob_pt_512[ptype]),
            _ => msac.symbol_n(&mut cdf.eob_pt_1024[ptype]),
        };
        let mut last = eob_bin;
        if eob_bin >= 2 {
            let eb = eob_bin - 2;
            let hi = msac.bool_adapt(&mut cdf.eob_extra[tx_sz_ctx][ptype][eb]) as usize;
            last = ((hi | 2) << eb) | msac.literal(eb as u32) as usize;
        }
        let eob = last + 1;

        // Levels are kept in a padded 2D array for context derivation;
        // non-zero entries are cleared while walking `nz` below.
        // `lbwl` maps a position to its levels index (`pos + (pos >> lbwl
        // << 2)`): the 2D class uses an unpadded array (see coef_tokens).
        let (dc_tok, num_ac, lbwl) = if eob == 1 {
            let mut tok = msac.symbol_n(&mut cdf.coeff_base_eob[tx_sz_ctx][ptype][0]) as u32 + 1;
            if tok == 3 {
                tok = msac.hi_tok(&mut cdf.coeff_br[tx_sz_ctx.min(3)][ptype][0]);
            }
            (tok, 0, 0)
        } else {
            let scan = coef_scan(tx_sz, tx_cls, tx_type);
            let Some(scan) = scan.get(..eob) else {
                return Err(Error::Invalid("eob"));
            };
            let adj = ADJUSTED_TX_SIZE[ts] as usize;
            let bwl = (TX_WIDTH_LOG2[adj] as usize).clamp(2, 5);
            let height = TX_HEIGHT[adj] as usize;
            let (dc_tok, num_ac) = match (tx_cls, bwl) {
                (0, 2) => self.coef_tokens_d::<0, 2>(scan, tx_sz_ctx, ptype, height),
                (0, 3) => self.coef_tokens_d::<0, 3>(scan, tx_sz_ctx, ptype, height),
                (0, 4) => self.coef_tokens_d::<0, 4>(scan, tx_sz_ctx, ptype, height),
                (0, _) => self.coef_tokens_d::<0, 5>(scan, tx_sz_ctx, ptype, height),
                (1, 2) => self.coef_tokens_d::<1, 2>(scan, tx_sz_ctx, ptype, height),
                (1, 3) => self.coef_tokens_d::<1, 3>(scan, tx_sz_ctx, ptype, height),
                (1, _) => self.coef_tokens_d::<1, 4>(scan, tx_sz_ctx, ptype, height),
                (_, 2) => self.coef_tokens_d::<2, 2>(scan, tx_sz_ctx, ptype, height),
                (_, 3) => self.coef_tokens_d::<2, 3>(scan, tx_sz_ctx, ptype, height),
                (_, _) => self.coef_tokens_d::<2, 4>(scan, tx_sz_ctx, ptype, height),
            };
            (dc_tok, num_ac, if tx_cls == 0 { 10 } else { bwl })
        };

        // Signs, golomb and dequantization (spec 7.12.3 step 1), in
        // scan order over the non-zero coefficients.
        let (dc_q, ac_q) = self.get_quant(plane);
        let fh = f.fh;
        let qm_level = if fh.quant.using_qmatrix && !self.b.lossless {
            fh.seg_qm_level[plane][self.b.segment_id] as usize
        } else {
            15
        };
        let use_qm =
            fh.quant.using_qmatrix && (tx_type as usize) < TxType::IDTX as usize && qm_level < 15;
        let qm: &[u8] = if use_qm {
            crate::decoder::av1::tables::qm::quantizer_matrix(qm_level, ptype, tx_sz)
        } else {
            &[]
        };
        let dq_shift = match tx_sz {
            TxSize::Tx32x32
            | TxSize::Tx16x32
            | TxSize::Tx32x16
            | TxSize::Tx16x64
            | TxSize::Tx64x16 => 1,
            TxSize::Tx64x64 | TxSize::Tx32x64 | TxSize::Tx64x32 => 2,
            _ => 0,
        };
        // Largest magnitude of a positive coefficient; negative ones
        // may be one larger.
        let cf_max = (1u64 << (7 + f.bd)) - 1;
        let dequant_qm = |q: u32, qv: i32, pos: usize, sign: bool| -> i32 {
            let qv = qv as u64;
            let q2 = if use_qm { (qv * qm[pos] as u64 + 16) >> 5 } else { qv };
            let dq = (((q as u64 * q2) & 0xFFFFFF) >> dq_shift).min(cf_max + sign as u64) as i32;
            if sign {
                -dq
            } else {
                dq
            }
        };
        let mut cul_level = 0u32;
        let mut dc_category = 0u8;
        if dc_tok != 0 {
            let dc_sign = sign_sum(&self.s.above.dc[plane][ax0..ax1])
                + sign_sum(&self.s.left.dc[plane][ly0..ly1]);
            let ctx = (dc_sign < 0) as usize + 2 * (dc_sign > 0) as usize;
            let sign = self.t.msac.bool_adapt(&mut self.t.cdf.dc_sign[ptype][ctx]);
            let mut q = dc_tok;
            let dq = if !use_qm && q < 15 {
                let d = ((q * dc_q as u32) >> dq_shift) as i32;
                if sign {
                    -d
                } else {
                    d
                }
            } else {
                if q == 15 {
                    q = self
                        .t
                        .msac
                        .golomb_coef()
                        .ok_or(Error::Invalid("golomb"))?
                        .wrapping_add(14)
                        & 0xFFFFF;
                }
                dequant_qm(q, dc_q, 0, sign)
            };
            dc_category = if sign { 1 } else { 2 };
            cul_level += q;
            self.sc.coeffs[0] = dq;
        }
        let n_ac = num_ac.min(1024);
        if n_ac != 0 {
            let mut m = self.t.msac.clone();
            let msac = &mut m;
            let cf = &mut *self.sc.coeffs;
            let levels = &mut *self.sc.levels;
            let nz = &self.sc.nz[..n_ac];
            if !use_qm {
                let ac_q = ac_q as u32;
                for &e in nz.iter().rev() {
                    let rc = (e & 0x3FF) as usize;
                    levels[rc + ((rc >> lbwl) << 2)] = 0;
                    let sign = msac.bool();
                    let mut q = e >> 16;
                    let dq = if q < 15 {
                        ((q * ac_q) >> dq_shift) as i32
                    } else {
                        q = msac
                            .golomb_coef()
                            .ok_or(Error::Invalid("golomb"))?
                            .wrapping_add(14)
                            & 0xFFFFF;
                        (((q as u64 * ac_q as u64) & 0xFFFFFF) >> dq_shift)
                            .min(cf_max + sign as u64) as i32
                    };
                    cul_level += q;
                    cf[rc] = if sign { -dq } else { dq };
                }
            } else {
                for &e in nz.iter().rev() {
                    let rc = (e & 0x3FF) as usize;
                    levels[rc + ((rc >> lbwl) << 2)] = 0;
                    let sign = msac.bool();
                    let mut q = e >> 16;
                    if q == 15 {
                        q = msac
                            .golomb_coef()
                            .ok_or(Error::Invalid("golomb"))?
                            .wrapping_add(14)
                            & 0xFFFFF;
                    }
                    cul_level += q;
                    cf[rc] = dequant_qm(q, ac_q, rc, sign);
                }
            }
            self.t.msac = m;
        }
        cul_level = cul_level.min(63);
        fill_ctx2(
            &mut self.s.above.level[plane],
            &mut self.s.above.dc[plane],
            ax0,
            w4,
            cul_level as u8,
            dc_category,
        );
        fill_ctx2(
            &mut self.s.left.level[plane],
            &mut self.s.left.dc[plane],
            ly,
            h4,
            cul_level as u8,
            dc_category,
        );
        Ok(eob)
    }

    /// Reads the coefficient tokens (levels up to 15) of a transform block
    /// in reverse scan order (`coeff_base_eob`, `coeff_base`, `coeff_br`)
    /// for `eob > 1`.
    ///
    /// Like dav1d, the `levels` array holds per position
    /// `min(level, 3) << 6 | level`, so that one sum over the neighbours
    /// gives both the `coeff_base` and the `coeff_br` magnitudes. Non-zero
    /// AC tokens are appended to `nz` as `level << 16 | pos`, in reverse
    /// scan order (so the sign pass walks `nz` backwards without a
    /// dependent load chain). `coeffs` is not touched.
    ///
    /// The 1D classes pad each row of `levels` with `LEVELS_PAD` zero
    /// columns. The 2D class uses no padding (stride = width), like
    /// dav1d: a neighbour read past the end of a row lands at the start
    /// of a later row, i.e. on a smaller anti-diagonal, which the 2D
    /// (diagonal) scan has not decoded yet, so it still reads 0. Its
    /// `scan` entries carry the context offset of the position in bits
    /// 10.. (see `coef_scan`).
    ///
    /// Returns the DC token and the number of non-zero AC tokens in `nz`.
    /// `coef_tokens` specialized on the CDF adaptation flag (it is
    /// constant per tile and almost always set), which keeps the flag
    /// test out of the per-coefficient symbol reads.
    #[inline(always)]
    fn coef_tokens_d<const CLS: usize, const BWL: usize>(
        &mut self,
        scan: &[u16],
        tx_sz_ctx: usize,
        ptype: usize,
        height: usize,
    ) -> (u32, usize) {
        if self.t.msac.allow_update() {
            self.coef_tokens::<CLS, BWL, true>(scan, tx_sz_ctx, ptype, height)
        } else {
            self.coef_tokens::<CLS, BWL, false>(scan, tx_sz_ctx, ptype, height)
        }
    }

    #[inline(never)]
    fn coef_tokens<const CLS: usize, const BWL: usize, const UPD: bool>(
        &mut self,
        scan: &[u16],
        tx_sz_ctx: usize,
        ptype: usize,
        height: usize,
    ) -> (u32, usize) {
        const { assert!(BWL >= 2 && BWL <= 5) };
        let ls = (1usize << BWL) + if CLS == 0 { 0 } else { LEVELS_PAD };
        // Work on a local copy of the decoder state so that it stays in
        // registers.
        let mut m = self.t.msac.clone();
        let msac = &mut m;
        let cdf = &mut *self.t.cdf;
        let eob_cdf = &mut cdf.coeff_base_eob[tx_sz_ctx][ptype];
        let lo_cdf = &mut cdf.coeff_base[tx_sz_ctx][ptype];
        let hi_cdf = &mut cdf.coeff_br[tx_sz_ctx.min(3)][ptype];
        let levels = &mut *self.sc.levels;
        let nz = &mut *self.sc.nz;

        // Last coefficient (coeff_base_eob, with last = eob - 1 >= 1).
        // All levels are still zero, so the coeff_br magnitude is 0.
        let last = scan.len() - 1;
        let area = height << BWL;
        let pos = (scan[last] & 0x3FF) as usize;
        let ctx = if last <= area / 8 {
            1
        } else if last <= area / 4 {
            2
        } else {
            3
        };
        let mut tok = msac.symbol_n(&mut eob_cdf[ctx]) as u32 + 1;
        let lb = if tok == 3 {
            tok = msac.hi_tok_c::<UPD>(&mut hi_cdf[br_pos_ctx::<CLS, BWL>(pos)]);
            (tok as u8) | (3 << 6)
        } else {
            (tok as u8) * 0x41
        };
        levels[level_idx::<CLS, BWL>(pos)] = lb;
        nz[0] = (tok << 16) | pos as u32;
        let mut k = 1usize;

        for &p in scan[1..last].iter().rev() {
            let pos = (p & 0x3FF) as usize;
            let idx = level_idx::<CLS, BWL>(pos);
            let ctx = lo_ctx::<CLS, BWL>(levels, idx, ls, p);
            let tok = msac.symbol5_c::<UPD>(&mut lo_cdf[ctx]);
            // k < scan.len() <= 1024.
            let slot = &mut nz[k & 1023];
            if tok == 3 {
                let hi = hi_mag::<CLS>(levels, idx, ls);
                let tok =
                    msac.hi_tok_c::<UPD>(&mut hi_cdf[br_pos_ctx::<CLS, BWL>(pos) + hi_ctx(hi)]);
                levels[idx] = (tok as u8) | (3 << 6);
                *slot = (tok << 16) | pos as u32;
                k += 1;
            } else {
                // Branch free (zero / non-zero is unpredictable), like
                // dav1d: zero tokens store 0 (levels are zero there
                // already) and are dropped from `nz` by not advancing k.
                levels[idx] = (tok * 0x41) as u8;
                *slot = (tok << 16) | pos as u32;
                k += (tok != 0) as usize;
            }
        }

        // DC.
        let mut tok = if CLS == 0 {
            msac.symbol5_c::<UPD>(&mut lo_cdf[0])
        } else {
            let ctx = lo_ctx::<CLS, BWL>(levels, 0, ls, 0);
            msac.symbol5_c::<UPD>(&mut lo_cdf[ctx])
        };
        if tok == 3 {
            let hi = hi_mag::<CLS>(levels, 0, ls);
            tok = msac.hi_tok_c::<UPD>(&mut hi_cdf[hi_ctx(hi)]);
        }
        self.t.msac = m;
        (tok, k)
    }

    /// Returns (dc quantizer, ac quantizer) for `plane` (spec 7.12.2).
    fn get_quant(&self, plane: usize) -> (i32, i32) {
        use crate::decoder::av1::tables::quant::{ac_q, dc_q};
        let fh = self.f.fh;
        let bd = self.f.bd;
        let qindex = fh.get_qindex(false, self.b.segment_id, self.t.current_qindex);
        let q = &fh.quant;
        match plane {
            0 => (dc_q(qindex + q.delta_q_y_dc, bd), ac_q(qindex, bd)),
            1 => (
                dc_q(qindex + q.delta_q_u_dc, bd),
                ac_q(qindex + q.delta_q_u_ac, bd),
            ),
            _ => (
                dc_q(qindex + q.delta_q_v_dc, bd),
                ac_q(qindex + q.delta_q_v_ac, bd),
            ),
        }
    }
}

/// Spec `get_scan(txSz)` given PlaneTxType.
#[inline]
pub fn get_scan(tx_sz: TxSize, tx_type: TxType) -> &'static [u16] {
    use crate::decoder::av1::tables::scan::*;
    use TxSize::*;
    if tx_sz == Tx16x64 {
        return &DEFAULT_SCAN_16X32;
    }
    if tx_sz == Tx64x16 {
        return &DEFAULT_SCAN_32X16;
    }
    if TX_SIZE_SQR_UP[tx_sz as usize] == Tx64x64 {
        return &DEFAULT_SCAN_32X32;
    }
    let default = || -> &'static [u16] {
        match tx_sz {
            Tx4x4 => &DEFAULT_SCAN_4X4,
            Tx4x8 => &DEFAULT_SCAN_4X8,
            Tx8x4 => &DEFAULT_SCAN_8X4,
            Tx8x8 => &DEFAULT_SCAN_8X8,
            Tx8x16 => &DEFAULT_SCAN_8X16,
            Tx16x8 => &DEFAULT_SCAN_16X8,
            Tx16x16 => &DEFAULT_SCAN_16X16,
            Tx16x32 => &DEFAULT_SCAN_16X32,
            Tx32x16 => &DEFAULT_SCAN_32X16,
            Tx4x16 => &DEFAULT_SCAN_4X16,
            Tx16x4 => &DEFAULT_SCAN_16X4,
            Tx8x32 => &DEFAULT_SCAN_8X32,
            Tx32x8 => &DEFAULT_SCAN_32X8,
            _ => &DEFAULT_SCAN_32X32,
        }
    };
    if tx_type == TxType::IDTX {
        return default();
    }
    use TxType::*;
    let prefer_row = matches!(tx_type, V_DCT | V_ADST | V_FLIPADST);
    let prefer_col = matches!(tx_type, H_DCT | H_ADST | H_FLIPADST);
    if prefer_row {
        match tx_sz {
            Tx4x4 => &MROW_SCAN_4X4,
            Tx4x8 => &MROW_SCAN_4X8,
            Tx8x4 => &MROW_SCAN_8X4,
            Tx8x8 => &MROW_SCAN_8X8,
            Tx8x16 => &MROW_SCAN_8X16,
            Tx16x8 => &MROW_SCAN_16X8,
            Tx16x16 => &MROW_SCAN_16X16,
            Tx4x16 => &MROW_SCAN_4X16,
            _ => &MROW_SCAN_16X4,
        }
    } else if prefer_col {
        match tx_sz {
            Tx4x4 => &MCOL_SCAN_4X4,
            Tx4x8 => &MCOL_SCAN_4X8,
            Tx8x4 => &MCOL_SCAN_8X4,
            Tx8x8 => &MCOL_SCAN_8X8,
            Tx8x16 => &MCOL_SCAN_8X16,
            Tx16x8 => &MCOL_SCAN_16X8,
            Tx16x16 => &MCOL_SCAN_16X16,
            Tx4x16 => &MCOL_SCAN_4X16,
            _ => &MCOL_SCAN_16X4,
        }
    } else {
        default()
    }
}

pub(super) const _UNUSED: i32 = INTRABC_DELAY_PIXELS;

#[inline(always)]
fn or_slice(s: &[u8]) -> u8 {
    match s {
        &[a] => a,
        &[a, b] => a | b,
        &[a, b, c, d] => {
            let v = u32::from_ne_bytes([a, b, c, d]);
            let v = v | (v >> 16);
            (v | (v >> 8)) as u8
        }
        &[a0, a1, a2, a3, a4, a5, a6, a7] => {
            let v = u64::from_ne_bytes([a0, a1, a2, a3, a4, a5, a6, a7]);
            let v = v | (v >> 32);
            let v = v | (v >> 16);
            (v | (v >> 8)) as u8
        }
        &[a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7] => {
            let v0 = u64::from_ne_bytes([a0, a1, a2, a3, a4, a5, a6, a7]);
            let v1 = u64::from_ne_bytes([b0, b1, b2, b3, b4, b5, b6, b7]);
            let v = v0 | v1;
            let v = v | (v >> 32);
            let v = v | (v >> 16);
            (v | (v >> 8)) as u8
        }
        _ => or_slice_slow(s),
    }
}

#[cold]
#[inline(never)]
fn or_slice_slow(s: &[u8]) -> u8 {
    s.iter().fold(0u8, |m, &v| m | v)
}

#[inline(always)]
fn any_nz2(a: &[u8], b: &[u8]) -> bool {
    match (a, b) {
        (&[a0], &[b0]) => (a0 | b0) != 0,
        (&[a0, a1], &[b0, b1]) => {
            (u16::from_ne_bytes([a0, a1]) | u16::from_ne_bytes([b0, b1])) != 0
        }
        (&[a0, a1, a2, a3], &[b0, b1, b2, b3]) => {
            (u32::from_ne_bytes([a0, a1, a2, a3]) | u32::from_ne_bytes([b0, b1, b2, b3])) != 0
        }
        (&[a0, a1, a2, a3, a4, a5, a6, a7], &[b0, b1, b2, b3, b4, b5, b6, b7]) => {
            (u64::from_ne_bytes([a0, a1, a2, a3, a4, a5, a6, a7])
                | u64::from_ne_bytes([b0, b1, b2, b3, b4, b5, b6, b7]))
                != 0
        }
        (
            &[a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15],
            &[b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15],
        ) => {
            (u64::from_ne_bytes([a0, a1, a2, a3, a4, a5, a6, a7])
                | u64::from_ne_bytes([a8, a9, a10, a11, a12, a13, a14, a15])
                | u64::from_ne_bytes([b0, b1, b2, b3, b4, b5, b6, b7])
                | u64::from_ne_bytes([b8, b9, b10, b11, b12, b13, b14, b15]))
                != 0
        }
        _ => any_nz2_slow(a, b),
    }
}

#[cold]
#[inline(never)]
fn any_nz2_slow(a: &[u8], b: &[u8]) -> bool {
    a.iter().zip(b).any(|(&x, &y)| (x | y) != 0)
}

#[inline(always)]
fn sign_sum(dc: &[u8]) -> i32 {
    const ONES32: u32 = 0x0101_0101;
    const ONES64: u64 = 0x0101_0101_0101_0101;
    match dc {
        &[d0] => (d0 >> 1) as i32 - (d0 & 1) as i32,
        &[d0, d1] => ((d0 >> 1) + (d1 >> 1)) as i32 - ((d0 & 1) + (d1 & 1)) as i32,
        &[d0, d1, d2, d3] => {
            let v = u32::from_ne_bytes([d0, d1, d2, d3]);
            let x = ((v >> 1) & ONES32) + (!v & ONES32);
            (x.wrapping_mul(ONES32) >> 24) as i32 - 4
        }
        &[d0, d1, d2, d3, d4, d5, d6, d7] => {
            let v = u64::from_ne_bytes([d0, d1, d2, d3, d4, d5, d6, d7]);
            let x = ((v >> 1) & ONES64) + (!v & ONES64);
            (x.wrapping_mul(ONES64) >> 56) as i32 - 8
        }
        &[d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15] => {
            let v0 = u64::from_ne_bytes([d0, d1, d2, d3, d4, d5, d6, d7]);
            let v1 = u64::from_ne_bytes([d8, d9, d10, d11, d12, d13, d14, d15]);
            let x0 = ((v0 >> 1) & ONES64) + (!v0 & ONES64);
            let x1 = ((v1 >> 1) & ONES64) + (!v1 & ONES64);
            ((x0 + x1).wrapping_mul(ONES64) >> 56) as i32 - 16
        }
        _ => sign_sum_slow(dc),
    }
}

#[cold]
#[inline(never)]
fn sign_sum_slow(dc: &[u8]) -> i32 {
    dc.iter()
        .fold(0i32, |s, &d| s + (d >> 1) as i32 - (d & 1) as i32)
}

#[inline(always)]
fn fill_ctx2(a: &mut [u8], b: &mut [u8], pos: usize, len: usize, va: u8, vb: u8) {
    debug_assert!(matches!(len, 1 | 2 | 4 | 8 | 16 | 32));
    let sa = &mut a[pos..pos + len];
    let sb = &mut b[pos..pos + len];
    match (sa, sb) {
        ([a0], [b0]) => {
            *a0 = va;
            *b0 = vb;
        }
        ([a0, a1], [b0, b1]) => {
            *a0 = va;
            *a1 = va;
            *b0 = vb;
            *b1 = vb;
        }
        (sa @ [_, _, _, _], sb @ [_, _, _, _]) => {
            sa.fill(va);
            sb.fill(vb);
        }
        (sa @ [_, _, _, _, _, _, _, _], sb @ [_, _, _, _, _, _, _, _]) => {
            sa.fill(va);
            sb.fill(vb);
        }
        (
            sa @ [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            sb @ [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
        ) => {
            sa.fill(va);
            sb.fill(vb);
        }
        (sa, sb) => {
            sa.fill(va);
            sb.fill(vb);
        }
    }
}

/// Context offset of `coeff_br` from the position (for pos != 0).
#[inline(always)]
fn br_pos_ctx<const CLS: usize, const BWL: usize>(pos: usize) -> usize {
    let mask = (1usize << BWL) - 1;
    let near = match CLS {
        0 => (pos & !((1usize << BWL) | 1)) == 0,
        1 => (pos & mask) == 0,
        _ => pos <= mask,
    };
    if near {
        7
    } else {
        14
    }
}

/// Raw `coeff_br` neighbour sum at `idx` in the padded levels array.
#[inline(always)]
fn hi_mag<const CLS: usize>(levels: &[u8], idx: usize, ls: usize) -> u32 {
    match CLS {
        0 => {
            let l = &levels[idx..idx + ls + 2];
            l[1] as u32 + l[ls] as u32 + l[ls + 1] as u32
        }
        1 => {
            let l = &levels[idx..idx + ls + 1];
            l[1] as u32 + l[ls] as u32 + l[2] as u32
        }
        _ => {
            let l = &levels[idx..idx + 2 * ls + 1];
            l[1] as u32 + l[ls] as u32 + l[2 * ls] as u32
        }
    }
}

/// Index of position `pos` in the levels array (see `coef_tokens`).
#[inline(always)]
fn level_idx<const CLS: usize, const BWL: usize>(pos: usize) -> usize {
    if CLS == 0 {
        pos
    } else {
        pos + ((pos >> BWL) << 2)
    }
}

/// Returns the `coeff_base` context of the coefficient at `idx` in the
/// levels array (stride `ls`), without the special case of the 2D DC.
/// `p` is the scan entry (position, plus the 2D context offset in bits
/// 10..).
#[inline(always)]
fn lo_ctx<const CLS: usize, const BWL: usize>(
    levels: &[u8],
    idx: usize,
    ls: usize,
    p: u16,
) -> usize {
    let pos = (p & 0x3FF) as usize;
    let (mag, off) = match CLS {
        0 => {
            let l = &levels[idx..idx + 2 * ls + 1];
            let mag =
                l[1] as u32 + l[2] as u32 + l[ls] as u32 + l[ls + 1] as u32 + l[2 * ls] as u32;
            (mag, ((p >> 10) & 31) as usize)
        }
        1 => {
            let l = &levels[idx..idx + ls + 1];
            let mag = l[1] as u32 + l[2] as u32 + l[3] as u32 + l[4] as u32 + l[ls] as u32;
            (mag, 26 + (pos & ((1 << BWL) - 1)).min(2) * 5)
        }
        _ => {
            let l = &levels[idx..idx + 4 * ls + 1];
            let mag =
                l[1] as u32 + l[ls] as u32 + l[2 * ls] as u32 + l[3 * ls] as u32 + l[4 * ls] as u32;
            (mag, 26 + (pos >> BWL).min(2) * 5)
        }
    };
    // The low 6 bits of the sum are at most 30 when the (min 3) high
    // parts sum to less than 8, so they never disturb the rounding below.
    let mag_ctx = ((mag + 64) >> 7).min(4) as usize;
    off + mag_ctx
}

/// Scan of the coefficient reader: `get_scan`, except that for the 2D
/// class every entry also carries the 2D `coeff_base` context offset of
/// its position in bits 10.. (`Coeff_Base_Ctx_Offset[txSz]`).
#[inline]
fn coef_scan(tx_sz: TxSize, tx_cls: usize, tx_type: TxType) -> &'static [u16] {
    use crate::decoder::av1::tables::scan::*;
    use TxSize::*;
    if tx_cls != 0 {
        return get_scan(tx_sz, tx_type);
    }
    match tx_sz {
        Tx4x4 => &COEF_SCAN_4X4,
        Tx8x8 => &COEF_SCAN_8X8,
        Tx16x16 => &COEF_SCAN_16X16,
        Tx32x32 | Tx64x64 => &COEF_SCAN_32X32,
        Tx4x8 => &COEF_SCAN_4X8,
        Tx8x4 => &COEF_SCAN_8X4,
        Tx8x16 => &COEF_SCAN_8X16,
        Tx16x8 => &COEF_SCAN_16X8,
        Tx16x32 | Tx16x64 => &COEF_SCAN_16X32,
        Tx32x16 | Tx64x16 => &COEF_SCAN_32X16,
        Tx32x64 => &COEF_SCAN_32X32_TALL,
        Tx64x32 => &COEF_SCAN_32X32_WIDE,
        Tx4x16 => &COEF_SCAN_4X16,
        Tx16x4 => &COEF_SCAN_16X4,
        Tx8x32 => &COEF_SCAN_8X32,
        Tx32x8 => &COEF_SCAN_32X8,
    }
}

/// `coeff_br` magnitude context from the raw neighbour sum.
#[inline(always)]
fn hi_ctx(hi: u32) -> usize {
    HI_CTX[(hi & 63) as usize] as usize
}

/// `coeff_br` magnitude context: `mag > 12 ? 6 : (mag + 1) >> 1`.
static HI_CTX: [u8; 64] = {
    let mut t = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        t[i] = if i > 12 { 6 } else { ((i + 1) >> 1) as u8 };
        i += 1;
    }
    t
};

mod modeinfo;

#[cfg(test)]
mod tests {
    use super::*;

    /// `coef_scan` is `get_scan` plus, for the 2D class, the spec's
    /// `Coeff_Base_Ctx_Offset[txSz][min(row, 4)][min(col, 4)]` in bits 10...
    #[test]
    fn coef_scan_ctx_offsets() {
        for ts in 0..19u8 {
            let tx_sz = TxSize::from_u8(ts);
            let adj = ADJUSTED_TX_SIZE[ts as usize] as usize;
            let bwl = TX_WIDTH_LOG2[adj] as usize;
            for tx_type in [TxType::DCT_DCT, TxType::IDTX, TxType::V_DCT, TxType::H_DCT] {
                let cls = tx_class(tx_type);
                let plain = get_scan(tx_sz, tx_type);
                let scan = coef_scan(tx_sz, cls, tx_type);
                assert_eq!(plain.len(), scan.len());
                for (&p, &s) in plain.iter().zip(scan) {
                    assert_eq!(p, s & 0x3FF);
                    if cls == 0 {
                        let (row, col) = ((p as usize) >> bwl, (p as usize) & ((1 << bwl) - 1));
                        let off = COEFF_BASE_CTX_OFFSET[ts as usize][row.min(4)][col.min(4)];
                        assert_eq!((s >> 10) as u8, off, "{tx_sz:?} pos {p}");
                    } else {
                        assert_eq!(s, p);
                    }
                }
            }
        }
    }
}
