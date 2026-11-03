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

/// Register-sized internal result for block/partition decoding (`Ok(())` is
/// `ptr == 0` in `x0`/`rax`, avoiding the 24-byte stack return of `Result<_, Error>`).
type BlockResult<T = ()> = std::result::Result<T, &'static str>;

const TX_DQ_SHIFT: [u8; TxSize::COUNT] = {
    let mut t = [0u8; TxSize::COUNT];
    t[TxSize::Tx32x32 as usize] = 1;
    t[TxSize::Tx16x32 as usize] = 1;
    t[TxSize::Tx32x16 as usize] = 1;
    t[TxSize::Tx16x64 as usize] = 1;
    t[TxSize::Tx64x16 as usize] = 1;
    t[TxSize::Tx64x64 as usize] = 2;
    t[TxSize::Tx32x64 as usize] = 2;
    t[TxSize::Tx64x32 as usize] = 2;
    t
};

const TX_SZ_CTX: [u8; TxSize::COUNT] = {
    let mut t = [0u8; TxSize::COUNT];
    let mut i = 0;
    while i < TxSize::COUNT {
        t[i] = (TX_SIZE_SQR[i] as u8 + TX_SIZE_SQR_UP[i] as u8 + 1) >> 1;
        i += 1;
    }
    t
};

const EOB_MULTISIZE: [u8; TxSize::COUNT] = {
    let mut t = [0u8; TxSize::COUNT];
    let mut i = 0;
    while i < TxSize::COUNT {
        let w = if TX_WIDTH_LOG2[i] < 5 { TX_WIDTH_LOG2[i] } else { 5 };
        let h = if TX_HEIGHT_LOG2[i] < 5 { TX_HEIGHT_LOG2[i] } else { 5 };
        t[i] = w + h - 4;
        i += 1;
    }
    t
};

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
    pub dc_q: [u32; 3],
    pub ac_q: [u32; 3],
    pub qm_level: [u8; 3],
    pub cdef_unread: u8,
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
            dc_q: [0; 3],
            ac_q: [0; 3],
            qm_level: [15; 3],
            cdef_unread: 0,
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
    /// Dequantized coefficients for the inverse transform.
    pub coeffs: [i32; 1024],
    /// Padded 2D copy of the coefficient levels for context derivation.
    pub levels: [u8; 2200],
    /// Non-zero AC tokens of the current transform block in reverse scan
    /// order, as `level << 10 | pos` (see `coef_tokens`).
    pub nz: [u16; 1024],
    /// Transform types per 4x4 of the current block (`TxTypes`), indexed
    /// relative to the block's top-left (32x32 4x4 blocks max).
    pub tx_types: [[TxType; 32]; 32],
    /// `BlockDecoded[plane][y + 1]` bitmask (bit `x + 1` is column `x`).
    /// Only rows `0..(sb4 >> ss_y) + 2` (at most 34) are cleared per
    /// superblock and meaningfully read. In `transform_block`, `by <= 32`
    /// (`sub_y` is masked with `& 31`) and the real `step_y <= 16`, but
    /// `step_y` comes from the `u8` `TX_HEIGHT` table, so the compiler only
    /// knows `step_y <= 63`; 96 rows (`> 32 + 63`) keep the `by + step_y`
    /// index and the `by..by + 16` slice free of bounds checks.
    pub block_decoded: [[BlockDecodedRow; 96]; 3],
    pub color_map_y: Box<[[u8; 64]; 64]>,
    pub color_map_uv: Box<[[u8; 64]; 64]>,
}

impl Scratch {
    pub fn new() -> Self {
        Self {
            coeffs: [0; 1024],
            levels: [0; 2200],
            nz: [0; 1024],
            tx_types: [[TxType::DCT_DCT; 32]; 32],
            block_decoded: [[BlockDecodedRow(0); 96]; 3],
            color_map_y: Box::new([[0; 64]; 64]),
            color_map_uv: Box::new([[0; 64]; 64]),
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

    #[inline(always)]
    fn residual(&mut self) -> BlockResult {
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

    #[cold]
    #[inline(never)]
    fn transform_tree(
        &mut self,
        start_x: usize,
        start_y: usize,
        w: usize,
        h: usize,
    ) -> BlockResult {
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
    ) -> BlockResult {
        let f = self.f;
        let pt = CHROMA as usize;
        let start_x = base_x + 4 * x;
        let start_y = base_y + 4 * y;
        if start_x >= f.max_x[pt] || start_y >= f.max_y[pt] {
            return Ok(());
        }
        let sub_x = (start_x >> 2) & (f.sb_mask_x[pt] & 31);
        let sub_y = (start_y >> 2) & (f.sb_mask_y[pt] & 31);
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
            let eob = self.coeffs::<CHROMA>(plane, start_x, start_y, sub_y, step_x, step_y, tx_sz);
            if eob > 0 {
                if eob == u32::MAX {
                    return Err("coef");
                }
                if c.recon {
                    self.reconstruct(plane, start_x, start_y, tx_sz, eob as usize);
                } else {
                    self.clear_coeffs(tx_sz);
                }
            }
        }
        // BlockDecoded.
        let mask = ((1u64 << step_x) - 1) << bx;
        let bd = if !CHROMA {
            &mut self.sc.block_decoded[0][by..by + 16]
        } else {
            &mut self.sc.block_decoded[plane][by..by + 16]
        };
        if step_y < 4 {
            bd[0].0 |= mask;
            bd[step_y & 1 ^ 1].0 |= mask;
        } else if step_y == 4 {
            for r in &mut bd[..4] {
                r.0 |= mask;
            }
        } else {
            let o = (step_y >> 1) & 8;
            for r in &mut bd[..8] {
                r.0 |= mask;
            }
            for r in &mut bd[o..o + 8] {
                r.0 |= mask;
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Transform type
    // ------------------------------------------------------------------

    #[inline(always)]
    fn get_tx_set(&self, tx_sz: TxSize) -> usize {
        let sqr = TX_SIZE_SQR[tx_sz as usize];
        let sqr_up = TX_SIZE_SQR_UP[tx_sz as usize];
        if sqr_up > TxSize::Tx32x32 {
            return TX_SET_DCTONLY;
        }
        if !self.b.is_inter {
            if sqr_up == TxSize::Tx32x32 {
                TX_SET_DCTONLY
            } else if self.f.fh.reduced_tx_set || sqr == TxSize::Tx16x16 {
                TX_SET_INTRA_2
            } else {
                TX_SET_INTRA_1
            }
        } else if self.f.fh.reduced_tx_set || sqr_up == TxSize::Tx32x32 {
            TX_SET_INTER_3
        } else if sqr == TxSize::Tx16x16 {
            TX_SET_INTER_2
        } else {
            TX_SET_INTER_1
        }
    }

    #[inline(always)]
    fn is_tx_type_in_set(&self, set: usize, t: TxType) -> bool {
        const INTRA_MASK: u64 = {
            let mut m = 0u64;
            let mut s = 0;
            while s < 3 {
                let mut i = 0;
                while i < 16 {
                    if TX_TYPE_IN_SET_INTRA[s][i] {
                        m |= 1u64 << (s * 16 + i);
                    }
                    i += 1;
                }
                s += 1;
            }
            m
        };
        const INTER_MASK: u64 = {
            let mut m = 0u64;
            let mut s = 0;
            while s < 4 {
                let mut i = 0;
                while i < 16 {
                    if TX_TYPE_IN_SET_INTER[s][i] {
                        m |= 1u64 << (s * 16 + i);
                    }
                    i += 1;
                }
                s += 1;
            }
            m
        };
        let mask = if self.b.is_inter { INTER_MASK } else { INTRA_MASK };
        ((mask >> ((set << 4) | t as usize)) & 1) != 0
    }

    /// Reads `transform_type` for a luma transform block at (x4, y4)
    /// (absolute 4x4 units), stores it in the block's TxTypes when needed
    /// for inter chroma derivation, and returns it.
    #[inline(always)]
    fn transform_type(&mut self, x4: usize, y4: usize, tx_sz: TxSize) -> TxType {
        let set = self.get_tx_set(tx_sz);
        if !self.b.is_inter {
            if set > 0 && self.f.fh.seg_qindex_nz[self.b.segment_id & 7] {
                let sqr = (TX_SIZE_SQR[tx_sz as usize] as usize).min(2);
                let intra_dir = if self.b.use_filter_intra {
                    FILTER_INTRA_MODE_TO_INTRA_DIR[(self.b.filter_intra_mode as usize).min(4)]
                        as usize
                } else {
                    (self.b.y_mode as usize).min(12)
                };
                let msac = &mut self.t.msac;
                let cdf = &mut *self.t.cdf;
                if set == TX_SET_INTRA_1 {
                    let v = msac.symbol_n(&mut cdf.intra_tx_type_set1[sqr.min(1)][intra_dir]);
                    TX_TYPE_INTRA_INV_SET1[v & 7]
                } else {
                    let v = msac.symbol6(&mut cdf.intra_tx_type_set2[sqr][intra_dir]);
                    TX_TYPE_INTRA_INV_SET2[v.min(4)]
                }
            } else {
                TxType::DCT_DCT
            }
        } else {
            self.transform_type_inter(x4, y4, tx_sz, set)
        }
    }

    #[cold]
    #[inline(never)]
    fn transform_type_inter(&mut self, x4: usize, y4: usize, tx_sz: TxSize, set: usize) -> TxType {
        let tx_type = if set > 0 && self.f.fh.seg_qindex_nz[self.b.segment_id & 7] {
            // inter_tx_type_set3 has one CDF per square transform size up to
            // TX_32X32, so the index must not be clamped below 3.
            let sqr = (TX_SIZE_SQR[tx_sz as usize] as usize).min(3);
            let msac = &mut self.t.msac;
            let cdf = &mut *self.t.cdf;
            match set {
                TX_SET_INTER_1 => {
                    let v = msac.symbol_n(&mut cdf.inter_tx_type_set1[sqr.min(1)]);
                    TX_TYPE_INTER_INV_SET1[v & 15]
                }
                TX_SET_INTER_2 => {
                    let v = msac.symbol_n(&mut cdf.inter_tx_type_set2);
                    TX_TYPE_INTER_INV_SET2[v.min(11)]
                }
                _ => {
                    let v = msac.symbol_n(&mut cdf.inter_tx_type_set3[sqr]);
                    TX_TYPE_INTER_INV_SET3[v & 7]
                }
            }
        } else {
            TxType::DCT_DCT
        };
        self.set_tx_types(x4, y4, tx_sz, tx_type);
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

    /// Spec `compute_tx_type(plane, txSz, blockX, blockY)` for `plane > 0`.
    #[inline(always)]
    fn compute_chroma_tx_type(&self, tx_sz: TxSize, block_x: usize, block_y: usize) -> TxType {
        let sqr_up = TX_SIZE_SQR_UP[tx_sz as usize];
        if !self.b.is_inter {
            return if self.b.lossless || sqr_up >= TxSize::Tx32x32 {
                TxType::DCT_DCT
            } else {
                MODE_TO_TXFM[(self.b.uv_mode as usize).min(13)]
            };
        }
        if self.b.lossless || sqr_up > TxSize::Tx32x32 {
            return TxType::DCT_DCT;
        }
        let set = self.get_tx_set(tx_sz);
        let x4 = self.b.mi_col.max(block_x << self.f.ss_x);
        let y4 = self.b.mi_row.max(block_y << self.f.ss_y);
        let t = self.sc.tx_types[(y4 - self.b.mi_row) & 31][(x4 - self.b.mi_col) & 31];
        if !self.is_tx_type_in_set(set, t) {
            TxType::DCT_DCT
        } else {
            t
        }
    }

    // ------------------------------------------------------------------
    // Coefficients
    // ------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn coeffs<const CHROMA: bool>(
        &mut self,
        plane: usize,
        start_x: usize,
        start_y: usize,
        ly: usize,
        w4: usize,
        h4: usize,
        tx_sz: TxSize,
    ) -> u32 {
        let f = self.f;
        let x4 = start_x >> 2;
        let y4 = start_y >> 2;
        let tx_sz_ctx = TX_SZ_CTX[tx_sz as usize] as usize;
        let (sx, sy) = if !CHROMA { (0, 0) } else { (f.ss_x, f.ss_y) };
        let ax0 = x4 - (self.s.above.col0 >> sx);

        let all_zero_ctx = if !CHROMA && self.b.bw4 == w4 && self.b.bh4 == h4 {
            0
        } else if !CHROMA {
            let kx = if w4 == 1 { 1 } else { w4.min(f.mi_cols - x4) };
            let ky = if h4 == 1 { 1 } else { h4.min(f.mi_rows - y4) };
            let top = or_slice(&self.s.above.level[0][ax0..ax0 + kx]) as usize;
            let left = or_slice(&self.s.left.level[0][ly..ly + ky]) as usize;
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
        } else {
            let kx = if w4 == 1 { 1 } else { w4.min((f.mi_cols >> sx) - x4) };
            let ky = if h4 == 1 { 1 } else { h4.min((f.mi_rows >> sy) - y4) };
            let bsize = f.uv_residual_size[self.b.bsize as usize];
            let above = (or_slice(&self.s.above.level[plane][ax0..ax0 + kx])
                | or_slice(&self.s.above.dc[plane][ax0..ax0 + kx]))
                != 0;
            let left = (or_slice(&self.s.left.level[plane][ly..ly + ky])
                | or_slice(&self.s.left.dc[plane][ly..ly + ky]))
                != 0;
            let mut ctx = above as usize + left as usize + 7;
            if (bw4(bsize) != w4) | (bh4(bsize) != h4) {
                ctx += 3;
            }
            ctx
        };

        let all_zero = self
            .t
            .msac
            .bool_adapt(&mut self.t.cdf.txb_skip[tx_sz_ctx][all_zero_ctx]);
        if all_zero {
            if !CHROMA && self.b.is_inter {
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
            return 0;
        }
        self.coeffs_nz(plane, tx_sz, ax0, ly, y4)
    }

    #[inline(never)]
    fn coeffs_nz(&mut self, plane: usize, tx_sz: TxSize, ax0: usize, ly: usize, y4: usize) -> u32 {
        let f = self.f;
        let ts = tx_sz as usize;
        let w4 = (TX_WIDTH[ts] >> 2) as usize;
        let h4 = (TX_HEIGHT[ts] >> 2) as usize;
        let tx_sz_ctx = TX_SZ_CTX[ts] as usize;
        let ptype = (plane > 0) as usize;
        let x4 = ax0 + (self.s.above.col0 >> if plane == 0 { 0 } else { f.ss_x });
        let tx_type = if plane == 0 {
            self.transform_type(x4, y4, tx_sz)
        } else {
            self.compute_chroma_tx_type(tx_sz, x4, y4)
        };
        self.plane_tx_type = tx_type;
        let tx_cls = tx_class(tx_type);
        let eob_multisize = EOB_MULTISIZE[ts];
        let eob_ctx = (tx_cls != 0) as usize;
        let msac = &mut self.t.msac;
        let cdf = &mut *self.t.cdf;
        let eob_bin = if eob_multisize == 0 {
            msac.symbol6(&mut cdf.eob_pt_16[ptype][eob_ctx])
        } else {
            match eob_multisize {
                1 => msac.symbol_n(&mut cdf.eob_pt_32[ptype][eob_ctx]),
                2 => msac.symbol_n(&mut cdf.eob_pt_64[ptype][eob_ctx]),
                3 => msac.symbol_n(&mut cdf.eob_pt_128[ptype][eob_ctx]),
                4 => msac.symbol_n(&mut cdf.eob_pt_256[ptype][eob_ctx]),
                5 => msac.symbol_n(&mut cdf.eob_pt_512[ptype]),
                _ => msac.symbol_n(&mut cdf.eob_pt_1024[ptype]),
            }
        };
        let mut last = eob_bin;
        if eob_bin >= 2 {
            let eb = (eob_bin - 2).min(8);
            let hi = msac.bool_adapt(&mut cdf.eob_extra[tx_sz_ctx][ptype][eb]) as usize;
            last = ((hi | 2) << eb) | if eb != 0 { msac.literal(eb as u32) as usize } else { 0 };
        }
        let eob = last + 1;

        // Levels are kept in a padded 2D array for context derivation;
        // non-zero entries are cleared while walking `nz` below.
        // `lbwl` maps a position to its levels index (`pos + (pos >> lbwl
        // << 2)`): the 2D class uses an unpadded array (see coef_tokens).
        if eob != 1 {
            return self.coeffs_ac(plane, tx_sz, ax0, ly, y4, eob);
        }
        // DC only (eob == 1), the common case of small transforms: kept
        // separate from the coefficient loops so that this path stays
        // small (few spills) and needs none of the AC setup.
        let mut tok = msac.symbol_n(&mut cdf.coeff_base_eob[tx_sz_ctx][ptype][0]) as u32 + 1;
        if tok == 3 {
            tok = msac.hi_tok(&mut cdf.coeff_br[tx_sz_ctx.min(3)][ptype][0]);
        }
        let Some((q, dc_category)) = self.dc_coef(plane, tx_sz, ax0, ly, y4, tok) else {
            return u32::MAX;
        };
        let cul_level = q.min(63) as u8;
        fill_ctx2(
            &mut self.s.above.level[plane],
            &mut self.s.above.dc[plane],
            ax0,
            w4,
            cul_level,
            dc_category,
        );
        fill_ctx2(
            &mut self.s.left.level[plane],
            &mut self.s.left.dc[plane],
            ly,
            h4,
            cul_level,
            dc_category,
        );
        1
    }

    /// Reads the sign of the non-zero DC token `dc_tok` (plus its golomb
    /// remainder) and stores the dequantized DC coefficient. Returns the
    /// level and the DC category (1 negative, 2 positive) for the context
    /// arrays, or `None` on an invalid golomb code.
    #[inline(always)]
    fn dc_coef(
        &mut self,
        plane: usize,
        tx_sz: TxSize,
        ax0: usize,
        ly: usize,
        y4: usize,
        dc_tok: u32,
    ) -> Option<(u32, u8)> {
        let f = self.f;
        let ts = tx_sz as usize;
        let w4 = (TX_WIDTH[ts] >> 2) as usize;
        let h4 = (TX_HEIGHT[ts] >> 2) as usize;
        let ptype = (plane > 0) as usize;
        let dc_sign = if (w4 | h4) == 1 {
            let da = self.s.above.dc[plane][ax0] as i32;
            let dl = self.s.left.dc[plane][ly] as i32;
            (da >> 1) + (dl >> 1) - (da & 1) - (dl & 1)
        } else {
            let (sx, sy) = f.sub(plane);
            let x4 = ax0 + (self.s.above.col0 >> sx);
            let kx = w4.min((f.mi_cols >> sx) - x4);
            let ky = h4.min((f.mi_rows >> sy) - y4);
            sign_sum(&self.s.above.dc[plane][ax0..ax0 + kx])
                + sign_sum(&self.s.left.dc[plane][ly..ly + ky])
        };
        let ctx = (dc_sign < 0) as usize + 2 * (dc_sign > 0) as usize;
        let sign = self.t.msac.bool_adapt(&mut self.t.cdf.dc_sign[ptype][ctx]);
        let dc_q = self.b.dc_q[plane];
        let qm_level = self.b.qm_level[plane] as usize;
        let use_qm = qm_level < 15 && (self.plane_tx_type as usize) < TxType::IDTX as usize;
        let dq_shift = (TX_DQ_SHIFT[ts] & 31) as u32;
        let mut q = dc_tok;
        let dq = if q < 15 {
            // No `& 0xFFFFFF` / `cf_max` clamp needed: with `q <= 14` and
            // QM DC weights `<= QM_MAX_DC_WEIGHT` (35, checked at compile
            // time in tables/qm.rs) the worst case at 12-bit is
            // `14 * ((21387 * 35 + 16) >> 5) = 327488 < cf_max = 524287`
            // (see the `qm_dequant_fast_paths` test).
            let q2 = if use_qm {
                let qm0 =
                    crate::decoder::av1::tables::qm::quantizer_matrix_1024(qm_level, ptype, tx_sz)
                        [0] as u32;
                (dc_q * qm0 + 16) >> 5
            } else {
                dc_q
            };
            let d = ((q * q2) >> dq_shift) as i32;
            if sign {
                -d
            } else {
                d
            }
        } else {
            let g = self.t.msac.golomb_coef()?;
            q = g.wrapping_add(14) & 0xFFFFF;
            // Largest magnitude of a positive coefficient; negative ones
            // may be one larger.
            let cf_max = (1u64 << (7 + f.bd)) - 1;
            self.dequant_dc_slow(
                q, dc_q, qm_level, ptype, tx_sz, dq_shift, cf_max, sign, use_qm,
            )
        };
        self.sc.coeffs[0] = dq;
        Some((q, if sign { 1 } else { 2 }))
    }

    /// `coeffs_nz` for `eob > 1`: coefficient tokens, then signs, golomb
    /// remainders and dequantization (spec 7.12.3 step 1) in scan order
    /// over the non-zero coefficients.
    #[inline(never)]
    fn coeffs_ac(
        &mut self,
        plane: usize,
        tx_sz: TxSize,
        ax0: usize,
        ly: usize,
        y4: usize,
        eob: usize,
    ) -> u32 {
        let f = self.f;
        let ts = tx_sz as usize;
        let w4 = (TX_WIDTH[ts] >> 2) as usize;
        let h4 = (TX_HEIGHT[ts] >> 2) as usize;
        let tx_sz_ctx = TX_SZ_CTX[ts] as usize;
        let ptype = (plane > 0) as usize;
        let tx_type = self.plane_tx_type;
        let tx_cls = tx_class(tx_type);
        // Levels are kept in a padded 2D array for context derivation;
        // non-zero entries are cleared while walking `nz` below.
        // `lbwl` maps a position to its levels index (`pos + (pos >> lbwl
        // << 2)`): the 2D class uses an unpadded array (see coef_tokens).
        let (dc_tok, num_ac, lbwl) = {
            let scan = coef_scan(tx_sz, tx_cls);
            let Some(scan) = scan.get(..eob) else {
                return u32::MAX;
            };
            let adj = ADJUSTED_TX_SIZE[ts] as usize;
            let bwl = (TX_WIDTH_LOG2[adj] as usize).clamp(2, 5);
            let height = TX_HEIGHT[adj] as usize;
            let (dc_tok, num_ac) = if self.t.msac.allow_update() {
                match (tx_cls, bwl) {
                    (0, 2) => self.coef_tokens::<0, 2, true>(scan, tx_sz_ctx, ptype, height),
                    (0, 3) => self.coef_tokens::<0, 3, true>(scan, tx_sz_ctx, ptype, height),
                    (0, 4) => self.coef_tokens::<0, 4, true>(scan, tx_sz_ctx, ptype, height),
                    (0, _) => self.coef_tokens::<0, 5, true>(scan, tx_sz_ctx, ptype, height),
                    (1, 2) => self.coef_tokens::<1, 2, true>(scan, tx_sz_ctx, ptype, height),
                    (1, 3) => self.coef_tokens::<1, 3, true>(scan, tx_sz_ctx, ptype, height),
                    (1, _) => self.coef_tokens::<1, 4, true>(scan, tx_sz_ctx, ptype, height),
                    (_, 2) => self.coef_tokens::<2, 2, true>(scan, tx_sz_ctx, ptype, height),
                    (_, 3) => self.coef_tokens::<2, 3, true>(scan, tx_sz_ctx, ptype, height),
                    (_, _) => self.coef_tokens::<2, 4, true>(scan, tx_sz_ctx, ptype, height),
                }
            } else {
                self.coef_tokens_no_upd(tx_cls, bwl, scan, tx_sz_ctx, ptype, height)
            };
            (dc_tok, num_ac, if tx_cls == 0 { 10 } else { bwl })
        };

        let ac_q = self.b.ac_q[plane];
        let qm_level = self.b.qm_level[plane] as usize;
        let use_qm = qm_level < 15 && (tx_type as usize) < TxType::IDTX as usize;
        let dq_shift = (TX_DQ_SHIFT[ts] & 31) as u32;
        // Largest magnitude of a positive coefficient; negative ones
        // may be one larger.
        let cf_max = (1u64 << (7 + f.bd)) - 1;
        let mut cul_level = 0u32;
        let mut dc_category = 0u8;
        if dc_tok != 0 {
            let Some((q, cat)) = self.dc_coef(plane, tx_sz, ax0, ly, y4, dc_tok) else {
                return u32::MAX;
            };
            cul_level += q;
            dc_category = cat;
        }
        let n_ac = num_ac.min(1024);
        if n_ac != 0 {
            if use_qm {
                let Some(c) = self.dequant_ac_qm(n_ac, plane, tx_sz, lbwl.max(2)) else {
                    return u32::MAX;
                };
                cul_level += c;
            } else {
                #[cold]
                #[inline(always)]
                fn cold_hi_tok() {}

                let mut m = self.t.msac.clone();
                let msac = &mut m;
                let sc = &mut *self.sc;
                let cf = &mut sc.coeffs;
                let levels = &mut sc.levels;
                let nz = &sc.nz[..n_ac];
                if tx_cls == 0 {
                    for &e in nz.iter().rev() {
                        let e = e as u32;
                        let rc = (e & 0x3FF) as usize;
                        levels[rc] = 0;
                        let q = e >> 10;
                        if q < 15 {
                            let d = ((q * ac_q) >> dq_shift) as i32;
                            cul_level += q;
                            msac.bool_store_neg(&mut cf[rc], d);
                        } else {
                            cold_hi_tok();
                            let (q_hi, dq) = msac.ac_golomb(ac_q, dq_shift, cf_max);
                            if q_hi == 0 {
                                return u32::MAX;
                            }
                            cul_level += q_hi & 0xFFFFF;
                            cf[rc] = dq;
                        }
                    }
                } else {
                    let lbwl = lbwl.max(2);
                    for &e in nz.iter().rev() {
                        let e = e as u32;
                        let rc = (e & 0x3FF) as usize;
                        levels[(rc as u32 + ((rc as u32 >> lbwl) << 2)) as usize] = 0;
                        let q = e >> 10;
                        if q < 15 {
                            let d = ((q * ac_q) >> dq_shift) as i32;
                            cul_level += q;
                            msac.bool_store_neg(&mut cf[rc], d);
                        } else {
                            cold_hi_tok();
                            let (q_hi, dq) = msac.ac_golomb(ac_q, dq_shift, cf_max);
                            if q_hi == 0 {
                                return u32::MAX;
                            }
                            cul_level += q_hi & 0xFFFFF;
                            cf[rc] = dq;
                        }
                    }
                }
                self.t.msac.copy_state(&m);
            }
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
        eob as u32
    }

    #[cold]
    #[inline(never)]
    fn dequant_ac_qm(
        &mut self,
        n_ac: usize,
        plane: usize,
        tx_sz: TxSize,
        lbwl: usize,
    ) -> Option<u32> {
        #[cold]
        #[inline(always)]
        fn cold_hi_tok() {}

        let ptype = (plane > 0) as usize;
        let qm_level = (self.b.qm_level[plane] as usize).min(14);
        let ac_q = self.b.ac_q[plane];
        let dq_shift = (TX_DQ_SHIFT[tx_sz as usize] & 31) as u32;
        let cf_max = (1u64 << (7 + self.f.bd)) - 1;
        let tx_cls = tx_class(self.plane_tx_type);
        let mut m = self.t.msac.clone();
        let msac = &mut m;
        let sc = &mut *self.sc;
        let cf = &mut sc.coeffs;
        let levels = &mut sc.levels;
        let nz = &sc.nz[..n_ac];
        let qm = crate::decoder::av1::tables::qm::quantizer_matrix_1024(qm_level, ptype, tx_sz);
        let cf_max32 = cf_max as u32;
        // Largest `q < 15` coefficient magnitude for any QM weight (<= 255).
        // If it fits in `cf_max`, the clamp is a no-op; `& 0xFFFFFF` is
        // always a no-op for `q < 15` (see the `qm_dequant_fast_paths` test).
        let max_d = (14 * ((ac_q * 255 + 16) >> 5)) >> dq_shift;
        let mut cul_level = 0u32;
        if max_d <= cf_max32 && tx_cls == 0 {
            for &e in nz.iter().rev() {
                let e = e as u32;
                let rc = (e & 0x3FF) as usize;
                let q2 = (ac_q * qm[rc] as u32 + 16) >> 5;
                levels[rc] = 0;
                let q = e >> 10;
                if q < 15 {
                    let d = ((q * q2) >> dq_shift) as i32;
                    cul_level += q;
                    msac.bool_store_neg(&mut cf[rc], d);
                } else {
                    cold_hi_tok();
                    let (q_hi, dq) = msac.ac_golomb(q2, dq_shift, cf_max);
                    if q_hi == 0 {
                        return None;
                    }
                    cul_level += q_hi & 0xFFFFF;
                    cf[rc] = dq;
                }
            }
        } else {
            for &e in nz.iter().rev() {
                let e = e as u32;
                let rc = (e & 0x3FF) as usize;
                let q2 = (ac_q * qm[rc] as u32 + 16) >> 5;
                levels[(rc as u32 + ((rc as u32 >> lbwl) << 2)) as usize] = 0;
                let q = e >> 10;
                if q < 15 {
                    let d = (q * q2) >> dq_shift;
                    cul_level += q;
                    msac.bool_store_neg_qm(&mut cf[rc], d, cf_max32);
                } else {
                    cold_hi_tok();
                    let (q_hi, dq) = msac.ac_golomb(q2, dq_shift, cf_max);
                    if q_hi == 0 {
                        return None;
                    }
                    cul_level += q_hi & 0xFFFFF;
                    cf[rc] = dq;
                }
            }
        }
        self.t.msac.copy_state(&m);
        Some(cul_level)
    }

    #[cold]
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn dequant_dc_slow(
        &self,
        q: u32,
        dc_q: u32,
        qm_level: usize,
        ptype: usize,
        tx_sz: TxSize,
        dq_shift: u32,
        cf_max: u64,
        sign: bool,
        use_qm: bool,
    ) -> i32 {
        let qv = dc_q as u64;
        let q2 = if use_qm {
            let qm =
                crate::decoder::av1::tables::qm::quantizer_matrix(qm_level.min(14), ptype, tx_sz);
            (qv * qm[0] as u64 + 16) >> 5
        } else {
            qv
        };
        let dq = (((q as u64 * q2) & 0xFFFFFF) >> dq_shift).min(cf_max + sign as u64) as i32;
        if sign {
            -dq
        } else {
            dq
        }
    }

    /// `coef_tokens` dispatch for tiles with CDF updates disabled.
    #[cold]
    #[inline(never)]
    fn coef_tokens_no_upd(
        &mut self,
        tx_cls: usize,
        bwl: usize,
        scan: &[u16],
        tx_sz_ctx: usize,
        ptype: usize,
        height: usize,
    ) -> (u32, usize) {
        match (tx_cls, bwl) {
            (0, 2) => self.coef_tokens::<0, 2, false>(scan, tx_sz_ctx, ptype, height),
            (0, 3) => self.coef_tokens::<0, 3, false>(scan, tx_sz_ctx, ptype, height),
            (0, 4) => self.coef_tokens::<0, 4, false>(scan, tx_sz_ctx, ptype, height),
            (0, _) => self.coef_tokens::<0, 5, false>(scan, tx_sz_ctx, ptype, height),
            (1, 2) => self.coef_tokens::<1, 2, false>(scan, tx_sz_ctx, ptype, height),
            (1, 3) => self.coef_tokens::<1, 3, false>(scan, tx_sz_ctx, ptype, height),
            (1, _) => self.coef_tokens::<1, 4, false>(scan, tx_sz_ctx, ptype, height),
            (_, 2) => self.coef_tokens::<2, 2, false>(scan, tx_sz_ctx, ptype, height),
            (_, 3) => self.coef_tokens::<2, 3, false>(scan, tx_sz_ctx, ptype, height),
            (_, _) => self.coef_tokens::<2, 4, false>(scan, tx_sz_ctx, ptype, height),
        }
    }

    /// Reads the coefficient tokens (levels up to 15) of a transform block
    /// in reverse scan order (`coeff_base_eob`, `coeff_base`, `coeff_br`)
    /// for `eob > 1`.
    ///
    /// Like dav1d, the `levels` array holds per position
    /// `min(level, 3) << 6 | level`, so that one sum over the neighbours
    /// gives both the `coeff_base` and the `coeff_br` magnitudes. Non-zero
    /// AC tokens are appended to `nz` as `level << 10 | pos`, in reverse
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
        let ts_ctx = tx_sz_ctx.min(4);
        let pt = ptype & 1;
        let eob_cdf = &mut cdf.coeff_base_eob[ts_ctx][pt];
        let lo_cdf = &mut cdf.coeff_base[ts_ctx][pt];
        let hi_cdf = &mut cdf.coeff_br[tx_sz_ctx.min(3)][pt];
        let sc = &mut *self.sc;
        let levels = &mut sc.levels;
        let nz = &mut sc.nz;

        // Last coefficient (coeff_base_eob, with last = eob - 1 >= 1).
        // All levels are still zero, so the coeff_br magnitude is 0.
        let Some((&last_p, rest)) = scan.split_last() else {
            return (0, 0);
        };
        let Some((_, ac_scan)) = rest.split_first() else {
            return (0, 0);
        };
        let last = rest.len();
        let area = height << BWL;
        let pos = (last_p & 0x3FF) as usize;
        let ctx = if last <= area / 8 {
            1
        } else if last <= area / 4 {
            2
        } else {
            3
        };
        #[cold]
        #[inline(always)]
        fn cold_hi_tok() {}

        let mut tok = msac.symbol4_c::<UPD, true>(&mut eob_cdf[ctx]) + 1;
        let lb = if tok == 3 {
            cold_hi_tok();
            tok = msac.hi_tok_c::<UPD>(&mut hi_cdf[br_pos_ctx::<CLS, BWL>(pos)]);
            (tok as u8) | (3 << 6)
        } else {
            (tok as u8) * 0x41
        };
        levels[level_idx::<CLS, BWL>(pos)] = lb;
        nz[0] = ((tok as u16) << 10) | pos as u16;
        let mut k = 1usize;

        for &p in ac_scan.iter().rev() {
            let p32 = p as u32;
            let pos = (p32 & 0x3FF) as usize;
            let idx = level_idx::<CLS, BWL>(pos);
            let ctx = lo_ctx::<CLS, BWL>(levels, idx, ls, p32);
            let tok = msac.symbol5_c::<UPD>(&mut lo_cdf[ctx]);
            // k < scan.len() <= 1024.
            if tok == 3 {
                cold_hi_tok();
                let hi = hi_mag::<CLS>(levels, idx, ls);
                let tok =
                    msac.hi_tok_c::<UPD>(&mut hi_cdf[br_pos_ctx::<CLS, BWL>(pos) + hi_ctx(hi)]);
                levels[idx] = (tok as u8) | (3 << 6);
                nz[k & 1023] = ((tok as u16) << 10) | pos as u16;
                k += 1;
            } else {
                // Branch free (zero / non-zero is unpredictable), like
                // dav1d: zero tokens store 0 (levels are zero there
                // already) and are dropped from `nz` by not advancing k.
                levels[idx] = (tok * 0x41) as u8;
                nz[k & 1023] = ((tok as u16) << 10) | pos as u16;
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
            cold_hi_tok();
            let hi = hi_mag::<CLS>(levels, 0, ls);
            tok = msac.hi_tok_c::<UPD>(&mut hi_cdf[hi_ctx(hi)]);
        }
        self.t.msac.copy_state(&m);
        (tok, k)
    }

    /// Updates cached `(dc_q, ac_q, qm_level)` for all 3 planes (spec 7.12.2).
    #[inline(always)]
    pub(crate) fn update_quant(&mut self) {
        use crate::decoder::av1::tables::quant::{ac_q, dc_q};
        let fh = self.f.fh;
        let bd = self.f.bd;
        let s = self.b.segment_id & 7;
        let qindex = if fh.seg.enabled {
            fh.get_qindex(false, s, self.t.current_qindex)
        } else {
            self.t.current_qindex
        };
        let q = &fh.quant;
        self.b.dc_q = [
            dc_q(qindex + q.delta_q_y_dc, bd) as u32,
            dc_q(qindex + q.delta_q_u_dc, bd) as u32,
            dc_q(qindex + q.delta_q_v_dc, bd) as u32,
        ];
        self.b.ac_q = [
            ac_q(qindex, bd) as u32,
            ac_q(qindex + q.delta_q_u_ac, bd) as u32,
            ac_q(qindex + q.delta_q_v_ac, bd) as u32,
        ];
        self.b.qm_level = if q.using_qmatrix && !fh.lossless_array[s] {
            [
                fh.seg_qm_level[0][s],
                fh.seg_qm_level[1][s],
                fh.seg_qm_level[2][s],
            ]
        } else {
            [15; 3]
        };
    }
}

/// Spec `get_scan(txSz)` given PlaneTxType.
#[inline]
pub const fn get_scan(tx_sz: TxSize, tx_type: TxType) -> &'static [u16] {
    use crate::decoder::av1::tables::scan::*;
    use TxSize::*;
    use TxType::*;
    if matches!(tx_sz, Tx16x64) {
        return &DEFAULT_SCAN_16X32;
    }
    if matches!(tx_sz, Tx64x16) {
        return &DEFAULT_SCAN_32X16;
    }
    if matches!(TX_SIZE_SQR_UP[tx_sz as usize], Tx64x64) {
        return &DEFAULT_SCAN_32X32;
    }
    if !matches!(tx_type, IDTX) {
        if matches!(tx_type, V_DCT | V_ADST | V_FLIPADST) {
            return match tx_sz {
                Tx4x4 => &MROW_SCAN_4X4,
                Tx4x8 => &MROW_SCAN_4X8,
                Tx8x4 => &MROW_SCAN_8X4,
                Tx8x8 => &MROW_SCAN_8X8,
                Tx8x16 => &MROW_SCAN_8X16,
                Tx16x8 => &MROW_SCAN_16X8,
                Tx16x16 => &MROW_SCAN_16X16,
                Tx4x16 => &MROW_SCAN_4X16,
                _ => &MROW_SCAN_16X4,
            };
        }
        if matches!(tx_type, H_DCT | H_ADST | H_FLIPADST) {
            return match tx_sz {
                Tx4x4 => &MCOL_SCAN_4X4,
                Tx4x8 => &MCOL_SCAN_4X8,
                Tx8x4 => &MCOL_SCAN_8X4,
                Tx8x8 => &MCOL_SCAN_8X8,
                Tx8x16 => &MCOL_SCAN_8X16,
                Tx16x8 => &MCOL_SCAN_16X8,
                Tx16x16 => &MCOL_SCAN_16X16,
                Tx4x16 => &MCOL_SCAN_4X16,
                _ => &MCOL_SCAN_16X4,
            };
        }
    }
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
}

#[inline(always)]
fn or_slice(s: &[u8]) -> u8 {
    let n = s.len();
    if n == 1 {
        return s[0];
    }
    let v = if let (Some(a), Some(b)) = (s.first_chunk::<8>(), s.last_chunk::<8>()) {
        let v = u64::from_ne_bytes(*a) | u64::from_ne_bytes(*b);
        (v as u32) | ((v >> 32) as u32)
    } else if let (Some(a), Some(b)) = (s.first_chunk::<4>(), s.last_chunk::<4>()) {
        u32::from_ne_bytes(*a) | u32::from_ne_bytes(*b)
    } else if n > 1 {
        return s[0] | s[1] | s[n - 1];
    } else {
        return 0;
    };
    let v = v | (v >> 16);
    (v | (v >> 8)) as u8
}

#[inline(always)]
fn sign_sum(dc: &[u8]) -> i32 {
    const ONES32: u32 = 0x0101_0101;
    const ONES64: u64 = 0x0101_0101_0101_0101;
    let n = dc.len();
    if n <= 2 {
        if n == 1 {
            let d0 = dc[0];
            (d0 >> 1) as i32 - (d0 & 1) as i32
        } else if n == 2 {
            let (d0, d1) = (dc[0], dc[1]);
            ((d0 >> 1) + (d1 >> 1)) as i32 - ((d0 & 1) + (d1 & 1)) as i32
        } else {
            0
        }
    } else if let Ok(a) = <[u8; 4]>::try_from(dc) {
        let v = u32::from_ne_bytes(a);
        let x = ((v >> 1) & ONES32) + (!v & ONES32);
        (x.wrapping_mul(ONES32) >> 24) as i32 - 4
    } else if let Ok(a) = <[u8; 8]>::try_from(dc) {
        let v = u64::from_ne_bytes(a);
        let x = ((v >> 1) & ONES64) + (!v & ONES64);
        (x.wrapping_mul(ONES64) >> 56) as i32 - 8
    } else if let Ok(a) = <[u8; 16]>::try_from(dc) {
        let v = u128::from_ne_bytes(a);
        let (v0, v1) = (v as u64, (v >> 64) as u64);
        let x0 = ((v0 >> 1) & ONES64) + (!v0 & ONES64);
        let x1 = ((v1 >> 1) & ONES64) + (!v1 & ONES64);
        ((x0 + x1).wrapping_mul(ONES64) >> 56) as i32 - 16
    } else {
        sign_sum_slow(dc)
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
    debug_assert!(matches!(len, 1 | 2 | 4 | 8 | 16));
    let a = &mut a[pos..pos + len];
    let b = &mut b[pos..pos + len];
    if len >= 8 {
        a[..8].fill(va);
        b[..8].fill(vb);
        a[len - 8..len].fill(va);
        b[len - 8..len].fill(vb);
    } else if len >= 4 {
        a[..4].fill(va);
        b[..4].fill(vb);
    } else if len >= 2 {
        a[..2].fill(va);
        b[..2].fill(vb);
    } else if len == 1 {
        a[0] = va;
        b[0] = vb;
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
        (pos as u32 + ((pos as u32 >> BWL) << 2)) as usize
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
    p: u32,
) -> usize {
    let pos = p & 0x3FF;
    let (mag, off) = match CLS {
        0 => {
            let l = &levels[idx..idx + 2 * ls + 1];
            let far = (64 + l[2] as u32) + (l[ls + 1] as u32 + l[2 * ls] as u32);
            let near = l[1] as u32 + l[ls] as u32;
            (far + near, (p >> 10) & 31)
        }
        1 => {
            let l = &levels[idx..idx + ls + 1];
            let far = (64 + l[2] as u32) + (l[3] as u32 + l[4] as u32 + l[ls] as u32);
            (far + l[1] as u32, 26 + (pos & ((1 << BWL) - 1)).min(2) * 5)
        }
        _ => {
            let l = &levels[idx..idx + 4 * ls + 1];
            let far = (64 + l[1] as u32) + (l[2 * ls] as u32 + l[3 * ls] as u32 + l[4 * ls] as u32);
            (far + l[ls] as u32, 26 + (pos >> BWL).min(2) * 5)
        }
    };
    // The low 6 bits of the sum are at most 30 when the (min 3) high
    // parts sum to less than 8, so they never disturb the rounding below.
    (off + (mag >> 7).min(4)) as usize
}

/// `coef_scan` table indexed by `[tx_class][tx_size]`. `tx_class` is
/// 0..=2; entry 3 duplicates entry 2 only so that the `tx_cls & 3` index
/// in `coef_scan` needs no bounds check.
const COEF_SCAN: [[&[u16]; TxSize::COUNT]; 4] = {
    use crate::decoder::av1::tables::scan::*;
    use TxSize::*;
    let mut table = [[&COEF_SCAN_4X4 as &[u16]; TxSize::COUNT]; 4];
    let mut i = 0;
    while i < TxSize::COUNT {
        let tx_sz = TxSize::ALL[i];
        table[0][i] = match tx_sz {
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
        };
        table[1][i] = get_scan(tx_sz, TxType::H_DCT);
        table[2][i] = get_scan(tx_sz, TxType::V_DCT);
        table[3][i] = table[2][i];
        i += 1;
    }
    table
};

/// Scan of the coefficient reader: `get_scan`, except that for the 2D
/// class every entry also carries the 2D `coeff_base` context offset of
/// its position in bits 10.. (`Coeff_Base_Ctx_Offset[txSz]`).
#[inline(always)]
fn coef_scan(tx_sz: TxSize, tx_cls: usize) -> &'static [u16] {
    COEF_SCAN[tx_cls & 3][tx_sz as usize]
}

/// `coeff_br` magnitude context from the raw neighbour sum:
/// `mag > 12 ? 6 : (mag + 1) >> 1`.
#[inline(always)]
fn hi_ctx(hi: u32) -> usize {
    (((hi & 63) + 1) >> 1).min(6) as usize
}

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
                let scan = coef_scan(tx_sz, cls);
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

    /// Spec 7.12.3 dequantization of level `q` with quantizer `q2`:
    /// `((q * q2) & 0xFFFFFF) >> dq_shift`, clipped to `cf_max` (positive)
    /// or `cf_max + 1` (negative), then signed.
    fn spec_dq(q: u32, q2: u32, dq_shift: u32, cf_max: u64, neg: bool) -> i64 {
        let d = ((q as u64 * q2 as u64) & 0xFFFFFF) >> dq_shift;
        let d = d.min(cf_max + neg as u64) as i64;
        if neg {
            -d
        } else {
            d
        }
    }

    /// The unclamped `q < 15` dequantization fast paths of `dc_coef` and
    /// `dequant_ac_qm` (and the clamped `bool_store_neg_qm` path) match the
    /// spec formula at 8, 10 and 12 bits for every quantizer.
    #[test]
    fn qm_dequant_fast_paths() {
        use crate::decoder::av1::tables::qm::{quantizer_matrix_1024, QM_MAX_DC_WEIGHT};
        use crate::decoder::av1::tables::quant::{AC_QLOOKUP, DC_QLOOKUP};
        // Every DC weight is within the bound used by the DC fast path.
        for level in 0..15 {
            for ptype in 0..2 {
                for &tx_sz in TxSize::ALL.iter() {
                    assert!(quantizer_matrix_1024(level, ptype, tx_sz)[0] <= QM_MAX_DC_WEIGHT);
                }
            }
        }
        let max_shift = *TX_DQ_SHIFT.iter().max().unwrap() as u32;
        assert_eq!(max_shift, 2);
        for (bd_idx, bd) in [8u32, 10, 12].into_iter().enumerate() {
            let cf_max = (1u64 << (7 + bd)) - 1;
            let cf_max32 = cf_max as u32;
            for dq_shift in 0..=max_shift {
                // DC (`dc_coef`): never clamped, with or without QM.
                for &dc_q in DC_QLOOKUP[bd_idx].iter() {
                    let dc_q = dc_q as u32;
                    let q2s = (0..=QM_MAX_DC_WEIGHT as u32)
                        .map(|w| (dc_q * w + 16) >> 5)
                        .chain(std::iter::once(dc_q));
                    for q2 in q2s {
                        for q in 1..15u32 {
                            let d = ((q * q2) >> dq_shift) as i32 as i64;
                            assert_eq!(d, spec_dq(q, q2, dq_shift, cf_max, false), "{bd} {q2}");
                            assert_eq!(-d, spec_dq(q, q2, dq_shift, cf_max, true), "{bd} {q2}");
                        }
                    }
                }
                // AC with QM (`dequant_ac_qm`).
                for &ac_q in AC_QLOOKUP[bd_idx].iter() {
                    let ac_q = ac_q as u32;
                    let max_d = (14 * ((ac_q * 255 + 16) >> 5)) >> dq_shift;
                    let fast = max_d <= cf_max32;
                    for w in 0..=255u32 {
                        let q2 = (ac_q * w + 16) >> 5;
                        for q in 1..15u32 {
                            let d = (q * q2) >> dq_shift;
                            let (pos, neg) = if fast {
                                (d as i32 as i64, -(d as i32) as i64)
                            } else {
                                (d.min(cf_max32) as i64, -(d.min(cf_max32 + 1) as i64))
                            };
                            assert_eq!(pos, spec_dq(q, q2, dq_shift, cf_max, false), "{bd} {q2}");
                            assert_eq!(neg, spec_dq(q, q2, dq_shift, cf_max, true), "{bd} {q2}");
                        }
                    }
                }
            }
        }
    }
}
