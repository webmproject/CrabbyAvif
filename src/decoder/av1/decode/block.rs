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

/// Packed `(TX_HEIGHT_LOG2[adj] << 4) | TX_WIDTH_LOG2[adj]` for `adj = ADJUSTED_TX_SIZE[ts]`.
const ADJ_BWL_HLOG2: [u8; TxSize::COUNT] = {
    let mut t = [0u8; TxSize::COUNT];
    let mut i = 0;
    while i < TxSize::COUNT {
        let adj = ADJUSTED_TX_SIZE[i] as usize;
        let bwl = TX_WIDTH_LOG2[adj];
        let hl = TX_HEIGHT_LOG2[adj];
        assert!(bwl >= 2 && bwl <= 5 && hl >= 2 && hl <= 5);
        assert!((1usize << hl) == TX_HEIGHT[adj] as usize);
        t[i] = (hl << 4) | bwl;
        i += 1;
    }
    t
};

#[derive(Clone, Copy)]
#[repr(C, align(8))]
struct TxMeta {
    tx_sz: TxSize,
    w4_log2: u8,
    h4_log2: u8,
    tx_sz_ctx: u8,
    eob_multisize: u8,
    dq_shift: u8,
    /// `tx_mode` for luma intra when `seg_qindex_nz && !reduced_tx_set`.
    intra_mode_full: u8,
    /// `tx_mode` for luma intra when `seg_qindex_nz && reduced_tx_set`.
    intra_mode_red: u8,
}

const TX_META: [TxMeta; TxSize::COUNT] = {
    let mut t = [TxMeta {
        tx_sz: TxSize::Tx4x4,
        w4_log2: 0,
        h4_log2: 0,
        tx_sz_ctx: 0,
        eob_multisize: 0,
        dq_shift: 0,
        intra_mode_full: 0,
        intra_mode_red: 0,
    }; TxSize::COUNT];
    let mut i = 0;
    while i < TxSize::COUNT {
        let sqr = TX_SIZE_SQR[i] as u8;
        let sqr_up = TX_SIZE_SQR_UP[i] as u8;
        let (full, red) = if sqr_up >= TxSize::Tx32x32 as u8 {
            (0, 0)
        } else if sqr == TxSize::Tx16x16 as u8 {
            (18 + 2, 18 + 2)
        } else {
            (16 + sqr, 18 + sqr)
        };
        t[i] = TxMeta {
            tx_sz: TxSize::ALL[i],
            w4_log2: TX_WIDTH_LOG2[i] - 2,
            h4_log2: TX_HEIGHT_LOG2[i] - 2,
            tx_sz_ctx: TX_SZ_CTX[i],
            eob_multisize: EOB_MULTISIZE[i],
            dq_shift: TX_DQ_SHIFT[i],
            intra_mode_full: full,
            intra_mode_red: red,
        };
        i += 1;
    }
    t
};

/// Per-(coding block, plane type) transform constants, packed into 8 bytes
/// so it fits and passes in a single 64-bit GPR.
#[derive(Clone, Copy)]
#[repr(C, align(8))]
struct TxPlan {
    tx_sz: TxSize,
    w4_log2: u8,
    h4_log2: u8,
    tx_sz_ctx: u8,
    eob_multisize: u8,
    dq_shift: u8,
    /// `0..16`: fixed `TxType`; `16..18`: luma intra set1 (`sqr = tx_mode & 1`);
    /// `18..21`: luma intra set2 (`sqr = tx_mode - 18`);
    /// `24..28`: luma inter (`set = tx_mode - 24`);
    /// `29..32`: chroma inter (`set = tx_mode - 28`).
    tx_mode: u8,
    /// Low 4 bits: luma `intra_dir` (`0..=12`); high 4 bits: `zero_ctx_base` (`0, 1, 7, 10`).
    dir_and_zbase: u8,
}

const _: () = assert!(std::mem::size_of::<TxPlan>() == 8);

impl TxPlan {
    #[inline(always)]
    fn luma(f: &FrameCtx<'_>, b: &Block, tx_sz: TxSize, zero_ctx_base: u8) -> Self {
        let m = TX_META[tx_sz as usize];
        let seg_q_nz = f.fh.seg_qindex_nz[b.segment_id & 7];
        let (tx_mode, intra_dir) = if !b.is_inter {
            let mode = if !seg_q_nz {
                0
            } else if f.fh.reduced_tx_set {
                m.intra_mode_red
            } else {
                m.intra_mode_full
            };
            let dir = if b.use_filter_intra {
                FILTER_INTRA_MODE_TO_INTRA_DIR[(b.filter_intra_mode as usize).min(4)] as u8
            } else {
                (b.y_mode as u8).min(12)
            };
            (mode, dir)
        } else {
            let sqr = TX_SIZE_SQR[tx_sz as usize];
            let sqr_up = TX_SIZE_SQR_UP[tx_sz as usize];
            let set = if sqr_up > TxSize::Tx32x32 {
                TX_SET_DCTONLY
            } else if f.fh.reduced_tx_set || sqr_up == TxSize::Tx32x32 {
                TX_SET_INTER_3
            } else if sqr == TxSize::Tx16x16 {
                TX_SET_INTER_2
            } else {
                TX_SET_INTER_1
            };
            let mode = if set == 0 || !seg_q_nz { 24 } else { (24 + set) as u8 };
            (mode, 0)
        };
        Self {
            tx_sz: m.tx_sz,
            w4_log2: m.w4_log2,
            h4_log2: m.h4_log2,
            tx_sz_ctx: m.tx_sz_ctx,
            eob_multisize: m.eob_multisize,
            dq_shift: m.dq_shift,
            tx_mode,
            dir_and_zbase: intra_dir | (zero_ctx_base << 4),
        }
    }

    #[inline(always)]
    fn chroma(f: &FrameCtx<'_>, b: &Block, tx_sz: TxSize, zero_ctx_base: u8) -> Self {
        let m = TX_META[tx_sz as usize];
        let sqr_up = TX_SIZE_SQR_UP[tx_sz as usize];
        let tx_mode = if !b.is_inter {
            if b.lossless || sqr_up >= TxSize::Tx32x32 {
                TxType::DCT_DCT as u8
            } else {
                MODE_TO_TXFM[(b.uv_mode as usize).min(13)] as u8
            }
        } else if b.lossless || sqr_up > TxSize::Tx32x32 {
            TxType::DCT_DCT as u8
        } else {
            let sqr = TX_SIZE_SQR[tx_sz as usize];
            let set = if f.fh.reduced_tx_set || sqr_up == TxSize::Tx32x32 {
                TX_SET_INTER_3
            } else if sqr == TxSize::Tx16x16 {
                TX_SET_INTER_2
            } else {
                TX_SET_INTER_1
            };
            (28 + set) as u8
        };
        Self {
            tx_sz: m.tx_sz,
            w4_log2: m.w4_log2,
            h4_log2: m.h4_log2,
            tx_sz_ctx: m.tx_sz_ctx,
            eob_multisize: m.eob_multisize,
            dq_shift: m.dq_shift,
            tx_mode,
            dir_and_zbase: zero_ctx_base << 4,
        }
    }
}

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
    ((0x6660_0000u32 >> ((t as u32) << 1)) & 3) as usize
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

#[cold]
#[inline(never)]
fn reset_scratch_on_err(sc: &mut Scratch) {
    sc.coeffs.fill(0);
    sc.levels.fill(0);
}

impl Default for Scratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Decodes superblocks of one tile.
pub struct SbDecoder<'a, 'b, 's, T: Pixel> {
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
    pub stage: super::stage::DecoderStage<'s>,
    pub filter_type: [bool; 2],
}

macro_rules! cdf {
    ($self:ident) => {
        $self.t.cdf
    };
}

const TXL_PREDICT: u8 = 1 << 0;
const TXL_PALETTE: u8 = 1 << 1;
const TXL_CFL: u8 = 1 << 2;
const TXL_AVAIL_L: u8 = 1 << 3;
const TXL_AVAIL_U: u8 = 1 << 4;
const TXL_COEFS: u8 = 1 << 5;
const TXL_RECON: u8 = 1 << 6;

/// Bit-packed flags of one plane type's transform block loop in `residual`,
/// kept in a single byte register across inner prediction / reconstruction calls.
#[derive(Clone, Copy)]
#[repr(transparent)]
struct TxLoop(u8);

impl TxLoop {
    #[inline(always)]
    fn new(
        predict: bool,
        palette: bool,
        cfl: bool,
        avail_l: bool,
        avail_u: bool,
        coefs: bool,
        recon: bool,
    ) -> Self {
        Self(
            (predict as u8 * TXL_PREDICT)
                | (palette as u8 * TXL_PALETTE)
                | (cfl as u8 * TXL_CFL)
                | (avail_l as u8 * TXL_AVAIL_L)
                | (avail_u as u8 * TXL_AVAIL_U)
                | (coefs as u8 * TXL_COEFS)
                | (recon as u8 * TXL_RECON),
        )
    }

    #[inline(always)]
    fn predict(self) -> bool {
        (self.0 & TXL_PREDICT) != 0
    }
    #[inline(always)]
    fn palette(self) -> bool {
        (self.0 & TXL_PALETTE) != 0
    }
    #[inline(always)]
    fn cfl(self) -> bool {
        (self.0 & TXL_CFL) != 0
    }
    #[inline(always)]
    fn avail_l(self) -> bool {
        (self.0 & TXL_AVAIL_L) != 0
    }
    #[inline(always)]
    fn avail_u(self) -> bool {
        (self.0 & TXL_AVAIL_U) != 0
    }
    #[inline(always)]
    fn coefs(self) -> bool {
        (self.0 & TXL_COEFS) != 0
    }
    #[inline(always)]
    fn recon(self) -> bool {
        (self.0 & TXL_RECON) != 0
    }
}

#[inline(always)]
fn set_block_decoded(bd: &mut [BlockDecodedRow; 16], bx: usize, w4_log2: u8, h4_log2: u8) {
    let mask = ((1u64 << (1usize << (w4_log2 & 7))) - 1) << bx;
    match h4_log2 {
        0 => bd[0].0 |= mask,
        1 => {
            bd[0].0 |= mask;
            bd[1].0 |= mask;
        }
        2 => {
            for r in &mut bd[..4] {
                r.0 |= mask;
            }
        }
        3 => {
            for r in &mut bd[..8] {
                r.0 |= mask;
            }
        }
        _ => {
            for r in bd {
                r.0 |= mask;
            }
        }
    }
}

#[cold]
#[inline(never)]
fn record_parse_tx_qm(
    sc: &mut Scratch,
    plane_tx_type: TxType,
    log: &mut super::stage::LogChunk,
    eob: usize,
    has_dc: bool,
    n_ac: usize,
) {
    let count = has_dc as usize + n_ac;
    log.tx_entries.reserve(count);
    if has_dc {
        let val = sc.coeffs[0];
        sc.coeffs[0] = 0;
        log.tx_entries.push((0, val));
    }
    let coeffs = &mut sc.coeffs;
    log.tx_entries.extend(sc.nz[..n_ac].iter().rev().map(|&e| {
        let rc = (e & 0x3FF) as usize;
        let val = coeffs[rc];
        coeffs[rc] = 0;
        (rc as u16, val)
    }));
    log.tx_headers.push(super::stage::TxRecHeader {
        eob: eob as u16,
        tx_type: plane_tx_type,
        count: count as u16,
    });
}

impl<'a, 'b, 's, T: Pixel> SbDecoder<'a, 'b, 's, T> {
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

    #[inline(always)]
    pub(crate) fn residual(&mut self) -> BlockResult {
        match self.stage {
            super::stage::DecoderStage::Fused => self.residual_fused(),
            super::stage::DecoderStage::Parse(..) => {
                if self.b.skip {
                    Ok(())
                } else {
                    self.residual_parse()
                }
            }
            super::stage::DecoderStage::Recon(..) => self.residual_recon(),
        }
    }

    #[inline(always)]
    fn residual_fused(&mut self) -> BlockResult {
        let f = self.f;
        let b = &self.b;
        if b.skip && !self.recon && !f.fh.allow_intrabc {
            return Ok(());
        }
        let bw = b.bw4;
        let bh = b.bh4;
        let width_chunks = 1.max(bw >> 4);
        let height_chunks = 1.max(bh >> 4);
        let y_tx = if b.lossless { TxSize::Tx4x4 } else { b.tx_size };
        let ym = TX_META[y_tx as usize];
        let y_w4_log2 = ym.w4_log2.min(4);
        let y_h4_log2 = ym.h4_log2.min(4);
        let y_step_x = 1usize << y_w4_log2;
        let y_step_y = 1usize << y_h4_log2;
        let y_zero_base = !(bw == y_step_x && bh == y_step_y) as u8;
        let y_plan = TxPlan::luma(f, b, y_tx, y_zero_base);
        let y_n4w = bw.min(16);
        let y_n4h = bh.min(16);
        let y_base_x = b.mi_col * MI_SIZE;
        let y_base_y = b.mi_row * MI_SIZE;
        let tree = b.is_inter && !b.lossless;
        let has_chroma = b.has_chroma;
        let cfl = has_chroma && b.uv_mode == IntraMode::UV_CFL_PRED;
        let pred = !b.is_inter && self.recon;
        let lc = TxLoop::new(
            pred && b.palette_size_y == 0,
            pred && b.palette_size_y > 0,
            false,
            b.avail_l,
            b.avail_u,
            !b.skip,
            self.recon,
        );

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
                            self.transform_block_fused::<false>(
                                lc,
                                0,
                                y_base_x,
                                y_base_y,
                                y_plan,
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
                        let f = self.f;
                        self.b.max_luma_w = y_base_x
                            + ((f.max_x[0] - y_base_x + sw - 1) & !(sw - 1)).min(y_n4w * 4);
                        self.b.max_luma_h = y_base_y
                            + ((f.max_y[0] - y_base_y + sh - 1) & !(sh - 1)).min(y_n4h * 4);
                    }
                }
                if has_chroma {
                    let f = self.f;
                    let b = &self.b;
                    let (sx, sy) = (f.ss_x, f.ss_y);
                    let mi_size_chunk = if (bw | bh) >= 32 { BlockSize::B64x64 } else { b.bsize };
                    let uv_tx =
                        if b.lossless { TxSize::Tx4x4 } else { f.uv_tx_size[b.bsize as usize] };
                    let uvm = TX_META[uv_tx as usize];
                    let uv_step_x = 1usize << uvm.w4_log2.min(4);
                    let uv_step_y = 1usize << uvm.h4_log2.min(4);
                    let plane_sz = f.uv_residual_size[mi_size_chunk as usize];
                    let uv_bsize = f.uv_residual_size[b.bsize as usize];
                    let uv_zero_base =
                        if (bw4(uv_bsize) != uv_step_x) | (bh4(uv_bsize) != uv_step_y) {
                            10
                        } else {
                            7
                        };
                    let uv_plan = TxPlan::chroma(f, b, uv_tx, uv_zero_base);
                    let uv_n4w = bw4(plane_sz);
                    let uv_n4h = bh4(plane_sz);
                    let uv_base_x = (b.mi_col >> sx) * MI_SIZE;
                    let uv_base_y = (b.mi_row >> sy) * MI_SIZE;
                    let cc = TxLoop::new(
                        pred && b.palette_size_uv == 0,
                        pred && b.palette_size_uv > 0,
                        cfl,
                        b.avail_l_chroma,
                        b.avail_u_chroma,
                        lc.coefs(),
                        lc.recon(),
                    );
                    let ox = ox0 >> sx;
                    let oy = oy0 >> sy;
                    for plane in 1..3 {
                        let mut y = 0;
                        while y < uv_n4h {
                            let mut x = 0;
                            while x < uv_n4w {
                                self.transform_block_fused::<true>(
                                    cc,
                                    plane,
                                    uv_base_x,
                                    uv_base_y,
                                    uv_plan,
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

    #[inline(never)]
    fn residual_parse(&mut self) -> BlockResult {
        let f = self.f;
        let b = &self.b;
        debug_assert!(!self.recon && !f.fh.allow_intrabc && !b.is_inter && !b.skip);
        let bw = b.bw4;
        let bh = b.bh4;
        let width_chunks = 1.max(bw >> 4);
        let height_chunks = 1.max(bh >> 4);
        let y_tx = if b.lossless { TxSize::Tx4x4 } else { b.tx_size };
        let ym = TX_META[y_tx as usize];
        let y_w4_log2 = ym.w4_log2.min(4);
        let y_h4_log2 = ym.h4_log2.min(4);
        let y_step_x = 1usize << y_w4_log2;
        let y_step_y = 1usize << y_h4_log2;
        let y_zero_base = !(bw == y_step_x && bh == y_step_y) as u8;
        let y_plan = TxPlan::luma(f, b, y_tx, y_zero_base);
        let y_n4w = bw.min(16);
        let y_n4h = bh.min(16);
        let y_base_x = b.mi_col * MI_SIZE;
        let y_base_y = b.mi_row * MI_SIZE;
        let y_max_x = f.max_x[0];
        let y_max_y = f.max_y[0];
        let y_sb_mask = f.sb_mask_y[0] & 31;
        let has_chroma = b.has_chroma;

        // At most `32 * 32 * 3 = 3072` 4x4 transform blocks in a 128x128 coding block.
        if let super::stage::DecoderStage::Parse(log) = &mut self.stage {
            log.tx_headers.reserve(3072);
        }

        for chunk_y in 0..height_chunks {
            for chunk_x in 0..width_chunks {
                let ox0 = chunk_x << 4;
                let oy0 = chunk_y << 4;
                let mut y = 0;
                while y < y_n4h {
                    let start_y = y_base_y + 4 * (y + oy0);
                    if start_y < y_max_y {
                        let sub_y = (start_y >> 2) & y_sb_mask;
                        let mut x = 0;
                        while x < y_n4w {
                            let start_x = y_base_x + 4 * (x + ox0);
                            if start_x < y_max_x
                                && self.coeffs::<false, true>(0, start_x, start_y, sub_y, y_plan)
                                    == u32::MAX
                            {
                                return Err("coef");
                            }
                            x += y_step_x;
                        }
                    }
                    y += y_step_y;
                }
                if has_chroma {
                    let f = self.f;
                    let b = &self.b;
                    let (sx, sy) = (f.ss_x, f.ss_y);
                    let mi_size_chunk = if (bw | bh) >= 32 { BlockSize::B64x64 } else { b.bsize };
                    let uv_tx =
                        if b.lossless { TxSize::Tx4x4 } else { f.uv_tx_size[b.bsize as usize] };
                    let uvm = TX_META[uv_tx as usize];
                    let uv_step_x = 1usize << uvm.w4_log2.min(4);
                    let uv_step_y = 1usize << uvm.h4_log2.min(4);
                    let plane_sz = f.uv_residual_size[mi_size_chunk as usize];
                    let uv_bsize = f.uv_residual_size[b.bsize as usize];
                    let uv_zero_base =
                        if (bw4(uv_bsize) != uv_step_x) | (bh4(uv_bsize) != uv_step_y) {
                            10
                        } else {
                            7
                        };
                    let uv_plan = TxPlan::chroma(f, b, uv_tx, uv_zero_base);
                    let uv_n4w = bw4(plane_sz);
                    let uv_n4h = bh4(plane_sz);
                    let uv_base_x = (b.mi_col >> sx) * MI_SIZE;
                    let uv_base_y = (b.mi_row >> sy) * MI_SIZE;
                    let uv_max_x = f.max_x[1];
                    let uv_max_y = f.max_y[1];
                    let uv_sb_mask = f.sb_mask_y[1] & 31;
                    let ox = ox0 >> sx;
                    let oy = oy0 >> sy;
                    for plane in 1..3 {
                        let mut y = 0;
                        while y < uv_n4h {
                            let start_y = uv_base_y + 4 * (y + oy);
                            if start_y < uv_max_y {
                                let sub_y = (start_y >> 2) & uv_sb_mask;
                                let mut x = 0;
                                while x < uv_n4w {
                                    let start_x = uv_base_x + 4 * (x + ox);
                                    if start_x < uv_max_x
                                        && self.coeffs::<true, true>(
                                            plane, start_x, start_y, sub_y, uv_plan,
                                        ) == u32::MAX
                                    {
                                        return Err("coef");
                                    }
                                    x += uv_step_x;
                                }
                            }
                            y += uv_step_y;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[inline(never)]
    fn residual_recon(&mut self) -> BlockResult {
        let b = &self.b;
        let bw = b.bw4;
        let bh = b.bh4;
        let width_chunks = 1.max(bw >> 4);
        let height_chunks = 1.max(bh >> 4);
        let y_tx = if b.lossless { TxSize::Tx4x4 } else { b.tx_size };
        let ym = TX_META[y_tx as usize];
        let y_w4_log2 = ym.w4_log2.min(4);
        let y_h4_log2 = ym.h4_log2.min(4);
        let y_step_x = 1usize << y_w4_log2;
        let y_step_y = 1usize << y_h4_log2;
        let y_n4w = bw.min(16);
        let y_n4h = bh.min(16);
        let y_base_x = b.mi_col * MI_SIZE;
        let y_base_y = b.mi_row * MI_SIZE;
        let has_chroma = b.has_chroma;
        let cfl = has_chroma && b.uv_mode == IntraMode::UV_CFL_PRED;
        let pred = !b.is_inter && self.recon;
        let lc = TxLoop::new(
            pred && b.palette_size_y == 0,
            pred && b.palette_size_y > 0,
            false,
            b.avail_l,
            b.avail_u,
            !b.skip,
            self.recon,
        );

        let (mut hdrs, mut ents): (&'s [super::stage::TxRecHeader], &'s [(u16, i32)]) =
            if lc.coefs() {
                let super::stage::DecoderStage::Recon(chunk, cursor) = &self.stage else {
                    return Err("invalid log");
                };
                let chunk = *chunk;
                let (Some(h), Some(e)) = (
                    chunk.tx_headers.get(cursor.tx_header_idx..),
                    chunk.tx_entries.get(cursor.tx_entry_idx..),
                ) else {
                    return Err("invalid log");
                };
                (h, e)
            } else {
                (&[], &[])
            };

        for chunk_y in 0..height_chunks {
            for chunk_x in 0..width_chunks {
                let ox0 = chunk_x << 4;
                let oy0 = chunk_y << 4;
                let mut y = 0;
                while y < y_n4h {
                    let mut x = 0;
                    while x < y_n4w {
                        self.transform_block_recon::<false>(
                            lc,
                            0,
                            y_base_x,
                            y_base_y,
                            y_tx,
                            y_w4_log2,
                            y_h4_log2,
                            x + ox0,
                            y + oy0,
                            &mut hdrs,
                            &mut ents,
                        )?;
                        x += y_step_x;
                    }
                    y += y_step_y;
                }
                if cfl {
                    let sw = y_step_x * 4;
                    let sh = y_step_y * 4;
                    let f = self.f;
                    self.b.max_luma_w =
                        y_base_x + ((f.max_x[0] - y_base_x + sw - 1) & !(sw - 1)).min(y_n4w * 4);
                    self.b.max_luma_h =
                        y_base_y + ((f.max_y[0] - y_base_y + sh - 1) & !(sh - 1)).min(y_n4h * 4);
                }
                if has_chroma {
                    let f = self.f;
                    let b = &self.b;
                    let (sx, sy) = (f.ss_x, f.ss_y);
                    let mi_size_chunk = if (bw | bh) >= 32 { BlockSize::B64x64 } else { b.bsize };
                    let uv_tx =
                        if b.lossless { TxSize::Tx4x4 } else { f.uv_tx_size[b.bsize as usize] };
                    let uvm = TX_META[uv_tx as usize];
                    let uv_w4_log2 = uvm.w4_log2.min(4);
                    let uv_h4_log2 = uvm.h4_log2.min(4);
                    let uv_step_x = 1usize << uv_w4_log2;
                    let uv_step_y = 1usize << uv_h4_log2;
                    let plane_sz = f.uv_residual_size[mi_size_chunk as usize];
                    let uv_n4w = bw4(plane_sz);
                    let uv_n4h = bh4(plane_sz);
                    let uv_base_x = (b.mi_col >> sx) * MI_SIZE;
                    let uv_base_y = (b.mi_row >> sy) * MI_SIZE;
                    let cc = TxLoop::new(
                        pred && b.palette_size_uv == 0,
                        pred && b.palette_size_uv > 0,
                        cfl,
                        b.avail_l_chroma,
                        b.avail_u_chroma,
                        lc.coefs(),
                        lc.recon(),
                    );
                    let ox = ox0 >> sx;
                    let oy = oy0 >> sy;
                    for plane in 1..3 {
                        let mut y = 0;
                        while y < uv_n4h {
                            let mut x = 0;
                            while x < uv_n4w {
                                self.transform_block_recon::<true>(
                                    cc,
                                    plane,
                                    uv_base_x,
                                    uv_base_y,
                                    uv_tx,
                                    uv_w4_log2,
                                    uv_h4_log2,
                                    x + ox,
                                    y + oy,
                                    &mut hdrs,
                                    &mut ents,
                                )?;
                                x += uv_step_x;
                            }
                            y += uv_step_y;
                        }
                    }
                }
            }
        }
        if lc.coefs() {
            if let super::stage::DecoderStage::Recon(chunk, cursor) = &mut self.stage {
                cursor.tx_header_idx = chunk.tx_headers.len() - hdrs.len();
                cursor.tx_entry_idx = chunk.tx_entries.len() - ents.len();
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
            let step_x = w >> 2;
            let step_y = h >> 2;
            // Inter (intra block copy) blocks: no prediction per
            // transform block.
            let zero_base = !(self.b.bw4 == step_x && self.b.bh4 == step_y) as u8;
            let plan = TxPlan::luma(f, &self.b, tx_sz, zero_base);
            let c = TxLoop::new(false, false, false, false, false, !self.b.skip, self.recon);
            self.transform_block_fused::<false>(c, 0, start_x, start_y, plan, 0, 0)
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
    fn transform_block_fused<const CHROMA: bool>(
        &mut self,
        c: TxLoop,
        plane: usize,
        base_x: usize,
        base_y: usize,
        plan: TxPlan,
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
        let w4_log2 = plan.w4_log2.min(4);
        let h4_log2 = plan.h4_log2.min(4);
        let step_x = 1usize << w4_log2;
        let step_y = 1usize << h4_log2;
        let tx_sz = plan.tx_sz;
        if c.predict() {
            let have_left = c.avail_l() || x > 0;
            let have_above = c.avail_u() || y > 0;
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
            if CHROMA && c.cfl() {
                self.predict_cfl(plane, start_x, start_y, tx_sz);
            }
        } else if c.palette() {
            self.predict_palette(plane, start_x, start_y, x, y, tx_sz);
        }
        if c.coefs() {
            let eob = self.coeffs::<CHROMA, false>(plane, start_x, start_y, sub_y, plan);
            if eob > 0 {
                if eob == u32::MAX {
                    return Err("coef");
                }
                if c.recon() {
                    self.reconstruct(plane, start_x, start_y, tx_sz, eob as usize);
                } else {
                    self.clear_coeffs(tx_sz);
                }
            }
        }
        let bd_plane = if !CHROMA {
            &mut self.sc.block_decoded[0]
        } else {
            &mut self.sc.block_decoded[plane]
        };
        if let Some(bd) = bd_plane[by..].first_chunk_mut::<16>() {
            set_block_decoded(bd, bx, w4_log2, h4_log2);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn transform_block_recon<const CHROMA: bool>(
        &mut self,
        c: TxLoop,
        plane: usize,
        base_x: usize,
        base_y: usize,
        tx_sz: TxSize,
        w4_log2: u8,
        h4_log2: u8,
        x: usize,
        y: usize,
        hdrs: &mut &'s [super::stage::TxRecHeader],
        ents: &mut &'s [(u16, i32)],
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
        let step_x = 1usize << w4_log2;
        let step_y = 1usize << h4_log2;
        if c.predict() {
            let have_left = c.avail_l() || x > 0;
            let have_above = c.avail_u() || y > 0;
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
            if CHROMA && c.cfl() {
                self.predict_cfl(plane, start_x, start_y, tx_sz);
            }
        } else if c.palette() {
            self.predict_palette(plane, start_x, start_y, x, y, tx_sz);
        }
        if c.coefs() {
            let Some((&th, rest_h)) = hdrs.split_first() else {
                return Err("invalid log");
            };
            *hdrs = rest_h;
            let eob = th.eob as usize;
            if eob > 0 {
                self.plane_tx_type = th.tx_type;
                let Some((entries, rest_e)) = ents.split_at_checked(th.count as usize) else {
                    return Err("invalid log");
                };
                *ents = rest_e;
                let coeffs = &mut self.sc.coeffs;
                for &(rc, val) in entries {
                    coeffs[(rc & 0x3FF) as usize] = val;
                }
                self.reconstruct(plane, start_x, start_y, tx_sz, eob);
            }
        }
        let bd_plane = if !CHROMA {
            &mut self.sc.block_decoded[0]
        } else {
            &mut self.sc.block_decoded[plane]
        };
        if let Some(bd) = bd_plane[by..].first_chunk_mut::<16>() {
            set_block_decoded(bd, bx, w4_log2, h4_log2);
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Transform type
    // ------------------------------------------------------------------

    #[cold]
    #[inline(never)]
    fn resolve_inter_tx_type(&mut self, plan: TxPlan, ax0: usize, y4: usize) -> TxType {
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
        if plan.tx_mode < 28 {
            let set = (plan.tx_mode - 24) as usize;
            let tx_type = if set > 0 {
                let sqr = (TX_SIZE_SQR[plan.tx_sz as usize] as usize).min(3);
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
            let x4 = ax0 + self.s.above.col0;
            set_tx_types(
                &mut self.sc.tx_types,
                x4 - self.b.mi_col,
                y4 - self.b.mi_row,
                plan.w4_log2,
                plan.h4_log2,
                tx_type,
            );
            tx_type
        } else {
            let set = ((plan.tx_mode - 28) as usize).min(3);
            let block_x = ax0 + (self.s.above.col0 >> self.f.ss_x);
            let x4 = self.b.mi_col.max(block_x << self.f.ss_x);
            let y4_luma = self.b.mi_row.max(y4 << self.f.ss_y);
            let t = self.sc.tx_types[(y4_luma - self.b.mi_row) & 31][(x4 - self.b.mi_col) & 31];
            if ((INTER_MASK >> ((set << 4) | t as usize)) & 1) == 0 {
                TxType::DCT_DCT
            } else {
                t
            }
        }
    }

    // ------------------------------------------------------------------
    // Coefficients
    // ------------------------------------------------------------------

    #[inline(always)]
    fn coeffs<const CHROMA: bool, const PARSE: bool>(
        &mut self,
        plane: usize,
        start_x: usize,
        start_y: usize,
        ly: usize,
        plan: TxPlan,
    ) -> u32 {
        let f = self.f;
        let x4 = start_x >> 2;
        let y4 = start_y >> 2;
        let tx_sz_ctx = (plan.tx_sz_ctx as usize).min(4);
        let w4 = 1usize << plan.w4_log2.min(4);
        let h4 = 1usize << plan.h4_log2.min(4);
        let zero_ctx_base = (plan.dir_and_zbase >> 4) as usize;
        let (sx, sy) = if !CHROMA { (0, 0) } else { (f.ss_x, f.ss_y) };
        let ax0 = x4 - (self.s.above.col0 >> sx);

        let all_zero_ctx = if !CHROMA && zero_ctx_base == 0 {
            0
        } else if !CHROMA {
            let kx = if w4 == 1 { 1 } else { w4.min(f.mi_cols - x4) };
            let ky = if h4 == 1 { 1 } else { h4.min(f.mi_rows - y4) };
            let tc = level_cat(&self.s.above.level[0], ax0, kx);
            let lc = level_cat(&self.s.left.level[0], ly, ky);
            ((0x653_0542_0321u64 >> ((tc << 4) | (lc << 2))) & 7) as usize
        } else {
            let kx = if w4 == 1 { 1 } else { w4.min((f.mi_cols >> sx) - x4) };
            let ky = if h4 == 1 { 1 } else { h4.min((f.mi_rows >> sy) - y4) };
            let above = any_nz2(&self.s.above.level[plane], &self.s.above.dc[plane], ax0, kx);
            let left = any_nz2(&self.s.left.level[plane], &self.s.left.dc[plane], ly, ky);
            zero_ctx_base.min(10) + above as usize + left as usize
        };

        let all_zero = self
            .t
            .msac
            .bool_adapt(&mut self.t.cdf.txb_skip[tx_sz_ctx][all_zero_ctx]);
        if all_zero {
            if !PARSE && !CHROMA && plan.tx_mode >= 24 {
                set_tx_types(
                    &mut self.sc.tx_types,
                    x4 - self.b.mi_col,
                    y4 - self.b.mi_row,
                    plan.w4_log2,
                    plan.h4_log2,
                    TxType::DCT_DCT,
                );
            }
            set_ctx2_log2(
                &mut self.s.above.level[plane],
                &mut self.s.above.dc[plane],
                ax0,
                plan.w4_log2,
                0,
                0,
            );
            set_ctx2_log2(
                &mut self.s.left.level[plane],
                &mut self.s.left.dc[plane],
                ly,
                plan.h4_log2,
                0,
                0,
            );
            if PARSE {
                if let super::stage::DecoderStage::Parse(log) = &mut self.stage {
                    log.tx_headers.push(super::stage::TxRecHeader {
                        eob: 0,
                        tx_type: TxType::DCT_DCT,
                        count: 0,
                    });
                }
            }
            return 0;
        }
        self.coeffs_nz::<PARSE>(plane, plan, ax0, ly, y4)
    }

    #[inline(always)]
    fn update_coeff_ctx(
        &mut self,
        plane: usize,
        plan: TxPlan,
        ax0: usize,
        ly: usize,
        cul_level: u8,
        dc_category: u8,
    ) {
        if plan.tx_sz == TxSize::Tx4x4 {
            if let (Some(al), Some(ad)) = (
                self.s.above.level[plane].get_mut(ax0),
                self.s.above.dc[plane].get_mut(ax0),
            ) {
                *al = cul_level;
                *ad = dc_category;
            }
            if ly < 32 {
                self.s.left.level[plane][ly] = cul_level;
                self.s.left.dc[plane][ly] = dc_category;
            }
        } else {
            set_ctx2_log2(
                &mut self.s.above.level[plane],
                &mut self.s.above.dc[plane],
                ax0,
                plan.w4_log2,
                cul_level,
                dc_category,
            );
            set_ctx2_log2(
                &mut self.s.left.level[plane],
                &mut self.s.left.dc[plane],
                ly,
                plan.h4_log2,
                cul_level,
                dc_category,
            );
        }
    }

    #[inline(never)]
    fn coeffs_nz<const PARSE: bool>(
        &mut self,
        plane: usize,
        plan: TxPlan,
        ax0: usize,
        ly: usize,
        y4: usize,
    ) -> u32 {
        let tx_sz_ctx = (plan.tx_sz_ctx as usize).min(4);
        let ptype = (plane > 0) as usize;
        let tx_type = if plan.tx_mode < 16 {
            TxType::ALL[(plan.tx_mode & 15) as usize]
        } else if !PARSE && plan.tx_mode >= 24 {
            self.resolve_inter_tx_type(plan, ax0, y4)
        } else {
            let intra_dir = (plan.dir_and_zbase & 15).min(12) as usize;
            let msac = &mut self.t.msac;
            let cdf = &mut *self.t.cdf;
            if plan.tx_mode < 18 {
                let v = msac
                    .symbol_n(&mut cdf.intra_tx_type_set1[(plan.tx_mode & 1) as usize][intra_dir]);
                TX_TYPE_INTRA_INV_SET1[v & 7]
            } else {
                let sqr = (plan.tx_mode.wrapping_sub(18) as usize).min(2);
                let v = msac.symbol6(&mut cdf.intra_tx_type_set2[sqr][intra_dir]);
                TX_TYPE_INTRA_INV_SET2[v.min(4)]
            }
        };
        self.plane_tx_type = tx_type;
        let tx_cls = tx_class(tx_type);
        let eob_multisize = plan.eob_multisize;
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
            last = ((hi | 2) << eb) | if eb != 0 { msac.literal_short(eb) } else { 0 };
        }
        let eob = last + 1;

        if eob != 1 {
            return self.coeffs_ac::<PARSE>(plane, plan, ax0, ly, y4, eob, tx_cls);
        }
        // DC only (eob == 1), the common case of small transforms: kept
        // separate from the coefficient loops so that this path stays
        // small (few spills) and needs none of the AC setup.
        let mut tok = msac.symbol_n(&mut cdf.coeff_base_eob[tx_sz_ctx][ptype][0]) as u32 + 1;
        if tok == 3 {
            tok = msac.hi_tok(&mut cdf.coeff_br[tx_sz_ctx.min(3)][ptype][0]);
        }
        let Some((q, dc_category, dq)) = self.dc_coef(plane, plan, ax0, ly, y4, tok) else {
            return u32::MAX;
        };
        if PARSE {
            if let super::stage::DecoderStage::Parse(log) = &mut self.stage {
                log.tx_entries.push((0, dq));
                log.tx_headers.push(super::stage::TxRecHeader {
                    eob: 1,
                    tx_type,
                    count: 1,
                });
            }
        } else {
            self.sc.coeffs[0] = dq;
        }
        let cul_level = q.min(63) as u8;
        self.update_coeff_ctx(plane, plan, ax0, ly, cul_level, dc_category);
        1
    }

    /// Reads the sign of the non-zero DC token `dc_tok` (plus its golomb
    /// remainder) and computes the dequantized DC coefficient. Returns
    /// `(level, dc_category, dq)` (`dc_category`: 1 negative, 2 positive),
    /// or `None` on an invalid golomb code.
    #[inline(always)]
    fn dc_coef(
        &mut self,
        plane: usize,
        plan: TxPlan,
        ax0: usize,
        ly: usize,
        y4: usize,
        dc_tok: u32,
    ) -> Option<(u32, u8, i32)> {
        let f = self.f;
        let tx_sz = plan.tx_sz;
        let ptype = (plane > 0) as usize;
        let ctx = if tx_sz == TxSize::Tx4x4 {
            let da = self.s.above.dc[plane][ax0] as i32;
            let dl = self.s.left.dc[plane][ly] as i32;
            let dc_sign = (da >> 1)
                .wrapping_add(dl >> 1)
                .wrapping_sub(da & 1)
                .wrapping_sub(dl & 1);
            (dc_sign < 0) as usize + 2 * (dc_sign > 0) as usize
        } else {
            let w4 = 1usize << plan.w4_log2.min(4);
            let h4 = 1usize << plan.h4_log2.min(4);
            let (sx, sy) = f.sub(plane);
            let x4 = ax0.wrapping_add(self.s.above.col0 >> sx);
            let kx = w4.min((f.mi_cols >> sx).saturating_sub(x4));
            let ky = h4.min((f.mi_rows >> sy).saturating_sub(y4));
            dc_sign_ctx(
                &self.s.above.dc[plane],
                ax0,
                kx,
                &self.s.left.dc[plane],
                ly,
                ky,
            )
        };
        let sign = self.t.msac.bool_adapt(&mut self.t.cdf.dc_sign[ptype][ctx]);
        let dc_q = self.b.dc_q[plane];
        let qm_level = self.b.qm_level[plane] as usize;
        let use_qm = qm_level < 15 && (self.plane_tx_type as usize) < TxType::IDTX as usize;
        let dq_shift = (plan.dq_shift & 31) as u32;
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
                (dc_q.wrapping_mul(qm0).wrapping_add(16)) >> 5
            } else {
                dc_q
            };
            let d = ((q.wrapping_mul(q2)) >> dq_shift) as i32;
            std::hint::select_unpredictable(sign, d.wrapping_neg(), d)
        } else {
            let g = self.t.msac.golomb_coef()?;
            q = g.wrapping_add(14) & 0xFFFFF;
            // Largest magnitude of a positive coefficient; negative ones
            // may be one larger.
            let cf_max = (1u64 << (7 + f.bd)) - 1;
            dequant_dc_slow(
                q, dc_q, qm_level, ptype, tx_sz, dq_shift, cf_max, sign, use_qm,
            )
        };
        Some((q, 2u8.wrapping_sub(sign as u8), dq))
    }

    /// `coeffs_nz` for `eob > 1`: coefficient tokens, then signs, golomb
    /// remainders and dequantization (spec 7.12.3 step 1) in scan order
    /// over the non-zero coefficients.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn coeffs_ac<const PARSE: bool>(
        &mut self,
        plane: usize,
        plan: TxPlan,
        ax0: usize,
        ly: usize,
        y4: usize,
        eob: usize,
        tx_cls: usize,
    ) -> u32 {
        let f = self.f;
        let tx_sz = plan.tx_sz;
        let ts = tx_sz as usize;
        let tx_sz_ctx = (plan.tx_sz_ctx as usize).min(4);
        let ptype = (plane > 0) as usize;
        // Levels are kept in a padded 2D array for context derivation;
        // non-zero entries are cleared while walking `nz` below.
        // `lbwl` maps a position to its levels index (`pos + (pos >> lbwl
        // << 2)`): the 2D class uses an unpadded array (see coef_tokens).
        let (dc_tok, num_ac, lbwl) = {
            let scan = coef_scan(tx_sz, tx_cls);
            let Some(scan) = scan.get(..eob) else {
                return u32::MAX;
            };
            let adj_bh = ADJ_BWL_HLOG2[ts];
            let bwl = ((adj_bh & 15) as usize).clamp(2, 5);
            let height = 1usize << ((adj_bh >> 4) & 7);
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
        let use_qm = qm_level < 15 && (self.plane_tx_type as usize) < TxType::IDTX as usize;
        let dq_shift = (plan.dq_shift & 31) as u32;
        // Largest magnitude of a positive coefficient; negative ones
        // may be one larger.
        let cf_max = (1u64 << (7 + f.bd)) - 1;
        let mut cul_level = 0u32;
        let mut dc_category = 0u8;
        let mut dc_dq = 0i32;
        let has_dc = dc_tok != 0;
        if has_dc {
            let Some((q, cat, dq)) = self.dc_coef(plane, plan, ax0, ly, y4, dc_tok) else {
                reset_scratch_on_err(self.sc);
                return u32::MAX;
            };
            cul_level = q;
            dc_category = cat;
            if !PARSE || use_qm {
                self.sc.coeffs[0] = dq;
            } else {
                dc_dq = dq;
            }
        }
        let n_ac = num_ac.min(1024);
        if use_qm {
            if n_ac != 0 {
                let Some(c) = self.dequant_ac_qm(n_ac, plane, tx_sz, lbwl.max(2)) else {
                    reset_scratch_on_err(self.sc);
                    return u32::MAX;
                };
                cul_level += c;
            }
            if PARSE {
                if let super::stage::DecoderStage::Parse(log) = &mut self.stage {
                    record_parse_tx_qm(self.sc, self.plane_tx_type, log, eob, has_dc, n_ac);
                }
            }
        } else if PARSE {
            #[cold]
            #[inline(always)]
            fn cold_hi_tok() {}

            let mut ok = true;
            let mut m = self.t.msac.clone();
            let msac = &mut m;
            let sc = &mut *self.sc;
            let levels = &mut sc.levels;
            let nz = &sc.nz[..n_ac];
            if let super::stage::DecoderStage::Parse(log) = &mut self.stage {
                let count = has_dc as usize + n_ac;
                log.tx_entries.reserve(count);
                if has_dc {
                    log.tx_entries.push((0, dc_dq));
                }
                if tx_cls == 0 {
                    log.tx_entries.extend(nz.iter().rev().map(|&e| {
                        let e = e as u32;
                        let rc = (e & 0x3FF) as usize;
                        levels[rc] = 0;
                        let q = e >> 10;
                        let mut dq = 0i32;
                        if q < 15 {
                            let d = ((q.wrapping_mul(ac_q)) >> dq_shift) as i32;
                            cul_level = cul_level.wrapping_add(q);
                            msac.bool_store_neg(&mut dq, d);
                        } else {
                            cold_hi_tok();
                            let (q_hi, dq_hi) = msac.ac_golomb(ac_q, dq_shift, cf_max);
                            if q_hi == 0 {
                                ok = false;
                            }
                            cul_level = cul_level.wrapping_add(q_hi & 0xFFFFF);
                            dq = dq_hi;
                        }
                        (rc as u16, dq)
                    }));
                } else {
                    let lbwl = lbwl.max(2);
                    log.tx_entries.extend(nz.iter().rev().map(|&e| {
                        let e = e as u32;
                        let rc = (e & 0x3FF) as usize;
                        levels[((rc as u32).wrapping_add((rc as u32 >> lbwl) << 2) & 2047)
                            as usize] = 0;
                        let q = e >> 10;
                        let mut dq = 0i32;
                        if q < 15 {
                            let d = ((q.wrapping_mul(ac_q)) >> dq_shift) as i32;
                            cul_level = cul_level.wrapping_add(q);
                            msac.bool_store_neg(&mut dq, d);
                        } else {
                            cold_hi_tok();
                            let (q_hi, dq_hi) = msac.ac_golomb(ac_q, dq_shift, cf_max);
                            if q_hi == 0 {
                                ok = false;
                            }
                            cul_level = cul_level.wrapping_add(q_hi & 0xFFFFF);
                            dq = dq_hi;
                        }
                        (rc as u16, dq)
                    }));
                }
                log.tx_headers.push(super::stage::TxRecHeader {
                    eob: eob as u16,
                    tx_type: self.plane_tx_type,
                    count: count as u16,
                });
            }
            self.t.msac.copy_state(&m);
            if !ok {
                reset_scratch_on_err(self.sc);
                return u32::MAX;
            }
        } else if n_ac != 0 {
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
                        let d = ((q.wrapping_mul(ac_q)) >> dq_shift) as i32;
                        cul_level = cul_level.wrapping_add(q);
                        msac.bool_store_neg(&mut cf[rc], d);
                    } else {
                        cold_hi_tok();
                        let (q_hi, dq) = msac.ac_golomb(ac_q, dq_shift, cf_max);
                        if q_hi == 0 {
                            reset_scratch_on_err(self.sc);
                            return u32::MAX;
                        }
                        cul_level = cul_level.wrapping_add(q_hi & 0xFFFFF);
                        cf[rc] = dq;
                    }
                }
            } else {
                let lbwl = lbwl.max(2);
                for &e in nz.iter().rev() {
                    let e = e as u32;
                    let rc = (e & 0x3FF) as usize;
                    levels[((rc as u32).wrapping_add((rc as u32 >> lbwl) << 2) & 2047) as usize] =
                        0;
                    let q = e >> 10;
                    if q < 15 {
                        let d = ((q.wrapping_mul(ac_q)) >> dq_shift) as i32;
                        cul_level = cul_level.wrapping_add(q);
                        msac.bool_store_neg(&mut cf[rc], d);
                    } else {
                        cold_hi_tok();
                        let (q_hi, dq) = msac.ac_golomb(ac_q, dq_shift, cf_max);
                        if q_hi == 0 {
                            reset_scratch_on_err(self.sc);
                            return u32::MAX;
                        }
                        cul_level = cul_level.wrapping_add(q_hi & 0xFFFFF);
                        cf[rc] = dq;
                    }
                }
            }
            self.t.msac.copy_state(&m);
        }
        cul_level = cul_level.min(63);
        self.update_coeff_ctx(plane, plan, ax0, ly, cul_level as u8, dc_category);
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
                levels[((rc as u32 + ((rc as u32 >> lbwl) << 2)) & 2047) as usize] = 0;
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

    /// `coef_tokens` dispatch for tiles with CDF updates disabled.
    #[cold]
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
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
    /// 11.. with bit 10 set to `1` for rounding (see `coef_scan`).
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

        let mut tok = msac
            .symbol4_c::<UPD, true>(&mut eob_cdf[ctx])
            .wrapping_add(1);
        let lb = if tok == 3 {
            cold_hi_tok();
            tok = msac.hi_tok_c::<UPD>(&mut hi_cdf[br_pos_ctx::<CLS, BWL>(pos)]);
            (tok as u8) | (3 << 6)
        } else {
            (tok as u8).wrapping_mul(0x41)
        };
        // `& 2047` is a no-op (level indices are < 2048) that drops bounds checks.
        levels[level_idx::<CLS, BWL>(pos) & 2047] = lb;
        nz[0] = ((tok as u16) << 10) | pos as u16;
        let mut k = 1usize;

        for &p in ac_scan.iter().rev() {
            let p32 = p as u32;
            let pos = (p32 & 0x3FF) as usize;
            let idx = level_idx::<CLS, BWL>(pos);
            let ctx = lo_ctx::<CLS, BWL>(levels, idx, p32);
            let cdf_row = if CLS == 0 { &mut lo_cdf[ctx] } else { &mut lo_cdf[22 + ctx] };
            let tok = msac.symbol5_c::<UPD>(cdf_row);
            // k < scan.len() <= 1024.
            if tok == 3 {
                cold_hi_tok();
                let hi = hi_mag::<CLS, BWL>(levels, idx);
                let tok = msac.hi_tok_c::<UPD>(
                    &mut hi_cdf[br_pos_ctx::<CLS, BWL>(pos).wrapping_add(hi_ctx(hi))],
                );
                levels[level_idx::<CLS, BWL>(pos) & 2047] = (tok as u8) | (3 << 6);
                nz[k & 1023] = ((tok as u16) << 10) | pos as u16;
                k = k.wrapping_add(1);
            } else {
                // Branch free (zero / non-zero is unpredictable), like
                // dav1d: zero tokens store 0 (levels are zero there
                // already) and are dropped from `nz` by not advancing k.
                levels[idx & 2047] = (tok.wrapping_mul(0x41)) as u8;
                nz[k & 1023] = ((tok as u16) << 10) | pos as u16;
                k = k.wrapping_add((tok != 0) as usize);
            }
        }

        // DC (`(4 << 11) | (1 << 10)` encodes 1D relative offset `4` + rounding bit `10`).
        let mut tok = if CLS == 0 {
            msac.symbol5_c::<UPD>(&mut lo_cdf[0])
        } else {
            let ctx = lo_ctx::<CLS, BWL>(levels, 0, (4 << 11) | (1 << 10));
            msac.symbol5_c::<UPD>(&mut lo_cdf[22 + ctx])
        };
        if tok == 3 {
            cold_hi_tok();
            let hi = hi_mag::<CLS, BWL>(levels, 0);
            tok = msac.hi_tok_c::<UPD>(&mut hi_cdf[hi_ctx(hi)]);
        }
        self.t.msac.copy_state(&m);
        (tok, k)
    }
}

#[inline(never)]
fn set_tx_types(
    tx_types: &mut [[TxType; 32]; 32],
    bx: usize,
    by: usize,
    w4_log2: u8,
    h4_log2: u8,
    t: TxType,
) {
    let bx_end = (bx + (1usize << w4_log2.min(4))).min(32);
    let by_end = (by + (1usize << h4_log2.min(4))).min(32);
    if bx >= bx_end || by >= by_end {
        return;
    }
    let n = bx_end - bx;
    let rows = &mut tx_types[by..by_end];
    if n == 1 {
        for row in rows {
            row[bx] = t;
        }
    } else if n >= 8 {
        let oct = [t; 8];
        for row in rows {
            let s = &mut row[bx..bx_end];
            if let Some(f) = s.first_chunk_mut::<8>() {
                *f = oct;
            }
            if let Some(l) = s.last_chunk_mut::<8>() {
                *l = oct;
            }
        }
    } else if n >= 4 {
        let quad = [t; 4];
        for row in rows {
            let s = &mut row[bx..bx_end];
            if let Some(f) = s.first_chunk_mut::<4>() {
                *f = quad;
            }
            if n > 4 {
                if let Some(l) = s.last_chunk_mut::<4>() {
                    *l = quad;
                }
            }
        }
    } else {
        let pair = [t; 2];
        for row in rows {
            let s = &mut row[bx..bx_end];
            if let Some(f) = s.first_chunk_mut::<2>() {
                *f = pair;
            }
            if n > 2 {
                if let Some(l) = s.last_chunk_mut::<2>() {
                    *l = pair;
                }
            }
        }
    }
}

#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn dequant_dc_slow(
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
        let qm = crate::decoder::av1::tables::qm::quantizer_matrix(qm_level.min(14), ptype, tx_sz);
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

/// Categorizes `max(s[pos..pos + n])` (with `n <= 16`) into `0` (`== 0`),
/// `1` (`1..=3`), or `2` (`> 3`) without horizontal byte-reduction shifts.
#[inline(always)]
fn level_cat(s: &[u8], pos: usize, n: usize) -> u32 {
    if n == 1 {
        let b = s.get(pos).copied().unwrap_or(0);
        return (b != 0) as u32 + (b > 3) as u32;
    }
    let Some(s) = s.get(pos..pos + n) else {
        return 0;
    };
    if let (Some(a), Some(b)) = (s.first_chunk::<8>(), s.last_chunk::<8>()) {
        let v = u64::from_ne_bytes(*a) | u64::from_ne_bytes(*b);
        (v != 0) as u32 + ((v & 0xFCFC_FCFC_FCFC_FCFC) != 0) as u32
    } else if let (Some(a), Some(b)) = (s.first_chunk::<4>(), s.last_chunk::<4>()) {
        let v = u32::from_ne_bytes(*a) | u32::from_ne_bytes(*b);
        (v != 0) as u32 + ((v & 0xFCFC_FCFC) != 0) as u32
    } else if n > 1 {
        let b = s[0] | s[1] | s[n - 1];
        (b != 0) as u32 + (b > 3) as u32
    } else {
        0
    }
}

/// Returns true if any byte in `a[pos..pos + n]` or `b[pos..pos + n]` (`n <= 16`) is non-zero.
#[inline(always)]
fn any_nz2(a: &[u8], b: &[u8], pos: usize, n: usize) -> bool {
    if n == 1 {
        return match (a.get(pos), b.get(pos)) {
            (Some(&va), Some(&vb)) => (va | vb) != 0,
            _ => false,
        };
    }
    let (Some(a), Some(b)) = (a.get(pos..pos + n), b.get(pos..pos + n)) else {
        return false;
    };
    if let (Some(a0), Some(a1), Some(b0), Some(b1)) = (
        a.first_chunk::<8>(),
        a.last_chunk::<8>(),
        b.first_chunk::<8>(),
        b.last_chunk::<8>(),
    ) {
        (u64::from_ne_bytes(*a0)
            | u64::from_ne_bytes(*a1)
            | u64::from_ne_bytes(*b0)
            | u64::from_ne_bytes(*b1))
            != 0
    } else if let (Some(a0), Some(a1), Some(b0), Some(b1)) = (
        a.first_chunk::<4>(),
        a.last_chunk::<4>(),
        b.first_chunk::<4>(),
        b.last_chunk::<4>(),
    ) {
        (u32::from_ne_bytes(*a0)
            | u32::from_ne_bytes(*a1)
            | u32::from_ne_bytes(*b0)
            | u32::from_ne_bytes(*b1))
            != 0
    } else if n > 1 {
        (a[0] | a[1] | a[n - 1] | b[0] | b[1] | b[n - 1]) != 0
    } else {
        false
    }
}

#[inline(always)]
fn prefix_mask8(n: usize) -> u64 {
    if n == 0 {
        0
    } else {
        !0u64 >> ((n.wrapping_neg() & 7) << 3)
    }
}

#[cold]
#[inline(never)]
fn load_short_dc16(above: &[u8], ax0: usize, kx: usize) -> (u64, u64) {
    let mut b = [0u8; 16];
    if let Some(tail) = above.get(ax0..) {
        let n = tail.len().min(kx).min(16);
        b[..n].copy_from_slice(&tail[..n]);
    }
    (
        u64::from_le_bytes(*b.first_chunk::<8>().unwrap()),
        u64::from_le_bytes(*b.last_chunk::<8>().unwrap()),
    )
}

#[inline(always)]
fn dc_sign_ctx(
    above: &[u8],
    ax0: usize,
    kx: usize,
    left: &[u8; 32],
    ly: usize,
    ky: usize,
) -> usize {
    const ONES: u64 = 0x0101_0101_0101_0101;
    let kx = kx.min(16);
    let (mut a0, mut a1) = if let Some(c) = above.get(ax0..).and_then(|s| s.first_chunk::<16>()) {
        (
            u64::from_le_bytes(*c.first_chunk::<8>().unwrap()),
            u64::from_le_bytes(*c.last_chunk::<8>().unwrap()),
        )
    } else {
        load_short_dc16(above, ax0, kx)
    };
    a0 &= prefix_mask8(kx.min(8));
    a1 &= prefix_mask8(kx.saturating_sub(8));

    let ky = ky.min(32usize.saturating_sub(ly)).min(16);
    let base = ly.min(16);
    let l_chunk = left[base..].first_chunk::<16>().unwrap_or(&[0; 16]);
    let lv = u128::from_le_bytes(*l_chunk) >> (((ly.min(31) - base) & 15) << 3);
    let l0 = (lv as u64) & prefix_mask8(ky.min(8));
    let l1 = ((lv >> 64) as u64) & prefix_mask8(ky.saturating_sub(8));

    let pos = ((a0 >> 1) & ONES)
        .wrapping_add((a1 >> 1) & ONES)
        .wrapping_add((l0 >> 1) & ONES)
        .wrapping_add((l1 >> 1) & ONES);
    let neg = (a0 & ONES)
        .wrapping_add(a1 & ONES)
        .wrapping_add(l0 & ONES)
        .wrapping_add(l1 & ONES);
    let sum = 0x0404_0404_0404_0404u64
        .wrapping_add(pos)
        .wrapping_sub(neg)
        .wrapping_mul(ONES)
        >> 56;
    (sum < 32) as usize + 2 * (sum > 32) as usize
}

#[inline(always)]
fn set_ctx2_log2(a: &mut [u8], b: &mut [u8], pos: usize, log2: u8, va: u8, vb: u8) {
    match log2 {
        0 => {
            if let (Some(sa), Some(sb)) = (a.get_mut(pos), b.get_mut(pos)) {
                *sa = va;
                *sb = vb;
            }
        }
        1 => {
            if let (Some(sa), Some(sb)) = (
                a.get_mut(pos..).and_then(|s| s.first_chunk_mut::<2>()),
                b.get_mut(pos..).and_then(|s| s.first_chunk_mut::<2>()),
            ) {
                *sa = ((va as u16).wrapping_mul(0x0101)).to_ne_bytes();
                *sb = ((vb as u16).wrapping_mul(0x0101)).to_ne_bytes();
            }
        }
        2 => {
            if let (Some(sa), Some(sb)) = (
                a.get_mut(pos..).and_then(|s| s.first_chunk_mut::<4>()),
                b.get_mut(pos..).and_then(|s| s.first_chunk_mut::<4>()),
            ) {
                *sa = ((va as u32).wrapping_mul(0x0101_0101)).to_ne_bytes();
                *sb = ((vb as u32).wrapping_mul(0x0101_0101)).to_ne_bytes();
            }
        }
        3 => {
            if let (Some(sa), Some(sb)) = (
                a.get_mut(pos..).and_then(|s| s.first_chunk_mut::<8>()),
                b.get_mut(pos..).and_then(|s| s.first_chunk_mut::<8>()),
            ) {
                *sa = ((va as u64).wrapping_mul(0x0101_0101_0101_0101)).to_ne_bytes();
                *sb = ((vb as u64).wrapping_mul(0x0101_0101_0101_0101)).to_ne_bytes();
            }
        }
        _ => {
            if let (Some(sa), Some(sb)) = (
                a.get_mut(pos..).and_then(|s| s.first_chunk_mut::<16>()),
                b.get_mut(pos..).and_then(|s| s.first_chunk_mut::<16>()),
            ) {
                let wa = ((va as u64).wrapping_mul(0x0101_0101_0101_0101)).to_ne_bytes();
                let wb = ((vb as u64).wrapping_mul(0x0101_0101_0101_0101)).to_ne_bytes();
                *sa.first_chunk_mut::<8>().unwrap() = wa;
                *sa.last_chunk_mut::<8>().unwrap() = wa;
                *sb.first_chunk_mut::<8>().unwrap() = wb;
                *sb.last_chunk_mut::<8>().unwrap() = wb;
            }
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
fn hi_mag<const CLS: usize, const BWL: usize>(levels: &[u8; 2200], idx: usize) -> u32 {
    let ls = (1usize << BWL) + if CLS == 0 { 0 } else { LEVELS_PAD };
    match CLS {
        0 => levels[idx + 1] as u32 + levels[idx + ls] as u32 + levels[idx + ls + 1] as u32,
        1 => levels[idx + 1] as u32 + levels[idx + ls] as u32 + levels[idx + 2] as u32,
        _ => levels[idx + 1] as u32 + levels[idx + ls] as u32 + levels[idx + 2 * ls] as u32,
    }
}

/// Index of position `pos` in the levels array (see `coef_tokens`).
#[inline(always)]
fn level_idx<const CLS: usize, const BWL: usize>(pos: usize) -> usize {
    if CLS == 0 {
        pos
    } else {
        (pos as u32).wrapping_add((pos as u32 >> BWL) << 2) as usize
    }
}

/// Returns the `coeff_base` context of the coefficient at `idx` in the
/// levels array (stride `ls`), without the special case of the 2D DC.
/// For `CLS == 0`, returns the absolute context in `0..36`; for `CLS != 0`,
/// returns the context relative to `lo_cdf[22]` in `0..20` (`<= 19`, so
/// `lo_cdf[22 + ctx]` needs no bounds check on the 42-entry `coeff_base`).
#[inline(always)]
fn lo_ctx<const CLS: usize, const BWL: usize>(levels: &[u8; 2200], idx: usize, p: u32) -> usize {
    let ls = (1usize << BWL) + if CLS == 0 { 0 } else { LEVELS_PAD };
    let off = if CLS == 0 { (p >> 11) & 31 } else { (p >> 11) & 15 };
    // `p` encodes `(off << 11) | (1 << 10) | pos`, so `(p & mask) >> 4`
    // equals `(off << 7) + 64` as a runtime value derived from `p` before
    // the `levels` loads finish (preventing LLVM Reassociate from deferring
    // `+ 64` past the most recent neighbour loads).
    let bias = (p & if CLS == 0 { 0xFC00 } else { 0x7C00 }) >> 4;
    let cap = off + 4;
    let mag = match CLS {
        0 => {
            let far = bias
                + (levels[idx + 2] as u32 + levels[idx + 2 * ls] as u32)
                + levels[idx + ls + 1] as u32;
            (far + levels[idx + ls] as u32) + levels[idx + 1] as u32
        }
        1 => {
            // Column scan (`MCOL_SCAN`): `idx + 1..=4` are 4 contiguous far
            // bytes (each `<= 0xCF`, so 16-bit lane sums are `<= 414`), loaded
            // as one GPR `u32` to avoid a cross-domain NEON horizontal sum;
            // `idx + ls` was visited 1 step ago and is added last.
            let v = u32::from_ne_bytes(*levels[idx + 1..idx + 5].first_chunk::<4>().unwrap());
            let s16 = (v & 0x00FF_00FF).wrapping_add((v >> 8) & 0x00FF_00FF);
            let far = bias + (s16.wrapping_add(s16 >> 16) & 0xFFFF);
            far + levels[idx + ls] as u32
        }
        _ => {
            // Row scan (`MROW_SCAN`): reverse scan visited `idx + 1` one step ago.
            let far = bias
                + (levels[idx + 2 * ls] as u32 + levels[idx + 3 * ls] as u32)
                + (levels[idx + 4 * ls] as u32 + levels[idx + ls] as u32);
            far + levels[idx + 1] as u32
        }
    };
    // The low 6 bits of the 5-neighbour sum are at most 31 when the (min 3)
    // high parts sum to less than 8, so they never disturb the rounding.
    (mag >> 7).min(cap) as usize
}

/// Repacks a 2D `COEF_SCAN_*` table (`pos | (off << 10)`) into
/// `pos | (1 << 10) | (off << 11)` so bit 10 provides the `+ 64` rounding bit.
const fn with_2d_bias<const N: usize>(scan: &[u16; N]) -> [u16; N] {
    let mut out = [0u16; N];
    let mut i = 0;
    while i < N {
        let s = scan[i];
        let pos = s & 0x3FF;
        let off = s >> 10;
        assert!(off <= 21);
        out[i] = pos | (1 << 10) | (off << 11);
        i += 1;
    }
    out
}

macro_rules! scan_2d {
    ($t:expr) => {{
        const T: [u16; $t.len()] = with_2d_bias(&$t);
        &T as &[u16]
    }};
}

/// Returns `scan` for a 1D class (`cls == 1` or `2`) with bit 10 set to `1`
/// (rounding) and the `coeff_base` context offset relative to index 22
/// (`4, 9, 14` = `26 - 22 + {0, 5, 10}`) packed in bits 11..14.
const fn with_1d_ctx<const N: usize>(scan: &[u16; N], cls: usize, bwl: usize) -> [u16; N] {
    let mut out = *scan;
    let mut i = 0;
    while i < N {
        assert!(scan[i] & 0xFC00 == 0);
        let pos = (scan[i] & 0x3FF) as usize;
        let coord = if cls == 1 { pos & ((1 << bwl) - 1) } else { pos >> bwl };
        let c_off = 4 + if coord < 2 { coord } else { 2 } * 5;
        out[i] = (pos as u16) | (1 << 10) | ((c_off as u16) << 11);
        i += 1;
    }
    out
}

/// `with_1d_ctx` of a 1D scan table, as a `&'static [u16]`.
macro_rules! scan_1d {
    ($t:expr, $cls:expr, $bwl:expr) => {{
        const T: [u16; $t.len()] = with_1d_ctx(&$t, $cls, $bwl);
        &T as &[u16]
    }};
}

/// `coef_scan` table indexed by `[tx_class][tx_size]`. `tx_class` is
/// 0..=2; entry 3 duplicates entry 2 only so that the `tx_cls & 3` index
/// in `coef_scan` needs no bounds check. Every entry carries bit 10 = `1`
/// and its relative `coeff_base` context offset in bits 11..15.
const COEF_SCAN: [[&[u16]; TxSize::COUNT]; 4] = {
    use crate::decoder::av1::tables::scan::*;
    use TxSize::*;
    let mut table = [[scan_2d!(COEF_SCAN_4X4); TxSize::COUNT]; 4];
    let mut i = 0;
    while i < TxSize::COUNT {
        let tx_sz = TxSize::ALL[i];
        table[0][i] = match tx_sz {
            Tx4x4 => scan_2d!(COEF_SCAN_4X4),
            Tx8x8 => scan_2d!(COEF_SCAN_8X8),
            Tx16x16 => scan_2d!(COEF_SCAN_16X16),
            Tx32x32 | Tx64x64 => scan_2d!(COEF_SCAN_32X32),
            Tx4x8 => scan_2d!(COEF_SCAN_4X8),
            Tx8x4 => scan_2d!(COEF_SCAN_8X4),
            Tx8x16 => scan_2d!(COEF_SCAN_8X16),
            Tx16x8 => scan_2d!(COEF_SCAN_16X8),
            Tx16x32 | Tx16x64 => scan_2d!(COEF_SCAN_16X32),
            Tx32x16 | Tx64x16 => scan_2d!(COEF_SCAN_32X16),
            Tx32x64 => scan_2d!(COEF_SCAN_32X32_TALL),
            Tx64x32 => scan_2d!(COEF_SCAN_32X32_WIDE),
            Tx4x16 => scan_2d!(COEF_SCAN_4X16),
            Tx16x4 => scan_2d!(COEF_SCAN_16X4),
            Tx8x32 => scan_2d!(COEF_SCAN_8X32),
            Tx32x8 => scan_2d!(COEF_SCAN_32X8),
        };
        // 1D classes (`get_scan` with H_DCT / V_DCT). The 1D `BWL` is at
        // most 4 (see `coeffs_ac`). The sizes with a 32 or 64 dimension
        // are never 1D in valid streams.
        (table[1][i], table[2][i]) = match tx_sz {
            Tx4x4 => (scan_1d!(MCOL_SCAN_4X4, 1, 2), scan_1d!(MROW_SCAN_4X4, 2, 2)),
            Tx4x8 => (scan_1d!(MCOL_SCAN_4X8, 1, 2), scan_1d!(MROW_SCAN_4X8, 2, 2)),
            Tx8x4 => (scan_1d!(MCOL_SCAN_8X4, 1, 3), scan_1d!(MROW_SCAN_8X4, 2, 3)),
            Tx8x8 => (scan_1d!(MCOL_SCAN_8X8, 1, 3), scan_1d!(MROW_SCAN_8X8, 2, 3)),
            Tx8x16 => (
                scan_1d!(MCOL_SCAN_8X16, 1, 3),
                scan_1d!(MROW_SCAN_8X16, 2, 3),
            ),
            Tx16x8 => (
                scan_1d!(MCOL_SCAN_16X8, 1, 4),
                scan_1d!(MROW_SCAN_16X8, 2, 4),
            ),
            Tx16x16 => (
                scan_1d!(MCOL_SCAN_16X16, 1, 4),
                scan_1d!(MROW_SCAN_16X16, 2, 4),
            ),
            Tx4x16 => (
                scan_1d!(MCOL_SCAN_4X16, 1, 2),
                scan_1d!(MROW_SCAN_4X16, 2, 2),
            ),
            Tx16x4 | Tx16x32 | Tx32x16 | Tx32x8 | Tx32x32 => (
                scan_1d!(MCOL_SCAN_16X4, 1, 4),
                scan_1d!(MROW_SCAN_16X4, 2, 4),
            ),
            Tx8x32 => (
                scan_1d!(MCOL_SCAN_16X4, 1, 3),
                scan_1d!(MROW_SCAN_16X4, 2, 3),
            ),
            Tx16x64 => (
                scan_1d!(DEFAULT_SCAN_16X32, 1, 4),
                scan_1d!(DEFAULT_SCAN_16X32, 2, 4),
            ),
            Tx64x16 => (
                scan_1d!(DEFAULT_SCAN_32X16, 1, 4),
                scan_1d!(DEFAULT_SCAN_32X16, 2, 4),
            ),
            Tx64x64 | Tx32x64 | Tx64x32 => (
                scan_1d!(DEFAULT_SCAN_32X32, 1, 4),
                scan_1d!(DEFAULT_SCAN_32X32, 2, 4),
            ),
        };
        table[3][i] = table[2][i];
        i += 1;
    }
    table
};

/// Scan of the coefficient reader: `get_scan`, with every entry carrying
/// bit 10 = `1` and the relative `coeff_base` context offset in bits 11..15.
#[inline(always)]
fn coef_scan(tx_sz: TxSize, tx_cls: usize) -> &'static [u16] {
    COEF_SCAN[tx_cls & 3][tx_sz as usize]
}

/// `coeff_br` magnitude context from the raw 3-neighbour sum:
/// `mag > 12 ? 6 : (mag + 1) >> 1`. Each neighbour's low 6 bits are `<= 15`,
/// so `(hi & 63) + 1 <= 46 < 64` never carries into bit 6.
#[inline(always)]
fn hi_ctx(hi: u32) -> usize {
    (((hi + 1) >> 1) & 31).min(6) as usize
}

mod modeinfo;

#[cfg(test)]
mod tests {
    use super::*;

    /// `coef_scan` is `get_scan` plus bit 10 = 1 and, for the 2D class, the
    /// spec's `Coeff_Base_Ctx_Offset[txSz][min(row, 4)][min(col, 4)]` in bits 11..15.
    #[test]
    fn coef_scan_ctx_offsets() {
        for ts in 0..19u8 {
            let tx_sz = TxSize::from_u8(ts);
            let adj = ADJUSTED_TX_SIZE[ts as usize] as usize;
            let bwl = TX_WIDTH_LOG2[adj] as usize;
            for tx_type in TxType::ALL {
                let ref_cls = match tx_type {
                    TxType::V_DCT | TxType::V_ADST | TxType::V_FLIPADST => 2,
                    TxType::H_DCT | TxType::H_ADST | TxType::H_FLIPADST => 1,
                    _ => 0,
                };
                let cls = tx_class(*tx_type);
                assert_eq!(cls, ref_cls, "{tx_type:?}");
                let plain = get_scan(tx_sz, *tx_type);
                let scan = coef_scan(tx_sz, cls);
                assert_eq!(plain.len(), scan.len());
                for (&p, &s) in plain.iter().zip(scan) {
                    assert_eq!(p, s & 0x3FF);
                    assert_eq!((s >> 10) & 1, 1);
                    if cls == 0 {
                        let (row, col) = ((p as usize) >> bwl, (p as usize) & ((1 << bwl) - 1));
                        let off = COEFF_BASE_CTX_OFFSET[ts as usize][row.min(4)][col.min(4)];
                        assert_eq!(((s >> 11) & 31) as u8, off, "{tx_sz:?} pos {p}");
                    } else {
                        let bwl1d = bwl.clamp(2, 4);
                        let coord = if cls == 1 {
                            (p as usize) & ((1 << bwl1d) - 1)
                        } else {
                            (p as usize) >> bwl1d
                        };
                        let off = (4 + coord.min(2) * 5) as u8;
                        assert_eq!(((s >> 11) & 15) as u8, off, "{tx_sz:?} cls {cls} pos {p}");
                        assert_eq!(22 + off as usize, 26 + coord.min(2) * 5);
                    }
                }
            }
        }
    }

    #[test]
    fn txb_skip_helpers_and_lut() {
        for top in 0..=8usize {
            for left in 0..=8usize {
                let ref_ctx = if (top | left) == 0 {
                    1
                } else if top == 0 || left == 0 {
                    2 + ((top | left) > 3) as usize
                } else if (top | left) <= 3 {
                    4
                } else if top.min(left) <= 3 {
                    5
                } else {
                    6
                };
                let tc = (top != 0) as u32 + (top > 3) as u32;
                let lc = (left != 0) as u32 + (left > 3) as u32;
                let got = ((0x653_0542_0321u64 >> ((tc << 4) | (lc << 2))) & 7) as usize;
                assert_eq!(got, ref_ctx, "top={top} left={left}");
            }
        }
        for n in 1..=16usize {
            for idx in 0..n {
                for val in [0u8, 1, 2, 3, 4, 63, 255] {
                    let mut buf = [0u8; 24];
                    let buf2 = [0u8; 24];
                    buf[3 + idx] = val;
                    let expected = (val != 0) as u32 + (val > 3) as u32;
                    assert_eq!(level_cat(&buf, 3, n), expected, "n={n} idx={idx} val={val}");
                    assert_eq!(any_nz2(&buf, &buf2, 3, n), val != 0);
                    assert_eq!(any_nz2(&buf2, &buf, 3, n), val != 0);
                }
            }
        }
        for (log2, len) in [1usize, 2, 4, 8, 16].into_iter().enumerate() {
            let mut a = [0xAAu8; 24];
            let mut b = [0x55u8; 24];
            set_ctx2_log2(&mut a, &mut b, 3, log2 as u8, 0x12, 0x34);
            assert!(a[..3].iter().all(|&x| x == 0xAA));
            assert!(a[3..3 + len].iter().all(|&x| x == 0x12));
            assert!(a[3 + len..].iter().all(|&x| x == 0xAA));
            assert!(b[..3].iter().all(|&x| x == 0x55));
            assert!(b[3..3 + len].iter().all(|&x| x == 0x34));
            assert!(b[3 + len..].iter().all(|&x| x == 0x55));
        }
    }

    #[test]
    fn dc_sign_ctx_and_block_decoded() {
        let mut above = [0u8; 40];
        let mut left = [0u8; 32];
        for i in 0..40 {
            above[i] = ((i * 7 + 1) % 3) as u8;
        }
        for i in 0..32 {
            left[i] = ((i * 5 + 2) % 3) as u8;
        }
        let ref_dc_sign_ctx =
            |ab: &[u8], ax0: usize, kx: usize, lf: &[u8; 32], ly: usize, ky: usize| {
                let mut s = 0i32;
                for i in 0..kx {
                    let d = ab.get(ax0 + i).copied().unwrap_or(0);
                    s += (d >> 1) as i32 - (d & 1) as i32;
                }
                for j in 0..32 {
                    if (j as u8).wrapping_sub(ly as u8) < (ky as u8) {
                        let d = lf[j];
                        s += (d >> 1) as i32 - (d & 1) as i32;
                    }
                }
                (s < 0) as usize + 2 * (s > 0) as usize
            };
        for &short_len in &[0usize, 5, 15, 25, 40] {
            let ab = &above[..short_len];
            for ax0 in [0usize, 3, 12, 24] {
                for kx in [0usize, 1, 2, 4, 7, 8, 9, 15, 16] {
                    for ly in 0..=32usize {
                        for ky in [0usize, 1, 2, 4, 7, 8, 9, 15, 16] {
                            assert_eq!(
                                dc_sign_ctx(ab, ax0, kx, &left, ly, ky),
                                ref_dc_sign_ctx(ab, ax0, kx, &left, ly, ky),
                                "len={short_len} ax0={ax0} kx={kx} ly={ly} ky={ky}"
                            );
                        }
                    }
                }
            }
        }
        for w4_log2 in 0..=4u8 {
            for h4_log2 in 0..=4u8 {
                let mut bd = [BlockDecodedRow(0); 16];
                set_block_decoded(&mut bd, 5, w4_log2, h4_log2);
                let expected = ((1u64 << (1usize << w4_log2)) - 1) << 5;
                let h4 = 1usize << h4_log2;
                for (r, row) in bd.iter().enumerate() {
                    assert_eq!(row.0, if r < h4 { expected } else { 0 });
                }
            }
        }
    }

    #[test]
    fn tx_meta_and_adj_bwl_equivalence() {
        for ts in 0..TxSize::COUNT {
            let m = TX_META[ts];
            assert_eq!(m.tx_sz, TxSize::ALL[ts]);
            assert_eq!(1usize << m.w4_log2, txw(m.tx_sz) >> 2);
            assert_eq!(1usize << m.h4_log2, txh(m.tx_sz) >> 2);
            let adj = ADJUSTED_TX_SIZE[ts] as usize;
            let adj_bh = ADJ_BWL_HLOG2[ts];
            assert_eq!((adj_bh & 15) as usize, TX_WIDTH_LOG2[adj] as usize);
            assert_eq!(1usize << (adj_bh >> 4), TX_HEIGHT[adj] as usize);
        }
    }

    #[test]
    fn lo_ctx_and_hi_ctx_equivalence() {
        let mut valid_bytes = vec![0u8, 0x41, 0x82];
        for tok in 3..=15u8 {
            valid_bytes.push(tok | (3 << 6));
        }
        // Exhaustively check 3-neighbour hi_ctx equivalence.
        for &a in &valid_bytes {
            for &b in &valid_bytes {
                for &c in &valid_bytes {
                    let hi = a as u32 + b as u32 + c as u32;
                    let ref_ctx = (((hi & 63) + 1) >> 1).min(6) as usize;
                    assert_eq!(hi_ctx(hi), ref_ctx);
                }
            }
        }
        // Check 5-neighbour pre-biased lo_ctx equivalence across 2D and both 1D classes.
        let mut levels = [0u8; 2200];
        for &b0 in &[0u8, 0x41, 0x82, 0xC3, 0xCF] {
            for &b1 in &[0u8, 0x41, 0x82, 0xC3, 0xCF] {
                for &b2 in &[0u8, 0x41, 0x82, 0xCF] {
                    for &b3 in &[0u8, 0x41, 0x82, 0xCF] {
                        for &b4 in &[0u8, 0x41, 0x82, 0xCF] {
                            levels.fill(0);
                            levels[1] = b0;
                            levels[2] = b1;
                            levels[4] = b2;
                            levels[5] = b3;
                            levels[8] = b4;
                            let raw_sum = b0 as u32 + b1 as u32 + b2 as u32 + b3 as u32 + b4 as u32;
                            for off in [0u32, 1, 13, 21] {
                                let p = (off << 11) | (1 << 10);
                                let ref_ctx = (off + ((raw_sum + 64) >> 7).min(4)) as usize;
                                assert_eq!(lo_ctx::<0, 2>(&levels, 0, p), ref_ctx);
                            }
                            // CLS = 1 (`BWL = 2`, `ls = 8`): idx + {1, 2, 3, 4, 8}
                            levels.fill(0);
                            levels[1] = b0;
                            levels[2] = b1;
                            levels[3] = b2;
                            levels[4] = b3;
                            levels[8] = b4;
                            for coord in 0..=2u32 {
                                let c_off = 4 + coord * 5;
                                let p = (c_off << 11) | (1 << 10);
                                let ref_ctx =
                                    (26 + coord * 5 + ((raw_sum + 64) >> 7).min(4)) as usize;
                                assert_eq!(22 + lo_ctx::<1, 2>(&levels, 0, p), ref_ctx);
                            }
                            // CLS = 2 (`BWL = 2`, `ls = 8`): idx + {1, 8, 16, 24, 32}
                            levels.fill(0);
                            levels[1] = b0;
                            levels[8] = b1;
                            levels[16] = b2;
                            levels[24] = b3;
                            levels[32] = b4;
                            for coord in 0..=2u32 {
                                let c_off = 4 + coord * 5;
                                let p = (c_off << 11) | (1 << 10);
                                let ref_ctx =
                                    (26 + coord * 5 + ((raw_sum + 64) >> 7).min(4)) as usize;
                                assert_eq!(22 + lo_ctx::<2, 2>(&levels, 0, p), ref_ctx);
                            }
                        }
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
