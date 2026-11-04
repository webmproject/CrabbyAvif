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

//! Two-stage pipeline data structures for single-tile frames.
//! Entropy parsing runs on one thread (Stage P) and reconstruction on
//! another (Stage R) using a compact log of decoded block records and
//! transform coefficients.

use crate::decoder::av1::decode::block::{bh4, bw4, Block, Scratch};
use crate::decoder::av1::decode::state::FrameCtx;
use crate::decoder::av1::pixel::Pixel;
use crate::decoder::av1::types::{BlockSize, IntraMode, TxSize, TxType};

pub const FLAG_HAS_CHROMA: u16 = 1 << 0;
pub const FLAG_AVAIL_U: u16 = 1 << 1;
pub const FLAG_AVAIL_L: u16 = 1 << 2;
pub const FLAG_AVAIL_U_CHROMA: u16 = 1 << 3;
pub const FLAG_AVAIL_L_CHROMA: u16 = 1 << 4;
pub const FLAG_SKIP: u16 = 1 << 5;
pub const FLAG_LOSSLESS: u16 = 1 << 6;
pub const FLAG_USE_FILTER_INTRA: u16 = 1 << 7;
pub const FLAG_FILTER_TYPE_Y: u16 = 1 << 8;
pub const FLAG_FILTER_TYPE_UV: u16 = 1 << 9;

#[derive(Clone, Copy, Default)]
#[repr(C, align(8))]
pub struct BlockRec {
    pub mi_row: u16,
    pub mi_col: u16,
    pub flags: u16,
    pub bsize: BlockSize,
    pub tx_size: TxSize,
    pub y_mode: IntraMode,
    pub uv_mode: IntraMode,
    pub angle_delta_y: i8,
    pub angle_delta_uv: i8,
    pub filter_intra_mode: u8,
    pub cfl_alpha_u: i8,
    pub cfl_alpha_v: i8,
    pub palette_sizes: u8,
}

const _: () = assert!(std::mem::size_of::<BlockRec>() == 16);

#[derive(Clone, Copy, Default)]
pub struct PaletteRec {
    pub palette_y: [u16; 8],
    pub palette_u: [u16; 8],
    pub palette_v: [u16; 8],
    pub map_y_w: u8,
    pub map_y_h: u8,
    pub map_uv_w: u8,
    pub map_uv_h: u8,
    pub map_offset: u32,
}

#[derive(Clone, Copy, Default)]
pub struct TxRecHeader {
    pub eob: u16,
    pub tx_type: TxType,
    pub count: u16,
}

#[derive(Default)]
pub struct LogChunk {
    /// Superblocks in this chunk: (mi_row, mi_col, num_blocks)
    pub sbs: Vec<(usize, usize, usize)>,
    pub blocks: Vec<BlockRec>,
    pub palettes: Vec<PaletteRec>,
    pub palette_bytes: Vec<u8>,
    pub tx_headers: Vec<TxRecHeader>,
    pub tx_entries: Vec<(u16, i32)>,
}

impl LogChunk {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.sbs.is_empty()
    }

    pub fn approx_size(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<BlockRec>()
            + self.palettes.len() * std::mem::size_of::<PaletteRec>()
            + self.palette_bytes.len()
            + self.tx_headers.len() * std::mem::size_of::<TxRecHeader>()
            + self.tx_entries.len() * std::mem::size_of::<(u16, i32)>()
    }

    pub fn clear(&mut self) {
        self.sbs.clear();
        self.blocks.clear();
        self.palettes.clear();
        self.palette_bytes.clear();
        self.tx_headers.clear();
        self.tx_entries.clear();
    }

    #[cold]
    #[inline(never)]
    fn record_palette(&mut self, b: &Block, sc: &Scratch, f: &FrameCtx) {
        let map_offset = self.palette_bytes.len() as u32;
        let block_h = b.bh4 * 4;
        let block_w = b.bw4 * 4;
        let (mut yw, mut yh) = (0, 0);
        if b.palette_size_y > 0 {
            yw = block_w;
            yh = block_h;
            for row in &sc.color_map_y[..yh] {
                self.palette_bytes.extend_from_slice(&row[..yw]);
            }
        }
        let (mut uvw, mut uvh) = (0, 0);
        if b.palette_size_uv > 0 {
            uvw = block_w >> f.ss_x;
            uvh = block_h >> f.ss_y;
            if uvw < 4 {
                uvw += 2;
            }
            if uvh < 4 {
                uvh += 2;
            }
            for row in &sc.color_map_uv[..uvh] {
                self.palette_bytes.extend_from_slice(&row[..uvw]);
            }
        }
        self.palettes.push(PaletteRec {
            palette_y: b.palette_y,
            palette_u: b.palette_u,
            palette_v: b.palette_v,
            map_y_w: yw as u8,
            map_y_h: yh as u8,
            map_uv_w: uvw as u8,
            map_uv_h: uvh as u8,
            map_offset,
        });
    }

    pub fn record_block(
        &mut self,
        b: &Block,
        sc: &Scratch,
        f: &FrameCtx,
        filter_type_y: bool,
        filter_type_uv: bool,
    ) {
        let flags = (b.has_chroma as u16 * FLAG_HAS_CHROMA)
            | (b.avail_u as u16 * FLAG_AVAIL_U)
            | (b.avail_l as u16 * FLAG_AVAIL_L)
            | (b.avail_u_chroma as u16 * FLAG_AVAIL_U_CHROMA)
            | (b.avail_l_chroma as u16 * FLAG_AVAIL_L_CHROMA)
            | (b.skip as u16 * FLAG_SKIP)
            | (b.lossless as u16 * FLAG_LOSSLESS)
            | (b.use_filter_intra as u16 * FLAG_USE_FILTER_INTRA)
            | (filter_type_y as u16 * FLAG_FILTER_TYPE_Y)
            | (filter_type_uv as u16 * FLAG_FILTER_TYPE_UV);

        let palette_sizes = (b.palette_size_y as u8) | ((b.palette_size_uv as u8) << 4);
        if palette_sizes != 0 {
            self.record_palette(b, sc, f);
        }

        self.blocks.push(BlockRec {
            mi_row: b.mi_row as u16,
            mi_col: b.mi_col as u16,
            flags,
            bsize: b.bsize,
            tx_size: b.tx_size,
            y_mode: b.y_mode,
            uv_mode: b.uv_mode,
            angle_delta_y: b.angle_delta_y as i8,
            angle_delta_uv: b.angle_delta_uv as i8,
            filter_intra_mode: b.filter_intra_mode,
            cfl_alpha_u: b.cfl_alpha_u as i8,
            cfl_alpha_v: b.cfl_alpha_v as i8,
            palette_sizes,
        });
    }
}

#[derive(Default)]
pub struct LogCursor {
    pub block_idx: usize,
    pub palette_idx: usize,
    pub tx_header_idx: usize,
    pub tx_entry_idx: usize,
}

impl LogCursor {
    pub fn reset(&mut self) {
        self.block_idx = 0;
        self.palette_idx = 0;
        self.tx_header_idx = 0;
        self.tx_entry_idx = 0;
    }
}

pub enum DecoderStage<'l> {
    Fused,
    Parse(&'l mut LogChunk),
    Recon(&'l LogChunk, &'l mut LogCursor),
}

impl<'l> DecoderStage<'l> {
    #[inline(always)]
    pub fn is_recon(&self) -> bool {
        matches!(self, DecoderStage::Recon(..))
    }
}

pub(crate) enum PipeMsg<T: Pixel> {
    Chunk(LogChunk),
    RowEnd {
        r: usize,
        last: bool,
        bufs: Option<Box<super::BandBufs<T>>>,
        chunk: Option<LogChunk>,
    },
}

impl<'a, 'b, 's, T: Pixel> super::block::SbDecoder<'a, 'b, 's, T> {
    /// Restores the mode info of a block recorded by Stage P (except for
    /// the palette, see `decode_recon_palette_block`).
    #[inline(always)]
    fn unpack_block_rec(&mut self, rec: &BlockRec) {
        let flags = rec.flags;
        self.b.mi_row = rec.mi_row as usize;
        self.b.mi_col = rec.mi_col as usize;
        self.b.bsize = rec.bsize;
        self.b.bw4 = bw4(rec.bsize);
        self.b.bh4 = bh4(rec.bsize);
        self.b.tx_size = rec.tx_size;
        self.b.y_mode = rec.y_mode;
        self.b.uv_mode = rec.uv_mode;
        self.b.angle_delta_y = rec.angle_delta_y as i32;
        self.b.angle_delta_uv = rec.angle_delta_uv as i32;
        self.b.filter_intra_mode = rec.filter_intra_mode;
        self.b.cfl_alpha_u = rec.cfl_alpha_u as i32;
        self.b.cfl_alpha_v = rec.cfl_alpha_v as i32;
        self.b.has_chroma = (flags & FLAG_HAS_CHROMA) != 0;
        self.b.avail_u = (flags & FLAG_AVAIL_U) != 0;
        self.b.avail_l = (flags & FLAG_AVAIL_L) != 0;
        self.b.avail_u_chroma = (flags & FLAG_AVAIL_U_CHROMA) != 0;
        self.b.avail_l_chroma = (flags & FLAG_AVAIL_L_CHROMA) != 0;
        self.b.skip = (flags & FLAG_SKIP) != 0;
        self.b.lossless = (flags & FLAG_LOSSLESS) != 0;
        self.b.use_filter_intra = (flags & FLAG_USE_FILTER_INTRA) != 0;
        self.filter_type = [
            (flags & FLAG_FILTER_TYPE_Y) != 0,
            (flags & FLAG_FILTER_TYPE_UV) != 0,
        ];
    }

    /// `decode_recon_block` for a block with a palette. Only called in the
    /// `DecoderStage::Recon` stage.
    #[cold]
    #[inline(never)]
    fn decode_recon_palette_block(
        &mut self,
        rec: &BlockRec,
        chunk: &LogChunk,
    ) -> crate::decoder::av1::error::Result<()> {
        let prec = match &mut self.stage {
            DecoderStage::Recon(_, cursor) => {
                let p = *chunk.palettes.get(cursor.palette_idx).ok_or(
                    crate::decoder::av1::error::Error::Invalid(
                        "pipeline palette cursor out of bounds",
                    ),
                )?;
                cursor.palette_idx += 1;
                p
            }
            _ => return Ok(()),
        };
        self.unpack_block_rec(rec);
        self.b.palette_size_y = (rec.palette_sizes & 0x0F) as usize;
        self.b.palette_size_uv = (rec.palette_sizes >> 4) as usize;
        self.b.palette_y = prec.palette_y;
        self.b.palette_u = prec.palette_u;
        self.b.palette_v = prec.palette_v;
        let mut offset = prec.map_offset as usize;
        if prec.map_y_w > 0 {
            let (w, h) = (prec.map_y_w as usize, prec.map_y_h as usize);
            for row in &mut self.sc.color_map_y[..h] {
                if offset + w <= chunk.palette_bytes.len() {
                    row[..w].copy_from_slice(&chunk.palette_bytes[offset..offset + w]);
                }
                offset += w;
            }
        }
        if prec.map_uv_w > 0 {
            let (w, h) = (prec.map_uv_w as usize, prec.map_uv_h as usize);
            for row in &mut self.sc.color_map_uv[..h] {
                if offset + w <= chunk.palette_bytes.len() {
                    row[..w].copy_from_slice(&chunk.palette_bytes[offset..offset + w]);
                }
                offset += w;
            }
        }
        self.setup_intra();
        self.residual()
            .map_err(crate::decoder::av1::error::Error::Invalid)?;
        self.b.palette_size_y = 0;
        self.b.palette_size_uv = 0;
        Ok(())
    }

    pub(crate) fn decode_recon_block(
        &mut self,
        chunk: &LogChunk,
    ) -> crate::decoder::av1::error::Result<()> {
        let rec = match &mut self.stage {
            DecoderStage::Recon(_, cursor) => {
                let r = *chunk.blocks.get(cursor.block_idx).ok_or(
                    crate::decoder::av1::error::Error::Invalid(
                        "pipeline block cursor out of bounds",
                    ),
                )?;
                cursor.block_idx += 1;
                r
            }
            _ => return Ok(()),
        };
        if rec.palette_sizes != 0 {
            return self.decode_recon_palette_block(&rec, chunk);
        }
        self.unpack_block_rec(&rec);
        self.setup_intra();
        self.residual()
            .map_err(crate::decoder::av1::error::Error::Invalid)?;
        Ok(())
    }
}
