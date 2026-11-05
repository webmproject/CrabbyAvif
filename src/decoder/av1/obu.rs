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

//! OBU parsing (spec 5.3) and collection of the data needed to decode the
//! first frame of a bitstream.

use crate::decoder::av1::bitreader::BitReader;
use crate::decoder::av1::error::{Error, Result};
use crate::decoder::av1::headers::{FrameHeader, SequenceHeader};

const OBU_SEQUENCE_HEADER: u32 = 1;
const OBU_TEMPORAL_DELIMITER: u32 = 2;
const OBU_FRAME_HEADER: u32 = 3;
const OBU_TILE_GROUP: u32 = 4;
const OBU_METADATA: u32 = 5;
const OBU_FRAME: u32 = 6;
const OBU_REDUNDANT_FRAME_HEADER: u32 = 7;
const OBU_TILE_LIST: u32 = 8;
const OBU_PADDING: u32 = 15;

struct Obu<'a> {
    obu_type: u32,
    temporal_id: u32,
    spatial_id: u32,
    has_extension: bool,
    payload: &'a [u8],
}

/// Splits the next OBU off `data`. Returns the OBU and the remaining data.
fn next_obu(data: &[u8]) -> Result<(Obu<'_>, &[u8])> {
    let mut r = BitReader::new(data);
    let _obu_forbidden_bit = r.bit()?;
    let obu_type = r.f(4)?;
    let has_extension = r.flag()?;
    let has_size = r.flag()?;
    let _reserved = r.bit()?;
    let (mut temporal_id, mut spatial_id) = (0, 0);
    if has_extension {
        temporal_id = r.f(3)?;
        spatial_id = r.f(2)?;
        let _ = r.f(3)?;
    }
    let size = if has_size {
        r.leb128()? as usize
    } else {
        data.len() - 1 - has_extension as usize
    };
    let start = r.byte_position();
    let end = start.checked_add(size).ok_or(Error::Truncated)?;
    if end > data.len() {
        return Err(Error::Truncated);
    }
    Ok((
        Obu {
            obu_type,
            temporal_id,
            spatial_id,
            has_extension,
            payload: &data[start..end],
        },
        &data[end..],
    ))
}

/// Everything needed to decode one frame: headers and the coded data of
/// every tile (indexed by tile number).
pub struct FrameData<'a> {
    pub seq: SequenceHeader,
    pub frame: FrameHeader,
    pub single_tile: &'a [u8],
    pub tiles: Vec<&'a [u8]>,
}

impl<'a> FrameData<'a> {
    #[inline(always)]
    pub fn tiles(&self) -> &[&'a [u8]] {
        if self.tiles.is_empty() {
            std::slice::from_ref(&self.single_tile)
        } else {
            &self.tiles
        }
    }
}

/// Parses a `tile_group_obu(sz)` payload (spec 5.11.1) and stores the
/// tile data slices.
fn parse_tile_group<'a>(
    data: &'a [u8],
    fh: &FrameHeader,
    single_tile: &mut &'a [u8],
    tiles: &mut Vec<&'a [u8]>,
) -> Result<bool> {
    let ti = &fh.tile_info;
    let num_tiles = ti.cols * ti.rows;
    if num_tiles == 1 {
        *single_tile = data;
        return Ok(true);
    }
    let mut r = BitReader::new(data);
    let tile_start_and_end_present = r.flag()?;
    let (tg_start, tg_end) = if !tile_start_and_end_present {
        (0, num_tiles - 1)
    } else {
        let bits = ti.cols_log2 + ti.rows_log2;
        (r.f(bits)? as usize, r.f(bits)? as usize)
    };
    if tg_end >= num_tiles || tg_start > tg_end {
        return Err(Error::Invalid("tile group range"));
    }
    // Tile groups must arrive in order (tg_start == TileNum).
    if tg_start != tiles.len() {
        return Err(Error::Invalid("tile group start"));
    }
    r.byte_align();
    let mut pos = r.byte_position();
    for tile_num in tg_start..=tg_end {
        let size = if tile_num == tg_end {
            data.len().checked_sub(pos).ok_or(Error::Truncated)?
        } else {
            let n = ti.tile_size_bytes as usize;
            let bytes = data.get(pos..pos + n).ok_or(Error::Truncated)?;
            let mut s = 0usize;
            for (i, b) in bytes.iter().enumerate() {
                s |= (*b as usize) << (8 * i);
            }
            pos += n;
            s + 1
        };
        let tile = data.get(pos..pos + size).ok_or(Error::Truncated)?;
        tiles.push(tile);
        pos += size;
    }
    Ok(tg_end == num_tiles - 1)
}

/// Validates a `metadata_obu()` payload (spec 5.8.1) to match `dav1d`.
fn parse_metadata(data: &[u8]) -> Result<()> {
    const OBU_META_HDR_CLL: u64 = 1;
    const OBU_META_HDR_MDCV: u64 = 2;
    let mut r = BitReader::new(data);
    match r.leb128()? {
        OBU_META_HDR_CLL => {
            let _ = r.f(32)?;
            let _ = r.bit()?;
        }
        OBU_META_HDR_MDCV => {
            for _ in 0..6 {
                let _ = r.f(32)?;
            }
            let _ = r.bit()?;
        }
        _ => {}
    }
    Ok(())
}

/// Parses OBUs until the first frame is complete.
pub fn parse_first_frame(mut data: &[u8]) -> Result<FrameData<'_>> {
    let mut seq: Option<SequenceHeader> = None;
    let mut frame: Option<FrameHeader> = None;
    let mut single_tile: &[u8] = &[];
    let mut tiles: Vec<&[u8]> = Vec::new();
    let mut complete = false;
    while !data.is_empty() {
        let (obu, rest) = next_obu(data)?;
        data = rest;
        if obu.obu_type != OBU_SEQUENCE_HEADER
            && obu.obu_type != OBU_TEMPORAL_DELIMITER
            && obu.has_extension
        {
            if let Some(s) = &seq {
                let idc = s.operating_point_idc;
                if idc != 0 {
                    let in_t = (idc >> obu.temporal_id) & 1 != 0;
                    let in_s = (idc >> (obu.spatial_id + 8)) & 1 != 0;
                    if !in_t || !in_s {
                        continue;
                    }
                }
            }
        }
        match obu.obu_type {
            OBU_SEQUENCE_HEADER => {
                let new_seq = SequenceHeader::parse(obu.payload)?;
                if seq.as_ref() != Some(&new_seq) {
                    frame = None;
                    single_tile = &[];
                    tiles.clear();
                }
                seq = Some(new_seq);
            }
            OBU_FRAME_HEADER | OBU_REDUNDANT_FRAME_HEADER | OBU_FRAME => {
                let s = seq
                    .as_ref()
                    .ok_or(Error::Invalid("frame before sequence header"))?;
                if obu.obu_type == OBU_REDUNDANT_FRAME_HEADER && frame.is_some() {
                    continue;
                }
                single_tile = &[];
                tiles.clear();
                let (fh, header_bits) =
                    FrameHeader::parse(obu.payload, s, obu.temporal_id, obu.spatial_id)?;
                let num_tiles = fh.tile_info.cols * fh.tile_info.rows;
                if num_tiles > 1 {
                    tiles.reserve(num_tiles);
                }
                if obu.obu_type == OBU_FRAME {
                    let offset = header_bits.div_ceil(8);
                    let tg = obu.payload.get(offset..).ok_or(Error::Truncated)?;
                    if parse_tile_group(tg, &fh, &mut single_tile, &mut tiles)? {
                        return Ok(FrameData {
                            seq: seq.take().ok_or(Error::Invalid("no sequence header"))?,
                            frame: fh,
                            single_tile,
                            tiles,
                        });
                    }
                } else if header_bits >= obu.payload.len() * 8 {
                    return Err(Error::Truncated);
                }
                frame = Some(fh);
            }
            OBU_TILE_GROUP => {
                let fh = frame
                    .as_ref()
                    .ok_or(Error::Invalid("tile group before frame header"))?;
                if parse_tile_group(obu.payload, fh, &mut single_tile, &mut tiles)? {
                    complete = true;
                    break;
                }
            }
            OBU_METADATA => parse_metadata(obu.payload)?,
            OBU_TEMPORAL_DELIMITER | OBU_PADDING => {}
            OBU_TILE_LIST => return Err(Error::Unsupported("large scale tile")),
            _ => {}
        }
    }
    let seq = seq.ok_or(Error::Invalid("no sequence header"))?;
    let frame = frame.ok_or(Error::Invalid("no frame header"))?;
    if !complete {
        return Err(Error::Truncated);
    }
    Ok(FrameData {
        seq,
        frame,
        single_tile,
        tiles,
    })
}
