// Copyright 2024 Google LLC
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

use crate::internal_utils::stream::*;
#[cfg(feature = "png")]
use crate::internal_utils::*;
use crate::parser::mp4box::BoxSize;
use crate::*;

pub(crate) fn parse_exif_tiff_header_offset(stream: &mut IStream) -> AvifResult<u32> {
    const TIFF_HEADER_BE: u32 = 0x4D4D002A; // MM0* (read as a big endian u32)
    const TIFF_HEADER_LE: u32 = 0x49492A00; // II*0 (read as a big endian u32)
    let mut expected_offset: u32 = 0;
    let mut size = u32::try_from(stream.bytes_left()?).unwrap_or(u32::MAX);
    while size > 0 {
        let value = stream
            .read_u32()
            .map_err(AvifError::map_invalid_exif_payload)?;
        if value == TIFF_HEADER_BE || value == TIFF_HEADER_LE {
            stream.rewind(4)?;
            return Ok(expected_offset);
        }
        stream.rewind(3)?;
        checked_decr!(size, 1);
        checked_incr!(expected_offset, 1);
    }
    // Could not find the TIFF header.
    AvifError::invalid_exif_payload()
}

pub(crate) fn parse(stream: &mut IStream) -> AvifResult<()> {
    // unsigned int(32) exif_tiff_header_offset;
    let offset = stream
        .read_u32()
        .map_err(AvifError::map_invalid_exif_payload)?;

    let bytes_left = stream.bytes_left()?;
    let mut sub_stream = stream.sub_stream(&BoxSize::FixedSize(bytes_left))?;
    let expected_offset = parse_exif_tiff_header_offset(&mut sub_stream)?;
    if offset != expected_offset {
        return AvifError::invalid_exif_payload();
    }
    stream.rewind(bytes_left)?;
    Ok(())
}

#[cfg(feature = "png")]
pub(crate) fn get_orientation_offset(exif: &[u8]) -> AvifResult<Option<usize>> {
    let mut temp_stream = IStream::create(exif);
    let tiff_offset = usize_from_u32(parse_exif_tiff_header_offset(&mut temp_stream)?)?;
    let tiff_data = &exif[tiff_offset..];
    let little_endian = tiff_data[0] == b'I';
    let mut stream = IStream::create(tiff_data);
    let _ = stream.skip(4); // skip tiff header

    let offset_to_0th_ifd =
        if little_endian { stream.read_u32_le()? } else { stream.read_u32()? };
    stream.offset = usize_from_u32(offset_to_0th_ifd)?;

    let field_count = if little_endian { stream.read_u16_le()? } else { stream.read_u16()? };
    for _ in 0..field_count {
        let (tag, field_type, count, value_offset) = if little_endian {
            (
                stream.read_u16_le()?,
                stream.read_u16_le()?,
                stream.read_u32_le()?,
                stream.read_u16_le()?,
            )
        } else {
            (
                stream.read_u16()?,
                stream.read_u16()?,
                stream.read_u32()?,
                stream.read_u16()?,
            )
        };
        let _ = stream.skip(2);

        // Orientation tag is 0x0112, type is SHORT (3), count is 1.
        if tag == 0x0112 && field_type == 3 && count == 1 && (1..=8).contains(&value_offset) {
            // Offset to the least meaningful byte of value_offset.
            // In a 12-byte field, the value/offset starts at byte 8.
            // If it fits in 2 or 4 bytes, it's stored directly there.
            // Our stream.offset is at the end of the 12-byte field (after skip(2)).
            return Ok(Some(
                tiff_offset + stream.offset - if little_endian { 4 } else { 3 },
            ));
        }
    }
    Ok(None)
}

/// The return value is a tuple containing (irot_angle, imir_axis).
#[cfg(feature = "jpeg")]
pub(crate) fn get_orientation(exif: &[u8]) -> AvifResult<(Option<u8>, Option<u8>)> {
    Ok(match get_orientation_offset(exif) {
        Ok(Some(offset)) => match exif[offset] {
            2 => (None, Some(1)),
            3 => (Some(2), None),
            4 => (None, Some(0)),
            5 => (Some(1), Some(0)),
            6 => (Some(3), None),
            7 => (Some(3), Some(0)),
            8 => (Some(1), None),
            _ => (None, None),
        },
        _ => (None, None),
    })
}

#[cfg(feature = "png")]
pub(crate) fn set_orientation(exif: &mut [u8], orientation: u8) -> AvifResult<()> {
    match get_orientation_offset(exif)? {
        Some(offset) => {
            exif[offset] = orientation;
            Ok(())
        }
        None => {
            if orientation == 1 {
                Ok(())
            } else {
                Err(AvifError::NotImplemented)
            }
        }
    }
}
