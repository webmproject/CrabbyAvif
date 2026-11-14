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
#[cfg(any(feature = "png", feature = "jpeg"))]
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

#[cfg(any(feature = "png", feature = "jpeg"))]
pub(crate) fn get_orientation_offset(exif: &[u8]) -> AvifResult<Option<usize>> {
    let mut temp_stream = IStream::create(exif);
    let tiff_offset = usize_from_u32(parse_exif_tiff_header_offset(&mut temp_stream)?)?;
    let tiff_data = exif
        .get(tiff_offset..)
        .ok_or(AvifError::InvalidExifPayload)?;
    let little_endian = tiff_data.first() == Some(&b'I');
    let mut stream = IStream::create(tiff_data);
    stream
        .skip(4)
        .map_err(AvifError::map_invalid_exif_payload)?; // skip tiff header

    let read_u16 = |stream: &mut IStream| -> AvifResult<u16> {
        (if little_endian { stream.read_u16_le() } else { stream.read_u16() })
            .map_err(AvifError::map_invalid_exif_payload)
    };
    let read_u32 = |stream: &mut IStream| -> AvifResult<u32> {
        (if little_endian { stream.read_u32_le() } else { stream.read_u32() })
            .map_err(AvifError::map_invalid_exif_payload)
    };

    let offset_to_0th_ifd = read_u32(&mut stream)?;
    stream.offset = usize_from_u32(offset_to_0th_ifd)?;

    let field_count = read_u16(&mut stream)?;
    for _ in 0..field_count {
        let tag = read_u16(&mut stream)?;
        let field_type = read_u16(&mut stream)?;
        let count = read_u32(&mut stream)?;
        let value_offset = read_u16(&mut stream)?;
        stream
            .skip(2)
            .map_err(AvifError::map_invalid_exif_payload)?;

        // Orientation tag is 0x0112, type is SHORT (3), count is 1.
        if tag == 0x0112 && field_type == 3 && count == 1 && (1..=8).contains(&value_offset) {
            // Offset to the least meaningful byte of value_offset.
            // In a 12-byte field, the value/offset starts at byte 8.
            // If it fits in 2 or 4 bytes, it's stored directly there.
            // Our stream.offset is at the end of the 12-byte field (after skip(2)).
            let delta = if little_endian { 4 } else { 3 };
            let rel_offset = checked_sub!(stream.offset, delta)?;
            let offset = checked_add!(tiff_offset, rel_offset)?;
            return if exif.get(offset).is_some() {
                Ok(Some(offset))
            } else {
                AvifError::invalid_exif_payload()
            };
        }
    }
    Ok(None)
}

/// The return value is a tuple containing (irot_angle, imir_axis).
#[cfg(feature = "jpeg")]
pub(crate) fn get_orientation(exif: &[u8]) -> AvifResult<(Option<u8>, Option<u8>)> {
    Ok(match get_orientation_offset(exif) {
        Ok(Some(offset)) => match exif.get(offset).copied().unwrap_or(0) {
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
            let slot = exif.get_mut(offset).ok_or(AvifError::InvalidExifPayload)?;
            *slot = orientation;
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

// Searches for the HDR headroom in the Exif metadata for JPEGs captured on iPhones.
// References:
// https://developer.apple.com/documentation/appkit/applying-apple-hdr-effect-to-your-photos
// https://www.media.mit.edu/pia/Research/deepview/exif.html
// https://www.cipa.jp/std/documents/download_e.html?CIPA_DC-008-2024-E
// https://exiftool.org/TagNames/EXIF.html
// Exif metadata consists of a list of IFDs (Image File Directory), each containing a list of tags.
// The first IFD (IFD0) is expected to contain a tag called ExifOffset (id 0x8769) which contains
// the offset to another IFD, the Exif IFD.
// The Exif IFD is expected to contain a tag called MakerNotes (id 0x927c) which contains an offset
// to proprietary notes data specific to the camera vendor. In the case of Apple, it consists of a
// header starting with 'Apple iOS'  etc. followed by another IFD. This last IFD contains the tags
// 33 and 48 which are used to compute the headroom.
#[cfg(feature = "jpeg")]
pub(crate) fn apple_headroom(exif: &[u8]) -> AvifResult<Option<f64>> {
    if exif.is_empty() {
        return Ok(None);
    }
    let tiff_offset = match parse_exif_tiff_header_offset(&mut IStream::create(exif)) {
        Ok(offset) => offset as usize,
        Err(_) => return Ok(None),
    };
    let Some(tiff_data) = exif.get(tiff_offset..) else {
        return Ok(None);
    };
    if tiff_data.len() < 8 {
        return Ok(None);
    }

    let read_u16 = |stream: &mut IStream, is_big_endian: bool| -> AvifResult<u16> {
        if is_big_endian {
            stream.read_u16()
        } else {
            stream.read_u16_le()
        }
    };
    let read_u32 = |stream: &mut IStream, is_big_endian: bool| -> AvifResult<u32> {
        if is_big_endian {
            stream.read_u32()
        } else {
            stream.read_u32_le()
        }
    };

    let mut is_big_endian = tiff_data[0] == b'M';
    let mut stream = IStream::create(tiff_data);
    stream.skip(4)?; // skip tiff header
    let mut offset_to_ifd = read_u32(&mut stream, is_big_endian)?;
    let mut maker33: Option<f64> = None;
    let mut maker48: Option<f64> = None;
    let mut in_apple_maker_notes = false;

    let mut num_ifds = 0;
    while offset_to_ifd != 0 && num_ifds < 3 {
        num_ifds += 1;
        stream.offset = offset_to_ifd as usize;
        let mut offset_to_next_ifd_already_set = false;
        let field_count = read_u16(&mut stream, is_big_endian)?;
        for _ in 0..field_count {
            let tag_id = read_u16(&mut stream, is_big_endian)?;
            let data_format = read_u16(&mut stream, is_big_endian)?;
            stream.skip_u32()?; // num_components.
            let tag_data = read_u32(&mut stream, is_big_endian)?;

            if tag_id == 0x8769 {
                // Exif Offset
                stream.rewind(4)?;
                break;
            } else if tag_id == 0x927c {
                // Maker Notes
                let maker_notes_offset = usize_from_u32(tag_data)?;
                if let Some(end) = maker_notes_offset.checked_add(14) {
                    if tiff_data.get(maker_notes_offset..end) == Some(b"Apple iOS\0\0\x01MM") {
                        offset_to_ifd = u32_from_usize(end)?;
                        in_apple_maker_notes = true;
                        offset_to_next_ifd_already_set = true;
                        is_big_endian = true;
                        break;
                    }
                }
            } else if in_apple_maker_notes && (tag_id == 33 || tag_id == 48) && data_format == 10 {
                let start_of_maker_notes = checked_sub!(usize_from_u32(offset_to_ifd)?, 14)?;
                let tag_offset = usize_from_u32(tag_data)?;
                if let Some(tmp_offset) = start_of_maker_notes.checked_add(tag_offset) {
                    if let Some(slice) = tiff_data.get(tmp_offset..) {
                        if slice.len() >= 8 {
                            let mut rational_stream = IStream::create(slice);
                            let numerator = rational_stream.read_i32()?;
                            let denominator = read_u32(&mut rational_stream, is_big_endian)?;
                            if denominator != 0 {
                                let v = numerator as f64 / denominator as f64;
                                if tag_id == 33 {
                                    maker33 = Some(v);
                                } else {
                                    maker48 = Some(v);
                                }
                            }
                        }
                    }
                }
            }
        }

        if !offset_to_next_ifd_already_set {
            if in_apple_maker_notes || stream.bytes_left()? < 4 {
                break;
            }
            offset_to_ifd = read_u32(&mut stream, is_big_endian)?;
        }
    }

    if maker33.is_none() && maker48.is_none() {
        return Ok(None);
    }
    let m33 = maker33.unwrap_or(0.0);
    let m48 = maker48.unwrap_or(0.0);

    Ok(Some(if m33 < 1.0 {
        if m48 <= 0.01 {
            -20.0 * m48 + 1.8
        } else {
            -0.101 * m48 + 1.601
        }
    } else if m48 <= 0.01 {
        -70.0 * m48 + 3.0
    } else {
        -0.303 * m48 + 2.303
    }))
}
#[cfg(all(test, any(feature = "png", feature = "jpeg")))]
mod tests {
    use super::*;

    // Little endian TIFF header followed by a 0th IFD containing an orientation tag.
    const EXIF: [u8; 26] = [
        b'I', b'I', 42, 0, // TIFF header
        8, 0, 0, 0, // Offset to the 0th IFD
        1, 0, // Number of directory entries
        0x12, 0x01, // Tag: Orientation
        3, 0, // Type: SHORT
        1, 0, 0, 0, // Count
        6, 0, 0, 0, // Value
        0, 0, 0, 0, // Offset to the next IFD
    ];
    const ORIENTATION_OFFSET: usize = 18;

    #[test]
    fn orientation_offset() {
        assert_eq!(get_orientation_offset(&EXIF), Ok(Some(ORIENTATION_OFFSET)));
        // The offset to the next IFD is not needed.
        assert_eq!(
            get_orientation_offset(&EXIF[..22]),
            Ok(Some(ORIENTATION_OFFSET))
        );
    }

    #[test]
    fn orientation_offset_truncated_exif() {
        for size in 0..22 {
            assert_eq!(
                get_orientation_offset(&EXIF[..size]),
                Err(AvifError::InvalidExifPayload),
                "size {size}"
            );
        }
    }

    #[test]
    fn orientation_offset_invalid_ifd_offset() {
        let mut exif = EXIF;
        exif[4] = 0xFF; // Offset to the 0th IFD beyond the end of the Exif payload.
        assert_eq!(
            get_orientation_offset(&exif),
            Err(AvifError::InvalidExifPayload)
        );
    }

    #[cfg(feature = "png")]
    #[test]
    fn set_orientation_truncated_exif() {
        let mut exif = EXIF[..20].to_vec();
        assert_eq!(
            set_orientation(&mut exif, 1),
            Err(AvifError::InvalidExifPayload)
        );
        assert_eq!(exif, EXIF[..20]);
    }

    fn push_ifd_entry_be(out: &mut Vec<u8>, tag: u16, format: u16, count: u32, data: u32) {
        out.extend_from_slice(&tag.to_be_bytes());
        out.extend_from_slice(&format.to_be_bytes());
        out.extend_from_slice(&count.to_be_bytes());
        out.extend_from_slice(&data.to_be_bytes());
    }

    #[cfg(feature = "jpeg")]
    #[test]
    fn apple_headroom_with_next_ifd_before_maker_notes_does_not_panic() {
        for ifd0_offset in [8u32, 16u32] {
            let mut exif = b"MM\0*".to_vec();
            exif.extend_from_slice(&ifd0_offset.to_be_bytes());
            exif.resize(ifd0_offset as usize, 0);
            // 0th IFD: a rational tag 33 and the maker notes tag pointing past IFD0.
            let maker_notes_offset = ifd0_offset + 2 + 24 + 4;
            exif.extend_from_slice(&2u16.to_be_bytes());
            push_ifd_entry_be(&mut exif, 33, 10, 1, 0);
            push_ifd_entry_be(&mut exif, 0x927c, 7, 14, maker_notes_offset);
            exif.extend_from_slice(&0u32.to_be_bytes());
            // Apple maker notes header followed by an empty IFD whose next_ifd points back to IFD0.
            exif.extend_from_slice(b"Apple iOS\0\0\x01MM");
            exif.extend_from_slice(&0u16.to_be_bytes());
            exif.extend_from_slice(&ifd0_offset.to_be_bytes());
            assert_eq!(apple_headroom(&exif), Ok(None));
        }
    }

    #[cfg(feature = "jpeg")]
    #[test]
    fn apple_headroom_supports_negative_srational_numerator() {
        let mut exif = b"MM\0*".to_vec();
        exif.extend_from_slice(&8u32.to_be_bytes());
        // IFD0 at 8: 1 entry (MakerNotes 0x927c -> offset 26).
        exif.extend_from_slice(&1u16.to_be_bytes());
        push_ifd_entry_be(&mut exif, 0x927c, 7, 14, 26);
        exif.extend_from_slice(&0u32.to_be_bytes());
        // Apple MakerNotes at 26 (14-byte header -> IFD at 40):
        // 2 entries: tag 33 -> rel offset 44 (abs 70), tag 48 -> rel offset 52 (abs 78).
        exif.extend_from_slice(b"Apple iOS\0\0\x01MM");
        exif.extend_from_slice(&2u16.to_be_bytes());
        push_ifd_entry_be(&mut exif, 33, 10, 1, 44);
        push_ifd_entry_be(&mut exif, 48, 10, 1, 52);
        exif.extend_from_slice(&0u32.to_be_bytes());
        assert_eq!(exif.len(), 70);
        // Tag 33 = -1 / 2 (-0.5 < 1.0), Tag 48 = 0 / 100 (0.0 <= 0.01) -> headroom = 1.8.
        exif.extend_from_slice(&(-1i32).to_be_bytes());
        exif.extend_from_slice(&2u32.to_be_bytes());
        exif.extend_from_slice(&0i32.to_be_bytes());
        exif.extend_from_slice(&100u32.to_be_bytes());
        let hr = apple_headroom(&exif).unwrap().expect("expected headroom");
        assert!((hr - 1.8).abs() < 1e-9);
    }

    #[cfg(feature = "jpeg")]
    #[test]
    fn orientation_of_truncated_payloads() {
        let mut exif = b"MM\0*".to_vec();
        exif.extend_from_slice(&8u32.to_be_bytes());
        exif.extend_from_slice(&1u16.to_be_bytes());
        push_ifd_entry_be(&mut exif, 0x0112, 3, 1, 6 << 16);
        assert_eq!(exif.len(), 22);
        assert_eq!(get_orientation(&exif), Ok((Some(3), None)));
        for len in 0..exif.len() {
            assert!(get_orientation_offset(&exif[..len]).is_err());
            assert_eq!(get_orientation(&exif[..len]), Ok((None, None)));
            assert!(!matches!(apple_headroom(&exif[..len]), Ok(Some(_))));
        }
    }

    #[cfg(feature = "png")]
    #[test]
    fn set_orientation_be_le_and_truncated_entry() {
        // Big-endian TIFF: orientation at byte 19 (22 - 3).
        let mut be = b"MM\0*".to_vec();
        be.extend_from_slice(&8u32.to_be_bytes());
        be.extend_from_slice(&1u16.to_be_bytes());
        push_ifd_entry_be(&mut be, 0x0112, 3, 1, 6 << 16);
        assert_eq!(get_orientation_offset(&be), Ok(Some(19)));
        assert_eq!(set_orientation(&mut be, 1), Ok(()));
        assert_eq!(be[19], 1);

        // Truncated at bytes 20..21 (missing trailing 2 bytes of 12-byte IFD entry) must fail
        // without overwriting `count`.
        let mut trunc = be[..20].to_vec();
        let before = trunc.clone();
        assert!(set_orientation(&mut trunc, 1).is_err());
        assert_eq!(trunc, before);

        // Little-endian TIFF ("II*\0"): orientation at byte 18 (22 - 4).
        let mut le = b"II*\0".to_vec();
        le.extend_from_slice(&8u32.to_le_bytes());
        le.extend_from_slice(&1u16.to_le_bytes());
        le.extend_from_slice(&0x0112u16.to_le_bytes());
        le.extend_from_slice(&3u16.to_le_bytes());
        le.extend_from_slice(&1u32.to_le_bytes());
        le.extend_from_slice(&6u16.to_le_bytes());
        le.extend_from_slice(&0u16.to_le_bytes());
        assert_eq!(get_orientation_offset(&le), Ok(Some(18)));
        assert_eq!(set_orientation(&mut le, 1), Ok(()));
        assert_eq!(le[18], 1);
    }
}
