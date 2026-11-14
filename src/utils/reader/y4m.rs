// Copyright 2025 Google LLC
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

use crate::gainmap::GainMap;
use crate::image::*;
use crate::internal_utils::*;
use crate::utils::pixels::Pixels;
use crate::*;

use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;

use super::Config;
use super::Reader;

const Y4M_BUF_READER_CAP: usize = 64 * 1024;
const Y4M_MAX_LINE_BYTES: usize = 2048;

#[derive(Debug, Default)]
pub struct Y4MReader {
    width: u32,
    height: u32,
    depth: u8,
    has_alpha: bool,
    format: PixelFormat,
    range: YuvRange,
    chroma_sample_position: ChromaSamplePosition,
    timescale: u64,
    frame_duration: u64,
    reader: Option<BufReader<File>>,
    frame_header_buf: Vec<u8>,
}

fn parse_frame_header(header_bytes: &[u8], default_range: YuvRange) -> AvifResult<YuvRange> {
    if !header_bytes.ends_with(b"\n") {
        return AvifError::unknown_error("unterminated frame header");
    }
    let frame_marker = std::str::from_utf8(header_bytes).map_err(AvifError::map_unknown_error)?;
    let frame_header = frame_marker.trim_end_matches(['\r', '\n']);
    let mut parts = frame_header.split(' ');
    if parts.next() != Some("FRAME") {
        return AvifError::unknown_error("could not find frame marker");
    }
    let mut frame_range = default_range;
    for part in parts {
        if part == "XCOLORRANGE=FULL" {
            frame_range = YuvRange::Full;
        } else if part == "XCOLORRANGE=LIMITED" {
            frame_range = YuvRange::Limited;
        }
    }
    Ok(frame_range)
}

#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
/// # Safety
/// Caller must verify `avx2` support at runtime and pass `src.len() == dst.len() * 2`.
#[target_feature(enable = "avx2")]
unsafe fn clamp_le_u16_avx2(src: &[u8], dst: &mut [u16], max_sample: u16) {
    use std::arch::x86_64::*;
    let count = dst.len();
    let vmax = _mm256_set1_epi16(max_sample as i16);
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let mut i = 0usize;
    while i + 16 <= count {
        // SAFETY: `i + 16 <= count` and `src.len() == count * 2`, so `src_ptr.add(i * 2)` has
        // at least 32 readable bytes.
        let v = unsafe { _mm256_loadu_si256(src_ptr.add(i * 2) as *const __m256i) };
        let clamped = _mm256_min_epu16(v, vmax);
        // SAFETY: `i + 16 <= count` and `dst.len() == count`, so `dst_ptr.add(i)` has at least
        // 16 writable `u16` elements.
        unsafe {
            _mm256_storeu_si256(dst_ptr.add(i) as *mut __m256i, clamped);
        }
        i += 16;
    }
    if i + 8 <= count {
        let vmax128 = _mm256_castsi256_si128(vmax);
        // SAFETY: `i + 8 <= count` and `src.len() == count * 2`, so `src_ptr.add(i * 2)` has
        // at least 16 readable bytes.
        let v = unsafe { _mm_loadu_si128(src_ptr.add(i * 2) as *const __m128i) };
        let clamped = _mm_min_epu16(v, vmax128);
        // SAFETY: `i + 8 <= count` and `dst.len() == count`, so `dst_ptr.add(i)` has at least
        // 8 writable `u16` elements.
        unsafe {
            _mm_storeu_si128(dst_ptr.add(i) as *mut __m128i, clamped);
        }
        i += 8;
    }
    for j in i..count {
        dst[j] = u16::from_le_bytes([src[j * 2], src[j * 2 + 1]]).min(max_sample);
    }
}

fn clamp_le_u16_into(src: &[u8], dst: &mut [u16], max_sample: u16) {
    assert_eq!(src.len(), dst.len() * 2);
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    if dst.len() >= 16 && is_x86_feature_detected!("avx2") {
        // SAFETY: `avx2` support verified at runtime and `src.len() == dst.len() * 2` asserted.
        unsafe {
            clamp_le_u16_avx2(src, dst, max_sample);
        }
        return;
    }
    for (j, slot) in dst.iter_mut().enumerate() {
        *slot = u16::from_le_bytes([src[j * 2], src[j * 2 + 1]]).min(max_sample);
    }
}

fn read_and_clamp_u16(
    reader: &mut BufReader<File>,
    dst: &mut [u16],
    max_sample: u16,
) -> AvifResult<()> {
    let mut offset = 0usize;
    let mut pending_lo: Option<u8> = None;
    while offset < dst.len() {
        let avail = match reader.fill_buf() {
            Ok([]) => return AvifError::unknown_error("unexpected end of file"),
            Ok(buf) => buf,
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(AvifError::map_unknown_error(err)),
        };
        if let Some(lo) = pending_lo.take() {
            let hi = avail[0];
            reader.consume(1);
            dst[offset] = u16::from_le_bytes([lo, hi]).min(max_sample);
            offset += 1;
        } else if avail.len() >= 2 {
            let pairs = (avail.len() / 2).min(dst.len() - offset);
            let bytes = pairs * 2;
            clamp_le_u16_into(
                &avail[..bytes],
                &mut dst[offset..offset + pairs],
                max_sample,
            );
            reader.consume(bytes);
            offset += pairs;
        } else {
            pending_lo = Some(avail[0]);
            reader.consume(1);
        }
    }
    Ok(())
}

fn read_bounded_line(
    reader: &mut BufReader<File>,
    buf: &mut Vec<u8>,
    max_bytes: usize,
) -> AvifResult<usize> {
    buf.clear();
    let mut total = 0usize;
    loop {
        let avail = match reader.fill_buf() {
            Ok(b) => b,
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(AvifError::map_unknown_error(err)),
        };
        if avail.is_empty() {
            return Ok(total);
        }
        let rem_limit = max_bytes.saturating_add(1).saturating_sub(total);
        let search = &avail[..avail.len().min(rem_limit)];
        if let Some(nl_pos) = search.iter().position(|&b| b == b'\n') {
            let take = nl_pos + 1;
            buf.try_extend_from_slice(&search[..take])?;
            reader.consume(take);
            total += take;
            return Ok(total);
        }
        let take = search.len();
        if total + take > max_bytes {
            return AvifError::unknown_error("y4m header line exceeds maximum length");
        }
        buf.try_extend_from_slice(search)?;
        reader.consume(take);
        total += take;
    }
}

impl Y4MReader {
    fn parse_colorspace(&mut self, colorspace: &str) -> AvifResult<()> {
        (
            self.depth,
            self.format,
            self.chroma_sample_position,
            self.has_alpha,
        ) = match colorspace {
            "420jpeg" => (
                8,
                PixelFormat::Yuv420,
                ChromaSamplePosition::default(),
                false,
            ),
            "420mpeg2" => (
                8,
                PixelFormat::Yuv420,
                ChromaSamplePosition::Vertical,
                false,
            ),
            "420paldv" => (
                8,
                PixelFormat::Yuv420,
                ChromaSamplePosition::Colocated,
                false,
            ),
            "444p10" => (
                10,
                PixelFormat::Yuv444,
                ChromaSamplePosition::default(),
                false,
            ),
            "422p10" => (
                10,
                PixelFormat::Yuv422,
                ChromaSamplePosition::default(),
                false,
            ),
            "420p10" => (
                10,
                PixelFormat::Yuv420,
                ChromaSamplePosition::default(),
                false,
            ),
            "444p12" => (
                12,
                PixelFormat::Yuv444,
                ChromaSamplePosition::default(),
                false,
            ),
            "422p12" => (
                12,
                PixelFormat::Yuv422,
                ChromaSamplePosition::default(),
                false,
            ),
            "420p12" => (
                12,
                PixelFormat::Yuv420,
                ChromaSamplePosition::default(),
                false,
            ),
            "444" => (
                8,
                PixelFormat::Yuv444,
                ChromaSamplePosition::default(),
                false,
            ),
            "422" => (
                8,
                PixelFormat::Yuv422,
                ChromaSamplePosition::default(),
                false,
            ),
            "420" => (
                8,
                PixelFormat::Yuv420,
                ChromaSamplePosition::default(),
                false,
            ),
            "444alpha" => (
                8,
                PixelFormat::Yuv444,
                ChromaSamplePosition::default(),
                true,
            ),
            "mono" => (
                8,
                PixelFormat::Yuv400,
                ChromaSamplePosition::default(),
                false,
            ),
            "mono10" => (
                10,
                PixelFormat::Yuv400,
                ChromaSamplePosition::default(),
                false,
            ),
            "mono12" => (
                12,
                PixelFormat::Yuv400,
                ChromaSamplePosition::default(),
                false,
            ),
            _ => return AvifError::unknown_error("invalid colorspace string"),
        };
        Ok(())
    }

    pub fn create(filename: &str) -> AvifResult<Y4MReader> {
        let mut reader = BufReader::with_capacity(
            Y4M_BUF_READER_CAP,
            File::open(filename).map_err(AvifError::map_unknown_error)?,
        );
        let mut header_buf = create_vec_exact(128)?;
        let bytes_read = read_bounded_line(&mut reader, &mut header_buf, Y4M_MAX_LINE_BYTES)?;
        if bytes_read == 0 || !header_buf.ends_with(b"\n") {
            return AvifError::unknown_error("no terminated header line in y4m file");
        }
        let header_str = std::str::from_utf8(&header_buf).map_err(AvifError::map_unknown_error)?;
        let y4m_line = header_str.trim_end_matches(['\r', '\n']);
        let parts: Vec<&str> = y4m_line.split(' ').try_collect()?;
        if parts[0] != "YUV4MPEG2" {
            return AvifError::unknown_error("Not a Y4M file");
        }
        let mut y4m = Y4MReader {
            depth: 8,
            format: PixelFormat::Yuv420,
            range: YuvRange::Limited,
            timescale: 30,
            frame_duration: 1,
            frame_header_buf: create_vec_exact(64)?,
            ..Default::default()
        };
        for part in parts[1..].iter() {
            match part.get(0..1).unwrap_or("") {
                "W" => y4m.width = part[1..].parse::<u32>().unwrap_or(0),
                "H" => y4m.height = part[1..].parse::<u32>().unwrap_or(0),
                "C" => y4m.parse_colorspace(&part[1..])?,
                "F" => {
                    if let Some((num_s, den_s)) = part[1..].split_once(':') {
                        if let (Ok(num), Ok(den)) = (num_s.parse::<u32>(), den_s.parse::<u64>()) {
                            if num > 0 && den > 0 {
                                y4m.timescale = num as u64;
                                y4m.frame_duration = den;
                            }
                        }
                    }
                }
                "X" if part[1..] == *"COLORRANGE=FULL" => {
                    y4m.range = YuvRange::Full;
                }
                "X" if part[1..] == *"COLORRANGE=LIMITED" => {
                    y4m.range = YuvRange::Limited;
                }
                _ => {}
            }
        }
        if y4m.width == 0 || y4m.height == 0 || y4m.depth == 0 {
            return AvifError::invalid_argument();
        }
        y4m.reader = Some(reader);
        Ok(y4m)
    }
}

impl Reader for Y4MReader {
    fn read_frame(&mut self, config: &Config) -> AvifResult<(Image, u64, Option<GainMap>)> {
        if config.image_size_limit > 0
            && (self.width as u64) * (self.height as u64) > config.image_size_limit as u64
        {
            return AvifError::unknown_error("Too big Y4M dimensions");
        }
        let default_range = self.range;
        let reader = self
            .reader
            .as_mut()
            .ok_or(AvifError::UnknownError("reader not initialized".into()))?;
        let frame_range = {
            let buf = reader.fill_buf().map_err(AvifError::map_unknown_error)?;
            if buf.is_empty() {
                return AvifError::unknown_error("could not read frame marker");
            }
            if buf.starts_with(b"FRAME\n") {
                reader.consume(6);
                default_range
            } else if buf.starts_with(b"FRAME\r\n") {
                reader.consume(7);
                default_range
            } else {
                let bytes_read =
                    read_bounded_line(reader, &mut self.frame_header_buf, Y4M_MAX_LINE_BYTES)?;
                if bytes_read == 0 {
                    return AvifError::unknown_error("could not read frame marker");
                }
                parse_frame_header(&self.frame_header_buf, default_range)?
            }
        };
        let mut image = image::Image {
            width: self.width,
            height: self.height,
            depth: self.depth,
            yuv_format: self.format,
            yuv_range: frame_range,
            alpha_present: self.has_alpha,
            chroma_sample_position: self.chroma_sample_position,
            matrix_coefficients: config
                .matrix_coefficients
                .unwrap_or(MatrixCoefficients::Bt601),
            ..Default::default()
        };
        image.allocate_planes(Category::Color)?;
        if self.has_alpha {
            image.allocate_planes(Category::Alpha)?;
        }
        let max_sample = image.max_channel();
        for plane in ALL_PLANES {
            if !image.has_plane(plane) {
                continue;
            }
            let actual_width = image.width(plane);
            let actual_height = image.height(plane);
            let padded_width = round2_usize(actual_width);
            let plane_index = plane.as_usize();
            if self.depth == 8 {
                if padded_width == actual_width {
                    let active_len = checked_mul!(actual_width, actual_height)?;
                    let Some(Pixels::Buffer(buf)) = image.planes[plane_index].as_mut() else {
                        return AvifError::no_content();
                    };
                    reader
                        .read_exact(&mut buf[..active_len])
                        .map_err(AvifError::map_unknown_error)?;
                } else {
                    for y in 0..u32_from_usize(actual_height)? {
                        let row = image.row_mut(plane, y)?;
                        reader
                            .read_exact(&mut row[..actual_width])
                            .map_err(AvifError::map_unknown_error)?;
                    }
                }
            } else if padded_width == actual_width {
                let active_samples = checked_mul!(actual_width, actual_height)?;
                let Some(Pixels::Buffer16(buf16)) = image.planes[plane_index].as_mut() else {
                    return AvifError::no_content();
                };
                read_and_clamp_u16(reader, &mut buf16[..active_samples], max_sample)?;
            } else {
                for y in 0..u32_from_usize(actual_height)? {
                    let row = image.row16_mut(plane, y)?;
                    read_and_clamp_u16(reader, &mut row[..actual_width], max_sample)?;
                }
            }
        }
        Ok((image, self.frame_duration, None))
    }

    fn has_more_frames(&mut self) -> bool {
        let Some(reader) = self.reader.as_mut() else {
            return false;
        };
        let buffer = match reader.fill_buf() {
            Ok(buffer) => buffer,
            Err(_) => return false,
        };
        !buffer.is_empty()
    }

    fn timescale(&self) -> u64 {
        self.timescale
    }
}
