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

#![allow(clippy::disallowed_methods, clippy::disallowed_macros)]

use crabby_avif::image::*;
use crabby_avif::utils::reader::y4m::Y4MReader;
use crabby_avif::utils::reader::Config;
use crabby_avif::utils::reader::Reader;
use crabby_avif::utils::writer::y4m::Y4MWriter;
use crabby_avif::utils::writer::Writer;
use crabby_avif::*;

mod utils;
use utils::*;

use std::fs::File;
use std::io::Write;
use test_case::test_matrix;

fn write_temp_y4m(header: &[u8], payload: &[u8]) -> tempfile::TempPath {
    let mut tmp = tempfile::Builder::new().suffix(".y4m").tempfile().unwrap();
    tmp.write_all(header).unwrap();
    tmp.write_all(payload).unwrap();
    tmp.flush().unwrap();
    tmp.into_temp_path()
}

#[test_matrix(
    [100, 121],
    [200, 107],
    [8, 10, 12],
    [PixelFormat::Yuv420, PixelFormat::Yuv422, PixelFormat::Yuv444, PixelFormat::Yuv400],
    [YuvRange::Limited, YuvRange::Full],
    [false, true]
)]
fn roundtrip(
    width: u32,
    height: u32,
    depth: u8,
    yuv_format: PixelFormat,
    yuv_range: YuvRange,
    alpha: bool,
) -> AvifResult<()> {
    if alpha && (depth != 8 || yuv_format != PixelFormat::Yuv444) {
        // alpha in y4m is supported only for 8-bit 444 images.
        return Ok(());
    }
    let mut image1 = generate_gradient_image(width, height, depth, yuv_format, yuv_range, alpha)?;
    image1.matrix_coefficients = MatrixCoefficients::Bt601;
    let path = tempfile::Builder::new()
        .suffix(".y4m")
        .tempfile()
        .unwrap()
        .into_temp_path();
    let path_str = path.to_str().unwrap();
    // Write the image.
    {
        let mut writer = Y4MWriter::create(false);
        let mut output_file = File::create(path_str).expect("output file creation failed");
        writer.write_frame(&mut output_file, &image1)?;
    }
    // Read the image.
    let mut reader = Y4MReader::create(path_str)?;
    let (image2, _, _) = reader.read_frame(&Config::default())?;
    assert!(are_images_equal(&image1, &image2)?);
    // Read the image using the generic test utils API.
    assert!(are_images_equal(&image1, &read_image(path_str)?)?);
    Ok(())
}

#[test_matrix(
    [10, 12],
    [35, 64],
    [17, 20]
)]
fn read_10_12bit_clamp_and_odd_dims(depth: u8, width: u32, height: u32) -> AvifResult<()> {
    let max_sample = (1u16 << depth) - 1;
    let cs = if depth == 10 { "444p10" } else { "420p12" };
    let path = tempfile::Builder::new()
        .suffix(".y4m")
        .tempfile()
        .unwrap()
        .into_temp_path();
    let path_str = path.to_str().unwrap();
    {
        let mut f = File::create(path_str).unwrap();
        writeln!(
            f,
            "YUV4MPEG2 W{width} H{height} F30:1 C{cs} XCOLORRANGE=FULL"
        )
        .unwrap();
        writeln!(f, "FRAME").unwrap();
        let y_count = (width * height) as usize;
        let uv_w = if depth == 10 { width } else { width.div_ceil(2) };
        let uv_h = if depth == 10 { height } else { height.div_ceil(2) };
        let uv_count = (uv_w * uv_h) as usize;
        // Include values below, at, just above max_sample, and >= 32768 (tests unsigned min_epu16).
        let pattern = [0u16, 17, max_sample, max_sample + 1, 32767, 32768, 65535];
        for i in 0..y_count {
            let v = pattern[i % pattern.len()];
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        for _ in 0..2 {
            for i in 0..uv_count {
                let v = pattern[(i + 2) % pattern.len()];
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
    }
    let mut reader = Y4MReader::create(path_str)?;
    let (img, dur, _) = reader.read_frame(&Config::default())?;
    assert_eq!(dur, 1);
    assert_eq!(img.width, width);
    assert_eq!(img.height, height);
    assert_eq!(img.depth, depth);
    assert_eq!(img.yuv_range, YuvRange::Full);
    assert!(!reader.has_more_frames());

    let pattern = [0u16, 17, max_sample, max_sample + 1, 32767, 32768, 65535];
    for y in 0..height {
        let row = img.row16(Plane::Y, y)?;
        assert_eq!(row.len(), width as usize);
        for x in 0..width as usize {
            let idx = (y as usize) * (width as usize) + x;
            let expected = pattern[idx % pattern.len()].min(max_sample);
            assert_eq!(row[x], expected);
        }
    }
    Ok(())
}

#[test]
fn read_multiframe_range_and_odd_8bit() -> AvifResult<()> {
    let width = 33u32;
    let height = 19u32;
    let path = tempfile::Builder::new()
        .suffix(".y4m")
        .tempfile()
        .unwrap()
        .into_temp_path();
    let path_str = path.to_str().unwrap();
    {
        let mut f = File::create(path_str).unwrap();
        writeln!(
            f,
            "YUV4MPEG2 W{width} H{height} F24000:1001 C420jpeg XCOLORRANGE=LIMITED"
        )
        .unwrap();
        for frame_idx in 0..3u8 {
            if frame_idx == 1 {
                writeln!(f, "FRAME XCOLORRANGE=FULL").unwrap();
            } else {
                writeln!(f, "FRAME").unwrap();
            }
            let y_len = (width * height) as usize;
            let uv_len = (width.div_ceil(2) * height.div_ceil(2)) as usize;
            let y_bytes = vec![16 + frame_idx * 40; y_len];
            let u_bytes = vec![128 + frame_idx; uv_len];
            let v_bytes = vec![128 - frame_idx; uv_len];
            f.write_all(&y_bytes).unwrap();
            f.write_all(&u_bytes).unwrap();
            f.write_all(&v_bytes).unwrap();
        }
    }
    let mut reader = Y4MReader::create(path_str)?;
    assert_eq!(reader.timescale(), 24000);
    for frame_idx in 0..3u8 {
        assert!(reader.has_more_frames());
        let (img, dur, _) = reader.read_frame(&Config::default())?;
        assert_eq!(dur, 1001);
        assert_eq!(
            img.yuv_range,
            if frame_idx == 1 { YuvRange::Full } else { YuvRange::Limited }
        );
        assert_eq!(img.row(Plane::Y, 0)?[0], 16 + frame_idx * 40);
        assert_eq!(img.row(Plane::U, 0)?[0], 128 + frame_idx);
        assert_eq!(img.row(Plane::V, 0)?[0], 128 - frame_idx);
    }
    assert!(!reader.has_more_frames());
    Ok(())
}

#[test]
fn read_444alpha_and_even_odd_10_12bit() -> AvifResult<()> {
    for (width, height) in [(32u32, 19u32), (17, 13)] {
        let path = tempfile::Builder::new()
            .suffix(".y4m")
            .tempfile()
            .unwrap()
            .into_temp_path();
        let path_str = path.to_str().unwrap();
        let n = (width * height) as usize;
        {
            let mut f = File::create(path_str).unwrap();
            writeln!(f, "YUV4MPEG2 W{width} H{height} F30:1 C444alpha").unwrap();
            writeln!(f, "FRAME").unwrap();
            f.write_all(&vec![100u8; n]).unwrap();
            f.write_all(&vec![110u8; n]).unwrap();
            f.write_all(&vec![140u8; n]).unwrap();
            f.write_all(&vec![220u8; n]).unwrap();
        }
        let mut reader = Y4MReader::create(path_str)?;
        let (img, _, _) = reader.read_frame(&Config::default())?;
        assert!(img.alpha_present);
        assert!(img.has_alpha());
        assert_eq!(img.yuv_format, PixelFormat::Yuv444);
        assert_eq!(img.row(Plane::Y, height - 1)?[width as usize - 1], 100);
        assert_eq!(img.row(Plane::A, height - 1)?[width as usize - 1], 220);
    }

    for (cs, depth, fmt) in [
        ("420p10", 10u8, PixelFormat::Yuv420),
        ("444p12", 12u8, PixelFormat::Yuv444),
    ] {
        let (width, height) = (32u32, 19u32);
        let path = tempfile::Builder::new()
            .suffix(".y4m")
            .tempfile()
            .unwrap()
            .into_temp_path();
        let path_str = path.to_str().unwrap();
        let max_v = (1u16 << depth) - 1;
        let cw = if fmt == PixelFormat::Yuv420 { width.div_ceil(2) } else { width };
        let ch = if fmt == PixelFormat::Yuv420 { height.div_ceil(2) } else { height };
        {
            let mut f = File::create(path_str).unwrap();
            writeln!(f, "YUV4MPEG2 W{width} H{height} F25:1 C{cs}").unwrap();
            writeln!(f, "FRAME").unwrap();
            for _ in 0..(width * height) {
                f.write_all(&(max_v + 5).to_le_bytes()).unwrap();
            }
            for _ in 0..(2 * cw * ch) {
                f.write_all(&(1u16 << (depth - 1)).to_le_bytes()).unwrap();
            }
        }
        let mut reader = Y4MReader::create(path_str)?;
        let (img, _, _) = reader.read_frame(&Config::default())?;
        assert_eq!(img.depth, depth);
        assert_eq!(img.yuv_format, fmt);
        assert_eq!(img.row16(Plane::Y, height - 1)?[width as usize - 1], max_v);
    }
    Ok(())
}

#[test]
fn read_frame_10bit_unpadded_multi_chunk_and_straddle() {
    // Odd-length header (43 bytes) forces u16 sample pairs to straddle the 64 KiB BufReader
    // refill boundary across a 400x120 444p10 frame (96,000 bytes/plane).
    let w = 400usize;
    let h = 120usize;
    let samples = w * h * 3;
    let mut payload = Vec::with_capacity(samples * 2);
    for idx in 0..samples {
        let val = if idx % 17 == 0 { 1500u16 } else { ((idx * 13 + 5) % 1024) as u16 };
        payload.extend_from_slice(&val.to_le_bytes());
    }
    let header = b"YUV4MPEG2 W400 H120 F30:1 Ip C444p10\nFRAME\n";
    assert_eq!(header.len() % 2, 1);
    let path = write_temp_y4m(header, &payload);
    let mut reader = Y4MReader::create(path.to_str().unwrap()).unwrap();
    let (img, dur, _) = reader.read_frame(&Config::default()).unwrap();
    assert_eq!(dur, 1);
    assert_eq!(img.width, 400);
    assert_eq!(img.height, 120);
    assert_eq!(img.depth, 10);
    let mut global_idx = 0usize;
    for plane in [Plane::Y, Plane::U, Plane::V] {
        for y in 0..h as u32 {
            let row = img.row16(plane, y).unwrap();
            for &px in row.iter().take(w) {
                let expected = if global_idx % 17 == 0 {
                    1023u16
                } else {
                    ((global_idx * 13 + 5) % 1024) as u16
                };
                assert_eq!(px, expected);
                global_idx += 1;
            }
        }
    }
}

#[test]
fn read_frame_10bit_odd_padded_spanning_multiple_buffers() {
    let w = 333usize;
    let h = 199usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let total = w * h + cw * ch * 2;
    let mut payload = Vec::with_capacity(total * 2);
    for idx in 0..total {
        let val = if idx % 11 == 0 { 2000u16 } else { ((idx * 29 + 7) % 1024) as u16 };
        payload.extend_from_slice(&val.to_le_bytes());
    }
    let header = b"YUV4MPEG2 W333 H199 F24:1 Ip C420p10\nFRAME\n";
    let path = write_temp_y4m(header, &payload);
    let mut reader = Y4MReader::create(path.to_str().unwrap()).unwrap();
    let (img, _, _) = reader.read_frame(&Config::default()).unwrap();
    let mut idx = 0usize;
    for (plane, pw, ph) in [(Plane::Y, w, h), (Plane::U, cw, ch), (Plane::V, cw, ch)] {
        for y in 0..ph as u32 {
            let row = img.row16(plane, y).unwrap();
            for &px in row.iter().take(pw) {
                let expected = if idx % 11 == 0 { 1023u16 } else { ((idx * 29 + 7) % 1024) as u16 };
                assert_eq!(px, expected);
                idx += 1;
            }
        }
    }
}

#[test]
fn read_frame_10bit_wide_row_exceeding_bufreader_capacity() {
    // Odd width where row byte size (65,538 B) exceeds Y4M_BUF_READER_CAP (65,536 B).
    let wide_w = 32769usize;
    let mut wide_payload = Vec::with_capacity(wide_w * 2);
    for x in 0..wide_w {
        let val = if x == 0 || x == wide_w - 1 || x == 32768 {
            2047u16
        } else {
            ((x * 7 + 3) % 1024) as u16
        };
        wide_payload.extend_from_slice(&val.to_le_bytes());
    }
    let wide_header = b"YUV4MPEG2 W32769 H1 F30:1 Ip Cmono10\nFRAME\n";
    let wide_path = write_temp_y4m(wide_header, &wide_payload);
    let mut wide_reader = Y4MReader::create(wide_path.to_str().unwrap()).unwrap();
    let (wide_img, _, _) = wide_reader.read_frame(&Config::default()).unwrap();
    let row = wide_img.row16(Plane::Y, 0).unwrap();
    for (x, &px) in row.iter().take(wide_w).enumerate() {
        let expected = if x == 0 || x == wide_w - 1 || x == 32768 {
            1023u16
        } else {
            ((x * 7 + 3) % 1024) as u16
        };
        assert_eq!(px, expected, "mismatch at x={x}");
    }
}

#[test]
fn y4m_header_validation_and_limits() {
    // 1) F0:0 falls back to default 30:1 framerate instead of failing.
    let p_f0 = write_temp_y4m(b"YUV4MPEG2 W2 H2 F0:0 Cmono\nFRAME\n", &[10, 20, 30, 40]);
    let mut r_f0 = Y4MReader::create(p_f0.to_str().unwrap()).unwrap();
    assert_eq!(r_f0.timescale(), 30);
    let (_, dur, _) = r_f0.read_frame(&Config::default()).unwrap();
    assert_eq!(dur, 1);

    // 2) Unterminated "FRAME" without trailing '\n' is rejected.
    let p_unterm = write_temp_y4m(b"YUV4MPEG2 W2 H2 F30:1 Cmono\nFRAME", &[]);
    let mut r_unterm = Y4MReader::create(p_unterm.to_str().unwrap()).unwrap();
    assert!(r_unterm.read_frame(&Config::default()).is_err());

    // 3) Overlong header (> 2048 bytes without '\n') is rejected without unbounded growth.
    let mut huge_hdr = b"YUV4MPEG2 W2 H2 Cmono X".to_vec();
    huge_hdr.resize(3000, b'A');
    huge_hdr.push(b'\n');
    let p_huge = write_temp_y4m(&huge_hdr, &[]);
    assert!(Y4MReader::create(p_huge.to_str().unwrap()).is_err());

    // 4) image_size_limit is enforced before plane allocation.
    let p_lim = write_temp_y4m(b"YUV4MPEG2 W4 H4 F30:1 Cmono\nFRAME\n", &[0u8; 16]);
    let mut r_lim = Y4MReader::create(p_lim.to_str().unwrap()).unwrap();
    let cfg_lim = Config {
        image_size_limit: 8,
        ..Default::default()
    };
    assert!(r_lim.read_frame(&cfg_lim).is_err());
}
