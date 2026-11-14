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

// Not all sub-modules are used by all targets. Ignore dead code warnings.
#![allow(dead_code)]

pub mod avif;
#[cfg(feature = "gif")]
pub mod gif;
#[cfg(feature = "jpeg")]
pub mod jpeg;
#[cfg(feature = "png")]
pub mod png;
pub mod y4m;

#[cfg(feature = "png")]
mod icc;
#[cfg(feature = "jpeg")]
mod xmp;

use crate::gainmap::GainMap;
use crate::image::Image;
use crate::image::YuvRange;
use crate::AvifError;
use crate::AvifResult;
use crate::ColorPrimaries;
use crate::MatrixCoefficients;
use crate::PixelFormat;

#[derive(Default)]
pub struct Config {
    pub yuv_format: Option<PixelFormat>,
    pub yuv_range: Option<YuvRange>,
    pub depth: Option<u8>,
    pub color_primaries: Option<ColorPrimaries>,
    pub matrix_coefficients: Option<MatrixCoefficients>,
    pub ignore_icc: bool,
    pub ignore_exif: bool,
    pub ignore_xmp: bool,
    pub image_size_limit: u32,
    pub allow_sample_transform: bool,
    pub max_threads: u32,
}

pub trait Reader {
    // Returns the next frame, its duration in timescale units, and the gain map if any.
    fn read_frame(&mut self, config: &Config) -> AvifResult<(Image, u64, Option<GainMap>)>;
    // Returns true if the last call to read_frame() returned another frame than the last frame.
    // Meaningless if read_frame() was never called or if read_frame() returned an error.
    fn has_more_frames(&mut self) -> bool;
    // Timescale (ticks per second) for frame durations returned by read_frame().
    fn timescale(&self) -> u64 {
        1000
    }
}

impl dyn Reader {
    pub fn create(path: &str) -> AvifResult<Box<dyn Reader>> {
        Ok(
            match std::path::Path::new(path)
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_lowercase()
                .as_str()
            {
                "y4m" => Box::new(y4m::Y4MReader::create(path)?),
                #[cfg(feature = "jpeg")]
                "jpg" | "jpeg" => Box::new(jpeg::JpegReader::create(path)?),
                #[cfg(not(feature = "jpeg"))]
                "jpg" | "jpeg" => {
                    return AvifError::unknown_error(format!(
                    "Cannot read {path} because CrabbyAvif was not compiled with the jpeg feature"
                ))
                }
                #[cfg(feature = "png")]
                "png" => Box::new(png::PngReader::create(path)?),
                #[cfg(not(feature = "png"))]
                "png" => {
                    return AvifError::unknown_error(format!(
                    "Cannot read {path} because CrabbyAvif was not compiled with the png feature"
                ))
                }
                #[cfg(feature = "gif")]
                "gif" => Box::new(gif::GifReader::create(path)?),
                #[cfg(not(feature = "gif"))]
                "gif" => {
                    return AvifError::unknown_error(format!(
                    "Cannot read {path} because CrabbyAvif was not compiled with the gif feature"
                ))
                }
                "avif" => Box::new(avif::AvifReader::create(path)?),
                _ => {
                    return AvifError::unknown_error(format!(
                        "Unknown input file extension for {path}"
                    ))
                }
            },
        )
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods, clippy::disallowed_macros)]
mod tests {
    use super::*;
    #[cfg(feature = "png")]
    use crate::image::Plane;

    #[cfg(feature = "png")]
    fn assert_images_bit_identical(a: &Image, b: &Image) -> AvifResult<()> {
        assert_eq!(a.width, b.width);
        assert_eq!(a.height, b.height);
        assert_eq!(a.depth, b.depth);
        assert_eq!(a.yuv_format, b.yuv_format);
        for plane in [Plane::Y, Plane::U, Plane::V, Plane::A] {
            assert_eq!(a.has_plane(plane), b.has_plane(plane));
            if !a.has_plane(plane) {
                continue;
            }
            let ph = a.height(plane) as u32;
            let pw = a.width(plane);
            for y in 0..ph {
                if a.depth == 8 {
                    assert_eq!(&a.row(plane, y)?[..pw], &b.row(plane, y)?[..pw]);
                } else {
                    assert_eq!(&a.row16(plane, y)?[..pw], &b.row16(plane, y)?[..pw]);
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "png")]
    fn png_crc32(data: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &byte in data {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
        !crc
    }

    #[cfg(feature = "png")]
    fn png_adler32(data: &[u8]) -> u32 {
        let mut s1 = 1u32;
        let mut s2 = 0u32;
        for &byte in data {
            s1 = (s1 + u32::from(byte)) % 65521;
            s2 = (s2 + s1) % 65521;
        }
        (s2 << 16) | s1
    }

    #[cfg(feature = "png")]
    fn push_png_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], payload: &[u8]) {
        out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        let start = out.len();
        out.extend_from_slice(chunk_type);
        out.extend_from_slice(payload);
        let crc = png_crc32(&out[start..]);
        out.extend_from_slice(&crc.to_be_bytes());
    }

    #[cfg(feature = "png")]
    fn build_uncompressed_rgba8_png(width: u32, height: u32, post_idat_text: bool) -> Vec<u8> {
        let w = width as usize;
        let h = height as usize;
        let mut raw = Vec::with_capacity(h * (1 + w * 4));
        for y in 0..h {
            raw.push(0); // Filter: None
            for x in 0..w {
                raw.push(((x * 37 + y * 13 + 7) & 0xFF) as u8);
                raw.push(((x * 19 + y * 41 + 11) & 0xFF) as u8);
                raw.push(((x * 53 + y * 29 + 3) & 0xFF) as u8);
                raw.push(((x * 7 + y * 61 + 128) & 0xFF) as u8);
            }
        }

        let mut zlib = vec![0x78, 0x01];
        let mut offset = 0usize;
        while offset < raw.len() {
            let block_len = (raw.len() - offset).min(65535);
            let is_final = u8::from(offset + block_len == raw.len());
            zlib.push(is_final);
            let len_u16 = block_len as u16;
            zlib.extend_from_slice(&len_u16.to_le_bytes());
            zlib.extend_from_slice(&(!len_u16).to_le_bytes());
            zlib.extend_from_slice(&raw[offset..offset + block_len]);
            offset += block_len;
        }
        zlib.extend_from_slice(&png_adler32(&raw).to_be_bytes());

        let mut png = Vec::new();
        png.extend_from_slice(b"\x89PNG\r\n\x1a\n");
        let mut ihdr = Vec::with_capacity(13);
        ihdr.extend_from_slice(&width.to_be_bytes());
        ihdr.extend_from_slice(&height.to_be_bytes());
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit RGBA, non-interlaced
        push_png_chunk(&mut png, b"IHDR", &ihdr);
        push_png_chunk(&mut png, b"IDAT", &zlib);
        if post_idat_text {
            push_png_chunk(&mut png, b"tEXt", b"XML:com.adobe.xmp\0<x:xmpmeta/>");
        }
        push_png_chunk(&mut png, b"IEND", &[]);
        png
    }

    #[test]
    fn config_default_max_threads_is_zero() {
        assert_eq!(Config::default().max_threads, 0);
    }

    #[cfg(feature = "png")]
    #[test]
    fn png_reader_16bit_max_threads_matches_single_thread_bit_exact() -> AvifResult<()> {
        let weld_path = &crate::internal_utils::get_test_file("weld_16bit.png");
        let cfg_default = Config::default();
        let cfg_1t = Config {
            max_threads: 1,
            ..Default::default()
        };
        let cfg_4t = Config {
            max_threads: 4,
            ..Default::default()
        };
        let (img_def, _, _) = png::PngReader::create(weld_path)?.read_frame(&cfg_default)?;
        let (img_1t, _, _) = png::PngReader::create(weld_path)?.read_frame(&cfg_1t)?;
        let (img_4t, _, _) = png::PngReader::create(weld_path)?.read_frame(&cfg_4t)?;
        assert_images_bit_identical(&img_def, &img_1t)?;
        assert_images_bit_identical(&img_1t, &img_4t)?;
        Ok(())
    }

    #[cfg(feature = "png")]
    #[test]
    fn png_reader_8bit_rgba_max_threads_matches_single_thread_bit_exact() -> AvifResult<()> {
        for post_idat_text in [false, true] {
            let bytes = build_uncompressed_rgba8_png(600, 520, post_idat_text);
            let tmp = tempfile::Builder::new()
                .suffix(".png")
                .tempfile()
                .map_err(AvifError::map_unknown_error)?;
            std::fs::write(tmp.path(), &bytes).map_err(AvifError::map_unknown_error)?;
            let path_str = tmp.path().to_str().unwrap();
            let (st, _, _) = png::PngReader::create(path_str)?.read_frame(&Config {
                yuv_format: Some(PixelFormat::Yuv420),
                max_threads: 1,
                ..Default::default()
            })?;
            let (mt, _, _) = png::PngReader::create(path_str)?.read_frame(&Config {
                yuv_format: Some(PixelFormat::Yuv420),
                max_threads: 4,
                ..Default::default()
            })?;
            assert_images_bit_identical(&st, &mt)?;
            if post_idat_text {
                assert_eq!(st.xmp.as_slice(), b"<x:xmpmeta/>");
            }
        }
        Ok(())
    }
}
