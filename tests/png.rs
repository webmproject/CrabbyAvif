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

#![allow(clippy::disallowed_methods, clippy::disallowed_macros)]
#![cfg(feature = "png")]

use crabby_avif::image::*;
use crabby_avif::*;

mod utils;
use utils::*;

use test_case::test_matrix;

#[test_matrix(
    [8, 16],
    [false, true]
)]
fn roundtrip(depth: u8, alpha: bool) -> AvifResult<()> {
    let image = generate_gradient_image(1, 1, depth, PixelFormat::Yuv444, YuvRange::Full, alpha)?;
    let path = tempfile::Builder::new()
        .suffix(".png")
        .tempfile()
        .unwrap()
        .into_temp_path();
    let path_str = path.to_str().unwrap();
    write_png(&image, path_str)?;
    let decoded = read_image(path_str)?;
    are_images_equal(&image, &decoded)?;
    Ok(())
}

mod extended_png_tests {
    use super::get_test_file;
    use crabby_avif::image::*;
    use crabby_avif::utils::reader::png::PngReader;
    use crabby_avif::utils::reader::{Config, Reader};
    use crabby_avif::{AvifResult, ColorPrimaries, PixelFormat, TransferCharacteristics};

    fn crc32(data: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &b in data {
            crc ^= b as u32;
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
        !crc
    }

    fn adler32(data: &[u8]) -> u32 {
        let mut s1 = 1u32;
        let mut s2 = 0u32;
        for &b in data {
            s1 = (s1 + b as u32) % 65521;
            s2 = (s2 + s1) % 65521;
        }
        (s2 << 16) | s1
    }

    fn append_chunk(dst: &mut Vec<u8>, tag: &[u8; 4], data: &[u8]) {
        dst.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let crc_start = dst.len();
        dst.extend_from_slice(tag);
        dst.extend_from_slice(data);
        let c = crc32(&dst[crc_start..]);
        dst.extend_from_slice(&c.to_be_bytes());
    }

    fn zlib_stored(raw: &[u8]) -> Vec<u8> {
        let mut out = vec![0x78, 0x01];
        if raw.is_empty() {
            out.extend_from_slice(&[1, 0, 0, 0xFF, 0xFF]);
        } else {
            let mut pos = 0usize;
            while pos < raw.len() {
                let take = (raw.len() - pos).min(65535);
                let final_flag = if pos + take == raw.len() { 1u8 } else { 0u8 };
                out.push(final_flag);
                let len = take as u16;
                out.extend_from_slice(&len.to_le_bytes());
                out.extend_from_slice(&(!len).to_le_bytes());
                out.extend_from_slice(&raw[pos..pos + take]);
                pos += take;
            }
        }
        out.extend_from_slice(&adler32(raw).to_be_bytes());
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn build_png(
        width: u32,
        height: u32,
        bit_depth: u8,
        color_type: u8,
        interlace: u8,
        pre_chunks: &[(&[u8; 4], &[u8])],
        idat_chunks: &[&[u8]],
        post_chunks: &[(&[u8; 4], &[u8])],
    ) -> Vec<u8> {
        let mut out = Vec::from(b"\x89PNG\r\n\x1a\n".as_slice());
        let mut ihdr = Vec::with_capacity(13);
        ihdr.extend_from_slice(&width.to_be_bytes());
        ihdr.extend_from_slice(&height.to_be_bytes());
        ihdr.push(bit_depth);
        ihdr.push(color_type);
        ihdr.push(0); // compression
        ihdr.push(0); // filter
        ihdr.push(interlace);
        append_chunk(&mut out, b"IHDR", &ihdr);
        for (tag, data) in pre_chunks {
            append_chunk(&mut out, tag, data);
        }
        for idat in idat_chunks {
            append_chunk(&mut out, b"IDAT", idat);
        }
        for (tag, data) in post_chunks {
            append_chunk(&mut out, tag, data);
        }
        append_chunk(&mut out, b"IEND", &[]);
        out
    }

    fn read_png_bytes(bytes: &[u8], config: &Config) -> AvifResult<Image> {
        use std::io::Write;
        let mut tmp = tempfile::Builder::new().suffix(".png").tempfile().unwrap();
        tmp.write_all(bytes).unwrap();
        let path = tmp.into_temp_path();
        let mut reader = PngReader::create(path.to_str().expect("utf8 path"))?;
        let (img, _, _) = reader.read_frame(config)?;
        Ok(img)
    }

    fn encode_filtered_scanlines(
        pixels: &[u8],
        height: usize,
        row_bytes: usize,
        bpp: usize,
        filters: &[u8],
    ) -> Vec<u8> {
        let mut out = Vec::with_capacity(height * (row_bytes + 1));
        for y in 0..height {
            let f = filters[y % filters.len()];
            out.push(f);
            let cur = &pixels[y * row_bytes..(y + 1) * row_bytes];
            let prev = if y > 0 { Some(&pixels[(y - 1) * row_bytes..y * row_bytes]) } else { None };
            for i in 0..row_bytes {
                let left = if i >= bpp { cur[i - bpp] } else { 0 };
                let up = prev.map_or(0, |p| p[i]);
                let up_left = if i >= bpp { prev.map_or(0, |p| p[i - bpp]) } else { 0 };
                let pred = match f {
                    0 => 0,
                    1 => left,
                    2 => up,
                    3 => (((left as u16) + (up as u16)) >> 1) as u8,
                    4 => {
                        let a = left as i32;
                        let b = up as i32;
                        let c = up_left as i32;
                        let p = a + b - c;
                        let pa = (p - a).abs();
                        let pb = (p - b).abs();
                        let pc = (p - c).abs();
                        if pa <= pb && pa <= pc {
                            left
                        } else if pb <= pc {
                            up
                        } else {
                            up_left
                        }
                    }
                    _ => unreachable!(),
                };
                out.push(cur[i].wrapping_sub(pred));
            }
        }
        out
    }

    #[test]
    fn test_png_palette_with_trns() -> AvifResult<()> {
        let plte = [255u8, 0, 0, 0, 255, 0, 0, 0, 255];
        let trns = [128u8, 64];
        let mut raw = Vec::new();
        for y in 0..4u8 {
            raw.push(0); // None filter
            raw.extend_from_slice(&[y % 3, (y + 1) % 3, (y + 2) % 3, 0]);
        }
        let z = zlib_stored(&raw);
        let png = build_png(
            4,
            4,
            8,
            3, // PALETTE
            0,
            &[(b"PLTE", &plte), (b"tRNS", &trns)],
            &[&z],
            &[],
        );
        let img = read_png_bytes(&png, &Config::default())?;
        assert_eq!(img.width, 4);
        assert_eq!(img.height, 4);
        assert!(img.has_alpha());
        let alpha_row0 = img.row(Plane::A, 0)?;
        assert_eq!(&alpha_row0[..4], &[128, 64, 255, 128]);
        Ok(())
    }

    #[test]
    fn test_png_gray_and_gray_alpha_and_4bit_gray() -> AvifResult<()> {
        let mut raw_gray = Vec::new();
        for y in 0..4u8 {
            raw_gray.push(y % 4);
            raw_gray.extend_from_slice(&[10, 20, 30, 40]);
        }
        let z_gray = zlib_stored(&raw_gray);
        let png_gray = build_png(4, 4, 8, 0, 0, &[], &[&z_gray], &[]);
        let img_gray = read_png_bytes(&png_gray, &Config::default())?;
        assert_eq!(img_gray.yuv_format, PixelFormat::Yuv400);
        assert!(!img_gray.has_alpha());

        let mut raw_gray4 = Vec::new();
        for _ in 0..2 {
            raw_gray4.push(0);
            raw_gray4.extend_from_slice(&[0x0F, 0x84]);
        }
        let z_gray4 = zlib_stored(&raw_gray4);
        let png_gray4 = build_png(4, 2, 4, 0, 0, &[], &[&z_gray4], &[]);
        let img_gray4 = read_png_bytes(&png_gray4, &Config::default())?;
        assert_eq!(img_gray4.yuv_format, PixelFormat::Yuv400);
        assert_eq!(img_gray4.width, 4);

        let mut raw_ga = Vec::new();
        for _ in 0..4 {
            raw_ga.push(0);
            raw_ga.extend_from_slice(&[200, 100, 150, 250, 50, 10, 120, 255]);
        }
        let z_ga = zlib_stored(&raw_ga);
        let png_ga = build_png(4, 4, 8, 4, 0, &[], &[&z_ga], &[]);
        let img_ga = read_png_bytes(&png_ga, &Config::default())?;
        assert_eq!(img_ga.yuv_format, PixelFormat::Yuv400);
        assert!(img_ga.has_alpha());
        assert_eq!(&img_ga.row(Plane::A, 0)?[..4], &[100, 250, 10, 255]);
        Ok(())
    }

    #[test]
    fn test_png_adam7_interlaced() -> AvifResult<()> {
        let pass_dims = [(1, 1), (1, 1), (2, 1), (2, 2), (4, 2), (4, 4), (8, 4)];
        let mut raw = Vec::new();
        for (pw, ph) in pass_dims {
            for _ in 0..ph {
                raw.push(0); // filter None
                raw.extend_from_slice(&[12u8, 34, 56].repeat(pw));
            }
        }
        let z = zlib_stored(&raw);
        let png = build_png(8, 8, 8, 2, 1, &[], &[&z], &[]);
        let img = read_png_bytes(&png, &Config::default())?;
        assert_eq!(img.width, 8);
        assert_eq!(img.height, 8);
        Ok(())
    }

    #[test]
    fn test_png_cicp_and_color_primaries_override() -> AvifResult<()> {
        let mut raw = Vec::new();
        for _ in 0..2 {
            raw.push(0);
            raw.extend_from_slice(&[100, 120, 140, 100, 120, 140]);
        }
        let z = zlib_stored(&raw);
        let cicp = [9u8, 16, 0, 1]; // BT.2020 + PQ + RGB + full
        let png = build_png(2, 2, 8, 2, 0, &[(b"cICP", &cicp)], &[&z], &[]);
        let img = read_png_bytes(&png, &Config::default())?;
        assert_eq!(img.color_primaries, ColorPrimaries::Bt2020);
        assert_eq!(img.transfer_characteristics, TransferCharacteristics::Pq);

        let img_override = read_png_bytes(
            &png,
            &Config {
                color_primaries: Some(ColorPrimaries::Srgb),
                ..Default::default()
            },
        )?;
        assert_eq!(img_override.color_primaries, ColorPrimaries::Srgb);
        assert_eq!(
            img_override.transfer_characteristics,
            TransferCharacteristics::Pq
        );
        Ok(())
    }

    #[test]
    fn test_png_16bit_all_filters_and_weld_16bit() -> AvifResult<()> {
        let width = 12u32;
        let height = 5u32;
        let mut raw = Vec::new();
        for y in 0..height {
            raw.push((y % 5) as u8);
            for x in 0..width {
                let r = ((x * 5000 + y * 1234) as u16).to_be_bytes();
                let g = ((x * 3000 + y * 2345) as u16).to_be_bytes();
                let b = ((x * 1000 + y * 3456) as u16).to_be_bytes();
                raw.extend_from_slice(&r);
                raw.extend_from_slice(&g);
                raw.extend_from_slice(&b);
            }
        }
        let z = zlib_stored(&raw);
        let png = build_png(width, height, 16, 2, 0, &[], &[&z], &[]);
        let mut cfg = Config {
            allow_sample_transform: true,
            ..Default::default()
        };
        let img = read_png_bytes(&png, &cfg)?;
        assert_eq!(img.depth, 16);
        assert_eq!(img.width, width);
        assert_eq!(img.height, height);

        cfg.allow_sample_transform = false;
        let fixture = get_test_file("weld_16bit.png");
        let mut reader = PngReader::create(&fixture)?;
        let (weld_img, _, _) = reader.read_frame(&cfg)?;
        assert_eq!(weld_img.depth, 12);
        assert!(weld_img.width > 0 && weld_img.height > 0);
        Ok(())
    }

    #[test]
    fn test_png_corrupt_crc_rejected() {
        let mut raw = Vec::new();
        for _ in 0..2 {
            raw.push(0);
            raw.extend_from_slice(&[10, 20, 30, 40, 50, 60]);
        }
        let z = zlib_stored(&raw);
        let mut png_ihdr = build_png(2, 2, 8, 2, 0, &[], &[&z], &[]);
        png_ihdr[32] ^= 0xFF;
        assert!(read_png_bytes(&png_ihdr, &Config::default()).is_err());

        let mut png_idat = build_png(2, 2, 8, 2, 0, &[], &[&z], &[]);
        let idat_crc_pos = png_idat.len() - 12 - 1;
        png_idat[idat_crc_pos] ^= 0xFF;
        assert!(read_png_bytes(&png_idat, &Config::default()).is_err());
    }

    #[test]
    fn test_png_post_idat_exif_xmp_and_edge_cases() -> AvifResult<()> {
        let mut raw = Vec::new();
        for _ in 0..2 {
            raw.push(0);
            raw.extend_from_slice(&[10, 20, 30, 40, 50, 60]);
        }
        let z = zlib_stored(&raw);
        let exif_bytes = b"MM\x00*\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00";
        // Embedded NUL byte in tEXt payload must truncate at the first NUL like libpng strlen.
        let xmp_chunk = b"XML:com.adobe.xmp\x00<x:xmpmeta/>\x00trailing";
        let png = build_png(
            2,
            2,
            8,
            2,
            0,
            &[],
            &[&z],
            &[(b"eXIf", exif_bytes), (b"tEXt", xmp_chunk)],
        );

        let img = read_png_bytes(&png, &Config::default())?;
        assert_eq!(img.exif.as_slice(), exif_bytes);
        assert_eq!(img.xmp.as_slice(), b"<x:xmpmeta/>");

        // Malformed pre-IDAT eXIf without TIFF magic is benignly skipped so a valid raw profile
        // Exif text chunk is still extracted.
        let mut hex = String::new();
        for b in exif_bytes {
            hex.push_str(&format!("{b:02x}"));
        }
        let raw_exif_profile = format!(
            "Raw profile type exif\0\nexif\n{}\n{hex}\n",
            exif_bytes.len()
        );
        let png_bad_exif = build_png(
            2,
            2,
            8,
            2,
            0,
            &[(b"eXIf", b"NOPE")],
            &[&z],
            &[(b"tEXt", raw_exif_profile.as_bytes())],
        );
        let img_fallback_exif = read_png_bytes(&png_bad_exif, &Config::default())?;
        assert_eq!(img_fallback_exif.exif.as_slice(), exif_bytes);

        let cfg_ignore = Config {
            ignore_exif: true,
            ignore_xmp: true,
            ..Default::default()
        };
        let img_ignored = read_png_bytes(&png, &cfg_ignore)?;
        assert!(img_ignored.exif.is_empty());
        assert!(img_ignored.xmp.is_empty());
        Ok(())
    }

    #[test]
    fn test_png_metadata_priority_and_ignore_flags() -> AvifResult<()> {
        let mut raw = Vec::new();
        for _ in 0..2 {
            raw.push(0);
            raw.extend_from_slice(&[64, 128, 192, 64, 128, 192]);
        }
        let z = zlib_stored(&raw);

        let srgb = [0u8];
        let mut chrm = Vec::with_capacity(32);
        for v in [31270u32, 32900, 64000, 33000, 30000, 60000, 15000, 6000] {
            chrm.extend_from_slice(&v.to_be_bytes());
        }
        let gama = 45455u32.to_be_bytes();
        let itxt = b"XML:com.adobe.xmp\x00\x00\x00\x00\x00<x:xmpmeta>lazy</x:xmpmeta>";
        let exif_bytes = b"MM\x00*\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00";

        let png_srgb = build_png(
            2,
            2,
            8,
            2,
            0,
            &[
                (b"sRGB", &srgb),
                (b"gAMA", &gama),
                (b"cHRM", &chrm),
                (b"eXIf", exif_bytes),
                (b"iTXt", itxt),
            ],
            &[&z],
            &[],
        );
        let img_srgb = read_png_bytes(&png_srgb, &Config::default())?;
        assert_eq!(img_srgb.color_primaries, ColorPrimaries::Srgb);
        assert_eq!(
            img_srgb.transfer_characteristics,
            TransferCharacteristics::Srgb
        );
        assert!(img_srgb.icc.is_empty());
        assert_eq!(img_srgb.exif.as_slice(), exif_bytes);
        assert_eq!(img_srgb.xmp.as_slice(), b"<x:xmpmeta>lazy</x:xmpmeta>");

        // Recognized gAMA (45455 -> 2.2) + cHRM (BT.709) sets CICP without generating ICC.
        let png_gama_chrm_known = build_png(
            2,
            2,
            8,
            2,
            0,
            &[(b"gAMA", &gama), (b"cHRM", &chrm)],
            &[&z],
            &[],
        );
        let img_known = read_png_bytes(&png_gama_chrm_known, &Config::default())?;
        assert_eq!(img_known.color_primaries, ColorPrimaries::Srgb);
        assert_eq!(
            img_known.transfer_characteristics,
            TransferCharacteristics::Bt470m
        );
        assert!(img_known.icc.is_empty());

        // Non-standard gAMA (1.8) + cHRM generates ICC unless ignore_icc is set.
        let gama_unknown = 55556u32.to_be_bytes();
        let png_gama_chrm = build_png(
            2,
            2,
            8,
            2,
            0,
            &[(b"gAMA", &gama_unknown), (b"cHRM", &chrm)],
            &[&z],
            &[],
        );
        let img_icc = read_png_bytes(&png_gama_chrm, &Config::default())?;
        assert!(!img_icc.icc.is_empty());
        let img_no_icc = read_png_bytes(
            &png_gama_chrm,
            &Config {
                ignore_icc: true,
                ..Default::default()
            },
        )?;
        assert!(img_no_icc.icc.is_empty());

        // cICP takes precedence over sRGB / gAMA / cHRM.
        let cicp = [9u8, 16, 0, 1];
        let png_cicp = build_png(
            2,
            2,
            8,
            2,
            0,
            &[
                (b"cICP", &cicp),
                (b"sRGB", &srgb),
                (b"gAMA", &gama_unknown),
                (b"cHRM", &chrm),
            ],
            &[&z],
            &[],
        );
        let img_cicp = read_png_bytes(&png_cicp, &Config::default())?;
        assert_eq!(img_cicp.color_primaries, ColorPrimaries::Bt2020);
        assert_eq!(
            img_cicp.transfer_characteristics,
            TransferCharacteristics::Pq
        );
        assert!(img_cicp.icc.is_empty());
        Ok(())
    }

    #[test]
    fn test_png_16bit_edge_widths_and_filters() -> AvifResult<()> {
        let cases = [
            (1usize, 10usize, 0u8, 1usize),
            (7, 10, 4, 2),
            (17, 10, 2, 3),
            (33, 10, 6, 4),
        ];
        let filters = [0u8, 1, 2, 3, 4, 3, 4, 1, 2, 0];
        for (width, height, color_type, channels) in cases {
            let row_bytes = width * channels * 2;
            let mut pixels = vec![0u8; height * row_bytes];
            for y in 0..height {
                for x in 0..width {
                    for c in 0..channels {
                        let val = if color_type == 0 && (x + y) % 2 == 0 {
                            0xFF00u16
                        } else if color_type == 0 {
                            0x00FFu16
                        } else {
                            (((y * 257 + x * 41 + c * 997) ^ 0x5A3C) & 0xFFFF) as u16
                        };
                        let off = y * row_bytes + (x * channels + c) * 2;
                        let be = val.to_be_bytes();
                        pixels[off] = be[0];
                        pixels[off + 1] = be[1];
                    }
                }
            }

            let raw_none =
                encode_filtered_scanlines(&pixels, height, row_bytes, channels * 2, &[0]);
            let raw_mix =
                encode_filtered_scanlines(&pixels, height, row_bytes, channels * 2, &filters);
            let z_none = zlib_stored(&raw_none);
            let z_mix = zlib_stored(&raw_mix);

            let png_none = build_png(
                width as u32,
                height as u32,
                16,
                color_type,
                0,
                &[],
                &[&z_none],
                &[],
            );
            let png_mix = build_png(
                width as u32,
                height as u32,
                16,
                color_type,
                0,
                &[],
                &[&z_mix],
                &[],
            );
            let cfg16 = Config {
                allow_sample_transform: true,
                ..Default::default()
            };
            let img_none = read_png_bytes(&png_none, &cfg16)?;
            let img_mix = read_png_bytes(&png_mix, &cfg16)?;
            assert_eq!(img_mix.depth, 16);
            for y in 0..height as u32 {
                assert_eq!(
                    &img_mix.row16(Plane::Y, y)?[..width],
                    &img_none.row16(Plane::Y, y)?[..width]
                );
                if channels == 2 || channels == 4 {
                    assert_eq!(
                        &img_mix.row16(Plane::A, y)?[..width],
                        &img_none.row16(Plane::A, y)?[..width]
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_png_gama_zero_and_unknown_chunks_and_multi_idat() -> AvifResult<()> {
        let raw_2x2 = vec![
            0, 10, 20, 30, 40, 50, 60, // row 0 (None)
            0, 70, 80, 90, 100, 110, 120, // row 1 (None)
        ];
        let z_2x2 = zlib_stored(&raw_2x2);
        // 1) gAMA == 0 is treated as benignly absent (defaults to gamma 2.2, never 1.0 / 0.0).
        let png_gama0 = build_png(2, 2, 8, 2, 0, &[(b"gAMA", &[0, 0, 0, 0])], &[&z_2x2], &[]);
        let img_gama0 = read_png_bytes(&png_gama0, &Config::default())?;
        assert_eq!(img_gama0.width, 2);
        assert_eq!(img_gama0.height, 2);

        // 2) Unknown ancillary chunk is skipped, while unknown critical chunk is rejected.
        let png_ancillary = build_png(2, 2, 8, 2, 0, &[(b"baDc", &[1, 2, 3, 4])], &[&z_2x2], &[]);
        assert_eq!(read_png_bytes(&png_ancillary, &Config::default())?.width, 2);
        let png_bad_crit = build_png(2, 2, 8, 2, 0, &[(b"BADc", &[1, 2, 3, 4])], &[&z_2x2], &[]);
        assert!(read_png_bytes(&png_bad_crit, &Config::default()).is_err());

        // 3) Multi-IDAT split into 1-byte IDAT chunks + image_size_limit check.
        let chunks: Vec<&[u8]> = z_2x2.chunks(1).collect();
        let png_multi = build_png(2, 2, 8, 2, 0, &[], &chunks, &[]);
        assert_eq!(read_png_bytes(&png_multi, &Config::default())?.width, 2);
        assert!(read_png_bytes(
            &png_multi,
            &Config {
                image_size_limit: 3,
                ..Default::default()
            }
        )
        .is_err());
        Ok(())
    }
}
