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
use crate::parser::exif;
use crate::reformat::*;
use crate::utils::pixels::Pixels;
use crate::utils::*;
use crate::AvifError;
use crate::AvifResult;

use super::icc;
use super::Config;
use super::Reader;

use std::fs::File;
use std::io::BufReader;

// Match create_vec_exact's Chromium PartitionAlloc < 2 GiB cap (b/342251590).
const PNG_ALLOC_LIMIT_BYTES: usize = 2_145_386_495;
// Match libpng's default PNG_USER_WIDTH_MAX and PNG_USER_HEIGHT_MAX.
const PNG_MAX_DIMENSION: u32 = 1_000_000;
const BT709_PRIMARIES: [f32; 8] = [0.64, 0.33, 0.3, 0.6, 0.15, 0.06, 0.3127, 0.329];

pub struct PngReader {
    filename: String,
}

fn set_png_cicp(image: &mut Image, cicp: &[u8; 4]) {
    image.color_primaries = (cicp[0] as u16).into();
    image.transfer_characteristics = (cicp[1] as u16).into();
    // PNG specification Third Edition Section 4.3:
    // RGB is currently the only supported color model in PNG, and as such Matrix Coefficients
    // shall be set to 0. Note that we don't set image.matrix_coefficients to this value as the
    // AVIF YUV matrix is independent from the PNG's.
    if cicp[2] != 0 {
        println!(
            "Warning: Unsupported PNG CICP matrix coefficients value {}. Expected to be 0.",
            cicp[2]
        );
    }
    // Limited range PNG files are uncommon and would require conversion to full range which we
    // don't support for now for simplicity. Similarly, we don't set image.yuv_range to this value
    // as the AVIF YUV range is independent from the PNG's.
    if cicp[3] != 1 {
        println!(
            "Warning: Unsupported PNG CICP full range flag value {}. Expected to be 1.",
            cicp[3]
        );
    }
}

fn text_bytes(text: &str, is_latin1: bool) -> AvifResult<Vec<u8>> {
    let text = text.split('\0').next().unwrap_or("");
    if is_latin1 {
        let mut out = create_vec_exact(text.chars().count())?;
        for ch in text.chars() {
            out.try_push(ch as u8)?;
        }
        Ok(out)
    } else {
        let bytes = text.as_bytes();
        let mut out = create_vec_exact(bytes.len())?;
        out.try_extend_from_slice(bytes)?;
        Ok(out)
    }
}

fn apply_raw_text_metadata(
    image: &mut Image,
    key: &str,
    text: &str,
    is_latin1: bool,
    ignore_exif: &mut bool,
    ignore_xmp: &mut bool,
) -> AvifResult<()> {
    const EXIF_PREFIX: &[u8] = b"Exif\0\0";
    const XMP_PREFIX: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
    if !*ignore_exif && key == "Raw profile type exif" {
        let raw = text_bytes(text, is_latin1)?;
        let mut exif = icc::copy_raw_profile(&raw)?;
        if exif.starts_with(EXIF_PREFIX) {
            exif.drain(..EXIF_PREFIX.len());
        }
        let _ = exif::set_orientation(&mut exif, 1);
        image.exif = exif;
        *ignore_exif = true;
    } else if !*ignore_xmp && key == "Raw profile type xmp" {
        let raw = text_bytes(text, is_latin1)?;
        let mut xmp = icc::copy_raw_profile(&raw)?;
        if xmp.starts_with(XMP_PREFIX) {
            xmp.drain(..XMP_PREFIX.len());
        }
        image.xmp = xmp;
        *ignore_xmp = true;
    } else if (!*ignore_exif || !*ignore_xmp)
        && (key == "Raw profile type APP1" || key == "Raw profile type app1")
    {
        // This can be either Exif, XMP, or something else.
        let raw = text_bytes(text, is_latin1)?;
        let mut data = icc::copy_raw_profile(&raw)?;
        if !*ignore_exif && data.starts_with(EXIF_PREFIX) {
            data.drain(..EXIF_PREFIX.len());
            let _ = exif::set_orientation(&mut data, 1);
            image.exif = data;
            *ignore_exif = true;
        } else if !*ignore_xmp && data.starts_with(XMP_PREFIX) {
            data.drain(..XMP_PREFIX.len());
            image.xmp = data;
            *ignore_xmp = true;
        }
    } else if !*ignore_xmp && key == "XML:com.adobe.xmp" {
        let xmp = text_bytes(text, is_latin1)?;
        if xmp.is_empty() {
            return Err(AvifError::UnknownError(
                "XMP extraction failed: empty XML:com.adobe.xmp payload".into(),
            ));
        }
        image.xmp = xmp;
        *ignore_xmp = true;
    }
    Ok(())
}

#[derive(Default)]
struct TextChunkOffsets {
    latin1: usize,
    compressed_latin1: usize,
    utf8: usize,
}

fn extract_exif_and_xmp(
    image: &mut Image,
    info: &png::Info<'_>,
    offsets: &mut TextChunkOffsets,
    ignore_exif: &mut bool,
    ignore_xmp: &mut bool,
) -> AvifResult<()> {
    if !*ignore_exif {
        if let Some(exif_slice) = info.exif_metadata.as_deref() {
            if exif_slice.len() >= 4
                && (exif_slice.starts_with(b"II*\0") || exif_slice.starts_with(b"MM\0*"))
            {
                let mut exif = create_vec_exact(exif_slice.len())?;
                exif.try_extend_from_slice(exif_slice)?;
                let _ = exif::set_orientation(&mut exif, 1);
                image.exif = exif;
                *ignore_exif = true;
            }
        }
    }
    for chunk in &info.uncompressed_latin1_text[offsets.latin1..] {
        if *ignore_exif && *ignore_xmp {
            break;
        }
        apply_raw_text_metadata(
            image,
            &chunk.keyword,
            &chunk.text,
            true,
            ignore_exif,
            ignore_xmp,
        )?;
    }
    offsets.latin1 = info.uncompressed_latin1_text.len();
    for chunk in &info.compressed_latin1_text[offsets.compressed_latin1..] {
        if *ignore_exif && *ignore_xmp {
            break;
        }
        if let Ok(text) = chunk.get_text() {
            apply_raw_text_metadata(image, &chunk.keyword, &text, true, ignore_exif, ignore_xmp)?;
        }
    }
    offsets.compressed_latin1 = info.compressed_latin1_text.len();
    for chunk in &info.utf8_text[offsets.utf8..] {
        if *ignore_exif && *ignore_xmp {
            break;
        }
        if let Ok(text) = chunk.get_text() {
            apply_raw_text_metadata(image, &chunk.keyword, &text, false, ignore_exif, ignore_xmp)?;
        }
    }
    offsets.utf8 = info.utf8_text.len();
    image.remove_trailing_null_from_xmp();
    Ok(())
}

impl PngReader {
    pub fn create(filename: &str) -> AvifResult<Self> {
        Ok(Self {
            filename: filename.into(),
        })
    }
}

impl Reader for PngReader {
    fn read_frame(&mut self, config: &Config) -> AvifResult<(Image, u64, Option<GainMap>)> {
        let file = File::open(&self.filename)
            .map_err(|_| AvifError::UnknownError("failed to open file".into()))?;
        let mut decoder = png::Decoder::new_with_limits(
            BufReader::with_capacity(64 * 1024, file),
            png::Limits {
                bytes: PNG_ALLOC_LIMIT_BYTES,
            },
        );
        decoder.set_transformations(png::Transformations::EXPAND);
        if config.ignore_icc {
            decoder.set_ignore_iccp_chunk(true);
        }
        if config.ignore_exif && config.ignore_xmp {
            decoder.set_ignore_text_chunk(true);
        }

        let mut reader = decoder
            .read_info()
            .map_err(|e| AvifError::UnknownError(format!("png read_info failed: {e}")))?;

        let raw_width = reader.info().width;
        let raw_height = reader.info().height;
        if raw_width == 0
            || raw_height == 0
            || raw_width > PNG_MAX_DIMENSION
            || raw_height > PNG_MAX_DIMENSION
        {
            return Err(AvifError::UnknownError(format!(
                "Invalid PNG dimensions ({raw_width} x {raw_height})"
            )));
        }
        if config.image_size_limit > 0 {
            let total_pixels = (raw_width as u64) * (raw_height as u64);
            if total_pixels > config.image_size_limit as u64 {
                return Err(AvifError::UnknownError(format!(
                    "Too big PNG dimensions ({raw_width} x {raw_height} > {} px)",
                    config.image_size_limit
                )));
            }
        }

        let raw_color_type = reader.info().color_type;
        let raw_color_type_is_gray = matches!(
            raw_color_type,
            png::ColorType::Grayscale | png::ColorType::GrayscaleAlpha
        );
        let (out_color_type, out_bit_depth) = reader.output_color_type();
        let bit_depth = if out_bit_depth == png::BitDepth::Sixteen { 16 } else { 8 };

        let mut yuv = Image {
            width: raw_width,
            height: raw_height,
            yuv_range: config.yuv_range.unwrap_or(YuvRange::Full),
            matrix_coefficients: config
                .matrix_coefficients
                .unwrap_or(MatrixCoefficients::Bt601),
            ..Default::default()
        };
        if yuv.matrix_coefficients == MatrixCoefficients::YcgcoRo {
            return Err(AvifError::UnknownError(
                "YcgcoRo cannot be used with PNG because it has an even bit depth.".into(),
            ));
        }
        yuv.yuv_format = match config.yuv_format {
            Some(yuv_format) => yuv_format,
            None => {
                if raw_color_type_is_gray {
                    PixelFormat::Yuv400
                } else {
                    PixelFormat::Yuv444
                }
            }
        };
        yuv.depth = match config.depth {
            Some(depth) => depth,
            None => {
                if bit_depth == 8 {
                    8
                } else if config.allow_sample_transform {
                    16
                } else {
                    12
                }
            }
        };
        if yuv.matrix_coefficients == MatrixCoefficients::YcgcoRe {
            if bit_depth != 8 {
                return Err(AvifError::UnknownError(
                    "YcgcoRe cannot be used on 16 bit input because it adds two bits.".into(),
                ));
            }
            if let Some(d) = config.depth {
                if d != 10 {
                    return Err(AvifError::UnknownError(format!(
                        "Cannot request {d} bits for YCgCo-Re as it uses two extra bits"
                    )));
                }
            }
            yuv.depth = 10;
        }

        if !config.ignore_icc {
            let info = reader.info();
            if let Some(cicp) = info
                .coding_independent_code_points
                .filter(|c| c.matrix_coefficients == 0)
            {
                set_png_cicp(
                    &mut yuv,
                    &[
                        cicp.color_primaries,
                        cicp.transfer_function,
                        cicp.matrix_coefficients,
                        u8::from(cicp.is_video_full_range_image),
                    ],
                );
            } else if let Some(icc_slice) = info.icc_profile.as_deref() {
                if (!raw_color_type_is_gray && yuv.yuv_format == PixelFormat::Yuv400)
                    || (raw_color_type_is_gray && yuv.yuv_format != PixelFormat::Yuv400)
                {
                    return Err(AvifError::UnknownError(
                        "Image contains an ICC profile that is incompatible with the requested \
                         output. Pass --ignore-icc to discard the ICC profile"
                            .into(),
                    ));
                }
                yuv.icc = create_vec_exact(icc_slice.len())?;
                yuv.icc.try_extend_from_slice(icc_slice)?;
            } else if info.srgb.is_some() {
                yuv.color_primaries = ColorPrimaries::Srgb;
                yuv.transfer_characteristics = TransferCharacteristics::Srgb;
            } else {
                let mut should_generate_icc = false;
                let gamma = if let Some(g) = info.gama_chunk.filter(|g| g.into_scaled() > 0) {
                    let g_val = 1.0 / (g.into_scaled() as f64 * 0.00001);
                    yuv.transfer_characteristics =
                        TransferCharacteristics::from_gamma(g_val as f32);
                    if yuv.transfer_characteristics == TransferCharacteristics::Unknown {
                        should_generate_icc = true;
                    }
                    g_val
                } else {
                    // No gamma information in file. Assume the default CRT exponent of 2.2
                    // (PNG specification 1.2 Section 10.5).
                    2.2
                };
                let primaries = if let Some(chrm) = info.chrm_chunk {
                    let scale = |sf: png::ScaledFloat| (sf.into_scaled() as f64 * 0.00001) as f32;
                    let p = [
                        scale(chrm.red.0),
                        scale(chrm.red.1),
                        scale(chrm.green.0),
                        scale(chrm.green.1),
                        scale(chrm.blue.0),
                        scale(chrm.blue.1),
                        scale(chrm.white.0),
                        scale(chrm.white.1),
                    ];
                    yuv.color_primaries = ColorPrimaries::find(&p);
                    if yuv.color_primaries == ColorPrimaries::Unknown {
                        should_generate_icc = true;
                    }
                    p
                } else {
                    BT709_PRIMARIES
                };
                if should_generate_icc {
                    yuv.color_primaries = ColorPrimaries::Unspecified;
                    yuv.transfer_characteristics = TransferCharacteristics::Unspecified;
                    yuv.icc = icc::generate_icc(yuv.yuv_format, gamma as f32, &primaries)
                        .unwrap_or_default();
                }
            }
        }

        if let Some(cp) = config.color_primaries {
            yuv.color_primaries = cp;
        }

        let mut rgb = rgb::Image::create_from_yuv(&yuv);
        rgb.max_threads = i32::try_from(config.max_threads).unwrap_or(i32::MAX);
        rgb.depth = bit_depth;
        rgb.format = match out_color_type {
            png::ColorType::Grayscale => rgb::Format::Gray,
            png::ColorType::GrayscaleAlpha => rgb::Format::GrayA,
            png::ColorType::Rgb => rgb::Format::Rgb,
            png::ColorType::Rgba => rgb::Format::Rgba,
            png::ColorType::Indexed => {
                return Err(AvifError::UnknownError(
                    "Unexpected unexpanded palette PNG output".into(),
                ));
            }
        };
        let buf_size = reader
            .output_buffer_size()
            .ok_or_else(|| AvifError::UnknownError("png output_buffer_size overflow".into()))?;
        rgb.allocate()?;
        let expected_bytes = usize_from_u32(checked_mul!(rgb.row_bytes, rgb.height)?)?;
        if expected_bytes != buf_size {
            return Err(AvifError::UnknownError(format!(
                "rowBytes mismatch libavif {expected_bytes} vs png {buf_size}"
            )));
        }
        if bit_depth == 8 {
            let Some(Pixels::Buffer(buf)) = rgb.pixels.as_mut() else {
                return Err(AvifError::UnknownError("missing 8-bit rgb buffer".into()));
            };
            reader
                .next_frame(buf)
                .map_err(|e| AvifError::UnknownError(format!("png next_frame failed: {e}")))?;
        } else {
            let mut be_bytes: Vec<u8> = try_vec_exact![0; buf_size]?;
            reader
                .next_frame(&mut be_bytes)
                .map_err(|e| AvifError::UnknownError(format!("png next_frame failed: {e}")))?;
            let Some(Pixels::Buffer16(buf16)) = rgb.pixels.as_mut() else {
                return Err(AvifError::UnknownError("missing 16-bit rgb buffer".into()));
            };
            #[allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
            for (dst, src) in buf16.iter_mut().zip(be_bytes.chunks_exact(2)) {
                *dst = u16::from_be_bytes([src[0], src[1]]);
            }
        }

        let mut ignore_exif = config.ignore_exif;
        let mut ignore_xmp = config.ignore_xmp;
        if !ignore_exif || !ignore_xmp {
            let mut offsets = TextChunkOffsets::default();
            extract_exif_and_xmp(
                &mut yuv,
                reader.info(),
                &mut offsets,
                &mut ignore_exif,
                &mut ignore_xmp,
            )?;
            if !ignore_exif || !ignore_xmp {
                let _ = reader.finish();
                extract_exif_and_xmp(
                    &mut yuv,
                    reader.info(),
                    &mut offsets,
                    &mut ignore_exif,
                    &mut ignore_xmp,
                )?;
            }
        }
        drop(reader);
        rgb.convert_to_yuv_with_threads(&mut yuv, config.max_threads.max(1))?;
        Ok((yuv, 0, None))
    }

    fn has_more_frames(&mut self) -> bool {
        // TODO: b/403090413 - maybe support APNG?
        false
    }
}
