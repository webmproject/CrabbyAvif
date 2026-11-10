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

#![forbid(unsafe_code)]

use crate::checked_mul;
use crate::image::*;
use crate::reformat::rgb;
use crate::utils::pixels::Pixels;
use crate::utils::*;

use std::borrow::Cow;
use std::fs::File;
use std::io::BufWriter;

use super::Writer;

// Uncompressed PNG iTXt payload prefix for Adobe XMP:
// keyword ("XML:com.adobe.xmp\0"), compression_flag (0), compression_method (0),
// language_tag ("\0"), translated_keyword ("\0").
const XMP_ITXT_PREFIX: &[u8] = b"XML:com.adobe.xmp\0\0\0\0\0";

#[derive(Default)]
pub struct PngWriter {
    pub depth: Option<u8>,
    pub compression_level: Option<i32>,
}

fn png_error(err: png::EncodingError) -> AvifError {
    AvifError::UnknownError(format!("PNG encoding error: {err}"))
}

impl Writer for PngWriter {
    fn write_frame(&mut self, file: &mut File, image: &Image) -> AvifResult<()> {
        if image.width == 0 || image.height == 0 {
            return AvifError::invalid_argument();
        }
        if image.matrix_coefficients == MatrixCoefficients::YcgcoRo {
            return Err(AvifError::UnknownError(
                "YcgcoRo cannot be used with PNG because it has an even bit depth.".into(),
            ));
        }
        let mut rgb_depth = self.depth.unwrap_or(0);
        if rgb_depth == 0 {
            rgb_depth = if image.depth > 8 { 16 } else { 8 };
        }
        if image.matrix_coefficients == MatrixCoefficients::YcgcoRe {
            if image.depth != 10 {
                return Err(AvifError::UnknownError(
                    "YcgcoRe can only be used with bit depth 10.".into(),
                ));
            }
            if self.depth.unwrap_or(0) != 0 && self.depth.unwrap_or(0) != 8 {
                return Err(AvifError::UnknownError(
                    "Cannot request non-8 bits for YCgCo-Re.".into(),
                ));
            }
            rgb_depth = 8;
        }
        let bit_depth = match rgb_depth {
            8 => png::BitDepth::Eight,
            16 => png::BitDepth::Sixteen,
            _ => return AvifError::invalid_argument(),
        };
        let deflate_compression = match self.compression_level {
            None => None,
            Some(0) => Some(png::DeflateCompression::NoCompression),
            Some(level @ 1..=9) => Some(png::DeflateCompression::Level(level as u8)),
            Some(_) => return AvifError::invalid_argument(),
        };
        let copy_y_plane = image.yuv_format == PixelFormat::Yuv400
            && !image.alpha_present
            && image.depth == 8
            && rgb_depth == 8
            && image.clap.is_none()
            && image.irot_angle.is_none()
            && image.imir_axis.is_none();
        let mut rgb = rgb::Image::create_from_yuv(image);
        let color_type = if copy_y_plane {
            png::ColorType::Grayscale
        } else {
            rgb.depth = rgb_depth;
            let color_type = match (image.yuv_format, image.alpha_present) {
                (PixelFormat::Yuv400, true) => {
                    rgb.format = rgb::Format::GrayA;
                    png::ColorType::GrayscaleAlpha
                }
                (PixelFormat::Yuv400, false) => {
                    rgb.format = rgb::Format::Gray;
                    png::ColorType::Grayscale
                }
                (_, true) => {
                    // TODO - b/479429854: Support specifying chroma upsampling.
                    if image.is_opaque()? {
                        rgb.format = rgb::Format::Rgb;
                        png::ColorType::Rgb
                    } else {
                        png::ColorType::Rgba
                    }
                }
                (_, false) => {
                    // TODO - b/479429854: Support specifying chroma upsampling.
                    rgb.format = rgb::Format::Rgb;
                    png::ColorType::Rgb
                }
            };
            rgb.allocate()?;
            rgb.convert_from_yuv(image)?;
            color_type
        };

        let mut info = png::Info::with_size(image.width, image.height);
        info.color_type = color_type;
        info.bit_depth = bit_depth;
        let is_srgb = image.icc.is_empty()
            && image.color_primaries == ColorPrimaries::Srgb
            && image.transfer_characteristics == TransferCharacteristics::Srgb;
        if image.icc.is_empty() {
            if !is_srgb {
                if let Some(primaries) = image.color_primaries.values() {
                    let [rx, ry, gx, gy, bx, by, wx, wy] = primaries;
                    info.source_chromaticities = Some(png::SourceChromaticities::new(
                        (wx, wy),
                        (rx, ry),
                        (gx, gy),
                        (bx, by),
                    ));
                }
                if let Some(gamma) = image.transfer_characteristics.gamma() {
                    info.source_gamma = Some(png::ScaledFloat::new(1.0 / gamma));
                }
            }
        } else {
            // If there is an ICC profile, the CICP values are irrelevant and only the ICC
            // profile is written. If we could extract the primaries/transfer curve from the
            // ICC profile, then they could be written in cHRM/gAMA chunks.
            info.icc_profile = Some(Cow::Borrowed(&image.icc));
        }
        if !image.exif.is_empty() {
            info.exif_metadata = Some(Cow::Borrowed(&image.exif));
        }

        let mut encoder = png::Encoder::with_info(BufWriter::new(file), info).map_err(png_error)?;
        if is_srgb {
            encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
        }
        if let Some(compression) = deflate_compression {
            encoder.set_deflate_compression(compression);
        }

        let mut writer = encoder.write_header().map_err(png_error)?;
        if image.icc.is_empty() {
            let cicp_data: [u8; 4] = [
                image.color_primaries as u8,
                image.transfer_characteristics as u8,
                MatrixCoefficients::Identity as u8,
                1, // full_range
            ];
            writer
                .write_chunk(png::chunk::cICP, &cicp_data)
                .map_err(png_error)?;
        }
        if !image.xmp.is_empty() {
            let xmp_len = XMP_ITXT_PREFIX
                .len()
                .checked_add(image.xmp.len())
                .ok_or(AvifError::OutOfMemory)?;
            let mut xmp_chunk = create_vec_exact(xmp_len)?;
            xmp_chunk.try_extend_from_slice(XMP_ITXT_PREFIX)?;
            xmp_chunk.try_extend_from_slice(&image.xmp)?;
            writer
                .write_chunk(png::chunk::iTXt, &xmp_chunk)
                .map_err(png_error)?;
        }

        if copy_y_plane {
            let width_usize = usize_from_u32(image.width)?;
            let total_len = usize_from_u32(checked_mul!(image.width, image.height)?)?;
            let mut y_bytes: Vec<u8> = create_vec_exact(total_len)?;
            for y in 0..image.height {
                let row = image
                    .row(Plane::Y, y)?
                    .get(..width_usize)
                    .ok_or(AvifError::NoContent)?;
                y_bytes.try_extend_from_slice(row)?;
            }
            writer.write_image_data(&y_bytes).map_err(png_error)?;
        } else if rgb_depth == 8 {
            writer
                .write_image_data(rgb.pixel_slice()?)
                .map_err(png_error)?;
        } else {
            let Some(Pixels::Buffer16(buf16)) = rgb.pixels.as_ref() else {
                return Err(AvifError::NoContent);
            };
            let byte_len = buf16.len().checked_mul(2).ok_or(AvifError::OutOfMemory)?;
            let mut be_bytes: Vec<u8> = try_vec_exact![0; byte_len]?;
            for (dst, &sample) in be_bytes.chunks_exact_mut(2).zip(buf16.iter()) {
                dst.copy_from_slice(&sample.to_be_bytes());
            }
            writer.write_image_data(&be_bytes).map_err(png_error)?;
        }

        writer.finish().map_err(png_error)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufReader;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TempPngPath(PathBuf);

    impl TempPngPath {
        fn new(label: &str) -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            Self(std::env::temp_dir().join(format!(
                "crabbyavif_png_writer_{label}_{}_{id}.png",
                std::process::id()
            )))
        }
    }

    impl Drop for TempPngPath {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn write_frame_rejects_zero_dimensions() {
        let tmp = TempPngPath::new("zero_dim");
        let mut file = File::create(&tmp.0).unwrap();

        let mut writer = PngWriter::default();
        assert!(matches!(
            writer.write_frame(&mut file, &Image::default()),
            Err(AvifError::InvalidArgument)
        ));

        let yuv400_zero_w = Image {
            width: 0,
            height: 4,
            depth: 8,
            yuv_format: PixelFormat::Yuv400,
            ..Image::default()
        };
        assert!(matches!(
            writer.write_frame(&mut file, &yuv400_zero_w),
            Err(AvifError::InvalidArgument)
        ));

        let yuv400_zero_h = Image {
            width: 4,
            height: 0,
            depth: 8,
            yuv_format: PixelFormat::Yuv400,
            ..Image::default()
        };
        assert!(matches!(
            writer.write_frame(&mut file, &yuv400_zero_h),
            Err(AvifError::InvalidArgument)
        ));
    }

    #[test]
    fn write_frame_rejects_invalid_configs() {
        let tmp = TempPngPath::new("invalid_cfg");
        let mut file = File::create(&tmp.0).unwrap();

        let mut writer = PngWriter::default();
        let ycgco_ro = Image {
            width: 2,
            height: 2,
            depth: 8,
            matrix_coefficients: MatrixCoefficients::YcgcoRo,
            ..Image::default()
        };
        assert!(matches!(
            writer.write_frame(&mut file, &ycgco_ro),
            Err(AvifError::UnknownError(_))
        ));

        let ycgco_re_bad_depth = Image {
            width: 2,
            height: 2,
            depth: 8,
            matrix_coefficients: MatrixCoefficients::YcgcoRe,
            ..Image::default()
        };
        assert!(matches!(
            writer.write_frame(&mut file, &ycgco_re_bad_depth),
            Err(AvifError::UnknownError(_))
        ));

        let ycgco_re_16bit_out = Image {
            width: 2,
            height: 2,
            depth: 10,
            matrix_coefficients: MatrixCoefficients::YcgcoRe,
            ..Image::default()
        };
        writer.depth = Some(16);
        assert!(matches!(
            writer.write_frame(&mut file, &ycgco_re_16bit_out),
            Err(AvifError::UnknownError(_))
        ));

        writer.depth = Some(10);
        let mut valid_img = Image {
            width: 2,
            height: 2,
            depth: 8,
            yuv_format: PixelFormat::Yuv444,
            ..Image::default()
        };
        valid_img.allocate_planes(Category::Color).unwrap();
        assert!(matches!(
            writer.write_frame(&mut file, &valid_img),
            Err(AvifError::InvalidArgument)
        ));

        writer.depth = None;
        writer.compression_level = Some(10);
        assert!(matches!(
            writer.write_frame(&mut file, &valid_img),
            Err(AvifError::InvalidArgument)
        ));
    }

    #[test]
    fn write_frame_roundtrips_strided_gray_and_metadata() -> AvifResult<()> {
        let tmp = TempPngPath::new("gray_meta");
        // Strided Yuv400 image (`row_bytes = 5` > `width = 3`) exercises `copy_y_plane`.
        let mut y_buf = try_vec_exact![0u8; 5 * 2]?;
        y_buf[..3].copy_from_slice(&[10, 20, 30]);
        y_buf[5..8].copy_from_slice(&[40, 50, 60]);
        let exif_payload = b"Exif\0\0MM\0*";
        let xmp_payload = b"<x:xmpmeta/>";
        let mut image = Image {
            width: 3,
            height: 2,
            depth: 8,
            yuv_format: PixelFormat::Yuv400,
            yuv_range: YuvRange::Full,
            color_primaries: ColorPrimaries::Srgb,
            transfer_characteristics: TransferCharacteristics::Srgb,
            exif: create_vec_exact(exif_payload.len())?,
            xmp: create_vec_exact(xmp_payload.len())?,
            ..Image::default()
        };
        image.exif.try_extend_from_slice(exif_payload)?;
        image.xmp.try_extend_from_slice(xmp_payload)?;
        image.planes[0] = Some(Pixels::Buffer(y_buf));
        image.row_bytes[0] = 5;

        {
            let mut file = File::create(&tmp.0).unwrap();
            let mut writer = PngWriter {
                depth: None,
                compression_level: Some(0),
            };
            writer.write_frame(&mut file, &image)?;
        }

        let decoder = png::Decoder::new(BufReader::new(File::open(&tmp.0).unwrap()));
        let mut reader = decoder.read_info().unwrap();
        let info = reader.info();
        assert_eq!(info.width, 3);
        assert_eq!(info.height, 2);
        assert_eq!(info.color_type, png::ColorType::Grayscale);
        assert_eq!(info.bit_depth, png::BitDepth::Eight);
        assert!(info.srgb.is_some());
        let cicp = info.coding_independent_code_points.unwrap();
        assert_eq!(cicp.color_primaries, ColorPrimaries::Srgb as u8);
        assert_eq!(cicp.transfer_function, TransferCharacteristics::Srgb as u8);
        assert_eq!(cicp.matrix_coefficients, 0);
        assert!(cicp.is_video_full_range_image);
        assert_eq!(info.exif_metadata.as_deref(), Some(exif_payload.as_slice()));
        assert_eq!(info.utf8_text.len(), 1);
        assert_eq!(info.utf8_text[0].keyword, "XML:com.adobe.xmp");
        assert_eq!(info.utf8_text[0].get_text().unwrap(), "<x:xmpmeta/>");

        let mut out = try_vec_exact![0u8; reader.output_buffer_size().unwrap()]?;
        let frame_info = reader.next_frame(&mut out).unwrap();
        assert_eq!(&out[..frame_info.buffer_size()], &[10, 20, 30, 40, 50, 60]);
        Ok(())
    }

    #[test]
    fn write_frame_roundtrips_16bit_rgb_with_iccp_and_cicp() {
        let tmp = TempPngPath::new("rgb16");
        let mut image = Image {
            width: 2,
            height: 2,
            depth: 10,
            yuv_format: PixelFormat::Yuv444,
            yuv_range: YuvRange::Full,
            color_primaries: ColorPrimaries::Bt2020,
            transfer_characteristics: TransferCharacteristics::Bt709,
            ..Image::default()
        };
        image.allocate_planes(Category::Color).unwrap();
        let icc_bytes = b"fake-icc-profile-bytes";
        image.icc = create_vec_exact(icc_bytes.len()).unwrap();
        image.icc.try_extend_from_slice(icc_bytes).unwrap();

        {
            let mut file = File::create(&tmp.0).unwrap();
            let mut writer = PngWriter {
                depth: Some(16),
                compression_level: Some(9),
            };
            writer.write_frame(&mut file, &image).unwrap();
        }

        let decoder = png::Decoder::new(BufReader::new(File::open(&tmp.0).unwrap()));
        let reader = decoder.read_info().unwrap();
        let info = reader.info();
        assert_eq!(info.color_type, png::ColorType::Rgb);
        assert_eq!(info.bit_depth, png::BitDepth::Sixteen);
        assert_eq!(info.icc_profile.as_deref(), Some(icc_bytes.as_slice()));
        assert!(info.coding_independent_code_points.is_none());
    }
}
