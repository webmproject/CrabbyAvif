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

use crabby_avif::image::*;
use crabby_avif::reformat::rgb;
use crabby_avif::AvifError;
use crabby_avif::AvifResult;
use crabby_avif::PixelFormat;

use std::fs::File;

use super::Writer;

use png;

#[derive(Default)]
pub(crate) struct PngWriter;

impl Writer for PngWriter {
    fn write_frame(&mut self, file: &mut File, image: &Image) -> AvifResult<()> {
        let is_monochrome = image.yuv_format == PixelFormat::Yuv400;
        let png_color_type = match (is_monochrome, image.alpha_present) {
            (true, _) => png::ColorType::Grayscale,
            (_, false) => png::ColorType::Rgb,
            (_, true) => png::ColorType::Rgba,
        };
        let mut rgb = rgb::Image::create_from_yuv(image);
        if !is_monochrome {
            rgb.depth = if image.depth == 8 { 8 } else { 16 };
            rgb.format = if image.alpha_present { rgb::Format::Rgba } else { rgb::Format::Rgb };
            rgb.allocate()?;
            rgb.convert_from_yuv(image)?;
        }

        let mut encoder = png::Encoder::new(file, image.width, image.height);
        encoder.set_color(png_color_type);
        encoder.set_depth(if image.depth == 8 {
            png::BitDepth::Eight
        } else {
            png::BitDepth::Sixteen
        });
        let mut writer = encoder.write_header().or(Err(AvifError::UnknownError(
            "Could not write the PNG header".into(),
        )))?;
        let mut rgba_pixel_buffer: Vec<u8> = Vec::new();
        let rgba_slice = if is_monochrome {
            for y in 0..image.height {
                if image.depth == 8 {
                    let y_row = image.row(Plane::Y, y)?;
                    rgba_pixel_buffer.extend_from_slice(&y_row[..image.width as usize]);
                } else {
                    let y_row = image.row16(Plane::Y, y)?;
                    for pixel in &y_row[..image.width as usize] {
                        // Scale the pixel to 16 bits.
                        let pixel16 = ((*pixel as u32 * 65535) / image.max_channel() as u32) as u16;
                        rgba_pixel_buffer.extend_from_slice(&pixel16.to_be_bytes());
                    }
                }
            }
            &rgba_pixel_buffer[..]
        } else if image.depth == 8 {
            let rgba_pixels = rgb.pixels.as_ref().unwrap();
            rgba_pixels.slice(0, rgba_pixels.size() as u32)?
        } else {
            let rgba_pixels = rgb.pixels.as_ref().unwrap();
            let rgba_slice16 = rgba_pixels.slice16(0, rgba_pixels.size() as u32).unwrap();
            for pixel in rgba_slice16 {
                rgba_pixel_buffer.extend_from_slice(&pixel.to_be_bytes());
            }
            &rgba_pixel_buffer[..]
        };
        writer
            .write_image_data(rgba_slice)
            .or(Err(AvifError::UnknownError(
                "Could not write PNG image data".into(),
            )))?;
        writer.finish().or(Err(AvifError::UnknownError(
            "Could not finalize the PNG encoder".into(),
        )))?;
        Ok(())
    }
}
