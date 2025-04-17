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

use crate::reformat::*;
use crate::utils::pixels::Pixels;
use crate::AvifError;
use crate::AvifResult;
use crate::*;

use super::Reader;

use std::fs::File;
use std::io::BufReader;

use ::image::codecs::jpeg;
use ::image::ColorType;
use ::image::ImageDecoder;

pub(crate) struct JpegReader {
    filename: String,
}

impl JpegReader {
    pub(crate) fn create(filename: &str) -> AvifResult<Self> {
        Ok(Self {
            filename: filename.into(),
        })
    }
}

impl Reader for JpegReader {
    fn read_frame(&mut self) -> AvifResult<Image> {
        let mut reader = BufReader::new(File::open(self.filename.clone()).or(Err(
            AvifError::UnknownError("error opening input file".into()),
        ))?);
        let decoder = jpeg::JpegDecoder::new(&mut reader).or(Err(AvifError::UnknownError(
            "failed to create jpeg decoder".into(),
        )))?;
        let color_type = decoder.color_type();
        if color_type != ColorType::Rgb8 {
            return Err(AvifError::UnknownError(format!(
                "jpeg color type was something other than rgb8: {:#?}",
                color_type
            )));
        }
        let (width, height) = decoder.dimensions();
        let total_bytes = decoder.total_bytes() as usize;
        let mut rgb_bytes = vec![0u8; total_bytes];
        decoder
            .read_image(&mut rgb_bytes)
            .or(Err(AvifError::UnknownError(
                "failed to read jpeg pixels".into(),
            )))?;
        let rgb = rgb::Image {
            width,
            height,
            depth: 8,
            format: rgb::Format::Rgb,
            pixels: Some(Pixels::Buffer(rgb_bytes)),
            row_bytes: width * 3,
            ..Default::default()
        };
        let mut yuv = Image {
            width,
            height,
            depth: 8,
            yuv_format: PixelFormat::Yuv420,
            yuv_range: YuvRange::Full,
            matrix_coefficients: MatrixCoefficients::Bt601,
            ..Default::default()
        };
        rgb.convert_to_yuv(&mut yuv)?;
        Ok(yuv)
    }

    fn has_more_frames(&mut self) -> bool {
        false
    }
}
