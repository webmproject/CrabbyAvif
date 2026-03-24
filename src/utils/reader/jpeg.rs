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

use super::Config;
use super::Reader;

use std::fs::File;
use std::io::BufReader;

use zune_jpeg::JpegDecoder;

pub struct JpegReader {
    filename: String,
}

impl JpegReader {
    pub fn create(filename: &str) -> AvifResult<Self> {
        Ok(Self {
            filename: filename.into(),
        })
    }
}

impl Reader for JpegReader {
    fn read_frame(&mut self, config: &Config) -> AvifResult<(Image, u64, Option<GainMap>)> {
        let file = File::open(self.filename.clone()).map_err(AvifError::map_unknown_error)?;
        let mut decoder = JpegDecoder::new(BufReader::new(file));
        decoder
            .decode_headers()
            .map_err(|e| AvifError::UnknownError(format!("jpeg header decode error: {e:?}")))?;
        let info = decoder
            .info()
            .ok_or(AvifError::UnknownError("jpeg info not found".into()))?;
        if info.components != 3 {
            return AvifError::unknown_error(format!(
                "jpeg components was something other than 3: {}",
                info.components
            ));
        }
        let width = info.width as u32;
        let height = info.height as u32;
        let icc = decoder.icc_profile().unwrap_or_default();
        let exif = info.exif_data.clone().unwrap_or_default();
        let xmp = info.xmp_data.clone().unwrap_or_default();
        let (irot_angle, imir_axis) = exif::get_orientation(&exif)?;
        let rgb_bytes = decoder
            .decode()
            .map_err(|e| AvifError::UnknownError(format!("jpeg decode error: {e:?}")))?;
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
            depth: config.depth.unwrap_or(8),
            yuv_format: config.yuv_format.unwrap_or(PixelFormat::Yuv420),
            yuv_range: YuvRange::Full,
            matrix_coefficients: config
                .matrix_coefficients
                .unwrap_or(MatrixCoefficients::Bt601),
            icc,
            exif,
            xmp,
            irot_angle,
            imir_axis,
            ..Default::default()
        };
        rgb.convert_to_yuv(&mut yuv)?;
        Ok((yuv, 0, None))
    }

    fn has_more_frames(&mut self) -> bool {
        false
    }
}
