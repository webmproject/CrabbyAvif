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

#[cfg(feature = "dav1d")]
pub mod dav1d;

#[cfg(feature = "android_mediacodec")]
pub mod android_mediacodec;

#[cfg(feature = "aom")]
pub mod aom;

#[cfg(feature = "avm")]
pub mod avm;

#[cfg(feature = "jpegxl")]
pub mod libjxl;

#[cfg(feature = "native_av1")]
pub mod native;

#[cfg(feature = "encoder")]
use crate::internal_utils::*;
#[cfg(feature = "encoder")]
use crate::AvifError;

use crate::decoder::item::Item;
use crate::decoder::GridImageHelper;
use crate::image::Image;
use crate::parser::mp4box::CodecConfiguration;
use crate::AndroidMediaCodecOutputColorFormat;
use crate::AvifResult;
use crate::Category;
use crate::CodecChoice;

#[cfg(feature = "encoder")]
use crate::encoder::*;

use std::num::NonZero;

// Not all fields of this struct are used in all the configurations.
#[allow(dead_code)]
pub(crate) struct DecoderConfig {
    pub operating_point: u8,
    pub all_layers: bool,
    pub width: u32,
    pub height: u32,
    pub depth: u8,
    pub max_threads: u32,
    pub image_size_limit: Option<NonZero<u32>>,
    pub max_input_size: usize,
    pub codec_config: CodecConfiguration,
    pub category: Category,
    pub android_mediacodec_output_color_format: AndroidMediaCodecOutputColorFormat,
}

pub(crate) trait Decoder {
    fn codec(&self) -> CodecChoice;
    fn initialize(&mut self, config: &DecoderConfig) -> AvifResult<()>;
    // Decode a single image and write the output into |image|.
    fn get_next_image(
        &mut self,
        av1_payload: &[u8],
        spatial_id: u8,
        image: &mut Image,
        category: Category,
        item: Option<&Item>,
        signal_eos: bool,
    ) -> AvifResult<()>;
    // Decode a list of input images and outputs them into the |grid_image_helper|.
    fn get_next_image_grid(
        &mut self,
        payloads: &[Vec<u8>],
        spatial_id: u8,
        grid_image_helper: &mut GridImageHelper,
    ) -> AvifResult<()>;
    // Decode a list of input images and write the output of the last decoded image into |image|.
    fn get_last_image(
        &mut self,
        payloads: &[Vec<u8>],
        spatial_id: u8,
        image: &mut Image,
        category: Category,
    ) -> AvifResult<()>;
    // Destruction must be implemented using Drop.
}

#[cfg(feature = "encoder")]
impl TryClone for CodecSpecificOptions {
    fn try_clone(&self) -> AvifResult<Self> {
        let mut cloned = CodecSpecificOptions::default();
        cloned
            .try_reserve(self.len())
            .map_err(AvifError::map_out_of_memory)?;
        for (key, value) in self {
            cloned.insert((key.0, key.1.try_clone()?), value.try_clone()?);
        }
        Ok(cloned)
    }
}

// Not all fields of this struct are used in all the configurations.
#[allow(dead_code)]
#[cfg(feature = "encoder")]
#[derive(Default, PartialEq)]
pub(crate) struct EncoderConfig {
    pub tile_rows_log2: i32,
    pub tile_columns_log2: i32,
    pub quality: f32, // From 0 (maximum distortion) to 100 (lossless) inclusive.
    pub quality_alpha: f32,
    pub disable_lagged_output: bool,
    pub is_single_image: bool,
    pub speed: Option<u32>,
    pub extra_layer_count: u32,
    pub threads: u32,
    pub keyframe_interval: i32,
    pub scaling_mode: ScalingMode,
    pub codec_specific_options: CodecSpecificOptions,
}

#[cfg(feature = "encoder")]
impl TryClone for EncoderConfig {
    fn try_clone(&self) -> AvifResult<Self> {
        Ok(Self {
            codec_specific_options: self.codec_specific_options.try_clone()?,
            ..*self
        })
    }
}

#[cfg(feature = "encoder")]
#[allow(dead_code)] // These functions are used only in tests and when aom is enabled.
impl EncoderConfig {
    pub(crate) fn codec_specific_option(&self, category: Category, key: &str) -> Option<&str> {
        for (k, v) in &self.codec_specific_options {
            if k.0 == Some(category) && k.1 == key {
                return Some(v.as_str());
            }
        }
        for (k, v) in &self.codec_specific_options {
            if k.0.is_none() && k.1 == key {
                return Some(v.as_str());
            }
        }
        None
    }

    pub(crate) fn codec_specific_options(
        &self,
        category: Category,
    ) -> AvifResult<Vec<(&str, &str)>> {
        self.codec_specific_options
            .iter()
            .filter(|(key, _value)| {
                // If there is a key in a requested category, return it. Otherwise, return the
                // value from the "None" category only if there is no value in the requested
                // category.
                key.0 == Some(category)
                    || (key.0.is_none()
                        && !self
                            .codec_specific_options
                            .iter()
                            .any(|(k, _)| k.0 == Some(category) && k.1 == key.1))
            })
            .map(|(key, value)| (key.1.as_str(), value.as_str()))
            .try_collect()
    }
}

#[cfg(feature = "encoder")]
pub(crate) trait Encoder {
    fn encode_image(
        &mut self,
        image: &Image,
        category: Category,
        config: &EncoderConfig,
        output_samples: &mut Vec<crate::encoder::Sample>,
    ) -> AvifResult<()>;
    fn finish(&mut self, output_samples: &mut Vec<crate::encoder::Sample>) -> AvifResult<()>;
    fn get_codec_config(
        &self,
        image: &Image,
        is_single_image: bool,
        is_lossless: bool,
        output_samples: &[crate::encoder::Sample],
    ) -> AvifResult<CodecConfiguration>;
    // Destruction must be implemented using Drop.
}

#[cfg(feature = "encoder")]
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn codec_specific_options() {
        let codec_specific_options = HashMap::from([
            (
                (Some(Category::Color), String::from("abcd")),
                String::from("color_value1"),
            ),
            ((None, String::from("abcd")), String::from("generic_value1")),
            (
                (Some(Category::Alpha), String::from("efgh")),
                String::from("alpha_value1"),
            ),
            ((None, String::from("hjkl")), String::from("generic_value2")),
        ]);
        let config = EncoderConfig {
            codec_specific_options,
            ..Default::default()
        };

        assert_eq!(
            config.codec_specific_option(Category::Color, "abcd"),
            Some("color_value1"),
        );
        assert_eq!(
            config.codec_specific_option(Category::Alpha, "abcd"),
            Some("generic_value1"),
        );
        assert_eq!(
            config.codec_specific_option(Category::Gainmap, "abcd"),
            Some("generic_value1"),
        );
        assert_eq!(
            config.codec_specific_option(Category::Color, "hjkl"),
            Some("generic_value2"),
        );

        let mut actual = config.codec_specific_options(Category::Color).unwrap();
        actual.sort();
        let mut expected = vec![("hjkl", "generic_value2"), ("abcd", "color_value1")];
        expected.sort();
        assert_eq!(expected, actual);

        actual = config.codec_specific_options(Category::Alpha).unwrap();
        actual.sort();
        expected = vec![
            ("hjkl", "generic_value2"),
            ("efgh", "alpha_value1"),
            ("abcd", "generic_value1"),
        ];
        expected.sort();
        assert_eq!(expected, actual);
    }
}
