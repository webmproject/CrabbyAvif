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

//! Wrapper around crate::decoder::av1, a pure Rust AV1 keyframe-only (still image) decoder.
//!
//! This is currently only a skeleton: all the decoding methods return NotImplemented.

use crate::codecs::Decoder;
use crate::codecs::DecoderConfig;
use crate::decoder::item::Item;
use crate::decoder::GridImageHelper;
use crate::image::Image;
use crate::internal_utils::*;
use crate::*;

#[derive(Default)]
pub struct Native {
    config: Option<DecoderConfig>,
}

impl Native {
    pub(crate) fn version() -> String {
        "native".into()
    }
}

impl Decoder for Native {
    fn codec(&self) -> CodecChoice {
        CodecChoice::Native
    }

    fn initialize(&mut self, config: &DecoderConfig) -> AvifResult<()> {
        self.config = Some(DecoderConfig {
            codec_config: config.codec_config.try_clone()?,
            ..*config
        });
        Ok(())
    }

    fn get_next_image(
        &mut self,
        _av1_payload: &[u8],
        _spatial_id: u8,
        _image: &mut Image,
        _category: Category,
        _item: Option<&Item>,
        _signal_eos: bool,
    ) -> AvifResult<()> {
        AvifError::not_implemented()
    }

    fn get_next_image_grid(
        &mut self,
        _payloads: &[Vec<u8>],
        _spatial_id: u8,
        _grid_image_helper: &mut GridImageHelper,
    ) -> AvifResult<()> {
        AvifError::not_implemented()
    }

    fn get_last_image(
        &mut self,
        _payloads: &[Vec<u8>],
        _spatial_id: u8,
        _image: &mut Image,
        _category: Category,
    ) -> AvifResult<()> {
        AvifError::not_implemented()
    }
}
