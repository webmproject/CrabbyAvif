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

use crate::*;

// To be used instead of direct AvifError enum variants in order to debug
// unexpected Err propagations as early as possible in the call stack.
impl AvifError {
    fn on_error() {
        // Use std::intrinsics::breakpoint() or manually add a breakpoint here.
        // Alternatively, uncomment the following to print the stack trace.
        // println!("{}", std::backtrace::Backtrace::force_capture());
    }

    pub fn invalid_ftyp<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::InvalidFtyp)
    }
    pub fn no_content<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::NoContent)
    }
    pub fn no_yuv_format_selected<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::NoYuvFormatSelected)
    }
    pub fn reformat_failed<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::ReformatFailed)
    }
    pub fn unsupported_depth<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::UnsupportedDepth)
    }
    pub fn encode_color_failed<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::EncodeColorFailed)
    }
    pub fn encode_alpha_failed<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::EncodeAlphaFailed)
    }

    pub fn missing_image_item<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::MissingImageItem)
    }
    pub fn decode_color_failed<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::DecodeColorFailed)
    }
    pub fn decode_alpha_failed<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::DecodeAlphaFailed)
    }
    pub fn color_alpha_size_mismatch<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::ColorAlphaSizeMismatch)
    }
    pub fn ispe_size_mismatch<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::IspeSizeMismatch)
    }
    pub fn no_codec_available<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::NoCodecAvailable)
    }
    pub fn no_images_remaining<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::NoImagesRemaining)
    }
    pub fn invalid_exif_payload<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::InvalidExifPayload)
    }

    pub fn invalid_codec_specific_option<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::InvalidCodecSpecificOption)
    }
    pub fn truncated_data<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::TruncatedData)
    }
    pub fn io_not_set<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::IoNotSet)
    }
    pub fn io_error<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::IoError)
    }
    pub fn waiting_on_io<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::WaitingOnIo)
    }
    pub fn invalid_argument<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::InvalidArgument)
    }
    pub fn not_implemented<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::NotImplemented)
    }
    pub fn out_of_memory<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::OutOfMemory)
    }
    pub fn cannot_change_setting<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::CannotChangeSetting)
    }
    pub fn incompatible_image<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::IncompatibleImage)
    }
    pub fn encode_gain_map_failed<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::EncodeGainMapFailed)
    }
    pub fn decode_gain_map_failed<T>() -> Result<T, AvifError> {
        AvifError::on_error();
        Err(AvifError::DecodeGainMapFailed)
    }

    pub fn unknown_error<T, O>(object: O) -> Result<T, AvifError>
    where
        O: std::fmt::Display,
    {
        AvifError::on_error();
        Err(AvifError::UnknownError(object.to_string()))
    }
    pub fn bmff_parse_failed<T, O>(object: O) -> Result<T, AvifError>
    where
        O: std::fmt::Display,
    {
        AvifError::on_error();
        Err(AvifError::BmffParseFailed(object.to_string()))
    }
    pub fn invalid_image_grid<T, O>(object: O) -> Result<T, AvifError>
    where
        O: std::fmt::Display,
    {
        AvifError::on_error();
        Err(AvifError::InvalidImageGrid(object.to_string()))
    }
    pub fn invalid_tone_mapped_image<T, O>(object: O) -> Result<T, AvifError>
    where
        O: std::fmt::Display,
    {
        AvifError::on_error();
        Err(AvifError::InvalidToneMappedImage(object.to_string()))
    }

    pub fn map_unknown_error<O>(object: O) -> AvifError
    where
        O: std::fmt::Display,
    {
        AvifError::on_error();
        AvifError::UnknownError(object.to_string())
    }
}
