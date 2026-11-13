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

#![allow(non_upper_case_globals)]

use crate::codecs::*;
use crate::encoder::Sample;
use crate::image::Image;
use crate::internal_utils::stream::OStreamLittleEndian;
use crate::internal_utils::*;
#[cfg(test)]
use crate::parser::mp4box::ColorInformation;
use crate::parser::mp4box::ItemProperty;
use crate::parser::mp4box::JpegXlCodecConfiguration;
#[cfg(test)]
use crate::parser::mp4box::PixelInformation;
#[cfg(test)]
use crate::parser::mp4box::PlanePixelInformation;
use crate::reformat::rgb::Format;
use crate::utils::pixels::ChannelIdc;
use crate::*;

// Encoding relies on the libjxl C API.
use libjxl_sys::bindings::*;

// Decoding relies on jxl-rs, the pure Rust JPEG XL decoder. Its API module is aliased because
// several of its type names (JxlDecoder, JxlPixelFormat etc.) collide with the libjxl C bindings
// glob-imported above.
use jxl::api as jxl_api;
use jxl_api::ExtraChannel;

use std::mem::MaybeUninit;
use std::ptr::null;
use std::ptr::NonNull;

#[derive(Default)]
pub struct Libjxl {
    // Encoding
    encoder: Option<NonNull<JxlEncoder>>,
    expected_header: Option<Vec<u8>>,

    // Decoding
    decoder: Option<jxl_api::JxlDecoder>,
    reconstructed_jxl: Option<Vec<u8>>,
    // Number of bytes of |reconstructed_jxl| that were already consumed by |decoder|.
    consumed_bytes: usize,
}

// Convenient error mapping.
trait JxlEncoderStatusTrait {
    fn map_enc_err(self, encoder: NonNull<JxlEncoder>) -> Result<(), AvifError>;
}
impl JxlEncoderStatusTrait for JxlEncoderStatus {
    fn map_enc_err(self, encoder: NonNull<JxlEncoder>) -> Result<(), AvifError> {
        match self {
            JxlEncoderStatus_JXL_ENC_SUCCESS => Ok(()),
            JxlEncoderStatus_JXL_ENC_ERROR => {
                // SAFETY: `encoder` is a valid, non-null `JxlEncoder` created by
                // `JxlEncoderCreate` and kept alive by the calling `Libjxl` instance.
                let error = unsafe { JxlEncoderGetError(encoder.as_ptr()) };
                AvifError::unknown_error(format!("JxlEncoderError {error}",))
            }
            _ => AvifError::unknown_error(format!("Unexpected JxlEncoderStatus {self}")),
        }
    }
}

fn map_dec_err(error: jxl::error::Error) -> AvifError {
    AvifError::UnknownError(format!("jxl-rs error: {error}"))
}

// The whole bitstream is available when decoding starts, so any jxl-rs operation is expected to
// reach its target event without asking for more input. Anything else means the payload was
// truncated or the decoder reached an unexpected state.
fn expect_event(
    result: jxl::error::Result<jxl_api::Event>,
    expected: impl FnOnce(&jxl_api::Event) -> bool,
) -> AvifResult<()> {
    match result.map_err(map_dec_err)? {
        jxl_api::Event::NeedMoreInput { size_hint } => AvifError::unknown_error(format!(
            "Truncated JPEG XL bitstream ({size_hint} more bytes needed)"
        )),
        event if expected(&event) => Ok(()),
        event => AvifError::unknown_error(format!("Unexpected jxl-rs event: {event:?}")),
    }
}

// Returns the jxl-rs pixel format that makes the decoder output samples directly into |rgb|.
fn rgb_image_to_jxl_rs_pixel_format(
    rgb: &reformat::rgb::Image,
    num_extra_channels: usize,
) -> AvifResult<jxl_api::JxlPixelFormat> {
    let color_data_format = match rgb.depth {
        8 => jxl_api::JxlDataFormat::U8 { bit_depth: 8 },
        10 | 12 | 16 => jxl_api::JxlDataFormat::U16 {
            endianness: jxl_api::Endianness::native(),
            bit_depth: rgb.depth,
        },
        _ => return AvifError::unknown_error(format!("Unexpected depth {}", rgb.depth)),
    };
    Ok(jxl_api::JxlPixelFormat {
        color_type: match (rgb.format.is_gray(), rgb.has_alpha()) {
            (false, false) => jxl_api::JxlColorType::Rgb,
            (false, true) => jxl_api::JxlColorType::Rgba,
            (true, false) => jxl_api::JxlColorType::Grayscale,
            (true, true) => jxl_api::JxlColorType::GrayscaleAlpha,
        },
        color_data_format: Some(color_data_format),
        // None means "do not output that extra channel to a dedicated buffer". The alpha channel,
        // if any, is interleaved with the color channels instead.
        extra_channel_format: try_vec_exact![None; num_extra_channels]?,
    })
}

fn rgb_image_to_jxl_pixel_format(
    rgb: &reformat::rgb::Image,
) -> Result<(JxlPixelFormat, usize), AvifError> {
    if rgb.width == 0 || rgb.height == 0 {
        return AvifError::invalid_argument();
    }
    let data_type = match rgb.depth {
        8 => JxlDataType_JXL_TYPE_UINT8,
        9..16 => JxlDataType_JXL_TYPE_UINT16,
        _ => return AvifError::unknown_error(format!("Unexpected depth {}", rgb.depth)),
    };
    let min_row_bytes = checked_mul!(rgb.width, rgb.pixel_size())?;
    if rgb.row_bytes < min_row_bytes {
        return AvifError::unknown_error(format!(
            "Invalid row_bytes {} < min_row_bytes {}",
            rgb.row_bytes, min_row_bytes
        ));
    }
    let row_bytes = usize_from_u32(rgb.row_bytes)?;
    let min_row_bytes_usize = usize_from_u32(min_row_bytes)?;
    let height = usize_from_u32(rgb.height)?;
    let size = checked_add!(
        checked_mul!(row_bytes, checked_sub!(height, 1)?)?,
        // The last row does not need rgb.row_bytes but only min_row_bytes.
        min_row_bytes_usize
    )?;
    Ok((
        JxlPixelFormat {
            num_channels: rgb.channel_count(),
            data_type,
            endianness: JxlEndianness_JXL_NATIVE_ENDIAN,
            align: if rgb.row_bytes == min_row_bytes { 0 } else { row_bytes },
        },
        size,
    ))
}

impl Encoder for Libjxl {
    fn encode_image(
        &mut self,
        image: &Image,
        category: Category,
        config: &EncoderConfig,
        _output_samples: &mut Vec<Sample>,
    ) -> AvifResult<()> {
        // TODO: b/456440247 - Remove RGB->YUV->RGB unnecessary conversion or pass YUV samples to libjxl.
        //                     Note that the libjxl API requires a single buffer anyway.
        let mut rgb = reformat::rgb::Image::create_from_yuv(image);
        rgb.format = match (image.yuv_format, image.alpha_present) {
            (PixelFormat::Yuv420 | PixelFormat::Yuv422 | PixelFormat::Yuv444, false) => Format::Rgb,
            (PixelFormat::Yuv420 | PixelFormat::Yuv422 | PixelFormat::Yuv444, true) => Format::Rgba,
            _ => return AvifError::not_implemented(),
        };
        rgb.premultiply_alpha = image.alpha_premultiplied;
        rgb.allocate()?;
        rgb.convert_from_yuv(image)?;
        let rgb = rgb;
        let is_lossless = config.quality == 100.0;
        if is_lossless {
            // Make sure the YUV->RGB->YUV conversion is lossless too.
            let mut yuv = image.shallow_clone();
            yuv.allocate_planes(Category::Color)?;
            rgb.convert_to_yuv(&mut yuv)?;
            if !are_images_equal(image, &yuv)? {
                return AvifError::unknown_error(
                    "Could not convert YUV to RGB losslessly for JPEG XL",
                );
            }
        }

        let is_gray = rgb.format.channel_count() < 3; // Meaning is_monochrome.
        let jxl_header = reconstruct_jxl_header(ContainerFeatures {
            width: rgb.width,
            height: rgb.height,
            num_channels: rgb.format.channel_count() - rgb.format.has_alpha() as u32,
            num_extra: rgb.format.has_alpha().into(),
            bit_depth: rgb.depth,
            float_sample: false, // container_float_sample
            premultiplied_alpha: rgb.premultiply_alpha,
            codec_config: &self.get_codec_configuration(image, config.is_single_image, is_lossless),
            colr_nclx_colour_primaries: ColorPrimaries::Srgb as u32, // TODO: b/456440247 - Use color_primaries
            colr_nclx_transfer_characteristics: TransferCharacteristics::Srgb as u32, // TODO: b/456440247 - Use transfer_characteristics
            intensity_target: image.clli.map_or(0, |clli| clli.max_cll.into()),
        })?;

        match category {
            Category::Color => {}
            Category::Alpha => return AvifError::invalid_argument(), // Should be a channel, not an auxiliary item.
            Category::Gainmap => return AvifError::not_implemented(),
        }
        let num_channels = rgb.channel_count();
        let num_alpha_channels = if rgb.has_alpha() { 1 } else { 0 };

        let encoder = if let Some(encoder) = self.encoder {
            if config.is_single_image {
                return AvifError::unknown_error("Cannot add another frame to a single image");
            } else if Some(jxl_header) != self.expected_header {
                return AvifError::unknown_error(
                    "Cannot add another frame with different features to a JPEG XL animation",
                );
            }
            encoder
        } else {
            // SAFETY: `JxlEncoderCreate` accepts a null `JxlMemoryManager` pointer to use
            // libjxl's default allocator and returns either a new `JxlEncoder` or null on failure.
            let Some(encoder) = NonNull::new(unsafe { JxlEncoderCreate(null()) }) else {
                return AvifError::unknown_error("JxlEncoderCreate() failed.");
            };
            self.encoder = Some(encoder);

            let mut basic_info = MaybeUninit::<JxlBasicInfo>::zeroed();
            // SAFETY: `basic_info.as_mut_ptr()` points to valid, aligned, zero-initialized
            // `JxlBasicInfo` storage. `JxlEncoderInitBasicInfo` populates libjxl's default field
            // values, and `zeroed()` ensures trailing padding bytes (`padding: [u8; 100]`) are
            // initialized before `assume_init()`.
            let mut basic_info = unsafe {
                JxlEncoderInitBasicInfo(basic_info.as_mut_ptr());
                basic_info.assume_init()
            };
            basic_info.xsize = rgb.width;
            basic_info.ysize = rgb.height;
            basic_info.bits_per_sample = rgb.depth.into();
            basic_info.uses_original_profile = is_lossless.into();
            basic_info.num_color_channels = num_channels - num_alpha_channels;
            if rgb.has_alpha() {
                basic_info.num_extra_channels = 1;
                basic_info.alpha_bits = basic_info.bits_per_sample;
                basic_info.alpha_premultiplied = rgb.premultiply_alpha.into();
            }
            if !config.is_single_image {
                basic_info.have_animation = true.into();
                // tps_numerator and tps_denominator do not matter.
            }
            // SAFETY: `encoder` is a valid `JxlEncoder` owned by `self.encoder`, and
            // `&basic_info` is a valid initialized `JxlBasicInfo` reference copied by libjxl.
            unsafe { JxlEncoderSetBasicInfo(encoder.as_ptr(), &basic_info) }
                .map_enc_err(encoder)?;
            if rgb.has_alpha() {
                // Explicitly set extra_channel_info because JxlEncoderSetBasicInfo() in
                // libjxl <= 0.11.2 did not copy basic_info.alpha_premultiplied.
                let mut extra_channel_info = MaybeUninit::<JxlExtraChannelInfo>::zeroed();
                // SAFETY: `extra_channel_info.as_mut_ptr()` points to valid, aligned,
                // zero-initialized `JxlExtraChannelInfo` storage, and
                // `JxlEncoderInitExtraChannelInfo` initializes its fields before `assume_init()`.
                let mut extra_channel_info = unsafe {
                    JxlEncoderInitExtraChannelInfo(
                        JxlExtraChannelType_JXL_CHANNEL_ALPHA,
                        extra_channel_info.as_mut_ptr(),
                    );
                    extra_channel_info.assume_init()
                };
                extra_channel_info.bits_per_sample = basic_info.alpha_bits;
                extra_channel_info.alpha_premultiplied = basic_info.alpha_premultiplied;
                // SAFETY: `encoder` is a valid `JxlEncoder` with `num_extra_channels == 1` set
                // above, so extra channel index `0` is in bounds and `&extra_channel_info` is
                // valid for reads during the call.
                unsafe { JxlEncoderSetExtraChannelInfo(encoder.as_ptr(), 0, &extra_channel_info) }
                    .map_enc_err(encoder)?;
            }

            let mut color_encoding = MaybeUninit::<JxlColorEncoding>::zeroed();
            // SAFETY: `color_encoding.as_mut_ptr()` points to valid, aligned, zero-initialized
            // `JxlColorEncoding` storage, and `JxlColorEncodingSetToSRGB` initializes its fields
            // before `assume_init()`.
            let mut color_encoding = unsafe {
                JxlColorEncodingSetToSRGB(color_encoding.as_mut_ptr(), is_gray.into());
                color_encoding.assume_init()
            };
            // TODO: b/456440247 - This should be RELATIVE but PERCEPTUAL forces
            //                     color_encoding.all_default to be false.
            color_encoding.rendering_intent = JxlRenderingIntent_JXL_RENDERING_INTENT_PERCEPTUAL;
            // SAFETY: `encoder` is a valid `JxlEncoder` after `JxlEncoderSetBasicInfo`, and
            // `&color_encoding` is a valid initialized `JxlColorEncoding` reference.
            unsafe { JxlEncoderSetColorEncoding(encoder.as_ptr(), &color_encoding) }
                .map_enc_err(encoder)?;

            if let Some(speed) = config.speed {
                if speed >= 11 {
                    // SAFETY: `encoder` is a valid `JxlEncoder` owned by `self.encoder`.
                    unsafe { JxlEncoderAllowExpertOptions(encoder.as_ptr()) };
                }
            }

            self.expected_header = Some(jxl_header);
            encoder
        };

        // SAFETY: `encoder` is a valid `JxlEncoder` owned by `self.encoder`, and passing null for
        // `source_frame_settings` creates default-initialized frame settings owned by `encoder`.
        let Some(frame_settings) =
            NonNull::new(unsafe { JxlEncoderFrameSettingsCreate(encoder.as_ptr(), null()) })
        else {
            return AvifError::unknown_error("JxlEncoderFrameSettingsCreate() failed.");
        };

        if is_lossless {
            // SAFETY: `frame_settings` is a valid non-null `JxlEncoderFrameSettings` owned by
            // `encoder`.
            unsafe { JxlEncoderSetFrameLossless(frame_settings.as_ptr(), true.into()) }
                .map_enc_err(encoder)?;
        } else {
            // SAFETY: `JxlEncoderDistanceFromQuality` is a pure numeric conversion helper with no
            // pointer or lifetime preconditions.
            let distance = unsafe { JxlEncoderDistanceFromQuality(config.quality) };
            // SAFETY: `frame_settings` is a valid non-null `JxlEncoderFrameSettings` owned by
            // `encoder`.
            unsafe { JxlEncoderSetFrameDistance(frame_settings.as_ptr(), distance) }
                .map_enc_err(encoder)?;
        }
        if let Some(speed) = config.speed {
            let e = JxlEncoderFrameSettingId_JXL_ENC_FRAME_SETTING_EFFORT;
            // SAFETY: `frame_settings` is a valid non-null `JxlEncoderFrameSettings` owned by
            // `encoder`.
            unsafe { JxlEncoderFrameSettingsSetOption(frame_settings.as_ptr(), e, speed.into()) }
                .map_enc_err(encoder)?;
        }

        // TODO: b/456440247 - Check if a non-zero duration is necessary for sequences.
        if !config.is_single_image {
            let mut frame_header = MaybeUninit::<JxlFrameHeader>::zeroed();
            // SAFETY: `frame_header.as_mut_ptr()` points to valid, aligned, zero-initialized
            // `JxlFrameHeader` storage, and `JxlEncoderInitFrameHeader` initializes its fields
            // before `assume_init()`.
            let mut frame_header = unsafe {
                JxlEncoderInitFrameHeader(frame_header.as_mut_ptr());
                frame_header.assume_init()
            };
            frame_header.duration = 1;
            // SAFETY: `frame_settings` is a valid non-null `JxlEncoderFrameSettings` owned by
            // `encoder`, and `&frame_header` is a valid initialized `JxlFrameHeader` reference.
            unsafe { JxlEncoderSetFrameHeader(frame_settings.as_ptr(), &frame_header) }
                .map_enc_err(encoder)?;
        }

        let (pixel_format, size) = rgb_image_to_jxl_pixel_format(&rgb)?;
        if !rgb.has_valid_pixels() {
            return AvifError::invalid_argument();
        }
        let pixels = rgb.pixels();
        // SAFETY: `frame_settings` is a valid non-null pointer owned by `encoder`. `rgb.allocate()`
        // and `rgb.has_valid_pixels()` guarantee `pixels` is non-null, aligned to `u8` (8-bit) or
        // `u16` (>8-bit) matching `pixel_format.data_type`, and valid for reads of `size` bytes
        // during `JxlEncoderAddImageFrame` (which copies the frame data internally).
        unsafe {
            JxlEncoderAddImageFrame(frame_settings.as_ptr(), &pixel_format, pixels.cast(), size)
        }
        .map_enc_err(encoder)?;

        Ok(())
    }

    fn finish(&mut self, output_samples: &mut Vec<Sample>) -> AvifResult<()> {
        let Some(encoder) = self.encoder else {
            return Ok(());
        };
        let encoded_bytes_with_header = Self::encode(encoder)?;

        let expected_header = self
            .expected_header
            .as_ref()
            .ok_or(AvifError::InvalidArgument)?;
        if encoded_bytes_with_header.len() <= expected_header.len()
            || encoded_bytes_with_header.get(..expected_header.len())
                != Some(expected_header.as_slice())
        {
            return AvifError::unknown_error(format!(
                "Unexpected JPEG XL header {:?} vs {:?}",
                encoded_bytes_with_header.get(..expected_header.len()),
                expected_header
            ));
        }
        let encoded_bytes_without_header = expected_header.len()..encoded_bytes_with_header.len();

        output_samples.try_push(Sample::create(
            encoded_bytes_with_header,
            encoded_bytes_without_header,
            true,
        ))?;

        Ok(())
    }

    fn get_codec_config(
        &self,
        image: &Image,
        is_single_image: bool,
        is_lossless: bool,
        _output_samples: &[crate::encoder::Sample],
    ) -> AvifResult<CodecConfiguration> {
        Ok(CodecConfiguration::JpegXl(self.get_codec_configuration(
            image,
            is_single_image,
            is_lossless,
        )))
    }
}

impl Libjxl {
    fn encode(encoder: NonNull<JxlEncoder>) -> AvifResult<Vec<u8>> {
        // SAFETY: `encoder` is a valid non-null `JxlEncoder` created by `JxlEncoderCreate` and
        // kept alive by the calling `Libjxl` instance.
        unsafe { JxlEncoderCloseInput(encoder.as_ptr()) };

        let mut data: Vec<u8> = Vec::new(); // Vector of encoded bytes, growing by chunks.
        let mut chunk = [0u8; 64]; // Arbitrary chunk size (>= 32 bytes required by libjxl).
        loop {
            let mut avail_out = chunk.len();
            let mut next_out: *mut u8 = chunk.as_mut_ptr();
            // SAFETY: `encoder` is a valid non-null `JxlEncoder` with input closed above,
            // `next_out` points to the writable `chunk` buffer of `avail_out` (`64`) bytes, and
            // `&mut next_out` / `&mut avail_out` are valid writable pointers updated in place by
            // `JxlEncoderProcessOutput`.
            let status =
                unsafe { JxlEncoderProcessOutput(encoder.as_ptr(), &mut next_out, &mut avail_out) };
            // avail_out now contains the number of unused bytes among the chunk.len() bytes
            // that were passed to JxlEncoderProcessOutput().
            let written_len = checked_sub!(chunk.len(), avail_out)?;
            if written_len == 0 && status == JxlEncoderStatus_JXL_ENC_NEED_MORE_OUTPUT {
                return AvifError::unknown_error("JxlEncoderProcessOutput() made no progress");
            }
            let Some(written_bytes) = chunk.get(..written_len) else {
                return AvifError::invalid_argument();
            };

            data.try_extend_from_slice(written_bytes)?;
            if status != JxlEncoderStatus_JXL_ENC_NEED_MORE_OUTPUT {
                status.map_enc_err(encoder)?; // JxlEncoderStatus_JXL_ENC_SUCCESS is expected.
                return Ok(data);
            }
        }
    }

    fn get_codec_configuration(
        &self,
        image: &Image,
        is_single_image: bool,
        is_lossless: bool,
    ) -> JpegXlCodecConfiguration {
        // Deduced from libjxl behavior.
        JpegXlCodecConfiguration {
            have_animation: !is_single_image,
            modular_16bit_buffers: image.depth <= 12, // See libjxl's SetUIntSamples().
            xyb_encoded: !is_lossless,
            level: 5,
        }
    }
}

impl Decoder for Libjxl {
    fn codec(&self) -> CodecChoice {
        CodecChoice::Libjxl
    }

    fn initialize(&mut self, _: &DecoderConfig) -> AvifResult<()> {
        Ok(())
    }

    fn get_next_image(
        &mut self,
        payload: &[u8],
        spatial_id: u8,
        image: &mut Image,
        category: Category,
        item: Option<&Item>,
        _signal_eos: bool,
    ) -> AvifResult<()> {
        // TODO: b/456440247 - Support tracks
        let item = if let Some(item) = item { item } else { return AvifError::not_implemented() };

        if spatial_id != 0xff {
            return AvifError::unknown_error(format!(
                "spatial_id {spatial_id} is not supported with JPEG XL"
            ));
        }
        match category {
            Category::Color => {}
            Category::Alpha => return AvifError::invalid_argument(), // Should be a channel, not an auxiliary item.
            Category::Gainmap => return AvifError::not_implemented(),
        }

        let reconstructed = match &self.reconstructed_jxl {
            Some(reconstructed) => reconstructed,
            None => self.reconstructed_jxl.insert({
                // Prepend the JPEG XL payload which is stripped from its header with a reconstructed header.

                let Some(pixi) = item.pixi() else {
                    return AvifError::bmff_parse_failed("pixi is mandatory with hxlI");
                };
                let premultiplied_alpha = item.alpi().is_some_and(|alpi| alpi.is_premultiplied);
                let Some(codec_config) = item.properties.iter().find_map(|property| {
                    if let ItemProperty::CodecConfiguration(CodecConfiguration::JpegXl(
                        codec_config,
                    )) = property
                    {
                        Some(codec_config)
                    } else {
                        None
                    }
                }) else {
                    return AvifError::bmff_parse_failed("hxlC is mandatory with hxlI");
                };
                let Some(_colr_nclx) = item.colr_nclx() else {
                    return AvifError::bmff_parse_failed("colr nclx is mandatory with hxlI");
                };
                let mut reconstructed = reconstruct_jxl_header(ContainerFeatures {
                    width: item.width,
                    height: item.height,
                    num_channels: pixi.num_color_channels()?,
                    num_extra: u32_from_usize(pixi.num_channels_with_idc(ChannelIdc::Alpha))?,
                    bit_depth: pixi.bit_depth()?,
                    float_sample: false, // TODO: b/456440247 - Support
                    premultiplied_alpha,
                    codec_config,
                    colr_nclx_colour_primaries: ColorPrimaries::Srgb as u32, // TODO: b/456440247 - Use colr_nclx.color_primaries
                    colr_nclx_transfer_characteristics: TransferCharacteristics::Srgb as u32, // TODO: b/456440247 - Use colr_nclx.transfer_characteristics
                    intensity_target: item.clli().map_or(0, |clli| clli.max_cll.into()),
                })?;
                reconstructed.try_extend_from_slice(payload)?;
                self.consumed_bytes = 0;
                reconstructed
            }),
        };

        self.decoder = Some(Self::decode_frame(
            self.decoder.take(),
            reconstructed,
            &mut self.consumed_bytes,
            image,
        )?);
        Ok(())
    }

    fn get_next_image_grid(
        &mut self,
        _payloads: &[Vec<u8>],
        _spatial_id: u8,
        _grid_image_helper: &mut GridImageHelper,
    ) -> AvifResult<()> {
        AvifError::not_implemented() // TODO: b/456440247
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

impl Libjxl {
    // Decodes the frame of |data| starting at |consumed_bytes| into |image|. |decoder| is None for
    // the first frame of the bitstream. Returns the decoder, ready to decode the next frame, if
    // any, and advances |consumed_bytes| accordingly.
    fn decode_frame(
        decoder: Option<jxl_api::JxlDecoder>,
        data: &[u8],
        consumed_bytes: &mut usize,
        image: &mut Image,
    ) -> AvifResult<jxl_api::JxlDecoder> {
        // jxl-rs reads from |input| and leaves the bytes it did not consume in it.
        let mut input = data.get(*consumed_bytes..).ok_or(AvifError::NoContent)?;

        let mut decoder = match decoder {
            Some(decoder) => decoder,
            None => {
                let mut decoder = jxl_api::JxlDecoder::new(jxl_api::JxlDecoderOptions::default());
                expect_event(decoder.process(&mut input, None, None), |e| {
                    *e == jxl_api::Event::BasicInfo
                })?;
                decoder
            }
        };

        let basic_info = decoder
            .basic_info()
            .ok_or_else(|| AvifError::UnknownError("Missing JPEG XL basic info".into()))?;
        let depth: u8 = match basic_info.bit_depth {
            jxl_api::JxlBitDepth::Int { bits_per_sample } => bits_per_sample
                .try_into()
                .map_err(|_| AvifError::InvalidArgument)?,
            // TODO: b/456440247 - Support floating point samples.
            jxl_api::JxlBitDepth::Float { .. } => return AvifError::not_implemented(),
        };
        let alpha_channel = match basic_info.extra_channels.as_slice() {
            [] => None,
            [extra_channel] if extra_channel.ec_type == ExtraChannel::Alpha => Some(extra_channel),
            extra_channels => {
                return AvifError::unknown_error(format!(
                    "Unexpected {} extra JPEG XL channels",
                    extra_channels.len()
                ))
            }
        };

        if image.width != 0 {
            return AvifError::invalid_argument();
        }
        image.width = u32_from_usize(basic_info.size.0)?;
        image.height = u32_from_usize(basic_info.size.1)?;
        image.depth = depth;

        // TODO: b/456440247 - Use information from pixi with px_flags&1=1 to fill these values?
        image.yuv_format = PixelFormat::Yuv444; // Expect RGB for now.
        image.yuv_range = YuvRange::Full;
        image.chroma_sample_position = ChromaSamplePosition::Unknown;
        // TODO: b/456440247 - Use information from colr nclx to fill these values?
        image.color_primaries = ColorPrimaries::Unspecified;
        image.transfer_characteristics = TransferCharacteristics::Unspecified;
        image.matrix_coefficients = MatrixCoefficients::Unspecified;

        image.allocate_planes(Category::Color)?;
        if let Some(alpha_channel) = alpha_channel {
            image.alpha_present = true;
            image.alpha_premultiplied = alpha_channel.alpha_associated;
            image.allocate_planes(Category::Alpha)?;
        }

        // TODO: b/456440247 - Remove RGB->YUV->RGB unnecessary conversion.
        let mut rgb = reformat::rgb::Image::create_from_yuv(image);
        rgb.format = match (image.yuv_format, image.alpha_present) {
            (PixelFormat::Yuv420 | PixelFormat::Yuv422 | PixelFormat::Yuv444, false) => Format::Rgb,
            (PixelFormat::Yuv420 | PixelFormat::Yuv422 | PixelFormat::Yuv444, true) => Format::Rgba,
            _ => return AvifError::not_implemented(),
        };
        rgb.premultiply_alpha = image.alpha_premultiplied;
        rgb.allocate()?;

        // The pixel format can only be set before the first frame header is decoded. Setting the
        // same value again for the following frames is allowed and is a no-op.
        let pixel_format = rgb_image_to_jxl_rs_pixel_format(&rgb, basic_info.extra_channels.len())?;
        decoder
            .set_pixel_format(pixel_format)
            .map_err(map_dec_err)?;

        expect_event(decoder.process(&mut input, None, None), |e| {
            *e == jxl_api::Event::FrameHeader
        })?;

        let num_rows = usize_from_u32(rgb.height)?;
        let bytes_per_row = usize_from_u32(checked_mul!(rgb.width, rgb.pixel_size())?)?;
        let byte_stride = usize_from_u32(rgb.row_bytes)?;
        if byte_stride < bytes_per_row {
            return AvifError::unknown_error(format!(
                "Invalid row_bytes {byte_stride} < min_row_bytes {bytes_per_row}"
            ));
        }

        let pixel_data = rgb.pixel_mut_slice()?;
        if pixel_data.len() < checked_mul!(num_rows, bytes_per_row)? {
            return AvifError::unknown_error(format!(
                "Invalid pixel_data.len() {} < num_rows {} * bytes_per_row {}",
                pixel_data.len(),
                num_rows,
                bytes_per_row,
            ));
        }

        {
            let buffer = jxl_api::JxlOutputBuffer::new_with_stride(
                pixel_data,
                num_rows,
                bytes_per_row,
                byte_stride,
            );
            let mut buffers = [buffer];
            expect_event(decoder.process(&mut input, Some(&mut buffers), None), |e| {
                matches!(e, jxl_api::Event::FrameComplete { .. })
            })?;
        }

        *consumed_bytes = checked_sub!(data.len(), input.len())?;

        rgb.convert_to_yuv(image)?;
        Ok(decoder)
    }
}

impl Drop for Libjxl {
    fn drop(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            // SAFETY: `encoder` is a valid non-null `JxlEncoder` uniquely owned by `self.encoder`
            // (created by `JxlEncoderCreate`), and `take()` ensures it is destroyed only once.
            unsafe { JxlEncoderDestroy(encoder.as_ptr()) };
        }
    }
}

impl Libjxl {
    pub(crate) fn version() -> String {
        // SAFETY: `JxlEncoderVersion` takes no arguments and returns a constant `u32` version.
        let version = unsafe { JxlEncoderVersion() };
        format!(
            "jxl: v{}.{}.{}",
            version / 1000000,
            (version % 1000000) / 1000,
            version % 1000
        )
    }
}

// FindAspectRatio() from libjxl's headers.cc
fn find_aspect_ratio(xsize: u32, ysize: u32) -> u32 {
    for (r, num, den) in [
        (1, 1, 1),   // square
        (2, 12, 10), //
        (3, 4, 3),   // camera
        (4, 3, 2),   // mobile camera
        (5, 16, 9),  // camera/display
        (6, 5, 4),   //
        (7, 2, 1),   //
    ] {
        let fixed_aspect_ratio = ysize as u64 * num as u64 / den;
        if xsize == fixed_aspect_ratio as u32 {
            return r;
        }
    }
    0 // Must send xsize instead
}

impl OStreamLittleEndian {
    // See Section B.2.2 of ISO/IEC 18181-1.
    fn enum_table(&mut self, value: u32) -> AvifResult<()> {
        match value {
            0 => self.write_bits(0b00, 2),
            1 => self.write_bits(0b01, 2),
            2..34 => {
                self.write_bits(0b10, 2)?;
                self.write_bits(value - 2, 4) // 2 + u(4)
            }
            34..146 => {
                self.write_bits(0b11, 2)?;
                self.write_bits(value - 18, 4) // 18 + u(6)
            }
            _ => AvifError::invalid_argument(),
        }
    }

    // Implementation of U32(1 + u(9), 1 + u(13), 1 + u(18), 1 + u(30)) as in ISO/IEC 18181-1.
    fn wsz(&mut self, dim: u32) -> AvifResult<()> {
        match dim {
            0..=511 /* 1<<9 */ => {
                self.write_bits(0b00, 2)?;
                self.write_bits(dim, 9)
            }
            512..=8191 /* 1<<13 */ => {
                self.write_bits(0b01, 2)?;
                self.write_bits(dim, 13)
            }
            8192..=262144 /* 1<<18 */ => {
                self.write_bits(0b10, 2)?;
                self.write_bits(dim, 18)
            }
            _ => {
                self.write_bits(0b11, 2)?;
                self.write_bits(dim, 30)
            }
        }
    }

    // See libjxl's BitDepth::VisitFields().
    fn bit_depth_bundle(&mut self, float_sample: bool, bits_per_sample: u8) -> AvifResult<()> {
        self.write_bits(float_sample.into(), 1)?; // bit_depth.floating_point_sample
        if float_sample {
            // bit_depth.bits_per_sample
            match bits_per_sample {
                16 => self.write_bits(0b01, 2)?,
                32 => self.write_bits(0b00, 2)?,
                64 => {
                    self.write_bits(0b11, 2)?;
                    self.write_bits(64 - 1, 6)?; // 1 + u(6)
                }
                _ => return AvifError::unsupported_depth(),
            }
            // bit_depth.exp_bits
            match bits_per_sample {
                16 => self.write_bits(5 - 1, 4)?,  // 1 + u(4)
                32 => self.write_bits(8 - 1, 2)?,  // 1 + u(4)
                64 => self.write_bits(11 - 1, 4)?, // 1 + u(4)
                _ => return AvifError::unsupported_depth(),
            }
        } else {
            // bit_depth.bits_per_sample
            match bits_per_sample {
                8 => self.write_bits(0b00, 2)?,
                10 => self.write_bits(0b01, 2)?,
                12 => self.write_bits(0b10, 2)?,
                16 => {
                    self.write_bits(0b11, 2)?;
                    self.write_bits(16 - 1, 6)?; // 1 + u(6)
                }
                _ => return AvifError::unsupported_depth(),
            }
        }
        Ok(())
    }
}

// Image metadata that can be retrieved at the HEIF level.
struct ContainerFeatures<'a> {
    width: u32,
    height: u32,
    num_channels: u32,
    num_extra: u32,
    bit_depth: u8,
    float_sample: bool,
    premultiplied_alpha: bool,
    codec_config: &'a JpegXlCodecConfiguration, // hxlC
    colr_nclx_colour_primaries: u32,
    colr_nclx_transfer_characteristics: u32,
    intensity_target: u32,
}

fn reconstruct_jxl_header(container: ContainerFeatures) -> AvifResult<Vec<u8>> {
    let mut writer = OStreamLittleEndian::default();

    // JPEG XL compliant signature (ISO/IEC 18181-1, Table D.1) set to 0xFF0A (2815).

    writer.write_slice(&[0xff, 0x0a])?; // signature

    // JPEG XL compliant SizeHeader bundle with its fields set according to C.4.1.

    // See libjxl's SizeHeader::Set().
    let xsize = container.width;
    let ysize = container.height;
    if xsize == 0 || ysize == 0 {
        return AvifError::invalid_argument();
    }
    let ratio = find_aspect_ratio(xsize, ysize);
    const BLOCK_DIM: u32 = 8; // kBlockDim
    let small = container.height <= 256
        && (container.height % BLOCK_DIM) == 0
        && (ratio != 0 || (container.width <= 256 && (container.width % BLOCK_DIM) == 0));

    // See libjxl's SizeHeader::VisitFields().
    writer.write_bits(if small { 1 } else { 0 }, 1)?;
    if small {
        writer.write_bits(ysize / 8 - 1, 5)?; // ysize_div8_minus_1_
    } else {
        writer.wsz(ysize - 1)?;
    }
    writer.write_bits(ratio, 3)?;
    if ratio == 0 && small {
        writer.write_bits(xsize / 8 - 1, 5)?; // xsize_div8_minus_1_
    } else if ratio == 0 {
        writer.wsz(xsize - 1)?;
    }

    // JPEG XL compliant ImageMetadata bundle.

    // See libjxl's ImageMetadata::VisitFields().
    writer.write_bits(0, 1)?; // all_default
    let extra_fields = container.codec_config.have_animation || container.intensity_target != 0;
    writer.write_bits(extra_fields.into(), 1)?; // extra_fields
    if extra_fields {
        writer.write_bits(0, 3)?; // orientation = 1 (minus one)
        writer.write_bits(0, 1)?; // have_intr_size
        writer.write_bits(0, 1)?; // have_preview
        writer.write_bits(container.codec_config.have_animation.into(), 1)?; // have_animation
        if container.codec_config.have_animation {
            // AnimationHeader bundle
            writer.write_bits(0b00, 2)?; // tps_numerator = 100
            writer.write_bits(0b00, 2)?; // tps_denominator = 1
            writer.write_bits(0b00, 2)?; // num_loops = 0
            writer.write_bits(0, 1)?; // have_timecodes
        }
    }
    writer.bit_depth_bundle(container.float_sample, container.bit_depth)?; // BitDepth bundle
    writer.write_bits(container.codec_config.modular_16bit_buffers.into(), 1)?; // modular_16bit_buffers
    match (container.num_extra, container.premultiplied_alpha) {
        (0, _) => writer.write_bits(0b00, 2)?, // num_extra
        (1, false) => {
            writer.write_bits(0b01, 2)?; // num_extra
            writer.write_bits(1, 1)?; // ec_info[0].d_alpha
        }
        (1, true) => {
            writer.write_bits(0b01, 2)?; // num_extra
            writer.write_bits(0, 1)?; // ec_info[0].d_alpha
            writer.enum_table(0)?; // ec_info[0].type = kAlpha

            // ec_info[0].bit_depth
            writer.bit_depth_bundle(container.float_sample, container.bit_depth)?;

            writer.write_bits(0b00, 2)?; // ec_info[0].dim_shift = 0
            writer.write_bits(0b00, 2)?; // ec_info[0].name_len = 0
            writer.write_bits(1, 1)?; // ec_info[0].alpha_associated
        }
        _ => return AvifError::invalid_argument(),
    }
    writer.write_bits(container.codec_config.xyb_encoded.into(), 1)?; // xyb_encoded

    // ColourEncoding bundle
    writer.write_bits(0, 1)?; // color_encoding.all_default
    writer.write_bits(0, 1)?; // color_encoding.want_icc
    writer.enum_table(match container.num_channels {
        1 => 1, // color_encoding.colour_space = kGrey
        3 => 0, // color_encoding.colour_space = kRGB
        _ => return AvifError::invalid_argument(),
    })?;
    writer.enum_table(1)?; // color_encoding.white_point = D65
    writer.enum_table(match container.colr_nclx_colour_primaries {
        1 | 9 | 11 => container.colr_nclx_colour_primaries, // color_encoding.primaries = kSRGB or k2100 or kP3
        _ => 1, // Unspecified or value not listed in table E.5 is mapped to color_encoding.primaries = kSRGB
    })?;
    {
        // CustomTransferFunction bundle
        writer.write_bits(0, 1)?; // color_encoding.tf.have_gamma
        writer.enum_table(match container.colr_nclx_transfer_characteristics {
            2 => 0, // color_encoding.tf = kUnknown
            13 | 16 | 17 | 18 => container.colr_nclx_transfer_characteristics, // color_encoding.tf = kSRGB or kPQ or kDCI or kHLG
            _ => 8, // color_encoding.tf = kLinear
        })?;
    }
    // TODO: b/456440247 - This should be kRelative but kPerceptual forces
    //                     color_encoding.all_default to be false.
    writer.enum_table(0)?; // color_encoding.rendering_intent = kPerceptual
    if extra_fields {
        // ToneMapping bundle
        match container.intensity_target {
            0 => writer.write_bits(1, 1)?,            // tone_mapping.all_default
            _ => return AvifError::not_implemented(), // TODO: b/456440247
        }
    }
    writer.write_bits(0b00, 2)?; // extensions.extensions = 0
    writer.write_bits(1, 1)?; // default_m = 1 // TODO: b/456440247

    // No ICC, no preview. Frame should start at byte boundary.
    writer.pad()?;

    Ok(writer.data)
}

#[test]
fn libjxl_enc_dec_test() -> Result<(), AvifError> {
    let mut encoder = Libjxl::default();
    let config = EncoderConfig {
        is_single_image: true,
        quality: 100.0,
        ..Default::default()
    };

    let image = {
        let mut image = Image {
            width: 1,
            height: 1,
            depth: 8,
            yuv_format: PixelFormat::Yuv444,
            ..Default::default()
        };
        image.allocate_planes(Category::Color)?;
        let mut rgb = reformat::rgb::Image::create_from_yuv(&image);
        rgb.format = Format::Rgb;
        rgb.allocate()?;
        rgb.row_mut(0)?[0] = 42;
        rgb.row_mut(0)?[1] = 80;
        rgb.row_mut(0)?[2] = 0;
        rgb.convert_to_yuv(&mut image)?;
        image
    };
    let item = Item {
        width: image.width,
        height: image.height,
        properties: try_vec_exact![
            ItemProperty::PixelInformation(PixelInformation {
                planes: try_vec_exact![
                    PlanePixelInformation {
                        depth: image.depth,
                        channel_idc: Some(ChannelIdc::FirstColorChannel),
                        subsampling_type: Some(image.yuv_format),
                        subsampling_location: None,
                    },
                    PlanePixelInformation {
                        depth: image.depth,
                        channel_idc: Some(ChannelIdc::SecondColorChannel),
                        subsampling_type: Some(image.yuv_format),
                        subsampling_location: None,
                    },
                    PlanePixelInformation {
                        depth: image.depth,
                        channel_idc: Some(ChannelIdc::ThirdColorChannel),
                        subsampling_type: Some(image.yuv_format),
                        subsampling_location: None,
                    },
                ]?,
            }),
            ItemProperty::ColorInformation(ColorInformation::Nclx(Nclx {
                color_primaries: image.color_primaries,
                transfer_characteristics: image.transfer_characteristics,
                matrix_coefficients: image.matrix_coefficients,
                yuv_range: image.yuv_range,
            })),
            ItemProperty::CodecConfiguration(CodecConfiguration::JpegXl(
                encoder.get_codec_configuration(
                    &image,
                    config.is_single_image,
                    config.quality == 100.0,
                ),
            )),
        ]?,
        is_made_up: true,
        ..Default::default()
    };

    let mut output_samples = vec![];
    encoder.encode_image(&image, Category::Color, &config, &mut output_samples)?;
    encoder.finish(&mut output_samples)?;
    assert_eq!(output_samples.len(), 1);

    let mut decoder = Libjxl::default();
    let mut decoded = Image::default();
    decoder.initialize(&DecoderConfig {
        operating_point: 0,
        all_layers: false,
        width: image.width,
        height: image.height,
        depth: image.depth,
        max_threads: 0,
        image_size_limit: None,
        max_input_size: 0,
        codec_config: item.codec_config().unwrap().try_clone()?,
        category: Category::Color,
        android_mediacodec_output_color_format: AndroidMediaCodecOutputColorFormat::default(),
    })?;
    decoder.get_next_image(
        &output_samples[0].sample_data(),
        0xff,
        &mut decoded,
        Category::Color,
        Some(&item),
        false,
    )?;

    assert!(are_images_equal(&image, &decoded)?);
    Ok(())
}
