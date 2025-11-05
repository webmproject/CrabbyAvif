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

use crate::codecs::*;
use crate::encoder::Sample;
use crate::image::Image;
use crate::reformat::rgb::Format;
use crate::*;

use libjxl_sys::bindings::*;

use std::mem::MaybeUninit;
use std::ptr::null;
use std::ptr::null_mut;

#[derive(Default)]
pub struct Libjxl {
    encoder: *mut JxlEncoder,
    decoder: *mut JxlDecoder,
}

// Convenient error mapping.
trait JxlEncoderStatusTrait {
    fn map_enc_err(self, encoder: *mut JxlEncoder) -> Result<(), AvifError>;
}
impl JxlEncoderStatusTrait for JxlEncoderStatus {
    fn map_enc_err(self, encoder: *mut JxlEncoder) -> Result<(), AvifError> {
        #![allow(non_upper_case_globals)]
        match self {
            JxlEncoderStatus_JXL_ENC_SUCCESS => Ok(()),
            JxlEncoderStatus_JXL_ENC_ERROR => {
                let error = unsafe { JxlEncoderGetError(encoder) };
                AvifError::unknown_error(format!("JxlEncoderError {error}",))
            }
            _ => AvifError::unknown_error(format!("Unexpected JxlEncoderStatus {self}")),
        }
    }
}
trait JxlDecoderStatusTrait {
    fn map_dec_err(self) -> Result<(), AvifError>;
}
impl JxlDecoderStatusTrait for JxlDecoderStatus {
    fn map_dec_err(self) -> Result<(), AvifError> {
        #![allow(non_upper_case_globals)]
        match self {
            JxlDecoderStatus_JXL_DEC_SUCCESS => Ok(()),
            _ => AvifError::unknown_error(format!("Unexpected JxlDecoderStatus {self}")),
        }
    }
}

fn rgb_image_to_jxl_pixel_format(
    rgb: &reformat::rgb::Image,
) -> Result<(JxlPixelFormat, usize), AvifError> {
    let data_type = match rgb.depth {
        8 => JxlDataType_JXL_TYPE_UINT8,
        9..16 => JxlDataType_JXL_TYPE_UINT16,
        _ => return AvifError::unknown_error(format!("Unexpected depth {}", rgb.depth)),
    };
    let min_row_bytes = checked_mul!(rgb.width, rgb.channel_count() * rgb.pixel_size())?;
    let size = checked_add!(
        checked_mul!(rgb.row_bytes as usize, rgb.height as usize - 1)?,
        min_row_bytes as usize
    )?;
    Ok((
        JxlPixelFormat {
            num_channels: rgb.channel_count(),
            data_type,
            endianness: JxlEndianness_JXL_NATIVE_ENDIAN,
            align: if rgb.row_bytes == min_row_bytes { 0 } else { rgb.row_bytes as usize },
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
        // Note that the libjxl API requires a single buffer anyway.
        let mut rgb = reformat::rgb::Image::create_from_yuv(image);
        rgb.allocate()?;
        rgb.convert_from_yuv(image)?;
        let rgb = rgb;
        let is_gray = false; // Meaning is_monochrome.

        match category {
            Category::Color => {}
            Category::Alpha => unreachable!(), // Should be a channel, not an auxiliary item.
            Category::Gainmap => return AvifError::not_implemented(),
        }
        if !matches!(rgb.format, Format::Rgb | Format::Rgba) {
            return AvifError::unknown_error(format!(
                "{:?} is not supported with JPEG XL",
                rgb.format
            ));
        }
        let lossless = config.quality == 100.0;
        let num_channels = rgb.pixel_size();
        let num_alpha_channels = if rgb.has_alpha() { 1 } else { 0 };

        if self.encoder.is_null() {
            // # Safety: Calling a C function.
            let encoder = unsafe { JxlEncoderCreate(null()) };
            if encoder.is_null() {
                return AvifError::unknown_error("JxlEncoderCreate() failed.");
            }
            self.encoder = encoder;

            let mut basic_info: MaybeUninit<JxlBasicInfo> = MaybeUninit::uninit();
            // # Safety: Calling a C function with valid parameters.
            unsafe { JxlEncoderInitBasicInfo(basic_info.as_mut_ptr()) };
            // # Safety: basic_info was initialized in the C function above.
            let mut basic_info = unsafe { basic_info.assume_init() };
            basic_info.xsize = rgb.width;
            basic_info.ysize = rgb.height;
            basic_info.bits_per_sample = rgb.depth.into();
            basic_info.uses_original_profile = lossless.into();
            basic_info.num_color_channels = num_channels - num_alpha_channels;
            if rgb.has_alpha() {
                basic_info.num_extra_channels = 1;
                basic_info.alpha_bits = basic_info.bits_per_sample;
                basic_info.alpha_premultiplied = rgb.premultiply_alpha.into();
                // JxlEncoderSetExtraChannelInfo() does not need to be called for alpha
                // apparently.
            }
            if !config.is_single_image {
                basic_info.have_animation = true.into();
                // tps_numerator and tps_denominator do not matter.
            }
            // # Safety: Calling a C function with valid parameters.
            unsafe { JxlEncoderSetBasicInfo(encoder, &basic_info) }.map_enc_err(encoder)?;

            let mut color_encoding: MaybeUninit<JxlColorEncoding> = MaybeUninit::uninit();
            // # Safety: Calling a C function with valid parameters.
            unsafe { JxlColorEncodingSetToSRGB(color_encoding.as_mut_ptr(), is_gray.into()) };
            // # Safety: color_encoding was initialized in the C function above.
            let mut color_encoding = unsafe { color_encoding.assume_init() };
            color_encoding.rendering_intent = JxlRenderingIntent_JXL_RENDERING_INTENT_PERCEPTUAL;
            // # Safety: Calling a C function with valid parameters.
            unsafe { JxlEncoderSetColorEncoding(encoder, &color_encoding) }.map_enc_err(encoder)?;

            if let Some(speed) = config.speed {
                if speed >= 11 {
                    // # Safety: Calling a C function with valid parameters.
                    unsafe { JxlEncoderAllowExpertOptions(encoder) };
                }
            }
        } else if config.is_single_image {
            return AvifError::unknown_error("Cannot add another frame to a single image");
        }
        let encoder = self.encoder;

        // # Safety: Calling a C function with valid parameters.
        let frame_settings = unsafe { JxlEncoderFrameSettingsCreate(encoder, null()) };
        if frame_settings.is_null() {
            return AvifError::unknown_error("JxlEncoderFrameSettingsCreate() failed.");
        }

        if lossless {
            // # Safety: Calling a C function with valid parameters.
            unsafe { JxlEncoderSetFrameLossless(frame_settings, true.into()) }
                .map_enc_err(encoder)?;
        } else {
            // # Safety: Calling a C function with valid parameters.
            let distance = unsafe { JxlEncoderDistanceFromQuality(config.quality) };
            // # Safety: Calling a C function with valid parameters.
            unsafe { JxlEncoderSetFrameDistance(frame_settings, distance) }.map_enc_err(encoder)?;
        }
        if let Some(speed) = config.speed {
            let e = JxlEncoderFrameSettingId_JXL_ENC_FRAME_SETTING_EFFORT;
            // # Safety: Calling a C function with valid parameters.
            unsafe { JxlEncoderFrameSettingsSetOption(frame_settings, e, speed.into()) }
                .map_enc_err(encoder)?;
        }

        // TODO: b/456440247 - Check if a non-zero duration is necessary for sequences.
        if !config.is_single_image {
            let mut frame_header: MaybeUninit<JxlFrameHeader> = MaybeUninit::uninit();
            // # Safety: Calling a C function with valid parameters.
            unsafe { JxlEncoderInitFrameHeader(frame_header.as_mut_ptr()) };
            // # Safety: frame_header was initialized in the C function above.
            let mut frame_header = unsafe { frame_header.assume_init() };
            frame_header.duration = 1;
            // # Safety: Calling a C function with valid parameters.
            unsafe { JxlEncoderSetFrameHeader(frame_settings, &frame_header) }
                .map_enc_err(encoder)?;
        }

        let (pixel_format, size) = rgb_image_to_jxl_pixel_format(&rgb)?;
        // # Safety: Calling a C function with valid parameters.
        unsafe {
            JxlEncoderAddImageFrame(frame_settings, &pixel_format, rgb.pixels().cast(), size)
        }
        .map_enc_err(encoder)?;

        Ok(())
    }

    fn finish(&mut self, output_samples: &mut Vec<Sample>) -> AvifResult<()> {
        if self.encoder.is_null() {
            return Ok(());
        }
        let encoder = self.encoder;

        // # Safety: Calling a C function with valid parameters.
        unsafe { JxlEncoderCloseInput(encoder) };

        let mut data: Vec<u8> = vec![];
        data.try_reserve(64).map_err(AvifError::map_out_of_memory)?; // Arbitrary initial size.
        data.resize(data.capacity(), 0);
        let mut avail_out = data.len();
        let mut next_out: *mut u8 = data.as_mut_ptr();
        loop {
            // # Safety: Calling a C function with valid parameters.
            let status = unsafe { JxlEncoderProcessOutput(encoder, &mut next_out, &mut avail_out) };
            // # Safety: Computing the offset between two pointers guaranteed to be from the same allocation.
            let num_written_bytes =
                unsafe { next_out.byte_offset_from_unsigned(data.as_mut_ptr()) };
            if status == JxlEncoderStatus_JXL_ENC_NEED_MORE_OUTPUT {
                data.try_reserve(data.capacity())
                    .map_err(AvifError::map_out_of_memory)?;
                data.resize(data.capacity(), 0);
                // # Safety: Offsetting pointer by a byte amount guaranteed to fit in the same allocation.
                next_out = unsafe { data.as_mut_ptr().byte_add(num_written_bytes) };
                avail_out = data.len() - num_written_bytes;
            } else {
                status.map_enc_err(encoder)?; // JxlEncoderStatus_JXL_ENC_SUCCESS is expected.
                assert!(num_written_bytes <= data.len());
                data.resize(num_written_bytes, 0); // Trim the unused allocated bytes.
                output_samples.push(Sample { data, sync: true });
                return Ok(());
            }
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
    ) -> AvifResult<()> {
        assert_eq!(spatial_id, 0xff); // Sentinel value.
        match category {
            Category::Color => {}
            Category::Alpha => unreachable!(), // Should be a channel, not an auxiliary item.
            Category::Gainmap => return AvifError::not_implemented(),
        }

        if self.decoder.is_null() {
            // # Safety: Calling a C function.
            let decoder = unsafe { JxlDecoderCreate(null()) };
            if decoder.is_null() {
                return AvifError::unknown_error("JxlDecoderCreate() failed.");
            }
            // # Safety: decoder cannot be null here thanks to the check above.
            self.decoder = decoder;

            const EVENTS: i32 =
                (JxlDecoderStatus_JXL_DEC_BASIC_INFO | JxlDecoderStatus_JXL_DEC_FULL_IMAGE) as i32;
            // # Safety: Calling a C function with valid parameters.
            unsafe { JxlDecoderSubscribeEvents(decoder, EVENTS) }.map_dec_err()?;

            // # Safety: Calling a C function with valid parameters.
            unsafe { JxlDecoderSetInput(decoder, payload.as_ptr(), payload.len()) }
                .map_dec_err()?;
            // # Safety: Calling a C function with valid parameters.
            unsafe { JxlDecoderCloseInput(decoder) };

            // # Safety: Calling a C function with valid parameters.
            let status = unsafe { JxlDecoderProcessInput(decoder) };
            if status != JxlDecoderStatus_JXL_DEC_BASIC_INFO {
                return AvifError::unknown_error(format!(
                    "Unexpected JxlDecoderStatus {status} instead of BASIC_INFO"
                ));
            }
        }
        let decoder = self.decoder;
        let mut basic_info: MaybeUninit<JxlBasicInfo> = MaybeUninit::uninit();
        // # Safety: Calling a C function with valid parameters.
        unsafe { JxlDecoderGetBasicInfo(decoder, basic_info.as_mut_ptr()) }.map_dec_err()?;
        // # Safety: basic_info was initialized in the C function above.
        let basic_info = unsafe { basic_info.assume_init() };

        // # Safety: Calling a C function with valid parameters.
        let status = unsafe { JxlDecoderProcessInput(decoder) };
        if status != JxlDecoderStatus_JXL_DEC_NEED_IMAGE_OUT_BUFFER {
            return AvifError::unknown_error(format!(
                "Unexpected JxlDecoderStatus {status} instead of NEED_IMAGE_OUT_BUFFER"
            ));
        }

        assert_eq!(image.width, 0);
        image.width = basic_info.xsize;
        image.height = basic_info.ysize;
        image.depth = basic_info.bits_per_sample.try_into().unwrap();

        // TODO: b/456440247 - Use information from pixi with px_flags&1=1 to fill these values?
        image.yuv_format = PixelFormat::Yuv444; // Expect RGB for now.
        image.yuv_range = YuvRange::Full;
        image.chroma_sample_position = ChromaSamplePosition::Unknown;
        // TODO: b/456440247 - Use information from colr nclx to fill these values?
        image.color_primaries = ColorPrimaries::Unspecified;
        image.transfer_characteristics = TransferCharacteristics::Unspecified;
        image.matrix_coefficients = MatrixCoefficients::Unspecified;

        image.allocate_planes(Category::Color)?;
        match basic_info.num_extra_channels {
            0 => {}
            1 => {
                image.alpha_present = true;
                image.allocate_planes(Category::Alpha)?
            }
            n => return AvifError::unknown_error(format!("Unexpected {n} extra JPEG XL channels")),
        }

        // TODO: b/456440247 - Remove RGB->YUV->RGB unnecessary conversion.
        let mut rgb = reformat::rgb::Image::create_from_yuv(image);
        rgb.allocate()?;

        let (pixel_format, size) = rgb_image_to_jxl_pixel_format(&rgb)?;
        // # Safety: Calling a C function with valid parameters.
        unsafe {
            JxlDecoderSetImageOutBuffer(decoder, &pixel_format, rgb.pixels_mut().cast(), size)
        }
        .map_dec_err()?;

        // # Safety: Calling a C function with valid parameters.
        let status = unsafe { JxlDecoderProcessInput(decoder) };
        if status != JxlDecoderStatus_JXL_DEC_FULL_IMAGE {
            return AvifError::unknown_error(format!(
                "Unexpected JxlDecoderStatus {status} instead of FULL_IMAGE"
            ));
        }
        // Note that JxlDecoderProcessInput() could be called a final time, expecting SUCCESS.
        // Whether this was the last frame or not is unknown here, so it is skipped.

        rgb.convert_to_yuv(image)?;
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
}

impl Drop for Libjxl {
    fn drop(&mut self) {
        if !self.encoder.is_null() {
            // # Safety: Calling a C function with valid parameters.
            unsafe { JxlEncoderDestroy(self.encoder) };
            self.encoder = null_mut();
        }
        if !self.decoder.is_null() {
            // # Safety: Calling a C function with valid parameters.
            unsafe { JxlDecoderDestroy(self.decoder) };
            self.decoder = null_mut();
        }
    }
}

impl Libjxl {
    pub(crate) fn version() -> String {
        // # Safety: Calling a C function.
        let version = unsafe { JxlEncoderVersion() };
        format!(
            "jxl: v{}.{}.{}",
            version / 1000000,
            (version % 1000000) / 1000,
            version % 1000
        )
    }
}

#[test]
fn libjxl_enc_dec_test() -> Result<(), AvifError> {
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
        rgb.allocate()?;
        rgb.row_mut(0)?[0] = 42;
        rgb.row_mut(0)?[1] = 80;
        rgb.row_mut(0)?[2] = 0;
        rgb.convert_to_yuv(&mut image)?;
        image
    };

    let mut encoder = Libjxl::default();
    let mut config = EncoderConfig::default();
    config.quality = 100.0;
    let mut output_samples = vec![];
    encoder.encode_image(&image, Category::Color, &config, &mut output_samples)?;
    encoder.finish(&mut output_samples)?;
    assert_eq!(output_samples.len(), 1);

    let mut decoder = Libjxl::default();
    let mut decoded = Image::default();
    decoder.initialize(&DecoderConfig::default())?;
    decoder.get_next_image(&output_samples[0].data, 0xff, &mut decoded, Category::Color)?;

    for plane in YUV_PLANES {
        assert_eq!(image.row(plane, 0), decoded.row(plane, 0));
    }
    Ok(())
}
