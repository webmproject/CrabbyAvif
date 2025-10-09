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

use crate::codecs::Decoder;
use crate::codecs::DecoderConfig;
use crate::decoder::CodecChoice;
use crate::decoder::GridImageHelper;
use crate::image::Image;
use crate::image::YuvRange;
use crate::internal_utils::stream::IStream;
use crate::internal_utils::*;
use crate::utils::pixels::*;
use crate::*;

use ndk_sys::bindings::*;

use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;

#[cfg(android_soong)]
include!(concat!(env!("OUT_DIR"), "/mediaimage2_bindgen.rs"));

// This sub-module is used by non-soong Android builds. It contains the bindings necessary to
// infer the YUV format that comes out of MediaCodec. The C struct source is here:
// https://cs.android.com/android/platform/superproject/main/+/main:frameworks/native/headers/media_plugin/media/hardware/VideoAPI.h;l=60;drc=a68f3a49e36e043b1640fe85010b0005d1bdb875
#[allow(non_camel_case_types, non_snake_case, unused)]
#[cfg(not(android_soong))]
mod android_soong_placeholder {
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub(crate) struct android_MediaImage2_PlaneInfo {
        pub mOffset: u32,
        pub mColInc: i32,
        pub mRowInc: i32,
        pub mHorizSubsampling: u32,
        pub mVertSubsampling: u32,
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    pub(crate) struct android_MediaImage2 {
        pub mType: u32,
        pub mNumPlanes: u32,
        pub mWidth: u32,
        pub mHeight: u32,
        pub mBitDepth: u32,
        pub mBitDepthAllocated: u32,
        pub mPlane: [android_MediaImage2_PlaneInfo; 4usize],
    }

    #[allow(non_upper_case_globals)]
    pub(crate) const android_MediaImage2_Type_MEDIA_IMAGE_TYPE_YUV: u32 = 1;
}

#[cfg(not(android_soong))]
use android_soong_placeholder::*;

#[derive(Debug)]
struct MediaFormat {
    format: *mut AMediaFormat,
}

macro_rules! c_str {
    ($var: ident, $var_tmp:ident, $str:expr) => {
        let $var_tmp = CString::new($str).unwrap();
        let $var = $var_tmp.as_ptr();
    };
}

#[derive(Debug, Default)]
struct PlaneInfo {
    color_format: AndroidMediaCodecOutputColorFormat,
    offset: [isize; 3],
    row_stride: [u32; 3],
    column_stride: [u32; 3],
}

impl PlaneInfo {
    fn pixel_format(&self) -> PixelFormat {
        match self.color_format {
            AndroidMediaCodecOutputColorFormat::P010 => PixelFormat::AndroidP010,
            AndroidMediaCodecOutputColorFormat::Yuv420Flexible => {
                let u_before_v = self.offset[2] == self.offset[1] + 1;
                let v_before_u = self.offset[1] == self.offset[2] + 1;
                let is_nv_format = self.column_stride == [1, 2, 2] && (u_before_v || v_before_u);
                match (is_nv_format, u_before_v) {
                    (true, true) => PixelFormat::AndroidNv12,
                    (true, false) => PixelFormat::AndroidNv21,
                    (false, _) => PixelFormat::Yuv420,
                }
            }
        }
    }

    fn depth(&self) -> u8 {
        match self.color_format {
            AndroidMediaCodecOutputColorFormat::P010 => 16,
            AndroidMediaCodecOutputColorFormat::Yuv420Flexible => 8,
        }
    }
}

impl MediaFormat {
    // These constants are documented in
    // https://developer.android.com/reference/android/media/MediaFormat
    const COLOR_RANGE_LIMITED: i32 = 2;

    const COLOR_STANDARD_BT709: i32 = 1;
    const COLOR_STANDARD_BT601_PAL: i32 = 2;
    const COLOR_STANDARD_BT601_NTSC: i32 = 4;
    const COLOR_STANDARD_BT2020: i32 = 6;

    const COLOR_TRANSFER_LINEAR: i32 = 1;
    const COLOR_TRANSFER_SDR_VIDEO: i32 = 3;
    const COLOR_TRANSFER_HLG: i32 = 7;

    fn get_i32(&self, key: *const c_char) -> Option<i32> {
        let mut value: i32 = 0;
        match unsafe { AMediaFormat_getInt32(self.format, key, &mut value as *mut _) } {
            true => Some(value),
            false => None,
        }
    }

    fn get_i32_from_str(&self, key: &str) -> Option<i32> {
        c_str!(key_str, key_str_tmp, key);
        self.get_i32(key_str)
    }

    fn width(&self) -> AvifResult<i32> {
        self.get_i32(unsafe { AMEDIAFORMAT_KEY_WIDTH })
            .ok_or(AvifError::UnknownError("".into()))
    }

    fn height(&self) -> AvifResult<i32> {
        self.get_i32(unsafe { AMEDIAFORMAT_KEY_HEIGHT })
            .ok_or(AvifError::UnknownError("".into()))
    }

    fn slice_height(&self) -> AvifResult<i32> {
        self.get_i32(unsafe { AMEDIAFORMAT_KEY_SLICE_HEIGHT })
            .ok_or(AvifError::UnknownError("".into()))
    }

    fn stride(&self) -> AvifResult<i32> {
        self.get_i32(unsafe { AMEDIAFORMAT_KEY_STRIDE })
            .ok_or(AvifError::UnknownError("".into()))
    }

    fn color_format(&self) -> AvifResult<i32> {
        self.get_i32(unsafe { AMEDIAFORMAT_KEY_COLOR_FORMAT })
            .ok_or(AvifError::UnknownError("".into()))
    }

    fn color_range(&self) -> YuvRange {
        // color-range is documented but isn't exposed as a constant in the NDK:
        // https://developer.android.com/reference/android/media/MediaFormat#KEY_COLOR_RANGE
        let color_range = self
            .get_i32_from_str("color-range")
            .unwrap_or(Self::COLOR_RANGE_LIMITED);
        if color_range == Self::COLOR_RANGE_LIMITED {
            YuvRange::Limited
        } else {
            YuvRange::Full
        }
    }

    fn color_primaries(&self) -> ColorPrimaries {
        // color-standard is documented but isn't exposed as a constant in the NDK:
        // https://developer.android.com/reference/android/media/MediaFormat#KEY_COLOR_STANDARD
        let color_standard = self.get_i32_from_str("color-standard").unwrap_or(-1);
        match color_standard {
            Self::COLOR_STANDARD_BT709 => ColorPrimaries::Bt709,
            Self::COLOR_STANDARD_BT2020 => ColorPrimaries::Bt2020,
            Self::COLOR_STANDARD_BT601_PAL | Self::COLOR_STANDARD_BT601_NTSC => {
                ColorPrimaries::Bt601
            }
            _ => ColorPrimaries::Unspecified,
        }
    }

    fn transfer_characteristics(&self) -> TransferCharacteristics {
        // color-transfer is documented but isn't exposed as a constant in the NDK:
        // https://developer.android.com/reference/android/media/MediaFormat#KEY_COLOR_TRANSFER
        match self.get_i32_from_str("color-transfer").unwrap_or(-1) {
            Self::COLOR_TRANSFER_LINEAR => TransferCharacteristics::Linear,
            Self::COLOR_TRANSFER_HLG => TransferCharacteristics::Hlg,
            Self::COLOR_TRANSFER_SDR_VIDEO => TransferCharacteristics::Bt601,
            _ => TransferCharacteristics::Unspecified,
        }
    }

    fn guess_plane_info(&self) -> AvifResult<PlaneInfo> {
        let height = self.height()?;
        let slice_height = self.slice_height().unwrap_or(height);
        let stride = self.stride()?;
        let color_format: AndroidMediaCodecOutputColorFormat = self.color_format()?.into();
        let mut plane_info = PlaneInfo {
            color_format,
            ..Default::default()
        };
        match color_format {
            AndroidMediaCodecOutputColorFormat::P010 => {
                plane_info.row_stride = [
                    u32_from_i32(stride)?,
                    u32_from_i32(stride)?,
                    0, // V plane is not used for P010.
                ];
                plane_info.column_stride = [
                    2, 2, 0, // V plane is not used for P010.
                ];
                plane_info.offset = [
                    0,
                    isize_from_i32(stride * slice_height)?,
                    0, // V plane is not used for P010.
                ];
            }
            AndroidMediaCodecOutputColorFormat::Yuv420Flexible => {
                plane_info.row_stride = [
                    u32_from_i32(stride)?,
                    u32_from_i32((stride + 1) / 2)?,
                    u32_from_i32((stride + 1) / 2)?,
                ];
                plane_info.column_stride = [1, 1, 1];
                plane_info.offset[0] = 0;
                plane_info.offset[1] = isize_from_i32(stride * slice_height)?;
                let u_plane_size = isize_from_i32(((stride + 1) / 2) * ((height + 1) / 2))?;
                // When color format is YUV_420_FLEXIBLE, the V plane comes before the U plane.
                plane_info.offset[2] = plane_info.offset[1] - u_plane_size;
            }
        }
        Ok(plane_info)
    }

    fn get_plane_info(&self) -> AvifResult<PlaneInfo> {
        c_str!(key_str, key_str_tmp, "image-data");
        let mut data: *mut std::ffi::c_void = ptr::null_mut();
        let mut size: usize = 0;
        if !unsafe {
            AMediaFormat_getBuffer(
                self.format,
                key_str,
                &mut data as *mut _,
                &mut size as *mut _,
            )
        } {
            return self.guess_plane_info();
        }
        if size != std::mem::size_of::<android_MediaImage2>() {
            return self.guess_plane_info();
        }
        let image_data = unsafe { *(data as *const android_MediaImage2) };
        if image_data.mType != android_MediaImage2_Type_MEDIA_IMAGE_TYPE_YUV {
            return self.guess_plane_info();
        }
        let planes = unsafe { ptr::read_unaligned(ptr::addr_of!(image_data.mPlane)) };
        let mut plane_info = PlaneInfo {
            color_format: self.color_format()?.into(),
            ..Default::default()
        };
        // Clippy suggests using an iterator with an enumerator which does not seem more readable
        // than using explicit indices.
        #[allow(clippy::needless_range_loop)]
        for plane_index in 0usize..3 {
            plane_info.offset[plane_index] = isize_from_u32(planes[plane_index].mOffset)?;
            plane_info.row_stride[plane_index] = u32_from_i32(planes[plane_index].mRowInc)?;
            plane_info.column_stride[plane_index] = u32_from_i32(planes[plane_index].mColInc)?;
        }
        Ok(plane_info)
    }
}

enum CodecInitializer {
    ByName(String),
    ByMimeType(String),
}

#[cfg(android_soong)]
fn prefer_hardware_decoder(config: &DecoderConfig) -> bool {
    let prefer_hw = rustutils::system_properties::read_bool(
        "media.stagefright.thumbnail.prefer_hw_codecs",
        false,
    )
    .unwrap_or(false);
    if config.codec_config.is_avif() {
        // We will return true when all of the below conditions are true:
        // 1) prefer_hw is true.
        // 2) category is not Alpha and category is not Gainmap. We do not prefer hardware for
        //    decoding these categories since they generally tend to be monochrome images and using
        //    hardware for that is unreliable.
        // 3) profile is 0. As of Sep 2024, there are no AV1 hardware decoders that support
        //    anything other than profile 0.
        // 4) depth is 8. Since we query for decoder simply by mime type, there is no way to know
        //    if an AV1 hardware decoder supports 10-bit or not.
        prefer_hw
            && config.category != Category::Alpha
            && config.category != Category::Gainmap
            && config.codec_config.profile() == 0
            && config.codec_config.depth() == 8
    } else {
        // We will return true when one of the following conditions are true:
        // 1) prefer_hw is true.
        // 2) depth is greater than 8. As of Nov 2024, the default HEVC software decoder on Android
        //    only supports 8-bit images.
        prefer_hw || config.depth > 8
    }
}

fn get_codec_initializers(config: &DecoderConfig) -> Vec<CodecInitializer> {
    #[cfg(android_soong)]
    {
        // Use a specific decoder if it is requested.
        if let Ok(Some(decoder)) =
            rustutils::system_properties::read("media.crabbyavif.debug.decoder")
        {
            if !decoder.is_empty() {
                return vec![CodecInitializer::ByName(decoder)];
            }
        }
    }
    let dav1d = String::from("c2.android.av1-dav1d.decoder");
    let gav1 = String::from("c2.android.av1.decoder");
    let hevc = String::from("c2.android.hevc.decoder");
    // As of Sep 2024, c2.android.av1.decoder is the only known decoder to support 12-bit AV1. So
    // prefer that for 12 bit images.
    let prefer_gav1 = config.depth == 12;
    let is_avif = config.codec_config.is_avif();
    let mime_type = if is_avif { MediaCodec::AV1_MIME } else { MediaCodec::HEVC_MIME };
    let prefer_hw = false;
    #[cfg(android_soong)]
    let prefer_hw = prefer_hardware_decoder(config);
    match (prefer_hw, is_avif, prefer_gav1) {
        (true, false, _) => vec![
            CodecInitializer::ByMimeType(mime_type.to_string()),
            CodecInitializer::ByName(hevc),
        ],
        (false, false, _) => vec![
            CodecInitializer::ByName(hevc),
            CodecInitializer::ByMimeType(mime_type.to_string()),
        ],
        (true, true, true) => vec![
            CodecInitializer::ByName(gav1),
            CodecInitializer::ByMimeType(mime_type.to_string()),
            CodecInitializer::ByName(dav1d),
        ],
        (true, true, false) => vec![
            CodecInitializer::ByMimeType(mime_type.to_string()),
            CodecInitializer::ByName(dav1d),
            CodecInitializer::ByName(gav1),
        ],
        (false, true, true) => vec![
            CodecInitializer::ByName(gav1),
            CodecInitializer::ByName(dav1d),
            CodecInitializer::ByMimeType(mime_type.to_string()),
        ],
        (false, true, false) => vec![
            CodecInitializer::ByName(dav1d),
            CodecInitializer::ByName(gav1),
            CodecInitializer::ByMimeType(mime_type.to_string()),
        ],
    }
}

#[derive(Default)]
pub struct MediaCodec {
    codec: Option<*mut AMediaCodec>,
    codec_index: usize,
    format: Option<MediaFormat>,
    output_buffer_index: Option<usize>,
    config: Option<DecoderConfig>,
    codec_initializers: Vec<CodecInitializer>,
}

impl MediaCodec {
    const AV1_MIME: &str = "video/av01";
    const HEVC_MIME: &str = "video/hevc";
    const MAX_RETRIES: u32 = 100;
    const TIMEOUT: u32 = 10000;

    fn initialize_impl(&mut self, low_latency: bool) -> AvifResult<()> {
        let config = self.config.unwrap_ref();
        if self.codec_index >= self.codec_initializers.len() {
            return AvifError::no_codec_available();
        }
        let format = unsafe { AMediaFormat_new() };
        if format.is_null() {
            return AvifError::unknown_error("");
        }
        c_str!(
            mime_type,
            mime_type_tmp,
            if config.codec_config.is_avif() { Self::AV1_MIME } else { Self::HEVC_MIME }
        );
        unsafe {
            AMediaFormat_setString(format, AMEDIAFORMAT_KEY_MIME, mime_type);
            AMediaFormat_setInt32(format, AMEDIAFORMAT_KEY_WIDTH, i32_from_u32(config.width)?);
            AMediaFormat_setInt32(
                format,
                AMEDIAFORMAT_KEY_HEIGHT,
                i32_from_u32(config.height)?,
            );
            AMediaFormat_setInt32(
                format,
                AMEDIAFORMAT_KEY_COLOR_FORMAT,
                if config.depth == 8 {
                    // For 8-bit images, always use Yuv420Flexible.
                    AndroidMediaCodecOutputColorFormat::Yuv420Flexible
                } else {
                    // For all other images, use whatever format is requested.
                    config.android_mediacodec_output_color_format
                } as i32,
            );
            if low_latency {
                // low-latency is documented but isn't exposed as a constant in the NDK:
                // https://developer.android.com/reference/android/media/MediaFormat#KEY_LOW_LATENCY
                c_str!(low_latency_str, low_latency_tmp, "low-latency");
                AMediaFormat_setInt32(format, low_latency_str, 1);
            }
            AMediaFormat_setInt32(
                format,
                AMEDIAFORMAT_KEY_MAX_INPUT_SIZE,
                i32_from_usize(config.max_input_size)?,
            );
            let codec_specific_data = config.codec_config.raw_data();
            if !codec_specific_data.is_empty() {
                AMediaFormat_setBuffer(
                    format,
                    AMEDIAFORMAT_KEY_CSD_0,
                    codec_specific_data.as_ptr() as *const _,
                    codec_specific_data.len(),
                );
            }
            // For video codecs, 0 is the highest importance (higher the number lesser the
            // importance). To make codec for images less important, give it a value more than 0.
            c_str!(importance, importance_tmp, "importance");
            AMediaFormat_setInt32(format, importance, 1);
        }

        let codec = match &self.codec_initializers[self.codec_index] {
            CodecInitializer::ByName(name) => {
                c_str!(codec_name, codec_name_tmp, name.as_str());
                unsafe { AMediaCodec_createCodecByName(codec_name) }
            }
            CodecInitializer::ByMimeType(mime_type) => {
                c_str!(codec_mime, codec_mime_tmp, mime_type.as_str());
                unsafe { AMediaCodec_createDecoderByType(codec_mime) }
            }
        };
        if codec.is_null() {
            unsafe { AMediaFormat_delete(format) };
            return AvifError::no_codec_available();
        }
        let status =
            unsafe { AMediaCodec_configure(codec, format, ptr::null_mut(), ptr::null_mut(), 0) };
        if status != media_status_t_AMEDIA_OK {
            unsafe {
                AMediaCodec_delete(codec);
                AMediaFormat_delete(format);
            }
            return AvifError::no_codec_available();
        }
        let status = unsafe { AMediaCodec_start(codec) };
        if status != media_status_t_AMEDIA_OK {
            unsafe {
                AMediaCodec_delete(codec);
                AMediaFormat_delete(format);
            }
            return AvifError::no_codec_available();
        }
        self.codec = Some(codec);
        Ok(())
    }

    fn output_buffer_to_image(
        &self,
        buffer: *mut u8,
        image: &mut Image,
        category: Category,
    ) -> AvifResult<()> {
        if self.format.is_none() {
            return AvifError::unknown_error("format is none");
        }
        let format = self.format.unwrap_ref();
        image.width = format.width()? as u32;
        image.height = format.height()? as u32;
        image.yuv_range = format.color_range();
        let plane_info = format.get_plane_info()?;
        image.depth = plane_info.depth();
        image.yuv_format = plane_info.pixel_format();
        match category {
            Category::Alpha => {
                image.row_bytes[3] = plane_info.row_stride[0];
                image.planes[3] = Some(Pixels::from_raw_pointer(
                    unsafe { buffer.offset(plane_info.offset[0]) },
                    image.depth as u32,
                    image.height,
                    image.row_bytes[3],
                )?);
            }
            _ => {
                image.chroma_sample_position = ChromaSamplePosition::Unknown;
                image.color_primaries = format.color_primaries();
                image.transfer_characteristics = format.transfer_characteristics();
                // MediaCodec does not expose matrix coefficients. Try to infer that based on color
                // primaries to get the most accurate color conversion possible.
                image.matrix_coefficients = match image.color_primaries {
                    ColorPrimaries::Bt601 => MatrixCoefficients::Bt601,
                    ColorPrimaries::Bt709 => MatrixCoefficients::Bt709,
                    ColorPrimaries::Bt2020 => MatrixCoefficients::Bt2020Ncl,
                    _ => MatrixCoefficients::Unspecified,
                };

                if image.yuv_format == PixelFormat::AndroidNv21 {
                    #[cfg(feature = "libyuv")]
                    {
                        // Convert Nv21 images into Nv12 for the following reasons:
                        // * Many of the yuv -> rgb conversions are optimized for Nv12 (Nv21 is also
                        //   missing several cases).
                        // * In Nv21 mode, some hardware decoders (e.g. c2.mtk.av1.decoder) will output
                        //   Nv21 for the first few frames and then switch to Nv12. crabbyavif does not
                        //   support cells within a same image to be of different pixel formats.
                        image.yuv_format = PixelFormat::AndroidNv12;
                        image.allocate_planes(category)?;
                        let planes = image.plane_ptrs_mut();
                        let row_bytes = image.plane_row_bytes()?;
                        if unsafe {
                            libyuv_sys::bindings::NV21ToNV12(
                                buffer.offset(plane_info.offset[0]),
                                i32_from_u32(plane_info.row_stride[0])?,
                                buffer.offset(plane_info.offset[2]),
                                i32_from_u32(plane_info.row_stride[2])?,
                                planes[0],
                                row_bytes[0],
                                planes[1],
                                row_bytes[1],
                                i32_from_u32(image.width)?,
                                i32_from_u32(image.height)?,
                            )
                        } != 0
                        {
                            return AvifError::reformat_failed();
                        }
                    }
                    #[cfg(not(feature = "libyuv"))]
                    {
                        return AvifError::not_implemented();
                    }
                } else {
                    for i in 0usize..3 {
                        if i == 2
                            && matches!(
                                image.yuv_format,
                                PixelFormat::AndroidP010 | PixelFormat::AndroidNv12
                            )
                        {
                            // V plane is not needed for these formats.
                            break;
                        }
                        image.row_bytes[i] = plane_info.row_stride[i];
                        let plane_height =
                            if i == 0 { image.height } else { (image.height + 1) / 2 };
                        image.planes[i] = Some(Pixels::from_raw_pointer(
                            unsafe { buffer.offset(plane_info.offset[i]) },
                            image.depth as u32,
                            plane_height,
                            image.row_bytes[i],
                        )?);
                    }
                }
            }
        }
        Ok(())
    }

    fn enqueue_payload(&self, input_index: isize, payload: &[u8], flags: u32) -> AvifResult<()> {
        let codec = self.codec.unwrap();
        let mut input_buffer_size: usize = 0;
        let input_buffer = unsafe {
            AMediaCodec_getInputBuffer(
                codec,
                input_index as usize,
                &mut input_buffer_size as *mut _,
            )
        };
        if input_buffer.is_null() {
            return AvifError::unknown_error(format!(
                "input buffer at index {input_index} was null"
            ));
        }
        let hevc_whole_nal_units = self.hevc_whole_nal_units(payload)?;
        let codec_payload = match &hevc_whole_nal_units {
            Some(hevc_payload) => hevc_payload,
            None => payload,
        };
        if input_buffer_size < codec_payload.len() {
            return AvifError::unknown_error(format!(
                "input buffer (size {input_buffer_size}) was not big enough. required size: {}",
                codec_payload.len()
            ));
        }
        unsafe {
            ptr::copy_nonoverlapping(codec_payload.as_ptr(), input_buffer, codec_payload.len());

            if AMediaCodec_queueInputBuffer(
                codec,
                usize_from_isize(input_index)?,
                /*offset=*/ 0,
                codec_payload.len(),
                /*pts=*/ 0,
                flags,
            ) != media_status_t_AMEDIA_OK
            {
                return AvifError::unknown_error("");
            }
        }
        Ok(())
    }

    fn get_next_image_impl(
        &mut self,
        payload: &[u8],
        _spatial_id: u8,
        image: &mut Image,
        category: Category,
    ) -> AvifResult<()> {
        if self.codec.is_none() {
            self.initialize_impl(/*low_latency=*/ true)?;
        }
        let codec = self.codec.unwrap();
        if self.output_buffer_index.is_some() {
            // Release any existing output buffer.
            unsafe {
                AMediaCodec_releaseOutputBuffer(codec, self.output_buffer_index.unwrap(), false);
            }
        }
        let mut retry_count = 0;
        unsafe {
            while retry_count < Self::MAX_RETRIES {
                retry_count += 1;
                let input_index = AMediaCodec_dequeueInputBuffer(codec, Self::TIMEOUT as _);
                if input_index >= 0 {
                    self.enqueue_payload(input_index, payload, 0)?;
                    break;
                } else if input_index == AMEDIACODEC_INFO_TRY_AGAIN_LATER as isize {
                    continue;
                } else {
                    return AvifError::unknown_error(format!("got input index < 0: {input_index}"));
                }
            }
        }
        let mut buffer: Option<*mut u8> = None;
        let mut buffer_size: usize = 0;
        let mut buffer_info = AMediaCodecBufferInfo::default();
        retry_count = 0;
        while retry_count < Self::MAX_RETRIES {
            retry_count += 1;
            unsafe {
                let output_index = AMediaCodec_dequeueOutputBuffer(
                    codec,
                    &mut buffer_info as *mut _,
                    Self::TIMEOUT as _,
                );
                if output_index >= 0 {
                    let output_buffer = AMediaCodec_getOutputBuffer(
                        codec,
                        usize_from_isize(output_index)?,
                        &mut buffer_size as *mut _,
                    );
                    if output_buffer.is_null() {
                        return AvifError::unknown_error("output buffer is null");
                    }
                    buffer = Some(output_buffer);
                    self.output_buffer_index = Some(usize_from_isize(output_index)?);
                    break;
                } else if output_index == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED as isize {
                    continue;
                } else if output_index == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED as isize {
                    let format = AMediaCodec_getOutputFormat(codec);
                    if format.is_null() {
                        return AvifError::unknown_error("output format was null");
                    }
                    self.format = Some(MediaFormat { format });
                    continue;
                } else if output_index == AMEDIACODEC_INFO_TRY_AGAIN_LATER as isize {
                    continue;
                } else {
                    return AvifError::unknown_error(format!(
                        "mediacodec dequeue_output_buffer failed: {output_index}"
                    ));
                }
            }
        }
        if buffer.is_none() {
            return AvifError::unknown_error("did not get buffer from mediacodec");
        }
        self.output_buffer_to_image(buffer.unwrap(), image, category)?;
        Ok(())
    }

    fn get_next_image_grid_impl(
        &mut self,
        payloads: &[Vec<u8>],
        grid_image_helper: &mut GridImageHelper,
    ) -> AvifResult<()> {
        if self.codec.is_none() {
            self.initialize_impl(/*low_latency=*/ false)?;
        }
        let codec = self.codec.unwrap();
        let mut retry_count = 0;
        let mut payloads_iter = payloads.iter().peekable();
        unsafe {
            while !grid_image_helper.is_grid_complete()? {
                // Queue as many inputs as we possibly can, then block on dequeuing outputs. After
                // getting each output, come back and queue the inputs again to keep the decoder as
                // busy as possible.
                while payloads_iter.peek().is_some() {
                    let input_index = AMediaCodec_dequeueInputBuffer(codec, 0);
                    if input_index < 0 {
                        if retry_count >= Self::MAX_RETRIES {
                            return AvifError::unknown_error("max retries exceeded");
                        }
                        break;
                    }
                    let payload = payloads_iter.next().unwrap();
                    self.enqueue_payload(
                        input_index,
                        payload,
                        if payloads_iter.peek().is_some() {
                            0
                        } else {
                            AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM as u32
                        },
                    )?;
                }
                loop {
                    let mut buffer_info = AMediaCodecBufferInfo::default();
                    let output_index = AMediaCodec_dequeueOutputBuffer(
                        codec,
                        &mut buffer_info as *mut _,
                        Self::TIMEOUT as _,
                    );
                    if output_index == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED as isize {
                        continue;
                    } else if output_index == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED as isize {
                        let format = AMediaCodec_getOutputFormat(codec);
                        if format.is_null() {
                            return AvifError::unknown_error("output format was null");
                        }
                        self.format = Some(MediaFormat { format });
                        continue;
                    } else if output_index == AMEDIACODEC_INFO_TRY_AGAIN_LATER as isize {
                        retry_count += 1;
                        if retry_count >= Self::MAX_RETRIES {
                            return AvifError::unknown_error("max retries exceeded");
                        }
                        break;
                    } else if output_index < 0 {
                        return AvifError::unknown_error("");
                    } else {
                        let mut buffer_size: usize = 0;
                        let output_buffer = AMediaCodec_getOutputBuffer(
                            codec,
                            usize_from_isize(output_index)?,
                            &mut buffer_size as *mut _,
                        );
                        if output_buffer.is_null() {
                            return AvifError::unknown_error("output buffer is null");
                        }
                        let mut cell_image = Image::default();
                        self.output_buffer_to_image(
                            output_buffer,
                            &mut cell_image,
                            grid_image_helper.category,
                        )?;
                        grid_image_helper.copy_from_cell_image(&mut cell_image)?;
                        if !grid_image_helper.is_grid_complete()? {
                            // The last output buffer will be released when the codec is dropped.
                            AMediaCodec_releaseOutputBuffer(codec, output_index as _, false);
                        }
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    fn drop_impl(&mut self) {
        if self.codec.is_some() {
            if self.output_buffer_index.is_some() {
                unsafe {
                    AMediaCodec_releaseOutputBuffer(
                        self.codec.unwrap(),
                        self.output_buffer_index.unwrap(),
                        false,
                    );
                }
                self.output_buffer_index = None;
            }
            unsafe {
                AMediaCodec_stop(self.codec.unwrap());
                AMediaCodec_delete(self.codec.unwrap());
            }
            self.codec = None;
        }
        self.format = None;
    }
}

impl Decoder for MediaCodec {
    fn codec(&self) -> CodecChoice {
        CodecChoice::MediaCodec
    }

    fn initialize(&mut self, config: &DecoderConfig) -> AvifResult<()> {
        self.codec_initializers = get_codec_initializers(config);
        self.config = Some(config.clone());
        // Actual codec initialization will be performed in get_next_image since we may try
        // multiple codecs.
        Ok(())
    }

    fn get_next_image(
        &mut self,
        payload: &[u8],
        spatial_id: u8,
        image: &mut Image,
        category: Category,
    ) -> AvifResult<()> {
        while self.codec_index < self.codec_initializers.len() {
            let res = self.get_next_image_impl(payload, spatial_id, image, category);
            if res.is_ok() {
                return Ok(());
            }
            // Drop the current codec and try the next one.
            self.drop_impl();
            self.codec_index += 1;
        }
        AvifError::unknown_error("all the codecs failed to extract an image")
    }

    fn get_next_image_grid(
        &mut self,
        payloads: &[Vec<u8>],
        _spatial_id: u8,
        grid_image_helper: &mut GridImageHelper,
    ) -> AvifResult<()> {
        let starting_cell_index = grid_image_helper.cell_index;
        while self.codec_index < self.codec_initializers.len() {
            let res = self.get_next_image_grid_impl(payloads, grid_image_helper);
            if res.is_ok() {
                return Ok(());
            }
            // Drop the current codec and try the next one.
            self.drop_impl();
            self.codec_index += 1;
            // Reset the cell_index so that each codec starts from the first cell. Mixing cells
            // between codecs could result in different color formats for each cell which is not
            // supported.
            grid_image_helper.cell_index = starting_cell_index;
        }
        AvifError::unknown_error("all the codecs failed to extract an image")
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

impl MediaCodec {
    fn hevc_whole_nal_units(&self, payload: &[u8]) -> AvifResult<Option<Vec<u8>>> {
        if !self.config.unwrap_ref().codec_config.is_heic() {
            return Ok(None);
        }
        // For HEVC, MediaCodec expects whole NAL units with each unit prefixed with a start code
        // of "\x00\x00\x00\x01".
        let nal_length_size = self.config.unwrap_ref().codec_config.nal_length_size() as usize;
        let mut offset = 0;
        let mut hevc_payload = Vec::new();
        while offset < payload.len() {
            let payload_slice = &payload[offset..];
            let mut stream = IStream::create(payload_slice);
            let nal_length = usize_from_u64(stream.read_uxx(nal_length_size as u8)?)?;
            let nal_unit_end = checked_add!(nal_length, nal_length_size)?;
            let nal_unit_range = nal_length_size..nal_unit_end;
            check_slice_range(payload_slice.len(), &nal_unit_range)?;
            // Start code.
            hevc_payload.extend_from_slice(&[0, 0, 0, 1]);
            // NAL Unit.
            hevc_payload.extend_from_slice(&payload_slice[nal_unit_range]);
            offset = checked_add!(offset, nal_unit_end)?;
        }
        Ok(Some(hevc_payload))
    }
}

impl Drop for MediaFormat {
    fn drop(&mut self) {
        unsafe { AMediaFormat_delete(self.format) };
    }
}

impl Drop for MediaCodec {
    fn drop(&mut self) {
        self.drop_impl();
    }
}
