use crate::codecs::Decoder;
use crate::image::Image;
use crate::internal_utils::pixels::*;
use crate::internal_utils::*;
use crate::*;

use ndk_sys::bindings::*;

use std::ffi::CString;
use std::ptr;

#[derive(Debug, Default)]
pub struct MediaCodec {
    codec: Option<*mut AMediaCodec>,
    format: Option<*mut AMediaFormat>,
    output_buffer_index: Option<usize>,
}

macro_rules! c_str {
    ($var: ident, $var_tmp:ident, $str:literal) => {
        let $var_tmp = CString::new($str).unwrap();
        let $var = $var_tmp.as_ptr();
    };
}

impl Decoder for MediaCodec {
    fn initialize(&mut self, _operating_point: u8, _all_layers: bool) -> AvifResult<()> {
        // Does not support operating point and all layers.
        if self.codec.is_some() {
            return Ok(()); // Already initialized.
        }
        //c_str!(codec_mime_type, codec_mime_type_tmp, "video/av01");
        //let codec = unsafe { AMediaCodec_createDecoderByType(codec_mime_type) };
        c_str!(codec_name, codec_name_tmp, "c2.android.av1.decoder");
        let codec = unsafe { AMediaCodec_createCodecByName(codec_name) };
        if codec.is_null() {
            return Err(AvifError::NoCodecAvailable);
        }
        let format = unsafe { AMediaFormat_new() };
        if format.is_null() {
            return Err(AvifError::UnknownError);
        }
        unsafe {
            c_str!(mime, mime_tmp, "mime");
            c_str!(mime_type, mime_type_tmp, "video/av01");
            AMediaFormat_setString(format, mime, mime_type);
            c_str!(width_str, width_str_tmp, "width");
            AMediaFormat_setInt32(format, width_str, 200);
            c_str!(height_str, height_str_tmp, "height");
            AMediaFormat_setInt32(format, height_str, 200);

            // https://developer.android.com/reference/android/media/MediaCodecInfo.CodecCapabilities#COLOR_FormatYUV420Flexible
            //c_str!(color_format_str, color_format_str_tmp, "color-format");
            //AMediaFormat_setInt32(format, color_format_str, 2135033992);
            //AMediaFormat_setInt32(format, color_format_str, 19);

            // TODO: for 10-bit need to set format to 54 in order to get 10-bit
            // output. Or maybe it is possible to get RGB 1010102 itself?
            // int32_t COLOR_FormatYUVP010 = 54;
            // rgb 1010102 = 2130750114
            //c_str!(color_format_str, color_format_str_tmp, "color-format");
            //AMediaFormat_setInt32(format, color_format_str, 54);

            // TODO: may have to set width and height.
            // fox is 1204x800.

            println!("mediacodec configure");
            if AMediaCodec_configure(codec, format, ptr::null_mut(), ptr::null_mut(), 0)
                != media_status_t_AMEDIA_OK
            {
                return Err(AvifError::NoCodecAvailable);
            }
            println!("mediacodec start");
            if AMediaCodec_start(codec) != media_status_t_AMEDIA_OK {
                return Err(AvifError::NoCodecAvailable);
            }
            AMediaFormat_delete(format);
        }
        self.codec = Some(codec);
        println!("codec: {:#?}", self.codec);
        Ok(())
    }

    fn get_next_image(
        &mut self,
        av1_payload: &[u8],
        _spatial_id: u8,
        image: &mut Image,
        category: usize,
    ) -> AvifResult<()> {
        if self.codec.is_none() {
            self.initialize(0, true)?;
        }
        let codec = self.codec.unwrap();
        if self.output_buffer_index.is_some() {
            // Release any existing output buffers.
            unsafe {
                AMediaCodec_releaseOutputBuffer(codec, self.output_buffer_index.unwrap(), false);
            }
        }
        println!("mediacodec dequeue_input_buffer");
        unsafe {
            let input_index = AMediaCodec_dequeueInputBuffer(codec, 0);
            if input_index >= 0 {
                let mut input_buffer_size: usize = 0;
                let input_buffer = AMediaCodec_getInputBuffer(
                    codec,
                    input_index as usize,
                    &mut input_buffer_size as *mut _,
                );
                if input_buffer.is_null() {
                    println!("input buffer at index {input_index} was null");
                    return Err(AvifError::UnknownError);
                }
                // TODO: Alternative is to create a slice from raw parts and use copy_from_slice.
                ptr::copy_nonoverlapping(av1_payload.as_ptr(), input_buffer, av1_payload.len());
                if AMediaCodec_queueInputBuffer(
                    codec,
                    usize_from_isize(input_index)?,
                    /*offset=*/ 0,
                    av1_payload.len(),
                    /*pts=*/ 0,
                    /*flags=*/ 0,
                ) != media_status_t_AMEDIA_OK
                {
                    return Err(AvifError::UnknownError);
                }
            } else {
                println!("got input index < 0: {input_index}");
                return Err(AvifError::UnknownError);
            }
        }
        let mut buffer: Option<*mut u8> = None;
        let mut buffer_size: usize = 0;
        let mut retry_count = 0;
        let mut buffer_info = AMediaCodecBufferInfo::default();
        while retry_count < 100 {
            retry_count += 1;
            //println!("mediacodec trying to dequeue output");
            unsafe {
                let output_index =
                    AMediaCodec_dequeueOutputBuffer(codec, &mut buffer_info as *mut _, 10000);
                if output_index >= 0 {
                    let output_buffer = AMediaCodec_getOutputBuffer(
                        codec,
                        usize_from_isize(output_index)?,
                        &mut buffer_size as *mut _,
                    );
                    if output_buffer.is_null() {
                        println!("output buffer is null");
                        return Err(AvifError::UnknownError);
                    }
                    buffer = Some(output_buffer);
                    self.output_buffer_index = Some(usize_from_isize(output_index)?);
                    break;
                } else if output_index == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED as isize {
                    println!("buffers changed.");
                    // TODO: what to do?
                    continue;
                } else if output_index == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED as isize {
                    let format = AMediaCodec_getOutputFormat(codec);
                    if format.is_null() {
                        println!("output format was null");
                        return Err(AvifError::UnknownError);
                    }
                    self.format = Some(format);
                    println!("format changed. {:#?}", format);
                    continue;
                } else if output_index == AMEDIACODEC_INFO_TRY_AGAIN_LATER as isize {
                    //println!("try again!");
                    continue;
                } else {
                    println!("mediacodec dequeue_output_buffer failed: {output_index}");
                    return Err(AvifError::UnknownError);
                }
            }
        }
        if buffer.is_none() {
            println!("did not get buffer from mediacodec");
            return Err(AvifError::UnknownError);
        }
        if self.format.is_none() {
            println!("format is none :(");
            return Err(AvifError::UnknownError);
        }
        let buffer = buffer.unwrap();
        let format = self.format.unwrap();
        let mut width: i32 = 0;
        let mut height: i32 = 0;
        let mut stride: i32 = 0;
        let mut color_format: i32 = 0;
        let mut color_range: i32 = 0;
        unsafe {
            println!("getting width");
            c_str!(width_str, width_str_tmp, "width");
            if !AMediaFormat_getInt32(format, width_str, &mut width as *mut _) {
                println!("error getting width");
                return Err(AvifError::UnknownError);
            }
            println!("getting height");
            c_str!(height_str, height_str_tmp, "height");
            if !AMediaFormat_getInt32(format, height_str, &mut height as *mut _) {
                println!("error getting height");
                return Err(AvifError::UnknownError);
            }
            // TODO: Figure out if this stride accounts for pixel size.
            println!("getting stride");
            c_str!(stride_str, stride_str_tmp, "stride");
            if !AMediaFormat_getInt32(format, stride_str, &mut stride as *mut _) {
                println!("error getting stride");
                return Err(AvifError::UnknownError);
            }
            println!("getting color format");
            c_str!(color_format_str, color_format_str_tmp, "color-format");
            if !AMediaFormat_getInt32(format, color_format_str, &mut color_format as *mut _) {
                println!("error getting color format. using default.");
                color_format = 2135033992;
            }
            println!("getting color range");
            c_str!(color_range_str, color_range_str_tmp, "color-range");
            if !AMediaFormat_getInt32(format, color_range_str, &mut color_range as *mut _) {
                println!("error getting color range. using default.");
                color_range = 0;
            }
        }
        println!("width: {:#?}", width);
        println!("height: {:#?}", height);
        println!("stride: {:#?}", stride);
        println!("color_format: {:#?}", color_format);
        println!("color_range: {:#?}", color_range);
        if category == 0 {
            image.width = width as u32;
            image.height = height as u32;
            image.depth = 8; // TODO: 10?
            let mut reverse_uv = true;
            image.yuv_format = match color_format {
                // Android maps all AV1 8-bit images into yuv 420.
                2135033992 => PixelFormat::Yuv420,
                19 => {
                    reverse_uv = false;
                    PixelFormat::Yuv420
                }
                _ => {
                    println!("unknown color format: {color_format}");
                    return Err(AvifError::UnknownError);
                }
            };
            image.full_range = color_range == 1;
            image.chroma_sample_position = ChromaSamplePosition::Unknown;

            image.color_primaries = ColorPrimaries::Unspecified;
            image.transfer_characteristics = TransferCharacteristics::Unspecified;
            image.matrix_coefficients = MatrixCoefficients::Unspecified;

            image.planes2[0] = Some(Pixels::Pointer(buffer));
            // TODO: u and v order must be inverted for color format 19.
            let u_plane_offset = isize_from_i32(stride * height)?;
            let u_index = if reverse_uv { 2 } else { 1 };
            image.planes2[u_index] =
                Some(Pixels::Pointer(unsafe { buffer.offset(u_plane_offset) }));
            let u_plane_size = isize_from_i32(((width + 1) / 2) * ((height + 1) / 2))?;
            let v_plane_offset = u_plane_offset + u_plane_size;
            let v_index = if reverse_uv { 1 } else { 2 };
            image.planes2[v_index] =
                Some(Pixels::Pointer(unsafe { buffer.offset(v_plane_offset) }));

            image.row_bytes[0] = stride as u32;
            image.row_bytes[1] = ((stride + 1) / 2) as u32;
            image.row_bytes[2] = ((stride + 1) / 2) as u32;
        } else if category == 1 {
            // TODO: make sure alpha plane matches previous alpha plane.
            image.width = width as u32;
            image.height = height as u32;
            image.depth = 8; // TODO: 10?
            image.full_range = color_range == 1;
            image.planes2[3] = Some(Pixels::Pointer(buffer));
            image.row_bytes[3] = stride as u32;
        }
        // TODO: gainmap category.
        Ok(())
    }
}

impl Drop for MediaCodec {
    fn drop(&mut self) {
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
        if self.format.is_some() {
            unsafe { AMediaFormat_delete(self.format.unwrap()) };
            self.format = None;
        }
    }
}
