use crate::codecs::Decoder;
use crate::image::Image;
use crate::*;

use ndk::media::media_codec::DequeuedInputBufferResult;
use ndk::media::media_codec::DequeuedOutputBufferInfoResult;
use ndk::media::media_codec::MediaCodec as NdkMediaCodec;
use ndk::media::media_codec::MediaCodecDirection;
use ndk::media::media_codec::MediaFormat;
use ndk::media::media_codec::OutputBuffer;

use std::time::Duration;

#[derive(Debug, Default)]
pub struct MediaCodec {
    codec: Option<NdkMediaCodec>,
    format: Option<MediaFormat>,
}

impl Decoder for MediaCodec {
    fn initialize(&mut self, _operating_point: u8, _all_layers: bool) -> AvifResult<()> {
        // Does not support operating point and all layers.
        if self.codec.is_some() {
            return Ok(()); // Already initialized.
        }
        self.codec = NdkMediaCodec::from_decoder_type("video/av01");
        if self.codec.is_none() {
            return Err(AvifError::NoCodecAvailable);
        }
        let format = MediaFormat::new();
        format.set_str("mime", "video/av01");
        format.set_i32("width", 200);
        format.set_i32("height", 200);

        // TODO: Need to set it to 2135033992 but seems to be having some
        // stride issues. Investigate.

        // format.set_i32("color-format", 2135033992);
        // format.set_i32("color-format", 19);

        // TODO: for 10-bit need to set format to 54 in order to get 10-bit
        // output. Or maybe it is possible to get RGB 1010102 itself?

        // int32_t COLOR_FormatYUVP010 = 54;

        // TODO: may have to set width and height.
        // fox is 1204x800.
        println!("mediacodec configure");
        self.codec
            .as_ref()
            .unwrap()
            .configure(&format, None, MediaCodecDirection::Decoder)
            .or(Err(AvifError::NoCodecAvailable))?;
        println!("mediacodec start");
        self.codec
            .as_ref()
            .unwrap()
            .start()
            .or(Err(AvifError::NoCodecAvailable))?;
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

        let codec = self.codec.as_ref().unwrap();
        println!("mediacodec dequeue_input_buffer");
        match codec.dequeue_input_buffer(Duration::from_micros(0)) {
            Ok(dequeue_result) => match dequeue_result {
                DequeuedInputBufferResult::Buffer(mut buffer) => {
                    println!("got input_buffer: {:#?}", buffer);
                    // TODO: is this unsafe necessary?
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            av1_payload.as_ptr(),
                            buffer.buffer_mut().as_mut_ptr().cast(),
                            av1_payload.len(),
                        )
                    };
                    println!(
                        "mediacodec queue_input_buffer of size: {}",
                        av1_payload.len()
                    );
                    codec
                        .queue_input_buffer(buffer, 0, av1_payload.len(), 0, 0)
                        .or(Err(AvifError::UnknownError))?;
                }
                DequeuedInputBufferResult::TryAgainLater => {
                    println!("got try again later");
                    return Err(AvifError::UnknownError);
                }
            },
            Err(_err) => {
                println!("mediacodec dequeue_input_buffer failed");
                return Err(AvifError::UnknownError);
            }
        }
        let mut buffer: Option<OutputBuffer> = None;
        let mut retry_count = 0;
        while retry_count < 100 {
            retry_count += 1;
            //println!("mediacodec trying to dequeue output");
            match codec.dequeue_output_buffer(Duration::from_millis(10)) {
                Ok(dequeue_result) => match dequeue_result {
                    DequeuedOutputBufferInfoResult::Buffer(output_buffer) => {
                        buffer = Some(output_buffer);
                        println!("got decoded buffer: {:#?}", buffer);
                        break;
                    }
                    DequeuedOutputBufferInfoResult::TryAgainLater => {
                        //println!("try again!");
                        continue;
                    }
                    DequeuedOutputBufferInfoResult::OutputFormatChanged => {
                        println!("format changed. {:#?}", codec.output_format());
                        self.format = Some(codec.output_format());
                        continue;
                    }
                    DequeuedOutputBufferInfoResult::OutputBuffersChanged => {
                        println!("buffers changed.");
                        // TODO: what to do?
                        continue;
                    }
                },
                Err(_err) => {
                    println!("mediacodec dequeue_output_buffer");
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
        let format = self.format.as_ref().unwrap();
        println!("getting width");
        let width = format.i32("width").ok_or(AvifError::UnknownError)?;
        println!("getting height");
        let height = format.i32("height").ok_or(AvifError::UnknownError)?;
        println!("getting stride");
        // TODO: Figure out if this stride accounts for pixel size.
        let stride = format.i32("stride").ok_or(AvifError::UnknownError)?;
        // https://developer.android.com/reference/android/media/MediaCodecInfo.CodecCapabilities#COLOR_FormatYUV420Planar
        let color_format = format.i32("color-format").unwrap_or(19);
        println!("width: {:#?}", width);
        println!("height: {:#?}", height);
        println!("stride: {:#?}", stride);
        println!("color_format: {:#?}", color_format);
        if category == 0 {
            image.width = width as u32;
            image.height = height as u32;
            image.depth = 8; // TODO: 10?
            image.yuv_format = match color_format {
                // Android maps all AV1 8-bit images into yuv 420.
                19 | 2135033992 => PixelFormat::Yuv420,
                _ => {
                    println!("unknown color format: {color_format}");
                    return Err(AvifError::UnknownError);
                }
            };
            image.full_range = format.i32("color-range").unwrap_or(0) == 1;
            image.chroma_sample_position = ChromaSamplePosition::Unknown;

            image.color_primaries = ColorPrimaries::Unspecified;
            image.transfer_characteristics = TransferCharacteristics::Unspecified;
            image.matrix_coefficients = MatrixCoefficients::Unspecified;

            image.copy_from_slice(buffer.buffer(), stride as u32, category)?;
        } else if category == 1 {
            // TODO: make sure alpha plane matches previous alpha plane.
            image.width = width as u32;
            image.height = height as u32;
            image.depth = 8; // TODO: 10?
            image.full_range = format.i32("color-range").unwrap_or(0) == 1;

            image.copy_from_slice(buffer.buffer(), stride as u32, category)?;
        }
        // TODO: gainmap category.
        Ok(())
    }
}
