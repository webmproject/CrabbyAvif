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
//! crate::decoder::av1 only decodes the first frame of an OBU stream, which must be an intra
//! frame. It is therefore suitable for still AVIF images (including grids, alpha and gain maps)
//! and for keyframes of image sequences, but not for inter frames of animations.

use crate::codecs::Decoder;
use crate::codecs::DecoderConfig;
use crate::decoder::av1;
use crate::decoder::item::Item;
use crate::decoder::GridImageHelper;
use crate::image::Image;
use crate::image::YuvRange;
use crate::internal_utils::*;
use crate::utils::pixels::Pixels;
use crate::*;

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

#[derive(Default)]
pub struct Native {
    config: Option<DecoderConfig>,
}

impl From<av1::Error> for AvifError {
    fn from(err: av1::Error) -> Self {
        match err {
            av1::Error::Truncated => AvifError::TruncatedData,
            _ => AvifError::UnknownError(format!("native: {err}")),
        }
    }
}

impl From<av1::ChromaLayout> for PixelFormat {
    fn from(layout: av1::ChromaLayout) -> Self {
        match layout {
            av1::ChromaLayout::Mono => PixelFormat::Yuv400,
            av1::ChromaLayout::Yuv420 => PixelFormat::Yuv420,
            av1::ChromaLayout::Yuv422 => PixelFormat::Yuv422,
            av1::ChromaLayout::Yuv444 => PixelFormat::Yuv444,
        }
    }
}

// A decoded frame along with the properties signaled in its sequence header.
struct DecodedFrame {
    info: av1::ImageInfo,
    frame: av1::Frame,
}

impl DecodedFrame {
    // Moves the decoded planes into CrabbyAvif pixels (stripping any leading alignment padding in
    // place), along with their respective row_bytes.
    fn into_planes(self) -> AvifResult<Vec<(Pixels, u32)>> {
        let mut pixels = create_vec_exact(3)?;
        match self.frame.planes {
            av1::Planes::U8(planes) => {
                for mut plane in planes {
                    if plane.offset > 0 {
                        if plane.offset > plane.data.len() {
                            return AvifError::unknown_error("native: invalid plane offset");
                        }
                        plane.data.drain(..plane.offset);
                    }
                    let row_bytes = u32_from_usize(plane.stride)?;
                    pixels.try_push((Pixels::Buffer(plane.data), row_bytes))?;
                }
            }
            av1::Planes::U16(planes) => {
                for mut plane in planes {
                    if plane.offset > 0 {
                        if plane.offset > plane.data.len() {
                            return AvifError::unknown_error("native: invalid plane offset");
                        }
                        plane.data.drain(..plane.offset);
                    }
                    let row_bytes = u32_from_usize(checked_mul!(plane.stride, 2)?)?;
                    pixels.try_push((Pixels::Buffer16(plane.data), row_bytes))?;
                }
            }
        }
        Ok(pixels)
    }
}

fn decode_payload(av1_payload: &[u8], options: &av1::DecodeOptions) -> AvifResult<DecodedFrame> {
    let (info, frame) = av1::decode_with_info(av1_payload, options)?;
    Ok(DecodedFrame { info, frame })
}

// Decodes |payloads| on |thread_count| threads, each one repeatedly picking the next payload to
// decode, and copies the decoded frames into |grid_image_helper| on the calling thread in the same
// order as |payloads|.
fn decode_payloads_in_parallel(
    payloads: &[Vec<u8>],
    options: &av1::DecodeOptions,
    thread_count: usize,
    grid_image_helper: &mut GridImageHelper,
) -> AvifResult<()> {
    // Frames decoded before all the previous ones were, waiting for their turn.
    let mut frames: Vec<Option<DecodedFrame>> = create_vec_exact(payloads.len())?;
    for _ in payloads {
        frames.try_push(None)?;
    }
    let next_index = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        let (sender, receiver) = std::sync::mpsc::channel();
        let mut handles = create_vec_exact(thread_count)?;
        for _ in 0..thread_count {
            #[allow(clippy::disallowed_methods)]
            let sender = sender.clone();
            let next_index = &next_index;
            let handle = std::thread::Builder::new()
                .spawn_scoped(scope, move || loop {
                    let index = next_index.fetch_add(1, Ordering::Relaxed);
                    let Some(payload) = payloads.get(index) else {
                        break;
                    };
                    let frame = decode_payload(payload, options);
                    let is_err = frame.is_err();
                    // Sending fails if the receiver is gone because of an error.
                    if sender.send((index, frame)).is_err() || is_err {
                        break;
                    }
                })
                .map_err(AvifError::map_unknown_error)?;
            handles.try_push(handle)?;
        }
        drop(sender);
        let mut next_frame = 0;
        for (index, frame) in receiver.iter() {
            // Fail as soon as any cell fails, even if previous cells are still being decoded.
            *frames
                .get_mut(index)
                .ok_or(AvifError::UnknownError("native: invalid cell index".into()))? =
                Some(frame?);
            while let Some(frame) = frames.get_mut(next_frame).and_then(Option::take) {
                let mut cell_image = Image::default();
                Native::frame_to_image(frame, &mut cell_image, grid_image_helper.category, true)?;
                grid_image_helper.copy_from_cell_image(&mut cell_image)?;
                next_frame += 1;
            }
        }
        for handle in handles {
            if handle.join().is_err() {
                return AvifError::unknown_error("native: decoding thread panicked");
            }
        }
        Ok(())
    })
}

impl Native {
    fn decode_options(&self, spatial_id: u8) -> AvifResult<av1::DecodeOptions> {
        let config = self.config.as_ref().ok_or(AvifError::InvalidArgument)?;
        // av1 always decodes operating point 0 and only outputs the first frame in the stream
        // (which is the base layer for layered images).
        if config.operating_point != 0 || config.all_layers || !matches!(spatial_id, 0 | 0xFF) {
            return AvifError::not_implemented();
        }
        let mut options = av1::DecodeOptions::default();
        if let Some(limit) = config.image_size_limit {
            options.max_frame_pixels = usize_from_u32(limit.get())?;
        }
        options.threads = (config.max_threads as usize).clamp(1, 32);
        Ok(options)
    }

    fn frame_to_image(
        decoded: DecodedFrame,
        image: &mut Image,
        category: Category,
        overwrite_alpha_plane_yuv_range: bool,
    ) -> AvifResult<()> {
        let width = u32_from_usize(decoded.frame.width)?;
        let height = u32_from_usize(decoded.frame.height)?;
        let depth = u8::try_from(decoded.frame.bit_depth).map_err(AvifError::map_unknown_error)?;
        let yuv_format: PixelFormat = decoded.frame.layout.into();
        let yuv_range =
            if decoded.info.color_range_full { YuvRange::Full } else { YuvRange::Limited };
        let color_primaries = decoded.info.color_primaries as u16;
        let transfer_characteristics = decoded.info.transfer_characteristics as u16;
        let matrix_coefficients = decoded.info.matrix_coefficients as u16;
        let planes = decoded.into_planes()?;
        match category {
            Category::Alpha => {
                if image.width > 0
                    && image.height > 0
                    && (image.width != width || image.height != height || image.depth != depth)
                {
                    return AvifError::unknown_error(format!(
                        "Alpha plane does not match existing image dimensions: {}-bit {}x{} vs \
                         {depth}-bit {width}x{height}",
                        image.depth, image.width, image.height
                    ));
                }
                // Only the luma plane of the decoded frame is used as the alpha plane.
                let (pixels, row_bytes) = planes
                    .into_iter()
                    .next()
                    .ok_or(AvifError::UnknownError("native returned no planes".into()))?;
                image.width = width;
                image.height = height;
                image.depth = depth;
                image.row_bytes[Plane::A.as_usize()] = row_bytes;
                image.planes[Plane::A.as_usize()] = Some(pixels);
                if overwrite_alpha_plane_yuv_range {
                    image.yuv_range = yuv_range;
                }
            }
            Category::Color | Category::Gainmap => {
                if planes.len() != yuv_format.plane_count() {
                    return AvifError::unknown_error(format!(
                        "native returned {} planes, expected {}",
                        planes.len(),
                        yuv_format.plane_count()
                    ));
                }
                image.width = width;
                image.height = height;
                image.depth = depth;
                image.yuv_format = yuv_format;
                image.yuv_range = yuv_range;
                image.color_primaries = color_primaries.into();
                image.transfer_characteristics = transfer_characteristics.into();
                image.matrix_coefficients = matrix_coefficients.into();
                for (plane, (pixels, row_bytes)) in planes.into_iter().enumerate() {
                    image.row_bytes[plane] = row_bytes;
                    image.planes[plane] = Some(pixels);
                }
                if yuv_format == PixelFormat::Yuv400 {
                    // Clear left over chroma planes from previous frames.
                    image.clear_chroma_planes();
                }
            }
        }
        Ok(())
    }

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
        av1_payload: &[u8],
        spatial_id: u8,
        image: &mut Image,
        category: Category,
        _item: Option<&Item>,
        _signal_eos: bool,
    ) -> AvifResult<()> {
        let options = self.decode_options(spatial_id)?;
        let frame = decode_payload(av1_payload, &options)?;
        Self::frame_to_image(frame, image, category, true)
    }

    fn get_next_image_grid(
        &mut self,
        payloads: &[Vec<u8>],
        spatial_id: u8,
        grid_image_helper: &mut GridImageHelper,
    ) -> AvifResult<()> {
        let mut options = self.decode_options(spatial_id)?;
        let thread_count = options.threads.min(payloads.len());
        if thread_count > 1 {
            // Decode several cells in parallel rather than each cell with several threads.
            options.threads = 1;
            decode_payloads_in_parallel(payloads, &options, thread_count, grid_image_helper)?;
        } else {
            for payload in payloads {
                let frame = decode_payload(payload, &options)?;
                let mut cell_image = Image::default();
                Self::frame_to_image(frame, &mut cell_image, grid_image_helper.category, true)?;
                grid_image_helper.copy_from_cell_image(&mut cell_image)?;
            }
        }
        if !grid_image_helper.is_grid_complete()? {
            return AvifError::unknown_error("native: not enough grid cells");
        }
        Ok(())
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
