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

#![allow(unused)]

use super::gainmap::*;
use super::image::*;
use super::io::*;
use super::types::*;

use std::ffi::CStr;
use std::num::NonZero;
use std::os::raw::c_char;

use crate::encoder::*;
use crate::gainmap::GainMap;
use crate::internal_utils::*;
use crate::*;

#[repr(C)]
pub struct avifEncoder {
    pub codecChoice: avifCodecChoice,
    pub maxThreads: i32,
    pub speed: i32,
    pub keyframeInterval: i32,
    pub timescale: u64,
    pub repetitionCount: i32,
    pub extraLayerCount: u32,
    pub quality: i32,
    pub qualityAlpha: i32,
    pub tileRowsLog2: i32,
    pub tileColsLog2: i32,
    pub autoTiling: avifBool,
    scalingMode: ScalingMode,
    pub ioStats: crate::decoder::IOStats,
    pub qualityGainMap: i32,
    rust_encoder: Box<Encoder>,
    rust_encoder_initialized: bool,
}

impl Default for avifEncoder {
    fn default() -> Self {
        let settings = Settings::default();
        Self {
            codecChoice: avifCodecChoice::Aom,
            maxThreads: settings.threads as _,
            speed: -1,
            keyframeInterval: settings.keyframe_interval,
            timescale: settings.timescale,
            repetitionCount: AVIF_REPETITION_COUNT_INFINITE,
            extraLayerCount: settings.extra_layer_count,
            quality: settings.mutable.quality,
            qualityAlpha: settings.mutable.quality,
            tileRowsLog2: 0,
            tileColsLog2: 0,
            autoTiling: AVIF_FALSE,
            scalingMode: settings.mutable.scaling_mode,
            ioStats: Default::default(),
            qualityGainMap: settings.mutable.quality,
            rust_encoder: Default::default(),
            rust_encoder_initialized: false,
        }
    }
}

impl From<&avifEncoder> for MutableSettings {
    fn from(encoder: &avifEncoder) -> Self {
        Self {
            quality: encoder.quality,
            // TODO - b/416560730: Convert to proper tiling mode.
            tiling_mode: TilingMode::Auto,
            scaling_mode: encoder.scalingMode,
        }
    }
}

impl From<&avifEncoder> for Settings {
    fn from(encoder: &avifEncoder) -> Self {
        Self {
            threads: encoder.maxThreads as u32,
            speed: Some(encoder.speed as u32),
            keyframe_interval: encoder.keyframeInterval,
            timescale: if encoder.timescale == 0 { 1 } else { encoder.timescale },
            repetition_count: RepetitionCount::create_from(encoder.repetitionCount),
            extra_layer_count: encoder.extraLayerCount,
            mutable: encoder.into(),
        }
    }
}

impl avifEncoder {
    fn initialize_or_update_rust_encoder(&mut self) -> avifResult {
        if self.rust_encoder_initialized {
            // TODO - b/416560730: Validate the immutable settings.
            let mutable_settings: MutableSettings = (&*self).into();
            self.rust_encoder.update_settings(&mutable_settings).into()
        } else {
            let settings: Settings = (&*self).into();
            match Encoder::create_with_settings(&settings) {
                Ok(encoder) => self.rust_encoder = Box::new(encoder),
                Err(err) => return (&err).into(),
            }
            self.rust_encoder_initialized = true;
            avifResult::Ok
        }
    }
}

fn rust_encoder<'a>(encoder: *mut avifEncoder) -> &'a mut Encoder {
    &mut deref_mut!(encoder).rust_encoder
}

#[no_mangle]
pub unsafe extern "C" fn crabby_avifEncoderCreate() -> *mut avifEncoder {
    Box::into_raw(Box::<avifEncoder>::default())
}

#[no_mangle]
pub unsafe extern "C" fn crabby_avifEncoderDestroy(encoder: *mut avifEncoder) {
    let _ = unsafe { Box::from_raw(encoder) };
}

#[no_mangle]
pub unsafe extern "C" fn crabby_avifEncoderWrite(
    encoder: *mut avifEncoder,
    image: *const avifImage,
    output: *mut avifRWData,
) -> avifResult {
    let res = unsafe { crabby_avifEncoderAddImage(encoder, image, 1, AVIF_ADD_IMAGE_FLAG_SINGLE) };
    if res != avifResult::Ok {
        return res;
    }
    unsafe { crabby_avifEncoderFinish(encoder, output) }
}

#[no_mangle]
pub unsafe extern "C" fn crabby_avifEncoderAddImage(
    encoder: *mut avifEncoder,
    image: *const avifImage,
    durationInTimescales: u64,
    addImageFlags: avifAddImageFlags,
) -> avifResult {
    let encoder_ref = deref_mut!(encoder);
    let res = encoder_ref.initialize_or_update_rust_encoder();
    if res != avifResult::Ok {
        return res;
    }
    let gainmap = deref_const!(image).gainmap();
    let image: image::Image = deref_const!(image).into();
    if (addImageFlags & AVIF_ADD_IMAGE_FLAG_SINGLE) != 0 || encoder_ref.extraLayerCount != 0 {
        match &gainmap {
            Some(gainmap) => rust_encoder(encoder)
                .add_image_gainmap(&image, gainmap)
                .into(),
            None => rust_encoder(encoder).add_image(&image).into(),
        }
    } else {
        rust_encoder(encoder)
            .add_image_for_sequence(&image, durationInTimescales)
            .into()
    }
}

#[no_mangle]
pub unsafe extern "C" fn crabby_avifEncoderAddImageGrid(
    encoder: *mut avifEncoder,
    gridCols: u32,
    gridRows: u32,
    cellImages: *const *const avifImage,
    addImageFlags: avifAddImageFlags,
) -> avifResult {
    let encoder_ref = deref_mut!(encoder);
    if cellImages.is_null()
        || gridCols == 0
        || gridRows == 0
        // If we are not encoding a grid image with multiple layers, AVIF_ADD_IMAGE_FLAG_SINGLE has
        // to be set.
        || ((addImageFlags & AVIF_ADD_IMAGE_FLAG_SINGLE) == 0 && encoder_ref.extraLayerCount == 0)
    {
        return avifResult::InvalidArgument;
    }
    let res = encoder_ref.initialize_or_update_rust_encoder();
    if res != avifResult::Ok {
        return res;
    }
    let cell_count = match gridCols.checked_mul(gridRows) {
        Some(value) => value as usize,
        None => return avifResult::InvalidArgument,
    };
    let mut images: Vec<image::Image> = match create_vec_exact(cell_count) {
        Ok(x) => x,
        Err(_) => return avifResult::OutOfMemory,
    };
    let image_ptrs: &[*const avifImage] =
        unsafe { std::slice::from_raw_parts(cellImages, cell_count) };
    let mut gainmaps: Vec<Option<GainMap>> = Vec::new();
    for image_ptr in image_ptrs {
        gainmaps.push(deref_const!(*image_ptr).gainmap());
        images.push(deref_const!(*image_ptr).into());
    }
    let image_refs: Vec<&Image> = images.iter().collect();
    if gainmaps.iter().all(|x| x.is_some()) {
        let gainmap_refs: Vec<&GainMap> = gainmaps.iter().map(|x| x.unwrap_ref()).collect();
        rust_encoder(encoder)
            .add_image_gainmap_grid(gridCols, gridRows, &image_refs, &gainmap_refs)
            .into()
    } else if gainmaps.iter().all(|x| x.is_none()) {
        rust_encoder(encoder)
            .add_image_grid(gridCols, gridRows, &image_refs)
            .into()
    } else {
        // Some cells had GainMap and some did not. This is invalid.
        avifResult::InvalidArgument
    }
}

#[no_mangle]
pub unsafe extern "C" fn crabby_avifEncoderFinish(
    encoder: *mut avifEncoder,
    output: *mut avifRWData,
) -> avifResult {
    match rust_encoder(encoder).finish() {
        Ok(encoded_data) => unsafe {
            crabby_avifRWDataSet(output, encoded_data.as_ptr(), encoded_data.len())
        },
        Err(err) => (&err).into(),
    }
}
