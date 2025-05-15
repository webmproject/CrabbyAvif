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
use crate::*;

#[repr(C)]
#[derive(Default)]
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

#[no_mangle]
pub unsafe extern "C" fn crabby_avifEncoderCreate() -> *mut avifEncoder {
    todo!();
}

#[no_mangle]
pub unsafe extern "C" fn crabby_avifEncoderDestroy(encoder: *mut avifEncoder) {
    todo!();
}

#[no_mangle]
pub unsafe extern "C" fn crabby_avifEncoderWrite(
    encoder: *mut avifEncoder,
    image: *const avifImage,
    output: *mut avifRWData,
) -> avifResult {
    todo!();
}

#[no_mangle]
pub unsafe extern "C" fn crabby_avifEncoderAddImage(
    encoder: *mut avifEncoder,
    image: *const avifImage,
    durationInTimescales: u64,
    addImageFlags: avifAddImageFlags,
) -> avifResult {
    todo!();
}

#[no_mangle]
pub unsafe extern "C" fn crabby_avifEncoderAddImageGrid(
    encoder: *mut avifEncoder,
    gridCols: u32,
    gridRows: u32,
    cellImages: *const *const avifImage,
    addImageFlags: avifAddImageFlags,
) -> avifResult {
    todo!();
}

#[no_mangle]
pub unsafe extern "C" fn crabby_avifEncoderFinish(
    encoder: *mut avifEncoder,
    output: *mut avifRWData,
) -> avifResult {
    todo!();
}
