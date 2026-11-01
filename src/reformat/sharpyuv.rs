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

use crate::image::*;
use crate::internal_utils::*;
use crate::reformat::rgb;
use crate::reformat::rgb::*;
use crate::*;

use libsharpyuv_sys::bindings::*;

use std::os::raw::c_void;

pub(crate) fn rgb_to_yuv(rgb: &rgb::Image, image: &mut image::Image) -> AvifResult<()> {
    let color_space = match (image as &image::Image).into() {
        Mode::YuvCoefficients(kr, _kg, kb) => SharpYuvColorSpace {
            kr,
            kb,
            bit_depth: image.depth as _,
            range: if image.yuv_range == YuvRange::Limited {
                SharpYuvRange_kSharpYuvRangeLimited
            } else {
                SharpYuvRange_kSharpYuvRangeFull
            },
        },
        _ => return AvifError::not_implemented(),
    };
    let mut matrix = SharpYuvConversionMatrix {
        rgb_to_y: [0; 4],
        rgb_to_u: [0; 4],
        rgb_to_v: [0; 4],
    };
    // SAFETY: `color_space` and `matrix` are valid, aligned, initialized stack structs;
    // `SharpYuvComputeConversionMatrix` only reads `color_space` and writes `matrix`.
    unsafe {
        SharpYuvComputeConversionMatrix(&color_space, &mut matrix);
    }
    // bindgen does not expose SHARPYUV_VERSION directly since it's populated with a macro. The
    // code to compute the version is duplicated from libwebp.
    const SHARPYUV_VERSION: i32 = ((SHARPYUV_VERSION_MAJOR << 24)
        | (SHARPYUV_VERSION_MINOR << 16)
        | SHARPYUV_VERSION_PATCH) as i32;
    let mut options = SharpYuvOptions {
        yuv_matrix: &matrix,
        transfer_type: SharpYuvTransferFunctionType_kSharpYuvTransferFunctionSrgb,
    };
    // SAFETY: `matrix` and `options` are valid, aligned, initialized stack structs and `matrix`
    // outlives `options` for the entire duration of this function.
    let init_ok = unsafe { SharpYuvOptionsInitInternal(&matrix, &mut options, SHARPYUV_VERSION) };
    if init_ok == 0 {
        return AvifError::reformat_failed();
    }
    if image.transfer_characteristics != TransferCharacteristics::Unspecified {
        options.transfer_type = image.transfer_characteristics as _;
    }
    let plane_u8 = image.plane_ptrs_mut();
    let plane_row_bytes = image.plane_row_bytes()?;
    let (r_ptr, g_ptr, b_ptr) = if rgb.depth == 8 {
        let row = rgb.row(0)?;
        (
            row.get(rgb.format.r_offset()..)
                .ok_or(AvifError::ReformatFailed)?
                .as_ptr() as *const c_void,
            row.get(rgb.format.g_offset()..)
                .ok_or(AvifError::ReformatFailed)?
                .as_ptr() as *const c_void,
            row.get(rgb.format.b_offset()..)
                .ok_or(AvifError::ReformatFailed)?
                .as_ptr() as *const c_void,
        )
    } else {
        let row = rgb.row16(0)?;
        (
            row.get(rgb.format.r_offset()..)
                .ok_or(AvifError::ReformatFailed)?
                .as_ptr() as *const c_void,
            row.get(rgb.format.g_offset()..)
                .ok_or(AvifError::ReformatFailed)?
                .as_ptr() as *const c_void,
            row.get(rgb.format.b_offset()..)
                .ok_or(AvifError::ReformatFailed)?
                .as_ptr() as *const c_void,
        )
    };
    // SAFETY: Calling the C library conversion function. Pointers and strides are guaranteed to be
    // valid.
    if unsafe {
        SharpYuvConvertWithOptions(
            r_ptr,
            g_ptr,
            b_ptr,
            rgb.format.pixel_size(rgb.depth as _) as _,
            i32_from_u32(rgb.row_bytes)?,
            rgb.depth as _,
            plane_u8[0] as *mut _,
            plane_row_bytes[0],
            plane_u8[1] as *mut _,
            plane_row_bytes[1],
            plane_u8[2] as *mut _,
            plane_row_bytes[2],
            image.depth as _,
            i32_from_u32(rgb.width)?,
            i32_from_u32(rgb.height)?,
            &options as *const _,
        )
    } == 0
    {
        AvifError::reformat_failed()
    } else {
        Ok(())
    }
}
