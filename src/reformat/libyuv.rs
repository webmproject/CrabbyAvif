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

use super::rgb;
use super::rgb::*;

use crate::image::*;
use crate::internal_utils::*;
use crate::*;

use libyuv_sys::bindings::*;

use std::os::raw::c_int;

#[derive(Clone, Copy)]
enum YuvMatrix {
    Bt709,
    Bt601,
    Bt2020,
}

fn find_constants(image: &image::Image) -> Option<(&'static YuvConstants, &'static YuvConstants)> {
    let matrix_coefficients = if image.yuv_format == PixelFormat::Yuv400
        && image.matrix_coefficients == MatrixCoefficients::Identity
    {
        MatrixCoefficients::Bt601
    } else {
        image.matrix_coefficients
    };
    // Android MediaCodec always uses Yuv420. So use Bt601 instead of Identity in that case.
    #[cfg(feature = "android_mediacodec")]
    let matrix_coefficients = if matrix_coefficients == MatrixCoefficients::Identity {
        MatrixCoefficients::Bt601
    } else {
        matrix_coefficients
    };
    let default = if cfg!(feature = "android_mediacodec") { Some(YuvMatrix::Bt601) } else { None };
    let matrix = match matrix_coefficients {
        MatrixCoefficients::Bt709 => Some(YuvMatrix::Bt709),
        MatrixCoefficients::Bt470bg
        | MatrixCoefficients::Bt601
        | MatrixCoefficients::Unspecified => Some(YuvMatrix::Bt601),
        MatrixCoefficients::Bt2020Ncl => Some(YuvMatrix::Bt2020),
        MatrixCoefficients::ChromaDerivedNcl => match image.color_primaries {
            ColorPrimaries::Srgb | ColorPrimaries::Unspecified => Some(YuvMatrix::Bt709),
            ColorPrimaries::Bt470bg | ColorPrimaries::Bt601 => Some(YuvMatrix::Bt601),
            ColorPrimaries::Bt2020 => Some(YuvMatrix::Bt2020),
            _ => default,
        },
        // default cannot be used here as this is expected to be near-lossless.
        MatrixCoefficients::YcgcoRe => None,
        _ => default,
    }?;
    // SAFETY: All `kYuv*Constants` and `kYvu*Constants` symbols are immutable, properly aligned
    // static `YuvConstants` coefficient tables defined by libyuv with `'static` lifetime and are
    // never mutated at runtime.
    Some(unsafe {
        match (image.yuv_range, matrix) {
            (YuvRange::Full, YuvMatrix::Bt709) => (&kYuvF709Constants, &kYvuF709Constants),
            (YuvRange::Full, YuvMatrix::Bt601) => (&kYuvJPEGConstants, &kYvuJPEGConstants),
            (YuvRange::Full, YuvMatrix::Bt2020) => (&kYuvV2020Constants, &kYvuV2020Constants),
            (YuvRange::Limited, YuvMatrix::Bt709) => (&kYuvH709Constants, &kYvuH709Constants),
            (YuvRange::Limited, YuvMatrix::Bt601) => (&kYuvI601Constants, &kYvuI601Constants),
            (YuvRange::Limited, YuvMatrix::Bt2020) => (&kYuv2020Constants, &kYvu2020Constants),
        }
    })
}

#[rustfmt::skip]
type YUV400ToRGBMatrix = unsafe extern "C" fn(
    *const u8, c_int, *mut u8, c_int, *const YuvConstants, c_int, c_int) -> c_int;
#[rustfmt::skip]
type YUVToRGBMatrixFilter = unsafe extern "C" fn(
    *const u8, c_int, *const u8, c_int, *const u8, c_int, *mut u8, c_int, *const YuvConstants,
    c_int, c_int, FilterMode) -> c_int;
#[rustfmt::skip]
type YUVAToRGBMatrixFilter = unsafe extern "C" fn(
    *const u8, c_int, *const u8, c_int, *const u8, c_int, *const u8, c_int, *mut u8, c_int,
    *const YuvConstants, c_int, c_int, c_int, FilterMode) -> c_int;
#[rustfmt::skip]
type YUVToRGBMatrix = unsafe extern "C" fn(
    *const u8, c_int, *const u8, c_int, *const u8, c_int, *mut u8, c_int, *const YuvConstants,
    c_int, c_int) -> c_int;
#[rustfmt::skip]
type YUVAToRGBMatrix = unsafe extern "C" fn(
    *const u8, c_int, *const u8, c_int, *const u8, c_int, *const u8, c_int, *mut u8, c_int,
    *const YuvConstants, c_int, c_int, c_int) -> c_int;
#[rustfmt::skip]
type YUVToRGBMatrixFilterHighBitDepth = unsafe extern "C" fn(
    *const u16, c_int, *const u16, c_int, *const u16, c_int, *mut u8, c_int, *const YuvConstants,
    c_int, c_int, FilterMode) -> c_int;
#[rustfmt::skip]
type YUVAToRGBMatrixFilterHighBitDepth = unsafe extern "C" fn(
    *const u16, c_int, *const u16, c_int, *const u16, c_int, *const u16, c_int, *mut u8, c_int,
    *const YuvConstants, c_int, c_int, c_int, FilterMode) -> c_int;
#[rustfmt::skip]
type YUVToRGBMatrixHighBitDepth = unsafe extern "C" fn(
    *const u16, c_int, *const u16, c_int, *const u16, c_int, *mut u8, c_int, *const YuvConstants,
    c_int, c_int) -> c_int;
#[rustfmt::skip]
type YUVAToRGBMatrixHighBitDepth = unsafe extern "C" fn(
    *const u16, c_int, *const u16, c_int, *const u16, c_int, *const u16, c_int, *mut u8, c_int,
    *const YuvConstants, c_int, c_int, c_int) -> c_int;
#[rustfmt::skip]
type P010ToRGBMatrix = unsafe extern "C" fn(
    *const u16, c_int, *const u16, c_int, *mut u8, c_int, *const YuvConstants, c_int,
    c_int) -> c_int;
#[rustfmt::skip]
type ARGBToABGR = unsafe extern "C" fn(
    *const u8, c_int, *mut u8, c_int, c_int, c_int) -> c_int;
#[rustfmt::skip]
type NVToARGBMatrix = unsafe extern "C" fn(
    *const u8, c_int, *const u8, c_int, *mut u8, c_int, *const YuvConstants, c_int,
    c_int) -> c_int;

#[derive(Debug)]
enum ConversionFunction {
    YUV400ToRGBMatrix(YUV400ToRGBMatrix),
    YUVToRGBMatrixFilter(YUVToRGBMatrixFilter),
    YUVAToRGBMatrixFilter(YUVAToRGBMatrixFilter),
    YUVToRGBMatrix(YUVToRGBMatrix),
    YUVAToRGBMatrix(YUVAToRGBMatrix),
    YUVToRGBMatrixFilterHighBitDepth(YUVToRGBMatrixFilterHighBitDepth),
    YUVAToRGBMatrixFilterHighBitDepth(YUVAToRGBMatrixFilterHighBitDepth),
    YUVToRGBMatrixHighBitDepth(YUVToRGBMatrixHighBitDepth),
    YUVAToRGBMatrixHighBitDepth(YUVAToRGBMatrixHighBitDepth),
    P010ToRGBMatrix(P010ToRGBMatrix, ARGBToABGR),
    YUVToAB30Matrix(YUVToRGBMatrixHighBitDepth, ARGBToABGR),
    NVToARGBMatrix(NVToARGBMatrix),
}

impl ConversionFunction {
    fn is_yuva(&self) -> bool {
        matches!(
            self,
            ConversionFunction::YUVAToRGBMatrixFilter(_)
                | ConversionFunction::YUVAToRGBMatrix(_)
                | ConversionFunction::YUVAToRGBMatrixFilterHighBitDepth(_)
                | ConversionFunction::YUVAToRGBMatrixHighBitDepth(_)
        )
    }
}

fn find_conversion_function(
    yuv_format: PixelFormat,
    yuv_depth: u8,
    rgb: &rgb::Image,
    alpha_preferred: bool,
) -> Option<ConversionFunction> {
    match (alpha_preferred, yuv_depth, rgb.format, yuv_format) {
        (_, 8, Format::Rgba, PixelFormat::AndroidNv12) => {
            // What Android considers to be NV12 is actually NV21 in libyuv.
            Some(ConversionFunction::NVToARGBMatrix(NV21ToARGBMatrix))
        }
        (_, 8, Format::Rgba, PixelFormat::AndroidNv21) => {
            // What Android considers to be NV21 is actually NV12 in libyuv.
            Some(ConversionFunction::NVToARGBMatrix(NV12ToARGBMatrix))
        }
        (_, 8, Format::Rgb565, PixelFormat::AndroidNv12) => {
            Some(ConversionFunction::NVToARGBMatrix(NV12ToRGB565Matrix))
        }
        (_, 16, Format::Rgba1010102, PixelFormat::AndroidP010) => Some(
            ConversionFunction::P010ToRGBMatrix(P010ToAR30Matrix, AR30ToAB30),
        ),
        (_, 10, Format::Rgba1010102, PixelFormat::Yuv420) => Some(
            ConversionFunction::YUVToAB30Matrix(I010ToAR30Matrix, AR30ToAB30),
        ),
        (_, 10, Format::Rgba1010102, PixelFormat::Yuv422) => Some(
            ConversionFunction::YUVToAB30Matrix(I210ToAR30Matrix, AR30ToAB30),
        ),
        (_, 10, Format::Rgba1010102, PixelFormat::Yuv444) => Some(
            ConversionFunction::YUVToAB30Matrix(I410ToAR30Matrix, AR30ToAB30),
        ),
        (_, 16, Format::Rgba, PixelFormat::AndroidP010) => Some(
            ConversionFunction::P010ToRGBMatrix(P010ToARGBMatrix, ARGBToABGR),
        ),
        (true, 10, Format::Rgba | Format::Bgra, PixelFormat::Yuv422)
            if rgb.chroma_upsampling.bilinear_or_better_filter_allowed() =>
        {
            Some(ConversionFunction::YUVAToRGBMatrixFilterHighBitDepth(
                I210AlphaToARGBMatrixFilter,
            ))
        }
        (true, 10, Format::Rgba | Format::Bgra, PixelFormat::Yuv420)
            if rgb.chroma_upsampling.bilinear_or_better_filter_allowed() =>
        {
            Some(ConversionFunction::YUVAToRGBMatrixFilterHighBitDepth(
                I010AlphaToARGBMatrixFilter,
            ))
        }
        (_, 10, Format::Rgba | Format::Bgra, PixelFormat::Yuv422)
            if rgb.chroma_upsampling.bilinear_or_better_filter_allowed() =>
        {
            Some(ConversionFunction::YUVToRGBMatrixFilterHighBitDepth(
                I210ToARGBMatrixFilter,
            ))
        }
        (_, 10, Format::Rgba | Format::Bgra, PixelFormat::Yuv420)
            if rgb.chroma_upsampling.bilinear_or_better_filter_allowed() =>
        {
            Some(ConversionFunction::YUVToRGBMatrixFilterHighBitDepth(
                I010ToARGBMatrixFilter,
            ))
        }

        (true, 10, Format::Rgba | Format::Bgra, PixelFormat::Yuv444) => Some(
            ConversionFunction::YUVAToRGBMatrixHighBitDepth(I410AlphaToARGBMatrix),
        ),
        (true, 10, Format::Rgba | Format::Bgra, PixelFormat::Yuv422)
            if rgb.chroma_upsampling.nearest_neighbor_filter_allowed() =>
        {
            Some(ConversionFunction::YUVAToRGBMatrixHighBitDepth(
                I210AlphaToARGBMatrix,
            ))
        }
        (true, 10, Format::Rgba | Format::Bgra, PixelFormat::Yuv420)
            if rgb.chroma_upsampling.nearest_neighbor_filter_allowed() =>
        {
            Some(ConversionFunction::YUVAToRGBMatrixHighBitDepth(
                I010AlphaToARGBMatrix,
            ))
        }
        (_, 10, Format::Rgba | Format::Bgra, PixelFormat::Yuv444) => Some(
            ConversionFunction::YUVToRGBMatrixHighBitDepth(I410ToARGBMatrix),
        ),
        (_, 10, Format::Rgba | Format::Bgra, PixelFormat::Yuv422)
            if rgb.chroma_upsampling.nearest_neighbor_filter_allowed() =>
        {
            Some(ConversionFunction::YUVToRGBMatrixHighBitDepth(
                I210ToARGBMatrix,
            ))
        }
        (_, 10, Format::Rgba | Format::Bgra, PixelFormat::Yuv420)
            if rgb.chroma_upsampling.nearest_neighbor_filter_allowed() =>
        {
            Some(ConversionFunction::YUVToRGBMatrixHighBitDepth(
                I010ToARGBMatrix,
            ))
        }
        (_, 12, Format::Rgba | Format::Bgra, PixelFormat::Yuv420)
            if rgb.chroma_upsampling.nearest_neighbor_filter_allowed() =>
        {
            Some(ConversionFunction::YUVToRGBMatrixHighBitDepth(
                I012ToARGBMatrix,
            ))
        }

        // The fall through here is intentional. If a high bitdepth function was not found, try to
        // see if we can use a low bitdepth function with a downshift.
        //
        (_, _, Format::Rgba | Format::Bgra, PixelFormat::Yuv400) => {
            Some(ConversionFunction::YUV400ToRGBMatrix(I400ToARGBMatrix))
        }

        (true, _, Format::Rgba | Format::Bgra, PixelFormat::Yuv422)
            if rgb.chroma_upsampling.bilinear_or_better_filter_allowed() =>
        {
            Some(ConversionFunction::YUVAToRGBMatrixFilter(
                I422AlphaToARGBMatrixFilter,
            ))
        }
        (true, _, Format::Rgba | Format::Bgra, PixelFormat::Yuv420)
            if rgb.chroma_upsampling.bilinear_or_better_filter_allowed() =>
        {
            Some(ConversionFunction::YUVAToRGBMatrixFilter(
                I420AlphaToARGBMatrixFilter,
            ))
        }

        (_, _, Format::Rgb | Format::Bgr, PixelFormat::Yuv422)
            if rgb.chroma_upsampling.bilinear_or_better_filter_allowed() =>
        {
            Some(ConversionFunction::YUVToRGBMatrixFilter(
                I422ToRGB24MatrixFilter,
            ))
        }
        (_, _, Format::Rgb | Format::Bgr, PixelFormat::Yuv420)
            if rgb.chroma_upsampling.bilinear_or_better_filter_allowed() =>
        {
            Some(ConversionFunction::YUVToRGBMatrixFilter(
                I420ToRGB24MatrixFilter,
            ))
        }
        (_, _, Format::Rgba | Format::Bgra, PixelFormat::Yuv422)
            if rgb.chroma_upsampling.bilinear_or_better_filter_allowed() =>
        {
            Some(ConversionFunction::YUVToRGBMatrixFilter(
                I422ToARGBMatrixFilter,
            ))
        }
        (_, _, Format::Rgba | Format::Bgra, PixelFormat::Yuv420)
            if rgb.chroma_upsampling.bilinear_or_better_filter_allowed() =>
        {
            Some(ConversionFunction::YUVToRGBMatrixFilter(
                I420ToARGBMatrixFilter,
            ))
        }

        (true, _, Format::Rgba | Format::Bgra, PixelFormat::Yuv444) => {
            Some(ConversionFunction::YUVAToRGBMatrix(I444AlphaToARGBMatrix))
        }
        (true, _, Format::Rgba | Format::Bgra, PixelFormat::Yuv422)
            if rgb.chroma_upsampling.nearest_neighbor_filter_allowed() =>
        {
            Some(ConversionFunction::YUVAToRGBMatrix(I422AlphaToARGBMatrix))
        }
        (true, _, Format::Rgba | Format::Bgra, PixelFormat::Yuv420)
            if rgb.chroma_upsampling.nearest_neighbor_filter_allowed() =>
        {
            Some(ConversionFunction::YUVAToRGBMatrix(I420AlphaToARGBMatrix))
        }

        (_, _, Format::Rgb | Format::Bgr, PixelFormat::Yuv444) => {
            Some(ConversionFunction::YUVToRGBMatrix(I444ToRGB24Matrix))
        }
        (_, _, Format::Rgb | Format::Bgr, PixelFormat::Yuv420)
            if rgb.chroma_upsampling.nearest_neighbor_filter_allowed() =>
        {
            Some(ConversionFunction::YUVToRGBMatrix(I420ToRGB24Matrix))
        }

        (_, _, Format::Rgba | Format::Bgra, PixelFormat::Yuv444) => {
            Some(ConversionFunction::YUVToRGBMatrix(I444ToARGBMatrix))
        }
        (_, _, Format::Rgba | Format::Bgra, PixelFormat::Yuv422)
            if rgb.chroma_upsampling.nearest_neighbor_filter_allowed() =>
        {
            Some(ConversionFunction::YUVToRGBMatrix(I422ToARGBMatrix))
        }
        (_, _, Format::Rgba | Format::Bgra, PixelFormat::Yuv420)
            if rgb.chroma_upsampling.nearest_neighbor_filter_allowed() =>
        {
            Some(ConversionFunction::YUVToRGBMatrix(I420ToARGBMatrix))
        }

        (_, _, Format::Argb | Format::Abgr, PixelFormat::Yuv422)
            if rgb.chroma_upsampling.nearest_neighbor_filter_allowed() =>
        {
            Some(ConversionFunction::YUVToRGBMatrix(I422ToRGBAMatrix))
        }
        (_, _, Format::Argb | Format::Abgr, PixelFormat::Yuv420)
            if rgb.chroma_upsampling.nearest_neighbor_filter_allowed() =>
        {
            Some(ConversionFunction::YUVToRGBMatrix(I420ToRGBAMatrix))
        }

        (_, _, Format::Rgb565, PixelFormat::Yuv422)
            if rgb.chroma_upsampling.nearest_neighbor_filter_allowed() =>
        {
            Some(ConversionFunction::YUVToRGBMatrix(I422ToRGB565Matrix))
        }
        (_, _, Format::Rgb565, PixelFormat::Yuv420)
            if rgb.chroma_upsampling.nearest_neighbor_filter_allowed() =>
        {
            Some(ConversionFunction::YUVToRGBMatrix(I420ToRGB565Matrix))
        }

        _ => None,
    }
}

fn validate_plane(image: &image::Image, plane: Plane) -> AvifResult<PlaneData> {
    let pd = image.plane_data(plane).ok_or(AvifError::ReformatFailed)?;
    if pd.width == 0 || pd.height == 0 {
        return AvifError::reformat_failed();
    }
    let min_row_bytes = checked_mul!(pd.width, pd.pixel_size)?;
    if pd.row_bytes < min_row_bytes {
        return AvifError::reformat_failed();
    }
    if image.depth > 8 {
        if pd.row_bytes % 2 != 0 {
            return AvifError::reformat_failed();
        }
        let _ = image
            .row16(plane, pd.height - 1)
            .map_err(|_| AvifError::ReformatFailed)?;
    } else {
        let _ = image
            .row(plane, pd.height - 1)
            .map_err(|_| AvifError::ReformatFailed)?;
    }
    Ok(pd)
}

fn validate_conversion_planes(
    image: &image::Image,
    conversion_function: &ConversionFunction,
) -> AvifResult<()> {
    match conversion_function {
        ConversionFunction::YUV400ToRGBMatrix(_) => {
            let _ = validate_plane(image, Plane::Y)?;
        }
        ConversionFunction::NVToARGBMatrix(_) | ConversionFunction::P010ToRGBMatrix(_, _) => {
            let _ = validate_plane(image, Plane::Y)?;
            let _ = validate_plane(image, Plane::U)?;
        }
        ConversionFunction::YUVToAB30Matrix(_, _)
        | ConversionFunction::YUVToRGBMatrixFilter(_)
        | ConversionFunction::YUVToRGBMatrix(_)
        | ConversionFunction::YUVToRGBMatrixFilterHighBitDepth(_)
        | ConversionFunction::YUVToRGBMatrixHighBitDepth(_) => {
            for plane in YUV_PLANES {
                let _ = validate_plane(image, plane)?;
            }
        }
        ConversionFunction::YUVAToRGBMatrixFilter(_)
        | ConversionFunction::YUVAToRGBMatrix(_)
        | ConversionFunction::YUVAToRGBMatrixFilterHighBitDepth(_)
        | ConversionFunction::YUVAToRGBMatrixHighBitDepth(_) => {
            for plane in ALL_PLANES {
                let _ = validate_plane(image, plane)?;
            }
        }
    }
    Ok(())
}

// Converts YUV(A) samples to RGB(A).
// Returns Ok(Some(true)) if the whole conversion was performed.
// Returns Ok(Some(false)) if only YUV was converted and alpha
// needs to be imported separately.
// Returns Ok(None) if the conversion is not implemented.
#[cfg_attr(feature = "disable_cfi", sanitize(cfi = "off"))]
pub(crate) fn yuv_to_rgb(image: &image::Image, rgb: &mut rgb::Image) -> AvifResult<Option<bool>> {
    if (rgb.depth != 8 && rgb.depth != 10) || !image.depth_valid() {
        return Ok(None); // Not implemented.
    }
    if rgb.depth == 10 && rgb.format != Format::Rgba1010102 {
        return Ok(None); // Not implemented.
    }
    if rgb.width != image.width || rgb.height != image.height || !rgb.has_valid_pixels() {
        return AvifError::reformat_failed();
    }
    let (matrix_yuv, matrix_yvu) = match find_constants(image) {
        Some((matrix_yuv, matrix_yvu)) => (matrix_yuv, matrix_yvu),
        None => return Ok(None), // Not implemented.
    };
    let alpha_preferred = rgb.has_alpha() && image.has_alpha();
    let conversion_function =
        match find_conversion_function(image.yuv_format, image.depth, rgb, alpha_preferred) {
            Some(conversion_function) => conversion_function,
            None => return Ok(None), // Not implemented.
        };
    validate_conversion_planes(image, &conversion_function)?;
    let is_yvu = matches!(rgb.format, Format::Rgb | Format::Rgba | Format::Argb);
    let matrix: *const YuvConstants = if is_yvu { matrix_yvu } else { matrix_yuv };
    let u_plane_index: usize = if is_yvu { 2 } else { 1 };
    let v_plane_index: usize = if is_yvu { 1 } else { 2 };
    let filter = if rgb.chroma_upsampling.bilinear_or_better_filter_allowed() {
        FilterMode_kFilterBilinear
    } else {
        FilterMode_kFilterNone
    };
    let mut plane_u8 = image.plane_ptrs();
    let plane_u16 = image.plane16_ptrs();
    let mut plane_row_bytes = image.plane_row_bytes()?;
    let rgb_row_bytes = i32_from_u32(rgb.row_bytes)?;
    let width = i32_from_u32(image.width)?;
    let height = i32_from_u32(image.height)?;
    let rgb_ptr = rgb.pixels_mut();
    if rgb_ptr.is_null() {
        return AvifError::reformat_failed();
    }
    let mut high_bd_matched = true;
    // Apply one of the high bitdepth functions if possible.
    let mut result: c_int = match conversion_function {
        ConversionFunction::P010ToRGBMatrix(func1, func2) => {
            let y_ptr = plane_u16[0];
            let y_stride = plane_row_bytes[0] / 2;
            let uv_ptr = plane_u16[1];
            let uv_stride = plane_row_bytes[1] / 2;
            // SAFETY: `validate_conversion_planes` verified that `y_ptr` and `uv_ptr` are
            // non-null, 2-byte aligned `u16` P010 planes covering `width x height` with valid
            // even byte strides; `rgb.has_valid_pixels()` verified that `rgb_ptr` is non-null,
            // uniquely borrowed via `&mut *rgb`, and large enough for `height` rows of
            // `rgb_row_bytes` bytes (4 bytes per pixel); `matrix` points to a `'static`
            // `YuvConstants`.
            let res = unsafe {
                func1(
                    y_ptr,
                    y_stride,
                    uv_ptr,
                    uv_stride,
                    rgb_ptr,
                    rgb_row_bytes,
                    matrix,
                    width,
                    height,
                )
            };
            if res == 0 {
                // libyuv explicitly supports in-place `src == dst` channel swizzling for
                // `AR30ToAB30` and `ARGBToABGR`.
                // SAFETY: `rgb_ptr` is non-null, uniquely borrowed via `&mut *rgb`, and valid for
                // reads and writes of `height` rows of `rgb_row_bytes` bytes (`>= width * 4`).
                unsafe {
                    func2(
                        rgb_ptr,
                        rgb_row_bytes,
                        rgb_ptr,
                        rgb_row_bytes,
                        width,
                        height,
                    )
                }
            } else {
                res
            }
        }
        ConversionFunction::YUVToAB30Matrix(func1, func2) => {
            let y_ptr = plane_u16[0];
            let y_stride = plane_row_bytes[0] / 2;
            let u_ptr = plane_u16[1];
            let u_stride = plane_row_bytes[1] / 2;
            let v_ptr = plane_u16[2];
            let v_stride = plane_row_bytes[2] / 2;
            // SAFETY: `validate_conversion_planes` verified that `y_ptr`, `u_ptr`, and `v_ptr` are
            // non-null, 2-byte aligned 10-bit `u16` planes with valid strides covering `width x
            // height`; `rgb.has_valid_pixels()` verified `rgb_ptr` is non-null, uniquely borrowed,
            // and valid for `height` rows of `rgb_row_bytes` (`>= width * 4` bytes); `matrix` is a
            // valid `'static` `YuvConstants` pointer.
            let res = unsafe {
                func1(
                    y_ptr,
                    y_stride,
                    u_ptr,
                    u_stride,
                    v_ptr,
                    v_stride,
                    rgb_ptr,
                    rgb_row_bytes,
                    matrix,
                    width,
                    height,
                )
            };
            if res == 0 {
                // libyuv explicitly supports in-place `src == dst` conversion for `AR30ToAB30`.
                // SAFETY: `rgb_ptr` is non-null, uniquely borrowed via `&mut *rgb`, and valid for
                // reads and writes of `height` rows of `rgb_row_bytes` bytes (`>= width * 4`).
                unsafe {
                    func2(
                        rgb_ptr,
                        rgb_row_bytes,
                        rgb_ptr,
                        rgb_row_bytes,
                        width,
                        height,
                    )
                }
            } else {
                res
            }
        }
        ConversionFunction::YUVToRGBMatrixFilterHighBitDepth(func) => {
            let y_ptr = plane_u16[0];
            let y_stride = plane_row_bytes[0] / 2;
            let u_ptr = plane_u16[u_plane_index];
            let u_stride = plane_row_bytes[u_plane_index] / 2;
            let v_ptr = plane_u16[v_plane_index];
            let v_stride = plane_row_bytes[v_plane_index] / 2;
            // SAFETY: `validate_conversion_planes` verified Y, U, and V are non-null, 2-byte
            // aligned `u16` planes with valid strides covering `width x height`; `rgb_ptr` is
            // non-null, uniquely borrowed, and valid for `height` rows of `rgb_row_bytes` bytes;
            // `matrix` points to a `'static` `YuvConstants`.
            unsafe {
                func(
                    y_ptr,
                    y_stride,
                    u_ptr,
                    u_stride,
                    v_ptr,
                    v_stride,
                    rgb_ptr,
                    rgb_row_bytes,
                    matrix,
                    width,
                    height,
                    filter,
                )
            }
        }
        ConversionFunction::YUVAToRGBMatrixFilterHighBitDepth(func) => {
            let y_ptr = plane_u16[0];
            let y_stride = plane_row_bytes[0] / 2;
            let u_ptr = plane_u16[u_plane_index];
            let u_stride = plane_row_bytes[u_plane_index] / 2;
            let v_ptr = plane_u16[v_plane_index];
            let v_stride = plane_row_bytes[v_plane_index] / 2;
            let a_ptr = plane_u16[3];
            let a_stride = plane_row_bytes[3] / 2;
            // SAFETY: `validate_conversion_planes` verified Y, U, V, and A are non-null, 2-byte
            // aligned `u16` planes with valid strides covering `width x height`; `rgb_ptr` is
            // non-null, uniquely borrowed, and valid for `height` rows of `rgb_row_bytes` bytes;
            // `matrix` points to a `'static` `YuvConstants`.
            unsafe {
                func(
                    y_ptr,
                    y_stride,
                    u_ptr,
                    u_stride,
                    v_ptr,
                    v_stride,
                    a_ptr,
                    a_stride,
                    rgb_ptr,
                    rgb_row_bytes,
                    matrix,
                    width,
                    height,
                    0, // attenuate
                    filter,
                )
            }
        }
        ConversionFunction::YUVToRGBMatrixHighBitDepth(func) => {
            let y_ptr = plane_u16[0];
            let y_stride = plane_row_bytes[0] / 2;
            let u_ptr = plane_u16[u_plane_index];
            let u_stride = plane_row_bytes[u_plane_index] / 2;
            let v_ptr = plane_u16[v_plane_index];
            let v_stride = plane_row_bytes[v_plane_index] / 2;
            // SAFETY: `validate_conversion_planes` verified Y, U, and V are non-null, 2-byte
            // aligned `u16` planes with valid strides covering `width x height`; `rgb_ptr` is
            // non-null, uniquely borrowed, and valid for `height` rows of `rgb_row_bytes` bytes;
            // `matrix` points to a `'static` `YuvConstants`.
            unsafe {
                func(
                    y_ptr,
                    y_stride,
                    u_ptr,
                    u_stride,
                    v_ptr,
                    v_stride,
                    rgb_ptr,
                    rgb_row_bytes,
                    matrix,
                    width,
                    height,
                )
            }
        }
        ConversionFunction::YUVAToRGBMatrixHighBitDepth(func) => {
            let y_ptr = plane_u16[0];
            let y_stride = plane_row_bytes[0] / 2;
            let u_ptr = plane_u16[u_plane_index];
            let u_stride = plane_row_bytes[u_plane_index] / 2;
            let v_ptr = plane_u16[v_plane_index];
            let v_stride = plane_row_bytes[v_plane_index] / 2;
            let a_ptr = plane_u16[3];
            let a_stride = plane_row_bytes[3] / 2;
            // SAFETY: `validate_conversion_planes` verified Y, U, V, and A are non-null, 2-byte
            // aligned `u16` planes with valid strides covering `width x height`; `rgb_ptr` is
            // non-null, uniquely borrowed, and valid for `height` rows of `rgb_row_bytes` bytes;
            // `matrix` points to a `'static` `YuvConstants`.
            unsafe {
                func(
                    y_ptr,
                    y_stride,
                    u_ptr,
                    u_stride,
                    v_ptr,
                    v_stride,
                    a_ptr,
                    a_stride,
                    rgb_ptr,
                    rgb_row_bytes,
                    matrix,
                    width,
                    height,
                    0, // attenuate
                )
            }
        }
        _ => {
            high_bd_matched = false;
            -1
        }
    };
    if high_bd_matched {
        return if result == 0 {
            Ok(Some(!image.has_alpha() || conversion_function.is_yuva()))
        } else {
            AvifError::reformat_failed()
        };
    }
    let mut image8 = image::Image::default();
    if image.depth > 8 {
        downshift_to_8bit(image, &mut image8, conversion_function.is_yuva())?;
        validate_conversion_planes(&image8, &conversion_function)?;
        plane_u8 = image8.plane_ptrs();
        plane_row_bytes = image8.plane_row_bytes()?;
    }
    result = match conversion_function {
        ConversionFunction::NVToARGBMatrix(func) => {
            let y_ptr = plane_u8[0];
            let y_stride = plane_row_bytes[0];
            let uv_ptr = plane_u8[1];
            let uv_stride = plane_row_bytes[1];
            // SAFETY: `validate_conversion_planes` verified `y_ptr` and `uv_ptr` are non-null 8-bit
            // luma and interleaved UV planes covering `width x height`; `rgb_ptr` is non-null,
            // uniquely borrowed, and valid for `height` rows of `rgb_row_bytes` bytes; `matrix`
            // points to a `'static` `YuvConstants`.
            unsafe {
                func(
                    y_ptr,
                    y_stride,
                    uv_ptr,
                    uv_stride,
                    rgb_ptr,
                    rgb_row_bytes,
                    matrix,
                    width,
                    height,
                )
            }
        }
        ConversionFunction::YUV400ToRGBMatrix(func) => {
            let y_ptr = plane_u8[0];
            let y_stride = plane_row_bytes[0];
            // SAFETY: `validate_conversion_planes` verified `y_ptr` is a non-null 8-bit luma plane
            // covering `width x height`; `rgb_ptr` is non-null, uniquely borrowed, and valid for
            // `height` rows of `rgb_row_bytes` bytes; `matrix` points to a `'static`
            // `YuvConstants`.
            unsafe {
                func(
                    y_ptr,
                    y_stride,
                    rgb_ptr,
                    rgb_row_bytes,
                    matrix,
                    width,
                    height,
                )
            }
        }
        ConversionFunction::YUVToRGBMatrixFilter(func) => {
            let y_ptr = plane_u8[0];
            let y_stride = plane_row_bytes[0];
            let u_ptr = plane_u8[u_plane_index];
            let u_stride = plane_row_bytes[u_plane_index];
            let v_ptr = plane_u8[v_plane_index];
            let v_stride = plane_row_bytes[v_plane_index];
            // SAFETY: `validate_conversion_planes` verified Y, U, and V are non-null 8-bit planes
            // covering `width x height` with valid strides; `rgb_ptr` is non-null, uniquely
            // borrowed, and valid for `height` rows of `rgb_row_bytes` bytes; `matrix` points to a
            // `'static` `YuvConstants`.
            unsafe {
                func(
                    y_ptr,
                    y_stride,
                    u_ptr,
                    u_stride,
                    v_ptr,
                    v_stride,
                    rgb_ptr,
                    rgb_row_bytes,
                    matrix,
                    width,
                    height,
                    filter,
                )
            }
        }
        ConversionFunction::YUVAToRGBMatrixFilter(func) => {
            let y_ptr = plane_u8[0];
            let y_stride = plane_row_bytes[0];
            let u_ptr = plane_u8[u_plane_index];
            let u_stride = plane_row_bytes[u_plane_index];
            let v_ptr = plane_u8[v_plane_index];
            let v_stride = plane_row_bytes[v_plane_index];
            let a_ptr = plane_u8[3];
            let a_stride = plane_row_bytes[3];
            // SAFETY: `validate_conversion_planes` verified Y, U, V, and A are non-null 8-bit
            // planes covering `width x height` with valid strides; `rgb_ptr` is non-null, uniquely
            // borrowed, and valid for `height` rows of `rgb_row_bytes` bytes; `matrix` points to a
            // `'static` `YuvConstants`.
            unsafe {
                func(
                    y_ptr,
                    y_stride,
                    u_ptr,
                    u_stride,
                    v_ptr,
                    v_stride,
                    a_ptr,
                    a_stride,
                    rgb_ptr,
                    rgb_row_bytes,
                    matrix,
                    width,
                    height,
                    0, // attenuate
                    filter,
                )
            }
        }
        ConversionFunction::YUVToRGBMatrix(func) => {
            let y_ptr = plane_u8[0];
            let y_stride = plane_row_bytes[0];
            let u_ptr = plane_u8[u_plane_index];
            let u_stride = plane_row_bytes[u_plane_index];
            let v_ptr = plane_u8[v_plane_index];
            let v_stride = plane_row_bytes[v_plane_index];
            // SAFETY: `validate_conversion_planes` verified Y, U, and V are non-null 8-bit planes
            // covering `width x height` with valid strides; `rgb_ptr` is non-null, uniquely
            // borrowed, and valid for `height` rows of `rgb_row_bytes` bytes; `matrix` points to a
            // `'static` `YuvConstants`.
            unsafe {
                func(
                    y_ptr,
                    y_stride,
                    u_ptr,
                    u_stride,
                    v_ptr,
                    v_stride,
                    rgb_ptr,
                    rgb_row_bytes,
                    matrix,
                    width,
                    height,
                )
            }
        }
        ConversionFunction::YUVAToRGBMatrix(func) => {
            let y_ptr = plane_u8[0];
            let y_stride = plane_row_bytes[0];
            let u_ptr = plane_u8[u_plane_index];
            let u_stride = plane_row_bytes[u_plane_index];
            let v_ptr = plane_u8[v_plane_index];
            let v_stride = plane_row_bytes[v_plane_index];
            let a_ptr = plane_u8[3];
            let a_stride = plane_row_bytes[3];
            // SAFETY: `validate_conversion_planes` verified Y, U, V, and A are non-null 8-bit
            // planes covering `width x height` with valid strides; `rgb_ptr` is non-null, uniquely
            // borrowed, and valid for `height` rows of `rgb_row_bytes` bytes; `matrix` points to a
            // `'static` `YuvConstants`.
            unsafe {
                func(
                    y_ptr,
                    y_stride,
                    u_ptr,
                    u_stride,
                    v_ptr,
                    v_stride,
                    a_ptr,
                    a_stride,
                    rgb_ptr,
                    rgb_row_bytes,
                    matrix,
                    width,
                    height,
                    0, // attenuate
                )
            }
        }
        _ => 0,
    };
    if result == 0 {
        Ok(Some(!image.has_alpha() || conversion_function.is_yuva()))
    } else {
        AvifError::reformat_failed()
    }
}

fn downshift_to_8bit(
    image: &image::Image,
    image8: &mut image::Image,
    alpha: bool,
) -> AvifResult<()> {
    if image.width == 0 || image.height == 0 || !matches!(image.depth, 10 | 12 | 16) {
        return AvifError::reformat_failed();
    }
    image8.width = image.width;
    image8.height = image.height;
    image8.depth = 8;
    image8.yuv_format = image.yuv_format;
    image8.allocate_planes(Category::Color)?;
    if alpha {
        image8.allocate_planes(Category::Alpha)?;
    }
    let scale = 1 << (24 - image.depth);
    for plane in ALL_PLANES {
        if (plane == Plane::A && !alpha) || image.width(plane) == 0 {
            continue;
        }
        let pd = validate_plane(image, plane)?;
        let pd8 = validate_plane(image8, plane)?;
        let source_ptr = image.planes[plane.as_usize()]
            .as_ref()
            .ok_or(AvifError::ReformatFailed)?
            .ptr16();
        let dst_ptr = image8.planes[plane.as_usize()]
            .as_mut()
            .ok_or(AvifError::ReformatFailed)?
            .ptr_mut();
        if source_ptr.is_null() || dst_ptr.is_null() {
            return AvifError::reformat_failed();
        }
        let src_stride = i32_from_u32(pd.row_bytes / 2)?;
        let dst_stride = i32_from_u32(pd8.row_bytes)?;
        let width = i32_from_u32(pd.width)?;
        let height = i32_from_u32(pd.height)?;
        // SAFETY: `validate_plane` and the null check above verified `source_ptr` is a non-null,
        // 2-byte aligned `u16` plane of `width x height` with `src_stride` `u16` elements per row,
        // and `dst_ptr` is a freshly allocated, disjoint `u8` buffer in `image8` of `width x
        // height` with `dst_stride` bytes per row.
        unsafe {
            Convert16To8Plane(
                source_ptr, src_stride, dst_ptr, dst_stride, scale, width, height,
            );
        }
    }
    Ok(())
}

pub(crate) fn process_alpha(rgb: &mut rgb::Image, multiply: bool) -> AvifResult<()> {
    if rgb.depth != 8 {
        return AvifError::not_implemented();
    }
    match rgb.format {
        Format::Rgba | Format::Bgra => {}
        _ => return AvifError::not_implemented(),
    }
    if !rgb.has_valid_pixels() {
        return AvifError::reformat_failed();
    }
    let pixels = rgb.pixels_mut();
    if pixels.is_null() {
        return AvifError::reformat_failed();
    }
    let row_bytes = i32_from_u32(rgb.row_bytes)?;
    let width = i32_from_u32(rgb.width)?;
    let height = i32_from_u32(rgb.height)?;
    let result = if multiply {
        // libyuv explicitly supports in-place `src_argb == dst_argb` for `ARGBAttenuate`.
        // SAFETY: `rgb.has_valid_pixels()` verified `pixels` is non-null, uniquely borrowed via
        // `&mut *rgb`, and valid for reads and writes of `height` rows of `row_bytes` bytes
        // (`>= width * 4`).
        unsafe { ARGBAttenuate(pixels, row_bytes, pixels, row_bytes, width, height) }
    } else {
        // libyuv explicitly supports in-place `src_argb == dst_argb` for `ARGBUnattenuate`.
        // SAFETY: `rgb.has_valid_pixels()` verified `pixels` is non-null, uniquely borrowed via
        // `&mut *rgb`, and valid for reads and writes of `height` rows of `row_bytes` bytes
        // (`>= width * 4`).
        unsafe { ARGBUnattenuate(pixels, row_bytes, pixels, row_bytes, width, height) }
    };
    if result == 0 {
        Ok(())
    } else {
        AvifError::reformat_failed()
    }
}

pub(crate) fn convert_to_half_float(rgb: &mut rgb::Image, scale: f32) -> AvifResult<Option<()>> {
    if !rgb.has_valid_pixels()
        || rgb.depth <= 8
        || matches!(rgb.format, Format::Rgb565 | Format::Rgba1010102)
    {
        return AvifError::invalid_argument();
    }
    let row_bytes = i32_from_u32(rgb.row_bytes)?;
    let width_channels = i32_from_u32(checked_mul!(rgb.width, rgb.channel_count())?)?;
    let height = i32_from_u32(rgb.height)?;
    let pixels = rgb
        .pixels
        .as_mut()
        .ok_or(AvifError::InvalidArgument)?
        .ptr16_mut();
    if pixels.is_null() {
        return AvifError::invalid_argument();
    }
    // Note: `HalfFloatPlane` takes `src_stride_y` and `dst_stride_y` in bytes, and supports
    // in-place conversion (`src_y == dst_y`).
    // SAFETY: `rgb.has_valid_pixels()`, `rgb.depth > 8`, and `ptr16_mut()` verified that `pixels`
    // is a non-null, 2-byte aligned, uniquely borrowed `u16` buffer with even `row_bytes` covering
    // `height` rows of `width_channels` `u16` elements.
    let res = unsafe {
        HalfFloatPlane(
            pixels,
            row_bytes,
            pixels,
            row_bytes,
            scale,
            width_channels,
            height,
        )
    };
    if res == 0 {
        Ok(Some(()))
    } else {
        AvifError::invalid_argument()
    }
}

#[rustfmt::skip]
type RGBToY = unsafe extern "C" fn(*const u8, c_int, *mut u8, c_int, c_int, c_int) -> c_int;
#[rustfmt::skip]
type ReorderARGB = unsafe extern "C" fn(*const u8, c_int, *mut u8, c_int, c_int, c_int) -> c_int;
#[rustfmt::skip]
type RGBToYUV = unsafe extern "C" fn(
    *const u8, c_int, *mut u8, c_int, *mut u8, c_int, *mut u8, c_int, c_int, c_int,
) -> c_int;

#[derive(Debug)]
enum RGBToYUVConversionFunction {
    RGBToY(RGBToY),
    RGBToYUV(RGBToYUV),
    ReorderThenYUV(ReorderARGB, RGBToYUV),
}

fn rgb_to_yuv_conversion_function(
    rgb: &rgb::Image,
    image: &mut image::Image,
) -> Option<RGBToYUVConversionFunction> {
    if image.depth != 8
        || rgb.depth != 8
        || !matches!(
            image.matrix_coefficients,
            MatrixCoefficients::Bt470bg | MatrixCoefficients::Bt601
        )
    {
        return None; // Not implemented.
    }
    // Matches libavif/src/reformat_libyuv.c lutRgbToY and lutRgbToYuv (LIBYUV_VERSION >= 1903).
    // Note (Yuv444, Full, Rgba|Argb|Bgr|Bgra|Abgr) intentionally return None to preserve exact
    // float32 rounding parity with libavif's avifImageRGBToYUV (no ABGRToJ444 entry in libavif).
    match (image.yuv_format, image.yuv_range, rgb.format) {
        (PixelFormat::Yuv400, YuvRange::Limited, Format::Bgra) => {
            Some(RGBToYUVConversionFunction::RGBToY(ARGBToI400))
        }
        (PixelFormat::Yuv400, YuvRange::Full, Format::Rgb) => {
            Some(RGBToYUVConversionFunction::RGBToY(RAWToJ400))
        }
        (PixelFormat::Yuv400, YuvRange::Full, Format::Rgba) => {
            Some(RGBToYUVConversionFunction::RGBToY(ABGRToJ400))
        }
        (PixelFormat::Yuv400, YuvRange::Full, Format::Bgr) => {
            Some(RGBToYUVConversionFunction::RGBToY(RGB24ToJ400))
        }
        (PixelFormat::Yuv400, YuvRange::Full, Format::Bgra) => {
            Some(RGBToYUVConversionFunction::RGBToY(ARGBToJ400))
        }
        (PixelFormat::Yuv400, YuvRange::Full, Format::Abgr) => {
            Some(RGBToYUVConversionFunction::RGBToY(RGBAToJ400))
        }
        // Limited range YUV444.
        (PixelFormat::Yuv444, YuvRange::Limited, Format::Rgb) if LIBYUV_VERSION >= 1903 => {
            Some(RGBToYUVConversionFunction::RGBToYUV(RAWToI444))
        }
        (PixelFormat::Yuv444, YuvRange::Limited, Format::Rgba) => Some(
            RGBToYUVConversionFunction::ReorderThenYUV(ABGRToARGB, ARGBToI444),
        ),
        (PixelFormat::Yuv444, YuvRange::Limited, Format::Argb) => Some(
            RGBToYUVConversionFunction::ReorderThenYUV(BGRAToARGB, ARGBToI444),
        ),
        (PixelFormat::Yuv444, YuvRange::Limited, Format::Bgr) => Some(
            RGBToYUVConversionFunction::ReorderThenYUV(RGB24ToARGB, ARGBToI444),
        ),
        (PixelFormat::Yuv444, YuvRange::Limited, Format::Bgra) => {
            Some(RGBToYUVConversionFunction::RGBToYUV(ARGBToI444))
        }
        (PixelFormat::Yuv444, YuvRange::Limited, Format::Abgr) => Some(
            RGBToYUVConversionFunction::ReorderThenYUV(RGBAToARGB, ARGBToI444),
        ),
        // Limited range YUV422.
        (PixelFormat::Yuv422, YuvRange::Limited, Format::Rgb) => Some(
            RGBToYUVConversionFunction::ReorderThenYUV(RAWToARGB, ARGBToI422),
        ),
        (PixelFormat::Yuv422, YuvRange::Limited, Format::Rgba) => Some(
            RGBToYUVConversionFunction::ReorderThenYUV(ABGRToARGB, ARGBToI422),
        ),
        (PixelFormat::Yuv422, YuvRange::Limited, Format::Argb) => Some(
            RGBToYUVConversionFunction::ReorderThenYUV(BGRAToARGB, ARGBToI422),
        ),
        (PixelFormat::Yuv422, YuvRange::Limited, Format::Bgr) => Some(
            RGBToYUVConversionFunction::ReorderThenYUV(RGB24ToARGB, ARGBToI422),
        ),
        (PixelFormat::Yuv422, YuvRange::Limited, Format::Bgra) => {
            Some(RGBToYUVConversionFunction::RGBToYUV(ARGBToI422))
        }
        (PixelFormat::Yuv422, YuvRange::Limited, Format::Abgr) => Some(
            RGBToYUVConversionFunction::ReorderThenYUV(RGBAToARGB, ARGBToI422),
        ),
        // Limited range YUV420.
        (PixelFormat::Yuv420, YuvRange::Limited, Format::Rgb) => {
            Some(RGBToYUVConversionFunction::RGBToYUV(RAWToI420))
        }
        (PixelFormat::Yuv420, YuvRange::Limited, Format::Rgba) => {
            Some(RGBToYUVConversionFunction::RGBToYUV(ABGRToI420))
        }
        (PixelFormat::Yuv420, YuvRange::Limited, Format::Argb) => {
            Some(RGBToYUVConversionFunction::RGBToYUV(BGRAToI420))
        }
        (PixelFormat::Yuv420, YuvRange::Limited, Format::Bgr) => {
            Some(RGBToYUVConversionFunction::RGBToYUV(RGB24ToI420))
        }
        (PixelFormat::Yuv420, YuvRange::Limited, Format::Bgra) => {
            Some(RGBToYUVConversionFunction::RGBToYUV(ARGBToI420))
        }
        (PixelFormat::Yuv420, YuvRange::Limited, Format::Abgr) => {
            Some(RGBToYUVConversionFunction::RGBToYUV(RGBAToI420))
        }
        // Full range YUV444.
        (PixelFormat::Yuv444, YuvRange::Full, Format::Rgb) if LIBYUV_VERSION >= 1902 => {
            Some(RGBToYUVConversionFunction::RGBToYUV(RAWToJ444))
        }
        // Full range YUV422.
        (PixelFormat::Yuv422, YuvRange::Full, Format::Rgb) => Some(
            RGBToYUVConversionFunction::ReorderThenYUV(RAWToARGB, ARGBToJ422),
        ),
        (PixelFormat::Yuv422, YuvRange::Full, Format::Rgba) => {
            Some(RGBToYUVConversionFunction::RGBToYUV(ABGRToJ422))
        }
        (PixelFormat::Yuv422, YuvRange::Full, Format::Argb) => Some(
            RGBToYUVConversionFunction::ReorderThenYUV(BGRAToARGB, ARGBToJ422),
        ),
        (PixelFormat::Yuv422, YuvRange::Full, Format::Bgr) => Some(
            RGBToYUVConversionFunction::ReorderThenYUV(RGB24ToARGB, ARGBToJ422),
        ),
        (PixelFormat::Yuv422, YuvRange::Full, Format::Bgra) => {
            Some(RGBToYUVConversionFunction::RGBToYUV(ARGBToJ422))
        }
        (PixelFormat::Yuv422, YuvRange::Full, Format::Abgr) => Some(
            RGBToYUVConversionFunction::ReorderThenYUV(RGBAToARGB, ARGBToJ422),
        ),
        // Full range YUV420.
        (PixelFormat::Yuv420, YuvRange::Full, Format::Rgb) => {
            Some(RGBToYUVConversionFunction::RGBToYUV(RAWToJ420))
        }
        (PixelFormat::Yuv420, YuvRange::Full, Format::Rgba) => {
            Some(RGBToYUVConversionFunction::RGBToYUV(ABGRToJ420))
        }
        (PixelFormat::Yuv420, YuvRange::Full, Format::Argb) => Some(
            RGBToYUVConversionFunction::ReorderThenYUV(BGRAToARGB, ARGBToJ420),
        ),
        (PixelFormat::Yuv420, YuvRange::Full, Format::Bgr) => {
            Some(RGBToYUVConversionFunction::RGBToYUV(RGB24ToJ420))
        }
        (PixelFormat::Yuv420, YuvRange::Full, Format::Bgra) => {
            Some(RGBToYUVConversionFunction::RGBToYUV(ARGBToJ420))
        }
        (PixelFormat::Yuv420, YuvRange::Full, Format::Abgr) => Some(
            RGBToYUVConversionFunction::ReorderThenYUV(RGBAToARGB, ARGBToJ420),
        ),
        _ => None, // Not implemented.
    }
}

/// # Safety
/// Caller must ensure `rgb_ptr` is valid for reading `height` rows with positive row stride
/// `rgb_row_bytes` (`>= width * channel_bytes`), `y_ptr`/`u_ptr`/`v_ptr` are valid, non-null,
/// disjoint buffers covering `width x height` (and subsampled chroma rows per `chroma_shift_y`)
/// with positive strides `y_stride`/`u_stride`/`v_stride`, and `temp_ptr` is valid for writing
/// `num_allocated_rows * stride_argb` disjoint bytes.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(feature = "disable_cfi", sanitize(cfi = "off"))]
unsafe fn reorder_argb_then_convert_to_yuv_loop(
    reorder: ReorderARGB,
    convert: RGBToYUV,
    rgb_ptr: *const u8,
    rgb_row_bytes: i32,
    y_ptr: *mut u8,
    y_stride: i32,
    u_ptr: *mut u8,
    u_stride: i32,
    v_ptr: *mut u8,
    v_stride: i32,
    chroma_shift_y: u32,
    width: i32,
    height: i32,
    stride_argb: i32,
    num_allocated_rows: i32,
    temp_ptr: *mut u8,
) -> c_int {
    let mut y = 0i32;
    while y < height {
        let num_rows = num_allocated_rows.min(height - y);
        let src_off = (y as usize) * (rgb_row_bytes as usize);
        let y_off = (y as usize) * (y_stride as usize);
        let uv_y = (y as usize) >> chroma_shift_y;
        let u_off = uv_y * (u_stride as usize);
        let v_off = uv_y * (v_stride as usize);
        // SAFETY: Caller guarantees `rgb_ptr`, `y_ptr`, `u_ptr`, and `v_ptr` span `height` (and
        // subsampled `uv_y`) rows with the given positive strides. All non-final chunks have
        // `num_rows` divisible by `min_num_rows`, advancing cleanly along chroma row boundaries.
        let (src_row, dst_y, dst_u, dst_v) = unsafe {
            (
                rgb_ptr.add(src_off),
                y_ptr.add(y_off),
                u_ptr.add(u_off),
                v_ptr.add(v_off),
            )
        };
        // SAFETY: `temp_ptr` has capacity `>= num_allocated_rows * stride_argb >= num_rows *
        // width * 4` writable bytes, and `src_row` has `num_rows` readable rows.
        let r1 = unsafe {
            reorder(
                src_row,
                rgb_row_bytes,
                temp_ptr,
                stride_argb,
                width,
                num_rows,
            )
        };
        if r1 != 0 {
            return r1;
        }
        // SAFETY: `reorder` initialized `num_rows * stride_argb` bytes in `temp_ptr`, and
        // `dst_y`, `dst_u`, `dst_v` are disjoint writable planes for `num_rows` rows.
        let r2 = unsafe {
            convert(
                temp_ptr,
                stride_argb,
                dst_y,
                y_stride,
                dst_u,
                u_stride,
                dst_v,
                v_stride,
                width,
                num_rows,
            )
        };
        if r2 != 0 {
            return r2;
        }
        y += num_rows;
    }
    0
}

/// # Safety
/// Same source/destination pointer and stride invariants as [`reorder_argb_then_convert_to_yuv`],
/// plus `alloc_bytes >= num_allocated_rows * stride_argb`.
#[cold]
#[allow(clippy::too_many_arguments)]
#[cfg_attr(feature = "disable_cfi", sanitize(cfi = "off"))]
unsafe fn reorder_argb_then_convert_to_yuv_heap(
    reorder: ReorderARGB,
    convert: RGBToYUV,
    rgb_ptr: *const u8,
    rgb_row_bytes: i32,
    y_ptr: *mut u8,
    y_stride: i32,
    u_ptr: *mut u8,
    u_stride: i32,
    v_ptr: *mut u8,
    v_stride: i32,
    chroma_shift_y: u32,
    width: i32,
    height: i32,
    stride_argb: i32,
    num_allocated_rows: i32,
    alloc_bytes: usize,
) -> AvifResult<c_int> {
    let mut heap_buf: Vec<u8> = create_vec_exact(alloc_bytes)?;
    let temp_ptr = heap_buf.spare_capacity_mut().as_mut_ptr().cast::<u8>();
    // SAFETY: `heap_buf` remains alive across the loop and owns `alloc_bytes` spare bytes;
    // remaining pointer invariants are forwarded from the caller.
    let rc = unsafe {
        reorder_argb_then_convert_to_yuv_loop(
            reorder,
            convert,
            rgb_ptr,
            rgb_row_bytes,
            y_ptr,
            y_stride,
            u_ptr,
            u_stride,
            v_ptr,
            v_stride,
            chroma_shift_y,
            width,
            height,
            stride_argb,
            num_allocated_rows,
            temp_ptr,
        )
    };
    Ok(rc)
}

/// # Safety
/// Caller must ensure `rgb_ptr` is valid for reading `height` rows with positive row stride
/// `rgb_row_bytes` (`>= width * channel_bytes`), and `y_ptr`, `u_ptr`, `v_ptr` are valid,
/// non-null, disjoint writable buffers covering `width x height` (and subsampled chroma rows per
/// `chroma_shift_y`) with positive strides `y_stride`, `u_stride`, and `v_stride`.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(feature = "disable_cfi", sanitize(cfi = "off"))]
unsafe fn reorder_argb_then_convert_to_yuv(
    reorder: ReorderARGB,
    convert: RGBToYUV,
    rgb_ptr: *const u8,
    rgb_row_bytes: i32,
    y_ptr: *mut u8,
    y_stride: i32,
    u_ptr: *mut u8,
    u_stride: i32,
    v_ptr: *mut u8,
    v_stride: i32,
    chroma_shift_y: u32,
    width: i32,
    height: i32,
) -> AvifResult<c_int> {
    if width <= 0 || height <= 0 {
        return AvifError::reformat_failed();
    }
    let min_num_rows: i32 = if chroma_shift_y == 1 { 2 } else { 1 };
    let stride_argb_i64 = (width as i64) * 4;
    if stride_argb_i64 > i32::MAX as i64 {
        return AvifError::reformat_failed();
    }
    let stride_argb = stride_argb_i64 as i32;
    const SOFT_ALLOCATION_LIMIT: usize = 16384;
    let num_allocated_rows: i32 =
        if height == 1 || stride_argb_i64 * (height as i64) <= SOFT_ALLOCATION_LIMIT as i64 {
            height
        } else {
            let min_chunk_bytes = stride_argb_i64 * (min_num_rows as i64);
            if min_chunk_bytes > i32::MAX as i64 {
                return AvifError::reformat_failed();
            }
            let mult = ((SOFT_ALLOCATION_LIMIT as i64) / min_chunk_bytes).max(1) as i32;
            mult.saturating_mul(min_num_rows)
        };
    let alloc_bytes = usize_from_u64((num_allocated_rows as u64) * (stride_argb as u64))?;
    if alloc_bytes > SOFT_ALLOCATION_LIMIT {
        // SAFETY: Forwarded from caller.
        return unsafe {
            reorder_argb_then_convert_to_yuv_heap(
                reorder,
                convert,
                rgb_ptr,
                rgb_row_bytes,
                y_ptr,
                y_stride,
                u_ptr,
                u_stride,
                v_ptr,
                v_stride,
                chroma_shift_y,
                width,
                height,
                stride_argb,
                num_allocated_rows,
                alloc_bytes,
            )
        };
    }
    let mut stack_buf = [std::mem::MaybeUninit::<u8>::uninit(); SOFT_ALLOCATION_LIMIT];
    let temp_ptr = stack_buf.as_mut_ptr().cast::<u8>();
    // SAFETY: `stack_buf` owns `SOFT_ALLOCATION_LIMIT >= alloc_bytes` writable bytes; remaining
    // pointer invariants are forwarded from the caller.
    let rc = unsafe {
        reorder_argb_then_convert_to_yuv_loop(
            reorder,
            convert,
            rgb_ptr,
            rgb_row_bytes,
            y_ptr,
            y_stride,
            u_ptr,
            u_stride,
            v_ptr,
            v_stride,
            chroma_shift_y,
            width,
            height,
            stride_argb,
            num_allocated_rows,
            temp_ptr,
        )
    };
    Ok(rc)
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct Raw444Stripe {
    src_ptr: *const u8,
    src_stride: usize,
    y_ptr: *mut u8,
    y_stride: usize,
    u_ptr: *mut u8,
    u_stride: usize,
    v_ptr: *mut u8,
    v_stride: usize,
    width: usize,
    height: usize,
    is_bgr: bool,
    is_full: bool,
}

// SAFETY: Each `Raw444Stripe` owns writes to a disjoint contiguous row range of the destination
// `UninitPlane8` buffers while reading from the shared immutable `rgb` source buffer, and all
// pointers remain valid for the duration of the scoped thread execution.
#[cfg(target_arch = "x86_64")]
unsafe impl Send for Raw444Stripe {}

/// # Safety
/// Caller must verify runtime `avx2` support, that `rgb_ptr` is valid for reading `height` rows
/// of `width * 3` bytes with row stride `src_stride`, and that `y_ptr`, `u_ptr`, and `v_ptr` are
/// disjoint writable buffers covering `height` rows of `width` bytes with strides `y_stride`,
/// `u_stride`, and `v_stride` (`>= width`).
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
unsafe fn run_raw_to_yuv444_avx2_stripes(
    rgb_ptr: *const u8,
    src_stride: usize,
    y_ptr: *mut u8,
    y_stride: usize,
    u_ptr: *mut u8,
    u_stride: usize,
    v_ptr: *mut u8,
    v_stride: usize,
    width: usize,
    height: usize,
    is_bgr: bool,
    is_full: bool,
    workers: usize,
) -> AvifResult<()> {
    let workers = workers.clamp(1, rgb::MAX_RGB_TO_YUV_THREADS).min(height);
    if workers <= 1 {
        // SAFETY: Caller verified AVX2 support and valid buffer dimensions/strides.
        unsafe {
            raw_to_yuv444_avx2(
                rgb_ptr, src_stride, y_ptr, y_stride, u_ptr, u_stride, v_ptr, v_stride, width,
                height, is_bgr, is_full,
            );
        }
        return Ok(());
    }
    let base_rows = height / workers;
    let extra_rows = height % workers;
    let mut stripes: Vec<Raw444Stripe> = create_vec_exact(workers)?;
    let mut start_row = 0usize;
    for idx in 0..workers {
        let stripe_h = base_rows + usize::from(idx < extra_rows);
        // SAFETY: `start_row < height` and caller guarantees all source/destination buffers span
        // at least `height` rows with the given strides.
        let (s_ptr, dy_ptr, du_ptr, dv_ptr) = unsafe {
            (
                rgb_ptr.add(start_row * src_stride),
                y_ptr.add(start_row * y_stride),
                u_ptr.add(start_row * u_stride),
                v_ptr.add(start_row * v_stride),
            )
        };
        stripes.try_push(Raw444Stripe {
            src_ptr: s_ptr,
            src_stride,
            y_ptr: dy_ptr,
            y_stride,
            u_ptr: du_ptr,
            u_stride,
            v_ptr: dv_ptr,
            v_stride,
            width,
            height: stripe_h,
            is_bgr,
            is_full,
        })?;
        start_row += stripe_h;
    }
    let mut iter = stripes.into_iter();
    let first = iter.next().ok_or(AvifError::ReformatFailed)?;
    let mut worker_panic = false;
    std::thread::scope(|scope| -> AvifResult<()> {
        let mut handles = create_vec_exact(workers - 1)?;
        for stripe in iter {
            let handle = std::thread::Builder::new()
                .spawn_scoped(scope, move || {
                    let stripe = stripe;
                    // SAFETY: AVX2 availability verified by caller; stripe covers disjoint rows.
                    unsafe {
                        raw_to_yuv444_avx2(
                            stripe.src_ptr,
                            stripe.src_stride,
                            stripe.y_ptr,
                            stripe.y_stride,
                            stripe.u_ptr,
                            stripe.u_stride,
                            stripe.v_ptr,
                            stripe.v_stride,
                            stripe.width,
                            stripe.height,
                            stripe.is_bgr,
                            stripe.is_full,
                        );
                    }
                })
                .map_err(|_| AvifError::ReformatFailed)?;
            handles.try_push(handle)?;
        }
        // SAFETY: AVX2 availability verified by caller; `first` covers disjoint `[0..stripe_h)`.
        unsafe {
            raw_to_yuv444_avx2(
                first.src_ptr,
                first.src_stride,
                first.y_ptr,
                first.y_stride,
                first.u_ptr,
                first.u_stride,
                first.v_ptr,
                first.v_stride,
                first.width,
                first.height,
                first.is_bgr,
                first.is_full,
            );
        }
        for handle in handles {
            if handle.join().is_err() {
                worker_panic = true;
            }
        }
        Ok(())
    })?;
    if worker_panic {
        return AvifError::reformat_failed();
    }
    Ok(())
}

/// # Safety
/// Caller must ensure `uninit_y`, `uninit_u`, and `uninit_v` are fully initialized for `image`.
unsafe fn finish_yuv8_planes(
    image: &mut image::Image,
    uninit_y: crate::image::UninitPlane8,
    uninit_u: crate::image::UninitPlane8,
    uninit_v: crate::image::UninitPlane8,
) -> AvifResult<()> {
    // SAFETY: Forwarded from caller.
    unsafe { uninit_y.finish_into(image, Plane::Y, 0)? };
    // SAFETY: Forwarded from caller.
    if let Err(err) = unsafe { uninit_u.finish_into(image, Plane::U, 0) } {
        image.free_planes(&[Plane::Y]);
        return Err(err);
    }
    // SAFETY: Forwarded from caller.
    if let Err(err) = unsafe { uninit_v.finish_into(image, Plane::V, 0) } {
        image.free_planes(&[Plane::Y, Plane::U]);
        return Err(err);
    }
    Ok(())
}

#[cfg_attr(feature = "disable_cfi", sanitize(cfi = "off"))]
pub(crate) fn rgb_to_yuv_with_threads(
    rgb: &rgb::Image,
    image: &mut image::Image,
    threads: u32,
) -> AvifResult<Option<()>> {
    let conversion_function = match rgb_to_yuv_conversion_function(rgb, image) {
        Some(conversion_function) => conversion_function,
        None => return Ok(None), // Not implemented.
    };
    if rgb.width != image.width || rgb.height != image.height || !rgb.has_valid_pixels() {
        return AvifError::reformat_failed();
    }
    let rgb_ptr = rgb.pixels();
    if rgb_ptr.is_null() {
        return AvifError::reformat_failed();
    }
    let chroma_shift_y = image.yuv_format.chroma_shift_y();
    let width = i32_from_u32(image.width)?;
    let height = i32_from_u32(image.height)?;
    let rgb_row_bytes = i32_from_u32(rgb.row_bytes)?;
    #[cfg(target_arch = "x86_64")]
    if image.yuv_format == PixelFormat::Yuv444
        && matches!(rgb.format, Format::Rgb | Format::Bgr)
        && std::arch::is_x86_feature_detected!("avx2")
    {
        let width_usize = usize_from_u32(image.width)?;
        let height_usize = usize_from_u32(image.height)?;
        let min_src_row_bytes = checked_mul!(width_usize, 3)?;
        let src_stride = usize_from_u32(rgb.row_bytes)?;
        if src_stride < min_src_row_bytes {
            return AvifError::reformat_failed();
        }
        let mut uninit_y = crate::image::UninitPlane8::for_image(image, Plane::Y)?;
        let mut uninit_u = crate::image::UninitPlane8::for_image(image, Plane::U)?;
        let mut uninit_v = crate::image::UninitPlane8::for_image(image, Plane::V)?;
        let y_ptr = uninit_y.as_mut_ptr();
        let u_ptr = uninit_u.as_mut_ptr();
        let v_ptr = uninit_v.as_mut_ptr();
        let y_stride = uninit_y.stride_elems();
        let u_stride = uninit_u.stride_elems();
        let v_stride = uninit_v.stride_elems();
        let is_bgr = rgb.format == Format::Bgr;
        let is_full = image.yuv_range == YuvRange::Full;
        let max_workers = (threads as usize).min(rgb::MAX_RGB_TO_YUV_THREADS);
        let row_worker_cap = height_usize / rgb::MIN_MT_ROWS_PER_WORKER;
        let workers = if threads > 1
            && width_usize.saturating_mul(height_usize) >= rgb::MIN_MT_PIXELS_LIBYUV_444
            && row_worker_cap >= 2
        {
            max_workers.min(row_worker_cap)
        } else {
            1
        };
        // SAFETY: `avx2` support was verified above, `rgb.has_valid_pixels()` checked `rgb_ptr`
        // across `height_usize` rows, and `uninit_y/u/v` own disjoint writable allocations.
        unsafe {
            run_raw_to_yuv444_avx2_stripes(
                rgb_ptr,
                src_stride,
                y_ptr,
                y_stride,
                u_ptr,
                u_stride,
                v_ptr,
                v_stride,
                width_usize,
                height_usize,
                is_bgr,
                is_full,
                workers,
            )?;
            finish_yuv8_planes(image, uninit_y, uninit_u, uninit_v)?;
        }
        return Ok(Some(()));
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = threads;
    match conversion_function {
        RGBToYUVConversionFunction::RGBToY(func) => {
            let mut uninit_y = crate::image::UninitPlane8::for_image(image, Plane::Y)?;
            let y_ptr = uninit_y.as_mut_ptr();
            let y_stride = uninit_y.row_bytes_i32()?;
            // SAFETY: `rgb.has_valid_pixels()` verified `rgb_ptr` is non-null and valid for reads
            // of `height` rows of `rgb_row_bytes` bytes; `uninit_y` owns a non-null, disjoint
            // allocation covering `round2(width) x round2(height)` bytes with `y_stride` row bytes.
            let result = unsafe { func(rgb_ptr, rgb_row_bytes, y_ptr, y_stride, width, height) };
            if result == 0 {
                // SAFETY: `func` initialized all `width x height` Y samples on success.
                unsafe { uninit_y.finish_into(image, Plane::Y, 0)? };
                image.free_planes(&[Plane::U, Plane::V]);
                Ok(Some(()))
            } else {
                AvifError::reformat_failed()
            }
        }
        RGBToYUVConversionFunction::RGBToYUV(func) => {
            let mut uninit_y = crate::image::UninitPlane8::for_image(image, Plane::Y)?;
            let mut uninit_u = crate::image::UninitPlane8::for_image(image, Plane::U)?;
            let mut uninit_v = crate::image::UninitPlane8::for_image(image, Plane::V)?;
            let y_ptr = uninit_y.as_mut_ptr();
            let u_ptr = uninit_u.as_mut_ptr();
            let v_ptr = uninit_v.as_mut_ptr();
            let y_stride = uninit_y.row_bytes_i32()?;
            let u_stride = uninit_u.row_bytes_i32()?;
            let v_stride = uninit_v.row_bytes_i32()?;
            // SAFETY: `rgb.has_valid_pixels()` verified `rgb_ptr` is non-null and valid for reads
            // of `height` rows of `rgb_row_bytes` bytes; `uninit_y`, `uninit_u`, and `uninit_v`
            // own disjoint 8-bit buffers covering `width x height` and subsampled chroma planes.
            let result = unsafe {
                func(
                    rgb_ptr,
                    rgb_row_bytes,
                    y_ptr,
                    y_stride,
                    u_ptr,
                    u_stride,
                    v_ptr,
                    v_stride,
                    width,
                    height,
                )
            };
            if result == 0 {
                // SAFETY: `func` initialized every logical `(row, col)` sample in Y, U, and V.
                unsafe { finish_yuv8_planes(image, uninit_y, uninit_u, uninit_v)? };
                Ok(Some(()))
            } else {
                AvifError::reformat_failed()
            }
        }
        RGBToYUVConversionFunction::ReorderThenYUV(reorder, convert) => {
            let mut uninit_y = crate::image::UninitPlane8::for_image(image, Plane::Y)?;
            let mut uninit_u = crate::image::UninitPlane8::for_image(image, Plane::U)?;
            let mut uninit_v = crate::image::UninitPlane8::for_image(image, Plane::V)?;
            let y_ptr = uninit_y.as_mut_ptr();
            let u_ptr = uninit_u.as_mut_ptr();
            let v_ptr = uninit_v.as_mut_ptr();
            let y_stride = uninit_y.row_bytes_i32()?;
            let u_stride = uninit_u.row_bytes_i32()?;
            let v_stride = uninit_v.row_bytes_i32()?;
            // SAFETY: `rgb.has_valid_pixels()` verified `rgb_ptr` and positive `rgb_row_bytes`
            // across `width x height`; `uninit_y/u/v` own disjoint writable allocations.
            let result = unsafe {
                reorder_argb_then_convert_to_yuv(
                    reorder,
                    convert,
                    rgb_ptr,
                    rgb_row_bytes,
                    y_ptr,
                    y_stride,
                    u_ptr,
                    u_stride,
                    v_ptr,
                    v_stride,
                    chroma_shift_y,
                    width,
                    height,
                )?
            };
            if result == 0 {
                // SAFETY: `reorder_argb_then_convert_to_yuv` initialized all logical Y/U/V samples.
                unsafe { finish_yuv8_planes(image, uninit_y, uninit_u, uninit_v)? };
                Ok(Some(()))
            } else {
                AvifError::reformat_failed()
            }
        }
    }
}

/// # Safety
/// Caller must verify `avx2` support, valid reads of `96` bytes from `src_raw`, and valid
/// writes of `32` bytes to `dst_y`, `dst_u`, and `dst_v`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn raw_to_yuv444_block32_avx2(
    src_raw: *const u8,
    dst_y: *mut u8,
    dst_u: *mut u8,
    dst_v: *mut u8,
    v_shuf_bg: std::arch::x86_64::__m256i,
    v_shuf_r: std::arch::x86_64::__m256i,
    v_bg_to_y: std::arch::x86_64::__m256i,
    v_r_to_y: std::arch::x86_64::__m256i,
    v_bg_to_u: std::arch::x86_64::__m256i,
    v_bg_to_v: std::arch::x86_64::__m256i,
    v_r_to_uv: std::arch::x86_64::__m256i,
    v_round_y: std::arch::x86_64::__m256i,
    v_80: std::arch::x86_64::__m256i,
    v_8000: std::arch::x86_64::__m256i,
) {
    use std::arch::x86_64::*;
    // SAFETY: Caller contract guarantees `src_raw` is valid for 96-byte reads (`0..96`).
    let (l0, h0, l1, h1) = unsafe {
        (
            _mm_loadu_si128(src_raw.cast()),
            _mm_loadu_si128(src_raw.add(48).cast()),
            _mm_loadu_si128(src_raw.add(8).cast()),
            _mm_loadu_si128(src_raw.add(56).cast()),
        )
    };
    let p0 = _mm256_inserti128_si256(_mm256_castsi128_si256(l0), h0, 1);
    let p1 = _mm256_inserti128_si256(_mm256_castsi128_si256(l1), h1, 1);

    let bg0 = _mm256_blend_epi32(
        _mm256_shuffle_epi8(p0, v_shuf_bg),
        _mm256_shuffle_epi8(p1, v_shuf_bg),
        0xcc,
    );
    let r0 = _mm256_blend_epi32(
        _mm256_shuffle_epi8(p0, v_shuf_r),
        _mm256_shuffle_epi8(p1, v_shuf_r),
        0xcc,
    );

    let u0 = _mm256_add_epi16(
        _mm256_maddubs_epi16(bg0, v_bg_to_u),
        _mm256_maddubs_epi16(r0, v_r_to_uv),
    );
    let u0 = _mm256_srli_epi16::<8>(_mm256_sub_epi16(v_8000, u0));

    let y0 = _mm256_add_epi16(
        _mm256_maddubs_epi16(v_bg_to_y, _mm256_sub_epi8(bg0, v_80)),
        _mm256_maddubs_epi16(r0, v_r_to_y),
    );
    let y0 = _mm256_srli_epi16::<8>(_mm256_add_epi16(y0, v_round_y));

    let v0 = _mm256_add_epi16(
        _mm256_maddubs_epi16(bg0, v_bg_to_v),
        _mm256_maddubs_epi16(_mm256_slli_epi16::<8>(r0), v_r_to_uv),
    );
    let v0 = _mm256_srli_epi16::<8>(_mm256_sub_epi16(v_8000, v0));

    // SAFETY: Caller contract guarantees `src_raw` is valid for 96-byte reads (`0..96`).
    let (l2, h2, l3, h3) = unsafe {
        (
            _mm_loadu_si128(src_raw.add(24).cast()),
            _mm_loadu_si128(src_raw.add(72).cast()),
            _mm_loadu_si128(src_raw.add(32).cast()),
            _mm_loadu_si128(src_raw.add(80).cast()),
        )
    };
    let p2 = _mm256_inserti128_si256(_mm256_castsi128_si256(l2), h2, 1);
    let p3 = _mm256_inserti128_si256(_mm256_castsi128_si256(l3), h3, 1);

    let bg1 = _mm256_blend_epi32(
        _mm256_shuffle_epi8(p2, v_shuf_bg),
        _mm256_shuffle_epi8(p3, v_shuf_bg),
        0xcc,
    );
    let r1 = _mm256_blend_epi32(
        _mm256_shuffle_epi8(p2, v_shuf_r),
        _mm256_shuffle_epi8(p3, v_shuf_r),
        0xcc,
    );

    let u1 = _mm256_add_epi16(
        _mm256_maddubs_epi16(bg1, v_bg_to_u),
        _mm256_maddubs_epi16(r1, v_r_to_uv),
    );
    let u1 = _mm256_srli_epi16::<8>(_mm256_sub_epi16(v_8000, u1));
    let u_packed = _mm256_packus_epi16(u0, u1);
    // SAFETY: Caller contract guarantees `dst_u` is valid for a 32-byte unaligned write.
    unsafe { _mm256_storeu_si256(dst_u.cast(), u_packed) };

    let y1 = _mm256_add_epi16(
        _mm256_maddubs_epi16(v_bg_to_y, _mm256_sub_epi8(bg1, v_80)),
        _mm256_maddubs_epi16(r1, v_r_to_y),
    );
    let y1 = _mm256_srli_epi16::<8>(_mm256_add_epi16(y1, v_round_y));
    let y_packed = _mm256_packus_epi16(y0, y1);
    // SAFETY: Caller contract guarantees `dst_y` is valid for a 32-byte unaligned write.
    unsafe { _mm256_storeu_si256(dst_y.cast(), y_packed) };

    let v1 = _mm256_add_epi16(
        _mm256_maddubs_epi16(bg1, v_bg_to_v),
        _mm256_maddubs_epi16(_mm256_slli_epi16::<8>(r1), v_r_to_uv),
    );
    let v1 = _mm256_srli_epi16::<8>(_mm256_sub_epi16(v_8000, v1));
    let v_packed = _mm256_packus_epi16(v0, v1);
    // SAFETY: Caller contract guarantees `dst_v` is valid for a 32-byte unaligned write.
    unsafe { _mm256_storeu_si256(dst_v.cast(), v_packed) };
}

/// # Safety
/// Caller must verify `avx2` support, valid reads of `height` rows of `width * 3` bytes with row
/// stride `src_stride` from `src_raw`, and valid disjoint writes of `height` rows of `width`
/// bytes with row strides `dst_stride_y/u/v` into `dst_y`, `dst_u`, and `dst_v`.
/// Note: `(is_bgr, is_full) == (true, true)` is supported here for symmetry and unit testing,
/// while `rgb_to_yuv_conversion_function` leaves `(Yuv444, Full, Bgr)` as `None` to match
/// `libavif` (which has no `RGB24ToJ444`/`ARGBToJ444` entry).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn raw_to_yuv444_avx2(
    src_raw: *const u8,
    src_stride: usize,
    dst_y: *mut u8,
    dst_stride_y: usize,
    dst_u: *mut u8,
    dst_stride_u: usize,
    dst_v: *mut u8,
    dst_stride_v: usize,
    width: usize,
    height: usize,
    is_bgr: bool,
    is_full: bool,
) {
    use std::arch::x86_64::*;
    let (width, height) = if height > 1
        && width.checked_mul(3) == Some(src_stride)
        && dst_stride_y == width
        && dst_stride_u == width
        && dst_stride_v == width
    {
        match width.checked_mul(height) {
            Some(total_px) if total_px.checked_mul(3).is_some() => (total_px, 1),
            _ => (width, height),
        }
    } else {
        (width, height)
    };

    let v_shuf_bg = if is_bgr {
        _mm256_setr_epi8(
            0, 1, 3, 4, 6, 7, 9, 10, 4, 5, 7, 8, 10, 11, 13, 14, 0, 1, 3, 4, 6, 7, 9, 10, 4, 5, 7,
            8, 10, 11, 13, 14,
        )
    } else {
        _mm256_setr_epi8(
            2, 1, 5, 4, 8, 7, 11, 10, 6, 5, 9, 8, 12, 11, 15, 14, 2, 1, 5, 4, 8, 7, 11, 10, 6, 5,
            9, 8, 12, 11, 15, 14,
        )
    };
    let v_shuf_r = if is_bgr {
        _mm256_setr_epi8(
            2, -1, 5, -1, 8, -1, 11, -1, 6, -1, 9, -1, 12, -1, 15, -1, 2, -1, 5, -1, 8, -1, 11, -1,
            6, -1, 9, -1, 12, -1, 15, -1,
        )
    } else {
        _mm256_setr_epi8(
            0, -1, 3, -1, 6, -1, 9, -1, 4, -1, 7, -1, 10, -1, 13, -1, 0, -1, 3, -1, 6, -1, 9, -1,
            4, -1, 7, -1, 10, -1, 13, -1,
        )
    };

    // Coefficients matching libyuv kArgbJPEGConstants (Full) and kArgbI601Constants (Limited).
    let (by, gy, ry, ay, neg_bu, neg_gu, neg_ru, neg_bv, neg_gv, neg_rv): (
        u16,
        u16,
        u16,
        i16,
        u16,
        u16,
        u16,
        u16,
        u16,
        u16,
    ) = if is_full {
        (
            29,
            150,
            77,
            128,
            (-128i8) as u8 as u16,
            85,
            43,
            21,
            107,
            (-128i8) as u8 as u16,
        )
    } else {
        (
            25,
            129,
            66,
            4224,
            (-112i8) as u8 as u16,
            74,
            38,
            18,
            94,
            (-112i8) as u8 as u16,
        )
    };

    let v_bg_to_y = _mm256_set1_epi16((by | (gy << 8)) as i16);
    let v_r_to_y = _mm256_set1_epi16(ry as i16);
    let v_bg_to_u = _mm256_set1_epi16((neg_bu | (neg_gu << 8)) as i16);
    let v_bg_to_v = _mm256_set1_epi16((neg_bv | (neg_gv << 8)) as i16);
    let v_r_to_uv = _mm256_set1_epi16((neg_ru | (neg_rv << 8)) as i16);
    let v_add_y = _mm256_set1_epi16(ay);
    let v_80 = _mm256_set1_epi8(0x80u8 as i8);
    let v_8000 = _mm256_set1_epi16(0x8000u16 as i16);
    let v_bias = _mm256_maddubs_epi16(v_bg_to_y, v_80);
    let v_round_y = _mm256_sub_epi16(v_add_y, v_bias);

    let full_px = width & !31usize;
    let rem_px = width & 31usize;
    let mut tail_raw = [0u8; 96];
    let mut tail_y = [0u8; 32];
    let mut tail_u = [0u8; 32];
    let mut tail_v = [0u8; 32];

    for row_idx in 0..height {
        // SAFETY: Caller guarantees `row_idx < height` and strides permit indexing each row.
        let (row_src, row_y, row_u, row_v) = unsafe {
            (
                src_raw.add(row_idx * src_stride),
                dst_y.add(row_idx * dst_stride_y),
                dst_u.add(row_idx * dst_stride_u),
                dst_v.add(row_idx * dst_stride_v),
            )
        };
        let mut x = 0usize;
        while x < full_px {
            // SAFETY: `x + 32 <= width`, so `row_src + x * 3` has >= 96 bytes and destination
            // pointers have >= 32 bytes remaining in the active row.
            unsafe {
                raw_to_yuv444_block32_avx2(
                    row_src.add(x * 3),
                    row_y.add(x),
                    row_u.add(x),
                    row_v.add(x),
                    v_shuf_bg,
                    v_shuf_r,
                    v_bg_to_y,
                    v_r_to_y,
                    v_bg_to_u,
                    v_bg_to_v,
                    v_r_to_uv,
                    v_round_y,
                    v_80,
                    v_8000,
                );
            }
            x += 32;
        }
        if rem_px > 0 {
            // SAFETY: `rem_px < 32`, so `rem_px * 3 <= 93 < 96`; copy the active tail bytes into
            // the 96-byte scratch buffer `tail_raw`, run the 32-px kernel into scratch outputs,
            // and copy the first `rem_px` valid output bytes to each destination row.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    row_src.add(full_px * 3),
                    tail_raw.as_mut_ptr(),
                    rem_px * 3,
                );
                raw_to_yuv444_block32_avx2(
                    tail_raw.as_ptr(),
                    tail_y.as_mut_ptr(),
                    tail_u.as_mut_ptr(),
                    tail_v.as_mut_ptr(),
                    v_shuf_bg,
                    v_shuf_r,
                    v_bg_to_y,
                    v_r_to_y,
                    v_bg_to_u,
                    v_bg_to_v,
                    v_r_to_uv,
                    v_round_y,
                    v_80,
                    v_8000,
                );
                std::ptr::copy_nonoverlapping(tail_y.as_ptr(), row_y.add(full_px), rem_px);
                std::ptr::copy_nonoverlapping(tail_u.as_ptr(), row_u.add(full_px), rem_px);
                std::ptr::copy_nonoverlapping(tail_v.as_ptr(), row_v.add(full_px), rem_px);
            }
        }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    fn run_libyuv_ref_444(
        rgb: &[u8],
        w: i32,
        h: i32,
        is_bgr: bool,
        is_full: bool,
    ) -> AvifResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        let len = (w as usize) * (h as usize);
        let mut y = try_vec_exact![0u8; len]?;
        let mut u = try_vec_exact![0u8; len]?;
        let mut v = try_vec_exact![0u8; len]?;
        let rc = if is_bgr && !is_full {
            let mut argb = try_vec_exact![0u8; len * 4]?;
            unsafe {
                let r1 = RGB24ToARGB(rgb.as_ptr(), w * 3, argb.as_mut_ptr(), w * 4, w, h);
                assert_eq!(r1, 0);
                ARGBToI444(
                    argb.as_ptr(),
                    w * 4,
                    y.as_mut_ptr(),
                    w,
                    u.as_mut_ptr(),
                    w,
                    v.as_mut_ptr(),
                    w,
                    w,
                    h,
                )
            }
        } else {
            let mut raw_buf = try_vec_exact![0u8; len * 3]?;
            if is_bgr {
                for px in 0..len {
                    raw_buf[px * 3] = rgb[px * 3 + 2];
                    raw_buf[px * 3 + 1] = rgb[px * 3 + 1];
                    raw_buf[px * 3 + 2] = rgb[px * 3];
                }
            } else {
                raw_buf.copy_from_slice(rgb);
            }
            unsafe {
                if is_full {
                    RAWToJ444(
                        raw_buf.as_ptr(),
                        w * 3,
                        y.as_mut_ptr(),
                        w,
                        u.as_mut_ptr(),
                        w,
                        v.as_mut_ptr(),
                        w,
                        w,
                        h,
                    )
                } else {
                    RAWToI444(
                        raw_buf.as_ptr(),
                        w * 3,
                        y.as_mut_ptr(),
                        w,
                        u.as_mut_ptr(),
                        w,
                        v.as_mut_ptr(),
                        w,
                        w,
                        h,
                    )
                }
            }
        };
        assert_eq!(rc, 0);
        Ok((y, u, v))
    }

    #[test]
    fn test_raw_to_yuv444_avx2_exact_0lsb_widths_and_edges() -> AvifResult<()> {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        // 1) Boundary widths (1, 7, 15, 31, 32, 33, 63, 64, 65) with multi-row padded strides.
        for is_bgr in [false, true] {
            for is_full in [false, true] {
                for &w in &[1usize, 7, 15, 31, 32, 33, 63, 64, 65] {
                    let h = 3usize;
                    let tight_len = w * h * 3;
                    let mut tight = try_vec_exact![0u8; tight_len]?;
                    for (idx, b) in tight.iter_mut().enumerate() {
                        *b = ((idx * 173 + 29) ^ (idx >> 3)) as u8;
                    }
                    let (ref_y, ref_u, ref_v) =
                        run_libyuv_ref_444(&tight, w as i32, h as i32, is_bgr, is_full)?;

                    let src_stride = w * 3 + 7;
                    let dst_stride = w.next_multiple_of(2);
                    let mut padded_src = try_vec_exact![0x5Au8; h * src_stride]?;
                    for r in 0..h {
                        padded_src[r * src_stride..r * src_stride + w * 3]
                            .copy_from_slice(&tight[r * w * 3..(r + 1) * w * 3]);
                    }
                    let mut dy = try_vec_exact![0u8; h * dst_stride]?;
                    let mut du = try_vec_exact![0u8; h * dst_stride]?;
                    let mut dv = try_vec_exact![0u8; h * dst_stride]?;
                    // SAFETY: AVX2 support checked above, buffers sized for padded strides.
                    unsafe {
                        raw_to_yuv444_avx2(
                            padded_src.as_ptr(),
                            src_stride,
                            dy.as_mut_ptr(),
                            dst_stride,
                            du.as_mut_ptr(),
                            dst_stride,
                            dv.as_mut_ptr(),
                            dst_stride,
                            w,
                            h,
                            is_bgr,
                            is_full,
                        );
                    }
                    for r in 0..h {
                        assert_eq!(
                            dy[r * dst_stride..r * dst_stride + w],
                            ref_y[r * w..(r + 1) * w],
                            "Y mismatch w={w} r={r} is_bgr={is_bgr} is_full={is_full}"
                        );
                        assert_eq!(
                            du[r * dst_stride..r * dst_stride + w],
                            ref_u[r * w..(r + 1) * w],
                            "U mismatch w={w} r={r} is_bgr={is_bgr} is_full={is_full}"
                        );
                        assert_eq!(
                            dv[r * dst_stride..r * dst_stride + w],
                            ref_v[r * w..(r + 1) * w],
                            "V mismatch w={w} r={r} is_bgr={is_bgr} is_full={is_full}"
                        );
                    }
                }
            }
        }

        // 2) Exhaustive edge / high-frequency combinations over [0..=255] sampled grid + extremes.
        let mut samples = [0u8; 16 * 16 * 16 * 3];
        let mut idx = 0usize;
        for r_step in 0..16u16 {
            let rv = if r_step == 15 { 255 } else { (r_step * 17) as u8 };
            for g_step in 0..16u16 {
                let gv = if g_step == 15 { 255 } else { (g_step * 17) as u8 };
                for b_step in 0..16u16 {
                    let bv = if b_step == 15 { 255 } else { (b_step * 17) as u8 };
                    samples[idx] = rv;
                    samples[idx + 1] = gv;
                    samples[idx + 2] = bv;
                    idx += 3;
                }
            }
        }
        let n_px = 16usize * 16 * 16;
        for is_bgr in [false, true] {
            for is_full in [false, true] {
                let (ref_y, ref_u, ref_v) =
                    run_libyuv_ref_444(&samples, n_px as i32, 1, is_bgr, is_full)?;
                let mut dy = try_vec_exact![0u8; n_px]?;
                let mut du = try_vec_exact![0u8; n_px]?;
                let mut dv = try_vec_exact![0u8; n_px]?;
                // SAFETY: AVX2 support checked above.
                unsafe {
                    raw_to_yuv444_avx2(
                        samples.as_ptr(),
                        n_px * 3,
                        dy.as_mut_ptr(),
                        n_px,
                        du.as_mut_ptr(),
                        n_px,
                        dv.as_mut_ptr(),
                        n_px,
                        n_px,
                        1,
                        is_bgr,
                        is_full,
                    );
                }
                assert_eq!(
                    dy, ref_y,
                    "grid Y mismatch is_bgr={is_bgr} is_full={is_full}"
                );
                assert_eq!(
                    du, ref_u,
                    "grid U mismatch is_bgr={is_bgr} is_full={is_full}"
                );
                assert_eq!(
                    dv, ref_v,
                    "grid V mismatch is_bgr={is_bgr} is_full={is_full}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_raw_to_yuv444_avx2_coalesced_and_tail_0lsb() -> AvifResult<()> {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        for is_bgr in [false, true] {
            for is_full in [false, true] {
                for &(w, h) in &[
                    (1usize, 5),
                    (3, 11),
                    (5, 7),
                    (7, 9),
                    (13, 5),
                    (17, 3),
                    (32, 4),
                    (65, 3),
                ] {
                    let len = w * h;
                    let mut src = try_vec_exact![0u8; len * 3]?;
                    for (idx, b) in src.iter_mut().enumerate() {
                        *b = ((idx * 197 + 41) ^ (idx >> 2)) as u8;
                    }
                    let (ref_y, ref_u, ref_v) =
                        run_libyuv_ref_444(&src, w as i32, h as i32, is_bgr, is_full)?;
                    let mut dy = try_vec_exact![0u8; len]?;
                    let mut du = try_vec_exact![0u8; len]?;
                    let mut dv = try_vec_exact![0u8; len]?;
                    // SAFETY: AVX2 support checked above, tightly packed buffers sized for `w * h`.
                    unsafe {
                        raw_to_yuv444_avx2(
                            src.as_ptr(),
                            w * 3,
                            dy.as_mut_ptr(),
                            w,
                            du.as_mut_ptr(),
                            w,
                            dv.as_mut_ptr(),
                            w,
                            w,
                            h,
                            is_bgr,
                            is_full,
                        );
                    }
                    assert_eq!(
                        dy, ref_y,
                        "coalesced Y w={w} h={h} bgr={is_bgr} full={is_full}"
                    );
                    assert_eq!(
                        du, ref_u,
                        "coalesced U w={w} h={h} bgr={is_bgr} full={is_full}"
                    );
                    assert_eq!(
                        dv, ref_v,
                        "coalesced V w={w} h={h} bgr={is_bgr} full={is_full}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_rgb_to_yuv_with_threads_444_0lsb() -> AvifResult<()> {
        for &(w, h) in &[(640u32, 520u32), (641u32, 521u32)] {
            for fmt in [Format::Rgb, Format::Bgr] {
                for yuv_range in [YuvRange::Limited, YuvRange::Full] {
                    let row_bytes = w * 3;
                    let total = (row_bytes * h) as usize;
                    let mut pixels = try_vec_exact![0u8; total]?;
                    for (idx, b) in pixels.iter_mut().enumerate() {
                        *b = ((idx * 89 + 17) ^ (idx >> 4)) as u8;
                    }
                    let is_bgr = fmt == Format::Bgr;
                    let is_full = yuv_range == YuvRange::Full;
                    let rgb = rgb::Image {
                        width: w,
                        height: h,
                        depth: 8,
                        format: fmt,
                        pixels: Some(crate::utils::pixels::Pixels::Buffer(pixels.clone())),
                        row_bytes,
                        ..Default::default()
                    };
                    let mut img_got = image::Image {
                        width: w,
                        height: h,
                        depth: 8,
                        yuv_format: PixelFormat::Yuv444,
                        yuv_range,
                        matrix_coefficients: MatrixCoefficients::Bt601,
                        ..Default::default()
                    };
                    let res = rgb_to_yuv_with_threads(&rgb, &mut img_got, 1)?;
                    if is_bgr && is_full {
                        // BGR Full Yuv444 is left to rgb_impl::avx2 (float32) to stay bit-exact.
                        assert!(res.is_none());
                        continue;
                    }
                    assert!(res.is_some());
                    let (ref_y, ref_u, ref_v) =
                        run_libyuv_ref_444(&pixels, w as i32, h as i32, is_bgr, is_full)?;
                    let wu = w as usize;
                    for row in 0..h as usize {
                        assert_eq!(
                            img_got.row(Plane::Y, row as u32)?,
                            &ref_y[row * wu..(row + 1) * wu]
                        );
                        assert_eq!(
                            img_got.row(Plane::U, row as u32)?,
                            &ref_u[row * wu..(row + 1) * wu]
                        );
                        assert_eq!(
                            img_got.row(Plane::V, row as u32)?,
                            &ref_v[row * wu..(row + 1) * wu]
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_rgb_to_yuv_with_threads_444_large_mt_0lsb() -> AvifResult<()> {
        // Exercises worker count > 1 above the 6_000_000 pixel threshold with odd dimensions.
        let (w, h) = (3001u32, 2001u32);
        for fmt in [Format::Rgb, Format::Bgr] {
            for yuv_range in [YuvRange::Limited, YuvRange::Full] {
                let row_bytes = w * 3;
                let total = (row_bytes * h) as usize;
                let mut pixels = try_vec_exact![0u8; total]?;
                for (idx, b) in pixels.iter_mut().enumerate() {
                    *b = ((idx.wrapping_mul(73) + 19) ^ (idx >> 5)) as u8;
                }
                let rgb = rgb::Image {
                    width: w,
                    height: h,
                    depth: 8,
                    format: fmt,
                    pixels: Some(crate::utils::pixels::Pixels::Buffer(pixels)),
                    row_bytes,
                    ..Default::default()
                };
                let mut img_st = image::Image {
                    width: w,
                    height: h,
                    depth: 8,
                    yuv_format: PixelFormat::Yuv444,
                    yuv_range,
                    matrix_coefficients: MatrixCoefficients::Bt601,
                    ..Default::default()
                };
                let mut img_mt = image::Image {
                    width: w,
                    height: h,
                    depth: 8,
                    yuv_format: PixelFormat::Yuv444,
                    yuv_range,
                    matrix_coefficients: MatrixCoefficients::Bt601,
                    ..Default::default()
                };
                let st_res = rgb_to_yuv_with_threads(&rgb, &mut img_st, 1)?;
                let mt_res = rgb_to_yuv_with_threads(&rgb, &mut img_mt, 8)?;
                if fmt == Format::Bgr && yuv_range == YuvRange::Full {
                    // BGR Full Yuv444 is left to rgb_impl::avx2 (float32) to stay bit-exact.
                    assert!(st_res.is_none());
                    assert!(mt_res.is_none());
                    continue;
                }
                assert!(st_res.is_some());
                assert!(mt_res.is_some());
                for plane in [Plane::Y, Plane::U, Plane::V] {
                    for row in 0..h {
                        assert_eq!(
                            img_mt.row(plane, row)?,
                            img_st.row(plane, row)?,
                            "large MT/ST mismatch fmt={fmt:?} range={yuv_range:?} \
                             plane={plane:?} row={row}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_reorder_then_yuv_422_420_and_large_chunked() -> AvifResult<()> {
        for &(w, h, yuv_format, yuv_range, fmt) in &[
            (
                65u32,
                12u32,
                PixelFormat::Yuv422,
                YuvRange::Full,
                Format::Rgb,
            ),
            (65, 12, PixelFormat::Yuv422, YuvRange::Limited, Format::Rgba),
            (65, 12, PixelFormat::Yuv422, YuvRange::Full, Format::Argb),
            (65, 12, PixelFormat::Yuv422, YuvRange::Limited, Format::Bgr),
            (65, 12, PixelFormat::Yuv422, YuvRange::Full, Format::Abgr),
            (64, 12, PixelFormat::Yuv420, YuvRange::Full, Format::Argb),
            (64, 12, PixelFormat::Yuv420, YuvRange::Full, Format::Abgr),
            // > SOFT_ALLOCATION_LIMIT (16384): 4105 * 4 = 16420 > 16384 triggers heap chunk loop.
            (4105, 6, PixelFormat::Yuv422, YuvRange::Full, Format::Abgr),
            (4106, 6, PixelFormat::Yuv420, YuvRange::Full, Format::Argb),
            // Odd height above chunk limit on 4:2:0 exercises final 1-row tail after 2-row chunks.
            (4106, 7, PixelFormat::Yuv420, YuvRange::Full, Format::Argb),
        ] {
            let ch = fmt.channel_count() as usize;
            let row_bytes = w * (ch as u32);
            let total = (row_bytes * h) as usize;
            let mut pixels = try_vec_exact![0u8; total]?;
            for (idx, b) in pixels.iter_mut().enumerate() {
                *b = ((idx.wrapping_mul(59) + 13) ^ (idx >> 3)) as u8;
            }
            let rgb = rgb::Image {
                width: w,
                height: h,
                depth: 8,
                format: fmt,
                pixels: Some(crate::utils::pixels::Pixels::Buffer(pixels.clone())),
                row_bytes,
                ..Default::default()
            };
            let bgra_ref = rgb::Image {
                width: w,
                height: h,
                depth: 8,
                format: fmt,
                pixels: Some(crate::utils::pixels::Pixels::Buffer(pixels)),
                row_bytes,
                ..Default::default()
            }
            .shuffle_channels_to(Format::Bgra)?;
            let mut img_got = image::Image {
                width: w,
                height: h,
                depth: 8,
                yuv_format,
                yuv_range,
                matrix_coefficients: MatrixCoefficients::Bt601,
                ..Default::default()
            };
            // `Format::Bgra` uses the direct whole-image `ARGBToI4xx`/`ARGBToJ4xx` libyuv entry,
            // catching any channel-order swap (`BGRAToARGB` vs `RGBAToARGB`) or chunk-seam bug.
            let mut img_ref = image::Image {
                width: w,
                height: h,
                depth: 8,
                yuv_format,
                yuv_range,
                matrix_coefficients: MatrixCoefficients::Bt601,
                ..Default::default()
            };
            assert!(rgb_to_yuv_with_threads(&rgb, &mut img_got, 1)?.is_some());
            assert!(rgb_to_yuv_with_threads(&bgra_ref, &mut img_ref, 1)?.is_some());
            for plane in [Plane::Y, Plane::U, Plane::V] {
                assert!(img_got.has_plane(plane));
                let ph = img_got.height(plane) as u32;
                let pw = img_got.width(plane);
                for row in 0..ph {
                    assert_eq!(
                        &img_got.row(plane, row)?[..pw],
                        &img_ref.row(plane, row)?[..pw],
                        "ReorderThenYUV mismatch fmt={fmt:?} {yuv_format:?} {w}x{h} \
                         plane={plane:?} row={row}"
                    );
                }
            }
        }
        Ok(())
    }
}
