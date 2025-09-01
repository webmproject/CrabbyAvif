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

fn find_constants(image: &image::Image) -> Option<(&YuvConstants, &YuvConstants)> {
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
    unsafe {
        match image.yuv_range {
            YuvRange::Full => match matrix_coefficients {
                MatrixCoefficients::Bt709 => Some((&kYuvF709Constants, &kYvuF709Constants)),
                MatrixCoefficients::Bt470bg
                | MatrixCoefficients::Bt601
                | MatrixCoefficients::Unspecified => Some((&kYuvJPEGConstants, &kYvuJPEGConstants)),
                MatrixCoefficients::Bt2020Ncl => Some((&kYuvV2020Constants, &kYvuV2020Constants)),
                MatrixCoefficients::ChromaDerivedNcl => match image.color_primaries {
                    ColorPrimaries::Srgb | ColorPrimaries::Unspecified => {
                        Some((&kYuvF709Constants, &kYvuF709Constants))
                    }
                    ColorPrimaries::Bt470bg | ColorPrimaries::Bt601 => {
                        Some((&kYuvJPEGConstants, &kYvuJPEGConstants))
                    }
                    ColorPrimaries::Bt2020 => Some((&kYuvV2020Constants, &kYvuV2020Constants)),
                    _ => None,
                },
                _ => None,
            },
            YuvRange::Limited => match matrix_coefficients {
                MatrixCoefficients::Bt709 => Some((&kYuvH709Constants, &kYvuH709Constants)),
                MatrixCoefficients::Bt470bg
                | MatrixCoefficients::Bt601
                | MatrixCoefficients::Unspecified => Some((&kYuvI601Constants, &kYvuI601Constants)),
                MatrixCoefficients::Bt2020Ncl => Some((&kYuv2020Constants, &kYvu2020Constants)),
                MatrixCoefficients::ChromaDerivedNcl => match image.color_primaries {
                    ColorPrimaries::Srgb | ColorPrimaries::Unspecified => {
                        Some((&kYuvH709Constants, &kYvuH709Constants))
                    }
                    ColorPrimaries::Bt470bg | ColorPrimaries::Bt601 => {
                        Some((&kYuvI601Constants, &kYvuI601Constants))
                    }
                    ColorPrimaries::Bt2020 => Some((&kYuv2020Constants, &kYvu2020Constants)),
                    _ => None,
                },
                _ => None,
            },
        }
    }
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

#[cfg_attr(feature = "disable_cfi", sanitize(cfi = "off"))]
pub(crate) fn yuv_to_rgb(image: &image::Image, rgb: &mut rgb::Image) -> AvifResult<Option<bool>> {
    if (rgb.depth != 8 && rgb.depth != 10) || !image.depth_valid() {
        return Ok(None); // Not implemented.
    }
    if rgb.depth == 10
        && (!matches!(
            image.yuv_format,
            PixelFormat::AndroidP010 | PixelFormat::Yuv420
        ) || rgb.format != Format::Rgba1010102)
    {
        return Ok(None); // Not implemented.
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
    let is_yvu = matches!(rgb.format, Format::Rgb | Format::Rgba | Format::Argb);
    let matrix = if is_yvu { matrix_yvu } else { matrix_yuv };
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
    let mut result: c_int;
    unsafe {
        let mut high_bd_matched = true;
        // Apply one of the high bitdepth functions if possible.
        result = match conversion_function {
            ConversionFunction::P010ToRGBMatrix(func1, func2) => {
                let result = func1(
                    plane_u16[0],
                    plane_row_bytes[0] / 2,
                    plane_u16[1],
                    plane_row_bytes[1] / 2,
                    rgb.pixels_mut(),
                    rgb_row_bytes,
                    matrix,
                    width,
                    height,
                );
                if result == 0 {
                    // It is okay to use the same pointer as source and destination for this
                    // conversion.
                    func2(
                        rgb.pixels_mut(),
                        rgb_row_bytes,
                        rgb.pixels_mut(),
                        rgb_row_bytes,
                        width,
                        height,
                    )
                } else {
                    result
                }
            }
            ConversionFunction::YUVToAB30Matrix(func1, func2) => {
                let result = func1(
                    plane_u16[0],
                    plane_row_bytes[0] / 2,
                    plane_u16[1],
                    plane_row_bytes[1] / 2,
                    plane_u16[2],
                    plane_row_bytes[2] / 2,
                    rgb.pixels_mut(),
                    rgb_row_bytes,
                    matrix,
                    width,
                    height,
                );
                if result == 0 {
                    // It is okay to use the same pointer as source and destination for this
                    // conversion.
                    func2(
                        rgb.pixels_mut(),
                        rgb_row_bytes,
                        rgb.pixels_mut(),
                        rgb_row_bytes,
                        width,
                        height,
                    )
                } else {
                    result
                }
            }
            ConversionFunction::YUVToRGBMatrixFilterHighBitDepth(func) => func(
                plane_u16[0],
                plane_row_bytes[0] / 2,
                plane_u16[u_plane_index],
                plane_row_bytes[u_plane_index] / 2,
                plane_u16[v_plane_index],
                plane_row_bytes[v_plane_index] / 2,
                rgb.pixels_mut(),
                rgb_row_bytes,
                matrix,
                width,
                height,
                filter,
            ),
            ConversionFunction::YUVAToRGBMatrixFilterHighBitDepth(func) => func(
                plane_u16[0],
                plane_row_bytes[0] / 2,
                plane_u16[u_plane_index],
                plane_row_bytes[u_plane_index] / 2,
                plane_u16[v_plane_index],
                plane_row_bytes[v_plane_index] / 2,
                plane_u16[3],
                plane_row_bytes[3] / 2,
                rgb.pixels_mut(),
                rgb_row_bytes,
                matrix,
                width,
                height,
                0, // attenuate
                filter,
            ),
            ConversionFunction::YUVToRGBMatrixHighBitDepth(func) => func(
                plane_u16[0],
                plane_row_bytes[0] / 2,
                plane_u16[u_plane_index],
                plane_row_bytes[u_plane_index] / 2,
                plane_u16[v_plane_index],
                plane_row_bytes[v_plane_index] / 2,
                rgb.pixels_mut(),
                rgb_row_bytes,
                matrix,
                width,
                height,
            ),
            ConversionFunction::YUVAToRGBMatrixHighBitDepth(func) => func(
                plane_u16[0],
                plane_row_bytes[0] / 2,
                plane_u16[u_plane_index],
                plane_row_bytes[u_plane_index] / 2,
                plane_u16[v_plane_index],
                plane_row_bytes[v_plane_index] / 2,
                plane_u16[3],
                plane_row_bytes[3] / 2,
                rgb.pixels_mut(),
                rgb_row_bytes,
                matrix,
                width,
                height,
                0, // attenuate
            ),
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
            plane_u8 = image8.plane_ptrs();
            plane_row_bytes = image8.plane_row_bytes()?;
        }
        result = match conversion_function {
            ConversionFunction::NVToARGBMatrix(func) => func(
                plane_u8[0],
                plane_row_bytes[0],
                plane_u8[1],
                plane_row_bytes[1],
                rgb.pixels_mut(),
                rgb_row_bytes,
                matrix,
                width,
                height,
            ),
            ConversionFunction::YUV400ToRGBMatrix(func) => func(
                plane_u8[0],
                plane_row_bytes[0],
                rgb.pixels_mut(),
                rgb_row_bytes,
                matrix,
                width,
                height,
            ),
            ConversionFunction::YUVToRGBMatrixFilter(func) => func(
                plane_u8[0],
                plane_row_bytes[0],
                plane_u8[u_plane_index],
                plane_row_bytes[u_plane_index],
                plane_u8[v_plane_index],
                plane_row_bytes[v_plane_index],
                rgb.pixels_mut(),
                rgb_row_bytes,
                matrix,
                width,
                height,
                filter,
            ),
            ConversionFunction::YUVAToRGBMatrixFilter(func) => func(
                plane_u8[0],
                plane_row_bytes[0],
                plane_u8[u_plane_index],
                plane_row_bytes[u_plane_index],
                plane_u8[v_plane_index],
                plane_row_bytes[v_plane_index],
                plane_u8[3],
                plane_row_bytes[3],
                rgb.pixels_mut(),
                rgb_row_bytes,
                matrix,
                width,
                height,
                0, // attenuate
                filter,
            ),
            ConversionFunction::YUVToRGBMatrix(func) => func(
                plane_u8[0],
                plane_row_bytes[0],
                plane_u8[u_plane_index],
                plane_row_bytes[u_plane_index],
                plane_u8[v_plane_index],
                plane_row_bytes[v_plane_index],
                rgb.pixels_mut(),
                rgb_row_bytes,
                matrix,
                width,
                height,
            ),
            ConversionFunction::YUVAToRGBMatrix(func) => func(
                plane_u8[0],
                plane_row_bytes[0],
                plane_u8[u_plane_index],
                plane_row_bytes[u_plane_index],
                plane_u8[v_plane_index],
                plane_row_bytes[v_plane_index],
                plane_u8[3],
                plane_row_bytes[3],
                rgb.pixels_mut(),
                rgb_row_bytes,
                matrix,
                width,
                height,
                0, // attenuate
            ),
            _ => 0,
        };
    }
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
        if plane == Plane::A && !alpha {
            continue;
        }
        let pd = image.plane_data(plane);
        if pd.is_none() {
            continue;
        }
        let pd = pd.unwrap();
        if pd.width == 0 {
            continue;
        }
        let source_ptr = image.planes[plane.as_usize()].unwrap_ref().ptr16();
        let pd8 = image8.plane_data(plane).unwrap();
        let dst_ptr = image8.planes[plane.as_usize()].unwrap_mut().ptr_mut();
        unsafe {
            Convert16To8Plane(
                source_ptr,
                i32_from_u32(pd.row_bytes / 2)?,
                dst_ptr,
                i32_from_u32(pd8.row_bytes)?,
                scale,
                i32_from_u32(pd.width)?,
                i32_from_u32(pd.height)?,
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
    let result = unsafe {
        if multiply {
            ARGBAttenuate(
                rgb.pixels_mut(),
                i32_from_u32(rgb.row_bytes)?,
                rgb.pixels_mut(),
                i32_from_u32(rgb.row_bytes)?,
                i32_from_u32(rgb.width)?,
                i32_from_u32(rgb.height)?,
            )
        } else {
            ARGBUnattenuate(
                rgb.pixels_mut(),
                i32_from_u32(rgb.row_bytes)?,
                rgb.pixels_mut(),
                i32_from_u32(rgb.row_bytes)?,
                i32_from_u32(rgb.width)?,
                i32_from_u32(rgb.height)?,
            )
        }
    };
    if result == 0 {
        Ok(())
    } else {
        AvifError::reformat_failed()
    }
}

pub(crate) fn convert_to_half_float(rgb: &mut rgb::Image, scale: f32) -> AvifResult<Option<()>> {
    let res = unsafe {
        HalfFloatPlane(
            rgb.pixels_mut() as *const u16,
            i32_from_u32(rgb.row_bytes)?,
            rgb.pixels_mut() as *mut u16,
            i32_from_u32(rgb.row_bytes)?,
            scale,
            i32_from_u32(rgb.width * rgb.channel_count())?,
            i32_from_u32(rgb.height)?,
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
type RGBToYUV = unsafe extern "C" fn(
    *const u8, c_int, *mut u8, c_int, *mut u8, c_int, *mut u8, c_int, c_int, c_int,
) -> c_int;

#[derive(Debug)]
enum RGBToYUVConversionFunction {
    RGBToY(RGBToY),
    RGBToYUV(RGBToYUV),
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
    // TODO: b/410088660 - Implement 2-step RGB conversion for functions which aren't directly
    // available in libyuv.
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
        (PixelFormat::Yuv422, YuvRange::Limited, Format::Bgra) => {
            Some(RGBToYUVConversionFunction::RGBToYUV(ARGBToI422))
        }
        (PixelFormat::Yuv444, YuvRange::Limited, Format::Bgra) => {
            Some(RGBToYUVConversionFunction::RGBToYUV(ARGBToI444))
        }
        (PixelFormat::Yuv420, YuvRange::Full, Format::Rgb) => {
            Some(RGBToYUVConversionFunction::RGBToYUV(RAWToJ420))
        }
        (PixelFormat::Yuv420, YuvRange::Full, Format::Rgba) => {
            Some(RGBToYUVConversionFunction::RGBToYUV(ABGRToJ420))
        }
        (PixelFormat::Yuv420, YuvRange::Full, Format::Bgr) => {
            Some(RGBToYUVConversionFunction::RGBToYUV(RGB24ToJ420))
        }
        (PixelFormat::Yuv420, YuvRange::Full, Format::Bgra) => {
            Some(RGBToYUVConversionFunction::RGBToYUV(ARGBToJ420))
        }
        (PixelFormat::Yuv422, YuvRange::Full, Format::Rgba) => {
            Some(RGBToYUVConversionFunction::RGBToYUV(ABGRToJ422))
        }
        (PixelFormat::Yuv422, YuvRange::Full, Format::Bgra) => {
            Some(RGBToYUVConversionFunction::RGBToYUV(ARGBToJ422))
        }
        _ => None, // Not implemented.
    }
}

#[cfg_attr(feature = "disable_cfi", no_sanitize(cfi))]
pub(crate) fn rgb_to_yuv(rgb: &rgb::Image, image: &mut image::Image) -> AvifResult<Option<()>> {
    let conversion_function = match rgb_to_yuv_conversion_function(rgb, image) {
        Some(conversion_function) => conversion_function,
        None => return Ok(None), // Not implemented.
    };
    let plane_u8 = image.plane_ptrs_mut();
    let plane_row_bytes = image.plane_row_bytes()?;
    let width = i32_from_u32(image.width)?;
    let height = i32_from_u32(image.height)?;
    let rgb_row_bytes = i32_from_u32(rgb.row_bytes)?;
    let result = unsafe {
        match conversion_function {
            RGBToYUVConversionFunction::RGBToY(func) => func(
                rgb.pixels(),
                rgb_row_bytes,
                plane_u8[0],
                plane_row_bytes[0],
                width,
                height,
            ),
            RGBToYUVConversionFunction::RGBToYUV(func) => func(
                rgb.pixels(),
                rgb_row_bytes,
                plane_u8[0],
                plane_row_bytes[0],
                plane_u8[1],
                plane_row_bytes[1],
                plane_u8[2],
                plane_row_bytes[2],
                width,
                height,
            ),
        }
    };
    if result == 0 {
        Ok(Some(()))
    } else {
        AvifError::reformat_failed()
    }
}
