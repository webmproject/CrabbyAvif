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

fn validate_plane(image: &image::Image, plane: Plane) -> AvifResult<()> {
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
    Ok(())
}

pub(crate) fn rgb_to_yuv(rgb: &rgb::Image, image: &mut image::Image) -> AvifResult<()> {
    if image.yuv_format != PixelFormat::Yuv420
        || !matches!(image.depth, 8 | 10 | 12)
        || rgb.format == Format::Rgb565
        || rgb.format == Format::Rgba1010102
        || rgb.format.is_gray()
        || rgb.is_float
        || rgb.width != image.width
        || rgb.height != image.height
        || !rgb.has_valid_pixels()
    {
        return AvifError::reformat_failed();
    }
    for plane in YUV_PLANES {
        validate_plane(image, plane)?;
    }
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
    let rgb_pixels = rgb.pixels.as_ref().ok_or(AvifError::ReformatFailed)?;
    let rgb_pixel_size = rgb.pixel_size();
    let (r_ptr, g_ptr, b_ptr) = if rgb.depth == 8 {
        let rgb_span_bytes = checked_add!(
            checked_mul!(rgb.height - 1, rgb.row_bytes)?,
            checked_mul!(rgb.width, rgb_pixel_size)?
        )?;
        let slice = rgb_pixels
            .slice(0, rgb_span_bytes)
            .map_err(|_| AvifError::ReformatFailed)?;
        (
            slice
                .get(rgb.format.r_offset()..)
                .ok_or(AvifError::ReformatFailed)?
                .as_ptr() as *const c_void,
            slice
                .get(rgb.format.g_offset()..)
                .ok_or(AvifError::ReformatFailed)?
                .as_ptr() as *const c_void,
            slice
                .get(rgb.format.b_offset()..)
                .ok_or(AvifError::ReformatFailed)?
                .as_ptr() as *const c_void,
        )
    } else {
        let rgb_span_u16 = checked_add!(
            checked_mul!(rgb.height - 1, rgb.row_bytes / 2)?,
            checked_mul!(rgb.width, rgb_pixel_size / 2)?
        )?;
        let slice16 = rgb_pixels
            .slice16(0, rgb_span_u16)
            .map_err(|_| AvifError::ReformatFailed)?;
        (
            slice16
                .get(rgb.format.r_offset()..)
                .ok_or(AvifError::ReformatFailed)?
                .as_ptr() as *const c_void,
            slice16
                .get(rgb.format.g_offset()..)
                .ok_or(AvifError::ReformatFailed)?
                .as_ptr() as *const c_void,
            slice16
                .get(rgb.format.b_offset()..)
                .ok_or(AvifError::ReformatFailed)?
                .as_ptr() as *const c_void,
        )
    };
    let rgb_step = i32_from_u32(rgb_pixel_size)?;
    let rgb_stride = i32_from_u32(rgb.row_bytes)?;
    let rgb_bit_depth = i32::from(rgb.depth);
    let width = i32_from_u32(rgb.width)?;
    let height = i32_from_u32(rgb.height)?;
    let plane_row_bytes = image.plane_row_bytes()?;
    let yuv_bit_depth = i32::from(image.depth);
    let y_stride = plane_row_bytes[0];
    let u_stride = plane_row_bytes[1];
    let v_stride = plane_row_bytes[2];
    let plane_u8 = image.plane_ptrs_mut();
    let y_ptr = plane_u8[0] as *mut c_void;
    let u_ptr = plane_u8[1] as *mut c_void;
    let v_ptr = plane_u8[2] as *mut c_void;
    // SAFETY:
    // - `rgb.has_valid_pixels()` verified non-zero dimensions, valid `rgb.depth` (8/10/12/16), and
    //   2-byte alignment and even `row_bytes` when `rgb.depth > 8`. `r_ptr`, `g_ptr`, and `b_ptr`
    //   are derived from a single slice covering all `height` rows up to the end of the last row's
    //   active pixels, so stepping across all `height` rows by `rgb_stride` stays in bounds with
    //   full multi-row provenance.
    // - `image.yuv_format == PixelFormat::Yuv420`, `image.depth` is 8/10/12, and `validate_plane`
    //   verified that `y_ptr`, `u_ptr`, and `v_ptr` are non-null, properly aligned (`u8` for 8-bit
    //   or `u16` with even strides for 10/12-bit), and backed by allocations with sufficient
    //   capacity for all luma and 4:2:0 chroma rows; `plane_ptrs_mut()` is taken after all shared
    //   reads of `image` so the mutable pointers remain uniquely borrowed for the FFI call.
    // - `options` was initialized by `SharpYuvOptionsInitInternal` (`init_ok != 0`) and borrows
    //   the live `matrix` struct on this stack frame.
    let converted = unsafe {
        SharpYuvConvertWithOptions(
            r_ptr,
            g_ptr,
            b_ptr,
            rgb_step,
            rgb_stride,
            rgb_bit_depth,
            y_ptr,
            y_stride,
            u_ptr,
            u_stride,
            v_ptr,
            v_stride,
            yuv_bit_depth,
            width,
            height,
            &options,
        )
    };
    if converted == 0 {
        AvifError::reformat_failed()
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::pixels::Pixels;
    use test_case::test_case;

    fn valid_pair(rgb_depth: u8, yuv_depth: u8) -> (rgb::Image, image::Image) {
        let mut yuv = image::Image {
            width: 4,
            height: 4,
            depth: yuv_depth,
            yuv_format: PixelFormat::Yuv420,
            matrix_coefficients: MatrixCoefficients::Bt601,
            ..Default::default()
        };
        yuv.allocate_planes(Category::Color).unwrap();
        let mut rgb = rgb::Image::create_from_yuv(&yuv);
        rgb.depth = rgb_depth;
        rgb.chroma_downsampling = ChromaDownsampling::SharpYuv;
        rgb.allocate().unwrap();
        (rgb, yuv)
    }

    #[test_case(8, 8)]
    #[test_case(10, 10)]
    #[test_case(16, 12)]
    fn rgb_to_yuv_valid(rgb_depth: u8, yuv_depth: u8) {
        let (rgb, mut yuv) = valid_pair(rgb_depth, yuv_depth);
        assert_eq!(rgb_to_yuv(&rgb, &mut yuv), Ok(()));
    }

    #[test_case(PixelFormat::Yuv444)]
    #[test_case(PixelFormat::Yuv422)]
    #[test_case(PixelFormat::Yuv400)]
    fn rgb_to_yuv_rejects_non_yuv420(yuv_format: PixelFormat) {
        let (rgb, mut yuv) = valid_pair(8, 8);
        yuv.yuv_format = yuv_format;
        yuv.allocate_planes(Category::Color).unwrap();
        assert_eq!(rgb_to_yuv(&rgb, &mut yuv), Err(AvifError::ReformatFailed));
    }

    #[test_case(0)]
    #[test_case(16)]
    fn rgb_to_yuv_rejects_unsupported_yuv_depth(yuv_depth: u8) {
        let (rgb, mut yuv) = valid_pair(8, 8);
        yuv.depth = yuv_depth;
        assert_eq!(rgb_to_yuv(&rgb, &mut yuv), Err(AvifError::ReformatFailed));
    }

    #[test_case(Format::Rgb565, 8, false)]
    #[test_case(Format::Rgba1010102, 10, false)]
    #[test_case(Format::Gray, 8, false)]
    #[test_case(Format::GrayA, 8, false)]
    #[test_case(Format::AGray, 8, false)]
    #[test_case(Format::Rgba, 16, true)]
    fn rgb_to_yuv_rejects_unsupported_rgb_format(format: Format, depth: u8, is_float: bool) {
        let (mut rgb, mut yuv) = valid_pair(8, 8);
        rgb.format = format;
        rgb.depth = depth;
        rgb.is_float = is_float;
        rgb.allocate().unwrap();
        assert_eq!(rgb_to_yuv(&rgb, &mut yuv), Err(AvifError::ReformatFailed));
    }

    #[test]
    fn rgb_to_yuv_rejects_invalid_rgb_or_mismatched_dimensions() {
        let (mut rgb, mut yuv) = valid_pair(8, 8);
        rgb.width = 2;
        assert_eq!(rgb_to_yuv(&rgb, &mut yuv), Err(AvifError::ReformatFailed));

        let (mut rgb, mut yuv) = valid_pair(8, 8);
        rgb.height = 2;
        assert_eq!(rgb_to_yuv(&rgb, &mut yuv), Err(AvifError::ReformatFailed));

        let (mut rgb, mut yuv) = valid_pair(8, 8);
        rgb.pixels = None;
        assert_eq!(rgb_to_yuv(&rgb, &mut yuv), Err(AvifError::ReformatFailed));
    }

    #[test]
    fn rgb_to_yuv_rejects_invalid_planes() {
        // Missing chroma plane.
        let (rgb, mut yuv) = valid_pair(8, 8);
        yuv.planes[Plane::U.as_usize()] = None;
        assert_eq!(rgb_to_yuv(&rgb, &mut yuv), Err(AvifError::ReformatFailed));

        // Row stride smaller than plane width * pixel_size.
        let (rgb, mut yuv) = valid_pair(8, 8);
        yuv.row_bytes[Plane::Y.as_usize()] = 3;
        assert_eq!(rgb_to_yuv(&rgb, &mut yuv), Err(AvifError::ReformatFailed));

        // Truncated 8-bit plane buffer.
        let (rgb, mut yuv) = valid_pair(8, 8);
        yuv.planes[Plane::Y.as_usize()] = Some(Pixels::Buffer(vec![0u8; 12]));
        assert_eq!(rgb_to_yuv(&rgb, &mut yuv), Err(AvifError::ReformatFailed));

        // Odd byte stride on a 10-bit plane.
        let (rgb, mut yuv) = valid_pair(10, 10);
        yuv.planes[Plane::Y.as_usize()] = Some(Pixels::Buffer16(vec![0u16; 32]));
        yuv.row_bytes[Plane::Y.as_usize()] = 9;
        assert_eq!(rgb_to_yuv(&rgb, &mut yuv), Err(AvifError::ReformatFailed));

        // Truncated 10-bit chroma plane buffer.
        let (rgb, mut yuv) = valid_pair(10, 10);
        yuv.planes[Plane::V.as_usize()] = Some(Pixels::Buffer16(vec![0u16; 3]));
        assert_eq!(rgb_to_yuv(&rgb, &mut yuv), Err(AvifError::ReformatFailed));

        // Mismatched Pixels variant (8-bit buffer on a 10-bit image).
        let (rgb, mut yuv) = valid_pair(10, 10);
        yuv.planes[Plane::Y.as_usize()] = Some(Pixels::Buffer(vec![0u8; 64]));
        assert_eq!(rgb_to_yuv(&rgb, &mut yuv), Err(AvifError::ReformatFailed));
    }

    #[test]
    fn rgb_to_yuv_strided_unpadded_last_row() {
        // 8-bit BGRA 4x4 with row_bytes=20 (> 16) and unpadded final row (3 * 20 + 16 = 76 bytes).
        let (mut rgb8, mut yuv8) = valid_pair(8, 8);
        rgb8.format = Format::Bgra;
        rgb8.row_bytes = 20;
        rgb8.pixels = Some(Pixels::Buffer(vec![128u8; 76]));
        assert_eq!(rgb_to_yuv(&rgb8, &mut yuv8), Ok(()));

        // 10-bit BGRA 4x4 with row_bytes=40 (> 32) and unpadded final row (3 * 20 + 16 = 76 u16s).
        let (mut rgb10, mut yuv10) = valid_pair(10, 10);
        rgb10.format = Format::Bgra;
        rgb10.row_bytes = 40;
        rgb10.pixels = Some(Pixels::Buffer16(vec![512u16; 76]));
        assert_eq!(rgb_to_yuv(&rgb10, &mut yuv10), Ok(()));
    }
}
