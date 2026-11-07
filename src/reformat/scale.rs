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

use crate::image::*;
use crate::internal_utils::*;
use crate::*;

use libyuv_sys::bindings::*;

fn fixed_div(num: u32, div: u32) -> u32 {
    if div == 0 {
        return 0;
    }
    (((num as u64) << 16) / div as u64) as u32
}

fn scale_slope(
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> (u32, u32, u32, u32) {
    let dx = fixed_div(src_width, dst_width);
    let dy = fixed_div(src_height, dst_height);
    let x = dx >> 1;
    let y = dy >> 1;
    (x, y, dx, dy)
}

fn scale_cols(dst: &mut [u8], src: &[u8], dst_width: u32, mut x: u32, dx: u32) {
    if dst_width == 0 {
        return;
    }
    let mut j = 0;
    let mut dst_idx = 0;
    while j + 1 < dst_width {
        dst[dst_idx] = src[(x >> 16) as usize];
        x += dx;
        dst[dst_idx + 1] = src[(x >> 16) as usize];
        x += dx;
        dst_idx += 2;
        j += 2;
    }
    if (dst_width & 1) != 0 {
        dst[dst_idx] = src[(x >> 16) as usize];
    }
}

fn scale_cols16(dst: &mut [u16], src: &[u16], dst_width: u32, mut x: u32, dx: u32) {
    if dst_width == 0 {
        return;
    }
    let mut j = 0;
    let mut dst_idx = 0;
    while j + 1 < dst_width {
        dst[dst_idx] = src[(x >> 16) as usize];
        x += dx;
        dst[dst_idx + 1] = src[(x >> 16) as usize];
        x += dx;
        dst_idx += 2;
        j += 2;
    }
    if (dst_width & 1) != 0 {
        dst[dst_idx] = src[(x >> 16) as usize];
    }
}

fn scale_uv_cols(dst: &mut [u8], src: &[u8], dst_width: u32, mut x: u64, dx: u32) {
    if dst_width == 0 {
        return;
    }
    let mut j = 0;
    let mut dst_idx = 0;
    while j + 1 < dst_width {
        let src_idx = ((x >> 16) as usize) * 2;
        dst[dst_idx] = src[src_idx];
        dst[dst_idx + 1] = src[src_idx + 1];
        x += dx as u64;

        let src_idx = ((x >> 16) as usize) * 2;
        dst[dst_idx + 2] = src[src_idx];
        dst[dst_idx + 3] = src[src_idx + 1];
        x += dx as u64;

        dst_idx += 4;
        j += 2;
    }
    if (dst_width & 1) != 0 {
        let src_idx = ((x >> 16) as usize) * 2;
        dst[dst_idx] = src[src_idx];
        dst[dst_idx + 1] = src[src_idx + 1];
    }
}

impl Image {
    fn scale_plane_impl(&mut self, src: &Image, plane: Plane) -> AvifResult<()> {
        let dst_width = self.width(plane);
        let dst_height = self.height(plane);
        if dst_width == 0 || dst_height == 0 {
            return Ok(());
        }
        let (x, mut y, dx, dy) = scale_slope(
            src.width(plane) as u32,
            src.height(plane) as u32,
            dst_width as u32,
            dst_height as u32,
        );
        if src.depth > 8 {
            for i in 0..dst_height as u32 {
                let src_row = src.row16(plane, y >> 16)?;
                let dst_row = self.row16_mut(plane, i)?;
                scale_cols16(dst_row, src_row, dst_width as u32, x, dx);
                y += dy;
            }
        } else {
            for i in 0..dst_height as u32 {
                let src_row = src.row(plane, y >> 16)?;
                let dst_row = self.row_mut(plane, i)?;
                scale_cols(dst_row, src_row, dst_width as u32, x, dx);
                y += dy;
            }
        }
        Ok(())
    }

    fn scale_impl(&mut self, src: &Image, category: Category) -> AvifResult<()> {
        if src.width < self.width || src.height < self.height {
            // This function only supports scaling down.
            return Err(AvifError::NotImplemented);
        }
        if category != Category::Alpha
            && (self.yuv_format == PixelFormat::AndroidNv12
                || self.yuv_format == PixelFormat::AndroidNv21)
        {
            if self.width < 2 || self.height < 2 {
                return Err(AvifError::NotImplemented);
            }
            self.scale_plane_impl(src, Plane::Y)?;
            let plane = Plane::U;
            // scale_slope and scale_uv_cols expect the width of one chroma plane whereas
            // self.width(plane) will return the width of the interleaved chrome planes in case of
            // Nv12/Nv21. So divide width by 2 when calling these functions.
            let dst_width = self.width(plane) / 2;
            let dst_height = self.height(plane);
            let (x, mut y, dx, dy) = scale_slope(
                src.width(plane) as u32 / 2,
                src.height(plane) as u32,
                dst_width as u32,
                dst_height as u32,
            );
            for i in 0..dst_height as u32 {
                let src_row = src.row(plane, y >> 16)?;
                let dst_row = self.row_mut(plane, i)?;
                scale_uv_cols(dst_row, src_row, dst_width as u32, x as u64, dx);
                y += dy;
            }
            return Ok(());
        }
        for plane in category.planes() {
            if !src.has_plane(*plane) || !self.has_plane(*plane) {
                continue;
            }
            self.scale_plane_impl(src, *plane)?;
        }
        Ok(())
    }
    // Converts 16-bit P010 alpha samples (stored in the upper 10 bits) into 10-bit samples.
    fn downshift_p010_alpha(&mut self) -> AvifResult<()> {
        for y in 0..self.height {
            for pixel in self.row16_mut(Plane::A, y)? {
                *pixel >>= 6;
            }
        }
        Ok(())
    }

    pub(crate) fn scale(&mut self, width: u32, height: u32, category: Category) -> AvifResult<()> {
        if self.width == width && self.height == height {
            return Ok(());
        }
        if width == 0 || height == 0 {
            return AvifError::invalid_argument();
        }
        // The destination planes are only reallocated when the Y plane is present. Reject images
        // that have chroma planes without a Y plane instead of silently dropping the chroma planes.
        if category != Category::Alpha
            && !self.has_plane(Plane::Y)
            && (self.has_plane(Plane::U) || self.has_plane(Plane::V))
        {
            return AvifError::invalid_argument();
        }
        let validate_plane = |img: &Image, plane: Plane| -> AvifResult<PlaneData> {
            let pd = img.plane_data(plane).ok_or(AvifError::NoContent)?;
            if pd.width == 0 || pd.height == 0 {
                return AvifError::no_content();
            }
            let min_row_bytes = checked_mul!(pd.width, pd.pixel_size)?;
            if pd.row_bytes < min_row_bytes {
                return AvifError::no_content();
            }
            if img.depth > 8 {
                if pd.row_bytes % 2 != 0 {
                    return AvifError::no_content();
                }
                let _ = img.row16(plane, pd.height - 1)?;
            } else {
                let _ = img.row(plane, pd.height - 1)?;
            }
            Ok(pd)
        };
        let planes = category.planes();
        let src = if category != Category::Alpha && self.yuv_format == PixelFormat::AndroidP010 {
            // P010 images cannot be scaled using ScalePlane_12 since the U and V planes are
            // interleaved. Convert them into I010 and then scale each plane using
            // ScalePlane_12.
            let src_y_pd = validate_plane(self, Plane::Y)?;
            let src_uv_pd = validate_plane(self, Plane::U)?;
            let mut i010 = image::Image {
                width: self.width,
                height: self.height,
                depth: 10,
                yuv_format: PixelFormat::Yuv420,
                ..image::Image::default()
            };
            i010.allocate_planes(Category::Color)?;
            let dst_y_pd = validate_plane(&i010, Plane::Y)?;
            let dst_u_pd = validate_plane(&i010, Plane::U)?;
            let dst_v_pd = validate_plane(&i010, Plane::V)?;
            let src_y = self.planes[Plane::Y.as_usize()]
                .as_ref()
                .ok_or(AvifError::NoContent)?
                .ptr16();
            let src_uv = self.planes[Plane::U.as_usize()]
                .as_ref()
                .ok_or(AvifError::NoContent)?
                .ptr16();
            let dst_y = i010.planes[Plane::Y.as_usize()]
                .as_mut()
                .ok_or(AvifError::NoContent)?
                .ptr16_mut();
            let dst_u = i010.planes[Plane::U.as_usize()]
                .as_mut()
                .ok_or(AvifError::NoContent)?
                .ptr16_mut();
            let dst_v = i010.planes[Plane::V.as_usize()]
                .as_mut()
                .ok_or(AvifError::NoContent)?
                .ptr16_mut();
            if src_y.is_null()
                || src_uv.is_null()
                || dst_y.is_null()
                || dst_u.is_null()
                || dst_v.is_null()
            {
                return AvifError::no_content();
            }
            let src_stride_y = i32_from_u32(src_y_pd.row_bytes / 2)?;
            let src_stride_uv = i32_from_u32(src_uv_pd.row_bytes / 2)?;
            let dst_stride_y = i32_from_u32(dst_y_pd.row_bytes / 2)?;
            let dst_stride_u = i32_from_u32(dst_u_pd.row_bytes / 2)?;
            let dst_stride_v = i32_from_u32(dst_v_pd.row_bytes / 2)?;
            let src_width = i32_from_u32(self.width)?;
            let src_height = i32_from_u32(self.height)?;
            // SAFETY: `validate_plane` and the null checks above verified that `src_y` and
            // `src_uv` point to valid, 2-byte aligned `u16` planes covering `src_width x
            // src_height` P010 samples, and `dst_y`, `dst_u`, `dst_v` point to freshly allocated,
            // disjoint, 2-byte aligned `u16` buffers in `i010` with valid strides and capacity.
            let ret = unsafe {
                P010ToI010(
                    src_y,
                    src_stride_y,
                    src_uv,
                    src_stride_uv,
                    dst_y,
                    dst_stride_y,
                    dst_u,
                    dst_stride_u,
                    dst_v,
                    dst_stride_v,
                    src_width,
                    src_height,
                )
            };
            if ret != 0 {
                return AvifError::reformat_failed();
            }
            if self.has_plane(Plane::A) {
                let _ = validate_plane(self, Plane::A)?;
                self.downshift_p010_alpha()?;
            }
            i010
        } else {
            // Alpha decoded separately from an AndroidP010 image (e.g. MediaCodec alpha tiles) is
            // also 16-bit. Downshift it so that it matches the 10-bit depth of the color planes.
            if self.yuv_format == PixelFormat::AndroidP010 && self.has_plane(Plane::A) {
                let _ = validate_plane(self, Plane::A)?;
                self.downshift_p010_alpha()?;
                self.depth = 10;
                self.yuv_format = PixelFormat::Yuv420;
            }
            // Move the planes for this category into `src`; they will be reallocated in `self`
            // below.
            let mut src = self.shallow_clone();
            for plane in planes {
                let plane_index = plane.as_usize();
                if let Some(self_plane) = self.planes[plane_index].take() {
                    src.planes[plane_index] = Some(self_plane);
                    src.row_bytes[plane_index] = std::mem::take(&mut self.row_bytes[plane_index]);
                }
            }
            src
        };

        let scale_factor = self.width / width;
        self.width = width;
        self.height = height;
        self.depth = src.depth;
        self.yuv_format = src.yuv_format;
        const LIBYUV_LIMIT: u32 = 16384;
        const LIMIT: u32 = 65536;
        if src.has_plane(Plane::Y) || src.has_plane(Plane::A) {
            if src.width > LIMIT || src.height > LIMIT {
                return AvifError::not_implemented();
            }
            if src.has_plane(Plane::Y) && category != Category::Alpha {
                self.allocate_planes(Category::Color)?;
            }
            if src.has_plane(Plane::A) && category == Category::Alpha {
                self.allocate_planes(Category::Alpha)?;
            }
        }
        if src.width > LIBYUV_LIMIT || src.height > LIBYUV_LIMIT {
            return self.scale_impl(&src, category);
        }
        if category != Category::Alpha
            && (self.yuv_format == PixelFormat::AndroidNv12
                || self.yuv_format == PixelFormat::AndroidNv21)
        {
            let src_y_pd = validate_plane(&src, Plane::Y)?;
            let src_uv_pd = validate_plane(&src, Plane::U)?;
            let dst_y_pd = validate_plane(self, Plane::Y)?;
            let dst_uv_pd = validate_plane(self, Plane::U)?;
            let src_y = src.planes[Plane::Y.as_usize()]
                .as_ref()
                .ok_or(AvifError::NoContent)?
                .ptr();
            let src_uv = src.planes[Plane::U.as_usize()]
                .as_ref()
                .ok_or(AvifError::NoContent)?
                .ptr();
            let dst_y = self.planes[Plane::Y.as_usize()]
                .as_mut()
                .ok_or(AvifError::NoContent)?
                .ptr_mut();
            let dst_uv = self.planes[Plane::U.as_usize()]
                .as_mut()
                .ok_or(AvifError::NoContent)?
                .ptr_mut();
            if src_y.is_null() || src_uv.is_null() || dst_y.is_null() || dst_uv.is_null() {
                return AvifError::no_content();
            }
            let src_stride_y = i32_from_u32(src_y_pd.row_bytes)?;
            let src_stride_uv = i32_from_u32(src_uv_pd.row_bytes)?;
            let src_width = i32_from_u32(src_y_pd.width)?;
            let src_height = i32_from_u32(src_y_pd.height)?;
            let dst_stride_y = i32_from_u32(dst_y_pd.row_bytes)?;
            let dst_stride_uv = i32_from_u32(dst_uv_pd.row_bytes)?;
            let dst_width = i32_from_u32(dst_y_pd.width)?;
            let dst_height = i32_from_u32(dst_y_pd.height)?;
            // SAFETY: `validate_plane` and the null checks above verified that `src_y` and
            // `src_uv` are valid 8-bit luma and interleaved UV buffers for `src_width x
            // src_height`, and `self.allocate_planes` allocated fresh, disjoint `Pixels::Buffer`
            // planes `dst_y` and `dst_uv` large enough for `dst_width x dst_height`.
            let ret = unsafe {
                NV12Scale(
                    src_y,
                    src_stride_y,
                    src_uv,
                    src_stride_uv,
                    src_width,
                    src_height,
                    dst_y,
                    dst_stride_y,
                    dst_uv,
                    dst_stride_uv,
                    dst_width,
                    dst_height,
                    FilterMode_kFilterBox,
                )
            };
            if ret != 0 {
                return AvifError::reformat_failed();
            } else {
                return Ok(());
            }
        }

        for plane in planes {
            if !src.has_plane(*plane) || !self.has_plane(*plane) {
                continue;
            }
            let src_pd = validate_plane(&src, *plane)?;
            let dst_pd = validate_plane(self, *plane)?;
            let src_width = i32_from_u32(src_pd.width)?;
            let src_height = i32_from_u32(src_pd.height)?;
            let dst_width = i32_from_u32(dst_pd.width)?;
            let dst_height = i32_from_u32(dst_pd.height)?;
            // libyuv versions >= 1880 report a return value here. Older versions do not. Ignore
            // the return value for now.
            if src.depth > 8 {
                let source_ptr = src.planes[plane.as_usize()]
                    .as_ref()
                    .ok_or(AvifError::NoContent)?
                    .ptr16();
                let dst_ptr = self.planes[plane.as_usize()]
                    .as_mut()
                    .ok_or(AvifError::NoContent)?
                    .ptr16_mut();
                if source_ptr.is_null() || dst_ptr.is_null() {
                    return AvifError::no_content();
                }
                let src_stride = i32_from_u32(src_pd.row_bytes / 2)?;
                let dst_stride = i32_from_u32(dst_pd.row_bytes / 2)?;
                // SAFETY: `validate_plane` and the null check above verified that `source_ptr`
                // points to a valid, 2-byte aligned `u16` plane of `src_width x src_height` with
                // `src_stride` `u16` elements per row, and `self.allocate_planes` allocated a
                // fresh, disjoint, 2-byte aligned `Pixels::Buffer16` at `dst_ptr` covering
                // `dst_width x dst_height` with `dst_stride` `u16` elements per row.
                #[allow(clippy::let_unit_value)]
                let _ret = unsafe {
                    ScalePlane_12(
                        source_ptr,
                        src_stride,
                        src_width,
                        src_height,
                        dst_ptr,
                        dst_stride,
                        dst_width,
                        dst_height,
                        FilterMode_kFilterBox,
                    )
                };
            } else {
                let source_ptr = src.planes[plane.as_usize()]
                    .as_ref()
                    .ok_or(AvifError::NoContent)?
                    .ptr();
                let dst_ptr = self.planes[plane.as_usize()]
                    .as_mut()
                    .ok_or(AvifError::NoContent)?
                    .ptr_mut();
                if source_ptr.is_null() || dst_ptr.is_null() {
                    return AvifError::no_content();
                }
                let src_stride = i32_from_u32(src_pd.row_bytes)?;
                let dst_stride = i32_from_u32(dst_pd.row_bytes)?;
                let filter = if cfg!(feature = "android_mediacodec") {
                    // On Android:
                    // * For chroma planes: always use bilinear filter for chroma planes.
                    //   libyuv uses only bilinear filtering for NV12Scale. So in order to
                    //   be consistent between Nv12 and Yuv420, we need to use bilinear
                    //   filter here so that the output looks the same irrespective of the
                    //   decoder's color format.
                    // * For luma planes: use bilinear filter when the scale factor is
                    //   large. This avoids some grid like artifacts when scaling down by a
                    //   large factor ( >= 25 ). See b/447547190.
                    if *plane == Plane::U || *plane == Plane::V || scale_factor >= 25 {
                        FilterMode_kFilterBilinear
                    } else {
                        FilterMode_kFilterBox
                    }
                } else {
                    FilterMode_kFilterBox
                };
                // SAFETY: `validate_plane` and the null check above verified that `source_ptr`
                // points to a valid `u8` plane of `src_width x src_height` with `src_stride` bytes
                // per row, and `self.allocate_planes` allocated a fresh, disjoint `Pixels::Buffer`
                // at `dst_ptr` covering `dst_width x dst_height` with `dst_stride` bytes per row.
                #[allow(clippy::let_unit_value)]
                let _ret = unsafe {
                    ScalePlane(
                        source_ptr, src_stride, src_width, src_height, dst_ptr, dst_stride,
                        dst_width, dst_height, filter,
                    )
                };
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::pixels::Pixels;
    use test_case::test_case;
    use test_case::test_matrix;

    #[test_matrix([PixelFormat::Yuv444, PixelFormat::Yuv422, PixelFormat::Yuv420, PixelFormat::Yuv400], [false, true], [false, true])]
    fn scale(yuv_format: PixelFormat, use_alpha: bool, is_pointer_input: bool) {
        let mut yuv = image::Image {
            width: 2,
            height: 2,
            depth: 8,
            yuv_format,
            ..Default::default()
        };

        let planes: &[Plane] = match (yuv_format, use_alpha) {
            (PixelFormat::Yuv400, false) => &[Plane::Y],
            (PixelFormat::Yuv400, true) => &[Plane::Y, Plane::A],
            (_, false) => &YUV_PLANES,
            (_, true) => &ALL_PLANES,
        };
        let mut values = [
            10, 20, //
            30, 40,
        ];
        for plane in planes {
            yuv.planes[plane.as_usize()] = if is_pointer_input {
                Pixels::from_raw_pointer(values.as_mut_ptr(), 8, 2, 2).ok()
            } else {
                Some(Pixels::Buffer(values.to_vec()))
            };
            yuv.row_bytes[plane.as_usize()] = 2;
        }
        let categories: &[Category] =
            if use_alpha { &[Category::Color, Category::Alpha] } else { &[Category::Color] };
        for category in categories {
            // Scale will update the width and height when scaling YUV planes. Reset it back before
            // calling it again.
            yuv.width = 2;
            yuv.height = 2;
            assert!(yuv.scale(4, 4, *category).is_ok());
        }
        for plane in planes {
            let expected_samples: &[u8] = match (yuv_format, plane) {
                (PixelFormat::Yuv422, Plane::U | Plane::V) => &[
                    10, 10, //
                    10, 10, //
                    30, 30, //
                    30, 30,
                ],
                (PixelFormat::Yuv420, Plane::U | Plane::V) => &[
                    10, 10, //
                    10, 10,
                ],
                (_, _) => &[
                    10, 13, 18, 20, //
                    15, 18, 23, 25, //
                    25, 28, 33, 35, //
                    30, 33, 38, 40,
                ],
            };
            match &yuv.planes[plane.as_usize()] {
                Some(Pixels::Buffer(samples)) => {
                    assert_eq!(*samples, expected_samples)
                }
                _ => panic!(),
            }
        }
    }

    #[test_case(PixelFormat::AndroidNv12, 8 ; "nv12")]
    #[test_case(PixelFormat::AndroidP010, 16 ; "p010")]
    fn scale_semi_planar_odd_dimension(yuv_format: PixelFormat, depth: u8) -> AvifResult<()> {
        let mut image = image::Image {
            width: 99,
            height: 49,
            depth,
            yuv_format,
            ..Default::default()
        };
        assert_eq!(image.width(Plane::U), 100);
        image.allocate_planes(Category::Color)?;
        assert!(image.scale(49, 24, Category::Color).is_ok());
        assert_eq!(image.width, 49);
        assert_eq!(image.height, 24);
        Ok(())
    }

    #[test]
    fn scale_android_p010_alpha_only() -> AvifResult<()> {
        let mut image = image::Image {
            width: 4,
            height: 4,
            depth: 16,
            yuv_format: PixelFormat::AndroidP010,
            ..Default::default()
        };
        image.allocate_planes(Category::Alpha)?;
        for y in 0..image.height {
            for pixel in image.row16_mut(Plane::A, y)? {
                *pixel = 512 << 6;
            }
        }
        image.scale(2, 2, Category::Alpha)?;
        assert_eq!(image.width, 2);
        assert_eq!(image.height, 2);
        assert_eq!(image.depth, 10);
        assert_eq!(image.yuv_format, PixelFormat::Yuv420);
        for y in 0..image.height {
            assert!(image.row16(Plane::A, y)?.iter().all(|&pixel| pixel == 512));
        }
        Ok(())
    }

    #[test_case(8 ; "8bit")]
    #[test_case(10 ; "10bit")]
    fn scale_without_y_plane(depth: u8) -> AvifResult<()> {
        let mut image = image::Image {
            width: 4,
            height: 4,
            depth,
            yuv_format: PixelFormat::Yuv444,
            ..Default::default()
        };
        image.allocate_planes(Category::Color)?;
        image.planes[Plane::Y.as_usize()] = None;
        image.row_bytes[Plane::Y.as_usize()] = 0;
        assert!(image.scale(2, 2, Category::Color).is_err());
        assert_eq!(image.width, 4);
        assert_eq!(image.height, 4);
        Ok(())
    }
}
