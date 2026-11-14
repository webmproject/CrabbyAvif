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

use crate::decoder::ProgressiveState;
use crate::internal_utils::*;
use crate::reformat::coeffs::*;
use crate::utils::clap::*;
use crate::utils::pixels::*;
use crate::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Plane {
    Y = 0,
    U = 1,
    V = 2,
    A = 3,
}

impl From<usize> for Plane {
    fn from(plane: usize) -> Self {
        match plane {
            1 => Plane::U,
            2 => Plane::V,
            3 => Plane::A,
            _ => Plane::Y,
        }
    }
}

impl Plane {
    pub(crate) fn as_usize(&self) -> usize {
        match self {
            Plane::Y => 0,
            Plane::U => 1,
            Plane::V => 2,
            Plane::A => 3,
        }
    }
}

/// cbindgen:ignore
pub const MAX_PLANE_COUNT: usize = 4;
pub const YUV_PLANES: [Plane; 3] = [Plane::Y, Plane::U, Plane::V];
pub const A_PLANE: [Plane; 1] = [Plane::A];
pub const ALL_PLANES: [Plane; MAX_PLANE_COUNT] = [Plane::Y, Plane::U, Plane::V, Plane::A];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
// VideoFullRangeFlag as specified in ISO/IEC 23091-2/ITU-T H.273.
pub enum YuvRange {
    Limited = 0,
    #[default]
    Full = 1,
}

#[derive(Default)]
pub struct Image {
    // Image dimensions in pixels.
    pub width: u32,
    pub height: u32,
    // Bit depth of the image per channel (8, 10, 12, 16, or 32).
    pub depth: u8,

    // Format of the luma and chroma planes.
    pub yuv_format: PixelFormat,
    pub yuv_range: YuvRange,
    pub chroma_sample_position: ChromaSamplePosition,

    // Presence and format of the alpha plane.
    pub alpha_present: bool,
    pub alpha_premultiplied: bool,

    // Number of bytes per row for each plane.
    pub row_bytes: [u32; MAX_PLANE_COUNT],

    // Luma, chroma, and alpha planes.
    pub planes: [Option<Pixels>; MAX_PLANE_COUNT],

    // CICP triplet.
    pub color_primaries: ColorPrimaries,
    pub transfer_characteristics: TransferCharacteristics,
    pub matrix_coefficients: MatrixCoefficients,

    // Render-related metadata.
    pub clli: Option<ContentLightLevelInformation>,
    pub pasp: Option<PixelAspectRatio>,
    pub clap: Option<CleanAperture>,
    pub irot_angle: Option<u8>,
    pub imir_axis: Option<u8>,

    // Sequences of metadata bytes.
    pub exif: Vec<u8>,
    pub icc: Vec<u8>,
    pub xmp: Vec<u8>,

    // These fields are only filled by the decoder and are ignored by the encoder.
    pub image_sequence_track_present: bool,
    pub progressive_state: ProgressiveState,
}

pub struct PlaneData {
    pub width: u32,
    pub height: u32,
    pub row_bytes: u32,
    pub pixel_size: u32,
}

impl Image {
    // Creates an instance with all the properties of self but with pixels and
    // Exif, XMP, ICC metadata left unallocated.
    pub(crate) fn shallow_clone(&self) -> Self {
        Self {
            // Fields requiring dynamic allocation.
            row_bytes: [0; MAX_PLANE_COUNT],
            planes: [const { None }; MAX_PLANE_COUNT],
            exif: Vec::new(),
            icc: Vec::new(),
            xmp: Vec::new(),

            // All other field values can be copied.
            ..*self
        }
    }

    pub fn try_deep_clone(&self) -> AvifResult<Self> {
        let mut image = self.shallow_clone();
        for plane in ALL_PLANES.iter().filter(|p| self.has_plane(**p)) {
            // Allocate and copy row by row to avoid carrying large row padding
            // over if any.
            image.allocate_plane(*plane)?;
            for y in 0..self.height(*plane) as u32 {
                if self.depth <= 8 {
                    image
                        .row_mut(*plane, y)?
                        .copy_from_slice(self.row(*plane, y)?);
                } else {
                    image
                        .row16_mut(*plane, y)?
                        .copy_from_slice(self.row16(*plane, y)?);
                }
            }
        }
        image.exif = self.exif.try_clone()?;
        image.icc = self.icc.try_clone()?;
        image.xmp = self.xmp.try_clone()?;
        Ok(image)
    }

    pub(crate) fn is_supported_depth(depth: u8) -> bool {
        match depth {
            8 | 10 | 12 | 16 => true,
            #[cfg(feature = "satofloat")]
            32 => true,
            _ => false,
        }
    }

    pub(crate) fn depth_valid(&self) -> bool {
        Self::is_supported_depth(self.depth)
    }

    pub fn max_channel(&self) -> u16 {
        if !self.depth_valid() {
            0
        } else {
            ((1i32 << self.depth) - 1) as u16
        }
    }

    pub(crate) fn max_channel_f(&self) -> f32 {
        self.max_channel() as f32
    }

    pub fn has_plane(&self, plane: Plane) -> bool {
        let plane_index = plane.as_usize();
        if self.planes[plane_index].is_none() || self.row_bytes[plane_index] == 0 {
            return false;
        }
        self.planes[plane_index].unwrap_ref().has_data()
    }

    pub fn has_alpha(&self) -> bool {
        self.has_plane(Plane::A)
    }

    pub(crate) fn has_same_properties(&self, other: &Image) -> bool {
        self.width == other.width && self.height == other.height && self.depth == other.depth
    }

    // TODO: b/392112497 - remove this annotation once encoder feature is enabled by default.
    #[allow(dead_code)]
    pub(crate) fn has_same_cicp(&self, other: &Image) -> bool {
        self.depth == other.depth
            && self.yuv_format == other.yuv_format
            && self.yuv_range == other.yuv_range
            && self.chroma_sample_position == other.chroma_sample_position
            && self.color_primaries == other.color_primaries
            && self.transfer_characteristics == other.transfer_characteristics
            && self.matrix_coefficients == other.matrix_coefficients
    }

    pub fn has_same_properties_and_cicp(&self, other: &Image) -> bool {
        self.has_same_properties(other) && self.has_same_cicp(other)
    }

    pub(crate) fn plane_width(yuv_format: PixelFormat, plane: Plane, luma_width: u32) -> usize {
        let width = luma_width as usize;
        match plane {
            Plane::Y | Plane::A => width,
            Plane::U => match yuv_format {
                PixelFormat::Yuv444 => width,
                PixelFormat::AndroidP010 | PixelFormat::AndroidNv12 | PixelFormat::AndroidNv21 => {
                    width.div_ceil(2) * 2
                }
                PixelFormat::Yuv420 | PixelFormat::Yuv422 => width.div_ceil(2),
                PixelFormat::None | PixelFormat::Yuv400 => 0,
            },
            Plane::V => match yuv_format {
                PixelFormat::Yuv444 => width,
                PixelFormat::Yuv420 | PixelFormat::Yuv422 => width.div_ceil(2),
                PixelFormat::None
                | PixelFormat::Yuv400
                | PixelFormat::AndroidP010
                | PixelFormat::AndroidNv12
                | PixelFormat::AndroidNv21 => 0,
            },
        }
    }

    pub(crate) fn plane_height(yuv_format: PixelFormat, plane: Plane, luma_height: u32) -> usize {
        let height = luma_height as usize;
        match plane {
            Plane::Y | Plane::A => height,
            Plane::U => match yuv_format {
                PixelFormat::Yuv444 | PixelFormat::Yuv422 => height,
                PixelFormat::Yuv420
                | PixelFormat::AndroidP010
                | PixelFormat::AndroidNv12
                | PixelFormat::AndroidNv21 => height.div_ceil(2),
                PixelFormat::None | PixelFormat::Yuv400 => 0,
            },
            Plane::V => match yuv_format {
                PixelFormat::Yuv444 | PixelFormat::Yuv422 => height,
                PixelFormat::Yuv420 => height.div_ceil(2),
                PixelFormat::None
                | PixelFormat::Yuv400
                | PixelFormat::AndroidP010
                | PixelFormat::AndroidNv12
                | PixelFormat::AndroidNv21 => 0,
            },
        }
    }

    pub fn width(&self, plane: Plane) -> usize {
        Self::plane_width(self.yuv_format, plane, self.width)
    }

    pub fn height(&self, plane: Plane) -> usize {
        Self::plane_height(self.yuv_format, plane, self.height)
    }

    pub fn plane_data(&self, plane: Plane) -> Option<PlaneData> {
        if !self.has_plane(plane) {
            return None;
        }
        Some(PlaneData {
            width: self.width(plane) as u32,
            height: self.height(plane) as u32,
            row_bytes: self.row_bytes[plane.as_usize()],
            pixel_size: if self.depth == 8 { 1 } else { 2 },
        })
    }

    pub fn row(&self, plane: Plane, row: u32) -> AvifResult<&[u8]> {
        let plane_data = self.plane_data(plane).ok_or(AvifError::NoContent)?;
        let start = checked_mul!(row, plane_data.row_bytes)?;
        self.planes[plane.as_usize()]
            .unwrap_ref()
            .slice(start, plane_data.width)
    }

    pub fn row_mut(&mut self, plane: Plane, row: u32) -> AvifResult<&mut [u8]> {
        let plane_data = self.plane_data(plane).ok_or(AvifError::NoContent)?;
        let start = checked_mul!(row, plane_data.row_bytes)?;
        self.planes[plane.as_usize()]
            .unwrap_mut()
            .slice_mut(start, plane_data.width)
    }

    pub fn row16(&self, plane: Plane, row: u32) -> AvifResult<&[u16]> {
        let plane_data = self.plane_data(plane).ok_or(AvifError::NoContent)?;
        let start = checked_mul!(row, plane_data.row_bytes / 2)?;
        self.planes[plane.as_usize()]
            .unwrap_ref()
            .slice16(start, plane_data.width)
    }

    pub fn row16_mut(&mut self, plane: Plane, row: u32) -> AvifResult<&mut [u16]> {
        let plane_data = self.plane_data(plane).ok_or(AvifError::NoContent)?;
        let start = checked_mul!(row, plane_data.row_bytes / 2)?;
        self.planes[plane.as_usize()]
            .unwrap_mut()
            .slice16_mut(start, plane_data.width)
    }

    // Replaces the given planes by owned copies of their region described by `rect` (with x and y
    // rounded down to the chroma subsampling of the image). The image dimensions are not modified.
    // self is left untouched if any of the planes cannot be cropped.
    pub(crate) fn crop_planes_in_place(
        &mut self,
        planes: &[Plane],
        rect: &CropRect,
    ) -> AvifResult<()> {
        let aligned = rect.align_to_pixel_format(self.yuv_format);
        if !aligned.is_within(self.width, self.height) {
            return AvifError::invalid_argument();
        }
        // Holds the cropped planes until all of them are successfully cropped.
        let mut dst = self.shallow_clone();
        dst.width = aligned.width;
        dst.height = aligned.height;
        for &plane in planes {
            if !self.has_plane(plane) {
                continue;
            }
            let (plane_x, plane_y) = if matches!(plane, Plane::Y | Plane::A) {
                (usize_from_u32(aligned.x)?, aligned.y)
            } else {
                (
                    usize_from_u32(self.yuv_format.apply_chroma_shift_x(aligned.x))?,
                    self.yuv_format.apply_chroma_shift_y(aligned.y),
                )
            };
            dst.allocate_plane(plane)?;
            if !dst.has_plane(plane) {
                continue;
            }
            let plane_x_end = checked_add!(plane_x, dst.width(plane))?;
            for y in 0..u32_from_usize(dst.height(plane))? {
                let src_y = checked_add!(plane_y, y)?;
                if self.depth == 8 {
                    let src_row = self
                        .row(plane, src_y)?
                        .get(plane_x..plane_x_end)
                        .ok_or(AvifError::InvalidArgument)?;
                    dst.row_mut(plane, y)?.copy_from_slice(src_row);
                } else {
                    let src_row = self
                        .row16(plane, src_y)?
                        .get(plane_x..plane_x_end)
                        .ok_or(AvifError::InvalidArgument)?;
                    dst.row16_mut(plane, y)?.copy_from_slice(src_row);
                }
            }
        }
        for &plane in planes {
            if self.has_plane(plane) {
                let plane_index = plane.as_usize();
                self.planes[plane_index] = dst.planes[plane_index].take();
                self.row_bytes[plane_index] = dst.row_bytes[plane_index];
            }
        }
        Ok(())
    }

    pub(crate) fn crop_in_place(&mut self, rect: &CropRect) -> AvifResult<()> {
        if rect.covers(self.width, self.height) {
            self.clap = None;
            return Ok(());
        }
        let aligned = rect.align_to_pixel_format(self.yuv_format);
        self.crop_planes_in_place(&ALL_PLANES, &aligned)?;
        self.width = aligned.width;
        self.height = aligned.height;
        self.clap = None;
        Ok(())
    }

    // Returns a view of the region of self described by `rect`. The returned image has the same
    // properties as self and points to the pixel values of self (no pixel data is copied). Exif,
    // XMP and ICC data are not copied. `rect` must be a valid rectangle within the bounds of self,
    // with x and y aligned to the chroma subsampling of the image.
    //
    // The returned image must not outlive self and the pixel buffers of self must not be
    // reallocated or freed while the view is alive.
    //
    // This function may not be used in all configurations.
    #[allow(dead_code)]
    pub(crate) fn view(&self, rect: &CropRect) -> AvifResult<Image> {
        if !rect.is_valid(self.width, self.height, self.yuv_format) {
            return AvifError::invalid_argument();
        }
        let mut image = self.shallow_clone();
        image.width = rect.width;
        image.height = rect.height;
        for plane in ALL_PLANES {
            if !self.has_plane(plane) {
                continue;
            }
            let (x, y) = if plane == Plane::Y || plane == Plane::A {
                (usize_from_u32(rect.x)?, rect.y)
            } else {
                (
                    usize_from_u32(image.yuv_format.apply_chroma_shift_x(rect.x))?,
                    image.yuv_format.apply_chroma_shift_y(rect.y),
                )
            };
            let ptr = if image.depth == 8 {
                let row = self.row(plane, y)?;
                row.get(x..).ok_or(AvifError::InvalidArgument)?.as_ptr() as *mut u8
            } else {
                let row = self.row16(plane, y)?;
                row.get(x..).ok_or(AvifError::InvalidArgument)?.as_ptr() as *mut u8
            };
            image.planes[plane.as_usize()] = Some(Pixels::from_raw_pointer(
                ptr,
                image.depth as _,
                u32_from_usize(image.height(plane))?,
                self.row_bytes[plane.as_usize()],
            )?);
            image.row_bytes[plane.as_usize()] = self.row_bytes[plane.as_usize()];
        }
        Ok(image)
    }

    // Returns a view with the same image properties as self and pointing to
    // the pixel values of self. Copies all Exif, XMP and ICC data if any.
    #[cfg(feature = "cli")]
    pub fn cropped_image(&self) -> AvifResult<Image> {
        match self.clap {
            Some(clap) => {
                let rect = CropRect::create_from(&clap, self.width, self.height, self.yuv_format)?;
                let mut image = self.view(&rect)?;
                image.exif = self.exif.try_clone()?;
                image.xmp = self.xmp.try_clone()?;
                image.icc = self.icc.try_clone()?;
                Ok(image)
            }
            None => Err(AvifError::InvalidArgument),
        }
    }

    #[cfg(feature = "libyuv")]
    pub(crate) fn plane_ptrs(&self) -> [*const u8; 4] {
        ALL_PLANES.map(|x| {
            if self.has_plane(x) {
                self.planes[x.as_usize()].unwrap_ref().ptr()
            } else {
                std::ptr::null()
            }
        })
    }

    #[cfg(any(feature = "libyuv", feature = "sharpyuv"))]
    pub(crate) fn plane_ptrs_mut(&mut self) -> [*mut u8; 4] {
        ALL_PLANES.map(|x| {
            if self.has_plane(x) {
                self.planes[x.as_usize()].unwrap_mut().ptr_mut_generic()
            } else {
                std::ptr::null_mut()
            }
        })
    }

    #[cfg(feature = "libyuv")]
    pub(crate) fn plane16_ptrs(&self) -> [*const u16; 4] {
        ALL_PLANES.map(|x| {
            if self.has_plane(x) {
                self.planes[x.as_usize()].unwrap_ref().ptr16()
            } else {
                std::ptr::null()
            }
        })
    }

    #[cfg(any(feature = "libyuv", feature = "jpegxl"))]
    pub(crate) fn plane_row_bytes(&self) -> AvifResult<[i32; 4]> {
        let mut row_bytes = [0i32; 4];
        for plane in ALL_PLANES {
            if self.has_plane(plane) {
                row_bytes[plane.as_usize()] = i32_from_u32(self.row_bytes[plane.as_usize()])?;
            }
        }
        Ok(row_bytes)
    }

    pub(crate) fn free_planes(&mut self, planes: &[Plane]) {
        for plane in planes {
            let plane = plane.as_usize();
            self.planes[plane] = None;
            self.row_bytes[plane] = 0;
        }
    }

    #[cfg(any(feature = "dav1d", feature = "native_av1"))]
    pub(crate) fn clear_chroma_planes(&mut self) {
        self.free_planes(&[Plane::U, Plane::V])
    }

    fn allocate_plane(&mut self, plane: Plane) -> AvifResult<()> {
        // Rust has no idiomatic way to allocate memory without initializing it.
        const DEFAULT_VALUE: u16 = 0; // The default value does not matter.
        self.allocate_plane_with_default_value(plane, DEFAULT_VALUE)
    }

    pub(crate) fn allocate_plane_with_default_value(
        &mut self,
        plane: Plane,
        default_value: u16,
    ) -> AvifResult<()> {
        let plane_index = plane.as_usize();
        let width = round2_usize(self.width(plane));
        let plane_size = checked_mul!(width, round2_usize(self.height(plane)))?;
        if plane_size == 0 {
            self.planes[plane_index] = None;
            self.row_bytes[plane_index] = 0;
        } else {
            self.planes[plane_index] = Some(if self.depth == 8 {
                Pixels::Buffer(Vec::new())
            } else {
                Pixels::Buffer16(Vec::new())
            });
            let pixels = self.planes[plane_index].unwrap_mut();
            pixels.resize(plane_size, default_value)?;
            let pixel_size: usize = if self.depth == 8 { 1 } else { 2 };
            self.row_bytes[plane_index] = u32_from_usize(checked_mul!(width, pixel_size)?)?;
        }
        Ok(())
    }

    pub(crate) fn allocate_planes_with_default_values(
        &mut self,
        category: Category,
        default_values: [u16; 4],
    ) -> AvifResult<()> {
        for plane in category.planes() {
            self.allocate_plane_with_default_value(*plane, default_values[plane.as_usize()])?;
        }
        Ok(())
    }

    pub fn allocate_planes(&mut self, category: Category) -> AvifResult<()> {
        self.allocate_planes_with_default_values(category, [0, 0, 0, self.max_channel()])
    }

    #[cfg(feature = "encoder")]
    pub(crate) fn copy_and_pad(&mut self, image: &Image) -> AvifResult<()> {
        if image.width > self.width || image.height > self.height {
            return AvifError::invalid_argument();
        }
        self.allocate_planes(Category::Color)?;
        if image.has_alpha() {
            self.allocate_planes(Category::Alpha)?;
        }
        for plane in ALL_PLANES {
            let src_plane = match image.plane_data(plane) {
                Some(pd) => pd,
                None => continue,
            };
            if self.depth == 8 {
                for y in 0..src_plane.height {
                    let src_row = image.row(plane, y)?;
                    let dst_row = self.row_mut(plane, y)?;
                    let src_len = src_row.len();
                    let dst_slice = dst_row
                        .get_mut(0..src_len)
                        .ok_or(AvifError::UnknownError("".into()))?;
                    dst_slice.copy_from_slice(src_row);
                    let last_pixel = *src_row.last().ok_or(AvifError::UnknownError("".into()))?;
                    let dst_slice = dst_row
                        .get_mut(src_len..)
                        .ok_or(AvifError::UnknownError("".into()))?;
                    dst_slice.fill(last_pixel);
                }
            } else {
                for y in 0..src_plane.height {
                    let src_row = image.row16(plane, y)?;
                    let dst_row = self.row16_mut(plane, y)?;
                    let src_len = src_row.len();
                    let dst_slice = dst_row
                        .get_mut(0..src_len)
                        .ok_or(AvifError::UnknownError("".into()))?;
                    dst_slice.copy_from_slice(src_row);
                    let last_pixel = *src_row.last().ok_or(AvifError::UnknownError("".into()))?;
                    let dst_slice = dst_row
                        .get_mut(src_len..)
                        .ok_or(AvifError::UnknownError("".into()))?;
                    dst_slice.fill(last_pixel);
                }
            }
        }
        Ok(())
    }

    pub(crate) fn convert_rgba16_to_yuva(&self, rgba: [u16; 4]) -> [u16; 4] {
        let r = rgba[0] as f32 / 65535.0;
        let g = rgba[1] as f32 / 65535.0;
        let b = rgba[2] as f32 / 65535.0;
        let coeffs = calculate_yuv_coefficients(self.color_primaries, self.matrix_coefficients);
        let y = coeffs[0] * r + coeffs[1] * g + coeffs[2] * b;
        let u = (b - y) / (2.0 * (1.0 - coeffs[2]));
        let v = (r - y) / (2.0 * (1.0 - coeffs[0]));
        let uv_bias = (1 << (self.depth - 1)) as f32;
        let max_channel = self.max_channel_f();
        [
            (y * max_channel).clamp(0.0, max_channel) as u16,
            (u * max_channel + uv_bias).clamp(0.0, max_channel) as u16,
            (v * max_channel + uv_bias).clamp(0.0, max_channel) as u16,
            ((rgba[3] as f32) / 65535.0 * max_channel).round() as u16,
        ]
    }

    // This function is not used in all configurations.
    #[allow(dead_code)]
    pub fn is_opaque(&self) -> AvifResult<bool> {
        let Some(plane_data) = self.plane_data(Plane::A) else {
            return Ok(true);
        };
        let opaque_value = self.max_channel();
        if self.depth == 8 {
            let opaque = opaque_value as u8;
            #[cfg(target_arch = "x86_64")]
            {
                if std::arch::is_x86_feature_detected!("avx2") {
                    // SAFETY: `is_x86_feature_detected!("avx2")` verified AVX2 support.
                    return unsafe { self.is_opaque_8_avx2(plane_data.height, opaque) };
                }
                // SAFETY: SSE2 is baseline on `x86_64`.
                unsafe { self.is_opaque_8_sse2(plane_data.height, opaque) }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                for y in 0..plane_data.height {
                    let row = self.row(Plane::A, y)?;
                    if !row.iter().all(|pixel| *pixel == opaque) {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
        } else {
            #[cfg(target_arch = "x86_64")]
            {
                if std::arch::is_x86_feature_detected!("avx2") {
                    // SAFETY: `is_x86_feature_detected!("avx2")` verified AVX2 support.
                    return unsafe { self.is_opaque_16_avx2(plane_data.height, opaque_value) };
                }
                // SAFETY: SSE2 is baseline on `x86_64`.
                unsafe { self.is_opaque_16_sse2(plane_data.height, opaque_value) }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                for y in 0..plane_data.height {
                    let row = self.row16(Plane::A, y)?;
                    if !row.iter().all(|pixel| *pixel == opaque_value) {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
        }
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn is_opaque_8_avx2(&self, height: u32, opaque: u8) -> AvifResult<bool> {
        use std::arch::x86_64::*;
        let target256 = _mm256_set1_epi8(opaque as i8);
        let target128 = _mm256_castsi256_si128(target256);
        for y in 0..height {
            let row = self.row(Plane::A, y)?;
            let len = row.len();
            let ptr = row.as_ptr();
            let mut i = 0usize;
            while i + 32 <= len {
                // SAFETY: `i + 32 <= len`, so `ptr.add(i)` is valid for an unaligned 32-byte read.
                let v = unsafe { _mm256_loadu_si256(ptr.add(i).cast()) };
                if _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, target256)) != -1 {
                    return Ok(false);
                }
                i += 32;
            }
            if i + 16 <= len {
                // SAFETY: `i + 16 <= len`, so `ptr.add(i)` is valid for an unaligned 16-byte read.
                let v = unsafe { _mm_loadu_si128(ptr.add(i).cast()) };
                if _mm_movemask_epi8(_mm_cmpeq_epi8(v, target128)) != 0xFFFF {
                    return Ok(false);
                }
                i += 16;
            }
            if let Some(tail) = row.get(i..) {
                if !tail.iter().all(|&pixel| pixel == opaque) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// # Safety
    /// The caller must ensure the CPU supports `sse2` (baseline on `x86_64`).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn is_opaque_8_sse2(&self, height: u32, opaque: u8) -> AvifResult<bool> {
        use std::arch::x86_64::*;
        let target128 = _mm_set1_epi8(opaque as i8);
        for y in 0..height {
            let row = self.row(Plane::A, y)?;
            let len = row.len();
            let ptr = row.as_ptr();
            let mut i = 0usize;
            while i + 16 <= len {
                // SAFETY: `i + 16 <= len`, so `ptr.add(i)` is valid for an unaligned 16-byte read.
                let v = unsafe { _mm_loadu_si128(ptr.add(i).cast()) };
                if _mm_movemask_epi8(_mm_cmpeq_epi8(v, target128)) != 0xFFFF {
                    return Ok(false);
                }
                i += 16;
            }
            if let Some(tail) = row.get(i..) {
                if !tail.iter().all(|&pixel| pixel == opaque) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// # Safety
    /// The caller must ensure the CPU supports `avx2`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn is_opaque_16_avx2(&self, height: u32, opaque: u16) -> AvifResult<bool> {
        use std::arch::x86_64::*;
        let target256 = _mm256_set1_epi16(opaque as i16);
        let target128 = _mm256_castsi256_si128(target256);
        for y in 0..height {
            let row = self.row16(Plane::A, y)?;
            let len = row.len();
            let ptr = row.as_ptr();
            let mut i = 0usize;
            while i + 16 <= len {
                // SAFETY: `i + 16 <= len`, so `ptr.add(i)` is valid for a 32-byte unaligned read.
                let v = unsafe { _mm256_loadu_si256(ptr.add(i).cast()) };
                if _mm256_movemask_epi8(_mm256_cmpeq_epi16(v, target256)) != -1 {
                    return Ok(false);
                }
                i += 16;
            }
            if i + 8 <= len {
                // SAFETY: `i + 8 <= len`, so `ptr.add(i)` is valid for a 16-byte unaligned read.
                let v = unsafe { _mm_loadu_si128(ptr.add(i).cast()) };
                if _mm_movemask_epi8(_mm_cmpeq_epi16(v, target128)) != 0xFFFF {
                    return Ok(false);
                }
                i += 8;
            }
            if let Some(tail) = row.get(i..) {
                if !tail.iter().all(|&pixel| pixel == opaque) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// # Safety
    /// The caller must ensure the CPU supports `sse2` (baseline on `x86_64`).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn is_opaque_16_sse2(&self, height: u32, opaque: u16) -> AvifResult<bool> {
        use std::arch::x86_64::*;
        let target128 = _mm_set1_epi16(opaque as i16);
        for y in 0..height {
            let row = self.row16(Plane::A, y)?;
            let len = row.len();
            let ptr = row.as_ptr();
            let mut i = 0usize;
            while i + 8 <= len {
                // SAFETY: `i + 8 <= len`, so `ptr.add(i)` is valid for a 16-byte unaligned read.
                let v = unsafe { _mm_loadu_si128(ptr.add(i).cast()) };
                if _mm_movemask_epi8(_mm_cmpeq_epi16(v, target128)) != 0xFFFF {
                    return Ok(false);
                }
                i += 8;
            }
            if let Some(tail) = row.get(i..) {
                if !tail.iter().all(|&pixel| pixel == opaque) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    pub fn fill_plane_with_value(&mut self, plane: Plane, value: u16) -> AvifResult<()> {
        if let Some(plane_data) = self.plane_data(plane) {
            let width = plane_data.width as usize;
            if self.depth == 8 {
                for y in 0..plane_data.height {
                    let row = self
                        .row_mut(plane, y)?
                        .get_mut(..width)
                        .ok_or(AvifError::NoContent)?;
                    row.fill(value as u8);
                }
            } else {
                for y in 0..plane_data.height {
                    let row = self
                        .row16_mut(plane, y)?
                        .get_mut(..width)
                        .ok_or(AvifError::NoContent)?;
                    row.fill(value);
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "png")]
    pub(crate) fn remove_trailing_null_from_xmp(&mut self) {
        if self.xmp.len() >= 2
            && self.xmp[self.xmp.len() - 1] == 0
            && self.xmp[self.xmp.len() - 2] != 0
        {
            self.xmp.pop();
        }
    }
}

#[cfg(feature = "satofloat")]
pub type FloatPlanes = [Option<Vec<f32>>; MAX_PLANE_COUNT];

pub(crate) trait PlanePixel: Copy + 'static {
    fn from_u16(val: u16) -> Self;
    fn into_pixels(vec: Vec<Self>) -> Pixels;
    fn byte_size() -> usize;
}

impl PlanePixel for u8 {
    fn from_u16(val: u16) -> Self {
        val as u8
    }

    fn into_pixels(vec: Vec<Self>) -> Pixels {
        Pixels::Buffer(vec)
    }

    fn byte_size() -> usize {
        1
    }
}

impl PlanePixel for u16 {
    fn from_u16(val: u16) -> Self {
        val
    }

    fn into_pixels(vec: Vec<Self>) -> Pixels {
        Pixels::Buffer16(vec)
    }

    fn byte_size() -> usize {
        2
    }
}

/// Sound builder for a YUV/A plane allocation without upfront zero-fill.
///
/// Safety invariants maintained by `new` until `finish_into`:
/// - `total_size == stride_elems * padded_height`, where `stride_elems == round2_usize(width)`
///   and `padded_height == round2_usize(height)`.
/// - `vec.len() == 0` and `vec.capacity() >= total_size`.
/// - `row_bytes == stride_elems * T::byte_size()`.
///
/// Only `finish_into` fills trailing `round2_usize` border padding and calls
/// `Vec::set_len(total_size)`.
pub(crate) struct UninitPlane<T: PlanePixel> {
    vec: Vec<T>,
    width: usize,
    height: usize,
    stride_elems: usize,
    padded_height: usize,
    row_bytes: u32,
    total_size: usize,
}

pub(crate) type UninitPlane8 = UninitPlane<u8>;
pub(crate) type UninitPlane16 = UninitPlane<u16>;

// Some methods on `UninitPlane` may not be used in all target/feature configurations.
#[allow(dead_code)]
impl<T: PlanePixel> UninitPlane<T> {
    pub(crate) fn new(width: usize, height: usize) -> AvifResult<Self> {
        let stride_elems = round2_usize(width);
        let padded_height = round2_usize(height);
        let total_size = checked_mul!(stride_elems, padded_height)?;
        let row_bytes = u32_from_usize(checked_mul!(stride_elems, T::byte_size())?)?;
        let vec = if total_size == 0 { Vec::new() } else { create_vec_exact(total_size)? };
        Ok(Self {
            vec,
            width,
            height,
            stride_elems,
            padded_height,
            row_bytes,
            total_size,
        })
    }

    pub(crate) fn for_image(image: &Image, plane: Plane) -> AvifResult<Self> {
        Self::new(image.width(plane), image.height(plane))
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }

    pub(crate) fn height(&self) -> usize {
        self.height
    }

    pub(crate) fn stride_elems(&self) -> usize {
        self.stride_elems
    }

    pub(crate) fn row_bytes_i32(&self) -> AvifResult<i32> {
        i32_from_u32(self.row_bytes)
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.vec.spare_capacity_mut().as_mut_ptr().cast()
    }

    pub(crate) fn spare_all_mut(&mut self) -> AvifResult<&mut [std::mem::MaybeUninit<T>]> {
        self.vec
            .spare_capacity_mut()
            .get_mut(..self.total_size)
            .ok_or(AvifError::NoContent)
    }

    pub(crate) fn spare_row_mut(
        &mut self,
        row: u32,
    ) -> AvifResult<&mut [std::mem::MaybeUninit<T>]> {
        let row_usize = usize_from_u32(row)?;
        if row_usize >= self.height {
            return AvifError::no_content();
        }
        let start = row_usize * self.stride_elems;
        self.vec
            .spare_capacity_mut()
            .get_mut(start..start + self.width)
            .ok_or(AvifError::NoContent)
    }

    /// Initializes any trailing odd-width columns and odd-height rows with `pad_value`, sets the
    /// vector length to `total_size`, and attaches the finalized buffer to `image.planes[plane]`.
    ///
    /// # Safety
    /// The caller MUST have initialized every element in `(0..self.height) x (0..self.width)` (with
    /// row stride `self.stride_elems`) before calling `finish_into`.
    pub(crate) unsafe fn finish_into(
        mut self,
        image: &mut Image,
        plane: Plane,
        pad_value: u16,
    ) -> AvifResult<()> {
        if (image.depth > 8) != (T::byte_size() == 2) {
            return AvifError::invalid_argument();
        }
        let plane_index = plane.as_usize();
        if self.total_size == 0 {
            image.planes[plane_index] = None;
            image.row_bytes[plane_index] = 0;
            return Ok(());
        }
        let pad = T::from_u16(pad_value);
        let spare = self
            .vec
            .spare_capacity_mut()
            .get_mut(..self.total_size)
            .ok_or(AvifError::NoContent)?;
        if self.stride_elems > self.width {
            for row in 0..self.height {
                let row_start = row * self.stride_elems;
                let Some(pad_cols) =
                    spare.get_mut(row_start + self.width..row_start + self.stride_elems)
                else {
                    return AvifError::no_content();
                };
                for cell in pad_cols {
                    cell.write(pad);
                }
            }
        }
        if self.padded_height > self.height {
            let tail_start = self.height * self.stride_elems;
            let Some(pad_rows) = spare.get_mut(tail_start..self.total_size) else {
                return AvifError::no_content();
            };
            for cell in pad_rows {
                cell.write(pad);
            }
        }
        // SAFETY: Caller contract ensures `(0..self.height) x (0..self.width)` was initialized, and
        // the border loops above initialized all odd-width trailing columns and odd-height trailing
        // rows across the entire `0..self.total_size` prefix of `self.vec`.
        unsafe {
            self.vec.set_len(self.total_size);
        }
        image.planes[plane_index] = Some(T::into_pixels(self.vec));
        image.row_bytes[plane_index] = self.row_bytes;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_opaque_depths_odd_widths_and_trailing_mismatch() {
        let img_no_alpha = Image {
            width: 33,
            height: 3,
            depth: 8,
            yuv_format: PixelFormat::Yuv420,
            ..Default::default()
        };
        assert!(img_no_alpha.is_opaque().unwrap());

        for &depth in &[8u8, 10, 12, 16] {
            for &width in &[1u32, 7, 8, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 97] {
                for &height in &[1u32, 2, 3] {
                    let mut img = Image {
                        width,
                        height,
                        depth,
                        yuv_format: PixelFormat::Yuv444,
                        ..Default::default()
                    };
                    img.allocate_planes(Category::Alpha).unwrap();
                    let opaque = img.max_channel();
                    img.fill_plane_with_value(Plane::A, opaque).unwrap();
                    assert!(
                        img.is_opaque().unwrap(),
                        "expected opaque depth={depth} {width}x{height}"
                    );

                    // Flip first pixel, vector boundary pixels, and trailing odd pixel.
                    let mismatch_val = opaque - 1;
                    let mut test_xs = [
                        0u32,
                        width - 1,
                        width / 2,
                        15.min(width - 1),
                        16.min(width - 1),
                        31.min(width - 1),
                        32.min(width - 1),
                    ];
                    test_xs.sort_unstable();
                    for &mx in &test_xs {
                        let my = height - 1;
                        if depth == 8 {
                            img.row_mut(Plane::A, my).unwrap()[mx as usize] = mismatch_val as u8;
                            assert!(
                                !img.is_opaque().unwrap(),
                                "mismatch missed depth=8 {width}x{height} at ({mx},{my})"
                            );
                            img.row_mut(Plane::A, my).unwrap()[mx as usize] = opaque as u8;
                        } else {
                            img.row16_mut(Plane::A, my).unwrap()[mx as usize] = mismatch_val;
                            assert!(
                                !img.is_opaque().unwrap(),
                                "mismatch missed depth={depth} {width}x{height} at ({mx},{my})"
                            );
                            img.row16_mut(Plane::A, my).unwrap()[mx as usize] = opaque;
                        }
                    }
                    assert!(img.is_opaque().unwrap());
                }
            }
        }
    }

    #[test]
    fn uninit_plane_odd_dimensions_padding_and_drop_safety() {
        let mut img = Image {
            width: 5,
            height: 3,
            depth: 8,
            yuv_format: PixelFormat::Yuv444,
            ..Default::default()
        };
        let mut p8 = UninitPlane8::for_image(&img, Plane::Y).unwrap();
        assert_eq!(p8.width(), 5);
        assert_eq!(p8.height(), 3);
        assert_eq!(p8.stride_elems(), 6);
        for row in 0..3u32 {
            let dst = p8.spare_row_mut(row).unwrap();
            for (col, cell) in dst.iter_mut().enumerate() {
                cell.write((row * 10 + col as u32) as u8);
            }
        }
        // SAFETY: All 3x5 logical samples were initialized above.
        unsafe { p8.finish_into(&mut img, Plane::Y, 77).unwrap() };
        assert_eq!(img.row_bytes[Plane::Y.as_usize()], 6);
        let Some(Pixels::Buffer(buf)) = &img.planes[Plane::Y.as_usize()] else {
            panic!("expected Buffer");
        };
        assert_eq!(buf.len(), 6 * 4);
        assert_eq!(buf[5], 77);
        assert_eq!(buf[6 + 5], 77);
        assert_eq!(buf[12 + 5], 77);
        assert_eq!(&buf[18..24], &[77; 6]);

        let mut img16 = Image {
            width: 3,
            height: 3,
            depth: 10,
            yuv_format: PixelFormat::Yuv444,
            ..Default::default()
        };
        let mut p16 = UninitPlane16::for_image(&img16, Plane::A).unwrap();
        for row in 0..3u32 {
            let dst = p16.spare_row_mut(row).unwrap();
            for (col, cell) in dst.iter_mut().enumerate() {
                cell.write((row * 100 + col as u32) as u16);
            }
        }
        // SAFETY: All 3x3 logical samples were initialized above.
        unsafe { p16.finish_into(&mut img16, Plane::A, 1023).unwrap() };
        let Some(Pixels::Buffer16(buf16)) = &img16.planes[Plane::A.as_usize()] else {
            panic!("expected Buffer16");
        };
        assert_eq!(buf16.len(), 4 * 4);
        assert_eq!(buf16[3], 1023);
        assert_eq!(&buf16[12..16], &[1023; 4]);
    }
}
