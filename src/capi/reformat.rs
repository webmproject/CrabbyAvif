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

use super::image::*;
use super::types::*;

use crate::image::*;
use crate::internal_utils::*;
use crate::reformat::rgb;
use crate::utils::pixels::*;
use crate::*;

use std::os::raw::c_void;

/// cbindgen:rename-all=CamelCase
#[repr(C)]
pub struct avifRGBImage {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub format: rgb::Format,
    pub chroma_upsampling: rgb::ChromaUpsampling,
    pub chroma_downsampling: rgb::ChromaDownsampling,
    pub ignore_alpha: bool,
    pub alpha_premultiplied: bool,
    pub is_float: bool,
    pub max_threads: i32,
    pub pixels: *mut u8,
    pub row_bytes: u32,
}

impl From<rgb::Image> for avifRGBImage {
    fn from(mut rgb: rgb::Image) -> avifRGBImage {
        avifRGBImage {
            width: rgb.width,
            height: rgb.height,
            depth: rgb.depth as u32,
            format: rgb.format,
            chroma_upsampling: rgb.chroma_upsampling,
            chroma_downsampling: rgb.chroma_downsampling,
            ignore_alpha: false,
            alpha_premultiplied: rgb.premultiply_alpha,
            is_float: rgb.is_float,
            max_threads: rgb.max_threads,
            pixels: rgb.pixels_mut(),
            row_bytes: rgb.row_bytes,
        }
    }
}

impl avifRGBImage {
    fn free_pixels(&mut self) {
        if !self.pixels.is_null() {
            // SAFETY: By caller precondition on `avifRGBImage` pixel management functions, a
            // non-null `self.pixels` owned by this `avifRGBImage` was allocated via
            // `crabby_avifAlloc`, and we immediately null `self.pixels` to prevent double-free.
            unsafe {
                crabby_avifFree(self.pixels as *mut c_void);
            }
        }
        self.pixels = std::ptr::null_mut();
        self.row_bytes = 0;
    }
}

impl TryFrom<&avifRGBImage> for rgb::Image {
    type Error = AvifError;

    fn try_from(rgb: &avifRGBImage) -> AvifResult<rgb::Image> {
        let dst = rgb::Image {
            width: rgb.width,
            height: rgb.height,
            depth: rgb.depth as u8,
            format: rgb.format,
            chroma_upsampling: rgb.chroma_upsampling,
            chroma_downsampling: rgb.chroma_downsampling,
            premultiply_alpha: rgb.alpha_premultiplied,
            is_float: rgb.is_float,
            max_threads: rgb.max_threads,
            pixels: Pixels::from_raw_pointer(rgb.pixels, rgb.depth, rgb.height, rgb.row_bytes).ok(),
            row_bytes: rgb.row_bytes,
        };
        let format = match (rgb.format, rgb.ignore_alpha) {
            (rgb::Format::Rgb, _) => rgb::Format::Rgb,
            (rgb::Format::Rgba, true) => rgb::Format::Rgb,
            (rgb::Format::Rgba, false) => rgb::Format::Rgba,
            (rgb::Format::Argb, true) => rgb::Format::Rgb,
            (rgb::Format::Argb, false) => rgb::Format::Argb,
            (rgb::Format::Bgr, _) => rgb::Format::Bgr,
            (rgb::Format::Bgra, true) => rgb::Format::Bgr,
            (rgb::Format::Bgra, false) => rgb::Format::Bgra,
            (rgb::Format::Abgr, true) => rgb::Format::Bgr,
            (rgb::Format::Abgr, false) => rgb::Format::Abgr,
            (rgb::Format::Rgb565, _) => rgb::Format::Rgb565,
            (rgb::Format::Rgba1010102, _) => rgb::Format::Rgba1010102,
            (rgb::Format::Gray, _) => rgb::Format::Gray,
            (rgb::Format::GrayA, true) => rgb::Format::Gray,
            (rgb::Format::GrayA, false) => rgb::Format::GrayA,
            (rgb::Format::AGray, true) => rgb::Format::Gray,
            (rgb::Format::AGray, false) => rgb::Format::AGray,
        };
        if !dst.has_valid_pixels() {
            return Ok(dst);
        }
        dst.shuffle_channels_to(format)
    }
}

/// # Safety
/// - If `rgb` is non-null, it must point to a valid, mutable [`avifRGBImage`] for the duration of
///   this call, not aliased by `image`.
/// - If `image` is non-null, it must point to a valid [`avifImage`] for the duration of this call.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifRGBImageSetDefaults(
    rgb: *mut avifRGBImage,
    image: *const avifImage,
) {
    check_pointer_or_return!(rgb);
    check_pointer_or_return!(image);
    let image: image::Image = deref_const!(image).as_rust_image_no_metadata();
    *deref_mut!(rgb) = rgb::Image::create_from_yuv(&image).into();
}

/// # Safety
/// - If `image` is non-null, it must point to a valid [`avifImage`] whose non-null plane pointers
///   are valid for reads of `height * rowBytes` bytes for each active plane.
/// - If `rgb` is non-null, it must point to a valid [`avifRGBImage`] whose `pixels` buffer is
///   valid for writes of `height * rowBytes` bytes (and disjoint from `image`'s planes).
#[no_mangle]
pub unsafe extern "C" fn crabby_avifImageYUVToRGB(
    image: *const avifImage,
    rgb: *mut avifRGBImage,
) -> avifResult {
    check_pointer!(image);
    check_pointer!(rgb);
    let image_ref = deref_const!(image);
    if image_ref.yuvPlanes[0].is_null() {
        return avifResult::Ok;
    }
    let mut rgb: rgb::Image = match deref_const!(rgb).try_into() {
        Ok(rgb) => rgb,
        Err(err) => return (&err).into(),
    };
    if rgb.pixels.is_none() {
        return avifResult::ReformatFailed;
    }
    let rust_image: image::Image = image_ref.as_rust_image_no_metadata();
    for (plane_ptr, rust_plane) in [
        image_ref.yuvPlanes[0],
        image_ref.yuvPlanes[1],
        image_ref.yuvPlanes[2],
        image_ref.alphaPlane,
    ]
    .into_iter()
    .zip(&rust_image.planes)
    {
        if !plane_ptr.is_null() && rust_plane.is_none() {
            return avifResult::ReformatFailed;
        }
    }
    rgb.convert_from_yuv(&rust_image).into()
}

/// # Safety
/// - If `image` is non-null, it must point to a valid, mutable [`avifImage`] whose owned plane
///   buffers (when `imageOwnsYUVPlanes` / `imageOwnsAlphaPlane` are set) were allocated by
///   `crabby_avifAlloc`.
/// - If `rgb` is non-null, it must point to a valid [`avifRGBImage`] whose `pixels` buffer is
///   valid for reads of `height * rowBytes` bytes (and disjoint from `image`).
#[no_mangle]
pub unsafe extern "C" fn crabby_avifImageRGBToYUV(
    image: *mut avifImage,
    rgb: *const avifRGBImage,
) -> avifResult {
    check_pointer!(image);
    check_pointer!(rgb);
    let rgb: rgb::Image = match deref_const!(rgb).try_into() {
        Ok(rgb) => rgb,
        Err(err) => return (&err).into(),
    };
    if rgb.pixels.is_none() {
        return avifResult::ReformatFailed;
    }
    let dst = deref_mut!(image);
    let mut tmp_image: image::Image = dst.as_rust_image_no_metadata();
    // tmp_image borrows the planes of |image|, which are freed below before being
    // copied back from tmp_image. convert_to_yuv() always reallocates the color planes,
    // but only reallocates the alpha plane if |rgb| has alpha. Replace any borrowed
    // alpha plane with an owned one so that it does not dangle after FreePlanes.
    if tmp_image.has_plane(Plane::A) && !rgb.has_alpha() {
        let res = tmp_image.allocate_planes(Category::Alpha);
        if res.is_err() {
            return res.into();
        }
    }
    let res = rgb.convert_to_yuv(&mut tmp_image);
    if res.is_err() {
        return res.into();
    }
    dst.free_planes(avifPlanesFlag::AvifPlanesAll as u32);
    let planes_to_alloc = if tmp_image.has_alpha() {
        avifPlanesFlag::AvifPlanesAll
    } else {
        avifPlanesFlag::AvifPlanesYuv
    } as u32;
    if let Err(err) = avif_image_allocate_planes_helper(dst, planes_to_alloc) {
        return (&err).into();
    }
    CopyPlanes(dst, &tmp_image).into()
}

fn CopyPlanes(dst: &mut avifImage, src: &Image) -> AvifResult<()> {
    // When scaling a P010 image, the scaling code converts the image into Yuv420 with an explicit
    // V plane. If the V plane is missing in `dst`, allocate it here so it will be freed alongside
    // the other owned plane buffers when `dst` is destroyed.
    if src.depth > 8 && dst.yuvPlanes[2].is_null() {
        if let Some(plane_data) = src.plane_data(Plane::V) {
            let row_bytes = checked_mul!(plane_data.width, 2)?;
            let plane_size = usize_from_u32(checked_mul!(row_bytes, plane_data.height)?)?;
            if plane_size > 0 {
                let _ = isize_from_usize(plane_size)?;
                // SAFETY: `plane_size` is in `1..=isize::MAX`; `crabby_avifAlloc` allocates a
                // zero-initialized buffer via `calloc` (or returns null on OOM).
                dst.yuvPlanes[2] = unsafe { crabby_avifAlloc(plane_size) } as *mut u8;
                if dst.yuvPlanes[2].is_null() {
                    return AvifError::out_of_memory();
                }
                dst.yuvRowBytes[2] = row_bytes;
            }
        }
    }

    let mut dst_image = dst.as_rust_image_no_metadata();
    for plane in ALL_PLANES {
        let Some(plane_data) = src.plane_data(plane) else {
            continue;
        };
        if !dst_image.has_plane(plane) {
            return AvifError::invalid_argument();
        }
        if src.depth == 8 {
            for y in 0..plane_data.height {
                let src_slice = src.row(plane, y)?;
                let dst_slice = dst_image.row_mut(plane, y)?;
                if dst_slice.len() != src_slice.len() {
                    return AvifError::invalid_argument();
                }
                dst_slice.copy_from_slice(src_slice);
            }
        } else {
            for y in 0..plane_data.height {
                let src_slice = src.row16(plane, y)?;
                let dst_slice = dst_image.row16_mut(plane, y)?;
                if dst_slice.len() != src_slice.len() {
                    return AvifError::invalid_argument();
                }
                dst_slice.copy_from_slice(src_slice);
            }
        }
    }
    Ok(())
}

/// # Safety
/// - If `image` is non-null, it must point to a valid, mutable [`avifImage`] whose active plane
///   buffers are valid for reads and writes of `plane_height * rowBytes` bytes for the duration of
///   this call.
/// - `_diag` is unused and may be null.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifImageScale(
    image: *mut avifImage,
    dstWidth: u32,
    dstHeight: u32,
    _diag: *mut avifDiagnostics,
) -> avifResult {
    check_pointer!(image);
    let dst_image = deref_mut!(image);
    let orig_width = dst_image.width;
    let orig_height = dst_image.height;
    if dstWidth == orig_width && dstHeight == orig_height {
        return avifResult::Ok;
    }
    if dstWidth == 0 || dstHeight == 0 {
        return avifResult::InvalidArgument;
    }
    if dstWidth > orig_width || dstHeight > orig_height {
        // To avoid buffer reallocations, we only support scaling to a smaller size.
        return avifResult::NotImplemented;
    }

    let mut rust_image: image::Image = dst_image.as_rust_image_no_metadata();
    for (plane_ptr, rust_plane) in [
        dst_image.yuvPlanes[0],
        dst_image.yuvPlanes[1],
        dst_image.yuvPlanes[2],
        dst_image.alphaPlane,
    ]
    .into_iter()
    .zip(&rust_image.planes)
    {
        if !plane_ptr.is_null() && rust_plane.is_none() {
            return avifResult::InvalidArgument;
        }
    }
    let res = rust_image.scale(dstWidth, dstHeight, Category::Color);
    if res.is_err() {
        return res.into();
    }
    // The scale function is designed to work only for one category at a time.
    // Restore the width and height to the original values before scaling the
    // alpha plane.
    rust_image.width = orig_width;
    rust_image.height = orig_height;
    let res = rust_image.scale(dstWidth, dstHeight, Category::Alpha);
    if res.is_err() {
        return res.into();
    }

    dst_image.width = rust_image.width;
    dst_image.height = rust_image.height;
    dst_image.depth = rust_image.depth as _;
    dst_image.yuvFormat = rust_image.yuv_format;
    CopyPlanes(dst_image, &rust_image).into()
}

#[no_mangle]
pub extern "C" fn crabby_avifRGBFormatChannelCount(format: rgb::Format) -> u32 {
    format.channel_count()
}

/// # Safety
/// If `rgb` is non-null, it must point to a valid [`avifRGBImage`] for the duration of this call.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifRGBImagePixelSize(rgb: *mut avifRGBImage) -> u32 {
    if rgb.is_null() {
        return 0;
    }
    let rgb = deref_const!(rgb);
    rgb.format.pixel_size(rgb.depth)
}

#[no_mangle]
pub extern "C" fn crabby_avifRGBFormatHasAlpha(format: rgb::Format) -> avifBool {
    format.has_alpha().into()
}

/// # Safety
/// If `rgb` is non-null, it must point to a valid, mutable [`avifRGBImage`] whose `pixels` field
/// is either null or a pointer previously allocated by `crabby_avifAlloc` /
/// [`crabby_avifRGBImageAllocatePixels`].
#[no_mangle]
pub unsafe extern "C" fn crabby_avifRGBImageAllocatePixels(rgb: *mut avifRGBImage) -> avifResult {
    check_pointer!(rgb);
    let rgb = deref_mut!(rgb);
    rgb.free_pixels();
    if rgb.width == 0 || rgb.height == 0 {
        return avifResult::InvalidArgument;
    }
    let pixel_size = rgb.format.pixel_size(rgb.depth);
    let Ok(row_bytes_u32) = checked_mul!(rgb.width, pixel_size) else {
        return avifResult::InvalidArgument;
    };
    let Ok(row_bytes) = usize_from_u32(row_bytes_u32) else {
        return avifResult::InvalidArgument;
    };
    let Ok(height) = usize_from_u32(rgb.height) else {
        return avifResult::InvalidArgument;
    };
    let Ok(raw_size) = checked_mul!(row_bytes, height) else {
        return avifResult::InvalidArgument;
    };
    let Some(alloc_size) = raw_size.checked_add(raw_size % 2) else {
        return avifResult::InvalidArgument;
    };
    if isize_from_usize(alloc_size).is_err() {
        return avifResult::InvalidArgument;
    }
    // SAFETY: `alloc_size <= isize::MAX`; `crabby_avifAlloc` allocates `alloc_size` zeroed bytes
    // via `calloc` (or returns null on OOM), matching `crabby_avifFree` in `free_pixels`.
    rgb.pixels = unsafe { crabby_avifAlloc(alloc_size) } as *mut u8;
    if rgb.pixels.is_null() {
        return avifResult::OutOfMemory;
    }
    rgb.row_bytes = row_bytes_u32;
    avifResult::Ok
}

/// # Safety
/// If `rgb` is non-null, it must point to a valid, mutable [`avifRGBImage`] whose `pixels` field
/// is either null or a pointer previously allocated by `crabby_avifAlloc` /
/// [`crabby_avifRGBImageAllocatePixels`] that has not yet been freed.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifRGBImageFreePixels(rgb: *mut avifRGBImage) {
    check_pointer_or_return!(rgb);
    deref_mut!(rgb).free_pixels();
}
