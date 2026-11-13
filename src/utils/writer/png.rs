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
use crate::reformat::rgb;
use crate::utils::*;

use std::ffi::c_char;
use std::ffi::c_int;
use std::fs::File;
use std::io::Write;
use std::ptr;
use std::slice;

use super::Writer;

use libpng_sys::bindings::*;

const WRITE_ERROR_MSG: &[u8] = b"Write Error\0";
const ICC_PROFILE_NAME: &[u8] = b"libavif\0";
const XMP_KEY: &[u8] = b"XML:com.adobe.xmp\0";

#[derive(Default)]
pub struct PngWriter {
    pub depth: Option<u8>,
    pub compression_level: Option<i32>,
}

/// Owns a valid non-null `libpng` write struct (`png`) and info struct (`info`) pair, destroying
/// both on drop.
struct PngWriterNative {
    png: png_structp,
    info: png_infop,
}

impl PngWriterNative {
    fn new() -> AvifResult<Self> {
        let ver = PNG_LIBPNG_VER_STRING.as_ptr().cast::<c_char>();
        // SAFETY: `PNG_LIBPNG_VER_STRING` is a static NUL-terminated byte string generated from
        // `libpng`'s header, and null/`None` error arguments select `libpng`'s default handlers.
        let png = unsafe { png_create_write_struct(ver, ptr::null_mut(), None, None) };
        if png.is_null() {
            return Err(AvifError::UnknownError(
                "png_create_write_struct failed".into(),
            ));
        }
        let mut writer = Self {
            png,
            info: ptr::null_mut(),
        };
        // SAFETY: `writer.png` was just allocated by `png_create_write_struct` and verified
        // non-null; `writer` drops it automatically if `png_create_info_struct` fails.
        let info = unsafe {
            png_set_benign_errors(writer.png, 1);
            png_create_info_struct(writer.png)
        };
        if info.is_null() {
            return Err(AvifError::UnknownError(
                "png_create_info_struct failed".into(),
            ));
        }
        writer.info = info;
        Ok(writer)
    }
}

impl Drop for PngWriterNative {
    fn drop(&mut self) {
        if !self.png.is_null() {
            let png_ptr = ptr::from_mut(&mut self.png);
            let info_ptr = if self.info.is_null() {
                ptr::null_mut()
            } else {
                ptr::from_mut(&mut self.info)
            };
            // SAFETY: `self.png` is non-null and was allocated by `png_create_write_struct`;
            // `info_ptr` is either null or points to `self.info` allocated by
            // `png_create_info_struct`. Both are owned exclusively by `self` and destroyed once.
            unsafe {
                png_destroy_write_struct(png_ptr, info_ptr);
            }
        }
    }
}

/// # Safety
/// Must only be invoked by `libpng` as the `write_data_fn` callback registered via
/// `png_set_write_fn`. `png_ptr` must be a valid `libpng` write struct whose `io_ptr` points to a
/// valid, aligned `File` exclusively accessible for the duration of this call, and when
/// `length > 0`, `data` must point to a readable buffer of at least `length` initialized bytes
/// disjoint from the `File`.
unsafe extern "C" fn crabbyavif_png_write_data(
    png_ptr: png_structp,
    data: png_bytep,
    length: png_size_t,
) {
    if length == 0 || png_ptr.is_null() {
        return;
    }
    // SAFETY: `png_ptr` is verified non-null and valid per `libpng`'s callback contract;
    // `png_get_io_ptr` returns the `io_ptr` registered in `png_set_write_fn`.
    let io_ptr = unsafe { png_get_io_ptr(png_ptr) };
    let write_ok = if !data.is_null() && !io_ptr.is_null() {
        let data_const: *const u8 = data;
        let file_ptr = io_ptr.cast::<File>();
        // SAFETY: `data` is verified non-null and `length > 0`; per `libpng`'s `png_rw_ptr`
        // contract, `data` points to `length` initialized bytes valid for reading during this
        // callback (`u8` has alignment 1) and disjoint from `file`.
        let buf = unsafe { slice::from_raw_parts(data_const, length) };
        // SAFETY: `file_ptr` is verified non-null and was set to `file` (`*mut File`) in
        // `png_set_write_fn`; `file` outlives `png_ptr` and is not accessed elsewhere while
        // `libpng` invokes this callback.
        let file = unsafe { &mut *file_ptr };
        file.write_all(buf).is_ok()
    } else {
        false
    };
    if !write_ok {
        let err_msg = WRITE_ERROR_MSG.as_ptr().cast::<c_char>();
        // With no `setjmp` handler installed, `libpng`'s default error handler aborts; no `Drop`
        // values are live on this frame.
        // SAFETY: `png_ptr` is verified non-null and valid, and `err_msg` points to a static
        // NUL-terminated C string.
        unsafe {
            png_error(png_ptr, err_msg);
        }
    }
}

extern "C" fn crabbyavif_png_output_flush(_png_ptr: png_structp) {}

impl Writer for PngWriter {
    fn write_frame(&mut self, file: &mut File, image: &Image) -> AvifResult<()> {
        if image.width == 0 || image.height == 0 {
            return AvifError::invalid_argument();
        }
        if image.matrix_coefficients == MatrixCoefficients::YcgcoRo {
            return Err(AvifError::UnknownError(
                "YcgcoRo cannot be used with PNG because it has an even bit depth.".into(),
            ));
        }
        let mut rgb_depth = self.depth.unwrap_or(0);
        if rgb_depth == 0 {
            rgb_depth = if image.depth > 8 { 16 } else { 8 };
        }
        if image.matrix_coefficients == MatrixCoefficients::YcgcoRe {
            if image.depth != 10 {
                return Err(AvifError::UnknownError(
                    "YcgcoRe can only be used with bit depth 10.".into(),
                ));
            }
            if self.depth.unwrap_or(0) != 0 && self.depth.unwrap_or(0) != 8 {
                return Err(AvifError::UnknownError(
                    "Cannot request non-8 bits for YCgCo-Re.".into(),
                ));
            }
            rgb_depth = 8;
        }
        let copy_y_plane = image.yuv_format == PixelFormat::Yuv400
            && !image.alpha_present
            && image.depth == 8
            && rgb_depth == 8
            && image.clap.is_none()
            && image.irot_angle.is_none()
            && image.imir_axis.is_none();
        let mut rgb = rgb::Image::create_from_yuv(image);
        let color_type;
        if copy_y_plane {
            color_type = PNG_COLOR_TYPE_GRAY;
        } else {
            rgb.depth = rgb_depth;
            match (image.yuv_format, image.alpha_present) {
                (PixelFormat::Yuv400, true) => {
                    color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
                    rgb.format = rgb::Format::GrayA;
                }
                (PixelFormat::Yuv400, false) => {
                    color_type = PNG_COLOR_TYPE_GRAY;
                    rgb.format = rgb::Format::Gray;
                }
                (_, true) => {
                    // TODO - b/479429854: Support specifying chroma upsampling.
                    color_type = if image.is_opaque()? {
                        rgb.format = rgb::Format::Rgb;
                        PNG_COLOR_TYPE_RGB
                    } else {
                        PNG_COLOR_TYPE_RGBA
                    };
                }
                (_, false) => {
                    // TODO - b/479429854: Support specifying chroma upsampling.
                    color_type = PNG_COLOR_TYPE_RGB;
                    rgb.format = rgb::Format::Rgb;
                }
            }
            rgb.allocate()?;
            rgb.convert_from_yuv(image)?;
        }
        let height = image.height;
        let png = PngWriterNative::new()?;

        let io_ptr = ptr::from_mut(file).cast();
        let rgb_depth_c = c_int::from(rgb_depth);
        let color_type_c = color_type as c_int;
        let skip_srgb_check = PNG_SKIP_sRGB_CHECK_PROFILE as c_int;
        let option_on = PNG_OPTION_ON as c_int;
        let interlace_none = PNG_INTERLACE_NONE as c_int;
        let compression_default = PNG_COMPRESSION_TYPE_DEFAULT as c_int;
        let filter_default = PNG_FILTER_TYPE_DEFAULT as c_int;
        // SAFETY: `png.png` and `png.info` are valid non-null `libpng` structs owned by `png`.
        // `io_ptr` points to `file`, which outlives `png` and is not accessed directly while
        // `libpng` writes. `image.width` and `image.height` are verified non-zero above and match
        // the row pointers written below.
        unsafe {
            png_set_write_fn(
                png.png,
                io_ptr,
                Some(crabbyavif_png_write_data),
                Some(crabbyavif_png_output_flush),
            );
            png_set_option(png.png, skip_srgb_check, option_on);
            png_set_IHDR(
                png.png,
                png.info,
                image.width,
                image.height,
                rgb_depth_c,
                color_type_c,
                interlace_none,
                compression_default,
                filter_default,
            );
        }
        if let Some(compression_level) = self.compression_level {
            // SAFETY: `png.png` is a valid non-null `libpng` write struct owned by `png`.
            unsafe {
                png_set_compression_level(png.png, compression_level);
            }
        }
        if image.icc.is_empty() {
            if image.color_primaries == ColorPrimaries::Srgb
                && image.transfer_characteristics == TransferCharacteristics::Srgb
            {
                let intent = PNG_sRGB_INTENT_PERCEPTUAL as c_int;
                // SAFETY: `png.png` and `png.info` are valid non-null `libpng` structs owned by
                // `png`.
                unsafe {
                    png_set_sRGB_gAMA_and_cHRM(png.png, png.info, intent);
                }
            } else {
                if let Some(primaries) = image.color_primaries.values() {
                    let [rx, ry, gx, gy, bx, by, wx, wy] = primaries.map(f64::from);
                    // SAFETY: `png.png` and `png.info` are valid non-null `libpng` structs owned
                    // by `png`.
                    unsafe {
                        png_set_cHRM(png.png, png.info, wx, wy, rx, ry, gx, gy, bx, by);
                    }
                }
                if let Some(gamma) = image.transfer_characteristics.gamma() {
                    let file_gamma = f64::from(1.0 / gamma);
                    // SAFETY: `png.png` and `png.info` are valid non-null `libpng` structs owned
                    // by `png`.
                    unsafe {
                        png_set_gAMA(png.png, png.info, file_gamma);
                    }
                }
            }
        } else {
            // If there is an ICC profile, the CICP values are irrelevant and only the ICC
            // profile is written. If we could extract the primaries/transfer curve from the
            // ICC profile, then they could be written in cHRM/gAMA chunks.
            let icc_size = u32_from_usize(image.icc.len())?;
            let icc_ptr = image.icc.as_ptr();
            let icc_name = ICC_PROFILE_NAME.as_ptr().cast::<c_char>();
            // SAFETY: `png.png` and `png.info` are valid non-null `libpng` structs, `icc_name`
            // points to a static NUL-terminated C string, and `icc_ptr` points to `icc_size` (> 0)
            // initialized bytes in `image.icc`.
            unsafe {
                png_set_iCCP(png.png, png.info, icc_name, 0, icc_ptr, icc_size);
            }
        }
        if !image.exif.is_empty() {
            let exif_size = u32_from_usize(image.exif.len())?;
            let exif_ptr = image.exif.as_ptr().cast_mut();
            // SAFETY: `png.png` and `png.info` are valid non-null `libpng` structs, and `exif_ptr`
            // points to `exif_size` (> 0) initialized bytes in `image.exif`. `png_set_eXIf_1` only
            // reads from `exif_ptr` (`memcpy` into `info_ptr->exif`) despite lacking `const` in C.
            unsafe {
                png_set_eXIf_1(png.png, png.info, exif_size, exif_ptr);
            }
        }
        if !image.xmp.is_empty() {
            let xmp_with_nul_len = image
                .xmp
                .len()
                .checked_add(1)
                .ok_or(AvifError::OutOfMemory)?;
            let mut xmp = create_vec_exact(xmp_with_nul_len)?;
            xmp.try_extend_from_slice(&image.xmp)?;
            xmp.try_push(0)?;
            let text = png_text {
                compression: PNG_ITXT_COMPRESSION_NONE as c_int,
                key: XMP_KEY.as_ptr().cast::<c_char>().cast_mut(),
                text: xmp.as_mut_ptr().cast::<c_char>(),
                text_length: 0,
                itxt_length: xmp.len(),
                lang: ptr::null_mut(),
                lang_key: ptr::null_mut(),
            };
            let text_ptr = ptr::from_ref(&text);
            // SAFETY: `png.png` and `png.info` are valid non-null `libpng` structs, and `text_ptr`
            // points to an initialized `png_text` whose `key` and `text` are valid NUL-terminated
            // strings. `png_set_text` copies `key` and `text` into `png.info`-owned allocations
            // before returning without mutating the input buffers.
            unsafe {
                png_set_text(png.png, png.info, text_ptr, 1);
            }
        }
        // SAFETY: `png.png` and `png.info` are valid non-null `libpng` structs with `IHDR` and
        // write callbacks configured above.
        unsafe {
            png_write_info(png.png, png.info);
        }
        if image.icc.is_empty() {
            let cicp_name: &[png_byte; 5] = b"cICP\0";
            let cicp_data: [png_byte; 4] = [
                image.color_primaries as png_byte,
                image.transfer_characteristics as png_byte,
                MatrixCoefficients::Identity as png_byte,
                1, // full_range
            ];
            let cicp_name_ptr = cicp_name.as_ptr();
            let cicp_data_ptr = cicp_data.as_ptr();
            let cicp_len = cicp_data.len();
            // SAFETY: `png.png` is a valid non-null `libpng` write struct after `png_write_info`,
            // `cicp_name_ptr` points to 4 chunk-type bytes (plus NUL), and `cicp_data_ptr` points
            // to `cicp_len` (4) initialized bytes.
            unsafe {
                png_write_chunk(png.png, cicp_name_ptr, cicp_data_ptr, cicp_len);
            }
        }
        let height_usize = usize_from_u32(height)?;
        let mut row_pointers: Vec<png_bytep> = create_vec_exact(height_usize)?;
        if copy_y_plane {
            for y in 0..height {
                if image.depth == 8 {
                    row_pointers.try_push(image.row(Plane::Y, y)?.as_ptr().cast_mut())?;
                } else {
                    row_pointers
                        .try_push(image.row16(Plane::Y, y)?.as_ptr().cast::<u8>().cast_mut())?;
                }
            }
        } else {
            for y in 0..height {
                if rgb.depth == 8 {
                    row_pointers.try_push(rgb.row(y)?.as_ptr().cast_mut())?;
                } else {
                    row_pointers.try_push(rgb.row16(y)?.as_ptr().cast::<u8>().cast_mut())?;
                }
            }
        }
        if rgb_depth > 8 {
            // SAFETY: `png.png` is a valid non-null `libpng` write struct owned by `png`.
            unsafe {
                png_set_swap(png.png);
            }
        }
        let row_pointers_ptr = row_pointers.as_mut_ptr();
        // SAFETY: `png.png` is a valid non-null `libpng` write struct. `row_pointers_ptr` points
        // to `height` valid row pointers, each referencing a readable row slice of at least
        // `width * bytes_per_pixel` bytes in `image` or `rgb`, both of which remain alive and
        // immutable throughout `png_write_image` (`libpng` only reads the row buffers).
        unsafe {
            png_write_image(png.png, row_pointers_ptr);
            png_write_end(png.png, ptr::null_mut());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_frame_rejects_zero_dimensions() {
        let path = std::env::temp_dir().join(format!(
            "crabbyavif_png_writer_zero_dim_{}.png",
            std::process::id()
        ));
        let mut file = File::create(&path).unwrap();

        let mut writer = PngWriter::default();
        assert!(matches!(
            writer.write_frame(&mut file, &Image::default()),
            Err(AvifError::InvalidArgument)
        ));

        let yuv400_zero_w = Image {
            width: 0,
            height: 4,
            depth: 8,
            yuv_format: PixelFormat::Yuv400,
            ..Image::default()
        };
        assert!(matches!(
            writer.write_frame(&mut file, &yuv400_zero_w),
            Err(AvifError::InvalidArgument)
        ));

        let yuv400_zero_h = Image {
            width: 4,
            height: 0,
            depth: 8,
            yuv_format: PixelFormat::Yuv400,
            ..Image::default()
        };
        assert!(matches!(
            writer.write_frame(&mut file, &yuv400_zero_h),
            Err(AvifError::InvalidArgument)
        ));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn write_data_callback_ignores_zero_len_and_null_png_ptr() {
        // SAFETY: `crabbyavif_png_write_data` returns early before dereferencing any pointer when
        // `length == 0` or `png_ptr` is null.
        unsafe {
            crabbyavif_png_write_data(ptr::null_mut(), ptr::null_mut(), 0);
            crabbyavif_png_write_data(ptr::null_mut(), ptr::null_mut(), 16);
            let png = PngWriterNative::new().unwrap();
            crabbyavif_png_write_data(png.png, ptr::null_mut(), 0);
        }
    }
}
