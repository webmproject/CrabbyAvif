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

use crate::gainmap::GainMap;
use crate::parser::exif;
use crate::reformat::*;
use crate::utils::*;
use crate::AvifError;
use crate::AvifResult;

use super::icc;
use super::Config;
use super::Reader;

use std::ffi::c_char;
use std::ffi::c_int;
use std::ffi::CStr;
use std::fs::File;
use std::io::Read;
use std::ptr;
use std::slice;

use libpng_sys::bindings::*;

const PNG_SIGNATURE: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];
const EXIF_PREFIX: &[u8] = b"Exif\0\0";
const XMP_PREFIX: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
const READ_ERROR_MSG: &[u8] = b"Read Error\0";

pub struct PngReader {
    filename: String,
}

fn set_png_cicp(image: &mut Image, cicp: &[png_byte; 4]) {
    let &[cp, tc, mc, full_range] = cicp;
    image.color_primaries = u16::from(cp).into();
    image.transfer_characteristics = u16::from(tc).into();
    // PNG specification Third Edition Section 4.3:
    //  RGB is currently the only supported color model in PNG, and as such Matrix Coefficients shall be set to 0.
    // Note that we don't set image.matrix_coefficients to this value as the Avif's YUV matrix is independent from the PNG's.
    if mc != 0 {
        println!("Warning: Unsupported PNG CICP matrix coefficients value {mc}. Expected to be 0.");
    }
    // Limited range PNG files are uncommon and would require conversion to full range which we don't support for now for simplicity.
    // See also https://github.com/w3c/png/issues/312#issuecomment-2325281113 and https://svgees.us/blog/cICP.html#the-other-two-numbers
    // Similarly, we don't set image.yuv_range to this value as the Avif's YUV range is independent from the PNG's.
    if full_range != 1 {
        println!(
            "Warning: Unsupported PNG CICP full range flag value {full_range}. Expected to be 1."
        );
    }
}

#[derive(Default)]
struct UnknownCicpChunkData {
    cicp: Option<[png_byte; 4]>,
}

/// # Safety
/// Must only be invoked by `libpng` as the `read_user_chunk_fn` callback registered via
/// `png_set_read_user_chunk_fn`. `png_ptr` must be a valid `libpng` read struct whose
/// `user_chunk_ptr` points to a valid, aligned `UnknownCicpChunkData` exclusively accessible for
/// the duration of this call, and `chunk` must point to a valid `png_unknown_chunk` struct whose
/// `data` (when non-null) is readable for `chunk.size` bytes.
unsafe extern "C" fn crabbyavif_png_read_unknown_chunk(
    png_ptr: png_structp,
    chunk: png_unknown_chunkp,
) -> c_int {
    if png_ptr.is_null() || chunk.is_null() {
        return 0;
    }
    // SAFETY: `png_ptr` is verified non-null and valid per `libpng`'s callback contract;
    // `png_get_user_chunk_ptr` returns the `user_chunk_ptr` registered in
    // `png_set_read_user_chunk_fn`.
    let user_chunk_ptr = unsafe { png_get_user_chunk_ptr(png_ptr) };
    if user_chunk_ptr.is_null() {
        return 0;
    }
    let data_ptr = user_chunk_ptr.cast::<UnknownCicpChunkData>();
    // SAFETY: `data_ptr` is verified non-null and was set to `&mut unknown_chunk_data`
    // (`*mut UnknownCicpChunkData`) in `png_set_read_user_chunk_fn`, which outlives `png_ptr` and
    // is not accessed elsewhere while `libpng` invokes this callback.
    let data = unsafe { &mut *data_ptr };
    // SAFETY: `chunk` is verified non-null and points to `png_ptr->unknown_chunk` initialized by
    // `libpng` for the duration of this callback.
    let chunk = unsafe { &*chunk };
    if matches!(chunk.name, [b'c', b'I', b'C', b'P', _]) && chunk.size >= 4 && !chunk.data.is_null()
    {
        let chunk_data: *const u8 = chunk.data;
        // SAFETY: `chunk.data` is verified non-null and `libpng` guarantees `chunk.data` points to
        // a readable buffer of `chunk.size` (>= 4) initialized bytes (`u8` has alignment 1).
        let cicp_slice = unsafe { slice::from_raw_parts(chunk_data, 4) };
        if let Ok(cicp) = cicp_slice.try_into() {
            data.cicp = Some(cicp);
        }
    }
    1
}

/// Owns a valid non-null `libpng` read struct (`png`) and info struct (`info`) pair, destroying
/// both on drop.
struct PngReaderNative {
    png: png_structp,
    info: png_infop,
}

impl PngReaderNative {
    fn new() -> AvifResult<Self> {
        let ver = PNG_LIBPNG_VER_STRING.as_ptr().cast::<c_char>();
        // SAFETY: `PNG_LIBPNG_VER_STRING` is a static NUL-terminated byte string generated from
        // `libpng`'s header, and null/`None` error arguments select `libpng`'s default handlers.
        let png = unsafe { png_create_read_struct(ver, ptr::null_mut(), None, None) };
        if png.is_null() {
            return Err(AvifError::UnknownError(
                "png_create_read_struct failed".into(),
            ));
        }
        let mut reader = Self {
            png,
            info: ptr::null_mut(),
        };
        // SAFETY: `reader.png` was just allocated by `png_create_read_struct` and verified
        // non-null; `reader` drops it automatically if `png_create_info_struct` fails.
        let info = unsafe {
            png_set_benign_errors(reader.png, 1);
            png_create_info_struct(reader.png)
        };
        if info.is_null() {
            return Err(AvifError::UnknownError(
                "png_create_info_struct failed".into(),
            ));
        }
        reader.info = info;
        Ok(reader)
    }

    fn extract_exif_and_xmp(
        &self,
        image: &mut Image,
        ignore_exif: &mut bool,
        ignore_xmp: &mut bool,
    ) -> AvifResult<()> {
        if !*ignore_exif {
            let mut exif_size = 0u32;
            let mut exif: png_bytep = ptr::null_mut();
            let exif_size_ptr = ptr::from_mut(&mut exif_size);
            let exif_ptr = ptr::from_mut(&mut exif);
            // SAFETY: `self.png` and `self.info` are valid non-null `libpng` structs owned by
            // `self`, and `exif_size_ptr` and `exif_ptr` are valid mutable out-pointers.
            let exif_flag = unsafe { png_get_eXIf_1(self.png, self.info, exif_size_ptr, exif_ptr) };
            if exif_flag == PNG_INFO_eXIf {
                if exif_size == 0 || exif.is_null() {
                    return Err(AvifError::UnknownError(
                        "Exif extraction failed. Empty eXIf chunk".into(),
                    ));
                }
                let exif_len = usize_from_u32(exif_size)?;
                let exif_const: *const u8 = exif;
                // SAFETY: `png_get_eXIf_1` returned `PNG_INFO_eXIf`, we verified `exif` is
                // non-null and `exif_len > 0`, and `libpng` owns `info_ptr->exif` as a valid
                // readable buffer of `exif_len` bytes for the lifetime of `self.info`.
                let exif_slice = unsafe { slice::from_raw_parts(exif_const, exif_len) };
                image.exif = create_vec_exact(exif_slice.len())?;
                image.exif.try_extend_from_slice(exif_slice)?;
                let _ = exif::set_orientation(&mut image.exif, 1);
                *ignore_exif = true;
            }
        }

        let mut text_chunks: png_textp = ptr::null_mut();
        let text_chunks_ptr = ptr::from_mut(&mut text_chunks);
        // SAFETY: `self.png` and `self.info` are valid non-null `libpng` structs owned by `self`,
        // `text_chunks_ptr` is a valid out-pointer, and `num_text` may be null per `libpng` API.
        let num_text_chunks =
            unsafe { png_get_text(self.png, self.info, text_chunks_ptr, ptr::null_mut()) };
        let text_slice = if num_text_chunks > 0 && !text_chunks.is_null() {
            let count = usize::try_from(num_text_chunks).map_err(|_| AvifError::InvalidArgument)?;
            let text_const: *const png_text = text_chunks;
            // SAFETY: `png_get_text` returned `num_text_chunks > 0` and set `text_chunks` to
            // `info_ptr->text`, a non-null array of `count` initialized `png_text` structs owned
            // by `self.info` and unmutated for the duration of `extract_exif_and_xmp`.
            unsafe { slice::from_raw_parts(text_const, count) }
        } else {
            &[]
        };

        for text in text_slice {
            if *ignore_exif && *ignore_xmp {
                break;
            }
            if text.key.is_null() {
                continue;
            }
            let key_ptr: *const c_char = text.key;
            // SAFETY: `text.key` is verified non-null and comes from a `png_text` entry populated
            // by `libpng`, which guarantees `key` is a valid NUL-terminated string owned by
            // `self.info`.
            let key = unsafe { CStr::from_ptr(key_ptr) }.to_bytes();

            let is_raw_exif = !*ignore_exif && key == b"Raw profile type exif";
            let is_raw_xmp = !*ignore_xmp && key == b"Raw profile type xmp";
            let is_raw_app1 = key == b"Raw profile type APP1" || key == b"Raw profile type app1";
            let is_xmp_itxt = !*ignore_xmp && key == b"XML:com.adobe.xmp";
            if !is_raw_exif && !is_raw_xmp && !is_raw_app1 && !is_xmp_itxt {
                continue;
            }

            let length = if text.compression == PNG_ITXT_COMPRESSION_NONE as c_int
                || text.compression == PNG_ITXT_COMPRESSION_zTXt as c_int
            {
                text.itxt_length
            } else {
                text.text_length
            };
            let text_bytes = if length == 0 || text.text.is_null() {
                &[][..]
            } else {
                let text_ptr = text.text.cast::<u8>().cast_const();
                // SAFETY: `text.text` is verified non-null and `length > 0`; `libpng` guarantees
                // `text.text` points to a readable buffer of at least `length` bytes
                // (`itxt_length` for iTXt chunks, `text_length` for tEXt/zTXt chunks) owned by
                // `self.info` (`u8` has alignment 1).
                unsafe { slice::from_raw_parts(text_ptr, length) }
            };

            if is_raw_exif {
                let exif = icc::copy_raw_profile(text_bytes)?;
                let payload = exif.strip_prefix(EXIF_PREFIX).unwrap_or(&exif);
                image.exif = create_vec_exact(payload.len())?;
                image.exif.try_extend_from_slice(payload)?;
                let _ = exif::set_orientation(&mut image.exif, 1);
                *ignore_exif = true;
            } else if is_raw_xmp {
                let xmp = icc::copy_raw_profile(text_bytes)?;
                let payload = xmp.strip_prefix(XMP_PREFIX).unwrap_or(&xmp);
                image.xmp = create_vec_exact(payload.len())?;
                image.xmp.try_extend_from_slice(payload)?;
                *ignore_xmp = true;
            } else if is_raw_app1 {
                // This can be either exif, xmp or something else.
                let data = icc::copy_raw_profile(text_bytes)?;
                if !*ignore_exif {
                    if let Some(payload) = data.strip_prefix(EXIF_PREFIX) {
                        image.exif = create_vec_exact(payload.len())?;
                        image.exif.try_extend_from_slice(payload)?;
                        let _ = exif::set_orientation(&mut image.exif, 1);
                        *ignore_exif = true;
                    }
                }
                if !*ignore_xmp {
                    if let Some(payload) = data.strip_prefix(XMP_PREFIX) {
                        image.xmp = create_vec_exact(payload.len())?;
                        image.xmp.try_extend_from_slice(payload)?;
                        *ignore_xmp = true;
                    }
                }
            } else if is_xmp_itxt {
                if text_bytes.is_empty() {
                    return Err(AvifError::UnknownError(
                        "XMP extraction failed: empty XML:com.adobe.xmp payload".into(),
                    ));
                }
                image.xmp = create_vec_exact(text_bytes.len())?;
                image.xmp.try_extend_from_slice(text_bytes)?;
                *ignore_xmp = true;
            }
        }
        image.remove_trailing_null_from_xmp();
        Ok(())
    }
}

impl Drop for PngReaderNative {
    fn drop(&mut self) {
        if !self.png.is_null() {
            let png_ptr = ptr::from_mut(&mut self.png);
            let info_ptr = if self.info.is_null() {
                ptr::null_mut()
            } else {
                ptr::from_mut(&mut self.info)
            };
            // SAFETY: `self.png` is non-null and was allocated by `png_create_read_struct`;
            // `info_ptr` is either null or points to `self.info` allocated by
            // `png_create_info_struct`. Both are owned exclusively by `self` and destroyed once.
            unsafe {
                png_destroy_read_struct(png_ptr, info_ptr, ptr::null_mut());
            }
        }
    }
}

/// # Safety
/// Must only be invoked by `libpng` as the `read_data_fn` callback registered via
/// `png_set_read_fn`. `png_ptr` must be a valid `libpng` read struct whose `io_ptr` points to a
/// valid, aligned `File` exclusively accessible for the duration of this call, and when
/// `length > 0`, `data` must point to a writable buffer of at least `length` bytes disjoint from
/// the `File`.
unsafe extern "C" fn crabbyavif_png_read_data(
    png_ptr: png_structp,
    data: png_bytep,
    length: png_size_t,
) {
    if length == 0 || png_ptr.is_null() {
        return;
    }
    // SAFETY: `png_ptr` is verified non-null and valid per `libpng`'s callback contract;
    // `png_get_io_ptr` returns the `io_ptr` registered in `png_set_read_fn`.
    let io_ptr = unsafe { png_get_io_ptr(png_ptr) };
    let read_ok = if !data.is_null() && !io_ptr.is_null() {
        let file_ptr = io_ptr.cast::<File>();
        // SAFETY: `file_ptr` is verified non-null and was set to `&mut file` (`*mut File`) in
        // `png_set_read_fn`; `file` outlives `png_ptr` and is not accessed elsewhere while
        // `libpng` invokes this callback.
        let file = unsafe { &mut *file_ptr };
        // SAFETY: `data` is verified non-null and `length > 0`; per `libpng`'s `png_rw_ptr`
        // contract, `data` points to a valid writable buffer of `length` bytes (`u8` has
        // alignment 1) disjoint from `file`.
        let buf = unsafe { slice::from_raw_parts_mut(data, length) };
        file.read_exact(buf).is_ok()
    } else {
        false
    };
    if !read_ok {
        let err_msg = READ_ERROR_MSG.as_ptr().cast::<c_char>();
        // With no `setjmp` handler installed, `libpng`'s default error handler aborts; no `Drop`
        // values are live on this frame.
        // SAFETY: `png_ptr` is verified non-null and valid, and `err_msg` points to a static
        // NUL-terminated C string.
        unsafe {
            png_error(png_ptr, err_msg);
        }
    }
}

impl PngReader {
    pub fn create(filename: &str) -> AvifResult<Self> {
        Ok(Self {
            filename: filename.into(),
        })
    }
}

impl Reader for PngReader {
    fn read_frame(&mut self, config: &Config) -> AvifResult<(Image, u64, Option<GainMap>)> {
        let mut file = File::open(&self.filename)
            .map_err(|_| AvifError::UnknownError("failed to open file".into()))?;
        let mut header = [0u8; 8];
        file.read_exact(&mut header)
            .map_err(|_| AvifError::UnknownError("cannot read png header".into()))?;
        if header != PNG_SIGNATURE {
            return Err(AvifError::UnknownError("not a png".into()));
        }

        // Declared before `png` so both `file` and `unknown_chunk_data` strictly outlive `png`.
        let mut unknown_chunk_data = UnknownCicpChunkData::default();
        let png = PngReaderNative::new()?;
        let mut yuv = Image::default();

        let user_chunk_ptr = ptr::from_mut(&mut unknown_chunk_data).cast();
        let io_ptr = ptr::from_mut(&mut file).cast();
        // SAFETY: `png.png` is a valid non-null `libpng` read struct owned by `png`, and
        // `user_chunk_ptr` points to `unknown_chunk_data`, which outlives `png` and is not
        // accessed directly while `libpng` runs callbacks.
        unsafe {
            png_set_read_user_chunk_fn(
                png.png,
                user_chunk_ptr,
                Some(crabbyavif_png_read_unknown_chunk),
            );
        }
        // SAFETY: `png.png` is a valid non-null `libpng` read struct owned by `png`, and `io_ptr`
        // points to `file`, which outlives `png` and is not accessed directly while `libpng` runs
        // callbacks.
        unsafe {
            png_set_read_fn(png.png, io_ptr, Some(crabbyavif_png_read_data));
        }
        // SAFETY: `png.png` is a valid non-null `libpng` read struct, and the 8-byte PNG signature
        // was already read from `file` above.
        unsafe {
            png_set_sig_bytes(png.png, 8);
        }
        // SAFETY: `png.png` and `png.info` are valid non-null `libpng` structs with read callbacks
        // and signature offset configured above.
        unsafe {
            png_read_info(png.png, png.info);
        }
        // Disarm before reading `unknown_chunk_data.cicp` or running `png_read_image` /
        // `png_read_end` so trailing chunks cannot re-enter with an invalidated pointer.
        // SAFETY: `png.png` is a valid non-null `libpng` read struct owned by `png`.
        unsafe {
            png_set_read_user_chunk_fn(png.png, ptr::null_mut(), None);
        }
        // SAFETY: `png.png` and `png.info` are valid non-null `libpng` structs populated by
        // `png_read_info`.
        let raw_width = unsafe { png_get_image_width(png.png, png.info) };
        // SAFETY: `png.png` and `png.info` are valid non-null `libpng` structs populated by
        // `png_read_info`.
        let raw_height = unsafe { png_get_image_height(png.png, png.info) };
        // SAFETY: `png.png` and `png.info` are valid non-null `libpng` structs populated by
        // `png_read_info`.
        let raw_color_type = unsafe { png_get_color_type(png.png, png.info) };
        // SAFETY: `png.png` and `png.info` are valid non-null `libpng` structs populated by
        // `png_read_info`.
        let raw_bit_depth = unsafe { png_get_bit_depth(png.png, png.info) };

        // Defense-in-depth (`png_read_info` also rejects zero-dimension `IHDR` chunks).
        if raw_width == 0 || raw_height == 0 {
            return AvifError::invalid_argument();
        }
        if config.image_size_limit > 0 {
            let total_pixels = u64::from(raw_width) * u64::from(raw_height);
            if total_pixels > u64::from(config.image_size_limit) {
                return Err(AvifError::UnknownError(format!(
                    "Too big PNG dimensions ({raw_width} x {raw_height} > {} px)",
                    config.image_size_limit
                )));
            }
        }

        if raw_color_type == PNG_COLOR_TYPE_PALETTE as u8 {
            // SAFETY: `png.png` is a valid non-null `libpng` read struct owned by `png`.
            unsafe {
                png_set_palette_to_rgb(png.png);
            }
        }
        if raw_color_type == PNG_COLOR_TYPE_GRAY as u8 && raw_bit_depth < 8 {
            // SAFETY: `png.png` is a valid non-null `libpng` read struct owned by `png`.
            unsafe {
                png_set_expand_gray_1_2_4_to_8(png.png);
            }
        }
        // SAFETY: `png.png` and `png.info` are valid non-null `libpng` structs owned by `png`.
        let has_trns = unsafe { png_get_valid(png.png, png.info, PNG_INFO_tRNS) } != 0;
        if has_trns {
            // SAFETY: `png.png` is a valid non-null `libpng` read struct owned by `png`.
            unsafe {
                png_set_tRNS_to_alpha(png.png);
            }
        }
        let raw_color_type_is_gray = raw_color_type == PNG_COLOR_TYPE_GRAY as u8
            || raw_color_type == PNG_COLOR_TYPE_GRAY_ALPHA as u8;

        let bit_depth = if raw_bit_depth == 16 {
            // SAFETY: `png.png` is a valid non-null `libpng` read struct owned by `png`.
            unsafe {
                png_set_swap(png.png);
            }
            16
        } else {
            8
        };
        // SAFETY: `png.png` and `png.info` are valid non-null `libpng` structs owned by `png`,
        // called once after configuring all read transformations above.
        unsafe {
            png_read_update_info(png.png, png.info);
        }

        yuv.width = raw_width;
        yuv.height = raw_height;
        yuv.yuv_range = config.yuv_range.unwrap_or(YuvRange::Full);
        yuv.matrix_coefficients = config
            .matrix_coefficients
            .unwrap_or(MatrixCoefficients::Bt601);
        if yuv.matrix_coefficients == MatrixCoefficients::YcgcoRo {
            return Err(AvifError::UnknownError(
                "YcgcoRo cannot be used with PNG because it has an even bit depth.".into(),
            ));
        }
        yuv.yuv_format = match config.yuv_format {
            Some(yuv_format) => yuv_format,
            None => {
                if raw_color_type_is_gray {
                    PixelFormat::Yuv400
                } else if matches!(
                    yuv.matrix_coefficients,
                    MatrixCoefficients::Identity | MatrixCoefficients::YcgcoRe
                ) {
                    // Identity and YCgCo-R are valid only with Yuv444.
                    PixelFormat::Yuv444
                } else {
                    PixelFormat::Yuv444
                }
            }
        };
        yuv.depth = match config.depth {
            Some(depth) => depth,
            None => {
                if bit_depth == 8 {
                    8
                } else if config.allow_sample_transform {
                    16
                } else {
                    12
                }
            }
        };
        if yuv.matrix_coefficients == MatrixCoefficients::YcgcoRe {
            if bit_depth != 8 {
                return Err(AvifError::UnknownError(
                    "YcgcoRe cannot be used on 16 bit input because it adds two bits.".into(),
                ));
            }
            if let Some(d) = config.depth {
                if d != 10 {
                    return Err(AvifError::UnknownError(format!(
                        "Cannot request {d} bits for YCgCo-Re as it uses two extra bits"
                    )));
                }
            }
            yuv.depth = 10;
        }
        if !config.ignore_icc {
            let mut cicp: [png_byte; 4] = [0; 4];
            let has_cicp = {
                let [c0, c1, c2, c3] = &mut cicp;
                let (p0, p1, p2, p3) = (
                    ptr::from_mut(c0),
                    ptr::from_mut(c1),
                    ptr::from_mut(c2),
                    ptr::from_mut(c3),
                );
                // SAFETY: `png.png` and `png.info` are valid non-null `libpng` structs owned by
                // `png`, and `p0..p3` are valid disjoint mutable out-pointers to `png_byte`.
                let cicp_flag = unsafe { png_get_cICP(png.png, png.info, p0, p1, p2, p3) };
                cicp_flag != 0
            };
            if has_cicp {
                set_png_cicp(&mut yuv, &cicp);
            } else if let Some(unknown_cicp) = unknown_chunk_data.cicp {
                set_png_cicp(&mut yuv, &unknown_cicp);
            } else {
                let mut icc_profile_name: png_charp = ptr::null_mut();
                let mut icc_compression_type: c_int = 0;
                let mut iccp_data: png_bytep = ptr::null_mut();
                let mut iccp_data_len = 0u32;
                let name_ptr = ptr::from_mut(&mut icc_profile_name);
                let comp_ptr = ptr::from_mut(&mut icc_compression_type);
                let data_ptr = ptr::from_mut(&mut iccp_data);
                let len_ptr = ptr::from_mut(&mut iccp_data_len);
                // SAFETY: `png.png` and `png.info` are valid non-null `libpng` structs owned by
                // `png`, and `name_ptr`, `comp_ptr`, `data_ptr`, `len_ptr` are valid out-pointers.
                let has_iccp = unsafe {
                    png_get_iCCP(png.png, png.info, name_ptr, comp_ptr, data_ptr, len_ptr)
                } == PNG_INFO_iCCP;
                if has_iccp {
                    if (!raw_color_type_is_gray && yuv.yuv_format == PixelFormat::Yuv400)
                        || (raw_color_type_is_gray && yuv.yuv_format != PixelFormat::Yuv400)
                    {
                        return Err(AvifError::UnknownError("Image contains an ICC profile that is incompatible with the requested output. Pass --ignore-icc to discard the ICC profile".into()));
                    }
                    let icc_len = usize_from_u32(iccp_data_len)?;
                    if icc_len > 0 && !iccp_data.is_null() {
                        let iccp_const: *const u8 = iccp_data;
                        // SAFETY: `png_get_iCCP` returned `PNG_INFO_iCCP`, we verified `iccp_data`
                        // is non-null and `icc_len > 0`, and `libpng` owns `info_ptr->iccp_profile`
                        // as a readable buffer of `icc_len` bytes for the lifetime of `png.info`.
                        let icc_slice = unsafe { slice::from_raw_parts(iccp_const, icc_len) };
                        yuv.icc = create_vec_exact(icc_slice.len())?;
                        yuv.icc.try_extend_from_slice(icc_slice)?;
                    }
                } else {
                    let mut srgb_intent: c_int = 0;
                    let srgb_ptr = ptr::from_mut(&mut srgb_intent);
                    // SAFETY: `png.png` and `png.info` are valid non-null `libpng` structs, and
                    // `srgb_ptr` is a valid mutable out-pointer.
                    let has_srgb =
                        unsafe { png_get_sRGB(png.png, png.info, srgb_ptr) } == PNG_INFO_sRGB;
                    if has_srgb {
                        yuv.color_primaries = ColorPrimaries::Srgb;
                        yuv.transfer_characteristics = TransferCharacteristics::Srgb;
                    } else {
                        let mut should_generate_icc = false;
                        let mut gamma: f64 = 0.0;
                        let gamma_ptr = ptr::from_mut(&mut gamma);
                        // SAFETY: `png.png` and `png.info` are valid non-null `libpng` structs,
                        // and `gamma_ptr` is a valid mutable out-pointer.
                        let has_gama =
                            unsafe { png_get_gAMA(png.png, png.info, gamma_ptr) } == PNG_INFO_gAMA;
                        if has_gama {
                            gamma = 1.0 / gamma;
                            yuv.transfer_characteristics =
                                TransferCharacteristics::from_gamma(gamma as f32);
                            if yuv.transfer_characteristics == TransferCharacteristics::Unknown {
                                should_generate_icc = true;
                            }
                        } else {
                            // No gamma information in file. Assume the default value.
                            // PNG specification 1.2 Section 10.5:
                            // Assume a CRT exponent of 2.2 unless detailed calibration measurements
                            // of this particular CRT are available.
                            gamma = 2.2;
                        }
                        let mut png_primaries: [f64; 8] = [0.0; 8];
                        let has_chrm = {
                            let [rx, ry, gx, gy, bx, by, wx, wy] = &mut png_primaries;
                            let (pwx, pwy, prx, pry, pgx, pgy, pbx, pby) = (
                                ptr::from_mut(wx),
                                ptr::from_mut(wy),
                                ptr::from_mut(rx),
                                ptr::from_mut(ry),
                                ptr::from_mut(gx),
                                ptr::from_mut(gy),
                                ptr::from_mut(bx),
                                ptr::from_mut(by),
                            );
                            // SAFETY: `png.png` and `png.info` are valid non-null `libpng` structs,
                            // and `pwx..pby` are valid disjoint mutable `f64` out-pointers.
                            let chrm_flag = unsafe {
                                png_get_cHRM(
                                    png.png, png.info, pwx, pwy, prx, pry, pgx, pgy, pbx, pby,
                                )
                            };
                            chrm_flag == PNG_INFO_cHRM
                        };
                        let primaries: [f32; 8] = if has_chrm {
                            let mapped = png_primaries.map(|x| x as f32);
                            yuv.color_primaries = ColorPrimaries::find(&mapped);
                            if yuv.color_primaries == ColorPrimaries::Unknown {
                                should_generate_icc = true;
                            }
                            mapped
                        } else {
                            ColorPrimaries::Bt709.values().ok_or_else(|| {
                                AvifError::UnknownError("missing Bt709 primaries".into())
                            })?
                        };
                        if should_generate_icc {
                            yuv.color_primaries = ColorPrimaries::Unspecified;
                            yuv.transfer_characteristics = TransferCharacteristics::Unspecified;
                            yuv.icc = icc::generate_icc(yuv.yuv_format, gamma as f32, &primaries)
                                .unwrap_or_default();
                        }
                    }
                }
            }
        }

        if let Some(cp) = config.color_primaries {
            yuv.color_primaries = cp;
        }
        // SAFETY: `png.png` and `png.info` are valid non-null `libpng` structs updated by
        // `png_read_update_info`.
        let num_channels = unsafe { png_get_channels(png.png, png.info) };
        // SAFETY: `png.png` and `png.info` are valid non-null `libpng` structs updated by
        // `png_read_update_info`.
        let png_row_bytes = unsafe { png_get_rowbytes(png.png, png.info) };
        let mut rgb = rgb::Image::create_from_yuv(&yuv);
        rgb.depth = bit_depth;
        rgb.format = match num_channels {
            1 => rgb::Format::Gray,
            2 => rgb::Format::GrayA,
            3 => rgb::Format::Rgb,
            4 => rgb::Format::Rgba,
            _ => {
                return Err(AvifError::UnknownError(format!(
                    "png_get_channels() returned an invalid value: {num_channels}"
                )));
            }
        };
        rgb.allocate()?;
        let row_bytes = usize_from_u32(rgb.row_bytes)?;
        if row_bytes != png_row_bytes {
            return Err(AvifError::UnknownError(format!(
                "rowBytes mismatch libavif {} vs libpng {png_row_bytes}",
                rgb.row_bytes
            )));
        }
        let height = usize_from_u32(rgb.height)?;
        let mut row_pointers: Vec<png_bytep> = create_vec_exact(height)?;
        let Some(pixels) = rgb.pixels.as_mut() else {
            return AvifError::no_content();
        };
        match pixels {
            pixels::Pixels::Buffer(buffer) => {
                if row_bytes == 0 {
                    return AvifError::invalid_argument();
                }
                for row in buffer.chunks_exact_mut(row_bytes) {
                    row_pointers.try_push(row.as_mut_ptr())?;
                }
            }
            pixels::Pixels::Buffer16(buffer) => {
                let row_u16 = row_bytes / 2;
                if row_u16 == 0 || row_bytes % 2 != 0 {
                    return AvifError::invalid_argument();
                }
                for row in buffer.chunks_exact_mut(row_u16) {
                    row_pointers.try_push(row.as_mut_ptr().cast::<u8>())?;
                }
            }
            _ => return AvifError::no_content(),
        }
        if row_pointers.len() != height {
            return AvifError::invalid_argument();
        }
        let row_pointers_ptr = row_pointers.as_mut_ptr();
        // SAFETY: `png.png` is a valid non-null `libpng` read struct configured via
        // `png_read_update_info`. `row_pointers_ptr` points to `height` (`png_get_image_height`)
        // valid mutable row pointers, each pointing to a disjoint writable slice of `png_row_bytes`
        // (`png_get_rowbytes`) bytes in `rgb.pixels`, which remains alive and unaliased throughout
        // `png_read_image`.
        unsafe {
            png_read_image(png.png, row_pointers_ptr);
        }
        rgb.convert_to_yuv(&mut yuv)?;

        let mut ignore_exif = config.ignore_exif;
        let mut ignore_xmp = config.ignore_xmp;
        png.extract_exif_and_xmp(&mut yuv, &mut ignore_exif, &mut ignore_xmp)?;
        if !ignore_exif || !ignore_xmp {
            // SAFETY: `png.png` and `png.info` are valid non-null `libpng` structs after
            // `png_read_image` has consumed all image rows.
            unsafe {
                png_read_end(png.png, png.info);
            }
            png.extract_exif_and_xmp(&mut yuv, &mut ignore_exif, &mut ignore_xmp)?;
        }
        Ok((yuv, 0, None))
    }

    fn has_more_frames(&mut self) -> bool {
        // TODO: b/403090413 - maybe support APNG?
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_chunk_callback_validates_inputs_and_records_cicp() {
        let png = PngReaderNative::new().unwrap();
        let mut payload = [9u8, 16, 0, 1];
        let mut valid_cicp = png_unknown_chunk {
            name: *b"cICP\0",
            data: payload.as_mut_ptr(),
            size: payload.len(),
            location: 0,
        };

        // SAFETY: Tests null-pointer early-return guards before `user_chunk_ptr` is registered.
        unsafe {
            assert_eq!(
                crabbyavif_png_read_unknown_chunk(ptr::null_mut(), &mut valid_cicp),
                0
            );
            assert_eq!(
                crabbyavif_png_read_unknown_chunk(png.png, ptr::null_mut()),
                0
            );
            assert_eq!(
                crabbyavif_png_read_unknown_chunk(png.png, &mut valid_cicp),
                0
            );
        }

        let mut chunk_state = UnknownCicpChunkData::default();
        // SAFETY: `png.png` is valid and `chunk_state` outlives all callback invocations below.
        unsafe {
            png_set_read_user_chunk_fn(
                png.png,
                ptr::from_mut(&mut chunk_state).cast(),
                Some(crabbyavif_png_read_unknown_chunk),
            );

            let mut other = png_unknown_chunk {
                name: *b"tEXt\0",
                data: payload.as_mut_ptr(),
                size: payload.len(),
                location: 0,
            };
            assert_eq!(crabbyavif_png_read_unknown_chunk(png.png, &mut other), 1);
            assert_eq!(chunk_state.cicp, None);

            let mut short_cicp = png_unknown_chunk {
                name: *b"cICP\0",
                data: payload.as_mut_ptr(),
                size: 3,
                location: 0,
            };
            assert_eq!(
                crabbyavif_png_read_unknown_chunk(png.png, &mut short_cicp),
                1
            );
            assert_eq!(chunk_state.cicp, None);

            let mut null_data_cicp = png_unknown_chunk {
                name: *b"cICP\0",
                data: ptr::null_mut(),
                size: 4,
                location: 0,
            };
            assert_eq!(
                crabbyavif_png_read_unknown_chunk(png.png, &mut null_data_cicp),
                1
            );
            assert_eq!(chunk_state.cicp, None);

            assert_eq!(
                crabbyavif_png_read_unknown_chunk(png.png, &mut valid_cicp),
                1
            );
            assert_eq!(chunk_state.cicp, Some([9, 16, 0, 1]));
        }
    }

    #[test]
    fn read_data_callback_ignores_zero_len_and_null_png_ptr() {
        // SAFETY: `crabbyavif_png_read_data` returns early before dereferencing any pointer when
        // `length == 0` or `png_ptr` is null.
        unsafe {
            crabbyavif_png_read_data(ptr::null_mut(), ptr::null_mut(), 0);
            crabbyavif_png_read_data(ptr::null_mut(), ptr::null_mut(), 16);
            let png = PngReaderNative::new().unwrap();
            crabbyavif_png_read_data(png.png, ptr::null_mut(), 0);
        }
    }

    fn encode_raw_profile(name: &str, payload: &[u8]) -> std::ffi::CString {
        let hex: String = payload.iter().map(|b| format!("{b:02x}")).collect();
        std::ffi::CString::new(format!("\n{name}\n{}\n{hex}\n", payload.len())).unwrap()
    }

    fn set_png_text_entry(png: &PngReaderNative, compression: i32, key: &[u8], val: &mut [u8]) {
        let text = png_text {
            compression,
            key: key.as_ptr().cast::<c_char>().cast_mut(),
            text: val.as_mut_ptr().cast::<c_char>(),
            text_length: val.len().saturating_sub(1),
            itxt_length: val.len().saturating_sub(1),
            lang: ptr::null_mut(),
            lang_key: ptr::null_mut(),
        };
        // SAFETY: `png` owns valid `libpng` structs and `key`/`val` are NUL-terminated.
        unsafe {
            png_set_text(png.png, png.info, ptr::from_ref(&text), 1);
        }
    }

    #[test]
    fn extract_exif_and_xmp_handles_itxt_and_raw_profiles() {
        // 1. iTXt XML:com.adobe.xmp (strips trailing single NUL) + binary eXIf chunk.
        let png = PngReaderNative::new().unwrap();
        let mut exif_bytes = [0x49u8, 0x49, 0x2A, 0x00];
        // SAFETY: `png` is valid and `exif_bytes` is initialized.
        unsafe {
            png_set_eXIf_1(
                png.png,
                png.info,
                exif_bytes.len() as u32,
                exif_bytes.as_mut_ptr(),
            );
        }
        // Append an explicit NUL inside the payload plus the C-string NUL terminator so
        // `png_set_text` (`strlen`) copies the non-NUL prefix and `remove_trailing_null_from_xmp`
        // handles trailing NULs cleanly.
        let mut xmp_itxt = b"<x:xmpmeta/>\0".to_vec();
        set_png_text_entry(
            &png,
            PNG_ITXT_COMPRESSION_NONE as c_int,
            b"XML:com.adobe.xmp\0",
            &mut xmp_itxt,
        );
        let mut img = Image::default();
        let (mut ignore_exif, mut ignore_xmp) = (false, false);
        png.extract_exif_and_xmp(&mut img, &mut ignore_exif, &mut ignore_xmp)
            .unwrap();
        assert!(ignore_exif);
        assert!(ignore_xmp);
        assert_eq!(img.exif, exif_bytes);
        assert_eq!(img.xmp, b"<x:xmpmeta/>");

        // When ignore flags are already set, existing fields remain untouched.
        let mut skipped = Image::default();
        png.extract_exif_and_xmp(&mut skipped, &mut ignore_exif, &mut ignore_xmp)
            .unwrap();
        assert!(skipped.exif.is_empty());
        assert!(skipped.xmp.is_empty());

        // 2. Raw profile type exif and Raw profile type xmp strip EXIF_PREFIX / XMP_PREFIX.
        let png_raw = PngReaderNative::new().unwrap();
        let mut raw_exif =
            encode_raw_profile("exif", b"Exif\0\0\x4d\x4d\x00\x2a").into_bytes_with_nul();
        let mut raw_xmp = encode_raw_profile("xmp", b"http://ns.adobe.com/xap/1.0/\0<rdf/>")
            .into_bytes_with_nul();
        set_png_text_entry(
            &png_raw,
            PNG_TEXT_COMPRESSION_NONE,
            b"Raw profile type exif\0",
            &mut raw_exif,
        );
        set_png_text_entry(
            &png_raw,
            PNG_TEXT_COMPRESSION_NONE,
            b"Raw profile type xmp\0",
            &mut raw_xmp,
        );
        let mut img_raw = Image::default();
        let (mut ig_e, mut ig_x) = (false, false);
        png_raw
            .extract_exif_and_xmp(&mut img_raw, &mut ig_e, &mut ig_x)
            .unwrap();
        assert_eq!(img_raw.exif, b"\x4d\x4d\x00\x2a");
        assert_eq!(img_raw.xmp, b"<rdf/>");

        // 3. Raw profile type APP1 / app1 routes Exif and XMP prefixes independently.
        let png_app1 = PngReaderNative::new().unwrap();
        let mut app1_exif = encode_raw_profile("APP1", b"Exif\0\0ABCD").into_bytes_with_nul();
        let mut app1_xmp =
            encode_raw_profile("app1", b"http://ns.adobe.com/xap/1.0/\0XYZ").into_bytes_with_nul();
        set_png_text_entry(
            &png_app1,
            PNG_TEXT_COMPRESSION_NONE,
            b"Raw profile type APP1\0",
            &mut app1_exif,
        );
        set_png_text_entry(
            &png_app1,
            PNG_TEXT_COMPRESSION_NONE,
            b"Raw profile type app1\0",
            &mut app1_xmp,
        );
        let mut img_app1 = Image::default();
        let (mut ig_e2, mut ig_x2) = (false, false);
        png_app1
            .extract_exif_and_xmp(&mut img_app1, &mut ig_e2, &mut ig_x2)
            .unwrap();
        assert_eq!(img_app1.exif, b"ABCD");
        assert_eq!(img_app1.xmp, b"XYZ");
    }

    #[test]
    fn read_frame_roundtrips_8bit_and_16bit_rows_and_rejects_bad_header() {
        use crate::utils::writer::png::PngWriter;
        use crate::utils::writer::Writer;
        use crate::Category;

        for depth in [8u8, 12u8] {
            let mut src = Image {
                width: 3,
                height: 2,
                depth,
                yuv_format: PixelFormat::Yuv444,
                matrix_coefficients: MatrixCoefficients::Identity,
                color_primaries: ColorPrimaries::Srgb,
                transfer_characteristics: TransferCharacteristics::Srgb,
                ..Image::default()
            };
            src.allocate_planes(Category::Color).unwrap();
            let max_val = src.max_channel();
            for (i, plane) in [Plane::Y, Plane::U, Plane::V].into_iter().enumerate() {
                src.fill_plane_with_value(plane, max_val / ((i as u16) + 2))
                    .unwrap();
            }

            let path = std::env::temp_dir().join(format!(
                "crabbyavif_png_roundtrip_{depth}_{}.png",
                std::process::id()
            ));
            {
                let mut f = File::create(&path).unwrap();
                let mut writer = PngWriter::default();
                writer.write_frame(&mut f, &src).unwrap();
            }
            let mut reader = PngReader::create(path.to_str().unwrap()).unwrap();
            let (decoded, _, _) = reader
                .read_frame(&Config {
                    depth: Some(depth),
                    yuv_format: Some(PixelFormat::Yuv444),
                    matrix_coefficients: Some(MatrixCoefficients::Identity),
                    ..Config::default()
                })
                .unwrap();
            let _ = std::fs::remove_file(&path);
            assert_eq!(decoded.width, 3);
            assert_eq!(decoded.height, 2);
            assert_eq!(decoded.depth, depth);
            for plane in [Plane::Y, Plane::U, Plane::V] {
                for y in 0..decoded.height {
                    if depth == 8 {
                        assert_eq!(decoded.row(plane, y).unwrap(), src.row(plane, y).unwrap());
                    } else {
                        assert_eq!(
                            decoded.row16(plane, y).unwrap(),
                            src.row16(plane, y).unwrap()
                        );
                    }
                }
            }
        }

        let bad_path =
            std::env::temp_dir().join(format!("crabbyavif_png_bad_sig_{}.png", std::process::id()));
        std::fs::write(&bad_path, b"notapng!").unwrap();
        let mut bad_reader = PngReader::create(bad_path.to_str().unwrap()).unwrap();
        assert!(bad_reader.read_frame(&Config::default()).is_err());
        let _ = std::fs::remove_file(&bad_path);
    }
}
