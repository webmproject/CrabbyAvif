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

// The type of the fields from dav1d_sys::bindings::* are dependent on the
// compiler that is used to generate the bindings, version of dav1d, etc.
// So allow clippy to ignore unnecessary cast warnings.
#![allow(clippy::unnecessary_cast)]

use crate::codecs::Decoder;
use crate::codecs::DecoderConfig;
use crate::decoder::item::Item;
use crate::decoder::GridImageHelper;
use crate::image::Image;
use crate::image::YuvRange;
use crate::internal_utils::u32_from_usize;
use crate::internal_utils::TryClone;
use crate::utils::pixels::*;
use crate::*;

use dav1d_sys::bindings::*;

use std::ffi::CStr;
use std::mem::MaybeUninit;

#[derive(Default)]
pub struct Dav1d {
    context: Option<*mut Dav1dContext>,
    picture: Option<Dav1dPictureWrapper>,
    config: Option<DecoderConfig>,
}

// See https://code.videolan.org/videolan/dav1d/-/blob/9849ede1304da1443cfb4a86f197765081034205/include/dav1d/common.h#L55-59
const DAV1D_EAGAIN: i32 = if libc::EPERM > 0 { -libc::EAGAIN } else { libc::EAGAIN };

struct Dav1dPictureWrapper {
    picture: Dav1dPicture,
}

impl Default for Dav1dPictureWrapper {
    fn default() -> Self {
        Self {
            // SAFETY: Zero-initializing the C `Dav1dPicture` struct matches the C API usage
            // before passing it to `dav1d_get_picture` and remains compatible across dav1d
            // versions.
            picture: unsafe { std::mem::zeroed() },
        }
    }
}

impl Dav1dPictureWrapper {
    fn mut_ptr(&mut self) -> *mut Dav1dPicture {
        &mut self.picture
    }

    fn get(&self) -> &Dav1dPicture {
        &self.picture
    }

    fn use_layer(&self, spatial_id: u8) -> bool {
        if spatial_id == 0xFF {
            return true;
        }
        let frame_hdr = self.get().frame_hdr;
        if frame_hdr.is_null() {
            return false;
        }
        // SAFETY: `self.picture` was populated by a successful `dav1d_get_picture` call (`res == 0`),
        // and `frame_hdr` was verified to be non-null above. `dav1d` guarantees `frame_hdr` points
        // to an initialized `Dav1dFrameHeader` valid for the lifetime of this `Dav1dPicture` ref.
        let pic_spatial_id = unsafe { (*frame_hdr).spatial_id };
        spatial_id == pic_spatial_id as u8
    }
}

impl Drop for Dav1dPictureWrapper {
    fn drop(&mut self) {
        let ptr = self.mut_ptr();
        // SAFETY: `ptr` points to `self.picture`, which is either zero-initialized by `Default` or
        // populated by `dav1d_get_picture`. `dav1d_picture_unref` safely handles both states,
        // decrements any non-null internal `Dav1dRef`s, and zeroes the struct.
        unsafe {
            dav1d_picture_unref(ptr);
        }
    }
}

struct Dav1dDataWrapper {
    data: Dav1dData,
}

impl Default for Dav1dDataWrapper {
    fn default() -> Self {
        Self {
            // SAFETY: Zero-initializing the C `Dav1dData` struct matches the C API usage
            // before passing it to `dav1d_data_create` and remains compatible across dav1d
            // versions.
            data: unsafe { std::mem::zeroed() },
        }
    }
}

impl Dav1dDataWrapper {
    fn mut_ptr(&mut self) -> *mut Dav1dData {
        &mut self.data
    }

    fn has_data(&self) -> bool {
        self.data.sz > 0 && !self.data.data.is_null()
    }

    fn unref(&mut self) {
        if self.has_data() || !self.data.ref_.is_null() {
            let ptr = self.mut_ptr();
            // SAFETY: `ptr` points to `self.data`, which is either zero-initialized by `Default`
            // or managed by `dav1d_data_create` / `dav1d_send_data`. `dav1d_data_unref` safely
            // releases any owned `Dav1dRef` and resets `self.data` to empty.
            unsafe {
                dav1d_data_unref(ptr);
            }
        }
    }

    fn wrap(&mut self, payload: &[u8]) -> AvifResult<()> {
        self.unref();
        let data_ptr = self.mut_ptr();
        let len = payload.len();
        // SAFETY: `data_ptr` is a valid mutable pointer to `self.data` (which has no active
        // reference after `self.unref()`), and `dav1d_data_create` validates `len` internally.
        let data = unsafe { dav1d_data_create(data_ptr, len as _) };
        if data.is_null() {
            return Err(AvifError::UnknownError(
                "dav1d_data_create returned nullptr".into(),
            ));
        }
        if !payload.is_empty() {
            let src = payload.as_ptr();
            // SAFETY: `dav1d_data_create` succeeded and returned a non-null pointer `data` to a
            // newly heap-allocated buffer of at least `len` bytes disjoint from `payload`. Note
            // that `dav1d_data_create` returns uninitialized memory, so `copy_nonoverlapping` is
            // required here instead of constructing `&mut [u8]` via `slice::from_raw_parts_mut`.
            unsafe {
                std::ptr::copy_nonoverlapping(src, data, len);
            }
        }
        Ok(())
    }
}

impl Drop for Dav1dDataWrapper {
    fn drop(&mut self) {
        self.unref();
    }
}

impl Dav1d {
    fn initialize_impl(&mut self, low_latency: bool) -> AvifResult<()> {
        if self.context.is_some() {
            return Ok(());
        }
        let config = self.config.as_ref().ok_or(AvifError::InvalidArgument)?;
        let mut settings_uninit: MaybeUninit<Dav1dSettings> = MaybeUninit::uninit();
        let settings_mut_ptr = settings_uninit.as_mut_ptr();
        // SAFETY: `settings_mut_ptr` is a valid, aligned, non-null pointer to `settings_uninit`.
        unsafe { dav1d_default_settings(settings_mut_ptr) };
        // SAFETY: `dav1d_default_settings` completely initialized `settings_uninit`.
        let mut settings = unsafe { settings_uninit.assume_init() };
        if low_latency {
            settings.max_frame_delay = 1;
        }
        settings.n_threads = i32::try_from(config.max_threads)
            .unwrap_or(1)
            .clamp(1, DAV1D_MAX_THREADS as _);
        settings.operating_point = config.operating_point as i32;
        settings.all_layers = if config.all_layers { 1 } else { 0 };
        let frame_size_limit = match config.image_size_limit {
            Some(value) => value.get(),
            None => 0,
        };
        // Set a maximum frame size limit to avoid OOM'ing fuzzers. In 32-bit builds, if
        // frame_size_limit > 8192 * 8192, dav1d reduces frame_size_limit to 8192 * 8192 and logs
        // a message, so we set frame_size_limit to at most 8192 * 8192 to avoid the dav1d_log
        // message.
        settings.frame_size_limit = if cfg!(target_pointer_width = "32") {
            frame_size_limit.min(8192 * 8192)
        } else {
            frame_size_limit
        };

        let mut dec: *mut Dav1dContext = std::ptr::null_mut();
        let dec_ptr: *mut *mut Dav1dContext = &mut dec;
        let settings_ptr: *const Dav1dSettings = &settings;
        // SAFETY: `dec_ptr` and `settings_ptr` are valid non-null pointers to initialized stack
        // variables; `settings` was populated by `dav1d_default_settings` and valid config values.
        let ret = unsafe { dav1d_open(dec_ptr, settings_ptr) };
        if ret != 0 || dec.is_null() {
            return AvifError::unknown_error(format!("dav1d_open returned {ret}"));
        }
        self.context = Some(dec);
        Ok(())
    }

    fn drop_impl(&mut self) {
        self.picture = None;
        if let Some(mut context) = self.context.take() {
            let context_ptr: *mut *mut Dav1dContext = &mut context;
            // SAFETY: `context` was returned by a successful `dav1d_open` call and has not yet
            // been closed (`self.context.take()` ensures single close). `self.picture` was dropped
            // above before closing the decoder context.
            unsafe { dav1d_close(context_ptr) };
        }
    }

    fn picture_to_image(
        &self,
        dav1d_picture: &Dav1dPicture,
        image: &mut Image,
        category: Category,
        overwrite_alpha_plane_yuv_range: bool,
    ) -> AvifResult<()> {
        let seq_hdr_ptr = dav1d_picture.seq_hdr;
        if seq_hdr_ptr.is_null() {
            return AvifError::unknown_error("dav1d_picture.seq_hdr is null");
        }
        // SAFETY: `dav1d_picture` was populated by a successful `dav1d_get_picture` call,
        // `seq_hdr_ptr` was checked to be non-null above, and `dav1d` keeps `seq_hdr` valid and
        // immutable for the lifetime of `dav1d_picture` via `seq_hdr_ref`.
        let seq_hdr = unsafe { &*seq_hdr_ptr };
        match category {
            Category::Alpha => {
                if image.width > 0
                    && image.height > 0
                    && (image.width != (dav1d_picture.p.w as u32)
                        || image.height != (dav1d_picture.p.h as u32)
                        || image.depth != (dav1d_picture.p.bpc as u8))
                {
                    // Alpha plane does not match the previous alpha plane.
                    return AvifError::unknown_error("");
                }
                image.width = dav1d_picture.p.w as u32;
                image.height = dav1d_picture.p.h as u32;
                image.depth = dav1d_picture.p.bpc as u8;
                image.row_bytes[3] = dav1d_picture.stride[0] as u32;
                image.planes[3] = Some(Pixels::from_raw_pointer(
                    dav1d_picture.data[0] as *mut u8,
                    image.depth as u32,
                    image.height,
                    image.row_bytes[3],
                )?);
                if overwrite_alpha_plane_yuv_range {
                    image.yuv_range =
                        if seq_hdr.color_range == 0 { YuvRange::Limited } else { YuvRange::Full };
                }
            }
            _ => {
                image.width = dav1d_picture.p.w as u32;
                image.height = dav1d_picture.p.h as u32;
                image.depth = dav1d_picture.p.bpc as u8;

                image.yuv_format = match dav1d_picture.p.layout {
                    0 => PixelFormat::Yuv400,
                    1 => PixelFormat::Yuv420,
                    2 => PixelFormat::Yuv422,
                    3 => PixelFormat::Yuv444,
                    _ => return AvifError::unknown_error(""), // not reached.
                };
                image.yuv_range =
                    if seq_hdr.color_range == 0 { YuvRange::Limited } else { YuvRange::Full };
                image.chroma_sample_position = (seq_hdr.chr as u32).into();

                image.color_primaries = (seq_hdr.pri as u16).into();
                image.transfer_characteristics = (seq_hdr.trc as u16).into();
                image.matrix_coefficients = (seq_hdr.mtrx as u16).into();

                for plane in 0usize..image.yuv_format.plane_count() {
                    let stride_index = if plane == 0 { 0 } else { 1 };
                    image.row_bytes[plane] = dav1d_picture.stride[stride_index] as u32;
                    image.planes[plane] = Some(Pixels::from_raw_pointer(
                        dav1d_picture.data[plane] as *mut u8,
                        image.depth as u32,
                        u32_from_usize(image.height(plane.into()))?,
                        image.row_bytes[plane],
                    )?);
                }
                if image.yuv_format == PixelFormat::Yuv400 {
                    // Clear left over chroma planes from previous frames.
                    image.clear_chroma_planes();
                }
            }
        }
        Ok(())
    }

    fn get_next_image_grid_impl(
        &mut self,
        payloads: &[Vec<u8>],
        spatial_id: u8,
        grid_image_helper: &mut GridImageHelper,
    ) -> AvifResult<()> {
        if self.context.is_none() {
            self.initialize_impl(false)?;
        }
        let mut res;
        let context = self.context.unwrap();
        let mut payloads_iter = payloads.iter().peekable();
        let mut data = Dav1dDataWrapper::default();
        let max_retries = 500;
        let mut retries = 0;
        while !grid_image_helper.is_grid_complete()? {
            if !data.has_data() && payloads_iter.peek().is_some() {
                data.wrap(payloads_iter.next().unwrap())?;
            }
            if data.has_data() {
                // # Safety: Calling a C function with valid parameters.
                res = unsafe { dav1d_send_data(context, data.mut_ptr()) };
                if res != 0 && res != DAV1D_EAGAIN {
                    return AvifError::unknown_error(format!("dav1d_send_data returned {res}"));
                }
            }
            let mut picture = Dav1dPictureWrapper::default();
            // # Safety: Calling a C function with valid parameters.
            res = unsafe { dav1d_get_picture(context, picture.mut_ptr()) };
            if res != 0 && res != DAV1D_EAGAIN {
                return AvifError::unknown_error(format!("dav1d_get_picture returned {res}"));
            } else if res == 0 && picture.use_layer(spatial_id) {
                let mut cell_image = Image::default();
                self.picture_to_image(
                    picture.get(),
                    &mut cell_image,
                    grid_image_helper.category,
                    true,
                )?;
                grid_image_helper.copy_from_cell_image(&mut cell_image)?;
                retries = 0;
            } else {
                retries += 1;
                if retries > max_retries {
                    return AvifError::unknown_error(format!(
                        "dav1d_get_picture never returned a frame after {max_retries} calls"
                    ));
                }
            }
        }
        self.flush()?;
        Ok(())
    }

    fn flush(&mut self) -> AvifResult<()> {
        loop {
            let mut picture = Dav1dPictureWrapper::default();
            // # Safety: Calling a C function with valid parameters.
            let res = unsafe { dav1d_get_picture(self.context.unwrap(), picture.mut_ptr()) };
            if res < 0 && res != DAV1D_EAGAIN {
                return AvifError::unknown_error(format!("error draining buffered frames {res}"));
            }
            if res != 0 {
                break;
            }
        }
        Ok(())
    }

    pub(crate) fn version() -> String {
        // SAFETY: `dav1d_version` has no preconditions and returns a static NUL-terminated string.
        let version_ptr = unsafe { dav1d_version() };
        let version = if version_ptr.is_null() {
            String::new()
        } else {
            // SAFETY: `version_ptr` was checked to be non-null above and points to a static
            // NUL-terminated string literal owned by libdav1d.
            let version_cstr = unsafe { CStr::from_ptr(version_ptr) };
            version_cstr.to_str().unwrap_or_default().to_owned()
        };
        format!("dav1d: {version}")
    }
}

impl Decoder for Dav1d {
    fn codec(&self) -> CodecChoice {
        CodecChoice::Dav1d
    }

    fn initialize(&mut self, config: &DecoderConfig) -> AvifResult<()> {
        self.config = Some(DecoderConfig {
            codec_config: config.codec_config.try_clone()?,
            ..*config
        });
        Ok(())
    }

    fn get_next_image(
        &mut self,
        av1_payload: &[u8],
        spatial_id: u8,
        image: &mut Image,
        category: Category,
        _item: Option<&Item>,
        _signal_eos: bool,
    ) -> AvifResult<()> {
        if self.context.is_none() {
            self.initialize_impl(true)?;
        }
        let mut data = Dav1dDataWrapper::default();
        data.wrap(av1_payload)?;
        let next_picture: Option<Dav1dPictureWrapper>;
        loop {
            if data.has_data() {
                // # Safety: Calling a C function with valid parameters.
                let res = unsafe { dav1d_send_data(self.context.unwrap(), data.mut_ptr()) };
                if res < 0 && res != DAV1D_EAGAIN {
                    return AvifError::unknown_error(format!("dav1d_send_data returned {res}"));
                }
            }

            let mut picture = Dav1dPictureWrapper::default();
            // # Safety: Calling a C function with valid parameters.
            let res = unsafe { dav1d_get_picture(self.context.unwrap(), picture.mut_ptr()) };
            if res == DAV1D_EAGAIN {
                if data.has_data() {
                    continue;
                }
                return AvifError::unknown_error("");
            } else if res < 0 {
                return AvifError::unknown_error(format!("dav1d_send_picture returned {res}"));
            } else if picture.use_layer(spatial_id) {
                // Got a picture.
                next_picture = Some(picture);
                break;
            }
        }
        self.flush()?;
        if next_picture.is_some() {
            self.picture = next_picture;
        } else if category == Category::Alpha && self.picture.is_some() {
            // Special case for alpha, re-use last frame.
        } else {
            return AvifError::unknown_error("");
        }
        self.picture_to_image(self.picture.unwrap_ref().get(), image, category, true)?;
        Ok(())
    }

    fn get_next_image_grid(
        &mut self,
        payloads: &[Vec<u8>],
        spatial_id: u8,
        grid_image_helper: &mut GridImageHelper,
    ) -> AvifResult<()> {
        let res = self.get_next_image_grid_impl(payloads, spatial_id, grid_image_helper);
        if res.is_err() {
            self.drop_impl();
        }
        res
    }

    fn get_last_image(
        &mut self,
        payloads: &[Vec<u8>],
        spatial_id: u8,
        image: &mut Image,
        category: Category,
    ) -> AvifResult<()> {
        if self.context.is_none() {
            self.initialize_impl(false)?;
        }
        let mut res;
        let context = self.context.unwrap();
        let mut payloads_iter = payloads.iter().peekable();
        let mut data = Dav1dDataWrapper::default();
        let max_retries = 500;
        let mut retries = 0;
        let mut got_picture_count = 0;
        let next_picture: Option<Dav1dPictureWrapper>;
        loop {
            if !data.has_data() && payloads_iter.peek().is_some() {
                data.wrap(payloads_iter.next().unwrap())?;
            }
            if data.has_data() {
                // # Safety: Calling a C function with valid parameters.
                res = unsafe { dav1d_send_data(context, data.mut_ptr()) };
                if res != 0 && res != DAV1D_EAGAIN {
                    return AvifError::unknown_error(format!("dav1d_send_data returned {res}"));
                }
            }
            let mut picture = Dav1dPictureWrapper::default();
            // # Safety: Calling a C function with valid parameters.
            res = unsafe { dav1d_get_picture(context, picture.mut_ptr()) };
            if res != 0 && res != DAV1D_EAGAIN {
                return AvifError::unknown_error(format!("dav1d_get_picture returned {res}"));
            } else if res == 0 && picture.use_layer(spatial_id) {
                got_picture_count += 1;
                retries = 0;
                if got_picture_count == payloads.len() {
                    next_picture = Some(picture);
                    break;
                }
            } else {
                retries += 1;
                if retries > max_retries {
                    return AvifError::unknown_error(format!(
                        "dav1d_get_picture never returned a frame after {max_retries} calls"
                    ));
                }
            }
        }
        self.flush()?;
        if next_picture.is_some() {
            self.picture = next_picture;
        } else if category == Category::Alpha && self.picture.is_some() {
            // Special case for alpha, re-use last frame.
        } else {
            return AvifError::unknown_error("");
        }
        // In this case, the source image from the caller is used for both color and alpha planes,
        // so do not overwrite the yuv_range field when processing the alpha plane.
        self.picture_to_image(self.picture.unwrap_ref().get(), image, category, false)?;
        Ok(())
    }
}

impl Drop for Dav1d {
    fn drop(&mut self) {
        self.drop_impl();
    }
}
