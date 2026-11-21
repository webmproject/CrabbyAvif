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
use crate::internal_utils::create_vec_exact;
use crate::internal_utils::u32_from_usize;
use crate::internal_utils::TryClone;
use crate::internal_utils::VecExtension;
use crate::utils::pixels::*;
use crate::*;

use dav1d_sys::bindings::*;

use std::ffi::CStr;
use std::mem::MaybeUninit;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

#[derive(Default)]
pub struct Dav1d {
    context: Option<*mut Dav1dContext>,
    picture: Option<Dav1dPictureWrapper>,
    config: Option<DecoderConfig>,
    low_latency: bool,
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

struct SendCell {
    // Dropped before `_picture` so pointer slices remain valid for `image`'s lifetime.
    image: Image,
    _picture: Dav1dPictureWrapper,
}

// SAFETY: `Dav1d` uses `dav1d`'s default allocator (not a thread-affine custom allocator), and
// `decode_grid_cell()` flushes the context before constructing `SendCell` so no worker task threads
// or `c->refs` retain references to `_picture`'s pixel buffers. `Dav1dPictureWrapper` owns a
// ref-counted `Dav1dPicture` whose `Dav1dRef` / mutex-protected `Dav1dMemPool` are thread-safe to
// unref from another thread. `image` holds `Pixels::Pointer` / `Pixels::Pointer16` slices
// referencing only `_picture`'s pixel buffers, and field drop order ensures `image` is dropped
// before `_picture`.
unsafe impl Send for SendCell {}

impl Dav1d {
    // Safe to call repeatedly; reuses the existing context when `low_latency` matches and
    // recreates it otherwise.
    fn initialize_impl(&mut self, low_latency: bool) -> AvifResult<()> {
        if self.context.is_some() {
            if self.low_latency == low_latency {
                return Ok(());
            }
            // Do not reuse a Dav1dContext across incompatible low_latency modes.
            self.drop_impl();
        }
        let config = self.config.as_ref().ok_or(AvifError::InvalidArgument)?;
        let mut settings_uninit: MaybeUninit<Dav1dSettings> = MaybeUninit::zeroed();
        let settings_mut_ptr = settings_uninit.as_mut_ptr();
        // SAFETY: `settings_mut_ptr` is a valid, aligned, non-null pointer to zero-initialized
        // `settings_uninit`.
        unsafe { dav1d_default_settings(settings_mut_ptr) };
        // SAFETY: `settings_uninit` was zero-initialized (including `reserved` and any padding)
        // and all active fields were populated by `dav1d_default_settings`.
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
        self.low_latency = low_latency;
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

    fn send_data(&self, data: &mut Dav1dDataWrapper) -> AvifResult<i32> {
        let context = self
            .context
            .ok_or_else(|| AvifError::UnknownError("dav1d context is not initialized".into()))?;
        let data_ptr = data.mut_ptr();
        // SAFETY: `context` is a valid non-null `*mut Dav1dContext` opened by `dav1d_open` and
        // owned by `self`, and `data_ptr` points to a valid `Dav1dData` managed by `data`.
        Ok(unsafe { dav1d_send_data(context, data_ptr) })
    }

    fn get_picture(&self, picture: &mut Dav1dPictureWrapper) -> AvifResult<i32> {
        let context = self
            .context
            .ok_or_else(|| AvifError::UnknownError("dav1d context is not initialized".into()))?;
        let picture_ptr = picture.mut_ptr();
        // SAFETY: `context` is a valid non-null `*mut Dav1dContext` opened by `dav1d_open` and
        // owned by `self`, and `picture_ptr` points to a valid zero-initialized `Dav1dPicture`.
        Ok(unsafe { dav1d_get_picture(context, picture_ptr) })
    }

    fn picture_to_image(
        &self,
        dav1d_picture: &Dav1dPicture,
        image: &mut Image,
        category: Category,
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
                image.yuv_range =
                    if seq_hdr.color_range == 0 { YuvRange::Limited } else { YuvRange::Full };
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

    // Points the `category` planes of `image` to `next_picture`. `self.picture` is replaced only
    // if this succeeds, so that `image` never points to a released picture. On failure, the
    // `category` planes of `image` are cleared.
    fn set_picture_and_image(
        &mut self,
        next_picture: Dav1dPictureWrapper,
        image: &mut Image,
        category: Category,
    ) -> AvifResult<()> {
        let res = self.picture_to_image(next_picture.get(), image, category);
        if res.is_ok() {
            self.picture = Some(next_picture);
        } else {
            image.free_planes(category.planes());
        }
        res
    }

    fn flush_context(&mut self) {
        if let Some(context) = self.context {
            // SAFETY: `context` is a valid non-null `*mut Dav1dContext` opened by `dav1d_open`
            // and owned by `self`.
            unsafe { dav1d_flush(context) };
        }
    }

    fn decode_grid_cell(
        &mut self,
        payload: &[u8],
        spatial_id: u8,
        category: Category,
    ) -> AvifResult<SendCell> {
        let mut cell_image = Image::default();
        self.get_next_image(payload, spatial_id, &mut cell_image, category, None, true)?;
        let picture = self
            .picture
            .take()
            .ok_or_else(|| AvifError::UnknownError("dav1d returned no picture".into()))?;
        // Reset sequence and reference frame state so malformed cells cannot leak state into
        // subsequent cells on the same worker context.
        self.flush_context();
        Ok(SendCell {
            image: cell_image,
            _picture: picture,
        })
    }

    fn grid_thread_counts(max_threads: u32, payload_count: usize) -> (usize, usize) {
        const MAX_GRID_WORKERS: usize = 32;
        // Match `initialize_impl`'s `i32::try_from` fallback so out-of-range counts stay 1.
        let clamped_threads = i32::try_from(max_threads)
            .ok()
            .and_then(|t| usize::try_from(t).ok())
            .unwrap_or(1)
            .clamp(1, DAV1D_MAX_THREADS as usize);
        let worker_count = clamped_threads
            .min(payload_count)
            .clamp(1, MAX_GRID_WORKERS);
        (clamped_threads, worker_count)
    }

    fn get_next_image_grid_impl(
        &mut self,
        payloads: &[Vec<u8>],
        spatial_id: u8,
        grid_image_helper: &mut GridImageHelper,
    ) -> AvifResult<()> {
        if grid_image_helper.is_grid_complete()? {
            return Ok(());
        }
        if payloads.is_empty() {
            return AvifError::unknown_error("incomplete grid payloads");
        }
        let base_config = self.config.as_ref().ok_or(AvifError::InvalidArgument)?;
        let (max_threads, worker_count) =
            Self::grid_thread_counts(base_config.max_threads, payloads.len());
        if worker_count == 1 {
            for payload in payloads {
                if grid_image_helper.is_grid_complete()? {
                    break;
                }
                let mut cell =
                    self.decode_grid_cell(payload, spatial_id, grid_image_helper.category)?;
                grid_image_helper.copy_from_cell_image(&mut cell.image)?;
            }
            if !grid_image_helper.is_grid_complete()? {
                return AvifError::unknown_error("incomplete grid payloads");
            }
            return Ok(());
        }

        let per_worker_threads = u32_from_usize((max_threads / worker_count).max(1))?;
        let worker_config = DecoderConfig {
            max_threads: per_worker_threads,
            codec_config: base_config.codec_config.try_clone()?,
            ..*base_config
        };
        let category = grid_image_helper.category;
        let next_index = AtomicUsize::new(0);

        std::thread::scope(|scope| {
            // Keep `frames` inside `scope` so buffered `SendCell`s drop before returning.
            // `SendCell` relies on `Dav1dMemPool`'s mutex-protected refcounting to unref safely
            // even after a worker closes its `Dav1dContext`.
            let mut frames: Vec<Option<SendCell>> = create_vec_exact(payloads.len())?;
            for _ in payloads {
                frames.try_push(None)?;
            }
            let (sender, receiver) = std::sync::mpsc::channel();
            let mut handles = create_vec_exact(worker_count)?;
            let spawn_result: AvifResult<()> = (|| {
                for _ in 0..worker_count {
                    // Cloning Sender only increments an atomic reference count and does not
                    // allocate buffer memory.
                    #[allow(clippy::disallowed_methods)]
                    let sender = sender.clone();
                    let next_index = &next_index;
                    let worker_config = &worker_config;
                    let handle = std::thread::Builder::new()
                        .spawn_scoped(scope, move || {
                            let mut dav1d = Dav1d::default();
                            if let Err(err) = dav1d.initialize(worker_config) {
                                let _ = sender.send(Err(err));
                                return;
                            }
                            loop {
                                let cell_index = next_index.fetch_add(1, Ordering::Relaxed);
                                let Some(payload) = payloads.get(cell_index) else {
                                    break;
                                };
                                let res = dav1d
                                    .decode_grid_cell(payload, spatial_id, category)
                                    .map(|cell| (cell_index, cell));
                                let is_err = res.is_err();
                                if sender.send(res).is_err() || is_err {
                                    break;
                                }
                            }
                        })
                        .map_err(AvifError::map_unknown_error)?;
                    handles.try_push(handle)?;
                }
                Ok(())
            })();
            drop(sender);
            if let Err(err) = spawn_result {
                drop(receiver);
                drop(frames);
                for handle in handles {
                    let _ = handle.join();
                }
                return Err(err);
            }

            let mut result: AvifResult<()> = (|| {
                let mut next_frame = 0;
                'receive: for msg in receiver.iter() {
                    let (index, cell) = msg?;
                    *frames.get_mut(index).ok_or_else(|| {
                        AvifError::UnknownError("dav1d: invalid cell index".into())
                    })? = Some(cell);
                    while let Some(mut cell) = frames.get_mut(next_frame).and_then(Option::take) {
                        grid_image_helper.copy_from_cell_image(&mut cell.image)?;
                        next_frame += 1;
                        if grid_image_helper.is_grid_complete()? {
                            break 'receive;
                        }
                    }
                }
                Ok(())
            })();

            drop(receiver);
            drop(frames);
            for handle in handles {
                if handle.join().is_err() && result.is_ok() {
                    result = AvifError::unknown_error("dav1d: decoding thread panicked");
                }
            }
            if result.is_ok() && !grid_image_helper.is_grid_complete()? {
                result = AvifError::unknown_error("incomplete grid payloads");
            }
            result
        })
    }

    fn flush(&mut self) -> AvifResult<()> {
        loop {
            let mut picture = Dav1dPictureWrapper::default();
            let res = self.get_picture(&mut picture)?;
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
        // Do not leave stale planes in `image` if decoding fails before reaching
        // set_picture_and_image(), because the other fields of `image` may have changed since.
        image.free_planes(category.planes());
        self.initialize_impl(true)?;
        let mut data = Dav1dDataWrapper::default();
        data.wrap(av1_payload)?;
        let next_picture: Dav1dPictureWrapper;
        loop {
            if data.has_data() {
                let res = self.send_data(&mut data)?;
                if res < 0 && res != DAV1D_EAGAIN {
                    return AvifError::unknown_error(format!("dav1d_send_data returned {res}"));
                }
            }

            let mut picture = Dav1dPictureWrapper::default();
            let res = self.get_picture(&mut picture)?;
            if res == DAV1D_EAGAIN {
                if data.has_data() {
                    continue;
                }
                return AvifError::unknown_error("");
            } else if res < 0 {
                return AvifError::unknown_error(format!("dav1d_send_picture returned {res}"));
            } else if picture.use_layer(spatial_id) {
                // Got a picture.
                next_picture = picture;
                break;
            }
        }
        self.flush()?;
        self.set_picture_and_image(next_picture, image, category)
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
        // Do not leave stale planes in `image` if decoding fails before reaching
        // set_picture_and_image(), because the other fields of `image` may have changed since.
        image.free_planes(category.planes());
        self.initialize_impl(false)?;
        let mut payloads_iter = payloads.iter();
        let mut data = Dav1dDataWrapper::default();
        let max_retries = 500;
        let mut retries = 0;
        let mut got_picture_count = 0;
        let next_picture: Dav1dPictureWrapper;
        loop {
            if !data.has_data() {
                if let Some(payload) = payloads_iter.next() {
                    data.wrap(payload)?;
                }
            }
            if data.has_data() {
                let res = self.send_data(&mut data)?;
                if res != 0 && res != DAV1D_EAGAIN {
                    return AvifError::unknown_error(format!("dav1d_send_data returned {res}"));
                }
            }
            let mut picture = Dav1dPictureWrapper::default();
            let res = self.get_picture(&mut picture)?;
            if res != 0 && res != DAV1D_EAGAIN {
                return AvifError::unknown_error(format!("dav1d_get_picture returned {res}"));
            } else if res == 0 && picture.use_layer(spatial_id) {
                got_picture_count += 1;
                retries = 0;
                if got_picture_count == payloads.len() {
                    next_picture = picture;
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
        self.set_picture_and_image(next_picture, image, category)
    }
}

impl Drop for Dav1d {
    fn drop(&mut self) {
        self.drop_impl();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::mp4box::{Av1CodecConfiguration, CodecConfiguration};

    #[test]
    fn reinitialize_across_low_latency_modes() {
        let mut dav1d = Dav1d::default();
        dav1d
            .initialize(&DecoderConfig {
                operating_point: 0,
                all_layers: false,
                width: 64,
                height: 64,
                depth: 8,
                max_threads: 4,
                image_size_limit: None,
                max_input_size: 0,
                codec_config: CodecConfiguration::Av1(Av1CodecConfiguration::default()),
                category: Category::Color,
                android_mediacodec_output_color_format: AndroidMediaCodecOutputColorFormat::default(
                ),
            })
            .unwrap();
        dav1d.initialize_impl(false).unwrap();
        assert!(dav1d.context.is_some());
        assert!(!dav1d.low_latency);
        let first_ctx = dav1d.context.unwrap();

        // Calling again with the same mode reuses the existing context.
        dav1d.initialize_impl(false).unwrap();
        assert_eq!(dav1d.context.unwrap(), first_ctx);

        // Switching mode recreates the context instead of reusing an incompatible one.
        dav1d.initialize_impl(true).unwrap();
        assert!(dav1d.context.is_some());
        assert!(dav1d.low_latency);
    }

    #[test]
    fn grid_thread_counts_clamping() {
        assert_eq!(Dav1d::grid_thread_counts(0, 10), (1, 1));
        assert_eq!(Dav1d::grid_thread_counts(1, 10), (1, 1));
        assert_eq!(Dav1d::grid_thread_counts(4, 10), (4, 4));
        assert_eq!(Dav1d::grid_thread_counts(16, 5), (16, 5));
        assert_eq!(
            Dav1d::grid_thread_counts(u32::MAX, 10),
            (1, 1),
            "out-of-i32-range max_threads must match initialize_impl"
        );
        assert_eq!(Dav1d::grid_thread_counts(1000, 10), (256, 10));
        assert_eq!(Dav1d::grid_thread_counts(256, 100), (256, 32));
    }

    #[test]
    fn decode_grid_cell_and_flush_on_invalid_payload() {
        let mut dav1d = Dav1d::default();
        dav1d
            .initialize(&DecoderConfig {
                operating_point: 0,
                all_layers: false,
                width: 64,
                height: 64,
                depth: 8,
                max_threads: 2,
                image_size_limit: None,
                max_input_size: 0,
                codec_config: CodecConfiguration::Av1(Av1CodecConfiguration::default()),
                category: Category::Color,
                android_mediacodec_output_color_format: AndroidMediaCodecOutputColorFormat::default(
                ),
            })
            .unwrap();
        assert!(dav1d.decode_grid_cell(&[], 0, Category::Color).is_err());
        assert!(dav1d
            .decode_grid_cell(&[0xff, 0x00], 0, Category::Color)
            .is_err());
        dav1d.flush_context();
        assert!(dav1d.picture.is_none());
    }
}
