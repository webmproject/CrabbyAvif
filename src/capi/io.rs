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

use super::types::*;

use std::ffi::CStr;
use std::os::raw::c_char;
use std::os::raw::c_void;

use crate::decoder::GenericIO;
use crate::internal_utils::check_slice_from_raw_parts_safety;
use crate::internal_utils::create_vec_exact;
use crate::internal_utils::io::DecoderFileIO;
use crate::internal_utils::io::DecoderRawIO;
use crate::internal_utils::VecExtension;
use crate::*;

#[repr(C)]
#[derive(Default)]
pub struct avifROData {
    pub data: *const u8,
    pub size: usize,
}

/// Read-write byte buffer used across the C API.
///
/// Safety invariant for Rust conversions (`TryFrom<&avifRWData>`): whenever `size > 0`, `data`
/// must be non-null and point to at least `size` initialized readable bytes (`size <= isize::MAX`).
#[repr(C)]
#[derive(Debug, Default)]
pub struct avifRWData {
    pub data: *mut u8,
    pub size: usize,
}

impl From<&Vec<u8>> for avifRWData {
    fn from(v: &Vec<u8>) -> Self {
        avifRWData {
            data: if v.is_empty() { std::ptr::null_mut() } else { v.as_ptr() as *mut u8 },
            size: v.len(),
        }
    }
}

impl TryFrom<&avifRWData> for Vec<u8> {
    type Error = AvifError;
    fn try_from(data: &avifRWData) -> AvifResult<Vec<u8>> {
        if data.size == 0 {
            return Ok(Vec::new());
        }
        if !check_slice_from_raw_parts_safety(data.data, data.size) {
            return AvifError::invalid_argument();
        }
        let mut vec = create_vec_exact(data.size)?;
        // SAFETY: `data.size > 0`, `data.data` is non-null, `data.size <= isize::MAX`, and valid
        // `avifRWData` instances uphold that `data.data` points to `data.size` initialized bytes.
        let slice = unsafe { std::slice::from_raw_parts(data.data, data.size) };
        vec.try_extend_from_slice(slice)?;
        Ok(vec)
    }
}

/// # Safety
/// - If `raw` is non-null, it must be properly aligned and point to a valid, initialized
///   `avifRWData` valid for exclusive reads and writes during this call.
/// - Within `*raw`, `raw.data` must either be null (with `raw.size == 0`) or a pointer previously
///   allocated by `crabby_avifRWDataRealloc` / `crabby_avifRWDataSet` with length `raw.size` that
///   has not yet been freed.
/// - Passing null for `raw` is safe and returns `avifResult::InvalidArgument`.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifRWDataRealloc(
    raw: *mut avifRWData,
    newSize: usize,
) -> avifResult {
    check_pointer!(raw);
    let raw = deref_mut!(raw);
    if raw.size == newSize {
        return avifResult::Ok;
    }
    if newSize == 0 {
        // SAFETY: Pre-conditions are met to call this function.
        unsafe {
            crabby_avifRWDataFree(raw);
        }
        return avifResult::Ok;
    }
    // Ok to use size as capacity here since we use reserve_exact.
    let mut newData: Vec<u8> = Vec::new();
    if newData.try_reserve_exact(newSize).is_err() {
        return avifResult::OutOfMemory;
    }
    if !raw.data.is_null() {
        let old_slice = std::ptr::slice_from_raw_parts_mut(raw.data, raw.size);
        // SAFETY: By caller precondition, a non-null `raw.data` was allocated as a `Box<[u8]>` of
        // length `raw.size` by `crabby_avifRWDataRealloc` / `crabby_avifRWDataSet` and has not yet
        // been freed.
        let oldData = unsafe { Box::from_raw(old_slice) };
        let sizeToCopy = std::cmp::min(newSize, oldData.len());
        if let Some(prefix) = oldData.get(..sizeToCopy) {
            #[allow(clippy::disallowed_methods)]
            newData.extend_from_slice(prefix);
        }
    }
    #[allow(clippy::disallowed_methods)]
    newData.resize(newSize, 0);
    raw.data = Box::into_raw(newData.into_boxed_slice()) as *mut u8;
    raw.size = newSize;
    avifResult::Ok
}

/// # Safety
/// - If `raw` is non-null, it must be properly aligned and point to a valid, initialized
///   `avifRWData` whose non-null `raw.data` was allocated by `crabby_avifRWDataRealloc` /
///   `crabby_avifRWDataSet` with length `raw.size`.
/// - When `size > 0`, `data` (if non-null) must point to at least `size` initialized bytes valid
///   for reads during this call (`size <= isize::MAX`, and may alias `raw.data`).
/// - When `size == 0`, `data` is not dereferenced (may be null) and any existing buffer in `raw`
///   is freed. Passing null for `raw`, or null for `data` when `size > 0`, returns
///   `avifResult::InvalidArgument`.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifRWDataSet(
    raw: *mut avifRWData,
    data: *const u8,
    size: usize,
) -> avifResult {
    check_pointer!(raw);
    let mut new_data: Vec<u8> = Vec::new();
    if size != 0 {
        if !check_slice_from_raw_parts_safety(data, size) {
            return avifResult::InvalidArgument;
        }
        // Copy the data into a new buffer directly instead of calling crabby_avifRWDataRealloc(),
        // which would zero the new buffer before it is overwritten.
        if new_data.try_reserve_exact(size).is_err() {
            return avifResult::OutOfMemory;
        }
        // SAFETY: `data` is non-null, `size <= isize::MAX`, and the caller guarantees `data`
        // points to at least `size` initialized bytes valid for shared reads during this call.
        let src = unsafe { std::slice::from_raw_parts(data, size) };
        #[allow(clippy::disallowed_methods)]
        new_data.extend_from_slice(src);
    }
    // Free the old buffer only after the copy, in case `data` points into it.
    // SAFETY: `raw` is non-null, and the caller guarantees it points to a valid `avifRWData` whose
    // non-null `data` was allocated by `crabby_avifRWDataRealloc` / `crabby_avifRWDataSet`.
    unsafe {
        crabby_avifRWDataFree(raw);
    }
    if size != 0 {
        let raw = deref_mut!(raw);
        raw.data = Box::into_raw(new_data.into_boxed_slice()) as *mut u8;
        raw.size = size;
    }
    avifResult::Ok
}

/// # Safety
/// - If `raw` is non-null, it must be properly aligned and point to a valid, initialized
///   `avifRWData` whose non-null `raw.data` was allocated by `crabby_avifRWDataRealloc` /
///   `crabby_avifRWDataSet` with length `raw.size` and has not yet been freed.
/// - Passing null for `raw` (or an `avifRWData` with null `data`) is safe and is a no-op.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifRWDataFree(raw: *mut avifRWData) {
    check_pointer_or_return!(raw);
    let raw = deref_mut!(raw);
    if raw.data.is_null() {
        raw.size = 0;
        return;
    }
    let slice_ptr = std::ptr::slice_from_raw_parts_mut(raw.data, raw.size);
    raw.data = std::ptr::null_mut();
    raw.size = 0;
    // SAFETY: By caller precondition, `slice_ptr` was allocated as a `Box<[u8]>` of length
    // `raw.size` by `crabby_avifRWDataRealloc` / `crabby_avifRWDataSet`, and resetting `raw.data`
    // to null above prevents double-free on subsequent calls.
    let _ = unsafe { Box::from_raw(slice_ptr) };
}

pub type avifIODestroyFunc = Option<unsafe extern "C" fn(io: *mut avifIO)>;
pub type avifIOReadFunc = unsafe extern "C" fn(
    io: *mut avifIO,
    readFlags: u32,
    offset: u64,
    size: usize,
    out: *mut avifROData,
) -> avifResult;
pub type avifIOWriteFunc = unsafe extern "C" fn(
    io: *mut avifIO,
    writeFlags: u32,
    offset: u64,
    data: *const u8,
    size: usize,
) -> avifResult;

#[repr(C)]
pub struct avifIO {
    destroy: avifIODestroyFunc,
    read: avifIOReadFunc,
    write: avifIOWriteFunc,
    sizeHint: u64,
    persistent: avifBool,
    data: *mut c_void,
}

/// Rust `decoder::IO` adapter around a C `avifIO` pointer.
///
/// Invariant: `io` is either null or a valid pointer to an initialized `avifIO` (whose `read` and
/// optional `destroy` callbacks uphold the C `avifIO` contract) that outlives this wrapper.
pub(crate) struct avifIOWrapper {
    data: avifROData,
    io: *mut avifIO,
}

impl avifIOWrapper {
    pub(crate) fn create(io: *mut avifIO) -> Self {
        Self {
            io,
            data: Default::default(),
        }
    }
}

impl Drop for avifIOWrapper {
    fn drop(&mut self) {
        // SAFETY: By the struct invariant and `crabby_avifDecoderSetIO`'s preconditions,
        // `self.io` is either null or points to a valid `avifIO` that has not yet been destroyed,
        // and it is destroyed only once, when the decoder releases or replaces its IO.
        unsafe {
            crabby_avifIODestroy(self.io);
        }
    }
}

impl crate::decoder::IO for avifIOWrapper {
    #[cfg_attr(feature = "disable_cfi", sanitize(cfi = "off"))]
    fn read(&mut self, offset: u64, size: usize) -> AvifResult<&[u8]> {
        if self.io.is_null() {
            return AvifError::io_not_set();
        }
        let read_fn = deref_const!(self.io).read;
        let out = &mut self.data as *mut avifROData;
        // SAFETY: `self.io` is non-null and points to a valid `avifIO` per
        // `crabby_avifDecoderSetIO`'s preconditions, and `out` points to `self.data` which is a
        // valid writable `avifROData` for the duration of this call.
        let res = unsafe { read_fn(self.io, 0, offset, size, out) };
        if res != avifResult::Ok {
            let err: AvifError = res.into();
            return Err(err);
        }
        if self.data.size == 0 {
            Ok(&[])
        } else if !check_slice_from_raw_parts_safety(self.data.data, self.data.size) {
            AvifError::unknown_error("invalid data pointer or size returned by avifIO read")
        } else {
            // SAFETY: `self.data.data` is non-null, `self.data.size <= isize::MAX`, and the C
            // `avifIO::read` contract guarantees the returned buffer remains valid and unmodified
            // until the next `read` or `destroy` call on `self.io` (both of which require `&mut
            // self`, tying the returned slice lifetime `&[u8]` to `&mut self`).
            Ok(unsafe { std::slice::from_raw_parts(self.data.data, self.data.size) })
        }
    }
    fn size_hint(&self) -> u64 {
        if self.io.is_null() {
            return 0;
        }
        deref_const!(self.io).sizeHint
    }
    fn persistent(&self) -> bool {
        if self.io.is_null() {
            return false;
        }
        deref_const!(self.io).persistent != 0
    }
}

pub(crate) struct avifCIOWrapper {
    io: GenericIO,
    buf: Vec<u8>,
}

/// # Safety
/// If `io` is non-null, it must be a pointer previously returned by
/// `crabby_avifIOCreateMemoryReader` or `crabby_avifIOCreateFileReader` that has not yet been
/// destroyed, and must not be accessed after this call. Passing null is safe and is a no-op.
#[no_mangle]
unsafe extern "C" fn cioDestroy(io: *mut avifIO) {
    check_pointer_or_return!(io);
    // SAFETY: `io` is non-null and, by caller precondition, was allocated via `Box::into_raw` in
    // `crabby_avifIOCreateMemoryReader` or `crabby_avifIOCreateFileReader` and has not yet been
    // freed.
    let io_box = unsafe { Box::from_raw(io) };
    let data = io_box.data as *mut avifCIOWrapper;
    if !data.is_null() {
        // SAFETY: When non-null, `io_box.data` was allocated via `Box::into_raw` on a
        // `Box<avifCIOWrapper>` when `io` was created and has not yet been freed.
        let _ = unsafe { Box::from_raw(data) };
    }
}

/// # Safety
/// - If `io` is non-null, it must point to a valid, initialized `avifIO` whose non-null `data`
///   field points to a valid `avifCIOWrapper` for exclusive access during this call.
/// - If `out` is non-null, it must point to a valid, initialized `avifROData` writable during this
///   call and non-aliasing with `io` and `io.data`.
/// - Passing null for `io` or `out` (or an `avifIO` with null `data`) safely returns
///   `avifResult::IoError`.
#[no_mangle]
unsafe extern "C" fn cioRead(
    io: *mut avifIO,
    _readFlags: u32,
    offset: u64,
    size: usize,
    out: *mut avifROData,
) -> avifResult {
    if io.is_null() || out.is_null() {
        return avifResult::IoError;
    }
    let io = deref_mut!(io);
    if io.data.is_null() {
        return avifResult::IoError;
    }
    let cio = deref_mut!(io.data as *mut avifCIOWrapper);
    match cio.io.read(offset, size) {
        Ok(data) => {
            cio.buf.clear();
            if cio.buf.try_extend_from_slice(data).is_err() {
                return avifResult::OutOfMemory;
            }
        }
        Err(_) => return avifResult::IoError,
    }
    let out = deref_mut!(out);
    out.data = if cio.buf.is_empty() { std::ptr::null() } else { cio.buf.as_ptr() };
    out.size = cio.buf.len();
    avifResult::Ok
}

#[no_mangle]
extern "C" fn cioWrite(
    _io: *mut avifIO,
    _writeFlags: u32,
    _offset: u64,
    _data: *const u8,
    _size: usize,
) -> avifResult {
    avifResult::Ok
}

/// # Safety
/// - If `data` is non-null and `size <= isize::MAX as usize`, `data` must point to at least `size`
///   initialized bytes that remain valid and unmodified for shared reads until the returned
///   `avifIO` is destroyed (with `crabby_avifIODestroy` or by the decoder that owns it).
/// - Passing null for `data` or `size > isize::MAX as usize` safely returns null.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifIOCreateMemoryReader(
    data: *const u8,
    size: usize,
) -> *mut avifIO {
    if !check_slice_from_raw_parts_safety(data, size) {
        return std::ptr::null_mut();
    }
    let cio = Box::new(avifCIOWrapper {
        // SAFETY: `data` is non-null, `size <= isize::MAX`, and the caller guarantees `data`
        // points to `size` initialized bytes that outlive the returned `avifIO` reader.
        io: Box::new(unsafe { DecoderRawIO::create(data, size) }),
        buf: Vec::new(),
    });
    let io = Box::new(avifIO {
        destroy: Some(cioDestroy),
        read: cioRead,
        write: cioWrite,
        sizeHint: size as u64,
        persistent: 0,
        data: Box::into_raw(cio) as *mut c_void,
    });
    Box::into_raw(io)
}

/// # Safety
/// If `filename` is non-null, it must point to a valid, nul-terminated C string valid for reads
/// during this call. Passing null (or invalid UTF-8 / a file that cannot be opened) safely returns
/// null.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifIOCreateFileReader(filename: *const c_char) -> *mut avifIO {
    if filename.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: `filename` is non-null (checked above), and the caller guarantees it points to a
    // valid, nul-terminated C string valid for reads during this call.
    let Ok(filename) = (unsafe { CStr::from_ptr(filename) }).to_str() else {
        return std::ptr::null_mut();
    };
    let filename = String::from(filename);
    let file_io = match DecoderFileIO::create(&filename) {
        Ok(x) => x,
        Err(_) => return std::ptr::null_mut(),
    };
    let cio = Box::new(avifCIOWrapper {
        io: Box::new(file_io),
        buf: Vec::new(),
    });
    let io = Box::new(avifIO {
        destroy: Some(cioDestroy),
        read: cioRead,
        write: cioWrite,
        sizeHint: cio.io.size_hint(),
        persistent: 0,
        data: Box::into_raw(cio) as *mut c_void,
    });
    Box::into_raw(io)
}

/// # Safety
/// If `io` is non-null, it must point to a valid, initialized `avifIO` that has not yet been
/// destroyed and whose `destroy` callback (if any) upholds the C `avifIO` contract. `io` must not
/// be accessed after this call if it has a `destroy` callback. Passing null (or an `avifIO` with a
/// null `destroy` callback) is safe and is a no-op.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifIODestroy(io: *mut avifIO) {
    check_pointer_or_return!(io);
    if let Some(destroy) = deref_const!(io).destroy {
        // SAFETY: `io` is non-null and, by caller precondition, points to a valid `avifIO` that
        // has not yet been destroyed, so its `destroy` callback may be invoked once.
        unsafe {
            destroy(io);
        }
    }
}
