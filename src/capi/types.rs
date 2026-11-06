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

use std::ffi::CStr;
use std::ffi::CString;
use std::os::raw::c_char;
use std::os::raw::c_int;
use std::os::raw::c_uint;
use std::os::raw::c_void;

use crate::utils::clap::*;
use crate::*;

#[repr(C)]
#[derive(PartialEq)]
pub enum avifResult {
    Ok = 0,
    UnknownError = 1,
    InvalidFtyp = 2,
    NoContent = 3,
    NoYuvFormatSelected = 4,
    ReformatFailed = 5,
    UnsupportedDepth = 6,
    EncodeColorFailed = 7,
    EncodeAlphaFailed = 8,
    BmffParseFailed = 9,
    MissingImageItem = 10,
    DecodeColorFailed = 11,
    DecodeAlphaFailed = 12,
    ColorAlphaSizeMismatch = 13,
    IspeSizeMismatch = 14,
    NoCodecAvailable = 15,
    NoImagesRemaining = 16,
    InvalidExifPayload = 17,
    InvalidImageGrid = 18,
    InvalidCodecSpecificOption = 19,
    TruncatedData = 20,
    IoNotSet = 21,
    IoError = 22,
    WaitingOnIo = 23,
    InvalidArgument = 24,
    NotImplemented = 25,
    OutOfMemory = 26,
    CannotChangeSetting = 27,
    IncompatibleImage = 28,
    InternalError = 29,
    EncodeGainMapFailed = 30,
    DecodeGainMapFailed = 31,
    InvalidToneMappedImage = 32,
}

impl From<&AvifError> for avifResult {
    fn from(err: &AvifError) -> Self {
        match err {
            AvifError::Ok => avifResult::Ok,
            AvifError::UnknownError(_) => avifResult::UnknownError,
            AvifError::InvalidFtyp => avifResult::InvalidFtyp,
            AvifError::NoContent => avifResult::NoContent,
            AvifError::NoYuvFormatSelected => avifResult::NoYuvFormatSelected,
            AvifError::ReformatFailed => avifResult::ReformatFailed,
            AvifError::UnsupportedDepth => avifResult::UnsupportedDepth,
            AvifError::EncodeColorFailed => avifResult::EncodeColorFailed,
            AvifError::EncodeAlphaFailed => avifResult::EncodeAlphaFailed,
            AvifError::BmffParseFailed(_) => avifResult::BmffParseFailed,
            AvifError::MissingImageItem => avifResult::MissingImageItem,
            AvifError::DecodeColorFailed => avifResult::DecodeColorFailed,
            AvifError::DecodeAlphaFailed => avifResult::DecodeAlphaFailed,
            AvifError::ColorAlphaSizeMismatch => avifResult::ColorAlphaSizeMismatch,
            AvifError::IspeSizeMismatch => avifResult::IspeSizeMismatch,
            AvifError::NoCodecAvailable => avifResult::NoCodecAvailable,
            AvifError::NoImagesRemaining => avifResult::NoImagesRemaining,
            AvifError::InvalidExifPayload => avifResult::InvalidExifPayload,
            AvifError::InvalidImageGrid(_) => avifResult::InvalidImageGrid,
            AvifError::InvalidCodecSpecificOption => avifResult::InvalidCodecSpecificOption,
            AvifError::TruncatedData => avifResult::TruncatedData,
            AvifError::IoNotSet => avifResult::IoNotSet,
            AvifError::IoError => avifResult::IoError,
            AvifError::WaitingOnIo => avifResult::WaitingOnIo,
            AvifError::InvalidArgument => avifResult::InvalidArgument,
            AvifError::NotImplemented => avifResult::NotImplemented,
            AvifError::OutOfMemory => avifResult::OutOfMemory,
            AvifError::CannotChangeSetting => avifResult::CannotChangeSetting,
            AvifError::IncompatibleImage => avifResult::IncompatibleImage,
            AvifError::InternalError => avifResult::InternalError,
            AvifError::EncodeGainMapFailed => avifResult::EncodeGainMapFailed,
            AvifError::DecodeGainMapFailed => avifResult::DecodeGainMapFailed,
            AvifError::InvalidToneMappedImage(_) => avifResult::InvalidToneMappedImage,
        }
    }
}

impl<T> From<AvifResult<T>> for avifResult {
    fn from(res: AvifResult<T>) -> Self {
        match res {
            Ok(_) => avifResult::Ok,
            Err(err) => {
                let res: avifResult = (&err).into();
                res
            }
        }
    }
}

impl From<avifResult> for AvifError {
    fn from(res: avifResult) -> Self {
        match res {
            avifResult::Ok => AvifError::Ok,
            avifResult::UnknownError => AvifError::UnknownError("".into()),
            avifResult::InvalidFtyp => AvifError::InvalidFtyp,
            avifResult::NoContent => AvifError::NoContent,
            avifResult::NoYuvFormatSelected => AvifError::NoYuvFormatSelected,
            avifResult::ReformatFailed => AvifError::ReformatFailed,
            avifResult::UnsupportedDepth => AvifError::UnsupportedDepth,
            avifResult::EncodeColorFailed => AvifError::EncodeColorFailed,
            avifResult::EncodeAlphaFailed => AvifError::EncodeAlphaFailed,
            avifResult::BmffParseFailed => AvifError::BmffParseFailed("".into()),
            avifResult::MissingImageItem => AvifError::MissingImageItem,
            avifResult::DecodeColorFailed => AvifError::DecodeColorFailed,
            avifResult::DecodeAlphaFailed => AvifError::DecodeAlphaFailed,
            avifResult::ColorAlphaSizeMismatch => AvifError::ColorAlphaSizeMismatch,
            avifResult::IspeSizeMismatch => AvifError::IspeSizeMismatch,
            avifResult::NoCodecAvailable => AvifError::NoCodecAvailable,
            avifResult::NoImagesRemaining => AvifError::NoImagesRemaining,
            avifResult::InvalidExifPayload => AvifError::InvalidExifPayload,
            avifResult::InvalidImageGrid => AvifError::InvalidImageGrid("".into()),
            avifResult::InvalidCodecSpecificOption => AvifError::InvalidCodecSpecificOption,
            avifResult::TruncatedData => AvifError::TruncatedData,
            avifResult::IoNotSet => AvifError::IoNotSet,
            avifResult::IoError => AvifError::IoError,
            avifResult::WaitingOnIo => AvifError::WaitingOnIo,
            avifResult::InvalidArgument => AvifError::InvalidArgument,
            avifResult::NotImplemented => AvifError::NotImplemented,
            avifResult::OutOfMemory => AvifError::OutOfMemory,
            avifResult::CannotChangeSetting => AvifError::CannotChangeSetting,
            avifResult::IncompatibleImage => AvifError::IncompatibleImage,
            avifResult::InternalError => AvifError::InternalError,
            avifResult::EncodeGainMapFailed => AvifError::EncodeGainMapFailed,
            avifResult::DecodeGainMapFailed => AvifError::DecodeGainMapFailed,
            avifResult::InvalidToneMappedImage => AvifError::InvalidToneMappedImage("".into()),
        }
    }
}

pub type avifBool = c_int;
pub const AVIF_TRUE: c_int = 1;
pub const AVIF_FALSE: c_int = 0;

pub const AVIF_STRICT_DISABLED: u32 = 0;
pub const AVIF_STRICT_PIXI_REQUIRED: u32 = 1 << 0;
pub const AVIF_STRICT_CLAP_VALID: u32 = 1 << 1;
pub const AVIF_STRICT_ALPHA_ISPE_REQUIRED: u32 = 1 << 2;
pub const AVIF_STRICT_EXIF_VALID: u32 = 1 << 3;
pub const AVIF_STRICT_MULTIPLE_ILOC_ENTRIES_FOR_SAME_ITEM_DISALLOWED: u32 = 1 << 4;
pub const AVIF_STRICT_HEVC_BRAND_REQUIRES_MOOV: u32 = 1 << 5;
pub const AVIF_STRICT_ESSENTIAL_TRANSFORMATIVE_PROPERTIES: u32 = 1 << 6;
pub const AVIF_STRICT_ENABLED: u32 = AVIF_STRICT_PIXI_REQUIRED
    | AVIF_STRICT_CLAP_VALID
    | AVIF_STRICT_ALPHA_ISPE_REQUIRED
    | AVIF_STRICT_EXIF_VALID
    | AVIF_STRICT_MULTIPLE_ILOC_ENTRIES_FOR_SAME_ITEM_DISALLOWED
    | AVIF_STRICT_HEVC_BRAND_REQUIRES_MOOV
    | AVIF_STRICT_ESSENTIAL_TRANSFORMATIVE_PROPERTIES;
pub type avifStrictFlags = u32;

pub const AVIF_IMAGE_CONTENT_NONE: u32 = 0;
pub const AVIF_IMAGE_CONTENT_COLOR: u32 = 1 << 0;
/// Alpha only is not currently supported.
pub const AVIF_IMAGE_CONTENT_ALPHA: u32 = 1 << 1;
pub const AVIF_IMAGE_CONTENT_COLOR_AND_ALPHA: u32 =
    AVIF_IMAGE_CONTENT_COLOR | AVIF_IMAGE_CONTENT_ALPHA;
pub const AVIF_IMAGE_CONTENT_GAIN_MAP: u32 = 1 << 2;
pub const AVIF_IMAGE_CONTENT_COLOR_AND_GAIN_MAP: u32 =
    AVIF_IMAGE_CONTENT_COLOR | AVIF_IMAGE_CONTENT_GAIN_MAP;
pub const AVIF_IMAGE_CONTENT_ALL: u32 =
    AVIF_IMAGE_CONTENT_COLOR_AND_ALPHA | AVIF_IMAGE_CONTENT_GAIN_MAP;
pub type avifImageContentTypeFlags = u32;

#[repr(C)]
pub struct avifDecoderData {}

pub const AVIF_DIAGNOSTICS_ERROR_BUFFER_SIZE: usize = 256;
#[repr(C)]
pub struct avifDiagnostics {
    error: [c_char; AVIF_DIAGNOSTICS_ERROR_BUFFER_SIZE],
}

impl Default for avifDiagnostics {
    fn default() -> Self {
        Self {
            error: [0; AVIF_DIAGNOSTICS_ERROR_BUFFER_SIZE],
        }
    }
}

impl avifDiagnostics {
    pub(crate) fn set_from_result<T>(&mut self, res: &AvifResult<T>) {
        match res {
            Ok(_) => self.set_error_empty(),
            Err(AvifError::BmffParseFailed(s))
            | Err(AvifError::UnknownError(s))
            | Err(AvifError::InvalidImageGrid(s))
            | Err(AvifError::InvalidToneMappedImage(s)) => self.set_error_string(s),
            _ => self.set_error_empty(),
        }
    }

    fn set_error_string(&mut self, error: &str) {
        self.error.fill(0);
        let bytes = error.as_bytes();
        if bytes.contains(&0) {
            return;
        }
        let max_len = AVIF_DIAGNOSTICS_ERROR_BUFFER_SIZE - 1;
        for (dst, &src) in self.error.iter_mut().take(max_len).zip(bytes.iter()) {
            *dst = src as c_char;
        }
    }

    pub(crate) fn set_error_empty(&mut self) {
        self.error.fill(0);
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub enum avifCodecChoice {
    #[default]
    Auto = 0,
    Aom = 1,
    Dav1d = 2,
    Libgav1 = 3,
    Rav1e = 4,
    Svt = 5,
    Avm = 6,
    // Pure Rust AV1 keyframe-only decoder. Only available when built with the native_av1
    // feature. WARNING: experimental.
    Native = 7,
}

impl avifCodecChoice {
    fn from_name(name: &str) -> Self {
        let available_codecs: &[(avifCodecChoice, &str)] = &[
            #[cfg(feature = "aom")]
            (Self::Aom, "aom"),
            #[cfg(feature = "dav1d")]
            (Self::Dav1d, "dav1d"),
            #[cfg(feature = "native_av1")]
            (Self::Native, "native"),
        ];
        for available_codec in available_codecs {
            if name == available_codec.1 {
                return available_codec.0;
            }
        }
        avifCodecChoice::Auto
    }
}

/// # Safety
/// If `name` is non-null, it must point to a valid, nul-terminated C string that remains valid for
/// reads for the duration of this call. Passing null is safe and returns `avifCodecChoice::Auto`.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifCodecChoiceFromName(name: *const c_char) -> avifCodecChoice {
    if name.is_null() {
        return avifCodecChoice::Auto;
    }
    // SAFETY: `name` is non-null (checked above), and the caller guarantees it points to a valid,
    // nul-terminated C string valid for reads during this call.
    let Ok(name) = (unsafe { CStr::from_ptr(name) }).to_str() else {
        return avifCodecChoice::Auto;
    };
    avifCodecChoice::from_name(name)
}

macro_rules! cstr_ptr {
    ($bytes:literal) => {{
        const CSTR: &CStr = match CStr::from_bytes_with_nul($bytes) {
            Ok(s) => s,
            Err(_) => loop {},
        };
        CSTR.as_ptr()
    }};
}

#[no_mangle]
pub extern "C" fn crabby_avifCodecName(
    _choice: avifCodecChoice,
    requiredFlags: avifCodecFlags,
) -> *const c_char {
    // This function will just return "dav1d" or "aom" based on whether encoder or decoder is being
    // queried. It simply exists for compatibility with libavif.
    if (requiredFlags & avifCodecFlag::CanEncode as u32) != 0 {
        cstr_ptr!(b"aom\0")
    } else {
        cstr_ptr!(b"dav1d\0")
    }
}

#[no_mangle]
pub extern "C" fn crabby_avifPixelFormatToString(format: PixelFormat) -> *const c_char {
    match format {
        PixelFormat::Yuv444 => cstr_ptr!(b"YUV444\0"),
        PixelFormat::Yuv422 => cstr_ptr!(b"YUV422\0"),
        PixelFormat::Yuv420 => cstr_ptr!(b"YUV420\0"),
        PixelFormat::Yuv400 => cstr_ptr!(b"YUV400\0"),
        PixelFormat::AndroidP010 => cstr_ptr!(b"P010\0"),
        PixelFormat::AndroidNv12 => cstr_ptr!(b"NV12\0"),
        PixelFormat::AndroidNv21 => cstr_ptr!(b"NV21\0"),
        _ => cstr_ptr!(b"Unknown\0"),
    }
}

pub(crate) fn to_avifBool(val: bool) -> avifBool {
    if val {
        AVIF_TRUE
    } else {
        AVIF_FALSE
    }
}

#[no_mangle]
pub extern "C" fn crabby_avifResultToString(res: avifResult) -> *const c_char {
    match res {
        avifResult::Ok => cstr_ptr!(b"Ok\0"),
        avifResult::UnknownError => cstr_ptr!(b"Unknown Error\0"),
        avifResult::InvalidFtyp => cstr_ptr!(b"Invalid ftyp\0"),
        avifResult::NoContent => cstr_ptr!(b"No content\0"),
        avifResult::NoYuvFormatSelected => cstr_ptr!(b"No YUV format selected\0"),
        avifResult::ReformatFailed => cstr_ptr!(b"Reformat failed\0"),
        avifResult::UnsupportedDepth => cstr_ptr!(b"Unsupported depth\0"),
        avifResult::EncodeColorFailed => cstr_ptr!(b"Encoding of color planes failed\0"),
        avifResult::EncodeAlphaFailed => cstr_ptr!(b"Encoding of alpha plane failed\0"),
        avifResult::BmffParseFailed => cstr_ptr!(b"BMFF parsing failed\0"),
        avifResult::MissingImageItem => cstr_ptr!(b"Missing or empty image item\0"),
        avifResult::DecodeColorFailed => cstr_ptr!(b"Decoding of color planes failed\0"),
        avifResult::DecodeAlphaFailed => cstr_ptr!(b"Decoding of alpha plane failed\0"),
        avifResult::ColorAlphaSizeMismatch => cstr_ptr!(b"Color and alpha planes size mismatch\0"),
        avifResult::IspeSizeMismatch => cstr_ptr!(b"Plane sizes don't match ispe values\0"),
        avifResult::NoCodecAvailable => cstr_ptr!(b"No codec available\0"),
        avifResult::NoImagesRemaining => cstr_ptr!(b"No images remaining\0"),
        avifResult::InvalidExifPayload => cstr_ptr!(b"Invalid Exif payload\0"),
        avifResult::InvalidImageGrid => cstr_ptr!(b"Invalid image grid\0"),
        avifResult::InvalidCodecSpecificOption => cstr_ptr!(b"Invalid codec-specific option\0"),
        avifResult::TruncatedData => cstr_ptr!(b"Truncated data\0"),
        avifResult::IoNotSet => cstr_ptr!(b"IO not set\0"),
        avifResult::IoError => cstr_ptr!(b"IO Error\0"),
        avifResult::WaitingOnIo => cstr_ptr!(b"Waiting on IO\0"),
        avifResult::InvalidArgument => cstr_ptr!(b"Invalid argument\0"),
        avifResult::NotImplemented => cstr_ptr!(b"Not implemented\0"),
        avifResult::OutOfMemory => cstr_ptr!(b"Out of memory\0"),
        avifResult::CannotChangeSetting => {
            cstr_ptr!(b"Cannot change some setting during encoding\0")
        }
        avifResult::IncompatibleImage => {
            cstr_ptr!(b"The image is incompatible with already encoded images\0")
        }
        avifResult::InternalError => cstr_ptr!(b"Internal error\0"),
        avifResult::EncodeGainMapFailed => cstr_ptr!(b"Encoding of gain map planes failed\0"),
        avifResult::DecodeGainMapFailed => cstr_ptr!(b"Decoding of gain map planes failed\0"),
        avifResult::InvalidToneMappedImage => cstr_ptr!(b"Invalid tone mapped image item\0"),
    }
}

pub type avifCropRect = CropRect;

/// # Safety
/// - If both `cropRect` and `clap` are non-null:
///   - `clap` must be properly aligned and point to an initialized `avifCleanApertureBox` valid
///     for shared reads during this call.
///   - `cropRect` must be properly aligned and point to an initialized `avifCropRect` valid for
///     exclusive writes during this call, without aliasing `clap`.
/// - Passing null for either `cropRect` or `clap` is safe and returns `AVIF_FALSE`. `_diag` is
///   unused and may be null.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifCropRectConvertCleanApertureBox(
    cropRect: *mut avifCropRect,
    clap: *const avifCleanApertureBox,
    imageW: u32,
    imageH: u32,
    yuvFormat: PixelFormat,
    _diag: *mut avifDiagnostics,
) -> avifBool {
    if cropRect.is_null() || clap.is_null() {
        return AVIF_FALSE;
    }
    let rust_clap: CleanAperture = deref_const!(clap).into();
    let Ok(rect) = CropRect::create_from(&rust_clap, imageW, imageH, yuvFormat) else {
        return AVIF_FALSE;
    };
    *deref_mut!(cropRect) = rect;
    AVIF_TRUE
}

/// # Safety
/// - If both `clap` and `cropRect` are non-null:
///   - `cropRect` must be properly aligned and point to an initialized `avifCropRect` valid for
///     shared reads during this call.
///   - `clap` must be properly aligned and point to an initialized `avifCleanApertureBox` valid
///     for exclusive writes during this call, without aliasing `cropRect`.
/// - Passing null for either `clap` or `cropRect` is safe and returns `AVIF_FALSE`. `_diag` is
///   unused and may be null.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifCleanApertureBoxConvertCropRect(
    clap: *mut avifCleanApertureBox,
    cropRect: *const avifCropRect,
    imageW: u32,
    imageH: u32,
    yuvFormat: PixelFormat,
    _diag: *mut avifDiagnostics,
) -> avifBool {
    if cropRect.is_null() || clap.is_null() {
        return AVIF_FALSE;
    }
    let Ok(x) = CleanAperture::create_from(deref_const!(cropRect), imageW, imageH, yuvFormat)
    else {
        return AVIF_FALSE;
    };
    *deref_mut!(clap) = (&Some(x)).into();
    AVIF_TRUE
}

// Constants and definitions from libavif that are not used in rust.

pub const AVIF_PLANE_COUNT_YUV: usize = 3;
pub const AVIF_REPETITION_COUNT_INFINITE: i32 = -1;
pub const AVIF_REPETITION_COUNT_UNKNOWN: i32 = -2;

/// cbindgen:rename-all=ScreamingSnakeCase
#[repr(C)]
pub enum avifPlanesFlag {
    AvifPlanesYuv = 1 << 0,
    AvifPlanesA = 1 << 1,
    AvifPlanesAll = 0xFF,
}
pub type avifPlanesFlags = u32;

/// cbindgen:rename-all=ScreamingSnakeCase
#[repr(C)]
pub enum avifChannelIndex {
    AvifChanY = 0,
    AvifChanU = 1,
    AvifChanV = 2,
    AvifChanA = 3,
}

#[repr(C)]
pub struct avifPixelFormatInfo {
    monochrome: avifBool,
    chromaShiftX: c_int,
    chromaShiftY: c_int,
}

/// # Safety
/// If `info` is non-null, it must be properly aligned and point to an initialized
/// `avifPixelFormatInfo` valid for exclusive writes during this call. Passing null is safe and is
/// a no-op.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifGetPixelFormatInfo(
    format: PixelFormat,
    info: *mut avifPixelFormatInfo,
) {
    check_pointer_or_return!(info);
    let info = deref_mut!(info);
    match format {
        PixelFormat::Yuv444 => {
            info.chromaShiftX = 0;
            info.chromaShiftY = 0;
            info.monochrome = AVIF_FALSE;
        }
        PixelFormat::Yuv422 => {
            info.chromaShiftX = 1;
            info.chromaShiftY = 0;
            info.monochrome = AVIF_FALSE;
        }
        PixelFormat::Yuv420 => {
            info.chromaShiftX = 1;
            info.chromaShiftY = 1;
            info.monochrome = AVIF_FALSE;
        }
        PixelFormat::Yuv400 => {
            info.chromaShiftX = 1;
            info.chromaShiftY = 1;
            info.monochrome = AVIF_TRUE;
        }
        _ => {}
    }
}

/// # Safety
/// If `diag` is non-null, it must be properly aligned and point to an initialized
/// `avifDiagnostics` valid for exclusive writes during this call. Passing null is safe and is a
/// no-op.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDiagnosticsClearError(diag: *mut avifDiagnostics) {
    check_pointer_or_return!(diag);
    deref_mut!(diag).set_error_empty();
}

#[repr(C)]
pub enum avifCodecFlag {
    CanDecode = (1 << 0),
    CanEncode = (1 << 1),
}
pub type avifCodecFlags = u32;

pub const AVIF_TRANSFORM_NONE: u32 = 0;
pub const AVIF_TRANSFORM_PASP: u32 = 1 << 0;
pub const AVIF_TRANSFORM_CLAP: u32 = 1 << 1;
pub const AVIF_TRANSFORM_IROT: u32 = 1 << 2;
pub const AVIF_TRANSFORM_IMIR: u32 = 1 << 3;
pub type avifTransformFlags = u32;

pub const AVIF_COLOR_PRIMARIES_BT709: u16 = 1;
pub const AVIF_COLOR_PRIMARIES_IEC61966_2_4: u16 = 1;
pub const AVIF_COLOR_PRIMARIES_BT2100: u16 = 9;
pub const AVIF_COLOR_PRIMARIES_DCI_P3: u16 = 12;
pub const AVIF_TRANSFER_CHARACTERISTICS_SMPTE2084: u16 = 16;

/// # Safety
/// Allocates `size` zero-initialized bytes via the system allocator and returns a raw pointer that
/// must eventually be freed with `crabby_avifFree`. Returns null if `size == 0` or if allocation
/// fails. Safe for any `size` value; retained as `unsafe` because it performs raw FFI allocation
/// and existing C-API Rust callers invoke it inside `unsafe` blocks under `-Dwarnings`.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifAlloc(size: usize) -> *mut c_void {
    if size == 0 {
        return std::ptr::null_mut();
    }
    // SAFETY: `libc::calloc(1, size)` is safe to invoke for any non-zero `size` and returns either
    // a suitably aligned zeroed allocation of `size` bytes or null on allocation failure.
    unsafe { libc::calloc(1, size) }
}

/// # Safety
/// If `p` is non-null, it must be a pointer previously returned by `crabby_avifAlloc` (or standard
/// C `malloc`/`calloc`/`realloc`) that has not yet been freed, and must not be used after this
/// call. Passing null is safe and is a no-op.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifFree(p: *mut c_void) {
    if !p.is_null() {
        // SAFETY: `p` is non-null and, by caller precondition, was allocated by the C allocator
        // (`crabby_avifAlloc`) and has not yet been freed.
        unsafe { libc::free(p) }
    }
}

pub const AVIF_ADD_IMAGE_FLAG_NONE: u32 = 0;
pub const AVIF_ADD_IMAGE_FLAG_FORCE_KEYFRAME: u32 = 1 << 0;
pub const AVIF_ADD_IMAGE_FLAG_SINGLE: u32 = 1 << 1;
pub type avifAddImageFlags = u32;

pub const AVIF_QUALITY_WORST: u32 = 0;
pub const AVIF_QUALITY_BEST: u32 = 100;
pub const AVIF_QUALITY_LOSSLESS: u32 = 100;
pub const AVIF_QUALITY_DEFAULT: i32 = -1;

pub const AVIF_QUANTIZER_WORST_QUALITY: u32 = 63;
pub const AVIF_QUANTIZER_BEST_QUALITY: u32 = 0;
pub const AVIF_QUANTIZER_LOSSLESS: u32 = 0;

pub const AVIF_SPEED_SLOWEST: u32 = 0;
pub const AVIF_SPEED_FASTEST: u32 = 10;
pub const AVIF_SPEED_DEFAULT: u32 = 6;

/// # Safety
/// If `outBuffer` is non-null, it must point to a writable `c_char` allocation of at least 256
/// bytes (which may be uninitialized prior to this call) that does not overlap internal allocations
/// and remains valid for writes during this call. Passing null is safe and is a no-op.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifCodecVersions(outBuffer: *mut c_char) {
    if outBuffer.is_null() {
        return;
    }
    let versions = match CString::new(codec_versions()) {
        Ok(s) => s,
        Err(_) => return,
    };
    let bytes = versions.as_bytes_with_nul();
    if bytes.len() > 256 {
        return;
    }
    // SAFETY: `outBuffer` is non-null, `bytes.len() <= 256`, `bytes` is a freshly allocated local
    // `CString` so it cannot overlap `outBuffer`, and `copy_nonoverlapping` safely writes into
    // `outBuffer` even when the caller's C buffer is uninitialized (where constructing `&mut [u8]`
    // via `from_raw_parts_mut` would be UB).
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr().cast::<c_char>(), outBuffer, bytes.len());
    }
}

#[no_mangle]
pub extern "C" fn crabby_avifLibYUVVersion() -> c_uint {
    #[cfg(feature = "libyuv")]
    return libyuv_sys::bindings::LIBYUV_VERSION as _;
    #[cfg(not(feature = "libyuv"))]
    return 0;
}
