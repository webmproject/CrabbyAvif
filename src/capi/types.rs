use super::image::*;

use std::os::raw::c_char;
use std::os::raw::c_int;
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
    EncodeGainMapFailed = 29,
    DecodeGainMapFailed = 30,
    InvalidToneMappedImage = 31,
}

impl From<&AvifError> for avifResult {
    fn from(err: &AvifError) -> Self {
        match err {
            AvifError::Ok => avifResult::Ok,
            AvifError::UnknownError => avifResult::UnknownError,
            AvifError::InvalidFtyp => avifResult::InvalidFtyp,
            AvifError::NoContent => avifResult::NoContent,
            AvifError::NoYuvFormatSelected => avifResult::NoYuvFormatSelected,
            AvifError::ReformatFailed => avifResult::ReformatFailed,
            AvifError::UnsupportedDepth => avifResult::UnsupportedDepth,
            AvifError::EncodeColorFailed => avifResult::EncodeColorFailed,
            AvifError::EncodeAlphaFailed => avifResult::EncodeAlphaFailed,
            AvifError::BmffParseFailed => avifResult::BmffParseFailed,
            AvifError::MissingImageItem => avifResult::MissingImageItem,
            AvifError::DecodeColorFailed => avifResult::DecodeColorFailed,
            AvifError::DecodeAlphaFailed => avifResult::DecodeAlphaFailed,
            AvifError::ColorAlphaSizeMismatch => avifResult::ColorAlphaSizeMismatch,
            AvifError::IspeSizeMismatch => avifResult::IspeSizeMismatch,
            AvifError::NoCodecAvailable => avifResult::NoCodecAvailable,
            AvifError::NoImagesRemaining => avifResult::NoImagesRemaining,
            AvifError::InvalidExifPayload => avifResult::InvalidExifPayload,
            AvifError::InvalidImageGrid => avifResult::InvalidImageGrid,
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
            AvifError::EncodeGainMapFailed => avifResult::EncodeGainMapFailed,
            AvifError::DecodeGainMapFailed => avifResult::DecodeGainMapFailed,
            AvifError::InvalidToneMappedImage => avifResult::InvalidToneMappedImage,
        }
    }
}

impl From<avifResult> for AvifError {
    fn from(res: avifResult) -> Self {
        match res {
            avifResult::Ok => AvifError::Ok,
            avifResult::UnknownError => AvifError::UnknownError,
            avifResult::InvalidFtyp => AvifError::InvalidFtyp,
            avifResult::NoContent => AvifError::NoContent,
            avifResult::NoYuvFormatSelected => AvifError::NoYuvFormatSelected,
            avifResult::ReformatFailed => AvifError::ReformatFailed,
            avifResult::UnsupportedDepth => AvifError::UnsupportedDepth,
            avifResult::EncodeColorFailed => AvifError::EncodeColorFailed,
            avifResult::EncodeAlphaFailed => AvifError::EncodeAlphaFailed,
            avifResult::BmffParseFailed => AvifError::BmffParseFailed,
            avifResult::MissingImageItem => AvifError::MissingImageItem,
            avifResult::DecodeColorFailed => AvifError::DecodeColorFailed,
            avifResult::DecodeAlphaFailed => AvifError::DecodeAlphaFailed,
            avifResult::ColorAlphaSizeMismatch => AvifError::ColorAlphaSizeMismatch,
            avifResult::IspeSizeMismatch => AvifError::IspeSizeMismatch,
            avifResult::NoCodecAvailable => AvifError::NoCodecAvailable,
            avifResult::NoImagesRemaining => AvifError::NoImagesRemaining,
            avifResult::InvalidExifPayload => AvifError::InvalidExifPayload,
            avifResult::InvalidImageGrid => AvifError::InvalidImageGrid,
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
            avifResult::EncodeGainMapFailed => AvifError::EncodeGainMapFailed,
            avifResult::DecodeGainMapFailed => AvifError::DecodeGainMapFailed,
            avifResult::InvalidToneMappedImage => AvifError::InvalidToneMappedImage,
        }
    }
}

pub type avifBool = c_int;
pub const AVIF_TRUE: c_int = 1;
pub const AVIF_FALSE: c_int = 0;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum avifPixelFormat {
    None,
    Yuv444,
    Yuv422,
    Yuv420,
    Yuv400,
    Count,
}

impl From<PixelFormat> for avifPixelFormat {
    fn from(format: PixelFormat) -> Self {
        match format {
            PixelFormat::Yuv444 => Self::Yuv444,
            PixelFormat::Yuv422 => Self::Yuv422,
            PixelFormat::Yuv420 => Self::Yuv420,
            PixelFormat::Monochrome => Self::Yuv400,
        }
    }
}

impl From<avifPixelFormat> for PixelFormat {
    fn from(format: avifPixelFormat) -> Self {
        match format {
            avifPixelFormat::Yuv444 => Self::Yuv444,
            avifPixelFormat::Yuv422 => Self::Yuv422,
            avifPixelFormat::Yuv420 => Self::Yuv420,
            avifPixelFormat::Yuv400 => Self::Monochrome,
            _ => PixelFormat::Yuv420,
        }
    }
}

impl avifPixelFormat {
    // TODO: these functions can be removed if avifPixelFormat can be aliased to PixelFormat (with
    // constants None and Count.
    pub fn is_monochrome(&self) -> bool {
        matches!(self, Self::Yuv400)
    }

    pub fn chroma_shift_x(&self) -> u32 {
        match self {
            Self::Yuv422 | Self::Yuv420 => 1,
            _ => 0,
        }
    }

    pub fn chroma_shift_y(&self) -> u32 {
        match self {
            Self::Yuv420 => 1,
            _ => 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum avifRange {
    Limited = 0,
    Full = 1,
}

impl From<bool> for avifRange {
    fn from(full_range: bool) -> Self {
        match full_range {
            true => Self::Full,
            false => Self::Limited,
        }
    }
}

pub const AVIF_STRICT_DISABLED: u32 = 0;
pub const AVIF_STRICT_PIXI_REQUIRED: u32 = 1 << 0;
pub const AVIF_STRICT_CLAP_VALID: u32 = 1 << 1;
pub const AVIF_STRICT_ALPHA_ISPE_REQUIRED: u32 = 1 << 2;
pub const AVIF_STRICT_ENABLED: u32 =
    AVIF_STRICT_PIXI_REQUIRED | AVIF_STRICT_CLAP_VALID | AVIF_STRICT_ALPHA_ISPE_REQUIRED;
pub type avifStrictFlags = u32;

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

#[repr(C)]
pub enum avifCodecChoice {
    Auto = 0,
    Aom = 1,
    Dav1d = 2,
    Libgav1 = 3,
    Rav1e = 4,
    Svt = 5,
    Avm = 6,
}

pub fn to_avifBool(val: bool) -> avifBool {
    if val {
        AVIF_TRUE
    } else {
        AVIF_FALSE
    }
}

pub fn to_avifResult<T>(res: &AvifResult<T>) -> avifResult {
    match res {
        Ok(_) => avifResult::Ok,
        Err(err) => {
            let res: avifResult = err.into();
            res
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn crabby_avifResultToString(_res: avifResult) -> *const c_char {
    // TODO: implement this function.
    std::ptr::null()
}

pub type avifCropRect = CropRect;

#[no_mangle]
pub unsafe extern "C" fn crabby_avifCropRectConvertCleanApertureBox(
    cropRect: *mut avifCropRect,
    clap: *const avifCleanApertureBox,
    imageW: u32,
    imageH: u32,
    yuvFormat: avifPixelFormat,
    _diag: *mut avifDiagnostics,
) -> avifBool {
    let rust_clap: CleanAperture = unsafe { (&(*clap)).into() };
    let rect = unsafe { &mut (*cropRect) };
    *rect = match CropRect::create_from(&rust_clap, imageW, imageH, yuvFormat.into()) {
        Ok(x) => x,
        Err(_) => return AVIF_FALSE,
    };
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

/// cbindgen:rename-all=ScreamingSnakeCase
#[repr(C)]
pub enum avifHeaderFormat {
    AvifHeaderFull,
    AvifHeaderReduced,
}

#[repr(C)]
pub struct avifPixelFormatInfo {
    monochrome: avifBool,
    chromaShiftX: c_int,
    chromaShiftY: c_int,
}

#[no_mangle]
pub unsafe extern "C" fn crabby_avifGetPixelFormatInfo(
    format: avifPixelFormat,
    info: *mut avifPixelFormatInfo,
) {
    if info.is_null() {
        return;
    }
    let info = unsafe { &mut (*info) };
    match format {
        avifPixelFormat::Yuv444 => {
            info.chromaShiftX = 0;
            info.chromaShiftY = 0;
            info.monochrome = AVIF_FALSE;
        }
        avifPixelFormat::Yuv422 => {
            info.chromaShiftX = 1;
            info.chromaShiftY = 0;
            info.monochrome = AVIF_FALSE;
        }
        avifPixelFormat::Yuv420 => {
            info.chromaShiftX = 1;
            info.chromaShiftY = 1;
            info.monochrome = AVIF_FALSE;
        }
        avifPixelFormat::Yuv400 => {
            info.chromaShiftX = 1;
            info.chromaShiftY = 1;
            info.monochrome = AVIF_TRUE;
        }
        _ => {}
    }
}

#[no_mangle]
pub unsafe extern "C" fn crabby_avifDiagnosticsClearError(diag: *mut avifDiagnostics) {
    if diag.is_null() {
        return;
    }
    unsafe {
        (*diag).error[0] = 0;
    }
}

#[repr(C)]
pub enum avifCodecFlag {
    CanDecode = (1 << 0),
    CanEncode = (1 << 1),
}
pub type avifCodecFlags = u32;

// TODO: This can be moved into the rust layer and renamed.
#[repr(C)]
#[derive(Default)]
pub struct avifIOStats {
    colorOBUSize: usize,
    alphaOBUSize: usize,
}

pub const AVIF_TRANSFORM_NONE: u32 = 0;
pub const AVIF_TRANSFORM_PASP: u32 = 1 << 0;
pub const AVIF_TRANSFORM_CLAP: u32 = 1 << 1;
pub const AVIF_TRANSFORM_IROT: u32 = 1 << 2;
pub const AVIF_TRANSFORM_IMIR: u32 = 1 << 3;
pub type avifTransformFlags = u32;

pub const AVIF_COLOR_PRIMARIES_BT709: u32 = 1;
pub const AVIF_COLOR_PRIMARIES_IEC61966_2_4: u32 = 1;
pub const AVIF_COLOR_PRIMARIES_BT2100: u32 = 9;
pub const AVIF_COLOR_PRIMARIES_DCI_P3: u32 = 12;
pub const AVIF_TRANSFER_CHARACTERISTICS_SMPTE2084: u32 = 16;

#[no_mangle]
pub unsafe extern "C" fn crabby_avifAlloc(size: usize) -> *mut c_void {
    let mut data: Vec<u8> = Vec::new();
    data.reserve_exact(size);
    data.resize(size, 0);
    let mut boxed_slice = data.into_boxed_slice();
    let ptr = boxed_slice.as_mut_ptr();
    std::mem::forget(boxed_slice);
    ptr as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn crabby_avifFree(p: *mut c_void) {
    if !p.is_null() {
        let _ = unsafe { Box::from_raw(p as *mut u8) };
    }
}
