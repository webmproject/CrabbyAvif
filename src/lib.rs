use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;

/// cbindgen:ignore
mod bindings;
/// cbindgen:ignore
mod dav1d;
pub mod decoder;
pub mod io;
mod mp4box;
mod stream;
pub mod utils;

#[cfg(feature = "capi")]
pub mod capi;

macro_rules! println {
    ($($rest:tt)*) => {
        #[cfg(debug_assertions)]
        std::println!($($rest)*)
    }
}

#[derive(Debug, Default, PartialEq, Copy, Clone)]
pub enum PixelFormat {
    #[default]
    None,
    Yuv444,
    Yuv422,
    Yuv420,
    Monochrome,
}

impl PixelFormat {
    pub fn plane_count(&self) -> usize {
        match self {
            PixelFormat::None => 0,
            PixelFormat::Monochrome => 1,
            PixelFormat::Yuv420 | PixelFormat::Yuv422 | PixelFormat::Yuv444 => 3,
        }
    }
}

#[derive(Debug, Default, PartialEq, Copy, Clone)]
pub enum AvifError {
    #[default]
    Ok,
    UnknownError,
    InvalidFtyp,
    NoContent,
    NoYuvFormatSelected,
    ReformatFailed,
    UnsupportedDepth,
    EncodeColorFailed,
    EncodeAlphaFailed,
    BmffParseFailed, // TODO: this can contain an error string?
    MissingImageItem,
    DecodeColorFailed,
    DecodeAlphaFailed,
    ColorAlphaSizeMismatch,
    IspeSizeMismatch,
    NoCodecAvailable,
    NoImagesRemaining,
    InvalidExifPayload,
    InvalidImageGrid,
    InvalidCodecSpecificOption,
    TruncatedData,
    IoNotSet,
    IoError,
    WaitingOnIo,
    InvalidArgument,
    NotImplemented,
    OutOfMemory,
    CannotChangeSetting,
    IncompatibleImage,
    EncodeGainMapFailed,
    DecodeGainMapFailed,
    InvalidToneMappedImage,
}

pub type AvifResult<T> = Result<T, AvifError>;

#[derive(Debug)]
pub enum AvifStrictnessFlag {
    PixiRequired,
    ClapValid,
    AlphaIspeRequired,
}

#[derive(Debug, Default)]
pub enum AvifStrictness {
    None,
    #[default]
    All,
    SpecificInclude(Vec<AvifStrictnessFlag>),
    SpecificExclude(Vec<AvifStrictnessFlag>),
}
