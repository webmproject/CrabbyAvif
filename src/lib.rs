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

#[derive(Debug, Default, PartialEq, Copy, Clone)]
pub enum PixelFormat {
    Yuv444,
    Yuv422,
    #[default]
    Yuv420,
    Monochrome,
}

impl PixelFormat {
    pub fn plane_count(&self) -> usize {
        match self {
            PixelFormat::Monochrome => 1,
            PixelFormat::Yuv420 | PixelFormat::Yuv422 | PixelFormat::Yuv444 => 3,
        }
    }
}

#[derive(Debug, Default, PartialEq, Copy, Clone)]
pub enum ChromaSamplePosition {
    #[default]
    Unknown = 0,
    Vertical = 1,
    Colocated = 2,
}

impl From<u8> for ChromaSamplePosition {
    fn from(chroma_sample_position: u8) -> Self {
        match chroma_sample_position {
            0 => ChromaSamplePosition::Unknown,
            1 => ChromaSamplePosition::Vertical,
            2 => ChromaSamplePosition::Colocated,
            _ => ChromaSamplePosition::Unknown,
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

#[derive(Debug, Default)]
pub enum AvifProgressiveState {
    #[default]
    Unavailable,
    Available,
    Active,
}
