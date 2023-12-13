#![deny(unsafe_op_in_unsafe_fn)]

pub mod decoder;
pub mod image;
pub mod reformat;
pub mod utils;

#[cfg(feature = "capi")]
pub mod capi;

/// cbindgen:ignore
mod codecs;

mod internal_utils;
mod parser;

use num_derive::FromPrimitive;
use num_traits::cast::FromPrimitive;

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

macro_rules! impl_from_primitive {
    ($from:ty, $from_func: ident, $to: ident, $default: ident) => {
        impl From<$from> for $to {
            fn from(value: $from) -> Self {
                $to::$from_func(value).unwrap_or($to::$default)
            }
        }
    };
}

#[repr(C)]
#[derive(Debug, Default, PartialEq, Copy, Clone, FromPrimitive)]
pub enum ChromaSamplePosition {
    #[default]
    Unknown = 0,
    Vertical = 1,
    Colocated = 2,
}

impl_from_primitive!(u32, from_u32, ChromaSamplePosition, Unknown);

#[repr(C)]
#[derive(Debug, Default, PartialEq, Copy, Clone, FromPrimitive)]
pub enum ColorPrimaries {
    Unknown = 0,
    Srgb = 1,
    #[default]
    Unspecified = 2,
    Bt470m = 4,
    Bt470bg = 5,
    Bt601 = 6,
    Smpte240 = 7,
    GenericFilm = 8,
    Bt2020 = 9,
    Xyz = 10,
    Smpte431 = 11,
    Smpte432 = 12,
    Ebu3213 = 22,
}

impl_from_primitive!(u16, from_u16, ColorPrimaries, Unspecified);

#[allow(non_camel_case_types, non_upper_case_globals)]
impl ColorPrimaries {
    // TODO: expose these in the capi?
    pub const Bt709: Self = Self::Srgb;
    pub const Iec61966_2_4: Self = Self::Srgb;
    pub const Bt2100: Self = Self::Bt2020;
    pub const Dci_p3: Self = Self::Smpte432;
}

#[repr(C)]
#[derive(Debug, Default, PartialEq, Copy, Clone, FromPrimitive)]
pub enum TransferCharacteristics {
    Unknown = 0,
    Bt709 = 1,
    #[default]
    Unspecified = 2,
    Bt470m = 4,  // 2.2 gamma
    Bt470bg = 5, // 2.8 gamma
    Bt601 = 6,
    Smpte240 = 7,
    Linear = 8,
    Log100 = 9,
    Log100Sqrt10 = 10,
    Iec61966 = 11,
    Bt1361 = 12,
    Srgb = 13,
    Bt2020_10bit = 14,
    Bt2020_12bit = 15,
    Pq = 16, // Perceptual Quantizer (HDR); BT.2100 PQ

    Smpte428 = 17,
    Hlg = 18, // Hybrid Log-Gamma (HDR); ARIB STD-B67; BT.2100 HLG
}

impl_from_primitive!(u16, from_u16, TransferCharacteristics, Unspecified);

#[allow(non_upper_case_globals)]
impl TransferCharacteristics {
    pub const Smpte2084: Self = Self::Pq;
}

#[repr(C)]
#[derive(Debug, Default, PartialEq, Copy, Clone, FromPrimitive)]
pub enum MatrixCoefficients {
    Identity = 0,
    Bt709 = 1,
    #[default]
    Unspecified = 2,
    Fcc = 4,
    Bt470bg = 5,
    Bt601 = 6,
    Smpte240 = 7,
    Ycgco = 8,
    Bt2020Ncl = 9,
    Bt2020Cl = 10,
    Smpte2085 = 11,
    ChromaDerivedNcl = 12,
    ChromaDerivedCl = 13,
    Ictcp = 14,
    YcgcoRe = 15,
    YcgcoRo = 16,
}

impl_from_primitive!(u16, from_u16, MatrixCoefficients, Unspecified);

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
