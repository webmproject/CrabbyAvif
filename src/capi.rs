#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(clippy::missing_safety_doc)]

use std::os::raw::c_char;
use std::os::raw::c_int;

use std::ffi::CStr;

use std::slice;

use crate::decoder::gainmap::*;
use crate::decoder::*;
use crate::image::*;
use crate::parser::mp4box::*;
use crate::*;

#[repr(C)]
pub struct avifROData {
    pub data: *const u8,
    pub size: usize,
}

#[repr(C)]
#[derive(Debug)]
pub struct avifRWData {
    data: *mut u8,
    size: usize,
}

impl Default for avifRWData {
    fn default() -> Self {
        avifRWData {
            data: std::ptr::null_mut(),
            size: 0,
        }
    }
}

impl From<&Vec<u8>> for avifRWData {
    fn from(v: &Vec<u8>) -> Self {
        avifRWData {
            data: v.as_ptr() as *mut u8,
            size: v.len(),
        }
    }
}

#[repr(C)]
#[derive(PartialEq)]
pub enum avifResult {
    Ok,
    UnknownError,
    InvalidFtyp,
    NoContent,
    NoYuvFormatSelected,
    ReformatFailed,
    UnsupportedDepth,
    EncodeColorFailed,
    EncodeAlphaFailed,
    BmffParseFailed,
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

pub type avifBool = c_int;
pub const AVIF_TRUE: c_int = 1;
pub const AVIF_FALSE: c_int = 0;

#[repr(C)]
#[derive(Debug)]
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

#[repr(C)]
#[derive(Debug)]
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

type avifContentLightLevelInformationBox = ContentLightLevelInformation;

#[repr(C)]
#[derive(Debug, Default)]
pub struct avifGainMapMetadata {
    gainMapMinN: [i32; 3],
    gainMapMinD: [u32; 3],

    gainMapMaxN: [i32; 3],
    gainMapMaxD: [u32; 3],

    gainMapGammaN: [u32; 3],
    gainMapGammaD: [u32; 3],

    baseOffsetN: [i32; 3],
    baseOffsetD: [u32; 3],

    alternateOffsetN: [i32; 3],
    alternateOffsetD: [u32; 3],

    baseHdrHeadroomN: u32,
    baseHdrHeadroomD: u32,

    alternateHdrHeadroomN: u32,
    alternateHdrHeadroomD: u32,

    backwardDirection: avifBool,
    useBaseColorSpace: avifBool,
}

impl From<&GainMapMetadata> for avifGainMapMetadata {
    fn from(m: &GainMapMetadata) -> Self {
        avifGainMapMetadata {
            gainMapMinN: [m.min[0].0, m.min[1].0, m.min[2].0],
            gainMapMinD: [m.min[0].1, m.min[1].1, m.min[2].1],
            gainMapMaxN: [m.max[0].0, m.max[1].0, m.max[2].0],
            gainMapMaxD: [m.max[0].1, m.max[1].1, m.max[2].1],
            gainMapGammaN: [m.gamma[0].0, m.gamma[1].0, m.gamma[2].0],
            gainMapGammaD: [m.gamma[0].1, m.gamma[1].1, m.gamma[2].1],
            baseOffsetN: [m.base_offset[0].0, m.base_offset[1].0, m.base_offset[2].0],
            baseOffsetD: [m.base_offset[0].1, m.base_offset[1].1, m.base_offset[2].1],
            alternateOffsetN: [
                m.alternate_offset[0].0,
                m.alternate_offset[1].0,
                m.alternate_offset[2].0,
            ],
            alternateOffsetD: [
                m.alternate_offset[0].1,
                m.alternate_offset[1].1,
                m.alternate_offset[2].1,
            ],
            baseHdrHeadroomN: m.base_hdr_headroom.0,
            baseHdrHeadroomD: m.base_hdr_headroom.1,
            alternateHdrHeadroomN: m.alternate_hdr_headroom.0,
            alternateHdrHeadroomD: m.alternate_hdr_headroom.1,
            backwardDirection: m.backward_direction as avifBool,
            useBaseColorSpace: m.use_base_color_space as avifBool,
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct avifGainMap {
    image: *mut avifImage,
    metadata: avifGainMapMetadata,
    altICC: avifRWData,
    altColorPrimaries: ColorPrimaries,
    altTransferCharacteristics: TransferCharacteristics,
    altMatrixCoefficients: MatrixCoefficients,
    altYUVRange: avifRange,
    altDepth: u32,
    altPlaneCount: u32,
    //avifContentLightLevelInformationBox altCLLI;
}

impl Default for avifGainMap {
    fn default() -> Self {
        avifGainMap {
            image: std::ptr::null_mut(),
            metadata: avifGainMapMetadata::default(),
            altICC: avifRWData::default(),
            altColorPrimaries: ColorPrimaries::default(),
            altTransferCharacteristics: TransferCharacteristics::default(),
            altMatrixCoefficients: MatrixCoefficients::default(),
            altYUVRange: avifRange::Full,
            altDepth: 0,
            altPlaneCount: 0,
        }
    }
}

impl From<&GainMap> for avifGainMap {
    fn from(gainmap: &GainMap) -> Self {
        avifGainMap {
            metadata: (&gainmap.metadata).into(),
            altICC: (&gainmap.alt_icc).into(),
            altColorPrimaries: gainmap.alt_color_primaries,
            altTransferCharacteristics: gainmap.alt_transfer_characteristics,
            altMatrixCoefficients: gainmap.alt_matrix_coefficients,
            altYUVRange: gainmap.alt_full_range.into(),
            altDepth: u32::from(gainmap.alt_plane_depth),
            altPlaneCount: u32::from(gainmap.alt_plane_count),
            ..Self::default()
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct avifImage {
    width: u32,
    height: u32,
    depth: u32,

    yuvFormat: avifPixelFormat,
    yuvRange: avifRange,
    yuvChromaSamplePosition: ChromaSamplePosition,
    yuvPlanes: [*mut u8; 3],
    yuvRowBytes: [u32; 3],
    imageOwnsYUVPlanes: avifBool,

    alphaPlane: *mut u8,
    alphaRowBytes: u32,
    imageOwnsAlphaPlane: avifBool,
    alphaPremultiplied: avifBool,

    icc: avifRWData,
    colorPrimaries: ColorPrimaries,
    transferCharacteristics: TransferCharacteristics,
    matrixCoefficients: MatrixCoefficients,

    clli: avifContentLightLevelInformationBox,
    // avifTransformFlags transformFlags;
    // avifPixelAspectRatioBox pasp;
    // avifCleanApertureBox clap;
    // avifImageRotation irot;
    // avifImageMirror imir;
    exif: avifRWData,
    xmp: avifRWData,
    gainMap: *mut avifGainMap,
}

impl Default for avifImage {
    fn default() -> Self {
        avifImage {
            width: 0,
            height: 0,
            depth: 0,
            yuvFormat: avifPixelFormat::None,
            yuvRange: avifRange::Full,
            yuvChromaSamplePosition: ChromaSamplePosition::default(),
            yuvPlanes: [std::ptr::null_mut(); 3],
            yuvRowBytes: [0; 3],
            imageOwnsYUVPlanes: AVIF_FALSE,
            alphaPlane: std::ptr::null_mut(),
            alphaRowBytes: 0,
            imageOwnsAlphaPlane: AVIF_FALSE,
            alphaPremultiplied: AVIF_FALSE,
            icc: avifRWData::default(),
            colorPrimaries: ColorPrimaries::default(),
            transferCharacteristics: TransferCharacteristics::default(),
            matrixCoefficients: MatrixCoefficients::default(),
            clli: ContentLightLevelInformation::default(),
            exif: avifRWData::default(),
            xmp: avifRWData::default(),
            gainMap: std::ptr::null_mut(),
        }
    }
}

impl From<&ImageInfo> for avifImage {
    fn from(info: &ImageInfo) -> Self {
        avifImage {
            width: info.width,
            height: info.height,
            depth: info.depth as u32,
            yuvFormat: info.yuv_format.into(),
            yuvRange: info.full_range.into(),
            yuvChromaSamplePosition: info.chroma_sample_position.into(),
            alphaPremultiplied: info.alpha_premultiplied as avifBool,
            icc: (&info.icc).into(),
            colorPrimaries: info.color_primaries,
            transferCharacteristics: info.transfer_characteristics,
            matrixCoefficients: info.matrix_coefficients,
            clli: info.clli,
            exif: (&info.exif).into(),
            xmp: (&info.xmp).into(),
            ..Self::default()
        }
    }
}

impl From<&Image> for avifImage {
    fn from(image: &Image) -> Self {
        let mut dst_image: avifImage = (&image.info).into();
        for i in 0usize..3 {
            if image.planes[i].is_none() {
                continue;
            }
            dst_image.yuvPlanes[i] = image.planes[i].unwrap() as *mut u8;
            dst_image.yuvRowBytes[i] = image.row_bytes[i];
        }
        if image.planes[3].is_some() {
            dst_image.alphaPlane = image.planes[3].unwrap() as *mut u8;
            dst_image.alphaRowBytes = image.row_bytes[3];
        }
        dst_image
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

#[repr(C)]
pub struct avifDiagnostics {
    error: [c_char; 256],
}

impl Default for avifDiagnostics {
    fn default() -> Self {
        Self { error: [0; 256] }
    }
}

#[repr(C)]
pub struct avifDecoder {
    // avifCodecChoice codecChoice;
    pub maxThreads: i32,
    pub requestedSource: Source,
    pub allowIncremental: avifBool,
    pub allowProgressive: avifBool,
    pub ignoreExif: avifBool,
    pub ignoreXMP: avifBool,
    // uint32_t imageSizeLimit;
    // uint32_t imageDimensionLimit;
    // uint32_t imageCountLimit;
    pub strictFlags: avifStrictFlags,

    // Output params.
    pub image: *mut avifImage,
    pub imageIndex: i32,
    pub imageCount: i32,
    pub progressiveState: ProgressiveState,
    // avifImageTiming imageTiming;
    pub timescale: u64,
    pub duration: f64,
    pub durationInTimescales: u64,
    pub repetitionCount: i32,

    pub alphaPresent: avifBool,

    //avifIOStats ioStats;
    pub diag: avifDiagnostics,
    //avifIO * io;
    data: *mut avifDecoderData,
    gainMapPresent: avifBool,
    enableDecodingGainMap: avifBool,
    enableParsingGainMapMetadata: avifBool,
    // avifBool ignoreColorAndAlpha;
    pub imageSequenceTrackPresent: avifBool,

    // TODO: maybe wrap these fields in a private data kind of field?
    rust_decoder: Box<Decoder>,
    image_object: avifImage,
    gainmap_object: avifGainMap,
    gainmap_image_object: avifImage,
}

impl Default for avifDecoder {
    fn default() -> Self {
        Self {
            maxThreads: 1,
            requestedSource: Source::Auto,
            allowIncremental: AVIF_FALSE,
            allowProgressive: AVIF_FALSE,
            ignoreExif: AVIF_FALSE,
            ignoreXMP: AVIF_FALSE,
            strictFlags: AVIF_STRICT_ENABLED,
            image: std::ptr::null_mut(),
            imageIndex: -1,
            imageCount: 0,
            progressiveState: ProgressiveState::Unavailable,
            timescale: 0,
            duration: 0.0,
            durationInTimescales: 0,
            repetitionCount: 0,
            alphaPresent: AVIF_FALSE,
            diag: avifDiagnostics::default(),
            data: std::ptr::null_mut(),
            gainMapPresent: AVIF_FALSE,
            enableDecodingGainMap: AVIF_FALSE,
            enableParsingGainMapMetadata: AVIF_FALSE,
            imageSequenceTrackPresent: AVIF_FALSE,
            rust_decoder: Box::<Decoder>::default(),
            image_object: avifImage::default(),
            gainmap_image_object: avifImage::default(),
            gainmap_object: avifGainMap::default(),
        }
    }
}

fn to_avifBool(val: bool) -> avifBool {
    if val {
        AVIF_TRUE
    } else {
        AVIF_FALSE
    }
}

fn to_avifResult<T>(res: &AvifResult<T>) -> avifResult {
    match res {
        Ok(_) => avifResult::Ok,
        Err(err) => {
            let res: avifResult = err.into();
            res
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn avifPeekCompatibleFileType(input: *const avifROData) -> avifBool {
    let data = slice::from_raw_parts((*input).data, (*input).size);
    to_avifBool(Decoder::peek_compatible_file_type(data))
}

#[no_mangle]
pub unsafe extern "C" fn avifDecoderCreate() -> *mut avifDecoder {
    Box::into_raw(Box::<avifDecoder>::default())
}

#[no_mangle]
pub unsafe extern "C" fn avifDecoderSetIOFile(
    decoder: *mut avifDecoder,
    filename: *const c_char,
) -> avifResult {
    let rust_decoder = &mut (*decoder).rust_decoder;
    let filename = CStr::from_ptr(filename).to_str().unwrap_or("");
    let filename = String::from(filename);
    to_avifResult(&rust_decoder.set_io_file(&filename))
}

#[no_mangle]
pub unsafe extern "C" fn avifDecoderSetSource(
    decoder: *mut avifDecoder,
    source: Source,
) -> avifResult {
    (*decoder).requestedSource = source;
    // TODO: should decoder be reset here in case this is called after parse?
    avifResult::Ok
}

impl From<&avifDecoder> for Settings {
    fn from(decoder: &avifDecoder) -> Self {
        let strictness = if decoder.strictFlags == AVIF_STRICT_DISABLED {
            Strictness::None
        } else if decoder.strictFlags == AVIF_STRICT_ENABLED {
            Strictness::All
        } else {
            let mut flags: Vec<StrictnessFlag> = Vec::new();
            if (decoder.strictFlags & AVIF_STRICT_PIXI_REQUIRED) != 0 {
                flags.push(StrictnessFlag::PixiRequired);
            }
            if (decoder.strictFlags & AVIF_STRICT_CLAP_VALID) != 0 {
                flags.push(StrictnessFlag::ClapValid);
            }
            if (decoder.strictFlags & AVIF_STRICT_ALPHA_ISPE_REQUIRED) != 0 {
                flags.push(StrictnessFlag::AlphaIspeRequired);
            }
            Strictness::SpecificInclude(flags)
        };
        Self {
            source: decoder.requestedSource,
            strictness,
            allow_progressive: decoder.allowProgressive == AVIF_TRUE,
            ignore_exif: decoder.ignoreExif == AVIF_TRUE,
            ignore_xmp: decoder.ignoreXMP == AVIF_TRUE,
            enable_decoding_gainmap: decoder.enableDecodingGainMap == AVIF_TRUE,
            enable_parsing_gainmap_metadata: decoder.enableParsingGainMapMetadata == AVIF_TRUE,
            ..Self::default()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn avifDecoderParse(decoder: *mut avifDecoder) -> avifResult {
    let rust_decoder = &mut (*decoder).rust_decoder;
    rust_decoder.settings = (&(*decoder)).into();

    println!("settings: {:#?}", rust_decoder.settings);

    let res = rust_decoder.parse();
    if res.is_err() {
        return to_avifResult(&res);
    }

    // Copy image info.
    let info = res.unwrap();
    (*decoder).image_object = info.into();

    // Copy decoder properties. Properties from |info| must be copied first to
    // not mess with the borrow checker.
    (*decoder).alphaPresent = to_avifBool(info.alpha_present);
    (*decoder).imageSequenceTrackPresent = to_avifBool(info.image_sequence_track_present);
    (*decoder).progressiveState = info.progressive_state.into();
    (*decoder).imageCount = rust_decoder.image_count as i32;
    if rust_decoder.gainmap_present {
        (*decoder).gainMapPresent = AVIF_TRUE;
        (*decoder).gainmap_image_object = (&rust_decoder.gainmap.image).into();
        (*decoder).gainmap_object = (&rust_decoder.gainmap).into();
        (*decoder).gainmap_object.image = (&mut (*decoder).gainmap_image_object) as *mut avifImage;
        (*decoder).image_object.gainMap = (&mut (*decoder).gainmap_object) as *mut avifGainMap;
    }
    (*decoder).image = (&mut (*decoder).image_object) as *mut avifImage;

    avifResult::Ok
}

#[no_mangle]
pub unsafe extern "C" fn avifDecoderNextImage(decoder: *mut avifDecoder) -> avifResult {
    let rust_decoder = &mut (*decoder).rust_decoder;

    let res = rust_decoder.next_image();
    if res.is_err() {
        return to_avifResult(&res);
    }

    // Copy image.
    let image = res.unwrap();
    (*decoder).image_object = image.into();

    // Copy decoder properties. Properties from |image.info| must be copied first to
    // not mess with the borrow checker.
    (*decoder).alphaPresent = to_avifBool(image.info.alpha_present);
    (*decoder).imageSequenceTrackPresent = to_avifBool(image.info.image_sequence_track_present);
    (*decoder).progressiveState = image.info.progressive_state.into();
    (*decoder).imageCount = rust_decoder.image_count as i32;
    (*decoder).image = (&mut (*decoder).image_object) as *mut avifImage;

    avifResult::Ok
}

#[no_mangle]
pub unsafe extern "C" fn avifDecoderDestroy(decoder: *mut avifDecoder) {
    let _ = Box::from_raw(decoder);
}

#[no_mangle]
pub unsafe extern "C" fn avifImageDestroy(_image: *mut avifImage) {
    // Nothing to do.
}

#[no_mangle]
pub unsafe extern "C" fn avifResultToString(_res: avifResult) -> *const c_char {
    println!("hello:2323223222");
    // TODO: implement this function.
    std::ptr::null()
}
