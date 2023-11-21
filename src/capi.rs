use std::os::raw::c_char;
use std::os::raw::c_int;

use std::ffi::CStr;

use std::slice;

use libc::size_t;

use crate::decoder::*;
use crate::AvifError;
use crate::AvifResult;
use crate::AvifStrictness;
use crate::AvifStrictnessFlag;
use crate::PixelFormat;

#[repr(C)]
pub struct avifROData {
    pub data: *const u8,
    pub size: size_t,
}

#[repr(C)]
#[derive(PartialEq)]
pub enum avifResult {
    AVIF_RESULT_OK = 0,
    AVIF_RESULT_UNKNOWN_ERROR = 1,
    AVIF_RESULT_INVALID_FTYP = 2,
    AVIF_RESULT_NO_CONTENT = 3,
    AVIF_RESULT_NO_YUV_FORMAT_SELECTED = 4,
    AVIF_RESULT_REFORMAT_FAILED = 5,
    AVIF_RESULT_UNSUPPORTED_DEPTH = 6,
    AVIF_RESULT_ENCODE_COLOR_FAILED = 7,
    AVIF_RESULT_ENCODE_ALPHA_FAILED = 8,
    AVIF_RESULT_BMFF_PARSE_FAILED = 9,
    AVIF_RESULT_MISSING_IMAGE_ITEM = 10,
    AVIF_RESULT_DECODE_COLOR_FAILED = 11,
    AVIF_RESULT_DECODE_ALPHA_FAILED = 12,
    AVIF_RESULT_COLOR_ALPHA_SIZE_MISMATCH = 13,
    AVIF_RESULT_ISPE_SIZE_MISMATCH = 14,
    AVIF_RESULT_NO_CODEC_AVAILABLE = 15,
    AVIF_RESULT_NO_IMAGES_REMAINING = 16,
    AVIF_RESULT_INVALID_EXIF_PAYLOAD = 17,
    AVIF_RESULT_INVALID_IMAGE_GRID = 18,
    AVIF_RESULT_INVALID_CODEC_SPECIFIC_OPTION = 19,
    AVIF_RESULT_TRUNCATED_DATA = 20,
    AVIF_RESULT_IO_NOT_SET = 21,
    AVIF_RESULT_IO_ERROR = 22,
    AVIF_RESULT_WAITING_ON_IO = 23,
    AVIF_RESULT_INVALID_ARGUMENT = 24,
    AVIF_RESULT_NOT_IMPLEMENTED = 25,
    AVIF_RESULT_OUT_OF_MEMORY = 26,
    AVIF_RESULT_CANNOT_CHANGE_SETTING = 27,
    AVIF_RESULT_INCOMPATIBLE_IMAGE = 28,
    AVIF_RESULT_ENCODE_GAIN_MAP_FAILED = 29,
    AVIF_RESULT_DECODE_GAIN_MAP_FAILED = 30,
    AVIF_RESULT_INVALID_TONE_MAPPED_IMAGE = 31,
}

pub type avifBool = c_int;
pub const AVIF_TRUE: c_int = 1;
pub const AVIF_FALSE: c_int = 0;

#[repr(C)]
#[derive(Debug)]
pub enum avifPixelFormat {
    AVIF_PIXEL_FORMAT_NONE,
    AVIF_PIXEL_FORMAT_YUV444,
    AVIF_PIXEL_FORMAT_YUV422,
    AVIF_PIXEL_FORMAT_YUV420,
    AVIF_PIXEL_FORMAT_YUV400,
    AVIF_PIXEL_FORMAT_COUNT,
}

impl From<PixelFormat> for avifPixelFormat {
    fn from(format: PixelFormat) -> Self {
        match format {
            PixelFormat::None => Self::AVIF_PIXEL_FORMAT_NONE,
            PixelFormat::Yuv444 => Self::AVIF_PIXEL_FORMAT_YUV444,
            PixelFormat::Yuv422 => Self::AVIF_PIXEL_FORMAT_YUV422,
            PixelFormat::Yuv420 => Self::AVIF_PIXEL_FORMAT_YUV420,
            PixelFormat::Monochrome => Self::AVIF_PIXEL_FORMAT_YUV400,
        }
    }
}

#[repr(C)]
#[derive(Debug)]
enum avifRange {
    AVIF_RANGE_LIMITED = 0,
    AVIF_RANGE_FULL = 1,
}

impl From<bool> for avifRange {
    fn from(full_range: bool) -> Self {
        match full_range {
            true => Self::AVIF_RANGE_FULL,
            false => Self::AVIF_RANGE_LIMITED,
        }
    }
}

#[repr(C)]
#[derive(Debug)]
enum avifChromaSamplePosition {
    AVIF_CHROMA_SAMPLE_POSITION_UNKNOWN = 0,
    AVIF_CHROMA_SAMPLE_POSITION_VERTICAL = 1,
    AVIF_CHROMA_SAMPLE_POSITION_COLOCATED = 2,
}

#[repr(C)]
#[derive(Debug)]
pub struct avifImage {
    width: u32,
    height: u32,
    depth: u32,

    yuvFormat: avifPixelFormat,
    yuvRange: avifRange,
    yuvChromaSamplePosition: avifChromaSamplePosition,
    yuvPlanes: [*mut u8; 3],
    yuvRowBytes: [u32; 3],
    imageOwnsYUVPlanes: avifBool,

    alphaPlane: *mut u8,
    alphaRowBytes: u32,
    imageOwnsAlphaPlane: avifBool,
    alphaPremultiplied: avifBool,
    // avifRWData icc;
    // avifColorPrimaries colorPrimaries;
    // avifTransferCharacteristics transferCharacteristics;
    // avifMatrixCoefficients matrixCoefficients;
    // avifContentLightLevelInformationBox clli;
    // avifTransformFlags transformFlags;
    // avifPixelAspectRatioBox pasp;
    // avifCleanApertureBox clap;
    // avifImageRotation irot;
    // avifImageMirror imir;
    // avifRWData exif;
    // avifRWData xmp;
    // avifGainMap gainMap;
}

impl Default for avifImage {
    fn default() -> Self {
        avifImage {
            width: 0,
            height: 0,
            depth: 0,
            yuvFormat: avifPixelFormat::AVIF_PIXEL_FORMAT_NONE,
            yuvRange: avifRange::AVIF_RANGE_FULL,
            yuvChromaSamplePosition: avifChromaSamplePosition::AVIF_CHROMA_SAMPLE_POSITION_UNKNOWN,
            yuvPlanes: [std::ptr::null_mut(); 3],
            yuvRowBytes: [0; 3],
            imageOwnsYUVPlanes: AVIF_FALSE,
            alphaPlane: std::ptr::null_mut(),
            alphaRowBytes: 0,
            imageOwnsAlphaPlane: AVIF_FALSE,
            alphaPremultiplied: AVIF_FALSE,
        }
    }
}

impl From<&AvifImage> for avifImage {
    fn from(image: &AvifImage) -> Self {
        let mut dst_image = Self::default();
        dst_image.width = image.width;
        dst_image.height = image.height;
        dst_image.depth = image.depth as u32;
        dst_image.yuvFormat = image.yuv_format.into();
        dst_image.yuvRange = image.full_range.into();
        //dst_image.yuvChromaSamplePosition: avifChromaSamplePosition,
        for i in 0usize..3 {
            if image.yuv_planes[i].is_none() {
                continue;
            }
            dst_image.yuvPlanes[i] = image.yuv_planes[i].unwrap() as *mut u8;
            dst_image.yuvRowBytes[i] = image.yuv_row_bytes[i];
        }
        if image.alpha_plane.is_some() {
            dst_image.alphaPlane = image.alpha_plane.unwrap() as *mut u8;
            dst_image.alphaRowBytes = image.alpha_row_bytes;
            dst_image.alphaPremultiplied = image.alpha_premultiplied as avifBool;
        }
        dst_image
    }
}

pub const AVIF_STRICT_DISABLED: u32 = 0;
pub const AVIF_STRICT_PIXI_REQUIRED: u32 = (1 << 0);
pub const AVIF_STRICT_CLAP_VALID: u32 = (1 << 1);
pub const AVIF_STRICT_ALPHA_ISPE_REQUIRED: u32 = (1 << 2);
pub const AVIF_STRICT_ENABLED: u32 =
    AVIF_STRICT_PIXI_REQUIRED | AVIF_STRICT_CLAP_VALID | AVIF_STRICT_ALPHA_ISPE_REQUIRED;
pub type avifStrictFlags = u32;

#[repr(C)]
pub struct avifDecoder {
    // avifCodecChoice codecChoice;
    pub maxThreads: i32,
    //avifDecoderSource requestedSource;
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
    // avifProgressiveState progressiveState; // See avifProgressiveState declaration
    // avifImageTiming imageTiming;           //
    pub timescale: u64,
    pub duration: f64,
    pub durationInTimescales: u64,
    pub repetitionCount: i32,

    pub alphaPresent: avifBool,

    //avifIOStats ioStats;

    //avifDiagnostics diag;

    //avifIO * io;

    //struct avifDecoderData * data;

    //avifBool gainMapPresent;
    // avifBool enableDecodingGainMap;
    // avifBool enableParsingGainMapMetadata;
    // avifBool ignoreColorAndAlpha;
    // avifBool imageSequenceTrackPresent;

    // TODO: maybe wrap these fields in a private data kind of field?
    rust_decoder: Box<AvifDecoder>,
    image_object: avifImage,
}

impl Default for avifDecoder {
    fn default() -> Self {
        Self {
            maxThreads: 1,
            allowIncremental: AVIF_FALSE,
            allowProgressive: AVIF_FALSE,
            ignoreExif: AVIF_FALSE,
            ignoreXMP: AVIF_FALSE,
            strictFlags: AVIF_STRICT_ENABLED,
            image: std::ptr::null_mut(),
            imageIndex: -1,
            imageCount: 0,
            timescale: 0,
            duration: 0.0,
            durationInTimescales: 0,
            repetitionCount: 0,
            alphaPresent: AVIF_FALSE,
            rust_decoder: Box::new(AvifDecoder::default()),
            image_object: avifImage::default(),
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
        Ok(x) => avifResult::AVIF_RESULT_OK,
        Err(err) => avifResult::AVIF_RESULT_UNKNOWN_ERROR,
    }
}

#[no_mangle]
pub unsafe extern "C" fn avifPeekCompatibleFileType(input: *const avifROData) -> avifBool {
    let data = slice::from_raw_parts((*input).data, (*input).size);
    to_avifBool(AvifDecoder::peek_compatible_file_type(data))
}

#[no_mangle]
pub unsafe extern "C" fn avifDecoderCreate() -> *mut avifDecoder {
    Box::into_raw(Box::new(avifDecoder::default()))
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

impl From<&avifDecoder> for AvifDecoderSettings {
    fn from(decoder: &avifDecoder) -> Self {
        let strictness = if decoder.strictFlags == AVIF_STRICT_DISABLED {
            AvifStrictness::None
        } else if decoder.strictFlags == AVIF_STRICT_ENABLED {
            AvifStrictness::All
        } else {
            let mut flags: Vec<AvifStrictnessFlag> = Vec::new();
            if (decoder.strictFlags & AVIF_STRICT_PIXI_REQUIRED) != 0 {
                flags.push(AvifStrictnessFlag::PixiRequired);
            }
            if (decoder.strictFlags & AVIF_STRICT_CLAP_VALID) != 0 {
                flags.push(AvifStrictnessFlag::ClapValid);
            }
            if (decoder.strictFlags & AVIF_STRICT_ALPHA_ISPE_REQUIRED) != 0 {
                flags.push(AvifStrictnessFlag::AlphaIspeRequired);
            }
            AvifStrictness::SpecificInclude(flags)
        };
        Self {
            strictness,
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
    if !res.is_ok() {
        return to_avifResult(&res);
    }

    // Copy image.
    (*decoder).image_object = res.unwrap().into();

    // Copy decoder.
    (*decoder).imageCount = rust_decoder.image_count as i32;
    (*decoder).image = (&mut (*decoder).image_object) as *mut avifImage;

    return avifResult::AVIF_RESULT_OK;
}

#[no_mangle]
pub unsafe extern "C" fn avifDecoderNextImage(decoder: *mut avifDecoder) -> avifResult {
    let rust_decoder = &mut (*decoder).rust_decoder;

    let res = rust_decoder.next_image();
    if res.is_none() {
        return avifResult::AVIF_RESULT_UNKNOWN_ERROR;
    }

    // Copy image.
    (*decoder).image_object = res.unwrap().into();

    // Copy decoder.
    (*decoder).imageCount = rust_decoder.image_count as i32;
    (*decoder).image = (&mut (*decoder).image_object) as *mut avifImage;

    return avifResult::AVIF_RESULT_OK;
}

#[no_mangle]
pub unsafe extern "C" fn avifDecoderDestroy(decoder: *mut avifDecoder) {
    let _ = Box::from_raw(decoder);
}
