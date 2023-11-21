use std::os::raw::c_char;
use std::os::raw::c_int;

use std::ffi::CStr;

use std::slice;

use libc::size_t;

use crate::decoder::*;
use crate::AvifError;
use crate::AvifResult;

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
#[derive(Default)]
pub struct avifDecoder {
    // avifCodecChoice codecChoice;
    pub maxThreads: c_int,
    //avifDecoderSource requestedSource;
    pub allowIncremental: avifBool,
    pub allowProgressive: avifBool,
    pub ignoreExif: avifBool,
    pub ignoreXMP: avifBool,
    // uint32_t imageSizeLimit;
    // uint32_t imageDimensionLimit;
    // uint32_t imageCountLimit;
    // avifStrictFlags strictFlags;

    // Output params.

    // avifImage * image;
    pub imageIndex: c_int,
    pub imageCount: c_int,
    // avifProgressiveState progressiveState; // See avifProgressiveState declaration
    // avifImageTiming imageTiming;           //
    // uint64_t timescale;                    // timescale of the media (Hz)
    // double duration;                       // duration of a single playback of the image sequence in seconds
    //                                        // (durationInTimescales / timescale)
    // uint64_t durationInTimescales;         // duration of a single playback of the image sequence in "timescales"
    // int repetitionCount;                   // number of times the sequence has to be repeated. This can also be one of
    //                                        // AVIF_REPETITION_COUNT_INFINITE or AVIF_REPETITION_COUNT_UNKNOWN. Essentially, if
    //                                        // repetitionCount is a non-negative integer `n`, then the image sequence should be
    //                                        // played back `n + 1` times.

    //avifBool alphaPresent;

    //avifIOStats ioStats;

    //avifDiagnostics diag;

    //avifIO * io;

    //struct avifDecoderData * data;

    //avifBool gainMapPresent;
    // avifBool enableDecodingGainMap;
    // avifBool enableParsingGainMapMetadata;
    // avifBool ignoreColorAndAlpha;
    // avifBool imageSequenceTrackPresent;
    rust_decoder: Box<AvifDecoder>,
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

#[no_mangle]
pub unsafe extern "C" fn avifDecoderParse(decoder: *mut avifDecoder) -> avifResult {
    let rust_decoder = &mut (*decoder).rust_decoder;
    let res = rust_decoder.parse();
    let retval = to_avifResult(&res);
    if retval != avifResult::AVIF_RESULT_OK {
        return retval;
    }
    (*decoder).imageCount = rust_decoder.image_count as i32;
    retval
}

#[no_mangle]
pub unsafe extern "C" fn avifDecoderDestroy(decoder: *mut avifDecoder) {
    let _ = Box::from_raw(decoder);
}
