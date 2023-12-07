use super::gainmap::*;
use super::image::*;
use super::io::*;
use super::types::*;

use std::ffi::CStr;
use std::os::raw::c_char;

use crate::decoder::track::*;
use crate::decoder::*;

#[repr(C)]
pub struct avifDecoder {
    pub codecChoice: avifCodecChoice,
    pub maxThreads: i32,
    pub requestedSource: Source,
    pub allowProgressive: avifBool,
    pub allowIncremental: avifBool,
    pub ignoreExif: avifBool,
    pub ignoreXMP: avifBool,
    pub imageSizeLimit: u32,
    pub imageDimensionLimit: u32,
    pub imageCountLimit: u32,
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

    pub ioStats: avifIOStats,
    pub diag: avifDiagnostics,
    //avifIO * io;
    pub data: *mut avifDecoderData,
    pub gainMapPresent: avifBool,
    pub enableDecodingGainMap: avifBool,
    pub enableParsingGainMapMetadata: avifBool,
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
            codecChoice: avifCodecChoice::Auto,
            maxThreads: 1,
            requestedSource: Source::Auto,
            allowIncremental: AVIF_FALSE,
            allowProgressive: AVIF_FALSE,
            ignoreExif: AVIF_FALSE,
            ignoreXMP: AVIF_FALSE,
            imageSizeLimit: DEFAULT_IMAGE_SIZE_LIMIT,
            imageDimensionLimit: DEFAULT_IMAGE_DIMENSION_LIMIT,
            imageCountLimit: DEFAULT_IMAGE_COUNT_LIMIT,
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
            ioStats: Default::default(),
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

#[no_mangle]
pub unsafe extern "C" fn avifDecoderCreate() -> *mut avifDecoder {
    Box::into_raw(Box::<avifDecoder>::default())
}

#[no_mangle]
pub unsafe extern "C" fn avifDecoderSetIO(decoder: *mut avifDecoder, io: *mut avifIO) {
    let rust_decoder = &mut (*decoder).rust_decoder;
    rust_decoder.set_io(Box::new(avifIOWrapper::create(*io)));
}

#[no_mangle]
pub unsafe extern "C" fn avifDecoderSetIOFile(
    decoder: *mut avifDecoder,
    filename: *const c_char,
) -> avifResult {
    let rust_decoder = &mut (*decoder).rust_decoder;
    let filename = String::from(CStr::from_ptr(filename).to_str().unwrap_or(""));
    to_avifResult(&rust_decoder.set_io_file(&filename))
}

#[no_mangle]
pub unsafe extern "C" fn avifDecoderSetIOMemory(
    decoder: *mut avifDecoder,
    data: *const u8,
    size: usize,
) -> avifResult {
    let rust_decoder = &mut (*decoder).rust_decoder;
    to_avifResult(&rust_decoder.set_io_raw(data, size))
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
            codec_choice: match decoder.codecChoice {
                avifCodecChoice::Auto => CodecChoice::Auto,
                avifCodecChoice::Dav1d => CodecChoice::Dav1d,
                avifCodecChoice::Libgav1 => CodecChoice::Libgav1,
                // Silently treat all other choices the same as Auto.
                _ => CodecChoice::Auto,
            },
            image_size_limit: decoder.imageSizeLimit,
            image_dimension_limit: decoder.imageDimensionLimit,
            image_count_limit: decoder.imageCountLimit,
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

    // Copy image.
    let image = res.unwrap();
    (*decoder).image_object = image.into();

    // Copy decoder properties.
    (*decoder).alphaPresent = to_avifBool(image.alpha_present);
    (*decoder).imageSequenceTrackPresent = to_avifBool(image.image_sequence_track_present);
    (*decoder).progressiveState = image.progressive_state;
    (*decoder).imageCount = rust_decoder.image_count as i32;
    (*decoder).repetitionCount = match rust_decoder.repetition_count {
        RepetitionCount::Unknown => AVIF_REPETITION_COUNT_UNKNOWN,
        RepetitionCount::Infinite => AVIF_REPETITION_COUNT_INFINITE,
        RepetitionCount::Finite(x) => x,
    };

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

    // Copy decoder properties.
    (*decoder).alphaPresent = to_avifBool(image.alpha_present);
    (*decoder).imageSequenceTrackPresent = to_avifBool(image.image_sequence_track_present);
    (*decoder).progressiveState = image.progressive_state;
    (*decoder).imageCount = rust_decoder.image_count as i32;
    (*decoder).repetitionCount = match rust_decoder.repetition_count {
        RepetitionCount::Unknown => AVIF_REPETITION_COUNT_UNKNOWN,
        RepetitionCount::Infinite => AVIF_REPETITION_COUNT_INFINITE,
        RepetitionCount::Finite(x) => x,
    };

    (*decoder).image = (&mut (*decoder).image_object) as *mut avifImage;

    avifResult::Ok
}

#[no_mangle]
pub unsafe extern "C" fn avifDecoderDestroy(decoder: *mut avifDecoder) {
    let _ = Box::from_raw(decoder);
}

#[no_mangle]
pub unsafe extern "C" fn avifPeekCompatibleFileType(input: *const avifROData) -> avifBool {
    let data = std::slice::from_raw_parts((*input).data, (*input).size);
    to_avifBool(Decoder::peek_compatible_file_type(data))
}
