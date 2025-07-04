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

use super::gainmap::*;
use super::image::*;
use super::io::*;
use super::types::*;

use std::ffi::CStr;
use std::num::NonZero;
use std::os::raw::c_char;

use crate::decoder::track::*;
use crate::decoder::*;
use crate::internal_utils::*;
use crate::*;

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
    pub imageTiming: ImageTiming,
    pub timescale: u64,
    pub duration: f64,
    pub durationInTimescales: u64,
    pub repetitionCount: i32,
    pub alphaPresent: avifBool,
    pub ioStats: IOStats,
    pub diag: avifDiagnostics,
    pub data: *mut avifDecoderData,
    pub imageContentToDecode: avifImageContentTypeFlags,
    pub imageSequenceTrackPresent: avifBool,

    // These fields are not part of libavif. Any new fields that are to be header file compatible
    // with libavif must be added before this line.
    pub androidMediaCodecOutputColorFormat: AndroidMediaCodecOutputColorFormat,
    pub compressionFormat: CompressionFormat,

    // Rust specific fields that are not accessed from the C/C++ layer.
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
            imageTiming: ImageTiming::default(),
            timescale: 0,
            duration: 0.0,
            durationInTimescales: 0,
            repetitionCount: 0,
            alphaPresent: AVIF_FALSE,
            ioStats: Default::default(),
            diag: avifDiagnostics::default(),
            data: std::ptr::null_mut(),
            imageContentToDecode: AVIF_IMAGE_CONTENT_COLOR_AND_ALPHA,
            imageSequenceTrackPresent: AVIF_FALSE,
            androidMediaCodecOutputColorFormat: AndroidMediaCodecOutputColorFormat::default(),
            compressionFormat: CompressionFormat::default(),
            rust_decoder: Box::<Decoder>::default(),
            image_object: avifImage::default(),
            gainmap_image_object: avifImage::default(),
            gainmap_object: avifGainMap::default(),
        }
    }
}

fn rust_decoder<'a>(decoder: *mut avifDecoder) -> &'a mut Decoder {
    &mut deref_mut!(decoder).rust_decoder
}

fn rust_decoder_const<'a>(decoder: *const avifDecoder) -> &'a Decoder {
    &deref_const!(decoder).rust_decoder
}

/// # Safety
/// Used by the C API to create an avifDecoder object with default values.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderCreate() -> *mut avifDecoder {
    Box::into_raw(Box::<avifDecoder>::default())
}

/// # Safety
/// Used by the C API with the following pre-conditions:
/// - if decoder is not null, it has to point to a valid avifDecoder object.
/// - if io is not null, it has to point to a valid avifIO object.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderSetIO(decoder: *mut avifDecoder, io: *mut avifIO) {
    check_pointer_or_return!(decoder);
    check_pointer_or_return!(io);
    rust_decoder(decoder).set_io(Box::new(avifIOWrapper::create(io)));
}

/// # Safety
/// Used by the C API with the following pre-conditions:
/// - if decoder is not null, it has to point to a valid avifDecoder object.
/// - if filename is not null, it has to point to a valid C-style string.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderSetIOFile(
    decoder: *mut avifDecoder,
    filename: *const c_char,
) -> avifResult {
    check_pointer!(decoder);
    check_pointer!(filename);
    // SAFETY: filename is guaranteed to be not-null and contain a valid C-string as per the
    // pre-conditions of this function.
    let filename = String::from(unsafe { CStr::from_ptr(filename) }.to_str().unwrap_or(""));
    rust_decoder(decoder).set_io_file(&filename).into()
}

/// # Safety
/// Used by the C API with the following pre-conditions:
/// - if decoder is not null, it has to point to a valid avifDecoder object.
/// - if data is not null, it has to be a valid buffer of size bytes.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderSetIOMemory(
    decoder: *mut avifDecoder,
    data: *const u8,
    size: usize,
) -> avifResult {
    check_pointer!(decoder);
    if !check_slice_from_raw_parts_safety(data, size) {
        return avifResult::InvalidArgument;
    }
    // SAFETY: Pre-conditions are met to call this function.
    unsafe { rust_decoder(decoder).set_io_raw(data, size) }.into()
}

/// # Safety
/// Used by the C API with the following pre-conditions:
/// - if decoder is not null, it has to point to a valid avifDecoder object.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderSetSource(
    decoder: *mut avifDecoder,
    source: Source,
) -> avifResult {
    check_pointer!(decoder);
    deref_mut!(decoder).requestedSource = source;
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
        let image_content_to_decode_flags: ImageContentType = match decoder.imageContentToDecode {
            AVIF_IMAGE_CONTENT_ALL => ImageContentType::All,
            AVIF_IMAGE_CONTENT_COLOR_AND_ALPHA => ImageContentType::ColorAndAlpha,
            AVIF_IMAGE_CONTENT_GAIN_MAP => ImageContentType::GainMap,
            _ => ImageContentType::None,
        };
        Self {
            source: decoder.requestedSource,
            strictness,
            allow_progressive: decoder.allowProgressive == AVIF_TRUE,
            allow_incremental: decoder.allowIncremental == AVIF_TRUE,
            ignore_exif: decoder.ignoreExif == AVIF_TRUE,
            ignore_xmp: decoder.ignoreXMP == AVIF_TRUE,
            image_content_to_decode: image_content_to_decode_flags,
            codec_choice: match decoder.codecChoice {
                avifCodecChoice::Auto => CodecChoice::Auto,
                avifCodecChoice::Dav1d => CodecChoice::Dav1d,
                avifCodecChoice::Libgav1 => CodecChoice::Libgav1,
                // Silently treat all other choices the same as Auto.
                _ => CodecChoice::Auto,
            },
            image_size_limit: NonZero::new(decoder.imageSizeLimit),
            image_dimension_limit: NonZero::new(decoder.imageDimensionLimit),
            image_count_limit: NonZero::new(decoder.imageCountLimit),
            max_threads: u32::try_from(decoder.maxThreads).unwrap_or(0),
            android_mediacodec_output_color_format: decoder.androidMediaCodecOutputColorFormat,
        }
    }
}

fn rust_decoder_to_avifDecoder(src: &Decoder, dst: &mut avifDecoder) {
    // Copy image.
    let image = src.image().unwrap();
    dst.image_object = image.into();

    // Copy decoder properties.
    dst.alphaPresent = to_avifBool(image.alpha_present);
    dst.imageSequenceTrackPresent = to_avifBool(image.image_sequence_track_present);
    dst.progressiveState = image.progressive_state;

    dst.imageTiming = src.image_timing();
    dst.imageCount = src.image_count() as i32;
    dst.imageIndex = src.image_index();
    dst.repetitionCount = match src.repetition_count() {
        RepetitionCount::Unknown => AVIF_REPETITION_COUNT_UNKNOWN,
        RepetitionCount::Infinite => AVIF_REPETITION_COUNT_INFINITE,
        RepetitionCount::Finite(x) => x as i32,
    };
    dst.timescale = src.timescale();
    dst.durationInTimescales = src.duration_in_timescales();
    dst.duration = src.duration();
    dst.ioStats = src.io_stats();
    dst.compressionFormat = src.compression_format();

    if src.gainmap_present() {
        dst.gainmap_image_object = (&src.gainmap().image).into();
        dst.gainmap_object = src.gainmap().into();
        if src.settings.image_content_to_decode.gainmap() {
            dst.gainmap_object.image = (&mut dst.gainmap_image_object) as *mut avifImage;
        }
        dst.image_object.gainMap = (&mut dst.gainmap_object) as *mut avifGainMap;
    }
    dst.image = (&mut dst.image_object) as *mut avifImage;
}

/// # Safety
/// Used by the C API with the following pre-conditions:
/// - if decoder is not null, it has to point to a valid avifDecoder object.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderParse(decoder: *mut avifDecoder) -> avifResult {
    check_pointer!(decoder);
    let rust_decoder = rust_decoder(decoder);
    rust_decoder.settings = deref_const!(decoder).into();
    let res = rust_decoder.parse();
    deref_mut!(decoder).diag.set_from_result(&res);
    if res.is_err() {
        return res.into();
    }
    rust_decoder_to_avifDecoder(rust_decoder, deref_mut!(decoder));
    avifResult::Ok
}

/// # Safety
/// Used by the C API with the following pre-conditions:
/// - if decoder is not null, it has to point to a valid avifDecoder object.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderNextImage(decoder: *mut avifDecoder) -> avifResult {
    check_pointer!(decoder);
    let rust_decoder = rust_decoder(decoder);
    rust_decoder.settings = deref_const!(decoder).into();

    let previous_decoded_row_count = rust_decoder.decoded_row_count();

    let res = rust_decoder.next_image();
    deref_mut!(decoder).diag.set_from_result(&res);
    let mut early_return = false;
    if res.is_err() {
        early_return = true;
        if rust_decoder.settings.allow_incremental
            && matches!(res.as_ref().err().unwrap(), AvifError::WaitingOnIo)
        {
            early_return = previous_decoded_row_count == rust_decoder.decoded_row_count();
        }
    }
    if early_return {
        return res.into();
    }
    rust_decoder_to_avifDecoder(rust_decoder, deref_mut!(decoder));
    res.into()
}

/// # Safety
/// Used by the C API with the following pre-conditions:
/// - if decoder is not null, it has to point to a valid avifDecoder object.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderNthImage(
    decoder: *mut avifDecoder,
    frameIndex: u32,
) -> avifResult {
    check_pointer!(decoder);
    let rust_decoder = rust_decoder(decoder);
    rust_decoder.settings = deref_const!(decoder).into();

    let previous_decoded_row_count = rust_decoder.decoded_row_count();
    let image_index = (rust_decoder.image_index() + 1) as u32;

    let res = rust_decoder.nth_image(frameIndex);
    deref_mut!(decoder).diag.set_from_result(&res);
    let mut early_return = false;
    if res.is_err() {
        early_return = true;
        if rust_decoder.settings.allow_incremental
            && matches!(res.as_ref().err().unwrap(), AvifError::WaitingOnIo)
        {
            if image_index != frameIndex {
                early_return = false;
            } else {
                early_return = previous_decoded_row_count == rust_decoder.decoded_row_count();
            }
        }
    }
    if early_return {
        return res.into();
    }
    rust_decoder_to_avifDecoder(rust_decoder, deref_mut!(decoder));
    res.into()
}

/// # Safety
/// Used by the C API with the following pre-conditions:
/// - if decoder is not null, it has to point to a valid avifDecoder object.
/// - if outTiming is not null, it has to point to a valid ImageTiming object.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderNthImageTiming(
    decoder: *const avifDecoder,
    frameIndex: u32,
    outTiming: *mut ImageTiming,
) -> avifResult {
    check_pointer!(decoder);
    check_pointer!(outTiming);
    let image_timing = rust_decoder_const(decoder).nth_image_timing(frameIndex);
    if let Ok(timing) = image_timing {
        *deref_mut!(outTiming) = timing;
    }
    image_timing.into()
}

/// # Safety
/// Used by the C API with the following pre-conditions:
/// - if decoder is not null, it has to point to a valid avifDecoder object.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderDestroy(decoder: *mut avifDecoder) {
    check_pointer_or_return!(decoder);
    // SAFETY: decoder is guaranteed to be not null, so this is ok.
    let _ = unsafe { Box::from_raw(decoder) };
}

/// # Safety
/// Used by the C API with the following pre-conditions:
/// - if decoder is not null, it has to point to a valid avifDecoder object.
/// - if image is not null, it has to point to a valid avifImage object.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderRead(
    decoder: *mut avifDecoder,
    image: *mut avifImage,
) -> avifResult {
    check_pointer!(decoder);
    check_pointer!(image);
    let rust_decoder = rust_decoder(decoder);
    rust_decoder.settings = deref_const!(decoder).into();

    let res = rust_decoder.parse();
    if res.is_err() {
        return res.into();
    }
    let res = rust_decoder.next_image();
    if res.is_err() {
        return res.into();
    }
    rust_decoder_to_avifDecoder(rust_decoder, deref_mut!(decoder));
    *deref_mut!(image) = deref_mut!(decoder).image_object.clone();
    avifResult::Ok
}

/// # Safety
/// Used by the C API with the following pre-conditions:
/// - if decoder is not null, it has to point to a valid avifDecoder object.
/// - if image is not null, it has to point to a valid avifImage object.
/// - if data is not null, it has to be a valid buffer of size bytes.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderReadMemory(
    decoder: *mut avifDecoder,
    image: *mut avifImage,
    data: *const u8,
    size: usize,
) -> avifResult {
    // SAFETY: Pre-conditions are met to call this function.
    let res = unsafe { crabby_avifDecoderSetIOMemory(decoder, data, size) };
    if res != avifResult::Ok {
        return res;
    }
    // SAFETY: Pre-conditions are met to call this function.
    unsafe { crabby_avifDecoderRead(decoder, image) }
}

/// # Safety
/// Used by the C API with the following pre-conditions:
/// - if decoder is not null, it has to point to a valid avifDecoder object.
/// - if image is not null, it has to point to a valid avifImage object.
/// - if filename is not null, it has to point to a valid C-style string.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderReadFile(
    decoder: *mut avifDecoder,
    image: *mut avifImage,
    filename: *const c_char,
) -> avifResult {
    // SAFETY: Pre-conditions are met to call this function.
    let res = unsafe { crabby_avifDecoderSetIOFile(decoder, filename) };
    if res != avifResult::Ok {
        return res;
    }
    // SAFETY: Pre-conditions are met to call this function.
    unsafe { crabby_avifDecoderRead(decoder, image) }
}

/// # Safety
/// Used by the C API with the following pre-conditions:
/// - if decoder is not null, it has to point to a valid avifDecoder object.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderIsKeyframe(
    decoder: *const avifDecoder,
    frameIndex: u32,
) -> avifBool {
    if decoder.is_null() {
        return AVIF_FALSE;
    }
    to_avifBool(rust_decoder_const(decoder).is_keyframe(frameIndex))
}

/// # Safety
/// Used by the C API with the following pre-conditions:
/// - if decoder is not null, it has to point to a valid avifDecoder object.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderNearestKeyframe(
    decoder: *const avifDecoder,
    frameIndex: u32,
) -> u32 {
    if decoder.is_null() {
        return 0;
    }
    rust_decoder_const(decoder).nearest_keyframe(frameIndex)
}

/// # Safety
/// Used by the C API with the following pre-conditions:
/// - if decoder is not null, it has to point to a valid avifDecoder object.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderDecodedRowCount(decoder: *const avifDecoder) -> u32 {
    if decoder.is_null() {
        return 0;
    }
    rust_decoder_const(decoder).decoded_row_count()
}

#[allow(non_camel_case_types)]
pub type avifExtent = Extent;

/// # Safety
/// Used by the C API with the following pre-conditions:
/// - if decoder is not null, it has to point to a valid avifDecoder object.
/// - if outExtent is not null, it has to point to a valid avifExtent object.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderNthImageMaxExtent(
    decoder: *const avifDecoder,
    frameIndex: u32,
    outExtent: *mut avifExtent,
) -> avifResult {
    check_pointer!(decoder);
    check_pointer!(outExtent);
    let res = rust_decoder_const(decoder).nth_image_max_extent(frameIndex);
    if res.is_err() {
        return res.into();
    }
    *deref_mut!(outExtent) = res.unwrap();
    avifResult::Ok
}

/// # Safety
/// Used by the C API with the following pre-conditions:
/// - if input is not null, it has to point to a valid avifROData object.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifPeekCompatibleFileType(input: *const avifROData) -> avifBool {
    if input.is_null() {
        return AVIF_FALSE;
    }
    let input = deref_const!(input);
    if !check_slice_from_raw_parts_safety(input.data, input.size) {
        return AVIF_FALSE;
    }
    // SAFETY: The buffer is guaranteed to be valid based on the pre-condition and the checks
    // above.
    let data = unsafe { std::slice::from_raw_parts(input.data, input.size) };
    to_avifBool(Decoder::peek_compatible_file_type(data))
}

/// # Safety
/// Used by the C API with the following pre-conditions:
/// - if decoder is not null, it has to point to a valid avifDecoder object.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderReset(decoder: *mut avifDecoder) -> avifResult {
    // SAFETY: Pre-conditions are met to call this function.
    unsafe { crabby_avifDecoderParse(decoder) }
}
