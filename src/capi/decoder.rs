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
    pub allowSampleTransform: avifBool,

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
            allowSampleTransform: AVIF_TRUE,
            compressionFormat: CompressionFormat::default(),
            rust_decoder: Box::<Decoder>::default(),
            image_object: avifImage::default(),
            gainmap_image_object: avifImage::default(),
            gainmap_object: avifGainMap::default(),
        }
    }
}

#[no_mangle]
pub extern "C" fn crabby_avifDecoderCreate() -> *mut avifDecoder {
    Box::into_raw(Box::<avifDecoder>::default())
}

/// # Safety
/// - If `decoder` is non-null, it must be properly aligned and point to a valid, initialized
///   `avifDecoder` valid for exclusive access during this call.
/// - If `io` is non-null, it must point to a valid, initialized `avifIO` whose callbacks uphold the
///   C `avifIO` contract and that remains valid until `decoder` is destroyed or its IO is replaced.
/// - Passing null for either `decoder` or `io` is safe and is a no-op.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderSetIO(decoder: *mut avifDecoder, io: *mut avifIO) {
    check_pointer_or_return!(decoder);
    check_pointer_or_return!(io);
    deref_mut!(decoder)
        .rust_decoder
        .set_io(Box::new(avifIOWrapper::create(io)));
}

/// # Safety
/// - If `decoder` is non-null, it must be properly aligned and point to a valid, initialized
///   `avifDecoder` valid for exclusive access during this call.
/// - If `filename` is non-null, it must point to a valid, nul-terminated C string valid for reads
///   during this call.
/// - Passing null for `decoder` or `filename` safely returns `avifResult::InvalidArgument`.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderSetIOFile(
    decoder: *mut avifDecoder,
    filename: *const c_char,
) -> avifResult {
    check_pointer!(decoder);
    check_pointer!(filename);
    // SAFETY: `filename` is non-null (checked above), and the caller guarantees it points to a
    // valid, nul-terminated C string valid for reads during this call.
    let Ok(filename) = (unsafe { CStr::from_ptr(filename) }).to_str() else {
        return avifResult::IoError;
    };
    let filename = String::from(filename);
    deref_mut!(decoder)
        .rust_decoder
        .set_io_file(&filename)
        .into()
}

/// # Safety
/// - If `decoder` is non-null, it must be properly aligned and point to a valid, initialized
///   `avifDecoder` valid for exclusive access during this call.
/// - If `data` is non-null and `size <= isize::MAX as usize`, `data` must point to at least `size`
///   initialized bytes that remain valid and unmodified until `decoder` is destroyed or its IO is
///   replaced.
/// - Passing null for `decoder` or `data` (or `size > isize::MAX as usize`) safely returns
///   `avifResult::InvalidArgument`.
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
    let decoder = deref_mut!(decoder);
    // SAFETY: `data` is non-null, `size <= isize::MAX`, and the caller guarantees `data` points to
    // `size` initialized bytes that outlive the IO configured on `decoder`.
    unsafe { decoder.rust_decoder.set_io_raw(data, size) }.into()
}

/// # Safety
/// If `decoder` is non-null, it must be properly aligned and point to a valid, initialized
/// `avifDecoder` valid for exclusive writes during this call. Passing null safely returns
/// `avifResult::InvalidArgument`.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderSetSource(
    decoder: *mut avifDecoder,
    source: Source,
) -> avifResult {
    check_pointer!(decoder);
    deref_mut!(decoder).requestedSource = source;
    avifResult::Ok
}

impl TryFrom<&avifDecoder> for Settings {
    type Error = AvifError;
    fn try_from(decoder: &avifDecoder) -> AvifResult<Self> {
        let strictness = if decoder.strictFlags == AVIF_STRICT_DISABLED {
            Strictness::None
        } else if decoder.strictFlags == AVIF_STRICT_ENABLED {
            Strictness::All
        } else {
            let mut flags: Vec<StrictnessFlag> = Vec::new();
            if (decoder.strictFlags & AVIF_STRICT_PIXI_REQUIRED) != 0 {
                flags.try_push(StrictnessFlag::PixiRequired)?;
            }
            if (decoder.strictFlags & AVIF_STRICT_CLAP_VALID) != 0 {
                flags.try_push(StrictnessFlag::ClapValid)?;
            }
            if (decoder.strictFlags & AVIF_STRICT_ALPHA_ISPE_REQUIRED) != 0 {
                flags.try_push(StrictnessFlag::AlphaIspeRequired)?;
            }
            if (decoder.strictFlags & AVIF_STRICT_EXIF_VALID) != 0 {
                flags.try_push(StrictnessFlag::ExifValid)?;
            }
            if (decoder.strictFlags & AVIF_STRICT_MULTIPLE_ILOC_ENTRIES_FOR_SAME_ITEM_DISALLOWED)
                != 0
            {
                flags.try_push(StrictnessFlag::MultipleIlocEntriesForSameItemDisallowed)?;
            }
            if (decoder.strictFlags & AVIF_STRICT_HEVC_BRAND_REQUIRES_MOOV) != 0 {
                flags.try_push(StrictnessFlag::HevcBrandRequiresMoov)?;
            }
            if (decoder.strictFlags & AVIF_STRICT_ESSENTIAL_TRANSFORMATIVE_PROPERTIES) != 0 {
                flags.try_push(StrictnessFlag::EssentialTransformativeProperties)?;
            }
            Strictness::SpecificInclude(flags)
        };
        let image_content_to_decode_flags: ImageContentType = match decoder.imageContentToDecode {
            AVIF_IMAGE_CONTENT_ALL => ImageContentType::All,
            AVIF_IMAGE_CONTENT_COLOR_AND_ALPHA => ImageContentType::ColorAndAlpha,
            AVIF_IMAGE_CONTENT_COLOR_AND_GAIN_MAP => ImageContentType::ColorAndGainMap,
            AVIF_IMAGE_CONTENT_COLOR => ImageContentType::Color,
            AVIF_IMAGE_CONTENT_GAIN_MAP => ImageContentType::GainMap,
            // Alpha only is not currently supported and is mapped to ImageContentType::None.
            _ => ImageContentType::None,
        };
        Ok(Self {
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
                #[cfg(feature = "native_av1")]
                avifCodecChoice::Native => CodecChoice::Native,
                // Silently treat all other choices the same as Auto.
                _ => CodecChoice::Auto,
            },
            image_size_limit: NonZero::new(decoder.imageSizeLimit),
            image_dimension_limit: NonZero::new(decoder.imageDimensionLimit),
            image_count_limit: NonZero::new(decoder.imageCountLimit),
            max_threads: u32::try_from(decoder.maxThreads).unwrap_or(0),
            android_mediacodec_output_color_format: decoder.androidMediaCodecOutputColorFormat,
            allow_sample_transform: decoder.allowSampleTransform == AVIF_TRUE,
        })
    }
}

impl avifDecoder {
    fn sync_from_rust_decoder(&mut self) {
        let src = &self.rust_decoder;
        // Copy image.
        if let Some(image) = src.image() {
            self.image_object = image.into();

            // Copy decoder properties.
            self.alphaPresent = to_avifBool(image.alpha_present);
            self.imageSequenceTrackPresent = to_avifBool(image.image_sequence_track_present);
            self.progressiveState = image.progressive_state;
        }

        self.imageTiming = src.image_timing();
        self.imageCount = src.image_count() as i32;
        self.imageIndex = src.image_index();
        self.repetitionCount = match src.repetition_count() {
            RepetitionCount::Unknown => AVIF_REPETITION_COUNT_UNKNOWN,
            RepetitionCount::Infinite => AVIF_REPETITION_COUNT_INFINITE,
            RepetitionCount::Finite(x) => x as i32,
        };
        self.timescale = src.timescale();
        self.durationInTimescales = src.duration_in_timescales();
        self.duration = src.duration();
        self.ioStats = src.io_stats();
        self.compressionFormat = src.compression_format();

        if src.gainmap_present() {
            self.gainmap_image_object = (&src.gainmap().image).into();
            self.gainmap_object = src.gainmap().into();
            if src.settings.image_content_to_decode.gainmap() {
                self.gainmap_object.image = (&mut self.gainmap_image_object) as *mut avifImage;
            }
            self.image_object.gainMap = (&mut self.gainmap_object) as *mut avifGainMap;
        }
        self.image = if src.image().is_some() {
            (&mut self.image_object) as *mut avifImage
        } else {
            std::ptr::null_mut()
        };
    }
}

fn avif_decoder_parse(decoder: &mut avifDecoder) -> avifResult {
    let res: AvifResult<Settings> = (&*decoder).try_into();
    decoder.diag.set_from_result(&res);
    let Ok(settings) = res else {
        return res.into();
    };
    decoder.rust_decoder.settings = settings;
    let res = decoder.rust_decoder.parse();
    decoder.diag.set_from_result(&res);
    if res.is_err() {
        return res.into();
    }
    decoder.sync_from_rust_decoder();
    avifResult::Ok
}

/// # Safety
/// If `decoder` is non-null, it must be properly aligned and point to a valid, initialized
/// `avifDecoder` valid for exclusive access during this call (and any configured memory/custom IO
/// on `decoder` must still satisfy its lifetime/callback invariants). Passing null safely returns
/// `avifResult::InvalidArgument`.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderParse(decoder: *mut avifDecoder) -> avifResult {
    check_pointer!(decoder);
    avif_decoder_parse(deref_mut!(decoder))
}

/// # Safety
/// If `decoder` is non-null, it must be properly aligned and point to a valid, initialized
/// `avifDecoder` valid for exclusive access during this call. Passing null safely returns
/// `avifResult::InvalidArgument`.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderNextImage(decoder: *mut avifDecoder) -> avifResult {
    check_pointer!(decoder);
    let decoder = deref_mut!(decoder);
    let res: AvifResult<Settings> = (&*decoder).try_into();
    decoder.diag.set_from_result(&res);
    let Ok(settings) = res else {
        return res.into();
    };
    decoder.rust_decoder.settings = settings;

    let res = decoder.rust_decoder.next_image();
    decoder.diag.set_from_result(&res);
    decoder.sync_from_rust_decoder();
    res.into()
}

/// # Safety
/// If `decoder` is non-null, it must be properly aligned and point to a valid, initialized
/// `avifDecoder` valid for exclusive access during this call. Passing null safely returns
/// `avifResult::InvalidArgument`.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderNthImage(
    decoder: *mut avifDecoder,
    frameIndex: u32,
) -> avifResult {
    check_pointer!(decoder);
    let decoder = deref_mut!(decoder);
    let res: AvifResult<Settings> = (&*decoder).try_into();
    decoder.diag.set_from_result(&res);
    let Ok(settings) = res else {
        return res.into();
    };
    decoder.rust_decoder.settings = settings;

    let res = decoder.rust_decoder.nth_image(frameIndex);
    decoder.diag.set_from_result(&res);
    decoder.sync_from_rust_decoder();
    res.into()
}

/// # Safety
/// - If both `decoder` and `outTiming` are non-null:
///   - `decoder` must be properly aligned and point to a valid, initialized `avifDecoder` valid for
///     shared reads during this call.
///   - `outTiming` must be properly aligned and point to a valid, initialized `ImageTiming` valid
///     for exclusive writes during this call, without aliasing `decoder`.
/// - Passing null for either `decoder` or `outTiming` safely returns
///   `avifResult::InvalidArgument`.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderNthImageTiming(
    decoder: *const avifDecoder,
    frameIndex: u32,
    outTiming: *mut ImageTiming,
) -> avifResult {
    check_pointer!(decoder);
    check_pointer!(outTiming);
    let image_timing = deref_const!(decoder)
        .rust_decoder
        .nth_image_timing(frameIndex);
    if let Ok(timing) = image_timing {
        *deref_mut!(outTiming) = timing;
    }
    image_timing.into()
}

/// # Safety
/// - If both `decoder` and `outProperties` are non-null:
///   - `decoder` must be properly aligned and point to a valid, initialized `avifDecoder` valid for
///     shared reads during this call.
///   - `outProperties` must be properly aligned and point to a valid, initialized `avifImage`
///     (whose owned planes, if any, were allocated by `crabby_avifAlloc`) valid for exclusive
///     access during this call, without aliasing `decoder`.
/// - Passing null for either `decoder` or `outProperties` safely returns
///   `avifResult::InvalidArgument`.
///
/// Only properties that do not require an allocation are populated. On success, any planes
/// previously allocated in `outProperties` are freed.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderNthImageProperties(
    decoder: *const avifDecoder,
    frameIndex: u32,
    outProperties: *mut avifImage,
) -> avifResult {
    check_pointer!(decoder);
    check_pointer!(outProperties);
    let image = deref_const!(decoder)
        .rust_decoder
        .nth_image_properties(frameIndex);
    if let Ok(image) = &image {
        // SAFETY: `outProperties` is non-null (checked above), and the caller guarantees it points
        // to a valid `avifImage` whose owned plane pointers are valid to free with
        // `crabby_avifImageFreePlanes`.
        unsafe {
            crabby_avifImageFreePlanes(outProperties, avifPlanesFlag::AvifPlanesAll as u32);
        }
        deref_mut!(outProperties).copy_properties_from(image);
    }
    image.into()
}

/// # Safety
/// If `decoder` is non-null, it must be a pointer previously returned by `crabby_avifDecoderCreate`
/// that has not yet been destroyed, and neither `decoder` nor any borrowed pointers into it (such
/// as `decoder->image`) may be accessed after this call. Passing null is safe and is a no-op.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderDestroy(decoder: *mut avifDecoder) {
    check_pointer_or_return!(decoder);
    // SAFETY: `decoder` is non-null and, by caller precondition, was allocated via `Box::into_raw`
    // in `crabby_avifDecoderCreate` and has not yet been freed.
    let _ = unsafe { Box::from_raw(decoder) };
}

fn avif_decoder_read(decoder: &mut avifDecoder, image: &mut avifImage) -> avifResult {
    let res: AvifResult<Settings> = (&*decoder).try_into();
    decoder.diag.set_from_result(&res);
    let Ok(settings) = res else {
        return res.into();
    };
    decoder.rust_decoder.settings = settings;

    let res = decoder.rust_decoder.parse();
    if res.is_err() {
        return res.into();
    }
    let res = decoder.rust_decoder.next_image();
    if res.is_err() {
        return res.into();
    }
    decoder.sync_from_rust_decoder();
    if !decoder.image_object.gainMap.is_null() {
        return avifResult::NotImplemented;
    }
    let src = &decoder.image_object;
    *image = avifImage {
        icc: avifRWData {
            data: src.icc.data,
            size: src.icc.size,
        },
        exif: avifRWData {
            data: src.exif.data,
            size: src.exif.size,
        },
        xmp: avifRWData {
            data: src.xmp.data,
            size: src.xmp.size,
        },
        ..*src
    };
    avifResult::Ok
}

/// # Safety
/// - If both `decoder` and `image` are non-null, they must be properly aligned, non-aliasing
///   pointers to valid, initialized `avifDecoder` and `avifImage` objects valid for exclusive
///   access during this call.
/// - Passing null for either `decoder` or `image` safely returns `avifResult::InvalidArgument`.
///
/// This function does not support decoding gainmaps.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderRead(
    decoder: *mut avifDecoder,
    image: *mut avifImage,
) -> avifResult {
    check_pointer!(decoder);
    check_pointer!(image);
    avif_decoder_read(deref_mut!(decoder), deref_mut!(image))
}

/// # Safety
/// - If `decoder` and `image` are non-null, they must be properly aligned, non-aliasing pointers to
///   valid, initialized `avifDecoder` and `avifImage` objects valid for exclusive access during
///   this call.
/// - If `data` is non-null and `size <= isize::MAX as usize`, `data` must point to at least `size`
///   initialized bytes that remain valid and unmodified until `decoder` is destroyed or its IO is
///   replaced.
/// - Passing null for `decoder`, `image`, or `data` (or `size > isize::MAX as usize`) safely
///   returns `avifResult::InvalidArgument`.
///
/// This function does not support decoding gainmaps.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderReadMemory(
    decoder: *mut avifDecoder,
    image: *mut avifImage,
    data: *const u8,
    size: usize,
) -> avifResult {
    check_pointer!(decoder);
    check_pointer!(image);
    // SAFETY: `decoder` is non-null and valid, and the caller upholds `data` / `size` validity
    // preconditions for `crabby_avifDecoderSetIOMemory`.
    let res = unsafe { crabby_avifDecoderSetIOMemory(decoder, data, size) };
    if res != avifResult::Ok {
        return res;
    }
    avif_decoder_read(deref_mut!(decoder), deref_mut!(image))
}

/// # Safety
/// - If `decoder` and `image` are non-null, they must be properly aligned, non-aliasing pointers to
///   valid, initialized `avifDecoder` and `avifImage` objects valid for exclusive access during
///   this call.
/// - If `filename` is non-null, it must point to a valid, nul-terminated C string valid for reads
///   during this call.
/// - Passing null for `decoder`, `image`, or `filename` safely returns
///   `avifResult::InvalidArgument`.
///
/// This function does not support decoding gainmaps.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderReadFile(
    decoder: *mut avifDecoder,
    image: *mut avifImage,
    filename: *const c_char,
) -> avifResult {
    check_pointer!(decoder);
    check_pointer!(image);
    // SAFETY: `decoder` is non-null and valid, and the caller upholds that a non-null `filename`
    // points to a valid nul-terminated C string for `crabby_avifDecoderSetIOFile`.
    let res = unsafe { crabby_avifDecoderSetIOFile(decoder, filename) };
    if res != avifResult::Ok {
        return res;
    }
    avif_decoder_read(deref_mut!(decoder), deref_mut!(image))
}

/// # Safety
/// If `decoder` is non-null, it must be properly aligned and point to a valid, initialized
/// `avifDecoder` valid for shared reads during this call. Passing null safely returns `AVIF_FALSE`.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderIsKeyframe(
    decoder: *const avifDecoder,
    frameIndex: u32,
) -> avifBool {
    if decoder.is_null() {
        return AVIF_FALSE;
    }
    to_avifBool(deref_const!(decoder).rust_decoder.is_keyframe(frameIndex))
}

/// # Safety
/// If `decoder` is non-null, it must be properly aligned and point to a valid, initialized
/// `avifDecoder` valid for shared reads during this call. Passing null safely returns `0`.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderNearestKeyframe(
    decoder: *const avifDecoder,
    frameIndex: u32,
) -> u32 {
    if decoder.is_null() {
        return 0;
    }
    deref_const!(decoder)
        .rust_decoder
        .nearest_keyframe(frameIndex)
}

/// # Safety
/// If `decoder` is non-null, it must be properly aligned and point to a valid, initialized
/// `avifDecoder` valid for shared reads during this call. Passing null safely returns `0`.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderDecodedRowCount(decoder: *const avifDecoder) -> u32 {
    if decoder.is_null() {
        return 0;
    }
    deref_const!(decoder).rust_decoder.decoded_row_count()
}

#[allow(non_camel_case_types)]
pub type avifExtent = Extent;

/// # Safety
/// - If both `decoder` and `outExtent` are non-null:
///   - `decoder` must be properly aligned and point to a valid, initialized `avifDecoder` valid for
///     shared reads during this call.
///   - `outExtent` must be properly aligned and point to a valid, initialized `avifExtent` valid
///     for exclusive writes during this call, without aliasing `decoder`.
/// - Passing null for either `decoder` or `outExtent` safely returns
///   `avifResult::InvalidArgument`.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderNthImageMaxExtent(
    decoder: *const avifDecoder,
    frameIndex: u32,
    outExtent: *mut avifExtent,
) -> avifResult {
    check_pointer!(decoder);
    check_pointer!(outExtent);
    match deref_const!(decoder)
        .rust_decoder
        .nth_image_max_extent(frameIndex)
    {
        Ok(extent) => {
            *deref_mut!(outExtent) = extent;
            avifResult::Ok
        }
        Err(err) => (&err).into(),
    }
}

/// # Safety
/// - If `input` is non-null, it must be properly aligned and point to a valid, initialized
///   `avifROData` valid for shared reads during this call.
/// - When `input.size > 0` and `input.data` is non-null (`input.size <= isize::MAX as usize`),
///   `input.data` must point to at least `input.size` initialized bytes valid for shared reads
///   during this call.
/// - Passing null for `input` (or an `avifROData` with `size == 0`, null `data`, or
///   `size > isize::MAX as usize`) safely returns `AVIF_FALSE`.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifPeekCompatibleFileType(input: *const avifROData) -> avifBool {
    if input.is_null() {
        return AVIF_FALSE;
    }
    let input = deref_const!(input);
    if input.size == 0 || !check_slice_from_raw_parts_safety(input.data, input.size) {
        return AVIF_FALSE;
    }
    // SAFETY: `input.size > 0`, `input.data` is non-null, `input.size <= isize::MAX`, and the
    // caller guarantees `input.data` points to `input.size` initialized bytes valid for reads.
    let data = unsafe { std::slice::from_raw_parts(input.data, input.size) };
    to_avifBool(Decoder::peek_compatible_file_type(data))
}

/// # Safety
/// If `decoder` is non-null, it must be properly aligned and point to a valid, initialized
/// `avifDecoder` valid for exclusive access during this call. Passing null safely returns
/// `avifResult::InvalidArgument`.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifDecoderReset(decoder: *mut avifDecoder) -> avifResult {
    check_pointer!(decoder);
    avif_decoder_parse(deref_mut!(decoder))
}
