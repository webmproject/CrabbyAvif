// Copyright 2025 Google LLC
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

use super::image::*;
use super::io::*;
use super::types::*;

use crate::encoder::*;
use crate::gainmap::GainMap;
use crate::internal_utils::*;
use crate::*;

use std::ffi::CStr;
use std::os::raw::c_char;

#[repr(C)]
pub struct avifEncoder {
    pub codecChoice: avifCodecChoice,
    pub maxThreads: i32,
    pub speed: i32,
    pub keyframeInterval: i32,
    pub timescale: u64,
    pub repetitionCount: i32,
    pub extraLayerCount: u32,
    pub quality: i32,
    pub qualityAlpha: i32,
    pub minQuantizer: i32,
    pub maxQuantizer: i32,
    pub minQuantizerAlpha: i32,
    pub maxQuantizerAlpha: i32,
    pub tileRowsLog2: i32,
    pub tileColsLog2: i32,
    pub autoTiling: avifBool,
    scalingMode: ScalingMode,
    pub ioStats: crate::decoder::IOStats,
    pub diag: avifDiagnostics,
    pub qualityGainMap: i32,
    /// Used when encoding an image sequence. Specified in seconds since midnight, Jan. 1, 1970 UTC
    /// (the Unix epoch) If set to 0 (the default), now() is used.
    pub creationTime: u64,
    /// Used when encoding an image sequence. Specified in seconds since midnight, Jan. 1, 1970 UTC
    /// (the Unix epoch) If set to 0 (the default), now() is used.
    pub modificationTime: u64,
    rust_encoder: Box<Encoder>,
    rust_encoder_initialized: bool,
    codec_specific_options: Box<CodecSpecificOptions>,
}

impl Default for avifEncoder {
    fn default() -> Self {
        let settings = Settings::default();
        Self {
            codecChoice: avifCodecChoice::Aom,
            maxThreads: settings.threads as _,
            speed: -1,
            keyframeInterval: settings.keyframe_interval,
            timescale: settings.timescale,
            repetitionCount: AVIF_REPETITION_COUNT_INFINITE,
            extraLayerCount: settings.extra_layer_count,
            quality: AVIF_QUALITY_DEFAULT,
            qualityAlpha: AVIF_QUALITY_DEFAULT,
            minQuantizer: AVIF_QUANTIZER_BEST_QUALITY as i32,
            maxQuantizer: AVIF_QUANTIZER_WORST_QUALITY as i32,
            minQuantizerAlpha: AVIF_QUANTIZER_BEST_QUALITY as i32,
            maxQuantizerAlpha: AVIF_QUANTIZER_WORST_QUALITY as i32,
            tileRowsLog2: 0,
            tileColsLog2: 0,
            autoTiling: AVIF_FALSE,
            scalingMode: settings.mutable.scaling_mode,
            ioStats: Default::default(),
            diag: Default::default(),
            qualityGainMap: AVIF_QUALITY_DEFAULT,
            rust_encoder: Default::default(),
            creationTime: 0,
            modificationTime: 0,
            rust_encoder_initialized: false,
            codec_specific_options: Default::default(),
        }
    }
}

fn quality_from_quantizers(minQuantizer: i32, maxQuantizer: i32) -> i32 {
    100 - ((50 * (minQuantizer.clamp(0, 63) + maxQuantizer.clamp(0, 63)) - 50) / 63)
}

impl From<&avifEncoder> for MutableSettings {
    fn from(encoder: &avifEncoder) -> Self {
        Self {
            quality: if encoder.quality == -1 {
                quality_from_quantizers(encoder.minQuantizer, encoder.maxQuantizer)
            } else {
                encoder.quality
            } as f32,
            quality_alpha: if encoder.qualityAlpha == -1 {
                quality_from_quantizers(encoder.minQuantizerAlpha, encoder.maxQuantizerAlpha)
            } else {
                encoder.qualityAlpha
            } as f32,
            quality_gainmap: if encoder.qualityGainMap == -1 {
                quality_from_quantizers(encoder.minQuantizer, encoder.maxQuantizer)
            } else {
                encoder.qualityGainMap
            } as f32,
            tiling_mode: if encoder.autoTiling == AVIF_TRUE {
                TilingMode::Auto
            } else {
                TilingMode::Manual(encoder.tileRowsLog2, encoder.tileColsLog2)
            },
            scaling_mode: encoder.scalingMode,
        }
    }
}

impl From<&avifEncoder> for Settings {
    fn from(encoder: &avifEncoder) -> Self {
        Self {
            codec_choice: match encoder.codecChoice {
                avifCodecChoice::Auto => CodecChoice::Auto,
                avifCodecChoice::Aom => CodecChoice::Aom,
                // Silently treat all other choices the same as Auto.
                _ => CodecChoice::Auto,
            },
            threads: encoder.maxThreads as u32,
            speed: if encoder.speed >= 0 && encoder.speed <= 10 {
                Some(encoder.speed as u32)
            } else {
                None
            },
            header_format: HeaderFormat::default(),
            keyframe_interval: encoder.keyframeInterval,
            timescale: if encoder.timescale == 0 { 1 } else { encoder.timescale },
            repetition_count: RepetitionCount::create_from(encoder.repetitionCount),
            extra_layer_count: encoder.extraLayerCount,
            recipe: Recipe::None,
            force_write_extended_pixi: false,
            creation_time: if encoder.creationTime == 0 {
                None
            } else {
                Some(encoder.creationTime)
            },
            modification_time: if encoder.modificationTime == 0 {
                None
            } else {
                Some(encoder.modificationTime)
            },
            mutable: encoder.into(),
        }
    }
}

impl avifEncoder {
    #[allow(clippy::replace_box)]
    fn initialize_or_update_rust_encoder(&mut self) -> avifResult {
        if self.rust_encoder_initialized {
            // TODO - b/416560730: Validate the immutable settings.
            let mutable_settings: MutableSettings = (&*self).into();
            let res = self.rust_encoder.update_settings(&mutable_settings);
            self.diag.set_from_result(&res);
            res.into()
        } else {
            let settings: Settings = (&*self).into();
            let res = Encoder::create_with_settings(&settings);
            self.diag.set_from_result(&res);
            match res {
                Ok(encoder) => self.rust_encoder = Box::new(encoder),
                Err(err) => return (&err).into(),
            }
            self.rust_encoder_initialized = true;
            // Push any existing codec specific options.
            for (key, value) in self.codec_specific_options.iter() {
                self.rust_encoder.set_codec_specific_option(
                    key.0,
                    key.1.to_string(),
                    value.to_string(),
                );
            }
            self.codec_specific_options.clear();
            avifResult::Ok
        }
    }

    fn add_image(
        &mut self,
        image: &avifImage,
        duration_in_timescales: u64,
        add_image_flags: avifAddImageFlags,
    ) -> avifResult {
        let res = self.initialize_or_update_rust_encoder();
        if res != avifResult::Ok {
            return res;
        }
        let gainmap_res = image.gainmap();
        self.diag.set_from_result(&gainmap_res);
        let gainmap = match gainmap_res {
            Ok(gainmap) => gainmap,
            Err(err) => return (&err).into(),
        };
        let image_res: AvifResult<image::Image> = image.try_into();
        self.diag.set_from_result(&image_res);
        let rust_image = match image_res {
            Ok(rust_image) => rust_image,
            Err(err) => return (&err).into(),
        };
        let res =
            if (add_image_flags & AVIF_ADD_IMAGE_FLAG_SINGLE) != 0 || self.extraLayerCount != 0 {
                match &gainmap {
                    Some(gainmap) => self.rust_encoder.add_image_gainmap(&rust_image, gainmap),
                    None => self.rust_encoder.add_image(&rust_image),
                }
            } else {
                self.rust_encoder
                    .add_image_for_sequence(&rust_image, duration_in_timescales)
            };
        self.diag.set_from_result(&res);
        res.into()
    }

    fn finish(&mut self, output: &mut avifRWData) -> avifResult {
        let res = self.rust_encoder.finish();
        self.diag.set_from_result(&res);
        match res {
            Ok(encoded_data) => {
                let data_ptr = encoded_data.as_ptr();
                let data_len = encoded_data.len();
                // SAFETY: `output` points to a valid mutable `avifRWData` with a well-formed (or
                // default) allocation, and `data_ptr` is valid for reads of `data_len` initialized
                // bytes from `encoded_data` and disjoint from `output`.
                unsafe { crabby_avifRWDataSet(output, data_ptr, data_len) }
            }
            Err(err) => (&err).into(),
        }
    }
}

#[no_mangle]
pub extern "C" fn crabby_avifEncoderCreate() -> *mut avifEncoder {
    Box::into_raw(Box::<avifEncoder>::default())
}

/// # Safety
/// If `encoder` is non-null, it must be a valid pointer obtained from
/// [`crabby_avifEncoderCreate`] that has not yet been destroyed.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifEncoderDestroy(encoder: *mut avifEncoder) {
    check_pointer_or_return!(encoder);
    // SAFETY: `encoder` was verified to be non-null and, by caller precondition, was allocated via
    // `Box::into_raw` in `crabby_avifEncoderCreate` and has not yet been freed.
    let _ = unsafe { Box::from_raw(encoder) };
}

/// # Safety
/// - If `encoder` is non-null, it must point to a valid, mutable [`avifEncoder`] for the duration
///   of this call, not aliased by any other reference.
/// - If `image` is non-null, it must point to a valid [`avifImage`] (including valid plane,
///   metadata, and optional `gainMap` buffers matching its fields) for the duration of this call.
/// - If `output` is non-null, it must point to a valid, mutable [`avifRWData`] (either zeroed /
///   default-initialized or previously managed by `crabby_avifRWData*`) not aliasing `encoder` or
///   `image`.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifEncoderWrite(
    encoder: *mut avifEncoder,
    image: *const avifImage,
    output: *mut avifRWData,
) -> avifResult {
    check_pointer!(encoder);
    check_pointer!(image);
    check_pointer!(output);

    let encoder_ref = deref_mut!(encoder);
    let image_ref = deref_const!(image);
    let res = encoder_ref.add_image(image_ref, 1, AVIF_ADD_IMAGE_FLAG_SINGLE);
    if res != avifResult::Ok {
        return res;
    }
    let output_ref = deref_mut!(output);
    encoder_ref.finish(output_ref)
}

/// # Safety
/// - If `encoder` is non-null, it must point to a valid, mutable [`avifEncoder`] for the duration
///   of this call, not aliased by `image`.
/// - If `image` is non-null, it must point to a valid [`avifImage`] (including valid plane,
///   metadata, and optional `gainMap` buffers matching its fields) for the duration of this call.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifEncoderAddImage(
    encoder: *mut avifEncoder,
    image: *const avifImage,
    durationInTimescales: u64,
    addImageFlags: avifAddImageFlags,
) -> avifResult {
    check_pointer!(encoder);
    check_pointer!(image);

    let encoder_ref = deref_mut!(encoder);
    let image_ref = deref_const!(image);
    encoder_ref.add_image(image_ref, durationInTimescales, addImageFlags)
}

/// # Safety
/// - If `encoder` is non-null, it must point to a valid, mutable [`avifEncoder`] for the duration
///   of this call.
/// - If `cellImages` is non-null and `gridCols * gridRows > 0`, `cellImages` must be properly
///   aligned and point to a contiguous array of `gridCols * gridRows` pointers, where every
///   non-null element points to a valid [`avifImage`] (including valid plane, metadata, and
///   optional `gainMap` buffers) for the duration of this call.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifEncoderAddImageGrid(
    encoder: *mut avifEncoder,
    gridCols: u32,
    gridRows: u32,
    cellImages: *const *const avifImage,
    addImageFlags: avifAddImageFlags,
) -> avifResult {
    check_pointer!(encoder);
    check_pointer!(cellImages);

    let encoder_ref = deref_mut!(encoder);
    if gridCols == 0
        || gridRows == 0
        // If we are not encoding a grid image with multiple layers, AVIF_ADD_IMAGE_FLAG_SINGLE has
        // to be set.
        || ((addImageFlags & AVIF_ADD_IMAGE_FLAG_SINGLE) == 0 && encoder_ref.extraLayerCount == 0)
    {
        return avifResult::InvalidArgument;
    }
    let Some(cell_count_u32) = gridCols.checked_mul(gridRows) else {
        return avifResult::InvalidArgument;
    };
    let Ok(cell_count) = usize_from_u32(cell_count_u32) else {
        return avifResult::InvalidArgument;
    };
    let Some(cell_bytes) = cell_count.checked_mul(std::mem::size_of::<*const avifImage>()) else {
        return avifResult::InvalidArgument;
    };
    if isize_from_usize(cell_bytes).is_err() || !cellImages.is_aligned() {
        return avifResult::InvalidArgument;
    }
    let res = encoder_ref.initialize_or_update_rust_encoder();
    if res != avifResult::Ok {
        return res;
    }

    // SAFETY: `cellImages` was verified to be non-null and aligned, `cell_count >= 1`,
    // `cell_count * size_of::<*const avifImage>() <= isize::MAX`, and by caller precondition
    // `cellImages` points to `cell_count` initialized `*const avifImage` elements valid for this
    // call.
    let image_ptrs: &[*const avifImage] =
        unsafe { std::slice::from_raw_parts(cellImages, cell_count) };

    let res = || -> AvifResult<()> {
        let mut images: Vec<image::Image> = create_vec_exact(cell_count)?;
        let mut gainmaps: Vec<Option<GainMap>> = create_vec_exact(cell_count)?;
        for &image_ptr in image_ptrs {
            if image_ptr.is_null() {
                return AvifError::invalid_argument();
            }
            let cell_image = deref_const!(image_ptr);
            images.try_push(cell_image.try_into()?)?;
            gainmaps.try_push(cell_image.gainmap()?)?;
        }
        let mut image_refs: Vec<&image::Image> = create_vec_exact(images.len())?;
        for image in &images {
            image_refs.try_push(image)?;
        }
        if gainmaps.iter().all(|x| x.is_some()) {
            let mut gainmap_refs: Vec<&GainMap> = create_vec_exact(gainmaps.len())?;
            for gainmap in &gainmaps {
                gainmap_refs.try_push(gainmap.as_ref().ok_or(AvifError::InvalidArgument)?)?;
            }
            encoder_ref.rust_encoder.add_image_gainmap_grid(
                gridCols,
                gridRows,
                &image_refs,
                &gainmap_refs,
            )
        } else if gainmaps.iter().all(|x| x.is_none()) {
            encoder_ref
                .rust_encoder
                .add_image_grid(gridCols, gridRows, &image_refs)
        } else {
            // Some cells had GainMap and some did not. This is invalid.
            AvifError::invalid_argument()
        }
    }();

    encoder_ref.diag.set_from_result(&res);
    res.into()
}

/// # Safety
/// - If `encoder` is non-null, it must point to a valid, mutable [`avifEncoder`] for the duration
///   of this call.
/// - If `output` is non-null, it must point to a valid, mutable [`avifRWData`] (either zeroed /
///   default-initialized or previously managed by `crabby_avifRWData*`) not aliasing `encoder`.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifEncoderFinish(
    encoder: *mut avifEncoder,
    output: *mut avifRWData,
) -> avifResult {
    check_pointer!(encoder);
    check_pointer!(output);

    let encoder_ref = deref_mut!(encoder);
    let output_ref = deref_mut!(output);
    encoder_ref.finish(output_ref)
}

/// # Safety
/// - If `encoder` is non-null, it must point to a valid, mutable [`avifEncoder`] for the duration
///   of this call.
/// - If `key` is non-null, it must point to a valid NUL-terminated C string valid for reads for
///   the duration of this call.
/// - If `value` is non-null, it must point to a valid NUL-terminated C string valid for reads for
///   the duration of this call.
#[no_mangle]
pub unsafe extern "C" fn crabby_avifEncoderSetCodecSpecificOption(
    encoder: *mut avifEncoder,
    key: *const c_char,
    value: *const c_char,
) -> avifResult {
    check_pointer!(encoder);
    check_pointer!(key);
    check_pointer!(value);

    // SAFETY: `key` was verified to be non-null and, by caller precondition, points to a valid
    // NUL-terminated C string for the duration of this call.
    let key_cstr = unsafe { CStr::from_ptr(key) };
    // SAFETY: `value` was verified to be non-null and, by caller precondition, points to a valid
    // NUL-terminated C string for the duration of this call.
    let value_cstr = unsafe { CStr::from_ptr(value) };
    let (Ok(key), Ok(value)) = (key_cstr.to_str(), value_cstr.to_str()) else {
        return avifResult::InvalidArgument;
    };
    let (key, value) = (key.to_owned(), value.to_owned());
    let (key, category) = if let Some(suffix) = key.strip_prefix("c:") {
        (suffix.to_string(), Some(Category::Color))
    } else if let Some(suffix) = key.strip_prefix("color:") {
        (suffix.to_string(), Some(Category::Color))
    } else if let Some(suffix) = key.strip_prefix("a:") {
        (suffix.to_string(), Some(Category::Alpha))
    } else if let Some(suffix) = key.strip_prefix("alpha:") {
        (suffix.to_string(), Some(Category::Alpha))
    } else if let Some(suffix) = key.strip_prefix("g:") {
        (suffix.to_string(), Some(Category::Gainmap))
    } else if let Some(suffix) = key.strip_prefix("gainmap:") {
        (suffix.to_string(), Some(Category::Gainmap))
    } else {
        (key, None)
    };
    let encoder_ref = deref_mut!(encoder);
    if encoder_ref.rust_encoder_initialized {
        encoder_ref
            .rust_encoder
            .set_codec_specific_option(category, key, value);
    } else {
        encoder_ref
            .codec_specific_options
            .insert((category, key), value);
    }
    avifResult::Ok
}

#[cfg(test)]
mod tests {
    #[test]
    fn quality_from_quantizers() {
        // Test the extreme values and middle values.
        assert_eq!(super::quality_from_quantizers(0, 0), 100);
        assert_eq!(super::quality_from_quantizers(23, 32), 58);
        assert_eq!(super::quality_from_quantizers(63, 63), 1);
        // Invalid values should be clamped.
        assert_eq!(super::quality_from_quantizers(-1, -20), 100);
        assert_eq!(super::quality_from_quantizers(100, 200), 1);

        // Test all valid combinations to make sure they return a valid quality value.
        for min_quantizer in 0..63 {
            for max_quantizer in 0..63 {
                let quality = super::quality_from_quantizers(min_quantizer, max_quantizer);
                assert!((0..=100).contains(&quality));
            }
        }
    }
}
