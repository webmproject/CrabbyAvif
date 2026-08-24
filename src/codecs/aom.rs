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

#![allow(non_upper_case_globals)]

use crate::codecs::*;
use crate::encoder::Sample;
use crate::encoder::ScalingMode;
use crate::image::Image;
use crate::image::YuvRange;
use crate::internal_utils::*;
use crate::parser::obu::Av1SequenceHeader;
use crate::utils::IFraction;
use crate::*;

use aom_sys::bindings::*;

use std::cmp;
use std::ffi::CStr;
use std::ffi::CString;
use std::mem::MaybeUninit;

#[derive(Default)]
pub struct Aom {
    encoder: Option<aom_codec_ctx_t>,
    aom_config: Option<aom_codec_enc_cfg>,
    config: Option<EncoderConfig>,
    current_layer: u32,
    previous_frame_used_tune_iq: bool,
}

fn aom_format(image: &Image, category: Category) -> AvifResult<aom_img_fmt_t> {
    let format = match category {
        Category::Alpha => aom_img_fmt_AOM_IMG_FMT_I420,
        _ => match image.yuv_format {
            PixelFormat::Yuv420 | PixelFormat::Yuv400 => aom_img_fmt_AOM_IMG_FMT_I420,
            PixelFormat::Yuv422 => aom_img_fmt_AOM_IMG_FMT_I422,
            PixelFormat::Yuv444 => aom_img_fmt_AOM_IMG_FMT_I444,
            _ => return AvifError::invalid_argument(),
        },
    };
    Ok(if image.depth > 8 { format | AOM_IMG_FMT_HIGHBITDEPTH } else { format })
}

fn aom_bps(format: aom_img_fmt_t) -> i32 {
    match format {
        aom_img_fmt_AOM_IMG_FMT_I420 => 12,
        aom_img_fmt_AOM_IMG_FMT_I422 => 16,
        aom_img_fmt_AOM_IMG_FMT_I444 => 24,
        aom_img_fmt_AOM_IMG_FMT_I42016 => 24,
        aom_img_fmt_AOM_IMG_FMT_I42216 => 32,
        aom_img_fmt_AOM_IMG_FMT_I44416 => 48,
        _ => 16,
    }
}

fn aom_seq_profile(image: &Image, category: Category) -> AvifResult<u32> {
    if image.depth == 12 {
        // 12 bit is always profile 2.
        return Ok(2);
    }
    if category == Category::Alpha {
        // Alpha is monochrome, so it is always profile 0.
        return Ok(0);
    }
    match image.yuv_format {
        PixelFormat::Yuv420 | PixelFormat::Yuv400 => Ok(0),
        PixelFormat::Yuv422 => Ok(2),
        PixelFormat::Yuv444 => Ok(1),
        _ => AvifError::invalid_argument(),
    }
}

fn get_aom_scaling_mode_1d(mut fraction: IFraction) -> AvifResult<aom_scaling_mode_1d> {
    fraction.is_valid()?;
    fraction.simplify();
    Ok(match fraction {
        IFraction(1, 1) => aom_scaling_mode_1d_AOME_NORMAL,
        IFraction(1, 2) => aom_scaling_mode_1d_AOME_ONETWO,
        IFraction(1, 3) => aom_scaling_mode_1d_AOME_ONETHREE,
        IFraction(1, 4) => aom_scaling_mode_1d_AOME_ONEFOUR,
        IFraction(1, 8) => aom_scaling_mode_1d_AOME_ONEEIGHT,
        IFraction(2, 3) => aom_scaling_mode_1d_AOME_TWOTHREE,
        IFraction(3, 4) => aom_scaling_mode_1d_AOME_THREEFOUR,
        IFraction(3, 5) => aom_scaling_mode_1d_AOME_THREEFIVE,
        IFraction(4, 5) => aom_scaling_mode_1d_AOME_FOURFIVE,
        _ => return AvifError::not_implemented(),
    })
}

fn aom_scaling_mode(scaling_mode: &ScalingMode) -> AvifResult<aom_scaling_mode_t> {
    Ok(aom_scaling_mode_t {
        h_scaling_mode: get_aom_scaling_mode_1d(scaling_mode.horizontal)?,
        v_scaling_mode: get_aom_scaling_mode_1d(scaling_mode.vertical)?,
    })
}

macro_rules! codec_control {
    ($self: expr, $key: expr, $value: expr) => {
        // # Safety: Calling a C function with valid parameters.
        if unsafe { aom_codec_control($self.encoder.unwrap_mut() as *mut _, $key as _, $value) }
            != aom_codec_err_t_AOM_CODEC_OK
        {
            return AvifError::unknown_error(format!(
                "aom_codec_control failed: {}",
                $self.error_string()
            ));
        }
    };
}

macro_rules! c_str {
    ($var: ident, $var_tmp:ident, $str:expr) => {
        let $var_tmp = CString::new($str).unwrap();
        let $var = $var_tmp.as_ptr();
    };
}

// Quality (q) to quantizer (qp) formula for tune=iq (Image Quality), expressed as a look-up table for more clarity.
// The formula below is a piecewise linear function. Each segment was empirically selected to correct for the
// non-linear bitrate increase from encoding content with tune=iq relative to tune=ssim with the same qp.
//
// | Quality | Quantizer                          | Step size |
// |---------|------------------------------------|-----------|
// |  0 -  6 | 63 - floor(quality / 3)            |         3 |
// |  7 - 28 | 61 - round((quality - 7) / 2)      |         2 |
// | 29 - 53 | 50 - round((quality - 29) * 3 / 5) |      1.66 |
// | 54 - 99 | 35 - round((quality - 54) * 3 / 4) |      1.33 |
// |     100 | 0 (lossless)                       |         1 |
//
// The formula has these properties, in addition to the general conversion formula properties described in avif.h:
// - Encoding and decoding time with tune=iq are closer to tune=ssim's at a given quality level, with an
//   overall smaller (but still predictable) file size and a similar to better quality
// - The qp of tune=ssim <= qp of tune=iq for all quality values
// - Quality 60 (the default in avifenc) = qp 30
// - The step size of the quantizers monotonically decreases as quality increases (from 3 to 1)
//
// The x axis of the table represents the ones digit, while the y axis represents the tens digit
// of the q value [0-100], which is then mapped to a qp value [0-63].
#[rustfmt::skip]
static TUNE_IQ_QUALITY_TO_QUANTIZER: [i32; 101] = [
// 1s digit: *0  *1  *2  *3  *4  *5  *6  *7  *8  *9     10s digit:
             63, 63, 63, 62, 62, 62, 61, 61, 60, 60, // 0*
             59, 59, 58, 58, 57, 57, 56, 56, 55, 55, // 1*
             54, 54, 53, 53, 52, 52, 51, 51, 50, 50, // 2*
             49, 49, 48, 48, 47, 46, 46, 45, 45, 44, // 3*
             43, 43, 42, 42, 41, 40, 40, 39, 39, 38, // 4*
             37, 37, 36, 36, 35, 34, 33, 33, 32, 31, // 5*
             30, 30, 29, 28, 27, 27, 26, 25, 24, 24, // 6*
             23, 22, 21, 21, 20, 19, 18, 18, 17, 16, // 7*
             15, 15, 14, 13, 12, 12, 11, 10,  9,  9, // 8*
              8,  7,  6,  6,  5,  4,  3,  3,  2,  1, // 9*
              0  // quality 100
];

fn aom_quality_to_quantizer(quality: i32, is_tune_iq: bool) -> i32 {
    if is_tune_iq {
        TUNE_IQ_QUALITY_TO_QUANTIZER[quality as usize]
    } else {
        ((100 - quality) * 63 + 50) / 100
    }
}

fn aom_min_max_quantizers(quantizer: i32) -> (u32, u32) {
    if quantizer == 0 {
        (0, 0)
    } else {
        (
            cmp::max(quantizer - 4, 0) as u32,
            cmp::min(quantizer + 4, 63) as u32,
        )
    }
}

fn add_aom_pkt_to_output_samples(
    pkt: &aom_codec_cx_pkt,
    output_samples: &mut Vec<Sample>,
) -> AvifResult<bool> {
    if pkt.kind != aom_codec_cx_pkt_kind_AOM_CODEC_CX_FRAME_PKT {
        return Ok(false);
    }
    // # Safety: buf and sz are guaranteed to be valid as per libaom API contract. So
    // it is safe to construct a slice from it.
    let encoded_data =
        unsafe { std::slice::from_raw_parts(pkt.data.frame.buf as *const u8, pkt.data.frame.sz) };
    // # Safety: pkt.data is a union. pkt.kind == AOM_CODEC_CX_FRAME_PKT guarantees
    // that pkt.data.frame is the active field of the union (per libaom API contract).
    // So this access is safe.
    let sync = (unsafe { pkt.data.frame.flags } & AOM_FRAME_IS_KEY) != 0;
    output_samples.try_push(Sample::create_from(
        encoded_data,
        0..encoded_data.len(),
        sync,
    )?)?;
    Ok(true)
}

impl Encoder for Aom {
    fn encode_image(
        &mut self,
        image: &Image,
        category: Category,
        config: &EncoderConfig,
        output_samples: &mut Vec<Sample>,
    ) -> AvifResult<()> {
        let aom_usage = if config.is_single_image {
            AOM_USAGE_ALL_INTRA
        } else if config.speed.unwrap_or(0) >= 7 {
            AOM_USAGE_REALTIME
        } else {
            AOM_USAGE_GOOD_QUALITY
        };

        let quality = config.quality.clamp(0.0, 100.0) as i32;

        // If true, override libaom's default tune option.
        let mut use_crabbyavif_default_tune_metric = false;
        // Meaningless unless use_crabbyavif_default_tune_metric.
        let mut crabbyavif_default_tune_metric = aom_tune_metric_AOM_TUNE_PSNR;
        // True if CrabbyAvif knows that tune=iq is used, either set by
        // CrabbyAvif by default, or set by the user explicitly. False
        // otherwise (including if libaom uses tune=iq by default, which is not
        // the case as of v3.14.1 and earlier versions).
        let use_tune_iq: bool;
        // Check if codec-specific options for libaom contain a tune metric
        // setting. If there are multiple "tune" options specified, honor the
        // last one.
        let options = config.codec_specific_options(category);
        if let Some((_, value)) = options.iter().rfind(|&(k, _)| k == "tune") {
            // Check if the tune metric setting is AOM_TUNE_IQ. For consistent
            // behavior, handle both cases where tune was either specified as a
            // string (e.g. tune=iq) or as an enum value (e.g. tune=10).
            use_tune_iq = matches!(value.as_str(), "iq" | "10");
        } else if self.encoder.is_none() {
            // CrabbyAvif only needs to set the default tune metric for the
            // first frame, because libaom will persist that setting until
            // explicitly changed.

            if quality == 100 {
                // AOM_TUNE_IQ is not libaom's default tune option as of
                // v3.14.1. Even if it was, it does not matter for lossless.
                use_tune_iq = false;
            } else {
                use_crabbyavif_default_tune_metric = true;
                crabbyavif_default_tune_metric = if category == Category::Alpha {
                    // Minimize ringing for alpha.
                    aom_tune_metric_AOM_TUNE_PSNR
                } else if image.matrix_coefficients != MatrixCoefficients::Identity
                    && (config.is_single_image || config.extra_layer_count > 0)
                {
                    // AOM_TUNE_IQ has been tuned for the YCbCr family of color
                    // spaces, and is favored for its low perceptual
                    // distortion. AOM_TUNE_IQ partially generalizes to, and
                    // benefits from other "YUV-like" spaces (e.g. YCgCo and
                    // ICtCp) including monochrome (luma only).
                    //
                    // AOM_TUNE_IQ supports all-intra, good-quality and
                    // realtime modes (for single and layered images).
                    aom_tune_metric_AOM_TUNE_IQ
                } else {
                    aom_tune_metric_AOM_TUNE_SSIM
                };
                use_tune_iq = crabbyavif_default_tune_metric == aom_tune_metric_AOM_TUNE_IQ;
            }
        } else {
            // The tune option persists across frames in libaom until
            // explicitly set to another value.
            use_tune_iq = self.previous_frame_used_tune_iq;
        }
        // Remember the current tune option for the next frame.
        self.previous_frame_used_tune_iq = use_tune_iq;

        let quantizer = aom_quality_to_quantizer(quality, use_tune_iq);

        if self.encoder.is_none() {
            // Require libaom v3.14.0 or later.
            // # Safety: aom_codec_version() has no safety prerequisites.
            let aom_version = unsafe { aom_codec_version() };
            // aom_codec.h says: aom_codec_version() == (major<<16 | minor<<8 | patch)
            if aom_version < (3 << 16) | (14 << 8) {
                return AvifError::unknown_error(format!(
                    "{} is older than v3.14.0",
                    Aom::version()
                ));
            }
            // # Safety: Calling a C function.
            let encoder_iface = unsafe { aom_codec_av1_cx() };
            let mut cfg_uninit: MaybeUninit<aom_codec_enc_cfg> = MaybeUninit::uninit();
            // # Safety: Calling a C function with valid parameters.
            let err = unsafe {
                aom_codec_enc_config_default(encoder_iface, cfg_uninit.as_mut_ptr(), aom_usage)
            };
            if err != aom_codec_err_t_AOM_CODEC_OK {
                return AvifError::unknown_error(format!(
                    "aom_codec_enc_config_default failed. err: {err}"
                ));
            }
            // # Safety: cfg_uninit was initialized in the C function call above.
            let mut aom_config = unsafe { cfg_uninit.assume_init() };
            aom_config.rc_end_usage = match aom_usage {
                AOM_USAGE_REALTIME => aom_rc_mode_AOM_CBR,
                _ => aom_rc_mode_AOM_Q,
            };
            aom_config.g_profile = aom_seq_profile(image, category)?;
            aom_config.g_bit_depth = image.depth as _;
            aom_config.g_input_bit_depth = image.depth as _;
            aom_config.g_w = image.width;
            aom_config.g_h = image.height;

            if config.is_single_image {
                aom_config.g_limit = 1;
            }
            if aom_usage != AOM_USAGE_ALL_INTRA && config.keyframe_interval > 0 {
                aom_config.kf_max_dist = config.keyframe_interval as u32;
            }
            if config.disable_lagged_output {
                aom_config.g_lag_in_frames = 0;
            }
            if config.extra_layer_count > 0 {
                // For layered image, disable lagged encoding to always get output
                // frame for each input frame.
                aom_config.g_lag_in_frames = 0;
                aom_config.g_limit = config.extra_layer_count + 1;
                // Disable QP offsets, so CQ level = frame QP for every frame.
                if aom_config.rc_end_usage == aom_rc_mode_AOM_Q {
                    aom_config.use_fixed_qp_offsets = 2;
                }
            }
            if config.threads > 1 {
                aom_config.g_threads = cmp::min(config.threads, 64);
            }

            // Encode alpha as 4:0:0.
            // AVIF specification, Section 4 "Auxiliary Image Items and Sequences":
            //   The mono_chrome field in the Sequence Header OBU shall be set to 1
            aom_config.monochrome =
                (category == Category::Alpha || image.yuv_format == PixelFormat::Yuv400).into();
            // end-usage is the only codec specific option that has to be set before initializing
            // the libaom encoder
            if let Some(value) = config.codec_specific_option(category, String::from("end-usage")) {
                aom_config.rc_end_usage = if let Ok(value) = value.parse() {
                    if value == aom_rc_mode_AOM_VBR
                        || value == aom_rc_mode_AOM_CBR
                        || value == aom_rc_mode_AOM_CQ
                        || value == aom_rc_mode_AOM_Q
                    {
                        value
                    } else {
                        return AvifError::invalid_argument();
                    }
                } else {
                    match value.as_str() {
                        "vbr" => aom_rc_mode_AOM_VBR,
                        "cbr" => aom_rc_mode_AOM_CBR,
                        "cq" => aom_rc_mode_AOM_CQ,
                        "q" => aom_rc_mode_AOM_Q,
                        _ => return AvifError::invalid_argument(),
                    }
                };
            }
            if aom_config.rc_end_usage == aom_rc_mode_AOM_VBR
                || aom_config.rc_end_usage == aom_rc_mode_AOM_CBR
            {
                // cq-level is unused in these modes, so set the min and max quantizer instead.
                (aom_config.rc_min_quantizer, aom_config.rc_max_quantizer) =
                    aom_min_max_quantizers(quantizer);
            }

            let mut encoder_uninit: MaybeUninit<aom_codec_ctx_t> = MaybeUninit::uninit();
            // # Safety: Calling a C function with valid parameters.
            let err = unsafe {
                aom_codec_enc_init_ver(
                    encoder_uninit.as_mut_ptr(),
                    encoder_iface,
                    &aom_config as *const _,
                    if image.depth > 8 { AOM_CODEC_USE_HIGHBITDEPTH } else { 0 } as _,
                    AOM_ENCODER_ABI_VERSION as _,
                )
            };
            if err != aom_codec_err_t_AOM_CODEC_OK {
                return AvifError::unknown_error(format!(
                    "aom_codec_enc_init failed: {}",
                    self.error_string()
                ));
            }
            // # Safety: encoder_uninit was initialized in the C function call above.
            self.encoder = Some(unsafe { encoder_uninit.assume_init() });

            if aom_config.rc_end_usage == aom_rc_mode_AOM_CQ
                || aom_config.rc_end_usage == aom_rc_mode_AOM_Q
            {
                codec_control!(self, aome_enc_control_id_AOME_SET_CQ_LEVEL, quantizer);
            }
            if quantizer == 0 {
                codec_control!(self, aome_enc_control_id_AV1E_SET_LOSSLESS, 1);
            }
            if config.tile_rows_log2 != 0 {
                codec_control!(
                    self,
                    aome_enc_control_id_AV1E_SET_TILE_ROWS,
                    config.tile_rows_log2
                );
            }
            if config.tile_columns_log2 != 0 {
                codec_control!(
                    self,
                    aome_enc_control_id_AV1E_SET_TILE_COLUMNS,
                    config.tile_columns_log2
                );
            }
            if config.extra_layer_count > 0 {
                codec_control!(
                    self,
                    aome_enc_control_id_AOME_SET_NUMBER_SPATIAL_LAYERS,
                    config.extra_layer_count + 1
                );
            }
            if let Some(speed) = config.speed {
                codec_control!(
                    self,
                    aome_enc_control_id_AOME_SET_CPUUSED,
                    cmp::min(speed, 9)
                );
            }
            match category {
                Category::Alpha => {
                    // AVIF specification, Section 4 "Auxiliary Image Items and Sequences":
                    //   The color_range field in the Sequence Header OBU shall be set to 1.
                    codec_control!(
                        self,
                        aome_enc_control_id_AV1E_SET_COLOR_RANGE,
                        aom_color_range_AOM_CR_FULL_RANGE
                    )
                    // Keep the default AOM_CSP_UNKNOWN value.

                    // CICP (CP/TC/MC) does not apply to the alpha auxiliary image.
                    // Keep default Unspecified (2) colour primaries, transfer characteristics,
                    // and matrix coefficients.
                }
                _ => {
                    // libaom's defaults are AOM_CSP_UNKNOWN and 0 (studio/limited range).
                    // Call aom_codec_control() only if the values are not the defaults.
                    if image.chroma_sample_position != ChromaSamplePosition::Unknown {
                        codec_control!(
                            self,
                            aome_enc_control_id_AV1E_SET_CHROMA_SAMPLE_POSITION,
                            image.chroma_sample_position as i32
                        );
                    }
                    // AV1-ISOBMFF specification, Section 2.3.4:
                    //   The value of full_range_flag in the 'colr' box SHALL match the color_range
                    //   flag in the Sequence Header OBU.
                    if image.yuv_range != YuvRange::Limited {
                        codec_control!(
                            self,
                            aome_enc_control_id_AV1E_SET_COLOR_RANGE,
                            aom_color_range_AOM_CR_FULL_RANGE
                        );
                    }
                    // Section 2.3.4 of AV1-ISOBMFF says 'colr' with 'nclx' should be present and
                    // shall match CICP values in the Sequence Header OBU, unless the latter has
                    // 2/2/2 (Unspecified). So set CICP values to 2/2/2 (Unspecified) in the
                    // Sequence Header OBU for simplicity. libaom's defaults are
                    // AOM_CICP_CP_UNSPECIFIED, AOM_CICP_TC_UNSPECIFIED, and
                    // AOM_CICP_MC_UNSPECIFIED. No need to call aom_codec_control().
                }
            }
            if aom_config.g_usage == AOM_USAGE_ALL_INTRA {
                codec_control!(
                    self,
                    aome_enc_control_id_AV1E_SET_SKIP_POSTPROC_FILTERING,
                    1
                );
            }
            if use_crabbyavif_default_tune_metric {
                codec_control!(
                    self,
                    aome_enc_control_id_AOME_SET_TUNING,
                    crabbyavif_default_tune_metric
                );
            }
            let codec_specific_options = config.codec_specific_options(category);
            for (key, value) in &codec_specific_options {
                if key == "end-usage" {
                    // This key is already processed before initialization of the encoder.
                    continue;
                }
                c_str!(key_str, key_str_tmp, key.clone());
                c_str!(value_str, value_str_tmp, value.clone());
                // # Safety: Calling a C function with valid parameters.
                if unsafe {
                    aom_codec_set_option(self.encoder.unwrap_mut() as *mut _, key_str, value_str)
                } != aom_codec_err_t_AOM_CODEC_OK
                {
                    return AvifError::unknown_error(format!(
                        "Unable to set codec specific option: {key} to {value}: {}",
                        self.error_string()
                    ));
                }
            }
            if image.depth == 12 {
                // libaom may produce integer overflows with 12-bit input when loop restoration is
                // enabled. See crbug.com/aomedia/42302587.
                codec_control!(self, aome_enc_control_id_AV1E_SET_ENABLE_RESTORATION, 0);
            }

            self.aom_config = Some(aom_config);
            self.config = Some(config.clone());
        } else if self.config.unwrap_ref() != config {
            let aom_config = self.aom_config.unwrap_mut();
            if aom_config.g_w != image.width || aom_config.g_h != image.height {
                // Dimension changes aren't allowed.
                return AvifError::not_implemented();
            }
            let last_config = self.config.unwrap_ref();
            if last_config.quality != config.quality {
                if aom_config.rc_end_usage == aom_rc_mode_AOM_VBR
                    || aom_config.rc_end_usage == aom_rc_mode_AOM_CBR
                {
                    (aom_config.rc_min_quantizer, aom_config.rc_max_quantizer) =
                        aom_min_max_quantizers(quantizer);
                    // # Safety: Calling a C function with valid parameters.
                    let err = unsafe {
                        aom_codec_enc_config_set(
                            self.encoder.unwrap_mut() as *mut _,
                            self.aom_config.unwrap_ref() as *const _,
                        )
                    };
                    if err != aom_codec_err_t_AOM_CODEC_OK {
                        return AvifError::unknown_error(format!(
                            "aom_codec_enc_config_set failed: {}",
                            self.error_string()
                        ));
                    }
                } else if aom_config.rc_end_usage == aom_rc_mode_AOM_CQ
                    || aom_config.rc_end_usage == aom_rc_mode_AOM_Q
                {
                    codec_control!(self, aome_enc_control_id_AOME_SET_CQ_LEVEL, quantizer);
                }
                codec_control!(
                    self,
                    aome_enc_control_id_AV1E_SET_LOSSLESS,
                    if quantizer == 0 { 1 } else { 0 }
                );
            }
            if last_config.tile_rows_log2 != config.tile_rows_log2 {
                codec_control!(
                    self,
                    aome_enc_control_id_AV1E_SET_TILE_ROWS,
                    config.tile_rows_log2
                );
            }
            if last_config.tile_columns_log2 != config.tile_columns_log2 {
                codec_control!(
                    self,
                    aome_enc_control_id_AV1E_SET_TILE_COLUMNS,
                    config.tile_columns_log2
                );
            }
            self.config = Some(config.clone());
        }
        if self.current_layer > config.extra_layer_count {
            return AvifError::invalid_argument();
        }
        if config.extra_layer_count > 0 {
            codec_control!(
                self,
                aome_enc_control_id_AOME_SET_SPATIAL_LAYER_ID,
                self.current_layer
            );
        }
        let scaling_mode = aom_scaling_mode(&self.config.unwrap_ref().scaling_mode)?;
        if scaling_mode.h_scaling_mode != aom_scaling_mode_1d_AOME_NORMAL
            || scaling_mode.v_scaling_mode != aom_scaling_mode_1d_AOME_NORMAL
        {
            codec_control!(
                self,
                aome_enc_control_id_AOME_SET_SCALEMODE,
                &scaling_mode as *const _
            );
        }
        // # Safety: Zero initializing a C-struct. This is safe because this is the same usage
        // pattern as the equivalent C-code. The relevant fields are populated in the lines below.
        let mut aom_image: aom_image_t = unsafe { std::mem::zeroed() };
        aom_image.fmt = aom_format(image, category)?;
        aom_image.bit_depth = if image.depth > 8 { 16 } else { 8 };
        aom_image.w = image.width;
        aom_image.h = image.height;
        aom_image.d_w = image.width;
        aom_image.d_h = image.height;
        aom_image.bps = aom_bps(aom_image.fmt);
        aom_image.x_chroma_shift = image.yuv_format.chroma_shift_x().0;
        aom_image.y_chroma_shift = image.yuv_format.chroma_shift_y();
        match category {
            Category::Alpha => {
                aom_image.x_chroma_shift = 1;
                aom_image.y_chroma_shift = 1;
                aom_image.planes[0] = image.planes[3].unwrap_ref().ptr_generic() as *mut _;
                aom_image.stride[0] = image.row_bytes[3] as i32;
            }
            _ => {
                if image.yuv_format == PixelFormat::Yuv400 {
                    aom_image.x_chroma_shift = 1;
                    aom_image.y_chroma_shift = 1;
                    aom_image.planes[0] = image.planes[0].unwrap_ref().ptr_generic() as *mut _;
                    aom_image.stride[0] = image.row_bytes[0] as i32;
                } else {
                    for i in 0..=2 {
                        aom_image.planes[i] = image.planes[i].unwrap_ref().ptr_generic() as *mut _;
                        aom_image.stride[i] = image.row_bytes[i] as i32;
                    }
                }
            }
        }
        // TODO: b/392112497 - force keyframes when necessary.
        let mut encode_flags = 0i64;
        if self.current_layer > 0 {
            encode_flags |= AOM_EFLAG_NO_REF_GF as i64
                | AOM_EFLAG_NO_REF_ARF as i64
                | AOM_EFLAG_NO_REF_BWD as i64
                | AOM_EFLAG_NO_REF_ARF2 as i64
                | AOM_EFLAG_NO_UPD_GF as i64
                | AOM_EFLAG_NO_UPD_ARF as i64;
        }
        // # Safety: Calling a C function with valid parameters.
        let err = unsafe {
            aom_codec_encode(
                self.encoder.unwrap_mut() as *mut _,
                &aom_image as *const _,
                0,
                1,
                encode_flags as _,
            )
        };
        if err != aom_codec_err_t_AOM_CODEC_OK {
            return AvifError::unknown_error(format!(
                "aom_codec_encode failed: {}",
                self.error_string()
            ));
        }
        let mut iter: aom_codec_iter_t = std::ptr::null_mut();
        loop {
            // # Safety: Calling a C function with valid parameters.
            let pkt = unsafe {
                aom_codec_get_cx_data(self.encoder.unwrap_mut() as *mut _, &mut iter as *mut _)
            };
            if pkt.is_null() {
                break;
            }
            // # Safety: pkt is guaranteed to be valid and not null (libaom API contract).
            let pkt = unsafe { *pkt };
            add_aom_pkt_to_output_samples(&pkt, output_samples)?;
        }
        if config.is_single_image
            || (config.extra_layer_count > 0 && config.extra_layer_count == self.current_layer)
        {
            self.finish(output_samples)?;
            // # Safety: Calling a C function with valid parameters.
            unsafe {
                aom_codec_destroy(self.encoder.unwrap_mut() as *mut _);
            }
            self.encoder = None;
        }
        if config.extra_layer_count > 0 {
            self.current_layer += 1;
        }
        Ok(())
    }

    fn finish(&mut self, output_samples: &mut Vec<crate::encoder::Sample>) -> AvifResult<()> {
        if self.encoder.is_none() {
            return Ok(());
        }
        loop {
            // Flush the encoder.
            // # Safety: Calling a C function with valid parameters.
            let err = unsafe {
                aom_codec_encode(
                    self.encoder.unwrap_mut() as *mut _,
                    std::ptr::null(),
                    0,
                    1,
                    0,
                )
            };
            if err != aom_codec_err_t_AOM_CODEC_OK {
                return AvifError::unknown_error(format!(
                    "aom_codec_encode with null img failed: {}",
                    self.error_string()
                ));
            }
            let mut got_packet = false;
            let mut iter: aom_codec_iter_t = std::ptr::null_mut();
            loop {
                // # Safety: Calling a C function with valid parameters.
                let pkt = unsafe {
                    aom_codec_get_cx_data(self.encoder.unwrap_mut() as *mut _, &mut iter as *mut _)
                };
                if pkt.is_null() {
                    break;
                }
                // # Safety: pkt is guaranteed to be valid and not null (libaom API contract).
                let pkt = unsafe { *pkt };
                got_packet = add_aom_pkt_to_output_samples(&pkt, output_samples)?;
            }
            if !got_packet {
                break;
            }
        }
        Ok(())
    }

    fn get_codec_config(
        &self,
        _image: &Image,
        _is_single_image: bool,
        _is_lossless: bool,
        output_samples: &[crate::encoder::Sample],
    ) -> AvifResult<CodecConfiguration> {
        // Harvest codec configuration from AV1 sequence header.
        Ok(CodecConfiguration::Av1(
            Av1SequenceHeader::parse_from_obus(output_samples[0].sample_data())?.config,
        ))
    }
}

impl Drop for Aom {
    fn drop(&mut self) {
        if self.encoder.is_some() {
            // # Safety: Calling a C function with valid parameters.
            unsafe {
                aom_codec_destroy(self.encoder.unwrap_mut() as *mut _);
            }
        }
    }
}

impl Aom {
    pub(crate) fn version() -> String {
        let version = match unsafe { CStr::from_ptr(aom_codec_version_str()) }.to_str() {
            Ok(s) => s.to_owned(),
            Err(_) => String::new(),
        };
        format!("aom: {version}")
    }

    fn error_string(&self) -> String {
        let ctx = self.encoder.unwrap_ref();
        // # Safety: Calling aom_codec_error() with valid parameters is guaranteed to return a valid char* pointer.
        let err = unsafe { CStr::from_ptr(aom_codec_error(ctx)).to_string_lossy() };
        // # Safety: Calling aom_codec_error_detail() with valid parameters is guaranteed to return a null pointer or a valid char* pointer.
        let detail_ptr = unsafe { aom_codec_error_detail(ctx) };
        if detail_ptr.is_null() {
            format!("{err}: no error detail")
        } else {
            // # Safety: detail_ptr is a valid char* pointer.
            let detail = unsafe { CStr::from_ptr(detail_ptr).to_string_lossy() };
            format!("{err}: {detail}")
        }
    }
}
