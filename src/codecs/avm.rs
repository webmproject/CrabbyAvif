// Copyright 2026 Google LLC
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
use crate::parser::mp4box::Av2CodecConfiguration;
#[cfg(test)]
use crate::reformat::rgb::Format;
use crate::utils::pixels::Pixels;
use crate::utils::IFraction;
use crate::*;

use avm_sys::bindings::*;

use std::ffi::CStr;
use std::ffi::CString;
use std::mem::MaybeUninit;
use std::slice;

// OBU types
pub(crate) const AV2_OBU_SEQUENCE_HEADER: u8 = OBU_TYPE_OBU_SEQUENCE_HEADER;
pub(crate) const AV2_OBU_TEMPORAL_DELIMITER: u8 = OBU_TYPE_OBU_TEMPORAL_DELIMITER;
pub(crate) const AV2_OBU_CLOSED_LOOP_KEY: u8 = OBU_TYPE_OBU_CLOSED_LOOP_KEY;
pub(crate) const AV2_OBU_OPEN_LOOP_KEY: u8 = OBU_TYPE_OBU_OPEN_LOOP_KEY;
pub(crate) const AV2_OBU_LEADING_TILE_GROUP: u8 = OBU_TYPE_OBU_LEADING_TILE_GROUP;
pub(crate) const AV2_OBU_REGULAR_TILE_GROUP: u8 = OBU_TYPE_OBU_REGULAR_TILE_GROUP;
pub(crate) const AV2_OBU_SWITCH: u8 = OBU_TYPE_OBU_SWITCH;
pub(crate) const AV2_OBU_LEADING_TIP: u8 = OBU_TYPE_OBU_LEADING_TIP;
pub(crate) const AV2_OBU_REGULAR_TIP: u8 = OBU_TYPE_OBU_REGULAR_TIP;
pub(crate) const AV2_OBU_LAYER_CONFIGURATION_RECORD: u8 = OBU_TYPE_OBU_LAYER_CONFIGURATION_RECORD;
pub(crate) const AV2_OBU_BRIDGE_FRAME: u8 = OBU_TYPE_OBU_BRIDGE_FRAME;
pub(crate) const AV2_OBU_CONTENT_INTERPRETATION: u8 = OBU_TYPE_OBU_CONTENT_INTERPRETATION;
pub(crate) const AV2_OBU_PADDING: u8 = OBU_TYPE_OBU_PADDING;

// Chroma sample position
pub(crate) const AV2_CSP_LEFT: u32 = avm_chroma_sample_position_AVM_CSP_LEFT;
pub(crate) const AV2_CSP_CENTER: u32 = avm_chroma_sample_position_AVM_CSP_CENTER;
pub(crate) const AV2_CSP_TOPLEFT: u32 = avm_chroma_sample_position_AVM_CSP_TOPLEFT;

#[derive(Default)]
pub struct Avm {
    // Encoder or decoder.
    context: Option<avm_codec_ctx_t>,

    // Encoder.
    encoder_avm_config: Option<avm_codec_enc_cfg>,
    encoder_config: Option<EncoderConfig>,
    current_layer: u32,
    has_alpha_layer: bool,

    // Decoder.
    decoder_config: Option<DecoderConfig>,
    decoder_iter: avm_codec_iter_t,
    image: *mut avm_image_t,
}

// Functions mapping from CrabbyAvif structures to AV2 or libavm constants.

fn avm_format(image: &Image, category: Category) -> AvifResult<avm_img_fmt_t> {
    let format = match category {
        Category::Alpha => avm_img_fmt_AVM_IMG_FMT_I420,
        _ => match image.yuv_format {
            PixelFormat::Yuv420 | PixelFormat::Yuv400 => avm_img_fmt_AVM_IMG_FMT_I420,
            PixelFormat::Yuv422 => avm_img_fmt_AVM_IMG_FMT_I422,
            PixelFormat::Yuv444 => avm_img_fmt_AVM_IMG_FMT_I444,
            _ => return AvifError::invalid_argument(),
        },
    };
    Ok(if image.depth > 8 { format | AVM_IMG_FMT_HIGHBITDEPTH } else { format })
}

struct Av2Profile {
    seq_profile: u32,
    depths: &'static [u8],
    formats: &'static [PixelFormat],
}

fn av2_profile(image: &Image, category: Category) -> AvifResult<&Av2Profile> {
    // Based on AV2 spec Table A.1: AV2 profile definitions.
    // https://av2.aomedia.org/v1.0.0/index.html#table-profile-values

    // In increasing seq_profile order.
    let profiles = &[
        // Main_420_10_IP0 is the same as Main_420_10_IP1 except for the interoperability point.
        // An interoperability point specifies the number of extended and embedded layers a decoder
        // is capable of decoding simultaneously. Only keep Main_420_10_IP1 as IP1 should be
        // widespread enough; there is no need to get down to IP0.

        // Main_420_10_IP2 is the same as Main_420_10_IP1 except for the interoperability point.
        // Only keep Main_420_10_IP1 as there is no need for more than 2 mlayers in CrabbyAvif.

        // Main_420_10_IP1
        Av2Profile {
            seq_profile: 1,
            depths: &[8, 10], // bit_depth_idc 1 or 0
            formats: &[PixelFormat::Yuv400, PixelFormat::Yuv420],
        },
        // Main_422_10_IP1
        Av2Profile {
            seq_profile: 3,
            depths: &[8, 10], // bit_depth_idc 1 or 0
            formats: &[
                PixelFormat::Yuv400,
                PixelFormat::Yuv420,
                PixelFormat::Yuv422,
            ],
        },
        // Main_444_10_IP1
        Av2Profile {
            seq_profile: 4,
            depths: &[8, 10], // bit_depth_idc 1 or 0
            formats: &[
                PixelFormat::Yuv400,
                PixelFormat::Yuv420,
                PixelFormat::Yuv422,
                PixelFormat::Yuv444,
            ],
        },
        // Main_444C_12_IP2
        // Added in https://github.com/AOMediaCodec/avm/pull/5112. Not part of AV2 v1.0.0.
        Av2Profile {
            seq_profile: 5,       // Main_444C_12_IP2
            depths: &[8, 10, 12], // bit_depth_idc 1, 0, or 2
            formats: &[
                PixelFormat::Yuv400,
                PixelFormat::Yuv420,
                PixelFormat::Yuv422,
                PixelFormat::Yuv444,
            ],
        },
    ];
    let yuv_format = match category {
        Category::Color => image.yuv_format,
        Category::Alpha => PixelFormat::Yuv400,
        // TODO(b/437292541): Support AVM gainmaps.
        Category::Gainmap => return AvifError::not_implemented(),
    };
    // Return the smallest compatible seq_profile to maximize interoperability.
    for profile in profiles {
        if profile.depths.contains(&image.depth) && profile.formats.contains(&yuv_format) {
            return Ok(profile);
        }
    }
    AvifError::invalid_argument()
}

fn avm_scaling_mode(scaling_mode: &ScalingMode) -> AvifResult<avm_scaling_mode_t> {
    fn get_avm_scaling_mode_1d(mut fraction: IFraction) -> AvifResult<avm_scaling_mode_1d> {
        fraction.is_valid()?;
        fraction.simplify();
        Ok(match fraction {
            IFraction(1, 1) => avm_scaling_mode_1d_AVME_NORMAL,
            IFraction(1, 2) => avm_scaling_mode_1d_AVME_ONETWO,
            // IFraction(1, 3) => avm_scaling_mode_1d_AVME_ONETHREE, // Only exists in libaom.
            IFraction(1, 4) => avm_scaling_mode_1d_AVME_ONEFOUR,
            IFraction(1, 8) => avm_scaling_mode_1d_AVME_ONEEIGHT,
            // IFraction(2, 3) => avm_scaling_mode_1d_AVME_TWOTHREE, // Only exists in libaom.
            IFraction(3, 4) => avm_scaling_mode_1d_AVME_THREEFOUR,
            IFraction(3, 5) => avm_scaling_mode_1d_AVME_THREEFIVE,
            IFraction(4, 5) => avm_scaling_mode_1d_AVME_FOURFIVE,
            _ => return AvifError::not_implemented(),
        })
    }

    Ok(avm_scaling_mode_t {
        h_scaling_mode: get_avm_scaling_mode_1d(scaling_mode.horizontal)?,
        v_scaling_mode: get_avm_scaling_mode_1d(scaling_mode.vertical)?,
    })
}

fn avm_quality_to_quantizer(quality: i32) -> i32 {
    ((100 - quality) * 63 + 50) / 100
}

fn avm_min_max_quantizers(quantizer: i32) -> (i32, i32) {
    if quantizer == 0 {
        (0, 0)
    } else {
        ((quantizer - 4).max(0), (quantizer + 4).min(63))
    }
}

// Calls `handle_frame` for each frame packet available from `ctx`. Returns true if at least one
// frame packet was processed.
fn for_each_avm_frame_pkt(
    ctx: &mut avm_codec_ctx_t,
    mut handle_frame: impl FnMut(&[u8], bool) -> AvifResult<()>,
) -> AvifResult<bool> {
    let mut got_frame = false;
    let mut iter: avm_codec_iter_t = std::ptr::null_mut();
    loop {
        let ctx_ptr: *mut avm_codec_ctx_t = ctx;
        let iter_ptr: *mut avm_codec_iter_t = &mut iter;
        // SAFETY: `ctx_ptr` points to an initialized `avm_codec_ctx_t` and `iter_ptr` points to a
        // valid stack `avm_codec_iter_t` initialized to null per libavm's iteration protocol.
        let pkt = unsafe { avm_codec_get_cx_data(ctx_ptr, iter_ptr) };
        if pkt.is_null() {
            break;
        }
        // SAFETY: `pkt` was checked to be non-null above, and libavm guarantees the returned
        // packet pointer is valid and aligned until the next codec call on `ctx`.
        let pkt = unsafe { &*pkt };
        if pkt.kind != avm_codec_cx_pkt_kind_AVM_CODEC_CX_FRAME_PKT {
            continue;
        }
        // SAFETY: `pkt.data` is a C union; `pkt.kind == AVM_CODEC_CX_FRAME_PKT` guarantees that
        // `pkt.data.frame` is the active initialized variant per libavm's API contract.
        let frame = unsafe { &pkt.data.frame };
        let encoded_data = if frame.sz == 0 {
            &[][..]
        } else {
            if frame.buf.is_null() || frame.sz > isize::MAX as usize {
                return AvifError::unknown_error("libavm returned invalid frame buffer");
            }
            let buf_ptr = frame.buf as *const u8;
            // SAFETY: `frame.buf` is non-null, `u8` has alignment 1, `frame.sz <= isize::MAX`,
            // and libavm guarantees `frame.buf` points to `frame.sz` initialized bytes valid
            // until the next codec call on `ctx`.
            unsafe { slice::from_raw_parts(buf_ptr, frame.sz) }
        };
        // TODO(b/437292541): Make sure the sync definition below matches
        //                    https://aomediacodec.github.io/av2-isobmff/main#sync-sample
        let sync = (frame.flags & AVM_FRAME_IS_KEY) != 0;
        handle_frame(encoded_data, sync)?;
        got_frame = true;
    }
    Ok(got_frame)
}

// Adds frame packets to output_samples. Returns true if at least one sample was added.
fn add_avm_pkts_to_output_samples(
    ctx: &mut avm_codec_ctx_t,
    output_samples: &mut Vec<Sample>,
) -> AvifResult<bool> {
    for_each_avm_frame_pkt(ctx, |encoded_data, sync| {
        add_output_sample(encoded_data.try_to_vec()?, sync, output_samples)
    })
}

fn add_output_sample(
    data_output_by_codec: Vec<u8>,
    sync: bool,
    output_samples: &mut Vec<Sample>,
) -> AvifResult<()> {
    let sample_data_range = Av2CodecConfiguration::get_sample_obus_range(&data_output_by_codec)?;
    output_samples.try_push(Sample::create(
        data_output_by_codec,
        sample_data_range,
        sync,
    ))
}

impl Encoder for Avm {
    fn encode_image(
        &mut self,
        image: &Image,
        category: Category,
        config: &EncoderConfig,
        output_samples: &mut Vec<Sample>,
    ) -> AvifResult<()> {
        let quality = config.quality.clamp(0.0, 100.0) as i32;
        let quantizer = avm_quality_to_quantizer(quality);

        if self.context.is_none() {
            // SAFETY: `avm_codec_av2_cx` has no preconditions and returns a static descriptor.
            let encoder_iface = unsafe { avm_codec_av2_cx() };
            if encoder_iface.is_null() {
                return AvifError::unknown_error("avm_codec_av2_cx() returned null");
            }
            let avm_usage = AVM_USAGE_GOOD_QUALITY;
            let mut cfg_uninit: MaybeUninit<avm_codec_enc_cfg> = MaybeUninit::zeroed();
            let cfg_uninit_ptr = cfg_uninit.as_mut_ptr();
            // SAFETY: `encoder_iface` is non-null and valid, and `cfg_uninit_ptr` points to a
            // properly aligned `avm_codec_enc_cfg` stack slot to be initialized by libavm.
            let err =
                unsafe { avm_codec_enc_config_default(encoder_iface, cfg_uninit_ptr, avm_usage) };
            if err != avm_codec_err_t_AVM_CODEC_OK {
                return AvifError::unknown_error(format!(
                    "avm_codec_enc_config_default() failed: {err}"
                ));
            }
            // SAFETY: `avm_codec_enc_config_default` returned `AVM_CODEC_OK`, guaranteeing that
            // `cfg_uninit` has been completely initialized.
            let mut avm_config = unsafe { cfg_uninit.assume_init() };
            avm_config.rc_end_usage = avm_rc_mode_AVM_Q;
            avm_config.g_profile = av2_profile(image, category)?.seq_profile;
            avm_config.g_bit_depth = image.depth.into();
            avm_config.g_input_bit_depth = image.depth.into();
            avm_config.g_w = image.width;
            avm_config.g_h = image.height;

            self.has_alpha_layer = category == Category::Color
                && image.alpha_present
                && (!config.is_single_image || !image.is_opaque()?);
            if config.is_single_image {
                avm_config.g_limit = if self.has_alpha_layer { 2 } else { 1 };
                avm_config.g_lag_in_frames = 0;
                avm_config.kf_mode = avm_kf_mode_AVM_KF_DISABLED;
                avm_config.kf_max_dist = 0;
            }
            if config.disable_lagged_output || self.has_alpha_layer {
                avm_config.g_lag_in_frames = 0;
            }
            if config.extra_layer_count > 0 {
                avm_config.g_lag_in_frames = 0;
                avm_config.g_limit = config.extra_layer_count + 1;
            }
            if config.threads > 1 {
                avm_config.g_threads = config.threads.min(64);
            }

            avm_config.monochrome =
                (category == Category::Alpha || image.yuv_format == PixelFormat::Yuv400).into();
            // end-usage is the only codec specific option that has to be set before initializing
            // the libavm encoder
            if let Some(value) = config.codec_specific_option(category, "end-usage") {
                avm_config.rc_end_usage = match (value.parse(), value) {
                    (Ok(avm_rc_mode_AVM_VBR), _) | (Err(_), "vbr") => avm_rc_mode_AVM_VBR,
                    (Ok(avm_rc_mode_AVM_CBR), _) | (Err(_), "cbr") => avm_rc_mode_AVM_CBR,
                    (Ok(avm_rc_mode_AVM_CQ), _) | (Err(_), "cq") => avm_rc_mode_AVM_CQ,
                    (Ok(avm_rc_mode_AVM_Q), _) | (Err(_), "q") => avm_rc_mode_AVM_Q,
                    _ => return AvifError::invalid_argument(),
                };
            }
            if avm_config.rc_end_usage == avm_rc_mode_AVM_VBR
                || avm_config.rc_end_usage == avm_rc_mode_AVM_CBR
            {
                // cq-level is unused in these modes, so set the min and max quantizer instead.
                let (min, max) = avm_min_max_quantizers(quantizer);
                avm_config.rc_min_quantizer = min;
                avm_config.rc_max_quantizer = max;
            }

            // Zero initialize the struct first as some defaults (such as `err_detail`) are not
            // set by `avm_codec_enc_init_ver` when initialization succeeds.
            // SAFETY: Zero-initializing the C `avm_codec_ctx_t` struct matches the C API usage
            // before `avm_codec_enc_init_ver` and remains compatible across libavm versions.
            let mut encoder: avm_codec_ctx_t = unsafe { std::mem::zeroed() };
            let encoder_ptr: *mut avm_codec_ctx_t = &mut encoder;
            let cfg_ptr: *const avm_codec_enc_cfg = &avm_config;
            let init_flags = 0x0;
            let abi_version = AVM_ENCODER_ABI_VERSION as _;
            // SAFETY: `encoder_ptr`, `encoder_iface`, and `cfg_ptr` are valid non-null pointers,
            // and `abi_version` matches the headers linked into this build.
            let err = unsafe {
                avm_codec_enc_init_ver(encoder_ptr, encoder_iface, cfg_ptr, init_flags, abi_version)
            };
            if err != avm_codec_err_t_AVM_CODEC_OK {
                return AvifError::unknown_error(format!("avm_codec_enc_init() failed: {err}"));
            }
            self.context = Some(encoder);

            if avm_config.rc_end_usage == avm_rc_mode_AVM_CQ
                || avm_config.rc_end_usage == avm_rc_mode_AVM_Q
            {
                self.codec_control(avme_enc_control_id_AVME_SET_QP, quantizer)?;
            }
            if quantizer == 0 {
                self.codec_control(avme_enc_control_id_AV2E_SET_LOSSLESS, 1)?;
            }
            if config.tile_rows_log2 != 0 {
                self.codec_control(
                    avme_enc_control_id_AV2E_SET_TILE_ROWS,
                    config.tile_rows_log2,
                )?;
            }
            if config.tile_columns_log2 != 0 {
                self.codec_control(
                    avme_enc_control_id_AV2E_SET_TILE_COLUMNS,
                    config.tile_columns_log2,
                )?;
            }
            // TODO(b/437292541): Revisit using embedded layers (mlayer_id) vs. extended layers
            //                    (xlayer_id) and signaling the alpha layer in OBUs (e.g., via an
            //                    LCR, OPS, or ATLAS OBU) once supported by libavm and specified by
            //                    AV2-in-HEIF.
            let num_mlayers = if self.has_alpha_layer {
                if config.extra_layer_count > 0 {
                    // TODO(b/437292541): Support alpha layers with extra layers or explain why
                    //                    this is not possible.
                    return AvifError::not_implemented();
                }
                2
            } else {
                config.extra_layer_count + 1
            };
            if num_mlayers > 1 {
                self.codec_control(avme_enc_control_id_AVME_SET_NUMBER_MLAYERS, num_mlayers)?;
                self.codec_control(
                    avme_enc_control_id_AV2E_SET_ENABLE_FLAG_MULTI_LAYER_LAG_TEST,
                    1,
                )?;
            }
            if let Some(speed) = config.speed {
                self.codec_control(avme_enc_control_id_AVME_SET_CPUUSED, speed.min(9))?;
            }
            match category {
                Category::Alpha => {
                    // AVIF specification, Section 4 "Auxiliary Image Items and Sequences":
                    //   The color_range field in the Sequence Header OBU shall be set to 1.
                    self.codec_control(
                        avme_enc_control_id_AV2E_SET_COLOR_RANGE,
                        avm_color_range_AVM_CR_FULL_RANGE,
                    )?;
                    // Keep the default AVM_CSP_UNKNOWN value.

                    // CICP (CP/TC/MC) does not apply to the alpha auxiliary image.
                    // Keep default Unspecified (2) colour primaries, transfer characteristics,
                    // and matrix coefficients.
                }
                _ => {
                    // libavm's defaults are AVM_CSP_UNKNOWN and 0 (studio/limited range).
                    // Call avm_codec_control() only if the values are not the defaults.
                    // AV1-ISOBMFF specification, Section 2.3.4:
                    //   The value of full_range_flag in the 'colr' box SHALL match the color_range
                    //   flag in the Sequence Header OBU.
                    if image.yuv_range != YuvRange::Limited {
                        self.codec_control(
                            avme_enc_control_id_AV2E_SET_COLOR_RANGE,
                            avm_color_range_AVM_CR_FULL_RANGE,
                        )?;
                    }
                    // Section 2.3.4 of AV1-ISOBMFF says 'colr' with 'nclx' should be present and
                    // shall match CICP values in the Sequence Header OBU, unless the latter has
                    // 2/2/2 (Unspecified). So set CICP values to 2/2/2 (Unspecified) in the
                    // Sequence Header OBU for simplicity. libavm's defaults are
                    // AVM_CICP_CP_UNSPECIFIED, AVM_CICP_TC_UNSPECIFIED, and
                    // AVM_CICP_MC_UNSPECIFIED. No need to call avm_codec_control().
                }
            }

            let codec_specific_options = config.codec_specific_options(category)?;
            for &(key, value) in &codec_specific_options {
                if key == "end-usage" {
                    // This key is already processed before initialization of the encoder.
                    continue;
                }
                let key_str =
                    CString::new(key).map_err(|_| AvifError::InvalidCodecSpecificOption)?;
                let value_str =
                    CString::new(value).map_err(|_| AvifError::InvalidCodecSpecificOption)?;
                let ctx = self.context.as_mut().ok_or(AvifError::InvalidArgument)?;
                let ctx_ptr: *mut avm_codec_ctx_t = ctx;
                let key_ptr = key_str.as_ptr();
                let value_ptr = value_str.as_ptr();
                // SAFETY: `ctx_ptr` points to an initialized `avm_codec_ctx_t`, and `key_ptr` and
                // `value_ptr` are valid NUL-terminated C strings borrowed from `key_str` and
                // `value_str` which outlive the call.
                let err = unsafe { avm_codec_set_option(ctx_ptr, key_ptr, value_ptr) };
                if err != avm_codec_err_t_AVM_CODEC_OK {
                    return AvifError::unknown_error(format!(
                        "avm_codec_set_option({key}, {value}) failed: {err}"
                    ));
                }
            }
            // Set tune=SSIM unless explicitly defined by the user.
            if !codec_specific_options.iter().any(|&(key, _)| key == "tune") {
                self.codec_control(
                    avme_enc_control_id_AVME_SET_TUNING,
                    avm_tune_metric_AVM_TUNE_SSIM,
                )?;
            }

            self.encoder_avm_config = Some(avm_config);
            self.encoder_config = Some(config.try_clone()?);
        } else if self
            .encoder_config
            .as_ref()
            .ok_or(AvifError::InvalidArgument)?
            != config
        {
            let avm_config = self
                .encoder_avm_config
                .as_ref()
                .ok_or(AvifError::InvalidArgument)?;
            if avm_config.g_w != image.width || avm_config.g_h != image.height {
                // Dimension changes are forbidden.
                return AvifError::invalid_argument();
            }
            let rc_end_usage = avm_config.rc_end_usage;
            let last_config = self
                .encoder_config
                .as_ref()
                .ok_or(AvifError::InvalidArgument)?;
            let quality_changed = last_config.quality != config.quality;
            let tile_rows_changed = last_config.tile_rows_log2 != config.tile_rows_log2;
            let tile_cols_changed = last_config.tile_columns_log2 != config.tile_columns_log2;
            if quality_changed {
                if rc_end_usage == avm_rc_mode_AVM_VBR || rc_end_usage == avm_rc_mode_AVM_CBR {
                    let ctx = self.context.as_mut().ok_or(AvifError::InvalidArgument)?;
                    let avm_config = self
                        .encoder_avm_config
                        .as_mut()
                        .ok_or(AvifError::InvalidArgument)?;
                    (avm_config.rc_min_quantizer, avm_config.rc_max_quantizer) =
                        avm_min_max_quantizers(quantizer);
                    let ctx_ptr: *mut avm_codec_ctx_t = ctx;
                    let cfg_ptr: *const avm_codec_enc_cfg = avm_config;
                    // SAFETY: `ctx_ptr` points to an initialized `avm_codec_ctx_t` and `cfg_ptr`
                    // points to the initialized `avm_codec_enc_cfg` stored in `self`.
                    let err = unsafe { avm_codec_enc_config_set(ctx_ptr, cfg_ptr) };
                    if err != avm_codec_err_t_AVM_CODEC_OK {
                        return AvifError::unknown_error(format!(
                            "avm_codec_enc_config_set() failed: {err}"
                        ));
                    }
                } else if rc_end_usage == avm_rc_mode_AVM_CQ || rc_end_usage == avm_rc_mode_AVM_Q {
                    self.codec_control(avme_enc_control_id_AVME_SET_QP, quantizer)?;
                }
                self.codec_control(
                    avme_enc_control_id_AV2E_SET_LOSSLESS,
                    if quantizer == 0 { 1 } else { 0 },
                )?;
            }
            if tile_rows_changed {
                self.codec_control(
                    avme_enc_control_id_AV2E_SET_TILE_ROWS,
                    config.tile_rows_log2,
                )?;
            }
            if tile_cols_changed {
                self.codec_control(
                    avme_enc_control_id_AV2E_SET_TILE_COLUMNS,
                    config.tile_columns_log2,
                )?;
            }
            self.encoder_config = Some(config.try_clone()?);
        }
        if self.current_layer > config.extra_layer_count {
            return AvifError::invalid_argument();
        }
        if config.extra_layer_count > 0 {
            return AvifError::not_implemented();
        }
        if self.has_alpha_layer {
            self.codec_control(avme_enc_control_id_AVME_SET_MLAYER_ID, 0)?;
            self.codec_control(avme_enc_control_id_AVME_SET_QP, quantizer)?;
            self.codec_control(
                avme_enc_control_id_AV2E_SET_LOSSLESS,
                if quantizer == 0 { 1 } else { 0 },
            )?;
        }
        let current_encoder_config = self
            .encoder_config
            .as_ref()
            .ok_or(AvifError::InvalidArgument)?;
        let scaling_mode = avm_scaling_mode(&current_encoder_config.scaling_mode)?;
        if scaling_mode.h_scaling_mode != avm_scaling_mode_1d_AVME_NORMAL
            || scaling_mode.v_scaling_mode != avm_scaling_mode_1d_AVME_NORMAL
        {
            let scaling_mode_ptr: *const avm_scaling_mode_t = &scaling_mode;
            self.codec_control(avme_enc_control_id_AVME_SET_SCALEMODE, scaling_mode_ptr)?;
        }
        // SAFETY: Zero-initializing the C `avm_image_t` struct matches the C API usage
        // before populating active fields and remains compatible across libavm versions.
        let mut avm_image: avm_image_t = unsafe { std::mem::zeroed() };
        avm_image.fmt = avm_format(image, category)?;
        avm_image.bit_depth = if image.depth > 8 { 16 } else { 8 };
        avm_image.w = image.width;
        avm_image.h = image.height;
        avm_image.d_w = image.width; // Display dimensions equal to decoded dimensions.
        avm_image.d_h = image.height;
        avm_image.bps = match avm_image.fmt {
            avm_img_fmt_AVM_IMG_FMT_I420 => 12,   // 8 + 2 + 2 bits per sample
            avm_img_fmt_AVM_IMG_FMT_I422 => 16,   // 8 + 8 + 2 bits per sample
            avm_img_fmt_AVM_IMG_FMT_I444 => 24,   // 8 + 8 + 8 bits per sample
            avm_img_fmt_AVM_IMG_FMT_I42016 => 24, // 16 + 4 + 4 bits per sample
            avm_img_fmt_AVM_IMG_FMT_I42216 => 32, // 16 + 16 + 4 bits per sample
            avm_img_fmt_AVM_IMG_FMT_I44416 => 48, // 16 + 16 + 16 bits per sample
            _ => 16,
        };
        avm_image.x_chroma_shift = image.yuv_format.chroma_shift_x().0;
        avm_image.y_chroma_shift = image.yuv_format.chroma_shift_y();
        match category {
            Category::Alpha => {
                avm_image.range = avm_color_range_AVM_CR_FULL_RANGE;
                avm_image.monochrome = 1;
                avm_image.x_chroma_shift = 1;
                avm_image.y_chroma_shift = 1;
                avm_image.planes[0] = image.planes[3]
                    .as_ref()
                    .ok_or(AvifError::NoContent)?
                    .ptr_generic() as *mut _;
                avm_image.stride[0] = image.row_bytes[3] as i32;
            }
            _ => {
                avm_image.range = image.yuv_range as u32;
                if image.yuv_format == PixelFormat::Yuv400 {
                    avm_image.monochrome = 1;
                    avm_image.x_chroma_shift = 1;
                    avm_image.y_chroma_shift = 1;
                    avm_image.planes[0] = image.planes[0]
                        .as_ref()
                        .ok_or(AvifError::NoContent)?
                        .ptr_generic() as *mut _;
                    avm_image.stride[0] = image.row_bytes[0] as i32;
                } else {
                    avm_image.monochrome = 0;
                    for i in 0..=2 {
                        avm_image.planes[i] = image.planes[i]
                            .as_ref()
                            .ok_or(AvifError::NoContent)?
                            .ptr_generic() as *mut _;
                        avm_image.stride[i] = image.row_bytes[i] as i32;
                    }
                }
            }
        }
        avm_image.cp = image.color_primaries as u32;
        avm_image.tc = image.transfer_characteristics as u32;
        avm_image.mc = image.matrix_coefficients as u32;
        // TODO: b/392112497 - force keyframes when necessary.
        let mut encode_flags = 0;
        if self.current_layer > 0 {
            encode_flags |= AVM_EFLAG_NO_REF_GF as i64
                | AVM_EFLAG_NO_REF_ARF as i64
                | AVM_EFLAG_NO_REF_BWD as i64
                | AVM_EFLAG_NO_REF_ARF2 as i64;
        }
        let duration = 1;
        let ctx = self.context.as_mut().ok_or(AvifError::InvalidArgument)?;
        let ctx_ptr: *mut avm_codec_ctx_t = ctx;
        let avm_image_ptr: *const avm_image_t = &avm_image;
        let pts = avm_codec_pts_t::default();
        let flags = encode_flags as _;
        // SAFETY: `ctx_ptr` points to an initialized `avm_codec_ctx_t`, and `avm_image_ptr` points
        // to `avm_image` whose active plane pointers borrow from `image` (validated to have
        // allocated planes above) and remain valid for the duration of `avm_codec_encode`.
        let err = unsafe { avm_codec_encode(ctx_ptr, avm_image_ptr, pts, duration, flags) };
        if err != avm_codec_err_t_AVM_CODEC_OK {
            return AvifError::unknown_error(format!("avm_codec_encode() failed: {err}"));
        }
        if self.has_alpha_layer {
            let mut combined_data = Vec::new();
            let mut sync = false;
            for_each_avm_frame_pkt(ctx, |encoded_data, pkt_sync| {
                sync |= pkt_sync;
                combined_data.try_extend_from_slice(encoded_data)
            })?;

            self.codec_control(avme_enc_control_id_AVME_SET_MLAYER_ID, 1)?;
            let quality_alpha = config.quality_alpha.clamp(0.0, 100.0) as i32;
            let quantizer_alpha = avm_quality_to_quantizer(quality_alpha);
            self.codec_control(avme_enc_control_id_AVME_SET_QP, quantizer_alpha)?;
            self.codec_control(
                avme_enc_control_id_AV2E_SET_LOSSLESS,
                if quantizer_alpha == 0 { 1 } else { 0 },
            )?;

            // SAFETY: Zero-initializing the C `avm_image_t` struct matches the C API usage
            // before populating active fields and remains compatible across libavm versions.
            let mut avm_alpha_image: avm_image_t = unsafe { std::mem::zeroed() };
            // All embedded layers share a single SequenceHeader in libavm, so fmt, bps,
            // x_chroma_shift, and y_chroma_shift must match avm_image to pass libavm's
            // validate_img() checks even though the alpha plane is monochrome. Because
            // avm_alpha_image.monochrome is 1, libavm only copies planes[0] into its lookahead
            // buffer (leaving the U and V planes of the lookahead buffer untouched, i.e. zeroed
            // on initial allocation); if the color image is not monochrome, libavm still encodes
            // those U and V planes for mlayer_id == 1, and they are ignored on decode.
            avm_alpha_image.fmt = avm_image.fmt;
            avm_alpha_image.bit_depth = avm_image.bit_depth;
            avm_alpha_image.w = avm_image.w;
            avm_alpha_image.h = avm_image.h;
            avm_alpha_image.d_w = avm_image.d_w;
            avm_alpha_image.d_h = avm_image.d_h;
            avm_alpha_image.bps = avm_image.bps;
            avm_alpha_image.x_chroma_shift = avm_image.x_chroma_shift;
            avm_alpha_image.y_chroma_shift = avm_image.y_chroma_shift;
            avm_alpha_image.range = avm_color_range_AVM_CR_FULL_RANGE;
            avm_alpha_image.monochrome = 1;
            avm_alpha_image.planes[0] = image.planes[3]
                .as_ref()
                .ok_or(AvifError::NoContent)?
                .ptr_generic() as *mut _;
            avm_alpha_image.stride[0] = image.row_bytes[3] as i32;
            // Leave CICP values as unspecified.

            let ctx = self.context.as_mut().ok_or(AvifError::InvalidArgument)?;
            let ctx_ptr: *mut avm_codec_ctx_t = ctx;
            let alpha_img_ptr: *const avm_image_t = &avm_alpha_image;
            let alpha_pts = avm_codec_pts_t::default();
            let alpha_flags = encode_flags as _;
            // SAFETY: `ctx_ptr` points to an initialized `avm_codec_ctx_t`, and `alpha_img_ptr`
            // points to `avm_alpha_image` whose alpha plane pointer borrows from `image`
            // (validated to have an allocated alpha plane above) and remains valid for the call.
            let err = unsafe {
                avm_codec_encode(ctx_ptr, alpha_img_ptr, alpha_pts, duration, alpha_flags)
            };
            if err != avm_codec_err_t_AVM_CODEC_OK {
                return AvifError::unknown_error(format!(
                    "avm_codec_encode() failed for alpha: {err}"
                ));
            }

            for_each_avm_frame_pkt(ctx, |encoded_data, pkt_sync| {
                sync |= pkt_sync;
                let sample_range = Av2CodecConfiguration::get_sample_obus_range(encoded_data)?;
                let sample_slice = encoded_data.get(sample_range).ok_or(AvifError::NoContent)?;
                combined_data.try_extend_from_slice(sample_slice)
            })?;
            add_output_sample(combined_data, sync, output_samples)?;
        } else {
            add_avm_pkts_to_output_samples(ctx, output_samples)?;
        }
        if config.is_single_image
            || (config.extra_layer_count > 0 && config.extra_layer_count == self.current_layer)
        {
            self.finish(output_samples)?;
            self.destroy_context();
        }
        if config.extra_layer_count > 0 {
            self.current_layer += 1;
        }
        Ok(())
    }

    fn finish(&mut self, output_samples: &mut Vec<crate::encoder::Sample>) -> AvifResult<()> {
        let Some(ctx) = self.context.as_mut() else {
            return Ok(());
        };
        loop {
            // Flush the encoder.
            let duration = 1;
            let ctx_ptr: *mut avm_codec_ctx_t = ctx;
            let null_img: *const avm_image_t = std::ptr::null();
            let pts = avm_codec_pts_t::default();
            let flags = avm_enc_frame_flags_t::default();
            // SAFETY: `ctx_ptr` points to an initialized `avm_codec_ctx_t`, and passing a null
            // `img` pointer is the documented libavm API for flushing buffered frames.
            let err = unsafe { avm_codec_encode(ctx_ptr, null_img, pts, duration, flags) };
            if err != avm_codec_err_t_AVM_CODEC_OK {
                return AvifError::unknown_error(format!("Flush avm_codec_encode() failed: {err}"));
            }
            let got_packet = if let Some(last_sample) =
                output_samples.last_mut().filter(|_| self.has_alpha_layer)
            {
                for_each_avm_frame_pkt(ctx, |encoded_data, sync| {
                    let mut combined = last_sample.data_output_by_codec().try_to_vec()?;
                    let sample_range = Av2CodecConfiguration::get_sample_obus_range(encoded_data)?;
                    let sample_slice =
                        encoded_data.get(sample_range).ok_or(AvifError::NoContent)?;
                    combined.try_extend_from_slice(sample_slice)?;
                    let sample_data_range =
                        Av2CodecConfiguration::get_sample_obus_range(&combined)?;
                    *last_sample =
                        Sample::create(combined, sample_data_range, last_sample.sync || sync);
                    Ok(())
                })?
            } else {
                add_avm_pkts_to_output_samples(ctx, output_samples)?
            };
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
        // Harvest codec configuration (av2C) from AV2 OBUs.
        let first_sample = output_samples.first().ok_or(AvifError::NoContent)?;
        let all_obus = first_sample.data_output_by_codec();
        let config_obus = Av2CodecConfiguration::get_config_obus_range_and_count(all_obus)?;
        let codec_config = Av2CodecConfiguration::from_av2_config_obus(
            config_obus.count,
            &all_obus[config_obus.range],
        )?;
        Ok(CodecConfiguration::Av2(codec_config))
    }
}

fn copy_avm_planes(
    image: &mut Image,
    avm_image: &avm_image_t,
    category: Category,
    plane_mappings: &[(usize, Plane)],
) -> AvifResult<()> {
    // CrabbyAvif's Image assumes that a depth of 8 bits means an 8-bit buffer.
    // avm_image does not. The buffer depth depends on fmt|AVM_IMG_FMT_HIGHBITDEPTH, even for 8-bit values.
    if image.depth <= 8 && (avm_image.fmt & AVM_IMG_FMT_HIGHBITDEPTH) != 0 {
        image.allocate_planes(category)?;
        for &(src_plane_idx, dst_plane) in plane_mappings {
            let plane_width = image.width(dst_plane);
            let plane_height = image.height(dst_plane);
            if plane_width == 0 || plane_height == 0 {
                continue;
            }
            let src_plane = avm_image.planes[src_plane_idx] as *const u16;
            if src_plane.is_null() {
                return AvifError::unknown_error("AVM returned a NULL buffer");
            }
            if (src_plane as usize) % std::mem::align_of::<u16>() != 0 {
                return AvifError::unknown_error(
                    "AVM returned an unaligned buffer for 16-bit samples",
                );
            }
            let src_stride_in_bytes = avm_image.stride[src_plane_idx];
            if src_stride_in_bytes % 2 != 0 {
                return AvifError::unknown_error("AVM returned an odd stride for 16-bit samples");
            }
            let src_stride_in_samples: usize = (src_stride_in_bytes / 2)
                .try_into()
                .map_err(AvifError::map_unknown_error)?;
            if src_stride_in_samples < plane_width {
                return AvifError::unknown_error("AVM returned a stride smaller than width");
            }
            let src_len = checked_add!(
                checked_mul!(src_stride_in_samples, checked_sub!(plane_height, 1)?)?,
                plane_width
            )?;
            let _ = isize_from_usize(checked_mul!(src_len, std::mem::size_of::<u16>())?)?;
            // SAFETY: `src_plane` was checked to be non-null and aligned for `u16`, the total
            // byte length `src_len * size_of::<u16>()` does not exceed `isize::MAX`, and
            // libavm guarantees `planes[src_plane_idx]` points to `plane_height` initialized
            // rows of 16-bit samples with stride `src_stride_in_samples`.
            let src_plane = unsafe { slice::from_raw_parts(src_plane, src_len) };
            for y in 0..plane_height {
                let row_start = y * src_stride_in_samples;
                let src_row = &src_plane[row_start..row_start + plane_width];
                let dst_row = image.row_mut(dst_plane, u32_from_usize(y)?)?;
                for (&src_pixel, dst_pixel) in src_row.iter().zip(dst_row.iter_mut()) {
                    *dst_pixel = src_pixel as u8;
                }
            }
        }
    } else {
        // Steal the pointers from the decoder's image directly.
        for &(src_plane_idx, dst_plane) in plane_mappings {
            let dst_idx = dst_plane.as_usize();
            image.row_bytes[dst_idx] = avm_image.stride[src_plane_idx]
                .try_into()
                .map_err(AvifError::map_unknown_error)?;
            image.planes[dst_idx] = Some(Pixels::from_raw_pointer(
                avm_image.planes[src_plane_idx],
                image.depth.into(),
                u32_from_usize(image.height(dst_plane))?,
                image.row_bytes[dst_idx],
            )?);
        }
    }
    Ok(())
}

fn assign_avm_alpha_plane(image: &mut Image, avm_alpha_image: &avm_image_t) -> AvifResult<()> {
    image.free_planes(&A_PLANE);
    if !(image.depth <= 8 && (avm_alpha_image.fmt & AVM_IMG_FMT_HIGHBITDEPTH) != 0) {
        // Check that the alpha plane start pointer is distinct from the color plane start pointers.
        // This is a weak assertion that the AVM codec instance for color was not reused for alpha.
        // This check alone does not guarantee that the borrowed 'avm_alpha_image' outlast 'image'.
        for plane in image.planes[..3].iter().flatten() {
            if plane.ptr_generic() == avm_alpha_image.planes[0].cast_const() {
                return AvifError::unknown_error("AVM alpha plane aliases color plane");
            }
        }
    }
    copy_avm_planes(image, avm_alpha_image, Category::Alpha, &[(0, Plane::A)])?;
    image.alpha_present = true;
    Ok(())
}

impl Decoder for Avm {
    fn codec(&self) -> CodecChoice {
        CodecChoice::Avm
    }

    fn initialize(&mut self, config: &DecoderConfig) -> AvifResult<()> {
        self.decoder_config = Some(DecoderConfig {
            codec_config: config.codec_config.try_clone()?,
            ..*config
        });
        Ok(())
    }

    fn get_next_image(
        &mut self,
        av2_payload: &[u8],
        spatial_id: u8,
        image: &mut Image,
        category: Category,
        item: Option<&Item>,
        _signal_eos: bool,
    ) -> AvifResult<()> {
        let mut first_time = self.context.is_none();
        if first_time {
            let decoder_config = self
                .decoder_config
                .as_ref()
                .ok_or(AvifError::InvalidArgument)?;
            let max_threads = decoder_config.max_threads;
            let all_layers = decoder_config.all_layers;
            // SAFETY: `avm_codec_av2_dx` has no preconditions and returns a static descriptor.
            let decoder_iface = unsafe { avm_codec_av2_dx() };
            if decoder_iface.is_null() {
                return AvifError::unknown_error("avm_codec_av2_dx() returned null");
            }
            // SAFETY: Zero-initializing the C `avm_codec_dec_cfg` struct is valid for all its
            // integer and pointer fields and remains compatible across libavm versions.
            let mut avm_config: avm_codec_dec_cfg = unsafe { std::mem::zeroed() };
            avm_config.threads = max_threads;
            // SAFETY: Zero-initializing the C `avm_codec_ctx_t` struct matches the C API usage
            // before `avm_codec_dec_init_ver` and remains compatible across libavm versions.
            let mut context: avm_codec_ctx_t = unsafe { std::mem::zeroed() };
            let ctx_ptr: *mut avm_codec_ctx_t = &mut context;
            let cfg_ptr: *const avm_codec_dec_cfg = &avm_config;
            let init_flags = avm_codec_flags_t::default();
            let abi_version = AVM_DECODER_ABI_VERSION as _;
            // SAFETY: `ctx_ptr`, `decoder_iface`, and `cfg_ptr` are valid non-null pointers,
            // and `abi_version` matches the headers linked into this build.
            let err = unsafe {
                avm_codec_dec_init_ver(ctx_ptr, decoder_iface, cfg_ptr, init_flags, abi_version)
            };
            if err != avm_codec_err_t_AVM_CODEC_OK {
                return AvifError::unknown_error(format!("avm_codec_dec_init() failed: {err}"));
            }
            self.context = Some(context);

            self.codec_control(
                avm_dec_control_id_AV2D_SET_OUTPUT_ALL_LAYERS,
                if all_layers
                    || item.is_some_and(|item| item.has_native_alpha())
                    || (category == Category::Color && image.alpha_present)
                {
                    1
                } else {
                    0
                },
            )?;

            self.decoder_iter = std::ptr::null();
        }

        let mut next_frame;
        let mut target_spatial_id = 0xff;
        let mut av2_payload_can_be_used = true;
        loop {
            let ctx = self.context.as_mut().ok_or(AvifError::InvalidArgument)?;
            let ctx_ptr: *mut avm_codec_ctx_t = ctx;
            let iter_ptr: *mut avm_codec_iter_t = &mut self.decoder_iter;
            // SAFETY: `ctx_ptr` points to an initialized decoder `avm_codec_ctx_t` and `iter_ptr`
            // points to `self.decoder_iter`, which is initialized to null before each decode batch.
            next_frame = unsafe { avm_codec_get_frame(ctx_ptr, iter_ptr) };
            if !next_frame.is_null() {
                if target_spatial_id != 0xff {
                    // SAFETY: `next_frame` was checked to be non-null above, and libavm guarantees
                    // the returned `avm_image_t` pointer is valid until the next decode/destroy call.
                    let mlayer_id = unsafe { (*next_frame).mlayer_id };
                    if target_spatial_id == mlayer_id {
                        // Found the correct spatial_id.
                        break;
                    }
                } else {
                    // Got an image!
                    break;
                }
            } else if av2_payload_can_be_used {
                // Prepend the AV2CodecConfigurationBox config_obus to the
                // sample data OBUs.
                // TODO(b/437292541): Calling avm_codec_decode() twice fails.
                //                    Add an API to avoid allocating a buffer.
                let mut config_obus_then_av2_payload = Vec::new();
                let av2_payload = if first_time {
                    first_time = false;
                    let decoder_config = self
                        .decoder_config
                        .as_ref()
                        .ok_or(AvifError::InvalidArgument)?;
                    let config_obus = match &decoder_config.codec_config {
                        CodecConfiguration::Av2(av2c) => &av2c.config_obus,
                        _ => return AvifError::invalid_argument(),
                    };
                    let config_obus_then_av2_payload_len = config_obus
                        .len()
                        .checked_add(av2_payload.len())
                        .ok_or_else(|| AvifError::map_out_of_memory(()))?;
                    config_obus_then_av2_payload
                        .try_reserve_exact(config_obus_then_av2_payload_len)
                        .map_err(AvifError::map_out_of_memory)?;
                    config_obus_then_av2_payload.try_extend_from_slice(config_obus)?;
                    config_obus_then_av2_payload.try_extend_from_slice(av2_payload)?;
                    &config_obus_then_av2_payload
                } else {
                    av2_payload
                };

                self.decoder_iter = std::ptr::null();
                let ctx = self.context.as_mut().ok_or(AvifError::InvalidArgument)?;
                let ctx_ptr: *mut avm_codec_ctx_t = ctx;
                let data_ptr = av2_payload.as_ptr();
                let data_len = av2_payload.len();
                let user_priv = std::ptr::null_mut();
                // SAFETY: `ctx_ptr` points to an initialized decoder `avm_codec_ctx_t`, and
                // `data_ptr` is valid for `data_len` bytes borrowed from `av2_payload`.
                let err = unsafe { avm_codec_decode(ctx_ptr, data_ptr, data_len, user_priv) };
                if err != avm_codec_err_t_AVM_CODEC_OK {
                    // SAFETY: `ctx_ptr` points to an initialized decoder `avm_codec_ctx_t`
                    // (same context whose `avm_codec_decode` call just failed).
                    let ptr = unsafe { avm_codec_error_detail(ctx_ptr) };
                    let detail = if ptr.is_null() {
                        String::new()
                    } else {
                        // SAFETY: `ptr` was checked to be non-null above and libavm guarantees
                        // `avm_codec_error_detail` returns a valid NUL-terminated C string owned
                        // by `ctx_ptr` until the next codec call or destruction.
                        unsafe { CStr::from_ptr(ptr) }
                            .to_string_lossy()
                            .into_owned()
                    };
                    return AvifError::unknown_error(format!(
                        "avm_codec_decode() failed: {err}: {detail}"
                    ));
                }
                target_spatial_id = spatial_id.into();
                av2_payload_can_be_used = false;
            } else {
                break;
            }
        }

        if !next_frame.is_null() {
            self.image = next_frame;
        } else if category == Category::Alpha && !self.image.is_null() {
            // Special case: reuse last alpha frame
        } else {
            return AvifError::unknown_error("avm_codec_get_frame() failed");
        }
        // SAFETY: `self.image` is non-null (verified in the branches above and cleared before
        // `avm_codec_destroy`), and points to a valid `avm_image_t` owned by `self.context` with
        // no concurrent mutable access.
        let avm_image = unsafe { &*self.image };

        let color_planes = [(0usize, Plane::Y), (1, Plane::U), (2, Plane::V)];
        if category == Category::Color {
            // Color (YUV) planes - set image to correct size / format, fill color
            let yuv_format = if avm_image.monochrome != 0 {
                PixelFormat::Yuv400
            } else {
                match avm_image.fmt {
                    avm_img_fmt_AVM_IMG_FMT_I420 | avm_img_fmt_AVM_IMG_FMT_I42016 => {
                        PixelFormat::Yuv420
                    }
                    avm_img_fmt_AVM_IMG_FMT_I422 | avm_img_fmt_AVM_IMG_FMT_I42216 => {
                        PixelFormat::Yuv422
                    }
                    avm_img_fmt_AVM_IMG_FMT_I444 | avm_img_fmt_AVM_IMG_FMT_I44416 => {
                        PixelFormat::Yuv444
                    }
                    fmt => {
                        return AvifError::unknown_error(format!(
                            "Unrecognized AVM pixel format {fmt}"
                        ))
                    }
                }
            };

            if image.width != 0
                && image.height != 0
                && (image.width != avm_image.d_w
                    || image.height != avm_image.d_h
                    || image.depth as u32 != avm_image.bit_depth
                    || image.yuv_format != yuv_format)
            {
                image.free_planes(&ALL_PLANES);
            }
            image.width = avm_image.d_w;
            image.height = avm_image.d_h;
            image.depth = avm_image
                .bit_depth
                .try_into()
                .map_err(AvifError::map_unknown_error)?;
            image.yuv_format = yuv_format;
            image.yuv_range = match avm_image.range {
                avm_color_range_AVM_CR_STUDIO_RANGE => YuvRange::Limited,
                avm_color_range_AVM_CR_FULL_RANGE => YuvRange::Full,
                range => return AvifError::bmff_parse_failed(format!("Invalid AVM range {range}")),
            };
            image.chroma_sample_position = match avm_image.csp {
                // Horizontal offset 0, vertical offset 0.5
                avm_chroma_sample_position_AVM_CSP_LEFT => ChromaSamplePosition::Vertical,
                // Horizontal offset 0.5, vertical offset 0.5
                avm_chroma_sample_position_AVM_CSP_CENTER => ChromaSamplePosition::Unknown,
                // Horizontal offset 0, vertical offset 0
                avm_chroma_sample_position_AVM_CSP_TOPLEFT => ChromaSamplePosition::Colocated,
                _ => ChromaSamplePosition::Unknown,
            };

            image.color_primaries = (avm_image.cp as u16).into();
            image.transfer_characteristics = (avm_image.tc as u16).into();
            image.matrix_coefficients = (avm_image.mc as u16).into();

            image.free_planes(&YUV_PLANES);
            copy_avm_planes(
                image,
                avm_image,
                category,
                &color_planes[..image.yuv_format.plane_count()],
            )?;
            if item.is_some_and(|item| item.has_native_alpha()) || image.alpha_present {
                let ctx = self.context.as_mut().ok_or(AvifError::InvalidArgument)?;
                let ctx_ptr: *mut avm_codec_ctx_t = ctx;
                let iter_ptr: *mut avm_codec_iter_t = &mut self.decoder_iter;
                // SAFETY: `ctx_ptr` points to an initialized decoder `avm_codec_ctx_t` and
                // `iter_ptr` points to `self.decoder_iter`.
                let alpha_frame = unsafe { avm_codec_get_frame(ctx_ptr, iter_ptr) };
                if !alpha_frame.is_null() {
                    // SAFETY: `alpha_frame` was checked to be non-null above and points to a
                    // valid `avm_image_t` owned by `self.context`.
                    let avm_alpha_image = unsafe { &*alpha_frame };
                    // TODO(b/437292541): Revisit hardcoded mlayer_id == 1 once AV2-in-HEIF
                    //                    specifies how to find if and where there is alpha.
                    if avm_alpha_image.mlayer_id == 1 {
                        assign_avm_alpha_plane(image, avm_alpha_image)?;
                    }
                }
            }
        } else {
            // Alpha plane as an auxiliary image item: map AVM's luma plane (0)
            // to CrabbyAvif's alpha plane (3).
            if image.width != 0
                && image.height != 0
                && (image.width != avm_image.d_w
                    || image.height != avm_image.d_h
                    || image.depth as u32 != avm_image.bit_depth)
            {
                // Alpha plane doesn't match previous alpha plane decode, bail out.
                return AvifError::unknown_error(format!(
                    "Alpha plane does not match existing image dimensions: {}-bit {}x{} vs {}-bit {}x{}",
                    image.depth, image.width, image.height, avm_image.bit_depth, avm_image.d_w, avm_image.d_h,
                ));
            }
            image.width = avm_image.d_w;
            image.height = avm_image.d_h;
            image.depth = avm_image
                .bit_depth
                .try_into()
                .map_err(AvifError::map_unknown_error)?;

            assign_avm_alpha_plane(image, avm_image)?;
        }

        let ctx = self.context.as_mut().ok_or(AvifError::InvalidArgument)?;
        let ctx_ptr: *mut avm_codec_ctx_t = ctx;
        let iter_ptr: *mut avm_codec_iter_t = &mut self.decoder_iter;
        // SAFETY: `ctx_ptr` points to an initialized decoder `avm_codec_ctx_t` and `iter_ptr`
        // points to `self.decoder_iter`.
        while !unsafe { avm_codec_get_frame(ctx_ptr, iter_ptr) }.is_null() {}

        Ok(())
    }

    fn get_next_image_grid(
        &mut self,
        _payloads: &[Vec<u8>],
        _spatial_id: u8,
        _grid_image_helper: &mut GridImageHelper,
    ) -> AvifResult<()> {
        AvifError::not_implemented() // TODO: b/437292541 - Implement
    }

    fn get_last_image(
        &mut self,
        _payloads: &[Vec<u8>],
        _spatial_id: u8,
        _image: &mut Image,
        _category: Category,
    ) -> AvifResult<()> {
        AvifError::not_implemented()
    }
}

impl Drop for Avm {
    fn drop(&mut self) {
        self.destroy_context();
    }
}

impl Avm {
    fn destroy_context(&mut self) {
        self.image = std::ptr::null_mut();
        self.decoder_iter = std::ptr::null();
        if let Some(mut ctx) = self.context.take() {
            let ctx_ptr: *mut avm_codec_ctx_t = &mut ctx;
            // SAFETY: `ctx` was initialized by a successful `avm_codec_enc_init_ver` or
            // `avm_codec_dec_init_ver` call, and `self.context.take()` ensures `avm_codec_destroy`
            // is called at most once. `self.image` and `self.decoder_iter` are cleared first.
            unsafe {
                avm_codec_destroy(ctx_ptr);
            }
        }
    }

    pub(crate) fn version() -> String {
        // SAFETY: `avm_codec_version_str` has no preconditions and returns a static C string.
        let version_ptr = unsafe { avm_codec_version_str() };
        let version = if version_ptr.is_null() {
            String::new()
        } else {
            // SAFETY: `version_ptr` was checked to be non-null above and points to a static
            // NUL-terminated string literal owned by libavm.
            let version_cstr = unsafe { CStr::from_ptr(version_ptr) };
            version_cstr.to_str().unwrap_or_default().to_owned()
        };
        format!("avm: {version}")
    }

    // Convenience avm_codec_control() wrapper.
    fn codec_control<T>(&mut self, key: u32, value: T) -> AvifResult<()> {
        let ctx = self.context.as_mut().ok_or(AvifError::InvalidArgument)?;
        let ctx_ptr: *mut avm_codec_ctx_t = ctx;
        let key_id = key as i32;
        // SAFETY: `ctx_ptr` points to an initialized `avm_codec_ctx_t`, and every caller in this
        // module supplies the C-ABI control argument type matching `key` (`i32`, `u32`, or
        // `*const avm_scaling_mode_t`).
        let err = unsafe { avm_codec_control(ctx_ptr, key_id, value) };
        if err != avm_codec_err_t_AVM_CODEC_OK {
            return AvifError::unknown_error(format!("avm_codec_control({key}) failed: {err}"));
        }
        Ok(())
    }
}

#[test]
fn avm_enc_dec_test() -> Result<(), AvifError> {
    let mut encoder = Avm::default();
    let config = EncoderConfig {
        is_single_image: true,
        quality: 100.0,
        ..Default::default()
    };

    let image = {
        let mut image = Image {
            width: 1,
            height: 1,
            depth: 8,
            yuv_format: PixelFormat::Yuv444,
            ..Default::default()
        };
        image.allocate_planes(Category::Color)?;
        let mut rgb = reformat::rgb::Image::create_from_yuv(&image);
        rgb.format = Format::Rgb;
        rgb.allocate()?;
        rgb.row_mut(0)?[0] = 42;
        rgb.row_mut(0)?[1] = 80;
        rgb.row_mut(0)?[2] = 0;
        rgb.convert_to_yuv(&mut image)?;
        image
    };

    let mut output_samples = vec![];
    encoder.encode_image(&image, Category::Color, &config, &mut output_samples)?;
    encoder.finish(&mut output_samples)?;
    assert_eq!(output_samples.len(), 1);
    let all_obus = output_samples[0].data_output_by_codec();
    let config_obus = Av2CodecConfiguration::get_config_obus_range_and_count(all_obus)?;
    let codec_config = Av2CodecConfiguration::from_av2_config_obus(
        config_obus.count,
        &all_obus[config_obus.range],
    )?;

    let mut decoder = Avm::default();
    let mut decoded = Image::default();
    decoder.initialize(&DecoderConfig {
        operating_point: 0,
        all_layers: false,
        width: image.width,
        height: image.height,
        depth: image.depth,
        max_threads: 0,
        image_size_limit: None,
        max_input_size: 0,
        codec_config: CodecConfiguration::Av2(codec_config),
        category: Category::Color,
        android_mediacodec_output_color_format: AndroidMediaCodecOutputColorFormat::default(),
    })?;
    decoder.get_next_image(
        output_samples[0].sample_data(),
        0xff,
        &mut decoded,
        Category::Color,
        None,
        false,
    )?;

    assert!(are_images_equal(&image, &decoded)?);
    Ok(())
}

#[test]
fn avm_enc_dec_alpha_layer_test() -> Result<(), AvifError> {
    let mut encoder = Avm::default();
    let config = EncoderConfig {
        is_single_image: true,
        quality: 100.0,
        quality_alpha: 100.0,
        ..Default::default()
    };

    let image = {
        let mut image = Image {
            width: 1,
            height: 1,
            depth: 8,
            yuv_format: PixelFormat::Yuv444,
            ..Default::default()
        };
        image.allocate_planes(Category::Color)?;
        image.allocate_planes(Category::Alpha)?;
        let mut rgb = reformat::rgb::Image::create_from_yuv(&image);
        rgb.format = Format::Rgb;
        rgb.allocate()?;
        rgb.row_mut(0)?[0] = 42;
        rgb.row_mut(0)?[1] = 80;
        rgb.row_mut(0)?[2] = 0;
        rgb.convert_to_yuv(&mut image)?;
        image.row_mut(Plane::A, 0)?[0] = 128;
        image.alpha_present = true;
        image
    };

    let mut output_samples = vec![];
    encoder.encode_image(&image, Category::Color, &config, &mut output_samples)?;
    encoder.finish(&mut output_samples)?;
    assert_eq!(output_samples.len(), 1);
    let all_obus = output_samples[0].data_output_by_codec();
    let config_obus = Av2CodecConfiguration::get_config_obus_range_and_count(all_obus)?;
    let codec_config = Av2CodecConfiguration::from_av2_config_obus(
        config_obus.count,
        &all_obus[config_obus.range],
    )?;

    let mut decoder = Avm::default();
    let mut decoded = Image {
        alpha_present: true,
        ..Default::default()
    };
    decoder.initialize(&DecoderConfig {
        operating_point: 0,
        all_layers: false,
        width: image.width,
        height: image.height,
        depth: image.depth,
        max_threads: 0,
        image_size_limit: None,
        max_input_size: 0,
        codec_config: CodecConfiguration::Av2(codec_config),
        category: Category::Color,
        android_mediacodec_output_color_format: AndroidMediaCodecOutputColorFormat::default(),
    })?;
    decoder.get_next_image(
        output_samples[0].sample_data(),
        0xff,
        &mut decoded,
        Category::Color,
        None,
        false,
    )?;

    assert!(decoded.alpha_present);
    assert!(are_images_equal(&image, &decoded)?);
    Ok(())
}

#[test]
fn avm_enc_dec_opaque_alpha_test() -> Result<(), AvifError> {
    let mut encoder = Avm::default();
    let config = EncoderConfig {
        is_single_image: true,
        quality: 100.0,
        quality_alpha: 100.0,
        ..Default::default()
    };

    let image = {
        let mut image = Image {
            width: 1,
            height: 1,
            depth: 8,
            yuv_format: PixelFormat::Yuv444,
            ..Default::default()
        };
        image.allocate_planes(Category::Color)?;
        image.allocate_planes(Category::Alpha)?;
        let mut rgb = reformat::rgb::Image::create_from_yuv(&image);
        rgb.format = Format::Rgb;
        rgb.allocate()?;
        rgb.row_mut(0)?[0] = 42;
        rgb.row_mut(0)?[1] = 80;
        rgb.row_mut(0)?[2] = 0;
        rgb.convert_to_yuv(&mut image)?;
        image.row_mut(Plane::A, 0)?[0] = 255;
        image.alpha_present = true;
        image
    };

    let mut output_samples = vec![];
    encoder.encode_image(&image, Category::Color, &config, &mut output_samples)?;
    encoder.finish(&mut output_samples)?;
    assert_eq!(output_samples.len(), 1);
    let all_obus = output_samples[0].data_output_by_codec();
    let config_obus = Av2CodecConfiguration::get_config_obus_range_and_count(all_obus)?;
    let codec_config = Av2CodecConfiguration::from_av2_config_obus(
        config_obus.count,
        &all_obus[config_obus.range],
    )?;

    let mut decoder = Avm::default();
    let mut decoded = Image::default();
    decoder.initialize(&DecoderConfig {
        operating_point: 0,
        all_layers: false,
        width: image.width,
        height: image.height,
        depth: image.depth,
        max_threads: 0,
        image_size_limit: None,
        max_input_size: 0,
        codec_config: CodecConfiguration::Av2(codec_config),
        category: Category::Color,
        android_mediacodec_output_color_format: AndroidMediaCodecOutputColorFormat::default(),
    })?;
    decoder.get_next_image(
        output_samples[0].sample_data(),
        0xff,
        &mut decoded,
        Category::Color,
        None,
        false,
    )?;

    assert!(!decoded.alpha_present);
    assert!(!decoded.has_plane(Plane::A));
    for plane in YUV_PLANES {
        assert!(are_planes_equal(&image, &decoded, plane)?);
    }
    Ok(())
}
