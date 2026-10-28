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

//! Sequence header and frame header parsing (spec 5.5, 5.9), restricted
//! to what is needed for a single intra frame.

use crate::decoder::av1::bitreader::BitReader;
use crate::decoder::av1::dsp::film_grain::FilmGrainParams;
use crate::decoder::av1::error::{Error, Result};
use crate::decoder::av1::types::*;

pub const SELECT_SCREEN_CONTENT_TOOLS: u32 = 2;
pub const SELECT_INTEGER_MV: u32 = 2;
pub const MC_IDENTITY: u8 = 0;
pub const CP_BT_709: u8 = 1;
pub const TC_SRGB: u8 = 13;
const MAX_TILE_WIDTH: u32 = 4096;
const MAX_TILE_AREA: u32 = 4096 * 2304;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameType {
    Key = 0,
    Inter = 1,
    IntraOnly = 2,
    Switch = 3,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ColorConfig {
    pub bit_depth: u32,
    pub mono_chrome: bool,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub color_range: bool,
    pub chroma_sample_position: u8,
    pub separate_uv_delta_q: bool,
}

impl ColorConfig {
    pub fn num_planes(&self) -> usize {
        if self.mono_chrome {
            1
        } else {
            3
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SequenceHeader {
    pub seq_profile: u32,
    pub still_picture: bool,
    pub reduced_still_picture_header: bool,
    pub operating_point_idc: u32,
    pub decoder_model_info_present: bool,
    pub equal_picture_interval: bool,
    pub buffer_removal_time_length_minus_1: u32,
    pub frame_presentation_time_length_minus_1: u32,
    pub operating_points: Vec<(u32, bool)>, // (idc, decoder_model_present)
    pub frame_width_bits: u32,
    pub frame_height_bits: u32,
    pub max_frame_width: u32,
    pub max_frame_height: u32,
    pub frame_id_numbers_present: bool,
    pub delta_frame_id_length_minus_2: u32,
    pub additional_frame_id_length_minus_1: u32,
    pub use_128x128_superblock: bool,
    pub enable_filter_intra: bool,
    pub enable_intra_edge_filter: bool,
    pub enable_order_hint: bool,
    pub order_hint_bits: u32,
    pub seq_force_screen_content_tools: u32,
    pub seq_force_integer_mv: u32,
    pub enable_superres: bool,
    pub enable_cdef: bool,
    pub enable_restoration: bool,
    pub color: ColorConfig,
    pub film_grain_params_present: bool,
}

impl SequenceHeader {
    /// Parses `sequence_header_obu()` (spec 5.5.1).
    pub fn parse(data: &[u8]) -> Result<Self> {
        let mut r = BitReader::new(data);
        let mut s = SequenceHeader {
            seq_profile: r.f(3)?,
            ..Default::default()
        };
        if s.seq_profile > 2 {
            return Err(Error::Invalid("seq_profile"));
        }
        s.still_picture = r.flag()?;
        s.reduced_still_picture_header = r.flag()?;
        if s.reduced_still_picture_header {
            let _seq_level_idx = r.f(5)?;
            s.operating_points.push((0, false));
        } else {
            let timing_info_present = r.flag()?;
            if timing_info_present {
                let _num_units_in_display_tick = r.f(32)?;
                let _time_scale = r.f(32)?;
                s.equal_picture_interval = r.flag()?;
                if s.equal_picture_interval {
                    let _ = r.uvlc()?;
                }
                s.decoder_model_info_present = r.flag()?;
            }
            let mut buffer_delay_length_minus_1 = 0;
            if s.decoder_model_info_present {
                buffer_delay_length_minus_1 = r.f(5)?;
                let _num_units_in_decoding_tick = r.f(32)?;
                s.buffer_removal_time_length_minus_1 = r.f(5)?;
                s.frame_presentation_time_length_minus_1 = r.f(5)?;
            }
            let initial_display_delay_present = r.flag()?;
            let cnt = r.f(5)? + 1;
            for _ in 0..cnt {
                let idc = r.f(12)?;
                let seq_level_idx = r.f(5)?;
                if seq_level_idx > 7 {
                    let _seq_tier = r.f(1)?;
                }
                let mut dm_present = false;
                if s.decoder_model_info_present {
                    dm_present = r.flag()?;
                    if dm_present {
                        let n = buffer_delay_length_minus_1 + 1;
                        let _decoder_buffer_delay = r.f(n)?;
                        let _encoder_buffer_delay = r.f(n)?;
                        let _low_delay_mode_flag = r.f(1)?;
                    }
                }
                if initial_display_delay_present && r.flag()? {
                    let _initial_display_delay_minus_1 = r.f(4)?;
                }
                s.operating_points.push((idc, dm_present));
            }
        }
        // choose_operating_point(): always 0.
        s.operating_point_idc = s.operating_points[0].0;
        s.frame_width_bits = r.f(4)? + 1;
        s.frame_height_bits = r.f(4)? + 1;
        s.max_frame_width = r.f(s.frame_width_bits)? + 1;
        s.max_frame_height = r.f(s.frame_height_bits)? + 1;
        if !s.reduced_still_picture_header {
            s.frame_id_numbers_present = r.flag()?;
        }
        if s.frame_id_numbers_present {
            s.delta_frame_id_length_minus_2 = r.f(4)?;
            s.additional_frame_id_length_minus_1 = r.f(3)?;
        }
        s.use_128x128_superblock = r.flag()?;
        s.enable_filter_intra = r.flag()?;
        s.enable_intra_edge_filter = r.flag()?;
        if s.reduced_still_picture_header {
            s.seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS;
            s.seq_force_integer_mv = SELECT_INTEGER_MV;
        } else {
            let _enable_interintra_compound = r.flag()?;
            let _enable_masked_compound = r.flag()?;
            let _enable_warped_motion = r.flag()?;
            let _enable_dual_filter = r.flag()?;
            s.enable_order_hint = r.flag()?;
            if s.enable_order_hint {
                let _enable_jnt_comp = r.flag()?;
                let _enable_ref_frame_mvs = r.flag()?;
            }
            let seq_choose_screen_content_tools = r.flag()?;
            s.seq_force_screen_content_tools = if seq_choose_screen_content_tools {
                SELECT_SCREEN_CONTENT_TOOLS
            } else {
                r.f(1)?
            };
            if s.seq_force_screen_content_tools > 0 {
                let seq_choose_integer_mv = r.flag()?;
                s.seq_force_integer_mv =
                    if seq_choose_integer_mv { SELECT_INTEGER_MV } else { r.f(1)? };
            } else {
                s.seq_force_integer_mv = SELECT_INTEGER_MV;
            }
            if s.enable_order_hint {
                s.order_hint_bits = r.f(3)? + 1;
            }
        }
        s.enable_superres = r.flag()?;
        s.enable_cdef = r.flag()?;
        s.enable_restoration = r.flag()?;
        s.color = Self::parse_color_config(&mut r, s.seq_profile)?;
        s.film_grain_params_present = r.flag()?;
        Ok(s)
    }

    fn parse_color_config(r: &mut BitReader, seq_profile: u32) -> Result<ColorConfig> {
        let mut c = ColorConfig::default();
        let high_bitdepth = r.flag()?;
        c.bit_depth = if seq_profile == 2 && high_bitdepth {
            if r.flag()? {
                12
            } else {
                10
            }
        } else if high_bitdepth {
            10
        } else {
            8
        };
        c.mono_chrome = if seq_profile == 1 { false } else { r.flag()? };
        let color_description_present = r.flag()?;
        if color_description_present {
            c.color_primaries = r.f(8)? as u8;
            c.transfer_characteristics = r.f(8)? as u8;
            c.matrix_coefficients = r.f(8)? as u8;
        } else {
            c.color_primaries = 2;
            c.transfer_characteristics = 2;
            c.matrix_coefficients = 2;
        }
        if c.mono_chrome {
            c.color_range = r.flag()?;
            c.subsampling_x = true;
            c.subsampling_y = true;
            return Ok(c);
        } else if c.color_primaries == CP_BT_709
            && c.transfer_characteristics == TC_SRGB
            && c.matrix_coefficients == MC_IDENTITY
        {
            c.color_range = true;
        } else {
            c.color_range = r.flag()?;
            match seq_profile {
                0 => {
                    c.subsampling_x = true;
                    c.subsampling_y = true;
                }
                1 => {}
                _ => {
                    if c.bit_depth == 12 {
                        c.subsampling_x = r.flag()?;
                        c.subsampling_y = if c.subsampling_x { r.flag()? } else { false };
                    } else {
                        c.subsampling_x = true;
                    }
                }
            }
            if c.subsampling_x && c.subsampling_y {
                c.chroma_sample_position = r.f(2)? as u8;
            }
        }
        c.separate_uv_delta_q = r.flag()?;
        Ok(c)
    }
}

#[derive(Clone, Debug, Default)]
pub struct TileInfo {
    pub cols: usize,
    pub rows: usize,
    pub cols_log2: u32,
    pub rows_log2: u32,
    /// `MiColStarts[0..=cols]`.
    pub mi_col_starts: Vec<usize>,
    /// `MiRowStarts[0..=rows]`.
    pub mi_row_starts: Vec<usize>,
    pub context_update_tile_id: u32,
    pub tile_size_bytes: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct QuantParams {
    pub base_q_idx: u8,
    pub delta_q_y_dc: i32,
    pub delta_q_u_dc: i32,
    pub delta_q_u_ac: i32,
    pub delta_q_v_dc: i32,
    pub delta_q_v_ac: i32,
    pub using_qmatrix: bool,
    pub qm_y: u8,
    pub qm_u: u8,
    pub qm_v: u8,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SegmentationParams {
    pub enabled: bool,
    pub update_map: bool,
    pub temporal_update: bool,
    pub feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub feature_data: [[i32; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub seg_id_pre_skip: bool,
    pub last_active_seg_id: u8,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LoopFilterParams {
    pub level: [u8; 4],
    pub sharpness: u8,
    pub delta_enabled: bool,
    pub ref_deltas: [i8; 8],
    pub mode_deltas: [i8; 2],
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CdefParams {
    pub damping: u8,
    pub bits: u8,
    pub y_pri_strength: [u8; 8],
    pub y_sec_strength: [u8; 8],
    pub uv_pri_strength: [u8; 8],
    pub uv_sec_strength: [u8; 8],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RestorationType {
    #[default]
    None,
    Wiener,
    Sgrproj,
    Switchable,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LrParams {
    pub frame_restoration_type: [RestorationType; 3],
    pub uses_lr: bool,
    pub uses_chroma_lr: bool,
    /// `LoopRestorationSize[plane]` in samples of that plane.
    pub loop_restoration_size: [usize; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TxMode {
    #[default]
    Only4x4,
    Largest,
    Select,
}

#[derive(Clone, Debug)]
pub struct FrameHeader {
    pub frame_type: FrameType,
    pub show_frame: bool,
    pub showable_frame: bool,
    pub error_resilient_mode: bool,
    pub disable_cdf_update: bool,
    pub allow_screen_content_tools: bool,
    pub force_integer_mv: bool,
    pub frame_width: usize,
    pub frame_height: usize,
    pub upscaled_width: usize,
    pub render_width: usize,
    pub render_height: usize,
    pub use_superres: bool,
    pub superres_denom: u32,
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub allow_intrabc: bool,
    pub tile_info: TileInfo,
    pub quant: QuantParams,
    pub seg: SegmentationParams,
    pub delta_q_present: bool,
    pub delta_q_res: u8,
    pub delta_lf_present: bool,
    pub delta_lf_res: u8,
    pub delta_lf_multi: bool,
    pub coded_lossless: bool,
    pub all_lossless: bool,
    pub lossless_array: [bool; MAX_SEGMENTS],
    pub seg_qm_level: [[u8; MAX_SEGMENTS]; 3],
    pub lf: LoopFilterParams,
    pub cdef: CdefParams,
    pub lr: LrParams,
    pub tx_mode: TxMode,
    pub reduced_tx_set: bool,
    pub film_grain: FilmGrainParams,
}

const SEGMENTATION_FEATURE_BITS: [u32; SEG_LVL_MAX] = [8, 6, 6, 6, 6, 3, 0, 0];
const SEGMENTATION_FEATURE_SIGNED: [bool; SEG_LVL_MAX] =
    [true, true, true, true, true, false, false, false];
const SEGMENTATION_FEATURE_MAX: [i32; SEG_LVL_MAX] = [255, 63, 63, 63, 63, 7, 0, 0];

fn tile_log2(blk_size: u32, target: u32) -> u32 {
    let mut k = 0;
    while (blk_size << k) < target {
        k += 1;
    }
    k
}

impl FrameHeader {
    /// Parses `uncompressed_header()` (spec 5.9.2) for an intra frame.
    /// `temporal_id` / `spatial_id` come from the OBU extension header.
    /// Returns the header and the number of bits consumed.
    pub fn parse(
        data: &[u8],
        seq: &SequenceHeader,
        temporal_id: u32,
        spatial_id: u32,
    ) -> Result<(Self, usize)> {
        let mut r = BitReader::new(data);
        let cc = &seq.color;
        let id_len = seq.additional_frame_id_length_minus_1 + seq.delta_frame_id_length_minus_2 + 3;
        let all_frames = 0xffu32;
        let frame_type;
        let show_frame;
        let showable_frame;
        let error_resilient_mode;
        if seq.reduced_still_picture_header {
            frame_type = FrameType::Key;
            show_frame = true;
            showable_frame = false;
            error_resilient_mode = true;
        } else {
            let show_existing_frame = r.flag()?;
            if show_existing_frame {
                return Err(Error::Unsupported("show_existing_frame"));
            }
            frame_type = match r.f(2)? {
                0 => FrameType::Key,
                1 => FrameType::Inter,
                2 => FrameType::IntraOnly,
                _ => FrameType::Switch,
            };
            if !matches!(frame_type, FrameType::Key | FrameType::IntraOnly) {
                return Err(Error::Unsupported("inter frame"));
            }
            show_frame = r.flag()?;
            if show_frame && seq.decoder_model_info_present && !seq.equal_picture_interval {
                let _frame_presentation_time =
                    r.f(seq.frame_presentation_time_length_minus_1 + 1)?;
            }
            showable_frame = if show_frame { frame_type != FrameType::Key } else { r.flag()? };
            error_resilient_mode =
                if frame_type == FrameType::Key && show_frame { true } else { r.flag()? };
        }
        let disable_cdf_update = r.flag()?;
        let allow_screen_content_tools =
            if seq.seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS {
                r.flag()?
            } else {
                seq.seq_force_screen_content_tools != 0
            };
        if allow_screen_content_tools && seq.seq_force_integer_mv == SELECT_INTEGER_MV {
            let _force_integer_mv = r.flag()?;
        }
        let force_integer_mv = true; // FrameIsIntra
        if seq.frame_id_numbers_present {
            let _current_frame_id = r.f(id_len)?;
        }
        let frame_size_override_flag =
            if seq.reduced_still_picture_header { false } else { r.flag()? };
        let _order_hint = r.f(seq.order_hint_bits)?;
        // primary_ref_frame = PRIMARY_REF_NONE for intra frames.
        if seq.decoder_model_info_present {
            let buffer_removal_time_present = r.flag()?;
            if buffer_removal_time_present {
                for &(idc, dm_present) in &seq.operating_points {
                    if dm_present {
                        let in_temporal = (idc >> temporal_id) & 1 != 0;
                        let in_spatial = (idc >> (spatial_id + 8)) & 1 != 0;
                        if idc == 0 || (in_temporal && in_spatial) {
                            let _ = r.f(seq.buffer_removal_time_length_minus_1 + 1)?;
                        }
                    }
                }
            }
        }
        let refresh_frame_flags =
            if frame_type == FrameType::Key && show_frame { all_frames } else { r.f(8)? };
        if refresh_frame_flags != all_frames && error_resilient_mode && seq.enable_order_hint {
            for _ in 0..8 {
                let _ref_order_hint = r.f(seq.order_hint_bits)?;
            }
        }
        // frame_size()
        let (mut frame_width, frame_height) = if frame_size_override_flag {
            // Note: frame_width_minus_1 <= max_frame_width_minus_1 is a
            // conformance requirement, but SVT-AV1 superres streams violate
            // it (and dav1d / libaom accept them), so it is not enforced.
            (
                r.f(seq.frame_width_bits)? as usize + 1,
                r.f(seq.frame_height_bits)? as usize + 1,
            )
        } else {
            (seq.max_frame_width as usize, seq.max_frame_height as usize)
        };
        // superres_params()
        let use_superres = seq.enable_superres && r.flag()?;
        let superres_denom = if use_superres {
            r.f(SUPERRES_DENOM_BITS)? + SUPERRES_DENOM_MIN
        } else {
            SUPERRES_NUM
        };
        let upscaled_width = frame_width;
        frame_width = (upscaled_width * SUPERRES_NUM as usize + (superres_denom as usize / 2))
            / superres_denom as usize;
        // compute_image_size()
        let mi_cols = 2 * ((frame_width + 7) >> 3);
        let mi_rows = 2 * ((frame_height + 7) >> 3);
        // render_size()
        let (render_width, render_height) = if r.flag()? {
            (r.f(16)? as usize + 1, r.f(16)? as usize + 1)
        } else {
            (upscaled_width, frame_height)
        };
        let allow_intrabc =
            allow_screen_content_tools && upscaled_width == frame_width && r.flag()?;

        let _disable_frame_end_update_cdf =
            if seq.reduced_still_picture_header || disable_cdf_update {
                true
            } else {
                r.flag()?
            };

        let mut fh = FrameHeader {
            frame_type,
            show_frame,
            showable_frame,
            error_resilient_mode,
            disable_cdf_update,
            allow_screen_content_tools,
            force_integer_mv,
            frame_width,
            frame_height,
            upscaled_width,
            render_width,
            render_height,
            use_superres,
            superres_denom,
            mi_cols,
            mi_rows,
            allow_intrabc,
            tile_info: TileInfo::default(),
            quant: QuantParams::default(),
            seg: SegmentationParams::default(),
            delta_q_present: false,
            delta_q_res: 0,
            delta_lf_present: false,
            delta_lf_res: 0,
            delta_lf_multi: false,
            coded_lossless: false,
            all_lossless: false,
            lossless_array: [false; MAX_SEGMENTS],
            seg_qm_level: [[0; MAX_SEGMENTS]; 3],
            lf: LoopFilterParams::default(),
            cdef: CdefParams::default(),
            lr: LrParams::default(),
            tx_mode: TxMode::Only4x4,
            reduced_tx_set: false,
            film_grain: FilmGrainParams::default(),
        };
        fh.parse_tile_info(&mut r, seq)?;
        fh.parse_quantization_params(&mut r, cc)?;
        fh.parse_segmentation_params(&mut r)?;
        // delta_q_params()
        if fh.quant.base_q_idx > 0 {
            fh.delta_q_present = r.flag()?;
        }
        if fh.delta_q_present {
            fh.delta_q_res = r.f(2)? as u8;
        }
        // delta_lf_params()
        if fh.delta_q_present {
            if !fh.allow_intrabc {
                fh.delta_lf_present = r.flag()?;
            }
            if fh.delta_lf_present {
                fh.delta_lf_res = r.f(2)? as u8;
                fh.delta_lf_multi = r.flag()?;
            }
        }
        fh.coded_lossless = true;
        for seg_id in 0..MAX_SEGMENTS {
            let qindex = fh.get_qindex(true, seg_id, 0);
            let q = &fh.quant;
            let lossless = qindex == 0
                && q.delta_q_y_dc == 0
                && q.delta_q_u_ac == 0
                && q.delta_q_u_dc == 0
                && q.delta_q_v_ac == 0
                && q.delta_q_v_dc == 0;
            fh.lossless_array[seg_id] = lossless;
            if !lossless {
                fh.coded_lossless = false;
            }
            if q.using_qmatrix {
                if lossless {
                    fh.seg_qm_level[0][seg_id] = 15;
                    fh.seg_qm_level[1][seg_id] = 15;
                    fh.seg_qm_level[2][seg_id] = 15;
                } else {
                    fh.seg_qm_level[0][seg_id] = q.qm_y;
                    fh.seg_qm_level[1][seg_id] = q.qm_u;
                    fh.seg_qm_level[2][seg_id] = q.qm_v;
                }
            }
        }
        fh.all_lossless = fh.coded_lossless && fh.frame_width == fh.upscaled_width;
        if fh.coded_lossless && fh.delta_q_present {
            return Err(Error::Invalid("delta_q_present in a lossless frame"));
        }
        fh.parse_loop_filter_params(&mut r, cc)?;
        fh.parse_cdef_params(&mut r, seq)?;
        fh.parse_lr_params(&mut r, seq)?;
        // read_tx_mode()
        fh.tx_mode = if fh.coded_lossless {
            TxMode::Only4x4
        } else if r.flag()? {
            TxMode::Select
        } else {
            TxMode::Largest
        };
        // frame_reference_mode(), skip_mode_params(), allow_warped_motion:
        // nothing coded for intra frames.
        fh.reduced_tx_set = r.flag()?;
        // global_motion_params(): nothing coded for intra frames.
        fh.parse_film_grain_params(&mut r, seq)?;
        Ok((fh, r.position()))
    }

    fn parse_tile_info(&mut self, r: &mut BitReader, seq: &SequenceHeader) -> Result<()> {
        let mi_cols = self.mi_cols as u32;
        let mi_rows = self.mi_rows as u32;
        let (sb_cols, sb_rows, sb_shift) = if seq.use_128x128_superblock {
            ((mi_cols + 31) >> 5, (mi_rows + 31) >> 5, 5)
        } else {
            ((mi_cols + 15) >> 4, (mi_rows + 15) >> 4, 4)
        };
        let sb_size = sb_shift + 2;
        let max_tile_width_sb = MAX_TILE_WIDTH >> sb_size;
        let mut max_tile_area_sb = MAX_TILE_AREA >> (2 * sb_size);
        let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
        let max_log2_tile_cols = tile_log2(1, sb_cols.min(MAX_TILE_COLS as u32));
        let max_log2_tile_rows = tile_log2(1, sb_rows.min(MAX_TILE_ROWS as u32));
        let min_log2_tiles = min_log2_tile_cols.max(tile_log2(max_tile_area_sb, sb_rows * sb_cols));
        let ti = &mut self.tile_info;
        ti.mi_col_starts.clear();
        ti.mi_row_starts.clear();
        let uniform = r.flag()?;
        if uniform {
            ti.cols_log2 = min_log2_tile_cols;
            while ti.cols_log2 < max_log2_tile_cols {
                if r.flag()? {
                    ti.cols_log2 += 1;
                } else {
                    break;
                }
            }
            let tile_width_sb = (sb_cols + (1 << ti.cols_log2) - 1) >> ti.cols_log2;
            let mut start = 0;
            while start < sb_cols {
                ti.mi_col_starts.push((start << sb_shift) as usize);
                start += tile_width_sb;
            }
            ti.mi_col_starts.push(mi_cols as usize);
            ti.cols = ti.mi_col_starts.len() - 1;
            let min_log2_tile_rows = min_log2_tiles.saturating_sub(ti.cols_log2);
            ti.rows_log2 = min_log2_tile_rows;
            while ti.rows_log2 < max_log2_tile_rows {
                if r.flag()? {
                    ti.rows_log2 += 1;
                } else {
                    break;
                }
            }
            let tile_height_sb = (sb_rows + (1 << ti.rows_log2) - 1) >> ti.rows_log2;
            let mut start = 0;
            while start < sb_rows {
                ti.mi_row_starts.push((start << sb_shift) as usize);
                start += tile_height_sb;
            }
            ti.mi_row_starts.push(mi_rows as usize);
            ti.rows = ti.mi_row_starts.len() - 1;
        } else {
            let mut widest_tile_sb = 0;
            let mut start = 0;
            while start < sb_cols {
                ti.mi_col_starts.push((start << sb_shift) as usize);
                let max_width = (sb_cols - start).min(max_tile_width_sb);
                let size_sb = r.ns(max_width)? + 1;
                widest_tile_sb = widest_tile_sb.max(size_sb);
                start += size_sb;
            }
            ti.mi_col_starts.push(mi_cols as usize);
            ti.cols = ti.mi_col_starts.len() - 1;
            ti.cols_log2 = tile_log2(1, ti.cols as u32);
            if min_log2_tiles > 0 {
                max_tile_area_sb = (sb_rows * sb_cols) >> (min_log2_tiles + 1);
            } else {
                max_tile_area_sb = sb_rows * sb_cols;
            }
            let max_tile_height_sb = (max_tile_area_sb / widest_tile_sb).max(1);
            let mut start = 0;
            while start < sb_rows {
                ti.mi_row_starts.push((start << sb_shift) as usize);
                let max_height = (sb_rows - start).min(max_tile_height_sb);
                let size_sb = r.ns(max_height)? + 1;
                start += size_sb;
            }
            ti.mi_row_starts.push(mi_rows as usize);
            ti.rows = ti.mi_row_starts.len() - 1;
            ti.rows_log2 = tile_log2(1, ti.rows as u32);
        }
        if ti.cols > MAX_TILE_COLS || ti.rows > MAX_TILE_ROWS {
            return Err(Error::Invalid("too many tiles"));
        }
        if ti.cols_log2 > 0 || ti.rows_log2 > 0 {
            ti.context_update_tile_id = r.f(ti.rows_log2 + ti.cols_log2)?;
            if ti.context_update_tile_id as usize >= ti.cols * ti.rows {
                return Err(Error::Invalid("context_update_tile_id"));
            }
            ti.tile_size_bytes = r.f(2)? + 1;
        } else {
            ti.context_update_tile_id = 0;
        }
        Ok(())
    }

    fn read_delta_q(r: &mut BitReader) -> Result<i32> {
        if r.flag()? {
            r.su(7)
        } else {
            Ok(0)
        }
    }

    fn parse_quantization_params(&mut self, r: &mut BitReader, cc: &ColorConfig) -> Result<()> {
        let q = &mut self.quant;
        q.base_q_idx = r.f(8)? as u8;
        q.delta_q_y_dc = Self::read_delta_q(r)?;
        if cc.num_planes() > 1 {
            let diff_uv_delta = if cc.separate_uv_delta_q { r.flag()? } else { false };
            q.delta_q_u_dc = Self::read_delta_q(r)?;
            q.delta_q_u_ac = Self::read_delta_q(r)?;
            if diff_uv_delta {
                q.delta_q_v_dc = Self::read_delta_q(r)?;
                q.delta_q_v_ac = Self::read_delta_q(r)?;
            } else {
                q.delta_q_v_dc = q.delta_q_u_dc;
                q.delta_q_v_ac = q.delta_q_u_ac;
            }
        }
        q.using_qmatrix = r.flag()?;
        if q.using_qmatrix {
            q.qm_y = r.f(4)? as u8;
            q.qm_u = r.f(4)? as u8;
            q.qm_v = if !cc.separate_uv_delta_q { q.qm_u } else { r.f(4)? as u8 };
        }
        Ok(())
    }

    fn parse_segmentation_params(&mut self, r: &mut BitReader) -> Result<()> {
        let s = &mut self.seg;
        s.enabled = r.flag()?;
        if s.enabled {
            // primary_ref_frame == PRIMARY_REF_NONE
            s.update_map = true;
            s.temporal_update = false;
            for i in 0..MAX_SEGMENTS {
                for j in 0..SEG_LVL_MAX {
                    let enabled = r.flag()?;
                    s.feature_enabled[i][j] = enabled;
                    let mut clipped = 0;
                    if enabled {
                        let bits = SEGMENTATION_FEATURE_BITS[j];
                        let limit = SEGMENTATION_FEATURE_MAX[j];
                        if SEGMENTATION_FEATURE_SIGNED[j] {
                            clipped = r.su(1 + bits)?.clamp(-limit, limit);
                        } else {
                            clipped = (r.f(bits)? as i32).clamp(0, limit);
                        }
                    }
                    s.feature_data[i][j] = clipped;
                }
            }
        }
        s.seg_id_pre_skip = false;
        s.last_active_seg_id = 0;
        for i in 0..MAX_SEGMENTS {
            for j in 0..SEG_LVL_MAX {
                if s.feature_enabled[i][j] {
                    s.last_active_seg_id = i as u8;
                    if j >= SEG_LVL_REF_FRAME {
                        s.seg_id_pre_skip = true;
                    }
                }
            }
        }
        Ok(())
    }

    fn parse_loop_filter_params(&mut self, r: &mut BitReader, cc: &ColorConfig) -> Result<()> {
        let lf = &mut self.lf;
        // Defaults (setup_past_independence).
        lf.ref_deltas = [1, 0, 0, 0, -1, 0, -1, -1];
        lf.mode_deltas = [0, 0];
        if self.coded_lossless || self.allow_intrabc {
            lf.level = [0; 4];
            return Ok(());
        }
        lf.level[0] = r.f(6)? as u8;
        lf.level[1] = r.f(6)? as u8;
        if cc.num_planes() > 1 && (lf.level[0] != 0 || lf.level[1] != 0) {
            lf.level[2] = r.f(6)? as u8;
            lf.level[3] = r.f(6)? as u8;
        }
        lf.sharpness = r.f(3)? as u8;
        lf.delta_enabled = r.flag()?;
        if lf.delta_enabled {
            let update = r.flag()?;
            if update {
                for i in 0..8 {
                    if r.flag()? {
                        lf.ref_deltas[i] = r.su(7)? as i8;
                    }
                }
                for i in 0..2 {
                    if r.flag()? {
                        lf.mode_deltas[i] = r.su(7)? as i8;
                    }
                }
            }
        }
        Ok(())
    }

    fn parse_cdef_params(&mut self, r: &mut BitReader, seq: &SequenceHeader) -> Result<()> {
        let c = &mut self.cdef;
        if self.coded_lossless || self.allow_intrabc || !seq.enable_cdef {
            *c = CdefParams {
                damping: 3,
                ..Default::default()
            };
            return Ok(());
        }
        c.damping = r.f(2)? as u8 + 3;
        c.bits = r.f(2)? as u8;
        for i in 0..(1usize << c.bits) {
            c.y_pri_strength[i] = r.f(4)? as u8;
            c.y_sec_strength[i] = r.f(2)? as u8;
            if c.y_sec_strength[i] == 3 {
                c.y_sec_strength[i] += 1;
            }
            if seq.color.num_planes() > 1 {
                c.uv_pri_strength[i] = r.f(4)? as u8;
                c.uv_sec_strength[i] = r.f(2)? as u8;
                if c.uv_sec_strength[i] == 3 {
                    c.uv_sec_strength[i] += 1;
                }
            }
        }
        Ok(())
    }

    fn parse_lr_params(&mut self, r: &mut BitReader, seq: &SequenceHeader) -> Result<()> {
        let lr = &mut self.lr;
        *lr = LrParams::default();
        if self.all_lossless || self.allow_intrabc || !seq.enable_restoration {
            return Ok(());
        }
        let cc = &seq.color;
        for i in 0..cc.num_planes() {
            let t = match r.f(2)? {
                0 => RestorationType::None,
                1 => RestorationType::Switchable,
                2 => RestorationType::Wiener,
                _ => RestorationType::Sgrproj,
            };
            lr.frame_restoration_type[i] = t;
            if t != RestorationType::None {
                lr.uses_lr = true;
                if i > 0 {
                    lr.uses_chroma_lr = true;
                }
            }
        }
        if lr.uses_lr {
            let mut lr_unit_shift;
            if seq.use_128x128_superblock {
                lr_unit_shift = r.f(1)? + 1;
            } else {
                lr_unit_shift = r.f(1)?;
                if lr_unit_shift != 0 {
                    lr_unit_shift += r.f(1)?;
                }
            }
            lr.loop_restoration_size[0] = RESTORATION_TILESIZE_MAX >> (2 - lr_unit_shift);
            let uv_shift = if cc.subsampling_x && cc.subsampling_y && lr.uses_chroma_lr {
                r.f(1)?
            } else {
                0
            };
            lr.loop_restoration_size[1] = lr.loop_restoration_size[0] >> uv_shift;
            lr.loop_restoration_size[2] = lr.loop_restoration_size[0] >> uv_shift;
        }
        Ok(())
    }

    fn parse_film_grain_params(&mut self, r: &mut BitReader, seq: &SequenceHeader) -> Result<()> {
        let fg = &mut self.film_grain;
        *fg = FilmGrainParams::default();
        if !seq.film_grain_params_present || (!self.show_frame && !self.showable_frame) {
            return Ok(());
        }
        fg.apply_grain = r.flag()?;
        if !fg.apply_grain {
            return Ok(());
        }
        fg.grain_seed = r.f(16)? as u16;
        // update_grain = 1 for intra frames.
        fg.num_y_points = r.f(4)? as usize;
        if fg.num_y_points > 14 {
            return Err(Error::Invalid("num_y_points"));
        }
        for i in 0..fg.num_y_points {
            fg.point_y_value[i] = r.f(8)? as u8;
            fg.point_y_scaling[i] = r.f(8)? as u8;
            if i > 0 && fg.point_y_value[i] <= fg.point_y_value[i - 1] {
                return Err(Error::Invalid("point_y_value not increasing"));
            }
        }
        let cc = &seq.color;
        fg.chroma_scaling_from_luma = if cc.mono_chrome { false } else { r.flag()? };
        if cc.mono_chrome
            || fg.chroma_scaling_from_luma
            || (cc.subsampling_x && cc.subsampling_y && fg.num_y_points == 0)
        {
            fg.num_cb_points = 0;
            fg.num_cr_points = 0;
        } else {
            fg.num_cb_points = r.f(4)? as usize;
            if fg.num_cb_points > 10 {
                return Err(Error::Invalid("num_cb_points"));
            }
            for i in 0..fg.num_cb_points {
                fg.point_cb_value[i] = r.f(8)? as u8;
                fg.point_cb_scaling[i] = r.f(8)? as u8;
                if i > 0 && fg.point_cb_value[i] <= fg.point_cb_value[i - 1] {
                    return Err(Error::Invalid("point_cb_value not increasing"));
                }
            }
            fg.num_cr_points = r.f(4)? as usize;
            if fg.num_cr_points > 10 {
                return Err(Error::Invalid("num_cr_points"));
            }
            for i in 0..fg.num_cr_points {
                fg.point_cr_value[i] = r.f(8)? as u8;
                fg.point_cr_scaling[i] = r.f(8)? as u8;
                if i > 0 && fg.point_cr_value[i] <= fg.point_cr_value[i - 1] {
                    return Err(Error::Invalid("point_cr_value not increasing"));
                }
            }
        }
        fg.grain_scaling_minus_8 = r.f(2)? as u8;
        fg.ar_coeff_lag = r.f(2)? as u8;
        let lag = fg.ar_coeff_lag as usize;
        let num_pos_luma = 2 * lag * (lag + 1);
        let num_pos_chroma;
        if fg.num_y_points > 0 {
            num_pos_chroma = num_pos_luma + 1;
            for i in 0..num_pos_luma {
                fg.ar_coeffs_y_plus_128[i] = r.f(8)? as u8;
            }
        } else {
            num_pos_chroma = num_pos_luma;
        }
        if fg.chroma_scaling_from_luma || fg.num_cb_points > 0 {
            for i in 0..num_pos_chroma {
                fg.ar_coeffs_cb_plus_128[i] = r.f(8)? as u8;
            }
        }
        if fg.chroma_scaling_from_luma || fg.num_cr_points > 0 {
            for i in 0..num_pos_chroma {
                fg.ar_coeffs_cr_plus_128[i] = r.f(8)? as u8;
            }
        }
        fg.ar_coeff_shift_minus_6 = r.f(2)? as u8;
        fg.grain_scale_shift = r.f(2)? as u8;
        if fg.num_cb_points > 0 {
            fg.cb_mult = r.f(8)? as u8;
            fg.cb_luma_mult = r.f(8)? as u8;
            fg.cb_offset = r.f(9)? as u16;
        }
        if fg.num_cr_points > 0 {
            fg.cr_mult = r.f(8)? as u8;
            fg.cr_luma_mult = r.f(8)? as u8;
            fg.cr_offset = r.f(9)? as u16;
        }
        fg.overlap_flag = r.flag()?;
        fg.clip_to_restricted_range = r.flag()?;
        Ok(())
    }

    /// Spec `get_qindex(ignoreDeltaQ, segmentId)` (7.12.2).
    pub fn get_qindex(&self, ignore_delta_q: bool, segment_id: usize, current_q_index: i32) -> i32 {
        if self.seg.enabled && self.seg.feature_enabled[segment_id][SEG_LVL_ALT_Q] {
            let data = self.seg.feature_data[segment_id][SEG_LVL_ALT_Q];
            let mut qindex = self.quant.base_q_idx as i32 + data;
            if !ignore_delta_q && self.delta_q_present {
                qindex = current_q_index + data;
            }
            qindex.clamp(0, 255)
        } else if !ignore_delta_q && self.delta_q_present {
            current_q_index
        } else {
            self.quant.base_q_idx as i32
        }
    }

    /// Number of superblock columns / rows.
    pub fn sb_cols(&self, sb128: bool) -> usize {
        let s = if sb128 { 32 } else { 16 };
        self.mi_cols.div_ceil(s)
    }

    pub fn sb_rows(&self, sb128: bool) -> usize {
        let s = if sb128 { 32 } else { 16 };
        self.mi_rows.div_ceil(s)
    }
}
