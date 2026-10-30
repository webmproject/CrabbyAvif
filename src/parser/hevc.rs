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

// Minimal parser for the HEVC sequence parameter set (ISO/IEC 23008-2 / ITU-T H.265,
// Section 7.3.2.2.1), used to extract the color information from the VUI parameters.
//
// All the section references in this file are to the H.265 specification:
// https://www.itu.int/rec/T-REC-H.265-202601-I/

use crate::image::YuvRange;
use crate::internal_utils::stream::*;
use crate::internal_utils::*;
use crate::parser::mp4box::HevcCodecConfiguration;
use crate::*;

// Removes the emulation prevention bytes (0x000003 -> 0x0000) from a NAL unit. See
// emulation_prevention_three_byte in H.265 Sections 7.3.1.1 and 7.4.2.
fn nal_unit_to_rbsp(nal_unit: &[u8]) -> AvifResult<Vec<u8>> {
    let mut rbsp: Vec<u8> = create_vec_exact(nal_unit.len())?;
    let mut zero_count = 0;
    for &byte in nal_unit {
        if zero_count >= 2 && byte == 3 {
            zero_count = 0;
            continue;
        }
        zero_count = if byte == 0 { zero_count + 1 } else { 0 };
        rbsp.try_push(byte)?;
    }
    Ok(rbsp)
}

// profile_tier_level(1, sps_max_sub_layers_minus1), H.265 Section 7.3.3.
fn skip_profile_tier_level(stream: &mut IStream, max_sub_layers_minus1: usize) -> AvifResult<()> {
    // general_profile_space .. general_level_idc.
    stream.skip_bits(2 + 1 + 5 + 32 + 4 + 43 + 1 + 8)?;
    let mut sub_layer_profile_present_flag = [false; 8];
    let mut sub_layer_level_present_flag = [false; 8];
    for i in 0..max_sub_layers_minus1 {
        sub_layer_profile_present_flag[i] = stream.read_bool()?;
        sub_layer_level_present_flag[i] = stream.read_bool()?;
    }
    if max_sub_layers_minus1 > 0 {
        // reserved_zero_2bits.
        stream.skip_bits(2 * (8 - max_sub_layers_minus1))?;
    }
    for i in 0..max_sub_layers_minus1 {
        if sub_layer_profile_present_flag[i] {
            stream.skip_bits(2 + 1 + 5 + 32 + 4 + 43 + 1)?;
        }
        if sub_layer_level_present_flag[i] {
            stream.skip_bits(8)?;
        }
    }
    Ok(())
}

// scaling_list_data(), H.265 Section 7.3.4.
fn skip_scaling_list_data(stream: &mut IStream) -> AvifResult<()> {
    for size_id in 0..4 {
        let step = if size_id == 3 { 3 } else { 1 };
        for _matrix_id in (0..6).step_by(step) {
            let scaling_list_pred_mode_flag = stream.read_bool()?;
            if !scaling_list_pred_mode_flag {
                stream.skip_uvlc()?; // scaling_list_pred_matrix_id_delta
            } else {
                let coef_num = std::cmp::min(64, 1 << (4 + (size_id << 1)));
                if size_id > 1 {
                    stream.skip_uvlc()?; // scaling_list_dc_coef_minus8 (se(v))
                }
                for _ in 0..coef_num {
                    stream.skip_uvlc()?; // scaling_list_delta_coef (se(v))
                }
            }
        }
    }
    Ok(())
}

// st_ref_pic_set(st_rps_idx) in the SPS, H.265 Section 7.3.7. Returns NumDeltaPocs[st_rps_idx].
fn skip_st_ref_pic_set(
    stream: &mut IStream,
    st_rps_idx: usize,
    num_delta_pocs: &[u32],
) -> AvifResult<u32> {
    let inter_ref_pic_set_prediction_flag = st_rps_idx != 0 && stream.read_bool()?;
    if inter_ref_pic_set_prediction_flag {
        // delta_idx_minus1 is only present in slice headers, so RefRpsIdx = st_rps_idx - 1.
        stream.skip_bits(1)?; // delta_rps_sign
        stream.skip_uvlc()?; // abs_delta_rps_minus1
        let mut count = 0;
        for _j in 0..=num_delta_pocs[st_rps_idx - 1] {
            let used_by_curr_pic_flag = stream.read_bool()?;
            let use_delta_flag = used_by_curr_pic_flag || stream.read_bool()?;
            if use_delta_flag {
                count += 1;
            }
        }
        Ok(count)
    } else {
        // Both values are bounded by sps_max_dec_pic_buffering_minus1 (at most 15).
        let num_negative_pics = stream.read_uvlc_max(16)?;
        let num_positive_pics = stream.read_uvlc_max(16)?;
        for _ in 0..(num_negative_pics + num_positive_pics) {
            stream.skip_uvlc()?; // delta_poc_s{0,1}_minus1
            stream.skip_bits(1)?; // used_by_curr_pic_s{0,1}_flag
        }
        Ok(num_negative_pics + num_positive_pics)
    }
}

// seq_parameter_set_rbsp(), H.265 Section 7.3.2.2.1.
// Parses the HEVC SPS NAL unit in config.sps (including its 2-byte NAL unit header) and sets the
// color information signaled in (or inferred from the absence of) its VUI parameters in config.
// The color information in config is not modified if parsing fails.
pub(crate) fn parse_sps_color_info(config: &mut HevcCodecConfiguration) -> AvifResult<()> {
    let rbsp = nal_unit_to_rbsp(&config.sps)?;
    let mut stream = IStream::create(&rbsp);
    stream.skip_bits(16)?; // nal_unit_header(), H.265 Section 7.3.1.2.
    stream.skip_bits(4)?; // sps_video_parameter_set_id
    let max_sub_layers_minus1 = stream.read_bits(3)? as usize;
    if max_sub_layers_minus1 > 6 {
        return AvifError::bmff_parse_failed("invalid sps_max_sub_layers_minus1");
    }
    stream.skip_bits(1)?; // sps_temporal_id_nesting_flag
    skip_profile_tier_level(&mut stream, max_sub_layers_minus1)?;
    stream.skip_uvlc()?; // sps_seq_parameter_set_id
    let chroma_format_idc = stream.read_uvlc_max(3)?;
    if chroma_format_idc == 3 {
        stream.skip_bits(1)?; // separate_colour_plane_flag
    }
    stream.skip_uvlc()?; // pic_width_in_luma_samples
    stream.skip_uvlc()?; // pic_height_in_luma_samples
    if stream.read_bool()? {
        // conformance_window_flag
        for _ in 0..4 {
            stream.skip_uvlc()?; // conf_win_{left,right,top,bottom}_offset
        }
    }
    stream.skip_uvlc()?; // bit_depth_luma_minus8
    stream.skip_uvlc()?; // bit_depth_chroma_minus8
    let log2_max_pic_order_cnt_lsb_minus4 = stream.read_uvlc_max(12)?;
    let sps_sub_layer_ordering_info_present_flag = stream.read_bool()?;
    let first = if sps_sub_layer_ordering_info_present_flag { 0 } else { max_sub_layers_minus1 };
    for _ in first..=max_sub_layers_minus1 {
        stream.skip_uvlc()?; // sps_max_dec_pic_buffering_minus1
        stream.skip_uvlc()?; // sps_max_num_reorder_pics
        stream.skip_uvlc()?; // sps_max_latency_increase_plus1
    }
    for _ in 0..6 {
        // log2_min_luma_coding_block_size_minus3, log2_diff_max_min_luma_coding_block_size,
        // log2_min_luma_transform_block_size_minus2, log2_diff_max_min_luma_transform_block_size,
        // max_transform_hierarchy_depth_inter, max_transform_hierarchy_depth_intra
        stream.skip_uvlc()?;
    }
    if stream.read_bool()? && stream.read_bool()? {
        // scaling_list_enabled_flag && sps_scaling_list_data_present_flag
        skip_scaling_list_data(&mut stream)?;
    }
    stream.skip_bits(1)?; // amp_enabled_flag
    stream.skip_bits(1)?; // sample_adaptive_offset_enabled_flag
    if stream.read_bool()? {
        // pcm_enabled_flag
        // pcm_sample_bit_depth_luma_minus1, pcm_sample_bit_depth_chroma_minus1
        stream.skip_bits(4 + 4)?;
        stream.skip_uvlc()?; // log2_min_pcm_luma_coding_block_size_minus3
        stream.skip_uvlc()?; // log2_diff_max_min_pcm_luma_coding_block_size
        stream.skip_bits(1)?; // pcm_loop_filter_disabled_flag
    }
    let num_short_term_ref_pic_sets = stream.read_uvlc_max(64)? as usize;
    let mut num_delta_pocs: Vec<u32> = create_vec_exact(num_short_term_ref_pic_sets)?;
    for i in 0..num_short_term_ref_pic_sets {
        let count = skip_st_ref_pic_set(&mut stream, i, &num_delta_pocs)?;
        num_delta_pocs.try_push(count)?;
    }
    if stream.read_bool()? {
        // long_term_ref_pics_present_flag
        let num_long_term_ref_pics_sps = stream.read_uvlc_max(32)?;
        for _ in 0..num_long_term_ref_pics_sps {
            // lt_ref_pic_poc_lsb_sps, used_by_curr_pic_lt_sps_flag
            stream.skip_bits(log2_max_pic_order_cnt_lsb_minus4 as usize + 4 + 1)?;
        }
    }
    stream.skip_bits(1)?; // sps_temporal_mvp_enabled_flag
    stream.skip_bits(1)?; // strong_intra_smoothing_enabled_flag

    // Values inferred by H.265 Section E.3.1 when the corresponding syntax elements are absent.
    let mut color_primaries = ColorPrimaries::Unspecified;
    let mut transfer_characteristics = TransferCharacteristics::Unspecified;
    let mut matrix_coefficients = MatrixCoefficients::Unspecified;
    let mut yuv_range = YuvRange::Limited;
    if stream.read_bool()? {
        // vui_parameters_present_flag
        // vui_parameters(), H.265 Section E.2.1, up to and including the colour description.
        if stream.read_bool()? {
            // aspect_ratio_info_present_flag
            let aspect_ratio_idc = stream.read_bits(8)?;
            if aspect_ratio_idc == 255 {
                // EXTENDED_SAR
                stream.skip_bits(16 + 16)?; // sar_width, sar_height
            }
        }
        if stream.read_bool()? {
            // overscan_info_present_flag
            stream.skip_bits(1)?; // overscan_appropriate_flag
        }
        if stream.read_bool()? {
            // video_signal_type_present_flag
            stream.skip_bits(3)?; // video_format
            yuv_range = if stream.read_bool()? { YuvRange::Full } else { YuvRange::Limited };
            if stream.read_bool()? {
                // colour_description_present_flag
                color_primaries = (stream.read_bits(8)? as u16).into();
                transfer_characteristics = (stream.read_bits(8)? as u16).into();
                matrix_coefficients = (stream.read_bits(8)? as u16).into();
            }
        }
    }
    config.color_primaries = color_primaries;
    config.transfer_characteristics = transfer_characteristics;
    config.matrix_coefficients = matrix_coefficients;
    config.yuv_range = yuv_range;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rbsp_removes_emulation_prevention() {
        assert_eq!(
            nal_unit_to_rbsp(&[0, 0, 3, 1, 0, 0, 3, 0, 3]).unwrap(),
            vec![0, 0, 1, 0, 0, 0, 3]
        );
    }

    type ColorInfo = (
        ColorPrimaries,
        TransferCharacteristics,
        MatrixCoefficients,
        YuvRange,
    );

    fn color_info(sps: &[u8]) -> AvifResult<ColorInfo> {
        let mut config = HevcCodecConfiguration {
            sps: sps.to_vec(),
            ..Default::default()
        };
        parse_sps_color_info(&mut config)?;
        Ok((
            config.color_primaries,
            config.transfer_characteristics,
            config.matrix_coefficients,
            config.yuv_range,
        ))
    }

    const UNSPECIFIED_LIMITED: ColorInfo = (
        ColorPrimaries::Unspecified,
        TransferCharacteristics::Unspecified,
        MatrixCoefficients::Unspecified,
        YuvRange::Limited,
    );

    const UNSPECIFIED_FULL: ColorInfo = (
        ColorPrimaries::Unspecified,
        TransferCharacteristics::Unspecified,
        MatrixCoefficients::Unspecified,
        YuvRange::Full,
    );

    #[test]
    fn sps_color_info() {
        // SPS with a VUI with video_full_range_flag=0.
        assert_eq!(
            color_info(&[
                0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03,
                0x00, 0x00, 0x03, 0x00, 0x78, 0xa0, 0x02, 0x80, 0x80, 0x2d, 0x1f, 0xe5, 0xf9, 0x24,
                0x6d, 0x9e, 0xd9
            ])
            .unwrap(),
            UNSPECIFIED_LIMITED
        );
        // SPS of tests/data/heic/blue_gh_issue_692.heic: VUI with video_full_range_flag=1.
        assert_eq!(
            color_info(&[
                0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00, 0x03, 0x00,
                0x00, 0x03, 0x00, 0x3c, 0xa0, 0x0a, 0x08, 0x0f, 0x16, 0x59, 0x98, 0xa9, 0x24, 0x6d,
                0xae, 0x6d, 0xc2, 0x00, 0x00, 0x03, 0x00, 0x02, 0x00, 0x00, 0x03, 0x00, 0x02, 0x10
            ])
            .unwrap(),
            UNSPECIFIED_FULL
        );
        // SPS of tests/data/heic/blue.heic: VUI with full range and no colour description.
        assert_eq!(
            color_info(&[
                0x42, 0x01, 0x01, 0x03, 0x70, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00, 0x03, 0x00,
                0x00, 0x03, 0x00, 0x3c, 0xa0, 0x0a, 0x08, 0x0f, 0x16, 0x5b, 0xa9, 0x24, 0xa6, 0xb9,
                0xb0, 0x20, 0x00, 0x00, 0x03, 0x00, 0x20, 0x00, 0x00, 0x03, 0x00, 0x21
            ])
            .unwrap(),
            UNSPECIFIED_FULL
        );
        // SPS of tests/data/heic/yuv420_image_with_yuv400_gainmap.heic: scaling list data and a
        // VUI with full range and a colour description (12, 1, 6).
        assert_eq!(
            color_info(&[
                0x42, 0x01, 0x01, 0x03, 0x70, 0x00, 0x00, 0x03, 0x00, 0xb0, 0x00, 0x00, 0x03, 0x00,
                0x00, 0x03, 0x00, 0x3c, 0xa0, 0x0a, 0x08, 0x0f, 0x16, 0x21, 0xee, 0x45, 0xe1, 0x0f,
                0x8a, 0x3c, 0xc2, 0x06, 0x0d, 0x06, 0x90, 0x3a, 0x42, 0x1f, 0x14, 0x79, 0x84, 0x0c,
                0x1a, 0x0d, 0x20, 0x74, 0x84, 0x3e, 0x28, 0xf3, 0x08, 0x18, 0x34, 0x1a, 0x40, 0xe9,
                0x08, 0x7c, 0x9c, 0x56, 0x10, 0x83, 0xa8, 0x43, 0xe4, 0xe2, 0xb0, 0x84, 0x1d, 0x42,
                0x1f, 0x27, 0x15, 0x84, 0x20, 0xea, 0x10, 0xf9, 0x1d, 0x18, 0xe9, 0x98, 0x50, 0xbe,
                0x28, 0x28, 0x11, 0x7a, 0x08, 0x07, 0x03, 0x22, 0x55, 0x88, 0x0c, 0x18, 0x23, 0x31,
                0xe1, 0x00, 0xb1, 0xc4, 0x44, 0x98, 0x16, 0x18, 0x40, 0x93, 0x81, 0x38, 0x29, 0x0a,
                0x80, 0x92, 0x09, 0x04, 0xc0, 0xe5, 0xc2, 0x1f, 0x23, 0xa3, 0x1d, 0x33, 0x0a, 0x17,
                0xc5, 0x05, 0x02, 0x2f, 0x41, 0x00, 0xe0, 0x64, 0x4a, 0xb1, 0x01, 0x83, 0x04, 0x66,
                0x3c, 0x20, 0x16, 0x38, 0x88, 0x93, 0x02, 0xc3, 0x08, 0x12, 0x70, 0x27, 0x05, 0x21,
                0x50, 0x12, 0x41, 0x20, 0x98, 0x1c, 0xb8, 0x43, 0xe4, 0x74, 0x63, 0xa6, 0x61, 0x42,
                0xf8, 0xa0, 0xa0, 0x45, 0xe8, 0x20, 0x1c, 0x0c, 0x89, 0x56, 0x20, 0x30, 0x60, 0x8c,
                0xc7, 0x84, 0x02, 0xc7, 0x11, 0x12, 0x60, 0x58, 0x61, 0x02, 0x4e, 0x04, 0xe0, 0xa4,
                0x2a, 0x02, 0x48, 0x24, 0x13, 0x03, 0x97, 0x08, 0x7f, 0xd7, 0xaf, 0x93, 0xf1, 0x1f,
                0xd7, 0xe6, 0xf8, 0xaf, 0x09, 0x70, 0x49, 0x83, 0x08, 0x3c, 0x84, 0x3f, 0xeb, 0xd7,
                0xc9, 0xf8, 0x8f, 0xeb, 0xf3, 0x7c, 0x57, 0x84, 0xb8, 0x24, 0xc1, 0x84, 0x1e, 0x42,
                0x1f, 0xf5, 0xeb, 0xe4, 0xfc, 0x47, 0xf5, 0xf9, 0xbe, 0x2b, 0xc2, 0x5c, 0x12, 0x60,
                0xc2, 0x0f, 0x21, 0x0f, 0xc8, 0xe8, 0xc7, 0x4c, 0xc2, 0x85, 0xf1, 0x41, 0x40, 0x8b,
                0xd0, 0x40, 0x38, 0x19, 0x12, 0xac, 0x40, 0x60, 0xc1, 0x19, 0x8f, 0x08, 0x05, 0x8e,
                0x22, 0x24, 0xc0, 0xb0, 0xc2, 0x04, 0x9c, 0x09, 0xc1, 0x48, 0x54, 0x04, 0x90, 0x48,
                0x26, 0x07, 0x2e, 0x10, 0xfc, 0x8e, 0x8c, 0x74, 0xcc, 0x28, 0x5f, 0x14, 0x14, 0x08,
                0xbd, 0x04, 0x03, 0x81, 0x91, 0x2a, 0xc4, 0x06, 0x0c, 0x11, 0x98, 0xf0, 0x80, 0x58,
                0xe2, 0x22, 0x4c, 0x0b, 0x0c, 0x20, 0x49, 0xc0, 0x9c, 0x14, 0x85, 0x40, 0x49, 0x04,
                0x82, 0x60, 0x72, 0xe1, 0x0f, 0xc8, 0xe8, 0xc7, 0x4c, 0xc2, 0x85, 0xf1, 0x41, 0x40,
                0x8b, 0xd0, 0x40, 0x38, 0x19, 0x12, 0xac, 0x40, 0x60, 0xc1, 0x19, 0x8f, 0x08, 0x05,
                0x8e, 0x22, 0x24, 0xc0, 0xb0, 0xc2, 0x04, 0x9c, 0x09, 0xc1, 0x48, 0x54, 0x04, 0x90,
                0x48, 0x26, 0x07, 0x2e, 0x10, 0xff, 0xd7, 0xaf, 0x93, 0xf1, 0x1f, 0xd7, 0xe6, 0xf8,
                0xaf, 0x09, 0x70, 0x49, 0x83, 0x08, 0x3c, 0x84, 0x3f, 0xf5, 0xeb, 0xe4, 0xfc, 0x47,
                0xf5, 0xf9, 0xbe, 0x2b, 0xc2, 0x5c, 0x12, 0x60, 0xc2, 0x0f, 0x21, 0x0f, 0xfd, 0x7a,
                0xf9, 0x3f, 0x11, 0xfd, 0x7e, 0x6f, 0x8a, 0xf0, 0x97, 0x04, 0x98, 0x30, 0x83, 0xc8,
                0x43, 0xf2, 0x3a, 0x31, 0xd3, 0x30, 0xa1, 0x7c, 0x50, 0x50, 0x22, 0xf4, 0x10, 0x0e,
                0x06, 0x44, 0xab, 0x10, 0x18, 0x30, 0x46, 0x63, 0xc2, 0x01, 0x63, 0x88, 0x89, 0x30,
                0x2c, 0x30, 0x81, 0x27, 0x02, 0x70, 0x52, 0x15, 0x01, 0x24, 0x12, 0x09, 0x81, 0xcb,
                0x84, 0x3f, 0xf5, 0xeb, 0xe4, 0xfc, 0x47, 0xf5, 0xf9, 0xbe, 0x2b, 0xc2, 0x5c, 0x12,
                0x60, 0xc2, 0x0f, 0x15, 0x4d, 0xc3, 0x00, 0x41, 0x80, 0x20
            ])
            .unwrap(),
            (
                ColorPrimaries::Smpte432,
                TransferCharacteristics::Bt709,
                MatrixCoefficients::Bt601,
                YuvRange::Full,
            )
        );
        // Truncated SPS.
        assert!(color_info(&[0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03]).is_err());
    }
}
