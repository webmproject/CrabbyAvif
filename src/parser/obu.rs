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

use crate::image::YuvRange;
use crate::internal_utils::stream::*;
use crate::internal_utils::*;
use crate::parser::mp4box::*;
use crate::*;

#[derive(Debug)]
struct ObuHeader {
    obu_type: u8,
    size: u32,
}

#[derive(Debug, Default)]
pub struct Av1SequenceHeader {
    reduced_still_picture_header: bool,
    max_width: u32,
    max_height: u32,
    bit_depth: u8,
    yuv_format: PixelFormat,
    pub color_primaries: ColorPrimaries,
    pub transfer_characteristics: TransferCharacteristics,
    pub matrix_coefficients: MatrixCoefficients,
    pub yuv_range: YuvRange,
    pub config: Av1CodecConfiguration,
}

impl Av1SequenceHeader {
    fn parse_profile(&mut self, stream: &mut IStream) -> AvifResult<()> {
        self.config.seq_profile = stream.read_bits(3)? as u8;
        if self.config.seq_profile > 2 {
            return AvifError::bmff_parse_failed("invalid seq_profile");
        }
        let still_picture = stream.read_bool()?;
        self.reduced_still_picture_header = stream.read_bool()?;
        if self.reduced_still_picture_header && !still_picture {
            return AvifError::bmff_parse_failed("invalid reduced_still_picture_header");
        }
        if self.reduced_still_picture_header {
            self.config.seq_level_idx0 = stream.read_bits(5)? as u8;
        } else {
            let mut buffer_delay_length = 0;
            let mut decoder_model_info_present_flag = false;
            let timing_info_present_flag = stream.read_bool()?;
            if timing_info_present_flag {
                // num_units_in_display_tick
                stream.skip_bits(32)?;
                // time_scale
                stream.skip_bits(32)?;
                let equal_picture_interval = stream.read_bool()?;
                if equal_picture_interval {
                    // num_ticks_per_picture_minus_1
                    stream.skip_uvlc()?;
                }
                decoder_model_info_present_flag = stream.read_bool()?;
                if decoder_model_info_present_flag {
                    let buffer_delay_length_minus_1 = stream.read_bits(5)?;
                    buffer_delay_length = buffer_delay_length_minus_1 + 1;
                    // num_units_in_decoding_tick
                    stream.skip_bits(32)?;
                    // buffer_removal_time_length_minus_1
                    stream.skip_bits(5)?;
                    // frame_presentation_time_length_minus_1
                    stream.skip_bits(5)?;
                }
            }
            let initial_display_delay_present_flag = stream.read_bool()?;
            let operating_points_cnt_minus_1 = stream.read_bits(5)?;
            let operating_points_cnt = operating_points_cnt_minus_1 + 1;
            for i in 0..operating_points_cnt {
                // operating_point_idc
                stream.skip_bits(12)?;
                let seq_level_idx = stream.read_bits(5)?;
                if i == 0 {
                    self.config.seq_level_idx0 = seq_level_idx as u8;
                }
                if seq_level_idx > 7 {
                    let seq_tier = stream.read_bits(1)?;
                    if i == 0 {
                        self.config.seq_tier0 = seq_tier as u8;
                    }
                }
                if decoder_model_info_present_flag {
                    let decoder_model_present_for_this_op = stream.read_bool()?;
                    if decoder_model_present_for_this_op {
                        // decoder_buffer_delay
                        stream.skip_bits(buffer_delay_length as usize)?;
                        // encoder_buffer_delay
                        stream.skip_bits(buffer_delay_length as usize)?;
                        // low_delay_mode_flag
                        stream.skip_bits(1)?;
                    }
                }
                if initial_display_delay_present_flag {
                    let initial_display_delay_present_for_this_op = stream.read_bool()?;
                    if initial_display_delay_present_for_this_op {
                        // initial_display_delay_minus_1
                        stream.skip_bits(4)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn parse_frame_max_dimensions(&mut self, stream: &mut IStream) -> AvifResult<()> {
        let frame_width_bits_minus_1 = stream.read_bits(4)?;
        let frame_height_bits_minus_1 = stream.read_bits(4)?;
        let max_frame_width_minus_1 = stream.read_bits(frame_width_bits_minus_1 as usize + 1)?;
        let max_frame_height_minus_1 = stream.read_bits(frame_height_bits_minus_1 as usize + 1)?;
        self.max_width = checked_add!(max_frame_width_minus_1, 1)?;
        self.max_height = checked_add!(max_frame_height_minus_1, 1)?;
        let frame_id_numbers_present_flag =
            if self.reduced_still_picture_header { false } else { stream.read_bool()? };
        if frame_id_numbers_present_flag {
            // delta_frame_id_length_minus_2
            stream.skip_bits(4)?;
            // additional_frame_id_length_minus_1
            stream.skip_bits(3)?;
        }
        Ok(())
    }

    fn parse_enabled_features(&mut self, stream: &mut IStream) -> AvifResult<()> {
        // use_128x128_superblock
        stream.skip_bits(1)?;
        // enable_filter_intra
        stream.skip_bits(1)?;
        // enable_intra_edge_filter
        stream.skip_bits(1)?;
        if self.reduced_still_picture_header {
            return Ok(());
        }
        // enable_interintra_compound
        stream.skip_bits(1)?;
        // enable_masked_compound
        stream.skip_bits(1)?;
        // enable_warped_motion
        stream.skip_bits(1)?;
        // enable_dual_filter
        stream.skip_bits(1)?;
        let enable_order_hint = stream.read_bool()?;
        if enable_order_hint {
            // enable_jnt_comp
            stream.skip_bits(1)?;
            // enable_ref_frame_mvs
            stream.skip_bits(1)?;
        }
        let seq_choose_screen_content_tools = stream.read_bool()?;
        let seq_force_screen_content_tools = if seq_choose_screen_content_tools {
            2 // SELECT_SCREEN_CONTENT_TOOLS
        } else {
            stream.read_bits(1)?
        };
        if seq_force_screen_content_tools > 0 {
            let seq_choose_integer_mv = stream.read_bool()?;
            if !seq_choose_integer_mv {
                // seq_force_integer_mv
                stream.skip_bits(1)?;
            }
        }
        if enable_order_hint {
            // order_hint_bits_minus_1
            stream.skip_bits(3)?;
        }
        Ok(())
    }

    fn parse_color_config(&mut self, stream: &mut IStream) -> AvifResult<()> {
        self.config.high_bitdepth = stream.read_bool()?;
        if self.config.seq_profile == 2 && self.config.high_bitdepth {
            self.config.twelve_bit = stream.read_bool()?;
            self.bit_depth = if self.config.twelve_bit { 12 } else { 10 };
        } else {
            self.bit_depth = if self.config.high_bitdepth { 10 } else { 8 };
        }
        if self.config.seq_profile != 1 {
            self.config.monochrome = stream.read_bool()?;
        }
        let color_description_present_flag = stream.read_bool()?;
        if color_description_present_flag {
            self.color_primaries = (stream.read_bits(8)? as u16).into();
            self.transfer_characteristics = (stream.read_bits(8)? as u16).into();
            self.matrix_coefficients = (stream.read_bits(8)? as u16).into();
        } else {
            self.color_primaries = ColorPrimaries::Unspecified;
            self.transfer_characteristics = TransferCharacteristics::Unspecified;
            self.matrix_coefficients = MatrixCoefficients::Unspecified;
        }
        if self.config.monochrome {
            let color_range = stream.read_bool()?;
            self.yuv_range = if color_range { YuvRange::Full } else { YuvRange::Limited };
            self.config.chroma_subsampling_x = 1;
            self.config.chroma_subsampling_y = 1;
            self.yuv_format = PixelFormat::Yuv400;
            return Ok(());
        } else if self.color_primaries == ColorPrimaries::Bt709
            && self.transfer_characteristics == TransferCharacteristics::Srgb
            && self.matrix_coefficients == MatrixCoefficients::Identity
        {
            self.yuv_range = YuvRange::Full;
            self.yuv_format = PixelFormat::Yuv444;
        } else {
            let color_range = stream.read_bool()?;
            self.yuv_range = if color_range { YuvRange::Full } else { YuvRange::Limited };
            match self.config.seq_profile {
                0 => {
                    self.config.chroma_subsampling_x = 1;
                    self.config.chroma_subsampling_y = 1;
                    self.yuv_format = PixelFormat::Yuv420;
                }
                1 => {
                    self.yuv_format = PixelFormat::Yuv444;
                }
                2 => {
                    if self.bit_depth == 12 {
                        self.config.chroma_subsampling_x = stream.read_bits(1)? as u8;
                        if self.config.chroma_subsampling_x == 1 {
                            self.config.chroma_subsampling_y = stream.read_bits(1)? as u8;
                        }
                    } else {
                        self.config.chroma_subsampling_x = 1;
                    }
                    self.yuv_format = if self.config.chroma_subsampling_x == 1 {
                        if self.config.chroma_subsampling_y == 1 {
                            PixelFormat::Yuv420
                        } else {
                            PixelFormat::Yuv422
                        }
                    } else {
                        PixelFormat::Yuv444
                    };
                }
                _ => {} // Not reached.
            }
            if self.config.chroma_subsampling_x == 1 && self.config.chroma_subsampling_y == 1 {
                // chroma_sample_position.
                stream.skip_bits(2)?;
            }
        }
        // separate_uv_delta_q
        stream.skip_bits(1)?;
        Ok(())
    }

    fn parse_obu_header(stream: &mut IStream) -> AvifResult<ObuHeader> {
        // Section 5.3.2 of AV1 specification.
        // https://aomediacodec.github.io/av1-spec/#obu-header-syntax
        let obu_forbidden_bit = stream.read_bits(1)?;
        if obu_forbidden_bit != 0 {
            return AvifError::bmff_parse_failed("invalid obu_forbidden_bit");
        }
        let obu_type = stream.read_bits(4)? as u8;
        let obu_extension_flag = stream.read_bool()?;
        let obu_has_size_field = stream.read_bool()?;
        // obu_reserved_1bit
        stream.skip_bits(1)?; // "The value is ignored by a decoder."

        if obu_extension_flag {
            // temporal_id
            stream.skip_bits(3)?;
            // spatial_id
            stream.skip_bits(2)?;
            // extension_header_reserved_3bits
            stream.skip_bits(3)?;
        }

        let size = if obu_has_size_field {
            stream.read_uleb128()?
        } else {
            u32_from_usize(stream.bytes_left()?)? // sz - 1 - obu_extension_flag
        };

        Ok(ObuHeader { obu_type, size })
    }

    pub(crate) fn parse_from_obus(data: &[u8]) -> AvifResult<Self> {
        let mut stream = IStream::create(data);

        while stream.has_bytes_left()? {
            let obu = Self::parse_obu_header(&mut stream)?;
            if obu.obu_type != /*OBU_SEQUENCE_HEADER=*/1 {
                // Not a sequence header. Skip this obu.
                stream.skip(usize_from_u32(obu.size)?)?;
                continue;
            }
            let mut stream = stream.sub_stream(&BoxSize::FixedSize(usize_from_u32(obu.size)?))?;
            let mut sequence_header = Av1SequenceHeader::default();
            sequence_header.parse_profile(&mut stream)?;
            sequence_header.parse_frame_max_dimensions(&mut stream)?;
            sequence_header.parse_enabled_features(&mut stream)?;
            // enable_superres
            stream.skip_bits(1)?;
            // enable_cdef
            stream.skip_bits(1)?;
            // enable_restoration
            stream.skip_bits(1)?;
            sequence_header.parse_color_config(&mut stream)?;
            // film_grain_params_present
            stream.skip_bits(1)?;
            return Ok(sequence_header);
        }
        AvifError::bmff_parse_failed("could not parse sequence header")
    }
}

#[cfg(feature = "avm")]
impl Av2CodecConfiguration {
    // Returns the OBU type on success.
    fn parse_av2_obu_header(stream: &mut IStream) -> AvifResult<u8> {
        // https://av2.aomedia.org/v1.0.0/index.html#obu_header_syntax
        let obu_header_extension_flag = stream.read_bits(1)? as usize;
        let obu_type = stream.read_bits(5)? as u8;
        let _obu_tlayer_id = stream.read_bits(2)?;
        if obu_header_extension_flag == 1 {
            let _obu_mlayer_id = stream.read_bits(3)?;
            let _obu_xlayer_id = stream.read_bits(5)?;
        }
        Ok(obu_type)
    }

    // Returns the profile, pixel format and depth on success.
    fn parse_av2_sequence_header_obu(stream: &mut IStream) -> AvifResult<(u8, PixelFormat, u8)> {
        // https://av2.aomedia.org/v1.0.0/index.html#general_sequence_header_obu_syntax
        let _seq_header_id = stream.read_uvlc()?;
        let seq_profile_idc = stream.read_bits(5)?;
        let single_picture_header_flag = stream.read_bool()?;
        let seq_level_idx = stream.read_bits(5)?;
        let _seq_tier = if seq_level_idx > 3 && !single_picture_header_flag {
            stream.read_bits(1)?
        } else {
            0
        };
        let chroma_format_idc = stream.read_uvlc()?;
        let pixel_format = match chroma_format_idc {
            // https://av2.aomedia.org/v1.0.0/index.html#table-chroma-format
            0 => PixelFormat::Yuv420,
            1 => PixelFormat::Yuv400,
            2 => PixelFormat::Yuv444,
            3 => PixelFormat::Yuv422,
            _ => return AvifError::bmff_parse_failed("It is a requirement of bitstream conformance that chroma_format_idc is less than or equal to 3."),
        };
        let bit_depth_idc = stream.read_uvlc()?;
        let depth = match bit_depth_idc {
            // https://av2.aomedia.org/v1.0.0/index.html#table-bit-depth
            0 => 10,
            1 => 8,
            _ => return AvifError::bmff_parse_failed("It is a requirement of bitstream conformance that bit_depth_idc is less than or equal to 1."),
        };

        // TODO(b/437292541): Parse all other fields of sequence_header_obu()?
        Ok((seq_profile_idc as u8, pixel_format, depth))
    }

    // Returns the chroma sample position on success.
    fn parse_av2_content_interpretation_obu(
        stream: &mut IStream,
    ) -> AvifResult<ChromaSamplePosition> {
        use crate::codecs::avm::*;

        // https://av2.aomedia.org/v1.0.0/index.html#content_interpretation_obu_syntax
        let ci_scan_type_idc = stream.read_bits(2)?;
        let ci_color_description_present_flag = stream.read_bool()?;
        let ci_chroma_sample_position_present_flag = stream.read_bool()?;
        let _ci_aspect_ratio_info_present_flag = stream.read_bool()?;
        let _ci_timing_info_present_flag = stream.read_bool()?;
        let _ci_reserved_2bit = stream.read_bits(2)?;

        if ci_color_description_present_flag {
            let ci_color_description_idc = stream.read_rg(2)?;
            if ci_color_description_idc == 0 {
                let _ci_color_primaries = stream.read_bits(8)?;
                let _ci_transfer_characteristics = stream.read_bits(8)?;
                let _ci_matrix_coefficients = stream.read_bits(8)?;
            }
            let _ci_full_range_flag = stream.read_bool()?;
        }

        if ci_chroma_sample_position_present_flag {
            let ci_chroma_sample_position_top = stream.read_uvlc()?;
            let _ci_chroma_sample_position_bottom = if ci_scan_type_idc != 1 {
                stream.read_uvlc()?
            } else {
                ci_chroma_sample_position_top
            };
            return Ok(match ci_chroma_sample_position_top {
                // Horizontal offset 0, vertical offset 0.5
                AV2_CSP_LEFT => ChromaSamplePosition::Vertical,
                // Horizontal offset 0.5, vertical offset 0.5
                AV2_CSP_CENTER => ChromaSamplePosition::Unknown,
                // Horizontal offset 0, vertical offset 0
                AV2_CSP_TOPLEFT => ChromaSamplePosition::Colocated,
                // TODO(b/437292541): Implement other values in CrabbyAvif?
                _ => ChromaSamplePosition::Unknown,
            });
            // TODO(b/437292541): What to do with _ci_chroma_sample_position_bottom?
        }

        // TODO(b/437292541): Parse all other fields of content_interpretation_obu()?
        Ok(ChromaSamplePosition::Unknown)
    }

    fn parse_av2_num_bytes_in_obu(stream: &mut IStream) -> AvifResult<usize> {
        // https://av2.aomedia.org/v1.0.0/index.html#length_delimited_bitstream_syntax
        let num_bytes_in_obu = stream.read_leb128()? as usize; // At most (1 << 32) - 1.
        if num_bytes_in_obu > stream.bytes_left()? {
            return AvifError::bmff_parse_failed(format!(
                "AV2 num_bytes_in_obu {} exceeds the {} bytes left",
                num_bytes_in_obu,
                stream.bytes_left()?
            ));
        }
        Ok(num_bytes_in_obu)
    }

    fn get_config_and_sample_obus_range(
        obus: &[u8],
    ) -> AvifResult<((usize, usize), (usize, usize))> {
        use crate::codecs::avm::*;
        fn get_one_obu(obus: &[u8], offset: usize) -> AvifResult<(usize, u8)> {
            let mut stream = IStream::create(&obus[offset..]);
            let num_bytes_in_obu = Av2CodecConfiguration::parse_av2_num_bytes_in_obu(&mut stream)?;
            let num_bytes_in_obu_length = stream.offset;
            let obu_type = Av2CodecConfiguration::parse_av2_obu_header(
                &mut stream.sub_stream(&BoxSize::FixedSize(num_bytes_in_obu))?,
            )?;
            Ok((num_bytes_in_obu_length + num_bytes_in_obu, obu_type))
        }

        let mut offset = 0;

        // Skip any Temporal Delimiter OBU.
        // A temporal delimiter cannot be part of the AV2CodecConfigurationBox config_obus
        // and cannot be part of any sample data.
        // TODO(b/437292541): Are all TD OBUs mandatorily here? Is there at least one? At most one?
        while offset < obus.len() {
            let (obu_length, obu_type) = get_one_obu(obus, offset)?;
            if !matches!(obu_type, AV2_OBU_TEMPORAL_DELIMITER) {
                break;
            }
            offset += obu_length;
        }

        // Iterate over the OBUs that will go into the AV2CodecConfigurationBox config_obus.
        let config_obus_start = offset;
        while offset < obus.len() {
            let (obu_length, obu_type) = get_one_obu(obus, offset)?;
            if matches!(
                obu_type,
                // These are explicitly forbidden in the AV2CodecConfigurationBox config_obus.
                // TODO(b/437292541): What is a switch-eligible OBU? See
                //                    https://aomediacodec.github.io/av2-isobmff/pr/40/index.html#configobus
                AV2_OBU_CLOSED_LOOP_KEY
                    | AV2_OBU_OPEN_LOOP_KEY
                    | AV2_OBU_LEADING_TILE_GROUP
                    | AV2_OBU_REGULAR_TILE_GROUP
                    | AV2_OBU_SWITCH
                    | AV2_OBU_LEADING_TIP
                    | AV2_OBU_REGULAR_TIP
                    | AV2_OBU_BRIDGE_FRAME
            ) {
                break;
            }
            // TODO(b/437292541): What is expected here besides AV2_OBU_SEQUENCE_HEADER,
            //                    AV2_OBU_CONTENT_INTERPRETATION, AV2_OBU_LAYER_CONFIGURATION_RECORD?
            if matches!(obu_type, AV2_OBU_TEMPORAL_DELIMITER | AV2_OBU_PADDING) {
                return AvifError::unknown_error(format!("Unexpected obu_type {}", obu_type));
            }
            offset += obu_length;
        }
        let config_obus = (config_obus_start, offset);

        // Everything else. See https://aomediacodec.github.io/av2-isobmff/pr/40/index.html#sample-obus
        let sample_obus_start = offset;
        while offset < obus.len() {
            let (obu_length, obu_type) = get_one_obu(obus, offset)?;
            if matches!(
                obu_type,
                AV2_OBU_SEQUENCE_HEADER
                    | AV2_OBU_CONTENT_INTERPRETATION
                    | AV2_OBU_LAYER_CONFIGURATION_RECORD
                    | AV2_OBU_TEMPORAL_DELIMITER
            ) {
                return AvifError::unknown_error(format!("Unexpected obu_type {}", obu_type));
            }
            if matches!(obu_type, AV2_OBU_PADDING) {
                break;
            }
            offset += obu_length;
        }
        let sample_obus = (sample_obus_start, offset);

        // Maybe some trailing Padding OBUs.
        while offset < obus.len() {
            let (obu_length, obu_type) = get_one_obu(obus, offset)?;
            if !matches!(obu_type, AV2_OBU_PADDING) {
                return AvifError::unknown_error(format!("Unexpected obu_type {}", obu_type));
            }
            offset += obu_length;
        }
        Ok((config_obus, sample_obus))
    }

    pub(crate) fn get_config_obus_range(obus: &[u8]) -> AvifResult<(usize, usize)> {
        Ok(Self::get_config_and_sample_obus_range(obus)?.0)
    }

    pub(crate) fn get_sample_obus_range(obus: &[u8]) -> AvifResult<(usize, usize)> {
        Ok(Self::get_config_and_sample_obus_range(obus)?.1)
    }

    pub(crate) fn from_av2_config_obus(data: &[u8]) -> AvifResult<Self> {
        use crate::codecs::avm::*;
        let mut stream = IStream::create(data);

        // https://aomediacodec.github.io/av2-isobmff/pr/40/index.html#configobus
        if !stream.has_bytes_left()? {
            return AvifError::bmff_parse_failed(
                "configOBUs shall contain a Sequence Header OBU as its first OBU",
            );
        }

        let (seq_profile, pixel_format, depth) = {
            let num_bytes_in_obu = Self::parse_av2_num_bytes_in_obu(&mut stream)?;
            let mut stream = stream.sub_stream(&BoxSize::FixedSize(num_bytes_in_obu))?;
            let first_obu_type = Self::parse_av2_obu_header(&mut stream)?;
            if first_obu_type != AV2_OBU_SEQUENCE_HEADER {
                return AvifError::bmff_parse_failed(
                    "configOBUs shall contain a Sequence Header OBU as its first OBU",
                );
            }
            Self::parse_av2_sequence_header_obu(&mut stream)?
        };

        let mut chroma_sample_position = ChromaSamplePosition::Unknown;
        while stream.has_bytes_left()? {
            let num_bytes_in_obu = Self::parse_av2_num_bytes_in_obu(&mut stream)?;
            let mut stream = stream.sub_stream(&BoxSize::FixedSize(num_bytes_in_obu))?;
            // https://av2.aomedia.org/v1.0.0/index.html#general_obu_syntax
            let obu_type = Self::parse_av2_obu_header(&mut stream)?;
            match obu_type {
                AV2_OBU_SEQUENCE_HEADER => {
                    // TODO(b/437292541): Reject, parse, or skip?
                }
                AV2_OBU_CONTENT_INTERPRETATION => {
                    // TODO(b/437292541): Reject, parse, or skip if multiple ones?
                    chroma_sample_position =
                        Self::parse_av2_content_interpretation_obu(&mut stream)?;
                }
                AV2_OBU_LAYER_CONFIGURATION_RECORD => {
                    // Allowed as mentioned at
                    // https://aomediacodec.github.io/av2-isobmff/pr/40/index.html#sample-obus
                }
                AV2_OBU_TEMPORAL_DELIMITER
                | AV2_OBU_CLOSED_LOOP_KEY
                | AV2_OBU_OPEN_LOOP_KEY
                | AV2_OBU_LEADING_TILE_GROUP
                | AV2_OBU_REGULAR_TILE_GROUP
                | AV2_OBU_SWITCH
                | AV2_OBU_LEADING_TIP
                | AV2_OBU_REGULAR_TIP
                | AV2_OBU_BRIDGE_FRAME => {
                    return AvifError::bmff_parse_failed(
                        "configOBUs shall NOT contain any of the following:
a Temporal Delimiter OBU;
any OBU carrying coded frame data, including Closed Loop Key OBUs, Open Loop Key OBUs, Switch OBUs,
Bridge Frame OBUs, regular or leading tile-group OBUs, and switch-eligible or TIP frame OBUs;",
                    );
                    // TODO(b/437292541): What is a "switch-eligible" OBU?
                }
                _ => {
                    // TODO(b/437292541): How to check what is expected here?
                }
            }
        }

        let mut config_obus = vec![];
        config_obus
            .try_reserve_exact(data.len())
            .map_err(AvifError::map_out_of_memory)?;
        config_obus.extend_from_slice(data);
        Ok(Av2CodecConfiguration {
            config_obus,
            seq_profile,
            pixel_format,
            depth,
            chroma_sample_position,
        })
    }
}
