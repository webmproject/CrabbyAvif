use crate::internal_utils::stream::*;
use crate::internal_utils::*;
use crate::parser::mp4box::CodecConfiguration;
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
    #[allow(unused)]
    chroma_sample_position: ChromaSamplePosition,
    pub color_primaries: ColorPrimaries,
    pub transfer_characteristics: TransferCharacteristics,
    pub matrix_coefficients: MatrixCoefficients,
    pub full_range: bool,
    config: CodecConfiguration,
}

impl Av1SequenceHeader {
    fn parse_profile(&mut self, bits: &mut IBitStream) -> AvifResult<()> {
        self.config.seq_profile = bits.read(3)? as u8;
        if self.config.seq_profile > 2 {
            println!("invalid seq_profile");
            return Err(AvifError::BmffParseFailed);
        }
        let still_picture = bits.read_bool()?;
        self.reduced_still_picture_header = bits.read_bool()?;
        if self.reduced_still_picture_header && !still_picture {
            return Err(AvifError::BmffParseFailed);
        }
        if self.reduced_still_picture_header {
            self.config.seq_level_idx0 = bits.read(5)? as u8;
        } else {
            let mut buffer_delay_length = 0;
            let mut decoder_model_info_present = false;
            // timing_info_present_flag
            if bits.read_bool()? {
                // num_units_in_display_tick
                bits.skip(32)?;
                // time_scale
                bits.skip(32)?;
                // equal_picture_interval
                if bits.read_bool()? {
                    // num_ticks_per_picture
                    bits.skip_uvlc()?;
                }
                // decoder_model_info_present_flag
                decoder_model_info_present = bits.read_bool()?;
                if decoder_model_info_present {
                    buffer_delay_length = bits.read(5)? + 1;
                    // num_units_in_decoding_tick
                    bits.skip(32)?;
                    // buffer_removal_time_length_minus_1, frame_presentation_time_length_minus_1
                    bits.skip(10)?;
                }
            }
            let initial_display_delay_present = bits.read_bool()?;
            let operaing_points_count = bits.read(5)? + 1;
            for i in 0..operaing_points_count {
                // operating_point_idc
                bits.skip(12)?;
                let seq_level_idx = bits.read(5)?;
                if i == 0 {
                    self.config.seq_level_idx0 = seq_level_idx as u8;
                }
                if seq_level_idx > 7 {
                    let seq_tier = bits.read(1)?;
                    if i == 0 {
                        self.config.seq_tier0 = seq_tier as u8;
                    }
                }
                if decoder_model_info_present {
                    // decoder_model_present_for_this_op
                    if bits.read_bool()? {
                        // decoder_buffer_delay
                        bits.skip(buffer_delay_length as usize)?;
                        // encoder_buffer_delay
                        bits.skip(buffer_delay_length as usize)?;
                        // low_delay_mode_flag
                        bits.skip(1)?;
                    }
                }
                if initial_display_delay_present {
                    // initial_display_delay_present_for_this_op
                    if bits.read_bool()? {
                        // initial_display_delay_minus_1
                        bits.skip(4)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn parse_frame_max_dimensions(&mut self, bits: &mut IBitStream) -> AvifResult<()> {
        let frame_width_bits = bits.read(4)? + 1;
        let frame_height_bits = bits.read(4)? + 1;
        self.max_width = bits.read(frame_width_bits as usize)? + 1;
        self.max_height = bits.read(frame_height_bits as usize)? + 1;
        let mut frame_id_numbers_present = false;
        if !self.reduced_still_picture_header {
            frame_id_numbers_present = bits.read_bool()?;
        }
        if frame_id_numbers_present {
            // delta_frame_id_length_minus_2, additional_frame_id_length_minus_1
            bits.skip(7)?;
        }
        Ok(())
    }

    fn parse_enabled_features(&mut self, bits: &mut IBitStream) -> AvifResult<()> {
        // use_128x128_superblock, enable_filter_intra, enable_intra_edge_filter
        bits.skip(3)?;
        if self.reduced_still_picture_header {
            return Ok(());
        }
        // enable_interintra_compound, enable_masked_compound
        // enable_warped_motion, enable_dual_filter
        bits.skip(4)?;
        let enable_order_hint = bits.read_bool()?;
        if enable_order_hint {
            // enable_jnt_comp, enable_ref_frame_mvs
            bits.skip(2)?;
        }
        let seq_force_screen_content_tools = if bits.read_bool()? { 2 } else { bits.read(1)? };
        if seq_force_screen_content_tools > 0 {
            // seq_choose_integer_mv
            if !bits.read_bool()? {
                // seq_force_integer_mv
                bits.skip(1)?;
            }
        }
        if enable_order_hint {
            // order_hint_bits_minus_1
            bits.skip(3)?;
        }
        Ok(())
    }

    fn parse_color_config(&mut self, bits: &mut IBitStream) -> AvifResult<()> {
        self.config.high_bitdepth = bits.read_bool()?;
        if self.config.seq_profile == 2 && self.config.high_bitdepth {
            self.config.twelve_bit = bits.read_bool()?;
            self.bit_depth = if self.config.twelve_bit { 12 } else { 10 };
        } else {
            self.bit_depth = if self.config.high_bitdepth { 10 } else { 8 };
        }
        if self.config.seq_profile != 1 {
            self.config.monochrome = bits.read_bool()?;
        }
        // color_description_present_flag
        if bits.read_bool()? {
            // color_primaries
            self.color_primaries = (bits.read(8)? as u16).into();
            // transfer_characteristics
            self.transfer_characteristics = (bits.read(8)? as u16).into();
            // matrix_coefficients
            self.matrix_coefficients = (bits.read(8)? as u16).into();
        } else {
            self.color_primaries = ColorPrimaries::Unspecified;
            self.transfer_characteristics = TransferCharacteristics::Unspecified;
            self.matrix_coefficients = MatrixCoefficients::Unspecified;
        }
        if self.config.monochrome {
            self.full_range = bits.read_bool()?;
            self.config.chroma_subsampling_x = 1;
            self.config.chroma_subsampling_y = 1;
            self.yuv_format = PixelFormat::Monochrome;
            return Ok(());
        }
        if self.color_primaries == ColorPrimaries::Srgb
            && self.transfer_characteristics == TransferCharacteristics::Srgb
            && self.matrix_coefficients == MatrixCoefficients::Identity
        {
            self.full_range = true;
            self.yuv_format = PixelFormat::Yuv444;
        } else {
            self.full_range = bits.read_bool()?;
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
                        self.config.chroma_subsampling_x = bits.read(1)? as u8;
                        if self.config.chroma_subsampling_x == 1 {
                            self.config.chroma_subsampling_y = bits.read(1)? as u8;
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
                self.config.chroma_sample_position = bits.read(2)?.into();
            }
        }
        // separate_uv_delta_q
        bits.skip(1)?;
        Ok(())
    }

    fn parse_obu_header(stream: &mut IStream) -> AvifResult<ObuHeader> {
        let mut bits = stream.sub_bit_stream(1)?;
        // obu_forbidden_bit
        bits.skip(1)?;
        // obu_type
        let obu_type = bits.read(4)? as u8;
        // obu_extension_flag
        let obu_extension_flag = bits.read_bool()?;
        // obu_has_size_field
        let obu_has_size_field = bits.read_bool()?;
        // obu_reserved_1bit
        bits.skip(1)?;

        if obu_extension_flag {
            // temporal_id, spatial_id, extension_header_reserved_3bits
            stream.skip(1)?;
        }

        let size = if obu_has_size_field {
            stream.read_uleb128()?
        } else {
            u32_from_usize(stream.bytes_left()?)?
        };

        Ok(ObuHeader { obu_type, size })
    }

    pub fn parse_from_obus(data: &[u8]) -> AvifResult<Self> {
        let mut stream = IStream::create(data);

        while stream.has_bytes_left()? {
            let obu = Self::parse_obu_header(&mut stream)?;
            if obu.obu_type != 1 {
                // Not a sequence header. Skip this obu.
                stream.skip(usize_from_u32(obu.size)?)?;
                continue;
            }
            let mut bits = stream.sub_bit_stream(usize_from_u32(obu.size)?)?;
            let mut sequence_header = Av1SequenceHeader::default();
            sequence_header.parse_profile(&mut bits)?;
            sequence_header.parse_frame_max_dimensions(&mut bits)?;
            sequence_header.parse_enabled_features(&mut bits)?;
            // enable_superres, enable_cdef, enable_restoration
            bits.skip(3)?;
            sequence_header.parse_color_config(&mut bits)?;
            // film_grain_params_present
            bits.skip(1)?;
            return Ok(sequence_header);
        }
        // Failed to parse a sequence header.
        Err(AvifError::BmffParseFailed)
    }
}
