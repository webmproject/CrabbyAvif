use std::io::prelude::*;

use crate::io::*;
use crate::stream::*;
use crate::*;

#[derive(Debug)]
struct BoxHeader {
    size: u64,
    box_type: String,
}

#[derive(Debug, Default)]
pub struct FileTypeBox {
    pub major_brand: String,
    minor_version: u32,
    compatible_brands: Vec<String>,
}

#[derive(Debug, Default)]
pub struct ItemLocationExtent {
    pub offset: u64,
    pub length: u64,
}

#[derive(Debug, Default)]
pub struct ItemLocationEntry {
    pub item_id: u32,
    pub construction_method: u8,
    pub base_offset: u64,
    pub extent_count: u16,
    pub extents: Vec<ItemLocationExtent>,
}

#[derive(Debug, Default)]
pub struct ItemLocationBox {
    offset_size: u8,
    length_size: u8,
    base_offset_size: u8,
    pub items: Vec<ItemLocationEntry>,
}

const MAX_PLANE_COUNT: usize = 4;

#[derive(Debug, Default, Clone)]
pub struct ImageSpatialExtents {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Default, Clone)]
pub struct PixelInformation {
    plane_count: u8,
    plane_depths: [u8; MAX_PLANE_COUNT],
}

#[derive(Debug, Default, Clone)]
pub struct CodecConfiguration {
    seq_profile: u8,
    seq_level_idx0: u8,
    seq_tier0: u8,
    high_bitdepth: bool,
    twelve_bit: bool,
    pub monochrome: bool,
    pub chroma_subsampling_x: u8,
    pub chroma_subsampling_y: u8,
    pub chroma_sample_position: u8,
}

impl CodecConfiguration {
    pub fn depth(&self) -> u8 {
        match self.twelve_bit {
            true => 12,
            false => match (self.high_bitdepth) {
                true => 10,
                false => 8,
            },
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct Icc {
    offset: u64,
    size: usize,
}

#[derive(Debug, Default, Clone)]
pub struct Nclx {
    pub color_primaries: u16,
    pub transfer_characteristics: u16,
    pub matrix_coefficients: u16,
    pub full_range: bool,
}

#[derive(Debug, Clone)]
pub enum ColorInformation {
    Icc(Icc),
    Nclx(Nclx),
}

#[derive(Debug, Default, Clone)]
pub struct PixelAspectRatio {
    h_spacing: u32,
    v_spacing: u32,
}

#[derive(Debug, Default, Clone)]
pub struct ClearAperture {
    width_n: u32,
    width_d: u32,
    height_n: u32,
    height_d: u32,
    horiz_off_n: u32,
    horiz_off_d: u32,
    vert_off_n: u32,
    vert_off_d: u32,
}

#[derive(Debug, Default, Clone)]
pub struct ContentLightLevelInformation {
    max_cll: u16,
    max_pall: u16,
}

#[derive(Debug, Clone)]
pub enum ItemProperty {
    ImageSpatialExtents(ImageSpatialExtents),
    PixelInformation(PixelInformation),
    CodecConfiguration(CodecConfiguration),
    ColorInformation(ColorInformation),
    PixelAspectRatio(PixelAspectRatio),
    AuxiliaryType(String),
    ClearAperture(ClearAperture),
    ImageRotation(u8),
    ImageMirror(u8),
    OperatingPointSelector(u8),
    LayerSelector(u16),
    AV1LayeredImageIndexing([usize; 3]),
    ContentLightLevelInformation(ContentLightLevelInformation),
    Unknown(String),
}

#[derive(Debug, Default)]
pub struct ItemPropertyAssociation {
    version: u8,
    flags: u32,
    pub item_id: u32,
    pub associations: Vec<(u16, bool)>,
}

#[derive(Debug, Default)]
pub struct ItemInfo {
    pub item_id: u32,
    item_protection_index: u16,
    pub item_type: String,
    item_name: String,
    pub content_type: String,
    content_encoding: String,
}

#[derive(Debug, Default)]
pub struct ItemPropertyBox {
    pub properties: Vec<ItemProperty>,
    pub associations: Vec<ItemPropertyAssociation>,
}

#[derive(Debug, Default)]
pub struct ItemReference {
    // Read this reference as "{from_item_id} is a {reference_type} for
    // {to_item_id}" (except for dimg where it is in the opposite
    // direction).
    pub from_item_id: u32,
    pub to_item_id: u32,
    pub reference_type: String,
}

#[derive(Debug, Default)]
pub struct MetaBox {
    pub iinf: Vec<ItemInfo>,
    pub iloc: ItemLocationBox,
    pub primary_item_id: u32,
    pub iprp: ItemPropertyBox,
    pub iref: Vec<ItemReference>,
    pub idat: Vec<u8>,
}

#[derive(Debug, Default)]
pub struct TimeToSample {
    sample_count: u32,
    sample_delta: u32,
}

#[derive(Debug, Default)]
pub struct SampleToChunk {
    first_chunk: u32,
    samples_per_chunk: u32,
    sample_description_index: u32,
}

#[derive(Debug, Default)]
pub struct SampleDescription {
    format: String,
    properties: Vec<ItemProperty>,
}

#[derive(Debug, Default)]
pub struct SampleTable {
    pub chunk_offsets: Vec<u64>,
    pub sample_to_chunk: Vec<SampleToChunk>,
    pub sample_sizes: Vec<u32>,
    // If this is non-zero, sampleSizes will be empty and all samples will be this size.
    // TODO: candidate for rust enum ?
    pub all_samples_size: u32,
    pub sync_samples: Vec<u32>,
    pub time_to_sample: Vec<TimeToSample>,
    pub sample_descriptions: Vec<SampleDescription>,
}

impl SampleTable {
    pub fn has_av1_sample(&self) -> bool {
        // TODO: replace with vector find.
        for sample_description in &self.sample_descriptions {
            if sample_description.format == "av01" {
                return true;
            }
        }
        return false;
    }

    // returns the number of samples in the chunk.
    pub fn get_sample_count_of_chunk(&self, chunk_index: usize) -> u32 {
        for entry in self.sample_to_chunk.iter().rev() {
            if (entry.first_chunk as usize) <= chunk_index + 1 {
                return entry.samples_per_chunk;
            }
        }
        0
    }

    pub fn get_properties(&self) -> Option<&Vec<ItemProperty>> {
        for sample_description in &self.sample_descriptions {
            if sample_description.format == "av01" {
                return Some(&sample_description.properties);
            }
        }
        None
    }
}

#[derive(Debug, Default)]
pub struct AvifTrack {
    pub id: u32,
    pub aux_for_id: u32,
    pub prem_by_id: u32,
    pub media_timescale: u32,
    pub media_duration: u64,
    pub track_duration: u64,
    pub segment_duration: u64,
    pub is_repeating: bool,
    pub repetition_count: i32,
    pub width: u32,
    pub height: u32,
    pub sample_table: Option<SampleTable>,
    elst_seen: bool,
}

impl AvifTrack {
    pub fn is_aux(&self, primary_track_id: u32) -> bool {
        if self.sample_table.is_none() || self.id == 0 {
            return false;
        }
        let sample_table = self.sample_table.as_ref().unwrap();
        if sample_table.chunk_offsets.is_empty() || !sample_table.has_av1_sample() {
            return false;
        }
        self.aux_for_id == primary_track_id
    }

    pub fn is_color(&self) -> bool {
        return self.is_aux(0); // If aux_for_id is 0, then it is the color track.
    }

    pub fn get_properties(&self) -> Option<&Vec<ItemProperty>> {
        self.sample_table.as_ref()?.get_properties()
    }
}

#[derive(Debug, Default)]
pub struct MovieBox {
    pub tracks: Vec<AvifTrack>,
}

#[derive(Debug, Default)]
pub struct AvifBoxes {
    pub ftyp: FileTypeBox,
    pub meta: MetaBox,
    pub moov: MovieBox,
}

pub struct MP4Box {}

impl MP4Box {
    fn parse_header(stream: &mut IStream) -> AvifResult<BoxHeader> {
        let start_offset = stream.offset;
        let mut size: u64 = stream.read_u32()? as u64;
        let box_type = stream.read_string(4)?;
        println!("box_type: {}", box_type);
        if size == 1 {
            size = stream.read_u64()?;
        }
        if box_type == "uuid" {
            stream.skip(16)?;
        }
        size -= (stream.offset - start_offset) as u64;
        Ok(BoxHeader {
            box_type,
            size, // TODO: check if size will fit in usize.
        })
    }

    fn parse_ftyp(stream: &mut IStream) -> AvifResult<FileTypeBox> {
        let major_brand = stream.read_string(4)?;
        let minor_version = stream.read_u32()?;
        let mut compatible_brands: Vec<String> = Vec::new();
        while stream.has_bytes_left() {
            // TODO: check if remaining size is a multiple of 4.
            compatible_brands.push(stream.read_string(4)?);
        }
        Ok(FileTypeBox {
            major_brand,
            minor_version,
            compatible_brands,
        })
    }

    fn parse_hdlr(stream: &mut IStream) -> AvifResult<()> {
        // TODO: version must be 0.
        let (_version, _flags) = stream.read_version_and_flags()?;
        // unsigned int(32) pre_defined = 0;
        let predefined = stream.read_u32()?;
        if predefined != 0 {
            println!("invalid predefined value in hdlr");
            return Err(AvifError::BmffParseFailed);
        }
        // unsigned int(32) handler_type;
        let handler_type = stream.read_string(4)?;
        if handler_type != "pict" {
            println!("handler type is not pict");
            return Err(AvifError::BmffParseFailed);
        }
        // const unsigned int(32)[3] reserved = 0;
        stream.skip(4 * 3)?;
        // string name;
        // Verify that a valid string is here, but don't bother to store it.
        let name = stream.read_c_string()?;
        println!("hdlr: {name}");
        Ok(())
    }

    fn parse_iloc(stream: &mut IStream) -> AvifResult<ItemLocationBox> {
        let start_offset = stream.offset;
        println!("iloc start: {start_offset}");
        let (version, _flags) = stream.read_version_and_flags()?;
        if version > 2 {
            println!("Invalid version in iloc.");
            return Err(AvifError::BmffParseFailed);
        }
        let mut iloc: ItemLocationBox = Default::default();
        let mut bit_reader = stream.get_bitreader()?;
        // unsigned int(4) offset_size;
        iloc.offset_size = bit_reader.read(4);
        // unsigned int(4) length_size;
        iloc.length_size = bit_reader.read(4);
        bit_reader = stream.get_bitreader()?;
        // unsigned int(4) base_offset_size;
        iloc.base_offset_size = bit_reader.read(4);
        if (version == 1 || version == 2) && iloc.base_offset_size != 0 {
            println!("Invalid base_offset_size in iloc.");
            return Err(AvifError::BmffParseFailed);
        }
        // unsigned int(4) reserved; The last 4 bits left in the bit_reader.
        let item_count: u32;
        if version < 2 {
            // unsigned int(16) item_count;
            item_count = stream.read_u16()? as u32;
        } else {
            // unsigned int(32) item_count;
            item_count = stream.read_u32()?;
        }
        for _i in 0..item_count {
            let mut entry: ItemLocationEntry = Default::default();
            if version < 2 {
                // unsigned int(16) item_ID;
                entry.item_id = stream.read_u16()? as u32;
            } else {
                // unsigned int(32) item_ID;
                entry.item_id = stream.read_u32()?;
            }
            if entry.item_id == 0 {
                println!("Invalid item id.");
                return Err(AvifError::BmffParseFailed);
            }
            if version == 1 || version == 2 {
                // unsigned int(12) reserved = 0;
                // unsigned int(4) construction_method;
                stream.skip(1)?;
                let mut byte = stream.get_bitreader()?;
                byte.read(4);
                entry.construction_method = byte.read(4);
                // 0: file, 1: idat.
                if entry.construction_method != 0 && entry.construction_method != 1 {
                    println!("unknown construction_method");
                    return Err(AvifError::BmffParseFailed);
                }
            }
            // unsigned int(16) data_reference_index;
            stream.skip(2)?;
            // unsigned int(base_offset_size*8) base_offset;
            entry.base_offset = stream.read_uxx(iloc.base_offset_size)?;
            // unsigned int(16) extent_count;
            entry.extent_count = stream.read_u16()?;
            for _j in 0..entry.extent_count {
                let mut extent: ItemLocationExtent = Default::default();
                // If extent_index is ever supported, this spec must be implemented here:
                // ::  if (((version == 1) || (version == 2)) && (index_size > 0)) {
                // ::      unsigned int(index_size*8) extent_index;
                // ::  }

                // unsigned int(offset_size*8) extent_offset;
                extent.offset = stream.read_uxx(iloc.offset_size)?;
                // unsigned int(length_size*8) extent_length;
                // TODO: this comment is incorrect in libavif.
                extent.length = stream.read_uxx(iloc.length_size)?;
                entry.extents.push(extent);
            }
            iloc.items.push(entry);
        }

        println!("end of iloc, skiping {} bytes", stream.bytes_left());
        Ok(iloc)
    }

    fn parse_pitm(stream: &mut IStream) -> AvifResult<u32> {
        // TODO: check for multiple pitms.
        let (version, _flags) = stream.read_version_and_flags()?;
        let primary_item_id: u32;
        if version == 0 {
            primary_item_id = stream.read_u16()? as u32;
        } else {
            primary_item_id = stream.read_u32()?;
        }
        Ok(primary_item_id)
    }

    fn parse_ispe(stream: &mut IStream) -> AvifResult<ItemProperty> {
        // TODO: enforce version 0.
        let (_version, _flags) = stream.read_version_and_flags()?;
        let ispe = ImageSpatialExtents {
            // unsigned int(32) image_width;
            width: stream.read_u32()?,
            // unsigned int(32) image_height;
            height: stream.read_u32()?,
        };
        Ok(ItemProperty::ImageSpatialExtents(ispe))
    }

    fn parse_pixi(stream: &mut IStream) -> AvifResult<ItemProperty> {
        // TODO: enforce version 0.
        let (_version, _flags) = stream.read_version_and_flags()?;
        let mut pixi: PixelInformation = Default::default();
        // unsigned int (8) num_channels;
        pixi.plane_count = stream.read_u8()?;
        if usize::from(pixi.plane_count) > MAX_PLANE_COUNT {
            println!("Invalid plane count in pixi box");
            return Err(AvifError::BmffParseFailed);
        }
        for i in 0..pixi.plane_count {
            // unsigned int (8) bits_per_channel;
            pixi.plane_depths[i as usize] = stream.read_u8()?;
        }
        Ok(ItemProperty::PixelInformation(pixi))
    }

    #[allow(non_snake_case)]
    fn parse_av1C(stream: &mut IStream) -> AvifResult<ItemProperty> {
        // unsigned int (1) marker = 1;
        // unsigned int (7) version = 1;
        let mut byte = stream.get_bitreader()?;
        let marker = byte.read(1);
        if marker != 1 {
            println!("Invalid marker in av1C");
            return Err(AvifError::BmffParseFailed);
        }
        let version = byte.read(7);
        if version != 1 {
            println!("Invalid version in av1C");
            return Err(AvifError::BmffParseFailed);
        }
        let mut av1C: CodecConfiguration = Default::default();
        // unsigned int(3) seq_profile;
        // unsigned int(5) seq_level_idx_0;
        byte = stream.get_bitreader()?;
        av1C.seq_profile = byte.read(3);
        av1C.seq_level_idx0 = byte.read(5);

        // unsigned int(1) seq_tier_0;
        // unsigned int(1) high_bitdepth;
        // unsigned int(1) twelve_bit;
        // unsigned int(1) monochrome;
        // unsigned int(1) chroma_subsampling_x;
        // unsigned int(1) chroma_subsampling_y;
        // unsigned int(2) chroma_sample_position;
        byte = stream.get_bitreader()?;
        av1C.seq_tier0 = byte.read(1);
        av1C.high_bitdepth = byte.read(1) == 1;
        av1C.twelve_bit = byte.read(1) == 1;
        av1C.monochrome = byte.read(1) == 1;
        av1C.chroma_subsampling_x = byte.read(1);
        av1C.chroma_subsampling_y = byte.read(1);
        av1C.chroma_sample_position = byte.read(2);

        // unsigned int(3) reserved = 0;
        // unsigned int(1) initial_presentation_delay_present;
        // if(initial_presentation_delay_present) {
        // unsigned int(4) initial_presentation_delay_minus_one;
        // } else {
        // unsigned int(4) reserved = 0;
        // }
        // unsigned int(8) configOBUs[];
        // We skip all these.
        println!("end of av1C, skiping {} bytes", stream.bytes_left());
        Ok(ItemProperty::CodecConfiguration(av1C))
    }

    fn parse_colr(stream: &mut IStream) -> AvifResult<Option<ItemProperty>> {
        // unsigned int(32) colour_type;
        let color_type = stream.read_string(4)?;
        if color_type == "rICC" || color_type == "prof" {
            let mut icc: Icc = Default::default();
            // TODO: perhaps this can be a slice or something?
            // TODO: this offset is relative. needs to be absolute.
            // TODO: maybe just clone the data?
            icc.offset = stream.offset as u64;
            icc.size = stream.bytes_left();
            return Ok(Some(ItemProperty::ColorInformation(ColorInformation::Icc(
                icc,
            ))));
        }
        if color_type == "nclx" {
            let mut nclx: Nclx = Default::default();
            // unsigned int(16) colour_primaries;
            nclx.color_primaries = stream.read_u16()?;
            // unsigned int(16) transfer_characteristics;
            nclx.transfer_characteristics = stream.read_u16()?;
            // unsigned int(16) matrix_coefficients;
            nclx.matrix_coefficients = stream.read_u16()?;
            // unsigned int(1) full_range_flag;
            // unsigned int(7) reserved = 0;
            let mut byte = stream.get_bitreader()?;
            nclx.full_range = byte.read(1) == 1;
            if byte.read(7) != 0 {
                println!("colr box contains invalid reserve bits");
                return Err(AvifError::BmffParseFailed);
            }
            return Ok(Some(ItemProperty::ColorInformation(
                ColorInformation::Nclx(nclx),
            )));
        }
        Ok(None)
    }

    fn parse_pasp(stream: &mut IStream) -> AvifResult<ItemProperty> {
        let mut pasp: PixelAspectRatio = Default::default();
        // unsigned int(32) hSpacing;
        pasp.h_spacing = stream.read_u32()?;
        // unsigned int(32) vSpacing;
        pasp.v_spacing = stream.read_u32()?;
        Ok(ItemProperty::PixelAspectRatio(pasp))
    }

    #[allow(non_snake_case)]
    fn parse_auxC(stream: &mut IStream) -> AvifResult<ItemProperty> {
        // TODO: enforce version 0.
        let (_version, _flags) = stream.read_version_and_flags()?;
        // string aux_type;
        let auxiliary_type = stream.read_c_string()?;
        Ok(ItemProperty::AuxiliaryType(auxiliary_type))
    }

    fn parse_clap(stream: &mut IStream) -> AvifResult<ItemProperty> {
        let mut clap: ClearAperture = Default::default();
        // unsigned int(32) cleanApertureWidthN;
        clap.width_n = stream.read_u32()?;
        // unsigned int(32) cleanApertureWidthD;
        clap.width_d = stream.read_u32()?;
        // unsigned int(32) cleanApertureHeightN;
        clap.height_n = stream.read_u32()?;
        // unsigned int(32) cleanApertureHeightD;
        clap.height_d = stream.read_u32()?;
        // unsigned int(32) horizOffN;
        clap.horiz_off_n = stream.read_u32()?;
        // unsigned int(32) horizOffD;
        clap.horiz_off_d = stream.read_u32()?;
        // unsigned int(32) vertOffN;
        clap.vert_off_n = stream.read_u32()?;
        // unsigned int(32) vertOffD;
        clap.vert_off_d = stream.read_u32()?;
        Ok(ItemProperty::ClearAperture(clap))
    }

    fn parse_irot(stream: &mut IStream) -> AvifResult<ItemProperty> {
        let mut byte = stream.get_bitreader()?;
        // unsigned int (6) reserved = 0;
        if byte.read(6) != 0 {
            println!("invalid reserve bits in irot");
            return Err(AvifError::BmffParseFailed);
        }
        // unsigned int (2) angle;
        let angle = byte.read(2);
        Ok(ItemProperty::ImageRotation(angle))
    }

    fn parse_imir(stream: &mut IStream) -> AvifResult<ItemProperty> {
        let mut byte = stream.get_bitreader()?;
        // unsigned int(7) reserved = 0;
        if byte.read(7) != 0 {
            println!("invalid reserve bits in imir");
            return Err(AvifError::BmffParseFailed);
        }
        let axis = byte.read(1);
        Ok(ItemProperty::ImageMirror(axis))
    }

    fn parse_a1op(stream: &mut IStream) -> AvifResult<ItemProperty> {
        // unsigned int(8) op_index;
        let op_index = stream.read_u8()?;
        if op_index > 31 {
            // 31 is AV1's maximum operating point value.
            println!("Invalid op_index in a1op");
            return Err(AvifError::BmffParseFailed);
        }
        Ok(ItemProperty::OperatingPointSelector(op_index))
    }

    fn parse_lsel(stream: &mut IStream) -> AvifResult<ItemProperty> {
        // unsigned int(16) layer_id;
        let layer_id = stream.read_u16()?;
        if layer_id != 0xFFFF && layer_id >= 4 {
            println!("Invalid layer_id in lsel");
            return Err(AvifError::BmffParseFailed);
        }
        Ok(ItemProperty::LayerSelector(layer_id))
    }

    fn parse_a1lx(stream: &mut IStream) -> AvifResult<ItemProperty> {
        let mut byte = stream.get_bitreader()?;
        // unsigned int(7) reserved = 0;
        if byte.read(7) != 0 {
            println!("Invalid reserve bits in a1lx");
            return Err(AvifError::BmffParseFailed);
        }
        // unsigned int(1) large_size;
        let large_size = byte.read(1) == 1;
        let mut layer_sizes: [usize; 3] = [0; 3];
        for layer_size in &mut layer_sizes {
            if large_size {
                *layer_size = stream.read_u32()? as usize;
            } else {
                *layer_size = stream.read_u16()? as usize;
            }
        }
        Ok(ItemProperty::AV1LayeredImageIndexing(layer_sizes))
    }

    fn parse_clli(stream: &mut IStream) -> AvifResult<ItemProperty> {
        let mut clli: ContentLightLevelInformation = Default::default();
        // unsigned int(16) max_content_light_level
        clli.max_cll = stream.read_u16()?;
        // unsigned int(16) max_pic_average_light_level
        clli.max_pall = stream.read_u16()?;
        Ok(ItemProperty::ContentLightLevelInformation(clli))
    }

    #[allow(non_snake_case)]
    fn parse_ipco(stream: &mut IStream) -> AvifResult<Vec<ItemProperty>> {
        let mut properties: Vec<ItemProperty> = Vec::new();
        while stream.has_bytes_left() {
            let header = Self::parse_header(stream)?;
            let mut sub_stream = stream.sub_stream(header.size as usize)?;
            match header.box_type.as_str() {
                "ispe" => properties.push(Self::parse_ispe(&mut sub_stream)?),
                "pixi" => properties.push(Self::parse_pixi(&mut sub_stream)?),
                "av1C" => properties.push(Self::parse_av1C(&mut sub_stream)?),
                "colr" => match Self::parse_colr(&mut sub_stream)? {
                    Some(colr) => properties.push(colr),
                    None => {}
                },
                "pasp" => properties.push(Self::parse_pasp(&mut sub_stream)?),
                "auxC" => properties.push(Self::parse_auxC(&mut sub_stream)?),
                "clap" => properties.push(Self::parse_clap(&mut sub_stream)?),
                "irot" => properties.push(Self::parse_irot(&mut sub_stream)?),
                "imir" => properties.push(Self::parse_imir(&mut sub_stream)?),
                "a1op" => properties.push(Self::parse_a1op(&mut sub_stream)?),
                "lsel" => properties.push(Self::parse_lsel(&mut sub_stream)?),
                "a1lx" => properties.push(Self::parse_a1lx(&mut sub_stream)?),
                "clli" => properties.push(Self::parse_clli(&mut sub_stream)?),
                _ => properties.push(ItemProperty::Unknown(header.box_type)),
            }
        }
        Ok(properties)
    }

    fn parse_ipma(stream: &mut IStream) -> AvifResult<Vec<ItemPropertyAssociation>> {
        let (version, flags) = stream.read_version_and_flags()?;
        // unsigned int(32) entry_count;
        let entry_count = stream.read_u32()?;
        let mut previous_item_id = 0; // TODO: there is no need for this. can simply look up the vector.
        let mut ipma: Vec<ItemPropertyAssociation> = Vec::new();
        for _i in 0..entry_count {
            let mut entry: ItemPropertyAssociation = Default::default();
            entry.version = version;
            entry.flags = flags;
            // ISO/IEC 23008-12, First edition, 2017-12, Section 9.3.1:
            //   Each ItemPropertyAssociation box shall be ordered by increasing item_ID, and there shall
            //   be at most one association box for each item_ID, in any ItemPropertyAssociation box.
            if version < 1 {
                // unsigned int(16) item_ID;
                entry.item_id = stream.read_u16()? as u32;
            } else {
                // unsigned int(32) item_ID;
                entry.item_id = stream.read_u32()?;
            }
            if entry.item_id == 0 {
                println!("invalid item id in ipma");
                return Err(AvifError::BmffParseFailed);
            }
            if entry.item_id <= previous_item_id {
                println!("ipma item ids are not ordered by increasing id");
                return Err(AvifError::BmffParseFailed);
            }
            previous_item_id = entry.item_id;
            // unsigned int(8) association_count;
            let association_count = stream.read_u8()?;
            for _j in 0..association_count {
                // bit(1) essential;
                let mut byte = stream.get_bitreader()?;
                let essential = byte.read(1) == 1;
                // unsigned int(7 or 15) property_index;
                let mut property_index: u16 = byte.read(7) as u16;
                if (flags & 0x1) == 1 {
                    let property_index_lsb: u16 = stream.read_u8()? as u16;
                    property_index <<= 8;
                    property_index |= property_index_lsb;
                }
                // TODO: verify the correctness of essential.
                entry.associations.push((property_index, essential));
            }
            ipma.push(entry);
        }
        Ok(ipma)
    }

    fn parse_iprp(stream: &mut IStream) -> AvifResult<ItemPropertyBox> {
        println!("iprp start: {}", stream.offset);
        let header = Self::parse_header(stream)?;
        if header.box_type != "ipco" {
            println!("First box in iprp is not ipco");
            return Err(AvifError::BmffParseFailed);
        }
        let mut iprp: ItemPropertyBox = Default::default();
        // Parse ipco box.
        {
            let mut sub_stream = stream.sub_stream(header.size as usize)?;
            match Self::parse_ipco(&mut sub_stream) {
                Ok(properties) => {
                    iprp.properties = properties;
                }
                Err(err) => {
                    // TODO: re-using err here results in some weird borrow checker error:
                    // https://old.reddit.com/r/rust/comments/qi3ye9/why_does_returning_a_value_mess_with_borrows/
                    println!("ipco parsing failed");
                    return Err(err);
                }
            }
        }
        // Parse ipma boxes.
        while stream.has_bytes_left() {
            let header = Self::parse_header(stream)?;
            if header.box_type != "ipma" {
                println!("Found non ipma box in iprp");
                return Err(AvifError::BmffParseFailed);
            }
            let mut sub_stream = stream.sub_stream(header.size as usize)?;
            match Self::parse_ipma(&mut sub_stream) {
                Ok(mut ipma) => iprp.associations.append(&mut ipma),
                Err(err) => {
                    // TODO: re-using err here results in some weird borrow checker error:
                    println!("ipma parsing failed");
                    return Err(err);
                }
            }
        }
        println!("end of iprp, skiping {} bytes", stream.bytes_left());
        Ok(iprp)
    }

    fn parse_iinf(stream: &mut IStream) -> AvifResult<Vec<ItemInfo>> {
        let start_offset = stream.offset;
        let (version, _flags) = stream.read_version_and_flags()?;
        let entry_count: u32;
        if version == 0 {
            // unsigned int(16) entry_count;
            entry_count = stream.read_u16()? as u32;
        } else {
            // unsigned int(32) entry_count;
            entry_count = stream.read_u32()?;
        }
        let mut iinf: Vec<ItemInfo> = Vec::new();
        for _i in 0..entry_count {
            let header = Self::parse_header(stream)?;
            if header.box_type != "infe" {
                println!("Found non infe box in iinf");
                return Err(AvifError::BmffParseFailed);
            }
            let (version, _flags) = stream.read_version_and_flags()?;
            if version != 2 && version != 3 {
                println!("infe box version 2 or 3 expected.");
                return Err(AvifError::BmffParseFailed);
            }

            // TODO: check flags. ISO/IEC 23008-12:2017, Section 9.2 says:
            //   The flags field of ItemInfoEntry with version greater than or equal to 2 is specified as
            //   follows:
            //
            //   (flags & 1) equal to 1 indicates that the item is not intended to be a part of the
            //   presentation. For example, when (flags & 1) is equal to 1 for an image item, the image
            //   item should not be displayed.
            //   (flags & 1) equal to 0 indicates that the item is intended to be a part of the
            //   presentation.
            //
            // See also Section 6.4.2.

            let mut entry: ItemInfo = Default::default();
            if version == 2 {
                // unsigned int(16) item_ID;
                entry.item_id = stream.read_u16()? as u32;
            } else {
                // unsigned int(16) item_ID;
                entry.item_id = stream.read_u32()?;
            }
            if entry.item_id == 0 {
                println!("Invalid item id found in infe");
                return Err(AvifError::BmffParseFailed);
            }
            // unsigned int(16) item_protection_index;
            entry.item_protection_index = stream.read_u16()?;
            // unsigned int(32) item_type;
            entry.item_type = stream.read_string(4)?;

            // TODO: libavif read vs write does not seem to match. check it out.
            // The rust code follows the spec.

            // utf8string item_name;
            entry.item_name = stream.read_c_string()?;
            if entry.item_type == "mime" {
                // string content_type;
                entry.content_type = stream.read_c_string()?;
                // string content_encoding;
                entry.content_encoding = stream.read_c_string()?;
            } else if entry.item_type == "uri" {
                // string item_uri_type; (skipped)
                _ = stream.read_c_string()?;
            }
            iinf.push(entry);
        }
        println!("end of iinf, skiping {} bytes", stream.bytes_left());
        Ok(iinf)
    }

    fn parse_iref(stream: &mut IStream) -> AvifResult<Vec<ItemReference>> {
        let start_offset = stream.offset;
        let (version, _flags) = stream.read_version_and_flags()?;
        let mut iref: Vec<ItemReference> = Vec::new();
        // versions > 1 are not supported. ignore them.
        if version <= 1 {
            while stream.has_bytes_left() {
                let header = Self::parse_header(stream)?;
                let from_item_id: u32;
                if version == 0 {
                    // unsigned int(16) from_item_ID;
                    from_item_id = stream.read_u16()? as u32;
                } else {
                    // unsigned int(32) from_item_ID;
                    from_item_id = stream.read_u32()?;
                }
                if from_item_id == 0 {
                    println!("invalid from_item_id in iref");
                    return Err(AvifError::BmffParseFailed);
                }
                // unsigned int(16) reference_count;
                let reference_count = stream.read_u16()?;
                for reference_index in 0..reference_count {
                    let to_item_id: u32;
                    if version == 0 {
                        // unsigned int(16) to_item_ID;
                        to_item_id = stream.read_u16()? as u32;
                    } else {
                        // unsigned int(32) to_item_ID;
                        to_item_id = stream.read_u32()?;
                    }
                    if to_item_id == 0 {
                        println!("invalid to_item_id in iref");
                        return Err(AvifError::BmffParseFailed);
                    }
                    iref.push(ItemReference {
                        from_item_id,
                        to_item_id,
                        reference_type: header.box_type.clone(),
                    });
                }
            }
        }
        println!("end of iref, skiping {} bytes", stream.bytes_left());
        Ok(iref)
    }

    fn parse_idat(stream: &mut IStream) -> AvifResult<Vec<u8>> {
        // TODO: check if multiple idats were seen for this meta box.
        if !stream.has_bytes_left() {
            println!("Invalid idat size");
            return Err(AvifError::BmffParseFailed);
        }
        let mut idat: Vec<u8> = Vec::new();
        idat.reserve(stream.bytes_left());
        idat.extend_from_slice(stream.get_slice(stream.bytes_left())?);
        Ok(idat)
    }

    fn parse_meta(stream: &mut IStream) -> AvifResult<MetaBox> {
        println!("parsing meta size: {}", stream.data.len());
        // TODO: version must be 0.
        let (_version, _flags) = stream.read_version_and_flags()?;
        let mut first_box = true;
        let mut meta: MetaBox = Default::default();

        // TODO: add box unique checks.

        while stream.has_bytes_left() {
            let header = Self::parse_header(stream)?;
            if first_box {
                if header.box_type != "hdlr" {
                    println!("first box in meta is not hdlr");
                    return Err(AvifError::BmffParseFailed);
                }
                let mut sub_stream = stream.sub_stream(header.size as usize)?;
                Self::parse_hdlr(&mut sub_stream)?;
                first_box = false;
                continue;
            }
            let mut sub_stream = stream.sub_stream(header.size as usize)?;
            match header.box_type.as_str() {
                "iloc" => meta.iloc = Self::parse_iloc(&mut sub_stream)?,
                "pitm" => meta.primary_item_id = Self::parse_pitm(&mut sub_stream)?,
                "iprp" => meta.iprp = Self::parse_iprp(&mut sub_stream)?,
                "iinf" => meta.iinf = Self::parse_iinf(&mut sub_stream)?,
                "iref" => meta.iref = Self::parse_iref(&mut sub_stream)?,
                "idat" => meta.idat = Self::parse_idat(&mut sub_stream)?,
                _ => println!("skipping box {}", header.box_type),
            }
        }
        if first_box {
            // The meta box must not be empty (it must contain at least a hdlr box).
            println!("Meta box has no child boxes");
            return Err(AvifError::BmffParseFailed);
        }
        Ok(meta)
    }

    fn parse_tkhd(stream: &mut IStream, track: &mut AvifTrack) -> AvifResult<()> {
        let (version, _flags) = stream.read_version_and_flags()?;
        if version == 1 {
            // unsigned int(64) creation_time;
            stream.skip_u64()?;
            // unsigned int(64) modification_time;
            stream.skip_u64()?;
            // unsigned int(32) track_ID;
            track.id = stream.read_u32()?;
            // const unsigned int(32) reserved = 0;
            stream.skip_u32()?;
            // unsigned int(64) duration;
            track.track_duration = stream.read_u64()?;
        } else if version == 0 {
            // unsigned int(32) creation_time;
            stream.skip_u32()?;
            // unsigned int(32) modification_time;
            stream.skip_u32()?;
            // unsigned int(32) track_ID;
            track.id = stream.read_u32()?;
            // const unsigned int(32) reserved = 0;
            stream.skip_u32()?;
            // unsigned int(32) duration;
            track.track_duration = stream.read_u32()? as u64;
        } else {
            println!("unsupported version in trak");
            return Err(AvifError::BmffParseFailed);
        }

        // Skip the following 52 bytes.
        // const unsigned int(32)[2] reserved = 0;
        // template int(16) layer = 0;
        // template int(16) alternate_group = 0;
        // template int(16) volume = {if track_is_audio 0x0100 else 0};
        // const unsigned int(16) reserved = 0;
        // template int(32)[9] matrix= { 0x00010000,0,0,0,0x00010000,0,0,0,0x40000000 }; // unity matrix
        stream.skip(52)?;

        // unsigned int(32) width;
        track.width = stream.read_u32()? >> 16;
        // unsigned int(32) height;
        track.height = stream.read_u32()? >> 16;

        if track.width == 0 || track.height == 0 {
            println!("invalid track dimensions");
            return Err(AvifError::BmffParseFailed);
        }

        // TODO: check if track dims are too large.

        Ok(())
    }

    fn parse_mdhd(stream: &mut IStream, track: &mut AvifTrack) -> AvifResult<()> {
        let (version, _flags) = stream.read_version_and_flags()?;
        if version == 1 {
            // unsigned int(64) creation_time;
            stream.skip_u64()?;
            // unsigned int(64) modification_time;
            stream.skip_u64()?;
            // unsigned int(32) timescale;
            track.media_timescale = stream.read_u32()?;
            // unsigned int(64) duration;
            track.media_duration = stream.read_u64()?;
        } else if version == 0 {
            // unsigned int(32) creation_time;
            stream.skip_u32()?;
            // unsigned int(32) modification_time;
            stream.skip_u32()?;
            // unsigned int(32) timescale;
            track.media_timescale = stream.read_u32()?;
            // unsigned int(32) duration;
            track.media_duration = stream.read_u32()? as u64;
        } else {
            println!("unsupported version in mdhd");
            return Err(AvifError::BmffParseFailed);
        }

        // Skip the following 4 bytes.
        // bit(1) pad = 0;
        // unsigned int(5)[3] language; // ISO-639-2/T language code
        // unsigned int(16) pre_defined = 0;
        stream.skip(4)?;
        Ok(())
    }

    fn parse_stco(
        stream: &mut IStream,
        sample_table: &mut SampleTable,
        large_offset: bool,
    ) -> AvifResult<()> {
        // TODO: version must be 0.
        let (_version, _flags) = stream.read_version_and_flags()?;
        // unsigned int(32) entry_count;
        let entry_count = stream.read_u32()?;
        sample_table.chunk_offsets.reserve(entry_count as usize);
        for i in 0..entry_count {
            let chunk_offset: u64;
            if large_offset {
                // TODO: this comment is wrong in libavif.
                // unsigned int(64) chunk_offset;
                chunk_offset = stream.read_u64()?;
            } else {
                // unsigned int(32) chunk_offset;
                chunk_offset = stream.read_u32()? as u64;
            }
            sample_table.chunk_offsets.push(chunk_offset);
        }
        Ok(())
    }

    fn parse_stsc(stream: &mut IStream, sample_table: &mut SampleTable) -> AvifResult<()> {
        // TODO: version must be 0.
        let (_version, _flags) = stream.read_version_and_flags()?;
        // unsigned int(32) entry_count;
        let entry_count = stream.read_u32()?;
        sample_table.sample_to_chunk.reserve(entry_count as usize);
        for i in 0..entry_count {
            let mut stsc: SampleToChunk = Default::default();
            // unsigned int(32) first_chunk;
            stsc.first_chunk = stream.read_u32()?;
            // unsigned int(32) samples_per_chunk;
            stsc.samples_per_chunk = stream.read_u32()?;
            // unsigned int(32) sample_description_index;
            stsc.sample_description_index = stream.read_u32()?;

            if i == 0 {
                if stsc.first_chunk != 1 {
                    println!("stsc does not begin with chunk 1.");
                    return Err(AvifError::BmffParseFailed);
                }
            } else {
                if stsc.first_chunk <= sample_table.sample_to_chunk.last().unwrap().first_chunk {
                    println!("stsc chunks are not strictly increasing.");
                    return Err(AvifError::BmffParseFailed);
                }
            }
            sample_table.sample_to_chunk.push(stsc);
        }
        Ok(())
    }

    fn parse_stsz(stream: &mut IStream, sample_table: &mut SampleTable) -> AvifResult<()> {
        // TODO: version must be 0.
        let (_version, _flags) = stream.read_version_and_flags()?;
        // unsigned int(32) sample_size;
        sample_table.all_samples_size = stream.read_u32()?;
        // unsigned int(32) sample_count;
        let sample_count = stream.read_u32()?;

        if sample_table.all_samples_size > 0 {
            return Ok(());
        }
        sample_table.sample_sizes.reserve(sample_count as usize);
        for i in 0..sample_count {
            // unsigned int(32) entry_size;
            let entry_size = stream.read_u32()?;
            sample_table.sample_sizes.push(entry_size);
        }
        Ok(())
    }

    fn parse_stss(stream: &mut IStream, sample_table: &mut SampleTable) -> AvifResult<()> {
        // TODO: version must be 0.
        let (_version, _flags) = stream.read_version_and_flags()?;
        // unsigned int(32) entry_count;
        let entry_count = stream.read_u32()?;
        sample_table.sync_samples.reserve(entry_count as usize);
        for i in 0..entry_count {
            // unsigned int(32) sample_number;
            let sample_number = stream.read_u32()?;
            sample_table.sync_samples.push(sample_number);
        }
        Ok(())
    }

    fn parse_stts(stream: &mut IStream, sample_table: &mut SampleTable) -> AvifResult<()> {
        // TODO: version must be 0.
        let (_version, _flags) = stream.read_version_and_flags()?;
        // unsigned int(32) entry_count;
        let entry_count = stream.read_u32()?;
        sample_table.time_to_sample.reserve(entry_count as usize);
        for i in 0..entry_count {
            let mut stts: TimeToSample = Default::default();
            // unsigned int(32) sample_count;
            stts.sample_count = stream.read_u32()?;
            // unsigned int(32) sample_delta;
            stts.sample_delta = stream.read_u32()?;
            sample_table.time_to_sample.push(stts);
        }
        Ok(())
    }

    fn parse_stsd(stream: &mut IStream, sample_table: &mut SampleTable) -> AvifResult<()> {
        // TODO: version must be 0.
        let (_version, _flags) = stream.read_version_and_flags()?;
        // unsigned int(32) entry_count;
        let entry_count = stream.read_u32()?;
        sample_table
            .sample_descriptions
            .reserve(entry_count as usize);
        for i in 0..entry_count {
            let header = Self::parse_header(stream)?;
            let mut stsd: SampleDescription = Default::default();
            stsd.format = header.box_type.clone();
            if stsd.format == "av01" {
                // Skip 78 bytes for visual sample entry size.
                stream.skip(78)?;
                // TODO: check subtraction is ok.
                let mut sub_stream = stream.sub_stream((header.size - 78) as usize)?;
                stsd.properties = Self::parse_ipco(&mut sub_stream)?;
            }
            sample_table.sample_descriptions.push(stsd);
        }
        Ok(())
    }

    fn parse_stbl(stream: &mut IStream, track: &mut AvifTrack) -> AvifResult<()> {
        if track.sample_table.is_some() {
            println!("duplciate stbl for track.");
            return Err(AvifError::BmffParseFailed);
        }
        let mut sample_table: SampleTable = Default::default();
        while stream.has_bytes_left() {
            let header = Self::parse_header(stream)?;
            let mut sub_stream = stream.sub_stream(header.size as usize)?;
            match header.box_type.as_str() {
                "stco" => Self::parse_stco(&mut sub_stream, &mut sample_table, false)?,
                "co64" => Self::parse_stco(&mut sub_stream, &mut sample_table, true)?,
                "stsc" => Self::parse_stsc(&mut sub_stream, &mut sample_table)?,
                "stsz" => Self::parse_stsz(&mut sub_stream, &mut sample_table)?,
                "stss" => Self::parse_stss(&mut sub_stream, &mut sample_table)?,
                "stts" => Self::parse_stts(&mut sub_stream, &mut sample_table)?,
                "stsd" => Self::parse_stsd(&mut sub_stream, &mut sample_table)?,
                _ => println!("skipping box {}", header.box_type),
            }
        }
        track.sample_table = Some(sample_table);
        Ok(())
    }

    fn parse_minf(stream: &mut IStream, track: &mut AvifTrack) -> AvifResult<()> {
        while stream.has_bytes_left() {
            let header = Self::parse_header(stream)?;
            let mut sub_stream = stream.sub_stream(header.size as usize)?;
            match header.box_type.as_str() {
                "stbl" => Self::parse_stbl(&mut sub_stream, track)?,
                _ => println!("skipping box {}", header.box_type),
            }
        }
        Ok(())
    }

    fn parse_mdia(stream: &mut IStream, track: &mut AvifTrack) -> AvifResult<()> {
        while stream.has_bytes_left() {
            let header = Self::parse_header(stream)?;
            let mut sub_stream = stream.sub_stream(header.size as usize)?;
            match header.box_type.as_str() {
                "mdhd" => Self::parse_mdhd(&mut sub_stream, track)?,
                "minf" => Self::parse_minf(&mut sub_stream, track)?,
                _ => println!("skipping box {}", header.box_type),
            }
        }
        Ok(())
    }

    fn parse_tref(stream: &mut IStream, track: &mut AvifTrack) -> AvifResult<()> {
        while stream.has_bytes_left() {
            let header = Self::parse_header(stream)?;
            let mut sub_stream = stream.sub_stream(header.size as usize)?;
            match header.box_type.as_str() {
                "auxl" => {
                    // unsigned int(32) track_IDs[];
                    // Use only the first one and skip the rest.
                    track.aux_for_id = sub_stream.read_u32()?;
                }
                "prem" => {
                    // unsigned int(32) track_IDs[];
                    // Use only the first one and skip the rest.
                    track.prem_by_id = sub_stream.read_u32()?;
                }
                _ => println!("skipping box {}", header.box_type),
            }
        }
        Ok(())
    }

    fn parse_elst(stream: &mut IStream, track: &mut AvifTrack) -> AvifResult<()> {
        if track.elst_seen {
            println!("more than one elst box was found for track");
            return Err(AvifError::BmffParseFailed);
        }
        track.elst_seen = true;
        let (version, flags) = stream.read_version_and_flags()?;
        if (flags & 1) == 0 {
            track.is_repeating = false;
            return Ok(());
        }
        track.is_repeating = true;
        // unsigned int(32) entry_count;
        let entry_count = stream.read_u32()?;
        if entry_count != 1 {
            println!("elst has entry_count != 1");
            return Err(AvifError::BmffParseFailed);
        }
        if version == 1 {
            // unsigned int(64) segment_duration;
            track.segment_duration = stream.read_u64()?;
        } else if version == 0 {
            // unsigned int(32) segment_duration;
            track.segment_duration = stream.read_u32()? as u64;
        } else {
            println!("unsupported version in elst");
            return Err(AvifError::BmffParseFailed);
        }
        if track.segment_duration == 0 {
            println!("invalid value for segment_duration (0)");
            return Err(AvifError::BmffParseFailed);
        }
        Ok(())
    }

    fn parse_edts(stream: &mut IStream, track: &mut AvifTrack) -> AvifResult<()> {
        // TODO: add uniqueness check.
        while stream.has_bytes_left() {
            let header = Self::parse_header(stream)?;
            let mut sub_stream = stream.sub_stream(header.size as usize)?;
            match header.box_type.as_str() {
                "elst" => Self::parse_elst(&mut sub_stream, track)?,
                _ => println!("skipping box {}", header.box_type),
            }
        }
        if !track.elst_seen {
            println!("elst box was not found in edts");
            return Err(AvifError::BmffParseFailed);
        }
        Ok(())
    }

    fn parse_trak(stream: &mut IStream) -> AvifResult<AvifTrack> {
        let mut track: AvifTrack = Default::default();
        println!("parsing trak size: {}", stream.bytes_left());
        let mut edts_seen = false;
        while stream.has_bytes_left() {
            let header = Self::parse_header(stream)?;
            let mut sub_stream = stream.sub_stream(header.size as usize)?;
            match header.box_type.as_str() {
                "tkhd" => Self::parse_tkhd(&mut sub_stream, &mut track)?,
                "mdia" => Self::parse_mdia(&mut sub_stream, &mut track)?,
                "tref" => Self::parse_tref(&mut sub_stream, &mut track)?,
                "edts" => Self::parse_edts(&mut sub_stream, &mut track)?,
                // TODO: track meta can be ignored? probably not becuase of xmp/exif.
                _ => println!("skipping box {}", header.box_type),
            }
        }
        if track.elst_seen {
            if track.track_duration == u64::MAX {
                // If isRepeating is true and the track duration is
                // unknown/indefinite, then set the repetition count to
                // infinite(Section 9.6.1 of ISO/IEC 23008-12 Part 12).
                track.repetition_count = -1;
            } else {
                // Section 9.6.1. of ISO/IEC 23008-12 Part 12: 1, the entire
                // edit list is repeated a sufficient number of times to
                // equal the track duration.
                //
                // Since libavif uses repetitionCount (which is 0-based), we
                // subtract the value by 1 to derive the number of
                // repetitions.
                assert!(track.segment_duration != 0);
                // We specifically check for trackDuration == 0 here and not
                // when it is actually read in order to accept files which
                // inadvertently has a trackDuration of 0 without any edit
                // lists.
                if track.track_duration == 0 {
                    println!("invalid track duration 0");
                    return Err(AvifError::BmffParseFailed);
                }
                let remainder = if track.track_duration % track.segment_duration != 0 {
                    1u64
                } else {
                    0u64
                };
                let repetition_count: u64 =
                    (track.track_duration / track.segment_duration) + remainder - 1u64;
                if repetition_count > (i32::MAX as u64) {
                    track.repetition_count = -1;
                } else {
                    track.repetition_count = repetition_count as i32;
                }
            }
        } else {
            track.repetition_count = -2;
        }
        Ok(track)
    }

    fn parse_moov(stream: &mut IStream) -> AvifResult<MovieBox> {
        println!("parsing moov size: {}", stream.bytes_left());
        let mut moov: MovieBox = Default::default();
        while stream.has_bytes_left() {
            let header = Self::parse_header(stream)?;
            let mut sub_stream = stream.sub_stream(header.size as usize)?;
            match header.box_type.as_str() {
                "trak" => moov.tracks.push(Self::parse_trak(&mut sub_stream)?),
                _ => println!("skipping box {}", header.box_type),
            }
        }
        Ok(moov)
    }

    pub fn parse(io: &mut Box<dyn AvifDecoderIO>) -> AvifResult<AvifBoxes> {
        let mut ftyp_seen = false;
        let mut avif_boxes: AvifBoxes = Default::default();
        let mut meta_seen = false;
        let mut parse_offset: u64 = 0;
        loop {
            // Read just enough to get the next box header (32 bytes).
            let header_data = io.read(parse_offset, 32)?;
            if header_data.len() == 0 {
                // No error and size is 0. We have reached the end of the stream.
                break;
            }
            let mut header_stream = IStream::create(header_data);
            let header = MP4Box::parse_header(&mut header_stream)?;
            println!("{:#?}", header);
            parse_offset += header_stream.offset as u64;

            // Read the rest of the box if necessary.
            match header.box_type.as_str() {
                "ftyp" | "meta" | "moov" => {
                    // TODO: check overflow of header.size to usize cast.
                    let box_data = io.read(parse_offset, header.size as usize)?;
                    if box_data.len() != header.size as usize {
                        return Err(AvifError::TruncatedData);
                    }
                    let mut box_stream = IStream::create(box_data);
                    match header.box_type.as_str() {
                        "ftyp" => {
                            avif_boxes.ftyp = MP4Box::parse_ftyp(&mut box_stream)?;
                            ftyp_seen = true;
                        }
                        "meta" => {
                            avif_boxes.meta = MP4Box::parse_meta(&mut box_stream)?;
                            meta_seen = true;
                            println!("{:#?}", avif_boxes);
                        }
                        "moov" => {
                            avif_boxes.moov = MP4Box::parse_moov(&mut box_stream)?;
                        }
                        _ => {} // Not reached.
                    }
                }
                _ => {
                    println!("skipping box: {}", header.box_type);
                }
            }
            parse_offset += header.size;
        }
        println!("{:#?}", avif_boxes);
        Ok(avif_boxes)
    }
}
