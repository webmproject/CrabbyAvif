use std::io::prelude::*;

use crate::stream::*;

#[derive(Debug)]
struct BoxHeader {
    full_size: u64,
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
pub struct ItemData {
    pub size: u64,
    pub offset: usize,
}

#[derive(Debug, Default)]
pub struct MetaBox {
    pub iinf: Vec<ItemInfo>,
    pub iloc: ItemLocationBox,
    pub primary_item_id: u32,
    pub iprp: ItemPropertyBox,
    pub iref: Vec<ItemReference>,
    pub idat: ItemData,
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
    fn parse_header(stream: &mut IStream) -> BoxHeader {
        let start_offset = stream.offset;
        let mut size: u64 = stream.read_u32().into();
        let box_type = stream.read_string(4);
        println!("box_type: {}", box_type);
        if size == 1 {
            size = stream.read_u64();
        }

        // if uuid, skip 16.

        let bytes_read: u64 = (stream.offset - start_offset).try_into().unwrap();
        BoxHeader {
            box_type,
            size: size - bytes_read, // do overflow check for bytes_read?
            full_size: size,
        }
    }

    fn parse_ftyp(stream: &mut IStream, size: u64) -> FileTypeBox {
        let major_brand = stream.read_string(4);
        let minor_version = stream.read_u32();
        let mut remaining_size = size - 8;
        let mut compatible_brands: Vec<String> = Vec::new();
        while remaining_size > 0 {
            // TODO: check if remaining size is a multiple of 4.
            compatible_brands.push(stream.read_string(4));
            remaining_size -= 4;
        }
        FileTypeBox {
            major_brand,
            minor_version,
            compatible_brands,
        }
    }

    fn parse_hdlr(stream: &mut IStream) -> bool {
        // TODO: version must be 0.
        let (_version, _flags) = stream.read_version_and_flags();
        // unsigned int(32) pre_defined = 0;
        let predefined = stream.read_u32();
        if predefined != 0 {
            return false;
        }
        // unsigned int(32) handler_type;
        let handler_type = stream.read_string(4);
        if handler_type != "pict" {
            return false;
        }
        // const unsigned int(32)[3] reserved = 0;
        stream.skip(4 * 3);
        // string name;
        // Verify that a valid string is here, but don't bother to store it.
        let name = stream.read_c_string();
        println!("{name}");
        true
    }

    fn parse_iloc(stream: &mut IStream, size: u64) -> Result<ItemLocationBox, &str> {
        let start_offset = stream.offset;
        println!("iloc start: {start_offset}");
        let (version, _flags) = stream.read_version_and_flags();
        if version > 2 {
            return Err("Invalid version in iloc.");
        }
        let mut iloc: ItemLocationBox = Default::default();
        let mut bit_reader = stream.get_bitreader();
        // unsigned int(4) offset_size;
        iloc.offset_size = bit_reader.read(4);
        // unsigned int(4) length_size;
        iloc.length_size = bit_reader.read(4);
        bit_reader = stream.get_bitreader();
        // unsigned int(4) base_offset_size;
        iloc.base_offset_size = bit_reader.read(4);
        if (version == 1 || version == 2) && iloc.base_offset_size != 0 {
            return Err("Invalid base_offset_size in iloc.");
        }
        // unsigned int(4) reserved; The last 4 bits left in the bit_reader.
        let item_count: u32;
        if version < 2 {
            // unsigned int(16) item_count;
            item_count = stream.read_u16().into();
        } else {
            // unsigned int(32) item_count;
            item_count = stream.read_u32();
        }
        for _i in 0..item_count {
            let mut entry: ItemLocationEntry = Default::default();
            if version < 2 {
                // unsigned int(16) item_ID;
                entry.item_id = stream.read_u16().into();
            } else {
                // unsigned int(32) item_ID;
                entry.item_id = stream.read_u32();
            }
            if entry.item_id == 0 {
                return Err("Invalid item id.");
            }
            if version == 1 || version == 2 {
                // unsigned int(12) reserved = 0;
                // unsigned int(4) construction_method;
                stream.skip(1);
                let mut byte = stream.get_bitreader();
                byte.read(4);
                entry.construction_method = byte.read(4);
                // 0: file, 1: idat.
                if entry.construction_method != 0 && entry.construction_method != 1 {
                    return Err("unknown construction_method");
                }
            }
            // unsigned int(16) data_reference_index;
            stream.skip(2);
            // unsigned int(base_offset_size*8) base_offset;
            entry.base_offset = stream.read_uxx(iloc.base_offset_size);
            // unsigned int(16) extent_count;
            entry.extent_count = stream.read_u16();
            for _j in 0..entry.extent_count {
                let mut extent: ItemLocationExtent = Default::default();
                // If extent_index is ever supported, this spec must be implemented here:
                // ::  if (((version == 1) || (version == 2)) && (index_size > 0)) {
                // ::      unsigned int(index_size*8) extent_index;
                // ::  }

                // unsigned int(offset_size*8) extent_offset;
                extent.offset = stream.read_uxx(iloc.offset_size);
                // unsigned int(length_size*8) extent_length;
                // TODO: this comment is incorrect in libavif.
                extent.length = stream.read_uxx(iloc.length_size);
                entry.extents.push(extent);
            }
            iloc.items.push(entry);
        }

        let bytes_read = stream.offset - start_offset;
        let remaining_size: u64 = size - (bytes_read as u64);
        println!("end of iloc, skiping {remaining_size} bytes");
        stream.skip(remaining_size.try_into().unwrap());
        Ok(iloc)
    }

    fn parse_pitm(stream: &mut IStream) -> Option<u32> {
        // TODO: check for multiple pitms.
        let (version, _flags) = stream.read_version_and_flags();
        if version == 0 {
            return Some(stream.read_u16() as u32);
        }
        Some(stream.read_u32())
    }

    fn parse_ispe(stream: &mut IStream) -> ItemProperty {
        // TODO: enforce version 0.
        let (_version, _flags) = stream.read_version_and_flags();
        let ispe = ImageSpatialExtents {
            // unsigned int(32) image_width;
            width: stream.read_u32(),
            // unsigned int(32) image_height;
            height: stream.read_u32(),
        };
        ItemProperty::ImageSpatialExtents(ispe)
    }

    fn parse_pixi(stream: &mut IStream) -> Option<ItemProperty> {
        // TODO: enforce version 0.
        let (_version, _flags) = stream.read_version_and_flags();
        let mut pixi: PixelInformation = Default::default();
        // unsigned int (8) num_channels;
        pixi.plane_count = stream.read_u8();
        if usize::from(pixi.plane_count) > MAX_PLANE_COUNT {
            println!("Invalid plane count in pixi box");
            return None;
        }
        for i in 0..pixi.plane_count {
            // unsigned int (8) bits_per_channel;
            pixi.plane_depths[i as usize] = stream.read_u8();
        }
        Some(ItemProperty::PixelInformation(pixi))
    }

    #[allow(non_snake_case)]
    fn parse_av1C(stream: &mut IStream, size: u64) -> Option<ItemProperty> {
        let start_offset = stream.offset;

        // unsigned int (1) marker = 1;
        // unsigned int (7) version = 1;
        let mut byte = stream.get_bitreader();
        let marker = byte.read(1);
        if marker != 1 {
            println!("Invalid marker in av1C");
            return None;
        }
        let version = byte.read(7);
        if version != 1 {
            println!("Invalid version in av1C");
            return None;
        }
        let mut av1C: CodecConfiguration = Default::default();
        // unsigned int(3) seq_profile;
        // unsigned int(5) seq_level_idx_0;
        byte = stream.get_bitreader();
        av1C.seq_profile = byte.read(3);
        av1C.seq_level_idx0 = byte.read(5);

        // unsigned int(1) seq_tier_0;
        // unsigned int(1) high_bitdepth;
        // unsigned int(1) twelve_bit;
        // unsigned int(1) monochrome;
        // unsigned int(1) chroma_subsampling_x;
        // unsigned int(1) chroma_subsampling_y;
        // unsigned int(2) chroma_sample_position;
        byte = stream.get_bitreader();
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
        let bytes_read = stream.offset - start_offset;
        let remaining_size: u64 = size - (bytes_read as u64);
        println!("end of av1C, skiping {remaining_size} bytes");
        stream.skip(remaining_size.try_into().unwrap());
        Some(ItemProperty::CodecConfiguration(av1C))
    }

    fn parse_colr(stream: &mut IStream, mut size: u64) -> Option<ItemProperty> {
        // unsigned int(32) colour_type;
        let color_type = stream.read_string(4);
        size -= 4;
        if color_type == "rICC" || color_type == "prof" {
            let mut icc: Icc = Default::default();
            // TODO: perhaps this can be a slice or something?
            icc.offset = stream.offset.try_into().unwrap();
            icc.size = size.try_into().unwrap();
            return Some(ItemProperty::ColorInformation(ColorInformation::Icc(icc)));
        }
        if color_type == "nclx" {
            let mut nclx: Nclx = Default::default();
            // unsigned int(16) colour_primaries;
            nclx.color_primaries = stream.read_u16();
            // unsigned int(16) transfer_characteristics;
            nclx.transfer_characteristics = stream.read_u16();
            // unsigned int(16) matrix_coefficients;
            nclx.matrix_coefficients = stream.read_u16();
            // unsigned int(1) full_range_flag;
            // unsigned int(7) reserved = 0;
            let mut byte = stream.get_bitreader();
            nclx.full_range = byte.read(1) == 1;
            if byte.read(7) != 0 {
                println!("colr box contains invalid reserve bits");
                return None;
            }
            return Some(ItemProperty::ColorInformation(ColorInformation::Nclx(nclx)));
        }
        None
    }

    fn parse_pasp(stream: &mut IStream) -> ItemProperty {
        let mut pasp: PixelAspectRatio = Default::default();
        // unsigned int(32) hSpacing;
        pasp.h_spacing = stream.read_u32();
        // unsigned int(32) vSpacing;
        pasp.v_spacing = stream.read_u32();
        ItemProperty::PixelAspectRatio(pasp)
    }

    #[allow(non_snake_case)]
    fn parse_auxC(stream: &mut IStream) -> ItemProperty {
        // TODO: enforce version 0.
        let (_version, _flags) = stream.read_version_and_flags();
        // string aux_type;
        let auxiliary_type = stream.read_c_string();
        ItemProperty::AuxiliaryType(auxiliary_type)
    }

    fn parse_clap(stream: &mut IStream) -> ItemProperty {
        let mut clap: ClearAperture = Default::default();
        // unsigned int(32) cleanApertureWidthN;
        clap.width_n = stream.read_u32();
        // unsigned int(32) cleanApertureWidthD;
        clap.width_d = stream.read_u32();
        // unsigned int(32) cleanApertureHeightN;
        clap.height_n = stream.read_u32();
        // unsigned int(32) cleanApertureHeightD;
        clap.height_d = stream.read_u32();
        // unsigned int(32) horizOffN;
        clap.horiz_off_n = stream.read_u32();
        // unsigned int(32) horizOffD;
        clap.horiz_off_d = stream.read_u32();
        // unsigned int(32) vertOffN;
        clap.vert_off_n = stream.read_u32();
        // unsigned int(32) vertOffD;
        clap.vert_off_d = stream.read_u32();
        ItemProperty::ClearAperture(clap)
    }

    fn parse_irot(stream: &mut IStream) -> Option<ItemProperty> {
        let mut byte = stream.get_bitreader();
        // unsigned int (6) reserved = 0;
        if byte.read(6) != 0 {
            return None;
        }
        // unsigned int (2) angle;
        let angle = byte.read(2);
        Some(ItemProperty::ImageRotation(angle))
    }

    fn parse_imir(stream: &mut IStream) -> Option<ItemProperty> {
        let mut byte = stream.get_bitreader();
        // unsigned int(7) reserved = 0;
        if byte.read(7) != 0 {
            return None;
        }
        let axis = byte.read(1);
        Some(ItemProperty::ImageMirror(axis))
    }

    fn parse_a1op(stream: &mut IStream) -> Option<ItemProperty> {
        // unsigned int(8) op_index;
        let op_index = stream.read_u8();
        if op_index > 31 {
            // 31 is AV1's maximum operating point value.
            println!("Invalid op_index in a1op");
            return None;
        }
        Some(ItemProperty::OperatingPointSelector(op_index))
    }

    fn parse_lsel(stream: &mut IStream) -> Option<ItemProperty> {
        // unsigned int(16) layer_id;
        let layer_id = stream.read_u16();
        if layer_id != 0xFFFF && layer_id >= 4 {
            println!("Invalid layer_id in lsel");
            return None;
        }
        Some(ItemProperty::LayerSelector(layer_id))
    }

    fn parse_a1lx(stream: &mut IStream) -> Option<ItemProperty> {
        let mut byte = stream.get_bitreader();
        // unsigned int(7) reserved = 0;
        if byte.read(7) != 0 {
            println!("Invalid reserve bits in a1lx");
            return None;
        }
        // unsigned int(1) large_size;
        let large_size = byte.read(1) == 1;
        let mut layer_sizes: [usize; 3] = [0; 3];
        for layer_size in &mut layer_sizes {
            if large_size {
                *layer_size = stream.read_u32() as usize;
            } else {
                *layer_size = stream.read_u16() as usize;
            }
        }
        Some(ItemProperty::AV1LayeredImageIndexing(layer_sizes))
    }

    fn parse_clli(stream: &mut IStream) -> ItemProperty {
        let mut clli: ContentLightLevelInformation = Default::default();
        // unsigned int(16) max_content_light_level
        clli.max_cll = stream.read_u16();
        // unsigned int(16) max_pic_average_light_level
        clli.max_pall = stream.read_u16();
        ItemProperty::ContentLightLevelInformation(clli)
    }

    #[allow(non_snake_case)]
    fn parse_ipco(stream: &mut IStream, mut size: u64) -> Result<Vec<ItemProperty>, &str> {
        let mut properties: Vec<ItemProperty> = Vec::new();
        while size > 0 {
            let header = Self::parse_header(stream);
            size -= header.full_size;
            match header.box_type.as_str() {
                "ispe" => {
                    properties.push(Self::parse_ispe(stream));
                }
                "pixi" => match Self::parse_pixi(stream) {
                    Some(pixi) => properties.push(pixi),
                    None => return Err("Parsing pixi failed"),
                },
                "av1C" => match Self::parse_av1C(stream, header.size) {
                    Some(av1C) => properties.push(av1C),
                    None => return Err("Parsing av1C failed"),
                },
                "colr" => match Self::parse_colr(stream, header.size) {
                    Some(colr) => properties.push(colr),
                    None => return Err("Parsing colr failed"),
                },
                "pasp" => {
                    properties.push(Self::parse_pasp(stream));
                }
                "auxC" => {
                    properties.push(Self::parse_auxC(stream));
                }
                "clap" => {
                    properties.push(Self::parse_clap(stream));
                }
                "irot" => match Self::parse_irot(stream) {
                    Some(irot) => properties.push(irot),
                    None => return Err("Parsing irot failed"),
                },
                "imir" => match Self::parse_imir(stream) {
                    Some(imir) => properties.push(imir),
                    None => return Err("Parsing imir failed"),
                },
                "a1op" => match Self::parse_a1op(stream) {
                    Some(a1op) => properties.push(a1op),
                    None => return Err("Parsing a1op failed"),
                },
                "lsel" => match Self::parse_lsel(stream) {
                    Some(lsel) => properties.push(lsel),
                    None => return Err("Parsing lsel failed"),
                },
                "a1lx" => match Self::parse_a1lx(stream) {
                    Some(a1lx) => properties.push(a1lx),
                    None => return Err("Parsing a1lx failed"),
                },
                "clli" => {
                    properties.push(Self::parse_clli(stream));
                }
                _ => {
                    println!("adding unknown property {}", header.box_type);
                    properties.push(ItemProperty::Unknown(header.box_type));
                    stream.skip(header.size.try_into().unwrap());
                }
            }
        }
        Ok(properties)
    }

    fn parse_ipma(stream: &mut IStream) -> Result<Vec<ItemPropertyAssociation>, &str> {
        let (version, flags) = stream.read_version_and_flags();
        // unsigned int(32) entry_count;
        let entry_count = stream.read_u32();
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
                entry.item_id = stream.read_u16().into();
            } else {
                // unsigned int(32) item_ID;
                entry.item_id = stream.read_u32();
            }
            if entry.item_id == 0 {
                return Err("invalid item id in ipma");
            }
            if entry.item_id <= previous_item_id {
                return Err("ipma item ids are not ordered by increasing id");
            }
            previous_item_id = entry.item_id;
            // unsigned int(8) association_count;
            let association_count = stream.read_u8();
            for _j in 0..association_count {
                // bit(1) essential;
                let mut byte = stream.get_bitreader();
                let essential = byte.read(1) == 1;
                // unsigned int(7 or 15) property_index;
                let mut property_index: u16 = byte.read(7).into();
                if (flags & 0x1) == 1 {
                    let property_index_lsb: u16 = stream.read_u8().into();
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

    fn parse_iprp(stream: &mut IStream, mut size: u64) -> Result<ItemPropertyBox, &str> {
        let orig_size = size;
        let start_offset = stream.offset;
        println!("iprp start: {start_offset}");
        let header = Self::parse_header(stream);
        if header.box_type != "ipco" {
            return Err("First box in iprp is not ipco");
        }
        let mut iprp: ItemPropertyBox = Default::default();

        match Self::parse_ipco(stream, header.size) {
            Ok(properties) => {
                iprp.properties = properties;
            }
            Err(err) => {
                // TODO: re-using err here results in some weird borrow checker error:
                // https://old.reddit.com/r/rust/comments/qi3ye9/why_does_returning_a_value_mess_with_borrows/
                return Err("ipco parsing failed");
            }
        }
        size -= header.full_size;
        while size > 0 {
            let header = Self::parse_header(stream);
            size -= header.full_size;
            if header.box_type != "ipma" {
                return Err("Found non ipma box in iprp");
            }
            match Self::parse_ipma(stream) {
                Ok(mut ipma) => iprp.associations.append(&mut ipma),
                Err(err) => {
                    // TODO: re-using err here results in some weird borrow checker error:
                    return Err("ipma parsing failed");
                }
            }
        }

        let bytes_read = stream.offset - start_offset;
        let remaining_size: u64 = orig_size - (bytes_read as u64);
        println!("end of iprp, skiping {remaining_size} bytes");
        stream.skip(remaining_size.try_into().unwrap());

        Ok(iprp)
    }

    fn parse_iinf(stream: &mut IStream, size: u64) -> Result<Vec<ItemInfo>, &str> {
        let start_offset = stream.offset;
        let (version, _flags) = stream.read_version_and_flags();
        let entry_count: u32;
        if version == 0 {
            // unsigned int(16) entry_count;
            entry_count = stream.read_u16().into();
        } else {
            // unsigned int(32) entry_count;
            entry_count = stream.read_u32();
        }
        let mut iinf: Vec<ItemInfo> = Vec::new();
        for _i in 0..entry_count {
            let header = Self::parse_header(stream);
            if header.box_type != "infe" {
                return Err("Found non infe box in iinf");
            }
            let (version, _flags) = stream.read_version_and_flags();
            if version != 2 && version != 3 {
                return Err("infe box version 2 or 3 expected.");
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
                entry.item_id = stream.read_u16().into();
            } else {
                // unsigned int(16) item_ID;
                entry.item_id = stream.read_u32();
            }
            if entry.item_id == 0 {
                return Err("Invalid item id found in infe");
            }
            // unsigned int(16) item_protection_index;
            entry.item_protection_index = stream.read_u16();
            // unsigned int(32) item_type;
            entry.item_type = stream.read_string(4);

            // TODO: libavif read vs write does not seem to match. check it out.
            // The rust code follows the spec.

            // utf8string item_name;
            entry.item_name = stream.read_c_string();
            if entry.item_type == "mime" {
                // string content_type;
                entry.content_type = stream.read_c_string();
                // string content_encoding;
                entry.content_encoding = stream.read_c_string();
            } else if entry.item_type == "uri" {
                // string item_uri_type; (skipped)
                _ = stream.read_c_string();
            }
            iinf.push(entry);
        }
        // TODO: this skip may not be necessary.
        let bytes_read = stream.offset - start_offset;
        let remaining_size: u64 = size - (bytes_read as u64);
        println!("end of iinf, skiping {remaining_size} bytes");
        stream.skip(remaining_size.try_into().unwrap());
        Ok(iinf)
    }

    fn parse_iref(stream: &mut IStream, mut size: u64) -> Result<Vec<ItemReference>, &str> {
        let orig_size = size;
        let start_offset = stream.offset;
        let (version, _flags) = stream.read_version_and_flags();
        size -= 4;
        let mut iref: Vec<ItemReference> = Vec::new();
        // versions > 1 are not supported. ignore them.
        if version <= 1 {
            while size > 0 {
                let header = Self::parse_header(stream);
                size -= header.full_size;

                let from_item_id: u32;
                if version == 0 {
                    // unsigned int(16) from_item_ID;
                    from_item_id = stream.read_u16().into();
                } else {
                    // unsigned int(32) from_item_ID;
                    from_item_id = stream.read_u32();
                }
                if from_item_id == 0 {
                    return Err("invalid from_item_id in iref");
                }
                // unsigned int(16) reference_count;
                let reference_count = stream.read_u16();
                for reference_index in 0..reference_count {
                    let to_item_id: u32;
                    if version == 0 {
                        // unsigned int(16) to_item_ID;
                        to_item_id = stream.read_u16().into();
                    } else {
                        // unsigned int(32) to_item_ID;
                        to_item_id = stream.read_u32();
                    }
                    if to_item_id == 0 {
                        return Err("invalid to_item_id in iref");
                    }
                    iref.push(ItemReference {
                        from_item_id,
                        to_item_id,
                        reference_type: header.box_type.clone(),
                    });
                }
            }
        }
        let bytes_read = stream.offset - start_offset;
        let remaining_size: u64 = orig_size - (bytes_read as u64);
        println!("end of iref, skiping {remaining_size} bytes");
        stream.skip(remaining_size.try_into().unwrap());
        Ok(iref)
    }

    fn parse_idat(stream: &mut IStream, size: u64) -> Result<ItemData, &str> {
        // TODO: check if multiple idats were seen for this meta box.
        if size == 0 {
            return Err("Invalid idat size");
        }
        Ok(ItemData {
            size: size,
            offset: stream.offset,
        })
    }

    fn parse_meta(stream: &mut IStream, mut size: u64) -> MetaBox {
        println!("parsing meta size: {size}");
        // TODO: version must be 0.
        let (_version, _flags) = stream.read_version_and_flags();
        size -= 4;
        let mut first_box = true;
        let mut meta: MetaBox = Default::default();
        let empty: MetaBox = Default::default();

        // TODO: add box unique checks.

        while size > 0 {
            let header = Self::parse_header(stream);
            size -= header.full_size;
            //println!("{:#?}", header);
            if first_box {
                if header.box_type != "hdlr" {
                    println!("first box in meta is not hdlr");
                    return empty;
                }
                if !Self::parse_hdlr(stream) {
                    return empty;
                }
                first_box = false;
                continue;
            }
            match header.box_type.as_str() {
                "iloc" => {
                    meta.iloc = match Self::parse_iloc(stream, header.size) {
                        Ok(iloc) => iloc,
                        Err(err) => {
                            println!("Parsing iloc failed: {err}");
                            return empty;
                        }
                    };
                }
                "pitm" => {
                    meta.primary_item_id = match Self::parse_pitm(stream) {
                        Some(item_id) => item_id,
                        None => {
                            println!("Error parsing pitm box.");
                            return empty;
                        }
                    }
                }
                "iprp" => {
                    meta.iprp = match Self::parse_iprp(stream, header.size) {
                        Ok(iprp) => iprp,
                        Err(err) => {
                            println!("Parsing iprp failed: {err}");
                            return empty;
                        }
                    };
                }
                "iinf" => {
                    meta.iinf = match Self::parse_iinf(stream, header.size) {
                        Ok(iinf) => iinf,
                        Err(err) => {
                            println!("Parsing iinf failed: {err}");
                            return empty;
                        }
                    };
                }
                "iref" => {
                    meta.iref = match Self::parse_iref(stream, header.size) {
                        Ok(iref) => iref,
                        Err(err) => {
                            println!("Parsing iref failed: {err}");
                            return empty;
                        }
                    }
                }
                "idat" => {
                    meta.idat = match Self::parse_idat(stream, header.size) {
                        Ok(idat) => idat,
                        Err(err) => {
                            println!("Parsing idat failed: {err}");
                            return empty;
                        }
                    }
                }
                _ => {
                    // TODO: handle idat.
                    println!("skipping box {}", header.box_type);
                    stream.skip(header.size.try_into().unwrap());
                }
            }
        }
        if first_box {
            // The meta box must not be empty (it must contain at least a hdlr box).
            println!("Meta box has no child boxes");
            return empty;
        }
        meta
    }

    fn parse_tkhd(stream: &mut IStream, track: &mut AvifTrack) -> bool {
        let (version, _flags) = stream.read_version_and_flags();
        if version == 1 {
            // unsigned int(64) creation_time;
            stream.skip_u64();
            // unsigned int(64) modification_time;
            stream.skip_u64();
            // unsigned int(32) track_ID;
            track.id = stream.read_u32();
            // const unsigned int(32) reserved = 0;
            stream.skip_u32();
            // unsigned int(64) duration;
            track.track_duration = stream.read_u64();
        } else if version == 0 {
            // unsigned int(32) creation_time;
            stream.skip_u32();
            // unsigned int(32) modification_time;
            stream.skip_u32();
            // unsigned int(32) track_ID;
            track.id = stream.read_u32();
            // const unsigned int(32) reserved = 0;
            stream.skip_u32();
            // unsigned int(32) duration;
            track.track_duration = stream.read_u32() as u64;
        } else {
            println!("unsupported version in trak");
            return false;
        }

        // Skip the following 52 bytes.
        // const unsigned int(32)[2] reserved = 0;
        // template int(16) layer = 0;
        // template int(16) alternate_group = 0;
        // template int(16) volume = {if track_is_audio 0x0100 else 0};
        // const unsigned int(16) reserved = 0;
        // template int(32)[9] matrix= { 0x00010000,0,0,0,0x00010000,0,0,0,0x40000000 }; // unity matrix
        stream.skip(52);

        // unsigned int(32) width;
        track.width = stream.read_u32() >> 16;
        // unsigned int(32) height;
        track.height = stream.read_u32() >> 16;

        if track.width == 0 || track.height == 0 {
            println!("invalid track dimensions");
            return false;
        }

        // TODO: check if track dims are too large.

        true
    }

    fn parse_mdhd(stream: &mut IStream, track: &mut AvifTrack) -> bool {
        let (version, _flags) = stream.read_version_and_flags();
        if version == 1 {
            // unsigned int(64) creation_time;
            stream.skip_u64();
            // unsigned int(64) modification_time;
            stream.skip_u64();
            // unsigned int(32) timescale;
            track.media_timescale = stream.read_u32();
            // unsigned int(64) duration;
            track.media_duration = stream.read_u64();
        } else if version == 0 {
            // unsigned int(32) creation_time;
            stream.skip_u32();
            // unsigned int(32) modification_time;
            stream.skip_u32();
            // unsigned int(32) timescale;
            track.media_timescale = stream.read_u32();
            // unsigned int(32) duration;
            track.media_duration = stream.read_u32() as u64;
        } else {
            println!("unsupported version in mdhd");
            return false;
        }

        // Skip the following 4 bytes.
        // bit(1) pad = 0;
        // unsigned int(5)[3] language; // ISO-639-2/T language code
        // unsigned int(16) pre_defined = 0;
        stream.skip(4);

        println!("track after mdhd: {:#?}", track);
        true
    }

    fn parse_stco(
        stream: &mut IStream,
        sample_table: &mut SampleTable,
        large_offset: bool,
    ) -> bool {
        // TODO: version must be 0.
        let (_version, _flags) = stream.read_version_and_flags();
        // unsigned int(32) entry_count;
        let entry_count = stream.read_u32();
        sample_table.chunk_offsets.reserve(entry_count as usize);
        for i in 0..entry_count {
            let chunk_offset: u64;
            if large_offset {
                // TODO: this comment is wrong in libavif.
                // unsigned int(64) chunk_offset;
                chunk_offset = stream.read_u64();
            } else {
                // unsigned int(32) chunk_offset;
                chunk_offset = stream.read_u32() as u64;
            }
            sample_table.chunk_offsets.push(chunk_offset);
        }
        true
    }

    fn parse_stsc(stream: &mut IStream, sample_table: &mut SampleTable) -> bool {
        // TODO: version must be 0.
        let (_version, _flags) = stream.read_version_and_flags();
        // unsigned int(32) entry_count;
        let entry_count = stream.read_u32();
        sample_table.sample_to_chunk.reserve(entry_count as usize);
        for i in 0..entry_count {
            let mut stsc: SampleToChunk = Default::default();
            // unsigned int(32) first_chunk;
            stsc.first_chunk = stream.read_u32();
            // unsigned int(32) samples_per_chunk;
            stsc.samples_per_chunk = stream.read_u32();
            // unsigned int(32) sample_description_index;
            stsc.sample_description_index = stream.read_u32();

            if i == 0 {
                if stsc.first_chunk != 1 {
                    println!("stsc does not begin with chunk 1.");
                    return false;
                }
            } else {
                if stsc.first_chunk <= sample_table.sample_to_chunk.last().unwrap().first_chunk {
                    println!("stsc chunks are not strictly increasing.");
                    return false;
                }
            }
            sample_table.sample_to_chunk.push(stsc);
        }
        true
    }

    fn parse_stsz(stream: &mut IStream, sample_table: &mut SampleTable) -> bool {
        // TODO: version must be 0.
        let (_version, _flags) = stream.read_version_and_flags();
        // unsigned int(32) sample_size;
        sample_table.all_samples_size = stream.read_u32();
        // unsigned int(32) sample_count;
        let sample_count = stream.read_u32();

        if sample_table.all_samples_size > 0 {
            return true;
        }
        sample_table.sample_sizes.reserve(sample_count as usize);
        for i in 0..sample_count {
            // unsigned int(32) entry_size;
            let entry_size = stream.read_u32();
            sample_table.sample_sizes.push(entry_size);
        }
        true
    }

    fn parse_stss(stream: &mut IStream, sample_table: &mut SampleTable) -> bool {
        // TODO: version must be 0.
        let (_version, _flags) = stream.read_version_and_flags();
        // unsigned int(32) entry_count;
        let entry_count = stream.read_u32();
        sample_table.sync_samples.reserve(entry_count as usize);
        for i in 0..entry_count {
            // unsigned int(32) sample_number;
            let sample_number = stream.read_u32();
            sample_table.sync_samples.push(sample_number);
        }
        true
    }

    fn parse_stts(stream: &mut IStream, sample_table: &mut SampleTable) -> bool {
        // TODO: version must be 0.
        let (_version, _flags) = stream.read_version_and_flags();
        // unsigned int(32) entry_count;
        let entry_count = stream.read_u32();
        sample_table.time_to_sample.reserve(entry_count as usize);
        for i in 0..entry_count {
            let mut stts: TimeToSample = Default::default();
            // unsigned int(32) sample_count;
            stts.sample_count = stream.read_u32();
            // unsigned int(32) sample_delta;
            stts.sample_delta = stream.read_u32();
            sample_table.time_to_sample.push(stts);
        }
        true
    }

    fn parse_stsd(stream: &mut IStream, sample_table: &mut SampleTable) -> bool {
        // TODO: version must be 0.
        let (_version, _flags) = stream.read_version_and_flags();
        // unsigned int(32) entry_count;
        let entry_count = stream.read_u32();
        sample_table
            .sample_descriptions
            .reserve(entry_count as usize);
        for i in 0..entry_count {
            let header = Self::parse_header(stream);
            let mut stsd: SampleDescription = Default::default();
            stsd.format = header.box_type.clone();
            if stsd.format == "av01" {
                // Skip 78 bytes for visual sample entry size.
                stream.skip(78);
                // TODO: check subtraction is ok.
                match Self::parse_ipco(stream, header.size - 78) {
                    Ok(properties) => stsd.properties = properties,
                    Err(err) => {
                        println!("{}", err);
                        return false;
                    }
                }
            }
            sample_table.sample_descriptions.push(stsd);
        }
        true
    }

    fn parse_stbl(stream: &mut IStream, track: &mut AvifTrack, mut size: u64) -> bool {
        if track.sample_table.is_some() {
            println!("duplciate stbl for track.");
            return false;
        }
        let mut sample_table: SampleTable = Default::default();
        while size > 0 {
            let header = Self::parse_header(stream);
            size -= header.full_size;
            match header.box_type.as_str() {
                "stco" => {
                    if !Self::parse_stco(stream, &mut sample_table, false) {
                        return false;
                    }
                }
                "co64" => {
                    if !Self::parse_stco(stream, &mut sample_table, true) {
                        return false;
                    }
                }
                "stsc" => {
                    if !Self::parse_stsc(stream, &mut sample_table) {
                        return false;
                    }
                }
                "stsz" => {
                    if !Self::parse_stsz(stream, &mut sample_table) {
                        return false;
                    }
                }
                "stss" => {
                    if !Self::parse_stss(stream, &mut sample_table) {
                        return false;
                    }
                }
                "stts" => {
                    if !Self::parse_stts(stream, &mut sample_table) {
                        return false;
                    }
                }
                "stsd" => {
                    if !Self::parse_stsd(stream, &mut sample_table) {
                        return false;
                    }
                }
                _ => {
                    println!("skipping box {}", header.box_type);
                    stream.skip(header.size.try_into().unwrap());
                }
            }
        }
        track.sample_table = Some(sample_table);
        true
    }

    fn parse_minf(stream: &mut IStream, track: &mut AvifTrack, mut size: u64) -> bool {
        while size > 0 {
            let header = Self::parse_header(stream);
            size -= header.full_size;
            match header.box_type.as_str() {
                "stbl" => {
                    if !Self::parse_stbl(stream, track, header.size) {
                        return false;
                    }
                }
                _ => {
                    println!("skipping box {}", header.box_type);
                    stream.skip(header.size.try_into().unwrap());
                }
            }
        }
        true
    }

    fn parse_mdia(stream: &mut IStream, track: &mut AvifTrack, mut size: u64) -> bool {
        while size > 0 {
            let header = Self::parse_header(stream);
            size -= header.full_size;
            match header.box_type.as_str() {
                "mdhd" => {
                    if !Self::parse_mdhd(stream, track) {
                        return false;
                    }
                }
                "minf" => {
                    if !Self::parse_minf(stream, track, header.size) {
                        return false;
                    }
                }
                _ => {
                    println!("skipping box {}", header.box_type);
                    stream.skip(header.size.try_into().unwrap());
                }
            }
        }
        true
    }

    fn parse_tref(stream: &mut IStream, track: &mut AvifTrack, mut size: u64) -> bool {
        while size > 0 {
            let header = Self::parse_header(stream);
            size -= header.full_size;
            match header.box_type.as_str() {
                "auxl" => {
                    // unsigned int(32) track_IDs[];
                    track.aux_for_id = stream.read_u32(); // Use the first one.
                    stream.skip((header.size - 4) as usize); // Skip the rest.
                }
                "prem" => {
                    // unsigned int(32) track_IDs[];
                    track.prem_by_id = stream.read_u32(); // Use the first one.
                    stream.skip((header.size - 4) as usize); // Skip the rest.
                }
                _ => {
                    println!("skipping box {}", header.box_type);
                    stream.skip(header.size.try_into().unwrap());
                }
            }
        }
        true
    }

    fn parse_trak(stream: &mut IStream, mut size: u64) -> Option<AvifTrack> {
        let mut track: AvifTrack = Default::default();
        println!("parsing trak size: {size}");
        while size > 0 {
            let header = Self::parse_header(stream);
            size -= header.full_size;
            match header.box_type.as_str() {
                "tkhd" => {
                    if !Self::parse_tkhd(stream, &mut track) {
                        return None;
                    }
                }
                "mdia" => {
                    if !Self::parse_mdia(stream, &mut track, header.size) {
                        return None;
                    }
                }
                "tref" => {
                    if !Self::parse_tref(stream, &mut track, header.size) {
                        return None;
                    }
                }
                _ => {
                    // TODO: edts.
                    // TODO: track meta can be ignored? probably not becuase of xmp/exif.
                    println!("skipping box {}", header.box_type);
                    stream.skip(header.size.try_into().unwrap());
                }
            }
            println!("track after {}: {:#?}", header.box_type, track);
        }
        // TODO: compute repetition count here.
        Some(track)
    }

    fn parse_moov(stream: &mut IStream, mut size: u64) -> Option<MovieBox> {
        println!("parsing moov size: {size}");
        let mut moov: MovieBox = Default::default();
        while size > 0 {
            let header = Self::parse_header(stream);
            size -= header.full_size;
            match header.box_type.as_str() {
                "trak" => match Self::parse_trak(stream, header.size) {
                    Some(trak) => moov.tracks.push(trak),
                    None => return None,
                },
                _ => {
                    println!("skipping box {}", header.box_type);
                    stream.skip(header.size.try_into().unwrap());
                }
            }
        }
        Some(moov)
    }

    pub fn parse(stream: &mut IStream) -> AvifBoxes {
        let mut ftyp_seen = false;
        let mut avif_boxes: AvifBoxes = Default::default();
        let mut meta_seen = false;
        loop {
            let header = MP4Box::parse_header(stream);
            println!("{:#?}", header);
            match header.box_type.as_str() {
                "ftyp" => {
                    avif_boxes.ftyp = MP4Box::parse_ftyp(stream, header.size);
                    ftyp_seen = true;
                }
                "meta" => {
                    avif_boxes.meta = MP4Box::parse_meta(stream, header.size);
                    meta_seen = true;
                    //break;
                }
                "moov" => {
                    match MP4Box::parse_moov(stream, header.size) {
                        Some(moov) => avif_boxes.moov = moov,
                        None => {
                            // TODO: lol panic.
                            panic!("error parsing moov");
                        }
                    }
                    break;
                }
                _ => {
                    println!("unknown box: {}", header.box_type);
                    break;
                }
            }
        }
        println!("{:#?}", avif_boxes);
        avif_boxes
    }
}
