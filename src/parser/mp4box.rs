use crate::decoder::gainmap::GainMapMetadata;
use crate::decoder::track::*;
use crate::decoder::GenericIO;
use crate::image::MAX_PLANE_COUNT;
use crate::internal_utils::stream::*;
use crate::internal_utils::*;
use crate::utils::clap::CleanAperture;
use crate::*;

use std::collections::HashSet;

#[derive(Debug)]
struct BoxHeader {
    size: usize,
    box_type: String,
}

#[derive(Debug)]
pub struct FileTypeBox {
    pub major_brand: String,
    #[allow(unused)]
    minor_version: u32,
    compatible_brands: Vec<String>,
}

impl FileTypeBox {
    fn has_brand(&self, brand: &str) -> bool {
        if self.major_brand.as_str() == brand {
            return true;
        }
        self.compatible_brands.iter().any(|x| x.as_str() == brand)
    }

    pub fn is_avif(&self) -> bool {
        self.has_brand("avif") || self.has_brand("avis")
    }

    pub fn needs_meta(&self) -> bool {
        self.has_brand("avif")
    }

    pub fn needs_moov(&self) -> bool {
        self.has_brand("avis")
    }
}

#[derive(Debug)]
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

#[derive(Debug, Clone)]
pub struct ImageSpatialExtents {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Default, Clone)]
pub struct PixelInformation {
    pub plane_count: u8,
    pub plane_depths: [u8; MAX_PLANE_COUNT],
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct CodecConfiguration {
    pub seq_profile: u8,
    pub seq_level_idx0: u8,
    pub seq_tier0: u8,
    pub high_bitdepth: bool,
    pub twelve_bit: bool,
    pub monochrome: bool,
    pub chroma_subsampling_x: u8,
    pub chroma_subsampling_y: u8,
    pub chroma_sample_position: ChromaSamplePosition,
}

impl CodecConfiguration {
    pub fn depth(&self) -> u8 {
        match self.twelve_bit {
            true => 12,
            false => match self.high_bitdepth {
                true => 10,
                false => 8,
            },
        }
    }

    pub fn pixel_format(&self) -> PixelFormat {
        if self.monochrome {
            PixelFormat::Monochrome
        } else if self.chroma_subsampling_x == 1 && self.chroma_subsampling_y == 1 {
            PixelFormat::Yuv420
        } else if self.chroma_subsampling_x == 1 {
            PixelFormat::Yuv422
        } else {
            PixelFormat::Yuv444
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct Nclx {
    pub color_primaries: ColorPrimaries,
    pub transfer_characteristics: TransferCharacteristics,
    pub matrix_coefficients: MatrixCoefficients,
    pub full_range: bool,
}

#[derive(Debug, Clone)]
#[allow(unused)]
pub enum ColorInformation {
    Icc(Vec<u8>),
    Nclx(Nclx),
}

/// cbindgen:rename-all=CamelCase
#[derive(Debug, Clone, Default, Copy)]
#[repr(C)]
pub struct PixelAspectRatio {
    pub h_spacing: u32,
    pub v_spacing: u32,
}

/// cbindgen:field-names=[maxCLL, maxPALL]
#[repr(C)]
#[derive(Debug, Clone, Default, Copy)]
pub struct ContentLightLevelInformation {
    pub max_cll: u16,
    pub max_pall: u16,
}

#[derive(Debug, Clone)]
pub enum ItemProperty {
    ImageSpatialExtents(ImageSpatialExtents),
    PixelInformation(PixelInformation),
    CodecConfiguration(CodecConfiguration),
    ColorInformation(ColorInformation),
    PixelAspectRatio(PixelAspectRatio),
    AuxiliaryType(String),
    CleanAperture(CleanAperture),
    ImageRotation(u8),
    ImageMirror(u8),
    OperatingPointSelector(u8),
    LayerSelector(u16),
    AV1LayeredImageIndexing([usize; 3]),
    ContentLightLevelInformation(ContentLightLevelInformation),
    Unknown(String),
}

#[derive(Debug, Default)]
#[allow(unused)]
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
}

#[derive(Debug, Default)]
pub struct ItemPropertyBox {
    pub properties: Vec<ItemProperty>,
    pub associations: Vec<ItemPropertyAssociation>,
}

#[derive(Debug)]
pub struct ItemReference {
    // Read this reference as "{from_item_id} is a {reference_type} for{to_item_id}" (except for
    // dimg where it is in the opposite direction).
    pub from_item_id: u32,
    pub to_item_id: u32,
    pub reference_type: String,
    pub index: u32, // Index of the reference within the iref type.
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

#[derive(Debug)]
pub struct AvifBoxes {
    pub ftyp: FileTypeBox,
    pub meta: MetaBox,
    pub tracks: Vec<Track>,
}

fn parse_header(stream: &mut IStream) -> AvifResult<BoxHeader> {
    let start_offset = stream.offset;
    let mut size = stream.read_u32()? as u64;
    let box_type = stream.read_string(4)?;
    println!("box_type: {}", box_type);
    if size == 1 {
        size = stream.read_u64()?;
    }
    if box_type == "uuid" {
        stream.skip(16)?;
    }
    size = size
        .checked_sub((stream.offset - start_offset) as u64)
        .ok_or(AvifError::BmffParseFailed)?;
    Ok(BoxHeader {
        box_type,
        size: usize_from_u64(size)?,
    })
}

fn parse_ftyp(stream: &mut IStream) -> AvifResult<FileTypeBox> {
    let major_brand = stream.read_string(4)?;
    let minor_version = stream.read_u32()?;
    let mut compatible_brands: Vec<String> = Vec::new();
    while stream.has_bytes_left() {
        compatible_brands.push(stream.read_string(4)?);
    }
    Ok(FileTypeBox {
        major_brand,
        minor_version,
        compatible_brands,
    })
}

fn parse_hdlr(stream: &mut IStream) -> AvifResult<()> {
    let (_version, _flags) = stream.read_and_enforce_version_and_flags(0)?;
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
    let mut iloc = ItemLocationBox::default();
    let mut bits = stream.sub_bit_stream(2)?;
    // unsigned int(4) offset_size;
    iloc.offset_size = bits.read(4)? as u8;
    // unsigned int(4) length_size;
    iloc.length_size = bits.read(4)? as u8;
    // unsigned int(4) base_offset_size;
    iloc.base_offset_size = bits.read(4)? as u8;
    if (version == 1 || version == 2) && iloc.base_offset_size != 0 {
        println!("Invalid base_offset_size in iloc.");
        return Err(AvifError::BmffParseFailed);
    }
    // unsigned int(4) reserved; The last 4 bits left in the bits.
    let item_count: u32 = if version < 2 {
        // unsigned int(16) item_count;
        stream.read_u16()? as u32
    } else {
        // unsigned int(32) item_count;
        stream.read_u32()?
    };
    for _i in 0..item_count {
        let mut entry = ItemLocationEntry {
            item_id: if version < 2 {
                // unsigned int(16) item_ID;
                stream.read_u16()? as u32
            } else {
                // unsigned int(32) item_ID;
                stream.read_u32()?
            },
            ..ItemLocationEntry::default()
        };
        if entry.item_id == 0 {
            println!("Invalid item id.");
            return Err(AvifError::BmffParseFailed);
        }
        if version == 1 || version == 2 {
            // unsigned int(12) reserved = 0;
            // unsigned int(4) construction_method;
            stream.skip(1)?;
            let mut bits = stream.sub_bit_stream(1)?;
            bits.read(4)?;
            entry.construction_method = bits.read(4)? as u8;
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
            let extent = ItemLocationExtent {
                // unsigned int(offset_size*8) extent_offset;
                offset: stream.read_uxx(iloc.offset_size)?,
                // unsigned int(length_size*8) extent_length;
                length: stream.read_uxx(iloc.length_size)?,
            };
            entry.extents.push(extent);
        }
        iloc.items.push(entry);
    }

    println!("end of iloc, skiping {} bytes", stream.bytes_left());
    Ok(iloc)
}

fn parse_pitm(stream: &mut IStream) -> AvifResult<u32> {
    let (version, _flags) = stream.read_version_and_flags()?;
    let primary_item_id = if version == 0 {
        stream.read_u16()? as u32
    } else {
        stream.read_u32()?
    };
    Ok(primary_item_id)
}

fn parse_ispe(stream: &mut IStream) -> AvifResult<ItemProperty> {
    let (_version, _flags) = stream.read_and_enforce_version_and_flags(0)?;
    let ispe = ImageSpatialExtents {
        // unsigned int(32) image_width;
        width: stream.read_u32()?,
        // unsigned int(32) image_height;
        height: stream.read_u32()?,
    };
    Ok(ItemProperty::ImageSpatialExtents(ispe))
}

fn parse_pixi(stream: &mut IStream) -> AvifResult<ItemProperty> {
    let (_version, _flags) = stream.read_and_enforce_version_and_flags(0)?;
    let mut pixi = PixelInformation {
        // unsigned int (8) num_channels;
        plane_count: stream.read_u8()?,
        ..PixelInformation::default()
    };
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
    let mut bits = stream.sub_bit_stream(3)?;
    let marker = bits.read(1)?;
    if marker != 1 {
        println!("Invalid marker in av1C");
        return Err(AvifError::BmffParseFailed);
    }
    let version = bits.read(7)?;
    if version != 1 {
        println!("Invalid version in av1C");
        return Err(AvifError::BmffParseFailed);
    }
    let av1C = CodecConfiguration {
        // unsigned int(3) seq_profile;
        // unsigned int(5) seq_level_idx_0;
        seq_profile: bits.read(3)? as u8,
        seq_level_idx0: bits.read(5)? as u8,
        // unsigned int(1) seq_tier_0;
        // unsigned int(1) high_bitdepth;
        // unsigned int(1) twelve_bit;
        // unsigned int(1) monochrome;
        // unsigned int(1) chroma_subsampling_x;
        // unsigned int(1) chroma_subsampling_y;
        // unsigned int(2) chroma_sample_position;
        seq_tier0: bits.read(1)? as u8,
        high_bitdepth: bits.read_bool()?,
        twelve_bit: bits.read_bool()?,
        monochrome: bits.read_bool()?,
        chroma_subsampling_x: bits.read(1)? as u8,
        chroma_subsampling_y: bits.read(1)? as u8,
        chroma_sample_position: bits.read(2)?.into(),
    };

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
        return Ok(Some(ItemProperty::ColorInformation(ColorInformation::Icc(
            stream.get_slice(stream.bytes_left())?.to_vec(),
        ))));
    }
    if color_type == "nclx" {
        let mut nclx = Nclx {
            // unsigned int(16) colour_primaries;
            color_primaries: stream.read_u16()?.into(),
            // unsigned int(16) transfer_characteristics;
            transfer_characteristics: stream.read_u16()?.into(),
            // unsigned int(16) matrix_coefficients;
            matrix_coefficients: stream.read_u16()?.into(),
            ..Nclx::default()
        };
        // unsigned int(1) full_range_flag;
        // unsigned int(7) reserved = 0;
        let mut bits = stream.sub_bit_stream(1)?;
        nclx.full_range = bits.read_bool()?;
        if bits.read(7)? != 0 {
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
    let pasp = PixelAspectRatio {
        // unsigned int(32) hSpacing;
        h_spacing: stream.read_u32()?,
        // unsigned int(32) vSpacing;
        v_spacing: stream.read_u32()?,
    };
    Ok(ItemProperty::PixelAspectRatio(pasp))
}

#[allow(non_snake_case)]
fn parse_auxC(stream: &mut IStream) -> AvifResult<ItemProperty> {
    let (_version, _flags) = stream.read_and_enforce_version_and_flags(0)?;
    // string aux_type;
    let auxiliary_type = stream.read_c_string()?;
    Ok(ItemProperty::AuxiliaryType(auxiliary_type))
}

fn parse_clap(stream: &mut IStream) -> AvifResult<ItemProperty> {
    let clap = CleanAperture {
        // unsigned int(32) cleanApertureWidthN;
        // unsigned int(32) cleanApertureWidthD;
        width: stream.read_ufraction()?,
        // unsigned int(32) cleanApertureHeightN;
        // unsigned int(32) cleanApertureHeightD;
        height: stream.read_ufraction()?,
        // unsigned int(32) horizOffN;
        // unsigned int(32) horizOffD;
        horiz_off: stream.read_ufraction()?,
        // unsigned int(32) vertOffN;
        // unsigned int(32) vertOffD;
        vert_off: stream.read_ufraction()?,
    };
    Ok(ItemProperty::CleanAperture(clap))
}

fn parse_irot(stream: &mut IStream) -> AvifResult<ItemProperty> {
    let mut bits = stream.sub_bit_stream(1)?;
    // unsigned int (6) reserved = 0;
    if bits.read(6)? != 0 {
        println!("invalid reserve bits in irot");
        return Err(AvifError::BmffParseFailed);
    }
    // unsigned int (2) angle;
    let angle = bits.read(2)? as u8;
    Ok(ItemProperty::ImageRotation(angle))
}

fn parse_imir(stream: &mut IStream) -> AvifResult<ItemProperty> {
    let mut bits = stream.sub_bit_stream(1)?;
    // unsigned int(7) reserved = 0;
    if bits.read(7)? != 0 {
        println!("invalid reserve bits in imir");
        return Err(AvifError::BmffParseFailed);
    }
    let axis = bits.read(1)? as u8;
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
    let mut bits = stream.sub_bit_stream(1)?;
    // unsigned int(7) reserved = 0;
    if bits.read(7)? != 0 {
        println!("Invalid reserve bits in a1lx");
        return Err(AvifError::BmffParseFailed);
    }
    // unsigned int(1) large_size;
    let large_size = bits.read_bool()?;
    let mut layer_sizes: [usize; 3] = [0; 3];
    for layer_size in &mut layer_sizes {
        if large_size {
            *layer_size = usize_from_u32(stream.read_u32()?)?;
        } else {
            *layer_size = usize_from_u16(stream.read_u16()?)?;
        }
    }
    Ok(ItemProperty::AV1LayeredImageIndexing(layer_sizes))
}

fn parse_clli(stream: &mut IStream) -> AvifResult<ItemProperty> {
    let clli = ContentLightLevelInformation {
        // unsigned int(16) max_content_light_level
        max_cll: stream.read_u16()?,
        // unsigned int(16) max_pic_average_light_level
        max_pall: stream.read_u16()?,
    };
    Ok(ItemProperty::ContentLightLevelInformation(clli))
}

#[allow(non_snake_case)]
fn parse_ipco(stream: &mut IStream) -> AvifResult<Vec<ItemProperty>> {
    let mut properties: Vec<ItemProperty> = Vec::new();
    while stream.has_bytes_left() {
        let header = parse_header(stream)?;
        let mut sub_stream = stream.sub_stream(header.size)?;
        match header.box_type.as_str() {
            "ispe" => properties.push(parse_ispe(&mut sub_stream)?),
            "pixi" => properties.push(parse_pixi(&mut sub_stream)?),
            "av1C" => properties.push(parse_av1C(&mut sub_stream)?),
            "colr" => {
                if let Some(colr) = parse_colr(&mut sub_stream)? {
                    properties.push(colr)
                }
            }
            "pasp" => properties.push(parse_pasp(&mut sub_stream)?),
            "auxC" => properties.push(parse_auxC(&mut sub_stream)?),
            "clap" => properties.push(parse_clap(&mut sub_stream)?),
            "irot" => properties.push(parse_irot(&mut sub_stream)?),
            "imir" => properties.push(parse_imir(&mut sub_stream)?),
            "a1op" => properties.push(parse_a1op(&mut sub_stream)?),
            "lsel" => properties.push(parse_lsel(&mut sub_stream)?),
            "a1lx" => properties.push(parse_a1lx(&mut sub_stream)?),
            "clli" => properties.push(parse_clli(&mut sub_stream)?),
            _ => properties.push(ItemProperty::Unknown(header.box_type)),
        }
    }
    Ok(properties)
}

fn parse_ipma(stream: &mut IStream) -> AvifResult<Vec<ItemPropertyAssociation>> {
    let (version, flags) = stream.read_version_and_flags()?;
    // unsigned int(32) entry_count;
    let entry_count = stream.read_u32()?;
    let mut ipma: Vec<ItemPropertyAssociation> = Vec::new();
    for _i in 0..entry_count {
        let mut entry = ItemPropertyAssociation {
            version,
            flags,
            ..ItemPropertyAssociation::default()
        };
        // ISO/IEC 23008-12, First edition, 2017-12, Section 9.3.1:
        //   Each ItemPropertyAssociation box shall be ordered by increasing item_ID, and there
        //   shall be at most one association box for each item_ID, in any
        //   ItemPropertyAssociation box.
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
        if !ipma.is_empty() {
            let previous_item_id = ipma.last().unwrap().item_id;
            if entry.item_id <= previous_item_id {
                println!("ipma item ids are not ordered by increasing id");
                return Err(AvifError::BmffParseFailed);
            }
        }
        // unsigned int(8) association_count;
        let association_count = stream.read_u8()?;
        for _j in 0..association_count {
            // bit(1) essential;
            let mut bits = stream.sub_bit_stream(1)?;
            let essential = bits.read_bool()?;
            // unsigned int(7 or 15) property_index;
            let mut property_index: u16 = bits.read(7)? as u16;
            if (flags & 0x1) == 1 {
                let property_index_lsb: u16 = stream.read_u8()? as u16;
                property_index <<= 8;
                property_index |= property_index_lsb;
            }
            entry.associations.push((property_index, essential));
        }
        ipma.push(entry);
    }
    Ok(ipma)
}

fn parse_iprp(stream: &mut IStream) -> AvifResult<ItemPropertyBox> {
    println!("iprp start: {}", stream.offset);
    let header = parse_header(stream)?;
    if header.box_type != "ipco" {
        println!("First box in iprp is not ipco");
        return Err(AvifError::BmffParseFailed);
    }
    let mut iprp = ItemPropertyBox::default();
    // Parse ipco box.
    {
        let mut sub_stream = stream.sub_stream(header.size)?;
        iprp.properties = parse_ipco(&mut sub_stream)?;
    }
    // Parse ipma boxes.
    while stream.has_bytes_left() {
        let header = parse_header(stream)?;
        if header.box_type != "ipma" {
            println!("Found non ipma box in iprp");
            return Err(AvifError::BmffParseFailed);
        }
        let mut sub_stream = stream.sub_stream(header.size)?;
        iprp.associations.append(&mut parse_ipma(&mut sub_stream)?);
    }
    println!("end of iprp, skiping {} bytes", stream.bytes_left());
    Ok(iprp)
}

fn parse_infe(stream: &mut IStream) -> AvifResult<ItemInfo> {
    let (version, _flags) = stream.read_version_and_flags()?;
    if version != 2 && version != 3 {
        println!("infe box version 2 or 3 expected.");
        return Err(AvifError::BmffParseFailed);
    }

    // TODO: check flags. ISO/IEC 23008-12:2017, Section 9.2 says:
    // The flags field of ItemInfoEntry with version greater than or equal to 2 is specified
    // as follows:
    //   (flags & 1) equal to 1 indicates that the item is not intended to be a part of the
    //   presentation. For example, when (flags & 1) is equal to 1 for an image item, the
    //   image item should not be displayed.(flags & 1) equal to 0 indicates that the item
    //   is intended to be a part of the presentation.
    //
    // See also Section 6.4.2.
    let mut entry = ItemInfo::default();
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
    // The rust code follows libavif.

    if entry.item_type == "mime" {
        // string item_name;
        entry.item_name = stream.read_c_string()?;
        // string content_type;
        entry.content_type = stream.read_c_string()?;
    }
    Ok(entry)
}

fn parse_iinf(stream: &mut IStream) -> AvifResult<Vec<ItemInfo>> {
    let (version, _flags) = stream.read_version_and_flags()?;
    let entry_count: u32 = if version == 0 {
        // unsigned int(16) entry_count;
        stream.read_u16()? as u32
    } else {
        // unsigned int(32) entry_count;
        stream.read_u32()?
    };
    let mut iinf: Vec<ItemInfo> = Vec::new();
    for _i in 0..entry_count {
        let header = parse_header(stream)?;
        if header.box_type != "infe" {
            println!("Found non infe box in iinf");
            return Err(AvifError::BmffParseFailed);
        }
        let mut sub_stream = stream.sub_stream(header.size)?;
        iinf.push(parse_infe(&mut sub_stream)?);
    }
    println!("end of iinf, skiping {} bytes", stream.bytes_left());
    Ok(iinf)
}

fn parse_iref(stream: &mut IStream) -> AvifResult<Vec<ItemReference>> {
    let (version, _flags) = stream.read_version_and_flags()?;
    let mut iref: Vec<ItemReference> = Vec::new();
    // versions > 1 are not supported. ignore them.
    if version <= 1 {
        while stream.has_bytes_left() {
            let header = parse_header(stream)?;
            let from_item_id: u32 = if version == 0 {
                // unsigned int(16) from_item_ID;
                stream.read_u16()? as u32
            } else {
                // unsigned int(32) from_item_ID;
                stream.read_u32()?
            };
            if from_item_id == 0 {
                println!("invalid from_item_id in iref");
                return Err(AvifError::BmffParseFailed);
            }
            // unsigned int(16) reference_count;
            let reference_count = stream.read_u16()?;
            for index in 0..reference_count {
                let to_item_id: u32 = if version == 0 {
                    // unsigned int(16) to_item_ID;
                    stream.read_u16()? as u32
                } else {
                    // unsigned int(32) to_item_ID;
                    stream.read_u32()?
                };
                if to_item_id == 0 {
                    println!("invalid to_item_id in iref");
                    return Err(AvifError::BmffParseFailed);
                }
                iref.push(ItemReference {
                    from_item_id,
                    to_item_id,
                    reference_type: header.box_type.clone(),
                    index: index as u32,
                });
            }
        }
    }
    println!("end of iref, skiping {} bytes", stream.bytes_left());
    Ok(iref)
}

fn parse_idat(stream: &mut IStream) -> AvifResult<Vec<u8>> {
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
    let (_version, _flags) = stream.read_and_enforce_version_and_flags(0)?;
    let mut meta = MetaBox::default();

    // Parse the first hdlr box.
    {
        let header = parse_header(stream)?;
        if header.box_type != "hdlr" {
            println!("first box in meta is not hdlr");
            return Err(AvifError::BmffParseFailed);
        }
        let mut sub_stream = stream.sub_stream(header.size)?;
        parse_hdlr(&mut sub_stream)?;
    }

    let mut boxes_seen = HashSet::from([String::from("hdlr")]);
    while stream.has_bytes_left() {
        let header = parse_header(stream)?;
        match header.box_type.as_str() {
            "hdlr" | "iloc" | "pitm" | "iprp" | "iinf" | "iref" | "idat" => {
                if boxes_seen.contains(&header.box_type) {
                    println!("duplicate {} box in meta.", header.box_type);
                    return Err(AvifError::BmffParseFailed);
                }
                boxes_seen.insert(header.box_type.clone());
            }
            _ => {}
        }
        let mut sub_stream = stream.sub_stream(header.size)?;
        match header.box_type.as_str() {
            "iloc" => meta.iloc = parse_iloc(&mut sub_stream)?,
            "pitm" => meta.primary_item_id = parse_pitm(&mut sub_stream)?,
            "iprp" => meta.iprp = parse_iprp(&mut sub_stream)?,
            "iinf" => meta.iinf = parse_iinf(&mut sub_stream)?,
            "iref" => meta.iref = parse_iref(&mut sub_stream)?,
            "idat" => {
                if !meta.idat.is_empty() {
                    println!("meta contains multiple idat boxes");
                    return Err(AvifError::BmffParseFailed);
                }
                meta.idat = parse_idat(&mut sub_stream)?;
            }
            _ => println!("skipping box {}", header.box_type),
        }
    }
    Ok(meta)
}

fn parse_tkhd(stream: &mut IStream, track: &mut Track) -> AvifResult<()> {
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
    Ok(())
}

fn parse_mdhd(stream: &mut IStream, track: &mut Track) -> AvifResult<()> {
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
    let (_version, _flags) = stream.read_and_enforce_version_and_flags(0)?;
    // unsigned int(32) entry_count;
    let entry_count = usize_from_u32(stream.read_u32()?)?;
    sample_table.chunk_offsets.reserve(entry_count);
    for _ in 0..entry_count {
        let chunk_offset: u64 = if large_offset {
            // unsigned int(64) chunk_offset;
            stream.read_u64()?
        } else {
            // unsigned int(32) chunk_offset;
            stream.read_u32()? as u64
        };
        sample_table.chunk_offsets.push(chunk_offset);
    }
    Ok(())
}

fn parse_stsc(stream: &mut IStream, sample_table: &mut SampleTable) -> AvifResult<()> {
    let (_version, _flags) = stream.read_and_enforce_version_and_flags(0)?;
    // unsigned int(32) entry_count;
    let entry_count = usize_from_u32(stream.read_u32()?)?;
    sample_table.sample_to_chunk.reserve(entry_count);
    for i in 0..entry_count {
        let stsc = SampleToChunk {
            // unsigned int(32) first_chunk;
            first_chunk: stream.read_u32()?,
            // unsigned int(32) samples_per_chunk;
            samples_per_chunk: stream.read_u32()?,
            // unsigned int(32) sample_description_index;
            sample_description_index: stream.read_u32()?,
        };
        if i == 0 {
            if stsc.first_chunk != 1 {
                println!("stsc does not begin with chunk 1.");
                return Err(AvifError::BmffParseFailed);
            }
        } else if stsc.first_chunk <= sample_table.sample_to_chunk.last().unwrap().first_chunk {
            println!("stsc chunks are not strictly increasing.");
            return Err(AvifError::BmffParseFailed);
        }
        sample_table.sample_to_chunk.push(stsc);
    }
    Ok(())
}

fn parse_stsz(stream: &mut IStream, sample_table: &mut SampleTable) -> AvifResult<()> {
    let (_version, _flags) = stream.read_and_enforce_version_and_flags(0)?;
    // unsigned int(32) sample_size;
    let sample_size = stream.read_u32()?;
    // unsigned int(32) sample_count;
    let sample_count = usize_from_u32(stream.read_u32()?)?;

    if sample_size > 0 {
        sample_table.sample_size = SampleSize::FixedSize(sample_size);
        return Ok(());
    }
    let mut sample_sizes: Vec<u32> = Vec::new();
    sample_sizes.reserve(sample_count);
    for _ in 0..sample_count {
        // unsigned int(32) entry_size;
        sample_sizes.push(stream.read_u32()?);
    }
    sample_table.sample_size = SampleSize::Sizes(sample_sizes);
    Ok(())
}

fn parse_stss(stream: &mut IStream, sample_table: &mut SampleTable) -> AvifResult<()> {
    let (_version, _flags) = stream.read_and_enforce_version_and_flags(0)?;
    // unsigned int(32) entry_count;
    let entry_count = usize_from_u32(stream.read_u32()?)?;
    sample_table.sync_samples.reserve(entry_count);
    for _ in 0..entry_count {
        // unsigned int(32) sample_number;
        let sample_number = stream.read_u32()?;
        sample_table.sync_samples.push(sample_number);
    }
    Ok(())
}

fn parse_stts(stream: &mut IStream, sample_table: &mut SampleTable) -> AvifResult<()> {
    let (_version, _flags) = stream.read_and_enforce_version_and_flags(0)?;
    // unsigned int(32) entry_count;
    let entry_count = usize_from_u32(stream.read_u32()?)?;
    sample_table.time_to_sample.reserve(entry_count);
    for _ in 0..entry_count {
        let stts = TimeToSample {
            // unsigned int(32) sample_count;
            sample_count: stream.read_u32()?,
            // unsigned int(32) sample_delta;
            sample_delta: stream.read_u32()?,
        };
        sample_table.time_to_sample.push(stts);
    }
    Ok(())
}

fn parse_stsd(stream: &mut IStream, sample_table: &mut SampleTable) -> AvifResult<()> {
    let (_version, _flags) = stream.read_and_enforce_version_and_flags(0)?;
    // unsigned int(32) entry_count;
    let entry_count = usize_from_u32(stream.read_u32()?)?;
    sample_table.sample_descriptions.reserve(entry_count);
    for _ in 0..entry_count {
        let header = parse_header(stream)?;
        let mut stsd = SampleDescription {
            format: header.box_type.clone(),
            ..SampleDescription::default()
        };

        if stsd.format == "av01" {
            // Skip 78 bytes for visual sample entry size.
            stream.skip(78)?;
            if header.size <= 78 {
                println!("Not enough bytes to parse stsd");
                return Err(AvifError::BmffParseFailed);
            }
            let mut sub_stream = stream.sub_stream(header.size - 78)?;
            stsd.properties = parse_ipco(&mut sub_stream)?;
        }
        sample_table.sample_descriptions.push(stsd);
    }
    Ok(())
}

fn parse_stbl(stream: &mut IStream, track: &mut Track) -> AvifResult<()> {
    if track.sample_table.is_some() {
        println!("duplciate stbl for track.");
        return Err(AvifError::BmffParseFailed);
    }
    let mut sample_table = SampleTable::default();
    while stream.has_bytes_left() {
        let header = parse_header(stream)?;
        let mut sub_stream = stream.sub_stream(header.size)?;
        match header.box_type.as_str() {
            "stco" => parse_stco(&mut sub_stream, &mut sample_table, false)?,
            "co64" => parse_stco(&mut sub_stream, &mut sample_table, true)?,
            "stsc" => parse_stsc(&mut sub_stream, &mut sample_table)?,
            "stsz" => parse_stsz(&mut sub_stream, &mut sample_table)?,
            "stss" => parse_stss(&mut sub_stream, &mut sample_table)?,
            "stts" => parse_stts(&mut sub_stream, &mut sample_table)?,
            "stsd" => parse_stsd(&mut sub_stream, &mut sample_table)?,
            _ => println!("skipping box {}", header.box_type),
        }
    }
    track.sample_table = Some(sample_table);
    Ok(())
}

fn parse_minf(stream: &mut IStream, track: &mut Track) -> AvifResult<()> {
    while stream.has_bytes_left() {
        let header = parse_header(stream)?;
        let mut sub_stream = stream.sub_stream(header.size)?;
        match header.box_type.as_str() {
            "stbl" => parse_stbl(&mut sub_stream, track)?,
            _ => println!("skipping box {}", header.box_type),
        }
    }
    Ok(())
}

fn parse_mdia(stream: &mut IStream, track: &mut Track) -> AvifResult<()> {
    while stream.has_bytes_left() {
        let header = parse_header(stream)?;
        let mut sub_stream = stream.sub_stream(header.size)?;
        match header.box_type.as_str() {
            "mdhd" => parse_mdhd(&mut sub_stream, track)?,
            "minf" => parse_minf(&mut sub_stream, track)?,
            _ => println!("skipping box {}", header.box_type),
        }
    }
    Ok(())
}

fn parse_tref(stream: &mut IStream, track: &mut Track) -> AvifResult<()> {
    while stream.has_bytes_left() {
        let header = parse_header(stream)?;
        let mut sub_stream = stream.sub_stream(header.size)?;
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

fn parse_elst(stream: &mut IStream, track: &mut Track) -> AvifResult<()> {
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

fn parse_edts(stream: &mut IStream, track: &mut Track) -> AvifResult<()> {
    if track.elst_seen {
        // This function always exits with track.elst_seen set to true. So it is sufficient to
        // check track.elst_seen to verify the uniqueness of the edts box.
        println!("multiple edts boxes found for track.");
        return Err(AvifError::BmffParseFailed);
    }
    while stream.has_bytes_left() {
        let header = parse_header(stream)?;
        let mut sub_stream = stream.sub_stream(header.size)?;
        match header.box_type.as_str() {
            "elst" => parse_elst(&mut sub_stream, track)?,
            _ => println!("skipping box {}", header.box_type),
        }
    }
    if !track.elst_seen {
        println!("elst box was not found in edts");
        return Err(AvifError::BmffParseFailed);
    }
    Ok(())
}

fn parse_trak(stream: &mut IStream) -> AvifResult<Track> {
    let mut track = Track::default();
    println!("parsing trak size: {}", stream.bytes_left());
    while stream.has_bytes_left() {
        let header = parse_header(stream)?;
        let mut sub_stream = stream.sub_stream(header.size)?;
        match header.box_type.as_str() {
            "tkhd" => parse_tkhd(&mut sub_stream, &mut track)?,
            "mdia" => parse_mdia(&mut sub_stream, &mut track)?,
            "tref" => parse_tref(&mut sub_stream, &mut track)?,
            "edts" => parse_edts(&mut sub_stream, &mut track)?,
            "meta" => track.meta = Some(parse_meta(&mut sub_stream)?),
            _ => println!("skipping box {}", header.box_type),
        }
    }
    println!("track: {:#?}", track);
    Ok(track)
}

fn parse_moov(stream: &mut IStream) -> AvifResult<Vec<Track>> {
    println!("parsing moov size: {}", stream.bytes_left());
    let mut tracks: Vec<Track> = Vec::new();
    while stream.has_bytes_left() {
        let header = parse_header(stream)?;
        let mut sub_stream = stream.sub_stream(header.size)?;
        match header.box_type.as_str() {
            "trak" => tracks.push(parse_trak(&mut sub_stream)?),
            _ => println!("skipping box {}", header.box_type),
        }
    }
    Ok(tracks)
}

pub fn parse(io: &mut GenericIO) -> AvifResult<AvifBoxes> {
    let mut ftyp: Option<FileTypeBox> = None;
    let mut meta: Option<MetaBox> = None;
    let mut tracks: Option<Vec<Track>> = None;
    let mut parse_offset: u64 = 0;
    loop {
        // Read just enough to get the next box header (32 bytes).
        let header_data = io.read(parse_offset, 32)?;
        if header_data.is_empty() {
            // No error and size is 0. We have reached the end of the stream.
            break;
        }
        let mut header_stream = IStream::create(header_data);
        let header = parse_header(&mut header_stream)?;
        println!("{:#?}", header);
        parse_offset = parse_offset
            .checked_add(header_stream.offset as u64)
            .ok_or(AvifError::BmffParseFailed)?;

        // Read the rest of the box if necessary.
        match header.box_type.as_str() {
            "ftyp" | "meta" | "moov" => {
                let box_data = io.read(parse_offset, header.size)?;
                if box_data.len() != header.size {
                    return Err(AvifError::TruncatedData);
                }
                let mut box_stream = IStream::create(box_data);
                match header.box_type.as_str() {
                    "ftyp" => {
                        ftyp = Some(parse_ftyp(&mut box_stream)?);
                        if !ftyp.as_ref().unwrap().is_avif() {
                            return Err(AvifError::InvalidFtyp);
                        }
                    }
                    "meta" => meta = Some(parse_meta(&mut box_stream)?),
                    "moov" => {
                        tracks = Some(parse_moov(&mut box_stream)?);
                        // decoder.image_sequence_track_present = true;
                    }
                    _ => {} // Not reached.
                }
                if ftyp.is_some() {
                    let ftyp = ftyp.as_ref().unwrap();
                    if (!ftyp.needs_meta() || meta.is_some())
                        && (!ftyp.needs_moov() || tracks.is_some())
                    {
                        // Enough information has been parsed to consider parse a success.
                        break;
                    }
                }
            }
            _ => {
                println!("skipping box: {}", header.box_type);
            }
        }
        parse_offset = parse_offset
            .checked_add(header.size as u64)
            .ok_or(AvifError::BmffParseFailed)?;
    }
    if ftyp.is_none() {
        return Err(AvifError::InvalidFtyp);
    }
    let ftyp = ftyp.unwrap();
    if (ftyp.needs_meta() && meta.is_none()) || (ftyp.needs_moov() && tracks.is_none()) {
        return Err(AvifError::TruncatedData);
    }
    Ok(AvifBoxes {
        ftyp,
        meta: meta.unwrap_or_default(),
        tracks: tracks.unwrap_or_default(),
    })
}

pub fn peek_compatible_file_type(data: &[u8]) -> AvifResult<bool> {
    let mut stream = IStream::create(data);
    let header = parse_header(&mut stream)?;
    if header.box_type != "ftyp" {
        return Ok(false);
    }
    let ftyp = parse_ftyp(&mut stream)?;
    //println!("ftyp: {:#?}", ftyp);
    Ok(ftyp.is_avif())
}

pub fn parse_tmap(stream: &mut IStream) -> AvifResult<GainMapMetadata> {
    // unsigned int(8) version = 0;
    let version = stream.read_u8()?;
    if version != 0 {
        println!("unsupported version in tmap box");
        return Err(AvifError::InvalidToneMappedImage);
    }
    // unsigned int(8) flags;
    let flags = stream.read_u8()?;
    let channel_count: usize = ((flags & 1) * 2 + 1).into();
    let mut metadata = GainMapMetadata {
        use_base_color_space: (flags & 2) != 0,
        backward_direction: (flags & 4) != 0,
        ..GainMapMetadata::default()
    };
    let use_common_denominator = (flags & 8) != 0;
    println!("channel_count: {channel_count}");
    println!("use_common_denominator: {use_common_denominator}");
    if use_common_denominator {
        let common_denominator = stream.read_u32()?;
        metadata.base_hdr_headroom = (stream.read_u32()?, common_denominator);
        metadata.alternate_hdr_headroom = (stream.read_u32()?, common_denominator);
        for i in 0..channel_count {
            metadata.min[i] = (stream.read_i32()?, common_denominator);
            metadata.max[i] = (stream.read_i32()?, common_denominator);
            metadata.gamma[i] = (stream.read_u32()?, common_denominator);
            metadata.base_offset[i] = (stream.read_i32()?, common_denominator);
            metadata.alternate_offset[i] = (stream.read_i32()?, common_denominator);
        }
    } else {
        metadata.base_hdr_headroom = stream.read_ufraction()?;
        //println!("here1 offset: {}", stream.offset);
        metadata.alternate_hdr_headroom = stream.read_ufraction()?;
        //println!("here2 offset: {}", stream.offset);
        for i in 0..channel_count {
            metadata.min[i] = stream.read_fraction()?;
            //println!("here3 offset: {}", stream.offset);
            metadata.max[i] = stream.read_fraction()?;
            //println!("here4 offset: {}", stream.offset);
            metadata.gamma[i] = stream.read_ufraction()?;
            //println!("here5 offset: {}", stream.offset);
            metadata.base_offset[i] = stream.read_fraction()?;
            //println!("here6 offset: {}", stream.offset);
            metadata.alternate_offset[i] = stream.read_fraction()?;
            //println!("here7 offset: {}", stream.offset);
        }
    }
    // Fill the remaining values by copying those from the first channel.
    for i in channel_count..3 {
        metadata.min[i] = metadata.min[0];
        metadata.max[i] = metadata.max[0];
        metadata.gamma[i] = metadata.gamma[0];
        metadata.base_offset[i] = metadata.base_offset[0];
        metadata.alternate_offset[i] = metadata.alternate_offset[0];
    }
    if stream.has_bytes_left() {
        println!("invalid trailing bytes in tmap box");
        return Err(AvifError::InvalidToneMappedImage);
    }
    Ok(metadata)
}
