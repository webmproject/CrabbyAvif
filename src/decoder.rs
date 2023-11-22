use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::ops::Range;

use crate::dav1d::*;
use crate::io::*;
use crate::mp4box::ItemProperty::AuxiliaryType;
use crate::mp4box::ItemProperty::ImageSpatialExtents;
use crate::mp4box::*;
use crate::stream::*;
use crate::*;

// TODO: needed only for debug to AvifImage. Can be removed it AvifIMage does not have to be debug printable.
use derivative::Derivative;

#[derive(Default, Debug)]
pub struct AvifImageInfo {
    pub width: u32,
    pub height: u32,
    pub depth: u8,

    pub yuv_format: PixelFormat,
    pub full_range: bool,
    pub chroma_sample_position: u8,

    pub alpha_present: bool,
    pub alpha_premultiplied: bool,

    pub icc: u8, //Option<Vec<u8>>,

    pub color_primaries: u16,
    pub transfer_characteristics: u16,
    pub matrix_coefficients: u16,
}

#[derive(Derivative, Default)]
#[derivative(Debug)]
pub struct AvifImage {
    pub info: AvifImageInfo,

    pub yuv_planes: [Option<*const u8>; 3],
    pub yuv_row_bytes: [u32; 3], // TODO: named constant
    pub image_owns_yuv_planes: bool,

    pub alpha_plane: Option<*const u8>,
    pub alpha_row_bytes: u32,
    pub image_owns_alpha_plane: bool,

    // some more boxes. clli, transformations. pasp, clap, irot, imir.

    // exif, xmp.

    // gainmap.
    #[derivative(Debug = "ignore")]
    plane_buffers: [Vec<u8>; 4],
}

#[derive(Debug)]
pub struct AvifPlane {
    pub data: *const u8,
    pub width: u32,
    pub height: u32,
    pub row_bytes: u32,
    pub pixel_size: u32,
}

impl AvifImage {
    pub fn plane(&self, plane: usize) -> Option<AvifPlane> {
        assert!(plane < 4);
        let pixel_size = if self.info.depth == 8 { 1 } else { 2 };
        if plane < 3 {
            if self.yuv_planes[plane].is_none() {
                return None;
            }
            let mut plane_width = self.info.width;
            let mut plane_height = self.info.height;
            if plane > 0 {
                if self.info.yuv_format == PixelFormat::Yuv420 {
                    plane_width = (plane_width + 1) / 2;
                    plane_height = (plane_height + 1) / 2;
                } else if self.info.yuv_format == PixelFormat::Yuv422 {
                    plane_width = (plane_width + 1) / 2;
                }
            }
            let stride_index: usize = if plane == 0 { 0 } else { 1 };
            return Some(AvifPlane {
                data: self.yuv_planes[plane].unwrap(),
                width: plane_width,
                height: plane_height,
                row_bytes: self.yuv_row_bytes[plane],
                pixel_size,
            });
        }
        if self.alpha_plane.is_none() {
            return None;
        }
        return Some(AvifPlane {
            data: self.alpha_plane.unwrap(),
            width: self.info.width,
            height: self.info.height,
            row_bytes: self.alpha_row_bytes,
            pixel_size,
        });
    }

    fn allocate_planes(&mut self, category: usize) -> AvifResult<()> {
        // TODO : assumes 444. do other stuff.
        let pixel_size: u32 = if self.info.depth == 8 { 1 } else { 2 };
        let plane_size = (self.info.width * self.info.height * pixel_size) as usize;
        if category == 0 {
            for plane_index in 0usize..3 {
                self.plane_buffers[plane_index].reserve(plane_size);
                self.plane_buffers[plane_index].resize(plane_size, 0);
                self.yuv_row_bytes[plane_index] = self.info.width * pixel_size;
                self.yuv_planes[plane_index] = Some(self.plane_buffers[plane_index].as_ptr());
            }
            self.image_owns_yuv_planes = true;
        } else if category == 1 {
            self.plane_buffers[3].reserve(plane_size);
            self.plane_buffers[3].resize(plane_size, 255);
            self.alpha_row_bytes = self.info.width * pixel_size;
            self.alpha_plane = Some(self.plane_buffers[3].as_ptr());
            self.image_owns_alpha_plane = true;
        } else {
            println!("unknown category {category}. cannot allocate.");
            return Err(AvifError::UnknownError);
        }
        Ok(())
    }

    fn copy_from_tile(
        &mut self,
        tile: &AvifImage,
        tile_info: &AvifTileInfo,
        tile_index: u32,
        category: usize,
    ) {
        let row_index: usize = (tile_index / tile_info.grid.columns) as usize;
        let column_index: usize = (tile_index % tile_info.grid.columns) as usize;
        println!("copying tile {tile_index} {row_index} {column_index}");

        let plane_range = if category == 1 { 3usize..4 } else { 0usize..3 };
        // TODO: what about gainmap category?

        for plane_index in plane_range {
            println!("plane_index {plane_index}");
            let src_plane = tile.plane(plane_index);
            if src_plane.is_none() {
                continue;
            }
            let src_plane = src_plane.unwrap();
            let src_width_to_copy;
            // If this is the last tile column, clamp to left over width.
            if (column_index as u32) == tile_info.grid.columns - 1 {
                let width_so_far = src_plane.width * (column_index as u32);
                // TODO: does self.width need to be accounted for subsampling?
                src_width_to_copy = self.info.width - width_so_far;
            } else {
                src_width_to_copy = src_plane.width;
            }
            let src_byte_count: usize = (src_width_to_copy * src_plane.pixel_size)
                .try_into()
                .unwrap();
            let dst_row_bytes = if plane_index < 3 {
                self.yuv_row_bytes[plane_index]
            } else {
                self.alpha_row_bytes
            };

            let mut dst_base_offset: usize = 0;
            dst_base_offset += row_index * ((src_plane.height * dst_row_bytes) as usize);
            dst_base_offset += column_index * ((src_plane.width * src_plane.pixel_size) as usize);
            //println!("dst base_offset: {dst_base_offset}");

            let src_height_to_copy;
            // If this is the last tile row, clamp to left over height.
            if (row_index as u32) == tile_info.grid.rows - 1 {
                let height_so_far = src_plane.height * (row_index as u32);
                // TODO: does self.height need to be accounted for subsampling?
                src_height_to_copy = self.info.height - height_so_far;
            } else {
                src_height_to_copy = src_plane.height;
            }

            for y in 0..src_height_to_copy {
                let src_stride_offset: isize = (y * src_plane.row_bytes).try_into().unwrap();
                let ptr = unsafe { src_plane.data.offset(src_stride_offset) };
                let pixels = unsafe { std::slice::from_raw_parts(ptr, src_byte_count) };
                let dst_stride_offset: usize = dst_base_offset + ((y * dst_row_bytes) as usize);
                let dst_end_offset: usize = dst_stride_offset + src_byte_count;

                let mut dst_slice =
                    &mut self.plane_buffers[plane_index][dst_stride_offset..dst_end_offset];
                if y == 0 {
                    println!(
                        "src slice len: {} dst_slice_len: {}",
                        pixels.len(),
                        dst_slice.len()
                    );
                }
                dst_slice.copy_from_slice(pixels);
            }
        }
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub enum AvifDecoderSource {
    Tracks,
    PrimaryItem,
    #[default]
    Auto,
    // TODO: Thumbnail,
}

#[derive(Debug, Default)]
pub struct AvifDecoderSettings {
    pub source: AvifDecoderSource,
    pub ignore_exif: bool,
    pub ignore_icc: bool,
    pub strictness: AvifStrictness,
}

impl AvifStrictness {
    pub fn alpha_ispe_required(&self) -> bool {
        match self {
            AvifStrictness::All => true,
            AvifStrictness::SpecificInclude(flags) => flags
                .iter()
                .find(|x| matches!(x, AvifStrictnessFlag::AlphaIspeRequired))
                .is_some(),
            AvifStrictness::SpecificExclude(flags) => flags
                .iter()
                .find(|x| matches!(x, AvifStrictnessFlag::AlphaIspeRequired))
                .is_none(),
            _ => false,
        }
    }
}

#[derive(Default)]
pub struct AvifDecoder {
    pub settings: AvifDecoderSettings,
    image: AvifImage,
    codec: Dav1d,
    source: AvifDecoderSource,
    tile_info: [AvifTileInfo; 3],
    tiles: [Vec<AvifTile>; 3],
    image_index: i32,
    pub image_count: u32,
    pub timescale: u32,
    pub duration_in_timescales: u64,
    pub duration: f64,
    pub repetition_count: i32,
    avif_items: HashMap<u32, AvifItem>,
    tracks: Vec<AvifTrack>,
    // To replicate the C-API, we need to keep this optional. Otherwise this
    // could be part of the initialization.
    io: Option<Box<dyn AvifDecoderIO>>,
    codecs: Vec<Dav1d>,
}

#[derive(Debug, Default)]
struct AvifGrid {
    rows: u32,
    columns: u32,
    width: u32,
    height: u32,
}

#[derive(Debug, Default)]
struct AvifTileInfo {
    tile_count: u32,
    decoded_tile_count: u32,
    grid: AvifGrid,
}

#[derive(Debug, Default)]
struct AvifItem {
    id: u32,
    item_type: String,
    size: usize,
    width: u32,
    height: u32,
    content_type: String,
    properties: Vec<ItemProperty>,
    extents: Vec<ItemLocationExtent>,
    // TODO mergedExtents stuff.
    thumbnail_for_id: u32,
    aux_for_id: u32,
    desc_for_id: u32,
    dimg_for_id: u32,
    dimg_index: u32,
    prem_by_id: u32,
    has_unsupported_essential_property: bool,
    ipma_seen: bool,
    progressive: bool,
    idat: Vec<u8>,
}

macro_rules! find_property {
    ($self:ident, $a:ident) => {
        $self
            .properties
            .iter()
            .find(|x| matches!(x, ItemProperty::$a(_)))
    };
}

macro_rules! find_properties {
    ($self:ident, $a:ident) => {
        $self
            .properties
            .iter()
            .filter(|x| matches!(x, ItemProperty::$a(_)))
            .collect()
    };
}

impl AvifItem {
    fn data_offset(&self) -> u64 {
        self.extents[0].offset as u64
    }

    fn read_and_parse(
        &self,
        io: &mut Box<dyn AvifDecoderIO>,
        grid: &mut AvifGrid,
    ) -> AvifResult<()> {
        // TODO: this function also has to extract codec type.
        if self.item_type != "grid" {
            return Ok(());
        }
        // TODO: handle multiple extents.
        let mut io_data = match self.idat.is_empty() {
            true => io.read(self.data_offset(), self.size)?,
            false => {
                // TODO: assumes idat offset is 0.
                self.idat.as_slice()
            }
        };
        let mut stream = IStream::create(io_data);
        // unsigned int(8) version = 0;
        let version = stream.read_u8()?;
        if version != 0 {
            println!("unsupported version for grid");
            return Err(AvifError::InvalidImageGrid);
        }
        // unsigned int(8) flags;
        let flags = stream.read_u8()?;
        // unsigned int(8) rows_minus_one;
        grid.rows = stream.read_u8()? as u32;
        grid.rows += 1;
        // unsigned int(8) columns_minus_one;
        grid.columns = stream.read_u8()? as u32;
        grid.columns += 1;
        if (flags & 1) == 1 {
            // unsigned int(32) output_width;
            grid.width = stream.read_u32()?;
            // unsigned int(32) output_height;
            grid.height = stream.read_u32()?;
        } else {
            // unsigned int(16) output_width;
            grid.width = stream.read_u16()? as u32;
            // unsigned int(16) output_height;
            grid.height = stream.read_u16()? as u32;
        }
        if grid.width == 0 || grid.height == 0 {
            println!("invalid dimensions in grid box");
            return Err(AvifError::InvalidImageGrid);
        }
        println!("grid: {:#?}", grid);
        // TODO: check for too large of a grid.
        Ok(())
    }

    fn operating_point(&self) -> u8 {
        match find_property!(self, OperatingPointSelector) {
            Some(a1op) => match a1op {
                ItemProperty::OperatingPointSelector(operating_point) => *operating_point,
                _ => 0, // not reached.
            },
            None => 0, // default operating point.
        }
    }

    fn harvest_ispe(&mut self, alpha_ispe_required: bool) -> AvifResult<()> {
        if self.size == 0 {
            return Ok(());
        }
        if self.has_unsupported_essential_property {
            // An essential property isn't supported by libavif. Ignore.
            return Ok(());
        }

        let is_grid = self.item_type == "grid";
        if self.item_type != "av01" && !is_grid {
            // probably exif or some other data.
            return Ok(());
        }
        match find_property!(self, ImageSpatialExtents) {
            Some(property) => match property {
                ItemProperty::ImageSpatialExtents(x) => {
                    self.width = x.width;
                    self.height = x.height;
                    if self.width == 0 || self.height == 0 {
                        println!("item id has invalid size.");
                        return Err(AvifError::BmffParseFailed);
                    }
                }
                _ => return Err(AvifError::UnknownError), // not reached.
            },
            None => {
                // No ispe was found.
                if self.is_auxiliary_alpha() {
                    if alpha_ispe_required {
                        println!("alpha auxiliary image is missing mandatory ispe");
                        return Err(AvifError::BmffParseFailed);
                    }
                } else {
                    println!("item id is missing mandatory ispe property");
                    return Err(AvifError::BmffParseFailed);
                }
            }
        }
        Ok(())
    }

    fn av1C(&self) -> Option<&CodecConfiguration> {
        match find_property!(self, CodecConfiguration) {
            Some(property) => match property {
                ItemProperty::CodecConfiguration(av1C) => Some(&av1C),
                _ => None, // not reached.
            },
            None => None,
        }
    }

    fn a1lx(&self) -> Option<&[usize; 3]> {
        match find_property!(self, AV1LayeredImageIndexing) {
            Some(property) => match property {
                ItemProperty::AV1LayeredImageIndexing(a1lx) => Some(&a1lx),
                _ => None, // not reached.
            },
            None => None,
        }
    }

    fn lsel(&self) -> Option<u16> {
        match find_property!(self, LayerSelector) {
            Some(property) => match property {
                ItemProperty::LayerSelector(lsel) => Some(*lsel),
                _ => None, // not reached.
            },
            None => None,
        }
    }

    fn is_auxiliary_alpha(&self) -> bool {
        match find_property!(self, AuxiliaryType) {
            Some(auxC) => match auxC {
                ItemProperty::AuxiliaryType(aux_type) => {
                    aux_type == "urn:mpeg:mpegB:cicp:systems:auxiliary:alpha"
                        || aux_type == "urn:mpeg:hevc:2015:auxid:1"
                }
                _ => false, // not reached.
            },
            None => false,
        }
    }

    fn should_skip(&self) -> bool {
        self.size == 0
            || self.has_unsupported_essential_property
            || (self.item_type != "av01" && self.item_type != "grid")
            || self.thumbnail_for_id != 0
    }
}

fn find_nclx(properties: &Vec<ItemProperty>) -> Result<&Nclx, bool> {
    let nclx_properties: Vec<_> = properties
        .iter()
        .filter(|x| match x {
            ItemProperty::ColorInformation(colr) => match colr {
                ColorInformation::Nclx(_) => true,
                _ => false,
            },
            _ => false,
        })
        .collect();
    match nclx_properties.len() {
        0 => Err(false),
        1 => match nclx_properties[0] {
            ItemProperty::ColorInformation(colr) => match colr {
                ColorInformation::Nclx(nclx) => Ok(&nclx),
                _ => Err(false), // not reached.
            },
            _ => Err(false), // not reached.
        },
        _ => Err(true), // multiple nclx were found.
    }
}

fn find_icc(properties: &Vec<ItemProperty>) -> Result<&Icc, bool> {
    let icc_properties: Vec<_> = properties
        .iter()
        .filter(|x| match x {
            ItemProperty::ColorInformation(colr) => match colr {
                ColorInformation::Icc(_) => true,
                _ => false,
            },
            _ => false,
        })
        .collect();
    match icc_properties.len() {
        0 => Err(false),
        1 => match icc_properties[0] {
            ItemProperty::ColorInformation(colr) => match colr {
                ColorInformation::Icc(icc) => Ok(&icc),
                _ => Err(false), // not reached.
            },
            _ => Err(false), // not reached.
        },
        _ => Err(true), // multiple icc were found.
    }
}

fn find_av1C(properties: &Vec<ItemProperty>) -> Option<&CodecConfiguration> {
    match properties
        .iter()
        .find(|x| matches!(x, ItemProperty::CodecConfiguration(_)))
    {
        Some(property) => match property {
            ItemProperty::CodecConfiguration(av1C) => Some(&av1C),
            _ => None, // not reached.
        },
        None => None,
    }
}

fn read_file(filename: &String) -> Vec<u8> {
    let mut file = File::open(filename).expect("file not found");
    let mut data: Vec<u8> = Vec::new();
    let _ = file.read_to_end(&mut data);
    data
}

// This design is not final. It's possible to do this in the same loop where boxes are parsed. But it
// seems a little cleaner to do this after the fact.
fn construct_avif_items(meta: &MetaBox) -> AvifResult<HashMap<u32, AvifItem>> {
    let mut avif_items: HashMap<u32, AvifItem> = HashMap::new();
    for item in &meta.iinf {
        let mut avif_item: AvifItem = Default::default();
        avif_item.id = item.item_id;
        avif_item.item_type = item.item_type.clone();
        avif_item.content_type = item.content_type.clone();
        avif_items.insert(avif_item.id, avif_item);
    }
    for item in &meta.iloc.items {
        // TODO: Make sure item id exists before unwrapping.
        let avif_item = avif_items.get_mut(&item.item_id).unwrap();
        if !avif_item.extents.is_empty() {
            println!("item already has extents.");
            return Err(AvifError::BmffParseFailed);
        }
        if item.construction_method == 1 {
            avif_item.idat = meta.idat.clone();
        }
        // TODO: handle overflows in the addition below.
        for extent in &item.extents {
            avif_item.extents.push(ItemLocationExtent {
                offset: item.base_offset + extent.offset,
                length: extent.length,
            });
            avif_item.size += extent.length as usize;
        }
    }
    for association in &meta.iprp.associations {
        // TODO: Make sure item id exists before unwrapping.
        let avif_item = avif_items.get_mut(&association.item_id).unwrap();
        if avif_item.ipma_seen {
            // TODO: ipma_seen can be a local hashmap or set here instea of being in the
            // struct as it is only used for this validation.
            println!("item has duplictate ipma.");
            return Err(AvifError::BmffParseFailed);
        }
        avif_item.ipma_seen = true;
        for (property_index_ref, essential_ref) in &association.associations {
            let property_index: usize = *property_index_ref as usize;
            let essential = *essential_ref;
            if property_index == 0 {
                // Not associated with any item.
                continue;
            }
            if property_index > meta.iprp.properties.len() {
                println!(
                    "property index: {} len: {}",
                    property_index,
                    meta.iprp.properties.len()
                );
                println!("invalid property_index in ipma.");
                return Err(AvifError::BmffParseFailed);
            }
            // property_index is 1-indexed.
            let property = meta.iprp.properties[property_index - 1].clone();
            // TODO: Add more boxes here once they are supported.
            let is_supported_property = match property {
                ItemProperty::ImageSpatialExtents(_)
                | ItemProperty::ColorInformation(_)
                | ItemProperty::CodecConfiguration(_)
                | ItemProperty::PixelInformation(_)
                | ItemProperty::PixelAspectRatio(_)
                | ItemProperty::AuxiliaryType(_)
                | ItemProperty::ClearAperture(_)
                | ItemProperty::ImageRotation(_)
                | ItemProperty::ImageMirror(_)
                | ItemProperty::OperatingPointSelector(_)
                | ItemProperty::LayerSelector(_)
                | ItemProperty::AV1LayeredImageIndexing(_)
                | ItemProperty::ContentLightLevelInformation(_) => true,
                _ => false,
            };
            if is_supported_property {
                if essential {
                    // a1lx is not allowed to be marked as essential.
                    // TODO: enforce that.
                } else {
                    // a1op and lsel must be marked as essential.
                    // TODO: enforce that.
                }
                avif_item.properties.push(property);
            } else {
                if essential {
                    avif_item.has_unsupported_essential_property = true;
                }
            }
        }
    }
    for (reference_index, reference) in meta.iref.iter().enumerate() {
        let item = avif_items
            .get_mut(&reference.from_item_id)
            .ok_or(AvifError::BmffParseFailed)?;
        match reference.reference_type.as_str() {
            "thmb" => item.thumbnail_for_id = reference.to_item_id,
            "auxl" => item.aux_for_id = reference.to_item_id,
            "cdsc" => item.desc_for_id = reference.to_item_id,
            "prem" => item.prem_by_id = reference.to_item_id,
            "dimg" => {
                // derived images refer in the opposite direction.
                let dimg_item = avif_items
                    .get_mut(&reference.to_item_id)
                    .ok_or(AvifError::BmffParseFailed)?;
                dimg_item.dimg_for_id = reference.from_item_id;
                dimg_item.dimg_index = reference_index as u32;
            }
            _ => {
                // unknown reference type, ignore.
            }
        }
    }
    Ok(avif_items)
}

fn find_alpha_item(avif_items: &HashMap<u32, AvifItem>, color_item: &AvifItem) -> Option<u32> {
    match avif_items
        .iter()
        .find(|x| !x.1.should_skip() && x.1.aux_for_id == color_item.id && x.1.is_auxiliary_alpha())
    {
        Some(item) => return Some(*item.0),
        None => {} // Do nothing.
    };
    if color_item.item_type != "grid" {
        return Some(0);
    }
    // TODO: If color item is a grid, check if there is an alpha channel which is represented as an auxl item to each color tile item.
    Some(0)
}

#[derive(Debug, Default)]
struct AvifDecodeSample {
    // owns_data
    // partial_data
    item_id: u32,
    offset: u64,
    size: usize,
    spatial_id: u8,
    sync: bool,
    // TODO: these two can be some enum?
    data_buffer: Option<Vec<u8>>,
}

impl AvifDecodeSample {
    pub fn data<'a>(&'a self, io: &'a mut Box<impl AvifDecoderIO + ?Sized>) -> AvifResult<&[u8]> {
        match &self.data_buffer {
            Some(data_buffer) => Ok(&data_buffer),
            None => io.read(self.offset, self.size),
        }
    }
}

#[derive(Debug, Default)]
struct AvifDecodeInput {
    samples: Vec<AvifDecodeSample>,
    all_layers: bool,
    category: u8,
}

#[derive(Debug, Default)]
struct AvifTile {
    width: u32,
    height: u32,
    operating_point: u8,
    image: AvifImage,
    input: AvifDecodeInput,
    codec_index: usize,
}

fn create_tile(item: &AvifItem) -> AvifResult<AvifTile> {
    let mut tile = AvifTile::default();
    tile.width = item.width;
    tile.height = item.height;
    tile.operating_point = item.operating_point();
    tile.image = AvifImage::default();
    // TODO: do all the layer stuff (a1op and lsel) in avifCodecDecodeInputFillFromDecoderItem.
    let mut layer_sizes: [usize; 4] = [0; 4];
    let mut layer_count = 0;
    let a1lx = item.a1lx();
    if a1lx.is_some() {
        let a1lx = a1lx.unwrap();
        println!("item size: {} a1lx: {:#?}", item.size, a1lx);
        let mut remaining_size: usize = item.size;
        for i in 0usize..3 {
            layer_count += 1;
            if a1lx[i] > 0 {
                // >= instead of > because there must be room for the last layer
                if a1lx[i] >= remaining_size {
                    println!("a1lx layer index [{i}] does not fit in item size");
                    return Err(AvifError::BmffParseFailed);
                }
                layer_sizes[i] = a1lx[i];
                remaining_size -= a1lx[i];
            } else {
                layer_sizes[i] = remaining_size;
                remaining_size = 0;
                break;
            }
        }
        if remaining_size > 0 {
            assert!(layer_count == 3);
            layer_count += 1;
            layer_sizes[3] = remaining_size;
        }
        println!("layer count: {layer_count} layer_sizes: {:#?}", layer_sizes);
    }
    let lsel = item.lsel();
    // TODO: account for progressive (avifCodecDecodeInputFillFromDecoderItem).
    if lsel.is_some() && lsel.unwrap() != 0xFFFF {
        // Layer selection. This requires that the underlying AV1 codec decodes all layers,
        // and then only returns the requested layer as a single frame. To the user of libavif,
        // this appears to be a single frame.
        tile.input.all_layers = true;
        let mut sample_size: usize = 0;
        let layer_id = lsel.unwrap();
        if layer_count > 0 {
            // TODO: test this with a case?
            panic!("im here");
            // Optimization: If we're selecting a layer that doesn't require
            // the entire image's payload (hinted via the a1lx box).
            if layer_id >= layer_count {
                println!("lsel layer index not found in a1lx.");
                return Err(AvifError::InvalidImageGrid);
            }
            let layer_id_plus_1: usize = (layer_id + 1) as usize;
            for i in 0usize..layer_id_plus_1 {
                sample_size += layer_sizes[i];
            }
        } else {
            // This layer payload subsection is not known. Use the whole payload.
            sample_size = item.size;
        }
        let sample = AvifDecodeSample {
            item_id: item.id,
            offset: 0,
            size: sample_size,
            spatial_id: lsel.unwrap() as u8,
            sync: true,
            data_buffer: None,
        };
        tile.input.samples.push(sample);
    } else if (false) {
        // TODO: case for progressive and allow progressive.
    } else {
        // Typical case: Use the entire item's payload for a single frame output
        let sample = AvifDecodeSample {
            item_id: item.id,
            offset: 0,
            size: item.size,
            // Legal spatial_id values are [0,1,2,3], so this serves as a sentinel
            // value for "do not filter by spatial_id"
            spatial_id: 0xff,
            sync: true,
            data_buffer: None,
        };
        tile.input.samples.push(sample);
    }
    Ok(tile)
}

fn create_tile_from_track(track: &AvifTrack) -> AvifResult<AvifTile> {
    let mut tile = AvifTile::default();
    tile.width = track.width;
    tile.height = track.height;
    tile.operating_point = 0; // No way to set operating point via tracks

    // TODO: implement the imagecount check in avifCodecDecodeInputFillFromSampleTable.

    let mut sample_size_index = 0;
    let sample_table = &track.sample_table.as_ref().unwrap();
    for (chunk_index, chunk_offset) in sample_table.chunk_offsets.iter().enumerate() {
        // Figure out how many samples are in this chunk.
        let sample_count = sample_table.get_sample_count_of_chunk(chunk_index);
        if sample_count == 0 {
            println!("chunk with 0 samples found");
            return Err(AvifError::BmffParseFailed);
        }

        let mut sample_offset = *chunk_offset;
        for sample_index in 0..sample_count {
            let mut sample_size = sample_table.all_samples_size;
            if sample_size == 0 {
                if sample_size_index >= sample_table.sample_sizes.len() {
                    println!("not enough sampel sizes in the table");
                    return Err(AvifError::BmffParseFailed);
                }
                sample_size = sample_table.sample_sizes[sample_size_index];
            }
            let sample = AvifDecodeSample {
                item_id: 0,
                offset: sample_offset,
                size: sample_size as usize,
                // Legal spatial_id values are [0,1,2,3], so this serves as a sentinel
                // value for "do not filter by spatial_id"
                spatial_id: 0xff,
                // Assume first sample is always sync (in case stss box was missing).
                sync: tile.input.samples.is_empty(),
                data_buffer: None,
            };
            tile.input.samples.push(sample);
            // TODO: verify if sample size math can be done here.
            sample_offset += sample_size as u64;
            sample_size_index += 1;
        }
    }
    for sync_sample_number in &sample_table.sync_samples {
        let index: usize = (*sync_sample_number - 1) as usize; // sample_table.sync_samples is 1-based.
        if index < tile.input.samples.len() {
            tile.input.samples[index].sync = true;
        }
    }
    Ok(tile)
}

fn generate_tiles(
    avif_items: &mut HashMap<u32, AvifItem>,
    iinf: &Vec<ItemInfo>,
    item_id: u32,
    info: &AvifTileInfo,
    category: usize,
) -> AvifResult<Vec<AvifTile>> {
    let mut tiles: Vec<AvifTile> = Vec::new();
    if info.grid.rows > 0 && info.grid.columns > 0 {
        println!("grid###: {:#?}", info.grid);
        let mut grid_item_ids: Vec<u32> = Vec::new();
        let mut first_av1C: CodecConfiguration = Default::default();
        // Collect all the dimg items.
        // Cannot directly iterate through avif_items here directly because HashMap is not ordered.
        for item_info in iinf {
            let dimg_item = avif_items
                .get(&item_info.item_id)
                .ok_or(AvifError::InvalidImageGrid)?;
            if dimg_item.dimg_for_id != item_id {
                continue;
            }
            if dimg_item.item_type != "av01" {
                println!("invalid item_type in dimg grid");
                return Err(AvifError::InvalidImageGrid);
            }
            if dimg_item.has_unsupported_essential_property {
                println!(
                    "Grid image contains tile with an unsupported property marked as essential"
                );
                return Err(AvifError::InvalidImageGrid);
            }
            let mut tile = create_tile(dimg_item)?;
            tile.input.category = category as u8;
            tiles.push(tile);

            if tiles.len() == 1 {
                // Adopt the configuration property of the first tile.
                first_av1C = dimg_item.av1C().ok_or(AvifError::BmffParseFailed)?.clone();
            }
            grid_item_ids.push(item_info.item_id);
        }
        // TODO: check if there are enough grids.
        avif_items
            .get_mut(&item_id)
            .ok_or(AvifError::InvalidImageGrid)?
            .properties
            .push(ItemProperty::CodecConfiguration(first_av1C));
        println!("grid item ids: {:#?}", grid_item_ids);
    } else {
        let item = avif_items
            .get(&item_id)
            .ok_or(AvifError::MissingImageItem)?;
        if item.size == 0 {
            return Err(AvifError::MissingImageItem);
        }
        let mut tile = create_tile(item)?;
        tile.input.category = category as u8;
        tiles.push(tile);
    }
    Ok(tiles)
}

fn steal_planes(dst: &mut AvifImage, src: &mut AvifImage, category: usize) {
    match category {
        0 => {
            dst.yuv_planes[0] = src.yuv_planes[0];
            dst.yuv_planes[1] = src.yuv_planes[1];
            dst.yuv_planes[2] = src.yuv_planes[2];
            dst.yuv_row_bytes[0] = src.yuv_row_bytes[0];
            dst.yuv_row_bytes[1] = src.yuv_row_bytes[1];
            dst.yuv_row_bytes[2] = src.yuv_row_bytes[2];
            src.yuv_planes[0] = None;
            src.yuv_planes[1] = None;
            src.yuv_planes[2] = None;
            src.yuv_row_bytes[0] = 0;
            src.yuv_row_bytes[1] = 0;
            src.yuv_row_bytes[2] = 0;
        }
        1 => {
            dst.alpha_plane = src.alpha_plane;
            dst.alpha_row_bytes = src.alpha_row_bytes;
            src.alpha_plane = None;
            src.alpha_row_bytes = 0;
        }
        _ => {
            // do nothing.
        }
    }
}

impl AvifDecoder {
    pub fn set_io_file(&mut self, filename: &String) -> AvifResult<()> {
        self.io = Some(Box::new(AvifDecoderFileIO::create(filename)?));
        Ok(())
    }

    pub fn set_io(&mut self, io: Box<dyn AvifDecoderIO>) -> AvifResult<()> {
        self.io = Some(io);
        Ok(())
    }

    pub fn parse(&mut self) -> AvifResult<&AvifImageInfo> {
        if self.io.is_none() {
            return Err(AvifError::IoNotSet);
        }
        let avif_boxes = MP4Box::parse(&mut self.io.as_mut().unwrap())?;
        self.tracks = avif_boxes.tracks;
        self.avif_items = construct_avif_items(&avif_boxes.meta)?;
        for (id, item) in &mut self.avif_items {
            item.harvest_ispe(self.settings.strictness.alpha_ispe_required())?;
        }
        //println!("{:#?}", self.avif_items);

        self.source = match self.settings.source {
            // Decide the source based on the major brand.
            AvifDecoderSource::Auto => match avif_boxes.ftyp.major_brand.as_str() {
                "avis" => AvifDecoderSource::Tracks,
                "avif" => AvifDecoderSource::PrimaryItem,
                _ => {
                    if self.tracks.is_empty() {
                        AvifDecoderSource::PrimaryItem
                    } else {
                        AvifDecoderSource::Tracks
                    }
                }
            },
            AvifDecoderSource::Tracks => AvifDecoderSource::Tracks,
            AvifDecoderSource::PrimaryItem => AvifDecoderSource::PrimaryItem,
        };

        let color_properties: &Vec<ItemProperty>;
        match self.source {
            AvifDecoderSource::Tracks => {
                let color_track = self
                    .tracks
                    .iter()
                    .find(|x| x.is_color())
                    .ok_or(AvifError::NoContent)?;
                color_properties = color_track
                    .get_properties()
                    .ok_or(AvifError::BmffParseFailed)?;

                // TODO: exif/xmp from meta.

                self.tiles[0].push(create_tile_from_track(&color_track)?);
                //println!("color_tile: {:#?}", self.tiles[0]);
                self.tile_info[0].tile_count = 1;

                match self.tracks.iter().find(|x| x.is_aux(color_track.id)) {
                    Some(alpha_track) => {
                        self.tiles[1].push(create_tile_from_track(alpha_track)?);
                        //println!("alpha_tile: {:#?}", self.tiles[1]);
                        self.tile_info[1].tile_count = 1;
                        self.image.info.alpha_present = true;
                        self.image.info.alpha_premultiplied =
                            color_track.prem_by_id == alpha_track.id;
                    }
                    None => {}
                }

                self.image_index = -1;
                self.image_count = self.tiles[0][0].input.samples.len() as u32;
                self.timescale = color_track.media_timescale;
                self.duration_in_timescales = color_track.media_duration;
                if self.timescale != 0 {
                    self.duration = (self.duration_in_timescales as f64) / (self.timescale as f64);
                } else {
                    self.duration = 0.0;
                }
                self.repetition_count = color_track.repetition_count;
                // TODO: self.image timing.

                println!("image_count: {}", self.image_count);
                println!("timescale: {}", self.timescale);
                println!("duration_in_timescales: {}", self.duration_in_timescales);

                self.image.info.width = color_track.width;
                self.image.info.height = color_track.height;
            }
            AvifDecoderSource::PrimaryItem => {
                // 0 color, 1 alpha, 2 gainmap
                let mut item_ids: [u32; 3] = [0; 3];

                // Mandatory color item.
                item_ids[0] = *self
                    .avif_items
                    .iter()
                    .find(|x| {
                        !x.1.should_skip()
                            && x.1.id != 0
                            && x.1.id == avif_boxes.meta.primary_item_id
                    })
                    .ok_or(AvifError::NoContent)?
                    .0;
                self.read_and_parse_item(item_ids[0], 0)?;

                // Optional alpha auxiliary item
                item_ids[1] =
                    find_alpha_item(&self.avif_items, self.avif_items.get(&item_ids[0]).unwrap())
                        .unwrap_or(0);
                self.read_and_parse_item(item_ids[1], 1)?;

                println!("item ids: {:#?}", item_ids);

                // TODO: gainmap item.

                // TODO: find exif or xmp metadata.

                self.image_index = -1;
                self.image_count = 1;
                self.timescale = 1;
                self.duration_in_timescales = 1;
                // TODO: duration, imagetiming.

                for (index, item_id) in item_ids.iter().enumerate() {
                    if *item_id == 0 {
                        continue;
                    }
                    {
                        let item = self.avif_items.get(&item_id).unwrap();
                        if index == 1 && item.width == 0 && item.height == 0 {
                            // NON-STANDARD: Alpha subimage does not have an ispe
                            // property; adopt width/height from color item.
                            // TODO: need to assert for strict flag.
                            // item.width = items[0].unwrap().width;
                            // item.height = items[0].unwrap().height;
                            // TODO: make this work. some mut problem.
                        }
                    }
                    self.tiles[index] = generate_tiles(
                        &mut self.avif_items,
                        &avif_boxes.meta.iinf,
                        *item_id,
                        &self.tile_info[index],
                        index,
                    )?;
                    // TODO: validate item properties.
                }

                let color_item = self.avif_items.get(&item_ids[0]).unwrap();
                self.image.info.width = color_item.width;
                self.image.info.height = color_item.height;
                self.image.info.alpha_present = item_ids[1] != 0;
                // alphapremultiplied.

                // This borrow has to be in the end of this branch.
                color_properties = &self.avif_items.get(&item_ids[0]).unwrap().properties;
            }
            _ => return Err(AvifError::UnknownError), // not reached.
        }

        // Check validity of samples.
        for tiles in &self.tiles {
            for tile in tiles {
                for sample in &tile.input.samples {
                    if sample.size == 0 {
                        println!("sample has invalid size.");
                        return Err(AvifError::BmffParseFailed);
                    }
                    // TODO: iostats?
                }
            }
        }

        // Find and adopt all colr boxes "at most one for a given value of
        // colour type" (HEIF 6.5.5.1, from Amendment 3) Accept one of each
        // type, and bail out if more than one of a given type is provided.
        //match color_item.nclx() {
        match find_nclx(color_properties) {
            Ok(nclx) => {
                self.image.info.color_primaries = nclx.color_primaries;
                self.image.info.transfer_characteristics = nclx.transfer_characteristics;
                self.image.info.matrix_coefficients = nclx.matrix_coefficients;
                self.image.info.full_range = nclx.full_range;
            }
            Err(multiple_nclx_found) => {
                if multiple_nclx_found {
                    println!("multiple nclx were found");
                    return Err(AvifError::BmffParseFailed);
                }
            }
        }
        match find_icc(color_properties) {
            Ok(icc) => {
                // TODO: attach icc to self.image.
            }
            Err(multiple_icc_found) => {
                if multiple_icc_found {
                    println!("multiple icc were found");
                    return Err(AvifError::BmffParseFailed);
                }
            }
        }

        // TODO: clli, pasp, clap, irot, imir

        // TODO: if cicp was not found, harvest it from the seq hdr.

        let av1C = find_av1C(color_properties).ok_or(AvifError::BmffParseFailed)?;
        self.image.info.depth = av1C.depth();
        self.image.info.yuv_format = if av1C.monochrome {
            PixelFormat::Monochrome
        } else {
            if av1C.chroma_subsampling_x == 1 && av1C.chroma_subsampling_y == 1 {
                PixelFormat::Yuv420
            } else if (av1C.chroma_subsampling_x == 1) {
                PixelFormat::Yuv422
            } else {
                PixelFormat::Yuv444
            }
        };
        self.image.info.chroma_sample_position = av1C.chroma_sample_position;

        Ok(&self.image.info)
    }

    fn read_and_parse_item(&mut self, item_id: u32, index: usize) -> AvifResult<()> {
        if item_id == 0 {
            return Ok(());
        }
        self.avif_items.get(&item_id).unwrap().read_and_parse(
            &mut self.io.as_mut().unwrap(),
            &mut self.tile_info[index as usize].grid,
        )
    }

    fn can_use_single_codec(&self) -> bool {
        let total_tile_count = self.tiles[0].len() + self.tiles[1].len() + self.tiles[2].len();
        if total_tile_count == 1 {
            return true;
        }
        if self.image_count != 1 {
            return false;
        }
        let mut image_buffers = 0;
        let mut stolen_image_buffers = 0;
        for category in 0usize..3 {
            if self.tile_info[category].tile_count > 0 {
                image_buffers += 1;
            }
            if self.tile_info[category].tile_count > 1 {
                stolen_image_buffers += 1;
            }
        }
        if stolen_image_buffers > 0 && image_buffers > 1 {
            // Stealing will cause problems. So we need separate codec instances.
            return false;
        }
        let operating_point = self.tiles[0][0].operating_point;
        let all_layers = self.tiles[0][0].input.all_layers;
        for tiles in &self.tiles {
            for tile in tiles {
                if tile.operating_point != operating_point || tile.input.all_layers != all_layers {
                    return false;
                }
            }
        }
        true
    }

    fn create_codec(&mut self, operating_point: u8, all_layers: bool) -> AvifResult<()> {
        let mut codec = Dav1d::default();
        codec.initialize(operating_point, all_layers)?;
        self.codecs.push(codec);
        Ok(())
    }

    fn create_codecs(&mut self) -> AvifResult<()> {
        if matches!(self.source, AvifDecoderSource::Tracks) {
            // In this case, we will use at most two codec instances (one for
            // the color planes and one for the alpha plane). Gain maps are
            // not supported.
            self.create_codec(
                self.tiles[0][0].operating_point,
                self.tiles[0][0].input.all_layers,
            )?;
            self.tiles[0][0].codec_index = 0;
            if !self.tiles[1].is_empty() {
                self.create_codec(
                    self.tiles[1][0].operating_point,
                    self.tiles[1][0].input.all_layers,
                )?;
                self.tiles[1][0].codec_index = 1;
            }
        } else {
            if self.can_use_single_codec() {
                self.create_codec(
                    self.tiles[0][0].operating_point,
                    self.tiles[0][0].input.all_layers,
                )?;
                for tiles in &mut self.tiles {
                    for tile in tiles {
                        tile.codec_index = 0;
                    }
                }
            } else {
                for category in 0usize..3 {
                    for tile_index in 0..self.tiles[category].len() {
                        let tile = &self.tiles[category][tile_index];
                        self.create_codec(tile.operating_point, tile.input.all_layers)?;
                        self.tiles[category][tile_index].codec_index = self.codecs.len() - 1;
                    }
                }
            }
        }
        Ok(())
    }

    fn prepare_samples(&mut self, image_index: usize) -> AvifResult<()> {
        // TODO: this function can probably be moved into AvifDecodeSample.data().
        for tiles in &mut self.tiles {
            for tile in tiles {
                if tile.input.samples.len() <= image_index {
                    println!("sample for index {image_index} not found.");
                    return Err(AvifError::NoImagesRemaining);
                }
                let sample = &mut tile.input.samples[image_index];
                if sample.item_id != 0 {
                    // Data comes from an item.
                    let item = self
                        .avif_items
                        .get(&sample.item_id)
                        .ok_or(AvifError::BmffParseFailed)?;
                    if item.extents.len() > 1 {
                        // Item has multiple extents, merge them into a contiguous buffer.
                        if sample.data_buffer.is_none() {
                            let mut data: Vec<u8> = Vec::new();
                            data.reserve(item.size);
                            for extent in &item.extents {
                                let io = self.io.as_mut().unwrap();
                                // TODO: extent.length usize cast safety?
                                data.extend_from_slice(
                                    io.read(extent.offset, extent.length as usize)?,
                                );
                            }
                            sample.data_buffer = Some(data);
                        }
                    } else {
                        sample.offset = item.data_offset();
                    }
                } else {
                    // TODO: handle tracks.
                }
            }
        }
        Ok(())
    }

    fn decode_tiles(&mut self, image_index: usize) -> AvifResult<()> {
        for category in 0usize..3 {
            let grid = &self.tile_info[category].grid;
            let is_grid = grid.rows > 0 && grid.columns > 0;
            if is_grid {
                self.image.allocate_planes(category)?;
            }
            for (tile_index, tile) in self.tiles[category].iter_mut().enumerate() {
                let sample = &tile.input.samples[image_index];
                let io = &mut self.io.as_mut().unwrap();
                {
                    let codec = &mut self.codecs[tile.codec_index];
                    codec.get_next_image(
                        sample.data(io)?,
                        sample.spatial_id,
                        &mut tile.image,
                        category,
                    )?;
                }

                // TODO: convert alpha from limited range to full range.
                // TODO: scale tile to match output dimension.

                if is_grid {
                    // TODO: make sure all tiles decoded properties match.
                    // Need to figure out a way to do it with proper borrows.
                    self.image.copy_from_tile(
                        &tile.image,
                        &self.tile_info[category],
                        tile_index as u32,
                        category,
                    );
                } else {
                    // Non grid path, steal planes from the only tile.
                    if category == 0 {
                        self.image.info.width = tile.image.info.width;
                        self.image.info.height = tile.image.info.height;
                        self.image.info.depth = tile.image.info.depth;
                        self.image.info.yuv_format = tile.image.info.yuv_format;
                    } else if category == 1 {
                        // check width height mismatch.
                    }

                    steal_planes(&mut self.image, &mut tile.image, category);
                }
            }
        }
        Ok(())
    }

    pub fn next_image(&mut self) -> AvifResult<&AvifImage> {
        if self.io.is_none() {
            return Err(AvifError::IoNotSet);
        }
        if self.tiles[0].is_empty() {
            return Err(AvifError::NoContent);
        }
        let next_image_index = self.image_index + 1;
        if next_image_index == 0 {
            // TODO: this may accidentally create more when nth image is added.
            // so make sure this function is called only once.
            self.create_codecs()?;
        }
        self.prepare_samples(next_image_index as usize)?;
        self.decode_tiles(next_image_index as usize)?;
        self.image_index = next_image_index;
        // TODO provide timing info for tracks.
        Ok(&self.image)
    }

    pub fn peek_compatible_file_type(data: &[u8]) -> bool {
        match MP4Box::peek_compatible_file_type(data) {
            Ok(x) => x,
            Err(err) => false,
        }
    }
}
