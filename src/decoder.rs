use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;

use crate::dav1d::*;
use crate::mp4box::ItemProperty::AuxiliaryType;
use crate::mp4box::ItemProperty::ImageSpatialExtents;
use crate::mp4box::*;
use crate::stream::*;

#[derive(Debug, Default, Copy, Clone)]
pub struct AvifImage {
    pub width: u32,
    pub height: u32,
    pub depth: u8,

    pub yuv_format: u8,
    pub full_range: bool,
    pub chroma_sample_position: u8,

    pub yuv_planes: [Option<*mut u8>; 3],
    pub yuv_row_bytes: [u32; 3], // TODO: named constant
    pub image_owns_yuv_planes: bool,

    pub alpha_plane: Option<*mut u8>,
    pub alpha_row_bytes: u32,
    pub image_owns_alpha_plane: bool,
    pub alpha_premultiplied: bool,

    pub icc: u8, //Option<Vec<u8>>,

    pub color_primaries: u16,
    pub transfer_characteristics: u16,
    pub matrix_coefficients: u16,
    // some more boxes. clli, transformations. pasp, clap, irot, imir.

    // exif, xmp.

    // gainmap.
}

#[derive(Debug)]
pub struct AvifPlane {
    pub data: *mut u8,
    pub width: u32,
    pub height: u32,
    pub row_bytes: u32,
    pub pixel_size: u32,
}

impl AvifImage {
    pub fn plane(&self, plane: usize) -> Option<AvifPlane> {
        assert!(plane < 4);
        let pixel_size = if self.depth == 8 { 1 } else { 2 };
        if plane < 3 {
            if self.yuv_planes[plane].is_none() {
                return None;
            }
            let mut plane_width = self.width;
            let mut plane_height = self.height;
            if plane > 0 {
                if self.yuv_format == 1 {
                    plane_width = (plane_width + 1) / 2;
                    plane_height = (plane_height + 1) / 2;
                } else if self.yuv_format == 2 {
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
            width: self.width,
            height: self.height,
            row_bytes: self.alpha_row_bytes,
            pixel_size,
        });
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
}

#[derive(Debug, Default)]
pub struct AvifDecoder {
    pub settings: AvifDecoderSettings,
    image: AvifImage,
    data: Vec<u8>,
    codec: Dav1d,
    source: AvifDecoderSource,
    tile_info: [AvifTileInfo; 3],
    tiles: [Vec<AvifTile>; 3],
    alpha_present: bool,
    image_index: i32,
    avif_items: HashMap<u32, AvifItem>,
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
    offset_relative_to_idat: bool,
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
    fn read_and_parse(&self) -> bool {
        // TODO: this function also has to extract codec type.
        if self.item_type != "grid" {
            return true;
        }
        // TODO: read grid info.
        true
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

    fn harvest_ispe(&mut self) -> bool {
        if self.size == 0 {
            return true;
        }
        if self.has_unsupported_essential_property {
            // An essential property isn't supported by libavif. Ignore.
            return true;
        }

        let is_grid = self.item_type == "grid";
        if self.item_type != "av01" && !is_grid {
            // probably exif or some other data.
            return true;
        }
        match find_property!(self, ImageSpatialExtents) {
            Some(property) => match property {
                ItemProperty::ImageSpatialExtents(x) => {
                    self.width = x.width;
                    self.height = x.height;
                    if self.width == 0 || self.height == 0 {
                        println!("item id has invalid size.");
                        return false;
                    }
                }
                _ => return false, // not reached.
            },
            None => {
                // No ispe was found.
                if self.is_auxiliary_alpha() {
                    // TODO: provide a strict flag to bypass this check.
                    println!("alpha auxiliary image is missing mandatory ispe");
                    return false;
                } else {
                    println!("item id is missing mandatory ispe property");
                    return false;
                }
            }
        }
        true
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

    fn nclx(&self) -> Result<&Nclx, bool> {
        let nclx_properties: Vec<_> = self
            .properties
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

    fn icc(&self) -> Result<&Icc, bool> {
        let icc_properties: Vec<_> = self
            .properties
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
}

fn read_file(filename: &String) -> Vec<u8> {
    let mut file = File::open(filename).expect("file not found");
    let mut data: Vec<u8> = Vec::new();
    let _ = file.read_to_end(&mut data);
    data
}

// This design is not final. It's possible to do this in the same loop where boxes are parsed. But it
// seems a little cleaner to do this after the fact.
fn construct_avif_items(meta: &MetaBox) -> Result<HashMap<u32, AvifItem>, &str> {
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
            return Err("item already has extents.");
        }
        // TODO: infer idat stored once construction method is implemented.
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
            return Err("item has duplictate ipma.");
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
                return Err("invalid property_index in ipma.");
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
        let item = avif_items.get_mut(&reference.from_item_id);
        if item.is_none() {
            return Err("invalid from_item_id in iref");
        }
        let item = item.unwrap();
        match reference.reference_type.as_str() {
            "thmb" => item.thumbnail_for_id = reference.to_item_id,
            "auxl" => item.aux_for_id = reference.to_item_id,
            "cdsc" => item.desc_for_id = reference.to_item_id,
            "prem" => item.prem_by_id = reference.to_item_id,
            "dimg" => {
                // derived images refer in the opposite direction.
                let dimg_item = avif_items.get_mut(&reference.to_item_id);
                if dimg_item.is_none() {
                    return Err("invalid to_item_id in iref");
                }
                let dimg_item = dimg_item.unwrap();
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

fn should_skip_decoder_item(item: &AvifItem) -> bool {
    item.size == 0
        || item.has_unsupported_essential_property
        || (item.item_type != "av01" && item.item_type != "grid")
        || item.thumbnail_for_id != 0
}

fn find_color_item(avif_items: &HashMap<u32, AvifItem>, primary_item_id: u32) -> Option<&AvifItem> {
    if primary_item_id == 0 {
        return None;
    }
    // TODO: perhaps this can be an idiomatic oneliner ?
    for (_, item) in avif_items.iter() {
        if should_skip_decoder_item(item) {
            continue;
        }
        if item.id == primary_item_id {
            return Some(item);
        }
    }
    None
}

fn find_alpha_item<'a>(
    avif_items: &'a HashMap<u32, AvifItem>,
    color_item: &AvifItem,
) -> Option<&'a AvifItem> {
    for (_, item) in avif_items.iter() {
        if should_skip_decoder_item(item) {
            continue;
        }
        if item.aux_for_id != color_item.id {
            continue;
        }
        if !item.is_auxiliary_alpha() {
            continue;
        }
        return Some(item);
    }
    if color_item.item_type != "grid" {
        return None;
    }
    // TODO: If color item is a grid, check if there is an alpha channel which is represented as an auxl item to each color tile item.
    None
}

#[derive(Debug, Default)]
struct AvifDecodeSample {
    data_offset: u64,
    // owns_data
    // partial_data
    item_id: u32,
    offset: u64,
    size: usize,
    spatial_id: u8,
    sync: bool,
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
    codec: Dav1d,
}

fn create_tile(item: &AvifItem) -> Option<AvifTile> {
    let mut tile = AvifTile::default();
    tile.width = item.width;
    tile.height = item.height;
    tile.operating_point = item.operating_point();
    tile.image = AvifImage::default();
    // TODO: do all the layer stuff in avifCodecDecodeInputFillFromDecoderItem.
    // Typical case: Use the entire item's payload for a single frame output
    let sample = AvifDecodeSample {
        data_offset: 0,
        item_id: item.id,
        offset: 0,
        size: item.size,
        // Legal spatial_id values are [0,1,2,3], so this serves as a sentinel
        // value for "do not filter by spatial_id"
        spatial_id: 0xff,
        sync: true,
    };
    tile.input.samples.push(sample);
    Some(tile)
}

fn generate_tiles(item: &AvifItem, info: &AvifTileInfo, category: usize) -> Option<Vec<AvifTile>> {
    let mut tiles: Vec<AvifTile> = Vec::new();
    if info.grid.rows > 0 && info.grid.columns > 0 {
        // TODO: grid tiles.
    } else {
        if item.size == 0 {
            return None;
        }
        let tile = create_tile(item);
        if tile.is_none() {
            return None;
        }
        let mut tile = tile.unwrap();
        tile.input.category = category as u8;
        tiles.push(tile);
    }
    Some(tiles)
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
    pub fn set_file(&mut self, filename: &String) {
        self.data = read_file(filename);
    }

    pub fn parse(&mut self) -> Option<&AvifImage> {
        let mut stream = IStream {
            // TODO: learn to store references in struct.
            data: self.data.clone(),
            offset: 0,
        };
        let avif_boxes = MP4Box::parse(&mut stream);
        self.avif_items = match construct_avif_items(&avif_boxes.meta) {
            Ok(items) => items,
            Err(err) => {
                println!("failed to construct_avif_items: {err}");
                return None;
            }
        };
        for (id, item) in &mut self.avif_items {
            if !item.harvest_ispe() {
                println!("failed to harvest ispe");
                return None;
            }
        }
        println!("{:#?}", self.avif_items);

        // Build the decoder input.
        self.source = self.settings.source;
        match self.settings.source {
            AvifDecoderSource::Auto => {
                // Decide the source based on the major brand.
                if avif_boxes.ftyp.major_brand == "avis" {
                    self.source = AvifDecoderSource::Tracks;
                } else if avif_boxes.ftyp.major_brand == "avif" {
                    self.source = AvifDecoderSource::PrimaryItem;
                } else {
                    // TODO: add a else if for if track count > 0, then use tracks.
                    self.source = AvifDecoderSource::PrimaryItem;
                }
            }
            _ => {}
        }

        // 0 color, 1 alpha, 2 gainmap
        let mut items: [Option<&AvifItem>; 3] = [None; 3];

        match self.source {
            AvifDecoderSource::Tracks => {
                // TODO: implement.
            }
            AvifDecoderSource::PrimaryItem => {
                // Mandatory color item.
                items[0] = find_color_item(&self.avif_items, avif_boxes.meta.primary_item_id);
                if items[0].is_none() {
                    println!("primary color item not found.");
                    return None;
                }
                println!("color item: {:#?}", items[0].unwrap());
                if !items[0].unwrap().read_and_parse() {
                    println!("failed to read_and_parse color item");
                    return None;
                }

                // Optional alpha auxiliary item
                items[1] = find_alpha_item(&self.avif_items, items[0].unwrap());
                if items[1].is_some() && !items[1].unwrap().read_and_parse() {
                    println!("failed to read_and_parse alpha item");
                    return None;
                }

                // TODO: gainmap item.

                // TODO: find exif or xmp metadata.

                self.image_index = -1;
                //self.image_count = 1;
                // TODO: image timing for avis.
                for (index, item) in items.iter_mut().enumerate() {
                    if item.is_none() {
                        continue;
                    }
                    let item = item.unwrap();
                    if index == 1 && item.width == 0 && item.height == 0 {
                        // NON-STANDARD: Alpha subimage does not have an ispe
                        // property; adopt width/height from color item.
                        // TODO: need to assert for strict flag.
                        // item.width = items[0].unwrap().width;
                        // item.height = items[0].unwrap().height;
                        // TODO: make this work. some mut problem.
                    }
                    let tiles = generate_tiles(item, &self.tile_info[index], index);
                    if tiles.is_none() {
                        println!("Failed to generate_tiles");
                        return None;
                    }
                    self.tiles[index] = tiles.unwrap();
                    // TODO: validate item properties.
                }
                println!("^^^=====");
                println!("{:#?}", self.tiles);
                println!("$$$=====");

                self.image.width = items[0].unwrap().width;
                self.image.height = items[0].unwrap().height;
                self.alpha_present = items[1].is_some();
                // alphapremultiplied.
            }
            _ => {}
        }

        // Check validity of samples.
        for tiles in &self.tiles {
            for tile in tiles {
                for sample in &tile.input.samples {
                    if sample.size == 0 {
                        println!("sample has invalid size.");
                        return None;
                    }
                    // TODO: iostats?
                }
            }
        }

        // Find and adopt all colr boxes "at most one for a given value of
        // colour type" (HEIF 6.5.5.1, from Amendment 3) Accept one of each
        // type, and bail out if more than one of a given type is provided.
        match items[0].unwrap().nclx() {
            Ok(nclx) => {
                self.image.color_primaries = nclx.color_primaries;
                self.image.transfer_characteristics = nclx.transfer_characteristics;
                self.image.matrix_coefficients = nclx.matrix_coefficients;
                self.image.full_range = nclx.full_range;
            }
            Err(multiple_nclx_found) => {
                if multiple_nclx_found {
                    println!("multiple nclx were found");
                    return None;
                }
            }
        }
        match items[0].unwrap().icc() {
            Ok(icc) => {
                // TODO: attach icc to self.image.
            }
            Err(multiple_icc_found) => {
                if multiple_icc_found {
                    println!("multiple icc were found");
                    return None;
                }
            }
        }

        // TODO: clli, pasp, clap, irot, imir

        // TODO: if cicp was not found, harvest it from the seq hdr.

        // TODO: copy info from av1c. avifReadCodecConfigProperty.
        let av1C = items[0].unwrap().av1C();
        if av1C.is_none() {
            println!("missing av1C");
            return None;
        }
        let av1C = av1C.unwrap();
        if av1C.monochrome {
            self.image.yuv_format = 0;
        } else {
            if av1C.chroma_subsampling_x == 1 && av1C.chroma_subsampling_y == 1 {
                self.image.yuv_format = 1;
            } else if (av1C.chroma_subsampling_x == 1) {
                self.image.yuv_format = 2;
            } else {
                self.image.yuv_format = 3;
            }
        }
        self.image.chroma_sample_position = av1C.chroma_sample_position;

        Some(&self.image)
    }

    fn create_codecs(&mut self) -> bool {
        // TODO: share codecs for grid, etc.
        for tiles in &mut self.tiles {
            for tile in tiles {
                tile.codec
                    .initialize(tile.operating_point, tile.input.all_layers);
            }
        }
        true
    }

    fn prepare_samples(&mut self, image_index: usize) -> bool {
        for tiles in &mut self.tiles {
            for tile in tiles {
                let sample = &mut tile.input.samples[image_index];
                if sample.item_id != 0 {
                    // Data comes from an item.
                    let item = self.avif_items.get(&sample.item_id);
                    if item.is_none() {
                        return false;
                    }
                    let item = item.unwrap();
                    // TODO: account for merged extent ?
                    sample.data_offset = item.extents[0].offset;
                } else {
                    // TODO: handle tracks.
                }
            }
        }
        true
    }

    fn decode_tiles(&mut self, image_index: usize) -> bool {
        for (category, tiles) in self.tiles.iter_mut().enumerate() {
            for tile in tiles {
                let sample = &tile.input.samples[image_index];
                let payload_start: usize = sample.data_offset as usize;
                let payload_size: usize = sample.size as usize;
                let sample_payload = &self.data[payload_start..payload_start + payload_size];
                if !tile
                    .codec
                    .get_next_image(sample_payload, &mut tile.image, category)
                {
                    return false;
                }
                // TODO: convert alpha from limited range to full range.
                // TODO: scale tile to match output dimension.

                let grid = &self.tile_info[category].grid;
                if grid.rows > 0 && grid.columns > 0 {
                    // TODO: grid path.
                } else {
                    // Non grid path, steal planes from the only tile.

                    if category == 0 {
                        self.image.width = tile.image.width;
                        self.image.height = tile.image.height;
                        self.image.depth = tile.image.depth;
                        self.image.yuv_format = tile.image.yuv_format;
                    } else if category == 1 {
                        // check width height mismatch.
                    }

                    steal_planes(&mut self.image, &mut tile.image, category);
                }
            }
        }
        true
    }

    pub fn next_image(&mut self) -> Option<&AvifImage> {
        if self.tiles[0].is_empty() && self.tiles[1].is_empty() && self.tiles[2].is_empty() {
            // Nothing has been parsed yet.
            return None;
        }

        let next_image_index = self.image_index + 1;
        if !self.create_codecs() {
            return None;
        }
        if !self.prepare_samples(next_image_index as usize) {
            return None;
        }
        if !self.decode_tiles(next_image_index as usize) {
            return None;
        }

        self.image_index = next_image_index;
        // TODO provide timing info for tracks.
        Some(&self.image)
    }
}
