pub mod gainmap;
pub mod item;
pub mod tile;
pub mod track;

use std::collections::HashSet;

use crate::decoder::gainmap::*;
use crate::decoder::item::*;
use crate::decoder::tile::*;
use crate::decoder::track::*;

use crate::codecs::dav1d::Dav1d;
use crate::codecs::Decoder as DecoderTrait;
use crate::image::*;
use crate::internal_utils::io::*;
use crate::internal_utils::*;
use crate::parser::exif;
use crate::parser::mp4box;
use crate::parser::mp4box::*;
use crate::parser::obu;
use crate::*;

pub trait IO {
    fn read(&mut self, offset: u64, size: usize) -> AvifResult<&[u8]>;
    fn size_hint(&self) -> u64;
    fn persistent(&self) -> bool;
}

pub type GenericIO = Box<dyn IO>;
pub type Codec = Box<dyn crate::codecs::Decoder>;

#[derive(Debug, Copy, Clone, Default)]
pub enum Source {
    Tracks,
    PrimaryItem,
    #[default]
    Auto,
    // TODO: Thumbnail,
}

#[derive(Debug, Default)]
pub struct Settings {
    pub source: Source,
    pub ignore_exif: bool,
    pub ignore_xmp: bool,
    pub strictness: Strictness,
    pub allow_progressive: bool,
    pub enable_decoding_gainmap: bool,
    pub enable_parsing_gainmap_metadata: bool,
}

#[derive(Debug)]
pub enum StrictnessFlag {
    PixiRequired,
    ClapValid,
    AlphaIspeRequired,
}

#[derive(Debug, Default)]
pub enum Strictness {
    None,
    #[default]
    All,
    SpecificInclude(Vec<StrictnessFlag>),
    SpecificExclude(Vec<StrictnessFlag>),
}

#[derive(Debug, Default, Copy, Clone)]
pub enum ProgressiveState {
    #[default]
    Unavailable,
    Available,
    Active,
}

impl Strictness {
    pub fn pixi_required(&self) -> bool {
        match self {
            Strictness::All => true,
            Strictness::SpecificInclude(flags) => flags
                .iter()
                .any(|x| matches!(x, StrictnessFlag::PixiRequired)),
            Strictness::SpecificExclude(flags) => !flags
                .iter()
                .any(|x| matches!(x, StrictnessFlag::PixiRequired)),
            _ => false,
        }
    }

    pub fn alpha_ispe_required(&self) -> bool {
        match self {
            Strictness::All => true,
            Strictness::SpecificInclude(flags) => flags
                .iter()
                .any(|x| matches!(x, StrictnessFlag::AlphaIspeRequired)),
            Strictness::SpecificExclude(flags) => !flags
                .iter()
                .any(|x| matches!(x, StrictnessFlag::AlphaIspeRequired)),
            _ => false,
        }
    }
}

#[derive(Default)]
pub struct Decoder {
    pub settings: Settings,
    image: Image,
    source: Source,
    tile_info: [TileInfo; 3],
    tiles: [Vec<Tile>; 3],
    image_index: i32,
    pub image_count: u32,
    pub timescale: u32,
    pub duration_in_timescales: u64,
    pub duration: f64,
    pub repetition_count: i32,
    pub gainmap: GainMap,
    pub gainmap_present: bool,
    items: Items,
    tracks: Vec<Track>,
    // To replicate the C-API, we need to keep this optional. Otherwise this
    // could be part of the initialization.
    io: Option<GenericIO>,
    codecs: Vec<Codec>,
}

fn find_nclx(properties: &[ItemProperty]) -> Result<&Nclx, bool> {
    let nclx_properties: Vec<_> = properties
        .iter()
        .filter(|x| match x {
            ItemProperty::ColorInformation(colr) => matches!(colr, ColorInformation::Nclx(_)),
            _ => false,
        })
        .collect();
    match nclx_properties.len() {
        0 => Err(false),
        1 => match nclx_properties[0] {
            ItemProperty::ColorInformation(ColorInformation::Nclx(nclx)) => Ok(nclx),
            _ => Err(false), // not reached.
        },
        _ => Err(true), // multiple nclx were found.
    }
}

fn find_icc(properties: &[ItemProperty]) -> Result<Vec<u8>, bool> {
    let icc_properties: Vec<_> = properties
        .iter()
        .filter(|x| match x {
            ItemProperty::ColorInformation(colr) => matches!(colr, ColorInformation::Icc(_)),
            _ => false,
        })
        .collect();
    match icc_properties.len() {
        0 => Err(false),
        1 => match icc_properties[0] {
            ItemProperty::ColorInformation(ColorInformation::Icc(icc)) => Ok(icc.to_vec()),
            _ => Err(false), // not reached.
        },
        _ => Err(true), // multiple icc were found.
    }
}

#[allow(non_snake_case)]
fn find_av1C(properties: &[ItemProperty]) -> Option<&CodecConfiguration> {
    match properties
        .iter()
        .find(|x| matches!(x, ItemProperty::CodecConfiguration(_)))
    {
        Some(ItemProperty::CodecConfiguration(av1C)) => Some(av1C),
        _ => None,
    }
}

impl Decoder {
    pub fn set_io_file(&mut self, filename: &String) -> AvifResult<()> {
        self.io = Some(Box::new(DecoderFileIO::create(filename)?));
        Ok(())
    }

    pub fn set_io(&mut self, io: GenericIO) -> AvifResult<()> {
        self.io = Some(io);
        Ok(())
    }

    #[allow(non_snake_case)]
    fn find_alpha_item(&self, color_item_index: u32) -> (u32, Option<Item>) {
        let color_item = self.items.get(&color_item_index).unwrap();
        if let Some(item) = self.items.iter().find(|x| {
            !x.1.should_skip() && x.1.aux_for_id == color_item.id && x.1.is_auxiliary_alpha()
        }) {
            return (*item.0, None);
        }
        if color_item.item_type != "grid" || color_item.grid_item_ids.is_empty() {
            return (0, None);
        }
        // If color item is a grid, check if there is an alpha channel which is represented as an
        // auxl item to each color tile item.
        let mut alpha_item_indices: Vec<u32> = Vec::new();
        for color_grid_item_id in &color_item.grid_item_ids {
            match self
                .items
                .iter()
                .find(|x| x.1.aux_for_id == *color_grid_item_id && x.1.is_auxiliary_alpha())
            {
                Some(item) => alpha_item_indices.push(*item.0),
                None => {
                    // TODO: This case must be an error.
                    println!("alpha aux item was not found for color tile.");
                    return (0, None);
                }
            }
        }
        assert!(color_item.grid_item_ids.len() == alpha_item_indices.len());
        let first_item = self.items.get(&alpha_item_indices[0]).unwrap();
        let properties = match first_item.av1C() {
            Some(av1C) => vec![ItemProperty::CodecConfiguration(av1C.clone())],
            None => return (0, None),
        };
        (
            0,
            Some(Item {
                id: self.items.keys().max().unwrap() + 1,
                item_type: String::from("grid"),
                width: color_item.width,
                height: color_item.height,
                grid_item_ids: alpha_item_indices,
                properties,
                ..Item::default()
            }),
        )
    }

    // returns (tone_mapped_image_item_id, gain_map_item_id)
    fn find_tone_mapped_image_item(&self, color_item_id: u32) -> AvifResult<(u32, u32)> {
        // TODO: may have to find_all instead of find one.
        let tmap_items: Vec<_> = self.items.values().filter(|x| x.is_tmap()).collect();
        for item in tmap_items {
            println!("found a tonemapped item: {:#?}", item.id);
            let dimg_items: Vec<_> = self
                .items
                .values()
                .filter(|x| x.dimg_for_id == item.id)
                .collect();
            if dimg_items.len() != 2 {
                println!("Expected tmap to have 2 dimg items");
                return Err(AvifError::InvalidToneMappedImage);
            }
            let item0 = if dimg_items[0].dimg_index == 0 {
                dimg_items[0]
            } else {
                dimg_items[1]
            };
            if item0.id != color_item_id {
                continue;
            }
            let item1 = if dimg_items[0].dimg_index == 0 {
                dimg_items[1]
            } else {
                dimg_items[0]
            };
            return Ok((item.id, item1.id));
        }
        Ok((0, 0))
    }

    fn find_gainmap_item(&self, color_item_id: u32) -> AvifResult<(u32, u32)> {
        let (tonemap_id, gainmap_id) = self.find_tone_mapped_image_item(color_item_id)?;
        if tonemap_id == 0 || gainmap_id == 0 {
            return Ok((0, 0));
        }
        println!("tonemap_id: {tonemap_id} gainmap_id: {gainmap_id}");
        let gainmap_item = self
            .items
            .get(&gainmap_id)
            .ok_or(AvifError::InvalidToneMappedImage)?;
        if gainmap_item.should_skip() {
            return Err(AvifError::InvalidToneMappedImage);
        }
        Ok((tonemap_id, gainmap_id))
    }

    fn validate_gainmap_item(&mut self, gainmap_id: u32) -> AvifResult<()> {
        let gainmap_item = self
            .items
            .get(&gainmap_id)
            .ok_or(AvifError::InvalidToneMappedImage)?;
        if let Ok(nclx) = find_nclx(&gainmap_item.properties) {
            println!("found nclx: {:#?}", nclx);
            self.gainmap.image.info.color_primaries = nclx.color_primaries;
            self.gainmap.image.info.transfer_characteristics = nclx.transfer_characteristics;
            self.gainmap.image.info.matrix_coefficients = nclx.matrix_coefficients;
            self.gainmap.image.info.full_range = nclx.full_range;
        }
        // Find and adopt all colr boxes "at most one for a given value of colour type"
        // (HEIF 6.5.5.1, from Amendment 3) Accept one of each type, and bail out if more than one
        // of a given type is provided.
        let tonemap_item = self
            .items
            .get(&gainmap_id)
            .ok_or(AvifError::InvalidToneMappedImage)?;
        match find_nclx(&tonemap_item.properties) {
            Ok(nclx) => {
                self.gainmap.alt_color_primaries = nclx.color_primaries;
                self.gainmap.alt_transfer_characteristics = nclx.transfer_characteristics;
                self.gainmap.alt_matrix_coefficients = nclx.matrix_coefficients;
                self.gainmap.alt_full_range = nclx.full_range;
            }
            Err(multiple_nclx_found) => {
                if multiple_nclx_found {
                    println!("multiple nclx were found for tonemap");
                    return Err(AvifError::BmffParseFailed);
                }
            }
        }
        match find_icc(&tonemap_item.properties) {
            Ok(icc) => {
                self.gainmap.alt_icc = icc;
            }
            Err(multiple_icc_found) => {
                if multiple_icc_found {
                    println!("multiple icc were found for tonemap");
                    return Err(AvifError::BmffParseFailed);
                }
            }
        }
        if let Some(clli) = tonemap_item.clli() {
            self.gainmap.alt_clli = *clli;
        }
        if let Some(pixi) = tonemap_item.pixi() {
            if pixi.plane_count == 0 {
                println!("invalid plane count in tonemap");
                return Err(AvifError::BmffParseFailed);
            }
            self.gainmap.alt_plane_count = pixi.plane_count;
            self.gainmap.alt_plane_depth = pixi.plane_depths[0];
        }
        Ok(())
    }

    fn search_exif_or_xmp_metadata(&mut self, color_item_index: u32) -> AvifResult<()> {
        if self.settings.ignore_exif && self.settings.ignore_xmp {
            return Ok(());
        }
        if !self.settings.ignore_exif {
            if let Some(exif) = self.items.iter().find(|x| x.1.is_exif(color_item_index)) {
                let mut stream = exif.1.stream(self.io.as_mut().unwrap())?;
                exif::parse(&mut stream)?;
                self.image
                    .info
                    .exif
                    .extend_from_slice(stream.get_slice(stream.bytes_left())?);
            }
        }
        if !self.settings.ignore_xmp {
            if let Some(xmp) = self.items.iter().find(|x| x.1.is_xmp(color_item_index)) {
                let mut stream = xmp.1.stream(self.io.as_mut().unwrap())?;
                self.image
                    .info
                    .xmp
                    .extend_from_slice(stream.get_slice(stream.bytes_left())?);
            }
        }
        Ok(())
    }

    fn generate_tiles(&mut self, item_id: u32, category: usize) -> AvifResult<Vec<Tile>> {
        let mut tiles: Vec<Tile> = Vec::new();
        let item = self
            .items
            .get(&item_id)
            .ok_or(AvifError::MissingImageItem)?;
        if !item.grid_item_ids.is_empty() {
            let grid = &self.tile_info[category].grid;
            if grid.rows == 0 || grid.columns == 0 {
                println!("multiple dimg items were found but image is not grid.");
                return Err(AvifError::InvalidImageGrid);
            }
            println!("grid###: {:#?}", grid);
            let grid_item_ids = item.grid_item_ids.clone();
            for grid_item_id in &grid_item_ids {
                let grid_item = self
                    .items
                    .get_mut(grid_item_id)
                    .ok_or(AvifError::InvalidImageGrid)?;
                let mut tile = Tile::create_from_item(grid_item, self.settings.allow_progressive)?;
                tile.input.category = category as u8;
                tiles.push(tile);
            }

            if category == 0 && self.items.get(&grid_item_ids[0]).unwrap().progressive {
                // Propagate the progressive status to the top-level grid item.
                let item = self
                    .items
                    .get_mut(&item_id)
                    .ok_or(AvifError::MissingImageItem)?;
                item.progressive = true;
            }
        } else {
            if item.size == 0 {
                return Err(AvifError::MissingImageItem);
            }
            let item = self
                .items
                .get_mut(&item_id)
                .ok_or(AvifError::MissingImageItem)?;
            let mut tile = Tile::create_from_item(item, self.settings.allow_progressive)?;
            tile.input.category = category as u8;
            tiles.push(tile);
        }
        Ok(tiles)
    }

    fn harvest_cicp_from_sequence_header(&mut self) -> AvifResult<()> {
        println!("HARVESTING!");
        if self.tiles[0].is_empty() {
            return Ok(());
        }
        // TODO: do this incrementally instead of preparing whole sample. Start with 64 and go up to
        // 4096 incrementally.
        self.prepare_sample(0, 0, 0)?;
        let io = &mut self.io.as_mut().unwrap();
        let sample = &self.tiles[0][0].input.samples[0];
        match obu::parse_sequence_header(sample.data(io)?) {
            Ok(sequence_header) => {
                self.image.info.color_primaries = sequence_header.color_primaries;
                self.image.info.transfer_characteristics = sequence_header.transfer_characteristics;
                self.image.info.matrix_coefficients = sequence_header.matrix_coefficients;
                self.image.info.full_range = sequence_header.full_range;
            }
            Err(_) => {
                println!("errored :(");
            }
        }
        Ok(())
    }

    #[allow(non_snake_case)]
    fn populate_grid_item_ids(
        &mut self,
        iinf: &Vec<ItemInfo>,
        item_id: u32,
        category: usize,
    ) -> AvifResult<()> {
        if self.items.get(&item_id).unwrap().item_type != "grid" {
            return Ok(());
        }
        let mut grid_item_ids: Vec<u32> = Vec::new();
        let mut first_av1C = CodecConfiguration::default();
        let mut is_first = true;
        // Collect all the dimg items. Cannot directly iterate through items here directly
        // because HashMap is not ordered.
        for item_info in iinf {
            let dimg_item = self
                .items
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
            if is_first {
                // Adopt the configuration property of the first tile.
                first_av1C = dimg_item.av1C().ok_or(AvifError::BmffParseFailed)?.clone();
                is_first = false;
            }
            grid_item_ids.push(item_info.item_id);
        }
        println!(
            "### category {category} grid item ids: {:#?}",
            grid_item_ids
        );
        let grid_count = self.tile_info[category].grid.rows * self.tile_info[category].grid.columns;
        if grid_item_ids.len() as u32 != grid_count {
            println!("Expected number of tiles not found");
            return Err(AvifError::InvalidImageGrid);
        }
        let item = self
            .items
            .get_mut(&item_id)
            .ok_or(AvifError::InvalidImageGrid)?;
        item.properties
            .push(ItemProperty::CodecConfiguration(first_av1C));
        item.grid_item_ids = grid_item_ids;
        Ok(())
    }

    #[allow(non_snake_case)]
    pub fn parse(&mut self) -> AvifResult<&ImageInfo> {
        if self.io.is_none() {
            return Err(AvifError::IoNotSet);
        }
        let avif_boxes = mp4box::parse(self.io.as_mut().unwrap())?;
        self.tracks = avif_boxes.tracks;
        self.items = construct_items(&avif_boxes.meta)?;
        for item in self.items.values_mut() {
            item.harvest_ispe(self.settings.strictness.alpha_ispe_required())?;
        }
        self.image.info.image_sequence_track_present = !self.tracks.is_empty();
        //println!("{:#?}", self.items);

        self.source = match self.settings.source {
            // Decide the source based on the major brand.
            Source::Auto => match avif_boxes.ftyp.major_brand.as_str() {
                "avis" => Source::Tracks,
                "avif" => Source::PrimaryItem,
                _ => {
                    if self.tracks.is_empty() {
                        Source::PrimaryItem
                    } else {
                        Source::Tracks
                    }
                }
            },
            Source::Tracks => Source::Tracks,
            Source::PrimaryItem => Source::PrimaryItem,
        };

        let color_properties: &Vec<ItemProperty>;
        match self.source {
            Source::Tracks => {
                let color_track = self
                    .tracks
                    .iter()
                    .find(|x| x.is_color())
                    .ok_or(AvifError::NoContent)?;
                color_properties = color_track
                    .get_properties()
                    .ok_or(AvifError::BmffParseFailed)?;

                // TODO: exif/xmp from meta.

                self.tiles[0].push(Tile::create_from_track(color_track)?);
                self.tile_info[0].tile_count = 1;

                if let Some(alpha_track) = self.tracks.iter().find(|x| x.is_aux(color_track.id)) {
                    self.tiles[1].push(Tile::create_from_track(alpha_track)?);
                    //println!("alpha_tile: {:#?}", self.tiles[1]);
                    self.tile_info[1].tile_count = 1;
                    self.image.info.alpha_present = true;
                    self.image.info.alpha_premultiplied = color_track.prem_by_id == alpha_track.id;
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
            Source::PrimaryItem => {
                // 0 color, 1 alpha, 2 gainmap
                let mut item_ids: [u32; 3] = [0; 3];

                // Mandatory color item.
                item_ids[0] = *self
                    .items
                    .iter()
                    .find(|x| {
                        !x.1.should_skip()
                            && x.1.id != 0
                            && x.1.id == avif_boxes.meta.primary_item_id
                    })
                    .ok_or(AvifError::NoContent)?
                    .0;
                self.read_and_parse_item(item_ids[0], 0)?;
                self.populate_grid_item_ids(&avif_boxes.meta.iinf, item_ids[0], 0)?;

                // Optional alpha auxiliary item
                let mut ignore_pixi_validation_for_alpha = false;
                let (alpha_item_id, alpha_item) = self.find_alpha_item(item_ids[0]);
                if alpha_item_id != 0 {
                    item_ids[1] = alpha_item_id;
                    self.read_and_parse_item(item_ids[1], 1)?;
                    self.populate_grid_item_ids(&avif_boxes.meta.iinf, item_ids[1], 1)?;
                } else if alpha_item.is_some() {
                    // Alpha item was made up and not part of the input. Make it part of the items
                    // array.
                    let alpha_item = alpha_item.unwrap();
                    item_ids[1] = alpha_item.id;
                    self.tile_info[1].grid = self.tile_info[0].grid;
                    self.items.insert(item_ids[1], alpha_item);
                    // Made up alpha item does not contain the pixi property. So do not try to
                    // validate it.
                    ignore_pixi_validation_for_alpha = true;
                } else {
                    // No alpha channel.
                    item_ids[1] = 0;
                }

                // Optional gainmap item
                let (tonemap_id, gainmap_id) = self.find_gainmap_item(item_ids[0])?;
                if tonemap_id != 0 && gainmap_id != 0 {
                    self.read_and_parse_item(gainmap_id, 2)?;
                    self.populate_grid_item_ids(&avif_boxes.meta.iinf, gainmap_id, 2)?;
                    if self.settings.enable_decoding_gainmap {
                        self.validate_gainmap_item(gainmap_id)?;
                        item_ids[2] = gainmap_id;
                    } else {
                        self.gainmap_present = true;
                    }
                    if self.settings.enable_parsing_gainmap_metadata {
                        let tonemap_item = self
                            .items
                            .get(&tonemap_id)
                            .ok_or(AvifError::InvalidToneMappedImage)?;
                        let mut stream = tonemap_item.stream(self.io.as_mut().unwrap())?;
                        println!("tonemap stream size: {}", stream.data.len());
                        self.gainmap.metadata = mp4box::parse_tmap(&mut stream)?;
                    }
                    println!("gainmap: {:#?}", self.gainmap);
                }

                println!("item ids: {:#?}", item_ids);

                self.search_exif_or_xmp_metadata(item_ids[0])?;

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
                        let item = self.items.get(item_id).unwrap();
                        if index == 1 && item.width == 0 && item.height == 0 {
                            // NON-STANDARD: Alpha subimage does not have an ispe property; adopt
                            // width/height from color item.
                            assert!(!self.settings.strictness.alpha_ispe_required());
                            let color_item = self.items.get(&item_ids[0]).unwrap();
                            let width = color_item.width;
                            let height = color_item.height;
                            let alpha_item = self.items.get_mut(item_id).unwrap();
                            // Note: We cannot directly use color_item.width here because borrow
                            // checker won't allow that.
                            alpha_item.width = width;
                            alpha_item.height = height;
                        }
                    }
                    self.tiles[index] = self.generate_tiles(*item_id, index)?;
                    let pixi_required = self.settings.strictness.pixi_required()
                        && (index != 1 || !ignore_pixi_validation_for_alpha);
                    let item = self.items.get(item_id).unwrap();
                    item.validate_properties(&self.items, pixi_required)?;
                }
                println!("hello");

                let color_item = self.items.get(&item_ids[0]).unwrap();
                self.image.info.width = color_item.width;
                self.image.info.height = color_item.height;
                self.image.info.alpha_present = item_ids[1] != 0;
                // alphapremultiplied.

                if color_item.progressive {
                    self.image.info.progressive_state = ProgressiveState::Available;
                    let sample_count = self.tiles[0][0].input.samples.len();
                    if sample_count > 1 {
                        self.image.info.progressive_state = ProgressiveState::Active;
                        self.image_count = sample_count as u32;
                    }
                }

                if item_ids[2] != 0 {
                    self.gainmap_present = true;
                    let gainmap_item = self.items.get(&item_ids[2]).unwrap();
                    self.gainmap.image.info.width = gainmap_item.width;
                    self.gainmap.image.info.height = gainmap_item.height;
                    let av1C = gainmap_item.av1C().ok_or(AvifError::BmffParseFailed)?;
                    self.gainmap.image.info.depth = av1C.depth();
                    self.gainmap.image.info.yuv_format = av1C.pixel_format();
                    self.gainmap.image.info.chroma_sample_position = av1C.chroma_sample_position;
                }

                // This borrow has to be in the end of this branch.
                color_properties = &self.items.get(&item_ids[0]).unwrap().properties;
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

        // Find and adopt all colr boxes "at most one for a given value of colour type"
        // (HEIF 6.5.5.1, from Amendment 3) Accept one of each type, and bail out if more than one
        // of a given type is provided.
        let mut cicp_set = false;
        match find_nclx(color_properties) {
            Ok(nclx) => {
                self.image.info.color_primaries = nclx.color_primaries;
                self.image.info.transfer_characteristics = nclx.transfer_characteristics;
                self.image.info.matrix_coefficients = nclx.matrix_coefficients;
                self.image.info.full_range = nclx.full_range;
                cicp_set = true;
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
                self.image.info.icc = icc;
            }
            Err(multiple_icc_found) => {
                if multiple_icc_found {
                    println!("multiple icc were found");
                    return Err(AvifError::BmffParseFailed);
                }
            }
        }

        // TODO: clli, pasp, clap, irot, imir

        let av1C = find_av1C(color_properties).ok_or(AvifError::BmffParseFailed)?;
        self.image.info.depth = av1C.depth();
        self.image.info.yuv_format = av1C.pixel_format();
        self.image.info.chroma_sample_position = av1C.chroma_sample_position;

        if !cicp_set {
            // If cicp was not set, try to harvest it from the sequence header.
            self.harvest_cicp_from_sequence_header()?;
        }

        Ok(&self.image.info)
    }

    fn read_and_parse_item(&mut self, item_id: u32, category: usize) -> AvifResult<()> {
        if item_id == 0 {
            return Ok(());
        }
        self.items.get(&item_id).unwrap().read_and_parse(
            self.io.as_mut().unwrap(),
            &mut self.tile_info[category].grid,
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
        let mut codec = Box::new(Dav1d::default());
        codec.initialize(operating_point, all_layers)?;
        self.codecs.push(codec);
        Ok(())
    }

    fn create_codecs(&mut self) -> AvifResult<()> {
        if matches!(self.source, Source::Tracks) {
            // In this case, we will use at most two codec instances (one for the color planes and
            // one for the alpha plane). Gain maps are not supported.
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
        } else if self.can_use_single_codec() {
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
        Ok(())
    }

    fn prepare_sample(
        &mut self,
        image_index: usize,
        category: usize,
        tile_index: usize,
    ) -> AvifResult<()> {
        // TODO: this function can probably be moved into DecodeSample.data().
        // println!(
        //     "prepare sample: image_index {image_index} category {category} tile_index {tile_index}"
        // );
        let tile = &mut self.tiles[category][tile_index];
        if tile.input.samples.len() <= image_index {
            println!("sample for index {image_index} not found.");
            return Err(AvifError::NoImagesRemaining);
        }
        let sample = &mut tile.input.samples[image_index];
        // the rest of the code can be in sample struct.
        if sample.item_id != 0 {
            // Data comes from an item.
            let item = self
                .items
                .get(&sample.item_id)
                .ok_or(AvifError::BmffParseFailed)?;
            if item.extents.len() > 1 {
                // Item has multiple extents, merge them into a contiguous buffer.
                if sample.data_buffer.is_none() {
                    let mut data: Vec<u8> = Vec::new();
                    data.reserve(item.size);
                    for extent in &item.extents {
                        let io = self.io.as_mut().unwrap();
                        data.extend_from_slice(
                            io.read(extent.offset, usize_from_u64(extent.length)?)?,
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
        Ok(())
    }

    fn prepare_samples(&mut self, image_index: usize) -> AvifResult<()> {
        for category in 0usize..3 {
            for tile_index in 0..self.tiles[category].len() {
                self.prepare_sample(image_index, category, tile_index)?;
            }
        }
        Ok(())
    }

    fn decode_tiles(&mut self, image_index: usize) -> AvifResult<()> {
        for category in 0usize..3 {
            let grid = &self.tile_info[category].grid;
            let is_grid = grid.rows > 0 && grid.columns > 0;
            if is_grid {
                if category == 2 {
                    self.gainmap.image.allocate_planes(category)?;
                } else {
                    self.image.allocate_planes(category)?;
                }
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
                    // TODO: make sure all tiles decoded properties match. Need to figure out a way
                    // to do it with proper borrows.
                    if category == 2 {
                        self.gainmap.image.copy_from_tile(
                            &tile.image,
                            &self.tile_info[category],
                            tile_index as u32,
                            category,
                        )?;
                    } else {
                        self.image.copy_from_tile(
                            &tile.image,
                            &self.tile_info[category],
                            tile_index as u32,
                            category,
                        )?;
                    }
                } else {
                    // Non grid path, steal planes from the only tile.
                    if category == 0 {
                        self.image.info.width = tile.image.info.width;
                        self.image.info.height = tile.image.info.height;
                        self.image.info.depth = tile.image.info.depth;
                        self.image.info.yuv_format = tile.image.info.yuv_format;
                    } else if category == 1 {
                        // check width height mismatch.
                    } else if category == 2 {
                        self.gainmap.image.info.width = tile.image.info.width;
                        self.gainmap.image.info.height = tile.image.info.height;
                        self.gainmap.image.info.depth = tile.image.info.depth;
                        self.gainmap.image.info.yuv_format = tile.image.info.yuv_format;
                    }

                    if category == 0 || category == 1 {
                        self.image.steal_from(&tile.image, category);
                    } else {
                        self.gainmap.image.steal_from(&tile.image, category);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn next_image(&mut self) -> AvifResult<&Image> {
        if self.io.is_none() {
            return Err(AvifError::IoNotSet);
        }
        if self.tiles[0].is_empty() {
            return Err(AvifError::NoContent);
        }
        let next_image_index = self.image_index + 1;
        if next_image_index == 0 {
            // TODO: this may accidentally create more when nth image is added. so make sure this
            // function is called only once.
            self.create_codecs()?;
        }
        self.prepare_samples(next_image_index as usize)?;
        self.decode_tiles(next_image_index as usize)?;
        self.image_index = next_image_index;
        // TODO provide timing info for tracks.
        Ok(&self.image)
    }

    pub fn peek_compatible_file_type(data: &[u8]) -> bool {
        mp4box::peek_compatible_file_type(data).unwrap_or(false)
    }
}
