pub mod gainmap;
pub mod item;
pub mod tile;
pub mod track;

use crate::decoder::gainmap::*;
use crate::decoder::item::*;
use crate::decoder::tile::*;
use crate::decoder::track::*;

#[cfg(feature = "dav1d")]
use crate::codecs::dav1d::Dav1d;

#[cfg(feature = "libgav1")]
use crate::codecs::libgav1::Libgav1;

#[cfg(feature = "android_mediacodec")]
use crate::codecs::android_mediacodec::MediaCodec;

use crate::image::*;
use crate::internal_utils::io::*;
use crate::internal_utils::*;
use crate::parser::exif;
use crate::parser::mp4box;
use crate::parser::mp4box::*;
use crate::parser::obu::Av1SequenceHeader;
use crate::*;

use std::cmp::max;
use std::cmp::min;

pub trait IO {
    fn read(&mut self, offset: u64, size: usize) -> AvifResult<&[u8]>;
    fn size_hint(&self) -> u64;
    fn persistent(&self) -> bool;
}

pub type GenericIO = Box<dyn IO>;
pub type Codec = Box<dyn crate::codecs::Decoder>;

#[derive(Debug, Default)]
pub enum CodecChoice {
    #[default]
    Auto,
    Dav1d,
    Libgav1,
    MediaCodec,
}

impl CodecChoice {
    #[allow(unreachable_code)]
    fn get_codec(&self) -> AvifResult<Codec> {
        match self {
            CodecChoice::Auto => {
                #[cfg(feature = "dav1d")]
                {
                    return Ok(Box::<Dav1d>::default());
                }
                #[cfg(feature = "libgav1")]
                {
                    return Ok(Box::<Libgav1>::default());
                }
                #[cfg(feature = "android_mediacodec")]
                {
                    return Ok(Box::<MediaCodec>::default());
                }
                Err(AvifError::NoCodecAvailable)
            }
            CodecChoice::Dav1d => {
                #[cfg(feature = "dav1d")]
                {
                    return Ok(Box::<Dav1d>::default());
                }
                Err(AvifError::NoCodecAvailable)
            }
            CodecChoice::Libgav1 => {
                #[cfg(feature = "libgav1")]
                {
                    return Ok(Box::<Libgav1>::default());
                }
                Err(AvifError::NoCodecAvailable)
            }
            CodecChoice::MediaCodec => {
                #[cfg(feature = "android_mediacodec")]
                {
                    return Ok(Box::<MediaCodec>::default());
                }
                Err(AvifError::NoCodecAvailable)
            }
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub enum Source {
    #[default]
    Auto = 0,
    PrimaryItem = 1,
    Tracks = 2,
    // TODO: Thumbnail,
}

pub const DEFAULT_IMAGE_SIZE_LIMIT: u32 = 16384 * 16384;
pub const DEFAULT_IMAGE_DIMENSION_LIMIT: u32 = 32768;
pub const DEFAULT_IMAGE_COUNT_LIMIT: u32 = 12 * 3600 * 60;

#[derive(Debug)]
pub struct Settings {
    pub source: Source,
    pub ignore_exif: bool,
    pub ignore_xmp: bool,
    pub strictness: Strictness,
    pub allow_progressive: bool,
    pub allow_incremental: bool,
    pub enable_decoding_gainmap: bool,
    pub enable_parsing_gainmap_metadata: bool,
    pub codec_choice: CodecChoice,
    pub image_size_limit: u32,
    pub image_dimension_limit: u32,
    pub image_count_limit: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            source: Default::default(),
            ignore_exif: false,
            ignore_xmp: false,
            strictness: Default::default(),
            allow_progressive: false,
            allow_incremental: false,
            enable_decoding_gainmap: false,
            enable_parsing_gainmap_metadata: false,
            codec_choice: Default::default(),
            image_size_limit: DEFAULT_IMAGE_SIZE_LIMIT,
            image_dimension_limit: DEFAULT_IMAGE_DIMENSION_LIMIT,
            image_count_limit: DEFAULT_IMAGE_COUNT_LIMIT,
        }
    }
}

#[derive(Debug, Default, Copy, Clone)]
#[repr(C)]
pub struct Extent {
    pub offset: u64,
    pub size: usize,
}

impl Extent {
    fn merge(&mut self, extent: &Extent) -> AvifResult<()> {
        if self.size == 0 {
            *self = *extent;
            return Ok(());
        }
        if extent.size == 0 {
            return Ok(());
        }
        let max_extent_1 = self.offset + u64_from_usize(self.size)?;
        let max_extent_2 = extent.offset + u64_from_usize(extent.size)?;
        self.offset = min(self.offset, extent.offset);
        self.size = usize_from_u64(max(max_extent_1, max_extent_2) - self.offset)?;
        Ok(())
    }
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

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub enum ProgressiveState {
    #[default]
    Unavailable = 0,
    Available = 1,
    Active = 2,
}

#[derive(Default, PartialEq)]
enum ParseState {
    #[default]
    None,
    AwaitingSequenceHeader,
    Complete,
}

#[derive(Default)]
pub struct Decoder {
    pub settings: Settings,
    pub image_count: u32,
    pub image_index: i32,
    pub image_timing: ImageTiming,
    pub timescale: u64,
    pub duration_in_timescales: u64,
    pub duration: f64,
    pub repetition_count: RepetitionCount,
    pub gainmap: GainMap,
    pub gainmap_present: bool,
    image: Image,
    source: Source,
    tile_info: [TileInfo; 3],
    tiles: [Vec<Tile>; 3],
    items: Items,
    tracks: Vec<Track>,
    // To replicate the C-API, we need to keep this optional. Otherwise this
    // could be part of the initialization.
    io: Option<GenericIO>,
    codecs: Vec<Codec>,
    color_track_id: Option<u32>,
    parse_state: ParseState,
}

impl Decoder {
    fn parsing_complete(&self) -> bool {
        self.parse_state == ParseState::Complete
    }

    pub fn set_io_file(&mut self, filename: &String) -> AvifResult<()> {
        self.io = Some(Box::new(DecoderFileIO::create(filename)?));
        self.parse_state = ParseState::None;
        Ok(())
    }

    // This has an unsafe block and is intended for use only from the C API.
    pub fn set_io_raw(&mut self, data: *const u8, size: usize) -> AvifResult<()> {
        self.io = Some(Box::new(DecoderRawIO::create(data, size)));
        self.parse_state = ParseState::None;
        Ok(())
    }

    pub fn set_io(&mut self, io: GenericIO) {
        self.io = Some(io);
        self.parse_state = ParseState::None;
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
                    //println!("alpha aux item was not found for color tile.");
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
        let tmap_items: Vec<_> = self.items.values().filter(|x| x.is_tmap()).collect();
        for item in tmap_items {
            let dimg_items: Vec<_> = self
                .items
                .values()
                .filter(|x| x.dimg_for_id == item.id)
                .collect();
            if dimg_items.len() != 2 {
                println!("Expected tmap to have 2 dimg items");
                return Err(AvifError::InvalidToneMappedImage);
            }
            let item0 = if dimg_items[0].dimg_index == 0 { dimg_items[0] } else { dimg_items[1] };
            if item0.id != color_item_id {
                continue;
            }
            let item1 = if dimg_items[0].dimg_index == 0 { dimg_items[1] } else { dimg_items[0] };
            return Ok((item.id, item1.id));
        }
        Ok((0, 0))
    }

    fn find_gainmap_item(&self, color_item_id: u32) -> AvifResult<(u32, u32)> {
        let (tonemap_id, gainmap_id) = self.find_tone_mapped_image_item(color_item_id)?;
        if tonemap_id == 0 || gainmap_id == 0 {
            return Ok((0, 0));
        }
        let gainmap_item = self
            .items
            .get(&gainmap_id)
            .ok_or(AvifError::InvalidToneMappedImage)?;
        if gainmap_item.should_skip() {
            return Err(AvifError::InvalidToneMappedImage);
        }
        Ok((tonemap_id, gainmap_id))
    }

    fn validate_gainmap_item(&mut self, gainmap_id: u32, tonemap_id: u32) -> AvifResult<()> {
        let gainmap_item = self
            .items
            .get(&gainmap_id)
            .ok_or(AvifError::InvalidToneMappedImage)?;
        if let Ok(nclx) = find_nclx(&gainmap_item.properties) {
            println!("found nclx: {:#?}", nclx);
            self.gainmap.image.color_primaries = nclx.color_primaries;
            self.gainmap.image.transfer_characteristics = nclx.transfer_characteristics;
            self.gainmap.image.matrix_coefficients = nclx.matrix_coefficients;
            self.gainmap.image.full_range = nclx.full_range;
        }
        if tonemap_id == 0 {
            return Ok(());
        }
        // Find and adopt all colr boxes "at most one for a given value of colour type"
        // (HEIF 6.5.5.1, from Amendment 3) Accept one of each type, and bail out if more than one
        // of a given type is provided.
        let tonemap_item = self
            .items
            .get(&tonemap_id)
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
                    .exif
                    .extend_from_slice(stream.get_slice(stream.bytes_left())?);
            }
        }
        if !self.settings.ignore_xmp {
            if let Some(xmp) = self.items.iter().find(|x| x.1.is_xmp(color_item_index)) {
                let mut stream = xmp.1.stream(self.io.as_mut().unwrap())?;
                self.image
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
            if !self.tile_info[category].is_grid() {
                println!("multiple dimg items were found but image is not grid.");
                return Err(AvifError::InvalidImageGrid);
            }
            let grid_item_ids = item.grid_item_ids.clone();
            for grid_item_id in &grid_item_ids {
                let grid_item = self
                    .items
                    .get_mut(grid_item_id)
                    .ok_or(AvifError::InvalidImageGrid)?;
                let mut tile = Tile::create_from_item(
                    grid_item,
                    self.settings.allow_progressive,
                    self.settings.image_count_limit,
                )?;
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
            let mut tile = Tile::create_from_item(
                item,
                self.settings.allow_progressive,
                self.settings.image_count_limit,
            )?;
            tile.input.category = category as u8;
            tiles.push(tile);
        }
        self.tile_info[category].tile_count = u32_from_usize(tiles.len())?;
        Ok(tiles)
    }

    fn harvest_cicp_from_sequence_header(&mut self) -> AvifResult<()> {
        if self.tiles[0].is_empty() {
            return Ok(());
        }
        // TODO: This will read the entire first sample if there are multiple extents. Might want
        // to fix that.
        self.prepare_sample(0, 0, 0)?;
        let io = &mut self.io.as_mut().unwrap();
        let sample = &self.tiles[0][0].input.samples[0];
        let item_data_buffer = if sample.item_id == 0 {
            &None
        } else {
            &self.items.get(&sample.item_id).unwrap().data_buffer
        };
        let mut search_size = 64;
        while search_size < 4096 {
            match Av1SequenceHeader::parse_from_obus(sample.partial_data(
                io,
                item_data_buffer,
                search_size,
            )?) {
                Ok(sequence_header) => {
                    self.image.color_primaries = sequence_header.color_primaries;
                    self.image.transfer_characteristics = sequence_header.transfer_characteristics;
                    self.image.matrix_coefficients = sequence_header.matrix_coefficients;
                    self.image.full_range = sequence_header.full_range;
                    break;
                }
                Err(_) => {
                    println!("errored :(");
                }
            }
            search_size += 64;
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
        if grid_item_ids.len() as u32 != self.tile_info[category].grid_tile_count() {
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

    fn reset(&mut self) {
        let decoder = Decoder::default();
        // Reset all fields to default except the following: settings, io, source.
        self.image_count = decoder.image_count;
        self.image_timing = decoder.image_timing;
        self.timescale = decoder.timescale;
        self.duration_in_timescales = decoder.duration_in_timescales;
        self.duration = decoder.duration;
        self.repetition_count = decoder.repetition_count;
        self.gainmap = decoder.gainmap;
        self.gainmap_present = decoder.gainmap_present;
        self.image = decoder.image;
        self.tile_info = decoder.tile_info;
        self.tiles = decoder.tiles;
        self.image_index = decoder.image_index;
        self.items = decoder.items;
        self.tracks = decoder.tracks;
        self.codecs = decoder.codecs;
        self.color_track_id = decoder.color_track_id;
        self.parse_state = decoder.parse_state;
    }

    #[allow(non_snake_case)]
    pub fn parse(&mut self) -> AvifResult<()> {
        if self.parsing_complete() {
            // Parse was called again. Reset the data and start over.
            self.parse_state = ParseState::None;
        }
        if self.io.is_none() {
            return Err(AvifError::IoNotSet);
        }
        if self.parse_state == ParseState::None {
            self.reset();
            let avif_boxes = mp4box::parse(self.io.as_mut().unwrap())?;
            self.tracks = avif_boxes.tracks;
            if !self.tracks.is_empty() {
                self.image.image_sequence_track_present = true;
                for track in &self.tracks {
                    if !track.check_limits(
                        self.settings.image_size_limit,
                        self.settings.image_dimension_limit,
                    ) {
                        println!("track dimension too large");
                        return Err(AvifError::BmffParseFailed);
                    }
                }
            }
            self.items = construct_items(&avif_boxes.meta)?;
            for item in self.items.values_mut() {
                item.harvest_ispe(
                    self.settings.strictness.alpha_ispe_required(),
                    self.settings.image_size_limit,
                    self.settings.image_dimension_limit,
                )?;
            }
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
                    self.color_track_id = Some(color_track.id);
                    color_properties = color_track
                        .get_properties()
                        .ok_or(AvifError::BmffParseFailed)?;

                    // TODO: exif/xmp from meta.

                    self.tiles[0].push(Tile::create_from_track(
                        color_track,
                        self.settings.image_count_limit,
                    )?);
                    self.tile_info[0].tile_count = 1;

                    if let Some(alpha_track) = self.tracks.iter().find(|x| x.is_aux(color_track.id))
                    {
                        self.tiles[1].push(Tile::create_from_track(
                            alpha_track,
                            self.settings.image_count_limit,
                        )?);
                        self.tile_info[1].tile_count = 1;
                        self.image.alpha_present = true;
                        self.image.alpha_premultiplied = color_track.prem_by_id == alpha_track.id;
                    }

                    self.image_index = -1;
                    self.image_count = self.tiles[0][0].input.samples.len() as u32;
                    self.timescale = color_track.media_timescale as u64;
                    self.duration_in_timescales = color_track.media_duration;
                    if self.timescale != 0 {
                        self.duration =
                            (self.duration_in_timescales as f64) / (self.timescale as f64);
                    } else {
                        self.duration = 0.0;
                    }
                    self.repetition_count = color_track.repetition_count()?;
                    self.image_timing = Default::default();

                    self.image.width = color_track.width;
                    self.image.height = color_track.height;
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
                        self.validate_gainmap_item(gainmap_id, tonemap_id)?;
                        self.gainmap_present = true;
                        if self.settings.enable_decoding_gainmap {
                            item_ids[2] = gainmap_id;
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
                        //println!("gainmap: {:#?}", self.gainmap);
                    }

                    //println!("item ids: {:#?}", item_ids);

                    self.search_exif_or_xmp_metadata(item_ids[0])?;

                    self.image_index = -1;
                    self.image_count = 1;
                    self.timescale = 1;
                    self.duration_in_timescales = 1;
                    self.image_timing.timescale = 1;
                    self.image_timing.duration = 1.0;
                    self.image_timing.duration_in_timescales = 1;

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

                    let color_item = self.items.get(&item_ids[0]).unwrap();
                    self.image.width = color_item.width;
                    self.image.height = color_item.height;
                    self.image.alpha_present = item_ids[1] != 0;
                    // alphapremultiplied.

                    if color_item.progressive {
                        self.image.progressive_state = ProgressiveState::Available;
                        let sample_count = self.tiles[0][0].input.samples.len();
                        if sample_count > 1 {
                            self.image.progressive_state = ProgressiveState::Active;
                            self.image_count = sample_count as u32;
                        }
                    }

                    if item_ids[2] != 0 {
                        let gainmap_item = self.items.get(&item_ids[2]).unwrap();
                        self.gainmap.image.width = gainmap_item.width;
                        self.gainmap.image.height = gainmap_item.height;
                        let av1C = gainmap_item.av1C().ok_or(AvifError::BmffParseFailed)?;
                        self.gainmap.image.depth = av1C.depth();
                        self.gainmap.image.yuv_format = av1C.pixel_format();
                        self.gainmap.image.chroma_sample_position = av1C.chroma_sample_position;
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
                    self.image.color_primaries = nclx.color_primaries;
                    self.image.transfer_characteristics = nclx.transfer_characteristics;
                    self.image.matrix_coefficients = nclx.matrix_coefficients;
                    self.image.full_range = nclx.full_range;
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
                    self.image.icc = icc;
                }
                Err(multiple_icc_found) => {
                    if multiple_icc_found {
                        println!("multiple icc were found");
                        return Err(AvifError::BmffParseFailed);
                    }
                }
            }

            self.image.clli = find_clli(color_properties);
            self.image.pasp = find_pasp(color_properties);
            self.image.clap = find_clap(color_properties);
            self.image.irot_angle = find_irot_angle(color_properties);
            self.image.imir_axis = find_imir_axis(color_properties);

            let av1C = find_av1C(color_properties).ok_or(AvifError::BmffParseFailed)?;
            self.image.depth = av1C.depth();
            self.image.yuv_format = av1C.pixel_format();
            self.image.chroma_sample_position = av1C.chroma_sample_position;

            if cicp_set {
                self.parse_state = ParseState::Complete;
                return Ok(());
            }
            self.parse_state = ParseState::AwaitingSequenceHeader;
        }

        // If cicp was not set, try to harvest it from the sequence header.
        self.harvest_cicp_from_sequence_header()?;
        self.parse_state = ParseState::Complete;

        Ok(())
    }

    fn read_and_parse_item(&mut self, item_id: u32, category: usize) -> AvifResult<()> {
        if item_id == 0 {
            return Ok(());
        }
        self.items.get(&item_id).unwrap().read_and_parse(
            self.io.as_mut().unwrap(),
            &mut self.tile_info[category].grid,
            self.settings.image_size_limit,
            self.settings.image_dimension_limit,
        )
    }

    #[allow(unreachable_code)]
    fn can_use_single_codec(&self) -> bool {
        #[cfg(feature = "android_mediacodec")]
        {
            // Android MediaCodec does not support using a single codec
            // instance for images of varying formats (which could happen
            // when image contains alpha).
            return false;
        }
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
        let mut codec: Codec = self.settings.codec_choice.get_codec()?;
        codec.initialize(operating_point, all_layers)?;
        self.codecs.push(codec);
        Ok(())
    }

    fn create_codecs(&mut self) -> AvifResult<()> {
        if !self.codecs.is_empty() {
            return Ok(());
        }
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
        let tile = &mut self.tiles[category][tile_index];
        if tile.input.samples.len() <= image_index {
            println!("sample for index {image_index} not found.");
            return Err(AvifError::NoImagesRemaining);
        }
        let sample = &tile.input.samples[image_index];
        if sample.item_id == 0 {
            // Data comes from a track. Nothing to prepare.
            return Ok(());
        }
        // Data comes from an item.
        let item = self
            .items
            .get_mut(&sample.item_id)
            .ok_or(AvifError::BmffParseFailed)?;
        if item.extents.len() == 1 {
            // Item has only one extent. Nothing to prepare.
            return Ok(());
        }
        if item.data_buffer.is_some() {
            // Extents have already been merged.
            return Ok(());
        }
        // Item has multiple extents, merge them into a contiguous buffer.
        let mut data: Vec<u8> = Vec::new();
        data.reserve(item.size);
        for extent in &item.extents {
            let io = self.io.as_mut().unwrap();
            // TODO: check if enough bytes were actually read.
            data.extend_from_slice(io.read(extent.offset, extent.size)?);
        }
        item.data_buffer = Some(data);
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
            let is_grid = self.tile_info[category].is_grid();
            if is_grid {
                if category == 2 {
                    self.gainmap.image.allocate_planes(category)?;
                } else {
                    self.image.allocate_planes(category)?;
                }
            }
            let previous_decoded_tile_count = self.tile_info[category].decoded_tile_count;
            for tile_index in previous_decoded_tile_count as usize..self.tiles[category].len() {
                let tile = &mut self.tiles[category][tile_index];
                let sample = &tile.input.samples[image_index];
                let io = &mut self.io.as_mut().unwrap();

                let codec = &mut self.codecs[tile.codec_index];
                let item_data_buffer = if sample.item_id == 0 {
                    &None
                } else {
                    &self.items.get(&sample.item_id).unwrap().data_buffer
                };
                let data = sample.data(io, item_data_buffer)?;
                codec.get_next_image(data, sample.spatial_id, &mut tile.image, category)?;
                self.tile_info[category].decoded_tile_count += 1;

                if is_grid {
                    if category == 1 && !tile.image.full_range {
                        tile.image.alpha_to_full_range()?;
                    }
                    tile.image.scale(tile.width, tile.height)?;
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
                        self.image.width = tile.image.width;
                        self.image.height = tile.image.height;
                        self.image.depth = tile.image.depth;
                        self.image.yuv_format = tile.image.yuv_format;
                    } else if category == 1 {
                        // check width height mismatch.
                    } else if category == 2 {
                        self.gainmap.image.width = tile.image.width;
                        self.gainmap.image.height = tile.image.height;
                        self.gainmap.image.depth = tile.image.depth;
                        self.gainmap.image.yuv_format = tile.image.yuv_format;
                    }

                    if category == 0 || category == 1 {
                        self.image.steal_from(&tile.image, category);
                        // TODO: These likely may not work with android mediacodec since it does
                        // not use pointer.
                        if category == 1 && !tile.image.full_range {
                            self.image.alpha_to_full_range()?;
                        }
                        self.image.scale(tile.width, tile.height)?;
                    } else {
                        self.gainmap.image.steal_from(&tile.image, category);
                        self.gainmap.image.scale(tile.width, tile.height)?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn next_image(&mut self) -> AvifResult<()> {
        if self.io.is_none() {
            return Err(AvifError::IoNotSet);
        }
        if !self.parsing_complete() {
            return Err(AvifError::NoContent);
        }
        if self.is_current_frame_fully_decoded() {
            for category in 0usize..3 {
                self.tile_info[category].decoded_tile_count = 0;
            }
        }

        let next_image_index = self.image_index + 1;
        self.create_codecs()?;
        self.prepare_samples(next_image_index as usize)?;
        self.decode_tiles(next_image_index as usize)?;
        self.image_index = next_image_index;
        self.image_timing = self.nth_image_timing(self.image_index as u32)?;
        Ok(())
    }

    fn is_current_frame_fully_decoded(&self) -> bool {
        if !self.parsing_complete() {
            return false;
        }
        for category in 0usize..3 {
            if !self.tile_info[category].is_fully_decoded() {
                return false;
            }
        }
        true
    }

    pub fn nth_image(&mut self, index: u32) -> AvifResult<()> {
        if !self.parsing_complete() {
            return Err(AvifError::NoContent);
        }
        if index >= self.image_count {
            return Err(AvifError::NoImagesRemaining);
        }
        let requested_index = i32_from_u32(index)?;
        if requested_index == (self.image_index + 1) {
            return self.next_image();
        }
        if requested_index == self.image_index && self.is_current_frame_fully_decoded() {
            // Current frame which is already fully decoded has been requested. Do nothing.
            return Ok(());
        }
        let nearest_keyframe = i32_from_u32(self.nearest_keyframe(index))?;
        if nearest_keyframe > (self.image_index + 1) || requested_index <= self.image_index {
            // Start decoding from the nearest keyframe.
            self.image_index = nearest_keyframe - 1;
        }
        loop {
            self.next_image()?;
            if requested_index == self.image_index {
                break;
            }
        }
        Ok(())
    }

    pub fn image(&self) -> &Image {
        // TODO: make this optional and reutrn none if parsing is not complete.
        &self.image
    }

    pub fn nth_image_timing(&self, n: u32) -> AvifResult<ImageTiming> {
        if !self.parsing_complete() {
            return Err(AvifError::NoContent);
        }
        if n > self.settings.image_count_limit {
            return Err(AvifError::NoImagesRemaining);
        }
        if self.color_track_id.is_none() {
            return Ok(self.image_timing);
        }
        let color_track_id = self.color_track_id.unwrap();
        let color_track = self
            .tracks
            .iter()
            .find(|x| x.id == color_track_id)
            .ok_or(AvifError::NoContent)?;
        color_track.image_timing(n)
    }

    pub fn decoded_row_count(&self) -> u32 {
        let mut min_row_count = self.image.height;
        for category in 0usize..3 {
            if self.tiles[category].is_empty() {
                continue;
            }
            if category == 2 {
                // TODO: handle gainmap.
            }
            let first_tile_height = self.tiles[category][0].height;
            let row_count =
                self.tile_info[category].decoded_row_count(self.image.height, first_tile_height);
            min_row_count = std::cmp::min(min_row_count, row_count);
        }
        min_row_count
    }

    pub fn is_keyframe(&self, index: u32) -> bool {
        if !self.parsing_complete() {
            return false;
        }
        let index = index as usize;
        // All the tiles for the requested index must be a keyframe.
        for category in 0usize..3 {
            for tile in &self.tiles[category] {
                if index >= tile.input.samples.len() || !tile.input.samples[index].sync {
                    return false;
                }
            }
        }
        true
    }

    pub fn nearest_keyframe(&self, index: u32) -> u32 {
        if !self.parsing_complete() {
            return 0;
        }
        for i in (0..index).rev() {
            if self.is_keyframe(i) {
                return i;
            }
        }
        0
    }

    pub fn nth_image_max_extent(&self, index: u32) -> AvifResult<Extent> {
        if !self.parsing_complete() {
            return Err(AvifError::NoContent);
        }
        let mut extent = Extent::default();
        let start_index = self.nearest_keyframe(index) as usize;
        let end_index = index as usize;
        for current_index in start_index..=end_index {
            for category in 0usize..3 {
                for tile in &self.tiles[category] {
                    if current_index >= tile.input.samples.len() {
                        return Err(AvifError::NoImagesRemaining);
                    }
                    let sample = &tile.input.samples[current_index];
                    let sample_extent = if sample.item_id != 0 {
                        let item = self.items.get(&sample.item_id).unwrap();
                        item.max_extent(sample)?
                    } else {
                        Extent {
                            offset: sample.offset,
                            size: sample.size,
                        }
                    };
                    extent.merge(&sample_extent)?;
                }
            }
        }
        Ok(extent)
    }

    pub fn peek_compatible_file_type(data: &[u8]) -> bool {
        mp4box::peek_compatible_file_type(data).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    #[test_case(10, 20, 50, 100, 10, 140 ; "case 1")]
    #[test_case(100, 20, 50, 100, 50, 100 ; "case 2")]
    fn merge_extents(
        offset1: u64,
        size1: usize,
        offset2: u64,
        size2: usize,
        expected_offset: u64,
        expected_size: usize,
    ) {
        let mut e1 = Extent {
            offset: offset1,
            size: size1,
        };
        let e2 = Extent {
            offset: offset2,
            size: size2,
        };
        assert!(e1.merge(&e2).is_ok());
        assert_eq!(e1.offset, expected_offset);
        assert_eq!(e1.size, expected_size);
    }
}
