use std::collections::HashMap;

use crate::decoder::*;
use crate::io::*;
use crate::parser::mp4box::*;
use crate::*;

#[derive(Debug, Default)]
pub struct Item {
    pub id: u32,
    pub item_type: String,
    pub size: usize,
    pub width: u32,
    pub height: u32,
    pub content_type: String,
    pub properties: Vec<ItemProperty>,
    pub extents: Vec<ItemLocationExtent>,
    // TODO mergedExtents stuff.
    pub thumbnail_for_id: u32,
    pub aux_for_id: u32,
    pub desc_for_id: u32,
    pub dimg_for_id: u32,
    pub dimg_index: u32,
    pub prem_by_id: u32,
    pub has_unsupported_essential_property: bool,
    pub progressive: bool,
    pub idat: Vec<u8>,
    pub grid_item_ids: Vec<u32>,
}

macro_rules! find_property {
    ($self:ident, $a:ident) => {
        $self
            .properties
            .iter()
            .find(|x| matches!(x, ItemProperty::$a(_)))
    };
}

impl Item {
    pub fn data_offset(&self) -> u64 {
        self.extents[0].offset
    }

    pub fn stream<'a>(&'a self, io: &'a mut Box<dyn DecoderIO>) -> AvifResult<IStream> {
        // TODO: handle multiple extents.
        let io_data = match self.idat.is_empty() {
            true => io.read(self.data_offset(), self.size)?,
            false => {
                // TODO: assumes idat offset is 0.
                self.idat.as_slice()
            }
        };
        Ok(IStream::create(io_data))
    }

    pub fn read_and_parse(&self, io: &mut Box<dyn DecoderIO>, grid: &mut Grid) -> AvifResult<()> {
        // TODO: this function also has to extract codec type.
        if self.item_type != "grid" {
            return Ok(());
        }
        let mut stream = self.stream(io)?;
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

    pub fn operating_point(&self) -> u8 {
        match find_property!(self, OperatingPointSelector) {
            Some(ItemProperty::OperatingPointSelector(operating_point)) => *operating_point,
            _ => 0, // default operating point.
        }
    }

    pub fn harvest_ispe(&mut self, alpha_ispe_required: bool) -> AvifResult<()> {
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

    #[allow(non_snake_case)]
    pub fn validate_properties(
        &self,
        avif_items: &HashMap<u32, Item>,
        pixi_required: bool,
    ) -> AvifResult<()> {
        println!("validating item: {:#?}", self);
        let av1C = self.av1C().ok_or(AvifError::BmffParseFailed)?;
        if self.item_type == "grid" {
            for grid_item_id in &self.grid_item_ids {
                let grid_item = avif_items.get(&grid_item_id).unwrap();
                let grid_av1C = grid_item.av1C().ok_or(AvifError::BmffParseFailed)?;
                if av1C != grid_av1C {
                    println!("av1c of grid items do not match");
                    return Err(AvifError::BmffParseFailed);
                }
            }
        }
        match self.pixi() {
            Some(pixi) => {
                for i in 0..pixi.plane_count as usize {
                    if pixi.plane_depths[i] != av1C.depth() {
                        println!("pixi depth does not match av1C depth");
                        return Err(AvifError::BmffParseFailed);
                    }
                }
            }
            None => {
                if pixi_required {
                    println!("missing pixi property");
                    return Err(AvifError::BmffParseFailed);
                }
            }
        }
        Ok(())
    }

    #[allow(non_snake_case)]
    pub fn av1C(&self) -> Option<&CodecConfiguration> {
        match find_property!(self, CodecConfiguration) {
            Some(ItemProperty::CodecConfiguration(av1C)) => Some(av1C),
            _ => None,
        }
    }

    pub fn pixi(&self) -> Option<&PixelInformation> {
        match find_property!(self, PixelInformation) {
            Some(ItemProperty::PixelInformation(pixi)) => Some(pixi),
            _ => None,
        }
    }

    pub fn a1lx(&self) -> Option<&[usize; 3]> {
        match find_property!(self, AV1LayeredImageIndexing) {
            Some(ItemProperty::AV1LayeredImageIndexing(a1lx)) => Some(a1lx),
            _ => None,
        }
    }

    pub fn lsel(&self) -> Option<u16> {
        match find_property!(self, LayerSelector) {
            Some(ItemProperty::LayerSelector(lsel)) => Some(*lsel),
            _ => None,
        }
    }

    #[allow(non_snake_case)]
    pub fn is_auxiliary_alpha(&self) -> bool {
        match find_property!(self, AuxiliaryType) {
            Some(ItemProperty::AuxiliaryType(aux_type)) => {
                aux_type == "urn:mpeg:mpegB:cicp:systems:auxiliary:alpha"
                    || aux_type == "urn:mpeg:hevc:2015:auxid:1"
            }
            _ => false,
        }
    }

    pub fn should_skip(&self) -> bool {
        self.size == 0
            || self.has_unsupported_essential_property
            || (self.item_type != "av01" && self.item_type != "grid")
            || self.thumbnail_for_id != 0
    }

    pub fn is_metadata(&self, item_type: &str, color_id: u32) -> bool {
        self.size != 0
            && !self.has_unsupported_essential_property
            && (color_id == 0 || self.desc_for_id == color_id)
            && self.item_type == *item_type
    }

    pub fn is_exif(&self, color_id: u32) -> bool {
        self.is_metadata("Exif", color_id)
    }

    pub fn is_xmp(&self, color_id: u32) -> bool {
        self.is_metadata("mime", color_id) && self.content_type == "application/rdf+xml"
    }

    pub fn is_tmap(&self) -> bool {
        self.is_metadata("tmap", 0) && self.thumbnail_for_id == 0
    }
}
