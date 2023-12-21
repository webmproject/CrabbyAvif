use ahash::AHashMap;

use crate::decoder::*;
use crate::internal_utils::stream::*;
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
    pub extents: Vec<Extent>,
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
    pub data_buffer: Option<Vec<u8>>,
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

    pub fn stream<'a>(&'a self, io: &'a mut GenericIO) -> AvifResult<IStream> {
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

    pub fn read_and_parse(
        &self,
        io: &mut GenericIO,
        grid: &mut Grid,
        size_limit: u32,
        dimension_limit: u32,
    ) -> AvifResult<()> {
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
        if !check_limits(grid.width, grid.height, size_limit, dimension_limit) {
            println!("grid dimensions too large");
            return Err(AvifError::InvalidImageGrid);
        }
        Ok(())
    }

    pub fn operating_point(&self) -> u8 {
        match find_property!(self, OperatingPointSelector) {
            Some(ItemProperty::OperatingPointSelector(operating_point)) => *operating_point,
            _ => 0, // default operating point.
        }
    }

    pub fn harvest_ispe(
        &mut self,
        alpha_ispe_required: bool,
        size_limit: u32,
        dimension_limit: u32,
    ) -> AvifResult<()> {
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
                    if !check_limits(x.width, x.height, size_limit, dimension_limit) {
                        println!("item dimensions too large.");
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
    pub fn validate_properties(&self, items: &Items, pixi_required: bool) -> AvifResult<()> {
        //println!("validating item: {:#?}", self);
        let av1C = self.av1C().ok_or(AvifError::BmffParseFailed)?;
        if self.item_type == "grid" {
            for grid_item_id in &self.grid_item_ids {
                let grid_item = items.get(grid_item_id).unwrap();
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

    pub fn clli(&self) -> Option<&ContentLightLevelInformation> {
        match find_property!(self, ContentLightLevelInformation) {
            Some(ItemProperty::ContentLightLevelInformation(clli)) => Some(clli),
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

    pub fn max_extent(&self, sample: &DecodeSample) -> AvifResult<Extent> {
        if self.extents.is_empty() {
            return Err(AvifError::TruncatedData);
        }
        if !self.idat.is_empty() {
            return Ok(Extent::default());
        }
        if sample.size == 0 {
            return Err(AvifError::TruncatedData);
        }
        let mut remaining_offset = sample.offset;
        let mut remaining_size = sample.size;
        let mut min_offset = u64::MAX;
        let mut max_offset = 0;
        if self.extents.len() > 1 {
            for extent in &self.extents {
                let mut start_offset = extent.offset;
                let mut size = extent.size;
                let sizeu64 = u64_from_usize(size)?;
                if remaining_offset != 0 {
                    if remaining_offset >= sizeu64 {
                        remaining_offset -= sizeu64;
                        continue;
                    } else {
                        start_offset = start_offset
                            .checked_add(remaining_offset)
                            .ok_or(AvifError::BmffParseFailed)?;
                        size -= usize_from_u64(remaining_offset)?;
                        remaining_offset = 0;
                    }
                }
                let used_extent_size = std::cmp::min(size, remaining_size);
                let end_offset = start_offset
                    .checked_add(u64_from_usize(used_extent_size)?)
                    .ok_or(AvifError::BmffParseFailed)?;
                min_offset = std::cmp::min(min_offset, start_offset);
                max_offset = std::cmp::max(max_offset, end_offset);
                remaining_size -= used_extent_size;
                if remaining_size == 0 {
                    break;
                }
            }
        } else {
            min_offset = sample.offset;
            max_offset = sample.offset + u64_from_usize(sample.size)?;
            remaining_size = 0;
        }
        if remaining_size != 0 {
            return Err(AvifError::TruncatedData);
        }
        Ok(Extent {
            offset: min_offset,
            size: usize_from_u64(max_offset - min_offset)?,
        })
    }
}

pub type Items = AHashMap<u32, Item>;

pub fn construct_items(meta: &MetaBox) -> AvifResult<Items> {
    let mut items: Items = AHashMap::new();
    for iinf in &meta.iinf {
        items.insert(
            iinf.item_id,
            Item {
                id: iinf.item_id,
                item_type: iinf.item_type.clone(),
                content_type: iinf.content_type.clone(),
                ..Item::default()
            },
        );
    }
    for iloc in &meta.iloc.items {
        let item = items
            .get_mut(&iloc.item_id)
            .ok_or(AvifError::BmffParseFailed)?;
        if !item.extents.is_empty() {
            println!("item already has extents.");
            return Err(AvifError::BmffParseFailed);
        }
        if iloc.construction_method == 1 {
            item.idat = meta.idat.clone();
        }
        for extent in &iloc.extents {
            item.extents.push(Extent {
                offset: iloc.base_offset + extent.offset,
                size: extent.size,
            });
            item.size = item
                .size
                .checked_add(extent.size)
                .ok_or(AvifError::BmffParseFailed)?;
        }
    }
    let mut ipma_seen: AHashSet<u32> = AHashSet::new();
    for association in &meta.iprp.associations {
        if ipma_seen.contains(&association.item_id) {
            println!("item has duplictate ipma.");
            return Err(AvifError::BmffParseFailed);
        }
        ipma_seen.insert(association.item_id);
        let item = items
            .get_mut(&association.item_id)
            .ok_or(AvifError::BmffParseFailed)?;
        for (property_index_ref, essential_ref) in &association.associations {
            let property_index: usize = *property_index_ref as usize;
            let essential = *essential_ref;
            if property_index == 0 {
                // Not associated with any item.
                continue;
            }
            if property_index > meta.iprp.properties.len() {
                println!("invalid property_index in ipma.");
                return Err(AvifError::BmffParseFailed);
            }
            // property_index is 1-indexed.
            let property = meta.iprp.properties[property_index - 1].clone();
            let is_supported_property = matches!(
                property,
                ItemProperty::ImageSpatialExtents(_)
                    | ItemProperty::ColorInformation(_)
                    | ItemProperty::CodecConfiguration(_)
                    | ItemProperty::PixelInformation(_)
                    | ItemProperty::PixelAspectRatio(_)
                    | ItemProperty::AuxiliaryType(_)
                    | ItemProperty::CleanAperture(_)
                    | ItemProperty::ImageRotation(_)
                    | ItemProperty::ImageMirror(_)
                    | ItemProperty::OperatingPointSelector(_)
                    | ItemProperty::LayerSelector(_)
                    | ItemProperty::AV1LayeredImageIndexing(_)
                    | ItemProperty::ContentLightLevelInformation(_)
            );
            if is_supported_property {
                if essential {
                    if let ItemProperty::AV1LayeredImageIndexing(_) = property {
                        println!("invalid essential property.");
                        return Err(AvifError::BmffParseFailed);
                    }
                } else {
                    match property {
                        ItemProperty::OperatingPointSelector(_)
                        | ItemProperty::LayerSelector(_) => {
                            println!("required essential property not marked as essential.");
                            return Err(AvifError::BmffParseFailed);
                        }
                        _ => {}
                    }
                }
                item.properties.push(property);
            } else if essential {
                item.has_unsupported_essential_property = true;
            }
        }
    }
    for reference in &meta.iref {
        let item = items
            .get_mut(&reference.from_item_id)
            .ok_or(AvifError::BmffParseFailed)?;
        match reference.reference_type.as_str() {
            "thmb" => item.thumbnail_for_id = reference.to_item_id,
            "auxl" => item.aux_for_id = reference.to_item_id,
            "cdsc" => item.desc_for_id = reference.to_item_id,
            "prem" => item.prem_by_id = reference.to_item_id,
            "dimg" => {
                // derived images refer in the opposite direction.
                let dimg_item = items
                    .get_mut(&reference.to_item_id)
                    .ok_or(AvifError::BmffParseFailed)?;
                dimg_item.dimg_for_id = reference.from_item_id;
                dimg_item.dimg_index = reference.index;
            }
            _ => {
                // unknown reference type, ignore.
            }
        }
    }
    Ok(items)
}
