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
    ($properties:expr, $property_name:ident) => {
        $properties.iter().find_map(|p| match p {
            ItemProperty::$property_name(value) => Some(value),
            _ => None,
        })
    };
}

impl Item {
    pub fn data_offset(&self) -> u64 {
        self.extents[0].offset
    }

    pub fn stream<'a>(&'a self, io: &'a mut GenericIO) -> AvifResult<IStream> {
        // TODO: handle multiple extents.
        let io_data = match self.idat.is_empty() {
            true => io.read_exact(self.data_offset(), self.size)?,
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
            return Err(AvifError::InvalidImageGrid(
                "unsupported version for grid".into(),
            ));
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
            return Err(AvifError::InvalidImageGrid(
                "invalid dimensions in grid box".into(),
            ));
        }
        if !check_limits(grid.width, grid.height, size_limit, dimension_limit) {
            return Err(AvifError::InvalidImageGrid(
                "grid dimensions too large".into(),
            ));
        }
        Ok(())
    }

    pub fn operating_point(&self) -> u8 {
        match find_property!(self.properties, OperatingPointSelector) {
            Some(operating_point_selector) => *operating_point_selector,
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
        match find_property!(self.properties, ImageSpatialExtents) {
            Some(image_spatial_extents) => {
                self.width = image_spatial_extents.width;
                self.height = image_spatial_extents.height;
                if self.width == 0 || self.height == 0 {
                    return Err(AvifError::BmffParseFailed(
                        "item id has invalid size.".into(),
                    ));
                }
                if !check_limits(
                    image_spatial_extents.width,
                    image_spatial_extents.height,
                    size_limit,
                    dimension_limit,
                ) {
                    return Err(AvifError::BmffParseFailed(
                        "item dimensions too large.".into(),
                    ));
                }
            }
            None => {
                // No ispe was found.
                if self.is_auxiliary_alpha() {
                    if alpha_ispe_required {
                        return Err(AvifError::BmffParseFailed(
                            "alpha auxiliary image is missing mandatory ispe".into(),
                        ));
                    }
                } else {
                    return Err(AvifError::BmffParseFailed(
                        "item id is missing mandatory ispe property".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    #[allow(non_snake_case)]
    pub fn validate_properties(&self, items: &Items, pixi_required: bool) -> AvifResult<()> {
        let av1C = self.av1C().ok_or(AvifError::BmffParseFailed("".into()))?;
        if self.item_type == "grid" {
            for grid_item_id in &self.grid_item_ids {
                let grid_item = items.get(grid_item_id).unwrap();
                let grid_av1C = grid_item
                    .av1C()
                    .ok_or(AvifError::BmffParseFailed("".into()))?;
                if av1C != grid_av1C {
                    return Err(AvifError::BmffParseFailed(
                        "av1c of grid items do not match".into(),
                    ));
                }
            }
        }
        match self.pixi() {
            Some(pixi) => {
                for depth in &pixi.plane_depths {
                    if *depth != av1C.depth() {
                        return Err(AvifError::BmffParseFailed(
                            "pixi depth does not match av1C depth".into(),
                        ));
                    }
                }
            }
            None => {
                if pixi_required {
                    return Err(AvifError::BmffParseFailed("missing pixi property".into()));
                }
            }
        }
        Ok(())
    }

    #[allow(non_snake_case)]
    pub fn av1C(&self) -> Option<&CodecConfiguration> {
        find_property!(self.properties, CodecConfiguration)
    }

    pub fn pixi(&self) -> Option<&PixelInformation> {
        find_property!(self.properties, PixelInformation)
    }

    pub fn a1lx(&self) -> Option<&[usize; 3]> {
        find_property!(self.properties, AV1LayeredImageIndexing)
    }

    pub fn lsel(&self) -> Option<&u16> {
        find_property!(self.properties, LayerSelector)
    }

    pub fn clli(&self) -> Option<&ContentLightLevelInformation> {
        find_property!(self.properties, ContentLightLevelInformation)
    }

    #[allow(non_snake_case)]
    pub fn is_auxiliary_alpha(&self) -> bool {
        matches!(find_property!(self.properties, AuxiliaryType),
                 Some(aux_type) if aux_type == "urn:mpeg:mpegB:cicp:systems:auxiliary:alpha" ||
                                   aux_type == "urn:mpeg:hevc:2015:auxid:1")
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
                            .ok_or(AvifError::BmffParseFailed("".into()))?;
                        size -= usize_from_u64(remaining_offset)?;
                        remaining_offset = 0;
                    }
                }
                let used_extent_size = std::cmp::min(size, remaining_size);
                let end_offset = start_offset
                    .checked_add(u64_from_usize(used_extent_size)?)
                    .ok_or(AvifError::BmffParseFailed("".into()))?;
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

pub type Items = HashMap<u32, Item>;

pub fn construct_items(meta: &MetaBox) -> AvifResult<Items> {
    let mut items: Items = HashMap::with_hasher(NonRandomHasherState);
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
            .ok_or(AvifError::BmffParseFailed("".into()))?;
        if !item.extents.is_empty() {
            return Err(AvifError::BmffParseFailed(
                "item already has extents".into(),
            ));
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
                .ok_or(AvifError::BmffParseFailed("".into()))?;
        }
    }
    let mut ipma_seen: HashSet<u32> = HashSet::with_hasher(NonRandomHasherState);
    for association in &meta.iprp.associations {
        if ipma_seen.contains(&association.item_id) {
            return Err(AvifError::BmffParseFailed(
                "item has duplictate ipma.".into(),
            ));
        }
        ipma_seen.insert(association.item_id);
        let item = items
            .get_mut(&association.item_id)
            .ok_or(AvifError::BmffParseFailed("".into()))?;
        for (property_index_ref, essential_ref) in &association.associations {
            let property_index: usize = *property_index_ref as usize;
            let essential = *essential_ref;
            if property_index == 0 {
                // Not associated with any item.
                continue;
            }
            if property_index > meta.iprp.properties.len() {
                return Err(AvifError::BmffParseFailed(
                    "invalid property_index in ipma".into(),
                ));
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
                        return Err(AvifError::BmffParseFailed(
                            "invalid essential property".into(),
                        ));
                    }
                } else {
                    match property {
                        ItemProperty::OperatingPointSelector(_)
                        | ItemProperty::LayerSelector(_) => {
                            return Err(AvifError::BmffParseFailed(
                                "required essential property not marked as essential".into(),
                            ));
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
            .ok_or(AvifError::BmffParseFailed("".into()))?;
        match reference.reference_type.as_str() {
            "thmb" => item.thumbnail_for_id = reference.to_item_id,
            "auxl" => item.aux_for_id = reference.to_item_id,
            "cdsc" => item.desc_for_id = reference.to_item_id,
            "prem" => item.prem_by_id = reference.to_item_id,
            "dimg" => {
                // derived images refer in the opposite direction.
                let dimg_item = items
                    .get_mut(&reference.to_item_id)
                    .ok_or(AvifError::BmffParseFailed("".into()))?;
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
