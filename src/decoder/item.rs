// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::decoder::*;
use crate::internal_utils::stream::*;
use crate::parser::mp4box::*;
use crate::utils::pixels::ChannelIdc;
use crate::*;

use std::collections::BTreeMap;
use std::num::NonZero;

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
    // Item ids of source items of a derived image item, in the same order as
    // they appear in the `dimg` box. E.g. item ids for the cells of a grid
    // item, or for the layers of an overlay item.
    pub source_item_ids: Vec<u32>,
    pub data_buffer: Option<Vec<u8>>,
    pub is_made_up: bool, // Placeholder grid alpha item if true.
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
    pub(crate) fn stream<'a>(&'a mut self, io: &'a mut GenericIO) -> AvifResult<IStream<'a>> {
        if !self.idat.is_empty() {
            match self.extents.len() {
                0 => return AvifError::unknown_error("no extent"),
                1 => {
                    let idat = self.idat.as_slice();
                    let offset = usize_from_u64(self.extents[0].offset)?;
                    let range = offset..checked_add!(offset, self.size)?;
                    check_slice_range(idat.len(), &range)?;
                    return Ok(IStream::create(&idat[range]));
                }
                _ => {
                    return AvifError::unknown_error("idat with multiple extents is not supported");
                }
            }
        }

        let io_data = match self.extents.len() {
            0 => return AvifError::unknown_error("no extent"),
            1 => io.read_exact(self.extents[0].offset, self.size)?,
            _ => {
                if self.data_buffer.is_none() {
                    // Decoder::prepare_sample() will merge the extents the same way but only for
                    // image items. It may be necessary here for Exif/XMP metadata for example.
                    let mut data_buffer: Vec<u8> = create_vec_exact(self.size)?;
                    for extent in &self.extents {
                        data_buffer.extend_from_slice(io.read_exact(extent.offset, extent.size)?);
                    }
                    self.data_buffer = Some(data_buffer);
                }
                self.data_buffer.as_ref().unwrap().as_slice()
            }
        };
        Ok(IStream::create(io_data))
    }

    fn validate_derived_image_dimensions(
        width: u32,
        height: u32,
        size_limit: Option<NonZero<u32>>,
        dimension_limit: Option<NonZero<u32>>,
    ) -> AvifResult<()> {
        if width == 0 || height == 0 || !check_limits(width, height, size_limit, dimension_limit) {
            return AvifError::invalid_image_grid("invalid derived image dimensions");
        }
        Ok(())
    }

    pub(crate) fn read_and_parse(
        &mut self,
        io: &mut GenericIO,
        tile_info: &mut TileInfo,
        size_limit: Option<NonZero<u32>>,
        dimension_limit: Option<NonZero<u32>>,
    ) -> AvifResult<()> {
        if self.is_grid_item() {
            let grid = &mut tile_info.grid;
            let mut stream = self.stream(io)?;
            // unsigned int(8) version = 0;
            let version = stream.read_u8()?;
            if version != 0 {
                return AvifError::not_implemented();
            }
            // unsigned int(8) flags;
            let flags = stream.read_u8()?;
            // unsigned int(8) rows_minus_one;
            grid.rows = stream.read_u8()? as u32 + 1;
            // unsigned int(8) columns_minus_one;
            grid.columns = stream.read_u8()? as u32 + 1;
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
            Self::validate_derived_image_dimensions(
                grid.width,
                grid.height,
                size_limit,
                dimension_limit,
            )?;
            if stream.has_bytes_left()? {
                return AvifError::invalid_image_grid("found unknown extra bytes in the grid box");
            }
        } else if self.is_overlay_item() {
            let overlay = &mut tile_info.overlay;
            let reference_count = self.source_item_ids.len();
            let mut stream = self.stream(io)?;
            // unsigned int(8) version = 0;
            let version = stream.read_u8()?;
            if version != 0 {
                return AvifError::not_implemented();
            }
            // unsigned int(8) flags;
            let flags = stream.read_u8()?;
            for j in 0..4 {
                // unsigned int(16) canvas_fill_value;
                overlay.canvas_fill_value[j] = stream.read_u16()?;
            }
            if (flags & 1) == 1 {
                // unsigned int(32) output_width;
                overlay.width = stream.read_u32()?;
                // unsigned int(32) output_height;
                overlay.height = stream.read_u32()?;
            } else {
                // unsigned int(16) output_width;
                overlay.width = stream.read_u16()? as u32;
                // unsigned int(16) output_height;
                overlay.height = stream.read_u16()? as u32;
            }
            Self::validate_derived_image_dimensions(
                overlay.width,
                overlay.height,
                size_limit,
                dimension_limit,
            )?;
            for _ in 0..reference_count {
                if (flags & 1) == 1 {
                    // unsigned int(32) horizontal_offset;
                    overlay.horizontal_offsets.push(stream.read_i32()?);
                    // unsigned int(32) vertical_offset;
                    overlay.vertical_offsets.push(stream.read_i32()?);
                } else {
                    // unsigned int(16) horizontal_offset;
                    overlay.horizontal_offsets.push(stream.read_i16()? as i32);
                    // unsigned int(16) vertical_offset;
                    overlay.vertical_offsets.push(stream.read_i16()? as i32);
                }
            }
            if stream.has_bytes_left()? {
                return AvifError::invalid_image_grid("found unknown extra bytes in the iovl box");
            }
        } else if self.is_tone_mapped_item() {
            let mut stream = self.stream(io)?;
            tile_info.gainmap_metadata = mp4box::parse_tmap(&mut stream)?;
        } else if self.is_sample_transform_item() {
            let num_inputs = self.source_item_ids.len();
            tile_info.sample_transform = mp4box::parse_sato(&mut self.stream(io)?, num_inputs)?;
        }
        Ok(())
    }

    pub(crate) fn operating_point(&self) -> u8 {
        match find_property!(self.properties, OperatingPointSelector) {
            Some(operating_point_selector) => *operating_point_selector,
            _ => 0, // default operating point.
        }
    }

    pub(crate) fn harvest_ispe(
        &mut self,
        alpha_ispe_required: bool,
        size_limit: Option<NonZero<u32>>,
        dimension_limit: Option<NonZero<u32>>,
    ) -> AvifResult<()> {
        if self.should_skip() {
            return Ok(());
        }

        match find_property!(self.properties, ImageSpatialExtents) {
            Some(image_spatial_extents) => {
                self.width = image_spatial_extents.width;
                self.height = image_spatial_extents.height;
                if self.width == 0 || self.height == 0 {
                    return AvifError::bmff_parse_failed("item id has invalid size.");
                }
                if !check_limits(
                    image_spatial_extents.width,
                    image_spatial_extents.height,
                    size_limit,
                    dimension_limit,
                ) {
                    return AvifError::bmff_parse_failed("item dimensions too large");
                }
            }
            None => {
                // No ispe was found.
                if self.is_auxiliary_alpha() {
                    if alpha_ispe_required {
                        return AvifError::bmff_parse_failed(
                            "alpha auxiliary image item is missing mandatory ispe",
                        );
                    }
                } else {
                    return AvifError::bmff_parse_failed("item is missing mandatory ispe property");
                }
            }
        }
        Ok(())
    }

    pub(crate) fn validate_properties(&self, items: &Items, pixi_required: bool) -> AvifResult<()> {
        let codec_config = self
            .codec_config()
            .ok_or(AvifError::BmffParseFailed("missing av1C property".into()))?;
        if self.is_derived_image_item() {
            for derived_item_id in &self.source_item_ids {
                let source_item = items.get(derived_item_id).unwrap();
                let source_codec_config =
                    source_item
                        .codec_config()
                        .ok_or(AvifError::BmffParseFailed(
                            "missing codec config property".into(),
                        ))?;
                // ISO/IEC 23000-22:2019 (MIAF), Section 7.3.11.4.1:
                // All input image of a grid image item shall use the same coding format, chroma
                // sampling format, and the same decoder configuration (see 7.3.6.2).
                // TODO: this is only a requirement for grids, the check for overlays is kept
                // for now to avoid behavior changes but it should be possible to remove it.
                if (self.is_grid_item() || self.is_overlay_item())
                    && codec_config != source_codec_config
                {
                    return AvifError::bmff_parse_failed(
                        "codec config of derived items do not match",
                    );
                }
                if self.is_sample_transform_item()
                    && (codec_config.pixel_format() != source_codec_config.pixel_format()
                        || source_item.width != self.width
                        || source_item.height != self.height)
                {
                    return AvifError::bmff_parse_failed(
                            "pixel format or dimensions of input images for sato derived item do not match"
                        );
                }
            }
        }
        match self.pixi() {
            Some(pixi) => {
                for plane in &pixi.planes {
                    // Check that the depth in pixi matches the codec config.
                    // For derived image items, the codec config comes from the first source item.
                    // Sample transform items can have a depth different from their source items.
                    if plane.depth != codec_config.depth() && !self.is_sample_transform_item() {
                        return AvifError::bmff_parse_failed(
                            "pixi depth does not match codec config depth",
                        );
                    }
                    // Extended pixi. Check that the subsampling of the chroma planes matches
                    // the codec config.
                    if matches!(
                        plane.channel_idc,
                        Some(ChannelIdc::SecondColorChannel | ChannelIdc::ThirdColorChannel)
                    ) {
                        if let Some(subsampling_type) = plane.subsampling_type {
                            if subsampling_type != codec_config.pixel_format() {
                                return AvifError::bmff_parse_failed(format!(
                                    "pixi {:?} does not match codec config {:?}",
                                    subsampling_type,
                                    codec_config.pixel_format()
                                ));
                            }
                        }
                        // Do not check subsampling_location.
                        // It does not matter enough to fail the decoding just because of that.
                    }
                }
            }
            None => {
                if pixi_required {
                    return AvifError::bmff_parse_failed("missing pixi property");
                }
            }
        }
        Ok(())
    }

    pub(crate) fn codec_config(&self) -> Option<&CodecConfiguration> {
        find_property!(self.properties, CodecConfiguration)
    }

    pub(crate) fn pixi(&self) -> Option<&PixelInformation> {
        find_property!(self.properties, PixelInformation)
    }

    pub(crate) fn a1lx(&self) -> Option<&[usize; 3]> {
        find_property!(self.properties, AV1LayeredImageIndexing)
    }

    pub(crate) fn lsel(&self) -> Option<&u16> {
        find_property!(self.properties, LayerSelector)
    }

    pub(crate) fn clli(&self) -> Option<&ContentLightLevelInformation> {
        find_property!(self.properties, ContentLightLevelInformation)
    }

    pub(crate) fn is_auxiliary_alpha(&self) -> bool {
        matches!(find_property!(&self.properties, AuxiliaryType),
                 Some(aux_type) if is_auxiliary_type_alpha(aux_type))
            && !self.is_sample_transform_item()
    }

    pub(crate) fn is_image_codec_item(&self) -> bool {
        [
            "av01",
            #[cfg(feature = "heic")]
            "hvc1",
        ]
        .contains(&self.item_type.as_str())
    }

    pub(crate) fn is_grid_item(&self) -> bool {
        self.item_type == "grid"
    }

    pub(crate) fn is_overlay_item(&self) -> bool {
        self.item_type == "iovl"
    }

    pub(crate) fn is_tone_mapped_item(&self) -> bool {
        self.item_type == "tmap"
    }

    pub(crate) fn is_sample_transform_item(&self) -> bool {
        self.item_type == "sato"
    }

    pub(crate) fn is_derived_image_item(&self) -> bool {
        self.is_grid_item()
            || self.is_overlay_item()
            || self.is_tone_mapped_item()
            || self.is_sample_transform_item()
    }

    pub(crate) fn is_image_item(&self) -> bool {
        self.is_image_codec_item() || self.is_derived_image_item()
    }

    pub(crate) fn should_skip(&self) -> bool {
        // The item has no payload in idat or mdat. It cannot be a coded image item, a
        // non-identity derived image item, or Exif/XMP metadata.
        self.size == 0
            // An essential property isn't supported by libavif. Ignore the whole item.
            || self.has_unsupported_essential_property
            // Probably Exif/XMP or some other data.
            || !self.is_image_item()
            // libavif does not support thumbnails.
            || self.thumbnail_for_id != 0
    }

    fn is_metadata(&self, item_type: &str, color_id: Option<u32>) -> bool {
        self.size != 0
            && !self.has_unsupported_essential_property
            && (color_id.is_none() || self.desc_for_id == color_id.unwrap())
            && self.item_type == *item_type
    }

    pub(crate) fn is_exif(&self, color_id: Option<u32>) -> bool {
        self.is_metadata("Exif", color_id)
    }

    pub(crate) fn is_xmp(&self, color_id: Option<u32>) -> bool {
        self.is_metadata("mime", color_id) && self.content_type == "application/rdf+xml"
    }

    pub(crate) fn max_extent(&self, sample: &DecodeSample) -> AvifResult<Extent> {
        if !self.idat.is_empty() {
            return Ok(Extent::default());
        }
        if sample.size == 0 {
            return AvifError::truncated_data();
        }
        let mut remaining_offset = sample.offset;
        let mut min_offset = u64::MAX;
        let mut max_offset = 0;
        if self.extents.is_empty() {
            return AvifError::truncated_data();
        } else if self.extents.len() == 1 {
            min_offset = sample.offset;
            max_offset = checked_add!(sample.offset, u64_from_usize(sample.size)?)?;
        } else {
            let mut remaining_size = sample.size;
            for extent in &self.extents {
                let mut start_offset = extent.offset;
                let mut size = extent.size;
                let sizeu64 = u64_from_usize(size)?;
                if remaining_offset != 0 {
                    if remaining_offset >= sizeu64 {
                        remaining_offset -= sizeu64;
                        continue;
                    } else {
                        checked_incr!(start_offset, remaining_offset);
                        checked_decr!(size, usize_from_u64(remaining_offset)?);
                        remaining_offset = 0;
                    }
                }
                // TODO(yguyon): Add comment to explain why it is fine to clip the extent size.
                let used_extent_size = std::cmp::min(size, remaining_size);
                let end_offset = checked_add!(start_offset, u64_from_usize(used_extent_size)?)?;
                min_offset = std::cmp::min(min_offset, start_offset);
                max_offset = std::cmp::max(max_offset, end_offset);
                remaining_size -= used_extent_size;
                if remaining_size == 0 {
                    break;
                }
            }
            if remaining_size != 0 {
                return AvifError::truncated_data();
            }
        }
        Ok(Extent {
            offset: min_offset,
            size: usize_from_u64(checked_sub!(max_offset, min_offset)?)?,
        })
    }
}

pub type Items = BTreeMap<u32, Item>;

fn insert_item_if_not_exists(id: u32, items: &mut Items) {
    if items.contains_key(&id) {
        return;
    }
    items.insert(
        id,
        Item {
            id,
            ..Item::default()
        },
    );
}

pub(crate) fn construct_items(meta: &MetaBox) -> AvifResult<Items> {
    let mut items: Items = BTreeMap::new();
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
        insert_item_if_not_exists(iloc.item_id, &mut items);
        let item = items.get_mut(&iloc.item_id).unwrap();
        if !item.extents.is_empty() {
            return AvifError::bmff_parse_failed("item already has extents");
        }
        if iloc.construction_method == 1 {
            item.idat.clone_from(&meta.idat);
        }
        for extent in &iloc.extents {
            item.extents.push(Extent {
                offset: checked_add!(iloc.base_offset, extent.offset)?,
                size: extent.size,
            });
            checked_incr!(item.size, extent.size);
        }
    }
    let mut ipma_seen: HashSet<u32> = HashSet::with_hasher(NonRandomHasherState);
    for association in &meta.iprp.associations {
        if association.associations.is_empty() {
            continue;
        }
        if ipma_seen.contains(&association.item_id) {
            return AvifError::bmff_parse_failed("item has duplicate ipma entry");
        }
        ipma_seen.insert(association.item_id);

        insert_item_if_not_exists(association.item_id, &mut items);
        let item = items.get_mut(&association.item_id).unwrap();
        for (property_index_ref, essential_ref) in &association.associations {
            let property_index: usize = *property_index_ref as usize;
            let essential = *essential_ref;
            if property_index == 0 {
                if essential {
                    return AvifError::bmff_parse_failed(format!(
                        "item id {} contains an illegal essential property index 0",
                        { item.id }
                    ));
                }
                continue;
            }
            // property_index is 1-based.
            if property_index > meta.iprp.properties.len() {
                return AvifError::bmff_parse_failed("invalid property_index in ipma");
            }

            match (&meta.iprp.properties[property_index - 1], essential) {
                (ItemProperty::Unknown(_), true) => item.has_unsupported_essential_property = true,
                (ItemProperty::AV1LayeredImageIndexing(_), true) => {
                    return AvifError::bmff_parse_failed(
                        "invalid essential property"
                    );
                }
                (
                    ItemProperty::OperatingPointSelector(_)
                    | ItemProperty::LayerSelector(_)
                    // MIAF 2019/Amd. 2:2021: Section 7.3.9:
                    //   All transformative properties associated with coded and derived images
                    //   shall be marked as essential.
                    | ItemProperty::CleanAperture(_)
                    | ItemProperty::ImageRotation(_)
                    | ItemProperty::ImageMirror(_),
                    false,
                ) => {
                    return AvifError::bmff_parse_failed(
                        "required essential property not marked as essential"
                    );
                }
                (property, _) => item.properties.push(property.clone()),
            }
        }
    }

    for reference in &meta.iref {
        insert_item_if_not_exists(reference.from_item_id, &mut items);
        let item = items.get_mut(&reference.from_item_id).unwrap();
        match reference.reference_type.as_str() {
            "thmb" => item.thumbnail_for_id = reference.to_item_id,
            "auxl" => item.aux_for_id = reference.to_item_id,
            "cdsc" => item.desc_for_id = reference.to_item_id,
            "prem" => item.prem_by_id = reference.to_item_id,
            "dimg" => {
                // derived images refer in the opposite direction.
                insert_item_if_not_exists(reference.to_item_id, &mut items);
                let dimg_item = items.get_mut(&reference.to_item_id).unwrap();
                if dimg_item.dimg_for_id != 0 {
                    return Err(if dimg_item.dimg_for_id == reference.from_item_id {
                        // Section 8.11.12.1 of ISO/IEC 14496-12:
                        //   The items linked to are then represented by an array of to_item_IDs;
                        //   within a given array, a given value shall occur at most once.
                        AvifError::BmffParseFailed(format!(
                            "multiple dimg references for item ID {}",
                            dimg_item.dimg_for_id
                        ))
                    } else {
                        AvifError::NotImplemented
                    });
                }
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
