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
use crate::internal_utils::sampletransform::*;
use crate::internal_utils::*;
use crate::*;

use std::num::NonZero;

#[derive(Clone, Copy, Debug, Default)]
pub struct DecodeSample {
    pub item_id: u32, // 1-based. 0 if it comes from a track.
    pub offset: u64,
    pub size: usize,
    pub spatial_id: u8,
    pub sync: bool,
}

impl DecodeSample {
    pub(crate) fn partial_data<'a>(
        &'a self,
        io: &'a mut Box<impl decoder::IO + ?Sized>,
        buffer: &'a Option<Vec<u8>>,
        size: usize,
    ) -> AvifResult<&'a [u8]> {
        match buffer {
            Some(x) => {
                let start_offset = usize_from_u64(self.offset)?;
                let end_offset = checked_add!(start_offset, size)?;
                let range = start_offset..end_offset;
                x.get(range).ok_or(AvifError::NoContent)
            }
            None => {
                let data = io.read(self.offset, size)?;
                if data.len() != size {
                    AvifError::truncated_data()
                } else {
                    Ok(data)
                }
            }
        }
    }

    pub(crate) fn data<'a>(
        &'a self,
        io: &'a mut Box<impl decoder::IO + ?Sized>,
        buffer: &'a Option<Vec<u8>>,
    ) -> AvifResult<&'a [u8]> {
        self.partial_data(io, buffer, self.size)
    }
}

#[derive(Debug, Default)]
pub struct DecodeInput {
    pub samples: Vec<DecodeSample>,
    pub all_layers: bool,
    pub decoding_item: DecodingItem,
}

#[derive(Debug, Default)]
pub struct Overlay {
    pub canvas_fill_value: [u16; 4],
    pub width: u32,
    pub height: u32,
    pub horizontal_offsets: Vec<i32>,
    pub vertical_offsets: Vec<i32>,
}

#[derive(Debug, Default)]
pub(crate) struct TileInfo {
    pub tile_count: u32,
    pub decoded_tile_count: u32,
    pub grid: Grid,
    pub overlay: Overlay,
    pub gainmap_metadata: GainMapMetadata,
    pub sample_transform: SampleTransform,
}

impl TileInfo {
    pub(crate) fn is_grid(&self) -> bool {
        self.grid.rows > 0 && self.grid.columns > 0
    }

    pub(crate) fn is_overlay(&self) -> bool {
        !self.overlay.horizontal_offsets.is_empty() && !self.overlay.vertical_offsets.is_empty()
    }

    pub(crate) fn is_sample_transform(&self) -> bool {
        !self.sample_transform.tokens.is_empty()
    }

    pub(crate) fn is_derived_image(&self) -> bool {
        self.is_grid() || self.is_overlay() || self.is_sample_transform()
    }

    pub(crate) fn grid_tile_count(&self) -> AvifResult<u32> {
        if self.is_grid() {
            checked_mul!(self.grid.rows, self.grid.columns)
        } else {
            Ok(1)
        }
    }

    pub(crate) fn decoded_row_count(&self, image_height: u32, tile_height: u32) -> u32 {
        if self.decoded_tile_count == 0 {
            return 0;
        }
        if self.decoded_tile_count == self.tile_count || !self.is_grid() {
            return image_height;
        }
        std::cmp::min(
            (self.decoded_tile_count / self.grid.columns) * tile_height,
            image_height,
        )
    }

    pub(crate) fn is_fully_decoded(&self) -> bool {
        self.tile_count == self.decoded_tile_count
    }
}

pub struct Tile {
    pub width: u32,
    pub height: u32,
    pub operating_point: u8,
    pub image: Image,
    pub input: DecodeInput,
    pub codec_index: usize,
    pub codec_config: CodecConfiguration,
}

impl Tile {
    pub(crate) fn create_from_item(
        item: &mut Item,
        allow_progressive: bool,
        image_count_limit: Option<NonZero<u32>>,
        size_hint: u64,
    ) -> AvifResult<Tile> {
        if size_hint != 0 && item.size as u64 > size_hint {
            return AvifError::bmff_parse_failed("exceeded size_hint");
        }
        let mut tile = Tile {
            width: item.width,
            height: item.height,
            operating_point: item.operating_point(),
            image: Image::default(),
            input: DecodeInput::default(),
            codec_index: 0,
            codec_config: item
                .codec_config()
                .ok_or(AvifError::BmffParseFailed(
                    "missing codec config property".into(),
                ))?
                .try_clone()?,
        };
        let mut layer_sizes: [usize; MAX_AV1_LAYER_COUNT] = [0; MAX_AV1_LAYER_COUNT];
        let mut layer_count: usize = 0;
        let a1lx = item.a1lx();
        let has_a1lx = a1lx.is_some();
        if let Some(a1lx) = a1lx {
            let mut remaining_size: usize = item.size;
            for i in 0usize..3 {
                layer_count += 1;
                if a1lx[i] > 0 {
                    // >= instead of > because there must be room for the last layer
                    if a1lx[i] >= remaining_size {
                        return AvifError::bmff_parse_failed(format!(
                            "a1lx layer index [{i}] does not fit in item size"
                        ));
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
        }
        let (lsel, has_lsel) = match item.lsel() {
            Some(x) => (*x, true),
            None => (0, false),
        };
        // Progressive images offer layers via the a1lxProp, but don't specify a layer selection with
        // lsel.
        item.progressive = has_a1lx && (!has_lsel || lsel == 0xFFFF);
        let base_item_offset = match item.extents.as_slice() {
            [extent] => extent.offset,
            _ => 0,
        };
        if has_lsel && lsel != 0xFFFF {
            // Layer selection. This requires that the underlying AV1 codec decodes all layers, and
            // then only returns the requested layer as a single frame. To the user of libavif,
            // this appears to be a single frame.
            tile.input.all_layers = true;
            let mut sample_size: usize = 0;
            let layer_id = usize_from_u16(lsel)?;
            if layer_count > 0 {
                // Optimization: If we're selecting a layer that doesn't require the entire image's
                // payload (hinted via the a1lx box).
                if layer_id >= layer_count {
                    return AvifError::invalid_image_grid("lsel layer index not found in a1lx.");
                }
                let layer_id_plus_1 = layer_id + 1;
                for layer_size in layer_sizes.iter().take(layer_id_plus_1) {
                    checked_incr!(sample_size, *layer_size);
                }
            } else {
                // This layer payload subsection is not known. Use the whole payload.
                sample_size = item.size;
            }
            let sample = DecodeSample {
                item_id: item.id,
                offset: base_item_offset,
                size: sample_size,
                spatial_id: lsel as u8,
                sync: true,
            };
            tile.input.samples.try_push(sample)?;
        } else if item.progressive && allow_progressive {
            // Progressive image. Decode all layers and expose them all to the
            // user.
            if let Some(limit) = image_count_limit {
                if layer_count as u32 > limit.get() {
                    return AvifError::bmff_parse_failed(
                        "exceeded image_count_limit (progressive)",
                    );
                }
            }
            tile.input.all_layers = true;
            let mut offset = 0;
            for (i, layer_size) in layer_sizes.iter().take(layer_count).enumerate() {
                let sample = DecodeSample {
                    item_id: item.id,
                    offset: checked_add!(base_item_offset, offset)?,
                    size: *layer_size,
                    spatial_id: 0xff,
                    sync: i == 0, // Assume all layers depend on the first layer.
                };
                tile.input.samples.try_push(sample)?;
                offset = checked_add!(offset, *layer_size as u64)?;
            }
        } else {
            // Typical case: Use the entire item's payload for a single frame output
            if item.has_native_alpha() {
                tile.input.all_layers = true;
            }
            let sample = DecodeSample {
                item_id: item.id,
                offset: base_item_offset,
                size: item.size,
                // Legal spatial_id values are [0,1,2,3], so this serves as a sentinel value for
                // "do not filter by spatial_id"
                spatial_id: 0xff,
                sync: true,
            };
            tile.input.samples.try_push(sample)?;
        }
        Ok(tile)
    }

    pub(crate) fn create_from_track(
        track: &Track,
        image_count_limit: Option<NonZero<u32>>,
        size_hint: u64,
        decoding_item: DecodingItem,
    ) -> AvifResult<Tile> {
        let properties = track
            .get_properties()
            .ok_or(AvifError::BmffParseFailed("".into()))?;
        let codec_config = find_property!(properties, CodecConfiguration)
            .ok_or(AvifError::BmffParseFailed("".into()))?
            .try_clone()?;
        let mut tile = Tile {
            width: track.width,
            height: track.height,
            operating_point: 0, // No way to set operating point via tracks
            image: Image::default(),
            input: DecodeInput {
                decoding_item,
                ..DecodeInput::default()
            },
            codec_index: 0,
            codec_config,
        };
        let sample_table = track
            .sample_table
            .as_ref()
            .ok_or(AvifError::BmffParseFailed("missing sample table".into()))?;

        if let Some(limit) = image_count_limit {
            let mut limit = limit.get();
            for (chunk_index, _chunk_offset) in sample_table.chunk_offsets.iter().enumerate() {
                // Figure out how many samples are in this chunk.
                let sample_count = sample_table.get_sample_count_of_chunk(chunk_index as u32);
                if sample_count == 0 {
                    return AvifError::bmff_parse_failed("chunk with 0 samples found");
                }
                if sample_count > limit {
                    return AvifError::bmff_parse_failed("exceeded image_count_limit");
                }
                limit -= sample_count;
            }
        }

        let mut sample_size_index: usize = 0;
        for (chunk_index, chunk_offset) in sample_table.chunk_offsets.iter().enumerate() {
            // Figure out how many samples are in this chunk.
            let sample_count = sample_table.get_sample_count_of_chunk(chunk_index as u32);
            if sample_count == 0 {
                return AvifError::bmff_parse_failed("chunk with 0 samples found");
            }

            let mut sample_offset = *chunk_offset;
            for _ in 0..sample_count {
                let sample_size = sample_table.sample_size(sample_size_index)?;
                let sample_size_hint = checked_add!(sample_offset, sample_size as u64)?;
                if size_hint != 0 && sample_size_hint > size_hint {
                    return AvifError::bmff_parse_failed("exceeded size_hint");
                }
                let sample = DecodeSample {
                    item_id: 0,
                    offset: sample_offset,
                    size: sample_size,
                    // Legal spatial_id values are [0,1,2,3], so this serves as a sentinel value for "do
                    // not filter by spatial_id"
                    spatial_id: 0xff,
                    // Assume first sample is always sync (in case stss box was missing).
                    sync: tile.input.samples.is_empty(),
                };
                tile.input.samples.try_push(sample)?;
                checked_incr!(sample_offset, sample_size as u64);
                checked_incr!(sample_size_index, 1);
            }
        }
        for sync_sample_number in &sample_table.sync_samples {
            let index = usize_from_u32(*sync_sample_number)?;
            // sample_table.sync_samples is 1-based.
            let Some(sample) = index
                .checked_sub(1)
                .and_then(|i| tile.input.samples.get_mut(i))
            else {
                return AvifError::bmff_parse_failed(format!("invalid sync sample number {index}"));
            };
            sample.sync = true;
        }
        Ok(tile)
    }

    pub(crate) fn max_sample_size(&self) -> usize {
        match self.input.samples.iter().max_by_key(|sample| sample.size) {
            Some(sample) => sample.size,
            None => 0,
        }
    }
}

impl Image {
    pub(crate) fn copy_from_tile(
        &mut self,
        tile: &Image,
        grid: &Grid,
        tile_index: u32,
        category: Category,
        crop_rect: Option<&CropRect>,
    ) -> AvifResult<()> {
        // When copying non-alpha planes, the source and destination yuv_format has to be the
        // same. This restriction does not apply to alpha planes since only Y plane is copied.
        if category != Category::Alpha && self.yuv_format != tile.yuv_format {
            return AvifError::invalid_image_grid("mismatch in grid tile yuv formats.");
        }
        // The samples are copied as is, so the source and destination depth has to be the same
        // for all planes.
        if self.depth != tile.depth {
            return AvifError::invalid_image_grid("mismatch in grid tile depths.");
        }
        let row_index = tile_index / grid.columns;
        let column_index = tile_index % grid.columns;
        let planes: &[Plane] = if category == Category::Color && tile.has_plane(Plane::A) {
            &ALL_PLANES
        } else {
            category.planes()
        };
        if let Some(target) = crop_rect {
            if category != Category::Alpha
                && ((self.yuv_format.chroma_shift_x().0 > 0 && tile.width % 2 != 0)
                    || (self.yuv_format.chroma_shift_y() > 0 && tile.height % 2 != 0))
            {
                return AvifError::invalid_image_grid(
                    "grid tile dimensions must be even along chroma-subsampled axes.",
                );
            }
            let target = target.align_to_pixel_format(self.yuv_format);
            if !target.is_within(grid.width, grid.height)
                || self.width != target.width
                || self.height != target.height
            {
                return AvifError::invalid_argument();
            }
            let tile_x0 = checked_mul!(column_index, tile.width)?;
            let tile_x1 = std::cmp::min(checked_add!(tile_x0, tile.width)?, grid.width);
            let tile_y0 = checked_mul!(row_index, tile.height)?;
            let tile_y1 = std::cmp::min(checked_add!(tile_y0, tile.height)?, grid.height);

            let inter_x0 = std::cmp::max(tile_x0, target.x);
            let inter_x1 = std::cmp::min(tile_x1, checked_add!(target.x, target.width)?);
            let inter_y0 = std::cmp::max(tile_y0, target.y);
            let inter_y1 = std::cmp::min(tile_y1, checked_add!(target.y, target.height)?);
            if inter_x0 >= inter_x1 || inter_y0 >= inter_y1 {
                return Ok(());
            }

            let luma_src_x = inter_x0 - tile_x0;
            let luma_dst_x = inter_x0 - target.x;
            let luma_w = inter_x1 - inter_x0;
            let luma_src_y = inter_y0 - tile_y0;
            let luma_dst_y = inter_y0 - target.y;
            let luma_h = inter_y1 - inter_y0;

            for plane in planes {
                let plane = *plane;
                if !tile.has_plane(plane) {
                    continue;
                }
                let (src, dst, size) = if matches!(plane, Plane::Y | Plane::A) {
                    (
                        (usize_from_u32(luma_src_x)?, luma_src_y),
                        (usize_from_u32(luma_dst_x)?, luma_dst_y),
                        (usize_from_u32(luma_w)?, luma_h),
                    )
                } else {
                    (
                        (
                            usize_from_u32(self.yuv_format.apply_chroma_shift_x(luma_src_x))?,
                            self.yuv_format.apply_chroma_shift_y(luma_src_y),
                        ),
                        (
                            usize_from_u32(self.yuv_format.apply_chroma_shift_x(luma_dst_x))?,
                            self.yuv_format.apply_chroma_shift_y(luma_dst_y),
                        ),
                        (
                            Self::plane_width(self.yuv_format, plane, luma_w),
                            u32_from_usize(Self::plane_height(self.yuv_format, plane, luma_h))?,
                        ),
                    )
                };
                self.copy_plane_block(tile, plane, src, dst, size)?;
            }
            return Ok(());
        }
        for plane in planes {
            let plane = *plane;
            let src_plane = tile.plane_data(plane);
            if src_plane.is_none() {
                continue;
            }
            let src_plane = src_plane.unwrap();
            // If this is the last tile column, clamp to left over width.
            let src_width_to_copy = if column_index == grid.columns - 1 {
                let width_so_far = checked_mul!(src_plane.width, column_index)?;
                checked_sub!(self.width(plane), usize_from_u32(width_so_far)?)?
            } else {
                usize_from_u32(src_plane.width)?
            };

            // If this is the last tile row, clamp to left over height.
            let src_height_to_copy = if row_index == grid.rows - 1 {
                let height_so_far = checked_mul!(src_plane.height, row_index)?;
                checked_sub!(u32_from_usize(self.height(plane))?, height_so_far)?
            } else {
                src_plane.height
            };

            let dst_y_start = checked_mul!(row_index, src_plane.height)?;
            let dst_x_offset = usize_from_u32(checked_mul!(column_index, src_plane.width)?)?;
            self.copy_plane_block(
                tile,
                plane,
                (0, 0),
                (dst_x_offset, dst_y_start),
                (src_width_to_copy, src_height_to_copy),
            )?;
        }
        Ok(())
    }

    // Maps a cumulative cell boundary `pos` in `0..=src_len` to `0..=dst_len`, rounding interior
    // boundaries (`0 < pos < src_len`) to multiples of `align` (1 or 2) so chroma-subsampled cell
    // seams never split a 2x2 chroma sample.
    fn map_grid_boundary(pos: u32, src_len: u32, dst_len: u32, align: u32) -> AvifResult<u32> {
        if pos == 0 {
            return Ok(0);
        }
        if pos >= src_len {
            return Ok(dst_len);
        }
        let pos = u128::from(pos);
        let src = u128::from(src_len);
        let dst = u128::from(dst_len);
        let mapped = if align == 2 {
            2 * ((pos * dst + src) / (2 * src))
        } else {
            (2 * pos * dst + src) / (2 * src)
        };
        u32::try_from(mapped).map_err(AvifError::map_unknown_error)
    }

    #[allow(dead_code)] // Used in a follow-up change.
    pub(crate) fn copy_scaled_from_tile(
        &mut self,
        tile: &Image,
        grid: &Grid,
        tile_index: u32,
        category: Category,
        crop_rect: Option<&CropRect>,
        scaled_size: ImageSize,
    ) -> AvifResult<()> {
        let tile_count = checked_mul!(grid.rows, grid.columns)?;
        if grid.rows == 0 || grid.columns == 0 || tile_index >= tile_count {
            return AvifError::invalid_image_grid("grid tile index out of bounds.");
        }
        let p010_to_i010 = tile.yuv_format == PixelFormat::AndroidP010
            && self.yuv_format == PixelFormat::Yuv420
            && tile.depth == 16
            && self.depth == 10;
        if !p010_to_i010 {
            if category != Category::Alpha && self.yuv_format != tile.yuv_format {
                return AvifError::invalid_image_grid("mismatch in grid tile yuv formats.");
            }
            if self.depth != tile.depth {
                return AvifError::invalid_image_grid("mismatch in grid tile depths.");
            }
        }
        // Unlike `copy_from_tile`, interior cell boundaries are rounded to `align_x` / `align_y`
        // (2 along chroma-subsampled axes of `self.yuv_format`) for both Color and Alpha so color
        // and alpha seams land on identical output coordinates. That requires `tile.width` and
        // `tile.height` to be even along `self.yuv_format`'s chroma-subsampled axes even for
        // `Category::Alpha` (otherwise rounding an odd seam to an even output coordinate can
        // exceed `luma_w` / `luma_h` and attempt upscaling; `grid_tiles_for_crop` enforces the
        // same check and falls back to full-grid alpha decode + scale for odd alpha tiles).
        if (self.yuv_format.chroma_shift_x().0 > 0 && tile.width % 2 != 0)
            || (self.yuv_format.chroma_shift_y() > 0 && tile.height % 2 != 0)
        {
            return AvifError::invalid_image_grid(
                "grid tile dimensions must be even along chroma-subsampled axes.",
            );
        }
        let default_rect = CropRect {
            x: 0,
            y: 0,
            width: grid.width,
            height: grid.height,
        };
        let target = match crop_rect {
            Some(rect) => rect.align_to_pixel_format(self.yuv_format),
            None => default_rect,
        };
        if !target.is_within(grid.width, grid.height)
            || scaled_size.width == 0
            || scaled_size.height == 0
            || scaled_size.width > target.width
            || scaled_size.height > target.height
            || self.width != scaled_size.width
            || self.height != scaled_size.height
        {
            return AvifError::invalid_argument();
        }
        if (tile.yuv_format == PixelFormat::AndroidP010
            || self.yuv_format == PixelFormat::AndroidP010)
            && !p010_to_i010
            && (scaled_size.width != target.width || scaled_size.height != target.height)
        {
            return AvifError::invalid_image_grid(
                "AndroidP010 grid scaling requires a 10-bit Yuv420 destination.",
            );
        }

        let row_index = tile_index / grid.columns;
        let column_index = tile_index % grid.columns;
        let tile_x0 = checked_mul!(column_index, tile.width)?;
        let tile_x1 = std::cmp::min(checked_add!(tile_x0, tile.width)?, grid.width);
        let tile_y0 = checked_mul!(row_index, tile.height)?;
        let tile_y1 = std::cmp::min(checked_add!(tile_y0, tile.height)?, grid.height);

        let inter_x0 = std::cmp::max(tile_x0, target.x);
        let inter_x1 = std::cmp::min(tile_x1, checked_add!(target.x, target.width)?);
        let inter_y0 = std::cmp::max(tile_y0, target.y);
        let inter_y1 = std::cmp::min(tile_y1, checked_add!(target.y, target.height)?);
        if inter_x0 >= inter_x1 || inter_y0 >= inter_y1 {
            return Ok(());
        }

        let luma_src_x = inter_x0 - tile_x0;
        let luma_w = inter_x1 - inter_x0;
        let luma_src_y = inter_y0 - tile_y0;
        let luma_h = inter_y1 - inter_y0;

        let align_x = if self.yuv_format.chroma_shift_x().0 > 0 { 2 } else { 1 };
        let align_y = if self.yuv_format.chroma_shift_y() > 0 { 2 } else { 1 };
        let src_x0_rel = inter_x0 - target.x;
        let src_x1_rel = inter_x1 - target.x;
        let src_y0_rel = inter_y0 - target.y;
        let src_y1_rel = inter_y1 - target.y;

        let dst_x0 = Self::map_grid_boundary(src_x0_rel, target.width, scaled_size.width, align_x)?;
        let dst_x1 = Self::map_grid_boundary(src_x1_rel, target.width, scaled_size.width, align_x)?;
        let dst_y0 =
            Self::map_grid_boundary(src_y0_rel, target.height, scaled_size.height, align_y)?;
        let dst_y1 =
            Self::map_grid_boundary(src_y1_rel, target.height, scaled_size.height, align_y)?;
        let dst_w = checked_sub!(dst_x1, dst_x0)?;
        let dst_h = checked_sub!(dst_y1, dst_y0)?;
        if dst_w == 0 || dst_h == 0 {
            return Ok(());
        }

        let cell_rect = CropRect {
            x: luma_src_x,
            y: luma_src_y,
            width: luma_w,
            height: luma_h,
        };
        let mut cell_view = if category == Category::Alpha {
            tile.view_plane_a(&cell_rect)?
        } else {
            tile.view(&cell_rect)?
        };
        if (dst_w, dst_h) != (luma_w, luma_h) {
            cell_view.scale_planes_in_place(dst_w, dst_h, category)?;
        } else if p010_to_i010 {
            cell_view.convert_p010_to_i010()?;
        }

        let planes: &[Plane] = if category == Category::Color && cell_view.has_plane(Plane::A) {
            &ALL_PLANES
        } else {
            category.planes()
        };
        for &plane in planes {
            if !cell_view.has_plane(plane) {
                continue;
            }
            let (dst, size) = if matches!(plane, Plane::Y | Plane::A) {
                (
                    (usize_from_u32(dst_x0)?, dst_y0),
                    (usize_from_u32(dst_w)?, dst_h),
                )
            } else {
                (
                    (
                        usize_from_u32(self.yuv_format.apply_chroma_shift_x(dst_x0))?,
                        self.yuv_format.apply_chroma_shift_y(dst_y0),
                    ),
                    (
                        Self::plane_width(self.yuv_format, plane, dst_w),
                        u32_from_usize(Self::plane_height(self.yuv_format, plane, dst_h))?,
                    ),
                )
            };
            self.copy_plane_block(&cell_view, plane, (0, 0), dst, size)?;
        }
        Ok(())
    }

    // Copies the block of samples of the given size (width, height) at position src (x, y) in
    // the given plane of tile to position dst (x, y) in the same plane of self. Positions and
    // dimensions are expressed in samples of that plane.
    fn copy_plane_block(
        &mut self,
        tile: &Image,
        plane: Plane,
        src: (usize, u32),
        dst: (usize, u32),
        size: (usize, u32),
    ) -> AvifResult<()> {
        let (src_x, src_y) = src;
        let (dst_x, dst_y) = dst;
        let (width, height) = size;
        let src_x_end = checked_add!(src_x, width)?;
        let dst_x_end = checked_add!(dst_x, width)?;
        if self.depth == 8 {
            for y in 0..height {
                let src_row = tile.row(plane, checked_add!(src_y, y)?)?;
                let src_slice = src_row
                    .get(src_x..src_x_end)
                    .ok_or(AvifError::InvalidArgument)?;
                let dst_row = self.row_mut(plane, checked_add!(dst_y, y)?)?;
                let dst_slice = dst_row
                    .get_mut(dst_x..dst_x_end)
                    .ok_or(AvifError::InvalidArgument)?;
                dst_slice.copy_from_slice(src_slice);
            }
        } else {
            for y in 0..height {
                let src_row = tile.row16(plane, checked_add!(src_y, y)?)?;
                let src_slice = src_row
                    .get(src_x..src_x_end)
                    .ok_or(AvifError::InvalidArgument)?;
                let dst_row = self.row16_mut(plane, checked_add!(dst_y, y)?)?;
                let dst_slice = dst_row
                    .get_mut(dst_x..dst_x_end)
                    .ok_or(AvifError::InvalidArgument)?;
                dst_slice.copy_from_slice(src_slice);
            }
        }
        Ok(())
    }

    pub(crate) fn copy_and_overlay_from_tile(
        &mut self,
        tile: &Image,
        tile_info: &TileInfo,
        tile_index: u32,
        category: Category,
    ) -> AvifResult<()> {
        // When copying non-alpha planes, the source and destination yuv_format has to be the
        // same. This restriction does not apply to alpha planes since only Y plane is copied.
        if category != Category::Alpha && self.yuv_format != tile.yuv_format {
            return AvifError::invalid_image_grid("mismatch in overlay tile yuv formats.");
        }
        // The samples are copied as is, so the source and destination depth has to be the same
        // for all planes.
        if self.depth != tile.depth {
            return AvifError::invalid_image_grid("mismatch in overlay tile depths.");
        }
        // This function is used only when |tile| contains pointers and self contains buffers.
        let planes: &[Plane] = if category == Category::Color && tile.has_plane(Plane::A) {
            &ALL_PLANES
        } else {
            category.planes()
        };
        for plane in planes {
            let plane = *plane;
            let (Some(_), Some(dst_plane)) = (tile.plane_data(plane), self.plane_data(plane))
            else {
                continue;
            };
            let tile_index = usize_from_u32(tile_index)?;

            let vertical_offset = *tile_info.overlay.vertical_offsets.get(tile_index).ok_or(
                AvifError::InvalidImageGrid("overlay tile index out of bounds".into()),
            )? as i128;
            let horizontal_offset = *tile_info.overlay.horizontal_offsets.get(tile_index).ok_or(
                AvifError::InvalidImageGrid("overlay tile index out of bounds".into()),
            )? as i128;
            let src_height = tile.height as i128;
            let src_width = tile.width as i128;
            let dst_height = dst_plane.height as i128;
            let dst_width = dst_plane.width as i128;

            if matches!(plane, Plane::Y | Plane::A)
                && (vertical_offset + src_height < 0
                    || horizontal_offset + src_width < 0
                    || vertical_offset >= dst_height
                    || horizontal_offset >= dst_width)
            {
                // Entire tile outside of the canvas. It is sufficient to perform this check only
                // for Y and A plane since they are never sub-sampled.
                return Ok(());
            }

            let mut src_y_start: u32;
            let mut src_height_to_copy: u32;
            let mut dst_y_start: u32;
            if vertical_offset >= 0 {
                src_y_start = 0;
                src_height_to_copy = src_height as u32;
                dst_y_start = vertical_offset as u32;
            } else {
                src_y_start = vertical_offset.unsigned_abs() as u32;
                src_height_to_copy = (src_height - vertical_offset.abs()) as u32;
                dst_y_start = 0;
            }

            let mut src_x_start: u32;
            let mut src_width_to_copy: u32;
            let mut dst_x_start: u32;
            if horizontal_offset >= 0 {
                src_x_start = 0;
                src_width_to_copy = src_width as u32;
                dst_x_start = horizontal_offset as u32;
            } else {
                src_x_start = horizontal_offset.unsigned_abs() as u32;
                src_width_to_copy = (src_width - horizontal_offset.abs()) as u32;
                dst_x_start = 0;
            }

            // Clamp width to the canvas width.
            if self.width - dst_x_start < src_width_to_copy {
                src_width_to_copy = self.width - dst_x_start;
            }

            // Clamp height to the canvas height.
            if self.height - dst_y_start < src_height_to_copy {
                src_height_to_copy = self.height - dst_y_start;
            }

            // Apply chroma subsampling to the offsets.
            if plane == Plane::U || plane == Plane::V {
                src_y_start = tile.yuv_format.apply_chroma_shift_y(src_y_start);
                src_height_to_copy = tile.yuv_format.apply_chroma_shift_y(src_height_to_copy);
                dst_y_start = tile.yuv_format.apply_chroma_shift_y(dst_y_start);
                src_x_start = tile.yuv_format.apply_chroma_shift_x(src_x_start);
                src_width_to_copy = tile.yuv_format.apply_chroma_shift_x(src_width_to_copy);
                dst_x_start = tile.yuv_format.apply_chroma_shift_x(dst_x_start);
            }

            let src_width_to_copy = usize_from_u32(src_width_to_copy)?;
            let src_y_range = src_y_start..checked_add!(src_y_start, src_height_to_copy)?;
            let dst_x_start = usize_from_u32(dst_x_start)?;
            let dst_x_end = checked_add!(dst_x_start, src_width_to_copy)?;
            let src_x_start = usize_from_u32(src_x_start)?;
            let src_x_end = checked_add!(src_x_start, src_width_to_copy)?;
            let mut dst_y = dst_y_start;
            if self.depth == 8 {
                for src_y in src_y_range {
                    let src_row = tile.row(plane, src_y)?;
                    let src_slice = src_row
                        .get(src_x_start..src_x_end)
                        .ok_or(AvifError::InvalidArgument)?;
                    let dst_row = self.row_mut(plane, dst_y)?;
                    let dst_slice = dst_row
                        .get_mut(dst_x_start..dst_x_end)
                        .ok_or(AvifError::InvalidArgument)?;
                    dst_slice.copy_from_slice(src_slice);
                    checked_incr!(dst_y, 1);
                }
            } else {
                for src_y in src_y_range {
                    let src_row = tile.row16(plane, src_y)?;
                    let src_slice = src_row
                        .get(src_x_start..src_x_end)
                        .ok_or(AvifError::InvalidArgument)?;
                    let dst_row = self.row16_mut(plane, dst_y)?;
                    let dst_slice = dst_row
                        .get_mut(dst_x_start..dst_x_end)
                        .ok_or(AvifError::InvalidArgument)?;
                    dst_slice.copy_from_slice(src_slice);
                    checked_incr!(dst_y, 1);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    fn create_image(width: u32, height: u32, depth: u8, category: Category) -> Image {
        let mut image = Image {
            width,
            height,
            depth,
            yuv_format: PixelFormat::Yuv444,
            ..Default::default()
        };
        assert!(image.allocate_planes(category).is_ok());
        image
    }

    #[test]
    fn copy_from_tile_depth_mismatch() {
        let grid = Grid {
            rows: 1,
            columns: 2,
            width: 4,
            height: 2,
        };
        for category in [Category::Color, Category::Alpha] {
            for (grid_depth, tile_depth) in [(8, 10), (10, 8), (10, 12)] {
                let mut grid_image = create_image(4, 2, grid_depth, category);
                let tile = create_image(2, 2, tile_depth, category);
                assert!(matches!(
                    grid_image.copy_from_tile(&tile, &grid, 0, category, None),
                    Err(AvifError::InvalidImageGrid(_))
                ));
            }
            for depth in [8, 10] {
                let mut grid_image = create_image(4, 2, depth, category);
                let tile = create_image(2, 2, depth, category);
                for tile_index in 0..2 {
                    assert!(grid_image
                        .copy_from_tile(&tile, &grid, tile_index, category, None)
                        .is_ok());
                }
            }
        }
    }

    #[test]
    fn copy_and_overlay_from_tile_depth_mismatch() {
        let tile_info = TileInfo {
            overlay: Overlay {
                width: 4,
                height: 2,
                horizontal_offsets: vec![0, 2],
                vertical_offsets: vec![0, 0],
                ..Default::default()
            },
            ..Default::default()
        };
        for category in [Category::Color, Category::Alpha] {
            for (overlay_depth, tile_depth) in [(8, 10), (10, 8), (10, 12)] {
                let mut overlay_image = create_image(4, 2, overlay_depth, category);
                let tile = create_image(2, 2, tile_depth, category);
                assert!(matches!(
                    overlay_image.copy_and_overlay_from_tile(&tile, &tile_info, 0, category),
                    Err(AvifError::InvalidImageGrid(_))
                ));
            }
            for depth in [8, 10] {
                let mut overlay_image = create_image(4, 2, depth, category);
                let tile = create_image(2, 2, depth, category);
                for tile_index in 0..2 {
                    assert!(overlay_image
                        .copy_and_overlay_from_tile(&tile, &tile_info, tile_index, category)
                        .is_ok());
                }
            }
        }
    }

    // Returns a tile whose samples depend on the tile index, the plane and the sample position.
    fn create_tile(
        yuv_format: PixelFormat,
        depth: u8,
        category: Category,
        width: u32,
        height: u32,
        tile_index: u32,
    ) -> AvifResult<Image> {
        let mut tile = Image {
            width,
            height,
            depth,
            yuv_format,
            ..Default::default()
        };
        tile.allocate_planes(category)?;
        for plane in category.planes() {
            let plane = *plane;
            if !tile.has_plane(plane) {
                continue;
            }
            for y in 0..u32_from_usize(tile.height(plane))? {
                let value = |x: usize| {
                    tile_index as usize * 1009 + plane.as_usize() * 101 + y as usize * 31 + x * 7
                };
                if depth == 8 {
                    for (x, sample) in tile.row_mut(plane, y)?.iter_mut().enumerate() {
                        *sample = (value(x) % 251) as u8;
                    }
                } else {
                    for (x, sample) in tile.row16_mut(plane, y)?.iter_mut().enumerate() {
                        *sample = (value(x) % 1021) as u16;
                    }
                }
            }
        }
        Ok(tile)
    }

    // Copies all tiles to an image of the given dimensions with the given crop_rect.
    fn copy_tiles(
        tiles: &[Image],
        grid: &Grid,
        yuv_format: PixelFormat,
        category: Category,
        (width, height): (u32, u32),
        crop_rect: Option<&CropRect>,
    ) -> AvifResult<Image> {
        let mut image = Image {
            width,
            height,
            depth: tiles[0].depth,
            yuv_format,
            ..Default::default()
        };
        image.allocate_planes(category)?;
        for (tile_index, tile) in tiles.iter().enumerate() {
            image.copy_from_tile(tile, grid, u32_from_usize(tile_index)?, category, crop_rect)?;
        }
        Ok(image)
    }

    fn assert_same_planes(image1: &Image, image2: &Image, category: Category) -> AvifResult<()> {
        assert_eq!((image1.width, image1.height), (image2.width, image2.height));
        for plane in category.planes() {
            let plane = *plane;
            assert_eq!(image1.has_plane(plane), image2.has_plane(plane));
            if !image1.has_plane(plane) {
                continue;
            }
            for y in 0..u32_from_usize(image1.height(plane))? {
                if image1.depth == 8 {
                    assert_eq!(image1.row(plane, y)?, image2.row(plane, y)?);
                } else {
                    assert_eq!(image1.row16(plane, y)?, image2.row16(plane, y)?);
                }
            }
        }
        Ok(())
    }

    #[test_case(PixelFormat::Yuv444, 8, Category::Color, 6, 4)]
    #[test_case(PixelFormat::Yuv444, 10, Category::Color, 5, 3)]
    #[test_case(PixelFormat::Yuv422, 8, Category::Color, 6, 3)]
    #[test_case(PixelFormat::Yuv422, 12, Category::Color, 6, 4)]
    #[test_case(PixelFormat::Yuv420, 8, Category::Color, 6, 4)]
    #[test_case(PixelFormat::Yuv420, 10, Category::Color, 8, 6)]
    #[test_case(PixelFormat::Yuv400, 8, Category::Color, 5, 3)]
    #[test_case(PixelFormat::AndroidNv12, 8, Category::Color, 6, 4)]
    #[test_case(PixelFormat::AndroidP010, 16, Category::Color, 6, 4)]
    #[test_case(PixelFormat::Yuv420, 8, Category::Alpha, 5, 3)]
    #[test_case(PixelFormat::Yuv420, 10, Category::Alpha, 6, 4)]
    #[test_case(PixelFormat::AndroidNv12, 8, Category::Alpha, 5, 3)]
    #[test_case(PixelFormat::Yuv444, 8, Category::Gainmap, 6, 4)]
    fn copy_from_tile_with_crop_rect(
        yuv_format: PixelFormat,
        depth: u8,
        category: Category,
        tile_width: u32,
        tile_height: u32,
    ) -> AvifResult<()> {
        // The last column and the last row of cells are trimmed.
        let grid = Grid {
            rows: 2,
            columns: 3,
            width: tile_width * 2 + tile_width / 2 + 1,
            height: tile_height + tile_height / 2 + 1,
        };
        let tile_format =
            if category == Category::Alpha { PixelFormat::Yuv400 } else { yuv_format };
        let tiles = (0..grid.rows * grid.columns)
            .map(|i| create_tile(tile_format, depth, category, tile_width, tile_height, i))
            .collect::<AvifResult<Vec<_>>>()?;
        for x in 0..grid.width {
            for y in 0..grid.height {
                for width in [1, 2, 3, tile_width + 1, grid.width - x] {
                    for height in [1, 2, 3, tile_height + 1, grid.height - y] {
                        let rect = CropRect {
                            x,
                            y,
                            width,
                            height,
                        };
                        if !rect.is_within(grid.width, grid.height) {
                            continue;
                        }
                        let mut expected = copy_tiles(
                            &tiles,
                            &grid,
                            yuv_format,
                            category,
                            (grid.width, grid.height),
                            None,
                        )?;
                        expected.crop_in_place(&rect)?;
                        let cropped = copy_tiles(
                            &tiles,
                            &grid,
                            yuv_format,
                            category,
                            (width, height),
                            Some(&rect),
                        )?;
                        assert_same_planes(&cropped, &expected, category)?;
                    }
                }
            }
        }
        Ok(())
    }

    #[test_case(PixelFormat::Yuv420, 5, 4)]
    #[test_case(PixelFormat::Yuv420, 6, 3)]
    #[test_case(PixelFormat::Yuv422, 5, 4)]
    #[test_case(PixelFormat::AndroidNv12, 5, 4)]
    #[test_case(PixelFormat::AndroidP010, 6, 3)]
    fn copy_from_tile_with_crop_rect_odd_tile(
        yuv_format: PixelFormat,
        tile_width: u32,
        tile_height: u32,
    ) -> AvifResult<()> {
        let grid = Grid {
            rows: 2,
            columns: 2,
            width: tile_width * 2,
            height: tile_height * 2,
        };
        let depth = if yuv_format == PixelFormat::AndroidP010 { 16 } else { 8 };
        let tile = create_tile(
            yuv_format,
            depth,
            Category::Color,
            tile_width,
            tile_height,
            0,
        )?;
        let rect = CropRect {
            x: 0,
            y: 0,
            width: 2,
            height: 2,
        };
        let mut image = Image {
            width: 2,
            height: 2,
            depth,
            yuv_format,
            ..Default::default()
        };
        image.allocate_planes(Category::Color)?;
        assert!(matches!(
            image.copy_from_tile(&tile, &grid, 0, Category::Color, Some(&rect)),
            Err(AvifError::InvalidImageGrid(_))
        ));
        Ok(())
    }

    #[test_case(CropRect { x: 0, y: 0, width: 0, height: 2 }, (0, 2) ; "empty")]
    #[test_case(CropRect { x: 10, y: 0, width: 4, height: 2 }, (4, 2) ; "outside grid width")]
    #[test_case(CropRect { x: 0, y: 6, width: 2, height: 4 }, (2, 4) ; "outside grid height")]
    #[test_case(CropRect { x: 2, y: 2, width: 4, height: 4 }, (6, 4) ; "image too wide")]
    #[test_case(CropRect { x: 2, y: 2, width: 4, height: 4 }, (4, 2) ; "image too short")]
    #[test_case(CropRect { x: 2, y: 2, width: 4, height: 4 }, (12, 8) ; "grid sized image")]
    fn copy_from_tile_with_invalid_crop_rect(
        rect: CropRect,
        (width, height): (u32, u32),
    ) -> AvifResult<()> {
        let grid = Grid {
            rows: 2,
            columns: 2,
            width: 12,
            height: 8,
        };
        let tile = create_tile(PixelFormat::Yuv420, 8, Category::Color, 6, 4, 0)?;
        let mut image = Image {
            width,
            height,
            depth: 8,
            yuv_format: PixelFormat::Yuv420,
            ..Default::default()
        };
        image.allocate_planes(Category::Color)?;
        assert_eq!(
            image.copy_from_tile(&tile, &grid, 0, Category::Color, Some(&rect)),
            Err(AvifError::InvalidArgument)
        );
        Ok(())
    }

    #[test]
    fn map_grid_boundary_invariants() -> AvifResult<()> {
        for align in [1u32, 2] {
            for tile_len in [64u32, 66, 100] {
                for cols in 1u32..=4 {
                    for trim in [0u32, 1, 7, 33] {
                        let grid_len = tile_len * cols - trim.min(tile_len - 1);
                        for crop_x in [0u32, align, tile_len - align] {
                            if crop_x >= grid_len {
                                continue;
                            }
                            let src_len = grid_len - crop_x;
                            for dst_len in [1u32, 2, 3, src_len / 3, src_len / 2, src_len] {
                                if dst_len == 0 || dst_len > src_len {
                                    continue;
                                }
                                let mut seams = vec![0u32];
                                for c in 1..cols {
                                    let raw = c * tile_len;
                                    if raw > crop_x && raw < grid_len {
                                        seams.push(raw - crop_x);
                                    }
                                }
                                seams.push(src_len);

                                let mut mapped = Vec::new();
                                for &s in &seams {
                                    mapped.push(Image::map_grid_boundary(
                                        s, src_len, dst_len, align,
                                    )?);
                                }
                                assert_eq!(mapped[0], 0);
                                assert_eq!(*mapped.last().unwrap(), dst_len);
                                for i in 0..mapped.len() - 1 {
                                    assert!(mapped[i] <= mapped[i + 1]);
                                    let cell_src = seams[i + 1] - seams[i];
                                    let cell_dst = mapped[i + 1] - mapped[i];
                                    assert!(cell_dst <= cell_src);
                                    if i + 1 < mapped.len() - 1 {
                                        assert_eq!(mapped[i + 1] % align, 0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(
            Image::map_grid_boundary(u32::MAX - 1, u32::MAX, u32::MAX - 2, 2)?,
            u32::MAX - 3
        );
        Ok(())
    }

    #[cfg(feature = "libyuv")]
    #[test_case(PixelFormat::Yuv444, 8, Category::Color, 8, 6)]
    #[test_case(PixelFormat::Yuv422, 10, Category::Color, 8, 6)]
    #[test_case(PixelFormat::Yuv420, 8, Category::Color, 8, 6)]
    #[test_case(PixelFormat::Yuv420, 12, Category::Color, 8, 6)]
    #[test_case(PixelFormat::Yuv400, 8, Category::Color, 7, 5)]
    #[test_case(PixelFormat::AndroidNv12, 8, Category::Color, 8, 6)]
    #[test_case(PixelFormat::AndroidP010, 16, Category::Color, 8, 6)]
    #[test_case(PixelFormat::Yuv420, 8, Category::Alpha, 8, 6)]
    #[test_case(PixelFormat::Yuv444, 10, Category::Alpha, 7, 5)]
    #[test_case(PixelFormat::AndroidP010, 16, Category::Alpha, 8, 6)]
    fn copy_scaled_from_tile_covers_canvas(
        yuv_format: PixelFormat,
        depth: u8,
        category: Category,
        tile_width: u32,
        tile_height: u32,
    ) -> AvifResult<()> {
        let grid = Grid {
            rows: 2,
            columns: 3,
            width: tile_width * 2 + tile_width / 2 + 1,
            height: tile_height + tile_height / 2 + 1,
        };
        let tile_format = if category == Category::Alpha && yuv_format != PixelFormat::AndroidP010 {
            PixelFormat::Yuv420
        } else {
            yuv_format
        };
        let mut tiles = (0..grid.rows * grid.columns)
            .map(|i| create_tile(tile_format, depth, category, tile_width, tile_height, i))
            .collect::<AvifResult<Vec<_>>>()?;
        if yuv_format == PixelFormat::AndroidP010 {
            for tile in &mut tiles {
                for plane in category.planes() {
                    if !tile.has_plane(*plane) {
                        continue;
                    }
                    for y in 0..u32_from_usize(tile.height(*plane))? {
                        for p in tile.row16_mut(*plane, y)? {
                            *p = (*p % 1000) << 6;
                        }
                    }
                }
            }
        }

        let crop = CropRect {
            x: 2,
            y: 2,
            width: grid.width - 3,
            height: grid.height - 2,
        };
        for opt_crop in [None, Some(&crop)] {
            let (base_w, base_h) = match opt_crop {
                Some(r) => {
                    let aligned = r.align_to_pixel_format(yuv_format);
                    (aligned.width, aligned.height)
                }
                None => (grid.width, grid.height),
            };
            // Identity 1:1 scale must match copy_from_tile bit-exactly.
            let expected_1to1 = copy_tiles(
                &tiles,
                &grid,
                yuv_format,
                category,
                (base_w, base_h),
                opt_crop,
            )?;
            let mut actual_1to1 = Image {
                width: base_w,
                height: base_h,
                depth,
                yuv_format,
                ..Default::default()
            };
            actual_1to1.allocate_planes(category)?;
            for (i, tile) in tiles.iter().enumerate() {
                actual_1to1.copy_scaled_from_tile(
                    tile,
                    &grid,
                    u32_from_usize(i)?,
                    category,
                    opt_crop,
                    ImageSize {
                        width: base_w,
                        height: base_h,
                    },
                )?;
            }
            assert_same_planes(&actual_1to1, &expected_1to1, category)?;

            // Downscaling to various sizes (including 1x1, 2x2, odd sizes) must fill all pixels
            // without error or out-of-bounds access, and convert P010 -> 10-bit Yuv420.
            for &(dst_w, dst_h) in &[
                (1u32, 1u32),
                (2, 2),
                (3, 3),
                (base_w / 2, base_h / 2),
                (base_w - 1, base_h - 1),
            ] {
                if dst_w == 0 || dst_h == 0 || dst_w > base_w || dst_h > base_h {
                    continue;
                }
                let (dst_depth, dst_fmt) = if yuv_format == PixelFormat::AndroidP010 {
                    (10, PixelFormat::Yuv420)
                } else {
                    (depth, yuv_format)
                };
                let mut dst = Image {
                    width: dst_w,
                    height: dst_h,
                    depth: dst_depth,
                    yuv_format: dst_fmt,
                    ..Default::default()
                };
                let sentinel: u16 = if dst_depth == 8 { 255 } else { 4095 };
                dst.allocate_planes_with_default_values(category, [sentinel; 4])?;
                for (i, tile) in tiles.iter().enumerate() {
                    dst.copy_scaled_from_tile(
                        tile,
                        &grid,
                        u32_from_usize(i)?,
                        category,
                        opt_crop,
                        ImageSize {
                            width: dst_w,
                            height: dst_h,
                        },
                    )?;
                }
                for &plane in category.planes() {
                    if !dst.has_plane(plane) {
                        continue;
                    }
                    for y in 0..u32_from_usize(dst.height(plane))? {
                        if dst.depth == 8 {
                            assert!(dst.row(plane, y)?.iter().all(|&v| v != sentinel as u8));
                        } else {
                            assert!(dst.row16(plane, y)?.iter().all(|&v| v != sentinel));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn copy_scaled_from_tile_edge_validations() -> AvifResult<()> {
        let grid = Grid {
            rows: 2,
            columns: 2,
            width: 8,
            height: 8,
        };
        let tile = create_tile(PixelFormat::Yuv420, 8, Category::Color, 4, 4, 0)?;
        let mut dst = Image {
            width: 4,
            height: 4,
            depth: 8,
            yuv_format: PixelFormat::Yuv420,
            ..Default::default()
        };
        dst.allocate_planes(Category::Color)?;

        // Out-of-bounds tile index and empty grid dimensions.
        assert!(matches!(
            dst.copy_scaled_from_tile(
                &tile,
                &grid,
                4,
                Category::Color,
                None,
                ImageSize {
                    width: 4,
                    height: 4
                },
            ),
            Err(AvifError::InvalidImageGrid(_))
        ));
        for bad_grid in [
            Grid {
                rows: 0,
                columns: 2,
                width: 8,
                height: 8,
            },
            Grid {
                rows: 2,
                columns: 0,
                width: 8,
                height: 8,
            },
        ] {
            assert!(matches!(
                dst.copy_scaled_from_tile(
                    &tile,
                    &bad_grid,
                    0,
                    Category::Color,
                    None,
                    ImageSize {
                        width: 4,
                        height: 4
                    },
                ),
                Err(AvifError::InvalidImageGrid(_))
            ));
        }

        // Invalid scaled_size (zero, upscaling, or mismatched with destination dimensions).
        for bad_size in [
            ImageSize {
                width: 0,
                height: 4,
            },
            ImageSize {
                width: 10,
                height: 4,
            },
            ImageSize {
                width: 2,
                height: 2,
            },
        ] {
            assert_eq!(
                dst.copy_scaled_from_tile(&tile, &grid, 0, Category::Color, None, bad_size),
                Err(AvifError::InvalidArgument)
            );
        }

        // Downscaling AndroidP010 directly into an AndroidP010 destination is rejected.
        let p010_tile = create_tile(PixelFormat::AndroidP010, 16, Category::Color, 4, 4, 0)?;
        let mut p010_dst = Image {
            width: 4,
            height: 4,
            depth: 16,
            yuv_format: PixelFormat::AndroidP010,
            ..Default::default()
        };
        p010_dst.allocate_planes(Category::Color)?;
        assert!(matches!(
            p010_dst.copy_scaled_from_tile(
                &p010_tile,
                &grid,
                0,
                Category::Color,
                None,
                ImageSize {
                    width: 4,
                    height: 4
                },
            ),
            Err(AvifError::InvalidImageGrid(_))
        ));

        // Odd Alpha tile dimensions on a chroma-subsampled destination image are rejected
        // (decoder falls back to full-grid alpha decode + scale in that case).
        let odd_alpha_tile = create_tile(PixelFormat::Yuv400, 8, Category::Alpha, 5, 4, 0)?;
        let mut alpha_dst = Image {
            width: 4,
            height: 4,
            depth: 8,
            yuv_format: PixelFormat::Yuv420,
            ..Default::default()
        };
        alpha_dst.allocate_planes(Category::Alpha)?;
        assert!(matches!(
            alpha_dst.copy_scaled_from_tile(
                &odd_alpha_tile,
                &Grid {
                    rows: 2,
                    columns: 2,
                    width: 10,
                    height: 8,
                },
                0,
                Category::Alpha,
                None,
                ImageSize {
                    width: 4,
                    height: 4
                },
            ),
            Err(AvifError::InvalidImageGrid(_))
        ));
        Ok(())
    }
}
