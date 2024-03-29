use crate::decoder::*;
use crate::*;

pub const MAX_AV1_LAYER_COUNT: usize = 4;

#[derive(Debug, Default)]
pub struct DecodeSample {
    pub item_id: u32,
    pub offset: u64,
    pub size: usize,
    pub spatial_id: u8,
    pub sync: bool,
}

impl DecodeSample {
    pub fn partial_data<'a>(
        &'a self,
        io: &'a mut Box<impl decoder::IO + ?Sized>,
        buffer: &'a Option<Vec<u8>>,
        size: usize,
    ) -> AvifResult<&[u8]> {
        match buffer {
            Some(x) => {
                let start_offset = usize_from_u64(self.offset)?;
                let end_offset = start_offset + size;
                Ok(&x[start_offset..end_offset])
            }
            None => io.read(self.offset, size),
        }
    }

    pub fn data<'a>(
        &'a self,
        io: &'a mut Box<impl decoder::IO + ?Sized>,
        buffer: &'a Option<Vec<u8>>,
    ) -> AvifResult<&[u8]> {
        self.partial_data(io, buffer, self.size)
    }
}

#[derive(Debug, Default)]
pub struct DecodeInput {
    pub samples: Vec<DecodeSample>,
    pub all_layers: bool,
    pub category: Category,
}

#[derive(Debug, Default, Copy, Clone)]
pub struct Grid {
    pub rows: u32,
    pub columns: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Default)]
pub struct TileInfo {
    pub tile_count: u32,
    pub decoded_tile_count: u32,
    pub grid: Grid,
}

impl TileInfo {
    pub fn is_grid(&self) -> bool {
        self.grid.rows > 0 && self.grid.columns > 0
    }

    pub fn grid_tile_count(&self) -> u32 {
        if self.is_grid() {
            self.grid.rows * self.grid.columns
        } else {
            1
        }
    }

    pub fn decoded_row_count(&self, image_height: u32, tile_height: u32) -> u32 {
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

    pub fn is_fully_decoded(&self) -> bool {
        self.tile_count == self.decoded_tile_count
    }
}

#[derive(Default)]
pub struct Tile {
    pub width: u32,
    pub height: u32,
    pub operating_point: u8,
    pub image: Image,
    pub input: DecodeInput,
    pub codec_index: usize,
}

impl Tile {
    pub fn create_from_item(
        item: &mut Item,
        allow_progressive: bool,
        image_count_limit: u32,
    ) -> AvifResult<Tile> {
        let mut tile = Tile {
            width: item.width,
            height: item.height,
            operating_point: item.operating_point(),
            image: Image::default(),
            ..Tile::default()
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
        }
        let lsel;
        let has_lsel;
        match item.lsel() {
            Some(x) => {
                lsel = x;
                has_lsel = true;
            }
            None => {
                lsel = 0;
                has_lsel = false;
            }
        }
        // Progressive images offer layers via the a1lxProp, but don't specify a layer selection with
        // lsel.
        item.progressive = has_a1lx && (!has_lsel || lsel == 0xFFFF);
        let base_item_offset = if item.extents.len() == 1 { item.extents[0].offset } else { 0 };
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
                    println!("lsel layer index not found in a1lx.");
                    return Err(AvifError::InvalidImageGrid);
                }
                let layer_id_plus_1 = layer_id.checked_add(1).ok_or(AvifError::BmffParseFailed)?;
                for layer_size in layer_sizes.iter().take(layer_id_plus_1) {
                    sample_size += layer_size;
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
            tile.input.samples.push(sample);
        } else if item.progressive && allow_progressive {
            // Progressive image. Decode all layers and expose them all to the
            // user.
            if image_count_limit != 0 && layer_count as u32 > image_count_limit {
                println!("exceeded image_count_limit (progressive)");
                return Err(AvifError::BmffParseFailed);
            }
            tile.input.all_layers = true;
            let mut offset = 0;
            for (i, layer_size) in layer_sizes.iter().take(layer_count).enumerate() {
                let sample = DecodeSample {
                    item_id: item.id,
                    offset: base_item_offset + offset,
                    size: *layer_size,
                    spatial_id: 0xff,
                    sync: i == 0, // Assume all layers depend on the first layer.
                };
                tile.input.samples.push(sample);
                offset += *layer_size as u64;
            }
        } else {
            // Typical case: Use the entire item's payload for a single frame output
            let sample = DecodeSample {
                item_id: item.id,
                offset: base_item_offset,
                size: item.size,
                // Legal spatial_id values are [0,1,2,3], so this serves as a sentinel value for
                // "do not filter by spatial_id"
                spatial_id: 0xff,
                sync: true,
            };
            tile.input.samples.push(sample);
        }
        Ok(tile)
    }

    pub fn create_from_track(track: &Track, mut image_count_limit: u32) -> AvifResult<Tile> {
        let mut tile = Tile {
            width: track.width,
            height: track.height,
            operating_point: 0, // No way to set operating point via tracks
            ..Tile::default()
        };
        let sample_table = &track.sample_table.unwrap_ref();

        if image_count_limit != 0 {
            for (chunk_index, _chunk_offset) in sample_table.chunk_offsets.iter().enumerate() {
                // Figure out how many samples are in this chunk.
                let sample_count = sample_table.get_sample_count_of_chunk(chunk_index as u32);
                if sample_count == 0 {
                    println!("chunk with 0 samples found");
                    return Err(AvifError::BmffParseFailed);
                }
                if sample_count > image_count_limit {
                    println!("exceeded image_count_limit");
                    return Err(AvifError::BmffParseFailed);
                }
                image_count_limit -= sample_count;
            }
        }

        let mut sample_size_index: usize = 0;
        for (chunk_index, chunk_offset) in sample_table.chunk_offsets.iter().enumerate() {
            // Figure out how many samples are in this chunk.
            let sample_count = sample_table.get_sample_count_of_chunk(chunk_index as u32);
            if sample_count == 0 {
                println!("chunk with 0 samples found");
                return Err(AvifError::BmffParseFailed);
            }

            let mut sample_offset = *chunk_offset;
            for _ in 0..sample_count {
                let sample_size = sample_table.sample_size(sample_size_index)?;
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
                tile.input.samples.push(sample);
                sample_offset = sample_offset
                    .checked_add(sample_size as u64)
                    .ok_or(AvifError::BmffParseFailed)?;
                sample_size_index += 1;
            }
        }
        for sync_sample_number in &sample_table.sync_samples {
            // sample_table.sync_samples is 1-based.
            let index: usize = (*sync_sample_number - 1) as usize;
            if index < tile.input.samples.len() {
                tile.input.samples[index].sync = true;
            }
        }
        Ok(tile)
    }
}
