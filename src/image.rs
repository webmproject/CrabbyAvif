use crate::decoder::tile::TileInfo;
use crate::decoder::ProgressiveState;
use crate::internal_utils::*;
use crate::*;

// TODO: needed only for debug to Image and PlaneData. Can be removed it those
// do not have to be debug printable.
use derivative::Derivative;

#[derive(Derivative, Default)]
#[derivative(Debug)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub depth: u8,

    pub yuv_format: PixelFormat,
    pub full_range: bool,
    pub chroma_sample_position: ChromaSamplePosition,

    pub alpha_present: bool,
    pub alpha_premultiplied: bool,

    #[derivative(Debug = "ignore")]
    pub exif: Vec<u8>,
    #[derivative(Debug = "ignore")]
    pub icc: Vec<u8>,
    #[derivative(Debug = "ignore")]
    pub xmp: Vec<u8>,

    pub color_primaries: ColorPrimaries,
    pub transfer_characteristics: TransferCharacteristics,
    pub matrix_coefficients: MatrixCoefficients,

    // TODO: these can go in a "global" image info struct. which can then
    // contain an ImageInfo as well.
    pub image_sequence_track_present: bool,

    pub progressive_state: ProgressiveState,
}

impl ImageInfo {
    // TODO: replace plane_index with an enum.
    pub fn height(&self, plane_index: usize) -> usize {
        assert!(plane_index <= 3);
        if plane_index == 0 || plane_index == 3 {
            // Y and Alpha planes are never subsampled.
            return self.height as usize;
        }
        match self.yuv_format {
            PixelFormat::Yuv444 | PixelFormat::Yuv422 | PixelFormat::Monochrome => {
                self.height as usize
            }
            PixelFormat::Yuv420 => (self.height as usize + 1) / 2,
        }
    }

    pub fn width(&self, plane_index: usize) -> usize {
        assert!(plane_index <= 3);
        if plane_index == 0 || plane_index == 3 {
            // Y and Alpha planes are never subsampled.
            return self.width as usize;
        }
        match self.yuv_format {
            PixelFormat::Yuv444 | PixelFormat::Monochrome => self.width as usize,
            PixelFormat::Yuv420 | PixelFormat::Yuv422 => (self.width as usize + 1) / 2,
        }
    }

    pub fn subsampled_value(&self, value: u32, plane_index: usize) -> usize {
        assert!(plane_index <= 3);
        if plane_index == 0 || plane_index == 3 {
            // Y and Alpha planes are never subsampled.
            return value as usize;
        }
        match self.yuv_format {
            PixelFormat::Yuv444 | PixelFormat::Monochrome => value as usize,
            PixelFormat::Yuv420 | PixelFormat::Yuv422 => (value as usize + 1) / 2,
        }
    }
}

#[derive(Derivative, Default)]
#[derivative(Debug)]
pub struct Image {
    pub info: ImageInfo,

    pub planes: [Option<*const u8>; 4],
    pub row_bytes: [u32; 4], // TODO: named constant
    pub image_owns_planes: bool,
    pub image_owns_alpha_plane: bool,

    // some more boxes. clli, transformations. pasp, clap, irot, imir.

    // exif, xmp.

    // gainmap.
    #[derivative(Debug = "ignore")]
    plane_buffers: [Vec<u8>; 4],
}

#[derive(Derivative)]
#[derivative(Debug)]
pub struct PlaneData<'a> {
    #[derivative(Debug = "ignore")]
    pub data: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub row_bytes: u32,
    pub pixel_size: u32,
}

impl Image {
    pub fn plane(&self, plane: usize) -> Option<PlaneData> {
        assert!(plane < 4);
        self.planes[plane]?;
        let pixel_size = if self.info.depth == 8 { 1 } else { 2 };
        let height = self.info.height(plane);
        let row_bytes = self.row_bytes[plane] as usize;
        let plane_size = height * row_bytes;
        let data = unsafe { std::slice::from_raw_parts(self.planes[plane].unwrap(), plane_size) };
        Some(PlaneData {
            data,
            width: self.info.width(plane) as u32,
            height: height as u32,
            row_bytes: row_bytes as u32,
            pixel_size,
        })
    }

    pub fn allocate_planes(&mut self, category: usize) -> AvifResult<()> {
        // TODO : assumes 444. do other stuff.
        // TODO: do not realloc if size is already big enough.
        let pixel_size: u32 = if self.info.depth == 8 { 1 } else { 2 };
        let plane_size = (self.info.width * self.info.height * pixel_size) as usize;
        if category == 0 || category == 2 {
            for plane_index in 0usize..3 {
                self.plane_buffers[plane_index].reserve(plane_size);
                self.plane_buffers[plane_index].resize(plane_size, 0);
                self.row_bytes[plane_index] = self.info.width * pixel_size;
                self.planes[plane_index] = Some(self.plane_buffers[plane_index].as_ptr());
            }
            self.image_owns_planes = true;
        } else {
            assert!(category == 1);
            self.plane_buffers[3].reserve(plane_size);
            self.plane_buffers[3].resize(plane_size, 255);
            self.row_bytes[3] = self.info.width * pixel_size;
            self.planes[3] = Some(self.plane_buffers[3].as_ptr());
            self.image_owns_alpha_plane = true;
        }
        Ok(())
    }

    pub fn copy_from_slice(
        &mut self,
        source: &[u8],
        stride: u32,
        category: usize,
    ) -> AvifResult<()> {
        // TODO: deal with integer math safety in this function.
        self.allocate_planes(category)?;
        let pixel_size: usize = if self.info.depth == 8 { 1 } else { 2 };
        // TODO: if width == stride, the whole thing can be one copy.
        if category == 0 || category == 2 {
            let mut src_offset = 0;
            for plane_index in 0usize..3 {
                let plane_stride = self.info.subsampled_value(stride, plane_index);
                let width = self.info.width(plane_index);
                let height = self.info.height(plane_index);
                let row_width = width * pixel_size;
                let mut dst_offset = 0;
                for _y in 0usize..height {
                    let src_y_start = src_offset;
                    let src_y_end = src_y_start + row_width;
                    let src_slice = &source[src_y_start..src_y_end];

                    let dst_y_start = dst_offset;
                    let dst_y_end = dst_y_start + row_width;
                    let dst_slice = &mut self.plane_buffers[plane_index][dst_y_start..dst_y_end];

                    dst_slice.copy_from_slice(src_slice);

                    // TODO: does plane_stride account for pixel size?
                    src_offset += plane_stride;
                    dst_offset += self.row_bytes[plane_index] as usize;
                }
            }
        } else {
            assert!(category == 1);
            let mut src_offset = 0;
            let width = self.info.width(3);
            let height = self.info.height(3);
            let row_width = width * pixel_size;
            let mut dst_offset = 0;
            for _y in 0usize..height {
                let src_y_start = src_offset;
                let src_y_end = src_y_start + row_width;
                let src_slice = &source[src_y_start..src_y_end];

                let dst_y_start = dst_offset;
                let dst_y_end = dst_y_start + row_width;
                let dst_slice = &mut self.plane_buffers[3][dst_y_start..dst_y_end];

                dst_slice.copy_from_slice(src_slice);

                // TODO: does stride account for pixel size?
                src_offset += stride as usize;
                dst_offset += self.row_bytes[3] as usize;
            }
        }
        Ok(())
    }

    pub fn steal_from(&mut self, src: &Image, category: usize) {
        match category {
            0 | 2 => {
                self.planes[0] = src.planes[0];
                self.planes[1] = src.planes[1];
                self.planes[2] = src.planes[2];
                self.row_bytes[0] = src.row_bytes[0];
                self.row_bytes[1] = src.row_bytes[1];
                self.row_bytes[2] = src.row_bytes[2];
            }
            1 => {
                self.planes[3] = src.planes[3];
                self.row_bytes[3] = src.row_bytes[3];
            }
            _ => {
                panic!("invalid category in steal planes");
            }
        }
    }

    pub fn copy_from_tile(
        &mut self,
        tile: &Image,
        tile_info: &TileInfo,
        tile_index: u32,
        category: usize,
    ) -> AvifResult<()> {
        let err = AvifError::BmffParseFailed;
        let row_index = u64::from(tile_index / tile_info.grid.columns);
        let column_index = u64::from(tile_index % tile_info.grid.columns);
        //println!("copying tile {tile_index} {row_index} {column_index}");

        let plane_range = if category == 1 { 3usize..4 } else { 0usize..3 };
        for plane_index in plane_range {
            //println!("plane_index {plane_index}");
            let src_plane = tile.plane(plane_index);
            if src_plane.is_none() {
                continue;
            }
            let src_plane = src_plane.unwrap();
            // If this is the last tile column, clamp to left over width.
            let src_width_to_copy = if column_index == (tile_info.grid.columns - 1).into() {
                let width_so_far = u64::from(src_plane.width)
                    .checked_mul(column_index)
                    .ok_or(err)?;
                u64_from_usize(self.info.width(plane_index))?
                    .checked_sub(width_so_far)
                    .ok_or(err)?
            } else {
                u64::from(src_plane.width)
            };
            //println!("src_width_to_copy: {src_width_to_copy}");
            let src_byte_count = src_width_to_copy * u64::from(src_plane.pixel_size);
            let dst_row_bytes = u64::from(self.row_bytes[plane_index]);
            let dst_base_offset = (row_index * (u64::from(src_plane.height) * dst_row_bytes))
                + (column_index * u64::from(src_plane.width * src_plane.pixel_size));
            //println!("dst base_offset: {dst_base_offset}");

            // If this is the last tile row, clamp to left over height.
            let src_height_to_copy = if row_index == (tile_info.grid.rows - 1).into() {
                let height_so_far = u64::from(src_plane.height)
                    .checked_mul(row_index)
                    .ok_or(err)?;
                u64_from_usize(self.info.height(plane_index))?
                    .checked_sub(height_so_far)
                    .ok_or(err)?
            } else {
                u64::from(src_plane.height)
            };

            //println!("src_height_to_copy: {src_height_to_copy}");
            for y in 0..src_height_to_copy {
                let src_stride_offset = y.checked_mul(u64::from(src_plane.row_bytes)).ok_or(err)?;
                let src_end_offset = src_stride_offset.checked_add(src_byte_count).ok_or(err)?;
                let dst_row_offset = y.checked_mul(dst_row_bytes).ok_or(err)?;
                let dst_stride_offset = dst_base_offset.checked_add(dst_row_offset).ok_or(err)?;
                let dst_end_offset = dst_stride_offset.checked_add(src_byte_count).ok_or(err)?;

                let src_slice = &src_plane.data
                    [usize_from_u64(src_stride_offset)?..usize_from_u64(src_end_offset)?];
                let dst_slice = &mut self.plane_buffers[plane_index]
                    [usize_from_u64(dst_stride_offset)?..usize_from_u64(dst_end_offset)?];
                dst_slice.copy_from_slice(src_slice);
            }
        }
        Ok(())
    }
}
