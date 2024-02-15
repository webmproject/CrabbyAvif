use crate::decoder::tile::TileInfo;
use crate::decoder::Category;
use crate::decoder::ProgressiveState;
use crate::internal_utils::pixels::*;
use crate::internal_utils::*;
use crate::parser::mp4box::*;
use crate::utils::clap::CleanAperture;
use crate::*;

#[derive(PartialEq, Copy, Clone, Debug)]
pub enum Plane {
    Y = 0,
    U = 1,
    V = 2,
    A = 3,
}

impl From<usize> for Plane {
    fn from(plane: usize) -> Self {
        match plane {
            1 => Plane::U,
            2 => Plane::V,
            3 => Plane::A,
            _ => Plane::Y,
        }
    }
}

impl Plane {
    pub fn to_usize(&self) -> usize {
        match self {
            Plane::Y => 0,
            Plane::U => 1,
            Plane::V => 2,
            Plane::A => 3,
        }
    }
}

/// cbindgen:ignore
pub const MAX_PLANE_COUNT: usize = 4;
pub const YUV_PLANES: [Plane; 3] = [Plane::Y, Plane::U, Plane::V];
pub const A_PLANE: [Plane; 1] = [Plane::A];
pub const ALL_PLANES: [Plane; MAX_PLANE_COUNT] = [Plane::Y, Plane::U, Plane::V, Plane::A];

#[derive(Default)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub depth: u8,

    pub yuv_format: PixelFormat,
    pub full_range: bool,
    pub chroma_sample_position: ChromaSamplePosition,

    pub alpha_present: bool,
    pub alpha_premultiplied: bool,

    pub row_bytes: [u32; MAX_PLANE_COUNT],
    pub image_owns_planes: [bool; MAX_PLANE_COUNT],

    pub planes2: [Option<Pixels>; MAX_PLANE_COUNT],

    pub color_primaries: ColorPrimaries,
    pub transfer_characteristics: TransferCharacteristics,
    pub matrix_coefficients: MatrixCoefficients,

    pub clli: Option<ContentLightLevelInformation>,
    pub pasp: Option<PixelAspectRatio>,
    pub clap: Option<CleanAperture>,
    pub irot_angle: Option<u8>,
    pub imir_axis: Option<u8>,

    pub exif: Vec<u8>,
    pub icc: Vec<u8>,
    pub xmp: Vec<u8>,

    pub image_sequence_track_present: bool,
    pub progressive_state: ProgressiveState,
    // TODO: gainmap image ?
}

pub struct PlaneData<'a> {
    pub data: Option<&'a [u8]>,
    pub data16: Option<&'a [u16]>,
    pub width: u32,
    pub height: u32,
    pub row_bytes: u32,
    pub pixel_size: u32,
}

// TODO: unify this into the struct above with an enum for mut/const.
pub struct PlaneMutData<'a> {
    pub data: Option<&'a mut [u8]>,
    pub data16: Option<&'a mut [u16]>,
    pub width: u32,
    pub height: u32,
    pub row_bytes: u32,
    pub pixel_size: u32,
}

impl Image {
    pub fn depth_valid(&self) -> bool {
        matches!(self.depth, 8 | 10 | 12 | 16)
    }

    pub fn max_channel(&self) -> u16 {
        ((1i32 << self.depth) - 1) as u16
    }

    pub fn max_channel_f(&self) -> f32 {
        self.max_channel() as f32
    }

    pub fn has_plane(&self, plane: Plane) -> bool {
        let plane_index = plane.to_usize();
        if self.planes2[plane_index].is_none() || self.row_bytes[plane_index] == 0 {
            return false;
        }
        self.planes2[plane_index].as_ref().unwrap().has_data()
    }

    pub fn has_alpha(&self) -> bool {
        self.has_plane(Plane::A)
    }

    pub fn subsampled_width(&self, width: u32, plane: Plane) -> usize {
        match plane {
            Plane::Y | Plane::A => width as usize,
            _ => match self.yuv_format {
                PixelFormat::Yuv444 | PixelFormat::Monochrome => width as usize,
                PixelFormat::Yuv420 | PixelFormat::Yuv422 => (width as usize + 1) / 2,
            },
        }
    }

    pub fn width(&self, plane: Plane) -> usize {
        self.subsampled_width(self.width, plane)
    }

    pub fn height(&self, plane: Plane) -> usize {
        match plane {
            Plane::Y | Plane::A => self.height as usize,
            _ => match self.yuv_format {
                PixelFormat::Yuv444 | PixelFormat::Monochrome | PixelFormat::Yuv422 => {
                    self.height as usize
                }
                PixelFormat::Yuv420 => (self.height as usize + 1) / 2,
            },
        }
    }

    pub fn plane(&self, plane: Plane) -> Option<PlaneData> {
        if !self.has_plane(plane) {
            return None;
        }
        let plane_index = plane.to_usize();
        let pixel_size = if self.depth == 8 { 1 } else { 2 };
        let height = self.height(plane);
        let row_bytes = self.row_bytes[plane_index] as usize;
        let plane_size = height * row_bytes;
        let planes2 = self.planes2[plane_index].as_ref().unwrap();
        let (data, data16) = planes2.slices(0, plane_size as u32).unwrap();
        Some(PlaneData {
            data,
            data16,
            width: self.width(plane) as u32,
            height: height as u32,
            row_bytes: row_bytes as u32,
            pixel_size,
        })
    }

    pub fn plane_mut(&mut self, plane: Plane) -> Option<PlaneMutData> {
        if !self.has_plane(plane) {
            return None;
        }
        let plane_index = plane.to_usize();
        let pixel_size = if self.depth == 8 { 1 } else { 2 };
        let height = self.height(plane);
        let width = self.width(plane) as u32;
        let row_bytes = self.row_bytes[plane_index] as usize;
        let plane_size = height * row_bytes;
        let planes2 = self.planes2[plane_index].as_mut().unwrap();
        let (data, data16) = planes2.slices_mut(0, plane_size as u32).unwrap();
        Some(PlaneMutData {
            data,
            data16,
            width,
            height: height as u32,
            row_bytes: row_bytes as u32,
            pixel_size,
        })
    }

    pub fn row(&self, plane: Plane, row: u32) -> AvifResult<&[u8]> {
        let plane = self.plane(plane).ok_or(AvifError::NoContent)?;
        let row_bytes = usize_from_u32(plane.row_bytes)?;
        let start = usize_from_u32(row * plane.row_bytes)?;
        let end = start + row_bytes;
        Ok(&plane.data.ok_or(AvifError::NoContent)?[start..end])
    }

    pub fn row_mut(&mut self, plane: Plane, row: u32) -> AvifResult<&mut [u8]> {
        let plane = self.plane_mut(plane).ok_or(AvifError::NoContent)?;
        let row_bytes = usize_from_u32(plane.row_bytes)?;
        let start = usize_from_u32(row * plane.row_bytes)?;
        let end = start + row_bytes;
        Ok(&mut plane.data.ok_or(AvifError::NoContent)?[start..end])
    }

    pub fn row16(&self, plane: Plane, row: u32) -> AvifResult<&[u16]> {
        let plane = self.plane(plane).ok_or(AvifError::NoContent)?;
        let row_bytes = usize_from_u32(plane.row_bytes)? / 2;
        let start = usize_from_u32(row * plane.row_bytes / 2)?;
        let end = start + row_bytes;
        Ok(&plane.data16.ok_or(AvifError::NoContent)?[start..end])
    }

    pub fn row16_mut(&mut self, plane: Plane, row: u32) -> AvifResult<&mut [u16]> {
        let plane = self.plane_mut(plane).ok_or(AvifError::NoContent)?;
        let row_bytes = usize_from_u32(plane.row_bytes)? / 2;
        let start = usize_from_u32(row * plane.row_bytes / 2)?;
        let end = start + row_bytes;
        Ok(&mut plane.data16.ok_or(AvifError::NoContent)?[start..end])
    }

    pub fn allocate_planes(&mut self, category: Category) -> AvifResult<()> {
        let pixel_size: usize = if self.depth == 8 { 1 } else { 2 };
        for plane in category.planes() {
            let plane = *plane;
            let plane_index = plane.to_usize();
            let width = self.width(plane);
            let plane_size = width * self.height(plane);
            let default_value =
                if plane == Plane::A { ((1i32 << self.depth) - 1) as u16 } else { 0 };
            if self.planes2[plane_index].is_some()
                && self.planes2[plane_index].as_ref().unwrap().size() == plane_size
            {
                // TODO: need to memset to 0 maybe?
                continue;
            }
            if self.planes2[plane_index].is_none()
                || self.planes2[plane_index].as_ref().unwrap().is_pointer()
            {
                self.planes2[plane_index] = Some(if self.depth == 8 {
                    Pixels::Buffer(Vec::new())
                } else {
                    Pixels::Buffer16(Vec::new())
                });
            }
            let pixels = self.planes2[plane_index].as_mut().unwrap();
            pixels.resize(plane_size, default_value)?;
            self.row_bytes[plane_index] = u32_from_usize(width * pixel_size)?;
            self.image_owns_planes[plane_index] = true;
        }
        Ok(())
    }

    pub fn steal_from(&mut self, src: &Image, category: Category) {
        // This function is used only when both src and self contains only pointers.
        match category {
            Category::Alpha => {
                if src.planes2[3].is_some() {
                    self.planes2[3] =
                        Some(Pixels::Pointer(src.planes2[3].as_ref().unwrap().pointer()));
                }
                self.row_bytes[3] = src.row_bytes[3];
            }
            _ => {
                if src.planes2[0].is_some() {
                    self.planes2[0] =
                        Some(Pixels::Pointer(src.planes2[0].as_ref().unwrap().pointer()));
                }
                if src.planes2[1].is_some() {
                    self.planes2[1] =
                        Some(Pixels::Pointer(src.planes2[1].as_ref().unwrap().pointer()));
                }
                if src.planes2[2].is_some() {
                    self.planes2[2] =
                        Some(Pixels::Pointer(src.planes2[2].as_ref().unwrap().pointer()));
                }
                self.row_bytes[0] = src.row_bytes[0];
                self.row_bytes[1] = src.row_bytes[1];
                self.row_bytes[2] = src.row_bytes[2];
            }
        }
    }

    pub fn copy_from_tile(
        &mut self,
        tile: &Image,
        tile_info: &TileInfo,
        tile_index: u32,
        category: Category,
    ) -> AvifResult<()> {
        // This function is used only when |tile| contains pointers and self contains buffers.
        let err = AvifError::BmffParseFailed;
        let row_index = u64::from(tile_index / tile_info.grid.columns);
        let column_index = u64::from(tile_index % tile_info.grid.columns);
        //println!("copying tile {tile_index} {row_index} {column_index}");
        for plane in category.planes() {
            let plane = *plane;
            let src_plane = tile.plane(plane);
            if src_plane.is_none() {
                continue;
            }
            let src_plane = src_plane.unwrap();
            // If this is the last tile column, clamp to left over width.
            let src_width_to_copy = if column_index == (tile_info.grid.columns - 1).into() {
                let width_so_far = u64::from(src_plane.width)
                    .checked_mul(column_index)
                    .ok_or(err)?;
                u64_from_usize(self.width(plane))?
                    .checked_sub(width_so_far)
                    .ok_or(err)?
            } else {
                u64::from(src_plane.width)
            };
            let src_width_to_copy = usize_from_u64(src_width_to_copy)?;

            // If this is the last tile row, clamp to left over height.
            let src_height_to_copy = if row_index == (tile_info.grid.rows - 1).into() {
                let height_so_far = u64::from(src_plane.height)
                    .checked_mul(row_index)
                    .ok_or(err)?;
                u64_from_usize(self.height(plane))?
                    .checked_sub(height_so_far)
                    .ok_or(err)?
            } else {
                u64::from(src_plane.height)
            };

            let dst_y_start = row_index * u64::from(src_plane.height);
            let dst_x_offset = usize_from_u64(column_index * u64::from(src_plane.width))?;
            // TODO: src_height_to_copy can just be u32?
            if self.depth == 8 {
                for y in 0..src_height_to_copy {
                    let src_row = tile.row(plane, u32_from_u64(y)?)?;
                    let src_slice = &src_row[0..src_width_to_copy];
                    let dst_row = self.row_mut(plane, u32_from_u64(dst_y_start + y)?)?;
                    let dst_slice = &mut dst_row[dst_x_offset..dst_x_offset + src_width_to_copy];
                    dst_slice.copy_from_slice(src_slice);
                }
            } else {
                for y in 0..src_height_to_copy {
                    let src_row = tile.row16(plane, u32_from_u64(y)?)?;
                    let src_slice = &src_row[0..src_width_to_copy];
                    let dst_row = self.row16_mut(plane, u32_from_u64(dst_y_start + y)?)?;
                    let dst_slice = &mut dst_row[dst_x_offset..dst_x_offset + src_width_to_copy];
                    dst_slice.copy_from_slice(src_slice);
                }
            }
        }
        Ok(())
    }
}
