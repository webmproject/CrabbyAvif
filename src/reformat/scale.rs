use crate::decoder::Category;
use crate::image::*;
use crate::internal_utils::*;
use crate::*;

use libyuv_sys::bindings::*;

impl Image {
    pub fn scale(&mut self, width: u32, height: u32) -> AvifResult<()> {
        if self.width == width && self.height == height {
            return Ok(());
        }
        if width == 0 || height == 0 {
            return Err(AvifError::InvalidArgument);
        }
        if (self.planes[0].is_some() && !self.planes[0].as_ref().unwrap().is_pointer())
            || (self.planes[1].is_some() && !self.planes[1].as_ref().unwrap().is_pointer())
            || (self.planes[2].is_some() && !self.planes[2].as_ref().unwrap().is_pointer())
            || (self.planes[3].is_some() && !self.planes[3].as_ref().unwrap().is_pointer())
        {
            // TODO: implement this function for non-pointer inputs.
            return Err(AvifError::NotImplemented);
        }
        let src = image::Image {
            width: self.width,
            height: self.height,
            depth: self.depth,
            yuv_format: self.yuv_format,
            planes: [
                if self.planes[0].is_some() {
                    self.planes[0].as_ref().unwrap().clone_pointer()
                } else {
                    None
                },
                if self.planes[1].is_some() {
                    self.planes[1].as_ref().unwrap().clone_pointer()
                } else {
                    None
                },
                if self.planes[2].is_some() {
                    self.planes[2].as_ref().unwrap().clone_pointer()
                } else {
                    None
                },
                if self.planes[3].is_some() {
                    self.planes[3].as_ref().unwrap().clone_pointer()
                } else {
                    None
                },
            ],
            row_bytes: self.row_bytes,
            ..image::Image::default()
        };

        self.width = width;
        self.height = height;
        if src.has_plane(Plane::Y) || src.has_plane(Plane::A) {
            if src.width > 16384 || src.height > 16384 {
                return Err(AvifError::NotImplemented);
            }
            if src.has_plane(Plane::Y) {
                self.allocate_planes(Category::Color)?;
            }
            if src.has_plane(Plane::A) {
                self.allocate_planes(Category::Alpha)?;
            }
        }
        for plane in ALL_PLANES {
            if !src.has_plane(plane) {
                continue;
            }
            let src_pd = src.plane_data(plane).unwrap();
            let pd = self.plane_data(plane).unwrap();
            // libyuv versions >= 1880 reports a return value here. Older versions do not. Ignore
            // the return value for now.
            #[allow(clippy::let_unit_value)]
            let _ret = unsafe {
                if src.depth > 8 {
                    let source_ptr = src.planes[plane.to_usize()].as_ref().unwrap().ptr16();
                    let dst_ptr = self.planes[plane.to_usize()].as_mut().unwrap().ptr16_mut();
                    ScalePlane_12(
                        source_ptr,
                        i32_from_u32(src_pd.row_bytes / 2)?,
                        i32_from_u32(src_pd.width)?,
                        i32_from_u32(src_pd.height)?,
                        dst_ptr,
                        i32_from_u32(pd.row_bytes / 2)?,
                        i32_from_u32(pd.width)?,
                        i32_from_u32(pd.height)?,
                        FilterMode_kFilterBox,
                    )
                } else {
                    let source_ptr = src.planes[plane.to_usize()].as_ref().unwrap().ptr();
                    let dst_ptr = self.planes[plane.to_usize()].as_mut().unwrap().ptr_mut();
                    ScalePlane(
                        source_ptr,
                        i32_from_u32(src_pd.row_bytes)?,
                        i32_from_u32(src_pd.width)?,
                        i32_from_u32(src_pd.height)?,
                        dst_ptr,
                        i32_from_u32(pd.row_bytes)?,
                        i32_from_u32(pd.width)?,
                        i32_from_u32(pd.height)?,
                        FilterMode_kFilterBox,
                    )
                }
            };
        }
        Ok(())
    }
}
