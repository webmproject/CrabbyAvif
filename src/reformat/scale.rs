use crate::image::*;
use crate::internal_utils::pixels::*;
use crate::internal_utils::*;
use crate::reformat::bindings::libyuv::*;
use crate::*;

impl Image {
    pub fn scale(&mut self, width: u32, height: u32) -> AvifResult<()> {
        if self.width == width && self.height == height {
            return Ok(());
        }
        if width == 0 || height == 0 {
            return Err(AvifError::InvalidArgument);
        }
        if (self.planes2[0].is_some() && !self.planes2[0].as_ref().unwrap().is_pointer())
            || (self.planes2[1].is_some() && !self.planes2[1].as_ref().unwrap().is_pointer())
            || (self.planes2[2].is_some() && !self.planes2[2].as_ref().unwrap().is_pointer())
            || (self.planes2[3].is_some() && !self.planes2[3].as_ref().unwrap().is_pointer())
        {
            // TODO: implement this function for non-pointer inputs.
            return Err(AvifError::NotImplemented);
        }
        let src = image::Image {
            width: self.width,
            height: self.height,
            depth: self.depth,
            yuv_format: self.yuv_format,
            planes2: [
                if self.planes2[0].is_some() {
                    Some(Pixels::Pointer(self.planes2[0].as_ref().unwrap().pointer()))
                } else {
                    None
                },
                if self.planes2[1].is_some() {
                    Some(Pixels::Pointer(self.planes2[1].as_ref().unwrap().pointer()))
                } else {
                    None
                },
                if self.planes2[2].is_some() {
                    Some(Pixels::Pointer(self.planes2[2].as_ref().unwrap().pointer()))
                } else {
                    None
                },
                if self.planes2[3].is_some() {
                    Some(Pixels::Pointer(self.planes2[3].as_ref().unwrap().pointer()))
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
                self.allocate_planes(0)?;
            }
            if src.has_plane(Plane::A) {
                self.allocate_planes(1)?;
            }
        }
        for plane in ALL_PLANES {
            if !src.has_plane(plane) {
                continue;
            }
            let src_pd = src.plane(plane).unwrap();
            let pd = self.plane_mut(plane).unwrap();
            let ret = unsafe {
                if src.depth > 8 {
                    ScalePlane_12(
                        src_pd.data16.unwrap().as_ptr(),
                        i32_from_u32(src_pd.row_bytes / 2)?,
                        i32_from_u32(src_pd.width)?,
                        i32_from_u32(src_pd.height)?,
                        pd.data16.unwrap().as_mut_ptr(),
                        i32_from_u32(pd.row_bytes / 2)?,
                        i32_from_u32(pd.width)?,
                        i32_from_u32(pd.height)?,
                        FilterMode_kFilterBox,
                    )
                } else {
                    ScalePlane(
                        src_pd.data.unwrap().as_ptr(),
                        i32_from_u32(src_pd.row_bytes)?,
                        i32_from_u32(src_pd.width)?,
                        i32_from_u32(src_pd.height)?,
                        pd.data.unwrap().as_mut_ptr(),
                        i32_from_u32(pd.row_bytes)?,
                        i32_from_u32(pd.width)?,
                        i32_from_u32(pd.height)?,
                        FilterMode_kFilterBox,
                    )
                }
            };
            if ret != 0 {
                return Err(if ret == 1 {
                    AvifError::OutOfMemory
                } else {
                    AvifError::UnknownError
                });
            }
        }
        Ok(())
    }
}
