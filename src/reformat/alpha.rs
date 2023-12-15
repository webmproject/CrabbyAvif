use super::libyuv;
use super::rgb;

use crate::internal_utils::*;
use crate::*;

impl rgb::Image {
    pub fn premultiply_alpha(&mut self) -> AvifResult<()> {
        if self.pixels.is_null() || self.row_bytes == 0 {
            return Err(AvifError::ReformatFailed);
        }
        if !self.has_alpha() {
            return Err(AvifError::InvalidArgument);
        }
        match libyuv::process_alpha(self, true) {
            Ok(_) => return Ok(()),
            Err(err) => {
                if err != AvifError::NotImplemented {
                    return Err(err);
                }
            }
        }
        unimplemented!("native alpha multiply implementation");
    }

    pub fn unpremultiply_alpha(&mut self) -> AvifResult<()> {
        if self.pixels.is_null() || self.row_bytes == 0 {
            return Err(AvifError::ReformatFailed);
        }
        if !self.has_alpha() {
            return Err(AvifError::InvalidArgument);
        }
        match libyuv::process_alpha(self, false) {
            Ok(_) => return Ok(()),
            Err(err) => {
                if err != AvifError::NotImplemented {
                    return Err(err);
                }
            }
        }
        unimplemented!("native alpha unmultiply implementation");
    }

    pub fn fill_alpha(&mut self, offset_bytes_a: isize) -> AvifResult<()> {
        if self.depth > 8 {
            let max_channel = ((1 << self.depth) - 1) as u16;
            for y in 0..self.height {
                let ptr =
                    unsafe { self.pixels.offset(isize_from_u32(y * self.row_bytes)?) } as *mut u16;
                for x in 0..isize_from_u32(self.width)? {
                    unsafe {
                        *ptr.offset(x + offset_bytes_a) = max_channel;
                    }
                }
            }
        } else {
            for y in 0..self.height {
                let ptr = unsafe { self.pixels.offset(isize_from_u32(y * self.row_bytes)?) };
                for x in 0..isize_from_u32(self.width)? {
                    unsafe {
                        *ptr.offset(x + offset_bytes_a) = 255;
                    }
                }
            }
        }
        Ok(())
    }
}
