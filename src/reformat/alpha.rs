#![allow(dead_code, unused)] // TODO: remove

use super::libyuv;
use super::rgb;

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
}
