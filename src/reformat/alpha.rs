use super::libyuv;
use super::rgb;

use crate::internal_utils::*;
use crate::*;

impl rgb::Image {
    pub fn premultiply_alpha(&mut self) -> AvifResult<()> {
        if self.pixels().is_null() || self.row_bytes == 0 {
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
        if self.pixels().is_null() || self.row_bytes == 0 {
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

    pub fn fill_alpha(&mut self, offset_bytes_a: usize) -> AvifResult<()> {
        if self.depth > 8 {
            let max_channel = ((1 << self.depth) - 1) as u16;
            for y in 0..self.height {
                let width = usize_from_u32(self.width)?;
                let row = self.mut_row16(y)?;
                for x in 0..width {
                    row[(x * 4) + offset_bytes_a] = max_channel;
                }
            }
        } else {
            for y in 0..self.height {
                let width = usize_from_u32(self.width)?;
                let row = self.mut_row(y)?;
                for x in 0..width {
                    row[(x * 4) + offset_bytes_a] = 255;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use test_case::test_matrix;

    #[test_matrix(20, 10, [8, 10, 12, 16], 0..=3, [true, false])]
    fn fill_alpha_rgb_buffer(
        width: u32,
        height: u32,
        depth: u32,
        alpha_byte_offset: usize,
        use_pointer: bool,
    ) -> AvifResult<()> {
        let mut rgb = rgb::Image {
            width,
            height,
            depth,
            ..rgb::Image::default()
        };
        let mut buffer: Vec<u8> = Vec::new();
        if use_pointer {
            let pixel_size = if depth == 8 { 1 } else { 2 };
            let buffer_size = (width * height * 4 * pixel_size) as usize;
            buffer.reserve_exact(buffer_size);
            buffer.resize(buffer_size, 0);
            rgb.pixels = Some(rgb::Pixels::Pointer(buffer.as_mut_ptr()));
            rgb.row_bytes = width * 4 * pixel_size;
        } else {
            rgb.allocate()?;
        }
        rgb.fill_alpha(alpha_byte_offset)?;
        if depth == 8 {
            for y in 0..height {
                let row = rgb.row(y)?;
                assert_eq!(row.len(), (width * 4) as usize);
                for x in 0..width as usize {
                    for idx in 0usize..4 {
                        let expected_value = if idx == alpha_byte_offset { 255 } else { 0 };
                        assert_eq!(row[(x * 4) + idx], expected_value);
                    }
                }
            }
        } else {
            let max_channel = ((1 << depth) - 1) as u16;
            for y in 0..height {
                let row = rgb.row16(y)?;
                assert_eq!(row.len(), (width * 4) as usize);
                for x in 0..width as usize {
                    for idx in 0usize..4 {
                        let expected_value = if idx == alpha_byte_offset { max_channel } else { 0 };
                        assert_eq!(row[(x * 4) + idx], expected_value);
                    }
                }
            }
        }
        Ok(())
    }
}
