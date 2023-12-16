use super::libyuv;
use super::rgb;

use crate::image::Plane;
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

    pub fn fill_alpha(&mut self) -> AvifResult<()> {
        let alpha_offset = self.format.alpha_offset();
        if self.depth > 8 {
            let max_channel = ((1 << self.depth) - 1) as u16;
            for y in 0..self.height {
                let width = usize_from_u32(self.width)?;
                let row = self.row16_mut(y)?;
                for x in 0..width {
                    row[(x * 4) + alpha_offset] = max_channel;
                }
            }
        } else {
            for y in 0..self.height {
                let width = usize_from_u32(self.width)?;
                let row = self.row_mut(y)?;
                for x in 0..width {
                    row[(x * 4) + alpha_offset] = 255;
                }
            }
        }
        Ok(())
    }

    pub fn reformat_alpha(&mut self, image: &image::Image) -> AvifResult<()> {
        if self.depth == image.depth as u32 {
            let dst_alpha_offset = self.format.alpha_offset();
            let dst_pixel_size = self.pixel_size() as usize;
            let width = usize_from_u32(self.width)?;
            if self.depth > 8 {
                /*
                TODO: uncomment after image.row16 is implemented.
                for y in 0..self.height {
                    let dst_row = self.row16_mut(y)?;
                    let src_row = image.row16(Plane::A, y)?;
                    for x in 0..width as usize {
                        dst_row[x * dst_pixel_size + dst_alpha_offset] = src_row[x];
                    }
                }
                */
            } else {
                for y in 0..self.height {
                    let dst_row = self.row_mut(y)?;
                    let src_row = image.row(Plane::A, y)?;
                    for x in 0..width as usize {
                        dst_row[x * dst_pixel_size + dst_alpha_offset] = src_row[x];
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal_utils::pixels::*;

    use rand::Rng;
    use test_case::test_matrix;

    const ALPHA_RGB_FORMATS: [rgb::Format; 4] = [
        rgb::Format::Rgba,
        rgb::Format::Argb,
        rgb::Format::Bgra,
        rgb::Format::Abgr,
    ];

    fn rgb_image(
        width: u32,
        height: u32,
        depth: u32,
        format: rgb::Format,
        use_pointer: bool,
        buffer: &mut Vec<u8>,
    ) -> AvifResult<rgb::Image> {
        let mut rgb = rgb::Image {
            width,
            height,
            depth,
            format,
            ..rgb::Image::default()
        };
        if use_pointer {
            let pixel_size = if depth == 8 { 1 } else { 2 };
            let buffer_size = (width * height * 4 * pixel_size) as usize;
            buffer.reserve_exact(buffer_size);
            buffer.resize(buffer_size, 0);
            rgb.pixels = Some(Pixels::Pointer(buffer.as_mut_ptr()));
            rgb.row_bytes = width * 4 * pixel_size;
        } else {
            rgb.allocate()?;
        }
        Ok(rgb)
    }

    #[test_matrix(20, 10, [8, 10, 12, 16], 0..4, [true, false])]
    fn fill_alpha(
        width: u32,
        height: u32,
        depth: u32,
        format_index: usize,
        use_pointer: bool,
    ) -> AvifResult<()> {
        let format = ALPHA_RGB_FORMATS[format_index];
        let mut buffer: Vec<u8> = vec![];
        let mut rgb = rgb_image(width, height, depth, format, use_pointer, &mut buffer)?;

        rgb.fill_alpha()?;

        let alpha_offset = rgb.format.alpha_offset();
        if depth == 8 {
            for y in 0..height {
                let row = rgb.row(y)?;
                assert_eq!(row.len(), (width * 4) as usize);
                for x in 0..width as usize {
                    for idx in 0usize..4 {
                        let expected_value = if idx == alpha_offset { 255 } else { 0 };
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
                        let expected_value = if idx == alpha_offset { max_channel } else { 0 };
                        assert_eq!(row[(x * 4) + idx], expected_value);
                    }
                }
            }
        }
        Ok(())
    }

    #[test_matrix(20, 10, [8], 0..4, [8], [true, false])]
    fn reformat_alpha(
        width: u32,
        height: u32,
        rgb_depth: u32,
        format_index: usize,
        yuv_depth: u8,
        use_pointer: bool,
    ) -> AvifResult<()> {
        let format = ALPHA_RGB_FORMATS[format_index];
        let mut buffer: Vec<u8> = vec![];
        let mut rgb = rgb_image(width, height, rgb_depth, format, use_pointer, &mut buffer)?;

        let mut image = image::Image::default();
        image.width = width;
        image.height = height;
        image.depth = yuv_depth;
        image.allocate_planes(1)?;

        let mut rng = rand::thread_rng();
        for y in 0..height {
            let row = image.row_mut(Plane::A, y)?;
            for x in 0..width as usize {
                // TODO: this can just be 0..255.
                row[x] = rng.gen_range(0..(1i32 << yuv_depth)) as u8;
            }
        }

        rgb.reformat_alpha(&image)?;

        let alpha_offset = rgb.format.alpha_offset();
        if rgb_depth == 8 {
            for y in 0..height {
                let rgb_row = rgb.row(y)?;
                let yuv_row = image.row(Plane::A, y)?;
                assert_eq!(rgb_row.len(), (width * 4) as usize);
                for x in 0..width as usize {
                    for idx in 0usize..4 {
                        let expected_value = if idx == alpha_offset { yuv_row[x] } else { 0 };
                        assert_eq!(rgb_row[(x * 4) + idx], expected_value);
                    }
                }
            }
        } else {
            let max_channel = ((1 << rgb_depth) - 1) as u16;
            for y in 0..height {
                let row = rgb.row16(y)?;
                assert_eq!(row.len(), (width * 4) as usize);
                for x in 0..width as usize {
                    for idx in 0usize..4 {
                        let expected_value = if idx == alpha_offset { max_channel } else { 0 };
                        assert_eq!(row[(x * 4) + idx], expected_value);
                    }
                }
            }
        }
        Ok(())
    }
}
