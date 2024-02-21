use super::libyuv;
use super::rgb_impl;

use crate::image;
use crate::image::Plane;
use crate::internal_utils::pixels::*;
use crate::internal_utils::*;
use crate::*;

#[repr(C)]
#[derive(Default, PartialEq, Copy, Clone)]
pub enum Format {
    Rgb,
    #[default]
    Rgba,
    Argb,
    Bgr,
    Bgra,
    Abgr,
    Rgb565,
}

impl Format {
    pub fn offsets(&self) -> [usize; 4] {
        match self {
            Format::Rgb => [0, 1, 2, 0],
            Format::Rgba => [0, 1, 2, 3],
            Format::Argb => [1, 2, 3, 0],
            Format::Bgr => [2, 1, 0, 0],
            Format::Bgra => [2, 1, 0, 3],
            Format::Abgr => [3, 2, 1, 0],
            Format::Rgb565 => [0; 4],
        }
    }

    pub fn r_offset(&self) -> usize {
        self.offsets()[0]
    }

    pub fn g_offset(&self) -> usize {
        self.offsets()[1]
    }

    pub fn b_offset(&self) -> usize {
        self.offsets()[2]
    }

    pub fn alpha_offset(&self) -> usize {
        self.offsets()[3]
    }
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub enum ChromaUpsampling {
    #[default]
    Automatic,
    Fastest,
    BestQuality,
    Nearest,
    Bilinear,
}

impl ChromaUpsampling {
    pub fn nearest_neighbor_filter_allowed(&self) -> bool {
        // TODO: this function has to return different values based on whether libyuv is used.
        !matches!(self, Self::Bilinear | Self::BestQuality)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub enum ChromaDownsampling {
    #[default]
    Automatic,
    Fastest,
    BestQuality,
    Average,
    SharpYuv,
}

#[derive(Default)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub format: Format,
    pub chroma_upsampling: ChromaUpsampling,
    pub chroma_downsampling: ChromaDownsampling,
    pub ignore_alpha: bool,
    pub alpha_premultiplied: bool,
    pub is_float: bool,
    pub max_threads: i32,
    pub pixels: Option<Pixels>,
    pub row_bytes: u32,
}

#[derive(Default, Debug, PartialEq)]
pub enum AlphaMultiplyMode {
    #[default]
    NoOp,
    Multiply,
    UnMultiply,
}

impl Image {
    pub fn max_channel(&self) -> u16 {
        ((1i32 << self.depth) - 1) as u16
    }

    pub fn max_channel_f(&self) -> f32 {
        self.max_channel() as f32
    }

    pub fn create_from_yuv(image: &image::Image) -> Self {
        Self {
            width: image.width,
            height: image.height,
            depth: image.depth as u32,
            format: Format::Rgba,
            chroma_upsampling: ChromaUpsampling::Automatic,
            chroma_downsampling: ChromaDownsampling::Automatic,
            ignore_alpha: false,
            alpha_premultiplied: false,
            is_float: false,
            max_threads: 1,
            pixels: None,
            row_bytes: 0,
        }
    }

    pub fn pixels(&mut self) -> *mut u8 {
        if self.pixels.is_none() {
            return std::ptr::null_mut();
        }
        match self.pixels.as_mut().unwrap() {
            Pixels::Pointer(ptr) => *ptr,
            Pixels::Buffer(buffer) => buffer.as_mut_ptr(),
            Pixels::Buffer16(buffer) => buffer.as_mut_ptr() as *mut u8,
        }
    }

    pub fn row(&self, row: u32) -> AvifResult<&[u8]> {
        match &self.pixels {
            Some(pixels) => pixels.slice(row * self.row_bytes, self.row_bytes),
            None => Err(AvifError::NoContent),
        }
    }

    pub fn row_mut(&mut self, row: u32) -> AvifResult<&mut [u8]> {
        match &mut self.pixels {
            Some(pixels) => pixels.slice_mut(row * self.row_bytes, self.row_bytes),
            None => Err(AvifError::NoContent),
        }
    }

    pub fn row16(&self, row: u32) -> AvifResult<&[u16]> {
        match &self.pixels {
            Some(pixels) => pixels.slice16(row * self.row_bytes / 2, self.row_bytes / 2),
            None => Err(AvifError::NoContent),
        }
    }

    pub fn row16_mut(&mut self, row: u32) -> AvifResult<&mut [u16]> {
        match &mut self.pixels {
            Some(pixels) => pixels.slice16_mut(row * self.row_bytes / 2, self.row_bytes / 2),
            None => Err(AvifError::NoContent),
        }
    }

    pub fn rows_mut(&mut self, row: u32) -> AvifResult<(Option<&mut [u8]>, Option<&mut [u16]>)> {
        if self.depth == 8 {
            Ok((Some(self.row_mut(row)?), None))
        } else {
            Ok((None, Some(self.row16_mut(row)?)))
        }
    }

    pub fn allocate(&mut self) -> AvifResult<()> {
        let row_bytes = self.width * self.pixel_size();
        if self.channel_size() == 1 {
            let buffer_size: usize = usize_from_u32(row_bytes * self.height)?;
            let buffer: Vec<u8> = vec![0; buffer_size];
            self.pixels = Some(Pixels::Buffer(buffer));
        } else {
            let buffer_size: usize = usize_from_u32((row_bytes / 2) * self.height)?;
            let buffer: Vec<u16> = vec![0; buffer_size];
            self.pixels = Some(Pixels::Buffer16(buffer));
        }
        self.row_bytes = row_bytes;
        Ok(())
    }

    pub fn depth_valid(&self) -> bool {
        matches!(self.depth, 8 | 10 | 12 | 16)
    }

    pub fn has_alpha(&self) -> bool {
        !matches!(self.format, Format::Rgb | Format::Bgr | Format::Rgb565)
    }

    pub fn channel_size(&self) -> u32 {
        if self.depth == 8 {
            1
        } else {
            2
        }
    }

    pub fn channel_count(&self) -> u32 {
        if self.has_alpha() {
            4
        } else {
            3
        }
    }

    pub fn pixel_size(&self) -> u32 {
        if self.format == Format::Rgb565 {
            return 2;
        }
        self.channel_count() * self.channel_size()
    }

    fn convert_to_half_float(&mut self) -> AvifResult<()> {
        let scale = 1.0 / self.max_channel_f();
        match libyuv::convert_to_half_float(self, scale) {
            Ok(_) => return Ok(()),
            Err(err) => {
                if err != AvifError::NotImplemented {
                    return Err(err);
                }
            }
        }
        // This constant comes from libyuv. For details, see here:
        // https://chromium.googlesource.com/libyuv/libyuv/+/2f87e9a7/source/row_common.cc#3537
        let multiplier = 1.925_93e-34 * scale;
        for y in 0..self.height {
            let row = self.row16_mut(y)?;
            for pixel in row {
                *pixel = ((((*pixel as f32) * multiplier) as u32) >> 13) as u16;
            }
        }
        Ok(())
    }

    pub fn convert_from_yuv(&mut self, image: &image::Image) -> AvifResult<()> {
        // TODO: use plane constant here and elsewhere.
        if !image.has_plane(Plane::Y) {
            return Err(AvifError::ReformatFailed);
        }
        let mut alpha_multiply_mode = AlphaMultiplyMode::NoOp;
        if image.has_alpha() {
            if !self.has_alpha() || self.ignore_alpha {
                if !image.alpha_premultiplied {
                    alpha_multiply_mode = AlphaMultiplyMode::Multiply;
                }
            } else if !image.alpha_premultiplied && self.alpha_premultiplied {
                alpha_multiply_mode = AlphaMultiplyMode::Multiply;
            } else if image.alpha_premultiplied && !self.alpha_premultiplied {
                alpha_multiply_mode = AlphaMultiplyMode::UnMultiply;
            }
        }

        let mut converted_with_libyuv: bool = false;
        let reformat_alpha = self.has_alpha()
            && (!self.ignore_alpha || alpha_multiply_mode != AlphaMultiplyMode::NoOp);
        let mut alpha_reformatted_with_libyuv = false;
        if alpha_multiply_mode == AlphaMultiplyMode::NoOp || self.has_alpha() {
            match libyuv::yuv_to_rgb(image, self, reformat_alpha) {
                Ok(alpha_reformatted) => {
                    alpha_reformatted_with_libyuv = alpha_reformatted;
                    converted_with_libyuv = true;
                }
                Err(err) => {
                    if err != AvifError::NotImplemented {
                        return Err(err);
                    }
                }
            }
        }
        if reformat_alpha && !alpha_reformatted_with_libyuv {
            if image.has_alpha() {
                self.import_alpha_from(image)?;
            } else {
                self.set_opaque()?;
            }
        }
        if !converted_with_libyuv {
            if let Err(err) = rgb_impl::yuv_to_rgb(image, self) {
                if err != AvifError::NotImplemented {
                    return Err(err);
                }
                rgb_impl::yuv_to_rgb_any(image, self, alpha_multiply_mode)?;
                alpha_multiply_mode = AlphaMultiplyMode::NoOp;
            }
        }
        match alpha_multiply_mode {
            AlphaMultiplyMode::Multiply => self.premultiply_alpha()?,
            AlphaMultiplyMode::UnMultiply => self.unpremultiply_alpha()?,
            _ => {}
        }
        if self.is_float {
            self.convert_to_half_float()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::decoder::Category;
    use crate::image::ALL_PLANES;
    use crate::image::MAX_PLANE_COUNT;

    use test_case::test_matrix;

    const WIDTH: usize = 3;
    const HEIGHT: usize = 3;
    struct YuvParams {
        width: u32,
        height: u32,
        depth: u8,
        format: PixelFormat,
        full_range: bool,
        color_primaries: ColorPrimaries,
        matrix_coefficients: MatrixCoefficients,
        planes: [[&'static [u16]; HEIGHT]; MAX_PLANE_COUNT],
    }

    const YUV_PARAMS: [YuvParams; 1] = [YuvParams {
        width: WIDTH as u32,
        height: HEIGHT as u32,
        depth: 12,
        format: PixelFormat::Yuv420,
        full_range: false,
        color_primaries: ColorPrimaries::Srgb,
        matrix_coefficients: MatrixCoefficients::Bt709,
        planes: [
            [
                &[1001, 1001, 1001],
                &[1001, 1001, 1001],
                &[1001, 1001, 1001],
            ],
            [&[1637, 1637], &[1637, 1637], &[1637, 1637]],
            [&[3840, 3840], &[3840, 3840], &[3840, 3840]],
            [&[0, 0, 2039], &[0, 2039, 4095], &[2039, 4095, 4095]],
        ],
    }];

    struct RgbParams {
        params: (
            /*yuv_param_index:*/ usize,
            /*format:*/ Format,
            /*depth:*/ u32,
            /*alpha_premultiplied:*/ bool,
            /*is_float:*/ bool,
        ),
        expected_rgba: [&'static [u16]; HEIGHT],
    }

    const RGB_PARAMS: [RgbParams; 5] = [
        RgbParams {
            params: (0, Format::Rgba, 16, true, false),
            expected_rgba: [
                &[0, 0, 0, 0, 0, 0, 0, 0, 32631, 1, 0, 32631],
                &[0, 0, 0, 0, 32631, 1, 0, 32631, 65535, 2, 0, 65535],
                &[32631, 1, 0, 32631, 65535, 2, 0, 65535, 65535, 2, 0, 65535],
            ],
        },
        RgbParams {
            params: (0, Format::Rgba, 16, true, true),
            expected_rgba: [
                &[0, 0, 0, 0, 0, 0, 0, 0, 14327, 256, 0, 14327],
                &[0, 0, 0, 0, 14327, 256, 0, 14327, 15360, 512, 0, 15360],
                &[
                    14327, 256, 0, 14327, 15360, 512, 0, 15360, 15360, 512, 0, 15360,
                ],
            ],
        },
        RgbParams {
            params: (0, Format::Rgba, 16, false, true),
            expected_rgba: [
                &[15360, 512, 0, 0, 15360, 512, 0, 0, 15360, 512, 0, 14327],
                &[15360, 512, 0, 0, 15360, 512, 0, 14327, 15360, 512, 0, 15360],
                &[
                    15360, 512, 0, 14327, 15360, 512, 0, 15360, 15360, 512, 0, 15360,
                ],
            ],
        },
        RgbParams {
            params: (0, Format::Rgba, 16, false, false),
            expected_rgba: [
                &[65535, 2, 0, 0, 65535, 2, 0, 0, 65535, 2, 0, 32631],
                &[65535, 2, 0, 0, 65535, 2, 0, 32631, 65535, 2, 0, 65535],
                &[65535, 2, 0, 32631, 65535, 2, 0, 65535, 65535, 2, 0, 65535],
            ],
        },
        RgbParams {
            params: (0, Format::Bgra, 16, true, false),
            expected_rgba: [
                &[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 32631, 32631],
                &[0, 0, 0, 0, 0, 1, 32631, 32631, 0, 2, 65535, 65535],
                &[0, 1, 32631, 32631, 0, 2, 65535, 65535, 0, 2, 65535, 65535],
            ],
        },
    ];

    #[test_matrix(0usize..5)]
    fn rgb_conversion(rgb_param_index: usize) -> AvifResult<()> {
        let rgb_params = &RGB_PARAMS[rgb_param_index];
        let yuv_params = &YUV_PARAMS[rgb_params.params.0];
        let mut image = image::Image {
            width: yuv_params.width,
            height: yuv_params.height,
            depth: yuv_params.depth,
            yuv_format: yuv_params.format,
            color_primaries: yuv_params.color_primaries,
            matrix_coefficients: yuv_params.matrix_coefficients,
            full_range: yuv_params.full_range,
            ..image::Image::default()
        };
        image.allocate_planes(Category::Color)?;
        image.allocate_planes(Category::Alpha)?;
        let yuva_planes = &yuv_params.planes;
        for plane in ALL_PLANES {
            let plane_index = plane.to_usize();
            if yuva_planes[plane_index].is_empty() {
                continue;
            }
            for y in 0..image.height(plane) {
                let row16 = image.row16_mut(plane, y as u32)?;
                assert_eq!(row16.len(), yuva_planes[plane_index][y].len());
                let dst = &mut row16[..];
                dst.copy_from_slice(yuva_planes[plane_index][y]);
            }
        }

        let mut rgb = Image::create_from_yuv(&image);
        assert_eq!(rgb.width, image.width);
        assert_eq!(rgb.height, image.height);
        assert_eq!(rgb.depth, image.depth as u32);

        rgb.format = rgb_params.params.1;
        rgb.depth = rgb_params.params.2;
        rgb.alpha_premultiplied = rgb_params.params.3;
        rgb.is_float = rgb_params.params.4;

        rgb.allocate()?;
        rgb.convert_from_yuv(&image)?;

        for y in 0..rgb.height as usize {
            let row16 = rgb.row16(y as u32)?;
            assert_eq!(&row16[..], rgb_params.expected_rgba[y]);
        }
        Ok(())
    }
}
