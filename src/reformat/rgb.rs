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
enum AlphaMultiplyMode {
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

    pub fn allocate(&mut self) -> AvifResult<()> {
        let row_bytes = self.width * self.pixel_size();
        if self.channel_size() == 1 {
            let buffer_size: usize = usize_from_u32(row_bytes * self.height)?;
            let mut buffer: Vec<u8> = Vec::new();
            buffer.reserve(buffer_size);
            buffer.resize(buffer_size, 0);
            self.pixels = Some(Pixels::Buffer(buffer));
        } else {
            let buffer_size: usize = usize_from_u32((row_bytes / 2) * self.height)?;
            let mut buffer: Vec<u16> = Vec::new();
            buffer.reserve(buffer_size);
            buffer.resize(buffer_size, 0);
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
        let multiplier = 1.9259299444e-34 * scale;
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
        println!(
            "alpha_multiply_mode: {:#?} reformat_alpha: {reformat_alpha}",
            alpha_multiply_mode
        );
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
                self.reformat_alpha(image)?;
            } else {
                self.fill_alpha()?;
            }
        }
        if !converted_with_libyuv {
            rgb_impl::yuv_to_rgb(image, self)?;
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
