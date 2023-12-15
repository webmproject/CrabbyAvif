use super::libyuv;
use crate::image;
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

#[repr(C)]
#[derive(Copy, Clone)]
pub enum ChromaUpsampling {
    Automatic,
    Fastest,
    BestQuality,
    Nearest,
    Bilinear,
}

impl ChromaUpsampling {
    pub fn nearest_neighbor_filter_allowed(&self) -> bool {
        !matches!(self, Self::Bilinear | Self::BestQuality | Self::Automatic)
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub enum ChromaDownsampling {
    Automatic,
    Fastest,
    BestQuality,
    Average,
    SharpYuv,
}

pub enum Pixels {
    Pointer(*mut u8),
    Buffer(Vec<u8>),
}

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
    pub pixel_buffer: Vec<u8>,
}

#[allow(unused)]
struct RgbColorSpaceInfo {
    channel_bytes: u32,
    pixel_bytes: u32,
    offset_bytes_r: isize,
    offset_bytes_g: isize,
    offset_bytes_b: isize,
    offset_bytes_a: isize,
    max_channel: i32,
    max_channel_f: f32,
}

impl RgbColorSpaceInfo {
    fn create_from(rgb: &Image) -> AvifResult<Self> {
        if !rgb.depth_valid()
            || (rgb.is_float && rgb.depth != 16)
            || (rgb.format == Format::Rgb565 && rgb.depth != 8)
        {
            return Err(AvifError::ReformatFailed);
        }
        let offsets: [u32; 4] = match rgb.format {
            Format::Rgb => [0, 1, 2, 0],
            Format::Rgba => [0, 1, 2, 3],
            Format::Argb => [1, 2, 3, 0],
            Format::Bgr => [2, 1, 0, 0],
            Format::Bgra => [2, 1, 0, 3],
            Format::Abgr => [3, 2, 1, 0],
            Format::Rgb565 => [0; 4],
        };
        let max_channel = i32_from_u32((1 << rgb.depth) - 1)?;
        Ok(Self {
            channel_bytes: rgb.channel_size(),
            pixel_bytes: rgb.pixel_size(),
            offset_bytes_r: (rgb.channel_size() * offsets[0]) as isize,
            offset_bytes_g: (rgb.channel_size() * offsets[1]) as isize,
            offset_bytes_b: (rgb.channel_size() * offsets[2]) as isize,
            offset_bytes_a: (rgb.channel_size() * offsets[3]) as isize,
            max_channel,
            max_channel_f: max_channel as f32,
        })
    }
}

enum Mode {
    YuvCoefficients,
    Identity,
    Ycgco,
}

#[allow(unused)]
struct YuvColorSpaceInfo {
    kr: f32,
    kg: f32,
    kb: f32,
    channel_bytes: u32,
    depth: u32,
    full_range: bool,
    max_channel: i32,
    bias_y: f32,
    bias_uv: f32,
    range_y: f32,
    range_uv: f32,
    format: PixelFormat,
    mode: Mode,
}

impl YuvColorSpaceInfo {
    fn create_from(image: &image::Image) -> AvifResult<Self> {
        if !image.depth_valid() {
            return Err(AvifError::ReformatFailed);
        }
        // Unsupported matrix coefficients.
        match image.matrix_coefficients {
            MatrixCoefficients::Ycgco
            | MatrixCoefficients::Bt2020Cl
            | MatrixCoefficients::Smpte2085
            | MatrixCoefficients::ChromaDerivedCl
            | MatrixCoefficients::Ictcp => return Err(AvifError::ReformatFailed),
            _ => {}
        }
        if image.matrix_coefficients == MatrixCoefficients::Identity
            && image.yuv_format != PixelFormat::Yuv444
            && image.yuv_format != PixelFormat::Monochrome
        {
            return Err(AvifError::ReformatFailed);
        }
        let kr: f32 = 0.0;
        let kg: f32 = 0.0;
        let kb: f32 = 0.0;
        let mode = match image.matrix_coefficients {
            MatrixCoefficients::Identity => Mode::Identity,
            MatrixCoefficients::Ycgco => Mode::Ycgco,
            _ => {
                // TODO: compute kr, kg and kb here.
                Mode::YuvCoefficients
            }
        };
        let max_channel = (1 << image.depth) - 1;
        Ok(Self {
            kr,
            kg,
            kb,
            channel_bytes: if image.depth == 8 { 1 } else { 2 },
            depth: image.depth as u32,
            full_range: image.full_range,
            max_channel,
            bias_y: if image.full_range { 0.0 } else { (16 << (image.depth - 8)) as f32 },
            bias_uv: (1 << (image.depth - 1)) as f32,
            range_y: if image.full_range { max_channel } else { 219 << (image.depth - 8) } as f32,
            range_uv: if image.full_range { max_channel } else { 224 << (image.depth - 8) } as f32,
            format: image.yuv_format,
            mode,
        })
    }
}

struct State {
    rgb: RgbColorSpaceInfo,
    #[allow(unused)]
    yuv: YuvColorSpaceInfo,
}

#[derive(Default, Debug, PartialEq)]
enum AlphaMultiplyMode {
    #[default]
    NoOp,
    Multiply,
    UnMultiply,
}

impl Image {
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
            pixel_buffer: Vec::new(),
        }
    }

    pub fn pixels(&mut self) -> *mut u8 {
        if self.pixels.is_none() {
            return std::ptr::null_mut();
        }
        match self.pixels.as_mut().unwrap() {
            Pixels::Pointer(ptr) => *ptr,
            Pixels::Buffer(buffer) => buffer.as_mut_ptr(),
        }
    }

    pub fn allocate(&mut self) -> AvifResult<()> {
        let row_bytes = self.width * self.pixel_size();
        let buffer_size: usize = usize_from_u32(row_bytes * self.height)?;
        let mut buffer: Vec<u8> = Vec::new();
        buffer.reserve(buffer_size);
        buffer.resize(buffer_size, 0);
        self.pixels = Some(Pixels::Buffer(buffer));
        self.row_bytes = row_bytes;
        Ok(())
    }

    fn depth_valid(&self) -> bool {
        matches!(self.depth, 8 | 10 | 12 | 16)
    }

    pub fn has_alpha(&self) -> bool {
        !matches!(self.format, Format::Rgb | Format::Bgr | Format::Rgb565)
    }

    fn channel_size(&self) -> u32 {
        if self.depth == 8 {
            1
        } else {
            2
        }
    }

    fn channel_count(&self) -> u32 {
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

    pub fn convert_from_yuv(&mut self, image: &image::Image) -> AvifResult<()> {
        // TODO: use plane constant here and elsewhere.
        if image.planes[0].is_none() {
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
        let state = State {
            rgb: RgbColorSpaceInfo::create_from(self)?,
            yuv: YuvColorSpaceInfo::create_from(image)?,
        };
        if reformat_alpha && !alpha_reformatted_with_libyuv {
            if image.has_alpha() {
                unimplemented!("reformat_alpha");
            } else {
                self.fill_alpha(state.rgb.offset_bytes_a)?;
            }
        }
        if !converted_with_libyuv {
            unimplemented!("libyuv could not convet this");
        }
        match alpha_multiply_mode {
            AlphaMultiplyMode::Multiply => self.premultiply_alpha()?,
            AlphaMultiplyMode::UnMultiply => {
                unimplemented!("needs alpha unmultiply!");
            }
            _ => {}
        }
        if self.is_float {
            unimplemented!("needs is float");
        }
        Ok(())
    }
}
