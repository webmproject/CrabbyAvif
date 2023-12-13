use super::libyuv;
use crate::image;
use crate::image::Plane;
use crate::internal_utils::*;
use crate::*;

#[derive(Default, PartialEq)]
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

pub enum ChromaUpsampling {
    Automatic,
    Fastest,
    BestQuality,
    Nearest,
    Bilinear,
}

pub enum ChromaDownsampling {
    Automatic,
    Fastest,
    BestQuality,
    Average,
    SharpYuv,
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
    pub pixels: *mut u8, // TODO: slice?
    pub row_bytes: u32,
}

struct RgbColorSpaceInfo {
    channel_bytes: u32,
    pixel_bytes: u32,
    offset_bytes_r: u32,
    offset_bytes_g: u32,
    offset_bytes_b: u32,
    offset_bytes_a: u32,
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
            Format::Rgb565 | _ => [0; 4],
        };
        let max_channel = i32_from_u32((1 << rgb.depth) - 1)?;
        Ok(Self {
            channel_bytes: rgb.channel_size(),
            pixel_bytes: rgb.pixel_size(),
            offset_bytes_r: rgb.channel_size() * offsets[0],
            offset_bytes_g: rgb.channel_size() * offsets[1],
            offset_bytes_b: rgb.channel_size() * offsets[2],
            offset_bytes_a: rgb.channel_size() * offsets[3],
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
            && image.yuv_format != PixelFormat::Yuv420
        {
            return Err(AvifError::ReformatFailed);
        }
        let mut kr: f32 = 0.0;
        let mut kg: f32 = 0.0;
        let mut kb: f32 = 0.0;
        let mode = match image.matrix_coefficients {
            MatrixCoefficients::Identity => Mode::Identity,
            MatrixCoefficients::Ycgco => Mode::Ycgco,
            _ => {
                // TODO: compute kr, kg and kb here.
                Mode::YuvCoefficients
            }
        };
        let max_channel = ((1 << image.depth) - 1) as i32;
        Ok(Self {
            kr,
            kg,
            kb,
            channel_bytes: if image.depth == 8 { 1 } else { 2 },
            depth: image.depth as u32,
            full_range: image.full_range,
            max_channel,
            bias_y: if image.full_range {
                0.0
            } else {
                (16 << (image.depth - 8)) as f32
            },
            bias_uv: (1 << (image.depth - 1)) as f32,
            range_y: if image.full_range {
                max_channel
            } else {
                219 << (image.depth - 8)
            } as f32,
            range_uv: if image.full_range {
                max_channel
            } else {
                224 << (image.depth - 8)
            } as f32,
            format: image.yuv_format,
            mode,
        })
    }
}

struct State {
    rgb: RgbColorSpaceInfo,
    yuv: YuvColorSpaceInfo,
}

#[derive(Default, PartialEq)]
enum AlphaMultiplyMode {
    #[default]
    NoOp,
    Multiply,
    UnMultiply,
}

impl Image {
    fn depth_valid(&self) -> bool {
        match self.depth {
            8 | 10 | 12 | 16 => true,
            _ => false,
        }
    }

    fn has_alpha(&self) -> bool {
        match self.format {
            Format::Rgb | Format::Bgr | Format::Rgb565 => false,
            _ => true,
        }
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

    fn pixel_size(&self) -> u32 {
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
        let state = State {
            rgb: RgbColorSpaceInfo::create_from(self)?,
            yuv: YuvColorSpaceInfo::create_from(image)?,
        };
        let mut alpha_multiply_mode = AlphaMultiplyMode::NoOp;
        if image.planes[3].is_none() {
            if !self.has_alpha() || self.ignore_alpha {
                if !image.alpha_premultiplied {
                    alpha_multiply_mode = AlphaMultiplyMode::Multiply;
                }
            } else {
                if !image.alpha_premultiplied && self.alpha_premultiplied {
                    alpha_multiply_mode = AlphaMultiplyMode::Multiply;
                } else if image.alpha_premultiplied && !self.alpha_premultiplied {
                    alpha_multiply_mode = AlphaMultiplyMode::UnMultiply;
                }
            }
        }

        let mut converted_with_libyuv: bool = false;
        let reformat_alpha = self.has_alpha()
            && (!self.ignore_alpha || alpha_multiply_mode != AlphaMultiplyMode::NoOp);
        // TODO: alpha_reformatted_with_libyuv.
        if alpha_multiply_mode == AlphaMultiplyMode::NoOp || self.has_alpha() {
            libyuv::yuv_to_rgb(image, &self, reformat_alpha)?;
        }
        Err(AvifError::ReformatFailed)
    }
}
