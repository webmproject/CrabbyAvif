use super::rgb;
use super::rgb::*;

use crate::image;
use crate::image::Plane;
use crate::internal_utils::*;
use crate::*;

#[allow(unused)]
struct RgbColorSpaceInfo {
    channel_bytes: u32,
    pixel_bytes: u32,
    offset_bytes_r: usize,
    offset_bytes_g: usize,
    offset_bytes_b: usize,
    offset_bytes_a: usize,
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
        let offsets = rgb.format.offsets();
        let max_channel = i32_from_u32((1 << rgb.depth) - 1)?;
        Ok(Self {
            channel_bytes: rgb.channel_size(),
            pixel_bytes: rgb.pixel_size(),
            offset_bytes_r: (rgb.channel_size() as usize * offsets[0]),
            offset_bytes_g: (rgb.channel_size() as usize * offsets[1]),
            offset_bytes_b: (rgb.channel_size() as usize * offsets[2]),
            offset_bytes_a: (rgb.channel_size() as usize * offsets[3]),
            max_channel,
            max_channel_f: max_channel as f32,
        })
    }
}

#[derive(PartialEq)]
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
    #[allow(unused)]
    rgb: RgbColorSpaceInfo,
    #[allow(unused)]
    yuv: YuvColorSpaceInfo,
}

fn identity_yuv8_to_rgb8_full_range(image: &image::Image, rgb: &mut rgb::Image) -> AvifResult<()> {
    let r_offset = rgb.format.r_offset();
    let g_offset = rgb.format.g_offset();
    let b_offset = rgb.format.b_offset();
    let rgb565 = rgb.format == Format::Rgb565;
    for i in 0..image.height {
        let y = image.row(Plane::Y, i)?;
        let u = image.row(Plane::U, i)?;
        let v = image.row(Plane::V, i)?;
        let rgb_pixels = rgb.row_mut(i)?;
        if rgb565 {
            unimplemented!("rgb 565 is not implemented");
        } else {
            for j in 0..image.width as usize {
                rgb_pixels[j + r_offset] = v[j];
                rgb_pixels[j + g_offset] = y[j];
                rgb_pixels[j + b_offset] = u[j];
            }
        }
    }
    Ok(())
}

pub fn yuv_to_rgb(image: &image::Image, rgb: &mut rgb::Image) -> AvifResult<()> {
    // TODO: This function is equivalent to libavif's "fast" path. Implement the slow path too.
    let state = State {
        rgb: RgbColorSpaceInfo::create_from(rgb)?,
        yuv: YuvColorSpaceInfo::create_from(image)?,
    };
    if state.yuv.mode == Mode::Identity {
        if image.depth == 8
            && rgb.depth == 8
            && image.yuv_format == PixelFormat::Yuv444
            && image.full_range
        {
            return identity_yuv8_to_rgb8_full_range(image, rgb);
        }
        // TODO: Add more fast paths for identity.
        return Err(AvifError::NotImplemented);
    }
    Err(AvifError::NotImplemented)
}
