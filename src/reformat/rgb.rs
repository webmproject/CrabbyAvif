use crate::image;
use crate::image::Plane;
use crate::*;

pub enum Format {
    Rgb,
    Rgba,
    Argb,
    Bgr,
    Bgra,
    Abgr,
    Rgb565,
}

enum ChromaUpsampling {
    Automatic,
    Fastest,
    BestQuality,
    Nearest,
    Bilinear,
}

enum ChromaDownsampling {
    Automatic,
    Fastest,
    BestQuality,
    Average,
    SharpYuv,
}

pub struct Image {
    width: u32,
    height: u32,
    depth: u32,
    format: Format,
    chroma_upsampling: ChromaUpsampling,
    chroma_downsampling: ChromaDownsampling,
    avoid_libyuv: bool,
    ignore_alpha: bool,
    alpha_premultiplied: bool,
    is_float: bool,
    max_threads: i32,
    pixels: *mut u8, // TODO: slice?
    row_bytes: u32,
}

struct RGBColorSpaceInfo {
    channel_bytes: u32,
    pixel_bytes: u32,
    offset_bytes_r: u32,
    offset_bytes_g: u32,
    offset_bytes_b: u32,
    offset_bytes_a: u32,
    max_channel: i32,
    max_channel_f: f32,
}

enum Mode {
    YuvCoefficients,
    Identity,
    YCgCo,
}

struct YUVColorSpaceInfo {
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

struct State {
    rgb: RGBColorSpaceInfo,
    yuv: YUVColorSpaceInfo,
}

impl Image {
    fn prepare_state(&self, image: &image::Image) -> AvifResult<State> {
        Err(AvifError::ReformatFailed)
    }

    pub fn convert_from_yuv(&mut self, image: &image::Image) -> AvifResult<()> {
        // TODO: use plane constant.
        if image.planes[0].is_none() {
            return Err(AvifError::ReformatFailed);
        }
        let state = self.prepare_state(image)?;
        Err(AvifError::ReformatFailed)
    }
}
