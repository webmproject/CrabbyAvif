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
    pixels: *mut u8,
    row_bytes: u32,
}

impl Image {
    pub fn convert_from_yuv(&mut self, image: &image::Image) -> AvifResult<()> {
        Ok(())
    }
}
