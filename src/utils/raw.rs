use crate::image::Image;
use crate::image::ALL_PLANES;
use crate::reformat::rgb;

use std::fs::File;
use std::io::prelude::*;

#[derive(Default)]
pub struct RawWriter {
    pub filename: Option<String>,
    pub rgb: bool,
    file: Option<File>,
}

impl RawWriter {
    pub fn create(filename: &str) -> Self {
        Self {
            filename: Some(filename.to_owned()),
            ..Self::default()
        }
    }

    fn write_header(&mut self) -> bool {
        if self.file.is_none() {
            assert!(self.filename.is_some());
            let file = File::create(self.filename.as_ref().unwrap());
            if file.is_err() {
                return false;
            }
            self.file = Some(file.unwrap());
        }
        true
    }

    pub fn write_frame(&mut self, image: &Image) -> bool {
        if !self.write_header() {
            return false;
        }
        if self.rgb {
            let mut rgb = rgb::Image::create_from_yuv(image);
            rgb.format = rgb::Format::Bgra;
            rgb.depth = 8;
            rgb.alpha_premultiplied = true;
            if let Err(_) = rgb.allocate() {
                return false;
            }
            if let Err(_) = rgb.convert_from_yuv(image) {
                println!("conversion failed");
                return false;
            }
            for y in 0..rgb.height {
                let stride_offset = (y * rgb.row_bytes) as isize;
                let ptr = unsafe { rgb.pixels.offset(stride_offset) };
                let byte_count = (rgb.width * rgb.pixel_size()) as usize;
                let pixels = unsafe { std::slice::from_raw_parts(ptr, byte_count) };
                if self.file.as_ref().unwrap().write_all(pixels).is_err() {
                    return false;
                }
            }
            return true;
        }
        for plane in ALL_PLANES {
            let avif_plane = image.plane(plane);
            println!("{:#?}", avif_plane);
            if avif_plane.is_none() {
                continue;
            }
            let avif_plane = avif_plane.unwrap();
            let byte_count: usize = (avif_plane.width * avif_plane.pixel_size)
                .try_into()
                .unwrap();
            for y in 0..avif_plane.height {
                let stride_offset: usize = (y * avif_plane.row_bytes).try_into().unwrap();
                //println!("{y}: {stride_offset} plane_height: {}", avif_plane.height);
                let pixels = &avif_plane.data[stride_offset..stride_offset + byte_count];
                if self.file.as_ref().unwrap().write_all(pixels).is_err() {
                    return false;
                }
            }
        }
        true
    }
}
