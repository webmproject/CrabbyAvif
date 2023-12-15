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
            rgb.format = rgb::Format::Bgr;
            rgb.depth = 8;
            //rgb.alpha_premultiplied = true;
            if rgb.allocate().is_err() || rgb.convert_from_yuv(image).is_err() {
                println!("conversion failed");
                return false;
            }
            for y in 0..rgb.height {
                if rgb.depth == 8 {
                    let row = rgb.row(y).unwrap();
                    if self.file.as_ref().unwrap().write_all(row).is_err() {
                        return false;
                    }
                } else {
                    unimplemented!("rgb bitdepth higher than 8");
                    //let row = rgb.row16(y).unwrap();
                    //if self.file.as_ref().unwrap().write_all(row).is_err() {
                    //    return false;
                    //}
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
