use crate::image::Image;
use std::fs::File;
use std::io::prelude::*;

#[derive(Default)]
pub struct RawWriter {
    pub filename: Option<String>,
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
        for plane in 0usize..4 {
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
