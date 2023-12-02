use crate::decoder::Image;
use crate::*;
use std::fs::File;
use std::io::prelude::*;

#[derive(Default)]
pub struct Y4MWriter {
    pub filename: Option<String>,
    header_written: bool,
    file: Option<File>,
    write_alpha: bool,
}

impl Y4MWriter {
    pub fn create(filename: &str) -> Self {
        Self {
            filename: Some(filename.to_owned()),
            ..Self::default()
        }
    }

    pub fn create_from_file(file: File) -> Self {
        Self {
            file: Some(file),
            ..Self::default()
        }
    }

    fn write_header(&mut self, image: &Image) -> bool {
        if self.header_written {
            return true;
        }
        self.write_alpha = false;

        if image.info.alpha_present
            && (image.info.depth != 8 || image.info.yuv_format != PixelFormat::Yuv444)
        {
            println!("WARNING: writing alpha is currently only supported in 8bpc YUV444, ignoring alpha channel");
        }

        let y4m_format = match image.info.depth {
            8 => match image.info.yuv_format {
                PixelFormat::Yuv444 => {
                    if image.info.alpha_present {
                        self.write_alpha = true;
                        "C444alpha XYSCSS=444"
                    } else {
                        "C444 XYSCSS=444"
                    }
                }
                PixelFormat::Yuv422 => "C422 XYSCSS=422",
                PixelFormat::Yuv420 => "C420jpeg XYSCSS=420JPEG",
                PixelFormat::Monochrome => "Cmono XYSCSS=400",
            },
            10 => match image.info.yuv_format {
                PixelFormat::Yuv444 => "C444p10 XYSCSS=444P10",
                PixelFormat::Yuv422 => "C422p10 XYSCSS=422P10",
                PixelFormat::Yuv420 => "C420p10 XYSCSS=420P10",
                PixelFormat::Monochrome => "Cmono10 XYSCSS=400",
            },
            12 => match image.info.yuv_format {
                PixelFormat::Yuv444 => "C444p12 XYSCSS=444P12",
                PixelFormat::Yuv422 => "C422p12 XYSCSS=422P12",
                PixelFormat::Yuv420 => "C420p12 XYSCSS=420P12",
                PixelFormat::Monochrome => "Cmono12 XYSCSS=400",
            },
            _ => {
                println!("image depth is invalid: {}", image.info.depth);
                return false;
            }
        };
        let y4m_color_range = if image.info.full_range {
            "XCOLORRANGE=FULL"
        } else {
            "XCOLORRANGE=LIMITED"
        };
        let header = format!(
            "YUV4MPEG2 W{} H{} F25:1 Ip A0:0 {y4m_format} {y4m_color_range}\n",
            image.info.width, image.info.height
        );
        println!("{header}");
        if self.file.is_none() {
            assert!(self.filename.is_some());
            let file = File::create(self.filename.as_ref().unwrap());
            if file.is_err() {
                return false;
            }
            self.file = Some(file.unwrap());
        }
        if self
            .file
            .as_ref()
            .unwrap()
            .write_all(header.as_bytes())
            .is_err()
        {
            return false;
        }
        self.header_written = true;
        true
    }

    pub fn write_frame(&mut self, image: &Image) -> bool {
        if !self.write_header(image) {
            return false;
        }
        let frame_marker = "FRAME\n";
        if self
            .file
            .as_ref()
            .unwrap()
            .write_all(frame_marker.as_bytes())
            .is_err()
        {
            return false;
        }
        let plane_count = if self.write_alpha { 4 } else { 3 };
        for plane in 0usize..plane_count {
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
