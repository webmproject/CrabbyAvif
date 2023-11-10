use crate::decoder::AvifImage;
use std::fs::File;
use std::io::prelude::*;

pub fn y4m_write(filename: &String, image: &AvifImage) -> bool {
    let has_alpha = image.alpha_plane.is_some() && image.alpha_row_bytes > 0;
    let mut write_alpha = false;

    if has_alpha && (image.depth != 8 || image.yuv_format != 3) {
        println!("WARNING: writing alpha is currently only supported in 8bpc YUV444, ignoring alpha channel");
    }

    // do some has alpha stuff.

    let y4m_format = match image.depth {
        8 => match image.yuv_format {
            3 => {
                if has_alpha {
                    write_alpha = true;
                    "C444alpha XYSCSS=444"
                } else {
                    "C444 XYSCSS=444"
                }
            }
            2 => "C422 XYSCSS=422",
            1 => "C420jpeg XYSCSS=420JPEG",
            0 => "Cmono XYSCSS=400",
            _ => return false,
        },
        10 => match image.yuv_format {
            3 => "C444p10 XYSCSS=444P10",
            2 => "C422p10 XYSCSS=422P10",
            1 => "C420p10 XYSCSS=420P10",
            0 => "Cmono10 XYSCSS=400",
            _ => return false,
        },
        12 => match image.yuv_format {
            3 => "C444p12 XYSCSS=444P12",
            2 => "C422p12 XYSCSS=422P12",
            1 => "C420p12 XYSCSS=420P12",
            0 => "Cmono12 XYSCSS=400",
            _ => return false,
        },
        _ => return false,
    };
    let y4m_color_range = if image.full_range {
        "XCOLORRANGE=FULL"
    } else {
        "XCOLORRANGE=LIMITED"
    };
    let header = format!(
        "YUV4MPEG2 W{} H{} F25:1 Ip A0:0 {y4m_format} {y4m_color_range}\nFRAME\n",
        image.width, image.height
    );
    println!("{header}");

    let file = File::create(filename);
    if !file.is_ok() {
        return false;
    }
    let mut file = file.unwrap();
    match file.write_all(header.as_bytes()) {
        Err(e) => return false,
        _ => {}
    }
    let plane_count = if write_alpha { 4 } else { 3 };
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
            let stride_offset: isize = (y * avif_plane.row_bytes).try_into().unwrap();
            //println!("{y}: {stride_offset}");
            let ptr = unsafe { avif_plane.data.offset(stride_offset) };
            let pixels = unsafe { std::slice::from_raw_parts(ptr, byte_count) };
            match file.write_all(pixels) {
                Err(e) => return false,
                _ => {}
            }
        }
    }
    true
}
