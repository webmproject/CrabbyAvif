#![allow(dead_code, unused)] // TODO: remove

use super::gainmap::*;
use super::image::*;
use super::io::*;
use super::types::*;

use crate::image;
use crate::internal_utils::pixels::*;
use crate::internal_utils::*;
use crate::parser::mp4box::*;
use crate::reformat::rgb;
use crate::utils::clap::*;
use crate::*;

use std::os::raw::c_int;

/// cbindgen:rename-all=CamelCase
#[repr(C)]
pub struct avifRGBImage {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub format: rgb::Format,
    pub chroma_upsampling: rgb::ChromaUpsampling,
    pub chroma_downsampling: rgb::ChromaDownsampling,
    pub ignore_alpha: bool,
    pub alpha_premultiplied: bool,
    pub is_float: bool,
    pub max_threads: i32,
    pub pixels: *mut u8,
    pub row_bytes: u32,
}

impl From<rgb::Image> for avifRGBImage {
    fn from(mut rgb: rgb::Image) -> avifRGBImage {
        avifRGBImage {
            width: rgb.width,
            height: rgb.height,
            depth: rgb.depth,
            format: rgb.format,
            chroma_upsampling: rgb.chroma_upsampling,
            chroma_downsampling: rgb.chroma_downsampling,
            ignore_alpha: rgb.ignore_alpha,
            alpha_premultiplied: rgb.alpha_premultiplied,
            is_float: rgb.is_float,
            max_threads: rgb.max_threads,
            pixels: rgb.pixels(),
            row_bytes: rgb.row_bytes,
        }
    }
}

impl From<*mut avifRGBImage> for rgb::Image {
    fn from(rgb: *mut avifRGBImage) -> rgb::Image {
        let rgb = unsafe { &(*rgb) };
        rgb::Image {
            width: rgb.width,
            height: rgb.height,
            depth: rgb.depth,
            format: rgb.format,
            chroma_upsampling: rgb.chroma_upsampling,
            chroma_downsampling: rgb.chroma_downsampling,
            ignore_alpha: rgb.ignore_alpha,
            alpha_premultiplied: rgb.alpha_premultiplied,
            is_float: rgb.is_float,
            max_threads: rgb.max_threads,
            pixels: Some(Pixels::Pointer(rgb.pixels)),
            row_bytes: rgb.row_bytes,
        }
    }
}

impl From<*const avifImage> for image::Image {
    // Only copies fields necessary for reformatting.
    fn from(image: *const avifImage) -> image::Image {
        let image = unsafe { &(*image) };
        image::Image {
            width: image.width,
            height: image.height,
            depth: image.depth as u8,
            yuv_format: image.yuvFormat.into(),
            full_range: image.yuvRange == avifRange::Full,
            alpha_present: !image.alphaPlane.is_null(),
            alpha_premultiplied: image.alphaPremultiplied == AVIF_TRUE,
            planes2: [
                Some(Pixels::Pointer(image.yuvPlanes[0])),
                Some(Pixels::Pointer(image.yuvPlanes[1])),
                Some(Pixels::Pointer(image.yuvPlanes[2])),
                Some(Pixels::Pointer(image.alphaPlane)),
            ],
            row_bytes: [
                image.yuvRowBytes[0],
                image.yuvRowBytes[1],
                image.yuvRowBytes[2],
                image.alphaRowBytes,
            ],
            color_primaries: image.colorPrimaries,
            transfer_characteristics: image.transferCharacteristics,
            matrix_coefficients: image.matrixCoefficients,
            ..Default::default()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn avifRGBImageSetDefaults(rgb: *mut avifRGBImage, image: *const avifImage) {
    let rgb = unsafe { &mut (*rgb) };
    let image: image::Image = image.into();
    *rgb = rgb::Image::create_from_yuv(&image).into();
}

#[no_mangle]
pub unsafe extern "C" fn avifImageYUVToRGB(
    image: *const avifImage,
    rgb: *mut avifRGBImage,
) -> avifResult {
    unsafe {
        if (*image).yuvPlanes[0].is_null() {
            return avifResult::Ok;
        }
    }
    let mut rgb: rgb::Image = rgb.into();
    let image: image::Image = image.into();
    to_avifResult(&rgb.convert_from_yuv(&image))
}
