#![allow(dead_code, unused)] // TODO: remove

use super::gainmap::*;
use super::image::*;
use super::io::*;
use super::types::*;

use crate::image;
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
            pixels: Some(rgb::Pixels::Pointer(rgb.pixels)),
            row_bytes: rgb.row_bytes,
            pixel_buffer: Vec::new(),
        }
    }
}

impl From<*const avifImage> for image::Image {
    // Only copies fields necessary for reformatting.
    fn from(image: *const avifImage) -> image::Image {
        let image = unsafe { &(*image) };
        let mut ret = image::Image::default();
        ret.width = image.width;
        ret.height = image.height;
        ret.depth = image.depth as u8;
        ret.yuv_format = image.yuvFormat.into();
        ret.full_range = image.yuvRange == avifRange::Full;
        ret.alpha_present = !image.alphaPlane.is_null();
        ret.alpha_premultiplied = image.alphaPremultiplied == AVIF_TRUE;
        ret.planes = [
            Some(image.yuvPlanes[0]),
            Some(image.yuvPlanes[1]),
            Some(image.yuvPlanes[2]),
            Some(image.alphaPlane),
        ];
        ret.row_bytes = [
            image.yuvRowBytes[0],
            image.yuvRowBytes[1],
            image.yuvRowBytes[2],
            image.alphaRowBytes,
        ];
        ret.color_primaries = image.colorPrimaries;
        ret.transfer_characteristics = image.transferCharacteristics;
        ret.matrix_coefficients = image.matrixCoefficients;
        ret
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
    let mut rgb: rgb::Image = rgb.into();
    let image: image::Image = image.into();
    to_avifResult(&rgb.convert_from_yuv(&image))
}
