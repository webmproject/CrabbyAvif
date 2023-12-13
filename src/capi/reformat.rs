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

pub type avifRGBImage = rgb::Image;

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
    *rgb = rgb::Image::create_from_yuv(&image);
}

#[no_mangle]
pub unsafe extern "C" fn avifImageYUVToRGB(
    image: *const avifImage,
    rgb: *mut avifRGBImage,
) -> avifResult {
    let rgb = unsafe { &mut (*rgb) };
    let image: image::Image = image.into();
    to_avifResult(&rgb.convert_from_yuv(&image))
}
