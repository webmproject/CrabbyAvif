use super::image::*;
use super::types::*;

use crate::utils::clap::*;

pub type avifCropRect = CropRect;

#[no_mangle]
pub unsafe extern "C" fn avifCropRectConvertCleanApertureBox(
    cropRect: *mut avifCropRect,
    clap: *const avifCleanApertureBox,
    imageW: u32,
    imageH: u32,
    yuvFormat: avifPixelFormat,
    _diag: *mut avifDiagnostics,
) -> avifBool {
    let rust_clap: CleanAperture = (&(*clap)).into();
    *cropRect = match CropRect::create_from(&rust_clap, imageW, imageH, yuvFormat.into()) {
        Ok(x) => x,
        Err(_) => return AVIF_FALSE,
    };
    AVIF_TRUE
}
