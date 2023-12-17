use super::rgb;
use super::rgb::*;

use crate::image;
use crate::image::*;
use crate::internal_utils::*;
use crate::reformat::bindings::libyuv::*;
use crate::*;

use std::os::raw::c_int;

fn find_constants(image: &image::Image) -> Option<(&YuvConstants, &YuvConstants)> {
    let matrix_coefficients = if image.yuv_format == PixelFormat::Monochrome
        && image.matrix_coefficients == MatrixCoefficients::Identity
    {
        MatrixCoefficients::Bt601
    } else {
        image.matrix_coefficients
    };
    /*
    // TODO: workaround to allow identity for now.
    let matrix_coefficients = if matrix_coefficients == MatrixCoefficients::Identity {
        MatrixCoefficients::Bt601
    } else {
        matrix_coefficients
    };
    */
    unsafe {
        match image.full_range {
            true => match matrix_coefficients {
                MatrixCoefficients::Bt709 => Some((&kYuvF709Constants, &kYvuF709Constants)),
                MatrixCoefficients::Bt470bg
                | MatrixCoefficients::Bt601
                | MatrixCoefficients::Unspecified => Some((&kYuvJPEGConstants, &kYvuJPEGConstants)),
                MatrixCoefficients::Bt2020Ncl => Some((&kYuvV2020Constants, &kYvuV2020Constants)),
                MatrixCoefficients::ChromaDerivedNcl => match image.color_primaries {
                    ColorPrimaries::Srgb | ColorPrimaries::Unspecified => {
                        Some((&kYuvF709Constants, &kYvuF709Constants))
                    }
                    ColorPrimaries::Bt470bg | ColorPrimaries::Bt601 => {
                        Some((&kYuvJPEGConstants, &kYvuJPEGConstants))
                    }
                    ColorPrimaries::Bt2020 => Some((&kYuvV2020Constants, &kYvuV2020Constants)),
                    _ => None,
                },
                _ => None,
            },
            false => match matrix_coefficients {
                MatrixCoefficients::Bt709 => Some((&kYuvH709Constants, &kYvuH709Constants)),
                MatrixCoefficients::Bt470bg
                | MatrixCoefficients::Bt601
                | MatrixCoefficients::Unspecified => Some((&kYuvI601Constants, &kYvuI601Constants)),
                MatrixCoefficients::Bt2020Ncl => Some((&kYuv2020Constants, &kYvu2020Constants)),
                MatrixCoefficients::ChromaDerivedNcl => match image.color_primaries {
                    ColorPrimaries::Srgb | ColorPrimaries::Unspecified => {
                        Some((&kYuvH709Constants, &kYvuH709Constants))
                    }
                    ColorPrimaries::Bt470bg | ColorPrimaries::Bt601 => {
                        Some((&kYuvI601Constants, &kYvuI601Constants))
                    }
                    ColorPrimaries::Bt2020 => Some((&kYuv2020Constants, &kYvu2020Constants)),
                    _ => None,
                },
                _ => None,
            },
        }
    }
}

#[rustfmt::skip]
type YUV400ToRGBMatrix = unsafe extern "C" fn(
    *const u8, c_int, *mut u8, c_int, *const YuvConstants, c_int, c_int) -> c_int;
#[rustfmt::skip]
type YUVToRGBMatrixFilter = unsafe extern "C" fn(
    *const u8, c_int, *const u8, c_int, *const u8, c_int, *mut u8, c_int, *const YuvConstants,
    c_int, c_int, FilterMode) -> c_int;
#[rustfmt::skip]
type YUVAToRGBMatrixFilter = unsafe extern "C" fn(
    *const u8, c_int, *const u8, c_int, *const u8, c_int, *const u8, c_int, *mut u8, c_int,
    *const YuvConstants, c_int, c_int, c_int, FilterMode) -> c_int;
#[rustfmt::skip]
type YUVToRGBMatrix = unsafe extern "C" fn(
    *const u8, c_int, *const u8, c_int, *const u8, c_int, *mut u8, c_int, *const YuvConstants,
    c_int, c_int) -> c_int;
#[rustfmt::skip]
type YUVAToRGBMatrix = unsafe extern "C" fn(
    *const u8, c_int, *const u8, c_int, *const u8, c_int, *const u8, c_int, *mut u8, c_int,
    *const YuvConstants, c_int, c_int, c_int) -> c_int;
#[rustfmt::skip]
type YUVToRGBMatrixFilterHighBitDepth = unsafe extern "C" fn(
    *const u16, c_int, *const u16, c_int, *const u16, c_int, *mut u8, c_int, *const YuvConstants,
    c_int, c_int, FilterMode) -> c_int;
#[rustfmt::skip]
type YUVAToRGBMatrixFilterHighBitDepth = unsafe extern "C" fn(
    *const u16, c_int, *const u16, c_int, *const u16, c_int, *const u16, c_int, *mut u8, c_int,
    *const YuvConstants, c_int, c_int, c_int, FilterMode) -> c_int;
#[rustfmt::skip]
type YUVToRGBMatrixHighBitDepth = unsafe extern "C" fn(
    *const u16, c_int, *const u16, c_int, *const u16, c_int, *mut u8, c_int, *const YuvConstants,
    c_int, c_int) -> c_int;
#[rustfmt::skip]
type YUVAToRGBMatrixHighBitDepth = unsafe extern "C" fn(
    *const u16, c_int, *const u16, c_int, *const u16, c_int, *const u16, c_int, *mut u8, c_int,
    *const YuvConstants, c_int, c_int, c_int) -> c_int;

#[derive(Debug)]
enum ConversionFunction {
    YUV400ToRGBMatrix(YUV400ToRGBMatrix),
    YUVToRGBMatrixFilter(YUVToRGBMatrixFilter),
    YUVAToRGBMatrixFilter(YUVAToRGBMatrixFilter),
    YUVToRGBMatrix(YUVToRGBMatrix),
    YUVAToRGBMatrix(YUVAToRGBMatrix),
    YUVToRGBMatrixFilterHighBitDepth(YUVToRGBMatrixFilterHighBitDepth),
    YUVAToRGBMatrixFilterHighBitDepth(YUVAToRGBMatrixFilterHighBitDepth),
    YUVToRGBMatrixHighBitDepth(YUVToRGBMatrixHighBitDepth),
    YUVAToRGBMatrixHighBitDepth(YUVAToRGBMatrixHighBitDepth),
}

impl ConversionFunction {
    fn is_yuva(&self) -> bool {
        matches!(
            self,
            ConversionFunction::YUVAToRGBMatrixFilter(_)
                | ConversionFunction::YUVAToRGBMatrix(_)
                | ConversionFunction::YUVAToRGBMatrixFilterHighBitDepth(_)
                | ConversionFunction::YUVAToRGBMatrixHighBitDepth(_)
        )
    }
}

#[allow(clippy::single_match)]
fn find_conversion_function(
    yuv_format: PixelFormat,
    yuv_depth: u8,
    rgb: &rgb::Image,
    alpha_preferred: bool,
) -> Option<ConversionFunction> {
    if yuv_depth > 8 {
        if yuv_format != PixelFormat::Yuv444 {
            if alpha_preferred {
                match yuv_depth {
                    10 => match rgb.format {
                        Format::Rgba | Format::Bgra => match yuv_format {
                            PixelFormat::Yuv422 => {
                                return Some(ConversionFunction::YUVAToRGBMatrixFilterHighBitDepth(
                                    I210AlphaToARGBMatrixFilter,
                                ))
                            }
                            PixelFormat::Yuv420 => {
                                return Some(ConversionFunction::YUVAToRGBMatrixFilterHighBitDepth(
                                    I010AlphaToARGBMatrixFilter,
                                ))
                            }
                            _ => {}
                        },
                        _ => {}
                    },
                    _ => {}
                }
            }
            match yuv_depth {
                10 => match rgb.format {
                    Format::Rgba | Format::Bgra => match yuv_format {
                        PixelFormat::Yuv422 => {
                            return Some(ConversionFunction::YUVToRGBMatrixFilterHighBitDepth(
                                I210ToARGBMatrixFilter,
                            ))
                        }
                        PixelFormat::Yuv420 => {
                            return Some(ConversionFunction::YUVToRGBMatrixFilterHighBitDepth(
                                I010ToARGBMatrixFilter,
                            ))
                        }
                        _ => {}
                    },
                    _ => {}
                },
                _ => {}
            }
        }
        if yuv_format == PixelFormat::Yuv444
            || rgb.chroma_upsampling.nearest_neighbor_filter_allowed()
        {
            if alpha_preferred {
                match yuv_depth {
                    10 => match rgb.format {
                        Format::Rgba | Format::Bgra => match yuv_format {
                            PixelFormat::Yuv444 => {
                                return Some(ConversionFunction::YUVAToRGBMatrixHighBitDepth(
                                    I410AlphaToARGBMatrix,
                                ))
                            }
                            PixelFormat::Yuv422 => {
                                return Some(ConversionFunction::YUVAToRGBMatrixHighBitDepth(
                                    I210AlphaToARGBMatrix,
                                ))
                            }
                            PixelFormat::Yuv420 => {
                                return Some(ConversionFunction::YUVAToRGBMatrixHighBitDepth(
                                    I010AlphaToARGBMatrix,
                                ))
                            }
                            _ => {}
                        },
                        _ => {}
                    },
                    _ => {}
                }
            }
            match yuv_depth {
                10 => match rgb.format {
                    Format::Rgba | Format::Bgra => match yuv_format {
                        PixelFormat::Yuv444 => {
                            return Some(ConversionFunction::YUVToRGBMatrixHighBitDepth(
                                I410ToARGBMatrix,
                            ))
                        }
                        PixelFormat::Yuv422 => {
                            return Some(ConversionFunction::YUVToRGBMatrixHighBitDepth(
                                I210ToARGBMatrix,
                            ))
                        }
                        PixelFormat::Yuv420 => {
                            return Some(ConversionFunction::YUVToRGBMatrixHighBitDepth(
                                I010ToARGBMatrix,
                            ))
                        }
                        _ => {}
                    },
                    _ => {}
                },
                12 => match rgb.format {
                    Format::Rgba | Format::Bgra => match yuv_format {
                        PixelFormat::Yuv420 => {
                            return Some(ConversionFunction::YUVToRGBMatrixHighBitDepth(
                                I012ToARGBMatrix,
                            ))
                        }
                        _ => {}
                    },
                    _ => {}
                },
                _ => {}
            }
        }
        // The fall through here is intentional. If a high bitdepth function was not found, try to
        // see if we can use a low bitdepth function with a downshift.
    }
    if yuv_format == PixelFormat::Monochrome {
        return match rgb.format {
            Format::Rgba | Format::Bgra => {
                Some(ConversionFunction::YUV400ToRGBMatrix(I400ToARGBMatrix))
            }
            _ => None,
        };
    }
    if yuv_format != PixelFormat::Yuv444 {
        if alpha_preferred {
            match rgb.format {
                Format::Rgba | Format::Bgra => match yuv_format {
                    PixelFormat::Yuv422 => {
                        return Some(ConversionFunction::YUVAToRGBMatrixFilter(
                            I422AlphaToARGBMatrixFilter,
                        ))
                    }
                    PixelFormat::Yuv420 => {
                        return Some(ConversionFunction::YUVAToRGBMatrixFilter(
                            I420AlphaToARGBMatrixFilter,
                        ))
                    }
                    _ => {}
                },
                _ => {}
            }
        }
        match rgb.format {
            Format::Rgb | Format::Bgr => match yuv_format {
                PixelFormat::Yuv422 => {
                    return Some(ConversionFunction::YUVToRGBMatrixFilter(
                        I422ToRGB24MatrixFilter,
                    ))
                }
                PixelFormat::Yuv420 => {
                    return Some(ConversionFunction::YUVToRGBMatrixFilter(
                        I420ToRGB24MatrixFilter,
                    ))
                }
                _ => {}
            },
            Format::Rgba | Format::Bgra => match yuv_format {
                PixelFormat::Yuv422 => {
                    return Some(ConversionFunction::YUVToRGBMatrixFilter(
                        I422ToARGBMatrixFilter,
                    ))
                }
                PixelFormat::Yuv420 => {
                    return Some(ConversionFunction::YUVToRGBMatrixFilter(
                        I420ToARGBMatrixFilter,
                    ))
                }
                _ => {}
            },
            _ => {}
        }
        if !rgb.chroma_upsampling.nearest_neighbor_filter_allowed() {
            return None;
        }
    }
    if alpha_preferred {
        match rgb.format {
            Format::Rgba | Format::Bgra => match yuv_format {
                PixelFormat::Yuv444 => {
                    return Some(ConversionFunction::YUVAToRGBMatrix(I444AlphaToARGBMatrix))
                }
                PixelFormat::Yuv422 => {
                    return Some(ConversionFunction::YUVAToRGBMatrix(I422AlphaToARGBMatrix))
                }
                PixelFormat::Yuv420 => {
                    return Some(ConversionFunction::YUVAToRGBMatrix(I420AlphaToARGBMatrix))
                }
                _ => {}
            },
            _ => {}
        }
    }
    match rgb.format {
        Format::Rgb | Format::Bgr => match yuv_format {
            PixelFormat::Yuv444 => {
                return Some(ConversionFunction::YUVToRGBMatrix(I444ToRGB24Matrix))
            }
            PixelFormat::Yuv420 => {
                return Some(ConversionFunction::YUVToRGBMatrix(I420ToRGB24Matrix))
            }
            _ => {}
        },
        Format::Rgba | Format::Bgra => match yuv_format {
            PixelFormat::Yuv444 => {
                return Some(ConversionFunction::YUVToRGBMatrix(I444ToARGBMatrix))
            }
            PixelFormat::Yuv422 => {
                return Some(ConversionFunction::YUVToRGBMatrix(I422ToARGBMatrix))
            }
            PixelFormat::Yuv420 => {
                return Some(ConversionFunction::YUVToRGBMatrix(I420ToARGBMatrix))
            }
            _ => {}
        },
        Format::Argb | Format::Abgr => match yuv_format {
            PixelFormat::Yuv422 => {
                return Some(ConversionFunction::YUVToRGBMatrix(I422ToRGBAMatrix))
            }
            PixelFormat::Yuv420 => {
                return Some(ConversionFunction::YUVToRGBMatrix(I420ToRGBAMatrix))
            }
            _ => {}
        },
        Format::Rgb565 => match yuv_format {
            PixelFormat::Yuv422 => {
                return Some(ConversionFunction::YUVToRGBMatrix(I422ToRGB565Matrix))
            }
            PixelFormat::Yuv420 => {
                return Some(ConversionFunction::YUVToRGBMatrix(I420ToRGB565Matrix))
            }
            _ => {}
        },
    }
    None
}

pub fn yuv_to_rgb(
    image: &image::Image,
    rgb: &mut rgb::Image,
    reformat_alpha: bool,
) -> AvifResult<bool> {
    if rgb.depth != 8 || (image.depth != 8 && image.depth != 10 && image.depth != 12) {
        return Err(AvifError::NotImplemented);
    }
    let (matrix_yuv, matrix_yvu) = find_constants(image).ok_or(AvifError::NotImplemented)?;
    let alpha_preferred = reformat_alpha && image.has_alpha();
    let conversion_function =
        find_conversion_function(image.yuv_format, image.depth, rgb, alpha_preferred)
            .ok_or(AvifError::NotImplemented)?;
    println!("conversion_function: {:#?}", conversion_function);
    let is_yvu = matches!(rgb.format, Format::Rgb | Format::Rgba | Format::Argb);
    let matrix = if is_yvu { matrix_yvu } else { matrix_yuv };
    let u_plane_index: usize = if is_yvu { 2 } else { 1 };
    let v_plane_index: usize = if is_yvu { 1 } else { 2 };
    let filter = if rgb.chroma_upsampling.nearest_neighbor_filter_allowed() {
        FilterMode_kFilterNone
    } else {
        FilterMode_kFilterBilinear
    };
    let mut pd: [Option<PlaneData>; 4] = [
        image.plane(Plane::Y),
        image.plane(Plane::U),
        image.plane(Plane::V),
        image.plane(Plane::A),
    ];
    let mut plane_u8: [*const u8; 4] = pd
        .iter()
        .map(|x| {
            if x.is_some() {
                match x.as_ref().unwrap().data {
                    Some(data) => data.as_ptr(),
                    None => std::ptr::null(),
                }
            } else {
                std::ptr::null()
            }
        })
        .collect::<Vec<*const u8>>()
        .try_into()
        .unwrap();
    let plane_u16: [*const u16; 4] = pd
        .iter()
        .map(|x| {
            if x.is_some() {
                match x.as_ref().unwrap().data16 {
                    Some(data16) => data16.as_ptr(),
                    None => std::ptr::null(),
                }
            } else {
                std::ptr::null()
            }
        })
        .collect::<Vec<*const u16>>()
        .try_into()
        .unwrap();
    let mut plane_row_bytes: [i32; 4] = pd
        .iter()
        .map(|x| {
            if x.is_some() {
                i32_from_u32(x.as_ref().unwrap().row_bytes).unwrap_or_default()
            } else {
                0
            }
        })
        .collect::<Vec<i32>>()
        .try_into()
        .unwrap();
    let rgb_row_bytes = i32_from_u32(rgb.row_bytes)?;
    let width = i32_from_u32(image.width)?;
    let height = i32_from_u32(image.height)?;
    let mut result: c_int;
    unsafe {
        let mut high_bd_matched = true;
        // Apply one of the high bitdepth functions if possible.
        result = match conversion_function {
            ConversionFunction::YUVToRGBMatrixFilterHighBitDepth(func) => func(
                plane_u16[0],
                plane_row_bytes[0] / 2,
                plane_u16[u_plane_index],
                plane_row_bytes[u_plane_index] / 2,
                plane_u16[v_plane_index],
                plane_row_bytes[v_plane_index] / 2,
                rgb.pixels(),
                rgb_row_bytes,
                matrix,
                width,
                height,
                filter,
            ),
            ConversionFunction::YUVAToRGBMatrixFilterHighBitDepth(func) => func(
                plane_u16[0],
                plane_row_bytes[0] / 2,
                plane_u16[u_plane_index],
                plane_row_bytes[u_plane_index] / 2,
                plane_u16[v_plane_index],
                plane_row_bytes[v_plane_index] / 2,
                plane_u16[3],
                plane_row_bytes[3] / 2,
                rgb.pixels(),
                rgb_row_bytes,
                matrix,
                width,
                height,
                0, // attenuate
                filter,
            ),
            ConversionFunction::YUVToRGBMatrixHighBitDepth(func) => func(
                plane_u16[0],
                plane_row_bytes[0] / 2,
                plane_u16[u_plane_index],
                plane_row_bytes[u_plane_index] / 2,
                plane_u16[v_plane_index],
                plane_row_bytes[v_plane_index] / 2,
                rgb.pixels(),
                rgb_row_bytes,
                matrix,
                width,
                height,
            ),
            ConversionFunction::YUVAToRGBMatrixHighBitDepth(func) => func(
                plane_u16[0],
                plane_row_bytes[0] / 2,
                plane_u16[u_plane_index],
                plane_row_bytes[u_plane_index] / 2,
                plane_u16[v_plane_index],
                plane_row_bytes[v_plane_index] / 2,
                plane_u16[3],
                plane_row_bytes[3] / 2,
                rgb.pixels(),
                rgb_row_bytes,
                matrix,
                width,
                height,
                0, // attenuate
            ),
            _ => {
                high_bd_matched = false;
                -1
            }
        };
        if high_bd_matched {
            return if result == 0 {
                Ok(!image.has_alpha() || conversion_function.is_yuva())
            } else {
                Err(AvifError::ReformatFailed)
            };
        }
        let mut image8 = image::Image::default();
        if image.depth > 8 {
            downshift_to_8bit(image, &mut image8, conversion_function.is_yuva())?;
            pd = [
                image8.plane(Plane::Y),
                image8.plane(Plane::U),
                image8.plane(Plane::V),
                image8.plane(Plane::A),
            ];
            plane_u8 = pd
                .iter()
                .map(|x| {
                    if x.is_some() {
                        x.as_ref().unwrap().data.unwrap().as_ptr()
                    } else {
                        std::ptr::null()
                    }
                })
                .collect::<Vec<*const u8>>()
                .try_into()
                .unwrap();
            plane_row_bytes = pd
                .iter()
                .map(|x| {
                    if x.is_some() {
                        i32_from_u32(x.as_ref().unwrap().row_bytes).unwrap_or_default()
                    } else {
                        0
                    }
                })
                .collect::<Vec<i32>>()
                .try_into()
                .unwrap();
        }
        result = match conversion_function {
            ConversionFunction::YUV400ToRGBMatrix(func) => func(
                plane_u8[0],
                plane_row_bytes[0],
                rgb.pixels(),
                rgb_row_bytes,
                matrix,
                width,
                height,
            ),
            ConversionFunction::YUVToRGBMatrixFilter(func) => func(
                plane_u8[0],
                plane_row_bytes[0],
                plane_u8[u_plane_index],
                plane_row_bytes[u_plane_index],
                plane_u8[v_plane_index],
                plane_row_bytes[v_plane_index],
                rgb.pixels(),
                rgb_row_bytes,
                matrix,
                width,
                height,
                filter,
            ),
            ConversionFunction::YUVAToRGBMatrixFilter(func) => func(
                plane_u8[0],
                plane_row_bytes[0],
                plane_u8[u_plane_index],
                plane_row_bytes[u_plane_index],
                plane_u8[v_plane_index],
                plane_row_bytes[v_plane_index],
                plane_u8[3],
                plane_row_bytes[3],
                rgb.pixels(),
                rgb_row_bytes,
                matrix,
                width,
                height,
                0, // attenuate
                filter,
            ),
            ConversionFunction::YUVToRGBMatrix(func) => func(
                plane_u8[0],
                plane_row_bytes[0],
                plane_u8[u_plane_index],
                plane_row_bytes[u_plane_index],
                plane_u8[v_plane_index],
                plane_row_bytes[v_plane_index],
                rgb.pixels(),
                rgb_row_bytes,
                matrix,
                width,
                height,
            ),
            ConversionFunction::YUVAToRGBMatrix(func) => func(
                plane_u8[0],
                plane_row_bytes[0],
                plane_u8[u_plane_index],
                plane_row_bytes[u_plane_index],
                plane_u8[v_plane_index],
                plane_row_bytes[v_plane_index],
                plane_u8[3],
                plane_row_bytes[3],
                rgb.pixels(),
                rgb_row_bytes,
                matrix,
                width,
                height,
                0, // attenuate
            ),
            _ => 0,
        };
    }
    if result == 0 {
        Ok(!image.has_alpha() || conversion_function.is_yuva())
    } else {
        Err(AvifError::ReformatFailed)
    }
}

fn downshift_to_8bit(
    image: &image::Image,
    image8: &mut image::Image,
    alpha: bool,
) -> AvifResult<()> {
    image8.width = image.width;
    image8.height = image.height;
    image8.depth = 8;
    image8.yuv_format = image.yuv_format;
    image8.allocate_planes(0)?;
    if alpha {
        image8.allocate_planes(1)?;
    }
    let scale = 1 << (24 - image.depth);
    for plane in ALL_PLANES {
        if plane == Plane::A && !alpha {
            continue;
        }
        let pd = image.plane(plane);
        if pd.is_none() {
            continue;
        }
        let pd = pd.unwrap();
        if pd.width == 0 {
            continue;
        }
        let pd8 = image8.plane_mut(plane).unwrap();
        unsafe {
            Convert16To8Plane(
                pd.data16.unwrap().as_ptr(),
                i32_from_u32(pd.row_bytes / 2)?,
                pd8.data.unwrap().as_mut_ptr(),
                i32_from_u32(pd8.row_bytes)?,
                scale,
                i32_from_u32(pd.width)?,
                i32_from_u32(pd.height)?,
            );
        }
    }
    Ok(())
}

pub fn process_alpha(rgb: &mut rgb::Image, multiply: bool) -> AvifResult<()> {
    if rgb.depth != 8 {
        return Err(AvifError::NotImplemented);
    }
    match rgb.format {
        Format::Rgba | Format::Bgra => {}
        _ => return Err(AvifError::NotImplemented),
    }
    let result = unsafe {
        if multiply {
            ARGBAttenuate(
                rgb.pixels(),
                i32_from_u32(rgb.row_bytes)?,
                rgb.pixels(),
                i32_from_u32(rgb.row_bytes)?,
                i32_from_u32(rgb.width)?,
                i32_from_u32(rgb.height)?,
            )
        } else {
            ARGBUnattenuate(
                rgb.pixels(),
                i32_from_u32(rgb.row_bytes)?,
                rgb.pixels(),
                i32_from_u32(rgb.row_bytes)?,
                i32_from_u32(rgb.width)?,
                i32_from_u32(rgb.height)?,
            )
        }
    };
    if result == 0 {
        Ok(())
    } else {
        Err(AvifError::ReformatFailed)
    }
}

pub fn convert_to_half_float(rgb: &mut rgb::Image, scale: f32) -> AvifResult<()> {
    let res = unsafe {
        HalfFloatPlane(
            rgb.pixels() as *const u16,
            i32_from_u32(rgb.row_bytes)?,
            rgb.pixels() as *mut u16,
            i32_from_u32(rgb.row_bytes)?,
            scale,
            i32_from_u32(rgb.width * rgb.channel_count())?,
            i32_from_u32(rgb.height)?,
        )
    };
    if res == 0 {
        Ok(())
    } else {
        Err(AvifError::InvalidArgument)
    }
}
