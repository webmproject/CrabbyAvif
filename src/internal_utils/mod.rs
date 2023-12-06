pub mod io;
pub mod stream;

use crate::parser::mp4box::*;
use crate::*;

pub type Fraction = (i32, u32);
pub type UFraction = (u32, u32);

macro_rules! conversion_function {
    ($func:ident, $to: ident, $from:ty) => {
        pub fn $func(value: $from) -> AvifResult<$to> {
            $to::try_from(value).or(Err(AvifError::BmffParseFailed))
        }
    };
}

conversion_function!(usize_from_u64, usize, u64);
conversion_function!(usize_from_u32, usize, u32);
conversion_function!(usize_from_u16, usize, u16);
conversion_function!(u64_from_usize, u64, usize);
conversion_function!(u32_from_usize, u32, usize);
conversion_function!(u32_from_u64, u32, u64);

pub fn find_nclx(properties: &[ItemProperty]) -> Result<&Nclx, bool> {
    let nclx_properties: Vec<_> = properties
        .iter()
        .filter(|x| match x {
            ItemProperty::ColorInformation(colr) => matches!(colr, ColorInformation::Nclx(_)),
            _ => false,
        })
        .collect();
    match nclx_properties.len() {
        0 => Err(false),
        1 => match nclx_properties[0] {
            ItemProperty::ColorInformation(ColorInformation::Nclx(nclx)) => Ok(nclx),
            _ => Err(false), // not reached.
        },
        _ => Err(true), // multiple nclx were found.
    }
}

pub fn find_icc(properties: &[ItemProperty]) -> Result<Vec<u8>, bool> {
    let icc_properties: Vec<_> = properties
        .iter()
        .filter(|x| match x {
            ItemProperty::ColorInformation(colr) => matches!(colr, ColorInformation::Icc(_)),
            _ => false,
        })
        .collect();
    match icc_properties.len() {
        0 => Err(false),
        1 => match icc_properties[0] {
            ItemProperty::ColorInformation(ColorInformation::Icc(icc)) => Ok(icc.to_vec()),
            _ => Err(false), // not reached.
        },
        _ => Err(true), // multiple icc were found.
    }
}

#[allow(non_snake_case)]
pub fn find_av1C(properties: &[ItemProperty]) -> Option<&CodecConfiguration> {
    match properties
        .iter()
        .find(|x| matches!(x, ItemProperty::CodecConfiguration(_)))
    {
        Some(ItemProperty::CodecConfiguration(av1C)) => Some(av1C),
        _ => None,
    }
}

macro_rules! find_property_function {
    ($func:ident, $prop: ident, $ret:ty) => {
        pub fn $func(properties: &[ItemProperty]) -> Option<$ret> {
            match properties
                .iter()
                .find(|x| matches!(x, ItemProperty::$prop(_)))
            {
                Some(ItemProperty::$prop(x)) => Some(*x),
                _ => None,
            }
        }
    };
}

find_property_function!(
    find_clli,
    ContentLightLevelInformation,
    ContentLightLevelInformation
);
find_property_function!(find_pasp, PixelAspectRatio, PixelAspectRatio);
find_property_function!(find_clap, CleanAperture, CleanAperture);
find_property_function!(find_irot_angle, ImageRotation, u8);
find_property_function!(find_imir_axis, ImageMirror, u8);

pub fn check_limits(width: u32, height: u32, size_limit: u32, dimension_limit: u32) -> bool {
    println!("w: {width} h: {height} s: {size_limit} d: {dimension_limit}");
    if height == 0 {
        return false;
    }
    if width > size_limit / height {
        return false;
    }
    if dimension_limit != 0 && (width > dimension_limit || height > dimension_limit) {
        return false;
    }
    return true;
}
