pub mod io;
pub mod stream;

use crate::parser::mp4box::*;
use crate::*;

pub type Fraction = (i32, u32);
pub type UFraction = (u32, u32);

pub fn usize_from_u64(value: u64) -> AvifResult<usize> {
    usize::try_from(value).or(Err(AvifError::BmffParseFailed))
}

pub fn usize_from_u32(value: u32) -> AvifResult<usize> {
    usize::try_from(value).or(Err(AvifError::BmffParseFailed))
}

pub fn usize_from_u16(value: u16) -> AvifResult<usize> {
    usize::try_from(value).or(Err(AvifError::BmffParseFailed))
}

pub fn u64_from_usize(value: usize) -> AvifResult<u64> {
    u64::try_from(value).or(Err(AvifError::BmffParseFailed))
}

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

pub fn find_clli(properties: &[ItemProperty]) -> Option<&ContentLightLevelInformation> {
    match properties
        .iter()
        .find(|x| matches!(x, ItemProperty::ContentLightLevelInformation(_)))
    {
        Some(ItemProperty::ContentLightLevelInformation(clli)) => Some(clli),
        _ => None,
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
