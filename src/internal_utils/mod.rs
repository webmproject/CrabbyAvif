pub mod io;
pub mod pixels;
pub mod stream;

use crate::parser::mp4box::*;
use crate::utils::clap::*;
use crate::*;

#[derive(Copy, Clone, Default, Debug)]
pub struct Fraction(pub i32, pub u32);
#[derive(Copy, Clone, Default, Debug)]
pub struct UFraction(pub u32, pub u32);
#[derive(Copy, Clone, Default, Debug)]
pub struct IFraction(pub i32, pub i32);

impl TryFrom<UFraction> for IFraction {
    type Error = AvifError;

    fn try_from(uf: UFraction) -> AvifResult<IFraction> {
        Ok(IFraction(uf.0 as i32, i32_from_u32(uf.1)?))
    }
}

impl IFraction {
    fn gcd(a: i32, b: i32) -> i32 {
        let mut a = if a < 0 { -a as i64 } else { a as i64 };
        let mut b = if b < 0 { -b as i64 } else { b as i64 };
        while b != 0 {
            let r = a % b;
            a = b;
            b = r;
        }
        a as i32
    }

    pub fn simplified(n: i32, d: i32) -> Self {
        let mut fraction = IFraction(n, d);
        fraction.simplify();
        fraction
    }

    pub fn simplify(&mut self) {
        let gcd = Self::gcd(self.0, self.1);
        if gcd > 1 {
            self.0 /= gcd;
            self.1 /= gcd;
        }
    }

    pub fn get_i32(&self) -> i32 {
        assert!(self.1 != 0);
        self.0 / self.1
    }

    pub fn get_u32(&self) -> AvifResult<u32> {
        u32_from_i32(self.get_i32())
    }

    pub fn is_integer(&self) -> bool {
        self.0 % self.1 == 0
    }

    fn common_denominator(&mut self, val: &mut IFraction) -> AvifResult<()> {
        self.simplify();
        if self.1 == val.1 {
            return Ok(());
        }
        let self_d = self.1;
        self.0 = self.0.checked_mul(val.1).ok_or(AvifError::UnknownError)?;
        self.1 = self.1.checked_mul(val.1).ok_or(AvifError::UnknownError)?;
        val.0 = val.0.checked_mul(self_d).ok_or(AvifError::UnknownError)?;
        val.1 = val.1.checked_mul(self_d).ok_or(AvifError::UnknownError)?;
        Ok(())
    }

    pub fn add(&mut self, val: &IFraction) -> AvifResult<()> {
        let mut val = *val;
        val.simplify();
        self.common_denominator(&mut val)?;
        self.0 = self.0.checked_add(val.0).ok_or(AvifError::UnknownError)?;
        self.simplify();
        Ok(())
    }

    pub fn sub(&mut self, val: &IFraction) -> AvifResult<()> {
        let mut val = *val;
        val.simplify();
        self.common_denominator(&mut val)?;
        self.0 = self.0.checked_sub(val.0).ok_or(AvifError::UnknownError)?;
        self.simplify();
        Ok(())
    }
}

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
#[cfg(feature = "android_mediacodec")]
conversion_function!(usize_from_isize, usize, isize);
conversion_function!(u64_from_usize, u64, usize);
conversion_function!(u32_from_usize, u32, usize);
conversion_function!(u32_from_u64, u32, u64);
conversion_function!(u32_from_i32, u32, i32);
conversion_function!(i32_from_u32, i32, u32);
#[cfg(feature = "android_mediacodec")]
conversion_function!(isize_from_i32, isize, i32);
#[cfg(feature = "capi")]
conversion_function!(isize_from_u32, isize, u32);
conversion_function!(isize_from_usize, isize, usize);

macro_rules! clamp_function {
    ($func:ident, $type:ty) => {
        pub fn $func(value: $type, low: $type, high: $type) -> $type {
            if value < low {
                low
            } else if value > high {
                high
            } else {
                value
            }
        }
    };
}

clamp_function!(clamp_u16, u16);
clamp_function!(clamp_f32, f32);
clamp_function!(clamp_i32, i32);

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
    if height == 0 {
        return false;
    }
    if width > size_limit / height {
        return false;
    }
    if dimension_limit != 0 && (width > dimension_limit || height > dimension_limit) {
        return false;
    }
    true
}

fn limited_to_full(min: i32, max: i32, full: i32, v: u16) -> u16 {
    let v = v as i32;
    clamp_i32(
        (((v - min) * full) + ((max - min) / 2)) / (max - min),
        0,
        full,
    ) as u16
}

pub fn limited_to_full_y(depth: u8, v: u16) -> u16 {
    match depth {
        8 => limited_to_full(16, 235, 255, v),
        10 => limited_to_full(64, 940, 1023, v),
        12 => limited_to_full(256, 3760, 4095, v),
        _ => 0,
    }
}

pub fn create_vec_exact<T>(size: usize) -> AvifResult<Vec<T>> {
    let mut v = Vec::<T>::new();
    if v.try_reserve_exact(size).is_err() {
        return Err(AvifError::OutOfMemory);
    }
    Ok(v)
}

#[cfg(test)]
pub fn assert_f32_array(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        assert!((a[i] - b[i]).abs() <= std::f32::EPSILON);
    }
}
