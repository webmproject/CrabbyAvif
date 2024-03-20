pub mod io;
pub mod pixels;
pub mod stream;

use crate::parser::mp4box::*;
use crate::utils::clap::*;
use crate::*;

#[derive(Copy, Clone, Default, Debug)]
pub struct Fraction {
    pub n: u32, // numerator
    pub d: u32, // denominator
    pub is_negative: bool,
}

impl Fraction {
    pub const fn new(n: u32, d: u32) -> Fraction {
        let is_negative = false;
        Fraction { n, d, is_negative }
    }
    pub const fn new_i32(n: i32, d: u32) -> Fraction {
        let is_negative = n < 0;
        let n: u32 = n.unsigned_abs();
        Fraction { n, d, is_negative }
    }

    fn gcd(mut a: u32, mut b: u32) -> u32 {
        while b != 0 {
            let r = a % b;
            a = b;
            b = r;
        }
        a
    }

    pub fn negate(self) -> Fraction {
        Fraction {
            n: self.n,
            d: self.d,
            is_negative: !self.is_negative,
        }
    }

    pub fn simplify(self) -> AvifResult<Fraction> {
        let gcd = Self::gcd(self.n, self.d);
        Ok(Fraction {
            n: self.n.checked_div(gcd).ok_or(AvifError::UnknownError)?,
            d: self.d.checked_div(gcd).ok_or(AvifError::UnknownError)?,
            is_negative: self.is_negative,
        })
    }

    pub fn get_u32(self) -> AvifResult<u32> {
        if self.is_negative || !self.is_integer() {
            return Err(AvifError::UnknownError);
        }
        self.n.checked_div(self.d).ok_or(AvifError::UnknownError)
    }

    pub fn is_integer(self) -> bool {
        self.n % self.d == 0
    }

    fn common_denominator(a: Fraction, b: Fraction) -> AvifResult<(Fraction, Fraction)> {
        if a.d == b.d {
            return Ok((a, b));
        }
        let common_d = a.d.checked_mul(b.d).ok_or(AvifError::UnknownError)?;
        Ok((
            Fraction {
                n: a.n.checked_mul(b.d).ok_or(AvifError::UnknownError)?,
                d: common_d,
                is_negative: a.is_negative,
            },
            Fraction {
                n: b.n.checked_mul(a.d).ok_or(AvifError::UnknownError)?,
                d: common_d,
                is_negative: b.is_negative,
            },
        ))
    }

    pub fn add(self, val: Fraction) -> AvifResult<Fraction> {
        let (a, b) = Self::common_denominator(self.simplify()?, val.simplify()?)?;
        if a.is_negative == b.is_negative {
            Ok(Fraction {
                n: a.n.checked_add(b.n).ok_or(AvifError::UnknownError)?,
                d: a.d,
                is_negative: a.is_negative,
            })
        } else if a.n >= b.n {
            Ok(Fraction {
                n: a.n - b.n,
                d: a.d,
                is_negative: a.is_negative,
            })
        } else {
            Ok(Fraction {
                n: b.n - a.n,
                d: a.d,
                is_negative: !a.is_negative,
            })
        }
    }

    pub fn sub(self, val: Fraction) -> AvifResult<Fraction> {
        self.add(val.negate())
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
    //println!("w: {width} h: {height} s: {size_limit} d: {dimension_limit}");
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

pub fn reinterpret_f32_as_u32(f: f32) -> u32 {
    u32::from_be_bytes(f.to_be_bytes())
}

#[cfg(test)]
pub fn assert_f32_array(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        assert!((a[i] - b[i]).abs() <= std::f32::EPSILON);
    }
}
