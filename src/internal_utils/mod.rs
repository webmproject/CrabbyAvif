// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

pub mod io;
pub mod pixels;
pub mod stream;

use crate::parser::mp4box::*;
use crate::utils::*;
use crate::*;

use std::num::NonZero;
use std::ops::Range;

// 'clap' fractions do not follow this pattern: both numerators and denominators
// are used as i32, but they are signalled as u32 according to the specification
// as of 2024. This may be fixed in later versions of the specification, see
// https://github.com/AOMediaCodec/libavif/pull/1749#discussion_r1391612932.
#[derive(Clone, Copy, Debug, Default)]
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

    pub(crate) fn simplified(n: i32, d: i32) -> Self {
        let mut fraction = IFraction(n, d);
        fraction.simplify();
        fraction
    }

    pub(crate) fn simplify(&mut self) {
        let gcd = Self::gcd(self.0, self.1);
        if gcd > 1 {
            self.0 /= gcd;
            self.1 /= gcd;
        }
    }

    pub(crate) fn get_i32(&self) -> i32 {
        assert!(self.1 != 0);
        self.0 / self.1
    }

    pub(crate) fn get_u32(&self) -> AvifResult<u32> {
        u32_from_i32(self.get_i32())
    }

    pub(crate) fn is_integer(&self) -> bool {
        self.0 % self.1 == 0
    }

    fn common_denominator(&mut self, val: &mut IFraction) -> AvifResult<()> {
        self.simplify();
        if self.1 == val.1 {
            return Ok(());
        }
        let self_d = self.1;
        self.0 = self
            .0
            .checked_mul(val.1)
            .ok_or(AvifError::UnknownError("".into()))?;
        self.1 = self
            .1
            .checked_mul(val.1)
            .ok_or(AvifError::UnknownError("".into()))?;
        val.0 = val
            .0
            .checked_mul(self_d)
            .ok_or(AvifError::UnknownError("".into()))?;
        val.1 = val
            .1
            .checked_mul(self_d)
            .ok_or(AvifError::UnknownError("".into()))?;
        Ok(())
    }

    pub(crate) fn add(&mut self, val: &IFraction) -> AvifResult<()> {
        let mut val = *val;
        val.simplify();
        self.common_denominator(&mut val)?;
        self.0 = self
            .0
            .checked_add(val.0)
            .ok_or(AvifError::UnknownError("".into()))?;
        self.simplify();
        Ok(())
    }

    pub(crate) fn sub(&mut self, val: &IFraction) -> AvifResult<()> {
        let mut val = *val;
        val.simplify();
        self.common_denominator(&mut val)?;
        self.0 = self
            .0
            .checked_sub(val.0)
            .ok_or(AvifError::UnknownError("".into()))?;
        self.simplify();
        Ok(())
    }
}

macro_rules! conversion_function {
    ($func:ident, $to: ident, $from:ty) => {
        pub(crate) fn $func(value: $from) -> AvifResult<$to> {
            $to::try_from(value).or(Err(AvifError::BmffParseFailed("".into())))
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
#[cfg(feature = "encoder")]
conversion_function!(u16_from_usize, u16, usize);
#[cfg(feature = "encoder")]
conversion_function!(u8_from_usize, u8, usize);
conversion_function!(u32_from_u64, u32, u64);
conversion_function!(u32_from_i32, u32, i32);
conversion_function!(i32_from_u32, i32, u32);
#[cfg(feature = "encoder")]
conversion_function!(u16_from_u32, u16, u32);
#[cfg(feature = "android_mediacodec")]
conversion_function!(isize_from_i32, isize, i32);
#[cfg(any(feature = "capi", feature = "android_mediacodec"))]
conversion_function!(isize_from_u32, isize, u32);
conversion_function!(isize_from_usize, isize, usize);
#[cfg(feature = "android_mediacodec")]
conversion_function!(i32_from_usize, i32, usize);

macro_rules! clamp_function {
    ($func:ident, $type:ty) => {
        pub(crate) fn $func(value: $type, low: $type, high: $type) -> $type {
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

macro_rules! find_property {
    ($properties:expr, $property_name:ident) => {
        $properties.iter().find_map(|p| match p {
            ItemProperty::$property_name(value) => Some(value.clone()),
            _ => None,
        })
    };
}

// Returns the colr nclx property. Returns an error if there are multiple ones.
pub(crate) fn find_nclx(properties: &[ItemProperty]) -> AvifResult<Option<&Nclx>> {
    let mut single_nclx: Option<&Nclx> = None;
    for property in properties {
        if let ItemProperty::ColorInformation(ColorInformation::Nclx(nclx)) = property {
            if single_nclx.is_some() {
                return Err(AvifError::BmffParseFailed(
                    "multiple nclx were found".into(),
                ));
            }
            single_nclx = Some(nclx);
        }
    }
    Ok(single_nclx)
}

// Returns the colr icc property. Returns an error if there are multiple ones.
pub(crate) fn find_icc(properties: &[ItemProperty]) -> AvifResult<Option<&Vec<u8>>> {
    let mut single_icc: Option<&Vec<u8>> = None;
    for property in properties {
        if let ItemProperty::ColorInformation(ColorInformation::Icc(icc)) = property {
            if single_icc.is_some() {
                return Err(AvifError::BmffParseFailed("multiple icc were found".into()));
            }
            single_icc = Some(icc);
        }
    }
    Ok(single_icc)
}

pub(crate) fn check_limits(
    width: u32,
    height: u32,
    size_limit: Option<NonZero<u32>>,
    dimension_limit: Option<NonZero<u32>>,
) -> bool {
    if height == 0 {
        return false;
    }
    if let Some(limit) = size_limit {
        if width > limit.get() / height {
            return false;
        }
    }
    if let Some(limit) = dimension_limit {
        if width > limit.get() || height > limit.get() {
            return false;
        }
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

pub(crate) fn limited_to_full_y(depth: u8, v: u16) -> u16 {
    match depth {
        8 => limited_to_full(16, 235, 255, v),
        10 => limited_to_full(64, 940, 1023, v),
        12 => limited_to_full(256, 3760, 4095, v),
        _ => 0,
    }
}

pub(crate) fn create_vec_exact<T>(size: usize) -> AvifResult<Vec<T>> {
    let mut v = Vec::<T>::new();
    let allocation_size = size
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(AvifError::OutOfMemory)?;
    // TODO: b/342251590 - Do not request allocations of more than what is allowed in Chromium's
    // partition allocator. This is the allowed limit in the chromium fuzzers. The value comes
    // from:
    // https://source.chromium.org/chromium/chromium/src/+/main:base/allocator/partition_allocator/src/partition_alloc/partition_alloc_constants.h;l=433-440;drc=c0265133106c7647e90f9aaa4377d28190b1a6a9.
    // Requesting an allocation larger than this value will cause the fuzzers to crash instead of
    // returning null. Remove this check once that behavior is fixed.
    if u64_from_usize(allocation_size)? >= 2_145_386_496 {
        return Err(AvifError::OutOfMemory);
    }
    if v.try_reserve_exact(size).is_err() {
        return Err(AvifError::OutOfMemory);
    }
    Ok(v)
}

#[cfg(test)]
pub(crate) fn assert_eq_f32_array(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        assert!((a[i] - b[i]).abs() <= std::f32::EPSILON);
    }
}

pub(crate) fn check_slice_range(len: usize, range: &Range<usize>) -> AvifResult<()> {
    if range.start >= len || range.end > len {
        return Err(AvifError::NoContent);
    }
    Ok(())
}

pub(crate) fn is_auxiliary_type_alpha(aux_type: &str) -> bool {
    aux_type == "urn:mpeg:mpegB:cicp:systems:auxiliary:alpha"
        || aux_type == "urn:mpeg:hevc:2015:auxid:1"
}

pub(crate) fn validate_grid_image_dimensions(image: &Image, grid: &Grid) -> AvifResult<()> {
    if checked_mul!(image.width, grid.columns)? < grid.width
        || checked_mul!(image.height, grid.rows)? < grid.height
    {
        return Err(AvifError::InvalidImageGrid(
                        "Grid image tiles do not completely cover the image (HEIF (ISO/IEC 23008-12:2017), Section 6.6.2.3.1)".into(),
                    ));
    }
    if checked_mul!(image.width, grid.columns)? < grid.width
        || checked_mul!(image.height, grid.rows)? < grid.height
    {
        return Err(AvifError::InvalidImageGrid(
            "Grid image tiles do not completely cover the image (HEIF (ISO/IEC 23008-12:2017), \
                    Section 6.6.2.3.1)"
                .into(),
        ));
    }
    if checked_mul!(image.width, grid.columns - 1)? >= grid.width
        || checked_mul!(image.height, grid.rows - 1)? >= grid.height
    {
        return Err(AvifError::InvalidImageGrid(
            "Grid image tiles in the rightmost column and bottommost row do not overlap the \
                     reconstructed image grid canvas. See MIAF (ISO/IEC 23000-22:2019), Section \
                     7.3.11.4.2, Figure 2"
                .into(),
        ));
    }
    // ISO/IEC 23000-22:2019, Section 7.3.11.4.2:
    //   - the tile_width shall be greater than or equal to 64, and should be a multiple of 64
    //   - the tile_height shall be greater than or equal to 64, and should be a multiple of 64
    // The "should" part is ignored here.
    if image.width < 64 || image.height < 64 {
        return Err(AvifError::InvalidImageGrid(format!(
            "Grid image tile width ({}) or height ({}) cannot be smaller than 64. See MIAF \
                     (ISO/IEC 23000-22:2019), Section 7.3.11.4.2",
            image.width, image.height
        )));
    }
    // ISO/IEC 23000-22:2019, Section 7.3.11.4.2:
    //   - when the images are in the 4:2:2 chroma sampling format the horizontal tile offsets
    //     and widths, and the output width, shall be even numbers;
    //   - when the images are in the 4:2:0 chroma sampling format both the horizontal and
    //     vertical tile offsets and widths, and the output width and height, shall be even
    //     numbers.
    if ((image.yuv_format == PixelFormat::Yuv420 || image.yuv_format == PixelFormat::Yuv422)
        && (grid.width % 2 != 0 || image.width % 2 != 0))
        || (image.yuv_format == PixelFormat::Yuv420
            && (grid.height % 2 != 0 || image.height % 2 != 0))
    {
        return Err(AvifError::InvalidImageGrid(format!(
            "Grid image width ({}) or height ({}) or tile width ({}) or height ({}) shall be \
                    even if chroma is subsampled in that dimension. See MIAF \
                    (ISO/IEC 23000-22:2019), Section 7.3.11.4.2",
            grid.width, grid.height, image.width, image.height
        )));
    }
    Ok(())
}
