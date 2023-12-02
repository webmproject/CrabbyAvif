pub mod io;
pub mod stream;

use crate::AvifError;
use crate::AvifResult;

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
