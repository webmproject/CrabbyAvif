// Copyright 2026 Google LLC
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

//! Pixel storage abstraction.
//!
//! All pixel processing code is generic over [`Pixel`], which is implemented
//! for `u8` (8-bit streams) and `u16` (10/12-bit streams). The actual bit
//! depth is always passed separately at runtime as `bd: u32`.

/// A pixel sample type.
pub trait Pixel:
    Copy + Clone + Default + PartialEq + Eq + Send + Sync + std::fmt::Debug + 'static
{
    /// Largest bit depth representable by this type.
    const MAX_BITDEPTH: u32;

    fn to_i32(self) -> i32;

    /// Converts from `i32`. `v` must already be in the valid pixel range
    /// (`0..(1 << bd)`); out of range values are truncated.
    fn from_i32(v: i32) -> Self;

    #[inline(always)]
    fn to_u16(self) -> u16 {
        self.to_i32() as u16
    }
}

impl Pixel for u8 {
    const MAX_BITDEPTH: u32 = 8;
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v as u8
    }
}

impl Pixel for u16 {
    const MAX_BITDEPTH: u32 = 12;
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v as u16
    }
}

/// Clamps `v` to `[0, (1 << bd) - 1]` and converts it to a pixel.
#[inline(always)]
pub fn clip_pixel<T: Pixel>(v: i32, bd: u32) -> T {
    T::from_i32(v.clamp(0, (1 << bd) - 1))
}

/// Spec `Clip3(lo, hi, v)`.
#[inline(always)]
pub fn clip3(lo: i32, hi: i32, v: i32) -> i32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

/// Spec `Round2(x, n)` for signed values (arithmetic shift).
#[inline(always)]
pub fn round2(x: i32, n: u32) -> i32 {
    if n == 0 {
        x
    } else {
        (x + (1 << (n - 1))) >> n
    }
}

/// Spec `Round2Signed(x, n)`.
#[inline(always)]
pub fn round2_signed(x: i32, n: u32) -> i32 {
    if x >= 0 {
        round2(x, n)
    } else {
        -round2(-x, n)
    }
}

/// Spec `FloorLog2(x)` for x > 0.
#[inline(always)]
pub fn floor_log2(x: u32) -> u32 {
    31 - x.leading_zeros()
}

/// Spec `CeilLog2(x)`.
#[inline(always)]
pub fn ceil_log2(x: u32) -> u32 {
    if x < 2 {
        0
    } else {
        32 - (x - 1).leading_zeros()
    }
}
