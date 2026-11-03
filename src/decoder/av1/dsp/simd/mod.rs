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

//! SIMD kernels written with `std::simd` (portable SIMD, nightly only).
//!
//! Every kernel has the same contract and produces bit-identical output to
//! the portable kernel it replaces (see the parent module);
//! `dsp::Dsp::new(bd)` always selects these kernels. The portable kernels
//! remain as the reference implementation for the tests and as fallbacks;
//! kernels that have not been converted are re-exported from them.
//!
//! Each submodule must contain tests that compare its kernels with the
//! portable ones on random inputs for every supported block size and bit
//! depth (`cargo test`).

pub mod cdef;
pub mod film_grain;
pub mod ipred;
pub mod itx;
pub mod loopfilter;
pub mod lr;

use std::simd::prelude::*;

use crate::decoder::av1::pixel::Pixel;

/// Loads `N` pixels from `src[..N]` widened to `i32` lanes.
#[inline(always)]
pub fn load_i32<T: Pixel, const N: usize>(src: &[T]) -> Simd<i32, N> {
    Simd::from_array(std::array::from_fn(|i| src[i].to_i32()))
}

/// Clamps `v` to `[0, (1 << bd) - 1]` and stores it to `dst[..N]`.
#[inline(always)]
pub fn store_clip<T: Pixel, const N: usize>(dst: &mut [T], v: Simd<i32, N>, bd: u32) {
    let v = v.simd_clamp(Simd::splat(0), Simd::splat((1 << bd) - 1));
    for (d, &x) in dst[..N].iter_mut().zip(v.as_array()) {
        *d = T::from_i32(x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_store_round_trip() {
        let src: Vec<u16> = (0..8).map(|i| i * 600).collect();
        let v = load_i32::<u16, 8>(&src);
        let mut dst = [0u16; 8];
        store_clip(&mut dst, v, 12);
        assert_eq!(&dst[..7], &src[..7]);
        assert_eq!(dst[7], 4095);
    }
}
