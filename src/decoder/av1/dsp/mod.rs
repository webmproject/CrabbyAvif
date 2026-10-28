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

//! Pixel processing kernels (DSP).
//!
//! Every kernel here is a plain function over slices + strides, generic
//! over [`crate::decoder::av1::pixel::Pixel`]. Higher level code only calls kernels via
//! these entry points so that SIMD versions can later be dispatched from
//! here without touching the decoding logic.

pub mod cdef;
pub mod film_grain;
pub mod ipred;
pub mod itx;
pub mod loopfilter;
pub mod lr;

use crate::decoder::av1::pixel::Pixel;
use crate::decoder::av1::types::{TxSize, TxType};

/// Inverse transform + add, see [`itx::inv_txfm_add`].
pub type InvTxfmAddFn<T> = fn(&mut [T], usize, &mut [i32], TxSize, TxType, bool, usize, u32);
/// Intra prediction, see [`ipred::predict_intra`].
pub type PredictIntraFn<T> = fn(&mut [T], usize, &ipred::IntraEdges<T>, &ipred::IntraPredParams);
/// Chroma from luma AC values, see [`ipred::cfl_ac`].
pub type CflAcFn<T> = fn(&mut [i16], &[T], usize, usize, usize, usize, usize, usize, usize);
/// Chroma from luma prediction, see [`ipred::cfl_pred`].
pub type CflPredFn<T> = fn(&mut [T], usize, &[i16], usize, usize, i32, u32);
/// Deblocking of one whole edge made of `sizes.len()` 4-sample
/// segments: `fn(buf, stride, x, y, sizes, levels, lut, bd)`.
///
/// Vertical edges (`lf_v_edge`): the edge lies between columns `x - 1`
/// and `x`, segment `i` covers rows `y + 4 * i .. y + 4 * i + 4`.
/// Horizontal edges (`lf_h_edge`): the edge lies between rows `y - 1`
/// and `y`, segment `i` covers columns `x + 4 * i .. x + 4 * i + 4`.
/// `sizes[i]` is the kernel filter size of the segment (0 = not
/// filtered, else 4, 6, 8 or 16 as returned by
/// [`loopfilter::kernel_filter_size`]; all non-zero sizes of one call
/// are from one plane class, {4, 8, 16} or {4, 6}), `levels[i]` its
/// filter level (1..=63, index into `lut`, which holds
/// [`loopfilter::filter_limits`] of every level). `sizes.len() ==
/// levels.len()`.
pub type LoopFilterEdgeFn<T> =
    fn(&mut [T], usize, usize, usize, &[u8], &[u8], &[loopfilter::FilterLimits; 64], u32);
/// CDEF of one unit (direction search and filter of its blocks), see
/// [`cdef::cdef_unit`].
pub type CdefUnitFn<T> = fn(&mut cdef::CdefUnit<'_, T>);
/// Wiener filter, see [`lr::wiener_filter`].
pub type WienerFn<T> = fn(&mut [T], usize, &[T], usize, usize, usize, &[[i32; 3]; 2], u32);
/// Self guided filter, see [`lr::sgr_filter`].
pub type SgrFn<T> = fn(&mut [T], usize, &[T], usize, usize, usize, usize, [i32; 2], u32);

/// Table of the kernels used by the decoder for one pixel type.
///
/// The decoder only calls kernels through this table. [`Dsp::new`] fills
/// it with the portable implementations; SIMD versions can be selected
/// here at runtime (e.g. from CPU feature detection) without any change
/// to the decoding logic.
#[derive(Clone, Copy)]
pub struct Dsp<T: Pixel> {
    pub inv_txfm_add: InvTxfmAddFn<T>,
    pub predict_intra: PredictIntraFn<T>,
    pub cfl_ac: CflAcFn<T>,
    pub cfl_pred: CflPredFn<T>,
    pub lf_v_edge: LoopFilterEdgeFn<T>,
    pub lf_h_edge: LoopFilterEdgeFn<T>,
    pub cdef_unit: CdefUnitFn<T>,
    pub wiener: WienerFn<T>,
    pub sgr: SgrFn<T>,
}

impl<T: Pixel> Dsp<T> {
    /// The kernels of this build for bit depth `bd` (8, 10 or 12).
    pub fn new(bd: u32) -> Self {
        let _ = bd;
        Self::portable()
    }

    /// Portable (scalar, auto-vectorized) kernels.
    pub fn portable() -> Self {
        Self {
            inv_txfm_add: itx::inv_txfm_add::<T>,
            predict_intra: ipred::predict_intra::<T>,
            cfl_ac: ipred::cfl_ac::<T>,
            cfl_pred: ipred::cfl_pred::<T>,
            lf_v_edge: loopfilter::filter_v_edge::<T>,
            lf_h_edge: loopfilter::filter_h_edge::<T>,
            cdef_unit: cdef::cdef_unit::<T, cdef::PortableCdef>,
            wiener: lr::wiener_filter::<T>,
            sgr: lr::sgr_filter::<T>,
        }
    }
}
