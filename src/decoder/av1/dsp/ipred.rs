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

//! Intra prediction (spec 7.11.2.2 - 7.11.2.12).
//!
//! OWNER: ipred worker. Contract below must be kept.
//!
//! The caller performs the edge gathering of spec 7.11.2.1 (AboveRow /
//! LeftCol derivation from the frame buffer with availability rules) and
//! hands the prepared, *unfiltered* edges to [`predict_intra`], which does
//! everything else: mode dispatch, intra edge filtering / upsampling for
//! directional modes, recursive filter intra, smooth, DC, paeth.
//! Palette and CFL are handled by the caller.
//!
//! # Structure
//!
//! [`predict_intra`] is a thin dispatcher (end of spec 7.11.2.1). Every
//! predictor is a separate kernel function operating on plain slices and a
//! stride (`ipred_*`), so that SIMD implementations can later replace
//! individual kernels:
//!
//! * [`ipred_dc`], [`ipred_dc_top`], [`ipred_dc_left`], [`ipred_dc_128`]
//!   (spec 7.11.2.5),
//! * [`ipred_v`], [`ipred_h`] (pAngle 90 / 180 of spec 7.11.2.4),
//! * [`ipred_paeth`] (spec 7.11.2.2),
//! * [`ipred_smooth`], [`ipred_smooth_v`], [`ipred_smooth_h`]
//!   (spec 7.11.2.6),
//! * [`ipred_z1`] (pAngle < 90), [`ipred_z2`] (90 < pAngle < 180),
//!   [`ipred_z3`] (pAngle > 180) (spec 7.11.2.4 steps 7 - 9),
//! * [`ipred_filter`] (recursive filter intra, spec 7.11.2.3).
//!
//! The edge preparation of directional prediction (spec 7.11.2.4 step 4)
//! uses [`filter_corner`], [`edge_filter_strength`], [`use_upsample`],
//! [`filter_edge`] and [`upsample_edge`].
//!
//! Nothing here allocates on the heap; all scratch buffers are fixed size
//! stack arrays.

use crate::decoder::av1::pixel::{clip_pixel, round2, round2_signed, Pixel};
use crate::decoder::av1::types::IntraMode;

/// Maximum number of edge samples needed (`w + h` with w, h <= 64).
pub const MAX_EDGE: usize = 128;

/// Prepared edge samples (spec 7.11.2.1 output).
///
/// Only the samples the selected predictor reads need to be valid, see
/// [`edge_needs`]; the other entries are unspecified.
#[derive(Clone)]
pub struct IntraEdges<T: Pixel> {
    /// `AboveRow[-1]` (== `LeftCol[-1]`).
    pub top_left: T,
    /// `AboveRow[i]` for `i = 0..edge_needs().0` (entries beyond are
    /// unspecified).
    pub above: [T; MAX_EDGE],
    /// `LeftCol[i]` for `i = 0..edge_needs().1` (entries beyond are
    /// unspecified).
    pub left: [T; MAX_EDGE],
}

impl<T: Pixel> Default for IntraEdges<T> {
    fn default() -> Self {
        Self {
            top_left: T::default(),
            above: [T::default(); MAX_EDGE],
            left: [T::default(); MAX_EDGE],
        }
    }
}

/// Parameters of one intra prediction call (one transform block).
#[derive(Clone, Copy, Debug, Default)]
pub struct IntraPredParams {
    /// Intra mode, spec value 0 (DC_PRED) ..= 12 (PAETH_PRED). For chroma CFL
    /// blocks the caller passes DC_PRED and applies CFL afterwards.
    pub mode: u8,
    /// AngleDeltaY / AngleDeltaUV in -3..=3 (only used by directional modes).
    pub angle_delta: i32,
    /// `Some(filter_intra_mode)` (0..=4) if use_filter_intra applies to this
    /// call (luma only), `None` otherwise.
    pub filter_intra_mode: Option<u8>,
    pub have_left: bool,
    pub have_above: bool,
    /// Sequence header `enable_intra_edge_filter`.
    pub enable_intra_edge_filter: bool,
    /// `filterType` from spec 7.11.2.8 (computed by the caller): true if an
    /// available above/left neighbouring block uses a smooth mode.
    pub smooth_filter_type: bool,
    /// Sample position of the transform block in the plane (spec x, y).
    pub x: i32,
    pub y: i32,
    /// Spec maxX / maxY for this plane.
    pub max_x: i32,
    pub max_y: i32,
    /// Transform block width / height in samples (4..=64, powers of two).
    pub w: usize,
    pub h: usize,
    /// Bit depth.
    pub bd: u32,
}

// ---------------------------------------------------------------------------
// Constants and tables (spec section 3 and "Additional tables").
// ---------------------------------------------------------------------------

/// Spec `ANGLE_STEP`: degrees per unit of angle delta.
pub const ANGLE_STEP: i32 = 3;
/// Spec `INTRA_FILTER_SCALE_BITS`.
pub const INTRA_FILTER_SCALE_BITS: u32 = 4;
/// Spec `INTRA_EDGE_TAPS`.
pub const INTRA_EDGE_TAPS: usize = 5;
/// Spec `INTRA_EDGE_KERNELS`.
pub const INTRA_EDGE_KERNELS: usize = 3;
/// Spec `INTRA_FILTER_MODES`.
pub const INTRA_FILTER_MODES: usize = 5;

/// Spec `Mode_To_Angle[INTRA_MODES]`.
pub const MODE_TO_ANGLE: [i32; 13] = [0, 90, 180, 45, 135, 113, 157, 203, 67, 0, 0, 0, 0];

/// Spec `Dr_Intra_Derivative[90]`.
#[rustfmt::skip]
pub const DR_INTRA_DERIVATIVE: [i32; 90] = [
    0, 0, 0, 1023, 0, 0, 547, 0, 0, 372, 0, 0, 0, 0,
    273, 0, 0, 215, 0, 0, 178, 0, 0, 151, 0, 0, 132, 0, 0,
    116, 0, 0, 102, 0, 0, 0, 90, 0, 0, 80, 0, 0, 71, 0, 0,
    64, 0, 0, 57, 0, 0, 51, 0, 0, 45, 0, 0, 0, 40, 0, 0,
    35, 0, 0, 31, 0, 0, 27, 0, 0, 23, 0, 0, 19, 0, 0,
    15, 0, 0, 0, 0, 11, 0, 0, 7, 0, 0, 3, 0, 0,
];

/// Spec `Sm_Weights_Tx_4x4`.
pub const SM_WEIGHTS_TX_4X4: [u8; 4] = [255, 149, 85, 64];
/// Spec `Sm_Weights_Tx_8x8`.
pub const SM_WEIGHTS_TX_8X8: [u8; 8] = [255, 197, 146, 105, 73, 50, 37, 32];
/// Spec `Sm_Weights_Tx_16x16`.
#[rustfmt::skip]
pub const SM_WEIGHTS_TX_16X16: [u8; 16] = [
    255, 225, 196, 170, 145, 123, 102, 84, 68, 54, 43, 33, 26, 20, 17, 16,
];
/// Spec `Sm_Weights_Tx_32x32`.
#[rustfmt::skip]
pub const SM_WEIGHTS_TX_32X32: [u8; 32] = [
    255, 240, 225, 210, 196, 182, 169, 157, 145, 133, 122, 111, 101, 92, 83, 74,
    66, 59, 52, 45, 39, 34, 29, 25, 21, 17, 14, 12, 10, 9, 8, 8,
];
/// Spec `Sm_Weights_Tx_64x64`.
#[rustfmt::skip]
pub const SM_WEIGHTS_TX_64X64: [u8; 64] = [
    255, 248, 240, 233, 225, 218, 210, 203, 196, 189, 182, 176, 169, 163, 156, 150,
    144, 138, 133, 127, 121, 116, 111, 106, 101, 96, 91, 86, 82, 77, 73, 69,
    65, 61, 57, 54, 50, 47, 44, 41, 38, 35, 32, 29, 27, 25, 22, 20,
    18, 16, 15, 13, 12, 10, 9, 8, 7, 6, 6, 5, 5, 4, 4, 4,
];

/// Returns the smooth weights table for a dimension of `n` samples
/// (`n` in 4, 8, 16, 32, 64), i.e. `Sm_Weights_Tx_NxN`.
#[inline]
pub fn sm_weights(n: usize) -> &'static [u8] {
    match n {
        4 => &SM_WEIGHTS_TX_4X4,
        8 => &SM_WEIGHTS_TX_8X8,
        16 => &SM_WEIGHTS_TX_16X16,
        32 => &SM_WEIGHTS_TX_32X32,
        64 => &SM_WEIGHTS_TX_64X64,
        _ => panic!("invalid smooth prediction size {n}"),
    }
}

/// Spec `Intra_Filter_Taps[INTRA_FILTER_MODES][8][7]`.
#[rustfmt::skip]
pub const INTRA_FILTER_TAPS: [[[i8; 7]; 8]; INTRA_FILTER_MODES] = [
    [
        [-6, 10, 0, 0, 0, 12, 0],
        [-5, 2, 10, 0, 0, 9, 0],
        [-3, 1, 1, 10, 0, 7, 0],
        [-3, 1, 1, 2, 10, 5, 0],
        [-4, 6, 0, 0, 0, 2, 12],
        [-3, 2, 6, 0, 0, 2, 9],
        [-3, 2, 2, 6, 0, 2, 7],
        [-3, 1, 2, 2, 6, 3, 5],
    ],
    [
        [-10, 16, 0, 0, 0, 10, 0],
        [-6, 0, 16, 0, 0, 6, 0],
        [-4, 0, 0, 16, 0, 4, 0],
        [-2, 0, 0, 0, 16, 2, 0],
        [-10, 16, 0, 0, 0, 0, 10],
        [-6, 0, 16, 0, 0, 0, 6],
        [-4, 0, 0, 16, 0, 0, 4],
        [-2, 0, 0, 0, 16, 0, 2],
    ],
    [
        [-8, 8, 0, 0, 0, 16, 0],
        [-8, 0, 8, 0, 0, 16, 0],
        [-8, 0, 0, 8, 0, 16, 0],
        [-8, 0, 0, 0, 8, 16, 0],
        [-4, 4, 0, 0, 0, 0, 16],
        [-4, 0, 4, 0, 0, 0, 16],
        [-4, 0, 0, 4, 0, 0, 16],
        [-4, 0, 0, 0, 4, 0, 16],
    ],
    [
        [-2, 8, 0, 0, 0, 10, 0],
        [-1, 3, 8, 0, 0, 6, 0],
        [-1, 2, 3, 8, 0, 4, 0],
        [0, 1, 2, 3, 8, 2, 0],
        [-1, 4, 0, 0, 0, 3, 10],
        [-1, 3, 4, 0, 0, 4, 6],
        [-1, 2, 3, 4, 0, 4, 4],
        [-1, 2, 2, 3, 4, 3, 3],
    ],
    [
        [-12, 14, 0, 0, 0, 14, 0],
        [-10, 0, 14, 0, 0, 12, 0],
        [-9, 0, 0, 14, 0, 11, 0],
        [-8, 0, 0, 0, 14, 10, 0],
        [-10, 12, 0, 0, 0, 0, 14],
        [-9, 1, 12, 0, 0, 0, 12],
        [-8, 0, 0, 12, 0, 1, 11],
        [-7, 0, 0, 1, 12, 1, 9],
    ],
];

/// Spec `Intra_Edge_Kernel[INTRA_EDGE_KERNELS][INTRA_EDGE_TAPS]`.
pub const INTRA_EDGE_KERNEL: [[i32; INTRA_EDGE_TAPS]; INTRA_EDGE_KERNELS] =
    [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];

pub(crate) const DC_PRED: u8 = IntraMode::DC_PRED as u8;
pub(crate) const V_PRED: u8 = IntraMode::V_PRED as u8;
pub(crate) const D67_PRED: u8 = IntraMode::D67_PRED as u8;
pub(crate) const SMOOTH_PRED: u8 = IntraMode::SMOOTH_PRED as u8;
pub(crate) const SMOOTH_V_PRED: u8 = IntraMode::SMOOTH_V_PRED as u8;
pub(crate) const SMOOTH_H_PRED: u8 = IntraMode::SMOOTH_H_PRED as u8;
pub(crate) const PAETH_PRED: u8 = IntraMode::PAETH_PRED as u8;

// ---------------------------------------------------------------------------
// Directional edge buffers.
// ---------------------------------------------------------------------------

/// Number of padding samples in front of spec index 0 in a directional
/// edge buffer (see [`EDGE_BUF_LEN`]). Spec index `k` of AboveRow / LeftCol
/// is stored at buffer index `EDGE_PAD + k`, so indices down to `-EDGE_PAD`
/// are addressable (the spec uses down to -2).
pub const EDGE_PAD: usize = 16;

/// Length of a directional edge buffer: `EDGE_PAD` samples before spec
/// index 0, `MAX_EDGE` samples for indices `0..w+h` and some slack.
pub const EDGE_BUF_LEN: usize = EDGE_PAD + MAX_EDGE + 16;

/// Maximum `numPx` of the intra edge upsample process. Upsampling is only
/// selected when `w + h <= 16` (spec 7.11.2.10).
pub const MAX_UPSAMPLE_PX: usize = 16;

/// Maximum size of the intra edge filter process (spec 7.11.2.12).
pub const MAX_FILTER_EDGE: usize = MAX_EDGE + 1;

// ---------------------------------------------------------------------------
// Dispatcher.
// ---------------------------------------------------------------------------

/// Table of the individual predictor kernels used by [`predict_intra_with`].
///
/// [`IpredKernels::SCALAR`] holds the portable kernels of this module. The
/// decoder uses a table where some entries are replaced by
/// `dsp::simd::ipred` kernels; every kernel has the same contract as the
/// scalar function it replaces.
pub struct IpredKernels<T: Pixel> {
    pub dc: fn(&mut [T], usize, &[T], &[T], usize, usize),
    pub dc_top: fn(&mut [T], usize, &[T], usize, usize),
    pub dc_left: fn(&mut [T], usize, &[T], usize, usize),
    pub dc_128: fn(&mut [T], usize, usize, usize, u32),
    pub v: fn(&mut [T], usize, &[T], usize, usize),
    pub h: fn(&mut [T], usize, &[T], usize, usize),
    pub paeth: fn(&mut [T], usize, T, &[T], &[T], usize, usize),
    pub smooth: fn(&mut [T], usize, &[T], &[T], usize, usize),
    pub smooth_v: fn(&mut [T], usize, &[T], &[T], usize, usize),
    pub smooth_h: fn(&mut [T], usize, &[T], &[T], usize, usize),
    pub z1: fn(&mut [T], usize, &[T], usize, usize, i32, bool),
    pub z2: fn(&mut [T], usize, &[T], &[T], usize, usize, i32, i32, bool, bool),
    pub z3: fn(&mut [T], usize, &[T], usize, usize, i32, bool),
    pub filter: fn(&mut [T], usize, T, &[T], &[T], usize, usize, usize, u32),
}

impl<T: Pixel> IpredKernels<T> {
    /// The portable kernels.
    pub const SCALAR: Self = Self {
        dc: ipred_dc::<T>,
        dc_top: ipred_dc_top::<T>,
        dc_left: ipred_dc_left::<T>,
        dc_128: ipred_dc_128::<T>,
        v: ipred_v::<T>,
        h: ipred_h::<T>,
        paeth: ipred_paeth::<T>,
        smooth: ipred_smooth::<T>,
        smooth_v: ipred_smooth_v::<T>,
        smooth_h: ipred_smooth_h::<T>,
        z1: ipred_z1::<T>,
        z2: ipred_z2::<T>,
        z3: ipred_z3::<T>,
        filter: ipred_filter::<T>,
    };
}

/// Writes the `w x h` prediction into `dst` (`dst[0]` = top-left sample of
/// the block, rows `stride` apart). Never reads from `dst`.
pub fn predict_intra<T: Pixel>(
    dst: &mut [T],
    stride: usize,
    edges: &IntraEdges<T>,
    p: &IntraPredParams,
) {
    predict_intra_with(&IpredKernels::SCALAR, dst, stride, edges, p);
}

/// Number of `AboveRow` / `LeftCol` samples (from index 0) that the
/// prediction selected by `p` reads (portable and SIMD kernels, including
/// edge filtering and upsampling); `top_left` is always read. The spec
/// derives `w + h` samples of each edge, most predictors use fewer:
///
/// * DC: `w` above if `have_above`, `h` left if `have_left`,
/// * pAngle 90 / 180: `w` above / `h` left,
/// * pAngle < 90 (z1): `w + h` above (up to `maxBaseX`), no left,
/// * 90 < pAngle < 180 (z2): `w` above, `h` left (the positions are below
///   `w - 1` / `h - 1` before the 2-tap interpolation, see
///   `z2_left_index_bounds`; the edge filter covers at most `w` / `h`),
/// * pAngle > 180 (z3): `w + h` left, no above,
/// * smooth, paeth, filter intra: `w` above and `h` left.
pub fn edge_needs(p: &IntraPredParams) -> (usize, usize) {
    let (w, h) = (p.w, p.h);
    if p.filter_intra_mode.is_some() {
        return (w, h);
    }
    match p.mode {
        DC_PRED => (
            if p.have_above { w } else { 0 },
            if p.have_left { h } else { 0 },
        ),
        V_PRED..=D67_PRED => {
            let p_angle = MODE_TO_ANGLE[p.mode as usize] + p.angle_delta * ANGLE_STEP;
            match p_angle {
                90 => (w, 0),
                180 => (0, h),
                a if a < 90 => (w + h, 0),
                a if a < 180 => (w, h),
                _ => (0, w + h),
            }
        }
        _ => (w, h),
    }
}

/// [`predict_intra`] using the predictor kernels of `k`.
pub fn predict_intra_with<T: Pixel>(
    k: &IpredKernels<T>,
    dst: &mut [T],
    stride: usize,
    edges: &IntraEdges<T>,
    p: &IntraPredParams,
) {
    let (w, h) = (p.w, p.h);
    debug_assert!(w.is_power_of_two() && (4..=64).contains(&w));
    debug_assert!(h.is_power_of_two() && (4..=64).contains(&h));
    debug_assert!(stride >= w && dst.len() >= (h - 1) * stride + w);
    let above = &edges.above[..];
    let left = &edges.left[..];

    if let Some(fm) = p.filter_intra_mode {
        (k.filter)(
            dst,
            stride,
            edges.top_left,
            above,
            left,
            w,
            h,
            fm as usize,
            p.bd,
        );
        return;
    }
    match p.mode {
        V_PRED..=D67_PRED => predict_directional(k, dst, stride, edges, p),
        SMOOTH_PRED => (k.smooth)(dst, stride, above, left, w, h),
        SMOOTH_V_PRED => (k.smooth_v)(dst, stride, above, left, w, h),
        SMOOTH_H_PRED => (k.smooth_h)(dst, stride, above, left, w, h),
        DC_PRED => match (p.have_left, p.have_above) {
            (true, true) => (k.dc)(dst, stride, above, left, w, h),
            (true, false) => (k.dc_left)(dst, stride, left, w, h),
            (false, true) => (k.dc_top)(dst, stride, above, w, h),
            (false, false) => (k.dc_128)(dst, stride, w, h, p.bd),
        },
        PAETH_PRED => (k.paeth)(dst, stride, edges.top_left, above, left, w, h),
        m => panic!("invalid intra mode {m}"),
    }
}

/// Directional intra prediction process (spec 7.11.2.4): edge preparation
/// (corner filter, edge filter, upsampling) followed by the z1 / z2 / z3
/// kernels, or plain vertical / horizontal copies for pAngle 90 / 180.
fn predict_directional<T: Pixel>(
    k: &IpredKernels<T>,
    dst: &mut [T],
    stride: usize,
    edges: &IntraEdges<T>,
    p: &IntraPredParams,
) {
    let (w, h) = (p.w, p.h);
    let p_angle = MODE_TO_ANGLE[p.mode as usize] + p.angle_delta * ANGLE_STEP;
    // For pAngle 90 / 180 the edges are never modified: the edge filter is
    // skipped and neither upsample selection can be true (|delta| is 0 or
    // 90 for both edges).
    if p_angle == 90 {
        (k.v)(dst, stride, &edges.above, w, h);
        return;
    }
    if p_angle == 180 {
        (k.h)(dst, stride, &edges.left, w, h);
        return;
    }

    let n = w + h;
    let filter = p.enable_intra_edge_filter;
    let filter_type = p.smooth_filter_type;

    if p_angle < 90 {
        let dx = DR_INTRA_DERIVATIVE[p_angle as usize];
        let delta = p_angle - 90;
        let upsample_above = filter && use_upsample(w, h, filter_type, delta);
        let strength = if !upsample_above && filter && p.have_above {
            edge_filter_strength(w, h, filter_type, delta)
        } else {
            0
        };
        if !upsample_above && strength == 0 {
            (k.z1)(dst, stride, &edges.above, w, h, dx, false);
            return;
        }
        let mut above = [T::default(); EDGE_BUF_LEN];
        above[EDGE_PAD - 1] = edges.top_left;
        above[EDGE_PAD..EDGE_PAD + n].copy_from_slice(&edges.above[..n]);
        if strength != 0 {
            let num_px = (w as i32).min(p.max_x - p.x + 1) + h as i32 + 1;
            filter_edge(&mut above[EDGE_PAD - 1..], num_px, strength);
        }
        if upsample_above {
            upsample_edge(&mut above, n, p.bd);
        }
        (k.z1)(dst, stride, &above[EDGE_PAD..], w, h, dx, upsample_above);
        return;
    }

    if p_angle > 180 {
        let dy = DR_INTRA_DERIVATIVE[(270 - p_angle) as usize];
        let delta = p_angle - 180;
        let upsample_left = filter && use_upsample(w, h, filter_type, delta);
        let strength = if !upsample_left && filter && p.have_left {
            edge_filter_strength(w, h, filter_type, delta)
        } else {
            0
        };
        if !upsample_left && strength == 0 {
            (k.z3)(dst, stride, &edges.left, w, h, dy, false);
            return;
        }
        let mut left = [T::default(); EDGE_BUF_LEN];
        left[EDGE_PAD - 1] = edges.top_left;
        left[EDGE_PAD..EDGE_PAD + n].copy_from_slice(&edges.left[..n]);
        if strength != 0 {
            let num_px = (h as i32).min(p.max_y - p.y + 1) + w as i32 + 1;
            filter_edge(&mut left[EDGE_PAD - 1..], num_px, strength);
        }
        if upsample_left {
            upsample_edge(&mut left, n, p.bd);
        }
        (k.z3)(dst, stride, &left[EDGE_PAD..], w, h, dy, upsample_left);
        return;
    }

    // 90 < pAngle < 180 (z2): reads at most w above samples and h left
    // samples (plus indices -1 / -2).
    let mut above = [T::default(); EDGE_BUF_LEN];
    let mut left = [T::default(); EDGE_BUF_LEN];
    let corner = if filter && n >= 24 {
        T::from_i32(filter_corner(edges.left[0], edges.top_left, edges.above[0]))
    } else {
        edges.top_left
    };
    above[EDGE_PAD - 1] = corner;
    above[EDGE_PAD..EDGE_PAD + w].copy_from_slice(&edges.above[..w]);
    left[EDGE_PAD - 1] = corner;
    left[EDGE_PAD..EDGE_PAD + h].copy_from_slice(&edges.left[..h]);

    let mut upsample_above = false;
    let mut upsample_left = false;
    if filter {
        if p.have_above {
            let strength = edge_filter_strength(w, h, filter_type, p_angle - 90);
            let num_px = (w as i32).min(p.max_x - p.x + 1) + 1;
            filter_edge(&mut above[EDGE_PAD - 1..], num_px, strength);
        }
        if p.have_left {
            let strength = edge_filter_strength(w, h, filter_type, p_angle - 180);
            let num_px = (h as i32).min(p.max_y - p.y + 1) + 1;
            filter_edge(&mut left[EDGE_PAD - 1..], num_px, strength);
        }
        upsample_above = use_upsample(w, h, filter_type, p_angle - 90);
        if upsample_above {
            upsample_edge(&mut above, w, p.bd);
        }
        upsample_left = use_upsample(w, h, filter_type, p_angle - 180);
        if upsample_left {
            upsample_edge(&mut left, h, p.bd);
        }
    }

    let dx = DR_INTRA_DERIVATIVE[(180 - p_angle) as usize];
    let dy = DR_INTRA_DERIVATIVE[(p_angle - 90) as usize];
    (k.z2)(
        dst,
        stride,
        &above,
        &left,
        w,
        h,
        dx,
        dy,
        upsample_above,
        upsample_left,
    );
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

/// Dispatches `$f::<T, W>(args)` on the block width `$w` (4..=64).
macro_rules! by_width {
    ($w:expr, $f:ident::<$($t:ty),+>($($a:expr),* $(,)?)) => {
        match $w {
            4 => $f::<$($t),+, 4>($($a),*),
            8 => $f::<$($t),+, 8>($($a),*),
            16 => $f::<$($t),+, 16>($($a),*),
            32 => $f::<$($t),+, 32>($($a),*),
            64 => $f::<$($t),+, 64>($($a),*),
            w => panic!("invalid block width {w}"),
        }
    };
}

/// Bounds-checks `dst` for a `W x h` block (`h` in 1..=64) with row
/// stride `stride`, returning the bounded slice and masked `stride` so
/// row offsets `i * stride` for `i < h` cannot overflow 64-bit `usize`.
#[inline(always)]
fn check_block<T, const W: usize>(dst: &mut [T], stride: usize, h: usize) -> (&mut [T], usize) {
    let stride = stride & 0xffff_ffff;
    assert!(stride >= W && (1..=64).contains(&h));
    (&mut dst[..(h - 1) * stride + W], stride)
}

/// Bounds-checks `dst` for a `W x h` block where `h` is a multiple of 4
/// in `4..=64`.
#[inline(always)]
fn check_block4<T, const W: usize>(
    dst: &mut [T],
    stride: usize,
    h: usize,
) -> (&mut [T], usize, usize) {
    let stride = stride & 0xffff_ffff;
    let h4 = h >> 2;
    assert!(stride >= W && (1..=16).contains(&h4) && h == h4 * 4);
    (&mut dst[..(4 * h4 - 1) * stride + W], stride, h4)
}

/// Returns the `W`-sample row starting at `off` in `dst`.
#[inline(always)]
fn row_mut<T, const W: usize>(dst: &mut [T], off: usize) -> &mut [T; W] {
    (&mut dst[off..off + W]).try_into().unwrap()
}

/// Fills a `W x h` block with a single value (4 rows per iteration).
#[inline(always)]
fn fill_w<T: Pixel, const W: usize>(dst: &mut [T], stride: usize, h: usize, v: T) {
    let (dst, stride, h4) = check_block4::<T, W>(dst, stride, h);
    let src = [v; W];
    for i4 in 0..h4 {
        let base = 4 * i4 * stride;
        *row_mut::<T, W>(dst, base) = src;
        *row_mut::<T, W>(dst, base + stride) = src;
        *row_mut::<T, W>(dst, base + 2 * stride) = src;
        *row_mut::<T, W>(dst, base + 3 * stride) = src;
    }
}

/// Fills a `w x h` block with a single value.
#[inline(always)]
fn fill<T: Pixel>(dst: &mut [T], stride: usize, w: usize, h: usize, v: T) {
    by_width!(w, fill_w::<T>(dst, stride, h, v));
}

/// Sum of `e[..n]` for `n` in `4, 8, 16, 32, 64`.
#[inline(always)]
fn edge_sum_n<T: Pixel>(e: &[T], n: usize) -> u32 {
    #[inline(always)]
    fn sum_c<T: Pixel, const N: usize>(e: &[T]) -> u32 {
        let s: &[T; N] = (&e[..N]).try_into().unwrap();
        s.iter().map(|&v| v.to_i32() as u32).sum()
    }
    match n {
        4 => sum_c::<T, 4>(e),
        8 => sum_c::<T, 8>(e),
        16 => sum_c::<T, 16>(e),
        32 => sum_c::<T, 32>(e),
        64 => sum_c::<T, 64>(e),
        _ => e[..n].iter().map(|&v| v.to_i32() as u32).sum(),
    }
}

/// `Round2(a * (32 - shift) + b * shift, 5)`: the 2-tap interpolation of
/// the directional predictor.
#[inline(always)]
fn interp<T: Pixel>(a: T, b: T, shift: i32) -> T {
    let a = a.to_i32();
    T::from_i32(a + (((b.to_i32() - a) * shift + 16) >> 5))
}

// ---------------------------------------------------------------------------
// DC / V / H kernels.
// ---------------------------------------------------------------------------

/// DC prediction with both edges available (spec 7.11.2.5, first case).
pub fn ipred_dc<T: Pixel>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    w: usize,
    h: usize,
) {
    let sum = edge_sum_n(above, w) + edge_sum_n(left, h);
    let n = (w + h) as u32;
    let k = n.trailing_zeros();
    let x = (sum + (n >> 1)) >> k;
    let avg = match n >> k {
        1 => x,
        3 => ((x as u64 * 0xAAAA_AAAB) >> 33) as u32,
        _ => ((x as u64 * 0xCCCC_CCCD) >> 34) as u32,
    };
    fill(dst, stride, w, h, T::from_i32(avg as i32));
}

/// DC prediction from the above edge only (haveLeft = 0, haveAbove = 1).
pub fn ipred_dc_top<T: Pixel>(dst: &mut [T], stride: usize, above: &[T], w: usize, h: usize) {
    let sum = edge_sum_n(above, w);
    let avg = (sum + (w as u32 >> 1)) >> w.trailing_zeros();
    fill(dst, stride, w, h, T::from_i32(avg as i32));
}

/// DC prediction from the left edge only (haveLeft = 1, haveAbove = 0).
pub fn ipred_dc_left<T: Pixel>(dst: &mut [T], stride: usize, left: &[T], w: usize, h: usize) {
    let sum = edge_sum_n(left, h);
    let avg = (sum + (h as u32 >> 1)) >> h.trailing_zeros();
    fill(dst, stride, w, h, T::from_i32(avg as i32));
}

/// DC prediction without any edge: `1 << (bd - 1)`.
pub fn ipred_dc_128<T: Pixel>(dst: &mut [T], stride: usize, w: usize, h: usize, bd: u32) {
    fill(dst, stride, w, h, T::from_i32(1 << (bd - 1)));
}

/// Vertical prediction: each row is a copy of `above[0..w]`.
pub fn ipred_v<T: Pixel>(dst: &mut [T], stride: usize, above: &[T], w: usize, h: usize) {
    by_width!(w, v_w::<T>(dst, stride, above, h));
}

#[inline(always)]
fn v_w<T: Pixel, const W: usize>(dst: &mut [T], stride: usize, above: &[T], h: usize) {
    let (dst, stride, h4) = check_block4::<T, W>(dst, stride, h);
    let src: [T; W] = above[..W].try_into().unwrap();
    for i4 in 0..h4 {
        let base = 4 * i4 * stride;
        *row_mut::<T, W>(dst, base) = src;
        *row_mut::<T, W>(dst, base + stride) = src;
        *row_mut::<T, W>(dst, base + 2 * stride) = src;
        *row_mut::<T, W>(dst, base + 3 * stride) = src;
    }
}

/// Horizontal prediction: row `i` is filled with `left[i]`.
pub fn ipred_h<T: Pixel>(dst: &mut [T], stride: usize, left: &[T], w: usize, h: usize) {
    by_width!(w, h_w::<T>(dst, stride, left, h));
}

#[inline(always)]
fn h_w<T: Pixel, const W: usize>(dst: &mut [T], stride: usize, left: &[T], h: usize) {
    let (dst, stride, h4) = check_block4::<T, W>(dst, stride, h);
    let left = &left[..4 * h4];
    for i4 in 0..h4 {
        let base = 4 * i4 * stride;
        *row_mut::<T, W>(dst, base) = [left[4 * i4]; W];
        *row_mut::<T, W>(dst, base + stride) = [left[4 * i4 + 1]; W];
        *row_mut::<T, W>(dst, base + 2 * stride) = [left[4 * i4 + 2]; W];
        *row_mut::<T, W>(dst, base + 3 * stride) = [left[4 * i4 + 3]; W];
    }
}

// ---------------------------------------------------------------------------
// Paeth and smooth kernels.
// ---------------------------------------------------------------------------

/// Paeth prediction (spec 7.11.2.2).
pub fn ipred_paeth<T: Pixel>(
    dst: &mut [T],
    stride: usize,
    top_left: T,
    above: &[T],
    left: &[T],
    w: usize,
    h: usize,
) {
    by_width!(w, paeth_w::<T>(dst, stride, top_left, above, left, h));
}

#[inline(always)]
fn paeth_w<T: Pixel, const W: usize>(
    dst: &mut [T],
    stride: usize,
    top_left: T,
    above: &[T],
    left: &[T],
    h: usize,
) {
    let (dst, stride) = check_block::<T, W>(dst, stride, h);
    let tl = top_left.to_i32();
    let above: &[T; W] = (&above[..W]).try_into().unwrap();
    let mut a_val = [0i32; W];
    let mut a_tl = [0i32; W];
    let mut p_left = [0i32; W];
    for j in 0..W {
        let a = above[j].to_i32();
        a_val[j] = a;
        a_tl[j] = a - tl;
        p_left[j] = (a - tl).abs();
    }
    for (i, &l) in left[..h].iter().enumerate() {
        let row = row_mut::<T, W>(dst, i * stride);
        let l = l.to_i32();
        let l_tl = l - tl;
        let p_top = l_tl.abs();
        for j in 0..W {
            let pl = p_left[j];
            let ptl = (a_tl[j] + l_tl).abs();
            row[j] = T::from_i32(if pl <= p_top && pl <= ptl {
                l
            } else if p_top <= ptl {
                a_val[j]
            } else {
                tl
            });
        }
    }
}

/// Smooth prediction (spec 7.11.2.6, SMOOTH_PRED).
pub fn ipred_smooth<T: Pixel>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    w: usize,
    h: usize,
) {
    by_width!(w, smooth_w::<T>(dst, stride, above, left, h));
}

#[inline(always)]
fn smooth_w<T: Pixel, const W: usize>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    h: usize,
) {
    let (dst, stride) = check_block::<T, W>(dst, stride, h);
    let wx: &[u8; W] = (&sm_weights(W)[..W]).try_into().unwrap();
    let wy = sm_weights(h);
    let bottom = left[h - 1].to_i32();
    let right = above[W - 1].to_i32();
    let above: &[T; W] = (&above[..W]).try_into().unwrap();
    let mut a_val = [0i32; W];
    let mut wx_val = [0i32; W];
    let mut hbase = [0i32; W];
    for j in 0..W {
        a_val[j] = above[j].to_i32();
        let wxj = wx[j] as i32;
        wx_val[j] = wxj;
        hbase[j] = (256 - wxj) * right + 256;
    }
    for (i, (&l, &wyi)) in left[..h].iter().zip(wy).enumerate() {
        let row = row_mut::<T, W>(dst, i * stride);
        let wyi = wyi as i32;
        let l = l.to_i32();
        let vert_base = (256 - wyi) * bottom;
        for j in 0..W {
            let pred = wyi * a_val[j] + wx_val[j] * l + vert_base + hbase[j];
            row[j] = T::from_i32(pred >> 9);
        }
    }
}

/// Smooth vertical prediction (spec 7.11.2.6, SMOOTH_V_PRED).
pub fn ipred_smooth_v<T: Pixel>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    w: usize,
    h: usize,
) {
    by_width!(w, smooth_v_w::<T>(dst, stride, above, left, h));
}

#[inline(always)]
fn smooth_v_w<T: Pixel, const W: usize>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    h: usize,
) {
    let (dst, stride) = check_block::<T, W>(dst, stride, h);
    let wy = sm_weights(h);
    let bottom = left[h - 1].to_i32();
    let above: &[T; W] = (&above[..W]).try_into().unwrap();
    let mut a_val = [0i32; W];
    for j in 0..W {
        a_val[j] = above[j].to_i32();
    }
    for (i, &wyi) in wy.iter().enumerate() {
        let row = row_mut::<T, W>(dst, i * stride);
        let wyi = wyi as i32;
        let base = (256 - wyi) * bottom + 128;
        for j in 0..W {
            row[j] = T::from_i32((wyi * a_val[j] + base) >> 8);
        }
    }
}

/// Smooth horizontal prediction (spec 7.11.2.6, SMOOTH_H_PRED).
pub fn ipred_smooth_h<T: Pixel>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    w: usize,
    h: usize,
) {
    by_width!(w, smooth_h_w::<T>(dst, stride, above, left, h));
}

#[inline(always)]
fn smooth_h_w<T: Pixel, const W: usize>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    h: usize,
) {
    let (dst, stride) = check_block::<T, W>(dst, stride, h);
    let wx: &[u8; W] = (&sm_weights(W)[..W]).try_into().unwrap();
    let right = above[W - 1].to_i32();
    let mut wx_val = [0i32; W];
    let mut base = [0i32; W];
    for j in 0..W {
        let wxj = wx[j] as i32;
        wx_val[j] = wxj;
        base[j] = (256 - wxj) * right + 128;
    }
    for (i, &l) in left[..h].iter().enumerate() {
        let row = row_mut::<T, W>(dst, i * stride);
        let l = l.to_i32();
        for j in 0..W {
            row[j] = T::from_i32((wx_val[j] * l + base[j]) >> 8);
        }
    }
}

// ---------------------------------------------------------------------------
// Directional edge preparation.
// ---------------------------------------------------------------------------

/// Filter corner process (spec 7.11.2.7): returns the filtered top-left
/// sample from `LeftCol[0]`, `AboveRow[-1]` and `AboveRow[0]`.
#[inline]
pub fn filter_corner<T: Pixel>(left0: T, top_left: T, above0: T) -> i32 {
    round2(
        left0.to_i32() * 5 + top_left.to_i32() * 6 + above0.to_i32() * 5,
        4,
    )
}

/// Intra edge filter strength selection process (spec 7.11.2.9). Returns a
/// strength in 0..=3.
///
/// Branch free: the strength is the number of thresholds of
/// [`STRENGTH_THRESHOLDS`] that `|delta|` reaches (checked against the
/// spec's if/else chains by the `strength_tables` test).
#[inline]
pub fn edge_filter_strength(w: usize, h: usize, filter_type: bool, delta: i32) -> u32 {
    let d = delta.unsigned_abs();
    let blk_wh = w + h;
    let class = (blk_wh > 8) as usize
        + (blk_wh > 16) as usize
        + (blk_wh > 24) as usize
        + (blk_wh > 32) as usize;
    let t = &STRENGTH_THRESHOLDS[filter_type as usize][class];
    (d >= t[0] as u32) as u32 + (d >= t[1] as u32) as u32 + (d >= t[2] as u32) as u32
}

/// Thresholds of [`edge_filter_strength`] by `[filter_type][class]`, with
/// class 0..=4 for `w + h` in `..=8`, `..=16`, `..=24`, `..=32`, `33..`.
/// `u16::MAX` is never reached.
const STRENGTH_THRESHOLDS: [[[u16; 3]; 5]; 2] = {
    const N: u16 = u16::MAX;
    [
        [[56, N, N], [40, N, N], [8, 16, 32], [0, 4, 32], [0, 0, 0]],
        [[40, 64, N], [20, 48, N], [4, 4, 4], [0, 0, 0], [0, 0, 0]],
    ]
};

/// Intra edge upsample selection process (spec 7.11.2.10).
#[inline]
pub fn use_upsample(w: usize, h: usize, filter_type: bool, delta: i32) -> bool {
    let d = delta.abs();
    if d <= 0 || d >= 40 {
        false
    } else if !filter_type {
        w + h <= 16
    } else {
        w + h <= 8
    }
}

/// Intra edge filter process (spec 7.11.2.12).
///
/// `edge[i]` is the spec `edge[i]`, i.e. `AboveRow[i - 1]` / `LeftCol[i - 1]`
/// (so `edge[0]` is the top-left sample). Filters `edge[1..sz]` in place
/// using `edge[0..sz]` as input. Does nothing if `strength` is 0 or
/// `sz <= 1`. `sz` must be at most [`MAX_FILTER_EDGE`].
pub fn filter_edge<T: Pixel>(edge: &mut [T], sz: i32, strength: u32) {
    if strength == 0 || sz <= 1 {
        return;
    }
    let sz = sz as usize;
    debug_assert!(sz <= MAX_FILTER_EDGE);
    // Input copy, padded by replicating edge[0] / edge[sz - 1] twice on each
    // side, which implements the Clip3(0, sz - 1, ...) of the spec.
    let mut tmp = [0i32; MAX_FILTER_EDGE + 4];
    for (t, e) in tmp[2..sz + 2].iter_mut().zip(&edge[..sz]) {
        *t = e.to_i32();
    }
    tmp[0] = tmp[2];
    tmp[1] = tmp[2];
    tmp[sz + 2] = tmp[sz + 1];
    tmp[sz + 3] = tmp[sz + 1];
    match strength {
        1 => {
            for (i, e) in edge[1..sz].iter_mut().enumerate() {
                let s = (tmp[i + 2] + tmp[i + 4]) * 4 + tmp[i + 3] * 8;
                *e = T::from_i32((s + 8) >> 4);
            }
        }
        2 => {
            for (i, e) in edge[1..sz].iter_mut().enumerate() {
                let s = (tmp[i + 2] + tmp[i + 4]) * 5 + tmp[i + 3] * 6;
                *e = T::from_i32((s + 8) >> 4);
            }
        }
        _ => {
            for (i, e) in edge[1..sz].iter_mut().enumerate() {
                let s = (tmp[i + 1] + tmp[i + 5]) * 2 + (tmp[i + 2] + tmp[i + 3] + tmp[i + 4]) * 4;
                *e = T::from_i32((s + 8) >> 4);
            }
        }
    }
}

/// Intra edge upsample process (spec 7.11.2.11).
///
/// `buf` is a directional edge buffer: spec index `k` is stored at
/// `buf[EDGE_PAD + k]`. On entry spec entries `-1..numPx` are valid, on exit
/// entries `-2..=2 * numPx - 2` hold the upsampled edge. `num_px` must be at
/// most [`MAX_UPSAMPLE_PX`].
pub fn upsample_edge<T: Pixel>(buf: &mut [T], num_px: usize, bd: u32) {
    debug_assert!((1..=MAX_UPSAMPLE_PX).contains(&num_px));
    // dup[0] = buf[-1], dup[i + 2] = buf[i] for i = -1..numPx-1,
    // dup[numPx + 2] = buf[numPx - 1].
    let mut dup = [0i32; MAX_UPSAMPLE_PX + 3];
    for (d, b) in dup[1..num_px + 2]
        .iter_mut()
        .zip(&buf[EDGE_PAD - 1..EDGE_PAD + num_px])
    {
        *d = b.to_i32();
    }
    dup[0] = dup[1];
    dup[num_px + 2] = dup[num_px + 1];

    let out = &mut buf[EDGE_PAD - 2..];
    // out[k] is spec buf[k - 2].
    out[0] = T::from_i32(dup[0]);
    for i in 0..num_px {
        let s = 9 * (dup[i + 1] + dup[i + 2]) - dup[i] - dup[i + 3];
        out[2 * i + 1] = clip_pixel(round2(s, 4), bd);
        out[2 * i + 2] = T::from_i32(dup[i + 2]);
    }
}

// ---------------------------------------------------------------------------
// Directional kernels.
// ---------------------------------------------------------------------------

/// Directional prediction for pAngle < 90 (spec 7.11.2.4 step 7).
///
/// `above[k]` is `AboveRow[k]` (prepared, possibly upsampled) for
/// `k = 0..=maxBaseX` with `maxBaseX = (w + h - 1) << upsample`.
pub fn ipred_z1<T: Pixel>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    w: usize,
    h: usize,
    dx: i32,
    upsample: bool,
) {
    by_width!(w, z1_w::<T>(dst, stride, above, h, dx, upsample));
}

#[inline(always)]
fn z1_w<T: Pixel, const W: usize>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    h: usize,
    dx: i32,
    upsample: bool,
) {
    let (dst, stride) = check_block::<T, W>(dst, stride, h);
    let u = upsample as u32;
    let max_base_x = (W + h - 1) << u;
    let above = &above[..=max_base_x];
    let edge_val = above[max_base_x];
    for i in 0..h {
        let row = row_mut::<T, W>(dst, i * stride);
        let idx = (i as i32 + 1) * dx;
        let base0 = (idx >> (6 - u)) as usize;
        if base0 >= max_base_x {
            *row = [edge_val; W];
            continue;
        }
        let shift = ((idx << u) >> 1) & 0x1f;
        if !upsample && base0 + W <= max_base_x {
            // Full row of W contiguous samples: auto-vectorizable at fixed width W.
            let s0: &[T; W] = (&above[base0..base0 + W]).try_into().unwrap();
            let s1: &[T; W] = (&above[base0 + 1..base0 + W + 1]).try_into().unwrap();
            for j in 0..W {
                row[j] = interp(s0[j], s1[j], shift);
            }
            continue;
        }
        let n = ((max_base_x - base0 + (1 << u) - 1) >> u).min(W);
        let (interp_part, fill_part) = row.split_at_mut(n);
        if u == 0 {
            let src = &above[base0..base0 + n + 1];
            for (d, s) in interp_part.iter_mut().zip(src.windows(2)) {
                *d = interp(s[0], s[1], shift);
            }
        } else {
            for (j, d) in interp_part.iter_mut().enumerate() {
                let base = base0 + (j << u);
                *d = interp(above[base], above[base + 1], shift);
            }
        }
        fill_part.fill(edge_val);
    }
}

/// Directional prediction for 90 < pAngle < 180 (spec 7.11.2.4 step 8).
///
/// `above` and `left` are full directional edge buffers: spec index `k` is
/// stored at `EDGE_PAD + k` (spec indices down to -2 are read when
/// upsampled, -1 otherwise).
#[allow(clippy::too_many_arguments)]
pub fn ipred_z2<T: Pixel>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    w: usize,
    h: usize,
    dx: i32,
    dy: i32,
    upsample_above: bool,
    upsample_left: bool,
) {
    by_width!(
        w,
        z2_w::<T>(
            dst,
            stride,
            above,
            left,
            h,
            dx,
            dy,
            upsample_above,
            upsample_left
        )
    );
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn z2_w<T: Pixel, const W: usize>(
    dst: &mut [T],
    stride: usize,
    above: &[T],
    left: &[T],
    h: usize,
    dx: i32,
    dy: i32,
    upsample_above: bool,
    upsample_left: bool,
) {
    let (dst, stride) = check_block::<T, W>(dst, stride, h);
    let ua = upsample_above as u32;
    let ul = upsample_left as u32;
    let pad = EDGE_PAD as i32;
    // Left-edge base offset and shift depend only on column j:
    // idx = (i << 6) - (j + 1) * dy, so base = (i << ul) + base0_y[j].
    let mut base0_y = [0i32; W];
    let mut shift_y = [0i32; W];
    for j in 0..W {
        let idx0 = -(j as i32 + 1) * dy;
        base0_y[j] = (idx0 >> (6 - ul)) + pad;
        shift_y[j] = ((idx0 << ul) >> 1) & 0x1f;
    }
    for i in 0..h {
        let row = row_mut::<T, W>(dst, i * stride);
        let i_i32 = i as i32;
        // The above edge is used where base >= -(1 << ua), which is
        // equivalent to idx = (j << 6) - (i + 1) * dx >= -64.
        let t = (i_i32 + 1) * dx - 64;
        let j0 = if t <= 0 { 0 } else { (((t + 63) >> 6) as usize).min(W) };
        if j0 == 0 && !upsample_above {
            let idx = -(i_i32 + 1) * dx;
            let b = ((idx >> 6) + pad) as usize;
            let shift = (idx >> 1) & 0x1f;
            let s0: &[T; W] = (&above[b..b + W]).try_into().unwrap();
            let s1: &[T; W] = (&above[b + 1..b + W + 1]).try_into().unwrap();
            for j in 0..W {
                row[j] = interp(s0[j], s1[j], shift);
            }
            continue;
        }
        let (left_part, above_part) = row.split_at_mut(j0);
        let y_off = i_i32 << ul;
        for (j, d) in left_part.iter_mut().enumerate() {
            let b = (base0_y[j] + y_off) as usize;
            *d = interp(left[b], left[b + 1], shift_y[j]);
        }
        if ua == 0 && !above_part.is_empty() {
            let idx = (j0 as i32) * 64 - (i_i32 + 1) * dx;
            let b = ((idx >> 6) + pad) as usize;
            let shift = (idx >> 1) & 0x1f;
            let src = &above[b..b + above_part.len() + 1];
            for (d, s) in above_part.iter_mut().zip(src.windows(2)) {
                *d = interp(s[0], s[1], shift);
            }
        } else {
            for (j, d) in above_part.iter_mut().enumerate() {
                let idx = ((j + j0) as i32) * 64 - (i_i32 + 1) * dx;
                let base = (idx >> (6 - ua)) + pad;
                let shift = ((idx << ua) >> 1) & 0x1f;
                let b = base as usize;
                *d = interp(above[b], above[b + 1], shift);
            }
        }
    }
}

/// Directional prediction for pAngle > 180 (spec 7.11.2.4 step 9).
///
/// `left[k]` is `LeftCol[k]` (prepared, possibly upsampled) for `k >= 0`.
pub fn ipred_z3<T: Pixel>(
    dst: &mut [T],
    stride: usize,
    left: &[T],
    w: usize,
    h: usize,
    dy: i32,
    upsample: bool,
) {
    by_width!(w, z3_w::<T>(dst, stride, left, h, dy, upsample));
}

#[inline(always)]
fn z3_w<T: Pixel, const W: usize>(
    dst: &mut [T],
    stride: usize,
    left: &[T],
    h: usize,
    dy: i32,
    upsample: bool,
) {
    let (dst, stride) = check_block::<T, W>(dst, stride, h);
    let u = upsample as usize;
    let mut base0 = [0usize; W];
    let mut shift = [0i32; W];
    for j in 0..W {
        let idx = (j as i32 + 1) * dy;
        base0[j] = (idx >> (6 - u)) as usize;
        shift[j] = ((idx << u) >> 1) & 0x1f;
    }
    for i in 0..h {
        let row = row_mut::<T, W>(dst, i * stride);
        let off = i << u;
        for j in 0..W {
            let b = base0[j] + off;
            row[j] = interp(left[b], left[b + 1], shift[j]);
        }
    }
}

// ---------------------------------------------------------------------------
// Recursive filter intra.
// ---------------------------------------------------------------------------

/// Recursive intra prediction process (spec 7.11.2.3) with filter intra
/// mode `mode` (0..=4). `above[0..w]`, `left[0..h]` are the unfiltered
/// edges, `top_left` is `AboveRow[-1]`.
#[allow(clippy::too_many_arguments)]
pub fn ipred_filter<T: Pixel>(
    dst: &mut [T],
    stride: usize,
    top_left: T,
    above: &[T],
    left: &[T],
    w: usize,
    h: usize,
    mode: usize,
    bd: u32,
) {
    let stride = stride & 0xffff_ffff;
    assert!(stride >= w && (2..=32).contains(&h) && (4..=32).contains(&w));
    let dst = &mut dst[..(h - 1) * stride + w];
    let taps = &INTRA_FILTER_TAPS[mode];
    let max = (1 << bd) - 1;
    // top[k] is the sample above column k - 1 of the current 2-row strip
    // (top[0] is the sample above-left of column 0).
    let mut top = [0i32; 32 + 1];
    top[0] = top_left.to_i32();
    for (t, a) in top[1..=w].iter_mut().zip(&above[..w]) {
        *t = a.to_i32();
    }
    for i2 in 0..h / 2 {
        let r0 = 2 * i2 * stride;
        let r1 = r0 + stride;
        let mut p0 = top[0];
        let mut p5 = left[2 * i2].to_i32();
        let mut p6 = left[2 * i2 + 1].to_i32();
        for j4 in 0..w / 4 {
            let c = 4 * j4;
            let p1 = top[c + 1];
            let p2 = top[c + 2];
            let p3 = top[c + 3];
            let p4 = top[c + 4];
            let mut out = [0i32; 8];
            for (k, tap) in taps.iter().enumerate() {
                let pr = tap[0] as i32 * p0
                    + tap[1] as i32 * p1
                    + tap[2] as i32 * p2
                    + tap[3] as i32 * p3
                    + tap[4] as i32 * p4
                    + tap[5] as i32 * p5
                    + tap[6] as i32 * p6;
                out[k] = round2_signed(pr, INTRA_FILTER_SCALE_BITS).clamp(0, max);
            }
            let d0 = row_mut::<T, 4>(dst, r0 + c);
            d0[0] = T::from_i32(out[0]);
            d0[1] = T::from_i32(out[1]);
            d0[2] = T::from_i32(out[2]);
            d0[3] = T::from_i32(out[3]);
            let d1 = row_mut::<T, 4>(dst, r1 + c);
            d1[0] = T::from_i32(out[4]);
            d1[1] = T::from_i32(out[5]);
            d1[2] = T::from_i32(out[6]);
            d1[3] = T::from_i32(out[7]);
            p0 = p4;
            p5 = out[3];
            p6 = out[7];
            top[c + 1] = out[4];
            top[c + 2] = out[5];
            top[c + 3] = out[6];
            top[c + 4] = out[7];
        }
        top[0] = left[2 * i2 + 1].to_i32();
    }
}

// ---------------------------------------------------------------------------
// Chroma from luma (spec 7.11.5).
// ---------------------------------------------------------------------------

/// Maximum number of CFL AC values (chroma transform blocks are at most
/// 32x32 when CFL is allowed).
pub const MAX_CFL_AC: usize = 32 * 32;

/// CFL luma AC values (spec 7.11.5 up to the average subtraction).
///
/// `luma[0]` is the top-left luma sample of the block (luma position
/// `(startX << ss_x, startY << ss_y)`), rows `stride` apart. Output
/// `ac[i * w + j]` (`i < h`, `j < w`) is the `(ss_x, ss_y)` subsampled
/// luma sum at output position `(min(j, wv - 1), min(i, hv - 1))`,
/// scaled to 3 fractional bits, minus the rounded average over all
/// `w * h` values. `wv` / `hv` (`1..=w` / `1..=h`) are the number of
/// output columns / rows whose luma samples lie inside the block's luma
/// area (spec `MaxLumaW` / `MaxLumaH`): the clamping of the spec luma
/// coordinates replicates column `wv - 1` and row `hv - 1`.
///
/// Reads luma rows `(i << ss_y) + dy` for `i < hv`, `dy <= ss_y` and
/// columns `(j << ss_x) + dx` for `j < wv`, `dx <= ss_x`. All values fit
/// in `i16` (at most `8 * 4095` before and `+-8 * 4095` after the average
/// subtraction).
#[allow(clippy::too_many_arguments)]
pub fn cfl_ac<T: Pixel>(
    ac: &mut [i16],
    luma: &[T],
    stride: usize,
    w: usize,
    h: usize,
    wv: usize,
    hv: usize,
    ss_x: usize,
    ss_y: usize,
) {
    debug_assert!((1..=w).contains(&wv) && (1..=h).contains(&hv) && w * h <= MAX_CFL_AC);
    macro_rules! go {
        ($sx:literal, $sy:literal) => {
            match w {
                4 => cfl_ac_w::<T, $sx, $sy, 4>(ac, luma, stride, h, wv, hv),
                8 => cfl_ac_w::<T, $sx, $sy, 8>(ac, luma, stride, h, wv, hv),
                16 => cfl_ac_w::<T, $sx, $sy, 16>(ac, luma, stride, h, wv, hv),
                32 => cfl_ac_w::<T, $sx, $sy, 32>(ac, luma, stride, h, wv, hv),
                w => panic!("invalid cfl width {w}"),
            }
        };
    }
    match (ss_x, ss_y) {
        (1, 1) => go!(1, 1),
        (1, 0) => go!(1, 0),
        (0, 0) => go!(0, 0),
        _ => go!(0, 1),
    }
}

#[inline(always)]
fn cfl_ac_w<T: Pixel, const SX: usize, const SY: usize, const W: usize>(
    ac: &mut [i16],
    luma: &[T],
    stride: usize,
    h: usize,
    wv: usize,
    hv: usize,
) {
    let ac = &mut ac[..W * h];
    let shift = 3 - SX - SY;
    for i in 0..hv {
        let r0 = &luma[(i << SY) * stride..];
        let r1 = &luma[((i << SY) + SY) * stride..];
        let row: &mut [i16; W] = (&mut ac[i * W..(i + 1) * W]).try_into().unwrap();
        if wv == W {
            let s0 = &r0[..W << SX];
            let s1 = if SY == 1 { &r1[..W << SX] } else { s0 };
            for j in 0..W {
                let x = j << SX;
                let mut t = s0[x].to_i32();
                if SX == 1 {
                    t += s0[x + 1].to_i32();
                }
                if SY == 1 {
                    t += s1[x].to_i32();
                    if SX == 1 {
                        t += s1[x + 1].to_i32();
                    }
                }
                row[j] = (t << shift) as i16;
            }
        } else {
            for (j, d) in row[..wv].iter_mut().enumerate() {
                let x = j << SX;
                let mut t = r0[x].to_i32();
                if SX == 1 {
                    t += r0[x + 1].to_i32();
                }
                if SY == 1 {
                    t += r1[x].to_i32();
                    if SX == 1 {
                        t += r1[x + 1].to_i32();
                    }
                }
                *d = (t << shift) as i16;
            }
            let last = row[wv - 1];
            row[wv..].fill(last);
        }
    }
    for i in hv..h {
        ac.copy_within((hv - 1) * W..hv * W, i * W);
    }
    let sum: i32 = ac.iter().map(|&v| v as i32).sum();
    let avg = round2(sum, (W * h).trailing_zeros()) as i16;
    for a in ac.iter_mut() {
        *a -= avg;
    }
}

/// CFL prediction (spec 7.11.5, last step): `dst` holds the DC prediction
/// of the `w x h` block, `ac` the values from [`cfl_ac`]; each sample
/// becomes `Clip1(dc + Round2Signed(alpha * ac, 6))`.
pub fn cfl_pred<T: Pixel>(
    dst: &mut [T],
    stride: usize,
    ac: &[i16],
    w: usize,
    h: usize,
    alpha: i32,
    bd: u32,
) {
    match w {
        4 => cfl_pred_w::<T, 4>(dst, stride, ac, h, alpha, bd),
        8 => cfl_pred_w::<T, 8>(dst, stride, ac, h, alpha, bd),
        16 => cfl_pred_w::<T, 16>(dst, stride, ac, h, alpha, bd),
        32 => cfl_pred_w::<T, 32>(dst, stride, ac, h, alpha, bd),
        w => panic!("invalid cfl width {w}"),
    }
}

#[inline(always)]
fn cfl_pred_w<T: Pixel, const W: usize>(
    dst: &mut [T],
    stride: usize,
    ac: &[i16],
    h: usize,
    alpha: i32,
    bd: u32,
) {
    let (dst, stride) = check_block::<T, W>(dst, stride, h);
    let max = (1i32 << bd) - 1;
    for (i, acr) in ac[..W * h].chunks_exact(W).enumerate() {
        let row = row_mut::<T, W>(dst, i * stride);
        let acr: &[i16; W] = acr.try_into().unwrap();
        for j in 0..W {
            let p = alpha * acr[j] as i32;
            let diff = (p + (p >> 31) + 32) >> 6;
            row[j] = T::from_i32((row[j].to_i32() + diff).clamp(0, max));
        }
    }
}

#[cfg(test)]
mod tests {
    //! Differential tests against a direct transcription of the spec
    //! pseudo code (sections 7.11.2.2 - 7.11.2.12).

    use super::*;
    use crate::decoder::av1::pixel::clip3;

    /// Offset of spec index 0 in the reference edge arrays.
    const OFF: i32 = 16;
    const LEN: usize = 320;

    /// Reference implementation state: spec AboveRow / LeftCol arrays with
    /// negative index support.
    struct Spec {
        above: [i32; LEN],
        left: [i32; LEN],
        bd: u32,
    }

    impl Spec {
        fn a(&self, i: i32) -> i32 {
            self.above[(i + OFF) as usize]
        }
        fn l(&self, i: i32) -> i32 {
            self.left[(i + OFF) as usize]
        }
        fn buf(&mut self, dir: i32) -> &mut [i32; LEN] {
            if dir == 0 {
                &mut self.above
            } else {
                &mut self.left
            }
        }
        fn clip1(&self, v: i32) -> i32 {
            clip3(0, (1 << self.bd) - 1, v)
        }

        /// 7.11.2.9 (kept literal, hence the duplicated branches).
        #[allow(clippy::if_same_then_else)]
        fn strength(w: i32, h: i32, filter_type: i32, delta: i32) -> i32 {
            let d = delta.abs();
            let blk_wh = w + h;
            let mut strength = 0;
            if filter_type == 0 {
                if blk_wh <= 8 {
                    if d >= 56 {
                        strength = 1
                    }
                } else if blk_wh <= 12 {
                    if d >= 40 {
                        strength = 1
                    }
                } else if blk_wh <= 16 {
                    if d >= 40 {
                        strength = 1
                    }
                } else if blk_wh <= 24 {
                    if d >= 8 {
                        strength = 1
                    }
                    if d >= 16 {
                        strength = 2
                    }
                    if d >= 32 {
                        strength = 3
                    }
                } else if blk_wh <= 32 {
                    strength = 1;
                    if d >= 4 {
                        strength = 2
                    }
                    if d >= 32 {
                        strength = 3
                    }
                } else {
                    strength = 3
                }
            } else if blk_wh <= 8 {
                if d >= 40 {
                    strength = 1
                }
                if d >= 64 {
                    strength = 2
                }
            } else if blk_wh <= 16 {
                if d >= 20 {
                    strength = 1
                }
                if d >= 48 {
                    strength = 2
                }
            } else if blk_wh <= 24 {
                if d >= 4 {
                    strength = 3
                }
            } else {
                strength = 3
            }
            strength
        }

        /// 7.11.2.10
        fn use_upsample(w: i32, h: i32, filter_type: i32, delta: i32) -> i32 {
            let d = delta.abs();
            let blk_wh = w + h;
            if d <= 0 || d >= 40 {
                0
            } else if filter_type == 0 {
                (blk_wh <= 16) as i32
            } else {
                (blk_wh <= 8) as i32
            }
        }

        /// 7.11.2.11
        fn upsample(&mut self, num_px: i32, dir: i32) {
            let bd = self.bd;
            let buf = self.buf(dir);
            let at = |i: i32| (i + OFF) as usize;
            let mut dup = vec![0i32; (num_px + 3) as usize];
            dup[0] = buf[at(-1)];
            for i in -1..num_px {
                dup[(i + 2) as usize] = buf[at(i)];
            }
            dup[(num_px + 2) as usize] = buf[at(num_px - 1)];
            buf[at(-2)] = dup[0];
            for i in 0..num_px {
                let iu = i as usize;
                let s = -dup[iu] + (9 * dup[iu + 1]) + (9 * dup[iu + 2]) - dup[iu + 3];
                let s = clip3(0, (1 << bd) - 1, round2(s, 4));
                buf[at(2 * i - 1)] = s;
                buf[at(2 * i)] = dup[iu + 2];
            }
        }

        /// 7.11.2.12
        fn edge_filter(&mut self, sz: i32, strength: i32, left: i32) {
            if strength == 0 {
                return;
            }
            let buf = self.buf(left);
            let mut edge = vec![0i32; sz.max(0) as usize];
            for i in 0..sz {
                edge[i as usize] = buf[(i - 1 + OFF) as usize];
            }
            for i in 1..sz {
                let mut s = 0;
                for j in 0..INTRA_EDGE_TAPS as i32 {
                    let k = clip3(0, sz - 1, i - 2 + j);
                    s += INTRA_EDGE_KERNEL[(strength - 1) as usize][j as usize] * edge[k as usize];
                }
                buf[(i - 1 + OFF) as usize] = (s + 8) >> 4;
            }
        }

        /// Full prediction per spec 7.11.2.1 (dispatch part) and below.
        fn predict(&mut self, p: &IntraPredParams) -> Vec<i32> {
            let w = p.w as i32;
            let h = p.h as i32;
            let log2w = w.trailing_zeros();
            let log2h = h.trailing_zeros();
            let mut pred = vec![0i32; (w * h) as usize];
            let mode = p.mode as i32;
            {
                let mut set = |i: i32, j: i32, v: i32| pred[(i * w + j) as usize] = v;
                if let Some(fm) = p.filter_intra_mode {
                    // 7.11.2.3
                    let mut pr2 = vec![0i32; (w * h) as usize];
                    let w4 = w >> 2;
                    let h2 = h >> 1;
                    for i2 in 0..h2 {
                        for j4 in 0..w4 {
                            let mut pp = [0i32; 7];
                            for i in 0..7 {
                                pp[i as usize] = if i < 5 {
                                    if i2 == 0 {
                                        self.a((j4 << 2) + i - 1)
                                    } else if j4 == 0 && i == 0 {
                                        self.l((i2 << 1) - 1)
                                    } else {
                                        pr2[(((i2 << 1) - 1) * w + (j4 << 2) + i - 1) as usize]
                                    }
                                } else if j4 == 0 {
                                    self.l((i2 << 1) + i - 5)
                                } else {
                                    pr2[(((i2 << 1) + i - 5) * w + (j4 << 2) - 1) as usize]
                                };
                            }
                            for i1 in 0..2 {
                                for j1 in 0..4 {
                                    let mut pr = 0;
                                    for i in 0..7 {
                                        pr += INTRA_FILTER_TAPS[fm as usize]
                                            [((i1 << 2) + j1) as usize][i]
                                            as i32
                                            * pp[i];
                                    }
                                    pr2[(((i2 << 1) + i1) * w + (j4 << 2) + j1) as usize] =
                                        self.clip1(round2_signed(pr, INTRA_FILTER_SCALE_BITS));
                                }
                            }
                        }
                    }
                    return pr2;
                } else if (1..=8).contains(&mode) {
                    // 7.11.2.4
                    let p_angle = MODE_TO_ANGLE[mode as usize] + p.angle_delta * ANGLE_STEP;
                    let mut upsample_above = 0;
                    let mut upsample_left = 0;
                    if p.enable_intra_edge_filter {
                        if p_angle != 90 && p_angle != 180 {
                            if p_angle > 90 && p_angle < 180 && (w + h) >= 24 {
                                let s = self.l(0) * 5 + self.a(-1) * 6 + self.a(0) * 5;
                                let c = round2(s, 4);
                                self.left[(OFF - 1) as usize] = c;
                                self.above[(OFF - 1) as usize] = c;
                            }
                            let filter_type = p.smooth_filter_type as i32;
                            if p.have_above {
                                let strength = Self::strength(w, h, filter_type, p_angle - 90);
                                let num_px =
                                    w.min(p.max_x - p.x + 1) + if p_angle < 90 { h } else { 0 } + 1;
                                self.edge_filter(num_px, strength, 0);
                            }
                            if p.have_left {
                                let strength = Self::strength(w, h, filter_type, p_angle - 180);
                                let num_px = h.min(p.max_y - p.y + 1)
                                    + if p_angle > 180 { w } else { 0 }
                                    + 1;
                                self.edge_filter(num_px, strength, 1);
                            }
                        }
                        let filter_type = p.smooth_filter_type as i32;
                        upsample_above = Self::use_upsample(w, h, filter_type, p_angle - 90);
                        let num_px = w + if p_angle < 90 { h } else { 0 };
                        if upsample_above == 1 {
                            self.upsample(num_px, 0);
                        }
                        upsample_left = Self::use_upsample(w, h, filter_type, p_angle - 180);
                        let num_px = h + if p_angle > 180 { w } else { 0 };
                        if upsample_left == 1 {
                            self.upsample(num_px, 1);
                        }
                    }
                    let dx = if p_angle < 90 {
                        DR_INTRA_DERIVATIVE[p_angle as usize]
                    } else if p_angle > 90 && p_angle < 180 {
                        DR_INTRA_DERIVATIVE[(180 - p_angle) as usize]
                    } else {
                        0
                    };
                    let dy = if p_angle > 90 && p_angle < 180 {
                        DR_INTRA_DERIVATIVE[(p_angle - 90) as usize]
                    } else if p_angle > 180 {
                        DR_INTRA_DERIVATIVE[(270 - p_angle) as usize]
                    } else {
                        0
                    };
                    for i in 0..h {
                        for j in 0..w {
                            let v = if p_angle < 90 {
                                let idx = (i + 1) * dx;
                                let base = (idx >> (6 - upsample_above)) + (j << upsample_above);
                                let shift = ((idx << upsample_above) >> 1) & 0x1F;
                                let max_base_x = (w + h - 1) << upsample_above;
                                if base < max_base_x {
                                    round2(
                                        self.a(base) * (32 - shift) + self.a(base + 1) * shift,
                                        5,
                                    )
                                } else {
                                    self.a(max_base_x)
                                }
                            } else if p_angle > 90 && p_angle < 180 {
                                let idx = (j << 6) - (i + 1) * dx;
                                let base = idx >> (6 - upsample_above);
                                if base >= -(1 << upsample_above) {
                                    let shift = ((idx << upsample_above) >> 1) & 0x1F;
                                    round2(
                                        self.a(base) * (32 - shift) + self.a(base + 1) * shift,
                                        5,
                                    )
                                } else {
                                    let idx = (i << 6) - (j + 1) * dy;
                                    let base = idx >> (6 - upsample_left);
                                    // Not stated by the spec, but relied
                                    // upon by the optimized kernel.
                                    assert!(base >= -(1 << upsample_left));
                                    let shift = ((idx << upsample_left) >> 1) & 0x1F;
                                    round2(
                                        self.l(base) * (32 - shift) + self.l(base + 1) * shift,
                                        5,
                                    )
                                }
                            } else if p_angle > 180 {
                                let idx = (j + 1) * dy;
                                let base = (idx >> (6 - upsample_left)) + (i << upsample_left);
                                let shift = ((idx << upsample_left) >> 1) & 0x1F;
                                round2(self.l(base) * (32 - shift) + self.l(base + 1) * shift, 5)
                            } else if p_angle == 90 {
                                self.a(j)
                            } else {
                                self.l(i)
                            };
                            set(i, j, v);
                        }
                    }
                } else if mode == SMOOTH_PRED as i32 {
                    let wx = sm_weights(w as usize);
                    let wy = sm_weights(h as usize);
                    for i in 0..h {
                        for j in 0..w {
                            let (wyi, wxj) = (wy[i as usize] as i32, wx[j as usize] as i32);
                            let sp = wyi * self.a(j)
                                + (256 - wyi) * self.l(h - 1)
                                + wxj * self.l(i)
                                + (256 - wxj) * self.a(w - 1);
                            set(i, j, round2(sp, 9));
                        }
                    }
                } else if mode == SMOOTH_V_PRED as i32 {
                    let wy = sm_weights(h as usize);
                    for i in 0..h {
                        for j in 0..w {
                            let wyi = wy[i as usize] as i32;
                            set(
                                i,
                                j,
                                round2(wyi * self.a(j) + (256 - wyi) * self.l(h - 1), 8),
                            );
                        }
                    }
                } else if mode == SMOOTH_H_PRED as i32 {
                    let wx = sm_weights(w as usize);
                    for i in 0..h {
                        for j in 0..w {
                            let wxj = wx[j as usize] as i32;
                            set(
                                i,
                                j,
                                round2(wxj * self.l(i) + (256 - wxj) * self.a(w - 1), 8),
                            );
                        }
                    }
                } else if mode == DC_PRED as i32 {
                    let v = if p.have_left && p.have_above {
                        let mut sum = 0;
                        for k in 0..h {
                            sum += self.l(k);
                        }
                        for k in 0..w {
                            sum += self.a(k);
                        }
                        sum += (w + h) >> 1;
                        sum / (w + h)
                    } else if p.have_left {
                        let mut sum = 0;
                        for k in 0..h {
                            sum += self.l(k);
                        }
                        self.clip1((sum + (h >> 1)) >> log2h)
                    } else if p.have_above {
                        let mut sum = 0;
                        for k in 0..w {
                            sum += self.a(k);
                        }
                        self.clip1((sum + (w >> 1)) >> log2w)
                    } else {
                        1 << (self.bd - 1)
                    };
                    for i in 0..h {
                        for j in 0..w {
                            set(i, j, v);
                        }
                    }
                } else {
                    assert_eq!(mode, PAETH_PRED as i32);
                    for i in 0..h {
                        for j in 0..w {
                            let base = self.a(j) + self.l(i) - self.a(-1);
                            let p_left = (base - self.l(i)).abs();
                            let p_top = (base - self.a(j)).abs();
                            let p_top_left = (base - self.a(-1)).abs();
                            let v = if p_left <= p_top && p_left <= p_top_left {
                                self.l(i)
                            } else if p_top <= p_top_left {
                                self.a(j)
                            } else {
                                self.a(-1)
                            };
                            set(i, j, v);
                        }
                    }
                }
            }
            pred
        }
    }

    /// Small deterministic PRNG (xorshift64*).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
        }
        fn below(&mut self, n: u32) -> u32 {
            self.next() % n
        }
    }

    /// All (w, h) transform dimensions used by AV1.
    fn sizes() -> Vec<(usize, usize)> {
        let mut v = Vec::new();
        for lw in 2..=6i32 {
            for lh in 2..=6i32 {
                if (lw - lh).abs() <= 2 {
                    v.push((1usize << lw, 1usize << lh));
                }
            }
        }
        assert_eq!(v.len(), 19);
        v
    }

    /// Fills `edges` with test data of one of several flavours.
    fn gen_edges<T: Pixel>(rng: &mut Rng, bd: u32, n: usize, e: &mut IntraEdges<T>) {
        let max = (1i32 << bd) - 1;
        let flavour = rng.below(4);
        let sample = |rng: &mut Rng, prev: i32| -> i32 {
            match flavour {
                // Uniform random.
                0 => rng.below(max as u32 + 1) as i32,
                // Extremes only (stresses clipping).
                1 => {
                    if rng.below(2) == 0 {
                        0
                    } else {
                        max
                    }
                }
                // Smooth random walk.
                2 => clip3(0, max, prev + rng.below(33) as i32 - 16),
                // Near-constant.
                _ => clip3(0, max, prev + rng.below(3) as i32 - 1),
            }
        };
        let start = rng.below(max as u32 + 1) as i32;
        let tl = sample(rng, start);
        e.top_left = T::from_i32(tl);
        let mut prev = tl;
        for k in 0..MAX_EDGE {
            prev = sample(rng, prev);
            e.above[k] = T::from_i32(if k < n { prev } else { 0 });
        }
        prev = tl;
        for k in 0..MAX_EDGE {
            prev = sample(rng, prev);
            e.left[k] = T::from_i32(if k < n { prev } else { 0 });
        }
    }

    fn spec_from<T: Pixel>(e: &IntraEdges<T>, bd: u32) -> Spec {
        let mut s = Spec {
            above: [0; LEN],
            left: [0; LEN],
            bd,
        };
        s.above[(OFF - 1) as usize] = e.top_left.to_i32();
        s.left[(OFF - 1) as usize] = e.top_left.to_i32();
        for k in 0..MAX_EDGE {
            s.above[OFF as usize + k] = e.above[k].to_i32();
            s.left[OFF as usize + k] = e.left[k].to_i32();
        }
        s
    }

    /// Runs the optimized predictor and the spec reference and compares.
    fn check<T: Pixel>(e: &IntraEdges<T>, p: &IntraPredParams, rng: &mut Rng) {
        let (w, h) = (p.w, p.h);
        let stride = w + rng.below(9) as usize;
        let sentinel = T::from_i32(0x5a);
        let mut dst = vec![sentinel; stride * h];
        predict_intra(&mut dst, stride, e, p);
        let expected = spec_from(e, p.bd).predict(p);
        for i in 0..h {
            for j in 0..stride {
                let got = dst[i * stride + j];
                if j >= w {
                    assert_eq!(got, sentinel, "write outside block at ({i},{j}) {p:?}");
                    continue;
                }
                assert_eq!(
                    got.to_i32(),
                    expected[i * w + j],
                    "mismatch at ({i},{j}) for {p:?}"
                );
            }
        }
    }

    fn run_all<T: Pixel>(bd: u32, seed: u64, trials: usize) {
        let mut rng = Rng(seed);
        let mut e = IntraEdges::<T>::default();
        for (w, h) in sizes() {
            // (mode, angle_delta, filter_intra_mode)
            let mut configs: Vec<(u8, i32, Option<u8>)> = Vec::new();
            for mode in 0u8..=12 {
                if (V_PRED..=D67_PRED).contains(&mode) {
                    for d in -3..=3 {
                        configs.push((mode, d, None));
                    }
                } else {
                    configs.push((mode, 0, None));
                }
            }
            if w <= 32 && h <= 32 {
                for fm in 0u8..5 {
                    configs.push((DC_PRED, 0, Some(fm)));
                }
            }
            for &(mode, angle_delta, filter_intra_mode) in &configs {
                for flags in 0..16u32 {
                    for _ in 0..trials {
                        gen_edges(&mut rng, bd, w + h, &mut e);
                        let x = 4 * rng.below(64) as i32;
                        let y = 4 * rng.below(64) as i32;
                        // Distance to the frame edge: often beyond the
                        // block, sometimes inside it.
                        let max_x = x + rng.below(2 * w as u32 + 2) as i32;
                        let max_y = y + rng.below(2 * h as u32 + 2) as i32;
                        let p = IntraPredParams {
                            mode,
                            angle_delta,
                            filter_intra_mode,
                            have_left: flags & 1 != 0,
                            have_above: flags & 2 != 0,
                            enable_intra_edge_filter: flags & 4 != 0,
                            smooth_filter_type: flags & 8 != 0,
                            x,
                            y,
                            max_x,
                            max_y,
                            w,
                            h,
                            bd,
                        };
                        check(&e, &p, &mut rng);
                    }
                }
            }
        }
    }

    #[test]
    fn matches_spec_8bit() {
        run_all::<u8>(8, 0x1234_5678_9abc_def1, 3);
    }

    #[test]
    fn matches_spec_8bit_in_u16() {
        run_all::<u16>(8, 0x0f0f_1234_5555_aaaa, 1);
    }

    #[test]
    fn matches_spec_10bit() {
        run_all::<u16>(10, 0xdead_beef_cafe_f00d, 3);
    }

    #[test]
    fn matches_spec_12bit() {
        run_all::<u16>(12, 0x0123_4567_89ab_cdef, 3);
    }

    /// CFL (spec 7.11.5, transcribed literally with the MaxLumaW /
    /// MaxLumaH clamping) against `cfl_ac` + `cfl_pred`, with `wv` / `hv`
    /// derived from MaxLumaW / MaxLumaH as a caller does.
    fn run_cfl<T: Pixel>(bd: u32, seed: u64, trials: usize) {
        let mut rng = Rng(seed);
        let max = (1i32 << bd) - 1;
        const LS: usize = 80;
        let mut luma = vec![T::default(); LS * LS];
        let mut ac = [0i16; MAX_CFL_AC];
        for (w, h) in sizes() {
            if w > 32 || h > 32 {
                continue;
            }
            for (sx, sy) in [(0usize, 0usize), (1, 0), (1, 1)] {
                for _ in 0..trials {
                    let extreme = rng.below(4) == 0;
                    for v in luma.iter_mut() {
                        *v = T::from_i32(if extreme {
                            [0, max][rng.below(2) as usize]
                        } else {
                            rng.below(max as u32 + 1) as i32
                        });
                    }
                    // Chroma block at (start_x, start_y); the luma area
                    // (a multiple of 4 luma samples in the decoder, so
                    // the clamped coordinate is on the subsampling grid)
                    // ends somewhere inside or after the block.
                    let start_x = rng.below(4) as usize;
                    let start_y = rng.below(4) as usize;
                    let max_luma_w =
                        (start_x << sx) + ((1 + rng.below(w as u32 + 2) as usize) << sx);
                    let max_luma_h =
                        (start_y << sy) + ((1 + rng.below(h as u32 + 2) as usize) << sy);
                    let alpha = rng.below(33) as i32 - 16;
                    let dc = rng.below(max as u32 + 1) as i32;
                    let lp = |x: usize, y: usize| luma[y * LS + x].to_i32();
                    // Spec.
                    let mut l = vec![0i32; w * h];
                    let mut sum = 0;
                    for i in 0..h {
                        let luma_y = ((start_y + i) << sy).min(max_luma_h - (1 << sy));
                        for j in 0..w {
                            let luma_x = ((start_x + j) << sx).min(max_luma_w - (1 << sx));
                            let mut t = 0;
                            for dy in 0..=sy {
                                for dx in 0..=sx {
                                    t += lp(luma_x + dx, luma_y + dy);
                                }
                            }
                            l[i * w + j] = t << (3 - sx - sy);
                            sum += l[i * w + j];
                        }
                    }
                    let avg = round2(sum, w.trailing_zeros() + h.trailing_zeros());
                    let expected: Vec<i32> = l
                        .iter()
                        .map(|&v| clip3(0, max, dc + round2_signed(alpha * (v - avg), 6)))
                        .collect();
                    // Kernels.
                    let (lx0, ly0) = (start_x << sx, start_y << sy);
                    let wv = (((max_luma_w - (1 << sx) - lx0) >> sx) + 1).min(w);
                    let hv = (((max_luma_h - (1 << sy) - ly0) >> sy) + 1).min(h);
                    cfl_ac(&mut ac, &luma[ly0 * LS + lx0..], LS, w, h, wv, hv, sx, sy);
                    let stride = w + rng.below(5) as usize;
                    let mut dst = vec![T::from_i32(dc); stride * (h - 1) + w];
                    cfl_pred(&mut dst, stride, &ac, w, h, alpha, bd);
                    for i in 0..h {
                        for j in 0..w {
                            assert_eq!(
                                dst[i * stride + j].to_i32(),
                                expected[i * w + j],
                                "cfl {w}x{h} ss {sx},{sy} at {i},{j}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn cfl_matches_spec() {
        run_cfl::<u8>(8, 0xcf10_0008, 50);
        run_cfl::<u16>(10, 0xcf10_0010, 50);
        run_cfl::<u16>(12, 0xcf10_0012, 50);
    }

    /// The z2 kernel indexes the left edge from spec index -(1 << upsample)
    /// on; check this holds for every block size, angle and upsampling
    /// combination (it depends only on these, not on sample values).
    #[test]
    fn z2_left_index_bounds() {
        for (w, h) in sizes() {
            let (w, h) = (w as i32, h as i32);
            for p_angle in 91..180 {
                let dx = DR_INTRA_DERIVATIVE[(180 - p_angle) as usize];
                let dy = DR_INTRA_DERIVATIVE[(p_angle - 90) as usize];
                if dx == 0 || dy == 0 {
                    continue;
                }
                for ua in 0..2 {
                    for ul in 0..2 {
                        for i in 0..h {
                            for j in 0..w {
                                let idx = (j << 6) - (i + 1) * dx;
                                let base = idx >> (6 - ua);
                                if base >= -(1 << ua) {
                                    assert!(base < (w << ua));
                                } else {
                                    let idx = (i << 6) - (j + 1) * dy;
                                    let base = idx >> (6 - ul);
                                    assert!(base >= -(1 << ul), "w={w} h={h} angle={p_angle}");
                                    assert!(base < (h << ul));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Upsampling is only selected when the edge filter strength is 0
    /// and the corner is not filtered (w + h < 24); the SIMD edge
    /// preparation relies on this.
    #[test]
    fn upsample_implies_no_filter() {
        for (w, h) in sizes() {
            for ft in [false, true] {
                for delta in -180..=180 {
                    if use_upsample(w, h, ft, delta) {
                        assert_eq!(
                            edge_filter_strength(w, h, ft, delta),
                            0,
                            "{w}x{h} ft {ft} delta {delta}"
                        );
                        assert!(w + h < 24);
                    }
                }
            }
        }
    }

    #[test]
    fn strength_tables() {
        // Spot checks of 7.11.2.9 / 7.11.2.10 against the reference.
        for w in [4usize, 8, 16, 32, 64] {
            for h in [4usize, 8, 16, 32, 64] {
                for ft in [false, true] {
                    for delta in -120..=120 {
                        assert_eq!(
                            edge_filter_strength(w, h, ft, delta) as i32,
                            Spec::strength(w as i32, h as i32, ft as i32, delta)
                        );
                        assert_eq!(
                            use_upsample(w, h, ft, delta) as i32,
                            Spec::use_upsample(w as i32, h as i32, ft as i32, delta)
                        );
                    }
                }
            }
        }
    }
}
