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

//! Inverse transforms (spec 7.13) and reconstruction add (spec 7.12.3
//! step 2 and 3).
//!
//! OWNER: itx worker. Contract below must be kept.
//!
//! # Structure
//!
//! * The 1D kernels (`idct`, `iadst4/8/16`, `iidentity*`, `iwht4`) are
//!   direct transcriptions of the spec butterfly networks (7.13.2). They
//!   share the signature `Tx1dFn` and work in place on the first
//!   `1 << n` entries of a `[i32; 64]` scratch array. Internally they are
//!   monomorphised per length on `[i32; L]`, so loop bounds, indices and
//!   angles are compile-time constants and the optimiser emits
//!   straight-line code. A SIMD implementation can replace these kernels
//!   (or the whole row / column passes in `transform_add`) without
//!   touching the 2D logic.
//! * [`inv_txfm_add`] runs the 2D process of 7.13.3: row transforms (all
//!   zero rows are skipped since every 1D transform maps zero to zero),
//!   intermediate clamping, column transforms, then the add/clip step of
//!   7.12.3 with flipping. A DC-only fast path is used for `DCT_DCT`.
//!
//! # Arithmetic
//!
//! Products are computed in `i64`. For conformant streams every value fits
//! in 32 bits, but non-conformant (yet decodable) 12-bit streams could
//! overflow 32-bit products; using 64-bit products keeps the code free of
//! overflow for every input that respects the `Dequant` range, and keeps
//! it bit exact with the spec's (unbounded integer) arithmetic everywhere.
//! Stored intermediates always fit in `i32`.

use crate::decoder::av1::pixel::{clip_pixel, Pixel};

mod kernels;
use crate::decoder::av1::types::{TxSize, TxType};

/// Spec `Tx_Width_Log2`, indexed by `TxSize`.
pub(crate) const TX_WIDTH_LOG2: [u8; TxSize::COUNT] =
    [2, 3, 4, 5, 6, 2, 3, 3, 4, 4, 5, 5, 6, 2, 4, 3, 5, 4, 6];
/// Spec `Tx_Height_Log2`, indexed by `TxSize`.
pub(crate) const TX_HEIGHT_LOG2: [u8; TxSize::COUNT] =
    [2, 3, 4, 5, 6, 3, 2, 4, 3, 5, 4, 6, 5, 4, 2, 5, 3, 6, 4];
/// Spec `Transform_Row_Shift`, indexed by `TxSize`.
pub(crate) const TRANSFORM_ROW_SHIFT: [u8; TxSize::COUNT] =
    [0, 1, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2];

/// Spec `Cos128_Lookup` (the generated kernels have the values inlined).
#[cfg(test)]
const COS128_LOOKUP: [i32; 65] = [
    4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973, 3948, 3920, 3889, 3857, 3822,
    3784, 3745, 3703, 3659, 3612, 3564, 3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967,
    2896, 2824, 2751, 2675, 2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019, 1931, 1842, 1751, 1660,
    1567, 1474, 1380, 1285, 1189, 1092, 995, 897, 799, 700, 601, 501, 401, 301, 201, 101, 0,
];

const SINPI_1_9: i64 = 1321;
const SINPI_2_9: i64 = 2482;
const SINPI_3_9: i64 = 3344;
const SINPI_4_9: i64 = 3803;

/// One dimensional transform kind (spec 7.13.3 classification).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Tx1d {
    Dct,
    /// ADST and FLIPADST (flipping is done when adding to the frame).
    Adst,
    Identity,
}

/// Returns the (column / vertical, row / horizontal) 1D transform kinds of
/// a 2D transform type.
#[inline]
pub(crate) fn tx1d_kinds(tx_type: TxType) -> (Tx1d, Tx1d) {
    use Tx1d::*;
    use TxType::*;
    match tx_type {
        DCT_DCT => (Dct, Dct),
        ADST_DCT | FLIPADST_DCT => (Adst, Dct),
        DCT_ADST | DCT_FLIPADST => (Dct, Adst),
        ADST_ADST | FLIPADST_FLIPADST | ADST_FLIPADST | FLIPADST_ADST => (Adst, Adst),
        IDTX => (Identity, Identity),
        V_DCT => (Dct, Identity),
        H_DCT => (Identity, Dct),
        V_ADST | V_FLIPADST => (Adst, Identity),
        H_ADST | H_FLIPADST => (Identity, Adst),
    }
}

/// Spec `flipUD` (7.12.3).
#[inline]
pub(crate) fn flip_ud(tx_type: TxType) -> bool {
    use TxType::*;
    const M: u16 = bits(&[FLIPADST_DCT, FLIPADST_ADST, V_FLIPADST, FLIPADST_FLIPADST]);
    M >> tx_type as u16 & 1 != 0
}

/// Bit mask of a set of transform types.
const fn bits(t: &[TxType]) -> u16 {
    let mut m = 0;
    let mut i = 0;
    while i < t.len() {
        m |= 1 << t[i] as u16;
        i += 1;
    }
    m
}

/// Spec `flipLR` (7.12.3).
#[inline]
pub(crate) fn flip_lr(tx_type: TxType) -> bool {
    use TxType::*;
    const M: u16 = bits(&[DCT_FLIPADST, ADST_FLIPADST, H_FLIPADST, FLIPADST_FLIPADST]);
    M >> tx_type as u16 & 1 != 0
}

/// Signed clamping range of `r` bits: `[-(1 << (r - 1)), (1 << (r - 1)) - 1]`.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Clamp {
    pub(crate) lo: i32,
    pub(crate) hi: i32,
}

impl Clamp {
    #[inline]
    pub(crate) fn bits(r: u32) -> Self {
        Clamp {
            lo: -(1 << (r - 1)),
            hi: (1 << (r - 1)) - 1,
        }
    }

    #[inline(always)]
    pub(crate) fn apply(self, v: i32) -> i32 {
        v.clamp(self.lo, self.hi)
    }
}

/// A 1D inverse transform working in place on the first `1 << n` entries.
type Tx1dFn = fn(&mut [i32; 64], Clamp);

// ---------------------------------------------------------------------------
// Butterfly helpers (7.13.2.1)
// ---------------------------------------------------------------------------

/// `Round2(x, 12)` of a 64-bit product sum, narrowed to 32 bits.
#[inline(always)]
pub(crate) fn round12(x: i64) -> i32 {
    ((x + (1 << 11)) >> 12) as i32
}

/// `Round2(x, n)` for 32-bit values (n may be 0).
#[inline(always)]
pub(crate) fn round2(x: i32, n: u32) -> i32 {
    if n == 0 {
        x
    } else {
        (x + (1 << (n - 1))) >> n
    }
}

/// `Round2(x, 12)` (short name for the generated kernels).
#[inline(always)]
fn r12(x: i64) -> i32 {
    let v = round12(x);
    #[cfg(test)]
    conformance::check(v as i64, 0);
    v
}

/// Test-only tracking of the bitstream conformance requirements on the
/// butterfly outputs (7.13.2.1: representable with `r` bits; 7.13.2.6:
/// ADST4 sums with `r + 12` bits). Implementations with narrower
/// intermediates (`simd::itx::S16`) only match the spec for inputs that
/// meet them.
#[cfg(test)]
pub(crate) mod conformance {
    use std::cell::Cell;

    thread_local! {
        static RANGE: Cell<u32> = const { Cell::new(0) };
        static VIOLATED: Cell<bool> = const { Cell::new(false) };
    }

    /// Sets the clamping range `r` of the current pass (0: no checks).
    pub(crate) fn set_range(r: u32) {
        RANGE.with(|c| c.set(r));
    }

    /// Records a violation if `v` does not fit in `r + extra` bits.
    #[inline(always)]
    pub(crate) fn check(v: i64, extra: u32) {
        let r = RANGE.with(|c| c.get());
        if r != 0 {
            let b = r + extra;
            if v < -(1i64 << (b - 1)) || v > (1i64 << (b - 1)) - 1 {
                VIOLATED.with(|c| c.set(true));
            }
        }
    }

    /// Returns and clears the violation flag.
    pub(crate) fn take_violated() -> bool {
        VIOLATED.with(|c| c.replace(false))
    }
}

// Lane helpers for the column pass kernels (`*_l`), which transform `L`
// columns at once. Their i32 arithmetic cannot overflow: inputs and
// Hadamard outputs are clamped to colClampRange <= 18 bits and the
// generator checks the magnitude of every product sum.

/// Lane-wise `Round2(a * ca + b * cb, 12)`.
#[inline(always)]
fn lmul2<const L: usize>(a: &[i32; L], ca: i32, b: &[i32; L], cb: i32) -> [i32; L] {
    let mut o = [0; L];
    for l in 0..L {
        let s = match (ca < 0, cb < 0) {
            (false, false) => a[l] * ca + b[l] * cb,
            (false, true) => a[l] * ca - b[l] * -cb,
            (true, false) => b[l] * cb - a[l] * -ca,
            (true, true) => -(a[l] * -ca + b[l] * -cb),
        };
        o[l] = (s + (1 << 11)) >> 12;
        #[cfg(test)]
        conformance::check(o[l] as i64, 0);
    }
    o
}

/// Lane-wise `Round2(a * ca, 12)`.
#[inline(always)]
fn lmul1<const L: usize>(a: &[i32; L], ca: i32) -> [i32; L] {
    let mut o = [0; L];
    for l in 0..L {
        o[l] = (a[l] * ca + (1 << 11)) >> 12;
        #[cfg(test)]
        conformance::check(o[l] as i64, 0);
    }
    o
}

/// Lane-wise clamp.
#[inline(always)]
fn lclamp<const L: usize>(a: [i32; L], r: Clamp) -> [i32; L] {
    a.map(|v| v.max(r.lo).min(r.hi))
}

/// Lane-wise negation.
#[inline(always)]
fn lneg<const L: usize>(a: [i32; L]) -> [i32; L] {
    a.map(|v| -v)
}

/// Lane-wise clamped sum (Hadamard rotation output `a + b`).
#[inline(always)]
fn ladd<const L: usize>(a: &[i32; L], b: &[i32; L], r: Clamp) -> [i32; L] {
    let mut o = [0; L];
    for l in 0..L {
        o[l] = (a[l] + b[l]).max(r.lo).min(r.hi);
    }
    o
}

/// Lane-wise clamped difference (Hadamard rotation output `a - b`).
#[inline(always)]
fn lsub<const L: usize>(a: &[i32; L], b: &[i32; L], r: Clamp) -> [i32; L] {
    let mut o = [0; L];
    for l in 0..L {
        o[l] = (a[l] - b[l]).max(r.lo).min(r.hi);
    }
    o
}

// ---------------------------------------------------------------------------
// 1D transforms (7.13.2)
// ---------------------------------------------------------------------------

/// Inverse ADST4 (7.13.2.6). No intermediate clamping.
fn iadst4(t: &mut [i32; 64], _r: Clamp) {
    let t0 = t[0] as i64;
    let t1 = t[1] as i64;
    let t2 = t[2] as i64;
    let t3 = t[3] as i64;
    let mut s0 = SINPI_1_9 * t0;
    let mut s1 = SINPI_2_9 * t0;
    let mut s2 = SINPI_3_9 * t1;
    let mut s3 = SINPI_4_9 * t2;
    let s4 = SINPI_1_9 * t2;
    let s5 = SINPI_2_9 * t3;
    let s6 = SINPI_4_9 * t3;
    let a7 = t0 - t2;
    let b7 = a7 + t3;

    s0 += s3;
    s1 -= s4;
    s3 = s2;
    s2 = SINPI_3_9 * b7;

    s0 += s5;
    s1 -= s6;

    let x0 = s0 + s3;
    let x1 = s1 + s3;
    let x2 = s2;
    let x3 = s0 + s1 - s3;

    #[cfg(test)]
    for x in [x0, x1, x2, x3] {
        conformance::check(x, 12);
    }
    t[0] = round12(x0);
    t[1] = round12(x1);
    t[2] = round12(x2);
    t[3] = round12(x3);
}

/// Inverse identity transform 4 (7.13.2.11).
fn iidentity4(t: &mut [i32; 64], _r: Clamp) {
    for v in &mut t[..4] {
        *v = round12(*v as i64 * 5793);
    }
}

/// Inverse identity transform 8 (7.13.2.12).
fn iidentity8(t: &mut [i32; 64], _r: Clamp) {
    for v in &mut t[..8] {
        *v *= 2;
    }
}

/// Inverse identity transform 16 (7.13.2.13).
fn iidentity16(t: &mut [i32; 64], _r: Clamp) {
    for v in &mut t[..16] {
        *v = round12(*v as i64 * 11586);
    }
}

/// Inverse identity transform 32 (7.13.2.14).
fn iidentity32(t: &mut [i32; 64], _r: Clamp) {
    for v in &mut t[..32] {
        *v *= 4;
    }
}

/// Inverse Walsh-Hadamard transform (7.13.2.10).
#[inline(always)]
fn iwht4(t: &mut [i32; 4], shift: u32) {
    let mut a = t[0] >> shift;
    let mut c = t[1] >> shift;
    let mut d = t[2] >> shift;
    let mut b = t[3] >> shift;
    a += c;
    d -= b;
    let e = (a - d) >> 1;
    b = e - b;
    c = e - c;
    a -= b;
    d += c;
    *t = [a, b, c, d];
}

/// Selects the 1D kernel for a transform kind and length `1 << log2n`,
/// given that only the first `nz` inputs (`1 <= nz <= 32`) can be
/// non-zero. The straight-line DCT / ADST kernels are generated
/// (`itx/gen_kernels.py`) in variants for 4, 8, 16 and 32 non-zero inputs
/// that skip all operations on known zeros.
///
/// Combinations that cannot occur in a conformant stream (ADST longer than
/// 16, identity longer than 32) fall back to the DCT so that the function
/// never panics in release builds.
fn select_1d(kind: Tx1d, log2n: usize, nz: usize) -> Tx1dFn {
    use kernels::*;
    // Index of the smallest variant with at least `nz` inputs.
    let v = match nz {
        0..=4 => 0,
        5..=8 => 1,
        9..=16 => 2,
        _ => 3,
    };
    match (kind, log2n) {
        (Tx1d::Adst, 2) => iadst4,
        (Tx1d::Adst, 3) => [iadst8_nz4, iadst8_nz8][v.min(1)],
        (Tx1d::Adst, 4) => [iadst16_nz4, iadst16_nz8, iadst16_nz16][v.min(2)],
        (Tx1d::Identity, 2) => iidentity4,
        (Tx1d::Identity, 3) => iidentity8,
        (Tx1d::Identity, 4) => iidentity16,
        (Tx1d::Identity, 5) => iidentity32,
        (kind, n) => {
            debug_assert!(
                kind == Tx1d::Dct,
                "illegal {kind:?} transform of length {}",
                1 << n
            );
            match n {
                2 => idct4_nz4,
                3 => [idct8_nz4, idct8_nz8][v.min(1)],
                4 => [idct16_nz4, idct16_nz8, idct16_nz16][v.min(2)],
                5 => [idct32_nz4, idct32_nz8, idct32_nz16, idct32_nz32][v],
                _ => [idct64_nz4, idct64_nz8, idct64_nz16, idct64_nz32][v],
            }
        }
    }
}

/// A column pass kernel transforming `L` columns at once (`t[i][l]` is
/// row `i` of column `l`).
type Tx1dLanesFn<const L: usize> = fn(&mut [[i32; L]], Clamp);

/// Lane version of [`select_1d`] for the column pass (only valid with the
/// column clamping range).
fn select_1d_lanes<const L: usize>(kind: Tx1d, log2n: usize, nz: usize) -> Option<Tx1dLanesFn<L>> {
    use kernels::*;
    let v = match nz {
        0..=4 => 0,
        5..=8 => 1,
        9..=16 => 2,
        _ => 3,
    };
    Some(match (kind, log2n) {
        (Tx1d::Adst, 2) => iadst4_l::<L>,
        (Tx1d::Adst, 3) => [iadst8_nz4_l::<L>, iadst8_nz8_l::<L>][v.min(1)],
        (Tx1d::Adst, 4) => [iadst16_nz4_l::<L>, iadst16_nz8_l::<L>, iadst16_nz16_l::<L>][v.min(2)],
        (Tx1d::Identity, 2) => iidentity_l::<L, 4>,
        (Tx1d::Identity, 3) => iidentity_l::<L, 8>,
        (Tx1d::Identity, 4) => iidentity_l::<L, 16>,
        (Tx1d::Identity, 5) => iidentity_l::<L, 32>,
        (_, n) => match n {
            2 => idct4_nz4_l::<L>,
            3 => [idct8_nz4_l::<L>, idct8_nz8_l::<L>][v.min(1)],
            4 => [idct16_nz4_l::<L>, idct16_nz8_l::<L>, idct16_nz16_l::<L>][v.min(2)],
            5 => [
                idct32_nz4_l::<L>,
                idct32_nz8_l::<L>,
                idct32_nz16_l::<L>,
                idct32_nz32_l::<L>,
            ][v],
            _ => [
                idct64_nz4_l::<L>,
                idct64_nz8_l::<L>,
                idct64_nz16_l::<L>,
                idct64_nz32_l::<L>,
            ][v],
        },
    })
}

/// Lane version of [`iadst4`] for the column pass (with `colClampRange <= 18`
/// bits, every product sum has magnitude `< 2^31 - 2048` and fits in `i32`).
fn iadst4_l<const L: usize>(t: &mut [[i32; L]], _r: Clamp) {
    let [t0, t1, t2, t3] = [t[0], t[1], t[2], t[3]];
    for l in 0..L {
        let (x0, x1, x2, x3) = (t0[l], t1[l], t2[l], t3[l]);
        let s0 = 1321 * x0 + 3803 * x2 + 2482 * x3;
        let s1 = 2482 * x0 - 1321 * x2 - 3803 * x3;
        let s2 = 3344 * (x0 - x2 + x3);
        let s3 = 3344 * x1;
        let y0 = s0 + s3;
        let y1 = s1 + s3;
        let y2 = s2;
        let y3 = s0 + s1 - s3;
        #[cfg(test)]
        for y in [y0, y1, y2, y3] {
            conformance::check(y as i64, 12);
        }
        t[0][l] = (y0 + 2048) >> 12;
        t[1][l] = (y1 + 2048) >> 12;
        t[2][l] = (y2 + 2048) >> 12;
        t[3][l] = (y3 + 2048) >> 12;
    }
}

/// Lane version of the inverse identity transforms of length `N`
/// (7.13.2.11 - 7.13.2.14). With inputs below 2^17 the products fit in
/// 32 bits.
fn iidentity_l<const L: usize, const N: usize>(t: &mut [[i32; L]], _r: Clamp) {
    for row in &mut t[..N] {
        for v in row.iter_mut() {
            *v = match N {
                4 => (*v * 5793 + (1 << 11)) >> 12,
                8 => *v * 2,
                16 => (*v * 11586 + (1 << 11)) >> 12,
                _ => *v * 4,
            };
        }
    }
}

/// [`select_1d`] for an arbitrary (dense) input, including DCT64 inputs
/// 32..63 (never coded in AV1).
#[cfg(test)]
fn select_1d_full(kind: Tx1d, log2n: usize) -> Tx1dFn {
    if kind == Tx1d::Dct && log2n == 6 {
        kernels::idct64_nz64
    } else {
        select_1d(kind, log2n, 32)
    }
}

// ---------------------------------------------------------------------------
// 2D transform + reconstruction
// ---------------------------------------------------------------------------

/// Upper bounds `(eob_rows, eob_cols)` on `1 + max_r` and `1 + max_c`
/// of the first `eob` scan positions (spec 5.11.41 `get_scan`).
#[inline(always)]
pub(crate) fn eob_bounds(
    eob: usize,
    tw: usize,
    th: usize,
    row_kind: Tx1d,
    col_kind: Tx1d,
) -> (usize, usize) {
    if eob <= 1 {
        return (eob, eob);
    }
    match (
        tw.max(th) <= 16,
        row_kind == Tx1d::Identity,
        col_kind == Tx1d::Identity,
    ) {
        (true, true, false) => (eob.div_ceil(tw).min(th), eob.min(tw)),
        (true, false, true) => (eob.min(th), eob.div_ceil(th).min(tw)),
        _ => {
            // DEFAULT_SCAN visits anti-diagonals r + c = 0, 1, ... in order.
            let m = tw.min(th);
            let k = if eob <= 3 {
                2
            } else if eob <= 6 {
                3
            } else if eob <= 10 {
                4
            } else if eob <= if m == 4 { 26 } else { 36 } {
                8
            } else if eob <= if m <= 8 { 100 } else { 136 } {
                16
            } else {
                32
            };
            (k.min(th), k.min(tw))
        }
    }
}

/// Runs the 2D inverse transform (spec 7.13.3) on dequantized
/// coefficients and adds the residual to `dst` with clipping (spec 7.12.3
/// step 3, including flipUD / flipLR handling).
///
/// * `dst`: `dst[0]` is the top-left sample of the transform block; rows are
///   `stride` samples apart. The full `w x h` block (w/h = transform width /
///   height, up to 64) is guaranteed to be inside `dst`.
/// * `coeffs`: `Dequant[i][j]` (row `i`, column `j`) stored at
///   `coeffs[i * tw + j]` where `tw = min(32, w)`, `th = min(32, h)`.
///   `coeffs.len() >= tw * th`. Values are already dequantized and clamped
///   as in spec 7.12.3 step 1. The function may use it as scratch, and
///   leaves `coeffs[..tw * th]` all zero on return (the decoder relies on
///   it to start the next block from zeros).
/// * `tx_type`: the plane transform type (`PlaneTxType`).
/// * `lossless`: when true the Walsh-Hadamard transform is used (tx_size is
///   then always `Tx4x4`).
/// * `eob`: end of block (number of coded coefficients in scan order, >= 1).
///   Only a hint for fast paths (e.g. DC-only); may be ignored.
/// * `bd`: bit depth (8, 10 or 12).
#[allow(clippy::too_many_arguments)]
pub fn inv_txfm_add<T: Pixel>(
    dst: &mut [T],
    stride: usize,
    coeffs: &mut [i32],
    tx_size: TxSize,
    tx_type: TxType,
    lossless: bool,
    eob: usize,
    bd: u32,
) {
    if eob == 0 {
        return;
    }
    let col_clamp = Clamp::bits((bd + 6).max(16));
    let fud = flip_ud(tx_type);
    let flr = flip_lr(tx_type);
    if lossless {
        debug_assert_eq!(tx_size, TxSize::Tx4x4);
        inv_wht4x4_add(dst, stride, coeffs, col_clamp, fud, flr, bd);
        coeffs[..16].fill(0);
        return;
    }

    let log2w = TX_WIDTH_LOG2[tx_size.idx()] as usize;
    let log2h = TX_HEIGHT_LOG2[tx_size.idx()] as usize;
    let w = 1usize << log2w;
    let h = 1usize << log2h;
    let tw = w.min(32);
    let th = h.min(32);
    let row_shift = TRANSFORM_ROW_SHIFT[tx_size.idx()] as u32;
    let rect2 = log2w.abs_diff(log2h) == 1;
    let row_clamp = Clamp::bits(bd + 8);
    let coeffs = &mut coeffs[..tw * th];

    if eob == 1
        && matches!(
            tx_type,
            TxType::DCT_DCT | TxType::IDTX | TxType::V_DCT | TxType::H_DCT
        )
    {
        // Only Dequant[0][0] can be nonzero.
        let mut dc = coeffs[0];
        coeffs[0] = 0;
        if rect2 {
            dc = round12(dc as i64 * 2896);
        }
        dc = if matches!(tx_type, TxType::DCT_DCT | TxType::H_DCT) {
            row_clamp.apply(round12(dc as i64 * 2896))
        } else {
            match log2w {
                2 => round12(dc as i64 * 5793),
                3 => dc * 2,
                4 => round12(dc as i64 * 11586),
                _ => dc * 4,
            }
        };
        dc = col_clamp.apply(round2(dc, row_shift));
        dc = if matches!(tx_type, TxType::DCT_DCT | TxType::V_DCT) {
            col_clamp.apply(round12(dc as i64 * 2896))
        } else {
            match log2h {
                2 => round12(dc as i64 * 5793),
                3 => dc * 2,
                4 => round12(dc as i64 * 11586),
                _ => dc * 4,
            }
        };
        let dc = round2(dc, 4);
        if dc == 0 {
            return;
        }
        match tx_type {
            TxType::IDTX => dst[0] = clip_pixel(dst[0].to_i32() + dc, bd),
            TxType::V_DCT => {
                for y in 0..h {
                    let d = &mut dst[y * stride];
                    *d = clip_pixel(d.to_i32() + dc, bd);
                }
            }
            TxType::H_DCT => add_dc(dst, stride, w, 1, dc, bd),
            _ => add_dc(dst, stride, w, h, dc, bd),
        }
        return;
    }

    let (col_kind, row_kind) = tx1d_kinds(tx_type);
    let (eob_rows, eob_cols) = eob_bounds(eob, tw, th, row_kind, col_kind);
    let p = Params {
        w,
        h,
        tw,
        th,
        log2w,
        log2h,
        rect2,
        row_shift,
        row_kind,
        col_kind,
        row_clamp,
        col_clamp,
        flip_ud: fud,
        flip_lr: flr,
        bd,
        eob_rows,
        eob_cols,
    };
    if w <= 32 {
        // tw == w: row transforms in place in the coefficient buffer.
        let nz_rows = transform_add(dst, stride, coeffs, &p);
        coeffs[..nz_rows * w].fill(0);
    } else {
        // Only the first eob_rows <= th <= 32 rows can be non-zero.
        let mut buf = [0i32; 64 * 32];
        let active = eob_rows * tw;
        buf[..active].copy_from_slice(&coeffs[..active]);
        coeffs[..active].fill(0);
        transform_add(dst, stride, &mut buf[..w * th], &p);
    }
}

/// Parameters of a (non-lossless) 2D inverse transform.
struct Params {
    w: usize,
    h: usize,
    tw: usize,
    th: usize,
    log2w: usize,
    log2h: usize,
    rect2: bool,
    row_shift: u32,
    row_kind: Tx1d,
    col_kind: Tx1d,
    row_clamp: Clamp,
    col_clamp: Clamp,
    flip_ud: bool,
    flip_lr: bool,
    bd: u32,
    eob_rows: usize,
    eob_cols: usize,
}

/// 2D transform core. On entry `buf[i * tw + j]` holds `Dequant[i][j]` for
/// `i < th`, `j < tw`. `buf.len() == w * th`. The row transform outputs
/// are computed in place (`buf[i * w + j]`), then the column transforms
/// are run and their outputs added to `dst`.
///
/// Zero rows / columns are tracked so that the 1D kernels for "only the
/// first `nz` inputs non-zero" can be used (see [`select_1d`]).
fn transform_add<T: Pixel>(dst: &mut [T], stride: usize, buf: &mut [i32], p: &Params) -> usize {
    let (w, h, tw) = (p.w, p.h, p.tw);
    let mut t = [0i32; 64];

    // Row transforms. Rows are processed bottom-up so that writing row i
    // (stride w) never clobbers unread input rows (stride tw <= w).
    // Input rows at or beyond eob_rows are zero and so are their outputs; so are
    // the outputs of all-zero rows. `nz_rows` is one more than the index
    // of the last non-zero row.
    #[cfg(test)]
    conformance::set_range(p.row_clamp.hi.trailing_ones() + 1);
    let mut nz_rows = 0;
    for i in (0..p.eob_rows).rev() {
        let src = &buf[i * tw..i * tw + p.eob_cols];
        let nz = match src.iter().rposition(|&v| v != 0) {
            Some(k) => k + 1,
            None => {
                // Rows at or beyond nz_rows are never read.
                if tw != w && nz_rows != 0 {
                    buf[i * w..i * w + w].fill(0);
                }
                continue;
            }
        };
        if nz_rows == 0 {
            nz_rows = i + 1;
        }
        if nz == 1 && p.row_kind == Tx1d::Dct {
            // DC only: every DCT output is clamp(Round2(in * cos128(32),
            // 12)) (see dct_dc).
            let mut v = src[0];
            if p.rect2 {
                v = round12(v as i64 * 2896);
            }
            let v = p
                .col_clamp
                .apply(round2(dct_dc(v, p.row_clamp), p.row_shift));
            buf[i * w..i * w + w].fill(v);
            continue;
        }
        // The kernels read at most t[..tw] (DCT64 kernels only t[..32]).
        t[..nz].copy_from_slice(&src[..nz]);
        t[nz..tw].fill(0);
        if p.rect2 {
            for v in &mut t[..nz] {
                *v = round12(*v as i64 * 2896);
            }
        }
        select_1d(p.row_kind, p.log2w, nz)(&mut t, p.row_clamp);
        for (o, &v) in buf[i * w..i * w + w].iter_mut().zip(&t[..w]) {
            *o = p.col_clamp.apply(round2(v, p.row_shift));
        }
    }
    if nz_rows == 0 {
        return 0;
    }

    #[cfg(test)]
    conformance::set_range(p.col_clamp.hi.trailing_ones() + 1);
    // Column transforms, with the final Round2(T[i], colShift = 4), added
    // to the frame with the flips of 7.12.3 step 3.
    if nz_rows == 1 && p.col_kind == Tx1d::Dct {
        // Every column is constant.
        let mut res = [0i32; 64];
        for (r, &v) in res[..w].iter_mut().zip(&buf[..w]) {
            *r = round2(dct_dc(v, p.col_clamp), 4);
        }
        if p.flip_lr {
            res[..w].reverse();
        }
        for y in 0..h {
            add_row(&mut dst[y * stride..y * stride + w], &res[..w], p.bd);
        }
        return nz_rows;
    }
    match (w, h) {
        (4, 4) => col_pass::<T, 4, 4>(dst, stride, buf, p, nz_rows),
        (4, 8) => col_pass::<T, 4, 8>(dst, stride, buf, p, nz_rows),
        (4, _) => col_pass::<T, 4, 16>(dst, stride, buf, p, nz_rows),
        (_, 4) => col_pass::<T, 8, 4>(dst, stride, buf, p, nz_rows),
        (_, 8) => col_pass::<T, 8, 8>(dst, stride, buf, p, nz_rows),
        (_, 16) => col_pass::<T, 8, 16>(dst, stride, buf, p, nz_rows),
        (_, 32) => col_pass::<T, 8, 32>(dst, stride, buf, p, nz_rows),
        _ => col_pass::<T, 8, 64>(dst, stride, buf, p, nz_rows),
    }
    nz_rows
}

/// Column transforms of a transform of height `H`, `L` columns at a time
/// (`L <= w`), with the final Round2(T[i], colShift = 4), added to the
/// frame with the flips of 7.12.3 step 3. Rows at or beyond `nz_rows`
/// of `buf` are zero.
fn col_pass<T: Pixel, const L: usize, const H: usize>(
    dst: &mut [T],
    stride: usize,
    buf: &[i32],
    p: &Params,
    nz_rows: usize,
) {
    debug_assert_eq!(H, p.h);
    let w = p.w;
    if p.col_kind == Tx1d::Identity {
        for i in 0..nz_rows {
            let y = if p.flip_ud { H - 1 - i } else { i };
            let y_off = y * stride;
            let src_row = &buf[i * w..i * w + w];
            for j0 in (0..w).step_by(L) {
                let c: [i32; L] = src_row[j0..j0 + L].try_into().unwrap();
                let mut res = c.map(|v| {
                    let y = match H {
                        4 => (v * 5793 + 2048) >> 12,
                        8 => v * 2,
                        16 => (v * 11586 + 2048) >> 12,
                        _ => v * 4,
                    };
                    round2(y, 4)
                });
                let x0 = if p.flip_lr {
                    res.reverse();
                    w - j0 - L
                } else {
                    j0
                };
                add_row(&mut dst[y_off + x0..y_off + x0 + L], &res, p.bd);
            }
        }
        return;
    }
    let kernel = select_1d_lanes::<L>(p.col_kind, p.log2h, nz_rows).unwrap();
    let k = nz_rows.next_power_of_two().max(4).min(H);
    let mut cols = [[0i32; L]; H];
    for j0 in (0..w).step_by(L) {
        for (i, c) in cols[..nz_rows].iter_mut().enumerate() {
            *c = buf[i * w + j0..i * w + j0 + L].try_into().unwrap();
        }
        // The kernel reads at most the first `k` rows and overwrites all H rows.
        cols[nz_rows..k].fill([0; L]);
        kernel(&mut cols, p.col_clamp);
        if !p.flip_lr && !p.flip_ud {
            let mut off = j0;
            for c in &cols {
                let res = c.map(|v| round2(v, 4));
                add_row(&mut dst[off..off + L], &res, p.bd);
                off += stride;
            }
        } else {
            let x0 = if p.flip_lr { w - j0 - L } else { j0 };
            let mut off = if p.flip_ud { (H - 1) * stride + x0 } else { x0 };
            for c in &cols {
                let mut res = c.map(|v| round2(v, 4));
                if p.flip_lr {
                    res.reverse();
                }
                add_row(&mut dst[off..off + L], &res, p.bd);
                off = if p.flip_ud { off.wrapping_sub(stride) } else { off + stride };
            }
        }
    }
}

/// Output of a DCT (any length) whose only non-zero input is `t0`:
/// `B(0, 1, 32)` turns it into two copies of `Round2(t0 * 2896, 12)` and
/// every later stage either adds zeros (Hadamard steps, which clamp) or
/// rotates zeros. As the clamp is idempotent, every output equals
/// `clamp(Round2(t0 * 2896, 12))`.
#[inline(always)]
fn dct_dc(t0: i32, r: Clamp) -> i32 {
    r.apply(round12(t0 as i64 * 2896))
}

/// Adds residuals to a row of pixels with clipping to `[0, (1 << bd) - 1]`.
#[inline(always)]
fn add_row<T: Pixel>(dst: &mut [T], res: &[i32], bd: u32) {
    let max = (1 << bd) - 1;
    for (d, &r) in dst.iter_mut().zip(res) {
        *d = T::from_i32((d.to_i32() + r).clamp(0, max));
    }
}

/// Adds a `w x h` residual block (row-major, stride `w`) to `dst` with
/// clipping, applying the flips of spec 7.12.3 step 3.
#[allow(clippy::too_many_arguments)]
fn add_residual<T: Pixel>(
    dst: &mut [T],
    stride: usize,
    res: &[i32],
    w: usize,
    h: usize,
    flip_ud: bool,
    flip_lr: bool,
    bd: u32,
) {
    for (i, res_row) in res.chunks_exact(w).take(h).enumerate() {
        let y = if flip_ud { h - 1 - i } else { i };
        let dst_row = &mut dst[y * stride..y * stride + w];
        if flip_lr {
            for (d, &r) in dst_row.iter_mut().rev().zip(res_row) {
                *d = clip_pixel(d.to_i32() + r, bd);
            }
        } else {
            for (d, &r) in dst_row.iter_mut().zip(res_row) {
                *d = clip_pixel(d.to_i32() + r, bd);
            }
        }
    }
}

/// Adds a constant residual to a `w x h` block with clipping.
fn add_dc<T: Pixel>(dst: &mut [T], stride: usize, w: usize, h: usize, dc: i32, bd: u32) {
    for y in 0..h {
        for d in &mut dst[y * stride..y * stride + w] {
            *d = clip_pixel(d.to_i32() + dc, bd);
        }
    }
}

/// Lossless 4x4 Walsh-Hadamard reconstruction (7.13.3 with Lossless = 1).
fn inv_wht4x4_add<T: Pixel>(
    dst: &mut [T],
    stride: usize,
    coeffs: &mut [i32],
    col_clamp: Clamp,
    flip_ud: bool,
    flip_lr: bool,
    bd: u32,
) {
    let mut res = [0i32; 16];
    for i in 0..4 {
        let mut t = [
            coeffs[i * 4],
            coeffs[i * 4 + 1],
            coeffs[i * 4 + 2],
            coeffs[i * 4 + 3],
        ];
        iwht4(&mut t, 2);
        for (o, v) in res[i * 4..i * 4 + 4].iter_mut().zip(t) {
            *o = col_clamp.apply(v);
        }
    }
    for j in 0..4 {
        let mut t = [res[j], res[4 + j], res[8 + j], res[12 + j]];
        iwht4(&mut t, 0);
        for (i, v) in t.into_iter().enumerate() {
            res[i * 4 + j] = v;
        }
    }
    add_residual(dst, stride, &res, 4, 4, flip_ud, flip_lr, bd);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct transcription of the spec pseudo code (7.13 and 7.12.3 steps
    /// 2-3) using 64-bit integers everywhere, runtime sizes and no fast
    /// paths. Used as the reference for differential tests.
    #[allow(
        clippy::neg_multiply,
        clippy::assign_op_pattern,
        clippy::needless_range_loop
    )]
    mod spec {
        pub struct Tx {
            pub t: [i64; 64],
        }

        fn brev(num_bits: u32, x: usize) -> usize {
            let mut t = 0;
            for i in 0..num_bits {
                let bit = (x >> i) & 1;
                t += bit << (num_bits - 1 - i);
            }
            t
        }

        const COS128_LOOKUP: [i64; 65] = super::COS128_LOOKUP_I64;

        fn cos128(angle: i64) -> i64 {
            let angle2 = angle & 255;
            if (0..=64).contains(&angle2) {
                COS128_LOOKUP[angle2 as usize]
            } else if angle2 > 64 && angle2 <= 128 {
                COS128_LOOKUP[(128 - angle2) as usize] * -1
            } else if angle2 > 128 && angle2 <= 192 {
                COS128_LOOKUP[(angle2 - 128) as usize] * -1
            } else {
                COS128_LOOKUP[(256 - angle2) as usize]
            }
        }

        fn sin128(angle: i64) -> i64 {
            cos128(angle - 64)
        }

        fn round2(x: i64, n: u32) -> i64 {
            if n == 0 {
                x
            } else {
                (x + (1 << (n - 1))) >> n
            }
        }

        impl Tx {
            fn b(&mut self, a: usize, b: usize, angle: i64, flip: u32, _r: u32) {
                let x = self.t[a] * cos128(angle) - self.t[b] * sin128(angle);
                let y = self.t[a] * sin128(angle) + self.t[b] * cos128(angle);
                self.t[a] = round2(x, 12);
                self.t[b] = round2(y, 12);
                if flip == 1 {
                    self.t.swap(a, b);
                }
            }

            fn h(&mut self, a: usize, b: usize, flip: usize, r: u32) {
                if flip == 1 {
                    return self.h(b, a, 0, r);
                }
                let x = self.t[a];
                let y = self.t[b];
                let lo = -(1i64 << (r - 1));
                let hi = (1i64 << (r - 1)) - 1;
                self.t[a] = (x + y).clamp(lo, hi);
                self.t[b] = (x - y).clamp(lo, hi);
            }

            fn dct_perm(&mut self, n: u32) {
                let copy = self.t;
                for i in 0..(1usize << n) {
                    self.t[i] = copy[brev(n, i)];
                }
            }

            pub fn dct(&mut self, n: u32, r: u32) {
                self.dct_perm(n);
                if n == 6 {
                    for i in 0..16 {
                        self.b(32 + i, 63 - i, 63 - 4 * brev(4, i) as i64, 0, r);
                    }
                }
                if n >= 5 {
                    for i in 0..8 {
                        self.b(16 + i, 31 - i, 6 + ((brev(3, 7 - i) as i64) << 3), 0, r);
                    }
                }
                if n == 6 {
                    for i in 0..16 {
                        self.h(32 + i * 2, 33 + i * 2, i & 1, r);
                    }
                }
                if n >= 4 {
                    for i in 0..4 {
                        self.b(8 + i, 15 - i, 12 + ((brev(2, 3 - i) as i64) << 4), 0, r);
                    }
                }
                if n >= 5 {
                    for i in 0..8 {
                        self.h(16 + 2 * i, 17 + 2 * i, i & 1, r);
                    }
                }
                if n == 6 {
                    for i in 0..4 {
                        for j in 0..2 {
                            self.b(
                                62 - i * 4 - j,
                                33 + i * 4 + j,
                                60 - 16 * brev(2, i) as i64 + 64 * j as i64,
                                1,
                                r,
                            );
                        }
                    }
                }
                if n >= 3 {
                    for i in 0..2 {
                        self.b(4 + i, 7 - i, 56 - 32 * i as i64, 0, r);
                    }
                }
                if n >= 4 {
                    for i in 0..4 {
                        self.h(8 + 2 * i, 9 + 2 * i, i & 1, r);
                    }
                }
                if n >= 5 {
                    for i in 0..2 {
                        for j in 0..2 {
                            self.b(
                                30 - 4 * i - j,
                                17 + 4 * i + j,
                                24 + ((j as i64) << 6) + ((1 - i as i64) << 5),
                                1,
                                r,
                            );
                        }
                    }
                }
                if n == 6 {
                    for i in 0..8 {
                        for j in 0..2 {
                            self.h(32 + i * 4 + j, 35 + i * 4 - j, i & 1, r);
                        }
                    }
                }
                for i in 0..2 {
                    self.b(2 * i, 2 * i + 1, 32 + 16 * i as i64, 1 - i as u32, r);
                }
                if n >= 3 {
                    for i in 0..2 {
                        self.h(4 + 2 * i, 5 + 2 * i, i, r);
                    }
                }
                if n >= 4 {
                    for i in 0..2 {
                        self.b(14 - i, 9 + i, 48 + 64 * i as i64, 1, r);
                    }
                }
                if n >= 5 {
                    for i in 0..4 {
                        for j in 0..2 {
                            self.h(16 + 4 * i + j, 19 + 4 * i - j, i & 1, r);
                        }
                    }
                }
                if n == 6 {
                    for i in 0..2 {
                        for j in 0..4 {
                            self.b(
                                61 - i * 8 - j,
                                34 + i * 8 + j,
                                56 - i as i64 * 32 + (j as i64 >> 1) * 64,
                                1,
                                r,
                            );
                        }
                    }
                }
                for i in 0..2 {
                    self.h(i, 3 - i, 0, r);
                }
                if n >= 3 {
                    self.b(6, 5, 32, 1, r);
                }
                if n >= 4 {
                    for i in 0..2 {
                        for j in 0..2 {
                            self.h(8 + 4 * i + j, 11 + 4 * i - j, i, r);
                        }
                    }
                }
                if n >= 5 {
                    for i in 0..4 {
                        self.b(29 - i, 18 + i, 48 + (i as i64 >> 1) * 64, 1, r);
                    }
                }
                if n == 6 {
                    for i in 0..4 {
                        for j in 0..4 {
                            self.h(32 + 8 * i + j, 39 + 8 * i - j, i & 1, r);
                        }
                    }
                }
                if n >= 3 {
                    for i in 0..4 {
                        self.h(i, 7 - i, 0, r);
                    }
                }
                if n >= 4 {
                    for i in 0..2 {
                        self.b(13 - i, 10 + i, 32, 1, r);
                    }
                }
                if n >= 5 {
                    for i in 0..2 {
                        for j in 0..4 {
                            self.h(16 + i * 8 + j, 23 + i * 8 - j, i, r);
                        }
                    }
                }
                if n == 6 {
                    for i in 0..8 {
                        self.b(59 - i, 36 + i, if i < 4 { 48 } else { 112 }, 1, r);
                    }
                }
                if n >= 4 {
                    for i in 0..8 {
                        self.h(i, 15 - i, 0, r);
                    }
                }
                if n >= 5 {
                    for i in 0..4 {
                        self.b(27 - i, 20 + i, 32, 1, r);
                    }
                }
                if n == 6 {
                    for i in 0..8 {
                        self.h(32 + i, 47 - i, 0, r);
                        self.h(48 + i, 63 - i, 1, r);
                    }
                }
                if n >= 5 {
                    for i in 0..16 {
                        self.h(i, 31 - i, 0, r);
                    }
                }
                if n == 6 {
                    for i in 0..8 {
                        self.b(55 - i, 40 + i, 32, 1, r);
                    }
                }
                if n == 6 {
                    for i in 0..32 {
                        self.h(i, 63 - i, 0, r);
                    }
                }
            }

            fn adst_in_perm(&mut self, n: u32) {
                let n0 = 1usize << n;
                let copy = self.t;
                for i in 0..n0 {
                    let idx = if i & 1 == 1 { i - 1 } else { n0 - i - 1 };
                    self.t[i] = copy[idx];
                }
            }

            fn adst_out_perm(&mut self, n: u32) {
                let n0 = 1usize << n;
                let copy = self.t;
                for i in 0..n0 {
                    let a = (i >> 3) & 1;
                    let b = ((i >> 2) & 1) ^ ((i >> 3) & 1);
                    let c = ((i >> 1) & 1) ^ ((i >> 2) & 1);
                    let d = (i & 1) ^ ((i >> 1) & 1);
                    let idx = ((d << 3) | (c << 2) | (b << 1) | a) >> (4 - n);
                    self.t[i] = if i & 1 == 1 { -copy[idx] } else { copy[idx] };
                }
            }

            fn adst4(&mut self) {
                let t = &mut self.t;
                let (sinpi_1_9, sinpi_2_9, sinpi_3_9, sinpi_4_9) = (1321, 2482, 3344, 3803);
                let mut s = [0i64; 7];
                let mut x = [0i64; 4];
                s[0] = sinpi_1_9 * t[0];
                s[1] = sinpi_2_9 * t[0];
                s[2] = sinpi_3_9 * t[1];
                s[3] = sinpi_4_9 * t[2];
                s[4] = sinpi_1_9 * t[2];
                s[5] = sinpi_2_9 * t[3];
                s[6] = sinpi_4_9 * t[3];
                let a7 = t[0] - t[2];
                let b7 = a7 + t[3];
                s[0] = s[0] + s[3];
                s[1] = s[1] - s[4];
                s[3] = s[2];
                s[2] = sinpi_3_9 * b7;
                s[0] = s[0] + s[5];
                s[1] = s[1] - s[6];
                x[0] = s[0] + s[3];
                x[1] = s[1] + s[3];
                x[2] = s[2];
                x[3] = s[0] + s[1];
                x[3] = x[3] - s[3];
                for i in 0..4 {
                    t[i] = round2(x[i], 12);
                }
            }

            fn adst8(&mut self, r: u32) {
                self.adst_in_perm(3);
                for i in 0..4 {
                    self.b(2 * i, 2 * i + 1, 60 - 16 * i as i64, 1, r);
                }
                for i in 0..4 {
                    self.h(i, 4 + i, 0, r);
                }
                for i in 0..2 {
                    self.b(4 + 3 * i, 5 + i, 48 - 32 * i as i64, 1, r);
                }
                for i in 0..2 {
                    for j in 0..2 {
                        self.h(4 * j + i, 2 + 4 * j + i, 0, r);
                    }
                }
                for i in 0..2 {
                    self.b(2 + 4 * i, 3 + 4 * i, 32, 1, r);
                }
                self.adst_out_perm(3);
            }

            fn adst16(&mut self, r: u32) {
                self.adst_in_perm(4);
                for i in 0..8 {
                    self.b(2 * i, 2 * i + 1, 62 - 8 * i as i64, 1, r);
                }
                for i in 0..8 {
                    self.h(i, 8 + i, 0, r);
                }
                for i in 0..2 {
                    self.b(8 + 2 * i, 9 + 2 * i, 56 - 32 * i as i64, 1, r);
                    self.b(13 + 2 * i, 12 + 2 * i, 8 + 32 * i as i64, 1, r);
                }
                for i in 0..4 {
                    for j in 0..2 {
                        self.h(8 * j + i, 4 + 8 * j + i, 0, r);
                    }
                }
                for i in 0..2 {
                    for j in 0..2 {
                        self.b(4 + 8 * j + 3 * i, 5 + 8 * j + i, 48 - 32 * i as i64, 1, r);
                    }
                }
                for i in 0..2 {
                    for j in 0..4 {
                        self.h(4 * j + i, 2 + 4 * j + i, 0, r);
                    }
                }
                for i in 0..4 {
                    self.b(2 + 4 * i, 3 + 4 * i, 32, 1, r);
                }
                self.adst_out_perm(4);
            }

            pub fn adst(&mut self, n: u32, r: u32) {
                match n {
                    2 => self.adst4(),
                    3 => self.adst8(r),
                    _ => self.adst16(r),
                }
            }

            pub fn wht(&mut self, shift: u32) {
                let t = &mut self.t;
                let mut a = t[0] >> shift;
                let mut c = t[1] >> shift;
                let mut d = t[2] >> shift;
                let mut b = t[3] >> shift;
                a += c;
                d -= b;
                let e = (a - d) >> 1;
                b = e - b;
                c = e - c;
                a -= b;
                d += c;
                t[0] = a;
                t[1] = b;
                t[2] = c;
                t[3] = d;
            }

            pub fn identity(&mut self, n: u32) {
                for i in 0..(1usize << n) {
                    self.t[i] = match n {
                        2 => round2(self.t[i] * 5793, 12),
                        3 => self.t[i] * 2,
                        4 => round2(self.t[i] * 11586, 12),
                        _ => self.t[i] * 4,
                    };
                }
            }
        }

        /// 7.13.3 followed by 7.12.3 step 3. `dequant[i][j]`, frame
        /// samples in `frame` (stride `stride`).
        #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
        pub fn reconstruct(
            frame: &mut [i64],
            stride: usize,
            dequant: &[[i64; 64]; 64],
            tx_sz: usize,
            plane_tx_type: usize,
            lossless: bool,
            bit_depth: u32,
        ) {
            let log2w = super::TX_WIDTH_LOG2[tx_sz] as u32;
            let log2h = super::TX_HEIGHT_LOG2[tx_sz] as u32;
            let w = 1usize << log2w;
            let h = 1usize << log2h;
            let row_shift = if lossless { 0 } else { super::TRANSFORM_ROW_SHIFT[tx_sz] as u32 };
            let col_shift = if lossless { 0 } else { 4 };
            let row_clamp_range = bit_depth + 8;
            let col_clamp_range = (bit_depth + 6).max(16);
            // TxType numbering (spec order).
            const DCT_DCT: usize = 0;
            const ADST_DCT: usize = 1;
            const DCT_ADST: usize = 2;
            const ADST_ADST: usize = 3;
            const FLIPADST_DCT: usize = 4;
            const DCT_FLIPADST: usize = 5;
            const FLIPADST_FLIPADST: usize = 6;
            const ADST_FLIPADST: usize = 7;
            const FLIPADST_ADST: usize = 8;
            const V_DCT: usize = 10;
            const H_DCT: usize = 11;
            const V_ADST: usize = 12;
            const H_ADST: usize = 13;
            const V_FLIPADST: usize = 14;
            const H_FLIPADST: usize = 15;
            let ptt = plane_tx_type;
            let mut residual = [[0i64; 64]; 64];
            let mut tx = Tx { t: [0; 64] };
            for i in 0..h {
                for j in 0..w {
                    tx.t[j] = if i < 32 && j < 32 { dequant[i][j] } else { 0 };
                }
                if log2w.abs_diff(log2h) == 1 {
                    for j in 0..w {
                        tx.t[j] = round2(tx.t[j] * 2896, 12);
                    }
                }
                if lossless {
                    tx.wht(2);
                } else if [DCT_DCT, ADST_DCT, FLIPADST_DCT, H_DCT].contains(&ptt) {
                    tx.dct(log2w, row_clamp_range);
                } else if [
                    DCT_ADST,
                    ADST_ADST,
                    DCT_FLIPADST,
                    FLIPADST_FLIPADST,
                    ADST_FLIPADST,
                    FLIPADST_ADST,
                    H_ADST,
                    H_FLIPADST,
                ]
                .contains(&ptt)
                {
                    tx.adst(log2w, row_clamp_range);
                } else {
                    tx.identity(log2w);
                }
                for j in 0..w {
                    residual[i][j] = round2(tx.t[j], row_shift);
                }
            }
            let lo = -(1i64 << (col_clamp_range - 1));
            let hi = (1i64 << (col_clamp_range - 1)) - 1;
            for i in 0..h {
                for j in 0..w {
                    residual[i][j] = residual[i][j].clamp(lo, hi);
                }
            }
            for j in 0..w {
                for i in 0..h {
                    tx.t[i] = residual[i][j];
                }
                if lossless {
                    tx.wht(0);
                } else if [DCT_DCT, DCT_ADST, DCT_FLIPADST, V_DCT].contains(&ptt) {
                    tx.dct(log2h, col_clamp_range);
                } else if [
                    ADST_DCT,
                    ADST_ADST,
                    FLIPADST_DCT,
                    FLIPADST_FLIPADST,
                    ADST_FLIPADST,
                    FLIPADST_ADST,
                    V_ADST,
                    V_FLIPADST,
                ]
                .contains(&ptt)
                {
                    tx.adst(log2h, col_clamp_range);
                } else {
                    tx.identity(log2h);
                }
                for i in 0..h {
                    residual[i][j] = round2(tx.t[i], col_shift);
                }
            }
            let flip_ud =
                [FLIPADST_DCT, FLIPADST_ADST, V_FLIPADST, FLIPADST_FLIPADST].contains(&ptt);
            let flip_lr =
                [DCT_FLIPADST, ADST_FLIPADST, H_FLIPADST, FLIPADST_FLIPADST].contains(&ptt);
            for i in 0..h {
                for j in 0..w {
                    let xx = if flip_lr { w - j - 1 } else { j };
                    let yy = if flip_ud { h - i - 1 } else { i };
                    let p = &mut frame[yy * stride + xx];
                    *p = (*p + residual[i][j]).clamp(0, (1 << bit_depth) - 1);
                }
            }
        }
    }

    const COS128_LOOKUP_I64: [i64; 65] = {
        let mut out = [0i64; 65];
        let mut i = 0;
        while i < 65 {
            out[i] = COS128_LOOKUP[i] as i64;
            i += 1;
        }
        out
    };

    /// Small deterministic PRNG (xorshift64*).
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        /// Uniform in `[lo, hi]`.
        fn range(&mut self, lo: i64, hi: i64) -> i64 {
            lo + (self.next() % ((hi - lo + 1) as u64)) as i64
        }
    }

    /// Transform types whose 1D components are defined for this size.
    fn types_for(tx_size: TxSize) -> Vec<TxType> {
        let w = 1 << TX_WIDTH_LOG2[tx_size.idx()];
        let h = 1 << TX_HEIGHT_LOG2[tx_size.idx()];
        let max = w.max(h);
        if max == 64 {
            vec![TxType::DCT_DCT]
        } else if max == 32 {
            // Spec only allows DCT_DCT and IDTX here; also test the mixed
            // DCT / identity types where each 1D transform is defined
            // (identity up to 32, ADST up to 16).
            TxType::ALL
                .iter()
                .copied()
                .filter(|&t| {
                    let (c, r) = tx1d_kinds(t);
                    let ok = |k: Tx1d, n: usize| k != Tx1d::Adst || n <= 16;
                    ok(c, h) && ok(r, w)
                })
                .collect()
        } else {
            TxType::ALL.to_vec()
        }
    }

    /// Kinds of coefficient blocks generated by the tests.
    #[derive(Copy, Clone, Debug)]
    enum Dist {
        /// Only the DC coefficient (eob = 1).
        DcOnly,
        /// A few low-frequency coefficients with realistic magnitudes.
        Sparse,
        /// All coefficients, moderate magnitudes.
        Dense,
        /// All coefficients over the full Dequant range (non-conformant,
        /// exercises the clamps).
        Extreme,
        /// All zero coefficients (eob > 1 still possible after masking).
        Zero,
    }

    fn gen_coeffs(rng: &mut Rng, dist: Dist, tw: usize, th: usize, bd: u32) -> (Vec<i32>, usize) {
        let max = (1i64 << (7 + bd)) - 1;
        let min = -(1i64 << (7 + bd));
        let mut c = vec![0i32; tw * th];
        let mut eob = tw * th;
        match dist {
            Dist::DcOnly => {
                c[0] = rng.range(min, max) as i32;
                eob = 1;
            }
            Dist::Sparse => {
                let n = rng.range(1, 8);
                let amp = 1i64 << (bd + rng.range(0, 6) as u32);
                for _ in 0..n {
                    let i = (rng.range(0, th as i64 - 1) * rng.range(0, 3) / 3) as usize;
                    let j = (rng.range(0, tw as i64 - 1) * rng.range(0, 3) / 3) as usize;
                    c[i * tw + j] = rng.range(-amp, amp).clamp(min, max) as i32;
                }
            }
            Dist::Dense => {
                let amp = 1i64 << (bd as i64 + rng.range(-2, 3)) as u32;
                for v in &mut c {
                    *v = rng.range(-amp, amp) as i32;
                }
            }
            Dist::Extreme => {
                for v in &mut c {
                    *v = match rng.range(0, 3) {
                        0 => min,
                        1 => max,
                        _ => rng.range(min, max),
                    } as i32;
                }
            }
            Dist::Zero => {}
        }
        (c, eob)
    }

    fn run_case<T: Pixel>(
        rng: &mut Rng,
        tx_size: TxSize,
        tx_type: TxType,
        lossless: bool,
        dist: Dist,
        bd: u32,
    ) {
        let w = 1usize << TX_WIDTH_LOG2[tx_size.idx()];
        let h = 1usize << TX_HEIGHT_LOG2[tx_size.idx()];
        let (tw, th) = (w.min(32), h.min(32));
        let (mut coeffs, eob) = gen_coeffs(rng, dist, tw, th, bd);
        let stride = w + rng.range(0, 5) as usize;
        let pix_max = (1i64 << bd) - 1;
        let frame: Vec<i64> = (0..stride * h).map(|_| rng.range(0, pix_max)).collect();

        let mut dequant = Box::new([[0i64; 64]; 64]);
        for i in 0..th {
            for j in 0..tw {
                dequant[i][j] = coeffs[i * tw + j] as i64;
            }
        }
        let mut expected = frame.clone();
        spec::reconstruct(
            &mut expected,
            stride,
            &dequant,
            tx_size.idx(),
            tx_type.idx(),
            lossless,
            bd,
        );

        let mut got: Vec<T> = frame.iter().map(|&v| T::from_i32(v as i32)).collect();
        inv_txfm_add(
            &mut got,
            stride,
            &mut coeffs,
            tx_size,
            tx_type,
            lossless,
            eob,
            bd,
        );
        for (k, (&g, &e)) in got.iter().zip(&expected).enumerate() {
            assert_eq!(
                g.to_i32() as i64,
                e,
                "mismatch at ({}, {}): {tx_size:?} {tx_type:?} lossless={lossless} \
                 {dist:?} bd={bd}",
                k / stride,
                k % stride
            );
        }
    }

    fn run_all<T: Pixel>(bd: u32, seed: u64) {
        let mut rng = Rng(seed);
        let dists = [
            Dist::DcOnly,
            Dist::Sparse,
            Dist::Dense,
            Dist::Extreme,
            Dist::Zero,
        ];
        for &tx_size in TxSize::ALL {
            let w = 1usize << TX_WIDTH_LOG2[tx_size.idx()];
            let h = 1usize << TX_HEIGHT_LOG2[tx_size.idx()];
            let iters = if w * h >= 2048 { 2 } else { 6 };
            for tx_type in types_for(tx_size) {
                for &dist in &dists {
                    for _ in 0..iters {
                        run_case::<T>(&mut rng, tx_size, tx_type, false, dist, bd);
                    }
                }
            }
        }
        // Lossless (WHT); flips still apply per 7.12.3.
        for &tx_type in TxType::ALL {
            for &dist in &[Dist::DcOnly, Dist::Sparse, Dist::Dense, Dist::Zero] {
                for _ in 0..10 {
                    run_case::<T>(&mut rng, TxSize::Tx4x4, tx_type, true, dist, bd);
                }
            }
        }
    }

    #[test]
    fn matches_spec_8bit() {
        run_all::<u8>(8, 0x1234_5678_9abc_def1);
    }

    #[test]
    fn matches_spec_10bit() {
        run_all::<u16>(10, 0x0fed_cba9_8765_4321);
    }

    #[test]
    fn matches_spec_12bit() {
        run_all::<u16>(12, 0x5555_aaaa_3333_cccc);
    }

    #[test]
    fn matches_spec_8bit_in_u16() {
        run_all::<u16>(8, 0x7777_1111_2222_9999);
    }

    /// The zero-skipping kernel variants equal the dense kernels whenever
    /// the inputs they assume zero are zero (the dense kernels are checked
    /// against the spec by the 2D tests).
    #[test]
    fn nz_variants_match_dense() {
        let mut rng = Rng(99);
        for (kind, log2n) in [
            (Tx1d::Dct, 3),
            (Tx1d::Dct, 4),
            (Tx1d::Dct, 5),
            (Tx1d::Dct, 6),
            (Tx1d::Adst, 2),
            (Tx1d::Adst, 3),
            (Tx1d::Adst, 4),
        ] {
            let n = 1usize << log2n;
            for nz in 1..=n.min(32) {
                for iter in 0..200 {
                    let bits = [8, 12, 16, 20][iter % 4];
                    let r = Clamp::bits(bits + 2);
                    let mut a = [0i32; 64];
                    for v in &mut a[..nz] {
                        *v = rng.range(-(1 << (bits - 1)), (1 << (bits - 1)) - 1) as i32;
                    }
                    let mut b = a;
                    select_1d_full(kind, log2n)(&mut a, r);
                    select_1d(kind, log2n, nz)(&mut b, r);
                    assert_eq!(a[..n], b[..n], "{kind:?}{n} nz {nz}");
                }
            }
        }
    }

    /// The lane (column pass) kernels equal the scalar kernels for inputs
    /// within the column clamping range (up to 18 bits for 12-bit video),
    /// including values at the range limits.
    #[test]
    fn lane_kernels_match_scalar() {
        let mut rng = Rng(123);
        let kinds = [
            (Tx1d::Dct, 2),
            (Tx1d::Dct, 3),
            (Tx1d::Dct, 4),
            (Tx1d::Dct, 5),
            (Tx1d::Dct, 6),
            (Tx1d::Adst, 2),
            (Tx1d::Adst, 3),
            (Tx1d::Adst, 4),
            (Tx1d::Identity, 2),
            (Tx1d::Identity, 3),
            (Tx1d::Identity, 4),
            (Tx1d::Identity, 5),
        ];
        for (kind, log2n) in kinds {
            let n = 1usize << log2n;
            for nz in 1..=n.min(32) {
                for iter in 0..100 {
                    let bits = [16u32, 18][iter % 2];
                    let r = Clamp::bits(bits);
                    let mut lanes = [[0i32; 8]; 64];
                    for l in 0..8 {
                        for row in lanes[..nz].iter_mut() {
                            row[l] = match rng.range(0, 3) {
                                0 => r.lo,
                                1 => r.hi,
                                _ => rng.range(r.lo as i64, r.hi as i64) as i32,
                            };
                        }
                    }
                    let mut want = lanes;
                    for l in 0..8 {
                        let mut t = [0i32; 64];
                        for i in 0..n {
                            t[i] = want[i][l];
                        }
                        select_1d(kind, log2n, nz)(&mut t, r);
                        for i in 0..n {
                            want[i][l] = t[i];
                        }
                    }
                    select_1d_lanes::<8>(kind, log2n, nz).unwrap()(&mut lanes[..n], r);
                    assert_eq!(lanes[..n], want[..n], "{kind:?}{n} nz {nz}");
                }
            }
        }
    }

    /// Cross-check of the 1D kernels against vectors produced by dav1d's
    /// `itx_1d.c` (an independent implementation). Run with
    /// `ITX_DAV1D_VECTORS=<file> cargo test -- --ignored`. Each line is
    /// `kind log2n min max in[0..n] | out[0..n]` with kind 0 = DCT,
    /// 1 = ADST, 2 = identity.
    #[test]
    #[ignore]
    fn matches_dav1d_1d_vectors() {
        let path = std::env::var("ITX_DAV1D_VECTORS").expect("ITX_DAV1D_VECTORS not set");
        let text = std::fs::read_to_string(path).unwrap();
        let mut mismatches = 0;
        let mut total = 0;
        for line in text.lines() {
            let (lhs, rhs) = line.split_once('|').unwrap();
            let v: Vec<i64> = lhs.split_whitespace().map(|x| x.parse().unwrap()).collect();
            let out: Vec<i32> = rhs.split_whitespace().map(|x| x.parse().unwrap()).collect();
            let kind = [Tx1d::Dct, Tx1d::Adst, Tx1d::Identity][v[0] as usize];
            let log2n = v[1] as usize;
            let r = Clamp {
                lo: v[2] as i32,
                hi: v[3] as i32,
            };
            let n = 1 << log2n;
            let mut t = [0i32; 64];
            for i in 0..n {
                t[i] = v[4 + i] as i32;
            }
            select_1d_full(kind, log2n)(&mut t, r);
            total += 1;
            if t[..n] != out[..] {
                mismatches += 1;
                if mismatches < 5 {
                    eprintln!(
                        "mismatch {kind:?}{n} {:?}:\n got {:?}\nwant {:?}",
                        r,
                        &t[..n],
                        out
                    );
                }
            }
        }
        assert!(total > 0);
        assert_eq!(mismatches, 0, "{mismatches}/{total} mismatches");
    }

    /// Rough timing of the 2D path (`cargo test --release -- --ignored
    /// bench_itx --nocapture`).
    #[test]
    #[ignore]
    fn bench_itx() {
        let mut rng = Rng(7);
        let cases = [
            (TxSize::Tx4x4, TxType::DCT_DCT),
            (TxSize::Tx4x4, TxType::ADST_ADST),
            (TxSize::Tx8x8, TxType::DCT_DCT),
            (TxSize::Tx16x16, TxType::DCT_DCT),
            (TxSize::Tx16x16, TxType::ADST_ADST),
            (TxSize::Tx32x32, TxType::DCT_DCT),
            (TxSize::Tx64x64, TxType::DCT_DCT),
        ];
        let mut dst = vec![128u8; 64 * 64];
        for (name, f, n) in [
            ("idct4", kernels::idct4_nz4 as Tx1dFn, 4),
            ("idct16", kernels::idct16_nz16, 16),
            ("iadst16", kernels::iadst16_nz16, 16),
            ("idct64", kernels::idct64_nz32, 64),
        ] {
            let mut t = [0i32; 64];
            for v in &mut t[..n] {
                *v = rng.range(-1000, 1000) as i32;
            }
            let iters = 1_000_000;
            let start = std::time::Instant::now();
            for _ in 0..iters {
                f(std::hint::black_box(&mut t), Clamp::bits(16));
            }
            let ns = start.elapsed().as_nanos() as f64 / iters as f64;
            eprintln!("{name}: {ns:.1} ns");
        }
        for (tx_size, tx_type) in cases.into_iter().chain(cases) {
            let w = 1usize << TX_WIDTH_LOG2[tx_size.idx()];
            let h = 1usize << TX_HEIGHT_LOG2[tx_size.idx()];
            let (tw, th) = (w.min(32), h.min(32));
            let (orig, eob) = gen_coeffs(&mut rng, Dist::Dense, tw, th, 8);
            let mut c = orig.clone();
            let iters = 2_000_000 / (w * h);
            let start = std::time::Instant::now();
            for _ in 0..iters {
                c.copy_from_slice(&orig);
                inv_txfm_add(&mut dst, 64, &mut c, tx_size, tx_type, false, eob, 8);
            }
            let ns = start.elapsed().as_nanos() as f64 / iters as f64;
            eprintln!(
                "{tx_size:?} {tx_type:?}: {ns:.0} ns/block, {:.2} ns/pixel",
                ns / (w * h) as f64
            );
        }
    }

    /// Sanity check of the 1D DCT/ADST kernels against floating point
    /// (catches angle / index transcription errors common to both the
    /// implementation and the spec transcription above).
    #[test]
    fn dct_adst_close_to_float() {
        use std::f64::consts::PI;
        let mut rng = Rng(42);
        let big = Clamp::bits(30);
        for log2n in 2..=6usize {
            let n = 1usize << log2n;
            for kind in [Tx1d::Dct, Tx1d::Adst] {
                if kind == Tx1d::Adst && n > 16 {
                    continue;
                }
                let f = select_1d_full(kind, log2n);
                for _ in 0..50 {
                    let input: Vec<i32> = (0..n).map(|_| rng.range(-4096, 4096) as i32).collect();
                    let mut t = [0i32; 64];
                    t[..n].copy_from_slice(&input);
                    f(&mut t, big);
                    for (k, &got) in t[..n].iter().enumerate() {
                        let mut acc = 0f64;
                        for (i, &x) in input.iter().enumerate() {
                            let (i, k, nf) = (i as f64, k as f64, n as f64);
                            let basis = match kind {
                                // Unnormalised DCT-II inverse (DC weight
                                // 1/sqrt2), as used by AV1.
                                Tx1d::Dct => {
                                    let s = if i == 0.0 { 0.5f64.sqrt() } else { 1.0 };
                                    s * (PI * (2.0 * k + 1.0) * i / (2.0 * nf)).cos()
                                }
                                // AV1 ADST4 is a DST-VII scaled by
                                // 2*sqrt(2)/3; ADST8/16 are DST-IV.
                                _ if n == 4 => {
                                    let s = 2.0 * 2f64.sqrt() / 3.0;
                                    s * (PI * (2.0 * i + 1.0) * (k + 1.0) / 9.0).sin()
                                }
                                _ => (PI * (2.0 * i + 1.0) * (2.0 * k + 1.0) / (4.0 * nf)).sin(),
                            };
                            acc += x as f64 * basis;
                        }
                        let tol = 2.0 + n as f64 * 0.75;
                        assert!(
                            (acc - got as f64).abs() <= tol,
                            "{kind:?}{n}: out[{k}] = {got}, float {acc}"
                        );
                    }
                }
            }
        }
    }
}
