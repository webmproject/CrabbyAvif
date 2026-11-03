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

//! SIMD film grain kernels (see `dsp::film_grain`).
//!
//! Film grain is applied through `FilmGrain::apply_band`; its inner
//! kernels dispatch here.

use std::simd::prelude::*;

use crate::decoder::av1::dsp::film_grain::{
    self as portable, BlockNeighbourhood, BlockOffsets, ChromaNoiseParams, GrainTemplate,
    NoiseParams, BLOCK,
};
use crate::decoder::av1::pixel::Pixel;

// `std::simd` forms of NEON instruction sequences, written so that LLVM
// selects exactly those instructions on aarch64 (checked in the asm).
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::simd::prelude::*;

    /// Pairwise average with rounding of adjacent pairs from `a` and `b`:
    /// `[(a[2*i] + a[2*i+1] + 1) >> 1, ..., (b[2*i] + b[2*i+1] + 1) >> 1]`
    /// (`addp.8h` + `shadd.8h` with 1).
    /// Inputs must be in `0..=16383` so adjacent pairs do not overflow `i16`.
    #[inline(always)]
    pub(super) fn avg_pairs(a: i16x8, b: i16x8) -> i16x8 {
        let (even, odd) = a.deinterleave(b);
        // The rounding is done in `i32` (exact for every `i16` sum), which
        // LLVM lowers to `shadd(sum, 1)`.
        ((even + odd).cast::<i32>() + i32x8::splat(1) >> 1).cast::<i16>()
    }

    /// Multiply-accumulate two `i16x8` pairs `a0 * b0 + a1 * b1`,
    /// shifted right by 6 and narrowed to `i16x8`
    /// (`smull` + `smlal` + `smull2` + `smlal2` + `shrn #6` + `shrn2 #6`).
    #[inline(always)]
    pub(super) fn madd_shrn6(a0: i16x8, b0: i16x8, a1: i16x8, b1: i16x8) -> i16x8 {
        ((a0.cast::<i32>() * b0.cast::<i32>() + a1.cast::<i32>() * b1.cast::<i32>()) >> 6)
            .cast::<i16>()
    }
}

/// Returns `&grain[y][x..x + M]`.
///
/// # Safety invariant
/// All block offsets comes from `BlockOffsets::new`, where `x <= 39`, `y <= 39`,
/// `row < 34`, and `M <= 32` (with `x + N + 4 <= 75 <= 82` and `y + row <= 72 < 73`).
#[inline(always)]
fn grain_slice<const M: usize>(grain: &GrainTemplate, y: usize, x: usize) -> &[i16; M] {
    debug_assert!(y < grain.len() && x + M <= grain[0].len());
    grain[y][x..x + M].try_into().unwrap()
}

/// Asserts that all block offsets in `nb` satisfy `BlockOffsets::new`'s range
/// (`x < 40 && y < 40`) so that `grain_slice` bounds checks inside row loops
/// are proven in-bounds by LLVM.
#[inline(always)]
fn check_nb(nb: &BlockNeighbourhood) {
    assert!(nb.cur.x < 40 && nb.cur.y < 40);
    if let Some(l) = nb.left {
        assert!(l.x < 40 && l.y < 40);
    }
    if let Some(t) = nb.top {
        assert!(t.x < 40 && t.y < 40);
    }
    if let Some(tl) = nb.top_left {
        assert!(tl.x < 40 && tl.y < 40);
    }
}

/// Vertically blends `old` (weight `wo`) and `new` (weight `wn`) with rounding
/// right shift by 5 and clamps to `[GrainMin, GrainMax]`.
#[inline(always)]
fn vblend<const N: usize, const BD: u32>(
    old: Simd<i16, N>,
    new: Simd<i16, N>,
    wo: i32,
    wn: i32,
) -> Simd<i16, N> {
    let gmin = -(128i32 << (BD - 8));
    let gmax = (128i32 << (BD - 8)) - 1;
    if BD <= 10 {
        ((old * Simd::splat(wo as i16) + new * Simd::splat(wn as i16) + Simd::splat(16)) >> 5)
            .simd_max(Simd::splat(gmin as i16))
            .simd_min(Simd::splat(gmax as i16))
    } else {
        (((old.cast::<i32>() * Simd::splat(wo)
            + new.cast::<i32>() * Simd::splat(wn)
            + Simd::splat(16))
            >> 5)
            .simd_max(Simd::splat(gmin))
            .simd_min(Simd::splat(gmax)))
        .cast::<i16>()
    }
}

/// Row `row` of the noise stripe of a block of width `N` with offsets
/// `cur`, blended horizontally with its left neighbour `left`.
#[inline(always)]
fn stripe<const N: usize, const BD: u32>(
    grain: &GrainTemplate,
    cur: BlockOffsets,
    left: Option<BlockOffsets>,
    row: usize,
) -> Simd<i16, N> {
    let v = Simd::<i16, N>::from_slice(grain_slice::<N>(grain, cur.y + row, cur.x));
    let Some(l) = left else {
        return v;
    };
    let gmin = -(128i32 << (BD - 8));
    let gmax = (128i32 << (BD - 8)) - 1;
    let ow = if N == BLOCK { 2 } else { 1 };
    let old = Simd::<i16, 4>::from_slice(grain_slice::<4>(grain, l.y + row, l.x + N));
    let g = if BD <= 10 {
        let wo =
            Simd::<i16, 4>::from_array(if N == BLOCK { [27, 17, 0, 0] } else { [23, 0, 0, 0] });
        let wn =
            Simd::<i16, 4>::from_array(if N == BLOCK { [17, 27, 0, 0] } else { [22, 0, 0, 0] });
        ((old * wo + v.extract::<0, 4>() * wn + Simd::splat(16)) >> 5)
            .simd_max(Simd::splat(gmin as i16))
            .simd_min(Simd::splat(gmax as i16))
    } else {
        let wo =
            Simd::<i32, 4>::from_array(if N == BLOCK { [27, 17, 0, 0] } else { [23, 0, 0, 0] });
        let wn =
            Simd::<i32, 4>::from_array(if N == BLOCK { [17, 27, 0, 0] } else { [22, 0, 0, 0] });
        (((old.cast::<i32>() * wo + v.extract::<0, 4>().cast::<i32>() * wn + Simd::splat(16)) >> 5)
            .simd_max(Simd::splat(gmin))
            .simd_min(Simd::splat(gmax)))
        .cast::<i16>()
    };
    let head = Mask::<i16, N>::from_array(std::array::from_fn(|j| j < ow));
    head.select(g.resize::<N>(0), v)
}

/// Computes row `i` of a block of width `N`, including vertical overlap when
/// `i < oh` and `nb.top` is present.
#[inline(always)]
fn noise_row<const N: usize, const BD: u32>(
    grain: &GrainTemplate,
    nb: &BlockNeighbourhood,
    sy: bool,
    i: usize,
) -> Simd<i16, N> {
    let bh = BLOCK >> sy as usize;
    let oh = if sy { 1 } else { 2 };
    let v = stripe::<N, BD>(grain, nb.cur, nb.left, i);
    if let Some(top) = nb.top {
        if i < oh {
            let old = stripe::<N, BD>(grain, top, nb.top_left, i + bh);
            let (wo, wn) = match (oh, i) {
                (1, _) => (23, 22),
                (_, 0) => (27, 17),
                _ => (17, 27),
            };
            return vblend::<N, BD>(old, v, wo, wn);
        }
    }
    v
}

/// `dsp::film_grain::noise_block` with the rows of the block processed as
/// vectors of `32 >> sx` samples. Only used to test [`noise_row`].
#[cfg(test)]
pub(crate) fn noise_block_bd<const BD: u32>(
    grain: &GrainTemplate,
    nb: &BlockNeighbourhood,
    sx: bool,
    sy: bool,
    w: usize,
    h: usize,
    out: &mut [[i16; BLOCK]; BLOCK],
) {
    if sx {
        noise_block_n::<16, BD>(grain, nb, sy, w, h, out);
    } else {
        noise_block_n::<32, BD>(grain, nb, sy, w, h, out);
    }
}

/// [`noise_block_bd`] for a block width of `N` samples (`N == 32 >> sx`).
#[cfg(test)]
fn noise_block_n<const N: usize, const BD: u32>(
    grain: &GrainTemplate,
    nb: &BlockNeighbourhood,
    sy: bool,
    w: usize,
    h: usize,
    out: &mut [[i16; BLOCK]; BLOCK],
) {
    check_nb(nb);
    assert!(h <= BLOCK >> sy as usize);
    for (i, out_row) in out.iter_mut().enumerate().take(h) {
        let v = noise_row::<N, BD>(grain, nb, sy, i);
        if w == N {
            v.copy_to_slice(&mut out_row[..N]);
        } else {
            out_row[..w].copy_from_slice(&v.as_array()[..w]);
        }
    }
}

/// Number of samples processed per vector by the row kernels.
const LANES: usize = 16;

/// Loads `N` pixels from `src[..N]` as `i16` lanes (pixels have at most
/// 12 bits).
#[inline(always)]
fn load_i16<T: Pixel, const N: usize>(src: &[T]) -> Simd<i16, N> {
    debug_assert!(src.len() >= N);
    let src = &src[..N];
    if T::MAX_BITDEPTH == 8 {
        Simd::<u8, N>::from_array(std::array::from_fn(|i| src[i].to_i32() as u8)).cast::<i16>()
    } else {
        Simd::from_array(std::array::from_fn(|i| src[i].to_i32() as i16))
    }
}

/// Stores `v`, whose lanes are valid pixel values, to `dst[..N]`.
#[inline(always)]
fn store<T: Pixel, const N: usize>(dst: &mut [T], v: Simd<i16, N>) {
    debug_assert!(dst.len() >= N);
    let dst = &mut dst[..N];
    for (d, &x) in dst.iter_mut().zip(v.as_array()) {
        *d = T::from_i32(x as i32);
    }
}

/// Looks up 16 scaling values `lut[index & 4095]` for 16 lanes.
///
/// For `BD == 8`, all indices are in `0..=255`, so narrowing `index` to `u8`
/// packs all 16 indices into two 64-bit registers (or loads directly from the
/// source byte slice when `index` comes from `load_i16::<u8, 16>`) and needs
/// no `4095` mask.
#[inline(always)]
fn lookup_scale_16<const BD: u32>(index: Simd<i16, 16>, lut: &[u8; 4096]) -> Simd<u8, 16> {
    if BD == 8 {
        let [w0, w1] = Simd::<u64, 2>::from_ne_bytes(index.cast::<u8>().to_ne_bytes()).to_array();
        let lk = |w: u64, sh: u32| lut[((w >> sh) as u8) as usize];
        Simd::<u8, 16>::from_array(if cfg!(target_endian = "little") {
            [
                lk(w0, 0),
                lk(w0, 8),
                lk(w0, 16),
                lk(w0, 24),
                lk(w0, 32),
                lk(w0, 40),
                lk(w0, 48),
                lk(w0, 56),
                lk(w1, 0),
                lk(w1, 8),
                lk(w1, 16),
                lk(w1, 24),
                lk(w1, 32),
                lk(w1, 40),
                lk(w1, 48),
                lk(w1, 56),
            ]
        } else {
            [
                lk(w0, 56),
                lk(w0, 48),
                lk(w0, 40),
                lk(w0, 32),
                lk(w0, 24),
                lk(w0, 16),
                lk(w0, 8),
                lk(w0, 0),
                lk(w1, 56),
                lk(w1, 48),
                lk(w1, 40),
                lk(w1, 32),
                lk(w1, 24),
                lk(w1, 16),
                lk(w1, 8),
                lk(w1, 0),
            ]
        })
    } else {
        let [w0, w1, w2, w3] = Simd::<u64, 4>::from_ne_bytes(index.to_ne_bytes()).to_array();
        let lk = |w: u64, sh: u32| lut[((w >> sh) as usize) & 4095];
        Simd::<u8, 16>::from_array(if cfg!(target_endian = "little") {
            [
                lk(w0, 0),
                lk(w0, 16),
                lk(w0, 32),
                lk(w0, 48),
                lk(w1, 0),
                lk(w1, 16),
                lk(w1, 32),
                lk(w1, 48),
                lk(w2, 0),
                lk(w2, 16),
                lk(w2, 32),
                lk(w2, 48),
                lk(w3, 0),
                lk(w3, 16),
                lk(w3, 32),
                lk(w3, 48),
            ]
        } else {
            [
                lk(w0, 48),
                lk(w0, 32),
                lk(w0, 16),
                lk(w0, 0),
                lk(w1, 48),
                lk(w1, 32),
                lk(w1, 16),
                lk(w1, 0),
                lk(w2, 48),
                lk(w2, 32),
                lk(w2, 16),
                lk(w2, 0),
                lk(w3, 48),
                lk(w3, 32),
                lk(w3, 16),
                lk(w3, 0),
            ]
        })
    }
}

/// Noise application of 16 samples: `orig` plus `noise` scaled by
/// `lut[index & 4095]`, rounded and clipped to `[min, max]`.
///
/// Since `scale` is in `0..=255` and `scaling_shift` is in `8..=11`,
/// `scale * scale_mul` (where `scale_mul = 1 << (15 - scaling_shift)` is in
/// `16..=128`) is in `0..=32640 <= i16::MAX`, and
/// `(scale_shifted * noise + 2^14) >> 15 == round2(scale * noise, s)`
/// bit-exactly (NEON `sqrdmulh`, which LLVM does not select from
/// generic IR, would compute the same in a single instruction).
#[inline(always)]
fn apply_noise_16<const BD: u32>(
    orig: Simd<i16, 16>,
    index: Simd<i16, 16>,
    noise: Simd<i16, 16>,
    lut: &[u8; 4096],
    scale_mul: Simd<i16, 16>,
    min: Simd<i16, 16>,
    max: Simd<i16, 16>,
) -> Simd<i16, 16> {
    let scale_shifted = lookup_scale_16::<BD>(index, lut).cast::<i16>() * scale_mul;
    #[cfg(target_arch = "aarch64")]
    let v = {
        // `smull` + `smull2` + `rshrn #15` + `rshrn2 #15` per 8 lanes on aarch64.
        ((scale_shifted.cast::<i32>() * noise.cast::<i32>() + Simd::splat(1 << 14)) >> 15)
            .cast::<i16>()
    };
    #[cfg(not(target_arch = "aarch64"))]
    let v = {
        // `noise` is in `-2048..=2047` (even at 12-bit), so `b = noise << 1` fits
        // in `i16`, and `(scale_shifted * noise + 2^14) >> 15` equals
        // `(scale_shifted * b + 2^15) >> 16 == mulhi(scale_shifted, b) - (mullo(scale_shifted, b) >> 15)`
        // (`vpsllw` + `vpmulhw` + `vpmullw` + `vpsraw` + `vpsubw` in one `ymm`).
        let b = noise << 1;
        let hi = ((scale_shifted.cast::<i32>() * b.cast::<i32>()) >> 16).cast::<i16>();
        let lo = scale_shifted * b;
        hi - (lo >> 15)
    };
    (orig + v).simd_max(min).simd_min(max)
}

/// Applies luma noise vector `nz` to 32 pixels at `row[..32]`.
#[inline(always)]
fn apply_luma_32<T: Pixel, const BD: u32>(
    row: &mut [T],
    nz: Simd<i16, 32>,
    lut: &[u8; 4096],
    scale_mul: Simd<i16, 16>,
    min: Simd<i16, 16>,
    max: Simd<i16, 16>,
) {
    let orig0 = load_i16::<T, 16>(row);
    let orig1 = load_i16::<T, 16>(&row[16..]);
    let nz0 = nz.extract::<0, 16>();
    let nz1 = nz.extract::<16, 16>();
    store::<T, 16>(
        row,
        apply_noise_16::<BD>(orig0, orig0, nz0, lut, scale_mul, min, max),
    );
    store::<T, 16>(
        &mut row[16..],
        apply_noise_16::<BD>(orig1, orig1, nz1, lut, scale_mul, min, max),
    );
}

/// Fused noise generation + application for one luma block (`w x h` samples at
/// `(x0, y_rel)` in `luma`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_block_bd<T: Pixel, const BD: u32>(
    luma: &mut [T],
    stride: usize,
    grain: &GrainTemplate,
    nb: &BlockNeighbourhood,
    x0: usize,
    y_rel: usize,
    w: usize,
    h: usize,
    p: &NoiseParams,
) {
    if h == 0 || w == 0 {
        return;
    }
    check_nb(nb);
    assert!(h <= BLOCK);
    let stride = stride & 0xffff_ffff;
    let end = (y_rel + h - 1) * stride + x0 + w;
    let luma = &mut luma[..end];

    if w == BLOCK {
        let scale_mul = Simd::<i16, 16>::splat(1i16 << (15 - p.scaling_shift));
        let min = Simd::<i16, 16>::splat(p.min as i16);
        let max = Simd::<i16, 16>::splat(p.max as i16);
        let mut i = 0;
        if let Some(top) = nb.top {
            let top_rows = 2.min(h);
            while i < top_rows {
                let v = stripe::<BLOCK, BD>(grain, nb.cur, nb.left, i);
                let old = stripe::<BLOCK, BD>(grain, top, nb.top_left, i + BLOCK);
                let (wo, wn) = if i == 0 { (27, 17) } else { (17, 27) };
                let nz = vblend::<BLOCK, BD>(old, v, wo, wn);
                let start = (y_rel + i) * stride + x0;
                let row = &mut luma[start..start + BLOCK];
                apply_luma_32::<T, BD>(row, nz, p.lut, scale_mul, min, max);
                i += 1;
            }
        }
        while i < h {
            let nz = stripe::<BLOCK, BD>(grain, nb.cur, nb.left, i);
            let start = (y_rel + i) * stride + x0;
            let row = &mut luma[start..start + BLOCK];
            apply_luma_32::<T, BD>(row, nz, p.lut, scale_mul, min, max);
            i += 1;
        }
    } else {
        for i in 0..h {
            let nz = noise_row::<BLOCK, BD>(grain, nb, false, i);
            let start = (y_rel + i) * stride + x0;
            let row = &mut luma[start..start + w];
            luma_row_bd::<T, BD>(row, nz.as_array(), p);
        }
    }
}

/// Applies chroma noise to 16 samples at `row[..16]` using co-located
/// `luma[..16 << SX as usize]`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn apply_chroma_16<T: Pixel, const SX: bool, const BD: u32>(
    row: &mut [T],
    luma: &[T],
    nz: Simd<i16, 16>,
    csfl: bool,
    luma_mult: Simd<i16, 8>,
    mult: Simd<i16, 8>,
    offset: Simd<i16, 16>,
    pixel_max: Simd<i16, 16>,
    lut: &[u8; 4096],
    scale_mul: Simd<i16, 16>,
    min: Simd<i16, 16>,
    max: Simd<i16, 16>,
) {
    let avg = if SX {
        #[cfg(target_arch = "aarch64")]
        {
            let l0 = load_i16::<T, 8>(luma);
            let l1 = load_i16::<T, 8>(&luma[8..]);
            let l2 = load_i16::<T, 8>(&luma[16..]);
            let l3 = load_i16::<T, 8>(&luma[24..]);
            let lo = neon::avg_pairs(l0, l1);
            let hi = neon::avg_pairs(l2, l3);
            simd_swizzle!(
                lo,
                hi,
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            )
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let (even, odd) = load_i16::<T, 16>(luma).deinterleave(load_i16::<T, 16>(&luma[16..]));
            (even + odd + Simd::splat(1)) >> 1
        }
    } else {
        load_i16::<T, 16>(luma)
    };
    let orig = load_i16::<T, 16>(row);
    let merged = if csfl {
        avg
    } else {
        #[cfg(target_arch = "aarch64")]
        let combined = {
            let lo = neon::madd_shrn6(
                avg.extract::<0, 8>(),
                luma_mult,
                orig.extract::<0, 8>(),
                mult,
            );
            let hi = neon::madd_shrn6(
                avg.extract::<8, 8>(),
                luma_mult,
                orig.extract::<8, 8>(),
                mult,
            );
            simd_swizzle!(
                lo,
                hi,
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            )
        };
        #[cfg(not(target_arch = "aarch64"))]
        let combined = {
            const INTERLEAVE: [usize; 32] = {
                let mut t = [0; 32];
                let mut i = 0;
                while i < 32 {
                    t[i] = i / 2 + (i % 2) * 16;
                    i += 1;
                }
                t
            };
            const INTERLEAVE_P: [usize; 32] = {
                const Q: [usize; 4] = [0, 2, 1, 3];
                let mut t = [0; 32];
                let mut q = 0;
                while q < 4 {
                    let src_q = Q[q];
                    let mut k = 0;
                    while k < 4 {
                        t[q * 8 + 2 * k] = src_q * 4 + k;
                        t[q * 8 + 2 * k + 1] = src_q * 4 + k + 16;
                        k += 1;
                    }
                    q += 1;
                }
                t
            };
            const EVEN: [usize; 16] = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30];
            const ODD: [usize; 16] = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31];
            let lm = simd_swizzle!(
                luma_mult,
                luma_mult,
                [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]
            );
            let m = simd_swizzle!(mult, mult, [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]);
            let x: Simd<i16, 32> = simd_swizzle!(avg, orig, INTERLEAVE_P);
            let f: Simd<i16, 32> = simd_swizzle!(lm, m, INTERLEAVE);
            let p = x.cast::<i32>() * f.cast::<i32>();
            let even: Simd<i32, 16> = simd_swizzle!(p, EVEN);
            let odd: Simd<i32, 16> = simd_swizzle!(p, ODD);
            let sum = (even + odd) >> 6;
            let n = sum
                .simd_clamp(Simd::splat(i16::MIN as i32), Simd::splat(i16::MAX as i32))
                .cast::<i16>();
            let q = Simd::<u64, 4>::from_ne_bytes(n.to_ne_bytes());
            Simd::<i16, 16>::from_ne_bytes(simd_swizzle!(q, [0, 2, 1, 3]).to_ne_bytes())
        };
        (combined + offset)
            .simd_max(Simd::splat(0))
            .simd_min(pixel_max)
    };
    store::<T, 16>(
        row,
        apply_noise_16::<BD>(orig, merged, nz, lut, scale_mul, min, max),
    );
}

/// Fused noise generation + application for one chroma block (`w x h` samples).
#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma_block_bd<T: Pixel, const BD: u32>(
    plane: &mut [T],
    stride: usize,
    luma: &[T],
    luma_stride: usize,
    grain: &GrainTemplate,
    nb: &BlockNeighbourhood,
    sy: bool,
    x0: usize,
    y_rel: usize,
    luma_y_rel: usize,
    frame_w: usize,
    w: usize,
    h: usize,
    p: &ChromaNoiseParams,
) {
    if p.sx {
        chroma_block_n::<T, 16, true, BD>(
            plane,
            stride,
            luma,
            luma_stride,
            grain,
            nb,
            sy,
            x0,
            y_rel,
            luma_y_rel,
            frame_w,
            w,
            h,
            p,
        );
    } else {
        chroma_block_n::<T, 32, false, BD>(
            plane,
            stride,
            luma,
            luma_stride,
            grain,
            nb,
            sy,
            x0,
            y_rel,
            luma_y_rel,
            frame_w,
            w,
            h,
            p,
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn chroma_block_n<T: Pixel, const N: usize, const SX: bool, const BD: u32>(
    plane: &mut [T],
    stride: usize,
    luma: &[T],
    luma_stride: usize,
    grain: &GrainTemplate,
    nb: &BlockNeighbourhood,
    sy: bool,
    x0: usize,
    y_rel: usize,
    luma_y_rel: usize,
    frame_w: usize,
    w: usize,
    h: usize,
    p: &ChromaNoiseParams,
) {
    if h == 0 || w == 0 {
        return;
    }
    check_nb(nb);
    assert!(h <= BLOCK >> sy as usize);
    let stride = stride & 0xffff_ffff;
    let luma_stride = luma_stride & 0xffff_ffff;
    let sx_shift = SX as usize;
    let sy_shift = sy as usize;
    let lx0 = x0 << sx_shift;
    let plane_end = (y_rel + h - 1) * stride + x0 + w;
    let plane = &mut plane[..plane_end];
    let luma_end = (luma_y_rel + ((h - 1) << sy_shift)) * luma_stride + frame_w;
    let luma = &luma[..luma_end];

    if w == N && lx0 + (N << sx_shift) <= frame_w {
        let scale_mul = Simd::<i16, 16>::splat(1i16 << (15 - p.noise.scaling_shift));
        let min = Simd::<i16, 16>::splat(p.noise.min as i16);
        let max = Simd::<i16, 16>::splat(p.noise.max as i16);
        let luma_mult = Simd::<i16, 8>::splat(p.cp.luma_mult as i16);
        let mult = Simd::<i16, 8>::splat(p.cp.mult as i16);
        let offset = Simd::<i16, 16>::splat(p.cp.offset as i16);
        let pixel_max = Simd::<i16, 16>::splat(p.pixel_max as i16);
        let csfl = p.csfl;
        let lut = p.noise.lut;

        let apply_row = |row: &mut [T], luma_row: &[T], nz: Simd<i16, N>| {
            apply_chroma_16::<T, SX, BD>(
                row,
                luma_row,
                nz.extract::<0, 16>(),
                csfl,
                luma_mult,
                mult,
                offset,
                pixel_max,
                lut,
                scale_mul,
                min,
                max,
            );
            if N == 32 {
                apply_chroma_16::<T, SX, BD>(
                    &mut row[16..],
                    &luma_row[16 << sx_shift..],
                    Simd::<i16, 16>::from_slice(&nz.as_array()[16..]),
                    csfl,
                    luma_mult,
                    mult,
                    offset,
                    pixel_max,
                    lut,
                    scale_mul,
                    min,
                    max,
                );
            }
        };

        let bh = BLOCK >> sy_shift;
        let oh = if sy { 1 } else { 2 };
        let mut i = 0;
        if let Some(top) = nb.top {
            let top_rows = oh.min(h);
            while i < top_rows {
                let v = stripe::<N, BD>(grain, nb.cur, nb.left, i);
                let old = stripe::<N, BD>(grain, top, nb.top_left, i + bh);
                let (wo, wn) = match (oh, i) {
                    (1, _) => (23, 22),
                    (_, 0) => (27, 17),
                    _ => (17, 27),
                };
                let nz = vblend::<N, BD>(old, v, wo, wn);
                let start = (y_rel + i) * stride + x0;
                let row = &mut plane[start..start + N];
                let lstart = (luma_y_rel + (i << sy_shift)) * luma_stride + lx0;
                let luma_row = &luma[lstart..lstart + (N << sx_shift)];
                apply_row(row, luma_row, nz);
                i += 1;
            }
        }
        while i < h {
            let nz = stripe::<N, BD>(grain, nb.cur, nb.left, i);
            let start = (y_rel + i) * stride + x0;
            let row = &mut plane[start..start + N];
            let lstart = (luma_y_rel + (i << sy_shift)) * luma_stride + lx0;
            let luma_row = &luma[lstart..lstart + (N << sx_shift)];
            apply_row(row, luma_row, nz);
            i += 1;
        }
    } else {
        for i in 0..h {
            let nz = noise_row::<N, BD>(grain, nb, sy, i);
            let start = (y_rel + i) * stride + x0;
            let row = &mut plane[start..start + w];
            let lstart = (luma_y_rel + (i << sy_shift)) * luma_stride;
            let luma_row = &luma[lstart + lx0..lstart + frame_w];
            chroma_row_bd::<T, BD>(row, nz.as_array(), luma_row, p);
        }
    }
}

/// `dsp::film_grain::luma_row`, `LANES` samples at a time.
pub(crate) fn luma_row_bd<T: Pixel, const BD: u32>(row: &mut [T], noise: &[i16], p: &NoiseParams) {
    let n = row.len().min(noise.len()) / LANES * LANES;
    if n > 0 {
        let scale_mul = Simd::<i16, 16>::splat(1i16 << (15 - p.scaling_shift));
        let min = Simd::<i16, 16>::splat(p.min as i16);
        let max = Simd::<i16, 16>::splat(p.max as i16);
        for (r, nz) in row[..n]
            .chunks_exact_mut(LANES)
            .zip(noise.chunks_exact(LANES))
        {
            let orig = load_i16::<T, LANES>(r);
            let nz = Simd::<i16, LANES>::from_slice(nz);
            store::<T, LANES>(
                r,
                apply_noise_16::<BD>(orig, orig, nz, p.lut, scale_mul, min, max),
            );
        }
    }
    if n < row.len() {
        portable::luma_row(&mut row[n..], &noise[n..], p);
    }
}

/// `dsp::film_grain::chroma_row`, `LANES` samples at a time.
pub(crate) fn chroma_row_bd<T: Pixel, const BD: u32>(
    row: &mut [T],
    noise: &[i16],
    luma: &[T],
    p: &ChromaNoiseParams,
) {
    let sx = p.sx as usize;
    // Rows at the right edge of an odd width frame (whose last
    // averageLuma uses a single luma sample) use the portable kernel.
    let n = if (row.len() << sx) <= luma.len() {
        row.len().min(noise.len()) / LANES * LANES
    } else {
        0
    };
    if n > 0 {
        let scale_mul = Simd::<i16, 16>::splat(1i16 << (15 - p.noise.scaling_shift));
        let min = Simd::<i16, 16>::splat(p.noise.min as i16);
        let max = Simd::<i16, 16>::splat(p.noise.max as i16);
        let luma_mult = Simd::<i16, 8>::splat(p.cp.luma_mult as i16);
        let mult = Simd::<i16, 8>::splat(p.cp.mult as i16);
        let offset = Simd::<i16, 16>::splat(p.cp.offset as i16);
        let pixel_max = Simd::<i16, 16>::splat(p.pixel_max as i16);
        let chunks = row[..n]
            .chunks_exact_mut(LANES)
            .zip(noise.chunks_exact(LANES));
        for (k, (r, nz)) in chunks.enumerate() {
            let l = &luma[(k * LANES) << sx..];
            let nz = Simd::<i16, LANES>::from_slice(nz);
            if p.sx {
                apply_chroma_16::<T, true, BD>(
                    r,
                    l,
                    nz,
                    p.csfl,
                    luma_mult,
                    mult,
                    offset,
                    pixel_max,
                    p.noise.lut,
                    scale_mul,
                    min,
                    max,
                );
            } else {
                apply_chroma_16::<T, false, BD>(
                    r,
                    l,
                    nz,
                    p.csfl,
                    luma_mult,
                    mult,
                    offset,
                    pixel_max,
                    p.noise.lut,
                    scale_mul,
                    min,
                    max,
                );
            }
        }
    }
    if n < row.len() {
        portable::chroma_row(&mut row[n..], &noise[n..], &luma[n << sx..], p);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::av1::dsp::film_grain::ChromaParams;

    /// xorshift64* test generator.
    struct TestRng(u64);

    impl TestRng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        fn below(&mut self, n: u64) -> u64 {
            (self.next() >> 11) % n
        }

        fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + self.below((hi - lo + 1) as u64) as i32
        }

        fn bool(&mut self) -> bool {
            self.next() >> 63 != 0
        }
    }

    /// Grain range `(GrainMin, GrainMax)` of bit depth `bd`.
    fn grain_range(bd: u32) -> (i32, i32) {
        let center = 128 << (bd - 8);
        (-center, (256 << (bd - 8)) - 1 - center)
    }

    #[test]
    fn noise_block_matches_portable() {
        let mut r = TestRng(0x5eed_f11e_9a17_0001);
        for bd in [8, 10, 12] {
            let (gmin, gmax) = grain_range(bd);
            for _ in 0..40 {
                let mut grain = Box::new([[0i16; 82]; 73]);
                for v in grain.iter_mut().flatten() {
                    *v = r.range(gmin, gmax) as i16;
                }
                for (sx, sy) in [(false, false), (true, false), (false, true), (true, true)] {
                    let (bw, bh) = (BLOCK >> sx as usize, BLOCK >> sy as usize);
                    for _ in 0..20 {
                        let mut o = || BlockOffsets::new(r.range(0, 255), sx, sy);
                        let cur = o();
                        let (left, top, top_left) = (o(), o(), o());
                        let (has_left, has_top) = (r.bool(), r.bool());
                        let nb = BlockNeighbourhood {
                            cur,
                            left: has_left.then_some(left),
                            top: has_top.then_some(top),
                            top_left: (has_left && has_top).then_some(top_left),
                        };
                        let w = if r.bool() { bw } else { r.range(1, bw as i32) as usize };
                        let h = if r.bool() { bh } else { r.range(1, bh as i32) as usize };
                        let mut want = [[0i16; BLOCK]; BLOCK];
                        let mut got = [[0i16; BLOCK]; BLOCK];
                        portable::noise_block(&grain, &nb, sx, sy, w, h, gmin, gmax, &mut want);
                        if bd == 8 {
                            noise_block_bd::<8>(&grain, &nb, sx, sy, w, h, &mut got);
                        } else if bd == 10 {
                            noise_block_bd::<10>(&grain, &nb, sx, sy, w, h, &mut got);
                        } else {
                            noise_block_bd::<12>(&grain, &nb, sx, sy, w, h, &mut got);
                        }
                        assert_eq!(got, want, "bd {bd} sub {sx},{sy} {w}x{h} {nb:?}");
                    }
                }
            }
        }
    }

    /// Random scaling LUT, as built by `expand_scaling_lut` (entries past
    /// `1 << bd` are never read by valid samples but are random too).
    fn random_lut(r: &mut TestRng) -> Box<[u8; 4096]> {
        let mut lut = Box::new([0u8; 4096]);
        for v in lut.iter_mut() {
            *v = r.below(256) as u8;
        }
        lut
    }

    /// Random `(minValue, maxLuma or maxChroma)`, full or restricted.
    fn random_clip(r: &mut TestRng, bd: u32) -> (i32, i32) {
        if r.bool() {
            (0, (1 << bd) - 1)
        } else {
            (
                16 << (bd - 8),
                (if r.bool() { 235 } else { 240 }) << (bd - 8),
            )
        }
    }

    fn random_pixels<T: Pixel>(r: &mut TestRng, n: usize, bd: u32) -> Vec<T> {
        (0..n)
            .map(|_| T::from_i32(r.range(0, (1 << bd) - 1)))
            .collect()
    }

    fn random_noise(r: &mut TestRng, bd: u32) -> [i16; BLOCK] {
        let (gmin, gmax) = grain_range(bd);
        std::array::from_fn(|_| r.range(gmin, gmax) as i16)
    }

    fn check_luma_row<T: Pixel>(r: &mut TestRng, bd: u32) {
        for _ in 0..300 {
            let lut = random_lut(r);
            let (min, max) = random_clip(r, bd);
            let p = NoiseParams {
                lut: &lut,
                scaling_shift: r.range(8, 11) as u32,
                min,
                max,
            };
            for w in 1..=BLOCK {
                let noise = random_noise(r, bd);
                let row: Vec<T> = random_pixels(r, w, bd);
                let mut want = row.clone();
                let mut got = row.clone();
                portable::luma_row(&mut want, &noise, &p);
                if bd == 8 {
                    luma_row_bd::<T, 8>(&mut got, &noise, &p);
                } else if bd == 10 {
                    luma_row_bd::<T, 10>(&mut got, &noise, &p);
                } else {
                    luma_row_bd::<T, 12>(&mut got, &noise, &p);
                }
                assert_eq!(got, want, "bd {bd} w {w}");
            }
        }
    }

    #[test]
    fn luma_row_matches_portable() {
        let mut r = TestRng(0x5eed_f11e_9a17_0002);
        check_luma_row::<u8>(&mut r, 8);
        for bd in [8, 10, 12] {
            check_luma_row::<u16>(&mut r, bd);
        }
    }

    fn check_chroma_row<T: Pixel>(r: &mut TestRng, bd: u32) {
        for _ in 0..300 {
            let lut = random_lut(r);
            let (min, max) = random_clip(r, bd);
            let sx = r.bool();
            let p = ChromaNoiseParams {
                noise: NoiseParams {
                    lut: &lut,
                    scaling_shift: r.range(8, 11) as u32,
                    min,
                    max,
                },
                csfl: r.below(4) == 0,
                cp: ChromaParams {
                    mult: r.range(-128, 127),
                    luma_mult: r.range(-128, 127),
                    offset: (r.range(0, 511) - 256) << (bd - 8),
                },
                pixel_max: (1 << bd) - 1,
                sx,
            };
            for w in 1..=BLOCK >> sx as usize {
                let noise = random_noise(r, bd);
                let row: Vec<T> = random_pixels(r, w, bd);
                // Luma up to the right frame edge: exactly the co-located
                // samples, one less (odd width) or more.
                let luma_w = match r.below(3) {
                    0 => w << sx as usize,
                    1 if sx => 2 * w - 1,
                    _ => (w << sx as usize) + r.range(1, 40) as usize,
                };
                let luma: Vec<T> = random_pixels(r, luma_w, bd);
                let mut want = row.clone();
                let mut got = row.clone();
                portable::chroma_row(&mut want, &noise, &luma, &p);
                if bd == 8 {
                    chroma_row_bd::<T, 8>(&mut got, &noise, &luma, &p);
                } else if bd == 10 {
                    chroma_row_bd::<T, 10>(&mut got, &noise, &luma, &p);
                } else {
                    chroma_row_bd::<T, 12>(&mut got, &noise, &luma, &p);
                }
                assert_eq!(
                    got, want,
                    "bd {bd} w {w} sx {sx} luma_w {luma_w} csfl {}",
                    p.csfl
                );
            }
        }
    }

    #[test]
    fn chroma_row_matches_portable() {
        let mut r = TestRng(0x5eed_f11e_9a17_0003);
        check_chroma_row::<u8>(&mut r, 8);
        for bd in [8, 10, 12] {
            check_chroma_row::<u16>(&mut r, bd);
        }
    }
}
