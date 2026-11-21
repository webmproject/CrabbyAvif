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

//! Symbol (arithmetic) decoder, spec 8.2.
//!
//! CDF arrays use the spec layout: `N` cumulative values increasing to
//! 32768 followed by an adaptation counter, i.e. `cdf.len() == N + 1`.
//! The implementation keeps a 64-bit window of (inverted) bitstream bits
//! like dav1d's msac, which is equivalent to the bit-serial spec process
//! (including the zero padding past the end of the tile data).

const EC_PROB_SHIFT: u32 = 6;
const EC_MIN_PROB: u32 = 4;
const WIN_SIZE: i32 = 64;

#[derive(Clone)]
struct ByteReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

#[cold]
#[inline(never)]
fn refill_slow(buf: &[u8], mut pos: usize, mut dif: u64, mut c: i32) -> u64 {
    loop {
        let b = if pos < buf.len() { buf[pos] ^ 0xff } else { 0xff };
        dif |= (b as u64) << c;
        pos += 1;
        c = c.wrapping_sub(8);
        if c < 0 {
            return dif;
        }
    }
}

// The cold out-of-line helpers below take an unused `const DUMMY: usize`
// generic, always instantiated as `::<0>`. It has no semantic effect: it
// only makes the function generic, so it is monomorphized in the codegen
// unit of its caller instead of once in this module's, which places the
// cold code next to its (hot) caller and measurably changed the generated
// code. Keep it unless benchmarks show otherwise.
#[cold]
#[inline(never)]
fn refill_cold<const DUMMY: usize>(buf: &[u8], pos: usize, mut dif: u64, cnt: i32) -> (u64, u64) {
    debug_assert!((-16..=40).contains(&cnt));
    let nb = (47i32.wrapping_sub(cnt) >> 3) as usize;
    if let Some(bytes) = buf.get(pos..).and_then(|s| s.first_chunk::<8>()) {
        let w = !u64::from_be_bytes(*bytes);
        dif |= w >> (16i32.wrapping_add(cnt) as u32 & 63);
    } else {
        dif = refill_slow(buf, pos, dif, 40i32.wrapping_sub(cnt));
    }
    let new_cnt = ((cnt & 7) | 40) as u32;
    (dif, (nb as u64) | ((new_cnt as u64) << 32))
}

#[inline(always)]
fn refill_buf(buf: &[u8], cur_pos: &mut usize, dif: &mut u64, cnt: &mut i32) {
    let c = *cnt;
    debug_assert!((-16..=40).contains(&c));
    let nb = (47i32.wrapping_sub(c) >> 3) as usize;
    if let Some(bytes) = buf.get(*cur_pos..).and_then(|s| s.first_chunk::<8>()) {
        let w = !u64::from_be_bytes(*bytes);
        *dif |= w >> (16i32.wrapping_add(c) as u32 & 63);
    } else {
        *dif = refill_slow(buf, *cur_pos, *dif, 40i32.wrapping_sub(c));
    }
    *cur_pos = (*cur_pos).wrapping_add(nb);
    *cnt = (c & 7) | 40;
}

/// Reads one equiprobable bit (`read_bool`) without a refill check.
/// Across any `K >= 1` consecutive equiprobable bits, total bits shifted out of
/// `dif` is at most `K + 1` (`s1 == 2` only for `rng in 0x8000..=0x8003` with
/// `ret == true`, which sets `rng >= 0xfff0` so `rng` must pass through an
/// `s1 == 0` step at `rng >= 0xff84` before `s1 == 2` can occur again).
#[inline(always)]
fn equi_nobuf(dif: &mut u64, rng: &mut u32, cnt: &mut i32) -> bool {
    use std::hint::select_unpredictable as sel;
    let r = *rng;
    let d = *dif;
    let v = ((r & 0xff00) >> 1) + EC_MIN_PROB;
    let vw = (v as u64) << (WIN_SIZE - 16);
    let ret = d >= vw;
    let s1 = ((r < 0xff84) as u32) + ((r < 0x8004) as u32);
    let sh = sel(ret, s1, 1);
    *dif = sel(ret, d.wrapping_sub(vw), d) << sh;
    *rng = sel(ret, r.wrapping_sub(v), v) << sh;
    *cnt = cnt.wrapping_sub(sh as i32);
    !ret
}

#[cold]
#[inline(never)]
fn golomb_coef_very_slow(
    buf: &[u8],
    pos: usize,
    mut cur_pos: usize,
    mut dif: u64,
    mut rng: u32,
    mut cnt: i32,
) -> (u64, u64) {
    let mut length = 8u32;
    loop {
        if cnt < 2 {
            refill_buf(buf, &mut cur_pos, &mut dif, &mut cnt);
        }
        length += 1;
        let v = ((rng & 0xff00) >> 1) + EC_MIN_PROB;
        let vw = (v as u64) << (WIN_SIZE - 16);
        if dif < vw {
            dif <<= 1;
            rng = v << 1;
            cnt = cnt.wrapping_sub(1);
            break;
        }
        let s1 = ((rng < 0xff84) as u32) + ((rng < 0x8004) as u32);
        dif = dif.wrapping_sub(vw) << s1;
        rng = rng.wrapping_sub(v) << s1;
        cnt = cnt.wrapping_sub(s1 as i32);
        if length > 32 {
            break;
        }
    }
    let n_suf = length - 1;
    if cnt <= n_suf as i32 {
        refill_buf(buf, &mut cur_pos, &mut dif, &mut cnt);
    }
    let mut x = 1u32;
    for _ in 0..n_suf {
        x = (x << 1) | equi_nobuf(&mut dif, &mut rng, &mut cnt) as u32;
    }
    let out = (rng as u64)
        | (((cnt as u8) as u64) << 16)
        | ((cur_pos.wrapping_sub(pos) as u64) << 24)
        | ((x as u64) << 32);
    (dif, out)
}

#[cold]
#[inline(never)]
fn golomb_coef_eof(buf: &[u8], pos: usize, mut dif: u64, rng: u32, mut cnt: i32) -> (u64, u64) {
    let mut cur_pos = pos;
    refill_buf(buf, &mut cur_pos, &mut dif, &mut cnt);
    let (dif, out) = golomb_coef_slow::<0>(buf, cur_pos, dif, rng, cnt);
    let total_nb = cur_pos.wrapping_sub(pos) + (((out >> 24) & 0xff) as usize);
    let out = (out & !0xff00_0000u64) | (((total_nb as u8) as u64) << 24);
    (dif, out)
}

#[inline(never)]
fn golomb_coef_slow<const DUMMY: usize>(
    buf: &[u8],
    pos: usize,
    mut dif: u64,
    mut rng: u32,
    mut cnt: i32,
) -> (u64, u64) {
    let mut cur_pos = pos;
    // Ensure `cnt >= 24` once upfront so up to 8 prefix bits + terminating bit
    // (`<= 9` bits shifted out) still leaves `cnt >= 15` (so `dif >> 33` reads
    // only valid bits even near EOF), with EOF tail-called so the common path
    // stays frameless.
    if cnt < 24 {
        cold_refill();
        if let Some(bytes) = buf.get(pos..).and_then(|s| s.first_chunk::<8>()) {
            let nb = (47i32.wrapping_sub(cnt) >> 3) as usize;
            let w = !u64::from_be_bytes(*bytes);
            dif |= w >> (16i32.wrapping_add(cnt) as u32 & 63);
            cur_pos = pos.wrapping_add(nb);
            cnt = (cnt & 7) | 40;
        } else {
            return golomb_coef_eof(buf, pos, dif, rng, cnt);
        }
    }
    let mut length = 0u32;
    loop {
        length += 1;
        let v = ((rng & 0xff00) >> 1) + EC_MIN_PROB;
        let vw = (v as u64) << (WIN_SIZE - 16);
        if dif < vw {
            dif <<= 1;
            rng = v << 1;
            cnt = cnt.wrapping_sub(1);
            let rem = length - 1;
            let mut x = 1u32;
            if rem != 0 {
                let recip = EQUI_RECIP[((rng >> 8) & 127) as usize] as u64;
                let t = ((((dif >> 33).wrapping_mul(recip)) >> 46) as u32) >> (16 - rem);
                x = (1u32 << rem) | ((!t) & ((1u32 << rem) - 1));
                dif = (dif << rem).wrapping_sub((t as u64).wrapping_mul((rng as u64) << 48));
                cnt = cnt.wrapping_sub(rem as i32);
            }
            debug_assert!(cnt >= 0);
            let out = (rng as u64)
                | (((cnt as u8) as u64) << 16)
                | ((cur_pos.wrapping_sub(pos) as u64) << 24)
                | ((x as u64) << 32);
            return (dif, out);
        }
        let s1 = ((rng < 0xff84) as u32) + ((rng < 0x8004) as u32);
        dif = dif.wrapping_sub(vw) << s1;
        rng = rng.wrapping_sub(v) << s1;
        cnt = cnt.wrapping_sub(s1 as i32);
        if length == 8 {
            return golomb_coef_very_slow(buf, pos, cur_pos, dif, rng, cnt);
        }
    }
}

#[cold]
#[inline(never)]
fn sign_golomb_very_slow(
    buf: &[u8],
    pos: usize,
    mut cur_pos: usize,
    mut dif: u64,
    mut rng: u32,
    mut cnt: i32,
    sign_hi: u32,
) -> (u64, u64) {
    let mut length = 8u32;
    loop {
        if cnt < 2 {
            refill_buf(buf, &mut cur_pos, &mut dif, &mut cnt);
        }
        length += 1;
        let v = ((rng & 0xff00) >> 1) + EC_MIN_PROB;
        let vw = (v as u64) << (WIN_SIZE - 16);
        if dif < vw {
            dif <<= 1;
            rng = v << 1;
            cnt = cnt.wrapping_sub(1);
            break;
        }
        let s1 = ((rng < 0xff84) as u32) + ((rng < 0x8004) as u32);
        dif = dif.wrapping_sub(vw) << s1;
        rng = rng.wrapping_sub(v) << s1;
        cnt = cnt.wrapping_sub(s1 as i32);
        if length > 32 {
            break;
        }
    }
    let n_suf = length - 1;
    if cnt <= n_suf as i32 {
        refill_buf(buf, &mut cur_pos, &mut dif, &mut cnt);
    }
    let mut x = 1u32;
    for _ in 0..n_suf {
        x = (x << 1) | equi_nobuf(&mut dif, &mut rng, &mut cnt) as u32;
    }
    let q = x.wrapping_add(14) & 0xFFFFF;
    let hi = q | sign_hi;
    let out = (rng as u64)
        | (((cnt as u8) as u64) << 16)
        | ((cur_pos.wrapping_sub(pos) as u64) << 24)
        | ((hi as u64) << 32);
    (dif, out)
}

#[cold]
#[inline(never)]
fn sign_golomb_eof(buf: &[u8], pos: usize, mut dif: u64, rng: u32, mut cnt: i32) -> (u64, u64) {
    let mut cur_pos = pos;
    refill_buf(buf, &mut cur_pos, &mut dif, &mut cnt);
    let (dif, out) = sign_golomb_slow::<0>(buf, cur_pos, dif, rng, cnt);
    let total_nb = cur_pos.wrapping_sub(pos) + (((out >> 24) & 0xff) as usize);
    let out = (out & !0xff00_0000u64) | (((total_nb as u8) as u64) << 24);
    (dif, out)
}

#[cold]
#[inline(never)]
fn sign_golomb_slow<const DUMMY: usize>(
    buf: &[u8],
    pos: usize,
    mut dif: u64,
    mut rng: u32,
    mut cnt: i32,
) -> (u64, u64) {
    use std::hint::select_unpredictable as sel;
    let mut cur_pos = pos;
    // Ensure `cnt >= 25` once upfront so 1 sign bit + up to 8 prefix bits
    // (`<= 10` bits shifted out) still leaves `cnt >= 15` (so `dif >> 33`
    // reads only valid bits even near EOF), with EOF tail-called so the common
    // path compiles as a zero-frame leaf function.
    if cnt < 25 {
        cold_refill();
        if let Some(bytes) = buf.get(pos..).and_then(|s| s.first_chunk::<8>()) {
            let nb = (47i32.wrapping_sub(cnt) >> 3) as usize;
            let w = !u64::from_be_bytes(*bytes);
            dif |= w >> (16i32.wrapping_add(cnt) as u32 & 63);
            cur_pos = pos.wrapping_add(nb);
            cnt = (cnt & 7) | 40;
        } else {
            return sign_golomb_eof(buf, pos, dif, rng, cnt);
        }
    }
    let sign = equi_nobuf(&mut dif, &mut rng, &mut cnt);
    let sign_hi = sel(sign, 0xc000_0000u32, 0x8000_0000u32);
    let mut length = 0u32;
    loop {
        length += 1;
        let v = ((rng & 0xff00) >> 1) + EC_MIN_PROB;
        let vw = (v as u64) << (WIN_SIZE - 16);
        if dif < vw {
            dif <<= 1;
            rng = v << 1;
            cnt = cnt.wrapping_sub(1);
            let rem = length - 1;
            let mut x = 1u32;
            if rem != 0 {
                let recip = EQUI_RECIP[((rng >> 8) & 127) as usize] as u64;
                let t = ((((dif >> 33).wrapping_mul(recip)) >> 46) as u32) >> (16 - rem);
                x = (1u32 << rem) | ((!t) & ((1u32 << rem) - 1));
                dif = (dif << rem).wrapping_sub((t as u64).wrapping_mul((rng as u64) << 48));
                cnt = cnt.wrapping_sub(rem as i32);
            }
            debug_assert!(cnt >= 0);
            let q = x.wrapping_add(14) & 0xFFFFF;
            let hi = q | sign_hi;
            let out = (rng as u64)
                | (((cnt as u8) as u64) << 16)
                | ((cur_pos.wrapping_sub(pos) as u64) << 24)
                | ((hi as u64) << 32);
            return (dif, out);
        }
        let s1 = ((rng < 0xff84) as u32) + ((rng < 0x8004) as u32);
        dif = dif.wrapping_sub(vw) << s1;
        rng = rng.wrapping_sub(v) << s1;
        cnt = cnt.wrapping_sub(s1 as i32);
        if length == 8 {
            return sign_golomb_very_slow(buf, pos, cur_pos, dif, rng, cnt, sign_hi);
        }
    }
}

/// `c < v` for `c < 2^16`, `v < 2^17`. On x86_64, expressing this as the sign
/// of `c - v` turns selects into single-uop `cmovs` instead of 2-uop `cmova`;
/// on aarch64, direct unsigned `c < v` maps cleanly to `cmp` + `csel ..., lo`
/// without keeping live subtraction results in GPRs across renormalization.
#[inline(always)]
fn lt(c: u32, v: u32) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        (c as i32).wrapping_sub(v as i32) < 0
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        c < v
    }
}

/// CDF adaptation counter update (saturating at 32).
#[inline(always)]
fn inc_count(count: u16) -> u16 {
    count.wrapping_add((count < 32) as u16)
}

const MR4_BASE: u32 = 0x0fff_0fff;
const MR5_BASE: u64 = 0x07ff_07ff_07ff_07ff;
const MR7_64: u64 = 0x0000_00ff_00ff_00ff;
const MR5_64: u64 = 0x0000_03ff_03ff_03ff;
const M1_ALL: u64 = 0x0000_0001_0001_0001;
const TOP3_64: u64 = 0x0000_8000_8000_8000;

pub(crate) const EQUI_RECIP: [u32; 128] = {
    let mut t = [0u32; 128];
    let mut i = 0;
    while i < 128 {
        let v = (((128 + i) as u64) << 7) + 4;
        t[i] = ((1u64 << 46).div_ceil(v)) as u32;
        i += 1;
    }
    t
};

#[inline(always)]
fn adapt5_m(f: u64, m1: u64) -> u64 {
    let t = f.wrapping_sub(m1);
    if (f as i64) < 0 {
        t.wrapping_add(m1 << 8).wrapping_sub((t >> 7) & MR7_64)
    } else {
        cold_refill();
        let r_idx = (f >> 62) as u32;
        let mr = (MR5_64 >> r_idx) & MR5_64;
        let add = (m1 << (10 - r_idx)) | (1u64 << 58);
        t.wrapping_add(add).wrapping_sub((t >> (5 + r_idx)) & mr)
    }
}

use crate::decoder::av1::tables::cdf::{CdfRow16, CdfRow3, CdfRow5, CdfRow8, CdfRow9};

/// CDF row types accepted by [`SymbolDecoder::symbol_cdf`].
pub trait SymbolCdf {
    fn decode_from(msac: &mut SymbolDecoder<'_>, cdf: &mut Self) -> usize;
}

impl<const N: usize> SymbolCdf for [u16; N] {
    #[inline(always)]
    fn decode_from(msac: &mut SymbolDecoder<'_>, cdf: &mut Self) -> usize {
        msac.symbol_n(cdf)
    }
}

impl<const SYMS: usize> SymbolCdf for CdfRow8<SYMS> {
    #[inline(always)]
    fn decode_from(msac: &mut SymbolDecoder<'_>, cdf: &mut Self) -> usize {
        msac.symbol_cdf8(cdf)
    }
}

impl SymbolCdf for CdfRow9 {
    #[inline(always)]
    fn decode_from(msac: &mut SymbolDecoder<'_>, cdf: &mut Self) -> usize {
        msac.symbol_cdf9(cdf)
    }
}

impl<const SYMS: usize> SymbolCdf for CdfRow16<SYMS> {
    #[inline(always)]
    fn decode_from(msac: &mut SymbolDecoder<'_>, cdf: &mut Self) -> usize {
        msac.symbol_cdf16(cdf)
    }
}

/// Register copy of a 4-symbol CDF in `CdfRow5` layout (3 inverted
/// cumulative probabilities in bits 0..48, 10 zero guard bits in 48..58, and
/// unbiased counter `min(count, 32)` in bits 58..64 so `cnt == 32` sets bit 63).
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Cdf5(pub u64);

impl Cdf5 {
    #[inline(always)]
    pub fn from_row(c: &CdfRow5) -> Self {
        Self(c.0)
    }

    #[inline(always)]
    pub fn store_row(self, c: &mut CdfRow5) {
        c.0 = self.0;
    }

    #[inline(always)]
    fn load_spec(c: &[u16; 5]) -> Self {
        let x = (c[0] as u64) | ((c[1] as u64) << 16) | ((c[2] as u64) << 32);
        let f = TOP3_64.wrapping_sub(x);
        Self(f | ((c[4].min(32) as u64) << 58))
    }

    #[inline(always)]
    fn store_spec_f(self, c: &mut [u16; 5]) {
        let x = TOP3_64.wrapping_sub(self.0);
        c[0] = x as u16;
        c[1] = (x >> 16) as u16;
        c[2] = (x >> 32) as u16;
        c[3] = 32768;
    }
}

#[cfg(feature = "native_av1_simd")]
#[inline(always)]
fn load_partial_u16x8(s: &[u16], k: usize) -> std::simd::u16x8 {
    use std::simd::u16x8;
    if k == 2 {
        let c = *s.first_chunk::<2>().unwrap();
        u16x8::from_array([c[0], c[1], 0, 0, 0, 0, 0, 0])
    } else if k == 4 {
        let c = *s.first_chunk::<4>().unwrap();
        u16x8::from_array([c[0], c[1], c[2], c[3], 0, 0, 0, 0])
    } else {
        let lo = *s.first_chunk::<4>().unwrap();
        let hi = *s[4..6].first_chunk::<2>().unwrap();
        u16x8::from_array([lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], 0, 0])
    }
}

#[cfg(all(
    feature = "native_av1_simd",
    target_arch = "aarch64",
    target_endian = "little"
))]
#[inline(always)]
fn simd_thresh8(f: std::simd::u16x8, r: u32, min_prob: std::simd::u16x8) -> std::simd::u16x8 {
    use std::simd::prelude::*;
    // Every active AV1 CDF probability is in `1..=32768`, so `f = 32768 - cdf`
    // is in `0..=32767` and both `f & 0xffc0` and `(r & 0xff00) ^ 0x8000`
    // (`== r & 0x7f00` since `r >= 0x8000`) fit in non-negative `i16`.
    // Using `(r & 0xff00) ^ 0x8000` preserves `SIGN_EXTEND` on `r_lo` so
    // `smin(sra(mul(sext, sext), 15), 32767)` matches AArch64's
    // `trySQDMULHCombine` -> single `sqdmulh`, computing
    // `((f >> 6) * ((r >> 8) & 127)) >> 1`; adding `(f & 0xffc0) + min_prob`
    // yields `(((f >> 6) * (r >> 8)) >> 1) + min_prob`.
    debug_assert!(f.simd_lt(u16x8::splat(0x8000)).all());
    let f_m = f & u16x8::splat(0xffc0);
    let r_lo = i16x8::splat(((r & 0xff00) ^ 0x8000) as i16);
    let hi = ((f_m.cast::<i16>().cast::<i32>() * r_lo.cast::<i32>()) >> i32x8::splat(15))
        .simd_min(i32x8::splat(i16::MAX as i32))
        .cast::<i16>()
        .cast::<u16>();
    hi + (f_m + min_prob)
}

/// Body of `symbol_simd8` (`8` lanes of `u16x8`).
#[cfg(feature = "native_av1_simd")]
macro_rules! simd_symbol_body {
    ($self:ident, $cdf:ident, $l:literal, $v:ident, $b:ident) => {{
        use std::simd::prelude::*;
        let n = N - 1;
        let k = (n & !1).min($l);
        let c = ($self.dif >> (WIN_SIZE - 16)) as u16;
        let (dif0, cnt) = $self.prefill_16();
        let r = $self.rng;
        let x = if N >= $l {
            $v::from_array(*$cdf.first_chunk::<$l>().unwrap())
        } else {
            load_partial_u16x8($cdf, k)
        };
        let f = if N >= 10 {
            $v::splat(0x8000) - x
        } else {
            let top = const {
                let mut t = [0u16; $l];
                let mut i = 0;
                while i < (N - 2) && i < $l {
                    t[i] = 0x8000;
                    i += 1;
                }
                $v::from_array(t)
            };
            top.saturating_sub(x)
        };
        let min_prob = const {
            let mut mp = [0u16; $l];
            let mut i = 0;
            while i < N - 2 && i < $l {
                mp[i] = (EC_MIN_PROB * (N - 2 - i) as u32) as u16;
                i += 1;
            }
            $v::from_array(mp)
        };
        // x86 / non-aarch64: dav1d's pmulhuw form + movemask / tzcnt index into a
        // stack copy of [rng, v..] (avoids u8x32::swizzle_dyn scalar fallback).
        #[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
        let (ge, new_rng, vv, val) = {
            let fm = (f & $v::splat(0xffc0)) << 1;
            let rr = $v::splat((r & 0xff00) as u16).cast::<u32>();
            let prod = ((fm.cast::<u32>() * rr) >> 16).cast::<u16>();
            let v = prod + min_prob;
            let ge = $v::splat(c).simd_ge(v);
            let ge_b = $b::from_ne_bytes(ge.to_simd().cast::<u16>().to_ne_bytes());
            let mask = ge_b.simd_lt($b::splat(0)).to_bitmask() as u32 | (1u32 << (2 * (N - 2)));
            let val = (mask.trailing_zeros() >> 1) as usize;
            let vs: [u16; 8] = *v.as_array();
            let u =
                std::hint::select_unpredictable(val == 0, r, vs[val.wrapping_sub(1) & 7] as u32);
            let vv = if N <= 9 {
                vs[val & 7] as u32
            } else {
                std::hint::select_unpredictable(val < 8, vs[val & 7] as u32, 0)
            };
            (ge, u.wrapping_sub(vv), vv, val)
        };
        // aarch64: `ge` (`c >= v`) is `0` for lanes `< val` and `-1` for lanes
        // `>= val` (including padding lanes with `v == 0`). Reinterpreting as
        // `i8x16` yields `2 * val - 16` via one `addv b`; `swizzle_dyn` (`tbl`)
        // extracts `vv` and `u - vv` directly in NEON registers without stack
        // traffic or a GPR round-trip.
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        let (ge, new_rng, vv, val) = {
            let v = simd_thresh8(f, r, min_prob);
            let ge = $v::splat(c).simd_ge(v);
            let u_vec =
                std::simd::simd_swizzle!($v::splat(r as u16), v, [7, 8, 9, 10, 11, 12, 13, 14]);
            let rng_vec = u_vec - v;
            let sum = $b::from_ne_bytes(ge.to_simd().to_ne_bytes()).reduce_sum();
            let val = if N <= 9 {
                (((sum as u8) >> 1) & 7) as usize
            } else {
                (16u8.wrapping_add(sum as u8) >> 1) as usize
            };
            let idx = u8x16::splat(sum as u8)
                + u8x16::from_array([
                    16, 17, 16, 17, 16, 17, 16, 17, 16, 17, 16, 17, 16, 17, 16, 17,
                ]);
            let v_sel = u8x16::from_ne_bytes(v.to_ne_bytes()).swizzle_dyn(idx);
            let r_sel = u8x16::from_ne_bytes(rng_vec.to_ne_bytes()).swizzle_dyn(idx);
            let vv = u16::from_ne_bytes([v_sel[0], v_sel[1]]) as u32;
            let rng0 = u16::from_ne_bytes([r_sel[0], r_sel[1]]) as u32;
            let new_rng = if N <= 9 {
                rng0
            } else {
                std::hint::select_unpredictable(val < 8, rng0, v[7] as u32)
            };
            (ge, new_rng, vv, val)
        };
        let new_dif = dif0.wrapping_sub((vv as u64) << (WIN_SIZE - 16));
        let d = (new_rng.leading_zeros() ^ 16) as i32;
        debug_assert!(cnt >= d);
        $self.dif = new_dif << ((d as u32) & 63);
        $self.rng = new_rng << ((d as u32) & 31);
        $self.cnt = cnt.wrapping_sub(d);
        if $self.allow_update {
            let gm = ge.to_simd().cast::<u16>();
            if N == $l {
                let rate = $v::splat(5 + ((x[7] >> 4) & 3));
                let delta = ge.select(f, x) >> rate;
                let y = (x + gm) - (delta ^ gm);
                let inc = y
                    .simd_lt($v::from_array([0, 0, 0, 0, 0, 0, 0, 32]))
                    .to_simd()
                    .cast::<u16>();
                let y = y - inc;
                *$cdf.first_chunk_mut::<$l>().unwrap() = *y.as_array();
            } else {
                let count = $cdf[n];
                let rate = $v::splat(5 + ((count >> 4) & 3));
                let delta = ge.select(f, x) >> rate;
                let y = (x + gm) - (delta ^ gm);
                if N > $l {
                    *$cdf.first_chunk_mut::<$l>().unwrap() = *y.as_array();
                } else {
                    $cdf[..k].copy_from_slice(&y.as_array()[..k]);
                }
                $cdf[n] = inc_count(count);
            }
        }
        val
    }};
}

#[cfg(feature = "native_av1_simd")]
macro_rules! simd_symbol_body_hot {
    ($self:ident, $cdf:ident, $n_:ident, $l:literal, $v:ident, $b:ident, $upd:ident) => {{
        use std::simd::prelude::*;
        let n = N - 1;
        let k = (n & !1).min($l);
        let dif0 = $self.dif;
        let c = (dif0 >> (WIN_SIZE - 16)) as u16;
        let r = $self.rng;
        let cnt = $self.cnt;
        let count = if $upd && N != $l { $cdf[n] } else { 0 };
        let x = if N >= $l {
            $v::from_array(*$cdf.first_chunk::<$l>().unwrap())
        } else {
            load_partial_u16x8($cdf, k)
        };
        let f = if N >= 10 {
            $v::splat(0x8000) - x
        } else {
            let top = const {
                let mut t = [0u16; $l];
                let mut i = 0;
                while i < (N - 2) && i < $l {
                    t[i] = 0x8000;
                    i += 1;
                }
                $v::from_array(t)
            };
            top.saturating_sub(x)
        };
        let min_prob = const {
            let mut mp = [0u16; $l];
            let mut i = 0;
            while i < N - 2 && i < $l {
                mp[i] = (EC_MIN_PROB * (N - 2 - i) as u32) as u16;
                i += 1;
            }
            $v::from_array(mp)
        };
        #[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
        let (ge, new_rng, vv, val) = {
            let fm = (f & $v::splat(0xffc0)) << 1;
            let rr = $v::splat((r & 0xff00) as u16).cast::<u32>();
            let prod = ((fm.cast::<u32>() * rr) >> 16).cast::<u16>();
            let v = prod + min_prob;
            let ge = $v::splat(c).simd_ge(v);
            let ge_b = $b::from_ne_bytes(ge.to_simd().cast::<u16>().to_ne_bytes());
            let mask = ge_b.simd_lt($b::splat(0)).to_bitmask() as u32 | (1u32 << (2 * (N - 2)));
            let val = (mask.trailing_zeros() >> 1) as usize;
            let vs: [u16; 8] = *v.as_array();
            let u =
                std::hint::select_unpredictable(val == 0, r, vs[val.wrapping_sub(1) & 7] as u32);
            let vv = vs[val & 7] as u32;
            (ge, u.wrapping_sub(vv), vv, val)
        };
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        let (ge, new_rng, vv, val) = {
            let v = simd_thresh8(f, r, min_prob);
            let ge = $v::splat(c).simd_ge(v);
            let u_vec =
                std::simd::simd_swizzle!($v::splat(r as u16), v, [7, 8, 9, 10, 11, 12, 13, 14]);
            let rng_vec = u_vec - v;
            let sum = $b::from_ne_bytes(ge.to_simd().to_ne_bytes()).reduce_sum();
            let val = if N <= 9 {
                (((sum as u8) >> 1) & 7) as usize
            } else {
                (16u8.wrapping_add(sum as u8) >> 1) as usize
            };
            let idx = u8x16::splat(sum as u8)
                + u8x16::from_array([
                    16, 17, 16, 17, 16, 17, 16, 17, 16, 17, 16, 17, 16, 17, 16, 17,
                ]);
            let v_sel = u8x16::from_ne_bytes(v.to_ne_bytes()).swizzle_dyn(idx);
            let r_sel = u8x16::from_ne_bytes(rng_vec.to_ne_bytes()).swizzle_dyn(idx);
            let vv = u16::from_ne_bytes([v_sel[0], v_sel[1]]) as u32;
            let rng0 = u16::from_ne_bytes([r_sel[0], r_sel[1]]) as u32;
            let new_rng = if N <= 9 {
                rng0
            } else {
                std::hint::select_unpredictable(val < 8, rng0, v[7] as u32)
            };
            (ge, new_rng, vv, val)
        };
        let new_dif = dif0.wrapping_sub((vv as u64) << (WIN_SIZE - 16));
        let d = (new_rng.leading_zeros() ^ 16) as i32;
        $self.dif = new_dif << ((d as u32) & 63);
        $self.rng = new_rng << ((d as u32) & 31);
        $self.cnt = cnt.wrapping_sub(d);
        if $upd {
            let gm = ge.to_simd().cast::<u16>();
            if N == $l {
                let rate = $v::splat(5 + ((x[7] >> 4) & 3));
                let delta = ge.select(f, x) >> rate;
                let y = (x + gm) - (delta ^ gm);
                let inc = y
                    .simd_lt($v::from_array([0, 0, 0, 0, 0, 0, 0, 32]))
                    .to_simd()
                    .cast::<u16>();
                let y = y - inc;
                *$cdf.first_chunk_mut::<$l>().unwrap() = *y.as_array();
            } else {
                let rate = $v::splat(5 + ((count >> 4) & 3));
                let delta = ge.select(f, x) >> rate;
                let y = (x + gm) - (delta ^ gm);
                if N > $l {
                    *$cdf.first_chunk_mut::<$l>().unwrap() = *y.as_array();
                } else {
                    $cdf[..k].copy_from_slice(&y.as_array()[..k]);
                }
                $cdf[n] = inc_count(count);
            }
        }
        if (cnt as u32) < (d as u32) {
            cold_refill();
            $self.refill_hot();
        }
        val
    }};
}

#[derive(Clone)]
pub struct SymbolDecoder<'a> {
    raw: ByteReader<'a>,
    dif: u64,
    rng: u32,
    cnt: i32,
    allow_update: bool,
}

#[cold]
#[inline(always)]
fn cold_refill() {}

impl<'a> SymbolDecoder<'a> {
    /// Spec `init_symbol(sz)` over the tile data `buf`.
    pub fn new(buf: &'a [u8], disable_cdf_update: bool) -> Self {
        let mut s = Self {
            raw: ByteReader { buf, pos: 0 },
            dif: 0,
            rng: 0x8000,
            cnt: -15,
            allow_update: !disable_cdf_update,
        };
        s.refill();
        s
    }

    #[inline(always)]
    fn refill(&mut self) {
        let cnt = self.cnt;
        debug_assert!((-16..=40).contains(&cnt));
        let nb = (47i32.wrapping_sub(cnt) >> 3) as usize;
        if let Some(bytes) = self
            .raw
            .buf
            .get(self.raw.pos..)
            .and_then(|s| s.first_chunk::<8>())
        {
            // Fast path: load the next 8 bytes and align them directly into
            // `dif` with a single shift. Any extra low bits (< bit 48 while
            // `cnt >= 0`) belong to `buf[pos + nb]` and are idempotently OR'd
            // at the exact same bit positions on the next refill.
            let w = !u64::from_be_bytes(*bytes);
            self.dif |= w >> (16i32.wrapping_add(cnt) as u32 & 63);
        } else {
            self.dif = refill_slow(
                self.raw.buf,
                self.raw.pos,
                self.dif,
                40i32.wrapping_sub(cnt),
            );
        }
        self.raw.pos = self.raw.pos.wrapping_add(nb);
        self.cnt = (cnt & 7) | 40;
    }

    /// Copies the mutable decoding state (`pos`, `dif`, `rng`, `cnt`) from
    /// `other`, leaving `buf` and `allow_update` untouched.
    #[inline(always)]
    pub fn copy_state(&mut self, other: &Self) {
        self.raw.pos = other.raw.pos;
        self.dif = other.dif;
        self.rng = other.rng;
        self.cnt = other.cnt;
    }

    /// `refill` with no function calls even on the EOF tail, so hot loops
    /// (`coef_tokens`, sign pass) compile as true leaf functions with all
    /// caller-saved registers available across iterations.
    #[inline(always)]
    fn refill_hot(&mut self) {
        let cnt = self.cnt;
        debug_assert!((-16..=40).contains(&cnt));
        let nb = (47i32.wrapping_sub(cnt) >> 3) as usize;
        if let Some(bytes) = self
            .raw
            .buf
            .get(self.raw.pos..)
            .and_then(|s| s.first_chunk::<8>())
        {
            let w = !u64::from_be_bytes(*bytes);
            self.dif |= w >> (16i32.wrapping_add(cnt) as u32 & 63);
        } else {
            cold_refill();
            let mut pos = self.raw.pos;
            let mut shift = 40i32.wrapping_sub(cnt);
            loop {
                let b = if pos < self.raw.buf.len() { self.raw.buf[pos] ^ 0xff } else { 0xff };
                self.dif |= (b as u64) << (shift as u32 & 63);
                pos += 1;
                shift = shift.wrapping_sub(8);
                if shift < 0 {
                    break;
                }
            }
        }
        self.raw.pos = self.raw.pos.wrapping_add(nb);
        self.cnt = (cnt & 7) | 40;
    }

    /// Byte-at-a-time refill tail for the last 7 bytes of tile data (no calls).
    #[inline(always)]
    fn refill_eof(&mut self) {
        let cnt = self.cnt;
        debug_assert!((-16..=40).contains(&cnt));
        let nb = (47i32.wrapping_sub(cnt) >> 3) as usize;
        let mut pos = self.raw.pos;
        let mut shift = 40i32.wrapping_sub(cnt);
        loop {
            let b = if pos < self.raw.buf.len() { self.raw.buf[pos] ^ 0xff } else { 0xff };
            self.dif |= (b as u64) << (shift as u32 & 63);
            pos += 1;
            shift = shift.wrapping_sub(8);
            if shift < 0 {
                break;
            }
        }
        self.raw.pos = self.raw.pos.wrapping_add(nb);
        self.cnt = (cnt & 7) | 40;
    }

    /// Branch-free refill ensuring `cnt >= 16`: whenever 8 bytes remain, tops
    /// `dif` up to `cnt in 40..=47` unconditionally; near EOF, falls back to
    /// `refill_eof` if `cnt < 16`. Returns `(dif, cnt)` in registers so callers
    /// that immediately renormalize avoid dead stores to `self.dif`/`self.cnt`.
    #[inline(always)]
    fn prefill_16(&mut self) -> (u64, i32) {
        let len = self.raw.buf.len();
        let pos = self.raw.pos;
        let dif = self.dif;
        let cnt = self.cnt;
        if len >= 8 && pos <= len - 8 {
            if let Some(bytes) = self.raw.buf[pos..].first_chunk::<8>() {
                debug_assert!((0..=47).contains(&cnt));
                let w = !u64::from_be_bytes(*bytes);
                let new_dif = dif | (w >> (16i32.wrapping_add(cnt) as u32 & 63));
                self.raw.pos = pos.wrapping_add(5).wrapping_sub((cnt as u32 >> 3) as usize);
                return (new_dif, (cnt & 7) | 40);
            }
        }
        cold_refill();
        if cnt < 16 {
            self.refill_eof();
            (self.dif, self.cnt)
        } else {
            (dif, cnt)
        }
    }

    /// Spec `SymbolMaxBits`: bits of the tile data still unread (negative
    /// once zero padding past the end is being consumed).
    #[inline]
    pub fn symbol_max_bits(&self) -> i64 {
        8 * (self.raw.buf.len() as i64 - self.raw.pos as i64) + self.cnt as i64
    }

    /// True once more than 14 padding bits have been consumed. The spec
    /// requires `SymbolMaxBits >= -14` when the tile ends (8.2.4) and the
    /// value never increases, so the tile data is known to be truncated
    /// (or corrupt) and decoding can stop early.
    #[inline]
    pub fn overrun(&self) -> bool {
        self.symbol_max_bits() < -14
    }

    /// Renormalization after a symbol. The refill (every few symbols) is
    /// kept out of line: inlined at every symbol read it was the largest
    /// single item of the per-block code (L1i misses).
    #[inline(always)]
    fn norm(&mut self, dif: u64, rng: u32) {
        let d = (rng.leading_zeros() ^ 16) as i32;
        let cnt = self.cnt;
        self.dif = dif << ((d as u32) & 63);
        self.rng = rng << ((d as u32) & 31);
        self.cnt = cnt.wrapping_sub(d);
        if (cnt as u32) < (d as u32) {
            cold_refill();
            let (dif, out) = refill_cold::<0>(self.raw.buf, self.raw.pos, self.dif, self.cnt);
            self.dif = dif;
            self.raw.pos = self.raw.pos.wrapping_add((out as u32) as usize);
            self.cnt = (out >> 32) as i32;
        }
    }

    /// `norm` with the refill inlined, for the coefficient loops.
    #[inline(always)]
    fn norm_hot(&mut self, dif: u64, rng: u32) {
        let d = (rng.leading_zeros() ^ 16) as i32;
        let cnt = self.cnt;
        self.dif = dif << ((d as u32) & 63);
        self.rng = rng << ((d as u32) & 31);
        self.cnt = cnt.wrapping_sub(d);
        if (cnt as u32) < (d as u32) {
            cold_refill();
            self.refill_hot();
        }
    }

    /// Decodes a boolean with probability `f / 32768` of being 0, where `f`
    /// is the inverted CDF value (`32768 - cdf[0]`).
    #[inline(always)]
    pub fn bool_inv(&mut self, f: u32) -> bool {
        use std::hint::select_unpredictable as sel;
        let r = self.rng;
        let dif = self.dif;
        let v = (((r >> 8).wrapping_mul(f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
            .wrapping_add(EC_MIN_PROB);
        let vw = (v as u64) << (WIN_SIZE - 16);
        let ret = dif >= vw;
        let new_dif = sel(ret, dif.wrapping_sub(vw), dif);
        let new_rng = sel(ret, r.wrapping_sub(v), v);
        self.norm(new_dif, new_rng);
        !ret
    }

    /// Spec `read_symbol(cdf)`.
    pub fn symbol(&mut self, cdf: &mut [u16]) -> usize {
        macro_rules! dispatch {
            ($($n:literal)*) => {
                match cdf.len() {
                    $($n => self.symbol_n::<$n>(cdf.try_into().unwrap()),)*
                    _ => self.symbol_generic(cdf),
                }
            };
        }
        dispatch!(3 4 5 6 7 8 9 10 11 12 13 14 15 16 17)
    }

    /// Dispatches `read_symbol` for any CDF row representation (`[u16; N]`,
    /// `CdfRow8<SYMS>`, or `CdfRow16<SYMS>`).
    #[inline(always)]
    pub fn symbol_cdf<T: SymbolCdf>(&mut self, cdf: &mut T) -> usize {
        T::decode_from(self, cdf)
    }

    #[cfg(not(feature = "native_av1_simd"))]
    fn symbol_inv_scalar(
        &mut self,
        lo: &mut [u16],
        hi: &mut [u16],
        syms: usize,
        count: &mut u16,
    ) -> usize {
        let mut s = [0u16; 17];
        let n_lo = lo.len().min(syms - 1);
        for (dst, &src) in s[..n_lo].iter_mut().zip(&lo[..n_lo]) {
            *dst = 32768u16.wrapping_sub(src);
        }
        for (dst, &src) in s[n_lo..syms - 1].iter_mut().zip(hi.iter()) {
            *dst = 32768u16.wrapping_sub(src);
        }
        s[syms - 1] = 32768;
        s[syms] = *count;
        let val = self.symbol_generic(&mut s[..=syms]);
        if self.allow_update {
            for (dst, &src) in lo[..n_lo].iter_mut().zip(&s[..n_lo]) {
                *dst = 32768u16.wrapping_sub(src);
            }
            for (dst, &src) in hi.iter_mut().zip(&s[n_lo..syms - 1]) {
                *dst = 32768u16.wrapping_sub(src);
            }
            *count = s[syms].min(32);
        }
        val
    }

    /// `read_symbol` for a 16-byte aligned inverted 8-lane CDF (`6 <= SYMS <= 8`,
    /// `count` in lane 7).
    #[inline(always)]
    pub fn symbol_cdf8<const SYMS: usize>(&mut self, cdf: &mut CdfRow8<SYMS>) -> usize {
        #[cfg(not(feature = "native_av1_simd"))]
        {
            let (head, tail) = cdf.0.split_at_mut(7);
            return self.symbol_inv_scalar(head, &mut [], SYMS, &mut tail[0]);
        }
        #[cfg(feature = "native_av1_simd")]
        {
            use std::simd::prelude::*;
            const { assert!(SYMS >= 6 && SYMS <= 8) };
            let c = (self.dif >> (WIN_SIZE - 16)) as u16;
            let (dif0, cnt) = self.prefill_16();
            let r = self.rng;
            let f = u16x8::from_array(cdf.0);
            let min_prob = const {
                let mut mp = [0u16; 8];
                let mut i = 0;
                while i < SYMS - 1 {
                    mp[i] = (EC_MIN_PROB * (SYMS - 1 - i) as u32) as u16;
                    i += 1;
                }
                u16x8::from_array(mp)
            };
            #[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
            let (ge, new_rng, vv, val) = {
                let fm = (f & u16x8::splat(0xffc0)) << 1;
                let rr = u16x8::splat((r & 0xff00) as u16).cast::<u32>();
                let v = ((fm.cast::<u32>() * rr) >> 16).cast::<u16>() + min_prob;
                let ge = u16x8::splat(c).simd_ge(v);
                let ge_b = i8x16::from_ne_bytes(ge.to_simd().cast::<u16>().to_ne_bytes());
                let mask =
                    ge_b.simd_lt(i8x16::splat(0)).to_bitmask() as u32 | (1u32 << (2 * (SYMS - 1)));
                let val = (mask.trailing_zeros() >> 1) as usize;
                let vs: [u16; 8] = *v.as_array();
                let u = std::hint::select_unpredictable(
                    val == 0,
                    r,
                    vs[val.wrapping_sub(1) & 7] as u32,
                );
                let vv = vs[val & 7] as u32;
                (ge, u.wrapping_sub(vv), vv, val)
            };
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            let (ge, new_rng, vv, val) = {
                let v = simd_thresh8(f, r, min_prob);
                let ge = u16x8::splat(c).simd_ge(v);
                let u_vec = std::simd::simd_swizzle!(
                    u16x8::splat(r as u16),
                    v,
                    [7, 8, 9, 10, 11, 12, 13, 14]
                );
                let rng_vec = u_vec - v;
                let sum = i8x16::from_ne_bytes(ge.to_simd().to_ne_bytes()).reduce_sum();
                let val = (((sum as u8) >> 1) & 7) as usize;
                let idx = u8x16::splat(sum as u8)
                    + u8x16::from_array([
                        16, 17, 16, 17, 16, 17, 16, 17, 16, 17, 16, 17, 16, 17, 16, 17,
                    ]);
                let v_sel = u8x16::from_ne_bytes(v.to_ne_bytes()).swizzle_dyn(idx);
                let r_sel = u8x16::from_ne_bytes(rng_vec.to_ne_bytes()).swizzle_dyn(idx);
                let vv = u16::from_ne_bytes([v_sel[0], v_sel[1]]) as u32;
                let new_rng = u16::from_ne_bytes([r_sel[0], r_sel[1]]) as u32;
                (ge, new_rng, vv, val)
            };
            let new_dif = dif0.wrapping_sub((vv as u64) << (WIN_SIZE - 16));
            let d = (new_rng.leading_zeros() ^ 16) as i32;
            debug_assert!(cnt >= d);
            self.dif = new_dif << ((d as u32) & 63);
            self.rng = new_rng << ((d as u32) & 31);
            self.cnt = cnt.wrapping_sub(d);
            if self.allow_update {
                let gm = ge.to_simd();
                let fi = f.cast::<i16>();
                let rate = i16x8::splat(5 + ((f[7] >> 4) & 3) as i16);
                let delta = ((gm | i16x8::splat(i16::MIN)) - fi) >> rate;
                let y = ((fi - gm) + delta).cast::<u16>();
                let inc = y
                    .simd_lt(u16x8::from_array([0, 0, 0, 0, 0, 0, 0, 32]))
                    .to_simd()
                    .cast::<u16>();
                cdf.0 = *(y - inc).as_array();
            }
            val
        }
    }

    /// `read_symbol` for a 16-byte aligned inverted 9-symbol CDF (`eob_pt_256`):
    /// single 8-lane vector for `f[0..8]` with a 2-vector `tbl` for `new_rng`
    /// (`[rng_vec, [v[7], 0..]]`) and 1-vector `tbl` for `vv` (out-of-range
    /// index 16 returns `0` when `val == 8`).
    #[inline(always)]
    pub fn symbol_cdf9(&mut self, cdf: &mut CdfRow9) -> usize {
        #[cfg(not(feature = "native_av1_simd"))]
        {
            return self.symbol_inv_scalar(&mut cdf.f, &mut [], 9, &mut cdf.count);
        }
        #[cfg(feature = "native_av1_simd")]
        {
            use std::simd::prelude::*;
            let c = (self.dif >> (WIN_SIZE - 16)) as u16;
            let (dif0, cnt) = self.prefill_16();
            let r = self.rng;
            let f = u16x8::from_array(cdf.f);
            let min_prob = u16x8::from_array([32, 28, 24, 20, 16, 12, 8, 4]);
            #[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
            let (ge, new_rng, vv, val) = {
                let fm = (f & u16x8::splat(0xffc0)) << 1;
                let rr = u16x8::splat((r & 0xff00) as u16).cast::<u32>();
                let v = ((fm.cast::<u32>() * rr) >> 16).cast::<u16>() + min_prob;
                let ge = u16x8::splat(c).simd_ge(v);
                let ge_b = i8x16::from_ne_bytes(ge.to_simd().cast::<u16>().to_ne_bytes());
                let mask = ge_b.simd_lt(i8x16::splat(0)).to_bitmask() as u32 | (1u32 << 16);
                let val = (mask.trailing_zeros() >> 1) as usize;
                let vs: [u16; 8] = *v.as_array();
                let u = std::hint::select_unpredictable(
                    val == 0,
                    r,
                    vs[val.wrapping_sub(1) & 7] as u32,
                );
                let vv = std::hint::select_unpredictable(val < 8, vs[val & 7] as u32, 0);
                (ge, u.wrapping_sub(vv), vv, val)
            };
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            let (ge, new_rng, vv, val) = {
                let v = simd_thresh8(f, r, min_prob);
                let ge = u16x8::splat(c).simd_ge(v);
                let u_vec = std::simd::simd_swizzle!(
                    u16x8::splat(r as u16),
                    v,
                    [7, 8, 9, 10, 11, 12, 13, 14]
                );
                let rng_vec = u_vec - v;
                let rng_tail = std::simd::simd_swizzle!(v, [7, 7, 7, 7, 7, 7, 7, 7]);
                let sum = i8x16::from_ne_bytes(ge.to_simd().to_ne_bytes()).reduce_sum();
                let val = (16u8.wrapping_add(sum as u8) >> 1) as usize;
                let idx16 = u8x16::splat(sum as u8)
                    + u8x16::from_array([
                        16, 17, 16, 17, 16, 17, 16, 17, 16, 17, 16, 17, 16, 17, 16, 17,
                    ]);
                let idx32 = std::simd::simd_swizzle!(
                    idx16,
                    [
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6,
                        7, 8, 9, 10, 11, 12, 13, 14, 15,
                    ]
                );
                let v_sel = u8x16::from_ne_bytes(v.to_ne_bytes()).swizzle_dyn(idx16);
                let r_cat = std::simd::simd_swizzle!(
                    rng_vec.to_ne_bytes(),
                    rng_tail.to_ne_bytes(),
                    [
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                    ]
                );
                let r_sel = r_cat.swizzle_dyn(idx32);
                let vv = u16::from_ne_bytes([v_sel[0], v_sel[1]]) as u32;
                let new_rng = u16x16::from_ne_bytes(r_sel)[0] as u32;
                (ge, new_rng, vv, val)
            };
            let new_dif = dif0.wrapping_sub((vv as u64) << (WIN_SIZE - 16));
            let d = (new_rng.leading_zeros() ^ 16) as i32;
            debug_assert!(cnt >= d);
            self.dif = new_dif << ((d as u32) & 63);
            self.rng = new_rng << ((d as u32) & 31);
            self.cnt = cnt.wrapping_sub(d);
            if self.allow_update {
                let count = cdf.count;
                let gm = ge.to_simd();
                let fi = f.cast::<i16>();
                let rate = i16x8::splat(5 + ((count >> 4) & 3) as i16);
                let delta = ((gm | i16x8::splat(i16::MIN)) - fi) >> rate;
                let y = ((fi - gm) + delta).cast::<u16>();
                cdf.f = *y.as_array();
                cdf.count = inc_count(count);
            }
            val
        }
    }

    /// `read_symbol` for a 16-byte aligned inverted 16-lane CDF
    /// (`9 <= SYMS <= 16`, `count` in lane 15).
    #[inline(always)]
    pub fn symbol_cdf16<const SYMS: usize>(&mut self, cdf: &mut CdfRow16<SYMS>) -> usize {
        #[cfg(not(feature = "native_av1_simd"))]
        {
            let [lo, hi_all] = &mut cdf.0;
            let (hi, count) = hi_all.split_at_mut(7);
            return self.symbol_inv_scalar(lo, hi, SYMS, &mut count[0]);
        }
        #[cfg(feature = "native_av1_simd")]
        {
            use std::simd::prelude::*;
            const { assert!(SYMS >= 9 && SYMS <= 16) };
            let c = (self.dif >> (WIN_SIZE - 16)) as u16;
            let (dif0, cnt) = self.prefill_16();
            let r = self.rng;
            let f0 = u16x8::from_array(cdf.0[0]);
            let f1 = u16x8::from_array(cdf.0[1]);
            let (mp0, mp1) = const {
                let mut m0 = [0u16; 8];
                let mut m1 = [0u16; 8];
                let mut i = 0;
                while i < 8 {
                    if i < SYMS - 1 {
                        m0[i] = (EC_MIN_PROB * (SYMS - 1 - i) as u32) as u16;
                    }
                    if i + 8 < SYMS - 1 {
                        m1[i] = (EC_MIN_PROB * (SYMS - 1 - (i + 8)) as u32) as u16;
                    }
                    i += 1;
                }
                (u16x8::from_array(m0), u16x8::from_array(m1))
            };
            #[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
            let (ge0, ge1, new_rng, vv, val) = {
                let mask_ffc0 = u16x8::splat(0xffc0);
                let rr = u16x8::splat((r & 0xff00) as u16).cast::<u32>();
                let fm0 = (f0 & mask_ffc0) << 1;
                let fm1 = (f1 & mask_ffc0) << 1;
                let v0 = ((fm0.cast::<u32>() * rr) >> 16).cast::<u16>() + mp0;
                let v1 = ((fm1.cast::<u32>() * rr) >> 16).cast::<u16>() + mp1;
                let vc = u16x8::splat(c);
                let ge0 = vc.simd_ge(v0);
                let ge1 = vc.simd_ge(v1);
                let b0 = i8x16::from_ne_bytes(ge0.to_simd().cast::<u16>().to_ne_bytes());
                let b1 = i8x16::from_ne_bytes(ge1.to_simd().cast::<u16>().to_ne_bytes());
                let m0 = b0.simd_lt(i8x16::splat(0)).to_bitmask() as u32;
                let m1 = b1.simd_lt(i8x16::splat(0)).to_bitmask() as u32;
                let mask = m0 | (m1 << 16) | (1u32 << (2 * (SYMS - 1)));
                let val = (mask.trailing_zeros() >> 1) as usize;
                let mut vs = [0u16; 16];
                vs[..8].copy_from_slice(v0.as_array());
                vs[8..].copy_from_slice(v1.as_array());
                let u = std::hint::select_unpredictable(
                    val == 0,
                    r,
                    vs[val.wrapping_sub(1) & 15] as u32,
                );
                let vv = vs[val & 15] as u32;
                (ge0, ge1, u.wrapping_sub(vv), vv, val)
            };
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            let (ge0, ge1, new_rng, vv, val) = {
                let v0 = simd_thresh8(f0, r, mp0);
                let v1 = simd_thresh8(f1, r, mp1);
                let vc = u16x8::splat(c);
                let ge0 = vc.simd_ge(v0);
                let ge1 = vc.simd_ge(v1);
                let u0 = std::simd::simd_swizzle!(
                    u16x8::splat(r as u16),
                    v0,
                    [7, 8, 9, 10, 11, 12, 13, 14]
                );
                let u1 = std::simd::simd_swizzle!(v0, v1, [7, 8, 9, 10, 11, 12, 13, 14]);
                let rng0 = u0 - v0;
                let rng1 = u1 - v1;
                let sum = (i8x16::from_ne_bytes(ge0.to_simd().to_ne_bytes())
                    + i8x16::from_ne_bytes(ge1.to_simd().to_ne_bytes()))
                .reduce_sum();
                let val = (((sum as u8) >> 1) & 15) as usize;
                let idx = u8x32::splat(sum as u8)
                    + u8x32::from_array([
                        32, 33, 32, 33, 32, 33, 32, 33, 32, 33, 32, 33, 32, 33, 32, 33, 32, 33, 32,
                        33, 32, 33, 32, 33, 32, 33, 32, 33, 32, 33, 32, 33,
                    ]);
                let v_cat = std::simd::simd_swizzle!(
                    v0.to_ne_bytes(),
                    v1.to_ne_bytes(),
                    [
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
                    ]
                );
                let r_cat = std::simd::simd_swizzle!(
                    rng0.to_ne_bytes(),
                    rng1.to_ne_bytes(),
                    [
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
                    ]
                );
                let v_sel = v_cat.swizzle_dyn(idx);
                let r_sel = r_cat.swizzle_dyn(idx);
                let vv = u16x16::from_ne_bytes(v_sel)[0] as u32;
                let new_rng = u16x16::from_ne_bytes(r_sel)[0] as u32;
                (ge0, ge1, new_rng, vv, val)
            };
            let new_dif = dif0.wrapping_sub((vv as u64) << (WIN_SIZE - 16));
            let d = (new_rng.leading_zeros() ^ 16) as i32;
            debug_assert!(cnt >= d);
            self.dif = new_dif << ((d as u32) & 63);
            self.rng = new_rng << ((d as u32) & 31);
            self.cnt = cnt.wrapping_sub(d);
            if self.allow_update {
                let gm0 = ge0.to_simd();
                let gm1 = ge1.to_simd();
                let fi0 = f0.cast::<i16>();
                let fi1 = f1.cast::<i16>();
                let rate = i16x8::splat(5 + ((f1[7] >> 4) & 3) as i16);
                let d0 = ((gm0 | i16x8::splat(i16::MIN)) - fi0) >> rate;
                let d1 = ((gm1 | i16x8::splat(i16::MIN)) - fi1) >> rate;
                let y0 = ((fi0 - gm0) + d0).cast::<u16>();
                let y1 = ((fi1 - gm1) + d1).cast::<u16>();
                let inc = y1
                    .simd_lt(u16x8::from_array([0, 0, 0, 0, 0, 0, 0, 32]))
                    .to_simd()
                    .cast::<u16>();
                cdf.0[0] = *y0.as_array();
                cdf.0[1] = *(y1 - inc).as_array();
            }
            val
        }
    }

    /// `read_symbol` for a CDF with `N - 1` symbols (`N` entries
    /// including the adaptation counter). Branch free search and
    /// adaptation.
    #[inline(always)]
    pub fn symbol_n<const N: usize>(&mut self, cdf: &mut [u16; N]) -> usize {
        const { assert!(N >= 3 && N <= 17) };
        let n = N - 1;
        debug_assert!(cdf[n - 1] == 32768);
        if N == 3 {
            return self.bool_adapt(cdf.first_chunk_mut().unwrap()) as usize;
        }
        if N == 4 {
            return self.symbol4(cdf.first_chunk_mut().unwrap());
        }
        if N == 5 {
            let upd = self.allow_update;
            return self.symbol5(cdf.first_chunk_mut().unwrap(), upd);
        }
        if N == 6 {
            return self.symbol6(cdf.first_chunk_mut().unwrap());
        }
        #[cfg(feature = "native_av1_simd")]
        {
            if N <= 10 {
                return self.symbol_simd8::<N>(cdf);
            }
            self.symbol_simd16::<N>(cdf)
        }
        #[cfg(not(feature = "native_av1_simd"))]
        {
            self.symbol_generic(cdf)
        }
    }

    /// `symbol_n::<4>` (3-symbol CDFs).
    #[inline(always)]
    pub fn symbol4(&mut self, cdf: &mut [u16; 4]) -> usize {
        use std::hint::select_unpredictable as sel;
        let c = (self.dif >> (WIN_SIZE - 16)) as u32;
        let r = self.rng >> 8;
        let x = (cdf[0] as u32) | ((cdf[1] as u32) << 16);
        let f = 0x8000_8000u32.wrapping_sub(x);
        let v0 = ((r.wrapping_mul((f >> 6) & 0x3ff)) >> 1).wrapping_add(8);
        let v1 = ((r.wrapping_mul(f >> 22)) >> 1).wrapping_add(4);
        let c0 = lt(c, v0);
        let c1 = lt(c, v1);
        let val = (c0 as u32).wrapping_add(c1 as u32);
        let u0 = sel(c0, v0, self.rng);
        let u = sel(c1, v1, u0);
        let vv0 = sel(c0, v1, v0);
        let v = sel(c1, 0, vv0);
        let new_dif = self.dif.wrapping_sub((v as u64) << (WIN_SIZE - 16));
        let new_rng = u.wrapping_sub(v);
        if self.allow_update {
            let count = cdf[3];
            let nm1 = (!c0 as u32) | ((!c1 as u32) << 16);
            let t = x.wrapping_sub(nm1);
            let r_idx = ((count as u32) >> 4).min(2);
            let mr = (MR4_BASE >> r_idx) & MR4_BASE;
            cdf[3] = inc_count(count);
            let y = t
                .wrapping_add(nm1 << (11 - r_idx))
                .wrapping_sub((t >> (4 + r_idx)) & mr);
            cdf[0] = y as u16;
            cdf[1] = (y >> 16) as u16;
        }
        self.norm(new_dif, new_rng);
        val as usize
    }

    /// `symbol_n::<4>` for loops specialized on the adaptation flag.
    #[inline(always)]
    pub fn symbol4_c<const UPD: bool, const HOT: bool>(&mut self, cdf: &mut [u16; 4]) -> u32 {
        use std::hint::select_unpredictable as sel;
        let c = (self.dif >> (WIN_SIZE - 16)) as u32;
        let r = self.rng >> 8;
        let x = (cdf[0] as u32) | ((cdf[1] as u32) << 16);
        let f = 0x8000_8000u32.wrapping_sub(x);
        let v0 = ((r.wrapping_mul((f >> 6) & 0x3ff)) >> 1).wrapping_add(8);
        let v1 = ((r.wrapping_mul(f >> 22)) >> 1).wrapping_add(4);
        let c0 = lt(c, v0);
        let c1 = lt(c, v1);
        let val0 = c0 as u32;
        let u0 = sel(c0, v0, self.rng);
        let vv0 = sel(c0, v1, v0);
        let nm0 = sel(c0, 0x0001_0000u32, 0x0001_0001u32);

        let val = val0.wrapping_add(c1 as u32);
        let u = sel(c1, v1, u0);
        let v = sel(c1, 0, vv0);
        let new_dif = self.dif.wrapping_sub((v as u64) << (WIN_SIZE - 16));
        let new_rng = u.wrapping_sub(v);
        if UPD {
            let count = cdf[3];
            let nm1 = sel(c1, 0u32, nm0);
            let t = x.wrapping_sub(nm1);
            let r_idx = ((count as u32) >> 4).min(2);
            let mr = (MR4_BASE >> r_idx) & MR4_BASE;
            cdf[3] = inc_count(count);
            let y = t
                .wrapping_add(nm1 << (11 - r_idx))
                .wrapping_sub((t >> (4 + r_idx)) & mr);
            cdf[0] = y as u16;
            cdf[1] = (y >> 16) as u16;
        }
        if HOT {
            self.norm_hot(new_dif, new_rng);
        } else {
            self.norm(new_dif, new_rng);
        }
        val
    }

    /// `symbol_n::<5>` (4-symbol CDFs). `upd` is the CDF adaptation flag.
    #[inline(always)]
    fn symbol5(&mut self, cdf: &mut [u16; 5], upd: bool) -> usize {
        let count = cdf[4];
        let (val, y) = self.symbol5_r::<false>(Cdf5::load_spec(cdf), upd);
        if upd {
            y.store_spec_f(cdf);
            cdf[4] = inc_count(count);
        }
        val as usize
    }

    /// `symbol_n::<6>` (5-symbol CDFs: 4 values below 32768, 32768, count).
    #[inline(always)]
    pub fn symbol6(&mut self, cdf: &mut [u16; 6]) -> usize {
        self.symbol6_c::<true, false>(cdf)
    }

    #[inline(always)]
    pub fn symbol6_c<const UPD: bool, const HOT: bool>(&mut self, cdf: &mut [u16; 6]) -> usize {
        use std::hint::select_unpredictable as sel;
        let c = (self.dif >> (WIN_SIZE - 16)) as u32;
        let r = self.rng >> 8;
        let x = (cdf[0] as u64)
            | ((cdf[1] as u64) << 16)
            | ((cdf[2] as u64) << 32)
            | ((cdf[3] as u64) << 48);
        let f = 0x8000_8000_8000_8000u64.wrapping_sub(x);
        let f_lo = f as u32;
        let f_hi = (f >> 32) as u32;
        let v0 = ((r.wrapping_mul((f_lo >> 6) & 0x3ff)) >> 1).wrapping_add(16);
        let v1 = ((r.wrapping_mul(f_lo >> 22)) >> 1).wrapping_add(12);
        let v2 = ((r.wrapping_mul((f_hi >> 6) & 0x3ff)) >> 1).wrapping_add(8);
        let v3 = ((r.wrapping_mul(f_hi >> 22)) >> 1).wrapping_add(4);
        let c0 = lt(c, v0);
        let c1 = lt(c, v1);
        let c2 = lt(c, v2);
        let c3 = lt(c, v3);
        let val = (c0 as u32)
            .wrapping_add(c2 as u32)
            .wrapping_add(c3 as u32)
            .wrapping_add(c1 as u32);
        let u_lo = sel(c0, v0, self.rng);
        let u_hi = sel(c3, v3, sel(c2, v2, v1));
        let u = sel(c1, u_hi, u_lo);
        let v_lo = sel(c0, v1, v0);
        let v_hi = sel(c3, 0, sel(c2, v3, v2));
        let v = sel(c1, v_hi, v_lo);
        let new_dif = self.dif.wrapping_sub((v as u64) << (WIN_SIZE - 16));
        let new_rng = u.wrapping_sub(v);
        let upd_flag = if UPD { self.allow_update } else { false };
        if upd_flag {
            let count = cdf[5];
            let nm1 =
                (!c0 as u64) | ((!c1 as u64) << 16) | ((!c2 as u64) << 32) | ((!c3 as u64) << 48);
            let t = x.wrapping_sub(nm1);
            let r_idx = ((count as u32) >> 4).min(2);
            let mr = (MR5_BASE >> r_idx) & MR5_BASE;
            cdf[5] = inc_count(count);
            let y = t
                .wrapping_add(nm1 << (10 - r_idx))
                .wrapping_sub((t >> (5 + r_idx)) & mr);
            cdf[0] = y as u16;
            cdf[1] = (y >> 16) as u16;
            cdf[2] = (y >> 32) as u16;
            cdf[3] = (y >> 48) as u16;
        }
        if HOT {
            self.norm_hot(new_dif, new_rng);
        } else {
            self.norm(new_dif, new_rng);
        }
        val as usize
    }

    /// `symbol_n::<5>` on a pre-inverted `Cdf5` held in registers: returns
    /// the symbol and the adapted `Cdf5` (unchanged unless `upd`).
    #[inline(always)]
    pub fn symbol5_r<const HOT: bool>(&mut self, cdf: Cdf5, upd: bool) -> (u32, Cdf5) {
        use std::hint::select_unpredictable as sel;
        let c = (self.dif >> (WIN_SIZE - 16)) as u32;
        let r = self.rng >> 8;
        let f = cdf.0;
        let v0 = ((r.wrapping_mul(((f as u32) >> 6) & 0x3ff)) >> 1).wrapping_add(12);
        let v1 = ((r.wrapping_mul((f as u32) >> 22)) >> 1).wrapping_add(8);
        let v2 = ((r.wrapping_mul(((f >> 38) as u32) & 0x3ff)) >> 1).wrapping_add(4);
        let c0 = lt(c, v0);
        let c1 = lt(c, v1);
        let c2 = lt(c, v2);
        let val = (c0 as u32).wrapping_add(c2 as u32).wrapping_add(c1 as u32);
        let u_lo = sel(c0, v0, self.rng);
        let u_hi = sel(c2, v2, v1);
        let u = sel(c1, u_hi, u_lo);
        let v_lo = sel(c0, v1, v0);
        let v_hi = sel(c2, 0, v2);
        let v = sel(c1, v_hi, v_lo);
        let new_dif = self.dif.wrapping_sub((v as u64) << (WIN_SIZE - 16));
        let new_rng = u.wrapping_sub(v);
        let y = if upd {
            let m1 = (c0 as u64) | ((c1 as u64) << 16) | ((c2 as u64) << 32);
            Cdf5(adapt5_m(f, m1))
        } else {
            cdf
        };
        if HOT {
            self.norm_hot(new_dif, new_rng);
        } else {
            self.norm(new_dif, new_rng);
        }
        (val, y)
    }

    /// Whether CDF adaptation is enabled.
    #[inline(always)]
    pub fn allow_update(&self) -> bool {
        self.allow_update
    }

    /// `symbol_n::<5>` on a pre-inverted 8-byte `CdfRow5` (`coeff_base`),
    /// specialized on the adaptation flag.
    #[inline(always)]
    pub fn symbol5_c<const UPD: bool>(&mut self, cdf: &mut CdfRow5) -> u32 {
        use std::hint::select_unpredictable as sel;
        let c = (self.dif >> (WIN_SIZE - 16)) as u32;
        let r = self.rng >> 8;
        let f = cdf.0;
        let v0 = ((r.wrapping_mul(((f as u32) >> 6) & 0x3ff)) >> 1).wrapping_add(12);
        let v1 = ((r.wrapping_mul((f as u32) >> 22)) >> 1).wrapping_add(8);
        let v2 = ((r.wrapping_mul(((f >> 38) as u32) & 0x3ff)) >> 1).wrapping_add(4);
        let c0 = lt(c, v0);
        let c1 = lt(c, v1);
        let c2 = lt(c, v2);
        let val0 = c0 as u32;
        let u_lo = sel(c0, v0, self.rng);
        let v_lo = sel(c0, v1, v0);

        let val_hi = sel(c2, 3u32, 2u32);
        let u_hi = sel(c2, v2, v1);
        let v_hi = sel(c2, 0, v2);
        let m_hi = sel(c2, M1_ALL, 0x0001_0001u64);

        let val = sel(c1, val_hi, val0);
        let u = sel(c1, u_hi, u_lo);
        let v = sel(c1, v_hi, v_lo);
        let new_dif = self.dif.wrapping_sub((v as u64) << (WIN_SIZE - 16));
        let new_rng = u.wrapping_sub(v);
        if UPD {
            let m1 = sel(c1, m_hi, val0 as u64);
            cdf.0 = adapt5_m(f, m1);
        }
        self.norm_hot(new_dif, new_rng);
        val
    }

    /// `symbol_n::<5>` on the `CdfRow5` value `f` (`coeff_base`), fused with
    /// `levels` / `nz` stores before adaptation and renormalization to hide
    /// store-to-load forwarding latency in `coef_tokens`. Returns whether
    /// the symbol is 3 and the adapted CDF (`f` itself if `!UPD`), which the
    /// caller stores back.
    #[inline(always)]
    pub fn symbol5_coef<const UPD: bool>(
        &mut self,
        f: u64,
        levels: &mut [u8; 2200],
        nz: &mut [u16; 1024],
        idx: usize,
        pos: usize,
        k: &mut usize,
    ) -> (bool, u64) {
        use std::hint::select_unpredictable as sel;
        let c = (self.dif >> (WIN_SIZE - 16)) as u32;
        let r = self.rng >> 8;
        let v0 = ((r.wrapping_mul(((f as u32) >> 6) & 0x3ff)) >> 1).wrapping_add(12);
        let v1 = ((r.wrapping_mul((f as u32) >> 22)) >> 1).wrapping_add(8);
        let v2 = ((r.wrapping_mul(((f >> 38) as u32) & 0x3ff)) >> 1).wrapping_add(4);
        let c0 = lt(c, v0);
        let c1 = lt(c, v1);
        let c2 = lt(c, v2);
        // Group all uses of c0, then c2, then c1 so each comparison only
        // sets flags once and `m1` uses a 2-level select tree (2 cycles).
        let val0 = c0 as u32;
        let u_lo = sel(c0, v0, self.rng);
        let v_lo = sel(c0, v1, v0);
        let cur_k = *k;
        *k = cur_k.wrapping_add(c0 as usize);

        let val02 = val0.wrapping_add(c2 as u32);
        let u_hi = sel(c2, v2, v1);
        let v_hi = sel(c2, 0, v2);
        let m_hi = sel(c2, M1_ALL, 0x0001_0001u64);

        let val = val02.wrapping_add(c1 as u32);
        let u = sel(c1, u_hi, u_lo);
        let v = sel(c1, v_hi, v_lo);
        let new_dif = self.dif.wrapping_sub((v as u64) << (WIN_SIZE - 16));
        let new_rng = u.wrapping_sub(v);
        levels[idx & 2047] = (val.wrapping_mul(0x41)) as u8;
        nz[cur_k & 1023] = ((val as u16) << 10) | (pos as u16);
        let new_f = if UPD {
            let m1 = sel(c1, m_hi, val0 as u64);
            adapt5_m(f, m1)
        } else {
            f
        };
        self.norm_hot(new_dif, new_rng);
        (c2, new_f)
    }

    /// `hi_tok` for loops specialized on the adaptation flag.
    #[inline(always)]
    pub fn hi_tok_c<const UPD: bool>(&mut self, cdf: &mut CdfRow5) -> u32 {
        use std::hint::select_unpredictable as sel;
        let mut f = cdf.0;
        let mut base = 3u32;
        let (c, r, v2) = loop {
            let c = (self.dif >> (WIN_SIZE - 16)) as u32;
            let r = self.rng >> 8;
            let v2 = ((r.wrapping_mul(((f >> 38) as u32) & 0x3ff)) >> 1).wrapping_add(4);
            if !lt(c, v2) {
                break (c, r, v2);
            }
            if UPD {
                f = adapt5_m(f, M1_ALL);
            }
            let d = (v2.leading_zeros() ^ 16) as i32;
            self.dif <<= (d as u32) & 63;
            self.rng = v2 << ((d as u32) & 31);
            self.cnt = self.cnt.wrapping_sub(d);
            if base == 3 && self.cnt < 39 {
                self.refill_hot();
            }
            base = base.wrapping_add(3);
            if base == 15 {
                if UPD {
                    cdf.0 = f;
                }
                return 15;
            }
        };
        let v0 = ((r.wrapping_mul(((f as u32) >> 6) & 0x3ff)) >> 1).wrapping_add(12);
        let v1 = ((r.wrapping_mul((f as u32) >> 22)) >> 1).wrapping_add(8);
        let c0 = lt(c, v0);
        let c1 = lt(c, v1);
        let val0 = c0 as u32;
        let u0 = sel(c0, v0, self.rng);
        let vv0 = sel(c0, v1, v0);

        let tok = base.wrapping_add(val0).wrapping_add(c1 as u32);
        let u = sel(c1, v1, u0);
        let v = sel(c1, v2, vv0);
        let new_dif = self.dif.wrapping_sub((v as u64) << (WIN_SIZE - 16));
        let new_rng = u.wrapping_sub(v);
        if UPD {
            let m1 = sel(c1, 0x0001_0001u64, val0 as u64);
            cdf.0 = adapt5_m(f, m1);
        }
        self.norm_hot(new_dif, new_rng);
        tok
    }

    /// Dedicated hot symbol helper for palette decoding.
    #[inline(always)]
    pub fn symbol_palette_hot<const N: usize, const UPD: bool>(
        &mut self,
        cdf: &mut [u16; N],
    ) -> usize {
        const { assert!(N >= 3 && N <= 9) };
        if N == 3 {
            let c3: &mut [u16; 3] = cdf.first_chunk_mut().unwrap();
            use std::hint::select_unpredictable as sel;
            let p0 = c3[0] as u32;
            let count = if UPD { c3[2] } else { 0 };
            let f = 32768 - p0;
            let r = self.rng;
            let dif = self.dif;
            let v = (((r >> 8).wrapping_mul(f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
                .wrapping_add(EC_MIN_PROB);
            let vw = (v as u64) << (WIN_SIZE - 16);
            let ret = dif >= vw;
            let new_dif = sel(ret, dif.wrapping_sub(vw), dif);
            let new_rng = sel(ret, r.wrapping_sub(v), v);
            if UPD {
                let rate = 4 + ((count >> 4) & 3) as u32;
                c3[0] = sel(ret, p0.wrapping_add(f >> rate), p0.wrapping_sub(p0 >> rate)) as u16;
                c3[2] = inc_count(count);
            }
            self.norm_hot(new_dif, new_rng);
            (!ret) as usize
        } else if N == 4 {
            self.symbol4_c::<UPD, true>(cdf.first_chunk_mut().unwrap()) as usize
        } else if N == 5 {
            let c5: &mut [u16; 5] = cdf.first_chunk_mut().unwrap();
            let count = if UPD { c5[4] } else { 0 };
            let (val, y) = self.symbol5_r::<true>(Cdf5::load_spec(c5), UPD);
            if UPD {
                y.store_spec_f(c5);
                c5[4] = inc_count(count);
            }
            val as usize
        } else if N == 6 {
            self.symbol6_c::<UPD, true>(cdf.first_chunk_mut().unwrap())
        } else {
            #[cfg(feature = "native_av1_simd")]
            {
                self.simd_symbol_hot::<N, UPD>(cdf)
            }
            #[cfg(not(feature = "native_av1_simd"))]
            {
                self.symbol_generic_upd::<UPD>(cdf)
            }
        }
    }

    #[cfg(feature = "native_av1_simd")]
    #[inline(always)]
    fn simd_symbol_hot<const N: usize, const UPD: bool>(&mut self, cdf: &mut [u16; N]) -> usize {
        simd_symbol_body_hot!(self, cdf, N, 8, u16x8, i8x16, UPD)
    }

    /// `symbol_n` for `N <= 10` with 8 lanes. On x86_64, `u` and `v` of the
    /// decoded symbol are looked up in a stack copy of `[rng, v..]` at the
    /// index given by the compare movemask; on little-endian non-x86_64
    /// targets, `v` and `u - v` are extracted directly in SIMD registers
    /// via `swizzle_dyn` (`tbl` on aarch64).
    #[cfg(feature = "native_av1_simd")]
    #[inline(always)]
    fn symbol_simd8<const N: usize>(&mut self, cdf: &mut [u16; N]) -> usize {
        simd_symbol_body!(self, cdf, 8, u16x8, i8x16)
    }

    /// `symbol_n` for `11 <= N <= 17` with two 8-lane vectors.
    #[cfg(feature = "native_av1_simd")]
    #[inline(always)]
    fn symbol_simd16<const N: usize>(&mut self, cdf: &mut [u16; N]) -> usize {
        use std::simd::prelude::*;
        let n = N - 1;
        let k2 = (n & !1) - 8;
        let c = (self.dif >> (WIN_SIZE - 16)) as u16;
        let (dif0, cnt) = self.prefill_16();
        let r = self.rng;
        let x0 = u16x8::from_array(*cdf.first_chunk::<8>().unwrap());
        let x1 = if N >= 16 {
            u16x8::from_array(*cdf[8..16].first_chunk::<8>().unwrap())
        } else {
            load_partial_u16x8(&cdf[8..], k2)
        };
        let top1 = const {
            let mut t = [0u16; 8];
            let mut i = 0;
            while i < 8 && i + 8 < (N - 2) {
                t[i] = 0x8000;
                i += 1;
            }
            u16x8::from_array(t)
        };
        let f0 = u16x8::splat(0x8000) - x0;
        let f1 = top1.saturating_sub(x1);
        let (mp0, mp1) = const {
            let mut m0 = [0u16; 8];
            let mut m1 = [0u16; 8];
            let mut i = 0;
            while i < 8 {
                if i < N - 2 {
                    m0[i] = (EC_MIN_PROB * (N - 2 - i) as u32) as u16;
                }
                if i + 8 < N - 2 {
                    m1[i] = (EC_MIN_PROB * (N - 2 - (i + 8)) as u32) as u16;
                }
                i += 1;
            }
            (u16x8::from_array(m0), u16x8::from_array(m1))
        };
        #[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
        let (ge0, ge1, new_rng, vv, val) = {
            let mask_ffc0 = u16x8::splat(0xffc0);
            let rr = u16x8::splat((r & 0xff00) as u16).cast::<u32>();
            let fm0 = (f0 & mask_ffc0) << 1;
            let fm1 = (f1 & mask_ffc0) << 1;
            let v0 = ((fm0.cast::<u32>() * rr) >> 16).cast::<u16>() + mp0;
            let v1 = ((fm1.cast::<u32>() * rr) >> 16).cast::<u16>() + mp1;
            let vc = u16x8::splat(c);
            let ge0 = vc.simd_ge(v0);
            let ge1 = vc.simd_ge(v1);
            let b0 = i8x16::from_ne_bytes(ge0.to_simd().cast::<u16>().to_ne_bytes());
            let b1 = i8x16::from_ne_bytes(ge1.to_simd().cast::<u16>().to_ne_bytes());
            let m0 = b0.simd_lt(i8x16::splat(0)).to_bitmask() as u32;
            let m1 = b1.simd_lt(i8x16::splat(0)).to_bitmask() as u32;
            let mask = m0 | (m1 << 16) | (1u32 << (2 * (N - 2)));
            let val = (mask.trailing_zeros() >> 1) as usize;
            let mut vs = [0u16; 16];
            vs[..8].copy_from_slice(v0.as_array());
            vs[8..].copy_from_slice(v1.as_array());
            let u =
                std::hint::select_unpredictable(val == 0, r, vs[val.wrapping_sub(1) & 15] as u32);
            let vv = vs[val & 15] as u32;
            (ge0, ge1, u.wrapping_sub(vv), vv, val)
        };
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        let (ge0, ge1, new_rng, vv, val) = {
            let v0 = simd_thresh8(f0, r, mp0);
            let v1 = simd_thresh8(f1, r, mp1);
            let vc = u16x8::splat(c);
            let ge0 = vc.simd_ge(v0);
            let ge1 = vc.simd_ge(v1);
            let u0 =
                std::simd::simd_swizzle!(u16x8::splat(r as u16), v0, [7, 8, 9, 10, 11, 12, 13, 14]);
            let u1 = std::simd::simd_swizzle!(v0, v1, [7, 8, 9, 10, 11, 12, 13, 14]);
            let rng0 = u0 - v0;
            let rng1 = u1 - v1;
            let sum = (i8x16::from_ne_bytes(ge0.to_simd().to_ne_bytes())
                + i8x16::from_ne_bytes(ge1.to_simd().to_ne_bytes()))
            .reduce_sum();
            let val = (((sum as u8) >> 1) & 15) as usize;
            let idx = u8x32::splat(sum as u8)
                + u8x32::from_array([
                    32, 33, 32, 33, 32, 33, 32, 33, 32, 33, 32, 33, 32, 33, 32, 33, 32, 33, 32, 33,
                    32, 33, 32, 33, 32, 33, 32, 33, 32, 33, 32, 33,
                ]);
            let v_cat = std::simd::simd_swizzle!(
                v0.to_ne_bytes(),
                v1.to_ne_bytes(),
                [
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                    22, 23, 24, 25, 26, 27, 28, 29, 30, 31
                ]
            );
            let r_cat = std::simd::simd_swizzle!(
                rng0.to_ne_bytes(),
                rng1.to_ne_bytes(),
                [
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                    22, 23, 24, 25, 26, 27, 28, 29, 30, 31
                ]
            );
            let v_sel = v_cat.swizzle_dyn(idx);
            let r_sel = r_cat.swizzle_dyn(idx);
            let vv = u16x16::from_ne_bytes(v_sel)[0] as u32;
            let new_rng = u16x16::from_ne_bytes(r_sel)[0] as u32;
            (ge0, ge1, new_rng, vv, val)
        };
        let new_dif = dif0.wrapping_sub((vv as u64) << (WIN_SIZE - 16));
        let d = (new_rng.leading_zeros() ^ 16) as i32;
        debug_assert!(cnt >= d);
        self.dif = new_dif << ((d as u32) & 63);
        self.rng = new_rng << ((d as u32) & 31);
        self.cnt = cnt.wrapping_sub(d);
        if self.allow_update {
            let gm0 = ge0.to_simd().cast::<u16>();
            let gm1 = ge1.to_simd().cast::<u16>();
            if N == 16 {
                let rate = u16x8::splat(5 + ((x1[7] >> 4) & 3));
                let d0 = ge0.select(f0, x0) >> rate;
                let d1 = ge1.select(f1, x1) >> rate;
                let y0 = (x0 + gm0) - (d0 ^ gm0);
                let y1 = (x1 + gm1) - (d1 ^ gm1);
                let inc = y1
                    .simd_lt(u16x8::from_array([0, 0, 0, 0, 0, 0, 0, 32]))
                    .to_simd()
                    .cast::<u16>();
                let y1 = y1 - inc;
                *cdf.first_chunk_mut::<8>().unwrap() = *y0.as_array();
                *cdf[8..16].first_chunk_mut::<8>().unwrap() = *y1.as_array();
            } else {
                let count = cdf[n];
                let rate = u16x8::splat(5 + ((count >> 4) & 3));
                let d0 = ge0.select(f0, x0) >> rate;
                let d1 = ge1.select(f1, x1) >> rate;
                let y0 = (x0 + gm0) - (d0 ^ gm0);
                let y1 = (x1 + gm1) - (d1 ^ gm1);
                *cdf.first_chunk_mut::<8>().unwrap() = *y0.as_array();
                if N > 16 {
                    *cdf[8..16].first_chunk_mut::<8>().unwrap() = *y1.as_array();
                } else {
                    cdf[8..8 + k2].copy_from_slice(&y1.as_array()[..k2]);
                }
                cdf[n] = inc_count(count);
            }
        }
        val
    }

    /// Coefficient `coeff_br` loop: starting from level 3, adds up to four
    /// `coeff_br` symbols (stopping after one below 3). Returns the level.
    #[inline(always)]
    pub fn hi_tok(&mut self, cdf: &mut CdfRow5) -> u32 {
        if self.allow_update {
            self.hi_tok_c::<true>(cdf)
        } else {
            self.hi_tok_c::<false>(cdf)
        }
    }

    /// Golomb code used for coefficient levels above 14: returns the
    /// decoded value plus one (the spec's `x`), clamping the prefix
    /// length at 32 bits like `dav1d`.
    #[inline(always)]
    pub fn golomb_coef(&mut self) -> Option<u32> {
        if self.cnt >= 1 {
            let v2 = (self.rng & 0xff00).wrapping_add(2 * EC_MIN_PROB);
            let vw = (v2 as u64) << (WIN_SIZE - 17);
            if self.dif < vw {
                self.dif <<= 1;
                self.rng = v2;
                self.cnt = self.cnt.wrapping_sub(1);
                return Some(1);
            }
        }
        let (dif, out) =
            golomb_coef_slow::<0>(self.raw.buf, self.raw.pos, self.dif, self.rng, self.cnt);
        self.dif = dif;
        self.rng = (out & 0xFFFF) as u32;
        self.cnt = ((out >> 16) as i8) as i32;
        self.raw.pos = self.raw.pos.wrapping_add(((out >> 24) & 0xFF) as usize);
        Some((out >> 32) as u32)
    }

    /// Reads an equiprobable sign bit followed by `golomb_coef` and
    /// dequantizes an AC coefficient with `tok == 15`. Returns `(q | (1 << 31), dq)`
    /// or `(0, 0)` on an invalid stream.
    #[inline(always)]
    pub fn ac_golomb(&mut self, q2: u32, dq_shift: u32, cf_max: u64) -> (u32, i32) {
        let (dif, out) =
            sign_golomb_slow::<0>(self.raw.buf, self.raw.pos, self.dif, self.rng, self.cnt);
        self.dif = dif;
        self.rng = (out & 0xFFFF) as u32;
        self.cnt = ((out >> 16) as i8) as i32;
        self.raw.pos = self.raw.pos.wrapping_add(((out >> 24) & 0xFF) as usize);
        let hi = (out >> 32) as u32;
        if hi == 0 {
            return (0, 0);
        }
        let sign = (hi >> 30) & 1;
        let q = hi & 0xFFFFF;
        let dq = (((q as u64).wrapping_mul(q2 as u64) & 0xFFFFFF) >> (dq_shift & 31))
            .min(cf_max.wrapping_add(sign as u64)) as i32;
        let dq_signed = std::hint::select_unpredictable(sign != 0, dq.wrapping_neg(), dq);
        (q | (1 << 31), dq_signed)
    }

    fn symbol_generic(&mut self, cdf: &mut [u16]) -> usize {
        if self.allow_update {
            self.symbol_generic_upd::<true>(cdf)
        } else {
            self.symbol_generic_upd::<false>(cdf)
        }
    }

    fn symbol_generic_upd<const UPD: bool>(&mut self, cdf: &mut [u16]) -> usize {
        let n = cdf.len() - 1;
        debug_assert!(n >= 2 && cdf[n - 1] == 32768);
        let c = (self.dif >> (WIN_SIZE - 16)) as u32;
        let r = self.rng >> 8;
        let mut u;
        let mut v = self.rng;
        let mut val = 0usize;
        loop {
            u = v;
            let f = 32768 - cdf[val] as u32;
            v = ((r * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
                + EC_MIN_PROB * (n - 1 - val) as u32;
            if c >= v {
                break;
            }
            val += 1;
        }
        self.norm(self.dif - ((v as u64) << (WIN_SIZE - 16)), u - v);
        if UPD {
            let count = cdf[n];
            let rate = 3
                + (count > 15) as u32
                + (count > 31) as u32
                + (crate::decoder::av1::pixel::floor_log2(n as u32)).min(2);
            for (i, p) in cdf[..n - 1].iter_mut().enumerate() {
                if i < val {
                    *p -= *p >> rate;
                } else {
                    *p += (32768 - *p) >> rate;
                }
            }
            cdf[n] = count + (count < 32) as u16;
        }
        val
    }

    /// `read_symbol` for a binary CDF (`[p, 32768, count]`).
    ///
    /// Branch free like dav1d's `cmov` based asm: the decoded bit is
    /// unpredictable, so a branch on it costs a mispredict half the time.
    #[inline(always)]
    pub fn bool_adapt(&mut self, cdf: &mut [u16; 3]) -> bool {
        use std::hint::select_unpredictable as sel;
        let p0 = cdf[0] as u32;
        let f = 32768 - p0;
        let r = self.rng;
        let dif = self.dif;
        let v = (((r >> 8).wrapping_mul(f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
            .wrapping_add(EC_MIN_PROB);
        let vw = (v as u64) << (WIN_SIZE - 16);
        let ret = dif >= vw;
        let new_dif = sel(ret, dif.wrapping_sub(vw), dif);
        let new_rng = sel(ret, r.wrapping_sub(v), v);
        if self.allow_update {
            let count = cdf[2];
            let rate = 4 + ((count >> 4) & 3) as u32;
            cdf[0] = sel(ret, p0.wrapping_add(f >> rate), p0.wrapping_sub(p0 >> rate)) as u16;
            cdf[2] = inc_count(count);
        }
        self.norm(new_dif, new_rng);
        !ret
    }

    /// `read_symbol` for a packed 4-byte binary CDF (`CdfRow3`).
    #[inline(always)]
    pub fn bool_adapt3<const HOT: bool>(&mut self, cdf: &mut CdfRow3) -> bool {
        use std::hint::select_unpredictable as sel;
        let w = cdf.0;
        let f = w as u16 as u32;
        let r = self.rng;
        let dif = self.dif;
        let v = (((r >> 8).wrapping_mul(f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
            .wrapping_add(EC_MIN_PROB);
        let vw = (v as u64) << (WIN_SIZE - 16);
        let ret = dif >= vw;
        let new_dif = sel(ret, dif.wrapping_sub(vw), dif);
        let new_rng = sel(ret, r.wrapping_sub(v), v);
        if self.allow_update {
            let rate = 4 + (w >> 30);
            let delta = sel(ret, f, 32768 - f) >> rate;
            let inc = (!w >> 5) & (1 << 26);
            cdf.0 = sel(ret, w.wrapping_sub(delta), w.wrapping_add(delta)).wrapping_add(inc);
        }
        if HOT {
            self.norm_hot(new_dif, new_rng);
        } else {
            self.norm(new_dif, new_rng);
        }
        !ret
    }

    /// Spec `read_bool()` (equiprobable, no adaptation). Only used by the
    /// tests: the decoder uses `literal`, `bool_cold` and the
    /// `bool_store_neg*` variants.
    #[cfg(test)]
    #[inline(always)]
    pub fn bool(&mut self) -> bool {
        use std::hint::select_unpredictable as sel;
        let r = self.rng;
        let dif = self.dif;
        let v = ((r >> 8) << 7).wrapping_add(EC_MIN_PROB);
        let vw = (v as u64) << (WIN_SIZE - 16);
        let ret = dif >= vw;
        let new_dif = sel(ret, dif.wrapping_sub(vw), dif);
        let new_rng = sel(ret, r.wrapping_sub(v), v);
        self.norm_hot(new_dif, new_rng);
        !ret
    }

    /// Reads an equiprobable sign bit and stores `if sign { -d } else { d }`
    /// to `*dst` before renormalization so `dif`, `vw`, and `d` do not stay
    /// live across `refill`.
    #[inline(always)]
    pub fn bool_store_neg(&mut self, dst: &mut i32, d: i32) {
        let (ret, new_rng, new_dif, shift) = Self::equi_step(self.rng, self.dif);
        *dst = std::hint::select_unpredictable(ret, d, d.wrapping_neg());
        let cnt = self.cnt;
        self.dif = new_dif;
        self.rng = new_rng;
        self.cnt = cnt.wrapping_sub(shift as i32);
        if (cnt as u32) < shift {
            cold_refill();
            self.refill_hot();
        }
    }

    /// Core equiprobable `read_bool()` step; returns `(ret, new_rng, new_dif, shift)`
    /// where `ret` is `true` for a 0 bit (`dif >= vw`). For `rng` in `0x8000..=0xffff`:
    /// - 1-bit outcome (`rng = v`) is in `0x4004..=0x7f84`, so `shift == 1` and
    ///   `v << 1 == (rng & 0xff00) + 8`.
    /// - 0-bit outcome (`rng = r - v`) shifts by `(r < 0xff84) + (r < 0x8004)`,
    ///   computed directly from `r` in parallel with `dif >= vw` so `clz`
    ///   is off the serial `rng` dependency chain.
    #[inline(always)]
    fn equi_step(r: u32, dif: u64) -> (bool, u32, u64, u32) {
        use std::hint::select_unpredictable as sel;
        debug_assert!((0x8000..=0xffff).contains(&r));
        let v2 = (r & 0xff00).wrapping_add(2 * EC_MIN_PROB);
        let vw = (v2 as u64) << (WIN_SIZE - 17);
        let r0 = r.wrapping_sub(v2 >> 1);
        let d0 = ((r < 0xff84) as u32).wrapping_add((r < 0x8004) as u32);
        let ret = dif >= vw;
        let new_rng = sel(ret, r0 << (d0 & 31), v2);
        let shift = sel(ret, d0, 1);
        let new_dif = sel(ret, dif.wrapping_sub(vw) << (d0 & 63), dif << 1);
        (ret, new_rng, new_dif, shift)
    }

    #[inline(always)]
    pub(crate) fn equi_locked(&self) -> bool {
        (self.rng & 0xff) == 8
    }

    #[inline(always)]
    pub(crate) fn equi_recip(&self) -> u64 {
        EQUI_RECIP[((self.rng >> 8) & 127) as usize] as u64
    }

    #[inline(always)]
    pub(crate) fn equi_r_sh48(&self) -> u64 {
        (self.rng as u64) << 48
    }

    #[inline(always)]
    pub(crate) fn equi_peek16_with(&mut self, recip: u64) -> u32 {
        if self.cnt < 16 {
            cold_refill();
            self.refill_hot();
        }
        (((self.dif >> 33).wrapping_mul(recip)) >> 46) as u32
    }

    #[inline(always)]
    pub(crate) fn equi_consume_with(&mut self, t16: u32, n: u32, r_sh48: u64) {
        let t = (t16 >> (16u32.wrapping_sub(n) & 31)) as u64;
        self.dif = (self.dif << (n & 63)).wrapping_sub(t.wrapping_mul(r_sh48));
        self.cnt = self.cnt.wrapping_sub(n as i32);
    }

    /// Equiprobable `read_literal(n)` for `1 <= n <= 8` (`eob` extra bits).
    /// Runs plain `equi_step` per bit instead of branching on `(r & 0xff) == 8`
    /// to enter `EQUI_RECIP` (each step locks with ~50% probability, so checking
    /// for lock mispredicts on almost every call). Across `n <= 8` bits, total
    /// renormalization shift is at most `n + 1 <= 9`, so one `cnt < 9` check
    /// suffices.
    #[inline(always)]
    pub fn literal_short(&mut self, n: usize) -> usize {
        debug_assert!((1..=8).contains(&n));
        if self.cnt < 9 {
            cold_refill();
            self.refill_hot();
        }
        let mut r = self.rng;
        let mut dif = self.dif;
        let mut cnt = self.cnt;
        let mut x = 0usize;
        for _ in 0..n {
            let (ret, nr, nd, shift) = Self::equi_step(r, dif);
            r = nr;
            dif = nd;
            cnt = cnt.wrapping_sub(shift as i32);
            x = (x << 1) | (!ret as usize);
        }
        self.dif = dif;
        self.rng = r;
        self.cnt = cnt;
        x
    }

    /// Like `bool_store_neg`, clamping `d` to `cf_max + sign` before negating.
    #[inline(always)]
    pub fn bool_store_neg_qm(&mut self, dst: &mut i32, d: u32, cf_max: u32) {
        let pos = d.min(cf_max) as i32;
        let neg = (d.min(cf_max.wrapping_add(1)) as i32).wrapping_neg();
        let (ret, new_rng, new_dif, shift) = Self::equi_step(self.rng, self.dif);
        *dst = std::hint::select_unpredictable(ret, pos, neg);
        let cnt = self.cnt;
        self.dif = new_dif;
        self.rng = new_rng;
        self.cnt = cnt.wrapping_sub(shift as i32);
        if (cnt as u32) < shift {
            cold_refill();
            self.refill_hot();
        }
    }

    /// Spec `read_literal(n)` / `L(n)` (`n <= 32`).
    #[inline]
    pub fn literal(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        debug_assert!(n <= 32);
        let mut dif = self.dif;
        let mut cnt = self.cnt;
        if cnt <= n as i32 {
            cold_refill();
            let (d_out, out) = refill_cold::<0>(self.raw.buf, self.raw.pos, dif, cnt);
            dif = d_out;
            self.raw.pos = self.raw.pos.wrapping_add((out as u32) as usize);
            cnt = (out >> 32) as i32;
        }
        let mut r = self.rng;
        let mut x = 0;
        for _ in 0..n {
            x = (x << 1) | equi_nobuf(&mut dif, &mut r, &mut cnt) as u32;
        }
        self.dif = dif;
        self.rng = r;
        self.cnt = cnt;
        x
    }

    /// Spec `NS(n)` coded with the symbol decoder.
    pub fn ns(&mut self, n: u32) -> u32 {
        if n <= 1 {
            return 0;
        }
        let w = crate::decoder::av1::pixel::floor_log2(n) + 1;
        let m = (1 << w) - n;
        let v = self.literal(w - 1);
        if v < m {
            return v;
        }
        let extra = self.literal(1);
        (v << 1) - m + extra
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct bit-serial transcription of spec 8.2 for cross checking.
    struct SpecDecoder<'a> {
        data: &'a [u8],
        bitpos: usize,
        value: u32,
        range: u32,
        max_bits: i64,
    }

    impl<'a> SpecDecoder<'a> {
        fn f(&mut self, n: u32) -> u32 {
            let mut x = 0;
            for _ in 0..n {
                let b = self
                    .data
                    .get(self.bitpos >> 3)
                    .map(|b| (b >> (7 - (self.bitpos & 7))) & 1)
                    .unwrap_or(0);
                self.bitpos += 1;
                x = (x << 1) | b as u32;
            }
            x
        }
        fn new(data: &'a [u8]) -> Self {
            let sz = data.len() as i64;
            let mut s = Self {
                data,
                bitpos: 0,
                value: 0,
                range: 1 << 15,
                max_bits: 8 * sz - 15,
            };
            let num_bits = (sz * 8).min(15) as u32;
            let buf = s.f(num_bits);
            let padded = buf << (15 - num_bits);
            s.value = ((1 << 15) - 1) ^ padded;
            s
        }
        fn symbol(&mut self, cdf: &mut [u16], update: bool) -> usize {
            let n = cdf.len() - 1;
            let mut cur = self.range;
            let mut symbol: i32 = -1;
            let mut prev;
            loop {
                symbol += 1;
                prev = cur;
                let f = (1 << 15) - cdf[symbol as usize] as u32;
                cur = ((self.range >> 8) * (f >> 6)) >> 1;
                cur += 4 * (n as u32 - symbol as u32 - 1);
                if self.value >= cur {
                    break;
                }
            }
            self.range = prev - cur;
            self.value -= cur;
            let bits = 15 - crate::decoder::av1::pixel::floor_log2(self.range);
            self.range <<= bits;
            let num_bits = (bits as i64).min(self.max_bits.max(0)) as u32;
            let new_data = self.f(num_bits);
            let padded = new_data << (bits - num_bits);
            self.value = padded ^ (((self.value + 1) << bits) - 1);
            self.max_bits -= bits as i64;
            if update {
                let cnt = cdf[n];
                let rate = 3
                    + (cnt > 15) as u32
                    + (cnt > 31) as u32
                    + crate::decoder::av1::pixel::floor_log2(n as u32).min(2);
                let mut tmp = 0u32;
                for i in 0..n - 1 {
                    if i as i32 == symbol {
                        tmp = 1 << 15;
                    }
                    let c = cdf[i] as u32;
                    if tmp < c {
                        cdf[i] -= ((c - tmp) >> rate) as u16;
                    } else {
                        cdf[i] += ((tmp - c) >> rate) as u16;
                    }
                }
                cdf[n] += (cdf[n] < 32) as u16;
            }
            symbol as usize
        }

        /// Spec `read_bool()`.
        fn bool(&mut self) -> bool {
            self.symbol(&mut [1 << 14, 1 << 15, 0], false) == 1
        }

        /// The Golomb reader of `golomb_coef`: returns the spec's `x`
        /// (value plus one), clamping the prefix length at 32 bits like `dav1d`.
        fn golomb_coef(&mut self) -> Option<u32> {
            let mut length = 0;
            loop {
                length += 1;
                if self.bool() || length > 32 {
                    let mut x = 1u32;
                    for _ in 0..length - 1 {
                        x = (x << 1) | self.bool() as u32;
                    }
                    return Some(x);
                }
            }
        }
    }

    #[test]
    fn matches_spec_decoder() {
        let mut seed = 12345u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed >> 8
        };
        for len in [0usize, 1, 2, 3, 7, 50, 300] {
            let data: Vec<u8> = (0..len).map(|_| rnd() as u8).collect();
            let mut a = SymbolDecoder::new(&data, false);
            let mut b = SpecDecoder::new(&data);
            assert_eq!(a.symbol_max_bits(), b.max_bits);
            let mut cdfs_a = vec![[8000u16, 16000, 24000, 32768, 0]; 4];
            let mut cdfs_b = cdfs_a.clone();
            for i in 0..2000 {
                let k = i % 4;
                match rnd() % 3 {
                    0 => {
                        let sa = a.symbol(&mut cdfs_a[k]);
                        let sb = b.symbol(&mut cdfs_b[k], true);
                        assert_eq!(sa, sb, "len {len} iter {i}");
                    }
                    1 => {
                        let sa = a.bool() as usize;
                        let sb = b.symbol(&mut [1 << 14, 1 << 15, 0], false);
                        assert_eq!(sa, sb);
                    }
                    _ => {
                        let mut ca = [12000u16, 32768, (i % 40) as u16];
                        let mut cb = ca;
                        let sa = a.bool_adapt(&mut ca) as usize;
                        let sb = b.symbol(&mut cb, true);
                        assert_eq!(sa, sb);
                        assert_eq!(ca, cb);
                    }
                }
                assert_eq!(cdfs_a[k], cdfs_b[k]);
                assert_eq!(a.symbol_max_bits(), b.max_bits, "len {len} iter {i}");
            }
        }
    }

    /// Exercises every CDF size (fast path) against the spec decoder,
    /// plus `hi_tok` and the golomb reader.
    #[test]
    fn symbol_sizes_match_spec() {
        let mut seed = 777u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed >> 8
        };
        fn mk(n: usize, rnd: &mut dyn FnMut() -> u32) -> Vec<u16> {
            let mut v: Vec<u16> = (0..n - 1).map(|_| (1 + rnd() % 32767) as u16).collect();
            v.sort();
            v.push(32768);
            v.push((rnd() % 33) as u16);
            v
        }
        for len in [0usize, 1, 5, 9, 17, 64, 1000] {
            let data: Vec<u8> = (0..len).map(|_| rnd() as u8).collect();
            for update in [false, true] {
                let mut a = SymbolDecoder::new(&data, !update);
                let mut b = SpecDecoder::new(&data);
                let mut cdfs: Vec<Vec<u16>> = (2..=16).map(|n| mk(n, &mut rnd)).collect();
                let mut cdfs_b = cdfs.clone();
                let mut br_b = [8000u16, 16000, 24000, 32768, 0];
                let mut br_a = CdfRow5::from_spec(br_b);
                for i in 0..3000 {
                    match rnd() % 4 {
                        0 | 1 => {
                            let k = (rnd() % 15) as usize;
                            let sa = a.symbol(&mut cdfs[k]);
                            let sb = b.symbol(&mut cdfs_b[k], update);
                            assert_eq!(sa, sb, "len {len} iter {i} n {}", k + 2);
                            assert_eq!(cdfs[k], cdfs_b[k]);
                        }
                        2 => {
                            let ta = a.hi_tok(&mut br_a);
                            let mut tb = 3;
                            for _ in 0..4 {
                                let s = b.symbol(&mut br_b, update) as u32;
                                tb += s;
                                if s < 3 {
                                    break;
                                }
                            }
                            assert_eq!(ta, tb);
                            assert_eq!(br_a, br_b);
                        }
                        _ => {
                            let ga = a.golomb_coef();
                            let gb = b.golomb_coef();
                            assert_eq!(ga, gb);
                        }
                    }
                    assert_eq!(a.symbol_max_bits(), b.max_bits, "len {len} iter {i}");
                }
            }
        }
    }

    /// Exercises `literal`, `ac_golomb`, `bool_store_neg[_qm]`,
    /// `symbol4_c` and `symbol5_c` against the spec decoder, with short
    /// buffers so that the refill tail past the end of the data is hit.
    #[test]
    fn specialized_readers_match_spec() {
        let mut seed = 4242u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed >> 8
        };
        for len in [0usize, 1, 2, 7, 9, 33, 400] {
            let data: Vec<u8> = (0..len).map(|_| rnd() as u8).collect();
            for update in [false, true] {
                let mut a = SymbolDecoder::new(&data, !update);
                let mut b = SpecDecoder::new(&data);
                let mut c4a = [10000u16, 22000, 32768, 0];
                let mut c4b = c4a;
                let mut c5b = [8000u16, 16000, 24000, 32768, 0];
                let mut c5a = CdfRow5::from_spec(c5b);
                let mut c5_coef_b = [6000u16, 14000, 25000, 32768, 0];
                let mut c5_coef_a = CdfRow5::from_spec(c5_coef_b);
                let mut br_b = [14000u16, 20000, 24000, 32768, 0];
                let mut br_a = CdfRow5::from_spec(br_b);
                let mut c3b = [12000u16, 32768, 0];
                let mut c3a = CdfRow3::from_spec(c3b);
                let mut c6a = [5000u16, 12000, 20000, 28000, 32768, 0];
                let mut c6b = c6a;
                let mut levels = [0u8; 2200];
                let mut nz = [0u16; 1024];
                let mut k = 0usize;
                for i in 0..2000 {
                    match rnd() % 11 {
                        0 => {
                            let n = 1 + rnd() % 32;
                            let la = a.literal(n);
                            let mut lb = 0u32;
                            for _ in 0..n {
                                lb = (lb << 1) | b.bool() as u32;
                            }
                            assert_eq!(la, lb, "len {len} iter {i} n {n}");
                        }
                        1 => {
                            let q2 = 1 + rnd() % 30000;
                            let dq_shift = rnd() % 3;
                            // (1 << (7 + bd)) - 1 for bd 8, 10, 12.
                            let cf_max = [(1u64 << 15) - 1, (1 << 17) - 1, (1 << 19) - 1]
                                [(rnd() % 3) as usize];
                            let (qa, dqa) = a.ac_golomb(q2, dq_shift, cf_max);
                            let sign = b.bool();
                            let (qb, dqb) = match b.golomb_coef() {
                                None => (0, 0),
                                Some(x) => {
                                    let q = x.wrapping_add(14) & 0xFFFFF;
                                    let dq =
                                        (((q as u64 * q2 as u64) & 0xFFFFFF) >> dq_shift) as i64;
                                    let dq = if sign { -dq } else { dq };
                                    let dq = dq.clamp(-(cf_max as i64) - 1, cf_max as i64);
                                    (q | (1 << 31), dq as i32)
                                }
                            };
                            assert_eq!((qa, dqa), (qb, dqb), "len {len} iter {i}");
                        }
                        2 => {
                            let d = (rnd() % 100000) as i32;
                            let mut dst = 0;
                            a.bool_store_neg(&mut dst, d);
                            let sign = b.bool();
                            assert_eq!(dst, if sign { -d } else { d }, "len {len} iter {i}");
                        }
                        3 => {
                            let d = rnd() % 600000;
                            let cf_max = [(1u32 << 15) - 1, (1 << 17) - 1, (1 << 19) - 1]
                                [(rnd() % 3) as usize];
                            let mut dst = 0;
                            a.bool_store_neg_qm(&mut dst, d, cf_max);
                            let sign = b.bool();
                            let expected = if sign {
                                -(d.min(cf_max + 1) as i32)
                            } else {
                                d.min(cf_max) as i32
                            };
                            assert_eq!(dst, expected, "len {len} iter {i}");
                        }
                        4 => {
                            let hot = rnd() % 2 == 0;
                            let sa = match (update, hot) {
                                (false, false) => a.symbol4_c::<false, false>(&mut c4a),
                                (false, true) => a.symbol4_c::<false, true>(&mut c4a),
                                (true, false) => a.symbol4_c::<true, false>(&mut c4a),
                                (true, true) => a.symbol4_c::<true, true>(&mut c4a),
                            };
                            let sb = b.symbol(&mut c4b, update);
                            assert_eq!(sa as usize, sb, "len {len} iter {i}");
                            assert_eq!(c4a, c4b);
                        }
                        5 => {
                            let sa = if update {
                                a.symbol5_c::<true>(&mut c5a)
                            } else {
                                a.symbol5_c::<false>(&mut c5a)
                            };
                            let sb = b.symbol(&mut c5b, update);
                            assert_eq!(sa as usize, sb, "len {len} iter {i}");
                            assert_eq!(c5a, c5b);
                        }
                        6 => {
                            let idx = (rnd() % 1800) as usize;
                            let pos = (rnd() % 1024) as usize;
                            let k_before = k;
                            let (is_hi, f) = if update {
                                a.symbol5_coef::<true>(
                                    c5_coef_a.0,
                                    &mut levels,
                                    &mut nz,
                                    idx,
                                    pos,
                                    &mut k,
                                )
                            } else {
                                a.symbol5_coef::<false>(
                                    c5_coef_a.0,
                                    &mut levels,
                                    &mut nz,
                                    idx,
                                    pos,
                                    &mut k,
                                )
                            };
                            c5_coef_a.0 = f;
                            let sb = b.symbol(&mut c5_coef_b, update) as u32;
                            assert_eq!(is_hi, sb == 3, "len {len} iter {i}");
                            assert_eq!(c5_coef_a, c5_coef_b, "len {len} iter {i}");
                            assert_eq!(levels[idx & 2047], (sb * 0x41) as u8);
                            assert_eq!(nz[k_before & 1023], ((sb as u16) << 10) | (pos as u16));
                            assert_eq!(k, k_before + (sb != 0) as usize);
                        }
                        7 => {
                            let ta = if update {
                                a.hi_tok_c::<true>(&mut br_a)
                            } else {
                                a.hi_tok_c::<false>(&mut br_a)
                            };
                            let mut tb = 3;
                            for _ in 0..4 {
                                let s = b.symbol(&mut br_b, update) as u32;
                                tb += s;
                                if s < 3 {
                                    break;
                                }
                            }
                            assert_eq!(ta, tb, "len {len} iter {i}");
                            assert_eq!(br_a, br_b, "len {len} iter {i}");
                        }
                        8 => {
                            let hot = rnd() % 2 == 0;
                            let ba = if hot {
                                a.bool_adapt3::<true>(&mut c3a)
                            } else {
                                a.bool_adapt3::<false>(&mut c3a)
                            };
                            let bb = b.symbol(&mut c3b, update) != 0;
                            assert_eq!(ba, bb, "len {len} iter {i}");
                            assert_eq!(c3a, c3b, "len {len} iter {i}");
                        }
                        9 => {
                            let n = 1 + (rnd() % 8) as usize;
                            let la = a.literal_short(n);
                            let mut lb = 0usize;
                            for _ in 0..n {
                                lb = (lb << 1) | b.bool() as usize;
                            }
                            assert_eq!(la, lb, "literal_short len {len} iter {i} n {n}");
                        }
                        _ => {
                            let sa = if update {
                                a.symbol6_c::<true, true>(&mut c6a)
                            } else {
                                a.symbol6_c::<false, false>(&mut c6a)
                            };
                            let sb = b.symbol(&mut c6b, update);
                            assert_eq!(sa, sb, "len {len} iter {i}");
                            assert_eq!(c6a, c6b, "len {len} iter {i}");
                        }
                    }
                    assert_eq!(a.symbol_max_bits(), b.max_bits, "len {len} iter {i}");
                }
            }
        }
        // Explicitly test `equi_peek16_with` + `equi_consume_with` (including
        // short/EOF buffers 0, 1, 2, 7 bytes) and `golomb_coef` against `SpecDecoder`.
        for len in [0usize, 1, 2, 7, 15, 64, 256] {
            let data: Vec<u8> = (0..len).map(|_| rnd() as u8).collect();
            let mut a = SymbolDecoder::new(&data, false);
            let mut b = SpecDecoder::new(&data);
            // Drive until `equi_locked()` becomes true.
            while !a.equi_locked() {
                let mut dst = 0;
                a.bool_store_neg(&mut dst, 1);
                let sb = b.bool();
                assert_eq!(dst, if sb { -1 } else { 1 });
            }
            let recip = a.equi_recip();
            let r_sh48 = a.equi_r_sh48();
            for step in 0..64 {
                let n = 1 + (rnd() % 16);
                let t16 = a.equi_peek16_with(recip);
                let mut sbits = (t16 as i32) << 16;
                for bit in 0..n {
                    let expected = b.bool();
                    let got = sbits >= 0;
                    assert_eq!(got, expected, "locked len {len} step {step} bit {bit}");
                    sbits = sbits.wrapping_shl(1);
                }
                a.equi_consume_with(t16, n, r_sh48);
                assert_eq!(
                    a.symbol_max_bits(),
                    b.max_bits,
                    "locked len {len} step {step}"
                );
                if step % 7 == 0 {
                    assert_eq!(
                        a.golomb_coef(),
                        b.golomb_coef(),
                        "golomb len {len} step {step}"
                    );
                    assert!(a.equi_locked());
                    assert_eq!(a.symbol_max_bits(), b.max_bits);
                }
            }
        }
    }

    #[test]
    fn long_golomb_and_simd_reduction_match_spec() {
        let mut seed = 99991u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed >> 8
        };
        // Craft streams with long runs of zero bytes so Golomb prefixes span
        // `length = 1..=34` (exercising inline `x == 1`, `length <= 8`, and
        // `*_very_slow` up to and past the 32-bit limit).
        for zero_run in 0..=6usize {
            let mut data = Vec::with_capacity(256);
            for k in 0..64usize {
                for _ in 0..(k % (zero_run + 1)) {
                    data.push(0);
                }
                data.push((1 + (rnd() % 255)) as u8);
            }
            let mut a = SymbolDecoder::new(&data, false);
            let mut b = SpecDecoder::new(&data);
            for i in 0..200 {
                if i % 2 == 0 {
                    assert_eq!(a.golomb_coef(), b.golomb_coef(), "run {zero_run} iter {i}");
                } else {
                    let q2 = 1 + rnd() % 100000;
                    let dq_shift = rnd() % 3;
                    let cf_max = (1u64 << 15) - 1;
                    let (qa, dqa) = a.ac_golomb(q2, dq_shift, cf_max);
                    let sign = b.bool();
                    let (qb, dqb) = match b.golomb_coef() {
                        None => (0, 0),
                        Some(x) => {
                            let q = x.wrapping_add(14) & 0xFFFFF;
                            let dq = (((q as u64 * q2 as u64) & 0xFFFFFF) >> dq_shift) as i64;
                            let dq = if sign { -dq } else { dq };
                            let dq = dq.clamp(-(cf_max as i64) - 1, cf_max as i64);
                            (q | (1 << 31), dq as i32)
                        }
                    };
                    assert_eq!((qa, dqa), (qb, dqb), "run {zero_run} iter {i}");
                }
                assert_eq!(a.symbol_max_bits(), b.max_bits, "run {zero_run} iter {i}");
            }
        }
        // Explicitly verify the non-x86_64 SIMD reduction formulas on all N in 7..=17,
        // including boundary CDFs (`cdf[0] = 1`, duplicate entries, extreme `r` and `c`).
        #[cfg(feature = "native_av1_simd")]
        {
            use std::simd::prelude::*;
            for n in 7..=17usize {
                for trial in 0..200 {
                    let mut cdf = vec![0u16; n];
                    for p in &mut cdf[..n - 2] {
                        *p = if trial == 0 {
                            1
                        } else if trial == 1 {
                            32767
                        } else {
                            (1 + rnd() % 32767) as u16
                        };
                    }
                    cdf[..n - 2].sort();
                    cdf[n - 2] = 32768;
                    cdf[n - 1] = (rnd() % 33) as u16;
                    let r = match trial % 4 {
                        0 => 0x8000u32,
                        1 => 0xffffu32,
                        _ => 0x8000 + (rnd() & 0x7fff),
                    };
                    let c = match trial % 5 {
                        0 => 0u16,
                        1 => 0xffffu16,
                        _ => rnd() as u16,
                    };
                    // Reference scalar search
                    let mut u_ref;
                    let mut vv_ref = r;
                    let mut val_ref = 0usize;
                    loop {
                        u_ref = vv_ref;
                        let f = 32768 - cdf[val_ref] as u32;
                        vv_ref = (((r >> 8) * (f >> 6)) >> 1) + 4 * (n - 2 - val_ref) as u32;
                        if (c as u32) >= vv_ref {
                            break;
                        }
                        val_ref += 1;
                    }
                    let r7 = u16x8::splat(((r >> 8) & 0x7f) as u16);
                    let vr = u16x8::splat(r as u16);
                    let vc = u16x8::splat(c);
                    let mut f_arr = [0u16; 16];
                    let mut mp_arr = [0u16; 16];
                    for i in 0..(n - 2) {
                        f_arr[i] = 32768 - cdf[i];
                        mp_arr[i] = (4 * (n - 2 - i)) as u16;
                    }
                    if n <= 10 {
                        let f0 = u16x8::from_slice(&f_arr[..8]);
                        let mp0 = u16x8::from_slice(&mp_arr[..8]);
                        let v0 = (f0 & u16x8::splat(0xffc0)) + mp0 + (((f0 >> 6) * r7) >> 1);
                        let lt = v0.simd_gt(vc);
                        let lm = lt.to_simd().cast::<u16>();
                        let u = lt.select(v0, vr).reduce_min() as u32;
                        let vv = (v0 & !lm).reduce_max() as u32;
                        let val = lt.to_simd().cast::<i16>().reduce_sum().wrapping_neg() as usize;
                        assert_eq!(
                            (u, vv, val),
                            (u_ref, vv_ref, val_ref),
                            "n={n} r={r:#x} c={c:#x}"
                        );
                        let ge = vc.simd_ge(v0);
                        let sent = if n < 10 { 1u64 << (8 * (n - 2)) } else { 0 };
                        let m64 = u64::from_le_bytes(ge.to_simd().cast::<u8>().to_array()) | sent;
                        let val_m = (m64.trailing_zeros() >> 3) as usize;
                        let vs: [u16; 8] = *v0.as_array();
                        let u_m = std::hint::select_unpredictable(
                            val_m == 0,
                            r,
                            vs[val_m.wrapping_sub(1) & 7] as u32,
                        );
                        let vv_m = if n <= 9 {
                            vs[val_m & 7] as u32
                        } else {
                            std::hint::select_unpredictable(val_m < 8, vs[val_m & 7] as u32, 0)
                        };
                        assert_eq!(
                            (u_m, vv_m, val_m),
                            (u_ref, vv_ref, val_ref),
                            "bitmask n={n} r={r:#x} c={c:#x}"
                        );
                        #[cfg(target_endian = "little")]
                        {
                            let u_vec =
                                std::simd::simd_swizzle!(vr, v0, [7, 8, 9, 10, 11, 12, 13, 14]);
                            let rng_vec = u_vec - v0;
                            let sum = i8x16::from_ne_bytes(ge.to_simd().to_ne_bytes()).reduce_sum();
                            let val_t = if n <= 9 {
                                (((sum as u8) >> 1) & 7) as usize
                            } else {
                                (16u8.wrapping_add(sum as u8) >> 1) as usize
                            };
                            let idx16 = u8x16::splat(sum as u8)
                                + u8x16::from_array([
                                    16, 17, 16, 17, 16, 17, 16, 17, 16, 17, 16, 17, 16, 17, 16, 17,
                                ]);
                            let v_sel = u8x16::from_ne_bytes(v0.to_ne_bytes()).swizzle_dyn(idx16);
                            let vv_t = u16::from_ne_bytes([v_sel[0], v_sel[1]]) as u32;
                            let new_rng_t = if n <= 9 {
                                let r_sel =
                                    u8x16::from_ne_bytes(rng_vec.to_ne_bytes()).swizzle_dyn(idx16);
                                u16::from_ne_bytes([r_sel[0], r_sel[1]]) as u32
                            } else {
                                let rng_tail =
                                    std::simd::simd_swizzle!(v0, [7, 7, 7, 7, 7, 7, 7, 7]);
                                let idx32 = std::simd::simd_swizzle!(
                                    idx16,
                                    [
                                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1,
                                        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                    ]
                                );
                                let r_cat = std::simd::simd_swizzle!(
                                    rng_vec.to_ne_bytes(),
                                    rng_tail.to_ne_bytes(),
                                    [
                                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                    ]
                                );
                                let r_sel = r_cat.swizzle_dyn(idx32);
                                u16x16::from_ne_bytes(r_sel)[0] as u32
                            };
                            assert_eq!(
                                (u_ref.wrapping_sub(vv_ref), vv_t, val_t),
                                (new_rng_t, vv_ref, val_ref),
                                "tbl8 n={n} r={r:#x} c={c:#x}"
                            );
                        }
                    } else {
                        let f0 = u16x8::from_slice(&f_arr[..8]);
                        let f1 = u16x8::from_slice(&f_arr[8..16]);
                        let mp0 = u16x8::from_slice(&mp_arr[..8]);
                        let mp1 = u16x8::from_slice(&mp_arr[8..16]);
                        let v0 = (f0 & u16x8::splat(0xffc0)) + mp0 + (((f0 >> 6) * r7) >> 1);
                        let v1 = (f1 & u16x8::splat(0xffc0)) + mp1 + (((f1 >> 6) * r7) >> 1);
                        let ge0 = vc.simd_ge(v0);
                        let ge1 = vc.simd_ge(v1);
                        let gm0 = ge0.to_simd().cast::<u16>();
                        let gm1 = ge1.to_simd().cast::<u16>();
                        let u = ge0.select(vr, v0).simd_min(ge1.select(vr, v1)).reduce_min() as u32;
                        let vv = (v0 & gm0).simd_max(v1 & gm1).reduce_max() as u32;
                        let val = (16i32
                            + (ge0.to_simd().cast::<i16>() + ge1.to_simd().cast::<i16>())
                                .reduce_sum() as i32) as u32
                            as usize;
                        assert_eq!(
                            (u, vv, val),
                            (u_ref, vv_ref, val_ref),
                            "n={n} r={r:#x} c={c:#x}"
                        );
                        let u0 = u32x4::from_ne_bytes(ge0.to_simd().cast::<u16>().to_ne_bytes());
                        let u1 = u32x4::from_ne_bytes(ge1.to_simd().cast::<u16>().to_ne_bytes());
                        let u8s = std::simd::simd_swizzle!(u0, u1, [0, 1, 2, 3, 4, 5, 6, 7]);
                        let m64 = u64::from_le_bytes((u8s >> 12).cast::<u8>().to_array())
                            | (1u64 << (4 * (n - 2)));
                        let val_m = (m64.trailing_zeros() >> 2) as usize;
                        let mut vs = [0u16; 16];
                        vs[..8].copy_from_slice(v0.as_array());
                        vs[8..].copy_from_slice(v1.as_array());
                        let u_m = std::hint::select_unpredictable(
                            val_m == 0,
                            r,
                            vs[val_m.wrapping_sub(1) & 15] as u32,
                        );
                        let vv_m = vs[val_m & 15] as u32;
                        assert_eq!(
                            (u_m, vv_m, val_m),
                            (u_ref, vv_ref, val_ref),
                            "bitmask n={n} r={r:#x} c={c:#x}"
                        );
                        #[cfg(target_endian = "little")]
                        {
                            let u0 =
                                std::simd::simd_swizzle!(vr, v0, [7, 8, 9, 10, 11, 12, 13, 14]);
                            let u1 =
                                std::simd::simd_swizzle!(v0, v1, [7, 8, 9, 10, 11, 12, 13, 14]);
                            let rng0 = u0 - v0;
                            let rng1 = u1 - v1;
                            let sum = (i8x16::from_ne_bytes(ge0.to_simd().to_ne_bytes())
                                + i8x16::from_ne_bytes(ge1.to_simd().to_ne_bytes()))
                            .reduce_sum();
                            let val_t = (((sum as u8) >> 1) & 15) as usize;
                            let idx = u8x32::splat(sum as u8)
                                + u8x32::from_array([
                                    32, 33, 32, 33, 32, 33, 32, 33, 32, 33, 32, 33, 32, 33, 32, 33,
                                    32, 33, 32, 33, 32, 33, 32, 33, 32, 33, 32, 33, 32, 33, 32, 33,
                                ]);
                            let v_cat = std::simd::simd_swizzle!(
                                v0.to_ne_bytes(),
                                v1.to_ne_bytes(),
                                [
                                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                                    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
                                ]
                            );
                            let r_cat = std::simd::simd_swizzle!(
                                rng0.to_ne_bytes(),
                                rng1.to_ne_bytes(),
                                [
                                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                                    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
                                ]
                            );
                            let vv_t = u16x16::from_ne_bytes(v_cat.swizzle_dyn(idx))[0] as u32;
                            let new_rng_t = u16x16::from_ne_bytes(r_cat.swizzle_dyn(idx))[0] as u32;
                            assert_eq!(
                                (u_ref.wrapping_sub(vv_ref), vv_t, val_t),
                                (new_rng_t, vv_ref, val_ref),
                                "tbl16 n={n} r={r:#x} c={c:#x}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[cfg(feature = "native_av1_simd")]
    #[test]
    fn sqdmulh_threshold_formula_exact() {
        use std::simd::prelude::*;
        // Exhaustively verify the sqdmulh threshold identity over all 512
        // values of `f >> 6` (`0 <= f <= 32767`) and all 128 values of
        // `r >> 8` (`0x8000 <= r <= 0xffff`).
        let mp = u16x8::from_array([60, 56, 52, 48, 44, 40, 36, 32]);
        for r_hi in 128u32..=255 {
            let r = (r_hi << 8) | 0xff;
            let r_lo = i16x8::splat(((r & 0xff00) ^ 0x8000) as i16);
            for f_base in (0u16..512).step_by(8) {
                let mut f_arr = [0u16; 8];
                for j in 0..8 {
                    f_arr[j] = ((f_base + j as u16) << 6) | 63;
                }
                let f = u16x8::from_array(f_arr);
                let f_m = f & u16x8::splat(0xffc0);
                let hi = ((f_m.cast::<i16>().cast::<i32>() * r_lo.cast::<i32>())
                    >> i32x8::splat(15))
                .simd_min(i32x8::splat(i16::MAX as i32))
                .cast::<i16>()
                .cast::<u16>();
                let got = hi + (f_m + mp);
                for j in 0..8 {
                    let expected = (((r >> 8) * ((f_arr[j] as u32) >> 6)) >> 1) + mp[j] as u32;
                    assert_eq!(got[j] as u32, expected, "r={r:#x} f={:#x}", f_arr[j]);
                }
            }
        }
    }

    #[test]
    fn inverted_simd_cdfs_match_spec() {
        let mut seed = 31337u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed >> 8
        };
        fn mk_arr<const N: usize>(rnd: &mut dyn FnMut() -> u32, init_cnt: u16) -> [u16; N] {
            let mut v = [0u16; N];
            for p in &mut v[..N - 2] {
                *p = (1 + rnd() % 32767) as u16;
            }
            v[..N - 2].sort();
            v[N - 2] = 32768;
            v[N - 1] = init_cnt;
            v
        }
        macro_rules! check_cdf8 {
            ($a:expr, $b:expr, $upd:expr, $syms:literal, $n:literal, $rnd:expr) => {{
                let init_cnt = ($rnd() % 33) as u16;
                let mut sb = mk_arr::<$n>($rnd, init_cnt);
                let mut sa = CdfRow8::<$syms>::from_spec(sb);
                for iter in 0..48 {
                    let ga = $a.symbol_cdf(&mut sa);
                    let gb = $b.symbol(&mut sb, $upd);
                    assert_eq!(ga, gb, "CdfRow8<{}> upd={} iter={iter}", $syms, $upd);
                    assert_eq!(sa, sb, "CdfRow8<{}> cdf upd={} iter={iter}", $syms, $upd);
                    assert_eq!($a.symbol_max_bits(), $b.max_bits);
                }
            }};
        }
        macro_rules! check_cdf16 {
            ($a:expr, $b:expr, $upd:expr, $syms:literal, $n:literal, $rnd:expr) => {{
                let init_cnt = ($rnd() % 33) as u16;
                let mut sb = mk_arr::<$n>($rnd, init_cnt);
                let mut sa = CdfRow16::<$syms>::from_spec(sb);
                for iter in 0..48 {
                    let ga = $a.symbol_cdf(&mut sa);
                    let gb = $b.symbol(&mut sb, $upd);
                    assert_eq!(ga, gb, "CdfRow16<{}> upd={} iter={iter}", $syms, $upd);
                    assert_eq!(sa, sb, "CdfRow16<{}> cdf upd={} iter={iter}", $syms, $upd);
                    assert_eq!($a.symbol_max_bits(), $b.max_bits);
                }
            }};
        }
        for len in [0usize, 1, 5, 8, 17, 64, 512] {
            let data: Vec<u8> = (0..len).map(|_| rnd() as u8).collect();
            for upd in [false, true] {
                let mut a = SymbolDecoder::new(&data, !upd);
                let mut b = SpecDecoder::new(&data);
                check_cdf8!(a, b, upd, 6, 7, &mut rnd);
                check_cdf8!(a, b, upd, 7, 8, &mut rnd);
                check_cdf8!(a, b, upd, 8, 9, &mut rnd);
                {
                    let init_cnt = (rnd() % 33) as u16;
                    let mut sb = mk_arr::<10>(&mut rnd, init_cnt);
                    let mut sa = CdfRow9::from_spec(sb);
                    for iter in 0..48 {
                        let ga = a.symbol_cdf(&mut sa);
                        let gb = b.symbol(&mut sb, upd);
                        assert_eq!(ga, gb, "CdfRow9 upd={upd} iter={iter}");
                        assert_eq!(sa, sb, "CdfRow9 cdf upd={upd} iter={iter}");
                        assert_eq!(a.symbol_max_bits(), b.max_bits);
                    }
                }
                check_cdf16!(a, b, upd, 9, 10, &mut rnd);
                check_cdf16!(a, b, upd, 10, 11, &mut rnd);
                check_cdf16!(a, b, upd, 11, 12, &mut rnd);
                check_cdf16!(a, b, upd, 12, 13, &mut rnd);
                check_cdf16!(a, b, upd, 13, 14, &mut rnd);
                check_cdf16!(a, b, upd, 14, 15, &mut rnd);
                check_cdf16!(a, b, upd, 15, 16, &mut rnd);
                check_cdf16!(a, b, upd, 16, 17, &mut rnd);
            }
        }
    }
}
