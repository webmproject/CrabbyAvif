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
        c -= 8;
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
    let nb = ((48 - cnt) >> 3) as usize;
    if let Some(bytes) = buf.get(pos..).and_then(|s| s.first_chunk::<8>()) {
        let w = !u64::from_be_bytes(*bytes);
        dif |= w >> ((16 + cnt) as u32 & 63);
    } else {
        dif = refill_slow(buf, pos, dif, 40 - cnt);
    }
    let new_cnt = (cnt + ((nb as i32) << 3)) as u32;
    (dif, (nb as u64) | ((new_cnt as u64) << 32))
}

#[inline(always)]
fn refill_buf(buf: &[u8], cur_pos: &mut usize, dif: &mut u64, cnt: &mut i32) {
    let c = *cnt;
    debug_assert!((-16..=40).contains(&c));
    let nb = ((48 - c) >> 3) as usize;
    if let Some(bytes) = buf.get(*cur_pos..).and_then(|s| s.first_chunk::<8>()) {
        let w = !u64::from_be_bytes(*bytes);
        *dif |= w >> ((16 + c) as u32 & 63);
    } else {
        *dif = refill_slow(buf, *cur_pos, *dif, 40 - c);
    }
    *cur_pos = (*cur_pos).wrapping_add(nb);
    *cnt = c + ((nb as i32) << 3);
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
    *cnt -= sh as i32;
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
            cnt -= 1;
            break;
        }
        let s1 = ((rng < 0xff84) as u32) + ((rng < 0x8004) as u32);
        dif = dif.wrapping_sub(vw) << s1;
        rng = rng.wrapping_sub(v) << s1;
        cnt -= s1 as i32;
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

#[cold]
#[inline(never)]
fn golomb_coef_slow<const DUMMY: usize>(
    buf: &[u8],
    pos: usize,
    mut dif: u64,
    mut rng: u32,
    mut cnt: i32,
) -> (u64, u64) {
    let mut cur_pos = pos;
    // Ensure `cnt >= 16` once upfront so up to 8 prefix bits + 7 suffix bits
    // (`<= 15` equiprobable bits, max `dif` shift `<= 16`) need no per-bit
    // refill checks, with EOF tail-called so the common path stays frameless.
    if cnt < 16 {
        cold_refill();
        if let Some(bytes) = buf.get(pos..).and_then(|s| s.first_chunk::<8>()) {
            let nb = ((48 - cnt) >> 3) as usize;
            let w = !u64::from_be_bytes(*bytes);
            dif |= w >> ((16 + cnt) as u32 & 63);
            cur_pos = pos.wrapping_add(nb);
            cnt += (nb as i32) << 3;
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
            cnt -= 1;
            let mut x = 1u32;
            let mut rem = length - 1;
            while rem != 0 {
                x = (x << 1) | equi_nobuf(&mut dif, &mut rng, &mut cnt) as u32;
                rem -= 1;
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
        cnt -= s1 as i32;
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
            cnt -= 1;
            break;
        }
        let s1 = ((rng < 0xff84) as u32) + ((rng < 0x8004) as u32);
        dif = dif.wrapping_sub(vw) << s1;
        rng = rng.wrapping_sub(v) << s1;
        cnt -= s1 as i32;
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
    // Ensure `cnt >= 18` once upfront so 1 sign bit + up to 8 prefix bits +
    // up to 7 suffix bits (`<= 16` equiprobable bits, max `dif` shift `<= 17`)
    // need no per-bit refill checks, with EOF tail-called so the common path
    // compiles as a zero-frame leaf function.
    if cnt < 18 {
        cold_refill();
        if let Some(bytes) = buf.get(pos..).and_then(|s| s.first_chunk::<8>()) {
            let nb = ((48 - cnt) >> 3) as usize;
            let w = !u64::from_be_bytes(*bytes);
            dif |= w >> ((16 + cnt) as u32 & 63);
            cur_pos = pos.wrapping_add(nb);
            cnt += (nb as i32) << 3;
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
            cnt -= 1;
            let mut x = 1u32;
            let mut rem = length - 1;
            while rem != 0 {
                x = (x << 1) | equi_nobuf(&mut dif, &mut rng, &mut cnt) as u32;
                rem -= 1;
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
        cnt -= s1 as i32;
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
const MR5_SHIFT_SRC: u64 = 0xffe0_ffe0_ffe0_ffe0;
const TOP3_64: u64 = 0x0000_8000_8000_8000;

use crate::decoder::av1::tables::cdf::CdfRow5;

/// Register copy of a 4-symbol CDF in `CdfRow5` layout (`f`: 3 inverted
/// cumulative probabilities in bits 0..48 with bits 48..63 zero; `cb`:
/// biased counter `80 + min(count, 32)` so `cb >> 4` is `rate` in `5..=7`).
#[derive(Clone, Copy)]
pub struct Cdf5 {
    f: u64,
    cb: u64,
}

impl Cdf5 {
    #[inline(always)]
    pub fn from_row(c: &CdfRow5) -> Self {
        Self { f: c.f, cb: c.cb }
    }

    #[inline(always)]
    pub fn store_row(self, c: &mut CdfRow5) {
        *c = CdfRow5 {
            f: self.f,
            cb: self.cb,
        };
    }

    #[inline(always)]
    fn load_spec(c: &[u16; 5]) -> Self {
        let x = (c[0] as u64) | ((c[1] as u64) << 16) | ((c[2] as u64) << 32);
        let f = TOP3_64.wrapping_sub(x);
        let cb = 80 + (c[4].min(32) as u64);
        Self { f, cb }
    }

    #[inline(always)]
    fn store_spec_f(self, c: &mut [u16; 5]) {
        let x = TOP3_64.wrapping_sub(self.f);
        c[0] = x as u16;
        c[1] = (x >> 16) as u16;
        c[2] = (x >> 32) as u16;
        c[3] = 32768;
    }
}

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

/// Body of `symbol_simd8` (`8` lanes of `u16x8`).
macro_rules! simd_symbol_body {
    ($self:ident, $cdf:ident, $l:literal, $v:ident, $b:ident) => {{
        use std::simd::prelude::*;
        let n = N - 1;
        let k = (n & !1).min($l);
        let c0 = $self.dif;
        let c = (c0 >> (WIN_SIZE - 16)) as u16;
        let r = $self.rng;
        let cnt0 = $self.cnt;
        let (dif0, cnt) = if $self.raw.buf.len() >= 8 && $self.raw.pos <= $self.raw.buf.len() - 8 {
            let bytes = $self.raw.buf[$self.raw.pos..].first_chunk::<8>().unwrap();
            let nb = ((48 - cnt0) >> 3) as usize;
            let w = !u64::from_be_bytes(*bytes);
            $self.raw.pos = $self.raw.pos.wrapping_add(nb);
            (
                c0 | ((w >> 1) >> ((15 + cnt0) as u32 & 63)),
                cnt0 + ((nb as i32) << 3),
            )
        } else if cnt0 >= 16 {
            cold_refill();
            (c0, cnt0)
        } else {
            cold_refill();
            let nb = ((48 - cnt0) >> 3) as usize;
            let mut d = c0;
            let mut pos = $self.raw.pos;
            let mut shift = 40 - cnt0;
            loop {
                let b = if pos < $self.raw.buf.len() { $self.raw.buf[pos] ^ 0xff } else { 0xff };
                d |= (b as u64) << shift;
                pos += 1;
                shift -= 8;
                if shift < 0 {
                    break;
                }
            }
            $self.raw.pos = $self.raw.pos.wrapping_add(nb);
            (d, cnt0 + ((nb as i32) << 3))
        };
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
        // x86: dav1d's pmulhuw form + movemask / tzcnt index into a stack copy
        // of [rng, v..] (measured faster there than horizontal reductions).
        #[cfg(target_arch = "x86_64")]
        let (lt, u, vv, val) = {
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
            (!ge, u, vv, val)
        };
        // aarch64: `lt` (`v > c`) drives `u`, `vv`, `val`, and adaptation;
        // `lt.select(v, splat(r16))` folds the `val == 0 -> r` fallback into
        // the NEON mask before `uminv` so no scalar `min(r)` follows `fmov`.
        #[cfg(not(target_arch = "x86_64"))]
        let (lt, u, vv, val) = {
            let r7 = $v::splat(((r >> 8) & 0x7f) as u16);
            let vr = $v::splat(r as u16);
            let b = (f & $v::splat(0xffc0)) + min_prob;
            let v = b + (((f >> 6) * r7) >> 1);
            let lt = v.simd_gt($v::splat(c));
            let lm = lt.to_simd().cast::<u16>();
            let u = lt.select(v, vr).reduce_min() as u32;
            let vv = (v & !lm).reduce_max() as u32;
            let val = lt.to_simd().cast::<i16>().reduce_sum().wrapping_neg() as usize;
            (lt, u, vv, val)
        };
        let new_dif = dif0.wrapping_sub((vv as u64) << (WIN_SIZE - 16));
        let new_rng = u.wrapping_sub(vv);
        let d = (new_rng.leading_zeros() ^ 16) as i32;
        debug_assert!(cnt >= d);
        $self.dif = new_dif << ((d as u32) & 63);
        $self.rng = new_rng << ((d as u32) & 31);
        $self.cnt = cnt.wrapping_sub(d);
        if $self.allow_update {
            let lm = lt.to_simd().cast::<u16>();
            if N == $l {
                let count_v = std::simd::simd_swizzle!(x, [7, 7, 7, 7, 7, 7, 7, 7]);
                let rate = $v::splat(5) + ((count_v >> 4) & $v::splat(3));
                let delta = lt.select(x, f) >> rate;
                let y = (x - lm) + (delta ^ lm);
                let inc = y
                    .simd_lt($v::from_array([0, 0, 0, 0, 0, 0, 0, 32]))
                    .to_simd()
                    .cast::<u16>();
                let y = y - inc;
                *$cdf.first_chunk_mut::<$l>().unwrap() = *y.as_array();
            } else {
                let count = $cdf[n];
                let rate = $v::splat(5 + ((count >> 4) & 3));
                let delta = lt.select(x, f) >> rate;
                let y = (x - lm) + (delta ^ lm);
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
        #[cfg(target_arch = "x86_64")]
        let (lt, u, vv, val) = {
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
            (!ge, u, vv, val)
        };
        #[cfg(not(target_arch = "x86_64"))]
        let (lt, u, vv, val) = {
            let r7 = $v::splat(((r >> 8) & 0x7f) as u16);
            let vr = $v::splat(r as u16);
            let b = (f & $v::splat(0xffc0)) + min_prob;
            let v = b + (((f >> 6) * r7) >> 1);
            let lt = v.simd_gt($v::splat(c));
            let lm = lt.to_simd().cast::<u16>();
            let u = lt.select(v, vr).reduce_min() as u32;
            let vv = (v & !lm).reduce_max() as u32;
            let val = lt.to_simd().cast::<i16>().reduce_sum().wrapping_neg() as usize;
            (lt, u, vv, val)
        };
        let new_dif = dif0.wrapping_sub((vv as u64) << (WIN_SIZE - 16));
        let new_rng = u.wrapping_sub(vv);
        let d = (new_rng.leading_zeros() ^ 16) as i32;
        $self.dif = new_dif << ((d as u32) & 63);
        $self.rng = new_rng << ((d as u32) & 31);
        $self.cnt = cnt.wrapping_sub(d);
        if $upd {
            let lm = lt.to_simd().cast::<u16>();
            if N == $l {
                let count_v = std::simd::simd_swizzle!(x, [7, 7, 7, 7, 7, 7, 7, 7]);
                let rate = $v::splat(5) + ((count_v >> 4) & $v::splat(3));
                let delta = lt.select(x, f) >> rate;
                let y = (x - lm) + (delta ^ lm);
                let inc = y
                    .simd_lt($v::from_array([0, 0, 0, 0, 0, 0, 0, 32]))
                    .to_simd()
                    .cast::<u16>();
                let y = y - inc;
                *$cdf.first_chunk_mut::<$l>().unwrap() = *y.as_array();
            } else {
                let rate = $v::splat(5 + ((count >> 4) & 3));
                let delta = lt.select(x, f) >> rate;
                let y = (x - lm) + (delta ^ lm);
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
        let nb = ((48 - cnt) >> 3) as usize;
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
            self.dif |= w >> ((16 + cnt) as u32 & 63);
        } else {
            self.dif = refill_slow(self.raw.buf, self.raw.pos, self.dif, 40 - cnt);
        }
        self.raw.pos = self.raw.pos.wrapping_add(nb);
        self.cnt = cnt + ((nb as i32) << 3);
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
        let nb = ((48 - cnt) >> 3) as usize;
        if let Some(bytes) = self
            .raw
            .buf
            .get(self.raw.pos..)
            .and_then(|s| s.first_chunk::<8>())
        {
            let w = !u64::from_be_bytes(*bytes);
            self.dif |= w >> ((16 + cnt) as u32 & 63);
        } else {
            cold_refill();
            let mut pos = self.raw.pos;
            let mut shift = 40 - cnt;
            loop {
                let b = if pos < self.raw.buf.len() { self.raw.buf[pos] ^ 0xff } else { 0xff };
                self.dif |= (b as u64) << shift;
                pos += 1;
                shift -= 8;
                if shift < 0 {
                    break;
                }
            }
        }
        self.raw.pos = self.raw.pos.wrapping_add(nb);
        self.cnt = cnt + ((nb as i32) << 3);
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
        if N <= 10 {
            return self.symbol_simd8::<N>(cdf);
        }
        self.symbol_simd16::<N>(cdf)
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
            let r_idx = ((count >> 4) & 3) as u32;
            let rate = 4 + r_idx;
            let mr = (MR4_BASE >> r_idx) & MR4_BASE;
            let y_inc = x.wrapping_add((f >> rate) & mr);
            let y_dec = x.wrapping_sub((x >> rate) & mr);
            let y_mid = (y_inc & 0xffff_0000) | (y_dec & 0x0000_ffff);
            let y = sel(c1, y_dec, sel(c0, y_mid, y_inc));
            cdf[0] = y as u16;
            cdf[1] = (y >> 16) as u16;
            cdf[3] = inc_count(count);
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
        let count = if UPD { cdf[3] } else { 0 };
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
        if UPD {
            let r_idx = ((count >> 4) & 3) as u32;
            let rate = 4 + r_idx;
            let mr = (MR4_BASE >> r_idx) & MR4_BASE;
            let y_inc = x.wrapping_add((f >> rate) & mr);
            let y_dec = x.wrapping_sub((x >> rate) & mr);
            let y_mid = (y_inc & 0xffff_0000) | (y_dec & 0x0000_ffff);
            let y = sel(c1, y_dec, sel(c0, y_mid, y_inc));
            cdf[0] = y as u16;
            cdf[1] = (y >> 16) as u16;
            cdf[3] = inc_count(count);
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
            let r_idx = ((count >> 4) & 3) as u32;
            let rate = 5 + r_idx;
            let mr = (MR5_BASE >> r_idx) & MR5_BASE;
            let y_inc = x.wrapping_add((f >> rate) & mr);
            let y_dec = x.wrapping_sub((x >> rate) & mr);
            let y_1 = (y_inc & !0xffffu64) | (y_dec & 0xffffu64);
            let y_2 = (y_inc & !0xffff_ffffu64) | (y_dec & 0xffff_ffffu64);
            let y_3 = (y_inc & 0xffff_0000_0000_0000u64) | (y_dec & 0x0000_ffff_ffff_ffffu64);
            let y_lo = sel(c0, y_1, y_inc);
            let y_hi = sel(c3, y_dec, sel(c2, y_3, y_2));
            let y = sel(c1, y_hi, y_lo);
            cdf[0] = y as u16;
            cdf[1] = (y >> 16) as u16;
            cdf[2] = (y >> 32) as u16;
            cdf[3] = (y >> 48) as u16;
            cdf[5] = inc_count(count);
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
        let f = cdf.f;
        let v0 = ((r.wrapping_mul(((f as u32) >> 6) & 0x3ff)) >> 1).wrapping_add(12);
        let v1 = ((r.wrapping_mul((f as u32) >> 22)) >> 1).wrapping_add(8);
        let v2 = ((r.wrapping_mul((f >> 38) as u32)) >> 1).wrapping_add(4);
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
            let cb = cdf.cb;
            let rate = (cb >> 4) as u32;
            let mr = (MR5_SHIFT_SRC >> rate) & MR5_BASE;
            let f_dec = f.wrapping_sub((f >> rate) & mr);
            let f_inc = f.wrapping_add((TOP3_64.wrapping_sub(f) >> rate) & mr);
            let f_1 = (f_dec & !0xffffu64) | (f_inc & 0xffffu64);
            let f_2 = (f_dec & !0xffff_ffffu64) | (f_inc & 0xffff_ffffu64);
            let y_lo = sel(c0, f_1, f_dec);
            let y_hi = sel(c2, f_inc, f_2);
            let f_new = sel(c1, y_hi, y_lo);
            Cdf5 {
                f: f_new,
                cb: cb.wrapping_add((cb < 112) as u64),
            }
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

    /// `symbol_n::<5>` on a pre-inverted 16-byte `CdfRow5` (`coeff_base`),
    /// specialized on the adaptation flag.
    #[inline(always)]
    pub fn symbol5_c<const UPD: bool>(&mut self, cdf: &mut CdfRow5) -> u32 {
        use std::hint::select_unpredictable as sel;
        let c = (self.dif >> (WIN_SIZE - 16)) as u32;
        let r = self.rng >> 8;
        let loaded = Cdf5::from_row(cdf);
        let f = loaded.f;
        let v0 = ((r.wrapping_mul(((f as u32) >> 6) & 0x3ff)) >> 1).wrapping_add(12);
        let v1 = ((r.wrapping_mul((f as u32) >> 22)) >> 1).wrapping_add(8);
        let v2 = ((r.wrapping_mul((f >> 38) as u32)) >> 1).wrapping_add(4);
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
        if UPD {
            let cb = loaded.cb;
            let rate = (cb >> 4) as u32;
            let mr = (MR5_SHIFT_SRC >> rate) & MR5_BASE;
            let f_dec = f.wrapping_sub((f >> rate) & mr);
            let f_inc = f.wrapping_add((TOP3_64.wrapping_sub(f) >> rate) & mr);
            let f_1 = (f_dec & !0xffffu64) | (f_inc & 0xffffu64);
            let f_2 = (f_dec & !0xffff_ffffu64) | (f_inc & 0xffff_ffffu64);
            let y_lo = sel(c0, f_1, f_dec);
            let y_hi = sel(c2, f_inc, f_2);
            let f_new = sel(c1, y_hi, y_lo);
            (Cdf5 {
                f: f_new,
                cb: cb.wrapping_add((cb < 112) as u64),
            })
            .store_row(cdf);
        }
        self.norm_hot(new_dif, new_rng);
        val
    }

    /// `hi_tok` for loops specialized on the adaptation flag.
    #[inline(always)]
    pub fn hi_tok_c<const UPD: bool>(&mut self, cdf: &mut CdfRow5) -> u32 {
        let r = Cdf5::from_row(cdf);
        let (br0, r0) = self.symbol5_r::<true>(r, UPD);
        'done: {
            if br0 < 3 {
                if UPD {
                    r0.store_row(cdf);
                }
                break 'done 3u32.wrapping_add(br0);
            }
            let (br1, r1) = self.symbol5_r::<true>(r0, UPD);
            if br1 < 3 {
                if UPD {
                    r1.store_row(cdf);
                }
                break 'done 6u32.wrapping_add(br1);
            }
            let (br2, r2) = self.symbol5_r::<true>(r1, UPD);
            if br2 < 3 {
                if UPD {
                    r2.store_row(cdf);
                }
                break 'done 9u32.wrapping_add(br2);
            }
            let (br3, r3) = self.symbol5_r::<true>(r2, UPD);
            if UPD {
                r3.store_row(cdf);
            }
            if br3 < 3 {
                break 'done 12u32.wrapping_add(br3);
            }
            15
        }
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
                let rate = 4 + (count >> 4) as u32;
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
            self.simd_symbol_hot::<N, UPD>(cdf)
        }
    }

    #[inline(always)]
    fn simd_symbol_hot<const N: usize, const UPD: bool>(&mut self, cdf: &mut [u16; N]) -> usize {
        simd_symbol_body_hot!(self, cdf, N, 8, u16x8, i8x16, UPD)
    }

    /// `symbol_n` for `N <= 10` with 8 lanes. On x86_64, `u` and `v` of the
    /// decoded symbol are looked up in a stack copy of `[rng, v..]` at the
    /// index given by the compare movemask (measured faster there); on
    /// other targets they come from min / max reductions of the compare
    /// masked `v` lanes.
    #[inline(never)]
    fn symbol_simd8<const N: usize>(&mut self, cdf: &mut [u16; N]) -> usize {
        simd_symbol_body!(self, cdf, 8, u16x8, i8x16)
    }

    /// `symbol_n` for `11 <= N <= 17` with two 8-lane vectors.
    #[inline(never)]
    fn symbol_simd16<const N: usize>(&mut self, cdf: &mut [u16; N]) -> usize {
        use std::simd::prelude::*;
        let n = N - 1;
        let k2 = (n & !1) - 8;
        let c0 = self.dif;
        let c = (c0 >> (WIN_SIZE - 16)) as u16;
        let r = self.rng;
        let cnt0 = self.cnt;
        let (dif0, cnt) = if self.raw.buf.len() >= 8 && self.raw.pos <= self.raw.buf.len() - 8 {
            let bytes = self.raw.buf[self.raw.pos..].first_chunk::<8>().unwrap();
            let nb = ((48 - cnt0) >> 3) as usize;
            let w = !u64::from_be_bytes(*bytes);
            self.raw.pos = self.raw.pos.wrapping_add(nb);
            (
                c0 | ((w >> 1) >> ((15 + cnt0) as u32 & 63)),
                cnt0 + ((nb as i32) << 3),
            )
        } else if cnt0 >= 16 {
            cold_refill();
            (c0, cnt0)
        } else {
            cold_refill();
            let nb = ((48 - cnt0) >> 3) as usize;
            let mut d = c0;
            let mut pos = self.raw.pos;
            let mut shift = 40 - cnt0;
            loop {
                let b = if pos < self.raw.buf.len() { self.raw.buf[pos] ^ 0xff } else { 0xff };
                d |= (b as u64) << shift;
                pos += 1;
                shift -= 8;
                if shift < 0 {
                    break;
                }
            }
            self.raw.pos = self.raw.pos.wrapping_add(nb);
            (d, cnt0 + ((nb as i32) << 3))
        };
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
        #[cfg(target_arch = "x86_64")]
        let (ge0, ge1, u, vv, val) = {
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
            (ge0, ge1, u, vv, val)
        };
        #[cfg(not(target_arch = "x86_64"))]
        let (ge0, ge1, u, vv, val) = {
            let r7 = u16x8::splat(((r >> 8) & 0x7f) as u16);
            let vr = u16x8::splat(r as u16);
            let mask_ffc0 = u16x8::splat(0xffc0);
            let b0 = (f0 & mask_ffc0) + mp0;
            let b1 = (f1 & mask_ffc0) + mp1;
            let v0 = b0 + (((f0 >> 6) * r7) >> 1);
            let v1 = b1 + (((f1 >> 6) * r7) >> 1);
            let vc = u16x8::splat(c);
            let ge0 = vc.simd_ge(v0);
            let ge1 = vc.simd_ge(v1);
            let gm0 = ge0.to_simd().cast::<u16>();
            let gm1 = ge1.to_simd().cast::<u16>();
            let u = ge0.select(vr, v0).simd_min(ge1.select(vr, v1)).reduce_min() as u32;
            let vv = (v0 & gm0).simd_max(v1 & gm1).reduce_max() as u32;
            let val = (16i32
                + (ge0.to_simd().cast::<i16>() + ge1.to_simd().cast::<i16>()).reduce_sum() as i32)
                as u32 as usize;
            (ge0, ge1, u, vv, val)
        };
        let new_dif = dif0.wrapping_sub((vv as u64) << (WIN_SIZE - 16));
        let new_rng = u.wrapping_sub(vv);
        let d = (new_rng.leading_zeros() ^ 16) as i32;
        debug_assert!(cnt >= d);
        self.dif = new_dif << ((d as u32) & 63);
        self.rng = new_rng << ((d as u32) & 31);
        self.cnt = cnt.wrapping_sub(d);
        if self.allow_update {
            let gm0 = ge0.to_simd().cast::<u16>();
            let gm1 = ge1.to_simd().cast::<u16>();
            if N == 16 {
                let count_v = std::simd::simd_swizzle!(x1, [7, 7, 7, 7, 7, 7, 7, 7]);
                let rate = u16x8::splat(5) + ((count_v >> 4) & u16x8::splat(3));
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
                self.cnt -= 1;
                return Some(1);
            }
        }
        cold_refill();
        let (dif, out) =
            golomb_coef_slow::<0>(self.raw.buf, self.raw.pos, self.dif, self.rng, self.cnt);
        self.dif = dif;
        self.rng = (out & 0xFFFF) as u32;
        self.cnt = ((out >> 16) as i8) as i32;
        self.raw.pos += ((out >> 24) & 0xFF) as usize;
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
        self.raw.pos += ((out >> 24) & 0xFF) as usize;
        let hi = (out >> 32) as u32;
        if hi == 0 {
            return (0, 0);
        }
        let sign = (hi >> 30) & 1;
        let q = hi & 0xFFFFF;
        let dq = (((q as u64 * q2 as u64) & 0xFFFFFF) >> dq_shift).min(cf_max + sign as u64) as i32;
        let dq_signed = std::hint::select_unpredictable(sign != 0, dq.wrapping_neg(), dq);
        (q | (1 << 31), dq_signed)
    }

    fn symbol_generic(&mut self, cdf: &mut [u16]) -> usize {
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
        if self.allow_update {
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
            let rate = 4 + (count >> 4) as u32;
            cdf[0] = sel(ret, p0.wrapping_add(f >> rate), p0.wrapping_sub(p0 >> rate)) as u16;
            cdf[2] = inc_count(count);
        }
        self.norm(new_dif, new_rng);
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
        let new_dif = sel(ret, dif.wrapping_sub(vw), dif) << (shift & 63);
        (ret, new_rng, new_dif, shift)
    }

    /// Equiprobable `read_literal(n)` for `1 <= n <= 8` (`eob` extra bits).
    /// Pre-refills once when `cnt < 16` so the bit loop needs no per-bit
    /// refill branch or duplicate cold refill blocks.
    #[inline(always)]
    pub fn literal_short(&mut self, n: usize) -> usize {
        debug_assert!((1..=8).contains(&n));
        if self.cnt < 16 {
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
                for i in 0..2000 {
                    match rnd() % 6 {
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
                        _ => {
                            let sa = if update {
                                a.symbol5_c::<true>(&mut c5a)
                            } else {
                                a.symbol5_c::<false>(&mut c5a)
                            };
                            let sb = b.symbol(&mut c5b, update);
                            assert_eq!(sa as usize, sb, "len {len} iter {i}");
                            assert_eq!(c5a, c5b);
                        }
                    }
                    assert_eq!(a.symbol_max_bits(), b.max_bits, "len {len} iter {i}");
                }
            }
        }
    }

    #[test]
    fn long_golomb_and_simd_reduction_match_spec() {
        use std::simd::prelude::*;
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
                let mut u_ref = r;
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
                        + (ge0.to_simd().cast::<i16>() + ge1.to_simd().cast::<i16>()).reduce_sum()
                            as i32) as u32 as usize;
                    assert_eq!(
                        (u, vv, val),
                        (u_ref, vv_ref, val_ref),
                        "n={n} r={r:#x} c={c:#x}"
                    );
                }
            }
        }
    }
}
