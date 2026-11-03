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

// The cold out-of-line helpers below (and `SymbolDecoder::refill_ret`)
// take an unused `const DUMMY: usize` generic, always instantiated as
// `::<0>`. It has no semantic effect: it only makes the function generic,
// so it is monomorphized in the codegen unit of its caller instead of once
// in this module's, which places the cold code next to its (hot) caller
// and measurably changed the generated code. Keep it unless benchmarks
// show otherwise.
#[cold]
#[inline(never)]
fn refill_cold<const DUMMY: usize>(buf: &[u8], pos: usize, mut dif: u64, cnt: i32) -> (u64, u64) {
    let c = WIN_SIZE - cnt - 24;
    debug_assert!((0..=56).contains(&c));
    let nb = ((c >> 3) + 1) as usize;
    if let Some(bytes) = buf.get(pos..pos + 8) {
        let w = !u64::from_be_bytes(bytes.try_into().unwrap());
        dif |= (w >> (56 - (c & !7))) << (c & 7);
    } else {
        dif = refill_slow(buf, pos, dif, c);
    }
    let new_cnt = (48 - (c & 7)) as u32;
    (dif, (nb as u64) | ((new_cnt as u64) << 32))
}

#[cold]
#[inline(never)]
fn golomb_coef_slow<const DUMMY: usize>(
    buf: &[u8],
    pos: usize,
    dif: u64,
    state: u64,
) -> (u64, u64) {
    let mut s = SymbolDecoder {
        raw: ByteReader { buf, pos },
        dif,
        rng: (state & 0xFFFF) as u32,
        cnt: (state >> 16) as i8 as i32,
        allow_update: false,
    };
    let mut length = 0u32;
    loop {
        length += 1;
        if s.bool_cold() {
            break;
        }
        if length > 32 {
            let out = (s.rng as u64)
                | (((s.cnt as i8 as u8) as u64) << 16)
                | (((s.raw.pos - pos) as u64) << 24);
            return (s.dif, out);
        }
    }
    let mut x = 1u32;
    for _ in 0..(length - 1) {
        x = (x << 1) | s.bool_cold() as u32;
    }
    let out = (s.rng as u64)
        | (((s.cnt as i8 as u8) as u64) << 16)
        | (((s.raw.pos - pos) as u64) << 24)
        | ((x as u64) << 32);
    (s.dif, out)
}

#[cold]
#[inline(never)]
fn ac_golomb_slow<const DUMMY: usize>(
    buf: &[u8],
    pos: usize,
    dif: u64,
    state: u64,
    q2: u32,
    dq_shift: u32,
    cf_max: u64,
) -> (u64, u64, i32) {
    let mut s = SymbolDecoder {
        raw: ByteReader { buf, pos },
        dif,
        rng: (state & 0xFFFF) as u32,
        cnt: (state >> 16) as i8 as i32,
        allow_update: false,
    };
    let sign = s.bool_cold();
    let mut length = 0u32;
    loop {
        length += 1;
        if s.bool_cold() {
            break;
        }
        if length > 32 {
            let out = (s.rng as u64)
                | (((s.cnt as i8 as u8) as u64) << 16)
                | (((s.raw.pos - pos) as u64) << 24);
            return (s.dif, out, 0);
        }
    }
    let mut x = 1u32;
    for _ in 0..(length - 1) {
        x = (x << 1) | s.bool_cold() as u32;
    }
    let q = x.wrapping_add(14) & 0xFFFFF;
    let dq = (((q as u64 * q2 as u64) & 0xFFFFFF) >> dq_shift).min(cf_max + sign as u64) as i32;
    let dq_signed = if sign { -dq } else { dq };
    let out = (s.rng as u64)
        | (((s.cnt as i8 as u8) as u64) << 16)
        | (((s.raw.pos - pos) as u64) << 24)
        | (((q | (1 << 31)) as u64) << 32);
    (s.dif, out, dq_signed)
}

/// `c < v` for `c < 2^16`, `v < 2^17`, as the sign of `c - v`: the
/// selects on it become single uop `cmovs` instead of `cmova` (two uops,
/// two cycles of latency on Intel) in the symbol decoders' critical path.
#[inline(always)]
fn lt(c: u32, v: u32) -> bool {
    (c as i32).wrapping_sub(v as i32) < 0
}

/// CDF adaptation counter update (saturating at 32).
#[inline(always)]
fn inc_count(count: u16) -> u16 {
    count + (count < 32) as u16
}

const TOP_MR4: [u32; 8] = [
    0x8000_8000,
    0x8000_ffff,
    0xffff_ffff,
    0xffff_ffff,
    0x0fff_0fff,
    0x07ff_07ff,
    0x03ff_03ff,
    0x01ff_01ff,
];

const TOP_MR5: [u64; 12] = [
    0x8000_8000_8000_8000,
    0x8000_8000_8000_ffff,
    0x8000_8000_ffff_ffff,
    0x8000_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_ffff_ffff,
    0x07ff_07ff_07ff_07ff,
    0x03ff_03ff_03ff_03ff,
    0x01ff_01ff_01ff_01ff,
    0x00ff_00ff_00ff_00ff,
];

/// Register copy of a 4-symbol CDF (`[u16; 5]`: 3 values below 32768,
/// 32768 and the adaptation counter), in the layout `symbol5_r` works on.
#[derive(Clone, Copy)]
pub struct Cdf5 {
    x: u64,
    count: u16,
}

impl Cdf5 {
    #[inline(always)]
    pub fn load(c: &[u16; 5]) -> Self {
        let x =
            (c[0] as u64) | ((c[1] as u64) << 16) | ((c[2] as u64) << 32) | ((c[3] as u64) << 48);
        Self { x, count: c[4] }
    }

    #[inline(always)]
    pub fn store(self, c: &mut [u16; 5]) {
        c[0] = self.x as u16;
        c[1] = (self.x >> 16) as u16;
        c[2] = (self.x >> 32) as u16;
        c[3] = (self.x >> 48) as u16;
        c[4] = self.count;
    }
}

/// Body of `symbol_simd8` (`8` lanes of `u16x8`).
macro_rules! simd_symbol_body {
    ($self:ident, $cdf:ident, $l:literal, $v:ident, $b:ident) => {{
        use std::simd::prelude::*;
        let n = N - 1;
        let k = (n & !1).min($l);
        let c = ($self.dif >> (WIN_SIZE - 16)) as u16;
        let r = $self.rng;
        let x = if N >= $l {
            $v::from_array(*$cdf.first_chunk::<$l>().unwrap())
        } else {
            let mut arr = [0u16; $l];
            arr[..k].copy_from_slice(&$cdf[..k]);
            $v::from_array(arr)
        };
        let top = const {
            let mut t = [0u16; $l];
            let mut i = 0;
            while i < (N - 2) && i < $l {
                t[i] = 0x8000;
                i += 1;
            }
            $v::from_array(t)
        };
        let f = top.saturating_sub(x);
        let min_prob = const {
            let mut mp = [0u16; $l];
            let mut i = 0;
            while i < N - 2 && i < $l {
                mp[i] = (EC_MIN_PROB * (N - 2 - i) as u32) as u16;
                i += 1;
            }
            $v::from_array(mp)
        };
        // ((rng >> 8) * (f >> 6)) >> 1: dav1d's pmulhuw form on x86, the
        // split (f & 0xffc0) + ((f >> 6) * (r7 = (rng >> 8) & 0x7f) >> 1)
        // in 16 bits elsewhere.
        #[cfg(target_arch = "x86_64")]
        let prod = {
            let fm = (f & $v::splat(0xffc0)) << 1;
            let rr = $v::splat((r & 0xff00) as u16).cast::<u32>();
            ((fm.cast::<u32>() * rr) >> 16).cast::<u16>()
        };
        #[cfg(not(target_arch = "x86_64"))]
        let prod = {
            let r7 = $v::splat(((r >> 8) & 0x7f) as u16);
            (f & $v::splat(0xffc0)) + (((f >> 6) * r7) >> 1)
        };
        // x86: movemask / tzcnt index into a stack copy of [rng, v..]
        // (measured faster there than the reductions below).
        #[cfg(target_arch = "x86_64")]
        let (ge, u, vv, val) = {
            let v = prod + min_prob;
            let ge = $v::splat(c).simd_ge(v);
            let ge_b = $b::from_ne_bytes(ge.to_simd().cast::<u16>().to_ne_bytes());
            let mask = ge_b.simd_lt($b::splat(0)).to_bitmask() as u32 | (1u32 << (2 * (N - 2)));
            let val = (mask.trailing_zeros() >> 1) as usize;
            if N <= $l + 1 {
                let mut vs = [0u16; $l + 1];
                vs[0] = r as u16;
                *vs[1..1 + $l].first_chunk_mut::<$l>().unwrap() = *v.as_array();
                let idx = val.min($l - 1);
                (ge, vs[idx] as u32, vs[idx + 1] as u32, val)
            } else {
                let mut vs = [0u16; $l + 2];
                vs[0] = r as u16;
                *vs[1..1 + $l].first_chunk_mut::<$l>().unwrap() = *v.as_array();
                let idx = val.min($l);
                (ge, vs[idx] as u32, vs[idx + 1] as u32, val)
            }
        };
        // aarch64: u and v of the decoded symbol by uminv / umaxv of the
        // compare masked lanes. Lanes N - 2.. have v = 0 <= c, so lt is
        // false there without needing a special sentinel lane.
        #[cfg(not(target_arch = "x86_64"))]
        let (ge, u, vv, val) = {
            let v = prod + min_prob;
            let lt = v.simd_gt($v::splat(c));
            let lm = lt.to_simd().cast::<u16>();
            let u = ((v | !lm).reduce_min() as u32).min(r);
            let vv = (v & !lm).reduce_max() as u32;
            let val = lt.to_simd().cast::<i16>().reduce_sum().wrapping_neg() as usize;
            (!lt, u, vv, val)
        };
        let new_dif = $self.dif - ((vv as u64) << (WIN_SIZE - 16));
        let new_rng = u - vv;
        let d = (new_rng.leading_zeros() ^ 16) as i32;
        $self.dif = new_dif << d;
        $self.rng = new_rng << d;
        if $self.allow_update {
            let count = $cdf[n];
            let rate = $v::splat(5 + (count >> 4));
            let gm = ge.to_simd().cast::<u16>();
            let y = x + ((f >> rate) & gm) - ((x >> rate) & !gm);
            if N > $l {
                *$cdf.first_chunk_mut::<$l>().unwrap() = *y.as_array();
                $cdf[n] = inc_count(count);
            } else if N == $l {
                let mut ya = *y.as_array();
                ya[$l - 1] = inc_count(count);
                *$cdf.first_chunk_mut::<$l>().unwrap() = ya;
            } else {
                $cdf[..k].copy_from_slice(&y.as_array()[..k]);
                $cdf[n] = inc_count(count);
            }
        }
        let cnt = $self.cnt;
        $self.cnt = cnt - d;
        if (cnt as u32) < (d as u32) {
            return $self.refill_ret::<0>(val);
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
        let c = WIN_SIZE - self.cnt - 24;
        debug_assert!((0..=56).contains(&c));
        let nb = ((c >> 3) + 1) as usize;
        if let Some(bytes) = self.raw.buf.get(self.raw.pos..self.raw.pos + 8) {
            // Fast path: load the next 8 bytes at once and insert as many
            // whole bytes as fit, exactly like the byte loop in refill_slow.
            let w = !u64::from_be_bytes(bytes.try_into().unwrap());
            self.dif |= (w >> (56 - (c & !7))) << (c & 7);
        } else {
            self.dif = refill_slow(self.raw.buf, self.raw.pos, self.dif, c);
        }
        self.raw.pos += nb;
        self.cnt = 48 - (c & 7);
    }

    /// `refill` for the tail of the SIMD symbol decoders, passing their
    /// result through. `DUMMY`: see `refill_cold`. `black_box` keeps LLVM
    /// from deducing that the result is `val` (a `returned` argument),
    /// which would make callers keep `val` live across the call (in a
    /// callee-saved register, saved / restored in the hot path) instead of
    /// taking it from the return register. Codegen only: correctness does
    /// not depend on it.
    #[cold]
    #[inline(never)]
    fn refill_ret<const DUMMY: usize>(&mut self, val: usize) -> usize {
        self.refill();
        std::hint::black_box(val)
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
        let c = WIN_SIZE - self.cnt - 24;
        debug_assert!((0..=56).contains(&c));
        let nb = ((c >> 3) + 1) as usize;
        if let Some(bytes) = self.raw.buf.get(self.raw.pos..self.raw.pos + 8) {
            let w = !u64::from_be_bytes(bytes.try_into().unwrap());
            self.dif |= (w >> (56 - (c & !7))) << (c & 7);
        } else {
            cold_refill();
            let mut pos = self.raw.pos;
            let mut shift = c;
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
        self.raw.pos += nb;
        self.cnt = 48 - (c & 7);
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
        self.dif = dif << d;
        self.rng = rng << d;
        self.cnt = cnt - d;
        if (cnt as u32) < (d as u32) {
            cold_refill();
            let (dif, out) = refill_cold::<0>(self.raw.buf, self.raw.pos, self.dif, self.cnt);
            self.dif = dif;
            self.raw.pos += (out as u32) as usize;
            self.cnt = (out >> 32) as i32;
        }
    }

    /// `norm` with the refill inlined, for the coefficient loops.
    #[inline(always)]
    fn norm_hot(&mut self, dif: u64, rng: u32) {
        let d = (rng.leading_zeros() ^ 16) as i32;
        let cnt = self.cnt;
        self.dif = dif << d;
        self.rng = rng << d;
        self.cnt = cnt - d;
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
        let v = (((r >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB;
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
        let v0 = ((r * ((f >> 6) & 0x3ff)) >> 1) + 8;
        let v1 = ((r * (f >> 22)) >> 1) + 4;
        let c0 = lt(c, v0);
        let c1 = lt(c, v1);
        let val = (c0 as u32) + (c1 as u32);
        let u0 = sel(c0, v0, self.rng);
        let u = sel(c1, v1, u0);
        let vv0 = sel(c0, v1, v0);
        let v = sel(c1, 0, vv0);
        let new_dif = self.dif - ((v as u64) << (WIN_SIZE - 16));
        let new_rng = u - v;
        if self.allow_update {
            let top = TOP_MR4[val as usize & 3];
            let count = cdf[3];
            let r_idx = (count >> 4) as usize & 3;
            let rate = 4 + r_idx as u32;
            let mr = TOP_MR4[4 + r_idx];
            let y = x
                .wrapping_sub(top & mr)
                .wrapping_add((top.wrapping_sub(x) >> rate) & mr);
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
        let f = 0x8000_8000u32.wrapping_sub(x);
        let v0 = ((r * ((f >> 6) & 0x3ff)) >> 1) + 8;
        let v1 = ((r * (f >> 22)) >> 1) + 4;
        let c0 = lt(c, v0);
        let c1 = lt(c, v1);
        let val = (c0 as u32) + (c1 as u32);
        let u0 = sel(c0, v0, self.rng);
        let u = sel(c1, v1, u0);
        let vv0 = sel(c0, v1, v0);
        let v = sel(c1, 0, vv0);
        let new_dif = self.dif - ((v as u64) << (WIN_SIZE - 16));
        let new_rng = u - v;
        if UPD {
            let top = TOP_MR4[val as usize & 3];
            let count = cdf[3];
            let r_idx = (count >> 4) as usize & 3;
            let rate = 4 + r_idx as u32;
            let mr = TOP_MR4[4 + r_idx];
            let y = x
                .wrapping_sub(top & mr)
                .wrapping_add((top.wrapping_sub(x) >> rate) & mr);
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
        let (val, y) = self.symbol5_r::<false>(Cdf5::load(cdf), upd);
        if upd {
            y.store(cdf);
        }
        val as usize
    }

    /// `symbol_n::<6>` (5-symbol CDFs: 4 values below 32768, 32768, count).
    #[inline(always)]
    pub fn symbol6(&mut self, cdf: &mut [u16; 6]) -> usize {
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
        let v0 = ((r * ((f_lo >> 6) & 0x3ff)) >> 1) + 16;
        let v1 = ((r * (f_lo >> 22)) >> 1) + 12;
        let v2 = ((r * ((f_hi >> 6) & 0x3ff)) >> 1) + 8;
        let v3 = ((r * (f_hi >> 22)) >> 1) + 4;
        let c0 = lt(c, v0);
        let c1 = lt(c, v1);
        let c2 = lt(c, v2);
        let c3 = lt(c, v3);
        let val = (c0 as u32) + (c1 as u32) + (c2 as u32) + (c3 as u32);
        let u_lo = sel(c0, v0, self.rng);
        let u_hi = sel(c3, v3, sel(c2, v2, v1));
        let u = sel(c1, u_hi, u_lo);
        let v_lo = sel(c0, v1, v0);
        let v_hi = sel(c3, 0, sel(c2, v3, v2));
        let v = sel(c1, v_hi, v_lo);
        let new_dif = self.dif - ((v as u64) << (WIN_SIZE - 16));
        let new_rng = u - v;
        if self.allow_update {
            let top = TOP_MR5[val as usize & 7];
            let count = cdf[5];
            let r_idx = (count >> 4) as usize & 3;
            let rate = 5 + r_idx as u32;
            let mr = TOP_MR5[8 + r_idx];
            let y = x
                .wrapping_sub(top & mr)
                .wrapping_add((top.wrapping_sub(x) >> rate) & mr);
            cdf[0] = y as u16;
            cdf[1] = (y >> 16) as u16;
            cdf[2] = (y >> 32) as u16;
            cdf[3] = (y >> 48) as u16;
            cdf[5] = inc_count(count);
        }
        self.norm(new_dif, new_rng);
        val as usize
    }

    /// `symbol_n::<5>` on a CDF held in registers: returns the symbol and
    /// the adapted CDF (unchanged unless `upd`).
    #[inline(always)]
    pub fn symbol5_r<const HOT: bool>(&mut self, cdf: Cdf5, upd: bool) -> (u32, Cdf5) {
        use std::hint::select_unpredictable as sel;
        let c = (self.dif >> (WIN_SIZE - 16)) as u32;
        let r = self.rng >> 8;
        let x = cdf.x;
        let f = 0x8000_8000_8000_8000u64.wrapping_sub(x);
        let v0 = ((r * (((f as u32) >> 6) & 0x3ff)) >> 1) + 12;
        let v1 = ((r * ((f as u32) >> 22)) >> 1) + 8;
        let v2 = ((r * (((f >> 38) as u32) & 0x3ff)) >> 1) + 4;
        let c0 = lt(c, v0);
        let c1 = lt(c, v1);
        let c2 = lt(c, v2);
        let val = (c0 as u32) + (c1 as u32) + (c2 as u32);
        let u_lo = sel(c0, v0, self.rng);
        let u_hi = sel(c2, v2, v1);
        let u = sel(c1, u_hi, u_lo);
        let v_lo = sel(c0, v1, v0);
        let v_hi = sel(c2, 0, v2);
        let v = sel(c1, v_hi, v_lo);
        let new_dif = self.dif - ((v as u64) << (WIN_SIZE - 16));
        let new_rng = u - v;
        let y = if upd {
            let top = TOP_MR5[val as usize & 3];
            let count = cdf.count;
            let r_idx = (count >> 4) as usize & 3;
            let rate = 5 + r_idx as u32;
            let mr = TOP_MR5[8 + r_idx];
            let y = x
                .wrapping_sub(top & mr)
                .wrapping_add((top.wrapping_sub(x) >> rate) & mr);
            Cdf5 {
                x: y,
                count: inc_count(count),
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

    /// `symbol_n::<5>` for loops specialized on the adaptation flag.
    #[inline(always)]
    pub fn symbol5_c<const UPD: bool>(&mut self, cdf: &mut [u16; 5]) -> u32 {
        use std::hint::select_unpredictable as sel;
        let c = (self.dif >> (WIN_SIZE - 16)) as u32;
        let r = self.rng >> 8;
        let loaded = Cdf5::load(cdf);
        let x = loaded.x;
        let f = 0x8000_8000_8000_8000u64.wrapping_sub(x);
        let v0 = ((r * (((f as u32) >> 6) & 0x3ff)) >> 1) + 12;
        let v1 = ((r * ((f as u32) >> 22)) >> 1) + 8;
        let v2 = ((r * (((f >> 38) as u32) & 0x3ff)) >> 1) + 4;
        let c0 = lt(c, v0);
        let c1 = lt(c, v1);
        let c2 = lt(c, v2);
        let val = (c0 as u32) + (c1 as u32) + (c2 as u32);
        let u0 = sel(c0, v0, self.rng);
        let u1 = sel(c1, v1, u0);
        let u = sel(c2, v2, u1);
        let vv0 = sel(c0, v1, v0);
        let vv1 = sel(c1, v2, vv0);
        let v = sel(c2, 0, vv1);
        let new_dif = self.dif - ((v as u64) << (WIN_SIZE - 16));
        let new_rng = u - v;
        if UPD {
            let top = TOP_MR5[val as usize & 3];
            let count = loaded.count;
            let r_idx = (count >> 4) as usize & 3;
            let rate = 5 + r_idx as u32;
            let mr = TOP_MR5[8 + r_idx];
            let y = x
                .wrapping_sub(top & mr)
                .wrapping_add((top.wrapping_sub(x) >> rate) & mr);
            (Cdf5 {
                x: y,
                count: inc_count(count),
            })
            .store(cdf);
        }
        self.norm_hot(new_dif, new_rng);
        val
    }

    /// `hi_tok` for loops specialized on the adaptation flag.
    #[inline(always)]
    pub fn hi_tok_c<const UPD: bool>(&mut self, cdf: &mut [u16; 5]) -> u32 {
        let mut tok = 3u32;
        let mut r = Cdf5::load(cdf);
        for _ in 0..4 {
            let (br, y) = self.symbol5_r::<true>(r, UPD);
            r = y;
            tok += br;
            if br < 3 {
                break;
            }
        }
        if UPD {
            r.store(cdf);
        }
        tok
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
        let c = (self.dif >> (WIN_SIZE - 16)) as u16;
        let r = self.rng;
        let x0 = u16x8::from_array(*cdf.first_chunk::<8>().unwrap());
        let x1 = if N >= 16 {
            u16x8::from_array(*cdf[8..16].first_chunk::<8>().unwrap())
        } else {
            let mut arr1 = [0u16; 8];
            arr1[..k2].copy_from_slice(&cdf[8..8 + k2]);
            u16x8::from_array(arr1)
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
            let mut vs = [0u16; 17];
            vs[0] = r as u16;
            *vs[1..9].first_chunk_mut::<8>().unwrap() = *v0.as_array();
            *vs[9..17].first_chunk_mut::<8>().unwrap() = *v1.as_array();
            let idx = val.min(15);
            (ge0, ge1, vs[idx] as u32, vs[idx + 1] as u32, val)
        };
        #[cfg(not(target_arch = "x86_64"))]
        let (ge0, ge1, u, vv, val) = {
            let r7 = u16x8::splat(((r >> 8) & 0x7f) as u16);
            let mask_ffc0 = u16x8::splat(0xffc0);
            let v0 = (f0 & mask_ffc0) + (((f0 >> 6) * r7) >> 1) + mp0;
            let v1 = (f1 & mask_ffc0) + (((f1 >> 6) * r7) >> 1) + mp1;
            let vc = u16x8::splat(c);
            let lt0 = v0.simd_gt(vc);
            let lt1 = v1.simd_gt(vc);
            let lm0 = lt0.to_simd().cast::<u16>();
            let lm1 = lt1.to_simd().cast::<u16>();
            let u = ((v0 | !lm0).simd_min(v1 | !lm1).reduce_min() as u32).min(r);
            let vv = (v0 & !lm0).simd_max(v1 & !lm1).reduce_max() as u32;
            let val = (lt0.to_simd().cast::<i16>() + lt1.to_simd().cast::<i16>())
                .reduce_sum()
                .wrapping_neg() as usize;
            (!lt0, !lt1, u, vv, val)
        };
        let new_dif = self.dif - ((vv as u64) << (WIN_SIZE - 16));
        let new_rng = u - vv;
        let d = (new_rng.leading_zeros() ^ 16) as i32;
        self.dif = new_dif << d;
        self.rng = new_rng << d;
        if self.allow_update {
            let count = cdf[n];
            let rate = u16x8::splat(5 + (count >> 4));
            let gm0 = ge0.to_simd().cast::<u16>();
            let gm1 = ge1.to_simd().cast::<u16>();
            let y0 = x0 + ((f0 >> rate) & gm0) - ((x0 >> rate) & !gm0);
            let y1 = x1 + ((f1 >> rate) & gm1) - ((x1 >> rate) & !gm1);
            *cdf.first_chunk_mut::<8>().unwrap() = *y0.as_array();
            if N > 16 {
                *cdf[8..16].first_chunk_mut::<8>().unwrap() = *y1.as_array();
                cdf[n] = inc_count(count);
            } else if N == 16 {
                let mut ya1 = *y1.as_array();
                ya1[7] = inc_count(count);
                *cdf[8..16].first_chunk_mut::<8>().unwrap() = ya1;
            } else {
                cdf[8..8 + k2].copy_from_slice(&y1.as_array()[..k2]);
                cdf[n] = inc_count(count);
            }
        }
        let cnt = self.cnt;
        self.cnt = cnt - d;
        if (cnt as u32) < (d as u32) {
            return self.refill_ret::<0>(val);
        }
        val
    }

    /// Coefficient `coeff_br` loop: starting from level 3, adds up to four
    /// `coeff_br` symbols (stopping after one below 3). Returns the level.
    #[inline(always)]
    pub fn hi_tok(&mut self, cdf: &mut [u16; 5]) -> u32 {
        if self.allow_update {
            self.hi_tok_c::<true>(cdf)
        } else {
            self.hi_tok_c::<false>(cdf)
        }
    }

    /// Golomb code used for coefficient levels above 14: returns the
    /// decoded value plus one (the spec's `x`), or `None` for a prefix
    /// longer than 32 bits.
    #[inline(always)]
    pub fn golomb_coef(&mut self) -> Option<u32> {
        let state = (self.rng as u64) | (((self.cnt as i8 as u8) as u64) << 16);
        let (dif, out) = golomb_coef_slow::<0>(self.raw.buf, self.raw.pos, self.dif, state);
        self.dif = dif;
        self.rng = (out & 0xFFFF) as u32;
        self.cnt = ((out >> 16) as u8 as i8) as i32;
        self.raw.pos += ((out >> 24) & 0xFF) as usize;
        let x = (out >> 32) as u32;
        if x == 0 {
            None
        } else {
            Some(x)
        }
    }

    /// Reads an equiprobable sign bit followed by `golomb_coef` and
    /// dequantizes an AC coefficient with `tok == 15`. Returns `(q | (1 << 31), dq)`
    /// or `(0, 0)` on an invalid stream.
    #[inline(always)]
    pub fn ac_golomb(&mut self, q2: u32, dq_shift: u32, cf_max: u64) -> (u32, i32) {
        let state = (self.rng as u64) | (((self.cnt as i8 as u8) as u64) << 16);
        let (dif, out, dq) = ac_golomb_slow::<0>(
            self.raw.buf,
            self.raw.pos,
            self.dif,
            state,
            q2,
            dq_shift,
            cf_max,
        );
        self.dif = dif;
        self.rng = (out & 0xFFFF) as u32;
        self.cnt = ((out >> 16) as u8 as i8) as i32;
        self.raw.pos += ((out >> 24) & 0xFF) as usize;
        ((out >> 32) as u32, dq)
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
        let v = ((r >> 8) * (f >> EC_PROB_SHIFT) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB;
        let vw = (v as u64) << (WIN_SIZE - 16);
        let ret = dif >= vw;
        let new_dif = sel(ret, dif.wrapping_sub(vw), dif);
        let new_rng = sel(ret, r.wrapping_sub(v), v);
        if self.allow_update {
            let count = cdf[2];
            let rate = 4 + (count >> 4) as u32;
            cdf[0] = sel(ret, p0 + (f >> rate), p0 - (p0 >> rate)) as u16;
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
        let v = ((r >> 8) << 7) + EC_MIN_PROB;
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
        use std::hint::select_unpredictable as sel;
        let r = self.rng;
        let dif = self.dif;
        let v = ((r >> 8) << 7) + EC_MIN_PROB;
        let vw = (v as u64) << (WIN_SIZE - 16);
        let ret = dif >= vw;
        let new_dif = sel(ret, dif.wrapping_sub(vw), dif);
        let new_rng = sel(ret, r.wrapping_sub(v), v);
        *dst = sel(ret, d, -d);
        self.norm_hot(new_dif, new_rng);
    }

    /// Like `bool_store_neg`, clamping `d` to `cf_max + sign` before negating.
    #[inline(always)]
    pub fn bool_store_neg_qm(&mut self, dst: &mut i32, d: u32, cf_max: u32) {
        use std::hint::select_unpredictable as sel;
        let r = self.rng;
        let dif = self.dif;
        let v = ((r >> 8) << 7) + EC_MIN_PROB;
        let vw = (v as u64) << (WIN_SIZE - 16);
        let ret = dif >= vw;
        let new_dif = sel(ret, dif.wrapping_sub(vw), dif);
        let new_rng = sel(ret, r.wrapping_sub(v), v);
        let pos = d.min(cf_max) as i32;
        let neg = -(d.min(cf_max + 1) as i32);
        *dst = sel(ret, pos, neg);
        self.norm_hot(new_dif, new_rng);
    }

    /// Spec `read_bool()` with out-of-line refill (`norm`), for
    /// `golomb_coef_slow` and `ac_golomb_slow`.
    #[inline(always)]
    fn bool_cold(&mut self) -> bool {
        use std::hint::select_unpredictable as sel;
        let r = self.rng;
        let dif = self.dif;
        let v = ((r >> 8) << 7) + EC_MIN_PROB;
        let vw = (v as u64) << (WIN_SIZE - 16);
        let ret = dif >= vw;
        let new_dif = sel(ret, dif.wrapping_sub(vw), dif);
        let new_rng = sel(ret, r.wrapping_sub(v), v);
        self.norm(new_dif, new_rng);
        !ret
    }

    /// Spec `read_literal(n)` / `L(n)`.
    #[inline]
    pub fn literal(&mut self, n: u32) -> u32 {
        use std::hint::select_unpredictable as sel;
        if n == 0 {
            return 0;
        }
        let mut r = self.rng;
        let mut dif = self.dif;
        let mut cnt = self.cnt;
        let mut pos = self.raw.pos;
        let buf = self.raw.buf;
        let mut x = 0;
        for _ in 0..n {
            let v = ((r >> 8) << 7) + EC_MIN_PROB;
            let vw = (v as u64) << (WIN_SIZE - 16);
            let ret = dif >= vw;
            let new_dif = sel(ret, dif.wrapping_sub(vw), dif);
            let new_rng = sel(ret, r.wrapping_sub(v), v);
            x = (x << 1) | (!ret as u32);
            let d = (new_rng.leading_zeros() ^ 16) as i32;
            dif = new_dif << d;
            r = new_rng << d;
            let old_cnt = cnt;
            cnt = old_cnt - d;
            if (old_cnt as u32) < (d as u32) {
                cold_refill();
                let (d_out, out) = refill_cold::<0>(buf, pos, dif, cnt);
                dif = d_out;
                pos += (out as u32) as usize;
                cnt = (out >> 32) as i32;
            }
        }
        self.raw.pos = pos;
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
        /// (value plus one) or `None` for a prefix longer than 32 bits.
        fn golomb_coef(&mut self) -> Option<u32> {
            let mut length = 0;
            loop {
                length += 1;
                if self.bool() {
                    let mut x = 1u32;
                    for _ in 0..length - 1 {
                        x = (x << 1) | self.bool() as u32;
                    }
                    return Some(x);
                }
                if length > 32 {
                    return None;
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
                let mut br_a = [8000u16, 16000, 24000, 32768, 0];
                let mut br_b = br_a;
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
                            let mut length = 0;
                            let gb = loop {
                                length += 1;
                                if b.symbol(&mut [1 << 14, 1 << 15, 0], false) == 1 {
                                    let mut x = 1u32;
                                    for _ in 0..length - 1 {
                                        x = (x << 1)
                                            | b.symbol(&mut [1 << 14, 1 << 15, 0], false) as u32;
                                    }
                                    break Some(x);
                                }
                                if length > 32 {
                                    break None;
                                }
                            };
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
                let mut c5a = [8000u16, 16000, 24000, 32768, 0];
                let mut c5b = c5a;
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
}
