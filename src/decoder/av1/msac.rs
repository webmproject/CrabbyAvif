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

#[cold]
#[inline(never)]
fn golomb_coef_slow(buf: &[u8], pos: usize, dif: u64, state: u64) -> (u64, u64) {
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
        if s.bool() {
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
        x = (x << 1) | s.bool() as u32;
    }
    let out = (s.rng as u64)
        | (((s.cnt as i8 as u8) as u64) << 16)
        | (((s.raw.pos - pos) as u64) << 24)
        | ((x as u64) << 32);
    (s.dif, out)
}

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

#[derive(Clone)]
pub struct SymbolDecoder<'a> {
    raw: ByteReader<'a>,
    dif: u64,
    rng: u32,
    cnt: i32,
    allow_update: bool,
}

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
            self.refill_cold();
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
            self.refill();
        }
    }

    #[cold]
    #[inline(never)]
    fn refill_cold(&mut self) {
        self.refill();
    }

    /// Decodes a boolean with probability `f / 32768` of being 0, where `f`
    /// is the inverted CDF value (`32768 - cdf[0]`).
    #[inline(always)]
    fn bool_inv(&mut self, f: u32) -> bool {
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
        let c = (self.dif >> (WIN_SIZE - 16)) as u32;
        let r = self.rng >> 8;
        if N == 4 {
            const ONES: u32 = 0x0001_0001;
            let x = (cdf[0] as u32) | ((cdf[1] as u32) << 16);
            let f = (0x8000 * ONES).wrapping_sub(x);
            let v0 = ((r * ((f >> 6) & 0x3ff)) >> 1) + 8;
            let v1 = ((r * (f >> 22)) >> 1) + 4;
            let c0 = c < v0;
            let c1 = c < v1;
            let val = c0 as usize + c1 as usize;
            let u = if c1 {
                v1
            } else if c0 {
                v0
            } else {
                self.rng
            };
            let v = if c1 {
                0
            } else if c0 {
                v1
            } else {
                v0
            };
            if self.allow_update {
                let count = cdf[3];
                let rate = 4 + (count >> 4) as u32;
                let mr = (0xFFFFu32 >> rate) * ONES;
                let m = ((1u64 << (16 * val)) - 1) as u32;
                let top = m | (0x8000 * ONES);
                let y = x
                    .wrapping_add((top.wrapping_sub(x) >> rate) & mr)
                    .wrapping_sub(m & mr);
                cdf[0] = y as u16;
                cdf[1] = (y >> 16) as u16;
                cdf[3] = count + 1 - (count >> 5);
            }
            self.norm(self.dif - ((v as u64) << (WIN_SIZE - 16)), u - v);
            return val;
        }
        if N == 5 {
            let upd = self.allow_update;
            return self.symbol5(cdf.first_chunk_mut().unwrap(), upd);
        }
        // vs[0] = rng, vs[i + 1] = v(i); v(n - 1) = 0 since cdf[n - 1]
        // is 32768. v(i) is strictly decreasing for monotone CDFs, so
        // c < vs[i + 1] holds for a prefix 0..val.
        let mut vs = [0u16; 17];
        vs[0] = self.rng as u16;
        let mut val = 0usize;
        for i in 0..n - 1 {
            let f = 32768 - cdf[i] as u32;
            let vi = ((r * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
                + EC_MIN_PROB * (n - 1 - i) as u32;
            vs[i + 1] = vi as u16;
            val += (c < vi) as usize;
        }
        let idx = val & 15;
        let u = vs[idx] as u32;
        let v = vs[idx + 1] as u32;
        if self.allow_update {
            let count = cdf[n];
            let rate = 5 + (count >> 4) as u32;
            let c16 = c as u16;
            for (p, &vi) in cdf[..n - 1].iter_mut().zip(&vs[1..n]) {
                let x = *p as i32;
                let m = -((c16 < vi) as i32);
                *p = (x - m + (((m | 32768) - x) >> rate)) as u16;
            }
            cdf[n] = count + 1 - (count >> 5);
        }
        self.norm(self.dif - ((v as u64) << (WIN_SIZE - 16)), u - v);
        val
    }

    /// `symbol_n::<5>` (coefficient tokens). `upd` is the CDF adaptation
    /// flag, passed as a constant by the specialized coefficient loops.
    #[inline(always)]
    fn symbol5(&mut self, cdf: &mut [u16; 5], upd: bool) -> usize {
        let (val, y) = self.symbol5_r(Cdf5::load(cdf), upd);
        if upd {
            y.store(cdf);
        }
        val as usize
    }

    /// `symbol_n::<5>` on a CDF held in registers: returns the symbol and
    /// the adapted CDF (unchanged unless `upd`).
    #[inline(always)]
    pub fn symbol5_r(&mut self, cdf: Cdf5, upd: bool) -> (u32, Cdf5) {
        let c = (self.dif >> (WIN_SIZE - 16)) as u32;
        let r = self.rng >> 8;
        const ONES: u64 = 0x0001_0001_0001_0001;
        let x = cdf.x;
        let f = (0x8000 * ONES).wrapping_sub(x);
        let v0 = ((r * (((f as u32) >> 6) & 0x3ff)) >> 1) + 12;
        let v1 = ((r * (((f >> 22) as u32) & 0x3ff)) >> 1) + 8;
        let v2 = ((r * (((f >> 38) as u32) & 0x3ff)) >> 1) + 4;
        let c0 = c < v0;
        let c1 = c < v1;
        let c2 = c < v2;
        let val = c0 as u32 + c1 as u32 + c2 as u32;
        let u = if c2 {
            v2
        } else if c1 {
            v1
        } else if c0 {
            v0
        } else {
            self.rng
        };
        let v = if c2 {
            0
        } else if c1 {
            v2
        } else if c0 {
            v1
        } else {
            v0
        };
        let y = if upd {
            let count = cdf.count;
            let rate = 5 + (count >> 4) as u32;
            let mr = (0xFFFFu64 >> rate) * ONES;
            let m = (1u64 << (16 * val)) - 1;
            let top = m | (0x8000 * ONES);
            let y = x
                .wrapping_add((top.wrapping_sub(x) >> rate) & mr)
                .wrapping_sub(m & mr);
            Cdf5 {
                x: y,
                count: count + (count < 32) as u16,
            }
        } else {
            cdf
        };
        self.norm_hot(self.dif - ((v as u64) << (WIN_SIZE - 16)), u - v);
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
        self.symbol5(cdf, UPD) as u32
    }

    /// `hi_tok` for loops specialized on the adaptation flag.
    #[inline(always)]
    pub fn hi_tok_c<const UPD: bool>(&mut self, cdf: &mut [u16; 5]) -> u32 {
        let mut tok = 3u32;
        for _ in 0..4 {
            let br = self.symbol5(cdf, UPD) as u32;
            tok += br;
            if br < 3 {
                break;
            }
        }
        tok
    }

    /// Coefficient `coeff_br` loop: starting from level 3, adds up to four
    /// `coeff_br` symbols (stopping after one below 3). Returns the level.
    #[inline(always)]
    pub fn hi_tok(&mut self, cdf: &mut [u16; 5]) -> u32 {
        let mut tok = 3u32;
        for _ in 0..4 {
            let br = self.symbol_n(cdf) as u32;
            tok += br;
            if br < 3 {
                break;
            }
        }
        tok
    }

    /// Golomb code used for coefficient levels above 14: returns the
    /// decoded value plus one (the spec's `x`), or `None` for a prefix
    /// longer than 32 bits.
    #[inline(always)]
    pub fn golomb_coef(&mut self) -> Option<u32> {
        let state = (self.rng as u64) | (((self.cnt as i8 as u8) as u64) << 16);
        let (dif, out) = golomb_coef_slow(self.raw.buf, self.raw.pos, self.dif, state);
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
            cdf[2] = count + 1 - (count >> 5);
        }
        self.norm(new_dif, new_rng);
        !ret
    }

    /// Spec `read_bool()` (equiprobable, no adaptation).
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

    /// Spec `read_literal(n)` / `L(n)`.
    #[inline]
    pub fn literal(&mut self, n: u32) -> u32 {
        let mut x = 0;
        for _ in 0..n {
            x = 2 * x + self.bool() as u32;
        }
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

    /// Spec `read_golomb()` used by coefficient decoding.
    pub fn golomb(&mut self) -> u32 {
        let mut num_bits = 0u32;
        while !self.bool() {
            num_bits += 1;
            if num_bits >= 32 {
                // Invalid stream; avoid overflow.
                return u32::MAX >> 1;
            }
        }
        let mut x = 1u32;
        for _ in 0..num_bits {
            x = 2 * x + self.bool() as u32;
        }
        x - 1
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
}
