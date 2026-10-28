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

//! Bit reader for the header syntax (spec 4.10 descriptors and 8.1).

use crate::decoder::av1::error::{Error, Result};

/// MSB-first bit reader over a byte slice.
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Position in bits.
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Current position in bits (spec `get_position()`).
    #[inline]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Current position in whole bytes (rounded up).
    #[inline]
    pub fn byte_position(&self) -> usize {
        self.pos.div_ceil(8)
    }

    #[inline]
    pub fn bit(&mut self) -> Result<u32> {
        let byte = *self.data.get(self.pos >> 3).ok_or(Error::Truncated)?;
        let bit = (byte >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        Ok(bit as u32)
    }

    #[inline]
    pub fn flag(&mut self) -> Result<bool> {
        Ok(self.bit()? != 0)
    }

    /// `f(n)` for n <= 32.
    pub fn f(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32);
        let mut x: u64 = 0;
        for _ in 0..n {
            x = (x << 1) | self.bit()? as u64;
        }
        Ok(x as u32)
    }

    /// `su(n)`: n-bit two's complement signed value.
    pub fn su(&mut self, n: u32) -> Result<i32> {
        let value = self.f(n)? as i64;
        let sign_mask = 1i64 << (n - 1);
        let v = if value & sign_mask != 0 { value - 2 * sign_mask } else { value };
        Ok(v as i32)
    }

    /// `ns(n)`: non-symmetric unsigned value in `0..n`.
    pub fn ns(&mut self, n: u32) -> Result<u32> {
        if n <= 1 {
            return Ok(0);
        }
        let w = crate::decoder::av1::pixel::floor_log2(n) + 1;
        let m = (1u32 << w) - n;
        let v = self.f(w - 1)?;
        if v < m {
            return Ok(v);
        }
        let extra_bit = self.bit()?;
        Ok((v << 1) - m + extra_bit)
    }

    /// `le(n)`: n-byte little endian value.
    pub fn le(&mut self, n: u32) -> Result<u64> {
        let mut t = 0u64;
        for i in 0..n {
            t += (self.f(8)? as u64) << (i * 8);
        }
        Ok(t)
    }

    /// `leb128()`.
    pub fn leb128(&mut self) -> Result<u64> {
        let mut value = 0u64;
        for i in 0..8 {
            let byte = self.f(8)? as u64;
            value |= (byte & 0x7f) << (i * 7);
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(Error::Invalid("leb128 too long"))
    }

    /// `uvlc()`.
    pub fn uvlc(&mut self) -> Result<u32> {
        let mut leading_zeros = 0u32;
        while self.bit()? == 0 {
            leading_zeros += 1;
            if leading_zeros >= 32 {
                return Ok(u32::MAX);
            }
        }
        let value = self.f(leading_zeros)?;
        Ok(value.wrapping_add((1u32 << leading_zeros).wrapping_sub(1)))
    }

    /// `byte_alignment()`.
    pub fn byte_align(&mut self) {
        self.pos = (self.pos + 7) & !7;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_reads() {
        let data = [0b1010_0000, 0xff, 0x01];
        let mut r = BitReader::new(&data);
        assert_eq!(r.f(3).unwrap(), 0b101);
        assert_eq!(r.su(2).unwrap(), 0);
        r.byte_align();
        assert_eq!(r.position(), 8);
        assert_eq!(r.leb128().unwrap(), 0x7f | (1 << 7));
        assert!(r.bit().is_err());
    }

    #[test]
    fn ns_values() {
        // ns(5): w = 3, m = 3. v = f(2); v < 3 -> v, else (v << 1) - 3 + bit.
        let data = [0b1110_0000];
        let mut r = BitReader::new(&data);
        assert_eq!(r.ns(5).unwrap(), 4);
    }
}
