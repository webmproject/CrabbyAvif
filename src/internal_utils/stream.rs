use crate::internal_utils::*;
use crate::AvifError;
use crate::AvifResult;

#[derive(Debug)]
pub struct IBitStream<'a> {
    pub data: &'a [u8],
    pub bit_offset: usize,
}

impl IBitStream<'_> {
    fn read_bit(&mut self) -> AvifResult<u8> {
        let byte_offset = self.bit_offset / 8;
        if byte_offset >= self.data.len() {
            return Err(AvifError::BmffParseFailed);
        }
        let byte = self.data[byte_offset];
        let shift = 7 - (self.bit_offset % 8);
        self.bit_offset += 1;
        // println!(
        //     "read bit at offset {} is {}",
        //     self.bit_offset - 1,
        //     (byte >> shift) & 0x01
        // );
        Ok((byte >> shift) & 0x01)
    }

    pub fn read(&mut self, n: usize) -> AvifResult<u32> {
        assert!(n <= 32);
        let mut value: u32 = 0;
        for _i in 0..n {
            value <<= 1;
            value |= self.read_bit()? as u32;
        }
        // println!("read byte({n}): {value}");
        Ok(value)
    }

    pub fn read_bool(&mut self) -> AvifResult<bool> {
        let bit = self.read_bit()?;
        // println!("read bool: {}", bit == 1);
        Ok(bit == 1)
    }

    pub fn skip(&mut self, n: usize) {
        self.bit_offset += n;
    }

    pub fn skip_uvlc(&mut self) -> AvifResult<()> {
        let mut bit_count = 0;
        while !self.read_bool()? {
            bit_count += 1;
            if bit_count == 32 {
                return Err(AvifError::BmffParseFailed);
            }
        }
        self.skip(bit_count);
        Ok(())
    }
}

#[derive(Debug)]
pub struct IStream<'a> {
    pub data: &'a [u8],
    pub offset: usize,
}

impl IStream<'_> {
    pub fn create(data: &[u8]) -> IStream {
        IStream { data, offset: 0 }
    }

    fn check(&self, size: usize) -> AvifResult<()> {
        if self.bytes_left() < size {
            return Err(AvifError::BmffParseFailed);
        }
        Ok(())
    }

    pub fn sub_stream(&mut self, size: usize) -> AvifResult<IStream> {
        self.check(size)?;
        let offset = self.offset;
        self.offset += size;
        Ok(IStream {
            data: &self.data[offset..self.offset],
            offset: 0,
        })
    }

    pub fn sub_bit_stream(&mut self, size: usize) -> AvifResult<IBitStream> {
        self.check(size)?;
        let offset = self.offset;
        self.offset += size;
        Ok(IBitStream {
            data: &self.data[offset..self.offset],
            bit_offset: 0,
        })
    }

    pub fn bytes_left(&self) -> usize {
        self.data.len() - self.offset
    }

    pub fn has_bytes_left(&self) -> bool {
        self.bytes_left() > 0
    }

    pub fn get_slice(&mut self, size: usize) -> AvifResult<&[u8]> {
        self.check(size)?;
        let offset_start = self.offset;
        self.offset += size;
        Ok(&self.data[offset_start..offset_start + size])
    }

    fn get_vec(&mut self, size: usize) -> AvifResult<Vec<u8>> {
        Ok(self.get_slice(size)?.to_vec())
    }

    pub fn read_u8(&mut self) -> AvifResult<u8> {
        self.check(1)?;
        let value = self.data[self.offset];
        self.offset += 1;
        Ok(value)
    }

    pub fn read_u16(&mut self) -> AvifResult<u16> {
        Ok(u16::from_be_bytes(self.get_slice(2)?.try_into().unwrap()))
    }

    pub fn read_u24(&mut self) -> AvifResult<u32> {
        Ok(self.read_uxx(3)? as u32)
    }

    pub fn read_u32(&mut self) -> AvifResult<u32> {
        Ok(u32::from_be_bytes(self.get_slice(4)?.try_into().unwrap()))
    }

    pub fn read_u64(&mut self) -> AvifResult<u64> {
        Ok(u64::from_be_bytes(self.get_slice(8)?.try_into().unwrap()))
    }

    pub fn read_i32(&mut self) -> AvifResult<i32> {
        // For now this is used only for gainmap fractions where we need
        // wrapping conversion from u32 to i32.
        Ok(self.read_u32()? as i32)
        //i32::try_from(val).or(Err(AvifError::BmffParseFailed))
    }

    pub fn skip_u32(&mut self) -> AvifResult<()> {
        self.skip(4)
    }

    pub fn skip_u64(&mut self) -> AvifResult<()> {
        self.skip(8)
    }

    pub fn read_fraction(&mut self) -> AvifResult<Fraction> {
        Ok(Fraction(self.read_i32()?, self.read_u32()?))
    }

    pub fn read_ufraction(&mut self) -> AvifResult<UFraction> {
        Ok(UFraction(self.read_u32()?, self.read_u32()?))
    }

    pub fn read_string(&mut self, size: usize) -> AvifResult<String> {
        String::from_utf8(self.get_vec(size)?).or(Err(AvifError::BmffParseFailed))
    }

    pub fn read_uxx(&mut self, xx: u8) -> AvifResult<u64> {
        let n: usize = xx.into();
        if n == 0 {
            return Ok(0);
        }
        if n > 8 {
            return Err(AvifError::NotImplemented);
        }
        let mut out = [0; 8];
        let start = out.len() - n;
        out[start..].copy_from_slice(self.get_slice(n)?);
        Ok(u64::from_be_bytes(out))
    }

    pub fn read_c_string(&mut self) -> AvifResult<String> {
        self.check(1)?;
        let null_position = self.data[self.offset..]
            .iter()
            .position(|&x| x == 0)
            .ok_or(AvifError::BmffParseFailed)?;
        let range = self.offset..self.offset + null_position;
        self.offset += null_position + 1;
        String::from_utf8(self.data[range].to_vec()).or(Err(AvifError::BmffParseFailed))
    }

    pub fn read_version_and_flags(&mut self) -> AvifResult<(u8, u32)> {
        let version = self.read_u8()?;
        let flags = self.read_u24()?;
        Ok((version, flags))
    }

    pub fn read_and_enforce_version_and_flags(
        &mut self,
        enforced_version: u8,
    ) -> AvifResult<(u8, u32)> {
        let (version, flags) = self.read_version_and_flags()?;
        if version != enforced_version {
            return Err(AvifError::BmffParseFailed);
        }
        Ok((version, flags))
    }

    pub fn skip(&mut self, size: usize) -> AvifResult<()> {
        self.check(size)?;
        self.offset += size;
        Ok(())
    }

    pub fn rewind(&mut self, size: usize) -> AvifResult<()> {
        self.offset = self
            .offset
            .checked_sub(size)
            .ok_or(AvifError::BmffParseFailed)?;
        Ok(())
    }

    pub fn read_uleb128(&mut self) -> AvifResult<u32> {
        let mut val: u64 = 0;
        for i in 0..8 {
            let byte = self.read_u8()?;
            val |= ((byte & 0x7F) << (i * 7)) as u64;
            if (byte & 0x80) == 0 {
                return u32_from_u64(val);
            }
        }
        println!("uleb value did not terminate after 8 bytes");
        Err(AvifError::BmffParseFailed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_uxx() {
        let mut stream = IStream::create(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(stream.read_uxx(0), Ok(0));
        assert_eq!(stream.offset, 0);
        assert_eq!(stream.read_uxx(1), Ok(1));
        assert_eq!(stream.offset, 1);
        stream.offset = 0;
        assert_eq!(stream.read_uxx(2), Ok(258));
        stream.offset = 0;
        assert_eq!(stream.read_u16(), Ok(258));
        stream.offset = 0;
        assert_eq!(stream.read_uxx(3), Ok(66051));
        stream.offset = 0;
        assert_eq!(stream.read_u24(), Ok(66051));
        stream.offset = 0;
        assert_eq!(stream.read_uxx(4), Ok(16909060));
        stream.offset = 0;
        assert_eq!(stream.read_u32(), Ok(16909060));
        stream.offset = 0;
        assert_eq!(stream.read_uxx(5), Ok(4328719365));
        stream.offset = 0;
        assert_eq!(stream.read_uxx(6), Ok(1108152157446));
        stream.offset = 0;
        assert_eq!(stream.read_uxx(7), Ok(283686952306183));
        stream.offset = 0;
        assert_eq!(stream.read_uxx(8), Ok(72623859790382856));
        stream.offset = 0;
        assert_eq!(stream.read_u64(), Ok(72623859790382856));
        stream.offset = 0;
        assert_eq!(stream.read_uxx(9), Err(AvifError::NotImplemented));
    }
}
