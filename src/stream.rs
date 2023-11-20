use crate::AvifError;
use crate::AvifResult;
use byteorder::{BigEndian, ByteOrder};

#[derive(Debug)]
pub struct BitReader {
    byte: u8,
    offset: u8,
}

impl BitReader {
    pub fn read(&mut self, n: u8) -> u8 {
        assert!(n <= 8);
        let shift: u8 = 8 - n - self.offset;
        let mask: u8 = (1 << n) - 1;
        self.offset += n;
        (self.byte >> shift) & mask
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
        Ok(BigEndian::read_u16(self.get_slice(2)?))
    }

    pub fn read_u24(&mut self) -> AvifResult<u32> {
        Ok(BigEndian::read_u24(self.get_slice(3)?))
    }

    pub fn read_u32(&mut self) -> AvifResult<u32> {
        Ok(BigEndian::read_u32(self.get_slice(4)?))
    }

    pub fn read_u64(&mut self) -> AvifResult<u64> {
        Ok(BigEndian::read_u64(self.get_slice(8)?))
    }

    pub fn skip_u32(&mut self) -> AvifResult<()> {
        self.skip(4)
    }

    pub fn skip_u64(&mut self) -> AvifResult<()> {
        self.skip(8)
    }

    pub fn read_string(&mut self, size: usize) -> AvifResult<String> {
        String::from_utf8(self.get_vec(size)?).or(Err(AvifError::BmffParseFailed))
    }

    pub fn read_uxx(&mut self, xx: u8) -> AvifResult<u64> {
        let value: u64;
        if xx == 0 {
            value = 0;
        } else if xx == 4 {
            value = self.read_u32()? as u64;
        } else {
            return Err(AvifError::NotImplemented);
        }
        Ok(value)
    }

    pub fn read_c_string(&mut self) -> AvifResult<String> {
        let null_position = self.data[self.offset..]
            .iter()
            .position(|&x| x == 0)
            .ok_or(AvifError::BmffParseFailed)?;
        self.check(null_position + 1)?;
        let range = self.offset..self.offset + null_position;
        self.offset += null_position + 1;
        String::from_utf8(self.data[range].to_vec()).or(Err(AvifError::BmffParseFailed))
    }

    pub fn read_version_and_flags(&mut self) -> AvifResult<(u8, u32)> {
        // TODO: this must also add an option to enforce version.
        let version = self.read_u8()?;
        let flags = self.read_u24()?;
        Ok((version, flags))
    }

    pub fn skip(&mut self, size: usize) -> AvifResult<()> {
        self.check(size)?;
        self.offset += size;
        Ok(())
    }

    // TODO: rename this function and bitreader struct.
    pub fn get_bitreader(&mut self) -> AvifResult<BitReader> {
        let byte = self.read_u8()?;
        Ok(BitReader { byte, offset: 0 })
    }
}
