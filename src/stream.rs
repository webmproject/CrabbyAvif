use byteorder::{BigEndian, ByteOrder};

#[derive(Debug)]
pub struct IStream {
    pub data: Vec<u8>,
    pub offset: usize,
}

#[derive(Debug)]
pub struct BitReader {
    byte: u8,
    offset: u8,
}

impl BitReader {
    pub fn read(&mut self, n: u8) -> u8 {
        let shift: u8 = 8 - n - self.offset;
        let mask: u8 = (1 << n) - 1;
        self.offset += n;
        (self.byte >> shift) & mask
    }
}

impl IStream {
    fn done(&self) -> bool {
        self.offset >= self.data.len()
    }

    fn get_slice(&mut self, size: usize) -> &[u8] {
        let offset_start = self.offset;
        self.offset += size;
        &self.data[offset_start..offset_start + size]
    }

    fn get_vec(&mut self, size: usize) -> Vec<u8> {
        self.get_slice(size).to_vec()
    }

    // TODO: should these functions return Option

    pub fn read_u8(&mut self) -> u8 {
        self.offset += 1;
        self.data[self.offset - 1]
    }

    pub fn read_u16(&mut self) -> u16 {
        BigEndian::read_u16(self.get_slice(2))
    }

    pub fn read_u24(&mut self) -> u32 {
        BigEndian::read_u24(self.get_slice(3))
    }

    pub fn read_u32(&mut self) -> u32 {
        BigEndian::read_u32(self.get_slice(4))
    }

    pub fn read_u64(&mut self) -> u64 {
        BigEndian::read_u64(self.get_slice(8))
    }

    pub fn read_string(&mut self, size: usize) -> String {
        String::from_utf8(self.get_vec(size)).unwrap()
    }

    pub fn read_uxx(&mut self, xx: u8) -> u64 {
        if xx == 0 {
            return 0;
        } else if xx == 4 {
            return self.read_u32() as u64;
        } else {
            panic!("read-uxx with {xx}. whoa!");
        }
        0
    }

    pub fn read_c_string(&mut self) -> String {
        // TODO: handle none.
        let null_position = self.data[self.offset..]
            .iter()
            .position(|&x| x == 0)
            .unwrap();
        let range = self.offset..self.offset + null_position;
        self.offset += null_position + 1;
        String::from_utf8(self.data[range].to_vec()).unwrap()
    }

    pub fn read_version_and_flags(&mut self) -> (u8, u32) {
        // TODO: this must also add an option to enforce version.
        (self.read_u8(), self.read_u24())
    }

    pub fn skip(&mut self, size: usize) {
        self.offset += size;
    }

    pub fn get_bitreader(&mut self) -> BitReader {
        BitReader {
            byte: self.read_u8(),
            offset: 0,
        }
    }
}
