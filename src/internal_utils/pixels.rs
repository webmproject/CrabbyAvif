use crate::internal_utils::*;
use crate::*;

pub enum Pixels {
    // Intended for use from the C API. Used for all bitdepths.
    Pointer(*mut u8),
    // Used for 8-bit images.
    Buffer(Vec<u8>),
    // Used for 10-bit, 12-bit and 16-bit images.
    Buffer16(Vec<u16>),
}

impl Pixels {
    pub fn slice(&self, offset: u32, size: u32) -> AvifResult<&[u8]> {
        let offset: usize = usize_from_u32(offset)?;
        let size: usize = usize_from_u32(size)?;
        match self {
            Pixels::Pointer(ptr) => {
                let offset = isize_from_usize(offset)?;
                Ok(unsafe { std::slice::from_raw_parts(ptr.offset(offset), size) })
            }
            Pixels::Buffer(buffer) => {
                let end = offset.checked_add(size).ok_or(AvifError::NoContent)?;
                Ok(&buffer[offset..end])
            }
            Pixels::Buffer16(_) => Err(AvifError::NoContent),
        }
    }

    pub fn slice_mut(&mut self, offset: u32, size: u32) -> AvifResult<&mut [u8]> {
        let offset: usize = usize_from_u32(offset)?;
        let size: usize = usize_from_u32(size)?;
        match self {
            Pixels::Pointer(ptr) => {
                let offset = isize_from_usize(offset)?;
                Ok(unsafe { std::slice::from_raw_parts_mut(ptr.offset(offset), size) })
            }
            Pixels::Buffer(buffer) => {
                let end = offset.checked_add(size).ok_or(AvifError::NoContent)?;
                Ok(&mut buffer[offset..end])
            }
            Pixels::Buffer16(_) => Err(AvifError::NoContent),
        }
    }

    pub fn slice16(&self, offset: u32, size: u32) -> AvifResult<&[u16]> {
        let offset: usize = usize_from_u32(offset)?;
        let size: usize = usize_from_u32(size)?;
        match self {
            Pixels::Pointer(ptr) => {
                let offset = isize_from_usize(offset)?;
                let ptr = (*ptr) as *const u16;
                Ok(unsafe { std::slice::from_raw_parts(ptr.offset(offset), size) })
            }
            Pixels::Buffer(_) => Err(AvifError::NoContent),
            Pixels::Buffer16(buffer) => {
                let end = offset.checked_add(size).ok_or(AvifError::NoContent)?;
                Ok(&buffer[offset..end])
            }
        }
    }

    pub fn slice16_mut(&mut self, offset: u32, size: u32) -> AvifResult<&mut [u16]> {
        let offset: usize = usize_from_u32(offset)?;
        let size: usize = usize_from_u32(size)?;
        match self {
            Pixels::Pointer(ptr) => {
                let offset = isize_from_usize(offset)?;
                let ptr = (*ptr) as *mut u16;
                Ok(unsafe { std::slice::from_raw_parts_mut(ptr.offset(offset), size) })
            }
            Pixels::Buffer(_) => Err(AvifError::NoContent),
            Pixels::Buffer16(buffer) => {
                let end = offset.checked_add(size).ok_or(AvifError::NoContent)?;
                Ok(&mut buffer[offset..end])
            }
        }
    }
}
