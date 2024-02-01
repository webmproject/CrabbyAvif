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
    pub fn size(&self) -> usize {
        match self {
            Pixels::Pointer(_) => 0,
            Pixels::Buffer(buffer) => buffer.len(),
            Pixels::Buffer16(buffer) => buffer.len(),
        }
    }

    pub fn has_data(&self) -> bool {
        match self {
            Pixels::Pointer(ptr) => !ptr.is_null(),
            Pixels::Buffer(buffer) => !buffer.is_empty(),
            Pixels::Buffer16(buffer) => !buffer.is_empty(),
        }
    }

    pub fn resize(&mut self, size: usize, default: u16) -> AvifResult<()> {
        match self {
            Pixels::Pointer(_) => {}
            Pixels::Buffer(buffer) => {
                if buffer.capacity() < size {
                    if buffer.try_reserve_exact(size).is_err() {
                        return Err(AvifError::OutOfMemory);
                    }
                }
                buffer.resize(size, default as u8);
            }
            Pixels::Buffer16(buffer) => {
                if buffer.capacity() < size {
                    if buffer.try_reserve_exact(size).is_err() {
                        return Err(AvifError::OutOfMemory);
                    }
                }
                buffer.resize(size, default);
            }
        }
        Ok(())
    }

    pub fn is_pointer(&self) -> bool {
        matches!(self, Pixels::Pointer(_))
    }

    pub fn pointer(&self) -> *mut u8 {
        match self {
            Pixels::Pointer(ptr) => *ptr,
            _ => std::ptr::null_mut(),
        }
    }

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

    pub fn slices(&self, offset: u32, size: u32) -> AvifResult<(Option<&[u8]>, Option<&[u16]>)> {
        match self {
            Pixels::Pointer(ptr) => {
                let offset = isize_from_u32(offset)?;
                let size = usize_from_u32(size)?;
                let ptr16 = (*ptr) as *const u16;
                unsafe {
                    Ok((
                        Some(std::slice::from_raw_parts(ptr.offset(offset), size)),
                        Some(std::slice::from_raw_parts(ptr16.offset(offset), size)),
                    ))
                }
            }
            Pixels::Buffer(_) => Ok((Some(self.slice(offset, size)?), None)),
            Pixels::Buffer16(_) => Ok((None, Some(self.slice16(offset, size / 2)?))),
        }
    }

    pub fn slices_mut(
        &mut self,
        offset: u32,
        size: u32,
    ) -> AvifResult<(Option<&mut [u8]>, Option<&mut [u16]>)> {
        match self {
            Pixels::Pointer(ptr) => {
                let offset = isize_from_u32(offset)?;
                let size = usize_from_u32(size)?;
                let ptr16 = (*ptr) as *mut u16;
                unsafe {
                    Ok((
                        Some(std::slice::from_raw_parts_mut(ptr.offset(offset), size)),
                        Some(std::slice::from_raw_parts_mut(ptr16.offset(offset), size)),
                    ))
                }
            }
            Pixels::Buffer(_) => Ok((Some(self.slice_mut(offset, size)?), None)),
            Pixels::Buffer16(_) => Ok((None, Some(self.slice16_mut(offset, size / 2)?))),
        }
    }
}
