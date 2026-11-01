// Copyright 2024 Google LLC
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

use crate::internal_utils::*;
use crate::*;

// This struct must not be derived from the default `Clone` trait as it has to be cloned with error
// checking using the `try_clone` function.
#[derive(Debug)]
pub enum Pixels {
    // Intended for holding data from underlying native libraries. Used for 8-bit images.
    Pointer(PointerSlice<u8>),
    // Intended for holding data from underlying native libraries. Used for 10-bit, 12-bit and
    // 16-bit images.
    Pointer16(PointerSlice<u16>),
    // Used for 8-bit images.
    Buffer(Vec<u8>),
    // Used for 10-bit, 12-bit and 16-bit images.
    Buffer16(Vec<u16>),
}

impl Pixels {
    // This function may not be used in all configurations.
    #[allow(dead_code)]
    pub(crate) fn from_raw_pointer(
        ptr: *mut u8,
        depth: u32,
        height: u32,
        mut row_bytes: u32,
    ) -> AvifResult<Self> {
        if depth > 8 {
            if !row_bytes.is_multiple_of(2) {
                return AvifError::no_content();
            }
            row_bytes /= 2;
        }
        let size = usize_from_u32(checked_mul!(height, row_bytes)?)?;
        Self::from_raw_pointer_and_size(ptr, depth, size)
    }

    // This function may not be used in all configurations.
    #[allow(dead_code)]
    pub(crate) fn from_raw_pointer_and_size(
        ptr: *mut u8,
        depth: u32,
        size: usize,
    ) -> AvifResult<Self> {
        if depth > 8 {
            let ptr16 = ptr as *mut u16;
            // SAFETY: Crate-internal callers (C API wrappers, native codec outputs, and scoped
            // image views) pass a pointer to a single live allocation of at least `size`
            // initialized `u16` elements that outlives the returned `Pixels` and obeys Rust's
            // shared/mutable borrowing rules through `Pixels` accessors. Null pointers, unaligned
            // pointers, zero sizes, and byte sizes exceeding `isize::MAX` are rejected by
            // `PointerSlice::create`.
            let slice = unsafe { PointerSlice::create(ptr16, size) }?;
            Ok(Pixels::Pointer16(slice))
        } else {
            // SAFETY: Crate-internal callers (C API wrappers, native codec outputs, and scoped
            // image views) pass a pointer to a single live allocation of at least `size`
            // initialized `u8` bytes that outlives the returned `Pixels` and obeys Rust's
            // shared/mutable borrowing rules through `Pixels` accessors. Null pointers, zero
            // sizes, and sizes exceeding `isize::MAX` are rejected by `PointerSlice::create`.
            let slice = unsafe { PointerSlice::create(ptr, size) }?;
            Ok(Pixels::Pointer(slice))
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Pixels::Pointer(_) => 0,
            Pixels::Pointer16(_) => 0,
            Pixels::Buffer(buffer) => buffer.len(),
            Pixels::Buffer16(buffer) => buffer.len(),
        }
    }

    pub(crate) fn has_data(&self) -> bool {
        match self {
            Pixels::Pointer(ptr) => !ptr.is_empty(),
            Pixels::Pointer16(ptr) => !ptr.is_empty(),
            Pixels::Buffer(buffer) => !buffer.is_empty(),
            Pixels::Buffer16(buffer) => !buffer.is_empty(),
        }
    }

    pub(crate) fn resize(&mut self, size: usize, default: u16) -> AvifResult<()> {
        match self {
            Pixels::Pointer(_) => return AvifError::invalid_argument(),
            Pixels::Pointer16(_) => return AvifError::invalid_argument(),
            Pixels::Buffer(buffer) => {
                if buffer.len() < size {
                    buffer
                        .try_reserve_exact(size - buffer.len())
                        .map_err(AvifError::map_out_of_memory)?;
                }
                #[allow(clippy::disallowed_methods)]
                buffer.resize(size, default as u8);
            }
            Pixels::Buffer16(buffer) => {
                if buffer.len() < size {
                    buffer
                        .try_reserve_exact(size - buffer.len())
                        .map_err(AvifError::map_out_of_memory)?;
                }
                #[allow(clippy::disallowed_methods)]
                buffer.resize(size, default);
            }
        }
        Ok(())
    }

    // This function may not be used in all configurations.
    #[allow(dead_code)]
    pub(crate) fn ptr(&self) -> *const u8 {
        match self {
            Pixels::Pointer(ptr) => ptr.ptr(),
            Pixels::Buffer(buffer) => buffer.as_ptr(),
            _ => std::ptr::null(),
        }
    }

    // This function may not be used in all configurations.
    #[allow(dead_code)]
    pub(crate) fn ptr16(&self) -> *const u16 {
        match self {
            Pixels::Pointer16(ptr) => ptr.ptr(),
            Pixels::Buffer16(buffer) => buffer.as_ptr(),
            _ => std::ptr::null(),
        }
    }

    // This function may not be used in all configurations.
    #[allow(dead_code)]
    pub(crate) fn ptr_mut(&mut self) -> *mut u8 {
        match self {
            Pixels::Pointer(ptr) => ptr.ptr_mut(),
            Pixels::Buffer(buffer) => buffer.as_mut_ptr(),
            _ => std::ptr::null_mut(),
        }
    }

    // This function may not be used in all configurations.
    #[allow(dead_code)]
    pub(crate) fn ptr16_mut(&mut self) -> *mut u16 {
        match self {
            Pixels::Pointer16(ptr) => ptr.ptr_mut(),
            Pixels::Buffer16(buffer) => buffer.as_mut_ptr(),
            _ => std::ptr::null_mut(),
        }
    }

    pub(crate) fn ptr_generic(&self) -> *const u8 {
        match self {
            Pixels::Pointer(ptr) => ptr.ptr(),
            Pixels::Pointer16(ptr) => ptr.ptr() as *const u8,
            Pixels::Buffer(buffer) => buffer.as_ptr(),
            Pixels::Buffer16(buffer) => buffer.as_ptr() as *const u8,
        }
    }

    pub fn ptr_mut_generic(&mut self) -> *mut u8 {
        match self {
            Pixels::Pointer(ptr) => ptr.ptr_mut(),
            Pixels::Pointer16(ptr) => ptr.ptr_mut() as *mut u8,
            Pixels::Buffer(buffer) => buffer.as_mut_ptr(),
            Pixels::Buffer16(buffer) => buffer.as_mut_ptr() as *mut u8,
        }
    }

    pub fn slice(&self, offset: u32, size: u32) -> AvifResult<&[u8]> {
        let offset: usize = usize_from_u32(offset)?;
        let size: usize = usize_from_u32(size)?;
        match self {
            Pixels::Pointer(ptr) => {
                let end = offset.checked_add(size).ok_or(AvifError::NoContent)?;
                ptr.slice(offset..end)
            }
            Pixels::Pointer16(_) => AvifError::no_content(),
            Pixels::Buffer(buffer) => {
                let end = offset.checked_add(size).ok_or(AvifError::NoContent)?;
                let range = offset..end;
                check_slice_range(buffer.len(), &range)?;
                Ok(&buffer[range])
            }
            Pixels::Buffer16(_) => AvifError::no_content(),
        }
    }

    pub(crate) fn slice_mut(&mut self, offset: u32, size: u32) -> AvifResult<&mut [u8]> {
        let offset: usize = usize_from_u32(offset)?;
        let size: usize = usize_from_u32(size)?;
        match self {
            Pixels::Pointer(ptr) => {
                let end = offset.checked_add(size).ok_or(AvifError::NoContent)?;
                ptr.slice_mut(offset..end)
            }
            Pixels::Pointer16(_) => AvifError::no_content(),
            Pixels::Buffer(buffer) => {
                let end = offset.checked_add(size).ok_or(AvifError::NoContent)?;
                let range = offset..end;
                check_slice_range(buffer.len(), &range)?;
                Ok(&mut buffer[range])
            }
            Pixels::Buffer16(_) => AvifError::no_content(),
        }
    }

    pub fn slice16(&self, offset: u32, size: u32) -> AvifResult<&[u16]> {
        let offset: usize = usize_from_u32(offset)?;
        let size: usize = usize_from_u32(size)?;
        match self {
            Pixels::Pointer(_) => AvifError::no_content(),
            Pixels::Pointer16(ptr) => {
                let end = offset.checked_add(size).ok_or(AvifError::NoContent)?;
                ptr.slice(offset..end)
            }
            Pixels::Buffer(_) => AvifError::no_content(),
            Pixels::Buffer16(buffer) => {
                let end = offset.checked_add(size).ok_or(AvifError::NoContent)?;
                let range = offset..end;
                check_slice_range(buffer.len(), &range)?;
                Ok(&buffer[range])
            }
        }
    }

    pub(crate) fn slice16_mut(&mut self, offset: u32, size: u32) -> AvifResult<&mut [u16]> {
        let offset: usize = usize_from_u32(offset)?;
        let size: usize = usize_from_u32(size)?;
        match self {
            Pixels::Pointer(_) => AvifError::no_content(),
            Pixels::Pointer16(ptr) => {
                let end = offset.checked_add(size).ok_or(AvifError::NoContent)?;
                ptr.slice_mut(offset..end)
            }
            Pixels::Buffer(_) => AvifError::no_content(),
            Pixels::Buffer16(buffer) => {
                let end = offset.checked_add(size).ok_or(AvifError::NoContent)?;
                let range = offset..end;
                check_slice_range(buffer.len(), &range)?;
                Ok(&mut buffer[range])
            }
        }
    }
}

// From PixelInformationProperty semantics in ISO/IEC 23008-12/DAM 2.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum ChannelIdc {
    Unused = 0,
    Unspecified = 1,
    FirstColorChannel = 2,  // e.g. monochrome, Y, R, C
    SecondColorChannel = 3, // e.g. U, Cb, G, M
    ThirdColorChannel = 4,  // e.g. V, Cr, B, Y
    Alpha = 5,
    Depth = 6,
    FourthColorChannel = 7, // e.g. K
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixels_from_raw_pointer_validation() {
        let mut buf16 = [1u16, 2, 3, 4];
        let raw_ptr = buf16.as_mut_ptr() as *mut u8;

        // Odd row_bytes for >8-bit depth is rejected so row strides stay u16-aligned.
        assert!(Pixels::from_raw_pointer(raw_ptr, 10, 2, 3).is_err());
        // Unaligned pointer for >8-bit depth is rejected.
        assert!(Pixels::from_raw_pointer(raw_ptr.wrapping_add(1), 10, 1, 4).is_err());

        let mut pixels16 = Pixels::from_raw_pointer(raw_ptr, 10, 2, 4).unwrap();
        assert!(pixels16.has_data());
        assert!(pixels16.ptr().is_null());
        assert!(!pixels16.ptr16().is_null());
        assert_eq!(pixels16.slice16(0, 4).unwrap(), &[1, 2, 3, 4]);
        pixels16
            .slice16_mut(1, 2)
            .unwrap()
            .copy_from_slice(&[20, 30]);
        assert_eq!(pixels16.slice16(0, 4).unwrap(), &[1, 20, 30, 4]);

        let pixels8 = Pixels::from_raw_pointer_and_size(raw_ptr, 8, 4).unwrap();
        assert!(pixels8.ptr16().is_null());
        assert!(!pixels8.ptr().is_null());
        assert_eq!(pixels8.slice(0, 4).unwrap().len(), 4);
    }
}
