use super::types::*;

use std::os::raw::c_void;

use crate::*;

#[repr(C)]
pub struct avifROData {
    pub data: *const u8,
    pub size: usize,
}

impl Default for avifROData {
    fn default() -> Self {
        avifROData {
            data: std::ptr::null(),
            size: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct avifRWData {
    data: *mut u8,
    size: usize,
}

impl Default for avifRWData {
    fn default() -> Self {
        avifRWData {
            data: std::ptr::null_mut(),
            size: 0,
        }
    }
}

impl From<&Vec<u8>> for avifRWData {
    fn from(v: &Vec<u8>) -> Self {
        avifRWData {
            data: v.as_ptr() as *mut u8,
            size: v.len(),
        }
    }
}

pub type avifIODestroyFunc = fn(io: *mut avifIO);
pub type avifIOReadFunc = fn(
    io: *mut avifIO,
    readFlags: u32,
    offset: u64,
    size: usize,
    out: *mut avifROData,
) -> avifResult;
pub type avifIOWriteFunc =
    fn(io: *mut avifIO, writeFlags: u32, offset: u64, data: *const u8, size: usize) -> avifResult;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct avifIO {
    destroy: avifIODestroyFunc,
    read: avifIOReadFunc,
    write: avifIOWriteFunc,
    sizeHint: u64,
    persistent: avifBool,
    data: *mut c_void,
}

pub struct avifIOWrapper {
    data: avifROData,
    io: avifIO,
}

impl avifIOWrapper {
    pub fn create(io: avifIO) -> Self {
        Self {
            io,
            data: Default::default(),
        }
    }
}

impl crate::decoder::IO for avifIOWrapper {
    fn read(&mut self, offset: u64, size: usize) -> AvifResult<&[u8]> {
        let res = (self.io.read)(
            &mut self.io as *mut avifIO,
            0,
            offset,
            size,
            &mut self.data as *mut avifROData,
        );
        if res != avifResult::Ok {
            // TODO: Some other return values may be allowed?
            return Err(AvifError::IoError);
        }
        Ok(unsafe { std::slice::from_raw_parts(self.data.data, self.data.size) })
    }
    fn size_hint(&self) -> u64 {
        self.io.sizeHint
    }
    fn persistent(&self) -> bool {
        self.io.persistent != 0
    }
}
