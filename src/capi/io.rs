use super::types::*;

use std::ffi::CStr;
use std::os::raw::c_char;
use std::os::raw::c_void;

use crate::decoder::GenericIO;
use crate::internal_utils::io::DecoderFileIO;
use crate::internal_utils::io::DecoderRawIO;
use crate::*;

#[derive(Clone)]
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

#[derive(Debug, Clone)]
#[repr(C)]
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

#[no_mangle]
pub unsafe extern "C" fn avifRWDataRealloc(raw: *mut avifRWData, newSize: usize) -> avifResult {
    unsafe {
        if (*raw).size == newSize {
            return avifResult::Ok;
        }
        // Ok to use size as capacity here since we use reserve_exact.
        let mut newData: Vec<u8> = Vec::new();
        newData.reserve_exact(newSize);
        if !(*raw).data.is_null() {
            let oldData = Box::from_raw(std::slice::from_raw_parts_mut((*raw).data, (*raw).size));
            let sizeToCopy = std::cmp::min(newSize, oldData.len());
            newData.extend_from_slice(&oldData[..sizeToCopy]);
        }
        newData.resize(newSize, 0);
        let mut b = newData.into_boxed_slice();
        (*raw).data = b.as_mut_ptr();
        std::mem::forget(b);
        (*raw).size = newSize;
        avifResult::Ok
    }
}

#[no_mangle]
pub unsafe extern "C" fn avifRWDataSet(
    raw: *mut avifRWData,
    data: *const u8,
    size: usize,
) -> avifResult {
    unsafe {
        if size != 0 {
            let res = avifRWDataRealloc(raw, size);
            if res != avifResult::Ok {
                return res;
            }
            std::ptr::copy_nonoverlapping(data, (*raw).data, size);
        } else {
            avifRWDataFree(raw);
        }
        avifResult::Ok
    }
}

#[no_mangle]
pub unsafe extern "C" fn avifRWDataFree(raw: *mut avifRWData) {
    unsafe {
        let _ = Box::from_raw(std::slice::from_raw_parts_mut((*raw).data, (*raw).size));
    }
}

pub type avifIODestroyFunc = unsafe extern "C" fn(io: *mut avifIO);
pub type avifIOReadFunc = unsafe extern "C" fn(
    io: *mut avifIO,
    readFlags: u32,
    offset: u64,
    size: usize,
    out: *mut avifROData,
) -> avifResult;
pub type avifIOWriteFunc = unsafe extern "C" fn(
    io: *mut avifIO,
    writeFlags: u32,
    offset: u64,
    data: *const u8,
    size: usize,
) -> avifResult;

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
        let res = unsafe {
            (self.io.read)(
                &mut self.io as *mut avifIO,
                0,
                offset,
                size,
                &mut self.data as *mut avifROData,
            )
        };
        if res != avifResult::Ok {
            let err: AvifError = res.into();
            //println!("returning {:#?} from IO::read", err);
            return Err(err);
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

pub struct avifCIOWrapper {
    io: GenericIO,
    buf: Vec<u8>,
}

#[no_mangle]
unsafe extern "C" fn cioDestroy(_io: *mut avifIO) {}

#[no_mangle]
unsafe extern "C" fn cioRead(
    io: *mut avifIO,
    _readFlags: u32,
    offset: u64,
    size: usize,
    out: *mut avifROData,
) -> avifResult {
    unsafe {
        if io.is_null() {
            return avifResult::IoError;
        }
        let cio = (*io).data as *mut avifCIOWrapper;
        match (*cio).io.read(offset, size) {
            Ok(data) => {
                (*cio).buf.clear();
                (*cio).buf.reserve(data.len());
                (*cio).buf.extend_from_slice(data);
            }
            Err(_) => return avifResult::IoError,
        }
        (*out).data = (*cio).buf.as_ptr();
        (*out).size = (*cio).buf.len();
        avifResult::Ok
    }
}

#[no_mangle]
unsafe extern "C" fn cioWrite(
    _io: *mut avifIO,
    _writeFlags: u32,
    _offset: u64,
    _data: *const u8,
    _size: usize,
) -> avifResult {
    avifResult::Ok
}

#[no_mangle]
pub unsafe extern "C" fn avifIOCreateMemoryReader(data: *const u8, size: usize) -> *mut avifIO {
    let cio = Box::new(avifCIOWrapper {
        io: Box::new(DecoderRawIO::create(data, size)),
        buf: Vec::new(),
    });
    let io = Box::new(avifIO {
        destroy: cioDestroy,
        read: cioRead,
        write: cioWrite,
        sizeHint: size as u64,
        persistent: 0,
        data: Box::into_raw(cio) as *mut c_void,
    });
    Box::into_raw(io)
}

#[no_mangle]
pub unsafe extern "C" fn avifIOCreateFileReader(filename: *const c_char) -> *mut avifIO {
    let filename = unsafe { String::from(CStr::from_ptr(filename).to_str().unwrap_or("")) };
    let file_io = match DecoderFileIO::create(&filename) {
        Ok(x) => x,
        Err(_) => return std::ptr::null_mut(),
    };
    let cio = Box::new(avifCIOWrapper {
        io: Box::new(file_io),
        buf: Vec::new(),
    });
    let io = Box::new(avifIO {
        destroy: cioDestroy,
        read: cioRead,
        write: cioWrite,
        sizeHint: cio.io.size_hint(),
        persistent: 0,
        data: Box::into_raw(cio) as *mut c_void,
    });
    Box::into_raw(io)
}

#[no_mangle]
pub unsafe extern "C" fn avifIODestroy(io: *mut avifIO) {
    unsafe {
        let _ = Box::from_raw((*io).data as *mut avifCIOWrapper);
        let _ = Box::from_raw(io);
    }
}
