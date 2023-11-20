use crate::AvifError;
use crate::AvifResult;
use std::fs::File;
use std::io::prelude::*;
use std::os::unix::fs::FileExt; // TODO: what happens when this is compiled for windows?

pub struct AvifIOData {
    data: *const u8,
    size: usize,
}

impl AvifIOData {
    pub fn empty() -> AvifIOData {
        AvifIOData {
            data: std::ptr::null(),
            size: 0,
        }
    }
}

pub trait AvifDecoderIO {
    fn read(&mut self, offset: u64, size: usize) -> AvifResult<&[u8]>;
    fn size_hint(&self) -> u64;
    fn persistent(&self) -> bool;
}

#[derive(Default, Debug)]
pub struct AvifDecoderFileIO {
    file: Option<File>,
    buffer: Vec<u8>,
}

impl AvifDecoderFileIO {
    pub fn create(filename: &String) -> AvifResult<AvifDecoderFileIO> {
        let file = File::open(filename).or(Err(AvifError::IoError))?;
        Ok(AvifDecoderFileIO {
            file: Some(file),
            buffer: Vec::new(),
        })
    }
}

impl AvifDecoderIO for AvifDecoderFileIO {
    fn read(&mut self, offset: u64, size: usize) -> AvifResult<&[u8]> {
        let file_size = self.size_hint();
        if offset > file_size {
            return Err(AvifError::IoError);
        }
        let available_size: usize = (file_size - offset) as usize;
        let size_to_read: usize = if size > available_size {
            available_size
        } else {
            size
        };
        if size_to_read > 0 {
            if self.buffer.capacity() < size_to_read {
                self.buffer.reserve(size_to_read);
            }
            self.buffer.resize(size_to_read, 0);
            if let Err(_) = self
                .file
                .as_ref()
                .unwrap()
                .read_exact_at(self.buffer.as_mut_slice(), offset)
            {
                return Err(AvifError::IoError);
            }
        } else {
            self.buffer.resize(0, 0);
        }
        Ok(self.buffer.as_slice())
    }

    fn size_hint(&self) -> u64 {
        let metadata = self.file.as_ref().unwrap().metadata();
        if !metadata.is_ok() {
            return 0;
        }
        metadata.unwrap().len()
    }

    fn persistent(&self) -> bool {
        false
    }
}
