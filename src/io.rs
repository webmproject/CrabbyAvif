use std::fs::File;
use std::io::prelude::*;
// TODO: what happens when this is compiled for windows?
use std::os::unix::fs::FileExt;

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
    // TODO: replace result err type with something else.
    fn read(&mut self, offset: u64, size: usize) -> Result<&[u8], i32>;
    fn size_hint(&self) -> u64;
    fn persistent(&self) -> bool;
}

#[derive(Default, Debug)]
pub struct AvifDecoderFileIO {
    file: Option<File>,
    buffer: Vec<u8>,
}

impl AvifDecoderFileIO {
    pub fn create(filename: &String) -> Option<AvifDecoderFileIO> {
        let file = File::open(filename);
        if !file.is_ok() {
            return None;
        }
        Some(AvifDecoderFileIO {
            file: Some(file.unwrap()),
            buffer: Vec::new(),
        })
    }
}

impl AvifDecoderIO for AvifDecoderFileIO {
    fn read(&mut self, offset: u64, size: usize) -> Result<&[u8], i32> {
        let file_size = self.size_hint();
        if offset > file_size {
            return Err(-1);
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
            match self
                .file
                .as_ref()
                .unwrap()
                .read_exact_at(self.buffer.as_mut_slice(), offset)
            {
                Err(err) => return Err(-1),
                Ok(ok) => {}
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
