use crate::AvifError;
use crate::AvifResult;
use std::fs::File;
use std::os::unix::fs::FileExt; // TODO: what happens when this is compiled for windows?

pub trait DecoderIO {
    fn read(&mut self, offset: u64, size: usize) -> AvifResult<&[u8]>;
    fn size_hint(&self) -> u64;
    fn persistent(&self) -> bool;
}

#[derive(Default, Debug)]
pub struct DecoderFileIO {
    file: Option<File>,
    buffer: Vec<u8>,
}

impl DecoderFileIO {
    pub fn create(filename: &String) -> AvifResult<DecoderFileIO> {
        let file = File::open(filename).or(Err(AvifError::IoError))?;
        Ok(DecoderFileIO {
            file: Some(file),
            buffer: Vec::new(),
        })
    }
}

impl DecoderIO for DecoderFileIO {
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
            if self
                .file
                .as_ref()
                .unwrap()
                .read_exact_at(self.buffer.as_mut_slice(), offset)
                .is_err()
            {
                return Err(AvifError::IoError);
            }
        } else {
            self.buffer.clear();
        }
        Ok(self.buffer.as_slice())
    }

    fn size_hint(&self) -> u64 {
        let metadata = self.file.as_ref().unwrap().metadata();
        if metadata.is_err() {
            return 0;
        }
        metadata.unwrap().len()
    }

    fn persistent(&self) -> bool {
        false
    }
}
