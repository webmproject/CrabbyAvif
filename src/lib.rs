use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;

mod bindings;
mod dav1d;
pub mod decoder;
pub mod io;
mod mp4box;
mod stream;
pub mod utils;

macro_rules! println {
    ($($rest:tt)*) => {
        #[cfg(debug_assertions)]
        std::println!($($rest)*)
    }
}

#[derive(Debug, Default, PartialEq, Copy, Clone)]
pub enum PixelFormat {
    #[default]
    None,
    Yuv444,
    Yuv422,
    Yuv420,
    Monochrome,
}

impl PixelFormat {
    pub fn plane_count(&self) -> usize {
        match self {
            PixelFormat::None => 0,
            PixelFormat::Monochrome => 1,
            PixelFormat::Yuv420 | PixelFormat::Yuv422 | PixelFormat::Yuv444 => 3,
        }
    }
}
