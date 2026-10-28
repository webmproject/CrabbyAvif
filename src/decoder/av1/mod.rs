// Copyright 2026 Google LLC
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

//! av1: an AV1 keyframe-only decoder for still images (AVIF).
//!
//! This is currently only a skeleton: the public API is in place but
//! decoding is not implemented yet and every entry point returns
//! [`Error::Unsupported`]. The decoder will be filled in by follow-up
//! changes.

// The decoder is 100% safe Rust: unsafe code is forbidden in this module
// and cannot be re-enabled locally with `#[allow(unsafe_code)]`.
#![forbid(unsafe_code)]

mod error;

pub use error::{Error, Result};

/// Chroma subsampling of the decoded frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromaLayout {
    Mono,
    Yuv420,
    Yuv422,
    Yuv444,
}

/// Options for [`decode`].
#[derive(Clone, Debug, Default)]
pub struct DecodeOptions {}

/// Basic information about the coded image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageInfo {
    /// Frame width (after super-resolution upscaling) and height.
    pub width: usize,
    pub height: usize,
    pub bit_depth: u32,
    pub layout: ChromaLayout,
    pub color_range_full: bool,
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
}

/// A decoded frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub bit_depth: u32,
    pub layout: ChromaLayout,
}

/// Parses the sequence header of the OBU stream in `data` and returns the
/// basic properties of the coded image without decoding it.
pub fn probe(_data: &[u8]) -> Result<ImageInfo> {
    Err(Error::Unsupported("probe is not implemented"))
}

/// Decodes the first frame (which must be a keyframe) of the OBU stream in
/// `data`.
pub fn decode(_data: &[u8], _opts: &DecodeOptions) -> Result<Frame> {
    Err(Error::Unsupported("decode is not implemented"))
}
