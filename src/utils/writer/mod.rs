// Copyright 2025 Google LLC
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

// Not all sub-modules are used by all targets. Ignore dead code warnings.
#![allow(dead_code)]

#[cfg(feature = "jpeg")]
#[forbid(unsafe_code)]
pub mod jpeg;
// Unsafe is required for FFI calls and C callbacks into the `libpng` C library (`libpng-sys`).
#[cfg(feature = "png")]
#[allow(unsafe_code)]
pub mod png;
#[forbid(unsafe_code)]
pub mod y4m;

use crate::image::Image;
use crate::AvifResult;

use std::fs::File;

pub trait Writer {
    fn write_frame(&mut self, file: &mut File, image: &Image) -> AvifResult<()>;
}
