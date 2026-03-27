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

#![cfg(feature = "png")]

use crabby_avif::image::*;
use crabby_avif::*;

mod utils;
use utils::*;

use tempfile::NamedTempFile;
use test_case::test_matrix;

#[test_matrix(
    [8, 16],
    [false, true]
)]
fn roundtrip(depth: u8, alpha: bool) -> AvifResult<()> {
    let image = generate_gradient_image(1, 1, depth, PixelFormat::Yuv444, YuvRange::Full, alpha)?;
    let path = NamedTempFile::new().unwrap().into_temp_path();
    let path = format!("{}.png", path.to_str().unwrap());
    write_png(&image, &path)?;
    let decoded = read_image(&path)?;
    are_images_equal(&image, &decoded)?;
    Ok(())
}
