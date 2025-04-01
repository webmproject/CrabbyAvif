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

// Not all functions are used from all test targets. So allow unused functions in this module.
#![allow(unused)]

use crabby_avif::image::Image;
use crabby_avif::*;
use std::fs::File;

#[cfg(test)]
pub fn get_test_file(filename: &str) -> String {
    let base_path = if cfg!(google3) {
        format!(
            "{}/google3/third_party/crabbyavif/",
            std::env::var("TEST_SRCDIR").expect("TEST_SRCDIR is not defined")
        )
    } else {
        "".to_string()
    };
    format!("{base_path}tests/data/{filename}")
}

#[cfg(test)]
pub fn get_decoder(filename: &str) -> decoder::Decoder {
    let abs_filename = get_test_file(filename);
    let mut decoder = decoder::Decoder::default();
    decoder
        .set_io_file(&abs_filename)
        .expect("Failed to set IO");
    decoder
}

#[cfg(test)]
pub fn decode_png(filename: &str) -> Vec<u8> {
    let decoder = png::Decoder::new(File::open(get_test_file(filename)).unwrap());
    let mut reader = decoder.read_info().unwrap();
    // Indexed colors are not supported.
    assert_ne!(reader.output_color_type().0, png::ColorType::Indexed);
    let mut pixels = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut pixels).unwrap();
    pixels
}

#[cfg(test)]
fn squared_diff_sum(pixel1: u16, pixel2: u16) -> u64 {
    let diff = pixel1 as i32 - pixel2 as i32;
    (diff * diff) as u64
}

#[cfg(test)]
pub fn psnr(image1: &Image, image2: &Image) -> AvifResult<f64> {
    assert!(image1.has_same_properties_and_cicp(image2));
    let mut diff_sum = 0u64;
    let mut num_samples = 0;
    for plane in image::ALL_PLANES {
        assert_eq!(image1.has_plane(plane), image2.has_plane(plane));
        if !image1.has_plane(plane) {
            continue;
        }
        let width = image1.width(plane);
        let height = image1.height(plane);
        if width == 0 || height == 0 {
            continue;
        }
        for y in 0..height as u32 {
            if image1.depth > 8 {
                let row1 = image1.row16(plane, y)?;
                let row2 = image2.row16(plane, y)?;
                for x in 0..width {
                    diff_sum += squared_diff_sum(row1[x], row2[x]);
                }
            } else {
                let row1 = image1.row(plane, y)?;
                let row2 = image2.row(plane, y)?;
                for x in 0..width {
                    diff_sum += squared_diff_sum(row1[x] as u16, row2[x] as u16);
                }
            }
            num_samples += width;
        }
    }
    if diff_sum == 0 {
        return Ok(99.0);
    }
    let max_channel_f = image1.max_channel() as f64;
    let normalized_error = diff_sum as f64 / (num_samples as f64 * max_channel_f * max_channel_f);
    if normalized_error <= f64::EPSILON {
        Ok(98.99)
    } else {
        Ok((-10.0 * normalized_error.log10()).min(98.99))
    }
}

#[cfg(test)]
#[allow(dead_code)]
pub const HAS_DECODER: bool = cfg!(any(
    feature = "dav1d",
    feature = "libgav1",
    feature = "android_mediacodec"
));

#[cfg(test)]
#[allow(dead_code)]
pub const HAS_ENCODER: bool = cfg!(feature = "aom");
