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

#![cfg(feature = "jpegxl")]

use crabby_avif::decoder::CompressionFormat;
use crabby_avif::image::*;
use crabby_avif::*;

mod utils;
use utils::*;

use test_case::test_matrix;

#[test_matrix(
    [100, 121],
    [200, 107],
    [8], // TODO: b/456440247 - Support 16-bit
    [false] // TODO: b/456440247 - Support alpha
)]
fn encode_decode(width: u32, height: u32, depth: u8, alpha: bool) -> AvifResult<()> {
    let image = generate_gradient_image(
        width,
        height,
        depth,
        PixelFormat::Yuv444,
        YuvRange::Full,
        alpha,
    )?;
    let encoded = {
        let settings = encoder::Settings {
            codec_choice: CodecChoice::Libjxl,
            speed: Some(1), // Fastest libjxl setting.
            mutable: encoder::MutableSettings {
                quality: 90.0,
                quality_alpha: 90.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
        encoder.add_image(&image)?;
        encoder.finish()?
    };
    assert!(!encoded.is_empty());

    let mut decoder = decoder::Decoder::default();
    // Explicitly selecting libjxl should not be necessary.
    decoder.settings.codec_choice = CodecChoice::Auto;
    decoder.set_io_vec(encoded);
    assert!(decoder.parse().is_ok());
    assert_eq!(decoder.compression_format(), CompressionFormat::JpegXl);
    assert_eq!(decoder.image_count(), 1);

    let decoded = decoder.image().unwrap();
    assert_eq!(decoded.alpha_present, image.alpha_present);
    assert_eq!(
        decoded.image_sequence_track_present,
        image.image_sequence_track_present
    );
    assert_eq!(decoded.width, image.width);
    assert_eq!(decoded.height, image.height);
    assert_eq!(decoded.depth, image.depth);
    assert_eq!(decoded.yuv_format, image.yuv_format);
    assert_eq!(decoded.yuv_range, image.yuv_range);

    assert!(decoder.next_image().is_ok());
    let image = decoder.image().unwrap();
    let psnr = psnr(image, &image)?;
    assert!(psnr >= 50.0);
    Ok(())
}
