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

#![cfg(feature = "encoder")]

use crabby_avif::decoder::CompressionFormat;
use crabby_avif::decoder::ProgressiveState;
use crabby_avif::image::*;
use crabby_avif::*;

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

#[path = "./mod.rs"]
mod tests;

use tests::*;

fn generate_random_image(
    width: u32,
    height: u32,
    depth: u8,
    yuv_format: PixelFormat,
    yuv_range: YuvRange,
    alpha: bool,
) -> AvifResult<Image> {
    let mut image = image::Image {
        width,
        height,
        depth,
        yuv_format,
        yuv_range,
        ..Default::default()
    };
    image.allocate_planes(Category::Color)?;
    if alpha {
        image.allocate_planes(Category::Alpha)?;
    }
    let mut rng: StdRng = SeedableRng::seed_from_u64(0xABCDEF);
    for plane in ALL_PLANES {
        if !image.has_plane(plane) {
            continue;
        }
        let plane_data = image.plane_data(plane).unwrap();
        for y in 0..plane_data.height {
            if image.depth == 8 {
                let row = image.row_mut(plane, y)?;
                let row_slice = &mut row[..plane_data.width as usize];
                for pixel in row_slice {
                    *pixel = rng.gen_range(0..=255);
                }
            } else {
                let max_channel = image.max_channel();
                let row = image.row16_mut(plane, y)?;
                let row_slice = &mut row[..plane_data.width as usize];
                for pixel in row_slice {
                    *pixel = rng.gen_range(0..=max_channel);
                }
            }
        }
    }
    if rng.gen() {
        image.pasp = Some(PixelAspectRatio {
            h_spacing: rng.gen(),
            v_spacing: rng.gen(),
        });
    }
    if rng.gen() {
        image.clli = Some(ContentLightLevelInformation {
            max_cll: rng.gen(),
            max_pall: rng.gen(),
        });
    }
    Ok(image)
}

#[test_case::test_matrix(
    [100, 121],
    [200, 107],
    [8, 10, 12],
    [PixelFormat::Yuv420, PixelFormat::Yuv422, PixelFormat::Yuv444, PixelFormat::Yuv400],
    [YuvRange::Limited, YuvRange::Full],
    [false, true]
)]
fn encode_decode(
    width: u32,
    height: u32,
    depth: u8,
    yuv_format: PixelFormat,
    yuv_range: YuvRange,
    alpha: bool,
) -> AvifResult<()> {
    if !HAS_ENCODER {
        return Ok(());
    }
    let input_image = generate_random_image(width, height, depth, yuv_format, yuv_range, alpha)?;
    let settings = encoder::Settings {
        speed: Some(10),
        mutable: encoder::MutableSettings {
            quality: 90,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    encoder.add_image(&input_image)?;
    let edata = encoder.finish()?;
    assert!(!edata.is_empty());

    let mut decoder = decoder::Decoder::default();
    decoder.set_io_vec(edata);
    assert!(decoder.parse().is_ok());
    assert_eq!(decoder.compression_format(), CompressionFormat::Avif);
    assert_eq!(decoder.image_count(), 1);

    let image = decoder.image().expect("image was none");
    assert_eq!(image.alpha_present, alpha);
    assert!(!image.image_sequence_track_present);
    assert_eq!(image.width, width);
    assert_eq!(image.height, height);
    assert_eq!(image.depth, depth);
    assert_eq!(image.yuv_format, yuv_format);
    assert_eq!(image.yuv_range, yuv_range);
    assert_eq!(image.pasp, input_image.pasp);
    assert_eq!(image.clli, input_image.clli);
    // TODO: test for other properties.

    if !HAS_DECODER {
        return Ok(());
    }
    assert!(decoder.next_image().is_ok());
    Ok(())
}

#[test_case::test_matrix(
    [100, 121],
    [200, 107],
    [8, 10, 12],
    [PixelFormat::Yuv420, PixelFormat::Yuv422, PixelFormat::Yuv444, PixelFormat::Yuv400],
    [YuvRange::Limited, YuvRange::Full],
    [false, true]
)]
fn encode_decode_sequence(
    width: u32,
    height: u32,
    depth: u8,
    yuv_format: PixelFormat,
    yuv_range: YuvRange,
    alpha: bool,
) -> AvifResult<()> {
    if !HAS_ENCODER {
        return Ok(());
    }
    let mut input_images = Vec::new();
    let frame_count = 10;
    for _ in 0..frame_count {
        input_images.push(generate_random_image(
            width, height, depth, yuv_format, yuv_range, alpha,
        )?);
    }
    let images: Vec<&Image> = input_images.iter().collect();
    let settings = encoder::Settings {
        speed: Some(6),
        timescale: 10000,
        mutable: encoder::MutableSettings {
            quality: 50,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    for image in images {
        encoder.add_image_for_sequence(image, 1000)?;
    }
    let edata = encoder.finish()?;
    assert!(!edata.is_empty());

    let mut decoder = decoder::Decoder::default();
    decoder.set_io_vec(edata);
    assert!(decoder.parse().is_ok());
    assert_eq!(decoder.compression_format(), CompressionFormat::Avif);
    assert_eq!(decoder.image_count(), 10);

    let image = decoder.image().expect("image was none");
    assert_eq!(image.alpha_present, alpha);
    assert!(image.image_sequence_track_present);
    assert_eq!(image.width, width);
    assert_eq!(image.height, height);
    assert_eq!(image.depth, depth);
    assert_eq!(image.yuv_format, yuv_format);
    assert_eq!(image.yuv_range, yuv_range);

    if !HAS_DECODER {
        return Ok(());
    }
    for _ in 0..frame_count {
        assert!(decoder.next_image().is_ok());
    }
    Ok(())
}

fn test_progressive_decode(
    edata: Vec<u8>,
    width: u32,
    height: u32,
    extra_layer_count: u32,
) -> AvifResult<()> {
    let mut decoder = decoder::Decoder::default();
    decoder.settings.allow_progressive = true;
    decoder.set_io_vec(edata);
    assert!(decoder.parse().is_ok());
    let image = decoder.image().expect("image was none");
    assert!(matches!(image.progressive_state, ProgressiveState::Active));
    assert_eq!(decoder.image_count(), extra_layer_count + 1);
    assert_eq!(image.width, width);
    assert_eq!(image.height, height);
    if !HAS_DECODER {
        return Ok(());
    }
    for _ in 0..extra_layer_count + 1 {
        let res = decoder.next_image();
        assert!(res.is_ok());
        let image = decoder.image().expect("image was none");
        assert_eq!(image.width, width);
        assert_eq!(image.height, height);
    }
    Ok(())
}

#[test_case::test_matrix([true, false])]
fn progressive_quality_change(use_grid: bool) -> AvifResult<()> {
    if !HAS_ENCODER {
        return Ok(());
    }
    let image = generate_random_image(256, 256, 8, PixelFormat::Yuv444, YuvRange::Full, false)?;
    let mut settings = encoder::Settings {
        speed: Some(10),
        extra_layer_count: 1,
        mutable: encoder::MutableSettings {
            quality: 2,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    let images = [&image, &image];
    if use_grid {
        encoder.add_image_grid(2, 1, &images)?;
    } else {
        encoder.add_image(&image)?;
    }
    settings.mutable.quality = 90;
    encoder.update_settings(&settings.mutable)?;
    if use_grid {
        encoder.add_image_grid(2, 1, &images)?;
    } else {
        encoder.add_image(&image)?;
    }
    let edata = encoder.finish()?;
    assert!(!edata.is_empty());
    test_progressive_decode(
        edata,
        if use_grid { 512 } else { 256 },
        256,
        settings.extra_layer_count,
    )?;
    Ok(())
}

#[test]
fn progressive_same_layers() -> AvifResult<()> {
    if !HAS_ENCODER {
        return Ok(());
    }
    let image = generate_random_image(256, 256, 8, PixelFormat::Yuv444, YuvRange::Full, false)?;
    let settings = encoder::Settings {
        extra_layer_count: 3,
        speed: Some(10),
        mutable: encoder::MutableSettings {
            quality: 50,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    for _ in 0..4 {
        encoder.add_image(&image)?;
    }
    let edata = encoder.finish()?;
    assert!(!edata.is_empty());
    test_progressive_decode(edata, 256, 256, settings.extra_layer_count)?;
    Ok(())
}

#[test]
fn progressive_incorrect_number_of_layers() -> AvifResult<()> {
    if !HAS_ENCODER {
        return Ok(());
    }
    let image = generate_random_image(256, 256, 8, PixelFormat::Yuv444, YuvRange::Full, false)?;
    let settings = encoder::Settings {
        speed: Some(10),
        extra_layer_count: 1,
        mutable: encoder::MutableSettings {
            quality: 50,
            ..Default::default()
        },
        ..Default::default()
    };

    // Too many layers.
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    assert!(encoder.add_image(&image).is_ok());
    assert!(encoder.add_image(&image).is_ok());
    assert!(encoder.add_image(&image).is_err());

    // Too few layers.
    encoder = encoder::Encoder::create_with_settings(&settings)?;
    assert!(encoder.add_image(&image).is_ok());
    assert!(encoder.finish().is_err());
    Ok(())
}
