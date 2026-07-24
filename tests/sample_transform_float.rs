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

#![cfg(feature = "encoder")]
#![cfg(feature = "satofloat")]

mod utils;
use crabby_avif::encoder::Recipe;
use crabby_avif::image::*;
use crabby_avif::*;
use test_case::test_matrix;
use utils::*;

// Encode losslessly and decode float samples. Verify they match.
#[test_matrix([false, true])]
fn lossless_sample_transform_roundtrip(allow_sample_transform: bool) -> AvifResult<()> {
    if !HAS_ENCODER {
        return Ok(());
    }

    let image_metadata = Image {
        width: 4,
        height: 4,
        depth: 8, // Of the primary image item. 'sato' item depth is 32.
        yuv_format: PixelFormat::Yuv444,
        yuv_range: YuvRange::Full,
        ..Default::default()
    };
    let num_pixels = image_metadata.width * image_metadata.height;
    let mut float_data = Vec::with_capacity(num_pixels as usize);
    // Easy-to-debug values.
    float_data.extend_from_slice(&[1., 2., 3., 4.]);
    // Notable values.
    float_data.extend_from_slice(&[f32::EPSILON, f32::MIN, f32::MIN_POSITIVE, f32::MAX]);
    float_data.extend_from_slice(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.]);
    // Notable values from
    // https://en.wikipedia.org/wiki/Single-precision_floating-point_format#Notable_single-precision_cases
    const SMALLEST_POSITIVE_SUBNORMAL: f32 = f32::from_bits(0b0_00000000_00000000000000000000001);
    const SMALLEST_NEGATIVE_SUBNORMAL: f32 = f32::from_bits(0b1_00000000_00000000000000000000001);
    const LARGEST_SUBNORMAL: f32 = f32::from_bits(0b0_00000000_11111111111111111111111);
    const LARGEST_NUMBER_LESS_THAN_ONE: f32 = f32::from_bits(0b0_01111110_11111111111111111111111);
    float_data.extend_from_slice(&[
        SMALLEST_POSITIVE_SUBNORMAL,
        SMALLEST_NEGATIVE_SUBNORMAL,
        LARGEST_SUBNORMAL,
        LARGEST_NUMBER_LESS_THAN_ONE,
    ]);

    let settings = encoder::Settings {
        speed: Some(10),
        mutable: encoder::MutableSettings {
            quality: 100.0,
            ..Default::default()
        },
        recipe: Recipe::Float32b,
        ..Default::default()
    };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    encoder.add_image_float(
        &image_metadata,
        [
            Some(float_data.as_slice()),
            Some(float_data.as_slice()),
            Some(float_data.as_slice()),
            None,
        ],
    )?;
    let encoded = encoder.finish()?;
    assert!(!encoded.is_empty());

    let mut decoder = decoder::Decoder::default();
    decoder.set_io_vec(encoded.clone());
    decoder.settings.allow_sample_transform = allow_sample_transform;
    decoder.parse()?;

    assert!(are_images_equal(
        &decoder.image().unwrap(),
        &Image {
            depth: if allow_sample_transform { 32 } else { 8 },
            planes: [const { None }; MAX_PLANE_COUNT],
            ..image_metadata
        }
    )?);

    if !HAS_DECODER {
        return Ok(());
    }

    decoder.next_image()?;
    if allow_sample_transform {
        // Sample Transform derived image item output.
        let decoded_float_samples = decoder.image_float()?;
        let expected_float_data = Vec::from_iter(float_data.iter().map(|&f| {
            if f.is_normal() || f.is_subnormal() {
                f
            } else {
                0.0
            }
        }));
        for plane in YUV_PLANES {
            let decoded_plane = decoded_float_samples[plane as usize].as_ref().unwrap();
            assert_eq!(decoded_plane.len(), expected_float_data.len());
            for i in 0..expected_float_data.len() {
                assert_eq!(decoded_plane[i], expected_float_data[i]);
            }
        }
    } else {
        // Primary image item output.
        let primary_image = decoder.image().unwrap();
        assert_eq!(primary_image.depth, 8);
        let expected_first_byte_of_float_data = Vec::from_iter(float_data.iter().map(|&f| {
            if f.is_normal() || f.is_subnormal() {
                (f.to_bits() >> 24) as u8
            } else {
                0
            }
        }));
        for plane in YUV_PLANES {
            for y in 0..primary_image.height {
                let row = primary_image.row(plane, y)?;
                for x in 0..primary_image.width as usize {
                    assert_eq!(
                        row[x],
                        expected_first_byte_of_float_data[(y * primary_image.width) as usize + x]
                    );
                }
            }
        }
        assert!(decoder.image_float().is_err());
    }

    Ok(())
}
