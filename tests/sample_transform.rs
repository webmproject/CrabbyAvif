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

#![allow(clippy::disallowed_methods, clippy::disallowed_macros)]
#![cfg(feature = "png")]
#![cfg(feature = "encoder")]

mod utils;
use crabby_avif::encoder::Recipe;
use crabby_avif::image::*;
use crabby_avif::utils::reader::png::PngReader;
use crabby_avif::utils::reader::Config;
use crabby_avif::utils::reader::Reader;
use crabby_avif::*;
use test_case::test_matrix;
use utils::*;

#[test_matrix(
    [Recipe::BitDepthExtension8b8b, Recipe::BitDepthExtension12b4b, Recipe::BitDepthExtension12b8bOverlap4b]
)]
fn lossless_sample_transform_roundtrip(recipe: Recipe) -> AvifResult<()> {
    if !HAS_ENCODER {
        return Ok(());
    }
    if recipe == Recipe::BitDepthExtension12b8bOverlap4b && !HAS_DECODER {
        // This recipe decodes the primary image item at encoding.
        return Ok(());
    }
    let input_file = get_test_file("weld_16bit.png");
    let (image, _, _) = PngReader::create(&input_file)?.read_frame(&Config {
        yuv_format: Some(PixelFormat::Yuv444),
        matrix_coefficients: Some(MatrixCoefficients::Identity),
        allow_sample_transform: true,
        ..Default::default()
    })?;
    assert_eq!(image.depth, 16);

    let settings = encoder::Settings {
        speed: Some(10),
        mutable: encoder::MutableSettings {
            quality: 100.0,
            ..Default::default()
        },
        recipe,
        ..Default::default()
    };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    encoder.add_image(&image)?;
    let edata = encoder.finish()?;
    assert!(!edata.is_empty());

    if !HAS_DECODER {
        return Ok(());
    }

    let mut decoder = decoder::Decoder::default();
    decoder.set_io_vec(edata.clone());
    decoder.settings.allow_sample_transform = true;
    assert!(decoder.parse().is_ok());
    assert!(decoder.next_image().is_ok());
    let decoded_image = decoder.image().unwrap();
    assert!(are_images_equal(&image, decoded_image)?);
    Ok(())
}

#[test_matrix(
    [Recipe::BitDepthExtension8b8b, Recipe::BitDepthExtension12b4b, Recipe::BitDepthExtension12b8bOverlap4b],
    [false, true]
)]
fn lossless_sample_transform_grid_roundtrip(recipe: Recipe, alpha: bool) -> AvifResult<()> {
    if !HAS_ENCODER {
        return Ok(());
    }
    if recipe == Recipe::BitDepthExtension12b8bOverlap4b && !HAS_DECODER {
        // This recipe decodes the primary image item at encoding.
        return Ok(());
    }
    let cell0 = generate_gradient_image(64, 64, 16, PixelFormat::Yuv444, YuvRange::Full, alpha)?;
    let mut cell1 = cell0.try_deep_clone()?;
    // Make the cells different.
    for plane in ALL_PLANES {
        if !cell1.has_plane(plane) {
            continue;
        }
        for y in 0..cell1.height(plane) as u32 {
            for sample in cell1.row16_mut(plane, y)? {
                *sample = u16::MAX - *sample;
            }
        }
    }
    let cells = [&cell0, &cell1];

    let settings = encoder::Settings {
        speed: Some(10),
        mutable: encoder::MutableSettings {
            quality: 100.0,
            quality_alpha: 100.0,
            ..Default::default()
        },
        recipe,
        ..Default::default()
    };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    // The 'sato' derived image item has two 'grid' input image items.
    encoder.add_image_grid(/*columns=*/ 2, /*rows=*/ 1, &cells)?;
    let edata = encoder.finish()?;

    let mut decoder = decoder::Decoder::default();
    decoder.set_io_vec(edata);
    decoder.settings.allow_sample_transform = true;
    assert_eq!(decoder.parse(), Ok(()));
    let image = decoder.image().unwrap();
    assert_eq!(image.width, 128);
    assert_eq!(image.height, 64);
    assert_eq!(image.depth, 16);
    assert_eq!(image.alpha_present, alpha);

    if !HAS_DECODER {
        return Ok(());
    }
    assert_eq!(decoder.next_image(), Ok(()));
    let expected_image =
        merge_cells_into_grid_image(/*columns=*/ 2, /*rows=*/ 1, &cells)?;
    assert!(are_images_equal(&expected_image, decoder.image().unwrap())?);
    Ok(())
}

#[test_matrix(
    [8, 10, 12, 16],
    [PixelFormat::Yuv420, PixelFormat::Yuv422, PixelFormat::Yuv444, PixelFormat::Yuv400],
    [YuvRange::Limited, YuvRange::Full],
    [false, true]
)]
fn recipe_auto(
    depth: u8,
    yuv_format: PixelFormat,
    yuv_range: YuvRange,
    alpha: bool,
) -> AvifResult<()> {
    if !HAS_ENCODER {
        return Ok(());
    }
    let input_image = generate_gradient_image(64, 64, depth, yuv_format, yuv_range, alpha)?;
    let settings = encoder::Settings {
        speed: Some(10),
        recipe: Recipe::Auto,
        ..Default::default()
    };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    encoder.add_image(&input_image)?;
    let edata = encoder.finish()?;

    let mut decoder = decoder::Decoder::default();
    decoder.set_io_vec(edata);
    decoder.settings.allow_sample_transform = true;
    assert_eq!(decoder.parse(), Ok(()));

    if !HAS_DECODER {
        return Ok(());
    }
    assert_eq!(decoder.next_image(), Ok(()));
    assert!(psnr(decoder.image().unwrap(), &input_image)? >= 30.0);
    Ok(())
}

#[test]
fn sample_transform_grid_mismatched_dimensions() -> AvifResult<()> {
    if !HAS_ENCODER || !HAS_DECODER {
        return Ok(());
    }
    let cell = generate_gradient_image(64, 66, 16, PixelFormat::Yuv444, YuvRange::Full, false)?;
    let cells = [&cell, &cell];
    let settings = encoder::Settings {
        speed: Some(10),
        mutable: encoder::MutableSettings {
            quality: 100.0,
            ..Default::default()
        },
        recipe: Recipe::BitDepthExtension8b8b,
        ..Default::default()
    };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    encoder.add_image_grid(/*columns=*/ 2, /*rows=*/ 1, &cells)?;
    let edata = encoder.finish()?;

    // Patch the deduplicated 'grid' payload's output_width from 128 to 100 while leaving the
    // 'ispe' properties at 128x66 (100 still overlaps both 64x66 cells so grid tile validation
    // succeeds).
    let valid_grid_payload: [u8; 8] = [0, 0, 0, 1, 0, 128, 0, 66];
    let offsets: Vec<usize> = edata
        .windows(valid_grid_payload.len())
        .enumerate()
        .filter_map(|(i, window)| (window == valid_grid_payload).then_some(i))
        .collect();
    assert_eq!(offsets.len(), 1);
    for offset in offsets {
        let mut patched = edata.clone();
        patched[offset + 5] = 100;
        let mut decoder = decoder::Decoder::default();
        decoder.set_io_vec(patched);
        decoder.settings.allow_sample_transform = true;
        assert_eq!(decoder.parse(), Ok(()));
        assert!(matches!(
            decoder.next_image(),
            Err(AvifError::BmffParseFailed(_))
        ));
    }
    Ok(())
}

#[test]
fn sample_transform_grid_per_tile_alpha() -> AvifResult<()> {
    if !HAS_ENCODER || !HAS_DECODER {
        return Ok(());
    }
    let cell0 = generate_gradient_image(64, 64, 16, PixelFormat::Yuv444, YuvRange::Full, true)?;
    let mut cell1 = cell0.try_deep_clone()?;
    // Keep the alpha plane identical so a single alpha tile can be shared via 'auxl' across both
    // color cells, while making the color planes different.
    for plane in YUV_PLANES {
        for y in 0..cell1.height(plane) as u32 {
            for sample in cell1.row16_mut(plane, y)? {
                *sample = u16::MAX - *sample;
            }
        }
    }
    let cells = [&cell0, &cell1];
    let settings = encoder::Settings {
        speed: Some(10),
        mutable: encoder::MutableSettings {
            quality: 100.0,
            quality_alpha: 100.0,
            ..Default::default()
        },
        recipe: Recipe::BitDepthExtension8b8b,
        ..Default::default()
    };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    encoder.add_image_grid(/*columns=*/ 2, /*rows=*/ 1, &cells)?;
    let mut edata = encoder.finish()?;

    // Replace the top-level alpha 'grid' items (4 for MSB grid 1, 11 for LSB grid 8) in 'iref'
    // with per-tile 'auxl' references directly on the color cells ([2, 3] and [9, 10]), exercising
    // synthesized ('is_made_up') alpha grids for 'sato' inputs.
    for (alpha_grid_id, color_grid_id, alpha_cell_id, color_cell0_id, color_cell1_id) in
        [(4u8, 1u8, 5u8, 2u8, 3u8), (11u8, 8u8, 12u8, 9u8, 10u8)]
    {
        let old_dimg = [
            0,
            0,
            0,
            16,
            b'd',
            b'i',
            b'm',
            b'g',
            0,
            alpha_grid_id,
            0,
            2,
            0,
            alpha_cell_id,
            0,
            alpha_cell_id + 1,
        ];
        let new_auxl = [
            0,
            0,
            0,
            16,
            b'a',
            b'u',
            b'x',
            b'l',
            0,
            alpha_cell_id,
            0,
            2,
            0,
            color_cell0_id,
            0,
            color_cell1_id,
        ];
        let dimg_pos = edata
            .windows(old_dimg.len())
            .position(|w| w == old_dimg)
            .unwrap();
        edata[dimg_pos..dimg_pos + old_dimg.len()].copy_from_slice(&new_auxl);

        let old_auxl = [
            0,
            0,
            0,
            14,
            b'a',
            b'u',
            b'x',
            b'l',
            0,
            alpha_grid_id,
            0,
            1,
            0,
            color_grid_id,
        ];
        let new_noop = [
            0,
            0,
            0,
            14,
            b'x',
            b'x',
            b'x',
            b'x',
            0,
            alpha_grid_id,
            0,
            1,
            0,
            color_grid_id,
        ];
        let auxl_pos = edata
            .windows(old_auxl.len())
            .position(|w| w == old_auxl)
            .unwrap();
        edata[auxl_pos..auxl_pos + old_auxl.len()].copy_from_slice(&new_noop);
    }

    let mut decoder = decoder::Decoder::default();
    decoder.set_io_vec(edata);
    decoder.settings.allow_sample_transform = true;
    assert_eq!(decoder.parse(), Ok(()));
    assert_eq!(decoder.next_image(), Ok(()));
    let expected_image =
        merge_cells_into_grid_image(/*columns=*/ 2, /*rows=*/ 1, &cells)?;
    assert!(are_images_equal(&expected_image, decoder.image().unwrap())?);
    Ok(())
}
