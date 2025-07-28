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
use crabby_avif::decoder::ImageContentType;
use crabby_avif::decoder::ProgressiveState;
use crabby_avif::encoder::*;
use crabby_avif::gainmap::*;
use crabby_avif::image::*;
use crabby_avif::utils::*;
use crabby_avif::*;

mod utils;
use utils::*;

use test_case::test_case;
use test_case::test_matrix;

#[test_matrix(
    [100, 121],
    [200, 107],
    [8, 10, 12],
    [PixelFormat::Yuv420, PixelFormat::Yuv422, PixelFormat::Yuv444, PixelFormat::Yuv400],
    [YuvRange::Limited, YuvRange::Full],
    [false, true],
    [TilingMode::Manual(0, 0), TilingMode::Manual(1, 0)]
)]
fn encode_decode(
    width: u32,
    height: u32,
    depth: u8,
    yuv_format: PixelFormat,
    yuv_range: YuvRange,
    alpha: bool,
    tiling_mode: TilingMode,
) -> AvifResult<()> {
    if !HAS_ENCODER {
        return Ok(());
    }
    let input_image = generate_gradient_image(width, height, depth, yuv_format, yuv_range, alpha)?;
    let settings = encoder::Settings {
        speed: Some(10),
        mutable: encoder::MutableSettings {
            quality: 90,
            tiling_mode,
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
    let image = decoder.image().expect("image was none");
    assert!(psnr(image, &input_image)? >= 50.0);
    Ok(())
}

fn encode_decode_grid_impl(
    cells_and_expect_success: (Vec<Vec<(u32, u32)>>, bool),
    yuv_format: PixelFormat,
    depth: u8,
) -> AvifResult<()> {
    if !HAS_ENCODER {
        return Ok(());
    }
    let cells = cells_and_expect_success.0;
    let expect_success = cells_and_expect_success.1;
    let mut cell_images = Vec::new();
    for cell_row in &cells {
        for cell_column in cell_row {
            cell_images.push(generate_gradient_image(
                cell_column.0,
                cell_column.1,
                depth,
                yuv_format,
                YuvRange::Full,
                /*alpha=*/ true,
            )?);
        }
    }
    let settings = encoder::Settings {
        speed: Some(10),
        mutable: encoder::MutableSettings {
            // Encode losslessly for easier comparison of outputs.
            quality: 100,
            quality_alpha: 100,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    let cell_image_refs: Vec<&Image> = cell_images.iter().collect();
    let columns = cells[0].len() as u32;
    let rows = cells.len() as u32;
    let res = encoder.add_image_grid(columns, rows, &cell_image_refs);
    if !expect_success {
        assert!(res.is_err());
        return Ok(());
    }
    assert!(res.is_ok());
    let edata = encoder.finish()?;
    assert!(!edata.is_empty());

    let reference_image = merge_cells_into_grid_image(columns, rows, &cell_image_refs)?;
    let mut decoder = decoder::Decoder::default();
    decoder.set_io_vec(edata);
    assert!(decoder.parse().is_ok());
    assert_eq!(decoder.compression_format(), CompressionFormat::Avif);
    assert_eq!(decoder.image_count(), 1);

    let image = decoder.image().expect("image was none");
    assert!(image.alpha_present);
    assert_eq!(image.width, reference_image.width);
    assert_eq!(image.height, reference_image.height);
    assert_eq!(image.depth, reference_image.depth);
    assert_eq!(image.yuv_format, reference_image.yuv_format);

    if !HAS_DECODER {
        return Ok(());
    }
    assert!(decoder.next_image().is_ok());
    let decoded_image = decoder.image().expect("image was none");
    assert!(are_images_equal(decoded_image, &reference_image)?);
    Ok(())
}

#[test_matrix(
    [
        // Single cell.
        (vec![vec![(1, 1)]], true),
        (vec![vec![(1, 64)]], true),
        (vec![vec![(64, 1)]], true),
        (vec![vec![(64, 64)]], true),
        (vec![vec![(127, 127)]], true),
        // Cells of same dimensions.
        (
            vec![
                vec![(64, 64), (64, 64), (64, 64)],
            ],
            true,
        ),
        (
            vec![
                vec![(100, 110)],
                vec![(100, 110)],
                vec![(100, 110)],
            ],
            true,
        ),
        (
            vec![
                vec![(64, 64), (64, 64), (64, 64)],
                vec![(64, 64), (64, 64), (64, 64)],
                vec![(64, 64), (64, 64), (64, 64)],
            ],
            true,
        ),
        (
            vec![
                vec![(2, 64), (2, 64)],
            ],
            false, // The cell image size is too small.
        ),
        (
            vec![
                vec![(64, 62), (64, 62)],
            ],
            false, // The cell image size is too small.
        ),
        (
            vec![
                vec![(64, 2)],
                vec![(64, 2)],
            ],
            false, // The cell image size is too small.
        ),
        (
            vec![
                vec![(62, 64)],
                vec![(62, 64)],
            ],
            false, // The cell image size is too small.
        ),
        // Right-most cells are narrower.
        (
            vec![
                vec![(100, 100), (100, 100), (66, 100)],
            ],
            true,
        ),
        // Bottom-most cells are shorter.
        (
            vec![
                vec![(100, 100), (100, 100)],
                vec![(100, 100), (100, 100)],
                vec![(100, 66), (100, 66)],
            ],
            true,
        ),
        // Right-most cells are narrower and bottom-most cells are shorter.
        (
            vec![
                vec![(100, 100), (100, 100), (66, 100)],
                vec![(100, 100), (100, 100), (66, 100)],
                vec![(100, 66), (100, 66), (66, 66)],
            ],
            true,
        ),
        // Right-most cells are wider.
        (
            vec![
                vec![(100, 100), (100, 100), (222, 100)],
                vec![(100, 100), (100, 100), (222, 100)],
                vec![(100, 100), (100, 100), (222, 100)],
            ],
            false,
        ),
        // Bottom-most cells are taller.
        (
            vec![
                vec![(100, 100), (100, 100), (100, 100)],
                vec![(100, 100), (100, 100), (100, 100)],
                vec![(100, 222), (100, 222), (100, 222)],
            ],
            false,
        ),
        // One cell dimension is off - case 1.
        (
            vec![
                vec![(100, 100), (100, 100), (100, 100)],
                vec![(100, 100), (66, 100), (100, 100)],
                vec![(100, 100), (100, 100), (100, 100)],
            ],
            false,
        ),
        // One cell dimension is off - case 2.
        (
            vec![
                vec![(100, 100), (100, 100), (66, 100)],
                vec![(100, 100), (100, 100), (66, 100)],
                vec![(100, 66), (100, 66), (66, 100)],
            ],
            false,
        ),
    ],
    [PixelFormat::Yuv420, PixelFormat::Yuv422, PixelFormat::Yuv444, PixelFormat::Yuv400],
    [8, 10, 12]
)]
fn encode_decode_grid(
    cells_and_expect_success: (Vec<Vec<(u32, u32)>>, bool),
    yuv_format: PixelFormat,
    depth: u8,
) -> AvifResult<()> {
    encode_decode_grid_impl(cells_and_expect_success, yuv_format, depth)
}

#[test_case(vec![vec![(64, 65), (64, 65)]], PixelFormat::Yuv422, true; "422 valid")]
#[test_case(vec![vec![(65, 64), (65, 64)]], PixelFormat::Yuv422, false; "422 invalid")]
#[test_case(vec![vec![(64, 65), (64, 65)]], PixelFormat::Yuv420, false; "420 invalid width")]
#[test_case(vec![vec![(65, 64), (65, 64)]], PixelFormat::Yuv420, false; "420 invalid height")]
fn encode_decode_grid_odd_dimensions(
    cells: Vec<Vec<(u32, u32)>>,
    yuv_format: PixelFormat,
    expect_success: bool,
) -> AvifResult<()> {
    encode_decode_grid_impl((cells, expect_success), yuv_format, /*depth=*/ 8)
}

#[test_matrix([true, false])]
fn encode_decode_grid_matrix_coefficients(same_matrix_coefficients: bool) -> AvifResult<()> {
    if !HAS_ENCODER {
        return Ok(());
    }
    let mut cell_images = Vec::new();
    for _ in 0..2 {
        cell_images.push(generate_gradient_image(
            /*width=*/ 64,
            /*height=*/ 64,
            /*depth=*/ 8,
            PixelFormat::Yuv444,
            YuvRange::Full,
            /*alpha=*/ true,
        )?);
    }
    cell_images[0].matrix_coefficients = MatrixCoefficients::Bt601;
    cell_images[1].matrix_coefficients = if same_matrix_coefficients {
        MatrixCoefficients::Bt601
    } else {
        MatrixCoefficients::Unspecified
    };
    let settings = encoder::Settings {
        speed: Some(10),
        ..Default::default()
    };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    let cell_image_refs: Vec<&Image> = cell_images.iter().collect();
    let res = encoder.add_image_grid(/*columns=*/ 2, /*rows=*/ 1, &cell_image_refs);
    if !same_matrix_coefficients {
        assert!(res.is_err());
        return Ok(());
    }
    assert!(res.is_ok());
    let edata = encoder.finish()?;
    assert!(!edata.is_empty());

    let mut decoder = decoder::Decoder::default();
    decoder.set_io_vec(edata);
    assert!(decoder.parse().is_ok());
    if !HAS_DECODER {
        return Ok(());
    }
    assert!(decoder.next_image().is_ok());
    Ok(())
}

#[test_matrix(
    [100, 121],
    [200, 107],
    [8, 10, 12],
    [PixelFormat::Yuv420, PixelFormat::Yuv422, PixelFormat::Yuv444, PixelFormat::Yuv400],
    [YuvRange::Limited, YuvRange::Full],
    [false, true],
    [
        RepetitionCount::Infinite,
        RepetitionCount::Finite(0),
        RepetitionCount::Finite(5),
        RepetitionCount::Finite(i32::MAX as u32 - 1),
        RepetitionCount::Finite(i32::MAX as u32),
    ]
)]
fn encode_decode_sequence(
    width: u32,
    height: u32,
    depth: u8,
    yuv_format: PixelFormat,
    yuv_range: YuvRange,
    alpha: bool,
    repetition_count: RepetitionCount,
) -> AvifResult<()> {
    if !HAS_ENCODER {
        return Ok(());
    }
    let mut input_images = Vec::new();
    const FRAME_COUNT: usize = 10;
    let durations: [u64; FRAME_COUNT] = [1000, 2000, 1500, 1100, 2300, 5000, 10000, 9000, 10, 500];
    let pts: [u64; FRAME_COUNT] = [0, 1000, 3000, 4500, 5600, 7900, 12900, 22900, 31900, 31910];
    for _ in 0..FRAME_COUNT {
        input_images.push(generate_gradient_image(
            width, height, depth, yuv_format, yuv_range, alpha,
        )?);
    }
    let images: Vec<&Image> = input_images.iter().collect();
    let settings = encoder::Settings {
        speed: Some(10),
        repetition_count,
        mutable: encoder::MutableSettings {
            quality: 50,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    for (index, image) in images.iter().enumerate() {
        encoder.add_image_for_sequence(image, durations[index])?;
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
    assert_eq!(
        decoder.repetition_count(),
        match repetition_count {
            RepetitionCount::Finite(x) if (0..i32::MAX as u32).contains(&x) => repetition_count,
            _ => RepetitionCount::Infinite,
        }
    );

    if !HAS_DECODER {
        return Ok(());
    }
    for index in 0..FRAME_COUNT {
        assert!(decoder.next_image().is_ok());
        let image_timing = decoder.nth_image_timing(index as u32)?;
        assert_eq!(image_timing.pts_in_timescales, pts[index]);
        assert_eq!(image_timing.duration_in_timescales, durations[index]);
    }
    Ok(())
}

#[test_matrix([true, false])]
fn sequence_alpha_combinations(first_image_has_alpha: bool) -> AvifResult<()> {
    if !HAS_ENCODER {
        return Ok(());
    }
    let width = 20;
    let height = 10;
    let depth = 8;
    let yuv_format = PixelFormat::Yuv420;
    let yuv_range = YuvRange::Full;
    let image1 = generate_gradient_image(
        width,
        height,
        depth,
        yuv_format,
        yuv_range,
        first_image_has_alpha,
    )?;
    let image2 = generate_gradient_image(
        width,
        height,
        depth,
        yuv_format,
        yuv_range,
        !first_image_has_alpha,
    )?;
    let settings = encoder::Settings {
        speed: Some(10),
        mutable: encoder::MutableSettings {
            quality: 50,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    // Adding the first image should always succeed.
    assert!(encoder.add_image_for_sequence(&image1, 0).is_ok());
    let second_image_res = encoder.add_image_for_sequence(&image2, 0);
    if first_image_has_alpha {
        // In this case, adding a second image without an alpha plane should fail.
        assert!(second_image_res.is_err());
        return Ok(());
    } else {
        // In this case, adding a second image with an alpha plane should succeed, but the
        // resulting image sequence will not have alpha in any of the frames.
        assert!(second_image_res.is_ok());
    }

    let edata = encoder.finish()?;
    assert!(!edata.is_empty());

    let mut decoder = decoder::Decoder::default();
    decoder.set_io_vec(edata);
    assert!(decoder.parse().is_ok());
    assert_eq!(decoder.compression_format(), CompressionFormat::Avif);
    assert_eq!(decoder.image_count(), 2);

    let image = decoder.image().expect("image was none");
    assert!(!image.alpha_present);
    assert!(image.image_sequence_track_present);
    assert_eq!(image.width, width);
    assert_eq!(image.height, height);
    assert_eq!(image.depth, depth);
    assert_eq!(image.yuv_format, yuv_format);
    assert_eq!(image.yuv_range, yuv_range);

    if !HAS_DECODER {
        return Ok(());
    }
    for _ in 0..2 {
        assert!(decoder.next_image().is_ok());
        let image = decoder.image().expect("image was none");
        assert!(!image.alpha_present);
    }
    Ok(())
}

#[test_matrix([0, 1, 65535], [0, 1, 65535])]
fn clli(max_cll: u16, max_pall: u16) -> AvifResult<()> {
    if !HAS_ENCODER || !HAS_DECODER {
        return Ok(());
    }
    let mut image = generate_gradient_image(8, 8, 8, PixelFormat::Yuv444, YuvRange::Full, false)?;
    image.clli = Some(ContentLightLevelInformation { max_cll, max_pall });

    let settings = encoder::Settings {
        speed: Some(10),
        ..Default::default()
    };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    encoder.add_image(&image)?;
    let edata = encoder.finish()?;
    assert!(!edata.is_empty());

    let mut decoder = decoder::Decoder::default();
    decoder.set_io_vec(edata);
    assert!(decoder.parse().is_ok());
    let decoded_image = decoder.image().unwrap();
    assert_eq!(decoded_image.clli, image.clli);

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

#[test_matrix([true, false])]
fn progressive_quality_change(use_grid: bool) -> AvifResult<()> {
    if !HAS_ENCODER {
        return Ok(());
    }
    let image = generate_gradient_image(256, 256, 8, PixelFormat::Yuv444, YuvRange::Full, false)?;
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

#[test_matrix([IFraction(1,2), IFraction(2, 6), IFraction(4, 32)], [true, false])]
fn progressive_dimension_change(scaling_fraction: IFraction, use_grid: bool) -> AvifResult<()> {
    if !HAS_ENCODER {
        return Ok(());
    }
    let image = generate_gradient_image(256, 256, 8, PixelFormat::Yuv444, YuvRange::Full, false)?;
    let mut settings = encoder::Settings {
        speed: Some(10),
        extra_layer_count: 1,
        mutable: encoder::MutableSettings {
            quality: 100,
            scaling_mode: ScalingMode {
                horizontal: scaling_fraction,
                vertical: scaling_fraction,
            },
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
    settings.mutable.scaling_mode = ScalingMode::default();
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
    let image = generate_gradient_image(256, 256, 8, PixelFormat::Yuv444, YuvRange::Full, false)?;
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
    let image = generate_gradient_image(256, 256, 8, PixelFormat::Yuv444, YuvRange::Full, false)?;
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

fn gainmap_metadata(base_is_hdr: bool) -> GainMapMetadata {
    let mut metadata = GainMapMetadata {
        use_base_color_space: true,
        base_hdr_headroom: if base_is_hdr { UFraction(6, 2) } else { UFraction(0, 1) },
        alternate_hdr_headroom: if base_is_hdr { UFraction(0, 1) } else { UFraction(6, 2) },
        ..Default::default()
    };
    for c in 0..3u32 {
        metadata.base_offset[c as usize] = Fraction(c as i32 * 10, 1000);
        metadata.alternate_offset[c as usize] = Fraction(c as i32 * 20, 1000);
        metadata.gamma[c as usize] = UFraction(1, c + 1);
        metadata.min[c as usize] = Fraction(-1, c + 1);
        metadata.max[c as usize] = Fraction(c as i32 + 11, c + 1);
    }
    metadata
}

fn generate_gainmap_image(base_is_hdr: bool) -> AvifResult<(Image, GainMap)> {
    let mut image = generate_gradient_image(12, 34, 10, PixelFormat::Yuv420, YuvRange::Full, true)?;
    image.transfer_characteristics = if base_is_hdr {
        TransferCharacteristics::Pq
    } else {
        TransferCharacteristics::Srgb
    };
    let mut gainmap = GainMap {
        image: generate_gradient_image(6, 17, 8, PixelFormat::Yuv420, YuvRange::Full, false)?,
        metadata: gainmap_metadata(base_is_hdr),
        ..Default::default()
    };
    gainmap.alt_plane_count = 3;
    gainmap.alt_matrix_coefficients = MatrixCoefficients::Smpte2085;
    let clli = ContentLightLevelInformation {
        max_cll: 10,
        max_pall: 5,
    };
    if base_is_hdr {
        image.clli = Some(clli);
        gainmap.alt_plane_depth = 8;
        gainmap.alt_color_primaries = ColorPrimaries::Bt601;
        gainmap.alt_transfer_characteristics = TransferCharacteristics::Srgb;
    } else {
        gainmap.alt_clli = clli;
        gainmap.alt_plane_depth = 10;
        gainmap.alt_color_primaries = ColorPrimaries::Bt2020;
        gainmap.alt_transfer_characteristics = TransferCharacteristics::Pq;
    }
    Ok((image, gainmap))
}

#[test]
fn gainmap_base_image_sdr() -> AvifResult<()> {
    let (image, gainmap) = generate_gainmap_image(false)?;
    let settings = encoder::Settings {
        speed: Some(10),
        mutable: encoder::MutableSettings {
            quality: 80,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    encoder.add_image_gainmap(&image, &gainmap)?;
    let edata = encoder.finish()?;
    assert!(!edata.is_empty());

    let mut decoder = decoder::Decoder::default();
    decoder.set_io_vec(edata);
    decoder.settings.image_content_to_decode = ImageContentType::All;
    assert!(decoder.parse().is_ok());
    assert!(decoder.image().unwrap().alpha_present);
    assert!(decoder.gainmap_present());
    let decoded_gainmap = decoder.gainmap();
    assert_eq!(
        decoded_gainmap.image.matrix_coefficients,
        gainmap.image.matrix_coefficients
    );
    assert_eq!(decoded_gainmap.alt_clli, gainmap.alt_clli);
    assert_eq!(decoded_gainmap.alt_plane_depth, 10);
    assert_eq!(decoded_gainmap.alt_plane_count, 3);
    assert_eq!(decoded_gainmap.alt_color_primaries, ColorPrimaries::Bt2020);
    assert_eq!(
        decoded_gainmap.alt_transfer_characteristics,
        TransferCharacteristics::Pq
    );
    assert_eq!(
        decoded_gainmap.alt_matrix_coefficients,
        MatrixCoefficients::Smpte2085
    );
    assert_eq!(decoded_gainmap.image.width, gainmap.image.width);
    assert_eq!(decoded_gainmap.image.height, gainmap.image.height);
    assert_eq!(decoded_gainmap.image.depth, gainmap.image.depth);
    assert_eq!(decoded_gainmap.metadata, gainmap.metadata);
    assert!(decoder.next_image().is_ok());
    let decoded_image = decoder.image().expect("failed to get image");
    assert!(decoded_image.has_plane(Plane::A));
    assert!(psnr(&image, decoded_image)? >= 40.0);
    let decoded_gainmap = decoder.gainmap();
    assert!(psnr(&gainmap.image, &decoded_gainmap.image)? >= 40.0);
    Ok(())
}

#[test]
fn gainmap_base_image_hdr() -> AvifResult<()> {
    let (image, gainmap) = generate_gainmap_image(true)?;
    let settings = encoder::Settings {
        speed: Some(10),
        mutable: encoder::MutableSettings {
            quality: 80,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    encoder.add_image_gainmap(&image, &gainmap)?;
    let edata = encoder.finish()?;
    assert!(!edata.is_empty());

    let mut decoder = decoder::Decoder::default();
    decoder.set_io_vec(edata);
    decoder.settings.image_content_to_decode = ImageContentType::All;
    assert!(decoder.parse().is_ok());
    assert!(decoder.gainmap_present());
    let decoded_gainmap = decoder.gainmap();
    let decoded_image = decoder.image().expect("failed to get decoded image");
    assert_eq!(
        decoded_gainmap.image.matrix_coefficients,
        gainmap.image.matrix_coefficients
    );
    assert_eq!(decoded_image.clli, image.clli);
    assert_eq!(
        decoded_gainmap.alt_clli,
        ContentLightLevelInformation::default()
    );
    assert_eq!(decoded_gainmap.alt_plane_depth, 8);
    assert_eq!(decoded_gainmap.alt_plane_count, 3);
    assert_eq!(decoded_gainmap.alt_color_primaries, ColorPrimaries::Bt601);
    assert_eq!(
        decoded_gainmap.alt_transfer_characteristics,
        TransferCharacteristics::Srgb
    );
    assert_eq!(
        decoded_gainmap.alt_matrix_coefficients,
        MatrixCoefficients::Smpte2085
    );
    assert_eq!(decoded_gainmap.image.width, gainmap.image.width);
    assert_eq!(decoded_gainmap.image.height, gainmap.image.height);
    assert_eq!(decoded_gainmap.image.depth, gainmap.image.depth);
    assert_eq!(decoded_gainmap.metadata, gainmap.metadata);
    assert!(decoder.next_image().is_ok());
    let decoded_image = decoder.image().expect("failed to get image");
    assert!(psnr(&image, decoded_image)? >= 40.0);
    let decoded_gainmap = decoder.gainmap();
    assert!(psnr(&gainmap.image, &decoded_gainmap.image)? >= 40.0);
    Ok(())
}

// From aviftransformtest.cc
#[test]
#[cfg(feature = "png")]
fn alpha_transformative_properties() -> AvifResult<()> {
    let mut reference = get_decoder("abc_color_irot_alpha_irot.avif");
    reference.parse()?;
    reference.next_image()?;
    let reference = reference.image().unwrap();

    let (mut image, _) = reader::Reader::read_frame(
        &mut reader::png::PngReader::create(&get_test_file("abc.png"))?,
        &reader::Config {
            yuv_format: Some(PixelFormat::Yuv444),
            matrix_coefficients: Some(reference.matrix_coefficients),
            ..Default::default()
        },
    )?;
    image.color_primaries = reference.color_primaries;
    image.transfer_characteristics = reference.transfer_characteristics;
    let mut encoder = encoder::Encoder::create_with_settings(&encoder::Settings {
        speed: Some(10),
        mutable: MutableSettings {
            quality: 100,
            quality_alpha: 100,
            quality_gainmap: 100,
            ..Default::default()
        },
        ..Default::default()
    })?;
    encoder.add_image(&image)?;
    let encoded = encoder.finish()?;
    assert!(!encoded.is_empty());

    let mut decoder = decoder::Decoder::default();
    decoder.set_io_vec(encoded);
    decoder.settings.image_content_to_decode = ImageContentType::All;
    decoder.parse()?;
    decoder.next_image()?;
    let decoded = decoder.image().unwrap();

    // Check with existing correct AVIF file.
    assert!(are_images_equal(decoded, reference)?);

    // The rendering of the images should be compared but that is outside the
    // scope of CrabbyAvif.
    Ok(())
}

#[test]
fn gainmap_oriented() -> AvifResult<()> {
    let (mut image, gainmap) = generate_gainmap_image(false)?;
    image.irot_angle = Some(1);
    image.imir_axis = Some(0);
    let settings = encoder::Settings {
        speed: Some(10),
        ..Default::default()
    };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    encoder.add_image_gainmap(&image, &gainmap)?;
    let edata = encoder.finish()?;
    assert!(!edata.is_empty());

    let mut decoder = decoder::Decoder::default();
    decoder.set_io_vec(edata);
    decoder.settings.image_content_to_decode = ImageContentType::All;
    assert!(decoder.parse().is_ok());
    assert!(decoder.gainmap_present());
    let decoded_image = decoder.image().expect("failed to get decoded image");
    assert_eq!(decoded_image.irot_angle, image.irot_angle);
    assert_eq!(decoded_image.imir_axis, image.imir_axis);
    let decoded_gainmap = decoder.gainmap();
    assert!(decoded_gainmap.image.irot_angle.is_none());
    assert!(decoded_gainmap.image.imir_axis.is_none());
    Ok(())
}

#[test]
fn gainmap_image_alpha_invalid() -> AvifResult<()> {
    let (image, mut gainmap) = generate_gainmap_image(false)?;
    // Invalid: gainmap.image must not have alpha plane.
    gainmap.image = generate_gradient_image(6, 17, 8, PixelFormat::Yuv420, YuvRange::Full, true)?;
    let settings = encoder::Settings {
        speed: Some(10),
        ..Default::default()
    };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    assert!(encoder.add_image_gainmap(&image, &gainmap).is_err());
    Ok(())
}

#[test_matrix([0, 1, 2])]
fn gainmap_oriented_invalid(transformation_index: u8) -> AvifResult<()> {
    let (image, mut gainmap) = generate_gainmap_image(false)?;
    // Gainmap image should not have a transformative property. Expect a failure.
    match transformation_index {
        0 => gainmap.image.irot_angle = Some(1),
        1 => gainmap.image.imir_axis = Some(0),
        2 => gainmap.image.pasp = Some(PixelAspectRatio::default()),
        _ => {} // not reached.
    }
    let settings = encoder::Settings {
        speed: Some(10),
        ..Default::default()
    };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    encoder.add_image_gainmap(&image, &gainmap)?;
    assert!(encoder.finish().is_err());
    Ok(())
}

#[test]
fn gainmap_all_channels_identical() -> AvifResult<()> {
    let (image, mut gainmap) = generate_gainmap_image(true)?;
    for c in 0..3 {
        gainmap.metadata.base_offset[c] = Fraction(1, 2);
        gainmap.metadata.alternate_offset[c] = Fraction(3, 4);
        gainmap.metadata.gamma[c] = UFraction(5, 6);
        gainmap.metadata.min[c] = Fraction(7, 8);
        gainmap.metadata.max[c] = Fraction(9, 10);
    }
    let settings = encoder::Settings {
        speed: Some(10),
        ..Default::default()
    };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    encoder.add_image_gainmap(&image, &gainmap)?;
    let edata = encoder.finish()?;
    assert!(!edata.is_empty());

    let mut decoder = decoder::Decoder::default();
    decoder.set_io_vec(edata);
    decoder.settings.image_content_to_decode = ImageContentType::All;
    assert!(decoder.parse().is_ok());
    assert!(decoder.gainmap_present());
    let decoded_gainmap = decoder.gainmap();
    assert_eq!(decoded_gainmap.metadata, gainmap.metadata);
    Ok(())
}

#[test]
fn gainmap_grid() -> AvifResult<()> {
    let grid_columns = 2;
    let grid_rows = 2;
    let cell_width = 128;
    let cell_height = 200;
    let mut cells = Vec::new();
    for _ in 0..grid_rows * grid_columns {
        let mut image = generate_gradient_image(
            cell_width,
            cell_height,
            10,
            PixelFormat::Yuv444,
            YuvRange::Full,
            false,
        )?;
        image.transfer_characteristics = TransferCharacteristics::Pq;
        let gainmap = GainMap {
            image: generate_gradient_image(
                cell_width / 2,
                cell_height / 2,
                8,
                PixelFormat::Yuv420,
                YuvRange::Full,
                false,
            )?,
            metadata: gainmap_metadata(true),
            ..Default::default()
        };
        cells.push((image, gainmap));
    }
    let mut images = Vec::new();
    let mut gainmaps = Vec::new();
    for cell in &cells {
        images.push(&cell.0);
        gainmaps.push(&cell.1);
    }

    let settings = encoder::Settings {
        speed: Some(10),
        ..Default::default()
    };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    encoder.add_image_gainmap_grid(grid_columns, grid_rows, &images, &gainmaps)?;
    let edata = encoder.finish()?;
    assert!(!edata.is_empty());

    let mut decoder = decoder::Decoder::default();
    decoder.set_io_vec(edata);
    decoder.settings.image_content_to_decode = ImageContentType::All;
    assert!(decoder.parse().is_ok());
    assert!(decoder.gainmap_present());
    assert!(decoder.next_image().is_ok());
    Ok(())
}

#[test_matrix([0, 1, 2, 3, 4])]
fn invalid_grid(test_case_index: u8) -> AvifResult<()> {
    let grid_columns = 2;
    let grid_rows = 2;
    let cell_width = 128;
    let cell_height = 200;
    let mut cells = Vec::new();
    for _ in 0..grid_rows * grid_columns {
        let mut image = generate_gradient_image(
            cell_width,
            cell_height,
            10,
            PixelFormat::Yuv444,
            YuvRange::Full,
            false,
        )?;
        image.transfer_characteristics = TransferCharacteristics::Pq;
        let gainmap = GainMap {
            image: generate_gradient_image(
                cell_width / 2,
                cell_height / 2,
                8,
                PixelFormat::Yuv420,
                YuvRange::Full,
                false,
            )?,
            metadata: gainmap_metadata(true),
            ..Default::default()
        };
        cells.push((image, gainmap));
    }

    let settings = encoder::Settings {
        speed: Some(10),
        ..Default::default()
    };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;

    match test_case_index {
        0 => {
            // Invalid: one gainmap cell has the wrong size.
            cells[1].1.image.height = 90;
        }
        1 => {
            // Invalid: one gainmap cell has a different depth.
            cells[1].1.image.depth = 12;
        }
        2 => {
            // Invalid: one gainmap cell has different gainmap metadata.
            cells[1].1.metadata.gamma[0] = UFraction(42, 1);
        }
        3 => {
            // Invalid: one image cell has the wrong size.
            cells[1].0.height = 90;
        }
        4 => {
            // Invalid: one gainmap cell has a different depth.
            cells[1].0.depth = 12;
        }
        _ => unreachable!(),
    }
    let images: Vec<_> = cells.iter().map(|x| &x.0).collect();
    let gainmaps: Vec<_> = cells.iter().map(|x| &x.1).collect();
    assert!(encoder
        .add_image_gainmap_grid(grid_columns, grid_rows, &images, &gainmaps)
        .is_err());
    Ok(())
}

#[test_matrix([8, 10, 12], [false, true])]
fn opaque_alpha(depth: u8, is_sequence: bool) -> AvifResult<()> {
    if !HAS_ENCODER {
        return Ok(());
    }
    let width = 2;
    let height = 2;
    let mut input_image = generate_gradient_image(
        width,
        height,
        depth,
        PixelFormat::Yuv420,
        YuvRange::Full,
        /*alpha=*/ true,
    )?;
    let opaque_value = input_image.max_channel();
    fill_plane(&mut input_image, Plane::A, opaque_value)?;
    let settings = encoder::Settings {
        speed: Some(10),
        ..Default::default()
    };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    if is_sequence {
        for i in 0..2 {
            encoder.add_image_for_sequence(&input_image, i)?;
        }
    } else {
        encoder.add_image(&input_image)?;
    }
    let edata = encoder.finish()?;
    assert!(!edata.is_empty());

    let mut decoder = decoder::Decoder::default();
    decoder.set_io_vec(edata);
    assert!(decoder.parse().is_ok());
    let image = decoder.image().expect("image was none");

    if is_sequence {
        assert_eq!(decoder.image_count(), 2);
        assert!(image.alpha_present);
        if !HAS_DECODER {
            return Ok(());
        }
        for _ in 0..2 {
            let res = decoder.next_image();
            assert!(res.is_ok());
            let image = decoder.image().expect("image was none");
            assert!(image.alpha_present);
            let alpha_plane = image.plane_data(Plane::A);
            assert!(alpha_plane.is_some());
            assert!(alpha_plane.unwrap().row_bytes > 0);
        }
    } else {
        assert_eq!(decoder.image_count(), 1);
        assert!(!image.alpha_present);
        if !HAS_DECODER {
            return Ok(());
        }
        let res = decoder.next_image();
        assert!(res.is_ok());
        let image = decoder.image().expect("image was none");
        assert!(!image.alpha_present);
        assert!(image.plane_data(Plane::A).is_none());
    }
    Ok(())
}

#[test_matrix([8, 10, 12], [true, false])]
fn opaque_alpha_grid(depth: u8, all_cells_opaque: bool) -> AvifResult<()> {
    if !HAS_ENCODER {
        return Ok(());
    }
    let width = 100;
    let height = 100;
    let mut image1 = generate_gradient_image(
        width,
        height,
        depth,
        PixelFormat::Yuv420,
        YuvRange::Full,
        /*alpha=*/ true,
    )?;
    let mut image2 = generate_gradient_image(
        width,
        height,
        depth,
        PixelFormat::Yuv420,
        YuvRange::Full,
        /*alpha=*/ true,
    )?;
    let opaque_value = image1.max_channel();
    fill_plane(&mut image1, Plane::A, opaque_value)?;
    fill_plane(&mut image2, Plane::A, opaque_value)?;
    if !all_cells_opaque {
        // Set some alpha pixels as not opaque in one of the cells.
        if depth == 8 {
            image2.row_mut(Plane::A, 0)?[0] = 10;
        } else {
            image2.row16_mut(Plane::A, 0)?[0] = 10;
        }
    }
    let settings = encoder::Settings {
        speed: Some(10),
        ..Default::default()
    };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    encoder.add_image_grid(1, 2, &[&image1, &image2])?;
    let edata = encoder.finish()?;
    assert!(!edata.is_empty());

    let mut decoder = decoder::Decoder::default();
    decoder.set_io_vec(edata);
    assert!(decoder.parse().is_ok());
    let image = decoder.image().expect("image was none");
    assert_eq!(decoder.image_count(), 1);
    assert_eq!(image.alpha_present, !all_cells_opaque);
    if !HAS_DECODER {
        return Ok(());
    }
    let res = decoder.next_image();
    assert!(res.is_ok());
    let image = decoder.image().expect("image was none");
    if all_cells_opaque {
        assert!(!image.alpha_present);
        assert!(image.plane_data(Plane::A).is_none());
    } else {
        assert!(image.alpha_present);
        let alpha_plane = image.plane_data(Plane::A);
        assert!(alpha_plane.is_some());
        assert!(alpha_plane.unwrap().row_bytes > 0);
    }
    Ok(())
}

#[test]
fn quality_categories() -> AvifResult<()> {
    if !HAS_ENCODER {
        return Ok(());
    }
    let width = 100;
    let height = 200;
    let depth = 8;
    let yuv_format = PixelFormat::Yuv420;
    let yuv_range = YuvRange::Full;
    let input_image = generate_gradient_image(
        width, height, depth, yuv_format, yuv_range, /*alpha=*/ true,
    )?;
    let settings = encoder::Settings {
        speed: Some(10),
        mutable: encoder::MutableSettings {
            quality: 5,
            quality_alpha: 100,
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
    let image = decoder.image().expect("image was none");
    assert!(image.alpha_present);
    if !HAS_DECODER {
        return Ok(());
    }
    assert!(decoder.next_image().is_ok());
    let image = decoder.image().expect("image was none");

    // Color planes should have some loss because the quality was set to a low value.
    assert!(!are_planes_equal(image, &input_image, Plane::Y)?);
    assert!(!are_planes_equal(image, &input_image, Plane::U)?);
    assert!(!are_planes_equal(image, &input_image, Plane::V)?);

    // Alpha plane should be lossless.
    assert!(are_planes_equal(image, &input_image, Plane::A)?);

    Ok(())
}

#[test]
fn codec_versions() {
    assert!(crabby_avif::codec_versions().contains("dav1d"));
    assert!(crabby_avif::codec_versions().contains("aom"));
}
