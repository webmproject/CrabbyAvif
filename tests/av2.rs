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

#![allow(clippy::disallowed_methods, clippy::disallowed_macros)]
#![cfg(feature = "avm")]

use crabby_avif::decoder::CompressionFormat;
use crabby_avif::image::*;
use crabby_avif::*;

mod utils;
use utils::*;

use test_case::test_matrix;

// Equivalent to Image::free_planes which is not public.
fn free_planes(image: &mut Image, planes: &[Plane]) {
    for plane in planes {
        let plane = *plane as usize;
        image.planes[plane] = None;
        image.row_bytes[plane] = 0;
    }
}

#[derive(Debug, PartialEq)]
enum Alpha {
    None,
    Opaque,
    Unpremultiplied,
    Premultiplied,
}

#[test_matrix(
    [1, 2],
    [1, 2],
    [8, 10, 12],
    [PixelFormat::Yuv420, PixelFormat::Yuv444],
    [YuvRange::Limited, YuvRange::Full],
    [
        Alpha::None,
        Alpha::Opaque,
        Alpha::Unpremultiplied,
        Alpha::Premultiplied
    ],
    [1, 2]
)]
fn encode_decode(
    width: u32,
    height: u32,
    depth: u8,
    format: PixelFormat,
    range: YuvRange,
    alpha: Alpha,
    image_count: usize,
) -> AvifResult<()> {
    let mut image =
        generate_gradient_image(width, height, depth, format, range, alpha != Alpha::None)?;
    if alpha == Alpha::Opaque {
        let max_channel = image.max_channel();
        fill_plane(&mut image, Plane::A, max_channel)?;
    }
    // This may result in invalid premultiplied color samples but CrabbyAvif
    // does not reject that for now.
    image.alpha_premultiplied = alpha == Alpha::Premultiplied;

    // Generate different frames for sequence encoding.
    let mut next_frames = Vec::new();
    for i in 1..image_count {
        let mut image = image.try_deep_clone()?;
        image.fill_plane_with_value((i % image.yuv_format.plane_count()).into(), i as u16)?;
        next_frames.push(image);
    }

    let encoded = {
        let settings = encoder::Settings {
            codec_choice: CodecChoice::Avm,
            speed: Some(9), // Fastest libavm setting.
            mutable: encoder::MutableSettings {
                quality: 90.0,
                quality_alpha: 90.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
        if next_frames.is_empty() {
            encoder.add_image(&image)?;
        } else {
            encoder.add_image_for_sequence(&image, 1)?;
            for image in &next_frames {
                encoder.add_image_for_sequence(image, 1)?;
            }
        }
        encoder.finish()?
    };
    assert!(!encoded.is_empty());
    // When using AV2, alpha is stored as an embedded layer, not a separate auxiliary item.
    assert!(!encoded.windows(4).any(|w| w == b"auxl"));
    let expected_alpha_present =
        alpha != Alpha::None && (image_count > 1 || alpha != Alpha::Opaque);
    if !expected_alpha_present {
        assert!(!encoded.windows(4).any(|w| w == b"alpi"));
    }

    let mut decoder = decoder::Decoder::default();
    // Explicitly selecting libavm should not be necessary.
    decoder.settings.codec_choice = CodecChoice::Auto;
    decoder.set_io_vec(encoded);
    assert!(decoder.parse().is_ok());
    assert_eq!(decoder.compression_format(), CompressionFormat::Avif2);
    assert_eq!(decoder.image_count(), image_count as u32);

    let decoded = decoder.image().unwrap();
    assert!(decoded.has_same_properties_and_cicp(&image));
    assert_eq!(decoded.alpha_present, expected_alpha_present);
    if decoded.alpha_present {
        assert_eq!(decoded.alpha_premultiplied, image.alpha_premultiplied);
    }
    assert_eq!(decoded.image_sequence_track_present, image_count > 1);

    for frame in std::iter::once(&mut image).chain(next_frames.iter_mut()) {
        assert!(decoder.next_image().is_ok());
        let decoded = decoder.image().unwrap();
        assert_eq!(decoded.has_plane(Plane::A), expected_alpha_present);
        // Expect no alpha plane in the decoded image if the original alpha
        // is present but opaque in a single image.
        if !expected_alpha_present {
            free_planes(frame, &[Plane::A]);
        }
        let frame_psnr = psnr(decoded, frame)?;
        assert!(frame_psnr >= 50.0, "PSNR: {frame_psnr}");
    }
    Ok(())
}

// Finds the byte offset within `sample` where the first AV2 OBU with `mlayer_id == 1` starts.
fn find_av2_mlayer1_offset(sample: &[u8]) -> Option<usize> {
    let mut pos = 0;
    while pos < sample.len() {
        let obu_start = pos;
        let mut obu_size = 0usize;
        for i in 0..8 {
            let b = *sample.get(pos)?;
            pos += 1;
            obu_size |= ((b & 0x7f) as usize) << (7 * i);
            if b & 0x80 == 0 {
                break;
            }
        }
        let header0 = *sample.get(pos)?;
        let ext_flag = (header0 >> 7) & 1;
        if ext_flag == 1 {
            let header1 = *sample.get(pos + 1)?;
            let mlayer_id = (header1 >> 5) & 0x7;
            if mlayer_id == 1 {
                return Some(obu_start);
            }
        }
        pos += obu_size;
    }
    None
}

// Verifies that when an AV2 sequence with embedded (native) alpha fails on a subsequent frame—
// either because the embedded alpha layer is missing or because decoding the frame fails—
// decoding returns an error and clears Plane::A (as well as color planes) on decoder.image().
#[test]
fn embedded_alpha_failure_in_sequence() -> AvifResult<()> {
    let frame0 = generate_gradient_image(16, 16, 8, PixelFormat::Yuv420, YuvRange::Full, true)?;
    let mut frame1 = frame0.try_deep_clone()?;
    frame1.fill_plane_with_value(Plane::Y, 100)?;
    fill_plane(&mut frame1, Plane::A, 200)?;

    let encoded = {
        let settings = encoder::Settings {
            codec_choice: CodecChoice::Avm,
            speed: Some(9),
            mutable: encoder::MutableSettings {
                quality: 90.0,
                quality_alpha: 90.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
        encoder.add_image_for_sequence(&frame0, 1)?;
        encoder.add_image_for_sequence(&frame1, 1)?;
        encoder.finish()?
    };
    assert!(!encoded.windows(4).any(|w| w == b"auxl"));

    // Locate the single track's stco and stsz entries.
    let stco_pos = encoded.windows(4).position(|w| w == b"stco").unwrap();
    let chunk_offset =
        u32::from_be_bytes(encoded[stco_pos + 12..stco_pos + 16].try_into().unwrap()) as usize;
    let stsz_pos = encoded.windows(4).position(|w| w == b"stsz").unwrap();
    let sample0_size =
        u32::from_be_bytes(encoded[stsz_pos + 16..stsz_pos + 20].try_into().unwrap()) as usize;
    let sample1_size =
        u32::from_be_bytes(encoded[stsz_pos + 20..stsz_pos + 24].try_into().unwrap()) as usize;
    let sample1_offset = chunk_offset + sample0_size;

    // Case 1: Drop the embedded alpha OBU (mlayer_id == 1) from frame 1 so color decodes cleanly
    // but alpha_frame is null.
    let mut missing_alpha = encoded.clone();
    let color_only_len =
        find_av2_mlayer1_offset(&encoded[sample1_offset..sample1_offset + sample1_size])
            .expect("expected embedded alpha OBU in frame 1");
    assert!(color_only_len > 0 && color_only_len < sample1_size);
    missing_alpha[stsz_pos + 20..stsz_pos + 24]
        .copy_from_slice(&(color_only_len as u32).to_be_bytes());

    let mut decoder = decoder::Decoder::default();
    decoder.settings.codec_choice = CodecChoice::Avm;
    decoder.set_io_vec(missing_alpha);
    assert_eq!(decoder.parse(), Ok(()));
    assert!(decoder.image().unwrap().alpha_present);
    assert_eq!(decoder.next_image(), Ok(()));
    assert!(decoder.image().unwrap().has_plane(Plane::A));
    assert_eq!(decoder.next_image(), Err(AvifError::DecodeAlphaFailed));
    assert!(decoder.next_image().is_err());
    let decoded = decoder.image().unwrap();
    assert!(!decoded.has_plane(Plane::Y));
    assert!(!decoded.has_plane(Plane::A));

    // Case 2: Corrupt frame 1 so avm_codec_decode() fails after releasing frame 0's buffers.
    let mut corrupt_frame1 = encoded;
    corrupt_frame1[sample1_offset..sample1_offset + sample1_size].fill(0xff);

    let mut decoder = decoder::Decoder::default();
    decoder.settings.codec_choice = CodecChoice::Avm;
    decoder.set_io_vec(corrupt_frame1);
    assert_eq!(decoder.parse(), Ok(()));
    assert!(decoder.image().unwrap().alpha_present);
    assert_eq!(decoder.next_image(), Ok(()));
    assert!(decoder.image().unwrap().has_plane(Plane::A));
    assert!(decoder.next_image().is_err());
    assert!(decoder.next_image().is_err());
    let decoded = decoder.image().unwrap();
    assert!(!decoded.has_plane(Plane::Y));
    assert!(!decoded.has_plane(Plane::A));

    Ok(())
}
