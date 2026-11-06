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
    // TODO(b/437292541): 1x1 does not pass (PSNR < 15dB). Investigate.
    [2],
    [2],
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
