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

use crabby_avif::image::*;
use crabby_avif::*;

mod utils;
use utils::*;

#[test]
fn encode_decode_extended_pixi() -> AvifResult<()> {
    if !HAS_ENCODER {
        return Ok(());
    }
    let input_image =
        generate_gradient_image(64, 64, 8, PixelFormat::Yuv420, YuvRange::Full, true)?;
    let settings = encoder::Settings {
        speed: Some(10),
        write_extended_pixi: true,
        ..Default::default()
    };
    let mut encoder = encoder::Encoder::create_with_settings(&settings)?;
    encoder.add_image(&input_image)?;
    let edata = encoder.finish()?;
    // Make sure that a PixelInformationProperty was written with px_flags=1.
    assert_ne!(
        edata
            .as_slice()
            .windows(8)
            .position(|eight_bytes| eight_bytes == [b'p', b'i', b'x', b'i', 0, 0, 0, 1]),
        None
    );

    let mut decoder = decoder::Decoder::default();
    decoder.set_io_vec(edata);
    assert_eq!(decoder.parse(), Ok(()));
    if !HAS_DECODER {
        return Ok(());
    }
    assert_eq!(decoder.next_image(), Ok(()));
    assert_eq!(decoder.image().unwrap().yuv_format, input_image.yuv_format);
    assert_eq!(
        decoder.image().unwrap().chroma_sample_position,
        input_image.chroma_sample_position
    );
    Ok(())
}
