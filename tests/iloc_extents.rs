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

mod utils;

#[cfg(feature = "png")]
use crabby_avif::utils::reader::png::PngReader;
#[cfg(feature = "png")]
use crabby_avif::utils::reader::Config;
#[cfg(feature = "png")]
use crabby_avif::utils::reader::Reader;
use crabby_avif::AvifResult;
use utils::*;

#[test]
fn iloc_extents() -> AvifResult<()> {
    let mut decoder = get_decoder("sacre_coeur_2extents.avif");
    assert!(decoder.parse().is_ok());
    if !HAS_DECODER {
        return Ok(());
    }
    assert!(decoder.next_image().is_ok());
    #[cfg(feature = "png")]
    {
        let decoded = decoder.image().expect("image was none");
        // sacre_coeur_2extents.avif was generated with
        //   avifenc --lossless --ignore-exif --ignore-xmp --ignore-icc sacre_coeur.png
        // so pixels can be compared byte by byte.
        let mut png_reader = PngReader::create(&get_test_file("sacre_coeur.png"))?;
        let (mut png_image, _, _) = png_reader.read_frame(&Config {
            yuv_format: Some(decoded.yuv_format),
            depth: Some(decoded.depth),
            matrix_coefficients: Some(decoded.matrix_coefficients),
            ..Default::default()
        })?;
        // PngReader sets these to Unspecified if there is no CICP info in the file.
        png_image.color_primaries = decoded.color_primaries;
        png_image.transfer_characteristics = decoded.transfer_characteristics;
        assert!(are_images_equal(decoded, &png_image)?);
    }
    Ok(())
}

#[test]
fn nth_image_max_extent() {
    let mut decoder = get_decoder("sacre_coeur_2extents.avif");
    assert!(decoder.parse().is_ok());

    let max_extent = decoder.nth_image_max_extent(0).unwrap();
    assert_eq!(max_extent.offset, 290);
    assert_eq!(max_extent.size, 1000 + 1 + 5778); // '\0' in the middle.
}
