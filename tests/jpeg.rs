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
#![cfg(feature = "jpeg")]

use crabby_avif::gainmap::GainMapMetadata;
use crabby_avif::utils::reader::jpeg::JpegReader;
use crabby_avif::utils::reader::Config;
use crabby_avif::utils::reader::Reader;
use crabby_avif::utils::Fraction;
use crabby_avif::utils::UFraction;
use crabby_avif::*;

mod utils;
use utils::*;

use test_case::test_case;

#[test_case("apple_gainmap_v1.jpg", true, true, false, true)]
#[test_case("apple_gainmap_v2.jpg", true, true, false, true)]
#[test_case("apple_gainmap_v3.jpg", true, true, false, true)]
#[test_case("dog_exif_extended_xmp_icc.jpg", true, true, true, false)]
#[test_case("paris_exif_orientation_5.jpg", true, false, false, false)]
#[test_case("paris_exif_xmp_gainmap_bigendian.jpg", true, false, true, true)]
#[test_case("paris_exif_xmp_gainmap_littleendian.jpg", true, false, true, true)]
#[test_case("paris_exif_xmp_icc_gainmap_bigendian.jpg", true, true, true, true)]
#[test_case("paris_exif_xmp_icc.jpg", true, true, true, false)]
#[test_case("paris_exif_xmp_modified_icc.jpg", true, true, true, false)]
#[test_case("paris_extended_xmp.jpg", false, false, true, false)]
#[test_case("paris_xmp_trailing_null.jpg", false, false, true, false)]
#[test_case("seine_sdr_gainmap_srgb.jpg", true, true, true, true)]
fn reader(
    filename: &str,
    has_exif: bool,
    has_icc: bool,
    has_xmp: bool,
    has_gainmap: bool,
) -> AvifResult<()> {
    let mut reader = JpegReader::create(&get_test_file(filename))?;
    let (image, _, gainmap) = reader.read_frame(&Config::default())?;
    assert_eq!(!image.exif.is_empty(), has_exif);
    assert_eq!(!image.icc.is_empty(), has_icc);
    assert_eq!(!image.xmp.is_empty(), has_xmp);
    assert_eq!(gainmap.is_some(), has_gainmap);
    Ok(())
}

#[test]
fn exif_orientation() -> AvifResult<()> {
    let mut reader = JpegReader::create(&get_test_file("paris_exif_orientation_5.jpg"))?;
    let (image, _, _) = reader.read_frame(&Config::default())?;
    assert_eq!(image.irot_angle, Some(1));
    assert_eq!(image.imir_axis, Some(0));
    Ok(())
}

#[test_case("apple_gainmap_v1.jpg")]
#[test_case("apple_gainmap_v2.jpg")]
#[test_case("apple_gainmap_v3.jpg")]
fn read_apple_jpeg_with_gain_map(filename: &str) -> AvifResult<()> {
    let mut reader = JpegReader::create(&get_test_file(filename))?;
    let (_, _, gainmap) = reader.read_frame(&Config::default())?;
    let gainmap = gainmap.expect("expected gain map");
    assert_eq!(gainmap.image.width, 192);
    assert_eq!(gainmap.image.height, 256);

    let expected_metadata = if filename == "apple_gainmap_v3.jpg" {
        GainMapMetadata {
            min: [Fraction::from(-0.332520); 3],
            max: [Fraction::from(1.757545); 3],
            gamma: [UFraction::from(0.873047); 3],
            base_offset: [Fraction::from(0.000010); 3],
            alternate_offset: [Fraction::from(0.000010); 3],
            base_hdr_headroom: UFraction(0, 1),
            alternate_hdr_headroom: UFraction::from(1.757545),
            use_base_color_space: true,
        }
    } else {
        let headroom = if filename == "apple_gainmap_v1.jpg" { 3.0 } else { 4.532783f64.log2() };
        GainMapMetadata {
            min: [Fraction(0, 1); 3],
            max: [Fraction::from(headroom); 3],
            gamma: [UFraction(1, 1); 3],
            base_offset: [Fraction(0, 1); 3],
            alternate_offset: [Fraction(0, 1); 3],
            base_hdr_headroom: UFraction(0, 1),
            alternate_hdr_headroom: UFraction::from(headroom),
            use_base_color_space: true,
        }
    };
    assert_eq!(gainmap.metadata, expected_metadata);
    Ok(())
}
