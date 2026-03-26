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

#![cfg(feature = "jpeg")]

use crabby_avif::utils::reader::jpeg::JpegReader;
use crabby_avif::utils::reader::Config;
use crabby_avif::utils::reader::Reader;
use crabby_avif::*;

mod utils;
use utils::*;

use test_case::test_case;

#[test_case("apple_gainmap_new.jpg", true, true, false, true)]
#[test_case("apple_gainmap_old.jpg", true, true, false, true)]
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
