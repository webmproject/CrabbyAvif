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

#![cfg(feature = "png")]

mod utils;
use utils::*;

#[cfg(all(feature = "jpeg", feature = "encoder"))]
use crabby_avif::utils::reader::jpeg::JpegReader;
use crabby_avif::utils::reader::png::PngReader;
use crabby_avif::utils::reader::Config;
use crabby_avif::utils::reader::Reader;
use crabby_avif::*;

use test_case::test_case;
#[cfg(feature = "encoder")]
use test_case::test_matrix;

#[test_case("paris_identity.avif", "paris_icc_exif_xmp.png"; "lossless_identity")]
#[test_case("paris_ycgco_re.avif", "paris_icc_exif_xmp.png"; "lossless_ycgco_re")]
fn lossless(avif_file: &str, png_file: &str) -> AvifResult<()> {
    let mut decoder = get_decoder(avif_file);
    assert!(decoder.parse().is_ok());
    if !HAS_DECODER {
        return Ok(());
    }
    assert!(decoder.next_image().is_ok());
    let avif_image = decoder.image().expect("image was none");
    let mut png_reader = PngReader::create(&get_test_file(png_file))?;
    let (png_image, _) = png_reader.read_frame(&Config {
        yuv_format: Some(avif_image.yuv_format),
        depth: Some(avif_image.depth),
        matrix_coefficients: Some(avif_image.matrix_coefficients),
    })?;
    assert!(are_images_equal(avif_image, &png_image)?);
    Ok(())
}

#[test_matrix(
    ["paris_icc_exif_xmp.png", "paris_exif_xmp_icc.jpg"],
    [MatrixCoefficients::Identity, MatrixCoefficients::Ycgco, MatrixCoefficients::YcgcoRe],
    [PixelFormat::Yuv444, PixelFormat::Yuv420]
)]
#[cfg(feature = "encoder")]
fn lossless_roundtrip(
    input_file: &str,
    matrix_coefficients: MatrixCoefficients,
    yuv_format: PixelFormat,
) -> AvifResult<()> {
    if !HAS_ENCODER {
        return Ok(());
    }
    if input_file.ends_with("jpg") && !cfg!(feature = "jpeg") {
        return Ok(());
    }
    if matrix_coefficients == MatrixCoefficients::Identity && yuv_format != PixelFormat::Yuv444 {
        // The AV1 spec does not allow identity with subsampling.
        return Ok(());
    }
    let input_file_abs = get_test_file(input_file);
    let mut reader: Box<dyn Reader> = if input_file.ends_with("png") {
        Box::new(PngReader::create(&input_file_abs)?)
    } else {
        #[cfg(feature = "jpeg")]
        {
            Box::new(JpegReader::create(&input_file_abs)?)
        }
        #[cfg(not(feature = "jpeg"))]
        unreachable!();
    };
    let (image, _) = reader.read_frame(&Config {
        yuv_format: Some(yuv_format),
        matrix_coefficients: Some(matrix_coefficients),
        ..Default::default()
    })?;

    let settings = encoder::Settings {
        speed: Some(10),
        mutable: encoder::MutableSettings {
            quality: 100.0,
            ..Default::default()
        },
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
    decoder.set_io_vec(edata);
    assert!(decoder.parse().is_ok());
    assert!(decoder.next_image().is_ok());
    let decoded_image = decoder.image().expect("image was none");
    assert!(are_images_equal(&image, decoded_image)?);
    Ok(())
}
