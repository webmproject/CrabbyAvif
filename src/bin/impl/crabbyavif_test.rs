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

use crabby_avif::*;

use crate::r#impl::main_impl;

#[test]
fn crabbyavif_test() -> AvifResult<()> {
    assert_eq!(
        main_impl(["crabbyavif"].iter()),
        Err(AvifError::UnknownError("input_file is required".into()))
    );
    assert_eq!(
        main_impl(["crabbyavif", "--info", "--extract-thumbnail", "input.avif"].iter()),
        Err(AvifError::UnknownError(
            "--info contains unsupported extra arguments".into()
        ))
    );
    assert_eq!(
        main_impl(["crabbyavif", "--info", "--crop", "0,0,8,8", "input.avif"].iter()),
        Err(AvifError::UnknownError(
            "--info contains unsupported extra arguments".into()
        ))
    );
    assert_eq!(
        main_impl(
            [
                "crabbyavif",
                "--progressive",
                "--crop",
                "0,0,8,8",
                "input.avif",
                "output.y4m"
            ]
            .iter()
        ),
        Err(AvifError::UnknownError(
            "crop is not supported with progressive decoding".into()
        ))
    );
    assert_eq!(
        main_impl(["crabbyavif", "--info", "--target-size", "8,8", "input.avif"].iter()),
        Err(AvifError::UnknownError(
            "--info contains unsupported extra arguments".into()
        ))
    );
    assert_eq!(
        main_impl(
            [
                "crabbyavif",
                "--progressive",
                "--target-size",
                "8,8",
                "input.avif",
                "output.y4m"
            ]
            .iter()
        ),
        Err(AvifError::UnknownError(
            "target-size is not supported with progressive decoding".into()
        ))
    );
    assert_eq!(
        main_impl(
            [
                "crabbyavif",
                "--target-size",
                "8,8",
                "input.png",
                "output.avif"
            ]
            .iter()
        ),
        Err(AvifError::UnknownError(
            "target-size is only supported when decoding".into()
        ))
    );
    Ok(())
}

#[test]
fn grid_parser_and_validation() {
    use crate::r#impl::grid_parser;
    use crate::r#impl::GridSize;
    assert_eq!(grid_parser("1x1"), Ok(GridSize { cols: 1, rows: 1 }));
    assert_eq!(grid_parser("2x3"), Ok(GridSize { cols: 2, rows: 3 }));
    for invalid in ["0x2", "2x0", "2X2", "257x1", "256x256", "abc"] {
        assert!(grid_parser(invalid).is_err(), "{invalid} should fail");
    }
    assert_eq!(
        main_impl(["crabbyavif", "--grid", "2x2", "--info", "input.avif"].iter()),
        Err(AvifError::UnknownError(
            "--info contains unsupported extra arguments".into()
        ))
    );
    assert_eq!(
        main_impl(["crabbyavif", "-g", "2x2", "--info", "input.avif"].iter()),
        Err(AvifError::UnknownError(
            "--info contains unsupported extra arguments".into()
        ))
    );
    assert_eq!(
        main_impl(["crabbyavif", "--grid", "2x2", "input.avif", "output.png"].iter()),
        Err(AvifError::UnknownError(
            "grid is only supported when encoding".into()
        ))
    );
    assert_eq!(
        main_impl(
            [
                "crabbyavif",
                "--grid",
                "2x2",
                "--progressive",
                "input.avif",
                "output.avif",
            ]
            .iter()
        ),
        Err(AvifError::UnknownError(
            "grid is not supported with progressive encoding".into()
        ))
    );
    assert_eq!(
        main_impl(
            [
                "crabbyavif",
                "--grid",
                "2x2",
                "--progressive",
                "input.png",
                "output.avif",
            ]
            .iter()
        ),
        Err(AvifError::UnknownError(
            "grid is not supported with progressive encoding".into()
        ))
    );
}

#[test]
fn gainmap_info_rows_formatting_and_clli_regression() {
    use crabby_avif::gainmap::GainMap;
    use crabby_avif::gainmap::GainMapMetadata;
    use crabby_avif::utils::Fraction;
    use crabby_avif::utils::UFraction;

    let gm = GainMap {
        image: image::Image {
            width: 100,
            height: 50,
            depth: 8,
            yuv_format: PixelFormat::Yuv444,
            clli: Some(ContentLightLevelInformation {
                max_cll: 999,
                max_pall: 333,
            }),
            ..Default::default()
        },
        metadata: GainMapMetadata {
            base_hdr_headroom: UFraction(1, 2),
            alternate_hdr_headroom: UFraction(0, 1),
            min: [Fraction(i32::MIN, 2), Fraction(1, 0), Fraction(-3, 2)],
            max: [Fraction(2, 1), Fraction(4, 1), Fraction(6, 1)],
            base_offset: [Fraction(1, 64), Fraction(2, 64), Fraction(3, 64)],
            alternate_offset: [Fraction(4, 64), Fraction(5, 64), Fraction(6, 64)],
            gamma: [UFraction(1, 1), UFraction(2, 1), UFraction(0, 0)],
            use_base_color_space: true,
            ..Default::default()
        },
        alt_color_primaries: ColorPrimaries::Bt2020,
        alt_transfer_characteristics: TransferCharacteristics::Pq,
        alt_matrix_coefficients: MatrixCoefficients::Bt2020Ncl,
        alt_plane_count: 3,
        alt_plane_depth: 10,
        alt_clli: ContentLightLevelInformation::default(),
        ..Default::default()
    };

    // Regression: gm.image.clli set while alt_clli is zero must omit Alternate CLLI.
    let rows_decoded = crate::r#impl::gainmap_info_rows(&gm, true);
    fn val_of<'a>(rows: &'a [(usize, &'static str, String)], label: &str) -> Option<&'a str> {
        rows.iter()
            .find(|(_, k, _)| *k == label)
            .map(|(_, _, v)| v.as_str())
    }
    assert_eq!(
        val_of(&rows_decoded, "Gainmap"),
        Some("100x50 pixels, 8 bit, Yuv444, Full Range, Matrix Coeffs. Unspecified, Base Image is HDR")
    );
    assert_eq!(
        val_of(&rows_decoded, "Base headroom"),
        Some("0.5 (as fraction: 1/2)")
    );
    assert_eq!(
        val_of(&rows_decoded, "Alternate headroom"),
        Some("0 (as fraction: 0/1)")
    );
    let min_row = val_of(&rows_decoded, "Gain Map Min").unwrap();
    assert!(min_row.contains("-1073741824 (as fraction: -2147483648/2)"));
    assert!(min_row.contains("0 (as fraction: 1/0)"));
    assert!(min_row.contains("-1.5 (as fraction: -3/2)"));
    assert_eq!(
        val_of(&rows_decoded, "Gain Map Max").unwrap(),
        format!(
            "R {:<40} G {:<40} B {}",
            "2 (as fraction: 2/1)", "4 (as fraction: 4/1)", "6 (as fraction: 6/1)"
        )
    );
    assert_eq!(val_of(&rows_decoded, "Use Base Color Space"), Some("True"));
    assert_eq!(val_of(&rows_decoded, "Bit Depth"), Some("10"));
    assert_eq!(val_of(&rows_decoded, "Planes"), Some("3"));
    assert_eq!(val_of(&rows_decoded, "CLLI"), None);

    // Non-decoded HDR/SDR summary + alt_clli non-zero + use_base_color_space=false.
    let mut gm_undecoded = GainMap {
        metadata: GainMapMetadata {
            use_base_color_space: false,
            ..Default::default()
        },
        alt_clli: ContentLightLevelInformation {
            max_cll: 1000,
            max_pall: 400,
        },
        ..Default::default()
    };
    gm_undecoded.image.clli = None;
    let rows_undecoded = crate::r#impl::gainmap_info_rows(&gm_undecoded, false);
    assert_eq!(
        val_of(&rows_undecoded, "Gainmap"),
        Some("Present (not decoded), Base Image is SDR")
    );
    assert_eq!(
        val_of(&rows_undecoded, "Use Base Color Space"),
        Some("False")
    );
    assert_eq!(val_of(&rows_undecoded, "CLLI"), Some("1000, 400"));
}

// The tests below decode images, so they require an AV1 decoder.
#[cfg(any(
    feature = "dav1d",
    feature = "android_mediacodec",
    feature = "native_av1"
))]
mod decode_tests {
    use super::*;
    use crabby_avif::utils::reader::y4m::Y4MReader;
    use crabby_avif::utils::reader::Config;
    use crabby_avif::utils::reader::Reader;

    fn get_test_file(filename: &str) -> String {
        let base_path = if cfg!(google3) {
            format!(
                "{}/google3/third_party/crabbyavif/",
                std::env::var("TEST_SRCDIR").expect("TEST_SRCDIR is not defined")
            )
        } else {
            "".to_string()
        };
        format!("{base_path}tests/data/{filename}")
    }

    // Decodes input_file with the given extra arguments into a Y4M file and returns its dimensions.
    fn decode_to_y4m(input_file: &str, extra_args: &[&str]) -> AvifResult<(u32, u32)> {
        let temp_dir = tempfile::tempdir()
            .map_err(|_| AvifError::UnknownError("Failed to create temp dir".into()))?;
        let output_file = temp_dir.path().join("output.y4m");
        let output_file = output_file.to_str().unwrap();
        let input_file = get_test_file(input_file);
        let mut args = vec!["crabbyavif"];
        args.extend_from_slice(extra_args);
        args.extend_from_slice(&[input_file.as_str(), output_file]);
        main_impl(args.iter())?;
        let image = Y4MReader::create(output_file)?
            .read_frame(&Config::default())?
            .0;
        Ok((image.width, image.height))
    }

    #[test]
    fn decode_without_crop() {
        assert_eq!(decode_to_y4m("paris_10bpc.avif", &[]), Ok((403, 302)));
    }

    #[test]
    fn crop() {
        assert_eq!(
            decode_to_y4m("paris_10bpc.avif", &["--crop", "10,20,100,50"]),
            Ok((100, 50))
        );
    }

    #[test]
    fn crop_out_of_bounds() {
        assert_eq!(
            decode_to_y4m("paris_10bpc.avif", &["--crop", "400,0,10,10"]),
            Err(AvifError::UnknownError(
                "Invalid crop rectangle: it must fit in the image (or in its clean aperture)"
                    .into()
            ))
        );
    }

    #[test]
    fn extract_missing_gainmap() {
        assert_eq!(
            decode_to_y4m("paris_10bpc.avif", &["--extract-gainmap"]),
            Err(AvifError::UnknownError(
                "Input image does not contain a gain map".into()
            ))
        );
    }

    #[test]
    fn crop_with_gainmap() {
        // Only the color image is decoded when cropping an image with a gain map.
        assert_eq!(
            decode_to_y4m("seine_sdr_gainmap_srgb.avif", &["--crop", "0,0,64,32"]),
            Ok((64, 32))
        );
    }

    #[test]
    fn crop_extracted_gainmap() {
        assert_eq!(
            decode_to_y4m(
                "seine_sdr_gainmap_srgb.avif",
                &["--extract-gainmap", "--crop", "8,8,40,30"]
            ),
            Ok((40, 30))
        );
    }

    #[cfg(feature = "libyuv")]
    #[test]
    fn target_size() {
        assert_eq!(
            decode_to_y4m("paris_10bpc.avif", &["--target-size", "100,75"]),
            Ok((100, 75))
        );
        assert_eq!(
            decode_to_y4m("sofa_grid1x5_420.avif", &["--target-size", "256,192"]),
            Ok((256, 192))
        );
    }

    #[cfg(feature = "libyuv")]
    #[test]
    fn crop_and_target_size() {
        assert_eq!(
            decode_to_y4m(
                "sofa_grid1x5_420.avif",
                &["--crop", "100,50,400,300", "--target-size", "200,150"]
            ),
            Ok((200, 150))
        );
    }

    #[cfg(feature = "libyuv")]
    #[test]
    fn target_size_invalid() {
        for bad_size in ["0,10", "500,302"] {
            assert_eq!(
                decode_to_y4m("paris_10bpc.avif", &["--target-size", bad_size]),
                Err(AvifError::UnknownError(
                    "Invalid target size: width and height must be non-zero and not exceed the image (or clean aperture) dimensions"
                        .into()
                ))
            );
        }
        assert_eq!(
            decode_to_y4m(
                "paris_10bpc.avif",
                &["--crop", "0,0,100,100", "--target-size", "101,50"]
            ),
            Err(AvifError::UnknownError(
                "Invalid crop rectangle or target size: crop must fit in the image (or clean aperture) and target size must be non-zero and not exceed the crop dimensions"
                    .into()
            ))
        );
    }

    #[cfg(feature = "libyuv")]
    #[test]
    fn target_size_with_gainmap() {
        // Only the color image is decoded when scaling an image with a gain map.
        assert_eq!(
            decode_to_y4m("seine_sdr_gainmap_srgb.avif", &["--target-size", "100,75"]),
            Ok((100, 75))
        );
    }

    #[cfg(feature = "libyuv")]
    #[test]
    fn target_size_extracted_gainmap() {
        assert_eq!(
            decode_to_y4m(
                "seine_sdr_gainmap_srgb.avif",
                &["--extract-gainmap", "--target-size", "40,30"]
            ),
            Ok((40, 30))
        );
    }

    #[cfg(all(feature = "aom", feature = "png"))]
    #[test]
    fn encode_with_exif_only() -> AvifResult<()> {
        let temp_dir = tempfile::tempdir()
            .map_err(|_| AvifError::UnknownError("Failed to create temp dir".into()))?;
        let exif_path = temp_dir.path().join("meta.exif");
        let exif_bytes = b"II*\0\x08\0\0\0\0\0".to_vec();
        std::fs::write(&exif_path, &exif_bytes).unwrap();

        let input_path = get_test_file("sacre_coeur.png");
        let encoded_path = temp_dir.path().join("output.avif");
        main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                "--exif",
                exif_path.to_str().unwrap(),
                input_path.as_str(),
                encoded_path.to_str().unwrap(),
            ]
            .iter(),
        )?;

        let mut decoder = decoder::Decoder::default();
        decoder.set_io_file(&encoded_path.to_string_lossy().into_owned())?;
        decoder.parse()?;
        let parsed = decoder.image().expect("parsed image was none");
        assert_eq!(parsed.exif, exif_bytes);
        assert!(parsed.xmp.is_empty());
        Ok(())
    }

    #[cfg(all(feature = "aom", feature = "png"))]
    #[test]
    fn encode_with_icc_exif_xmp() -> AvifResult<()> {
        let temp_dir = tempfile::tempdir()
            .map_err(|_| AvifError::UnknownError("Failed to create temp dir".into()))?;
        let icc_path = temp_dir.path().join("meta.icc");
        let exif_path = temp_dir.path().join("meta.exif");
        let xmp_path = temp_dir.path().join("meta.xmp");
        let icc_bytes = b"cli test icc profile".to_vec();
        let exif_bytes = b"II*\0\x08\0\0\0\0\0".to_vec();
        let xmp_bytes = br#"<x:xmpmeta xmlns:x="adobe:ns:meta/">cli</x:xmpmeta>"#.to_vec();
        std::fs::write(&icc_path, &icc_bytes).unwrap();
        std::fs::write(&exif_path, &exif_bytes).unwrap();
        std::fs::write(&xmp_path, &xmp_bytes).unwrap();

        let input_path = get_test_file("sacre_coeur.png");
        let encoded_path = temp_dir.path().join("output.avif");
        main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                "--icc",
                icc_path.to_str().unwrap(),
                "--exif",
                exif_path.to_str().unwrap(),
                "--xmp",
                xmp_path.to_str().unwrap(),
                input_path.as_str(),
                encoded_path.to_str().unwrap(),
            ]
            .iter(),
        )?;

        let mut decoder = decoder::Decoder::default();
        decoder.set_io_file(&encoded_path.to_string_lossy().into_owned())?;
        decoder.parse()?;
        let parsed = decoder.image().expect("parsed image was none");
        assert_eq!(parsed.icc, icc_bytes);
        assert_eq!(parsed.exif, exif_bytes);
        assert_eq!(parsed.xmp, xmp_bytes);
        Ok(())
    }

    #[cfg(all(feature = "aom", feature = "jpeg"))]
    #[test]
    fn encode_jpeg_ignore_exif_and_xmp() -> AvifResult<()> {
        let temp_dir = tempfile::tempdir()
            .map_err(|_| AvifError::UnknownError("Failed to create temp dir".into()))?;
        let input_path = get_test_file("paris_exif_xmp_icc.jpg");

        // Positive control: Exif, XMP, and ICC are preserved by default.
        let preserved_path = temp_dir.path().join("preserved.avif");
        main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                input_path.as_str(),
                preserved_path.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut decoder = decoder::Decoder::default();
        decoder.set_io_file(&preserved_path.to_string_lossy().into_owned())?;
        decoder.parse()?;
        let parsed = decoder.image().expect("parsed image was none");
        assert!(!parsed.exif.is_empty());
        assert!(!parsed.xmp.is_empty());
        assert!(!parsed.icc.is_empty());

        let encoded_path = temp_dir.path().join("ignored.avif");
        main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                "--ignore-exif",
                "--ignore-xmp",
                input_path.as_str(),
                encoded_path.to_str().unwrap(),
            ]
            .iter(),
        )?;

        let mut decoder = decoder::Decoder::default();
        decoder.set_io_file(&encoded_path.to_string_lossy().into_owned())?;
        decoder.parse()?;
        let parsed = decoder.image().expect("parsed image was none");
        assert!(parsed.exif.is_empty());
        assert!(parsed.xmp.is_empty());
        assert!(!parsed.icc.is_empty());

        // --yuv 400 on a color JPEG with ICC fails without --ignore-icc or --icc, and succeeds
        // with --icc or --ignore-icc.
        let yuv400_fail = temp_dir.path().join("yuv400_fail.avif");
        assert!(main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                "--yuv",
                "400",
                input_path.as_str(),
                yuv400_fail.to_str().unwrap(),
            ]
            .iter(),
        )
        .is_err());

        let yuv400_ignore_icc = temp_dir.path().join("yuv400_ignore_icc.avif");
        main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                "--yuv",
                "400",
                "--ignore-icc",
                input_path.as_str(),
                yuv400_ignore_icc.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut decoder = decoder::Decoder::default();
        decoder.set_io_file(&yuv400_ignore_icc.to_string_lossy().into_owned())?;
        decoder.parse()?;
        assert!(decoder
            .image()
            .expect("parsed image was none")
            .icc
            .is_empty());

        let dummy_icc = temp_dir.path().join("dummy.icc");
        std::fs::write(&dummy_icc, b"test_icc").unwrap();
        let yuv400_override_icc = temp_dir.path().join("yuv400_override_icc.avif");
        main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                "--yuv",
                "400",
                "--icc",
                dummy_icc.to_str().unwrap(),
                input_path.as_str(),
                yuv400_override_icc.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut decoder = decoder::Decoder::default();
        decoder.set_io_file(&yuv400_override_icc.to_string_lossy().into_owned())?;
        decoder.parse()?;
        assert_eq!(
            decoder.image().expect("parsed image was none").icc,
            b"test_icc"
        );
        Ok(())
    }

    #[cfg(all(feature = "aom", feature = "jpeg"))]
    #[test]
    fn encode_jpeg_ignore_icc() -> AvifResult<()> {
        let temp_dir = tempfile::tempdir()
            .map_err(|_| AvifError::UnknownError("Failed to create temp dir".into()))?;
        let input_path = get_test_file("paris_exif_xmp_icc.jpg");
        let encoded_path = temp_dir.path().join("output.avif");
        main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                "--ignore-icc",
                input_path.as_str(),
                encoded_path.to_str().unwrap(),
            ]
            .iter(),
        )?;

        let mut decoder = decoder::Decoder::default();
        decoder.set_io_file(&encoded_path.to_string_lossy().into_owned())?;
        decoder.parse()?;
        let parsed = decoder.image().expect("parsed image was none");
        assert!(parsed.icc.is_empty());
        assert_eq!(parsed.color_primaries, ColorPrimaries::Srgb);
        assert_eq!(
            parsed.transfer_characteristics,
            TransferCharacteristics::Srgb
        );
        Ok(())
    }

    #[cfg(feature = "aom")]
    fn make_alpha_image(yuv_format: PixelFormat) -> AvifResult<image::Image> {
        let mut src = image::Image {
            width: 8,
            height: 8,
            depth: 8,
            yuv_format,
            alpha_present: true,
            ..Default::default()
        };
        src.allocate_planes(Category::Color)?;
        src.allocate_planes(Category::Alpha)?;
        src.fill_plane_with_value(image::Plane::Y, 128)?;
        if yuv_format != PixelFormat::Yuv400 {
            src.fill_plane_with_value(image::Plane::U, 128)?;
            src.fill_plane_with_value(image::Plane::V, 128)?;
        }
        src.fill_plane_with_value(image::Plane::A, 100)?;
        Ok(src)
    }

    #[cfg(all(feature = "aom", feature = "png"))]
    fn encode_alpha_png_to_avif(
        temp_dir: &std::path::Path,
        prefix: &str,
        yuv_format: PixelFormat,
        extra_encode_args: &[&str],
    ) -> AvifResult<std::path::PathBuf> {
        use crabby_avif::utils::writer::png::PngWriter;
        use crabby_avif::utils::writer::Writer;

        let src = make_alpha_image(yuv_format)?;
        let alpha_png = temp_dir.join(format!("{prefix}.png"));
        PngWriter::default().write_frame(&mut std::fs::File::create(&alpha_png).unwrap(), &src)?;
        let out_avif = temp_dir.join(format!("{prefix}.avif"));
        let mut args = vec!["crabbyavif", "--speed", "10"];
        args.extend_from_slice(extra_encode_args);
        args.push(alpha_png.to_str().unwrap());
        args.push(out_avif.to_str().unwrap());
        main_impl(args.iter())?;
        Ok(out_avif)
    }

    #[cfg(all(feature = "aom", feature = "png"))]
    #[test]
    fn encode_and_decode_png_ignore_icc() -> AvifResult<()> {
        let temp_dir = tempfile::tempdir()
            .map_err(|_| AvifError::UnknownError("Failed to create temp dir".into()))?;

        // Test --ignore-icc on a PNG with embedded ICC (with positive control).
        let icc_png = get_test_file("paris_icc_exif_xmp.png");
        let with_icc_avif = temp_dir.path().join("with_icc.avif");
        main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                icc_png.as_str(),
                with_icc_avif.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut decoder = decoder::Decoder::default();
        decoder.set_io_file(&with_icc_avif.to_string_lossy().into_owned())?;
        decoder.parse()?;
        assert!(!decoder.image().unwrap().icc.is_empty());

        let no_icc_avif = temp_dir.path().join("no_icc.avif");
        main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                "--ignore-icc",
                icc_png.as_str(),
                no_icc_avif.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut decoder = decoder::Decoder::default();
        decoder.set_io_file(&no_icc_avif.to_string_lossy().into_owned())?;
        decoder.parse()?;
        let parsed = decoder.image().unwrap();
        assert!(parsed.icc.is_empty());
        assert_eq!(parsed.color_primaries, ColorPrimaries::Srgb);
        assert_eq!(
            parsed.transfer_characteristics,
            TransferCharacteristics::Srgb
        );

        // Decode with_icc_avif to PNG without and with --ignore-icc to exercise strip_icc.
        let decoded_with_icc = temp_dir.path().join("decoded_with_icc.png");
        main_impl(
            [
                "crabbyavif",
                with_icc_avif.to_str().unwrap(),
                decoded_with_icc.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut reader = <dyn Reader>::create(decoded_with_icc.to_str().unwrap())?;
        let (png_with_icc, _, _) = reader.read_frame(&Config::default())?;
        assert!(!png_with_icc.icc.is_empty());

        let decoded_no_icc = temp_dir.path().join("decoded_no_icc.png");
        main_impl(
            [
                "crabbyavif",
                "--ignore-icc",
                with_icc_avif.to_str().unwrap(),
                decoded_no_icc.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut reader = <dyn Reader>::create(decoded_no_icc.to_str().unwrap())?;
        let (png_no_icc, _, _) = reader.read_frame(&Config::default())?;
        assert!(png_no_icc.icc.is_empty());
        Ok(())
    }

    #[cfg(all(feature = "aom", feature = "png"))]
    #[test]
    fn encode_png_ignore_alpha() -> AvifResult<()> {
        let temp_dir = tempfile::tempdir()
            .map_err(|_| AvifError::UnknownError("Failed to create temp dir".into()))?;

        let with_alpha =
            encode_alpha_png_to_avif(temp_dir.path(), "with_alpha", PixelFormat::Yuv444, &[])?;
        let mut decoder = decoder::Decoder::default();
        decoder.set_io_file(&with_alpha.to_string_lossy().into_owned())?;
        decoder.parse()?;
        assert!(decoder.image().unwrap().alpha_present);

        let without_alpha = encode_alpha_png_to_avif(
            temp_dir.path(),
            "without_alpha",
            PixelFormat::Yuv444,
            &["--ignore-alpha"],
        )?;
        let mut decoder = decoder::Decoder::default();
        decoder.set_io_file(&without_alpha.to_string_lossy().into_owned())?;
        decoder.parse()?;
        assert!(!decoder.image().unwrap().alpha_present);
        Ok(())
    }

    #[cfg(feature = "aom")]
    #[test]
    fn encode_y4m_ignore_alpha() -> AvifResult<()> {
        use crabby_avif::utils::writer::y4m::Y4MWriter;
        use crabby_avif::utils::writer::Writer;

        let temp_dir = tempfile::tempdir()
            .map_err(|_| AvifError::UnknownError("Failed to create temp dir".into()))?;
        let src = make_alpha_image(PixelFormat::Yuv444)?;
        let alpha_y4m = temp_dir.path().join("in_alpha.y4m");
        let mut y4m_file = std::fs::File::create(&alpha_y4m).unwrap();
        Y4MWriter::create(false).write_frame(&mut y4m_file, &src)?;
        drop(y4m_file);

        let y4m_no_alpha_avif = temp_dir.path().join("y4m_no_alpha.avif");
        main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                "--ignore-alpha",
                alpha_y4m.to_str().unwrap(),
                y4m_no_alpha_avif.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut y4m_dec = decoder::Decoder::default();
        y4m_dec.set_io_file(&y4m_no_alpha_avif.to_string_lossy().into_owned())?;
        y4m_dec.parse()?;
        assert!(!y4m_dec.image().unwrap().alpha_present);
        Ok(())
    }

    #[cfg(all(feature = "aom", feature = "png"))]
    #[test]
    fn decode_ignore_alpha() -> AvifResult<()> {
        let temp_dir = tempfile::tempdir()
            .map_err(|_| AvifError::UnknownError("Failed to create temp dir".into()))?;

        // Test decoding an alpha AVIF with --ignore-alpha to Y4M (no C444alpha header).
        let with_alpha =
            encode_alpha_png_to_avif(temp_dir.path(), "with_alpha", PixelFormat::Yuv444, &[])?;
        let decoded_y4m = temp_dir.path().join("decoded_no_alpha.y4m");
        main_impl(
            [
                "crabbyavif",
                "--ignore-alpha",
                with_alpha.to_str().unwrap(),
                decoded_y4m.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut y4m_reader = <dyn Reader>::create(decoded_y4m.to_str().unwrap())?;
        let (y4m_img, _, _) = y4m_reader.read_frame(&Config::default())?;
        assert!(!y4m_img.alpha_present);
        assert!(!y4m_img.has_alpha());

        // Test decoding Yuv400 + alpha AVIF with --ignore-alpha and --crop to PNG (1-channel Gray).
        let mono_avif = encode_alpha_png_to_avif(
            temp_dir.path(),
            "mono_alpha",
            PixelFormat::Yuv400,
            &["--yuv", "400"],
        )?;
        let mono_out_png = temp_dir.path().join("mono_no_alpha.png");
        main_impl(
            [
                "crabbyavif",
                "--ignore-alpha",
                "--crop",
                "0,0,4,4",
                mono_avif.to_str().unwrap(),
                mono_out_png.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut png_reader = <dyn Reader>::create(mono_out_png.to_str().unwrap())?;
        let (mono_decoded, _, _) = png_reader.read_frame(&Config::default())?;
        assert!(!mono_decoded.alpha_present);
        assert!(!mono_decoded.has_alpha());
        assert_eq!((mono_decoded.width, mono_decoded.height), (4, 4));
        Ok(())
    }

    #[cfg(all(feature = "aom", feature = "png", feature = "libyuv"))]
    #[test]
    fn decode_ignore_alpha_target_size() -> AvifResult<()> {
        let temp_dir = tempfile::tempdir()
            .map_err(|_| AvifError::UnknownError("Failed to create temp dir".into()))?;
        let with_alpha =
            encode_alpha_png_to_avif(temp_dir.path(), "with_alpha", PixelFormat::Yuv444, &[])?;
        let scaled_png = temp_dir.path().join("scaled_no_alpha.png");
        main_impl(
            [
                "crabbyavif",
                "--ignore-alpha",
                "--target-size",
                "4,4",
                with_alpha.to_str().unwrap(),
                scaled_png.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut scaled_reader = <dyn Reader>::create(scaled_png.to_str().unwrap())?;
        let (scaled_decoded, _, _) = scaled_reader.read_frame(&Config::default())?;
        assert!(!scaled_decoded.alpha_present);
        assert!(!scaled_decoded.has_alpha());
        assert_eq!((scaled_decoded.width, scaled_decoded.height), (4, 4));
        Ok(())
    }

    #[cfg(all(feature = "encoder", feature = "aom"))]
    #[test]
    fn encode_grid_cell_size_and_gainmap_edges() -> AvifResult<()> {
        use crate::r#impl::add_image_single_or_grid;
        use crate::r#impl::best_cell_size;
        use crate::r#impl::GridSize;
        use crabby_avif::encoder;
        use crabby_avif::gainmap::GainMap;

        // Odd div_ceil rounds up to even on subsampled axes, stays odd on Yuv400.
        assert_eq!(best_cell_size(200, 3, true)?, 68);
        assert_eq!(best_cell_size(200, 3, false)?, 67);
        // 1D grid still validates max cell dimension (<= 65536) without u32 overflow.
        assert!(matches!(
            best_cell_size(70000, 1, false),
            Err(AvifError::InvalidImageGrid(_))
        ));
        assert!(matches!(
            best_cell_size(u32::MAX, 1, true),
            Err(AvifError::InvalidImageGrid(_))
        ));
        assert!(matches!(
            best_cell_size(64, 2, false),
            Err(AvifError::InvalidImageGrid(_))
        ));

        let make_yuv420 = |width, height| -> AvifResult<image::Image> {
            let mut img = image::Image {
                width,
                height,
                depth: 8,
                yuv_format: PixelFormat::Yuv420,
                ..Default::default()
            };
            img.allocate_planes(Category::Color)?;
            img.fill_plane_with_value(image::Plane::Y, 128)?;
            img.fill_plane_with_value(image::Plane::U, 128)?;
            img.fill_plane_with_value(image::Plane::V, 128)?;
            Ok(img)
        };
        let settings = encoder::Settings {
            speed: Some(10),
            ..Default::default()
        };

        // 2x2 grid with 1/2-size gain map (128x128 -> 64x64 cells) succeeds and writes grid boxes.
        let base = make_yuv420(256, 256)?;
        let gm_ok = GainMap {
            image: make_yuv420(128, 128)?,
            ..Default::default()
        };
        let mut enc_ok = encoder::Encoder::create_with_settings(&settings)?;
        add_image_single_or_grid(
            &mut enc_ok,
            &base,
            Some(&gm_ok),
            Some(GridSize { cols: 2, rows: 2 }),
        )?;
        let bytes = enc_ok.finish()?;
        assert!(bytes.windows(4).any(|w| w == b"grid"));

        // 2x2 grid with too-small gain map (100x100 -> 50px < 64) fails with InvalidImageGrid.
        let gm_small = GainMap {
            image: make_yuv420(100, 100)?,
            ..Default::default()
        };
        let mut enc_small = encoder::Encoder::create_with_settings(&settings)?;
        assert!(matches!(
            add_image_single_or_grid(
                &mut enc_small,
                &base,
                Some(&gm_small),
                Some(GridSize { cols: 2, rows: 2 }),
            ),
            Err(AvifError::InvalidImageGrid(_))
        ));

        // Sequence input rejects --grid.
        use crabby_avif::utils::writer::y4m::Y4MWriter;
        use crabby_avif::utils::writer::Writer;
        let temp_dir = tempfile::tempdir()
            .map_err(|_| AvifError::UnknownError("Failed to create temp dir".into()))?;
        let seq_y4m = temp_dir.path().join("seq_grid.y4m");
        let mut seq_file = std::fs::File::create(&seq_y4m).unwrap();
        let mut y4m_writer = Y4MWriter::create(false);
        y4m_writer.write_frame(&mut seq_file, &base)?;
        y4m_writer.write_frame(&mut seq_file, &base)?;
        drop(seq_file);
        let seq_avif = temp_dir.path().join("seq_grid.avif");
        assert_eq!(
            main_impl(
                [
                    "crabbyavif",
                    "--speed",
                    "10",
                    "--grid",
                    "2x2",
                    seq_y4m.to_str().unwrap(),
                    seq_avif.to_str().unwrap(),
                ]
                .iter()
            ),
            Err(AvifError::UnknownError(
                "Grid encoding is not supported for image sequences.".into()
            ))
        );
        Ok(())
    }

    #[cfg(all(feature = "aom", feature = "png"))]
    #[test]
    fn encode_grid_png_odd_rounding_and_too_small() -> AvifResult<()> {
        use crabby_avif::utils::writer::png::PngWriter;
        use crabby_avif::utils::writer::Writer;

        let temp_dir = tempfile::tempdir()
            .map_err(|_| AvifError::UnknownError("Failed to create temp dir".into()))?;

        // 200x100 PNG with --yuv 420 --grid 3x1 rounds cell width 67 -> 68 and emits a grid.
        let mut odd_img = image::Image {
            width: 200,
            height: 100,
            depth: 8,
            yuv_format: PixelFormat::Yuv444,
            ..Default::default()
        };
        odd_img.allocate_planes(Category::Color)?;
        odd_img.fill_plane_with_value(image::Plane::Y, 128)?;
        odd_img.fill_plane_with_value(image::Plane::U, 128)?;
        odd_img.fill_plane_with_value(image::Plane::V, 128)?;
        let odd_png = temp_dir.path().join("200x100.png");
        PngWriter::default()
            .write_frame(&mut std::fs::File::create(&odd_png).unwrap(), &odd_img)?;
        let odd_grid_avif = temp_dir.path().join("odd_grid.avif");
        main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                "--yuv",
                "420",
                "--grid",
                "3x1",
                odd_png.to_str().unwrap(),
                odd_grid_avif.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let grid_bytes = std::fs::read(&odd_grid_avif).unwrap();
        assert!(grid_bytes.windows(4).any(|w| w == b"grid"));
        let mut decoder = decoder::Decoder::default();
        decoder.set_io_file(&odd_grid_avif.to_string_lossy().into_owned())?;
        decoder.parse()?;
        assert_eq!(
            (
                decoder.image().unwrap().width,
                decoder.image().unwrap().height
            ),
            (200, 100)
        );

        // 64x64 PNG is too small to split into 2x1 cells (>= 64px each).
        let mut small_img = image::Image {
            width: 64,
            height: 64,
            depth: 8,
            yuv_format: PixelFormat::Yuv444,
            ..Default::default()
        };
        small_img.allocate_planes(Category::Color)?;
        small_img.fill_plane_with_value(image::Plane::Y, 128)?;
        small_img.fill_plane_with_value(image::Plane::U, 128)?;
        small_img.fill_plane_with_value(image::Plane::V, 128)?;
        let small_png = temp_dir.path().join("64x64.png");
        PngWriter::default()
            .write_frame(&mut std::fs::File::create(&small_png).unwrap(), &small_img)?;
        let small_avif = temp_dir.path().join("small_fail.avif");
        assert!(matches!(
            main_impl(
                [
                    "crabbyavif",
                    "--speed",
                    "10",
                    "--grid",
                    "2x1",
                    small_png.to_str().unwrap(),
                    small_avif.to_str().unwrap(),
                ]
                .iter(),
            ),
            Err(AvifError::InvalidImageGrid(_))
        ));
        Ok(())
    }

    #[cfg(all(feature = "aom", feature = "jpeg"))]
    #[test]
    fn encode_grid() -> AvifResult<()> {
        let temp_dir = tempfile::tempdir()
            .map_err(|_| AvifError::UnknownError("Failed to create temp dir".into()))?;
        let input_path = get_test_file("seine_sdr_gainmap_srgb.jpg");
        let non_grid_path = temp_dir.path().join("single.avif");
        main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                input_path.as_str(),
                non_grid_path.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let non_grid_bytes = std::fs::read(&non_grid_path).unwrap();
        assert!(!non_grid_bytes.windows(4).any(|w| w == b"grid"));

        let encoded_path = temp_dir.path().join("grid.avif");
        main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                "--grid",
                "2x2",
                input_path.as_str(),
                encoded_path.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let grid_bytes = std::fs::read(&encoded_path).unwrap();
        assert!(grid_bytes.windows(4).any(|w| w == b"grid"));
        assert_ne!(grid_bytes, non_grid_bytes);

        let mut decoder = decoder::Decoder::default();
        decoder.settings.image_content_to_decode = decoder::ImageContentType::All;
        decoder.set_io_file(&encoded_path.to_string_lossy().into_owned())?;
        decoder.parse()?;
        decoder.next_image()?;
        let parsed = decoder.image().expect("parsed image was none");
        assert_eq!((parsed.width, parsed.height), (400, 300));
        assert!(decoder.gainmap_present());
        assert_eq!(
            (
                decoder.gainmap().image.width,
                decoder.gainmap().image.height
            ),
            (400, 300)
        );
        Ok(())
    }

    #[cfg(feature = "aom")]
    #[test]
    fn extract_gainmap_to_avif() -> AvifResult<()> {
        let temp_dir = tempfile::tempdir()
            .map_err(|_| AvifError::UnknownError("Failed to create temp dir".into()))?;
        let input_path = get_test_file("seine_sdr_gainmap_srgb.avif");

        // Extract gainmap directly to AVIF.
        let gm_avif = temp_dir.path().join("gainmap.avif");
        main_impl(
            [
                "crabbyavif",
                "--extract-gainmap",
                "--speed",
                "10",
                "-q",
                "80",
                input_path.as_str(),
                gm_avif.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut decoder = decoder::Decoder::default();
        decoder.set_io_file(&gm_avif.to_string_lossy().into_owned())?;
        decoder.parse()?;
        decoder.next_image()?;
        let gm_parsed = decoder.image().unwrap();
        assert_eq!((gm_parsed.width, gm_parsed.height), (400, 300));
        assert!(!decoder.gainmap_present());

        // --progressive with --extract-gainmap still writes a valid single-frame AVIF.
        let prog_gm_avif = temp_dir.path().join("prog_gainmap.avif");
        main_impl(
            [
                "crabbyavif",
                "--progressive",
                "--extract-gainmap",
                "--speed",
                "10",
                input_path.as_str(),
                prog_gm_avif.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut decoder = decoder::Decoder::default();
        decoder.set_io_file(&prog_gm_avif.to_string_lossy().into_owned())?;
        decoder.parse()?;
        decoder.next_image()?;
        assert_eq!(
            (
                decoder.image().unwrap().width,
                decoder.image().unwrap().height
            ),
            (400, 300)
        );
        Ok(())
    }

    #[cfg(feature = "aom")]
    #[test]
    fn decode_to_avif_options() -> AvifResult<()> {
        let temp_dir = tempfile::tempdir()
            .map_err(|_| AvifError::UnknownError("Failed to create temp dir".into()))?;
        let input_path = get_test_file("seine_sdr_gainmap_srgb.avif");

        // --crop to AVIF clears stale clap and writes cropped pixels cleanly.
        let cropped_avif = temp_dir.path().join("cropped.avif");
        main_impl(
            [
                "crabbyavif",
                "--crop",
                "0,0,100,80",
                "--speed",
                "10",
                input_path.as_str(),
                cropped_avif.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut decoder = decoder::Decoder::default();
        decoder.set_io_file(&cropped_avif.to_string_lossy().into_owned())?;
        decoder.parse()?;
        decoder.next_image()?;
        let parsed = decoder.image().unwrap();
        assert_eq!((parsed.width, parsed.height), (100, 80));
        assert!(parsed.clap.is_none());

        // Reject --depth 16 on AVIF decode output.
        let depth16_avif = temp_dir.path().join("depth16.avif");
        assert_eq!(
            main_impl(
                [
                    "crabbyavif",
                    "-d",
                    "16",
                    "--crop",
                    "0,0,64,64",
                    input_path.as_str(),
                    depth16_avif.to_str().unwrap(),
                ]
                .iter(),
            ),
            Err(AvifError::UnknownError(
                "depth 16 is not supported for avif output".into()
            ))
        );

        // Reject --yuv on non-AVIF decode output.
        let yuv_y4m = temp_dir.path().join("yuv_fail.y4m");
        assert_eq!(
            main_impl(
                [
                    "crabbyavif",
                    "--yuv",
                    "420",
                    input_path.as_str(),
                    yuv_y4m.to_str().unwrap(),
                ]
                .iter(),
            ),
            Err(AvifError::UnknownError(
                "yuv is only supported for avif output".into()
            ))
        );

        // Converting alpha AVIF with --crop --yuv 420 -d 10 preserves alpha and decodes cleanly.
        let mut src = image::Image {
            width: 8,
            height: 8,
            depth: 8,
            yuv_format: PixelFormat::Yuv444,
            alpha_present: true,
            alpha_premultiplied: true,
            matrix_coefficients: MatrixCoefficients::Bt601,
            icc: b"fake_rgb_icc".to_vec(),
            clap: Some(crabby_avif::utils::clap::CleanAperture::create_from(
                &crabby_avif::utils::clap::CropRect {
                    x: 0,
                    y: 0,
                    width: 4,
                    height: 4,
                },
                8,
                8,
                PixelFormat::Yuv444,
            )?),
            ..Default::default()
        };
        src.allocate_planes(Category::Color)?;
        src.allocate_planes(Category::Alpha)?;
        src.fill_plane_with_value(image::Plane::Y, 64)?;
        src.fill_plane_with_value(image::Plane::U, 128)?;
        src.fill_plane_with_value(image::Plane::V, 128)?;
        src.fill_plane_with_value(image::Plane::A, 128)?;
        let alpha_src = temp_dir.path().join("alpha_src.avif");
        let settings = crabby_avif::encoder::Settings {
            speed: Some(10),
            ..Default::default()
        };
        let mut enc = crabby_avif::encoder::Encoder::create_with_settings(&settings)?;
        enc.add_image(&src)?;
        std::fs::write(&alpha_src, enc.finish()?).unwrap();

        // Re-encoding an AVIF with an embedded clap (without --crop/--target-size) preserves
        // full dimensions and clap on the encoded AVIF.
        let clap_out = temp_dir.path().join("clap_preserved.avif");
        main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                alpha_src.to_str().unwrap(),
                clap_out.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut clap_dec = decoder::Decoder::default();
        clap_dec.set_io_file(&clap_out.to_string_lossy().into_owned())?;
        clap_dec.parse()?;
        clap_dec.next_image()?;
        let clap_parsed = clap_dec.image().unwrap();
        assert_eq!((clap_parsed.width, clap_parsed.height), (8, 8));
        assert!(clap_parsed.clap.is_some());

        #[cfg(feature = "dav1d")]
        {
            // Explicit decoder -c (dav1d) does not leak into the AVIF encoder settings.
            let dav1d_out = temp_dir.path().join("dav1d_reencoded.avif");
            main_impl(
                [
                    "crabbyavif",
                    "-c",
                    "dav1d",
                    "--speed",
                    "10",
                    alpha_src.to_str().unwrap(),
                    dav1d_out.to_str().unwrap(),
                ]
                .iter(),
            )?;
            let mut dav1d_dec = decoder::Decoder::default();
            dav1d_dec.set_io_file(&dav1d_out.to_string_lossy().into_owned())?;
            dav1d_dec.parse()?;
            dav1d_dec.next_image()?;
            assert_eq!(
                (
                    dav1d_dec.image().unwrap().width,
                    dav1d_dec.image().unwrap().height
                ),
                (8, 8)
            );
        }

        // Color ICC -> Yuv400 fails without --ignore-icc and succeeds with --ignore-icc.
        let mono_fail = temp_dir.path().join("mono_fail.avif");
        assert!(main_impl(
            [
                "crabbyavif",
                "--crop",
                "0,0,4,4",
                "--yuv",
                "400",
                "--speed",
                "10",
                alpha_src.to_str().unwrap(),
                mono_fail.to_str().unwrap(),
            ]
            .iter(),
        )
        .is_err());
        let mono_ok = temp_dir.path().join("mono_ok.avif");
        main_impl(
            [
                "crabbyavif",
                "--crop",
                "0,0,4,4",
                "--yuv",
                "400",
                "--ignore-icc",
                "--speed",
                "10",
                alpha_src.to_str().unwrap(),
                mono_ok.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut mono_dec = decoder::Decoder::default();
        mono_dec.set_io_file(&mono_ok.to_string_lossy().into_owned())?;
        mono_dec.parse()?;
        mono_dec.next_image()?;
        let mono_parsed = mono_dec.image().unwrap();
        assert!(mono_parsed.icc.is_empty());
        assert_eq!(mono_parsed.color_primaries, ColorPrimaries::Srgb);
        assert_eq!(
            mono_parsed.transfer_characteristics,
            TransferCharacteristics::Srgb
        );

        let converted_avif = temp_dir.path().join("converted_420_10bit.avif");
        main_impl(
            [
                "crabbyavif",
                "--crop",
                "0,0,4,4",
                "--yuv",
                "420",
                "-d",
                "10",
                "--speed",
                "10",
                alpha_src.to_str().unwrap(),
                converted_avif.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut conv_dec = decoder::Decoder::default();
        conv_dec.set_io_file(&converted_avif.to_string_lossy().into_owned())?;
        conv_dec.parse()?;
        conv_dec.next_image()?;
        let conv_parsed = conv_dec.image().unwrap();
        assert_eq!(conv_parsed.yuv_format, PixelFormat::Yuv420);
        assert_eq!(conv_parsed.depth, 10);
        assert!(conv_parsed.alpha_present);
        assert!(conv_parsed.alpha_premultiplied);
        let y_sample = conv_parsed.row16(image::Plane::Y, 0)?[0];
        assert!(
            y_sample < 320,
            "expected premultiplied Y ~256, got {y_sample}"
        );
        Ok(())
    }

    #[cfg(feature = "encoder")]
    #[test]
    fn convert_image_format_and_depth_edge_cases() -> AvifResult<()> {
        let make_yuv = |depth: u8,
                        yuv_format: PixelFormat,
                        matrix_coefficients: MatrixCoefficients,
                        chroma_sample_position: ChromaSamplePosition|
         -> AvifResult<image::Image> {
            let mut img = image::Image {
                width: 4,
                height: 4,
                depth,
                yuv_format,
                yuv_range: image::YuvRange::Full,
                chroma_sample_position,
                matrix_coefficients,
                ..Default::default()
            };
            img.allocate_planes(Category::Color)?;
            img.fill_plane_with_value(image::Plane::Y, 64)?;
            if yuv_format != PixelFormat::Yuv400 {
                img.fill_plane_with_value(image::Plane::U, 128)?;
                img.fill_plane_with_value(image::Plane::V, 128)?;
            }
            Ok(img)
        };

        // Unchanged format and depth returns Ok(None) without cloning planes.
        let unchanged = make_yuv(
            8,
            PixelFormat::Yuv420,
            MatrixCoefficients::Bt601,
            ChromaSamplePosition::Vertical,
        )?;
        assert!(crate::r#impl::convert_image_format_and_depth(
            &unchanged, None, None, None, false, false, true
        )?
        .is_none());

        // 32-bit float sample-transform images return NotImplemented instead of panicking.
        let float_img = image::Image {
            width: 4,
            height: 4,
            depth: 32,
            yuv_format: PixelFormat::Yuv444,
            ..Default::default()
        };
        assert_eq!(
            crate::r#impl::convert_image_format_and_depth(
                &float_img,
                Some(PixelFormat::Yuv420),
                Some(8),
                None,
                false,
                false,
                true,
            )
            .err(),
            Some(AvifError::NotImplemented)
        );

        // 16-bit input clamps default target depth to 12.
        let img16 = make_yuv(
            16,
            PixelFormat::Yuv444,
            MatrixCoefficients::Bt601,
            ChromaSamplePosition::Unknown,
        )?;
        let clamped = crate::r#impl::convert_image_format_and_depth(
            &img16, None, None, None, false, false, true,
        )?
        .unwrap();
        assert_eq!(clamped.depth, 12);

        // Identity falls back to Bt601 when converting away from Yuv444, and preserves Identity on
        // Yuv444 depth change.
        let id_img = make_yuv(
            8,
            PixelFormat::Yuv444,
            MatrixCoefficients::Identity,
            ChromaSamplePosition::Unknown,
        )?;
        let id_to_420 = crate::r#impl::convert_image_format_and_depth(
            &id_img,
            Some(PixelFormat::Yuv420),
            None,
            None,
            false,
            false,
            true,
        )?
        .unwrap();
        assert_eq!(id_to_420.matrix_coefficients, MatrixCoefficients::Bt601);
        let id_to_400 = crate::r#impl::convert_image_format_and_depth(
            &id_img,
            Some(PixelFormat::Yuv400),
            None,
            None,
            false,
            false,
            true,
        )?
        .unwrap();
        assert_eq!(id_to_400.matrix_coefficients, MatrixCoefficients::Bt601);
        let id_10 = crate::r#impl::convert_image_format_and_depth(
            &id_img,
            None,
            Some(10),
            None,
            false,
            false,
            true,
        )?
        .unwrap();
        assert_eq!(id_10.matrix_coefficients, MatrixCoefficients::Identity);

        // YcgcoRe uses rgb.depth = src.depth - 2, preserves YcgcoRe when depth is unchanged and
        // format is color, and falls back to Bt601 on depth change or Yuv400.
        let ycgco_re_10 = make_yuv(
            10,
            PixelFormat::Yuv444,
            MatrixCoefficients::YcgcoRe,
            ChromaSamplePosition::Unknown,
        )?;
        let re_420 = crate::r#impl::convert_image_format_and_depth(
            &ycgco_re_10,
            Some(PixelFormat::Yuv420),
            None,
            None,
            false,
            false,
            true,
        )?
        .unwrap();
        assert_eq!(re_420.matrix_coefficients, MatrixCoefficients::YcgcoRe);
        let re_8 = crate::r#impl::convert_image_format_and_depth(
            &ycgco_re_10,
            None,
            Some(8),
            None,
            false,
            false,
            true,
        )?
        .unwrap();
        assert_eq!(re_8.matrix_coefficients, MatrixCoefficients::Bt601);
        let re_400 = crate::r#impl::convert_image_format_and_depth(
            &ycgco_re_10,
            Some(PixelFormat::Yuv400),
            None,
            None,
            false,
            false,
            true,
        )?
        .unwrap();
        assert_eq!(re_400.matrix_coefficients, MatrixCoefficients::Bt601);

        // Unsupported YcgcoRe depth (8) and YcgcoRo return NotImplemented.
        let ycgco_re_8 = make_yuv(
            8,
            PixelFormat::Yuv444,
            MatrixCoefficients::YcgcoRe,
            ChromaSamplePosition::Unknown,
        )?;
        assert_eq!(
            crate::r#impl::convert_image_format_and_depth(
                &ycgco_re_8,
                None,
                Some(10),
                None,
                false,
                false,
                true,
            )
            .err(),
            Some(AvifError::NotImplemented)
        );
        let ycgco_ro_10 = make_yuv(
            10,
            PixelFormat::Yuv444,
            MatrixCoefficients::YcgcoRo,
            ChromaSamplePosition::Unknown,
        )?;
        assert_eq!(
            crate::r#impl::convert_image_format_and_depth(
                &ycgco_ro_10,
                None,
                Some(8),
                None,
                false,
                false,
                true,
            )
            .err(),
            Some(AvifError::NotImplemented)
        );

        // Reformatting through RGB resets chroma_sample_position to Unknown.
        #[cfg(feature = "libyuv")]
        {
            let csp_img = make_yuv(
                8,
                PixelFormat::Yuv420,
                MatrixCoefficients::Bt601,
                ChromaSamplePosition::Vertical,
            )?;
            let reformatted = crate::r#impl::convert_image_format_and_depth(
                &csp_img,
                Some(PixelFormat::Yuv444),
                None,
                None,
                false,
                false,
                true,
            )?
            .unwrap();
            assert_eq!(
                reformatted.chroma_sample_position,
                ChromaSamplePosition::Unknown
            );
        }

        Ok(())
    }

    #[cfg(feature = "aom")]
    #[test]
    fn reencode_avif() -> AvifResult<()> {
        let temp_dir = tempfile::tempdir()
            .map_err(|_| AvifError::UnknownError("Failed to create temp dir".into()))?;
        let input_path = get_test_file("seine_sdr_gainmap_srgb.avif");
        let mut src_dec = decoder::Decoder::default();
        src_dec.settings.image_content_to_decode = decoder::ImageContentType::All;
        src_dec.set_io_file(&input_path)?;
        src_dec.parse()?;
        src_dec.next_image()?;
        let src_gm = src_dec.gainmap();

        let reencoded_avif = temp_dir.path().join("reencoded.avif");
        main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                "--yuv",
                "420",
                "--grid",
                "2x2",
                input_path.as_str(),
                reencoded_avif.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let bytes = std::fs::read(&reencoded_avif).unwrap();
        assert!(bytes.windows(4).any(|w| w == b"grid"));

        let mut decoder = decoder::Decoder::default();
        decoder.settings.image_content_to_decode = decoder::ImageContentType::All;
        decoder.set_io_file(&reencoded_avif.to_string_lossy().into_owned())?;
        decoder.parse()?;
        decoder.next_image()?;
        let parsed = decoder.image().unwrap();
        assert_eq!((parsed.width, parsed.height), (400, 300));
        assert_eq!(parsed.yuv_format, PixelFormat::Yuv420);
        assert!(decoder.gainmap_present());
        let dst_gm = decoder.gainmap();
        assert_eq!((dst_gm.image.width, dst_gm.image.height), (400, 300));
        assert_eq!(dst_gm.metadata, src_gm.metadata);
        assert_eq!(dst_gm.alt_color_primaries, src_gm.alt_color_primaries);
        assert_eq!(
            dst_gm.alt_transfer_characteristics,
            src_gm.alt_transfer_characteristics
        );
        assert_eq!(
            dst_gm.alt_matrix_coefficients,
            src_gm.alt_matrix_coefficients
        );
        assert_eq!(dst_gm.alt_yuv_range, src_gm.alt_yuv_range);
        assert_eq!(dst_gm.alt_plane_count, src_gm.alt_plane_count);
        assert_eq!(dst_gm.alt_plane_depth, src_gm.alt_plane_depth);
        assert_eq!(dst_gm.alt_clli, src_gm.alt_clli);
        Ok(())
    }

    #[cfg(feature = "aom")]
    #[test]
    fn reencode_preserves_metadata_and_overrides() -> AvifResult<()> {
        use crabby_avif::encoder;
        use crabby_avif::gainmap::GainMap;
        use crabby_avif::gainmap::GainMapMetadata;
        use crabby_avif::utils::Fraction;
        use crabby_avif::utils::UFraction;

        let temp_dir = tempfile::tempdir()
            .map_err(|_| AvifError::UnknownError("Failed to create temp dir".into()))?;
        let mut base = image::Image {
            width: 8,
            height: 8,
            depth: 8,
            yuv_format: PixelFormat::Yuv444,
            color_primaries: ColorPrimaries::Srgb,
            transfer_characteristics: TransferCharacteristics::Srgb,
            matrix_coefficients: MatrixCoefficients::Identity,
            icc: b"fake_rgb_icc".to_vec(),
            pasp: Some(PixelAspectRatio {
                h_spacing: 4,
                v_spacing: 3,
            }),
            clli: Some(ContentLightLevelInformation {
                max_cll: 500,
                max_pall: 200,
            }),
            ..Default::default()
        };
        base.allocate_planes(Category::Color)?;
        // Pure green in Identity (GBR): Y=G=255, U=B=0, V=R=0.
        base.fill_plane_with_value(image::Plane::Y, 255)?;
        base.fill_plane_with_value(image::Plane::U, 0)?;
        base.fill_plane_with_value(image::Plane::V, 0)?;
        let mut gm_img = image::Image {
            width: 8,
            height: 8,
            depth: 8,
            yuv_format: PixelFormat::Yuv400,
            ..Default::default()
        };
        gm_img.allocate_planes(Category::Color)?;
        gm_img.fill_plane_with_value(image::Plane::Y, 64)?;
        let gm = GainMap {
            image: gm_img,
            metadata: GainMapMetadata {
                base_hdr_headroom: UFraction(0, 1),
                alternate_hdr_headroom: UFraction(2, 1),
                min: [Fraction(0, 1); 3],
                max: [Fraction(2, 1); 3],
                gamma: [UFraction(1, 1); 3],
                base_offset: [Fraction(1, 64); 3],
                alternate_offset: [Fraction(1, 64); 3],
                use_base_color_space: true,
                ..Default::default()
            },
            alt_color_primaries: ColorPrimaries::Bt2020,
            alt_transfer_characteristics: TransferCharacteristics::Pq,
            alt_matrix_coefficients: MatrixCoefficients::Bt2020Ncl,
            alt_plane_count: 3,
            alt_plane_depth: 10,
            alt_clli: ContentLightLevelInformation {
                max_cll: 1000,
                max_pall: 400,
            },
            ..Default::default()
        };
        let settings = encoder::Settings {
            speed: Some(10),
            ..Default::default()
        };
        let mut enc = encoder::Encoder::create_with_settings(&settings)?;
        enc.add_image_gainmap(&base, &gm)?;
        let src_avif = temp_dir.path().join("meta_src.avif");
        std::fs::write(&src_avif, enc.finish()?).unwrap();

        // Preserves pasp/clli/alt_clli when flags are omitted.
        let preserved_avif = temp_dir.path().join("preserved.avif");
        main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                src_avif.to_str().unwrap(),
                preserved_avif.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut dec = decoder::Decoder::default();
        dec.settings.image_content_to_decode = decoder::ImageContentType::All;
        dec.set_io_file(&preserved_avif.to_string_lossy().into_owned())?;
        dec.parse()?;
        assert_eq!(dec.image().unwrap().pasp, base.pasp);
        assert_eq!(dec.image().unwrap().clli, base.clli);
        assert_eq!(dec.gainmap().alt_clli, gm.alt_clli);

        // --yuv 420 --cicp 1/13/1 converts Identity -> Bt709 preserving green color and
        // overrides pasp/clli/alt_clli.
        let overridden_avif = temp_dir.path().join("overridden.avif");
        main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                "--yuv",
                "420",
                "--cicp",
                "1/13/1",
                "--pasp",
                "2,1",
                "--clli",
                "800,300",
                "--alt-clli",
                "1200,500",
                src_avif.to_str().unwrap(),
                overridden_avif.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let assert_pure_green = |avif_path: &std::path::Path| -> AvifResult<image::Image> {
            let mut dec = decoder::Decoder::default();
            dec.settings.image_content_to_decode = decoder::ImageContentType::All;
            dec.set_io_file(&avif_path.to_string_lossy().into_owned())?;
            dec.parse()?;
            dec.next_image()?;
            let parsed = dec.image().unwrap();
            let mut rgb = crabby_avif::reformat::rgb::Image::create_from_yuv(parsed);
            rgb.format = crabby_avif::reformat::rgb::Format::Rgb;
            rgb.allocate()?;
            rgb.convert_from_yuv(parsed)?;
            let row = rgb.pixels.as_ref().unwrap().slice(0, 3)?;
            assert!(
                row[0] < 15 && row[1] > 240 && row[2] < 15,
                "got RGB {row:?}"
            );
            Ok(parsed.try_deep_clone()?)
        };
        let parsed = assert_pure_green(&overridden_avif)?;
        assert_eq!(parsed.yuv_format, PixelFormat::Yuv420);
        assert_eq!(parsed.matrix_coefficients, MatrixCoefficients::Bt709);
        assert_eq!(
            parsed.pasp,
            Some(PixelAspectRatio {
                h_spacing: 2,
                v_spacing: 1,
            })
        );
        assert_eq!(
            parsed.clli,
            Some(ContentLightLevelInformation {
                max_cll: 800,
                max_pall: 300,
            })
        );
        let mut gm_dec = decoder::Decoder::default();
        gm_dec.settings.image_content_to_decode = decoder::ImageContentType::All;
        gm_dec.set_io_file(&overridden_avif.to_string_lossy().into_owned())?;
        gm_dec.parse()?;
        assert_eq!(
            gm_dec.gainmap().alt_clli,
            ContentLightLevelInformation {
                max_cll: 1200,
                max_pall: 500,
            }
        );

        // Same-format 444 --cicp 1/13/1 also re-converts Identity -> Bt709 rather than relabeling.
        let same_fmt_cicp = temp_dir.path().join("same_fmt_cicp.avif");
        main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                "--cicp",
                "1/13/1",
                src_avif.to_str().unwrap(),
                same_fmt_cicp.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let same_parsed = assert_pure_green(&same_fmt_cicp)?;
        assert_eq!(same_parsed.yuv_format, PixelFormat::Yuv444);
        assert_eq!(same_parsed.matrix_coefficients, MatrixCoefficients::Bt709);

        // Decode-to-AVIF path (--crop) also honors --cicp when converting Identity -> Bt709.
        let decoded_cicp = temp_dir.path().join("decoded_cicp.avif");
        main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                "--crop",
                "0,0,8,8",
                "--yuv",
                "420",
                "--cicp",
                "1/13/1",
                src_avif.to_str().unwrap(),
                decoded_cicp.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let decoded_cicp_parsed = assert_pure_green(&decoded_cicp)?;
        assert_eq!(decoded_cicp_parsed.yuv_format, PixelFormat::Yuv420);
        assert_eq!(
            decoded_cicp_parsed.matrix_coefficients,
            MatrixCoefficients::Bt709
        );

        // --yuv 400 on Identity without --cicp gives a clear error; with --cicp and --icc it
        // succeeds.
        let gray_icc_file = temp_dir.path().join("gray.icc");
        std::fs::write(&gray_icc_file, b"new_gray_icc").unwrap();
        let gray_reencode = temp_dir.path().join("gray_icc.avif");
        assert_eq!(
            main_impl(
                [
                    "crabbyavif",
                    "--speed",
                    "10",
                    "--yuv",
                    "400",
                    "--icc",
                    gray_icc_file.to_str().unwrap(),
                    src_avif.to_str().unwrap(),
                    gray_reencode.to_str().unwrap(),
                ]
                .iter(),
            ),
            Err(AvifError::UnknownError(
                "Identity matrix requires YUV 4:4:4; pass --cicp to convert matrix coefficients"
                    .into()
            ))
        );
        main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                "--yuv",
                "400",
                "--cicp",
                "1/13/6",
                "--icc",
                gray_icc_file.to_str().unwrap(),
                src_avif.to_str().unwrap(),
                gray_reencode.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut gray_dec = decoder::Decoder::default();
        gray_dec.set_io_file(&gray_reencode.to_string_lossy().into_owned())?;
        gray_dec.parse()?;
        assert_eq!(gray_dec.image().unwrap().yuv_format, PixelFormat::Yuv400);
        assert_eq!(gray_dec.image().unwrap().icc, b"new_gray_icc");

        // --progressive fails cleanly on gainmap input and succeeds on plain AVIF.
        let prog_fail = temp_dir.path().join("prog_fail.avif");
        assert!(main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                "--progressive",
                src_avif.to_str().unwrap(),
                prog_fail.to_str().unwrap(),
            ]
            .iter(),
        )
        .is_err());

        let mut plain_enc = encoder::Encoder::create_with_settings(&settings)?;
        plain_enc.add_image(&base)?;
        let plain_avif = temp_dir.path().join("plain.avif");
        std::fs::write(&plain_avif, plain_enc.finish()?).unwrap();
        let prog_ok = temp_dir.path().join("prog_ok.avif");
        main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                "--progressive",
                plain_avif.to_str().unwrap(),
                prog_ok.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut prog_dec = decoder::Decoder::default();
        prog_dec.settings.allow_progressive = true;
        prog_dec.set_io_file(&prog_ok.to_string_lossy().into_owned())?;
        prog_dec.parse()?;
        assert_eq!(prog_dec.image_count(), 2);

        // Re-encoding an already-progressive single image with --progressive succeeds.
        let prog_reencoded = temp_dir.path().join("prog_reencoded.avif");
        main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                "--progressive",
                prog_ok.to_str().unwrap(),
                prog_reencoded.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut prog_re_dec = decoder::Decoder::default();
        prog_re_dec.settings.allow_progressive = true;
        prog_re_dec.set_io_file(&prog_reencoded.to_string_lossy().into_owned())?;
        prog_re_dec.parse()?;
        assert_eq!(prog_re_dec.image_count(), 2);

        // Changing color primaries on ChromaDerivedNcl re-converts pixels rather than relabeling.
        let derived_src = temp_dir.path().join("derived_src.avif");
        main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                "--cicp",
                "1/13/12",
                src_avif.to_str().unwrap(),
                derived_src.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let derived_dst = temp_dir.path().join("derived_dst.avif");
        main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                "--cicp",
                "9/13/12",
                derived_src.to_str().unwrap(),
                derived_dst.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let derived_parsed = assert_pure_green(&derived_dst)?;
        assert_eq!(derived_parsed.color_primaries, ColorPrimaries::Bt2020);
        assert_eq!(
            derived_parsed.matrix_coefficients,
            MatrixCoefficients::ChromaDerivedNcl
        );

        // Explicit --cicp 1/13/0 with --crop and --yuv 420 is rejected instead of falling back.
        let bad_crop_cicp = temp_dir.path().join("bad_crop_cicp.avif");
        assert!(main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                "--crop",
                "0,0,8,8",
                "--yuv",
                "420",
                "--cicp",
                "1/13/0",
                src_avif.to_str().unwrap(),
                bad_crop_cicp.to_str().unwrap(),
            ]
            .iter(),
        )
        .is_err());

        // 8-bit -> 10-bit YCgCo-Re via -d 10 --cicp 1/13/16 succeeds cleanly.
        let ycgco_re_out = temp_dir.path().join("ycgco_re_out.avif");
        main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                "-d",
                "10",
                "--cicp",
                "1/13/16",
                plain_avif.to_str().unwrap(),
                ycgco_re_out.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut ycgco_re_dec = decoder::Decoder::default();
        ycgco_re_dec.set_io_file(&ycgco_re_out.to_string_lossy().into_owned())?;
        ycgco_re_dec.parse()?;
        ycgco_re_dec.next_image()?;
        let ycgco_re_parsed = ycgco_re_dec.image().unwrap();
        assert_eq!(ycgco_re_parsed.depth, 10);
        assert_eq!(
            ycgco_re_parsed.matrix_coefficients,
            MatrixCoefficients::YcgcoRe
        );
        let mut ycgco_re_rgb = crabby_avif::reformat::rgb::Image::create_from_yuv(ycgco_re_parsed);
        ycgco_re_rgb.depth = 8;
        ycgco_re_rgb.format = crabby_avif::reformat::rgb::Format::Rgb;
        ycgco_re_rgb.allocate()?;
        ycgco_re_rgb.convert_from_yuv(ycgco_re_parsed)?;
        let row = ycgco_re_rgb.pixels.as_ref().unwrap().slice(0, 3)?;
        assert!(
            row[0] < 15 && row[1] > 240 && row[2] < 15,
            "got RGB {row:?}"
        );

        // Unsupported --cicp target matrices are rejected up front.
        let invalid_cicp_out = temp_dir.path().join("invalid_cicp.avif");
        for cicp in ["1/13/10", "1/13/16"] {
            assert!(main_impl(
                [
                    "crabbyavif",
                    "--speed",
                    "10",
                    "--cicp",
                    cicp,
                    plain_avif.to_str().unwrap(),
                    invalid_cicp_out.to_str().unwrap(),
                ]
                .iter(),
            )
            .is_err());
        }
        let mut limited_img = base.try_deep_clone()?;
        limited_img.yuv_range = image::YuvRange::Limited;
        limited_img.matrix_coefficients = MatrixCoefficients::Bt709;
        let mut limited_enc = encoder::Encoder::create_with_settings(&settings)?;
        limited_enc.add_image(&limited_img)?;
        let limited_avif = temp_dir.path().join("limited.avif");
        std::fs::write(&limited_avif, limited_enc.finish()?).unwrap();
        assert!(main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                "--cicp",
                "1/13/8",
                limited_avif.to_str().unwrap(),
                invalid_cicp_out.to_str().unwrap(),
            ]
            .iter(),
        )
        .is_err());

        // --grid with --progressive on non-encodable output reports the non-encoding grid error.
        let invalid_png = temp_dir.path().join("out.png");
        assert_eq!(
            main_impl(
                [
                    "crabbyavif",
                    "--grid",
                    "2x2",
                    "--progressive",
                    plain_avif.to_str().unwrap(),
                    invalid_png.to_str().unwrap(),
                ]
                .iter(),
            ),
            Err(AvifError::UnknownError(
                "grid is only supported when encoding".into()
            ))
        );

        // --ignore-alpha during RGB conversion drops alpha without premultiplying onto black.
        let mut alpha_src = base.try_deep_clone()?;
        alpha_src.alpha_present = true;
        alpha_src.alpha_premultiplied = false;
        alpha_src.allocate_planes(Category::Alpha)?;
        alpha_src.fill_plane_with_value(image::Plane::A, 0)?;
        let converted = crate::r#impl::convert_image_format_and_depth(
            &alpha_src,
            Some(PixelFormat::Yuv420),
            None,
            Some(&Nclx {
                color_primaries: ColorPrimaries::Srgb,
                transfer_characteristics: TransferCharacteristics::Srgb,
                matrix_coefficients: MatrixCoefficients::Bt709,
                ..Default::default()
            }),
            false,
            true,
            false,
        )?
        .expect("expected converted image");
        assert!(!converted.has_alpha());
        let mut converted_rgb = crabby_avif::reformat::rgb::Image::create_from_yuv(&converted);
        converted_rgb.format = crabby_avif::reformat::rgb::Format::Rgb;
        converted_rgb.allocate()?;
        converted_rgb.convert_from_yuv(&converted)?;
        let row = converted_rgb.pixels.as_ref().unwrap().slice(0, 3)?;
        assert!(
            row[0] < 15 && row[1] > 240 && row[2] < 15,
            "got RGB {row:?}"
        );

        // Multi-frame sequence rejects re-encode unless --index is passed.
        let seq_src = temp_dir.path().join("seq_src.avif");
        let mut seq_enc = encoder::Encoder::create_with_settings(&settings)?;
        seq_enc.add_image_for_sequence(&base, 100)?;
        seq_enc.add_image_for_sequence(&base, 100)?;
        std::fs::write(&seq_src, seq_enc.finish()?).unwrap();
        let seq_out = temp_dir.path().join("seq_out.avif");
        assert_eq!(
            main_impl(
                [
                    "crabbyavif",
                    "--speed",
                    "10",
                    seq_src.to_str().unwrap(),
                    seq_out.to_str().unwrap(),
                ]
                .iter(),
            ),
            Err(AvifError::UnknownError(
                "Re-encoding multi-frame AVIF sequences requires --index to select a frame".into()
            ))
        );
        main_impl(
            [
                "crabbyavif",
                "--speed",
                "10",
                "--index",
                "1",
                seq_src.to_str().unwrap(),
                seq_out.to_str().unwrap(),
            ]
            .iter(),
        )?;
        Ok(())
    }

    #[test]
    fn info_gainmap_metadata_rows() -> AvifResult<()> {
        let input_path = get_test_file("seine_sdr_gainmap_srgb.avif");
        main_impl(["crabbyavif", "--info", input_path.as_str()].iter())?;

        let mut decoder = decoder::Decoder::default();
        decoder.settings.image_content_to_decode = decoder::ImageContentType::All;
        decoder.set_io_file(&input_path)?;
        decoder.parse()?;
        let rows = crate::r#impl::gainmap_info_rows(decoder.gainmap(), true);
        let val_of = |label: &str| {
            rows.iter()
                .find(|(_, k, _)| *k == label)
                .map(|(_, _, v)| v.as_str())
        };
        assert_eq!(
            val_of("Gainmap"),
            Some(
                "400x300 pixels, 8 bit, Yuv444, Full Range, Matrix Coeffs. Bt601, Base Image is SDR"
            )
        );
        assert_eq!(val_of("Base headroom"), Some("0 (as fraction: 0/1)"));
        assert_eq!(
            val_of("Alternate headroom"),
            Some("1.3 (as fraction: 13/10)")
        );
        assert!(val_of("Gain Map Min").unwrap().contains("-256907/1000000"));
        assert!(val_of("Gain Map Max").unwrap().contains("1277177/1000000"));
        assert!(val_of("Base Offset")
            .unwrap()
            .contains("0.015625 (as fraction: 1/64)"));
        assert!(val_of("Alternate Offset")
            .unwrap()
            .contains("0.015625 (as fraction: 1/64)"));
        assert!(val_of("Gain Map Gamma").unwrap().contains("119223/125000"));
        assert_eq!(val_of("Use Base Color Space"), Some("True"));
        assert_eq!(val_of("Bit Depth"), Some("8"));
        assert_eq!(val_of("Planes"), Some("3"));
        assert_eq!(val_of("CLLI"), None);
        Ok(())
    }

    #[cfg(all(feature = "aom", feature = "png", feature = "jpeg"))]
    #[test]
    fn tonemap_gainmap_cli() -> AvifResult<()> {
        let temp_dir = tempfile::tempdir()
            .map_err(|_| AvifError::UnknownError("Failed to create temp dir".into()))?;
        let avif_input = get_test_file("seine_sdr_gainmap_srgb.avif");

        // Tonemap AVIF -> HDR AVIF at headroom 2.0.
        let hdr_out = temp_dir.path().join("tonemapped_hdr.avif");
        main_impl(
            [
                "crabbyavif",
                "--tonemap",
                "--headroom",
                "2.0",
                "--speed",
                "10",
                "--cicp-output",
                "9/16/6",
                avif_input.as_str(),
                hdr_out.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut decoder = decoder::Decoder::default();
        decoder.set_io_file(&hdr_out.to_string_lossy().into_owned())?;
        decoder.parse()?;
        decoder.next_image()?;
        let parsed = decoder.image().unwrap();
        assert_eq!((parsed.width, parsed.height), (400, 300));
        assert_eq!(parsed.color_primaries, ColorPrimaries::Bt2020);
        assert_eq!(parsed.transfer_characteristics, TransferCharacteristics::Pq);
        assert!(parsed.clli.is_some());
        assert!(!decoder.gainmap_present());
        assert!(!parsed.exif.is_empty());
        assert!(!parsed.xmp.is_empty());

        // Tonemap AVIF -> SDR PNG ignoring EXIF/XMP.
        let sdr_png = temp_dir.path().join("tonemapped_sdr.png");
        main_impl(
            [
                "crabbyavif",
                "--tonemap",
                "--headroom",
                "0.0",
                "--ignore-exif",
                "--ignore-xmp",
                avif_input.as_str(),
                sdr_png.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut sdr_png_reader = <dyn Reader>::create(sdr_png.to_str().unwrap())?;
        let (sdr_png_img, _, _) = sdr_png_reader.read_frame(&Config::default())?;
        assert_eq!(sdr_png_img.depth, 8);
        assert!(sdr_png_img.exif.is_empty());
        assert!(sdr_png_img.xmp.is_empty());

        // JPEG with gainmap + ICC: errors without --ignore-icc, succeeds with --ignore-icc.
        let jpg_input = get_test_file("seine_sdr_gainmap_srgb.jpg");
        let jpg_hdr_out = temp_dir.path().join("from_jpg_hdr.avif");
        assert!(main_impl(
            [
                "crabbyavif",
                "--tonemap",
                "--headroom",
                "2.0",
                "--speed",
                "10",
                jpg_input.as_str(),
                jpg_hdr_out.to_str().unwrap(),
            ]
            .iter(),
        )
        .is_err());

        main_impl(
            [
                "crabbyavif",
                "--tonemap",
                "--headroom",
                "2.0",
                "--ignore-icc",
                "--ignore-alpha",
                "--speed",
                "10",
                jpg_input.as_str(),
                jpg_hdr_out.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut decoder = decoder::Decoder::default();
        decoder.set_io_file(&jpg_hdr_out.to_string_lossy().into_owned())?;
        decoder.parse()?;
        assert_eq!(
            decoder.image().unwrap().color_primaries,
            ColorPrimaries::Srgb
        );
        assert_eq!(
            decoder.image().unwrap().transfer_characteristics,
            TransferCharacteristics::Pq
        );
        assert_eq!(
            decoder.image().unwrap().matrix_coefficients,
            MatrixCoefficients::Bt601
        );

        // -d 16 to PNG preserves 16-bit depth without clamping to 12.
        let sdr_png16 = temp_dir.path().join("tonemapped_sdr_16.png");
        main_impl(
            [
                "crabbyavif",
                "--tonemap",
                "--headroom",
                "0.0",
                "-d",
                "16",
                avif_input.as_str(),
                sdr_png16.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut sdr_png16_reader = <dyn Reader>::create(sdr_png16.to_str().unwrap())?;
        let (sdr_png16_img, _, _) = sdr_png16_reader.read_frame(&Config {
            allow_sample_transform: true,
            ..Default::default()
        })?;
        assert_eq!(sdr_png16_img.depth, 16);

        // HDR-base -> SDR (--headroom 0.0) omits spurious CLLI on SDR output.
        let mut sdr_dec = decoder::Decoder::default();
        sdr_dec.settings.image_content_to_decode = decoder::ImageContentType::All;
        sdr_dec.set_io_file(&avif_input)?;
        sdr_dec.parse()?;
        sdr_dec.next_image()?;
        let (hdr_base, hdr_base_gm) = crate::r#impl::gainmap::change_base(
            sdr_dec.image().unwrap(),
            sdr_dec.gainmap(),
            10,
            PixelFormat::Yuv444,
        )?;
        let hdr_base_input = temp_dir.path().join("hdr_base.avif");
        let mut hdr_enc = encoder::Encoder::create_with_settings(&encoder::Settings {
            speed: Some(10),
            ..Default::default()
        })?;
        hdr_enc.add_image_gainmap(&hdr_base, &hdr_base_gm)?;
        std::fs::write(&hdr_base_input, hdr_enc.finish()?).unwrap();
        let hdr_to_sdr_avif = temp_dir.path().join("hdr_to_sdr.avif");
        main_impl(
            [
                "crabbyavif",
                "--tonemap",
                "--headroom",
                "0.0",
                "--speed",
                "10",
                hdr_base_input.to_str().unwrap(),
                hdr_to_sdr_avif.to_str().unwrap(),
            ]
            .iter(),
        )?;
        let mut sdr_out_dec = decoder::Decoder::default();
        sdr_out_dec.set_io_file(&hdr_to_sdr_avif.to_string_lossy().into_owned())?;
        sdr_out_dec.parse()?;
        assert_eq!(sdr_out_dec.image().unwrap().clli, None);

        // Negative validation tests: NaN headroom, -d 10/12 PNG, unsupported --cicp-output, --crop, --target-size, --progressive.
        for bad_depth in ["10", "12"] {
            assert!(main_impl(
                [
                    "crabbyavif",
                    "--tonemap",
                    "--headroom",
                    "0.0",
                    "-d",
                    bad_depth,
                    avif_input.as_str(),
                    sdr_png.to_str().unwrap(),
                ]
                .iter(),
            )
            .is_err());
        }
        assert!(main_impl(
            [
                "crabbyavif",
                "--tonemap",
                "--headroom",
                "2.0",
                "--cicp-output",
                "9/16/14",
                avif_input.as_str(),
                hdr_out.to_str().unwrap(),
            ]
            .iter(),
        )
        .is_err());
        assert!(main_impl(
            [
                "crabbyavif",
                "--tonemap",
                "--headroom",
                "2.0",
                "--yuv",
                "420",
                "--cicp-output",
                "1/13/0",
                avif_input.as_str(),
                hdr_out.to_str().unwrap(),
            ]
            .iter(),
        )
        .is_err());
        assert!(main_impl(
            [
                "crabbyavif",
                "--tonemap",
                "--headroom",
                "nan",
                avif_input.as_str(),
                sdr_png.to_str().unwrap(),
            ]
            .iter(),
        )
        .is_err());
        assert!(main_impl(
            [
                "crabbyavif",
                "--tonemap",
                "--crop",
                "0,0,10,10",
                avif_input.as_str(),
                sdr_png.to_str().unwrap(),
            ]
            .iter(),
        )
        .is_err());
        assert!(main_impl(
            [
                "crabbyavif",
                "--tonemap",
                "--target-size",
                "100,100",
                avif_input.as_str(),
                sdr_png.to_str().unwrap(),
            ]
            .iter(),
        )
        .is_err());
        assert!(main_impl(
            [
                "crabbyavif",
                "--tonemap",
                "--progressive",
                avif_input.as_str(),
                hdr_out.to_str().unwrap(),
            ]
            .iter(),
        )
        .is_err());
        Ok(())
    }
}
