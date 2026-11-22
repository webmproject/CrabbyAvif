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
        main_impl(["crabbyavif", "--grid", "2x2", "input.avif", "output.avif"].iter()),
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
}
