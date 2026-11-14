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

#[cfg(all(feature = "aom", feature = "dav1d", feature = "png"))]
use crabby_avif::image::YuvRange;
use crabby_avif::*;

use crate::r#impl::main_impl;

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
        main_impl(
            [
                "crabbyavif",
                "--autotiling",
                "--tilerowslog2",
                "1",
                "input.png",
                "output.avif"
            ]
            .iter()
        ),
        Err(AvifError::UnknownError(
            "--autotiling cannot be combined with --tilerowslog2 or --tilecolslog2".into()
        ))
    );
    assert_eq!(
        main_impl(
            [
                "crabbyavif",
                "--autotiling",
                "--no-autotiling",
                "input.png",
                "output.avif"
            ]
            .iter()
        ),
        Err(AvifError::UnknownError(
            "--autotiling and --no-autotiling are mutually exclusive".into()
        ))
    );
    assert_eq!(
        main_impl(["crabbyavif", "-s", "6", "--info", "input.avif"].iter()),
        Err(AvifError::UnknownError(
            "encode-only arguments cannot be used when decoding".into()
        ))
    );
    assert_eq!(
        main_impl(["crabbyavif", "--header", "mini", "--info", "input.avif"].iter()),
        Err(AvifError::UnknownError(
            "encode-only arguments cannot be used when decoding".into()
        ))
    );
    assert_eq!(
        main_impl(
            [
                "crabbyavif",
                "--repetition-count",
                "2",
                "--info",
                "input.avif"
            ]
            .iter()
        ),
        Err(AvifError::UnknownError(
            "encode-only arguments cannot be used when decoding".into()
        ))
    );
    assert_eq!(
        main_impl(["crabbyavif", "input.png"].iter()),
        Err(AvifError::UnknownError("output_file is required".into()))
    );
    assert_eq!(
        main_impl(["crabbyavif", "-", "output.avif"].iter()),
        Err(AvifError::UnknownError(
            "stdin/stdout ('-') is not supported".into()
        ))
    );
    assert_eq!(
        main_impl(
            [
                "crabbyavif",
                "--clap",
                "8,1,8,1,0,1,0,1",
                "--crop",
                "0,0,8,8",
                "input.png",
                "output.avif"
            ]
            .iter()
        ),
        Err(AvifError::UnknownError(
            "--clap and --crop are mutually exclusive".into()
        ))
    );
    Ok(())
}

#[cfg(all(feature = "aom", feature = "dav1d", feature = "png"))]
#[test]
fn cli_exif_and_xmp_flags() -> AvifResult<()> {
    let tmp = tempfile::tempdir().unwrap();
    let input_png = get_test_file("abc.png");
    let exif_path = tmp.path().join("sample.exif");
    let xmp_path = tmp.path().join("sample.xmp");
    let out_exif_only = tmp.path().join("exif_only.avif");
    let out_both = tmp.path().join("exif_and_xmp.avif");

    let exif_bytes: &[u8] = b"MM\x00*\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00";
    let xmp_bytes: &[u8] = b"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"/>";
    std::fs::write(&exif_path, exif_bytes).unwrap();
    std::fs::write(&xmp_path, xmp_bytes).unwrap();

    main_impl(
        [
            "crabbyavif",
            "--exif",
            exif_path.to_str().unwrap(),
            &input_png,
            out_exif_only.to_str().unwrap(),
        ]
        .iter(),
    )?;
    let mut dec = crabby_avif::decoder::Decoder::default();
    dec.set_io_file(&out_exif_only.to_string_lossy().into_owned())?;
    dec.parse()?;
    let img = dec.image().unwrap();
    assert_eq!(img.exif.as_slice(), exif_bytes);
    assert!(img.xmp.is_empty());

    main_impl(
        [
            "crabbyavif",
            "--exif",
            exif_path.to_str().unwrap(),
            "--xmp",
            xmp_path.to_str().unwrap(),
            &input_png,
            out_both.to_str().unwrap(),
        ]
        .iter(),
    )?;
    let mut dec2 = crabby_avif::decoder::Decoder::default();
    dec2.set_io_file(&out_both.to_string_lossy().into_owned())?;
    dec2.parse()?;
    let img2 = dec2.image().unwrap();
    assert_eq!(img2.exif.as_slice(), exif_bytes);
    assert_eq!(img2.xmp.as_slice(), xmp_bytes);
    Ok(())
}

#[cfg(all(feature = "aom", feature = "dav1d", feature = "png"))]
#[test]
fn cli_avifenc_defaults_and_options() -> AvifResult<()> {
    let tmp = tempfile::tempdir().unwrap();
    let y4m_in = tmp.path().join("seq.y4m");
    // 2 frames of 16x16 C420mpeg2 at 25:1 fps (16*16 + 2*8*8 = 384 bytes/frame).
    let frame0 = [120u8; 384];
    let frame1 = [140u8; 384];
    let y4m_bytes = [
        &b"YUV4MPEG2 W16 H16 F25:1 Ip C420mpeg2\n"[..],
        &b"FRAME\n"[..],
        &frame0[..],
        &b"FRAME\n"[..],
        &frame1[..],
    ]
    .concat();
    std::fs::write(&y4m_in, &y4m_bytes).unwrap();

    let out_y4m = tmp.path().join("seq.avif");
    main_impl(
        [
            "crabbyavif",
            "-j",
            "2",
            y4m_in.to_str().unwrap(),
            out_y4m.to_str().unwrap(),
        ]
        .iter(),
    )?;
    let mut dec_y4m = crabby_avif::decoder::Decoder::default();
    dec_y4m.set_io_file(&out_y4m.to_string_lossy().into_owned())?;
    dec_y4m.parse()?;
    assert_eq!(dec_y4m.timescale(), 25);
    assert_eq!(dec_y4m.image_count(), 2);
    assert_eq!(dec_y4m.duration_in_timescales(), 2);
    let y4m_img = dec_y4m.image().unwrap();
    assert_eq!(y4m_img.color_primaries, ColorPrimaries::Srgb);
    assert_eq!(
        y4m_img.transfer_characteristics,
        TransferCharacteristics::Srgb
    );
    assert_eq!(y4m_img.matrix_coefficients, MatrixCoefficients::Bt601);
    assert_eq!(y4m_img.yuv_range, YuvRange::Limited);
    assert_eq!(
        y4m_img.chroma_sample_position,
        ChromaSamplePosition::Vertical
    );

    let input_png = get_test_file("abc.png");
    let out_default = tmp.path().join("default.avif");
    let out_q90 = tmp.path().join("q90.avif");
    let out_opts = tmp.path().join("opts.avif");
    main_impl(["crabbyavif", &input_png, out_default.to_str().unwrap()].iter())?;
    main_impl(
        [
            "crabbyavif",
            "-q",
            "90",
            &input_png,
            out_q90.to_str().unwrap(),
        ]
        .iter(),
    )?;
    assert!(
        std::fs::metadata(&out_default).unwrap().len() < std::fs::metadata(&out_q90).unwrap().len()
    );

    main_impl(
        [
            "crabbyavif",
            "-j",
            "all",
            "-s",
            "d",
            "-y",
            "420",
            "-r",
            "limited",
            "--cicp",
            "9/16/9",
            "-a",
            "tune=ssim",
            "-a",
            "a:cq-level=20",
            &input_png,
            out_opts.to_str().unwrap(),
        ]
        .iter(),
    )?;
    let mut dec_opts = crabby_avif::decoder::Decoder::default();
    dec_opts.set_io_file(&out_opts.to_string_lossy().into_owned())?;
    dec_opts.parse()?;
    dec_opts.next_image()?;
    let img_opts = dec_opts.image().unwrap();
    assert_eq!(img_opts.yuv_format, PixelFormat::Yuv420);
    assert_eq!(img_opts.yuv_range, YuvRange::Limited);
    assert_eq!(img_opts.color_primaries, ColorPrimaries::Bt2020);
    assert_eq!(
        img_opts.transfer_characteristics,
        TransferCharacteristics::Pq
    );
    assert_eq!(img_opts.matrix_coefficients, MatrixCoefficients::Bt2020Ncl);

    assert!(main_impl(
        [
            "crabbyavif",
            "-a",
            "not-a-valid-aom-opt=1",
            &input_png,
            out_opts.to_str().unwrap(),
        ]
        .iter(),
    )
    .is_err());

    // AVIF re-encode (1-frame AVIF -> AVIF) and --cicp 1/13/0 -y 420 Identity->Bt601 downgrade.
    let out_reencode = tmp.path().join("reencoded.avif");
    main_impl(
        [
            "crabbyavif",
            "-j",
            "0",
            out_default.to_str().unwrap(),
            out_reencode.to_str().unwrap(),
        ]
        .iter(),
    )?;
    let out_cicp_420 = tmp.path().join("cicp_420.avif");
    main_impl(
        [
            "crabbyavif",
            "--cicp",
            "1/13/0",
            "-y",
            "420",
            &input_png,
            out_cicp_420.to_str().unwrap(),
        ]
        .iter(),
    )?;
    let mut dec_420 = crabby_avif::decoder::Decoder::default();
    dec_420.set_io_file(&out_cicp_420.to_string_lossy().into_owned())?;
    dec_420.parse()?;
    assert_eq!(
        dec_420.image().unwrap().matrix_coefficients,
        MatrixCoefficients::Bt601
    );
    Ok(())
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
}
