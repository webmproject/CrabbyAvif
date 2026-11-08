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
}
