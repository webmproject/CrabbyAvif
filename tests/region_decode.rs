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

//! Region decoding tests: Decoder::nth_image_with_options() and
//! Decoder::next_image_with_options() with a crop_rect must produce exactly the
//! same pixels as a full decode followed by a crop, for every codec. Codecs may
//! decode only the part of the image needed for the region (such as the
//! superblocks of a non-grid image with the native AV1 decoder), so these tests
//! exercise many files, bit depths, pixel formats, dimensions and regions.

#![allow(clippy::disallowed_methods, clippy::disallowed_macros)]

use crabby_avif::decoder::ImageContentType;
use crabby_avif::image::*;
use crabby_avif::utils::clap::CropRect;
use crabby_avif::*;

mod utils;
use utils::*;

use std::cell::RefCell;
use std::rc::Rc;
use test_case::test_matrix;

const DAV1D: CodecChoice =
    if cfg!(feature = "dav1d") { CodecChoice::Dav1d } else { CodecChoice::Auto };
#[cfg(feature = "native_av1")]
const NATIVE: CodecChoice = CodecChoice::Native;
// Placeholder that is never used for decoding: the tests return early for
// NATIVE when the native AV1 decoder is not available.
#[cfg(not(feature = "native_av1"))]
const NATIVE: CodecChoice = CodecChoice::Aom;

fn codec_unavailable(codec: CodecChoice) -> bool {
    !HAS_NON_ANDROID_DECODER
        || (!cfg!(feature = "dav1d") && codec == DAV1D)
        || (!cfg!(feature = "native_av1") && codec == NATIVE)
}

fn crop(x: u32, y: u32, width: u32, height: u32) -> CropRect {
    CropRect {
        x,
        y,
        width,
        height,
    }
}

// Same as Image::crop_in_place(), implemented independently of the library.
fn crop_reference_image(src: &Image, rect: &CropRect) -> AvifResult<Image> {
    let x = match src.yuv_format {
        PixelFormat::Yuv420 | PixelFormat::Yuv422 => rect.x & !1,
        _ => rect.x,
    };
    let y = match src.yuv_format {
        PixelFormat::Yuv420 => rect.y & !1,
        _ => rect.y,
    };
    let mut dst = Image {
        width: rect.width,
        height: rect.height,
        depth: src.depth,
        yuv_format: src.yuv_format,
        yuv_range: src.yuv_range,
        chroma_sample_position: src.chroma_sample_position,
        alpha_present: src.alpha_present,
        alpha_premultiplied: src.alpha_premultiplied,
        color_primaries: src.color_primaries,
        transfer_characteristics: src.transfer_characteristics,
        matrix_coefficients: src.matrix_coefficients,
        clli: src.clli,
        pasp: src.pasp,
        clap: None,
        irot_angle: src.irot_angle,
        imir_axis: src.imir_axis,
        ..Default::default()
    };
    if src.has_plane(Plane::Y) {
        dst.allocate_planes(Category::Color)?;
    }
    if src.has_plane(Plane::A) {
        dst.allocate_planes(Category::Alpha)?;
    }
    for plane in ALL_PLANES {
        if !src.has_plane(plane) {
            continue;
        }
        let (src_x, src_y) = if matches!(plane, Plane::Y | Plane::A) {
            (x as usize, y)
        } else {
            (
                src.yuv_format.apply_chroma_shift_x(x) as usize,
                src.yuv_format.apply_chroma_shift_y(y),
            )
        };
        let copy_w = dst.width(plane);
        for row in 0..dst.height(plane) as u32 {
            if src.depth == 8 {
                let s = &src.row(plane, src_y + row)?[src_x..src_x + copy_w];
                dst.row_mut(plane, row)?[..copy_w].copy_from_slice(s);
            } else {
                let s = &src.row16(plane, src_y + row)?[src_x..src_x + copy_w];
                dst.row16_mut(plane, row)?[..copy_w].copy_from_slice(s);
            }
        }
    }
    Ok(dst)
}

// Regions of a width x height image: edges, corners, odd positions and dimensions, 1x1, the whole
// image, regions within a single 64x64 or 128x128 superblock and regions straddling superblock
// (and thus potentially tile) boundaries.
fn crops_for(width: u32, height: u32) -> Vec<CropRect> {
    let (w, h) = (width, height);
    let candidates = [
        crop(0, 0, w, h),
        crop(0, 0, 1, 1),
        crop(w - 1, h - 1, 1, 1),
        crop(w / 2, h / 2, 1, 1),
        crop((w / 2) | 1, (h / 2) | 1, 1, 1),
        crop(1, 1, w - 1, h - 1),
        crop(0, 0, w.div_ceil(2), h.div_ceil(2)),
        crop((w / 4) | 1, (h / 4) | 1, (w / 2) | 1, (h / 2) | 1),
        crop(
            w - w.div_ceil(3),
            h - h.div_ceil(3),
            w.div_ceil(3),
            h.div_ceil(3),
        ),
        crop(w - 3.min(w), h - 3.min(h), 3.min(w), 3.min(h)),
        crop(w - 1, 0, 1, h),
        crop(0, h - 1, w, 1),
        crop(1, h - 1, w - 1, 1),
        crop(0, h / 3, w, (h / 3).max(1)),
        crop(w / 3, 0, (w / 3).max(1), h),
        // Within a single 64x64 superblock.
        crop(10, 20, 31, 17),
        crop(65, 67, 30, 29),
        crop(70, 30, 33, 21),
        crop(3, 5, 7, 9),
        // Straddling 64x64 superblock boundaries.
        crop(60, 61, 9, 7),
        crop(63, 63, 2, 2),
        crop(127, 127, 3, 3),
        // Within a single 128x128 superblock.
        crop(130, 131, 100, 99),
        // Straddling 128x128 superblock boundaries.
        crop(120, 121, 20, 19),
        crop(255, 255, 2, 2),
        crop(250, 3, 270, 515),
        crop(w / 2 - 1.min(w / 2), 0, 2.min(w), h),
    ];
    // Keep the tests reasonably fast: apart from the whole image, only regions of at most a
    // quarter of large images are tested.
    let area = u64::from(w) * u64::from(h);
    let max_area = if area > 1 << 20 { area / 4 } else { area };
    let mut crops = Vec::new();
    for c in candidates {
        if c.width != 0
            && c.height != 0
            && c.x + c.width <= width
            && c.y + c.height <= height
            && (u64::from(c.width) * u64::from(c.height) <= max_area || c == candidates[0])
            && !crops.contains(&c)
        {
            crops.push(c);
        }
    }
    crops
}

// Returns a decoder of the gain map if gainmap is true, of the color and alpha otherwise.
fn decoder_for_data(data: &[u8], codec: CodecChoice, gainmap: bool) -> decoder::Decoder {
    let mut decoder = decoder::Decoder::default();
    decoder.settings.codec_choice = codec;
    decoder.settings.image_content_to_decode = if gainmap {
        ImageContentType::GainMap
    } else {
        ImageContentType::ColorAndAlpha
    };
    decoder.set_io_vec(data.to_vec());
    decoder
}

fn decoded_image(decoder: &decoder::Decoder, gainmap: bool) -> &Image {
    if gainmap {
        &decoder.gainmap().image
    } else {
        decoder.image().unwrap()
    }
}

// Decodes the frame at index of data entirely and then each region of crops_for() with the same
// decoder instance (to also check that no state leaks from one decode to the next), and checks that
// each region is identical to the crop of the full decode. Finally decodes the whole frame again
// with the same decoder. Returns false if the full decode fails, true otherwise.
fn check_regions(
    data: &[u8],
    codec: CodecChoice,
    gainmap: bool,
    index: u32,
    name: &str,
) -> AvifResult<bool> {
    // Invalid clean apertures are ignored, regions are then relative to the whole image.
    let strictness =
        || decoder::Strictness::SpecificExclude(vec![decoder::StrictnessFlag::ClapValid]);
    let mut ref_decoder = decoder_for_data(data, codec, gainmap);
    ref_decoder.settings.strictness = strictness();
    if ref_decoder.parse().is_err() || ref_decoder.nth_image(index).is_err() {
        return Ok(false);
    }
    let ref_full = decoded_image(&ref_decoder, gainmap).try_deep_clone()?;
    if !ref_full.has_plane(Plane::Y) {
        return Ok(false);
    }
    // Regions are relative to the clean aperture, if any.
    let full_rect = crop(0, 0, ref_full.width, ref_full.height);
    let base = match &ref_full.clap {
        Some(clap) => {
            CropRect::create_from(clap, ref_full.width, ref_full.height, ref_full.yuv_format)
                .unwrap_or(full_rect)
        }
        None => full_rect,
    };
    let crops = crops_for(base.width, base.height);
    let mut decoder = decoder_for_data(data, codec, gainmap);
    decoder.settings.strictness = strictness();
    decoder.parse()?;
    for crop in &crops {
        let options = decoder::DecodeOptions {
            crop_rect: Some(*crop),
        };
        decoder
            .nth_image_with_options(index, &options)
            .unwrap_or_else(|err| {
                panic!("{name}: region decode of {crop:?} failed: {err:?}");
            });
        let mut rect = *crop;
        rect.x += base.x;
        rect.y += base.y;
        let expected = crop_reference_image(&ref_full, &rect)?;
        let actual = decoded_image(&decoder, gainmap);
        assert_eq!(
            (actual.width, actual.height),
            (crop.width, crop.height),
            "{name} {crop:?}"
        );
        assert!(actual.clap.is_none(), "{name} {crop:?}");
        assert!(
            are_images_equal(actual, &expected)?,
            "{name}: region {crop:?} of {codec:?} differs from full decode + crop"
        );
    }
    decoder.nth_image(index)?;
    assert!(
        are_images_equal(decoded_image(&decoder, gainmap), &ref_full)?,
        "{name}: full decode after region decodes differs"
    );
    Ok(true)
}

fn check_regions_of_file(path: &str, codec: CodecChoice) -> AvifResult<bool> {
    let data = std::fs::read(path).expect("Unable to read file");
    let mut decoded = check_regions(&data, codec, false, 0, path)?;
    let mut probe = decoder_for_data(&data, codec, false);
    if probe.parse().is_ok() && probe.gainmap_present() {
        decoded |= check_regions(&data, codec, true, 0, path)?;
    }
    Ok(decoded)
}

fn tests_data_dir(dir: &str) -> String {
    let path = get_test_file(dir);
    path.trim_end_matches('/').to_string()
}

// Returns the paths of all the .avif files in dir and its subdirectories, sorted.
fn avif_files(dir: &str) -> Vec<String> {
    let mut files = Vec::new();
    let mut dirs = vec![std::path::PathBuf::from(dir)];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).expect("Unable to read dir") {
            let path = entry.expect("Unable to read dir entry").path();
            if path.is_dir() {
                dirs.push(path);
            } else if path.extension().is_some_and(|ext| ext == "avif") {
                files.push(path.to_string_lossy().into_owned());
            }
        }
    }
    files.sort();
    files
}

// Checks all the regions of the AVIF files in the given directory. min_decoded is the minimum
// number of these files that must be decodable (to make sure the test does not silently test
// nothing).
fn check_regions_of_dir(dir: &str, codec: CodecChoice, min_decoded: usize) -> AvifResult<()> {
    let files = avif_files(dir);
    let mut decoded = 0;
    for file in &files {
        if check_regions_of_file(file, codec)? {
            decoded += 1;
        }
    }
    eprintln!("{decoded} of {} files of {dir} were decoded", files.len());
    assert!(
        decoded >= min_decoded,
        "only {decoded} of {} files of {dir} were decoded",
        files.len()
    );
    Ok(())
}

#[test_matrix([DAV1D, NATIVE])]
fn region_decode_tests_data(codec: CodecChoice) -> AvifResult<()> {
    if codec_unavailable(codec) {
        return Ok(());
    }
    // Not recursive: tests/data/region only contains grids, tested separately below.
    let dir = tests_data_dir("");
    let mut files: Vec<String> = std::fs::read_dir(&dir)
        .expect("Unable to read dir")
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.is_file() && path.extension().is_some_and(|ext| ext == "avif"))
        .map(|path| path.to_string_lossy().into_owned())
        .collect();
    files.sort();
    let mut decoded = 0;
    for file in &files {
        if check_regions_of_file(file, codec)? {
            decoded += 1;
        }
    }
    assert!(
        decoded >= 60,
        "only {decoded} of {} files were decoded",
        files.len()
    );
    Ok(())
}

#[test_matrix([DAV1D, NATIVE])]
fn region_decode_tests_data_region(codec: CodecChoice) -> AvifResult<()> {
    if codec_unavailable(codec) {
        return Ok(());
    }
    check_regions_of_dir(&tests_data_dir("region"), codec, 25)
}

// The files of tests/data/region_decode were pre-encoded with libavif (see tests/data/README.md).
fn region_decode_file(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}/{name}", tests_data_dir("region_decode")))
        .unwrap_or_else(|err| panic!("Unable to read {name}: {err}"))
}

fn yuv_format_name(yuv_format: PixelFormat) -> &'static str {
    match yuv_format {
        PixelFormat::Yuv400 => "400",
        PixelFormat::Yuv420 => "420",
        PixelFormat::Yuv422 => "422",
        PixelFormat::Yuv444 => "444",
        _ => unreachable!(),
    }
}

fn check_file(name: &str, codec: CodecChoice) -> AvifResult<()> {
    let data = region_decode_file(name);
    assert!(check_regions(&data, codec, false, 0, name)?, "{name}");
    Ok(())
}

// Images with and without alpha (with limited range YUV) of various dimensions.
#[test_matrix(
    [PixelFormat::Yuv400, PixelFormat::Yuv420, PixelFormat::Yuv422, PixelFormat::Yuv444],
    [8, 10, 12],
    [false, true],
    [DAV1D, NATIVE]
)]
fn region_decode_formats(
    yuv_format: PixelFormat,
    depth: u8,
    alpha: bool,
    codec: CodecChoice,
) -> AvifResult<()> {
    if codec_unavailable(codec) {
        return Ok(());
    }
    let yuv = yuv_format_name(yuv_format);
    let alpha = if alpha { "_alpha" } else { "" };
    for size in ["301x203", "64x64", "1x1", "2x3", "129x66"] {
        check_file(
            &format!("formats_{yuv}_{depth}bpc{alpha}_{size}.avif"),
            codec,
        )?;
    }
    Ok(())
}

// IO that records the byte ranges that are read, to check whether a decode reads any data.
struct TrackingIO {
    data: Vec<u8>,
    reads: Rc<RefCell<Vec<(usize, usize)>>>,
}

impl decoder::IO for TrackingIO {
    fn read(&mut self, offset: u64, max_read_size: usize) -> AvifResult<&[u8]> {
        let start = usize::try_from(offset).map_err(|_| AvifError::IoError)?;
        if start > self.data.len() {
            return Err(AvifError::IoError);
        }
        let end = start + std::cmp::min(max_read_size, self.data.len() - start);
        self.reads.borrow_mut().push((start, end));
        Ok(&self.data[start..end])
    }

    fn size_hint(&self) -> u64 {
        self.data.len() as u64
    }

    fn persistent(&self) -> bool {
        true
    }
}

// Region and full decodes of a single-tile image with alpha and of a gain map, in any order,
// with the same decoder instance.
#[test_matrix(
    [
        ("alpha.avif", false),
        ("alpha_premultiplied.avif", false),
        ("seine_sdr_gainmap_srgb.avif", true),
        ("seine_sdr_gainmap_srgb.avif", false),
        ("paris_10bpc.avif", false)
    ],
    [DAV1D, NATIVE]
)]
fn region_decode_cache_and_state_transitions(
    filename_and_gainmap: (&str, bool),
    codec: CodecChoice,
) -> AvifResult<()> {
    if codec_unavailable(codec) {
        return Ok(());
    }
    let (filename, gainmap) = filename_and_gainmap;
    let data = std::fs::read(get_test_file(filename)).expect("Unable to read file");
    let mut ref_decoder = decoder_for_data(&data, codec, gainmap);
    ref_decoder.parse()?;
    ref_decoder.nth_image(0)?;
    let ref_full = decoded_image(&ref_decoder, gainmap).try_deep_clone()?;
    let (w, h) = (ref_full.width, ref_full.height);

    let reads = Rc::new(RefCell::new(Vec::new()));
    let mut decoder = decoder::Decoder::default();
    decoder.settings.codec_choice = codec;
    if gainmap {
        decoder.settings.image_content_to_decode = ImageContentType::GainMap;
    }
    decoder.set_io(Box::new(TrackingIO {
        data,
        reads: reads.clone(),
    }));
    decoder.parse()?;
    let crop1 = crop(3, 5, w / 2, h / 3);
    let crop2 = crop(w / 3, h / 2, w - w / 3, h - h / 2);
    let opts1 = decoder::DecodeOptions {
        crop_rect: Some(crop1),
    };
    let opts2 = decoder::DecodeOptions {
        crop_rect: Some(crop2),
    };
    let check = |decoder: &decoder::Decoder, expected_crop: Option<&CropRect>| -> AvifResult<()> {
        let expected = match expected_crop {
            Some(rect) => crop_reference_image(&ref_full, rect)?,
            None => ref_full.try_deep_clone()?,
        };
        assert!(
            are_images_equal(decoded_image(decoder, gainmap), &expected)?,
            "{filename}: {expected_crop:?}"
        );
        Ok(())
    };

    decoder.nth_image_with_options(0, &opts1)?;
    check(&decoder, Some(&crop1))?;
    // Same frame and same options: cache hit, nothing is read.
    reads.borrow_mut().clear();
    decoder.nth_image_with_options(0, &opts1)?;
    assert!(reads.borrow().is_empty());
    check(&decoder, Some(&crop1))?;
    // Different region: decoded again.
    decoder.nth_image_with_options(0, &opts2)?;
    assert!(!reads.borrow().is_empty());
    check(&decoder, Some(&crop2))?;
    // No image past the last one: the decoded region is left untouched.
    assert!(decoder.next_image_with_options(&opts1).is_err());
    check(&decoder, Some(&crop2))?;
    // Full decode after a region decode.
    decoder.nth_image(0)?;
    check(&decoder, None)?;
    reads.borrow_mut().clear();
    decoder.nth_image(0)?;
    assert!(reads.borrow().is_empty());
    check(&decoder, None)?;
    // Region decode after a full decode.
    decoder.nth_image_with_options(0, &opts2)?;
    check(&decoder, Some(&crop2))?;
    decoder.nth_image_with_options(0, &opts1)?;
    check(&decoder, Some(&crop1))?;
    // An invalid region fails without affecting later decodes.
    assert!(decoder
        .nth_image_with_options(
            0,
            &decoder::DecodeOptions {
                crop_rect: Some(crop(w - 1, 0, 2, 1)),
            }
        )
        .is_err());
    decoder.nth_image_with_options(0, &opts1)?;
    check(&decoder, Some(&crop1))?;
    decoder.nth_image(0)?;
    check(&decoder, None)?;

    // A fresh decoder starting with next_image_with_options().
    let mut decoder = decoder_for_data(
        &std::fs::read(get_test_file(filename)).unwrap(),
        codec,
        gainmap,
    );
    decoder.parse()?;
    decoder.next_image_with_options(&opts2)?;
    check(&decoder, Some(&crop2))?;
    Ok(())
}

// Keyframes of an image sequence can be decoded by the native AV1 decoder. Region decodes of
// these frames, in any order and with different regions, must never return a stale frame.
#[test_matrix([DAV1D, NATIVE])]
fn region_decode_sequence_keyframes(codec: CodecChoice) -> AvifResult<()> {
    // dav1d decodes the reference frames, including the non-keyframes.
    if codec_unavailable(codec) || !cfg!(feature = "dav1d") {
        return Ok(());
    }
    let data = std::fs::read(get_test_file("colors-animated-12bpc-keyframes-0-2-3.avif"))
        .expect("Unable to read file");
    let mut ref_decoder = decoder_for_data(&data, DAV1D, false);
    ref_decoder.parse()?;
    let mut ref_frames = Vec::new();
    for index in 0..ref_decoder.image_count() {
        ref_decoder.nth_image(index)?;
        ref_frames.push(ref_decoder.image().unwrap().try_deep_clone()?);
    }
    let keyframes: Vec<u32> = (0..ref_decoder.image_count())
        .filter(|&i| ref_decoder.is_keyframe(i))
        .collect();
    assert_eq!(keyframes, [0, 2, 3]);
    let (w, h) = (ref_frames[0].width, ref_frames[0].height);
    let crops = [
        crop(1, 3, w / 2, h / 3),
        crop(w / 2, h / 2, w / 2, h / 2),
        crop(0, 0, 1, 1),
    ];
    let mut decoder = decoder_for_data(&data, codec, false);
    decoder.parse()?;
    for &index in [3u32, 0, 2, 2, 0, 3].iter() {
        for crop in &crops {
            decoder.nth_image_with_options(
                index,
                &decoder::DecodeOptions {
                    crop_rect: Some(*crop),
                },
            )?;
            assert!(
                are_images_equal(
                    decoder.image().unwrap(),
                    &crop_reference_image(&ref_frames[index as usize], crop)?
                )?,
                "frame {index} region {crop:?}"
            );
        }
        decoder.nth_image(index)?;
        assert!(are_images_equal(
            decoder.image().unwrap(),
            &ref_frames[index as usize]
        )?);
    }
    Ok(())
}

// The 'ispe' dimensions of the items differ from the dimensions of the AV1 frames, so a full decode
// scales the frames. A region decode must behave the same way.
#[test_matrix([(150, 90), (260, 150)], [DAV1D, NATIVE])]
fn region_decode_scaled_frame(ispe: (u32, u32), codec: CodecChoice) -> AvifResult<()> {
    if codec_unavailable(codec) {
        return Ok(());
    }
    // Without alpha: decoding an alpha item again after scaling it fails with all the codecs.
    let mut data = region_decode_file("scaled_200x120.avif");
    let mut patched = 0;
    for offset in 0..data.len() - 16 {
        if &data[offset..offset + 4] == b"ispe" {
            assert_eq!(
                data[offset + 4..offset + 16],
                [0, 0, 0, 0, 0, 0, 0, 200, 0, 0, 0, 120]
            );
            data[offset + 8..offset + 12].copy_from_slice(&ispe.0.to_be_bytes());
            data[offset + 12..offset + 16].copy_from_slice(&ispe.1.to_be_bytes());
            patched += 1;
        }
    }
    assert_eq!(patched, 1);
    assert!(check_regions(&data, codec, false, 0, "scaled")?);
    Ok(())
}
